"""Re-score a HaluEval results file offline — no model, no GPU, no re-run.

`evaluate.py` writes one row per example carrying `ground_truth`, the parsed
`judgement`, and (since this fork stores it) the judge's `raw_judgement`. That is
enough to recompute every metric anyone might want from a run that already happened,
which is the point: the expensive part is generation, and it only has to be paid once.

Two scorings are reported side by side:

  strict   The published HaluEval rule, reproduced exactly: capitalised `Yes`/`No`
           substring search over the judgement with "." removed, ambiguous or absent
           verdicts counted as WRONG. This is the number that is comparable to the
           paper, and it is the only one that belongs in a results table without a
           caveat.

  lenient  Case-insensitive, word-boundary, first-verdict-wins. NOT comparable to the
           published numbers — it exists to size the gap. When strict and lenient
           agree, the strict parser is costing you nothing and the failures are real
           refusals; when they diverge, the difference is the share of your score that
           is output formatting rather than hallucination detection.

Usage:
    python score_results.py qa_<label>_results.json
    python score_results.py <run>/halueval/*_results.json --json scores.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# --- The two parsers ------------------------------------------------------------

def parse_strict(text):
    """The upstream rule, character for character (evaluate.py's scoring branch).

    Kept as its own function ONLY so it can be applied to a stored file; the copy in
    `evaluate.py` remains the one that scores a live run. The two must stay in sync —
    `tests/test_scoring.py` pins them against each other.
    """
    ans = text.replace(".", "")
    if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
        return None
    return "Yes" if "Yes" in ans else "No"


_VERDICT = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_lenient(text):
    """First standalone yes/no token, any case.

    Word boundaries are what the strict rule lacks: bare `"No" in ans` also fires on
    "Not", "Note", "None", "November", so a `Yes` verdict that goes on to use any of
    them reads as ambiguous and is thrown away.
    """
    match = _VERDICT.search(text)
    return match.group(1).capitalize() if match else None


# --- Scoring --------------------------------------------------------------------

def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def score(rows, parser, field):
    """Metrics for one (parser, field) combination over already-judged rows.

    `n` counts every row, and an unparseable verdict counts as incorrect — matching
    how the live run computes accuracy, so `strict` here reproduces `summary.json`.
    """
    n = correct = failed = judged_yes = 0
    gt_total = {"Yes": 0, "No": 0}
    gt_correct = {"Yes": 0, "No": 0}

    for row in rows:
        ground_truth = row.get("ground_truth")
        if ground_truth not in gt_total:
            continue  # not a scored row (the writer can emit a blank line on interrupt)
        n += 1
        gt_total[ground_truth] += 1

        text = row.get(field)
        verdict = None if text is None else parser(text)
        if verdict is None:
            failed += 1
            continue
        if verdict == "Yes":
            judged_yes += 1
        if verdict == ground_truth:
            correct += 1
            gt_correct[ground_truth] += 1

    parsed = n - failed
    return {
        "num_examples": n,
        "num_correct": correct,
        "num_incorrect": n - correct,
        "accuracy": ratio(correct, n),
        "num_failed": failed,
        "format_compliance": ratio(parsed, n),
        "num_ground_truth_yes": gt_total["Yes"],
        "num_ground_truth_no": gt_total["No"],
        "tpr": ratio(gt_correct["Yes"], gt_total["Yes"]),
        "tnr": ratio(gt_correct["No"], gt_total["No"]),
        "judged_yes_rate": ratio(judged_yes, parsed),
    }


def score_file(path):
    """Both scorings for one results file, plus which field each could use.

    A run predating the `raw_judgement` field (or the shipped gpt-3.5 reference file)
    still scores strictly — the stored `judgement` is already the parsed verdict, so
    re-parsing it is a no-op — but its lenient scoring is unavailable, because the
    text that failed to parse was never kept. That is exactly the gap the field closes.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    has_raw = any("raw_judgement" in row for row in rows)
    # `judgement` is "failed!" for unparseable rows, which no parser matches — so it
    # scores identically to re-parsing the raw text, and works on legacy files too.
    strict = score(rows, parse_strict, "judgement")
    # Rows without a `ground_truth` are not scorable and are excluded from the
    # denominator — which means this file's accuracy is over FEWER rows than the split
    # has. Surfaced rather than silently absorbed: it is the signature of a run that was
    # interrupted and appended to, and it makes the number non-comparable to a full run.
    result = {"path": str(path), "num_rows": len(rows), "has_raw_judgement": has_raw,
              "num_skipped": len(rows) - strict["num_examples"], "strict": strict}
    if has_raw:
        result["lenient"] = score(rows, parse_lenient, "raw_judgement")
    return result


# --- Reporting ------------------------------------------------------------------

def _print_report(result):
    strict, lenient = result["strict"], result.get("lenient")
    print(f"\n=== {Path(result['path']).name}  ({strict['num_examples']} scored rows) ===")
    print(f"  labels: {strict['num_ground_truth_yes']} hallucinated / "
          f"{strict['num_ground_truth_no']} correct")
    if result["num_skipped"]:
        print(f"  !! {result['num_skipped']} unscorable row(s) skipped (no ground_truth) - "
              f"partial or interrupted run; the denominator is not the full split.")

    def block(name, s):
        print(f"  {name:<8} accuracy {s['accuracy']:.4f} | "
              f"compliance {s['format_compliance']:.4f} ({s['num_failed']} unparseable) | "
              f"TPR {s['tpr']:.3f} | TNR {s['tnr']:.3f} | yes-rate {s['judged_yes_rate']:.3f}")

    block("strict", strict)
    if lenient is None:
        print("  lenient  unavailable (no raw_judgement stored - pre-instrumentation run)")
    else:
        block("lenient", lenient)
        gap = lenient["accuracy"] - strict["accuracy"]
        recovered = strict["num_failed"] - lenient["num_failed"]
        print(f"  -> parser accounts for {gap:+.4f} accuracy "
              f"({recovered} of {strict['num_failed']} unparseable rows recoverable)")

    if strict["num_failed"]:
        print(f"  !! {strict['num_failed'] / strict['num_examples']:.1%} of rows scored as wrong "
              f"because no verdict could be read - not because the judge was wrong.")
    if strict["judged_yes_rate"] in (0.0, 1.0) and strict["format_compliance"] > 0:
        constant = "No" if strict["judged_yes_rate"] == 0.0 else "Yes"
        print(f"  !! degenerate judge: answered \"{constant}\" on every parsed row; "
              f"its accuracy is just the base rate of that label.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="one or more *_results.json files")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="also write all scorings to this JSON file")
    args = parser.parse_args()

    results = [score_file(Path(p)) for p in args.results]
    for result in results:
        _print_report(result)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
