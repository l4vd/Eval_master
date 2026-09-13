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

A *constrained* results file (`*_constrained_results.json`, from `evaluate.py --scoring
constrained`) carries `logp_yes` / `logp_no` instead of a parsed judgement. It is scored
without any parser: AUROC of logP(Yes) - logP(No) against the label, with its
Hanley-McNeil SE. `evaluate.py` imports the same functions, so offline and live
constrained scores cannot drift apart. This is a modified protocol, never a replacement
for the strict number.

Decontamination (a modified protocol too): `--exclude LIST.json` restricts scoring to the
rows an exclusion list keeps (`analysis/overlap.py --write-exclusions`), and
`--emit-summary DIR` writes `<task>_<label>[_constrained]_decontam_summary.json` scoring the
original, decontaminated, seen, clean and unseen-exposed row sets side by side, with SEs and
the seen-minus-clean contrast. The original summary is never touched.

Usage:
    python score_results.py qa_<label>_results.json
    python score_results.py <run>/halueval/*_results.json --json scores.json
    python score_results.py <run>/halueval/summarization_<label>_results.json \\
        --exclude <Eval_master>/decontamination/ragtruth/halueval__summarization.json \\
        --emit-summary <modified_run>/halueval
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
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


def binomial_se(p, n):
    """Standard error of an accuracy `p` over `n` rows; None when undefined."""
    return math.sqrt(p * (1 - p) / n) if n else None


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


# --- Constrained (parser-free) scoring --------------------------------------------

def constrained_verdict(logp_yes, logp_no):
    """The argmax verdict between the two verdict masses. An exact tie reads as "No"."""
    return "Yes" if logp_yes > logp_no else "No"


def verdict_mass(logp_yes, logp_no):
    """Probability the judge puts on any verdict token at all (1.0 = never anything else)."""
    return math.exp(logp_yes) + math.exp(logp_no)


def auroc(scores, labels):
    """Mann-Whitney AUROC, ties at their average rank; None when a class is empty.

    `labels` are truthy for the positive class (ground truth "Yes" = hallucinated).
    Equivalent to P(score_pos > score_neg) + 0.5 * P(tie).
    """
    n = len(scores)
    n_pos = sum(1 for label in labels if label)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    start = 0
    while start < n:
        end = start
        while end + 1 < n and scores[order[end + 1]] == scores[order[start]]:
            end += 1
        average = (start + end) / 2 + 1  # 1-based rank shared by the tied block
        for k in range(start, end + 1):
            ranks[order[k]] = average
        start = end + 1
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auroc_se(auc, n_pos, n_neg):
    """Hanley & McNeil (1982) standard error of an AUROC; None when undefined."""
    if auc is None or not n_pos or not n_neg:
        return None
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    variance = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc)
                + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(max(variance, 0.0))


def score_constrained(rows):
    """AUROC and argmax statistics over rows carrying `logp_yes` / `logp_no`."""
    scores, labels, masses = [], [], []
    correct = judged_yes = 0
    gt_total = {"Yes": 0, "No": 0}
    gt_correct = {"Yes": 0, "No": 0}
    for row in rows:
        ground_truth = row.get("ground_truth")
        logp_yes, logp_no = row.get("logp_yes"), row.get("logp_no")
        if ground_truth not in gt_total or logp_yes is None or logp_no is None:
            continue
        logp_yes, logp_no = float(logp_yes), float(logp_no)
        verdict = constrained_verdict(logp_yes, logp_no)
        gt_total[ground_truth] += 1
        scores.append(logp_yes - logp_no)
        labels.append(ground_truth == "Yes")
        masses.append(verdict_mass(logp_yes, logp_no))
        judged_yes += verdict == "Yes"
        if verdict == ground_truth:
            correct += 1
            gt_correct[ground_truth] += 1

    n = len(scores)
    auc = auroc(scores, labels)
    return {
        "num_examples": n,
        "auroc": auc,
        "auroc_se": auroc_se(auc, gt_total["Yes"], gt_total["No"]),
        "accuracy_argmax": ratio(correct, n),
        "num_correct_argmax": correct,
        "num_ground_truth_yes": gt_total["Yes"],
        "num_ground_truth_no": gt_total["No"],
        "tpr": ratio(gt_correct["Yes"], gt_total["Yes"]),
        "tnr": ratio(gt_correct["No"], gt_total["No"]),
        "judged_yes_rate": ratio(judged_yes, n),
        "mean_verdict_mass": statistics.fmean(masses) if masses else None,
        "median_verdict_mass": statistics.median(masses) if masses else None,
        # Share of rows where "Yes"/"No" is not even the judge's majority continuation:
        # there the argmax is a choice between two unlikely tokens.
        "frac_mass_below_half": ratio(sum(1 for m in masses if m < 0.5), n),
    }


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_constrained(rows):
    return any("logp_yes" in row for row in rows)


def score_file(path):
    """Both scorings for one results file, plus which field each could use.

    A run predating the `raw_judgement` field (or the shipped gpt-3.5 reference file)
    still scores strictly — the stored `judgement` is already the parsed verdict, so
    re-parsing it is a no-op — but its lenient scoring is unavailable, because the
    text that failed to parse was never kept. That is exactly the gap the field closes.
    """
    return score_rows(load_rows(path), path)


def score_rows(rows, path):
    if is_constrained(rows):
        constrained = score_constrained(rows)
        return {"path": str(path), "num_rows": len(rows), "scoring": "constrained",
                "num_skipped": len(rows) - constrained["num_examples"],
                "constrained": constrained}

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


# --- Decontamination ------------------------------------------------------------

ROW_SETS = ("original", "decontaminated", "seen", "clean", "unseen_exposed")
# Copied from the run's own summary so a decontaminated summary is comparable on its own.
PROVENANCE_KEYS = ("model", "backend", "batch_size", "sort_by_length", "seed",
                   "num_samples_requested", "max_new_tokens", "prompt_format", "verdict_tokens")
_HEADLINE = {
    "strict": ("accuracy", "accuracy_se", "num_correct", "num_failed", "format_compliance",
               "tpr", "tnr", "judged_yes_rate"),
    "constrained": ("auroc", "auroc_se", "accuracy_argmax", "tpr", "tnr", "judged_yes_rate",
                    "mean_verdict_mass", "median_verdict_mass", "frac_mass_below_half"),
}


def load_exclusion_list(path):
    """An `analysis/overlap.py` exclusion list, plus the sha256 of the file as read."""
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "train": data.get("train"),
        "benchmark": data.get("benchmark"),
        "task": data.get("task"),
        "policy": data.get("policy"),
        "report_sha256": data.get("report_sha256"),
        "n_rows": data.get("n_rows"),
        "excluded": {int(i) for i in data.get("indices", [])},
        "unseen_exposed": {int(i) for i in data.get("unseen_exposed_indices", [])},
    }


def results_stem(path):
    """`summarization_<label>[_constrained]` for `summarization_<label>[_constrained]_results.json`."""
    name = Path(path).name
    if not name.endswith("_results.json"):
        raise ValueError(f"{name} is not a *_results.json file")
    return name[: -len("_results.json")]


def results_task(path):
    """The task a results file belongs to (task names carry no underscore)."""
    return results_stem(path).split("_", 1)[0]


def split_row_sets(rows, exclusion):
    """Partition rows by an exclusion list.

    original = every row; decontaminated = not excluded; seen = excluded; within the kept
    rows, unseen_exposed = matched training text only the test side / release saw, and
    clean = the rest.
    """
    sets = {name: [] for name in ROW_SETS}
    for row in rows:
        sets["original"].append(row)
        if row.get("index") is None:
            continue
        index = int(row["index"])
        if index in exclusion["excluded"]:
            sets["seen"].append(row)
            continue
        sets["decontaminated"].append(row)
        sets["unseen_exposed" if index in exclusion["unseen_exposed"] else "clean"].append(row)
    return sets


def _set_scores(rows, constrained):
    if constrained:
        scores = score_constrained(rows)
        return {"num_examples": scores["num_examples"], "constrained": scores}
    strict = score(rows, parse_strict, "judgement")
    strict["accuracy_se"] = binomial_se(strict["accuracy"], strict["num_examples"])
    out = {"num_examples": strict["num_examples"], "strict": strict}
    if any("raw_judgement" in row for row in rows):
        lenient = score(rows, parse_lenient, "raw_judgement")
        lenient["accuracy_se"] = binomial_se(lenient["accuracy"], lenient["num_examples"])
        out["lenient"] = lenient
    return out


def _contrast(a, b):
    """a - b for every scoring both row sets carry, with the SE of the difference."""
    out = {}
    for scoring, metric in (("strict", "accuracy"), ("lenient", "accuracy"), ("constrained", "auroc")):
        if scoring not in a or scoring not in b:
            continue
        va, vb = a[scoring].get(metric), b[scoring].get(metric)
        if va is None or vb is None:
            continue
        sa, sb = a[scoring].get(f"{metric}_se"), b[scoring].get(f"{metric}_se")
        out[scoring] = {"metric": metric, "diff": va - vb,
                        "se": math.sqrt(sa * sa + sb * sb) if sa is not None and sb is not None else None,
                        "n": [a["num_examples"], b["num_examples"]]}
    return out


def decontam_summary(results_path, exclusion, rows=None):
    """The `*_decontam_summary.json` payload for one results file and exclusion list."""
    results_path = Path(results_path)
    rows = load_rows(results_path) if rows is None else rows
    task = results_task(results_path)
    if exclusion.get("task") not in (None, task):
        raise ValueError(f"{results_path.name} is task {task!r}; the exclusion list is for "
                         f"{exclusion['task']!r}")
    constrained = is_constrained(rows)
    scoring = "constrained" if constrained else "strict"
    sets = split_row_sets(rows, exclusion)
    scored = {name: _set_scores(subset, constrained) for name, subset in sets.items()}

    stem = results_stem(results_path)
    provenance = {}
    own_summary = results_path.with_name(f"{stem}_summary.json")
    if own_summary.is_file():
        data = json.loads(own_summary.read_text(encoding="utf-8"))
        provenance = {k: data[k] for k in PROVENANCE_KEYS if k in data}
    label = stem[len(task) + 1:]
    if constrained and label.endswith("_constrained"):
        label = label[: -len("_constrained")]

    expected = exclusion.get("n_rows")
    summary = {
        "task": task,
        "model": provenance.pop("model", label),
        "variant": "decontam",
        "scoring": "constrained" if constrained else "generate",
        "source_results": results_path.name,
        "exclusion_list": exclusion["path"],
        "exclusion_list_sha256": exclusion["sha256"],
        "exclusion_report_sha256": exclusion.get("report_sha256"),
        "train": exclusion.get("train"),
        "policy": exclusion.get("policy"),
        "n_excluded": len(sets["seen"]),
        "n_unseen_exposed": len(sets["unseen_exposed"]),
        "n_rows_expected": expected,
        # A --num-samples or interrupted run scores a subset; its row sets are not the
        # policy's, and the summary says so rather than passing for a full one.
        "complete": expected is None or scored["original"]["num_examples"] == expected,
        **provenance,
        "num_examples": scored["decontaminated"]["num_examples"],
    }
    head = scored["decontaminated"][scoring]
    summary.update({k: head[k] for k in _HEADLINE[scoring] if k in head})
    if "lenient" in scored["decontaminated"]:
        summary["accuracy_lenient"] = scored["decontaminated"]["lenient"]["accuracy"]
    summary["row_sets"] = scored
    summary["contrasts"] = {
        "seen_minus_clean": _contrast(scored["seen"], scored["clean"]),
        "seen_minus_unseen_exposed": _contrast(scored["seen"], scored["unseen_exposed"]),
    }
    return summary


def decontam_summary_name(results_path):
    return f"{results_stem(results_path)}_decontam_summary.json"


# --- Reporting ------------------------------------------------------------------

def _print_report(result):
    if result.get("scoring") == "constrained":
        c = result["constrained"]
        auc = "n/a" if c["auroc"] is None else f"{c['auroc']:.4f} (SE {c['auroc_se']:.4f})"
        print(f"\n=== {Path(result['path']).name}  ({c['num_examples']} scored rows, constrained) ===")
        print(f"  labels: {c['num_ground_truth_yes']} hallucinated / {c['num_ground_truth_no']} correct")
        print(f"  AUROC {auc} | argmax accuracy {c['accuracy_argmax']:.4f} | TPR {c['tpr']:.3f} | "
              f"TNR {c['tnr']:.3f} | yes-rate {c['judged_yes_rate']:.3f}")
        if c["mean_verdict_mass"] is not None:
            print(f"  verdict mass: mean {c['mean_verdict_mass']:.3f}, median "
                  f"{c['median_verdict_mass']:.3f}, below 0.5 on {c['frac_mass_below_half']:.1%} of rows")
        return

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


def _print_decontam(summary):
    scoring, metric = ("constrained", "auroc") if summary["scoring"] == "constrained" else ("strict", "accuracy")
    print(f"  decontaminated ({summary['n_excluded']} rows excluded"
          f"{'' if summary['complete'] else ', INCOMPLETE run'}):")
    for name in ROW_SETS:
        block = summary["row_sets"][name]
        value = block[scoring][metric]
        se = block[scoring][f"{metric}_se"]
        shown = "n/a" if value is None else f"{value:.4f}" + ("" if se is None else f" (SE {se:.4f})")
        print(f"    {name:<15} n={block['num_examples']:>6}  {metric} {shown}")
    contrast = summary["contrasts"]["seen_minus_clean"].get(scoring)
    if contrast and contrast["se"] is not None:
        print(f"    seen - clean: {contrast['diff']:+.4f} (SE {contrast['se']:.4f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="one or more *_results.json files")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="also write all scorings to this JSON file")
    parser.add_argument("--exclude", metavar="LIST.json", default=None,
                        help="score only the rows this exclusion list keeps (analysis/overlap.py "
                             "--write-exclusions); results of another task are skipped")
    parser.add_argument("--emit-summary", metavar="DIR", default=None,
                        help="with --exclude: write <task>_<label>[_constrained]_decontam_summary.json "
                             "into DIR. DIR must not be the results' own directory: the "
                             "decontaminated variant belongs in the modified-protocol root.")
    args = parser.parse_args()
    if args.emit_summary and not args.exclude:
        parser.error("--emit-summary needs --exclude")

    exclusion = load_exclusion_list(args.exclude) if args.exclude else None
    emit_dir = Path(args.emit_summary) if args.emit_summary else None
    results = []
    for p in args.results:
        path = Path(p)
        if exclusion is None:
            result = score_file(path)
            _print_report(result)
            results.append(result)
            continue

        if exclusion["task"] and results_task(path) != exclusion["task"]:
            print(f"\n--- skip {path.name}: not task {exclusion['task']!r}")
            continue
        rows = load_rows(path)
        summary = decontam_summary(path, exclusion, rows)
        result = score_rows(split_row_sets(rows, exclusion)["decontaminated"], path)
        result["decontam"] = summary
        _print_report(result)
        _print_decontam(summary)
        if emit_dir is not None:
            if emit_dir.resolve() == path.parent.resolve():
                raise SystemExit(f"refusing to write a decontaminated summary next to {path.name}: "
                                 "that directory holds the original protocol's artifacts. Pass "
                                 "the mirrored directory under the modified-protocol root.")
            emit_dir.mkdir(parents=True, exist_ok=True)
            out = emit_dir / decontam_summary_name(path)
            out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"  wrote {out}")
        results.append(result)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
