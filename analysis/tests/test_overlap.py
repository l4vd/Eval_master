"""analysis.overlap: tiers, exposure and exclusion lists, on synthetic data and on the real data.

The synthetic tests run offline in well under a second. The real-data test indexes
SP-DPO-Base's data/ragtruth against Eval_master's benchmark files and must reproduce the
tables of SP-DPO-Base/KNOWN_ISSUES.md §5; it is skipped when either side is absent.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from analysis import overlap
from analysis.overlap import Params, TrainIndex, TrainSide, check_row

P = Params(ngram=5, near_dup=0.5, df_max=2, exposure_topk=3)
_WORDS = [f"w{i}" for i in range(5000)]


def doc(seed: int, n: int = 60) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(_WORDS) for _ in range(n))


def side(groups: dict[str, tuple[str, list[str]]], params: Params = P) -> TrainSide:
    """A train side from ``{group: (exposure, [texts])}``."""
    index = TrainIndex(params)
    for name, (_, texts) in groups.items():
        for text in texts:
            index.add(name, text)
    return TrainSide("t", "synthetic", index, [groups[g][0] for g in index.group_names], [])


def write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# --- matching ----------------------------------------------------------------------

def test_normalize_folds_case_punctuation_width_and_underscores():
    assert overlap.normalize("Hello,  WORLD!! ｆｕｌｌ_width") == "hello world full width"


def test_exact_match_survives_case_and_punctuation():
    base = doc(1)
    s = side({"s1": ("train", [base])})
    r = check_row(0, [("document", base.upper().replace(" ", ", ") + "!")], s)
    assert (r.tier, r.containment, r.exposure, r.source_ids) == ("exact", 1.0, "train", ["s1"])


def test_truncated_document_is_a_near_duplicate_not_exact():
    base = doc(2, 100)
    s = side({"s1": ("val", [base])})
    r = check_row(0, [("document", " ".join(base.split()[:60]))], s)
    assert (r.tier, r.containment, r.exposure) == ("near_duplicate", 1.0, "val")


def test_shared_quote_is_partial_and_its_span_is_reported():
    quote = doc(3, 8)
    s = side({"s1": ("train", [f"{doc(4)} {quote} {doc(5)}"])})
    r = check_row(0, [("document", f"{doc(6)} {quote} {doc(7)}")], s)
    assert r.tier == "partial"
    assert r.containment == pytest.approx(4 / 124)  # 8-5+1 shared of 128-5+1 distinct 5-grams
    assert r.span == quote


def test_boilerplate_ngrams_above_df_max_are_ignored():
    boiler = doc(8, 10)
    three = side({f"s{i}": ("train", [f"{doc(10 + i)} {boiler}"]) for i in range(3)})  # df 3 > 2
    assert check_row(0, [("document", f"{doc(20)} {boiler}")], three).tier == "none"
    assert three.index.n_boilerplate == 10 - 5 + 1
    two = side({f"s{i}": ("train", [f"{doc(10 + i)} {boiler}"]) for i in range(2)})  # df 2 <= 2
    assert check_row(0, [("document", f"{doc(20)} {boiler}")], two).tier == "partial"


def test_short_texts_are_checked_for_exact_matches_only():
    s = side({"s1": ("train", ["Capital of France", doc(31)])})
    assert check_row(0, [("question", "capital of france?")], s).tier == "exact"
    contained = " ".join(doc(31).split()[:4])  # inside a training text, but shorter than n
    assert check_row(0, [("question", contained)], s).tier == "none"
    assert check_row(0, [("question", "  !!  ")], s).tier == "none"  # empty never matches


def test_exposure_is_the_strongest_among_exact_sources():
    base = doc(40)
    s = side({"t_src": ("test", [base]), "v_src": ("val", [base])})
    r = check_row(0, [("document", base)], s)
    assert r.exposure == "val" and sorted(r.source_ids) == ["t_src", "v_src"]


def test_exposure_otherwise_comes_from_the_top_k_sources_by_shared_ngrams():
    a, b, c, d = (doc(50 + i, 30) for i in range(4))
    groups = {"A": ("release_only", [a]), "B": ("test", [b]), "C": ("train", [c]), "D": ("val", [d])}
    text = " ".join([a, *b.split()[:20], *c.split()[:10], *d.split()[:6]])  # shares 26/16/6/2
    r3 = check_row(0, [("document", text)], side(groups))
    assert r3.source_ids == ["A", "B", "C"] and r3.exposure == "train"
    r2 = check_row(0, [("document", text)], side(groups, Params(ngram=5, df_max=2, exposure_topk=2)))
    assert r2.source_ids == ["A", "B"] and r2.exposure == "test"


def test_a_row_takes_its_best_field():
    base = doc(60)
    r = check_row(7, [("summary", doc(61)), ("document", base)], side({"s": ("train", [base])}))
    assert (r.row_id, r.tier, r.field) == (7, "exact", "document")


# --- inputs ------------------------------------------------------------------------

def test_row_ids_follow_line_order(tmp_path):
    write_jsonl(tmp_path / "HaluEval-reproduce" / "data" / "qa_data.json", [{"question": f"q{i}"} for i in range(3)])
    _, rows = overlap.halueval_rows(tmp_path, "qa")
    assert [(i, fields[0][1]) for i, fields in rows] == [(0, "q0"), (1, "q1"), (2, "q2")]


def test_missing_eval_file_is_not_checked_rather_than_zero(tmp_path):
    with pytest.raises(overlap.NotChecked, match="not found"):
        overlap.halueval_rows(tmp_path, "qa")


def test_harness_samples_merge_every_task_by_doc_id(tmp_path):
    rows = [
        {"task": "truthfulqa_gen", "doc_id": "0", "doc": json.dumps({"question": "Q zero?", "best_answer": "A0"})},
        {"task": "truthfulqa_mc1", "doc_id": "0",
         "doc": json.dumps({"question": "Q zero?", "mc1_targets": {"choices": ["x", "y"], "labels": [1, 0]}})},
        {"task": "truthfulqa_mc2", "doc_id": "1", "doc": json.dumps({"question": "Q one?"})},
    ]
    _, merged = overlap.harness_rows(write_jsonl(tmp_path / "samples.jsonl", rows))
    assert [row_id for row_id, _ in merged] == [0, 1]
    fields = dict(merged)[0]
    assert fields.count(("question", "Q zero?")) == 1
    assert ("mc1_targets.choices[]", "x") in fields


def test_jsonl_train_spec_takes_windows_paths_and_a_group_field(tmp_path):
    m = overlap._JSONL_SPEC.match(r"jsonl:C:\data\train.jsonl:context,chosen:source_id")
    assert (m["path"], m["fields"], m["group"]) == (r"C:\data\train.jsonl", "context,chosen", "source_id")
    path = write_jsonl(tmp_path / "train.jsonl", [{"context": doc(70), "source_id": "a"}, {"context": doc(71), "source_id": "b"}])
    s = overlap.load_jsonl_train("added", f"jsonl:{path}:context:source_id", P)
    assert s.index.group_names == ["a", "b"] and set(s.exposure_by_group) == {"train"}


def test_run_config_flags_halueval_and_truthfulqa_presets(tmp_path):
    pytest.importorskip("yaml")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("data:\n  name: halueval_mixture\n  sources: [halueval_qa, halueval_summarization]\n", encoding="utf-8")
    flagged = {(i["benchmark"], tuple(i["tasks"])) for i in overlap.read_run_config(cfg)["identity_overlaps"]}
    assert {("halueval", ("qa",)), ("halueval", ("summarization",))} <= flagged
    cfg.write_text("data:\n  name: truthfulqa\n", encoding="utf-8")
    assert {i["benchmark"] for i in overlap.read_run_config(cfg)["identity_overlaps"]} == {"truthfulqa", "harness"}
    cfg.write_text("data:\n  name: ragtruth_mixture\n  sources: [ragtruth_qa]\n", encoding="utf-8")
    assert overlap.read_run_config(cfg)["identity_overlaps"] == []


# --- end to end ----------------------------------------------------------------------

def _pair(sid: str, context: str, k: int) -> dict:
    return {"pair_id": f"{sid}:{k}", "source_id": sid, "prompt": "", "context": context,
            "author_prompt": "Summarize: " + context, "chosen": doc(1000 + 10 * int(sid) + k, 20),
            "rejected": doc(2000 + 10 * int(sid) + k, 20), "category": "Summary", "corpus": "CNN/DM"}


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Sources 1/2/3 have train/val/test pairs, 4 is release-only; 5 exists only in the benchmark."""
    monkeypatch.delenv("FAITHEVAL_DATA_DIR", raising=False)
    docs = {sid: doc(100 + int(sid), 40) for sid in "12345"}
    rt = tmp_path / "ragtruth"
    write_jsonl(rt / "summary" / "train.jsonl", [_pair("1", docs["1"], 0), _pair("1", docs["1"], 1)])
    write_jsonl(rt / "summary" / "val.jsonl", [_pair("2", docs["2"], 0)])
    write_jsonl(rt / "summary" / "test.jsonl", [_pair("3", docs["3"], 0)])
    sources = [{"source_id": sid, "task_type": "Summary", "source": "CNN/DM", "source_info": docs[sid],
                "prompt": "Summarize: " + docs[sid]} for sid in "12345"]
    write_jsonl(rt / "raw" / "source_info.jsonl", sources[:4])

    ev = tmp_path / "eval"
    quote = " ".join(docs["1"].split()[10:18])
    blank = {"right_summary": "fine", "hallucinated_summary": "wrong"}
    write_jsonl(ev / "HaluEval-reproduce" / "data" / "summarization_data.json", [
        {"document": doc(900), **blank},                           # 0 clean
        {"document": docs["1"].upper(), **blank},                  # 1 exact, train
        {"document": docs["2"], **blank},                          # 2 exact, val
        {"document": docs["3"], **blank},                          # 3 exact, test
        {"document": docs["4"], **blank},                          # 4 exact, release_only
        {"document": f"{doc(903)} {quote} {doc(904)}", **blank},   # 5 partial, train
    ])
    ds = ev / "RAGTruth-reproduce" / "dataset"
    write_jsonl(ds / "source_info.jsonl", sources)
    split = {"1": "train", "2": "train", "3": "test", "4": "test", "5": "test"}
    write_jsonl(ds / "response.jsonl", [{"id": sid, "source_id": sid, "split": sp, "quality": "good",
                                         "labels": []} for sid, sp in split.items()])
    return rt, ev


def _run(rt: Path, ev: Path, out: Path, decon: Path) -> dict:
    argv = ["--train", f"ragtruth={rt}", "--eval-root", str(ev), "--ngram", "5", "--df-max", "2",
            "--out", str(out), "--write-exclusions", "--exclusions-dir", str(decon)]
    assert overlap.main(argv) == 0
    return json.loads((out / "ragtruth" / "overlap_report.json").read_text(encoding="utf-8"))


def test_end_to_end_report_and_exclusion_list(world, tmp_path):
    rt, ev = world
    report = _run(rt, ev, tmp_path / "out", tmp_path / "decon")

    s = report["benchmarks"]["halueval"]["summarization"]
    assert s["tiers"] == {"exact": 4, "near_duplicate": 0, "partial": 1, "none": 1}
    assert s["exposure_by_tier"]["train"] == {"exact": 1, "near_duplicate": 0, "partial": 1}
    assert [s["exposure_by_tier"][e]["exact"] for e in ("val", "test", "release_only")] == [1, 1, 1]
    assert (s["excluded"], s["unseen_exposed"], s["clean"], s["kept"]) == (2, 2, 2, 4)
    assert s["affected_train_data"]["summary/train"] == {"sources": 1, "sources_total": 1, "pairs": 2, "pairs_total": 2}

    missing = {(n["benchmark"], n["task"]) for n in report["not_checked"]}
    assert {("halueval", "qa"), ("faitheval", "counterfactual"), ("truthfulqa", "TruthfulQA.csv"),
            ("harness", "truthfulqa")} <= missing
    assert "qa" not in report["benchmarks"]["halueval"]  # missing is not reported as zero

    rb = report["benchmarks"]["ragtruth_benchmark"]
    assert rb["test"]["identity"] == {"train": 0, "val": 0, "test": 1, "release_only": 1, "not_in_train": 1}
    assert (rb["all"]["identity"]["train"], rb["all"]["identity"]["val"]) == (1, 1)
    assert rb["test"]["excluded"] == 0 and "Summary" in rb["internal_test_vs_train"]

    decon = tmp_path / "decon" / "ragtruth"
    lists = sorted(p.name for p in decon.glob("*__*.json"))
    assert lists == ["halueval__summarization.json"]  # only non-empty lists are written
    payload = json.loads((decon / lists[0]).read_text(encoding="utf-8"))
    assert payload["indices"] == [1, 2] and payload["unseen_exposed_indices"] == [3, 4]
    assert payload["n_rows"] == 6 and payload["per_index"]["1"]["source_ids"] == ["1"]
    report_bytes = (tmp_path / "out" / "ragtruth" / "overlap_report.json").read_bytes()
    assert payload["report_sha256"] == hashlib.sha256(report_bytes).hexdigest()
    assert (decon / "overlap_report.json").read_bytes() == report_bytes  # the committed copy


def test_report_is_deterministic(world, tmp_path):
    rt, ev = world
    _run(rt, ev, tmp_path / "a", tmp_path / "da")
    _run(rt, ev, tmp_path / "b", tmp_path / "db")
    for rel in ("ragtruth/overlap_report.json", "ragtruth/overlap_report.md"):
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()
    rel = "ragtruth/halueval__summarization.json"
    assert (tmp_path / "da" / rel).read_bytes() == (tmp_path / "db" / rel).read_bytes()


# --- the real data: SP-DPO-Base/KNOWN_ISSUES.md §5 -----------------------------------

_HALUEVAL_SUMMARIZATION = overlap.ROOT / "HaluEval-reproduce" / "data" / "summarization_data.json"
_REAL_AVAILABLE = (overlap.DEFAULT_RAGTRUTH_DATA / "summary" / "train.jsonl").is_file() and _HALUEVAL_SUMMARIZATION.is_file()


@pytest.mark.slow
@pytest.mark.skipif(not _REAL_AVAILABLE, reason="SP-DPO-Base data/ragtruth or HaluEval data not present")
def test_real_data_reproduces_known_issues_section_5(tmp_path):
    harness = next(iter(sorted((overlap.ROOT / "outputs" / "eval").glob("*/*/*/harness/samples.jsonl"))), None)
    argv = ["--out", str(tmp_path / "out"), "--write-exclusions", "--exclusions-dir", str(tmp_path / "decon")]
    if harness is not None:
        argv += ["--harness-samples", str(harness)]
    assert overlap.main(argv) == 0
    report = json.loads((tmp_path / "out" / "ragtruth" / "overlap_report.json").read_text(encoding="utf-8"))
    b = report["benchmarks"]

    s = b["halueval"]["summarization"]
    assert s["tiers"] == {"exact": 628, "near_duplicate": 5, "partial": 188, "none": 9179}
    assert s["exposure_by_tier"] == {
        "train": {"exact": 424, "near_duplicate": 3, "partial": 126},
        "val": {"exact": 76, "near_duplicate": 0, "partial": 15},
        "test": {"exact": 86, "near_duplicate": 2, "partial": 43},
        "release_only": {"exact": 42, "near_duplicate": 0, "partial": 4},
    }
    assert (s["excluded"], s["unseen_exposed"], s["clean"], s["kept"]) == (503, 130, 9367, 9497)
    assert s["affected_train_data"] == {
        "summary/test": {"sources": 86, "sources_total": 130, "pairs": 197, "pairs_total": 300},
        "summary/train": {"sources": 424, "sources_total": 633, "pairs": 1339, "pairs_total": 2000},
        "summary/val": {"sources": 76, "sources_total": 112, "pairs": 206, "pairs_total": 300},
    }
    partial = s["partial_containment"]
    assert partial["median"] == pytest.approx(0.019, abs=5e-4) and partial["max"] == pytest.approx(0.384, abs=5e-4)
    assert (partial["n_ge_0.10"], partial["n_ge_0.20"]) == (31, 18)

    assert b["halueval"]["qa"]["tiers"]["partial"] == 3 and b["halueval"]["qa"]["fields"]["knowledge"]["partial"] == 3
    assert b["halueval"]["qa"]["fields"]["knowledge"]["max_containment"] == pytest.approx(0.089, abs=5e-4)
    assert b["halueval"]["dialogue"]["tiers"]["none"] == 10000
    assert b["halueval"]["general"]["tiers"]["partial"] == 1
    assert b["faitheval"]["counterfactual"]["tiers"]["partial"] == 2
    assert b["faitheval"]["counterfactual"]["fields"]["context"]["max_containment"] == pytest.approx(0.007, abs=5e-4)
    assert b["faitheval"]["unanswerable"]["tiers"]["partial"] == 1
    assert b["faitheval"]["inconsistent"]["tiers"]["none"] == 1500
    assert b["truthfulqa"]["TruthfulQA.csv"]["tiers"]["none"] == 790
    if harness is not None:
        assert b["harness"]["truthfulqa"]["tiers"] == {"exact": 0, "near_duplicate": 0, "partial": 0, "none": 817}

    rb = b["ragtruth_benchmark"]
    assert (rb["test"]["identity"]["train"], rb["test"]["identity"]["val"], rb["test"]["identity"]["test"]) == (0, 0, 368)
    assert (rb["all"]["identity"]["train"], rb["all"]["identity"]["val"]) == (1845, 332)
    internal = rb["internal_test_vs_train"]
    assert {d["source_id"] for d in internal["Summary"]["near_duplicates"]} == {"15786", "11911"}
    assert internal["Summary"]["near_duplicate"] + internal["Summary"]["partial"] == 29
    assert internal["QA"]["partial"] == 7
    assert internal["Data2txt"]["partial"] == internal["Data2txt"]["n_sources"] == 150

    lists = sorted(p.name for p in (tmp_path / "decon" / "ragtruth").glob("*__*.json"))
    assert lists == ["halueval__summarization.json"]
    fresh =json.loads((tmp_path / "decon" / "ragtruth" / lists[0]).read_text(encoding="utf-8"))
    assert (fresh["n_rows"], fresh["n_excluded"], fresh["n_unseen_exposed"]) == (10000, 503, 130)
    committed = overlap.ROOT / "decontamination" / "ragtruth" / lists[0]
    if committed.is_file():  # the committed list must still be what the checker produces
        stored = json.loads(committed.read_text(encoding="utf-8"))
        assert stored["indices"] == fresh["indices"]
        assert stored["unseen_exposed_indices"] == fresh["unseen_exposed_indices"]
