"""Report writers: JSON keys, LaTeX rendering, records round-trip."""

from __future__ import annotations

import json

from analysis.aggregate import aggregate_all
from analysis.model import RecordSet
from analysis.parse import parse_run_dir
from analysis.report import (
    aggregate_to_latex,
    load_records,
    write_aggregate,
    write_records,
)
from analysis import fixtures


def _records(tmp_path):
    recs = []
    for seed, acc in {42: 0.3, 7: 0.5}.items():
        run = fixtures.write_full_run(tmp_path / f"dpo_{seed}", seed=seed,
                                      faitheval={"counterfactual": acc},
                                      ragtruth_rate=0.2)
        recs += parse_run_dir(run, "dpo", seed)
    return RecordSet(recs)


def test_records_round_trip(tmp_path):
    rs = _records(tmp_path)
    path = write_records(rs, tmp_path / "records.jsonl")
    back = load_records(path)
    assert len(back) == len(rs)
    assert {r.key for r in back} == {r.key for r in rs}


def test_write_aggregate_json_and_tex(tmp_path):
    rs = _records(tmp_path)
    arms = aggregate_all(rs)
    paths = write_aggregate(arms, tmp_path / "analysis")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "dpo" in payload
    tex = paths["tex"].read_text(encoding="utf-8")
    assert r"\begin{tabular}" in tex and r"\bottomrule" in tex


def test_latex_marks_lower_is_better_and_escapes(tmp_path):
    rs = _records(tmp_path)
    arms = aggregate_all(rs)
    tex = aggregate_to_latex(arms, primary_only=False)
    assert r"\downarrow" in tex               # ragtruth hallucination_rate marked
    assert r"\_" in tex or "faitheval" in tex  # underscores escaped where present
