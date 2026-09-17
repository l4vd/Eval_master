"""Optional adoption of legacy modified-protocol GPU artifacts into overview sibling dirs."""

from __future__ import annotations

import json

from analysis import adopt, fixtures


def _snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _roots(tmp_path, *, model="ckpt", target_model="ckpt"):
    src, dst = tmp_path / "mod", tmp_path / "orig"
    for root, mid in ((src, model), (dst, target_model)):
        run = root / "arm" / "seed_1"
        run.mkdir(parents=True)
        (run / "run_metadata.json").write_text(json.dumps({"seed": 1, "model_id": mid}), encoding="utf-8")
    fixtures.write_halueval(dst / "arm" / "seed_1" / "halueval", {"qa": 0.5}, label="L")
    mod = src / "arm" / "seed_1"
    fixtures.write_halueval_constrained(mod / "halueval", {"qa": 0.7}, label="L")
    (mod / "halueval" / "qa_L_constrained_results.json").write_text("{}\n", encoding="utf-8")
    fixtures.write_halueval_decontam(mod / "halueval", {"summarization": (0.5, 0.4)}, label="L", constrained=True)
    fixtures.write_faitheval_mc(mod / "faitheval", 0.6)
    (mod / "faitheval" / "counterfactual_mc_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    # a legacy strict_match run is indistinguishable from an original: never adopted
    fixtures.write_faitheval(mod / "faitheval", {"unanswerable": 0.1})
    return src, dst


def test_gpu_artifacts_are_copied_into_sibling_dirs_and_the_source_is_unchanged(tmp_path):
    src, dst = _roots(tmp_path)
    before = _snapshot(src)
    assert adopt.main(["--from", str(src), "--into", str(dst)]) == 0
    run = dst / "arm" / "seed_1"
    assert sorted(p.name for p in (run / "halueval.constrained").iterdir()) == [
        "qa_L_constrained_results.json", "qa_L_constrained_summary.json"]
    assert sorted(p.name for p in (run / "faitheval.mc").iterdir()) == [
        "counterfactual_mc_predictions.jsonl", "counterfactual_mc_summary.json"]
    assert not (run / "faitheval").exists()  # FaithEval generation files are never copied
    assert not list(dst.rglob("*decontam*"))  # CPU artifacts are re-derived, not copied
    assert _snapshot(src) == before


def test_never_overwrites_and_dry_run_copies_nothing(tmp_path):
    src, dst = _roots(tmp_path)
    assert adopt.main(["--from", str(src), "--into", str(dst), "--dry-run"]) == 0
    assert not (dst / "arm" / "seed_1" / "faitheval.mc").exists()
    target = dst / "arm" / "seed_1" / "faitheval.mc" / "counterfactual_mc_summary.json"
    target.parent.mkdir(parents=True)
    target.write_text("mine", encoding="utf-8")
    copies, skipped = adopt.plan_adoption(src, dst)
    assert target not in [d for _, d in copies] and any("not overwritten" in s for s in skipped)
    adopt.main(["--from", str(src), "--into", str(dst)])
    assert target.read_text(encoding="utf-8") == "mine"


def test_a_different_model_or_label_is_not_adopted(tmp_path):
    src, dst = _roots(tmp_path / "a", target_model="other")
    copies, skipped = adopt.plan_adoption(src, dst)
    assert copies == [] and "model_id differs" in skipped[0]

    src, dst = _roots(tmp_path / "b")
    for p in (dst / "arm" / "seed_1" / "halueval").iterdir():
        p.rename(p.with_name(p.name.replace("_L_", "_M_")))
    copies, skipped = adopt.plan_adoption(src, dst)
    assert {d.parent.name for _, d in copies} == {"faitheval.mc"}
    assert sum("label" in s for s in skipped) == 2
