"""Command-line entry point for the analysis layer.

Ties parsing -> aggregation -> comparison -> plotting together, honouring the toggles:
``--no-compare`` (aggregate only), ``--no-plot`` (skip figures), and per-benchmark
selection (``--benchmarks`` / ``--exclude``). A single ``--arm`` with ``--no-compare`` is
the single-ensemble path; N arms drive the multi-arm comparison against a reference.

``--run-evals`` reuses the existing Hydra launcher via ``--multirun`` to *produce* run
dirs before analysing them (no new launch plumbing) — see :func:`run_evals`.

Examples
--------
    # aggregate one ensemble, no comparison, with plots
    python -m analysis.cli --arm dpo='outputs/.../seed_*' --no-compare

    # base vs SFT vs DPO ensembles, full compare + plots
    python -m analysis.cli \
        --arm base=outputs/.../base_run \
        --arm sft='outputs/.../*_sft_ensemble/seed_*' \
        --arm dpo='outputs/.../*_dpo_ensemble/seed_*' \
        --reference base --out outputs/analysis
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from analysis.aggregate import aggregate_all
from analysis.compare import compare_all
from analysis.plot import plot_all
from analysis.report import write_aggregate, write_comparisons, write_records
from analysis.spec import AnalysisConfig, ArmMeta, ArmSpec, build_records, parse_arm_arg

# The launcher lives one level up from this package.
_LAUNCHER = Path(__file__).resolve().parents[1] / "run_benchmarks.py"


def default_reference(arm_meta: dict[str, ArmMeta]) -> str:
    """Prefer an arm literally named 'base'; else a fixed-point arm; else the first."""
    if "base" in arm_meta:
        return "base"
    for name, meta in arm_meta.items():
        if meta.is_fixed_point:
            return name
    return next(iter(arm_meta))


def run_analysis(config: AnalysisConfig) -> dict[str, list[Path]]:
    """Run the whole pipeline for one config; returns the written paths per artifact."""
    build = build_records(config)
    outdir = Path(config.out)
    aggregates = aggregate_all(build.records, seed=config.rng_seed)

    written: dict[str, list[Path]] = {
        "records": [write_records(build.records, outdir / "records.jsonl")],
        "aggregate": list(write_aggregate(aggregates, outdir).values()),
    }

    comparisons = None
    if config.compare and len(build.arm_meta) >= 2:
        reference = config.reference or default_reference(build.arm_meta)
        comparisons = compare_all(
            aggregates, build.arm_meta, reference,
            require_matched=config.require_matched, rng_seed=config.rng_seed,
        )
        written["comparisons"] = list(write_comparisons(comparisons, outdir).values())

    if config.plot:
        figs = plot_all(
            build.records, outdir / "figures",
            reference=config.reference, comparisons=comparisons,
        )
        written["figures"] = figs

    return written


def run_evals(models: list[str], sweep_dir: Path, extra: list[str], *, dry_run: bool) -> Path:
    """Drive the Hydra launcher's ``--multirun`` to produce run dirs for a model list.

    Produces ``<sweep_dir>/{0,1,...}/`` — one numbered run dir per model, each with the
    five benchmark subfolders. Point ``--arm`` at those to analyse them. Requires the
    launcher + per-benchmark venvs (not exercised by the offline test suite).
    """
    ids = ",".join(models)
    cmd = [
        sys.executable, str(_LAUNCHER), "--multirun",
        f"model.id={ids}", f"hydra.sweep.dir={sweep_dir}", *extra,
    ]
    print("[run-evals]", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)
    return sweep_dir


def _load_primary_map(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".json",):
        return json.loads(text)
    # YAML via omegaconf (a launcher dependency) to avoid a hard PyYAML requirement.
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(p), resolve=True)  # type: ignore[return-value]


def build_config(args: argparse.Namespace) -> AnalysisConfig:
    arms = [parse_arm_arg(a) for a in (args.arm or [])]
    if args.run_evals:
        # Analyse the freshly produced numbered sweep subdirs as one arm each,
        # unless the user already declared arms explicitly.
        sweep = run_evals(args.models.split(","), Path(args.eval_sweep_dir),
                          args.eval_extra or [], dry_run=args.dry_run)
        if not arms:
            models = args.models.split(",")
            arms = [ArmSpec(name=f"model{i}", spec=str(sweep / str(i)))
                    for i in range(len(models))]
    if not arms:
        raise SystemExit("No arms given. Use --arm NAME=SPEC (repeatable) or --run-evals.")

    return AnalysisConfig(
        arms=arms,
        out=args.out,
        reference=args.reference,
        benchmarks=args.benchmarks.split(",") if args.benchmarks else None,
        exclude=args.exclude.split(",") if args.exclude else None,
        compare=not args.no_compare,
        plot=not args.no_plot,
        require_matched=not args.allow_seed_mismatch,
        primary_map=_load_primary_map(args.primary_map),
        rng_seed=args.rng_seed,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate + compare + plot Eval_master benchmark results.")
    ap.add_argument("--arm", action="append", metavar="NAME=SPEC",
                    help="Arm as NAME=SPEC (dir/group-dir/glob). Repeatable. "
                         "Append !fixed/!ensemble to force fixed-point-ness.")
    ap.add_argument("--out", default="outputs/analysis", help="Analysis output dir.")
    ap.add_argument("--reference", default=None,
                    help="Reference arm for comparisons (default: 'base' / a fixed arm / first).")
    ap.add_argument("--benchmarks", default=None, help="Comma-separated include list.")
    ap.add_argument("--exclude", default=None, help="Comma-separated exclude list.")
    ap.add_argument("--no-compare", action="store_true",
                    help="Aggregate each arm only; skip cross-arm comparison.")
    ap.add_argument("--no-plot", action="store_true", help="Skip figures.")
    ap.add_argument("--allow-seed-mismatch", action="store_true",
                    help="In paired mode, intersect shared seeds instead of failing loudly.")
    ap.add_argument("--primary-map", default=None,
                    help="YAML/JSON overriding the primary-metric-per-benchmark map.")
    ap.add_argument("--rng-seed", type=int, default=0, help="Bootstrap RNG seed.")
    # --run-evals group
    ap.add_argument("--run-evals", action="store_true",
                    help="Drive the Hydra launcher --multirun over --models first.")
    ap.add_argument("--models", default=None, help="Comma-separated model ids for --run-evals.")
    ap.add_argument("--eval-sweep-dir", default="outputs/analysis_sweep",
                    help="Hydra sweep dir for --run-evals.")
    ap.add_argument("--eval-extra", nargs=argparse.REMAINDER, default=None,
                    help="Extra Hydra overrides forwarded to the launcher (must come last).")
    ap.add_argument("--dry-run", action="store_true",
                    help="With --run-evals, print the launcher command without running it.")
    args = ap.parse_args(argv)

    if args.run_evals and not args.models:
        ap.error("--run-evals requires --models")

    config = build_config(args)
    written = run_analysis(config)
    print(f"\nAnalysis written under: {config.out}")
    for kind, paths in written.items():
        print(f"  {kind}: {len(paths)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
