"""Evaluate an ensemble of TRAINING checkpoints through the 5-benchmark launcher.

This is the missing bridge between the two sides of the repo:

* the **training** side (SP-DPO-Base) writes ``<...>_<METHOD>_ensemble/seed_<SEED>/``
  group dirs, each seed holding a ``final_checkpoint/`` (the trained model) — but *no*
  Eval_master benchmark results; and
* the **analysis** side ([`discover.py`](discover.py) → aggregate → compare) only ever
  reads run dirs that already contain the per-benchmark subfolders
  (``faitheval/``, ``harness/``, ...). It never loads a model.

Pointing ``analysis.cli --arm dpo=.../seed_*`` straight at a *training* group dir
therefore finds zero metric records — the ensemble "isn't recognised". This module
closes the gap: for each seed checkpoint it invokes [`run_benchmarks.py`](../run_benchmarks.py),
writing outputs to a **seed-preserving** layout ``<out>/seed_<SEED>/`` so that
``analysis.discover`` recovers the *same* seed value and the paired comparison lines up
by seed identity (never by list position). A ``run_metadata.json`` carrying the seed +
source checkpoint is dropped into each output dir so discovery is robust even if the
launcher never writes one itself.

Typical use (produce the DPO ensemble's eval outputs, then compare against base)::

    python -m analysis.eval_checkpoints \
        --checkpoints 'outputs/.../08-48-52_sft_ensemble/seed_*' \
        --out outputs/eval/sft_ensemble \
        --analyze --name sft --arm base=outputs/.../20-38-30 --reference base

Or produce the eval outputs only and hand them to ``run_analysis.sh`` yourself.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

from analysis import variants as V
from analysis.discover import infer_seed

# The Hydra launcher lives one level up from this package (mirrors cli._LAUNCHER).
_LAUNCHER = Path(__file__).resolve().parents[1] / "run_benchmarks.py"

# The five benchmarks the launcher knows, in the launcher's own default order.
_DEFAULT_BENCHMARKS = ("faitheval", "truthfulqa", "halueval", "ragtruth", "harness")


@dataclass
class CheckpointEvalJob:
    """One checkpoint to evaluate: where the model is, and where its eval goes."""

    seed: int | None
    checkpoint: Path
    out_dir: Path
    command: list[str] = field(default_factory=list)
    benchmarks: tuple[str, ...] = _DEFAULT_BENCHMARKS


def completed_benchmarks(
    out_dir: Path, benchmarks: tuple[str, ...] | list[str], overrides: list[str] | None = None,
) -> list[str]:
    """Which of `benchmarks` already have finished results under `out_dir`.

    All three launchers write their ``*summary*.json`` only after the benchmark
    finishes (``<task>_summary.json`` for FaithEval, ``<task>_<label>_summary.json``
    for HaluEval, ``summary.json`` for the harness), so its presence — unlike that of
    the incrementally-appended prediction files — means the benchmark ran to
    completion. A seed killed mid-benchmark therefore correctly reads as incomplete.

    Which summary counts depends on the variant the run was launched with, read from its
    Hydra ``overrides`` (the job's command line): ``halueval.scoring=constrained`` needs
    a ``*_constrained_summary.json`` (``both`` needs that and an original one), a
    ``faitheval.tasks`` list naming ``counterfactual_mc`` needs
    ``counterfactual_mc_summary.json``, and ``ragtruth.stage=generate`` needs
    ``generation_summary.json`` — otherwise a finished generate-only run would read as a
    finished benchmark, and a modified run as the original.
    """
    options = _override_map(overrides or [])
    done = []
    for name in benchmarks:
        bench_dir = out_dir / name
        if bench_dir.is_dir() and _is_complete(name, bench_dir, options):
            done.append(name)
    return done


_VARIANT_SUMMARY_SUFFIXES = ("_constrained_summary.json", "_decontam_summary.json")


def _override_map(overrides: list[str]) -> dict[str, str]:
    options = {}
    for token in overrides:
        key, sep, value = token.lstrip("+").partition("=")
        if sep:
            options[key] = value.strip("'\"")
    return options


def _list_override(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [t.strip().strip("'\"") for t in value.strip("[]").split(",") if t.strip()]


def _is_complete(name: str, bench_dir: Path, options: dict[str, str]) -> bool:
    summaries = [p.name for p in bench_dir.glob("*summary*.json")]
    if name == "halueval":
        original = any(s.endswith("_summary.json") and not s.endswith(_VARIANT_SUMMARY_SUFFIXES)
                       for s in summaries)
        constrained = any(s.endswith("_constrained_summary.json") for s in summaries)
        scoring = options.get("halueval.scoring", "generate")
        return {"constrained": constrained, "both": original and constrained}.get(scoring, original)
    if name == "faitheval":
        tasks = _list_override(options.get("faitheval.tasks"))
        if tasks and "counterfactual_mc" in tasks:
            others = [t for t in tasks if t != "counterfactual_mc"]
            return (bench_dir / "counterfactual_mc_summary.json").is_file() and (
                not others or any((bench_dir / f"{t}_summary.json").is_file() for t in others))
        return any(s != "counterfactual_mc_summary.json" for s in summaries)
    if name == "ragtruth":
        if options.get("ragtruth.stage") == "generate":
            return (bench_dir / "generation_summary.json").is_file()
        return (bench_dir / "summary.json").is_file()
    return bool(summaries)


def overview_variants(command: list[str]) -> list[str] | None:
    """The `variants=` list of an overview launch, or None for a legacy one."""
    value = _override_map(command).get("variants")
    if value in (None, "", "null", "None"):
        return None
    return _list_override(value)


def overview_pending(job: "CheckpointEvalJob") -> list[str]:
    """Units of an overview launch that are missing or stale in the job's output dir.

    The launch settings are not known here (the launcher composes them), so this checks
    presence and CPU source hashes only; the launcher re-checks every unit's settings.
    """
    options = _override_map(job.command)
    tasks = {}
    for bench in ("halueval", "faitheval"):
        listed = _list_override(options.get(f"{bench}.tasks"))
        if listed:
            tasks[bench] = tuple(listed)
    ctx = V.Context(tasks=tasks, halueval_label=V.run_label(str(job.checkpoint)),
                    ragtruth_stage=options.get("ragtruth.stage", "all"))
    chosen, explicit = V.resolve_variants(overview_variants(job.command), list(job.benchmarks))
    return [f"{u.variant.name}/{u.task}" for u in V.plan(job.out_dir, chosen, ctx, explicit=explicit)
            if u.status.state in (V.MISSING, V.STALE)]


def _is_seed_checkpoint_dir(path: Path, checkpoint_subdir: str) -> bool:
    """A seed dir carries a ``<checkpoint_subdir>/`` model, or a recoverable seed.

    The latter (``infer_seed`` != None) accepts a ``seed_<N>`` dir name or a
    ``run_metadata.json`` ``"seed"`` — so a group dir (``..._ensemble``, no seed of its
    own) and its ``ensemble/`` sibling both correctly read as *not* seed dirs.
    """
    if (path / checkpoint_subdir).is_dir():
        return True
    return infer_seed(path) is not None


def _resolve_checkpoint(seed_dir: Path, checkpoint_subdir: str) -> Path:
    """The model to hand the launcher: ``<seed_dir>/<subdir>`` if present, else the dir."""
    ckpt = seed_dir / checkpoint_subdir
    return ckpt if ckpt.exists() else seed_dir


def discover_checkpoints(
    spec: str, *, checkpoint_subdir: str = "final_checkpoint"
) -> list[tuple[int | None, Path]]:
    """Resolve a checkpoint spec to ``[(seed, checkpoint_path), ...]``, sorted by seed.

    ``spec`` accepts the same three shapes as an arm spec, but resolves to *training*
    checkpoints rather than eval run dirs:

    * a **glob** (``.../seed_*``) → each matching seed dir;
    * a **group dir** (``..._ensemble/``) → its ``seed_*`` children (``ensemble/`` and
      other non-seed children drop out);
    * a **single seed dir** → itself.

    The checkpoint handed to the launcher is ``<seed_dir>/<checkpoint_subdir>`` when that
    exists, else the seed dir itself (for layouts that save weights at the seed root).
    """
    matches = [Path(m) for m in glob(str(spec))]
    if not matches:
        p = Path(spec)
        matches = [p] if p.exists() else []

    seed_dirs: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        rp = path.resolve()
        if rp not in seen:
            seen.add(rp)
            seed_dirs.append(path)

    for m in matches:
        if not m.is_dir():
            continue
        if _is_seed_checkpoint_dir(m, checkpoint_subdir):
            _add(m)
            continue
        # Container (group dir / multirun root): pick up its seed-checkpoint children.
        for child in sorted(m.iterdir()):
            if child.is_dir() and _is_seed_checkpoint_dir(child, checkpoint_subdir):
                _add(child)

    pairs = [(infer_seed(d), _resolve_checkpoint(d, checkpoint_subdir)) for d in seed_dirs]
    # Sort by seed (None last), so output ordering is stable and readable.
    return sorted(pairs, key=lambda p: (p[0] is None, p[0] if p[0] is not None else 0))


def build_eval_command(
    checkpoint: Path,
    out_dir: Path,
    *,
    launcher: Path = _LAUNCHER,
    python: str | None = None,
    benchmarks: tuple[str, ...] | list[str] = _DEFAULT_BENCHMARKS,
    num_samples: int | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Build the ``run_benchmarks.py`` invocation for one checkpoint.

    ``output_dir`` and ``hydra.run.dir`` are both pinned to ``out_dir`` so the benchmark
    subfolders *and* Hydra's own bookkeeping land in the seed-preserving location.
    """
    cmd = [
        python or sys.executable,
        str(launcher),
        f"model.id={checkpoint}",
        f"output_dir={out_dir}",
        f"hydra.run.dir={out_dir}",
        f"run=[{','.join(benchmarks)}]",
    ]
    if num_samples is not None:
        cmd.append(f"num_samples={num_samples}")
    cmd.extend(extra or [])
    return cmd


def validate_overrides(overrides: list[str]) -> None:
    """Fail before any seed runs if an extra Hydra override does not parse.

    Otherwise the launcher dies on its command line once per seed, and all that is left
    per seed is the ``run_metadata.json`` written ahead of it. The usual cause is a
    shell word-split list (``faitheval.tasks=[a,`` + ``b]``).
    """
    try:
        from hydra.core.override_parser.overrides_parser import OverridesParser
    except ImportError:  # launcher env without hydra: leave it to the launcher
        return
    parser = OverridesParser.create()
    bad = []
    for token in overrides:
        try:
            parser.parse_override(token)
        except Exception as exc:  # HydraException; message is the ANTLR error
            bad.append(f"  {token!r}: {str(exc).splitlines()[0]}")
    if bad:
        raise SystemExit(
            "Invalid --launcher-extra Hydra override(s):\n" + "\n".join(bad)
            + "\nA list value must reach the launcher as ONE argument "
              "(e.g. 'faitheval.tasks=[a, b]'), not split on its spaces.")


def plan_evaluations(
    spec: str,
    out_root: Path,
    *,
    checkpoint_subdir: str = "final_checkpoint",
    launcher: Path = _LAUNCHER,
    python: str | None = None,
    benchmarks: tuple[str, ...] | list[str] = _DEFAULT_BENCHMARKS,
    num_samples: int | None = None,
    extra: list[str] | None = None,
) -> list[CheckpointEvalJob]:
    """Turn a checkpoint spec into the list of eval jobs (no side effects).

    Output dirs preserve seed identity: ``<out_root>/seed_<SEED>`` when the seed is
    known, else ``<out_root>/run_<i>`` (a fixed point — it can't be seed-paired).
    """
    out_root = Path(out_root)
    jobs: list[CheckpointEvalJob] = []
    for i, (seed, ckpt) in enumerate(discover_checkpoints(spec, checkpoint_subdir=checkpoint_subdir)):
        out_dir = out_root / (f"seed_{seed}" if seed is not None else f"run_{i}")
        cmd = build_eval_command(
            ckpt, out_dir, launcher=launcher, python=python,
            benchmarks=benchmarks, num_samples=num_samples, extra=extra,
        )
        jobs.append(CheckpointEvalJob(seed=seed, checkpoint=ckpt, out_dir=out_dir,
                                      command=cmd, benchmarks=tuple(benchmarks)))
    return jobs


def _write_run_metadata(job: CheckpointEvalJob) -> None:
    """Drop a ``run_metadata.json`` so discovery recovers the seed even if renamed."""
    meta = {"seed": job.seed, "model_id": str(job.checkpoint), "source": "eval_checkpoints"}
    (job.out_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")


def run_evaluations(
    jobs: list[CheckpointEvalJob], *, dry_run: bool = False, continue_on_error: bool = True,
    resume: bool = False,
) -> list[int | None]:
    """Execute each eval job in turn; returns the seeds that FAILED (empty = all fine).

    Each job's ``out_dir`` is created and stamped with ``run_metadata.json`` *before* the
    launcher runs, so a crash mid-benchmark still leaves a discoverable, seed-tagged dir.
    With ``continue_on_error`` (default), a failed seed is reported but the rest proceed —
    a partial ensemble still aggregates on the analysis side.

    With ``resume``, a seed whose every requested benchmark already wrote a summary is
    skipped. Nothing here is checkpointed *within* a seed, so an ensemble truncated by a
    walltime kill otherwise re-runs the seeds it already paid for from scratch.

    The failure list is the return value (rather than the jobs, which the caller already
    holds) because it is what the exit code must be derived from: an ensemble where 3 of 5
    seeds died used to print the failures and then exit 0, so the caller aggregated a
    2-seed ensemble believing it had 5.
    """
    failures: list[int | None] = []
    for job in jobs:
        header = f"seed_{job.seed}" if job.seed is not None else "(no seed)"
        command = job.command
        if resume and overview_variants(job.command) is not None:
            # The optional all-variants overview resumes per unit inside the launcher.
            if not any(t.startswith("resume=") for t in command):
                command = command + ["resume=true"]
            if not dry_run:
                pending = overview_pending(job)
                if not pending:
                    print(f"\n==> [eval-checkpoints] {header}: SKIPPED (every overview unit "
                          f"already done in {job.out_dir})")
                    continue
                more = " ..." if len(pending) > 8 else ""
                print(f"\n==> [eval-checkpoints] {header}: {len(pending)} overview unit(s) to "
                      f"compute: {', '.join(pending[:8])}{more}")
        elif resume and not dry_run:
            done = completed_benchmarks(job.out_dir, job.benchmarks, job.command)
            if len(done) == len(job.benchmarks):
                print(f"\n==> [eval-checkpoints] {header}: SKIPPED (all "
                      f"{len(done)} benchmark(s) already complete in {job.out_dir})")
                continue
        print(f"\n==> [eval-checkpoints] {header}: {job.checkpoint}")
        print("    " + " ".join(command))
        if dry_run:
            continue
        job.out_dir.mkdir(parents=True, exist_ok=True)
        _write_run_metadata(job)
        proc = subprocess.run(command)
        if proc.returncode != 0:
            failures.append(job.seed)
            print(f"!! seed {job.seed} FAILED (exit {proc.returncode})")
            if not continue_on_error:
                raise SystemExit(proc.returncode)
    if failures and not dry_run:
        print(f"\n!! {len(failures)} of {len(jobs)} seed(s) failed: {failures} "
              f"(continue_on_error kept the run going)")
    return failures


def _run_analysis(
    produced_out: Path,
    *,
    name: str,
    extra_arms: list[str],
    reference: str | None,
    analysis_out: Path,
    no_plot: bool,
    passthrough: list[str],
) -> int:
    """Chain into ``analysis.cli`` with the produced ensemble as one arm."""
    from analysis.cli import main as analysis_main

    argv = ["--arm", f"{name}={produced_out}/seed_*", "--out", str(analysis_out)]
    for arm in extra_arms:
        argv += ["--arm", arm]
    if reference:
        argv += ["--reference", reference]
    if no_plot:
        argv.append("--no-plot")
    argv += passthrough
    print(f"\n==> [eval-checkpoints] analysing: python -m analysis.cli {' '.join(argv)}")
    return analysis_main(argv)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate an ensemble of training checkpoints through the 5-benchmark "
                    "launcher, into an analysis-ready seed_<SEED>/ layout.")
    ap.add_argument("--checkpoints", required=True, metavar="SPEC",
                    help="Training ensemble spec: a group dir (..._ensemble), a glob "
                         "(.../seed_*), or a single seed dir.")
    ap.add_argument("--out", required=True, metavar="DIR",
                    help="Root for the produced eval outputs (one seed_<SEED>/ subdir each).")
    ap.add_argument("--checkpoint-subdir", default="final_checkpoint",
                    help="Model subdir inside each seed dir (default: final_checkpoint; "
                         "falls back to the seed dir itself if absent).")
    ap.add_argument("--benchmarks", default=None,
                    help="Comma-separated benchmarks to run (default: all five).")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="Per-benchmark sample cap forwarded to the launcher (default: full).")
    ap.add_argument("--python", default=None,
                    help="Interpreter to run the launcher with (default: this one).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the planned launcher commands without running them.")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="Abort on the first failed seed (default: keep going).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip seeds whose requested benchmarks all already wrote a "
                         "summary — re-submit after a walltime kill without redoing "
                         "the seeds that finished. With variants= in --launcher-extra (the "
                         "optional overview), a seed is skipped only when no unit is missing, "
                         "and resume=true is forwarded so the launcher skips finished units.")
    ap.add_argument("--launcher-extra", nargs=argparse.REMAINDER, default=None,
                    help="Extra Hydra overrides forwarded to the launcher (must come last).")
    # --analyze group: chain straight into analysis.cli on the produced outputs.
    ap.add_argument("--analyze", action="store_true",
                    help="After evaluating, run analysis.cli on the produced ensemble.")
    ap.add_argument("--analyze-partial", action="store_true",
                    help="Allow --analyze even when some seeds failed. Off by default: the "
                         "across-seed statistics would otherwise silently describe fewer "
                         "seeds than the run asked for.")
    ap.add_argument("--name", default="dpo",
                    help="Arm name for the produced ensemble when --analyze (default: dpo).")
    ap.add_argument("--arm", action="append", default=None, metavar="NAME=SPEC",
                    help="Extra arm(s) for --analyze, e.g. base=<eval_dir>. Repeatable.")
    ap.add_argument("--reference", default=None, help="Reference arm for --analyze.")
    ap.add_argument("--analysis-out", default=None,
                    help="Output dir for --analyze (default: <out>/analysis).")
    ap.add_argument("--no-plot", action="store_true", help="Skip figures in --analyze.")
    ap.add_argument("--exclude", default=None,
                    help="Comma-separated benchmark exclude-list, forwarded to analysis.cli "
                         "for --analyze.")
    ap.add_argument("--primary-map", default=None,
                    help="YAML/JSON overriding the primary-metric-per-benchmark map, "
                         "forwarded to analysis.cli for --analyze.")
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="Bootstrap RNG seed, forwarded to analysis.cli for --analyze.")
    ap.add_argument("--allow-seed-mismatch", action="store_true",
                    help="Forwarded to analysis.cli for --analyze: intersect shared seeds "
                         "instead of failing loudly in paired mode.")
    args = ap.parse_args(argv)

    benchmarks = tuple(args.benchmarks.split(",")) if args.benchmarks else _DEFAULT_BENCHMARKS
    validate_overrides(args.launcher_extra or [])
    jobs = plan_evaluations(
        args.checkpoints, Path(args.out),
        checkpoint_subdir=args.checkpoint_subdir,
        python=args.python, benchmarks=benchmarks,
        num_samples=args.num_samples, extra=args.launcher_extra or [],
    )
    if not jobs:
        raise SystemExit(f"No checkpoints found under: {args.checkpoints}")

    print(f"==> {len(jobs)} checkpoint(s) to evaluate -> {args.out}")
    failures = run_evaluations(
        jobs, dry_run=args.dry_run, continue_on_error=not args.stop_on_error,
        resume=args.resume,
    )

    if args.dry_run:
        return 0

    if args.analyze:
        # Analysing a partial ensemble as if it were complete is the silent-wrong-number
        # case this whole module exists to avoid: the missing seeds simply do not appear,
        # so the arm quietly has fewer seeds than it claims to. Make it an explicit choice.
        if failures and not args.analyze_partial:
            print(
                f"\n!! Refusing to --analyze: {len(failures)} of {len(jobs)} seed(s) failed "
                f"({failures}), so the ensemble is incomplete and its across-seed statistics "
                "would silently describe fewer seeds than intended. Re-run the failed seeds "
                "(--resume skips the ones that finished), or pass --analyze-partial to "
                "analyse what did complete."
            )
            return 1
        analysis_out = Path(args.analysis_out) if args.analysis_out else Path(args.out) / "analysis"
        # The --benchmarks restriction applied to the eval step above must also apply
        # here, or an --arm pointed at a differently-scoped eval dir (e.g. a full
        # five-benchmark `base`) silently pulls in benchmarks the produced ensemble
        # was never evaluated on.
        passthrough: list[str] = []
        if args.benchmarks:
            passthrough += ["--benchmarks", args.benchmarks]
        if args.exclude:
            passthrough += ["--exclude", args.exclude]
        if args.primary_map:
            passthrough += ["--primary-map", args.primary_map]
        if args.allow_seed_mismatch:
            passthrough.append("--allow-seed-mismatch")
        if args.rng_seed:
            passthrough += ["--rng-seed", str(args.rng_seed)]
        rc = _run_analysis(
            Path(args.out), name=args.name, extra_arms=args.arm or [],
            reference=args.reference, analysis_out=analysis_out,
            no_plot=args.no_plot, passthrough=passthrough,
        )
        # A clean analysis over a knowingly-partial ensemble still exits non-zero: the
        # caller asked for N seeds and got fewer.
        return rc or (1 if failures else 0)

    print(f"\n==> Eval outputs under: {args.out}")
    print("    Next: analyse them, e.g.")
    print(f"      ./run_analysis.sh --arm {args.name}='{args.out}/seed_*' "
          f"--arm base=<base_eval_dir> --reference base --out outputs/analysis")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
