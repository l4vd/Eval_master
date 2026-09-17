"""Derive every CPU variant of the optional overview over an existing eval root. No GPU.

Walks ``R/*/seed_*`` and ``R/*/run_*`` (which covers ``R/base/run_*``) and, for each run
dir, (re)derives every CPU unit (``halueval.parsed``, ``.lenient``, ``.decontam``,
``.constrained_decontam``, ``faitheval.strict``, ``.wordmatch``, ``.contains``) that is
missing or stale and whose source finished. Writes only into the sibling
``<run>/<bench>.<variant>/`` dirs; original dirs are read, never written.

**Overview only — descriptive, not pre-registered.** The official decontamination
workflow (``run_decontam.sh`` into ``$EVAL_ROOT_MOD``) is unchanged.

    python -m analysis.derive --root outputs/eval/v3 [--variants all] [--dry-run]

Interpreters: ``score_results.py`` is stdlib and runs with this interpreter;
``rescore.py`` needs only PyYAML and runs with ``--faitheval-python`` if given, else this
interpreter. It deliberately does not use the FaithEval venv the launcher would resolve.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from analysis import variants as V


def check_yaml(python: str) -> None:
    proc = subprocess.run([python, "-c", "import yaml"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"error: {python} cannot `import yaml`, which the FaithEval re-scorer needs. "
                         "Pass --faitheval-python with an interpreter that has PyYAML "
                         "(the Eval_master .venv does).")


def derive_root(root: Path, variant_names, *, ctx: V.Context | None = None, dry_run: bool = False,
                faitheval_python: str | None = None, resume: bool = True) -> dict[str, list]:
    ctx = ctx or V.Context()
    totals: dict[str, list] = {"ok": [], "failed": [], "unavailable": [], "waiting": [], "planned": []}
    run_dirs = V.run_dirs_under(root)
    if not run_dirs:
        raise SystemExit(f"no <arm>/seed_* or <arm>/run_* dirs under {root}")
    for run_dir in run_dirs:
        chosen, explicit = V.resolve_variants(variant_names, V.present_bases(run_dir))
        cpu = [v for v in chosen if v.kind == V.CPU]
        if not cpu:
            continue
        units = [u for u in V.plan(run_dir, chosen, ctx, explicit=explicit) if u.variant.kind == V.CPU]
        print(f"== {run_dir.parent.name}/{run_dir.name}")
        result = V.derive_run_dir(run_dir, units, ctx, resume=resume, dry_run=dry_run,
                                  faitheval_python=faitheval_python)
        for key, items in result.items():
            totals[key].extend((str(run_dir), *item) for item in items)
    return totals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"Derive the overview's CPU variants over a root ({V.OVERVIEW_LABEL}).")
    ap.add_argument("--root", required=True, type=Path, help="eval root holding <arm>/<run>/ dirs")
    ap.add_argument("--variants", default="all", help="'all' or comma-separated variant names")
    ap.add_argument("--faitheval-python", default=None,
                    help="interpreter for FaithEval-reproduce/src/rescore.py (needs PyYAML; default: this one)")
    ap.add_argument("--faitheval-data-dir", type=Path, default=None,
                    help="FaithEval JSONL root (default: $FAITHEVAL_DATA_DIR or the repo's data/)")
    ap.add_argument("--no-resume", action="store_true", help="re-derive units that are already done")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, write nothing")
    args = ap.parse_args(argv)

    names = args.variants.split(",")
    faitheval_python = args.faitheval_python or sys.executable
    if not args.dry_run and any(V.REGISTRY[n].base == "faitheval" for n in V.resolve_variants(
            names, ["faitheval", "halueval"])[1]):
        check_yaml(faitheval_python)
    ctx = V.Context(faitheval_data_dir=args.faitheval_data_dir)
    totals = derive_root(args.root, names, ctx=ctx, dry_run=args.dry_run,
                         faitheval_python=faitheval_python, resume=not args.no_resume)
    print(f"\n==> {V.OVERVIEW_LABEL}")
    for key in ("planned", "ok", "failed", "unavailable", "waiting"):
        if totals[key]:
            print(f"    {key}: {len(totals[key])} unit(s)")
    for item in totals["failed"]:
        print(f"    FAILED {item}")
    return 1 if totals["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
