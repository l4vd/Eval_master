"""Copy GPU-produced modified-protocol artifacts from a legacy root into overview sibling dirs.

**Optional convenience for the all-variants overview — the official procedure never needs it.**
The legacy layout keeps modified protocols in their own root (``$EVAL_ROOT_MOD``), flat in the
benchmark dir. The overview keeps them next to the originals, one dir per variant. This copies
what costs GPU time, so the overview does not recompute it:

* ``halueval/*_constrained_{results,summary}.json`` -> ``halueval.constrained/``
* ``faitheval/counterfactual_mc_*``                  -> ``faitheval.mc/``

Safety rules: only into run dirs that exist in the target root with the same
``run_metadata.json`` ``model_id`` and (HaluEval) the same file label as the target's
original files; never overwrites; copies, never moves, so the source root stays valid for
the official §5.8 analysis. CPU artifacts (decontam summaries) are not copied — ``derive``
recomputes them. FaithEval generation tasks are never copied from a modified root: a legacy
``strict_match`` run did not record the flag, so it cannot be told apart from an original.

    python -m analysis.adopt --from outputs/eval/v4-2_modified --into outputs/eval/v4-2 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from analysis import variants as V

_CONSTRAINED = ("_constrained_results.json", "_constrained_summary.json")


def _model_id(run_dir: Path) -> str | None:
    try:
        return json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8")).get("model_id")
    except (OSError, ValueError):
        return None


def _labels(halueval_dir: Path) -> set[str]:
    """Labels of the original HaluEval summaries in a dir (``<task>_<label>_summary.json``)."""
    out = set()
    for task in V.HALUEVAL_TASKS:
        for path in V._halueval_summaries(halueval_dir, task, "_summary.json"):
            out.add(path.name[len(task) + 1: -len("_summary.json")])
    return out


def _constrained_label(name: str) -> str | None:
    for suffix in _CONSTRAINED:
        if name.endswith(suffix):
            task, _, rest = name.partition("_")
            if task in V.HALUEVAL_TASKS:
                return rest[: -len(suffix)]
    return None


def plan_adoption(src_root: Path, dst_root: Path) -> tuple[list[tuple[Path, Path]], list[str]]:
    """``[(source file, target file)]`` to copy, and the reasons anything was skipped."""
    copies: list[tuple[Path, Path]] = []
    skipped: list[str] = []
    for src_run in V.run_dirs_under(src_root):
        rel = src_run.relative_to(src_root)
        dst_run = dst_root / rel
        candidates = []
        hal = src_run / "halueval"
        if hal.is_dir():
            candidates += [(p, "halueval.constrained") for p in sorted(hal.iterdir())
                           if _constrained_label(p.name) is not None]
        fe = src_run / "faitheval"
        if fe.is_dir():
            candidates += [(p, "faitheval.mc") for p in sorted(fe.glob("counterfactual_mc_*"))]
        if not candidates:
            continue
        if not dst_run.is_dir():
            skipped.append(f"{rel}: no such run dir in the target root")
            continue
        src_model, dst_model = _model_id(src_run), _model_id(dst_run)
        if src_model is None or src_model != dst_model:
            skipped.append(f"{rel}: model_id differs or is unrecorded ({src_model!r} vs {dst_model!r})")
            continue
        labels = _labels(dst_run / "halueval")
        for path, variant in candidates:
            target = dst_run / variant / path.name
            if variant == "halueval.constrained" and _constrained_label(path.name) not in labels:
                skipped.append(f"{rel}/{path.name}: label not among the target's original files")
            elif target.exists():
                skipped.append(f"{rel}/{variant}/{path.name}: exists, not overwritten")
            else:
                copies.append((path, target))
    return copies, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=f"Adopt legacy modified-protocol GPU artifacts into overview "
                                             f"sibling dirs ({V.OVERVIEW_LABEL}).")
    ap.add_argument("--from", dest="src", required=True, type=Path, help="legacy modified-protocol root")
    ap.add_argument("--into", dest="dst", required=True, type=Path, help="root holding the originals")
    ap.add_argument("--dry-run", action="store_true", help="list the copies, copy nothing")
    args = ap.parse_args(argv)
    if args.src.resolve() == args.dst.resolve():
        raise SystemExit("--from and --into must be different roots")

    copies, skipped = plan_adoption(args.src, args.dst)
    for src, dst in copies:
        print(f"{'would copy' if args.dry_run else 'copy'} {src.relative_to(args.src)} -> "
              f"{dst.relative_to(args.dst)}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for reason in skipped:
        print(f"skip {reason}")
    print(f"==> {len(copies)} file(s) {'to copy' if args.dry_run else 'copied'}, {len(skipped)} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
