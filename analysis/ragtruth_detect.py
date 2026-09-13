"""Shared RAG-Truth detection job: one 13B detector load for every generate-only run dir.

The RAG-Truth benchmark's cost is Stage 2: ``stage=all`` reloads the 13B detector for every
checkpoint. The sweep instead runs ``ragtruth.stage=generate`` per checkpoint (the small
model, batched) and then this job once over the whole eval root. It finds
``<root>/*/seed_*/ragtruth/`` and ``<root>/base/run_*/ragtruth/`` and hands them to
``RAGTruth-reproduce/src/run_eval.py --stage detect --output-dirs ...``, which loads the
detector once, skips every directory that already has a ``summary.json`` (so re-submitting
resumes), and writes each directory's ``detections.jsonl`` / ``summary.json`` exactly as
``stage=all`` would.

Detector settings default to ``conf/ragtruth/default.yaml`` (id, base model, tokenizer,
batch size, seed); flags override them. Runs in Eval_master's venv; the detection itself
runs in RAGTruth-reproduce's own, resolved as the launcher resolves it.

    python -m analysis.ragtruth_detect --root "$EVAL_ROOT" --dtype bfloat16
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAGTRUTH_DIR = ROOT / "RAGTruth-reproduce"
_PATTERNS = ("*/seed_*/ragtruth", "base/run_*/ragtruth")


def run_dir_patterns(root: Path) -> list[str]:
    return [str(Path(root).resolve() / pattern) for pattern in _PATTERNS]


def _conf_detector() -> dict:
    from omegaconf import OmegaConf

    conf = OmegaConf.load(ROOT / "conf" / "ragtruth" / "default.yaml")
    return OmegaConf.to_container(conf.detector, resolve=True)  # type: ignore[return-value]


def build_command(python: str, root: Path, args: argparse.Namespace, detector: dict) -> list[str]:
    def opt(flag, value):
        return [] if value in (None, "") else [flag, str(value)]

    return (
        [python, "src/run_eval.py", "--stage", "detect", "--output-dirs", *run_dir_patterns(root)]
        + ["--detector-model-id", str(args.detector_model_id or detector["id"])]
        + opt("--detector-base-model-id", args.detector_base_model_id or detector.get("base_model_id"))
        + opt("--detector-tokenizer-id", args.detector_tokenizer_id or detector.get("tokenizer_id"))
        + opt("--detector-batch-size", args.detector_batch_size or detector.get("batch_size"))
        + opt("--detector-seed", args.detector_seed if args.detector_seed is not None else detector.get("seed"))
        + ["--dtype", args.dtype, "--device-map", args.device_map]
        + opt("--cache-dir", args.cache_dir)
        + list(args.extra or [])
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True, help="Eval root holding <arm>/seed_<N>/ragtruth/ (and base/run_0/).")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="Detector dtype; use the eval block's model.dtype.")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--detector-model-id", default=None, help="Default: conf ragtruth.detector.id.")
    ap.add_argument("--detector-base-model-id", default=None)
    ap.add_argument("--detector-tokenizer-id", default=None)
    ap.add_argument("--detector-batch-size", type=int, default=None, help="Default: conf ragtruth.detector.batch_size.")
    ap.add_argument("--detector-seed", type=int, default=None, help="Default: conf ragtruth.detector.seed.")
    ap.add_argument("--python", default="auto",
                    help="Interpreter for RAGTruth-reproduce (default: its own venv, as the launcher finds it).")
    ap.add_argument("--dry-run", action="store_true", help="List the run dirs and print the command only.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=None,
                    help="Raw flags forwarded to run_eval.py (must come last).")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"--root {root} is not a directory")
    dirs = sorted({p for pattern in _PATTERNS for p in root.glob(pattern) if p.is_dir()})
    pending = [d for d in dirs if (d / "generations.jsonl").is_file() and not (d / "summary.json").is_file()]
    print(f"==> {len(dirs)} RAG-Truth run dir(s) under {root}: {len(pending)} awaiting detection")
    for d in pending:
        print(f"    {d}")

    import os

    from run_benchmarks import _resolve_interpreter  # the launcher's own venv resolution

    python = _resolve_interpreter(args.python, RAGTRUTH_DIR, os.environ.get("VENV_ROOT"), dry_run=args.dry_run)
    cmd = build_command(python, root, args, _conf_detector())
    print("    " + " ".join(cmd))
    if args.dry_run or not pending:
        return 0
    return subprocess.run(cmd, cwd=str(RAGTRUTH_DIR)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
