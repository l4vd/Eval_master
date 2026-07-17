#!/usr/bin/env python
"""Central Hydra launcher for the five benchmarks.

The benchmarks are separate repos with unified model interfaces (a Hub id, a
local checkpoint, or a LoRA/PEFT adapter). This launcher composes one shared
`model` config plus a per-benchmark config group, translates them into each
benchmark's own CLI, and runs the benchmarks as subprocesses in their folders.

Prefer the `./run_all.sh` wrapper; this file can also be called directly:
    python run_benchmarks.py model.id=/path/to/final_checkpoint run='[faitheval,ragtruth]'

See conf/config.yaml for all options, or `README-runner.md`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Benchmark folders resolve relative to THIS file, so the launcher works from any
# working directory (Hydra's `job.chdir: false` keeps cwd here, but we don't rely
# on it).
ROOT = Path(__file__).resolve().parent
FOLDERS = {
    "faitheval": ROOT / "FaithEval-reproduce",
    "truthfulqa": ROOT / "TruthfulQA-reproduce",
    "halueval": ROOT / "HaluEval-reproduce",
    "ragtruth": ROOT / "RAGTruth-reproduce",
    "harness": ROOT / "harness-eval",
}


def _opt(flag: str, value) -> list[str]:
    """`[flag, str(value)]` if value is set, else `[]` (skips None/empty)."""
    if value is None or value == "":
        return []
    return [flag, str(value)]


def _venv_python(venv: Path) -> Path | None:
    """The interpreter inside `venv`, or None if there isn't one.

    Checks the Windows layout and the POSIX one; `uv` creates whichever matches
    the platform.
    """
    for rel in ("bin/python", "Scripts/python.exe"):
        candidate = venv / rel
        if candidate.exists():
            return candidate
    return None


def _candidate_venvs(folder: Path, venv_root) -> list[Path]:
    """Where a benchmark's virtualenv may live, most specific first.

    Default is the benchmark's own `<folder>/.venv`. `venv_root` relocates the
    envs to `<venv_root>/<folder-name>`, which is needed on Windows when the repo
    sits deep enough that `.venv/Lib/site-packages/...` would exceed the 260-char
    MAX_PATH limit (and keeps multi-GB installs out of a synced folder).
    """
    candidates = []
    if venv_root not in (None, ""):
        candidates.append(Path(str(venv_root)).expanduser() / folder.name)
    candidates.append(folder / ".venv")
    return candidates


def _resolve_interpreter(spec, folder: Path, venv_root, *, dry_run: bool) -> str:
    """Resolve a `python:` config value to an interpreter path.

    `auto` (the default) finds the benchmark's own virtualenv, so each benchmark
    runs in its own environment — they have incompatible dependency stacks. Any
    other value is used verbatim, so `python=python` still forces one shared
    interpreter.
    """
    if spec != "auto":
        return str(spec)

    candidates = _candidate_venvs(folder, venv_root)
    for venv in candidates:
        found = _venv_python(venv)
        if found is not None:
            return str(found)

    searched = "\n".join(f"    - {c}" for c in candidates)
    msg = (
        f"No virtualenv found for {folder.name}. Searched:\n{searched}\n"
        f"  Run ./setup_envs.sh to create the per-benchmark environments, or pass an "
        f"explicit interpreter (e.g. python=python)."
    )
    if dry_run:
        # dry_run is how you inspect the commands before installing anything.
        print(f"!! {msg}\n   (dry-run: falling back to 'python' for display)")
        return "python"
    raise FileNotFoundError(msg)


def _resolve_samples(bench_cfg: DictConfig, cfg: DictConfig):
    """Per-benchmark num_samples, falling back to the global default."""
    val = bench_cfg.get("num_samples", None)
    return val if val is not None else cfg.get("num_samples", None)


def _model_common_map(cfg: DictConfig) -> list[str]:
    """Shared model flags using the `--base-model-id/--tokenizer-id/--cache-dir` naming."""
    m = cfg.model
    return (
        _opt("--base-model-id", m.base_model_id)
        + _opt("--tokenizer-id", m.tokenizer_id)
        + _opt("--cache-dir", m.cache_dir)
    )


def build_faitheval(cfg: DictConfig, out: Path) -> list[list[str]]:
    b = cfg.faitheval
    samples = _resolve_samples(b, cfg)
    cmds = []
    for task in b.tasks:
        cmd = (
            ["src/run_eval.py", "--task", str(task), "--model-id", str(cfg.model.id)]
            + _model_common_map(cfg)
            + ["--dtype", str(cfg.model.dtype), "--device-map", str(cfg.model.device_map)]
            + ["--split", str(b.split), "--max-new-tokens", str(b.max_new_tokens)]
            + _opt("--num-samples", samples)
            + (["--strict-match"] if b.strict_match else [])
            + ["--output-dir", str(out / "faitheval")]
            + list(b.extra_args)
        )
        cmds.append(cmd)
    return cmds


def build_truthfulqa(cfg: DictConfig, out: Path) -> list[list[str]]:
    b = cfg.truthfulqa
    cmd = (
        ["-m", "truthfulqa.evaluate", "--model_path", str(cfg.model.id)]
        + ["--metrics", *[str(x) for x in b.metrics]]
        + ["--preset", str(b.preset), "--input_path", str(b.input_path)]
        + ["--output_path", str(out / "truthfulqa" / "answers.csv")]
        + ["--prompt_style", str(b.prompt_style)]
        + ["--judge_backend", str(b.judge_backend)]
        + _opt("--truth_judge_id", b.truth_judge_id)
        + _opt("--info_judge_id", b.info_judge_id)
        # TruthfulQA uses its own flag spellings (--model_path/--base_model_id/...).
        + _opt("--base_model_id", cfg.model.base_model_id)
        + _opt("--tokenizer_id", cfg.model.tokenizer_id)
        + _opt("--cache_dir", cfg.model.cache_dir)
        + ["--dtype", str(cfg.model.dtype), "--device", str(cfg.model.device_index)]
        + list(b.extra_args)
    )
    return [cmd]


def build_halueval(cfg: DictConfig, out: Path) -> list[list[str]]:
    b = cfg.halueval
    samples = _resolve_samples(b, cfg)
    cmds = []
    for task in b.tasks:
        # HaluEval writes results under evaluation/<task>/; cwd is that folder.
        cmd = (
            ["evaluate.py", "--task", str(task), "--backend", str(b.backend)]
            + ["--model-path", str(cfg.model.id)]
            + _model_common_map(cfg)
            + ["--dtype", str(cfg.model.dtype), "--device-map", str(cfg.model.device_map)]
            + ["--max-new-tokens", str(b.max_new_tokens)]
            + _opt("--num-samples", samples)
            + list(b.extra_args)
        )
        cmds.append(cmd)
    return cmds


def build_ragtruth(cfg: DictConfig, out: Path) -> list[list[str]]:
    b = cfg.ragtruth
    samples = _resolve_samples(b, cfg)
    task_types = list(b.task_types) if b.task_types else None
    cmd = (
        ["src/run_eval.py", "--stage", str(b.stage)]
        + ["--model-id", str(cfg.model.id)]
        + _model_common_map(cfg)
        + ["--detector-model-id", str(b.detector.id)]
        + _opt("--detector-base-model-id", b.detector.base_model_id)
        + _opt("--detector-tokenizer-id", b.detector.tokenizer_id)
        + ["--dtype", str(cfg.model.dtype), "--device-map", str(cfg.model.device_map)]
        + _opt("--split", b.split)
        + _opt("--num-samples", samples)
        + (["--task-types", *task_types] if task_types else [])
        + (["--gold-f1"] if b.gold_f1 else [])
        + ["--output-dir", str(out / "ragtruth")]
        + list(b.extra_args)
    )
    return [cmd]


def build_harness(cfg: DictConfig, out: Path) -> list[list[str]]:
    b = cfg.harness
    samples = _resolve_samples(b, cfg)
    # One command for the whole task list: lm_eval loads the model once and
    # evaluates every task in a single pass. --tasks is nargs="+", terminated by
    # the following --dtype (no task name starts with "-").
    cmd = (
        ["src/run_eval.py", "--model-id", str(cfg.model.id)]
        + _model_common_map(cfg)
        + ["--tasks", *[str(t) for t in b.tasks]]
        + ["--dtype", str(cfg.model.dtype), "--device", str(cfg.model.device_index)]
        + _opt("--num-fewshot", b.num_fewshot)
        + _opt("--batch-size", b.batch_size)
        + _opt("--limit", samples)
        + _opt("--system-instruction", b.system_instruction)
        + (["--apply-chat-template"] if b.apply_chat_template else [])
        + (["--fewshot-as-multiturn"] if b.fewshot_as_multiturn else [])
        + (["--trust-remote-code"] if b.trust_remote_code else [])
        + (["--log-samples"] if b.log_samples else [])
        + ["--output-dir", str(out / "harness")]
        + list(b.extra_args)
    )
    return [cmd]


BUILDERS = {
    "faitheval": build_faitheval,
    "truthfulqa": build_truthfulqa,
    "halueval": build_halueval,
    "ragtruth": build_ragtruth,
    "harness": build_harness,
}
# Subdirectory (relative to a benchmark folder) to run each command from.
CWD_SUBDIR = {"halueval": "evaluation"}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    out = Path(cfg.output_dir).resolve()
    if not cfg.dry_run:
        # TruthfulQA's `--output_path` (a file) needs its parent to exist before
        # the subprocess writes to it; the others create their own --output-dir.
        for sub in ("faitheval", "truthfulqa", "ragtruth", "harness"):
            (out / sub).mkdir(parents=True, exist_ok=True)
    print(f"==> Output dir: {out}")
    print(f"==> Model: {cfg.model.id} (dtype={cfg.model.dtype})")

    results: list[tuple[str, str]] = []  # (label, status)
    for name in cfg.run:
        name = str(name)
        if name not in BUILDERS:
            print(f"!! Unknown benchmark '{name}' (choose from {list(BUILDERS)}); skipping.")
            results.append((name, "unknown"))
            continue

        bench_cfg = cfg[name]
        if not bench_cfg.get("enabled", True):
            print(f"== Skipping {name} (enabled=false)")
            results.append((name, "disabled"))
            continue

        folder = FOLDERS[name]
        interpreter = _resolve_interpreter(
            bench_cfg.get("python", None) or cfg.python,
            folder,
            cfg.get("venv_root", None),
            dry_run=cfg.dry_run,
        )
        cwd = folder / CWD_SUBDIR[name] if name in CWD_SUBDIR else folder

        for cmd in BUILDERS[name](cfg, out):
            full = [str(interpreter)] + cmd
            label = f"{name}: {' '.join(cmd[:3])}"
            print(f"\n==> [{name}] (cwd={cwd})\n    {' '.join(full)}")
            if cfg.dry_run:
                results.append((label, "dry-run"))
                continue
            proc = subprocess.run(full, cwd=str(cwd))
            status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
            results.append((label, status))
            if proc.returncode != 0 and not cfg.continue_on_error:
                _summary(results, out)
                sys.exit(proc.returncode)

    _summary(results, out)


def _summary(results: list[tuple[str, str]], out: Path) -> None:
    print("\n==================== Benchmark run summary ====================")
    for label, status in results:
        print(f"  [{status:>16}]  {label}")
    print(f"\nArtifacts under: {out}")
    if any(s.startswith("FAILED") for _, s in results):
        print("One or more benchmarks failed (continue_on_error kept the run going).")


if __name__ == "__main__":
    main()
