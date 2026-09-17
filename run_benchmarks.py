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

import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

# Stdlib-only unit registry of the optional all-variants overview.
from analysis import variants as V

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


# --- Hardware-aware config resolvers ----------------------------------------
#
# The launcher's own venv deliberately carries no torch (each benchmark brings
# its own, in its own env — see pyproject.toml), so GPU detection here shells
# out to `nvidia-smi` instead of importing torch. These back the `${auto_*:}`
# interpolations in conf/model/*.yaml and conf/harness/default.yaml, so a plain
# `./run_all.sh` picks sensible settings on both a laptop and a GPU node without
# any override.


def _gpu_available() -> bool:
    """Whether a CUDA GPU is visible to `nvidia-smi`."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _gpu_min_compute_capability() -> float | None:
    """Lowest compute capability across visible GPUs, or None if it can't be read."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        caps = [float(line) for line in proc.stdout.splitlines() if line.strip()]
        return min(caps) if caps else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _auto_device_index() -> int:
    """`${auto_device_index:}` -> 0 (cuda:0) if a GPU is visible, else -1 (CPU)."""
    return 0 if _gpu_available() else -1


def _auto_dtype() -> str:
    """`${auto_dtype:}` -> bf16 needs Ampere+ (compute capability >= 8.0); older
    GPUs fall back to fp16, and CPU falls back to fp32 (bf16 on a generic CPU is
    software-emulated and often slower than fp32, not a real speed win)."""
    if not _gpu_available():
        return "float32"
    cap = _gpu_min_compute_capability()
    return "bfloat16" if cap is not None and cap >= 8.0 else "float16"


def _auto_batch_size() -> str:
    """`${auto_batch_size:}` -> lm_eval's OOM-probing batch-size search
    (`batch_size="auto"`) is written and tested against CUDA OOM errors; it
    doesn't reliably detect CPU memory pressure, so only hand it "auto" when a
    GPU is present. CPU runs get a fixed, modest size instead of the unconditional
    batch_size=1 default."""
    return "auto" if _gpu_available() else "8"


OmegaConf.register_new_resolver("auto_device_index", _auto_device_index)
OmegaConf.register_new_resolver("auto_dtype", _auto_dtype)
OmegaConf.register_new_resolver("auto_batch_size", _auto_batch_size)


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
        f"  Run ./setup_envs_HPC.sh (cluster) or ./setup_envs_local.sh to create the "
        f"per-benchmark environments, or pass an "
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


def build_faitheval(cfg: DictConfig, out: Path, *, output_dir: Path | None = None) -> list[list[str]]:
    b = cfg.faitheval
    samples = _resolve_samples(b, cfg)
    cmds = []
    for task in b.tasks:
        cmd = (
            ["src/run_eval.py", "--task", str(task), "--model-id", str(cfg.model.id)]
            + _model_common_map(cfg)
            + ["--dtype", str(cfg.model.dtype), "--device-map", str(cfg.model.device_map)]
            + ["--split", str(b.split), "--max-new-tokens", str(b.max_new_tokens)]
            + _opt("--batch-size", b.get("batch_size", None))
            + _opt("--sort-by-length", b.get("sort_by_length", None))
            + _opt("--num-samples", samples)
            + (["--strict-match"] if b.strict_match else [])
            + ["--output-dir", str(output_dir or out / "faitheval")]
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


def build_halueval(
    cfg: DictConfig, out: Path, *, output_dir: Path | None = None,
    constrained_output_dir: Path | None = None,
) -> list[list[str]]:
    b = cfg.halueval
    samples = _resolve_samples(b, cfg)
    cmds = []
    for task in b.tasks:
        # cwd is evaluation/ (evaluate.py reads its instruction/data files relative
        # to it); --output-dir sends the results into the unified output tree anyway.
        cmd = (
            ["evaluate.py", "--task", str(task), "--backend", str(b.backend)]
            + ["--model-path", str(cfg.model.id)]
            + _model_common_map(cfg)
            + ["--dtype", str(cfg.model.dtype), "--device-map", str(cfg.model.device_map)]
            # max_new_tokens/seed are _opt so that leaving them null in the config hands
            # the decision to evaluate.py's own defaults (per-prompt-format budget,
            # upstream's unseeded draw) rather than stringifying "None" onto the CLI.
            + _opt("--max-new-tokens", b.get("max_new_tokens", None))
            + _opt("--seed", b.get("seed", None))
            + _opt("--batch-size", b.get("batch_size", None))
            + _opt("--sort-by-length", b.get("sort_by_length", None))
            + _opt("--num-samples", samples)
            # Modified protocols only when switched on, so a default run's command line is
            # exactly what it has always been.
            + (["--scoring", str(b.scoring)] if b.get("scoring", "generate") != "generate" else [])
            + _halueval_exclusion_args(b, str(task))
            + ["--output-dir", str(output_dir or out / "halueval")]
            + _opt("--constrained-output-dir", constrained_output_dir)
            + list(b.extra_args)
        )
        cmds.append(cmd)
    return cmds


def _halueval_exclusion_args(b: DictConfig, task: str) -> list[str]:
    """`--exclude-list` for `task`, only if `halueval.decontam.enabled` AND its list exists.

    A task without a list (today: every task but summarization) has nothing to exclude, so
    it runs with the original numbers only rather than failing.
    """
    decontam = b.get("decontam", None)
    if not decontam or not decontam.get("enabled", False):
        return []
    path = ROOT / str(decontam.get("exclusion_dir", "decontamination/ragtruth")) / f"halueval__{task}.json"
    if not path.is_file():
        print(f"== halueval.decontam: no exclusion list for '{task}' ({path}); original numbers only.")
        return []
    return ["--exclude-list", str(path)]


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
        + _opt("--batch-size", b.get("batch_size", None))
        + _opt("--detector-batch-size", b.detector.get("batch_size", None))
        + _opt("--detector-seed", b.detector.get("seed", None))
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
    if cfg.get("variants", None) is not None:
        # The optional all-variants overview; without `variants` nothing below changes.
        sys.exit(run_suite(cfg, out, overrides=_hydra_overrides()))
    if not cfg.dry_run:
        # TruthfulQA's `--output_path` (a file) needs its parent to exist before
        # the subprocess writes to it; the others create their own --output-dir.
        for sub in ("faitheval", "truthfulqa", "halueval", "ragtruth", "harness"):
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

    # continue_on_error keeps the run going, but the *process* must still report failure:
    # run_all.sh, a SLURM job and any CI wrapper read the exit code, and a run whose
    # benchmarks all died used to exit 0 and read as a complete evaluation.
    if _summary(results, out):
        sys.exit(1)


def _summary(results: list[tuple[str, str]], out: Path) -> bool:
    """Print the run summary; return whether anything failed."""
    print("\n==================== Benchmark run summary ====================")
    for label, status in results:
        print(f"  [{status:>16}]  {label}")
    print(f"\nArtifacts under: {out}")
    failed = [label for label, s in results if s.startswith("FAILED")]
    if failed:
        print(
            f"{len(failed)} of {len(results)} benchmark command(s) FAILED "
            "(continue_on_error kept the run going); exiting non-zero."
        )
    return bool(failed)


# =====================================================================================
# Optional all-variants overview (`variants=[all]`): descriptive, not pre-registered.
# Every variant goes to its own sibling dir <out>/<bench>.<variant>/, units that are done
# are skipped (`resume=true`), CPU variants are derived offline. See analysis/variants.py.
# =====================================================================================


def _hydra_overrides() -> list[str]:
    try:
        from hydra.core.hydra_config import HydraConfig

        return list(HydraConfig.get().overrides.task)
    except Exception:  # not under @hydra.main (tests)
        return []


def legacy_variant_switches(cfg: DictConfig) -> list[str]:
    """Legacy variant switches that would write a variant into an original dir."""
    found = []
    h, f = cfg.halueval, cfg.faitheval
    if str(h.get("scoring", "generate")) != "generate":
        found.append(f"halueval.scoring={h.get('scoring')}")
    decontam = h.get("decontam", None)
    if decontam is not None and decontam.get("enabled", False):
        found.append("halueval.decontam.enabled=true")
    if f.get("strict_match", False):
        found.append("faitheval.strict_match=true")
    if "counterfactual_mc" in [str(t) for t in f.tasks]:
        found.append("faitheval.tasks contains counterfactual_mc")
    return found


def overview_context(cfg: DictConfig) -> V.Context:
    """What the overview's units are checked against: this launch's settings."""
    h, f, hs = cfg.halueval, cfg.faitheval, cfg.harness
    decontam = h.get("decontam", None)
    exclusion_dir = decontam.get("exclusion_dir", "decontamination/ragtruth") if decontam is not None \
        else "decontamination/ragtruth"
    return V.Context(
        tasks={"halueval": tuple(str(t) for t in h.tasks), "faitheval": tuple(str(t) for t in f.tasks)},
        expected={
            "halueval": {"num_samples": _resolve_samples(h, cfg), "seed": h.get("seed", None),
                         "batch_size": h.get("batch_size", None),
                         "sort_by_length": h.get("sort_by_length", None),
                         "max_new_tokens": h.get("max_new_tokens", None)},
            "faitheval": {"num_samples": _resolve_samples(f, cfg), "batch_size": f.get("batch_size", None),
                          "sort_by_length": f.get("sort_by_length", None),
                          "max_new_tokens": f.max_new_tokens, "split": f.split,
                          "dtype": str(cfg.model.dtype)},
            "harness": {"num_samples": _resolve_samples(hs, cfg), "batch_size": hs.batch_size,
                        "dtype": str(cfg.model.dtype), "num_fewshot": hs.num_fewshot},
        },
        halueval_label=V.run_label(str(cfg.model.id)),
        ragtruth_stage=str(cfg.ragtruth.get("stage", "all")),
        exclusion_dir=ROOT / str(exclusion_dir),
        faitheval_split=str(f.split),
        strict_provenance=bool(cfg.get("strict_provenance", False)),
    )


def _one_task(cfg: DictConfig, bench: str, task: str, **fields) -> DictConfig:
    """A config copy running one task of `bench` (with `fields` overridden)."""
    copy = cfg.copy()
    with open_dict(copy):
        copy[bench].tasks = [task]
        for key, value in fields.items():
            copy[bench][key] = value
    return copy


def gpu_commands(cfg: DictConfig, out: Path, units: list) -> list[dict]:
    """One entry per process: the base benchmark, its argv, and the units it produces.

    Built with the legacy builders on a one-task config copy, so every flag matches a
    legacy run; only the output dirs differ. A HaluEval task missing both scorings gets
    `--scoring both` (one model load); one missing scoring runs alone, so a finished
    original is never re-decoded or truncated.
    """
    todo = {u.key: u for u in units}
    commands = []

    def add(bench, cmds, produced):
        for cmd in cmds:
            commands.append({"bench": bench, "cmd": cmd, "units": produced})

    for task in [str(t) for t in cfg.halueval.tasks]:
        gen, con = todo.get(("halueval", task)), todo.get(("halueval.constrained", task))
        if gen and con:
            add("halueval", build_halueval(_one_task(cfg, "halueval", task, scoring="both"), out,
                                           output_dir=out / "halueval",
                                           constrained_output_dir=out / "halueval.constrained"), [gen, con])
        elif gen:
            add("halueval", build_halueval(_one_task(cfg, "halueval", task, scoring="generate"), out,
                                           output_dir=out / "halueval"), [gen])
        elif con:
            add("halueval", build_halueval(_one_task(cfg, "halueval", task, scoring="constrained"), out,
                                           output_dir=out / "halueval.constrained"), [con])
    for (_name, task), unit in todo.items():
        if unit.variant.base == "faitheval":
            add("faitheval", build_faitheval(_one_task(cfg, "faitheval", task), out,
                                             output_dir=out / unit.variant.name), [unit])
    for name in ("truthfulqa", "ragtruth", "harness"):
        unit = todo.get((name, V.WHOLE))
        if unit:
            add(name, BUILDERS[name](cfg, out), [unit])
    order = [str(n) for n in cfg.run]
    commands.sort(key=lambda c: order.index(c["bench"]))
    return commands


def code_version() -> tuple[str | None, str]:
    """EVAL_MASTER_REV (the procedure's EVAL_BLOCK.txt value), else git HEAD, else None."""
    rev = os.environ.get("EVAL_MASTER_REV", "").strip()
    if rev:
        return rev, "EVAL_MASTER_REV"
    try:
        proc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip(), "git"
    except (OSError, subprocess.SubprocessError):
        pass
    print("!! launches.jsonl: no EVAL_MASTER_REV and no git revision; code version recorded as null")
    return None, "none"


def log_launch(out: Path, entry: dict) -> None:
    """Append to <out>/launches.jsonl — the only launch history (.hydra/ is overwritten)."""
    rev, source = code_version()
    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), **entry,
             "code_version": rev, "code_version_source": source, "label": V.OVERVIEW_LABEL}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "launches.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def run_suite(cfg: DictConfig, out: Path, *, overrides: list[str] | None = None,
              runner=subprocess.run) -> int:
    """The overview's unit planner. Returns the process exit code."""
    print(f"==> Overview mode ({V.OVERVIEW_LABEL}); output dir: {out}")
    print(f"==> Model: {cfg.model.id} (dtype={cfg.model.dtype})")
    legacy = legacy_variant_switches(cfg)
    if legacy:
        print(f"!! variants= cannot be combined with the legacy variant switches {legacy}: they "
              "would write a variant into an original dir. Drop them; the overview covers these "
              "variants in their own dirs.")
        return 2
    bases = [str(n) for n in cfg.run if str(n) in BUILDERS and cfg[str(n)].get("enabled", True)]
    requested = [cfg.variants] if isinstance(cfg.variants, str) else [str(v) for v in cfg.variants]
    try:
        chosen, explicit = V.resolve_variants(requested, bases)
    except ValueError as exc:
        print(f"!! {exc}")
        return 2
    if "halueval" in bases and str(cfg.halueval.backend) != "hf":
        print("!! overview mode needs halueval.backend=hf (constrained scoring reads log-probabilities)")
        return 2
    ctx = overview_context(cfg)
    resume, recompute = bool(cfg.get("resume", False)), bool(cfg.get("recompute_stale", False))
    units = V.plan(out, chosen, ctx, explicit=explicit)
    print(V.format_matrix(units, "\n==> Status (variant x task)"))

    stale_gpu = [u for u in units if u.variant.kind == V.GPU and u.status.state == V.STALE]
    if resume and stale_gpu and not recompute:
        print("\n!! GPU units whose recorded settings differ from this launch:")
        for u in stale_gpu:
            print(f"     {u.variant.name}/{u.task}: {u.status.reason}")
        print("   Refusing to mix settings: pass recompute_stale=true to re-run them, or fix the launch.")
        return 2
    for u in units:
        if u.status.state == V.UNVERIFIED:
            print(f"!! {u.variant.name}/{u.task}: {u.status.reason} could not be verified; treated as done")

    gpu_units = [u for u in units if u.variant.kind == V.GPU
                 and V.to_compute(u, resume=resume, recompute_stale=recompute)]
    results: list[tuple[str, str]] = []
    ran: list[dict] = []
    for entry in gpu_commands(cfg, out, gpu_units):
        name = entry["bench"]
        folder = FOLDERS[name]
        interpreter = _resolve_interpreter(cfg[name].get("python", None) or cfg.python, folder,
                                           cfg.get("venv_root", None), dry_run=cfg.dry_run)
        cwd = folder / CWD_SUBDIR[name] if name in CWD_SUBDIR else folder
        full = [str(interpreter)] + entry["cmd"]
        produced = ", ".join(f"{u.variant.name}/{u.task}" for u in entry["units"])
        label = f"{name}: {produced}"
        print(f"\n==> [{name}] {produced} (cwd={cwd})\n    {' '.join(full)}")
        if cfg.dry_run:
            results.append((label, "dry-run"))
            continue
        for u in entry["units"]:
            # A killed run must read as missing, not as the previous run's "done".
            for marker in V.markers(out, u.variant, u.task, ctx):
                marker.unlink(missing_ok=True)
            (out / u.variant.name).mkdir(parents=True, exist_ok=True)
        proc = runner(full, cwd=str(cwd))
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        results.append((label, status))
        ran += [{"unit": f"{u.variant.name}/{u.task}", "was": str(u.status), "result": status}
                for u in entry["units"]]
        if proc.returncode != 0 and not cfg.continue_on_error:
            break

    cpu_units = [u for u in units if u.variant.kind == V.CPU]
    if cpu_units:
        print("\n==> CPU derivations (no model)")
        derived = V.derive_run_dir(out, cpu_units, ctx, resume=resume, dry_run=bool(cfg.dry_run))
        outcomes = (("planned", "dry-run"), ("waiting", "after GPU" if cfg.dry_run else "waiting on source"),
                    ("unavailable", "unavailable"), ("ok", "ok"), ("failed", "FAILED (derive)"))
        for kind, status in outcomes:
            for variant, task in derived[kind]:
                results.append((f"derive: {variant}/{task}", status))
                if kind in ("ok", "failed", "unavailable"):
                    ran.append({"unit": f"{variant}/{task}", "result": status})

    if not cfg.dry_run:
        log_launch(out, {"model": str(cfg.model.id), "variants": requested, "resume": resume,
                         "recompute_stale": recompute, "overrides": overrides or [],
                         "status": {f"{u.variant.name}/{u.task}": str(u.status) for u in units},
                         "ran": ran})
    if not results:
        print("\n==> Nothing to do: every requested unit is done.")
    return 1 if _summary(results, out) else 0


if __name__ == "__main__":
    main()
