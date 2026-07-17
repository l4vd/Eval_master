# Central benchmark launcher

One entry point to run the five benchmarks (four `*-reproduce` plus `harness-eval`)
against a single model, configured with [Hydra](https://hydra.cc/). The benchmarks
stay separate repos; this launcher just composes one shared `model` config plus a
per-benchmark config group and translates them into each benchmark's own CLI.

```
run_all.sh              # shell wrapper (forwards Hydra overrides)
setup_envs.sh           # one-time: creates a virtualenv per benchmark
run_benchmarks.py       # Hydra launcher: composes config → subprocess per benchmark
conf/
├── config.yaml         # composition root: model + the five benchmark groups + run list
├── model/              # shared model target (id/path/LoRA, dtype, device)
│   ├── qwen_tiny.yaml  #   default: Qwen2.5-0.5B-Instruct (fast, CPU-friendly)
│   └── _template.yaml  #   copy to point at your own checkpoint
├── faitheval/default.yaml
├── truthfulqa/default.yaml
├── halueval/default.yaml
├── ragtruth/default.yaml
└── harness/default.yaml  # TruthfulQA via lm-evaluation-harness (English + multilingual)
```

## Install

The four benchmarks have incompatible dependency stacks, so each gets its own
virtualenv. `./setup_envs.sh` creates them all (plus one for the launcher) with
`uv sync`, honouring each folder's committed `uv.lock` so the stack is identical on
every machine:

```bash
./setup_envs.sh                  # envs at <benchmark>/.venv
./setup_envs.sh --hpc            # pinned cluster stack (pyproject-HPC.toml)
```

`run_all.sh` then finds each interpreter automatically — no flags needed.

> **Windows / OneDrive:** if this repo sits deep in the filesystem, an in-repo
> `.venv` overflows the 260-character `MAX_PATH` limit and imports fail with a
> confusing "No such file or directory" (`.venv/Lib/site-packages/transformers/
> models/...` is already ~260 chars under a synced OneDrive path). Put the envs
> somewhere short instead, and point the launcher at them:
>
> ```bash
> ./setup_envs.sh --venv-root "$LOCALAPPDATA/eval-venvs"
> export VENV_ROOT="$LOCALAPPDATA/eval-venvs"   # or: ./run_all.sh venv_root=...
> ```
>
> This also keeps multi-GB torch installs out of OneDrive sync.

## Usage

```bash
# All four, default tiny model (smoke test):
./run_all.sh model.dtype=float32 model.device_map=cpu num_samples=5

# Your own checkpoint or LoRA adapter across every benchmark:
./run_all.sh model.id=/path/to/final_checkpoint

# A LoRA adapter whose base isn't resolvable from its config:
./run_all.sh model.id=/path/to/lora_adapter \
    model.base_model_id=meta-llama/Meta-Llama-3.1-8B-Instruct

# Run a subset, in order:
./run_all.sh run='[faitheval,ragtruth]'

# Print the commands without running them:
./run_all.sh dry_run=true
```

`./run_all.sh` forwards every argument to Hydra, so any config key is overridable
on the command line. Prefer it, or call `python run_benchmarks.py <overrides>`
directly.

## Configuring subcomponents

Each benchmark is its own Hydra group — override any nested key:

```bash
# FaithEval: one task, strict matching
./run_all.sh faitheval.tasks='[unanswerable]' faitheval.strict_match=true

# TruthfulQA: truthfulness + informativeness judges alongside MC
./run_all.sh truthfulqa.metrics='[mc,judge,info]'

# TruthfulQA: reproduce the original paper's raw Q:/A: prompt instead of the
# model's chat template (see "Prompt format" below)
./run_all.sh truthfulqa.prompt_style=completion

# HaluEval: only the QA judge task
./run_all.sh halueval.tasks='[qa]'

# RAGTruth: your own detector; reproduction (gold-F1) mode; QA only
./run_all.sh ragtruth.detector.id=/path/to/detector \
    ragtruth.gold_f1=true ragtruth.split=test ragtruth.task_types='[QA]'

# harness (lm_eval): add the multilingual TruthfulQA tags; render with the chat
# template instead of the published completion protocol (see "Prompt format")
./run_all.sh harness.tasks='[truthfulqa,truthfulqa_multilingual,truthfulqa-multi]' \
    harness.apply_chat_template=true

# Disable one benchmark without removing it from `run`:
./run_all.sh halueval.enabled=false

# Pass raw flags straight through to a benchmark's own CLI:
./run_all.sh faitheval.extra_args='[--do-sample,--temperature,0.7]'
```

### Model / device

`conf/model/` is the shared model target. Key fields (`conf/model/qwen_tiny.yaml`):

| Field | Maps to | Notes |
| --- | --- | --- |
| `model.id` | `--model-id` / `--model_path` / `--model-path` | Hub id, local path, or LoRA adapter |
| `model.base_model_id` | `--base-model-id` (etc.) | base for a LoRA adapter |
| `model.tokenizer_id` | `--tokenizer-id` (etc.) | if not saved with the weights |
| `model.cache_dir` | `--cache-dir` (etc.) | HF cache |
| `model.dtype` | `--dtype` | `bfloat16` \| `float16` \| `float32` |
| `model.device_map` | `--device-map` | FaithEval / HaluEval / RAGTruth |
| `model.device_index` | `--device` | TruthfulQA / harness (`-1` = CPU, `0` = cuda:0) |

Point at your own checkpoint by editing `conf/model/_template.yaml` (then
`model=_template`) or just overriding `model.id=...` on the CLI.

## Prompt format

Every benchmark builds its prompt with the **evaluated model's own tokenizer**, so
an instruct checkpoint sees the turn markers it was tuned on:

| Benchmark | Prompt | Chat template |
| --- | --- | --- |
| FaithEval | task instruction + context/question as chat messages | yes (falls back to plain concatenation for a base model with no template) |
| HaluEval | system + user judge messages | yes (same fallback) |
| RAGTruth stage 1 | the dataset item's own prompt | yes, explicit |
| RAGTruth stage 2 | fixed `[INST] ... [/INST]` detector template | **no, by design** — the detector was fine-tuned on that exact string |
| TruthfulQA | `truthfulqa.prompt_style` (below) | `chat` by default |
| harness (lm_eval) | `harness.apply_chat_template` | `false` by default — the published completion-style protocol, so its numbers are leaderboard-comparable |

TruthfulQA is the one with a real choice, because the original benchmark predates
chat models and presents the same raw `Q:/A:` few-shot string to everything:

- `prompt_style=chat` (default) renders the preset's few-shot pairs as
  user/assistant turns through the model's chat template. The example *content* is
  parsed from the same preset, so only the framing differs.
- `prompt_style=completion` is the original string — use it to reproduce the
  published protocol.
- `prompt_style=auto` picks `chat` when the tokenizer has a template.

The style materially moves the scores (on a 5-question smoke test with
Qwen2.5-0.5B-Instruct, MC2 was 0.06 under `completion` vs 0.26 under `chat`), so
**report which one you used** and keep it fixed across the models you compare. The
resolved style is written to `run_config.csv` next to each run's answers.

Base and DPO checkpoints share a tokenizer, so either style compares them fairly.

## Global options (`conf/config.yaml`)

| Key | Purpose |
| --- | --- |
| `run` | which benchmarks to run, in order (subset of the five) |
| `num_samples` | global sample cap for FaithEval / HaluEval / RAGTruth, and harness (as lm_eval's `--limit`, applied **per task**); TruthfulQA has none |
| `python` | interpreter for every benchmark; `auto` (default) finds each one's venv; per-benchmark override `<benchmark>.python=...` |
| `venv_root` | where the per-benchmark venvs live; null = `<benchmark>/.venv`. Also read from `$VENV_ROOT` |
| `dry_run` | print commands instead of executing |
| `continue_on_error` | keep going if one benchmark fails (a summary prints regardless) |
| `output_dir` | run directory; each benchmark writes to a subfolder |

## Outputs

Each run gets a timestamped `outputs/<date>/<time>/` directory with per-benchmark
subfolders (`faitheval/`, `truthfulqa/`, `ragtruth/`, `harness/`). HaluEval writes
its results under its own `evaluation/<task>/` folder (upstream behavior). A summary
table of what ran (and pass/fail) prints at the end.

## Separate environments per benchmark

`./setup_envs.sh` (see [Install](#install)) gives each benchmark its own virtualenv
and `python: auto` finds them, so this normally needs no attention. To override:

```bash
# a specific interpreter for one benchmark
./run_all.sh faitheval.python=/envs/faitheval/bin/python

# one shared env for all four (they must all be installed in it)
./run_all.sh python=python
```

Dependency versions are capped at major boundaries and pinned in a per-benchmark
`uv.lock`. Install with `uv sync` (what `setup_envs.sh` does) rather than a bare
`pip install`, or you will silently get a different stack than the one your earlier
runs used — `transformers>=4.44` alone now resolves to 5.x, which this code does
not target.

## HPC

The launcher itself is stack-agnostic — it only shells out and needs Hydra —
but `./setup_envs.sh --hpc` also swaps *its own* `pyproject.toml` for
`pyproject-HPC.toml` (pinned to the cluster's Python 3.12), same as every
benchmark. Install each benchmark from its `pyproject-HPC.toml` (see each
folder's README "Install (HPC)"), pre-download models/detector into the HF
cache, and run `./run_all.sh` inside your SLURM job with
`model.dtype=float32`/`float16` and offline env vars set as usual. Because the
rewritten RAGTruth no longer needs a TGI server, the whole suite runs on a
stock GPU node.
