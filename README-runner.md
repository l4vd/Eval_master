# Central benchmark launcher

One entry point to run the four `*-reproduce` benchmarks against a single model,
configured with [Hydra](https://hydra.cc/). The benchmarks stay separate repos;
this launcher just composes one shared `model` config plus a per-benchmark config
group and translates them into each benchmark's own CLI.

```
run_all.sh              # shell wrapper (forwards Hydra overrides)
run_benchmarks.py       # Hydra launcher: composes config → subprocess per benchmark
conf/
├── config.yaml         # composition root: model + the four benchmark groups + run list
├── model/              # shared model target (id/path/LoRA, dtype, device)
│   ├── qwen_tiny.yaml  #   default: Qwen2.5-0.5B-Instruct (fast, CPU-friendly)
│   └── _template.yaml  #   copy to point at your own checkpoint
├── faitheval/default.yaml
├── truthfulqa/default.yaml
├── halueval/default.yaml
└── ragtruth/default.yaml
```

## Install

The launcher itself only needs Hydra (each benchmark's own deps come from its
`pyproject.toml`):

```bash
pip install hydra-core omegaconf     # already present in the SP-DPO-Base env
```

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

# TruthfulQA: add the (heavier) generation metrics as well as MC
./run_all.sh truthfulqa.metrics='[mc,bleu]'

# HaluEval: only the QA judge task
./run_all.sh halueval.tasks='[qa]'

# RAGTruth: your own detector; reproduction (gold-F1) mode; QA only
./run_all.sh ragtruth.detector.id=/path/to/detector \
    ragtruth.gold_f1=true ragtruth.split=test ragtruth.task_types='[QA]'

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
| `model.device_index` | `--device` | TruthfulQA (`-1` = CPU) |

Point at your own checkpoint by editing `conf/model/_template.yaml` (then
`model=_template`) or just overriding `model.id=...` on the CLI.

## Global options (`conf/config.yaml`)

| Key | Purpose |
| --- | --- |
| `run` | which benchmarks to run, in order (subset of the four) |
| `num_samples` | global sample cap for FaithEval / HaluEval / RAGTruth (TruthfulQA has none) |
| `python` | interpreter for every benchmark; per-benchmark override `<benchmark>.python=...` |
| `dry_run` | print commands instead of executing |
| `continue_on_error` | keep going if one benchmark fails (a summary prints regardless) |
| `output_dir` | run directory; each benchmark writes to a subfolder |

## Outputs

Each run gets a timestamped `outputs/<date>/<time>/` directory with per-benchmark
subfolders (`faitheval/`, `truthfulqa/`, `ragtruth/`). HaluEval writes its results
under its own `evaluation/<task>/` folder (upstream behavior). A summary table of
what ran (and pass/fail) prints at the end.

## Separate environments per benchmark

The four benchmarks have different dependency stacks. If you install each in its
own virtualenv, point the launcher at each interpreter:

```bash
./run_all.sh \
    faitheval.python=/envs/faitheval/bin/python \
    ragtruth.python=/envs/ragtruth/bin/python
```

Otherwise a single env with all four installed (`pip install -e .` in each
folder) works, and `python=...` sets one interpreter for all.

## HPC

The launcher is stack-agnostic — it only shells out. Install each benchmark from
its `pyproject-HPC.toml` (see each folder's README "Install (HPC)"), pre-download
models/detector into the HF cache, and run `./run_all.sh` inside your SLURM job
with `model.dtype=float32`/`float16` and offline env vars set as usual. Because
the rewritten RAGTruth no longer needs a TGI server, the whole suite runs on a
stock GPU node.
