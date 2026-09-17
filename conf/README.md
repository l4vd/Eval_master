# Configuration reference (`conf/`)

Full parameter reference for the Hydra config that drives
[`run_benchmarks.py`](../run_benchmarks.py) (launched via
[`run_all.sh`](../run_all.sh)). For the *workflow* — install, prompt-format
choices, HPC — see [`README-runner.md`](../README-runner.md). This file documents
**every key**: what it does, its type, its default, and what CLI flag it maps to in
each benchmark.

## How the config is composed

[`config.yaml`](config.yaml) is the composition root. Its `defaults:` list pulls one
file from each config group into a single merged config tree:

```yaml
defaults:
  - model: qwen_tiny     # -> conf/model/qwen_tiny.yaml   (merged at @_global_ -> cfg.model.*)
  - faitheval: default   # -> conf/faitheval/default.yaml (merged under cfg.faitheval.*)
  - truthfulqa: default
  - halueval: default
  - ragtruth: default
  - harness: default
  - _self_               # config.yaml's own top-level keys win last
```

```
conf/
├── config.yaml            # global keys + the defaults list (this composition root)
├── model/                 # -> model/README.md   (shared model target; @_global_ -> cfg.model.*)
│   ├── qwen_tiny.yaml
│   └── _template.yaml
├── faitheval/             # -> faitheval/README.md   (cfg.faitheval.*)
├── truthfulqa/            # -> truthfulqa/README.md
├── halueval/              # -> halueval/README.md
├── ragtruth/              # -> ragtruth/README.md
└── harness/               # -> harness/README.md
```

**Per-benchmark deep reference.** The tables in this file cover the config keys and their
flag mappings. Each group folder also has its own README with the **complete** underlying
CLI (every flag you can pass via `extra_args`), the task list, and the output shape:
[model](model/README.md) · [faitheval](faitheval/README.md) ·
[truthfulqa](truthfulqa/README.md) · [halueval](halueval/README.md) ·
[ragtruth](ragtruth/README.md) · [harness](harness/README.md).

**Overriding on the CLI.** Every key is a dotted path. `./run_all.sh` forwards all
arguments straight to Hydra:

```bash
./run_all.sh model.id=/path/to/ckpt faitheval.tasks='[unanswerable]' num_samples=5
```

Lists use Hydra/YAML list syntax and usually need quoting so the shell doesn't split
them: `run='[faitheval,ragtruth]'`. To swap a whole group file, use `group=file`
(no dot), e.g. `model=_template`.

**Precedence:** group defaults < `config.yaml` top-level keys (`_self_` is last in
`defaults`) < CLI overrides.

---

## Global keys ([`config.yaml`](config.yaml))

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `run` | list[str] | `[faitheval, truthfulqa, halueval, ragtruth, harness]` | Which benchmarks to run **and in what order**. Subset it to run only some: `run='[faitheval]'`. Unknown names are skipped with a warning. |
| `num_samples` | int \| null | `null` | Global sample cap for FaithEval / HaluEval / RAGTruth, and harness (as lm_eval `--limit`). `null` = full dataset. Each benchmark can override with its own `<bench>.num_samples`. **TruthfulQA ignores this** (it scores the whole CSV — pre-truncate the CSV for a subset). For `harness` the cap is applied **per task**. |
| `python` | str | `auto` | Interpreter used to launch each benchmark. `auto` resolves each benchmark's own virtualenv (they have incompatible stacks). Set `python=python` to force one shared interpreter, or override per-benchmark with `<bench>.python=...`. |
| `venv_root` | str \| null | `${oc.decode:${oc.env:VENV_ROOT,null}}` (i.e. `$VENV_ROOT` or `null`) | Where the per-benchmark venvs live. `null` = each benchmark's own `<folder>/.venv`. Relocate to `<venv_root>/<folder>` on Windows/OneDrive to dodge the 260-char `MAX_PATH` limit. Also read from the `VENV_ROOT` env var. |
| `dry_run` | bool | `false` | Print each subprocess command instead of running it. Also downgrades a missing-venv error to a warning so you can inspect commands before installing. |
| `continue_on_error` | bool | `true` | Keep going if one benchmark exits non-zero. A pass/fail summary prints either way; `false` aborts on the first failure. |
| `output_dir` | str | `${hydra:runtime.output_dir}` | Root for all artifacts (per-benchmark subfolders under it). Defaults to the timestamped Hydra run dir so runs isolate automatically. |
| `variants` | list[str] \| null | `null` | **Optional overview, descriptive only.** `null` = today's behaviour, unchanged. `[all]` or a list of variant names ([`analysis/variants.py`](../analysis/variants.py)) switches to the unit planner. Every variant goes to its own `<output_dir>/<bench>.<variant>/`, CPU variants are derived offline, and a status matrix prints first. Refused together with `halueval.scoring`, `halueval.decontam.enabled`, `faitheval.strict_match` or `counterfactual_mc` in `faitheval.tasks`. See [RUNNING_NEW_OPTIONS.md §7](../RUNNING_NEW_OPTIONS.md#7-optional-all-variants-in-one-run-overview). |
| `resume` | bool | `false` | Overview mode only: skip units that are done (marker present, recorded settings match). |
| `recompute_stale` | bool | `false` | Overview mode only: re-run GPU units whose recorded settings differ. Otherwise they are listed and the launch exits 2. |
| `strict_provenance` | bool | `false` | Overview mode only: treat `unverified` units as stale. A unit is `unverified` when a setting could not be checked, e.g. an old FaithEval summary where the dataset is absent. |

### Hydra block (bottom of `config.yaml`)

| Key | Default | Meaning |
| --- | --- | --- |
| `hydra.job.chdir` | `false` | Don't `cd` into the run dir; the launcher resolves benchmark folders relative to itself. |
| `hydra.run.dir` | `outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}` | Single-run output dir. |
| `hydra.sweep.dir` / `.subdir` | `outputs/multirun/<date>/<time>` / `${hydra.job.num}` | `--multirun` output layout: one numbered subdir per sweep job. The analysis layer's `--run-evals` relies on this numbered layout. |

---

## Shared model ([`conf/model/`](model/))

Selected with `model=<name>` (default `qwen_tiny`). Both files carry
`# @package _global_`, so their `model:` block lands at `cfg.model.*` and feeds **all
five** benchmarks. The launcher translates these fields into each benchmark's own flag
spelling (some use `--model-id`, TruthfulQA uses `--model_path`, etc.).

| Key | Type | Default (`qwen_tiny`) | Maps to (per benchmark) | Meaning |
| --- | --- | --- | --- | --- |
| `model.id` | str | `Qwen/Qwen2.5-0.5B-Instruct` | `--model-id` / `--model_path` / `--model-path` | The evaluated model: a Hub id, a local full-model dir, or a PEFT/LoRA adapter checkpoint (auto-detected and merged onto its base). |
| `model.base_model_id` | str \| null | `null` | `--base-model-id` / `--base_model_id` | Base to merge a LoRA adapter onto, when it isn't recorded in the adapter config. |
| `model.tokenizer_id` | str \| null | `null` | `--tokenizer-id` / `--tokenizer_id` | Tokenizer, if not saved alongside the weights. |
| `model.cache_dir` | str \| null | `null` | `--cache-dir` / `--cache_dir` | Hugging Face cache directory. |
| `model.dtype` | str | `${auto_dtype:}` | `--dtype` | `bfloat16` \| `float16` \| `float32`. The resolver picks bf16 on Ampere+ GPUs (compute capability ≥ 8.0), fp16 on older GPUs, fp32 on CPU. Pin explicitly to override. |
| `model.device_map` | str | `auto` | `--device-map` | HF `device_map`, used by **FaithEval / HaluEval / RAGTruth**. |
| `model.device_index` | int | `${auto_device_index:}` | `--device` | CUDA index for **TruthfulQA / harness**. `-1` = CPU, `0` = cuda:0. Resolver: `0` if a GPU is visible else `-1`. |

**`_template.yaml`** is an unselected copy pointing at `/path/to/final_checkpoint`.
Point your own checkpoint at it and select with `model=_template`, or just override
`model.id=...` on the CLI without touching any file.

### The `${auto_*:}` resolvers

`model.dtype`, `model.device_index`, and `harness.batch_size` default to OmegaConf
resolvers registered in [`run_benchmarks.py`](../run_benchmarks.py). The launcher's own
venv carries no torch, so hardware detection **shells out to `nvidia-smi`** (not a torch
import) at startup:

| Resolver | GPU present | No GPU |
| --- | --- | --- |
| `${auto_device_index:}` | `0` (cuda:0) | `-1` (CPU) |
| `${auto_dtype:}` | `bfloat16` if compute cap ≥ 8.0, else `float16` | `float32` |
| `${auto_batch_size:}` | `"auto"` (lm_eval OOM-probing search) | `"8"` (fixed; OOM probing is CUDA-only) |

A plain `./run_all.sh` therefore picks safe settings on a laptop and a GPU node alike.
Any explicit value bypasses the resolver.

---

## Per-benchmark keys

Every benchmark group shares four common keys:

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `enabled` | bool | `true` | `false` skips the benchmark even if it's listed in `run` (reported as `disabled` in the summary). |
| `python` | str \| null | `null` | Per-benchmark interpreter override. `null` = use the global `python`. |
| `num_samples` | int \| null | `null` | Per-benchmark sample cap. `null` = use the global `num_samples`. (Not present on TruthfulQA, which has no cap.) |
| `extra_args` | list[str] | `[]` | Raw flags appended verbatim to that benchmark's own CLI, e.g. `faitheval.extra_args='[--do-sample,--temperature,0.7]'`. Your escape hatch for any flag not surfaced as a config key. |

### FaithEval ([`faitheval/default.yaml`](faitheval/default.yaml)) · [full reference →](faitheval/README.md)

Contextual faithfulness; scores the model's own generations. **One subprocess per task
in `tasks`.**

| Key | Type | Default | `src/run_eval.py` flag | Meaning |
| --- | --- | --- | --- | --- |
| `tasks` | list[str] | `[unanswerable, inconsistent, counterfactual]` | `--task` (once each) | The FaithEval task splits to run. |
| `split` | str | `test` | `--split` | Dataset split. |
| `strict_match` | bool | `false` | `--strict-match` | Use each task's strict valid-phrase list for answer matching. |
| `max_new_tokens` | int | `256` | `--max-new-tokens` | Generation length cap. |

### TruthfulQA ([`truthfulqa/default.yaml`](truthfulqa/default.yaml)) · [full reference →](truthfulqa/README.md)

Truthfulness via MC scoring and/or local judges. **No `num_samples`** — it scores the
whole `input_path` CSV.

| Key | Type | Default | `truthfulqa.evaluate` flag | Meaning |
| --- | --- | --- | --- | --- |
| `metrics` | list[str] | `[mc]` | `--metrics` | Any of `mc` (MC1/MC2/MC3), `judge`, `info`, `bleu`, `rouge`, `bleurt`. `mc`/`judge`/`info` need only torch+transformers; `bleu`/`rouge`/`bleurt` need the `legacy-metrics` extra (TensorFlow) and don't run on the cluster. |
| `preset` | str | `qa` | `--preset` | Few-shot preset: `qa` \| `null` \| `chat` \| `long` \| `help` \| `harm`. |
| `input_path` | str | `TruthfulQA.csv` | `--input_path` | Questions CSV, relative to the `TruthfulQA-reproduce` folder. Truncate this file to subset the run. |
| `prompt_style` | str | `chat` | `--prompt_style` | How the question is presented: `chat` (few-shot pairs through the model's chat template — matches the other benchmarks), `completion` (the original raw `Q:/A:` string — reproduces the published protocol), or `auto` (chat iff the tokenizer has a template). **Materially moves scores — keep it fixed across compared models.** See [`README-runner.md`](../README-runner.md#prompt-format). |
| `judge_backend` | str | `local` | `--judge_backend` | For `judge`/`info`: `local` runs the authors' successor judge models in-process (no API key, offline); `openai` uses the legacy API path (needs `OPENAI_API_KEY` and a fine-tuned engine id). |
| `truth_judge_id` | str | `allenai/truthfulqa-truth-judge-llama2-7B` | `--truth_judge_id` | Truth judge weights (local backend) or engine id (openai). |
| `info_judge_id` | str | `allenai/truthfulqa-info-judge-llama2-7B` | `--info_judge_id` | Informativeness judge weights / engine id. |

### HaluEval ([`halueval/default.yaml`](halueval/default.yaml)) · [full reference →](halueval/README.md)

Hallucination recognition; the model acts as a Yes/No judge. **One subprocess per task.**
Runs from the `HaluEval-reproduce/evaluation/` subdir (its scripts read their instruction
files relative to it); results still land in the unified output tree.

| Key | Type | Default | `evaluate.py` flag | Meaning |
| --- | --- | --- | --- | --- |
| `tasks` | list[str] | `[qa, dialogue, summarization]` | `--task` (once each) | HaluEval judge tasks to run. |
| `backend` | str | `hf` | `--backend` | `hf` = local judge (the model under test); `openai` = reproduction path (needs `OPENAI_API_KEY`). |
| `max_new_tokens` | int | `16` | `--max-new-tokens` | Short — a Yes/No answer. |

### RAGTruth ([`ragtruth/default.yaml`](ragtruth/default.yaml)) · [full reference →](ragtruth/README.md)

Serverless two-stage RAG hallucination eval: the shared `model` generates responses
(stage 1), a **detector** model flags them (stage 2). No TGI server.

| Key | Type | Default | `src/run_eval.py` flag | Meaning |
| --- | --- | --- | --- | --- |
| `stage` | str | `all` | `--stage` | `generate` (stage 1 only), `detect` (stage 2 only), or `all`. |
| `split` | str \| null | `null` | `--split` | `null`/all, or `train` \| `dev` \| `test`. |
| `task_types` | list[str] \| null | `null` | `--task-types` | `null` = all, or a subset of `[QA, Summary, Data2txt]`. |
| `gold_f1` | bool | `false` | `--gold-f1` | Reproduction mode: score the detector against gold labels (precision/recall/F1), skipping generation. |
| `detector.id` | str | `llama2_13b_lora_0217` | `--detector-model-id` | The stage-2 hallucination detector. Defaults to the RAGTruth authors' weights; override with your own trained detector. |
| `detector.base_model_id` | str \| null | `null` | `--detector-base-model-id` | Base for a LoRA detector adapter. |
| `detector.tokenizer_id` | str \| null | `null` | `--detector-tokenizer-id` | Detector tokenizer, if separate. |

### harness ([`harness/default.yaml`](harness/default.yaml)) · [full reference →](harness/README.md)

TruthfulQA via EleutherAI's `lm-evaluation-harness`. A **second, independent** number
next to `TruthfulQA-reproduce` — this is what public leaderboards report, so the two
MC1/MC2 numbers are expected to differ. One subprocess evaluates the whole task list in
a single model load.

| Key | Type | Default | `src/run_eval.py` flag | Meaning |
| --- | --- | --- | --- | --- |
| `tasks` | list[str] | `[truthfulqa]` | `--tasks` | lm_eval task names/**tags**. `truthfulqa` → mc1/mc2/gen (English); `truthfulqa_multilingual` → 31 okapi languages (62 tasks); `truthfulqa-multi` → HiTZ en/es/ca/eu/gl. Tags report per-subtask with no aggregate. Globs work (`truthfulqa_de_*`). Validated against the installed lm_eval; `src/run_eval.py --list-tasks truthfulqa` prints what's available. |
| `num_fewshot` | int \| null | `null` | `--num-fewshot` | `null` = each task's own default (TruthfulQA is 0-shot). |
| `batch_size` | int \| str | `${auto_batch_size:}` | `--batch-size` | `"auto"` on GPU (OOM-probing search), fixed `"8"` on CPU. Set an int or `"auto:N"` to pin. |
| `apply_chat_template` | bool | `false` | `--apply-chat-template` | `false` (default) = the published completion-style protocol (leaderboard-comparable, and why this module exists); `true` = render with the model's chat template. **Materially moves scores — keep fixed across compared models.** |
| `fewshot_as_multiturn` | bool | `false` | `--fewshot-as-multiturn` | Present few-shot examples as separate chat turns. Only meaningful with `apply_chat_template=true`. Passed on every run because lm_eval's own default for it changed between versions. |
| `system_instruction` | str \| null | `null` | `--system-instruction` | Optional system prompt (typically with `apply_chat_template`). |
| `trust_remote_code` | bool | `false` | `--trust-remote-code` | For checkpoints with custom modeling code. |
| `log_samples` | bool | `true` | `--log-samples` | Write per-document records to `samples.jsonl`. |

> **`num_samples` for harness is lm_eval's `--limit`, applied _per task_.**
> `num_samples=5` over the 62 okapi tasks is 310 evaluations, not 5. Any non-null value
> makes the run non-comparable to published scores; the resolved value is recorded in
> `summary.json`.

---

## Outputs

Each run writes a timestamped `outputs/<date>/<time>/` (or, for `--multirun`,
`outputs/multirun/<date>/<time>/<job>/`) with a subfolder per benchmark: `faitheval/`,
`truthfulqa/`, `ragtruth/`, `harness/`. HaluEval also writes under its own
`evaluation/<task>/` folder (upstream behavior). A pass/fail summary table prints at the
end. These run dirs are exactly what the [`analysis/`](../analysis/README.md) layer
discovers, aggregates, and plots.

## Common recipes

```bash
# CPU smoke test, tiny model, 5 samples:
./run_all.sh model.dtype=float32 model.device_index=-1 model.device_map=cpu num_samples=5

# Your checkpoint / LoRA across every benchmark:
./run_all.sh model.id=/path/to/final_checkpoint
./run_all.sh model.id=/path/to/lora model.base_model_id=meta-llama/Meta-Llama-3.1-8B-Instruct

# A subset, in order; print without running:
./run_all.sh run='[faitheval,ragtruth]' dry_run=true

# Reproduce TruthfulQA's published protocol; add multilingual harness tasks:
./run_all.sh truthfulqa.prompt_style=completion
./run_all.sh harness.tasks='[truthfulqa,truthfulqa_multilingual]'

# Disable one benchmark without editing `run`:
./run_all.sh halueval.enabled=false
```
