# TruthfulQA config (`conf/truthfulqa/`)

**What it measures:** truthfulness — whether the model avoids imitative falsehoods on
questions humans often answer wrongly. Scores the model's **own generations / log-probs**
(MC) and, optionally, local judge models. Runs the original authors' scripts (a *second*,
independent TruthfulQA number lives in [`conf/harness/`](../harness/README.md)).

**Underlying CLI:** `python -m truthfulqa.evaluate` (in `TruthfulQA-reproduce/`), via the
launcher's [`build_truthfulqa`](../../run_benchmarks.py). See the parent
[config reference](../README.md) for global keys and the shared `model` block.

> **No `num_samples`.** TruthfulQA scores the whole `input_path` CSV — the global sample
> cap does **not** apply. To subset, pre-truncate the CSV or point `input_path` at a
> smaller file.

## Config keys ([`default.yaml`](default.yaml))

| Key | Type | Default | CLI flag | Meaning |
| --- | --- | --- | --- | --- |
| `enabled` | bool | `true` | — | `false` skips the benchmark. |
| `python` | str \| null | `null` | — | Per-benchmark interpreter; `null` = global. |
| `metrics` | list[str] | `[mc]` | `--metrics` | Any of `mc`, `judge`, `info`, `bleu`, `rouge`, `bleurt` (see below). |
| `preset` | str | `qa` | `--preset` | Few-shot preset: `qa` \| `null` \| `chat` \| `long` \| `help` \| `harm`. |
| `input_path` | str | `TruthfulQA.csv` | `--input_path` | Questions CSV, relative to `TruthfulQA-reproduce/`. |
| `prompt_style` | str | `chat` | `--prompt_style` | `chat` (few-shot pairs via chat template — matches the other benchmarks), `completion` (original raw `Q:/A:` string — reproduces the published protocol), or `auto`. **Materially moves scores; keep fixed across compared models.** |
| `judge_backend` | str | `local` | `--judge_backend` | For `judge`/`info`: `local` (authors' successor judges, in-process, no API key) or `openai` (legacy API path, needs `OPENAI_API_KEY`). |
| `truth_judge_id` | str | `allenai/truthfulqa-truth-judge-llama2-7B` | `--truth_judge_id` | Truth judge weights (local) or engine name (openai). |
| `info_judge_id` | str | `allenai/truthfulqa-info-judge-llama2-7B` | `--info_judge_id` | Informativeness judge weights / engine name. |
| `extra_args` | list[str] | `[]` | (appended raw) | Any flag below not surfaced. |

### Metrics

| Metric | Produces | Needs |
| --- | --- | --- |
| `mc` | MC1 / MC2 / MC3 (multiple-choice log-prob scoring) | torch + transformers only |
| `judge` | GPT-judge truthfulness score | a truth judge (`local` default works offline) |
| `info` | GPT-info informativeness score | an info judge (`local` default) |
| `bleu` / `rouge` / `bleurt` | reference-overlap scores | the benchmark's `legacy-metrics` extra (t5 → TensorFlow); **not supported on the cluster** |

`judge` + `info` are usually reported together (`metrics: [mc, judge, info]`) — the true
"% truthful *and* informative" headline needs both.

## Model fields consumed

| `model.*` | TruthfulQA flag (note the underscore spelling) |
| --- | --- |
| `id` | `--model_path` (a synthetic `local` model key is evaluated) |
| `base_model_id` | `--base_model_id` |
| `tokenizer_id` | `--tokenizer_id` |
| `cache_dir` | `--cache_dir` |
| `dtype` | `--dtype` |
| `device_index` | `--device` (`-1` = CPU, `0` = cuda:0) |

`--output_path` is set to `<output_dir>/truthfulqa/answers.csv` automatically.

## Full underlying CLI (for `extra_args`)

Everything meaningful is already surfaced as a key or driven by the `model` block. The
remaining passthrough flags:

| Flag | Meaning |
| --- | --- |
| `--gptj_path PATH` | Local GPT-J weights (only for the legacy `gptj` model preset — rarely needed). |
| `--models NAME...` | The built-in reference model keys; **ignored** when `model.id` is set (a `local` key is evaluated instead). |

## Output

Writes `<output_dir>/truthfulqa/answers.csv` (per-question answers + scores) and a
`summary.csv` pivot (rows = model, cols = metrics). The resolved `prompt_style` is
recorded alongside. The [analysis layer](../../analysis/README.md) reads `MC1`/`MC2` as
primary metrics (all TruthfulQA metrics are higher-is-better, task `overall`).
