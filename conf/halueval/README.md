# HaluEval config (`conf/halueval/`)

**What it measures:** hallucination *recognition* — the evaluated model acts as a Yes/No
judge deciding whether a given answer/response/summary contains a hallucination. Accuracy
against gold labels. One subprocess per task in `tasks`.

**Underlying CLI:** `HaluEval-reproduce/evaluation/evaluate.py`, via the launcher's
[`build_halueval`](../../run_benchmarks.py). It runs from the `evaluation/` subdir (the
script reads its instruction/data files relative to it); results still go to the unified
output tree. See the parent [config reference](../README.md) for global keys and the
shared `model` block.

## Config keys ([`default.yaml`](default.yaml))

| Key | Type | Default | CLI flag | Meaning |
| --- | --- | --- | --- | --- |
| `enabled` | bool | `true` | — | `false` skips the benchmark. |
| `python` | str \| null | `null` | — | Per-benchmark interpreter; `null` = global. |
| `tasks` | list[str] | `[qa, dialogue, summarization]` | `--task` (one run each) | Which HaluEval judge tasks to run. |
| `backend` | str | `hf` | `--backend` | `hf` = local judge (the model under test decides Yes/No); `openai` = reproduction path (needs `OPENAI_API_KEY`). |
| `num_samples` | int \| null | `null` | `--num-samples` | Evaluate only the first N examples; `null` = global. |
| `max_new_tokens` | int | `16` | `--max-new-tokens` | Short — a Yes/No answer. |
| `extra_args` | list[str] | `[]` | (appended raw) | Any flag below not surfaced. |

### Tasks

| Task | The judge decides whether… |
| --- | --- |
| `qa` | an answer to a question contains a hallucination. |
| `dialogue` | a dialogue response contains non-factual/hallucinated content. |
| `summarization` | a summary contains information unsupported by the source document. |

## Model fields consumed

The shared `model.id` is passed as `--model-path` (the HF judge), plus `--base-model-id`,
`--tokenizer-id`, `--cache-dir`, `--dtype`, `--device-map`. `backend` defaults to `hf`
whenever a model path is set. `--output-dir` → `<output_dir>/halueval/`.

## Full underlying CLI (for `extra_args`)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model NAME` | `davinci` | For the OpenAI backend, the engine name; with the HF backend it's just a **run label** used in output filenames. |

Everything else the script accepts (`--model-path`, `--base-model-id`, `--tokenizer-id`,
`--cache-dir`, `--dtype`, `--device-map`, `--max-new-tokens`, `--num-samples`,
`--output-dir`) is set for you from the `model` block / config keys — don't duplicate them.

## Output

Writes per-sample results and a `<task>_<label>_summary.json` (flat: `task`, `accuracy`,
`num_examples`, …) under `<output_dir>/halueval/`. When `--output-dir` is unset (not the
launcher path) it falls back to the in-repo `<task>/` folder (legacy behaviour). The
[analysis layer](../../analysis/README.md) reads `accuracy` (higher is better) as primary
and synthesizes a `mean` task across the three.
