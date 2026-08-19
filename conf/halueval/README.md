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
| `max_new_tokens` | int \| null | `null` | `--max-new-tokens` | `null` = the default for the prompt format in use: **16** for the flat/completion format, **128** for the chat format. See [Generation budget](#generation-budget). |
| `seed` | int \| null | `42` | `--seed` | Seed for the coin flip choosing each row's hallucinated-vs-correct output. See [Seeding](#seeding). |
| `batch_size` | int | `8` | `--batch-size` | Judge prompts per forward pass (`backend: hf` only). The dominant runtime lever here — see below. Lower it if `summarization` OOMs. |
| `extra_args` | list[str] | `[]` | (appended raw) | Any flag below not surfaced. |

### Tasks

| Task | The judge decides whether… |
| --- | --- |
| `qa` | an answer to a question contains a hallucination. |
| `dialogue` | a dialogue response contains non-factual/hallucinated content. |
| `summarization` | a summary contains information unsupported by the source document. |

## Generation budget

An unparseable judgement is scored as **incorrect**, not excluded — so a judge cut off
mid-verdict is counted as *wrong*, and the token budget silently becomes part of the
metric. The default therefore matches whichever OpenAI endpoint the prompt format stands
in for in the original script:

| Prompt format | Stands in for | Budget | Why |
| --- | --- | --- | --- |
| `chat_template` | `openai.ChatCompletion.create(...)` | 128 | The original passes only `temperature=0.0` — no `max_tokens`, so the published protocol never truncated a verbose judge. 128 rather than uncapped because 3 × 10,000 rows makes unbounded generation impractical; a judge silent for 128 tokens is refusing, not truncated. |
| `concat` | `openai.Completion.create(...)` | 16 | 16 *is* that endpoint's own default `max_tokens`, so this path was already faithful. |

Measured on this project's checkpoints, going 16 → 64 on the chat path **halved** the
unparseable rate over the same rows. Raise it further for reasoning models that think
before answering. The resolved value and the format are recorded in `summary.json`.

## Seeding

Upstream never seeds the per-row draw deciding whether the judge is shown the
hallucinated or the correct output, so every run scores a *different* random 50/50
partition and two models are compared against two different label draws. `seed: 42`
pins it: the metric's distribution is unchanged (the published number is itself one draw
from it), but runs become reproducible and compared arms become **paired**.

Keep the seed identical across every arm you compare. For error bars, sweep it
(`seed=42,43,44,…`) and report mean ± std across seeds — `halueval.seed` is a plain
Hydra key, so `--multirun halueval.seed=42,43,44` works. It is recorded in
`summary.json`. Set it to `null` to reproduce upstream's unseeded behaviour.

## Batch size

Each task is a **10,000-row** dataset and only 16 tokens are generated per row, so at
`batch_size: 1` this benchmark is almost entirely per-call overhead rather than real
decoding — batching is where the wall-clock goes. `summarization` carries by far the
longest prompts (full source documents) and is the split that will OOM first.

Batching changes throughput only — prompt construction, decoding and Yes/No parsing are
untouched, and the per-row `random()` draw that assigns the hallucinated/correct answer
still happens once per row in index order, so a given seed yields the same ground-truth
assignment as the unbatched loop.

Verified on the HPC pins (torch 2.2.2 / transformers 4.41.2): `batch_size` 1, 4 and 8
produced byte-identical judgements to each other and to the pre-batching single-call code,
in both prompt formats, and the three dataset loops produce identical per-sample result
files and accuracies at `batch_size` 1 vs 8. Padding could in principle flip a greedy
argmax tie on some model, so keep `batch_size` **fixed across every model you compare**; it
is recorded in each `<task>_<label>_summary.json`.

Full-size runs are the published protocol; use `num_samples` for iteration and debugging,
not for the numbers you report against published baselines.

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

### Reading the number

`accuracy` alone cannot distinguish the three ways a judge fails, so the summary carries
diagnostics next to it. They are derived from the same judgements and change no score:

| Field | Reading it |
| --- | --- |
| `num_failed`, `format_compliance` | Share of rows where no Yes/No verdict could be read. These count as **wrong** in `accuracy` (upstream's rule), so low compliance depresses the headline number without the judge ever having been wrong. |
| `tpr` / `tnr` | Recall on hallucinated / on correct outputs. A judge answering one constant label pins one of these to `0.0` while `accuracy` sits at the label base rate (~0.5) and looks merely mediocre. |
| `judged_yes_rate` | Share of parsed rows judged "Yes". `0.0` or `1.0` means degenerate. For scale, the bundled ChatGPT reference run sits at 0.247 — even a strong judge leans heavily "No". |

Each row also stores the judge's `raw_judgement`, so a finished run can be re-scored
offline — no model, no GPU:

```
python evaluation/score_results.py <output_dir>/halueval/*_results.json
```

That reports the strict (published, comparable) scoring alongside a lenient
case-insensitive one. The **gap between them is the share of your score that is output
formatting rather than hallucination detection**; when they agree, the strict parser is
costing you nothing. Only the strict number belongs in a results table uncaveated.
