# FaithEval config (`conf/faitheval/`)

**What it measures:** contextual faithfulness — given a context + question, does the
model answer *from the context* (and refuse when the context doesn't support an answer)
instead of fabricating. Scores the model's **own generations**. One subprocess per task
in `tasks`.

**Underlying CLI:** `FaithEval-reproduce/src/run_eval.py` (→ `faitheval.cli`).
The launcher's [`build_faitheval`](../../run_benchmarks.py) translates the keys below into
its flags. See the parent [config reference](../README.md) for global keys and the shared
`model` block.

## Config keys ([`default.yaml`](default.yaml))

| Key | Type | Default | CLI flag | Meaning |
| --- | --- | --- | --- | --- |
| `enabled` | bool | `true` | — | `false` skips the benchmark even if listed in `run`. |
| `python` | str \| null | `null` | — | Per-benchmark interpreter; `null` = global `python`. |
| `tasks` | list[str] | `[unanswerable, inconsistent, counterfactual]` | `--task` (one run each) | Which FaithEval task splits to run. |
| `split` | str | `test` | `--split` | Dataset split. |
| `strict_match` | bool | `false` | `--strict-match` | Match against each task's **strict** valid-phrase list rather than the lenient one (see task config below). |
| `num_samples` | int \| null | `null` | `--num-samples` | Evaluate only the first N examples; `null` = global `num_samples` = full split. |
| `max_new_tokens` | int | `256` | `--max-new-tokens` | Generation length cap per example. |
| `extra_args` | list[str] | `[]` | (appended raw) | Any flag below not surfaced as a key. |

### Tasks

| Task | What a *faithful* model does |
| --- | --- |
| `unanswerable` | The context lacks the answer → the model should refuse ("unknown"), not guess. |
| `inconsistent` | The context contains a contradiction → the model should flag the conflict. |
| `counterfactual` | The context asserts something counter to world knowledge → the model should follow the *context*, not its prior. |

## Full underlying CLI (for `extra_args`)

Flags **not** surfaced as config keys — pass them through `faitheval.extra_args`. Model
flags (`--model-id`, `--base-model-id`, `--tokenizer-id`, `--cache-dir`, `--dtype`,
`--device-map`) and `--output-dir` are set automatically from the shared `model` block /
`output_dir` — **don't** set those via `extra_args`.

| Flag | Type / default | Meaning |
| --- | --- | --- |
| `--config PATH` | Path / `configs/<task>.yaml` | Override the task config YAML (see below). |
| `--do-sample` | flag / off | Sample instead of greedy decoding. |
| `--temperature FLOAT` | float / none | Sampling temperature (requires `--do-sample`). |
| `--top-p FLOAT` | float / none | Nucleus sampling top-p (requires `--do-sample`). |
| `--system-prompt STR` | str / none | Optional system prompt prepended to each example. |
| `--log-level LEVEL` | DEBUG/INFO/WARNING/ERROR / INFO | Logging verbosity. |

```bash
# stochastic decoding on one task
./run_all.sh faitheval.tasks='[unanswerable]' \
    faitheval.extra_args='[--do-sample,--temperature,0.7,--top-p,0.9]'
```

## Task config YAML (`FaithEval-reproduce/configs/<task>.yaml`)

Each task loads a config file (override with `--config`). Fields, e.g.
[`unanswerable.yaml`](../../FaithEval-reproduce/configs/unanswerable.yaml):

| Field | Meaning |
| --- | --- |
| `dataset_name` | HF dataset id (e.g. `Salesforce/FaithEval-unanswerable-v1.0`). |
| `scoring` | Scoring method (`phrase_match`). |
| `task_specific_prompt` | Instruction appended to each example. |
| `valid_phrases` | Lenient accept list — any of these in the answer counts as correct. |
| `strict_valid_phrases` | Used instead when `strict_match=true` (a narrower list). |
| `context_column` / `question_column` | Dataset column names for the context and question. |

## Output

Writes `<output_dir>/faitheval/<task>_summary.json` (flat: `task`, `accuracy`,
`num_examples`, …) plus per-prediction files. The [analysis layer](../../analysis/README.md)
reads `accuracy` (higher is better) as the primary metric, and synthesizes a `mean` task
when more than one task ran.
