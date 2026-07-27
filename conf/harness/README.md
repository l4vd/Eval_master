# harness config (`conf/harness/`)

**What it measures:** TruthfulQA (and multilingual variants) via EleutherAI's
`lm-evaluation-harness`. This is the **leaderboard-comparable** number and is deliberately
kept alongside [`conf/truthfulqa/`](../truthfulqa/README.md) (the original authors'
scripts) — the two MC1/MC2 numbers use different prompt/scoring plumbing and are expected
to differ. One subprocess evaluates the whole task list in a single model load.

**Underlying CLI:** `harness-eval/src/run_eval.py` (→ `harness_eval.cli`), via the
launcher's [`build_harness`](../../run_benchmarks.py). See the parent
[config reference](../README.md) for global keys and the shared `model` block.

## Config keys ([`default.yaml`](default.yaml))

| Key | Type | Default | CLI flag | Meaning |
| --- | --- | --- | --- | --- |
| `enabled` | bool | `true` | — | `false` skips the benchmark. |
| `python` | str \| null | `null` | — | Per-benchmark interpreter; `null` = global. |
| `tasks` | list[str] | `[truthfulqa]` | `--tasks` | lm_eval task/tag names (see below). Globs allowed. |
| `num_samples` | int \| null | `null` | `--limit` | First N docs **per task**; `null` = global. |
| `num_fewshot` | int \| null | `null` | `--num-fewshot` | Few-shot count; `null` = each task's own default (TruthfulQA is 0-shot). |
| `batch_size` | int \| str | `${auto_batch_size:}` | `--batch-size` | `"auto"` on GPU (OOM-probing search), fixed `"8"` on CPU. Set an int or `"auto:N"` to pin. |
| `apply_chat_template` | bool | `false` | `--apply-chat-template` | `false` = published completion protocol (leaderboard-comparable); `true` = model's chat template. **Materially moves scores; keep fixed across compared models.** |
| `fewshot_as_multiturn` | bool | `false` | `--fewshot-as-multiturn` | Few-shot examples as separate chat turns. Requires `apply_chat_template=true`. Passed on every run because lm_eval's own default for it changed between versions. |
| `system_instruction` | str \| null | `null` | `--system-instruction` | Optional system prompt (typically with a chat template). |
| `trust_remote_code` | bool | `false` | `--trust-remote-code` | For checkpoints with custom modeling code. |
| `log_samples` | bool | `true` | `--log-samples` | Write per-document records to `samples.jsonl`. |
| `extra_args` | list[str] | `[]` | (appended raw) | Any flag below not surfaced. |

### Tasks — tags, not groups

Each name is an lm_eval **tag**: it expands to subtasks reported **per-subtask, with no
aggregate score**.

| `tasks` entry | Expands to |
| --- | --- |
| `truthfulqa` | `truthfulqa_mc1`, `truthfulqa_mc2`, `truthfulqa_gen` (English) |
| `truthfulqa_multilingual` | `truthfulqa_<lang>_mc1` / `_mc2` — 31 okapi languages (62 tasks) |
| `truthfulqa-multi` | `truthfulqa-multi_{mc1,mc2,gen}_{en,es,ca,eu,gl}` (HiTZ) |

Glob patterns work (`truthfulqa_de_*`). Names are validated against the installed lm_eval.
**Discover what's available without loading a model:**

```bash
python harness-eval/src/run_eval.py --list-tasks truthfulqa   # optional substring filter
```

> **`num_samples` is `--limit`, applied _per task_.** `num_samples=5` over the 62 okapi
> tasks is 310 evaluations, not 5. Any non-null value makes the run non-comparable to
> published scores; the resolved value is recorded in `summary.json`.

## Model fields consumed

`model.id` → `--model-id`, plus `--base-model-id`, `--tokenizer-id`, `--cache-dir`,
`--dtype`, and `model.device_index` → `--device` (`-1` = CPU → `cpu`, `0` → `cuda:0`;
an explicit string like `cuda:1`/`mps` is passed through). `--output-dir` →
`<output_dir>/harness/`.

## Full underlying CLI (for `extra_args`)

Nearly everything is surfaced. Remaining passthrough / diagnostic flags:

| Flag | Meaning |
| --- | --- |
| `--list-tasks [SUBSTRING]` | Print known task/tag names and exit (no model loaded). Diagnostic — run it directly rather than via `extra_args`. |
| `--log-level LEVEL` | DEBUG/INFO/WARNING/ERROR (default INFO). |

## Output

Writes `<output_dir>/harness/summary.json` (a flat `results` list — each row carries its
own `higher_is_better`) and, with `log_samples=true`, `samples.jsonl`. The
[analysis layer](../../analysis/README.md) reads `truthfulqa_mc1`/`truthfulqa_mc2` (metric
`acc`) as primary.
