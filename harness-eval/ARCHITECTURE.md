# harness-eval — architecture

A thin adapter between the `Eval_master` launcher and `lm_eval`. lm_eval owns the
tasks, prompts, and scoring; this module does exactly three translations and
nothing else.

> **Attribution.** The evaluation itself is *not* ours: it is EleutherAI's
> [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
> running the TruthfulQA benchmark (Lin, Hilton & Evans, 2022) and its multilingual
> extensions — Okapi (Lai et al., 2023) and TruthfulQA-multi (HiTZ; Calvo Figueras
> et al., 2025). Cite those projects, not this wrapper — full references and BibTeX
> are in [README.md](README.md#credits--citations).

## Why it exists: a second, independent number

`TruthfulQA-reproduce` already scores TruthfulQA, using the **original authors'
scripts**. That is not how the field reports TruthfulQA — modern papers run it
through EleutherAI's lm-evaluation-harness, which is what leaderboards publish. So
this module is deliberately **parallel**, not a replacement:

| | protocol | comparable to |
| --- | --- | --- |
| `TruthfulQA-reproduce` | `prompt_style: chat` (default) | the suite's other benchmarks |
| `harness-eval` | lm_eval defaults, `apply_chat_template: false` | published / leaderboard scores |

Both are kept and both reported; their MC1/MC2 will differ, and that difference is
itself informative. It also adds multilingual TruthfulQA (okapi's 31 languages and
HiTZ's 5), which the original-script path cannot reach.

## Module map

```
src/run_eval.py        thin entry point -> harness_eval.cli:main
src/harness_eval/
  cli.py               argparse; parse_args -> build_config -> run_evaluation; --list-tasks
  config.py            frozen EvalConfig + __post_init__ validation
  model.py             build_model_args(): model.id -> lm_eval model_args   [value-add 1]
  tasks.py             resolve_tasks() / list_tasks() against TaskManager    [value-add 2]
  results.py           EvalResults -> flat summary.json + samples.jsonl      [value-add 3]
  evaluator.py         run_evaluation(): the single seam onto simple_evaluate
```

Data flow: `cli` parses flags into a frozen `EvalConfig`; `evaluator` lazily imports
`lm_eval`, calls `model.build_model_args` and `tasks.resolve_tasks`, invokes
`simple_evaluate`, then hands the result to `results.write_outputs`.

`lm_eval` (and hence torch + datasets) is imported **only** inside `evaluator` and
the `--list-tasks` path, so `--help` and argument errors stay instant — the same
lazy-import discipline the sibling benchmarks apply to torch.

## Three design decisions

These are correctness issues, verified against lm_eval source at tags v0.4.5 /
v0.4.12; each has a dedicated regression test.

### 1. `model_args` is a dict, not a string

lm_eval accepts `model_args` as either. The string form (`pretrained=…,peft=…`) is
parsed with a bare `split(",")` and no escaping, so a checkpoint path containing a
comma silently corrupts the args — and it buys nothing, because both the string and
dict paths end in `HFLM(**kwargs)`. We pass the dict; `model_args_to_string` renders
the string for provenance only and **rejects** comma-bearing values rather than emit
one that corrupts on re-parse.

### 2. PEFT adapters keep the checkpoint's own tokenizer

lm_eval's native `pretrained=<base>,peft=<adapter>` loads the tokenizer from
`pretrained` = the **base**. But every sibling benchmark loads it from the
checkpoint dir (`tokenizer_id or model_id`), and a DPO checkpoint from this project
always saves its own tokenizer. Scoring an adapter with the base tokenizer here but
the checkpoint tokenizer everywhere else would break exactly the cross-benchmark
comparison the thesis rests on. So when `model.id` is an adapter that carries
tokenizer files, `build_model_args` passes `tokenizer=<adapter dir>` explicitly,
falling back to the base only when it doesn't.

`model.py` reads `adapter_config.json` with `json.load` rather than
`PeftConfig.from_pretrained`, and imports no torch/peft/transformers — an adapter is
always a local dir, so this is equivalent and keeps the module (and its tests)
heavy-dependency-free.

### 3. Every behaviour-bearing kwarg is passed explicitly

lm_eval's `simple_evaluate(fewshot_as_multiturn=…)` default flipped `False → True`
between 0.4.5 and 0.4.12, and the guard that rejected multiturn without a chat
template was removed. Rather than depend on a shim, `evaluator` passes every such
kwarg explicitly and calls `simple_evaluate` by keyword (it is
`@positional_deprecated`), `config.py` reinstates the multiturn-requires-chat-template
check, and `test_tasks.py` asserts by `inspect.signature` that the installed
`simple_evaluate` accepts our whole kwarg set. `uv.lock` pins the version underneath.

## Tags don't aggregate

All three benches are lm_eval *tags*, so lm_eval returns per-subtask rows and **no**
`groups` aggregate — there is no single cross-language MC2. `results.flatten_results`
handles `groups` if present but never depends on it, and keys rows on the task name
(group-member aliases arrive indented, `" - truthfulqa_mc1"`).

## Python floor: `>=3.10` (a deviation)

The sibling benchmarks pin `requires-python = ">=3.9"`. lm_eval 0.4.10+ requires
`>=3.10`, so a `>=3.9` floor forces the resolver to satisfy every 3.9 interpreter and
it silently backtracks to lm_eval ~0.4.9.x — you would believe you pinned the latest
0.4.x and get a two-year-old resolution. We pin `>=3.10` and `lm_eval[hf]>=0.4.12,<0.5`.
torch/transformers/accelerate/peft are still declared at the repo's shared caps, even
though lm_eval pulls them via its `hf` extra, so the stack is pinned the same way as
every sibling. (Note: lm_eval 0.4.6 never existed.)

## Testing

`tests/` runs offline with no model or dataset download. `test_model_args`,
`test_results`, and `test_cli` are pure-stdlib; only `test_tasks` needs lm_eval
(`pytest.importorskip`) to index the local task YAML via `TaskManager` — including a
canary that the three tag names are still registered in the installed version.
`run_evaluation` takes an `evaluate_fn` seam so the end-to-end path is tested with a
fake, i.e. with lm_eval absent.

## HPC

`pyproject-HPC.toml` pins the cluster stack (torch 2.2.2 / transformers 4.41 /
Python 3.12) and `lm_eval[hf]==0.4.12`. This pairing is metadata-clean but
**unverified at runtime** — 0.4.12's `HFLM` targets much newer transformers. If it
breaks on the cluster, fall back to `lm_eval[hf]==0.4.8` (Python ≥3.9, torch as a
base dep, contemporaneous with transformers 4.41). The three task groups need no
extra metrics deps and pull no TensorFlow (BLEURT is disabled upstream), so the
cluster install stays light. `rouge-score` (for `truthfulqa_gen`) is sdist-only —
make sure the offline mirror serves it.
