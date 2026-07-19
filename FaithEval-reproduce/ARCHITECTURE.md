# FaithEval-reproduce — Architecture

This folder packages the FaithEval contextual-faithfulness benchmark as a small,
installable evaluation CLI. It is the **reference layout** the sibling
`*-reproduce` benchmarks (TruthfulQA, HaluEval, RAGTruth) are being brought up to.

The benchmark evaluates a model's **own generations**: for each example the model
is shown a context + question and must answer faithfully (say "unknown" when the
context does not answer, flag conflicts, or follow a counterfactual context). The
answer string is then scored against the task's rule.

## Module map

```
src/
├── run_eval.py                # thin `python src/run_eval.py ...` entry point → cli.main
└── faitheval/
    ├── cli.py                 # argparse front-end; builds an EvalConfig, calls the evaluator
    ├── config.py              # TaskConfig (per-task YAML schema) + EvalConfig (per-run settings)
    ├── data.py                # load_task_dataset() → local JSONL split, optional truncation
    ├── prompting.py           # build_messages(): context+question → chat messages
    ├── model.py               # model loading (Hub id / local path / LoRA adapter) + HFChatGenerator
    ├── metrics.py             # normalize_answer, phrase_match, answer_match
    └── evaluator.py           # run_evaluation(): the loop tying data → generate → score → I/O
scripts/
└── prepare_datasets.py        # run online once: download Hub splits → data/faitheval/*.jsonl
data/
└── faitheval/                 # pre-materialised JSONL splits (populated by the script above)
configs/
├── unanswerable.yaml          # dataset name, task prompt, valid phrases, scoring rule
├── inconsistent.yaml
└── counterfactual.yaml
tests/
├── test_metrics.py            # unit tests for the scoring rules
└── test_data.py               # unit tests for the local JSONL loader
```

## Dataset loading (`data.py`)

`load_task_dataset` reads each split from a **local JSON Lines file** under
`data/faitheval/<slug>/<split>.jsonl` (slug = the dataset id's last path
component), builds a `datasets.Dataset` from the raw records, and optionally
truncates it. Nothing touches the Hub or the HF arrow cache at eval time.

This is a deliberate choice: the cluster pins `datasets<3`, which cannot read an
arrow cache written by `datasets>=4` (list columns use the newer `List` feature
type → `TypeError: must be called with a dataclass type or instance`). Plain
JSONL has no such version-stamped metadata. The files are produced by
`scripts/prepare_datasets.py` on an internet-connected machine (any `datasets`
version) and copied over; `FAITHEVAL_DATA_DIR` overrides the location. The
original Hub-loading path is preserved, commented, at the bottom of `data.py`
for online, unpinned use.

## Data flow

```
cli.parse_args ─▶ cli.build_config ─▶ EvalConfig
                        │  (reads configs/<task>.yaml via config.load_task_config)
                        ▼
                 evaluator.run_evaluation
                        │
      ┌─────────────────┼──────────────────────────┐
      ▼                 ▼                            ▼
 data.load_task    model.HFChatGenerator      metrics.phrase_match /
   _dataset          (load once)                 answer_match
      │                 │                            ▲
      └── example ─▶ prompting.build_messages ─▶ generator.generate ─┘
                                                     │
                                    stream record → outputs/<task>_predictions.jsonl
                                                     │
                                    outputs/<task>_summary.json  (accuracy)
```

Each example is generated one at a time and its prediction is streamed to
`<output_dir>/<task>_predictions.jsonl`; once the loop finishes, an accuracy
`<task>_summary.json` is written.

## The `configs/*.yaml` task mechanism

A task is fully described by data, not code. `config.load_task_config` reads a
YAML file into a frozen `TaskConfig`:

- `dataset_name` — the Hugging Face dataset id for the task split.
- `scoring` — either `phrase_match` (the prediction must *contain* one of a set of
  accepted phrases, e.g. "unknown"/"conflict") or `answer_match` (the normalized
  prediction must *equal* a reference answer, used for counterfactual).
- `task_specific_prompt` — a task instruction appended to the base instruction.
- `valid_phrases` / `strict_valid_phrases` — the accepted-phrase lists for
  `phrase_match`; `--strict-match` selects the strict list at run time.
- `*_column` — which dataset columns hold the context / question / answer.

Adding a new task or prompt variant is therefore a new YAML file plus (if the
name isn't already there) an entry in `SUPPORTED_TASKS` — no change to the
generation or scoring code.

## Running with your own model — the loading path in `model.py`

`model.py` is the single "run it with your own checkpoint" hook, reused verbatim
across the sibling benchmarks. `HFChatGenerator.__init__` calls `_load_causal_lm`,
which handles three cases behind one `--model-id` flag:

1. **Hub id** (`org/repo`) — loaded straight from the Hub / cache.
2. **Local full-model path** — `_looks_like_local_path` recognizes it (absolute,
   `./`, `~`, or a backslash Windows path); `_check_local_path_exists` fails fast
   with a `FileNotFoundError` if the directory is missing, instead of letting
   `from_pretrained` emit a confusing Hub-404.
3. **Local LoRA / PEFT adapter** — `_is_peft_adapter` detects an
   `adapter_config.json` in the directory. The base model is read from the
   adapter's own config (or overridden with `--base-model-id`), the adapter is
   loaded onto it with PEFT, and `merge_and_unload()` folds it into a plain
   causal LM for generation.

The tokenizer defaults to `--model-id`; `--tokenizer-id` is the escape hatch for
weights saved without their tokenizer. `--dtype`, `--device-map`, and `--cache-dir`
are passed straight through to `from_pretrained`.

This is the exact contract every benchmark in `Eval_master/` implements, so a
checkpoint from the sibling `SP-DPO-Base` training pipeline
(`outputs/<run>/final_checkpoint`, full **or** LoRA) drops into any of them.

## Prompt format

`prompting.build_messages` returns chat messages, and `HFChatGenerator.generate`
hands that list to the text-generation pipeline, which applies **the evaluated
model's own chat template** — so the wire format matches what the model was tuned
on, and differs per model by design.

A base (non-instruct) model has no chat template, which would make the pipeline
raise. `HFChatGenerator` detects that at load time, logs a warning, and falls back
to a plain `\n\n` concatenation of the message contents so such models remain
evaluable. `generator.prompt_format` reports which path is in use
(`chat_template` / `concat`).

## Install

See [`README.md`](./README.md) for the local install; on the cluster use
[`pyproject-HPC.toml`](./pyproject-HPC.toml) (torch 2.2.2 / transformers 4.41 /
Python 3.12 pins, HF backend only — no vLLM).
