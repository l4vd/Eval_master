# RAGTruth-reproduce — Architecture

RAGTruth is a word-level RAG hallucination corpus. A **detector model** reads a
`(reference, response)` pair and outputs the hallucinated spans as
`{"hallucination list": [...]}`. This fork adds a **serverless, two-stage,
in-process** pipeline so you can evaluate **your own generation model** with that
detector — replacing the upstream TGI Docker server.

## Two model roles

| Role | What it is | Where it plugs in |
| --- | --- | --- |
| **Generation model** | your checkpoint / LoRA (e.g. from `SP-DPO-Base`) | Stage 1 — produces a RAG response per source item (`--model-id`) |
| **Detector model** | `CodingLL/RAGTruth_Eval` or your own trained detector | Stage 2 — flags hallucinated spans (`--detector-model-id`) |

Both are loaded through the **same** `model.py` loader, so each independently
accepts a Hub id, a local path, or a LoRA/PEFT adapter.

## Package layout

```
src/
├── run_eval.py                 # `python src/run_eval.py --stage all ...` → cli.main
└── ragtruth_eval/
    ├── cli.py                  # argparse front-end; orchestrates the stages; prints the summary
    ├── model.py                # loader (Hub/path/LoRA) + HFGenerator.chat / .complete + GenerationParams
    ├── data.py                 # load_source_items (Stage 1) / load_gold_responses (gold-F1)
    ├── prompts.py              # generation messages, detector TEMPLATES + [INST] wrap, JSON parse
    ├── generate.py             # Stage 1: run_generation → generations.jsonl
    ├── detect.py               # Stage 2: run_detection → detections.jsonl + summary.json
    └── metrics.py              # hallucination_rate (primary) + gold_f1 (reproduction)
dataset/
├── source_info.jsonl          # source items ({source_id, task_type, source_info, prompt})
└── response.jsonl             # original responses + gold labels (+ split membership)
baseline/                      # RETAINED upstream detector-training + TGI-serving path
├── prepare_dataset.py, train.py, dataset.py
└── predict_and_evaluate.py    # the legacy TGI client (see README "why 127.0.0.1:8300")
tests/
└── test_metrics.py            # unit tests for the rate/F1 aggregation and JSON parsing
```

## Two-stage data flow

```
                    dataset/source_info.jsonl
                              │
         data.load_source_items(split, num_samples, task_types)
                              │  (derives reference/question per task_type)
                              ▼
  ┌─────────────────────── Stage 1: generate.py ───────────────────────┐
  │  HFGenerator(--model-id).chat( build_generation_messages(item) )    │
  │      → response                                                     │
  │  stream {source_id, task_type, prompt, reference, question, response}│
  └──────────────────────────────┬─────────────────────────────────────┘
                                  ▼
                 outputs/<run>/generations.jsonl
                                  │
  ┌─────────────────────── Stage 2: detect.py ─────────────────────────┐
  │  HFGenerator(--detector-model-id).complete(                        │
  │      build_detector_prompt(item)   # [INST] TEMPLATES[task] [/INST] │
  │  ) → raw text                                                       │
  │  parse_hallucination_list(raw) → (spans, ok)                        │
  │  record {..., hallucination_list, pred_halu, parse_failed}          │
  └──────────────────────────────┬─────────────────────────────────────┘
                                  ▼
       outputs/<run>/detections.jsonl   +   outputs/<run>/summary.json
                                  │
                     metrics.hallucination_rate(detections)
                       (overall + per QA/Summary/Data2txt)
```

`chat` vs. `complete` is deliberate: the generation model is an instruct model, so
Stage 1 uses its **chat template** (`HFGenerator.chat`, falling back to an
`[INST]` wrap if the tokenizer has none); the detector was trained on raw
`[INST] ... [/INST]` strings, so Stage 2 uses **raw completion**
(`HFGenerator.complete`) exactly as the TGI baseline did.

## Metrics

- **`hallucination_rate`** (primary) — fraction of generations flagged (non-empty
  span list), overall and per task type. Characterizes the generation model.
- **`gold_f1`** (optional, `--gold-f1`) — precision/recall/F1 of the detector's
  example-level "is this hallucinated" prediction against the corpus gold labels,
  reproducing `baseline/predict_and_evaluate.py`'s case-level scores. In this mode
  Stage 1 is skipped and Stage 2 reads the *original* responses via
  `data.load_gold_responses`.

Both are computed from the in-memory `detections` list and are unit-tested in
`tests/test_metrics.py` (no model download needed).

## Relation to the retained `baseline/` trainer

`baseline/` is kept for users who want to **train their own detector**: prepare
the split files, fine-tune a Llama-2 model on the `[INST]`-wrapped
`{"hallucination list": ...}` target (`baseline/dataset.py` — the source of the
ported `TEMPLATES` and prompt format in `prompts.py`), then either serve it via
TGI with the legacy `predict_and_evaluate.py`, **or** simply point the new
pipeline's `--detector-model-id` at the trained checkpoint. The new package reuses
the baseline's exact detector prompt, so a detector trained the upstream way drops
straight into Stage 2 with no server.

## The model loader (shared with the other benchmarks)

`model.py` ports `FaithEval-reproduce/src/faitheval/model.py`
(`_looks_like_local_path` / `_check_local_path_exists` / `_is_peft_adapter` /
`_load_causal_lm`). It is used for both model roles, which is why
`--model-id /path/to/final_checkpoint` (full or LoRA) "just works" for the
generation model *and* the detector, identically to FaithEval, TruthfulQA, and
HaluEval.
