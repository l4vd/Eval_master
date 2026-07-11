# TruthfulQA-reproduce — Architecture

TruthfulQA measures whether a model reproduces common human falsehoods. It has
two tasks over the same 817 questions:

- **Generation** — the model writes a 1–2 sentence answer; truthfulness is scored
  by comparing that answer to true/false reference answers (BLEU/ROUGE/BLEURT, or
  the fine-tuned GPT-judge/GPT-info metrics).
- **Multiple-choice (MC1/MC2)** — the model assigns log-probabilities to reference
  answers; MC1 = did the single best true answer beat all false ones; MC2 =
  normalized probability mass on the true answers.

This benchmark evaluates **your model's own generations / log-probabilities**.

## Module map

```
truthfulqa/
├── evaluate.py    # CLI + orchestration: loads questions, runs each model, then metrics
├── models.py      # per-family answer/log-prob functions (GPT-2, GPT-Neo, T5/UnifiedQA, GPT-3, GPT-J)
├── metrics.py     # BLEU/ROUGE (t5), BLEURT (datasets), GPT-judge/GPT-info (openai)
├── utilities.py   # CSV load/save, prompt formatting, answer splitting
├── presets.py     # the QA / null / chat / long / harm prompt preambles
├── configs.py     # ENGINE_MAP (key → HF id) + column names + score maps
└── hf_local.py    # NEW: load a local checkpoint / LoRA adapter as (model, tokenizer)
TruthfulQA.csv     # the questions + reference answers (pass via --input_path)
data/              # v0/v1 datasets, MC-task JSON, GPT-3 fine-tuning files
```

## Data flow

```
utilities.load_questions(--input_path)  ─▶  pandas DataFrame (one row per question)
                    │
   for each model key in --models:
                    │
        ┌───────────┴───────────────────────────────┐
        ▼                                            ▼
  models.run_answers  (generation)          models.run_probs  (MC, if 'mc' in --metrics)
   writes answer into frame[key]             writes MC1/MC2/MC3 + lprob cols into frame
        │                                            │
        └───────────────► answers.csv ◄──────────────┘   (streamed via save_questions)
                    │
   for each metric in --metrics:
        metrics.run_bleu_and_rouge / run_BLEURT / run_end2end_GPT3
                    │
        format_frame ─▶ mean over rows ─▶ summary.csv  (MC1, MC2, bleu acc, ... )
```

Each `run_*` function is keyed by a column `tag`; the generation answer and all
metric columns for a model live under that tag in the single results DataFrame,
which is checkpointed to `--output_path` after each step.

## MC1/MC2 vs. generation metrics

- **Generation metrics** (`bleu`, `rouge`, `bleurt`, `judge`, `info`) score the
  *text* the model produced (`run_answers`). `bleu`/`rouge` come from `t5`,
  `bleurt` from `datasets.load_metric`, and `judge`/`info` from fine-tuned GPT-3.
  These are the heavy/optional deps and are imported lazily.
- **Multiple-choice** (`mc`) never generates free text: `run_probs` scores the
  reference answers by summed token log-probability under the model and computes
  MC1/MC2/MC3 in `MC_calcs`. This path needs only `torch`/`transformers` — no
  `t5`, `openai`, or `datasets.load_metric`.

## Where the local loader plugs in

`models.run_answers` and `models.run_probs` already accept optional `model=` and
`tokenizer=` arguments; when both are supplied they skip the built-in
`AutoModelForCausalLM.from_pretrained(engine)` load. The `--model_path` support
uses exactly that seam:

1. `evaluate.main` sees `--model_path`, calls `hf_local.load_local_model(...)`
   once, and forces `args.models = ['local']`.
2. `hf_local.load_local_model` returns a `(model, tokenizer)` pair for a Hub id,
   a local full-model directory, or a LoRA/PEFT adapter (auto-detected via
   `adapter_config.json` and merged onto its base with `merge_and_unload`). It
   places the model on the chosen device, sets it to eval mode, and enables
   `return_dict_in_generate` so `run_answers` (which reads `outputs.sequences` /
   `outputs.scores`) works with a pre-loaded model.
3. The new `local` branch in the model loop passes that pair into `run_answers`
   (generation) and, when `mc` is requested, `run_probs` (log-probs).

`hf_local.py` is a port of `FaithEval-reproduce/src/faitheval/model.py` — the same
`_looks_like_local_path` / `_check_local_path_exists` / `_is_peft_adapter` /
`_load_causal_lm` helpers — so a checkpoint drops into every `*-reproduce`
benchmark the same way.

## Legacy paths retained

The GPT-3 model keys (`ada`/`babbage`/`curie`/`davinci`), the GPT-J path, and the
GPT-judge/GPT-info + BLEU/ROUGE/BLEURT metrics are all kept intact for
reproduction — `openai`, `t5`, and `datasets.load_metric` are just imported
lazily so they are only required when actually used (see the `openai` and
`legacy-metrics` extras in `pyproject.toml`).
