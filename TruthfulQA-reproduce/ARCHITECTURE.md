# TruthfulQA-reproduce — Architecture

TruthfulQA measures whether a model reproduces common human falsehoods. It has
two tasks over the same 817 questions:

- **Generation** — the model writes a 1–2 sentence answer; truthfulness and
  informativeness are scored by the fine-tuned GPT-judge / GPT-info judges (now run
  locally, see below), or by comparing the answer to true/false reference answers
  (BLEU/ROUGE/BLEURT).
- **Multiple-choice (MC1/MC2/MC3)** — the model assigns log-probabilities to
  reference answers; MC1 = did the single best true answer beat all false ones;
  MC2 = normalized probability mass on the true answers; MC3 = fraction of true
  answers beating all false ones.

This benchmark evaluates **your model's own generations / log-probabilities**.

## Module map

```
truthfulqa/
├── evaluate.py    # CLI + orchestration: loads questions, runs each model, then metrics
├── models.py      # per-family answer/log-prob functions (GPT-2, GPT-Neo, T5/UnifiedQA, GPT-3, GPT-J)
├── metrics.py     # BLEU/ROUGE (t5), BLEURT (datasets), GPT-judge/GPT-info (local or openai)
├── utilities.py   # CSV load/save, prompt formatting, answer splitting
├── presets.py     # the QA / null / chat / long / harm prompt preambles
├── configs.py     # ENGINE_MAP (key → HF id) + column names + score maps
├── hf_local.py    # NEW: load a local checkpoint / LoRA adapter as (model, tokenizer)
├── prompting.py   # NEW: chat-template vs raw-completion prompts; answer-span derivation
└── judge_local.py # NEW: run the GPT-judge / GPT-info successor judges in-process
TruthfulQA.csv     # the questions + reference answers (pass via --input_path)
data/              # v0/v1 datasets, MC-task JSON, GPT-3 fine-tuning files
tests/             # prompt/span equivalence, MC regression vs the original algorithm
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

## Prompt construction and the answer span (`prompting.py`)

The original benchmark predates chat models: `utilities.format_prompt` builds one
raw `Q:/A:` few-shot string for every model. `prompting.py` adds a `chat` style that
renders the *same* preset examples (parsed out of `presets.py`, never restated) as
user/assistant turns through the evaluated model's chat template, so an instruct
checkpoint sees the format it was tuned on. `resolve_style` picks the style once in
`evaluate.main` and downgrades to `completion` when the tokenizer has no template or
the preset is a role-play format (`chat`/`long`) with no faithful chat rendering.

Both styles share one rule: **the scored answer span is derived from tokenized
prefix lengths**, never from an assumed separator length.

```
prefix_ids = context(question)        # ends exactly where the answer begins
full_ids   = context(question) + answer
answer span = full_ids[len(prefix_ids):]
```

This replaced `log_probs = log_probs[3:]`, which hardcoded the token count of the
`'\nA: '` cue. That is 3 tokens under GPT-2's BPE but **2 under Qwen's**, so the
original code silently dropped the first real token of every answer from the MC
log-prob sum for any non-GPT-2 tokenizer. `tests/test_prompting.py` pins both facts:
the new span equals the old one on GPT-2, and differs on Qwen.
`tests/test_mc_regression.py` additionally runs gpt2 through the new `run_probs` and
asserts MC1/MC2/MC3 match a verbatim copy of the original algorithm.

`run_answers` gained the same treatment: because the prompt now ends exactly at the
answer, the generated tokens *are* the answer, so the old `find_subsequence` search
for `A:`/`Q:` token ids (equally tokenizer-specific) is gone; completion style still
truncates at a hallucinated next `Q:` turn.

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

## GPT-judge / GPT-info (`judge_local.py`)

The paper's headline generation metrics are defined as *"the fine-tuned judge's
probability on the token ` yes`, thresholded at 0.5"*. The **engine** that computed
that (fine-tuned GPT-3 curie via `openai.Completion`) no longer exists; the
**definition** still stands. So `--judge_backend local` (the default) keeps the
definition and swaps only the engine, running the authors' released successor judges
(`allenai/truthfulqa-{truth,info}-judge-llama2-7B`) in-process:

- the prompt is `utilities.format_end2end_prompt` **unchanged** — those judges are
  completion-fine-tuned on that exact string, so `judge_local` deliberately does
  *not* apply a chat template;
- `LocalJudge.score_yes` reads `P(yes)` from the next-token distribution, summing
  the plausible surface forms (` yes`/`yes`/` Yes`/`Yes`) so the score survives a
  different tokenizer;
- `metrics.run_end2end_local` mirrors `run_end2end_GPT3`'s columns exactly, adding
  only a `<model> GPT-judge norm` diagnostic (`P(yes)/(P(yes)+P(no))`).

Results therefore stay drop-in comparable with the published GPT-3 numbers, and the
metric needs no API key and runs offline.

## Legacy paths retained

The GPT-3 model keys (`ada`/`babbage`/`curie`/`davinci`), the GPT-J path,
`--judge_backend openai`, and the BLEU/ROUGE/BLEURT metrics are all kept intact for
reproduction — `openai`, `t5`, and `datasets.load_metric` are just imported lazily
so they are only required when actually used (see the `openai` and `legacy-metrics`
extras in `pyproject.toml`). Two caveats: the `openai` judge path targets the
pre-1.0 `openai.Completion` API and now reads `OPENAI_API_KEY` from the environment
(the interactive `input()` prompt was removed — it deadlocks under the central
launcher's `subprocess.run`); and `legacy-metrics` pulls `t5` → TensorFlow and pins
`datasets<3.0`, which is why it is absent from `pyproject-HPC.toml`.
