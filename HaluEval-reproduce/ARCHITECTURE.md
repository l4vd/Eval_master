# HaluEval-reproduce — Architecture

HaluEval measures whether a model can **recognize** hallucinations. For each
example the model is shown a `(question / context, answer)` pair — where the
answer is randomly either the ground-truth or a pre-generated hallucinated one —
and must reply `Yes` (hallucinated) or `No` (faithful). The score is the accuracy
of those Yes/No judgements.

So "your model" here is the **judge**, not the answer generator. This is
different from FaithEval/TruthfulQA/RAGTruth, which score your model's own
generations.

## Repository layout

```
generation/     # data-generation pipeline (build hallucinated samples via an LLM) — reproduction only
├── generate.py, filtering.py
└── <task>/*_instruction.txt, *_data.json     # per-task generation prompts + seeds
data/           # the released benchmark: qa_data.json, dialogue_data.json,
                #   summarization_data.json, general_data.json
evaluation/     # the judge — this is where "your model" plugs in
├── evaluate.py                 # CLI + the three get_*_response judges + dataset loops
├── hf_local.py                 # NEW: local HF judge loader + HFChatGenerator
└── <task>/<task>_evaluation_instruction.txt   # per-task judge instruction (read at run time)
analysis/       # LDA topic analysis of recognized/failed samples (optional)
└── analyze.py
```

## Data flow (evaluation)

```
evaluate.py __main__
    │  reads evaluation/<task>/<task>_evaluation_instruction.txt
    │  picks backend = --backend or ('hf' if --model-path else 'openai')
    │  if hf: build HFChatGenerator(model_path, ...) once
    ▼
evaluation_<task>_dataset(model, data/<task>_data.json, instruction, out, backend, generator, num_samples)
    │  for each example (first --num-samples):
    │     randomly pick ground-truth vs hallucinated answer  → ground_truth ∈ {No, Yes}
    │     build the judge messages (system + instruction + fields)
    │     ans = get_<task>_response(..., backend, generator)
    │        ├─ backend == 'hf'     → generator.generate(messages)  (in-process HF)
    │        └─ backend == 'openai' → openai.ChatCompletion/Completion (reproduction)
    │     normalize ans to 'Yes' / 'No' (malformed → 'failed!', counted incorrect)
    │     append {fields, ground_truth, judgement} to <task>/<task>_<label>_results.json
    ▼
prints "<correct> correct, <incorrect> incorrect, Accuracy: <correct/n>"
```

The three tasks differ only in field names and the system message:
`get_qa_response` (question/answer), `get_dialogue_response`
(dialogue_history/response), `get_summarization_response` (document/summary).

## generation/ vs evaluation/ vs analysis/

- **generation/** — how the benchmark itself was *built* (prompt an LLM to
  produce and then filter hallucinated samples from seed datasets). Not needed to
  evaluate a model; kept for reproduction.
- **evaluation/** — the actual benchmark: run a judge over `data/*.json` and score
  its Yes/No accuracy. This is where your model is used.
- **analysis/** — optional LDA topic modelling over the recognized/failed samples
  to see *what* a judge tends to miss. Behind the `analysis` extra
  (nltk/spacy/gensim/pyLDAvis).

## The task instruction files

Each task's judge instruction is a plain `.txt` under
`evaluation/<task>/<task>_evaluation_instruction.txt`, read at run time and
prepended to the per-example fields. Editing the instruction changes the judge
prompt without touching code — the same "data, not code" idea as FaithEval's
`configs/`.

## Where the judge model plugs in

The three `get_*_response` functions take `backend` and `generator` arguments:

- `backend='openai'` (default when no `--model-path`) keeps the original
  `openai.ChatCompletion` / `openai.Completion` calls with the retry loop.
  `openai` and `tiktoken` are imported lazily and the key is read from
  `OPENAI_API_KEY`, so this path is fully optional.
- `backend='hf'` (default when `--model-path` is set) calls
  `generator.generate(messages)`, where `generator` is an
  `HFChatGenerator` built once in `__main__`.

`evaluation/hf_local.py` is a port of
`FaithEval-reproduce/src/faitheval/model.py`: the same
`_looks_like_local_path` / `_check_local_path_exists` / `_is_peft_adapter` /
`_load_causal_lm` helpers, plus an `HFChatGenerator.generate(messages) -> str`
that runs a chat text-generation pipeline greedily and returns the assistant
turn. `--model-path` therefore accepts a Hub id, a local full-model path, or a
LoRA/PEFT adapter (auto-detected and merged onto its base), identically to the
other `*-reproduce` benchmarks.
