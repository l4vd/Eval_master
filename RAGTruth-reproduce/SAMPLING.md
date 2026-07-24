# Sampling considerations: Stage 1 (generation) vs. Stage 2 (detection)

The pipeline calls an LLM twice, for two different purposes, and the right
decoding strategy is not the same at both call sites. This note lays out the
trade-offs at each stage, what this repo currently does, and how that relates
to the upstream RAGTruth baseline (`baseline/predict_and_evaluate.py`).

## Why the two stages are not the same decision

| | Stage 1 — generation (`generate.py`) | Stage 2 — detection (`detect.py`) |
| --- | --- | --- |
| Model | your generation checkpoint (`--model-id`) | the hallucination detector (`--detector-model-id`) |
| Question it answers | "What would this model say?" | "Does this specific response contain a hallucination?" |
| Output | free-form RAG response | a fixed-schema JSON object, `{"hallucination list": [...]}` |
| What sampling changes | *which* response gets evaluated | *whether the same, already-fixed response* gets correctly labeled |

Stage 1's output is the thing under test — hallucination rate is a property
*of the generation model*, so how you decode there changes what you are
measuring. Stage 2's output is a judgment about a response that already
exists — sampling there doesn't change the object being evaluated, only the
judge's reliability.

## Stage 1: generation

Current default in this repo: **greedy** (`GenerationParams(do_sample=False)`
in `model.py`; `cli.py`'s `--do-sample`/`--temperature`/`--top-p`/`--top-k`
flags let you opt into sampling per run).

Arguments for greedy (the current default):
- **Reproducibility.** Re-running Stage 1 on the same checkpoint gives the
  same `generations.jsonl`, so a hallucination-rate change between two runs
  is attributable to the model, not to decoding luck.
- **Clean comparison across checkpoints.** This repo's actual use case is
  comparing a base model against a DPO/SFT-tuned checkpoint (see
  `SP-DPO-Base`). Greedy removes sampling variance as a confound, so a
  measured rate difference reflects the training intervention.
- **No retry logic downstream.** Nothing in Stage 2 depends on Stage 1 being
  stochastic (unlike the baseline's detector call — see below), so there's no
  hidden reason to sample here.

Arguments for sampling (available via `--do-sample`):
- Greedy decoding is not how the model is actually deployed; if you care
  about hallucination rate *under realistic decoding* (temperature/top-p as
  used at inference time), you need to sample with those settings.
- A single greedy transcript is one point estimate. Sampling multiple
  generations per prompt (not currently wired up — would need running Stage 1
  N times) would let you report a rate with a confidence interval instead of
  a single number.

**Note on the original RAGTruth corpus:** the corpus's own responses
(`dataset/response.jsonl`) were generated once, by six different source LLMs
via their respective APIs, with whatever decoding settings each provider used
at collection time — not something this pipeline controls or reproduces. This
pipeline's Stage 1 only generates fresh responses when you evaluate *your
own* generation model; it does not touch the corpus's original responses. Use
`--gold-f1` to skip Stage 1 entirely and run the detector against the
original corpus responses instead (see Stage 2 below).

## Stage 2: detection

Current default in this repo: **sampling with near-greedy settings**
(`DEFAULT_DETECTOR_PARAMS` in `detect.py`: `do_sample=True, temperature=0.05,
top_p=0.95, top_k=40`), hardcoded and not exposed as CLI flags. This exactly
mirrors the upstream baseline's decoding call:

```python
# baseline/predict_and_evaluate.py
answer = await client.text_generation(input_prompt,
                                    max_new_tokens=512,
                                    stream=False,
                                    do_sample=True,
                                    temperature=0.05,
                                    top_p=0.95,
                                    top_k=40)
```

Why keep sampling (even at near-zero temperature) for the detector, rather
than switching to greedy:
- **Faithfulness to the published numbers.** `--gold-f1` mode exists
  specifically to reproduce the baseline's case-level precision/recall/F1
  against the corpus's gold labels. Since the released detector was
  trained/served with these exact params, matching them is the right default
  for that comparison — not because sampling is *needed* by the model, but
  because it's what "the baseline's numbers" refers to.

The baseline's retry loop is the important, easy-to-miss detail:

```python
for i in range(10):
    try:
        answer = await client.text_generation(...)
        answer = json.loads(answer)
        break
    except:
        continue
```

The detector output must parse as JSON. `do_sample=True` is not incidental
here — it's what makes the retry loop meaningful. With `temperature=0.05`,
a resample after a parse failure has some chance of landing on valid JSON on
a retry; with `do_sample=False` (greedy), a parse failure would be exactly
reproduced on every retry, and the loop would just burn 10x the compute for
the same failed result. The temperature has to be low (so the label content
stays close to what a "confident" greedy decode would produce) but non-zero
(so retries can actually help).

**This repo's `detect.py` does not have that retry loop** — a parse failure
is recorded once as `parse_failed: true` and the record moves on. That's a
real divergence worth being deliberate about:
- Sampling still matters for **faithfulness** (matching the baseline's
  decoding distribution) in `--gold-f1` mode.
- But without the retry loop, sampling no longer buys the **robustness**
  benefit it gave the baseline — here it can only add run-to-run noise to
  `hallucination_rate` / `gold_f1` (a source item's label can flip between
  two otherwise-identical runs), with no compensating drop in parse
  failures. No seed is set anywhere in the pipeline (`model.py`), so this
  noise is not even reproducible run-to-run.

Options if this matters for your use case:
1. **Leave it as-is** if you only care about matching the baseline's
   published `--gold-f1` numbers on average over enough samples — the
   noise mostly averages out at corpus scale.
2. **Switch the detector to greedy** (`do_sample=False`) if you want
   deterministic `detections.jsonl` runs and are willing to accept a small
   deviation from the baseline's exact decoding recipe (temperature=0.05 is
   already close enough to greedy that the label distribution should barely
   change).
3. **Re-add a retry-on-parse-failure loop** in `detect.py` if you want to
   keep sampling *and* recover the baseline's robustness benefit from it.

## Summary

- Stage 1 defaults to greedy for reproducible, comparable generation-model
  evaluation; sampling is opt-in via CLI flags when you want realistic
  decoding or multi-sample rate estimates.
- Stage 2 defaults to the baseline's low-temperature sampling for
  faithfulness to the published detector numbers, but — unlike the
  baseline — this repo has no retry loop, so that sampling currently buys
  faithfulness without the robustness it was originally paired with.
