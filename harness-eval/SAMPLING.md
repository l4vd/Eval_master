# Sampling considerations in harness-eval

`RAGTruth-reproduce/SAMPLING.md` asks "should Stage 1 (generation) and Stage 2
(detection) sample?" for a pipeline that makes two separate model calls. This
module doesn't have that shape — there is one call into `lm_eval.simple_evaluate`
per run — so the question shows up differently here. This note explains where a
"sampling decision" exists at all in the TruthfulQA tasks this wrapper runs, and
why this repo deliberately does **not** expose decoding controls for it, unlike
`RAGTruth-reproduce`'s Stage 1.

## The three task families don't have the same relationship to sampling

`--tasks truthfulqa` expands to `truthfulqa_mc1`, `truthfulqa_mc2`, and
`truthfulqa_gen` (README.md "Tasks"; the multilingual tags follow the same
`_mc1`/`_mc2`/`_gen` split). They are not three variations on one decoding
question — two of them never generate text at all.

### `truthfulqa_mc1` / `truthfulqa_mc2` — no generation, so no sampling question

These are **loglikelihood** tasks: for each of the answer choices lm_eval does a
single teacher-forced forward pass and reads off the summed log-probability of
that continuation given the prompt. `model.generate()` is never called, so
`do_sample`/`temperature`/`top_p`/`top_k` don't apply — there is no decoding step
to configure. Scoring is a deterministic function of the model's weights and the
prompt: the same checkpoint scores identically on every run, on any hardware,
regardless of any setting in this wrapper.

This is the RAGTruth Stage-1 question answered by "the question doesn't arise" —
and it's most of what this module is used for (README leads with `mc1`/`mc2`;
the multilingual tags are MC-only).

### `truthfulqa_gen` — generation happens, but this wrapper doesn't control it

`truthfulqa_gen` does call `.generate()` (it needs free text to score against the
reference true/false answer sets with BLEU/ROUGE), so a decoding choice
(`do_sample`, `temperature`, …) genuinely exists here — the RAGTruth Stage-1
analogue. But `EvalConfig` (`config.py`), `cli.py`, and `evaluator.py` carry no
`temperature`/`do_sample`/`top_p`/`top_k`/seed field anywhere, and
`run_evaluation`'s call into `simple_evaluate` (`evaluator.py`) never passes
`gen_kwargs`. Whatever decoding parameters are baked into the **installed
`lm_eval` package's** `truthfulqa_gen` task YAML apply unmodified — this wrapper
has no opinion and exposes no way to override them short of patching lm_eval
itself. Check that YAML directly (in the installed `lm_eval` package, under
its TruthfulQA task config) if the exact decoding settings matter for your run;
this repo doesn't restate or pin them, and restating a specific value here would
risk going stale against the `lm_eval[hf]>=0.4.12,<0.5` pin in `pyproject.toml`.

### Scoring `truthfulqa_gen` is deterministic even though generating it might not be

Unlike RAGTruth's Stage 2, there is no LLM judge here: `truthfulqa_gen` is scored
by BLEU/ROUGE against a fixed set of reference true/false answers (README.md
"Tasks" table; BLEURT is disabled upstream per ARCHITECTURE.md, so it's
n-gram metrics only). So even in the one place this module does generate text,
there's no analogue of RAGTruth's "does the sampled judge call parse correctly,
and does that interact with retries" problem — the metric computation itself
never samples. Any non-determinism in a `truthfulqa_gen` score can only come
from Stage-1-equivalent generation, not from scoring.

## Why no decoding flags are exposed here, unlike `RAGTruth-reproduce`

This is a deliberate difference in the two repos' purpose, not an oversight.

`RAGTruth-reproduce`'s Stage 1 generates responses from **your own** checkpoint,
which is the object under test — so that repo exposes `--do-sample`/
`--temperature`/etc. and defaults to greedy, because the whole point is
controlling exactly how that model is sampled.

`harness-eval` exists (ARCHITECTURE.md "Why it exists") to produce a number
that's **comparable to what public leaderboards report**, by running
lm-evaluation-harness the way leaderboards do. Leaderboards run `truthfulqa_gen`
with lm_eval's stock task defaults. If this wrapper let you override its
`generation_kwargs`, a "harness TruthfulQA-gen" score here could silently stop
being the thing the module exists to produce — you'd have a number that looks
like the leaderboard protocol but isn't. So *not* exposing a decoding flag is
the correct choice for this module's stated purpose, the mirror image of
`RAGTruth-reproduce`'s choice to expose one.

The one prompting-adjacent knob this wrapper *does* expose,
`--apply-chat-template` (off by default — the completion-style protocol
leaderboards use), is not a sampling setting; it changes how the prompt is
built, not how the continuation is decoded. README.md already documents that it
materially moves MC2 (0.06 completion vs. 0.26 on a Qwen2.5-0.5B smoke test) —
mentioned here only to distinguish it from the decoding question this note is
about: it's a Stage-1-adjacent prompting choice this repo *does* let you pin,
decoding is one it deliberately doesn't.

## Practical implications

- **`mc1`/`mc2` runs**: fully deterministic. Re-running the same checkpoint
  through the same `lm_eval` version and task reproduces the exact score; no
  seed or decoding setting to track.
- **`gen` runs**: reproducibility depends on the installed `lm_eval` version's
  `truthfulqa_gen` generation_kwargs (pinned indirectly via `uv.lock` /
  `lm_eval[hf]>=0.4.12,<0.5`, per README.md "Credits" and ARCHITECTURE.md
  "Python floor") and, if those kwargs sample, on a seed this wrapper never
  sets. If you need bit-identical `gen` reruns and the installed task YAML
  turns out to sample, that is outside this module's control surface as it
  stands today.
- **Cross-repo comparison**: don't read a `RAGTruth-reproduce` hallucination
  rate and a `harness-eval` `truthfulqa_gen` BLEU/ROUGE score as "both measured
  under the same decoding policy" — one is decoded under flags you chose
  (default greedy), the other under whatever lm_eval ships.
