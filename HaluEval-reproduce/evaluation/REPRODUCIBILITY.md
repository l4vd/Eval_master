# Reproducibility of `evaluate.py`

## The problem

`evaluate.py` decides, per example, whether the judge is shown the
hallucinated answer/response/summary or the correct one via an unseeded coin
flip:

```python
if random.random() > 0.5:
    answer = hallucinated_answer
    ground_truth = "Yes"
else:
    answer = right_answer
    ground_truth = "No"
```

This appears once per task: `evaluation_qa_dataset` (evaluate.py:203),
`evaluation_dialogue_dataset` (evaluate.py:261), and
`evaluation_summarization_dataset` (evaluate.py:317). `random.seed(...)` is
never called anywhere in the script.

Consequences:

- Each run draws an independent, random 50/50 partition of "which examples
  are shown as hallucinated vs. correct." The judge effectively sees a
  different dataset every time the script is run.
- Reported accuracy is therefore a noisy random variable, not a fixed number
  for a given model/judge. Re-running the identical model can shift accuracy
  by a few points from sampling variance alone.
- When comparing two models or checkpoints without pinning the seed, part of
  any observed accuracy delta may reflect *which half got sampled* rather
  than a genuine difference in judge quality — the comparison is unpaired
  and confounded.

This is inherited from the original HaluEval repository, not something
introduced in this fork. Reproducing it as-is is defensible if the goal is
"run the exact published script," but it should not be relied on directly
for comparative claims (e.g., in a thesis) without addressing the sampling
noise.

Note what this issue is *not*: it is not about judge-side stochasticity.
OpenAI calls already use `temperature=0.0`, and the local HF judge backend
already uses `do_sample=False` (`hf_local.py:204`, `hf_local.py:210`,
greedy decoding). The unaddressed randomness is purely in the
label-sampling step described above.

## Options, in order of rigor

### 1. Minimum fix — seed it

Add `random.seed(SEED)` (ideally exposed as a `--seed` CLI argument) before
the sampling loop in each `evaluation_*_dataset` function. This makes runs
deterministic, and — more importantly — lets every model/judge under
comparison be evaluated against the *same* label partition, turning an
unpaired comparison into a paired one. Cheap to implement, but still only
evaluates one class per example, so it doesn't remove sampling variance
itself — it just fixes the draw.

### 2. Better — evaluate both classes, drop the sampling entirely

Instead of randomly picking one of `{hallucinated, right}` per example,
evaluate *both* for every example (2n judge calls instead of n). This:

- Gives an exact, deterministic accuracy for a given n — no sampling
  variance left at all.
- Decomposes the single blended accuracy into **TPR** (hallucination
  correctly flagged) and **TNR** (correct answer correctly cleared), which
  is far more diagnostic. A judge that always answers "Yes" scores ~50%
  under the random-sampling scheme but is immediately exposed once the two
  classes are checked separately.

This is the recommended approach for reporting results, since it is
strictly more informative than option 1 while also being fully
reproducible.

### 3. If the sampling design must be kept (e.g. to match compute budget or stay literally faithful to the original protocol)

Run the evaluation under **K fixed seeds** (5–10 is typical), computing
accuracy (and TPR/TNR) for each seed, and report **mean ± std** (or a
bootstrap/normal-approximation confidence interval) across seeds. Use the
*same* set of seeds for every model being compared (paired design) so that a
paired significance test can be used across seeds — a paired t-test,
Wilcoxon signed-rank test across seeds, or McNemar's test on the per-example
judgments — rather than comparing two unpaired point estimates.

## Recommendation

For thesis-quality comparisons, use **option 2** (evaluate both classes per
example, report accuracy/TPR/TNR) as the primary metric, and note in the
methods section that the original HaluEval script relies on unseeded random
sampling — this is the motivation for the change. Option 1 (seeding) is an
acceptable minimal fix if reproducing the original per-example sampling
protocol as closely as possible is a priority.

## What this fork actually does

**Option 1, plus the diagnostics option 2 was wanted for.** The sampling design is
kept — it is the published protocol, and staying on it is what keeps these numbers
comparable to the paper's — but everything that made the design hard to interpret
has been instrumented.

- `--seed` (launcher: `halueval.seed`, default `42`) pins the label draw. Unset
  reproduces upstream's unseeded behaviour. The draw happens once per row in
  ascending index order before any batching or reordering, so the same seed yields
  the same labels regardless of `batch_size` / `sort_by_length`
  (`tests/test_evaluate_ordering.py`).
- **TPR and TNR are reported anyway**, without paying option 2's 2n judge calls.
  Option 2's headline argument was that a constant "Yes" judge scores ~50% and
  looks unremarkable under the blended metric; per-class recall on the sampled
  half exposes exactly that, just on half the sample. This turned out to matter
  immediately: the checkpoints evaluated here answer `"No"` to *every* row.
- `format_compliance` / `num_failed` separate "judged wrongly" from "never emitted
  a verdict". Upstream scores an unparseable judgement as **incorrect**, which is
  reproduced faithfully — but it means accuracy silently mixes in a judge's
  inability to follow the output format, and for small instruction-tuned models
  that term can dominate.
- Each row stores the judge's `raw_judgement`, so a finished run can be re-scored
  offline under a different parser (`evaluation/score_results.py`) without
  re-running generation.

For error bars, option 3 is now a one-liner — sweep the seed and report mean ± std:

```
./run_all.sh --multirun run='[halueval]' halueval.seed=42,43,44,45,46
```

Use the same seed set for every arm so the comparison stays paired.
