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
