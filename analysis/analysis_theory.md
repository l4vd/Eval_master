# Theory of the Eval-Analysis Layer: Estimands, Regimes and Inference

This document derives what the `analysis` package is *estimating*, which randomness it does
and does not quantify, why the arm-vs-arm comparison branches into two regimes that must
never be mixed, and where the inference it produces is and is not trustworthy. It is
checkable against `analysis.{model, parse, discover, spec, aggregate, compare, stats, plot,
report}`.

Companion documents: `SP-DPO-Base/notebooks/selection_theory.md` (how the training pairs
were chosen, i.e. what makes an arm's *training* set a valid one) and
`SP-DPO-Base/notebooks/profiling_theory.md`. This one covers what happens after the runs
finish.

---

## 1. The object of study

Fix a benchmark suite. Let

$$
M(a, s, k) \;\in\; \mathbb R
\tag{1}
$$

be the scalar produced by **arm** $a$ (a training configuration: base / SFT / DPO /
Off-SP-DPO), at **seed** $s$, for **key** $k = (\text{benchmark}, \text{task},
\text{metric})$. That triple is the atom: `MetricRecord`. Everything downstream —
aggregation, comparison, plotting — consumes records; nothing re-reads a raw `summary.json`.

The design decisions encoded in Eq. (1) are worth naming, because each removes a special
case that would otherwise have to be handled everywhere:

* **The single-instance case is not a special path.** One model at one seed is an ensemble
  of size 1, with `seed` set and a degenerate CI. No branch anywhere tests for it.
* **`seed = None` means something specific**: a genuine fixed point carrying no seed
  identity — an untrained base evaluated once. It is not a missing value to be imputed;
  §4 turns it into a regime choice.
* **`higher_is_better` is a property of the record**, not of a lookup table consulted at
  plot time. `harness` rows carry their own flag; the others get one from a small
  benchmark×metric table in `parse.py`. Direction travels *with* the number, so a metric
  cannot be ranked the wrong way round in one figure and the right way in another.

### 1.1 Two sources of randomness, and which one is quantified

$M(a,s,k)$ varies for two independent reasons:

1. **Training/inference stochasticity** — initialization, data order, sampling — indexed by
   $s$. This is *between-run* variance.
2. **Finite benchmark size** — the benchmark is a sample of items from some population.
   This is *within-run*, item-level variance, and each benchmark's own `stderr` (where it
   reports one) estimates it.

This layer quantifies **(1) only**. Every CI, every $p$-value, every "$n$" below counts
*seeds*, not items. The estimand is therefore

$$
\mu(a,k) \;=\; \mathbb E_{s}\bigl[\,M(a,s,k)\,\bigr]
\tag{2}
$$

— the expected score of the *training procedure* $a$ on this *fixed* benchmark, averaging
over training randomness. It is **not** the expected score on a fresh sample of items.

This is the right choice for the question the thesis asks (does this training method beat
that one?), but the limitation must be stated rather than left implicit: a conclusion drawn
here generalizes across re-runs of training, not across benchmarks or item samples. The
per-record `stderr` is carried through the data model precisely so a future two-level
(seed × item) treatment is possible without re-parsing, but no current code path combines
the two.

### 1.2 Orientation, and the ceiling on what it can do

Metrics disagree about direction: accuracy up is good, `ragtruth/hallucination_rate` down is
good. Define

$$
\tilde M \;=\; \begin{cases} M & \text{higher is better}\\ -M & \text{otherwise}\end{cases}
\tag{3}
$$

so that $\tilde M(a) - \tilde M(b) > 0$ **iff** $a$ is the better arm, for every metric.

Eq. (3) fixes *sign* comparability and nothing else. It does **not** make magnitudes
commensurable: a $+0.02$ accuracy gain and a $+0.02$ hallucination-rate reduction are the
same number on incomparable scales with different noise floors. This matters at exactly one
place — `plot_ranked_deltas` sorts raw signed deltas across keys — and it is a documented
hazard rather than a bug, because the ranking it produces is a *reading aid*, not a result.
If a cross-metric ranking ever needs to carry weight, standardize first:

$$
\hat d(a,b,k) \;=\; \frac{\hat\mu(a,k) - \hat\mu(b,k)}{\hat\sigma_{\text{pooled}}(k)}
\tag{4}
$$

— a Cohen's-$d$ in units of across-seed standard deviation, which *is* comparable across
keys. Eq. (4) is not currently computed; `signed_value`'s own source note flags the gap.

---

## 2. Parsing as a contract, not a convenience

Five benchmarks write five different on-disk shapes: FaithEval and HaluEval one flat
`*_summary.json` per task, RAG-Truth a nested `summary.json`, TruthfulQA a `summary.csv`
pivot with no JSON at all, and the lm-eval harness a `summary.json` whose `results` is a
flat list of rows carrying their own direction flag. A registry maps benchmark → reader, so
adding or removing a benchmark is a one-line edit and nothing hardcodes "five".

Two parsing decisions have inferential consequences and belong in this document rather than
in a README:

**Synthesized `task = "mean"` records are not primary.** FaithEval and HaluEval get a
convenience record averaging accuracy over their tasks. It is excluded from the primary set
because including it would **double-count**: the per-task accuracies and their mean are not
independent, so a ranked-delta figure or a multiplicity correction over primary keys would
weight those benchmarks more heavily than the others by an amount determined by how many
tasks they happen to have.

**Diagnostics are parsed but never primary.** HaluEval's `format_compliance`, `tpr`, `tnr`,
`judged_yes_rate` explain a headline number rather than competing with it — `accuracy`
alone cannot distinguish "judged wrongly" from "never emitted a verdict" from "answered one
constant label". `judged_yes_rate` in particular is **non-directional**: 0.0 and 1.0 are
both degenerate and $\approx 0.5$ is healthy, so it carries a direction flag only because
the table needs one, and must never be ranked on.

### 2.1 The primary predicate as a pre-registration device

$$
\texttt{is\_primary}: (\text{benchmark},\text{task},\text{metric}) \to \{0,1\}
\tag{5}
$$

selects the headline keys, defaulting to per-task accuracy (FaithEval, HaluEval), overall
hallucination rate (RAG-Truth), MC1/MC2 (TruthfulQA) and the two TruthfulQA harness tasks.
It is overridable at runtime from a YAML map.

Functionally this is a **pre-registration** mechanism: fixing the primary set *before*
looking at the comparisons is what prevents the reported result from being the best of
however many keys happened to be produced. Its value as such is entirely a matter of when
the override is written. An override authored after reading `comparisons.json` provides no
protection whatsoever — it is the same selection effect with extra steps. The override
shape is deliberately explicit (a benchmark listed in the override uses *only* its listed
entries; benchmarks absent keep the built-in default), so what was pre-registered is
auditable from the config.

---

## 3. Aggregation across seeds

For arm $a$ and key $k$ with observed seeds $s_1,\dots,s_n$:

$$
\hat\mu = \frac1n\sum_i M(a,s_i,k),
\qquad
\hat\sigma = \sqrt{\frac{1}{n-1}\sum_i\bigl(M(a,s_i,k)-\hat\mu\bigr)^2}
\tag{6}
$$

with $\hat\sigma := 0$ at $n=1$ rather than undefined. The $n-1$ denominator is the
unbiased estimator of the across-seed variance; at these $n$ the distinction from $n$ is not
cosmetic (at $n=5$ it is a 12 % difference in $\hat\sigma$).

The interval is a **percentile bootstrap** (Efron, 1979 [1]) of the across-seed mean:
resample the $n$ seed values with replacement $B = 10^4$ times, and take the empirical
$[\alpha/2,\,1-\alpha/2]$ quantiles of the resampled means.

$$
\widehat{\mathrm{CI}}_{1-\alpha} \;=\;
\Bigl[\,Q_{\alpha/2}\bigl(\{\hat\mu^{*b}\}_{b=1}^{B}\bigr),\;\;
Q_{1-\alpha/2}\bigl(\{\hat\mu^{*b}\}_{b=1}^{B}\bigr)\,\Bigr]
\tag{7}
$$

Percentile, not BCa: no bias or acceleration correction is applied. The implementation is
copied **verbatim** from `hallu_mitigate.evaluation.metrics.bootstrap_ci` on the training
side, because the two projects live in separate virtualenvs with no clean shared import
path, and behaviour matching the training-side ensemble figures exactly is worth more here
than DRY.

### 3.1 What Eq. (7) is worth at $n = 5$

This is the single most important caveat in this document, so it is quantified rather than
gestured at. Simulating i.i.d. normal seed values and measuring how often the nominal-95 %
interval covers the true mean:

| seeds $n$ | empirical coverage of the nominal 95 % percentile CI |
|---|---|
| 3 | 0.75 |
| 4 | 0.80 |
| **5** | **0.84** |
| 6 | 0.86 |
| 8 | 0.88 |
| 10 | 0.90 |
| 20 | 0.93 |

The percentile bootstrap is **anti-conservative at small $n$**: it does not know about the
$t$-correction, so its intervals are systematically too narrow (at $n=5$, mean width 1.44
against the normal-theory 1.75 in $\sigma$ units). A "95 % CI" computed from 5 seeds is
closer to an 84 % CI.

Three responses, in order of preference:

1. **More seeds.** Coverage, and §5's power wall, both improve monotonically and cheaply.
2. **Report the interval descriptively.** "the spread across five seeds" is an honest
   statement; "a 95 % confidence interval" is not, at $n=5$.
3. **Use a $t$-interval or a BCa/studentized bootstrap** if a calibrated interval is
   required at small $n$. Neither is currently implemented.

Nothing about this is specific to this repository — it is the standard small-$n$ regime
warned about throughout the ML-benchmarking literature (Bouthillier et al., 2021 [2];
Dror et al., 2018 [3]).

---

## 4. Two regimes, never mixed

`compare_all` compares a reference arm against every other arm, per primary key. The
statistical machinery it uses is **not** a free choice: it is determined by whether both
arms carry seed variance.

$$
\text{regime}(a,b) =
\begin{cases}
\textsf{one\_sample} & \text{if } a \text{ or } b \text{ is a fixed point, or either lacks real seeds}\\
\textsf{paired} & \text{otherwise}
\end{cases}
\tag{8}
$$

A *fixed point* is an arm with no seed variance: a single run directory whose seed is
unrecoverable (auto-detected), or one forced with a `!fixed` marker on the spec.

### 4.1 One-sample: ensemble vs a constant

The base model evaluated once contributes a **single number with zero variance**. A paired
test against it is degenerate — there is nothing to pair with, and the "differences" it
would rank are just the arm's own values shifted by a constant, so a Wilcoxon on them tests
whether the *arm* differs from the point, using only the arm's variance while presenting
itself as a paired comparison.

The correct treatment is to characterize the arm and report its distance from the point:

$$
\hat\delta \;=\; \hat\mu(a,k) \;-\; m_{\text{base}},
\qquad
\text{uncertainty from } \{M(a,s_i,k)\}_i \text{ alone (Eqs. 6--7)}
\tag{9}
$$

with **no** $p$-value. The estimand in Eq. (9) is $\mu(a,k) - m_{\text{base}}$, in which
$m_{\text{base}}$ is a *constant*, not a random variable — so the interval on $\hat\delta$
is exactly the interval on $\hat\mu(a,k)$, shifted. That is a weaker claim than a paired
test, and the weakness is real rather than a modelling choice: the base's own run-to-run
variability was never measured, so it cannot be accounted for.

### 4.2 Paired: two ensembles over a shared seed set

When both arms are ensembles, form differences **by seed value**:

$$
D_i \;=\; M(a, s_i, k) - M(b, s_i, k), \qquad s_i \in S_a \cap S_b
\tag{10}
$$

and estimate $\mathbb E[D]$. This is a strictly better-conditioned estimand than the
difference of two independent means, by the elementary identity

$$
\operatorname{Var}(\bar D) \;=\; \frac{1}{n}\Bigl(\operatorname{Var}(A) + \operatorname{Var}(B) - 2\operatorname{Cov}(A,B)\Bigr).
\tag{11}
$$

Two arms trained at the *same* seed share their initialization draw and data order, so
$\operatorname{Cov}(A,B) > 0$ and the paired variance is strictly smaller than the unpaired
one. **Pairing is where the statistical power comes from at these sample sizes**, which is
why the pairing key matters so much.

### 4.3 Pairing by seed value, and the bug this fixes

The training side's `ensemble.aggregation.paired_comparison` paired two arms **by list
position** — `a[:n], b[:n]` after `n = min(sizes)` — with no seed-identity check. When the
two arms' seed sets differ, or are merely ordered differently on disk, that silently pairs
run $s=42$ of one arm with run $s=1337$ of the other. The consequences are not subtle:

* $\operatorname{Cov}(A,B)$ in Eq. (11) collapses to $\approx 0$, so the variance reduction
  that motivates pairing is thrown away while the report still claims a paired test;
* if the seed multisets differ, $\bar D$ is not an estimate of $\mathbb E[D]$ at all;
* the Wilcoxon null (§5) assumes the $D_i$ are symmetric about zero *under the null of no
  arm effect*, which a mispaired difference need not be even when the null holds.

`stats.wilcoxon_matched` pairs by seed value and, by default, **raises**
`SeedMismatchError` on any seed-set difference, naming the seeds present in only one arm.
`require_matched=False` intersects instead — still aligned by value, never by position.
Failing loudly is the right default because the failure it guards against is invisible in
the output: a mispaired comparison produces a perfectly plausible-looking $p$-value.

The same principle governs discovery. `analysis.discover.infer_seed` reads the seed from
`run_metadata.json`, else from a `seed_<N>` directory name, else returns **`None`**. There
is deliberately no `hash(name) % 100000` fallback of the kind the training side uses: a
fabricated seed identity would join two unrelated runs and produce exactly the failure above.
An unrecoverable seed becomes a fixed point (§4.1), which is a weaker but honest treatment.

`eval_checkpoints.py` exists to keep this invariant across the training/eval boundary: it
writes each checkpoint's benchmark outputs to a **seed-preserving** layout
`<out>/seed_<SEED>/` plus a `run_metadata.json`, so `discover` recovers the *same* seed
value the training run used and Eq. (10) lines up.

---

## 5. The Wilcoxon signed-rank test, and its hard power floor at these $n$

### 5.1 The test

Rank $|D_1|,\dots,|D_n|$; let $W^+$ be the sum of ranks of the positive $D_i$ and
$W^- = \tfrac{n(n+1)}{2} - W^+$; the statistic is $\min(W^+, W^-)$ (Wilcoxon, 1945 [4]).
Under $H_0$ the $D_i$ are i.i.d. and symmetric about zero, so their **signs are independent
Rademacher** and the exact null distribution is obtained by enumerating all $2^n$ sign
assignments. The test is distribution-free beyond that symmetry assumption, which is why it
is preferred over a paired $t$-test on 5 points from an unknown distribution.

Two degenerate cases are handled before SciPy is consulted, in this order:

* $n = 0$ shared seeds → statistic and $p$ are `NaN`, with `note = "no shared seeds"`;
* $A \equiv B$ (nothing to rank) → the conventional $(0, 1)$, returned *before* the import
  so the result stays correct without SciPy installed.

### 5.2 The floor

The two-sided exact $p$-value is bounded below by the probability of the most extreme sign
pattern:

$$
p_{\min}(n) \;=\; \frac{2}{2^{\,n}}
\tag{12}
$$

| $n$ | $p_{\min}$ | can reach $\alpha = 0.05$? |
|---|---|---|
| 3 | 0.250 | no |
| 4 | 0.125 | no |
| **5** | **0.0625** | **no** |
| 6 | 0.03125 | yes |
| 7 | 0.0156 | yes |
| 10 | 0.00195 | yes |

$$
\boxed{\;\text{With 5 seeds, no paired Wilcoxon result can ever be significant at }\alpha=0.05,
\text{ however large the effect.}\;}
$$

`significance_stars` can therefore return nothing but `"ns"` for a 5-seed paired
comparison — not because the arms are equivalent, but because the test has no resolution at
that $n$. This is a combinatorial fact about Eq. (12), independent of effect size, variance,
and every other property of the data. **Six seeds is the minimum at which the paired test
in this package can produce a significant result at all**, and $n \ge 10$ is where it has
usable power against realistic effect sizes.

Consequences for how results should be read and reported:

* At $n \le 5$, the $p$-value column carries no information and should not be presented as
  evidence of *equivalence*. Absence of significance here is guaranteed a priori.
* The paired **bootstrap CI of the differences**, which `wilcoxon_matched` returns
  alongside, is the more informative summary at small $n$ — subject to §3.1's coverage
  caveat, which applies to it identically.
* The cheapest fix is arithmetic: run 6–10 seeds. Eq. (12) is the reason the marginal
  6th seed is worth more than any amount of post-hoc analysis.

### 5.3 `NaN` is not `ns`

`significance_stars(NaN)` returns the **empty string**, and `"ns"` is reserved for a test
that ran and did not reject. The distinction is load-bearing: an uncomputed test rendered as
a blank cell next to computed ones reads as "not significant" when it means "not tested".
`PairedResult.note` carries the reason — `"no shared seeds"`, `"scipy not installed"`,
`"wilcoxon declined this sample"` — and the LaTeX writer prints `n/a (<note>)` rather than
a blank, because "SciPy is missing" and "SciPy declined this sample" call for different
fixes and neither is a result.

---

## 6. Multiplicity

The comparison grid has

$$
N_{\text{tests}} \;=\; \bigl(|\text{arms}| - 1\bigr)\times\bigl|\{k : \texttt{is\_primary}(k)\}\bigr|
\tag{13}
$$

cells, and across five benchmarks with per-task primaries that is routinely tens of tests.
An uncorrected 0.05 threshold controls nothing at the family level, so
`stats.adjust_pvalues` supplies two standard procedures and `compare_all` applies one
across the whole paired family:

* **Holm–Bonferroni** (Holm, 1979 [5]), the default — step-down,
  $\tilde p_{(k)} = \max_{j\le k}\,(m-j+1)\,p_{(j)}$, clipped to 1 and made monotone by the
  running maximum. Controls the **family-wise error rate**: the probability of *any* false
  positive. Uniformly more powerful than plain Bonferroni under the same (no) assumptions,
  so there is no reason to prefer Bonferroni. Appropriate when one spurious win would
  undermine the claim, which is the situation for a "this method beats the baseline" result.
* **Benjamini–Hochberg** (1995 [6]) — step-up,
  $\tilde p_{(k)} = \min_{j\ge k}\,\tfrac{m}{j}\,p_{(j)}$. Controls the **false discovery
  rate**: the expected *proportion* of false positives among rejections. Appropriate for a
  screening table where a known share of false positives is tolerable.
* **`none`** — offered explicitly, so switching the correction off is a recorded
  configuration choice rather than a silently absent key.

Three properties are load-bearing and are pinned by tests:

1. **The family is the paired comparisons only.** One-sample cells carry no $p$-value at
   all (§4.1), so including them would inflate $m$ on behalf of tests never performed.
2. **`NaN` is excluded from the family**, not counted in it. A test that did not run — no
   shared seeds, SciPy absent, SciPy declined the sample — must not penalise the tests that
   did. `mc_family_size` records the size actually corrected over.
3. **The raw $p$ is kept beside the adjusted one** in `comparisons.json` and in the LaTeX
   table, so the correction is auditable rather than baked in. The **stars carry the
   adjusted value**, since that is the one whose threshold means something.

At the present $n \le 5$ this changes no conclusion, for a specific and slightly absurd
reason: by §5.2 nothing can be significant anyway, so the family-wise error rate is zero by
construction. It becomes load-bearing the moment the seed count reaches 6 — which is also
the moment the paired test can reject at all.

Precedent elsewhere in this codebase: the selection layer's `diagnostics.py` offers an
explicit Bonferroni quantile ($z = 2.9353$ for 15 simultaneous matchup intervals) for the
same reason.

Demšar (2006 [7]) is the standard reference for the multi-arm/multi-benchmark case; his
Friedman-plus-post-hoc procedure is the natural next step if the design grows to enough
arms that pairwise-versus-reference stops being the right structure.

---

## 7. What the figures can and cannot assert

Five altitudes, each consuming only the intermediate data model — never a raw summary — so
figures can be restyled long after the evals finished by reloading `records.jsonl` and
`comparisons.json`.

| # | Figure | Claim it can support |
|---|---|---|
| 1 | `plot_benchmark` | one benchmark, arms side by side, CI bars + per-seed points |
| 2 | `plot_tasks` | per-task grouped bars within a benchmark |
| 3 | `plot_cross_benchmark_panels` | small multiples on **native scales** — deliberately not a shared axis |
| 4 | `plot_ranked_deltas` | direction-normalized deltas, sorted — a reading aid (§1.2), not a ranking |
| 5 | `plot_paired_deltas` | per-seed deltas + mean + CI + stars — **paired regime only** |

Figure 5 refuses to render for a one-sample comparison, with a warning rather than a
silently different plot: a paired-delta figure drawn against a zero-variance point would
display five "differences" that are one number minus a constant, and would read as though
the base had been measured five times.

Overplotting the individual seed points (figures 1 and 5) is not decoration. At $n=5$ it is
the most honest element on the page — it shows the reader the entire evidence base, which no
summary statistic at that $n$ can substitute for.

Colour is the Okabe-Ito colourblind-safe qualitative palette (Okabe & Ito, 2008 [8]),
assigned to arms in a **fixed** order: colour follows arm identity, never rank, and beyond
eight arms extras fold to grey rather than cycling — a repeated colour asserts an identity
that is not there.

---

## 8. Persisted artifacts

| File | Contents | Reload point |
|---|---|---|
| `records.jsonl` | the long-form `MetricRecord` dump, one JSON object per row | `report.load_records` → standalone plotting |
| `aggregate.json` | per arm, per key: $n$, mean, std, CI, and the full `per_seed` map | §3 |
| `aggregate.tex` | booktabs appendix table: mean $\pm$ std [CI], with a $\downarrow$ marker on lower-is-better metrics | §3 |
| `comparisons.json` | one record per (arm, key): regime, raw delta, signed delta, plus the regime-specific `paired` **or** `one_sample` block. The `paired` block carries `p_value`, `p_value_adjusted`, `mc_method`, `mc_family_size` | §4, §6 |
| `comparisons.tex` | booktabs table: signed $\Delta$, regime, and Wilcoxon $p$ + adjusted $p$ (stars on the adjusted one) **or** one-sample CI | §4–6 |
| `*.png` | the five figure altitudes | §7 |

Retaining `per_seed` in `aggregate.json` is what makes the whole layer re-analysable: the
raw seed values are the sufficient statistic for every test in §§3–5, so a corrected
interval, a multiplicity adjustment, or a different test can be computed later without
re-running a single evaluation.

JSON is written `indent=2, ensure_ascii=False` and the LaTeX follows the training side's
`ensemble.aggregation.to_latex_table` conventions, so appendix tables look consistent across
the two projects.

---

## 9. Summary of the standing caveats

Ordered by how much they should change what gets claimed:

1. **$n \le 5$ seeds makes the paired $p$-value uninformative by construction** (§5.2,
   Eq. 12). Fix: 6 seeds minimum, 10 preferred.
2. **The percentile bootstrap CI is anti-conservative at these $n$** — nominal 95 % is
   empirically ~84 % at $n=5$ (§3.1). Fix: more seeds, a $t$- or BCa interval, or
   descriptive framing.
3. **Multiplicity is corrected (Holm by default) but only across the *paired* family**
   (§6). One-sample comparisons carry no $p$-value and are excluded by construction, so a
   grid dominated by fixed-point references is effectively uncorrected — not wrongly, but
   the family being controlled is smaller than the number of claims being read.
4. **Item-level uncertainty is never combined with seed-level uncertainty** (§1.1). The
   estimand is "across re-runs on this benchmark", not "across item samples".
5. **Cross-metric magnitudes are not commensurable** (§1.2); only signs are. Fix: Eq. (4)
   before any cross-metric ranking is used as evidence.
6. **The primary-key set only functions as a pre-registration if it is fixed before the
   comparisons are read** (§2.1). This is a process property, not a code property, and no
   code can enforce it.

---

## References

[1] Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife.* Annals of
Statistics, 7(1). (Percentile bootstrap; see also Efron, 1987, for BCa.)

[2] Bouthillier, X., Delaunay, P., Bronzi, M., et al. (2021). *Accounting for Variance in
Machine Learning Benchmarks.* MLSys 2021.

[3] Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018). *The Hitchhiker's Guide to
Testing Statistical Significance in Natural Language Processing.* ACL 2018.

[4] Wilcoxon, F. (1945). *Individual Comparisons by Ranking Methods.* Biometrics Bulletin,
1(6).

[5] Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure.*
Scandinavian Journal of Statistics, 6(2).

[6] Benjamini, Y., & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A
Practical and Powerful Approach to Multiple Testing.* JRSS-B, 57(1).

[7] Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.*
JMLR, 7.

[8] Okabe, M., & Ito, K. (2008). *Color Universal Design (CUD): How to Make Figures and
Presentations That Are Friendly to Colorblind People.*

[9] Card, D., Henderson, P., Khandelwal, U., Jia, R., Mahowald, K., & Jurafsky, D. (2020).
*With Little Power Comes Great Responsibility.* EMNLP 2020. (Power analysis for NLP
experiments; the direct argument for §5.2.)
