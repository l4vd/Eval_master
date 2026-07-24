"""Arm-vs-arm comparison with the statistically correct regime, auto-selected.

Two regimes, never mixed:

* **one-sample** — a trained ensemble vs. a *fixed point* (an arm with no seed
  variance, e.g. the untrained base evaluated once). The point has zero variance so a
  paired test is degenerate; we characterise the ensemble (mean +/- std + bootstrap CI)
  and report its distance from the point. No Wilcoxon.
* **paired** — two multi-seed ensembles that share a seed set. We pair by seed **value**
  (never by list position) and require identical seed sets by default: a mismatch raises
  :class:`~analysis.stats.SeedMismatchError` (fail loudly), fixing the training-side bug.

Comparisons run over the reference arm vs. every other arm, per (benchmark, task, metric)
key — primary keys by default. Deltas are also reported in *signed* space (direction
aligned so positive = improvement) for cross-metric ranking / plotting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from analysis.aggregate import ArmAggregate, Key
from analysis.model import signed_value
from analysis.spec import ArmMeta
from analysis.stats import PairedResult, one_sample_summary, wilcoxon_matched

ONE_SAMPLE = "one_sample"
PAIRED = "paired"


@dataclass
class Comparison:
    reference: str
    arm: str
    benchmark: str
    task: str
    metric: str
    higher_is_better: bool
    regime: str
    arm_mean: float
    arm_std: float
    reference_point: float
    delta: float            # raw: arm_mean - reference_point
    signed_delta: float     # direction-aligned: positive = arm improves on reference
    n_seeds: int
    paired: dict | None = field(default=None)      # PairedResult.to_dict() when paired
    one_sample: dict | None = field(default=None)  # OneSampleSummary.to_dict() when one-sample

    def to_dict(self) -> dict:
        return asdict(self)


def choose_regime(a: ArmMeta, b: ArmMeta) -> str:
    """One-sample if either arm is a fixed point (or lacks real seeds); else paired."""
    if a.is_fixed_point or b.is_fixed_point:
        return ONE_SAMPLE
    if a.real_seeds and b.real_seeds:
        return PAIRED
    return ONE_SAMPLE


def compare_all(
    aggregates: dict[str, ArmAggregate],
    arm_meta: dict[str, ArmMeta],
    reference: str,
    *,
    require_matched: bool = True,
    primary_only: bool = True,
    rng_seed: int = 0,
) -> list[Comparison]:
    """Compare every non-reference arm against ``reference`` over shared keys."""
    if reference not in aggregates:
        raise KeyError(f"Reference arm '{reference}' not found among {list(aggregates)}")
    ref_agg = aggregates[reference]
    ref_meta = arm_meta[reference]

    out: list[Comparison] = []
    for arm, agg in aggregates.items():
        if arm == reference:
            continue
        regime = choose_regime(ref_meta, arm_meta[arm])
        for key, m in agg.metrics.items():
            if primary_only and not m.is_primary:
                continue
            ref_m = ref_agg.get(key)
            if ref_m is None:
                continue  # reference didn't produce this metric; nothing to compare
            out.append(
                _compare_one(reference, arm, key, ref_m, m, regime,
                             require_matched=require_matched, rng_seed=rng_seed)
            )
    return out


def _compare_one(
    reference, arm, key: Key, ref_m, arm_m, regime, *, require_matched, rng_seed
) -> Comparison:
    hib = arm_m.higher_is_better
    delta = arm_m.mean - ref_m.mean
    signed_delta = signed_value(arm_m.mean, hib) - signed_value(ref_m.mean, hib)

    comp = Comparison(
        reference=reference, arm=arm,
        benchmark=arm_m.benchmark, task=arm_m.task, metric=arm_m.metric,
        higher_is_better=hib, regime=regime,
        arm_mean=arm_m.mean, arm_std=arm_m.std,
        reference_point=ref_m.mean, delta=delta, signed_delta=signed_delta,
        n_seeds=arm_m.n_seeds,
    )

    if regime == PAIRED:
        a_by_seed = {s: v for s, v in arm_m.per_seed.items() if s is not None}
        b_by_seed = {s: v for s, v in ref_m.per_seed.items() if s is not None}
        paired: PairedResult = wilcoxon_matched(
            a_by_seed, b_by_seed, require_matched=require_matched, seed=rng_seed
        )
        comp.paired = paired.to_dict()
    else:
        summ = one_sample_summary(
            list(arm_m.per_seed.values()), base_point=ref_m.mean, seed=rng_seed
        )
        comp.one_sample = summ.to_dict()
    return comp
