"""Across-seed aggregation of long-form records into per-arm statistics.

For each arm and each (benchmark, task, metric) key, collect the per-seed values and
summarise them: mean, std (ddof=1), and a bootstrap CI of the across-seed mean — the
same two-granularity bootstrap the training side uses, here at the across-seed level.

Nothing hardcodes the number of seeds or benchmarks: whatever keys and seeds appear in
the :class:`~analysis.model.RecordSet` are aggregated. The single-instance case (one
seed) flows through unchanged — ``n_seeds == 1``, ``std == 0``, degenerate CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from analysis.model import MetricRecord, RecordSet
from analysis.stats import one_sample_summary

Key = tuple[str, str, str]  # (benchmark, task, metric)


@dataclass
class MetricAggregate:
    benchmark: str
    task: str
    metric: str
    higher_is_better: bool
    is_primary: bool
    n_seeds: int
    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float
    per_seed: dict[int | None, float] = field(default_factory=dict)

    @property
    def key(self) -> Key:
        return (self.benchmark, self.task, self.metric)

    @property
    def signed_mean(self) -> float:
        from analysis.model import signed_value

        return signed_value(self.mean, self.higher_is_better)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        d = asdict(self)
        # JSON object keys must be strings; keep seed ordering stable.
        d["per_seed"] = {str(k): v for k, v in self.per_seed.items()}
        return d


@dataclass
class ArmAggregate:
    arm: str
    metrics: dict[Key, MetricAggregate] = field(default_factory=dict)

    def get(self, key: Key) -> MetricAggregate | None:
        return self.metrics.get(key)

    def primary(self) -> list[MetricAggregate]:
        return [m for m in self.metrics.values() if m.is_primary]

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "metrics": [m.to_dict() for m in _sorted_metrics(self.metrics.values())],
        }


def aggregate_arm(records: RecordSet, arm: str, *, seed: int = 0) -> ArmAggregate:
    """Aggregate one arm's records across seeds into per-key statistics."""
    rs = records.filter(arm=arm)
    grouped: dict[Key, list[MetricRecord]] = {}
    for r in rs:
        grouped.setdefault(r.key, []).append(r)

    out: dict[Key, MetricAggregate] = {}
    for key, recs in grouped.items():
        # Guard against duplicate (seed,key) — keep the last seen, but a real run
        # never emits duplicates; this only protects against accidental double-parse.
        per_seed: dict[int | None, float] = {r.seed: r.value for r in recs}
        values = list(per_seed.values())
        summ = one_sample_summary(values, base_point=None, seed=seed)
        proto = recs[0]
        out[key] = MetricAggregate(
            benchmark=proto.benchmark, task=proto.task, metric=proto.metric,
            higher_is_better=proto.higher_is_better, is_primary=proto.is_primary,
            n_seeds=summ.n, mean=summ.mean, std=summ.std,
            ci_95_lower=summ.ci_95_lower, ci_95_upper=summ.ci_95_upper,
            per_seed=per_seed,
        )
    return ArmAggregate(arm=arm, metrics=out)


def aggregate_all(records: RecordSet, *, seed: int = 0) -> dict[str, ArmAggregate]:
    """Aggregate every arm present in ``records``."""
    return {arm: aggregate_arm(records, arm, seed=seed) for arm in records.arms()}


def _sorted_metrics(metrics):
    return sorted(metrics, key=lambda m: (m.benchmark, m.task, m.metric))
