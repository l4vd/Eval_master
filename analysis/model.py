"""The canonical intermediate data model.

Everything downstream — aggregation, comparison, plotting — consumes
:class:`MetricRecord`; nothing re-reads raw ``summary.json`` / ``summary.csv``.
A record is one scalar number produced by one benchmark for one arm at one seed.

Design notes
------------
* ``direction`` is folded into a single boolean ``higher_is_better``; the
  :func:`signed_value` helper flips lower-is-better metrics so that cross-metric
  deltas and rankings are always oriented "higher = better / improvement".
* The single-instance case (one model, one seed) is *not* a special path: it is
  simply an ensemble of size one, with ``seed`` set and no CI.
* ``seed`` is ``None`` only for a genuine fixed point that carries no seed
  identity (e.g. an untrained base evaluated once).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class MetricRecord:
    """One scalar metric for (arm, seed, benchmark, task, metric)."""

    arm: str
    seed: int | None
    benchmark: str
    task: str
    metric: str
    value: float
    stderr: float | None
    higher_is_better: bool
    n_samples: int | None
    is_primary: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        """The (benchmark, task, metric) identity a record is aggregated under."""
        return (self.benchmark, self.task, self.metric)

    @property
    def signed_value(self) -> float:
        """``value`` oriented so higher is always better (see :func:`signed_value`)."""
        return signed_value(self.value, self.higher_is_better)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetricRecord":
        # Tolerate extra keys from future schema growth; take only our fields.
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{k: d[k] for k in fields if k in d})


def signed_value(value: float, higher_is_better: bool) -> float:
    """Return ``value`` if higher-is-better else ``-value``.

    Used so that a delta ``signed(a) - signed(b)`` is positive exactly when ``a``
    is the better arm, regardless of the metric's native direction.
    """
    return value if higher_is_better else -value


class RecordSet:
    """A thin, immutable-ish wrapper over ``list[MetricRecord]`` with filters.

    Keeps the data layer plotting- and stats-agnostic: callers slice with
    :meth:`filter` and read back plain lists / sorted key sets.
    """

    def __init__(self, records: Iterable[MetricRecord]):
        self._records: list[MetricRecord] = list(records)

    def __iter__(self) -> Iterator[MetricRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __bool__(self) -> bool:
        return bool(self._records)

    @property
    def records(self) -> list[MetricRecord]:
        return list(self._records)

    def filter(
        self,
        *,
        arm: str | None = None,
        seed: int | None = None,
        benchmark: str | None = None,
        task: str | None = None,
        metric: str | None = None,
        primary: bool | None = None,
    ) -> "RecordSet":
        """Return a new RecordSet keeping only records matching every set predicate."""
        def keep(r: MetricRecord) -> bool:
            return (
                (arm is None or r.arm == arm)
                and (seed is None or r.seed == seed)
                and (benchmark is None or r.benchmark == benchmark)
                and (task is None or r.task == task)
                and (metric is None or r.metric == metric)
                and (primary is None or r.is_primary == primary)
            )

        return RecordSet(r for r in self._records if keep(r))

    def include_benchmarks(
        self, include: Iterable[str] | None, exclude: Iterable[str] | None
    ) -> "RecordSet":
        """Apply the per-benchmark enable/disable selection (``--benchmarks`` / ``--exclude``)."""
        inc = set(include) if include else None
        exc = set(exclude) if exclude else set()
        return RecordSet(
            r
            for r in self._records
            if (inc is None or r.benchmark in inc) and r.benchmark not in exc
        )

    # -- read-back helpers used by aggregation / plotting ------------------------

    def arms(self) -> list[str]:
        return _sorted_unique(r.arm for r in self._records)

    def benchmarks(self) -> list[str]:
        return _sorted_unique(r.benchmark for r in self._records)

    def seeds(self) -> list[int]:
        return sorted({r.seed for r in self._records if r.seed is not None})

    def keys(self) -> list[tuple[str, str, str]]:
        """Sorted unique (benchmark, task, metric) triples present."""
        return sorted({r.key for r in self._records})

    def values(self) -> list[float]:
        return [r.value for r in self._records]

    def to_jsonl_records(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    @classmethod
    def from_jsonl_records(cls, rows: Iterable[dict[str, Any]]) -> "RecordSet":
        return cls(MetricRecord.from_dict(r) for r in rows)


def _sorted_unique(items: Iterable[str]) -> list[str]:
    return sorted(set(items))
