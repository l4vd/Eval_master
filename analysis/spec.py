"""Arm declaration + config: turn ``name=spec`` inputs into a parsed RecordSet.

An arm resolves (via :mod:`analysis.discover`) to a set of run dirs, each with an
inferred seed; :func:`build_records` parses them all into one long-form RecordSet and
records per-arm metadata (seeds, fixed-point-ness) that the comparison layer needs to
choose the statistical regime.

A *fixed point* is an arm with no seed variance — a single run dir whose seed is
unknown/``None`` (e.g. an untrained base evaluated once). It can also be forced with a
trailing ``!fixed`` marker on the spec, or cleared with ``!ensemble``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from analysis.discover import discover_arm
from analysis.model import RecordSet
from analysis.parse import PrimaryPredicate, _default_is_primary, parse_run_dir
from analysis.stats import DEFAULT_MC_METHOD


@dataclass
class ArmSpec:
    name: str
    spec: str
    force_fixed: bool | None = None  # None -> auto-detect; True/False -> forced


@dataclass
class ArmMeta:
    name: str
    is_fixed_point: bool
    seeds: list[int | None]
    run_dirs: list[str]

    @property
    def real_seeds(self) -> set[int]:
        return {s for s in self.seeds if s is not None}


@dataclass
class AnalysisConfig:
    arms: list[ArmSpec]
    out: str = "outputs/analysis"
    reference: str | None = None
    benchmarks: list[str] | None = None   # include-list (None = all present)
    exclude: list[str] | None = None
    compare: bool = True
    plot: bool = True
    require_matched: bool = True
    primary_map: dict | None = None
    rng_seed: int = 0
    #: Multiplicity procedure across the paired comparison family; see
    #: :func:`analysis.stats.adjust_pvalues`. "none" disables it explicitly.
    mc_method: str = DEFAULT_MC_METHOD
    #: Which protocol trees to write: "original" (``out/``), "modified" (``out/modified/``)
    #: or "both". See :func:`analysis.model.protocol_of`.
    protocol: str = "both"


@dataclass
class BuildResult:
    records: RecordSet
    arm_meta: dict[str, ArmMeta] = field(default_factory=dict)


def parse_arm_arg(arg: str) -> ArmSpec:
    """Parse a ``NAME=SPEC`` CLI argument (with optional ``!fixed`` / ``!ensemble``)."""
    if "=" not in arg:
        raise ValueError(f"Arm must be NAME=SPEC, got: {arg!r}")
    name, spec = arg.split("=", 1)
    force_fixed: bool | None = None
    for marker, val in (("!fixed", True), ("!ensemble", False)):
        if spec.endswith(marker):
            force_fixed = val
            spec = spec[: -len(marker)].rstrip()
    return ArmSpec(name=name.strip(), spec=spec.strip(), force_fixed=force_fixed)


def build_primary_predicate(override: dict | None) -> PrimaryPredicate:
    """Return an is_primary predicate, optionally overridden per benchmark.

    Override shape: ``{benchmark: ["metric", "task:metric", ...]}``. A benchmark listed
    in the override uses ONLY its listed entries; benchmarks absent from the override
    keep the built-in default. Entry ``"metric"`` matches that metric on any task;
    ``"task:metric"`` matches exactly.
    """
    if not override:
        return _default_is_primary

    parsed: dict[str, set[tuple[str | None, str]]] = {}
    for bench, entries in override.items():
        acc: set[tuple[str | None, str]] = set()
        for e in entries:
            if ":" in e:
                task, metric = e.split(":", 1)
                acc.add((task, metric))
            else:
                acc.add((None, e))
        parsed[bench] = acc

    def predicate(benchmark: str, task: str, metric: str) -> bool:
        if benchmark not in parsed:
            return _default_is_primary(benchmark, task, metric)
        for t, m in parsed[benchmark]:
            if m == metric and (t is None or t == task):
                return True
        return False

    return predicate


def build_records(config: AnalysisConfig) -> BuildResult:
    """Discover + parse every arm into one RecordSet, plus per-arm metadata.

    Repeating ``--arm NAME=SPEC`` with the same NAME merges the run dirs of every spec
    into one arm — how an arm's original-protocol root and its modified-protocol root
    (``$EVAL_ROOT/<arm>`` + ``$EVAL_ROOT_MOD/<arm>``) are read together. The merged arm
    is a fixed point only if every spec is one, and two of its run dirs reporting the
    same (seed, benchmark, task, metric) is an error: the roots overlap.
    """
    is_primary = build_primary_predicate(config.primary_map)
    all_records = []
    arm_meta: dict[str, ArmMeta] = {}

    specs_by_name: dict[str, list[ArmSpec]] = {}
    for arm in config.arms:
        specs_by_name.setdefault(arm.name, []).append(arm)

    for name, specs in specs_by_name.items():
        all_pairs: list[tuple[Path, int | None]] = []
        fixed: list[bool] = []
        arm_records = []
        for arm in specs:
            pairs = discover_arm(arm.spec)
            if not pairs:
                raise FileNotFoundError(
                    f"Arm '{arm.name}': no run dirs matched spec {arm.spec!r}"
                )
            for run_dir, seed in pairs:
                arm_records.extend(
                    parse_run_dir(
                        Path(run_dir), arm.name, seed,
                        is_primary=is_primary, benchmarks=config.benchmarks,
                    )
                )
            all_pairs.extend(pairs)
            fixed.append(_resolve_fixed(arm, pairs))
        if len(specs) > 1:
            _check_no_duplicate_metrics(name, arm_records)
            seeds = list(dict.fromkeys(seed for _, seed in all_pairs))
        else:
            seeds = [seed for _, seed in all_pairs]
        all_records.extend(arm_records)
        arm_meta[name] = ArmMeta(
            name=name, is_fixed_point=all(fixed), seeds=seeds,
            run_dirs=[str(rd) for rd, _ in all_pairs],
        )

    records = RecordSet(all_records).include_benchmarks(config.benchmarks, config.exclude)
    return BuildResult(records=records, arm_meta=arm_meta)


def _check_no_duplicate_metrics(arm: str, records) -> None:
    seen: set[tuple] = set()
    for r in records:
        key = (r.seed, r.benchmark, r.task, r.metric)
        if key in seen:
            raise ValueError(
                f"Arm '{arm}': two of its run dirs report {r.benchmark}/{r.task}/{r.metric} "
                f"for seed {r.seed}. A merged arm's specs must not overlap (point the second "
                "--arm at the modified-protocol root only)."
            )
        seen.add(key)


def _resolve_fixed(arm: ArmSpec, pairs: list[tuple[Path, int | None]]) -> bool:
    if arm.force_fixed is not None:
        return arm.force_fixed
    # Auto: a single run dir with no recoverable seed is a fixed point.
    return len(pairs) == 1 and pairs[0][1] is None
