"""Variant registry and per-unit status for the optional "all variants" overview.

**Overview only — descriptive, not pre-registered.** Nothing in the official procedure
(EXPERIMENT_PROCEDURE §5.4–§5.8) reads what this module describes. The launcher uses it
only when ``variants=...`` is set, the analysis only with ``--variants-overview``.

Layout. The originals stay where they always were (``<run>/halueval/``, ``<run>/faitheval/``
...). Every variant lives in a sibling directory named exactly like its analysis
benchmark: ``<run>/halueval.constrained/``, ``<run>/faitheval.strict/`` and so on. The
procedure's tools all glob ``*/*/<bench>/...`` at a fixed depth, so they never see these
directories.

A *unit* is one (variant, task) pair: the smallest thing that is computed on its own and
can be skipped on resume. GPU units need a model; CPU units are re-scored offline from a
GPU unit's stored artifacts (their *source*) and never load one.

``python -m analysis.variants --root R`` prints the status of every unit under a root.

Stdlib only: the launcher (Eval_master ``.venv``) and every analysis entry point import it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

EVAL_MASTER = Path(__file__).resolve().parents[1]
HALUEVAL_EVALUATION = EVAL_MASTER / "HaluEval-reproduce" / "evaluation"
FAITHEVAL_REPO = EVAL_MASTER / "FaithEval-reproduce"

GPU = "gpu"
CPU = "cpu"

# A benchmark with no per-task split is one unit, under this task name.
WHOLE = "all"

HALUEVAL_TASKS = ("qa", "dialogue", "summarization")
FAITHEVAL_GEN_TASKS = ("unanswerable", "inconsistent", "counterfactual")
FAITHEVAL_PHRASE_TASKS = ("unanswerable", "inconsistent")
FAITHEVAL_MC_TASK = "counterfactual_mc"

# CPU rescoring exit code meaning "this source can never yield the variant".
EXIT_UNAVAILABLE = 3

OVERVIEW_LABEL = "overview - descriptive, not pre-registered"


@dataclass(frozen=True)
class Variant:
    """One registry row. ``name`` is both the sibling dir and the analysis benchmark."""

    name: str
    base: str
    kind: str
    tasks: tuple[str, ...]
    source: str | None = None
    rule: str | None = None
    note: str = ""

    @property
    def is_original(self) -> bool:
        return self.name == self.base


REGISTRY: dict[str, Variant] = {v.name: v for v in (
    Variant("harness", "harness", GPU, (WHOLE,), note="harness-eval (unchanged)"),
    Variant("ragtruth", "ragtruth", GPU, (WHOLE,), note="unchanged, no variants"),
    Variant("truthfulqa", "truthfulqa", GPU, (WHOLE,), note="unchanged, no variants"),
    Variant("halueval", "halueval", GPU, HALUEVAL_TASKS, note="evaluate.py generate"),
    Variant("halueval.constrained", "halueval", GPU, HALUEVAL_TASKS,
            note="evaluate.py constrained / both"),
    Variant("halueval.parsed", "halueval", CPU, HALUEVAL_TASKS, source="halueval", rule="parsed",
            note="score_results.py --emit-variant parsed"),
    Variant("halueval.lenient", "halueval", CPU, HALUEVAL_TASKS, source="halueval", rule="lenient",
            note="score_results.py --emit-variant lenient (needs raw_judgement)"),
    Variant("halueval.decontam", "halueval", CPU, HALUEVAL_TASKS, source="halueval", rule="decontam",
            note="score_results.py --exclude (tasks with an exclusion list)"),
    Variant("halueval.constrained_decontam", "halueval", CPU, HALUEVAL_TASKS,
            source="halueval.constrained", rule="decontam",
            note="score_results.py --exclude (tasks with an exclusion list)"),
    Variant("faitheval", "faitheval", GPU, FAITHEVAL_GEN_TASKS, note="run_eval.py (unchanged)"),
    Variant("faitheval.mc", "faitheval", GPU, (FAITHEVAL_MC_TASK,),
            note="run_eval.py --task counterfactual_mc"),
    Variant("faitheval.strict", "faitheval", CPU, FAITHEVAL_PHRASE_TASKS, source="faitheval",
            rule="strict", note="rescore.py --rule strict"),
    Variant("faitheval.wordmatch", "faitheval", CPU, FAITHEVAL_PHRASE_TASKS, source="faitheval",
            rule="wordmatch", note="rescore.py --rule wordmatch"),
    Variant("faitheval.contains", "faitheval", CPU, ("counterfactual",), source="faitheval",
            rule="contains", note="rescore.py --rule contains (length-stratified)"),
)}

# Suffixes a HaluEval summary carries when it is not the original protocol's.
_HALUEVAL_VARIANT_SUFFIXES = ("_constrained_summary.json", "_decontam_summary.json",
                              "_parsed_summary.json", "_lenient_summary.json")


def run_label(model_path: str) -> str:
    """HaluEval's filename label for a ``--model-path``; a copy of ``evaluate._run_label``.

    Copied rather than imported so this module stays importable without HaluEval on the
    path; ``test_variants`` pins the two against each other.
    """
    label = model_path.replace("/", "_").replace("\\", "_")
    for reserved in ':*?"<>|':
        label = label.replace(reserved, "_")
    return label


# =====================================================================================
# Context: what a status check is compared against
# =====================================================================================

@dataclass
class Context:
    """Everything :func:`unit_status` needs besides the run dir.

    ``expected`` maps a base benchmark to the settings a launch would use (as recorded in
    a summary: ``batch_size``, ``seed`` ...). An empty map checks presence and source
    hashes only, which is what a census over an existing root does.
    """

    tasks: dict[str, tuple[str, ...]] = field(default_factory=dict)
    expected: dict[str, dict] = field(default_factory=dict)
    halueval_label: str | None = None
    ragtruth_stage: str = "all"
    exclusion_dir: Path = EVAL_MASTER / "decontamination" / "ragtruth"
    faitheval_data_dir: Path | None = None
    faitheval_split: str = "test"
    strict_provenance: bool = False

    def tasks_of(self, variant: Variant) -> tuple[str, ...]:
        """The tasks ``variant`` has under this context (config task lists, exclusion lists)."""
        if variant.base == "faitheval":
            configured = self.tasks.get("faitheval", FAITHEVAL_GEN_TASKS)
            if variant.name == "faitheval.mc":
                return variant.tasks
            tasks = tuple(t for t in variant.tasks if t in configured)
        elif variant.base == "halueval":
            configured = self.tasks.get("halueval", HALUEVAL_TASKS)
            tasks = tuple(t for t in configured)
        else:
            return variant.tasks
        if variant.rule == "decontam":
            tasks = tuple(t for t in tasks if self.exclusion_list(t).is_file())
        return tasks

    def exclusion_list(self, task: str) -> Path:
        return Path(self.exclusion_dir) / f"halueval__{task}.json"

    def faitheval_data(self) -> Path:
        if self.faitheval_data_dir is not None:
            return Path(self.faitheval_data_dir)
        env = os.environ.get("FAITHEVAL_DATA_DIR")
        return Path(env) if env else FAITHEVAL_REPO / "data" / "faitheval"


# =====================================================================================
# Status
# =====================================================================================

DONE = "done"
MISSING = "missing"
STALE = "stale"
UNAVAILABLE = "unavailable"
UNVERIFIED = "unverified"


@dataclass(frozen=True)
class Status:
    state: str
    reason: str = ""

    def __str__(self) -> str:
        return f"{self.state}({self.reason})" if self.reason else self.state

    @property
    def counts_as_done(self) -> bool:
        return self.state in (DONE, UNVERIFIED)

    @property
    def settled(self) -> bool:
        """Nothing left to compute: done, or never computable."""
        return self.state in (DONE, UNVERIFIED, UNAVAILABLE)


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    """sha256 of a file, cached per (path, size, mtime) for the life of the process."""
    st = Path(path).stat()
    key = (str(Path(path).resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        _SHA_CACHE[key] = digest.hexdigest()
    return _SHA_CACHE[key]


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _halueval_summaries(directory: Path, task: str, suffix: str) -> list[Path]:
    """``<task>_<label><suffix>`` files; ``suffix == "_summary.json"`` means the original."""
    out = []
    for path in sorted(Path(directory).glob(f"{task}_*{suffix}")):
        name = path.name
        if suffix == "_summary.json" and name.endswith(_HALUEVAL_VARIANT_SUFFIXES):
            continue
        if suffix == "_constrained_summary.json" and name.endswith("_constrained_decontam_summary.json"):
            continue
        out.append(path)
    return out


_HALUEVAL_MARKER_SUFFIX = {
    "halueval": "_summary.json",
    "halueval.constrained": "_constrained_summary.json",
    "halueval.parsed": "_parsed_summary.json",
    "halueval.lenient": "_lenient_summary.json",
    "halueval.decontam": "_decontam_summary.json",
    "halueval.constrained_decontam": "_constrained_decontam_summary.json",
}


def markers(run_dir: Path, variant: Variant, task: str, ctx: Context) -> list[Path]:
    """The files whose presence means the unit finished (all of them must exist).

    For HaluEval without a known label this is whatever label is on disk; with one, it is
    exactly that label's file.
    """
    d = Path(run_dir) / variant.name
    if variant.name == "harness":
        # Written in this order, so the second one existing means both do.
        return [d / "summary.json", d / "lm_eval_results.json"]
    if variant.name == "ragtruth":
        return [d / ("generation_summary.json" if ctx.ragtruth_stage == "generate" else "summary.json")]
    if variant.name == "truthfulqa":
        return [d / "summary.csv"]
    if variant.base == "faitheval":
        return [d / f"{task}_summary.json"]
    suffix = _HALUEVAL_MARKER_SUFFIX[variant.name]
    if ctx.halueval_label is not None:
        return [d / f"{task}_{ctx.halueval_label}{suffix}"]
    found = _halueval_summaries(d, task, suffix)
    return found[:1] if found else [d / f"{task}_<label>{suffix}"]


def source_file(run_dir: Path, variant: Variant, task: str, ctx: Context) -> Path | None:
    """The artifact a CPU unit is derived from, if its source unit finished."""
    source = REGISTRY[variant.source]
    src_status = unit_status(run_dir, source, task, ctx)
    if not src_status.counts_as_done:
        return None
    marker = markers(run_dir, source, task, ctx)[0]
    if variant.base == "faitheval":
        path = marker.with_name(f"{task}_predictions.jsonl")
    else:
        path = marker.with_name(marker.name[: -len("_summary.json")] + "_results.json")
    return path if path.is_file() else None


def _has_raw_judgement(path: Path) -> bool:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                return "raw_judgement" in json.loads(line)
    return False


def _row_count(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _faitheval_dataset_file(task: str, split: str, ctx: Context) -> Path:
    slug = {"counterfactual_mc": "counterfactual"}.get(task, task)
    return ctx.faitheval_data() / f"FaithEval-{slug}-v1.0" / f"{split}.jsonl"


def _same(a, b) -> bool:
    return a == b or (a is not None and b is not None and str(a) == str(b))


def _gpu_provenance(variant: Variant, task: str, summary: dict, ctx: Context) -> Status:
    """Compare a finished GPU unit's recorded settings with the launch's."""
    expected = ctx.expected.get(variant.base) or {}
    if not expected:
        return Status(DONE)
    if variant.base == "halueval":
        want_label = ctx.halueval_label
        if want_label is not None and summary.get("model") not in (None, want_label):
            return Status(STALE, f"label {summary.get('model')!r}")
    unverified = ""
    for key, want in expected.items():
        if key == "num_samples":
            status = _num_samples_status(variant, task, summary, want, ctx)
            if status.state == STALE:
                return status
            if status.state == UNVERIFIED:
                unverified = status.reason
            continue
        if key not in summary:
            continue  # "where recorded": older summaries lack some fields
        if want is None and key == "max_new_tokens":
            continue  # None hands the choice to the benchmark; the summary holds the resolved value
        if not _same(summary[key], want):
            return Status(STALE, f"{key} {summary[key]!r} != {want!r}")
    if unverified:
        if ctx.strict_provenance:
            return Status(STALE, f"{unverified} unverifiable (strict_provenance)")
        return Status(UNVERIFIED, unverified)
    return Status(DONE)


def _num_samples_status(variant: Variant, task: str, summary: dict, want, ctx: Context) -> Status:
    for key in ("num_samples", "num_samples_requested", "limit"):
        if key in summary:
            if _same(summary[key], want):
                return Status(DONE)
            return Status(STALE, f"num_samples {summary[key]!r} != {want!r}")
    if variant.base != "faitheval" or "num_examples" not in summary:
        return Status(DONE)
    # Old FaithEval summaries: the row count is the only record of the cap.
    split = summary.get("split", ctx.faitheval_split)
    data = _faitheval_dataset_file(task, split, ctx)
    if not data.is_file():
        return Status(UNVERIFIED, "num_samples")
    rows = _row_count(data)
    expected_n = rows if want is None else min(int(want), rows)
    if int(summary["num_examples"]) != expected_n:
        return Status(STALE, f"num_examples {summary['num_examples']} != {expected_n}")
    return Status(DONE)


def unit_status(run_dir: Path, variant: Variant | str, task: str, ctx: Context | None = None) -> Status:
    """done / missing / stale(reason) / unavailable(reason) / unverified(field)."""
    ctx = ctx or Context()
    variant = REGISTRY[variant] if isinstance(variant, str) else variant
    marks = markers(run_dir, variant, task, ctx)
    present = all(m.is_file() for m in marks)

    if variant.kind == GPU:
        if not present:
            if variant.base == "halueval" and ctx.halueval_label is not None:
                other = _halueval_summaries(Path(run_dir) / variant.name, task,
                                            _HALUEVAL_MARKER_SUFFIX[variant.name])
                if other:
                    return Status(STALE, f"only another label on disk: {other[0].name}")
            return Status(MISSING)
        if variant.name in ("harness", "halueval", "halueval.constrained") or variant.base == "faitheval":
            summary = _read_json(marks[0])
            if summary is None:
                return Status(STALE, "unreadable summary")
            if variant.name == "faitheval" and summary.get("strict_match"):
                return Status(STALE, "strict_match run in the original dir")
            return _gpu_provenance(variant, task, summary, ctx)
        return Status(DONE)

    # CPU unit
    src = source_file(run_dir, variant, task, ctx)
    if src is None:
        return Status(MISSING, f"waiting on {variant.source}")
    if variant.rule == "lenient" and not _has_raw_judgement(src):
        return Status(UNAVAILABLE, "no raw_judgement")
    if not present:
        return Status(MISSING)
    summary = _read_json(marks[0])
    if summary is None:
        return Status(STALE, "unreadable summary")
    if summary.get("source_sha256") != sha256_file(src):
        return Status(STALE, "source changed" if summary.get("source_sha256") else "no source_sha256")
    if variant.rule == "decontam":
        lst = ctx.exclusion_list(task)
        if summary.get("exclusion_list_sha256") != sha256_file(lst):
            return Status(STALE, "exclusion list changed")
    return Status(DONE)


# =====================================================================================
# Selection and planning
# =====================================================================================

def resolve_variants(requested, bases) -> tuple[list[Variant], set[str]]:
    """``[all]`` or explicit names -> (registry rows whose base is in ``bases``, explicit names).

    A CPU variant pulls in its source, which is then not *explicit*: only the tasks its
    dependants need are planned (:func:`plan`).
    """
    requested = [str(r) for r in (requested or [])]
    bases = [str(b) for b in bases]
    unknown = [r for r in requested if r != "all" and r not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown variant(s) {unknown}; choose from {['all', *REGISTRY]}")
    if "all" in requested:
        names = [n for n, v in REGISTRY.items() if v.base in bases]
    else:
        names = [n for n in requested if REGISTRY[n].base in bases]
        skipped = [n for n in requested if REGISTRY[n].base not in bases]
        if skipped:
            print(f"!! variants {skipped}: base benchmark not in run=; skipped")
    explicit = set(names)
    chosen = list(dict.fromkeys(names))
    for name in list(chosen):
        src = REGISTRY[name].source
        while src is not None:
            if src not in chosen:
                chosen.append(src)
            src = REGISTRY[src].source
    order = list(REGISTRY)
    return [REGISTRY[n] for n in sorted(chosen, key=order.index)], explicit


@dataclass
class Unit:
    variant: Variant
    task: str
    status: Status

    @property
    def key(self) -> tuple[str, str]:
        return (self.variant.name, self.task)


def plan(run_dir: Path, variants: list[Variant], ctx: Context, *, explicit: set[str] | None = None) -> list[Unit]:
    """Every unit of ``variants`` in ``run_dir`` with its status.

    A source variant that was only pulled in by a CPU variant (not in ``explicit``) is
    restricted to the tasks its dependants need.
    """
    names = {v.name for v in variants}
    explicit = names if explicit is None else explicit
    needed: dict[str, set[str]] = {}
    for v in variants:
        if v.kind == CPU:
            src = v.source
            tasks = set(ctx.tasks_of(v))
            while src is not None:
                needed.setdefault(src, set()).update(tasks)
                src = REGISTRY[src].source
    units = []
    for v in variants:
        tasks = ctx.tasks_of(v)
        if v.name not in explicit:
            tasks = tuple(t for t in tasks if t in needed.get(v.name, set()))
        for task in tasks:
            units.append(Unit(v, task, unit_status(run_dir, v, task, ctx)))
    return units


def to_compute(unit: Unit, *, resume: bool, recompute_stale: bool) -> bool:
    """Whether a launch computes this unit."""
    state = unit.status.state
    if state == UNAVAILABLE:
        return False
    if unit.variant.kind == CPU:
        # Cheap and hash-checked: always re-derived when missing or stale; with resume
        # off, re-derived regardless.
        return (not resume) or state in (MISSING, STALE)
    if not resume:
        return True
    if state == MISSING:
        return True
    return state == STALE and recompute_stale


def format_matrix(units: list[Unit], title: str = "") -> str:
    """variant x task table of statuses."""
    variants = list(dict.fromkeys(u.variant.name for u in units))
    tasks = list(dict.fromkeys(u.task for u in units))
    cell = {u.key: str(u.status) for u in units}
    width_v = max([len("variant"), *map(len, variants)]) if variants else 7
    widths = {t: max(len(t), *(len(cell.get((v, t), "")) for v in variants)) for t in tasks}
    lines = [title] if title else []
    lines.append("  " + "variant".ljust(width_v) + "  " + "  ".join(t.ljust(widths[t]) for t in tasks))
    for v in variants:
        row = "  ".join(cell.get((v, t), "-").ljust(widths[t]) for t in tasks)
        lines.append("  " + v.ljust(width_v) + "  " + row)
    return "\n".join(lines)


# =====================================================================================
# CPU derivation commands
# =====================================================================================

def derive_command(run_dir: Path, unit: Unit, ctx: Context, *,
                   faitheval_python: str | None = None) -> tuple[list[str], Path] | None:
    """``(argv, cwd)`` that (re)derives a CPU unit, or None while its source is missing."""
    v, task = unit.variant, unit.task
    src = source_file(run_dir, v, task, ctx)
    if src is None:
        return None
    out = Path(run_dir) / v.name
    if v.base == "halueval":
        cmd = [sys.executable, str(HALUEVAL_EVALUATION / "score_results.py"), str(src)]
        if v.rule == "decontam":
            cmd += ["--exclude", str(ctx.exclusion_list(task))]
        else:
            cmd += ["--emit-variant", v.rule]
        return cmd + ["--emit-summary", str(out)], HALUEVAL_EVALUATION
    python = faitheval_python or sys.executable
    cmd = [python, str(FAITHEVAL_REPO / "src" / "rescore.py"), str(src), "--task", task,
           "--rule", v.rule, "--output-dir", str(out), "--data-dir", str(ctx.faitheval_data())]
    return cmd, FAITHEVAL_REPO


def derive_run_dir(run_dir: Path, units: list[Unit], ctx: Context, *, resume: bool = True,
                   dry_run: bool = False, faitheval_python: str | None = None) -> dict[str, list]:
    """Derive every CPU unit in ``units`` that needs it; statuses are re-read first.

    Returns ``{"ok": [...], "failed": [...], "unavailable": [...], "waiting": [...]}`` of
    unit keys. Writes only into the units' own sibling dirs.
    """
    result: dict[str, list] = {"ok": [], "failed": [], "unavailable": [], "waiting": [], "planned": []}
    for unit in units:
        if unit.variant.kind != CPU:
            continue
        unit = Unit(unit.variant, unit.task, unit_status(run_dir, unit.variant, unit.task, ctx))
        if unit.status.state == UNAVAILABLE:
            result["unavailable"].append(unit.key)
            continue
        if not to_compute(unit, resume=resume, recompute_stale=True):
            continue
        spec = derive_command(run_dir, unit, ctx, faitheval_python=faitheval_python)
        if spec is None:
            result["waiting"].append(unit.key)
            continue
        cmd, cwd = spec
        print(f"    [derive {unit.variant.name}/{unit.task}] {Path(cmd[1]).name} "
              f"{Path(cmd[2]).parent.name}/{Path(cmd[2]).name} {' '.join(cmd[3:5])}")
        if dry_run:
            result["planned"].append(unit.key)
            continue
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        if proc.returncode == EXIT_UNAVAILABLE:
            result["unavailable"].append(unit.key)
        elif proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:], file=sys.stderr)
            result["failed"].append(unit.key)
        else:
            result["ok"].append(unit.key)
    return result


# =====================================================================================
# Roots: census over <root>/<arm>/<run>/
# =====================================================================================

def run_dirs_under(root: Path) -> list[Path]:
    """``R/*/seed_*``, ``R/*/run_*`` and ``R/base/run_*`` (the latter is covered by the second)."""
    root = Path(root)
    found = []
    for pattern in ("*/seed_*", "*/run_*"):
        found.extend(p for p in sorted(root.glob(pattern)) if p.is_dir())
    return list(dict.fromkeys(found))


def present_bases(run_dir: Path) -> list[str]:
    return [b for b in ("faitheval", "truthfulqa", "halueval", "ragtruth", "harness")
            if (Path(run_dir) / b).is_dir()]


def _knobs(run_dir: Path, units: list[Unit], ctx: Context) -> dict[str, dict[str, set]]:
    """Recorded settings per variant (one set of values per knob)."""
    keys = ("batch_size", "sort_by_length", "max_new_tokens", "seed", "num_samples",
            "num_samples_requested", "limit", "dtype", "prompt_format", "split")
    out: dict[str, dict[str, set]] = {}
    for unit in units:
        if not unit.status.counts_as_done:
            continue
        data = _read_json(markers(run_dir, unit.variant, unit.task, ctx)[0]) or {}
        for k in keys:
            if k in data:
                out.setdefault(unit.variant.name, {}).setdefault(k, set()).add(json.dumps(data[k]))
    return out


def census(root: Path, variant_names, ctx: Context | None = None) -> int:
    ctx = ctx or Context()
    rows = []
    totals: dict[str, int] = {}
    knobs: dict[str, dict[str, set]] = {}
    for run_dir in run_dirs_under(root):
        variants, explicit = resolve_variants(variant_names, present_bases(run_dir))
        units = plan(run_dir, variants, ctx, explicit=explicit)
        for u in units:
            rows.append((run_dir.parent.name, run_dir.name, u.variant.name, u.task, str(u.status)))
            totals[u.status.state] = totals.get(u.status.state, 0) + 1
        for v, ks in _knobs(run_dir, units, ctx).items():
            for k, vals in ks.items():
                knobs.setdefault(v, {}).setdefault(k, set()).update(vals)
    print(f"# {OVERVIEW_LABEL}")
    print("arm\trun\tvariant\ttask\tstatus")
    for row in rows:
        print("\t".join(row))
    print("\n== totals: " + ", ".join(f"{k}={v}" for k, v in sorted(totals.items())))
    to_do: dict[tuple[str, str], int] = {}
    for _, _, variant, _, status in rows:
        state = status.split("(")[0]
        if state in (MISSING, STALE):
            kind = REGISTRY[variant].kind
            to_do[(kind, state)] = to_do.get((kind, state), 0) + 1
    for (kind, state), n in sorted(to_do.items()):
        print(f"   still to compute: {n} {kind} unit(s) {state}")
    print("\n== recorded knobs per variant (a knob with >1 value is not uniform)")
    for v in sorted(knobs):
        parts = [f"{k}={'|'.join(sorted(vals))}" for k, vals in sorted(knobs[v].items())]
        print(f"  {v}: " + "  ".join(parts))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Status of every overview unit under an eval root "
                                             f"({OVERVIEW_LABEL}).")
    ap.add_argument("--root", required=True, help="eval root holding <arm>/<run>/ dirs")
    ap.add_argument("--variants", default="all", help="'all' or comma-separated variant names")
    ap.add_argument("--faitheval-data-dir", default=None,
                    help="FaithEval JSONL root (default: $FAITHEVAL_DATA_DIR or the repo's data/)")
    args = ap.parse_args(argv)
    ctx = Context(faitheval_data_dir=Path(args.faitheval_data_dir) if args.faitheval_data_dir else None)
    return census(Path(args.root), args.variants.split(","), ctx)


if __name__ == "__main__":
    raise SystemExit(main())
