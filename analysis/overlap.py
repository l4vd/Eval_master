"""Train <-> eval text-overlap checker: which eval rows share text with the training data.

The maintained form of the read-only probe behind ``SP-DPO-Base/KNOWN_ISSUES.md`` §5. It
indexes a training corpus once and checks every Eval_master benchmark against it, row by
row, then writes a report and (optionally) the exclusion lists the eval-side
decontamination consumes (``HaluEval-reproduce/evaluation/score_results.py --exclude``).

Matching
--------
* **Normalization:** NFKC -> casefold -> ``[\\W_]+`` -> single space. Tokens become ints via
  one vocabulary dict, and an n-gram's key is ``hash(tuple(ints))`` — integer tuples hash
  identically in every process, so reports are reproducible.
* **Tiers, best tier per row** (a row is every field of one eval item):
  ``exact`` (normalized field == a normalized training text) > ``near_duplicate`` (at
  least ``--near-dup`` of the field's distinct ``--ngram``-grams occur in training text)
  > ``partial`` (> 0) > ``none``. Texts shorter than n tokens are checked for ``exact``
  only.
* **Boilerplate filter:** n-grams that occur in more than ``--df-max`` distinct training
  groups (sources) are ignored; how many were ignored is reported.
* **Exposure:** how strongly training saw a group — ``train`` > ``val`` > ``test`` >
  ``release_only`` (RAG-Truth: the selected splits a source has pairs in; ``release_only``
  when it has none). A row's exposure is the strongest one among its exact-match groups,
  or else among its top ``--exposure-topk`` groups by shared n-grams.

Exclusion policy: a row is excluded when its tier is ``exact`` or ``near_duplicate`` AND
its exposure is ``train`` or ``val``. Exposed rows that only the test side or the release
saw stay in, as the "unseen-exposed" control.

Stdlib only; ``--train-run-config`` imports PyYAML lazily.

Usage (from Eval_master/)::

    python -m analysis.overlap \\
        --harness-samples outputs/eval/<arm>/seed_42/harness/samples.jsonl --write-exclusions

A missing eval file is reported under ``not_checked`` with its reason — never as zero
overlap.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]  # Eval_master/
DEFAULT_RAGTRUTH_DATA = ROOT.parents[1] / "SP-DPO-Base" / "data" / "ragtruth"

SCHEMA_VERSION = 1
TIERS = ("exact", "near_duplicate", "partial", "none")
_TIER_RANK = {t: i for i, t in enumerate(TIERS)}
EXPOSURES = ("train", "val", "test", "release_only")
_EXPOSURE_RANK = {e: i for i, e in enumerate(EXPOSURES)}
EXCLUDE_TIERS = ("exact", "near_duplicate")
EXCLUDE_EXPOSURES = ("train", "val")
POLICY = (
    "exclude rows whose best tier is exact or near_duplicate and whose exposure is train "
    "or val; exact/near_duplicate rows with test or release_only exposure are kept as the "
    "unseen-exposed control; partial rows are kept"
)

BENCHMARKS = ("halueval", "faitheval", "truthfulqa", "harness", "ragtruth_benchmark")
HALUEVAL_TASKS = ("qa", "dialogue", "summarization", "general")
FAITHEVAL_TASKS = ("unanswerable", "inconsistent", "counterfactual")
RAGTRUTH_PAIR_FIELDS = ("context", "prompt", "chosen", "rejected")  # author_prompt embeds context
_RAGTRUTH_SPLITS = ("train", "val", "test")

_NON_WORD = re.compile(r"[\W_]+")
_BOILERPLATE = -1  # sentinel value in TrainIndex.grams: seen in more than df_max groups


@dataclass(frozen=True)
class Params:
    ngram: int = 13
    near_dup: float = 0.5
    df_max: int = 10
    exposure_topk: int = 3
    max_examples: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "ngram": self.ngram,
            "near_dup": self.near_dup,
            "df_max": self.df_max,
            "exposure_topk": self.exposure_topk,
            "normalization": "NFKC -> casefold -> [\\W_]+ -> ' '",
            "containment": "shared distinct n-grams / distinct n-grams of the eval text",
        }


# =================================================================================
# Normalization and the training index
# =================================================================================

def normalize(text: str) -> str:
    """NFKC -> casefold -> every run of non-word characters (and ``_``) to one space."""
    return _NON_WORD.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def as_text(value: Any) -> str | None:
    """A field value as text: strings as-is, containers as JSON, scalars via ``str``."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _digest(normalized: str) -> bytes:
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).digest()


class _Vocab:
    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def encode(self, tokens: list[str]) -> list[int]:
        ids = self._ids
        return [ids.setdefault(t, len(ids)) for t in tokens]


@dataclass
class TextMatch:
    tier: str
    containment: float
    exact_groups: frozenset[int]
    shared_by_group: dict[int, int]
    span: str | None


class TrainIndex:
    """Exact-text digests and n-gram -> group postings over one training corpus."""

    def __init__(self, params: Params) -> None:
        self.params = params
        self.vocab = _Vocab()
        self.grams: dict[int, int | set[int]] = {}
        self.exact: dict[bytes, set[int]] = {}
        self.group_names: list[str] = []
        self._group_ids: dict[str, int] = {}
        self._seen: set[tuple[int, bytes]] = set()
        self.n_texts = 0

    def group_id(self, name: str) -> int:
        gid = self._group_ids.get(name)
        if gid is None:
            gid = self._group_ids[name] = len(self.group_names)
            self.group_names.append(name)
        return gid

    def add(self, group: str, text: str) -> None:
        norm = normalize(text)
        if not norm:
            return
        gid = self.group_id(group)
        digest = _digest(norm)
        if (gid, digest) in self._seen:  # the same context recurs once per selected pair
            return
        self._seen.add((gid, digest))
        self.n_texts += 1
        self.exact.setdefault(digest, set()).add(gid)

        n = self.params.ngram
        ids = self.vocab.encode(norm.split())
        if len(ids) < n:
            return
        grams, df_max = self.grams, self.params.df_max
        for key in {hash(tuple(ids[i:i + n])) for i in range(len(ids) - n + 1)}:
            cur = grams.get(key)
            if cur is None:
                grams[key] = gid
            elif cur.__class__ is int:
                if cur != gid and cur != _BOILERPLATE:
                    grams[key] = {cur, gid} if df_max >= 2 else _BOILERPLATE
            else:
                cur.add(gid)
                if len(cur) > df_max:
                    grams[key] = _BOILERPLATE

    @property
    def n_boilerplate(self) -> int:
        return sum(1 for v in self.grams.values() if v.__class__ is int and v == _BOILERPLATE)

    def match(self, text: str, *, span_tokens: int = 40) -> TextMatch | None:
        """Tier one eval text against the index; ``None`` for an empty text."""
        norm = normalize(text)
        if not norm:
            return None
        exact = self.exact.get(_digest(norm))
        if exact:
            return TextMatch("exact", 1.0, frozenset(exact), {}, _clip(text))
        n = self.params.ngram
        tokens = norm.split()
        if len(tokens) < n:
            return TextMatch("none", 0.0, frozenset(), {}, None)

        ids = self.vocab.encode(tokens)
        positions: dict[int, int] = {}
        for i in range(len(ids) - n + 1):
            positions.setdefault(hash(tuple(ids[i:i + n])), i)
        grams = self.grams
        by_group: dict[int, int] = {}
        starts: set[int] = set()
        for key, pos in positions.items():
            posting = grams.get(key)
            if posting is None or (posting.__class__ is int and posting == _BOILERPLATE):
                continue
            starts.add(pos)
            if posting.__class__ is int:
                by_group[posting] = by_group.get(posting, 0) + 1
            else:
                for g in posting:
                    by_group[g] = by_group.get(g, 0) + 1
        if not starts:
            return TextMatch("none", 0.0, frozenset(), {}, None)

        containment = len(starts) / len(positions)
        tier = "near_duplicate" if containment >= self.params.near_dup else "partial"
        first = min(starts)
        last = first
        while last + 1 in starts and last + 1 - first + n < span_tokens:
            last += 1
        span = " ".join(tokens[first:last + n])
        return TextMatch(tier, containment, frozenset(), by_group, span)


def _clip(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


# =================================================================================
# Training inputs
# =================================================================================

@dataclass
class TrainSide:
    name: str
    kind: str
    index: TrainIndex
    exposure_by_group: list[str]
    inputs: list[dict[str, Any]]
    pools: dict[str, dict[str, int]] = field(default_factory=dict)  # "summary/train" -> sid -> pairs
    notes: list[str] = field(default_factory=list)

    def exposure(self, gid: int) -> str:
        return self.exposure_by_group[gid]

    def exposure_of_source(self, source_id: str) -> str | None:
        gid = self.index._group_ids.get(source_id)
        return None if gid is None else self.exposure_by_group[gid]


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _display_path(path: Path) -> str:
    """Path relative to Eval_master when possible, so committed reports carry no home dir."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()
    except ValueError:  # another drive on Windows
        return Path(path).resolve().as_posix()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": _display_path(path), "sha256": sha256_file(path)}


def is_ragtruth_layout(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and any(
        (task / f"{split}.jsonl").is_file()
        for task in path.iterdir() if task.is_dir()
        for split in _RAGTRUTH_SPLITS
    )


def load_ragtruth_train(name: str, root: Path, params: Params) -> TrainSide:
    """SP-DPO-Base ``data/ragtruth``: every selected pair plus all release sources."""
    root = Path(root)
    index = TrainIndex(params)
    pools: dict[str, dict[str, int]] = {}
    exposure_by_source: dict[str, str] = {}
    inputs: list[dict[str, Any]] = []
    notes: list[str] = []

    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "raw"):
        for split in _RAGTRUTH_SPLITS:
            path = task_dir / f"{split}.jsonl"
            if not path.is_file():
                continue
            inputs.append(_file_record(path))
            pool = pools.setdefault(f"{task_dir.name}/{split}", {})
            for row in _read_jsonl(path):
                sid = str(row["source_id"])
                pool[sid] = pool.get(sid, 0) + 1
                current = exposure_by_source.get(sid)
                if current is None or _EXPOSURE_RANK[split] < _EXPOSURE_RANK[current]:
                    exposure_by_source[sid] = split
                for fname in RAGTRUTH_PAIR_FIELDS:
                    text = as_text(row.get(fname))
                    if text:
                        index.add(sid, text)

    raw = root / "raw" / "source_info.jsonl"
    if raw.is_file():
        inputs.append(_file_record(raw))
        for row in _read_jsonl(raw):
            sid = str(row["source_id"])
            exposure_by_source.setdefault(sid, "release_only")
            text = as_text(row.get("source_info"))
            if text:
                index.add(sid, text)
    else:
        notes.append(f"{_display_path(raw)} missing: sources without a selected pair are not indexed")

    exposure = [exposure_by_source[g] for g in index.group_names]
    return TrainSide(name, "ragtruth", index, exposure, inputs, pools, notes)


_JSONL_SPEC = re.compile(r"^jsonl:(?P<path>.+?\.jsonl?):(?P<fields>[^:]+)(?::(?P<group>[^:]+))?$")


def load_jsonl_train(name: str, spec: str, params: Params) -> TrainSide:
    """``jsonl:PATH:field,field[:group_field]`` — a profiled train.jsonl or an added corpus.

    PATH may be a glob. Every row counts as ``train`` exposure; the group is the value of
    ``group_field``, or ``<file>:<line>`` without one.
    """
    m = _JSONL_SPEC.match(spec)
    if not m:
        raise ValueError(f"--train {name}: expected jsonl:PATH:field,field[:group_field], got {spec!r}")
    paths = sorted(Path(p) for p in glob.glob(m["path"])) or [Path(m["path"])]
    fields = [f for f in m["fields"].split(",") if f]
    group_field = m["group"]
    index = TrainIndex(params)
    inputs: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"--train {name}: {path} not found")
        inputs.append(_file_record(path))
        for i, row in enumerate(_read_jsonl(path)):
            group = str(row.get(group_field)) if group_field else f"{path.name}:{i}"
            for fname in fields:
                text = as_text(row.get(fname))
                if text:
                    index.add(group, text)
    exposure = ["train"] * len(index.group_names)
    return TrainSide(name, "jsonl", index, exposure, inputs,
                     notes=[f"fields {fields}, group {group_field or '<file>:<line>'}; all rows count as train"])


def load_train(name: str, spec: str, params: Params) -> TrainSide:
    if spec.startswith("jsonl:"):
        return load_jsonl_train(name, spec, params)
    path = Path(spec)
    if is_ragtruth_layout(path):
        return load_ragtruth_train(name, path, params)
    raise ValueError(
        f"--train {name}={spec}: not a jsonl:PATH:fields spec and not a data/ragtruth layout "
        "(<task>/{train,val,test}.jsonl + raw/source_info.jsonl)"
    )


def read_run_config(path: Path) -> dict[str, Any]:
    """A training run's ``.hydra/config.yaml``: its data, and identity overlaps it implies.

    Training on the HaluEval or TruthfulQA presets makes the matching benchmarks score rows
    the arm trained on. That is flagged from the config alone (the HF mirror is assumed to
    hold the same items as the local benchmark files); it is not text-measured here.
    """
    record: dict[str, Any] = {"path": _display_path(path)}
    try:
        import yaml
    except ImportError:
        record["error"] = "PyYAML not installed; run config not read"
        return record
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data = cfg.get("data") or {}
    record.update({
        "data_name": data.get("name"),
        "source_name": data.get("source_name"),
        "sources": list(data.get("sources") or []),
        "data_dir": data.get("data_dir"),
        "hf_path": data.get("hf_path"),
    })
    identity: list[dict[str, Any]] = []
    for via in [data.get("name"), data.get("source_name"), *(data.get("sources") or [])]:
        if not via:
            continue
        via = str(via)
        if via.startswith("halueval"):
            task = via[len("halueval_"):] if via.startswith("halueval_") else ""
            tasks = [task] if task in HALUEVAL_TASKS else list(HALUEVAL_TASKS[:3])
            identity.append({"benchmark": "halueval", "tasks": tasks, "via": via})
        elif via.startswith("truthfulqa"):
            identity.append({"benchmark": "truthfulqa", "via": via})
            identity.append({"benchmark": "harness", "via": via})
    record["identity_overlaps"] = identity
    return record


# =================================================================================
# Eval inputs: each yields (row_id, [(field, text), ...])
# =================================================================================

Row = tuple[Any, list[tuple[str, str]]]


class NotChecked(Exception):
    """An eval input that could not be read; reported, never counted as zero overlap."""


def _row_fields(row: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    """Every leaf of a (possibly nested) row as ``(field, text)``; list items share a name."""
    out: list[tuple[str, str]] = []
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.extend(_row_fields(value, f"{name}."))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    out.extend(_row_fields(item, f"{name}[]."))
                else:
                    text = as_text(item)
                    if text:
                        out.append((f"{name}[]", text))
        else:
            text = as_text(value)
            if text:
                out.append((name, text))
    return out


def halueval_rows(eval_root: Path, task: str) -> tuple[Path, list[Row]]:
    """Row id = line order = the ``index`` evaluate.py writes into results."""
    path = eval_root / "HaluEval-reproduce" / "data" / f"{task}_data.json"
    if not path.is_file():
        raise NotChecked(f"{_display_path(path)} not found")
    return path, [(i, _row_fields(row)) for i, row in enumerate(_read_jsonl(path))]


def _faitheval_dataset_slug(config_path: Path) -> str:
    match = re.search(r"^dataset_name:\s*['\"]?([^'\"\s#]+)", config_path.read_text(encoding="utf-8"), re.M)
    if not match:
        raise NotChecked(f"{_display_path(config_path)} has no dataset_name")
    return match.group(1).rstrip("/").split("/")[-1]


def faitheval_rows(eval_root: Path, task: str, split: str = "test") -> tuple[Path, list[Row]]:
    config = eval_root / "FaithEval-reproduce" / "configs" / f"{task}.yaml"
    if not config.is_file():
        raise NotChecked(f"{_display_path(config)} not found")
    data_dir = Path(os.environ.get("FAITHEVAL_DATA_DIR") or eval_root / "FaithEval-reproduce" / "data" / "faitheval")
    path = data_dir / _faitheval_dataset_slug(config) / f"{split}.jsonl"
    if not path.is_file():
        raise NotChecked(f"{_display_path(path)} not found (prepare it with scripts/prepare_datasets.py)")
    return path, [(i, _row_fields(row)) for i, row in enumerate(_read_jsonl(path))]


def truthfulqa_rows(eval_root: Path) -> tuple[Path, list[Row]]:
    path = eval_root / "TruthfulQA-reproduce" / "TruthfulQA.csv"
    if not path.is_file():
        raise NotChecked(f"{_display_path(path)} not found")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [(i, _row_fields(r)) for i, r in enumerate(csv.DictReader(f))]
    return path, rows


def harness_rows(samples: Path) -> tuple[Path, list[Row]]:
    """lm_eval ``samples.jsonl``: one row per ``doc_id``, its docs from every task merged."""
    if not samples.is_file():
        raise NotChecked(f"{_display_path(samples)} not found")
    docs: dict[Any, list[tuple[str, str]]] = {}
    for rec in _read_jsonl(samples):
        doc = rec.get("doc")
        if isinstance(doc, str):
            try:
                doc = json.loads(doc)
            except json.JSONDecodeError:
                doc = {"doc": doc}
        doc_id = rec.get("doc_id")
        doc_id = int(doc_id) if str(doc_id).isdigit() else doc_id
        fields = docs.setdefault(doc_id, [])
        for pair in _row_fields(doc if isinstance(doc, dict) else {"doc": doc}):
            if pair not in fields:
                fields.append(pair)
    return samples, sorted(docs.items(), key=lambda kv: (str(type(kv[0])), kv[0]))


def ragtruth_sources(dataset: Path, split: str | None) -> tuple[list[dict[str, Any]], list[Path]]:
    info = dataset / "source_info.jsonl"
    if not info.is_file():
        raise NotChecked(f"{_display_path(info)} not found")
    rows = list(_read_jsonl(info))
    used = [info]
    if split and split != "all":
        responses = dataset / "response.jsonl"
        if not responses.is_file():
            raise NotChecked(f"{_display_path(responses)} not found; cannot select split {split!r}")
        keep = {str(r["source_id"]) for r in _read_jsonl(responses) if r.get("split") == split}
        rows = [r for r in rows if str(r["source_id"]) in keep]
        used.append(responses)
    return rows, used


# =================================================================================
# Row checking and summaries
# =================================================================================

@dataclass
class RowResult:
    row_id: Any
    tier: str
    containment: float
    field: str | None
    exposure: str | None
    source_ids: list[str]
    field_tiers: dict[str, tuple[str, float]]
    span: str | None


def check_row(row_id: Any, fields: list[tuple[str, str]], train: TrainSide) -> RowResult:
    index, params = train.index, train.index.params
    best: tuple[str, TextMatch] | None = None
    exact_groups: set[int] = set()
    by_group: dict[int, int] = {}
    field_tiers: dict[str, tuple[str, float]] = {}
    for name, text in fields:
        m = index.match(text)
        if m is None:
            continue
        prev = field_tiers.get(name)
        if prev is None or (_TIER_RANK[m.tier], -m.containment) < (_TIER_RANK[prev[0]], -prev[1]):
            field_tiers[name] = (m.tier, m.containment)
        exact_groups |= m.exact_groups
        for g, c in m.shared_by_group.items():
            by_group[g] = by_group.get(g, 0) + c
        if best is None or (_TIER_RANK[m.tier], -m.containment) < (_TIER_RANK[best[1].tier], -best[1].containment):
            best = (name, m)

    if best is None or best[1].tier == "none":
        return RowResult(row_id, "none", 0.0, None, None, [], field_tiers, None)

    if exact_groups:
        groups = sorted(exact_groups)
    else:
        ranked = sorted(by_group.items(), key=lambda kv: (-kv[1], kv[0]))
        groups = [g for g, _ in ranked[: params.exposure_topk]]
    exposure = min((train.exposure(g) for g in groups), key=_EXPOSURE_RANK.__getitem__)
    names = [index.group_names[g] for g in groups]
    return RowResult(row_id, best[1].tier, best[1].containment, best[0], exposure, names,
                     field_tiers, best[1].span)


def is_excluded(r: RowResult) -> bool:
    return r.tier in EXCLUDE_TIERS and r.exposure in EXCLUDE_EXPOSURES


def is_unseen_exposed(r: RowResult) -> bool:
    return r.tier in EXCLUDE_TIERS and r.exposure not in EXCLUDE_EXPOSURES


def summarize(results: list[RowResult], train: TrainSide, params: Params) -> dict[str, Any]:
    tiers = {t: 0 for t in TIERS}
    matrix = {e: {t: 0 for t in TIERS[:3]} for e in EXPOSURES}
    fields: dict[str, dict[str, Any]] = {}
    for r in results:
        tiers[r.tier] += 1
        if r.exposure is not None:
            matrix[r.exposure][r.tier] += 1
        for name, (tier, containment) in r.field_tiers.items():
            if tier == "none":
                continue
            f = fields.setdefault(name, {"exact": 0, "near_duplicate": 0, "partial": 0, "max_containment": 0.0})
            f[tier] += 1
            f["max_containment"] = round(max(f["max_containment"], containment), 4)

    partial = sorted(r.containment for r in results if r.tier == "partial")
    excluded = [r for r in results if is_excluded(r)]
    control = [r for r in results if is_unseen_exposed(r)]
    summary: dict[str, Any] = {
        "n_rows": len(results),
        "tiers": tiers,
        "exposure_by_tier": matrix,
        "fields": dict(sorted(fields.items())),
        "partial_containment": {
            "median": round(statistics.median(partial), 4) if partial else None,
            "max": round(partial[-1], 4) if partial else None,
            "n_ge_0.10": sum(1 for c in partial if c >= 0.10),
            "n_ge_0.20": sum(1 for c in partial if c >= 0.20),
        },
        "excluded": len(excluded),
        "unseen_exposed": len(control),
        "kept": len(results) - len(excluded),
        "clean": len(results) - len(excluded) - len(control),
    }
    if train.pools and (excluded or control):
        summary["affected_train_data"] = affected_train_data(results, train)
    summary["examples"] = _examples(results, params.max_examples)
    return summary


def affected_train_data(results: list[RowResult], train: TrainSide) -> dict[str, dict[str, int]]:
    """Selected pools whose sources an exact / near-duplicate eval row matched.

    An exact row names its exact-match sources; a near-duplicate row its single strongest
    source by shared n-grams. Partial (same-story) matches do not make a source affected.
    """
    affected: set[str] = set()
    for r in results:
        if r.tier == "exact":
            affected.update(r.source_ids)
        elif r.tier == "near_duplicate" and r.source_ids:
            affected.add(r.source_ids[0])
    out: dict[str, dict[str, int]] = {}
    for pool_name, pool in sorted(train.pools.items()):
        hit = [sid for sid in pool if sid in affected]
        if hit:
            out[pool_name] = {
                "sources": len(hit), "sources_total": len(pool),
                "pairs": sum(pool[s] for s in hit), "pairs_total": sum(pool.values()),
            }
    return out


def _examples(results: list[RowResult], k: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for tier in TIERS[:3]:
        rows = [r for r in results if r.tier == tier]
        rows.sort(key=lambda r: (-r.containment, str(r.row_id)))
        if rows:
            out[tier] = [
                {"row": r.row_id, "field": r.field, "containment": round(r.containment, 4),
                 "exposure": r.exposure, "source_ids": r.source_ids[:3], "shared_span": r.span}
                for r in rows[:k]
            ]
    return out


def exclusion_payload(train: TrainSide, benchmark: str, task: str, results: list[RowResult],
                      params: Params, report_sha256: str) -> dict[str, Any]:
    def detail(rows: list[RowResult]) -> dict[str, dict[str, Any]]:
        return {
            str(r.row_id): {"tier": r.tier, "containment": round(r.containment, 4),
                            "exposure": r.exposure, "source_ids": r.source_ids}
            for r in rows
        }

    excluded = sorted((r for r in results if is_excluded(r)), key=lambda r: _sort_key(r.row_id))
    control = sorted((r for r in results if is_unseen_exposed(r)), key=lambda r: _sort_key(r.row_id))
    return {
        "schema_version": SCHEMA_VERSION,
        "train": train.name,
        "benchmark": benchmark,
        "task": task,
        "policy": POLICY,
        "params": params.to_dict(),
        "report_sha256": report_sha256,
        "n_rows": len(results),
        "n_excluded": len(excluded),
        "indices": [r.row_id for r in excluded],
        "per_index": detail(excluded),
        "n_unseen_exposed": len(control),
        "unseen_exposed_indices": [r.row_id for r in control],
        "per_unseen_exposed_index": detail(control),
    }


def _sort_key(row_id: Any) -> tuple[int, Any]:
    return (0, row_id) if isinstance(row_id, int) else (1, str(row_id))


# =================================================================================
# Benchmarks
# =================================================================================

@dataclass
class Checked:
    """One benchmark/task: its summary plus the per-row results exclusion lists need."""

    benchmark: str
    task: str
    summary: dict[str, Any]
    results: list[RowResult]


def _check_rows(train: TrainSide, rows: list[Row]) -> list[RowResult]:
    return [check_row(row_id, fields, train) for row_id, fields in rows]


def run_benchmarks(train: TrainSide, eval_root: Path, benchmarks: Iterable[str], *,
                   harness_samples: Path | None, ragtruth_split: str,
                   params: Params) -> tuple[list[Checked], dict[str, Any], list[dict[str, str]]]:
    checked: list[Checked] = []
    extra: dict[str, Any] = {}
    not_checked: list[dict[str, str]] = []

    def attempt(benchmark: str, task: str, loader) -> None:
        try:
            path, rows = loader()
        except NotChecked as exc:
            not_checked.append({"benchmark": benchmark, "task": task, "reason": str(exc)})
            return
        _log(f"  {benchmark}/{task}: {len(rows)} rows")
        results = _check_rows(train, rows)
        summary = {"input": _file_record(path), **summarize(results, train, params)}
        checked.append(Checked(benchmark, task, summary, results))

    benchmarks = list(benchmarks)
    if "halueval" in benchmarks:
        for task in HALUEVAL_TASKS:
            attempt("halueval", task, lambda t=task: halueval_rows(eval_root, t))
    if "faitheval" in benchmarks:
        for task in FAITHEVAL_TASKS:
            attempt("faitheval", task, lambda t=task: faitheval_rows(eval_root, t))
    if "truthfulqa" in benchmarks:
        attempt("truthfulqa", "TruthfulQA.csv", lambda: truthfulqa_rows(eval_root))
    if "harness" in benchmarks:
        if harness_samples is not None:
            attempt("harness", "truthfulqa", lambda: harness_rows(harness_samples))
        else:
            not_checked.append({
                "benchmark": "harness", "task": "truthfulqa",
                "reason": "no --harness-samples given; the TruthfulQA.csv result stands in, but "
                          "the CSV holds 790 of the 817 items the harness evaluates, so 27 are unchecked",
            })
    if "ragtruth_benchmark" in benchmarks:
        extra["ragtruth_benchmark"] = _ragtruth_benchmark(train, eval_root, ragtruth_split, params,
                                                          checked, not_checked)
    return checked, extra, not_checked


def _ragtruth_benchmark(train: TrainSide, eval_root: Path, split: str, params: Params,
                        checked: list[Checked], not_checked: list[dict[str, str]]) -> dict[str, Any]:
    """Source-id identity + text overlap at the benchmark split, plus the release's own
    test <-> train overlap (an upstream property that only concerns this benchmark)."""
    dataset = eval_root / "RAGTruth-reproduce" / "dataset"
    out: dict[str, Any] = {}
    for sp in dict.fromkeys([split, "all"]):
        try:
            sources, used = ragtruth_sources(dataset, sp)
        except NotChecked as exc:
            not_checked.append({"benchmark": "ragtruth_benchmark", "task": sp, "reason": str(exc)})
            continue
        identity = {e: 0 for e in (*EXPOSURES, "not_in_train")}
        for s in sources:
            identity[train.exposure_of_source(str(s["source_id"])) or "not_in_train"] += 1
        block: dict[str, Any] = {"inputs": [_file_record(p) for p in used], "identity": identity}
        if sp == split:
            _log(f"  ragtruth_benchmark/{sp}: {len(sources)} sources")
            rows = [(str(s["source_id"]),
                     [(k, t) for k in ("source_info", "prompt") if (t := as_text(s.get(k)))])
                    for s in sources]
            results = _check_rows(train, rows)
            # Identity is exposure too: a benchmark source the arms trained on is excluded
            # even if its rendering differs from every indexed text.
            for r in results:
                ident = train.exposure_of_source(str(r.row_id))
                if ident is not None and r.tier == "none":
                    r.tier, r.exposure, r.source_ids = "exact", ident, [str(r.row_id)]
            block.update(summarize(results, train, params))
            checked.append(Checked("ragtruth_benchmark", sp, block, results))
        out[sp] = block

    try:
        out["internal_test_vs_train"] = _ragtruth_internal(dataset, params)
    except NotChecked as exc:
        not_checked.append({"benchmark": "ragtruth_benchmark", "task": "internal_test_vs_train",
                            "reason": str(exc)})
    return out


def _ragtruth_internal(dataset: Path, params: Params) -> dict[str, Any]:
    train_rows, _ = ragtruth_sources(dataset, "train")
    test_rows, _ = ragtruth_sources(dataset, "test")
    internal = TrainSide("ragtruth_release_train", "ragtruth_release", TrainIndex(params), [], [])
    for s in train_rows:
        text = as_text(s.get("source_info"))
        if text:
            internal.index.add(str(s["source_id"]), text)
    internal.exposure_by_group = ["train"] * len(internal.index.group_names)

    by_task: dict[str, dict[str, Any]] = {}
    for s in test_rows:
        text = as_text(s.get("source_info"))
        r = check_row(str(s["source_id"]), [("source_info", text)] if text else [], internal)
        block = by_task.setdefault(s["task_type"], {"n_sources": 0, **{t: 0 for t in TIERS}, "near_duplicates": []})
        block["n_sources"] += 1
        block[r.tier] += 1
        if r.tier in EXCLUDE_TIERS:
            block["near_duplicates"].append({"source_id": r.row_id, "tier": r.tier,
                                             "containment": round(r.containment, 4),
                                             "train_source_ids": r.source_ids})
    return dict(sorted(by_task.items()))


# =================================================================================
# Reports
# =================================================================================

def git_rev(path: Path) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def build_report(train: TrainSide, checked: list[Checked], extra: dict[str, Any],
                 not_checked: list[dict[str, str]], run_configs: list[dict[str, Any]],
                 params: Params, eval_root: Path) -> dict[str, Any]:
    benchmarks: dict[str, dict[str, Any]] = {}
    for c in checked:
        if c.benchmark != "ragtruth_benchmark":
            benchmarks.setdefault(c.benchmark, {})[c.task] = c.summary
    benchmarks.update(extra)
    git = {"eval_master": git_rev(ROOT)}
    for inp in train.inputs[:1]:
        git["train_repo"] = git_rev((ROOT / inp["path"]).resolve().parent)
    exposure_counts = {e: 0 for e in EXPOSURES}
    for e in train.exposure_by_group:
        exposure_counts[e] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "train": {
            "name": train.name, "kind": train.kind, "inputs": train.inputs,
            "n_groups": len(train.index.group_names), "n_texts_indexed": train.index.n_texts,
            "n_ngrams": len(train.index.grams), "groups_by_exposure": exposure_counts,
            "notes": train.notes,
        },
        "params": params.to_dict(),
        "policy": POLICY,
        "eval_root": _display_path(eval_root),
        "git": git,
        "boilerplate_ngrams_ignored": train.index.n_boilerplate,
        "benchmarks": benchmarks,
        "not_checked": not_checked,
        "run_configs": run_configs,
    }


def render_markdown(report: dict[str, Any]) -> str:
    t = report["train"]
    lines = [
        f"# Train <-> eval overlap: `{t['name']}`",
        "",
        f"Generated by `python -m analysis.overlap` (schema {report['schema_version']}). "
        f"Eval_master `{report['git'].get('eval_master')}`, train repo `{report['git'].get('train_repo')}`.",
        "",
        f"- **Train side:** {t['n_groups']} groups ({_fmt_counts(t['groups_by_exposure'])}), "
        f"{t['n_texts_indexed']} distinct texts, {t['n_ngrams']} n-grams; "
        f"{report['boilerplate_ngrams_ignored']} boilerplate n-grams ignored (df > {report['params']['df_max']}).",
        f"- **Params:** {report['params']['ngram']}-grams, near-duplicate >= {report['params']['near_dup']}, "
        f"exposure from top-{report['params']['exposure_topk']} groups.",
        f"- **Policy:** {report['policy']}.",
        "",
        "## Tiers per benchmark",
        "",
        "| benchmark | task | rows | exact | near-dup | partial | none | excluded | unseen-exposed | kept |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    blocks = list(_iter_blocks(report["benchmarks"]))
    for bench, task, s in blocks:
        tiers = s["tiers"]
        lines.append(f"| {bench} | {task} | {s['n_rows']} | {tiers['exact']} | {tiers['near_duplicate']} | "
                     f"{tiers['partial']} | {tiers['none']} | {s['excluded']} | {s['unseen_exposed']} | {s['kept']} |")

    for bench, task, s in blocks:
        if not any(s["tiers"][x] for x in TIERS[:3]):
            continue
        lines += ["", f"## {bench} / {task}", "", "| exposure | exact | near-dup | partial |", "|---|---:|---:|---:|"]
        for e in EXPOSURES:
            m = s["exposure_by_tier"][e]
            lines.append(f"| `{e}` | {m['exact']} | {m['near_duplicate']} | {m['partial']} |")
        pc = s["partial_containment"]
        if pc["max"] is not None:
            lines += ["", f"Partial containment: median {pc['median']}, max {pc['max']}, "
                          f"{pc['n_ge_0.10']} rows >= 0.10, {pc['n_ge_0.20']} rows >= 0.20."]
        if s["fields"]:
            lines += ["", "Fields: " + "; ".join(
                f"`{k}` {v['exact']}/{v['near_duplicate']}/{v['partial']} (max {v['max_containment']})"
                for k, v in s["fields"].items())]
        if s.get("affected_train_data"):
            lines += ["", "| affected pool | sources | pairs |", "|---|---:|---:|"]
            for pool, a in s["affected_train_data"].items():
                lines.append(f"| `{pool}` | {a['sources']} / {a['sources_total']} | {a['pairs']} / {a['pairs_total']} |")
        for tier, rows in s["examples"].items():
            lines += ["", f"Examples, {tier}:"]
            for ex in rows:
                lines.append(f"- row {ex['row']} (`{ex['field']}`, {ex['containment']}, {ex['exposure']}, "
                             f"sources {', '.join(ex['source_ids'])}): \"{ex['shared_span']}\"")

    rb = report["benchmarks"].get("ragtruth_benchmark")
    if rb:
        lines += ["", "## RAG-Truth benchmark: source identity", "", "| split | " + " | ".join(
            f"`{e}`" for e in (*EXPOSURES, "not_in_train")) + " |", "|---|" + "---:|" * (len(EXPOSURES) + 1)]
        for sp, block in rb.items():
            if sp != "internal_test_vs_train":
                lines.append(f"| {sp} | " + " | ".join(str(block["identity"][e]) for e in (*EXPOSURES, "not_in_train")) + " |")
        internal = rb.get("internal_test_vs_train")
        if internal:
            lines += ["", "Release test <-> release train (upstream property):", ""]
            for task, b in internal.items():
                near = ", ".join(f"`{d['source_id']}` {d['containment']}" for d in b["near_duplicates"]) or "none"
                lines.append(f"- {task}: {b['n_sources']} test sources, exact {b['exact']}, near-dup "
                             f"{b['near_duplicate']} ({near}), partial {b['partial']}")

    if report["not_checked"]:
        lines += ["", "## Not checked", ""]
        lines += [f"- {n['benchmark']}/{n['task']}: {n['reason']}" for n in report["not_checked"]]
    if report["run_configs"]:
        lines += ["", "## Training run configs", ""]
        for rc in report["run_configs"]:
            ident = "; ".join(f"{i['benchmark']} via `{i['via']}`" for i in rc.get("identity_overlaps", []))
            lines.append(f"- `{rc['path']}`: data `{rc.get('data_name')}` sources {rc.get('sources')} -> "
                         f"{ident or 'no identity overlap'}{' (' + rc['error'] + ')' if rc.get('error') else ''}")
    lines += ["", "## Inputs", ""]
    lines += [f"- `{i['path']}` sha256 `{i['sha256']}`" for i in t["inputs"]]
    return "\n".join(lines) + "\n"


def _iter_blocks(benchmarks: dict[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    for bench, tasks in benchmarks.items():
        for task, s in tasks.items():
            if isinstance(s, dict) and "tiers" in s:
                yield bench, task, s


def _fmt_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items())


def _dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def write_outputs(train: TrainSide, report: dict[str, Any], checked: list[Checked], params: Params,
                  out_root: Path, exclusions_root: Path | None) -> dict[str, list[Path]]:
    out_dir = Path(out_root) / train.name
    out_dir.mkdir(parents=True, exist_ok=True)
    text = _dumps(report)
    # Bytes, not write_text: Windows would translate "\n" and the recorded sha256 of the
    # report would no longer be the sha256 of the file.
    json_path = out_dir / "overlap_report.json"
    json_path.write_bytes(text.encode("utf-8"))
    md_path = out_dir / "overlap_report.md"
    md_path.write_bytes(render_markdown(report).encode("utf-8"))
    written: dict[str, list[Path]] = {"report": [json_path, md_path], "exclusions": []}

    if exclusions_root is not None:
        report_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        target = Path(exclusions_root) / train.name
        target.mkdir(parents=True, exist_ok=True)
        # Every list names this report by sha256. outputs/ is not committed, so a byte copy
        # goes beside the lists: the reference then resolves wherever the lists travel.
        for source in (json_path, md_path):
            copy = target / source.name
            copy.write_bytes(source.read_bytes())
            written["report"].append(copy)
        expected: set[Path] = set()
        for c in checked:
            if not any(is_excluded(r) for r in c.results):
                continue
            path = target / f"{c.benchmark}__{c.task}.json"
            expected.add(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = exclusion_payload(train, c.benchmark, c.task, c.results, params, report_sha)
            path.write_bytes(_dumps(payload).encode("utf-8"))
            written["exclusions"].append(path)
        stale = sorted(p for p in target.glob("*__*.json") if p not in expected) if target.is_dir() else []
        for p in stale:
            print(f"!! stale exclusion list (this run excludes nothing there): {p}", file=sys.stderr)
    return written


# =================================================================================
# CLI
# =================================================================================

def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_train_arg(arg: str) -> tuple[str, str]:
    if "=" not in arg:
        raise SystemExit(f"--train must be NAME=SPEC, got {arg!r}")
    name, spec = arg.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise SystemExit(f"--train name {name!r} must be filename-safe")
    return name, spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="append", default=None, metavar="NAME=SPEC",
                    help="Training corpus, repeatable. SPEC is a SP-DPO-Base data/ragtruth dir or "
                         "jsonl:PATH:field,field[:group_field]. Default: ragtruth=<SP-DPO-Base>/data/ragtruth.")
    ap.add_argument("--train-run-config", action="append", default=[], metavar="PATH",
                    help="A training run's .hydra/config.yaml; flags halueval*/truthfulqa* data as "
                         "identity overlap. Repeatable.")
    ap.add_argument("--eval-root", default=str(ROOT), help="Eval_master root holding the benchmark folders.")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS),
                    help=f"Comma-separated subset of {','.join(BENCHMARKS)}.")
    ap.add_argument("--harness-samples", default=None, metavar="PATH",
                    help="A harness run's samples.jsonl (all 817 TruthfulQA items). Without it the "
                         "harness entry is not_checked and TruthfulQA.csv stands in.")
    ap.add_argument("--ragtruth-split", default="test",
                    help="RAG-Truth benchmark split to check (the 'all' identity counts are reported too).")
    ap.add_argument("--ngram", type=int, default=Params.ngram)
    ap.add_argument("--near-dup", type=float, default=Params.near_dup)
    ap.add_argument("--df-max", type=int, default=Params.df_max)
    ap.add_argument("--exposure-topk", type=int, default=Params.exposure_topk)
    ap.add_argument("--max-examples", type=int, default=Params.max_examples)
    ap.add_argument("--out", default="outputs/overlap", help="Report root: <out>/<train>/overlap_report.{json,md}.")
    ap.add_argument("--write-exclusions", action="store_true",
                    help="Also write <exclusions-dir>/<train>/<benchmark>__<task>.json for every "
                         "non-empty exclusion list.")
    ap.add_argument("--exclusions-dir", default=str(ROOT / "decontamination"))
    args = ap.parse_args(argv)

    params = Params(args.ngram, args.near_dup, args.df_max, args.exposure_topk, args.max_examples)
    benchmarks = [b for b in args.benchmarks.split(",") if b]
    unknown = sorted(set(benchmarks) - set(BENCHMARKS))
    if unknown:
        ap.error(f"unknown benchmark(s) {unknown}; choose from {BENCHMARKS}")
    trains = [_parse_train_arg(a) for a in args.train] if args.train else [("ragtruth", str(DEFAULT_RAGTRUTH_DATA))]
    run_configs = [read_run_config(Path(p)) for p in args.train_run_config]
    eval_root = Path(args.eval_root)

    for name, spec in trains:
        _log(f"==> indexing train side '{name}' ({spec})")
        train = load_train(name, spec, params)
        _log(f"    {len(train.index.group_names)} groups, {train.index.n_texts} texts, "
             f"{len(train.index.grams)} n-grams")
        checked, extra, not_checked = run_benchmarks(
            train, eval_root, benchmarks,
            harness_samples=Path(args.harness_samples) if args.harness_samples else None,
            ragtruth_split=args.ragtruth_split, params=params,
        )
        report = build_report(train, checked, extra, not_checked, run_configs, params, eval_root)
        written = write_outputs(train, report, checked, params, Path(args.out),
                                Path(args.exclusions_dir) if args.write_exclusions else None)
        print(f"\n== {name}")
        for bench, task, s in _iter_blocks(report["benchmarks"]):
            print(f"  {bench:<18} {task:<16} rows {s['n_rows']:>6}  " + "  ".join(
                f"{t} {s['tiers'][t]}" for t in TIERS) + f"  | excluded {s['excluded']}")
        for n in not_checked:
            print(f"  not checked: {n['benchmark']}/{n['task']}: {n['reason']}")
        for kind, paths in written.items():
            for p in paths:
                print(f"  wrote {kind}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
