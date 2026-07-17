"""Resolve and discover lm-evaluation-harness task names.

lm_eval owns the task catalogue; this module validates the names carried in the
Hydra config against the *installed* lm_eval (task spellings drift between
versions), expands glob patterns, and powers the `--list-tasks` discovery mode.
Adding a new benchmark is therefore a new config name, not a code change.

`truthfulqa`, `truthfulqa_multilingual`, and `truthfulqa-multi` are lm_eval *tags*
(each expands to its subtasks); a tag passes straight through to lm_eval, which
does the expansion.
"""

from __future__ import annotations

import difflib
import fnmatch
from collections.abc import Iterable
from typing import Any

# TaskManager attributes that enumerate names. The surface has shifted across
# lm_eval 0.4.x, so we union whatever exists rather than depend on any single one.
_NAME_ATTRS = ("all_tasks", "all_groups", "all_subtasks", "all_tags")


def _known_names(task_manager: Any) -> set[str]:
    """Every task, group, subtask, and tag name the TaskManager knows about."""
    names: set[str] = set()
    for attr in _NAME_ATTRS:
        names.update(getattr(task_manager, attr, None) or [])
    index = getattr(task_manager, "task_index", None)
    if isinstance(index, dict):
        names.update(index.keys())
    return names


def _is_glob(name: str) -> bool:
    return any(ch in name for ch in "*?[")


def resolve_tasks(names: Iterable[str], task_manager: Any) -> list[str]:
    """Validate/expand `names` against the installed lm_eval, preserving order.

    Exact names (including tags and groups) pass through; glob patterns expand to
    the matching known names, sorted. An unknown name raises `ValueError` with a
    close-match suggestion. Duplicates are collapsed while keeping first-seen order.
    """
    known = _known_names(task_manager)
    resolved: list[str] = []
    seen: set[str] = set()
    for name in names:
        if _is_glob(name):
            matches = sorted(n for n in known if fnmatch.fnmatchcase(n, name))
        elif name in known:
            matches = [name]
        else:
            matches = []
        if not matches:
            suggestions = difflib.get_close_matches(name, sorted(known), n=3)
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(f"Unknown lm_eval task or tag: {name!r}.{hint}")
        for match in matches:
            if match not in seen:
                seen.add(match)
                resolved.append(match)
    return resolved


def list_tasks(substring: str, task_manager: Any) -> list[str]:
    """Sorted known names, optionally filtered by a case-insensitive substring."""
    names = _known_names(task_manager)
    if substring:
        needle = substring.lower()
        names = {n for n in names if needle in n.lower()}
    return sorted(names)
