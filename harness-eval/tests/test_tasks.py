"""Task resolution tests. These need lm_eval installed (TaskManager indexes the
local task YAML only — no dataset or model download)."""

import inspect

import pytest

lm_eval = pytest.importorskip("lm_eval")

from harness_eval.tasks import _known_names, list_tasks, resolve_tasks  # noqa: E402

TRUTHFULQA_TAGS = ("truthfulqa", "truthfulqa_multilingual", "truthfulqa-multi")


@pytest.fixture(scope="module")
def task_manager():
    from lm_eval.tasks import TaskManager

    return TaskManager()


def test_three_truthfulqa_tags_are_known_to_installed_lm_eval(task_manager):
    # Canary: catches an upstream rename we can't see when lm_eval isn't installed.
    known = _known_names(task_manager)
    for tag in TRUTHFULQA_TAGS:
        assert tag in known, f"{tag!r} not registered in installed lm_eval"


def test_resolve_tasks_accepts_known_tag(task_manager):
    assert resolve_tasks(["truthfulqa"], task_manager) == ["truthfulqa"]


def test_resolve_tasks_rejects_unknown_name_with_suggestion(task_manager):
    with pytest.raises(ValueError, match="truthfulqa"):
        resolve_tasks(["truthfulqaa"], task_manager)


def test_resolve_tasks_deduplicates_preserving_order(task_manager):
    assert resolve_tasks(["truthfulqa", "truthfulqa"], task_manager) == ["truthfulqa"]


def test_resolve_tasks_expands_glob(task_manager):
    matches = resolve_tasks(["truthfulqa_*_mc1"], task_manager)
    assert len(matches) > 1
    assert all(name.endswith("_mc1") for name in matches)


def test_list_tasks_filters_by_substring(task_manager):
    names = list_tasks("truthfulqa", task_manager)
    assert names == sorted(names)
    assert all("truthfulqa" in n.lower() for n in names)
    assert "truthfulqa" in names


def test_simple_evaluate_accepts_every_kwarg_we_pass():
    # Zero-cost drift guard: our explicit kwarg set must be a subset of the
    # installed simple_evaluate signature.
    params = set(inspect.signature(lm_eval.simple_evaluate).parameters)
    ours = {
        "model",
        "model_args",
        "tasks",
        "num_fewshot",
        "batch_size",
        "device",
        "limit",
        "apply_chat_template",
        "fewshot_as_multiturn",
        "system_instruction",
        "log_samples",
        "task_manager",
    }
    assert ours <= params, f"simple_evaluate is missing: {ours - params}"
