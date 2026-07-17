"""Run lm-evaluation-harness and write the normalised outputs.

`run_evaluation` is the single seam between our config and `lm_eval`. It takes an
optional `evaluate_fn` so tests can drive the whole flow with a fake — and hence
with lm_eval (and torch/datasets) not installed at all.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from harness_eval.config import EvalConfig
from harness_eval.model import build_model_args, model_args_to_string
from harness_eval.results import build_summary, write_outputs

logger = logging.getLogger(__name__)


def _provenance(config: EvalConfig, model_args: dict[str, Any], resolved_tasks: list[str]) -> dict[str, Any]:
    """The reproducibility header written at the top of summary.json."""
    try:
        model_args_string = model_args_to_string(model_args)
    except ValueError as exc:
        # A comma in a checkpoint path defeats lm_eval's arg-string form; the dict
        # we actually pass is unaffected, so keep the run going and record the dict.
        logger.warning("model_args not renderable as a string for provenance: %s", exc)
        model_args_string = None
    return {
        "model_args": model_args,
        "model_args_string": model_args_string,
        "resolved_tasks": resolved_tasks,
        "num_fewshot": config.num_fewshot,
        "limit": config.limit,
        "batch_size": config.batch_size,
        "device": config.device,
        "dtype": config.dtype,
        "apply_chat_template": config.apply_chat_template,
        "fewshot_as_multiturn": config.fewshot_as_multiturn,
        "system_instruction": config.system_instruction,
    }


def run_evaluation(
    config: EvalConfig,
    *,
    evaluate_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Evaluate `config.tasks` with lm_eval and write outputs to `config.output_dir`.

    Returns the summary dict. When `evaluate_fn` is supplied it fully replaces
    `lm_eval.simple_evaluate` (used by the tests); otherwise lm_eval is imported
    lazily — importing it pulls in torch *and* datasets, which `--help` must not pay.
    """
    task_manager: Any = None
    if evaluate_fn is None:
        # Lazy: importing lm_eval drags in the heavy stack.
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager

        from harness_eval.tasks import resolve_tasks

        task_manager = TaskManager()
        resolved_tasks = resolve_tasks(config.tasks, task_manager)
        evaluate_fn = simple_evaluate
    else:
        # Fake injected: skip the lm_eval-dependent task resolution.
        resolved_tasks = list(config.tasks)

    model_args = build_model_args(
        config.model_id,
        base_model_id=config.base_model_id,
        tokenizer_id=config.tokenizer_id,
        cache_dir=config.cache_dir,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
    )

    logger.info("Evaluating %s on %s", model_args.get("pretrained"), resolved_tasks)
    # Every behaviour-bearing kwarg is passed EXPLICITLY (never left to an lm_eval
    # default): `fewshot_as_multiturn`'s default flipped between 0.4.5 and 0.4.12,
    # and simple_evaluate is @positional_deprecated, so we also call it by keyword.
    results = evaluate_fn(
        model="hf",
        model_args=model_args,
        tasks=resolved_tasks,
        num_fewshot=config.num_fewshot,
        batch_size=config.batch_size,
        device=config.device,
        limit=config.limit,
        apply_chat_template=config.apply_chat_template,
        fewshot_as_multiturn=config.fewshot_as_multiturn,
        system_instruction=config.system_instruction,
        log_samples=config.log_samples,
        task_manager=task_manager,
    )
    if results is None:
        # simple_evaluate returns None off the main rank in a distributed run.
        logger.warning("simple_evaluate returned None (non-primary rank?); nothing to write.")
        return {}

    summary = build_summary(results, _provenance(config, model_args, resolved_tasks))
    write_outputs(results, summary, config.output_dir, log_samples=config.log_samples)
    logger.info("Wrote summary and samples to %s", config.output_dir)
    return summary
