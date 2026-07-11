"""Load a local checkpoint / LoRA adapter for TruthfulQA evaluation.

This is a port of ``FaithEval-reproduce/src/faitheval/model.py`` — the same
path/adapter-detection helpers — adapted to return a ``(model, tokenizer)`` pair
that TruthfulQA's ``models.run_answers`` / ``models.run_probs`` accept via their
``model=`` / ``tokenizer=`` keyword arguments (so the usual
``AutoModelForCausalLM.from_pretrained(engine)`` load is bypassed).

It lets ``--model-path`` point at a Hugging Face Hub id, a local full-model
directory, or a PEFT/LoRA adapter checkpoint (e.g. a ``final_checkpoint``
produced by the sibling ``SP-DPO-Base`` training pipeline); an adapter is
auto-detected and merged onto its base model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _looks_like_local_path(model_id: str) -> bool:
    """Heuristically detect a local filesystem path (as opposed to a Hub repo id).

    Hub repo ids can contain "/" (`org/repo`), so that alone isn't a reliable
    signal. Absolute paths, explicit relative-path prefixes, and backslashes
    (Windows paths) are unambiguous, though.
    """
    return (
        os.path.isabs(model_id)
        or model_id.startswith(("/", "./", "../", ".\\", "..\\", "~/", "~\\"))
        or "\\" in model_id
    )


def _check_local_path_exists(model_id: str, *, what: str) -> None:
    """Fail fast with a clear error if `model_id` looks like a missing local path.

    Without this, `from_pretrained` treats any non-existent path as a Hub repo
    id and raises a confusing network/404-shaped error instead of a plain
    "path not found".
    """
    if _looks_like_local_path(model_id) and not Path(model_id).exists():
        raise FileNotFoundError(f"{what} path does not exist: {model_id}")


def _is_peft_adapter(model_path: str) -> bool:
    """True if `model_path` is a local directory holding a PEFT adapter checkpoint.

    A LoRA run (e.g. `training=dpo peft=lora`) saves only `adapter_config.json` +
    adapter weights, not a full model — `AutoModelForCausalLM.from_pretrained`
    can't load that directory directly.
    """
    return (Path(model_path) / "adapter_config.json").is_file()


def _load_causal_lm(
    model_id: str,
    base_model_id: str | None,
    cache_dir: str | None,
    torch_dtype: torch.dtype,
) -> PreTrainedModel:
    """Load a causal LM, transparently merging a PEFT/LoRA adapter if `model_id` is one."""
    _check_local_path_exists(model_id, what="Model")

    if not _is_peft_adapter(model_id):
        return AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch_dtype
        )

    from peft import PeftConfig, PeftModel

    adapter_config = PeftConfig.from_pretrained(model_id)
    resolved_base_id = base_model_id or adapter_config.base_model_name_or_path
    _check_local_path_exists(resolved_base_id, what="Base model")
    logger.info("Detected PEFT adapter at %s; loading base model %s", model_id, resolved_base_id)

    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base_id, cache_dir=cache_dir, torch_dtype=torch_dtype
    )
    model = PeftModel.from_pretrained(base_model, model_id)
    return model.merge_and_unload()


def load_local_model(
    model_path: str,
    base_model_id: str | None = None,
    tokenizer_id: str | None = None,
    cache_dir: str | None = None,
    device: str | None = None,
    dtype: str = "float32",
) -> tuple[PreTrainedModel, AutoTokenizer]:
    """Load a `(model, tokenizer)` for a Hub id, local path, or LoRA adapter.

    The returned model is placed on `device`, put in eval mode, and has
    `return_dict_in_generate` enabled so `models.run_answers` (which reads
    `outputs.sequences` / `outputs.scores`) works with a pre-loaded model.
    """
    if dtype not in _DTYPE_BY_NAME:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(_DTYPE_BY_NAME)}")

    logger.info("Loading model %s (dtype=%s, device=%s)", model_path, dtype, device)
    model = _load_causal_lm(model_path, base_model_id, cache_dir, _DTYPE_BY_NAME[dtype])
    if device is not None:
        model = model.to(device)
    model.eval()
    # `models.run_answers` calls `model.generate(..., output_scores=True)` and reads
    # `outputs.sequences` / `outputs.scores`; the original code gets this by passing
    # `return_dict_in_generate=True` to `from_pretrained`. Set it here so a pre-loaded
    # model behaves the same.
    model.generation_config.return_dict_in_generate = True

    # A checkpoint from this project's training pipeline saves its own tokenizer
    # (adapter or full); `tokenizer_id` is an escape hatch for weights saved without one.
    resolved_tokenizer_id = tokenizer_id or model_path
    _check_local_path_exists(resolved_tokenizer_id, what="Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
