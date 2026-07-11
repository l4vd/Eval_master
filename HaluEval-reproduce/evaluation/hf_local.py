"""Local HuggingFace judge backend for HaluEval evaluation.

HaluEval evaluates a model as a **Yes/No hallucination judge**: it is shown a
(question/context, answer) pair and must say whether the answer is hallucinated.
This module lets that judge be a checkpoint of your own instead of an OpenAI model.

It is a port of ``FaithEval-reproduce/src/faitheval/model.py`` — the same
path/adapter-detection helpers and chat-pipeline wrapper — so ``--model-path`` can
be a Hugging Face Hub id, a local full-model directory, or a PEFT/LoRA adapter
checkpoint (e.g. a ``final_checkpoint`` from the sibling ``SP-DPO-Base`` training
pipeline); an adapter is auto-detected and merged onto its base model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, pipeline

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
    device_map: str,
) -> PreTrainedModel:
    """Load a causal LM, transparently merging a PEFT/LoRA adapter if `model_id` is one."""
    _check_local_path_exists(model_id, what="Model")

    if not _is_peft_adapter(model_id):
        return AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, torch_dtype=torch_dtype, device_map=device_map
        )

    from peft import PeftConfig, PeftModel

    adapter_config = PeftConfig.from_pretrained(model_id)
    resolved_base_id = base_model_id or adapter_config.base_model_name_or_path
    _check_local_path_exists(resolved_base_id, what="Base model")
    logger.info("Detected PEFT adapter at %s; loading base model %s", model_id, resolved_base_id)

    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base_id, cache_dir=cache_dir, torch_dtype=torch_dtype, device_map=device_map
    )
    model = PeftModel.from_pretrained(base_model, model_id)
    return model.merge_and_unload()


class HFChatGenerator:
    """Thin wrapper around a Hugging Face causal-LM chat pipeline used as the judge."""

    def __init__(
        self,
        model_id: str,
        base_model_id: str | None = None,
        tokenizer_id: str | None = None,
        cache_dir: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 16,
    ) -> None:
        if dtype not in _DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(_DTYPE_BY_NAME)}")

        self.max_new_tokens = max_new_tokens

        logger.info("Loading judge model %s (dtype=%s, device_map=%s)", model_id, dtype, device_map)
        model = _load_causal_lm(model_id, base_model_id, cache_dir, _DTYPE_BY_NAME[dtype], device_map)

        # A checkpoint produced by this project's training pipeline always has its
        # own tokenizer saved alongside it (adapter or full); `tokenizer_id` is an
        # escape hatch for checkpoints (e.g. hand-merged weights) that don't.
        resolved_tokenizer_id = tokenizer_id or model_id
        _check_local_path_exists(resolved_tokenizer_id, what="Tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_id, cache_dir=cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map=device_map,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a judgement (ideally 'Yes'/'No') for chat-formatted messages.

        Greedy decoding; the caller normalizes the returned text to Yes/No.
        """
        outputs = self._generator(messages, max_new_tokens=self.max_new_tokens, do_sample=False)
        return outputs[0]["generated_text"][-1]["content"].strip()
