"""Model loading and text generation for the RAGTruth serverless pipeline.

Used for **both** model roles in the two-stage pipeline:
  * the *generation* model (Stage 1) — your own checkpoint / LoRA, producing a
    RAG response for each source item;
  * the *detector* model (Stage 2) — the RAGTruth authors' released hallucination
    detector (`CodingLL/RAGTruth_Eval`), or your own trained detector.

The loader is a port of ``FaithEval-reproduce/src/faitheval/model.py`` — the same
path/adapter-detection helpers — so ``--model-id`` / ``--detector-model-id`` can
each be a Hugging Face Hub id, a local full-model directory, or a PEFT/LoRA
adapter checkpoint (auto-detected via ``adapter_config.json`` and merged onto its
base model).

``HFGenerator`` exposes two generation modes:
  * ``chat(messages)`` — apply the model's chat template (used for the generation
    model, an instruct model);
  * ``complete(text)`` — raw text completion (used for the detector, which the
    RAGTruth baseline drives with a ``[INST] ... [/INST]`` string rather than a
    chat template).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path

import torch

# `transformers` is imported lazily inside the loader/generator so that importing
# this module (e.g. for `GenerationParams` in the CLI, or the metrics tests) does
# not pay the heavy transformers import cost — mirroring the other benchmarks.

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
) -> "PreTrainedModel":
    """Load a causal LM, transparently merging a PEFT/LoRA adapter if `model_id` is one."""
    from transformers import AutoModelForCausalLM

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


@dataclasses.dataclass(frozen=True)
class GenerationParams:
    """Decoding parameters shared by `HFGenerator.chat` / `.complete`."""

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    def to_kwargs(self) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
        return kwargs


class HFGenerator:
    """A loaded causal LM + tokenizer with chat and raw-completion generation."""

    def __init__(
        self,
        model_id: str,
        base_model_id: str | None = None,
        tokenizer_id: str | None = None,
        cache_dir: str | None = None,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
        from transformers import AutoTokenizer

        if dtype not in _DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(_DTYPE_BY_NAME)}")

        logger.info("Loading model %s (dtype=%s, device_map=%s)", model_id, dtype, device_map)
        self.model = _load_causal_lm(
            model_id, base_model_id, cache_dir, _DTYPE_BY_NAME[dtype], device_map
        )
        self.model.eval()

        resolved_tokenizer_id = tokenizer_id or model_id
        _check_local_path_exists(resolved_tokenizer_id, what="Tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_id, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def device(self) -> torch.device:
        return self.model.device

    @torch.no_grad()
    def _generate_from_ids(self, input_ids: torch.Tensor, params: GenerationParams) -> str:
        input_ids = input_ids.to(self.device)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            **params.to_kwargs(),
        )
        new_tokens = outputs[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def chat(self, messages: list[dict[str, str]], params: GenerationParams) -> str:
        """Generate a completion for chat messages via the model's chat template.

        Falls back to a plain ``[INST] ... [/INST]`` wrap if the tokenizer has no
        chat template (e.g. a base, non-instruct model).
        """
        if getattr(self.tokenizer, "chat_template", None):
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            text = "".join(f"[INST] {m['content'].strip()} [/INST]" for m in messages)
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        return self._generate_from_ids(input_ids, params)

    def complete(self, text: str, params: GenerationParams) -> str:
        """Generate a raw completion of `text` (no chat template)."""
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        return self._generate_from_ids(input_ids, params)
