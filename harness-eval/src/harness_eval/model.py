"""Translate the shared `model.id` interface into lm_eval `model_args`.

This is the module's core value-add. The launcher hands every benchmark one
`model.id` — a Hub id, a local full-model directory, or a PEFT/LoRA adapter
checkpoint (a `final_checkpoint` from the DPO training pipeline). lm_eval's `hf`
backend is configured through a `model_args` mapping instead, so we translate.

Deliberately free of `torch`/`peft`/`transformers` imports: a PEFT adapter is
always a *local* directory, so its `adapter_config.json` is read with `json.load`
(what `PeftConfig.from_pretrained` does under the hood anyway). This keeps the
whole module import-light, so its unit tests run in milliseconds without the heavy
stack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Files a saved Hugging Face tokenizer writes; the presence of any one means a
# checkpoint directory carries its own tokenizer.
_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "spiece.model",
    "sentencepiece.bpe.model",
)


def _looks_like_local_path(model_id: str) -> bool:
    """Heuristically detect a local filesystem path (as opposed to a Hub repo id).

    Hub repo ids can contain "/" (`org/repo`), so that alone isn't a reliable
    signal. Absolute paths, explicit relative-path prefixes, and backslashes
    (Windows paths) are unambiguous. Ported verbatim from the sibling benchmarks.
    """
    return (
        os.path.isabs(model_id)
        or model_id.startswith(("/", "./", "../", ".\\", "..\\", "~/", "~\\"))
        or "\\" in model_id
    )


def _check_local_path_exists(model_id: str, *, what: str) -> None:
    """Fail fast if `model_id` looks like a local path but doesn't exist.

    Without this, lm_eval treats a missing path as a Hub repo id and raises a
    confusing network/404-shaped error instead of a plain "path not found".
    """
    if _looks_like_local_path(model_id) and not Path(model_id).exists():
        raise FileNotFoundError(f"{what} path does not exist: {model_id}")


def _is_peft_adapter(model_path: str) -> bool:
    """True if `model_path` is a local directory holding a PEFT adapter checkpoint.

    A LoRA run saves only `adapter_config.json` + adapter weights, not a full
    model. Only ever matches a local directory, so a Hub id is never an adapter —
    the same limitation every sibling benchmark has.
    """
    return (Path(model_path) / "adapter_config.json").is_file()


def _has_tokenizer_files(path: str) -> bool:
    directory = Path(path)
    return any((directory / name).is_file() for name in _TOKENIZER_FILES)


def _read_adapter_base(model_path: str) -> str | None:
    """The base model recorded in an adapter's `adapter_config.json`, if any."""
    with (Path(model_path) / "adapter_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    base = config.get("base_model_name_or_path")
    return str(base) if base else None


def build_model_args(
    model_id: str,
    *,
    base_model_id: str | None = None,
    tokenizer_id: str | None = None,
    cache_dir: str | None = None,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Build the `model_args` mapping for `lm_eval.simple_evaluate(model="hf", ...)`.

    Three cases behind one `model_id`:

    * Hub id / local full-model dir -> ``pretrained=<model_id>``.
    * PEFT/LoRA adapter dir -> ``pretrained=<base>, peft=<adapter>``, the base
      taken from ``base_model_id`` or the adapter's own config.

    Tokenizer (the subtle one): lm_eval loads the tokenizer from ``pretrained`` =
    the *base* for an adapter, but every sibling benchmark loads it from the
    checkpoint dir (`tokenizer_id or model_id`). To keep a checkpoint scored the
    same way here as everywhere else, we pass ``tokenizer=<adapter dir>``
    explicitly when the adapter carries its own tokenizer files, falling back to
    the base otherwise.

    A dict (not lm_eval's comma-separated string) is returned so a checkpoint path
    containing a comma cannot corrupt the args, and to avoid a lossy re-parse — see
    :func:`model_args_to_string`.
    """
    _check_local_path_exists(model_id, what="Model")
    args: dict[str, Any] = {}

    if _is_peft_adapter(model_id):
        resolved_base = base_model_id or _read_adapter_base(model_id)
        if not resolved_base:
            raise ValueError(
                f"{model_id} is a PEFT adapter with no base model recorded in its "
                f"adapter_config.json; pass --base-model-id to name the base."
            )
        _check_local_path_exists(resolved_base, what="Base model")
        args["pretrained"] = resolved_base
        args["peft"] = model_id
        if tokenizer_id:
            args["tokenizer"] = tokenizer_id
        elif _has_tokenizer_files(model_id):
            # The sibling contract: the checkpoint's own tokenizer, not the base's.
            args["tokenizer"] = model_id
        # else: omit -> lm_eval falls back to `pretrained` (the base) tokenizer.
    else:
        args["pretrained"] = model_id
        if tokenizer_id:
            args["tokenizer"] = tokenizer_id
        # else: omit -> lm_eval uses `pretrained` (= model_id), which already
        # matches the sibling contract `tokenizer_id or model_id`.

    args["dtype"] = dtype
    if cache_dir:
        args["cache_dir"] = cache_dir
    if trust_remote_code:
        args["trust_remote_code"] = True
    return args


def model_args_to_string(model_args: dict[str, Any]) -> str:
    """Render `model_args` as lm_eval's `key=value,key=value` string.

    Provenance only — `simple_evaluate` receives the dict itself. lm_eval parses
    this string form with a bare `split(",")` and no escaping, so a value
    containing a comma is unrepresentable; we reject it loudly rather than emit a
    string that silently corrupts when re-parsed.
    """
    parts = []
    for key, value in model_args.items():
        text = str(value)
        if "," in text:
            raise ValueError(
                f"model_args[{key!r}] contains a comma ({text!r}); lm_eval's "
                f"comma-separated arg-string cannot represent it."
            )
        parts.append(f"{key}={text}")
    return ",".join(parts)
