"""Configuration schema for a single harness-eval run.

Run settings (model, tasks, decoding, I/O) are supplied via CLI flags and
assembled into :class:`EvalConfig` by :mod:`harness_eval.cli`. Unlike the sibling
benchmarks there is no per-task YAML: lm-evaluation-harness owns the task
definitions, so a task is just a name carried through to `lm_eval`.
"""

from __future__ import annotations

import dataclasses

_VALID_DTYPES = ("bfloat16", "float16", "float32")


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    """Full configuration for a single lm-evaluation-harness run."""

    model_id: str
    tasks: tuple[str, ...]
    base_model_id: str | None = None
    tokenizer_id: str | None = None
    cache_dir: str | None = None
    num_fewshot: int | None = None
    batch_size: str = "1"
    limit: int | None = None
    apply_chat_template: bool = False
    fewshot_as_multiturn: bool = False
    system_instruction: str | None = None
    trust_remote_code: bool = False
    log_samples: bool = True
    output_dir: str = "outputs"
    device: str = "cpu"
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("At least one task is required.")
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; choose from {list(_VALID_DTYPES)}")
        if self.fewshot_as_multiturn and not self.apply_chat_template:
            # lm_eval 0.4.5 raised this itself; 0.4.12 dropped the guard, so we
            # reinstate it for identical behaviour across the supported version band.
            raise ValueError("fewshot_as_multiturn requires apply_chat_template")
