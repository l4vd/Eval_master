"""Local GPT-judge / GPT-info models for TruthfulQA.

The original "GPT-judge" and "GPT-info" metrics score an answer's truthfulness and
informativeness with GPT-3 models fine-tuned by the TruthfulQA authors, queried
through ``openai.Completion``. Both that API and those fine-tuned engines are gone,
so ``metrics.run_end2end_GPT3`` is unrunnable in practice.

The authors released successor judges fine-tuned on the same data
(``allenai/truthfulqa-truth-judge-llama2-7B`` and ``-info-judge-llama2-7B``). They
are consumed exactly like the GPT-3 originals -- same prompt from
``utilities.format_end2end_prompt``, same score (the probability of the ``yes``
continuation), same 0.5 threshold -- so the metric *definition* is unchanged and
only the engine differs. They need no API key and run offline on a GPU node.

Because the judges are completion-fine-tuned on that literal prompt string, this
module deliberately does **not** apply a chat template.
"""

from __future__ import annotations

import logging

import torch

from .hf_local import _DTYPE_BY_NAME, _check_local_path_exists, _load_causal_lm

logger = logging.getLogger(__name__)


class LocalJudge:
    """A fine-tuned truth/info judge scoring `P(yes)` for a completion prompt."""

    def __init__(
        self,
        model_id: str,
        base_model_id: str | None = None,
        tokenizer_id: str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        dtype: str = "float32",
    ) -> None:
        from transformers import AutoTokenizer

        if dtype not in _DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(_DTYPE_BY_NAME)}")

        logger.info("Loading judge %s (dtype=%s, device=%s)", model_id, dtype, device)
        self.model = _load_causal_lm(model_id, base_model_id, cache_dir, _DTYPE_BY_NAME[dtype])
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()
        self.device = device

        resolved_tokenizer_id = tokenizer_id or model_id
        _check_local_path_exists(resolved_tokenizer_id, what="Judge tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_id, cache_dir=cache_dir)

        self._yes_ids = self._continuation_ids("yes")
        self._no_ids = self._continuation_ids("no")
        if not self._yes_ids or not self._no_ids:
            raise ValueError(f"Judge tokenizer {resolved_tokenizer_id!r} cannot encode 'yes'/'no'")

    def _continuation_ids(self, word: str) -> list[int]:
        """First-token ids for `word` as a continuation, in the surface forms it may take.

        The GPT-3 judges emitted `' yes'`; a Llama tokenizer may instead produce
        `'yes'`, `'▁yes'` or a capitalized variant. Collecting every plausible
        first token and summing their mass keeps the score faithful to
        "probability the judge says yes" across tokenizers.
        """
        ids = set()
        for surface in (f" {word}", word, f" {word.capitalize()}", word.capitalize()):
            encoded = self.tokenizer(surface, add_special_tokens=False).input_ids
            if encoded:
                ids.add(encoded[0])
        return sorted(ids)

    @torch.no_grad()
    def score_yes(self, prompt: str) -> tuple[float, float]:
        """`(p_yes, p_yes_normalized)` for `prompt`.

        `p_yes` is the raw probability mass on a `yes` continuation -- the
        upstream-comparable score that `metrics` thresholds at 0.5.
        `p_yes_normalized` is `P(yes) / (P(yes) + P(no))`, a diagnostic that is
        robust to the judge spreading mass over other tokens.
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if self.device is not None:
            input_ids = input_ids.to(self.device)

        logits = self.model(input_ids)[0][0, -1, :]
        probs = logits.softmax(-1)

        p_yes = probs[self._yes_ids].sum().item()
        p_no = probs[self._no_ids].sum().item()
        total = p_yes + p_no
        p_yes_norm = p_yes / total if total > 0 else 0.0

        return p_yes, p_yes_norm
