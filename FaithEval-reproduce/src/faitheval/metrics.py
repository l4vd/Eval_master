"""Answer normalization and scoring rules for FaithEval tasks."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_EXTRA_PUNCTUATION = "‘’´`"  # ' ' ´ `


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation, and collapse whitespace.

    Mirrors the standard SQuAD-style normalization used across FaithEval's
    reference evaluation snippets so that scores remain comparable.
    """
    text = text.replace("_", " ").lower()
    exclude = set(string.punctuation + _EXTRA_PUNCTUATION)
    text = "".join(ch if ch not in exclude else " " for ch in text)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def phrase_match(prediction: str, valid_phrases: Iterable[str]) -> bool:
    """True if any of `valid_phrases` occurs in the normalized prediction.

    Used for the unanswerable and inconsistent tasks, where correctness is
    defined by the model naming the right *category* of context issue rather
    than reproducing an exact reference string.
    """
    normalized = normalize_answer(prediction)
    return any(phrase in normalized for phrase in valid_phrases)


def answer_match(prediction: str, references: Iterable[str]) -> bool:
    """True if the normalized prediction exactly matches any reference answer.

    Used for the counterfactual task, where correctness means reproducing the
    (counterfactual) answer supported by the given context.
    """
    normalized_pred = normalize_answer(prediction)
    return any(normalized_pred == normalize_answer(ref) for ref in references)
