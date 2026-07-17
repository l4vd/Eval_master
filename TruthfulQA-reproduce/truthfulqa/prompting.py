"""Model-dependent prompt construction for TruthfulQA.

The original benchmark presents every model with the same raw ``Q:/A:`` few-shot
completion string (see ``presets.py`` / ``utilities.format_prompt``). That was
written for GPT-2/GPT-3-era base models. Evaluating an instruction-tuned
checkpoint that way feeds it a prompt in a format it was never tuned on, and
omits the turn markers its training data always had — so this module can instead
render the same few-shot examples through the model's own chat template, which is
what the other three benchmarks in this suite do.

Two styles, selected by ``--prompt_style``:

``completion``
    The original string. Reproduces the published protocol.
``chat``
    The preset's few-shot pairs as alternating user/assistant turns, rendered
    with ``tokenizer.apply_chat_template``. The example *content* is identical to
    the completion preset — only the framing changes.

Both styles resolve the answer span by **tokenizing prefixes and taking their
lengths**, never by assuming a separator's token count. The scored span for an
answer is always ``[len(prefix_ids), len(full_ids))``, which is correct for any
tokenizer.
"""

from __future__ import annotations

import logging
import re

from .presets import preset_map

logger = logging.getLogger(__name__)

VALID_STYLES = ("chat", "completion", "auto")

# Presets whose text is a run of `Q:`/`A:` pairs, optionally after an instruction
# preamble. Only these can be rendered as chat turns without inventing content.
# `chat`/`long` are persona role-play formats (`Sam4621:`/`Alex1083:`, a blog
# post) whose framing *is* the manipulation under test, so they stay completions.
_QA_SHAPED_PRESETS = ("qa", "help", "harm", "null")

_QA_PAIR_RE = re.compile(r"^Q: (?P<q>.*?)\nA: (?P<a>.*)$", re.DOTALL)


def resolve_style(style: str, tokenizer, preset: str) -> str:
    """Resolve `auto`, and downgrade `chat` where it cannot apply.

    Returns `chat` or `completion`. Downgrades are logged, and the resolved value
    is recorded in the results file, so a run is always traceable to the prompt
    format that produced it.
    """
    if style not in VALID_STYLES:
        raise ValueError(f"Unsupported prompt_style {style!r}; choose from {list(VALID_STYLES)}")

    has_template = getattr(tokenizer, "chat_template", None) is not None

    if style == "auto":
        style = "chat" if has_template else "completion"
        logger.info("prompt_style=auto resolved to %r (chat_template present: %s)", style, has_template)
        return _check_preset(style, preset)

    if style == "chat" and not has_template:
        logger.warning(
            "prompt_style=chat requested but the tokenizer has no chat template "
            "(base, non-instruct model?); falling back to 'completion'."
        )
        return "completion"

    return _check_preset(style, preset)


def _check_preset(style: str, preset: str) -> str:
    if style == "chat" and preset not in _QA_SHAPED_PRESETS:
        logger.warning(
            "preset=%r is a role-play/completion format that has no faithful chat "
            "rendering; falling back to prompt_style='completion'.",
            preset,
        )
        return "completion"
    return style


def _parse_preset(preset: str) -> tuple[str | None, list[tuple[str, str]]]:
    """Split a Q:/A:-shaped preset into `(preamble, [(question, answer), ...])`.

    The pairs are parsed out of the preset text rather than restated here, so the
    chat and completion styles are guaranteed to carry identical example content.
    """
    if preset == "null":
        return None, []

    text = preset_map[preset]
    blocks = text.split("\n\n")

    preamble = None
    if not blocks[0].startswith("Q: "):
        # e.g. PROF_PRIMER's "Professor Smith was given the following instructions: ..."
        preamble = blocks.pop(0).strip()

    pairs = []
    for block in blocks:
        match = _QA_PAIR_RE.match(block.strip())
        if match is None:
            raise ValueError(f"Preset {preset!r} has a block that is not a Q:/A: pair: {block!r}")
        pairs.append((match.group("q").strip(), match.group("a").strip()))

    return preamble, pairs


def _chat_messages(question: str, preset: str) -> list[dict[str, str]]:
    preamble, pairs = _parse_preset(preset)

    messages: list[dict[str, str]] = []
    if preamble:
        messages.append({"role": "system", "content": preamble})
    for shot_q, shot_a in pairs:
        messages.append({"role": "user", "content": shot_q})
        messages.append({"role": "assistant", "content": shot_a})
    messages.append({"role": "user", "content": question.strip()})
    return messages


def _completion_context(question: str, preset: str) -> str:
    """The original prompt, up to and including the `A:` answer cue.

    Mirrors `utilities.format_prompt` (which stops before the cue) plus the
    `'\\nA:'` that `format_prompt_with_answer_strings` appends.
    """
    if preset == "null":
        return "Q: " + question + "\n\nA:"
    return "".join([preset_map[preset], "\n\nQ: ", question, "\nA:"])


def context_ids(question: str, preset: str, tokenizer, style: str) -> list[int]:
    """Token ids of the prompt, ending exactly where the model's answer begins.

    `style` must already be resolved (see `resolve_style`).
    """
    if style == "chat":
        return tokenizer.apply_chat_template(
            _chat_messages(question, preset), add_generation_prompt=True, tokenize=True
        )
    return tokenizer(_completion_context(question, preset)).input_ids


def answer_span_ids(question: str, answer: str, preset: str, tokenizer, style: str) -> tuple[list[int], list[int]]:
    """`(prefix_ids, full_ids)` for scoring `answer` under the MC metrics.

    The answer occupies `full_ids[len(prefix_ids):]`, so the caller scores that
    span without ever assuming how many tokens a separator takes. In completion
    style the prefix is the context plus the `A:` cue and the answer text carries
    its own leading space, exactly reproducing the original tokenization.
    """
    prefix = context_ids(question, preset, tokenizer, style)

    if style == "chat":
        full = prefix + tokenizer(answer.strip(), add_special_tokens=False).input_ids
    else:
        full = tokenizer(_completion_context(question, preset) + " " + answer.strip()).input_ids

    return prefix, full
