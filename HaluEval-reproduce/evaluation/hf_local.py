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

import inspect
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

# Default generation budget per prompt format, chosen to match the OpenAI endpoint
# that each format stands in for in the original script:
#
#   concat        -> openai.Completion.create(...), whose own default max_tokens is 16.
#   chat_template -> openai.ChatCompletion.create(...), which the original calls with
#                    ONLY temperature=0.0 — no max_tokens, so no cap at all.
#
# The cap is not cosmetic. `evaluate.py` scores a judgement it cannot parse as
# INCORRECT, so a judge cut off mid-sentence ("Based on the given answers, it does not
# contain hallucin|") is counted as a wrong answer rather than an unfinished one. A
# 16-token cap on the chat format therefore penalises verbose-but-correct judges that
# the published protocol would have scored fine — measured on this project's
# checkpoints, going 16 -> 64 halved the unparseable rate on the same rows.
#
# 128 rather than literally uncapped: 10,000 rows x 3 splits makes an unbounded budget
# impractical, and a judge that has not said Yes/No within 128 tokens is not truncated,
# it is refusing. Raise it for reasoning models that think before answering.
DEFAULT_MAX_NEW_TOKENS = {"chat_template": 128, "concat": 16}

# The spellings `verdict_logprobs` pools for each verdict: the word the judge prompts ask for,
# in title, lower and upper case, bare or after one space. Which variant is a model's first
# answer token depends on the prompt format: after a chat template's assistant header it is
# "Yes", after the flat prompt's "#Your Judgement#:" it is " Yes". They are matched against
# each vocabulary token's DECODED text (`_resolve_verdict_tokens`), so every token id is
# pooled once however many spellings decode to it.
VERDICT_SPELLINGS = {
    "yes": ("Yes", " Yes", "yes", " yes", "YES", " YES"),
    "no": ("No", " No", "no", " no", "NO", " NO"),
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


def _chat_template_ids(encoded):
    """Normalize `apply_chat_template(tokenize=True)` output to a flat list of token ids.

    The return type is stack-dependent: transformers 4.41 (the HPC pin) hands back a plain
    list of ids, while 5.x returns a BatchEncoding. Taking `len()` of the latter yields the
    number of KEYS — 2 — which is a silent, uniform "length" that would make every batch
    look identical to the length sorter while still working correctly on the HPC stack.
    """
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    # A single conversation can come back either flat or wrapped in a batch dimension.
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return encoded


def _resolve_verdict_tokens(tokenizer) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    """Token ids, and their raw vocabulary tokens, of every single token spelling each verdict.

    Scans the vocabulary instead of encoding each spelling. `encode` collapses spellings onto
    one id wherever the tokenizer inserts its own word-boundary marker (SentencePiece encodes
    "Yes" and " Yes" both as "▁Yes"), and it never reaches the unmarked piece "Yes" that
    follows other text. A token counts when its decoded text is exactly one of
    `VERDICT_SPELLINGS`, so every id is pooled once, and "Yesterday", "no." or "\\nno" are not.
    The raw tokens ("ĠYes", "▁Yes") are what gets recorded: unlike decoded text, they are
    unique per id.
    """
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    found: dict[str, dict[int, str]] = {verdict: {} for verdict in VERDICT_SPELLINGS}
    for token, token_id in tokenizer.get_vocab().items():
        if token_id in special:
            continue
        # Cheap prefilter before decoding: ASCII letters keep their own characters in the raw
        # token of every HF vocabulary (only a bytes-keyed vocabulary is decoded unfiltered).
        if isinstance(token, str) and "yes" not in token.lower() and "no" not in token.lower():
            continue
        text = tokenizer.decode([token_id])
        for verdict, spellings in VERDICT_SPELLINGS.items():
            if text in spellings:
                found[verdict][int(token_id)] = token if isinstance(token, str) else text
    ids = {verdict: sorted(tokens) for verdict, tokens in found.items()}
    return ids, {verdict: [found[verdict][i] for i in ids[verdict]] for verdict in found}


def _forward_accepts(model, name: str) -> bool:
    try:
        return name in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _is_peft_adapter(model_path: str) -> bool:
    """True if `model_path` is a local directory holding a PEFT adapter checkpoint.

    A LoRA run (e.g. `training=dpo peft=lora`) saves only `adapter_config.json` +
    adapter weights, not a full model — `AutoModelForCausalLM.from_pretrained`
    can't load that directory directly.
    """
    return (Path(model_path) / "adapter_config.json").is_file()


def _is_hf_hub_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def _reraise_if_offline_cache_miss(model_id: str, exc: OSError) -> None:
    """Turn huggingface_hub's offline-cache-miss `OSError` into an actionable one.

    On this project's HPC setup (see README.md "Mirror / offline"), compute nodes
    export `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` and have no internet access, so
    any Hub id not already pre-downloaded into the shared cache on a login node fails
    here with a generic connection/404-shaped error. Re-raise with the actual fix.
    """
    if _looks_like_local_path(model_id) or not _is_hf_hub_offline():
        return
    raise OSError(
        f"'{model_id}' is not in the local Hugging Face cache and this node is offline "
        "(HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1). Pre-download it on a node with internet "
        f"access first, e.g.:\n    huggingface-cli download {model_id}\n"
        "then re-run on the compute node. See the 'Mirror / offline' section in README.md."
    ) from exc


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
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_id, cache_dir=cache_dir, torch_dtype=torch_dtype, device_map=device_map
            )
        except OSError as exc:
            _reraise_if_offline_cache_miss(model_id, exc)
            raise

    from peft import PeftConfig, PeftModel

    try:
        adapter_config = PeftConfig.from_pretrained(model_id)
    except OSError as exc:
        _reraise_if_offline_cache_miss(model_id, exc)
        raise
    resolved_base_id = base_model_id or adapter_config.base_model_name_or_path
    _check_local_path_exists(resolved_base_id, what="Base model")
    logger.info("Detected PEFT adapter at %s; loading base model %s", model_id, resolved_base_id)

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            resolved_base_id, cache_dir=cache_dir, torch_dtype=torch_dtype, device_map=device_map
        )
    except OSError as exc:
        _reraise_if_offline_cache_miss(resolved_base_id, exc)
        raise
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
        max_new_tokens: int | None = None,
        batch_size: int = 1,
    ) -> None:
        if dtype not in _DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(_DTYPE_BY_NAME)}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_new_tokens is not None and max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

        # Resolved below, once the tokenizer tells us which prompt format applies.
        self._requested_max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        logger.info("Loading judge model %s (dtype=%s, device_map=%s)", model_id, dtype, device_map)
        model = _load_causal_lm(model_id, base_model_id, cache_dir, _DTYPE_BY_NAME[dtype], device_map)

        # A checkpoint produced by this project's training pipeline always has its
        # own tokenizer saved alongside it (adapter or full); `tokenizer_id` is an
        # escape hatch for checkpoints (e.g. hand-merged weights) that don't.
        resolved_tokenizer_id = tokenizer_id or model_id
        _check_local_path_exists(resolved_tokenizer_id, what="Tokenizer")
        try:
            tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_id, cache_dir=cache_dir)
        except OSError as exc:
            _reraise_if_offline_cache_miss(resolved_tokenizer_id, exc)
            raise
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Batched generation on a decoder-only model REQUIRES left padding, or short
        # prompts in a batch start decoding from pad tokens. Harmless at batch_size=1.
        tokenizer.padding_side = "left"

        self.tokenizer = tokenizer
        # Resolved once. Only `verdict_logprobs` uses them, and it is what raises when a
        # tokenizer has no single-token verdict: a generate-only run never needs one.
        self.verdict_token_ids, self._verdict_spellings = _resolve_verdict_tokens(tokenizer)
        # The judge prompt is rendered with the model's own chat template, so the wire
        # format matches what it was tuned on. A base (non-instruct) model has no
        # template; `generate` falls back to a plain concatenation rather than failing.
        self._has_chat_template = getattr(tokenizer, "chat_template", None) is not None
        if not self._has_chat_template:
            logger.warning(
                "Tokenizer for %s has no chat template (base, non-instruct model?); "
                "falling back to plain concatenation of the message contents.",
                resolved_tokenizer_id,
            )

        # The generation budget depends on which prompt format we just settled on,
        # so it can only be resolved here (see DEFAULT_MAX_NEW_TOKENS).
        self.max_new_tokens = (
            self._requested_max_new_tokens
            if self._requested_max_new_tokens is not None
            else DEFAULT_MAX_NEW_TOKENS[self.prompt_format]
        )
        logger.info(
            "Judge prompt format: %s; max_new_tokens=%d%s",
            self.prompt_format, self.max_new_tokens,
            "" if self._requested_max_new_tokens is not None else " (default for this format)",
        )

        self._generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map=device_map,
        )

    @property
    def prompt_format(self) -> str:
        """'chat_template' or 'concat' — recorded alongside results for provenance."""
        return "chat_template" if self._has_chat_template else "concat"

    def prompt_token_lengths(self, requests: list[tuple[list[dict[str, str]], str]]) -> list[int]:
        """Tokenized length of each judge prompt, rendered as `generate_many` renders it.

        The sort key for length-bucketed batching, so it IS the quantity the pipeline pads
        to — a proxy (characters, words) would mis-order rows whose text tokenizes densely.
        Both branches below mirror `generate_many` line for line, INCLUDING the choice
        between `messages` and `completion_prompt`; if that method's rendering changes, this
        one must change with it or batches get bucketed on the wrong length.

        Costs one CPU tokenizer pass over the split — seconds against a multi-hour GPU run.
        """
        return [len(ids) for ids in self.prompt_token_ids(requests)]

    def prompt_token_ids(self, requests: list[tuple[list[dict[str, str]], str]]) -> list[list[int]]:
        """Token ids of each judge prompt, rendered as `generate_many` renders it.

        Shared by `prompt_token_lengths` (the sort key) and `verdict_logprobs` (the scored
        prefill), so the length the batches are cut on and the prompt the constrained
        scorer reads are the same tokens.
        """
        if not requests:
            return []

        if not self._has_chat_template:
            prompts = [completion_prompt for _, completion_prompt in requests]
            encoded = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
            return [list(ids) for ids in encoded]

        # `apply_chat_template` is per-conversation, so this cannot be batched. The
        # add_generation_prompt=True matches what TextGenerationPipeline applies internally
        # when it is handed message lists.
        return [
            list(
                _chat_template_ids(
                    self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                )
            )
            for messages, _ in requests
        ]

    def generate(self, messages: list[dict[str, str]], completion_prompt: str) -> str:
        """Generate a judgement (ideally 'Yes'/'No') for one judge prompt.

        Takes both renderings of the same judgement request, and picks by what the
        model supports: `messages` (chat turns) when the tokenizer has a chat
        template, else `completion_prompt` -- HaluEval's own flat prompt string, the
        one it sends to the legacy `openai.Completion` engines. Both are built
        side by side in `evaluate.get_*_response`; passing the benchmark's existing
        string keeps a base model on a format HaluEval actually defines, rather than
        one this wrapper invented.

        Greedy decoding; the caller normalizes the returned text to Yes/No.
        """
        return self.generate_many([(messages, completion_prompt)])[0]

    def generate_many(self, requests: list[tuple[list[dict[str, str]], str]]) -> list[str]:
        """Judge a batch of `(messages, completion_prompt)` pairs in one forward pass.

        Returned in input order. HaluEval's three splits are 10,000 rows each, so at
        `batch_size=1` this loop is dominated by per-call overhead rather than by the
        16 tokens it actually generates — batching is where the wall-clock goes.

        Prompt construction is untouched, so this changes throughput, not the protocol:
        on the pinned stack (torch 2.2.2 / transformers 4.41) batch sizes 1/4/8 reproduce
        the unbatched judgements byte-for-byte over ragged prompt lengths, in both prompt
        formats. Padding could still flip a greedy argmax tie on some other model, so keep
        `batch_size` fixed across the models you compare (it is recorded in the summary).
        """
        if not requests:
            return []

        if not self._has_chat_template:
            prompts = [completion_prompt for _, completion_prompt in requests]
            outputs = self._generator(
                prompts, max_new_tokens=self.max_new_tokens, do_sample=False,
                return_full_text=False, batch_size=self.batch_size,
            )
            return [out[0]["generated_text"].strip() for out in outputs]

        # Passing message lists makes the pipeline apply the tokenizer's template.
        chats = [messages for messages, _ in requests]
        outputs = self._generator(
            chats, max_new_tokens=self.max_new_tokens, do_sample=False,
            batch_size=self.batch_size,
        )
        return [out[0]["generated_text"][-1]["content"].strip() for out in outputs]

    # --- Constrained (prefill-only) scoring ---------------------------------------

    @property
    def verdict_tokens(self) -> dict[str, list[str]]:
        """The raw vocabulary tokens `verdict_logprobs` pools, one per id — recorded in the constrained summary."""
        return {verdict: list(texts) for verdict, texts in self._verdict_spellings.items()}

    @torch.no_grad()
    def last_token_logprobs(self, requests: list[tuple[list[dict[str, str]], str]]) -> torch.Tensor:
        """Float32 log-softmax over the vocabulary at each prompt's last position.

        One left-padded forward pass over the whole batch, on RAW logits: the pipeline's
        logits processors (the checkpoint's `repetition_penalty`, for one) are deliberately
        not applied, so this is the model's own next-token distribution. Returned on CPU,
        one row per request, in input order.
        """
        model = self._generator.model
        sequences = self.prompt_token_ids(requests)
        width = max(len(ids) for ids in sequences)
        input_ids = torch.full((len(sequences), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for row, ids in enumerate(sequences):
            input_ids[row, width - len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, width - len(ids):] = 1
        # generate() derives positions from the mask; a plain forward does not, so without
        # this every left-padded prompt would be read at shifted positions.
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        kwargs = {}
        # Only the last position's logits are needed; newer transformers can skip the rest.
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if _forward_accepts(model, name):
                kwargs[name] = 1
                break
        device = model.device
        logits = model(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
            position_ids=position_ids.to(device), use_cache=False, **kwargs,
        ).logits
        return torch.log_softmax(logits[:, -1, :].float(), dim=-1).cpu()

    def verdict_logprobs(self, requests: list[tuple[list[dict[str, str]], str]]) -> list[tuple[float, float]]:
        """``(logP("Yes"), logP("No"))`` for the judge's first answer token, per request.

        Each verdict's probability is pooled (logsumexp) over every vocabulary token spelling it
        (`VERDICT_SPELLINGS`), each id once, so a model that answers " Yes" or "YES" is not
        scored as if it put no mass on "Yes". Nothing is decoded: this is one prefill, and the judgement is
        the argmax of the two, scored threshold-free by AUROC downstream.
        """
        if not requests:
            return []
        yes_ids, no_ids = self.verdict_token_ids["yes"], self.verdict_token_ids["no"]
        if not yes_ids or not no_ids:
            raise ValueError(
                "constrained scoring needs a single-token spelling of both verdicts, but this "
                f"tokenizer has yes={self.verdict_tokens['yes']} no={self.verdict_tokens['no']} "
                f"(candidates: {VERDICT_SPELLINGS})"
            )
        logprobs = self.last_token_logprobs(requests)
        yes = torch.logsumexp(logprobs[:, yes_ids], dim=-1)
        no = torch.logsumexp(logprobs[:, no_ids], dim=-1)
        return [(float(y), float(n)) for y, n in zip(yes.tolist(), no.tolist())]
