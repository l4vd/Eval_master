"""Let ``datasets<4`` read cache metadata written by ``datasets>=4``.

``datasets>=4`` serialises list-valued columns with the newer ``List`` /
``LargeList`` feature types. Under the pinned ``datasets<4`` (see
``pyproject-HPC.toml``) those names resolve to :data:`typing.List` etc. —
non-dataclasses — so deserialising such a cached ``dataset_info.json`` (or the
equivalent Arrow schema metadata) dies with
``TypeError: must be called with a dataclass type or instance``.

lm_eval loads its task datasets (e.g. TruthfulQA) straight from the HF
``datasets`` cache at run time, so it trips over the exact same incompatibility
that :mod:`faitheval.data` guards against. :func:`install_list_feature_compat`
wraps the feature deserialiser to rewrite those ``_type``s to the equivalent
legacy ``Sequence``, which the installed version understands. It is a no-op on
``datasets>=4`` (where ``List`` reads natively) and safe to call more than once.
"""

from __future__ import annotations

import dataclasses
import logging

logger = logging.getLogger(__name__)


def install_list_feature_compat() -> None:
    """Patch ``datasets`` so ``List``/``LargeList`` metadata reads as ``Sequence``."""
    try:
        from datasets.features import features as _feat
    except Exception:  # pragma: no cover - datasets missing or internals moved
        return
    if getattr(_feat, "_harness_list_compat", False):
        return
    generate_from_dict = getattr(_feat, "generate_from_dict", None)
    if generate_from_dict is None or not hasattr(_feat, "Sequence"):
        return
    # True no-op on datasets>=4, where `List` is already a real feature dataclass.
    if dataclasses.is_dataclass(getattr(_feat, "List", None)):
        return

    def _patched(obj):
        # Rewrite List/LargeList -> Sequence; the wrapped original recurses
        # through this same patched module global, so nested list columns work.
        if isinstance(obj, dict) and obj.get("_type") in ("List", "LargeList"):
            obj = {**obj, "_type": "Sequence"}
        return generate_from_dict(obj)

    _feat.generate_from_dict = _patched
    _feat._harness_list_compat = True
    logger.debug("Installed List/LargeList -> Sequence datasets compatibility shim.")
