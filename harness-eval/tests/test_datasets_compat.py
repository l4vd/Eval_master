"""Tests for the datasets<4 List/LargeList compatibility shim."""

import dataclasses

import pytest

from harness_eval._datasets_compat import install_list_feature_compat


def test_install_is_safe_without_datasets():
    """Calling the shim never raises, even if datasets is absent."""
    install_list_feature_compat()  # no exception is the assertion


def test_shim_maps_new_list_types_to_sequence():
    """On datasets<4 a List/LargeList column deserialises as a Sequence."""
    _feat = pytest.importorskip("datasets.features.features")
    if dataclasses.is_dataclass(getattr(_feat, "List", None)):
        pytest.skip("datasets>=4 reads List natively; shim is a no-op")

    install_list_feature_compat()
    assert getattr(_feat, "_harness_list_compat", False) is True

    for new_type in ("List", "LargeList"):
        feature = _feat.generate_from_dict(
            {"feature": {"dtype": "string", "_type": "Value"}, "_type": new_type}
        )
        assert isinstance(feature, _feat.Sequence)
        assert isinstance(feature.feature, _feat.Value)
