"""Unit tests for model.id -> lm_eval model_args translation (no heavy imports)."""

import json

import pytest

from harness_eval.model import (
    _looks_like_local_path,
    build_model_args,
    model_args_to_string,
)


def _write_adapter(directory, base="base/model", with_tokenizer=False):
    (directory / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base} if base else {}), encoding="utf-8"
    )
    if with_tokenizer:
        (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def test_hub_id_maps_to_pretrained_only():
    args = build_model_args("Qwen/Qwen2.5-0.5B-Instruct")
    assert args["pretrained"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "peft" not in args
    assert "tokenizer" not in args


def test_hub_id_with_slash_is_not_treated_as_local_path():
    assert _looks_like_local_path("org/repo") is False


def test_local_full_model_dir_maps_to_pretrained_only(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    args = build_model_args(str(tmp_path))
    assert args["pretrained"] == str(tmp_path)
    assert "peft" not in args


def test_adapter_dir_maps_base_to_pretrained_and_adapter_to_peft(tmp_path):
    _write_adapter(tmp_path, base="base/model")
    args = build_model_args(str(tmp_path))
    assert args["pretrained"] == "base/model"
    assert args["peft"] == str(tmp_path)


def test_explicit_base_model_id_overrides_adapter_config(tmp_path):
    _write_adapter(tmp_path, base="base/model")
    args = build_model_args(str(tmp_path), base_model_id="other/base")
    assert args["pretrained"] == "other/base"


def test_adapter_without_base_anywhere_raises(tmp_path):
    _write_adapter(tmp_path, base=None)
    with pytest.raises(ValueError, match="--base-model-id"):
        build_model_args(str(tmp_path))


def test_adapter_tokenizer_defaults_to_adapter_dir_not_base(tmp_path):
    # Regression: lm_eval's native peft= loads the *base* tokenizer, but the
    # sibling contract uses the checkpoint's own tokenizer.
    _write_adapter(tmp_path, base="base/model", with_tokenizer=True)
    args = build_model_args(str(tmp_path))
    assert args["tokenizer"] == str(tmp_path)


def test_adapter_without_tokenizer_files_falls_back_to_base(tmp_path):
    _write_adapter(tmp_path, base="base/model", with_tokenizer=False)
    args = build_model_args(str(tmp_path))
    assert "tokenizer" not in args  # omitted -> lm_eval uses pretrained (the base)


def test_explicit_tokenizer_id_wins(tmp_path):
    _write_adapter(tmp_path, base="base/model", with_tokenizer=True)
    args = build_model_args(str(tmp_path), tokenizer_id="my/tokenizer")
    assert args["tokenizer"] == "my/tokenizer"


def test_missing_local_path_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        build_model_args("/nonexistent/checkpoint")


def test_none_values_are_omitted():
    args = build_model_args("Qwen/Qwen2.5-0.5B-Instruct", cache_dir=None)
    assert "cache_dir" not in args
    assert "trust_remote_code" not in args


def test_cache_dir_included_when_given():
    args = build_model_args("Qwen/Qwen2.5-0.5B-Instruct", cache_dir="/data/hf")
    assert args["cache_dir"] == "/data/hf"


def test_dtype_always_present():
    assert build_model_args("gpt2", dtype="float16")["dtype"] == "float16"


def test_model_args_string_round_trips_simple_values():
    text = model_args_to_string({"pretrained": "gpt2", "dtype": "float16"})
    assert text == "pretrained=gpt2,dtype=float16"


def test_model_args_string_rejects_value_containing_comma():
    # lm_eval's arg-string parser is a bare split(","); a comma is unrepresentable.
    with pytest.raises(ValueError, match="comma"):
        model_args_to_string({"pretrained": "/path/with,comma"})
