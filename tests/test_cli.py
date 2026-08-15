from __future__ import annotations

import pytest

from abliterator.cli import _parse_layers, build_parser, command_apply
from abliterator.errors import ConfigurationError


def test_layer_parser() -> None:
    assert _parse_layers("1,3:6") == [1, 3, 4, 5]
    assert _parse_layers("all") is None


def test_layer_parser_rejects_reverse_range() -> None:
    with pytest.raises(ConfigurationError):
        _parse_layers("5:3")


def test_layer_parser_wraps_non_numeric_input() -> None:
    with pytest.raises(ConfigurationError, match="Invalid layer"):
        _parse_layers("one:two")


def test_parser_exposes_expected_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["datasets", "--json"])
    assert args.command == "datasets"


def test_apply_validates_quantized_output_before_loading_model(tmp_path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "apply",
            "--model",
            "organization/model",
            "--directions",
            str(tmp_path / "missing.safetensors"),
            "--direction",
            "layer.1.resid_pre",
            "--output",
            str(tmp_path / "dense"),
            "--output-quantization",
            "bnb-fp4",
        ]
    )
    with pytest.raises(ConfigurationError, match="--quantized-output"):
        command_apply(args)
