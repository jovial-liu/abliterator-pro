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


def test_parser_exposes_portable_gguf_export() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "export",
            "--model",
            "/models/half-bliterated",
            "--quantization",
            "Q4_K_M",
        ]
    )

    assert args.command == "export"
    assert args.gguf_quantization == "q4_k_m"
    assert args.output is None
    assert args.context_length == 8192


def test_parser_exposes_one_command_obliteration_with_optional_output() -> None:
    parser = build_parser()
    args = parser.parse_args(["obliterate", "--model", "organization/model"])
    assert args.command == "obliterate"
    assert args.output is None
    assert args.candidate_layers == "auto"
    assert args.direction_samples == 40
    assert args.ranking_samples == 10
    assert args.objective == "refusal-removal"
    assert args.progress == "auto"


def test_obliterate_parser_exposes_plain_json_and_disabled_progress() -> None:
    parser = build_parser()

    assert (
        parser.parse_args(
            ["obliterate", "--model", "organization/model", "--progress", "plain"]
        ).progress
        == "plain"
    )
    assert (
        parser.parse_args(
            ["obliterate", "--model", "organization/model", "--progress", "json"]
        ).progress
        == "json"
    )
    assert (
        parser.parse_args(["obliterate", "--model", "organization/model", "--no-progress"]).progress
        == "off"
    )


def test_obliterate_parser_can_explicitly_select_guarded_objective() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["obliterate", "--model", "organization/model", "--objective", "guarded"]
    )

    assert args.objective == "guarded"


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
