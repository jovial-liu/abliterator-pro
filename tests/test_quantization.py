from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from abliterator.config import QuantizationConfig
from abliterator.errors import ConfigurationError, QuantizationError
from abliterator.quantization import (
    build_transformers_quantization,
    checkpoint_quantization_kind,
    quantize_saved_checkpoint,
)


def test_nested_text_checkpoint_quantization_is_detected() -> None:
    config = SimpleNamespace(
        text_config=SimpleNamespace(quantization_config={"quant_method": "compressed-tensors"})
    )
    assert checkpoint_quantization_kind(config) == "compressed-tensors"


def test_quantized_nested_source_is_rejected_before_export(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k25",
                "text_config": {"quantization_config": {"quant_method": "compressed-tensors"}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(QuantizationError, match="config.text_config.quantization_config"):
        quantize_saved_checkpoint(
            source,
            tmp_path / "output",
            QuantizationConfig(kind="bnb-fp4"),
        )


def test_fpquant_builder_uses_supported_bf16_backward_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeTransformers:
        @staticmethod
        def FPQuantConfig(**kwargs: object) -> object:
            calls.append(kwargs)
            return SimpleNamespace(**kwargs)

    def fake_require(name: str, extra: str | None = None) -> object:
        if name == "transformers":
            return FakeTransformers
        if name == "fp_quant":
            assert extra == "fpquant"
            return object()
        raise AssertionError(name)

    monkeypatch.setattr("abliterator.quantization.require", fake_require)
    result = build_transformers_quantization(
        QuantizationConfig(kind="nvfp4", pseudoquantization=True)
    )
    assert result.backward_dtype == "bf16"
    assert calls == [
        {
            "forward_dtype": "nvfp4",
            "backward_dtype": "bf16",
            "store_master_weights": False,
            "pseudoquantization": True,
        }
    ]


def test_backend_specific_options_are_not_silently_ignored() -> None:
    with pytest.raises(ConfigurationError, match="FP-Quant option"):
        QuantizationConfig(kind="bnb-fp4", pseudoquantization=True).validate()
    with pytest.raises(ConfigurationError, match="bitsandbytes option"):
        QuantizationConfig(kind="nvfp4", double_quant=True).validate()


def test_quantized_export_reloads_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    source = tmp_path / "dense"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "fake", "torch_dtype": "float32"}),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"dense-source")

    class FakeBnbLinear(torch.nn.Module):
        __module__ = "bitsandbytes.nn"

        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.zeros((2, 2), dtype=torch.uint8)

    class FakeQuantizedModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(model_type="unknown")
            self.packed = FakeBnbLinear()

        def named_modules(self):
            return [("", self), ("packed", self.packed)]

        def save_pretrained(self, path, **_: object) -> None:
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "fake",
                        "quantization_config": {
                            "quant_method": "bitsandbytes",
                            "bnb_4bit_quant_type": "fp4",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (path / "model.safetensors").write_bytes(b"packed-weights")

    class FakeSerializer:
        def save_pretrained(self, path) -> None:
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    load_calls: list[str] = []

    def fake_load(config):
        load_calls.append(config.model)
        return FakeQuantizedModel(), FakeSerializer(), config.model, None, (FakeSerializer(),)

    monkeypatch.setattr("abliterator.adapters.huggingface.load_pretrained_components", fake_load)
    output = tmp_path / "quantized"
    manifest = quantize_saved_checkpoint(
        source,
        output,
        QuantizationConfig(kind="bnb-fp4", compute_dtype="bfloat16"),
    )
    assert len(load_calls) == 2
    assert load_calls[0] == str(source.resolve())
    assert load_calls[1] != load_calls[0]
    assert load_calls[1].startswith(str(tmp_path / ".quantized.partial-"))
    assert manifest.verification is not None
    assert manifest.verification["reload_verified"] is True
    assert (output / "abliterator_quantization.json").is_file()
