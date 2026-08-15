from __future__ import annotations

import pytest

from abliterator.config import ArchitectureSpec, CaptureConfig, ModelLoadConfig, QuantizationConfig
from abliterator.errors import ConfigurationError


def test_remote_code_requires_pinned_revision() -> None:
    config = ModelLoadConfig(model="organization/model", trust_remote_code=True)
    with pytest.raises(ConfigurationError, match="commit SHA"):
        config.validate()


def test_remote_code_rejects_mutable_branch_name() -> None:
    config = ModelLoadConfig(model="organization/model", trust_remote_code=True, revision="main")
    with pytest.raises(ConfigurationError, match="commit SHA"):
        config.validate()


def test_explicit_remote_code_revision_is_accepted() -> None:
    config = ModelLoadConfig(
        model="organization/model",
        trust_remote_code=True,
        revision="0123456789abcdef0123456789abcdef01234567",
    )
    config.validate()


def test_quantization_names_are_unambiguous() -> None:
    for name in ("bnb-fp4", "bnb-nf4", "nvfp4", "mxfp4"):
        QuantizationConfig(kind=name).validate()
    with pytest.raises(ConfigurationError):
        QuantizationConfig(kind="fp4").validate()  # type: ignore[arg-type]


def test_capture_validation() -> None:
    with pytest.raises(ConfigurationError):
        CaptureConfig(batch_size=0).validate()


def test_architecture_spec_rejects_unknown_keys() -> None:
    with pytest.raises(ConfigurationError, match="Unknown architecture"):
        ArchitectureSpec.from_dict({"layer_path": "model.layers", "typo": True})


@pytest.mark.parametrize(
    "value",
    [
        {"hidden_size": "4096"},
        {"hidden_size": -1},
        {"include_experts": "false"},
        {"attention_output_patterns": "self_attn.o_proj"},
        {"mlp_output_patterns": [""]},
    ],
)
def test_architecture_spec_validates_override_types(value: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        ArchitectureSpec.from_dict(value)


def test_capture_config_rejects_unknown_point_and_pooling() -> None:
    with pytest.raises(ConfigurationError, match="point"):
        CaptureConfig(point="typo").validate()  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="pooling"):
        CaptureConfig(pooling="typo").validate()  # type: ignore[arg-type]


def test_separate_remote_code_tokenizer_requires_its_own_commit() -> None:
    config = ModelLoadConfig(
        model="organization/model",
        revision="a" * 40,
        tokenizer="organization/tokenizer",
        trust_remote_code=True,
    )
    with pytest.raises(ConfigurationError, match="tokenizer-revision"):
        config.validate()


def test_model_class_typo_is_rejected() -> None:
    config = ModelLoadConfig(model="organization/model", model_class="typo")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="model_class"):
        config.validate()
