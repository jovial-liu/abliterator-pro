from __future__ import annotations

import pytest
import torch

from abliterator.adapters.base import ArchitectureReport
from abliterator.artifacts import (
    DirectionArtifact,
    load_direction_artifact,
    local_checkpoint_receipt,
    save_direction_artifact,
    validate_direction_artifact_compatibility,
)
from abliterator.errors import ConfigurationError


def test_direction_artifact_round_trip(tmp_path) -> None:
    path = tmp_path / "direction.safetensors"
    artifact = DirectionArtifact(
        directions={"layer.2.resid_pre": torch.tensor([3.0, 4.0])},
        metadata={"model": "unit-test"},
    )
    digest = save_direction_artifact(path, artifact)
    loaded = load_direction_artifact(path)
    assert len(digest) == 64
    assert loaded.metadata == {"model": "unit-test"}
    torch.testing.assert_close(loaded.directions["layer.2.resid_pre"], torch.tensor([0.6, 0.8]))


def report(*, model_type: str = "qwen3", revision: str | None = "a" * 40) -> ArchitectureReport:
    return ArchitectureReport(
        model_type=model_type,
        checkpoint_revision=revision,
        tokenizer_revision=revision,
        architectures=["FakeForCausalLM"],
        layer_path="model.layers",
        layer_count=2,
        hidden_size=2,
        attention_targets=2,
        mlp_targets=2,
        expert_targets=0,
        packed_targets=0,
        capture_supported=True,
        weight_edit_supported=True,
    )


def compatible_artifact(model: str = "organization/model") -> DirectionArtifact:
    revision = "a" * 40
    return DirectionArtifact(
        directions={"layer.1.resid_pre": torch.tensor([1.0, 0.0])},
        metadata={
            "model": model,
            "revision": revision,
            "tokenizer": model,
            "capture": {"point": "resid_pre"},
            "prompt_rendering": {"raw_prompts": False, "enable_thinking": False},
            "architecture": report(revision=revision).to_dict(),
        },
    )


def test_artifact_provenance_rejects_wrong_model_and_allows_explicit_override() -> None:
    artifact = compatible_artifact()
    with pytest.raises(ConfigurationError, match="source model"):
        validate_direction_artifact_compatibility(
            artifact,
            report(),
            model="other/model",
            requested_revision="a" * 40,
            direction_keys=["layer.1.resid_pre"],
            intervention_point="resid_pre",
        )
    warnings = validate_direction_artifact_compatibility(
        artifact,
        report(),
        model="other/model",
        requested_revision="a" * 40,
        direction_keys=["layer.1.resid_pre"],
        intervention_point="resid_pre",
        allow_mismatch=True,
    )
    assert any("source model" in warning for warning in warnings)


def test_artifact_provenance_rejects_different_decoder_path() -> None:
    artifact = compatible_artifact()
    loaded = report()
    loaded.layer_path = "model.language_model.layers"
    with pytest.raises(ConfigurationError, match="layer_path"):
        validate_direction_artifact_compatibility(
            artifact,
            loaded,
            model="organization/model",
            requested_revision="a" * 40,
            direction_keys=["layer.1.resid_pre"],
        )


def test_artifact_provenance_rejects_prompt_rendering_mismatch() -> None:
    artifact = compatible_artifact()
    with pytest.raises(ConfigurationError, match="enable_thinking"):
        validate_direction_artifact_compatibility(
            artifact,
            report(),
            model="organization/model",
            requested_revision="a" * 40,
            direction_keys=["layer.1.resid_pre"],
            enable_thinking=True,
        )


def test_same_local_checkpoint_does_not_require_a_hub_revision(tmp_path) -> None:
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type":"qwen3"}\n')
    (checkpoint / "model.safetensors").write_bytes(b"test-weights")
    artifact = compatible_artifact(str(checkpoint))
    artifact.metadata["revision"] = None
    artifact.metadata["architecture"]["checkpoint_revision"] = None
    artifact.metadata["architecture"]["tokenizer_revision"] = None
    artifact.metadata["local_checkpoint_receipt"] = local_checkpoint_receipt(str(checkpoint))
    artifact.metadata["local_tokenizer_receipt"] = local_checkpoint_receipt(str(checkpoint))
    assert not validate_direction_artifact_compatibility(
        artifact,
        report(revision=None),
        model=str(checkpoint),
        requested_revision=None,
        direction_keys=["layer.1.resid_pre"],
    )


def test_artifact_metadata_must_be_an_object() -> None:
    artifact = DirectionArtifact(
        directions={"layer.0.resid_pre": torch.tensor([1.0, 0.0])},
        metadata=[],  # type: ignore[arg-type]
    )
    with pytest.raises(ConfigurationError, match="JSON object"):
        artifact.validate()


def test_artifact_save_refuses_to_overwrite(tmp_path) -> None:
    path = tmp_path / "direction.safetensors"
    artifact = compatible_artifact()
    save_direction_artifact(path, artifact)
    with pytest.raises(ConfigurationError, match="overwrite"):
        save_direction_artifact(path, artifact)
