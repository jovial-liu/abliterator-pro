from __future__ import annotations

import gc
import hashlib
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from abliterator.adapters.base import ArchitectureReport, ProjectionTarget
from abliterator.artifacts import DirectionArtifact
from abliterator.config import ArchitectureSpec, ModelLoadConfig
from abliterator.datasets import PromptRecord
from abliterator.errors import ConfigurationError
from abliterator.evaluation import EvaluationReport, ResponseResult
from abliterator.workflow import (
    ObliterateConfig,
    _validate_permanent_safety_result,
    automatic_candidate_layers,
    default_obliterated_output,
    obliteration_plan,
    run_obliteration_workflow,
    select_ranked_direction,
    verify_reloaded_targets,
)


def _architecture() -> ArchitectureReport:
    return ArchitectureReport(
        model_type="qwen3",
        checkpoint_revision=None,
        tokenizer_revision=None,
        architectures=["FakeForCausalLM"],
        layer_path="model.layers",
        layer_count=2,
        hidden_size=2,
        attention_targets=2,
        mlp_targets=0,
        expert_targets=0,
        packed_targets=0,
        capture_supported=True,
        weight_edit_supported=True,
    )


def test_default_output_for_local_model_is_a_sibling(tmp_path: Path) -> None:
    source = tmp_path / "Qwen3-4B"
    source.mkdir()
    assert default_obliterated_output(str(source)) == tmp_path / "Qwen3-4B-obliterated"


def test_default_output_for_hub_model_uses_repository_leaf(tmp_path: Path) -> None:
    assert default_obliterated_output("Qwen/Qwen3-4B", cwd=tmp_path) == (
        tmp_path / "Qwen3-4B-obliterated"
    )


def test_plan_does_not_create_output_parent(tmp_path: Path) -> None:
    source = tmp_path / "Qwen3-4B"
    source.mkdir()
    output = tmp_path / "not-created" / "Qwen3-4B-obliterated"

    plan = obliteration_plan(
        ModelLoadConfig(model=str(source), dtype="float32", device_map=None),
        ObliterateConfig(output=output),
    )

    assert plan["output"] == str(output)
    assert not output.parent.exists()


def test_obliterate_config_defaults_to_refusal_removal(tmp_path: Path) -> None:
    config = ObliterateConfig(output=tmp_path / "fork")

    assert config.objective == "refusal-removal"


def test_high_level_workflow_rejects_zero_strength(tmp_path: Path) -> None:
    source = tmp_path / "Qwen3-4B"
    source.mkdir()
    with pytest.raises(ConfigurationError, match="greater than zero"):
        obliteration_plan(
            ModelLoadConfig(model=str(source), dtype="float32", device_map=None),
            ObliterateConfig(output=tmp_path / "fork", strength=0.0),
        )


def test_workflow_wrapper_releases_models_and_retains_failure_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / ".direction.safetensors"
    snapshot.write_bytes(b"diagnostic")
    state: dict[str, object] = {}
    releases: list[bool] = []

    class Payload:
        pass

    def fail(*args: object) -> dict[str, object]:
        resources = args[-1]
        resources.adapter = Payload()
        resources.artifact_path = snapshot
        state["ref"] = weakref.ref(resources.adapter)
        raise RuntimeError("simulated failure")

    monkeypatch.setattr("abliterator.workflow._execute_obliteration_workflow", fail)
    monkeypatch.setattr(
        "abliterator.workflow._release_accelerator_memory",
        lambda: releases.append(True),
    )
    with pytest.raises(RuntimeError, match="simulated failure"):
        run_obliteration_workflow(
            ModelLoadConfig(model="unused"),
            ArchitectureSpec(),
            ObliterateConfig(output=tmp_path / "unused"),
        )
    gc.collect()
    assert state["ref"]() is None  # type: ignore[operator]
    assert snapshot.read_bytes() == b"diagnostic"
    assert releases


def test_automatic_candidate_layers_focus_on_the_middle_of_the_stack() -> None:
    assert automatic_candidate_layers(36, count=3) == (12, 18, 24)
    assert automatic_candidate_layers(2, count=3) == (0, 1)
    assert automatic_candidate_layers(1, count=1) == (0,)


def test_guarded_selection_rejects_measured_safety_loss() -> None:
    candidates = [
        {
            "direction": "layer.12.resid_pre",
            "source_layer": 12,
            "score": 0.5,
            "false_refusal_gain": 0.5,
            "safety_loss": 0.1,
        },
        {
            "direction": "layer.18.resid_pre",
            "source_layer": 18,
            "score": 0.2,
            "false_refusal_gain": 0.2,
            "safety_loss": 0.0,
        },
    ]
    selected, warnings = select_ranked_direction(candidates, objective="guarded")
    assert selected["direction"] == "layer.18.resid_pre"
    assert not any("safety loss" in warning for warning in warnings)


def test_selection_requires_explicit_override_when_every_candidate_loses_safety() -> None:
    candidates = [
        {
            "direction": "layer.12.resid_pre",
            "source_layer": 12,
            "score": -0.1,
            "false_refusal_gain": 0.0,
            "safety_loss": 0.1,
        }
    ]
    with pytest.raises(ConfigurationError, match="--allow-safety-loss"):
        select_ranked_direction(candidates, objective="guarded")
    selected, warnings = select_ranked_direction(
        candidates,
        objective="guarded",
        allow_safety_loss=True,
    )
    assert selected["direction"] == "layer.12.resid_pre"
    assert any("safety loss" in warning for warning in warnings)


def test_selection_records_a_tie_and_no_measured_benefit() -> None:
    candidates = [
        {
            "direction": "layer.12.resid_pre",
            "source_layer": 12,
            "score": 0.0,
            "false_refusal_gain": 0.0,
            "safety_loss": 0.0,
        },
        {
            "direction": "layer.18.resid_pre",
            "source_layer": 18,
            "score": 0.0,
            "false_refusal_gain": 0.0,
            "safety_loss": 0.0,
        },
    ]
    selected, warnings = select_ranked_direction(candidates, objective="guarded")
    assert selected["direction"] == "layer.12.resid_pre"
    assert any("tie" in warning for warning in warnings)
    assert any("no measured false-refusal improvement" in warning for warning in warnings)


def test_guarded_selection_requires_an_informative_harmful_baseline() -> None:
    candidates = [
        {
            "direction": "layer.18.resid_pre",
            "source_layer": 18,
            "score": 0.0,
            "false_refusal_gain": 0.0,
            "safety_loss": 0.0,
        }
    ]
    with pytest.raises(ConfigurationError, match="zero baseline refusals"):
        select_ranked_direction(
            candidates,
            objective="guarded",
            baseline_deny_refusal_rate=0.0,
        )
    _, warnings = select_ranked_direction(
        candidates,
        objective="guarded",
        allow_safety_loss=True,
        baseline_deny_refusal_rate=0.0,
    )
    assert any("no measurable safety signal" in warning for warning in warnings)


def test_refusal_removal_selection_maximizes_measured_refusal_reduction() -> None:
    candidates = [
        {
            "direction": "layer.12.resid_pre",
            "source_layer": 12,
            "score": -1.5,
            "false_refusal_gain": 0.0,
            "safety_loss": 0.75,
            "refusal_reduction": 0.75,
        },
        {
            "direction": "layer.18.resid_pre",
            "source_layer": 18,
            "score": 0.8,
            "false_refusal_gain": 0.8,
            "safety_loss": 0.25,
            "refusal_reduction": 0.25,
        },
    ]

    selected, warnings = select_ranked_direction(
        candidates,
        objective="refusal-removal",
    )

    assert selected["direction"] == "layer.12.resid_pre"
    assert selected["refusal_reduction"] == pytest.approx(0.75)
    assert not warnings


def test_refusal_removal_selection_rejects_zero_measured_improvement() -> None:
    candidates = [
        {
            "direction": "layer.12.resid_pre",
            "source_layer": 12,
            "score": 0.0,
            "false_refusal_gain": 0.0,
            "safety_loss": 0.0,
            "refusal_reduction": 0.0,
        },
        {
            "direction": "layer.18.resid_pre",
            "source_layer": 18,
            "score": 0.0,
            "false_refusal_gain": 0.0,
            "safety_loss": 0.0,
            "refusal_reduction": 0.0,
        },
    ]

    with pytest.raises(ConfigurationError, match="reduced harmful-prompt refusals"):
        select_ranked_direction(candidates, objective="refusal-removal")


def test_refusal_removal_prefers_the_verified_candidate_subspace_on_a_tie() -> None:
    candidates = [
        {
            "direction": "layer.16.resid_pre",
            "direction_keys": ["layer.16.resid_pre"],
            "direction_rank": 1,
            "source_layer": 16,
            "score": 0.9,
            "false_refusal_gain": 0.2,
            "safety_loss": 0.9,
            "refusal_reduction": 0.9,
        },
        {
            "direction": "subspace[layer.8.resid_pre,layer.12.resid_pre,layer.16.resid_pre]",
            "direction_keys": [
                "layer.8.resid_pre",
                "layer.12.resid_pre",
                "layer.16.resid_pre",
            ],
            "direction_rank": 3,
            "source_layer": 8,
            "score": 0.9,
            "false_refusal_gain": 0.2,
            "safety_loss": 0.9,
            "refusal_reduction": 0.9,
        },
    ]

    selected, _ = select_ranked_direction(
        candidates,
        objective="refusal-removal",
        baseline_deny_refusal_rate=0.9,
    )

    assert selected["direction_rank"] == 3
    assert len(selected["direction_keys"]) == 3


def test_permanent_safety_guard_catches_hook_to_weight_divergence() -> None:
    baseline = _evaluation()
    permanent = _evaluation()
    permanent.deny_refused = 0

    with pytest.raises(ConfigurationError, match="permanently edited checkpoint"):
        _validate_permanent_safety_result(
            baseline,
            permanent,
            objective="guarded",
            allow_safety_loss=False,
        )
    receipt = _validate_permanent_safety_result(
        baseline,
        permanent,
        objective="guarded",
        allow_safety_loss=True,
    )
    assert receipt["override_used"] is True
    assert receipt["safety_loss"] == pytest.approx(1.0)


def test_permanent_refusal_removal_requires_a_measured_refusal_drop() -> None:
    baseline = _evaluation()
    unchanged = _evaluation(changed=True)

    with pytest.raises(ConfigurationError, match="did not reduce harmful-prompt refusals"):
        _validate_permanent_safety_result(
            baseline,
            unchanged,
            objective="refusal-removal",
            allow_safety_loss=False,
        )

    improved = _evaluation(changed=True)
    improved.deny_refused = 0
    receipt = _validate_permanent_safety_result(
        baseline,
        improved,
        objective="refusal-removal",
        allow_safety_loss=False,
    )

    assert receipt["passed"] is True
    assert receipt["objective"] == "refusal-removal"
    assert receipt["refusal_reduction"] == pytest.approx(1.0)


def test_permanent_refusal_removal_rejects_a_partial_drop() -> None:
    baseline = _evaluation()
    baseline.deny_total = 10
    baseline.deny_refused = 9
    partial = _evaluation(changed=True)
    partial.deny_total = 10
    partial.deny_refused = 4

    with pytest.raises(ConfigurationError, match="still refused"):
        _validate_permanent_safety_result(
            baseline,
            partial,
            objective="refusal-removal",
            allow_safety_loss=False,
        )


class _TargetAdapter:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight
        self.layers = [SimpleNamespace(), SimpleNamespace()]

    def inspect(self) -> ArchitectureReport:
        return _architecture()

    def projection_targets(
        self,
        layers: object = None,
        kinds: object = None,
    ) -> list[ProjectionTarget]:
        assert layers == [0]
        assert kinds == ("attention",)
        return [
            ProjectionTarget(
                name="model.layers.0.self_attn.o_proj",
                layer=0,
                kind="attention",
                weight=self.weight,
                output_axis=0,
            )
        ]


def _edit_manifest() -> dict[str, object]:
    return {
        "architecture": _architecture().to_dict(),
        "targets": [
            {
                "name": "model.layers.0.self_attn.o_proj",
                "layer": 0,
                "kind": "attention",
                "shape": [2, 2],
                "output_axis": 0,
                "leakage_before": 2**-0.5,
                "leakage_after": 0.0,
                "weight_l2_after": 1.0,
            }
        ],
    }


def test_reloaded_target_verification_detects_unmodified_weights() -> None:
    adapter = _TargetAdapter(torch.eye(2))
    with pytest.raises(ConfigurationError, match="direction leakage"):
        verify_reloaded_targets(
            adapter,
            _edit_manifest(),
            torch.tensor([1.0, 0.0]),
            chunk_columns=2,
        )


def test_reloaded_target_verification_accepts_the_persisted_projection() -> None:
    adapter = _TargetAdapter(torch.tensor([[0.0, 0.0], [0.0, 1.0]]))
    receipt = verify_reloaded_targets(
        adapter,
        _edit_manifest(),
        torch.tensor([1.0, 0.0]),
        chunk_columns=2,
    )
    assert receipt["passed"] is True
    assert receipt["target_count"] == 1
    assert receipt["max_direction_leakage"] == pytest.approx(0.0)


def test_reloaded_target_verification_rejects_an_unmodified_partial_projection() -> None:
    manifest = _edit_manifest()
    target = manifest["targets"][0]  # type: ignore[index]
    target["leakage_after"] = 0.9 * 2**-0.5  # type: ignore[index]
    target["weight_l2_after"] = (1.0 + 0.9**2) ** 0.5  # type: ignore[index]
    adapter = _TargetAdapter(torch.eye(2))
    with pytest.raises(ConfigurationError, match="does not match"):
        verify_reloaded_targets(
            adapter,
            manifest,
            torch.tensor([1.0, 0.0]),
            chunk_columns=2,
        )


def _record(identifier: str, behavior: str) -> PromptRecord:
    return PromptRecord(
        id=identifier,
        prompt=f"prompt-{identifier}",
        cohort="cyber",
        expected_behavior=behavior,
        domain="cybersecurity",
        category=None,
        source_id="fake-source",
        source_revision="a" * 40,
        source_row_id=identifier,
        group_id=identifier,
        license_spdx="MIT",
        commercial_ok=True,
        gated=False,
        use_for_direction=True,
        use_for_eval=True,
    )


def _evaluation(*, changed: bool = False) -> EvaluationReport:
    results = [
        ResponseResult(
            prompt_id="allow",
            source_id="fake-source",
            expected_behavior="allow",
            refused=False,
            response_chars=2,
            response_sha256="allow-new" if changed else "allow-old",
        ),
        ResponseResult(
            prompt_id="deny",
            source_id="fake-source",
            expected_behavior="deny",
            refused=True,
            response_chars=2,
            response_sha256="deny-new" if changed else "deny-old",
        ),
    ]
    return EvaluationReport(
        total=2,
        allow_total=1,
        allow_refused=0,
        deny_total=1,
        deny_refused=1,
        unscored_total=0,
        capability_total=0,
        capability_correct=0,
        average_response_chars=2.0,
        results=results,
    )


def test_fake_one_command_workflow_verifies_before_atomic_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "source-obliterated"
    events: list[str] = []
    architecture = _architecture()
    adapter = SimpleNamespace(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        layers=[SimpleNamespace(), SimpleNamespace()],
        serialization_components=(),
        inspect=lambda: architecture,
    )
    verifier = SimpleNamespace(
        model=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        layers=[SimpleNamespace(), SimpleNamespace()],
        serialization_components=(),
        inspect=lambda: architecture,
    )

    def fake_from_pretrained(config: ModelLoadConfig, **_: object) -> object:
        if Path(config.model).resolve() == source.resolve():
            events.append("load-source")
            return adapter
        events.append("reload-stage")
        return verifier

    monkeypatch.setattr(
        "abliterator.workflow.HuggingFaceAdapter",
        SimpleNamespace(from_pretrained=fake_from_pretrained),
    )
    target = [_record("target-1", "deny"), _record("target-2", "deny")]
    contrast = [_record("contrast-1", "allow"), _record("contrast-2", "allow")]
    ranking_records = [_record("allow", "allow"), _record("deny", "deny")]
    monkeypatch.setattr(
        "abliterator.workflow._prepare_direction_records",
        lambda _: (target, contrast),
    )
    monkeypatch.setattr(
        "abliterator.workflow._prepare_ranking_records",
        lambda _: ranking_records,
    )

    def fake_smoke(model_adapter: object, **_: object) -> dict[str, object]:
        events.append("smoke-source" if model_adapter is adapter else "smoke-stage")
        return {
            "architecture": architecture.to_dict(),
            "forward_capture": {"passed": True},
        }

    monkeypatch.setattr("abliterator.workflow.inspect_and_smoke", fake_smoke)

    artifact = DirectionArtifact(
        directions={"layer.1.resid_pre": torch.tensor([1.0, 0.0])},
        metadata={"local_checkpoint_receipt": None},
    )

    def fake_discover(*_: object, **__: object) -> DirectionArtifact:
        events.append("discover")
        return artifact

    monkeypatch.setattr("abliterator.workflow.discover_direction_artifact", fake_discover)
    monkeypatch.setattr(
        "abliterator.workflow.validate_direction_artifact_compatibility",
        lambda *_args, **_kwargs: [],
    )
    baseline = _evaluation()
    hooked = _evaluation(changed=True)

    def fake_rank(*_: object, **__: object) -> tuple[dict[str, object], object, dict[str, object]]:
        events.append("rank")
        return (
            {
                "format": "abpro-direction-ranking-v1",
                "baseline": baseline.to_dict(include_rows=True),
                "candidates": [
                    {
                        "direction": "layer.1.resid_pre",
                        "source_layer": 1,
                        "score": 0.0,
                        "false_refusal_gain": 0.0,
                        "safety_loss": 0.0,
                    }
                ],
            },
            baseline,
            {"layer.1.resid_pre": hooked},
        )

    monkeypatch.setattr("abliterator.workflow.rank_direction_candidates", fake_rank)
    manifest: dict[str, object] = {
        "architecture": architecture.to_dict(),
        "layers": [0],
        "kinds": ["attention"],
        "targets": [
            {
                "name": "model.layers.0.self_attn.o_proj",
                "layer": 0,
                "kind": "attention",
                "shape": [2, 2],
                "output_axis": 0,
                "leakage_before": 0.5,
                "leakage_after": 0.0,
                "weight_l2_after": 1.0,
            }
        ],
    }

    def fake_save(*args: object, **kwargs: object) -> dict[str, object]:
        events.append("apply-stage")
        destination = Path(args[2])
        destination.mkdir()
        artifact_path = Path(str(kwargs["direction_artifact_path"]))
        manifest["direction_artifact_sha256"] = hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()
        (destination / "model.safetensors").write_bytes(b"fake")
        (destination / "abliterator_manifest.json").write_text(json.dumps(manifest))
        return manifest

    monkeypatch.setattr("abliterator.workflow.save_ablated_checkpoint", fake_save)

    def fake_verify(*_: object, **__: object) -> dict[str, object]:
        events.append("verify-leakage")
        return {
            "passed": True,
            "target_count": 1,
            "max_direction_leakage": 0.0,
            "mean_direction_leakage": 0.0,
            "architecture": architecture.to_dict(),
        }

    monkeypatch.setattr("abliterator.workflow.verify_reloaded_targets", fake_verify)

    def fake_evaluate(*_: object, **__: object) -> EvaluationReport:
        events.append("post-export-evaluate")
        return hooked

    monkeypatch.setattr("abliterator.workflow.evaluate_prompts", fake_evaluate)

    def fake_receipt(path: str, **_: object) -> dict[str, object] | None:
        if Path(path).resolve() == source.resolve():
            return None
        return {
            "path": str(Path(path).resolve()),
            "kind": "directory-content-v1",
            "metadata_files": {},
            "shards": [{"name": "model.safetensors", "size": 4, "sha256": "fake"}],
            "content_sha256": "output-receipt",
        }

    monkeypatch.setattr("abliterator.workflow.local_checkpoint_receipt", fake_receipt)

    result = run_obliteration_workflow(
        ModelLoadConfig(model=str(source), dtype="float32", device_map=None),
        ArchitectureSpec(),
        ObliterateConfig(
            output=output,
            objective="guarded",
            direction_samples=2,
            ranking_samples=1,
            candidate_count=1,
            batch_size=1,
            max_length=64,
            max_new_tokens=4,
            kinds=("attention",),
        ),
    )

    assert output.is_dir()
    assert result["output"] == str(output)
    assert (output / "abpro" / "direction.safetensors").is_file()
    assert (output / "abpro" / "ranking.json").is_file()
    assert (output / "abpro" / "post-export-validation.json").is_file()
    assert (output / "abpro" / "run.json").is_file()
    run_receipt = json.loads((output / "abpro" / "run.json").read_text())
    assert run_receipt["output_checkpoint_receipt"]["path"] == "."
    assert run_receipt["output_checkpoint_receipt"]["content_sha256"] == "output-receipt"
    assert events == [
        "load-source",
        "smoke-source",
        "discover",
        "rank",
        "apply-stage",
        "reload-stage",
        "verify-leakage",
        "smoke-stage",
        "post-export-evaluate",
    ]
