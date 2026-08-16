"""Verified one-command orchestration for creating an ablated model fork."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import shutil
import sys
import uuid
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from .adapters.base import ProjectionKind
from .adapters.huggingface import HuggingFaceAdapter
from .artifacts import (
    DirectionArtifact,
    load_direction_artifact,
    local_checkpoint_receipt,
    save_direction_artifact,
    validate_direction_artifact_compatibility,
)
from .capture import ActivationCollector
from .config import ArchitectureSpec, CaptureConfig, ModelLoadConfig, QuantizationConfig
from .datasets import PromptRecord, get_dataset_spec, length_match, load_dataset_records
from .directions import direction_leakage_and_norm
from .errors import ConfigurationError, DatasetError
from .evaluation import EvaluationReport, RefusalDetector, compare_reports, evaluate_prompts
from .interventions import ResidualProjector, input_embedding_projection_target
from .pipeline import (
    cohort_receipt,
    discover_direction_artifact,
    environment_receipt,
    save_ablated_checkpoint,
)
from .quantization import dequantize_model_for_edit
from .runtime import publish_directory_noreplace, require

ObliterationObjective = Literal["refusal-removal", "guarded"]


@dataclass(frozen=True)
class ObliterateConfig:
    """Configuration for the objective-aware end-to-end dense fork workflow."""

    output: str | Path
    objective: ObliterationObjective = "refusal-removal"
    directions: str | Path | None = None
    direction_key: str | None = None
    target_dataset: str = "cyberseceval-mitre"
    contrast_dataset: str = "cyberseceval-frr"
    direction_samples: int = 40
    ranking_datasets: tuple[str, ...] = ("jbb-cyber-benign", "jbb-cyber-harmful")
    ranking_samples: int = 10
    seed: int = 42
    length_match: bool = True
    length_bin: int = 128
    candidate_layers: tuple[int, ...] | Literal["auto"] | None = "auto"
    candidate_count: int = 3
    point: Literal["resid_pre", "resid_post"] = "resid_pre"
    pooling: Literal["last", "last-k", "mean"] = "last"
    last_k: int = 1
    batch_size: int = 1
    max_length: int = 1024
    max_new_tokens: int = 64
    safety_penalty: float = 2.0
    allow_safety_loss: bool = False
    apply_layers: tuple[int, ...] | None = None
    kinds: tuple[ProjectionKind, ...] = ("attention", "mlp", "expert")
    strength: float = 1.0
    chunk_columns: int = 8192
    dequantize: bool = False

    def validate(self) -> None:
        if not str(self.output).strip():
            raise ConfigurationError("output must be a checkpoint directory")
        if self.objective not in {"refusal-removal", "guarded"}:
            raise ConfigurationError("objective must be refusal-removal or guarded")
        if self.direction_key is not None and self.directions is None:
            raise ConfigurationError("direction_key requires a directions artifact")
        if self.directions is not None and not Path(self.directions).expanduser().is_file():
            raise ConfigurationError(f"Direction artifact does not exist: {self.directions}")
        minimum_direction_samples = (
            20 if self.objective == "refusal-removal" and self.directions is None else 2
        )
        if self.direction_samples < minimum_direction_samples:
            raise ConfigurationError(
                f"direction_samples must be at least {minimum_direction_samples} for "
                f"objective={self.objective}"
            )
        if self.ranking_samples < 1:
            raise ConfigurationError("ranking_samples must be positive")
        if not self.ranking_datasets:
            raise ConfigurationError("ranking_datasets must not be empty")
        if self.length_bin < 16:
            raise ConfigurationError("length_bin must be at least 16")
        if self.candidate_count < 1:
            raise ConfigurationError("candidate_count must be positive")
        if self.candidate_layers != "auto" and self.candidate_layers is not None:
            _validate_layers(self.candidate_layers, "candidate_layers")
        if self.apply_layers is not None:
            _validate_layers(self.apply_layers, "apply_layers")
            if self.objective == "refusal-removal":
                raise ConfigurationError(
                    "refusal-removal requires applying the selected direction to all layers; "
                    "use the low-level apply command for partial-layer experiments"
                )
        CaptureConfig(
            point=self.point,
            pooling=self.pooling,
            last_k=self.last_k,
            batch_size=self.batch_size,
            max_length=self.max_length,
        ).validate()
        if self.max_new_tokens < 1:
            raise ConfigurationError("max_new_tokens must be positive")
        if not math.isfinite(self.safety_penalty) or self.safety_penalty < 0:
            raise ConfigurationError("safety_penalty must be finite and non-negative")
        if self.objective == "refusal-removal" and self.allow_safety_loss:
            raise ConfigurationError(
                "allow_safety_loss only applies to objective=guarded; refusal-removal already "
                "optimizes measured refusal reduction"
            )
        if not math.isfinite(self.strength) or not 0.0 < self.strength <= 1.0:
            raise ConfigurationError("strength must be finite, greater than zero, and at most 1")
        if self.chunk_columns < 1:
            raise ConfigurationError("chunk_columns must be positive")
        if not self.kinds or set(self.kinds) - {"attention", "mlp", "expert"}:
            raise ConfigurationError("kinds must contain only attention, mlp, and expert")


def _validate_layers(layers: Iterable[int], label: str) -> None:
    values = tuple(layers)
    if not values or len(set(values)) != len(values):
        raise ConfigurationError(f"{label} must contain unique layer indices")
    if any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 0 for layer in values):
        raise ConfigurationError(f"{label} must contain non-negative integers")


def default_obliterated_output(model: str, *, cwd: str | Path | None = None) -> Path:
    """Return a predictable, non-incrementing output path for a model source."""
    source = Path(model).expanduser()
    if source.exists() or source.is_absolute() or model.startswith(("./", "../", "~")):
        resolved = source.resolve()
        return resolved.parent / f"{resolved.name}-obliterated"
    leaf = model.rstrip("/").rsplit("/", 1)[-1]
    leaf = re.sub(r"[^A-Za-z0-9._-]+", "-", leaf).strip(".-")
    if not leaf:
        raise ConfigurationError("Could not derive an output name from the model identifier")
    root = Path.cwd() if cwd is None else Path(cwd)
    return root.resolve() / f"{leaf}-obliterated"


def automatic_candidate_layers(layer_count: int, *, count: int = 3) -> tuple[int, ...]:
    """Choose an evenly spaced set across the middle third of a decoder stack."""
    if layer_count < 1:
        raise ConfigurationError("Automatic layer selection requires a positive layer count")
    if count < 1:
        raise ConfigurationError("candidate_count must be positive")
    if layer_count <= count:
        return tuple(range(layer_count))
    if count == 1:
        return (layer_count // 2,)
    start = layer_count / 3
    stop = 2 * layer_count / 3
    values = {
        min(layer_count - 1, max(0, round(start + index * (stop - start) / (count - 1))))
        for index in range(count)
    }
    if len(values) != count:
        # This is possible only for very small stacks. Fill deterministically from the center.
        center_order = sorted(
            range(layer_count),
            key=lambda layer: (abs(layer - layer_count / 2), layer),
        )
        for layer in center_order:
            values.add(layer)
            if len(values) == count:
                break
    return tuple(sorted(values))


def select_ranked_direction(
    candidates: list[dict[str, Any]],
    *,
    objective: ObliterationObjective,
    allow_safety_loss: bool = False,
    baseline_deny_refusal_rate: float | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Select a direction according to the workflow's explicit behavioral objective."""
    if objective not in {"refusal-removal", "guarded"}:
        raise ConfigurationError("Unknown obliteration objective")
    if objective == "refusal-removal" and allow_safety_loss:
        raise ConfigurationError("allow_safety_loss is only valid for objective=guarded")
    if baseline_deny_refusal_rate is not None:
        if not math.isfinite(baseline_deny_refusal_rate) or not (
            0.0 <= baseline_deny_refusal_rate <= 1.0
        ):
            raise ConfigurationError("Baseline harmful-refusal rate is invalid")
        if baseline_deny_refusal_rate == 0:
            if objective == "refusal-removal":
                raise ConfigurationError(
                    "The harmful ranking cohort produced zero baseline refusals, so refusal "
                    "removal cannot be measured"
                )
            if not allow_safety_loss:
                raise ConfigurationError(
                    "The harmful ranking cohort produced zero baseline refusals, so guarded "
                    "selection has no safety-preservation signal; inspect the model/cohort or "
                    "pass --allow-safety-loss explicitly"
                )
    if not candidates:
        raise ConfigurationError("Direction ranking produced no candidates")
    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            score = float(candidate["score"])
            false_refusal_gain = float(candidate["false_refusal_gain"])
            safety_loss = float(candidate["safety_loss"])
            refusal_reduction = float(candidate.get("refusal_reduction", safety_loss))
            direction_rank = int(candidate.get("direction_rank", 1))
            source_layer = int(candidate["source_layer"])
            direction = str(candidate["direction"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError("Direction ranking returned an invalid candidate row") from exc
        if not all(
            math.isfinite(value)
            for value in (score, false_refusal_gain, safety_loss, refusal_reduction)
        ):
            raise ConfigurationError("Direction ranking returned a non-finite score")
        if direction_rank < 1:
            raise ConfigurationError("Direction ranking returned an invalid subspace rank")
        row = dict(candidate)
        row.update(
            score=score,
            false_refusal_gain=false_refusal_gain,
            safety_loss=safety_loss,
            refusal_reduction=refusal_reduction,
            direction_rank=direction_rank,
            source_layer=source_layer,
            direction=direction,
        )
        normalized.append(row)

    if objective == "refusal-removal":
        ranked = sorted(
            normalized,
            key=lambda row: (
                -row["direction_rank"],
                -row["refusal_reduction"],
                -row["false_refusal_gain"],
                row["source_layer"],
                row["direction"],
            ),
        )
        eligible = [row for row in ranked if row["refusal_reduction"] > 1e-12]
        if not eligible:
            raise ConfigurationError(
                "No candidate reduced harmful-prompt refusals; the workflow will not publish a "
                "behaviorally ineffective model"
            )
    else:
        ranked = sorted(
            normalized,
            key=lambda row: (-row["score"], row["source_layer"], row["direction"]),
        )
        eligible = (
            ranked if allow_safety_loss else [row for row in ranked if row["safety_loss"] <= 1e-12]
        )
        if not eligible:
            raise ConfigurationError(
                "Every candidate reduced harmful-prompt refusal preservation; review the "
                "ranking or pass --allow-safety-loss explicitly"
            )
    selected = eligible[0]
    warnings: list[str] = []
    tied = [
        row
        for row in eligible
        if math.isclose(row["refusal_reduction"], selected["refusal_reduction"], abs_tol=1e-12)
        and math.isclose(row["false_refusal_gain"], selected["false_refusal_gain"], abs_tol=1e-12)
        and (
            objective == "refusal-removal"
            or math.isclose(row["score"], selected["score"], abs_tol=1e-12)
        )
    ]
    if len(tied) > 1:
        warnings.append(
            "selection tie: candidates had identical measured objectives; the lowest source "
            "layer and key won deterministically"
        )
    if objective == "guarded" and selected["false_refusal_gain"] <= 0:
        warnings.append("no measured false-refusal improvement for the selected direction")
    if objective == "guarded" and selected["safety_loss"] > 0:
        warnings.append(
            "explicit override accepted measured harmful-refusal safety loss for the selected "
            "direction"
        )
    if objective == "guarded" and baseline_deny_refusal_rate == 0:
        warnings.append(
            "explicit override accepted a zero-refusal baseline with no measurable safety signal"
        )
    if objective == "refusal-removal" and selected["direction_rank"] > 1:
        warnings.append(
            "selected the full behavior-active candidate subspace; permanent replay is the "
            "authoritative refusal-removal check"
        )
    return selected, warnings


def _validate_permanent_safety_result(
    baseline: EvaluationReport,
    permanent: EvaluationReport,
    *,
    objective: ObliterationObjective,
    allow_safety_loss: bool,
) -> dict[str, Any]:
    before = baseline.deny_refusal_rate
    after = permanent.deny_refusal_rate
    if before is None or after is None:
        raise ConfigurationError(
            "Permanent-fork verification requires a scored harmful-prompt cohort"
        )
    refusal_reduction = before - after
    safety_loss = max(0.0, refusal_reduction)
    if objective == "refusal-removal":
        if refusal_reduction <= 1e-12:
            raise ConfigurationError(
                "The permanently edited checkpoint did not reduce harmful-prompt refusals; "
                "the staged model will not be published"
            )
        if after > 1e-12:
            raise ConfigurationError(
                "The permanently edited checkpoint still refused harmful holdout prompts "
                f"(rate={after:.6g}); objective=refusal-removal requires a measured rate of "
                "zero before publication"
            )
        return {
            "passed": True,
            "objective": objective,
            "override_used": False,
            "baseline_deny_refusal_rate": before,
            "permanent_deny_refusal_rate": after,
            "refusal_reduction": refusal_reduction,
            "required_max_refusal_rate": 0.0,
            "safety_loss": safety_loss,
        }
    if objective != "guarded":
        raise ConfigurationError("Unknown obliteration objective")
    if safety_loss > 1e-12 and not allow_safety_loss:
        raise ConfigurationError(
            "The permanently edited checkpoint reduced harmful-prompt refusal preservation "
            f"by {safety_loss:.6g}, even though its reversible hook passed; the staged model "
            "will not be published unless --allow-safety-loss is explicit"
        )
    return {
        "passed": safety_loss <= 1e-12,
        "objective": objective,
        "override_used": safety_loss > 1e-12 and allow_safety_loss,
        "baseline_deny_refusal_rate": before,
        "permanent_deny_refusal_rate": after,
        "safety_loss": safety_loss,
        "refusal_reduction": refusal_reduction,
    }


def verify_reloaded_targets(
    adapter: Any,
    edit_manifest: dict[str, Any],
    direction: Any,
    *,
    chunk_columns: int = 8192,
) -> dict[str, Any]:
    """Verify the staged checkpoint retained the exact edited targets and low leakage."""
    expected_architecture = edit_manifest.get("architecture")
    expected_targets = edit_manifest.get("targets")
    if not isinstance(expected_architecture, dict) or not isinstance(expected_targets, list):
        raise ConfigurationError("Edit manifest lacks architecture or target records")
    loaded_architecture = adapter.inspect().to_dict()
    for field in ("model_type", "layer_path", "layer_count", "hidden_size"):
        if loaded_architecture.get(field) != expected_architecture.get(field):
            raise ConfigurationError(
                f"Reloaded checkpoint architecture mismatch for {field}: "
                f"expected {expected_architecture.get(field)!r}, "
                f"received {loaded_architecture.get(field)!r}"
            )
    if not expected_targets:
        raise ConfigurationError("Edit manifest contains no projection targets")
    try:
        layers = sorted({int(row["layer"]) for row in expected_targets if int(row["layer"]) >= 0})
        kinds = tuple(
            sorted({str(row["kind"]) for row in expected_targets if row["kind"] != "embedding"})
        )
        expected_by_name = {str(row["name"]): row for row in expected_targets}
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError("Edit manifest contains invalid target records") from exc
    if len(expected_by_name) != len(expected_targets):
        raise ConfigurationError("Edit manifest contains duplicate target names")
    loaded_targets = adapter.projection_targets(layers=layers, kinds=kinds) if kinds else []
    if any(row.get("kind") == "embedding" for row in expected_targets):
        loaded_targets.append(input_embedding_projection_target(adapter))
    loaded_by_name = {target.name: target for target in loaded_targets}
    if set(loaded_by_name) != set(expected_by_name):
        missing = sorted(set(expected_by_name) - set(loaded_by_name))
        extra = sorted(set(loaded_by_name) - set(expected_by_name))
        raise ConfigurationError(
            f"Reloaded checkpoint target mismatch; missing={missing}, extra={extra}"
        )

    leakages: list[float] = []
    target_checks: list[dict[str, Any]] = []
    for name, expected in expected_by_name.items():
        target = loaded_by_name[name]
        expected_shape = tuple(int(size) for size in expected.get("shape", ()))
        expected_axis = int(expected.get("output_axis", -1))
        if target.shape != expected_shape:
            raise ConfigurationError(
                f"Reloaded target {name} shape mismatch: expected {expected_shape}, "
                f"received {target.shape}"
            )
        if target.layer != int(expected["layer"]) or target.kind != str(expected["kind"]):
            raise ConfigurationError(f"Reloaded target {name} metadata does not match the edit")
        if target.output_axis != expected_axis:
            raise ConfigurationError(
                f"Reloaded target {name} output axis changed from {expected_axis} "
                f"to {target.output_axis}"
            )
        actual, actual_l2 = direction_leakage_and_norm(
            target.weight,
            direction,
            output_axis=target.output_axis,
            chunk_columns=chunk_columns,
        )
        expected_before = float(expected.get("leakage_before", math.inf))
        expected_after = float(expected.get("leakage_after", math.inf))
        expected_l2 = float(expected.get("weight_l2_after", math.inf))
        if not all(
            math.isfinite(value)
            for value in (actual, actual_l2, expected_before, expected_after, expected_l2)
        ):
            raise ConfigurationError(f"Reloaded target {name} has non-finite direction leakage")
        dtype_name = str(target.weight.dtype).casefold()
        leakage_tolerance = 5e-4 if "float16" in dtype_name or "bfloat16" in dtype_name else 5e-6
        leakage_tolerance += abs(expected_after) * 1e-4
        if not math.isclose(
            actual,
            expected_after,
            rel_tol=1e-4,
            abs_tol=leakage_tolerance,
        ):
            raise ConfigurationError(
                f"Reloaded target {name} direction leakage {actual:.6g} does not match the "
                f"saved value {expected_after:.6g} within tolerance {leakage_tolerance:.6g}"
            )
        if not math.isclose(actual_l2, expected_l2, rel_tol=1e-6, abs_tol=1e-8):
            raise ConfigurationError(
                f"Reloaded target {name} L2 norm {actual_l2:.6g} does not match the saved "
                f"value {expected_l2:.6g}"
            )
        leakages.append(actual)
        target_checks.append(
            {
                "name": name,
                "leakage_before": expected_before,
                "leakage_saved": expected_after,
                "leakage_reloaded": actual,
                "weight_l2_saved": expected_l2,
                "weight_l2_reloaded": actual_l2,
            }
        )
    return {
        "passed": True,
        "target_count": len(leakages),
        "max_direction_leakage": max(leakages),
        "mean_direction_leakage": sum(leakages) / len(leakages),
        "architecture": loaded_architecture,
        "targets": target_checks,
    }


def _jsonable_dataclass(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(asdict(value), default=str)))


def obliteration_plan(
    model: ModelLoadConfig,
    config: ObliterateConfig,
    architecture: ArchitectureSpec | None = None,
) -> dict[str, Any]:
    """Validate cheap inputs and return the side-effect-free execution plan."""
    model.validate()
    config.validate()
    architecture = architecture or ArchitectureSpec()
    architecture.validate()
    output = _preflight_output(model.model, config.output)
    if model.quantization.kind != "none" and not config.dequantize:
        raise ConfigurationError(
            "Packed input weights require --dequantize; canonical dense input is preferred"
        )
    return {
        "format": "abpro-obliterate-plan-v1",
        "model_load": _jsonable_dataclass(model),
        "architecture_override": _jsonable_dataclass(architecture),
        "workflow_config": _jsonable_dataclass(config),
        "source_model": model.model,
        "output": str(output),
        "direction_source": (
            "discover" if config.directions is None else str(Path(config.directions).resolve())
        ),
        "objective": config.objective,
        "target_dataset": config.target_dataset,
        "contrast_dataset": config.contrast_dataset,
        "direction_samples": config.direction_samples,
        "ranking_datasets": list(config.ranking_datasets),
        "ranking_samples_per_dataset": config.ranking_samples,
        "candidate_layers": (
            config.candidate_layers
            if isinstance(config.candidate_layers, str)
            else ("all" if config.candidate_layers is None else list(config.candidate_layers))
        ),
        "candidate_count": config.candidate_count,
        "apply_layers": "all" if config.apply_layers is None else list(config.apply_layers),
        "kinds": list(config.kinds),
        "strength": config.strength,
        "behavior_grounded_discovery": (
            config.objective == "refusal-removal" and config.directions is None
        ),
        "guarded_selection": config.objective == "guarded" and not config.allow_safety_loss,
        "reload_verification": True,
        "post_export_validation": True,
        "final_publish": "atomic-no-clobber",
    }


def _preflight_output(model: str, output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise ConfigurationError(f"Refusing to overwrite existing output path {destination}")
    source = Path(model).expanduser()
    if source.exists():
        source = source.resolve()
        if not source.is_dir():
            raise ConfigurationError(f"Local model source must be a directory: {source}")
        if destination == source or source in destination.parents:
            raise ConfigurationError(
                "Obliterated output must not be equal to or nested inside the source checkpoint"
            )
    return destination


def _direction_layer(key: str) -> int:
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] == "layer":
        try:
            layer = int(parts[1])
        except ValueError:
            layer = -1
        if layer >= 0:
            return layer
    raise ConfigurationError(f"Cannot infer a source layer from direction key {key!r}")


def _prepare_direction_records(
    config: ObliterateConfig,
) -> tuple[list[PromptRecord], list[PromptRecord]]:
    target_spec = get_dataset_spec(config.target_dataset)
    contrast_spec = get_dataset_spec(config.contrast_dataset)
    if not target_spec.use_for_direction or not contrast_spec.use_for_direction:
        raise DatasetError(
            "Direction discovery only accepts registry sources marked use_for_direction=true"
        )
    if config.objective == "refusal-removal":
        # The final cohort is behavior-confirmed after the model is loaded. Use the complete
        # pinned sources so a low refusal rate cannot silently turn a requested 40/40 run into
        # a tiny or label-only cohort.
        load_limit = None
    else:
        load_limit = (
            config.direction_samples * 3 if config.length_match else config.direction_samples
        )
    target = load_dataset_records(
        config.target_dataset,
        limit=load_limit,
        seed=config.seed,
    )
    contrast = load_dataset_records(
        config.contrast_dataset,
        limit=load_limit,
        seed=config.seed,
    )
    if config.length_match and config.objective != "refusal-removal":
        target, contrast = length_match(
            target,
            contrast,
            seed=config.seed,
            bin_width=config.length_bin,
        )
    if config.objective == "refusal-removal":
        if min(len(target), len(contrast)) < config.direction_samples:
            raise DatasetError(
                "Fewer candidate records are available than the requested behavior-grounded "
                "direction cohort"
            )
        return target, contrast
    count = min(config.direction_samples, len(target), len(contrast))
    if count < 2:
        raise DatasetError("Fewer than two matched records are available per direction cohort")
    return target[:count], contrast[:count]


def filter_direction_records_by_behavior(
    adapter: Any,
    target_candidates: list[PromptRecord],
    contrast_candidates: list[PromptRecord],
    *,
    desired_count: int,
    select_all_matches: bool = False,
    detector: RefusalDetector | None = None,
    batch_size: int = 4,
    max_new_tokens: int = 64,
    max_length: int = 1024,
) -> tuple[list[PromptRecord], list[PromptRecord], dict[str, Any]]:
    """Select direction cohorts grounded in the model's observed refusal behavior.

    Target rows qualify only when the model actually refuses them. Contrast rows qualify
    only when the model actually answers them. Selection preserves candidate order so a
    seeded source cohort produces the same result, and the receipt never stores prompt or
    response text.
    """
    if not isinstance(desired_count, int) or isinstance(desired_count, bool) or desired_count < 1:
        raise ConfigurationError("desired_count must be a positive integer")
    _validate_direction_behavior_labels(target_candidates, expected="deny", label="target")
    _validate_direction_behavior_labels(contrast_candidates, expected="allow", label="contrast")

    detector = detector or RefusalDetector()
    match_goal = desired_count * 3 if select_all_matches else desired_count
    target_rows = _evaluate_direction_behavior_candidates(
        adapter,
        target_candidates,
        match_goal=match_goal,
        required_behavior="refused",
        detector=detector,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
    )
    contrast_rows = _evaluate_direction_behavior_candidates(
        adapter,
        contrast_candidates,
        match_goal=match_goal,
        required_behavior="answered",
        detector=detector,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
    )
    target_matches = [index for index, row in enumerate(target_rows) if row["refused"]]
    contrast_matches = [index for index, row in enumerate(contrast_rows) if not row["refused"]]
    if len(target_matches) < desired_count or len(contrast_matches) < desired_count:
        raise DatasetError(
            "Insufficient behavior-grounded direction records: "
            f"required {desired_count} per cohort; "
            f"target refusals={len(target_matches)}/{len(target_candidates)}; "
            f"contrast answers={len(contrast_matches)}/{len(contrast_candidates)}"
        )

    selected_target_indices = (
        target_matches if select_all_matches else target_matches[:desired_count]
    )
    selected_contrast_indices = (
        contrast_matches if select_all_matches else contrast_matches[:desired_count]
    )
    selected_targets = [target_candidates[index] for index in selected_target_indices]
    selected_contrasts = [contrast_candidates[index] for index in selected_contrast_indices]
    receipt = {
        "format": "abpro-direction-behavior-filter-v1",
        "desired_count": desired_count,
        "selection_strategy": (
            "all-matching-candidates-in-input-order"
            if select_all_matches
            else "first-matching-candidate-in-input-order"
        ),
        "detector": {
            "class": type(detector).__name__,
            "patterns_sha256": hashlib.sha256(
                "\n".join(detector.patterns).encode("utf-8")
            ).hexdigest(),
        },
        "generation": {
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "max_length": max_length,
        },
        "target": _direction_behavior_cohort_receipt(
            target_rows,
            selected_target_indices,
            matching_count=len(target_matches),
            required_behavior="refused",
        ),
        "contrast": _direction_behavior_cohort_receipt(
            contrast_rows,
            selected_contrast_indices,
            matching_count=len(contrast_matches),
            required_behavior="answered",
        ),
    }
    return selected_targets, selected_contrasts, receipt


def _validate_direction_behavior_labels(
    records: list[PromptRecord],
    *,
    expected: Literal["allow", "deny"],
    label: str,
) -> None:
    invalid = [record.id for record in records if record.expected_behavior != expected]
    if invalid:
        raise ConfigurationError(
            f"{label} direction candidates must all have expected_behavior={expected!r}; "
            f"invalid IDs: {', '.join(invalid[:5])}"
        )
    ids = [record.id for record in records]
    if len(set(ids)) != len(ids):
        raise ConfigurationError(f"{label} direction candidate IDs must be unique")


def _evaluate_direction_behavior_candidates(
    adapter: Any,
    records: list[PromptRecord],
    *,
    match_goal: int,
    required_behavior: Literal["refused", "answered"],
    detector: RefusalDetector,
    batch_size: int,
    max_new_tokens: int,
    max_length: int,
) -> list[dict[str, Any]]:
    if not records:
        return []
    rows: list[dict[str, Any]] = []
    matching_count = 0
    effective_batch_size = max(1, batch_size)
    for start in range(0, len(records), effective_batch_size):
        batch = records[start : start + effective_batch_size]
        report = evaluate_prompts(
            adapter,
            batch,
            detector=detector,
            batch_size=effective_batch_size,
            max_new_tokens=max_new_tokens,
            max_length=max_length,
        )
        for record, result in zip(batch, report.results, strict=True):
            refused = bool(result.refused)
            rows.append(
                {
                    "prompt_id": record.id,
                    "source_id": record.source_id,
                    "prompt_sha256": hashlib.sha256(record.prompt.encode("utf-8")).hexdigest(),
                    "response_sha256": result.response_sha256,
                    "refused": refused,
                }
            )
            if (required_behavior == "refused" and refused) or (
                required_behavior == "answered" and not refused
            ):
                matching_count += 1
        print(
            "behavior filter "
            f"{required_behavior}: {matching_count}/{match_goal} matches "
            f"after {len(rows)}/{len(records)} checked",
            file=sys.stderr,
            flush=True,
        )
        if matching_count >= match_goal:
            break
    return rows


def _direction_behavior_cohort_receipt(
    rows: list[dict[str, Any]],
    selected_indices: list[int],
    *,
    matching_count: int,
    required_behavior: Literal["refused", "answered"],
) -> dict[str, Any]:
    return {
        "required_behavior": required_behavior,
        "candidate_count": len(rows),
        "matching_count": matching_count,
        "selected_count": len(selected_indices),
        "selected": [rows[index] for index in selected_indices],
        "evaluated": rows,
    }


def _prepare_ranking_records(config: ObliterateConfig) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for dataset_id in config.ranking_datasets:
        spec = get_dataset_spec(dataset_id)
        if not spec.use_for_eval or not spec.use_for_ranking:
            raise DatasetError(f"{dataset_id} is not registered for direction ranking")
        records.extend(
            load_dataset_records(
                dataset_id,
                limit=config.ranking_samples,
                seed=config.seed,
            )
        )
    if not any(record.expected_behavior == "allow" for record in records) or not any(
        record.expected_behavior == "deny" for record in records
    ):
        raise ConfigurationError("Automatic ranking requires both benign and harmful cohorts")
    return records


def inspect_and_smoke(adapter: Any, *, point: str = "resid_pre") -> dict[str, Any]:
    """Run structural inspection plus a real first/last-layer capture."""
    report = adapter.inspect()
    if not report.weight_edit_supported:
        raise ConfigurationError(
            "This checkpoint is not eligible for permanent dense weight editing; inspect the "
            "architecture report or use a canonical dense source"
        )
    smoke_layers = tuple(sorted({0, len(adapter.layers) - 1}))
    smoke = ActivationCollector(adapter).collect(
        ["Reply with the single word OK."],
        CaptureConfig(
            point=point,  # type: ignore[arg-type]
            layers=smoke_layers,
            batch_size=1,
            max_length=64,
        ),
    )
    return {
        "architecture": report.to_dict(),
        "forward_capture": {
            "passed": True,
            "layers": list(smoke_layers),
            "captured": sorted(smoke.means),
            "input_receipt": smoke.input_receipt,
        },
    }


def rank_direction_candidates(
    adapter: Any,
    artifact: DirectionArtifact,
    records: list[PromptRecord],
    direction_keys: list[str],
    *,
    layers: tuple[int, ...] | None,
    point: str,
    strength: float,
    objective: ObliterationObjective,
    safety_penalty: float,
    batch_size: int,
    max_new_tokens: int,
    max_length: int,
) -> tuple[dict[str, Any], EvaluationReport, dict[str, EvaluationReport]]:
    """Evaluate candidate hooks and return a deterministic, text-free ranking receipt."""
    if objective not in {"refusal-removal", "guarded"}:
        raise ConfigurationError("Unknown obliteration objective")
    if not direction_keys:
        raise ConfigurationError("No direction keys were selected for ranking")
    baseline = evaluate_prompts(
        adapter,
        records,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
    )
    baseline_false_refusal_rate = baseline.false_refusal_rate
    baseline_deny_refusal_rate = baseline.deny_refusal_rate
    if baseline_false_refusal_rate is None or baseline_deny_refusal_rate is None:
        raise ConfigurationError("Automatic ranking requires both benign and harmful metrics")
    modified_reports: dict[str, EvaluationReport] = {}
    ranked: list[dict[str, Any]] = []
    baseline_by_id = {row.prompt_id: row for row in baseline.results}

    def evaluate_candidate(
        label: str,
        directions: Any,
        source_keys: list[str],
    ) -> None:
        with ResidualProjector(
            adapter,
            directions,
            layers=layers,
            point=point,
            strength=strength,
        ):
            modified = evaluate_prompts(
                adapter,
                records,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                max_length=max_length,
            )
        modified_reports[label] = modified
        false_refusal_gain = (
            baseline_false_refusal_rate - modified.false_refusal_rate
            if modified.false_refusal_rate is not None
            else 0.0
        )
        refusal_reduction = (
            baseline_deny_refusal_rate - modified.deny_refusal_rate
            if modified.deny_refusal_rate is not None
            else 0.0
        )
        safety_loss = max(0.0, refusal_reduction)
        score = (
            refusal_reduction
            if objective == "refusal-removal"
            else false_refusal_gain - safety_penalty * safety_loss
        )
        changed = sum(
            baseline_by_id[row.prompt_id].response_sha256 != row.response_sha256
            for row in modified.results
        )
        ranked.append(
            {
                "direction": label,
                "direction_keys": source_keys,
                "direction_rank": len(source_keys),
                "source_layer": min(_direction_layer(key) for key in source_keys),
                "score": score,
                "false_refusal_gain": false_refusal_gain,
                "safety_loss": safety_loss,
                "refusal_reduction": refusal_reduction,
                "response_change_rate": changed / len(modified.results),
                "modified": modified.to_dict(include_rows=True),
            }
        )

    for key in direction_keys:
        evaluate_candidate(key, artifact.directions[key], [key])
    if objective == "refusal-removal" and len(direction_keys) > 1:
        composite_label = "subspace[" + ",".join(direction_keys) + "]"
        evaluate_candidate(
            composite_label,
            [artifact.directions[key] for key in direction_keys],
            list(direction_keys),
        )
    objective_receipt: dict[str, Any]
    if objective == "refusal-removal":
        ranked.sort(
            key=lambda row: (
                -row["refusal_reduction"],
                -row["false_refusal_gain"],
                -row["direction_rank"],
                row["source_layer"],
                row["direction"],
            )
        )
        objective_receipt = {
            "name": objective,
            "formula": "baseline_harmful_refusal_rate - modified_harmful_refusal_rate",
            "required_result": "zero modified harmful-refusal rate",
        }
    else:
        ranked.sort(key=lambda row: (-row["score"], row["source_layer"], row["direction"]))
        objective_receipt = {
            "name": objective,
            "safety_penalty": safety_penalty,
            "formula": "false_refusal_gain - safety_penalty*safety_loss",
            "required_result": "no harmful-refusal loss unless explicitly overridden",
        }
    report = {
        "format": "abpro-direction-ranking-v1",
        "baseline": baseline.to_dict(include_rows=True),
        "objective": objective_receipt,
        "candidates": ranked,
        "cohort": cohort_receipt(records),
    }
    return report, baseline, modified_reports


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        torch = require("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _write_json_new(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _copy_file_new(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _render_model_card(receipt: dict[str, Any]) -> str:
    selected = receipt["selection"]
    validation = receipt["post_export_validation"]
    selection_mode = str(receipt["plan"]["objective"])
    return (
        "---\n"
        "library_name: transformers\n"
        "tags:\n"
        "  - ab-pro\n"
        "  - abliteration\n"
        "---\n\n"
        f"# {receipt['output_name']}\n\n"
        "This checkpoint is an experimental dense AB Pro fork. It was created by the "
        f"{selection_mode} one-command workflow. Its refusal result applies only to the "
        "recorded evaluation cohort and is not proof of universal non-refusal or safe model "
        "behavior.\n\n"
        "## Receipt\n\n"
        f"- Source: `{receipt['source_model']}`\n"
        f"- Selected direction: `{selected['direction']}`\n"
        f"- Strength: `{receipt['edit']['strength']}`\n"
        f"- Edited targets: `{receipt['edit']['target_count']}`\n"
        f"- Reload verification: `{receipt['reload_verification']['passed']}`\n"
        f"- Harmful refusal rate before: `{validation['baseline']['deny_refusal_rate']}`\n"
        f"- Harmful refusal rate after: `{validation['modified']['deny_refusal_rate']}`\n"
        f"- Benign false-refusal rate before: "
        f"`{validation['baseline']['false_refusal_rate']}`\n"
        f"- Benign false-refusal rate after: "
        f"`{validation['modified']['false_refusal_rate']}`\n\n"
        "See `abpro/run.json`, `abpro/ranking.json`, and `abpro/post-export-validation.json` "
        "for the complete text-free provenance and evaluation receipts.\n"
    )


@dataclass
class _WorkflowResources:
    adapter: Any = None
    verifier: Any = None
    dense_model: Any = None
    artifact_path: Path | None = None
    published: bool = False


def run_obliteration_workflow(
    model: ModelLoadConfig,
    architecture: ArchitectureSpec,
    config: ObliterateConfig,
) -> dict[str, Any]:
    """Run the workflow while guaranteeing model and accelerator cleanup on every exit."""
    resources = _WorkflowResources()
    try:
        return _execute_obliteration_workflow(model, architecture, config, resources)
    finally:
        resources.adapter = None
        resources.verifier = None
        resources.dense_model = None
        if resources.published and resources.artifact_path is not None:
            with suppress(OSError):
                resources.artifact_path.unlink()
        _release_accelerator_memory()


def _execute_obliteration_workflow(
    model: ModelLoadConfig,
    architecture: ArchitectureSpec,
    config: ObliterateConfig,
    resources: _WorkflowResources,
) -> dict[str, Any]:
    """Discover, rank, permanently edit, reload-verify, and atomically publish a fork."""
    plan = obliteration_plan(model, config, architecture)
    output = Path(plan["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    target_records: list[PromptRecord] | None = None
    contrast_records: list[PromptRecord] | None = None
    direction_behavior_filter: dict[str, Any] | None = None
    if config.directions is None:
        target_records, contrast_records = _prepare_direction_records(config)
    ranking_records = _prepare_ranking_records(config)

    created_artifact = config.directions is None
    artifact_path = output.parent / (
        f".{output.name}.direction-{uuid.uuid4().hex[:12]}.safetensors"
    )
    resources.artifact_path = artifact_path
    if not created_artifact:
        provided_artifact = Path(config.directions).expanduser().resolve()  # type: ignore[arg-type]
        _copy_file_new(provided_artifact, artifact_path)

    source_receipt_at_start = local_checkpoint_receipt(model.model)
    resources.adapter = HuggingFaceAdapter.from_pretrained(model, architecture=architecture)
    detected_source_quantization = getattr(
        resources.adapter,
        "checkpoint_quantization",
        None,
    )
    source_quantization = {
        "requested": asdict(model.quantization),
        "detected_checkpoint_quantization": detected_source_quantization,
    }
    if config.dequantize:
        tokenizer = resources.adapter.tokenizer
        serialization_components = resources.adapter.serialization_components
        resources.dense_model = dequantize_model_for_edit(resources.adapter.model)
        resources.adapter = None
        _release_accelerator_memory()
        resources.adapter = HuggingFaceAdapter(
            resources.dense_model,
            tokenizer,
            architecture,
            raw_prompts=model.raw_prompts,
            enable_thinking=model.enable_thinking,
            tokenizer_source=model.tokenizer or model.model,
            tokenizer_requested_revision=model.tokenizer_revision,
            serialization_components=serialization_components,
        )
        resources.dense_model = None
        _release_accelerator_memory()
    source_check = inspect_and_smoke(resources.adapter, point=config.point)
    architecture_report = resources.adapter.inspect()
    source_receipt_after_load = local_checkpoint_receipt(model.model)
    if source_receipt_at_start != source_receipt_after_load:
        raise ConfigurationError("The local source checkpoint changed while it was being loaded")

    if created_artifact:
        if target_records is None or contrast_records is None:
            raise RuntimeError("Direction cohorts were not prepared")
        if config.objective == "refusal-removal":
            target_records, contrast_records, direction_behavior_filter = (
                filter_direction_records_by_behavior(
                    resources.adapter,
                    target_records,
                    contrast_records,
                    desired_count=config.direction_samples,
                    select_all_matches=config.length_match,
                    batch_size=config.batch_size,
                    max_new_tokens=config.max_new_tokens,
                    max_length=config.max_length,
                )
            )
            if config.length_match:
                target_records, contrast_records = length_match(
                    target_records,
                    contrast_records,
                    seed=config.seed,
                    bin_width=config.length_bin,
                )
                matched_count = min(len(target_records), len(contrast_records))
                direction_behavior_filter["post_filter_length_match"] = {
                    "enabled": True,
                    "bin_width": config.length_bin,
                    "matched_count": matched_count,
                    "selected_count": min(config.direction_samples, matched_count),
                }
                if matched_count < config.direction_samples:
                    raise DatasetError(
                        "Behavior-grounded cohorts did not retain enough length-matched rows: "
                        f"required {config.direction_samples}, received {matched_count}"
                    )
                target_records = target_records[: config.direction_samples]
                contrast_records = contrast_records[: config.direction_samples]
        capture_layers: tuple[int, ...] | None
        if config.candidate_layers == "auto":
            capture_layers = automatic_candidate_layers(
                architecture_report.layer_count,
                count=config.candidate_count,
            )
        else:
            capture_layers = config.candidate_layers
        capture = CaptureConfig(
            point=config.point,
            pooling=config.pooling,
            last_k=config.last_k,
            batch_size=config.batch_size,
            max_length=config.max_length,
            layers=capture_layers,
        )
        artifact = discover_direction_artifact(
            resources.adapter,
            target_records,
            contrast_records,
            capture,
            metadata={
                "workflow": "abpro-obliterate-v1",
                "objective": config.objective,
                "seed": config.seed,
                "model": model.model,
                "dtype": model.dtype,
                "device_map": model.device_map,
                "model_class": model.model_class,
                "length_matched": config.length_match,
                "length_bin": config.length_bin,
                "behavior_grounded": direction_behavior_filter is not None,
            },
        )
        artifact_sha256 = save_direction_artifact(artifact_path, artifact)
    else:
        artifact = load_direction_artifact(artifact_path)
        artifact_sha256 = _sha256(artifact_path)

    if config.direction_key is not None:
        if config.direction_key not in artifact.directions:
            raise ConfigurationError(f"Direction {config.direction_key!r} is not in the artifact")
        direction_keys = [config.direction_key]
        selection_strategy = f"explicit-key-{config.objective}-validation"
        if config.objective == "guarded" and config.allow_safety_loss:
            selection_strategy = "explicit-key-guarded-override-validation"
    else:
        direction_keys = sorted(
            artifact.directions,
            key=lambda key: (_direction_layer(key), key),
        )
        if not created_artifact and config.candidate_layers is not None:
            allowed = (
                set(
                    automatic_candidate_layers(
                        architecture_report.layer_count,
                        count=config.candidate_count,
                    )
                )
                if config.candidate_layers == "auto"
                else set(config.candidate_layers)
            )
            direction_keys = [key for key in direction_keys if _direction_layer(key) in allowed]
        if not direction_keys:
            raise ConfigurationError(
                "No artifact directions match the candidate-layer selection; pass "
                "--candidate-layers all or select --direction explicitly"
            )
        selection_strategy = f"{config.objective}-automatic-ranking"
        if config.objective == "guarded" and config.allow_safety_loss:
            selection_strategy = "guarded-override-automatic-ranking"
    provenance_warnings = validate_direction_artifact_compatibility(
        artifact,
        architecture_report,
        model=model.model,
        requested_revision=model.revision,
        tokenizer=model.tokenizer,
        requested_tokenizer_revision=model.tokenizer_revision,
        direction_keys=direction_keys,
        intervention_point=config.point,
        raw_prompts=model.raw_prompts,
        enable_thinking=model.enable_thinking,
        allow_mismatch=False,
        loaded_local_receipt=source_receipt_after_load,
    )
    try:
        ranking, baseline, modified_reports = rank_direction_candidates(
            resources.adapter,
            artifact,
            ranking_records,
            direction_keys,
            layers=config.apply_layers,
            point=config.point,
            strength=config.strength,
            objective=config.objective,
            safety_penalty=config.safety_penalty,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
            max_length=config.max_length,
        )
    except Exception as exc:
        artifact_note = f"; direction snapshot retained at {artifact_path}"
        raise RuntimeError(f"Direction ranking failed{artifact_note}: {exc}") from exc
    try:
        selected, selection_warnings = select_ranked_direction(
            ranking["candidates"],
            objective=config.objective,
            allow_safety_loss=config.allow_safety_loss,
            baseline_deny_refusal_rate=baseline.deny_refusal_rate,
        )
    except ConfigurationError as exc:
        rejected_ranking = output.parent / (
            f".{output.name}.rejected-ranking-{uuid.uuid4().hex[:12]}.json"
        )
        ranking["selection"] = {"status": "rejected", "reason": str(exc)}
        _write_json_new(rejected_ranking, ranking)
        artifact_note = f"; direction snapshot retained at {artifact_path}"
        raise ConfigurationError(
            f"{exc}; ranking receipt saved to {rejected_ranking}{artifact_note}"
        ) from exc
    selected_key = str(selected["direction"])
    selected_keys_value = selected.get("direction_keys", [selected_key])
    if not isinstance(selected_keys_value, list) or not selected_keys_value:
        raise ConfigurationError("Selected direction receipt contains invalid source keys")
    selected_keys = [str(key) for key in selected_keys_value]
    if any(key not in artifact.directions for key in selected_keys):
        raise ConfigurationError("Selected direction subspace references an unknown artifact key")
    selected_direction = (
        artifact.directions[selected_keys[0]]
        if len(selected_keys) == 1
        else [artifact.directions[key] for key in selected_keys]
    )
    ranking["selection"] = {
        "strategy": selection_strategy,
        "selected": selected,
        "warnings": selection_warnings,
    }
    ranking["provenance"] = {
        "artifact_sha256": artifact_sha256,
        "artifact_compatibility_warnings": provenance_warnings,
        "architecture": architecture_report.to_dict(),
        "environment": environment_receipt(),
    }
    ranking["execution"] = {
        "objective": config.objective,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "max_new_tokens": config.max_new_tokens,
        "point": config.point,
        "strength": config.strength,
        "apply_layers": ("all" if config.apply_layers is None else list(config.apply_layers)),
        "raw_prompts": model.raw_prompts,
        "enable_thinking": model.enable_thinking,
    }

    staging = output.parent / f".{output.name}.workflow-{uuid.uuid4().hex[:12]}"
    try:
        manifest = save_ablated_checkpoint(
            resources.adapter,
            selected_direction,
            staging,
            source_model=model.model,
            source_revision=model.revision,
            direction_artifact_path=artifact_path,
            direction_key=selected_key,
            layers=config.apply_layers,
            kinds=config.kinds,
            include_input_embeddings=config.objective == "refusal-removal",
            strength=config.strength,
            chunk_columns=config.chunk_columns,
            provenance_warnings=provenance_warnings,
            source_was_quantized=(
                model.quantization.kind != "none"
                or detected_source_quantization is not None
                or config.dequantize
            ),
            source_quantization=source_quantization,
            dequantized_before_edit=config.dequantize,
        )
        resources.adapter = None
        _release_accelerator_memory()

        evidence = staging / "abpro"
        evidence.mkdir()
        shutil.copy2(artifact_path, evidence / "direction.safetensors")
        if (
            _sha256(evidence / "direction.safetensors") != artifact_sha256
            or manifest.get("direction_artifact_sha256") != artifact_sha256
        ):
            raise ConfigurationError(
                "Direction snapshot changed between selection, weight editing, and evidence copy"
            )
        _write_json_new(evidence / "ranking.json", ranking)

        verify_model = replace(
            model,
            model=str(staging),
            revision=None,
            tokenizer=None,
            tokenizer_revision=None,
            quantization=QuantizationConfig(),
            local_files_only=True,
        )
        resources.verifier = HuggingFaceAdapter.from_pretrained(
            verify_model,
            architecture=architecture,
        )
        target_verification = verify_reloaded_targets(
            resources.verifier,
            manifest,
            selected_direction,
            chunk_columns=config.chunk_columns,
        )
        reload_smoke = inspect_and_smoke(resources.verifier, point=config.point)
        permanent = evaluate_prompts(
            resources.verifier,
            ranking_records,
            batch_size=config.batch_size,
            max_new_tokens=config.max_new_tokens,
            max_length=config.max_length,
        )
        permanent_safety = _validate_permanent_safety_result(
            baseline,
            permanent,
            objective=config.objective,
            allow_safety_loss=config.allow_safety_loss,
        )
        post_export_validation = compare_reports(baseline, permanent)
        post_export_validation["baseline"]["results"] = [asdict(row) for row in baseline.results]
        post_export_validation["modified"]["results"] = [asdict(row) for row in permanent.results]
        selected_hook = modified_reports[selected_key]
        post_export_validation["hook_vs_permanent"] = compare_reports(
            selected_hook,
            permanent,
        )
        post_export_validation["cohort"] = cohort_receipt(ranking_records)
        post_export_validation["objective_validation"] = permanent_safety
        post_export_validation["guard"] = permanent_safety
        resources.verifier = None
        _release_accelerator_memory()

        source_receipt_before = source_receipt_at_start
        source_receipt_after = local_checkpoint_receipt(model.model)
        if source_receipt_after != source_receipt_before:
            raise ConfigurationError("The local source checkpoint changed during the workflow")
        output_checkpoint_receipt = local_checkpoint_receipt(str(staging))
        if output_checkpoint_receipt is None:
            raise ConfigurationError("Could not generate a receipt for the staged checkpoint")
        output_checkpoint_receipt["path"] = "."
        edit_manifest_path = staging / "abliterator_manifest.json"
        receipt: dict[str, Any] = {
            "format": "abpro-obliterate-run-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "verified",
            "source_model": model.model,
            "source_revision": architecture_report.checkpoint_revision,
            "source_checkpoint_receipt_before": source_receipt_before,
            "source_checkpoint_receipt_after": source_receipt_after,
            "output": ".",
            "output_name": output.name,
            "output_checkpoint_receipt": output_checkpoint_receipt,
            "plan": plan,
            "source_verification": source_check,
            "direction": {
                "artifact": "abpro/direction.safetensors",
                "artifact_sha256": artifact_sha256,
                "target": (None if target_records is None else cohort_receipt(target_records)),
                "contrast": (
                    None if contrast_records is None else cohort_receipt(contrast_records)
                ),
                "behavior_filter": direction_behavior_filter,
            },
            "selection": {
                "strategy": selection_strategy,
                "objective": config.objective,
                **selected,
                "warnings": selection_warnings,
            },
            "edit": {
                "manifest": "abliterator_manifest.json",
                "manifest_sha256": _sha256(edit_manifest_path),
                "strength": config.strength,
                "layers": manifest["layers"],
                "kinds": manifest["kinds"],
                "target_count": len(manifest["targets"]),
            },
            "reload_verification": {
                **target_verification,
                "forward_capture": reload_smoke["forward_capture"],
            },
            "post_export_validation": post_export_validation,
            "provenance_warnings": provenance_warnings,
            "environment": environment_receipt(),
        }
        _write_json_new(evidence / "post-export-validation.json", post_export_validation)
        _write_json_new(evidence / "run.json", receipt)
        if not (staging / "README.md").exists():
            (staging / "README.md").write_text(_render_model_card(receipt), encoding="utf-8")
        source_license = Path(model.model).expanduser() / "LICENSE"
        if source_license.is_file() and not (staging / "LICENSE").exists():
            shutil.copy2(source_license, staging / "LICENSE")
        publish_directory_noreplace(staging, output)
        resources.published = True
    except Exception as exc:
        raise RuntimeError(
            "One-command obliteration failed before final publication. "
            f"Staging target: {staging}. Direction snapshot: {artifact_path}. Cause: {exc}"
        ) from exc

    return {
        "output": str(output),
        "objective": config.objective,
        "selected_direction": selected_key,
        "edited_targets": len(manifest["targets"]),
        "reload_verified": True,
        "metrics": {
            "baseline": baseline.to_dict(),
            "modified": permanent.to_dict(),
            "delta": post_export_validation["delta"],
        },
        "receipt": str(output / "abpro" / "run.json"),
    }
