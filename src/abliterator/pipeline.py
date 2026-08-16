"""High-level experiment operations used by both notebooks and the CLI."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import uuid
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters.base import ModelAdapter, ProjectionKind
from .artifacts import DirectionArtifact, local_checkpoint_receipt, save_direction_artifact
from .capture import ActivationCollector
from .config import CaptureConfig
from .datasets import PromptRecord
from .directions import estimate_directions
from .errors import ConfigurationError
from .interventions import WeightAblator
from .runtime import publish_directory_noreplace


def environment_receipt() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("torch", "transformers", "accelerate", "datasets", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def discover_direction_artifact(
    adapter: ModelAdapter,
    target_records: list[PromptRecord],
    contrast_records: list[PromptRecord],
    capture: CaptureConfig,
    *,
    metadata: dict[str, Any] | None = None,
) -> DirectionArtifact:
    if not target_records or not contrast_records:
        raise ConfigurationError("Both target and contrast cohorts must be non-empty")
    collector = ActivationCollector(adapter)
    target = collector.collect([record.prompt for record in target_records], capture)
    contrast = collector.collect([record.prompt for record in contrast_records], capture)
    directions = estimate_directions(target.means, contrast.means)
    report = adapter.inspect()
    artifact_metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "normalized-target-minus-contrast-mean-v1",
        "architecture": report.to_dict(),
        "capture": asdict(capture),
        "target": cohort_receipt(target_records),
        "contrast": cohort_receipt(contrast_records),
        "target_tokenization": target.input_receipt,
        "contrast_tokenization": contrast.input_receipt,
        "environment": environment_receipt(),
        "prompt_rendering": {
            "raw_prompts": bool(getattr(adapter, "raw_prompts", False)),
            "enable_thinking": bool(getattr(adapter, "enable_thinking", False)),
        },
    }
    model_config = getattr(adapter, "config", None)
    model_name = getattr(model_config, "_name_or_path", None)
    artifact_metadata["model"] = model_name
    artifact_metadata["revision"] = report.checkpoint_revision
    tokenizer_name = getattr(adapter, "_tokenizer_source", None) or model_name
    artifact_metadata["tokenizer"] = tokenizer_name
    model_receipt = local_checkpoint_receipt(model_name) if isinstance(model_name, str) else None
    artifact_metadata["local_checkpoint_receipt"] = model_receipt
    if tokenizer_name == model_name:
        artifact_metadata["local_tokenizer_receipt"] = model_receipt
    elif isinstance(tokenizer_name, str):
        artifact_metadata["local_tokenizer_receipt"] = local_checkpoint_receipt(
            tokenizer_name,
            include_weights=False,
        )
    if metadata:
        artifact_metadata["experiment"] = dict(metadata)
    return DirectionArtifact(directions=directions, metadata=artifact_metadata)


def cohort_receipt(records: list[PromptRecord]) -> dict[str, Any]:
    source_revisions = sorted(
        {f"{record.source_id}@{record.source_revision or 'unversioned'}" for record in records}
    )
    prompt_hashes = sorted(
        hashlib.sha256(record.prompt.encode("utf-8")).hexdigest() for record in records
    )
    combined = hashlib.sha256("\n".join(prompt_hashes).encode("ascii")).hexdigest()
    return {
        "count": len(records),
        "sources": source_revisions,
        "cohort_sha256": combined,
        "prompt_ids": [record.id for record in records],
    }


def select_direction(artifact: DirectionArtifact, key: str) -> Any:
    try:
        return artifact.directions[key]
    except KeyError as exc:
        available = ", ".join(sorted(artifact.directions)[:20])
        raise ConfigurationError(
            f"Unknown direction {key!r}; available examples: {available}"
        ) from exc


def save_ablated_checkpoint(
    adapter: ModelAdapter,
    direction: Any,
    output: str | Path,
    *,
    source_model: str,
    source_revision: str | None,
    direction_artifact_path: str | Path,
    direction_key: str,
    layers: Iterable[int] | None = None,
    kinds: Iterable[ProjectionKind] = ("attention", "mlp", "expert"),
    include_input_embeddings: bool = False,
    strength: float = 1.0,
    chunk_columns: int = 8192,
    provenance_warnings: list[str] | None = None,
    source_was_quantized: bool = False,
    source_quantization: dict[str, Any] | None = None,
    dequantized_before_edit: bool = False,
) -> dict[str, Any]:
    destination = Path(output).resolve()
    if destination.exists():
        raise ConfigurationError(f"Refusing to overwrite existing output path {destination}")
    artifact_source = Path(direction_artifact_path).resolve()
    try:
        artifact_bytes = artifact_source.read_bytes()
    except OSError as exc:
        raise ConfigurationError(
            f"Could not read direction artifact before beginning mutation: {artifact_source}"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex[:12]}"
    partial.mkdir()
    ablator = WeightAblator(adapter)
    records = ablator.apply(
        direction,
        layers=layers,
        kinds=kinds,
        include_input_embeddings=include_input_embeddings,
        strength=strength,
        chunk_columns=chunk_columns,
    )
    manifest = {
        "format": "abliterator-checkpoint-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": source_model,
        "source_requested_revision": source_revision,
        "source_revision": getattr(adapter, "resolved_revision", None) or source_revision,
        "source_was_quantized": source_was_quantized,
        "source_quantization": source_quantization or {"kind": "none"},
        "dequantized_before_edit": dequantized_before_edit,
        "direction_artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "direction_key": direction_key,
        "strength": strength,
        "layers": sorted({record.layer for record in records}),
        "kinds": sorted({record.kind for record in records}),
        "architecture": adapter.inspect().to_dict(),
        "targets": [record.to_dict() for record in records],
        "provenance_warnings": list(provenance_warnings or []),
        "environment": environment_receipt(),
    }
    try:
        adapter.model.save_pretrained(partial, safe_serialization=True)
        serialization_components = getattr(
            adapter, "serialization_components", (adapter.tokenizer,)
        )
        for component in serialization_components:
            component.save_pretrained(partial)
        (partial / "abliterator_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_directory_noreplace(partial, destination)
    except Exception as exc:
        raise RuntimeError(
            f"Checkpoint export failed; partial data was retained at {partial} for diagnosis"
        ) from exc
    return manifest


def write_direction_artifact(path: str | Path, artifact: DirectionArtifact) -> dict[str, Any]:
    digest = save_direction_artifact(path, artifact)
    return {
        "path": str(Path(path).resolve()),
        "sha256": digest,
        "directions": sorted(artifact.directions),
        "metadata": artifact.metadata,
    }
