"""Safe, pickle-free direction artifact serialization."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters.base import ArchitectureReport
from .directions import normalize_direction
from .errors import ConfigurationError
from .runtime import require


@dataclass
class DirectionArtifact:
    directions: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.metadata, dict):
            raise ConfigurationError("Direction artifact metadata must be a JSON object")
        if not self.directions:
            raise ConfigurationError("Direction artifact is empty")
        widths: set[int] = set()
        for key, direction in self.directions.items():
            if not key or not isinstance(key, str):
                raise ConfigurationError("Every direction needs a non-empty string key")
            widths.add(int(normalize_direction(direction).shape[0]))
        if len(widths) != 1:
            raise ConfigurationError(f"Direction widths do not match: {sorted(widths)}")

    @property
    def hidden_size(self) -> int:
        self.validate()
        first = next(iter(self.directions.values()))
        return int(normalize_direction(first).shape[0])


def _normalized_model_reference(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return value.strip().rstrip("/").casefold()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def local_checkpoint_receipt(
    model: str,
    *,
    include_weights: bool = True,
) -> dict[str, Any] | None:
    """Hash a local checkpoint without loading an entire shard into memory."""
    root = Path(model).expanduser()
    if not root.exists():
        return None
    root = root.resolve()
    if root.is_file():
        content_sha256 = _sha256_file(root)
        return {
            "path": str(root),
            "kind": "file",
            "size": root.stat().st_size,
            "sha256": content_sha256,
            "content_sha256": content_sha256,
        }
    names = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
    )
    metadata_files: dict[str, str] = {}
    for name in names:
        candidate = root / name
        if candidate.is_file():
            metadata_files[name] = _sha256_file(candidate)
    for pattern in ("*.py", "*.jinja", "*processor*.json", "chat_template*"):
        for candidate in sorted(root.glob(pattern)):
            if candidate.is_file():
                metadata_files[candidate.name] = _sha256_file(candidate)
    shards: list[dict[str, Any]] = []
    if include_weights:
        shard_paths: set[Path] = set()
        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_path = root / index_name
            if not index_path.is_file():
                continue
            try:
                index_data = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index_data.get("weight_map", {})
                if isinstance(weight_map, dict):
                    shard_paths.update(root / str(name) for name in weight_map.values())
            except (OSError, json.JSONDecodeError):
                pass
        if not shard_paths:
            for pattern in ("model*.safetensors", "pytorch_model*.bin"):
                shard_paths.update(root.glob(pattern))
        for candidate in sorted(shard_paths):
            if candidate.is_file():
                shards.append(
                    {
                        "name": candidate.name,
                        "size": candidate.stat().st_size,
                        "sha256": _sha256_file(candidate),
                    }
                )
    payload = {"metadata_files": metadata_files, "shards": shards}
    return {
        "path": str(root),
        "kind": "directory-content-v1",
        **payload,
        "content_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def validate_direction_artifact_compatibility(
    artifact: DirectionArtifact,
    report: ArchitectureReport,
    *,
    model: str,
    requested_revision: str | None,
    tokenizer: str | None = None,
    requested_tokenizer_revision: str | None = None,
    direction_keys: list[str] | None = None,
    intervention_point: str | None = None,
    raw_prompts: bool = False,
    enable_thinking: bool = False,
    allow_mismatch: bool = False,
) -> list[str]:
    """Fail closed when a direction artifact cannot be tied to the loaded checkpoint."""
    artifact.validate()
    metadata = artifact.metadata
    architecture = metadata.get("architecture")
    issues: list[str] = []
    if not isinstance(architecture, dict):
        architecture = {}
        issues.append("artifact has no architecture provenance")

    if artifact.hidden_size != report.hidden_size:
        issues.append(
            f"direction width is {artifact.hidden_size}, loaded model width is {report.hidden_size}"
        )
    prompt_rendering = metadata.get("prompt_rendering")
    if not isinstance(prompt_rendering, dict):
        issues.append("artifact does not record prompt_rendering policy")
    else:
        artifact_raw = prompt_rendering.get("raw_prompts")
        artifact_thinking = prompt_rendering.get("enable_thinking")
        if artifact_raw is not raw_prompts:
            issues.append(
                f"artifact raw_prompts={artifact_raw!r}, loaded raw_prompts={raw_prompts!r}"
            )
        if artifact_thinking is not enable_thinking:
            issues.append(
                "artifact enable_thinking="
                f"{artifact_thinking!r}, loaded enable_thinking={enable_thinking!r}"
            )
    artifact_hidden = architecture.get("hidden_size")
    if artifact_hidden is not None and artifact_hidden != report.hidden_size:
        issues.append(
            f"artifact architecture hidden_size={artifact_hidden}, loaded={report.hidden_size}"
        )
    artifact_type = architecture.get("model_type")
    if not artifact_type:
        issues.append("artifact does not record architecture.model_type")
    elif str(artifact_type).casefold() != report.model_type.casefold():
        issues.append(
            f"artifact model_type={artifact_type!r}, loaded model_type={report.model_type!r}"
        )
    artifact_layers = architecture.get("layer_count")
    if artifact_layers is not None and artifact_layers != report.layer_count:
        issues.append(
            f"artifact layer_count={artifact_layers}, loaded layer_count={report.layer_count}"
        )
    artifact_layer_path = architecture.get("layer_path")
    if not artifact_layer_path:
        issues.append("artifact does not record architecture.layer_path")
    elif artifact_layer_path != report.layer_path:
        issues.append(
            f"artifact layer_path={artifact_layer_path!r}, loaded layer_path={report.layer_path!r}"
        )

    artifact_model = metadata.get("model")
    if not isinstance(artifact_model, str) or not artifact_model.strip():
        issues.append("artifact does not record its source model")
    elif _normalized_model_reference(artifact_model) != _normalized_model_reference(model):
        issues.append(f"artifact source model={artifact_model!r}, loaded model={model!r}")

    loaded_tokenizer = tokenizer or model
    artifact_tokenizer = metadata.get("tokenizer") or artifact_model
    if isinstance(artifact_tokenizer, str) and (
        _normalized_model_reference(artifact_tokenizer)
        != _normalized_model_reference(loaded_tokenizer)
    ):
        issues.append(
            f"artifact tokenizer={artifact_tokenizer!r}, loaded tokenizer={loaded_tokenizer!r}"
        )
    artifact_tokenizer_revision = architecture.get("tokenizer_revision")
    if artifact_tokenizer_revision and report.tokenizer_revision:
        if artifact_tokenizer_revision != report.tokenizer_revision:
            issues.append(
                "artifact tokenizer commit "
                f"{artifact_tokenizer_revision!r}, loaded tokenizer commit "
                f"{report.tokenizer_revision!r}"
            )
    elif (
        requested_tokenizer_revision
        and artifact_tokenizer_revision
        and requested_tokenizer_revision != artifact_tokenizer_revision
    ):
        issues.append(
            "artifact tokenizer revision "
            f"{artifact_tokenizer_revision!r}, requested {requested_tokenizer_revision!r}"
        )

    loaded_local_receipt = local_checkpoint_receipt(model)
    artifact_local_receipt = metadata.get("local_checkpoint_receipt")
    if loaded_local_receipt is not None:
        if not isinstance(artifact_local_receipt, dict):
            issues.append("artifact does not contain a valid local checkpoint content receipt")
        elif artifact_local_receipt.get("content_sha256") != loaded_local_receipt.get(
            "content_sha256"
        ):
            issues.append("local checkpoint content receipt differs from direction discovery")

    same_local_source = loaded_local_receipt is not None and _normalized_model_reference(
        loaded_tokenizer
    ) == _normalized_model_reference(model)
    loaded_local_tokenizer_receipt = (
        loaded_local_receipt
        if same_local_source
        else local_checkpoint_receipt(loaded_tokenizer, include_weights=False)
    )
    artifact_local_tokenizer_receipt = metadata.get("local_tokenizer_receipt")
    if loaded_local_tokenizer_receipt is not None:
        if not isinstance(artifact_local_tokenizer_receipt, dict):
            issues.append("artifact does not contain a valid local tokenizer content receipt")
        elif artifact_local_tokenizer_receipt.get(
            "content_sha256"
        ) != loaded_local_tokenizer_receipt.get("content_sha256"):
            issues.append("local tokenizer content receipt differs from direction discovery")

    artifact_resolved_revision = architecture.get("checkpoint_revision")
    loaded_resolved_revision = report.checkpoint_revision
    if artifact_resolved_revision and loaded_resolved_revision:
        if artifact_resolved_revision != loaded_resolved_revision:
            issues.append(
                "artifact checkpoint commit "
                f"{artifact_resolved_revision!r}, loaded commit {loaded_resolved_revision!r}"
            )
    elif loaded_local_receipt is None:
        artifact_requested_revision = metadata.get("revision")
        if artifact_requested_revision and requested_revision:
            if artifact_requested_revision != requested_revision:
                issues.append(
                    "artifact requested revision "
                    f"{artifact_requested_revision!r}, loaded revision {requested_revision!r}"
                )
        elif not artifact_resolved_revision and not artifact_requested_revision:
            issues.append("artifact does not record a source checkpoint revision")
        elif not loaded_resolved_revision and not requested_revision:
            issues.append("loaded checkpoint revision could not be verified")

    capture = metadata.get("capture")
    captured_point = capture.get("point") if isinstance(capture, dict) else None
    if intervention_point is not None:
        if captured_point is None:
            issues.append("artifact does not record its residual capture point")
        elif captured_point != intervention_point:
            issues.append(
                f"artifact capture point={captured_point!r}, intervention "
                f"point={intervention_point!r}"
            )
    for key in direction_keys or []:
        if key not in artifact.directions:
            issues.append(f"artifact has no direction named {key!r}")
            continue
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "layer":
            try:
                layer = int(parts[1])
            except ValueError:
                layer = -1
            if layer < 0 or layer >= report.layer_count:
                issues.append(
                    f"direction {key!r} refers to layer {layer}, loaded model has "
                    f"{report.layer_count} layers"
                )
        if captured_point and not key.endswith(f".{captured_point}"):
            issues.append(
                f"direction key {key!r} does not match recorded capture point {captured_point!r}"
            )

    if issues and not allow_mismatch:
        detail = "\n  - ".join(issues)
        raise ConfigurationError(
            "Direction artifact provenance is incompatible with the loaded checkpoint:\n"
            f"  - {detail}\n"
            "Use --allow-artifact-mismatch only after independently verifying compatibility."
        )
    return issues


def save_direction_artifact(path: str | Path, artifact: DirectionArtifact) -> str:
    artifact.validate()
    safetensors = require("safetensors.torch")
    destination = Path(path).resolve()
    if destination.exists():
        raise ConfigurationError(f"Refusing to overwrite direction artifact {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex[:12]}"
    tensors = {
        key: normalize_direction(tensor).contiguous().cpu()
        for key, tensor in artifact.directions.items()
    }
    metadata = {
        "format": "abliterator-direction-v1",
        "metadata_json": json.dumps(artifact.metadata, sort_keys=True, separators=(",", ":")),
    }
    try:
        safetensors.save_file(tensors, str(partial), metadata=metadata)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(partial, destination)
    except Exception as exc:
        raise ConfigurationError(
            f"Could not publish direction artifact without overwriting; partial data is at "
            f"{partial}"
        ) from exc
    with suppress(OSError):
        partial.unlink()
    return _sha256_file(destination)


def load_direction_artifact(path: str | Path) -> DirectionArtifact:
    safetensors = require("safetensors")
    torch_io = require("safetensors.torch")
    source = Path(path)
    try:
        with safetensors.safe_open(str(source), framework="pt", device="cpu") as handle:
            header = handle.metadata() or {}
    except Exception as exc:
        raise ConfigurationError(f"Could not open direction artifact {source}") from exc
    if header.get("format") != "abliterator-direction-v1":
        raise ConfigurationError(f"Unsupported direction artifact format in {source}")
    try:
        metadata = json.loads(header.get("metadata_json", "{}"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid direction metadata in {source}") from exc
    try:
        directions = torch_io.load_file(str(source), device="cpu")
    except Exception as exc:
        raise ConfigurationError(f"Could not read direction tensors from {source}") from exc
    artifact = DirectionArtifact(directions=directions, metadata=metadata)
    artifact.validate()
    return artifact
