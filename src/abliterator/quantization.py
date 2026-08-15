"""Quantization capability checks and backend-specific load/export helpers."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .config import ModelLoadConfig, QuantizationConfig
from .errors import ArchitectureError, QuantizationError
from .runtime import publish_directory_noreplace, require, torch_dtype


@dataclass(frozen=True)
class QuantizationManifest:
    format: str
    backend: str
    source_dtype: str
    compute_dtype: str
    double_quant: bool = False
    keep_master_weights: bool = False
    pseudoquantization: bool = False
    source_was_quantized: bool = False
    library_versions: dict[str, str] | None = None
    source_checkpoint_sha256: str | None = None
    source_manifest_sha256: str | None = None
    verification: dict[str, Any] | None = None


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_transformers_quantization(config: QuantizationConfig) -> Any | None:
    """Build a Transformers quantizer without conflating incompatible FP4 formats."""
    config.validate()
    if config.kind == "none":
        return None
    transformers = require("transformers")
    compute_dtype = torch_dtype(config.compute_dtype)
    if config.kind in {"bnb-fp4", "bnb-nf4"}:
        require("bitsandbytes", extra="bnb")
        return transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.kind.removeprefix("bnb-"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.double_quant,
        )
    fp_quant = getattr(transformers, "FPQuantConfig", None)
    if fp_quant is None:
        raise QuantizationError(
            f"{config.kind} requires a Transformers release containing FPQuantConfig and the "
            "corresponding FP-Quant backend"
        )
    require("fp_quant", extra="fpquant")
    fp_compute_name = "bf16"
    try:
        return fp_quant(
            forward_dtype=config.kind,
            backward_dtype=fp_compute_name,
            store_master_weights=config.keep_master_weights,
            pseudoquantization=config.pseudoquantization,
        )
    except (TypeError, ValueError) as exc:
        raise QuantizationError(
            f"Installed Transformers/FP-Quant does not support {config.kind!r}; install the "
            "tested fpquant extra"
        ) from exc


def checkpoint_quantization_kind(config: Any, _seen: set[int] | None = None) -> str | None:
    """Return configured checkpoint quantization, including embedded FP8 metadata."""
    seen = set() if _seen is None else _seen
    if id(config) in seen:
        return None
    seen.add(id(config))
    if isinstance(config, dict):
        value = (
            config.get("quantization_config")
            or config.get("quantization_method")
            or config.get("quant_method")
        )
    else:
        value = getattr(config, "quantization_config", None)
        if value is None:
            value = getattr(config, "quantization_method", None)
        if value is None:
            value = getattr(config, "quant_method", None)
    if isinstance(value, dict):
        value = (
            value.get("quant_method")
            or value.get("quantization_method")
            or value.get("format")
            or value.get("dtype")
        )
    else:
        value = getattr(value, "quant_method", value)
        value = getattr(value, "value", value)
    if value is None:
        for name in ("text_config", "language_config"):
            nested = getattr(config, name, None)
            if nested is None and isinstance(config, dict):
                nested = config.get(name)
            if nested is not None:
                discovered = checkpoint_quantization_kind(nested, seen)
                if discovered:
                    return discovered
        return None
    text = str(value).strip().lower()
    return text or None


def describe_module_quantization(
    module: Any | None,
    weight: Any,
    *,
    checkpoint_quantization: str | None = None,
) -> tuple[str, bool]:
    """Return `(format, packed)` using module type, tensor dtype, and quantizer state."""
    module_name = (
        "" if module is None else f"{module.__class__.__module__}.{module.__class__.__name__}"
    )
    lowered = module_name.lower()
    weight_name = f"{weight.__class__.__module__}.{weight.__class__.__name__}".lower()
    dtype_name = str(getattr(weight, "dtype", "unknown")).lower()
    if "bitsandbytes" in lowered or "params4bit" in weight_name or hasattr(weight, "quant_state"):
        quant_type = getattr(getattr(weight, "quant_state", None), "quant_type", None)
        return f"bnb-{quant_type or '4bit'}", True
    if (
        "fpquant" in lowered
        or "fp_quant" in lowered
        or any(hasattr(module, name) for name in ("qweight", "dqweight"))
    ):
        master = getattr(module, "weight", None)
        editable_master = bool(master is not None and _is_float_tensor(master))
        # Even with a master copy, changing it does not automatically refresh qweight/scales.
        return "fp-quant-master" if editable_master else "fp-quant-packed", True
    if module is not None and any(
        hasattr(module, name) for name in ("weight_packed", "packed_weight")
    ):
        return checkpoint_quantization or f"packed-{module_name}", True
    quantized_module_tokens = (
        "awq",
        "gptq",
        "hqq",
        "quanto",
        "torchao",
        "compressed",
        "modelopt",
        "quantized",
        "fp8",
        "float8",
        "mxfp4",
        "nvfp4",
        "fouroversix",
    )
    if any(token in lowered for token in quantized_module_tokens):
        return module_name, True
    scale_attributes = (
        "weight_scale_inv",
        "weight_scale",
        "weight_scales",
        "scale_inv",
        "scales",
        "input_scale",
        "weight_quantizer",
    )
    if module is not None and any(hasattr(module, name) for name in scale_attributes):
        return f"scaled-{module_name}", True
    if "float8" in dtype_name or "fp8" in dtype_name:
        return dtype_name, True
    if any(token in weight_name for token in quantized_module_tokens):
        return weight_name, True
    if any(token in weight_name for token in ("dtensor", "shardedtensor", "flatparameter")):
        return weight_name, True
    if not _is_float_tensor(weight):
        return f"packed-{getattr(weight, 'dtype', 'unknown')}", True
    if checkpoint_quantization:
        return f"checkpoint-{checkpoint_quantization}", True
    return "dense", False


def _is_float_tensor(value: Any) -> bool:
    try:
        return bool(value.is_floating_point())
    except (AttributeError, TypeError):
        return False


def _embedded_quantization(config: Any, path: str = "config") -> tuple[str, Any] | None:
    if not isinstance(config, dict):
        return None
    for key in ("quantization_config", "quant_method", "quantization_method"):
        value = config.get(key)
        if value:
            return f"{path}.{key}", value
    for key in ("text_config", "language_config"):
        nested = config.get(key)
        found = _embedded_quantization(nested, f"{path}.{key}")
        if found is not None:
            return found
    return None


def _configured_source_dtype(config: Any) -> str:
    if not isinstance(config, dict):
        return "auto"
    for key in ("torch_dtype", "dtype"):
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("text_config", "language_config"):
        nested = _configured_source_dtype(config.get(key))
        if nested != "auto":
            return nested
    return "auto"


def assert_editable_targets(targets: Iterable[Any]) -> None:
    packed = [target for target in targets if target.packed]
    if not packed:
        return
    examples = ", ".join(f"{target.name} ({target.quantization})" for target in packed[:4])
    raise QuantizationError(
        "Permanent projection was blocked because packed low-bit weights are not ordinary "
        f"matrices. Examples: {examples}. Discover/test directions on the quantized model, then "
        "apply them to a dense source checkpoint and quantize the edited checkpoint, or "
        "explicitly dequantize first."
    )


def _runtime_packed_modules(model: Any) -> list[tuple[str, str]]:
    packed: list[tuple[str, str]] = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        for storage_name in ("qweight", "dqweight", "weight_packed", "packed_weight"):
            if weight is None:
                weight = getattr(module, storage_name, None)
        if weight is None:
            continue
        quantization, is_packed = describe_module_quantization(module, weight)
        if is_packed:
            packed.append((name, quantization))
    return packed


def _clear_quantization_metadata(config: Any, seen: set[int] | None = None) -> None:
    visited = set() if seen is None else seen
    if config is None or id(config) in visited:
        return
    visited.add(id(config))
    for name in ("quantization_config", "quant_method", "quantization_method"):
        if hasattr(config, name):
            config.__dict__[name] = None
    for name in ("text_config", "language_config"):
        _clear_quantization_metadata(getattr(config, name, None), visited)


def dequantize_model_for_edit(model: Any) -> Any:
    """Use the backend-supported whole-model conversion, never raw packed tensor arithmetic."""
    method = getattr(model, "dequantize", None)
    if method is None:
        raise QuantizationError(
            "This quantizer does not expose model.dequantize(); use a dense source checkpoint"
        )
    try:
        result = method()
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        raise QuantizationError(
            "Backend dequantization failed. Ensure the expanded model fits memory or use a dense "
            "source checkpoint."
        ) from exc
    dense_model = result if result is not None else model
    residual_packed = _runtime_packed_modules(dense_model)
    if residual_packed:
        examples = ", ".join(f"{name} ({kind})" for name, kind in residual_packed[:4])
        raise QuantizationError(
            "Backend dequantize() left packed/scaled modules, so permanent editing remains "
            f"unsafe. Examples: {examples}"
        )
    _clear_quantization_metadata(getattr(dense_model, "config", None))
    return dense_model


def quantize_saved_checkpoint(
    source: str | Path,
    output: str | Path,
    config: QuantizationConfig,
    *,
    trust_remote_code: bool = False,
    revision: str | None = None,
    model_class: Literal["causal-lm", "image-text-to-text"] = "causal-lm",
) -> QuantizationManifest:
    """Quantize an already-edited dense checkpoint through an official Transformers backend."""
    if config.kind == "none":
        raise QuantizationError("quantize_saved_checkpoint requires a non-dense target format")
    config.validate()
    from .adapters.huggingface import HuggingFaceAdapter, load_pretrained_components

    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise QuantizationError(f"Dense source checkpoint does not exist: {source_path}")
    if not source_path.is_dir():
        raise QuantizationError("Dense source checkpoint must be a save_pretrained directory")
    output_path = Path(output).resolve()
    if output_path.exists():
        raise QuantizationError(f"Refusing to overwrite existing output path {output_path}")
    if (
        output_path == source_path
        or source_path in output_path.parents
        or output_path in source_path.parents
    ):
        raise QuantizationError(
            "Dense source and quantized output must be separate, non-nested paths"
        )
    source_manifest_path = source_path / "abliterator_manifest.json"
    source_dtype = "auto"
    source_was_quantized = False
    config_path = source_path / "config.json" if source_path.is_dir() else None
    if config_path is not None and config_path.is_file():
        try:
            source_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QuantizationError(
                f"Could not validate dense source config {config_path}"
            ) from exc
        embedded = _embedded_quantization(source_config)
        if embedded is not None:
            embedded_path, _ = embedded
            raise QuantizationError(
                "Quantized export requires a dense source checkpoint; embedded quantization "
                f"metadata was found at {embedded_path} in {config_path}"
            )
        source_dtype = _configured_source_dtype(source_config)
    from .artifacts import local_checkpoint_receipt

    source_receipt = local_checkpoint_receipt(str(source_path))
    source_manifest_sha256 = (
        _sha256_path(source_manifest_path) if source_manifest_path.is_file() else None
    )
    if source_manifest_path.is_file():
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            source_was_quantized = bool(source_manifest.get("source_was_quantized", False))
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise QuantizationError(
                f"Could not validate source ablation manifest {source_manifest_path}"
            ) from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.parent / f".{output_path.name}.partial-{uuid.uuid4().hex[:12]}"
    partial.mkdir()
    try:
        (
            model,
            tokenizer,
            tokenizer_id,
            tokenizer_revision,
            serialization_components,
        ) = load_pretrained_components(
            ModelLoadConfig(
                model=str(source_path),
                revision=revision,
                trust_remote_code=trust_remote_code,
                quantization=config,
                dtype="auto",
                model_class=model_class,
            )
        )
        runtime_packed = _runtime_packed_modules(model)
        if not runtime_packed:
            raise QuantizationError(
                "The backend returned no packed modules; refusing to label the "
                f"checkpoint as {config.kind}"
            )
        if config.kind.startswith("bnb-") and not any(
            kind.startswith("bnb-") for _, kind in runtime_packed
        ):
            raise QuantizationError(
                "bitsandbytes export did not expose bitsandbytes-packed target modules"
            )
        if config.kind in {"nvfp4", "mxfp4"} and not any(
            kind.startswith("fp-quant") for _, kind in runtime_packed
        ):
            raise QuantizationError("FP-Quant export did not expose FP-Quant packed target modules")
        architecture: dict[str, Any]
        inspection_adapter = None
        try:
            inspection_adapter = HuggingFaceAdapter(
                model,
                tokenizer,
                tokenizer_source=tokenizer_id,
                tokenizer_requested_revision=tokenizer_revision,
            )
            architecture = inspection_adapter.inspect().to_dict()
        except ArchitectureError as exc:
            architecture = {
                "model_type": str(getattr(model.config, "model_type", "unknown")),
                "inspection": "unavailable",
                "reason": str(exc),
            }
        model.save_pretrained(partial, safe_serialization=True)
        for component in serialization_components:
            component.save_pretrained(partial)
        backend = "bitsandbytes" if config.kind.startswith("bnb-") else "fp-quant"
        manifest = QuantizationManifest(
            format=config.kind,
            backend=backend,
            source_dtype=source_dtype,
            compute_dtype=config.compute_dtype,
            double_quant=config.double_quant,
            keep_master_weights=config.keep_master_weights,
            pseudoquantization=config.pseudoquantization,
            source_was_quantized=source_was_quantized,
            library_versions={
                "torch": _version("torch"),
                "transformers": _version("transformers"),
                "bitsandbytes": _version("bitsandbytes"),
                "fp_quant": _version("fp_quant"),
            },
            source_checkpoint_sha256=(
                None if source_receipt is None else source_receipt.get("content_sha256")
            ),
            source_manifest_sha256=source_manifest_sha256,
            verification={
                "runtime_packed_modules": len(runtime_packed),
                "runtime_packed_examples": runtime_packed[:8],
                "runtime_architecture": architecture,
                "saved_config_quantization": True,
            },
        )
        saved_config_path = partial / "config.json"
        if not saved_config_path.is_file():
            raise QuantizationError("Quantized export did not produce config.json")
        saved_config = json.loads(saved_config_path.read_text(encoding="utf-8"))
        saved_quantization = _embedded_quantization(saved_config)
        if saved_quantization is None:
            raise QuantizationError(
                "Saved checkpoint has no embedded quantization metadata; refusing publication"
            )
        _, saved_quantization_value = saved_quantization
        saved_quantization_text = json.dumps(
            saved_quantization_value,
            sort_keys=True,
            default=str,
        ).lower()
        expected_format = config.kind.removeprefix("bnb-")
        if expected_format not in saved_quantization_text:
            raise QuantizationError(
                f"Saved quantization metadata does not identify requested format {config.kind}"
            )
        saved_weights = sorted(
            {
                *partial.glob("model*.safetensors"),
                *partial.glob("pytorch_model*.bin"),
            }
        )
        saved_weight_bytes = sum(path.stat().st_size for path in saved_weights if path.is_file())
        if not saved_weights or saved_weight_bytes < 1:
            raise QuantizationError(
                "Quantized export produced no non-empty model weight payload; refusing publication"
            )
        assert manifest.verification is not None
        manifest.verification["saved_weight_files"] = [path.name for path in saved_weights]
        manifest.verification["saved_weight_bytes"] = saved_weight_bytes
        del inspection_adapter, model
        gc.collect()
        torch = require("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reloaded_model, _, _, _, _ = load_pretrained_components(
            ModelLoadConfig(
                model=str(partial),
                trust_remote_code=trust_remote_code,
                dtype="auto",
                model_class=model_class,
                local_files_only=True,
            )
        )
        reloaded_packed = _runtime_packed_modules(reloaded_model)
        if not reloaded_packed:
            raise QuantizationError(
                "Saved checkpoint reloaded without packed backend modules; refusing publication"
            )
        if config.kind.startswith("bnb-") and not any(
            kind.startswith("bnb-") for _, kind in reloaded_packed
        ):
            raise QuantizationError(
                "Reloaded checkpoint does not contain bitsandbytes-packed modules"
            )
        if config.kind in {"nvfp4", "mxfp4"} and not any(
            kind.startswith("fp-quant") for _, kind in reloaded_packed
        ):
            raise QuantizationError("Reloaded checkpoint does not contain FP-Quant modules")
        manifest.verification["reload_verified"] = True
        manifest.verification["reloaded_packed_modules"] = len(reloaded_packed)
        del reloaded_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        (partial / "abliterator_quantization.json").write_text(
            json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_directory_noreplace(partial, output_path)
    except Exception as exc:
        raise QuantizationError(
            f"Quantized export failed ({exc}); partial data was retained at {partial} for diagnosis"
        ) from exc
    return manifest


def _sha256_path(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
