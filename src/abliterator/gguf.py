"""Verified GGUF export for LM Studio and Ollama."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .artifacts import local_checkpoint_receipt
from .errors import GgufExportError
from .quantization import checkpoint_quantization_kind
from .runtime import publish_directory_noreplace

GgufQuantization = Literal[
    "f16",
    "bf16",
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q5_k_s",
    "q4_k_m",
    "q4_k_s",
]

GGUF_QUANTIZATIONS: tuple[str, ...] = (
    "f16",
    "bf16",
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q5_k_s",
    "q4_k_m",
    "q4_k_s",
)

EXPORT_STAGES: tuple[str, ...] = (
    "preflight",
    "convert",
    "quantize",
    "verify",
    "publish",
)


@dataclass(frozen=True)
class GgufExportConfig:
    """Configuration for one atomic GGUF application package."""

    source: str | Path
    output: str | Path
    quantization: GgufQuantization = "q4_k_m"
    model_name: str | None = None
    llama_cpp_dir: str | Path | None = None
    python_executable: str | Path | None = None
    context_length: int = 8192
    keep_intermediate: bool = False
    keep_partial: bool = False

    def validate(self) -> None:
        source = Path(self.source).expanduser().resolve()
        output = Path(self.output).expanduser().resolve()
        if not source.is_dir():
            raise GgufExportError(f"Source checkpoint directory does not exist: {source}")
        if not (source / "config.json").is_file():
            raise GgufExportError(f"Source checkpoint has no config.json: {source}")
        weights = tuple(source.glob("*.safetensors")) + tuple(source.glob("*.bin"))
        if not weights or not any(path.is_file() and path.stat().st_size > 0 for path in weights):
            raise GgufExportError("Source checkpoint contains no non-empty model weights")
        if output.exists():
            raise GgufExportError(f"Refusing to overwrite existing export directory {output}")
        if output == source or source in output.parents or output in source.parents:
            raise GgufExportError("GGUF output and source must be separate, non-nested directories")
        if self.quantization not in GGUF_QUANTIZATIONS:
            raise GgufExportError(
                f"Unsupported GGUF quantization {self.quantization!r}; choose from "
                f"{', '.join(GGUF_QUANTIZATIONS)}"
            )
        if self.context_length < 256:
            raise GgufExportError("context_length must be at least 256")
        _safe_model_name(self.model_name or source.name)
        if self.python_executable is not None:
            # Keep virtual-environment interpreter symlinks intact. Resolving
            # one points at the base interpreter and loses the environment's
            # installed conversion dependencies.
            python = Path(os.path.abspath(Path(self.python_executable).expanduser()))
            if not python.is_file() or not os.access(python, os.X_OK):
                raise GgufExportError(f"Python executable is not usable: {python}")


@dataclass(frozen=True)
class GgufTools:
    converter: Path
    quantizer: Path | None
    cli: Path | None
    llama_cpp_root: Path
    revision: str | None


@dataclass(frozen=True)
class GgufExportEvent:
    stage: str
    status: Literal["started", "completed", "skipped", "failed"]
    index: int
    total: int
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GgufProgressCallback = Callable[[GgufExportEvent], None]


def locate_gguf_tools(llama_cpp_dir: str | Path | None = None) -> GgufTools:
    """Locate a llama.cpp checkout plus its optional compiled runtime tools."""
    candidates: list[Path] = []
    if llama_cpp_dir is not None:
        candidates.append(Path(llama_cpp_dir).expanduser().resolve())
    environment_root = os.environ.get("LLAMA_CPP_DIR")
    if environment_root:
        candidates.append(Path(environment_root).expanduser().resolve())
    candidates.extend(
        [
            Path.home() / "Library" / "Caches" / "AB-Pro" / "llama.cpp",
            Path.home() / ".cache" / "abpro" / "llama.cpp",
            Path.cwd() / "llama.cpp",
        ]
    )
    root = next(
        (path for path in candidates if (path / "convert_hf_to_gguf.py").is_file()),
        None,
    )
    if root is None:
        searched = ", ".join(str(path) for path in candidates)
        raise GgufExportError(
            "Could not find llama.cpp's convert_hf_to_gguf.py. Pass --llama-cpp-dir or set "
            f"LLAMA_CPP_DIR. Searched: {searched}"
        )
    quantizer = _find_tool(root, "llama-quantize")
    cli = _find_tool(root, "llama-cli")
    return GgufTools(
        converter=(root / "convert_hf_to_gguf.py").resolve(),
        quantizer=quantizer,
        cli=cli,
        llama_cpp_root=root.resolve(),
        revision=_git_revision(root),
    )


def inspect_gguf(path: str | Path) -> dict[str, int]:
    """Validate the fixed GGUF header without loading the tensor payload."""
    candidate = Path(path)
    if not candidate.is_file():
        raise GgufExportError(f"GGUF output does not exist: {candidate}")
    if candidate.stat().st_size < 24:
        raise GgufExportError(f"GGUF output is truncated: {candidate}")
    with candidate.open("rb") as handle:
        header = handle.read(24)
    magic, version, tensor_count, metadata_count = struct.unpack("<4sIQQ", header)
    if magic != b"GGUF":
        raise GgufExportError(f"Output does not have GGUF magic bytes: {candidate}")
    if version not in {2, 3}:
        raise GgufExportError(f"Unsupported GGUF version {version}: {candidate}")
    if tensor_count < 1 or metadata_count < 1:
        raise GgufExportError(
            f"GGUF header has no tensors or metadata: tensors={tensor_count}, "
            f"metadata={metadata_count}"
        )
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
    }


def export_gguf_package(
    config: GgufExportConfig,
    *,
    progress: GgufProgressCallback | None = None,
) -> dict[str, Any]:
    """Convert a dense HF checkpoint and atomically publish a portable GGUF package."""
    current_stage = "preflight"

    def emit(
        stage: str,
        status: Literal["started", "completed", "skipped", "failed"],
        message: str,
        **details: Any,
    ) -> None:
        nonlocal current_stage
        current_stage = stage
        if progress is None:
            return
        event = GgufExportEvent(
            stage=stage,
            status=status,
            index=EXPORT_STAGES.index(stage) + 1,
            total=len(EXPORT_STAGES),
            message=message,
            details=details,
        )
        try:
            progress(event)
        except Exception:
            return

    emit("preflight", "started", "Validating source, output, and llama.cpp tools")
    config.validate()
    source = Path(config.source).expanduser().resolve()
    output = Path(config.output).expanduser().resolve()
    model_name = _safe_model_name(config.model_name or source.name)
    source_config = _read_json_object(source / "config.json")
    embedded_quantization = checkpoint_quantization_kind(source_config)
    if embedded_quantization:
        raise GgufExportError(
            "GGUF conversion requires a dense source checkpoint; config declares "
            f"{embedded_quantization!r} quantization"
        )
    tools = locate_gguf_tools(config.llama_cpp_dir)
    if config.quantization not in {"f16", "bf16"} and tools.quantizer is None:
        raise GgufExportError(
            f"{config.quantization} export requires llama-quantize; build or install llama.cpp"
        )
    python = (
        Path(os.path.abspath(Path(config.python_executable).expanduser()))
        if config.python_executable is not None
        else Path(os.path.abspath(sys.executable))
    )
    source_receipt_before = local_checkpoint_receipt(str(source))
    if source_receipt_before is None:
        raise GgufExportError("Could not fingerprint the local source checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.gguf-export-{uuid.uuid4().hex[:12]}"
    staging.mkdir()
    quantization_label = config.quantization.upper()
    final_name = f"{model_name}-{quantization_label}.gguf"
    final_path = staging / final_name
    intermediate = staging / f".{model_name}-intermediate-f16.gguf"
    emit(
        "preflight",
        "completed",
        "Dense checkpoint and conversion tools are ready",
        source=str(source),
        quantization=config.quantization,
        llama_cpp_revision=tools.revision,
    )
    try:
        direct_output = config.quantization in {"f16", "bf16"}
        conversion_output = final_path if direct_output else intermediate
        conversion_type = config.quantization if direct_output else "f16"
        emit(
            "convert",
            "started",
            f"Converting Hugging Face weights to {conversion_type.upper()} GGUF",
        )
        _run_checked(
            [
                str(python),
                str(tools.converter),
                str(source),
                "--outfile",
                str(conversion_output),
                "--outtype",
                conversion_type,
            ],
            label="Hugging Face to GGUF conversion",
        )
        inspect_gguf(conversion_output)
        emit(
            "convert",
            "completed",
            f"Created {conversion_type.upper()} GGUF intermediate",
            bytes=conversion_output.stat().st_size,
        )

        if direct_output:
            emit("quantize", "skipped", "No secondary quantization was requested")
        else:
            if tools.quantizer is None:
                raise GgufExportError("llama-quantize disappeared after preflight")
            emit("quantize", "started", f"Quantizing GGUF weights to {quantization_label}")
            _run_checked(
                [
                    str(tools.quantizer),
                    str(intermediate),
                    str(final_path),
                    quantization_label,
                ],
                label="GGUF quantization",
            )
            inspect_gguf(final_path)
            emit(
                "quantize",
                "completed",
                f"Created {quantization_label} GGUF",
                bytes=final_path.stat().st_size,
            )
            if not config.keep_intermediate:
                intermediate.unlink()

        emit("verify", "started", "Hashing and packaging the portable model")
        header = inspect_gguf(final_path)
        final_bytes = final_path.stat().st_size
        final_sha256 = _sha256(final_path)
        source_receipt_after = local_checkpoint_receipt(str(source))
        if source_receipt_after != source_receipt_before:
            raise GgufExportError("Source checkpoint changed during GGUF conversion")
        modelfile = staging / "Modelfile"
        modelfile.write_text(
            f"FROM ./{final_name}\nPARAMETER num_ctx {config.context_length}\n",
            encoding="utf-8",
        )
        source_manifest = source / "abliterator_manifest.json"
        receipt: dict[str, Any] = {
            "format": "abpro-gguf-export-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "source_checkpoint_receipt": source_receipt_before,
            "source_abliterator_manifest_sha256": (
                _sha256(source_manifest) if source_manifest.is_file() else None
            ),
            "gguf": {
                "file": final_name,
                "quantization": config.quantization,
                "bytes": final_bytes,
                "sha256": final_sha256,
                "header": header,
            },
            "tools": {
                "converter": str(tools.converter),
                "converter_sha256": _sha256(tools.converter),
                "quantizer": None if tools.quantizer is None else str(tools.quantizer),
                "llama_cli": None if tools.cli is None else str(tools.cli),
                "llama_cpp_root": str(tools.llama_cpp_root),
                "llama_cpp_revision": tools.revision,
                "python": str(python),
            },
            "compatibility": {
                "lm_studio": {
                    "detected": shutil.which("lms") is not None,
                    "import_command": [
                        "lms",
                        "import",
                        str(output / final_name),
                        "--hard-link",
                        "--user-repo",
                        f"ab-pro/{model_name}",
                        "-y",
                    ],
                },
                "ollama": {
                    "detected": shutil.which("ollama") is not None,
                    "create_command": [
                        "ollama",
                        "create",
                        model_name,
                        "-f",
                        str(output / "Modelfile"),
                    ],
                    "run_command": ["ollama", "run", model_name],
                },
            },
        }
        (staging / "abpro-gguf-export.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            _render_package_readme(model_name, final_name, config.context_length),
            encoding="utf-8",
        )
        if (source / "LICENSE").is_file():
            shutil.copy2(source / "LICENSE", staging / "LICENSE")
        emit(
            "verify",
            "completed",
            "GGUF header, source stability, and SHA-256 receipt verified",
            sha256=final_sha256,
            tensor_count=header["tensor_count"],
        )

        emit("publish", "started", "Atomically publishing the GGUF application package")
        publish_directory_noreplace(staging, output)
        emit(
            "publish",
            "completed",
            "GGUF package is ready for LM Studio and Ollama",
            output=str(output),
        )
    except Exception as exc:
        emit(current_stage, "failed", str(exc))
        if not config.keep_partial and staging.exists():
            shutil.rmtree(staging)
        partial_note = f" Partial data retained at {staging}." if staging.exists() else ""
        if isinstance(exc, GgufExportError):
            raise GgufExportError(f"{exc}{partial_note}") from exc
        raise GgufExportError(f"GGUF export failed: {exc}.{partial_note}") from exc

    return {
        "output": str(output),
        "gguf": str(output / final_name),
        "quantization": config.quantization,
        "bytes": final_bytes,
        "sha256": final_sha256,
        "modelfile": str(output / "Modelfile"),
        "receipt": str(output / "abpro-gguf-export.json"),
        "lm_studio_import": receipt["compatibility"]["lm_studio"]["import_command"],
        "ollama_create": receipt["compatibility"]["ollama"]["create_command"],
    }


def _find_tool(root: Path, name: str) -> Path | None:
    candidates = [
        root / "build" / "bin" / name,
        root / "bin" / name,
        root / name,
    ]
    discovered = shutil.which(name)
    if discovered:
        candidates.append(Path(discovered))
    return next(
        (
            path.expanduser().resolve()
            for path in candidates
            if path.is_file() and os.access(path, os.X_OK)
        ),
        None,
    )


def _git_revision(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _run_checked(command: Sequence[str], *, label: str) -> None:
    result = subprocess.run(list(command), check=False)
    if result.returncode != 0:
        raise GgufExportError(f"{label} exited with status {result.returncode}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GgufExportError(f"Could not read checkpoint JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GgufExportError(f"Checkpoint JSON must contain an object: {path}")
    return value


def _safe_model_name(value: str) -> str:
    name = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name):
        raise GgufExportError(
            "model_name must start with a letter or number and contain only letters, numbers, "
            "periods, underscores, and hyphens"
        )
    return name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _render_package_readme(model_name: str, gguf_name: str, context_length: int) -> str:
    return f"""# {model_name} — GGUF application package

This directory was generated by AB Pro from a local dense Hugging Face checkpoint.

## LM Studio

```bash
lms import {gguf_name} --hard-link --user-repo ab-pro/{model_name} -y
```

You can also drag `{gguf_name}` into LM Studio or select it from **My Models** after import.

## Ollama

```bash
ollama create {model_name} -f Modelfile
ollama run {model_name}
```

The included `Modelfile` points to the local GGUF and sets `num_ctx` to {context_length}.
See `abpro-gguf-export.json` for hashes, source provenance, and converter details.
"""
