"""Small dependency and device helpers kept separate to make imports lightweight."""

from __future__ import annotations

import ctypes
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from .errors import DependencyError


def require(module: str, extra: str | None = None) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        suffix = f" Install with `pip install -e '.[{extra}]'`." if extra else ""
        raise DependencyError(f"Missing required dependency {module!r}.{suffix}") from exc


def torch_dtype(name: str) -> Any:
    torch = require("torch")
    aliases = {
        "auto": None,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise DependencyError(
            f"Unsupported dtype {name!r}; use auto, float32, float16, or bfloat16"
        ) from exc


def tensor_device_for_inputs(model: Any) -> Any:
    """Return the embedding device, which is valid for Accelerate-sharded models."""
    embeddings = model.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if weight is None:
        raise DependencyError("Model input embeddings do not expose a weight device")
    if weight.device.type != "meta":
        return weight.device
    execution_device = getattr(getattr(embeddings, "_hf_hook", None), "execution_device", None)
    if execution_device is not None and str(execution_device).lower() not in {"meta", "disk"}:
        torch = require("torch")
        return torch.device(execution_device)
    module_name = next(
        (name for name, module in model.named_modules() if module is embeddings),
        None,
    )
    device_map = getattr(model, "hf_device_map", {}) or {}
    if module_name is not None:
        candidates = [
            (key, value)
            for key, value in device_map.items()
            if module_name == key or module_name.startswith(f"{key}.")
        ]
        if candidates:
            _, mapped = max(candidates, key=lambda item: len(item[0]))
            if str(mapped).lower() not in {"meta", "disk"}:
                torch = require("torch")
                if isinstance(mapped, int):
                    return torch.device("cuda", mapped)
                return torch.device(mapped)
    raise DependencyError(
        "Input embeddings are on the meta device and no Accelerate execution device could be "
        "resolved"
    )


def publish_directory_noreplace(source: str | Path, destination: str | Path) -> None:
    """Atomically publish a directory while refusing a concurrently-created destination."""
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if destination_path.exists():
        raise FileExistsError(destination_path)

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source_path)
    destination_bytes = os.fsencode(destination_path)
    if sys.platform == "darwin":
        rename_exclusive = 0x00000004
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise RuntimeError("This macOS runtime does not expose atomic exclusive rename")
        result = renamex_np(source_bytes, destination_bytes, rename_exclusive)
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("This Linux runtime does not expose renameat2(RENAME_NOREPLACE)")
        at_fdcwd = -100
        rename_noreplace = 1
        result = renameat2(
            at_fdcwd,
            source_bytes,
            at_fdcwd,
            destination_bytes,
            rename_noreplace,
        )
    elif os.name == "nt":
        # MoveFile on Windows fails when the destination exists.
        os.rename(source_path, destination_path)
        return
    else:
        raise RuntimeError(
            "Atomic no-clobber directory publication is unsupported on this platform"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination_path))
