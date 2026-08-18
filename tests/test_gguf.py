from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

from abliterator.errors import GgufExportError
from abliterator.gguf import (
    GgufExportConfig,
    export_gguf_package,
    inspect_gguf,
    locate_gguf_tools,
)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "dense-model"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}),
        encoding="utf-8",
    )
    (source / "model.safetensors").write_bytes(b"dense-weights")
    (source / "tokenizer.json").write_text("{}", encoding="utf-8")
    (source / "abliterator_manifest.json").write_text(
        json.dumps({"format": "abliterator-checkpoint-v1"}),
        encoding="utf-8",
    )
    return source


def _tools(tmp_path: Path) -> Path:
    root = tmp_path / "llama.cpp"
    root.mkdir()
    (root / "convert_hf_to_gguf.py").write_text("# converter\n", encoding="utf-8")
    binary_dir = root / "build" / "bin"
    binary_dir.mkdir(parents=True)
    for name in ("llama-quantize", "llama-cli"):
        tool = binary_dir / name
        tool.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)
    return root


def _write_gguf(path: Path) -> None:
    path.write_bytes(struct.pack("<4sIQQ", b"GGUF", 3, 12, 24) + b"payload")


def test_inspect_gguf_validates_structural_header(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    _write_gguf(model)

    assert inspect_gguf(model) == {
        "version": 3,
        "tensor_count": 12,
        "metadata_count": 24,
    }

    invalid = tmp_path / "invalid.gguf"
    invalid.write_bytes(b"not-a-gguf")
    with pytest.raises(GgufExportError, match="truncated"):
        inspect_gguf(invalid)


def test_locate_gguf_tools_uses_checkout_and_compiled_binaries(tmp_path: Path) -> None:
    root = _tools(tmp_path)

    tools = locate_gguf_tools(root)

    assert tools.converter == (root / "convert_hf_to_gguf.py").resolve()
    assert tools.quantizer == (root / "build" / "bin" / "llama-quantize").resolve()
    assert tools.cli == (root / "build" / "bin" / "llama-cli").resolve()


def test_export_builds_verified_lm_studio_and_ollama_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    tools = _tools(tmp_path)
    output = tmp_path / "portable"
    environment = tmp_path / "environment"
    environment.mkdir()
    interpreter = environment / "python"
    interpreter.symlink_to(sys.executable)
    commands: list[tuple[list[str], str]] = []

    def fake_run(command: list[str], *, label: str) -> None:
        commands.append((command, label))
        if "--outfile" in command:
            _write_gguf(Path(command[command.index("--outfile") + 1]))
        else:
            _write_gguf(Path(command[2]))

    monkeypatch.setattr("abliterator.gguf._run_checked", fake_run)
    events: list[tuple[str, str]] = []
    result = export_gguf_package(
        GgufExportConfig(
            source=source,
            output=output,
            quantization="q4_k_m",
            model_name="half-bliterated",
            llama_cpp_dir=tools,
            python_executable=interpreter,
        ),
        progress=lambda event: events.append((event.stage, event.status)),
    )

    gguf = output / "half-bliterated-Q4_K_M.gguf"
    assert gguf.is_file()
    assert result["gguf"] == str(gguf)
    assert result["quantization"] == "q4_k_m"
    assert not any(output.glob(".*intermediate*.gguf"))
    assert (output / "Modelfile").read_text() == (
        "FROM ./half-bliterated-Q4_K_M.gguf\nPARAMETER num_ctx 8192\n"
    )
    receipt = json.loads((output / "abpro-gguf-export.json").read_text())
    assert receipt["format"] == "abpro-gguf-export-v1"
    assert receipt["gguf"]["header"]["tensor_count"] == 12
    assert receipt["gguf"]["sha256"] == result["sha256"]
    assert receipt["source_abliterator_manifest_sha256"] is not None
    assert receipt["compatibility"]["ollama"]["create_command"][0:2] == [
        "ollama",
        "create",
    ]
    assert len(commands) == 2
    assert commands[0][0][0] == str(interpreter.absolute())
    assert events == [
        ("preflight", "started"),
        ("preflight", "completed"),
        ("convert", "started"),
        ("convert", "completed"),
        ("quantize", "started"),
        ("quantize", "completed"),
        ("verify", "started"),
        ("verify", "completed"),
        ("publish", "started"),
        ("publish", "completed"),
    ]


def test_export_failure_removes_partial_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    tools = _tools(tmp_path)
    output = tmp_path / "portable"

    def fail(*_: object, **__: object) -> None:
        raise GgufExportError("converter failed")

    monkeypatch.setattr("abliterator.gguf._run_checked", fail)
    with pytest.raises(GgufExportError, match="converter failed"):
        export_gguf_package(GgufExportConfig(source=source, output=output, llama_cpp_dir=tools))

    assert not output.exists()
    assert not tuple(tmp_path.glob(".portable.gguf-export-*"))


def test_export_rejects_embedded_quantized_source_before_conversion(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "bitsandbytes"}}),
        encoding="utf-8",
    )

    with pytest.raises(GgufExportError, match="dense source"):
        export_gguf_package(
            GgufExportConfig(
                source=source,
                output=tmp_path / "portable",
                llama_cpp_dir=_tools(tmp_path),
            )
        )
