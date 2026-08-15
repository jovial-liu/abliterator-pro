"""Typed configuration objects shared by the CLI and Python API."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigurationError

QuantizationKind = Literal["none", "bnb-fp4", "bnb-nf4", "nvfp4", "mxfp4"]
CapturePoint = Literal["resid_pre", "resid_post"]
PoolingKind = Literal["last", "last-k", "mean"]


@dataclass(frozen=True)
class QuantizationConfig:
    """Describes a concrete low-bit backend; `fp4` is intentionally not ambiguous."""

    kind: QuantizationKind = "none"
    compute_dtype: str = "bfloat16"
    double_quant: bool = False
    keep_master_weights: bool = False
    pseudoquantization: bool = False

    def validate(self) -> None:
        allowed = {"none", "bnb-fp4", "bnb-nf4", "nvfp4", "mxfp4"}
        if self.kind not in allowed:
            raise ConfigurationError(
                f"Unknown quantization kind {self.kind!r}; choose one of {sorted(allowed)}"
            )
        if self.kind == "none":
            if self.double_quant or self.keep_master_weights or self.pseudoquantization:
                raise ConfigurationError("Low-bit options cannot be used with quantization='none'")
            return
        if self.kind in {"bnb-fp4", "bnb-nf4"}:
            if self.compute_dtype.lower() == "auto":
                raise ConfigurationError(
                    "bitsandbytes compute_dtype='auto' is ambiguous; choose an explicit dtype"
                )
            if self.keep_master_weights:
                raise ConfigurationError(
                    "bitsandbytes 4-bit loading does not retain editable master weights"
                )
            if self.pseudoquantization:
                raise ConfigurationError(
                    "pseudoquantization is an FP-Quant option, not a bitsandbytes option"
                )
        else:
            if self.double_quant:
                raise ConfigurationError(
                    "double_quant is a bitsandbytes option, not an FP-Quant option"
                )
            if self.compute_dtype.lower() not in {"bfloat16", "bf16"}:
                raise ConfigurationError(
                    "FP-Quant export currently requires compute_dtype='bfloat16'"
                )


@dataclass(frozen=True)
class ModelLoadConfig:
    model: str
    revision: str | None = None
    dtype: str = "bfloat16"
    device_map: str | dict[str, Any] | None = "auto"
    trust_remote_code: bool = False
    attn_implementation: str | None = None
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    raw_prompts: bool = False
    enable_thinking: bool = False
    model_class: Literal["causal-lm", "image-text-to-text"] = "causal-lm"
    local_files_only: bool = False

    def validate(self) -> None:
        if not self.model.strip():
            raise ConfigurationError("model must be a Hugging Face ID or local path")
        if self.model_class not in {"causal-lm", "image-text-to-text"}:
            raise ConfigurationError("model_class must be 'causal-lm' or 'image-text-to-text'")
        if not isinstance(self.raw_prompts, bool):
            raise ConfigurationError("raw_prompts must be true or false")
        if not isinstance(self.enable_thinking, bool):
            raise ConfigurationError("enable_thinking must be true or false")
        if (
            self.trust_remote_code
            and not Path(self.model).exists()
            and (not self.revision or re.fullmatch(r"[0-9a-fA-F]{40}", self.revision) is None)
        ):
            raise ConfigurationError(
                "trust_remote_code requires a full 40-character commit SHA; branch names "
                "and mutable tags are not sufficient"
            )
        if (
            self.trust_remote_code
            and self.tokenizer
            and self.tokenizer != self.model
            and not Path(self.tokenizer).exists()
            and (
                not self.tokenizer_revision
                or re.fullmatch(r"[0-9a-fA-F]{40}", self.tokenizer_revision) is None
            )
        ):
            raise ConfigurationError(
                "A separate tokenizer loaded with trust_remote_code requires its own full "
                "40-character --tokenizer-revision"
            )
        self.quantization.validate()


@dataclass(frozen=True)
class CaptureConfig:
    point: CapturePoint = "resid_pre"
    pooling: PoolingKind = "last"
    last_k: int = 1
    batch_size: int = 4
    max_length: int = 1024
    layers: tuple[int, ...] | None = None

    def validate(self) -> None:
        if self.point not in {"resid_pre", "resid_post"}:
            raise ConfigurationError("point must be resid_pre or resid_post")
        if self.pooling not in {"last", "last-k", "mean"}:
            raise ConfigurationError("pooling must be last, last-k, or mean")
        if self.last_k < 1:
            raise ConfigurationError("last_k must be positive")
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be positive")
        if self.max_length < 8:
            raise ConfigurationError("max_length must be at least 8")
        if self.layers is not None:
            if not self.layers:
                raise ConfigurationError("layers must not be empty")
            if len(set(self.layers)) != len(self.layers):
                raise ConfigurationError("layers must not contain duplicates")
            if any(
                not isinstance(layer, int) or isinstance(layer, bool) or layer < 0
                for layer in self.layers
            ):
                raise ConfigurationError("layers must contain only non-negative integers")


@dataclass(frozen=True)
class ArchitectureSpec:
    """Optional override for models whose decoder layout is not discoverable."""

    layer_path: str | None = None
    attention_output_patterns: tuple[str, ...] = ()
    mlp_output_patterns: tuple[str, ...] = ()
    hidden_size: int | None = None
    include_experts: bool = True

    def validate(self) -> None:
        if self.layer_path is not None and (
            not isinstance(self.layer_path, str) or not self.layer_path.strip()
        ):
            raise ConfigurationError("architecture.layer_path must be a non-empty string")
        if self.hidden_size is not None and (
            not isinstance(self.hidden_size, int)
            or isinstance(self.hidden_size, bool)
            or self.hidden_size < 1
        ):
            raise ConfigurationError("architecture.hidden_size must be a positive integer")
        if not isinstance(self.include_experts, bool):
            raise ConfigurationError("architecture.include_experts must be true or false")
        for name, patterns in (
            ("attention_output_patterns", self.attention_output_patterns),
            ("mlp_output_patterns", self.mlp_output_patterns),
        ):
            if not isinstance(patterns, (list, tuple)) or isinstance(patterns, (str, bytes)):
                raise ConfigurationError(f"architecture.{name} must be a list of strings")
            if not all(isinstance(item, str) and item.strip() for item in patterns):
                raise ConfigurationError(f"architecture.{name} must contain only non-empty strings")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ArchitectureSpec:
        known = {
            "layer_path",
            "attention_output_patterns",
            "mlp_output_patterns",
            "hidden_size",
            "include_experts",
        }
        extra = set(value) - known
        if extra:
            raise ConfigurationError(f"Unknown architecture keys: {sorted(extra)}")
        layer_path = value.get("layer_path")
        if layer_path is not None and (not isinstance(layer_path, str) or not layer_path.strip()):
            raise ConfigurationError("architecture.layer_path must be a non-empty string")
        hidden_size = value.get("hidden_size")
        if hidden_size is not None and (
            not isinstance(hidden_size, int) or isinstance(hidden_size, bool) or hidden_size < 1
        ):
            raise ConfigurationError("architecture.hidden_size must be a positive integer")
        include_experts = value.get("include_experts", True)
        if not isinstance(include_experts, bool):
            raise ConfigurationError("architecture.include_experts must be true or false")

        def patterns(name: str) -> tuple[str, ...]:
            raw = value.get(name, ())
            if not isinstance(raw, (list, tuple)) or isinstance(raw, (str, bytes)):
                raise ConfigurationError(f"architecture.{name} must be a list of strings")
            if not all(isinstance(item, str) and item.strip() for item in raw):
                raise ConfigurationError(f"architecture.{name} must contain only non-empty strings")
            return tuple(raw)

        result = cls(
            layer_path=layer_path,
            attention_output_patterns=patterns("attention_output_patterns"),
            mlp_output_patterns=patterns("mlp_output_patterns"),
            hidden_size=hidden_size,
            include_experts=include_experts,
        )
        result.validate()
        return result


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
