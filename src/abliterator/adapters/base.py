"""Interfaces and reports for architecture-specific model access."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ProjectionKind = Literal["attention", "mlp", "expert", "embedding"]


@dataclass
class ProjectionTarget:
    """A logical matrix whose output dimension is the model residual width."""

    name: str
    layer: int
    kind: ProjectionKind
    weight: Any
    output_axis: int
    bias: Any | None = None
    module: Any | None = None
    packed: bool = False
    quantization: str = "dense"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.weight.shape)


@dataclass
class ArchitectureReport:
    model_type: str
    checkpoint_revision: str | None
    tokenizer_revision: str | None
    architectures: list[str]
    layer_path: str
    layer_count: int
    hidden_size: int
    attention_targets: int
    mlp_targets: int
    expert_targets: int
    packed_targets: int
    capture_supported: bool
    weight_edit_supported: bool
    warnings: list[str] = field(default_factory=list)
    target_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelAdapter(ABC):
    """Minimal surface required by the architecture-independent experiment engine."""

    model: Any
    tokenizer: Any

    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @property
    @abstractmethod
    def layers(self) -> list[Any]: ...

    @abstractmethod
    def inspect(self) -> ArchitectureReport: ...

    @abstractmethod
    def projection_targets(
        self,
        layers: Iterable[int] | None = None,
        kinds: Iterable[ProjectionKind] = ("attention", "mlp", "expert"),
    ) -> list[ProjectionTarget]: ...

    @abstractmethod
    def format_prompts(self, prompts: list[str]) -> list[str]: ...

    @abstractmethod
    def tokenize(self, prompts: list[str], max_length: int) -> dict[str, Any]: ...

    @abstractmethod
    def generate(self, prompts: list[str], max_new_tokens: int, max_length: int) -> list[str]: ...
