"""Public AB Pro API layered over the stable ``abliterator`` compatibility namespace."""

from abliterator import (
    AbliteratorError,
    ArchitectureError,
    ArchitectureSpec,
    CaptureConfig,
    ConfigurationError,
    DependencyError,
    DirectionArtifact,
    ModelLoadConfig,
    QuantizationConfig,
    QuantizationError,
    __version__,
    estimate_directions,
    load_direction_artifact,
    normalize_direction,
    project_activations,
    save_direction_artifact,
)
from abliterator.adapters import HuggingFaceAdapter
from abliterator.capture import ActivationCollector

ABProError = AbliteratorError

__all__ = [
    "ABProError",
    "AbliteratorError",
    "ActivationCollector",
    "ArchitectureError",
    "ArchitectureSpec",
    "CaptureConfig",
    "ConfigurationError",
    "DependencyError",
    "DirectionArtifact",
    "HuggingFaceAdapter",
    "ModelLoadConfig",
    "QuantizationConfig",
    "QuantizationError",
    "estimate_directions",
    "load_direction_artifact",
    "normalize_direction",
    "project_activations",
    "save_direction_artifact",
    "__version__",
]
