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
    ObliterateConfig,
    QuantizationConfig,
    QuantizationError,
    __version__,
    default_obliterated_output,
    estimate_directions,
    load_direction_artifact,
    normalize_direction,
    obliteration_plan,
    project_activations,
    run_obliteration_workflow,
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
    "ObliterateConfig",
    "QuantizationConfig",
    "QuantizationError",
    "estimate_directions",
    "default_obliterated_output",
    "load_direction_artifact",
    "normalize_direction",
    "obliteration_plan",
    "project_activations",
    "run_obliteration_workflow",
    "save_direction_artifact",
    "__version__",
]
