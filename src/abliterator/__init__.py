"""AB Pro's compatibility namespace for activation and weight ablation tools."""

from .artifacts import DirectionArtifact, load_direction_artifact, save_direction_artifact
from .config import ArchitectureSpec, CaptureConfig, ModelLoadConfig, QuantizationConfig
from .directions import estimate_directions, normalize_direction, project_activations
from .errors import (
    AbliteratorError,
    ArchitectureError,
    ConfigurationError,
    DependencyError,
    QuantizationError,
)

__all__ = [
    "AbliteratorError",
    "ArchitectureError",
    "ArchitectureSpec",
    "CaptureConfig",
    "ConfigurationError",
    "DependencyError",
    "DirectionArtifact",
    "ModelLoadConfig",
    "QuantizationConfig",
    "QuantizationError",
    "estimate_directions",
    "load_direction_artifact",
    "normalize_direction",
    "project_activations",
    "save_direction_artifact",
]

__version__ = "0.1.0"
