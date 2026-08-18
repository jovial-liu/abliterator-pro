"""AB Pro's compatibility namespace for activation and weight ablation tools."""

from .artifacts import DirectionArtifact, load_direction_artifact, save_direction_artifact
from .config import ArchitectureSpec, CaptureConfig, ModelLoadConfig, QuantizationConfig
from .directions import estimate_directions, normalize_direction, project_activations
from .errors import (
    AbliteratorError,
    ArchitectureError,
    ConfigurationError,
    DependencyError,
    GgufExportError,
    QuantizationError,
)
from .gguf import GgufExportConfig, export_gguf_package, inspect_gguf, locate_gguf_tools
from .progress import JsonProgressRenderer, PlainProgressRenderer, WorkflowEvent
from .workflow import (
    ObliterateConfig,
    default_obliterated_output,
    obliteration_plan,
    run_obliteration_workflow,
)

__all__ = [
    "AbliteratorError",
    "ArchitectureError",
    "ArchitectureSpec",
    "CaptureConfig",
    "ConfigurationError",
    "DependencyError",
    "DirectionArtifact",
    "GgufExportConfig",
    "GgufExportError",
    "ModelLoadConfig",
    "ObliterateConfig",
    "JsonProgressRenderer",
    "PlainProgressRenderer",
    "QuantizationConfig",
    "QuantizationError",
    "WorkflowEvent",
    "estimate_directions",
    "export_gguf_package",
    "default_obliterated_output",
    "load_direction_artifact",
    "locate_gguf_tools",
    "normalize_direction",
    "obliteration_plan",
    "inspect_gguf",
    "project_activations",
    "run_obliteration_workflow",
    "save_direction_artifact",
]

__version__ = "0.1.0"
