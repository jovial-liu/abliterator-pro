"""Project-specific exceptions with actionable failure messages."""


class AbliteratorError(RuntimeError):
    """Base class for expected bench failures."""


class ConfigurationError(AbliteratorError):
    """Raised when an experiment configuration is invalid."""


class DependencyError(AbliteratorError):
    """Raised when an optional runtime dependency is unavailable."""


class ArchitectureError(AbliteratorError):
    """Raised when a model layout cannot be inspected safely."""


class QuantizationError(AbliteratorError):
    """Raised when a requested low-bit operation would be unsafe or unsupported."""


class GgufExportError(AbliteratorError):
    """Raised when a GGUF conversion cannot be verified or published safely."""


class DatasetError(AbliteratorError):
    """Raised when a dataset no longer matches its registered schema."""
