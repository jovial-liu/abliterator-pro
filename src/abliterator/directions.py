"""Precision-stable direction estimation and orthogonal projection primitives."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import Any

from .errors import ConfigurationError
from .runtime import require


def normalize_direction(direction: Any, eps: float = 1e-8) -> Any:
    torch = require("torch")
    vector = direction.detach().to(dtype=torch.float32)
    if vector.ndim != 1:
        raise ConfigurationError(
            f"A direction must be one-dimensional, got shape {tuple(vector.shape)}"
        )
    if not torch.isfinite(vector).all():
        raise ConfigurationError("Direction contains NaN or infinity")
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= eps:
        raise ConfigurationError("Cannot normalize a near-zero direction")
    return vector / norm


def orthonormalize_directions(directions: Any, eps: float = 1e-6) -> Any:
    """Return a rank-revealing hidden×rank basis for the supplied direction span."""
    torch = require("torch")
    matrix = directions
    if isinstance(directions, (list, tuple)):
        matrix = torch.stack([normalize_direction(item) for item in directions], dim=1)
    elif matrix.ndim == 1:
        matrix = matrix[:, None]
    elif matrix.ndim != 2:
        raise ConfigurationError("Directions must have shape [hidden] or [hidden, rank]")
    matrix = matrix.detach().to(dtype=torch.float32)
    if not torch.isfinite(matrix).all():
        raise ConfigurationError("Directions contain NaN or infinity")
    column_norms = torch.linalg.vector_norm(matrix, dim=0)
    nonzero_columns = column_norms > eps
    if not bool(nonzero_columns.any()):
        raise ConfigurationError("Directions are all near-zero or linearly dependent")
    matrix = matrix[:, nonzero_columns] / column_norms[nonzero_columns].unsqueeze(0)
    q, singular_values, _ = torch.linalg.svd(matrix, full_matrices=False)
    scale = float(singular_values.max()) if singular_values.numel() else 0.0
    numeric_tolerance = torch.finfo(matrix.dtype).eps * max(matrix.shape) * scale
    keep = singular_values > max(eps, numeric_tolerance)
    if not bool(keep.any()):
        raise ConfigurationError("Directions are all near-zero or linearly dependent")
    return q[:, keep]


def estimate_directions(
    target_means: Mapping[str, Any],
    contrast_means: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute normalized target-minus-contrast vectors for matching activation sites."""
    target_keys = set(target_means)
    contrast_keys = set(contrast_means)
    if target_keys != contrast_keys:
        missing_target = sorted(contrast_keys - target_keys)
        missing_contrast = sorted(target_keys - contrast_keys)
        raise ConfigurationError(
            f"Activation sites differ; missing target={missing_target}, "
            f"missing contrast={missing_contrast}"
        )
    return {
        key: normalize_direction(target_means[key].float() - contrast_means[key].float())
        for key in sorted(target_keys)
    }


def project_activations(activation: Any, directions: Any, strength: float = 1.0) -> Any:
    """Remove a direction subspace from the final hidden dimension."""
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ConfigurationError("strength must be finite and between 0 and 1")
    q = orthonormalize_directions(directions).to(
        device=activation.device,
        dtype=activation.dtype,
    )
    if activation.shape[-1] != q.shape[0]:
        raise ConfigurationError(
            f"Activation width {activation.shape[-1]} does not match direction width {q.shape[0]}"
        )
    coefficients = activation @ q
    return activation - strength * (coefficients @ q.transpose(0, 1))


def direction_leakage(
    weight: Any,
    directions: Any,
    output_axis: int = 0,
    chunk_columns: int = 8192,
) -> float:
    """Return ||QᵀW|| / ||W|| after moving the logical output axis first."""
    leakage, _ = direction_leakage_and_norm(
        weight,
        directions,
        output_axis=output_axis,
        chunk_columns=chunk_columns,
    )
    return leakage


def direction_leakage_and_norm(
    weight: Any,
    directions: Any,
    output_axis: int = 0,
    chunk_columns: int = 8192,
) -> tuple[float, float]:
    """Return direction leakage and the FP32-accumulated L2 norm of a weight."""
    torch = require("torch")
    if chunk_columns < 1:
        raise ConfigurationError("chunk_columns must be positive")
    q = orthonormalize_directions(directions).to(device=weight.device, dtype=torch.float32)
    moved = torch.movedim(weight.detach(), output_axis, 0)
    if moved.shape[0] != q.shape[0]:
        raise ConfigurationError(
            f"Weight output width {moved.shape[0]} does not match direction width {q.shape[0]}"
        )
    if moved.ndim == 1:
        moved = moved.reshape(moved.shape[0], 1)
    remaining = tuple(moved.shape[1:])
    prefix_ranges = [range(size) for size in remaining[:-1]]
    prefixes = itertools.product(*prefix_ranges) if prefix_ranges else [()]
    numerator_squared = 0.0
    denominator_squared = 0.0
    with torch.no_grad():
        for prefix in prefixes:
            width = remaining[-1]
            for start in range(0, width, chunk_columns):
                index = (slice(None), *prefix, slice(start, min(start + chunk_columns, width)))
                scratch = moved[index].float()
                coefficients = q.transpose(0, 1) @ scratch
                numerator_squared += float(torch.sum(coefficients * coefficients))
                denominator_squared += float(torch.sum(scratch * scratch))
    if denominator_squared <= 1e-24:
        return 0.0, 0.0
    return (
        float((numerator_squared / denominator_squared) ** 0.5),
        float(denominator_squared**0.5),
    )
