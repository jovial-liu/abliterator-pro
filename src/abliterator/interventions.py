"""Reversible residual projection and dense weight ablation."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from copy import copy
from dataclasses import asdict, dataclass
from typing import Any

from .adapters.base import ModelAdapter, ProjectionKind, ProjectionTarget
from .directions import direction_leakage_and_norm, orthonormalize_directions
from .errors import ArchitectureError, ConfigurationError
from .quantization import assert_editable_targets, describe_module_quantization
from .runtime import require


def _is_hidden_tensor(value: Any, hidden_size: int) -> bool:
    torch = require("torch")
    return isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == hidden_size


def _replace_first_hidden(value: Any, hidden_size: int, replacement: Any) -> tuple[Any, bool]:
    if _is_hidden_tensor(value, hidden_size):
        return replacement(value), True
    if isinstance(value, tuple):
        tuple_items = list(value)
        for index, item in enumerate(tuple_items):
            tuple_items[index], changed = _replace_first_hidden(item, hidden_size, replacement)
            if changed:
                fields = getattr(value, "_fields", None)
                replace = getattr(value, "_replace", None)
                if fields is not None and callable(replace):
                    return replace(**{fields[index]: tuple_items[index]}), True
                return tuple(tuple_items), True
    elif isinstance(value, list):
        list_items = list(value)
        for index, item in enumerate(list_items):
            list_items[index], changed = _replace_first_hidden(item, hidden_size, replacement)
            if changed:
                return list_items, True
    elif isinstance(value, dict):
        dict_items = copy(value)
        for key in ("hidden_states", "last_hidden_state"):
            if key in dict_items:
                dict_items[key], changed = _replace_first_hidden(
                    dict_items[key], hidden_size, replacement
                )
                if changed:
                    return dict_items, True
    return value, False


class ResidualProjector(AbstractContextManager["ResidualProjector"]):
    """Temporarily remove a direction subspace at decoder block boundaries."""

    def __init__(
        self,
        adapter: ModelAdapter,
        directions: Any | Mapping[int, Any],
        *,
        layers: Iterable[int] | None = None,
        point: str = "resid_pre",
        strength: float = 1.0,
    ) -> None:
        if point not in {"resid_pre", "resid_post"}:
            raise ConfigurationError("point must be resid_pre or resid_post")
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ConfigurationError(
                "Residual projection strength must be finite and between 0 and 1"
            )
        self.adapter = adapter
        self.directions = directions
        self.layer_indices = list(range(len(adapter.layers))) if layers is None else list(layers)
        if not self.layer_indices:
            raise ConfigurationError("Residual projection requires at least one layer")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ConfigurationError("Residual projection layers must not contain duplicates")
        self.point = point
        self.strength = strength
        self._handles: list[Any] = []

    def _direction(self, layer: int) -> Any:
        if isinstance(self.directions, Mapping):
            try:
                return self.directions[layer]
            except KeyError as exc:
                raise ConfigurationError(f"No direction was supplied for layer {layer}") from exc
        return self.directions

    def __enter__(self) -> ResidualProjector:
        if self._handles:
            raise ConfigurationError("ResidualProjector cannot be entered twice")
        invalid = [
            index for index in self.layer_indices if index < 0 or index >= len(self.adapter.layers)
        ]
        if invalid:
            raise ArchitectureError(f"Intervention layer indices out of range: {invalid}")
        prepared: list[tuple[int, Any, Any]] = []
        for index in self.layer_indices:
            prepared.append(
                (
                    index,
                    self.adapter.layers[index],
                    orthonormalize_directions(self._direction(index)),
                )
            )

        try:
            for index, layer, basis in prepared:
                basis_cache: dict[tuple[str, str], Any] = {}

                def replacement(
                    hidden: Any,
                    *,
                    _basis: Any = basis,
                    _cache: dict[tuple[str, str], Any] = basis_cache,
                ) -> Any:
                    cache_key = (str(hidden.device), str(hidden.dtype))
                    q = _cache.get(cache_key)
                    if q is None:
                        q = _basis.to(device=hidden.device, dtype=hidden.dtype)
                        _cache[cache_key] = q
                    coefficients = hidden @ q
                    return hidden - self.strength * (coefficients @ q.transpose(0, 1))

                if self.point == "resid_pre":

                    def pre_hook(
                        module: Any,
                        args: tuple[Any, ...],
                        kwargs: dict[str, Any],
                        *,
                        _replacement: Any = replacement,
                        _index: int = index,
                    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                        new_args, changed = _replace_first_hidden(
                            args, self.adapter.hidden_size, _replacement
                        )
                        if not changed:
                            new_kwargs, changed = _replace_first_hidden(
                                kwargs, self.adapter.hidden_size, _replacement
                            )
                            if not changed:
                                raise ArchitectureError(
                                    f"No residual input found for layer {_index}"
                                )
                            return args, new_kwargs
                        return new_args, kwargs

                    self._handles.append(
                        layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
                    )
                else:

                    def post_hook(
                        module: Any,
                        args: tuple[Any, ...],
                        kwargs: dict[str, Any],
                        output: Any,
                        *,
                        _replacement: Any = replacement,
                        _index: int = index,
                    ) -> Any:
                        new_output, changed = _replace_first_hidden(
                            output, self.adapter.hidden_size, _replacement
                        )
                        if not changed:
                            raise ArchitectureError(f"No residual output found for layer {_index}")
                        return new_output

                    self._handles.append(layer.register_forward_hook(post_hook, with_kwargs=True))
        except Exception:
            self._remove_hooks()
            raise
        return self

    def _remove_hooks(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._remove_hooks()


@dataclass
class WeightAblationRecord:
    name: str
    layer: int
    kind: str
    shape: tuple[int, ...]
    output_axis: int
    quantization: str
    leakage_before: float
    leakage_after: float
    weight_l2_before: float
    weight_l2_after: float
    strength: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _project_dense_weight_in_place(
    weight: Any,
    q: Any,
    *,
    output_axis: int,
    strength: float,
    chunk_columns: int,
) -> None:
    torch = require("torch")
    if not weight.is_floating_point():
        raise ConfigurationError(f"Expected floating-point weight, received {weight.dtype}")
    axis = output_axis % weight.ndim
    if weight.shape[axis] != q.shape[0]:
        raise ConfigurationError(
            f"Weight output width {weight.shape[axis]} does not match direction width {q.shape[0]}"
        )
    moved = torch.movedim(weight, axis, 0)
    remaining = tuple(moved.shape[1:])
    if not remaining:
        remaining = (1,)
        moved = moved.reshape(moved.shape[0], 1)
    prefix_ranges = [range(size) for size in remaining[:-1]]
    prefixes = itertools.product(*prefix_ranges) if prefix_ranges else [()]
    q_device = q.to(device=weight.device, dtype=torch.float32)
    with torch.no_grad():
        for prefix in prefixes:
            width = remaining[-1]
            for start in range(0, width, chunk_columns):
                index = (slice(None), *prefix, slice(start, min(start + chunk_columns, width)))
                target = moved[index]
                scratch = target.float()
                coefficients = q_device.transpose(0, 1) @ scratch
                projected = scratch - strength * (q_device @ coefficients)
                if not torch.isfinite(projected).all():
                    raise ConfigurationError("Projection produced NaN or infinity")
                target.copy_(projected.to(dtype=target.dtype))


def _project_bias_in_place(bias: Any, q: Any, strength: float) -> None:
    torch = require("torch")
    if bias is None:
        return
    if bias.ndim != 1 or bias.shape[0] != q.shape[0]:
        raise ConfigurationError(f"Bias shape {tuple(bias.shape)} does not match output direction")
    q_device = q.to(device=bias.device, dtype=torch.float32)
    with torch.no_grad():
        scratch = bias.float()
        projected = scratch - strength * (q_device @ (q_device.transpose(0, 1) @ scratch))
        bias.copy_(projected.to(dtype=bias.dtype))


def _tensor_is_finite_chunked(
    tensor: Any,
    *,
    output_axis: int,
    chunk_columns: int,
) -> bool:
    torch = require("torch")
    moved = torch.movedim(tensor.detach(), output_axis, 0)
    if moved.ndim == 1:
        moved = moved.reshape(moved.shape[0], 1)
    remaining = tuple(moved.shape[1:])
    prefix_ranges = [range(size) for size in remaining[:-1]]
    prefixes = itertools.product(*prefix_ranges) if prefix_ranges else [()]
    with torch.no_grad():
        for prefix in prefixes:
            for start in range(0, remaining[-1], chunk_columns):
                index = (
                    slice(None),
                    *prefix,
                    slice(start, min(start + chunk_columns, remaining[-1])),
                )
                if not bool(torch.isfinite(moved[index]).all()):
                    return False
    return True


def input_embedding_projection_target(adapter: ModelAdapter) -> ProjectionTarget:
    """Return the dense input-embedding matrix as a residual-output projection target."""
    model = getattr(adapter, "model", None)
    getter = getattr(model, "get_input_embeddings", None)
    if not callable(getter):
        raise ArchitectureError("The model does not expose input embeddings for projection")
    module = getter()
    weight = getattr(module, "weight", None)
    if weight is None or getattr(weight, "ndim", 0) != 2:
        raise ArchitectureError("The input embedding does not expose a two-dimensional weight")
    axes = [axis for axis, size in enumerate(weight.shape) if size == adapter.hidden_size]
    if len(axes) != 1:
        raise ArchitectureError(
            "Could not identify one residual-width axis in the input embedding weight"
        )
    checkpoint_quantization = getattr(adapter, "checkpoint_quantization", None)
    quantization, packed = describe_module_quantization(
        module,
        weight,
        checkpoint_quantization=checkpoint_quantization,
    )
    name = "input_embeddings"
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        for candidate_name, candidate in named_modules():
            if candidate is module:
                name = candidate_name or name
                break
    return ProjectionTarget(
        name=name,
        layer=-1,
        kind="embedding",
        weight=weight,
        output_axis=axes[0],
        module=module,
        packed=packed,
        quantization=quantization,
    )


class WeightAblator:
    def __init__(self, adapter: ModelAdapter) -> None:
        self.adapter = adapter

    def apply(
        self,
        directions: Any,
        *,
        layers: Iterable[int] | None = None,
        kinds: Iterable[ProjectionKind] = ("attention", "mlp", "expert"),
        include_input_embeddings: bool = False,
        strength: float = 1.0,
        chunk_columns: int = 8192,
    ) -> list[WeightAblationRecord]:
        if not 0.0 <= strength <= 1.0:
            raise ConfigurationError("Permanent weight-ablation strength must be between 0 and 1")
        if chunk_columns < 1:
            raise ConfigurationError("chunk_columns must be positive")
        requested_kinds = tuple(kinds)
        valid_kinds = {"attention", "mlp", "expert", "embedding"}
        invalid_kinds = sorted(set(requested_kinds) - valid_kinds)
        if not requested_kinds or invalid_kinds:
            raise ConfigurationError(f"Invalid projection kinds: {invalid_kinds}")
        include_input_embeddings = include_input_embeddings or "embedding" in requested_kinds
        decoder_kinds = tuple(kind for kind in requested_kinds if kind != "embedding")
        targets = (
            self.adapter.projection_targets(layers=layers, kinds=decoder_kinds)
            if decoder_kinds
            else []
        )
        if include_input_embeddings:
            targets.append(input_embedding_projection_target(self.adapter))
        if not targets:
            raise ArchitectureError("No eligible output projection matrices were found")
        assert_editable_targets(targets)
        q = orthonormalize_directions(directions)
        self._preflight(targets, q, chunk_columns=chunk_columns)
        model = getattr(self.adapter, "model", self.adapter)
        previous_state = getattr(model, "_abliterator_mutation_state", None)
        if previous_state is not None:
            raise ArchitectureError(
                "This model instance has already entered permanent mutation and must be discarded "
                "instead of edited a second time"
            )
        model.__dict__["_abliterator_mutation_state"] = "in-progress"
        records: list[WeightAblationRecord] = []
        try:
            for target in targets:
                before, weight_l2_before = direction_leakage_and_norm(
                    target.weight, q, target.output_axis, chunk_columns=chunk_columns
                )
                _project_dense_weight_in_place(
                    target.weight,
                    q,
                    output_axis=target.output_axis,
                    strength=strength,
                    chunk_columns=chunk_columns,
                )
                _project_bias_in_place(target.bias, q, strength)
                after, weight_l2_after = direction_leakage_and_norm(
                    target.weight, q, target.output_axis, chunk_columns=chunk_columns
                )
                records.append(
                    WeightAblationRecord(
                        name=target.name,
                        layer=target.layer,
                        kind=target.kind,
                        shape=target.shape,
                        output_axis=target.output_axis,
                        quantization=target.quantization,
                        leakage_before=before,
                        leakage_after=after,
                        weight_l2_before=weight_l2_before,
                        weight_l2_after=weight_l2_after,
                        strength=strength,
                    )
                )
        except Exception as exc:
            model.__dict__["_abliterator_mutation_state"] = "partial"
            raise ArchitectureError(
                "Weight projection failed after mutation began. The source checkpoint on disk is "
                "unchanged, but this in-memory model instance must be discarded."
            ) from exc
        model.__dict__["_abliterator_mutation_state"] = "complete"
        return records

    def _preflight(self, targets: list[Any], q: Any, *, chunk_columns: int) -> None:
        torch = require("torch")
        model = getattr(self.adapter, "model", self.adapter)
        parallelism = getattr(self.adapter, "declared_weight_parallelism", {})
        if parallelism:
            raise ArchitectureError(
                "Permanent editing is blocked for declared tensor/expert parallelism "
                f"{parallelism}; gather the full tensors, edit once, then reshard"
            )
        device_map = getattr(model, "hf_device_map", {}) or {}
        if any(str(device).lower() == "disk" for device in device_map.values()):
            raise ArchitectureError(
                "Permanent editing does not support Accelerate disk-offloaded weights; use an "
                "offline shard edit or a device map that materializes every target"
            )
        for target in targets:
            weight = target.weight
            if getattr(weight, "device", None) is None or weight.device.type == "meta":
                raise ArchitectureError(f"Target {target.name} is not materialized")
            editable_dtypes = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
            if weight.dtype not in editable_dtypes or weight.layout != torch.strided:
                raise ArchitectureError(
                    f"Target {target.name} is not an editable FP16/BF16/FP32/FP64 dense tensor"
                )
            axis = target.output_axis % weight.ndim
            if weight.shape[axis] != q.shape[0]:
                raise ArchitectureError(
                    f"Target {target.name} output width {weight.shape[axis]} does not match "
                    f"direction width {q.shape[0]}"
                )
            if target.bias is not None:
                if (
                    getattr(target.bias, "device", None) is None
                    or target.bias.device.type == "meta"
                    or target.bias.dtype not in editable_dtypes
                    or target.bias.layout != torch.strided
                ):
                    raise ArchitectureError(f"Target {target.name} has a non-editable bias")
                if target.bias.ndim != 1 or target.bias.shape[0] != q.shape[0]:
                    raise ArchitectureError(f"Target {target.name} has an incompatible bias shape")
            if not _tensor_is_finite_chunked(
                weight,
                output_axis=axis,
                chunk_columns=chunk_columns,
            ):
                raise ArchitectureError(f"Target {target.name} contains NaN or infinity")
            if target.bias is not None and not bool(torch.isfinite(target.bias).all()):
                raise ArchitectureError(f"Target {target.name} bias contains NaN or infinity")
