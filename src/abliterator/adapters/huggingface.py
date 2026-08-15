"""Generic Hugging Face adapter with registry-first, shape-validated discovery."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable
from typing import Any

from ..config import ArchitectureSpec, ModelLoadConfig
from ..errors import ArchitectureError, ConfigurationError, DependencyError
from ..quantization import (
    build_transformers_quantization,
    checkpoint_quantization_kind,
    describe_module_quantization,
)
from ..runtime import require, tensor_device_for_inputs, torch_dtype
from .base import ArchitectureReport, ModelAdapter, ProjectionKind, ProjectionTarget
from .layouts import (
    COMMON_ATTENTION_PATTERNS,
    COMMON_LAYER_PATHS,
    COMMON_MLP_PATTERNS,
    EXPERT_PATTERNS,
    profile_for,
)


def _resolve_path(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        if not hasattr(value, part):
            raise AttributeError(path)
        value = getattr(value, part)
    return value


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(name == pattern or fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _is_layer_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    try:
        if len(value) < 1:
            return False
        first = value[0]
    except (TypeError, KeyError, IndexError):
        return False
    return callable(getattr(first, "forward", None))


def _decoder_config(config: Any) -> Any:
    getter = getattr(config, "get_text_config", None)
    if getter is not None:
        try:
            return getter(decoder=True)
        except TypeError:
            return getter()
    for name in ("text_config", "language_config"):
        value = getattr(config, name, None)
        if value is not None:
            return value
    return config


def load_pretrained_components(
    config: ModelLoadConfig,
) -> tuple[Any, Any, str, str | None, tuple[Any, ...]]:
    """Load HF model/tokenizer components without requiring a known decoder layout."""
    config.validate()
    transformers = require("transformers")
    quantization_config = build_transformers_quantization(config.quantization)
    load_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
        "local_files_only": config.local_files_only,
        "low_cpu_mem_usage": True,
    }
    if config.device_map is not None:
        load_kwargs["device_map"] = config.device_map
    if config.dtype.lower() == "auto":
        load_kwargs["torch_dtype"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch_dtype(config.dtype)
    if config.attn_implementation:
        load_kwargs["attn_implementation"] = config.attn_implementation
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    load_kwargs = {key: value for key, value in load_kwargs.items() if value is not None}

    tokenizer_id = config.tokenizer or config.model
    tokenizer_revision = config.tokenizer_revision
    if tokenizer_revision is None and tokenizer_id == config.model:
        tokenizer_revision = config.revision
    serialization_components: list[Any] = []
    tokenizer = None
    if config.model_class == "image-text-to-text":
        processor_class = getattr(transformers, "AutoProcessor", None)
        if processor_class is None:
            raise DependencyError(
                "This Transformers version has no AutoProcessor; upgrade Transformers"
            )
        processor = processor_class.from_pretrained(
            config.model,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        serialization_components.append(processor)
        if config.tokenizer is None:
            tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=tokenizer_revision,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
        )
        serialization_components.append(tokenizer)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ArchitectureError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if config.model_class == "image-text-to-text":
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
        if model_class is None:
            model_class = transformers.AutoModelForCausalLM
    else:
        model_class = transformers.AutoModelForCausalLM
    try:
        model = model_class.from_pretrained(config.model, **load_kwargs)
    except ValueError as exc:
        mapping_error = "unrecognized configuration class" in str(exc).casefold()
        if config.model_class != "image-text-to-text" or not mapping_error:
            raise
        # Some multimodal repositories (notably Kimi K2.5 remote-code checkpoints) register the
        # conditional-generation class under AutoModelForCausalLM while still requiring an
        # AutoProcessor. Preserve processor handling and use the repository's declared mapping.
        model = transformers.AutoModelForCausalLM.from_pretrained(config.model, **load_kwargs)
    model.eval()
    return model, tokenizer, tokenizer_id, tokenizer_revision, tuple(serialization_components)


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for decoder-only Transformers models, including dense and MoE layouts."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        architecture: ArchitectureSpec | None = None,
        *,
        raw_prompts: bool = False,
        enable_thinking: bool = False,
        tokenizer_source: str | None = None,
        tokenizer_requested_revision: str | None = None,
        serialization_components: tuple[Any, ...] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.architecture = architecture or ArchitectureSpec()
        self.architecture.validate()
        self.raw_prompts = raw_prompts
        self.enable_thinking = enable_thinking
        self._tokenizer_source = tokenizer_source
        self._tokenizer_requested_revision = tokenizer_requested_revision
        self.serialization_components = serialization_components or (tokenizer,)
        self.config = model.config
        self.checkpoint_quantization = checkpoint_quantization_kind(self.config)
        self.text_config = _decoder_config(self.config)
        self.model_type = str(
            getattr(self.config, "model_type", None)
            or getattr(self.text_config, "model_type", "unknown")
        )
        self._profile = profile_for(self.model_type)
        self._layer_path, self._layers = self._find_layers()
        self._hidden_size = self._find_hidden_size()

    @classmethod
    def from_pretrained(
        cls,
        config: ModelLoadConfig,
        architecture: ArchitectureSpec | None = None,
    ) -> HuggingFaceAdapter:
        model, tokenizer, tokenizer_id, tokenizer_revision, serialization_components = (
            load_pretrained_components(config)
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            architecture=architecture,
            raw_prompts=config.raw_prompts,
            enable_thinking=config.enable_thinking,
            tokenizer_source=tokenizer_id,
            tokenizer_requested_revision=tokenizer_revision,
            serialization_components=serialization_components,
        )

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def layers(self) -> list[Any]:
        return list(self._layers)

    @property
    def layer_path(self) -> str:
        return self._layer_path

    @property
    def resolved_revision(self) -> str | None:
        return getattr(self.config, "_commit_hash", None)

    @property
    def tokenizer_revision(self) -> str | None:
        init_kwargs = getattr(self.tokenizer, "init_kwargs", {}) or {}
        discovered = init_kwargs.get("_commit_hash") or init_kwargs.get("revision")
        if discovered:
            return str(discovered)
        if self._tokenizer_requested_revision:
            return self._tokenizer_requested_revision
        model_source = getattr(self.config, "_name_or_path", None)
        if self._tokenizer_source is None or self._tokenizer_source == model_source:
            return self.resolved_revision
        return None

    @property
    def declared_weight_parallelism(self) -> dict[str, int]:
        declared: dict[str, int] = {}
        for prefix, config in (("config", self.config), ("text_config", self.text_config)):
            for name in (
                "ep_size",
                "expert_parallel_size",
                "tensor_parallel_size",
                "tp_size",
            ):
                value = getattr(config, name, None)
                if isinstance(value, int) and not isinstance(value, bool) and value > 1:
                    declared[f"{prefix}.{name}"] = value
        return declared

    def _find_layers(self) -> tuple[str, Any]:
        requested = self.architecture.layer_path
        if requested is not None:
            try:
                value = _resolve_path(self.model, requested)
            except AttributeError as exc:
                raise ArchitectureError(
                    f"Configured decoder layer path {requested!r} does not exist"
                ) from exc
            if not _is_layer_sequence(value):
                raise ArchitectureError(
                    f"Configured decoder layer path {requested!r} is not a non-empty layer sequence"
                )
            expected = getattr(self.text_config, "num_hidden_layers", None)
            if isinstance(expected, int) and expected > 0 and len(value) != expected:
                raise ArchitectureError(
                    f"Config declares {expected} decoder layers, but configured path "
                    f"{requested!r} contains {len(value)}"
                )
            return requested, value

        paths: list[str] = []
        if self._profile:
            paths.extend(self._profile.layer_paths)
        paths.extend(COMMON_LAYER_PATHS)
        seen: set[str] = set()
        candidates: list[tuple[str, Any]] = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            try:
                value = _resolve_path(self.model, path)
            except AttributeError:
                continue
            if _is_layer_sequence(value):
                candidates.append((path, value))
        if not candidates:
            raise ArchitectureError(
                "Could not locate a decoder layer sequence. Run `abpro probe` and provide "
                "architecture.layer_path in the experiment configuration."
            )
        expected = getattr(self.text_config, "num_hidden_layers", None)
        exact = [candidate for candidate in candidates if expected == len(candidate[1])]
        if isinstance(expected, int) and expected > 0 and not exact:
            discovered = {path: len(layers) for path, layers in candidates}
            raise ArchitectureError(
                f"Config declares {expected} decoder layers, but discovered candidates were "
                f"{discovered}"
            )
        return (exact or candidates)[0]

    def _find_hidden_size(self) -> int:
        if self.architecture.hidden_size:
            return self.architecture.hidden_size
        for name in ("hidden_size", "d_model", "n_embd"):
            value = getattr(self.text_config, name, None)
            if isinstance(value, int) and value > 0:
                return value
        embeddings = self.model.get_input_embeddings()
        value = getattr(embeddings, "embedding_dim", None)
        if isinstance(value, int) and value > 0:
            return value
        weight = getattr(embeddings, "weight", None)
        if weight is not None and len(weight.shape) == 2:
            return int(weight.shape[1])
        raise ArchitectureError("Could not determine the model residual width")

    def _patterns(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self.architecture.attention_output_patterns:
            attention = list(self.architecture.attention_output_patterns)
        else:
            attention = list(self._profile.attention_patterns) if self._profile else []
            if self._profile:
                attention.extend(COMMON_ATTENTION_PATTERNS)

        if self.architecture.mlp_output_patterns:
            mlp = list(self.architecture.mlp_output_patterns)
        else:
            mlp = list(self._profile.mlp_patterns) if self._profile else []
            if self._profile:
                mlp.extend(COMMON_MLP_PATTERNS)
                if self.architecture.include_experts:
                    mlp.extend(EXPERT_PATTERNS)
        return tuple(dict.fromkeys(attention)), tuple(dict.fromkeys(mlp))

    def projection_targets(
        self,
        layers: Iterable[int] | None = None,
        kinds: Iterable[ProjectionKind] = ("attention", "mlp", "expert"),
        *,
        validate_coverage: bool = True,
    ) -> list[ProjectionTarget]:
        layer_values = list(range(len(self._layers))) if layers is None else list(layers)
        if not layer_values:
            raise ConfigurationError("Projection target discovery requires at least one layer")
        if len(set(layer_values)) != len(layer_values):
            raise ConfigurationError("Projection layers must not contain duplicates")
        requested_layers = set(layer_values)
        invalid = sorted(
            index for index in requested_layers if index < 0 or index >= len(self._layers)
        )
        if invalid:
            raise ArchitectureError(f"Layer indices out of range: {invalid}")
        requested_kinds = set(kinds)
        invalid_kinds = sorted(requested_kinds - {"attention", "mlp", "expert"})
        if not requested_kinds or invalid_kinds:
            raise ConfigurationError(f"Invalid projection kinds: {invalid_kinds}")
        if self._profile is None and not (
            self.architecture.attention_output_patterns or self.architecture.mlp_output_patterns
        ):
            # Structural capture is safe, but names alone are insufficient authority to mutate
            # an unknown architecture's weights.
            return []
        attention_patterns, mlp_patterns = self._patterns()
        targets: list[ProjectionTarget] = []
        seen_weights: set[int] = set()

        for layer_index in sorted(requested_layers):
            layer = self._layers[layer_index]
            for relative_name, module in layer.named_modules():
                if not relative_name:
                    continue
                is_attention = _matches(relative_name, attention_patterns)
                is_mlp = _matches(relative_name, mlp_patterns)
                if not (is_attention or is_mlp):
                    continue
                is_expert = "expert" in relative_name
                if is_expert and not self.architecture.include_experts:
                    continue
                kind: ProjectionKind = (
                    "attention" if is_attention else ("expert" if is_expert else "mlp")
                )
                if kind not in requested_kinds:
                    continue
                weight = getattr(module, "weight", None)
                for storage_name in ("qweight", "dqweight", "weight_packed", "packed_weight"):
                    if weight is None:
                        weight = getattr(module, storage_name, None)
                if weight is None or id(weight) in seen_weights:
                    continue
                quantization, packed = describe_module_quantization(
                    module,
                    weight,
                    checkpoint_quantization=self.checkpoint_quantization,
                )
                output_axis = self._module_output_axis(module, weight, packed=packed)
                targets.append(
                    ProjectionTarget(
                        name=f"{self._layer_path}.{layer_index}.{relative_name}",
                        layer=layer_index,
                        kind=kind,
                        weight=weight,
                        output_axis=output_axis,
                        bias=getattr(module, "bias", None),
                        module=module,
                        packed=packed,
                        quantization=quantization,
                    )
                )
                seen_weights.add(id(weight))

            # Some DeepSeek/Kimi implementations pack all experts in one 3-D parameter.
            for relative_name, parameter in layer.named_parameters(recurse=True):
                if (
                    not self.architecture.include_experts
                    or id(parameter) in seen_weights
                    or not _matches(relative_name, mlp_patterns)
                ):
                    continue
                if "expert" not in requested_kinds or len(parameter.shape) != 3:
                    continue
                axes = [
                    axis for axis, size in enumerate(parameter.shape) if size == self.hidden_size
                ]
                if len(axes) != 1:
                    continue
                quantization, packed = describe_module_quantization(
                    None,
                    parameter,
                    checkpoint_quantization=self.checkpoint_quantization,
                )
                targets.append(
                    ProjectionTarget(
                        name=f"{self._layer_path}.{layer_index}.{relative_name}",
                        layer=layer_index,
                        kind="expert",
                        weight=parameter,
                        output_axis=axes[0],
                        packed=packed,
                        quantization=quantization,
                    )
                )
                seen_weights.add(id(parameter))
        self._validate_qwen3_next_attention_targets(
            targets,
            requested_layers=requested_layers,
            requested_kinds=requested_kinds,
        )
        if validate_coverage:
            self._validate_target_coverage(
                targets,
                requested_layers=requested_layers,
                requested_kinds=requested_kinds,
            )
        return targets

    def _validate_target_coverage(
        self,
        targets: list[ProjectionTarget],
        *,
        requested_layers: set[int],
        requested_kinds: set[ProjectionKind],
    ) -> None:
        """Prevent a known/explicit layout from silently skipping decoder sinks."""
        by_layer: dict[int, set[ProjectionKind]] = {index: set() for index in requested_layers}
        for target in targets:
            by_layer[target.layer].add(target.kind)
        missing_attention = (
            sorted(index for index, found in by_layer.items() if "attention" not in found)
            if "attention" in requested_kinds
            else []
        )
        feed_forward_kinds = requested_kinds & {"mlp", "expert"}
        missing_feed_forward = (
            sorted(index for index, found in by_layer.items() if not found & feed_forward_kinds)
            if feed_forward_kinds
            else []
        )
        if missing_attention or missing_feed_forward:
            details = []
            if missing_attention:
                details.append(f"attention sink missing in layers {missing_attention}")
            if missing_feed_forward:
                details.append(f"MLP/expert sink missing in layers {missing_feed_forward}")
            raise ArchitectureError(
                "Projection target coverage is incomplete: " + "; ".join(details)
            )

    def _validate_qwen3_next_attention_targets(
        self,
        targets: list[ProjectionTarget],
        *,
        requested_layers: set[int],
        requested_kinds: set[ProjectionKind],
    ) -> None:
        """Validate each hybrid Qwen3-Next layer against its declared mixer type."""

        if (
            self.model_type != "qwen3_next"
            or "attention" not in requested_kinds
            or self.architecture.attention_output_patterns
        ):
            return
        layer_types = getattr(self.text_config, "layer_types", None)
        if layer_types is None:
            return
        if len(layer_types) != len(self._layers):
            raise ArchitectureError(
                "Qwen3-Next config layer_types length does not match the discovered decoder "
                "layer count"
            )
        attention_names = {
            (target.layer, target.name.rsplit(f".{target.layer}.", 1)[-1])
            for target in targets
            if target.kind == "attention"
        }
        expected_by_type = {
            "full_attention": "self_attn.o_proj",
            "linear_attention": "linear_attn.out_proj",
        }
        for layer_index in requested_layers:
            layer_type = str(layer_types[layer_index])
            expected = expected_by_type.get(layer_type)
            if expected is None:
                raise ArchitectureError(
                    f"Unsupported Qwen3-Next layer type {layer_type!r} at layer {layer_index}"
                )
            if (layer_index, expected) not in attention_names:
                raise ArchitectureError(
                    f"Qwen3-Next layer {layer_index} declares {layer_type!r}, but its expected "
                    f"output projection {expected!r} was not found"
                )

    def _module_output_axis(self, module: Any, weight: Any, *, packed: bool) -> int:
        shape = tuple(weight.shape)
        logical_out = getattr(module, "out_features", None)
        if logical_out == self.hidden_size:
            # Quantized storage may not retain the logical matrix shape; Linear still reports
            # out_features.
            if not packed and (len(shape) < 1 or shape[0] != self.hidden_size):
                raise ArchitectureError(
                    f"Dense {module.__class__.__name__} declares out_features={logical_out}, "
                    f"but its weight shape is {shape}"
                )
            return 0
        if (
            module.__class__.__name__ == "Conv1D"
            and getattr(module, "nf", None) == self.hidden_size
        ):
            if not packed and (len(shape) < 2 or shape[1] != self.hidden_size):
                raise ArchitectureError(
                    f"Dense Conv1D declares nf={self.hidden_size}, but its weight shape is {shape}"
                )
            return 1
        axes = [axis for axis, size in enumerate(shape) if size == self.hidden_size]
        if len(axes) != 1:
            raise ArchitectureError(
                f"Target {module.__class__.__name__} has shape {shape}; cannot identify a unique "
                f"output axis of size {self.hidden_size}"
            )
        return axes[0]

    def inspect(self) -> ArchitectureReport:
        targets = self.projection_targets(validate_coverage=False)
        attention = sum(target.kind == "attention" for target in targets)
        mlp = sum(target.kind == "mlp" for target in targets)
        experts = sum(target.kind == "expert" for target in targets)
        packed = sum(target.packed for target in targets)
        warnings: list[str] = []
        found_by_layer: dict[int, set[ProjectionKind]] = {
            index: set() for index in range(len(self._layers))
        }
        for target in targets:
            found_by_layer[target.layer].add(target.kind)
        missing_attention = [
            index for index, found in found_by_layer.items() if "attention" not in found
        ]
        missing_feed_forward = [
            index for index, found in found_by_layer.items() if not found & {"mlp", "expert"}
        ]
        coverage_complete = not missing_attention and not missing_feed_forward
        if missing_attention:
            warnings.append(f"Attention sink coverage is missing in layers {missing_attention}")
        if missing_feed_forward:
            warnings.append(f"MLP/expert sink coverage is missing in layers {missing_feed_forward}")
        if attention == 0:
            warnings.append("No attention output projection was found")
        if mlp == 0 and experts == 0:
            warnings.append("No MLP or expert output projection was found")
        if packed:
            warnings.append(
                f"{packed} target(s) use packed/quantized storage; direct permanent editing "
                "is disabled"
            )
        configured_layers = getattr(self.text_config, "num_hidden_layers", None)
        if configured_layers and configured_layers != len(self._layers):
            warnings.append(
                f"Config declares {configured_layers} layers but {len(self._layers)} were "
                "discovered"
            )
        if self._profile is None and not (
            self.architecture.attention_output_patterns or self.architecture.mlp_output_patterns
        ):
            warnings.append(
                "Unknown architecture: residual capture is available, but permanent weight "
                "discovery requires explicit projection patterns"
            )
        if self.model_type in {"gemma2", "gemma3", "gemma3_text"}:
            warnings.append(
                "Gemma 2/3 branch normalization makes raw sink projection approximate; validate "
                "the post-normalization residual behavior"
            )
        if self.declared_weight_parallelism:
            warnings.append(
                "The configuration declares tensor/expert parallelism "
                f"({self.declared_weight_parallelism}); permanent local editing is blocked "
                "without a gather/edit/reshard workflow"
            )
        return ArchitectureReport(
            model_type=self.model_type,
            checkpoint_revision=self.resolved_revision,
            tokenizer_revision=self.tokenizer_revision,
            architectures=list(getattr(self.config, "architectures", None) or []),
            layer_path=self._layer_path,
            layer_count=len(self._layers),
            hidden_size=self.hidden_size,
            attention_targets=attention,
            mlp_targets=mlp,
            expert_targets=experts,
            packed_targets=packed,
            capture_supported=True,
            weight_edit_supported=(
                bool(targets)
                and coverage_complete
                and not packed
                and not self.declared_weight_parallelism
            ),
            warnings=warnings,
            target_examples=[target.name for target in targets[:8]],
        )

    def _render_prompts(self, prompts: list[str]) -> tuple[list[str], bool]:
        if self.raw_prompts:
            return list(prompts), False
        rendered: list[str] = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                    thinking=self.enable_thinking,
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise ArchitectureError(
                    "The tokenizer could not render its chat template. Pass --raw-prompts only "
                    "for a base model whose evaluation contract intentionally uses raw text."
                ) from exc
            rendered.append(text)
        return rendered, True

    def format_prompts(self, prompts: list[str]) -> list[str]:
        rendered, _ = self._render_prompts(prompts)
        return rendered

    def tokenize(self, prompts: list[str], max_length: int) -> dict[str, Any]:
        rendered, used_chat_template = self._render_prompts(prompts)
        encoded = self.tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=not used_chat_template,
            return_tensors="pt",
        )
        device = tensor_device_for_inputs(self.model)
        return {name: tensor.to(device) for name, tensor in encoded.items()}

    def generate(self, prompts: list[str], max_new_tokens: int, max_length: int) -> list[str]:
        torch = require("torch")
        context_limit = None
        for name in ("max_position_embeddings", "n_positions", "seq_length"):
            value = getattr(self.text_config, name, None)
            if isinstance(value, int) and value > 0:
                context_limit = value
                break
        effective_max_length = max_length
        if context_limit is not None:
            if max_new_tokens >= context_limit:
                raise ArchitectureError(
                    f"max_new_tokens={max_new_tokens} leaves no prompt space in the model's "
                    f"{context_limit}-token context"
                )
            effective_max_length = min(max_length, context_limit - max_new_tokens)
        encoded = self.tokenize(prompts, max_length=effective_max_length)
        input_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                num_return_sequences=1,
                return_dict_in_generate=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        sequences = getattr(generated, "sequences", generated)
        completions = sequences[:, input_length:]
        return list(self.tokenizer.batch_decode(completions, skip_special_tokens=True))
