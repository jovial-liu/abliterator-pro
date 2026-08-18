"""Command-line interface for reproducible ablation experiments."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import uuid
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters.huggingface import HuggingFaceAdapter
from .artifacts import (
    load_direction_artifact,
    validate_direction_artifact_compatibility,
)
from .capture import ActivationCollector
from .config import ArchitectureSpec, CaptureConfig, ModelLoadConfig, QuantizationConfig
from .datasets import (
    get_dataset_spec,
    length_match,
    load_dataset_records,
    registry,
)
from .errors import AbliteratorError, ConfigurationError, DatasetError
from .evaluation import compare_reports, evaluate_prompts
from .gguf import (
    GGUF_QUANTIZATIONS,
    GgufExportConfig,
    GgufExportEvent,
    export_gguf_package,
)
from .interventions import ResidualProjector
from .pipeline import (
    cohort_receipt,
    discover_direction_artifact,
    environment_receipt,
    save_ablated_checkpoint,
    select_direction,
    write_direction_artifact,
)
from .progress import JsonProgressRenderer, PlainProgressRenderer
from .quantization import dequantize_model_for_edit, quantize_saved_checkpoint
from .runtime import require
from .workflow import (
    ObliterateConfig,
    default_obliterated_output,
    obliteration_plan,
    run_obliteration_workflow,
)

QUANTIZATION_CHOICES = ("none", "bnb-fp4", "bnb-nf4", "nvfp4", "mxfp4")


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _write_json(path: str, value: Any) -> None:
    destination = Path(path).resolve()
    if destination.exists():
        raise ConfigurationError(f"Refusing to overwrite existing report {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex[:12]}"
    payload = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
    try:
        with partial.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(partial, destination)
    except Exception as exc:
        raise ConfigurationError(
            f"Could not publish report without overwriting; partial data is at {partial}"
        ) from exc
    with suppress(OSError):
        partial.unlink()


def _ensure_new_path(path: str | None, label: str) -> None:
    if path is not None and Path(path).resolve().exists():
        raise ConfigurationError(f"Refusing to overwrite existing {label} {Path(path).resolve()}")


def _validate_generation_arguments(args: argparse.Namespace) -> None:
    for name in ("samples", "batch_size", "max_new_tokens", "max_length"):
        value = getattr(args, name, None)
        minimum = 8 if name == "max_length" else 1
        if value is not None and value < minimum:
            raise ConfigurationError(f"--{name.replace('_', '-')} must be at least {minimum}")
    strength = getattr(args, "strength", 1.0)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ConfigurationError("--strength must be finite and between 0 and 1")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_layers(value: str | None) -> list[int] | None:
    if value is None or value == "all":
        return None
    output: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if ":" in part:
                start_text, stop_text = part.split(":", 1)
                start, stop = int(start_text), int(stop_text)
                if stop <= start:
                    raise ConfigurationError(f"Invalid layer range {part!r}; stop is exclusive")
                output.update(range(start, stop))
            else:
                output.add(int(part))
        except ValueError as exc:
            raise ConfigurationError(f"Invalid layer selection {part!r}") from exc
    if not output or min(output) < 0:
        raise ConfigurationError("Layer selection must contain non-negative indices")
    return sorted(output)


def _source_layer(direction_key: str) -> int:
    parts = direction_key.split(".")
    if len(parts) >= 2 and parts[0] == "layer":
        try:
            return int(parts[1])
        except ValueError:
            pass
    raise ConfigurationError(
        f"Cannot infer a source layer from direction key {direction_key!r}; pass --layers"
    )


def _architecture(path: str | None) -> ArchitectureSpec:
    if path is None:
        return ArchitectureSpec()
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not read architecture JSON {path}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError("Architecture JSON must contain an object")
    return ArchitectureSpec.from_dict(value)


def _quantization(args: argparse.Namespace, prefix: str = "") -> QuantizationConfig:
    def attr(name: str, default: Any) -> Any:
        return getattr(args, f"{prefix}{name}", default)

    return QuantizationConfig(
        kind=attr("quantization", "none"),
        compute_dtype=attr("compute_dtype", "bfloat16"),
        double_quant=attr("double_quant", False),
        keep_master_weights=attr("keep_master_weights", False),
        pseudoquantization=attr("pseudoquantization", False),
    )


def _model_config(
    args: argparse.Namespace, quantization: QuantizationConfig | None = None
) -> ModelLoadConfig:
    device_map: str | None = args.device_map
    if isinstance(device_map, str) and device_map.lower() == "none":
        device_map = None
    return ModelLoadConfig(
        model=args.model,
        revision=args.revision,
        dtype=args.dtype,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        quantization=quantization or _quantization(args),
        tokenizer=args.tokenizer,
        tokenizer_revision=args.tokenizer_revision,
        model_class=args.model_class,
        local_files_only=args.local_files_only,
        raw_prompts=args.raw_prompts,
        enable_thinking=args.enable_thinking,
    )


def _load_adapter(
    args: argparse.Namespace, quantization: QuantizationConfig | None = None
) -> HuggingFaceAdapter:
    return HuggingFaceAdapter.from_pretrained(
        _model_config(args, quantization=quantization),
        architecture=_architecture(args.architecture),
    )


def command_datasets(args: argparse.Namespace) -> int:
    specs = registry()
    rows = [asdict(spec) for spec in specs.values()]
    if args.json:
        _json_print(rows)
    else:
        for spec in specs.values():
            roles = []
            if spec.use_for_direction:
                roles.append("direction")
            if spec.use_for_eval:
                roles.append("evaluation")
            if spec.use_for_ranking:
                roles.append("ranking")
            print(f"{spec.id:26} {','.join(roles):16} {spec.license_spdx:14} {spec.description}")
    return 0


def command_probe(args: argparse.Namespace) -> int:
    adapter = _load_adapter(args)
    report = adapter.inspect().to_dict()
    smoke_layers = tuple(sorted({0, len(adapter.layers) - 1}))
    smoke = ActivationCollector(adapter).collect(
        ["Reply with the single word OK."],
        CaptureConfig(layers=smoke_layers, batch_size=1, max_length=64),
    )
    report["forward_capture_smoke"] = {
        "passed": True,
        "layers": list(smoke_layers),
        "captured": sorted(smoke.means),
        "input_receipt": smoke.input_receipt,
    }
    _json_print(report)
    return 0


def command_discover(args: argparse.Namespace) -> int:
    _validate_generation_arguments(args)
    if args.length_bin < 16:
        raise ConfigurationError("--length-bin must be at least 16")
    selected_layers = _parse_layers(args.layers)
    _ensure_new_path(args.output, "direction artifact")
    target_spec = get_dataset_spec(args.target_dataset)
    contrast_spec = get_dataset_spec(args.contrast_dataset)
    if not target_spec.use_for_direction or not contrast_spec.use_for_direction:
        raise DatasetError(
            "Direction discovery only accepts registry sources marked use_for_direction=true"
        )
    load_limit = args.samples * 3 if not args.no_length_match else args.samples
    target = load_dataset_records(args.target_dataset, limit=load_limit, seed=args.seed)
    contrast = load_dataset_records(args.contrast_dataset, limit=load_limit, seed=args.seed)
    if not args.no_length_match:
        target, contrast = length_match(target, contrast, seed=args.seed, bin_width=args.length_bin)
    count = min(args.samples, len(target), len(contrast))
    if count < 2:
        raise DatasetError("Fewer than two matched records are available per cohort")
    target, contrast = target[:count], contrast[:count]
    adapter = _load_adapter(args)
    capture = CaptureConfig(
        point=args.point,
        pooling=args.pooling,
        last_k=args.last_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        layers=None if selected_layers is None else tuple(selected_layers),
    )
    artifact = discover_direction_artifact(
        adapter,
        target,
        contrast,
        capture,
        metadata={
            "model": args.model,
            "revision": args.revision,
            "quantization": asdict(_quantization(args)),
            "dtype": args.dtype,
            "device_map": args.device_map,
            "attn_implementation": args.attn_implementation,
            "model_class": args.model_class,
            "seed": args.seed,
            "length_matched": not args.no_length_match,
        },
    )
    _json_print(write_direction_artifact(args.output, artifact))
    return 0


def _evaluation_records(args: argparse.Namespace, *, for_ranking: bool = False) -> list[Any]:
    records = []
    dataset_ids = args.dataset or ["jbb-cyber-benign", "jbb-cyber-harmful"]
    for dataset_id in dataset_ids:
        spec = get_dataset_spec(dataset_id)
        if not spec.use_for_eval:
            raise DatasetError(f"{dataset_id} is not registered as an evaluation dataset")
        if for_ranking and not spec.use_for_ranking:
            raise DatasetError(
                f"{dataset_id} is a final holdout or prompt-only source and cannot tune/rank "
                "directions"
            )
        records.extend(load_dataset_records(dataset_id, limit=args.samples, seed=args.seed))
    return records


def command_rank(args: argparse.Namespace) -> int:
    _validate_generation_arguments(args)
    if not math.isfinite(args.safety_penalty) or args.safety_penalty < 0:
        raise ConfigurationError("Ranking penalties must be finite and non-negative")
    if args.max_candidates is not None and args.max_candidates < 1:
        raise ConfigurationError("--max-candidates must be positive")
    candidate_layer_filter = _parse_layers(args.candidate_layers)
    shared_apply_layers = (
        None if args.apply_layers == "source" else _parse_layers(args.apply_layers)
    )
    _ensure_new_path(args.output, "ranking report")
    artifact = load_direction_artifact(args.directions)
    candidates: list[tuple[str, Any]] = []
    for key, direction in sorted(
        artifact.directions.items(), key=lambda item: (_source_layer(item[0]), item[0])
    ):
        source_layer = _source_layer(key)
        if candidate_layer_filter is not None and source_layer not in candidate_layer_filter:
            continue
        candidates.append((key, direction))
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise ConfigurationError("No candidate directions matched the requested layers")

    records = _evaluation_records(args, for_ranking=True)
    if not any(record.expected_behavior == "allow" for record in records) or not any(
        record.expected_behavior == "deny" for record in records
    ):
        raise ConfigurationError(
            "Ranking requires both benign allow and harmful deny cohorts; capability MCQ is "
            "optional"
        )
    adapter = _load_adapter(args)
    architecture = adapter.inspect()
    provenance_warnings = validate_direction_artifact_compatibility(
        artifact,
        architecture,
        model=args.model,
        requested_revision=args.revision,
        tokenizer=args.tokenizer,
        requested_tokenizer_revision=args.tokenizer_revision,
        direction_keys=[key for key, _ in candidates],
        intervention_point=args.point,
        raw_prompts=args.raw_prompts,
        enable_thinking=args.enable_thinking,
        allow_mismatch=args.allow_artifact_mismatch,
    )
    baseline = evaluate_prompts(
        adapter,
        records,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )
    if baseline.false_refusal_rate is None or baseline.deny_refusal_rate is None:
        raise ConfigurationError(
            "Ranking requires both benign allow and harmful deny cohorts; capability MCQ is "
            "optional"
        )

    ranked: list[dict[str, Any]] = []
    for key, direction in candidates:
        layers = [_source_layer(key)] if args.apply_layers == "source" else shared_apply_layers
        with ResidualProjector(
            adapter,
            direction,
            layers=layers,
            point=args.point,
            strength=args.strength,
        ):
            modified = evaluate_prompts(
                adapter,
                records,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
            )
        false_refusal_gain = (
            0.0
            if baseline.false_refusal_rate is None or modified.false_refusal_rate is None
            else baseline.false_refusal_rate - modified.false_refusal_rate
        )
        safety_loss = (
            0.0
            if baseline.deny_refusal_rate is None or modified.deny_refusal_rate is None
            else max(0.0, baseline.deny_refusal_rate - modified.deny_refusal_rate)
        )
        score = false_refusal_gain - args.safety_penalty * safety_loss
        ranked.append(
            {
                "direction": key,
                "source_layer": _source_layer(key),
                "score": score,
                "false_refusal_gain": false_refusal_gain,
                "safety_loss": safety_loss,
                "modified": modified.to_dict(),
            }
        )
    ranked.sort(key=lambda row: (-row["score"], row["source_layer"]))
    report = {
        "baseline": baseline.to_dict(),
        "objective": {
            "safety_penalty": args.safety_penalty,
            "formula": "false_refusal_gain - safety_penalty*safety_loss",
        },
        "candidates": ranked,
        "provenance": {
            "artifact_sha256": _sha256_file(args.directions),
            "artifact_compatibility_warnings": provenance_warnings,
            "loaded_architecture": architecture.to_dict(),
            "evaluation_cohort": cohort_receipt(records),
            "environment": environment_receipt(),
        },
    }
    if args.output:
        _write_json(args.output, report)
    _json_print(report)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    _validate_generation_arguments(args)
    _ensure_new_path(args.output, "evaluation report")
    if (args.directions is None) != (args.direction is None):
        raise ConfigurationError("--directions and --direction must be supplied together")
    layers = None
    artifact = None
    direction = None
    if args.directions is not None and args.direction is not None:
        artifact = load_direction_artifact(args.directions)
        direction = select_direction(artifact, args.direction)
        layers = (
            [_source_layer(args.direction)]
            if args.layers == "source"
            else _parse_layers(args.layers)
        )

    records = _evaluation_records(args)
    adapter = _load_adapter(args)
    architecture = adapter.inspect()
    provenance_warnings: list[str] = []
    if artifact is not None and args.direction is not None:
        provenance_warnings = validate_direction_artifact_compatibility(
            artifact,
            architecture,
            model=args.model,
            requested_revision=args.revision,
            tokenizer=args.tokenizer,
            requested_tokenizer_revision=args.tokenizer_revision,
            direction_keys=[args.direction],
            intervention_point=args.point,
            raw_prompts=args.raw_prompts,
            enable_thinking=args.enable_thinking,
            allow_mismatch=args.allow_artifact_mismatch,
        )
    baseline = evaluate_prompts(
        adapter,
        records,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
    )
    if artifact is None:
        report: dict[str, Any] = {"baseline": baseline.to_dict(include_rows=args.include_rows)}
    else:
        with ResidualProjector(
            adapter,
            direction,
            layers=layers,
            point=args.point,
            strength=args.strength,
        ):
            modified = evaluate_prompts(
                adapter,
                records,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
            )
        report = compare_reports(baseline, modified)
        if args.include_rows:
            report["baseline"]["results"] = [asdict(row) for row in baseline.results]
            report["modified"]["results"] = [asdict(row) for row in modified.results]
    report["provenance"] = {
        "artifact_sha256": (None if args.directions is None else _sha256_file(args.directions)),
        "artifact_compatibility_warnings": provenance_warnings,
        "loaded_architecture": architecture.to_dict(),
        "evaluation_cohort": cohort_receipt(records),
        "environment": environment_receipt(),
    }
    if args.output:
        _write_json(args.output, report)
    _json_print(report)
    return 0


def command_apply(args: argparse.Namespace) -> int:
    if not math.isfinite(args.strength) or not 0.0 <= args.strength <= 1.0:
        raise ConfigurationError("--strength must be finite and between 0 and 1")
    if args.chunk_columns < 1:
        raise ConfigurationError("--chunk-columns must be positive")
    layers = (
        [_source_layer(args.direction)] if args.layers == "source" else _parse_layers(args.layers)
    )
    kinds = tuple(part.strip() for part in args.kinds.split(",") if part.strip())
    allowed_kinds = {"attention", "mlp", "expert"}
    unknown_kinds = sorted(set(kinds) - allowed_kinds)
    if not kinds or unknown_kinds:
        raise ConfigurationError(
            f"--kinds must contain only attention, mlp, expert; unknown={unknown_kinds}"
        )
    _ensure_new_path(args.output, "dense checkpoint")
    source_path = Path(args.model).expanduser()
    if source_path.exists():
        source_path = source_path.resolve()
        dense_path = Path(args.output).resolve()
        if dense_path == source_path or source_path in dense_path.parents:
            raise ConfigurationError(
                "Dense output must not be equal to or nested inside the local source checkpoint"
            )

    quantized: QuantizationConfig | None = None
    if args.output_quantization == "none":
        if args.quantized_output or any(
            (
                args.output_double_quant,
                args.output_keep_master_weights,
                args.output_pseudoquantization,
            )
        ):
            raise ConfigurationError("Quantized-output options require --output-quantization")
    else:
        if not args.quantized_output:
            raise ConfigurationError(
                "--quantized-output is required when --output-quantization is requested"
            )
        _ensure_new_path(args.quantized_output, "quantized checkpoint")
        dense_path = Path(args.output).resolve()
        quantized_path = Path(args.quantized_output).resolve()
        if (
            dense_path == quantized_path
            or dense_path in quantized_path.parents
            or quantized_path in dense_path.parents
        ):
            raise ConfigurationError(
                "Dense and quantized output paths must be separate, non-nested directories"
            )
        if source_path.exists() and (
            quantized_path == source_path.resolve()
            or source_path.resolve() in quantized_path.parents
        ):
            raise ConfigurationError(
                "Quantized output must not be equal to or nested inside the local source checkpoint"
            )
        quantized = QuantizationConfig(
            kind=args.output_quantization,
            compute_dtype=args.output_compute_dtype,
            double_quant=args.output_double_quant,
            keep_master_weights=args.output_keep_master_weights,
            pseudoquantization=args.output_pseudoquantization,
        )
        quantized.validate()

    artifact = load_direction_artifact(args.directions)
    direction = select_direction(artifact, args.direction)
    input_quantization = _quantization(args)
    input_quantization.validate()
    if input_quantization.kind != "none" and not args.dequantize:
        raise ConfigurationError(
            "Packed input weights require --dequantize, or load the dense source checkpoint"
        )
    adapter = _load_adapter(args, quantization=input_quantization)
    dequantized_model: Any | None = None
    if args.dequantize:
        dequantized_model = dequantize_model_for_edit(adapter.model)
        adapter = HuggingFaceAdapter(
            dequantized_model,
            adapter.tokenizer,
            _architecture(args.architecture),
            raw_prompts=args.raw_prompts,
            enable_thinking=args.enable_thinking,
            tokenizer_source=args.tokenizer or args.model,
            tokenizer_requested_revision=args.tokenizer_revision,
            serialization_components=adapter.serialization_components,
        )
    architecture = adapter.inspect()
    provenance_warnings = validate_direction_artifact_compatibility(
        artifact,
        architecture,
        model=args.model,
        requested_revision=args.revision,
        tokenizer=args.tokenizer,
        requested_tokenizer_revision=args.tokenizer_revision,
        direction_keys=[args.direction],
        raw_prompts=args.raw_prompts,
        enable_thinking=args.enable_thinking,
        allow_mismatch=args.allow_artifact_mismatch,
    )
    manifest = save_ablated_checkpoint(
        adapter,
        direction,
        args.output,
        source_model=args.model,
        source_revision=args.revision,
        direction_artifact_path=args.directions,
        direction_key=args.direction,
        layers=layers,
        kinds=kinds,
        strength=args.strength,
        chunk_columns=args.chunk_columns,
        provenance_warnings=provenance_warnings,
        source_was_quantized=(input_quantization.kind != "none" or args.dequantize),
        source_quantization=asdict(input_quantization),
        dequantized_before_edit=args.dequantize,
    )
    result: dict[str, Any] = {
        "dense_output": str(Path(args.output).resolve()),
        "manifest": manifest,
    }
    if quantized is not None and args.quantized_output is not None:
        del adapter, artifact, direction
        if dequantized_model is not None:
            del dequantized_model
        gc.collect()
        try:
            torch = require("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        qmanifest = quantize_saved_checkpoint(
            args.output,
            args.quantized_output,
            quantized,
            trust_remote_code=args.trust_remote_code,
            revision=None,
            model_class=args.model_class,
        )
        result["quantized_output"] = str(Path(args.quantized_output).resolve())
        result["quantization_manifest"] = asdict(qmanifest)
    _json_print(result)
    return 0


def command_obliterate(args: argparse.Namespace) -> int:
    output = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else default_obliterated_output(args.model)
    )
    candidate_layers: tuple[int, ...] | str | None
    if args.candidate_layers == "auto":
        candidate_layers = "auto"
    else:
        parsed_candidates = _parse_layers(args.candidate_layers)
        candidate_layers = None if parsed_candidates is None else tuple(parsed_candidates)
    parsed_apply_layers = _parse_layers(args.apply_layers)
    kinds = tuple(part.strip() for part in args.kinds.split(",") if part.strip())
    workflow = ObliterateConfig(
        output=output,
        objective=args.objective,
        directions=args.directions,
        direction_key=args.direction,
        target_dataset=args.target_dataset,
        contrast_dataset=args.contrast_dataset,
        direction_samples=args.direction_samples,
        ranking_datasets=tuple(args.ranking_dataset or ("jbb-cyber-benign", "jbb-cyber-harmful")),
        ranking_samples=args.ranking_samples,
        truthfulness_datasets=tuple(args.truthfulness_dataset or ("truthfulness-smoke",)),
        truthfulness_samples=0 if args.no_truthfulness_check else args.truthfulness_samples,
        max_truthfulness_drop=args.max_truthfulness_drop,
        max_hallucination_increase=args.max_hallucination_increase,
        seed=args.seed,
        length_match=not args.no_length_match,
        length_bin=args.length_bin,
        candidate_layers=candidate_layers,  # type: ignore[arg-type]
        candidate_count=args.candidate_count,
        point=args.point,
        pooling=args.pooling,
        last_k=args.last_k,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        safety_penalty=args.safety_penalty,
        allow_safety_loss=args.allow_safety_loss,
        apply_layers=None if parsed_apply_layers is None else tuple(parsed_apply_layers),
        kinds=kinds,
        strength=args.strength,
        chunk_columns=args.chunk_columns,
        dequantize=args.dequantize,
    )
    model_config = _model_config(args)
    architecture = _architecture(args.architecture)
    plan = obliteration_plan(model_config, workflow, architecture)
    progress_mode = args.progress
    if progress_mode == "auto":
        progress_mode = "plain" if sys.stderr.isatty() else "off"
    progress_renderer: PlainProgressRenderer | JsonProgressRenderer | None
    if progress_mode == "plain":
        progress_renderer = PlainProgressRenderer(sys.stderr)
    elif progress_mode == "json":
        progress_renderer = JsonProgressRenderer(sys.stderr)
    else:
        progress_renderer = None
    if progress_renderer is not None:
        progress_renderer.show_plan(plan, dry_run=args.dry_run)
    if args.dry_run:
        _json_print(plan)
        return 0
    result = run_obliteration_workflow(
        model_config,
        architecture,
        workflow,
        progress=None if progress_renderer is None else progress_renderer,
    )
    if progress_renderer is not None:
        progress_renderer.show_result(result)
    _json_print(result)
    return 0


def command_quantize(args: argparse.Namespace) -> int:
    config = QuantizationConfig(
        kind=args.format,
        compute_dtype=args.compute_dtype,
        double_quant=args.double_quant,
        keep_master_weights=args.keep_master_weights,
        pseudoquantization=args.pseudoquantization,
    )
    manifest = quantize_saved_checkpoint(
        args.source,
        args.output,
        config,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        model_class=args.model_class,
    )
    _json_print(asdict(manifest))
    return 0


def command_export_gguf(args: argparse.Namespace) -> int:
    source = Path(args.model).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else source.parent / f"{source.name}-gguf"
    )
    progress_mode = args.progress
    if progress_mode == "auto":
        progress_mode = "plain" if sys.stderr.isatty() else "off"

    def progress(event: GgufExportEvent) -> None:
        if progress_mode == "json":
            print(
                json.dumps({"type": "gguf-progress", **event.to_dict()}, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            return
        marker = {
            "started": "...",
            "completed": "done",
            "skipped": "skip",
            "failed": "FAIL",
        }[event.status]
        print(
            f"[{event.index}/{event.total}] {marker:>4}  {event.message}",
            file=sys.stderr,
            flush=True,
        )

    if progress_mode == "plain":
        print("AB Pro · GGUF application export", file=sys.stderr)
        print(f"  Source:       {source}", file=sys.stderr)
        print(f"  Output:       {output}", file=sys.stderr)
        print(f"  Quantization: {args.gguf_quantization.upper()}", file=sys.stderr)
        print("  Targets:      LM Studio + Ollama", file=sys.stderr)
        print(file=sys.stderr)
    result = export_gguf_package(
        GgufExportConfig(
            source=source,
            output=output,
            quantization=args.gguf_quantization,
            model_name=args.name,
            llama_cpp_dir=args.llama_cpp_dir,
            python_executable=args.python_executable,
            context_length=args.context_length,
            keep_intermediate=args.keep_intermediate,
            keep_partial=args.keep_partial,
        ),
        progress=None if progress_mode == "off" else progress,
    )
    if progress_mode == "plain":
        print(file=sys.stderr)
        print("AB Pro · GGUF export complete", file=sys.stderr)
        print(f"  GGUF:       {result['gguf']}", file=sys.stderr)
        print(f"  Size:       {result['bytes'] / (1024**3):.2f} GiB", file=sys.stderr)
        print(f"  Modelfile:  {result['modelfile']}", file=sys.stderr)
        print(f"  Receipt:    {result['receipt']}", file=sys.stderr)
    _json_print(result)
    return 0


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint")
    parser.add_argument("--revision", help="Immutable model commit; required with remote code")
    parser.add_argument("--tokenizer", help="Optional separate tokenizer ID")
    parser.add_argument("--tokenizer-revision", help="Revision for a separate tokenizer repo")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="Accelerate device map or 'none'")
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_CHOICES,
        default="none",
        help="Input-loading backend; output quantization is selected separately",
    )
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument("--double-quant", action="store_true")
    parser.add_argument("--keep-master-weights", action="store_true")
    parser.add_argument("--pseudoquantization", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--model-class", choices=("causal-lm", "image-text-to-text"), default="causal-lm"
    )
    parser.add_argument("--attn-implementation")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--raw-prompts",
        action="store_true",
        help="Bypass chat templates intentionally (base models only)",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Allow reasoning-mode chat templates during capture/evaluation (off by default)",
    )
    parser.add_argument("--architecture", help="JSON architecture override")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abpro",
        description="AB Pro architecture-aware activation direction and weight ablation bench",
    )
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for failures")
    commands = parser.add_subparsers(dest="command", required=True)

    datasets_parser = commands.add_parser("datasets", help="List pinned prompt sources")
    datasets_parser.add_argument("--json", action="store_true")
    datasets_parser.set_defaults(handler=command_datasets)

    probe = commands.add_parser("probe", help="Load a model and verify its architecture")
    _add_model_arguments(probe)
    probe.set_defaults(handler=command_probe)

    discover = commands.add_parser("discover", help="Estimate target-minus-contrast directions")
    _add_model_arguments(discover)
    discover.add_argument("--target-dataset", default="cyberseceval-mitre")
    discover.add_argument("--contrast-dataset", default="cyberseceval-frr")
    discover.add_argument("--samples", type=int, default=128)
    discover.add_argument("--seed", type=int, default=42)
    discover.add_argument("--length-bin", type=int, default=128)
    discover.add_argument("--no-length-match", action="store_true")
    discover.add_argument("--point", choices=("resid_pre", "resid_post"), default="resid_pre")
    discover.add_argument("--pooling", choices=("last", "last-k", "mean"), default="last")
    discover.add_argument("--last-k", type=int, default=1)
    discover.add_argument("--batch-size", type=int, default=4)
    discover.add_argument("--max-length", type=int, default=1024)
    discover.add_argument("--layers", default="all", help="all, 8:24, or 8,10,12")
    discover.add_argument("--output", required=True, help="Output .safetensors artifact")
    discover.set_defaults(handler=command_discover)

    evaluate = commands.add_parser("evaluate", help="Compare baseline and reversible intervention")
    _add_model_arguments(evaluate)
    evaluate.add_argument("--dataset", action="append", required=True)
    evaluate.add_argument("--samples", type=int, default=32)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--directions")
    evaluate.add_argument("--direction")
    evaluate.add_argument("--allow-artifact-mismatch", action="store_true")
    evaluate.add_argument("--layers", default="source", help="source, all, 8:24, or 8,10")
    evaluate.add_argument("--point", choices=("resid_pre", "resid_post"), default="resid_pre")
    evaluate.add_argument("--strength", type=float, default=1.0)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--max-new-tokens", type=int, default=64)
    evaluate.add_argument("--max-length", type=int, default=1024)
    evaluate.add_argument("--include-rows", action="store_true")
    evaluate.add_argument("--output")
    evaluate.set_defaults(handler=command_evaluate)

    rank = commands.add_parser("rank", help="Rank directions with reversible validation runs")
    _add_model_arguments(rank)
    rank.add_argument("--directions", required=True)
    rank.add_argument("--allow-artifact-mismatch", action="store_true")
    rank.add_argument("--dataset", action="append")
    rank.add_argument("--samples", type=int, default=10)
    rank.add_argument("--seed", type=int, default=42)
    rank.add_argument("--candidate-layers", default="all")
    rank.add_argument("--max-candidates", type=int)
    rank.add_argument("--apply-layers", default="all", help="source, all, 8:24, or 8,10")
    rank.add_argument("--point", choices=("resid_pre", "resid_post"), default="resid_pre")
    rank.add_argument("--strength", type=float, default=1.0)
    rank.add_argument("--safety-penalty", type=float, default=2.0)
    rank.add_argument("--batch-size", type=int, default=4)
    rank.add_argument("--max-new-tokens", type=int, default=64)
    rank.add_argument("--max-length", type=int, default=1024)
    rank.add_argument("--output")
    rank.set_defaults(handler=command_rank)

    apply_parser = commands.add_parser(
        "apply", help="Project a selected direction from dense weights"
    )
    _add_model_arguments(apply_parser)
    apply_parser.add_argument("--directions", required=True)
    apply_parser.add_argument("--direction", required=True)
    apply_parser.add_argument("--allow-artifact-mismatch", action="store_true")
    apply_parser.add_argument("--layers", default="all", help="source, all, 8:24, or 8,10")
    apply_parser.add_argument("--kinds", default="attention,mlp,expert")
    apply_parser.add_argument("--strength", type=float, default=1.0)
    apply_parser.add_argument("--chunk-columns", type=int, default=8192)
    apply_parser.add_argument("--dequantize", action="store_true")
    apply_parser.add_argument("--output", required=True, help="New dense checkpoint directory")
    apply_parser.add_argument("--output-quantization", choices=QUANTIZATION_CHOICES, default="none")
    apply_parser.add_argument("--output-compute-dtype", default="bfloat16")
    apply_parser.add_argument("--output-double-quant", action="store_true")
    apply_parser.add_argument("--output-keep-master-weights", action="store_true")
    apply_parser.add_argument("--output-pseudoquantization", action="store_true")
    apply_parser.add_argument("--quantized-output")
    apply_parser.set_defaults(handler=command_apply)

    obliterate = commands.add_parser(
        "obliterate",
        aliases=("fork",),
        help="Discover, validate, and permanently save an ablated model copy in one run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  abpro obliterate --model /models/Qwen3-4B-Instruct\n"
            "  abpro fork --model organization/model --output ./model-obliterated\n\n"
            "Defaults: refusal-removal objective, 40 behavior-confirmed discovery prompts per "
            "cohort, 10 ranking prompts per JBB cohort, "
            "three middle-third candidates, strength 1.0, all decoder layers, and "
            "attention/MLP/expert output sinks. Refusal-removal also verifies the behavior-active "
            "subspace and input embedding. The result is always a dense checkpoint."
        ),
    )
    _add_model_arguments(obliterate)
    obliterate.add_argument(
        "--objective",
        choices=("refusal-removal", "guarded"),
        default="refusal-removal",
        help=(
            "refusal-removal selects and verifies a permanent zero-refusal replay on the fixed "
            "harmful cohort; guarded preserves harmful-prompt refusals (default: refusal-removal)"
        ),
    )
    obliterate.add_argument(
        "--output",
        help=(
            "New checkpoint directory; default is a local source sibling or, for a Hub ID, "
            "the current directory"
        ),
    )
    obliterate.add_argument("--directions", help="Reuse an existing direction artifact")
    obliterate.add_argument(
        "--direction",
        help="Explicit artifact key to validate and apply (requires --directions)",
    )
    obliterate.add_argument("--target-dataset", default="cyberseceval-mitre")
    obliterate.add_argument("--contrast-dataset", default="cyberseceval-frr")
    obliterate.add_argument(
        "--direction-samples",
        type=int,
        default=40,
        help="Behavior-confirmed prompts per discovery cohort (default: 40)",
    )
    obliterate.add_argument(
        "--ranking-dataset",
        action="append",
        help="Ranking dataset ID; repeat for multiple sources",
    )
    obliterate.add_argument(
        "--ranking-samples",
        type=int,
        default=10,
        help="Prompts per ranking dataset (default: 10)",
    )
    obliterate.add_argument(
        "--truthfulness-dataset",
        action="append",
        help="Truthfulness regression dataset ID; repeat for multiple sources",
    )
    obliterate.add_argument(
        "--truthfulness-samples",
        type=int,
        default=20,
        help="Prompts per truthfulness dataset (default: 20)",
    )
    obliterate.add_argument(
        "--max-truthfulness-drop",
        type=float,
        default=0.05,
        help="Maximum allowed truthfulness accuracy drop before publication fails (default: 0.05)",
    )
    obliterate.add_argument(
        "--max-hallucination-increase",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed increase in unknowable-question hallucination rate before "
            "publication fails (default: 0.0)"
        ),
    )
    obliterate.add_argument(
        "--no-truthfulness-check",
        action="store_true",
        help="Skip the packaged truthfulness/hallucination regression gate",
    )
    obliterate.add_argument("--seed", type=int, default=42)
    obliterate.add_argument("--length-bin", type=int, default=128)
    obliterate.add_argument("--no-length-match", action="store_true")
    obliterate.add_argument(
        "--candidate-layers",
        default="auto",
        help="auto, all, 8:24, or 8,10; auto samples the middle third",
    )
    obliterate.add_argument(
        "--candidate-count",
        type=int,
        default=3,
        help="Automatic middle-third candidate count (default: 3)",
    )
    obliterate.add_argument("--point", choices=("resid_pre", "resid_post"), default="resid_pre")
    obliterate.add_argument("--pooling", choices=("last", "last-k", "mean"), default="last")
    obliterate.add_argument("--last-k", type=int, default=1)
    obliterate.add_argument("--batch-size", type=int, default=1)
    obliterate.add_argument("--max-length", type=int, default=1024)
    obliterate.add_argument("--max-new-tokens", type=int, default=64)
    obliterate.add_argument("--safety-penalty", type=float, default=2.0)
    obliterate.add_argument(
        "--allow-safety-loss",
        action="store_true",
        help=(
            "Guarded objective only: permit an uninformative baseline or measured "
            "harmful-refusal loss"
        ),
    )
    obliterate.add_argument(
        "--apply-layers",
        default="all",
        help="Permanent edit layers: all, 8:24, or 8,10 (default: all)",
    )
    obliterate.add_argument(
        "--kinds",
        default="attention,mlp,expert",
        help="Permanent output sinks (default: attention,mlp,expert)",
    )
    obliterate.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Projection strength in (0,1] (default: 1.0)",
    )
    obliterate.add_argument("--chunk-columns", type=int, default=8192)
    obliterate.add_argument(
        "--dequantize",
        action="store_true",
        help=(
            "Expand a supported packed input to an approximate dense model before editing; "
            "a canonical dense source is preferred"
        ),
    )
    obliterate.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the plan without loading or changing a model",
    )
    obliterate.add_argument(
        "--progress",
        choices=("auto", "plain", "json", "off"),
        default="auto",
        help=(
            "Progress display on stderr: auto uses plain text in a terminal, json emits "
            "JSON Lines, and off stays silent (default: auto)"
        ),
    )
    obliterate.add_argument(
        "--no-progress",
        dest="progress",
        action="store_const",
        const="off",
        help="Alias for --progress off",
    )
    obliterate.set_defaults(handler=command_obliterate)

    quantize = commands.add_parser("quantize", help="Quantize an edited dense checkpoint")
    quantize.add_argument("--source", required=True)
    quantize.add_argument("--output", required=True)
    quantize.add_argument("--format", choices=QUANTIZATION_CHOICES[1:], required=True)
    quantize.add_argument("--compute-dtype", default="bfloat16")
    quantize.add_argument("--double-quant", action="store_true")
    quantize.add_argument("--keep-master-weights", action="store_true")
    quantize.add_argument("--pseudoquantization", action="store_true")
    quantize.add_argument("--trust-remote-code", action="store_true")
    quantize.add_argument("--revision")
    quantize.add_argument(
        "--model-class", choices=("causal-lm", "image-text-to-text"), default="causal-lm"
    )
    quantize.set_defaults(handler=command_quantize)

    export = commands.add_parser(
        "export",
        aliases=("gguf",),
        help="Package a dense local checkpoint as GGUF for LM Studio and Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  abpro export --model checkpoints/my-model --quantization q4_k_m\n\n"
            "The output is a new directory containing one GGUF file, an Ollama Modelfile, "
            "a README, and a provenance receipt. LM Studio and Ollama use the same GGUF."
        ),
    )
    export.add_argument(
        "--model",
        "--source",
        dest="model",
        required=True,
        help="Local dense Hugging Face checkpoint directory",
    )
    export.add_argument(
        "--output",
        help="New package directory (default: <model>-gguf beside the source)",
    )
    export.add_argument("--name", help="Portable model/file name (default: source directory name)")
    export.add_argument(
        "--quantization",
        dest="gguf_quantization",
        type=str.lower,
        choices=GGUF_QUANTIZATIONS,
        default="q4_k_m",
        help="GGUF tensor encoding (default: q4_k_m)",
    )
    export.add_argument(
        "--llama-cpp-dir",
        help="llama.cpp checkout containing convert_hf_to_gguf.py",
    )
    export.add_argument(
        "--python-executable",
        help="Python used to run llama.cpp's converter (default: current Python)",
    )
    export.add_argument(
        "--context-length",
        type=int,
        default=8192,
        help="Ollama num_ctx value written to Modelfile (default: 8192)",
    )
    export.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep the large F16 intermediate alongside a quantized GGUF",
    )
    export.add_argument(
        "--keep-partial",
        action="store_true",
        help="Retain hidden conversion data when an export fails",
    )
    export.add_argument(
        "--progress",
        choices=("auto", "plain", "json", "off"),
        default="auto",
        help="Progress on stderr (default: auto)",
    )
    export.set_defaults(handler=command_export_gguf)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (AbliteratorError, OSError, RuntimeError, ValueError) as exc:
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
