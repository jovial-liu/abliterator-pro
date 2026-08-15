"""Known decoder layouts plus conservative structural fallbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutProfile:
    model_types: tuple[str, ...]
    layer_paths: tuple[str, ...]
    attention_patterns: tuple[str, ...]
    mlp_patterns: tuple[str, ...]


COMMON_LAYER_PATHS = (
    "model.layers",
    "model.model.layers",
    "model.language_model.layers",
    "model.language_model.model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "transformer.h",
    "model.transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
    "transformer.blocks",
    "transformer.layers",
    "backbone.layers",
    "model.text_model.layers",
)

COMMON_ATTENTION_PATTERNS = (
    "self_attn.o_proj",
    "self_attn.out_proj",
    "self_attn.dense",
    "attention.o_proj",
    "attention.out_proj",
    "attention.wo",
    "attention.dense",
    "self_attention.dense",
    "attn.o_proj",
    "attn.out_proj",
    "attn.c_proj",
)

COMMON_MLP_PATTERNS = (
    "mlp.down_proj",
    "mlp.fc2",
    "mlp.c_proj",
    "mlp.dense_4h_to_h",
    "feed_forward.w2",
    "ffn.down_proj",
    "ffn.w2",
)

EXPERT_PATTERNS = (
    "mlp.experts.*.down_proj",
    "mlp.experts.*.w2",
    "mlp.experts.down_proj",
    "mlp.shared_experts.down_proj",
    "mlp.shared_expert.down_proj",
    "block_sparse_moe.experts.*.w2",
    "block_sparse_moe.experts.*.down_proj",
    "feed_forward.experts.*.w2",
)

PROFILES = (
    LayoutProfile(
        model_types=("qwen3_next",),
        layer_paths=("model.layers",),
        attention_patterns=("self_attn.o_proj", "linear_attn.out_proj"),
        mlp_patterns=("mlp.down_proj",),
    ),
    LayoutProfile(
        model_types=(
            "llama",
            "mistral",
            "mixtral",
            "gemma",
            "gemma2",
            "gemma3_text",
            "qwen2",
            "qwen2_moe",
            "qwen3",
            "qwen3_moe",
            "deepseek_v2",
            "deepseek_v3",
            "kimi_k2",
            "kimi_k25",
        ),
        layer_paths=(
            "model.layers",
            "model.model.layers",
            "model.language_model.layers",
            "model.language_model.model.layers",
        ),
        attention_patterns=("self_attn.o_proj",),
        mlp_patterns=("mlp.down_proj",),
    ),
    LayoutProfile(
        model_types=("phi",),
        layer_paths=("model.layers",),
        attention_patterns=("self_attn.dense",),
        mlp_patterns=("mlp.fc2",),
    ),
    LayoutProfile(
        model_types=("phi3",),
        layer_paths=("model.layers",),
        attention_patterns=("self_attn.o_proj",),
        mlp_patterns=("mlp.down_proj",),
    ),
    LayoutProfile(
        model_types=("gpt2", "gpt_bigcode"),
        layer_paths=("transformer.h",),
        attention_patterns=("attn.c_proj",),
        mlp_patterns=("mlp.c_proj",),
    ),
)


def profile_for(model_type: str) -> LayoutProfile | None:
    normalized = model_type.lower()
    return next((profile for profile in PROFILES if normalized in profile.model_types), None)
