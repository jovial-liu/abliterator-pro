# Architecture and extension contract

The experiment engine operates on four logical surfaces:

1. A sequence of decoder blocks carrying `[batch, sequence, hidden]` residual tensors.
2. A tokenizer or processor capable of applying the checkpoint’s chat template.
3. Attention/MLP output matrices whose logical output dimension equals `hidden_size`.
4. A quantization backend that states whether those logical matrices are editable.

`HuggingFaceAdapter` first consults known family profiles, then uses conservative structural
discovery. A candidate decoder stack must match `num_hidden_layers` where available. Every
selected block must execute exactly once during capture, and its input/output residual width must
match the model configuration. Known/explicit layouts must expose an attention sink and an
MLP-or-expert sink in every selected layer; partial coverage raises instead of silently editing a
subset. Qwen3-Next validates `linear_attn.out_proj` versus `self_attn.o_proj` per layer type.

## Custom layout override

Unknown models can provide a JSON file:

```json
{
  "layer_path": "model.language_model.layers",
  "attention_output_patterns": ["self_attn.o_proj"],
  "mlp_output_patterns": ["mlp.down_proj", "mlp.experts.*.down_proj"],
  "hidden_size": 4096,
  "include_experts": true
}
```

Pass it with `--architecture examples/architecture_override.json`. Explicit paths fail if absent
or if their depth differs from the config. Explicit patterns are exclusive for their category,
which narrows mutation authority; overrides do not bypass coverage, shape, dtype, or quantization
checks.

## Projection orientation

For `torch.nn.Linear`, weights are `[output, input]`. For normalized direction matrix `Q`, the
bench applies:

```text
W' = W - strength × Q(QᵀW)
b' = b - strength × Q(Qᵀb)
```

GPT-style `Conv1D` stores `[input, output]`, so its output axis is 1. Packed DeepSeek/Qwen expert
parameters may be `[experts, output, input]`, also making output axis 1. Projection iterates over
the remaining axes in bounded FP32 chunks and never constructs a `hidden × hidden` identity.

## Capability gates

The architecture report exposes separate facts:

- residual capture is available;
- reversible intervention is available;
- eligible output sinks were found;
- permanent editing is blocked by packed storage.

Unknown fused kernels, DTensor/FSDP parameters, ambiguous output axes, and unsupported quantizers
must raise an error before mutation.

`probe` additionally performs a real forward/capture smoke on the first and last decoder layers.
Permanent editing is blocked for meta/disk-offloaded targets and for configs declaring tensor or
expert parallelism; those require a verified gather/edit/reshard workflow.
