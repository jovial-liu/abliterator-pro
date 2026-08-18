# AB Pro — Abliterator Pro

[![CI](https://github.com/Infinityy7/abliterator-pro/actions/workflows/ci.yml/badge.svg)](https://github.com/Infinityy7/abliterator-pro/actions/workflows/ci.yml)

A conservative library and command-line bench for finding activation directions, testing them
reversibly, and projecting selected directions out of open-weight language models.

AB Pro is the public continuation of the Abliterator Bench work. Installations expose the new
`abpro` command and `ab_pro` top-level Python API. The existing `abliterator` command and import
namespace remain available for compatibility with scripts and direction artifacts created by the
earlier bench.

The current implementation uses native Hugging Face hooks and architecture inspection so the
experiment is not tied to TransformerLens or one Llama release.

> **Scope:** “Any model” means a locally loadable PyTorch model that exposes its decoder
> activations and weights. API-only models cannot be inspected or modified. Unknown, fused, or
> packed layouts fail closed and require an adapter/override instead of guessing.

## What is implemented

- Pinned cybersecurity target and contrast datasets with checksums and license metadata.
- Length-matched, provenance-preserving prompt cohorts.
- Streaming residual activation means; full activation caches are never retained.
- Normalized target-minus-contrast direction estimation.
- Pickle-free `.safetensors` direction artifacts.
- Reversible residual-stream projection through PyTorch hooks.
- Architecture probing for dense and mixture-of-experts decoder stacks.
- A real first/last-layer forward-hook smoke test in `probe`, not just name matching.
- Chunked FP32 weight projection for attention, MLP, and packed expert matrices.
- Separate bitsandbytes FP4, bitsandbytes NF4, NVIDIA NVFP4, and MXFP4 modes.
- Dense-edit → backend-quantize export, with packed-weight mutation blocked.
- Baseline-versus-intervention false-refusal and refusal-preservation reports.
- Atomic checkpoint export that never overwrites an existing directory.
- An objective-explicit `abpro obliterate` / `abpro fork` workflow that discovers, ranks, edits,
  reload-verifies, evaluates, and publishes a new model copy in one command.

The first real-checkpoint plumbing and behavior smoke test is documented in
[`docs/QWEN3_4B_BETA.md`](docs/QWEN3_4B_BETA.md).

## Architecture coverage

| Family | Residual capture | Reversible hooks | Dense weight projection | MoE projection |
|---|---:|---:|---:|---:|
| Llama / Mistral | Yes | Yes | Yes | N/A |
| Qwen 2/2.5/3 | Yes | Yes | Yes | Yes, including packed experts |
| Qwen 3 Next | Yes | Yes | Yes, including hybrid linear attention | Model dependent |
| Gemma 1 | Yes | Yes | Yes | N/A |
| Gemma 2/3 | Yes | Yes | Approximate at raw sinks; validate post-norm | N/A |
| Phi 1/2/3 | Yes | Yes | Yes | Model dependent |
| DeepSeek V2/V3/R1 | Yes | Yes | Yes | Yes, routed and shared experts |
| Kimi K2 | Yes; DeepSeek layout | Yes | Yes | Yes |
| Kimi K2.5+ | Text prompts through its processor | Yes | Dense text backbone only | Capability-gated |
| Unknown causal LM | Structurally probed | Structurally probed | Explicit strict override required | Fail closed |

Run `abpro probe` before every new model or revision. Architecture support and
quantization support are independent capabilities.

## Installation

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/Infinityy7/abliterator-pro.git
cd abliterator-pro
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For bitsandbytes FP4 or NF4:

```bash
pip install -e '.[bnb]'
```

For FP-Quant NVFP4/MXFP4 on a supported Linux accelerator environment:

```bash
pip install -e '.[fpquant]'
```

For current Kimi K2.5 processor and compressed-tensors dependencies:

```bash
pip install -e '.[kimi]'
```

Development tools:

```bash
pip install -e '.[dev]'
pytest
```

No large model is downloaded during installation or tests.

## Workflow

### One command: create a verified fork

For the all-in-one path, point AB Pro at a dense model:

```bash
abpro obliterate --model /models/Qwen3-4B-Instruct
```

For a Base checkpoint, intentionally bypass the chat template:

```bash
abpro obliterate \
  --model /models/Qwen3-4B-Base \
  --raw-prompts
```

Use an aligned Instruct checkpoint when the experiment is meant to measure refusal behavior.
A Base checkpoint is useful for architecture and numerical plumbing, but it may not contain a
stable refusal policy; a direction learned there can capture cybersecurity topic or writing style
instead of a clean refusal feature.

Unless `--output` is supplied, a local source is written to a sibling directory named
`<source>-obliterated`; a Hub model is written to `<repo-name>-obliterated` in the current
directory. Existing paths are never overwritten or silently numbered.

The automatic workflow has two explicit objectives:

- `refusal-removal` is the default. It first generates responses from an oversized discovery pool,
  then fits only on harmful prompts the source model actually refused and benign prompts it actually
  answered. A candidate is eligible only when a reversible test measures a lower harmful-prompt
  refusal rate, and the reloaded permanent fork must reach zero measured refusals on the fixed
  harmful validation cohort before publication.
- `guarded` preserves harmful-prompt refusals while looking for lower benign false-refusal rates.
  Select it with `--objective guarded`. Candidates that reduce harmful refusal are rejected unless
  `--allow-safety-loss` is explicitly supplied; that override applies only to guarded mode.

For example:

```bash
# Default: behavior-grounded refusal removal
abpro obliterate --model /models/Qwen3-4B-Instruct

# Safety-preserving false-refusal experiment
abpro obliterate \
  --model /models/Qwen3-4B-Instruct \
  --objective guarded
```

The command performs the complete workflow:

1. validates the source, output, datasets, architecture, and dense edit support;
2. forms the discovery cohorts required by the selected objective;
3. discovers directions at multiple candidate layers across the middle of the decoder;
4. tests every candidate reversibly by installing its hook across all decoder layers, using the
   benign and harmful JBB cyber cohorts;
5. selects a direction or refusal-removal subspace only when it passes the chosen objective's
   measured criterion;
6. projects the selected direction/subspace from supported weights across all decoder layers, and
   in refusal-removal mode also projects the input embedding, then writes the result to a hidden
   staging checkpoint;
7. unloads and reloads that checkpoint, repeats a real forward-capture smoke test, matches every
   edited target, and recomputes direction leakage from the saved weights;
8. replays the identical validation cohort on the permanent fork;
9. runs a packaged truthfulness/hallucination regression smoke test, including unknowable prompts
   where the model should admit missing information; and
10. bundles the direction, ranking, post-export validation, truthfulness validation, and run receipt before atomically
   publishing the final model directory.

Inspect the resolved plan without loading or modifying a model:

```bash
abpro obliterate --model /models/Qwen3-4B-Instruct --dry-run
```

In an interactive terminal, the one-command workflow now shows ten named stages, including cohort
loading, source inspection, direction discovery, reversible ranking, permanent editing, reload
verification, regression checks, and atomic publication. Each completed stage includes elapsed
time, followed by a short summary with the saved model and receipt paths. The detailed final JSON
still goes to stdout; progress goes to stderr, so scripts can consume the result unchanged.

```bash
# Force the readable stage display, including when output is redirected.
abpro obliterate --model /models/Qwen3-4B-Instruct --progress plain

# Emit machine-readable JSON Lines progress for a desktop or web application.
abpro obliterate --model /models/Qwen3-4B-Instruct --progress json

# Preserve the earlier silent behavior.
abpro obliterate --model /models/Qwen3-4B-Instruct --no-progress
```

`--progress auto` is the default: it shows the readable view in a real terminal and remains quiet
when stderr is redirected. No progress event contains prompt or generated-response text.

Reuse an existing artifact while retaining ranking and post-export verification:

```bash
abpro obliterate \
  --model /models/Qwen3-4B-Instruct \
  --directions artifacts/qwen-cyber-directions.safetensors \
  --direction layer.18.resid_pre \
  --output checkpoints/qwen3-4b-obliterated
```

The automatic command publishes a dense safetensors checkpoint. `--quantization` controls input
loading and does not select the output format. Use `abpro quantize` afterward for an explicitly
selected FP4/NF4 backend. Packed inputs still require `--dequantize`; AB Pro never mutates packed
storage directly, and a dequantized result is an expanded approximation rather than a canonical
dense source.

Every successful fork contains:

- `abliterator_manifest.json` — the exact permanent target edits;
- `abpro/direction.safetensors` — the immutable direction snapshot used for the edit;
- `abpro/ranking.json` — reversible candidate measurements and selection;
- `abpro/post-export-validation.json` — permanent-fork replay results; and
- `abpro/truthfulness-validation.json` — factuality and unknowable-question hallucination
  regression results, when enabled; and
- `abpro/run.json` — the complete execution, environment, source, and output receipts.

Local sources receive before/after content receipts; Hub sources rely on their resolved immutable
revision. Expected fail-closed stops include an existing output path, no safety-preserving
candidate in guarded mode, no measurable refusal-reducing candidate in refusal-removal mode, a
nonzero permanent harmful-refusal rate on the fixed validation cohort, an insufficient
behavior-grounded cohort, a zero-signal baseline, an incompatible artifact, unsupported packed
storage, a saved-weight/reload mismatch, a truthfulness accuracy drop above
`--max-truthfulness-drop`, or an unknowable-question hallucination increase above
`--max-hallucination-increase`. A failed post-edit run is not published and reports its hidden
staging or diagnostic path.

Testing multiple source layers is deliberate: architectures expose the same behavior at different
depths, and the trials are reversible. Guarded mode permanently applies one selected direction.
Refusal-removal mode can project the full behavior-active candidate subspace when that is what the
saved/reloaded checkpoint needs to pass the objective.

The reported behavioral evidence is the before/after refusal-phrase rate from AB Pro's regex
detector. The word “obliterated” describes the weight projection and objective, not proof that a
model will answer every possible request. A lower regex rate cannot establish universal
non-refusal, harmful compliance, correctness, or safety, and the post-export replay validates
persistence rather than serving as an independent final holdout.

The truthfulness check is a regression tripwire for post-training discipline. It scores simple
known-answer prompts plus deliberately unknowable prompts and records only row metadata and response
hashes. It can catch a fork becoming more willing to invent, but it is not a substitute for
evidence-backed task evaluation, tool-grounded verification, or domain-specific factual review.

### Export to LM Studio and Ollama

AB Pro can package any dense local Hugging Face checkpoint—including a verified AB Pro fork—as a
portable GGUF. Both applications use the same file; the package also includes an Ollama
`Modelfile`, a short README, and a source/tool/output hash receipt.

```bash
abpro export \
  --model checkpoints/qwen2.5-7b-obliterated \
  --output checkpoints/qwen2.5-7b-obliterated-gguf \
  --name qwen2.5-7b-obliterated \
  --quantization q4_k_m \
  --llama-cpp-dir ~/Library/Caches/AB-Pro/llama.cpp
```

`Q4_K_M` is the practical default for a 7B model on a 24 GB Apple Silicon machine. Export never
edits or deletes the dense checkpoint. It converts to a hidden F16 intermediate, quantizes it,
checks the GGUF header and source stability, hashes the result, removes the intermediate, and
atomically publishes a new directory. Use `--keep-intermediate` only when the extra disk usage is
intentional. See [`docs/GGUF.md`](docs/GGUF.md).

### Expert workflow

### 1. Inspect the model

Use a small model first while validating a new environment:

```bash
abpro probe \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype bfloat16 \
  --device-map auto
```

The command reports the decoder path, residual width, output projection counts, expert layout,
whether packed weights prevent permanent editing, and the result of a real residual-capture
forward smoke test.

For Hub repositories that require custom code, remote code is disabled by default. Enabling it
requires an immutable commit:

```bash
abpro probe \
  --model organization/model \
  --trust-remote-code \
  --revision 0123456789abcdef0123456789abcdef01234567
```

Kimi K2.5 uses a processor even for text-only bench prompts. Install `.[kimi]` and select the
multimodal loader so processor assets survive dense and quantized export:

```bash
abpro probe \
  --model /models/Kimi-K2.5 \
  --model-class image-text-to-text \
  --trust-remote-code \
  --dtype auto
```

The official packed INT4 Kimi K2.5 checkpoint is suitable for capture/probing, not direct weight
projection. A persistent edit still requires a compatible dense text-backbone checkpoint.
For a Hub ID with custom code, also supply its pinned 40-character `--revision`.

### 2. Discover cybersecurity directions

The defaults contrast Meta PurpleLlama’s unsafe MITRE prompts with its benign cybersecurity
false-refusal prompts. Prompt text is not written to artifacts or logs.

```bash
abpro discover \
  --model Qwen/Qwen2.5-7B-Instruct \
  --samples 128 \
  --layers 4:24 \
  --output artifacts/qwen-cyber-directions.safetensors
```

Each key is explicit, for example `layer.14.resid_pre`. Datasets are pinned and length-matched
before capture to reduce the chance of learning prose length instead of a behavioral direction.
Chat-template reasoning is disabled by default (`enable_thinking=False` and `thinking=False`) so
short evaluation budgets are not consumed by hidden reasoning. Use `--enable-thinking` only when
that is an intentional, recorded part of the experiment.

Discovery can run against bitsandbytes FP4 weights:

```bash
abpro discover \
  --model Qwen/Qwen2.5-7B-Instruct \
  --quantization bnb-fp4 \
  --compute-dtype bfloat16 \
  --samples 128 \
  --output artifacts/qwen-fp4-directions.safetensors
```

Rank candidates with reversible validation runs rather than a handful of tokenizer-specific
"Sure"/"cannot" token IDs:

```bash
abpro rank \
  --model Qwen/Qwen2.5-7B-Instruct \
  --directions artifacts/qwen-cyber-directions.safetensors \
  --candidate-layers 8:24 \
  --apply-layers all \
  --output reports/qwen-direction-ranking.json
```

Ranking requires both the benign and harmful JBB cohorts. The objective rewards lower false
refusals while penalizing lost refusal preservation. WMDP remains a final scored MCQ holdout after
selection, not a tuning input. Candidate directions are tested at all layers by default to match
the eventual global weight-edit geometry.

### 3. Test a direction reversibly

Evaluate benign and harmful validation cohorts separately:

```bash
abpro evaluate \
  --model Qwen/Qwen2.5-7B-Instruct \
  --directions artifacts/qwen-cyber-directions.safetensors \
  --direction layer.14.resid_pre \
  --layers source \
  --dataset jbb-cyber-benign \
  --dataset jbb-cyber-harmful \
  --samples 10 \
  --output reports/layer-14.json
```

The model is unchanged when the hook context exits. Reports contain prompt/response hashes and
aggregate metrics by default, not sensitive text.

Direction artifacts are tied to the model revision or local checkpoint hash, decoder path,
tokenizer, capture point, and prompt-rendering policy. A mismatch fails closed unless
`--allow-artifact-mismatch` is explicitly supplied; any override is recorded in the report.

### 4. Apply to dense weights

```bash
abpro apply \
  --model Qwen/Qwen2.5-7B-Instruct \
  --directions artifacts/qwen-cyber-directions.safetensors \
  --direction layer.14.resid_pre \
  --layers 8:24 \
  --strength 1.0 \
  --output checkpoints/qwen-cyber-dense
```

The end of a layer range is exclusive. Existing output directories are never overwritten.

### 5. Export FP4

The reliable workflow edits canonical dense weights and then quantizes the edited checkpoint:

```bash
abpro quantize \
  --source checkpoints/qwen-cyber-dense \
  --format bnb-fp4 \
  --compute-dtype bfloat16 \
  --output checkpoints/qwen-cyber-bnb-fp4
```

Or request both outputs during `apply` with `--output-quantization bnb-fp4` and
`--quantized-output PATH`.

`bnb-fp4`, `bnb-nf4`, `nvfp4`, and `mxfp4` are deliberately separate choices. They use
different codebooks, scale layouts, kernels, and checkpoint formats. See
[`docs/QUANTIZATION.md`](docs/QUANTIZATION.md).

## Cybersecurity datasets

List registered sources and licenses:

```bash
abpro datasets
abpro datasets --json
```

Direction fitting, ranking, and final evaluation roles are separated. WMDP is a scored final MCQ
holdout. `truthfulness-smoke` is a packaged scored regression source for factuality and
unknowable-question hallucination checks. IFEval and WildGuard remain pinned prompt-only sources because this release does not
implement their official constraint/annotation scorers; the CLI will not mislabel them as scored
evaluations. See
[`docs/DATASETS.md`](docs/DATASETS.md).

## Python API

```python
from ab_pro import (
    ActivationCollector,
    CaptureConfig,
    HuggingFaceAdapter,
    ModelLoadConfig,
    estimate_directions,
)

adapter = HuggingFaceAdapter.from_pretrained(ModelLoadConfig(model="Qwen/Qwen2.5-0.5B-Instruct"))
collector = ActivationCollector(adapter)
target = collector.collect(target_prompts, CaptureConfig(layers=(6, 7, 8)))
contrast = collector.collect(contrast_prompts, CaptureConfig(layers=(6, 7, 8)))
directions = estimate_directions(target.means, contrast.means)
```

## Important limits

- A mean-difference vector may encode topic, length, template, or tone rather than refusal.
- Removing refusal language does not establish technical correctness.
- Global projection can damage unrelated capabilities and safety boundaries.
- Quantization can reintroduce part of a removed direction; reload and evaluate the exported
  checkpoint.
- Declared tensor/expert-parallel models are blocked for permanent editing; use a verified offline
  gather/edit/reshard workflow.
- Multimodal Kimi variants require the text backbone and processor to be identified correctly.
- Gemma 2/3 normalize branch outputs; raw weight orthogonality is not sufficient evidence of
  post-normalization behavior.

“Bulletproof” here means conservative validation, content-receipted artifacts, safe failure, and
tests across the supported Python/Transformers range.
It does not mean every future architecture or quantizer can be modified without an adapter.

## Responsible operation

Use only models, datasets, and systems you are authorized to test. Keep an immutable base
checkpoint, evaluate false refusals and harmful compliance independently, and never publish a
modified checkpoint without its source revision, direction artifact, dataset receipt, and
before/after evaluation. See [`SECURITY.md`](SECURITY.md).

## License

MIT. The original FailSpy copyright and license are preserved. See [`NOTICE.md`](NOTICE.md) for
upstream attribution. Datasets retain their own licenses and attribution requirements.
