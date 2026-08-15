# Qwen3-4B Base beta receipt

AB Pro was exercised against a local dense BF16 Qwen3-4B Base checkpoint on Apple Silicon. No
checkpoint, direction artifact, prompt, or generated response is included in this repository.

## Structural and numerical result

- 4,022,468,096 parameters, 36 decoder layers, hidden width 2,560.
- All 36 attention and 36 MLP output sinks were found; no packed targets or warnings were present.
- A real forward pass captured the first and last residual streams.
- Three unit directions were estimated at layers 12, 18, and 24 from 8 unsafe and 8
  length-matched benign cybersecurity prompts; no prompt was truncated.
- A reversible layer-18 projection changed next-token logits and restored the exact baseline after
  hook removal.
- An in-memory projection of one layer-18 attention matrix reduced measured directional leakage
  from 0.01823281 to 0.00011495 (about 158.6x). No edited checkpoint was saved.

## Small behavioral check

A deterministic, short-generation check used three benign and four clearly harmful, non-graphic
prompts. The Base model refused three of four harmful prompts at baseline and two of four while all
three discovered directions were applied reversibly. Benign strict refusals were zero in both
conditions. All seven response hashes changed, but none of the tested harmful completions showed
code-block or numbered-procedure markers.

This does **not** show that model safety was removed. The change was small and non-monotonic: two
baseline refusals disappeared, one new refusal appeared, and one remained. A non-refusal is not the
same as harmful compliance. This Base checkpoint, 8+8 discovery cohort, seven-prompt behavior
check, short outputs, and regex refusal detector are suitable for plumbing validation only—not a
safety or deployment conclusion.
