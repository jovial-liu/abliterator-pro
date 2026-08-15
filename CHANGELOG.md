# Changelog

## 0.1.0

- Published the bench as AB Pro (Abliterator Pro), with the `abliterator-pro` distribution,
  primary `abpro` command, and `ab_pro` top-level API.
- Retained the `abliterator` command, import namespace, v1 artifact formats, manifest filenames,
  and legacy experiment for backward compatibility.
- Replaced the TransformerLens-only notebook-style module with an adapter-based package.
- Added native Hugging Face residual hooks and dense/MoE output-projection discovery.
- Added safe direction artifacts, streaming activation statistics, and atomic exports.
- Added pinned cybersecurity contrast datasets and separated evaluation holdouts.
- Added explicit bitsandbytes FP4/NF4 and FP-Quant NVFP4/MXFP4 workflows.
- Preserved the original implementation under `legacy/`.
- Added fail-closed model/artifact provenance, full local content receipts, strict decoder/sink
  coverage, reversible-hook cleanup, and preflighted atomic exports.
- Added Qwen3-Next hybrid attention and Kimi K2.5 processor/model-class handling.
- Added backend postconditions and separate installation paths for bitsandbytes and FP-Quant.
- Added Python 3.10–3.12 CI, wheel/CLI smoke tests, Ruff, MyPy, and regression coverage for packed
  FP8/INT4 modules, MoE axes, hook lifecycle, and partial-mutation prevention.
