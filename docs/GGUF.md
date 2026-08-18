# GGUF export for LM Studio and Ollama

`abpro export` converts a dense local Hugging Face checkpoint into one portable GGUF application
package. GGUF is an inference/export format: run AB Pro's activation discovery and permanent
weight projection against the dense checkpoint first, then export the verified fork.

## Prerequisites

AB Pro deliberately uses the official llama.cpp converter and quantizer rather than implementing
another tensor encoder. Provide a llama.cpp checkout containing `convert_hf_to_gguf.py`; for
low-bit output, `llama-quantize` must also be installed or built.

On macOS, one setup is:

```bash
brew install llama.cpp
git clone https://github.com/ggml-org/llama.cpp \
  ~/Library/Caches/AB-Pro/llama.cpp
python -m pip install -r ~/Library/Caches/AB-Pro/llama.cpp/requirements.txt
```

Pass the checkout with `--llama-cpp-dir` or set `LLAMA_CPP_DIR`. AB Pro records the exact checkout
commit, converter hash, binary paths, Python path, source checkpoint receipt, and output hash.

## Export

```bash
abpro export \
  --model /models/my-dense-fork \
  --output /models/my-dense-fork-gguf \
  --quantization q4_k_m
```

Supported encodings are `f16`, `bf16`, `q8_0`, `q6_k`, `q5_k_m`, `q5_k_s`, `q4_k_m`, and
`q4_k_s`. F16/BF16 preserve the most information but remain large. Q4_K_M is the default balance
for 7B-class local use. Quantization is lossy, so application-format smoke tests do not replace
the dense fork's behavioral and capability evaluation.

The command publishes a new directory containing:

- `<name>-<QUANT>.gguf` — shared by LM Studio and Ollama;
- `Modelfile` — relative GGUF reference plus the selected context length;
- `README.md` — exact import commands; and
- `abpro-gguf-export.json` — provenance, compatibility commands, header counts, and SHA-256.

Existing and nested output paths are rejected. Failed conversions are cleaned by default; use
`--keep-partial` only for diagnosis. The dense source is fingerprinted before and after conversion
and publication is atomic/no-clobber.

## LM Studio

Import without duplicating the GGUF on the same disk:

```bash
lms import /models/my-dense-fork-gguf/my-dense-fork-Q4_K_M.gguf \
  --hard-link --user-repo ab-pro/my-dense-fork -y
```

LM Studio also accepts a copy, symbolic link, or direct local GGUF selection. AB Pro prints and
records the import command but does not silently modify LM Studio's model library.

## Ollama

The generated `Modelfile` is immediately usable:

```bash
ollama create my-dense-fork -f /models/my-dense-fork-gguf/Modelfile
ollama run my-dense-fork
```

AB Pro creates the portable files but does not silently register a model with the Ollama service.
