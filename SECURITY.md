# AB Pro security and responsible-use notes

This repository changes model behavior. Operate it only on checkpoints and evaluation targets you
are authorized to use.

## Trust boundaries

- Hugging Face custom model code is disabled by default. If enabled, an immutable commit revision
  is mandatory because remote model code executes locally.
- Direction artifacts use `safetensors`; arbitrary pickle loading is not supported.
- Official raw dataset files are pinned and checksum-verified.
- The CLI refuses to overwrite checkpoint directories.
- Direction artifacts are checked against model/local-content receipts, decoder path, tokenizer,
  capture point, and prompt-rendering policy before evaluation or mutation.
- Packed low-bit weights are never modified through raw tensor arithmetic.
- Reports exclude prompt and response text unless a caller explicitly handles it outside the CLI.

Local checkpoint provenance hashes every model shard plus relevant config, tokenizer, processor,
chat-template, and custom-code files. This is intentionally I/O-heavy for very large local models.
Outputs may not be nested inside the immutable local source. Publication is no-clobber; a failed
export retains its uniquely named partial directory for diagnosis.

## Experimental controls

Maintain an immutable base checkpoint and capture:

- model ID, revision, tokenizer revision, and architecture report;
- direction artifact checksum and cohort receipt;
- selected source direction, target layers, matrix types, and strength;
- dense and quantized before/after evaluation;
- false-refusal, harmful-refusal-preservation, general capability, and cyber capability metrics.

Do not interpret a reduction in refusal phrases as proof of correctness or usefulness. Validate
technical answers independently and keep destructive testing inside owned or explicitly authorized
environments.

## Reporting problems

Do not attach sensitive prompts, generated exploit content, credentials, or private checkpoint
URLs to public bug reports. Report issues with synthetic tensors, prompt hashes, architecture
manifests, and minimal public models whenever possible.

Use a private GitHub security advisory for vulnerabilities that should not be disclosed publicly:
<https://github.com/Infinityy7/abliterator-pro/security/advisories/new>.
