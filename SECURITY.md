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
- The one-command workflow always records an explicit objective. Its default `refusal-removal`
  objective behavior-grounds discovery by retaining harmful prompts the source model actually
  refused and benign prompts it actually answered. It publishes only if the reversible trial lowers
  harmful-prompt refusal rate and the reloaded permanent fork reaches zero measured refusals on the
  fixed harmful validation cohort.
- `--objective guarded` instead preserves harmful-prompt refusals while measuring benign false
  refusals. It rejects a candidate that reduces harmful refusal unless the operator supplies the
  guarded-only `--allow-safety-loss` override.
- Candidate directions from multiple source layers are tested reversibly across all decoder
  layers. Guarded mode projects one selected direction permanently. Refusal-removal mode may project
  the full behavior-active direction subspace, across supported decoder-layer weights and the input
  embedding. This broad intervention can affect unrelated capabilities and safety boundaries.

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

AB Pro's built-in behavioral evidence is a regex-based refusal-phrase rate. Do not interpret a
reduction in that rate as proof that the model never refuses, complies with every request, is
correct, useful, or safe. Both objectives use bounded cohorts; neither is a certification. Validate
technical answers independently and keep destructive testing inside owned or explicitly authorized
environments.

## Reporting problems

Do not attach sensitive prompts, generated exploit content, credentials, or private checkpoint
URLs to public bug reports. Report issues with synthetic tensors, prompt hashes, architecture
manifests, and minimal public models whenever possible.

Use a private GitHub security advisory for vulnerabilities that should not be disclosed publicly:
<https://github.com/Infinityy7/abliterator-pro/security/advisories/new>.
