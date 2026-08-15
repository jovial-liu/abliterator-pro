# Contributing to AB Pro

AB Pro changes model behavior, so contributions should fail closed when architecture,
quantization, provenance, or evaluation assumptions cannot be verified.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
mypy src/abliterator src/ab_pro
```

Do not commit model weights, direction artifacts, generated responses, private prompt corpora, or
credentials. Tests should use synthetic tensors or small public configurations and must not require
downloading a large checkpoint.

When adding an architecture or quantizer, include a fail-closed negative test in addition to its
success path. Behavioral claims require separate benign, harmful, and capability cohorts; a lower
refusal rate alone is not evidence of improved usefulness.
