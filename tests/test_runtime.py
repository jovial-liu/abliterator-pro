from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from abliterator.runtime import publish_directory_noreplace, tensor_device_for_inputs


def test_directory_publication_is_atomic_and_no_clobber(tmp_path) -> None:
    source = tmp_path / "partial"
    source.mkdir()
    (source / "value.txt").write_text("complete", encoding="utf-8")
    destination = tmp_path / "published"
    publish_directory_noreplace(source, destination)
    assert not source.exists()
    assert (destination / "value.txt").read_text(encoding="utf-8") == "complete"

    second = tmp_path / "second-partial"
    second.mkdir()
    with pytest.raises((FileExistsError, OSError)):
        publish_directory_noreplace(second, destination)
    assert second.exists()


def test_meta_embeddings_use_accelerate_execution_device() -> None:
    embeddings = torch.nn.Embedding(4, 2, device="meta")
    embeddings._hf_hook = SimpleNamespace(execution_device=torch.device("cpu"))
    model = SimpleNamespace(get_input_embeddings=lambda: embeddings)
    assert tensor_device_for_inputs(model) == torch.device("cpu")
