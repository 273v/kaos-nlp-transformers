"""Offline tests for ``CrossEncoderReranker.from_local_path``.

The happy path (loading a real ONNX cross-encoder from disk) needs a
model export and lives in ``tests/integration/test_reranker_live.py``.
These unit tests cover the deterministic *rejected* cases — the input
validation that must hold without any model present — so they run on
every PR.
"""

from __future__ import annotations

import pytest

from kaos_nlp_transformers import CrossEncoderReranker, ModelLoadError


def test_from_local_path_rejects_nonexistent_directory(tmp_path) -> None:
    """A path that is not a directory raises ModelLoadError (Python guard)."""
    missing = tmp_path / "does-not-exist"
    with pytest.raises(ModelLoadError):
        CrossEncoderReranker.from_local_path(missing)


def test_from_local_path_rejects_file_path(tmp_path) -> None:
    """A regular file (not a directory) raises ModelLoadError."""
    f = tmp_path / "model.bin"
    f.write_bytes(b"not a model")
    with pytest.raises(ModelLoadError):
        CrossEncoderReranker.from_local_path(f)


def test_from_local_path_rejects_directory_missing_onnx(tmp_path) -> None:
    """An existing directory without ``onnx/model.onnx`` raises ModelLoadError
    (the Rust ``load_local`` backend guard)."""
    with pytest.raises(ModelLoadError):
        CrossEncoderReranker.from_local_path(tmp_path)
