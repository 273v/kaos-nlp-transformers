"""Unit tests for KaosNLPTransformersSettings."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_default_settings():
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    s = KaosNLPTransformersSettings()
    assert s.default_model == "BAAI/bge-small-en-v1.5"
    assert s.allow_unregistered is False


def test_excluded_model_load_raises_with_reason(monkeypatch):
    from kaos_nlp_transformers import EmbeddingModel
    from kaos_nlp_transformers.errors import ModelNotRegisteredError

    with pytest.raises(ModelNotRegisteredError, match="excluded"):
        EmbeddingModel.load("jinaai/jina-embeddings-v3")


def test_unregistered_model_load_raises_when_disallowed():
    from kaos_nlp_transformers import EmbeddingModel
    from kaos_nlp_transformers.errors import ModelNotRegisteredError

    with pytest.raises(ModelNotRegisteredError, match="not in the v0 registry"):
        EmbeddingModel.load("some-org/some-random-model-not-in-registry")
