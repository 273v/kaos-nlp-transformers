"""Unit tests for EmbeddingRetriever -- no network / model download needed.

These tests mock the EmbeddingModel to avoid downloading weights during
CI.  Integration tests that actually load the model live in
``tests/integration/``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from kaos_nlp_core.search import SearchHit

from kaos_nlp_transformers.retrieval import EmbeddingRetriever

pytestmark = pytest.mark.unit


# ---- Helpers ----------------------------------------------------------------


def make_mock_model(dim: int = 4) -> MagicMock:
    """Create a mock EmbeddingModel that returns deterministic vectors."""
    model = MagicMock()
    model.dim = dim
    model.model_id = "mock/test-model"

    def mock_embed(texts, *, batch_size=32):
        rng = np.random.default_rng(42)
        return rng.standard_normal((len(texts), dim)).astype(np.float32)

    model.embed = mock_embed
    return model


def make_retriever(
    n_docs: int = 5,
    dim: int = 4,
) -> tuple[EmbeddingRetriever, MagicMock]:
    """Build a retriever with *n_docs* random-embedded documents."""
    model = make_mock_model(dim=dim)
    rng = np.random.default_rng(123)
    embeddings = rng.standard_normal((n_docs, dim)).astype(np.float32)
    doc_ids = list(range(n_docs))
    texts = [f"Document number {i}" for i in range(n_docs)]
    external_ids = [f"ext-{i}" for i in range(n_docs)]
    metadata_list = [{"index": i} for i in range(n_docs)]

    retriever = EmbeddingRetriever(
        embeddings=embeddings,
        doc_ids=doc_ids,
        texts=texts,
        external_ids=external_ids,
        metadata_list=metadata_list,
        model=model,
    )
    return retriever, model


# ---- Construction tests -----------------------------------------------------


class TestEmbeddingRetrieverConstruction:
    def test_basic_construction(self) -> None:
        retriever, _ = make_retriever(n_docs=3, dim=4)
        assert retriever.num_documents == 3
        assert retriever.dim == 4

    def test_embeddings_are_normalized(self) -> None:
        retriever, _ = make_retriever(n_docs=5, dim=8)
        norms = np.linalg.norm(retriever._embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_mismatched_doc_ids_raises(self) -> None:
        model = make_mock_model(dim=4)
        embeddings = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="doc_ids length"):
            EmbeddingRetriever(
                embeddings=embeddings,
                doc_ids=[0, 1],  # only 2, but 3 rows
                texts=["a", "b", "c"],
                model=model,
            )

    def test_mismatched_texts_raises(self) -> None:
        model = make_mock_model(dim=4)
        embeddings = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="texts length"):
            EmbeddingRetriever(
                embeddings=embeddings,
                doc_ids=[0, 1, 2],
                texts=["a", "b"],  # only 2, but 3 rows
                model=model,
            )

    def test_1d_embeddings_raises(self) -> None:
        model = make_mock_model(dim=4)
        embeddings = np.zeros(4, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            EmbeddingRetriever(
                embeddings=embeddings,
                doc_ids=[0],
                texts=["a"],
                model=model,
            )

    def test_zero_documents(self) -> None:
        model = make_mock_model(dim=4)
        embeddings = np.zeros((0, 4), dtype=np.float32)
        retriever = EmbeddingRetriever(
            embeddings=embeddings,
            doc_ids=[],
            texts=[],
            model=model,
        )
        assert retriever.num_documents == 0


# ---- Retrieval tests --------------------------------------------------------


class TestEmbeddingRetrieverRetrieve:
    async def test_basic_retrieve(self) -> None:
        retriever, _ = make_retriever(n_docs=5)
        results = await retriever.retrieve("test query", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchHit) for r in results)

    async def test_top_k_clips_to_num_docs(self) -> None:
        retriever, _ = make_retriever(n_docs=3)
        results = await retriever.retrieve("test", top_k=10)
        assert len(results) == 3

    async def test_results_sorted_by_score(self) -> None:
        retriever, _ = make_retriever(n_docs=10)
        results = await retriever.retrieve("test", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_scores_are_cosine_similarities(self) -> None:
        retriever, _ = make_retriever(n_docs=5)
        results = await retriever.retrieve("test", top_k=5)
        for r in results:
            # Cosine similarity is in [-1, 1]
            assert -1.0 <= r.score <= 1.0 + 1e-6

    async def test_external_id_preserved(self) -> None:
        retriever, _ = make_retriever(n_docs=3)
        results = await retriever.retrieve("test", top_k=3)
        for r in results:
            assert r.external_id is not None
            assert r.external_id.startswith("ext-")

    async def test_metadata_preserved(self) -> None:
        retriever, _ = make_retriever(n_docs=3)
        results = await retriever.retrieve("test", top_k=3)
        for r in results:
            assert "index" in r.metadata

    async def test_empty_index_returns_empty(self) -> None:
        model = make_mock_model(dim=4)
        retriever = EmbeddingRetriever(
            embeddings=np.zeros((0, 4), dtype=np.float32),
            doc_ids=[],
            texts=[],
            model=model,
        )
        results = await retriever.retrieve("test", top_k=5)
        assert results == []


# ---- add_documents tests ----------------------------------------------------


class TestEmbeddingRetrieverAddDocuments:
    def test_add_documents_grows_index(self) -> None:
        retriever, _ = make_retriever(n_docs=3, dim=4)
        assert retriever.num_documents == 3

        retriever.add_documents(
            texts=["New document one", "New document two"],
            doc_ids=[100, 101],
        )
        assert retriever.num_documents == 5

    def test_add_documents_preserves_existing(self) -> None:
        retriever, _ = make_retriever(n_docs=3, dim=4)
        original_ids = list(retriever._doc_ids)

        retriever.add_documents(texts=["new"], doc_ids=[99])
        assert retriever._doc_ids[:3] == original_ids

    def test_add_empty_is_noop(self) -> None:
        retriever, _ = make_retriever(n_docs=3, dim=4)
        retriever.add_documents(texts=[], doc_ids=[])
        assert retriever.num_documents == 3

    async def test_added_documents_are_retrievable(self) -> None:
        retriever, model = make_retriever(n_docs=2, dim=4)

        # Override the mock to return a vector close to doc 100
        call_count = [0]
        original_embed = model.embed

        def embed_with_known_vectors(texts, *, batch_size=32):
            call_count[0] += 1
            return original_embed(texts, batch_size=batch_size)

        model.embed = embed_with_known_vectors

        retriever.add_documents(
            texts=["added document"],
            doc_ids=[100],
            external_ids=["ext-100"],
            metadata_list=[{"added": True}],
        )

        results = await retriever.retrieve("test", top_k=5)
        result_ids = [r.doc_id for r in results]
        assert 100 in result_ids

    def test_add_with_metadata(self) -> None:
        retriever, _ = make_retriever(n_docs=1, dim=4)
        retriever.add_documents(
            texts=["meta doc"],
            doc_ids=[50],
            external_ids=["ext-50"],
            metadata_list=[{"key": "value"}],
        )
        assert retriever._metadata_list[-1] == {"key": "value"}
        assert retriever._external_ids[-1] == "ext-50"

    def test_add_without_optional_fields(self) -> None:
        retriever, _ = make_retriever(n_docs=1, dim=4)
        retriever.add_documents(texts=["bare doc"], doc_ids=[50])
        assert retriever._external_ids[-1] is None
        assert retriever._metadata_list[-1] == {}


# ---- Embedding normalization edge cases -------------------------------------


class TestNormalizationEdgeCases:
    def test_zero_vector_handled(self) -> None:
        """A document with an all-zero embedding should not cause NaN."""
        model = make_mock_model(dim=4)
        embeddings = np.zeros((2, 4), dtype=np.float32)
        embeddings[1] = [1, 0, 0, 0]

        retriever = EmbeddingRetriever(
            embeddings=embeddings,
            doc_ids=[0, 1],
            texts=["zero vec", "unit vec"],
            model=model,
        )
        # The zero vector should be normalized to itself (all zeros)
        assert not np.any(np.isnan(retriever._embeddings))

    async def test_zero_query_vector(self) -> None:
        """A query that embeds to all-zeros should not crash."""
        model = make_mock_model(dim=4)
        # Override to return zero vector for query
        model.embed = lambda texts, **kw: np.zeros((len(texts), 4), dtype=np.float32)

        retriever = EmbeddingRetriever(
            embeddings=np.eye(2, 4, dtype=np.float32),
            doc_ids=[0, 1],
            texts=["a", "b"],
            model=model,
        )
        results = await retriever.retrieve("zero query", top_k=2)
        assert len(results) == 2
        assert not any(np.isnan(r.score) for r in results)
