"""EmbeddingRetriever -- dense retrieval via embedding similarity search.

Builds a numpy matrix of document embeddings at construction time.
Queries embed the query text, then compute cosine similarity against
all document embeddings.  For corpora up to ~50K documents, brute-force
numpy dot product is faster than FAISS overhead.

This module lives in kaos-nlp-transformers because it depends on
``EmbeddingModel`` (fastembed/numpy).  The ``Retriever`` protocol
it implements lives in kaos-nlp-core.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from kaos_nlp_core.retrieval.protocol import corpus_unit_passage_uri
from kaos_nlp_core.search import SearchHit

from kaos_nlp_transformers.embedding import EmbeddingModel


def _group_corpus_units_for_embedding(
    corpus: Any, group_by: str
) -> tuple[list[int], list[str], list[str | None], list[dict[str, Any]]]:
    """Group CorpusUnits by an attribute for embedding retrieval.

    Returns (doc_ids, texts, external_ids, metadata_list) where each
    entry corresponds to one group.
    """
    groups: OrderedDict[str, list[Any]] = OrderedDict()
    ungrouped_counter = 0
    for unit in corpus:
        value = getattr(unit, group_by)
        if value is None:
            key = f"{unit.doc_uri}#ungrouped-{ungrouped_counter}"
            ungrouped_counter += 1
        else:
            key = f"{unit.doc_uri}#{value}"
        groups.setdefault(key, []).append(unit)

    doc_ids: list[int] = []
    texts: list[str] = []
    external_ids: list[str | None] = []
    metadata_list: list[dict[str, Any]] = []
    for idx, (key, units) in enumerate(groups.items()):
        first = units[0]
        grouped_text = "\n".join(u.text for u in units)
        doc_ids.append(idx)
        texts.append(grouped_text)
        external_ids.append(key)
        metadata_list.append(
            {
                "doc_id": key,
                "doc_uri": first.doc_uri,
                "page": first.page,
                "section_ref": first.section_ref,
                "section_title": first.section_title,
            }
        )
    return doc_ids, texts, external_ids, metadata_list


class EmbeddingRetriever:
    """Dense retrieval via cosine similarity over pre-embedded documents.

    Typical usage::

        retriever = EmbeddingRetriever.from_texts(
            texts=["doc one", "doc two"],
            doc_ids=[0, 1],
        )
        results = await retriever.retrieve("query", top_k=5)

    Or from a ``DocumentCollection``::

        from kaos_nlp_core.documents import DocumentCollection
        collection = DocumentCollection.from_records(records)
        retriever = EmbeddingRetriever.from_collection(collection)
    """

    def __init__(
        self,
        *,
        embeddings: np.ndarray,
        doc_ids: list[int],
        texts: list[str],
        external_ids: list[str | None] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
        model: EmbeddingModel,
    ) -> None:
        """
        Args:
            embeddings: Pre-computed embeddings of shape ``(N, dim)``.
                Will be L2-normalized internally.
            doc_ids: Document IDs corresponding to each row of *embeddings*.
            texts: Document texts corresponding to each row.
            external_ids: Optional external IDs for each document.
            metadata_list: Optional metadata dicts for each document.
            model: The ``EmbeddingModel`` used to produce *embeddings*
                (also used to embed queries at retrieval time).
        """
        if embeddings.ndim != 2:
            msg = f"embeddings must be 2-D, got shape {embeddings.shape}"
            raise ValueError(msg)
        if embeddings.shape[0] != len(doc_ids):
            msg = (
                f"embeddings rows ({embeddings.shape[0]}) must match "
                f"doc_ids length ({len(doc_ids)})"
            )
            raise ValueError(msg)
        if embeddings.shape[0] != len(texts):
            msg = f"embeddings rows ({embeddings.shape[0]}) must match texts length ({len(texts)})"
            raise ValueError(msg)

        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero for all-zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = (embeddings / norms).astype(np.float32)

        self._doc_ids = list(doc_ids)
        self._texts = list(texts)
        self._external_ids = list(external_ids) if external_ids else [None] * len(doc_ids)
        self._metadata_list = list(metadata_list) if metadata_list else [{} for _ in doc_ids]
        self._model = model

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return len(self._doc_ids)

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._embeddings.shape[1]

    async def retrieve(self, query: str, top_k: int = 10, **kwargs: Any) -> list[SearchHit]:
        """Embed *query* and return the *top_k* most similar documents.

        Similarity is cosine similarity computed as a dot product over
        L2-normalized vectors.
        """
        q_vec = self._model.embed([query])[0]
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        sims = self._embeddings @ q_vec
        # Get top-k indices; clip to number of docs
        effective_k = min(top_k, len(self._doc_ids))
        if effective_k <= 0:
            return []

        # argpartition is faster than full argsort for large N
        if effective_k < len(self._doc_ids):
            top_indices = np.argpartition(-sims, effective_k)[:effective_k]
            # Sort the top-k by score descending
            top_indices = top_indices[np.argsort(-sims[top_indices])]
        else:
            top_indices = np.argsort(-sims)

        hits: list[SearchHit] = []
        for idx in top_indices:
            hits.append(
                SearchHit(
                    doc_id=self._doc_ids[idx],
                    score=float(sims[idx]),
                    text=self._texts[idx],
                    external_id=self._external_ids[idx],
                    metadata=dict(self._metadata_list[idx]),
                )
            )
        return hits

    def add_documents(
        self,
        texts: list[str],
        doc_ids: list[int],
        *,
        external_ids: list[str | None] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
        batch_size: int = 32,
    ) -> None:
        """Embed and add new documents to the index.

        Args:
            texts: Document texts to embed and add.
            doc_ids: Document IDs for each text.
            external_ids: Optional external IDs.
            metadata_list: Optional metadata dicts.
            batch_size: Batch size for embedding inference.
        """
        if not texts:
            return

        new_vecs = self._model.embed(texts, batch_size=batch_size)
        norms = np.linalg.norm(new_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        new_vecs = (new_vecs / norms).astype(np.float32)

        self._embeddings = np.vstack([self._embeddings, new_vecs])
        self._doc_ids.extend(doc_ids)
        self._texts.extend(texts)

        if external_ids:
            self._external_ids.extend(external_ids)
        else:
            self._external_ids.extend([None] * len(texts))

        if metadata_list:
            self._metadata_list.extend(metadata_list)
        else:
            self._metadata_list.extend([{} for _ in texts])

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        doc_ids: list[int] | None = None,
        *,
        external_ids: list[str | None] | None = None,
        metadata_list: list[dict[str, Any]] | None = None,
        model_id: str | None = None,
        batch_size: int = 32,
    ) -> EmbeddingRetriever:
        """Build a retriever by embedding a list of texts.

        Args:
            texts: Document texts to embed.
            doc_ids: Document IDs. Defaults to ``range(len(texts))``.
            external_ids: Optional external IDs.
            metadata_list: Optional metadata dicts.
            model_id: Model to load (defaults to registry default).
            batch_size: Batch size for embedding inference.
        """
        if doc_ids is None:
            doc_ids = list(range(len(texts)))

        em = EmbeddingModel.load(model_id)
        vecs = em.embed(texts, batch_size=batch_size)
        return cls(
            embeddings=vecs,
            doc_ids=doc_ids,
            texts=texts,
            external_ids=external_ids,
            metadata_list=metadata_list,
            model=em,
        )

    @classmethod
    def from_collection(
        cls,
        collection,
        *,
        model_id: str | None = None,
        batch_size: int = 32,
    ) -> EmbeddingRetriever:
        """Build a retriever from a ``DocumentCollection``.

        Args:
            collection: A ``kaos_nlp_core.documents.DocumentCollection``.
            model_id: Model to load (defaults to registry default).
            batch_size: Batch size for embedding inference.
        """
        doc_ids: list[int] = []
        texts: list[str] = []
        external_ids: list[str | None] = []
        metadata_list: list[dict[str, Any]] = []

        for doc in collection:
            doc_ids.append(doc.doc_id)
            texts.append(doc.text)
            external_ids.append(doc.external_id)
            metadata_list.append(dict(doc.metadata))

        return cls.from_texts(
            texts=texts,
            doc_ids=doc_ids,
            external_ids=external_ids,
            metadata_list=metadata_list,
            model_id=model_id,
            batch_size=batch_size,
        )

    @classmethod
    def from_corpus(
        cls,
        corpus: Any,
        *,
        group_by: str | None = None,
        model_id: str | None = None,
        batch_size: int = 32,
    ) -> EmbeddingRetriever:
        """Build an embedding retriever from a kaos-ml-core ``Corpus``.

        Uses ``kaos_ml_core.features.embed_corpus`` for vectorization
        (which in turn uses ``EmbeddingModel`` from this package).
        Provenance is threaded from ``CorpusUnit`` fields into
        ``external_id`` and ``metadata`` on each result.

        Args:
            corpus: A ``kaos_ml_core.Corpus`` instance.
            group_by: Optional attribute name on ``CorpusUnit`` to group
                by before indexing (e.g. ``"section_ref"``).  When set,
                units sharing the same attribute value are concatenated
                into one document whose ``external_id`` is
                ``{doc_uri}#{group_value}``.  The grouped text is
                embedded (not individual units).
            model_id: Embedding model id.  Defaults to the registry
                default (``BAAI/bge-small-en-v1.5``).
            batch_size: Batch size for embedding inference.

        Example::

            from kaos_ml_core import Corpus
            corpus = Corpus.from_documents([doc1, doc2])
            retriever = EmbeddingRetriever.from_corpus(corpus)
        """
        em = EmbeddingModel.load(model_id)

        if group_by is not None:
            doc_ids, texts, external_ids, metadata_list = _group_corpus_units_for_embedding(
                corpus, group_by
            )
        else:
            # Use Corpus.embed() if available (caches by model+batch_size).
            if hasattr(corpus, "embed"):
                vecs = corpus.embed(model=model_id, batch_size=batch_size)
            else:
                from kaos_ml_core.features import embed_corpus as _embed_corpus

                vecs = _embed_corpus(corpus, model=model_id, batch_size=batch_size)

            doc_ids: list[int] = []
            texts: list[str] = []
            external_ids: list[str | None] = []
            metadata_list: list[dict[str, Any]] = []

            for unit in corpus:
                doc_ids.append(unit.row)
                texts.append(unit.text)
                passage_uri = corpus_unit_passage_uri(unit)
                external_ids.append(passage_uri)
                metadata_list.append(
                    {
                        "doc_id": passage_uri,
                        "doc_uri": unit.doc_uri,
                        "page": unit.page,
                        "section_ref": unit.section_ref,
                        "section_title": unit.section_title,
                    }
                )

            return cls(
                embeddings=vecs,
                doc_ids=doc_ids,
                texts=texts,
                external_ids=external_ids,
                metadata_list=metadata_list,
                model=em,
            )

        # Embed grouped texts
        vecs = em.embed(texts, batch_size=batch_size)
        return cls(
            embeddings=vecs,
            doc_ids=doc_ids,
            texts=texts,
            external_ids=external_ids,
            metadata_list=metadata_list,
            model=em,
        )


__all__ = ["EmbeddingRetriever"]
