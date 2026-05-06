"""Semantic dedup level — implements ``kaos_content.dedup.types.DedupLevel``.

Embeds documents with a small fastembed model
(``BAAI/bge-small-en-v1.5`` by default) and clusters them with scipy
hierarchical agglomerative clustering on cosine distance. Catches
paraphrases, template variants, and topic clusters that lexical levels
miss.

Lives in kaos-nlp-transformers because it requires running an
embedding model at inference time. kaos-content owns the
``DedupLevel`` Protocol and the lightweight levels (binary hash, text
hash, MinHash, perceptual). Plugin shape — same as the BM25/`[nlp]`
extra: kaos-content defines the contract, kaos-nlp-transformers
registers the implementation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import ClassVar

from kaos_content.dedup.types import DedupCluster, DedupDocument, DedupLevel
from kaos_core.logging import get_logger

logger = get_logger(__name__)


class SemanticDedupLevel(DedupLevel):
    """Embedding + cosine + agglomerative clustering."""

    name: ClassVar[str] = "semantic"

    def __init__(
        self,
        *,
        model_id: str = "BAAI/bge-small-en-v1.5",
        distance_threshold: float = 0.10,
        batch_size: int = 64,
        max_chars: int = 8000,
        device: str | None = None,
    ) -> None:
        """
        Args:
            model_id: Embedding model identifier. Must be registered
                in ``kaos_nlp_transformers.models.REGISTRY``.
            distance_threshold: Cosine distance (1 - similarity)
                threshold for ``scipy.cluster.hierarchy.fcluster``.
                0.02 = near-exact semantic match (>0.98 cosine sim).
                0.10 = same template/form (~0.90 cosine sim).
                0.20 = same topic (~0.80 cosine sim).
            batch_size: Embedding batch size.
            max_chars: Truncate documents longer than this before
                embedding. The model context window is the hard limit;
                this avoids wasting memory on very long docs that
                won't fit anyway.
            device: Forwarded to ``EmbeddingModel.load(device=...)``.
                ``None`` defers to the package settings (default
                ``"auto"``). Pin to ``"cpu"`` to force fastembed even
                on GPU hosts.
        """
        self._model_id = model_id
        self._distance_threshold = distance_threshold
        self._batch_size = batch_size
        self._max_chars = max_chars
        self._device = device

    def find_clusters(
        self,
        documents: list[DedupDocument],
    ) -> list[DedupCluster]:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        from kaos_nlp_transformers.embedding import EmbeddingModel

        valid: list[tuple[int, DedupDocument]] = []
        texts: list[str] = []
        for i, doc in enumerate(documents):
            if doc.text and doc.text.strip():
                valid.append((i, doc))
                texts.append(doc.text[: self._max_chars])

        if len(valid) < 2:
            return []

        model = EmbeddingModel.load(self._model_id, device=self._device)
        embeddings = model.embed(texts, batch_size=self._batch_size)

        dists = pdist(embeddings, metric="cosine")
        linkage_matrix = linkage(dists, method="average")
        labels = fcluster(linkage_matrix, t=self._distance_threshold, criterion="distance")

        groups: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            groups[int(label)].append(idx)

        clusters: list[DedupCluster] = []
        for label, members in groups.items():
            if len(members) < 2:
                continue
            member_docs = [valid[m][1] for m in members]
            clusters.append(
                DedupCluster(
                    cluster_id=f"semantic_{label}_{member_docs[0].doc_id}",
                    canonical_doc_id=member_docs[0].doc_id,
                    member_doc_ids=tuple(d.doc_id for d in member_docs),
                    level=self.name,
                )
            )
        return clusters


__all__ = ["SemanticDedupLevel"]
