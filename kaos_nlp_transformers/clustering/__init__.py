"""Document-level clustering surfaces backed by dense embeddings.

The submodule registers implementations of kaos-content's clustering
protocols (today: ``DedupLevel``) so callers who install
``kaos-nlp-transformers`` automatically get the embedding-backed
variants of those operations.
"""

from __future__ import annotations

from kaos_nlp_transformers.clustering.semantic_dedup import SemanticDedupLevel

__all__ = ["SemanticDedupLevel"]
