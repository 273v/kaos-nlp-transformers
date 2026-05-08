"""Pinned model registry for kaos-nlp-transformers.

Every entry must carry an explicit revision SHA — never ``main``.
Every entry must declare a permissively-licensed model. The exclusion
list captures models that look attractive but have license problems
(CC-BY-NC, training-data ambiguity, etc.) and may not be added.

The registry is the binding contract — license review happens here, at
the point where a model becomes loadable. ``EmbeddingModel.load()``
checks the registry before delegating to the backend.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """A model that has passed license review and is loadable in v0/v1."""

    model_id: str
    """Hugging Face Hub model id (org/repo)."""

    revision: str
    """Pinned commit SHA — NEVER 'main'. Min 7 chars."""

    license: str
    """SPDX-style license identifier (must be permissive)."""

    params_m: int
    """Approximate parameter count in millions."""

    dim: int
    """Embedding dimension produced by this model."""

    backend: str
    """Which backend supports this model: 'fastembed' or 'sentence-transformers'."""

    notes: str = ""
    """Free-form notes (default model? legal-doc default? etc.)."""


# Embedding registry. Three model families covered in alpha:
#
# 1. fastembed — ONNX Runtime, CPU-friendly, the default for general retrieval.
#    Quality bench: BAAI/bge-small-en-v1.5 (33M, 384-dim, MIT).
#
# 2. model2vec — static lookup (vocab → vector + average), pure numpy at
#    inference, no torch. ~500x faster on CPU than the transformer source.
#    Quality bench (MTEB Retrieval): potion-retrieval-32M = 35.06 (~82% of
#    all-MiniLM-L6-v2). Use for: first-pass retrieval over 100K+ docs,
#    high-throughput dedup/clustering. Pair with a cross-encoder reranker
#    for final-pass quality.
#
# 3. sentence-transformers — torch-backed; opt-in via the [torch] extra.
#    No registry entries here today; reranker registry covers the BGE
#    cross-encoder via the same backend.
#
# Revision SHAs are validated against huggingface.co on every CI run by
# the optional ``test_registry_shas_exist_on_hub`` test (skipped offline).
# All SHAs were re-verified against huggingface.co/api/models/<id> on
# 2026-05-08 as part of the audit-04 sweep adding the model2vec entries.
REGISTRY: dict[str, RegisteredModel] = {
    "BAAI/bge-small-en-v1.5": RegisteredModel(
        model_id="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        license="MIT",
        params_m=33,
        dim=384,
        backend="fastembed",
        notes="Default v0 embedding model. CPU-friendly, English. Verified 2026-04-09.",
    ),
    "minishlab/potion-retrieval-32M": RegisteredModel(
        model_id="minishlab/potion-retrieval-32M",
        revision="6fc8051fab2a1e0ee76689cf08c853792ac285e7",
        license="MIT",
        params_m=32,
        # Matryoshka-trained at [32, 64, 128, 256, 512]; the on-disk vectors
        # are 512-dim and consumers can truncate at retrieval time. We pin
        # the full dim and document Matryoshka in the README rather than
        # branching the registry per-cut.
        dim=512,
        backend="model2vec",
        notes=(
            "Static retrieval-tuned distillation of bge-base-en-v1.5. "
            "MTEB Retrieval 35.06 (~82% of all-MiniLM-L6-v2) at >500x CPU "
            "throughput, ~30 MB on disk. Verified 2026-05-08. Requires the "
            "[model2vec] extra."
        ),
    ),
    "minishlab/potion-base-32M": RegisteredModel(
        model_id="minishlab/potion-base-32M",
        revision="1e5a03f8eeb2c98b928fbbd846f22f816360919f",
        license="MIT",
        params_m=32,
        # potion-base is the general-purpose static distillation; same
        # 512-dim vectors as potion-retrieval but tuned for the average-
        # over-tasks MTEB score rather than retrieval specifically.
        dim=512,
        backend="model2vec",
        notes=(
            "Static general-purpose distillation of bge-base-en-v1.5. "
            "MTEB avg 51.66 (~95% of all-MiniLM-L6-v2). Use for "
            "classification / dedup / clustering; for retrieval pick "
            "potion-retrieval-32M instead. Verified 2026-05-08. Requires the "
            "[model2vec] extra."
        ),
    ),
}


# Hard exclusion list. These models are flagged by license audit and
# may not enter the registry under any circumstances. The reason
# string is shown to the user when they try to load an excluded model
# so the rejection is informative, not silent.
EXCLUDED: dict[str, str] = {
    # CC-BY-NC family — non-commercial only
    "jinaai/jina-embeddings-v3": "CC-BY-NC 4.0 (non-commercial)",
    "nvidia/NV-Embed-v1": "CC-BY-NC 4.0 (non-commercial)",
    "nvidia/NV-Embed-v2": "CC-BY-NC 4.0 (non-commercial)",
    # MS MARCO training-data ambiguity (commercial license unclear)
    "Qwen/Qwen3-Embedding-0.6B": "Trained on MS MARCO (commercial license unclear)",
    "Qwen/Qwen3-Embedding-4B": "Trained on MS MARCO (commercial license unclear)",
    "Qwen/Qwen3-Embedding-8B": "Trained on MS MARCO (commercial license unclear)",
}


# ---------------------------------------------------------------------------
# Reranker registry (audit-02 KNT-104)
# ---------------------------------------------------------------------------

# Audit-02 KNT-104: rerankers go through the same license / revision /
# offline policy as embeddings. The reranker shape mirrors RegisteredModel
# but lives in its own dict so task-specific defaults stay clear and the
# embedding registry can never accidentally be used to load a reranker
# (or vice versa).
#
# Revision SHAs verified against huggingface.co/api/models/<id> on
# 2026-05-08; confirmation procedure documented in the model expansion
# checklist.
RERANKER_REGISTRY: dict[str, RegisteredModel] = {
    "BAAI/bge-reranker-base": RegisteredModel(
        model_id="BAAI/bge-reranker-base",
        revision="2cfc18c9415c912f9d8155881c133215df768a70",
        license="MIT",
        params_m=278,
        # Cross-encoders return a single relevance score per (query, passage)
        # pair, not a vector — dim is recorded as 1 for shape symmetry with
        # RegisteredModel rather than as an embedding dimension.
        dim=1,
        backend="sentence-transformers",
        notes="Default v0 reranker. CPU-friendly cross-encoder. Verified 2026-05-08.",
    ),
}


# Same shape as EXCLUDED but for rerankers. Currently empty — re-add
# entries here as license / data-licensing concerns surface.
RERANKER_EXCLUDED: dict[str, str] = {}


__all__ = [
    "EXCLUDED",
    "REGISTRY",
    "RERANKER_EXCLUDED",
    "RERANKER_REGISTRY",
    "RegisteredModel",
]
