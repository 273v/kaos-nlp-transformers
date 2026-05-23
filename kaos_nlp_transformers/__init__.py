"""kaos-nlp-transformers: Dense embeddings and small-model inference for KAOS.

The sibling package to ``kaos-nlp-core``. Hybrid Rust + Python via PyO3
/ maturin; no PyTorch in any in-tree runtime path. Ships
``EmbeddingModel`` (load + embed), ``CrossEncoderReranker`` (load +
rerank), ``EmbeddingRetriever`` (cosine search; deprecated in 0.2.0,
scheduled for removal in 0.3.0), ``NliModel``, ``GLiNERExtractor``, and
``PiiDetector``.

Embedding and cross-encoder inference goes through an in-tree Rust
cdylib (``kaos_nlp_transformers._rust``) that calls libonnxruntime via
the `ort` crate (KNT-601 retired the ``fastembed`` Python wrapper in
0.2.0). Static-embedding models load through ``model2vec`` (pure-numpy
lookup, ~500x CPU speedup) when the ``[model2vec]`` extra is installed.

GPU acceleration ships as a separate ``kaos-nlp-transformers-gpu``
companion wheel built with ``cargo build --features gpu``; the ``[gpu]``
extra is a reserved alias today. KNT-501 retired the ``[torch]`` extra
in 0.1.0a6 — it remains as a no-op alias scheduled for removal in 0.3.0.

See ``docs/internal/prd/kaos-nlp-transformers.md`` and
``docs/internal/plans/kaos-nlp-transformers-v0.md``.
"""

from kaos_nlp_transformers._version import __version__
from kaos_nlp_transformers.chunking import Embedder as ChunkerEmbedder
from kaos_nlp_transformers.chunking import SemanticChunker
from kaos_nlp_transformers.device import (
    DeviceInfo,
    LatentDevice,
    SystemDevices,
    detect_devices,
)
from kaos_nlp_transformers.embedding import EmbeddingModel
from kaos_nlp_transformers.errors import (
    BackendNotInstalledError,
    DeviceNotReachableError,
    EmbeddingError,
    KaosNLPTransformersError,
    ModelLoadError,
    ModelNotRegisteredError,
)
from kaos_nlp_transformers.extraction import (
    ExtractiveRanker,
    ScoredSegment,
)
from kaos_nlp_transformers.extraction import (
    Reranker as ExtractiveReranker,
)
from kaos_nlp_transformers.models import (
    EXCLUDED,
    NER_EXCLUDED,
    NER_REGISTRY,
    NLI_EXCLUDED,
    NLI_REGISTRY,
    PII_EXCLUDED,
    PII_REGISTRY,
    REGISTRY,
    RERANKER_EXCLUDED,
    RERANKER_REGISTRY,
    RegisteredModel,
)
from kaos_nlp_transformers.ner import Entity, GLiNERExtractor
from kaos_nlp_transformers.nli import NliModel, NliScore
from kaos_nlp_transformers.pii import PiiDetector
from kaos_nlp_transformers.reranker import CrossEncoderReranker
from kaos_nlp_transformers.retrieval import EmbeddingRetriever
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

__all__ = [
    "EXCLUDED",
    "NER_EXCLUDED",
    "NER_REGISTRY",
    "NLI_EXCLUDED",
    "NLI_REGISTRY",
    "PII_EXCLUDED",
    "PII_REGISTRY",
    "REGISTRY",
    "RERANKER_EXCLUDED",
    "RERANKER_REGISTRY",
    "BackendNotInstalledError",
    "ChunkerEmbedder",
    "CrossEncoderReranker",
    "DeviceInfo",
    "DeviceNotReachableError",
    "EmbeddingError",
    "EmbeddingModel",
    "EmbeddingRetriever",
    "Entity",
    "ExtractiveRanker",
    "ExtractiveReranker",
    "GLiNERExtractor",
    "KaosNLPTransformersError",
    "KaosNLPTransformersSettings",
    "LatentDevice",
    "ModelLoadError",
    "ModelNotRegisteredError",
    "NliModel",
    "NliScore",
    "PiiDetector",
    "RegisteredModel",
    "ScoredSegment",
    "SemanticChunker",
    "SystemDevices",
    "__version__",
    "detect_devices",
]
