"""kaos-nlp-transformers: Dense embeddings and small-model inference for KAOS.

The sibling package to ``kaos-nlp-core``. Pure Python (no Rust crate),
no PyTorch. Ships ``EmbeddingModel`` (load + embed),
``CrossEncoderReranker`` (load + rerank), and ``EmbeddingRetriever``
(cosine search). All inference goes through fastembed (Apache-2.0, ONNX
Runtime) or model2vec (pure-numpy static lookup, ~500x CPU speedup).

GPU acceleration is opt-in via the ``[gpu]`` extra (``onnxruntime-gpu``
+ CUDA). Audit-06 KNT-501 retired the ``[torch]`` extra in 0.1.0a6 — it
remains as a no-op alias for one release cycle so existing lockfiles
keep resolving; new code should use ``[gpu]`` for CUDA acceleration.

See ``docs/internal/prd/kaos-nlp-transformers.md`` and
``docs/internal/plans/kaos-nlp-transformers-v0.md``.
"""

from kaos_nlp_transformers._version import __version__
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
from kaos_nlp_transformers.models import (
    EXCLUDED,
    REGISTRY,
    RERANKER_EXCLUDED,
    RERANKER_REGISTRY,
    RegisteredModel,
)
from kaos_nlp_transformers.reranker import CrossEncoderReranker
from kaos_nlp_transformers.retrieval import EmbeddingRetriever
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

__all__ = [
    "EXCLUDED",
    "REGISTRY",
    "RERANKER_EXCLUDED",
    "RERANKER_REGISTRY",
    "BackendNotInstalledError",
    "CrossEncoderReranker",
    "DeviceInfo",
    "DeviceNotReachableError",
    "EmbeddingError",
    "EmbeddingModel",
    "EmbeddingRetriever",
    "KaosNLPTransformersError",
    "KaosNLPTransformersSettings",
    "LatentDevice",
    "ModelLoadError",
    "ModelNotRegisteredError",
    "RegisteredModel",
    "SystemDevices",
    "__version__",
    "detect_devices",
]
