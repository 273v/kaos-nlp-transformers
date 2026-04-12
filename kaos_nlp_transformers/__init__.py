"""kaos-nlp-transformers: Dense embeddings and small-model inference for KAOS.

The sibling package to ``kaos-nlp-core``. Pure Python (no Rust crate).
v0 ships one class with two methods — ``EmbeddingModel.load()`` and
``EmbeddingModel.embed()`` — backed by fastembed (Apache-2.0, ONNX-only,
no torch). v1+ phases broaden the model registry, add reranking, and
expose a ``[torch]`` extra for zero-shot NLI classification.

See ``docs/internal/prd/kaos-nlp-transformers.md`` and
``docs/internal/plans/kaos-nlp-transformers-v0.md``.
"""

from kaos_nlp_transformers._version import __version__
from kaos_nlp_transformers.device import DeviceInfo, SystemDevices, detect_devices
from kaos_nlp_transformers.embedding import EmbeddingModel
from kaos_nlp_transformers.errors import (
    BackendNotInstalledError,
    EmbeddingError,
    KaosNLPTransformersError,
    ModelLoadError,
    ModelNotRegisteredError,
)
from kaos_nlp_transformers.models import EXCLUDED, REGISTRY, RegisteredModel
from kaos_nlp_transformers.reranker import CrossEncoderReranker
from kaos_nlp_transformers.retrieval import EmbeddingRetriever
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

__all__ = [
    "EXCLUDED",
    "REGISTRY",
    "BackendNotInstalledError",
    "CrossEncoderReranker",
    "DeviceInfo",
    "EmbeddingError",
    "EmbeddingModel",
    "EmbeddingRetriever",
    "KaosNLPTransformersError",
    "KaosNLPTransformersSettings",
    "ModelLoadError",
    "ModelNotRegisteredError",
    "RegisteredModel",
    "SystemDevices",
    "__version__",
    "detect_devices",
]
