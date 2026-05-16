"""PiiDetector — closed-label PII detection via BERT token classifier.

Public surface: ``PiiDetector`` whose ``.detect(texts)`` returns a
list of :class:`Entity` lists — one ``list[Entity]`` per input text,
with byte-offset char-aligned spans against the original text.

The default model is ``onnx-community/bert-small-pii-detection-ONNX``
(28M params, ~27 MB int8 ONNX, Apache-2.0). It recognizes 24 PII
categories with B-/I- BIO encoding; the Python wrapper hands back
clean post-BIO labels like ``"PERSON"`` / ``"EMAIL_ADDRESS"`` /
``"US_SSN"`` etc.

How this complements :class:`~kaos_nlp_transformers.GLiNERExtractor`:

* **GLiNER** is zero-shot — slow, but you can throw any custom label
  set at it. Use when the entity types are domain-specific or
  open-ended ("warranty clause", "indemnification party").
* **PiiDetector** is closed-label — faster (roughly 10x per doc on
  comparable inputs), but the categories are fixed at training time.
  Use for standard PII redaction / compliance workflows where the
  24-category vocabulary covers your need.

The output ``Entity`` shape is shared with GLiNER's, so downstream
redaction pipelines / ``kaos_llm_core.programs.ner.GLiNERExtract``
can consume both extractors interchangeably.

Example::

    from kaos_nlp_transformers.pii import PiiDetector

    detector = PiiDetector.load()
    [spans] = detector.detect(
        ["Contact Jennifer Stacey at jen@galera.com or +1-555-0142."]
    )
    for e in spans:
        print(f"  [{e.score:.2f}] {e.label:<15} {e.text!r}")
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from kaos_core.logging import get_logger

from kaos_nlp_transformers.device import DeviceInfo, resolve_device
from kaos_nlp_transformers.embedding import _offline_env_scope
from kaos_nlp_transformers.errors import (
    BackendNotInstalledError,
    ModelLoadError,
    ModelNotRegisteredError,
)
from kaos_nlp_transformers.models import PII_EXCLUDED, PII_REGISTRY, RegisteredModel
from kaos_nlp_transformers.ner import Entity
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

logger = get_logger(__name__)


# Default PII model — Apache-2.0 chain, 28M params, CPU-friendly
# int8 (~27 MB). Pinned revision lives in ``PII_REGISTRY``. The
# settings field is the single source of truth so a single env-var
# override (``KAOS_NLP_TRANSFORMERS_DEFAULT_PII_MODEL``) updates
# every internal call site.
DEFAULT_PII_MODEL: str = KaosNLPTransformersSettings.model_fields["default_pii_model"].default


class PiiDetector:
    """Closed-label PII detector backed by a BERT token classifier.

    Output spans use the shared :class:`Entity` dataclass so the
    redaction / Program-Protocol code paths see GLiNER and PII
    output as the same type.
    """

    def __init__(
        self,
        _backend: Any,
        *,
        model_id: str = DEFAULT_PII_MODEL,
        device: DeviceInfo | None = None,
        labels: list[str] | None = None,
    ) -> None:
        self._backend = _backend
        self._model_id = model_id
        self._device = device
        self._labels = list(labels) if labels is not None else []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> DeviceInfo | None:
        return self._device

    @property
    def labels(self) -> list[str]:
        """The PII category vocabulary baked into the loaded model
        (post-BIO strip, e.g. ``["AGE", "EMAIL_ADDRESS", "PERSON",
        ...]``)."""
        return list(self._labels)

    @classmethod
    def load(
        cls,
        model_id: str | None = None,
        *,
        device: str | None = None,
        settings: KaosNLPTransformersSettings | None = None,
    ) -> PiiDetector:
        """Load a registered PII detector.

        Args:
            model_id: HuggingFace model id. Defaults to
                ``onnx-community/bert-small-pii-detection-ONNX``.
            device: Device override (``'auto'`` / ``'cpu'`` /
                ``'cuda'``). GPU acceleration requires the ``[gpu]``
                companion wheel.
            settings: Optional settings override.

        Raises:
            ModelNotRegisteredError: model is excluded by license
                policy or not in ``PII_REGISTRY`` and
                ``settings.allow_unregistered`` is false.
            BackendNotInstalledError: Rust cdylib not built.
            ModelLoadError: model failed to load.
        """
        s = settings if settings is not None else KaosNLPTransformersSettings()
        target = model_id or DEFAULT_PII_MODEL

        if target in PII_EXCLUDED:
            reason = PII_EXCLUDED[target]
            msg = (
                f"PII model {target!r} is excluded from the registry: "
                f"{reason}. Fix: pick a permissively-licensed alternative "
                "from kaos_nlp_transformers.models.PII_REGISTRY. "
                "Alternative: if you have a commercial license arrangement, "
                "set KAOS_NLP_TRANSFORMERS_ALLOW_UNREGISTERED=true (use with care)."
            )
            raise ModelNotRegisteredError(msg)

        if target not in PII_REGISTRY:
            if not s.allow_unregistered:
                available = ", ".join(sorted(PII_REGISTRY.keys()))
                msg = (
                    f"PII model {target!r} is not in the v0 registry. "
                    f"Fix: choose one of [{available}]. "
                    "Alternative: set KAOS_NLP_TRANSFORMERS_ALLOW_UNREGISTERED=true "
                    "to bypass the registry (you are responsible for license review)."
                )
                raise ModelNotRegisteredError(msg)
            registered = RegisteredModel(
                model_id=target,
                revision="main",
                license="UNKNOWN",
                params_m=0,
                dim=0,
                backend="ort",
                notes="unregistered PII model",
            )
        else:
            registered = PII_REGISTRY[target]

        req_device = device or s.device
        device_info = resolve_device(req_device)
        cache_dir = str(s.cache_dir) if s.cache_dir else None

        with _offline_env_scope(s.offline):
            backend = _load_pii_cached(
                model_id=registered.model_id,
                revision=registered.revision,
                device=device_info.device,
                cache_dir=cache_dir,
            )
        labels = list(backend.labels)
        logger.info(
            "Loaded PII detector %s @ %s on %s (%s) via ort (Rust); %d categories",
            registered.model_id,
            registered.revision,
            device_info.device,
            device_info.name,
            len(labels),
        )
        return cls(
            backend,
            model_id=registered.model_id,
            device=device_info,
            labels=labels,
        )

    def detect(
        self,
        texts: Sequence[str],
        *,
        score_threshold: float = 0.5,
    ) -> list[list[Entity]]:
        """Run PII detection over a batch of texts.

        Args:
            texts: Input text sequences.
            score_threshold: Minimum softmax confidence (min-across-
                span — conservative) to accept a detected span.
                Default 0.5.

        Returns:
            One ``list[Entity]`` per input text. Each ``Entity`` has
            char-offset ``start`` / ``end``, the substring ``text``,
            the category ``label`` (post-BIO strip, e.g. ``"PERSON"``
            / ``"EMAIL_ADDRESS"``), and a softmax confidence ``score``.
        """
        if not texts:
            return []
        if not (0.0 <= score_threshold <= 1.0):
            raise ValueError(f"score_threshold must be in [0, 1], got {score_threshold}")

        raw = self._backend.classify(
            list(texts),
            score_threshold=score_threshold,
        )

        out: list[list[Entity]] = []
        for per_text in raw:
            entities = [
                Entity(
                    start=int(d["start"]),
                    end=int(d["end"]),
                    text=str(d["text"]),
                    label=str(d["label"]),
                    score=float(d["score"]),
                )
                for d in per_text
            ]
            out.append(entities)
        return out


@lru_cache(maxsize=4)
def _load_pii_cached(
    model_id: str,
    revision: str,
    device: str,
    cache_dir: str | None = None,
):
    """Process-wide cache of loaded Rust ``TokenClassifierBackend``
    instances. Keyed by ``(model_id, revision, device, cache_dir)``
    so a registry SHA bump invalidates the cached backend.
    """
    try:
        from kaos_nlp_transformers import _rust

        TokenClassifierBackend = _rust.token_classify.TokenClassifierBackend
    except ImportError as exc:
        msg = (
            "kaos_nlp_transformers._rust extension is not built. "
            "Fix: run `uv run maturin develop --release` to compile the "
            "Rust cdylib for editable installs, or reinstall the package "
            "from a released wheel."
        )
        raise BackendNotInstalledError(msg) from exc

    try:
        _ = DeviceInfo
        return TokenClassifierBackend.load(model_id, device=device, cache_dir=cache_dir)
    except Exception as exc:
        if isinstance(exc, BackendNotInstalledError | ModelLoadError | ModelNotRegisteredError):
            raise
        msg = (
            f"Failed to load PII model {model_id!r} @ {revision} on "
            f"device {device!r}: {exc}. "
            f"Fix: try device='cpu' or model='{DEFAULT_PII_MODEL}'. "
            "Alternative: verify network access on first download "
            "(KAOS_NLP_TRANSFORMERS_OFFLINE)."
        )
        raise ModelLoadError(msg) from exc


__all__ = ["DEFAULT_PII_MODEL", "PiiDetector"]
