"""GLiNERExtractor — zero-shot NER via prompt-based span extraction.

Public surface: a ``GLiNERExtractor`` whose ``.extract(texts, labels)``
returns a list of :class:`Entity` lists — one ``list[Entity]`` per
input text, with byte-offset spans against the original text.

Backend: the in-tree Rust cdylib's ``_rust.ner.NerBackend`` (ort +
libonnxruntime). Same registry-gating pattern as ``EmbeddingModel`` /
``CrossEncoderReranker`` / ``NliModel``.

Default model: ``onnx-community/gliner_medium-v2.1`` (Apache-2.0
chain, 195M params, ~746 MiB fp32 ONNX). See ``NER_REGISTRY`` for the
full license chain and the quantization-tradeoff note.

Example::

    from kaos_nlp_transformers.ner import GLiNERExtractor

    extractor = GLiNERExtractor.load()
    [entities] = extractor.extract(
        ["Barack Obama was born in Hawaii."],
        labels=["person", "place"],
    )
    for e in entities:
        print(f"{e.text!r} -> {e.label} (score={e.score:.2f})")
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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
from kaos_nlp_transformers.models import NER_EXCLUDED, NER_REGISTRY, RegisteredModel
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

logger = get_logger(__name__)


# Default GLiNER model — Apache-2.0 chain, 195M params, CPU-friendly
# fp32. Pinned revision lives in ``NER_REGISTRY``. Derives from the
# settings field default so a single env-var override
# (``KAOS_NLP_TRANSFORMERS_DEFAULT_NER_MODEL``) updates every call
# site that does not pass an explicit ``model_id``.
DEFAULT_NER_MODEL: str = KaosNLPTransformersSettings.model_fields["default_ner_model"].default


@dataclass(frozen=True, slots=True)
class Entity:
    """A decoded named-entity span.

    Byte offsets are into the original input text — ``text[start:end]``
    yields ``text`` directly. ``score`` is the sigmoid-normalized
    probability returned by the GLiNER span head, in ``[0, 1]``.
    """

    start: int
    end: int
    text: str
    label: str
    score: float


class GLiNERExtractor:
    """Zero-shot NER via the GLiNER span-extraction model.

    Implements a synchronous ``extract`` contract that mirrors the
    upstream ``urchade/GLiNER`` Python library's
    ``predict_entities`` shape but batches over inputs and returns
    typed dataclasses instead of dicts.
    """

    def __init__(
        self,
        _backend: Any,
        *,
        model_id: str = DEFAULT_NER_MODEL,
        device: DeviceInfo | None = None,
    ) -> None:
        self._backend = _backend
        self._model_id = model_id
        self._device = device

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def device(self) -> DeviceInfo | None:
        return self._device

    @classmethod
    def load(
        cls,
        model_id: str | None = None,
        *,
        device: str | None = None,
        settings: KaosNLPTransformersSettings | None = None,
    ) -> GLiNERExtractor:
        """Load a registered GLiNER extractor.

        Args:
            model_id: HuggingFace model id. Defaults to
                ``onnx-community/gliner_medium-v2.1``.
            device: Device override (``'auto'``, ``'cpu'``, ``'cuda'``,
                etc.). GPU acceleration requires the ``[gpu]``
                companion wheel.
            settings: Optional settings override.

        Raises:
            ModelNotRegisteredError: If the model is in
                ``NER_EXCLUDED`` (license-blocked) or is not in
                ``NER_REGISTRY`` and ``settings.allow_unregistered``
                is false.
            BackendNotInstalledError: If the Rust cdylib was not built.
            ModelLoadError: If the model fails to load.
        """
        s = settings if settings is not None else KaosNLPTransformersSettings()
        target = model_id or DEFAULT_NER_MODEL

        if target in NER_EXCLUDED:
            reason = NER_EXCLUDED[target]
            msg = (
                f"NER model {target!r} is excluded from the registry: "
                f"{reason}. Fix: pick a permissively-licensed alternative "
                "from kaos_nlp_transformers.models.NER_REGISTRY. "
                "Alternative: if you have a commercial license arrangement, "
                "set KAOS_NLP_TRANSFORMERS_ALLOW_UNREGISTERED=true (use with care)."
            )
            raise ModelNotRegisteredError(msg)

        if target not in NER_REGISTRY:
            if not s.allow_unregistered:
                available = ", ".join(sorted(NER_REGISTRY.keys()))
                msg = (
                    f"NER model {target!r} is not in the v0 registry. "
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
                notes="unregistered NER model",
            )
        else:
            registered = NER_REGISTRY[target]

        req_device = device or s.device
        device_info = resolve_device(req_device)
        cache_dir = str(s.cache_dir) if s.cache_dir else None

        with _offline_env_scope(s.offline):
            backend = _load_ner_cached(
                model_id=registered.model_id,
                revision=registered.revision,
                device=device_info.device,
                cache_dir=cache_dir,
            )
        logger.info(
            "Loaded GLiNER %s @ %s on %s (%s) via ort (Rust)",
            registered.model_id,
            registered.revision,
            device_info.device,
            device_info.name,
        )
        return cls(backend, model_id=registered.model_id, device=device_info)

    def extract(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        threshold: float = 0.5,
        max_width: int = 12,
        flat_ner: bool = True,
        dup_label: bool = False,
        multi_label: bool = False,
    ) -> list[list[Entity]]:
        """Run NER extraction over a batch of texts.

        Args:
            texts: Input text sequences.
            labels: Entity-class labels to look for. Each (text, label)
                pair is scored independently; the model has not been
                tuned for any specific label set, so feel free to use
                custom labels like ``"medical condition"`` or
                ``"contract clause type"``.
            threshold: Minimum sigmoid score to accept a span.
                Default 0.5 — matches the upstream Python
                ``predict_entities`` default.
            max_width: Maximum span width in words.
            flat_ner: If True (default), no two output spans may
                overlap. If False, ``dup_label`` and ``multi_label``
                control whether same-label / different-label overlaps
                are allowed.
            dup_label: Permit overlapping spans with the SAME label
                (only effective when ``flat_ner=False``).
            multi_label: Permit overlapping spans with DIFFERENT
                labels (only effective when ``flat_ner=False``).

        Returns:
            One ``list[Entity]`` per input text, in the same order as
            ``texts``. Empty inputs yield empty lists.
        """
        if not texts:
            return []
        if not labels:
            raise ValueError("`labels` must contain at least one label")

        raw = self._backend.extract(
            list(texts),
            list(labels),
            threshold=threshold,
            max_width=max_width,
            flat_ner=flat_ner,
            dup_label=dup_label,
            multi_label=multi_label,
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
def _load_ner_cached(
    model_id: str,
    revision: str,
    device: str,
    cache_dir: str | None = None,
):
    """Process-wide cache of loaded Rust ``NerBackend`` instances.

    Keyed by ``(model_id, revision, device, cache_dir)`` so a registry
    SHA bump invalidates the cached backend.
    """
    try:
        from kaos_nlp_transformers import _rust

        NerBackend = _rust.ner.NerBackend
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
        return NerBackend.load(model_id, device=device, cache_dir=cache_dir)
    except Exception as exc:
        if isinstance(exc, BackendNotInstalledError | ModelLoadError | ModelNotRegisteredError):
            raise
        msg = (
            f"Failed to load NER model {model_id!r} @ {revision} on "
            f"device {device!r}: {exc}. "
            f"Fix: try device='cpu' or model='{DEFAULT_NER_MODEL}'. "
            "Alternative: verify network access on first download "
            "(KAOS_NLP_TRANSFORMERS_OFFLINE)."
        )
        raise ModelLoadError(msg) from exc


__all__ = ["DEFAULT_NER_MODEL", "Entity", "GLiNERExtractor"]
