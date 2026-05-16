"""NliModel — natural-language-inference cross-encoder scorer.

Public surface: an ``NliModel`` whose ``.score(premise, hypotheses)``
returns a list of :class:`NliScore` triples in the canonical
``(entailment, neutral, contradiction)`` order. The shape matches the
``NLIScorer`` Protocol declared in
``kaos_llm_core.programs.classify.nli`` so that
``ZeroShotNLIClassifier`` can be wired directly to an ``NliModel``
instance.

Backend: the in-tree Rust cdylib's ``_rust.nli.NliBackend`` (ort +
libonnxruntime). The Python wrapper does the same registry gate as
``EmbeddingModel`` / ``CrossEncoderReranker``: registered model ids
only (unless ``settings.allow_unregistered`` is true), and excluded
ids are rejected with the recorded license-audit reason.

Default model: ``Xenova/nli-deberta-v3-base`` (Apache-2.0 weight
chain, 184M params, ~244 MB ONNX). See ``NLI_REGISTRY`` for the full
license-chain note.

Example::

    from kaos_nlp_transformers.nli import NliModel
    from kaos_llm_core.programs.classify import ZeroShotNLIClassifier

    scorer = NliModel.load()
    program = ZeroShotNLIClassifier(labels=label_set, scorer=scorer)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from kaos_core.logging import get_logger

from kaos_nlp_transformers.device import DeviceInfo, resolve_device
from kaos_nlp_transformers.embedding import _offline_env_scope
from kaos_nlp_transformers.errors import (
    BackendNotInstalledError,
    ModelLoadError,
    ModelNotRegisteredError,
)
from kaos_nlp_transformers.models import NLI_EXCLUDED, NLI_REGISTRY, RegisteredModel
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

logger = get_logger(__name__)


# Default NLI model — Apache-2.0 chain, 184M params, CPU-friendly.
# Pinned revision lives in ``NLI_REGISTRY``. Derives from the settings
# field default so a single env-var override
# (``KAOS_NLP_TRANSFORMERS_DEFAULT_NLI_MODEL``) updates every call
# site that does not pass an explicit ``model_id``.
DEFAULT_NLI_MODEL: str = KaosNLPTransformersSettings.model_fields["default_nli_model"].default


@dataclass(frozen=True, slots=True)
class NliScore:
    """Single (premise, hypothesis) NLI score triple.

    Field order is the canonical ``(entailment, neutral,
    contradiction)`` tuple expected by the
    :class:`kaos_llm_core.programs.classify.NLIScore` Protocol. The
    Rust backend re-orders raw model outputs into this canonical
    layout regardless of the underlying ONNX checkpoint's
    ``id2label`` permutation.

    The three values are probabilities (post-softmax), so they
    sum to approximately ``1.0`` modulo float32 rounding.
    """

    entailment: float
    neutral: float
    contradiction: float


class NliModel:
    """NLI cross-encoder scorer satisfying the kaos-llm-core
    :class:`NLIScorer` Protocol.

    Implementations of the protocol expose a synchronous
    ``score(premise, hypotheses) -> Sequence[NLIScore]`` method.
    :class:`ZeroShotNLIClassifier` issues exactly one ``score`` call
    per :meth:`Program.forward`, so the underlying ort forward pass
    is amortised across all candidate labels for a given input.
    """

    def __init__(
        self,
        _backend: Any,
        *,
        model_id: str = DEFAULT_NLI_MODEL,
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
    ) -> NliModel:
        """Load an NLI cross-encoder.

        Args:
            model_id: HuggingFace model id. Defaults to
                ``Xenova/nli-deberta-v3-base`` (the ``NLI_REGISTRY``
                default).
            device: Device override (``'auto'``, ``'cpu'``, ``'cuda'``,
                etc.). GPU acceleration requires the ``[gpu]``
                companion wheel (ort/cuda EP).
            settings: Optional settings override. Defaults to a freshly
                constructed :class:`KaosNLPTransformersSettings`.

        Raises:
            ModelNotRegisteredError: If the model is in
                ``NLI_EXCLUDED`` (license-blocked) or is not in
                ``NLI_REGISTRY`` and ``settings.allow_unregistered`` is
                false.
            BackendNotInstalledError: If the Rust cdylib was not built
                or the requested device's feature flag is missing
                (e.g. ``cuda`` without ``--features gpu``).
            ModelLoadError: If the model fails to load.
        """
        s = settings if settings is not None else KaosNLPTransformersSettings()
        target = model_id or DEFAULT_NLI_MODEL

        if target in NLI_EXCLUDED:
            reason = NLI_EXCLUDED[target]
            msg = (
                f"NLI model {target!r} is excluded from the registry: "
                f"{reason}. Fix: pick a permissively-licensed alternative "
                "from kaos_nlp_transformers.models.NLI_REGISTRY. "
                "Alternative: if you have a commercial license arrangement, "
                "set KAOS_NLP_TRANSFORMERS_ALLOW_UNREGISTERED=true (use with care)."
            )
            raise ModelNotRegisteredError(msg)

        if target not in NLI_REGISTRY:
            if not s.allow_unregistered:
                available = ", ".join(sorted(NLI_REGISTRY.keys()))
                msg = (
                    f"NLI model {target!r} is not in the v0 registry. "
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
                dim=3,
                backend="ort",
                notes="unregistered NLI model",
            )
        else:
            registered = NLI_REGISTRY[target]

        req_device = device or s.device
        device_info = resolve_device(req_device)
        cache_dir = str(s.cache_dir) if s.cache_dir else None

        with _offline_env_scope(s.offline):
            backend = _load_nli_cached(
                model_id=registered.model_id,
                revision=registered.revision,
                device=device_info.device,
                cache_dir=cache_dir,
            )
        logger.info(
            "Loaded NLI %s @ %s on %s (%s) via ort (Rust)",
            registered.model_id,
            registered.revision,
            device_info.device,
            device_info.name,
        )
        return cls(backend, model_id=registered.model_id, device=device_info)

    def score(
        self,
        premise: str,
        hypotheses: Sequence[str],
    ) -> Sequence[NliScore]:
        """Score one premise against many hypotheses in one forward pass.

        Matches :class:`kaos_llm_core.programs.classify.NLIScorer`
        exactly: synchronous, ``Sequence[NLIScore]`` return type.

        Args:
            premise: The premise text.
            hypotheses: One or more hypothesis texts. The premise is
                paired against each hypothesis and all pairs are scored
                in a single batch.

        Returns:
            A list of :class:`NliScore` triples, one per hypothesis,
            in the same order as the input. Each triple is in canonical
            ``(entailment, neutral, contradiction)`` order with
            probabilities summing to approximately ``1.0``.
        """
        if not hypotheses:
            return []

        premises = [premise] * len(hypotheses)
        probs: np.ndarray = self._backend.score(premises, list(hypotheses))
        # Shape contract from the Rust side: (n_pairs, 3). Trust it
        # rather than re-validating on every call.
        return [
            NliScore(
                entailment=float(row[0]),
                neutral=float(row[1]),
                contradiction=float(row[2]),
            )
            for row in probs
        ]


@lru_cache(maxsize=4)
def _load_nli_cached(
    model_id: str,
    revision: str,
    device: str,
    cache_dir: str | None = None,
):
    """Process-wide cache of loaded Rust ``NliBackend`` instances.

    Keyed by ``(model_id, revision, device, cache_dir)`` so a registry
    SHA bump invalidates the cached backend. Mirrors the reranker
    cache pattern.
    """
    try:
        from kaos_nlp_transformers import _rust

        NliBackend = _rust.nli.NliBackend
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
        return NliBackend.load(model_id, device=device, cache_dir=cache_dir)
    except Exception as exc:
        if isinstance(exc, BackendNotInstalledError | ModelLoadError | ModelNotRegisteredError):
            raise
        msg = (
            f"Failed to load NLI model {model_id!r} @ {revision} on "
            f"device {device!r}: {exc}. "
            f"Fix: try device='cpu' or model='{DEFAULT_NLI_MODEL}'. "
            "Alternative: verify network access on first download "
            "(KAOS_NLP_TRANSFORMERS_OFFLINE)."
        )
        raise ModelLoadError(msg) from exc


__all__ = ["DEFAULT_NLI_MODEL", "NliModel", "NliScore"]
