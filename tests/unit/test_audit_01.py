"""Regression tests for audit-01 findings KNT-001..KNT-006.

These tests pin the fixes so a future refactor can't silently re-introduce
the original problems.
"""

from __future__ import annotations

import os
import re

import pytest

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# KNT-001 — no upward dependency on kaos-ml-core
# -----------------------------------------------------------------------------


def test_no_kaos_ml_core_import_anywhere():
    """The package must not import ``kaos_ml_core`` at any level — kaos-ml-core
    is the documented downstream consumer (Tier 4) and importing it from
    kaos-nlp-transformers (Tier 3) would invert the DAG and cause hidden
    runtime failures for users who install only the lower-tier package.
    """
    import pathlib

    import kaos_nlp_transformers

    pkg_root = pathlib.Path(kaos_nlp_transformers.__file__).parent
    offenders: list[tuple[pathlib.Path, int, str]] = []
    pat = re.compile(r"\b(?:from|import)\s+kaos_ml_core\b")
    for py in pkg_root.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            if pat.search(line):
                offenders.append((py.relative_to(pkg_root), lineno, line.strip()))
    assert not offenders, (
        "kaos_ml_core imports found in kaos-nlp-transformers (would invert the "
        f"dependency DAG): {offenders}"
    )


# -----------------------------------------------------------------------------
# KNT-002 — scipy is gated with an actionable install-hint
# -----------------------------------------------------------------------------


def test_semantic_dedup_raises_install_hint_when_scipy_missing(monkeypatch):
    """SemanticDedupLevel.find_clusters must raise an ImportError that
    references the [clustering] extra when scipy is unavailable, not let the
    raw ModuleNotFoundError from `import scipy` bubble up unhinted.
    """
    import sys

    from kaos_content.dedup.types import DedupDocument

    from kaos_nlp_transformers.clustering.semantic_dedup import SemanticDedupLevel

    # Hide every scipy submodule attempt by inserting None (pep 328 sentinel
    # that makes `from X import Y` fail).
    for mod in list(sys.modules):
        if mod == "scipy" or mod.startswith("scipy."):
            monkeypatch.setitem(sys.modules, mod, None)
    monkeypatch.setitem(sys.modules, "scipy.cluster.hierarchy", None)
    monkeypatch.setitem(sys.modules, "scipy.spatial.distance", None)

    docs = [DedupDocument(doc_id=str(i), text=f"document {i}") for i in range(3)]
    with pytest.raises(ImportError, match=r"\[clustering\]"):
        SemanticDedupLevel().find_clusters(docs)


# -----------------------------------------------------------------------------
# KNT-003 — RegisteredModel.revision is threaded through loader cache keys
# -----------------------------------------------------------------------------


def test_loader_signatures_accept_revision():
    """All backend loaders must accept a ``revision`` argument so the
    registered model SHA is part of the cache key (different revision →
    different cached backend).

    Audit-06 KNT-501: post-torch-removal, the surviving backends are
    fastembed (ONNX) and model2vec (static numpy). The
    ``_load_sentence_transformers_cached`` check was retired with the
    sentence-transformers backend.
    """
    import inspect

    from kaos_nlp_transformers.embedding import (
        _load_model2vec_cached,
        _load_rust_embedding_cached,
    )

    fe_params = inspect.signature(_load_rust_embedding_cached).parameters
    m2v_params = inspect.signature(_load_model2vec_cached).parameters
    assert "revision" in fe_params, (
        "fastembed loader must accept revision (cache-key participant) per audit-01 KNT-003."
    )
    assert "revision" in m2v_params, (
        "model2vec loader must accept revision (cache-key + snapshot pin) per audit-01 KNT-003."
    )


# -----------------------------------------------------------------------------
# KNT-004 — settings injection on retriever and dedup factories
# -----------------------------------------------------------------------------


def test_retriever_factories_accept_settings():
    """``EmbeddingRetriever.from_texts`` and ``.from_corpus`` must accept a
    ``settings`` keyword so cache/offline/device policy can be injected at
    the edge.
    """
    import inspect

    from kaos_nlp_transformers.retrieval import EmbeddingRetriever

    for fn_name in ("from_texts", "from_corpus"):
        fn = getattr(EmbeddingRetriever, fn_name)
        params = inspect.signature(fn).parameters
        assert "settings" in params, (
            f"EmbeddingRetriever.{fn_name} must accept settings kwarg per audit-01 KNT-004."
        )


def test_semantic_dedup_accepts_settings():
    """``SemanticDedupLevel.__init__`` must accept ``settings`` kwarg."""
    import inspect

    from kaos_nlp_transformers.clustering.semantic_dedup import SemanticDedupLevel

    params = inspect.signature(SemanticDedupLevel.__init__).parameters
    assert "settings" in params, (
        "SemanticDedupLevel must accept settings kwarg per audit-01 KNT-004."
    )


# -----------------------------------------------------------------------------
# KNT-005 — settings.offline is enforced at the load boundary
# -----------------------------------------------------------------------------


def test_offline_setting_sets_hf_env_vars(monkeypatch):
    """``KaosNLPTransformersSettings(offline=True)`` makes ``EmbeddingModel.load``
    set ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` to ``"1"`` while the
    backend is being constructed.

    Audit-02 KNT-103 changed the original setdefault-and-leak behavior into a
    scoped context manager. We capture the env-var state by intercepting the
    backend loader (which sees the env mid-scope), then assert restoration
    behavior in the audit-02 regression suite.
    """
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    # Audit-03 KNT-201 added a free-threaded Python guard at the top of
    # EmbeddingModel.load. On a Py_GIL_DISABLED interpreter (3.14t etc.)
    # the guard fires BEFORE the backend loader, so the offline env-var
    # capture below never runs. Force the GIL-enabled path here so this
    # audit-01 test exercises only the offline-scope contract.
    import sys

    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)

    from kaos_nlp_transformers import embedding as embedding_mod
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    captured: dict[str, str | None] = {}

    def _capture_then_explode(*args, **kwargs):
        captured["HF_HUB_OFFLINE"] = os.environ.get("HF_HUB_OFFLINE")
        captured["TRANSFORMERS_OFFLINE"] = os.environ.get("TRANSFORMERS_OFFLINE")
        msg = "stub backend (env-var capture only)"
        raise RuntimeError(msg)

    # Audit-06 KNT-501: only fastembed + model2vec backends remain, so we
    # only need to stub _load_rust_embedding_cached here. The previous
    # _load_sentence_transformers_cached stub was retired with the SE
    # backend.
    monkeypatch.setattr(embedding_mod, "_load_rust_embedding_cached", _capture_then_explode)

    s = KaosNLPTransformersSettings(offline=True)
    with pytest.raises(RuntimeError, match="stub backend"):
        embedding_mod.EmbeddingModel.load(settings=s)

    # Mid-scope (when the backend loader ran), both vars were "1".
    assert captured["HF_HUB_OFFLINE"] == "1"
    assert captured["TRANSFORMERS_OFFLINE"] == "1"


# -----------------------------------------------------------------------------
# KNT-006 — top-level __all__ has stable, ruff-validated ordering
# -----------------------------------------------------------------------------


def test_top_level_all_passes_isort_check():
    """The top-level ``__all__`` must satisfy ruff's RUF022 isort rule.

    The audit recommended pure-alphabetical sort, but ruff's authoritative
    interpretation (the rule the rest of the codebase enforces) groups
    SCREAMING_CASE constants ahead of mixed-case names. This test pins the
    ruff-validated ordering so a future re-sort can't drift the public API.
    """
    import kaos_nlp_transformers

    actual = list(kaos_nlp_transformers.__all__)
    # Constants first (uppercase-only), then mixed-case, then dunder, then
    # lowercase callables — matches ruff RUF022 when ruff check is green.
    # Audit-02 KNT-104 added RERANKER_EXCLUDED + RERANKER_REGISTRY constants.
    # audit-04 KNT-301..302 added LatentDevice + DeviceNotReachableError
    # for the OS-level GPU latent surface (see device.py / errors.py).
    expected_groups = {
        "constants": ["EXCLUDED", "REGISTRY", "RERANKER_EXCLUDED", "RERANKER_REGISTRY"],
        "classes": [
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
        ],
        "dunder": ["__version__"],
        "callables": ["detect_devices"],
    }
    expected_full = (
        expected_groups["constants"]
        + expected_groups["classes"]
        + expected_groups["dunder"]
        + expected_groups["callables"]
    )
    assert actual == expected_full, (
        f"__all__ drift detected. actual={actual}, expected={expected_full}"
    )
