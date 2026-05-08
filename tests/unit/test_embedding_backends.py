"""Unit tests for multi-backend embedding — backend resolution and settings."""

from __future__ import annotations

import pytest

from kaos_nlp_transformers.device import DeviceInfo
from kaos_nlp_transformers.embedding import _resolve_backend
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

pytestmark = pytest.mark.unit


# -- Settings fields -------------------------------------------------------


def test_settings_default_device():
    s = KaosNLPTransformersSettings()
    assert s.device == "auto"


def test_settings_default_backend():
    s = KaosNLPTransformersSettings()
    assert s.backend == "auto"


def test_settings_device_from_env(monkeypatch):
    monkeypatch.setenv("KAOS_NLP_TRANSFORMERS_DEVICE", "cuda:1")
    s = KaosNLPTransformersSettings()
    assert s.device == "cuda:1"


def test_settings_backend_from_env(monkeypatch):
    monkeypatch.setenv("KAOS_NLP_TRANSFORMERS_BACKEND", "sentence-transformers")
    s = KaosNLPTransformersSettings()
    assert s.backend == "sentence-transformers"


# -- _resolve_backend ------------------------------------------------------


def _cpu() -> DeviceInfo:
    return DeviceInfo(name="CPU", device="cpu", backend="fastembed")


def _gpu() -> DeviceInfo:
    return DeviceInfo(name="GPU", device="cuda:0", backend="sentence-transformers")


def test_resolve_backend_explicit_fastembed():
    assert _resolve_backend("fastembed", _gpu(), "fastembed") == "fastembed"


def test_resolve_backend_explicit_st():
    assert _resolve_backend("sentence-transformers", _cpu(), "fastembed") == "sentence-transformers"


def test_resolve_backend_auto_cpu_uses_registry():
    assert _resolve_backend("auto", _cpu(), "fastembed") == "fastembed"
    assert _resolve_backend("auto", _cpu(), "sentence-transformers") == "sentence-transformers"


def test_resolve_backend_auto_gpu_uses_device_backend():
    # GPU device recommends sentence-transformers
    assert _resolve_backend("auto", _gpu(), "fastembed") == "sentence-transformers"


# -- model2vec backend (audit-04 KNT-302) ----------------------------------


def test_resolve_backend_explicit_model2vec():
    assert _resolve_backend("model2vec", _cpu(), "fastembed") == "model2vec"
    # An explicit "model2vec" wins even on a GPU device — static models
    # have no GPU codepath, so honoring the user's choice (and pinning
    # CPU at load time) is the right behavior.
    assert _resolve_backend("model2vec", _gpu(), "fastembed") == "model2vec"


def test_resolve_backend_auto_routes_static_model_to_model2vec():
    """A registry entry with backend='model2vec' MUST stay on model2vec
    even when the device probe sees a GPU. Static models have no GPU
    codepath; routing them to sentence-transformers would force the user
    to install [torch] for a model that doesn't need it."""
    assert _resolve_backend("auto", _cpu(), "model2vec") == "model2vec"
    assert _resolve_backend("auto", _gpu(), "model2vec") == "model2vec"


def test_resolve_backend_rejects_unknown_value():
    """Audit-02 KNT-107: unknown backend names raise instead of falling
    through. Audit-04 update keeps the same shape but mentions model2vec
    in the install hint."""
    with pytest.raises(ValueError, match="Invalid backend"):
        _resolve_backend("tensorflow", _cpu(), "fastembed")


def test_load_model2vec_cached_missing_dep_raises_friendly_error(monkeypatch):
    """Without the [model2vec] extra, loader raises BackendNotInstalledError
    with the install hint — not a cryptic ImportError out of the encode call.
    Verifies the three-part message contract (what / fix / alternative)."""
    import builtins

    from kaos_nlp_transformers.embedding import _load_model2vec_cached
    from kaos_nlp_transformers.errors import BackendNotInstalledError

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "model2vec":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Bypass the lru_cache — the loader is module-level, so a previous
    # successful load in this process would otherwise short-circuit.
    _load_model2vec_cached.cache_clear()

    with pytest.raises(BackendNotInstalledError) as exc_info:
        _load_model2vec_cached(
            model_id="minishlab/potion-retrieval-32M",
            revision="6fc8051fab2a1e0ee76689cf08c853792ac285e7",
            cache_dir=None,
        )
    msg = str(exc_info.value)
    # All three parts of the contract should be present.
    assert "model2vec is not installed" in msg
    assert "kaos-nlp-transformers[model2vec]" in msg
    assert "Alternative" in msg
    # Reset the cache so subsequent tests get a fresh slate.
    _load_model2vec_cached.cache_clear()
