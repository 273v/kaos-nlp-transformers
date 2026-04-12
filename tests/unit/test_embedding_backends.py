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
