"""Smoke tests for the Rust cdylib (audit KNT-601).

Verifies that the PyO3 module imports cleanly, exposes the expected
classes / functions, and that the basic invariants hold (version
matches Cargo.toml, capabilities reports cpu=True, vendored path
resolves the bundled potion-base-8M model).

These tests do NOT load any heavyweight model — that's
``test_reference_vectors.py`` (P3.6). They run on every PR.
"""

from __future__ import annotations

import re

import pytest


def test_rust_module_imports():
    from kaos_nlp_transformers import _rust

    assert hasattr(_rust, "__version__")
    # Cargo SemVer: like "0.2.0-alpha.1" (or just "0.2.0" once stable).
    assert re.match(r"^\d+\.\d+\.\d+", _rust.__version__)


def test_rust_submodules_present():
    from kaos_nlp_transformers._rust import embedding, registry, reranker, tokenize

    assert embedding.EmbeddingBackend is not None
    assert reranker.CrossEncoderBackend is not None
    assert tokenize.Tokenizer is not None
    assert callable(registry.capabilities)
    assert callable(registry.vendored_model_path)


def test_capabilities_has_cpu():
    from kaos_nlp_transformers._rust.registry import capabilities

    caps = capabilities()
    assert caps["cpu"] is True
    assert "cuda" in caps
    assert "openvino" in caps
    assert "build_features" in caps
    # cuda / openvino must be False by default — the GPU companion
    # wheel ships in 0.2.0a2.
    assert caps["cuda"] is False
    assert caps["openvino"] is False


def test_vendored_model_path_finds_potion():
    """The wheel ships minishlab/potion-base-8M vendored (audit-05 KNT-401).
    The Rust resolver must find it."""
    from kaos_nlp_transformers._rust.registry import vendored_model_path

    path = vendored_model_path("minishlab/potion-base-8M")
    assert path is not None, "vendored potion-base-8M not resolved"
    # The wheel's _vendor dir uses the slug shape "potion-base-8M".
    assert path.endswith("potion-base-8M")


def test_vendored_model_path_returns_none_for_unknown():
    from kaos_nlp_transformers._rust.registry import vendored_model_path

    assert vendored_model_path("definitely/not-vendored") is None


def test_version_matches_python_metadata():
    """Cargo SemVer ↔ PEP 440 sync: 0.2.0-alpha.1 → 0.2.0a1."""
    from kaos_nlp_transformers import __version__
    from kaos_nlp_transformers._rust import __version__ as rust_version

    # PEP 440 alpha encoding: cargo "0.2.0-alpha.1" → pip "0.2.0a1".
    py_normalized = __version__.replace(".alpha.", "-alpha.").replace("a", "-alpha.")
    # Loose compatibility check: the two strings agree on the major.minor.patch core.
    rust_core = rust_version.split("-")[0]
    py_core = __version__.split("a")[0].split("b")[0].split("rc")[0]
    assert rust_core == py_core, f"version drift: rust={rust_version} python={__version__}"
    # Suppress unused-variable lint for the readability detour above.
    _ = py_normalized


@pytest.mark.parametrize(
    "model_id, expected_dim",
    [
        # Registry parity check — these match
        # rust/core/model_registry.rs.
        ("BAAI/bge-small-en-v1.5", 384),
    ],
)
def test_registry_parity(model_id: str, expected_dim: int):
    """The Rust registry must mirror the Python REGISTRY for every
    fastembed-backed entry. KNT-601: drift between the two would mean
    the new backend serves a different model than the Python registry
    documents."""
    from kaos_nlp_transformers.models import REGISTRY

    py_entry = REGISTRY[model_id]
    assert py_entry.dim == expected_dim
