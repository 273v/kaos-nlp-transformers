"""Static shape checks on the model registry — no network needed.

These tests are the binding license audit: they enforce the rules in
``docs/internal/prd/kaos-nlp-transformers.md`` §6 at every build, so a
malformed or non-permissively-licensed entry can never silently land.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_v0_registry_has_default_model():
    from kaos_nlp_transformers.models import REGISTRY

    assert "BAAI/bge-small-en-v1.5" in REGISTRY


def test_every_registered_model_pins_a_real_revision():
    from kaos_nlp_transformers.models import REGISTRY

    for model_id, entry in REGISTRY.items():
        assert entry.revision, f"{model_id} has no pinned revision"
        assert entry.revision != "main", (
            f"{model_id} pinned to 'main' (forbidden by hard rule §11.2)"
        )
        assert len(entry.revision) >= 7, f"{model_id} revision SHA too short ({entry.revision!r})"


def test_every_registered_model_declares_a_permissive_license():
    from kaos_nlp_transformers.models import REGISTRY

    permissive = {"MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause"}
    for model_id, entry in REGISTRY.items():
        assert entry.license in permissive, (
            f"{model_id} has non-permissive license {entry.license!r}; "
            f"must be one of {sorted(permissive)}"
        )


def test_every_registered_model_declares_a_dim():
    from kaos_nlp_transformers.models import REGISTRY

    for model_id, entry in REGISTRY.items():
        assert entry.dim > 0, f"{model_id} has dim={entry.dim} (must be > 0)"


def test_every_registered_model_has_a_supported_backend():
    from kaos_nlp_transformers.models import REGISTRY

    valid = {"fastembed", "sentence-transformers"}
    for model_id, entry in REGISTRY.items():
        assert entry.backend in valid, (
            f"{model_id} has backend={entry.backend!r}; must be one of {sorted(valid)}"
        )


def test_excluded_list_documented_with_reasons():
    from kaos_nlp_transformers.models import EXCLUDED

    # Sanity: the exclusion list isn't empty (license audit always finds at
    # least the CC-BY-NC family).
    assert len(EXCLUDED) >= 5
    for model_id, reason in EXCLUDED.items():
        assert reason, f"{model_id} excluded with no reason"


def test_excluded_models_are_not_in_registry():
    """A model can't be both excluded AND registered. Catch double-listing."""
    from kaos_nlp_transformers.models import EXCLUDED, REGISTRY

    for model_id in EXCLUDED:
        assert model_id not in REGISTRY, f"{model_id} is in BOTH EXCLUDED and REGISTRY — pick one"
