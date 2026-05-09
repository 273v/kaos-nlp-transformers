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

    # Audit history: KNT-501 (0.1.0a6) retired sentence-transformers;
    # KNT-601 (0.2.0) retired fastembed. The surviving backends are
    # ``"ort"`` (Rust + libonnxruntime) and ``"model2vec"`` (static
    # numpy lookup).
    valid = {"ort", "model2vec"}
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


# audit-04 KNT-301: model2vec entries with full 40-char SHA pins ----


def test_revisions_are_full_40char_shas():
    """Pinned revisions must be the full 40-char SHA from huggingface.co.

    The 7-char minimum in ``test_every_registered_model_pins_a_real_revision``
    catches obviously-broken pins; this stricter check catches sloppy ones
    (truncated SHAs, branch names disguised as SHAs) before they make it
    into a registry that downstream caches key against. The audit-04 entries
    were sourced via huggingface_hub.HfApi().model_info(...).sha so we hold
    every entry to that bar going forward.
    """
    import re

    from kaos_nlp_transformers.models import REGISTRY

    sha_re = re.compile(r"^[0-9a-f]{40}$")
    for model_id, entry in REGISTRY.items():
        assert sha_re.match(entry.revision), (
            f"{model_id} revision {entry.revision!r} is not a 40-char hex SHA "
            "(check huggingface.co/api/models/<id>.sha and re-pin)"
        )


def test_model2vec_entries_present():
    """The audit-04 sweep added the two pinned potion models."""
    from kaos_nlp_transformers.models import REGISTRY

    for model_id in ("minishlab/potion-retrieval-32M", "minishlab/potion-base-32M"):
        assert model_id in REGISTRY, f"{model_id} missing from REGISTRY"
        entry = REGISTRY[model_id]
        assert entry.backend == "model2vec"
        assert entry.license == "MIT"
        assert entry.dim == 512
        assert entry.params_m == 32
