"""Unit tests for :class:`kaos_nlp_transformers.nli.NliModel`.

Offline-friendly: uses an in-process fake backend (mirroring the Rust
``NliBackend.score`` contract) wired in via ``lru_cache`` injection so
the registry gating, score-shape contract, and Protocol conformance
are covered without downloading any model.

The real Rust backend is exercised separately in
``tests/integration/test_nli_live.py`` (marked ``live`` /
``integration``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from kaos_nlp_transformers import NLI_EXCLUDED, NLI_REGISTRY, NliModel, NliScore
from kaos_nlp_transformers.errors import ModelNotRegisteredError
from kaos_nlp_transformers.nli import DEFAULT_NLI_MODEL
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings


class _FakeBackend:
    """In-process stand-in for ``_rust.nli.NliBackend``.

    Returns a deterministic (n_pairs, 3) float32 array. The fixed
    triple `(0.7, 0.2, 0.1)` is enough for the structural tests; the
    actual three-class behavior is covered by the live integration
    test against the real model.
    """

    def __init__(self, *, dim: int = 3) -> None:
        self._dim = dim
        self.calls: list[tuple[list[str], list[str], int]] = []
        self.model_id = "Xenova/nli-deberta-v3-base"
        self.device = "cpu"

    def score(
        self,
        premises: list[str],
        hypotheses: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        assert len(premises) == len(hypotheses)
        self.calls.append((list(premises), list(hypotheses), batch_size))
        n = len(premises)
        # Constant triple, broadcast to (n, 3). The values sum to 1.0
        # so the resulting NliScore is a valid probability triple.
        return np.tile(np.array([0.7, 0.2, 0.1], dtype=np.float32), (n, 1))


def _make_model(backend: _FakeBackend) -> NliModel:
    """Construct an NliModel that bypasses the cdylib load path."""
    return NliModel(backend, model_id=backend.model_id, device=None)


# -- Registry / load gating ------------------------------------------------


def test_default_nli_model_matches_registry() -> None:
    """The settings default must be in the registry by construction."""
    assert DEFAULT_NLI_MODEL in NLI_REGISTRY


def test_load_rejects_excluded_model() -> None:
    """The exclusion list is hard policy — overriding via constructor arg
    must still raise."""
    # Pick any excluded id (registry-fixture guarantees at least one).
    excluded_id = next(iter(NLI_EXCLUDED))
    with pytest.raises(ModelNotRegisteredError) as exc_info:
        NliModel.load(excluded_id)
    assert "excluded" in str(exc_info.value).lower()
    assert excluded_id in str(exc_info.value)


def test_load_rejects_unregistered_when_not_allowed() -> None:
    s = KaosNLPTransformersSettings(allow_unregistered=False)
    with pytest.raises(ModelNotRegisteredError) as exc_info:
        NliModel.load("definitely/not-registered", settings=s)
    assert "not in the v0 registry" in str(exc_info.value)


# -- Score shape / NLIScorer Protocol --------------------------------------


def test_score_returns_one_triple_per_hypothesis() -> None:
    backend = _FakeBackend()
    model = _make_model(backend)

    scores = model.score("a premise", ["h1", "h2", "h3"])

    assert isinstance(scores, Sequence)
    assert len(scores) == 3
    assert all(isinstance(s, NliScore) for s in scores)


def test_score_broadcasts_premise_across_hypotheses() -> None:
    """The premise is paired against each hypothesis once."""
    backend = _FakeBackend()
    model = _make_model(backend)

    model.score("the premise", ["h1", "h2"])

    assert len(backend.calls) == 1
    premises, hypotheses, _ = backend.calls[0]
    assert premises == ["the premise", "the premise"]
    assert hypotheses == ["h1", "h2"]


def test_score_canonical_order_matches_protocol_fields() -> None:
    """The (entailment, neutral, contradiction) order is the
    Protocol contract — the wrapper must produce attributes in that
    canonical layout regardless of the underlying ONNX permutation
    (which is handled by the Rust side and stubbed in the fake)."""
    backend = _FakeBackend()
    model = _make_model(backend)

    [s] = model.score("p", ["h"])
    # Fake backend returns (0.7, 0.2, 0.1).
    assert s.entailment == pytest.approx(0.7)
    assert s.neutral == pytest.approx(0.2)
    assert s.contradiction == pytest.approx(0.1)


def test_score_probabilities_sum_to_one_approximately() -> None:
    backend = _FakeBackend()
    model = _make_model(backend)

    [s] = model.score("p", ["h"])
    total = s.entailment + s.neutral + s.contradiction
    assert total == pytest.approx(1.0, abs=1e-6)


def test_score_empty_hypotheses_short_circuits() -> None:
    backend = _FakeBackend()
    model = _make_model(backend)

    result = model.score("p", [])
    assert list(result) == []
    # No backend call should have happened on the empty path.
    assert backend.calls == []


def test_score_accepts_arbitrary_sequence() -> None:
    """The Protocol declares ``hypotheses: Sequence[str]`` — tuples
    must work the same as lists."""
    backend = _FakeBackend()
    model = _make_model(backend)

    scores = model.score("p", ("h1", "h2"))
    assert len(scores) == 2


# -- NLIScorer Protocol structural conformance -----------------------------


def test_nli_model_satisfies_runtime_nli_scorer_protocol() -> None:
    """isinstance() check against the runtime-checkable Protocol in
    kaos-llm-core. This is the closure-test for N6 — skipped when
    kaos-llm-core isn't installed (kaos-llm-core is a consumer of this
    package, not a dependency, so it isn't on the base test
    requirement list)."""
    nli_module = pytest.importorskip("kaos_llm_core.programs.classify.nli")
    NLIScorer = nli_module.NLIScorer

    backend = _FakeBackend()
    model = _make_model(backend)

    assert isinstance(model, NLIScorer)


def test_nli_score_satisfies_runtime_nli_score_protocol() -> None:
    nli_module = pytest.importorskip("kaos_llm_core.programs.classify.nli")
    NLIScore_proto = nli_module.NLIScore

    s = NliScore(entailment=0.6, neutral=0.3, contradiction=0.1)
    assert isinstance(s, NLIScore_proto)
    # Field access through the Protocol attributes:
    assert s.entailment == pytest.approx(0.6)
    assert s.neutral == pytest.approx(0.3)
    assert s.contradiction == pytest.approx(0.1)
