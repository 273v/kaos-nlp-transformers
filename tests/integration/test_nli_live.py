"""Live integration tests for ``NliModel``.

Hits a REAL Rust ``NliBackend`` (ort + libonnxruntime,
Xenova/nli-deberta-v3-base) — no mocks. Verifies that the canonical
``(entailment, neutral, contradiction)`` re-ordering is correct on
the SNLI/MNLI evaluation triple and that the public surface stays
type-stable.

Skips when ``KAOS_NLP_TRANSFORMERS_OFFLINE=1`` or when the Rust
extension hasn't been built.

Marked ``@pytest.mark.integration`` and ``@pytest.mark.live`` (network).
"""

from __future__ import annotations

import os

import pytest

from kaos_nlp_transformers import NliModel, NliScore

pytestmark = [pytest.mark.integration, pytest.mark.live]


def _skip_if_offline() -> None:
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set")


def _skip_if_no_rust_extension() -> None:
    try:
        from kaos_nlp_transformers._rust import nli as _nli  # noqa: F401
    except ImportError:
        pytest.skip(
            "kaos_nlp_transformers._rust extension is not built — "
            "run `uv run maturin develop --release` first."
        )


@pytest.fixture(scope="module")
def model() -> NliModel:
    """Module-scoped NLI model so the ~244 MB ONNX downloads once."""
    _skip_if_offline()
    _skip_if_no_rust_extension()

    return NliModel.load()  # default = Xenova/nli-deberta-v3-base


def test_load_returns_real_nli_model(model: NliModel) -> None:
    assert isinstance(model, NliModel)
    assert model.model_id == "Xenova/nli-deberta-v3-base"


def test_load_uses_rust_nli_backend(model: NliModel) -> None:
    """The underlying backend must be the in-tree Rust ``NliBackend``,
    not a Python fallback."""
    from kaos_nlp_transformers._rust import nli as _nli

    assert isinstance(model._backend, _nli.NliBackend)


def test_score_returns_n_scores_for_n_hypotheses(model: NliModel) -> None:
    scores = model.score(
        "A man is checking the uniform of a figure.",
        ["Someone is inspecting a uniform.", "The sky is blue.", "It is raining."],
    )
    assert len(scores) == 3
    assert all(isinstance(s, NliScore) for s in scores)


def test_score_probabilities_in_unit_interval(model: NliModel) -> None:
    scores = model.score(
        "Two dogs are running in a park.",
        ["The dogs are playing.", "The dogs are sleeping in bed."],
    )
    for s in scores:
        for value in (s.entailment, s.neutral, s.contradiction):
            assert 0.0 <= value <= 1.0, f"value {value} outside [0, 1]"


def test_score_probabilities_sum_to_one(model: NliModel) -> None:
    """Softmax invariant — each triple should sum to ~1.0 (float32
    rounding tolerance)."""
    scores = model.score(
        "Two dogs are running in a park.",
        ["The dogs are playing outside.", "It is sunny today."],
    )
    for s in scores:
        total = s.entailment + s.neutral + s.contradiction
        assert total == pytest.approx(1.0, abs=1e-3), f"probs sum {total}, expected ~1"


def test_canonical_order_entailment_wins_on_entailment_pair(model: NliModel) -> None:
    """Headline correctness check: when premise entails hypothesis,
    ``entailment`` must be the largest of the three classes.

    This is the test that catches a mistaken canonical permutation.
    If ``id2label`` is mis-decoded on the Rust side, ``entailment``
    will not be the argmax for the obviously-entailing pair below
    (SNLI gold-label style).
    """
    [s] = model.score(
        premise="A soccer game with multiple males playing.",
        hypotheses=["Some men are playing a sport."],
    )
    triple = (s.entailment, s.neutral, s.contradiction)
    assert max(triple) == s.entailment, (
        "entailment should be argmax for an obviously-entailing pair; "
        f"got entail={s.entailment}, neutral={s.neutral}, contradict={s.contradiction}"
    )


def test_canonical_order_contradiction_wins_on_contradiction_pair(
    model: NliModel,
) -> None:
    """Mirror of the entailment check: a contradictory hypothesis must
    have ``contradiction`` as the argmax of the canonical triple."""
    [s] = model.score(
        premise="A soccer game with multiple males playing.",
        hypotheses=["A group of women is baking bread."],
    )
    triple = (s.entailment, s.neutral, s.contradiction)
    assert max(triple) == s.contradiction, (
        "contradiction should be argmax for a contradictory pair; "
        f"got entail={s.entailment}, neutral={s.neutral}, contradict={s.contradiction}"
    )


def test_empty_hypotheses_returns_empty(model: NliModel) -> None:
    """Empty input short-circuits and returns an empty list."""
    result = model.score("any premise", [])
    assert list(result) == []
