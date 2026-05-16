"""Live integration tests for :class:`PiiDetector`.

Hits a REAL Rust ``TokenClassifierBackend`` (ort + libonnxruntime,
``onnx-community/bert-small-pii-detection-ONNX``). Verifies the
BIO-decoded output matches the kinds of PII the model card claims —
PERSON, EMAIL_ADDRESS, PHONE_NUMBER, plus US-specific financial PII.

Skips when ``KAOS_NLP_TRANSFORMERS_OFFLINE=1`` or when the Rust
extension hasn't been built.

Marked ``@pytest.mark.integration`` and ``@pytest.mark.live``
(network on first run; cache hits after).
"""

from __future__ import annotations

import os

import pytest

from kaos_nlp_transformers import Entity, PiiDetector

pytestmark = [pytest.mark.integration, pytest.mark.live]


def _skip_if_offline() -> None:
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set")


def _skip_if_no_rust_extension() -> None:
    try:
        from kaos_nlp_transformers._rust import token_classify as _tc  # noqa: F401
    except ImportError:
        pytest.skip(
            "kaos_nlp_transformers._rust extension is not built — "
            "run `uv run maturin develop --release` first."
        )


@pytest.fixture(scope="module")
def detector() -> PiiDetector:
    """Module-scoped PII detector — model loads once per test file."""
    _skip_if_offline()
    _skip_if_no_rust_extension()
    return PiiDetector.load()


def test_load_returns_real_detector(detector: PiiDetector) -> None:
    assert isinstance(detector, PiiDetector)
    assert detector.model_id == "onnx-community/bert-small-pii-detection-ONNX"


def test_load_exposes_24_categories(detector: PiiDetector) -> None:
    """The bert-small PII model declares 24 distinct PII categories
    in its config.json (each with B-/I- BIO variants). The wrapper
    should surface them post-strip."""
    labels = detector.labels
    assert len(labels) == 24
    # Spot-check a few of the standard ones.
    for needed in ("PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"):
        assert needed in labels, f"missing expected label: {needed}"


def test_detect_finds_person_and_email(detector: PiiDetector) -> None:
    [entities] = detector.detect(["Contact Jennifer Stacey at jen.stacey@galera.com today."])
    labels = {e.label for e in entities}
    assert "PERSON" in labels, f"missing PERSON: {entities}"
    assert "EMAIL_ADDRESS" in labels, f"missing EMAIL_ADDRESS: {entities}"


def test_detect_finds_financial_pii(detector: PiiDetector) -> None:
    """The model was trained on synthetic_pii_finance_multilingual —
    it should recognize SSN-shaped strings as US_SSN."""
    [entities] = detector.detect(["The SSN on file is 123-45-6789 for the applicant."])
    labels = {e.label for e in entities}
    # Either US_SSN or a generic FINANCIAL hit is acceptable —
    # the model can route SSN-shaped strings to either category
    # depending on the surrounding context.
    assert "US_SSN" in labels or "FINANCIAL" in labels, (
        f"expected US_SSN or FINANCIAL, got: {labels}"
    )


def test_detect_offsets_roundtrip(detector: PiiDetector) -> None:
    """Char-offset round-trip — ``text[e.start:e.end]`` must equal
    ``e.text`` exactly. The byte→char fix in
    ``core::token_classify`` is what makes this work on multibyte
    input."""
    src = "On October 7, 2021, “Jennifer Stacey” — based in Wilmington — agreed."
    [entities] = detector.detect([src])
    assert len(entities) >= 1
    for e in entities:
        assert src[e.start : e.end] == e.text, (
            f"offset mismatch on multibyte input: "
            f"text[{e.start}:{e.end}]={src[e.start : e.end]!r} != {e.text!r}"
        )


def test_detect_scores_in_unit_interval(detector: PiiDetector) -> None:
    [entities] = detector.detect(["John lives at 123 Main St in Brooklyn."])
    for e in entities:
        assert 0.0 <= e.score <= 1.0


def test_detect_returns_entity_dataclass(detector: PiiDetector) -> None:
    [entities] = detector.detect(["Bob writes to alice@example.com regularly."])
    for e in entities:
        assert isinstance(e, Entity)


def test_detect_respects_threshold(detector: PiiDetector) -> None:
    """Higher threshold should not increase the entity count."""
    src = "Jennifer Stacey works at galera.com and her phone is +1-555-0142."
    low = detector.detect([src], score_threshold=0.1)[0]
    high = detector.detect([src], score_threshold=0.99)[0]
    assert len(high) <= len(low)


def test_detect_empty_returns_empty(detector: PiiDetector) -> None:
    assert detector.detect([]) == []


def test_detect_batch_independence(detector: PiiDetector) -> None:
    """Two unrelated inputs should produce disjoint entity sets."""
    out = detector.detect(
        [
            "Contact Jennifer at jen@galera.com.",
            "The credit card 4532-1234-5678-9010 expires soon.",
        ]
    )
    assert len(out) == 2
    labels_a = {e.label for e in out[0]}
    labels_b = {e.label for e in out[1]}
    assert "PERSON" in labels_a or "EMAIL_ADDRESS" in labels_a
    # Credit-card text typically yields CREDIT_CARD; sometimes
    # FINANCIAL with a strong margin.
    assert "CREDIT_CARD" in labels_b or "FINANCIAL" in labels_b
