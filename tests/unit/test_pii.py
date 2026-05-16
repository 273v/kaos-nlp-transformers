"""Unit tests for :class:`kaos_nlp_transformers.pii.PiiDetector`.

Offline-friendly — uses a fake backend that mirrors the Rust
``TokenClassifierBackend.classify`` dict contract. Live behavior is
covered separately in ``tests/integration/test_pii_live.py``.
"""

from __future__ import annotations

import pytest

from kaos_nlp_transformers import PII_EXCLUDED, PII_REGISTRY, Entity, PiiDetector
from kaos_nlp_transformers.errors import ModelNotRegisteredError
from kaos_nlp_transformers.pii import DEFAULT_PII_MODEL
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings


class _FakeBackend:
    """Stand-in for ``_rust.token_classify.TokenClassifierBackend``."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], float]] = []
        self.model_id = "onnx-community/bert-small-pii-detection-ONNX"
        self.device = "cpu"
        self.labels = ["EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER"]

    def classify(self, texts, *, score_threshold: float = 0.5):
        self.calls.append((list(texts), score_threshold))
        # Fixed two-entity response per input.
        return [
            [
                {
                    "start": 0,
                    "end": 5,
                    "text": t[:5],
                    "label": "PERSON",
                    "score": 0.95,
                },
                {
                    "start": 6,
                    "end": 16,
                    "text": t[6:16] if len(t) >= 16 else t,
                    "label": "EMAIL_ADDRESS",
                    "score": 0.92,
                },
            ]
            for t in texts
        ]


def _make_detector(backend: _FakeBackend) -> PiiDetector:
    return PiiDetector(
        backend,
        model_id=backend.model_id,
        device=None,
        labels=list(backend.labels),
    )


# -- Registry / load gating ------------------------------------------------


def test_default_pii_model_in_registry() -> None:
    assert DEFAULT_PII_MODEL in PII_REGISTRY


def test_load_rejects_excluded_model() -> None:
    excluded_id = next(iter(PII_EXCLUDED))
    with pytest.raises(ModelNotRegisteredError) as exc_info:
        PiiDetector.load(excluded_id)
    assert excluded_id in str(exc_info.value)


def test_load_rejects_unregistered_when_not_allowed() -> None:
    s = KaosNLPTransformersSettings(allow_unregistered=False)
    with pytest.raises(ModelNotRegisteredError):
        PiiDetector.load("definitely/not-a-pii-model", settings=s)


# -- detect() shape contract -----------------------------------------------


def test_detect_returns_one_list_per_text() -> None:
    backend = _FakeBackend()
    det = _make_detector(backend)
    out = det.detect(["Alice walks home.", "Bob calls a friend."])
    assert len(out) == 2
    for per_text in out:
        assert all(isinstance(e, Entity) for e in per_text)


def test_detect_marshals_dicts_into_entities() -> None:
    backend = _FakeBackend()
    det = _make_detector(backend)
    [entities] = det.detect(["Alice was here today"])
    assert len(entities) == 2
    person = entities[0]
    assert person.label == "PERSON"
    assert person.start == 0
    assert person.end == 5
    assert person.score == pytest.approx(0.95)


def test_detect_forwards_threshold_to_backend() -> None:
    backend = _FakeBackend()
    det = _make_detector(backend)
    det.detect(["test input"], score_threshold=0.85)
    assert len(backend.calls) == 1
    _, threshold = backend.calls[0]
    assert threshold == pytest.approx(0.85)


def test_detect_empty_input_short_circuits() -> None:
    backend = _FakeBackend()
    det = _make_detector(backend)
    assert det.detect([]) == []
    assert backend.calls == []


def test_detect_rejects_out_of_range_threshold() -> None:
    backend = _FakeBackend()
    det = _make_detector(backend)
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            det.detect(["x"], score_threshold=bad)


def test_detect_accepts_arbitrary_sequence() -> None:
    backend = _FakeBackend()
    det = _make_detector(backend)
    out = det.detect(("first text", "second text"))
    assert len(out) == 2


def test_labels_property_returns_model_categories() -> None:
    backend = _FakeBackend()
    det = _make_detector(backend)
    assert det.labels == ["EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER"]


def test_entity_dataclass_is_shared_with_gliner() -> None:
    """Both PII and GLiNER outputs use the same `Entity` dataclass —
    downstream redaction pipelines can consume either source
    interchangeably."""
    from kaos_nlp_transformers import Entity as PiiEntity
    from kaos_nlp_transformers.ner import Entity as NerEntity

    assert PiiEntity is NerEntity
