"""Unit tests for :class:`kaos_nlp_transformers.ner.GLiNERExtractor`.

Offline-friendly: uses an in-process fake backend that mirrors the
Rust ``NerBackend.extract`` dict contract, so the registry gating,
Entity construction, and error paths are covered without downloading
any model. Live behavior is exercised separately in
``tests/integration/test_ner_live.py``.
"""

from __future__ import annotations

import pytest

from kaos_nlp_transformers import NER_EXCLUDED, NER_REGISTRY, Entity, GLiNERExtractor
from kaos_nlp_transformers.errors import ModelNotRegisteredError
from kaos_nlp_transformers.ner import DEFAULT_NER_MODEL
from kaos_nlp_transformers.settings import KaosNLPTransformersSettings


class _FakeBackend:
    """In-process stand-in for ``_rust.ner.NerBackend``.

    Returns a deterministic list-of-dicts response matching the Rust
    contract: one outer list per input text, inner items are dicts
    with ``start``, ``end``, ``text``, ``label``, ``score`` keys.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str], dict]] = []
        self.model_id = "onnx-community/gliner_medium-v2.1"
        self.device = "cpu"

    def extract(
        self,
        texts: list[str],
        labels: list[str],
        *,
        threshold: float = 0.5,
        max_width: int = 12,
        flat_ner: bool = True,
        dup_label: bool = False,
        multi_label: bool = False,
    ) -> list[list[dict]]:
        self.calls.append(
            (
                list(texts),
                list(labels),
                {
                    "threshold": threshold,
                    "max_width": max_width,
                    "flat_ner": flat_ner,
                    "dup_label": dup_label,
                    "multi_label": multi_label,
                },
            )
        )
        # Fixed response: 1 entity per input — a "person" span on the
        # first 5 characters with score 0.9.
        out: list[list[dict]] = []
        for t in texts:
            head = t[:5]
            out.append(
                [
                    {
                        "start": 0,
                        "end": len(head),
                        "text": head,
                        "label": labels[0],
                        "score": 0.9,
                    }
                ]
            )
        return out


def _make_extractor(backend: _FakeBackend) -> GLiNERExtractor:
    return GLiNERExtractor(backend, model_id=backend.model_id, device=None)


# -- Registry / load gating ------------------------------------------------


def test_default_ner_model_matches_registry() -> None:
    assert DEFAULT_NER_MODEL in NER_REGISTRY


def test_load_rejects_excluded_model() -> None:
    excluded_id = next(iter(NER_EXCLUDED))
    with pytest.raises(ModelNotRegisteredError) as exc_info:
        GLiNERExtractor.load(excluded_id)
    assert "excluded" in str(exc_info.value).lower()
    assert excluded_id in str(exc_info.value)


def test_load_rejects_unregistered_when_not_allowed() -> None:
    s = KaosNLPTransformersSettings(allow_unregistered=False)
    with pytest.raises(ModelNotRegisteredError) as exc_info:
        GLiNERExtractor.load("definitely/not-a-ner-model", settings=s)
    assert "not in the v0 registry" in str(exc_info.value)


# -- Extract shape / Entity construction ------------------------------------


def test_extract_returns_list_per_text() -> None:
    backend = _FakeBackend()
    extractor = _make_extractor(backend)

    out = extractor.extract(["Alice walks home.", "Bob is great."], ["person"])
    assert isinstance(out, list)
    assert len(out) == 2
    for per_text in out:
        assert isinstance(per_text, list)


def test_extract_marshals_dicts_into_entities() -> None:
    backend = _FakeBackend()
    extractor = _make_extractor(backend)

    [entities] = extractor.extract(["Alice was here."], ["person"])
    assert len(entities) == 1
    e = entities[0]
    assert isinstance(e, Entity)
    assert e.start == 0
    assert e.end == 5
    assert e.text == "Alice"
    assert e.label == "person"
    assert e.score == pytest.approx(0.9)


def test_extract_forwards_keyword_params_to_backend() -> None:
    backend = _FakeBackend()
    extractor = _make_extractor(backend)

    extractor.extract(
        ["foo bar"],
        ["thing"],
        threshold=0.3,
        max_width=5,
        flat_ner=False,
        dup_label=True,
        multi_label=True,
    )
    assert len(backend.calls) == 1
    _, _, params = backend.calls[0]
    assert params == {
        "threshold": 0.3,
        "max_width": 5,
        "flat_ner": False,
        "dup_label": True,
        "multi_label": True,
    }


def test_extract_empty_texts_short_circuits() -> None:
    backend = _FakeBackend()
    extractor = _make_extractor(backend)

    out = extractor.extract([], ["person"])
    assert out == []
    assert backend.calls == []


def test_extract_rejects_empty_labels() -> None:
    backend = _FakeBackend()
    extractor = _make_extractor(backend)

    with pytest.raises(ValueError) as exc_info:
        extractor.extract(["text"], [])
    assert "labels" in str(exc_info.value).lower()


def test_extract_accepts_arbitrary_sequence_types() -> None:
    """Sequence[str] in the type signature means tuples should work."""
    backend = _FakeBackend()
    extractor = _make_extractor(backend)

    out = extractor.extract(("a one", "a two"), ("person",))
    assert len(out) == 2
