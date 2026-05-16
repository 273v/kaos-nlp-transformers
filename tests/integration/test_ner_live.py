"""Live integration tests for ``GLiNERExtractor``.

Hits a REAL Rust ``NerBackend`` (ort + libonnxruntime,
onnx-community/gliner_medium-v2.1 fp32). Verifies that the GLiNER
prompt-based span extraction returns the expected high-confidence
spans on textbook inputs.

Skips when ``KAOS_NLP_TRANSFORMERS_OFFLINE=1`` or when the Rust
extension hasn't been built.

Marked ``@pytest.mark.integration`` and ``@pytest.mark.live`` (network).
"""

from __future__ import annotations

import os

import pytest

from kaos_nlp_transformers import Entity, GLiNERExtractor

pytestmark = [pytest.mark.integration, pytest.mark.live]


def _skip_if_offline() -> None:
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set")


def _skip_if_no_rust_extension() -> None:
    try:
        from kaos_nlp_transformers._rust import ner as _ner  # noqa: F401
    except ImportError:
        pytest.skip(
            "kaos_nlp_transformers._rust extension is not built — "
            "run `uv run maturin develop --release` first."
        )


@pytest.fixture(scope="module")
def extractor() -> GLiNERExtractor:
    """Module-scoped GLiNER extractor so the ~746 MiB fp32 ONNX
    downloads once."""
    _skip_if_offline()
    _skip_if_no_rust_extension()

    return GLiNERExtractor.load()  # default = onnx-community/gliner_medium-v2.1


def test_load_returns_real_extractor(extractor: GLiNERExtractor) -> None:
    assert isinstance(extractor, GLiNERExtractor)
    assert extractor.model_id == "onnx-community/gliner_medium-v2.1"


def test_load_uses_rust_ner_backend(extractor: GLiNERExtractor) -> None:
    from kaos_nlp_transformers._rust import ner as _ner

    assert isinstance(extractor._backend, _ner.NerBackend)


def test_extract_finds_person_and_place(extractor: GLiNERExtractor) -> None:
    """Headline correctness check: 'Barack Obama' is a person,
    'Hawaii' is a place. If this regresses, the inference path is
    broken."""
    [entities] = extractor.extract(
        ["Barack Obama was born in Hawaii."],
        labels=["person", "place"],
    )
    labels_by_text = {e.text: e.label for e in entities}
    assert labels_by_text.get("Barack Obama") == "person", (
        f"missing Barack Obama as person: {entities}"
    )
    assert labels_by_text.get("Hawaii") == "place", f"missing Hawaii as place: {entities}"


def test_extract_scores_in_unit_interval(extractor: GLiNERExtractor) -> None:
    [entities] = extractor.extract(
        ["Barack Obama was born in Hawaii."],
        labels=["person", "place"],
    )
    for e in entities:
        assert 0.0 <= e.score <= 1.0, f"score {e.score} outside [0, 1]"


def test_extract_offsets_roundtrip_to_substring(extractor: GLiNERExtractor) -> None:
    """``text[start:end]`` must reproduce ``entity.text`` exactly on
    pure-ASCII input."""
    src = "Barack Obama was born in Hawaii."
    [entities] = extractor.extract([src], labels=["person", "place"])
    for e in entities:
        assert src[e.start : e.end] == e.text, (
            f"offset mismatch: text[{e.start}:{e.end}] != {e.text!r}"
        )


def test_extract_offsets_roundtrip_on_multibyte_text(extractor: GLiNERExtractor) -> None:
    """Multi-byte characters (curly quotes, em-dashes) must not break
    the codepoint-offset round-trip. KNT-NLI-003: the Rust core was
    initially emitting byte offsets, which broke Python char-indexed
    slicing on any contract containing typographic punctuation."""
    src = (
        "On October 7, 2021, “Galera Therapeutics, Inc.” "
        "— a Delaware corporation — entered into this agreement."
    )
    [entities] = extractor.extract([src], labels=["company", "date", "jurisdiction"])
    assert len(entities) >= 1
    for e in entities:
        assert src[e.start : e.end] == e.text, (
            f"offset mismatch on multibyte input: "
            f"text[{e.start}:{e.end}]={src[e.start : e.end]!r} != {e.text!r}"
        )


def test_extract_batches_independent_texts(extractor: GLiNERExtractor) -> None:
    out = extractor.extract(
        [
            "Barack Obama was born in Hawaii.",
            "Acme Corporation announced earnings.",
        ],
        labels=["person", "place", "organization"],
    )
    assert len(out) == 2
    person_in_first = any(e.label == "person" for e in out[0])
    org_in_second = any(e.label == "organization" for e in out[1])
    assert person_in_first, f"no person in input 0: {out[0]}"
    assert org_in_second, f"no organization in input 1: {out[1]}"


def test_extract_empty_texts_returns_empty(extractor: GLiNERExtractor) -> None:
    assert extractor.extract([], labels=["person"]) == []


def test_extract_respects_threshold(extractor: GLiNERExtractor) -> None:
    """Higher threshold should reduce or preserve span count, never
    inflate it."""
    src = "Barack Obama was born in Hawaii."
    low = extractor.extract([src], labels=["person", "place"], threshold=0.1)[0]
    high = extractor.extract([src], labels=["person", "place"], threshold=0.95)[0]
    assert len(high) <= len(low)


def test_extract_entity_is_dataclass_instance(extractor: GLiNERExtractor) -> None:
    [entities] = extractor.extract(["John works at Acme."], labels=["person", "organization"])
    for e in entities:
        assert isinstance(e, Entity)
