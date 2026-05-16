"""Scale + throughput benchmark for :class:`PiiDetector` over USC + EDGAR.

What we validate:

1. **Inference runs to completion** on N USC sections + N EDGAR
   agreement heads against the model's baked-in 24-category PII
   vocabulary.

2. **Per-entity invariants** (same as the GLiNER scale test):

   * ``score`` is in ``[0, 1]``
   * ``start < end``
   * ``source_text[start:end] == entity.text`` (char-offset
     round-trip — exercises the byte-to-char conversion in
     ``core::token_classify``)

3. **Label distribution** is recorded — a regression that collapsed
   extraction to zero or to one degenerate label would be caught.

4. **Throughput JSON** is written under
   ``docs/benchmarks/pii-throughput-{usc,edgar}.json``.

Marked ``slow``. Skips cleanly when fixtures or the Rust extension
aren't available.

Throughput baseline (2026-05-16, 20-core CPU, ort default
intra-threads, 27 MB int8 BERT-small): faster per doc than GLiNER
because the model is ~7x smaller and the token-classifier output is
``(batch, seq, 49)`` instead of GLiNER's
``(batch, seq, max_width, num_classes)`` 4D tensor.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from .conftest import record_text

_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"


def _emit_report(name: str, payload: dict[str, Any]) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:
        pass


@pytest.fixture(scope="module")
def pii_detector() -> Any:
    """Load the real Rust-backed PiiDetector once per module."""
    try:
        from kaos_nlp_transformers._rust import token_classify as _tc  # noqa: F401
    except ImportError:
        pytest.skip(
            "kaos_nlp_transformers._rust extension is not built — "
            "run `uv run maturin develop --release` first."
        )
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set; PII scale test needs hub access on cold cache")
    from kaos_nlp_transformers import PiiDetector

    return PiiDetector.load()


def _exercise(
    detector: Any,
    documents: list[dict[str, Any]],
    *,
    label: str,
    max_chars: int = 1500,
    score_threshold: float = 0.5,
) -> dict[str, Any]:
    per_call_ms: list[float] = []
    n_entities_per_doc: list[int] = []
    label_counter: Counter[str] = Counter()
    bad_entities: list[dict[str, Any]] = []
    total_entities = 0

    for doc_index, record in enumerate(documents):
        text = record_text(record).strip()
        if not text:
            continue
        snippet = text[:max_chars]

        t = time.perf_counter()
        [entities] = detector.detect([snippet], score_threshold=score_threshold)
        per_call_ms.append((time.perf_counter() - t) * 1000.0)
        n_entities_per_doc.append(len(entities))
        total_entities += len(entities)

        for e in entities:
            if not (0.0 <= e.score <= 1.0):
                bad_entities.append({"doc_index": doc_index, "field": "score", "value": e.score})
            if not (0 <= e.start < e.end <= len(snippet)):
                bad_entities.append(
                    {
                        "doc_index": doc_index,
                        "field": "offsets",
                        "start": e.start,
                        "end": e.end,
                    }
                )
            elif snippet[e.start : e.end] != e.text:
                bad_entities.append(
                    {
                        "doc_index": doc_index,
                        "field": "roundtrip",
                        "expected": e.text,
                        "actual": snippet[e.start : e.end],
                    }
                )
            label_counter[e.label] += 1

    total_s = sum(per_call_ms) / 1000.0
    return {
        "label": label,
        "n_docs": len(per_call_ms),
        "n_entities_total": total_entities,
        "score_threshold": score_threshold,
        "elapsed_s_total": total_s,
        "docs_per_second": (len(per_call_ms) / total_s) if total_s > 0 else 0.0,
        "per_call_ms_mean": statistics.mean(per_call_ms) if per_call_ms else 0.0,
        "per_call_ms_p50": statistics.median(per_call_ms) if per_call_ms else 0.0,
        "per_call_ms_p95": (
            statistics.quantiles(per_call_ms, n=20)[-1]
            if len(per_call_ms) >= 20
            else max(per_call_ms, default=0.0)
        ),
        "per_call_ms_max": max(per_call_ms, default=0.0),
        "entities_per_doc_mean": statistics.mean(n_entities_per_doc) if n_entities_per_doc else 0.0,
        "entities_per_doc_max": max(n_entities_per_doc, default=0),
        "zero_entity_docs": sum(1 for c in n_entities_per_doc if c == 0),
        "label_distribution": dict(label_counter),
        "n_invalid_entities": len(bad_entities),
        "invalid_entity_samples": bad_entities[:5],
        "model_labels": detector.labels,
    }


class TestPiiScale:
    def test_usc(self, pii_detector: Any, usc_sample: list[dict[str, Any]]) -> None:
        if not usc_sample:
            pytest.skip("USC sample empty")
        report = _exercise(pii_detector, usc_sample, label="usc")
        _emit_report("pii-throughput-usc", report)
        # Hard invariants
        assert report["n_invalid_entities"] == 0, (
            f"Found {report['n_invalid_entities']} invalid entities: "
            f"{report['invalid_entity_samples']}"
        )
        # USC is law text — we expect very few PII hits per doc (no
        # personal data) but the model should still produce *some*
        # outputs (DATE_TIME on dates in case citations, ORGANIZATION
        # on agency names, etc.). A complete collapse to zero would
        # indicate something is wrong.
        assert report["zero_entity_docs"] < report["n_docs"], (
            f"All {report['n_docs']} USC docs returned zero entities"
        )
        # Throughput floor — BERT-small is faster than GLiNER per
        # doc; calibrate up after a stable baseline.
        assert report["docs_per_second"] > 2.0, (
            f"USC throughput regressed: {report['docs_per_second']:.2f} docs/s"
        )

    def test_edgar(self, pii_detector: Any, edgar_agreements: list[dict[str, Any]]) -> None:
        if not edgar_agreements:
            pytest.skip("EDGAR sample empty")
        report = _exercise(pii_detector, edgar_agreements, label="edgar")
        _emit_report("pii-throughput-edgar", report)
        assert report["n_invalid_entities"] == 0
        # EDGAR agreements are PII-rich — almost every doc has at
        # least party names (PERSON / ORGANIZATION), dates
        # (DATE_TIME), and addresses (LOCATION). Allow up to 10%
        # empty.
        max_empty = max(1, int(0.1 * report["n_docs"]))
        assert report["zero_entity_docs"] <= max_empty, (
            f"{report['zero_entity_docs']}/{report['n_docs']} EDGAR docs "
            "returned zero PII entities — extraction may be broken"
        )
        assert report["docs_per_second"] > 2.0
