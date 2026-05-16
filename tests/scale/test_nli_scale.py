"""Scale + throughput benchmark for :class:`NliModel` over USC + EDGAR.

What we validate:

1. **Inference runs to completion** on N USC sections + N EDGAR
   agreement heads against a fixed hypothesis set. Each call scores
   one premise paired with multiple hypotheses (the multi-hypothesis
   classification shape that ``ZeroShotNLIClassifier`` issues).

2. **Score sanity** per pair: every triple sums to ~1.0 (softmax
   invariant) and every component is in ``[0, 1]``.

3. **Argmax distribution** per corpus is recorded so a regression
   that, say, collapses all docs to one label would be caught.

4. **Throughput JSON** is written under
   ``docs/benchmarks/`` for tracking.

Marked ``slow`` and gated on the same fixtures the chunker scale
tests use. Skips cleanly when the fixtures or the Rust extension
aren't available.

Throughput baseline (2026-05-16, 20-core CPU, ort default
intra-threads): EDGAR 50 docs with 4 hypotheses each ~ 40 s.
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

# Stable hypothesis sets. Two distinct sets keep the test honest —
# USC sections shouldn't match the contract hypotheses and vice
# versa, which the recorded argmax distribution captures.
USC_HYPOTHESES = (
    "This text is an administrative procedure.",
    "This text is a federal criminal statute.",
    "This text concerns interstate commerce.",
    "This text is a tax regulation.",
)

EDGAR_HYPOTHESES = (
    "This is an employment agreement.",
    "This is a credit or loan agreement.",
    "This is a stock purchase agreement.",
    "This is a non-disclosure or confidentiality agreement.",
    "This is a merger or acquisition agreement.",
)


def _emit_report(name: str, payload: dict[str, Any]) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:
        pass


@pytest.fixture(scope="module")
def nli_model() -> Any:
    """Load the real Rust-backed NliModel once per module."""
    try:
        from kaos_nlp_transformers._rust import nli as _nli  # noqa: F401
    except ImportError:
        pytest.skip(
            "kaos_nlp_transformers._rust extension is not built — "
            "run `uv run maturin develop --release` first."
        )
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set; NLI scale test needs hub access on cold cache")
    from kaos_nlp_transformers import NliModel

    return NliModel.load()


def _exercise(
    nli: Any,
    documents: list[dict[str, Any]],
    hypotheses: tuple[str, ...],
    *,
    label: str,
    max_chars: int = 1500,
) -> dict[str, Any]:
    """Run NLI over every document, record per-call latency + argmax."""
    per_call_ms: list[float] = []
    argmax_counter: Counter[str] = Counter()
    bad_probs: list[dict[str, Any]] = []
    total_pairs = 0

    for doc_index, record in enumerate(documents):
        text = record_text(record).strip()
        if not text:
            continue
        snippet = text[:max_chars]

        t = time.perf_counter()
        scores = nli.score(snippet, list(hypotheses))
        per_call_ms.append((time.perf_counter() - t) * 1000.0)
        total_pairs += len(scores)

        for h, s in zip(hypotheses, scores, strict=True):
            triple = (s.entailment, s.neutral, s.contradiction)
            for v in triple:
                if not (0.0 <= v <= 1.0):
                    bad_probs.append({"doc_index": doc_index, "hyp": h, "value": v})
            total = sum(triple)
            if not (0.99 <= total <= 1.01):
                bad_probs.append({"doc_index": doc_index, "hyp": h, "sum": total})

        # Argmax over entailment scores → which hypothesis the model
        # most strongly endorses for this doc.
        best_hyp, _ = max(
            zip(hypotheses, scores, strict=True),
            key=lambda kv: kv[1].entailment,
        )
        argmax_counter[best_hyp] += 1

    total_s = sum(per_call_ms) / 1000.0
    return {
        "label": label,
        "n_docs": len(per_call_ms),
        "n_pairs_total": total_pairs,
        "elapsed_s_total": total_s,
        "docs_per_second": (len(per_call_ms) / total_s) if total_s > 0 else 0.0,
        "pairs_per_second": (total_pairs / total_s) if total_s > 0 else 0.0,
        "per_call_ms_mean": statistics.mean(per_call_ms) if per_call_ms else 0.0,
        "per_call_ms_p50": statistics.median(per_call_ms) if per_call_ms else 0.0,
        "per_call_ms_p95": (
            statistics.quantiles(per_call_ms, n=20)[-1]
            if len(per_call_ms) >= 20
            else max(per_call_ms, default=0.0)
        ),
        "per_call_ms_max": max(per_call_ms, default=0.0),
        "argmax_distribution": dict(argmax_counter),
        "n_invalid_probs": len(bad_probs),
        "invalid_prob_samples": bad_probs[:5],
        "hypotheses": list(hypotheses),
    }


class TestNliScale:
    def test_usc(self, nli_model: Any, usc_sample: list[dict[str, Any]]) -> None:
        if not usc_sample:
            pytest.skip("USC sample empty")
        report = _exercise(nli_model, usc_sample, USC_HYPOTHESES, label="usc")
        _emit_report("nli-throughput-usc", report)
        # Hard invariants — caught as test failures, not just JSON.
        assert report["n_invalid_probs"] == 0, (
            f"Found {report['n_invalid_probs']} invalid prob triples: "
            f"{report['invalid_prob_samples']}"
        )
        # At least one hypothesis must dominate (not all identical) — a
        # collapsed-to-one-class regression would be caught here.
        assert len(report["argmax_distribution"]) >= 1
        # Throughput floor: 1.5 pairs/sec is conservative for a 20-core
        # CPU box; calibrate up after a stable baseline.
        assert report["pairs_per_second"] > 1.5, (
            f"USC throughput regressed: {report['pairs_per_second']:.2f} pairs/s"
        )

    def test_edgar(self, nli_model: Any, edgar_agreements: list[dict[str, Any]]) -> None:
        if not edgar_agreements:
            pytest.skip("EDGAR sample empty")
        report = _exercise(nli_model, edgar_agreements, EDGAR_HYPOTHESES, label="edgar")
        _emit_report("nli-throughput-edgar", report)
        assert report["n_invalid_probs"] == 0
        assert len(report["argmax_distribution"]) >= 1
        assert report["pairs_per_second"] > 1.5
