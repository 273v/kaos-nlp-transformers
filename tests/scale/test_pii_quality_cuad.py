"""Quality test for :class:`PiiDetector` on the CUAD-sample fixture.

The PII model emits PERSON / ORGANIZATION / DATE_TIME etc. CUAD's
``Parties`` clause gives us ground truth for "what counts as an
organization-or-person party in this contract", and ``Agreement
Date`` gives us a gold date. We use both as a quality gate:

* **Party recall**: for each gold party string (after stripping
  CUAD's ``[*]`` redactions), check whether the PII model produced
  at least one ORGANIZATION or PERSON span matching the gold text
  (case-insensitive substring containment in either direction —
  same matcher as the GLiNER CUAD quality test).
* **Date recall**: same gating on DATE_TIME spans.
* **Precision proxy**: of all PII spans the model produced, how
  many match SOME gold party/date string?

5 contracts → corpus-level precision/recall. Results written to
``docs/benchmarks/pii-quality-cuad.json``.

Marked ``slow``. Skips cleanly when fixtures or the Rust extension
aren't available.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"

# Gravitee BERT-small PII labels we'll treat as "party-ish".
# PERSON for individuals; ORGANIZATION for entities (LLCs, Corps).
# The model also has TITLE (Mr./Mrs./Dr.) which we DON'T count.
PARTY_LIKE_LABELS = {"PERSON", "ORGANIZATION"}
DATE_LIKE_LABELS = {"DATE_TIME"}

CUAD_REDACTION = "[*]"


def _normalize(s: str) -> str:
    return s.lower().strip().strip(",.;:\"'() ").strip()


def _matches(gold: str, prediction: str) -> bool:
    g, p = _normalize(gold), _normalize(prediction)
    if not g or not p:
        return False
    return g in p or p in g


def _emit_report(name: str, payload: dict[str, Any]) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:
        pass


@pytest.fixture(scope="module")
def cuad_dir(scale_fixtures_dir: Path) -> Path:
    d = scale_fixtures_dir / "cuad-sample"
    if not d.exists():
        pytest.skip(f"CUAD sample directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def cuad_records(cuad_dir: Path) -> list[dict[str, Any]]:
    golden = cuad_dir / "cuad-extraction-golden.jsonl"
    if not golden.exists():
        pytest.skip(f"CUAD golden JSONL missing: {golden}")
    records: list[dict[str, Any]] = []
    with golden.open(encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            text_path = cuad_dir / f"{row['doc_id']}.txt"
            if not text_path.exists():
                continue
            row["text"] = text_path.read_text(encoding="utf-8")
            records.append(row)
    if not records:
        pytest.skip("CUAD records empty")
    return records


@pytest.fixture(scope="module")
def pii_detector() -> Any:
    try:
        from kaos_nlp_transformers._rust import token_classify as _tc  # noqa: F401
    except ImportError:
        pytest.skip("kaos_nlp_transformers._rust extension is not built")
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set; CUAD quality needs hub access on cold cache")
    from kaos_nlp_transformers import PiiDetector

    return PiiDetector.load()


class TestPiiCuadQuality:
    def test_party_and_date_recall(
        self,
        pii_detector: Any,
        cuad_records: list[dict[str, Any]],
    ) -> None:
        per_doc: list[dict[str, Any]] = []
        corpus_party_total = 0
        corpus_party_hit = 0
        corpus_date_total = 0
        corpus_date_hit = 0
        corpus_predictions = 0
        corpus_pred_matched = 0
        t = time.perf_counter()

        for record in cuad_records:
            text = record["text"]
            doc_id = record["doc_id"]
            clauses = record.get("clause_answers", {})
            gold_parties = [s for s in clauses.get("Parties", []) if CUAD_REDACTION not in s]
            gold_dates = [s for s in clauses.get("Agreement Date", []) if CUAD_REDACTION not in s]

            [entities] = pii_detector.detect([text], score_threshold=0.5)

            party_spans = [e for e in entities if e.label in PARTY_LIKE_LABELS]
            date_spans = [e for e in entities if e.label in DATE_LIKE_LABELS]

            party_hits = sum(
                1 for g in gold_parties if any(_matches(g, e.text) for e in party_spans)
            )
            date_hits = sum(1 for g in gold_dates if any(_matches(g, e.text) for e in date_spans))

            all_gold = gold_parties + gold_dates
            matched_preds = sum(1 for e in entities if any(_matches(g, e.text) for g in all_gold))

            per_doc.append(
                {
                    "doc_id": doc_id,
                    "n_gold_parties": len(gold_parties),
                    "party_hits": party_hits,
                    "n_gold_dates": len(gold_dates),
                    "date_hits": date_hits,
                    "n_predicted": len(entities),
                    "predicted_matched": matched_preds,
                    "predicted_sample": [
                        {
                            "text": e.text,
                            "label": e.label,
                            "score": round(e.score, 3),
                        }
                        for e in entities[:10]
                    ],
                    "missed_parties": [
                        g for g in gold_parties if not any(_matches(g, e.text) for e in party_spans)
                    ],
                    "missed_dates": [
                        g for g in gold_dates if not any(_matches(g, e.text) for e in date_spans)
                    ],
                }
            )
            corpus_party_total += len(gold_parties)
            corpus_party_hit += party_hits
            corpus_date_total += len(gold_dates)
            corpus_date_hit += date_hits
            corpus_predictions += len(entities)
            corpus_pred_matched += matched_preds

        elapsed_s = time.perf_counter() - t

        party_recall = (corpus_party_hit / corpus_party_total) if corpus_party_total else 0.0
        date_recall = (corpus_date_hit / corpus_date_total) if corpus_date_total else 0.0
        precision_proxy = (corpus_pred_matched / corpus_predictions) if corpus_predictions else 0.0

        report = {
            "party_like_labels": sorted(PARTY_LIKE_LABELS),
            "date_like_labels": sorted(DATE_LIKE_LABELS),
            "score_threshold": 0.5,
            "n_docs": len(per_doc),
            "elapsed_s": elapsed_s,
            "corpus": {
                "party_gold": corpus_party_total,
                "party_hit": corpus_party_hit,
                "party_recall": party_recall,
                "date_gold": corpus_date_total,
                "date_hit": corpus_date_hit,
                "date_recall": date_recall,
                "n_predictions": corpus_predictions,
                "n_predictions_matched": corpus_pred_matched,
                "precision_proxy": precision_proxy,
            },
            "per_doc": per_doc,
        }
        _emit_report("pii-quality-cuad", report)

        # Regression gates. Calibrated conservatively; tighten after
        # a stable baseline.
        assert party_recall >= 0.30, (
            f"Party recall regressed: {party_recall:.2%} "
            f"({corpus_party_hit}/{corpus_party_total}) — full report at "
            f"docs/benchmarks/pii-quality-cuad.json"
        )
        assert date_recall >= 0.40, (
            f"Agreement-date recall regressed: {date_recall:.2%} "
            f"({corpus_date_hit}/{corpus_date_total})"
        )
        _ = precision_proxy  # recorded but not gated (too noisy at n=5)
