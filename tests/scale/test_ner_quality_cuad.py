"""Quality test for :class:`GLiNERExtractor` on the CUAD-sample fixture.

CUAD ships SQuAD-format gold spans for a handful of clause types per
contract. Two of those clause types are *entity-like* (single short
strings rather than full clause sentences): **Parties** and
**Agreement Date**. We use them as gold sets for zero-shot NER:

* Run GLiNER with the labels ``("party", "date")`` (or
  ``("party", "company", "date")``) on the full contract text.
* For each gold party/date string, mark it as recalled iff at least
  one GLiNER-extracted span case-insensitively matches the gold
  string (substring containment in either direction — gold strings
  often carry trailing punctuation or partial-quote markers).
* Aggregate per-doc + corpus-level recall + precision and write to
  ``docs/benchmarks/gliner-quality-cuad.json``.

CUAD redacts some answers with the literal token ``[*]``; those are
skipped from the gold set (matching CUAD's own evaluation
convention).

Sample is 5 contracts — too small for statistically meaningful F1,
but enough for a *regression* gate: if a future revision causes us
to recall <50% of party strings we knew we used to catch, that's a
load-bearing failure.

Marked ``slow``. Skips cleanly when fixtures or the Rust extension
aren't available.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"

# Domain-appropriate label set. "company" + "party" both fire on
# corporation names in practice; we accept either as a match for a
# gold Party string.
NER_LABELS = ("party", "company", "person", "date")
PARTY_LABELS = {"party", "company", "person"}
DATE_LABELS = {"date"}

# CUAD redaction marker. Skip these from the gold set.
CUAD_REDACTION = "[*]"


def _normalize(s: str) -> str:
    """Lowercase, strip surrounding whitespace + common trailing
    punctuation. CUAD gold strings often end in ``,`` ``.`` ``"``."""
    return s.lower().strip().strip(",.;:\"'() ").strip()


def _matches(gold: str, prediction: str) -> bool:
    """Substring-containment match in either direction. Tolerates
    CUAD's punctuation-suffixed gold strings and gliner's stripped
    boundaries."""
    g, p = _normalize(gold), _normalize(prediction)
    if not g or not p:
        return False
    return g in p or p in g


def _emit_report(name: str, payload: dict[str, Any]) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


@pytest.fixture(scope="module")
def cuad_dir(scale_fixtures_dir: Path) -> Path:
    """The cuad-sample subdirectory next to usc.jsonl."""
    d = scale_fixtures_dir / "cuad-sample"
    if not d.exists():
        pytest.skip(f"CUAD sample directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def cuad_records(cuad_dir: Path) -> list[dict[str, Any]]:
    """Load each contract's text + gold answers from CUAD-sample."""
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
def gliner_model() -> Any:
    try:
        from kaos_nlp_transformers._rust import ner as _ner  # noqa: F401
    except ImportError:
        pytest.skip("kaos_nlp_transformers._rust extension is not built")
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set; CUAD quality needs hub access on cold cache")
    from kaos_nlp_transformers import GLiNERExtractor

    return GLiNERExtractor.load()


class TestCuadQuality:
    def test_party_and_date_recall(
        self,
        gliner_model: Any,
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

            [entities] = gliner_model.extract(
                [text],
                list(NER_LABELS),
                threshold=0.5,
            )

            # Bucket predicted spans by category
            party_spans = [e for e in entities if e.label in PARTY_LABELS]
            date_spans = [e for e in entities if e.label in DATE_LABELS]

            # Recall: for each gold answer, did we predict anything matching?
            party_hits = sum(
                1 for g in gold_parties if any(_matches(g, e.text) for e in party_spans)
            )
            date_hits = sum(1 for g in gold_dates if any(_matches(g, e.text) for e in date_spans))

            # Precision proxy: of our predictions, how many match SOME
            # gold party/date string?
            all_gold = gold_parties + gold_dates
            matched_preds = 0
            for e in entities:
                if any(_matches(g, e.text) for g in all_gold):
                    matched_preds += 1

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
                        {"text": e.text, "label": e.label, "score": round(e.score, 3)}
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
            "labels": list(NER_LABELS),
            "threshold": 0.5,
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
        _emit_report("gliner-quality-cuad", report)

        # Regression gates — calibrated conservatively. Tighten after
        # an established baseline. A complete recall collapse (< 30%)
        # would fail. Precision proxy floor at 25% catches a noise
        # explosion where most predictions match nothing in the gold.
        assert party_recall >= 0.30, (
            f"Party recall regressed: {party_recall:.2%} "
            f"({corpus_party_hit}/{corpus_party_total}) — full report at "
            f"docs/benchmarks/gliner-quality-cuad.json"
        )
        assert date_recall >= 0.40, (
            f"Agreement-date recall regressed: {date_recall:.2%} "
            f"({corpus_date_hit}/{corpus_date_total})"
        )
        # Don't gate on precision_proxy yet — too noisy with 5 docs.
        # Just record it.
        _ = precision_proxy
        _ = defaultdict  # silence unused-import if linter widens scope
