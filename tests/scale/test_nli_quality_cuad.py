"""Quality test for :class:`NliModel` on the CUAD-sample fixture.

The CUAD manifest records each contract's type in its title
(sponsorship, co-branding, outsourcing, web-site-hosting,
distributor). We use that as a small but ground-truth labeled
zero-shot classification benchmark:

* Build a hypothesis pool: one ``"This is a {type} agreement."``
  per type (5 hypotheses).
* For each contract, score the head text (first 1500 chars — NLI
  caps at 512 tokens) against every hypothesis.
* Pick the argmax over ``P(entailment)``.
* Check that the picked hypothesis names the CUAD type.

5 contracts → 5/5 expected if the NLI head behaves. We gate at
3/5 (60%) for now so a transient miss doesn't break CI; tighten
after a stable baseline is established. Results written to
``docs/benchmarks/nli-quality-cuad.json``.

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

# Map CUAD title keywords → canonical hypothesis text. Lowercase
# substring match against the title field of MANIFEST.json picks
# the gold type for each doc.
TYPE_KEYWORDS: dict[str, str] = {
    "sponsorship": "This is a sponsorship agreement.",
    "co-branding": "This is a co-branding agreement.",
    "outsourcing": "This is an outsourcing agreement.",
    "web site hosting": "This is a web site hosting agreement.",
    "distributor": "This is a distributor agreement.",
}


def _emit_report(name: str, payload: dict[str, Any]) -> None:
    if os.environ.get("KAOS_NLP_SCALE_NO_REPORT"):
        return
    try:
        _BENCH_DIR.mkdir(parents=True, exist_ok=True)
        (_BENCH_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:
        pass


def _gold_hypothesis(title: str) -> str | None:
    lower = title.lower()
    for keyword, hyp in TYPE_KEYWORDS.items():
        if keyword in lower:
            return hyp
    return None


@pytest.fixture(scope="module")
def cuad_dir(scale_fixtures_dir: Path) -> Path:
    d = scale_fixtures_dir / "cuad-sample"
    if not d.exists():
        pytest.skip(f"CUAD sample directory missing: {d}")
    return d


@pytest.fixture(scope="module")
def cuad_typed_records(cuad_dir: Path) -> list[dict[str, Any]]:
    """For each contract in MANIFEST.json, attach its full text and
    the gold-type hypothesis derived from its title."""
    manifest_path = cuad_dir / "MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip(f"CUAD MANIFEST.json missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for entry in manifest.get("contracts", []):
        title = entry.get("title", "")
        gold = _gold_hypothesis(title)
        if gold is None:
            continue  # title doesn't map to any of our 5 known types
        text_path = cuad_dir / entry["file"]
        if not text_path.exists():
            continue
        out.append(
            {
                "doc_id": entry["doc_id"],
                "title": title,
                "gold_hypothesis": gold,
                "text": text_path.read_text(encoding="utf-8"),
            }
        )
    if not out:
        pytest.skip("No CUAD records matched the known-type hypothesis pool")
    return out


@pytest.fixture(scope="module")
def nli_model() -> Any:
    try:
        from kaos_nlp_transformers._rust import nli as _nli  # noqa: F401
    except ImportError:
        pytest.skip("kaos_nlp_transformers._rust extension is not built")
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set; CUAD quality needs hub access on cold cache")
    from kaos_nlp_transformers import NliModel

    return NliModel.load()


class TestCuadNliQuality:
    def test_argmax_picks_correct_type(
        self,
        nli_model: Any,
        cuad_typed_records: list[dict[str, Any]],
    ) -> None:
        hypotheses = list(TYPE_KEYWORDS.values())
        per_doc: list[dict[str, Any]] = []
        correct = 0
        t = time.perf_counter()

        for record in cuad_typed_records:
            head = record["text"][:1500]
            scores = nli_model.score(head, hypotheses)
            ranked = sorted(
                zip(hypotheses, scores, strict=True),
                key=lambda kv: kv[1].entailment,
                reverse=True,
            )
            top_hyp, top_score = ranked[0]
            gold = record["gold_hypothesis"]
            is_correct = top_hyp == gold
            if is_correct:
                correct += 1
            per_doc.append(
                {
                    "doc_id": record["doc_id"],
                    "title": record["title"],
                    "gold_hypothesis": gold,
                    "predicted_hypothesis": top_hyp,
                    "predicted_entailment": round(top_score.entailment, 4),
                    "correct": is_correct,
                    "all_scores": [
                        {
                            "hyp": h,
                            "entail": round(s.entailment, 4),
                            "neutral": round(s.neutral, 4),
                            "contradict": round(s.contradiction, 4),
                        }
                        for h, s in ranked
                    ],
                }
            )

        elapsed_s = time.perf_counter() - t
        accuracy = correct / len(per_doc) if per_doc else 0.0

        report = {
            "hypotheses": hypotheses,
            "n_docs": len(per_doc),
            "n_correct": correct,
            "accuracy": accuracy,
            "elapsed_s": elapsed_s,
            "per_doc": per_doc,
        }
        _emit_report("nli-quality-cuad", report)

        # Regression gate. 3/5 is a low floor — CUAD titles are
        # explicit ("SPONSORSHIP AGREEMENT" etc.), so the model
        # should clear this comfortably. Tighten to 4/5 or 5/5
        # after a stable baseline.
        assert correct >= 3, (
            f"NLI CUAD type-classification regressed: {correct}/{len(per_doc)} "
            f"= {accuracy:.0%}. See docs/benchmarks/nli-quality-cuad.json"
        )
