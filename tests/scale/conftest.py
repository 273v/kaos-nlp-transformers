"""Shared scale-test fixtures for ``kaos-nlp-transformers``.

Mirrors ``kaos-nlp-core/tests/scale/conftest.py``: same fixture
files, same env var, same monorepo fallback. Tests in this directory
are marked ``slow`` and skipped cleanly when the HF fixtures aren't
available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow

_FIXTURE_FILES = ("usc.jsonl", "edgar_agreements.jsonl", "patents.jsonl")


def _resolve_fixtures_dir() -> Path | None:
    env = os.environ.get("KAOS_NLP_SCALE_FIXTURES")
    if env:
        path = Path(env).expanduser().resolve()
        if all((path / name).exists() for name in _FIXTURE_FILES):
            return path

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "kaos-modules" / "kaos-nlp-core" / "tests" / "fixtures"
        if all((candidate / name).exists() for name in _FIXTURE_FILES):
            return candidate

    return None


@pytest.fixture(scope="session")
def scale_fixtures_dir() -> Path:
    path = _resolve_fixtures_dir()
    if path is None:
        pytest.skip(
            "Scale fixtures not available. Set KAOS_NLP_SCALE_FIXTURES "
            "or run kaos-nlp-core/tests/fixtures/download_hf_fixtures.py."
        )
    return path


def _load_jsonl(path: Path, *, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if max_records is not None and index >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _record_text(record: dict[str, Any]) -> str:
    text = record.get("text") or record.get("content") or record.get("body") or ""
    return str(text)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Smaller defaults than the deterministic chunker tier: embedding
# inference is the cost driver, even with model2vec.
USC_SAMPLE_SIZE = _int_env("KAOS_NLP_SCALE_USC_SAMPLE", 200)
EDGAR_SAMPLE_SIZE = _int_env("KAOS_NLP_SCALE_EDGAR_SAMPLE", 50)
PATENTS_SAMPLE_SIZE = _int_env("KAOS_NLP_SCALE_PATENTS_SAMPLE", 50)


@pytest.fixture(scope="session")
def usc_sample(scale_fixtures_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(scale_fixtures_dir / "usc.jsonl", max_records=USC_SAMPLE_SIZE)


@pytest.fixture(scope="session")
def edgar_agreements(scale_fixtures_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(
        scale_fixtures_dir / "edgar_agreements.jsonl",
        max_records=EDGAR_SAMPLE_SIZE,
    )


@pytest.fixture(scope="session")
def patents(scale_fixtures_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(
        scale_fixtures_dir / "patents.jsonl",
        max_records=PATENTS_SAMPLE_SIZE,
    )


# ---------------------------------------------------------------------------
# Real local embedder (model2vec / potion-base-8M).
#
# The model is vendored inside the wheel at
# ``kaos_nlp_transformers/_vendor/potion-base-8M`` — no network access
# required to load it.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def local_embedder() -> Any:
    """Load the vendored model2vec embedder.

    Skips the test if model2vec or the vendored model aren't
    available — we never silently substitute a stub.
    """
    pytest.importorskip("model2vec", reason="model2vec extra not installed")
    from kaos_nlp_transformers import EmbeddingModel

    try:
        return EmbeddingModel.load("minishlab/potion-base-8M")
    except Exception as exc:  # pragma: no cover - hardware/license issues
        pytest.skip(f"Could not load local potion-base-8M embedder: {exc}")


record_text = _record_text
load_jsonl = _load_jsonl
