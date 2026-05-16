"""Unit tests for ``kaos-nlp-transformers prefetch``.

Offline-only — we use the ``--dry-run`` path for argument parsing
and resolution coverage, and monkeypatch ``_prefetch_one`` for the
real-execution path so no network or model load happens.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import Any

import pytest

from kaos_nlp_transformers.cli import (
    _prefetch_one,
    main,
    prefetch_models,
)


def _run(argv: list[str]) -> tuple[int, dict[str, Any] | str]:
    """Invoke ``main`` with ``--json`` and parse the envelope.

    Returns ``(exit_code, parsed_json_or_raw_text)``.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    out = buf.getvalue()
    try:
        return code, json.loads(out)
    except json.JSONDecodeError:
        return code, out


def test_dry_run_lists_every_registered_model() -> None:
    code, payload = _run(["prefetch", "--dry-run", "--json"])
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["command"] == "prefetch"
    assert payload["dry_run"] is True
    families = {m["family"] for m in payload["models"]}
    # All five families must be covered by default (PII added in 0.2.0a8 work).
    assert families == {"embedding", "reranker", "nli", "ner", "pii"}
    assert payload["n_planned"] == len(payload["models"])
    for entry in payload["models"]:
        assert entry["status"] == "planned"


def test_dry_run_with_include_filter() -> None:
    code, payload = _run(
        ["prefetch", "--include", "nli", "--include", "ner", "--dry-run", "--json"]
    )
    assert code == 0
    assert isinstance(payload, dict)
    families = {m["family"] for m in payload["models"]}
    assert families == {"nli", "ner"}


def test_dry_run_with_explicit_model_overrides_include() -> None:
    code, payload = _run(
        [
            "prefetch",
            "--include",
            "embedding",  # should be IGNORED when --model is given
            "--model",
            "Xenova/nli-deberta-v3-base",
            "--dry-run",
            "--json",
        ]
    )
    assert code == 0
    assert isinstance(payload, dict)
    assert len(payload["models"]) == 1
    assert payload["models"][0]["family"] == "nli"
    assert payload["models"][0]["model_id"] == "Xenova/nli-deberta-v3-base"


def test_unknown_model_rejected_by_default() -> None:
    """Without ``--allow-unregistered``, an unknown model id raises
    a clean SystemExit before any load attempt."""
    with pytest.raises(SystemExit) as exc:
        main(["prefetch", "--model", "definitely/not-real", "--dry-run", "--json"])
    msg = str(exc.value)
    assert "definitely/not-real" in msg
    assert "registry" in msg.lower()


def test_real_path_with_mocked_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the non-dry-run path without hitting the network.

    Monkeypatch ``_prefetch_one`` to return a synthetic OK result —
    we're testing the CLI plumbing (envelope shape, summary line,
    exit code), not the loader itself."""
    calls: list[tuple[str, str]] = []

    def fake_prefetch_one(
        family: str,
        model_id: str,
        *,
        cache_dir: Any = None,
        allow_unregistered: bool = False,
    ) -> dict[str, Any]:
        calls.append((family, model_id))
        return {
            "family": family,
            "model_id": model_id,
            "status": "ok",
            "elapsed_s": 0.001,
            "error": None,
        }

    monkeypatch.setattr("kaos_nlp_transformers.cli._prefetch_one", fake_prefetch_one)
    code, payload = _run(["prefetch", "--include", "nli", "--json"])
    assert code == 0
    assert isinstance(payload, dict)
    assert payload["n_ok"] == 1
    assert payload["n_failed"] == 0
    # Cache-delta is always present even on a no-op run.
    assert "cache_delta_mb" in payload
    # Loader was called once for NLI only.
    assert calls == [("nli", "Xenova/nli-deberta-v3-base")]


def test_failure_propagates_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "family": kwargs.get("family") or args[0],
            "model_id": kwargs.get("model_id") or args[1],
            "status": "failed",
            "elapsed_s": 0.001,
            "error": "ModelLoadError: simulated",
        }

    monkeypatch.setattr("kaos_nlp_transformers.cli._prefetch_one", boom)
    code, payload = _run(["prefetch", "--include", "nli", "--json"])
    assert code == 2  # non-zero on failure
    assert isinstance(payload, dict)
    assert payload["n_failed"] == 1
    assert payload["models"][0]["status"] == "failed"


def test_programmatic_prefetch_models_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``prefetch_models()`` function is the programmatic
    counterpart of the CLI subcommand. Smoke-test the dispatch path
    with the loader mocked out."""

    def fake_prefetch_one(
        family: str,
        model_id: str,
        *,
        cache_dir: Any = None,
        allow_unregistered: bool = False,
    ) -> dict[str, Any]:
        return {
            "family": family,
            "model_id": model_id,
            "status": "ok",
            "elapsed_s": 0.0,
            "error": None,
        }

    monkeypatch.setattr("kaos_nlp_transformers.cli._prefetch_one", fake_prefetch_one)
    envelope = prefetch_models(families=["nli"])
    assert envelope["n_ok"] == 1
    assert envelope["n_failed"] == 0
    assert envelope["models"][0]["model_id"] == "Xenova/nli-deberta-v3-base"


def test_prefetch_one_records_failure_on_unknown_family() -> None:
    """_prefetch_one catches the ValueError from _resolve_loader so a
    single bad row doesn't sink the whole prefetch run. The error
    surfaces in the returned envelope row."""
    result = _prefetch_one(
        "not_a_family",
        "x/y",
        cache_dir=None,
        allow_unregistered=False,
    )
    assert result["status"] == "failed"
    assert "unknown family" in (result["error"] or "")
