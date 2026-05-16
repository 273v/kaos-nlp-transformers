"""CLI for kaos-nlp-transformers.

v0 shipped ``info``, the diagnostic envelope (same JSON shape as the
``kaos-nlp-transformers-info`` MCP tool).

0.2.0a7 adds ``prefetch`` — downloads every registered model into the
HF Hub cache so the first inference call doesn't pay the full
network cost. Useful in Dockerfile builds, CI cache-warming jobs,
and air-gapped image preparation. See
``kaos-nlp-transformers prefetch --help`` for filtering.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid forcing model imports on `--help`
    from collections.abc import Iterable


# Mapping of CLI ``--include`` family name → (registry dict, loader callable).
# Resolved lazily inside ``prefetch`` so importing the cli module doesn't
# touch the Rust extension.
_FAMILIES = ("embedding", "reranker", "nli", "ner", "pii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kaos-nlp-transformers",
        description="Dense embeddings and small-model inference for KAOS",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)
    info = sub.add_parser("info", help="Show settings, registered models, devices")
    info.add_argument("--json", action="store_true", help="Emit JSON envelope")

    pre = sub.add_parser(
        "prefetch",
        help="Download every registered model into the HF Hub cache",
        description=(
            "Walk the embedding / reranker / NLI / NER / PII registries "
            "and call .load() for each entry, populating the HF Hub cache "
            "so first inference is no-network. Honors HF_HOME / "
            "KAOS_NLP_TRANSFORMERS_CACHE_DIR. Skips models in the EXCLUDED "
            "lists. Exits non-zero on any failure (exit 1 on argument "
            "errors, exit 2 on partial download failures)."
        ),
    )
    pre.add_argument(
        "--include",
        action="append",
        choices=list(_FAMILIES),
        # No metavar — letting argparse render the choices `{a,b,c,...}`
        # in the help output so users can see the family vocabulary
        # without reading docs.
        help=(
            "Limit prefetch to one or more model families "
            "(repeatable). Default: every registered family. "
            "Ignored when --model is given."
        ),
    )
    pre.add_argument(
        "--model",
        action="append",
        metavar="MODEL_ID",
        help=(
            "Prefetch only specified model id(s) (repeatable). Resolves "
            "against every registry; rejects unregistered ids unless "
            "--allow-unregistered is set."
        ),
    )
    pre.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Override the HF Hub cache directory for this run "
            "(equivalent to KAOS_NLP_TRANSFORMERS_CACHE_DIR)."
        ),
    )
    pre.add_argument(
        "--allow-unregistered",
        action="store_true",
        help="Permit --model ids that are not in any registry.",
    )
    pre.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without downloading.",
    )
    pre.add_argument(
        "--json",
        action="store_true",
        help="Emit a JSON envelope instead of human text.",
    )
    pre.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help=(
            "Suppress per-model progress lines; only print the final "
            "summary (or JSON envelope when --json is also set). Useful "
            "in CI logs."
        ),
    )

    args = parser.parse_args(argv)

    if args.cmd is None or args.cmd == "info":
        return _cmd_info(args)
    if args.cmd == "prefetch":
        return _cmd_prefetch(args)

    parser.print_help()
    return 1


# ---------------------------------------------------------------------------
# info subcommand (preserved verbatim from 0.2.0a6 plus NLI/NER additions)
# ---------------------------------------------------------------------------


def _cmd_info(args: argparse.Namespace) -> int:
    from kaos_nlp_transformers import __version__
    from kaos_nlp_transformers.device import get_system_devices, resolve_device
    from kaos_nlp_transformers.models import (
        EXCLUDED,
        NER_EXCLUDED,
        NER_REGISTRY,
        NLI_EXCLUDED,
        NLI_REGISTRY,
        PII_EXCLUDED,
        PII_REGISTRY,
        REGISTRY,
        RERANKER_EXCLUDED,
        RERANKER_REGISTRY,
    )
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    settings = KaosNLPTransformersSettings()
    system = get_system_devices()
    chosen = resolve_device(settings.device, system)

    payload = {
        "command": "info",
        "package": "kaos-nlp-transformers",
        "version": __version__,
        "settings": {
            "default_model": settings.default_model,
            "default_reranker_model": settings.default_reranker_model,
            "default_nli_model": settings.default_nli_model,
            "default_ner_model": settings.default_ner_model,
            "device": settings.device,
            "backend": settings.backend,
            "offline": settings.offline,
            "allow_unregistered": settings.allow_unregistered,
            "profile": settings.profile,
            "cache_dir": str(settings.cache_dir) if settings.cache_dir else None,
            "workspace_root": (str(settings.workspace_root) if settings.workspace_root else None),
        },
        "resolved_device": {
            "name": chosen.name,
            "device": chosen.device,
            "backend": chosen.backend,
            "memory_mb": chosen.memory_mb,
        },
        "reachable_devices": [
            {
                "name": d.name,
                "device": d.device,
                "backend": d.backend,
                "memory_mb": d.memory_mb,
            }
            for d in system.devices
        ],
        "latent_devices": [
            {
                "name": d.name,
                "kind": d.kind,
                "reason": d.reason,
                "install_extra": d.install_extra,
                "install_hint": (
                    f"pip install kaos-nlp-transformers[{d.install_extra}]"
                    if d.install_extra
                    else None
                ),
                "detail": d.detail,
            }
            for d in system.latent_devices
        ],
        "onnx_providers": list(system.onnx_providers),
        "embedding_models": {
            "registered": sorted(REGISTRY.keys()),
            "excluded": [{"model_id": k, "reason": v} for k, v in sorted(EXCLUDED.items())],
        },
        "reranker_models": {
            "registered": sorted(RERANKER_REGISTRY.keys()),
            "excluded": [
                {"model_id": k, "reason": v} for k, v in sorted(RERANKER_EXCLUDED.items())
            ],
        },
        "nli_models": {
            "registered": sorted(NLI_REGISTRY.keys()),
            "excluded": [{"model_id": k, "reason": v} for k, v in sorted(NLI_EXCLUDED.items())],
        },
        "ner_models": {
            "registered": sorted(NER_REGISTRY.keys()),
            "excluded": [{"model_id": k, "reason": v} for k, v in sorted(NER_EXCLUDED.items())],
        },
        "pii_models": {
            "registered": sorted(PII_REGISTRY.keys()),
            "excluded": [{"model_id": k, "reason": v} for k, v in sorted(PII_EXCLUDED.items())],
        },
    }

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2))
    else:
        _print_human(payload)
    return 0


# ---------------------------------------------------------------------------
# prefetch subcommand
# ---------------------------------------------------------------------------


def _cmd_prefetch(args: argparse.Namespace) -> int:
    # Lazy imports — the prefetch path is the most expensive entry point,
    # but the CLI's `--help` should never trigger them.
    from kaos_nlp_transformers.models import (
        NER_REGISTRY,
        NLI_REGISTRY,
        PII_REGISTRY,
        REGISTRY,
        RERANKER_REGISTRY,
    )
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    settings = KaosNLPTransformersSettings()
    cache_dir = args.cache_dir or settings.cache_dir
    target_dir = _resolved_cache_dir(cache_dir)

    # Resolve the set of (family, model_id) jobs to run.
    if args.model:
        jobs = _resolve_explicit_models(
            args.model,
            allow_unregistered=args.allow_unregistered,
        )
    else:
        families = args.include or list(_FAMILIES)
        jobs = []
        if "embedding" in families:
            jobs.extend(("embedding", mid) for mid in sorted(REGISTRY.keys()))
        if "reranker" in families:
            jobs.extend(("reranker", mid) for mid in sorted(RERANKER_REGISTRY.keys()))
        if "nli" in families:
            jobs.extend(("nli", mid) for mid in sorted(NLI_REGISTRY.keys()))
        if "ner" in families:
            jobs.extend(("ner", mid) for mid in sorted(NER_REGISTRY.keys()))
        if "pii" in families:
            jobs.extend(("pii", mid) for mid in sorted(PII_REGISTRY.keys()))

    if not jobs:
        print("nothing to prefetch (empty family / model selection)", file=sys.stderr)
        return 1

    envelope: dict[str, Any] = {
        "command": "prefetch",
        "cache_dir": str(target_dir) if target_dir else None,
        "dry_run": args.dry_run,
        "n_planned": len(jobs),
        "models": [],
    }

    if args.dry_run:
        for family, model_id in jobs:
            envelope["models"].append({"family": family, "model_id": model_id, "status": "planned"})
        if args.json:
            _emit_prefetch(envelope, as_json=True)
        else:
            print(
                f"Dry run — would fetch {len(jobs)} model(s) into "
                f"{target_dir or '(HF default cache)'}:"
            )
            for entry in envelope["models"]:
                print(f"  - {entry['family']:<10} {entry['model_id']}")
        return 0

    # Whether to emit per-model progress lines on stdout. JSON mode
    # is silent-until-end (the envelope is the output); --quiet
    # suppresses per-model output in human mode.
    show_progress = not args.json and not args.quiet

    if show_progress:
        print(
            f"Prefetching {len(jobs)} model(s) into {target_dir or '(HF default cache)'}",
            flush=True,
        )

    n_ok = 0
    n_failed = 0
    total_start = time.perf_counter()
    cache_size_before = _measure_cache_size_mb(target_dir)

    for i, (family, model_id) in enumerate(jobs, start=1):
        if show_progress:
            # Print the "...starting" line BEFORE the work starts so
            # users see something during cold downloads (~minutes for
            # the multi-GB GLiNER + Gemma-sized variants). The "OK"
            # line replaces this once the model finishes.
            print(
                f"  [{i:>2}/{len(jobs)}] fetching  {family:<10} {model_id} ...",
                flush=True,
            )
        result = _prefetch_one(
            family,
            model_id,
            cache_dir=cache_dir,
            allow_unregistered=args.allow_unregistered,
        )
        envelope["models"].append(result)
        if result["status"] == "ok":
            n_ok += 1
        else:
            n_failed += 1
        if show_progress:
            _print_prefetch_line(result)

    cache_size_after = _measure_cache_size_mb(target_dir)
    envelope["elapsed_s"] = time.perf_counter() - total_start
    envelope["n_ok"] = n_ok
    envelope["n_failed"] = n_failed
    envelope["cache_size_before_mb"] = cache_size_before
    envelope["cache_size_after_mb"] = cache_size_after
    envelope["cache_delta_mb"] = (
        round(cache_size_after - cache_size_before, 1)
        if cache_size_before is not None and cache_size_after is not None
        else None
    )

    _emit_prefetch(envelope, as_json=args.json)
    return 0 if n_failed == 0 else 2


def _resolved_cache_dir(explicit: Path | None) -> Path | None:
    """Best-guess at where hf-hub will write — used only for the
    cache-size summary. Never mutated."""
    import os

    if explicit:
        return Path(explicit).expanduser().resolve()
    for env_var in ("HF_HUB_CACHE", "HF_HOME"):
        val = os.environ.get(env_var)
        if val:
            base = Path(val).expanduser().resolve()
            # hf-hub appends ``/hub`` if HF_HOME is given.
            if env_var == "HF_HOME":
                return base / "hub"
            return base
    home = Path.home()
    return home / ".cache" / "huggingface" / "hub"


def _resolve_explicit_models(
    model_ids: list[str], *, allow_unregistered: bool
) -> list[tuple[str, str]]:
    from kaos_nlp_transformers.models import (
        NER_REGISTRY,
        NLI_REGISTRY,
        PII_REGISTRY,
        REGISTRY,
        RERANKER_REGISTRY,
    )

    family_for: dict[str, str] = {}
    for k in REGISTRY:
        family_for[k] = "embedding"
    for k in RERANKER_REGISTRY:
        family_for[k] = "reranker"
    for k in NLI_REGISTRY:
        family_for[k] = "nli"
    for k in NER_REGISTRY:
        family_for[k] = "ner"
    for k in PII_REGISTRY:
        family_for[k] = "pii"

    jobs: list[tuple[str, str]] = []
    for mid in model_ids:
        fam = family_for.get(mid)
        if fam is not None:
            jobs.append((fam, mid))
        elif allow_unregistered:
            # Can't infer the family — bail with a clear error.
            raise SystemExit(
                f"--model {mid!r}: --allow-unregistered does not help here "
                "because the prefetch needs to know which family to load it "
                "as. Pass a registered id."
            )
        else:
            raise SystemExit(
                f"--model {mid!r} is not in any registry. Choices: "
                + ", ".join(sorted(family_for.keys()))
            )
    return jobs


def _prefetch_one(
    family: str,
    model_id: str,
    *,
    cache_dir: Path | None,
    allow_unregistered: bool,
) -> dict[str, Any]:
    """Load one model. Returns a JSON-shaped result row."""
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    settings = KaosNLPTransformersSettings(
        cache_dir=cache_dir,
        allow_unregistered=allow_unregistered,
    )

    start = time.perf_counter()
    err: str | None = None
    try:
        loader = _resolve_loader(family)
        loader(model_id=model_id, settings=settings)
    except Exception as exc:
        # Broad catch is deliberate: the CLI surfaces the exception
        # text rather than crashing the whole prefetch run on one bad
        # model. The envelope's ``error`` field carries the typed
        # name so a caller can re-classify.
        err = f"{type(exc).__name__}: {exc}"

    return {
        "family": family,
        "model_id": model_id,
        "status": "ok" if err is None else "failed",
        "elapsed_s": time.perf_counter() - start,
        "error": err,
    }


def _resolve_loader(family: str):
    if family == "embedding":
        from kaos_nlp_transformers import EmbeddingModel

        return EmbeddingModel.load
    if family == "reranker":
        from kaos_nlp_transformers import CrossEncoderReranker

        return CrossEncoderReranker.load
    if family == "nli":
        from kaos_nlp_transformers import NliModel

        return NliModel.load
    if family == "ner":
        from kaos_nlp_transformers import GLiNERExtractor

        return GLiNERExtractor.load
    if family == "pii":
        from kaos_nlp_transformers import PiiDetector

        return PiiDetector.load
    raise ValueError(f"unknown family {family!r}")


def _measure_cache_size_mb(target_dir: Path | None) -> float | None:
    if target_dir is None or not target_dir.exists():
        return None
    total = 0
    for p in target_dir.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return round(total / (1024 * 1024), 1)


def _emit_prefetch(envelope: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(envelope, indent=2))
        return
    n_ok = envelope.get("n_ok", 0)
    n_failed = envelope.get("n_failed", 0)
    elapsed = envelope.get("elapsed_s")
    delta = envelope.get("cache_delta_mb")
    if envelope["dry_run"]:
        print(f"\nDry run — {envelope['n_planned']} model(s) would be fetched.")
        return
    print()
    line = f"Summary: {n_ok} ok, {n_failed} failed"
    if elapsed is not None:
        line += f"  ({elapsed:.1f}s total)"
    if delta is not None:
        # Negative delta is unlikely (would mean blob eviction during a load);
        # show 0 for warm-cache runs where nothing new was downloaded.
        line += f"  cache_delta={max(delta, 0.0):.1f} MB"
    print(line)


def _print_prefetch_line(result: dict[str, Any]) -> None:
    marker = "OK " if result["status"] == "ok" else "FAIL"
    fam = result["family"]
    mid = result["model_id"]
    elapsed = result["elapsed_s"]
    if result["status"] == "ok":
        print(f"  [{marker}] {fam:<10} {mid:<48} {elapsed:>6.1f}s")
    else:
        print(f"  [{marker}] {fam:<10} {mid:<48} {elapsed:>6.1f}s  {result['error']}")


# ---------------------------------------------------------------------------
# Programmatic prefetch API
# ---------------------------------------------------------------------------


def prefetch_models(
    *,
    families: Iterable[str] | None = None,
    model_ids: Iterable[str] | None = None,
    cache_dir: Path | None = None,
    allow_unregistered: bool = False,
) -> dict[str, Any]:
    """Programmatic equivalent of ``kaos-nlp-transformers prefetch``.

    Same resolution rules as the CLI:

    * ``families`` filters to a subset of ``{"embedding", "reranker", "nli", "ner"}``.
      Default = all four.
    * ``model_ids`` overrides ``families`` — load only the named ids.
    * ``cache_dir`` overrides ``KAOS_NLP_TRANSFORMERS_CACHE_DIR`` / ``HF_HOME``.

    Returns a JSON-serializable dict with per-model status + summary.
    Suitable for Dockerfile / CI / startup hooks::

        from kaos_nlp_transformers.cli import prefetch_models
        prefetch_models(families=["nli", "ner"])
    """
    from kaos_nlp_transformers.models import (
        NER_REGISTRY,
        NLI_REGISTRY,
        PII_REGISTRY,
        REGISTRY,
        RERANKER_REGISTRY,
    )

    if model_ids:
        jobs = _resolve_explicit_models(list(model_ids), allow_unregistered=allow_unregistered)
    else:
        selected = set(families) if families else set(_FAMILIES)
        jobs = []
        if "embedding" in selected:
            jobs.extend(("embedding", mid) for mid in sorted(REGISTRY.keys()))
        if "reranker" in selected:
            jobs.extend(("reranker", mid) for mid in sorted(RERANKER_REGISTRY.keys()))
        if "nli" in selected:
            jobs.extend(("nli", mid) for mid in sorted(NLI_REGISTRY.keys()))
        if "ner" in selected:
            jobs.extend(("ner", mid) for mid in sorted(NER_REGISTRY.keys()))
        if "pii" in selected:
            jobs.extend(("pii", mid) for mid in sorted(PII_REGISTRY.keys()))

    envelope: dict[str, Any] = {
        "command": "prefetch",
        "cache_dir": str(cache_dir) if cache_dir else None,
        "dry_run": False,
        "n_planned": len(jobs),
        "models": [],
    }

    target_dir = _resolved_cache_dir(cache_dir)
    cache_size_before = _measure_cache_size_mb(target_dir)
    start = time.perf_counter()
    n_ok = 0
    n_failed = 0
    for family, model_id in jobs:
        result = _prefetch_one(
            family,
            model_id,
            cache_dir=cache_dir,
            allow_unregistered=allow_unregistered,
        )
        envelope["models"].append(result)
        if result["status"] == "ok":
            n_ok += 1
        else:
            n_failed += 1

    cache_size_after = _measure_cache_size_mb(target_dir)
    envelope["elapsed_s"] = time.perf_counter() - start
    envelope["n_ok"] = n_ok
    envelope["n_failed"] = n_failed
    envelope["cache_size_before_mb"] = cache_size_before
    envelope["cache_size_after_mb"] = cache_size_after
    envelope["cache_delta_mb"] = (
        round(cache_size_after - cache_size_before, 1)
        if cache_size_before is not None and cache_size_after is not None
        else None
    )
    return envelope


# ---------------------------------------------------------------------------
# Human renderers
# ---------------------------------------------------------------------------


def _print_human(payload: dict) -> None:
    """Render the info envelope as plain-text sections.

    Latent devices get explicit install hints because that's the path the
    operator is most likely to act on — a GPU box where they didn't realize
    the base install is fastembed-only.
    """
    print(f"package: {payload['package']}")
    print(f"version: {payload['version']}")
    print()
    print("settings:")
    for key, value in payload["settings"].items():
        print(f"  {key}: {value}")
    print()
    print("resolved_device:")
    rd = payload["resolved_device"]
    print(f"  {rd['device']}  ({rd['name']}, backend={rd['backend']}, memory_mb={rd['memory_mb']})")
    print()
    print(f"reachable_devices ({len(payload['reachable_devices'])}):")
    for d in payload["reachable_devices"]:
        print(f"  - {d['device']}  {d['name']}  backend={d['backend']}  memory_mb={d['memory_mb']}")
    print()
    latent = payload["latent_devices"]
    if latent:
        print(f"latent_devices ({len(latent)}) — physically present, NOT reachable:")
        for d in latent:
            hint = d["install_hint"] or "(no single-extra fix; see reason)"
            print(f"  - {d['name']}  kind={d['kind']}")
            print(f"      install: {hint}")
            print(f"      reason:  {d['reason']}")
        print()
    print(f"onnx_providers: {', '.join(payload['onnx_providers']) or '(none)'}")
    print()
    print("embedding_models:")
    for m in payload["embedding_models"]["registered"]:
        print(f"  - {m}")
    if payload["embedding_models"]["excluded"]:
        print("  excluded:")
        for entry in payload["embedding_models"]["excluded"]:
            print(f"    - {entry['model_id']}: {entry['reason']}")
    print()
    print("reranker_models:")
    for m in payload["reranker_models"]["registered"]:
        print(f"  - {m}")
    print()
    print("nli_models:")
    for m in payload["nli_models"]["registered"]:
        print(f"  - {m}")
    if payload["nli_models"]["excluded"]:
        print("  excluded:")
        for entry in payload["nli_models"]["excluded"]:
            print(f"    - {entry['model_id']}: {entry['reason']}")
    print()
    print("ner_models:")
    for m in payload["ner_models"]["registered"]:
        print(f"  - {m}")
    if payload["ner_models"]["excluded"]:
        print("  excluded:")
        for entry in payload["ner_models"]["excluded"]:
            print(f"    - {entry['model_id']}: {entry['reason']}")
    print()
    print("pii_models:")
    for m in payload.get("pii_models", {}).get("registered", []):
        print(f"  - {m}")
    if payload.get("pii_models", {}).get("excluded"):
        print("  excluded:")
        for entry in payload["pii_models"]["excluded"]:
            print(f"    - {entry['model_id']}: {entry['reason']}")


if __name__ == "__main__":
    sys.exit(main())
