"""CLI for kaos-nlp-transformers.

v0 ships ``info``, the human / agent-readable diagnostic envelope. The
JSON shape mirrors the ``kaos-nlp-transformers-info`` MCP tool so that
operators and agents see the same information through both surfaces.
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kaos-nlp-transformers",
        description="Dense embeddings and small-model inference for KAOS",
    )
    sub = parser.add_subparsers(dest="cmd", required=False)
    info = sub.add_parser("info", help="Show settings, registered models, devices")
    info.add_argument("--json", action="store_true", help="Emit JSON envelope")

    args = parser.parse_args(argv)

    if args.cmd is None or args.cmd == "info":
        # Lazy imports keep `--help` snappy and avoid forcing model loads on a
        # CLI invocation that just wants to see the subcommand list.
        from kaos_nlp_transformers import __version__
        from kaos_nlp_transformers.device import get_system_devices, resolve_device
        from kaos_nlp_transformers.models import (
            EXCLUDED,
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
                "device": settings.device,
                "backend": settings.backend,
                "offline": settings.offline,
                "allow_unregistered": settings.allow_unregistered,
                "profile": settings.profile,
                "cache_dir": str(settings.cache_dir) if settings.cache_dir else None,
                "workspace_root": (
                    str(settings.workspace_root) if settings.workspace_root else None
                ),
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
        }

        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2))
        else:
            _print_human(payload)
        return 0

    parser.print_help()
    return 1


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


if __name__ == "__main__":
    sys.exit(main())
