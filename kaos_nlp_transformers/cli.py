"""Stub CLI for kaos-nlp-transformers.

Full CLI ships in Phase v1.3. v0 ships only ``info`` so the entry point
is registered.
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
    info = sub.add_parser("info", help="Show settings, registered models, status")
    info.add_argument("--json", action="store_true", help="Emit JSON envelope")

    args = parser.parse_args(argv)

    if args.cmd is None or args.cmd == "info":
        from kaos_nlp_transformers import __version__
        from kaos_nlp_transformers.models import EXCLUDED, REGISTRY
        from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

        s = KaosNLPTransformersSettings()
        payload = {
            "command": "info",
            "package": "kaos-nlp-transformers",
            "version": __version__,
            "default_model": s.default_model,
            "cache_dir": str(s.cache_dir) if s.cache_dir else None,
            "offline": s.offline,
            "allow_unregistered": s.allow_unregistered,
            "profile": s.profile,
            "registered_models": sorted(REGISTRY.keys()),
            "excluded_models": sorted(EXCLUDED.keys()),
        }
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2))
        else:
            for k, v in payload.items():
                if isinstance(v, list):
                    print(f"{k}:")
                    for item in v:
                        print(f"  - {item}")
                else:
                    print(f"{k}: {v}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
