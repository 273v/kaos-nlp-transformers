"""Stub MCP server entry for kaos-nlp-transformers.

Full MCP server ships in Phase v1.3 with 3-5 tools.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print(
        "kaos-nlp-transformers MCP server is not implemented in v0. "
        "It ships in Phase v1.3 — see "
        "docs/internal/plans/kaos-nlp-transformers-v0.md.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
