"""Run the kaos-nlp-transformers MCP server.

Mirrors the kaos-nlp-core / kaos-tabular serve pattern: argparse loads
instantly, heavy imports (``kaos-core``, ``kaos-mcp``) happen inside the
handler so the CLI ``--help`` path is fast and so a base install (no
``[mcp]`` extra) gets a friendly install hint instead of a stack trace.

Usage:
    # stdio (Claude Code / Claude Desktop) — single-tenant trust boundary
    kaos-nlp-transformers-serve

    # streamable HTTP — REQUIRES KAOS_NLP_TRANSFORMERS_HTTP_TOKEN to be set;
    # the value is an operator acknowledgement that the server's tool
    # surface will be fronted by a reverse proxy doing real authentication.
    # The token itself is not validated against incoming requests by this
    # server; it gates server *startup*, not callers.
    KAOS_NLP_TRANSFORMERS_HTTP_TOKEN=ops-ack \\
        kaos-nlp-transformers-serve --http --port 8000

    # debug logging
    kaos-nlp-transformers-serve --debug
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kaos-nlp-transformers-serve",
        description="kaos-nlp-transformers MCP server (embedding diagnostics).",
    )
    parser.add_argument(
        "--http", action="store_true", help="Use streamable HTTP transport (default: stdio)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    # kaos-core + kaos-mcp are gated behind the [mcp] extra. Import inside the
    # handler so a base install (fastembed-only) gets the actionable install
    # hint rather than a chained ImportError out of settings.py.
    try:
        from kaos_core import KaosRuntime

        # kaos-mcp is the optional sibling; ty cannot see it statically in
        # the per-package repo where it isn't installed, hence the ignore.
        from kaos_mcp import KaosMCPServer, KaosMCPSettings  # ty: ignore[unresolved-import]
    except ImportError as exc:
        print(
            f"kaos-nlp-transformers-serve requires kaos-core and kaos-mcp: {exc}\n"
            "Fix: pip install 'kaos-nlp-transformers[mcp]'.\n"
            "Alternative: use the Python API (EmbeddingModel, EmbeddingRetriever, "
            "CrossEncoderReranker) directly without the MCP surface.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve settings from env (KAOS_NLP_TRANSFORMERS_*) + .env file. The
    # HTTP-token gate below uses the typed SecretStr field rather than a raw
    # os.environ read so the value participates in the same redaction +
    # config-dump path as every other module setting.
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    settings = KaosNLPTransformersSettings()

    # F3 (matches kaos-nlp-core): --http exposes the tool surface to the
    # network. This server doesn't authenticate clients; running --http
    # without a reverse proxy is unsafe by construction. Require the operator
    # to set KAOS_NLP_TRANSFORMERS_HTTP_TOKEN as an explicit acknowledgement
    # that an external auth layer is in place.
    if args.http and (settings.http_token is None or not settings.http_token.get_secret_value()):
        print(
            "kaos-nlp-transformers-serve --http refuses to start without "
            "KAOS_NLP_TRANSFORMERS_HTTP_TOKEN.\n"
            "\n"
            "The HTTP transport does not validate incoming requests. To run safely:\n"
            "  1. Front this process with a reverse proxy that authenticates "
            "callers (mTLS, bearer-token, OAuth, …).\n"
            "  2. Set KAOS_NLP_TRANSFORMERS_HTTP_TOKEN=<any-non-empty-string> "
            "to confirm you have done (1).\n"
            "\n"
            "For local single-tenant use, prefer stdio: "
            "`kaos-nlp-transformers-serve` (no --http).",
            file=sys.stderr,
        )
        sys.exit(2)

    from kaos_nlp_transformers.tools import register_transformers_tools

    runtime = KaosRuntime()
    n_tools = register_transformers_tools(runtime)
    print(f"Registered {n_tools} kaos-nlp-transformers tool(s)", file=sys.stderr)

    mcp_settings = KaosMCPSettings(
        name="kaos-nlp-transformers-server",
        transport="streamable-http" if args.http else "stdio",
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    server = KaosMCPServer(runtime=runtime, settings=mcp_settings)

    if args.http:
        print(f"Starting HTTP server on {args.host}:{args.port}/mcp", file=sys.stderr)
        server.run_streamable_http()
    else:
        print("Starting stdio server", file=sys.stderr)
        server.run_stdio()


if __name__ == "__main__":
    main()
