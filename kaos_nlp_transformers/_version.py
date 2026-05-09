"""Version constant for kaos-nlp-transformers.

Audit KNT-601 (0.2.0): the version is now sourced from Cargo.toml via
the Rust extension. ``__version__`` reads from installed package
metadata so it matches what ``pip show kaos-nlp-transformers`` reports
(PEP 440 form, e.g. ``"0.2.0a1"``). Falling back to the Rust
extension's Cargo SemVer string (``"0.2.0-alpha.1"``) only happens for
editable / in-place builds where dist-info is missing — which would
otherwise cause a version drift between ``kaos_nlp_transformers.__version__``
and PyPI's metadata. Mirrors the kaos-nlp-core pattern documented in
that repo's ``per-package-release.md A7 / F009 lesson #3``.
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version

    try:
        __version__: str = _version("kaos-nlp-transformers")
    except _PackageNotFoundError:  # pragma: no cover - source/editable build only
        from kaos_nlp_transformers._rust import __version__  # type: ignore[import-not-found]
    del _version, _PackageNotFoundError
except Exception:  # pragma: no cover - defensive: importlib.metadata always present on 3.13+
    from kaos_nlp_transformers._rust import __version__  # type: ignore[import-not-found]

__all__ = ["__version__"]
