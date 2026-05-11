"""Type stubs for the Rust extension module.

The actual implementation lives in ``rust/lib.rs`` and is compiled by
maturin into ``kaos_nlp_transformers/_rust.abi3.so`` — a single file at
this same level. These stubs keep type checkers (``ty``, pyright,
mypy) resolving imports like ``from kaos_nlp_transformers._rust import
__version__`` even though the runtime artifact is a binary cdylib.

## Packaging shape note

This file lives next to ``_rust.abi3.so`` — NOT inside a ``_rust/``
directory. The previous layout shipped per-submodule stubs in a
``_rust/`` subdirectory next to the cdylib, which CPython's
namespace-package detector could ambiguously resolve as a package and
shadow the ``.so``. The wheel-install smoke test caught this with
``ImportError: cannot import name 'registry' from
'kaos_nlp_transformers._rust' (unknown location)``. See
``docs/oss/40-ci-cd/wheels.yml.md#direct-rust-submodule-imports-in-smoke``
in the kaos-modules monorepo for the full incident write-up.

## Stub structure

The cdylib registers four submodules at import time (``embedding``,
``registry``, ``reranker``, ``tokenize``) via PyO3 ``add_submodule`` +
``sys.modules`` insertion. Each one is a real Python module at
runtime. From a static-typing perspective there's no clean way to
declare "this single ``.pyi`` file is the type interface for both
``_rust`` AND its submodules" — pyi files map 1:1 to modules.

We use the **class-as-namespace** convention here: each runtime
submodule is represented as a ``class`` in this file containing the
relevant types. Type checkers resolve attribute access
(``_rust.embedding.EmbeddingBackend``) against the nested class
correctly. Direct-import forms (``from kaos_nlp_transformers._rust.embedding
import EmbeddingBackend``) still resolve at RUNTIME (the cdylib
populates ``sys.modules``) but type checkers report them as
``import-not-found``; consumers should add ``# ty: ignore[import-not-found]``
on those lines, or migrate to attribute-style access through the
parent ``_rust`` module if they want full static coverage.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__version__: str

# ────────────────────────────────────────────────────────────────────────
# Submodule namespaces — runtime modules, declared as classes here so
# static attribute access (`_rust.embedding.EmbeddingBackend`) resolves.
# ────────────────────────────────────────────────────────────────────────

class embedding:
    """Runtime module ``kaos_nlp_transformers._rust.embedding``.

    Registered by ``rust/bindings/embedding.rs::register_module``.
    """

    class EmbeddingBackend:
        """Rust-backed embedding inference. Wraps an ort Session loaded
        from a registered model."""

        @staticmethod
        def load(
            model_id: str,
            *,
            device: str = "cpu",
            cache_dir: str | None = None,
        ) -> embedding.EmbeddingBackend: ...
        def embed(
            self,
            texts: list[str],
            batch_size: int = 32,
        ) -> np.ndarray[Any, np.dtype[np.float32]]: ...
        def count_tokens(self, texts: list[str]) -> list[int]: ...
        @property
        def dim(self) -> int: ...
        @property
        def model_id(self) -> str: ...
        @property
        def device(self) -> str: ...
        @property
        def max_seq_len(self) -> int: ...

class registry:
    """Runtime module ``kaos_nlp_transformers._rust.registry``.

    Registered by ``rust/bindings/registry.rs::register_module``.
    """

    __version__: str

    @staticmethod
    def capabilities() -> dict[str, bool | list[str]]:
        """Compile-time + runtime capability snapshot.

        Returns a dict shaped like::

            {
              "cpu": True,
              "cuda": False,
              "openvino": False,
              "build_features": [],
            }
        """

    @staticmethod
    def vendored_model_path(model_id: str) -> str | None:
        """Return the absolute path of the vendored copy of ``model_id``,
        or ``None`` if no vendored copy ships in this wheel.
        """

class reranker:
    """Runtime module ``kaos_nlp_transformers._rust.reranker``.

    Registered by ``rust/bindings/reranker.rs::register_module``.
    """

    class CrossEncoderBackend:
        """Rust-backed cross-encoder inference. Wraps an ort Session loaded
        from a registered reranker model."""

        @staticmethod
        def load(
            model_id: str,
            *,
            device: str = "cpu",
            cache_dir: str | None = None,
        ) -> reranker.CrossEncoderBackend: ...
        def score(
            self,
            queries: list[str],
            passages: list[str],
            batch_size: int = 32,
        ) -> np.ndarray[Any, np.dtype[np.float32]]: ...
        @property
        def model_id(self) -> str: ...
        @property
        def device(self) -> str: ...

class tokenize:
    """Runtime module ``kaos_nlp_transformers._rust.tokenize``.

    Registered by ``rust/bindings/tokenize.rs::register_module``.
    Mostly an internal test surface; production paths use the
    Rust-side TokenizerWrapper via ``embedding.EmbeddingBackend`` /
    ``reranker.CrossEncoderBackend``.
    """

    class Tokenizer:
        @staticmethod
        def from_file(path: str, max_seq_len: int) -> tokenize.Tokenizer: ...
        def encode_batch(
            self,
            texts: list[str],
        ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]: ...
        @property
        def pad_id(self) -> int: ...
        @property
        def max_seq_len(self) -> int: ...
