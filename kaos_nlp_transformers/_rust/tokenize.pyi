"""Stub for kaos_nlp_transformers._rust.tokenize."""

from __future__ import annotations

class Tokenizer:
    """Thin wrapper over the HF tokenizers crate. Mostly an internal
    test surface; production paths use the Rust-side TokenizerWrapper
    via EmbeddingBackend / CrossEncoderBackend."""

    @staticmethod
    def from_file(path: str, max_seq_len: int) -> Tokenizer: ...
    def encode_batch(
        self,
        texts: list[str],
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]: ...
    @property
    def pad_id(self) -> int: ...
    @property
    def max_seq_len(self) -> int: ...
