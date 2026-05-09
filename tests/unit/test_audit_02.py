"""Regression tests for audit-02 findings KNT-101..KNT-107.

These tests pin the audit-02 fixes so a future refactor can't silently
reintroduce the original problems. Companion to ``test_audit_01.py``;
both run on every CI invocation.
"""

from __future__ import annotations

import inspect
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# -----------------------------------------------------------------------------
# KNT-101 — EmbeddingModel.embed enforces L2 normalization
# -----------------------------------------------------------------------------


class _FakeBackend:
    """Stand-in for fastembed/sentence-transformers that returns vectors with
    arbitrary magnitudes — lets us prove the model normalizes regardless of
    backend output."""

    def __init__(self, vecs: np.ndarray) -> None:
        self._vecs = vecs

    def embed(self, texts: list[str], batch_size: int = 32):
        return iter(self._vecs)

    def encode(self, texts: list[str], **kwargs):
        return self._vecs


def _make_embedding_model_with_fake(arr: np.ndarray):
    """Construct an EmbeddingModel directly via __init__ with a fake backend.

    Bypasses the registry / device / load() path so the test isolates
    embed() behavior.
    """
    from kaos_nlp_transformers import RegisteredModel
    from kaos_nlp_transformers.device import DeviceInfo
    from kaos_nlp_transformers.embedding import EmbeddingModel

    backend = _FakeBackend(arr)
    registered = RegisteredModel(
        model_id="test/fake",
        revision="abcdef0",
        license="MIT",
        params_m=0,
        dim=int(arr.shape[1]),
        backend="fastembed",
        notes="test fixture",
    )
    device = DeviceInfo(name="CPU", device="cpu", backend="fastembed", memory_mb=0)
    return EmbeddingModel(registered, backend, device=device, backend_name="fastembed")


def test_embed_returns_unit_norm_rows_for_arbitrary_backend_output():
    """Backend rows with arbitrary magnitudes must come out as unit vectors."""
    raw = np.array(
        [
            [3.0, 4.0],  # norm 5.0
            [0.5, 0.0],  # norm 0.5
            [10.0, 0.0],  # norm 10.0
        ],
        dtype=np.float32,
    )
    model = _make_embedding_model_with_fake(raw)
    out = model.embed(["a", "b", "c"])
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), norms
    # First row: (3,4)/5 = (0.6, 0.8).
    assert np.allclose(out[0], [0.6, 0.8], atol=1e-5)


def test_embed_handles_zero_vector_without_nan():
    """All-zero rows survive normalization as zero rows (no NaN)."""
    raw = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    # Construct the test model with dim=3 so the registry-dim check passes.
    from kaos_nlp_transformers import RegisteredModel
    from kaos_nlp_transformers.device import DeviceInfo
    from kaos_nlp_transformers.embedding import EmbeddingModel

    registered = RegisteredModel(
        model_id="test/fake-3d",
        revision="abcdef0",
        license="MIT",
        params_m=0,
        dim=3,
        backend="fastembed",
        notes="test fixture",
    )
    device = DeviceInfo(name="CPU", device="cpu", backend="fastembed", memory_mb=0)
    model = EmbeddingModel(registered, _FakeBackend(raw), device=device, backend_name="fastembed")

    out = model.embed(["zero", "unit"])
    assert not np.isnan(out).any()
    assert np.linalg.norm(out[0]) == 0.0
    assert np.allclose(np.linalg.norm(out[1]), 1.0, atol=1e-6)


# -----------------------------------------------------------------------------
# KNT-102 — retriever input-validation hardening
# -----------------------------------------------------------------------------


def _make_retriever_constructor_args(n: int = 3, dim: int = 4):
    """Build a minimum-viable kwargs bundle for EmbeddingRetriever(...).

    Embeddings are unit-norm; the model arg is a MagicMock since we do not
    exercise retrieve() here.
    """
    embeddings = np.eye(n, dim, dtype=np.float32)
    return {
        "embeddings": embeddings,
        "doc_ids": list(range(n)),
        "texts": [f"doc {i}" for i in range(n)],
        "model": MagicMock(),
    }


def test_retriever_init_rejects_external_ids_length_mismatch():
    from kaos_nlp_transformers.retrieval import EmbeddingRetriever

    args = _make_retriever_constructor_args(n=3)
    args["external_ids"] = ["a", "b"]  # len 2, mismatch
    with pytest.raises(ValueError, match=r"external_ids length"):
        EmbeddingRetriever(**args)


def test_retriever_init_rejects_metadata_list_length_mismatch():
    from kaos_nlp_transformers.retrieval import EmbeddingRetriever

    args = _make_retriever_constructor_args(n=3)
    args["metadata_list"] = [{"a": 1}, {"b": 2}]  # len 2, mismatch
    with pytest.raises(ValueError, match=r"metadata_list length"):
        EmbeddingRetriever(**args)


def test_retriever_init_rejects_explicit_empty_list():
    """The pre-fix falsy-empty-list behavior silently filled defaults; with
    KNT-102 the lengths must match exactly."""
    from kaos_nlp_transformers.retrieval import EmbeddingRetriever

    args = _make_retriever_constructor_args(n=3)
    args["external_ids"] = []
    with pytest.raises(ValueError, match=r"external_ids length"):
        EmbeddingRetriever(**args)


def test_retriever_init_accepts_omitted_optional_fields():
    """Passing None (not [] or omitted) must keep auto-fill defaults
    working — that's the documented sentinel for "use defaults"."""
    from kaos_nlp_transformers.retrieval import EmbeddingRetriever

    args = _make_retriever_constructor_args(n=3)
    args["external_ids"] = None
    args["metadata_list"] = None
    r = EmbeddingRetriever(**args)
    assert r.num_documents == 3


def test_add_documents_validates_before_mutating():
    """If add_documents detects a length mismatch, the retriever's
    pre-call state must remain intact (no partial commit)."""
    from kaos_nlp_transformers.retrieval import EmbeddingRetriever

    args = _make_retriever_constructor_args(n=3, dim=4)
    r = EmbeddingRetriever(**args)
    pre_n = r.num_documents
    pre_embedding_rows = r._embeddings.shape[0]

    fake_model = args["model"]
    fake_model.embed.return_value = np.eye(2, 4, dtype=np.float32)
    r._model = fake_model

    with pytest.raises(ValueError, match=r"external_ids length"):
        r.add_documents(
            texts=["x", "y"],
            doc_ids=[10, 11],
            external_ids=["only-one"],
        )

    # State is unchanged after the failed call.
    assert r.num_documents == pre_n
    assert r._embeddings.shape[0] == pre_embedding_rows


# -----------------------------------------------------------------------------
# KNT-103 — scoped offline mode (env vars restored on return + on exception)
# -----------------------------------------------------------------------------


def test_offline_env_scope_restores_on_clean_exit(monkeypatch):
    """After a successful load, env vars return to whatever the caller
    set before the call (or to "unset" if they were unset)."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    from kaos_nlp_transformers.embedding import _offline_env_scope

    with _offline_env_scope(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_offline_env_scope_overrides_hostile_zero(monkeypatch):
    """A user shell with HF_HUB_OFFLINE=0 must be overridden inside the
    scope — and restored to "0" after."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")

    from kaos_nlp_transformers.embedding import _offline_env_scope

    with _offline_env_scope(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"

    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_offline_env_scope_restores_on_exception(monkeypatch):
    """Backend exception inside the scope must not leave the env
    permanently mutated."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    from kaos_nlp_transformers.embedding import _offline_env_scope

    class Boom(Exception):
        pass

    with pytest.raises(Boom), _offline_env_scope(True):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        raise Boom("simulated backend failure")

    assert "HF_HUB_OFFLINE" not in os.environ


def test_offline_false_is_noop(monkeypatch):
    """offline=False must not touch the env at all — the user may have
    other reasons to set HF_HUB_OFFLINE."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    from kaos_nlp_transformers.embedding import _offline_env_scope

    with _offline_env_scope(False):
        assert os.environ["HF_HUB_OFFLINE"] == "1"

    assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_consecutive_offline_loads_dont_leak(monkeypatch):
    """Two consecutive load() calls with different offline values:
    the second one must NOT see the first one's env-var leftovers."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    # Audit-03 KNT-201: the free-threaded guard short-circuits load
    # BEFORE the backend loader runs. Force the GIL-enabled path so
    # this test exercises only the consecutive-load offline contract.
    import sys

    monkeypatch.setattr(sys, "_is_gil_enabled", lambda: True, raising=False)

    from kaos_nlp_transformers import embedding as embedding_mod
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    captured: list[str | None] = []

    def _capture_then_explode(*args, **kwargs):
        captured.append(os.environ.get("HF_HUB_OFFLINE"))
        msg = "stub"
        raise RuntimeError(msg)

    # Audit-06 KNT-501: SE loader retired; only fastembed needs stubbing.
    monkeypatch.setattr(embedding_mod, "_load_fastembed_cached", _capture_then_explode)

    s_on = KaosNLPTransformersSettings(offline=True)
    s_off = KaosNLPTransformersSettings(offline=False)

    with pytest.raises(RuntimeError):
        embedding_mod.EmbeddingModel.load(settings=s_on)
    with pytest.raises(RuntimeError):
        embedding_mod.EmbeddingModel.load(settings=s_off)

    assert captured == ["1", None], (
        f"Expected first call to see '1', second to see None; got {captured!r}. "
        "If second is '1', offline mode is leaking across loads."
    )


# -----------------------------------------------------------------------------
# KNT-104 — reranker registry parity
# -----------------------------------------------------------------------------


def test_reranker_registry_has_at_least_one_entry():
    from kaos_nlp_transformers import RERANKER_REGISTRY

    assert len(RERANKER_REGISTRY) >= 1
    assert "BAAI/bge-reranker-base" in RERANKER_REGISTRY
    entry = RERANKER_REGISTRY["BAAI/bge-reranker-base"]
    # Revision is pinned to a real SHA, never "main".
    assert entry.revision != "main"
    assert len(entry.revision) >= 7
    # License is permissive.
    assert entry.license in {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause"}


def test_reranker_load_rejects_unregistered(monkeypatch):
    """Without allow_unregistered=True, an unknown reranker model_id must
    raise ModelNotRegisteredError."""
    from kaos_nlp_transformers import KaosNLPTransformersSettings, ModelNotRegisteredError
    from kaos_nlp_transformers.reranker import CrossEncoderReranker

    monkeypatch.delenv("KAOS_NLP_TRANSFORMERS_ALLOW_UNREGISTERED", raising=False)
    s = KaosNLPTransformersSettings()  # allow_unregistered=False default
    assert s.allow_unregistered is False

    with pytest.raises(ModelNotRegisteredError, match=r"not in the v0 registry"):
        CrossEncoderReranker.load("some/random-org-model", settings=s)


def test_reranker_loader_signature_accepts_revision_and_cache_dir():
    """The cache layer must be keyed by revision and cache_dir so a
    registry SHA bump invalidates the cached backend."""
    from kaos_nlp_transformers.reranker import _load_cross_encoder_cached

    params = inspect.signature(_load_cross_encoder_cached).parameters
    assert "revision" in params, "reranker loader must accept revision"
    assert "cache_dir" in params, "reranker loader must accept cache_dir"


def test_reranker_excluded_blocks_load(monkeypatch):
    """Adding a model to RERANKER_EXCLUDED prevents loading even if the
    user supplies allow_unregistered=True."""
    from kaos_nlp_transformers import (
        KaosNLPTransformersSettings,
        ModelNotRegisteredError,
    )
    from kaos_nlp_transformers.models import RERANKER_EXCLUDED
    from kaos_nlp_transformers.reranker import CrossEncoderReranker

    # Inject a temporary exclusion entry (cleaned up by monkeypatch teardown).
    monkeypatch.setitem(RERANKER_EXCLUDED, "evil/non-commercial-reranker", "CC-BY-NC")
    s = KaosNLPTransformersSettings(allow_unregistered=True)

    with pytest.raises(ModelNotRegisteredError, match=r"excluded from the registry"):
        CrossEncoderReranker.load("evil/non-commercial-reranker", settings=s)


# -----------------------------------------------------------------------------
# KNT-105 — semantic dedup similarity reporting
# -----------------------------------------------------------------------------


def test_semantic_dedup_threshold_validated():
    """distance_threshold outside cosine domain [0, 2] is rejected at
    construction time."""
    from kaos_nlp_transformers.clustering.semantic_dedup import SemanticDedupLevel

    with pytest.raises(ValueError, match=r"distance_threshold"):
        SemanticDedupLevel(distance_threshold=-0.1)
    with pytest.raises(ValueError, match=r"distance_threshold"):
        SemanticDedupLevel(distance_threshold=3.0)
    # Valid thresholds construct cleanly.
    SemanticDedupLevel(distance_threshold=0.0)
    SemanticDedupLevel(distance_threshold=0.10)
    SemanticDedupLevel(distance_threshold=2.0)


def test_semantic_dedup_returns_real_similarity(monkeypatch):
    """With three near-duplicate embeddings, the cluster's similarity must
    be in (0.5, 1.0) — not the inherited 1.0 default."""
    pytest.importorskip("scipy", reason="SemanticDedupLevel requires the [clustering] extra")
    from kaos_content.dedup.types import DedupDocument

    from kaos_nlp_transformers.clustering import semantic_dedup as sd

    # Three rows: rows 0 and 1 are very close; row 2 is far away.
    fake_vecs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.31, 0.0],  # cosine ~0.95 vs row 0
            [0.0, 0.0, 1.0],  # cosine 0 vs row 0
        ],
        dtype=np.float32,
    )

    fake_model = MagicMock()
    fake_model.embed.return_value = fake_vecs
    # find_clusters lazy-imports EmbeddingModel from kaos_nlp_transformers.embedding,
    # so we monkeypatch that module rather than the dedup module.
    from kaos_nlp_transformers import embedding as embedding_mod

    monkeypatch.setattr(
        embedding_mod.EmbeddingModel, "load", classmethod(lambda *a, **k: fake_model)
    )

    docs = [
        DedupDocument(doc_id="a", text="this is text one"),
        DedupDocument(doc_id="b", text="this is text two close to one"),
        DedupDocument(doc_id="c", text="completely unrelated"),
    ]
    level = sd.SemanticDedupLevel(distance_threshold=0.20)
    clusters = level.find_clusters(docs)
    assert len(clusters) == 1, [c.member_doc_ids for c in clusters]
    cluster = clusters[0]
    # a and b should cluster; their mean cosine ≈ 0.95.
    assert set(cluster.member_doc_ids) == {"a", "b"}
    assert 0.90 <= cluster.similarity <= 1.0
    # And critically: NOT the inherited 1.0 default (real intra-cluster sim).
    assert cluster.similarity != 1.0


# -----------------------------------------------------------------------------
# KNT-106 — from_corpus single-path embedding
# -----------------------------------------------------------------------------


def test_from_corpus_uses_loaded_model_not_corpus_embed(monkeypatch):
    """The 0.1.0a1 fallback path called corpus.embed(model=, batch_size=)
    without forwarding device/backend/settings policy. KNT-106 routes
    everything through EmbeddingModel.embed so the loaded model's
    policy reaches every row.
    """
    from kaos_nlp_transformers.retrieval import EmbeddingRetriever

    fake_unit = MagicMock(
        row=0,
        text="hello",
        doc_uri="test://doc",
        page=None,
        section_ref=None,
        section_title=None,
    )

    class CorpusWithEmbed:
        """If KNT-106 ever regresses to the dual-path code, corpus.embed
        gets called and we'd see the spy fire."""

        def __init__(self):
            self.embed_was_called = False

        def embed(self, model=None, batch_size=32):
            self.embed_was_called = True
            return np.eye(1, 384, dtype=np.float32)

        def __iter__(self):
            return iter([fake_unit])

    # Stub EmbeddingModel.load so we don't hit the registry.
    fake_em = MagicMock()
    fake_em.embed.return_value = np.eye(1, 384, dtype=np.float32)
    monkeypatch.setattr(
        "kaos_nlp_transformers.retrieval.EmbeddingModel.load",
        classmethod(lambda *a, **k: fake_em),
    )

    corpus = CorpusWithEmbed()
    retriever = EmbeddingRetriever.from_corpus(corpus)
    assert retriever.num_documents == 1
    assert corpus.embed_was_called is False, (
        "from_corpus should NOT call corpus.embed() — single embedding "
        "path through the loaded EmbeddingModel (KNT-106)."
    )
    assert fake_em.embed.called is True


def test_from_corpus_materializes_iterator_once(monkeypatch):
    """Iterator-style corpus must be consumed exactly once (not twice
    as the 0.1.0a1 code did)."""
    from kaos_nlp_transformers.retrieval import EmbeddingRetriever

    fake_units = [
        MagicMock(
            row=i,
            text=f"text {i}",
            doc_uri=f"test://doc/{i}",
            page=None,
            section_ref=None,
            section_title=None,
        )
        for i in range(3)
    ]

    class IteratorOnlyCorpus:
        def __init__(self):
            self.iter_count = 0

        def __iter__(self):
            self.iter_count += 1
            return iter(fake_units)

    fake_em = MagicMock()
    fake_em.embed.return_value = np.eye(3, 384, dtype=np.float32)

    monkeypatch.setattr(
        "kaos_nlp_transformers.retrieval.EmbeddingModel.load",
        classmethod(lambda *a, **k: fake_em),
    )
    corpus = IteratorOnlyCorpus()
    retriever = EmbeddingRetriever.from_corpus(corpus)
    assert retriever.num_documents == 3
    assert corpus.iter_count == 1, f"Expected exactly 1 iteration, got {corpus.iter_count}"


# -----------------------------------------------------------------------------
# KNT-107 — invalid backend strings raise
# -----------------------------------------------------------------------------


def test_resolve_backend_rejects_unknown_backend():
    from kaos_nlp_transformers.device import DeviceInfo
    from kaos_nlp_transformers.embedding import _resolve_backend

    cpu = DeviceInfo(name="CPU", device="cpu", backend="fastembed", memory_mb=0)

    # Valid values still work. Audit-06 KNT-501: ``sentence-transformers`` was
    # retired alongside the torch backend; the surviving valid set is
    # {"auto", "fastembed", "model2vec"}.
    assert _resolve_backend("fastembed", cpu, "fastembed") == "fastembed"
    assert _resolve_backend("auto", cpu, "fastembed") == "fastembed"
    assert _resolve_backend("model2vec", cpu, "fastembed") == "model2vec"

    # Unknown values raise.
    with pytest.raises(ValueError, match=r"Invalid backend"):
        _resolve_backend("tensorflow", cpu, "fastembed")
    with pytest.raises(ValueError, match=r"Invalid backend"):
        _resolve_backend("", cpu, "fastembed")
    with pytest.raises(ValueError, match=r"Invalid backend"):
        _resolve_backend("FastEmbed", cpu, "fastembed")  # case-sensitive
    # Audit-06 KNT-501 regression guard: SE name is now invalid.
    with pytest.raises(ValueError, match=r"Invalid backend"):
        _resolve_backend("sentence-transformers", cpu, "fastembed")


# -----------------------------------------------------------------------------
# Public API contract — RERANKER_REGISTRY exported (KNT-104 follow-on)
# -----------------------------------------------------------------------------


def test_rerank_registries_are_top_level_exports():
    import kaos_nlp_transformers

    assert "RERANKER_REGISTRY" in kaos_nlp_transformers.__all__
    assert "RERANKER_EXCLUDED" in kaos_nlp_transformers.__all__
    assert kaos_nlp_transformers.RERANKER_REGISTRY  # non-empty


# -----------------------------------------------------------------------------
# Property-level invariants (no hypothesis dep — basic numpy fuzzers)
# -----------------------------------------------------------------------------


def test_retriever_length_invariant_after_random_appends():
    """Internal parallel arrays/lists stay length-equal across N random
    add_documents calls. A regression to the old "extend without
    validating" path would surface here."""
    from kaos_nlp_transformers.retrieval import EmbeddingRetriever

    rng = np.random.default_rng(42)
    args = _make_retriever_constructor_args(n=3, dim=8)
    r = EmbeddingRetriever(**args)

    fake_em = args["model"]
    r._model = fake_em

    for _ in range(20):
        n_new = int(rng.integers(1, 6))
        fake_em.embed.return_value = rng.standard_normal((n_new, 8)).astype(np.float32)
        r.add_documents(
            texts=[f"t-{i}" for i in range(n_new)],
            doc_ids=[100 + i for i in range(n_new)],
        )

        n = r.num_documents
        assert r._embeddings.shape[0] == n
        assert len(r._doc_ids) == n
        assert len(r._texts) == n
        assert len(r._external_ids) == n
        assert len(r._metadata_list) == n


def test_semantic_dedup_threshold_monotonicity(monkeypatch):
    """For a fixed embedding matrix, tightening the threshold must never
    INCREASE the total number of clustered members. (Membership is a
    monotonic non-increasing function of distance_threshold under
    average-linkage hierarchical clustering.)"""
    pytest.importorskip("scipy", reason="SemanticDedupLevel requires the [clustering] extra")
    from kaos_content.dedup.types import DedupDocument

    from kaos_nlp_transformers.clustering import semantic_dedup as sd

    rng = np.random.default_rng(0)
    n = 12
    dim = 16
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    # Make some pairs near-duplicates to populate clusters at low thresholds.
    vecs[1] = vecs[0] + 0.01 * rng.standard_normal(dim)
    vecs[3] = vecs[2] + 0.02 * rng.standard_normal(dim)
    vecs[5] = vecs[4] + 0.05 * rng.standard_normal(dim)
    # L2-normalize so the dedup level can do its math correctly.
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    fake_model = MagicMock()
    fake_model.embed.return_value = vecs
    from kaos_nlp_transformers import embedding as embedding_mod

    monkeypatch.setattr(
        embedding_mod.EmbeddingModel, "load", classmethod(lambda *a, **k: fake_model)
    )

    docs = [DedupDocument(doc_id=f"d{i}", text=f"document {i}") for i in range(n)]

    last_membership_count = n  # upper bound
    for thr in [0.50, 0.30, 0.20, 0.10, 0.05, 0.02]:
        clusters = sd.SemanticDedupLevel(distance_threshold=thr).find_clusters(docs)
        membership_count = sum(len(c.member_doc_ids) for c in clusters)
        assert membership_count <= last_membership_count, (
            f"Threshold {thr}: membership_count={membership_count} > "
            f"prior {last_membership_count}; threshold-monotonicity violated."
        )
        last_membership_count = membership_count
