"""Unit tests for the disk-backed incremental embedding cache (KNT-602).

All tests run offline through the real ``EmbeddingModel.embed`` entry
point using the wheel-vendored ``minishlab/potion-base-8M`` static model
(no network, no warm HF cache). They exercise: cold-miss writes, warm-hit
forward-pass skip, cross-"process" persistence, model/revision/shape
namespacing, bit-identical-vs-uncached output, and atomic-write
resilience to orphan temp files.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

_MODEL = "minishlab/potion-base-8M"
_TEXTS = ["hello world", "force majeure clauses excuse performance", "third distinct text"]


def _skip_if_no_model2vec() -> None:
    pytest.importorskip("model2vec")


@pytest.fixture(autouse=True)
def _clear_in_proc_lru():
    """Reset the process-wide LRU around every test so disk behavior is
    observed in isolation from the in-proc cache."""
    from kaos_nlp_transformers import embedding as E

    E._embed_cache_clear()
    yield
    E._embed_cache_clear()


def _offline_settings(cache_dir):
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    return KaosNLPTransformersSettings(offline=True, embedding_cache_dir=cache_dir)


def _load(cache_dir):
    from kaos_nlp_transformers import EmbeddingModel

    return EmbeddingModel.load(_MODEL, settings=_offline_settings(cache_dir))


class _ExplodingBackend:
    """Stands in for the real backend; any forward pass raises so a warm
    cache hit is proven by the *absence* of an exception."""

    def encode(self, *args, **kwargs):
        raise AssertionError("forward pass ran on a warm cache hit")

    def embed(self, *args, **kwargs):
        raise AssertionError("forward pass ran on a warm cache hit")


# -- (i) cold cache miss writes files --------------------------------------


def test_cold_miss_writes_npy_files(tmp_path):
    _skip_if_no_model2vec()
    model = _load(tmp_path)
    model.embed(_TEXTS)
    files = sorted(tmp_path.rglob("*.npy"))
    assert len(files) == len(_TEXTS)
    # Layout: <safe_model_id>/<revision>/<dim>-<dtype>/<hash>.npy
    rel = files[0].relative_to(tmp_path)
    assert rel.parts[0] == "minishlab_potion-base-8M"
    assert rel.parts[1] == model._registered.revision
    assert rel.parts[2] == f"{model.dim}-float32"
    assert rel.parts[3].endswith(".npy")


def test_disk_disabled_writes_nothing(tmp_path):
    _skip_if_no_model2vec()
    from kaos_nlp_transformers import EmbeddingModel
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    s = KaosNLPTransformersSettings(offline=True, embedding_cache_dir=None)
    model = EmbeddingModel.load(_MODEL, settings=s)
    model.embed(_TEXTS)
    assert not list(tmp_path.rglob("*.npy"))


# -- (ii) warm cache hit skips the forward pass ----------------------------


def test_warm_hit_skips_forward_pass(tmp_path):
    _skip_if_no_model2vec()
    from kaos_nlp_transformers import embedding as E

    warm = _load(tmp_path)
    warm.embed(_TEXTS)

    # New instance (disk warm), independent LRU cleared, exploding backend:
    # if the disk layer did NOT serve every text, the backend would raise.
    E._embed_cache_clear()
    cold = _load(tmp_path)
    cold._backend = _ExplodingBackend()
    out = cold.embed(_TEXTS)
    assert out.shape == (len(_TEXTS), warm.dim)


def test_partial_hit_only_embeds_misses(tmp_path):
    _skip_if_no_model2vec()
    from kaos_nlp_transformers import embedding as E

    warm = _load(tmp_path)
    # Seed only the first two texts.
    warm.embed(_TEXTS[:2])

    E._embed_cache_clear()
    model = _load(tmp_path)

    seen: list[list[str]] = []
    real_embed_uncached = model._embed_uncached

    def _spy(texts, *, batch_size):
        seen.append(list(texts))
        return real_embed_uncached(texts, batch_size=batch_size)

    model._embed_uncached = _spy  # type: ignore[method-assign]
    model.embed(_TEXTS)
    # Only the genuinely-new third text should reach the forward pass.
    assert seen == [[_TEXTS[2]]]


# -- (iii) cross-"process" persistence -------------------------------------


def test_persists_across_new_instance_and_cleared_lru(tmp_path):
    _skip_if_no_model2vec()
    from kaos_nlp_transformers import embedding as E

    first = _load(tmp_path)
    v1 = first.embed(_TEXTS)

    # Simulate a process restart: drop the instance, clear the in-proc LRU.
    del first
    E._embed_cache_clear()

    second = _load(tmp_path)
    second._backend = _ExplodingBackend()
    v2 = second.embed(_TEXTS)
    assert np.array_equal(v1, v2)


# -- (iv) model/revision/shape change never serves stale vectors -----------


def test_revision_change_namespaces_and_does_not_serve_stale(tmp_path):
    _skip_if_no_model2vec()
    from kaos_nlp_transformers.disk_cache import DiskEmbeddingCache

    text_hex = "deadbeef" * 4
    vec = np.arange(256, dtype=np.float32)

    rev_a = DiskEmbeddingCache(tmp_path, model_id=_MODEL, revision="rev-a", dim=256)
    rev_a.put(text_hex, vec)
    assert rev_a.get(text_hex) is not None

    # A different revision lands in a different namespace dir → miss.
    rev_b = DiskEmbeddingCache(tmp_path, model_id=_MODEL, revision="rev-b", dim=256)
    assert rev_b.get(text_hex) is None

    # A different model id → miss.
    other_model = DiskEmbeddingCache(tmp_path, model_id="other/model", revision="rev-a", dim=256)
    assert other_model.get(text_hex) is None


def test_dim_dtype_mismatch_is_a_miss(tmp_path):
    from kaos_nlp_transformers.disk_cache import DiskEmbeddingCache

    text_hex = "ab" * 16
    c256 = DiskEmbeddingCache(tmp_path, model_id=_MODEL, revision="r", dim=256)
    c256.put(text_hex, np.ones(256, dtype=np.float32))

    # Same model/revision but a different dim namespace → miss.
    c384 = DiskEmbeddingCache(tmp_path, model_id=_MODEL, revision="r", dim=384)
    assert c384.get(text_hex) is None

    # A file whose stored shape contradicts the namespace contract is
    # rejected on read (defends against a hand-corrupted cache dir).
    bad = c256.namespace_dir / f"{text_hex}.npy"
    np.save(bad, np.ones(99, dtype=np.float32))
    assert c256.get(text_hex) is None
    assert not bad.exists()  # corrupt entry is discarded


# -- (v) cached output bit-identical to uncached ---------------------------


def test_cached_output_bit_identical_to_uncached(tmp_path):
    _skip_if_no_model2vec()
    from kaos_nlp_transformers import EmbeddingModel
    from kaos_nlp_transformers import embedding as E
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

    # Uncached reference: no disk dir, no LRU.
    ref_model = EmbeddingModel.load(_MODEL, settings=KaosNLPTransformersSettings(offline=True))
    ref = ref_model.embed(_TEXTS)

    E._embed_cache_clear()
    cache_model = _load(tmp_path)
    cache_model.embed(_TEXTS)  # populate disk

    E._embed_cache_clear()
    read_back = _load(tmp_path)
    read_back._backend = _ExplodingBackend()
    cached = read_back.embed(_TEXTS)

    assert cached.dtype == np.float32
    assert np.array_equal(ref, cached)


def test_key_derivation_shared_with_in_proc_lru(tmp_path):
    """Disk and LRU keys derive from the same ``_hash_text``: a disk file
    name is the hex of the LRU digest for that text."""
    _skip_if_no_model2vec()
    from kaos_nlp_transformers import embedding as E

    model = _load(tmp_path)
    model.embed([_TEXTS[0]])
    files = list(tmp_path.rglob("*.npy"))
    assert len(files) == 1
    expected_hex = E._hash_text(_TEXTS[0]).hex()
    assert files[0].stem == expected_hex


# -- (vi) atomic-write resilience ------------------------------------------


def test_orphan_tmp_file_is_ignored(tmp_path):
    from kaos_nlp_transformers.disk_cache import DiskEmbeddingCache

    c = DiskEmbeddingCache(tmp_path, model_id=_MODEL, revision="r", dim=4)
    c.namespace_dir.mkdir(parents=True, exist_ok=True)
    # A crash mid-write leaves a *.tmp; it must never be read as an entry.
    orphan = c.namespace_dir / "feedface.12345.aabbccdd.tmp"
    orphan.write_bytes(b"partial junk")
    assert c.get("feedface") is None  # orphan is not a valid entry

    # A real write replaces atomically and reads back exactly.
    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    c.put("feedface", vec)
    got = c.get("feedface")
    assert got is not None
    assert np.array_equal(got, vec)
    # The successful write leaves no NEW tmp of its own; a pre-existing
    # orphan from a *different* writer is deliberately not touched (we
    # never clobber another process's in-flight temp file).
    assert list(c.namespace_dir.glob("*.tmp")) == [orphan]


def test_unregistered_zero_dim_skips_disk_cache(tmp_path):
    """Unregistered models have dim=0; the disk cache is skipped because
    the shape contract is unknown and entries could not be validated."""
    from kaos_nlp_transformers import EmbeddingModel
    from kaos_nlp_transformers.models import RegisteredModel

    reg = RegisteredModel(
        model_id="x/unregistered",
        revision="abc",
        license="UNKNOWN",
        params_m=0,
        dim=0,
        backend="ort",
        notes="unregistered",
    )
    model = EmbeddingModel(reg, _backend=object(), backend_name="ort", disk_cache=None)
    assert model._disk_cache is None
