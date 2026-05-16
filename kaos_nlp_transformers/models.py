"""Pinned model registry for kaos-nlp-transformers.

Every entry must carry an explicit revision SHA — never ``main``.
Every entry must declare a permissively-licensed model. The exclusion
list captures models that look attractive but have license problems
(CC-BY-NC, training-data ambiguity, etc.) and may not be added.

The registry is the binding contract — license review happens here, at
the point where a model becomes loadable. ``EmbeddingModel.load()``
checks the registry before delegating to the backend.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegisteredModel:
    """A model that has passed license review and is loadable in v0/v1."""

    model_id: str
    """Hugging Face Hub model id (org/repo)."""

    revision: str
    """Pinned commit SHA — NEVER 'main'. Min 7 chars."""

    license: str
    """SPDX-style license identifier (must be permissive)."""

    params_m: int
    """Approximate parameter count in millions."""

    dim: int
    """Embedding dimension produced by this model."""

    backend: str
    """Which backend supports this model: ``'ort'`` (Rust + libonnxruntime)
    or ``'model2vec'`` (static numpy lookup). Audit history:
    ``'sentence-transformers'`` retired in KNT-501 (0.1.0a6);
    ``'fastembed'`` retired in KNT-601 (0.2.0)."""

    notes: str = ""
    """Free-form notes (default model? legal-doc default? etc.)."""


# Embedding registry. Two model families covered in alpha:
#
# 1. fastembed — ONNX Runtime, CPU-friendly, the default for general retrieval.
#    Quality bench: BAAI/bge-small-en-v1.5 (33M, 384-dim, MIT). GPU
#    acceleration via the ``[gpu]`` extra (onnxruntime-gpu +
#    CUDAExecutionProvider).
#
# 2. model2vec — static lookup (vocab → vector + average), pure numpy at
#    inference, no torch, no ONNX. ~500x faster on CPU than the transformer
#    source. Quality bench (MTEB Retrieval): potion-retrieval-32M = 35.06
#    (~82% of all-MiniLM-L6-v2). Use for: first-pass retrieval over 100K+
#    docs, high-throughput dedup/clustering. Pair with a cross-encoder
#    reranker for final-pass quality.
#
# Audit-06 KNT-501 (0.1.0a6): the third "sentence-transformers" backend
# was retired. fastembed.TextCrossEncoder now serves the cross-encoder
# reranker via the same ONNX runtime as embedding does, so torch is no
# longer required anywhere in the package.
#
# Revision SHAs are validated against huggingface.co on every CI run by
# the optional ``test_registry_shas_exist_on_hub`` test (skipped offline).
# All SHAs were re-verified against huggingface.co/api/models/<id> on
# 2026-05-08 as part of the audit-04 sweep adding the model2vec entries.
REGISTRY: dict[str, RegisteredModel] = {
    "BAAI/bge-small-en-v1.5": RegisteredModel(
        model_id="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        license="MIT",
        params_m=33,
        dim=384,
        backend="ort",
        notes="Default v0 embedding model. CPU-friendly, English. Verified 2026-04-09.",
    ),
    "minishlab/potion-retrieval-32M": RegisteredModel(
        model_id="minishlab/potion-retrieval-32M",
        revision="6fc8051fab2a1e0ee76689cf08c853792ac285e7",
        license="MIT",
        params_m=32,
        # Matryoshka-trained at [32, 64, 128, 256, 512]; the on-disk vectors
        # are 512-dim and consumers can truncate at retrieval time. We pin
        # the full dim and document Matryoshka in the README rather than
        # branching the registry per-cut.
        dim=512,
        backend="model2vec",
        notes=(
            "Static retrieval-tuned distillation of bge-base-en-v1.5. "
            "MTEB Retrieval 35.06 (~82% of all-MiniLM-L6-v2) at >500x CPU "
            "throughput, ~30 MB on disk. Verified 2026-05-08. Requires the "
            "[model2vec] extra."
        ),
    ),
    "minishlab/potion-base-8M": RegisteredModel(
        model_id="minishlab/potion-base-8M",
        revision="bf8b056651a2c21b8d2565580b8569da283cab23",
        license="MIT",
        params_m=8,
        # Smaller potion variant — 256-dim PCA-reduced, ~30 MB safetensors,
        # ~31 MB total min subset (no ONNX). The "lightning-fast" entry-
        # point most blog posts reference. Lower MTEB scores than the 32M
        # siblings but small enough to vendor inside the wheel — see the
        # [bundled-static] extra (audit-05 KNT-401).
        dim=256,
        backend="model2vec",
        notes=(
            "Static general-purpose distillation of bge-base-en-v1.5, "
            "8M parameters, 256-dim. The smallest potion variant with "
            "respectable MTEB scores; ~31 MB on disk. Vendored in the "
            "wheel via [bundled-static] extra. Verified 2026-05-08."
        ),
    ),
    "minishlab/potion-base-32M": RegisteredModel(
        model_id="minishlab/potion-base-32M",
        revision="1e5a03f8eeb2c98b928fbbd846f22f816360919f",
        license="MIT",
        params_m=32,
        # potion-base is the general-purpose static distillation; same
        # 512-dim vectors as potion-retrieval but tuned for the average-
        # over-tasks MTEB score rather than retrieval specifically.
        dim=512,
        backend="model2vec",
        notes=(
            "Static general-purpose distillation of bge-base-en-v1.5. "
            "MTEB avg 51.66 (~95% of all-MiniLM-L6-v2). Use for "
            "classification / dedup / clustering; for retrieval pick "
            "potion-retrieval-32M instead. Verified 2026-05-08. Requires the "
            "[model2vec] extra."
        ),
    ),
}


# Hard exclusion list. These models are flagged by license audit and
# may not enter the registry under any circumstances. The reason
# string is shown to the user when they try to load an excluded model
# so the rejection is informative, not silent.
EXCLUDED: dict[str, str] = {
    # CC-BY-NC family — non-commercial only
    "jinaai/jina-embeddings-v3": "CC-BY-NC 4.0 (non-commercial)",
    "nvidia/NV-Embed-v1": "CC-BY-NC 4.0 (non-commercial)",
    "nvidia/NV-Embed-v2": "CC-BY-NC 4.0 (non-commercial)",
    # MS MARCO training-data ambiguity (commercial license unclear)
    "Qwen/Qwen3-Embedding-0.6B": "Trained on MS MARCO (commercial license unclear)",
    "Qwen/Qwen3-Embedding-4B": "Trained on MS MARCO (commercial license unclear)",
    "Qwen/Qwen3-Embedding-8B": "Trained on MS MARCO (commercial license unclear)",
}


# ---------------------------------------------------------------------------
# Reranker registry (audit-02 KNT-104)
# ---------------------------------------------------------------------------

# Audit-02 KNT-104: rerankers go through the same license / revision /
# offline policy as embeddings. The reranker shape mirrors RegisteredModel
# but lives in its own dict so task-specific defaults stay clear and the
# embedding registry can never accidentally be used to load a reranker
# (or vice versa).
#
# Revision SHAs verified against huggingface.co/api/models/<id> on
# 2026-05-08; confirmation procedure documented in the model expansion
# checklist.
RERANKER_REGISTRY: dict[str, RegisteredModel] = {
    "BAAI/bge-reranker-base": RegisteredModel(
        model_id="BAAI/bge-reranker-base",
        revision="2cfc18c9415c912f9d8155881c133215df768a70",
        license="MIT",
        params_m=278,
        # Cross-encoders return a single relevance score per (query, passage)
        # pair, not a vector — dim is recorded as 1 for shape symmetry with
        # RegisteredModel rather than as an embedding dimension.
        dim=1,
        # Audit-06 KNT-501: was "sentence-transformers" pre-0.1.0a6;
        # fastembed.TextCrossEncoder now serves this same model via ONNX,
        # so the registered backend is fastembed.
        backend="ort",
        notes="Default v0 reranker. CPU-friendly cross-encoder. Verified 2026-05-08.",
    ),
}


# Same shape as EXCLUDED but for rerankers. Currently empty — re-add
# entries here as license / data-licensing concerns surface.
RERANKER_EXCLUDED: dict[str, str] = {}


# Curated NLI (natural language inference) cross-encoders. Same shape
# as ``RERANKER_REGISTRY``: a cross-encoder over ``(premise, hypothesis)``
# pairs, but the head returns three logits (entailment / neutral /
# contradiction) instead of one relevance score. ``dim`` is recorded
# as 3 for the three-class output.
#
# The default entry uses ``Xenova/nli-deberta-v3-base`` — a pure 🤗
# Optimum re-export of the Apache-2.0 ``cross-encoder/nli-deberta-v3-base``
# upstream, chosen because the Xenova fork ships the full ONNX
# quantization matrix (fp32 / fp16 / int8 / quantized / q4 / uint8)
# whereas the upstream only ships an AVX-512-VNNI quantized variant
# (incompatible with CPUs that don't have the VNNI extension). The
# Xenova fork has no LICENSE file in the repo; the license chain is
# documented in ``notes`` so a counsel audit can follow it.
#
# Revision SHA verified against huggingface.co/api/models/{id} on
# 2026-05-15.
NLI_REGISTRY: dict[str, RegisteredModel] = {
    "Xenova/nli-deberta-v3-base": RegisteredModel(
        model_id="Xenova/nli-deberta-v3-base",
        revision="80a99030ce45a69a39ea2a6f50756d03859ff521",
        license="Apache-2.0",
        params_m=184,
        # Three-class head: entailment / neutral / contradiction.
        dim=3,
        backend="ort",
        notes=(
            "Default v0 NLI cross-encoder. Apache-2.0 chain: upstream "
            "weights ship as cross-encoder/nli-deberta-v3-base (Apache-2.0); "
            "Xenova fork is a pure 🤗 Optimum ONNX re-export with no "
            "fine-tuning added. Training data: SNLI + MultiNLI (both "
            "academically permissive). MNLI mismatched dev accuracy 90.04 / "
            "SNLI dev 92.38. Loaded variant is the portable 244 MB "
            "onnx/model_quantized.onnx — matches the bge-reranker-base "
            "quantized-by-default precedent. Verified 2026-05-15."
        ),
    ),
}


# Models considered for the NLI registry but excluded — typically
# blocked by training-data license issues (CC-BY-NC contamination)
# even when the model weights themselves are MIT/Apache-2.0. Mirrors
# the pattern in ``EXCLUDED`` for embeddings.
NLI_EXCLUDED: dict[str, str] = {
    "Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": (
        "Training data 'mil7' includes NC-licensed components (flagged on "
        "the model card). Multilingual NLI deferred until a clean-data "
        "alternative ships ONNX — most likely MoritzLaurer/"
        "mDeBERTa-v3-base-mnli-xnli (MIT, MNLI+XNLI only) once an ONNX "
        "export is vendored."
    ),
}


# Curated GLiNER (zero-shot NER via span extraction) checkpoints.
# Same shape as the other registries: a transformer cross-encoder over
# ``[ENT] label_1 [ENT] label_2 ... [SEP] text`` input, decoding
# (start_token, end_token, label_idx, score) spans from the model's
# output tensor. ``dim`` is recorded as 0 because GLiNER's output
# shape is span-and-label-dependent rather than a fixed embedding
# dimension; the actual head shape is decoded in
# ``core::ner::OrtGlinerExtractor``.
#
# The default ``onnx-community/gliner_medium-v2.1`` is the ONNX
# re-export of the original ``urchade/gliner_medium-v2.1``
# (Apache-2.0, DeBERTa-v3-base backbone, 195M params). Quantized
# variant is 243 MiB on disk — matches the "quantized-by-default"
# precedent of the reranker and NLI checkpoints.
#
# License chain documented in ``notes`` for counsel audit: the
# onnx-community fork has no LICENSE file in the repo, but the model
# card YAML declares ``base_model: urchade/gliner_medium-v2.1`` which
# is tagged Apache-2.0 on the Hub. The ONNX export is a derivative
# weight format of an Apache-2.0 work; redistribution is permitted
# under §4 of Apache-2.0 with upstream attribution preserved.
#
# Revision SHAs verified against huggingface.co/api/models/{id}/revision/{sha}
# on 2026-05-15 (agent-verified L1 model-hunt).
NER_REGISTRY: dict[str, RegisteredModel] = {
    "onnx-community/gliner_medium-v2.1": RegisteredModel(
        model_id="onnx-community/gliner_medium-v2.1",
        revision="959437589dc623d4c0a93f6e2828213567929cde",
        license="Apache-2.0",
        params_m=195,
        # GLiNER output shape is (batch, n_spans, n_labels); the
        # per-token head isn't a fixed embedding dim. We record 0
        # here for the registry shape and decode the actual shape
        # in core::ner. (Mirror of bge-reranker-base which records
        # dim=1 even though its head is scalar.)
        dim=0,
        backend="ort",
        notes=(
            "Default v0 GLiNER (English) zero-shot NER. Apache-2.0 chain: "
            "upstream weights ship as urchade/gliner_medium-v2.1 "
            "(Apache-2.0, DeBERTa-v3-base backbone, 195M params); "
            "onnx-community fork is a pure 🤗 Optimum / transformers.js "
            "ONNX re-export with no fine-tuning added. Loaded variant is "
            "the fp32 onnx/model.onnx (~746 MiB). The int8 "
            "model_quantized.onnx export was tested and rejected — its "
            "scores cap around 0.13 on examples where the PyTorch "
            "reference scores 0.99, so it produces zero spans at the "
            "default threshold. Verified 2026-05-15."
        ),
    ),
    "onnx-community/gliner_multi-v2.1": RegisteredModel(
        model_id="onnx-community/gliner_multi-v2.1",
        revision="6ddaeb9413b0e71ad8457da1aab378a165b24058",
        license="Apache-2.0",
        params_m=205,
        dim=0,
        backend="ort",
        notes=(
            "Multilingual GLiNER. Apache-2.0 chain: upstream "
            "urchade/gliner_multi-v2.1 (Apache-2.0, mDeBERTa-v3-base "
            "backbone); onnx-community fork is the standard ONNX "
            "re-export. Same gliner config + tokenizer family as the "
            "English medium variant, so a single core::ner code path "
            "covers both. Loaded variant is onnx/model.onnx (~1.08 GiB "
            "fp32); same quantization concern as the English variant "
            "applies — int8 export is unusable at default threshold. "
            "Verified 2026-05-15."
        ),
    ),
}


# Hard exclusion list for NER models. Same pattern as ``EXCLUDED`` for
# embeddings — license-blocked checkpoints flagged so future agents
# don't try to add them.
NER_EXCLUDED: dict[str, str] = {
    "urchade/gliner_base": "CC-BY-NC 4.0 (non-commercial) — upstream's smallest GLiNER variant",
    "onnx-community/gliner_base": (
        "Inherits CC-BY-NC 4.0 from upstream urchade/gliner_base. "
        "The onnx-community fork is just an ONNX re-export and does "
        "not relicense the weights."
    ),
}


# Curated PII (personally-identifiable-information) token-classifier
# models. Architecture differs from GLiNER: closed-label BERT-style
# token classifier with BIO encoding, ~10x faster than zero-shot
# span enumeration for fixed PII categories. The label set is baked
# into the model's ``config.json::id2label`` field, not supplied at
# inference time.
#
# Default model uses the ``onnx-community/bert-small-pii-detection-ONNX``
# 🤗 Optimum re-export of ``gravitee-io/bert-small-pii-detection``
# (Apache-2.0 chain — the ONNX repo card declares
# ``license: apache-2.0`` directly). 28M-param BERT-small (4-layer,
# 512 hidden, 512 context). Trained on ``beki/privy`` +
# ``gretelai/synthetic_pii_finance_multilingual`` + ``eriktks/conll2003``.
#
# 24 PII categories (each in B-/I- form): AGE, COORDINATE,
# CREDIT_CARD, DATE_TIME, EMAIL_ADDRESS, FINANCIAL, IBAN_CODE, IMEI,
# IP_ADDRESS, LOCATION, MAC_ADDRESS, NRP (nationality / religious /
# political), ORGANIZATION, PASSWORD, PERSON, PHONE_NUMBER, TITLE,
# URL, US_BANK_NUMBER, US_DRIVER_LICENSE, US_ITIN, US_LICENSE_PLATE,
# US_PASSPORT, US_SSN. Plus the ``O`` (outside) class.
#
# Revision SHA verified against huggingface.co/api/models/{id} on
# 2026-05-16.
PII_REGISTRY: dict[str, RegisteredModel] = {
    "onnx-community/bert-small-pii-detection-ONNX": RegisteredModel(
        model_id="onnx-community/bert-small-pii-detection-ONNX",
        revision="6cb4e77c2b2c7f81e731b88cffa9b7a6fc675a4c",
        license="Apache-2.0",
        params_m=28,
        # BERT-style token classifier outputs (batch, seq, num_classes)
        # logits — 49 classes total (O + 24 categories x 2 BIO prefixes).
        # ``dim`` records the number of classes for shape symmetry with
        # other registries.
        dim=49,
        backend="ort",
        notes=(
            "Default v0 PII (personally-identifiable-information) token "
            "classifier. Apache-2.0 chain: the onnx-community fork "
            "declares ``license: apache-2.0`` directly in its card "
            "YAML; upstream ``gravitee-io/bert-small-pii-detection`` is "
            "also Apache-2.0. 28M-param BERT-small (4-layer, 512 "
            "hidden); int8-quantized variant is the default load "
            "(27 MB). 24 PII categories covering general + US-specific "
            "financial PII (SSN, ITIN, IBAN, credit card, bank "
            "number). Complements GLiNER by serving the closed-label "
            "PII case roughly 10x faster than zero-shot span enumeration. "
            "Verified 2026-05-16."
        ),
    ),
}


# Hard exclusion list for PII / token-classifier models. Mirrors the
# pattern in other registries.
PII_EXCLUDED: dict[str, str] = {
    "urchade/gliner_multi_pii-v1": (
        "CC-BY-NC 4.0 (non-commercial). The 'obvious' PII GLiNER is "
        "blocked by training-data licensing; use the bert-small "
        "alternative above for closed-label PII, or fall back to "
        "GLiNER with custom labels for zero-shot domain-specific PII."
    ),
    "ai4privacy/pii-masking-200k": (
        "Training dataset is research-only (no clear commercial-use "
        "grant). Models fine-tuned on this corpus inherit the "
        "restriction; check individual model cards before adding."
    ),
}


__all__ = [
    "EXCLUDED",
    "NER_EXCLUDED",
    "NER_REGISTRY",
    "NLI_EXCLUDED",
    "NLI_REGISTRY",
    "PII_EXCLUDED",
    "PII_REGISTRY",
    "REGISTRY",
    "RERANKER_EXCLUDED",
    "RERANKER_REGISTRY",
    "RegisteredModel",
]
