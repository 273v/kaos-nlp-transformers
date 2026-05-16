//! GLiNER (zero-shot NER) extraction via prompt-based span scoring.
//!
//! The pipeline is a Rust port of the `gline-rs` span-mode pipeline
//! (Apache-2.0, https://github.com/fbilhaut/gline-rs), inlined here
//! rather than added as a crate dep because gline-rs pins ``ort
//! 2.0.0-rc.9`` / ``tokenizers 0.21`` / ``ndarray 0.16`` and depends
//! on a git-only sibling ``orp``. Reading its source is fine; linking
//! against it would force a version-skew diamond.
//!
//! ## Algorithm
//!
//! 1. **Word-split** the input text via the regex
//!    ``\w+(?:[-_]\w+)*|\S`` so every word has a byte-level
//!    ``(start, end)`` slice we can resolve back to the original text.
//! 2. **Build a prompt** of the form
//!    ``[<<ENT>>, label_1, <<ENT>>, label_2, ..., <<SEP>>, w_1, w_2, ..., w_n]``
//!    as a flat list of strings.
//! 3. **Sub-word encode** each word individually via the HF tokenizer
//!    (one Encoding per word). The flat ``input_ids`` row is then
//!    ``[CLS=1] + flat_subword_ids + [SEP=2] + padding(=0)`` — the
//!    GLiNER training contract hard-codes those bracket ids regardless
//!    of what the tokenizer's special-tokens config says.
//! 4. **Word mask**: a per-token id where the value at the FIRST
//!    subword of word k (zero-indexed past the entity-label region) is
//!    ``k+1``, and 0 everywhere else (including all continuation
//!    subwords, padding, and the prompt's label tokens). The model
//!    uses this to pool subwords back into word-level hidden states.
//! 5. **Span enumeration**: build ``span_idx[s, i, 0..1] = (start_word,
//!    start_word + width)`` and ``span_mask[s, i] = true/false`` over
//!    every ``(start_word, width)`` pair with start in
//!    ``[0, text_lengths[s])`` and width in
//!    ``[0, min(max_width, remaining)]``.
//! 6. **ort session run** with 6 named inputs (``input_ids``,
//!    ``attention_mask``, ``words_mask``, ``text_lengths``,
//!    ``span_idx``, ``span_mask``) → logits with shape
//!    ``(batch, sequence_length, num_spans, num_classes)``.
//!    NB: the model's ONNX export labels the second axis as
//!    ``sequence_length``, but the values are actually indexed by
//!    *start-word index* over ``[0, num_words)`` — the gline-rs
//!    decoder reads it as ``logits[s, start_word, width, class]``.
//! 7. **Decode** each cell whose ``sigmoid(score) >= threshold`` into
//!    an ``Entity`` at the byte offsets that the (start_word,
//!    start_word + width) words occupy in the original text.
//! 8. **Greedy search** (after sorting by start, end ascending) to
//!    enforce non-overlap if ``flat_ner = true``.

use crate::core::device::Device;
use crate::core::error::{BackendError, Result};
use crate::core::model_loader::{resolve_paths, ModelPaths};
use crate::core::model_registry::RegisteredModel;
use ndarray::{Array2, Array3, Array4};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use regex::Regex;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;

// -----------------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------------

/// A decoded named entity span. Byte offsets are into the original
/// input text. ``score`` is sigmoid-normalized in ``[0, 1]``.
#[derive(Debug, Clone)]
pub struct Entity {
    /// Index of the input sequence this span came from.
    pub sequence: usize,
    /// Byte offset of the first character in the source text.
    pub start: usize,
    /// Byte offset just past the last character (exclusive).
    pub end: usize,
    /// Substring of the source text from ``start`` to ``end``.
    pub text: String,
    /// The entity-class label string the user supplied in the
    /// ``labels`` list at extract time.
    pub label: String,
    /// Sigmoid-normalized score in ``[0, 1]``.
    pub score: f32,
}

/// Tunable parameters for GLiNER extraction. Mirrors gline-rs
/// `Parameters` with the same defaults (`threshold=0.5`,
/// `max_width=12`, `flat_ner=true`, `dup_label=false`,
/// `multi_label=false`).
#[derive(Debug, Clone, Copy)]
pub struct ExtractParams {
    /// Sigmoid score threshold for accepting a span.
    pub threshold: f32,
    /// Maximum width (number of words) of an enumerated span.
    pub max_width: usize,
    /// If true, no two output spans may overlap. If false, overlap is
    /// allowed subject to `dup_label` / `multi_label`.
    pub flat_ner: bool,
    /// If `flat_ner=false`: allow overlapping spans with the same label.
    pub dup_label: bool,
    /// If `flat_ner=false`: allow overlapping spans with different labels.
    pub multi_label: bool,
}

impl Default for ExtractParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            max_width: 12,
            flat_ner: true,
            dup_label: false,
            multi_label: false,
        }
    }
}

/// NER extractor trait. One Rust-side backend implementation
/// (`OrtGlinerExtractor`) is shipped today; future backends (e.g.
/// `ner-tiny` for small CPU footprints) would also implement this
/// surface.
pub trait NerExtractor: Send + Sync {
    /// Run extraction over a batch of input texts against the given
    /// label list. Returns one ``Vec<Entity>`` per input text.
    fn extract(
        &self,
        texts: &[&str],
        labels: &[&str],
        params: ExtractParams,
    ) -> Result<Vec<Vec<Entity>>>;

    /// HF Hub model id this extractor was loaded for.
    fn model_id(&self) -> &str;

    /// Device this extractor runs on.
    fn device(&self) -> &str;
}

// -----------------------------------------------------------------------------
// ort-backed GLiNER extractor
// -----------------------------------------------------------------------------

/// Hard-coded GLiNER bracket-token ids — the model is trained with
/// these specific ids and does not look them up from the tokenizer's
/// special-tokens table. Same constants as gline-rs.
const CLS_TOKEN_ID: i64 = 1;
const SEP_TOKEN_ID: i64 = 2;
const PAD_TOKEN_ID: i64 = 0;

/// Prompt label-marker tokens. These ARE expected to be in the
/// tokenizer's added-tokens table at the SHA we pin (verified for
/// onnx-community/gliner_medium-v2.1).
const ENT_TOKEN_STR: &str = "<<ENT>>";
const SEP_TOKEN_STR: &str = "<<SEP>>";

/// Default word-level splitter regex — words including internal
/// dashes/underscores, falling back to any non-whitespace single
/// character (punctuation). Same as gline-rs's default.
const SPLITTER_REGEX: &str = "\\w+(?:[-_]\\w+)*|\\S";

fn splitter() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(SPLITTER_REGEX).expect("static regex"))
}

/// ort-backed GLiNER extractor.
pub struct OrtGlinerExtractor {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    model_id: String,
    device_str: String,
    max_seq_len: usize,
}

impl OrtGlinerExtractor {
    /// Load a registered GLiNER model.
    pub fn load(
        model: &'static RegisteredModel,
        device: &Device,
        cache_dir: Option<&Path>,
    ) -> Result<Self> {
        let ModelPaths { onnx, tokenizer } = resolve_paths(model, cache_dir)?;

        let builder = Session::builder().map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("Session::builder(): {e}"),
            )
        })?;
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                BackendError::model_load(
                    model.model_id,
                    model.revision,
                    format!("optimization level: {e}"),
                )
            })?;

        // KNT-NLI-002 (2026-05-16): override intra-op thread count
        // from the env var ``KAOS_NLP_TRANSFORMERS_INTRA_THREADS``
        // when set. Otherwise leave ort's default in place — on a
        // 20-core host ort defaults to ~5 active threads for small
        // batches, which empirically beat both 1-thread and
        // 20-thread settings on the GLiNER short-sequence workload
        // (the thread-sync cost on tiny ops swamps the parallelism
        // win). Long-sequence / batched workloads can override.
        let builder = configure_intra_threads(builder, model)?;

        let mut builder = configure_eps(builder, device, model)?;

        let session = builder.commit_from_file(&onnx).map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("commit_from_file({}): {e}", onnx.display()),
            )
        })?;

        // GLiNER tokenizer is loaded raw — no padding/truncation
        // policy applied. We encode each WORD individually below and
        // concatenate manually with the hard-coded CLS/SEP bracket
        // ids, so the BERT-style batch-longest plumbing in
        // ``TokenizerWrapper`` would interfere.
        let tokenizer = Tokenizer::from_file(&tokenizer).map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("Tokenizer::from_file({}): {e}", tokenizer.display()),
            )
        })?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            model_id: model.model_id.to_string(),
            device_str: device.as_str().to_string(),
            max_seq_len: model.max_seq_len,
        })
    }
}

impl NerExtractor for OrtGlinerExtractor {
    fn extract(
        &self,
        texts: &[&str],
        labels: &[&str],
        params: ExtractParams,
    ) -> Result<Vec<Vec<Entity>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if labels.is_empty() {
            return Err(BackendError::inference(
                "labels must be non-empty for GLiNER extraction",
            ));
        }
        if params.max_width == 0 {
            return Err(BackendError::inference("max_width must be >= 1"));
        }

        // 1. Word-split each input text.
        let word_tokens: Vec<Vec<WordToken>> =
            texts.iter().map(|t| split_words(t)).collect::<Vec<_>>();

        // 2 + 3. Build prompt per sequence, sub-word encode each word.
        let encoded = encode_prompts(&self.tokenizer, &word_tokens, labels)?;

        // Bail before tensor work if every input was empty after
        // word-splitting — the model can't run on a zero-width batch.
        if encoded.num_words == 0 {
            return Ok(vec![Vec::new(); texts.len()]);
        }

        // 4. Span enumeration tensors.
        let (span_idx, span_mask) = make_span_tensors(&encoded.text_lengths, params.max_width);

        // 5. Run ort session.
        let logits = self.run_session(
            &encoded.input_ids,
            &encoded.attention_mask,
            &encoded.words_mask,
            &encoded.text_lengths,
            &span_idx,
            &span_mask,
            labels.len(),
        )?;

        // 6. Decode + threshold per sequence.
        let mut out = Vec::with_capacity(texts.len());
        for (s, text) in texts.iter().enumerate() {
            let raw_spans =
                decode_spans(&logits, s, text, &word_tokens[s], labels, params.threshold)?;
            // Sort by (start, end) ascending so greedy-search sees them
            // in offset order.
            let mut sorted = raw_spans;
            sorted.sort_by(|a, b| a.start.cmp(&b.start).then(a.end.cmp(&b.end)));

            // 7. Greedy non-overlap filter.
            let filtered = greedy_search(&sorted, params);
            out.push(filtered);
        }

        Ok(out)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn device(&self) -> &str {
        &self.device_str
    }
}

// -----------------------------------------------------------------------------
// Thread + EP plumbing (mirror of reranker.rs / nli.rs)
// -----------------------------------------------------------------------------

/// Apply the optional ``KAOS_NLP_TRANSFORMERS_INTRA_THREADS`` env
/// override to ``with_intra_threads``. When the env var is unset or
/// unparseable, we leave the builder alone so ort picks its own
/// default. On a 20-core host that default lands at ~5 active
/// threads — empirically optimal for the short-sequence GLiNER /
/// NLI workload; the thread-sync overhead on tiny matmuls exceeds
/// the parallelism win from saturating all cores. Long-sequence or
/// batched workloads can override via the env var.
fn configure_intra_threads(
    builder: ort::session::builder::SessionBuilder,
    model: &RegisteredModel,
) -> Result<ort::session::builder::SessionBuilder> {
    let override_n: Option<usize> = std::env::var("KAOS_NLP_TRANSFORMERS_INTRA_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0);
    match override_n {
        None => Ok(builder),
        Some(n) => builder.with_intra_threads(n).map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("with_intra_threads({n}): {e}"),
            )
        }),
    }
}

fn configure_eps(
    builder: ort::session::builder::SessionBuilder,
    device: &Device,
    model: &RegisteredModel,
) -> Result<ort::session::builder::SessionBuilder> {
    match device {
        Device::Cpu => Ok(builder),
        Device::Cuda(_) => Err(BackendError::BackendNotInstalled(format!(
            "GLiNER inference for {} requested on {:?}; CUDA execution requires the [gpu] companion wheel",
            model.model_id, device
        ))),
        Device::OpenVino => Err(BackendError::BackendNotInstalled(format!(
            "GLiNER inference for {} requested on {:?}; OpenVINO execution requires the [openvino] companion wheel",
            model.model_id, device
        ))),
    }
}

// -----------------------------------------------------------------------------
// Word-level splitting
// -----------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct WordToken {
    /// Byte offset of the first character — for Rust ``&str`` slicing.
    byte_start: usize,
    /// Byte offset just past the last character (exclusive).
    byte_end: usize,
    /// Character (codepoint) offset of the first character — for
    /// Python callers (``str[start:end]`` indexes codepoints, not
    /// bytes). The emitted ``Entity`` carries char offsets so the
    /// Python wrapper's round-trip ``text[e.start:e.end] == e.text``
    /// is correct on multi-byte content like curly quotes / em-dashes.
    char_start: usize,
    /// Character (codepoint) offset just past the last character.
    char_end: usize,
    /// The literal word text.
    text: String,
}

fn split_words(text: &str) -> Vec<WordToken> {
    // The `regex` crate gives us byte offsets. Build a byte→char map
    // by walking the text's ``char_indices`` once, so each WordToken
    // carries both views.
    let matches: Vec<regex::Match<'_>> = splitter().find_iter(text).collect();
    if matches.is_empty() {
        return Vec::new();
    }

    let mut byte_to_char: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(matches.len() * 2 + 1);
    let mut char_idx: usize = 0;
    for (byte_idx, _) in text.char_indices() {
        byte_to_char.insert(byte_idx, char_idx);
        char_idx += 1;
    }
    // End-of-string sentinel — the regex's m.end() can equal text.len().
    byte_to_char.insert(text.len(), char_idx);

    matches
        .into_iter()
        .map(|m: regex::Match<'_>| {
            let byte_start = m.start();
            let byte_end = m.end();
            WordToken {
                byte_start,
                byte_end,
                char_start: *byte_to_char.get(&byte_start).expect("byte offset in map"),
                char_end: *byte_to_char.get(&byte_end).expect("byte offset in map"),
                text: m.as_str().to_string(),
            }
        })
        .collect()
}

// -----------------------------------------------------------------------------
// Prompt encoding (per-word subword tokenization + flatten)
// -----------------------------------------------------------------------------

struct EncodedBatch {
    /// Shape ``(batch, num_tokens)``. int64 token ids.
    input_ids: Array2<i64>,
    /// Shape ``(batch, num_tokens)``. 1/0 mask.
    attention_mask: Array2<i64>,
    /// Shape ``(batch, num_tokens)``. word-id-or-zero per token.
    words_mask: Array2<i64>,
    /// Shape ``(batch, 1)``. Number of words in the text portion.
    text_lengths: Array2<i64>,
    /// Maximum number of words across all sequences.
    num_words: usize,
}

fn encode_prompts(
    tokenizer: &Tokenizer,
    word_tokens: &[Vec<WordToken>],
    labels: &[&str],
) -> Result<EncodedBatch> {
    // Per-sequence sub-word encodings: prompt is
    //   [ENT, label_1, ENT, label_2, ..., SEP, w_1, w_2, ..., w_n]
    // We encode every prompt-element string INDIVIDUALLY so we can:
    //   - track which encoding ranges belong to text vs labels
    //   - emit a words_mask that fires only on the first subword of
    //     each text word (matching the GLiNER training contract).

    let mut per_seq: Vec<EncodedPrompt> = Vec::with_capacity(word_tokens.len());
    let mut max_tokens: usize = 0;
    let mut max_words: usize = 0;

    for words in word_tokens {
        // Build the prompt as a flat sequence of strings.
        let mut prompt_tokens: Vec<String> = Vec::with_capacity(labels.len() * 2 + 1 + words.len());
        for label in labels {
            prompt_tokens.push(ENT_TOKEN_STR.to_string());
            prompt_tokens.push((*label).to_string());
        }
        prompt_tokens.push(SEP_TOKEN_STR.to_string());
        let entities_len = prompt_tokens.len(); // index where the text part starts
        for word in words {
            prompt_tokens.push(word.text.clone());
        }

        // Sub-word encode each prompt element individually. Use
        // ``add_special_tokens = false`` because we manage the
        // CLS/SEP bracket ids manually.
        let mut per_word_ids: Vec<Vec<u32>> = Vec::with_capacity(prompt_tokens.len());
        // Pre-count total subword tokens including the two
        // hard-coded CLS/SEP ids.
        let mut total_tokens: usize = 2;
        let mut total_entity_tokens: usize = 0;
        for (i, element) in prompt_tokens.iter().enumerate() {
            let enc = tokenizer
                .encode_fast(element.as_str(), false)
                .map_err(BackendError::tokenization)?;
            let ids: Vec<u32> = enc.get_ids().to_vec();
            total_tokens += ids.len();
            if i < entities_len {
                total_entity_tokens += ids.len();
            }
            per_word_ids.push(ids);
        }

        // Offset of the first text subword in the eventual flat
        // sequence: entity-part size + 1 for the leading CLS.
        let text_offset = total_entity_tokens + 1;
        per_seq.push(EncodedPrompt {
            per_word_ids,
            text_offset,
            text_word_count: words.len(),
        });
        max_tokens = max_tokens.max(total_tokens);
        max_words = max_words.max(words.len());
    }

    // Now flatten each per-sequence encoding into row vectors of length
    // ``max_tokens``, padded with PAD_TOKEN_ID.
    let batch_size = per_seq.len();
    let mut input_ids = Array2::<i64>::zeros((batch_size, max_tokens));
    let mut attention_mask = Array2::<i64>::zeros((batch_size, max_tokens));
    let mut words_mask = Array2::<i64>::zeros((batch_size, max_tokens));
    let mut text_lengths = Array2::<i64>::zeros((batch_size, 1));

    for (s, enc) in per_seq.iter().enumerate() {
        let mut idx: usize = 0;
        let mut word_id: i64 = 0; // 1-based once we emit; starts at 0 to pre-increment

        // CLS
        input_ids[[s, idx]] = CLS_TOKEN_ID;
        attention_mask[[s, idx]] = 1;
        // words_mask stays 0 for CLS
        idx += 1;

        for word_ids in &enc.per_word_ids {
            for (sub_pos, token_id) in word_ids.iter().enumerate() {
                if idx >= max_tokens {
                    // Should not happen — max_tokens was computed from
                    // exactly this enumeration. Defensive cap to avoid
                    // panic; the model would still get the slice it
                    // expects.
                    break;
                }
                input_ids[[s, idx]] = *token_id as i64;
                attention_mask[[s, idx]] = 1;
                // words_mask = word_id + 1 only on the FIRST subword
                // of each text word (past the entity-label region).
                if idx >= enc.text_offset && sub_pos == 0 {
                    words_mask[[s, idx]] = word_id + 1;
                }
                idx += 1;
            }
            // Bump the word counter once we've emitted a whole text
            // word's subwords.
            if idx > enc.text_offset {
                word_id += 1;
            }
        }

        // SEP
        if idx < max_tokens {
            input_ids[[s, idx]] = SEP_TOKEN_ID;
            attention_mask[[s, idx]] = 1;
        }
        // padding (PAD_TOKEN_ID = 0) is already in the Array2::zeros init.
        let _ = PAD_TOKEN_ID; // explicit reference for readers; init is zero.

        text_lengths[[s, 0]] = enc.text_word_count as i64;
    }

    Ok(EncodedBatch {
        input_ids,
        attention_mask,
        words_mask,
        text_lengths,
        num_words: max_words,
    })
}

struct EncodedPrompt {
    per_word_ids: Vec<Vec<u32>>,
    text_offset: usize,
    text_word_count: usize,
}

// -----------------------------------------------------------------------------
// Span enumeration tensors
// -----------------------------------------------------------------------------

/// Build the ``(span_idx, span_mask)`` tensors. ``span_idx`` is shape
/// ``(batch, num_words * max_width, 2)`` with (start_word, end_word)
/// pairs (inclusive endpoints). ``span_mask`` is shape
/// ``(batch, num_words * max_width)`` of bools — true iff the span is
/// real (vs padding).
///
/// num_words is derived from text_lengths in two ways: the SHARED
/// dimension across the batch must be the MAX of all sequences'
/// text_lengths (so the span-enumeration tensor is rectangular).
fn make_span_tensors(text_lengths: &Array2<i64>, max_width: usize) -> (Array3<i64>, Array2<bool>) {
    let batch = text_lengths.shape()[0];

    // Compute the batch-wide num_words (max over sequences).
    let mut num_words: usize = 0;
    for s in 0..batch {
        let n = text_lengths[[s, 0]] as usize;
        num_words = num_words.max(n);
    }

    let num_spans = num_words * max_width;
    let mut span_idx = Array3::<i64>::zeros((batch, num_spans, 2));
    let mut span_mask = Array2::<bool>::from_elem((batch, num_spans), false);

    for s in 0..batch {
        let text_width = text_lengths[[s, 0]] as usize;
        for start in 0..text_width {
            let remaining = text_width - start;
            let actual_max = max_width.min(remaining);
            for width in 0..actual_max {
                let dim = start * max_width + width;
                if dim < num_spans {
                    span_idx[[s, dim, 0]] = start as i64;
                    span_idx[[s, dim, 1]] = (start + width) as i64;
                    span_mask[[s, dim]] = true;
                }
            }
        }
    }

    (span_idx, span_mask)
}

// -----------------------------------------------------------------------------
// ort session execution
// -----------------------------------------------------------------------------

impl OrtGlinerExtractor {
    #[allow(clippy::too_many_arguments)]
    fn run_session(
        &self,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
        words_mask: &Array2<i64>,
        text_lengths: &Array2<i64>,
        span_idx: &Array3<i64>,
        span_mask: &Array2<bool>,
        num_classes: usize,
    ) -> Result<Array4<f32>> {
        let mut session = self
            .session
            .lock()
            .map_err(|e| BackendError::inference(format!("session mutex poisoned: {e}")))?;

        let input_ids_view = input_ids.view();
        let attention_view = attention_mask.view();
        let words_mask_view = words_mask.view();
        let text_lengths_view = text_lengths.view();
        let span_idx_view = span_idx.view();
        let span_mask_view = span_mask.view();

        let inputs = ort::inputs! {
            "input_ids" => TensorRef::from_array_view(input_ids_view).map_err(|e| BackendError::inference(format!("input_ids tensor: {e}")))?,
            "attention_mask" => TensorRef::from_array_view(attention_view).map_err(|e| BackendError::inference(format!("attention_mask tensor: {e}")))?,
            "words_mask" => TensorRef::from_array_view(words_mask_view).map_err(|e| BackendError::inference(format!("words_mask tensor: {e}")))?,
            "text_lengths" => TensorRef::from_array_view(text_lengths_view).map_err(|e| BackendError::inference(format!("text_lengths tensor: {e}")))?,
            "span_idx" => TensorRef::from_array_view(span_idx_view).map_err(|e| BackendError::inference(format!("span_idx tensor: {e}")))?,
            "span_mask" => TensorRef::from_array_view(span_mask_view).map_err(|e| BackendError::inference(format!("span_mask tensor: {e}")))?,
        };

        let outputs = session
            .run(inputs)
            .map_err(|e| BackendError::inference(format!("ort session.run: {e}")))?;

        let logits_value = outputs.get("logits").ok_or_else(|| {
            BackendError::inference("expected 'logits' output not found in ort outputs")
        })?;

        let (shape, raw) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(|e| BackendError::inference(format!("extract logits: {e}")))?;

        // Expected shape: (batch, num_words, max_width, num_classes).
        if shape.len() != 4 {
            return Err(BackendError::inference(format!(
                "logits has shape {:?}, expected 4D (batch, num_words, max_width, num_classes)",
                shape
            )));
        }
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if dims[3] != num_classes {
            return Err(BackendError::inference(format!(
                "logits last dim {} != num_classes {}",
                dims[3], num_classes
            )));
        }

        let flat: Vec<f32> = raw.to_vec();
        let arr = Array4::from_shape_vec((dims[0], dims[1], dims[2], dims[3]), flat)
            .map_err(|e| BackendError::inference(format!("logits reshape: {e}")))?;
        Ok(arr)
    }
}

// -----------------------------------------------------------------------------
// Span decoding
// -----------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn decode_spans(
    logits: &Array4<f32>,
    sequence_id: usize,
    text: &str,
    words: &[WordToken],
    labels: &[&str],
    threshold: f32,
) -> Result<Vec<Entity>> {
    let dims = logits.shape();
    let num_words = words.len();
    let max_width = dims[2];
    let num_classes = dims[3];

    if num_classes != labels.len() {
        return Err(BackendError::inference(format!(
            "decoder got {} classes but {} labels were supplied",
            num_classes,
            labels.len()
        )));
    }

    let mut spans = Vec::new();

    // logits axes are (batch, start_word, width, class). We bound the
    // start_word loop by num_words rather than by the tensor's dim 1
    // because the tensor is padded to the batch-wide max num_words.
    let max_start = num_words.min(dims[1]);
    for start in 0..max_start {
        for width in 0..max_width {
            let end_word = start + width;
            if end_word >= num_words {
                continue;
            }
            for class in 0..num_classes {
                let raw = logits[[sequence_id, start, width, class]];
                let score = sigmoid(raw);
                if score < threshold {
                    continue;
                }
                let start_byte = words[start].byte_start;
                let end_byte = words[end_word].byte_end;
                if start_byte >= end_byte || end_byte > text.len() {
                    continue;
                }
                // Entity offsets are codepoint offsets so Python's
                // ``text[e.start:e.end] == e.text`` round-trips.
                let char_start = words[start].char_start;
                let char_end = words[end_word].char_end;
                spans.push(Entity {
                    sequence: sequence_id,
                    start: char_start,
                    end: char_end,
                    text: text[start_byte..end_byte].to_string(),
                    label: labels[class].to_string(),
                    score,
                });
            }
        }
    }

    Ok(spans)
}

// -----------------------------------------------------------------------------
// Greedy non-overlap search
// -----------------------------------------------------------------------------

fn entities_disjoint(a: &Entity, b: &Entity) -> bool {
    // a.end is exclusive: disjoint iff one ends before the other starts.
    a.end <= b.start || b.end <= a.start
}

fn entity_accept(a: &Entity, b: &Entity, params: ExtractParams) -> bool {
    if entities_disjoint(a, b) {
        return true;
    }
    // Overlapping spans. Two reject conditions collapsed into one
    // branch (clippy::if_same_then_else flagged the structurally
    // identical `false` arms): flat_ner forbids overlap entirely;
    // otherwise we still reject when dup_label is off and the labels
    // match. The fallthrough is multi_label: True iff overlapping
    // spans with different labels are allowed.
    if params.flat_ner || (!params.dup_label && a.label == b.label) {
        return false;
    }
    params.multi_label
}

/// Greedy filter — input must be sorted by (start, end) ascending.
fn greedy_search(spans: &[Entity], params: ExtractParams) -> Vec<Entity> {
    if spans.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(spans.len());
    let mut prev: usize = 0;
    let mut next: usize = 1;

    while next < spans.len() {
        let p = &spans[prev];
        let n = &spans[next];
        if entity_accept(p, n, params) {
            out.push(p.clone());
            prev = next;
        } else if p.score < n.score {
            prev = next;
        }
        next += 1;
    }
    out.push(spans[prev].clone());
    out
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_endpoints() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn split_words_handles_basic_ascii() {
        let toks = split_words("Hello world");
        assert_eq!(toks.len(), 2);
        assert_eq!(toks[0].text, "Hello");
        assert_eq!(toks[0].byte_start, 0);
        assert_eq!(toks[0].byte_end, 5);
        assert_eq!(toks[0].char_start, 0);
        assert_eq!(toks[0].char_end, 5);
        assert_eq!(toks[1].text, "world");
        assert_eq!(toks[1].byte_start, 6);
        assert_eq!(toks[1].byte_end, 11);
        assert_eq!(toks[1].char_start, 6);
        assert_eq!(toks[1].char_end, 11);
    }

    #[test]
    fn split_words_emits_char_offsets_on_multibyte_text() {
        // Curly quotes are 3 bytes each in UTF-8 but 1 codepoint.
        // The word "world" lives at byte 12 but char 8 in this string.
        let text = "Hello \u{201C}quoted\u{201D} world";
        let toks = split_words(text);
        let world = toks
            .iter()
            .find(|t| t.text == "world")
            .expect("world token");
        // Byte offset still works for Rust slicing:
        assert_eq!(&text[world.byte_start..world.byte_end], "world");
        // Char offsets are smaller because each curly quote is one
        // codepoint occupying three bytes.
        assert!(
            world.char_start < world.byte_start,
            "expected char_start < byte_start on multibyte input"
        );
        let chars: Vec<char> = text.chars().collect();
        let slice: String = chars[world.char_start..world.char_end].iter().collect();
        assert_eq!(slice, "world");
    }

    #[test]
    fn split_words_keeps_internal_dashes_and_underscores() {
        let toks = split_words("Anti-fraud my_var foo");
        let texts: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["Anti-fraud", "my_var", "foo"]);
    }

    #[test]
    fn split_words_keeps_punctuation_as_single_tokens() {
        let toks = split_words("Hello, world!");
        let texts: Vec<&str> = toks.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["Hello", ",", "world", "!"]);
    }

    #[test]
    fn make_span_tensors_shape_and_mask() {
        // batch=2, sequence 0 has 3 words, sequence 1 has 5 words,
        // max_width=2. num_words = max(3, 5) = 5. num_spans = 5*2 = 10.
        let mut text_lengths = Array2::<i64>::zeros((2, 1));
        text_lengths[[0, 0]] = 3;
        text_lengths[[1, 0]] = 5;
        let (idx, mask) = make_span_tensors(&text_lengths, 2);
        assert_eq!(idx.shape(), &[2, 10, 2]);
        assert_eq!(mask.shape(), &[2, 10]);

        // Sequence 0 with 3 words and max_width=2:
        // start=0: widths 0,1 → (0,0) and (0,1) at dim 0,1; both true.
        // start=1: widths 0,1 → (1,1) and (1,2) at dim 2,3; both true.
        // start=2: only width=0 → (2,2) at dim 4; true. Dim 5 stays false.
        // dims 6..10 are start=3,4 which don't exist; stay false.
        assert!(mask[[0, 0]]);
        assert_eq!(idx[[0, 0, 0]], 0);
        assert_eq!(idx[[0, 0, 1]], 0);
        assert!(mask[[0, 4]]); // start=2, width=0
        assert_eq!(idx[[0, 4, 0]], 2);
        assert_eq!(idx[[0, 4, 1]], 2);
        assert!(!mask[[0, 5]]);
        assert!(!mask[[0, 9]]);
    }

    #[test]
    fn greedy_search_keeps_disjoint() {
        let a = Entity {
            sequence: 0,
            start: 0,
            end: 5,
            text: "Hello".into(),
            label: "X".into(),
            score: 0.9,
        };
        let b = Entity {
            sequence: 0,
            start: 6,
            end: 11,
            text: "world".into(),
            label: "Y".into(),
            score: 0.8,
        };
        let out = greedy_search(&[a, b], ExtractParams::default());
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn greedy_search_drops_overlap_in_flat_ner() {
        // Two overlapping spans; flat_ner picks the higher-score one.
        let a = Entity {
            sequence: 0,
            start: 0,
            end: 6,
            text: "Barack".into(),
            label: "Person".into(),
            score: 0.7,
        };
        let b = Entity {
            sequence: 0,
            start: 0,
            end: 12,
            text: "Barack Obama".into(),
            label: "Person".into(),
            score: 0.95,
        };
        let mut sorted = vec![a, b];
        sorted.sort_by(|x, y| x.start.cmp(&y.start).then(x.end.cmp(&y.end)));
        let out = greedy_search(&sorted, ExtractParams::default());
        assert_eq!(out.len(), 1);
        assert!((out[0].score - 0.95).abs() < 1e-6);
    }

    /// Live smoke for onnx-community/gliner_medium-v2.1. Requires
    /// network or a populated cache.
    #[test]
    #[ignore = "requires network or cached onnx-community/gliner_medium-v2.1 weights"]
    fn gliner_medium_smoke() {
        use crate::core::model_registry::lookup_ner;
        let model = lookup_ner("onnx-community/gliner_medium-v2.1").expect("registered");
        let backend = OrtGlinerExtractor::load(model, &Device::Cpu, None).expect("load");

        let entities = backend
            .extract(
                &["Barack Obama was born in Hawaii."],
                &["person", "place"],
                ExtractParams::default(),
            )
            .expect("extract");

        assert_eq!(entities.len(), 1);
        let spans = &entities[0];
        // Expect at least one Person span — Barack Obama — and one
        // Place span — Hawaii.
        let labels: Vec<&str> = spans.iter().map(|e| e.label.as_str()).collect();
        assert!(labels.contains(&"person"), "missing person: {:?}", labels);
        assert!(labels.contains(&"place"), "missing place: {:?}", labels);
    }
}
