//! BERT-style token classification — closed-label NER over BIO tags.
//!
//! Unlike the GLiNER zero-shot path in ``core::ner``, this module
//! serves models that bake their entire label vocabulary into
//! ``config.json::id2label`` at training time. The inference shape
//! is the classic BERT-NER one:
//!
//! ```text
//!  tokens → (input_ids, attention_mask[, token_type_ids]) → ort →
//!  logits (batch, seq, n_classes) → argmax + softmax confidence →
//!  BIO-decode to spans → map subword offsets to char offsets
//! ```
//!
//! Used today for the PII detection model
//! (``onnx-community/bert-small-pii-detection-ONNX``, 49 classes =
//! O + 24 PII categories × {B-, I-}). The trait surface is generic
//! — future closed-label NER models (POS taggers, legal-clause
//! taggers) would slot in here too.
//!
//! Output spans share the ``Entity`` shape from ``core::ner`` so
//! downstream Python code can consume PII + GLiNER spans
//! interchangeably (the Python ``Entity`` dataclass in
//! ``kaos_nlp_transformers.ner`` is the canonical surface).

use crate::core::device::Device;
use crate::core::error::{BackendError, Result};
use crate::core::model_loader::resolve_paths;
use crate::core::model_registry::RegisteredModel;
use crate::core::ner::Entity;
use ndarray::{Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

// -----------------------------------------------------------------------------
// Public surface
// -----------------------------------------------------------------------------

/// BERT-style token classifier trait. ``score_threshold`` filters
/// out low-confidence spans (softmax probability under the threshold);
/// 0.0 returns every span the BIO decoder produces.
pub trait TokenClassifier: Send + Sync {
    /// Run the classifier over a batch of input texts. Returns one
    /// ``Vec<Entity>`` per input.
    fn classify(&self, texts: &[&str], score_threshold: f32) -> Result<Vec<Vec<Entity>>>;

    /// HF Hub model id.
    fn model_id(&self) -> &str;

    /// Device this classifier runs on.
    fn device(&self) -> &str;

    /// Distinct entity labels exposed by the loaded model (post-BIO
    /// strip — e.g. "PERSON", "EMAIL_ADDRESS", not "B-PERSON" /
    /// "I-PERSON"). Useful for surface APIs that want to list the
    /// available categories before extracting.
    fn labels(&self) -> &[String];
}

// -----------------------------------------------------------------------------
// ort-backed implementation
// -----------------------------------------------------------------------------

/// ort-backed token classifier.
pub struct OrtTokenClassifier {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    /// Ordered raw label strings (index = class id). E.g.
    /// ``["O", "B-AGE", "I-AGE", ...]``.
    id2label: Vec<String>,
    /// Deduplicated entity categories with BIO prefixes stripped.
    /// E.g. ``["AGE", "COORDINATE", "PERSON", ...]``.
    distinct_labels: Vec<String>,
    /// True iff the loaded ONNX accepts a ``token_type_ids`` input.
    accepts_token_type_ids: bool,
    model_id: String,
    device_str: String,
    max_seq_len: usize,
}

impl OrtTokenClassifier {
    /// Load a registered token classifier. Fetches the model's
    /// ``config.json`` alongside the ONNX + tokenizer so we can read
    /// the ``id2label`` map at load time.
    pub fn load(
        model: &'static RegisteredModel,
        device: &Device,
        cache_dir: Option<&Path>,
    ) -> Result<Self> {
        let paths = resolve_paths(model, cache_dir)?;

        // Fetch config.json from the same SHA via hf-hub. The
        // standard model_loader path doesn't surface config.json
        // today; we go direct to hf-hub here for label decoding.
        let config_path = fetch_config_json(model, cache_dir)?;
        let id2label = parse_id2label(&config_path, model)?;
        let distinct_labels = derive_distinct_labels(&id2label);

        let builder = Session::builder().map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("Session::builder: {e}"),
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

        // KNT-NLI-002: env-var thread override; default to ort's
        // adaptive sizing (see rust/core/ner.rs for measurement).
        let builder = configure_intra_threads(builder, model)?;
        let mut builder = configure_eps(builder, device, model)?;

        let session = builder.commit_from_file(&paths.onnx).map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("commit_from_file({}): {e}", paths.onnx.display()),
            )
        })?;

        let accepts_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        // Load tokenizer raw — we drive padding/truncation ourselves
        // so the offsets are anchored to the original text without
        // BatchLongest auto-padding side effects.
        let tokenizer = Tokenizer::from_file(&paths.tokenizer).map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("Tokenizer::from_file({}): {e}", paths.tokenizer.display()),
            )
        })?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            id2label,
            distinct_labels,
            accepts_token_type_ids,
            model_id: model.model_id.to_string(),
            device_str: device.as_str().to_string(),
            max_seq_len: model.max_seq_len,
        })
    }
}

impl TokenClassifier for OrtTokenClassifier {
    fn classify(&self, texts: &[&str], score_threshold: f32) -> Result<Vec<Vec<Entity>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        if !(0.0..=1.0).contains(&score_threshold) {
            return Err(BackendError::inference(format!(
                "score_threshold must be in [0, 1], got {score_threshold}"
            )));
        }

        // 1. Encode the batch with offset tracking. ``encode_batch``
        //    (NOT ``encode_batch_fast``) is the variant that populates
        //    per-token byte offsets — the *_fast methods skip offset
        //    bookkeeping for throughput. We need offsets to map BIO
        //    spans back to source char ranges, so this is the right
        //    knob.
        let encodings = self
            .tokenizer
            .encode_batch(
                texts
                    .iter()
                    .map(|t| tokenizers::EncodeInput::Single((*t).into()))
                    .collect::<Vec<_>>(),
                true, // add_special_tokens: include [CLS]/[SEP]
            )
            .map_err(BackendError::tokenization)?;

        let batch_size = encodings.len();
        let mut seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        if seq_len > self.max_seq_len {
            seq_len = self.max_seq_len;
        }
        if seq_len == 0 {
            return Ok(vec![Vec::new(); batch_size]);
        }

        // 2. Build padded tensors. Manual padding so we keep the
        //    offsets aligned (the tokenizers crate's `with_padding`
        //    can interfere with truncation + offsets).
        let mut input_ids = Array2::<i64>::zeros((batch_size, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, seq_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch_size, seq_len));
        // Per-row offsets, post-truncation. None for padding/special tokens.
        let mut row_offsets: Vec<Vec<Option<(usize, usize)>>> = Vec::with_capacity(batch_size);
        let mut row_token_types: Vec<Vec<u32>> = Vec::with_capacity(batch_size);

        for (row, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let types = enc.get_type_ids();
            let offs = enc.get_offsets();
            let special = enc.get_special_tokens_mask();
            let take = ids.len().min(seq_len);
            let mut row_off: Vec<Option<(usize, usize)>> = Vec::with_capacity(seq_len);
            let mut row_typ: Vec<u32> = Vec::with_capacity(seq_len);
            for col in 0..take {
                input_ids[[row, col]] = ids[col] as i64;
                attention_mask[[row, col]] = mask[col] as i64;
                token_type_ids[[row, col]] = types[col] as i64;
                row_typ.push(types[col]);
                // Mask out special tokens (CLS / SEP) from offset
                // decoding so they don't become spurious "O"-labelled
                // spans.
                if special[col] == 1 {
                    row_off.push(None);
                } else {
                    let (a, b) = offs[col];
                    if b > a {
                        row_off.push(Some((a, b)));
                    } else {
                        row_off.push(None);
                    }
                }
            }
            for _ in take..seq_len {
                row_off.push(None);
                row_typ.push(0);
            }
            row_offsets.push(row_off);
            row_token_types.push(row_typ);
        }

        // 3. Run ort session.
        let logits = self.run_session(&input_ids, &attention_mask, &token_type_ids)?;
        // Expected shape: (batch, seq, n_classes).
        let shape = logits.shape();
        if shape.len() != 3 {
            return Err(BackendError::inference(format!(
                "logits has shape {shape:?}, expected (batch, seq, n_classes)"
            )));
        }
        let n_classes = shape[2];
        if n_classes != self.id2label.len() {
            return Err(BackendError::inference(format!(
                "logits last dim {} != id2label count {}",
                n_classes,
                self.id2label.len()
            )));
        }

        // 4. Decode per-sequence.
        let mut out = Vec::with_capacity(batch_size);
        for (s, text) in texts.iter().enumerate() {
            let entities =
                self.decode_sequence(&logits, s, text, &row_offsets[s], score_threshold)?;
            out.push(entities);
        }
        Ok(out)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
    fn device(&self) -> &str {
        &self.device_str
    }
    fn labels(&self) -> &[String] {
        &self.distinct_labels
    }
}

impl OrtTokenClassifier {
    fn run_session(
        &self,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
        token_type_ids: &Array2<i64>,
    ) -> Result<Array3<f32>> {
        let mut session = self
            .session
            .lock()
            .map_err(|e| BackendError::inference(format!("session mutex poisoned: {e}")))?;

        let input_ids_view = input_ids.view();
        let attention_view = attention_mask.view();
        let type_ids_view = token_type_ids.view();

        let ids_ref = TensorRef::from_array_view(input_ids_view)
            .map_err(|e| BackendError::inference(format!("input_ids tensor: {e}")))?;
        let mask_ref = TensorRef::from_array_view(attention_view)
            .map_err(|e| BackendError::inference(format!("attention_mask tensor: {e}")))?;
        let type_ref = TensorRef::from_array_view(type_ids_view)
            .map_err(|e| BackendError::inference(format!("token_type_ids tensor: {e}")))?;

        let outputs = if self.accepts_token_type_ids {
            let inputs = ort::inputs! {
                "input_ids" => ids_ref,
                "attention_mask" => mask_ref,
                "token_type_ids" => type_ref,
            };
            session
                .run(inputs)
                .map_err(|e| BackendError::inference(format!("ort session.run: {e}")))?
        } else {
            let inputs = ort::inputs! {
                "input_ids" => ids_ref,
                "attention_mask" => mask_ref,
            };
            session
                .run(inputs)
                .map_err(|e| BackendError::inference(format!("ort session.run: {e}")))?
        };

        let logits_value = outputs.get("logits").ok_or_else(|| {
            BackendError::inference("expected 'logits' output not found in ort outputs")
        })?;

        let (shape, raw) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(|e| BackendError::inference(format!("extract logits: {e}")))?;

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        if dims.len() != 3 {
            return Err(BackendError::inference(format!(
                "logits has unexpected shape {dims:?}"
            )));
        }
        let arr = Array3::from_shape_vec((dims[0], dims[1], dims[2]), raw.to_vec())
            .map_err(|e| BackendError::inference(format!("logits reshape: {e}")))?;
        Ok(arr)
    }

    fn decode_sequence(
        &self,
        logits: &Array3<f32>,
        sequence_id: usize,
        text: &str,
        offsets: &[Option<(usize, usize)>],
        score_threshold: f32,
    ) -> Result<Vec<Entity>> {
        // Precompute byte→char offset map for the source text. The
        // tokenizers crate emits byte offsets; Python callers index
        // text by codepoint so we hand back char offsets in Entity.
        let byte_to_char = build_byte_to_char(text);

        let n_classes = logits.shape()[2];
        let seq_len = logits.shape()[1].min(offsets.len());

        // BIO-decoder state.
        let mut current: Option<OpenSpan> = None;
        let mut out: Vec<Entity> = Vec::new();

        // ``offsets`` and the second dim of ``logits`` share the
        // index — iterate the offsets vec directly and pull
        // ``tok_idx`` for the ndarray slice. clippy::needless_range_loop
        // flagged the prior `for tok_idx in 0..seq_len` form.
        for (tok_idx, offset_opt) in offsets.iter().take(seq_len).enumerate() {
            let Some((byte_start, byte_end)) = *offset_opt else {
                // Special token or padding — close any open span.
                if let Some(span) = current.take() {
                    if let Some(e) = finalize_span(span, text, &byte_to_char, score_threshold) {
                        out.push(Entity {
                            sequence: sequence_id,
                            ..e
                        });
                    }
                }
                continue;
            };

            // Softmax over class logits at this position.
            let row = logits.slice(ndarray::s![sequence_id, tok_idx, ..]);
            let (best_id, best_score) = softmax_argmax(row.as_slice().unwrap_or(&[]), n_classes);
            let label = &self.id2label[best_id];

            let (prefix, category) = split_bio(label);
            match prefix {
                BioPrefix::Outside => {
                    if let Some(span) = current.take() {
                        if let Some(e) = finalize_span(span, text, &byte_to_char, score_threshold) {
                            out.push(Entity {
                                sequence: sequence_id,
                                ..e
                            });
                        }
                    }
                }
                BioPrefix::Begin => {
                    // Close any open span first, then start a fresh one.
                    if let Some(span) = current.take() {
                        if let Some(e) = finalize_span(span, text, &byte_to_char, score_threshold) {
                            out.push(Entity {
                                sequence: sequence_id,
                                ..e
                            });
                        }
                    }
                    current = Some(OpenSpan {
                        category: category.to_string(),
                        byte_start,
                        byte_end,
                        score_sum: best_score,
                        score_min: best_score,
                        score_count: 1,
                    });
                }
                BioPrefix::Inside => {
                    if let Some(ref mut span) = current {
                        if span.category == category {
                            // Continuation of the same span.
                            span.byte_end = byte_end;
                            span.score_sum += best_score;
                            span.score_min = span.score_min.min(best_score);
                            span.score_count += 1;
                        } else {
                            // Mismatched I- tag — close prev span and
                            // start a new one (lenient BIO recovery).
                            let prev = current.take().expect("matched above");
                            if let Some(e) =
                                finalize_span(prev, text, &byte_to_char, score_threshold)
                            {
                                out.push(Entity {
                                    sequence: sequence_id,
                                    ..e
                                });
                            }
                            current = Some(OpenSpan {
                                category: category.to_string(),
                                byte_start,
                                byte_end,
                                score_sum: best_score,
                                score_min: best_score,
                                score_count: 1,
                            });
                        }
                    } else {
                        // Stray I- with no preceding B- (BIO error in
                        // the model output). Treat as B- — open a new
                        // span.
                        current = Some(OpenSpan {
                            category: category.to_string(),
                            byte_start,
                            byte_end,
                            score_sum: best_score,
                            score_min: best_score,
                            score_count: 1,
                        });
                    }
                }
            }
        }
        // Close any still-open span at end-of-sequence.
        if let Some(span) = current.take() {
            if let Some(e) = finalize_span(span, text, &byte_to_char, score_threshold) {
                out.push(Entity {
                    sequence: sequence_id,
                    ..e
                });
            }
        }
        Ok(out)
    }
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BioPrefix {
    Begin,
    Inside,
    Outside,
}

fn split_bio(label: &str) -> (BioPrefix, &str) {
    if label == "O" {
        return (BioPrefix::Outside, "");
    }
    if let Some(rest) = label.strip_prefix("B-") {
        return (BioPrefix::Begin, rest);
    }
    if let Some(rest) = label.strip_prefix("I-") {
        return (BioPrefix::Inside, rest);
    }
    // Some models use plain category names without B-/I- prefixes;
    // treat as Begin (start a fresh span every time we see it). This
    // is a defensive fallback for non-standard label sets.
    (BioPrefix::Begin, label)
}

struct OpenSpan {
    category: String,
    byte_start: usize,
    byte_end: usize,
    score_sum: f32,
    score_min: f32,
    score_count: usize,
}

fn finalize_span(
    span: OpenSpan,
    text: &str,
    byte_to_char: &[usize],
    score_threshold: f32,
) -> Option<Entity> {
    // Use the MIN score across the span — most conservative; a
    // single low-confidence token disqualifies the whole span. The
    // alternative (mean) tends to dilute uncertainty.
    let score = span.score_min;
    if score < score_threshold {
        return None;
    }
    if span.byte_start >= span.byte_end || span.byte_end > text.len() {
        return None;
    }
    let char_start = *byte_to_char.get(span.byte_start)?;
    let char_end = *byte_to_char.get(span.byte_end)?;
    let surface = &text[span.byte_start..span.byte_end];
    Some(Entity {
        sequence: 0, // overridden by caller
        start: char_start,
        end: char_end,
        text: surface.to_string(),
        label: span.category,
        score,
    })
}

/// Build a flat byte→char map. ``out[byte_index]`` is the codepoint
/// index for that byte boundary. Length is ``text.len() + 1`` so the
/// end-of-string offset is also valid.
fn build_byte_to_char(text: &str) -> Vec<usize> {
    let mut map = vec![0usize; text.len() + 1];
    let mut char_idx = 0usize;
    for (byte_idx, _) in text.char_indices() {
        map[byte_idx] = char_idx;
        char_idx += 1;
    }
    map[text.len()] = char_idx;
    // Fill any unset entries (interior bytes of a multi-byte codepoint)
    // with the previous valid char index — defensive; the BIO decoder
    // only ever queries codepoint-aligned positions, but the
    // tokenizer's offsets are already aligned to byte boundaries that
    // are codepoint boundaries for well-formed tokenizers.
    let mut last = 0usize;
    for slot in map.iter_mut() {
        if *slot == 0 && last > 0 {
            *slot = last;
        } else {
            last = *slot;
        }
    }
    map
}

fn softmax_argmax(row: &[f32], n_classes: usize) -> (usize, f32) {
    // Numerically stable softmax + argmax in one pass.
    if row.is_empty() || n_classes == 0 {
        return (0, 0.0);
    }
    let max_logit = row
        .iter()
        .take(n_classes)
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    for &v in row.iter().take(n_classes) {
        exp_sum += (v - max_logit).exp();
    }
    let mut best_id = 0usize;
    let mut best_score = 0.0f32;
    for (i, &v) in row.iter().take(n_classes).enumerate() {
        let p = (v - max_logit).exp() / exp_sum;
        if p > best_score {
            best_score = p;
            best_id = i;
        }
    }
    (best_id, best_score)
}

fn derive_distinct_labels(id2label: &[String]) -> Vec<String> {
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for label in id2label {
        match split_bio(label) {
            (BioPrefix::Outside, _) => {}
            (_, category) => {
                if !category.is_empty() {
                    seen.insert(category.to_string());
                }
            }
        }
    }
    seen.into_iter().collect()
}

// -----------------------------------------------------------------------------
// Config.json fetch + parse (sibling to model_loader::resolve_paths)
// -----------------------------------------------------------------------------

fn fetch_config_json(
    model: &RegisteredModel,
    cache_dir: Option<&Path>,
) -> Result<std::path::PathBuf> {
    let mut api_builder = hf_hub::api::sync::ApiBuilder::new();
    if let Some(cd) = cache_dir {
        api_builder = api_builder.with_cache_dir(cd.to_path_buf());
    }
    let api = api_builder.build().map_err(|e| {
        BackendError::model_load(
            model.model_id,
            model.revision,
            format!("hf-hub builder (config.json): {e}"),
        )
    })?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        model.model_id.to_string(),
        hf_hub::RepoType::Model,
        model.revision.to_string(),
    ));
    repo.get("config.json").map_err(|e| {
        BackendError::model_load(
            model.model_id,
            model.revision,
            format!("download config.json: {e}"),
        )
    })
}

fn parse_id2label(config_path: &Path, model: &RegisteredModel) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(config_path).map_err(|e| {
        BackendError::model_load(
            model.model_id,
            model.revision,
            format!("read config.json: {e}"),
        )
    })?;
    let value: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
        BackendError::model_load(
            model.model_id,
            model.revision,
            format!("parse config.json: {e}"),
        )
    })?;
    let obj = value
        .get("id2label")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                "config.json missing id2label map",
            )
        })?;

    let mut pairs: Vec<(usize, String)> = Vec::with_capacity(obj.len());
    for (k, v) in obj {
        let id: usize = k.parse().map_err(|e| {
            BackendError::model_load(
                model.model_id,
                model.revision,
                format!("config.json id2label has non-integer key {k:?}: {e}"),
            )
        })?;
        let label = v
            .as_str()
            .ok_or_else(|| {
                BackendError::model_load(
                    model.model_id,
                    model.revision,
                    format!("config.json id2label[{k:?}] is not a string"),
                )
            })?
            .to_string();
        pairs.push((id, label));
    }
    pairs.sort_by_key(|(id, _)| *id);
    if pairs.is_empty() {
        return Err(BackendError::model_load(
            model.model_id,
            model.revision,
            "config.json id2label is empty",
        ));
    }
    // Validate the map is dense [0, n).
    for (expected, (id, _)) in pairs.iter().enumerate() {
        if *id != expected {
            return Err(BackendError::model_load(
                model.model_id,
                model.revision,
                format!("config.json id2label is sparse: missing class id {expected}"),
            ));
        }
    }
    Ok(pairs.into_iter().map(|(_, label)| label).collect())
}

// -----------------------------------------------------------------------------
// EP + thread plumbing (mirror of reranker.rs / nli.rs / ner.rs)
// -----------------------------------------------------------------------------

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
            "Token classifier {} requested on {:?}; CUDA execution requires the [gpu] companion wheel",
            model.model_id, device
        ))),
        Device::OpenVino => Err(BackendError::BackendNotInstalled(format!(
            "Token classifier {} requested on {:?}; OpenVINO execution requires the [openvino] companion wheel",
            model.model_id, device
        ))),
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_bio_handles_standard_labels() {
        assert_eq!(split_bio("O"), (BioPrefix::Outside, ""));
        assert_eq!(split_bio("B-PERSON"), (BioPrefix::Begin, "PERSON"));
        assert_eq!(
            split_bio("I-EMAIL_ADDRESS"),
            (BioPrefix::Inside, "EMAIL_ADDRESS")
        );
        // Defensive fallback for non-BIO label sets.
        assert_eq!(split_bio("CUSTOM"), (BioPrefix::Begin, "CUSTOM"));
    }

    #[test]
    fn derive_distinct_strips_bio_and_dedups() {
        let id2label = vec![
            "O".to_string(),
            "B-PERSON".to_string(),
            "I-PERSON".to_string(),
            "B-EMAIL".to_string(),
            "I-EMAIL".to_string(),
        ];
        let got = derive_distinct_labels(&id2label);
        assert_eq!(got, vec!["EMAIL".to_string(), "PERSON".to_string()]);
    }

    #[test]
    fn softmax_argmax_picks_max_class() {
        let logits = vec![-1.0_f32, 2.0, 0.0, -3.0];
        let (id, score) = softmax_argmax(&logits, 4);
        assert_eq!(id, 1);
        assert!(score > 0.7); // softmax([−1, 2, 0, −3]) puts class 1 well above 0.5
    }

    #[test]
    fn byte_to_char_handles_multibyte() {
        // Text layout (char count / byte count per piece):
        //   "Hello "     6 / 6
        //   "\u{201C}"   1 / 3   (curly opening quote)
        //   "quoted"     6 / 6
        //   "\u{201D}"   1 / 3   (curly closing quote)
        //   " world"     6 / 6
        // Total: 20 chars / 24 bytes. "world" lands at byte 19, char 15.
        let text = "Hello \u{201C}quoted\u{201D} world";
        let map = build_byte_to_char(text);
        assert_eq!(map[0], 0, "byte 0 -> char 0");
        // First curly quote byte boundary maps to char index 6.
        assert_eq!(map[6], 6, "first curly quote at byte 6 / char 6");
        let world_byte = text.find("world").expect("world present");
        assert_eq!(world_byte, 19, "sanity: 'world' starts at byte 19");
        assert_eq!(map[world_byte], 15, "'world' starts at char 15");
        // End-of-string sentinel.
        assert_eq!(map[text.len()], text.chars().count());
    }

    #[test]
    #[ignore = "debug-only — dumps PII raw logits"]
    fn pii_debug_top_per_token() {
        use crate::core::model_registry::lookup_pii;
        let model = lookup_pii("onnx-community/bert-small-pii-detection-ONNX").expect("registered");
        let backend = OrtTokenClassifier::load(model, &Device::Cpu, None).expect("load");

        let texts = &["Contact Jennifer Stacey at jen.stacey@galera.com or +1-555-0142."];
        let encodings = backend
            .tokenizer
            .encode_batch(
                texts
                    .iter()
                    .map(|t| tokenizers::EncodeInput::Single((*t).into()))
                    .collect::<Vec<_>>(),
                true,
            )
            .expect("encode");
        let enc = &encodings[0];
        println!("tokens: {:?}", enc.get_tokens());
        println!("offsets: {:?}", enc.get_offsets());
        println!("input_ids: {:?}", enc.get_ids());

        let seq_len = enc.get_ids().len();
        let mut input_ids = Array2::<i64>::zeros((1, seq_len));
        let mut attn = Array2::<i64>::zeros((1, seq_len));
        let mut typ = Array2::<i64>::zeros((1, seq_len));
        for i in 0..seq_len {
            input_ids[[0, i]] = enc.get_ids()[i] as i64;
            attn[[0, i]] = enc.get_attention_mask()[i] as i64;
            typ[[0, i]] = enc.get_type_ids()[i] as i64;
        }
        let logits = backend.run_session(&input_ids, &attn, &typ).expect("run");
        println!("logits shape: {:?}", logits.shape());

        // Dump argmax label per token + max softmax probability.
        let n_classes = logits.shape()[2];
        for i in 0..seq_len {
            let row = logits.slice(ndarray::s![0, i, ..]);
            let (best_id, best_score) = softmax_argmax(row.as_slice().unwrap_or(&[]), n_classes);
            let label = &backend.id2label[best_id];
            let tok = enc.get_tokens().get(i).cloned().unwrap_or_default();
            println!(
                "  [{:>2}] tok={:<20} -> {:<25} p={:.3}",
                i, tok, label, best_score
            );
        }
    }

    /// Live smoke for the registered PII model. Network or warm cache
    /// required.
    #[test]
    #[ignore = "requires network or cached onnx-community/bert-small-pii-detection-ONNX"]
    fn pii_bert_small_smoke() {
        use crate::core::model_registry::lookup_pii;
        let model = lookup_pii("onnx-community/bert-small-pii-detection-ONNX").expect("registered");
        let backend = OrtTokenClassifier::load(model, &Device::Cpu, None).expect("load");
        let labels = backend.labels();
        assert!(labels.iter().any(|l| l == "PERSON"));
        assert!(labels.iter().any(|l| l == "EMAIL_ADDRESS"));

        let texts = &["Contact Jennifer Stacey at jen.stacey@galera.com or +1-555-0142."];
        let out = backend.classify(texts, 0.5).expect("classify");
        assert_eq!(out.len(), 1);
        let entities = &out[0];
        let categories: std::collections::HashSet<&str> =
            entities.iter().map(|e| e.label.as_str()).collect();
        assert!(
            categories.contains("PERSON"),
            "expected PERSON: {:?}",
            entities
        );
        assert!(
            categories.contains("EMAIL_ADDRESS"),
            "expected EMAIL_ADDRESS: {:?}",
            entities
        );
    }
}
