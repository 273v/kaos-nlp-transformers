//! Pooling + L2 normalization for sentence embeddings.
//!
//! Two pooling strategies covered (the only ones in our REGISTRY today):
//!
//! * **CLS pooling** — take the first-token hidden state. This is what
//!   BAAI/bge-small-en-v1.5 was trained for; ``Pooling::Cls``.
//! * **Mean pooling** — average the token hidden states, weighted by
//!   the attention mask so pad positions don't dilute the mean. This
//!   is what sentence-transformers/all-MiniLM-L6-v2 was trained for;
//!   ``Pooling::Mean``.
//!
//! L2 normalization is centralized here per audit-02 KNT-101 — we
//! always emit unit-norm vectors regardless of what the upstream model
//! does, so the contract holds for every entry in REGISTRY.

use crate::core::error::{BackendError, Result};
use ndarray::{s, Array2, ArrayView2, ArrayView3, Axis};

/// Which pooling to apply to ``last_hidden_state``.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Take the [CLS] token (index 0). BGE-family default.
    Cls,
    /// Mean over tokens, mask-weighted. MiniLM-family default.
    Mean,
}

/// Apply pooling to a ``last_hidden_state`` tensor, returning a
/// ``(batch, hidden)`` matrix.
///
/// Args:
///   * ``hidden_states`` — shape ``(batch, seq_len, hidden)``, float32.
///   * ``attention_mask`` — shape ``(batch, seq_len)``, int64 (0 or 1).
///   * ``pooling`` — ``Pooling::Cls`` or ``Pooling::Mean``.
pub fn pool(
    hidden_states: ArrayView3<'_, f32>,
    attention_mask: ArrayView2<'_, i64>,
    pooling: Pooling,
) -> Result<Array2<f32>> {
    let (batch, seq, hidden) = hidden_states.dim();
    let (mask_b, mask_s) = attention_mask.dim();
    if batch != mask_b || seq != mask_s {
        return Err(BackendError::inference(format!(
            "shape mismatch: hidden_states {:?} vs attention_mask {:?}",
            hidden_states.dim(),
            attention_mask.dim()
        )));
    }

    match pooling {
        Pooling::Cls => Ok(hidden_states.slice(s![.., 0, ..]).to_owned()),
        Pooling::Mean => mean_pool(hidden_states, attention_mask, batch, seq, hidden),
    }
}

fn mean_pool(
    hidden_states: ArrayView3<'_, f32>,
    attention_mask: ArrayView2<'_, i64>,
    batch: usize,
    seq: usize,
    hidden: usize,
) -> Result<Array2<f32>> {
    let mut out: Array2<f32> = Array2::zeros((batch, hidden));
    for b in 0..batch {
        let mut count: f32 = 0.0;
        for s_idx in 0..seq {
            // Only sum positions where attention_mask == 1.
            if attention_mask[[b, s_idx]] == 0 {
                continue;
            }
            count += 1.0;
            for h in 0..hidden {
                out[[b, h]] += hidden_states[[b, s_idx, h]];
            }
        }
        if count > 0.0 {
            for h in 0..hidden {
                out[[b, h]] /= count;
            }
        }
        // count == 0 → an entirely-pad row, leave the row as zeros.
    }
    Ok(out)
}

/// L2-normalize each row of an ``(N, dim)`` matrix in place. All-zero
/// rows are preserved as zeros (no division-by-zero). Audit-02 KNT-101.
pub fn l2_normalize(arr: &mut Array2<f32>) {
    for mut row in arr.axis_iter_mut(Axis(0)) {
        let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in row.iter_mut() {
                *x /= norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    #[test]
    fn cls_pool_picks_first_token() {
        // (batch=1, seq=3, hidden=2)
        let hidden = array![[[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]]];
        let mask: Array2<i64> = Array2::ones((1, 3));
        let out = pool(hidden.view(), mask.view(), Pooling::Cls).unwrap();
        assert_eq!(out, array![[1.0, 2.0]]);
    }

    #[test]
    fn mean_pool_respects_mask() {
        // (batch=1, seq=3, hidden=2). Mask zeroes out the third token.
        let hidden = array![[[1.0_f32, 1.0], [3.0, 3.0], [99.0, 99.0]]];
        let mask = array![[1_i64, 1, 0]];
        let out = pool(hidden.view(), mask.view(), Pooling::Mean).unwrap();
        // Mean of (1,1) and (3,3) = (2,2); the masked-out (99,99) is ignored.
        assert_eq!(out, array![[2.0_f32, 2.0]]);
    }

    #[test]
    fn mean_pool_all_pad_yields_zeros() {
        let hidden = array![[[1.0_f32, 2.0], [3.0, 4.0]]];
        let mask = array![[0_i64, 0]];
        let out = pool(hidden.view(), mask.view(), Pooling::Mean).unwrap();
        assert_eq!(out, array![[0.0_f32, 0.0]]);
    }

    #[test]
    fn shape_mismatch_errors() {
        let hidden: Array3<f32> = Array3::zeros((2, 3, 4));
        let mask: Array2<i64> = Array2::ones((1, 3));
        let err = pool(hidden.view(), mask.view(), Pooling::Mean).unwrap_err();
        assert!(matches!(err, BackendError::Inference(_)));
    }

    #[test]
    fn l2_normalize_unit_vectors() {
        let mut a: Array2<f32> = array![[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]];
        l2_normalize(&mut a);
        // (3,4) → (0.6, 0.8); zero row → zero; (1,0) → (1,0).
        let expected: Array2<f32> = array![[0.6, 0.8], [0.0, 0.0], [1.0, 0.0]];
        for (got, want) in a.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6);
        }
    }
}
