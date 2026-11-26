// name: src/core/src/foundation/lora.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Training-free LoRA-to-Phase SVD projection (`project_lora`, `project_lora_with_block_scales`).

use crate::foundation::aligned::{AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32};
use crate::foundation::pod::Rank1Descriptor;
use crate::foundation::scalar_math::{
    branchless_atan2_to_q7_fast, branchless_flush_subnormal_f32, branchless_sqrt_f32,
};
use crate::foundation::simd_stats::{
    atan2_interleaved_pairs_to_q7_word_x8_simd, atan2_to_q7_word_x8_simd, sum_sq_f32_simd8,
};
use crate::liminal::{LMNK_ERR_INVALID_ARGUMENT, LmnkResult, PhaseQ7, branchless_select_f32};

// ===================================================================
// Training-Free LoRA-to-Phase SVD Projection (LoRA-to-Phase SVD)
// ===================================================================

/// Projects fine-tuned LoRA matrices ($W_A \in \mathbb{R}^{r \times d_{\mathrm{in}}}, W_B \in \mathbb{R}^{d_{\mathrm{out}} \times r}$)
/// via first-order SVD approximation into phase profiles and Rank-1 residual vectors (Section 5.2).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `rank == 0`, `dim_in == 0`, `dim_out == 0`,
/// `dim_out` is odd, matrix slices are smaller than `rank * dim_in` / `dim_out * rank`,
/// or `phase_profile_out` is empty.
#[inline]
pub fn project_lora(
    lora_a: &AlignedSlice32<'_, f32>,
    lora_b: &AlignedSlice32<'_, f32>,
    rank: usize,
    dim_in: usize,
    dim_out: usize,
    rank1_out: &mut Rank1Descriptor,
    phase_profile_out: &mut AlignedSliceMut16<'_, PhaseQ7>,
) -> LmnkResult<()> {
    if rank == 0
        || dim_in == 0
        || dim_out == 0
        || (dim_out & 1) != 0
        || lora_a.len() < rank.checked_mul(dim_in).ok_or(LMNK_ERR_INVALID_ARGUMENT)?
        || lora_b.len() < dim_out.checked_mul(rank).ok_or(LMNK_ERR_INVALID_ARGUMENT)?
        || phase_profile_out.is_empty()
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }

    let b_slice = lora_b.as_slice();
    let a_slice = lora_a.as_slice();

    let norm_u_sq = if rank == 1 {
        sum_sq_f32_simd8(&b_slice[..dim_out])
    } else {
        let mut total_u = 0.0_f32;
        for i in 0..dim_out {
            let v = b_slice[i * rank];
            total_u += v * v;
        }
        total_u
    };
    let norm_v_sq = sum_sq_f32_simd8(&a_slice[..dim_in]);

    let sigma_1 = branchless_flush_subnormal_f32(branchless_sqrt_f32(norm_u_sq * norm_v_sq));
    rank1_out.layer_idx = 0;
    rank1_out.rank = 1;
    rank1_out.u_vector_ptr = lora_b.as_ptr() as u64;
    rank1_out.v_vector_ptr = lora_a.as_ptr() as u64;
    rank1_out.gain_scalar_ptr = 0;
    rank1_out.scale_factor = branchless_select_f32(sigma_1 > 0.0, sigma_1, 1.0);

    let out_phases = phase_profile_out.as_mut_slice();
    let num_heads = out_phases.len();
    let num_pairs = (dim_out / 2).max(1);
    let step_pairs = (num_pairs / num_heads.max(1)).max(1);
    let full_cells = num_heads >> 3;
    let mut cell = 0usize;
    while cell < full_cells {
        let h_base = cell << 3;
        let word = if rank == 1 && step_pairs == 1 && (h_base + 8) <= num_pairs {
            let base_f32 = h_base << 1;
            atan2_interleaved_pairs_to_q7_word_x8_simd(&b_slice[base_f32..base_f32 + 16])
        } else {
            let mut sin_arr = [0.0_f32; 8];
            let mut cos_arr = [0.0_f32; 8];
            let mut lane = 0usize;
            while lane < 8 {
                let pair_idx = ((h_base + lane) * step_pairs) % num_pairs;
                let base_idx = (pair_idx << 1) * rank;
                cos_arr[lane] = b_slice[base_idx];
                sin_arr[lane] = b_slice[base_idx + rank];
                lane += 1;
            }
            atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr)
        };
        let w_bytes = word.to_le_bytes();
        let mut lane_out = 0usize;
        while lane_out < 8 {
            out_phases[h_base + lane_out] = w_bytes[lane_out] as PhaseQ7;
            lane_out += 1;
        }
        cell += 1;
    }
    let mut h = full_cells << 3;
    while h < num_heads {
        let pair_idx = (h * step_pairs) % num_pairs;
        let base_idx = (pair_idx << 1) * rank;
        out_phases[h] = branchless_atan2_to_q7_fast(b_slice[base_idx + rank], b_slice[base_idx]);
        h += 1;
    }

    Ok(())
}

/// Matrix shape dimensions for LoRA projection kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoraDimensions {
    /// Low-rank inner dimension $r$.
    pub rank: usize,
    /// Input hidden dimension $d_{\mathrm{in}}$.
    pub dim_in: usize,
    /// Output hidden dimension $d_{\mathrm{out}}$ (even).
    pub dim_out: usize,
}

/// Projects fine-tuned LoRA matrices into a Rank-1 residual descriptor, a Q7 rotational phase
/// profile, and an **Isotropic Block-Diagonal Scaling** vector $\mathbf{D}$ (Theorem 1.1 & Section 5.2).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `dims.rank == 0`, `dims.dim_in == 0`, `dims.dim_out == 0`,
/// `dims.dim_out` is odd, matrix slices are too small, `phase_profile_out` is empty, or
/// `block_scales_out.len() < dims.dim_out / 2`.
#[inline]
pub fn project_lora_with_block_scales(
    lora_a: &AlignedSlice32<'_, f32>,
    lora_b: &AlignedSlice32<'_, f32>,
    dims: LoraDimensions,
    rank1_out: &mut Rank1Descriptor,
    phase_profile_out: &mut AlignedSliceMut16<'_, PhaseQ7>,
    block_scales_out: &mut AlignedSliceMut32<'_, f32>,
) -> LmnkResult<()> {
    let LoraDimensions {
        rank,
        dim_in,
        dim_out,
    } = dims;
    let num_pairs = dim_out / 2;
    if num_pairs == 0 || block_scales_out.len() < num_pairs {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    project_lora(
        lora_a,
        lora_b,
        rank,
        dim_in,
        dim_out,
        rank1_out,
        phase_profile_out,
    )?;

    let b_slice = lora_b.as_slice();
    let pair_stride = 2 * rank;
    let scales_slice = &mut block_scales_out.as_mut_slice()[..num_pairs];

    for (p_idx, scale_slot) in scales_slice.iter_mut().enumerate() {
        let row0 = p_idx * pair_stride;
        let pair_energy = sum_sq_f32_simd8(&b_slice[row0..row0 + pair_stride]);
        let mag = branchless_flush_subnormal_f32(branchless_sqrt_f32(pair_energy));
        *scale_slot = branchless_select_f32(mag > 1.0e-9, mag, 1.0);
    }
    rank1_out.gain_scalar_ptr = block_scales_out.as_ptr() as u64;
    Ok(())
}
