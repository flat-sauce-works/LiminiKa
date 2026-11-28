// name: src/core/src/foundation/lora.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Training-free LoRA-to-Phase SVD projection (`project_lora`, `project_lora_with_block_scales`).

use crate::foundation::aligned::{AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32};
use crate::foundation::pod::{ArenaSpan, RANK1_GAIN_UNUSED, Rank1Descriptor};
use crate::foundation::scalar_math::{
    branchless_atan2_to_q7_fast, branchless_flush_subnormal_f32, branchless_inv_pos_f32,
    branchless_sqrt_f32,
};
use crate::foundation::simd_stats::{
    atan2_interleaved_pairs_to_q7_word_x8_simd, atan2_to_q7_word_x8_simd, sum_sq_f32_simd8,
};
use crate::liminal::{
    LMNK_ERR_BUFFER_TOO_SMALL, LMNK_ERR_INVALID_ARGUMENT, LmnkResult, PhaseQ7,
    branchless_select_f32,
};

// ===================================================================
// Training-Free LoRA-to-Phase SVD Projection (LoRA-to-Phase SVD)
// ===================================================================

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

/// Projects fine-tuned LoRA matrices ($W_A \in \mathbb{R}^{r \times d_{\mathrm{in}}}, W_B \in
/// \mathbb{R}^{d_{\mathrm{out}} \times r}$)
/// via first-order SVD approximation into a Q7 phase profile and normalized `(u, v)` factor vectors
/// packed contiguously into `arena_out[0 .. dim_out + dim_in]`.
///
/// Populates `rank1_out` with [`ArenaSpan`] offsets (`u_span = 0..dim_out`, `v_span = dim_out..dim_out+dim_in`,
/// `gain_offset = RANK1_GAIN_UNUSED`) and returns the number of `f32` elements written into `arena_out`.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any dimension is zero, `dim_out` is odd, dimensions
/// overflow `u32`, matrix slices are too small, or `phase_profile_out` is empty.
/// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `arena_out.len() < dims.dim_out + dims.dim_in`.
#[inline]
pub fn project_lora(
    lora_a: &AlignedSlice32<'_, f32>,
    lora_b: &AlignedSlice32<'_, f32>,
    dims: LoraDimensions,
    arena_out: &mut [f32],
    rank1_out: &mut Rank1Descriptor,
    phase_profile_out: &mut AlignedSliceMut16<'_, PhaseQ7>,
) -> LmnkResult<usize> {
    let LoraDimensions {
        rank,
        dim_in,
        dim_out,
    } = dims;
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
    let arena_needed = dim_out
        .checked_add(dim_in)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    let u_len_u32 = u32::try_from(dim_out).map_err(|_| LMNK_ERR_INVALID_ARGUMENT)?;
    let v_len_u32 = u32::try_from(dim_in).map_err(|_| LMNK_ERR_INVALID_ARGUMENT)?;
    let _ = u_len_u32
        .checked_add(v_len_u32)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    if arena_out.len() < arena_needed {
        return Err(LMNK_ERR_BUFFER_TOO_SMALL);
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
    let norm_u = branchless_sqrt_f32(norm_u_sq);
    let norm_v = branchless_sqrt_f32(norm_v_sq);
    let inv_u = branchless_select_f32(norm_u > 0.0, branchless_inv_pos_f32(norm_u), 0.0);
    let inv_v = branchless_select_f32(norm_v > 0.0, branchless_inv_pos_f32(norm_v), 0.0);

    let (u_dst, rest) = arena_out.split_at_mut(dim_out);
    let v_dst = &mut rest[..dim_in];
    if rank == 1 {
        for (dst, &src) in u_dst.iter_mut().zip(b_slice[..dim_out].iter()) {
            *dst = branchless_flush_subnormal_f32(src * inv_u);
        }
    } else {
        for (i, dst) in u_dst.iter_mut().enumerate() {
            *dst = branchless_flush_subnormal_f32(b_slice[i * rank] * inv_u);
        }
    }
    for (dst, &src) in v_dst.iter_mut().zip(a_slice[..dim_in].iter()) {
        *dst = branchless_flush_subnormal_f32(src * inv_v);
    }

    let sigma_1 = branchless_flush_subnormal_f32(branchless_sqrt_f32(norm_u_sq * norm_v_sq));
    *rank1_out = Rank1Descriptor {
        layer_idx: 0,
        rank: 1,
        u_span: ArenaSpan::new(0, u_len_u32),
        v_span: ArenaSpan::new(u_len_u32, v_len_u32),
        gain_offset: RANK1_GAIN_UNUSED,
        scale_factor: branchless_select_f32(sigma_1 > 0.0, sigma_1, 1.0),
    };

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

    Ok(arena_needed)
}

/// Projects fine-tuned LoRA matrices into normalized `(u, v, g)` arena vectors, a portable
/// 32-byte [`Rank1Descriptor`], a Q7 rotational phase profile, and per-pair **Isotropic
/// Block-Diagonal Scaling** values (`block_scales_out`).
///
/// Packs `u` (`dim_out`), `v` (`dim_in`), and expanded per-output isotropic gain `g` (`dim_out`,
/// where `g[2*p] = g[2*p + 1] = block_scales_out[p]`) into `arena_out[0 .. 2 * dim_out + dim_in]`
/// and sets `rank1_out.gain_offset = (dim_out + dim_in) as u32`.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] on invalid dimensions or overflow, or
/// [`LMNK_ERR_BUFFER_TOO_SMALL`] if `arena_out.len() < 2 * dims.dim_out + dims.dim_in`.
#[inline]
pub fn project_lora_with_block_scales(
    lora_a: &AlignedSlice32<'_, f32>,
    lora_b: &AlignedSlice32<'_, f32>,
    dims: LoraDimensions,
    arena_out: &mut [f32],
    rank1_out: &mut Rank1Descriptor,
    phase_profile_out: &mut AlignedSliceMut16<'_, PhaseQ7>,
    block_scales_out: &mut AlignedSliceMut32<'_, f32>,
) -> LmnkResult<usize> {
    let LoraDimensions {
        rank,
        dim_in,
        dim_out,
    } = dims;
    let num_pairs = dim_out / 2;
    if num_pairs == 0 || block_scales_out.len() < num_pairs {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let uv_len = dim_out
        .checked_add(dim_in)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    let total_arena = uv_len
        .checked_add(dim_out)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    let gain_off_u32 = u32::try_from(uv_len).map_err(|_| LMNK_ERR_INVALID_ARGUMENT)?;
    let u_len_u32 = u32::try_from(dim_out).map_err(|_| LMNK_ERR_INVALID_ARGUMENT)?;
    let _ = gain_off_u32
        .checked_add(u_len_u32)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    if arena_out.len() < total_arena {
        return Err(LMNK_ERR_BUFFER_TOO_SMALL);
    }

    let _ = project_lora(
        lora_a,
        lora_b,
        dims,
        arena_out,
        rank1_out,
        phase_profile_out,
    )?;

    let b_slice = lora_b.as_slice();
    let pair_stride = 2 * rank;
    let scales_slice = &mut block_scales_out.as_mut_slice()[..num_pairs];
    let gain_dst = &mut arena_out[uv_len..total_arena];

    for (p_idx, scale_slot) in scales_slice.iter_mut().enumerate() {
        let row0 = p_idx * pair_stride;
        let pair_energy = sum_sq_f32_simd8(&b_slice[row0..row0 + pair_stride]);
        let mag = branchless_flush_subnormal_f32(branchless_sqrt_f32(pair_energy));
        let eff = branchless_select_f32(mag > 1.0e-9, mag, 1.0);
        *scale_slot = eff;
        gain_dst[p_idx * 2] = eff;
        gain_dst[p_idx * 2 + 1] = eff;
    }
    rank1_out.gain_offset = gain_off_u32;
    Ok(total_arena)
}
