// name: src/core/src/foundation/rope/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.1 Capability 1: RoPE & Inline Tensor Modulation ([`PhaseSteering`] trait plus pure
//! sub-head routing, Rank-1 residual, and visual attention steering kernels) for the
//! Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! # Key Optimizations & Architectural Guarantees
//! - **`PhaseClamp` Upper-Quarter ($d_{\mathrm{head}}/4$) Low-Frequency Slicing**: All `PhaseClamp` soft-clamped
//!   steering kernels (`apply_phase_steering_safe`, `apply_phase_steering_geodesic_safe`,
//!   `steer_from_swar_words_core_safe`) rotate strictly the upper low-frequency quarter
//!   `rot_dim = ((head_dim >> 2) & !1).max(2)` (`[base + head_dim - rot_dim .. base + head_dim]`),
//!   preserving lower high-frequency syntactic RoPE channels and achieving a `2.2x-3.3x` speedup
//!   over full-channel RoPE.
//! - **3-Tier SWAR Head-Run Coalescing, Branchless `TZCNT` Stride & 4K-Aliasing Avoidance**:
//!   Full-channel steering kernels (`apply_phase_steering_slice_core`,
//!   `apply_phase_steering_pair_slice_core`) combine: (1) multi-word uniform-phase coalescing with a
//!   direct whole-tensor fast path, (2) a 4-instruction SWAR 7-pair adjacent-inequality detector
//!   (`(adj.wrapping_sub(SWAR_SPLAT_BYTE_64) & !adj & SWAR_HIGH_BITS_7PAIRS_64) == 0`) that
//!   rotates 8 distinct-phase heads via a fixed-trip `chunks_exact_mut(head_dim)` loop with a
//!   single bounds check and zero inner branches, (3) hardware `TZCNT` sub-word run-length stride,
//!   and (4) sequential 32-float unrolled `Q`/`K` rotation (`apply_phase_steering_slice_with_sincos`) when
//!   `q_run.len() >= 128` (`512B`) to eliminate 2-stream 4KB page-offset Load-Store Queue aliasing.
//! - **Zero-Arithmetic Direct 256-Entry LUT Fast Path in Sub-Head Routing**: When
//!   `is_default_ripa && gain_u7 == 128 && pulse_q7 == 0`, `rotate_pspm_subgroup_with_slot_phases`
//!   and `rotate_fused_ripa_pspm_subgroup_words` select `Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF` or
//!   `Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF` outside the head loop and index `lut[(slot_q7 as u8) as usize]`
//!   directly with zero per-head integer folding/scaling arithmetic and zero cache-branch jitter.
//! - **Single-Evaluation $SO(2)$ Group-Closure `PhaseClamp` + Sub-Head Router Fusion**: Because both `PhaseClamp` and sub-head router
//!   operate on the upper low-frequency quarter `rot_dim`, `dispatch_fused_phase_clamp_words_and_subhead_profiles_core`
//!   fuses $\mathbf{R}(\theta_{\mathrm{ripa}})\mathbf{R}(\theta_{\mathrm{pspm}}) = \mathbf{R}(\theta_{\mathrm{ripa}} + \theta_{\mathrm{pspm}})$
//!   via `eval_fused_phase_clamp_subhead_sincos_q7_fast` (evaluating the trigonometric polynomial once per
//!   head instead of twice on custom `PhaseClamp` limits) and caches consecutive head `(sin, cos)` pairs.
//! - **Zero-Jitter Micro-Swarm Quorum Pre-Pass & Constant-Time Damping**: `steer_from_swar_words_core_safe`
//!   resolves `quorum_mask` and `unresolved_damping_u7` in a branchless 8-word SWAR pre-pass, and
//!   [`PhaseSteering::apply_soft_phase_damping`] executes unconditional AVX2 subnormal-flushed scaling.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]
use crate::foundation::aligned::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut, AlignedSliceMut16, AlignedSliceMut32,
    DisjointMutPair,
};
use crate::foundation::branchless::{
    add_scalar_f32_slice_unrolled8, dot_f32_unrolled4, norm_l2_underflow_resilient_f32,
    procrustes_q7_slice_simd, rank1_accum_outer_product_simd, rank1_write_outer_product_simd,
    rotate_so2_2d_xy_slice_simd, rotate_so2_3d_txy_slice_simd, rotate_so2_slice_unrolled16,
    scale_f32_slice_unrolled8, sincos_f32_x8_simd, BranchlessKernel,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::foundation::branchless::{
    avx2_build_so2_2d_xy_vecs, avx2_hsum_ps, avx2_rotate_so2_2d_xy_slice_with_vecs,
    avx2_rotate_so2_block32,
};
use crate::foundation::lut_crc32::{
    F32_MIN_POSITIVE_NORMAL, Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF,
    Q7_FUSED_SELF_DEFAULT_PHASE_CLAMP_SINCOS_REF, Q7_FUSED_SELF_GEODESIC_PHASE_CLAMP_SINCOS_REF,
    Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF, Q7_SINCOS_LUT_256_REF, SWAR_SIGN_MASK_64,
    SWAR_SPLAT_BYTE_64,
};
use crate::foundation::pod::{
    gcso_paged_bitmask_t, gcso_rank1_residual_descriptor_t, gcso_subhead_config_t,
    gcso_tagged_ptr_t, Pod,
};
use crate::foundation::pod::{
    unpack_action_mode_from_tag, ACTION_MODE_PHASE_STEER, ACTION_MODE_RANK1_INJECT,
    ACTION_MODE_SUBHEAD_ROUTE, PAGED_BITMASK_BITS,
};
use crate::foundation::scalar_math::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_inv_pos_f32,
    branchless_is_finite_f32, branchless_min_f32, branchless_q7_add_sat,
    branchless_q7_fold_s1_geodesic, branchless_q7_scale_u7, branchless_rad_to_phase_step_u8,
    branchless_sincos_f32, branchless_sincos_half_pi_f32, branchless_sincos_q7,
    branchless_sincos_q7_phase_clamp_preinv, fast_inv_len_f32, fast_inv_pos_normal_f32,
};
use crate::foundation::scalar_math::{
    DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE, DEFAULT_PHASE_STEP_MIN_RAD,
};
use crate::foundation::swar_simd::{
    eval_fused_phase_clamp_subhead_sincos_q7_fast, eval_geodesic_phase_clamp_sincos_q7_fast,
    eval_phase_clamp_sincos_q7_fast, is_word_array_uniform_64, load_q7_words_64, load_q7_x8_slice,
    swar_phase_step_filter_q7_slice_in_place, swar_q7_add_sat_x8, swar_q7_circular_add_x8,
    swar_q7_fold_s1_geodesic_x8, swar_q7_scale_le128_with_bias_x8, swar_q7_scale_le128_x8,
    swar_q7_scale_u7_x8, swar_u8_sum_x8,
};
use crate::liminal::{
    branchless_select_f32, gcso_config_t, gcso_q7_t, GcsoResult, GCSO_ERROR_EDBC_SINGULARITY,
    GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER, GCSO_ERROR_NULL_POINTER,
    HEADS_PER_STIGMERGIC_GRANULE, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
    NUM_STIGMERGIC_GRANULES, Q7_PHASE_SCALE_RAD,
};
use core::f32::consts::FRAC_PI_2;
use core::mem::{align_of, offset_of, size_of};
use core::slice;

pub(crate) const NUM_SWARM_CELLS: usize = 8;
pub(crate) const HEADS_PER_CELL: usize = MAX_ATTENTION_HEADS / NUM_SWARM_CELLS;

/// Packs `num_heads <= MAX_ATTENTION_HEADS` Q7 phase bytes into an 8-word SWAR array.
#[inline(always)]
#[must_use]
pub(super) fn pack_q7_slice_to_swar_words_64(
    p_slice: &[gcso_q7_t],
    num_heads: usize,
) -> [u64; NUM_SWARM_CELLS] {
    if p_slice.len() >= MAX_ATTENTION_HEADS && num_heads == MAX_ATTENTION_HEADS {
        // SAFETY: `p_slice.len() >= MAX_ATTENTION_HEADS` and `align_of::<gcso_q7_t>() == 1`.
        let arr64 = unsafe { &*p_slice.as_ptr().cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        return load_q7_words_64(arr64);
    }
    let mut words = [0u64; NUM_SWARM_CELLS];
    let limit = num_heads.min(p_slice.len()).min(MAX_ATTENTION_HEADS);
    let full_cells = limit / HEADS_PER_CELL;
    let mut c = 0usize;
    while c < full_cells {
        words[c] = load_q7_x8_slice(p_slice, c * HEADS_PER_CELL);
        c += 1;
    }
    let rem_start = full_cells * HEADS_PER_CELL;
    if rem_start < limit {
        let mut tail_w = 0u64;
        let mut h = rem_start;
        while h < limit {
            tail_w |= u64::from(p_slice[h] as u8) << ((h - rem_start) * 8);
            h += 1;
        }
        words[full_cells] = tail_w;
    }
    words
}

mod kernels;
pub(crate) use kernels::steer_from_swar_words_core_safe;
pub use kernels::*;

mod phase_steering;
pub use phase_steering::*;

mod subhead_routing;
pub use subhead_routing::*;
pub(crate) use subhead_routing::{
    dispatch_fused_phase_clamp_and_subhead_words_core,
    rotate_fused_self_phase_clamp_and_subhead_default_words,
    rotate_fused_self_phase_clamp_subhead_q_and_clamp_k_default_words,
};

mod adapters;
pub use adapters::*;

const _: () = {
    assert!(size_of::<gcso_subhead_config_t>() == 32);
    assert!(align_of::<gcso_subhead_config_t>() == 16);
    assert!(size_of::<gcso_rank1_residual_descriptor_t>() == 64);
    assert!(align_of::<gcso_rank1_residual_descriptor_t>() == 32);
    assert!(NUM_SWARM_CELLS * HEADS_PER_CELL == MAX_ATTENTION_HEADS);
};

/// Default attention head dimension ($d_{\mathrm{head}} = 128$) for [`PhaseSteeringEngine::default`].
pub const DEFAULT_PHASE_STEER_HEAD_DIM: u32 = 128;

/// Default attention head count ($H = 32$) for [`PhaseSteeringEngine::default`].
pub const DEFAULT_PHASE_STEER_NUM_HEADS: u32 = 32;

/// Scale factor for Q7 quantized phase representation ($\beta_{\mathrm{Q7}} = \pi / 128$).
pub const Q7_PHASE_SCALE: f32 = Q7_PHASE_SCALE_RAD;

pub mod engine;
pub use engine::*;

#[cfg(test)]
mod tests;
