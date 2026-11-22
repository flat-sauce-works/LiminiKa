// name: src/core/src/foundation/simd_stats/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scalar & AVX2 constant-time norm/sum, `PhaseClamp` limit, Chebyshev deg-3, Procrustes, and Hodge dual kernels.

mod geometry;
mod procrustes;
mod reductions;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use geometry::{avx2_rotate_so2_block32, avx2_rotate_so2_triple_block32};
pub(crate) use geometry::{
    diff_norm_sq_f32_unrolled4, hodge_obstruction_and_dual_pairing_f32, rotate_so2_slice_unrolled16,
};
pub(crate) use procrustes::{
    atan2_interleaved_pairs_to_q7_word_x8_simd, atan2_to_q7_word_x8_simd, procrustes_q7_slice_simd,
    sincos_f32_x8_simd,
};
pub(crate) use reductions::{
    add_scalar_f32_slice_unrolled8, chebyshev_deg3_filter_slice_simd8, dot_f32_unrolled4,
    max_pos_f32_unrolled4, norm_l2_underflow_resilient_f32, scale_f32_slice_unrolled8,
};
pub use reductions::{sum_abs_and_sq_f32_simd8, sum_abs_sq_and_max_f32_simd8, sum_sq_f32_simd8};
