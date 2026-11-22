// name: src/core/src/foundation/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 1 (`Foundation`): Pure geometric, POD, aligned-slice, SWAR/AVX2 SIMD, lookup-table,
//! RoPE phase-steering, Rank-1 residual, LoRA SVD, and matrix-free cohomology kernels.
//!
//! Depends strictly on `core` and [`crate::liminal`].

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::manual_unwrap_or)]

pub mod aligned;
pub mod cohomology;
pub mod lora;
pub mod lut_crc32;
pub mod phase_ops;
pub mod pod;
pub mod projection;
pub mod rank1_kernels;
pub mod rope;
pub mod scalar_math;
pub mod simd_stats;
pub mod swar_simd;

pub use self::aligned::*;
pub use self::cohomology::*;
pub use self::lora::*;
pub use self::lut_crc32::*;
pub(crate) use self::lut_crc32::{
    branchless_popcount_rsqrt_u32, branchless_q7_default_gravity_from_diff_u8,
    compute_aligned64_payload_crc32_ieee, crc32_slice16_dep_lo0, crc32_slice16_indep_hi0_u32,
    crc32_slice16_indep_hi1_u32, crc32_slice16_indep_lo1_u32, crc32_slice16_step_u64x2,
    crc32_step_block64_words, crc32_step_l3, crc32_step_l4, crc32_step_l5, crc32_step_l6,
    CRC32_IEEE_SLICE16_REF, MAX_BAYESIAN_STIGMERGIC_DENSITY, POPCOUNT_RSQRT_LUT_257_REF,
    Q7_FULL_COS_REF, VON_MISES_KAPPA_AT_MAX_R_BITS,
};
pub use self::phase_ops::*;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use self::phase_ops::{
    avx2_coherence_from_diff_vec64, avx2_slerp_from_diff_vec64, avx2_vec64_to_words64,
    avx2_words64_to_vec64, avx2_words64_to_vec64_with_uniform,
};
pub(crate) use self::phase_ops::{
    branchless_q7_unified_manifold_fold_words_with_flags_64, coherence_from_uniform_words_x8,
    step_toroidal_centroid_words_cached_64,
};
pub use self::pod::*;
pub use self::projection::*;
pub use self::rank1_kernels::*;
pub use self::rope::*;
pub(crate) use self::rope::{
    dispatch_fused_phase_clamp_and_subhead_words_core, steer_from_swar_words_core_safe,
};
pub use self::scalar_math::*;
pub(crate) use self::scalar_math::{
    branchless_maturity_scale_u32, fast_inv_len_f32, fast_inv_pos_normal_f32,
    fast_ln_1p_bounded_f32,
};
pub use self::simd_stats::*;
pub(crate) use self::simd_stats::{atan2_to_q7_word_x8_simd, scale_f32_slice_unrolled8};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use self::swar_simd::avx2_q7_mean_i8_from_vec64;
pub use self::swar_simd::*;
pub(crate) use self::swar_simd::{
    store_q7_word_at, swar_or_avx2_q7_add_sat_words_64, swar_u8_nonzero_msb_mask,
    words_to_q7_array_64,
};

const _: () = {
    assert!(compute_crc32_ieee(b"123456789") == 0xCBF4_3926);
    assert!(core::mem::size_of::<KvMotionDeltaPacket>() == 64);
    assert!(core::mem::align_of::<KvMotionDeltaPacket>() == 64);
};

#[cfg(test)]
mod tests;
