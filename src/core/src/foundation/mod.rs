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
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
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
#![allow(clippy::manual_unwrap_or)]
#![allow(clippy::too_many_arguments)]

pub mod aligned;
pub mod branchless;
pub mod cohomology;
pub mod lora;
pub mod lut_crc32;
pub mod pod;
pub mod projection;
pub mod rank1;
pub mod rope;
pub mod scalar_math;
pub mod swar_simd;

pub use self::aligned::*;
pub use self::branchless::*;
pub(crate) use self::branchless::{
    atan2_to_q7_word_x8_simd, branchless_q7_unified_field_fold_words_with_flags_64,
    branchless_toroidal_centroid_step_words_with_coh_uni_and_changed_64,
    coherence_from_uniform_words_x8, dot_and_norms_f32_unrolled4, scale_f32_slice_unrolled8,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use self::branchless::{
    avx2_coherence_from_diff_vec64, avx2_slerp_from_diff_vec64, avx2_vec64_to_words64,
    avx2_words64_to_vec64, avx2_words64_to_vec64_with_uniform,
};
pub use self::cohomology::*;
pub use self::lora::*;
pub use self::lut_crc32::*;
pub(crate) use self::lut_crc32::{
    branchless_popcount_rsqrt_u32, branchless_q7_default_gravity_from_diff_u8,
    compute_aligned64_payload_crc32_ieee, crc32_crystal_be24_final_u32,
    crc32_crystal_gm36_final_u32, crc32_crystal_stamp_contrib_128b, crc32_crystal_zoff16_final_u32,
    crc32_fold_stamped_crystal_128b, crc32_slice16_dep_lo0, crc32_slice16_indep_hi0_u32,
    crc32_slice16_indep_hi1_u32, crc32_slice16_indep_lo1_u32, crc32_slice16_step_u64x2,
    crc32_step_block64_words, crc32_step_l16, crc32_step_l3, crc32_step_l32, crc32_step_l4,
    crc32_step_l5, crc32_step_l6, crc32_step_l8, crc32_update_crystal_absorption_delta,
    crc32_update_crystal_energy_and_mass_delta, crc32_update_crystal_stationary_fields_delta,
    update_aligned64_blocks_crc32_raw, CRC32_CRYSTAL_ACTION_BYTE57_REF,
    CRC32_CRYSTAL_AID_BYTE0_REF, CRC32_CRYSTAL_AID_BYTE1_REF, CRC32_CRYSTAL_PROV_BYTE0_REF,
    CRC32_CRYSTAL_PROV_BYTE1_REF, CRC32_CRYSTAL_SC_BYTE0_REF, CRC32_CRYSTAL_SLOT_BYTE56_LUT_8,
    CRC32_CRYSTAL_ZOFF_BYTE1_REF, CRC32_CRYSTAL_ZOFF_BYTE2_REF, CRC32_CRYSTAL_ZOFF_BYTE3_REF,
    CRC32_IEEE_SLICE16_REF, CRC32_MAPPED_STORAGE_DESC_BASE_SEED,
    CRYSTAL_CRC32_DEFAULT_W5_FINAL_CONTRIB, CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB,
    CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB, CRYSTAL_CRC32_RBAR_ONE_FINAL_CONTRIB,
    CRYSTAL_CRC32_W0_FINAL_CONTRIB, MAX_BAYESIAN_STIGMERGIC_DENSITY, POPCOUNT_RSQRT_LUT_257_REF,
    Q7_FULL_COS_REF, VON_MISES_KAPPA_AT_MAX_R_BITS,
};
pub use self::pod::*;
pub use self::projection::*;
pub use self::rank1::*;
pub use self::rope::*;
pub(crate) use self::rope::{
    dispatch_fused_phase_clamp_and_subhead_words_core, steer_from_swar_words_core_safe,
};
pub use self::scalar_math::*;
pub(crate) use self::scalar_math::{
    branchless_maturity_scale_u32, fast_inv_len_f32, fast_inv_pos_normal_f32,
    fast_ln_1p_bounded_f32,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use self::swar_simd::avx2_q7_mean_i8_from_vec64;
pub use self::swar_simd::*;
pub(crate) use self::swar_simd::{
    load_q7_word_at, store_q7_word_at, swar_or_avx2_q7_add_sat_words_64, swar_u8_nonzero_msb_mask,
    words_to_q7_array_64,
};

const _: () = {
    assert!(compute_crc32_ieee(b"123456789") == 0xCBF4_3926);
    assert!(core::mem::size_of::<KvMotionDeltaPacket>() == 64);
    assert!(core::mem::align_of::<KvMotionDeltaPacket>() == 64);
};

#[cfg(test)]
mod tests;
