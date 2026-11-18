// name: src/core/src/traits/branchless/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.0 Foundation 3: [`Liminal`] root epistemic boundary supertrait, [`BranchlessKernel`]
//! blanket trait, 64-head phase algebra
//! (word-uniform accelerated coherence & circular resultant, register-resident unified-field
//! folding, Phase-Proximity Gravity manifold folding, zero-hyperparameter resonant pull,
//! $S^1$ geodesic weighted folding, Online Toroidal $k$-Means M-step centroid updates,
//! SWAR $S^1$ synergy fusion, Attention $\times$ DPSR phase-interference & Bayesian
//! stigmergic score modulation, Limina transition log-probability, and AVX2 64-head
//! embedding projection), and strictly constant-time subnormal-flushed lockstep AVX2 SIMD kernels.
//! Base LUTs, IEEE 802.3 CRC32 tables, scalar selectors, and 8-lane SWAR primitives are
//! defined in [`crate::traits::primitives`].

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
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

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

use crate::traits::primitives::{
    DEFAULT_ENTROPY_EPS, F32_MIN_POSITIVE_NORMAL, GcsoResult,
    INV_64_Q14_F32, INV_8_Q14_F32, KvMotionDeltaPacket,
    MAX_ATTENTION_HEADS, MAX_BAYESIAN_STIGMERGIC_DENSITY, MAX_PHASE_INTERFERENCE_INV_TEMP,
    MIN_PHASE_INTERFERENCE_INV_TEMP, POPCOUNT_RSQRT_LUT_257_REF, Q14_BIAS_X8_I32,
    Q7_FULL_COS_I16_REF, Q7_FULL_COS_REF, Q7_FULL_SINCOS_BIASED_U64_REF, SUBNORMAL_SQ_RISK_FLOOR,
    SWAR_SPLAT_BYTE_64, VON_MISES_KAPPA_AT_MAX_R, branchless_abs_f32,
    branchless_atan2_to_q7_fast, branchless_clamp_f32,
    branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_ln_f32, branchless_max_f32,
    branchless_phase_proximity_gravity_f32,
    branchless_rad_to_q7,
    branchless_relu_f32, branchless_ripa_clamp_rad, branchless_rotate_so2_pair,
    branchless_rsqrt_f32,
    branchless_select_f32, branchless_select_i8,
    branchless_select_u32,
    branchless_select_u8, branchless_select_usize, branchless_sincos_f32,
    branchless_sincos_half_pi_f32, branchless_sincos_q7,
    branchless_sqrt_f32, branchless_von_mises_kappa,
    fast_inv_len_f32, fast_inv_pos_normal_f32, fast_ln_1p_bounded_f32, fast_sqrt_unit_f32,
    is_word_array_uniform_64, load_q7_words_64, sanitize_inplace_gain_f32, store_q7_words_64,
    store_q7_x8_slice, swar_q7_circular_add_x8,
    swar_q7_circular_diff_x8, swar_q7_fold_s1_geodesic_x8,
    swar_q7_neg_sat_x8, swar_q7_qdps_filter_words_64,
    swar_q7_scale_le128_with_bias_x8,
};
#[cfg(test)]
use crate::traits::primitives::{branchless_inv_pos_f32, swar_q7_add_sat_x8};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use crate::traits::primitives::Q14_BIAS_X64_I32;
use crate::abi::{GCSO_ERROR_INVALID_ARGUMENT, gcso_q7_t};
use core::f32::consts::{LN_2, PI};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::f32::consts::FRAC_PI_2;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::traits::primitives::{Q7_PHASE_INV_SCALE_RAD, branchless_q7_circular_lerp};


pub mod kernel_trait;
pub mod phase_field;
pub mod simd_stats;
pub mod visual_srl;

pub use self::kernel_trait::*;
pub use self::phase_field::*;
pub use self::simd_stats::*;
pub use self::visual_srl::*;

#[cfg(test)]
mod tests;
