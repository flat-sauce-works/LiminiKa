// name: src/core/src/traits/branchless.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.0 Foundation 3: `BranchlessKernel` blanket trait, `#![no_std]` constant-time bitwise
//! multiplexers, 64B-aligned 256-entry `f32`, Q14 (`i16` / biased `u64`), and 4KB Slice-by-4
//! IEEE 802.3 CRC32 LUTs, division-free (`0 fdiv`) IEEE-754 square roots and reciprocals,
//! 4/8-lane SWAR Q7 & `u8` arithmetic, 64-head phase algebra (word-uniform accelerated coherence
//! & circular resultant, register-resident unified-field folding, Phase-Proximity Gravity
//! manifold folding, zero-hyperparameter resonant pull, $S^1$ geodesic weighted folding,
//! Online Toroidal $k$-Means M-step centroid updates, SWAR $S^1$ synergy fusion,
//! Attention $\times$ DPSR phase-interference & Bayesian stigmergic score modulation, Limina
//! transition log-probability, and AVX2 64-head embedding projection), and strictly constant-time
//! subnormal-flushed lockstep AVX2 SIMD kernels.

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

use core::f32::consts::{FRAC_PI_2, LN_2, PI};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

use super::primitives::{
    DEFAULT_ENTROPY_EPS, DEFAULT_RIPA_CLAMP_MAX_RAD, MAX_ATTENTION_HEADS, Q7_MAX, Q7_MIN,
    Q7_PHASE_INV_SCALE_RAD, Q7_PHASE_SCALE_RAD,
};
use crate::abi::{gcso_pointer_trail_t, gcso_q7_t};

const SWAR_SIGN_MASK_64: u64 = 0x8080_8080_8080_8080;
const SWAR_LOW7_MASK_64: u64 = 0x7F7F_7F7F_7F7F_7F7F;
const SWAR_EVEN_BYTES_64: u64 = 0x00FF_00FF_00FF_00FF;
const SWAR_ODD_BYTES_64: u64 = 0xFF00_FF00_FF00_FF00;
const SWAR_SPLAT_BYTE_64: u64 = 0x0101_0101_0101_0101;
const SWAR_SUM_U16_64: u64 = 0x0001_0001_0001_0001;
const SWAR_SIGN_MASK_32: u32 = 0x8080_8080;
const SWAR_LOW7_MASK_32: u32 = 0x7F7F_7F7F;
const INV_TWO_PI_F64: f64 = 1.0 / (2.0 * core::f64::consts::PI);
const TWO_PI_F64: f64 = 2.0 * core::f64::consts::PI;
const INV_LN2_F32: f32 = core::f32::consts::LOG2_E;
const LN2_HI_F32: f32 = 0.693_145_75;
const LN2_LO_F32: f32 = 1.428_606_8e-6;
const INPLACE_GAIN_OVERFLOW_CEIL: f32 = 1.0e30;
const MAX_BAYESIAN_STIGMERGIC_DENSITY: f32 = 16.0;
const INV_64_Q14_F32: f32 = f32::from_bits((127 - 20) << 23);
const INV_8_Q14_F32: f32 = f32::from_bits((127 - 17) << 23);
const SUBNORMAL_SQ_RISK_FLOOR: f32 = 1.0e-18;
const Q14_BIAS_I32: i32 = 16_384;
const Q14_BIAS_X8_I32: i32 = 8 * Q14_BIAS_I32;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
const Q14_BIAS_X64_I32: i32 = 64 * Q14_BIAS_I32;

/// Minimum positive normal IEEE-754 `f32` value ($2^{-126} \approx 1.17549435 \times 10^{-38}$).
pub const F32_MIN_POSITIVE_NORMAL: f32 = f32::from_bits(0x0080_0000);
/// Default upper ceiling for token Z-score inputs ($10.0\sigma$).
pub const DEFAULT_MAX_VALID_Z_SCORE: f32 = 10.0;
/// Precomputed normalization scale `Q7_PHASE_SCALE_RAD / DEFAULT_RIPA_CLAMP_MAX_RAD`.
pub const DEFAULT_RIPA_Q7_TO_NORM_SCALE: f32 = Q7_PHASE_SCALE_RAD / DEFAULT_RIPA_CLAMP_MAX_RAD;
/// Minimum clamped inverse-temperature scale for destructive phase interference ($0.25$).
pub const MIN_PHASE_INTERFERENCE_INV_TEMP: f32 = 0.25;
/// Maximum clamped inverse-temperature scale for constructive phase interference ($4.0$).
pub const MAX_PHASE_INTERFERENCE_INV_TEMP: f32 = 4.0;
/// Default softening epsilon $\epsilon_{\mathrm{grav}}$ for Phase-Proximity Gravity on $SO(2)^{64}$ ($0.25$).
pub const DEFAULT_PHASE_GRAVITY_EPS: f32 = 0.25;

const TANH_LUT_MAX_X: f32 = 4.97;
const TANH_LUT_STEP: f32 = TANH_LUT_MAX_X / 256.0;
const TANH_LUT_INV_STEP: f32 = 256.0 / TANH_LUT_MAX_X;

#[repr(C, align(64))]
pub(crate) struct AlignedLut256<T>(pub [T; 256]);

#[repr(C, align(64))]
pub(crate) struct AlignedLut257<T>(pub [T; 257]);

#[repr(C, align(64))]
pub(crate) struct AlignedLut2x257<T>(pub [[T; 257]; 2]);

const VON_MISES_KAPPA_SPLIT_R: f32 = 0.875;
const VON_MISES_KAPPA_MAX_R: f32 = 0.995;
const VON_MISES_KAPPA_LO_STEP: f32 = VON_MISES_KAPPA_SPLIT_R / 256.0;
const VON_MISES_KAPPA_HI_STEP: f32 = (VON_MISES_KAPPA_MAX_R - VON_MISES_KAPPA_SPLIT_R) / 256.0;
const VON_MISES_KAPPA_LO_SCALE: f32 = 256.0 / VON_MISES_KAPPA_SPLIT_R;
const VON_MISES_KAPPA_HI_SCALE: f32 = 256.0 / (VON_MISES_KAPPA_MAX_R - VON_MISES_KAPPA_SPLIT_R);

#[repr(C, align(64))]
pub(crate) struct AlignedLut4x256<T>(pub [[T; 256]; 4]);

#[repr(C, align(64))]
pub(crate) struct AlignedLut16x256<T>(pub [[T; 256]; 16]);

/// 64B-aligned 16KB Slice-by-16 IEEE 802.3 CRC32 lookup table (`0xEDB8_8320` reflected polynomial)
/// enabling 16-byte (`2 x u64`) parallel XOR-table folding per step (`16x` fewer serial dependency
/// steps than byte-serial lookup and `2x` fewer than Slice-by-8 on `.gcso` ZIMMS containers and
/// 128-byte `CrystallizedAttractorRecord` blocks).
pub(crate) const CRC32_IEEE_SLICE16_LUT: AlignedLut16x256<u32> = {
    let mut lut = [[0u32; 256]; 16];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0usize;
        while bit < 8 {
            let mask = (c & 1).wrapping_neg();
            c = (c >> 1) ^ (0xEDB8_8320 & mask);
            bit += 1;
        }
        lut[0][i] = c;
        i += 1;
    }
    let mut s = 1usize;
    while s < 16 {
        let mut idx = 0usize;
        while idx < 256 {
            let prev = lut[s - 1][idx];
            lut[s][idx] = (prev >> 8) ^ lut[0][(prev & 0xFF) as usize];
            idx += 1;
        }
        s += 1;
    }
    AlignedLut16x256(lut)
};

const Q7_QUADRANT_SIN_LUT: [f32; 65] = {
    let mut lut = [0.0_f32; 65];
    let mut k = 1usize;
    while k < 64 {
        let x = (k as f32) * Q7_PHASE_SCALE_RAD;
        let x2 = x * x;
        lut[k] = x
            * (1.0
                + x2 * (-0.166_666_67
                    + x2 * (0.008_333_334
                        + x2 * (-0.000_198_412_7
                            + x2 * (0.000_002_755_732 + x2 * -0.000_000_025_052_108)))));
        k += 1;
    }
    lut[64] = 1.0;
    lut
};

pub(crate) const Q7_FULL_SIN_LUT: AlignedLut256<f32> = {
    let mut lut = [0.0_f32; 256];
    let mut b = 0usize;
    while b < 256 {
        let phase_q7 = (b as u8) as gcso_q7_t;
        let u = branchless_q7_abs_u8(phase_q7);
        let gt_64 = u > 64;
        let k = branchless_select_u8(gt_64, 128u8.wrapping_sub(u), u) as usize;
        let sin_mag = Q7_QUADRANT_SIN_LUT[k];
        let sin_neg = (phase_q7 < 0) & (k != 0);
        lut[b] = f32::from_bits(sin_mag.to_bits() | ((sin_neg as u32) << 31));
        b += 1;
    }
    AlignedLut256(lut)
};

pub(crate) const Q7_FULL_COS_LUT: AlignedLut256<f32> = {
    let mut lut = [1.0_f32; 256];
    let mut b = 0usize;
    while b < 256 {
        let phase_q7 = (b as u8) as gcso_q7_t;
        let u = branchless_q7_abs_u8(phase_q7);
        let gt_64 = u > 64;
        let k = branchless_select_u8(gt_64, 128u8.wrapping_sub(u), u) as usize;
        let cos_mag = Q7_QUADRANT_SIN_LUT[64 - k];
        let cos_neg = gt_64 & (k != 64);
        lut[b] = f32::from_bits(cos_mag.to_bits() | ((cos_neg as u32) << 31));
        b += 1;
    }
    AlignedLut256(lut)
};

pub(crate) const Q7_FULL_SINCOS_LUT: AlignedLut256<(f32, f32)> = {
    let mut lut = [(0.0_f32, 1.0_f32); 256];
    let mut b = 0usize;
    while b < 256 {
        lut[b] = (Q7_FULL_SIN_LUT.0[b], Q7_FULL_COS_LUT.0[b]);
        b += 1;
    }
    AlignedLut256(lut)
};

/// 64B-aligned 2KB interleaved `(sin, cos)` LUT indexed directly by `phase_q7 as u8`.
pub(crate) const Q7_SINCOS_LUT_256: AlignedLut256<(f32, f32)> = Q7_FULL_SINCOS_LUT;

/// 64B-aligned 512B Q14 fixed-point (`1.0 == 16384`) cosine LUT for 1-cycle integer ALU reductions.
pub(crate) const Q7_FULL_COS_I16_LUT: AlignedLut256<i16> = {
    let mut lut = [16384_i16; 256];
    let mut b = 0usize;
    while b < 256 {
        let c = Q7_FULL_COS_LUT.0[b] * 16384.0;
        lut[b] = (c + if c >= 0.0 { 0.5 } else { -0.5 }) as i16;
        b += 1;
    }
    AlignedLut256(lut)
};

/// 64B-aligned 1KB Q14 fixed-point `[sin_q14, cos_q14]` LUT for 1-cycle integer ALU reductions.
pub(crate) const Q7_FULL_SINCOS_I16_LUT: AlignedLut256<[i16; 2]> = {
    let mut lut = [[0_i16, 16384_i16]; 256];
    let mut b = 0usize;
    while b < 256 {
        let s = Q7_FULL_SIN_LUT.0[b] * 16384.0;
        let c = Q7_FULL_COS_LUT.0[b] * 16384.0;
        lut[b] = [
            (s + if s >= 0.0 { 0.5 } else { -0.5 }) as i16,
            (c + if c >= 0.0 { 0.5 } else { -0.5 }) as i16,
        ];
        b += 1;
    }
    AlignedLut256(lut)
};

/// 64B-aligned 2KB SWAR-packed biased Q14 `(sin, cos)` LUT where each `u64` entry stores:
/// - Bits `0..31`: `(cos_q14 + 16384) as u32` $\in [0, 32768]$
/// - Bits `32..63`: `((sin_q14 + 16384) as u32 as u64) << 32`
///
/// Summing 64 entries in a single `u64` register (`<= 64 * 32768 = 2^21 << 2^32`) accumulates
/// both `sum_sin_q14` and `sum_cos_q14` simultaneously with zero cross-lane carry.
pub(crate) const Q7_FULL_SINCOS_BIASED_U64_LUT: AlignedLut256<u64> = {
    let mut lut = [0u64; 256];
    let mut b = 0usize;
    while b < 256 {
        let [s_i16, c_i16] = Q7_FULL_SINCOS_I16_LUT.0[b];
        let u_sin = ((s_i16 as i32) + Q14_BIAS_I32) as u32 as u64;
        let u_cos = ((c_i16 as i32) + Q14_BIAS_I32) as u32 as u64;
        lut[b] = (u_sin << 32) | u_cos;
        b += 1;
    }
    AlignedLut256(lut)
};

const TANH_LUT_257: AlignedLut257<f32> = {
    let mut lut = [0.0_f32; 257];
    let mut k = 1usize;
    while k <= 256 {
        let x = (k as f32) * TANH_LUT_STEP;
        let x2 = x * x;
        let x4 = x2 * x2;
        lut[k] = (x * (945.0 + 105.0 * x2 + x4)) / (945.0 + 420.0 * x2 + 15.0 * x4);
        k += 1;
    }
    AlignedLut257(lut)
};

/// Evaluates $\tanh(u)$ for a non-negative finite float $u \ge 0$ using the 64-byte aligned
/// 1KB `TANH_LUT_257_REF` table (`step = 1/64`) and a 2nd-order analytic ODE Taylor expansion
/// ($\tanh' = 1 - y^2$, $\tanh'' = -2y(1 - y^2)$) in `0` divisions and `0` reciprocals.
#[inline(always)]
#[must_use]
pub const fn tanh_lut_eval_nonneg(u_nonneg: f32) -> f32 {
    let clamped = branchless_select_f32(u_nonneg < TANH_LUT_MAX_X, u_nonneg, TANH_LUT_MAX_X);
    let idx_raw = (clamped * TANH_LUT_INV_STEP) as usize;
    let idx = branchless_select_usize(idx_raw < 256, idx_raw, 255);
    let d = clamped - (idx as f32) * TANH_LUT_STEP;
    let y0 = TANH_LUT_257_REF[idx];
    y0 + d * (1.0 - y0 * y0) * (1.0 - y0 * d)
}

#[inline(always)]
const fn sincos_ripa_poly(safe_theta: f32) -> (f32, f32) {
    let x2 = safe_theta * safe_theta;
    let sin_t =
        safe_theta * (1.0 + x2 * (-0.166_666_67 + x2 * (0.008_333_334 - x2 * 0.000_198_412_7)));
    let cos_t = 1.0
        + x2 * (-0.5 + x2 * (0.041_666_668 + x2 * (-0.001_388_888_9 + x2 * 0.000_024_801_587)));
    (sin_t, cos_t)
}

pub(crate) const Q7_DEFAULT_RIPA_SINCOS_LUT_256: AlignedLut256<(f32, f32)> = {
    let mut lut = [(0.0_f32, 1.0_f32); 256];
    let mut b = 0usize;
    while b < 256 {
        let phase_q7 = (b as u8) as gcso_q7_t;
        let u = branchless_q7_abs_u8(phase_q7);
        let safe_theta = DEFAULT_RIPA_CLAMP_MAX_RAD
            * tanh_lut_eval_nonneg((u as f32) * DEFAULT_RIPA_Q7_TO_NORM_SCALE);
        let (sin_mag, cos_t) = sincos_ripa_poly(safe_theta);
        let sin_neg = (phase_q7 < 0) & (u != 0);
        lut[b] = (
            f32::from_bits(sin_mag.to_bits() | ((sin_neg as u32) << 31)),
            cos_t,
        );
        b += 1;
    }
    AlignedLut256(lut)
};

pub(crate) const Q7_GEODESIC_DEFAULT_RIPA_SINCOS_LUT_256: AlignedLut256<(f32, f32)> = {
    let mut table = [(0.0_f32, 1.0_f32); 256];
    let mut idx = 0usize;
    while idx < 256 {
        let folded = branchless_q7_fold_s1_geodesic(idx as i8);
        table[idx] = Q7_DEFAULT_RIPA_SINCOS_LUT_256.0[(folded as u8) as usize];
        idx += 1;
    }
    AlignedLut256(table)
};

pub(crate) const POPCOUNT_RSQRT_LUT_257: AlignedLut257<f32> = {
    let mut lut = [0.0_f32; 257];
    let mut k = 1usize;
    while k <= 256 {
        lut[k] = branchless_rsqrt_f32(k as f32);
        k += 1;
    }
    AlignedLut257(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 256-entry `(sin, cos)` Q7 table.
pub(crate) const Q7_SINCOS_LUT_256_REF: &[(f32, f32); 256] = &Q7_SINCOS_LUT_256.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry `sin` Q7 table.
pub(crate) const Q7_FULL_SIN_REF: &[f32; 256] = &Q7_FULL_SIN_LUT.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry `cos` Q7 table.
pub(crate) const Q7_FULL_COS_REF: &[f32; 256] = &Q7_FULL_COS_LUT.0;

/// 64B-aligned 2KB precomputed `(align_gain, inv_denom)` table for [`branchless_phase_proximity_gravity_f32`]
/// under `DEFAULT_PHASE_GRAVITY_EPS` (`0.25`) across all 256 Q7 circular difference bytes `0..=255`,
/// enabling 0-ULP bit-exact gravity evaluation in 1 LUT load + 2 `vmulss` (`0` Newton-Raphson iterations).
pub(crate) const Q7_DEFAULT_GRAVITY_PAIR_LUT_256: AlignedLut256<(f32, f32)> = {
    let mut lut = [(1.0_f32, 2.0_f32 / DEFAULT_PHASE_GRAVITY_EPS); 256];
    let mut b = 1usize;
    while b < 256 {
        let coh = Q7_FULL_COS_LUT.0[b];
        let chord_dist_sq = 2.0 * (1.0 - coh);
        let denom = chord_dist_sq + DEFAULT_PHASE_GRAVITY_EPS;
        let align_gain = branchless_max_f32(0.05, 1.0 + coh);
        let inv_denom = fast_inv_pos_normal_f32(denom);
        lut[b] = (align_gain, inv_denom);
        b += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 256-entry default gravity `(align_gain, inv_denom)` table.
pub(crate) const Q7_DEFAULT_GRAVITY_PAIR_REF: &[(f32, f32); 256] =
    &Q7_DEFAULT_GRAVITY_PAIR_LUT_256.0;

/// Evaluates Phase-Proximity Gravity $F_{\mathrm{grav}}(M_{\mathrm{grav}}, \cos(\Delta\theta_{\mathrm{Q7}}), \epsilon_{\mathrm{default}})$
/// directly from a uniform Q7 circular difference byte `diff_u8` in 1 LUT load + 2 `vmulss` (`0` Newton-Raphson reciprocals, `0` ULP drift).
#[inline(always)]
#[must_use]
pub(crate) const fn branchless_q7_default_gravity_from_diff_u8(
    gravity_mass: f32,
    diff_u8: u8,
) -> f32 {
    let clean_mass = branchless_flush_subnormal_f32(branchless_relu_f32(gravity_mass));
    let (align_gain, inv_denom) = Q7_DEFAULT_GRAVITY_PAIR_REF[diff_u8 as usize];
    branchless_flush_subnormal_f32((clean_mass * align_gain) * inv_denom)
}
/// Promoted `.rodata` reference to the 64B-aligned 256-entry Q14 `i16` cosine table.
pub(crate) const Q7_FULL_COS_I16_REF: &[i16; 256] = &Q7_FULL_COS_I16_LUT.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry SWAR-packed biased Q14 `(sin, cos)` table.
pub(crate) const Q7_FULL_SINCOS_BIASED_U64_REF: &[u64; 256] = &Q7_FULL_SINCOS_BIASED_U64_LUT.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry default RIPA `(sin, cos)` table.
pub(crate) const Q7_DEFAULT_RIPA_SINCOS_REF: &[(f32, f32); 256] = &Q7_DEFAULT_RIPA_SINCOS_LUT_256.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry geodesic default RIPA `(sin, cos)` table.
pub(crate) const Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_GEODESIC_DEFAULT_RIPA_SINCOS_LUT_256.0;

#[inline(always)]
const fn build_q7_fused_self_ripa_sincos_lut(fold_geodesic: bool) -> AlignedLut256<(f32, f32)> {
    let mut table = [(0.0_f32, 1.0_f32); 256];
    let mut i = 0usize;
    while i < 256 {
        let (sin_base, cos_base) = Q7_DEFAULT_RIPA_SINCOS_LUT_256.0[i];
        let (sin_pspm, cos_pspm) = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_RIPA_SINCOS_LUT_256.0[i]
        } else {
            Q7_DEFAULT_RIPA_SINCOS_LUT_256.0[i]
        };
        let sin_comb = sin_base * cos_pspm + cos_base * sin_pspm;
        let cos_comb = cos_base * cos_pspm - sin_base * sin_pspm;
        table[i] = (sin_comb, cos_comb);
        i += 1;
    }
    AlignedLut256(table)
}

/// Precomputed 64-byte-aligned 2KB LUT of fused $SO(2)$ double-rotation `(sin(2*theta), cos(2*theta))`
/// when `base_ripa_q7 == pspm_slot_q7` under `DEFAULT_RIPA_CLAMP_MAX_RAD` (non-geodesic PSPM).
pub(crate) const Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_LUT_256: AlignedLut256<(f32, f32)> =
    build_q7_fused_self_ripa_sincos_lut(false);

/// Precomputed 64-byte-aligned 2KB LUT of fused $SO(2)$ double-rotation `(sin(theta_b + theta_p), cos(theta_b + theta_p))`
/// when `base_ripa_q7 == pspm_slot_q7` under `DEFAULT_RIPA_CLAMP_MAX_RAD` (geodesic PSPM).
pub(crate) const Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_LUT_256: AlignedLut256<(f32, f32)> =
    build_q7_fused_self_ripa_sincos_lut(true);

pub(crate) const Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_LUT_256.0;

pub(crate) const Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_LUT_256.0;
/// Promoted `.rodata` reference to the 64B-aligned 257-entry `tanh` table.
pub(crate) const TANH_LUT_257_REF: &[f32; 257] = &TANH_LUT_257.0;
/// Promoted `.rodata` reference to the 64B-aligned 257-entry popcount `rsqrt` table.
pub(crate) const POPCOUNT_RSQRT_LUT_257_REF: &[f32; 257] = &POPCOUNT_RSQRT_LUT_257.0;

#[inline(always)]
pub(crate) const fn branchless_popcount_rsqrt_u32(popcount: u32) -> f32 {
    let idx = branchless_select_usize((popcount as usize) <= 256, popcount as usize, 256);
    POPCOUNT_RSQRT_LUT_257_REF[idx]
}

/// Promoted `.rodata` reference to the 64B-aligned 16KB Slice-by-16 IEEE 802.3 CRC32 table,
/// ensuring debug builds (`opt-level = 0`, `cargo test`) index directly into `.rodata` via an
/// 8-byte pointer rather than materializing 16KB stack temporaries per table lookup.
pub(crate) const CRC32_IEEE_SLICE16_REF: &[[u32; 256]; 16] = &CRC32_IEEE_SLICE16_LUT.0;

#[inline(always)]
const fn crc32_slice4_step(crc: u32, b0: u8, b1: u8, b2: u8, b3: u8) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    let word = u32::from_le_bytes([b0, b1, b2, b3]) ^ crc;
    lut[3][(word & 0xFF) as usize]
        ^ lut[2][((word >> 8) & 0xFF) as usize]
        ^ lut[1][((word >> 16) & 0xFF) as usize]
        ^ lut[0][(word >> 24) as usize]
}

#[inline(always)]
pub(crate) const fn crc32_slice8_step_u64(crc: u32, word64_le: u64) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    let lo = (word64_le as u32) ^ crc;
    let hi = (word64_le >> 32) as u32;
    (lut[7][(lo & 0xFF) as usize]
        ^ lut[6][((lo >> 8) & 0xFF) as usize]
        ^ lut[5][((lo >> 16) & 0xFF) as usize]
        ^ lut[4][(lo >> 24) as usize])
        ^ (lut[3][(hi & 0xFF) as usize]
            ^ lut[2][((hi >> 8) & 0xFF) as usize]
            ^ lut[1][((hi >> 16) & 0xFF) as usize]
            ^ lut[0][(hi >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_slice16_dep_lo0(lo0: u32) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    (lut[15][(lo0 & 0xFF) as usize]
        ^ lut[14][((lo0 >> 8) & 0xFF) as usize])
        ^ (lut[13][((lo0 >> 16) & 0xFF) as usize]
            ^ lut[12][(lo0 >> 24) as usize])
}

/// 64B-aligned 4KB GF(2) linear operator table for $L^2(x)$ (`32`-byte zero-shift modulo IEEE 802.3 CRC32),
/// enabling $2 \times 32\text{B}$ parallel tree reduction per 64B cacheline (`2x` shorter serial dependency chain).
pub(crate) const CRC32_IEEE_L2_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let shift = byte_pos * 8;
        let mut b = 0usize;
        while b < 256 {
            let x = (b as u32) << shift;
            lut[byte_pos][b] = crc32_slice16_dep_lo0(crc32_slice16_dep_lo0(x));
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) linear operator table for $L^3(x) = L^1(L^2(x))$ (`48`-byte zero-shift
/// modulo IEEE 802.3 CRC32), enabling 4-way parallel 16B chunk evaluation per 64B cacheline
/// (`2` serial L1D load stages instead of `3..4`).
pub(crate) const CRC32_IEEE_L3_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let l2 = CRC32_IEEE_L2_LUT.0[byte_pos][b];
            lut[byte_pos][b] = crc32_slice16_dep_lo0(l2);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) linear operator table for $L^4(x)$ (`64`-byte zero-shift modulo IEEE 802.3 CRC32),
/// enabling single-step 64B block folding (`128B` `CrystallizedAttractorRecord` in `2` parallel 64B halves).
pub(crate) const CRC32_IEEE_L4_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let l2 = CRC32_IEEE_L2_LUT.0[byte_pos][b];
            lut[byte_pos][b] = crc32_slice16_dep_lo0(crc32_slice16_dep_lo0(l2));
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) linear operator table for $L^8(x) = L^4(L^4(x))$ (`128`-byte zero-shift
/// modulo IEEE 802.3 CRC32), enabling single-step 128B block folding across stamped
/// `CrystallizedAttractorRecord` arrays and 256B ($4 \times 64\text{B}$) container blocks.
pub(crate) const CRC32_IEEE_L8_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let l4 = CRC32_IEEE_L4_LUT.0[byte_pos][b];
            let l6 = crc32_slice16_dep_lo0(crc32_slice16_dep_lo0(l4));
            lut[byte_pos][b] = crc32_slice16_dep_lo0(crc32_slice16_dep_lo0(l6));
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB $L^2(x)$ 32-byte shift table.
pub(crate) const CRC32_IEEE_L2_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L2_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^3(x)$ 48-byte shift table.
pub(crate) const CRC32_IEEE_L3_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L3_LUT.0;
/// 64B-aligned 4KB GF(2) operator table for $L^5(x) = L^1(L^4(x))$ (advancing a 32-bit CRC state by 80 zero bytes).
pub(crate) const CRC32_IEEE_L5_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            lut[byte_pos][b] = crc32_slice16_dep_lo0(CRC32_IEEE_L4_LUT.0[byte_pos][b]);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) operator table for $L^6(x) = L^2(L^4(x))$ (advancing a 32-bit CRC state by 96 zero bytes).
pub(crate) const CRC32_IEEE_L6_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            lut[byte_pos][b] = crc32_slice16_dep_lo0(CRC32_IEEE_L5_LUT.0[byte_pos][b]);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB $L^4(x)$ 64-byte shift table.
pub(crate) const CRC32_IEEE_L4_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L4_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^5(x)$ 80-byte shift table.
pub(crate) const CRC32_IEEE_L5_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L5_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^6(x)$ 96-byte shift table.
pub(crate) const CRC32_IEEE_L6_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L6_LUT.0;
/// 64B-aligned 4KB GF(2) operator table for $L^{16}(x) = L^8(L^8(x))$ (`256`-byte / 2-record zero-shift).
pub(crate) const CRC32_IEEE_L16_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            lut[byte_pos][b] = crc32_step_l8(CRC32_IEEE_L8_LUT.0[byte_pos][b]);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) operator table for $L^{32}(x) = L^{16}(L^{16}(x))$ (`512`-byte / 4-record zero-shift).
pub(crate) const CRC32_IEEE_L32_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let l16 = CRC32_IEEE_L16_LUT.0[byte_pos][b];
            lut[byte_pos][b] = crc32_step_l8(crc32_step_l8(l16));
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB $L^8(x)$ 128-byte shift table.
pub(crate) const CRC32_IEEE_L8_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L8_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^{16}(x)$ 256-byte shift table.
pub(crate) const CRC32_IEEE_L16_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L16_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^{32}(x)$ 512-byte shift table.
pub(crate) const CRC32_IEEE_L32_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L32_LUT.0;

#[inline(always)]
pub(crate) const fn crc32_step_l2(x: u32) -> u32 {
    let lut = CRC32_IEEE_L2_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l3(x: u32) -> u32 {
    let lut = CRC32_IEEE_L3_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l4(x: u32) -> u32 {
    let lut = CRC32_IEEE_L4_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l5(x: u32) -> u32 {
    let lut = CRC32_IEEE_L5_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l6(x: u32) -> u32 {
    let lut = CRC32_IEEE_L6_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l8(x: u32) -> u32 {
    let lut = CRC32_IEEE_L8_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l16(x: u32) -> u32 {
    let lut = CRC32_IEEE_L16_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l32(x: u32) -> u32 {
    let lut = CRC32_IEEE_L32_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

/// 64B-aligned 4KB GF(2) affine operator table fusing $L^4(L_{12}(\text{record\_crc32})) \oplus (\neg\text{record\_crc32}) \oplus L^8(\text{0xFFFF\_FFFF})$
/// into a single 4-lookup stage (`0` branches, `1` lookup stage instead of `2` serial stages).
pub(crate) const CRC32_IEEE_CRYSTAL_STAMP_LUT: AlignedLut4x256<u32> = {
    const L8_ONES: u32 = crc32_step_l8(0xFFFF_FFFFu32);
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let shift = byte_pos * 8;
        let mut b = 0usize;
        while b < 256 {
            let d = CRC32_IEEE_SLICE16_LUT.0[11 - byte_pos][b];
            let mut v = crc32_step_l4(d) ^ ((b as u32) << shift);
            if byte_pos == 0 {
                v ^= 0xFFFF_FFFFu32 ^ L8_ONES;
            }
            lut[byte_pos][b] = v;
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB 1-stage crystal stamp CRC32 folding table.
pub(crate) const CRC32_IEEE_CRYSTAL_STAMP_REF: &[[u32; 256]; 4] = &CRC32_IEEE_CRYSTAL_STAMP_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for low-byte `sample_count` increments
/// (`offset 12..16` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 112 bytes),
/// updating a stationary centroid's `checksum_crc32` in 1 table lookup (`1` cycle).
pub(crate) const CRC32_CRYSTAL_SC_BYTE0_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        let l0 = CRC32_IEEE_SLICE16_LUT.0[3][d];
        lut[d] = crc32_step_l4(crc32_step_l2(crc32_slice16_dep_lo0(l0)));
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `sample_count` byte-0 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_SC_BYTE0_REF: &[u32; 256] = &CRC32_CRYSTAL_SC_BYTE0_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for low-byte `anchor_id` values
/// (`offset 8..12` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 116 bytes via $L^7$),
/// evaluating `anchor_id < 256` contributions in 1 table lookup (`1` cycle, `0` serial $L^3/L^4$ stages).
pub(crate) const CRC32_CRYSTAL_AID_BYTE0_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        let l0 = CRC32_IEEE_SLICE16_LUT.0[7][d];
        lut[d] = crc32_step_l4(crc32_step_l3(l0));
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `anchor_id` byte-0 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_AID_BYTE0_REF: &[u32; 256] = &CRC32_CRYSTAL_AID_BYTE0_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-1 `anchor_id` values
/// (`offset 9` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 116 bytes via $L^7$),
/// evaluating `anchor_id < 65536` contributions in 2 parallel table lookups (`0` serial $L^3/L^4$ stages).
pub(crate) const CRC32_CRYSTAL_AID_BYTE1_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        let l0 = CRC32_IEEE_SLICE16_LUT.0[6][d];
        lut[d] = crc32_step_l4(crc32_step_l3(l0));
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `anchor_id` byte-1 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_AID_BYTE1_REF: &[u32; 256] = &CRC32_CRYSTAL_AID_BYTE1_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-1 (`(zoff >> 8) & 0xFF`, `offset 17`)
/// of `zimms_section_offset48` inside a 128-byte `CrystallizedAttractorRecord`, pre-shifted by
/// 96 bytes via $L^6$ so 4KB-aligned ZIMMS section offsets fold in 1 lookup stage (`0` serial $L^6$ stages).
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE1_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l6(CRC32_IEEE_SLICE16_LUT.0[14][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `zimms_section_offset48` byte-1 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE1_REF: &[u32; 256] = &CRC32_CRYSTAL_ZOFF_BYTE1_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-2 (`(zoff >> 16) & 0xFF`, `offset 18`)
/// of `zimms_section_offset48` inside a 128-byte `CrystallizedAttractorRecord`, pre-shifted by
/// 96 bytes via $L^6$ so 64KB-aligned ZIMMS section offsets fold in 1 lookup stage (`0` serial $L^6$ stages).
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE2_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l6(CRC32_IEEE_SLICE16_LUT.0[13][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `zimms_section_offset48` byte-2 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE2_REF: &[u32; 256] = &CRC32_CRYSTAL_ZOFF_BYTE2_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-3 (`(zoff >> 24) & 0xFF`, `offset 19`)
/// of `zimms_section_offset48` inside a 128-byte `CrystallizedAttractorRecord`, pre-shifted by
/// 96 bytes via $L^6$ so 64KB-aligned ZIMMS section offsets fold in 1 lookup stage (`0` serial $L^6$ stages).
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE3_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l6(CRC32_IEEE_SLICE16_LUT.0[12][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `zimms_section_offset48` byte-3 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE3_REF: &[u32; 256] = &CRC32_CRYSTAL_ZOFF_BYTE3_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-0 `provenance_bits` values
/// (`offset 48` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 64 bytes via $L^5$),
/// evaluating low-byte provenance bitmasks in 1 table lookup (`0` serial $L^4$ stages).
pub(crate) const CRC32_CRYSTAL_PROV_BYTE0_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l4(CRC32_IEEE_SLICE16_LUT.0[15][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `provenance_bits` byte-0 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_PROV_BYTE0_REF: &[u32; 256] = &CRC32_CRYSTAL_PROV_BYTE0_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-1 `provenance_bits` values
/// (`offset 49` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 64 bytes via $L^5$),
/// evaluating 16-bit provenance bitmasks (`provenance_bits <= 0xFFFF`) in 2 parallel table lookups.
pub(crate) const CRC32_CRYSTAL_PROV_BYTE1_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l4(CRC32_IEEE_SLICE16_LUT.0[14][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `provenance_bits` byte-1 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_PROV_BYTE1_REF: &[u32; 256] = &CRC32_CRYSTAL_PROV_BYTE1_LUT.0;

/// Pre-shifted 8-entry GF(2) differential CRC32 table for `dominant_slot & 7` (`offset 56`),
/// shifted by 64 bytes via $L^4$.
pub(crate) const CRC32_CRYSTAL_SLOT_BYTE56_LUT_8: [u32; 8] = {
    let mut lut = [0u32; 8];
    let mut s = 0usize;
    while s < 8 {
        lut[s] = crc32_step_l4(CRC32_IEEE_SLICE16_LUT.0[7][s]);
        s += 1;
    }
    lut
};

/// 64B-aligned 1KB GF(2) differential CRC32 table for `action_mode_flags` (`offset 57`, shifted by 64 bytes via $L^4$).
pub(crate) const CRC32_CRYSTAL_ACTION_BYTE57_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l4(CRC32_IEEE_SLICE16_LUT.0[6][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `action_mode_flags` (`offset 57`) differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ACTION_BYTE57_REF: &[u32; 256] = &CRC32_CRYSTAL_ACTION_BYTE57_LUT.0;

/// 64B-aligned 4KB GF(2) differential CRC32 table for `base_energy` (`offset 24..28`),
/// pre-shifted to the end of the 128-byte `CrystallizedAttractorRecord` via $L^6(\text{indep\_lo1}(x))$.
pub(crate) const CRC32_CRYSTAL_BE24_FINAL_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let c1 = CRC32_IEEE_SLICE16_LUT.0[7 - byte_pos][b];
            lut[byte_pos][b] = crc32_step_l6(c1);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB `base_energy` (`offset 24..28`) final 128B CRC32 table.
pub(crate) const CRC32_CRYSTAL_BE24_FINAL_REF: &[[u32; 256]; 4] = &CRC32_CRYSTAL_BE24_FINAL_LUT.0;

/// 64B-aligned 4KB GF(2) differential CRC32 table for `phase_gravity_mass` (`offset 36..40`),
/// pre-shifted to the end of the 128-byte `CrystallizedAttractorRecord` via $L^5(\text{indep\_hi0}(x))$.
pub(crate) const CRC32_CRYSTAL_GM36_FINAL_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let c2 = CRC32_IEEE_SLICE16_LUT.0[11 - byte_pos][b];
            lut[byte_pos][b] = crc32_step_l5(c2);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB `phase_gravity_mass` (`offset 36..40`) final 128B CRC32 table.
pub(crate) const CRC32_CRYSTAL_GM36_FINAL_REF: &[[u32; 256]; 4] = &CRC32_CRYSTAL_GM36_FINAL_LUT.0;

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 24..28`
/// (`base_energy`) in a single 4-lookup parallel stage (`0` serial $L^6$ dependency).
#[inline(always)]
pub(crate) const fn crc32_crystal_be24_final_u32(be_bits: u32) -> u32 {
    let lut = CRC32_CRYSTAL_BE24_FINAL_REF;
    (lut[0][(be_bits & 0xFF) as usize] ^ lut[1][((be_bits >> 8) & 0xFF) as usize])
        ^ (lut[2][((be_bits >> 16) & 0xFF) as usize] ^ lut[3][(be_bits >> 24) as usize])
}

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 36..40`
/// (`phase_gravity_mass`) in a single 4-lookup parallel stage (`0` serial $L^5$ dependency).
#[inline(always)]
pub(crate) const fn crc32_crystal_gm36_final_u32(gm_bits: u32) -> u32 {
    let lut = CRC32_CRYSTAL_GM36_FINAL_REF;
    (lut[0][(gm_bits & 0xFF) as usize] ^ lut[1][((gm_bits >> 8) & 0xFF) as usize])
        ^ (lut[2][((gm_bits >> 16) & 0xFF) as usize] ^ lut[3][(gm_bits >> 24) as usize])
}

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 16..20`
/// (`zimms_section_offset48 as u32`) using the L1D-resident `CRC32_IEEE_SLICE16_REF` and `CRC32_IEEE_L6_REF` tables.
#[inline(always)]
pub(crate) const fn crc32_crystal_zoff16_final_u32(zoff_lo: u32) -> u32 {
    if zoff_lo == 0 {
        0
    } else {
        crc32_step_l6(crc32_slice16_dep_lo0(zoff_lo))
    }
}

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 40..44`
/// (`ripa_clamp_rad`) using the L1D-resident `CRC32_IEEE_SLICE16_REF` and `CRC32_IEEE_L5_REF` tables.
#[inline(always)]
pub(crate) const fn crc32_crystal_cl40_final_u32(cl_bits: u32) -> u32 {
    if cl_bits == 0 {
        0
    } else {
        crc32_step_l5(crc32_slice16_indep_lo1_u32(cl_bits))
    }
}

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 44..48`
/// (`effective_inv_temp`) using the L1D-resident `CRC32_IEEE_SLICE16_REF` and `CRC32_IEEE_L5_REF` tables.
#[inline(always)]
pub(crate) const fn crc32_crystal_tm44_final_u32(tm_bits: u32) -> u32 {
    if tm_bits == 0 {
        0
    } else {
        crc32_step_l5(crc32_slice16_indep_hi1_u32(tm_bits))
    }
}

#[inline(always)]
pub(crate) const fn crc32_slice16_indep_lo1_u32(lo1: u32) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    (lut[7][(lo1 & 0xFF) as usize] ^ lut[6][((lo1 >> 8) & 0xFF) as usize])
        ^ (lut[5][((lo1 >> 16) & 0xFF) as usize] ^ lut[4][(lo1 >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_slice16_indep_hi0_u32(hi0: u32) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    (lut[11][(hi0 & 0xFF) as usize] ^ lut[10][((hi0 >> 8) & 0xFF) as usize])
        ^ (lut[9][((hi0 >> 16) & 0xFF) as usize] ^ lut[8][(hi0 >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_slice16_indep_hi1_u32(hi1: u32) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    (lut[3][(hi1 & 0xFF) as usize] ^ lut[2][((hi1 >> 8) & 0xFF) as usize])
        ^ (lut[1][((hi1 >> 16) & 0xFF) as usize] ^ lut[0][(hi1 >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_slice16_indep_u64x2(w0_le: u64, w1_le: u64) -> u32 {
    let hi0 = (w0_le >> 32) as u32;
    let c_hi0 = if hi0 == 0 {
        0
    } else {
        crc32_slice16_indep_hi0_u32(hi0)
    };
    if w1_le == 0 {
        return c_hi0;
    }
    let lo1 = w1_le as u32;
    let hi1 = (w1_le >> 32) as u32;
    c_hi0
        ^ (if lo1 == 0 {
            0
        } else {
            crc32_slice16_indep_lo1_u32(lo1)
        })
        ^ (if hi1 == 0 {
            0
        } else {
            crc32_slice16_indep_hi1_u32(hi1)
        })
}

/// Evaluates the accumulator-independent 128-byte stamped crystal contribution
/// $L^4(L_{12}(\text{record\_crc32})) \oplus (\neg\text{record\_crc32}) \oplus L^8(\text{0xFFFF\_FFFF})$
/// in 4 parallel table lookups (`0` branches).
#[inline(always)]
pub(crate) const fn crc32_crystal_stamp_contrib_128b(record_crc32: u32) -> u32 {
    let lut = CRC32_IEEE_CRYSTAL_STAMP_REF;
    (lut[0][(record_crc32 & 0xFF) as usize] ^ lut[1][((record_crc32 >> 8) & 0xFF) as usize])
        ^ (lut[2][((record_crc32 >> 16) & 0xFF) as usize]
            ^ lut[3][(record_crc32 >> 24) as usize])
}

/// Advances an uninverted IEEE 802.3 CRC32 accumulator (`crc`) across an entire 128-byte stamped
/// `CrystallizedAttractorRecord` using GF(2) linearity ($L^8(\text{crc}) \oplus \text{stamp}(\text{record\_crc32})$)
/// in a single 4-lookup stage (`0` branches).
#[inline(always)]
pub(crate) const fn crc32_fold_stamped_crystal_128b(crc: u32, record_crc32: u32) -> u32 {
    crc32_step_l8(crc) ^ crc32_crystal_stamp_contrib_128b(record_crc32)
}

/// Updates a stamped `CrystallizedAttractorRecord`'s IEEE 802.3 `checksum_crc32` in $\mathcal{O}(1)$
/// via GF(2) linearity when `phase_profile` (`offset 64..128`), `anchor_id`, `zimms_section_offset48`,
/// `resultant_r_bar`, and `von_mises_kappa` remained stationary while `sample_count` (`offset 12..16`),
/// `base_energy` (`offset 24..28`), `phase_gravity_mass` (`offset 36..40`), `ripa_clamp_rad`
/// (`offset 40..44`), `effective_inv_temp` (`offset 44..48`), `provenance_bits` (`offset 48..52`),
/// or `w7_lo` (`dominant_slot` / `action_mode_flags` at `offset 56..58`) changed.
#[inline(always)]
pub(crate) const fn crc32_update_crystal_stationary_fields_delta(
    old_crc32: u32,
    delta_sample_count: u32,
    delta_base_energy_bits: u32,
    delta_gravity_mass_bits: u32,
    delta_clamp_bits: u32,
    delta_inv_temp_bits: u32,
    delta_provenance_bits: u32,
    delta_w7_lo: u32,
) -> u32 {
    let sc_fast = (delta_sample_count >> 8) == 0;
    if sc_fast
        & (((delta_base_energy_bits | delta_gravity_mass_bits)
            | (delta_clamp_bits | delta_inv_temp_bits)
            | (delta_provenance_bits | delta_w7_lo))
            == 0)
    {
        return old_crc32 ^ CRC32_CRYSTAL_SC_BYTE0_REF[delta_sample_count as usize];
    }
    let prov_fast = (delta_provenance_bits >> 16) == 0;
    let mut direct_xor = if sc_fast {
        CRC32_CRYSTAL_SC_BYTE0_REF[delta_sample_count as usize]
    } else {
        0
    };
    if delta_base_energy_bits != 0 {
        direct_xor ^= crc32_crystal_be24_final_u32(delta_base_energy_bits);
    }
    if delta_gravity_mass_bits != 0 {
        direct_xor ^= crc32_crystal_gm36_final_u32(delta_gravity_mass_bits);
    }
    if delta_clamp_bits != 0 {
        direct_xor ^= crc32_crystal_cl40_final_u32(delta_clamp_bits);
    }
    if delta_inv_temp_bits != 0 {
        direct_xor ^= crc32_crystal_tm44_final_u32(delta_inv_temp_bits);
    }
    if delta_w7_lo != 0 {
        direct_xor ^= CRC32_CRYSTAL_SLOT_BYTE56_LUT_8[(delta_w7_lo & 7) as usize]
            ^ CRC32_CRYSTAL_ACTION_BYTE57_REF[((delta_w7_lo >> 8) & 0xFF) as usize];
    }
    if prov_fast {
        direct_xor ^= CRC32_CRYSTAL_PROV_BYTE0_REF[(delta_provenance_bits & 0xFF) as usize]
            ^ CRC32_CRYSTAL_PROV_BYTE1_REF[((delta_provenance_bits >> 8) & 0xFF) as usize];
        if sc_fast {
            return old_crc32 ^ direct_xor;
        }
    }
    let sc_l2 = if sc_fast {
        0
    } else {
        crc32_step_l2(crc32_slice16_indep_hi1_u32(delta_sample_count))
    };
    let prov_rem = if prov_fast { 0 } else { delta_provenance_bits };
    let l5_in = sc_l2 ^ prov_rem;
    let l5_out = if l5_in == 0 { 0 } else { crc32_step_l5(l5_in) };
    old_crc32 ^ direct_xor ^ l5_out
}

/// Updates a stamped `CrystallizedAttractorRecord`'s IEEE 802.3 `checksum_crc32` in $\mathcal{O}(1)$
/// via GF(2) linearity when only `base_energy` (`offset 24..28`) and `phase_gravity_mass`
/// (`offset 36..40`) changed (such as during multiplicative evaporative decay in
/// `merge_and_prune_centroids_64`), using 8 parallel table lookups with `0` branches.
#[inline(always)]
pub(crate) const fn crc32_update_crystal_energy_and_mass_delta(
    old_crc32: u32,
    delta_base_energy_bits: u32,
    delta_gravity_mass_bits: u32,
) -> u32 {
    old_crc32
        ^ crc32_crystal_be24_final_u32(delta_base_energy_bits)
        ^ crc32_crystal_gm36_final_u32(delta_gravity_mass_bits)
}

/// Precomputed Slice-by-16 accumulator contribution for the invariant 8-byte header
/// `w0 = (GCSO_CRYSTAL_RECORD_MAGIC as u64) | ((GCSO_ABI_VERSION_HEX as u64) << 32)`
/// XORed with the initial IEEE 802.3 seed `0xFFFF_FFFF`, eliminating 8 table lookups on every
/// `CrystallizedAttractorRecord` checksum evaluation.
pub(crate) const CRYSTAL_CRC32_W0_SEED_CONTRIB: u32 =
    crc32_slice16_dep_lo0(0x4C54_5343u32 ^ 0xFFFF_FFFFu32)
        ^ crc32_slice16_indep_hi0_u32(crate::abi::GCSO_ABI_VERSION_HEX);

/// Precomputed Slice-by-16 accumulator contribution for the default `w5` (`ripa_clamp_rad = 0.24`,
/// `effective_inv_temp = 1.0`) field pair inside a `CrystallizedAttractorRecord`, eliminating 8 table lookups.
pub(crate) const CRYSTAL_CRC32_DEFAULT_W5_CONTRIB: u32 =
    crc32_slice16_indep_lo1_u32(DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
        ^ crc32_slice16_indep_hi1_u32(0x3F80_0000u32);

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^7(\text{CRYSTAL\_CRC32\_W0\_SEED\_CONTRIB})$
/// for the invariant `(GCSO_CRYSTAL_RECORD_MAGIC, GCSO_CRYSTAL_RECORD_VERSION)` header and initial seed.
pub(crate) const CRYSTAL_CRC32_W0_FINAL_CONTRIB: u32 =
    crc32_step_l4(crc32_step_l3(CRYSTAL_CRC32_W0_SEED_CONTRIB));

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^5(\text{CRYSTAL\_CRC32\_DEFAULT\_W5\_CONTRIB})$
/// for the default `w5` (`ripa_clamp_rad = 0.24`, `effective_inv_temp = 1.0`) field pair.
pub(crate) const CRYSTAL_CRC32_DEFAULT_W5_FINAL_CONTRIB: u32 =
    crc32_step_l5(CRYSTAL_CRC32_DEFAULT_W5_CONTRIB);

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^6(\text{indep\_hi1}(\text{0x3F80\_0000}))$
/// for `resultant_r_bar == 1.0` (`offset 28..32`), eliminating 4 table lookups on coherent centroids.
pub(crate) const CRYSTAL_CRC32_RBAR_ONE_FINAL_CONTRIB: u32 =
    crc32_step_l6(crc32_slice16_indep_hi1_u32(0x3F80_0000u32));

/// Exact IEEE-754 `f32::to_bits()` representation of `VON_MISES_KAPPA_AT_MAX_R` ($\bar{R}_c \ge 0.995$).
pub(crate) const VON_MISES_KAPPA_AT_MAX_R_BITS: u32 = VON_MISES_KAPPA_AT_MAX_R.to_bits();

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^5(\text{dep\_lo0}(\text{VON\_MISES\_KAPPA\_AT\_MAX\_R\_BITS}))$
/// for `von_mises_kappa == VON_MISES_KAPPA_AT_MAX_R` (`offset 32..36`), eliminating 4 table lookups
/// on every byte-uniform or maximally coherent centroid ($\bar{R}_c \ge 0.995$).
pub(crate) const CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB: u32 =
    crc32_step_l5(crc32_slice16_dep_lo0(VON_MISES_KAPPA_AT_MAX_R_BITS));

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^5(\text{dep\_lo0}(\text{0x4248\_0000}))$ for
/// `von_mises_kappa == 50.0` (`offset 32..36`), eliminating 4 table lookups.
pub(crate) const CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB: u32 =
    crc32_step_l5(crc32_slice16_dep_lo0(0x4248_0000u32));

/// Precomputed 64-byte GF(2) shift $L^4(\text{0xFFFF\_FFFF})$ of the initial IEEE 802.3 CRC32 seed.
pub(crate) const CRC32_L4_INITIAL_SEED: u32 = crc32_step_l4(0xFFFF_FFFFu32);

/// Precomputed 64-byte IEEE 802.3 CRC32 accumulator state over the invariant fields of a default
/// 4-section `.gcso` `gcso_zimms_descriptor_t` (`mapped_address = 0`, `file_size_bytes = 0`,
/// `dma_buffer_handle = 0`, `flags = 4`, `fd_handle = -1`, `reserved = [0; 32]`).
pub(crate) const CRC32_ZIMMS_DESC_BASE_SEED: u32 =
    CRC32_L4_INITIAL_SEED ^ crc32_step_l2(crc32_slice16_indep_u64x2(0, 0xFFFF_FFFF_0000_0004u64));

/// Evaluates the 64-byte header (`offset 0..64`) GF(2) differential IEEE 802.3 CRC32 state
/// `(final_128b_xor, c3_at_offset_64)` using strictly the L1D-resident `CRC32_IEEE_SLICE16_REF`
/// table and scalar constants (`0` separate 4KB `FINAL_LUT` table accesses), allowing `c3` to fold
/// directly into `phase_profile` word 0 (`dw[0]`) with zero `crc32_step_l4` overhead.
#[inline(always)]
pub(crate) const fn crc32_crystal_header_delta_c3(
    delta_anchor_id: u32,
    delta_sample_count: u32,
    delta_zimms_offset48: u64,
    delta_base_energy_bits: u32,
    delta_r_bar_bits: u32,
    delta_kappa_bits: u32,
    delta_gravity_mass_bits: u32,
    delta_clamp_bits: u32,
    delta_inv_temp_bits: u32,
    delta_provenance_bits: u32,
    delta_w7_lo: u32,
) -> (u32, u32) {
    let mut final_xor = 0u32;

    if delta_base_energy_bits != 0 {
        final_xor ^= crc32_crystal_be24_final_u32(delta_base_energy_bits);
    }
    if delta_gravity_mass_bits != 0 {
        final_xor ^= crc32_crystal_gm36_final_u32(delta_gravity_mass_bits);
    }

    let c0 = if ((delta_anchor_id >> 16) == 0) & ((delta_sample_count >> 8) == 0) {
        final_xor ^= CRC32_CRYSTAL_AID_BYTE0_REF[(delta_anchor_id & 0xFF) as usize]
            ^ CRC32_CRYSTAL_AID_BYTE1_REF[((delta_anchor_id >> 8) & 0xFF) as usize]
            ^ CRC32_CRYSTAL_SC_BYTE0_REF[delta_sample_count as usize];
        0
    } else {
        let aid_c = if (delta_anchor_id >> 16) == 0 {
            final_xor ^= CRC32_CRYSTAL_AID_BYTE0_REF[(delta_anchor_id & 0xFF) as usize]
                ^ CRC32_CRYSTAL_AID_BYTE1_REF[((delta_anchor_id >> 8) & 0xFF) as usize];
            0
        } else {
            crc32_slice16_indep_lo1_u32(delta_anchor_id)
        };
        let sc_c = if (delta_sample_count >> 8) == 0 {
            final_xor ^= CRC32_CRYSTAL_SC_BYTE0_REF[delta_sample_count as usize];
            0
        } else {
            crc32_slice16_indep_hi1_u32(delta_sample_count)
        };
        aid_c ^ sc_c
    };

    let z_lo = (delta_zimms_offset48 as u32) ^ c0;
    let z_hi = (delta_zimms_offset48 >> 32) as u32;
    let z_c = if (z_lo | z_hi) == 0 {
        0
    } else if (c0 == 0) & (z_hi == 0) & ((z_lo & 0xFF) == 0) {
        final_xor ^= CRC32_CRYSTAL_ZOFF_BYTE1_REF[((z_lo >> 8) & 0xFF) as usize]
            ^ CRC32_CRYSTAL_ZOFF_BYTE2_REF[((z_lo >> 16) & 0xFF) as usize]
            ^ CRC32_CRYSTAL_ZOFF_BYTE3_REF[(z_lo >> 24) as usize];
        0
    } else if z_hi == 0 {
        crc32_slice16_dep_lo0(z_lo)
    } else {
        crc32_slice16_dep_lo0(z_lo) ^ crc32_slice16_indep_hi0_u32(z_hi)
    };
    let rb_c = if delta_r_bar_bits == 0 {
        0
    } else if delta_r_bar_bits == 0x3F80_0000u32 {
        final_xor ^= CRYSTAL_CRC32_RBAR_ONE_FINAL_CONTRIB;
        0
    } else {
        crc32_slice16_indep_hi1_u32(delta_r_bar_bits)
    };
    let c1 = z_c ^ rb_c;
    if c1 != 0 {
        final_xor ^= crc32_step_l6(c1);
    }

    let kp_c = if delta_kappa_bits == 0 {
        0
    } else if delta_kappa_bits == VON_MISES_KAPPA_AT_MAX_R_BITS {
        final_xor ^= CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB;
        0
    } else if delta_kappa_bits == 0x4248_0000u32 {
        final_xor ^= CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB;
        0
    } else {
        crc32_slice16_dep_lo0(delta_kappa_bits)
    };
    let cl_c = if delta_clamp_bits == 0 {
        0
    } else {
        crc32_slice16_indep_lo1_u32(delta_clamp_bits)
    };
    let tm_c = if delta_inv_temp_bits == 0 {
        0
    } else {
        crc32_slice16_indep_hi1_u32(delta_inv_temp_bits)
    };
    let c2 = kp_c ^ cl_c ^ tm_c;
    if c2 != 0 {
        final_xor ^= crc32_step_l5(c2);
    }

    let prov_c = if (delta_provenance_bits >> 16) == 0 {
        final_xor ^= CRC32_CRYSTAL_PROV_BYTE0_REF[(delta_provenance_bits & 0xFF) as usize]
            ^ CRC32_CRYSTAL_PROV_BYTE1_REF[((delta_provenance_bits >> 8) & 0xFF) as usize];
        0
    } else {
        crc32_slice16_dep_lo0(delta_provenance_bits)
    };
    let w7_c = if ((delta_w7_lo >> 16) == 0) & ((delta_w7_lo & 0xF8) == 0) {
        final_xor ^= CRC32_CRYSTAL_SLOT_BYTE56_LUT_8[(delta_w7_lo & 7) as usize]
            ^ CRC32_CRYSTAL_ACTION_BYTE57_REF[((delta_w7_lo >> 8) & 0xFF) as usize];
        0
    } else {
        crc32_slice16_indep_lo1_u32(delta_w7_lo)
    };
    (final_xor, prov_c ^ w7_c)
}

/// Updates a stamped `CrystallizedAttractorRecord`'s IEEE 802.3 `checksum_crc32` via GF(2)
/// linearity across any absorption step, folding `c3` directly into `dw[0]` via
/// `crc32_step_block64_words(c3, &dw)` when `phase_profile` changed (`0` `crc32_step_l4` lookups).
#[inline(always)]
pub(crate) const fn crc32_update_crystal_absorption_delta(
    old_crc32: u32,
    delta_anchor_id: u32,
    delta_sample_count: u32,
    delta_zimms_offset48: u64,
    delta_base_energy_bits: u32,
    delta_r_bar_bits: u32,
    delta_kappa_bits: u32,
    delta_gravity_mass_bits: u32,
    delta_clamp_bits: u32,
    delta_inv_temp_bits: u32,
    delta_provenance_bits: u32,
    delta_w7_lo: u32,
    cur_w: &[u64; 8],
    next_w: &[u64; 8],
) -> u32 {
    let (final_xor, c3) = crc32_crystal_header_delta_c3(
        delta_anchor_id,
        delta_sample_count,
        delta_zimms_offset48,
        delta_base_energy_bits,
        delta_r_bar_bits,
        delta_kappa_bits,
        delta_gravity_mass_bits,
        delta_clamp_bits,
        delta_inv_temp_bits,
        delta_provenance_bits,
        delta_w7_lo,
    );

    if cur_w[0] == next_w[0]
        && ((((cur_w[1] ^ next_w[1]) | (cur_w[2] ^ next_w[2]))
            | ((cur_w[3] ^ next_w[3]) | (cur_w[4] ^ next_w[4])))
            | (((cur_w[5] ^ next_w[5]) | (cur_w[6] ^ next_w[6])) | (cur_w[7] ^ next_w[7])))
            == 0
    {
        return old_crc32 ^ final_xor ^ (if c3 == 0 { 0 } else { crc32_step_l4(c3) });
    }
    let dw = [
        cur_w[0] ^ next_w[0],
        cur_w[1] ^ next_w[1],
        cur_w[2] ^ next_w[2],
        cur_w[3] ^ next_w[3],
        cur_w[4] ^ next_w[4],
        cur_w[5] ^ next_w[5],
        cur_w[6] ^ next_w[6],
        cur_w[7] ^ next_w[7],
    ];
    old_crc32 ^ final_xor ^ crc32_step_block64_words(c3, &dw)
}

#[inline(always)]
pub(crate) const fn crc32_slice16_step_u64x2(crc: u32, w0_le: u64, w1_le: u64) -> u32 {
    let lo0 = (w0_le as u32) ^ crc;
    let base = if lo0 == 0 {
        0
    } else {
        crc32_slice16_dep_lo0(lo0)
    };
    if ((w0_le >> 32) | w1_le) == 0 {
        return base;
    }
    base ^ crc32_slice16_indep_u64x2(w0_le, w1_le)
}

#[inline(always)]
pub(crate) const fn crc32_block16_zero_seed_u64x2(w0_le: u64, w1_le: u64) -> u32 {
    if (w0_le | w1_le) == 0 {
        return 0;
    }
    let lo0 = w0_le as u32;
    let base = if lo0 == 0 {
        0
    } else {
        crc32_slice16_dep_lo0(lo0)
    };
    if ((w0_le >> 32) | w1_le) == 0 {
        return base;
    }
    base ^ crc32_slice16_indep_u64x2(w0_le, w1_le)
}

#[inline(always)]
const fn crc32_uniform_word_zero_seed_raw(w0: u64) -> u32 {
    let p0 = crc32_block16_zero_seed_u64x2(w0, w0);
    let h1 = crc32_slice16_dep_lo0(p0) ^ p0;
    crc32_step_l2(h1) ^ h1
}

/// 64B-aligned 1KB GF(2)-linear lookup table mapping any uniform byte `b in 0..=255` to the exact
/// unseeded 64-byte IEEE 802.3 CRC32 operator $\mathcal{C}_0([b; 64])$ (`1` table lookup instead of `36`).
pub(crate) const CRC32_UNIFORM64_ZERO_SEED_LUT_256: AlignedLut256<u32> = {
    let mut basis = [0u32; 8];
    let mut k = 0usize;
    while k < 8 {
        let w_bit = (1u64 << k).wrapping_mul(SWAR_SPLAT_BYTE_64);
        basis[k] = crc32_uniform_word_zero_seed_raw(w_bit);
        k += 1;
    }
    let mut lut = [0u32; 256];
    let mut b = 1usize;
    while b < 256 {
        let lsb = b.trailing_zeros() as usize;
        lut[b] = lut[b & (b - 1)] ^ basis[lsb];
        b += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 256-entry uniform 64-byte zero-seed CRC32 LUT.
pub(crate) const CRC32_UNIFORM64_ZERO_SEED_REF: &[u32; 256] = &CRC32_UNIFORM64_ZERO_SEED_LUT_256.0;

#[inline(always)]
const fn crc32_uniform56_pad8_zero_seed_raw(w0: u64) -> u32 {
    let u64_crc = crc32_uniform_word_zero_seed_raw(w0);
    let lo7 = w0 as u32;
    let hi7 = (w0 >> 32) as u32;
    u64_crc
        ^ (if lo7 == 0 { 0 } else { crc32_slice16_indep_lo1_u32(lo7) })
        ^ (if hi7 == 0 { 0 } else { crc32_slice16_indep_hi1_u32(hi7) })
}

/// 64B-aligned 1KB GF(2)-linear lookup table mapping any uniform byte `b in 0..=255` to the exact
/// unseeded 64-byte IEEE 802.3 CRC32 operator over 56 uniform bytes followed by 8 zero bytes
/// (`[b; 56] ++ [0; 8]`, matching `gcso_pointer_trail_t` Block 1 `offset 64..128` in `1` lookup instead of `9`).
pub(crate) const CRC32_UNIFORM56_PAD8_ZERO_SEED_LUT_256: AlignedLut256<u32> = {
    let mut basis = [0u32; 8];
    let mut k = 0usize;
    while k < 8 {
        let w_bit = (1u64 << k).wrapping_mul(SWAR_SPLAT_BYTE_64);
        basis[k] = crc32_uniform56_pad8_zero_seed_raw(w_bit);
        k += 1;
    }
    let mut lut = [0u32; 256];
    let mut b = 1usize;
    while b < 256 {
        let lsb = b.trailing_zeros() as usize;
        lut[b] = lut[b & (b - 1)] ^ basis[lsb];
        b += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 256-entry 56-byte-uniform + 8-zero-byte CRC32 LUT.
pub(crate) const CRC32_UNIFORM56_PAD8_ZERO_SEED_REF: &[u32; 256] =
    &CRC32_UNIFORM56_PAD8_ZERO_SEED_LUT_256.0;

/// Evaluates the raw IEEE 802.3 CRC32 operator over a 64-byte (`[u64; 8]`) block starting from
/// `seed == 0` via 4-way independent 16B Slice-by-16 evaluation (`p0, p1, p2, p3` in parallel)
/// followed by GF(2) operator folding (`crc32_step_l3(p0) ^ crc32_step_l2(p1) ^ crc32_slice16_dep_lo0(p2) ^ p3`),
/// reducing serial L1D load dependency depth to 2 stages while fast-pathing byte/word-uniform blocks
/// and `gcso_pointer_trail_t` Block 1 (`56B` uniform phase delta + `8B` zero reserved padding).
/// Evaluates the zero-seed IEEE 802.3 CRC32 of a 64-byte block `[w0, w0, w0, w0, w0, w0, w0, w7]`
/// whose first 7 64-bit words are already known to equal `w0` (`0` duplicate uniformity checks).
#[inline(always)]
#[must_use]
const fn crc32_block64_uniform_prefix56_zero_seed(w0: u64, w7: u64) -> u32 {
    if w0 == 0 {
        if w7 == 0 {
            return 0;
        }
        let lo7 = w7 as u32;
        let hi7 = (w7 >> 32) as u32;
        return (if lo7 == 0 { 0 } else { crc32_slice16_indep_lo1_u32(lo7) })
            ^ (if hi7 == 0 { 0 } else { crc32_slice16_indep_hi1_u32(hi7) });
    }
    let b0 = (w0 & 0xFF) as usize;
    let is_byte_uni = w0 == w0.rotate_right(8);
    if w7 == w0 {
        return if is_byte_uni {
            CRC32_UNIFORM64_ZERO_SEED_REF[b0]
        } else {
            crc32_uniform_word_zero_seed_raw(w0)
        };
    }
    if w7 == 0 && is_byte_uni {
        return CRC32_UNIFORM56_PAD8_ZERO_SEED_REF[b0];
    }
    let u64_crc = if is_byte_uni {
        CRC32_UNIFORM64_ZERO_SEED_REF[b0]
    } else {
        crc32_uniform_word_zero_seed_raw(w0)
    };
    let dw7 = w0 ^ w7;
    let lo7 = dw7 as u32;
    let hi7 = (dw7 >> 32) as u32;
    u64_crc
        ^ (if lo7 == 0 { 0 } else { crc32_slice16_indep_lo1_u32(lo7) })
        ^ (if hi7 == 0 { 0 } else { crc32_slice16_indep_hi1_u32(hi7) })
}

/// Evaluates the raw IEEE 802.3 CRC32 operator over 8 native `u64` words (64 bytes)
/// starting from initial seed `0`, returning the unfinalized accumulator state.
#[inline(always)]
#[must_use]
pub const fn crc32_block64_zero_seed_words(w: &[u64; 8]) -> u32 {
    let w0 = w[0];
    if w0 == w[1]
        && (((w0 ^ w[2]) | (w0 ^ w[3]))
            | ((w0 ^ w[4]) | (w0 ^ w[5]))
            | (w0 ^ w[6]))
            == 0
    {
        return crc32_block64_uniform_prefix56_zero_seed(w0, w[7]);
    }
    let w1 = w[1];
    let p0 = crc32_block16_zero_seed_u64x2(w0, w1);
    if (w0 == w[2])
        & (w1 == w[3])
        & ((((w0 ^ w[4]) | (w1 ^ w[5])) | ((w0 ^ w[6]) | (w1 ^ w[7]))) == 0)
    {
        let p01 = crc32_slice16_dep_lo0(p0) ^ p0;
        return crc32_step_l2(p01) ^ p01;
    }
    let p1 = crc32_block16_zero_seed_u64x2(w[2], w[3]);
    let p2 = crc32_block16_zero_seed_u64x2(w[4], w[5]);
    let p3 = crc32_block16_zero_seed_u64x2(w[6], w[7]);
    let s0 = if p0 == 0 { 0 } else { crc32_step_l3(p0) };
    let s1 = if p1 == 0 { 0 } else { crc32_step_l2(p1) };
    let s2 = if p2 == 0 { 0 } else { crc32_slice16_dep_lo0(p2) };
    (s0 ^ s1) ^ (s2 ^ p3)
}

#[inline(always)]
pub(crate) const fn crc32_step_block64_words(crc: u32, w: &[u64; 8]) -> u32 {
    if crc == 0 {
        return crc32_block64_zero_seed_words(w);
    }
    let w0 = w[0];
    if w0 == w[1]
        && (((w0 ^ w[2]) | (w0 ^ w[3]))
            | ((w0 ^ w[4]) | (w0 ^ w[5]))
            | (w0 ^ w[6]))
            == 0
    {
        let shifted_seed = if crc == 0xFFFF_FFFFu32 {
            CRC32_L4_INITIAL_SEED
        } else {
            crc32_step_l4(crc)
        };
        return shifted_seed ^ crc32_block64_uniform_prefix56_zero_seed(w0, w[7]);
    }
    let p0 = crc32_slice16_step_u64x2(crc, w0, w[1]);
    let p1 = crc32_block16_zero_seed_u64x2(w[2], w[3]);
    let p2 = crc32_block16_zero_seed_u64x2(w[4], w[5]);
    let p3 = crc32_block16_zero_seed_u64x2(w[6], w[7]);
    let s0 = if p0 == 0 { 0 } else { crc32_step_l3(p0) };
    let s1 = if p1 == 0 { 0 } else { crc32_step_l2(p1) };
    let s2 = if p2 == 0 { 0 } else { crc32_slice16_dep_lo0(p2) };
    (s0 ^ s1) ^ (s2 ^ p3)
}

/// Computes the IEEE 802.3 CRC32 checksum over 16 little-endian 64-bit words (128 bytes) using
/// GF(2) parallel $2 \times 64\text{B}$ tree folding (`L^4(block0) ^ block1`), cutting the serial
/// dependency depth from 8 stages down to 2 and leveraging `CRC32_UNIFORM64_ZERO_SEED_REF`.
#[inline]
#[must_use]
pub const fn branchless_crc32_128b_words(w: &[u64; 16]) -> u32 {
    let w0 = [w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]];
    let w1 = [w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15]];
    let block0_crc = crc32_step_block64_words(0xFFFF_FFFFu32, &w0);
    let block1_crc = crc32_block64_zero_seed_words(&w1);
    !(crc32_step_l4(block0_crc) ^ block1_crc)
}

#[inline(always)]
const fn load_u64_le_bytes_at(bytes: &[u8], i: usize) -> u64 {
    u64::from_le_bytes([
        bytes[i],
        bytes[i + 1],
        bytes[i + 2],
        bytes[i + 3],
        bytes[i + 4],
        bytes[i + 5],
        bytes[i + 6],
        bytes[i + 7],
    ])
}

/// Incrementally updates an IEEE 802.3 CRC32 checksum (`crc_seed`) over `bytes` using the
/// 64B-aligned 16KB Slice-by-16 `CRC32_IEEE_SLICE16_LUT` table and GF(2) $2 \times 64\text{B}$ /
/// $2 \times 32\text{B}$ parallel tree reduction (`crc32_step_block64_words` + `crc32_step_l4`).
#[inline]
pub const fn branchless_update_crc32_ieee(crc_seed: u32, bytes: &[u8]) -> u32 {
    let mut crc = !crc_seed;
    let len = bytes.len();
    let mut i = 0usize;
    while i + 128 <= len {
        let w0 = [
            load_u64_le_bytes_at(bytes, i),
            load_u64_le_bytes_at(bytes, i + 8),
            load_u64_le_bytes_at(bytes, i + 16),
            load_u64_le_bytes_at(bytes, i + 24),
            load_u64_le_bytes_at(bytes, i + 32),
            load_u64_le_bytes_at(bytes, i + 40),
            load_u64_le_bytes_at(bytes, i + 48),
            load_u64_le_bytes_at(bytes, i + 56),
        ];
        let w1 = [
            load_u64_le_bytes_at(bytes, i + 64),
            load_u64_le_bytes_at(bytes, i + 72),
            load_u64_le_bytes_at(bytes, i + 80),
            load_u64_le_bytes_at(bytes, i + 88),
            load_u64_le_bytes_at(bytes, i + 96),
            load_u64_le_bytes_at(bytes, i + 104),
            load_u64_le_bytes_at(bytes, i + 112),
            load_u64_le_bytes_at(bytes, i + 120),
        ];
        let c0 = crc32_step_block64_words(crc, &w0);
        let c1 = crc32_block64_zero_seed_words(&w1);
        crc = crc32_step_l4(c0) ^ c1;
        i += 128;
    }
    while i + 64 <= len {
        let w = [
            load_u64_le_bytes_at(bytes, i),
            load_u64_le_bytes_at(bytes, i + 8),
            load_u64_le_bytes_at(bytes, i + 16),
            load_u64_le_bytes_at(bytes, i + 24),
            load_u64_le_bytes_at(bytes, i + 32),
            load_u64_le_bytes_at(bytes, i + 40),
            load_u64_le_bytes_at(bytes, i + 48),
            load_u64_le_bytes_at(bytes, i + 56),
        ];
        crc = crc32_step_block64_words(crc, &w);
        i += 64;
    }
    while i + 16 <= len {
        let w0 = load_u64_le_bytes_at(bytes, i);
        let w1 = load_u64_le_bytes_at(bytes, i + 8);
        crc = crc32_slice16_step_u64x2(crc, w0, w1);
        i += 16;
    }
    if i + 8 <= len {
        let w = load_u64_le_bytes_at(bytes, i);
        crc = crc32_slice8_step_u64(crc, w);
        i += 8;
    }
    if i + 4 <= len {
        crc = crc32_slice4_step(crc, bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]);
        i += 4;
    }
    while i < len {
        crc = (crc >> 8) ^ CRC32_IEEE_SLICE16_REF[0][((crc as u8) ^ bytes[i]) as usize];
        i += 1;
    }
    !crc
}

/// Computes the standard IEEE 802.3 CRC32 checksum over `bytes` using 8KB Slice-by-8 table lookups.
#[inline(always)]
pub const fn branchless_crc32_ieee(bytes: &[u8]) -> u32 {
    branchless_update_crc32_ieee(0, bytes)
}

/// Fast positive normal reciprocal $1.0 / x$ for $x \ge \text{F32\_MIN\_POSITIVE\_NORMAL}$
/// using IEEE-754 magic bit seed and 3 Newton-Raphson iterations (`0` subnormal/validity branches).
#[inline(always)]
pub(crate) const fn fast_inv_pos_normal_f32(x: f32) -> f32 {
    let y0 = f32::from_bits(0x7EF1_27EA_u32.wrapping_sub(x.to_bits()));
    let y1 = y0 * (2.0 - x * y0);
    let e1 = 1.0 - x * y1;
    y1 + y1 * (e1 + e1 * e1)
}

const LN_1P_LUT_STEP: f32 = 0.0625;
const LN_1P_LUT_INV_STEP: f32 = 16.0;

#[inline(always)]
const fn const_ln_1p_exact_f32(rho_nonneg: f32) -> f32 {
    if rho_nonneg <= 0.0 {
        return 0.0;
    }
    let val = 1.0 + rho_nonneg;
    let bits = val.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    let u = (mantissa - 1.0) / (mantissa + 1.0);
    let u2 = u * u;
    let poly = u
        * (2.0
            + u2 * (0.666_666_7
                + u2 * (0.4 + u2 * (0.285_714_3 + u2 * (0.222_222_22 + u2 * 0.181_818_19)))));
    (exponent as f32) * LN_2 + poly
}

/// 64-byte aligned 257-entry `(ln(1 + x0), 1 / (1 + x0))` lookup table over $[0.0, 16.0]$
/// with exact grid spacing $h = 1/16 = 0.0625$.
pub(crate) const LN_1P_LUT_257: AlignedLut257<(f32, f32)> = {
    let mut lut = [(0.0_f32, 1.0_f32); 257];
    let mut k = 1usize;
    while k <= 256 {
        let x0 = (k as f32) * LN_1P_LUT_STEP;
        lut[k] = (const_ln_1p_exact_f32(x0), 1.0 / (1.0 + x0));
        k += 1;
    }
    AlignedLut257(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 257-entry `(ln(1 + x0), 1 / (1 + x0))` table.
pub(crate) const LN_1P_LUT_257_REF: &[(f32, f32); 257] = &LN_1P_LUT_257.0;

/// Fast branchless natural logarithm $\ln(1 + \rho)$ for bounded non-negative density $\rho \in [0, 16]$
/// using `LN_1P_LUT_257` and 2nd-order Taylor interpolation (`0` branches, `0` NR reciprocals).
#[inline(always)]
pub(crate) const fn fast_ln_1p_bounded_f32(rho_clamped: f32) -> f32 {
    let clamped = branchless_clamp_f32(rho_clamped, 0.0, 16.0);
    let idx_raw = (clamped * LN_1P_LUT_INV_STEP) as usize;
    let idx = branchless_select_usize(idx_raw <= 256, idx_raw, 256);
    let d = clamped - (idx as f32) * LN_1P_LUT_STEP;
    let (y0, inv_1px0) = LN_1P_LUT_257_REF[idx];
    let u = d * inv_1px0;
    y0 + u * (1.0 - 0.5 * u)
}

/// Precomputed maturity gravity scale $1.0 + 0.10 \ln(1 + \min(N, 16))$ for integer sample counts $N \in [0, 16]$.
pub(crate) const MATURITY_SCALE_LUT_17: [f32; 17] = {
    let mut lut = [1.0_f32; 17];
    let mut n = 1usize;
    while n <= 16 {
        lut[n] = 1.0 + 0.10 * LN_1P_LUT_257.0[n << 4].0;
        n += 1;
    }
    lut
};

/// Evaluates the centroid maturity gravity multiplier $1.0 + 0.10 \ln(1 + \min(N, 16))$ in 1 cycle.
#[inline(always)]
pub(crate) const fn branchless_maturity_scale_u32(sample_count: u32) -> f32 {
    let idx = branchless_select_usize((sample_count as usize) <= 16, sample_count as usize, 16);
    MATURITY_SCALE_LUT_17[idx]
}

#[inline(always)]
pub(crate) const fn fast_inv_len_f32(n: usize) -> f32 {
    if n > 0 && (n & (n - 1)) == 0 && n <= (1usize << 62) {
        f32::from_bits((127u32.wrapping_sub(n.trailing_zeros())) << 23)
    } else if n > 0 && n <= 256 {
        let lut = POPCOUNT_RSQRT_LUT_257_REF;
        let r = lut[n];
        r * r
    } else if n > 0 {
        fast_inv_pos_normal_f32(n as f32)
    } else {
        0.0
    }
}

/// Branchless bitwise selection for 8-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_u8(cond: bool, if_true: u8, if_false: u8) -> u8 {
    let m = (cond as u8).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 8-bit signed integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_i8(cond: bool, if_true: i8, if_false: i8) -> i8 {
    let m = (cond as i8).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 16-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_u16(cond: bool, if_true: u16, if_false: u16) -> u16 {
    let m = (cond as u16).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 32-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_u32(cond: bool, if_true: u32, if_false: u32) -> u32 {
    let m = (cond as u32).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 64-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_u64(cond: bool, if_true: u64, if_false: u64) -> u64 {
    let m = (cond as u64).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for pointer-sized unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_usize(cond: bool, if_true: usize, if_false: usize) -> usize {
    let m = (cond as usize).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 32-bit signed integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_i32(cond: bool, if_true: i32, if_false: i32) -> i32 {
    let m = (cond as i32).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 64-bit signed integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_i64(cond: bool, if_true: i64, if_false: i64) -> i64 {
    let m = (cond as i64).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for Q7 fixed-point integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_q7(
    cond: bool,
    if_true: gcso_q7_t,
    if_false: gcso_q7_t,
) -> gcso_q7_t {
    branchless_select_i8(cond, if_true, if_false)
}

/// Branchless selection for IEEE-754 single-precision floats (`cond ? if_true : if_false`),
/// remaining in the floating-point XMM register file (`vminss` / `vmaxss` / `vblendvps`)
/// without XMM-to-GPR (`vmovd`) cross-domain bypass stalls.
#[inline(always)]
pub const fn branchless_select_f32(cond: bool, if_true: f32, if_false: f32) -> f32 {
    if cond {
        if_true
    } else {
        if_false
    }
}

/// Branchless IEEE-754 finiteness check (`!is_nan() && !is_infinite()`) evaluated directly
/// in XMM registers (`vucomiss`) without XMM-to-GPR `vmovd` transfer stalls.
#[inline(always)]
pub const fn branchless_is_finite_f32(x: f32) -> bool {
    (x > -f32::INFINITY) & (x < f32::INFINITY)
}

/// Branchless IEEE-754 `NaN` check evaluated directly in XMM registers (`vucomiss`).
#[inline(always)]
pub const fn branchless_is_nan_f32(x: f32) -> bool {
    x.is_nan()
}

/// Branchlessly flushes IEEE-754 subnormal (denormal) floats, non-finite values (`±Inf`, `NaN`),
/// and `-0.0` to `+0.0`.
#[inline(always)]
pub const fn branchless_flush_subnormal_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp_bits = bits & 0x7F80_0000;
    let is_normal_finite = (exp_bits != 0) & (exp_bits != 0x7F80_0000);
    f32::from_bits(bits & (is_normal_finite as u32).wrapping_neg())
}

/// Branchless IEEE-754 absolute value (`|x|`).
#[inline(always)]
pub const fn branchless_abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

/// Branchless IEEE-754 `copysign(mag, sgn)`.
#[inline(always)]
pub const fn branchless_copysign_f32(mag: f32, sgn: f32) -> f32 {
    f32::from_bits((mag.to_bits() & 0x7FFF_FFFF) | (sgn.to_bits() & 0x8000_0000))
}

/// Branchless signum returning `-1.0`, `0.0`, or `+1.0`.
#[inline(always)]
pub const fn branchless_signum_f32(x: f32) -> f32 {
    let is_nonzero_finite = ((x.to_bits() & 0x7FFF_FFFF) != 0) & (!branchless_is_nan_f32(x));
    let signed_one = branchless_copysign_f32(1.0, x);
    branchless_select_f32(is_nonzero_finite, signed_one, 0.0)
}

/// Branchless Rectified Linear Unit ($\max(0.0, x)$), flushing `NaN`, negatives, and `-0.0` to `+0.0`
/// in a single XMM `vmaxss` comparison (`0` XMM-to-GPR round-trips).
#[inline(always)]
pub const fn branchless_relu_f32(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Branchless `NaN`-resilient IEEE-754 maximum ($\max(a, b)$).
#[inline(always)]
pub const fn branchless_max_f32(a: f32, b: f32) -> f32 {
    if (a > b) | branchless_is_nan_f32(b) {
        a
    } else {
        b
    }
}

/// Branchless `NaN`-resilient IEEE-754 minimum ($\min(a, b)$).
#[inline(always)]
pub const fn branchless_min_f32(a: f32, b: f32) -> f32 {
    if (a < b) | branchless_is_nan_f32(b) {
        a
    } else {
        b
    }
}

/// Branchless interval clamp ($\mathrm{clamp}(x, \mathrm{lo}, \mathrm{hi})$), lowering to a pair
/// of hardware XMM `vmaxss` + `vminss` instructions when `lo` and `hi` are finite bounds.
#[inline(always)]
pub const fn branchless_clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
    let mx = if (x > lo) | branchless_is_nan_f32(lo) {
        x
    } else {
        lo
    };
    if (mx < hi) | branchless_is_nan_f32(hi) {
        mx
    } else {
        hi
    }
}

/// Sanitizes a single-token Z-score input into $[0.0, \text{max\_ceiling}]$.
#[inline(always)]
pub const fn branchless_sanitize_z_score_f32(z_score: f32, max_ceiling: f32) -> f32 {
    let safe_ceil = branchless_select_f32(
        (max_ceiling > 0.0) & (max_ceiling < f32::INFINITY),
        max_ceiling,
        DEFAULT_MAX_VALID_Z_SCORE,
    );
    let clean = branchless_select_f32((z_score > 0.0) & (z_score < f32::INFINITY), z_score, 0.0);
    branchless_min_f32(clean, safe_ceil)
}

/// Branchless linear interpolation: $a + t(b - a)$ for $t \in [0, 1]$.
#[inline(always)]
pub const fn branchless_lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    let tc = branchless_clamp_f32(t, 0.0, 1.0);
    a + tc * (b - a)
}

/// Branchless cubic Hermite smoothstep interpolation ($3u^2 - 2u^3$) for $x \in [\text{edge0}, \text{edge1}]$.
#[inline(always)]
pub const fn branchless_smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    let valid_span = (span > 1.0e-12) & branchless_is_finite_f32(span);
    let inv_span = branchless_inv_pos_f32(branchless_select_f32(valid_span, span, 1.0));
    let u = branchless_select_f32(
        valid_span,
        branchless_clamp_f32((x - edge0) * inv_span, 0.0, 1.0),
        branchless_select_f32(x >= edge1, 1.0, 0.0),
    );
    u * u * (3.0 - 2.0 * u)
}

/// Branchless $SO(2)$ 2D rotation operator applying $(q_0 \cos\theta - q_1 \sin\theta, q_0 \sin\theta + q_1 \cos\theta)$.
#[inline(always)]
pub const fn branchless_rotate_so2_pair(q0: f32, q1: f32, sin_t: f32, cos_t: f32) -> (f32, f32) {
    (q0 * cos_t - q1 * sin_t, q0 * sin_t + q1 * cos_t)
}

/// Branchless composite DPSR + Isotropic Block-Diagonal Scaling operator (Theorem 1.1).
#[inline(always)]
pub const fn branchless_rotate_so2_pair_scaled(
    q0: f32,
    q1: f32,
    sin_t: f32,
    cos_t: f32,
    scale: f32,
) -> (f32, f32) {
    (
        (q0 * cos_t - q1 * sin_t) * scale,
        (q0 * sin_t + q1 * cos_t) * scale,
    )
}

#[inline(always)]
const fn unpack_pos_f32_exp_and_mant(raw_bits: u32) -> (i32, u32) {
    if raw_bits < 0x0080_0000 {
        let lz = (raw_bits | 1).leading_zeros();
        let shift = lz.wrapping_sub(8);
        (-118i32 - (lz as i32), (raw_bits << shift) & 0x007F_FFFF)
    } else {
        ((((raw_bits >> 23) & 0xFF) as i32) - 127, raw_bits & 0x007F_FFFF)
    }
}

/// Division-free (`0 fdiv`) branchless `#![no_std]` IEEE-754 reciprocal ($1 / x$)
/// for positive finite floats (`val > 0.0`), executing a direct 3-step Newton-Raphson magic-seed
/// reciprocal on normal floats (`1.0e-36 <= val <= 1.0e36`, `0` exponent/mantissa unpacking) and
/// falling back to two-scale exponent reconstruction for subnormal or extreme floats.
#[inline(always)]
pub const fn branchless_inv_pos_f32(val: f32) -> f32 {
    let raw_bits = val.to_bits();
    if (raw_bits >= 0x0416_872B) & (raw_bits <= 0x7B60_435E) {
        return fast_inv_pos_normal_f32(val);
    }
    if (raw_bits == 0) | (raw_bits >= 0x7F80_0000) {
        return 0.0;
    }
    let (e_unbiased, mant23) = unpack_pos_f32_exp_and_mant(raw_bits);
    let m_bits = mant23 | 0x3F80_0000;
    let m = f32::from_bits(m_bits);

    let mut r = f32::from_bits(0x7EF3_11C2_u32.wrapping_sub(m_bits));
    r = r * (2.0 - m * r);
    r = r * (2.0 - m * r);
    r = r * (2.0 - m * r);

    let e_half1 = e_unbiased >> 1;
    let e_half2 = e_unbiased - e_half1;
    let scale1 = f32::from_bits(((127 - e_half1) as u32) << 23);
    let scale2 = f32::from_bits(((127 - e_half2) as u32) << 23);
    branchless_flush_subnormal_f32((r * scale1) * scale2)
}

/// Branchless `#![no_std]` hyperbolic tangent ($\tanh(x)$) using Padé $(5, 4)$ rational approximation.
#[inline(always)]
pub const fn branchless_tanh_f32(x: f32) -> f32 {
    let safe_x = branchless_select_f32(branchless_is_nan_f32(x), 0.0, x);
    let clamped = branchless_clamp_f32(safe_x, -TANH_LUT_MAX_X, TANH_LUT_MAX_X);
    let x2 = clamped * clamped;
    let x4 = x2 * x2;
    (clamped * (945.0 + 105.0 * x2 + x4))
        * fast_inv_pos_normal_f32(945.0 + 420.0 * x2 + 15.0 * x4)
}

/// Branchless `#![no_std]` logistic sigmoid ($\sigma(x) = 1 / (1 + e^{-x})$).
#[inline(always)]
pub const fn branchless_sigmoid_f32(x: f32) -> f32 {
    0.5 + 0.5 * branchless_tanh_f32(0.5 * x)
}

/// Branchless RIPA soft-bounded phase angle clamp: $\Delta\theta_{\mathrm{safe}} = \theta_{\max} \cdot \tanh(\Delta\theta / \theta_{\max})$.
#[inline(always)]
pub const fn branchless_ripa_clamp_rad(delta_rad: f32, max_rad: f32) -> f32 {
    let valid_max = (max_rad > 1.0e-9) & branchless_is_finite_f32(max_rad);
    let safe_max = branchless_select_f32(valid_max, max_rad, DEFAULT_RIPA_CLAMP_MAX_RAD);
    let inv_max = branchless_inv_pos_f32(safe_max);
    branchless_select_f32(
        valid_max,
        safe_max * branchless_tanh_f32(delta_rad * inv_max),
        0.0,
    )
}

/// Branchless `#![no_std]` natural logarithm ($\ln(x)$).
#[inline(always)]
pub const fn branchless_ln_f32(val: f32) -> f32 {
    let bits_raw = val.to_bits();
    if bits_raw == 0x3F80_0000 {
        return 0.0;
    }
    if (bits_raw >= 0x0080_0000) & (bits_raw < 0x7F80_0000) {
        let exponent = ((bits_raw >> 23) & 0xFF) as i32 - 127;
        let mantissa = f32::from_bits((bits_raw & 0x007F_FFFF) | 0x3F80_0000);
        let u = (mantissa - 1.0) * fast_inv_pos_normal_f32(mantissa + 1.0);
        let u2 = u * u;
        let poly =
            u * (2.0 + u2 * (0.666_666_7 + u2 * (0.4 + u2 * (0.285_714_3 + u2 * 0.222_222_2))));
        return (exponent as f32) * LN_2 + poly;
    }
    if (bits_raw == 0) | (bits_raw >= 0x7F80_0000) {
        return 0.0;
    }
    let (exponent, mant23) = unpack_pos_f32_exp_and_mant(bits_raw);
    let mantissa = f32::from_bits(mant23 | 0x3F80_0000);
    let u = (mantissa - 1.0) * fast_inv_pos_normal_f32(mantissa + 1.0);
    let u2 = u * u;
    let poly = u * (2.0 + u2 * (0.666_666_7 + u2 * (0.4 + u2 * (0.285_714_3 + u2 * 0.222_222_2))));
    (exponent as f32) * LN_2 + poly
}

/// Branchless `#![no_std]` fast exponential ($e^x$) using Cody-Waite base-2 range reduction and Padé $(3, 3)$.
#[inline(always)]
pub const fn branchless_exp_f32(x: f32) -> f32 {
    let invalid_or_under = branchless_is_nan_f32(x) | (x < -87.3);
    let clamped =
        branchless_clamp_f32(branchless_select_f32(invalid_or_under, 0.0, x), -87.0, 88.0);
    let k_float = clamped * INV_LN2_F32;
    let k = (k_float + branchless_select_f32(k_float >= 0.0, 0.5, -0.5)) as i32;
    let kf = k as f32;
    let r = (clamped - kf * LN2_HI_F32) - kf * LN2_LO_F32;
    let r2 = r * r;
    let even_part = 120.0 + 12.0 * r2;
    let odd_part = r * (60.0 + r2);
    let exp_r = (even_part + odd_part) * fast_inv_pos_normal_f32(even_part - odd_part);
    let scale_2k = f32::from_bits(((k + 127) as u32) << 23);
    branchless_select_f32(invalid_or_under, 0.0, exp_r * scale_2k)
}

/// Fast division-free (`0 fdiv`) square root $\sqrt{x} \in [0.0, 1.0]$ for bounded unit-interval
/// squared resultant lengths $x = \bar{S}^2 + \bar{C}^2 \in [0.0, 1.01]$ (`0` exponent unpacking).
#[inline(always)]
pub(crate) const fn fast_sqrt_unit_f32(r_sq: f32) -> f32 {
    if r_sq <= 1.0e-12 {
        return 0.0;
    }
    let half_x = 0.5 * r_sq;
    let r0 = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(r_sq.to_bits() >> 1));
    let r = r0 * (1.5 - half_x * r0 * r0);
    let s0 = r_sq * r;
    let y = s0 + (0.5 * r) * (r_sq - s0 * s0);
    branchless_select_f32(y < 1.0, y, 1.0)
}

/// Division-free (`0 fdiv`) branchless `#![no_std]` IEEE-754 square root ($\sqrt{x}$),
/// executing a direct magic-seed 2-step Newton-Raphson + Goldschmidt refinement on normal floats
/// (`F32_MIN_POSITIVE_NORMAL <= val <= 1.0e34`, `0` exponent/mantissa unpacking) and falling back
/// to exponent-normalized reconstruction for subnormal or extreme floats.
#[inline(always)]
pub const fn branchless_sqrt_f32(val: f32) -> f32 {
    let raw_bits = val.to_bits();
    if (raw_bits >= 0x0080_0000) & (raw_bits <= 0x781A_530C) {
        let half_x = 0.5 * val;
        let mut r = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(raw_bits >> 1));
        r = r * (1.5 - half_x * r * r);
        r = r * (1.5 - half_x * r * r);
        let s0 = val * r;
        return s0 + (0.5 * r) * (val - s0 * s0);
    }
    if (raw_bits == 0) | (raw_bits >= 0x7F80_0000) {
        return 0.0;
    }
    let (e_unbiased, mant23) = unpack_pos_f32_exp_and_mant(raw_bits);
    let odd_exp = (e_unbiased & 1) as u32;
    let m_bits = mant23 | ((127 + odd_exp) << 23);
    let m = f32::from_bits(m_bits);
    let half_m = 0.5 * m;

    let mut r = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(m_bits >> 1));
    r = r * (1.5 - half_m * r * r);
    r = r * (1.5 - half_m * r * r);
    r = r * (1.5 - half_m * r * r);

    let s0 = m * r;
    let y = s0 + (0.5 * r) * (m - s0 * s0);
    let half_exp_shift = ((e_unbiased >> 1) as u32) << 23;
    f32::from_bits(y.to_bits().wrapping_add(half_exp_shift))
}

/// Division-free (`0 fdiv`) branchless `#![no_std]` IEEE-754 reciprocal square root ($1 / \sqrt{x}$),
/// executing a direct magic-seed 3-step Newton-Raphson refinement on normal floats
/// (`F32_MIN_POSITIVE_NORMAL <= val <= 1.0e34`, `0` exponent/mantissa unpacking) and falling back
/// to exponent-normalized reconstruction for subnormal or extreme floats.
#[inline(always)]
pub const fn branchless_rsqrt_f32(val: f32) -> f32 {
    let raw_bits = val.to_bits();
    if (raw_bits >= 0x0080_0000) & (raw_bits <= 0x781A_530C) {
        let half_x = 0.5 * val;
        let mut y = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(raw_bits >> 1));
        y = y * (1.5 - half_x * y * y);
        y = y * (1.5 - half_x * y * y);
        return y * (1.5 - half_x * y * y);
    }
    if (raw_bits == 0) | (raw_bits >= 0x7F80_0000) {
        return 0.0;
    }
    let (e_unbiased, mant23) = unpack_pos_f32_exp_and_mant(raw_bits);
    let odd_exp = (e_unbiased & 1) as u32;
    let m_bits = mant23 | ((127 + odd_exp) << 23);
    let m = f32::from_bits(m_bits);
    let half_m = 0.5 * m;

    let mut y = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(m_bits >> 1));
    y = y * (1.5 - half_m * y * y);
    y = y * (1.5 - half_m * y * y);
    y = y * (1.5 - half_m * y * y);

    let neg_half_exp = ((-(e_unbiased >> 1)) as u32) << 23;
    f32::from_bits(y.to_bits().wrapping_add(neg_half_exp))
}

#[inline(always)]
const fn sincos_poly_half_pi(x_folded: f32, sign_cos: f32) -> (f32, f32) {
    let x2 = x_folded * x_folded;
    let sin_poly = x_folded
        * (1.0
            + x2 * (-0.166_666_67
                + x2 * (0.008_333_334
                    + x2 * (-0.000_198_412_7
                        + x2 * (0.000_002_755_732 + x2 * -0.000_000_025_052_108)))));
    let cos_poly = 1.0
        + x2 * (-0.5
            + x2 * (0.041_666_668
                + x2 * (-0.001_388_888_9 + x2 * (0.000_024_801_587 + x2 * -0.000_000_275_573_2))));
    (sin_poly, sign_cos * cos_poly)
}

#[inline(always)]
const fn sincos_reduced_pi(x: f32) -> (f32, f32) {
    let gt_half_pi = x > FRAC_PI_2;
    let lt_neg_half_pi = x < -FRAC_PI_2;
    let x_folded = branchless_select_f32(
        gt_half_pi,
        PI - x,
        branchless_select_f32(lt_neg_half_pi, -PI - x, x),
    );
    let sign_cos = branchless_select_f32(gt_half_pi | lt_neg_half_pi, -1.0, 1.0);
    sincos_poly_half_pi(x_folded, sign_cos)
}

/// Branchless `#![no_std]` fused $(\sin\theta, \cos\theta)$ evaluator using 53-bit `f64` $2\pi$ range reduction.
#[inline(always)]
pub const fn branchless_sincos_f32(val: f32) -> (f32, f32) {
    let valid = branchless_is_finite_f32(val);
    let safe_val = branchless_select_f32(valid, val, 0.0) as f64;
    let ratio = safe_val * INV_TWO_PI_F64;
    let k = ((ratio + if ratio >= 0.0 { 0.5 } else { -0.5 }) as i64) as f64;
    let x = (safe_val - k * TWO_PI_F64) as f32;
    let (sin_v, cos_v) = sincos_reduced_pi(x);
    (
        branchless_select_f32(valid, sin_v, 0.0),
        branchless_select_f32(valid, cos_v, 1.0),
    )
}

/// Branchless direct $(\sin\theta, \cos\theta)$ evaluator for angles bounded within $[-\pi/2, \pi/2]$.
#[inline(always)]
pub const fn branchless_sincos_half_pi_f32(theta_rad: f32) -> (f32, f32) {
    let valid = branchless_is_finite_f32(theta_rad);
    let safe_theta = branchless_clamp_f32(
        branchless_select_f32(valid, theta_rad, 0.0),
        -FRAC_PI_2,
        FRAC_PI_2,
    );
    let (sin_v, cos_v) = sincos_poly_half_pi(safe_theta, 1.0);
    (
        branchless_select_f32(valid, sin_v, 0.0),
        branchless_select_f32(valid, cos_v, 1.0),
    )
}

/// Direct $\mathcal{O}(1)$ 64B-aligned 256-entry LUT lookup for RIPA $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$
/// under [`DEFAULT_RIPA_CLAMP_MAX_RAD`].
#[inline(always)]
pub const fn branchless_sincos_q7_ripa_default_lut(phase_q7: gcso_q7_t) -> (f32, f32) {
    let lut = Q7_DEFAULT_RIPA_SINCOS_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Direct $\mathcal{O}(1)$ 64B-aligned 256-entry LUT lookup combining $S^1$ geodesic folding
/// with RIPA soft-clamping under [`DEFAULT_RIPA_CLAMP_MAX_RAD`].
#[inline(always)]
pub const fn branchless_sincos_q7_geodesic_ripa_default_lut(phase_q7: gcso_q7_t) -> (f32, f32) {
    let lut = Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Evaluates the signed RIPA-clamped radian angle $\Delta\theta_{\mathrm{safe}} = \mathrm{sgn}(q_7)\cdot |\theta_{\max}| \cdot \tanh(|q_7|\cdot s)$
/// using the 257-entry `TANH_LUT_257` table.
#[inline(always)]
pub(crate) const fn eval_ripa_clamped_theta_rad_preinv(
    phase_q7: gcso_q7_t,
    max_rad: f32,
    q7_to_norm_scale: f32,
) -> f32 {
    let u_abs = branchless_q7_abs_u8(phase_q7);
    let norm_u = (u_abs as f32) * branchless_abs_f32(q7_to_norm_scale);
    let safe_theta_mag = branchless_abs_f32(max_rad) * tanh_lut_eval_nonneg(norm_u);
    let is_neg = ((phase_q7 < 0) ^ (max_rad < 0.0)) & (u_abs != 0);
    f32::from_bits(safe_theta_mag.to_bits() | ((is_neg as u32) << 31))
}

/// Division-free (`0 fdiv`) branchless RIPA soft-clamped $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$
/// using precomputed `q7_to_norm_scale = Q7_PHASE_SCALE_RAD / max_rad`.
#[inline(always)]
pub const fn branchless_sincos_q7_ripa_preinv(
    phase_q7: gcso_q7_t,
    max_rad: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if (max_rad == DEFAULT_RIPA_CLAMP_MAX_RAD)
        & (q7_to_norm_scale == DEFAULT_RIPA_Q7_TO_NORM_SCALE)
    {
        return branchless_sincos_q7_ripa_default_lut(phase_q7);
    }
    let u_abs = branchless_q7_abs_u8(phase_q7);
    let norm_u = (u_abs as f32) * branchless_abs_f32(q7_to_norm_scale);
    let safe_theta_mag = branchless_abs_f32(max_rad) * tanh_lut_eval_nonneg(norm_u);
    let (sin_mag, cos_val) = sincos_ripa_poly(safe_theta_mag);
    let sin_neg = ((phase_q7 < 0) ^ (max_rad < 0.0)) & (u_abs != 0);
    (
        f32::from_bits(sin_mag.to_bits() | ((sin_neg as u32) << 31)),
        cos_val,
    )
}

/// Standalone division-free branchless RIPA soft-clamped $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ from Q7.
#[inline(always)]
pub const fn branchless_sincos_q7_ripa(phase_q7: gcso_q7_t, max_rad: f32) -> (f32, f32) {
    if max_rad == DEFAULT_RIPA_CLAMP_MAX_RAD {
        return branchless_sincos_q7_ripa_default_lut(phase_q7);
    }
    let valid_max = (max_rad > 1.0e-9) & branchless_is_finite_f32(max_rad);
    let eff_max = branchless_min_f32(
        branchless_select_f32(valid_max, max_rad, DEFAULT_RIPA_CLAMP_MAX_RAD),
        FRAC_PI_2,
    );
    let q7_to_norm_scale = Q7_PHASE_SCALE_RAD * branchless_inv_pos_f32(eff_max);
    let (sin_t, cos_t) = branchless_sincos_q7_ripa_preinv(phase_q7, eff_max, q7_to_norm_scale);
    (
        branchless_select_f32(valid_max, sin_t, 0.0),
        branchless_select_f32(valid_max, cos_t, 1.0),
    )
}

/// Direct 64B-aligned 256-entry interleaved LUT $(\sin\theta, \cos\theta)$ from Q7.
#[inline(always)]
pub const fn branchless_sincos_q7(phase_q7: gcso_q7_t) -> (f32, f32) {
    let lut = Q7_SINCOS_LUT_256_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Direct 256-entry LUT $\sin\theta$ from a Q7 phase angle (`q7 * pi / 128`).
#[inline(always)]
pub const fn branchless_sin_q7(phase_q7: gcso_q7_t) -> f32 {
    let lut = Q7_FULL_SIN_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Direct 256-entry LUT $\cos\theta$ from a Q7 phase angle (`q7 * pi / 128`).
#[inline(always)]
pub const fn branchless_cos_q7(phase_q7: gcso_q7_t) -> f32 {
    let lut = Q7_FULL_COS_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Branchless `#![no_std]` four-quadrant inverse tangent ($\mathrm{atan2}(y, x) \in [-\pi, \pi]$)
/// using `fast_inv_pos_normal_f32` (since $\max(|x|, |y|) \ge 10^{-30} \gg \text{F32\_MIN\_POSITIVE\_NORMAL}$).
#[inline(always)]
pub const fn branchless_atan2_f32(y: f32, x: f32) -> f32 {
    let valid = branchless_is_finite_f32(x) & branchless_is_finite_f32(y);
    let safe_x = branchless_select_f32(valid, x, 1.0);
    let safe_y = branchless_select_f32(valid, y, 0.0);
    let ax = branchless_abs_f32(safe_x);
    let ay = branchless_abs_f32(safe_y);
    let both_zero = (ax < 1.0e-30) & (ay < 1.0e-30);
    let swap = ax < ay;
    let min_v = branchless_select_f32(swap, ax, ay);
    let max_v = branchless_max_f32(branchless_select_f32(swap, ay, ax), 1.0e-30);
    let inv_max = if max_v <= 1.0e30 {
        fast_inv_pos_normal_f32(max_v)
    } else {
        branchless_inv_pos_f32(max_v)
    };
    let r = branchless_clamp_f32(min_v * inv_max, 0.0, 1.0);
    let r2 = r * r;
    let poly = r
        * (0.999_977_26
            + r2 * (-0.332_623_47
                + r2 * (0.193_543_46
                    + r2 * (-0.116_432_87 + r2 * (0.052_653_32 - 0.011_721_2 * r2)))));
    let mut angle = branchless_select_f32(swap, FRAC_PI_2 - poly, poly);
    angle = branchless_select_f32(safe_x < 0.0, PI - angle, angle);
    angle = branchless_copysign_f32(angle, safe_y);
    branchless_select_f32(valid & !both_zero, angle, 0.0)
}

/// Fused `#![no_std]` four-quadrant $\mathrm{atan2}(y, x)$ directly quantized into `gcso_q7_t`
/// ($[-128, 127]$), folding the $\frac{128}{\pi}$ scale factor directly into the minimax polynomial
/// coefficients (`0` Q7 error across $[-180^\circ, 180^\circ]$).
#[inline(always)]
pub const fn branchless_atan2_to_q7_fast(y: f32, x: f32) -> gcso_q7_t {
    let valid = branchless_is_finite_f32(x) & branchless_is_finite_f32(y);
    let safe_x = branchless_select_f32(valid, x, 1.0);
    let safe_y = branchless_select_f32(valid, y, 0.0);
    let ax = branchless_abs_f32(safe_x);
    let ay = branchless_abs_f32(safe_y);
    let both_zero = (ax < 1.0e-30) & (ay < 1.0e-30);
    if (!valid) | both_zero {
        return 0;
    }
    let swap = ax < ay;
    let min_v = branchless_select_f32(swap, ax, ay);
    let max_v = branchless_max_f32(branchless_select_f32(swap, ay, ax), 1.0e-30);
    let inv_max = if max_v <= 1.0e30 {
        fast_inv_pos_normal_f32(max_v)
    } else {
        branchless_inv_pos_f32(max_v)
    };
    let r = branchless_clamp_f32(min_v * inv_max, 0.0, 1.0);
    let r2 = r * r;
    let poly_q7 = r
        * (40.742_737
            + r2 * (-13.553_533
                + r2 * (7.886_416
                    + r2 * (-4.744_295 + r2 * (2.145_485 - 0.477_609 * r2)))));
    let mut ang_q7 = branchless_select_f32(swap, 64.0 - poly_q7, poly_q7);
    ang_q7 = branchless_select_f32(safe_x < 0.0, 128.0 - ang_q7, ang_q7);
    ang_q7 = branchless_copysign_f32(ang_q7, safe_y);
    let rounded = (ang_q7 + branchless_select_f32(ang_q7 >= 0.0, 0.5, -0.5)) as i32;
    branchless_select_i32(
        rounded < -128,
        -128,
        branchless_select_i32(rounded > 127, 127, rounded),
    ) as gcso_q7_t
}

#[inline(always)]
const fn eval_von_mises_kappa_clamped(r: f32) -> f32 {
    let r2 = r * r;
    let r3 = r2 * r;
    let k1 = 2.0 * r + r3 + 0.833_333_3 * (r3 * r2);
    let k2 = fast_inv_pos_normal_f32(1.28 * branchless_max_f32(1.0 - r2, 1.0e-4));
    let k3 = fast_inv_pos_normal_f32(
        branchless_max_f32(r, 0.5) * branchless_max_f32(1.0 - r, 1.0e-4) * (3.0 - r),
    );
    let w12 = branchless_sigmoid_f32((r - 0.53) * 40.0);
    let w23 = branchless_sigmoid_f32((r - 0.85) * 40.0);
    let blend12 = k1 + w12 * (k2 - k1);
    blend12 + w23 * (k3 - blend12)
}

pub(crate) const VON_MISES_KAPPA_AT_MAX_R: f32 =
    eval_von_mises_kappa_clamped(VON_MISES_KAPPA_MAX_R);

/// 64B-aligned 2-zone (`[0.0, 0.875]` and `[0.875, 0.995]`, `2 x 257` `f32` entries = `2056` bytes)
/// precomputed von Mises concentration $\kappa_c(\bar{R})$ table eliminating 4 Newton-Raphson
/// inversions and 2 Padé `tanh` evaluations per call on the hot/mezzo path.
pub(crate) static VON_MISES_KAPPA_2ZONE_LUT: AlignedLut2x257<f32> = {
    let mut lut = [[0.0_f32; 257]; 2];
    let mut i = 0usize;
    while i < 257 {
        lut[0][i] = eval_von_mises_kappa_clamped((i as f32) * VON_MISES_KAPPA_LO_STEP);
        lut[1][i] = eval_von_mises_kappa_clamped(
            VON_MISES_KAPPA_SPLIT_R + (i as f32) * VON_MISES_KAPPA_HI_STEP,
        );
        i += 1;
    }
    AlignedLut2x257(lut)
};

const VON_MISES_KAPPA_2ZONE_REF: &[[f32; 257]; 2] = &VON_MISES_KAPPA_2ZONE_LUT.0;

/// Branchless sigmoidal-blended von Mises concentration parameter $\kappa_c$ from $\bar{R}_c \in [0, 1)$,
/// evaluated in strictly constant time (`0` Newton-Raphson divisions, `0` Padé `tanh` evaluations)
/// via `VON_MISES_KAPPA_2ZONE_LUT` with a 1-cycle ceiling return for $\bar{R}_c \ge 0.995$.
#[inline(always)]
pub const fn branchless_von_mises_kappa(r_bar: f32) -> f32 {
    if r_bar >= VON_MISES_KAPPA_MAX_R {
        return VON_MISES_KAPPA_AT_MAX_R;
    }
    let clean_r = branchless_relu_f32(r_bar);
    if clean_r <= 0.0 {
        return 0.0;
    }
    let is_hi = clean_r > VON_MISES_KAPPA_SPLIT_R;
    let zone = is_hi as usize;
    let pos = branchless_select_f32(
        is_hi,
        (clean_r - VON_MISES_KAPPA_SPLIT_R) * VON_MISES_KAPPA_HI_SCALE,
        clean_r * VON_MISES_KAPPA_LO_SCALE,
    );
    let idx_raw = pos as usize;
    let idx = branchless_select_usize(idx_raw < 256, idx_raw, 255);
    let frac = pos - (idx as f32);
    let lut = &VON_MISES_KAPPA_2ZONE_REF[zone];
    let v0 = lut[idx];
    let v1 = lut[idx + 1];
    v0 + frac * (v1 - v0)
}

/// Evaluates Phase-Proximity Gravity $F_{\mathrm{grav}} = \frac{M_{\mathrm{grav}} \cdot \max(0, 1 + \mathcal{C}_{\mathrm{phase}})}{d_{\mathrm{chord}}^2 + \epsilon_{\mathrm{grav}}}$
/// on $SO(2)^{64}$ where $d_{\mathrm{chord}}^2 = 2(1 - \mathcal{C}_{\mathrm{phase}})$ (`0` branches, `0 fdiv`).
#[inline(always)]
pub const fn branchless_phase_proximity_gravity_f32(
    gravity_mass: f32,
    phase_coherence: f32,
    softening_eps: f32,
) -> f32 {
    let clean_mass = branchless_flush_subnormal_f32(branchless_relu_f32(gravity_mass));
    let is_default_eps = softening_eps == DEFAULT_PHASE_GRAVITY_EPS;
    if (phase_coherence == 1.0) & is_default_eps {
        return branchless_flush_subnormal_f32(clean_mass * (2.0 / DEFAULT_PHASE_GRAVITY_EPS));
    }
    let coh = branchless_clamp_f32(
        branchless_select_f32(branchless_is_finite_f32(phase_coherence), phase_coherence, 0.0),
        -1.0,
        1.0,
    );
    let eps = if is_default_eps {
        DEFAULT_PHASE_GRAVITY_EPS
    } else {
        branchless_max_f32(
            branchless_select_f32(
                branchless_is_finite_f32(softening_eps) & (softening_eps > 1.0e-4),
                softening_eps,
                DEFAULT_PHASE_GRAVITY_EPS,
            ),
            1.0e-4,
        )
    };
    let chord_dist_sq = 2.0 * (1.0 - coh);
    let denom = chord_dist_sq + eps;
    let align_gain = branchless_max_f32(0.05, 1.0 + coh);
    branchless_flush_subnormal_f32((clean_mass * align_gain) * fast_inv_pos_normal_f32(denom))
}

/// Branchless scale-invariant Eyring-Kramers Langevin pitchfork tunneling escape probability (Lemma 3.1).
#[inline(always)]
pub const fn branchless_eyring_kramers_escape(
    cvoid: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    d0: f32,
) -> f32 {
    let valid = branchless_is_finite_f32(cvoid)
        & branchless_is_finite_f32(tau_eff)
        & branchless_is_finite_f32(alpha)
        & branchless_is_finite_f32(beta)
        & branchless_is_finite_f32(d0)
        & (beta > 1.0e-12)
        & (d0 > 1.0e-12);
    let safe_denom = branchless_select_f32(valid, 4.0 * beta * d0, 1.0);
    let excess = branchless_relu_f32(cvoid - tau_eff);
    let safe_alpha = branchless_select_f32(valid, alpha, 0.0);
    let ratio = (safe_alpha * safe_alpha * excess * excess) * fast_inv_pos_normal_f32(safe_denom);
    branchless_select_f32(valid, branchless_exp_f32(-ratio), 0.0)
}

/// Converts a Q7 fixed-point phase value to radians ($\theta = \mathrm{q7} \cdot \pi / 128$).
#[inline(always)]
pub const fn branchless_q7_to_rad(phase_q7: gcso_q7_t) -> f32 {
    (phase_q7 as f32) * Q7_PHASE_SCALE_RAD
}

/// Branchlessly quantizes a radian angle into Q7 fixed-point format ($[-128, 127]$).
#[inline(always)]
pub const fn branchless_rad_to_q7(rad: f32) -> gcso_q7_t {
    let valid = branchless_is_finite_f32(rad);
    let scaled = branchless_select_f32(valid, rad, 0.0) * Q7_PHASE_INV_SCALE_RAD;
    let rounded = (scaled + branchless_select_f32(scaled >= 0.0, 0.5, -0.5)) as i32;
    let clamped = branchless_select_i32(
        rounded < -128,
        -128,
        branchless_select_i32(rounded > 127, 127, rounded),
    );
    branchless_select_i32(valid, clamped, 0) as gcso_q7_t
}

/// Converts a non-negative radian step threshold into a ceiling-rounded Q7 integer cutoff (`0..=128`).
#[inline(always)]
pub const fn branchless_rad_to_qdps_step_u8(min_step_rad: f32) -> u8 {
    let bits = min_step_rad.to_bits();
    if (bits & 0x7FFF_FFFF) == 0 || (bits >> 31) != 0 || (bits & 0x7F80_0000) == 0x7F80_0000 {
        return 0;
    }
    if bits <= Q7_PHASE_SCALE_RAD.to_bits() {
        return 1;
    }
    let raw_q7 = min_step_rad * Q7_PHASE_INV_SCALE_RAD;
    let int_part = raw_q7 as i32;
    let step = int_part + (((int_part as f32) < raw_q7) as i32);
    branchless_select_i32(step > 128, 128, branchless_select_i32(step < 0, 0, step)) as u8
}

/// Branchless Q7 fixed-point saturating addition ($[-128, 127]$).
#[inline(always)]
pub const fn branchless_q7_add_sat(a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
    let sum = (a as i32) + (b as i32);
    let clamped_hi = branchless_select_i32(sum > (Q7_MAX as i32), Q7_MAX as i32, sum);
    branchless_select_i32(clamped_hi < (Q7_MIN as i32), Q7_MIN as i32, clamped_hi) as gcso_q7_t
}

/// Branchless Q7 fixed-point saturating subtraction ($[-128, 127]$).
#[inline(always)]
pub const fn branchless_q7_sub_sat(a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
    let diff = (a as i32) - (b as i32);
    let clamped_hi = branchless_select_i32(diff > (Q7_MAX as i32), Q7_MAX as i32, diff);
    branchless_select_i32(clamped_hi < (Q7_MIN as i32), Q7_MIN as i32, clamped_hi) as gcso_q7_t
}

/// Branchless Q7 circular phase difference $(a - b) \pmod{256}$ on $S^1 \cong SO(2)$.
#[inline(always)]
pub const fn branchless_q7_circular_diff(a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
    a.wrapping_sub(b)
}

/// Branchlessly folds a circular Q7 phase angle on $S^1$ onto the principal geodesic wave `[-64, +64]`.
#[inline(always)]
pub const fn branchless_q7_fold_s1_geodesic(phase_q7: gcso_q7_t) -> gcso_q7_t {
    let b = phase_q7 as u8;
    let fold_mask = ((b ^ (b << 1)) >> 7).wrapping_neg();
    let mirrored = 0x80u8.wrapping_sub(b);
    ((b & !fold_mask) | (mirrored & fold_mask)) as gcso_q7_t
}

/// Branchless Q7 shortest-arc geodesic interpolation on $S^1 \cong SO(2)$ (`t_u7 in 0..=128`).
#[inline(always)]
pub const fn branchless_q7_circular_lerp(a: gcso_q7_t, b: gcso_q7_t, t_u7: u8) -> gcso_q7_t {
    let clamped_t = branchless_select_u8(t_u7 > 128, 128, t_u7);
    a.wrapping_add(branchless_q7_scale_u7(
        branchless_q7_circular_diff(b, a),
        clamped_t,
    ))
}

/// Branchless Q7 phase-conjugate negation ($-\Delta\theta$) saturating $-128 \to +127$.
#[inline(always)]
pub const fn branchless_q7_neg_sat(x: gcso_q7_t) -> gcso_q7_t {
    let neg = -(x as i32);
    branchless_select_i32(neg > 127, 127, neg) as gcso_q7_t
}

/// Branchless Q7 absolute value returning `[0, 128]` as `u8`.
#[inline(always)]
pub const fn branchless_q7_abs_u8(x: gcso_q7_t) -> u8 {
    let val = x as i32;
    let mask = val >> 31;
    ((val ^ mask) - mask) as u8
}

/// Branchless Q7 fixed-point integer scaling: $\lfloor (\mathrm{val} \cdot \mathrm{scale\_u7}) / 128 \rfloor$.
#[inline(always)]
pub const fn branchless_q7_scale_u7(val: gcso_q7_t, scale_u7: u8) -> gcso_q7_t {
    let scaled = ((val as i32) * (scale_u7 as i32)) >> 7;
    let clamped_hi = branchless_select_i32(scaled > 127, 127, scaled);
    branchless_select_i32(clamped_hi < -128, -128, clamped_hi) as gcso_q7_t
}

/// Packs 8 `gcso_q7_t` lanes into a single 64-bit little-endian SWAR word.
#[inline(always)]
pub const fn swar_q7_pack_x8(lanes: &[gcso_q7_t; 8]) -> u64 {
    u64::from_le_bytes([
        lanes[0] as u8,
        lanes[1] as u8,
        lanes[2] as u8,
        lanes[3] as u8,
        lanes[4] as u8,
        lanes[5] as u8,
        lanes[6] as u8,
        lanes[7] as u8,
    ])
}

/// Unpacks a 64-bit little-endian SWAR word into 8 `gcso_q7_t` lanes.
#[inline(always)]
pub const fn swar_q7_unpack_x8(packed: u64) -> [gcso_q7_t; 8] {
    let b = packed.to_le_bytes();
    [
        b[0] as gcso_q7_t,
        b[1] as gcso_q7_t,
        b[2] as gcso_q7_t,
        b[3] as gcso_q7_t,
        b[4] as gcso_q7_t,
        b[5] as gcso_q7_t,
        b[6] as gcso_q7_t,
        b[7] as gcso_q7_t,
    ]
}

/// True bit-parallel SWAR 8-lane unsigned byte halving decay (`b_i >> 1`).
#[inline(always)]
pub const fn swar_u8_halve_decay_x8(packed_u8: u64) -> u64 {
    (packed_u8 >> 1) & SWAR_LOW7_MASK_64
}

/// True bit-parallel SWAR 8-lane unsigned byte saturating addition (`min(a_i + b_i, 255)`).
#[inline(always)]
pub const fn swar_u8_add_sat_x8(a: u64, b: u64) -> u64 {
    let low7_sum = (a & SWAR_LOW7_MASK_64).wrapping_add(b & SWAR_LOW7_MASK_64);
    let high_bits = (a ^ b) & SWAR_SIGN_MASK_64;
    let carry_from_low7 = low7_sum & SWAR_SIGN_MASK_64;
    let overflow_msb = ((a & b) | ((a | b) & carry_from_low7)) & SWAR_SIGN_MASK_64;
    let one_hi = high_bits ^ carry_from_low7;
    let sat_mask = overflow_msb | overflow_msb.wrapping_sub(overflow_msb >> 7);
    (low7_sum ^ one_hi) | sat_mask
}

/// True bit-parallel SWAR 8-lane unsigned byte saturating subtraction (`max(b_i - sub_step, 0)`).
#[inline(always)]
pub const fn swar_u8_sub_sat_x8(packed_u8: u64, sub_step: u8) -> u64 {
    let b = (sub_step as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let diff_low7 = (packed_u8 | SWAR_SIGN_MASK_64).wrapping_sub(b & SWAR_LOW7_MASK_64);
    let ge_bit7 = ((packed_u8 & !b) | (!(packed_u8 ^ b) & diff_low7)) & SWAR_SIGN_MASK_64;
    let keep_mask = (ge_bit7 >> 7).wrapping_mul(0xFF);
    let diff_unwrapped = diff_low7 ^ ((packed_u8 ^ !b) & SWAR_SIGN_MASK_64);
    diff_unwrapped & keep_mask
}

/// True bit-parallel SWAR 8-lane unsigned byte elementwise maximum (`max(a_i, b_i)`).
#[inline(always)]
pub const fn swar_u8_max_x8(a: u64, b: u64) -> u64 {
    let diff_low7 =
        ((a & SWAR_LOW7_MASK_64) | SWAR_SIGN_MASK_64).wrapping_sub(b & SWAR_LOW7_MASK_64);
    let a_ge_b_msb = ((a & !b) | ((!(a ^ b)) & diff_low7)) & SWAR_SIGN_MASK_64;
    let mask = a_ge_b_msb | a_ge_b_msb.wrapping_sub(a_ge_b_msb >> 7);
    (a & mask) | (b & !mask)
}

#[inline(always)]
pub(crate) const fn swar_u8_nonzero_msb_mask(w: u64) -> u64 {
    let has_low7 = (w & SWAR_LOW7_MASK_64).wrapping_add(SWAR_LOW7_MASK_64);
    (w | has_low7) & SWAR_SIGN_MASK_64
}

/// Computes the exact horizontal sum (`0..=2040`) of 8 packed unsigned bytes in `packed_u8`.
#[inline(always)]
pub const fn swar_u8_sum_x8(packed_u8: u64) -> u32 {
    let sum16 =
        (packed_u8 & SWAR_EVEN_BYTES_64).wrapping_add((packed_u8 >> 8) & SWAR_EVEN_BYTES_64);
    (sum16.wrapping_mul(SWAR_SUM_U16_64) >> 48) as u32
}

/// Computes the exact rounded mean $\lfloor (\sum_{i=0}^7 b_i + 4) / 8 \rfloor$ of 8 packed `u8` lanes.
#[inline(always)]
pub const fn swar_u8_mean_round_x8(packed_u8: u64) -> u8 {
    ((swar_u8_sum_x8(packed_u8) + 4) >> 3) as u8
}

/// Counts the number of non-zero byte lanes (`0..=8`) inside a 64-bit word `packed_u8`.
#[inline(always)]
pub const fn swar_u8_nonzero_count_x8(packed_u8: u64) -> u32 {
    swar_u8_nonzero_msb_mask(packed_u8).count_ones()
}

/// True bit-parallel SWAR 4-lane Q7 saturating addition packed in a 32-bit word.
#[inline(always)]
pub const fn swar_q7_add_sat_x4(a_packed: u32, b_packed: u32) -> u32 {
    let sum_low7 = (a_packed & SWAR_LOW7_MASK_32).wrapping_add(b_packed & SWAR_LOW7_MASK_32);
    let sum_unwrapped = sum_low7 ^ ((a_packed ^ b_packed) & SWAR_SIGN_MASK_32);
    let ovf_sign = !(a_packed ^ b_packed) & (a_packed ^ sum_unwrapped) & SWAR_SIGN_MASK_32;
    let ovf_mask = (ovf_sign >> 7).wrapping_mul(0xFF);
    let sat_val = SWAR_LOW7_MASK_32.wrapping_add((a_packed & SWAR_SIGN_MASK_32) >> 7);
    (sum_unwrapped & !ovf_mask) | (sat_val & ovf_mask)
}

/// True bit-parallel SWAR 4-lane Q7 saturating subtraction packed in a 32-bit word.
#[inline(always)]
pub const fn swar_q7_sub_sat_x4(a_packed: u32, b_packed: u32) -> u32 {
    let diff_low7 = (a_packed | SWAR_SIGN_MASK_32).wrapping_sub(b_packed & SWAR_LOW7_MASK_32);
    let diff_unwrapped = diff_low7 ^ ((a_packed ^ !b_packed) & SWAR_SIGN_MASK_32);
    let ovf_sign = (a_packed ^ b_packed) & (a_packed ^ diff_unwrapped) & SWAR_SIGN_MASK_32;
    let ovf_mask = (ovf_sign >> 7).wrapping_mul(0xFF);
    let sat_val = SWAR_LOW7_MASK_32.wrapping_add((a_packed & SWAR_SIGN_MASK_32) >> 7);
    (diff_unwrapped & !ovf_mask) | (sat_val & ovf_mask)
}

/// True bit-parallel SWAR 4-lane Q7 phase-conjugate negation ($-\boldsymbol{\Delta\theta}$) in a 32-bit word.
#[inline(always)]
pub const fn swar_q7_neg_sat_x4(packed: u32) -> u32 {
    swar_q7_sub_sat_x4(0, packed)
}

/// True bit-parallel SWAR 8-lane Q7 saturating addition packed in a 64-bit word.
#[inline(always)]
pub const fn swar_q7_add_sat_x8(a_packed: u64, b_packed: u64) -> u64 {
    let sum_low7 = (a_packed & SWAR_LOW7_MASK_64).wrapping_add(b_packed & SWAR_LOW7_MASK_64);
    let sum_unwrapped = sum_low7 ^ ((a_packed ^ b_packed) & SWAR_SIGN_MASK_64);
    let ovf_sign = !(a_packed ^ b_packed) & (a_packed ^ sum_unwrapped) & SWAR_SIGN_MASK_64;
    let ovf_mask = (ovf_sign >> 7).wrapping_mul(0xFF);
    let sat_val = SWAR_LOW7_MASK_64.wrapping_add((a_packed & SWAR_SIGN_MASK_64) >> 7);
    (sum_unwrapped & !ovf_mask) | (sat_val & ovf_mask)
}

/// True bit-parallel SWAR 8-lane Q7 saturating subtraction packed in a 64-bit word.
#[inline(always)]
pub const fn swar_q7_sub_sat_x8(a_packed: u64, b_packed: u64) -> u64 {
    let diff_low7 = (a_packed | SWAR_SIGN_MASK_64).wrapping_sub(b_packed & SWAR_LOW7_MASK_64);
    let diff_unwrapped = diff_low7 ^ ((a_packed ^ !b_packed) & SWAR_SIGN_MASK_64);
    let ovf_sign = (a_packed ^ b_packed) & (a_packed ^ diff_unwrapped) & SWAR_SIGN_MASK_64;
    let ovf_mask = (ovf_sign >> 7).wrapping_mul(0xFF);
    let sat_val = SWAR_LOW7_MASK_64.wrapping_add((a_packed & SWAR_SIGN_MASK_64) >> 7);
    (diff_unwrapped & !ovf_mask) | (sat_val & ovf_mask)
}

/// True bit-parallel SWAR 8-lane Q7 phase-conjugate negation ($-\boldsymbol{\Delta\theta}$) in a 64-bit word.
#[inline(always)]
pub const fn swar_q7_neg_sat_x8(packed: u64) -> u64 {
    swar_q7_sub_sat_x8(0, packed)
}

/// True bit-parallel SWAR 8-lane Q7 circular phase difference $(a_i - b_i) \pmod{256}$ on $S^1$.
#[inline(always)]
pub const fn swar_q7_circular_diff_x8(a_packed: u64, b_packed: u64) -> u64 {
    let diff_low7 = (a_packed | SWAR_SIGN_MASK_64).wrapping_sub(b_packed & SWAR_LOW7_MASK_64);
    diff_low7 ^ ((a_packed ^ !b_packed) & SWAR_SIGN_MASK_64)
}

/// True bit-parallel SWAR 8-lane circular phase addition $(a_i + b_i) \pmod{256}$ on $S^1$.
#[inline(always)]
pub const fn swar_q7_circular_add_x8(a_packed: u64, b_packed: u64) -> u64 {
    let sum_low7 = (a_packed & SWAR_LOW7_MASK_64).wrapping_add(b_packed & SWAR_LOW7_MASK_64);
    sum_low7 ^ ((a_packed ^ b_packed) & SWAR_SIGN_MASK_64)
}

/// True bit-parallel SWAR 8-lane Q7 absolute value (`|x_i| in [0, 128]`) packed in a 64-bit word.
#[inline(always)]
pub const fn swar_q7_abs_u8_x8(packed: u64) -> u64 {
    let sign_bit = (packed & SWAR_SIGN_MASK_64) >> 7;
    let sign_mask = sign_bit.wrapping_mul(0xFF);
    (packed ^ sign_mask).wrapping_add(sign_bit)
}

/// True bit-parallel SWAR 8-lane $S^1$ geodesic folding (`[-128, 127] -> [-64, 64]`).
#[inline(always)]
pub const fn swar_q7_fold_s1_geodesic_x8(packed: u64) -> u64 {
    let fold_bit7 = (packed ^ (packed << 1)) & SWAR_SIGN_MASK_64;
    let fold_mask = (fold_bit7 >> 7).wrapping_mul(0xFF);
    let mirrored =
        SWAR_SIGN_MASK_64.wrapping_sub(packed & SWAR_LOW7_MASK_64) ^ (packed & SWAR_SIGN_MASK_64);
    (packed & !fold_mask) | (mirrored & fold_mask)
}

/// True bit-parallel SWAR 8-lane Q7 rounded halving average $\lfloor(a_i + b_i + 1) / 2\rfloor$.
#[inline(always)]
pub const fn swar_q7_avg_round_x8(a_packed: u64, b_packed: u64) -> u64 {
    let ua = a_packed ^ SWAR_SIGN_MASK_64;
    let ub = b_packed ^ SWAR_SIGN_MASK_64;
    let uavg = (ua | ub).wrapping_sub(((ua ^ ub) >> 1) & SWAR_LOW7_MASK_64);
    uavg ^ SWAR_SIGN_MASK_64
}

/// Precomputed-bias SWAR 8-lane Q7 scale for `s <= 128` where `bias_restore = (128 - s) * 0x0101_0101_0101_0101`.
#[inline(always)]
pub const fn swar_q7_scale_le128_with_bias_x8(packed: u64, s: u64, bias_restore: u64) -> u64 {
    let u = packed ^ SWAR_SIGN_MASK_64;
    let u_even = u & SWAR_EVEN_BYTES_64;
    let u_odd = (u >> 8) & SWAR_EVEN_BYTES_64;
    let scaled_even = (u_even.wrapping_mul(s) >> 7) & SWAR_EVEN_BYTES_64;
    let scaled_odd = (u_odd.wrapping_mul(s) << 1) & SWAR_ODD_BYTES_64;
    (scaled_even | scaled_odd).wrapping_add(bias_restore) ^ SWAR_SIGN_MASK_64
}

#[inline(always)]
pub(crate) const fn swar_q7_scale_le128_x8(packed: u64, s: u64) -> u64 {
    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    swar_q7_scale_le128_with_bias_x8(packed, s, bias_restore)
}

/// True bit-parallel SWAR 8-lane Q7 fixed-point gain scaling in strictly constant time (`0` branches).
#[inline(always)]
pub const fn swar_q7_scale_u7_x8(packed: u64, scale_u7: u8) -> u64 {
    let gt_128 = scale_u7 > 128;
    let s_sub = branchless_select_u8(gt_128, scale_u7.wrapping_sub(128), scale_u7) as u64;
    let base_scaled = swar_q7_scale_le128_x8(packed, s_sub);
    let extra = branchless_select_u64(gt_128, packed, 0);
    swar_q7_add_sat_x8(base_scaled, extra)
}

#[inline(always)]
const fn swar_q7_qdps_filter_with_thresh_x8(
    packed: u64,
    thresh_x8: u64,
    valid_step_mask: u64,
) -> u64 {
    let abs_lanes = swar_q7_abs_u8_x8(packed);
    let ge_bit7 = (((abs_lanes | SWAR_SIGN_MASK_64).wrapping_sub(thresh_x8))
        | (abs_lanes & SWAR_SIGN_MASK_64))
        & SWAR_SIGN_MASK_64;
    let keep_mask = (ge_bit7 >> 7).wrapping_mul(0xFF);
    packed & keep_mask & valid_step_mask
}

/// True bit-parallel SWAR 8-lane QDPS discrete phase step cutoff filter in strictly constant time (`0` branches).
#[inline(always)]
pub const fn swar_q7_qdps_filter_x8(packed: u64, min_step_q7: u8) -> u64 {
    let thresh_x8 = ((min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let valid_step_mask = ((min_step_q7 <= 127) as u64).wrapping_neg();
    swar_q7_qdps_filter_with_thresh_x8(packed, thresh_x8, valid_step_mask)
}

/// Applies the 8-lane SWAR QDPS discrete phase step cutoff filter across 8 register-resident
/// `u64` SWAR words using a pre-evaluated word-uniformity flag (`w_is_uniform`), returning
/// `(filtered_words, filtered_is_uniform)`.
#[inline(always)]
#[must_use]
pub const fn swar_q7_qdps_filter_words_with_uniform_flag_64(
    w: &[u64; 8],
    w_is_uniform: bool,
    min_step_q7: u8,
) -> ([u64; 8], bool) {
    if min_step_q7 <= 1 {
        return (*w, w_is_uniform);
    }
    if min_step_q7 > 127 {
        return ([0u64; 8], true);
    }
    let thresh_x8 = ((min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    if w_is_uniform {
        let w0 = w[0];
        if w0 == 0 {
            return ([0u64; 8], true);
        }
        let b0 = (w0 & 0xFF) as u8;
        if w0 == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
            let abs_b0 = (b0 as i8).unsigned_abs();
            return if abs_b0 >= min_step_q7 {
                (*w, true)
            } else {
                ([0u64; 8], true)
            };
        }
        return (
            [swar_q7_qdps_filter_with_thresh_x8(w0, thresh_x8, u64::MAX); 8],
            true,
        );
    }
    let out = [
        swar_q7_qdps_filter_with_thresh_x8(w[0], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[1], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[2], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[3], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[4], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[5], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[6], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[7], thresh_x8, u64::MAX),
    ];
    (out, is_word_array_uniform_64(&out))
}

/// Applies the 8-lane SWAR QDPS discrete phase step cutoff filter across all 8 register-resident
/// `u64` SWAR words (`64` attention heads) with hoisted `thresh_x8` and `valid_step_mask` in
/// strictly constant time (`0` data-dependent branches).
#[inline(always)]
#[must_use]
pub const fn swar_q7_qdps_filter_words_64(w: &[u64; 8], min_step_q7: u8) -> [u64; 8] {
    if min_step_q7 <= 1 {
        return *w;
    }
    if min_step_q7 > 127 {
        return [0u64; 8];
    }
    swar_q7_qdps_filter_words_with_uniform_flag_64(w, is_word_array_uniform_64(w), min_step_q7).0
}

#[inline(always)]
pub(crate) const fn load_q7_word_at(arr: &[gcso_q7_t; MAX_ATTENTION_HEADS], base: usize) -> u64 {
    u64::from_le_bytes([
        arr[base] as u8,
        arr[base + 1] as u8,
        arr[base + 2] as u8,
        arr[base + 3] as u8,
        arr[base + 4] as u8,
        arr[base + 5] as u8,
        arr[base + 6] as u8,
        arr[base + 7] as u8,
    ])
}

#[inline(always)]
pub(crate) const fn store_q7_word_at(
    arr: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    base: usize,
    word: u64,
) {
    let b = word.to_le_bytes();
    arr[base] = b[0] as gcso_q7_t;
    arr[base + 1] = b[1] as gcso_q7_t;
    arr[base + 2] = b[2] as gcso_q7_t;
    arr[base + 3] = b[3] as gcso_q7_t;
    arr[base + 4] = b[4] as gcso_q7_t;
    arr[base + 5] = b[5] as gcso_q7_t;
    arr[base + 6] = b[6] as gcso_q7_t;
    arr[base + 7] = b[7] as gcso_q7_t;
}

/// Loads all 64 Q7 head phases into 8 `u64` SWAR registers using 8 direct 64-bit GPR word loads
/// (`mov r64, qword ptr [arr + 8*c]`), avoiding 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF)
/// stalls and 64-byte cache-line split penalties when reading `trail.accumulated_phase_delta` (offset 56).
#[inline(always)]
pub const fn load_q7_words_64(arr: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> [u64; 8] {
    [
        load_q7_word_at(arr, 0),
        load_q7_word_at(arr, 8),
        load_q7_word_at(arr, 16),
        load_q7_word_at(arr, 24),
        load_q7_word_at(arr, 32),
        load_q7_word_at(arr, 40),
        load_q7_word_at(arr, 48),
        load_q7_word_at(arr, 56),
    ]
}

/// Borrows `trail.accumulated_phase_delta` (`offset 56` of `#[repr(C, align(128))]` [`gcso_pointer_trail_t`],
/// guaranteed 8-byte aligned) as a zero-copy `&[u64; 8]` SWAR reference on little-endian targets,
/// eliminating 64-byte stack copies and 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF) stalls.
#[inline(always)]
#[must_use]
pub const fn trail_phase_words_ref(trail: &gcso_pointer_trail_t) -> &[u64; 8] {
    const _: () = assert!(cfg!(target_endian = "little"));
    // SAFETY: `gcso_pointer_trail_t` is `#[repr(C, align(128))]` and `accumulated_phase_delta` is at
    // byte offset 56 (`56 % 8 == 0`, 8-byte aligned), with identical 64-byte layout to `[u64; 8]`.
    unsafe { &*(trail.accumulated_phase_delta.as_ptr().cast::<[u64; 8]>()) }
}

/// Mutably borrows `trail.accumulated_phase_delta` (`offset 56` of `#[repr(C, align(128))]`
/// [`gcso_pointer_trail_t`], guaranteed 8-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
#[inline(always)]
#[must_use]
pub const fn trail_phase_words_mut(trail: &mut gcso_pointer_trail_t) -> &mut [u64; 8] {
    const _: () = assert!(cfg!(target_endian = "little"));
    // SAFETY: `gcso_pointer_trail_t` is `#[repr(C, align(128))]` and `accumulated_phase_delta` is at
    // byte offset 56 (`56 % 8 == 0`, 8-byte aligned), with identical 64-byte layout to `[u64; 8]`.
    unsafe { &mut *(trail.accumulated_phase_delta.as_mut_ptr().cast::<[u64; 8]>()) }
}

/// Stores 8 `u64` SWAR registers into a 64-head Q7 phase array using 8 direct 64-bit GPR word stores
/// (`mov qword ptr [arr + 8*c], r64`), eliminating narrow-GPR-store-to-wide-YMM-load Store-to-Load
/// Forwarding (STLF) stalls (`~15–20 cycles`) and 64-byte cache-line split stores on `trail.accumulated_phase_delta`.
#[inline(always)]
pub const fn store_q7_words_64(arr: &mut [gcso_q7_t; MAX_ATTENTION_HEADS], w: &[u64; 8]) {
    store_q7_word_at(arr, 0, w[0]);
    store_q7_word_at(arr, 8, w[1]);
    store_q7_word_at(arr, 16, w[2]);
    store_q7_word_at(arr, 24, w[3]);
    store_q7_word_at(arr, 32, w[4]);
    store_q7_word_at(arr, 40, w[5]);
    store_q7_word_at(arr, 48, w[6]);
    store_q7_word_at(arr, 56, w[7]);
}

/// Returns `true` if all 8 `u64` SWAR words in `w` are identical (`w[0] == w[1] == ... == w[7]`)
/// using a branchless 8-word XOR-OR reduction tree (`0` branches).
#[inline(always)]
pub const fn is_word_array_uniform_64(w: &[u64; 8]) -> bool {
    let w0 = w[0];
    if w0 != w[1] {
        return false;
    }
    (((w0 ^ w[2]) | (w0 ^ w[3])) | ((w0 ^ w[4]) | (w0 ^ w[5])) | ((w0 ^ w[6]) | (w0 ^ w[7])))
        == 0
}

#[inline(always)]
const fn swar_u8_pair_sum16(packed_u8: u64) -> u64 {
    (packed_u8 & SWAR_EVEN_BYTES_64).wrapping_add((packed_u8 >> 8) & SWAR_EVEN_BYTES_64)
}

/// Computes the rounded arithmetic mean Q7 phase across 64 heads directly from two 256-bit
/// AVX2 vectors `(v0, v1)` using unsigned byte SAD (`vpsadbw`) against zero after sign-bit
/// flipping (`^ 0x80`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_q7_mean_i8_from_vec64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
) -> gcso_q7_t {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_add_epi64, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_sad_epu8,
            _mm256_set1_epi8, _mm256_setzero_si256, _mm256_xor_si256, _mm_add_epi64,
            _mm_cvtsi128_si64, _mm_unpackhi_epi64,
        };
        let sign = _mm256_set1_epi8(-128i8);
        let zero = _mm256_setzero_si256();
        let sad0 = _mm256_sad_epu8(_mm256_xor_si256(v0, sign), zero);
        let sad1 = _mm256_sad_epu8(_mm256_xor_si256(v1, sign), zero);
        let sum256 = _mm256_add_epi64(sad0, sad1);
        let sum128 = _mm_add_epi64(
            _mm256_castsi256_si128(sum256),
            _mm256_extracti128_si256::<1>(sum256),
        );
        let sum64 = _mm_add_epi64(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let total_u8 = _mm_cvtsi128_si64(sum64) as i32;
        let signed_sum = total_u8 - (64 * 128);
        let rounded = (signed_sum + branchless_select_i32(signed_sum >= 0, 32, -32)) / 64;
        branchless_select_i32(
            rounded < -128,
            -128,
            branchless_select_i32(rounded > 127, 127, rounded),
        ) as gcso_q7_t
    }
}

/// Computes the rounded arithmetic mean Q7 phase across 8 packed `u64` SWAR words (64 heads)
/// using a pre-evaluated word-uniformity flag (`words_is_uniform`).
#[inline(always)]
pub const fn swar_q7_mean_i8_from_words_with_uniform_flag_64(
    words: &[u64; 8],
    words_is_uniform: bool,
) -> gcso_q7_t {
    if words_is_uniform {
        let w0 = words[0];
        if w0 == (w0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
            return (w0 & 0xFF) as u8 as gcso_q7_t;
        }
        let signed_sum8 = (swar_u8_sum_x8(w0 ^ SWAR_SIGN_MASK_64) as i32) - (8 * 128);
        let rounded8 = (signed_sum8 + branchless_select_i32(signed_sum8 >= 0, 4, -4)) / 8;
        return branchless_select_i32(
            rounded8 < -128,
            -128,
            branchless_select_i32(rounded8 > 127, 127, rounded8),
        ) as gcso_q7_t;
    }
    let p01 = swar_u8_pair_sum16(words[0] ^ SWAR_SIGN_MASK_64)
        .wrapping_add(swar_u8_pair_sum16(words[1] ^ SWAR_SIGN_MASK_64));
    let p23 = swar_u8_pair_sum16(words[2] ^ SWAR_SIGN_MASK_64)
        .wrapping_add(swar_u8_pair_sum16(words[3] ^ SWAR_SIGN_MASK_64));
    let p45 = swar_u8_pair_sum16(words[4] ^ SWAR_SIGN_MASK_64)
        .wrapping_add(swar_u8_pair_sum16(words[5] ^ SWAR_SIGN_MASK_64));
    let p67 = swar_u8_pair_sum16(words[6] ^ SWAR_SIGN_MASK_64)
        .wrapping_add(swar_u8_pair_sum16(words[7] ^ SWAR_SIGN_MASK_64));
    let sum16_all = p01.wrapping_add(p23).wrapping_add(p45.wrapping_add(p67));
    let total_u8 = (sum16_all.wrapping_mul(SWAR_SUM_U16_64) >> 48) as i32;
    let signed_sum = total_u8 - (64 * 128);
    let rounded = (signed_sum + branchless_select_i32(signed_sum >= 0, 32, -32)) / 64;
    branchless_select_i32(
        rounded < -128,
        -128,
        branchless_select_i32(rounded > 127, 127, rounded),
    ) as gcso_q7_t
}

/// Computes the rounded arithmetic mean Q7 phase across all 8 packed `u64` SWAR words (64 heads)
/// in strictly constant time (`0` data-dependent branches).
#[inline(always)]
pub const fn swar_q7_mean_i8_from_words_64(words: &[u64; 8]) -> gcso_q7_t {
    swar_q7_mean_i8_from_words_with_uniform_flag_64(words, is_word_array_uniform_64(words))
}

/// Returns `true` if any of the 64 Q7 head phases in `phases` is non-zero (`0` branches).
#[inline(always)]
pub const fn is_phase_profile_nonzero_64(phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> bool {
    let w = load_q7_words_64(phases);
    (((w[0] | w[1]) | (w[2] | w[3])) | ((w[4] | w[5]) | (w[6] | w[7]))) != 0
}

/// Accumulates `deltas` into `acc` across all 64 attention heads using 8-word SWAR saturating addition.
#[inline(always)]
pub const fn branchless_q7_accumulate_64(
    acc: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    deltas: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) {
    let a = load_q7_words_64(acc);
    let d = load_q7_words_64(deltas);
    store_q7_words_64(
        acc,
        &[
            swar_q7_add_sat_x8(a[0], d[0]),
            swar_q7_add_sat_x8(a[1], d[1]),
            swar_q7_add_sat_x8(a[2], d[2]),
            swar_q7_add_sat_x8(a[3], d[3]),
            swar_q7_add_sat_x8(a[4], d[4]),
            swar_q7_add_sat_x8(a[5], d[5]),
            swar_q7_add_sat_x8(a[6], d[6]),
            swar_q7_add_sat_x8(a[7], d[7]),
        ],
    );
}

/// Subtracts `b` from `a` across all 64 attention heads using 8-word SWAR saturating subtraction.
#[inline(always)]
pub const fn branchless_q7_sub_64(
    a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let wa = load_q7_words_64(a);
    let wb = load_q7_words_64(b);
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    store_q7_words_64(
        &mut out,
        &[
            swar_q7_sub_sat_x8(wa[0], wb[0]),
            swar_q7_sub_sat_x8(wa[1], wb[1]),
            swar_q7_sub_sat_x8(wa[2], wb[2]),
            swar_q7_sub_sat_x8(wa[3], wb[3]),
            swar_q7_sub_sat_x8(wa[4], wb[4]),
            swar_q7_sub_sat_x8(wa[5], wb[5]),
            swar_q7_sub_sat_x8(wa[6], wb[6]),
            swar_q7_sub_sat_x8(wa[7], wb[7]),
        ],
    );
    out
}

/// Conjugates all 64 Q7 head phases ($-\boldsymbol{\Delta\theta}$) using 8-word SWAR saturating negation.
#[inline(always)]
pub const fn branchless_q7_conjugate_64(
    phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let w = load_q7_words_64(phases);
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    store_q7_words_64(
        &mut out,
        &[
            swar_q7_neg_sat_x8(w[0]),
            swar_q7_neg_sat_x8(w[1]),
            swar_q7_neg_sat_x8(w[2]),
            swar_q7_neg_sat_x8(w[3]),
            swar_q7_neg_sat_x8(w[4]),
            swar_q7_neg_sat_x8(w[5]),
            swar_q7_neg_sat_x8(w[6]),
            swar_q7_neg_sat_x8(w[7]),
        ],
    );
    out
}

/// Scales all 64 Q7 head phases in-place by `scale_u7 / 128` using 8-word SWAR arithmetic.
#[inline(always)]
pub const fn branchless_q7_scale_64(phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS], scale_u7: u8) {
    let gt_128 = scale_u7 > 128;
    let s = branchless_select_u8(gt_128, scale_u7.wrapping_sub(128), scale_u7) as u64;
    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let w = load_q7_words_64(phases);
    let s0 = swar_q7_scale_le128_with_bias_x8(w[0], s, bias_restore);
    let s1 = swar_q7_scale_le128_with_bias_x8(w[1], s, bias_restore);
    let s2 = swar_q7_scale_le128_with_bias_x8(w[2], s, bias_restore);
    let s3 = swar_q7_scale_le128_with_bias_x8(w[3], s, bias_restore);
    let s4 = swar_q7_scale_le128_with_bias_x8(w[4], s, bias_restore);
    let s5 = swar_q7_scale_le128_with_bias_x8(w[5], s, bias_restore);
    let s6 = swar_q7_scale_le128_with_bias_x8(w[6], s, bias_restore);
    let s7 = swar_q7_scale_le128_with_bias_x8(w[7], s, bias_restore);

    store_q7_words_64(
        phases,
        &[
            swar_q7_add_sat_x8(s0, branchless_select_u64(gt_128, w[0], 0)),
            swar_q7_add_sat_x8(s1, branchless_select_u64(gt_128, w[1], 0)),
            swar_q7_add_sat_x8(s2, branchless_select_u64(gt_128, w[2], 0)),
            swar_q7_add_sat_x8(s3, branchless_select_u64(gt_128, w[3], 0)),
            swar_q7_add_sat_x8(s4, branchless_select_u64(gt_128, w[4], 0)),
            swar_q7_add_sat_x8(s5, branchless_select_u64(gt_128, w[5], 0)),
            swar_q7_add_sat_x8(s6, branchless_select_u64(gt_128, w[6], 0)),
            swar_q7_add_sat_x8(s7, branchless_select_u64(gt_128, w[7], 0)),
        ],
    );
}

/// Applies the 8-lane SWAR QDPS discrete phase step cutoff filter across all 64 attention heads
/// in strictly constant time (`0` data-dependent branches) via [`swar_q7_qdps_filter_words_64`].
#[inline(always)]
pub const fn branchless_q7_qdps_filter_64(
    phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    min_step_q7: u8,
) {
    if min_step_q7 <= 1 {
        return;
    }
    if min_step_q7 > 127 {
        *phases = [0; MAX_ATTENTION_HEADS];
        return;
    }
    let w = load_q7_words_64(phases);
    let out_w = swar_q7_qdps_filter_words_64(&w, min_step_q7);
    store_q7_words_64(phases, &out_w);
}

/// Unchecked 8-lane SWAR shortest-arc $S^1$ geodesic interpolation for `1 <= s <= 127` and
/// precomputed `bias_restore = (128 - s) * 0x0101_0101_0101_0101` (`0` branches).
#[inline(always)]
pub const fn swar_slerp_word_unchecked_x8(a: u64, b: u64, s: u64, bias_restore: u64) -> u64 {
    swar_q7_circular_add_x8(
        a,
        swar_q7_scale_le128_with_bias_x8(swar_q7_circular_diff_x8(b, a), s, bias_restore),
    )
}

/// Interpolates two packed 8-lane Q7 words (`a` and `b`) along the shortest-arc $S^1$ geodesic
/// using a pre-hoisted `bias_restore = (128 - s) * 0x0101_0101_0101_0101` (`0` data-dependent branches).
#[inline(always)]
pub const fn swar_slerp_word_with_bias_x8(a: u64, b: u64, s: u64, bias_restore: u64) -> u64 {
    if s == 0 {
        return a;
    }
    if s >= 128 {
        return b;
    }
    swar_slerp_word_unchecked_x8(a, b, s, bias_restore)
}

/// Moves 8 `u64` words directly into two 256-bit YMM registers `(v0, v1)` via 64-bit XMM/YMM
/// register inserts (`vmovq` + `vpinsrq` + `vinserti128`), avoiding the 64-bit-store-to-256-bit-load
/// Store-to-Load Forwarding (STLF) pipeline stall caused by stack `transmute` or `_mm256_loadu_si256`
/// on recently stored 64-bit word arrays.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_u64x8_to_vec64(
    w0: u64,
    w1: u64,
    w2: u64,
    w3: u64,
    w4: u64,
    w5: u64,
    w6: u64,
    w7: u64,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castsi128_si256, _mm256_inserti128_si256, _mm_cvtsi64_si128, _mm_insert_epi64,
        };
        let lo0 = _mm_insert_epi64(_mm_cvtsi64_si128(w0 as i64), w1 as i64, 1);
        let hi0 = _mm_insert_epi64(_mm_cvtsi64_si128(w2 as i64), w3 as i64, 1);
        let lo1 = _mm_insert_epi64(_mm_cvtsi64_si128(w4 as i64), w5 as i64, 1);
        let hi1 = _mm_insert_epi64(_mm_cvtsi64_si128(w6 as i64), w7 as i64, 1);
        (
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo0), hi0, 1),
            _mm256_inserti128_si256(_mm256_castsi128_si256(lo1), hi1, 1),
        )
    }
}

/// Moves a `&[u64; 8]` word array into two 256-bit YMM registers `(v0, v1)` via `avx2_u64x8_to_vec64`
/// without triggering a 4-to-1 Store-to-Load Forwarding stall.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_words64_to_vec64(
    w: &[u64; 8],
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe { avx2_u64x8_to_vec64(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]) }
}

/// Moves a `&[u64; 8]` word array into two 256-bit YMM registers `(v0, v1)` using a single
/// `vpxor` or `vmovq + vpbroadcastq` when `w_is_uniform` is `true` (`1..2` instructions vs `10`
/// Port-5 inserts), or `avx2_words64_to_vec64` when heterogeneous.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_words64_to_vec64_with_uniform(
    w: &[u64; 8],
    w_is_uniform: bool,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{_mm256_broadcastq_epi64, _mm256_setzero_si256, _mm_cvtsi64_si128};
        if w_is_uniform {
            let w0 = w[0];
            if w0 == 0 {
                let z = _mm256_setzero_si256();
                return (z, z);
            }
            let v = _mm256_broadcastq_epi64(_mm_cvtsi64_si128(w0 as i64));
            return (v, v);
        }
        avx2_words64_to_vec64(w)
    }
}

/// Extracts two 256-bit YMM registers `(v0, v1)` directly into 8 `u64` GPR words (`[u64; 8]`)
/// via `vextracti128` + `vmovq` + `vpextrq`, avoiding the 256-bit-store-to-64-bit-load
/// Store-to-Load Forwarding (STLF) pipeline stall caused by stack `transmute`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_vec64_to_words64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
) -> [u64; 8] {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castsi256_si128, _mm256_extracti128_si256, _mm_cvtsi128_si64, _mm_extract_epi64,
        };
        let lo0 = _mm256_castsi256_si128(v0);
        let hi0 = _mm256_extracti128_si256(v0, 1);
        let lo1 = _mm256_castsi256_si128(v1);
        let hi1 = _mm256_extracti128_si256(v1, 1);
        [
            _mm_cvtsi128_si64(lo0) as u64,
            _mm_extract_epi64(lo0, 1) as u64,
            _mm_cvtsi128_si64(hi0) as u64,
            _mm_extract_epi64(hi0, 1) as u64,
            _mm_cvtsi128_si64(lo1) as u64,
            _mm_extract_epi64(lo1, 1) as u64,
            _mm_cvtsi128_si64(hi1) as u64,
            _mm_extract_epi64(hi1, 1) as u64,
        ]
    }
}

/// Scales precomputed circular difference vectors `(vd0, vd1) = (vb - va)` by `s in 0..=128`
/// and adds them onto `(va0, va1)` across all 64 Q7 lanes in YMM registers (`0` redundant `vpsubb`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_slerp_from_diff_vec64(
    va0: core::arch::x86_64::__m256i,
    va1: core::arch::x86_64::__m256i,
    vd0: core::arch::x86_64::__m256i,
    vd1: core::arch::x86_64::__m256i,
    s: u64,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_add_epi8, _mm256_and_si256, _mm256_mulhrs_epi16, _mm256_or_si256,
            _mm256_set1_epi16, _mm256_slli_epi16,
        };
        let v_s = _mm256_set1_epi16(s as i16);
        let mask_ff00 = _mm256_set1_epi16(-256i16);
        let mask_00ff = _mm256_set1_epi16(0x00FFi16);

        let ev0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd0, 8), v_s);
        let od0 = _mm256_mulhrs_epi16(_mm256_and_si256(vd0, mask_ff00), v_s);
        let sc0 = _mm256_or_si256(_mm256_and_si256(ev0, mask_00ff), _mm256_slli_epi16(od0, 8));
        let out0 = _mm256_add_epi8(va0, sc0);

        let ev1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd1, 8), v_s);
        let od1 = _mm256_mulhrs_epi16(_mm256_and_si256(vd1, mask_ff00), v_s);
        let sc1 = _mm256_or_si256(_mm256_and_si256(ev1, mask_00ff), _mm256_slli_epi16(od1, 8));
        let out1 = _mm256_add_epi8(va1, sc1);
        (out0, out1)
    }
}

/// Interpolates two 256-bit YMM register pairs `(va0, va1)` and `(vb0, vb1)` along the shortest-arc
/// $S^1$ geodesic across all 64 Q7 lanes in YMM registers (`0` memory or GPR roundtrips).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_slerp_vec64(
    va0: core::arch::x86_64::__m256i,
    va1: core::arch::x86_64::__m256i,
    vb0: core::arch::x86_64::__m256i,
    vb1: core::arch::x86_64::__m256i,
    s: u64,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::_mm256_sub_epi8;
        avx2_slerp_from_diff_vec64(
            va0,
            va1,
            _mm256_sub_epi8(vb0, va0),
            _mm256_sub_epi8(vb1, va1),
            s,
        )
    }
}

/// Scales all 64 signed Q7 phase lanes in `(v0, v1)` by `s in 0..=128` (`128 == 1.0x`) using
/// `_mm256_mulhrs_epi16`, matching `swar_q7_scale_le128_with_bias_x8` bit-for-bit across all inputs.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_scale_vec64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
    s: u64,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_and_si256, _mm256_mulhrs_epi16, _mm256_or_si256, _mm256_set1_epi16,
            _mm256_slli_epi16,
        };
        let v_s = _mm256_set1_epi16(s as i16);
        let mask_ff00 = _mm256_set1_epi16(-256i16);
        let mask_00ff = _mm256_set1_epi16(0x00FFi16);
        let ev0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(v0, 8), v_s);
        let od0 = _mm256_mulhrs_epi16(_mm256_and_si256(v0, mask_ff00), v_s);
        let sc0 = _mm256_or_si256(_mm256_and_si256(ev0, mask_00ff), _mm256_slli_epi16(od0, 8));
        let ev1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(v1, 8), v_s);
        let od1 = _mm256_mulhrs_epi16(_mm256_and_si256(v1, mask_ff00), v_s);
        let sc1 = _mm256_or_si256(_mm256_and_si256(ev1, mask_00ff), _mm256_slli_epi16(od1, 8));
        (sc0, sc1)
    }
}

/// Filters all 64 signed Q7 phase lanes in `(v0, v1)` against `min_step_q7` in 8 AVX2 instructions
/// (`vpabsb + vpmaxub + vpcmpeqb + vpand`), matching `swar_q7_qdps_filter_x8` bit-for-bit.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_qdps_filter_vec64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
    min_step_q7: u8,
) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_abs_epi8, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_max_epu8,
            _mm256_set1_epi8, _mm256_setzero_si256,
        };
        if min_step_q7 <= 1 {
            return (v0, v1);
        }
        if min_step_q7 > 127 {
            let z = _mm256_setzero_si256();
            return (z, z);
        }
        let v_step = _mm256_set1_epi8(min_step_q7 as i8);
        let a0 = _mm256_abs_epi8(v0);
        let a1 = _mm256_abs_epi8(v1);
        let k0 = _mm256_cmpeq_epi8(_mm256_max_epu8(a0, v_step), a0);
        let k1 = _mm256_cmpeq_epi8(_mm256_max_epu8(a1, v_step), a1);
        (_mm256_and_si256(v0, k0), _mm256_and_si256(v1, k1))
    }
}

/// Interpolates two 8-word (64-head) SWAR arrays (`a` and `b`) along the shortest-arc $S^1$
/// geodesic using pre-evaluated word-uniformity flags (`a_is_uniform`, `b_is_uniform`),
/// returning `(interpolated_words, out_is_uniform)` directly in 64-bit GPRs (`0` GPR-to-YMM or
/// YMM-to-GPR Store-to-Load Forwarding stalls, `1` shared `step0` scale when `b - a` is word-uniform).
#[inline(always)]
#[must_use]
pub const fn swar_slerp_words_with_uniform_flag_64(
    a: &[u64; 8],
    a_is_uniform: bool,
    b: &[u64; 8],
    b_is_uniform: bool,
    s: u64,
    bias_restore: u64,
) -> ([u64; 8], bool) {
    if s == 0 {
        return (*a, a_is_uniform);
    }
    if s >= 128 {
        return (*b, b_is_uniform);
    }
    if a_is_uniform & b_is_uniform {
        if a[0] == b[0] {
            return (*a, true);
        }
        let w0 = swar_slerp_word_unchecked_x8(a[0], b[0], s, bias_restore);
        return ([w0; 8], true);
    }
    let d0 = swar_q7_circular_diff_x8(b[0], a[0]);
    let d1 = swar_q7_circular_diff_x8(b[1], a[1]);
    let d2 = swar_q7_circular_diff_x8(b[2], a[2]);
    let d3 = swar_q7_circular_diff_x8(b[3], a[3]);
    let d4 = swar_q7_circular_diff_x8(b[4], a[4]);
    let d5 = swar_q7_circular_diff_x8(b[5], a[5]);
    let d6 = swar_q7_circular_diff_x8(b[6], a[6]);
    let d7 = swar_q7_circular_diff_x8(b[7], a[7]);
    if (d0 == d1)
        && ((((d0 ^ d2) | (d0 ^ d3)) | ((d0 ^ d4) | (d0 ^ d5))) | ((d0 ^ d6) | (d0 ^ d7))) == 0
    {
        if d0 == 0 {
            return (*a, a_is_uniform);
        }
        let step0 = swar_q7_scale_le128_with_bias_x8(d0, s, bias_restore);
        if step0 == 0 {
            return (*a, a_is_uniform);
        }
        let out = [
            swar_q7_circular_add_x8(a[0], step0),
            swar_q7_circular_add_x8(a[1], step0),
            swar_q7_circular_add_x8(a[2], step0),
            swar_q7_circular_add_x8(a[3], step0),
            swar_q7_circular_add_x8(a[4], step0),
            swar_q7_circular_add_x8(a[5], step0),
            swar_q7_circular_add_x8(a[6], step0),
            swar_q7_circular_add_x8(a[7], step0),
        ];
        return (out, a_is_uniform);
    }
    let out = [
        swar_q7_circular_add_x8(a[0], swar_q7_scale_le128_with_bias_x8(d0, s, bias_restore)),
        swar_q7_circular_add_x8(a[1], swar_q7_scale_le128_with_bias_x8(d1, s, bias_restore)),
        swar_q7_circular_add_x8(a[2], swar_q7_scale_le128_with_bias_x8(d2, s, bias_restore)),
        swar_q7_circular_add_x8(a[3], swar_q7_scale_le128_with_bias_x8(d3, s, bias_restore)),
        swar_q7_circular_add_x8(a[4], swar_q7_scale_le128_with_bias_x8(d4, s, bias_restore)),
        swar_q7_circular_add_x8(a[5], swar_q7_scale_le128_with_bias_x8(d5, s, bias_restore)),
        swar_q7_circular_add_x8(a[6], swar_q7_scale_le128_with_bias_x8(d6, s, bias_restore)),
        swar_q7_circular_add_x8(a[7], swar_q7_scale_le128_with_bias_x8(d7, s, bias_restore)),
    ];
    (out, is_word_array_uniform_64(&out))
}

/// Interpolates two 8-word (64-head) SWAR arrays (`a` and `b`) along the shortest-arc $S^1$
/// geodesic across all 8 words (`64` heads) in strictly constant time (`0` data-dependent branches).
#[inline(always)]
#[must_use]
pub const fn swar_slerp_words_with_bias_64(
    a: &[u64; 8],
    b: &[u64; 8],
    s: u64,
    bias_restore: u64,
) -> [u64; 8] {
    if s == 0 {
        return *a;
    }
    if s >= 128 {
        return *b;
    }
    let a_uni = is_word_array_uniform_64(a);
    let b_uni = if a_uni { is_word_array_uniform_64(b) } else { false };
    swar_slerp_words_with_uniform_flag_64(a, a_uni, b, b_uni, s, bias_restore).0
}

/// True bit-parallel SWAR 8-lane shortest-arc geodesic interpolation on $S^1$ (`t_u7 in 0..=128`).
#[inline(always)]
pub const fn swar_q7_geodesic_slerp_x8(a_packed: u64, b_packed: u64, t_u7: u8) -> u64 {
    let s = branchless_select_u8(t_u7 > 128, 128, t_u7) as u64;
    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    swar_slerp_word_with_bias_x8(a_packed, b_packed, s, bias_restore)
}

/// Performs shortest-arc geodesic interpolation on $S^1$ across all 64 attention heads in strictly
/// constant time, hoisting `bias_restore` once across all 8 SWAR words.
#[inline(always)]
pub fn branchless_q7_slerp_64(
    phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    t_u7: u8,
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let s = branchless_select_u8(t_u7 > 128, 128, t_u7) as u64;
    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let a = load_q7_words_64(phases_a);
    let b = load_q7_words_64(phases_b);
    let out_w = swar_slerp_words_with_bias_64(&a, &b, s, bias_restore);
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    store_q7_words_64(&mut out, &out_w);
    out
}

/// Folds a candidate 64-head phase vector `candidate_phases` (with weight `candidate_weight`)
/// into `running_phases` (with weight `running_weight`) along the shortest-arc $S^1$ geodesic
/// in strictly constant time (`0 fdiv`). Returns the updated total weight.
#[inline(always)]
pub fn branchless_q7_circular_weighted_step_64(
    running_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    running_weight: f32,
    candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    candidate_weight: f32,
) -> f32 {
    let w_run = branchless_flush_subnormal_f32(branchless_relu_f32(running_weight));
    let w_cand = branchless_flush_subnormal_f32(branchless_relu_f32(candidate_weight));
    let total_w = w_run + w_cand;
    if w_cand <= 1.0e-12 {
        return total_w;
    }
    if w_run <= 1.0e-12 {
        *running_phases = *candidate_phases;
        return total_w;
    }
    let ratio_f32 = branchless_clamp_f32(
        (w_cand * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
        0.0,
        128.0,
    );
    let s = (ratio_f32 as u8) as u64;
    if s >= 128 {
        *running_phases = *candidate_phases;
    } else if s > 0 {
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let a = load_q7_words_64(running_phases);
        let b = load_q7_words_64(candidate_phases);
        let a_uni = is_word_array_uniform_64(&a);
        let b_uni = if a_uni {
            is_word_array_uniform_64(&b)
        } else {
            false
        };
        let (out_w, _) =
            swar_slerp_words_with_uniform_flag_64(&a, a_uni, &b, b_uni, s, bias_restore);
        store_q7_words_64(running_phases, &out_w);
    }
    total_w
}

/// Sums 8 Q14 cosine values from a single packed `u64` circular-difference word in strictly
/// constant time (`0` branches) via a balanced 4-port ILP tree.
#[inline(always)]
const fn sum_cos_q14_from_diff_word(dw: u64) -> i32 {
    let lut = Q7_FULL_COS_I16_REF;
    let b = dw.to_le_bytes();
    let c01 = (lut[b[0] as usize] as i32) + (lut[b[1] as usize] as i32);
    let c23 = (lut[b[2] as usize] as i32) + (lut[b[3] as usize] as i32);
    let c45 = (lut[b[4] as usize] as i32) + (lut[b[5] as usize] as i32);
    let c67 = (lut[b[6] as usize] as i32) + (lut[b[7] as usize] as i32);
    (c01 + c23) + (c45 + c67)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_coherence_from_diff_vec64_with_floor(
    vd0: core::arch::x86_64::__m256i,
    vd1: core::arch::x86_64::__m256i,
    floor_coh: f32,
) -> f32 {
    // SAFETY: Guarded by `target_feature = "avx2"`. Evaluates the 6-instruction `vpabsb + vpsadbw`
    // 2-piece Lagrangian $L_1$ upper-bound screen (`min(1.052 - s_l1 / 8192, 1.215 - s_l1 / 4096)`)
    // and `s_l1 == 0` exact-match fast path BEFORE `vpbroadcastb`/`vpbroadcastq` uniformity checks,
    // cutting per-pair rejection work in `merge_and_prune_centroids_64` and $k$-Means E-steps in half.
    unsafe {
        use core::arch::x86_64::{
            _mm256_abs_epi8, _mm256_add_epi64, _mm256_broadcastb_epi8, _mm256_broadcastq_epi64,
            _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_or_si256, _mm256_sad_epu8,
            _mm256_setzero_si256, _mm256_testz_si256, _mm256_xor_si256, _mm_add_epi64,
            _mm_cvtsi128_si32, _mm_cvtsi128_si64, _mm_unpackhi_epi64,
        };
        let raw_abs0 = _mm256_abs_epi8(vd0);
        let raw_abs1 = _mm256_abs_epi8(vd1);
        let zero = _mm256_setzero_si256();
        let sad_l1 = _mm256_add_epi64(
            _mm256_sad_epu8(raw_abs0, zero),
            _mm256_sad_epu8(raw_abs1, zero),
        );
        let l1_128 = _mm_add_epi64(
            _mm256_castsi256_si128(sad_l1),
            _mm256_extracti128_si256(sad_l1, 1),
        );
        let s_l1_u64 =
            _mm_cvtsi128_si64(_mm_add_epi64(l1_128, _mm_unpackhi_epi64(l1_128, l1_128))) as u64;
        if s_l1_u64 == 0 {
            return if 1.0 < floor_coh { -2.0 } else { 1.0 };
        }
        if floor_coh > -0.95 {
            let s_l1 = s_l1_u64 as f32;
            if (1.052_f32 - s_l1 * (1.0 / 8192.0)) < floor_coh
                || (1.215_f32 - s_l1 * (1.0 / 4096.0)) < floor_coh
            {
                return -2.0;
            }
        }
        let lo128 = _mm256_castsi256_si128(vd0);
        let splat_b0 = _mm256_broadcastb_epi8(lo128);
        let diff_b0 = _mm256_or_si256(
            _mm256_xor_si256(vd0, splat_b0),
            _mm256_xor_si256(vd1, splat_b0),
        );
        if _mm256_testz_si256(diff_b0, diff_b0) != 0 {
            let b0 = (_mm_cvtsi128_si32(lo128) as u32) & 0xFF;
            let coh = Q7_FULL_COS_REF[b0 as usize];
            return if coh < floor_coh { -2.0 } else { coh };
        }
        let splat_q0 = _mm256_broadcastq_epi64(lo128);
        let diff_q0 = _mm256_or_si256(
            _mm256_xor_si256(vd0, splat_q0),
            _mm256_xor_si256(vd1, splat_q0),
        );
        if _mm256_testz_si256(diff_q0, diff_q0) != 0 {
            let d0 = _mm_cvtsi128_si64(lo128) as u64;
            let coh = branchless_clamp_f32(
                (sum_cos_q14_from_diff_word(d0) as f32) * INV_8_Q14_F32,
                -1.0,
                1.0,
            );
            return if coh < floor_coh { -2.0 } else { coh };
        }
        avx2_coherence_from_raw_abs_vec64(raw_abs0, raw_abs1)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_coherence_from_diff_vec64(
    vd0: core::arch::x86_64::__m256i,
    vd1: core::arch::x86_64::__m256i,
) -> f32 {
    // SAFETY: Guarded by `target_feature = "avx2"`. Fast-paths byte-uniform shifts in 1 LUT load
    // (`0` GPR `imul`) via `vpbroadcastb + vptest` and period-8 word-uniform shifts via
    // `vpbroadcastq + vptest` before running the 16-knot trigonometric pipeline.
    unsafe {
        use core::arch::x86_64::{
            _mm256_abs_epi8, _mm256_broadcastb_epi8, _mm256_broadcastq_epi64,
            _mm256_castsi256_si128, _mm256_or_si256, _mm256_testz_si256, _mm256_xor_si256,
            _mm_cvtsi128_si32, _mm_cvtsi128_si64,
        };
        let lo128 = _mm256_castsi256_si128(vd0);
        let splat_b0 = _mm256_broadcastb_epi8(lo128);
        let diff_b0 = _mm256_or_si256(
            _mm256_xor_si256(vd0, splat_b0),
            _mm256_xor_si256(vd1, splat_b0),
        );
        if _mm256_testz_si256(diff_b0, diff_b0) != 0 {
            let b0 = (_mm_cvtsi128_si32(lo128) as u32) & 0xFF;
            return Q7_FULL_COS_REF[b0 as usize];
        }
        let splat_q0 = _mm256_broadcastq_epi64(lo128);
        let diff_q0 = _mm256_or_si256(
            _mm256_xor_si256(vd0, splat_q0),
            _mm256_xor_si256(vd1, splat_q0),
        );
        if _mm256_testz_si256(diff_q0, diff_q0) != 0 {
            let d0 = _mm_cvtsi128_si64(lo128) as u64;
            return branchless_clamp_f32(
                (sum_cos_q14_from_diff_word(d0) as f32) * INV_8_Q14_F32,
                -1.0,
                1.0,
            );
        }
        avx2_coherence_from_raw_abs_vec64(_mm256_abs_epi8(vd0), _mm256_abs_epi8(vd1))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_coherence_from_raw_abs_vec64(
    raw_abs0: core::arch::x86_64::__m256i,
    raw_abs1: core::arch::x86_64::__m256i,
) -> f32 {
    // SAFETY: Guarded by `target_feature = "avx2"`. Evaluates 64-head cosine sum in registers
    // via 16-knot `vpshufb` + `vpsadbw` + saturation-free `vpmaddubsw` trigonometric angle-addition.
    unsafe {
        use core::arch::x86_64::{
            _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256,
            _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_madd_epi16,
            _mm256_maddubs_epi16, _mm256_min_epu8, _mm256_sad_epu8, _mm256_set1_epi16,
            _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
            _mm256_slli_epi64, _mm256_srli_epi16, _mm_add_epi32, _mm_add_epi64,
            _mm_cvtsi128_si32, _mm_cvtsi128_si64, _mm_shuffle_epi32, _mm_unpackhi_epi64,
        };
        let max_127 = _mm256_set1_epi8(127);
        let abs0 = _mm256_min_epu8(raw_abs0, max_127);
        let abs1 = _mm256_min_epu8(raw_abs1, max_127);

        let mask_15 = _mm256_set1_epi8(0x0F);
        let h0 = _mm256_and_si256(_mm256_srli_epi16(abs0, 3), mask_15);
        let h1 = _mm256_and_si256(_mm256_srli_epi16(abs1, 3), mask_15);

        let lut_c0_lo = _mm256_setr_epi8(
            0, -59, 33, 55, 65, -114, 126, 124, 0, -124, -126, 114, -65, -55, -33, 59,
            0, -59, 33, 55, 65, -114, 126, 124, 0, -124, -126, 114, -65, -55, -33, 59,
        );
        let zero = _mm256_setzero_si256();
        let sad_lo = _mm256_add_epi64(
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_lo, h0), zero),
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_lo, h1), zero),
        );
        let lut_c0_hi = _mm256_setr_epi8(
            -128, 126, 123, 117, 109, 99, 88, 76, 64, 51, 39, 28, 18, 10, 4, 1,
            -128, 126, 123, 117, 109, 99, 88, 76, 64, 51, 39, 28, 18, 10, 4, 1,
        );
        let sad_hi = _mm256_add_epi64(
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_hi, h0), zero),
            _mm256_sad_epu8(_mm256_shuffle_epi8(lut_c0_hi, h1), zero),
        );
        let sad_tot = _mm256_add_epi64(sad_lo, _mm256_slli_epi64(sad_hi, 8));
        let sad_128 = _mm_add_epi64(
            _mm256_castsi256_si128(sad_tot),
            _mm256_extracti128_si256(sad_tot, 1),
        );
        let base_sum = _mm_cvtsi128_si64(_mm_add_epi64(sad_128, _mm_unpackhi_epi64(sad_128, sad_128)))
            as i32
            - (64 * 16384);

        let mask_7 = _mm256_set1_epi8(0x07);
        let r0 = _mm256_and_si256(abs0, mask_7);
        let r1 = _mm256_and_si256(abs1, mask_7);

        let lut_nsin_h = _mm256_setr_epi8(
            0, -25, -49, -71, -90, -106, -117, -125, -127, -125, -117, -106, -90, -71, -49, -25,
            0, -25, -49, -71, -90, -106, -117, -125, -127, -125, -117, -106, -90, -71, -49, -25,
        );
        let lut_sin_r = _mm256_setr_epi8(
            0, 18, 36, 54, 72, 90, 108, 126, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 18, 36, 54, 72, 90, 108, 126, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        let sin0 = _mm256_maddubs_epi16(
            _mm256_shuffle_epi8(lut_sin_r, r0),
            _mm256_shuffle_epi8(lut_nsin_h, h0),
        );
        let sin1 = _mm256_maddubs_epi16(
            _mm256_shuffle_epi8(lut_sin_r, r1),
            _mm256_shuffle_epi8(lut_nsin_h, h1),
        );

        let lut_ncos_h = _mm256_setr_epi8(
            -127, -125, -117, -106, -90, -71, -49, -25, 0, 25, 49, 71, 90, 106, 117, 125,
            -127, -125, -117, -106, -90, -71, -49, -25, 0, 25, 49, 71, 90, 106, 117, 125,
        );
        let lut_omc_r = _mm256_setr_epi8(
            0, 0, 1, 2, 3, 6, 8, 11, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 2, 3, 6, 8, 11, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        let corr0 = _mm256_add_epi16(
            sin0,
            _mm256_maddubs_epi16(
                _mm256_shuffle_epi8(lut_omc_r, r0),
                _mm256_shuffle_epi8(lut_ncos_h, h0),
            ),
        );
        let corr1 = _mm256_add_epi16(
            sin1,
            _mm256_maddubs_epi16(
                _mm256_shuffle_epi8(lut_omc_r, r1),
                _mm256_shuffle_epi8(lut_ncos_h, h1),
            ),
        );
        let ones16 = _mm256_set1_epi16(1);
        let corr32 = _mm256_add_epi32(
            _mm256_madd_epi16(corr0, ones16),
            _mm256_madd_epi16(corr1, ones16),
        );
        let c128 = _mm_add_epi32(
            _mm256_castsi256_si128(corr32),
            _mm256_extracti128_si256(corr32, 1),
        );
        let c64 = _mm_add_epi32(c128, _mm_shuffle_epi32(c128, 0x4E));
        let c32 = _mm_add_epi32(c64, _mm_shuffle_epi32(c64, 0xB1));
        let corr_sum = _mm_cvtsi128_si32(c32);

        let total_f32 = (base_sum as f32) + 0.175_520_92_f32 * (corr_sum as f32);
        branchless_clamp_f32(total_f32 * INV_64_Q14_F32, -1.0, 1.0)
    }
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1.0, 1.0]$
/// from a single period-8 uniform circular-difference word `d0` (`d0 == d1 == ... == d7`) in
/// 1 LUT load when byte-uniform or 8 L1D `i16` lookups when heterogeneous (`0` SIMD instructions).
#[inline(always)]
#[must_use]
pub(crate) const fn coherence_from_uniform_diff_word_x8(d0: u64) -> f32 {
    if d0 == 0 {
        return 1.0;
    }
    if d0 == (d0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
        return Q7_FULL_COS_REF[(d0 & 0xFF) as usize];
    }
    branchless_clamp_f32(
        (sum_cos_q14_from_diff_word(d0) as f32) * INV_8_Q14_F32,
        -1.0,
        1.0,
    )
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// when both 64-head profiles are period-8 word-uniform (`a == [wa; 8]` and `b == [wb; 8]`),
/// operating strictly on the two 64-bit representative words `(wa, wb)` without touching the
/// 64-byte phase arrays or issuing vector instructions.
#[inline(always)]
#[must_use]
pub(crate) const fn coherence_from_uniform_words_x8(wa: u64, wb: u64) -> f32 {
    if wa == wb {
        return 1.0;
    }
    coherence_from_uniform_diff_word_x8(swar_q7_circular_diff_x8(wa, wb))
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1.0, 1.0]$
/// directly from 8 SWAR Q7 circular difference words `(d0..=d7)`.
#[inline(always)]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn coherence_from_diff_words_64(
    d0: u64,
    d1: u64,
    d2: u64,
    d3: u64,
    d4: u64,
    d5: u64,
    d6: u64,
    d7: u64,
) -> f32 {
    if (((d0 ^ d1) | (d0 ^ d2)) | ((d0 ^ d3) | (d0 ^ d4)) | ((d0 ^ d5) | (d0 ^ d6)) | (d0 ^ d7)) == 0 {
        return coherence_from_uniform_diff_word_x8(d0);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`. Value-level `[u64; 8] -> [__m256i; 2]`
        // transmute avoids `_mm256_setr_epi64x` 4-to-1 Store-to-Load Forwarding stalls.
        return unsafe {
            let (vd0, vd1) = avx2_u64x8_to_vec64(d0, d1, d2, d3, d4, d5, d6, d7);
            avx2_coherence_from_diff_vec64(vd0, vd1)
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let s01 = sum_cos_q14_from_diff_word(d0) + sum_cos_q14_from_diff_word(d1);
        let s23 = sum_cos_q14_from_diff_word(d2) + sum_cos_q14_from_diff_word(d3);
        let s45 = sum_cos_q14_from_diff_word(d4) + sum_cos_q14_from_diff_word(d5);
        let s67 = sum_cos_q14_from_diff_word(d6) + sum_cos_q14_from_diff_word(d7);
        let total_q14 = (s01 + s23) + (s45 + s67);
        branchless_clamp_f32((total_q14 as f32) * INV_64_Q14_F32, -1.0, 1.0)
    }
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// from two 8-word SWAR arrays (`a` and `b`) using pre-evaluated uniformity flags
/// (`a_is_uniform`, `b_is_uniform`), eliminating redundant 8-word XOR reductions inside multi-anchor
/// and multi-centroid loops.
#[inline(always)]
#[must_use]
pub fn coherence_from_word_arrays_with_both_uniform_flags_64(
    a: &[u64; 8],
    a_is_uniform: bool,
    b: &[u64; 8],
    b_is_uniform: bool,
) -> f32 {
    if a_is_uniform & b_is_uniform {
        return coherence_from_uniform_words_x8(a[0], b[0]);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`. Value-level `[u64; 8] -> [__m256i; 2]`
        // bitcast avoids 4-to-1 GPR-to-YMM Store-to-Load Forwarding stalls on stack-resident word arrays.
        return unsafe {
            use core::arch::x86_64::_mm256_sub_epi8;
            let (va0, va1) = avx2_words64_to_vec64_with_uniform(a, a_is_uniform);
            let (vb0, vb1) = avx2_words64_to_vec64_with_uniform(b, b_is_uniform);
            avx2_coherence_from_diff_vec64(_mm256_sub_epi8(va0, vb0), _mm256_sub_epi8(va1, vb1))
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        coherence_from_diff_words_64(
            swar_q7_circular_diff_x8(a[0], b[0]),
            swar_q7_circular_diff_x8(a[1], b[1]),
            swar_q7_circular_diff_x8(a[2], b[2]),
            swar_q7_circular_diff_x8(a[3], b[3]),
            swar_q7_circular_diff_x8(a[4], b[4]),
            swar_q7_circular_diff_x8(a[5], b[5]),
            swar_q7_circular_diff_x8(a[6], b[6]),
            swar_q7_circular_diff_x8(a[7], b[7]),
        )
    }
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// when $\mathcal{C}_{\mathrm{phase}}(a, b) \ge \text{floor\_coh}$ is possible, or returns `-2.0`
/// in 5 AVX2 instructions (`vpabsb + vpsadbw`) when the exact Lagrangian $L_1$ upper bound
/// $\mathcal{C}_{64}(a, b) \le 1.21052 - \frac{1}{4096}\sum_{h=0}^{63}|d_h|$ proves $\mathcal{C}_{64}(a, b) < \text{floor\_coh}$.
#[inline(always)]
#[must_use]
pub fn coherence_above_floor_from_word_arrays_64(
    a: &[u64; 8],
    a_is_uniform: bool,
    b: &[u64; 8],
    b_is_uniform: bool,
    floor_coh: f32,
) -> f32 {
    if a_is_uniform & b_is_uniform {
        return coherence_from_uniform_words_x8(a[0], b[0]);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`.
        return unsafe {
            use core::arch::x86_64::_mm256_sub_epi8;
            let (va0, va1) = avx2_words64_to_vec64_with_uniform(a, a_is_uniform);
            let (vb0, vb1) = avx2_words64_to_vec64_with_uniform(b, b_is_uniform);
            avx2_coherence_from_diff_vec64_with_floor(
                _mm256_sub_epi8(va0, vb0),
                _mm256_sub_epi8(va1, vb1),
                floor_coh,
            )
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let d0 = swar_q7_circular_diff_x8(a[0], b[0]);
        let d1 = swar_q7_circular_diff_x8(a[1], b[1]);
        let s01 = sum_cos_q14_from_diff_word(d0) + sum_cos_q14_from_diff_word(d1);
        if floor_coh > -0.95 && ((s01 as f32 + 48.05 * 16384.0) * INV_64_Q14_F32) < floor_coh {
            return -2.0;
        }
        let s23 = sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[2], b[2]))
            + sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[3], b[3]));
        let s45 = sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[4], b[4]))
            + sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[5], b[5]));
        let s67 = sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[6], b[6]))
            + sum_cos_q14_from_diff_word(swar_q7_circular_diff_x8(a[7], b[7]));
        let total_q14 = (s01 + s23) + (s45 + s67);
        branchless_clamp_f32((total_q14 as f32) * INV_64_Q14_F32, -1.0, 1.0)
    }
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// when `a_is_uniform = is_word_array_uniform_64(a)` has already been hoisted outside a loop.
#[inline(always)]
#[must_use]
pub fn coherence_from_word_arrays_with_uniform_flag_64(
    a: &[u64; 8],
    a_is_uniform: bool,
    b: &[u64; 8],
) -> f32 {
    let b_is_uniform = if a_is_uniform {
        is_word_array_uniform_64(b)
    } else {
        false
    };
    coherence_from_word_arrays_with_both_uniform_flags_64(a, a_is_uniform, b, b_is_uniform)
}

/// Evaluates normalized 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// from two 8-word SWAR arrays (`a` and `b`) in strictly constant time (`0` branches), fast-pathing
/// identical (`1.0`), byte-uniform (`1` LUT load), and word-uniform (`8` L1D LUT loads) arrays.
#[inline(always)]
#[must_use]
pub fn coherence_from_word_arrays_64(a: &[u64; 8], b: &[u64; 8]) -> f32 {
    coherence_from_word_arrays_with_uniform_flag_64(a, is_word_array_uniform_64(a), b)
}


/// Evaluates the mean circular phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$
/// using 8-word SWAR circular differences and register-shifted Q14 cosine lookups.
#[inline(always)]
pub fn branchless_q7_coherence_64(
    phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `phases_a` and `phases_b` are 64-byte arrays and AVX2 is enabled.
        // Direct `_mm256_loadu_si256` avoids 64-bit GPR stack copies and 4-to-1 STLF stalls.
        return unsafe {
            use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
            let pa = phases_a.as_ptr().cast::<__m256i>();
            let pb = phases_b.as_ptr().cast::<__m256i>();
            let va0 = _mm256_loadu_si256(pa);
            let va1 = _mm256_loadu_si256(pa.add(1));
            let vb0 = _mm256_loadu_si256(pb);
            let vb1 = _mm256_loadu_si256(pb.add(1));
            avx2_coherence_from_diff_vec64(_mm256_sub_epi8(va0, vb0), _mm256_sub_epi8(va1, vb1))
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        coherence_from_word_arrays_64(&load_q7_words_64(phases_a), &load_q7_words_64(phases_b))
    }
}

/// Evaluates **Zero-Hyperparameter Phase-Proximity Gravity** on the unified phase manifold,
/// loading `trail_phases` and `anchor_phases` once into 64-bit registers (`0` redundant reloads)
/// and bypassing the 64-head Q14 LUT reduction when either profile is zero.
#[inline(always)]
pub fn branchless_q7_resonant_pull_64(
    trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    base_energy: f32,
) -> (f32, f32) {
    let clean_energy = branchless_flush_subnormal_f32(branchless_relu_f32(base_energy));
    if clean_energy <= 1.0e-12 {
        return (0.0, 0.0);
    }
    let t = load_q7_words_64(trail_phases);
    let b = load_q7_words_64(anchor_phases);
    let both_nonzero =
        ((((t[0] | t[1]) | (t[2] | t[3])) | ((t[4] | t[5]) | (t[6] | t[7]))) != 0)
            & ((((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0);
    let eff_coh = if both_nonzero {
        coherence_from_word_arrays_64(&t, &b)
    } else {
        0.0
    };
    let proximity_gain = branchless_clamp_f32(1.0 + eff_coh, 0.25, 2.0);
    (
        branchless_flush_subnormal_f32(clean_energy * proximity_gain),
        eff_coh,
    )
}

/// Register-resident Unified-Field folder returning `([u64; 8], total_pull, mean_coherence, dominant_idx, active_count)`.
///
/// Includes a zero-division, zero-SLERP single-anchor fast path (`count == 1 && init_w <= 1.0e-12`)
/// and guards 64-head Q14 cosine LUT lookups behind `trail_nonzero & anchor_nonzero`.
/// Order-invariant (commutative) folding of an arbitrary slice of anchors
/// (`anchor_phases`, `anchor_base_energies`) directly in 8 `u64` SWAR / AVX2 registers,
/// returning `(folded_words, total_weight, mean_coherence, dominant_anchor_idx, active_anchor_count, any_resonant, any_liminal)`.
#[inline]
pub(crate) fn branchless_q7_unified_field_fold_words_with_flags_64(
    trail_words: &[u64; 8],
    trail_init_weight: f32,
    default_dominant_idx: u32,
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
) -> ([u64; 8], f32, f32, u32, u32, bool, bool) {
    let count = branchless_select_usize(
        anchor_phases.len() < anchor_base_energies.len(),
        anchor_phases.len(),
        anchor_base_energies.len(),
    );
    let init_w = branchless_flush_subnormal_f32(branchless_relu_f32(trail_init_weight));
    let t = *trail_words;
    if count == 0 {
        return (t, init_w, 0.0, default_dominant_idx, 0, false, false);
    }
    let trail_nonzero = (((t[0] | t[1]) | (t[2] | t[3])) | ((t[4] | t[5]) | (t[6] | t[7]))) != 0;

    if count == 1 {
        let raw_e = anchor_base_energies[0];
        let clean_energy = if branchless_is_finite_f32(raw_e) {
            branchless_flush_subnormal_f32(branchless_relu_f32(raw_e))
        } else {
            0.0
        };
        if clean_energy <= 1.0e-12 {
            return (t, init_w, 0.0, default_dominant_idx, 0, false, false);
        }
        let b = load_q7_words_64(&anchor_phases[0]);
        let anchor_nonzero =
            (((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0;
        let coh = if trail_nonzero & anchor_nonzero {
            coherence_from_word_arrays_64(&t, &b)
        } else {
            0.0
        };
        let res_pull =
            branchless_flush_subnormal_f32(clean_energy * branchless_clamp_f32(1.0 + coh, 0.25, 2.0));
        let is_active = res_pull > 1.0e-12;
        if !is_active {
            return (t, init_w, 0.0, default_dominant_idx, 0, false, false);
        }
        let total_w = init_w + res_pull;
        let out_w = if init_w <= 1.0e-12 {
            b
        } else {
            let ratio_f32 = branchless_clamp_f32(
                (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                0.0,
                128.0,
            );
            let s = (ratio_f32 as u8) as u64;
            let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            swar_slerp_words_with_bias_64(&t, &b, s, bias_restore)
        };
        return (
            out_w,
            branchless_flush_subnormal_f32(total_w),
            coh,
            0,
            1,
            coh > 0.25,
            coh < -0.10,
        );
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        return unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x, _mm256_sub_epi8,
                _mm256_testz_si256,
            };
            let t_uni = is_word_array_uniform_64(&t);
            let t_w0 = t[0];
            let t_b0 = (t_w0 & 0xFF) as u8;
            let t_is_byte_uniform =
                t_uni & (t_w0 == (t_b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64));
            let mut vt_ready = !t_uni;
            let (mut vt0, mut vt1) = if vt_ready {
                avx2_words64_to_vec64_with_uniform(&t, false)
            } else {
                let z = _mm256_set1_epi64x(t_w0 as i64);
                (z, z)
            };
            let mut r_uni = t_uni;
            let mut r_w0 = t_w0;
            let mut vr0 = vt0;
            let mut vr1 = vt1;
            let mut running_weight = init_w;
            let mut anchor_pull_sum = 0.0_f32;
            let mut weighted_coh_sum = 0.0_f32;
            let mut best_pull = 0.0_f32;
            let mut dominant_idx = default_dominant_idx;
            let mut active_count = 0u32;
            let mut any_resonant = false;
            let mut any_liminal = false;

            let mut idx = 0usize;
            while idx < count {
                let raw_e = anchor_base_energies[idx];
                let clean_energy = if branchless_is_finite_f32(raw_e) {
                    branchless_flush_subnormal_f32(branchless_relu_f32(raw_e))
                } else {
                    0.0
                };
                if clean_energy > 1.0e-12 {
                    let check_b_uni = r_uni & (t_uni | !trail_nonzero);
                    let mut b = [0u64; 8];
                    let b_uni = if check_b_uni {
                        b = load_q7_words_64(&anchor_phases[idx]);
                        is_word_array_uniform_64(&b)
                    } else {
                        false
                    };
                    if b_uni & r_uni & (t_uni | !trail_nonzero) {
                        let bw0 = b[0];
                        let coh = if !(trail_nonzero & (bw0 != 0)) {
                            0.0
                        } else if bw0 == t_w0 {
                            1.0
                        } else if t_is_byte_uniform
                            && (bw0 == ((bw0 & 0xFF) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64))
                        {
                            Q7_FULL_COS_REF[t_b0.wrapping_sub(bw0 as u8) as usize]
                        } else {
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &t, true, &b, true,
                            )
                        };
                        let res_pull = branchless_flush_subnormal_f32(
                            clean_energy * branchless_clamp_f32(1.0 + coh, 0.25, 2.0),
                        );
                        let is_active = res_pull > 1.0e-12;
                        let is_better = res_pull > best_pull;
                        let total_w = running_weight + res_pull;

                        if running_weight <= 1.0e-12 {
                            if is_active {
                                r_w0 = bw0;
                            }
                        } else if is_active {
                            let ratio_f32 = branchless_clamp_f32(
                                (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                0.0,
                                128.0,
                            );
                            let s = (ratio_f32 as u8) as u64;
                            if s >= 128 {
                                r_w0 = bw0;
                            } else if s > 0 && r_w0 != bw0 {
                                let rb0 = (r_w0 & 0xFF) as u8;
                                let bb0 = (bw0 & 0xFF) as u8;
                                if r_w0 == (rb0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64)
                                    && bw0 == (bb0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64)
                                {
                                    let b_lerp = branchless_q7_circular_lerp(
                                        rb0 as gcso_q7_t,
                                        bb0 as gcso_q7_t,
                                        s as u8,
                                    );
                                    r_w0 = ((b_lerp as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                } else {
                                    let bias_restore = (128u64.wrapping_sub(s))
                                        .wrapping_mul(SWAR_SPLAT_BYTE_64);
                                    r_w0 = swar_slerp_word_unchecked_x8(
                                        r_w0,
                                        bw0,
                                        s,
                                        bias_restore,
                                    );
                                }
                            }
                        }
                        running_weight = total_w;
                        anchor_pull_sum += branchless_select_f32(is_active, res_pull, 0.0);
                        weighted_coh_sum += branchless_select_f32(is_active, res_pull * coh, 0.0);
                        best_pull = branchless_select_f32(is_better, res_pull, best_pull);
                        dominant_idx =
                            branchless_select_u32(is_better, idx as u32, dominant_idx);
                        active_count = active_count.wrapping_add(u32::from(is_active));
                        any_resonant |= is_active & (coh > 0.25);
                        any_liminal |= is_active & (coh < -0.10);
                    } else {
                        let b_ptr = anchor_phases[idx].as_ptr().cast::<__m256i>();
                        let vb0 = _mm256_loadu_si256(b_ptr);
                        let vb1 = _mm256_loadu_si256(b_ptr.add(1));
                        let coh = if !trail_nonzero {
                            0.0
                        } else if t_uni & b_uni {
                            let bw0 = b[0];
                            if bw0 == 0 {
                                0.0
                            } else if bw0 == t_w0 {
                                1.0
                            } else {
                                coherence_from_word_arrays_with_both_uniform_flags_64(
                                    &t, true, &b, true,
                                )
                            }
                        } else {
                            if !vt_ready {
                                let tz = _mm256_set1_epi64x(t_w0 as i64);
                                vt0 = tz;
                                vt1 = tz;
                                vt_ready = true;
                            }
                            let b_or = _mm256_or_si256(vb0, vb1);
                            if _mm256_testz_si256(b_or, b_or) != 0 {
                                0.0
                            } else {
                                avx2_coherence_from_diff_vec64(
                                    _mm256_sub_epi8(vt0, vb0),
                                    _mm256_sub_epi8(vt1, vb1),
                                )
                            }
                        };
                        let res_pull = branchless_flush_subnormal_f32(
                            clean_energy * branchless_clamp_f32(1.0 + coh, 0.25, 2.0),
                        );
                        let is_active = res_pull > 1.0e-12;
                        let is_better = res_pull > best_pull;
                        let total_w = running_weight + res_pull;

                        if running_weight <= 1.0e-12 {
                            if is_active {
                                if b_uni {
                                    r_w0 = b[0];
                                    r_uni = true;
                                } else {
                                    vr0 = vb0;
                                    vr1 = vb1;
                                    r_uni = false;
                                }
                            }
                        } else if is_active {
                            if r_uni {
                                let rz = _mm256_set1_epi64x(r_w0 as i64);
                                vr0 = rz;
                                vr1 = rz;
                                r_uni = false;
                            }
                            let ratio_f32 = branchless_clamp_f32(
                                (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                0.0,
                                128.0,
                            );
                            let s = (ratio_f32 as u8) as u64;
                            if s >= 128 {
                                if b_uni {
                                    r_w0 = b[0];
                                    r_uni = true;
                                } else {
                                    vr0 = vb0;
                                    vr1 = vb1;
                                }
                            } else if s > 0 {
                                let diff0 = _mm256_sub_epi8(vb0, vr0);
                                let diff1 = _mm256_sub_epi8(vb1, vr1);
                                let diff_or = _mm256_or_si256(diff0, diff1);
                                if _mm256_testz_si256(diff_or, diff_or) == 0 {
                                    (vr0, vr1) = avx2_slerp_from_diff_vec64(vr0, vr1, diff0, diff1, s);
                                }
                            }
                        }

                        running_weight = total_w;
                        anchor_pull_sum += branchless_select_f32(is_active, res_pull, 0.0);
                        weighted_coh_sum += branchless_select_f32(is_active, res_pull * coh, 0.0);
                        best_pull = branchless_select_f32(is_better, res_pull, best_pull);
                        dominant_idx =
                            branchless_select_u32(is_better, idx as u32, dominant_idx);
                        active_count = active_count.wrapping_add(u32::from(is_active));
                        any_resonant |= is_active & (coh > 0.25);
                        any_liminal |= is_active & (coh < -0.10);
                    }
                }
                idx += 1;
            }

            let has_pull = (active_count > 0) & (anchor_pull_sum > 1.0e-12);
            let inv_pull =
                fast_inv_pos_normal_f32(branchless_select_f32(has_pull, anchor_pull_sum, 1.0));
            let mean_coh = branchless_select_f32(
                has_pull,
                branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
                0.0,
            );
            let out_w = if r_uni {
                [r_w0; 8]
            } else {
                avx2_vec64_to_words64(vr0, vr1)
            };
            (
                out_w,
                branchless_flush_subnormal_f32(running_weight),
                mean_coh,
                dominant_idx,
                active_count,
                any_resonant,
                any_liminal,
            )
        };
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let trail_is_uniform = is_word_array_uniform_64(&t);
        let mut r = t;
        let mut r_is_uniform = trail_is_uniform;
        let mut running_weight = init_w;
        let mut anchor_pull_sum = 0.0_f32;
        let mut weighted_coh_sum = 0.0_f32;
        let mut best_pull = 0.0_f32;
        let mut dominant_idx = default_dominant_idx;
        let mut active_count = 0u32;
        let mut any_resonant = false;
        let mut any_liminal = false;

        let mut idx = 0usize;
        while idx < count {
            let raw_e = anchor_base_energies[idx];
            let clean_energy = if branchless_is_finite_f32(raw_e) {
                branchless_flush_subnormal_f32(branchless_relu_f32(raw_e))
            } else {
                0.0
            };
            if clean_energy > 1.0e-12 {
                let b = load_q7_words_64(&anchor_phases[idx]);
                let b_is_uniform = is_word_array_uniform_64(&b);
                let anchor_nonzero = if b_is_uniform {
                    b[0] != 0
                } else {
                    (((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0
                };
                let coh = if trail_nonzero & anchor_nonzero {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &t,
                        trail_is_uniform,
                        &b,
                        b_is_uniform,
                    )
                } else {
                    0.0
                };
                let res_pull = branchless_flush_subnormal_f32(
                    clean_energy * branchless_clamp_f32(1.0 + coh, 0.25, 2.0),
                );
                let is_active = res_pull > 1.0e-12;
                let is_better = res_pull > best_pull;
                let total_w = running_weight + res_pull;

                if running_weight <= 1.0e-12 {
                    if is_active {
                        r = b;
                        r_is_uniform = b_is_uniform;
                    }
                } else if is_active {
                    let ratio_f32 = branchless_clamp_f32(
                        (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                        0.0,
                        128.0,
                    );
                    let s = (ratio_f32 as u8) as u64;
                    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let (next_r, next_uni) = swar_slerp_words_with_uniform_flag_64(
                        &r,
                        r_is_uniform,
                        &b,
                        b_is_uniform,
                        s,
                        bias_restore,
                    );
                    r = next_r;
                    r_is_uniform = next_uni;
                }

                running_weight = total_w;
                anchor_pull_sum += branchless_select_f32(is_active, res_pull, 0.0);
                weighted_coh_sum += branchless_select_f32(is_active, res_pull * coh, 0.0);
                best_pull = branchless_select_f32(is_better, res_pull, best_pull);
                dominant_idx = branchless_select_u32(is_better, idx as u32, dominant_idx);
                active_count = active_count.wrapping_add(u32::from(is_active));
                any_resonant |= is_active & (coh > 0.25);
                any_liminal |= is_active & (coh < -0.10);
            }
            idx += 1;
        }

        let has_pull = (active_count > 0) & (anchor_pull_sum > 1.0e-12);
        let inv_pull = fast_inv_pos_normal_f32(branchless_select_f32(has_pull, anchor_pull_sum, 1.0));
        let mean_coh = branchless_select_f32(
            has_pull,
            branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
            0.0,
        );
        (
            r,
            branchless_flush_subnormal_f32(running_weight),
            mean_coh,
            dominant_idx,
            active_count,
            any_resonant,
            any_liminal,
        )
    }
}

/// Register-resident core for [`branchless_q7_unified_field_fold_with_init_weight_64`] that folds
/// `anchor_phases` directly in 8 `u64` SWAR registers (`trail_words`) without intermediate memory
/// stores or reloads, returning `(folded_words_64, total_weight, mean_coherence, dominant_anchor_idx, active_anchor_count)`.
#[inline]
#[must_use]
pub fn branchless_q7_unified_field_fold_words_with_init_weight_64(
    trail_words: &[u64; 8],
    trail_init_weight: f32,
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
) -> ([u64; 8], f32, f32, u32, u32) {
    let count = branchless_select_usize(
        anchor_phases.len() < anchor_base_energies.len(),
        anchor_phases.len(),
        anchor_base_energies.len(),
    );
    let init_w = branchless_flush_subnormal_f32(branchless_relu_f32(trail_init_weight));
    let t = *trail_words;
    if count == 0 {
        return (t, init_w, 0.0, 0, 0);
    }
    let trail_nonzero = (((t[0] | t[1]) | (t[2] | t[3])) | ((t[4] | t[5]) | (t[6] | t[7]))) != 0;

    if count == 1 {
        let clean_energy =
            branchless_flush_subnormal_f32(branchless_relu_f32(anchor_base_energies[0]));
        if clean_energy <= 1.0e-12 {
            return (t, init_w, 0.0, 0, 0);
        }
        let b = load_q7_words_64(&anchor_phases[0]);
        let anchor_nonzero =
            (((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0;
        let coh = if trail_nonzero & anchor_nonzero {
            coherence_from_word_arrays_64(&t, &b)
        } else {
            0.0
        };
        let res_pull =
            branchless_flush_subnormal_f32(clean_energy * branchless_clamp_f32(1.0 + coh, 0.25, 2.0));
        let total_w = init_w + res_pull;
        let out_w = if init_w <= 1.0e-12 {
            b
        } else {
            let ratio_f32 = branchless_clamp_f32(
                (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                0.0,
                128.0,
            );
            let s = (ratio_f32 as u8) as u64;
            let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            swar_slerp_words_with_bias_64(&t, &b, s, bias_restore)
        };
        return (out_w, branchless_flush_subnormal_f32(total_w), coh, 0, 1);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`. Fast-paths word-uniform anchors directly in
        // a single 64-bit GPR (`r_w0`) with zero YMM loads or `vpextrq` extractions, and seamlessly
        // transitions to YMM-resident `vr0, vr1` (`avx2_slerp_vec64`) if any anchor is non-uniform.
        return unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x, _mm256_sub_epi8,
                _mm256_testz_si256,
            };
            let t_uni = is_word_array_uniform_64(&t);
            let t_w0 = t[0];
            let t_b0 = (t_w0 & 0xFF) as u8;
            let t_is_byte_uniform =
                t_uni & (t_w0 == (t_b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64));
            let mut vt_ready = !t_uni;
            let (mut vt0, mut vt1) = if vt_ready {
                avx2_words64_to_vec64_with_uniform(&t, false)
            } else {
                let z = _mm256_set1_epi64x(t_w0 as i64);
                (z, z)
            };
            let mut r_uni = t_uni;
            let mut r_w0 = t_w0;
            let mut vr0 = vt0;
            let mut vr1 = vt1;
            let mut running_weight = init_w;
            let mut anchor_pull_sum = 0.0_f32;
            let mut weighted_coh_sum = 0.0_f32;
            let mut best_pull = 0.0_f32;
            let mut dominant_idx = 0u32;
            let mut active_count = 0u32;

            let mut idx = 0usize;
            while idx < count {
                let clean_energy =
                    branchless_flush_subnormal_f32(branchless_relu_f32(anchor_base_energies[idx]));
                if clean_energy > 1.0e-12 {
                    let check_b_uni = r_uni & (t_uni | !trail_nonzero);
                    let mut b = [0u64; 8];
                    let b_uni = if check_b_uni {
                        b = load_q7_words_64(&anchor_phases[idx]);
                        is_word_array_uniform_64(&b)
                    } else {
                        false
                    };
                    if b_uni & r_uni & (t_uni | !trail_nonzero) {
                        let bw0 = b[0];
                        let coh = if !(trail_nonzero & (bw0 != 0)) {
                            0.0
                        } else if bw0 == t_w0 {
                            1.0
                        } else if t_is_byte_uniform
                            && (bw0 == ((bw0 & 0xFF) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64))
                        {
                            Q7_FULL_COS_REF[t_b0.wrapping_sub(bw0 as u8) as usize]
                        } else {
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &t, true, &b, true,
                            )
                        };
                        let proximity_gain = branchless_clamp_f32(1.0 + coh, 0.25, 2.0);
                        let res_pull =
                            branchless_flush_subnormal_f32(clean_energy * proximity_gain);
                        let is_better = res_pull > best_pull;
                        let total_w = running_weight + res_pull;
                        if running_weight <= 1.0e-12 {
                            r_w0 = bw0;
                        } else if r_w0 != bw0 {
                            let ratio_f32 = branchless_clamp_f32(
                                (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                0.0,
                                128.0,
                            );
                            let s = (ratio_f32 as u8) as u64;
                            if s >= 128 {
                                r_w0 = bw0;
                            } else if s > 0 {
                                let bias_restore =
                                    (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                r_w0 = swar_slerp_word_unchecked_x8(r_w0, bw0, s, bias_restore);
                            }
                        }
                        running_weight = total_w;
                        anchor_pull_sum += res_pull;
                        weighted_coh_sum += res_pull * coh;
                        best_pull = branchless_select_f32(is_better, res_pull, best_pull);
                        dominant_idx = branchless_select_u32(is_better, idx as u32, dominant_idx);
                        active_count = active_count.wrapping_add(1);
                    } else {
                        if !vt_ready {
                            let z = _mm256_set1_epi64x(t[0] as i64);
                            vt0 = z;
                            vt1 = z;
                            vt_ready = true;
                        }
                        if r_uni {
                            let rz = _mm256_set1_epi64x(r_w0 as i64);
                            vr0 = rz;
                            vr1 = rz;
                            r_uni = false;
                        }
                        let b_ptr = anchor_phases[idx].as_ptr().cast::<__m256i>();
                        let vb0 = _mm256_loadu_si256(b_ptr);
                        let vb1 = _mm256_loadu_si256(b_ptr.add(1));
                        let vb_or = _mm256_or_si256(vb0, vb1);
                        let coh = if trail_nonzero && _mm256_testz_si256(vb_or, vb_or) == 0 {
                            avx2_coherence_from_diff_vec64(
                                _mm256_sub_epi8(vt0, vb0),
                                _mm256_sub_epi8(vt1, vb1),
                            )
                        } else {
                            0.0
                        };
                        let proximity_gain = branchless_clamp_f32(1.0 + coh, 0.25, 2.0);
                        let res_pull =
                            branchless_flush_subnormal_f32(clean_energy * proximity_gain);
                        let is_better = res_pull > best_pull;
                        let total_w = running_weight + res_pull;
                        if running_weight <= 1.0e-12 {
                            if b_uni {
                                r_w0 = b[0];
                                r_uni = true;
                            } else {
                                vr0 = vb0;
                                vr1 = vb1;
                            }
                        } else {
                            let vd0 = _mm256_sub_epi8(vb0, vr0);
                            let vd1 = _mm256_sub_epi8(vb1, vr1);
                            let diff_or = _mm256_or_si256(vd0, vd1);
                            if _mm256_testz_si256(diff_or, diff_or) == 0 {
                                let ratio_f32 = branchless_clamp_f32(
                                    (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                    0.0,
                                    128.0,
                                );
                                let s = (ratio_f32 as u8) as u64;
                                if s >= 128 {
                                    vr0 = vb0;
                                    vr1 = vb1;
                                } else if s > 0 {
                                    (vr0, vr1) =
                                        avx2_slerp_from_diff_vec64(vr0, vr1, vd0, vd1, s);
                                }
                            }
                        }

                        running_weight = total_w;
                        anchor_pull_sum += res_pull;
                        weighted_coh_sum += res_pull * coh;
                        best_pull = branchless_select_f32(is_better, res_pull, best_pull);
                        dominant_idx = branchless_select_u32(is_better, idx as u32, dominant_idx);
                        active_count = active_count.wrapping_add(1);
                    }
                }
                idx += 1;
            }

            let has_pull = anchor_pull_sum > 1.0e-12;
            let inv_pull =
                fast_inv_pos_normal_f32(branchless_select_f32(has_pull, anchor_pull_sum, 1.0));
            let mean_coh = branchless_select_f32(
                has_pull,
                branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
                0.0,
            );
            let out_w = if r_uni {
                [r_w0; 8]
            } else {
                avx2_vec64_to_words64(vr0, vr1)
            };
            (
                out_w,
                branchless_flush_subnormal_f32(running_weight),
                mean_coh,
                dominant_idx,
                active_count,
            )
        };
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
    let t_is_uniform = is_word_array_uniform_64(&t);
    let mut r = t;
    let mut r_is_uniform = t_is_uniform;
    let mut running_weight = init_w;
    let mut anchor_pull_sum = 0.0_f32;
    let mut weighted_coh_sum = 0.0_f32;
    let mut best_pull = 0.0_f32;
    let mut dominant_idx = 0u32;
    let mut active_count = 0u32;

    let mut idx = 0usize;
    while idx < count {
        let clean_energy =
            branchless_flush_subnormal_f32(branchless_relu_f32(anchor_base_energies[idx]));
        if clean_energy > 1.0e-12 {
            let b = load_q7_words_64(&anchor_phases[idx]);
            let need_b_uni = (trail_nonzero & t_is_uniform) | (running_weight <= 1.0e-12) | r_is_uniform;
            let b_is_uniform = if need_b_uni {
                is_word_array_uniform_64(&b)
            } else {
                false
            };
            let anchor_nonzero = if b_is_uniform {
                b[0] != 0
            } else {
                (((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0
            };
            let coh = if trail_nonzero & anchor_nonzero {
                coherence_from_word_arrays_with_both_uniform_flags_64(
                    &t,
                    t_is_uniform,
                    &b,
                    b_is_uniform,
                )
            } else {
                0.0
            };
            let proximity_gain = branchless_clamp_f32(1.0 + coh, 0.25, 2.0);
            let res_pull = branchless_flush_subnormal_f32(clean_energy * proximity_gain);

            let is_better = res_pull > best_pull;
            let total_w = running_weight + res_pull;
            if running_weight <= 1.0e-12 {
                r = b;
                r_is_uniform = b_is_uniform;
            } else {
                let ratio_f32 = branchless_clamp_f32(
                    (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                    0.0,
                    128.0,
                );
                let s = (ratio_f32 as u8) as u64;
                let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                let (next_r, next_uni) = swar_slerp_words_with_uniform_flag_64(
                    &r,
                    r_is_uniform,
                    &b,
                    b_is_uniform,
                    s,
                    bias_restore,
                );
                r = next_r;
                r_is_uniform = next_uni;
            }

            running_weight = total_w;
            anchor_pull_sum += res_pull;
            weighted_coh_sum += res_pull * coh;
            best_pull = branchless_select_f32(is_better, res_pull, best_pull);
            dominant_idx = branchless_select_u32(is_better, idx as u32, dominant_idx);
            active_count = active_count.wrapping_add(1);
        }
        idx += 1;
    }

    let has_pull = anchor_pull_sum > 1.0e-12;
    let inv_pull = fast_inv_pos_normal_f32(branchless_select_f32(has_pull, anchor_pull_sum, 1.0));
    let mean_coh = branchless_select_f32(
        has_pull,
        branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
        0.0,
    );
    (
        r,
        branchless_flush_subnormal_f32(running_weight),
        mean_coh,
        dominant_idx,
        active_count,
    )
    }
}

/// Order-invariant (commutative) folding of an arbitrary slice of anchors
/// (`anchor_phases`, `anchor_base_energies`) on the **Single Unified Attractor Phase Manifold**
/// ($SO(2)^{d_{\mathrm{head}}/2}$) into `folded_phases_out` starting from `trail_phases` with
/// initial inertia weight `trail_init_weight`.
#[inline]
pub fn branchless_q7_unified_field_fold_with_init_weight_64(
    trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    trail_init_weight: f32,
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
    folded_phases_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32, u32, u32) {
    let t = load_q7_words_64(trail_phases);
    let (r, total_pull, mean_coh, dom_idx, act_cnt) =
        branchless_q7_unified_field_fold_words_with_init_weight_64(
            &t,
            trail_init_weight,
            anchor_phases,
            anchor_base_energies,
        );
    store_q7_words_64(folded_phases_out, &r);
    (total_pull, mean_coh, dom_idx, act_cnt)
}

/// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
/// **Single Unified Attractor Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) into `folded_phases_out`.
#[inline(always)]
pub fn branchless_q7_unified_field_fold_64(
    trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_base_energies: &[f32],
    folded_phases_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32, u32, u32) {
    branchless_q7_unified_field_fold_with_init_weight_64(
        trail_phases,
        0.0,
        anchor_phases,
        anchor_base_energies,
        folded_phases_out,
    )
}

/// Folds an arbitrary slice of crystallized attractors (`anchor_phases`, `anchor_gravity_masses`)
/// on $SO(2)^{64}$ weighted by inverse-chord **Phase-Proximity Gravity**
/// $F_{\mathrm{grav}}(k) = \frac{M_{\mathrm{grav},k} \cdot \max(0, 1 + \mathcal{C}_{\mathrm{phase}})}{2(1 - \mathcal{C}_{\mathrm{phase}}) + \epsilon_{\mathrm{grav}}}$.
///
/// Returns `(total_gravity_pull, mean_coherence, dominant_anchor_idx, active_count)`.
#[inline]
pub fn branchless_q7_gravity_fold_64(
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    anchor_gravity_masses: &[f32],
    softening_eps: f32,
    folded_phases_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32, u32, u32) {
    let count = branchless_select_usize(
        anchor_phases.len() < anchor_gravity_masses.len(),
        anchor_phases.len(),
        anchor_gravity_masses.len(),
    );
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if count > 0 {
            // SAFETY: Guarded by `target_feature = "avx2"`. Executes 64-head coherence and
            // geodesic SLERP directly in two 256-bit YMM registers (`vr0, vr1`).
            return unsafe {
                use core::arch::x86_64::{
                    _mm256_or_si256, _mm256_sub_epi8, _mm256_testz_si256,
                };
                let q_w = load_q7_words_64(query_phases);
                let q_uni = is_word_array_uniform_64(&q_w);
                let (vq0, vq1) = avx2_words64_to_vec64_with_uniform(&q_w, q_uni);
                let q_or = _mm256_or_si256(vq0, vq1);
                let q_nonzero = _mm256_testz_si256(q_or, q_or) == 0;
                let mut vr0 = vq0;
                let mut vr1 = vq1;
                let mut running_pull = 0.0_f32;
                let mut weighted_coh_sum = 0.0_f32;
                let mut best_pull = 0.0_f32;
                let mut dominant_idx = 0u32;
                let mut active_count = 0u32;
                let mut idx = 0usize;
                while idx < count {
                    let mass = branchless_flush_subnormal_f32(branchless_relu_f32(
                        anchor_gravity_masses[idx],
                    ));
                    if mass > 1.0e-12 {
                        let b_w = load_q7_words_64(&anchor_phases[idx]);
                        let (vb0, vb1) = avx2_words64_to_vec64(&b_w);
                        let b_or = _mm256_or_si256(vb0, vb1);
                        let coh = if q_nonzero && _mm256_testz_si256(b_or, b_or) == 0 {
                            avx2_coherence_from_diff_vec64(
                                _mm256_sub_epi8(vq0, vb0),
                                _mm256_sub_epi8(vq1, vb1),
                            )
                        } else {
                            0.0
                        };
                        let grav_pull =
                            branchless_phase_proximity_gravity_f32(mass, coh, softening_eps);
                        if grav_pull > 1.0e-12 {
                            let is_better = grav_pull > best_pull;
                            let total_w = running_pull + grav_pull;
                            if running_pull <= 1.0e-12 {
                                vr0 = vb0;
                                vr1 = vb1;
                            } else {
                                let ratio_f32 = branchless_clamp_f32(
                                    (grav_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                    0.0,
                                    128.0,
                                );
                                let s = (ratio_f32 as u8) as u64;
                                if s >= 128 {
                                    vr0 = vb0;
                                    vr1 = vb1;
                                } else if s > 0 {
                                    (vr0, vr1) = avx2_slerp_vec64(vr0, vr1, vb0, vb1, s);
                                }
                            }
                            running_pull = total_w;
                            weighted_coh_sum += grav_pull * coh;
                            best_pull = branchless_select_f32(is_better, grav_pull, best_pull);
                            dominant_idx =
                                branchless_select_u32(is_better, idx as u32, dominant_idx);
                            active_count = active_count.wrapping_add(1);
                        }
                    }
                    idx += 1;
                }
                let out_w = avx2_vec64_to_words64(vr0, vr1);
                store_q7_words_64(folded_phases_out, &out_w);
                let has_pull = (active_count > 0) & (running_pull > 1.0e-12);
                let inv_pull =
                    fast_inv_pos_normal_f32(branchless_select_f32(has_pull, running_pull, 1.0));
                let mean_coh = branchless_select_f32(
                    has_pull,
                    branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
                    0.0,
                );
                (
                    branchless_flush_subnormal_f32(running_pull),
                    mean_coh,
                    dominant_idx,
                    active_count,
                )
            };
        }
    }
    let q = load_q7_words_64(query_phases);
    let q_is_uniform = is_word_array_uniform_64(&q);
    let q_nonzero = if q_is_uniform {
        q[0] != 0
    } else {
        (((q[0] | q[1]) | (q[2] | q[3])) | ((q[4] | q[5]) | (q[6] | q[7]))) != 0
    };
    let mut r = q;
    let mut r_is_uniform = q_is_uniform;
    let mut running_pull = 0.0_f32;
    let mut weighted_coh_sum = 0.0_f32;
    let mut best_pull = 0.0_f32;
    let mut dominant_idx = 0u32;
    let mut active_count = 0u32;

    let mut idx = 0usize;
    while idx < count {
        let mass =
            branchless_flush_subnormal_f32(branchless_relu_f32(anchor_gravity_masses[idx]));
        if mass > 1.0e-12 {
            let b = load_q7_words_64(&anchor_phases[idx]);
            let b_is_uniform = is_word_array_uniform_64(&b);
            let b_nonzero = if b_is_uniform {
                b[0] != 0
            } else {
                (((b[0] | b[1]) | (b[2] | b[3])) | ((b[4] | b[5]) | (b[6] | b[7]))) != 0
            };
            let coh = if q_nonzero & b_nonzero {
                coherence_from_word_arrays_with_both_uniform_flags_64(
                    &q,
                    q_is_uniform,
                    &b,
                    b_is_uniform,
                )
            } else {
                0.0
            };
            let grav_pull = branchless_phase_proximity_gravity_f32(mass, coh, softening_eps);
            if grav_pull > 1.0e-12 {
                let is_better = grav_pull > best_pull;
                let total_w = running_pull + grav_pull;
                if running_pull <= 1.0e-12 {
                    r = b;
                    r_is_uniform = b_is_uniform;
                } else {
                    let ratio_f32 = branchless_clamp_f32(
                        (grav_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                        0.0,
                        128.0,
                    );
                    let s = (ratio_f32 as u8) as u64;
                    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let (next_r, next_uni) = swar_slerp_words_with_uniform_flag_64(
                        &r,
                        r_is_uniform,
                        &b,
                        b_is_uniform,
                        s,
                        bias_restore,
                    );
                    r = next_r;
                    r_is_uniform = next_uni;
                }

                running_pull = total_w;
                weighted_coh_sum += grav_pull * coh;
                best_pull = branchless_select_f32(is_better, grav_pull, best_pull);
                dominant_idx = branchless_select_u32(is_better, idx as u32, dominant_idx);
                active_count = active_count.wrapping_add(1);
            }
        }
        idx += 1;
    }

    store_q7_words_64(folded_phases_out, &r);
    let has_pull = running_pull > 1.0e-12;
    let inv_pull = fast_inv_pos_normal_f32(branchless_select_f32(has_pull, running_pull, 1.0));
    let mean_coh = branchless_select_f32(
        has_pull,
        branchless_clamp_f32(weighted_coh_sum * inv_pull, -1.0, 1.0),
        0.0,
    );
    (
        branchless_flush_subnormal_f32(running_pull),
        mean_coh,
        dominant_idx,
        active_count,
    )
}

/// Evaluates the conditional transition coherence $\mathcal{C}_{\mathrm{phase}}(M_{t-1}, M_t) \in [-1.0, 1.0]$
/// and log-transition probability $\ln P(M_t \mid M_{t-1}) \le 0$ directly on 8 register-resident `u64` SWAR words.
#[inline(always)]
#[must_use]
pub fn branchless_q7_transition_log_prob_words_64(
    prev_words: &[u64; 8],
    curr_words: &[u64; 8],
) -> (f32, f32) {
    let coh = coherence_from_word_arrays_64(prev_words, curr_words);
    let cond_prob = branchless_clamp_f32(0.5 * (1.0 + coh), DEFAULT_ENTROPY_EPS, 1.0);
    (coh, branchless_flush_subnormal_f32(branchless_ln_f32(cond_prob)))
}

/// Evaluates the conditional transition coherence $\mathcal{C}_{\mathrm{phase}}(M_{t-1}, M_t) \in [-1.0, 1.0]$
/// and log-transition probability $\ln P(M_t \mid M_{t-1}) = \ln(\max((1 + \mathcal{C}_{\mathrm{phase}})/2, \epsilon_{\mathrm{log}})) \le 0$.
#[inline(always)]
pub fn branchless_q7_transition_log_prob_64(
    prev_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    curr_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32) {
    branchless_q7_transition_log_prob_words_64(
        &load_q7_words_64(prev_phases),
        &load_q7_words_64(curr_phases),
    )
}

/// Evaluates the **Attention $\times$ DPSR Phase-Interference Logit Delta**
/// $\Delta z_{t,j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_j)$.
#[inline(always)]
pub fn branchless_q7_attention_score_delta_64(
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    head_dim: u32,
    interference_gain: f32,
) -> (f32, f32) {
    let coh = branchless_q7_coherence_64(query_phases, key_phases);
    let safe_dim = branchless_select_u32(head_dim > 0, head_dim, 128);
    let rsqrt_d = if safe_dim <= 256 {
        POPCOUNT_RSQRT_LUT_257_REF[safe_dim as usize]
    } else {
        branchless_rsqrt_f32(safe_dim as f32)
    };
    let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
    (coh, branchless_flush_subnormal_f32(clean_gain * rsqrt_d * coh))
}

/// Evaluates the **Bayes-Attention Stigmergic Logit Delta** uniting **Hypothesis 1** and **Hypothesis 2**
/// in `const fn` constant time (`0` branches, `0 fdiv`), returning `(phase_coherence, bayesian_logit_delta)`.
#[inline(always)]
pub fn branchless_q7_bayesian_attention_score_delta_64(
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_stigmergic_density: f32,
    is_pointer_linked: bool,
    head_dim: u32,
    interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
) -> (f32, f32) {
    let (coh, attn_delta) = branchless_q7_attention_score_delta_64(
        query_phases,
        key_phases,
        head_dim,
        interference_gain,
    );
    let clean_rho = branchless_clamp_f32(
        branchless_flush_subnormal_f32(branchless_relu_f32(key_stigmergic_density)),
        0.0,
        MAX_BAYESIAN_STIGMERGIC_DENSITY,
    );
    let prior_delta = branchless_flush_subnormal_f32(
        branchless_relu_f32(stigmergic_prior_gain) * fast_ln_1p_bounded_f32(clean_rho),
    );
    let link_delta = branchless_select_f32(
        is_pointer_linked,
        branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus)),
        0.0,
    );
    (
        coh,
        branchless_flush_subnormal_f32(attn_delta + prior_delta + link_delta),
    )
}

/// Computes the bounded **Phase-Interference Inverse-Temperature Scale**
/// $\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \cdot \mathcal{C}_{\mathrm{phase}}, 0.25, 4.0)$.
#[inline(always)]
pub const fn branchless_phase_coherence_inv_temp_scale(
    coherence: f32,
    interference_gain: f32,
) -> f32 {
    let clean_coh = branchless_clamp_f32(
        branchless_select_f32(branchless_is_finite_f32(coherence), coherence, 0.0),
        -1.0,
        1.0,
    );
    let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
    branchless_clamp_f32(
        1.0 + clean_gain * clean_coh,
        MIN_PHASE_INTERFERENCE_INV_TEMP,
        MAX_PHASE_INTERFERENCE_INV_TEMP,
    )
}

/// Directly modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
/// in-place by each memory $j$'s phase interference, preloading `query_phases` into 8 `u64` registers once.
#[inline]
pub fn modulate_attention_scores_by_phase_bank_64(
    attention_scores: &mut [f32],
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    head_dim: u32,
    interference_gain: f32,
) -> f32 {
    let count = attention_scores.len().min(key_anchor_phases.len());
    if count == 0 {
        return 0.0;
    }
    let q = load_q7_words_64(query_phases);
    let q_is_uniform = is_word_array_uniform_64(&q);
    let safe_dim = branchless_select_u32(head_dim > 0, head_dim, 128);
    let rsqrt_d = if safe_dim <= 256 {
        POPCOUNT_RSQRT_LUT_257_REF[safe_dim as usize]
    } else {
        branchless_rsqrt_f32(safe_dim as f32)
    };
    let eff_scale =
        branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain) * rsqrt_d);

    let mut coh_sum = 0.0_f32;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (vq0, vq1) = unsafe { avx2_words64_to_vec64_with_uniform(&q, q_is_uniform) };
    for (score_slot, key_p) in attention_scores[..count]
        .iter_mut()
        .zip(key_anchor_phases[..count].iter())
    {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let coh = {
            if q_is_uniform {
                let kw = load_q7_words_64(key_p);
                if is_word_array_uniform_64(&kw) {
                    coherence_from_word_arrays_with_both_uniform_flags_64(&q, true, &kw, true)
                } else {
                    unsafe {
                        use core::arch::x86_64::_mm256_sub_epi8;
                        let (vk0, vk1) = avx2_words64_to_vec64(&kw);
                        avx2_coherence_from_diff_vec64(
                            _mm256_sub_epi8(vq0, vk0),
                            _mm256_sub_epi8(vq1, vk1),
                        )
                    }
                }
            } else {
                let kw = load_q7_words_64(key_p);
                unsafe {
                    use core::arch::x86_64::_mm256_sub_epi8;
                    let (vk0, vk1) = avx2_words64_to_vec64(&kw);
                    avx2_coherence_from_diff_vec64(
                        _mm256_sub_epi8(vq0, vk0),
                        _mm256_sub_epi8(vq1, vk1),
                    )
                }
            }
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let coh = coherence_from_word_arrays_with_uniform_flag_64(
            &q,
            q_is_uniform,
            &load_q7_words_64(key_p),
        );
        *score_slot = branchless_flush_subnormal_f32(*score_slot + eff_scale * coh);
        coh_sum += coh;
    }
    branchless_clamp_f32(coh_sum * fast_inv_len_f32(count), -1.0, 1.0)
}

/// Directly modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
/// in-place by the full **Bayes-Attention Stigmergic Transition Score** (`0` branches, `0 fdiv`),
/// preloading `query_phases` into registers and using `fast_ln_1p_bounded_f32`.
#[inline]
pub fn modulate_attention_scores_by_bayesian_trails_64(
    attention_scores: &mut [f32],
    query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    key_anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
    key_stigmergic_densities: &[f32],
    linked_trail_id: Option<u32>,
    head_dim: u32,
    interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
) -> (f32, u32) {
    let count = attention_scores
        .len()
        .min(key_anchor_phases.len())
        .min(key_stigmergic_densities.len());
    if count == 0 {
        return (0.0, 0);
    }

    let q = load_q7_words_64(query_phases);
    let q_is_uniform = is_word_array_uniform_64(&q);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (vq0, vq1) = unsafe { avx2_words64_to_vec64_with_uniform(&q, q_is_uniform) };
    let safe_dim = branchless_select_u32(head_dim > 0, head_dim, 128);
    let rsqrt_d = if safe_dim <= 256 {
        POPCOUNT_RSQRT_LUT_257_REF[safe_dim as usize]
    } else {
        branchless_rsqrt_f32(safe_dim as f32)
    };
    let eff_interf_scale =
        branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain) * rsqrt_d);
    let clean_prior_gain = branchless_relu_f32(stigmergic_prior_gain);
    let clean_link_bonus =
        branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));

    let has_link = linked_trail_id.is_some();
    let link_target = linked_trail_id.unwrap_or(0);
    let mut coh_sum = 0.0_f32;
    let mut best_score = -1.0e30_f32;
    let mut best_idx = 0u32;

    for idx in 0..count {
        let is_linked = has_link & ((idx as u32) == link_target);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let coh = {
            if q_is_uniform {
                let kw = load_q7_words_64(&key_anchor_phases[idx]);
                if is_word_array_uniform_64(&kw) {
                    coherence_from_word_arrays_with_both_uniform_flags_64(&q, true, &kw, true)
                } else {
                    unsafe {
                        use core::arch::x86_64::_mm256_sub_epi8;
                        let (vk0, vk1) = avx2_words64_to_vec64(&kw);
                        avx2_coherence_from_diff_vec64(
                            _mm256_sub_epi8(vq0, vk0),
                            _mm256_sub_epi8(vq1, vk1),
                        )
                    }
                }
            } else {
                let kw = load_q7_words_64(&key_anchor_phases[idx]);
                unsafe {
                    use core::arch::x86_64::_mm256_sub_epi8;
                    let (vk0, vk1) = avx2_words64_to_vec64(&kw);
                    avx2_coherence_from_diff_vec64(
                        _mm256_sub_epi8(vq0, vk0),
                        _mm256_sub_epi8(vq1, vk1),
                    )
                }
            }
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let coh = coherence_from_word_arrays_with_uniform_flag_64(
            &q,
            q_is_uniform,
            &load_q7_words_64(&key_anchor_phases[idx]),
        );
        let attn_delta = branchless_flush_subnormal_f32(eff_interf_scale * coh);
        let clean_rho = branchless_clamp_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(key_stigmergic_densities[idx])),
            0.0,
            MAX_BAYESIAN_STIGMERGIC_DENSITY,
        );
        let prior_delta =
            branchless_flush_subnormal_f32(clean_prior_gain * fast_ln_1p_bounded_f32(clean_rho));
        let link_delta = branchless_select_f32(is_linked, clean_link_bonus, 0.0);
        let delta_z = branchless_flush_subnormal_f32(attn_delta + prior_delta + link_delta);

        let new_score = branchless_flush_subnormal_f32(attention_scores[idx] + delta_z);
        attention_scores[idx] = new_score;
        coh_sum += coh;
        let better = (idx == 0) | (new_score > best_score);
        best_score = branchless_select_f32(better, new_score, best_score);
        best_idx = branchless_select_u32(better, idx as u32, best_idx);
    }

    (
        branchless_clamp_f32(coh_sum * fast_inv_len_f32(count), -1.0, 1.0),
        best_idx,
    )
}

/// Accumulates 8 biased `(sin_q14 + 16384, cos_q14 + 16384)` pairs from a single `u64` word `w`
/// into a single `u64` register using `Q7_FULL_SINCOS_BIASED_U64_LUT` in strictly constant time
/// (`0` data-dependent branches).
#[inline(always)]
const fn sum_sincos_biased_u64_from_word(w: u64) -> u64 {
    let lut = Q7_FULL_SINCOS_BIASED_U64_REF;
    let b = w.to_le_bytes();
    let p01 = lut[b[0] as usize].wrapping_add(lut[b[1] as usize]);
    let p23 = lut[b[2] as usize].wrapping_add(lut[b[3] as usize]);
    let p45 = lut[b[4] as usize].wrapping_add(lut[b[5] as usize]);
    let p67 = lut[b[6] as usize].wrapping_add(lut[b[7] as usize]);
    p01.wrapping_add(p23).wrapping_add(p45.wrapping_add(p67))
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
const fn sum_sincos_q14_from_words_scalar_64(w: &[u64; 8]) -> (f32, f32) {
    let s01 = sum_sincos_biased_u64_from_word(w[0]).wrapping_add(sum_sincos_biased_u64_from_word(w[1]));
    let s23 = sum_sincos_biased_u64_from_word(w[2]).wrapping_add(sum_sincos_biased_u64_from_word(w[3]));
    let s45 = sum_sincos_biased_u64_from_word(w[4]).wrapping_add(sum_sincos_biased_u64_from_word(w[5]));
    let s67 = sum_sincos_biased_u64_from_word(w[6]).wrapping_add(sum_sincos_biased_u64_from_word(w[7]));
    let total = s01.wrapping_add(s23).wrapping_add(s45.wrapping_add(s67));
    let sum_cos_q14 = ((total & 0xFFFF_FFFF) as i32) - Q14_BIAS_X64_I32;
    let sum_sin_q14 = ((total >> 32) as i32) - Q14_BIAS_X64_I32;
    (
        (sum_sin_q14 as f32) * INV_64_Q14_F32,
        (sum_cos_q14 as f32) * INV_64_Q14_F32,
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_mean_sincos_from_words_64(w: &[u64; 8]) -> (f32, f32) {
    // SAFETY: Guarded by `target_feature = "avx2"`.
    unsafe {
        let (v0, v1) = avx2_words64_to_vec64(w);
        avx2_mean_sincos_from_vec64(v0, v1)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_mean_sincos_from_vec64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
) -> (f32, f32) {
    // SAFETY: Guarded by `target_feature = "avx2"`. Evaluates 64-head (mean_sin, mean_cos) in
    // registers via 16-nibble `vpshufb` + `vpsadbw` + `vpmaddubsw` with packed horizontal reduction.
    unsafe {
        use core::arch::x86_64::{
            _mm256_add_epi16, _mm256_add_epi32, _mm256_add_epi64, _mm256_and_si256,
            _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_madd_epi16,
            _mm256_maddubs_epi16, _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi16,
            _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8,
            _mm256_slli_epi64, _mm256_srli_epi16, _mm256_sub_epi16, _mm_add_epi32, _mm_add_epi64,
            _mm_cvtsi128_si32, _mm_cvtsi128_si64, _mm_extract_epi32, _mm_shuffle_epi32,
            _mm_unpackhi_epi64, _mm_unpacklo_epi64,
        };

        let mask_15 = _mm256_set1_epi8(0x0F);
        let l0 = _mm256_and_si256(v0, mask_15);
        let l1 = _mm256_and_si256(v1, mask_15);
        let h0 = _mm256_and_si256(_mm256_srli_epi16(v0, 4), mask_15);
        let h1 = _mm256_and_si256(_mm256_srli_epi16(v1, 4), mask_15);

        let lut_cos0_lo = _mm256_setr_epi8(
            0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126, 0, 126, 65, 33,
            0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126, 0, 126, 65, 33,
        );
        let lut_cos0_hi = _mm256_setr_epi8(
            -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39, 64, 88, 109, 123,
            -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39, 64, 88, 109, 123,
        );
        let lut_sin0_lo = _mm256_setr_epi8(
            0, 126, 65, 33, 0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126,
            0, 126, 65, 33, 0, 33, 65, 126, 0, -126, -65, -33, 0, -33, -65, -126,
        );
        let lut_sin0_hi = _mm256_setr_epi8(
            64, 88, 109, 123, -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39,
            64, 88, 109, 123, -128, 123, 109, 88, 64, 39, 18, 4, 0, 4, 18, 39,
        );
        let lut_cos_h = _mm256_setr_epi8(
            127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117,
            127, 117, 90, 49, 0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117,
        );
        let lut_nsin_h = _mm256_setr_epi8(
            0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117, 127, 117, 90, 49,
            0, -49, -90, -117, -127, -117, -90, -49, 0, 49, 90, 117, 127, 117, 90, 49,
        );
        let lut_sin_l = _mm256_setr_epi8(
            0, 6, 13, 19, 25, 32, 38, 44, 50, 57, 63, 69, 75, 81, 87, 93,
            0, 6, 13, 19, 25, 32, 38, 44, 50, 57, 63, 69, 75, 81, 87, 93,
        );
        let lut_omc_l = _mm256_setr_epi8(
            0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17,
            0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 8, 9, 11, 13, 15, 17,
        );

        let zero = _mm256_setzero_si256();
        let clo0 = _mm256_shuffle_epi8(lut_cos0_lo, h0);
        let clo1 = _mm256_shuffle_epi8(lut_cos0_lo, h1);
        let chi0 = _mm256_shuffle_epi8(lut_cos0_hi, h0);
        let chi1 = _mm256_shuffle_epi8(lut_cos0_hi, h1);
        let c_sad_lo = _mm256_add_epi64(_mm256_sad_epu8(clo0, zero), _mm256_sad_epu8(clo1, zero));
        let c_sad_hi = _mm256_add_epi64(_mm256_sad_epu8(chi0, zero), _mm256_sad_epu8(chi1, zero));
        let c_sad_tot = _mm256_add_epi64(c_sad_lo, _mm256_slli_epi64(c_sad_hi, 8));

        let slo0 = _mm256_shuffle_epi8(lut_sin0_lo, h0);
        let slo1 = _mm256_shuffle_epi8(lut_sin0_lo, h1);
        let shi0 = _mm256_shuffle_epi8(lut_sin0_hi, h0);
        let shi1 = _mm256_shuffle_epi8(lut_sin0_hi, h1);
        let s_sad_lo = _mm256_add_epi64(_mm256_sad_epu8(slo0, zero), _mm256_sad_epu8(slo1, zero));
        let s_sad_hi = _mm256_add_epi64(_mm256_sad_epu8(shi0, zero), _mm256_sad_epu8(shi1, zero));
        let s_sad_tot = _mm256_add_epi64(s_sad_lo, _mm256_slli_epi64(s_sad_hi, 8));

        let cs_sad = _mm256_or_si256(c_sad_tot, _mm256_slli_epi64(s_sad_tot, 32));
        let cs_sad_128 = _mm_add_epi64(
            _mm256_castsi256_si128(cs_sad),
            _mm256_extracti128_si256(cs_sad, 1),
        );
        let cs_base_packed = _mm_cvtsi128_si64(_mm_add_epi64(
            cs_sad_128,
            _mm_unpackhi_epi64(cs_sad_128, cs_sad_128),
        )) as u64;
        let cos_base = (cs_base_packed as i32) - (64 * 16384);
        let sin_base = ((cs_base_packed >> 32) as i32) - (64 * 16384);

        let ch0 = _mm256_shuffle_epi8(lut_cos_h, h0);
        let ch1 = _mm256_shuffle_epi8(lut_cos_h, h1);
        let nsh0 = _mm256_shuffle_epi8(lut_nsin_h, h0);
        let nsh1 = _mm256_shuffle_epi8(lut_nsin_h, h1);
        let sl0 = _mm256_shuffle_epi8(lut_sin_l, l0);
        let sl1 = _mm256_shuffle_epi8(lut_sin_l, l1);
        let ol0 = _mm256_shuffle_epi8(lut_omc_l, l0);
        let ol1 = _mm256_shuffle_epi8(lut_omc_l, l1);

        let ones16 = _mm256_set1_epi16(1);
        let c_corr0 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(sl0, nsh0),
            _mm256_maddubs_epi16(ol0, ch0),
        );
        let c_corr1 = _mm256_sub_epi16(
            _mm256_maddubs_epi16(sl1, nsh1),
            _mm256_maddubs_epi16(ol1, ch1),
        );
        let c_corr32 = _mm256_add_epi32(
            _mm256_madd_epi16(c_corr0, ones16),
            _mm256_madd_epi16(c_corr1, ones16),
        );

        let s_corr0 = _mm256_add_epi16(
            _mm256_maddubs_epi16(sl0, ch0),
            _mm256_maddubs_epi16(ol0, nsh0),
        );
        let s_corr1 = _mm256_add_epi16(
            _mm256_maddubs_epi16(sl1, ch1),
            _mm256_maddubs_epi16(ol1, nsh1),
        );
        let s_corr32 = _mm256_add_epi32(
            _mm256_madd_epi16(s_corr0, ones16),
            _mm256_madd_epi16(s_corr1, ones16),
        );

        let cc128 = _mm_add_epi32(
            _mm256_castsi256_si128(c_corr32),
            _mm256_extracti128_si256(c_corr32, 1),
        );
        let sc128 = _mm_add_epi32(
            _mm256_castsi256_si128(s_corr32),
            _mm256_extracti128_si256(s_corr32, 1),
        );
        let pair64 = _mm_add_epi32(
            _mm_unpacklo_epi64(cc128, sc128),
            _mm_unpackhi_epi64(cc128, sc128),
        );
        let pair32 = _mm_add_epi32(pair64, _mm_shuffle_epi32(pair64, 0xB1));
        let cos_corr = _mm_cvtsi128_si32(pair32);
        let sin_corr = _mm_extract_epi32::<2>(pair32);

        let mean_sin = ((sin_base as f32) + 0.5_f32 * (sin_corr as f32)) * INV_64_Q14_F32;
        let mean_cos = ((cos_base as f32) + 0.5_f32 * (cos_corr as f32)) * INV_64_Q14_F32;
        (mean_sin, mean_cos)
    }
}

#[inline(always)]
fn mean_sincos_from_words_with_uniform_flag_64(w: &[u64; 8], w_is_uniform: bool) -> (f32, f32) {
    if w_is_uniform {
        let w0 = w[0];
        if w0 == (w0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
            return branchless_sincos_q7((w0 & 0xFF) as u8 as gcso_q7_t);
        }
        let p = sum_sincos_biased_u64_from_word(w0);
        let sum_cos_q14 = ((p & 0xFFFF_FFFF) as i32) - Q14_BIAS_X8_I32;
        let sum_sin_q14 = ((p >> 32) as i32) - Q14_BIAS_X8_I32;
        return (
            (sum_sin_q14 as f32) * INV_8_Q14_F32,
            (sum_cos_q14 as f32) * INV_8_Q14_F32,
        );
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `target_feature = "avx2"`.
        return unsafe { avx2_mean_sincos_from_words_64(w) };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        sum_sincos_q14_from_words_scalar_64(w)
    }
}

/// Evaluates the circular resultant vector length $\bar{R}_c \in [0.0, 1.0]$ from 8 register-resident
/// `u64` SWAR words (`w`) and a pre-evaluated word-uniformity flag (`w_is_uniform`).
#[inline(always)]
#[must_use]
pub fn circular_r_bar_from_words_with_uniform_flag_64(w: &[u64; 8], w_is_uniform: bool) -> f32 {
    if w_is_uniform {
        let w0 = w[0];
        if w0 == (w0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
            return 1.0;
        }
    }
    let (mean_sin, mean_cos) = mean_sincos_from_words_with_uniform_flag_64(w, w_is_uniform);
    fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos)
}

/// Evaluates the circular resultant vector length
/// $\bar{R} = \sqrt{\left(\frac{1}{64}\sum_{h=0}^{63}\sin\theta_h\right)^2 + \left(\frac{1}{64}\sum_{h=0}^{63}\cos\theta_h\right)^2} \in [0.0, 1.0]$
/// directly from all 8 packed 64-bit SWAR words (`w`) in strictly constant time (`0` branches, `0 fdiv`).
#[inline(always)]
pub fn circular_r_bar_from_words_64(w: &[u64; 8]) -> f32 {
    circular_r_bar_from_words_with_uniform_flag_64(w, is_word_array_uniform_64(w))
}

/// Evaluates the 64-head circular statistics summary `(r_bar, mean_phase_q7, kappa_c)` directly
/// from 8 register-resident `u64` SWAR words (`0` intermediate stack arrays, `0` data-dependent
/// branches, `0 fdiv`).
#[inline(always)]
#[must_use]
pub fn circular_resultant_from_words_with_uniform_flag_64(
    w: &[u64; 8],
    w_is_uniform: bool,
) -> (f32, gcso_q7_t, f32) {
    if w_is_uniform {
        let w0 = w[0];
        if w0 == (w0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
            let q7 = (w0 & 0xFF) as u8 as gcso_q7_t;
            return (
                1.0,
                branchless_select_i8(q7 == -128, 127, q7),
                VON_MISES_KAPPA_AT_MAX_R,
            );
        }
    }
    let (mean_sin, mean_cos) = mean_sincos_from_words_with_uniform_flag_64(w, w_is_uniform);
    let r_bar = fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos);
    let mean_q7 = branchless_atan2_to_q7_fast(mean_sin, mean_cos);
    (r_bar, mean_q7, branchless_von_mises_kappa(r_bar))
}

/// Evaluates the 64-head circular statistics summary `(r_bar, mean_phase_q7, kappa_c)` (Section 4.2)
/// directly from pre-loaded 8-word SWAR registers `w` (`0` unaligned byte loads).
#[inline(always)]
#[must_use]
pub fn circular_resultant_from_words_64(w: &[u64; 8]) -> (f32, gcso_q7_t, f32) {
    circular_resultant_from_words_with_uniform_flag_64(w, is_word_array_uniform_64(w))
}

/// Evaluates the 64-head circular statistics summary `(r_bar, mean_phase_q7, kappa_c)` (Section 4.2)
/// using direct YMM vector loads on AVX2 (`0` Port-5 GPR-to-YMM inserts) or 8-word SWAR loads on scalar targets.
#[inline(always)]
pub fn branchless_q7_circular_resultant_64(
    phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, gcso_q7_t, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `phases` is 64 bytes and `target_feature = "avx2"` is enabled.
        return unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_broadcastb_epi8, _mm256_castsi256_si128, _mm256_loadu_si256,
                _mm256_or_si256, _mm256_testz_si256, _mm256_xor_si256,
            };
            let pa = phases.as_ptr().cast::<__m256i>();
            let va0 = _mm256_loadu_si256(pa);
            let va1 = _mm256_loadu_si256(pa.add(1));
            let bcast_a = _mm256_broadcastb_epi8(_mm256_castsi256_si128(va0));
            let a_diff = _mm256_or_si256(
                _mm256_xor_si256(va0, bcast_a),
                _mm256_xor_si256(va1, bcast_a),
            );
            if _mm256_testz_si256(a_diff, a_diff) != 0 {
                let q7 = phases[0];
                return (
                    1.0,
                    branchless_select_i8(q7 == -128, 127, q7),
                    VON_MISES_KAPPA_AT_MAX_R,
                );
            }
            let (mean_sin, mean_cos) = avx2_mean_sincos_from_vec64(va0, va1);
            let r_bar = fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos);
            let mean_q7 = branchless_atan2_to_q7_fast(mean_sin, mean_cos);
            (r_bar, mean_q7, branchless_von_mises_kappa(r_bar))
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        circular_resultant_from_words_64(&load_q7_words_64(phases))
    }
}

/// Evaluates both 64-head phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$ and
/// circular resultant statistics $(\bar{R}_a, \bar{\theta}_{a,\mathrm{Q7}}, \kappa_a)$ in a single
/// YMM-resident pass on AVX2 (`va0, va1` loaded once via `_mm256_loadu_si256` and shared across both
/// `avx2_coherence_from_diff_vec64` and `avx2_mean_sincos_from_vec64` with `0` Port-5 `vpinsrq` instructions).
#[inline(always)]
pub fn branchless_q7_coherence_and_resultant_64(
    phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32, gcso_q7_t, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `phases_a` and `phases_b` are 64 bytes each and `target_feature = "avx2"` is enabled.
        return unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_broadcastb_epi8, _mm256_castsi256_si128, _mm256_loadu_si256,
                _mm256_or_si256, _mm256_sub_epi8, _mm256_testz_si256, _mm256_xor_si256,
            };
            let pa = phases_a.as_ptr().cast::<__m256i>();
            let pb = phases_b.as_ptr().cast::<__m256i>();
            let va0 = _mm256_loadu_si256(pa);
            let va1 = _mm256_loadu_si256(pa.add(1));
            let vb0 = _mm256_loadu_si256(pb);
            let vb1 = _mm256_loadu_si256(pb.add(1));
            let vd0 = _mm256_sub_epi8(va0, vb0);
            let vd1 = _mm256_sub_epi8(va1, vb1);
            let vd_or = _mm256_or_si256(vd0, vd1);
            let coherence = if _mm256_testz_si256(vd_or, vd_or) != 0 {
                1.0
            } else {
                let bcast_d = _mm256_broadcastb_epi8(_mm256_castsi256_si128(vd0));
                let d_diff = _mm256_or_si256(
                    _mm256_xor_si256(vd0, bcast_d),
                    _mm256_xor_si256(vd1, bcast_d),
                );
                if _mm256_testz_si256(d_diff, d_diff) != 0 {
                    let d0 = (phases_a[0] as u8).wrapping_sub(phases_b[0] as u8);
                    Q7_FULL_COS_REF[d0 as usize]
                } else {
                    avx2_coherence_from_diff_vec64(vd0, vd1)
                }
            };
            let bcast_a = _mm256_broadcastb_epi8(_mm256_castsi256_si128(va0));
            let a_diff = _mm256_or_si256(
                _mm256_xor_si256(va0, bcast_a),
                _mm256_xor_si256(va1, bcast_a),
            );
            if _mm256_testz_si256(a_diff, a_diff) != 0 {
                let q7 = phases_a[0];
                return (
                    coherence,
                    1.0,
                    branchless_select_i8(q7 == -128, 127, q7),
                    VON_MISES_KAPPA_AT_MAX_R,
                );
            }
            let (mean_sin, mean_cos) = avx2_mean_sincos_from_vec64(va0, va1);
            let r_bar = fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos);
            let mean_q7 = branchless_atan2_to_q7_fast(mean_sin, mean_cos);
            (coherence, r_bar, mean_q7, branchless_von_mises_kappa(r_bar))
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let wa = load_q7_words_64(phases_a);
        let wb = load_q7_words_64(phases_b);
        let wa_is_uniform = is_word_array_uniform_64(&wa);
        let coherence = coherence_from_word_arrays_with_uniform_flag_64(&wa, wa_is_uniform, &wb);
        let (r_bar, mean_q7, kappa) =
            circular_resultant_from_words_with_uniform_flag_64(&wa, wa_is_uniform);
        (coherence, r_bar, mean_q7, kappa)
    }
}

/// Executes an **Online Toroidal $k$-Means M-Step Centroid Update** in 8 register-resident `u64`
/// SWAR words (`centroid_words`, `sample_words`) with an optional precomputed E-step phase
/// coherence `precomputed_coh_opt` and pre-evaluated word-uniformity flags `(c_uni, sample_uni)`,
/// in strictly constant time (`0` data-dependent branches on phase stationarity, normalized
/// directional cosine weighting $w_{\mathrm{coh}} = \operatorname{clamp}(0.5(1 + \mathcal{C}_{\mathrm{phase}}), 0.25, 1.0)$).
///
/// Returns `(updated_centroid_words, next_base_energy, resultant_r_bar, von_mises_kappa, words_changed)`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn branchless_toroidal_centroid_step_words_with_coh_uni_and_changed_64(
    centroid_words: &[u64; 8],
    prev_sample_count: u32,
    prev_base_energy: f32,
    prev_r_bar: f32,
    prev_kappa: f32,
    sample_words: &[u64; 8],
    precomputed_coh_opt: Option<f32>,
    sample_energy: f32,
    c_uni: bool,
    sample_uni: bool,
) -> ([u64; 8], f32, f32, f32, bool) {
    let clean_e = branchless_flush_subnormal_f32(branchless_relu_f32(sample_energy));
    if clean_e <= 1.0e-12 {
        return (*centroid_words, prev_base_energy, prev_r_bar, prev_kappa, false);
    }
    let both_uni = c_uni & sample_uni;
    let same_words = (sample_words[0] == centroid_words[0])
        && (both_uni
            || ((((sample_words[1] ^ centroid_words[1]) | (sample_words[2] ^ centroid_words[2]))
                | ((sample_words[3] ^ centroid_words[3]) | (sample_words[4] ^ centroid_words[4]))
                | ((sample_words[5] ^ centroid_words[5]) | (sample_words[6] ^ centroid_words[6]))
                | (sample_words[7] ^ centroid_words[7]))
                == 0));
    let prev_e = branchless_flush_subnormal_f32(branchless_relu_f32(prev_base_energy));
    if prev_sample_count == 0 || prev_e <= 1.0e-12 {
        let rb = circular_r_bar_from_words_with_uniform_flag_64(sample_words, sample_uni);
        return (
            *sample_words,
            clean_e,
            rb,
            branchless_von_mises_kappa(rb),
            !same_words,
        );
    }
    if same_words {
        let coh = precomputed_coh_opt.unwrap_or(1.0);
        let eff_sample_e = clean_e * branchless_clamp_f32(0.5 * (1.0 + coh), 0.25, 1.0);
        let next_e = prev_e + 0.5 * eff_sample_e;
        if prev_r_bar > 0.0 && prev_kappa > 0.0 {
            return (*centroid_words, next_e, prev_r_bar, prev_kappa, false);
        }
        let rb = circular_r_bar_from_words_with_uniform_flag_64(centroid_words, c_uni);
        return (*centroid_words, next_e, rb, branchless_von_mises_kappa(rb), false);
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (d0, d1, uniform_diff_words) = if both_uni {
        (
            swar_q7_circular_diff_x8(sample_words[0], centroid_words[0]),
            0u64,
            true,
        )
    } else {
        (0u64, 0u64, false)
    };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    let (d0, d1, uniform_diff_words) = {
        let d0_val = swar_q7_circular_diff_x8(sample_words[0], centroid_words[0]);
        if both_uni {
            (d0_val, d0_val, true)
        } else {
            let d1_val = swar_q7_circular_diff_x8(sample_words[1], centroid_words[1]);
            let is_uni = (d0_val == d1_val)
                && (swar_q7_circular_diff_x8(sample_words[2], centroid_words[2]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[3], centroid_words[3]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[4], centroid_words[4]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[5], centroid_words[5]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[6], centroid_words[6]) == d0_val)
                && (swar_q7_circular_diff_x8(sample_words[7], centroid_words[7]) == d0_val);
            (d0_val, d1_val, is_uni)
        }
    };

    if uniform_diff_words {
        let coh = match precomputed_coh_opt {
            Some(c) => c,
            None => coherence_from_uniform_diff_word_x8(d0),
        };
        let eff_sample_e = clean_e * branchless_clamp_f32(0.5 * (1.0 + coh), 0.25, 1.0);
        let next_e = prev_e + 0.5 * eff_sample_e;
        if d0 == 0 {
            if prev_r_bar > 0.0 && prev_kappa > 0.0 {
                return (*centroid_words, next_e, prev_r_bar, prev_kappa, false);
            }
            let rb = circular_r_bar_from_words_with_uniform_flag_64(centroid_words, c_uni);
            return (*centroid_words, next_e, rb, branchless_von_mises_kappa(rb), false);
        }
        let tot_e = prev_e + eff_sample_e;
        let safe_tot = branchless_select_f32(tot_e > 1.0e-12, tot_e, 1.0);
        let ratio = branchless_clamp_f32(
            eff_sample_e * fast_inv_pos_normal_f32(safe_tot),
            0.05,
            0.50,
        );
        let raw_s = (ratio * 128.0 + 0.5) as u8;
        let s = branchless_select_u8(raw_s > 128, 128, raw_s) as u64;
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let step0 = swar_q7_scale_le128_with_bias_x8(d0, s, bias_restore);
        if step0 == 0 {
            if prev_r_bar > 0.0 && prev_kappa > 0.0 {
                return (*centroid_words, next_e, prev_r_bar, prev_kappa, false);
            }
            let rb = circular_r_bar_from_words_with_uniform_flag_64(centroid_words, c_uni);
            return (*centroid_words, next_e, rb, branchless_von_mises_kappa(rb), false);
        }
        let next_w = if c_uni {
            [swar_q7_circular_add_x8(centroid_words[0], step0); 8]
        } else {
            [
                swar_q7_circular_add_x8(centroid_words[0], step0),
                swar_q7_circular_add_x8(centroid_words[1], step0),
                swar_q7_circular_add_x8(centroid_words[2], step0),
                swar_q7_circular_add_x8(centroid_words[3], step0),
                swar_q7_circular_add_x8(centroid_words[4], step0),
                swar_q7_circular_add_x8(centroid_words[5], step0),
                swar_q7_circular_add_x8(centroid_words[6], step0),
                swar_q7_circular_add_x8(centroid_words[7], step0),
            ]
        };
        if step0 == (step0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64)
            && prev_r_bar > 0.0
            && prev_kappa > 0.0
        {
            return (next_w, next_e, prev_r_bar, prev_kappa, true);
        }
        let rb = circular_r_bar_from_words_with_uniform_flag_64(&next_w, c_uni);
        return (next_w, next_e, rb, branchless_von_mises_kappa(rb), true);
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let _ = d1;
        // SAFETY: Guarded by `target_feature = "avx2"`. Keeps `va0..vb1`, `vd0..vd1`, `sc0..sc1`,
        // and `out0..out1` 100% resident in YMM registers across coherence, SLERP, stationarity,
        // rigid SO(2) invariance, and circular resultant evaluation (`0` GPR-to-YMM or YMM-to-GPR STLF stalls).
        return unsafe {
            use core::arch::x86_64::{
                _mm256_add_epi8, _mm256_and_si256, _mm256_broadcastb_epi8,
                _mm256_castsi256_si128, _mm256_mulhrs_epi16, _mm256_or_si256,
                _mm256_permute4x64_epi64, _mm256_set1_epi16, _mm256_slli_epi16, _mm256_sub_epi8,
                _mm256_testz_si256, _mm256_xor_si256, _mm_cvtsi128_si64,
            };
            let (va0, va1) = avx2_words64_to_vec64_with_uniform(centroid_words, c_uni);
            let (vb0, vb1) = avx2_words64_to_vec64_with_uniform(sample_words, sample_uni);
            let vd0 = _mm256_sub_epi8(vb0, va0);
            let vd1 = _mm256_sub_epi8(vb1, va1);
            let coh = match precomputed_coh_opt {
                Some(c) => c,
                None => avx2_coherence_from_diff_vec64(vd0, vd1),
            };
            let eff_sample_e = clean_e * branchless_clamp_f32(0.5 * (1.0 + coh), 0.25, 1.0);
            let next_e = prev_e + 0.5 * eff_sample_e;
            let tot_e = prev_e + eff_sample_e;
            let safe_tot = branchless_select_f32(tot_e > 1.0e-12, tot_e, 1.0);
            let ratio = branchless_clamp_f32(
                eff_sample_e * fast_inv_pos_normal_f32(safe_tot),
                0.05,
                0.50,
            );
            let raw_s = (ratio * 128.0 + 0.5) as u8;
            let s = branchless_select_u8(raw_s > 128, 128, raw_s);
            let v_s = _mm256_set1_epi16(s as i16);
            let mask_ff00 = _mm256_set1_epi16(-256i16);
            let mask_00ff = _mm256_set1_epi16(0x00FFi16);

            let ev0 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd0, 8), v_s);
            let od0 = _mm256_mulhrs_epi16(_mm256_and_si256(vd0, mask_ff00), v_s);
            let sc0 = _mm256_or_si256(_mm256_and_si256(ev0, mask_00ff), _mm256_slli_epi16(od0, 8));

            let ev1 = _mm256_mulhrs_epi16(_mm256_slli_epi16(vd1, 8), v_s);
            let od1 = _mm256_mulhrs_epi16(_mm256_and_si256(vd1, mask_ff00), v_s);
            let sc1 = _mm256_or_si256(_mm256_and_si256(ev1, mask_00ff), _mm256_slli_epi16(od1, 8));

            let sc_or = _mm256_or_si256(sc0, sc1);
            if _mm256_testz_si256(sc_or, sc_or) != 0 {
                if prev_r_bar > 0.0 && prev_kappa > 0.0 {
                    return (*centroid_words, next_e, prev_r_bar, prev_kappa, false);
                }
                let rb = circular_r_bar_from_words_with_uniform_flag_64(centroid_words, c_uni);
                return (*centroid_words, next_e, rb, branchless_von_mises_kappa(rb), false);
            }

            let out0 = _mm256_add_epi8(va0, sc0);
            let out1 = _mm256_add_epi8(va1, sc1);
            let next_w = avx2_vec64_to_words64(out0, out1);

            let sc_splat0 = _mm256_broadcastb_epi8(_mm256_castsi256_si128(sc0));
            let sc_diff = _mm256_or_si256(
                _mm256_xor_si256(sc0, sc_splat0),
                _mm256_xor_si256(sc1, sc_splat0),
            );
            if _mm256_testz_si256(sc_diff, sc_diff) != 0 && prev_r_bar > 0.0 && prev_kappa > 0.0 {
                return (next_w, next_e, prev_r_bar, prev_kappa, true);
            }

            let out_splat0 = _mm256_broadcastb_epi8(_mm256_castsi256_si128(out0));
            let out_byte_diff = _mm256_or_si256(
                _mm256_xor_si256(out0, out_splat0),
                _mm256_xor_si256(out1, out_splat0),
            );
            if _mm256_testz_si256(out_byte_diff, out_byte_diff) != 0 {
                return (next_w, next_e, 1.0, VON_MISES_KAPPA_AT_MAX_R, true);
            }

            let diff01 = _mm256_xor_si256(out0, out1);
            let shuf0 = _mm256_permute4x64_epi64::<0x39>(out0);
            let comb = _mm256_or_si256(diff01, _mm256_xor_si256(out0, shuf0));
            let (mean_sin, mean_cos) = if _mm256_testz_si256(comb, comb) != 0 {
                let w0 = _mm_cvtsi128_si64(_mm256_castsi256_si128(out0)) as u64;
                let p = sum_sincos_biased_u64_from_word(w0);
                let sum_cos_q14 = ((p & 0xFFFF_FFFF) as i32) - Q14_BIAS_X8_I32;
                let sum_sin_q14 = ((p >> 32) as i32) - Q14_BIAS_X8_I32;
                (
                    (sum_sin_q14 as f32) * INV_8_Q14_F32,
                    (sum_cos_q14 as f32) * INV_8_Q14_F32,
                )
            } else {
                avx2_mean_sincos_from_vec64(out0, out1)
            };
            let rb = fast_sqrt_unit_f32(mean_sin * mean_sin + mean_cos * mean_cos);
            (next_w, next_e, rb, branchless_von_mises_kappa(rb), true)
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let coh = match precomputed_coh_opt {
            Some(c) => c,
            None => coherence_from_diff_words_64(
                d0,
                d1,
                swar_q7_circular_diff_x8(sample_words[2], centroid_words[2]),
                swar_q7_circular_diff_x8(sample_words[3], centroid_words[3]),
                swar_q7_circular_diff_x8(sample_words[4], centroid_words[4]),
                swar_q7_circular_diff_x8(sample_words[5], centroid_words[5]),
                swar_q7_circular_diff_x8(sample_words[6], centroid_words[6]),
                swar_q7_circular_diff_x8(sample_words[7], centroid_words[7]),
            ),
        };
        let eff_sample_e = clean_e * branchless_clamp_f32(0.5 * (1.0 + coh), 0.25, 1.0);
        let next_e = prev_e + 0.5 * eff_sample_e;
        let tot_e = prev_e + eff_sample_e;
        let safe_tot = branchless_select_f32(tot_e > 1.0e-12, tot_e, 1.0);
        let ratio = branchless_clamp_f32(
            eff_sample_e * fast_inv_pos_normal_f32(safe_tot),
            0.05,
            0.50,
        );
        let raw_s = (ratio * 128.0 + 0.5) as u8;
        let s = branchless_select_u8(raw_s > 128, 128, raw_s) as u64;
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let next_w = [
            swar_q7_circular_add_x8(
                centroid_words[0],
                swar_q7_scale_le128_with_bias_x8(d0, s, bias_restore),
            ),
            swar_q7_circular_add_x8(
                centroid_words[1],
                swar_q7_scale_le128_with_bias_x8(d1, s, bias_restore),
            ),
            swar_slerp_word_unchecked_x8(centroid_words[2], sample_words[2], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[3], sample_words[3], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[4], sample_words[4], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[5], sample_words[5], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[6], sample_words[6], s, bias_restore),
            swar_slerp_word_unchecked_x8(centroid_words[7], sample_words[7], s, bias_restore),
        ];
        let next_is_uniform = is_word_array_uniform_64(&next_w);
        let stationary = next_w[0] == centroid_words[0]
            && ((((next_w[1] ^ centroid_words[1]) | (next_w[2] ^ centroid_words[2]))
                | ((next_w[3] ^ centroid_words[3]) | (next_w[4] ^ centroid_words[4])))
                | (((next_w[5] ^ centroid_words[5]) | (next_w[6] ^ centroid_words[6]))
                    | (next_w[7] ^ centroid_words[7])))
                == 0;
        if stationary && prev_r_bar > 0.0 && prev_kappa > 0.0 {
            return (next_w, next_e, prev_r_bar, prev_kappa, false);
        }
        let rb = circular_r_bar_from_words_with_uniform_flag_64(&next_w, next_is_uniform);
        (next_w, next_e, rb, branchless_von_mises_kappa(rb), !stationary)
    }
}

/// Performs a single-pass 8-word SWAR toroidal centroid update with precomputed coherence
/// and word-uniform flags, returning `(next_words, next_energy, next_r_bar, next_kappa)`.
#[inline(always)]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn branchless_toroidal_centroid_step_words_with_coh_and_uni_64(
    centroid_words: &[u64; 8],
    prev_sample_count: u32,
    prev_base_energy: f32,
    prev_r_bar: f32,
    prev_kappa: f32,
    sample_words: &[u64; 8],
    precomputed_coh_opt: Option<f32>,
    sample_energy: f32,
    c_uni: bool,
    sample_uni: bool,
) -> ([u64; 8], f32, f32, f32) {
    let (next_w, next_e, rb, kappa, _changed) =
        branchless_toroidal_centroid_step_words_with_coh_uni_and_changed_64(
            centroid_words,
            prev_sample_count,
            prev_base_energy,
            prev_r_bar,
            prev_kappa,
            sample_words,
            precomputed_coh_opt,
            sample_energy,
            c_uni,
            sample_uni,
        );
    (next_w, next_e, rb, kappa)
}

/// Executes an **Online Toroidal $k$-Means M-Step Centroid Update** in 8 register-resident `u64`
/// SWAR words (`centroid_words`, `sample_words`) with an optional precomputed E-step phase
/// coherence `precomputed_coh_opt`, in strictly constant time (`0` data-dependent branches on
/// phase stationarity, normalized directional cosine weighting $w_{\mathrm{coh}} = \operatorname{clamp}(0.5(1 + \mathcal{C}_{\mathrm{phase}}), 0.25, 1.0)$).
///
/// Returns `(updated_centroid_words, next_base_energy, resultant_r_bar, von_mises_kappa)`.
#[inline(always)]
pub fn branchless_toroidal_centroid_step_words_with_coh_64(
    centroid_words: &[u64; 8],
    prev_sample_count: u32,
    prev_base_energy: f32,
    prev_r_bar: f32,
    prev_kappa: f32,
    sample_words: &[u64; 8],
    precomputed_coh_opt: Option<f32>,
    sample_energy: f32,
) -> ([u64; 8], f32, f32, f32) {
    let c_uni = is_word_array_uniform_64(centroid_words);
    let sample_uni = is_word_array_uniform_64(sample_words);
    branchless_toroidal_centroid_step_words_with_coh_and_uni_64(
        centroid_words,
        prev_sample_count,
        prev_base_energy,
        prev_r_bar,
        prev_kappa,
        sample_words,
        precomputed_coh_opt,
        sample_energy,
        c_uni,
        sample_uni,
    )
}

/// Executes an **Online Toroidal $k$-Means M-Step Centroid Update** in 8 register-resident `u64`
/// SWAR words (`centroid_words`, `sample_words`) in strictly constant time (`0` data-dependent
/// branches on phase stationarity).
///
/// Returns `(updated_centroid_words, next_base_energy, resultant_r_bar, von_mises_kappa)`.
#[inline(always)]
pub fn branchless_toroidal_centroid_step_words_64(
    centroid_words: &[u64; 8],
    prev_sample_count: u32,
    prev_base_energy: f32,
    prev_r_bar: f32,
    prev_kappa: f32,
    sample_words: &[u64; 8],
    sample_energy: f32,
) -> ([u64; 8], f32, f32, f32) {
    branchless_toroidal_centroid_step_words_with_coh_64(
        centroid_words,
        prev_sample_count,
        prev_base_energy,
        prev_r_bar,
        prev_kappa,
        sample_words,
        None,
        sample_energy,
    )
}

#[inline(always)]
const fn swar_synergy_word_with_bias_x8(
    a: u64,
    d: u64,
    ratio_b: u64,
    bias_ratio: u64,
    beat_u7: u64,
    bias_beat: u64,
    mask: u64,
) -> u64 {
    swar_q7_circular_add_x8(
        swar_q7_circular_add_x8(
            a,
            swar_q7_scale_le128_with_bias_x8(d, ratio_b, bias_ratio),
        ),
        swar_q7_scale_le128_with_bias_x8(d, beat_u7, bias_beat),
    ) & mask
}

/// Synthesizes a non-linear synergistic 64-head Q7 phase profile directly in 8 register-resident
/// `u64` SWAR words (`0` intermediate stack arrays, `0` branches, `0 fdiv`), returning
/// `(fused_words, effective_pull, coherence)`.
#[inline(always)]
#[must_use]
pub fn branchless_q7_synergy_fuse_words_64(
    a: &[u64; 8],
    weight_a: f32,
    b: &[u64; 8],
    weight_b: f32,
    synergy_gain: f32,
) -> ([u64; 8], f32, f32) {
    let clean_wa = branchless_relu_f32(weight_a);
    let clean_wb = branchless_relu_f32(weight_b);
    let base_sum = clean_wa + clean_wb;
    let valid_sum = base_sum > 1.0e-9;
    let safe_sum = branchless_select_f32(valid_sum, base_sum, 1.0);

    let d = [
        swar_q7_circular_diff_x8(b[0], a[0]),
        swar_q7_circular_diff_x8(b[1], a[1]),
        swar_q7_circular_diff_x8(b[2], a[2]),
        swar_q7_circular_diff_x8(b[3], a[3]),
        swar_q7_circular_diff_x8(b[4], a[4]),
        swar_q7_circular_diff_x8(b[5], a[5]),
        swar_q7_circular_diff_x8(b[6], a[6]),
        swar_q7_circular_diff_x8(b[7], a[7]),
    ];
    let raw_coherence = coherence_from_diff_words_64(
        d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
    );
    let coherence = branchless_select_f32(valid_sum, raw_coherence, 0.0);
    let pos_resonance = branchless_relu_f32(coherence);
    let liminal_beat = branchless_relu_f32(-coherence);

    let safe_gain = branchless_relu_f32(synergy_gain);
    let synergy_boost = safe_gain * base_sum * pos_resonance;
    let effective_pull = branchless_select_f32(valid_sum, base_sum + synergy_boost, 0.0);

    let inv_sum = 128.0 * fast_inv_pos_normal_f32(safe_sum);
    let ratio_b = branchless_select_u8(
        valid_sum,
        branchless_clamp_f32(clean_wb * inv_sum + 0.5, 0.0, 128.0) as u8,
        0,
    ) as u64;
    let beat_u7 = branchless_select_u8(
        valid_sum,
        (liminal_beat * branchless_clamp_f32(safe_gain, 0.0, 1.0) * 64.0 + 0.5) as u8,
        0,
    ) as u64;
    let bias_ratio = (128u64.wrapping_sub(ratio_b)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let bias_beat = (128u64.wrapping_sub(beat_u7)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let m = (valid_sum as u64).wrapping_neg();

    let fused_w = [
        swar_synergy_word_with_bias_x8(a[0], d[0], ratio_b, bias_ratio, beat_u7, bias_beat, m),
        swar_synergy_word_with_bias_x8(a[1], d[1], ratio_b, bias_ratio, beat_u7, bias_beat, m),
        swar_synergy_word_with_bias_x8(a[2], d[2], ratio_b, bias_ratio, beat_u7, bias_beat, m),
        swar_synergy_word_with_bias_x8(a[3], d[3], ratio_b, bias_ratio, beat_u7, bias_beat, m),
        swar_synergy_word_with_bias_x8(a[4], d[4], ratio_b, bias_ratio, beat_u7, bias_beat, m),
        swar_synergy_word_with_bias_x8(a[5], d[5], ratio_b, bias_ratio, beat_u7, bias_beat, m),
        swar_synergy_word_with_bias_x8(a[6], d[6], ratio_b, bias_ratio, beat_u7, bias_beat, m),
        swar_synergy_word_with_bias_x8(a[7], d[7], ratio_b, bias_ratio, beat_u7, bias_beat, m),
    ];
    (fused_w, effective_pull, coherence)
}

/// Synthesizes a non-linear synergistic 64-head Q7 phase profile and amplified pull force in
/// strictly constant time (`0` branches, `0 fdiv`), reusing circular-difference words `d[0..8]`
/// and hoisted SWAR bias multipliers for both coherence evaluation and geodesic + liminal beat interpolation.
#[inline]
pub fn branchless_q7_synergy_fuse_64(
    phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    weight_a: f32,
    phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    weight_b: f32,
    synergy_gain: f32,
    fused_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
) -> (f32, f32) {
    let a = load_q7_words_64(phases_a);
    let b = load_q7_words_64(phases_b);
    let (fused_w, effective_pull, coherence) =
        branchless_q7_synergy_fuse_words_64(&a, weight_a, &b, weight_b, synergy_gain);
    store_q7_words_64(fused_out, &fused_w);
    (effective_pull, coherence)
}

/// Applies branchless mutual inhibition gating directly on 8 register-resident `u64` SWAR words
/// (`target_words` vs `inhibitor_words`, `0` stack roundtrips, hoisted SWAR bias multiplier).
#[inline(always)]
pub fn branchless_q7_inhibit_words_64(
    target_words: &mut [u64; 8],
    target_weight: f32,
    inhibitor_words: &[u64; 8],
    inhibition_strength: f32,
) -> f32 {
    let clean_w = branchless_relu_f32(target_weight);
    let safe_strength = branchless_clamp_f32(branchless_relu_f32(inhibition_strength), 0.0, 1.0);
    if safe_strength <= 1.0e-9 {
        return clean_w;
    }
    let coherence = coherence_from_word_arrays_64(target_words, inhibitor_words);
    let opposition = branchless_relu_f32(-coherence);
    let retain_factor = branchless_clamp_f32(1.0 - safe_strength * opposition, 0.0, 1.0);
    let s = (((retain_factor * 128.0) + 0.5) as u8) as u64;
    if s >= 128 {
        return clean_w * retain_factor;
    }
    if s == 0 {
        *target_words = [0u64; 8];
        return 0.0;
    }
    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    target_words[0] = swar_q7_scale_le128_with_bias_x8(target_words[0], s, bias_restore);
    target_words[1] = swar_q7_scale_le128_with_bias_x8(target_words[1], s, bias_restore);
    target_words[2] = swar_q7_scale_le128_with_bias_x8(target_words[2], s, bias_restore);
    target_words[3] = swar_q7_scale_le128_with_bias_x8(target_words[3], s, bias_restore);
    target_words[4] = swar_q7_scale_le128_with_bias_x8(target_words[4], s, bias_restore);
    target_words[5] = swar_q7_scale_le128_with_bias_x8(target_words[5], s, bias_restore);
    target_words[6] = swar_q7_scale_le128_with_bias_x8(target_words[6], s, bias_restore);
    target_words[7] = swar_q7_scale_le128_with_bias_x8(target_words[7], s, bias_restore);
    clean_w * retain_factor
}

/// Applies branchless mutual inhibition gating to a target 64-head Q7 phase profile in a single
/// register-resident pass (`0` redundant reloads of `target_phases`, hoisted SWAR bias multiplier).
#[inline]
pub fn branchless_q7_inhibit_64(
    target_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    target_weight: f32,
    inhibitor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    inhibition_strength: f32,
) -> f32 {
    let mut t = load_q7_words_64(target_phases);
    let b = load_q7_words_64(inhibitor_phases);
    let damped = branchless_q7_inhibit_words_64(&mut t, target_weight, &b, inhibition_strength);
    store_q7_words_64(target_phases, &t);
    damped
}

/// Projects an arbitrary `&[f32]` feature/embedding slice onto $SO(2)^{64}$ (`[gcso_q7_t; 64]`)
/// using 8-YMM (64-float) AVX2 SIMD accumulation for `values.len() >= 64` and branchless
/// sanitization of non-finite values.
#[inline]
pub fn branchless_project_f32_slice_to_q7_64(
    values: &[f32],
    gain: f32,
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    if values.is_empty() {
        return out;
    }
    let clean_gain = branchless_select_f32(
        branchless_is_finite_f32(gain) & (gain >= 0.0),
        gain,
        1.0,
    );
    let mut acc = [0.0_f32; MAX_ATTENTION_HEADS];
    let len = values.len();

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let rem_start = if len >= MAX_ATTENTION_HEADS {
        let ptr = values.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and `i + 64 <= len` is bounds-checked.
        unsafe {
            let inf_exp = _mm256_set1_epi32(0x7F80_0000);
            let mut v0 = _mm256_setzero_ps();
            let mut v1 = _mm256_setzero_ps();
            let mut v2 = _mm256_setzero_ps();
            let mut v3 = _mm256_setzero_ps();
            let mut v4 = _mm256_setzero_ps();
            let mut v5 = _mm256_setzero_ps();
            let mut v6 = _mm256_setzero_ps();
            let mut v7 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let x0 = _mm256_loadu_ps(ptr.add(i));
                let x1 = _mm256_loadu_ps(ptr.add(i + 8));
                let x2 = _mm256_loadu_ps(ptr.add(i + 16));
                let x3 = _mm256_loadu_ps(ptr.add(i + 24));
                let x4 = _mm256_loadu_ps(ptr.add(i + 32));
                let x5 = _mm256_loadu_ps(ptr.add(i + 40));
                let x6 = _mm256_loadu_ps(ptr.add(i + 48));
                let x7 = _mm256_loadu_ps(ptr.add(i + 56));

                let nf0 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x0), inf_exp),
                    inf_exp,
                ));
                let nf1 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x1), inf_exp),
                    inf_exp,
                ));
                let nf2 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x2), inf_exp),
                    inf_exp,
                ));
                let nf3 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x3), inf_exp),
                    inf_exp,
                ));
                let nf4 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x4), inf_exp),
                    inf_exp,
                ));
                let nf5 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x5), inf_exp),
                    inf_exp,
                ));
                let nf6 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x6), inf_exp),
                    inf_exp,
                ));
                let nf7 = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
                    _mm256_and_si256(_mm256_castps_si256(x7), inf_exp),
                    inf_exp,
                ));

                v0 = _mm256_add_ps(v0, _mm256_andnot_ps(nf0, x0));
                v1 = _mm256_add_ps(v1, _mm256_andnot_ps(nf1, x1));
                v2 = _mm256_add_ps(v2, _mm256_andnot_ps(nf2, x2));
                v3 = _mm256_add_ps(v3, _mm256_andnot_ps(nf3, x3));
                v4 = _mm256_add_ps(v4, _mm256_andnot_ps(nf4, x4));
                v5 = _mm256_add_ps(v5, _mm256_andnot_ps(nf5, x5));
                v6 = _mm256_add_ps(v6, _mm256_andnot_ps(nf6, x6));
                v7 = _mm256_add_ps(v7, _mm256_andnot_ps(nf7, x7));
                i += 64;
            }
            let ap = acc.as_mut_ptr();
            _mm256_storeu_ps(ap, v0);
            _mm256_storeu_ps(ap.add(8), v1);
            _mm256_storeu_ps(ap.add(16), v2);
            _mm256_storeu_ps(ap.add(24), v3);
            _mm256_storeu_ps(ap.add(32), v4);
            _mm256_storeu_ps(ap.add(40), v5);
            _mm256_storeu_ps(ap.add(48), v6);
            _mm256_storeu_ps(ap.add(56), v7);
            i
        }
    } else {
        0usize
    };

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    let rem_start = 0usize;

    let mut i = rem_start;
    while i < len {
        let v = values[i];
        acc[i & 63] += branchless_select_f32(branchless_is_finite_f32(v), v, 0.0);
        i += 1;
    }

    let folds = ((len + 63) >> 6) as f32;
    let inv_folds = clean_gain * fast_inv_pos_normal_f32(folds);
    let mut h = 0usize;
    while h < MAX_ATTENTION_HEADS {
        let rad = branchless_clamp_f32(acc[h] * inv_folds, -PI, PI);
        out[h] = branchless_rad_to_q7(rad);
        h += 1;
    }
    out
}

/// Branchless QDPS discrete phase step filter in the Q7 integer domain (Section 2.3).
#[inline(always)]
pub const fn branchless_qdps_filter_q7(delta: gcso_q7_t, min_step_q7: i32) -> gcso_q7_t {
    let val = delta as i32;
    let sign_mask = val >> 31;
    let abs_val = (val ^ sign_mask) - sign_mask;
    let keep_mask = (((abs_val >= min_step_q7) & (min_step_q7 <= 127)) as i32).wrapping_neg();
    (val & keep_mask) as gcso_q7_t
}

/// Branchless $\mathcal{O}(1)$ SplitMix64 bijective hash mixer for pointer trails and bitmasks.
#[inline(always)]
pub const fn branchless_splitmix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Branchless hysteresis quantizer converting continuous depth into discrete QDPT level (Section 1.3).
#[inline(always)]
pub const fn branchless_hysteresis_quantize_depth(
    continuous_depth: f32,
    current_discrete_level: u32,
    hysteresis_band: f32,
    max_level: u32,
) -> u32 {
    let cur_f = current_discrete_level as f32;
    let delta = branchless_relu_f32(hysteresis_band);
    let step_up = continuous_depth > (cur_f + 0.5 + delta);
    let step_down = (continuous_depth < (cur_f - 0.5 - delta)) & (current_discrete_level > 0);
    let candidate_up = branchless_select_u32(
        current_discrete_level < max_level,
        current_discrete_level.wrapping_add(1),
        max_level,
    );
    let candidate_down = current_discrete_level.saturating_sub(1);
    branchless_select_u32(
        step_up,
        candidate_up,
        branchless_select_u32(step_down, candidate_down, current_discrete_level),
    )
}

#[inline(always)]
pub(crate) fn load_q7_x8_slice(slice: &[gcso_q7_t], offset: usize) -> u64 {
    debug_assert!(offset + 8 <= slice.len());
    // SAFETY: Caller guarantees `offset + 8 <= slice.len()`.
    let raw = unsafe { core::ptr::read_unaligned(slice.as_ptr().add(offset).cast::<u64>()) };
    u64::from_le(raw)
}

#[inline(always)]
pub(crate) fn store_q7_x8_slice(slice: &mut [gcso_q7_t], offset: usize, word: u64) {
    debug_assert!(offset + 8 <= slice.len());
    // SAFETY: Caller guarantees `offset + 8 <= slice.len()`.
    unsafe {
        core::ptr::write_unaligned(slice.as_mut_ptr().add(offset).cast::<u64>(), word.to_le());
    }
}

#[inline(always)]
pub(crate) fn swar_add_sat_q7_slice_in_place(
    acc: &mut [gcso_q7_t],
    delta: &[gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` are checked before each 32-byte unaligned load/store.
        unsafe {
            use core::arch::x86_64::{__m256i, _mm256_adds_epi8, _mm256_loadu_si256, _mm256_storeu_si256};
            let acc_p = acc.as_mut_ptr().cast::<__m256i>();
            let del_p = delta.as_ptr().cast::<__m256i>();
            while i + 32 <= len {
                let off = i >> 5;
                let va = _mm256_loadu_si256(acc_p.add(off));
                let vd = _mm256_loadu_si256(del_p.add(off));
                _mm256_storeu_si256(acc_p.add(off), _mm256_adds_epi8(va, vd));
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let w0 = swar_q7_add_sat_x8(load_q7_x8_slice(acc, i), load_q7_x8_slice(delta, i));
        let w1 = swar_q7_add_sat_x8(load_q7_x8_slice(acc, i + 8), load_q7_x8_slice(delta, i + 8));
        store_q7_x8_slice(acc, i, w0);
        store_q7_x8_slice(acc, i + 8, w1);
        i += 16;
    }
    while i + 8 <= len {
        let w = swar_q7_add_sat_x8(load_q7_x8_slice(acc, i), load_q7_x8_slice(delta, i));
        store_q7_x8_slice(acc, i, w);
        i += 8;
    }
    while i < len {
        acc[i] = branchless_q7_add_sat(acc[i], delta[i]);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_sub_sat_q7_slices(
    a: &[gcso_q7_t],
    b: &[gcso_q7_t],
    out: &mut [gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` are checked before each 32-byte unaligned load/store.
        unsafe {
            use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_subs_epi8};
            let ap = a.as_ptr().cast::<__m256i>();
            let bp = b.as_ptr().cast::<__m256i>();
            let op = out.as_mut_ptr().cast::<__m256i>();
            while i + 32 <= len {
                let off = i >> 5;
                let va = _mm256_loadu_si256(ap.add(off));
                let vb = _mm256_loadu_si256(bp.add(off));
                _mm256_storeu_si256(op.add(off), _mm256_subs_epi8(va, vb));
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let w0 = swar_q7_sub_sat_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        let w1 = swar_q7_sub_sat_x8(load_q7_x8_slice(a, i + 8), load_q7_x8_slice(b, i + 8));
        store_q7_x8_slice(out, i, w0);
        store_q7_x8_slice(out, i + 8, w1);
        i += 16;
    }
    while i + 8 <= len {
        let w = swar_q7_sub_sat_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        store_q7_x8_slice(out, i, w);
        i += 8;
    }
    while i < len {
        out[i] = branchless_q7_sub_sat(a[i], b[i]);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_circular_diff_q7_slices(
    a: &[gcso_q7_t],
    b: &[gcso_q7_t],
    out: &mut [gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` (`32` or `64` fast paths) are checked before each 32B load/store.
        unsafe {
            use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_sub_epi8};
            let ap = a.as_ptr().cast::<__m256i>();
            let bp = b.as_ptr().cast::<__m256i>();
            let op = out.as_mut_ptr().cast::<__m256i>();
            if len == MAX_ATTENTION_HEADS {
                let va0 = _mm256_loadu_si256(ap);
                let vb0 = _mm256_loadu_si256(bp);
                let va1 = _mm256_loadu_si256(ap.add(1));
                let vb1 = _mm256_loadu_si256(bp.add(1));
                _mm256_storeu_si256(op, _mm256_sub_epi8(va0, vb0));
                _mm256_storeu_si256(op.add(1), _mm256_sub_epi8(va1, vb1));
                return;
            }
            if len == 32 {
                let va0 = _mm256_loadu_si256(ap);
                let vb0 = _mm256_loadu_si256(bp);
                _mm256_storeu_si256(op, _mm256_sub_epi8(va0, vb0));
                return;
            }
            while i + 32 <= len {
                let off = i >> 5;
                let va = _mm256_loadu_si256(ap.add(off));
                let vb = _mm256_loadu_si256(bp.add(off));
                _mm256_storeu_si256(op.add(off), _mm256_sub_epi8(va, vb));
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let w0 = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        let w1 = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i + 8), load_q7_x8_slice(b, i + 8));
        store_q7_x8_slice(out, i, w0);
        store_q7_x8_slice(out, i + 8, w1);
        i += 16;
    }
    while i + 8 <= len {
        let w = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        store_q7_x8_slice(out, i, w);
        i += 8;
    }
    while i < len {
        out[i] = branchless_q7_circular_diff(a[i], b[i]);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_circular_diff_and_fold_geodesic_q7_slices(
    a: &[gcso_q7_t],
    b: &[gcso_q7_t],
    out: &mut [gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` (`32` or `64` fast paths) are checked before each 32B load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_add_epi8, _mm256_blendv_epi8, _mm256_loadu_si256,
                _mm256_set1_epi8, _mm256_storeu_si256, _mm256_sub_epi8, _mm256_xor_si256,
            };
            let ap = a.as_ptr().cast::<__m256i>();
            let bp = b.as_ptr().cast::<__m256i>();
            let op = out.as_mut_ptr().cast::<__m256i>();
            let v_128 = _mm256_set1_epi8(-128);
            if len == MAX_ATTENTION_HEADS {
                let vd0 = _mm256_sub_epi8(_mm256_loadu_si256(ap), _mm256_loadu_si256(bp));
                let vd1 = _mm256_sub_epi8(_mm256_loadu_si256(ap.add(1)), _mm256_loadu_si256(bp.add(1)));
                let m0 = _mm256_xor_si256(vd0, _mm256_add_epi8(vd0, vd0));
                let m1 = _mm256_xor_si256(vd1, _mm256_add_epi8(vd1, vd1));
                _mm256_storeu_si256(op, _mm256_blendv_epi8(vd0, _mm256_sub_epi8(v_128, vd0), m0));
                _mm256_storeu_si256(op.add(1), _mm256_blendv_epi8(vd1, _mm256_sub_epi8(v_128, vd1), m1));
                return;
            }
            if len == 32 {
                let vd0 = _mm256_sub_epi8(_mm256_loadu_si256(ap), _mm256_loadu_si256(bp));
                let m0 = _mm256_xor_si256(vd0, _mm256_add_epi8(vd0, vd0));
                _mm256_storeu_si256(op, _mm256_blendv_epi8(vd0, _mm256_sub_epi8(v_128, vd0), m0));
                return;
            }
            while i + 32 <= len {
                let off = i >> 5;
                let vd = _mm256_sub_epi8(_mm256_loadu_si256(ap.add(off)), _mm256_loadu_si256(bp.add(off)));
                let mask = _mm256_xor_si256(vd, _mm256_add_epi8(vd, vd));
                let folded = _mm256_blendv_epi8(vd, _mm256_sub_epi8(v_128, vd), mask);
                _mm256_storeu_si256(op.add(off), folded);
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let diff_w0 = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        let diff_w1 =
            swar_q7_circular_diff_x8(load_q7_x8_slice(a, i + 8), load_q7_x8_slice(b, i + 8));
        store_q7_x8_slice(out, i, swar_q7_fold_s1_geodesic_x8(diff_w0));
        store_q7_x8_slice(out, i + 8, swar_q7_fold_s1_geodesic_x8(diff_w1));
        i += 16;
    }
    while i + 8 <= len {
        let diff_w = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        store_q7_x8_slice(out, i, swar_q7_fold_s1_geodesic_x8(diff_w));
        i += 8;
    }
    while i < len {
        out[i] = branchless_q7_fold_s1_geodesic(branchless_q7_circular_diff(a[i], b[i]));
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_circular_add_and_fold_geodesic_q7_slice_in_place(
    bus: &mut [gcso_q7_t],
    loc: &[gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` (`32` or `64` fast paths) are checked before each 32B load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_add_epi8, _mm256_blendv_epi8, _mm256_loadu_si256,
                _mm256_set1_epi8, _mm256_storeu_si256, _mm256_sub_epi8, _mm256_xor_si256,
            };
            let bp = bus.as_mut_ptr().cast::<__m256i>();
            let lp = loc.as_ptr().cast::<__m256i>();
            let v_128 = _mm256_set1_epi8(-128);
            if len == MAX_ATTENTION_HEADS {
                let vs0 = _mm256_add_epi8(_mm256_loadu_si256(bp), _mm256_loadu_si256(lp));
                let vs1 = _mm256_add_epi8(_mm256_loadu_si256(bp.add(1)), _mm256_loadu_si256(lp.add(1)));
                let m0 = _mm256_xor_si256(vs0, _mm256_add_epi8(vs0, vs0));
                let m1 = _mm256_xor_si256(vs1, _mm256_add_epi8(vs1, vs1));
                _mm256_storeu_si256(bp, _mm256_blendv_epi8(vs0, _mm256_sub_epi8(v_128, vs0), m0));
                _mm256_storeu_si256(bp.add(1), _mm256_blendv_epi8(vs1, _mm256_sub_epi8(v_128, vs1), m1));
                return;
            }
            if len == 32 {
                let vs0 = _mm256_add_epi8(_mm256_loadu_si256(bp), _mm256_loadu_si256(lp));
                let m0 = _mm256_xor_si256(vs0, _mm256_add_epi8(vs0, vs0));
                _mm256_storeu_si256(bp, _mm256_blendv_epi8(vs0, _mm256_sub_epi8(v_128, vs0), m0));
                return;
            }
            while i + 32 <= len {
                let off = i >> 5;
                let vs = _mm256_add_epi8(
                    _mm256_loadu_si256(bp.add(off)),
                    _mm256_loadu_si256(lp.add(off)),
                );
                let mask = _mm256_xor_si256(vs, _mm256_add_epi8(vs, vs));
                let folded = _mm256_blendv_epi8(vs, _mm256_sub_epi8(v_128, vs), mask);
                _mm256_storeu_si256(bp.add(off), folded);
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let b0 = load_q7_x8_slice(bus, i);
        let l0 = load_q7_x8_slice(loc, i);
        let b1 = load_q7_x8_slice(bus, i + 8);
        let l1 = load_q7_x8_slice(loc, i + 8);
        store_q7_x8_slice(
            bus,
            i,
            swar_q7_fold_s1_geodesic_x8(swar_q7_circular_add_x8(b0, l0)),
        );
        store_q7_x8_slice(
            bus,
            i + 8,
            swar_q7_fold_s1_geodesic_x8(swar_q7_circular_add_x8(b1, l1)),
        );
        i += 16;
    }
    while i + 8 <= len {
        let bw = load_q7_x8_slice(bus, i);
        let lw = load_q7_x8_slice(loc, i);
        store_q7_x8_slice(
            bus,
            i,
            swar_q7_fold_s1_geodesic_x8(swar_q7_circular_add_x8(bw, lw)),
        );
        i += 8;
    }
    while i < len {
        bus[i] = branchless_q7_fold_s1_geodesic(bus[i].wrapping_add(loc[i]));
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_fold_s1_geodesic_q7_slice_in_place(phases: &mut [gcso_q7_t], limit: usize) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= limit` (`limit <= phases.len()`) are checked before each 32B load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_add_epi8, _mm256_blendv_epi8, _mm256_loadu_si256,
                _mm256_set1_epi8, _mm256_storeu_si256, _mm256_sub_epi8, _mm256_xor_si256,
            };
            let pp = phases.as_mut_ptr().cast::<__m256i>();
            let v_128 = _mm256_set1_epi8(-128);
            if limit == MAX_ATTENTION_HEADS {
                let v0 = _mm256_loadu_si256(pp);
                let v1 = _mm256_loadu_si256(pp.add(1));
                let m0 = _mm256_xor_si256(v0, _mm256_add_epi8(v0, v0));
                let m1 = _mm256_xor_si256(v1, _mm256_add_epi8(v1, v1));
                _mm256_storeu_si256(pp, _mm256_blendv_epi8(v0, _mm256_sub_epi8(v_128, v0), m0));
                _mm256_storeu_si256(pp.add(1), _mm256_blendv_epi8(v1, _mm256_sub_epi8(v_128, v1), m1));
                return;
            }
            if limit == 32 {
                let v0 = _mm256_loadu_si256(pp);
                let m0 = _mm256_xor_si256(v0, _mm256_add_epi8(v0, v0));
                _mm256_storeu_si256(pp, _mm256_blendv_epi8(v0, _mm256_sub_epi8(v_128, v0), m0));
                return;
            }
            while i + 32 <= limit {
                let off = i >> 5;
                let v = _mm256_loadu_si256(pp.add(off));
                let mask = _mm256_xor_si256(v, _mm256_add_epi8(v, v));
                let folded = _mm256_blendv_epi8(v, _mm256_sub_epi8(v_128, v), mask);
                _mm256_storeu_si256(pp.add(off), folded);
                i += 32;
            }
        }
    }
    while i + 16 <= limit {
        let w0 = load_q7_x8_slice(phases, i);
        let w1 = load_q7_x8_slice(phases, i + 8);
        store_q7_x8_slice(phases, i, swar_q7_fold_s1_geodesic_x8(w0));
        store_q7_x8_slice(phases, i + 8, swar_q7_fold_s1_geodesic_x8(w1));
        i += 16;
    }
    while i + 8 <= limit {
        let w = load_q7_x8_slice(phases, i);
        store_q7_x8_slice(phases, i, swar_q7_fold_s1_geodesic_x8(w));
        i += 8;
    }
    while i < limit {
        phases[i] = branchless_q7_fold_s1_geodesic(phases[i]);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_qdps_filter_q7_slice_in_place(
    phases: &mut [gcso_q7_t],
    limit: usize,
    min_step_q7: i32,
) {
    let clamped_step = branchless_select_i32(
        min_step_q7 > 128,
        128,
        branchless_select_i32(min_step_q7 < 0, 0, min_step_q7),
    ) as u8;
    // Since `|x| >= 1` for every non-zero `gcso_q7_t` lane (`|0| = 0`), `clamped_step <= 1` is an exact identity.
    if clamped_step <= 1 {
        return;
    }
    let end = branchless_select_usize(phases.len() < limit, phases.len(), limit);
    if clamped_step > 127 {
        phases[..end].fill(0);
        return;
    }
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `i + 32 <= end <= phases.len()` is checked before each 32-byte unaligned load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_abs_epi8, _mm256_and_si256, _mm256_cmpeq_epi8,
                _mm256_loadu_si256, _mm256_max_epu8, _mm256_set1_epi8, _mm256_storeu_si256,
            };
            let pp = phases.as_mut_ptr().cast::<__m256i>();
            let v_thresh = _mm256_set1_epi8(clamped_step as i8);
            if end == MAX_ATTENTION_HEADS {
                let v0 = _mm256_loadu_si256(pp);
                let v1 = _mm256_loadu_si256(pp.add(1));
                let a0 = _mm256_abs_epi8(v0);
                let a1 = _mm256_abs_epi8(v1);
                let m0 = _mm256_cmpeq_epi8(_mm256_max_epu8(a0, v_thresh), a0);
                let m1 = _mm256_cmpeq_epi8(_mm256_max_epu8(a1, v_thresh), a1);
                _mm256_storeu_si256(pp, _mm256_and_si256(v0, m0));
                _mm256_storeu_si256(pp.add(1), _mm256_and_si256(v1, m1));
                return;
            }
            if end == 32 {
                let v0 = _mm256_loadu_si256(pp);
                let a0 = _mm256_abs_epi8(v0);
                let m0 = _mm256_cmpeq_epi8(_mm256_max_epu8(a0, v_thresh), a0);
                _mm256_storeu_si256(pp, _mm256_and_si256(v0, m0));
                return;
            }
            while i + 32 <= end {
                let off = i >> 5;
                let v0 = _mm256_loadu_si256(pp.add(off));
                let a0 = _mm256_abs_epi8(v0);
                let m0 = _mm256_cmpeq_epi8(_mm256_max_epu8(a0, v_thresh), a0);
                _mm256_storeu_si256(pp.add(off), _mm256_and_si256(v0, m0));
                i += 32;
            }
        }
    }
    let thresh_x8 = ((clamped_step & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    if i == 0 && end == MAX_ATTENTION_HEADS {
        // SAFETY: `end == MAX_ATTENTION_HEADS <= phases.len()` and `align_of::<gcso_q7_t>() == 1`.
        let arr = unsafe { &mut *phases.as_mut_ptr().cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        let w = load_q7_words_64(arr);
        store_q7_words_64(
            arr,
            &[
                swar_q7_qdps_filter_with_thresh_x8(w[0], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[1], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[2], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[3], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[4], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[5], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[6], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[7], thresh_x8, u64::MAX),
            ],
        );
        return;
    }
    while i + 16 <= end {
        let w0 = load_q7_x8_slice(phases, i);
        let w1 = load_q7_x8_slice(phases, i + 8);
        store_q7_x8_slice(
            phases,
            i,
            swar_q7_qdps_filter_with_thresh_x8(w0, thresh_x8, u64::MAX),
        );
        store_q7_x8_slice(
            phases,
            i + 8,
            swar_q7_qdps_filter_with_thresh_x8(w1, thresh_x8, u64::MAX),
        );
        i += 16;
    }
    while i + 8 <= end {
        let w = load_q7_x8_slice(phases, i);
        store_q7_x8_slice(
            phases,
            i,
            swar_q7_qdps_filter_with_thresh_x8(w, thresh_x8, u64::MAX),
        );
        i += 8;
    }
    while i < end {
        phases[i] = branchless_qdps_filter_q7(phases[i], i32::from(clamped_step));
        i += 1;
    }
}

#[inline(always)]
pub(crate) const fn sanitize_inplace_gain_f32(sample_elem: f32, raw_gain: f32) -> f32 {
    let valid_gain = branchless_is_finite_f32(raw_gain);
    let clean_gain = branchless_select_f32(valid_gain, raw_gain, 1.0);
    let snapped_gain = branchless_select_f32(
        branchless_abs_f32(clean_gain - 1.0) > 1.0e-7,
        clean_gain,
        1.0,
    );
    let abs_sample = branchless_abs_f32(sample_elem);
    let abs_gain = branchless_abs_f32(snapped_gain);
    let subnormal_sample_trap =
        (abs_sample > 0.0) & (abs_sample < F32_MIN_POSITIVE_NORMAL) & (abs_gain < 1.0);
    let overflow_trap = (abs_sample > INPLACE_GAIN_OVERFLOW_CEIL) & (abs_gain > 1.0);
    branchless_select_f32(subnormal_sample_trap | overflow_trap, 1.0, snapped_gain)
}

#[inline(always)]
pub(crate) const fn eval_ripa_sincos_q7_fast(
    phase_q7: gcso_q7_t,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if is_default {
        branchless_sincos_q7_ripa_default_lut(phase_q7)
    } else {
        branchless_sincos_q7_ripa_preinv(phase_q7, eff_max, q7_to_norm_scale)
    }
}

#[inline(always)]
pub(crate) const fn eval_geodesic_ripa_sincos_q7_fast(
    phase_q7: gcso_q7_t,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if is_default {
        branchless_sincos_q7_geodesic_ripa_default_lut(phase_q7)
    } else {
        let folded = branchless_q7_fold_s1_geodesic(phase_q7);
        branchless_sincos_q7_ripa_preinv(folded, eff_max, q7_to_norm_scale)
    }
}

/// Evaluates the $SO(2)$-composed $(\sin(\theta_{\mathrm{base}} + \theta_{\mathrm{pspm}}), \cos(\theta_{\mathrm{base}} + \theta_{\mathrm{pspm}}))$
/// for a single attention head, using two 1-cycle LUT loads when both RIPA clamps equal
/// [`DEFAULT_RIPA_CLAMP_MAX_RAD`], or summing the two `tanh`-clamped radian angles and evaluating
/// `sincos_ripa_poly` **once** instead of twice when custom RIPA clamps are active.
#[inline(always)]
pub(crate) const fn eval_fused_ripa_pspm_sincos_q7_fast(
    base_q7: gcso_q7_t,
    geodesic_base: bool,
    is_default_base: bool,
    eff_base_max: f32,
    base_norm_scale: f32,
    pspm_q7: gcso_q7_t,
    geodesic_pspm: bool,
    is_default_pspm: bool,
    eff_pspm_max: f32,
    pspm_norm_scale: f32,
) -> (f32, f32) {
    let b_q7 = if geodesic_base {
        branchless_q7_fold_s1_geodesic(base_q7)
    } else {
        base_q7
    };
    let p_q7 = if geodesic_pspm {
        branchless_q7_fold_s1_geodesic(pspm_q7)
    } else {
        pspm_q7
    };
    if is_default_base & is_default_pspm {
        if b_q7 == p_q7 {
            return Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF[(b_q7 as u8) as usize];
        }
        if (base_q7 == pspm_q7) & (!geodesic_base) & geodesic_pspm {
            return Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF[(base_q7 as u8) as usize];
        }
        let lut = Q7_DEFAULT_RIPA_SINCOS_REF;
        let (s_b, c_b) = lut[(b_q7 as u8) as usize];
        let (s_p, c_p) = lut[(p_q7 as u8) as usize];
        return (s_b * c_p + c_b * s_p, c_b * c_p - s_b * s_p);
    }
    if (b_q7 == p_q7) & (eff_base_max.to_bits() == eff_pspm_max.to_bits()) {
        if b_q7 == 0 {
            return (0.0, 1.0);
        }
        let theta_comb =
            eval_ripa_clamped_theta_rad_preinv(b_q7, eff_base_max + eff_base_max, base_norm_scale);
        let abs_comb = branchless_abs_f32(theta_comb);
        return if abs_comb <= FRAC_PI_2 {
            sincos_ripa_poly(theta_comb)
        } else {
            sincos_reduced_pi(theta_comb)
        };
    }
    let theta_b = if b_q7 == 0 {
        0.0
    } else {
        eval_ripa_clamped_theta_rad_preinv(b_q7, eff_base_max, base_norm_scale)
    };
    let theta_p = if p_q7 == 0 {
        0.0
    } else {
        eval_ripa_clamped_theta_rad_preinv(p_q7, eff_pspm_max, pspm_norm_scale)
    };
    let theta_comb = theta_b + theta_p;
    let abs_comb = branchless_abs_f32(theta_comb);
    if abs_comb <= FRAC_PI_2 {
        sincos_ripa_poly(theta_comb)
    } else {
        sincos_reduced_pi(theta_comb)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_hsum_ps(v: __m256) -> f32 {
    // SAFETY: Caller guarantees AVX2 is enabled.
    unsafe {
        let v128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let v64 = _mm_add_ps(v128, _mm_movehl_ps(v128, v128));
        _mm_cvtss_f32(_mm_add_ss(v64, _mm_shuffle_ps::<0x55>(v64, v64)))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_hsum2_ps(v0: __m256, v1: __m256) -> (f32, f32) {
    // SAFETY: Caller guarantees AVX2 is enabled.
    unsafe {
        let s0 = _mm_add_ps(_mm256_castps256_ps128(v0), _mm256_extractf128_ps::<1>(v0));
        let s1 = _mm_add_ps(_mm256_castps256_ps128(v1), _mm256_extractf128_ps::<1>(v1));
        let p64 = _mm_add_ps(_mm_unpacklo_ps(s0, s1), _mm_unpackhi_ps(s0, s1));
        let p32 = _mm_add_ps(p64, _mm_movehl_ps(p64, p64));
        (
            _mm_cvtss_f32(p32),
            _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(p32, p32)),
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_hsum3_ps(v0: __m256, v1: __m256, v2: __m256) -> (f32, f32, f32) {
    // SAFETY: Caller guarantees AVX2 is enabled.
    unsafe {
        let (s0, s1) = avx2_hsum2_ps(v0, v1);
        (s0, s1, avx2_hsum_ps(v2))
    }
}

/// Evaluates $(\sum_i |x_i|, \sum_i x_i^2)$ across `slice` using a 32-float (2-pair tree-reduced)
/// AVX2 SIMD kernel that computes `a = |x|` once per vector and uses $a^2 \equiv x^2$ to eliminate
/// Port 0/1 accumulator contention.
#[inline(always)]
pub fn sum_abs_and_sq_f32_simd8(slice: &[f32]) -> (f32, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Peak live YMM usage is capped at 5 registers (`sign_mask`, `abs_all`, `sq_all`, `a0`, `a1` <= 6)
        // to avoid non-volatile `xmm6..xmm15` stack saves on Windows x64 ABI, and remainder elements
        // use loop-free binary tail steps (`32`, `16`, `8`, `4`, `2`, `1`).
        unsafe {
            let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
            let mut abs_all = _mm256_setzero_ps();
            let mut sq_all = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                let a2 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 16)), sign_mask);
                let a3 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 24)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a2, a3));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a2, a2), _mm256_mul_ps(a3, a3)),
                );
                let a4 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 32)), sign_mask);
                let a5 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 40)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a4, a5));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a4, a4), _mm256_mul_ps(a5, a5)),
                );
                let a6 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 48)), sign_mask);
                let a7 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 56)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a6, a7));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a6, a6), _mm256_mul_ps(a7, a7)),
                );
                i += 64;
            }
            let rem = len - i;
            if (rem & 32) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                let a2 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 16)), sign_mask);
                let a3 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 24)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a2, a3));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a2, a2), _mm256_mul_ps(a3, a3)),
                );
                i += 32;
            }
            if (rem & 16) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                i += 16;
            }
            if (rem & 8) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, a0);
                sq_all = _mm256_add_ps(sq_all, _mm256_mul_ps(a0, a0));
                i += 8;
            }
            let (mut sum_abs, mut sum_sq) = avx2_hsum2_ps(abs_all, sq_all);
            if (rem & 4) != 0 {
                let v0 = branchless_abs_f32(*ptr.add(i));
                let v1 = branchless_abs_f32(*ptr.add(i + 1));
                let v2 = branchless_abs_f32(*ptr.add(i + 2));
                let v3 = branchless_abs_f32(*ptr.add(i + 3));
                sum_abs += (v0 + v1) + (v2 + v3);
                sum_sq += (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3);
                i += 4;
            }
            if (rem & 2) != 0 {
                let v0 = branchless_abs_f32(*ptr.add(i));
                let v1 = branchless_abs_f32(*ptr.add(i + 1));
                sum_abs += v0 + v1;
                sum_sq += v0 * v0 + v1 * v1;
                i += 2;
            }
            if (rem & 1) != 0 {
                let v0 = branchless_abs_f32(*ptr.add(i));
                sum_abs += v0;
                sum_sq += v0 * v0;
            }
            (sum_abs, sum_sq)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut acc_abs = [0.0_f32; 4];
        let mut acc_sq = [0.0_f32; 4];
        let mut chunks = slice.chunks_exact(4);
        for c in chunks.by_ref() {
            for lane in 0..4 {
                let v = branchless_abs_f32(c[lane]);
                acc_abs[lane] += v;
                acc_sq[lane] += v * v;
            }
        }
        let mut sum_abs = (acc_abs[0] + acc_abs[1]) + (acc_abs[2] + acc_abs[3]);
        let mut sum_sq = (acc_sq[0] + acc_sq[1]) + (acc_sq[2] + acc_sq[3]);
        for &v in chunks.remainder() {
            let a = branchless_abs_f32(v);
            sum_abs += a;
            sum_sq += a * a;
        }
        (sum_abs, sum_sq)
    }
}

/// Evaluates $(\sum_i |x_i|, \sum_i x_i^2, \max_i |x_i|)$ in a **single memory pass** across `slice`
/// using 32-float unrolled AVX2 SIMD accumulators while keeping peak live YMM usage `<= 6`
/// (`0` Windows x64 `xmm6..xmm15` stack saves/restores).
#[inline(always)]
pub fn sum_abs_sq_and_max_f32_simd8(slice: &[f32]) -> (f32, f32, f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Processing `a0, a1` then `a2, a3` into 3 accumulators (`abs_all, sq_all, max_all`) keeps peak
        // live YMM registers `<= 6` (volatile `ymm0..ymm5`), eliminating 5 `vmovaps` stack spills on Windows x64.
        unsafe {
            let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
            let mut abs_all = _mm256_setzero_ps();
            let mut sq_all = _mm256_setzero_ps();
            let mut max_all = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                max_all = _mm256_max_ps(max_all, _mm256_max_ps(a0, a1));

                let a2 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 16)), sign_mask);
                let a3 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 24)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a2, a3));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a2, a2), _mm256_mul_ps(a3, a3)),
                );
                max_all = _mm256_max_ps(max_all, _mm256_max_ps(a2, a3));
                i += 32;
            }
            let rem = len - i;
            if (rem & 16) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                let a1 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i + 8)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, _mm256_add_ps(a0, a1));
                sq_all = _mm256_add_ps(
                    sq_all,
                    _mm256_add_ps(_mm256_mul_ps(a0, a0), _mm256_mul_ps(a1, a1)),
                );
                max_all = _mm256_max_ps(max_all, _mm256_max_ps(a0, a1));
                i += 16;
            }
            if (rem & 8) != 0 {
                let a0 = _mm256_and_ps(_mm256_loadu_ps(ptr.add(i)), sign_mask);
                abs_all = _mm256_add_ps(abs_all, a0);
                sq_all = _mm256_add_ps(sq_all, _mm256_mul_ps(a0, a0));
                max_all = _mm256_max_ps(max_all, a0);
                i += 8;
            }
            let (mut sum_abs, mut sum_sq) = avx2_hsum2_ps(abs_all, sq_all);
            let m128 = _mm_max_ps(
                _mm256_castps256_ps128(max_all),
                _mm256_extractf128_ps::<1>(max_all),
            );
            let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            let mut max_abs = _mm_cvtss_f32(_mm_max_ss(m64, _mm_shuffle_ps::<0x55>(m64, m64)));
            while i < len {
                let a = branchless_abs_f32(*ptr.add(i));
                sum_abs += a;
                sum_sq += a * a;
                max_abs = branchless_max_f32(max_abs, a);
                i += 1;
            }
            (sum_abs, sum_sq, max_abs)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut sum_abs = 0.0_f32;
        let mut sum_sq = 0.0_f32;
        let mut max_abs = 0.0_f32;
        for &v in slice {
            let a = branchless_abs_f32(v);
            sum_abs += a;
            sum_sq += a * a;
            max_abs = branchless_max_f32(max_abs, a);
        }
        (sum_abs, sum_sq, max_abs)
    }
}

/// Evaluates $\sum_i x_i^2$ across `slice` using a 32-float (4-accumulator) unrolled AVX2 tree reduction.
#[inline(always)]
pub fn sum_sq_f32_simd8(slice: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Peak live YMM usage is capped at 4 registers (`sq0`, `sq1`, `v0`, `v1` <= 6) with loop-free
        // binary tail steps (`32`, `16`, `8`, `4`, `2`, `1`).
        unsafe {
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v0, v0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v1, v1));
                let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v2, v2));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v3, v3));
                let v4 = _mm256_loadu_ps(ptr.add(i + 32));
                let v5 = _mm256_loadu_ps(ptr.add(i + 40));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v4, v4));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v5, v5));
                let v6 = _mm256_loadu_ps(ptr.add(i + 48));
                let v7 = _mm256_loadu_ps(ptr.add(i + 56));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v6, v6));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v7, v7));
                i += 64;
            }
            let rem = len - i;
            if (rem & 32) != 0 {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v0, v0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v1, v1));
                let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v2, v2));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v3, v3));
                i += 32;
            }
            if (rem & 16) != 0 {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v0, v0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v1, v1));
                i += 16;
            }
            let mut sq_all = _mm256_add_ps(sq0, sq1);
            if (rem & 8) != 0 {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                sq_all = _mm256_add_ps(sq_all, _mm256_mul_ps(v0, v0));
                i += 8;
            }
            let mut sum_sq = avx2_hsum_ps(sq_all);
            if (rem & 4) != 0 {
                let v0 = *ptr.add(i);
                let v1 = *ptr.add(i + 1);
                let v2 = *ptr.add(i + 2);
                let v3 = *ptr.add(i + 3);
                sum_sq += (v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3);
                i += 4;
            }
            if (rem & 2) != 0 {
                let v0 = *ptr.add(i);
                let v1 = *ptr.add(i + 1);
                sum_sq += v0 * v0 + v1 * v1;
                i += 2;
            }
            if (rem & 1) != 0 {
                let v0 = *ptr.add(i);
                sum_sq += v0 * v0;
            }
            sum_sq
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        sum_sq_scaled_f32_simd8(slice, 1.0)
    }
}

/// Evaluates $\sum_i (|x_i| \cdot 2^{126})^2$ across a subnormal-dominated `slice` (`|x_i| < F32_MIN_POSITIVE_NORMAL`)
/// by lifting the 23-bit mantissa directly in IEEE-754 integer SIMD (`(bits & 0x007F_FFFF) | 0x3F80_0000`
/// minus `1.0_f32`), completely bypassing x86 Denormal-Operand microcode assists.
#[inline(always)]
fn sum_sq_subnormal_mantissa_f32_simd8(slice: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        unsafe {
            let mant_mask = _mm256_set1_epi32(0x007F_FFFF);
            let exp_mask = _mm256_set1_epi32(0x7F80_0000);
            let one_bits = _mm256_set1_epi32(0x3F80_0000);
            let v_one = _mm256_set1_ps(1.0_f32);
            let zero_i = _mm256_setzero_si256();
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 16 <= len {
                let p = slice.as_ptr().add(i).cast::<__m256i>();
                let r0 = _mm256_loadu_si256(p);
                let r1 = _mm256_loadu_si256(p.add(1));
                let is_sub0 = _mm256_cmpeq_epi32(_mm256_and_si256(r0, exp_mask), zero_i);
                let is_sub1 = _mm256_cmpeq_epi32(_mm256_and_si256(r1, exp_mask), zero_i);
                let m0 = _mm256_and_si256(_mm256_and_si256(r0, mant_mask), is_sub0);
                let m1 = _mm256_and_si256(_mm256_and_si256(r1, mant_mask), is_sub1);
                let u0 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(m0, one_bits)), v_one);
                let u1 = _mm256_sub_ps(_mm256_castsi256_ps(_mm256_or_si256(m1, one_bits)), v_one);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(u0, u0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(u1, u1));
                i += 16;
            }
            let mut sum_sq = avx2_hsum_ps(_mm256_add_ps(sq0, sq1));
            while i < len {
                let bits = (*slice.as_ptr().add(i)).to_bits();
                let is_sub = (bits & 0x7F80_0000) == 0;
                let mant = branchless_select_u32(is_sub, bits & 0x007F_FFFF, 0);
                let u = f32::from_bits(mant | 0x3F80_0000) - 1.0_f32;
                sum_sq += u * u;
                i += 1;
            }
            sum_sq
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut sum_sq = 0.0_f32;
        for &x in slice {
            let bits = x.to_bits();
            let is_sub = (bits & 0x7F80_0000) == 0;
            let mant = branchless_select_u32(is_sub, bits & 0x007F_FFFF, 0);
            let u = f32::from_bits(mant | 0x3F80_0000) - 1.0_f32;
            sum_sq += u * u;
        }
        sum_sq
    }
}

/// Evaluates $\sum_i (x_i \cdot \mathrm{scale})^2$ across `slice` using a 64-float unrolled AVX2 tree reduction.
#[inline(always)]
pub(crate) fn sum_sq_scaled_f32_simd8(slice: &[f32], scale: f32) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Note that `(x * scale)^2 = x^2 * scale^2`, but since `x^2` may underflow on near-zero inputs,
        // we scale `x` first across 64-float unrolled blocks with binary tail steps (`32`, `16`, `8`).
        unsafe {
            let v_scale = _mm256_set1_ps(scale);
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_scale);
                let v1 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_scale);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v0, v0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v1, v1));
                let v2 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 16)), v_scale);
                let v3 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 24)), v_scale);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v2, v2));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v3, v3));
                let v4 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 32)), v_scale);
                let v5 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 40)), v_scale);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v4, v4));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v5, v5));
                let v6 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 48)), v_scale);
                let v7 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 56)), v_scale);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v6, v6));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v7, v7));
                i += 64;
            }
            let rem = len - i;
            if (rem & 32) != 0 {
                let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_scale);
                let v1 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_scale);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v0, v0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v1, v1));
                let v2 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 16)), v_scale);
                let v3 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 24)), v_scale);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v2, v2));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v3, v3));
                i += 32;
            }
            if (rem & 16) != 0 {
                let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_scale);
                let v1 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i + 8)), v_scale);
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(v0, v0));
                sq1 = _mm256_add_ps(sq1, _mm256_mul_ps(v1, v1));
                i += 16;
            }
            let mut sq_all = _mm256_add_ps(sq0, sq1);
            if (rem & 8) != 0 {
                let v0 = _mm256_mul_ps(_mm256_loadu_ps(ptr.add(i)), v_scale);
                sq_all = _mm256_add_ps(sq_all, _mm256_mul_ps(v0, v0));
                i += 8;
            }
            let mut sum_sq = avx2_hsum_ps(sq_all);
            while i < len {
                let bits = (*ptr.add(i)).to_bits();
                let clean_x = branchless_select_f32((bits & 0x7F80_0000) != 0, *ptr.add(i), 0.0);
                let v = clean_x * scale;
                sum_sq += v * v;
                i += 1;
            }
            sum_sq
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut acc_sq = [0.0_f32; 4];
        let mut chunks = slice.chunks_exact(4);
        for c in chunks.by_ref() {
            for lane in 0..4 {
                let clean_x = branchless_flush_subnormal_f32(c[lane]);
                let v = clean_x * scale;
                acc_sq[lane] += v * v;
            }
        }
        let mut sum_sq = (acc_sq[0] + acc_sq[1]) + (acc_sq[2] + acc_sq[3]);
        for &x in chunks.remainder() {
            let v = branchless_flush_subnormal_f32(x) * scale;
            sum_sq += v * v;
        }
        sum_sq
    }
}

/// Evaluates $\max(0, \max_i x_i)$ across `slice` using a 32-float (4-YMM) unrolled AVX2 SIMD reduction.
#[inline(always)]
pub(crate) fn max_pos_f32_unrolled4(slice: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = slice.len();
        let ptr = slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads are bounds-checked against `len`.
        // Staged 2-vector pairs keep peak live YMMs <= 6 (`ymm0..ymm5`, `0` Windows x64 `xmm6..xmm15`
        // stack saves) while `_CMP_LT_OQ(v & 0x7F80_0000, +Inf)` stays 100% in the FP execution domain.
        unsafe {
            let inf_exp_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F80_0000));
            let mut m0 = _mm256_setzero_ps();
            let mut m1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let v1 = _mm256_loadu_ps(ptr.add(i + 8));
                let fin0 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v0, inf_exp_ps), inf_exp_ps);
                let fin1 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v1, inf_exp_ps), inf_exp_ps);
                m0 = _mm256_max_ps(m0, _mm256_and_ps(v0, fin0));
                m1 = _mm256_max_ps(m1, _mm256_and_ps(v1, fin1));

                let v2 = _mm256_loadu_ps(ptr.add(i + 16));
                let v3 = _mm256_loadu_ps(ptr.add(i + 24));
                let fin2 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v2, inf_exp_ps), inf_exp_ps);
                let fin3 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v3, inf_exp_ps), inf_exp_ps);
                m0 = _mm256_max_ps(m0, _mm256_and_ps(v2, fin2));
                m1 = _mm256_max_ps(m1, _mm256_and_ps(v3, fin3));
                i += 32;
            }
            while i + 8 <= len {
                let v0 = _mm256_loadu_ps(ptr.add(i));
                let fin0 = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v0, inf_exp_ps), inf_exp_ps);
                m0 = _mm256_max_ps(m0, _mm256_and_ps(v0, fin0));
                i += 8;
            }
            let mall = _mm256_max_ps(m0, m1);
            let m128 = _mm_max_ps(
                _mm256_castps256_ps128(mall),
                _mm256_extractf128_ps::<1>(mall),
            );
            let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            let mut max_val = _mm_cvtss_f32(_mm_max_ss(m64, _mm_shuffle_ps::<0x55>(m64, m64)));
            while i < len {
                max_val = branchless_max_f32(max_val, branchless_relu_f32(*ptr.add(i)));
                i += 1;
            }
            max_val
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut m = [0.0_f32; 4];
        let mut chunks = slice.chunks_exact(4);
        for c in chunks.by_ref() {
            m[0] = branchless_max_f32(m[0], branchless_relu_f32(c[0]));
            m[1] = branchless_max_f32(m[1], branchless_relu_f32(c[1]));
            m[2] = branchless_max_f32(m[2], branchless_relu_f32(c[2]));
            m[3] = branchless_max_f32(m[3], branchless_relu_f32(c[3]));
        }
        let mut max_val =
            branchless_max_f32(branchless_max_f32(m[0], m[1]), branchless_max_f32(m[2], m[3]));
        for &v in chunks.remainder() {
            max_val = branchless_max_f32(max_val, branchless_relu_f32(v));
        }
        max_val
    }
}

/// Scales `buf` in-place by `scale` in strictly constant time (`0` data-dependent early-return
/// branches) while flushing any subnormal product (`|y| < F32_MIN_POSITIVE_NORMAL`) to `+0.0`
/// directly in the AVX2 FP execution domain (`<= 6` live YMMs, `0` FP-to-INT domain-crossing stalls,
/// `0` Windows x64 `xmm6..xmm15` stack saves).
#[inline(always)]
pub(crate) fn scale_f32_slice_unrolled8(buf: &mut [f32], scale: f32) {
    let len = buf.len();
    if len == 0 {
        return;
    }
    let eff_scale = branchless_flush_subnormal_f32(sanitize_inplace_gain_f32(buf[0], scale));
    if len <= 16 {
        // Match preceding 32-bit/128-bit stores in the CPU store buffer on short logit/score slices
        // (`<= 16` floats) to prevent a ~14-cycle 32-bit-store -> 256-bit-YMM-load STLF stall.
        for val in buf.iter_mut() {
            *val = branchless_flush_subnormal_f32(*val * eff_scale);
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let ptr = buf.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and pointer offsets are bounds-checked against `len`.
        // Masking `prod` with `exp_mask_ps` (`0x7F80_0000`) clears the sign and mantissa bits so any
        // normal float (`1 <= exp <= 254`), `±Inf`, or `NaN` (`exp == 255 -> +Inf`) satisfies
        // `_CMP_GT_OQ(prod & exp_mask_ps, +0.0)`, while zero and subnormals (`exp == 0`) flush to `+0.0`
        // without crossing into the Integer SIMD execution domain. Staging 2 vectors (`64B` cache line)
        // at a time keeps peak live YMMs <= 6 (`ymm0..ymm5`).
        unsafe {
            let v_scale = _mm256_set1_ps(eff_scale);
            let exp_mask_ps = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F80_0000));
            let zero_ps = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= len {
                let p0 = ptr.add(i);
                let p1 = ptr.add(i + 8);
                let prod0 = _mm256_mul_ps(_mm256_loadu_ps(p0), v_scale);
                let prod1 = _mm256_mul_ps(_mm256_loadu_ps(p1), v_scale);
                let m0 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod0, exp_mask_ps), zero_ps);
                let m1 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod1, exp_mask_ps), zero_ps);
                _mm256_storeu_ps(p0, _mm256_and_ps(prod0, m0));
                _mm256_storeu_ps(p1, _mm256_and_ps(prod1, m1));

                let p2 = ptr.add(i + 16);
                let p3 = ptr.add(i + 24);
                let prod2 = _mm256_mul_ps(_mm256_loadu_ps(p2), v_scale);
                let prod3 = _mm256_mul_ps(_mm256_loadu_ps(p3), v_scale);
                let m2 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod2, exp_mask_ps), zero_ps);
                let m3 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod3, exp_mask_ps), zero_ps);
                _mm256_storeu_ps(p2, _mm256_and_ps(prod2, m2));
                _mm256_storeu_ps(p3, _mm256_and_ps(prod3, m3));

                let p4 = ptr.add(i + 32);
                let p5 = ptr.add(i + 40);
                let prod4 = _mm256_mul_ps(_mm256_loadu_ps(p4), v_scale);
                let prod5 = _mm256_mul_ps(_mm256_loadu_ps(p5), v_scale);
                let m4 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod4, exp_mask_ps), zero_ps);
                let m5 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod5, exp_mask_ps), zero_ps);
                _mm256_storeu_ps(p4, _mm256_and_ps(prod4, m4));
                _mm256_storeu_ps(p5, _mm256_and_ps(prod5, m5));

                let p6 = ptr.add(i + 48);
                let p7 = ptr.add(i + 56);
                let prod6 = _mm256_mul_ps(_mm256_loadu_ps(p6), v_scale);
                let prod7 = _mm256_mul_ps(_mm256_loadu_ps(p7), v_scale);
                let m6 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod6, exp_mask_ps), zero_ps);
                let m7 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod7, exp_mask_ps), zero_ps);
                _mm256_storeu_ps(p6, _mm256_and_ps(prod6, m6));
                _mm256_storeu_ps(p7, _mm256_and_ps(prod7, m7));
                i += 64;
            }
            let rem = len - i;
            if rem != 0 {
                if (rem & 32) != 0 {
                    let p0 = ptr.add(i);
                    let p1 = ptr.add(i + 8);
                    let prod0 = _mm256_mul_ps(_mm256_loadu_ps(p0), v_scale);
                    let prod1 = _mm256_mul_ps(_mm256_loadu_ps(p1), v_scale);
                    let m0 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod0, exp_mask_ps), zero_ps);
                    let m1 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod1, exp_mask_ps), zero_ps);
                    _mm256_storeu_ps(p0, _mm256_and_ps(prod0, m0));
                    _mm256_storeu_ps(p1, _mm256_and_ps(prod1, m1));

                    let p2 = ptr.add(i + 16);
                    let p3 = ptr.add(i + 24);
                    let prod2 = _mm256_mul_ps(_mm256_loadu_ps(p2), v_scale);
                    let prod3 = _mm256_mul_ps(_mm256_loadu_ps(p3), v_scale);
                    let m2 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod2, exp_mask_ps), zero_ps);
                    let m3 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod3, exp_mask_ps), zero_ps);
                    _mm256_storeu_ps(p2, _mm256_and_ps(prod2, m2));
                    _mm256_storeu_ps(p3, _mm256_and_ps(prod3, m3));
                    i += 32;
                }
                if (rem & 16) != 0 {
                    let p0 = ptr.add(i);
                    let p1 = ptr.add(i + 8);
                    let prod0 = _mm256_mul_ps(_mm256_loadu_ps(p0), v_scale);
                    let prod1 = _mm256_mul_ps(_mm256_loadu_ps(p1), v_scale);
                    let m0 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod0, exp_mask_ps), zero_ps);
                    let m1 = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod1, exp_mask_ps), zero_ps);
                    _mm256_storeu_ps(p0, _mm256_and_ps(prod0, m0));
                    _mm256_storeu_ps(p1, _mm256_and_ps(prod1, m1));
                    i += 16;
                }
                if (rem & 8) != 0 {
                    let p = ptr.add(i);
                    let prod = _mm256_mul_ps(_mm256_loadu_ps(p), v_scale);
                    let m = _mm256_cmp_ps::<_CMP_GT_OQ>(_mm256_and_ps(prod, exp_mask_ps), zero_ps);
                    _mm256_storeu_ps(p, _mm256_and_ps(prod, m));
                    i += 8;
                }
                while i < len {
                    *ptr.add(i) = branchless_flush_subnormal_f32(*ptr.add(i) * eff_scale);
                    i += 1;
                }
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for val in buf.iter_mut() {
            *val = branchless_flush_subnormal_f32(*val * eff_scale);
        }
    }
}

/// Applies RIPA hyperbolic tangent soft-clamping ($\theta_{\mathrm{safe}} = \theta_{\max}\tanh(\theta / \theta_{\max})$)
/// in-place across `angles` using an 8-float AVX2 Padé $(5, 4)$ rational polynomial + `_mm256_rcp_ps`
/// 2-step Newton-Raphson reciprocal kernel (`<= 6` live YMM registers `ymm0..ymm5`, `0` Windows x64
/// `xmm6..xmm15` stack saves, and branchless `NaN`/`±Inf`/subnormal flushing in the FP domain).
#[inline(always)]
pub(crate) fn clamp_angles_ripa_in_place_simd8(angles: &mut [f32], max_rad: f32) {
    let len = angles.len();
    if len == 0 {
        return;
    }
    let inv_max = branchless_inv_pos_f32(max_rad);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let ptr = angles.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads/stores are bounds-checked against `len`.
        // Constants are materialized inline (`vmovaps`/`vbroadcastss`) so peak live YMM usage is capped
        // at 5 registers (`v`, `x`, `x2`, `x4`, `den`/`num`) <= 6 (`0` Windows x64 `xmm6..xmm15` stack saves).
        unsafe {
            let mut i = 0usize;
            while i + 8 <= len {
                let p = ptr.add(i);
                let v = _mm256_loadu_ps(p);
                let inf_exp = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F80_0000));
                let fin_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(_mm256_and_ps(v, inf_exp), inf_exp);
                let clean_v = _mm256_and_ps(v, fin_mask);
                let u = _mm256_mul_ps(clean_v, _mm256_set1_ps(inv_max));
                let x = _mm256_min_ps(
                    _mm256_max_ps(u, _mm256_set1_ps(-TANH_LUT_MAX_X)),
                    _mm256_set1_ps(TANH_LUT_MAX_X),
                );
                let x2 = _mm256_mul_ps(x, x);
                let x4 = _mm256_mul_ps(x2, x2);
                let num = _mm256_mul_ps(
                    _mm256_mul_ps(x, _mm256_set1_ps(max_rad)),
                    _mm256_add_ps(
                        _mm256_add_ps(
                            _mm256_set1_ps(945.0_f32),
                            _mm256_mul_ps(_mm256_set1_ps(105.0_f32), x2),
                        ),
                        x4,
                    ),
                );
                let den = _mm256_add_ps(
                    _mm256_add_ps(
                        _mm256_set1_ps(945.0_f32),
                        _mm256_mul_ps(_mm256_set1_ps(420.0_f32), x2),
                    ),
                    _mm256_mul_ps(_mm256_set1_ps(15.0_f32), x4),
                );
                let r0 = _mm256_rcp_ps(den);
                let two = _mm256_set1_ps(2.0_f32);
                let r1 = _mm256_mul_ps(r0, _mm256_sub_ps(two, _mm256_mul_ps(den, r0)));
                let r2 = _mm256_mul_ps(r1, _mm256_sub_ps(two, _mm256_mul_ps(den, r1)));
                let prod = _mm256_mul_ps(num, r2);
                let norm_mask = _mm256_cmp_ps::<_CMP_GT_OQ>(
                    _mm256_and_ps(prod, inf_exp),
                    _mm256_setzero_ps(),
                );
                _mm256_storeu_ps(p, _mm256_and_ps(prod, norm_mask));
                i += 8;
            }
            while i < len {
                let a = *ptr.add(i);
                let clean_a = branchless_select_f32(branchless_is_finite_f32(a), a, 0.0);
                *ptr.add(i) =
                    branchless_flush_subnormal_f32(max_rad * branchless_tanh_f32(clean_a * inv_max));
                i += 1;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for angle in angles.iter_mut() {
            let a = *angle;
            let clean_a = branchless_select_f32(branchless_is_finite_f32(a), a, 0.0);
            *angle =
                branchless_flush_subnormal_f32(max_rad * branchless_tanh_f32(clean_a * inv_max));
        }
    }
}

/// Evaluates a 3rd-order Chebyshev Polynomial Spectral Filter $\sum_{k=0}^3 c_k T_k(\tilde{\lambda})$
/// across `eigenvalues_in` into `filtered_out` using 32-float (4-YMM) unrolled AVX2 Horner evaluation.
#[inline(always)]
pub(crate) fn chebyshev_deg3_filter_slice_simd8(
    eigenvalues_in: &[f32],
    filtered_out: &mut [f32],
    inv_half_lmax: f32,
    k0: f32,
    k1: f32,
    k2: f32,
    k3: f32,
) {
    let len = eigenvalues_in.len().min(filtered_out.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pin = eigenvalues_in.as_ptr();
        let pout = filtered_out.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and all loads/stores are bounds-checked against `len`.
        unsafe {
            let v_scale = _mm256_set1_ps(inv_half_lmax);
            let v_one = _mm256_set1_ps(1.0_f32);
            let v_neg_one = _mm256_set1_ps(-1.0_f32);
            let v_k0 = _mm256_set1_ps(k0);
            let v_k1 = _mm256_set1_ps(k1);
            let v_k2 = _mm256_set1_ps(k2);
            let v_k3 = _mm256_set1_ps(k3);
            let mut i = 0usize;
            while i + 32 <= len {
                let l0 = _mm256_loadu_ps(pin.add(i));
                let l1 = _mm256_loadu_ps(pin.add(i + 8));
                let l2 = _mm256_loadu_ps(pin.add(i + 16));
                let l3 = _mm256_loadu_ps(pin.add(i + 24));
                let x0 = _mm256_min_ps(
                    _mm256_max_ps(_mm256_sub_ps(_mm256_mul_ps(l0, v_scale), v_one), v_neg_one),
                    v_one,
                );
                let x1 = _mm256_min_ps(
                    _mm256_max_ps(_mm256_sub_ps(_mm256_mul_ps(l1, v_scale), v_one), v_neg_one),
                    v_one,
                );
                let x2 = _mm256_min_ps(
                    _mm256_max_ps(_mm256_sub_ps(_mm256_mul_ps(l2, v_scale), v_one), v_neg_one),
                    v_one,
                );
                let x3 = _mm256_min_ps(
                    _mm256_max_ps(_mm256_sub_ps(_mm256_mul_ps(l3, v_scale), v_one), v_neg_one),
                    v_one,
                );
                let p0 = _mm256_add_ps(
                    v_k0,
                    _mm256_mul_ps(
                        x0,
                        _mm256_add_ps(v_k1, _mm256_mul_ps(x0, _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x0)))),
                    ),
                );
                let p1 = _mm256_add_ps(
                    v_k0,
                    _mm256_mul_ps(
                        x1,
                        _mm256_add_ps(v_k1, _mm256_mul_ps(x1, _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x1)))),
                    ),
                );
                let p2 = _mm256_add_ps(
                    v_k0,
                    _mm256_mul_ps(
                        x2,
                        _mm256_add_ps(v_k1, _mm256_mul_ps(x2, _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x2)))),
                    ),
                );
                let p3 = _mm256_add_ps(
                    v_k0,
                    _mm256_mul_ps(
                        x3,
                        _mm256_add_ps(v_k1, _mm256_mul_ps(x3, _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x3)))),
                    ),
                );
                _mm256_storeu_ps(pout.add(i), p0);
                _mm256_storeu_ps(pout.add(i + 8), p1);
                _mm256_storeu_ps(pout.add(i + 16), p2);
                _mm256_storeu_ps(pout.add(i + 24), p3);
                i += 32;
            }
            while i + 8 <= len {
                let lam = _mm256_loadu_ps(pin.add(i));
                let raw_x = _mm256_sub_ps(_mm256_mul_ps(lam, v_scale), v_one);
                let x = _mm256_min_ps(_mm256_max_ps(raw_x, v_neg_one), v_one);
                let p2 = _mm256_add_ps(v_k2, _mm256_mul_ps(v_k3, x));
                let p1 = _mm256_add_ps(v_k1, _mm256_mul_ps(x, p2));
                let p0 = _mm256_add_ps(v_k0, _mm256_mul_ps(x, p1));
                _mm256_storeu_ps(pout.add(i), p0);
                i += 8;
            }
            while i < len {
                let x = branchless_clamp_f32(*pin.add(i) * inv_half_lmax - 1.0, -1.0, 1.0);
                *pout.add(i) = k0 + x * (k1 + x * (k2 + k3 * x));
                i += 1;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut i = 0usize;
        while i + 4 <= len {
            let x0 = branchless_clamp_f32(eigenvalues_in[i] * inv_half_lmax - 1.0, -1.0, 1.0);
            let x1 = branchless_clamp_f32(eigenvalues_in[i + 1] * inv_half_lmax - 1.0, -1.0, 1.0);
            let x2 = branchless_clamp_f32(eigenvalues_in[i + 2] * inv_half_lmax - 1.0, -1.0, 1.0);
            let x3 = branchless_clamp_f32(eigenvalues_in[i + 3] * inv_half_lmax - 1.0, -1.0, 1.0);
            filtered_out[i] = k0 + x0 * (k1 + x0 * (k2 + k3 * x0));
            filtered_out[i + 1] = k0 + x1 * (k1 + x1 * (k2 + k3 * x1));
            filtered_out[i + 2] = k0 + x2 * (k1 + x2 * (k2 + k3 * x2));
            filtered_out[i + 3] = k0 + x3 * (k1 + x3 * (k2 + k3 * x3));
            i += 4;
        }
        while i < len {
            let x = branchless_clamp_f32(eigenvalues_in[i] * inv_half_lmax - 1.0, -1.0, 1.0);
            filtered_out[i] = k0 + x * (k1 + x * (k2 + k3 * x));
            i += 1;
        }
    }
}

/// Adds scalar `delta` to `buf` in-place in strictly constant time (`0` data-dependent early-return
/// branches), grouping full 64-byte cacheline loads before stores across 64-float (`256B`) chunks
/// to prevent same-cacheline Load-Store Queue RFO stalls on large 32K+ vocabulary logit slices.
#[inline(always)]
pub(crate) fn add_scalar_f32_slice_unrolled8(buf: &mut [f32], delta: f32) {
    let len = buf.len();
    if len == 0 {
        return;
    }
    let clean_delta = branchless_flush_subnormal_f32(delta);
    if len <= 16 {
        for val in buf.iter_mut() {
            *val += clean_delta;
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let ptr = buf.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and pointer offsets are bounds-checked against `len`.
        unsafe {
            let v_delta = _mm256_set1_ps(clean_delta);
            let mut i = 0usize;
            while i + 64 <= len {
                let p0 = ptr.add(i);
                let p1 = ptr.add(i + 8);
                let p2 = ptr.add(i + 16);
                let p3 = ptr.add(i + 24);
                let x0 = _mm256_loadu_ps(p0);
                let x1 = _mm256_loadu_ps(p1);
                let x2 = _mm256_loadu_ps(p2);
                let x3 = _mm256_loadu_ps(p3);
                _mm256_storeu_ps(p0, _mm256_add_ps(x0, v_delta));
                _mm256_storeu_ps(p1, _mm256_add_ps(x1, v_delta));
                _mm256_storeu_ps(p2, _mm256_add_ps(x2, v_delta));
                _mm256_storeu_ps(p3, _mm256_add_ps(x3, v_delta));

                let p4 = ptr.add(i + 32);
                let p5 = ptr.add(i + 40);
                let p6 = ptr.add(i + 48);
                let p7 = ptr.add(i + 56);
                let x4 = _mm256_loadu_ps(p4);
                let x5 = _mm256_loadu_ps(p5);
                let x6 = _mm256_loadu_ps(p6);
                let x7 = _mm256_loadu_ps(p7);
                _mm256_storeu_ps(p4, _mm256_add_ps(x4, v_delta));
                _mm256_storeu_ps(p5, _mm256_add_ps(x5, v_delta));
                _mm256_storeu_ps(p6, _mm256_add_ps(x6, v_delta));
                _mm256_storeu_ps(p7, _mm256_add_ps(x7, v_delta));
                i += 64;
            }
            while i + 32 <= len {
                let p0 = ptr.add(i);
                let p1 = ptr.add(i + 8);
                let p2 = ptr.add(i + 16);
                let p3 = ptr.add(i + 24);
                let x0 = _mm256_loadu_ps(p0);
                let x1 = _mm256_loadu_ps(p1);
                let x2 = _mm256_loadu_ps(p2);
                let x3 = _mm256_loadu_ps(p3);
                _mm256_storeu_ps(p0, _mm256_add_ps(x0, v_delta));
                _mm256_storeu_ps(p1, _mm256_add_ps(x1, v_delta));
                _mm256_storeu_ps(p2, _mm256_add_ps(x2, v_delta));
                _mm256_storeu_ps(p3, _mm256_add_ps(x3, v_delta));
                i += 32;
            }
            while i + 8 <= len {
                let p = ptr.add(i);
                _mm256_storeu_ps(p, _mm256_add_ps(_mm256_loadu_ps(p), v_delta));
                i += 8;
            }
            while i < len {
                *ptr.add(i) += clean_delta;
                i += 1;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for val in buf.iter_mut() {
            *val += clean_delta;
        }
    }
}

#[inline(always)]
pub(crate) fn dot_f32_unrolled4(a: &[f32], b: &[f32], len: usize) -> f32 {
    let a_sub = &a[..len];
    let b_sub = &b[..len];
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pa = a_sub.as_ptr();
        let pb = b_sub.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and `pa`, `pb` are valid for `len` elements.
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut acc2 = _mm256_setzero_ps();
            let mut acc3 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                acc0 = _mm256_add_ps(
                    acc0,
                    _mm256_mul_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))),
                );
                acc1 = _mm256_add_ps(
                    acc1,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(pa.add(i + 8)),
                        _mm256_loadu_ps(pb.add(i + 8)),
                    ),
                );
                acc2 = _mm256_add_ps(
                    acc2,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(pa.add(i + 16)),
                        _mm256_loadu_ps(pb.add(i + 16)),
                    ),
                );
                acc3 = _mm256_add_ps(
                    acc3,
                    _mm256_mul_ps(
                        _mm256_loadu_ps(pa.add(i + 24)),
                        _mm256_loadu_ps(pb.add(i + 24)),
                    ),
                );
                i += 32;
            }
            let mut acc_all = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
            while i + 8 <= len {
                acc_all = _mm256_add_ps(
                    acc_all,
                    _mm256_mul_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))),
                );
                i += 8;
            }
            let mut total = avx2_hsum_ps(acc_all);
            for (&av, &bv) in a_sub[i..].iter().zip(b_sub[i..].iter()) {
                total += av * bv;
            }
            total
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut total = 0.0_f32;
        for (&av, &bv) in a_sub.iter().zip(b_sub.iter()) {
            total += av * bv;
        }
        total
    }
}

/// Evaluates the underflow-resilient L2 norm $(\|\mathbf{x}\|_2, \mathrm{is\_finite})$,
/// pre-detecting subnormal-square risk (`0.0 < |buf[0]| < 1.0e-18`) to bypass x86 subnormal
/// microcode traps and using AVX2 `sum_sq_scaled_f32_simd8` for $2^{64}$ rescue.
#[inline(always)]
pub(crate) fn norm_l2_underflow_resilient_f32(buf: &[f32]) -> (f32, bool) {
    let len = buf.len();
    if len == 0 {
        return (0.0, true);
    }
    const SCALE_UP_2P84: f32 = f32::from_bits((127 + 84) << 23);
    const SCALE_DOWN_2M84: f32 = f32::from_bits((127 - 84) << 23);
    const SUBNORMAL_SQ_RISK_BITS: u32 = SUBNORMAL_SQ_RISK_FLOOR.to_bits();

    let b0_bits = buf[0].to_bits() & 0x7FFF_FFFF;
    let bl_bits = buf[len - 1].to_bits() & 0x7FFF_FFFF;
    if (b0_bits >= 0x7F80_0000) | (bl_bits >= 0x7F80_0000) {
        return (f32::NAN, false);
    }
    let probe_bits = branchless_select_u32(b0_bits > bl_bits, b0_bits, bl_bits);
    if (probe_bits > 0) & (probe_bits < SUBNORMAL_SQ_RISK_BITS) {
        if probe_bits < 0x0080_0000 {
            let sum_mant = sum_sq_subnormal_mantissa_f32_simd8(buf);
            if !branchless_is_finite_f32(sum_mant) {
                return (f32::NAN, false);
            }
            let root = branchless_sqrt_f32(sum_mant);
            let root_bits = root.to_bits();
            let norm = if root_bits >= 0x7F00_0000 {
                f32::from_bits(root_bits - (126 << 23))
            } else {
                0.0
            };
            return (norm, true);
        }
        let sum_scaled = sum_sq_scaled_f32_simd8(buf, SCALE_UP_2P84);
        if !branchless_is_finite_f32(sum_scaled) {
            return (f32::NAN, false);
        }
        let root = branchless_sqrt_f32(sum_scaled);
        let root_bits = root.to_bits();
        let norm = if root_bits >= ((1 + 84) << 23) {
            root * SCALE_DOWN_2M84
        } else {
            0.0
        };
        return (norm, true);
    }

    let sum_sq = sum_sq_f32_simd8(buf);
    if !branchless_is_finite_f32(sum_sq) {
        return (f32::NAN, false);
    }
    if sum_sq >= F32_MIN_POSITIVE_NORMAL {
        return (branchless_sqrt_f32(sum_sq), true);
    }
    if (probe_bits == 0) & ((buf[len >> 1].to_bits() & 0x7FFF_FFFF) == 0) & (sum_sq == 0.0) {
        return (0.0, true);
    }
    let sum_scaled = sum_sq_scaled_f32_simd8(buf, SCALE_UP_2P84);
    if !branchless_is_finite_f32(sum_scaled) {
        return (f32::NAN, false);
    }
    let root = branchless_sqrt_f32(sum_scaled);
    let root_bits = root.to_bits();
    let norm = if root_bits >= ((1 + 84) << 23) {
        root * SCALE_DOWN_2M84
    } else {
        0.0
    };
    (norm, true)
}

/// Evaluates the $SO(2)$ Procrustes inner and exterior cross products using a 32-float (4-YMM)
/// unrolled dual-accumulator AVX2 tree reduction.
#[inline(always)]
pub(crate) fn procrustes_sincos_f32_simd(src_head: &[f32], tgt_head: &[f32]) -> (f32, f32) {
    let len = src_head.len().min(tgt_head.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let ps = src_head.as_ptr();
        let pt = tgt_head.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and `ps`, `pt` are valid for `len` elements.
        // Delegates 8-float chunks to `avx2_procrustes_head_rc128` (`<= 6` live YMMs, `0` Windows x64 spills).
        unsafe {
            let vec_len = len & !7;
            let rc = avx2_procrustes_head_rc128(ps, pt, vec_len);
            let sc = _mm_addsub_ps(
                _mm_shuffle_ps::<0x88>(rc, rc),
                _mm_shuffle_ps::<0xDD>(rc, rc),
            );
            let mut sum_sin = _mm_cvtss_f32(sc);
            let mut sum_cos = _mm_cvtss_f32(_mm_shuffle_ps::<0x55>(sc, sc));
            let mut i = vec_len;
            while i + 2 <= len {
                let (ux, uy) = (*ps.add(i), *ps.add(i + 1));
                let (vx, vy) = (*pt.add(i), *pt.add(i + 1));
                sum_cos += ux * vx + uy * vy;
                sum_sin += ux * vy - uy * vx;
                i += 2;
            }
            (sum_cos, sum_sin)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut sum_cos = 0.0_f32;
        let mut sum_sin = 0.0_f32;
        for (s_pair, t_pair) in src_head[..len]
            .chunks_exact(2)
            .zip(tgt_head[..len].chunks_exact(2))
        {
            sum_cos += s_pair[0] * t_pair[0] + s_pair[1] * t_pair[1];
            sum_sin += s_pair[0] * t_pair[1] - s_pair[1] * t_pair[0];
        }
        (sum_cos, sum_sin)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_procrustes_head_rc128(
    ps: *const f32,
    pt: *const f32,
    head_dim: usize,
) -> core::arch::x86_64::__m128 {
    // SAFETY: Caller guarantees `target_feature = "avx2"`, `(head_dim & 7) == 0`, and valid pointers.
    // Dual-accumulator `(cos0, crs0, cos1, crs1)` halves the 32-float FP add latency chain (`2x` ILP)
    // while staging 1 `(a_k, b_k)` vector pair at a time (`<= 6` live YMMs `ymm0..ymm5`, `0` Windows x64
    // `xmm6..xmm15` stack saves), then folds to `[R_even, R_odd, C_even, C_odd]` in a single XMM register.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_movehl_ps,
            _mm_movelh_ps,
        };
        let mut cos0 = _mm256_setzero_ps();
        let mut crs0 = _mm256_setzero_ps();
        let mut cos1 = _mm256_setzero_ps();
        let mut crs1 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 32 <= head_dim {
            let a0 = _mm256_loadu_ps(ps.add(i));
            let b0 = _mm256_loadu_ps(pt.add(i));
            cos0 = _mm256_add_ps(cos0, _mm256_mul_ps(a0, b0));
            crs0 = _mm256_add_ps(crs0, _mm256_mul_ps(a0, _mm256_permute_ps::<0xB1>(b0)));
            let a1 = _mm256_loadu_ps(ps.add(i + 8));
            let b1 = _mm256_loadu_ps(pt.add(i + 8));
            cos1 = _mm256_add_ps(cos1, _mm256_mul_ps(a1, b1));
            crs1 = _mm256_add_ps(crs1, _mm256_mul_ps(a1, _mm256_permute_ps::<0xB1>(b1)));
            let a2 = _mm256_loadu_ps(ps.add(i + 16));
            let b2 = _mm256_loadu_ps(pt.add(i + 16));
            cos0 = _mm256_add_ps(cos0, _mm256_mul_ps(a2, b2));
            crs0 = _mm256_add_ps(crs0, _mm256_mul_ps(a2, _mm256_permute_ps::<0xB1>(b2)));
            let a3 = _mm256_loadu_ps(ps.add(i + 24));
            let b3 = _mm256_loadu_ps(pt.add(i + 24));
            cos1 = _mm256_add_ps(cos1, _mm256_mul_ps(a3, b3));
            crs1 = _mm256_add_ps(crs1, _mm256_mul_ps(a3, _mm256_permute_ps::<0xB1>(b3)));
            i += 32;
        }
        cos0 = _mm256_add_ps(cos0, cos1);
        crs0 = _mm256_add_ps(crs0, crs1);
        while i + 8 <= head_dim {
            let s0 = _mm256_loadu_ps(ps.add(i));
            let t0 = _mm256_loadu_ps(pt.add(i));
            cos0 = _mm256_add_ps(cos0, _mm256_mul_ps(s0, t0));
            crs0 = _mm256_add_ps(crs0, _mm256_mul_ps(s0, _mm256_permute_ps::<0xB1>(t0)));
            i += 8;
        }
        let r4 = _mm_add_ps(
            _mm256_castps256_ps128(crs0),
            _mm256_extractf128_ps::<1>(crs0),
        );
        let c4 = _mm_add_ps(
            _mm256_castps256_ps128(cos0),
            _mm256_extractf128_ps::<1>(cos0),
        );
        _mm_add_ps(_mm_movelh_ps(r4, c4), _mm_movehl_ps(c4, r4))
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_procrustes_cell8_sincos_vec256(
    ps_base: *const f32,
    pt_base: *const f32,
    off0: usize,
    head_dim: usize,
) -> (core::arch::x86_64::__m256, core::arch::x86_64::__m256) {
    // SAFETY: Caller guarantees `target_feature = "avx2"` and valid pointer ranges for 8 heads.
    // Packs `[sin0..sin3]` and `[cos0..cos3]` directly in the XMM FP domain via `_mm_shuffle_ps::<0x88/0xDD>`,
    // stages the lower 128-bit half via exact-width 128-bit `_mm_storeu_ps` -> `_mm_loadu_ps` (`0` STLF
    // stalls, `0` XMM<->GPR domain transfers), and merges the upper 4 heads via `_mm256_insertf128_ps::<1>`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castps128_ps256, _mm256_insertf128_ps, _mm_addsub_ps, _mm_loadu_ps,
            _mm_shuffle_ps, _mm_storeu_ps,
        };
        let mut sin_lo_buf = [0.0_f32; 4];
        let mut cos_lo_buf = [0.0_f32; 4];
        let ps = ps_base.add(off0);
        let pt = pt_base.add(off0);
        let step2 = head_dim * 2;
        let step4 = head_dim * 4;

        let rc0 = avx2_procrustes_head_rc128(ps, pt, head_dim);
        let rc1 = avx2_procrustes_head_rc128(ps.add(head_dim), pt.add(head_dim), head_dim);
        let sc01 = _mm_addsub_ps(
            _mm_shuffle_ps::<0x88>(rc0, rc1),
            _mm_shuffle_ps::<0xDD>(rc0, rc1),
        );
        let ps2 = ps.add(step2);
        let pt2 = pt.add(step2);
        let rc2 = avx2_procrustes_head_rc128(ps2, pt2, head_dim);
        let rc3 = avx2_procrustes_head_rc128(ps2.add(head_dim), pt2.add(head_dim), head_dim);
        let sc23 = _mm_addsub_ps(
            _mm_shuffle_ps::<0x88>(rc2, rc3),
            _mm_shuffle_ps::<0xDD>(rc2, rc3),
        );
        _mm_storeu_ps(sin_lo_buf.as_mut_ptr(), _mm_shuffle_ps::<0x88>(sc01, sc23));
        _mm_storeu_ps(cos_lo_buf.as_mut_ptr(), _mm_shuffle_ps::<0xDD>(sc01, sc23));

        let ps4 = ps.add(step4);
        let pt4 = pt.add(step4);
        let rc4 = avx2_procrustes_head_rc128(ps4, pt4, head_dim);
        let rc5 = avx2_procrustes_head_rc128(ps4.add(head_dim), pt4.add(head_dim), head_dim);
        let sc45 = _mm_addsub_ps(
            _mm_shuffle_ps::<0x88>(rc4, rc5),
            _mm_shuffle_ps::<0xDD>(rc4, rc5),
        );
        let ps6 = ps4.add(step2);
        let pt6 = pt4.add(step2);
        let rc6 = avx2_procrustes_head_rc128(ps6, pt6, head_dim);
        let rc7 = avx2_procrustes_head_rc128(ps6.add(head_dim), pt6.add(head_dim), head_dim);
        let sc67 = _mm_addsub_ps(
            _mm_shuffle_ps::<0x88>(rc6, rc7),
            _mm_shuffle_ps::<0xDD>(rc6, rc7),
        );
        let sin_hi = _mm_shuffle_ps::<0x88>(sc45, sc67);
        let cos_hi = _mm_shuffle_ps::<0xDD>(sc45, sc67);
        (
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(_mm_loadu_ps(sin_lo_buf.as_ptr())), sin_hi),
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(_mm_loadu_ps(cos_lo_buf.as_ptr())), cos_hi),
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_atan2_to_q7_word_x8_vec(
    vy: core::arch::x86_64::__m256,
    vx: core::arch::x86_64::__m256,
) -> u64 {
    // SAFETY: `target_feature = "avx2"` is enabled; `vy` and `vx` are in YMM registers.
    // Schedules `swap_mask`, `x_neg_mask`, `y_sign`, and `valid_mask` strictly after `r`, `r2`,
    // and `p` are dead so peak live YMMs <= 6 (`ymm0..ymm5`, `0` Windows x64 `xmm6..xmm15` stack saves).
    unsafe {
        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let ax = _mm256_and_ps(vx, abs_mask);
        let ay = _mm256_and_ps(vy, abs_mask);
        let min_v = _mm256_min_ps(ax, ay);
        let max_v = _mm256_max_ps(_mm256_max_ps(ax, ay), _mm256_set1_ps(1.0e-30_f32));

        let rcp0 = _mm256_rcp_ps(max_v);
        let inv_max = _mm256_mul_ps(
            rcp0,
            _mm256_sub_ps(_mm256_set1_ps(2.0_f32), _mm256_mul_ps(max_v, rcp0)),
        );
        let r = _mm256_min_ps(
            _mm256_max_ps(_mm256_mul_ps(min_v, inv_max), _mm256_setzero_ps()),
            _mm256_set1_ps(1.0_f32),
        );
        let r2 = _mm256_mul_ps(r, r);

        let mut p = _mm256_add_ps(
            _mm256_mul_ps(r2, _mm256_set1_ps(-0.477_609_f32)),
            _mm256_set1_ps(2.145_485_f32),
        );
        p = _mm256_add_ps(_mm256_mul_ps(r2, p), _mm256_set1_ps(-4.744_295_f32));
        p = _mm256_add_ps(_mm256_mul_ps(r2, p), _mm256_set1_ps(7.886_416_f32));
        p = _mm256_add_ps(_mm256_mul_ps(r2, p), _mm256_set1_ps(-13.553_533_f32));
        p = _mm256_add_ps(_mm256_mul_ps(r2, p), _mm256_set1_ps(40.742_737_f32));
        let poly_q7 = _mm256_mul_ps(r, p);

        let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
        let swap_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(
            _mm256_and_ps(vx, abs_mask),
            _mm256_and_ps(vy, abs_mask),
        );
        let mut ang_q7 = _mm256_blendv_ps(
            poly_q7,
            _mm256_sub_ps(_mm256_set1_ps(64.0_f32), poly_q7),
            swap_mask,
        );
        let x_neg_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(vx, _mm256_setzero_ps());
        ang_q7 = _mm256_blendv_ps(
            ang_q7,
            _mm256_sub_ps(_mm256_set1_ps(128.0_f32), ang_q7),
            x_neg_mask,
        );
        let y_sign = _mm256_and_ps(vy, _mm256_set1_ps(f32::from_bits(0x8000_0000)));
        ang_q7 = _mm256_or_ps(_mm256_and_ps(ang_q7, abs_mask), y_sign);
        let rounded_q7 = _mm256_add_ps(ang_q7, _mm256_or_ps(_mm256_set1_ps(0.5_f32), y_sign));

        let norm_sq = _mm256_add_ps(_mm256_mul_ps(vx, vx), _mm256_mul_ps(vy, vy));
        let valid_mask = _mm256_and_ps(
            _mm256_cmp_ps::<_CMP_GT_OQ>(norm_sq, _mm256_set1_ps(1.0e-12_f32)),
            _mm256_cmp_ps::<_CMP_LT_OQ>(norm_sq, _mm256_set1_ps(f32::INFINITY)),
        );
        let v_i32 = _mm256_and_si256(
            _mm256_cvttps_epi32(rounded_q7),
            _mm256_castps_si256(valid_mask),
        );
        let lo128 = _mm256_castsi256_si128(v_i32);
        let hi128 = _mm256_extracti128_si256::<1>(v_i32);
        let p16 = _mm_packs_epi32(lo128, hi128);
        let p8 = _mm_packs_epi16(p16, p16);
        _mm_cvtsi128_si64(p8) as u64
    }
}

#[inline(always)]
pub(crate) fn atan2_to_q7_word_x8_simd(sin_arr: &[f32; 8], cos_arr: &[f32; 8]) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `target_feature = "avx2"` is enabled; `sin_arr` and `cos_arr` hold 8 `f32`s.
        unsafe {
            let vy = _mm256_loadu_ps(sin_arr.as_ptr());
            let vx = _mm256_loadu_ps(cos_arr.as_ptr());
            return avx2_atan2_to_q7_word_x8_vec(vy, vx);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut word = 0u64;
        let mut lane = 0usize;
        while lane < 8 {
            let sum_sin = sin_arr[lane];
            let sum_cos = cos_arr[lane];
            let norm_sq = sum_cos * sum_cos + sum_sin * sum_sin;
            let valid = (norm_sq > 1.0e-12) & branchless_is_finite_f32(norm_sq);
            let q7 = if valid {
                branchless_atan2_to_q7_fast(sum_sin, sum_cos)
            } else {
                0
            };
            word |= ((q7 as u8) as u64) << (lane << 3);
            lane += 1;
        }
        word
    }
}

/// Computes the Orthogonal Procrustes $SO(2)$ phase delta for a single 8-head SWAR cell
/// (`h_base .. h_base + cell_heads`, where `cell_heads <= 8`), returning the packed 64-bit Q7 SWAR word.
#[inline(always)]
pub(crate) fn procrustes_q7_cell_word_simd(
    src_slice: &[f32],
    tgt_slice: &[f32],
    head_dim: usize,
    h_base: usize,
    cell_heads: usize,
) -> u64 {
    if cell_heads == 8 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            if (head_dim & 7) == 0
                && head_dim >= 8
                && src_slice.len() >= (h_base + 8) * head_dim
                && tgt_slice.len() >= (h_base + 8) * head_dim
            {
                let ps_base = src_slice.as_ptr();
                let pt_base = tgt_slice.as_ptr();
                // SAFETY: `target_feature = "avx2"`, `(head_dim & 7) == 0`, and slice bounds verified above.
                return unsafe {
                    let (vy, vx) = avx2_procrustes_cell8_sincos_vec256(
                        ps_base,
                        pt_base,
                        h_base * head_dim,
                        head_dim,
                    );
                    avx2_atan2_to_q7_word_x8_vec(vy, vx)
                };
            }
        }
        let mut sin_arr = [0.0_f32; 8];
        let mut cos_arr = [0.0_f32; 8];
        let mut lane = 0usize;
        while lane < 8 {
            let base = (h_base + lane) * head_dim;
            let end = base + head_dim;
            let (sum_cos, sum_sin) =
                procrustes_sincos_f32_simd(&src_slice[base..end], &tgt_slice[base..end]);
            cos_arr[lane] = sum_cos;
            sin_arr[lane] = sum_sin;
            lane += 1;
        }
        return atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr);
    }
    let mut sin_arr = [0.0_f32; 8];
    let mut cos_arr = [0.0_f32; 8];
    let mut lane = 0usize;
    while lane < cell_heads {
        let base = (h_base + lane) * head_dim;
        let end = base + head_dim;
        let (sum_cos, sum_sin) =
            procrustes_sincos_f32_simd(&src_slice[base..end], &tgt_slice[base..end]);
        cos_arr[lane] = sum_cos;
        sin_arr[lane] = sum_sin;
        lane += 1;
    }
    atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr)
}

/// Computes the Orthogonal Procrustes $SO(2)$ phase delta across `num_heads` directly into
/// 8 register-resident `u64` SWAR words (`[u64; 8]`), evaluating 8-head AVX2 `atan2` in YMM
/// registers with zero scalar XMM-to-GPR roundtrips.
#[inline(always)]
pub(crate) fn procrustes_q7_words_64_simd(
    src_slice: &[f32],
    tgt_slice: &[f32],
    head_dim: usize,
    num_heads: usize,
) -> [u64; 8] {
    let mut words = [0u64; 8];
    let limit_heads = num_heads.min(MAX_ATTENTION_HEADS);
    let full_cells = limit_heads >> 3;
    let mut c = 0usize;
    while c < full_cells {
        words[c] = procrustes_q7_cell_word_simd(src_slice, tgt_slice, head_dim, c << 3, 8);
        c += 1;
    }
    let rem_start = full_cells << 3;
    if rem_start < limit_heads {
        words[full_cells] = procrustes_q7_cell_word_simd(
            src_slice,
            tgt_slice,
            head_dim,
            rem_start,
            limit_heads - rem_start,
        );
    }
    words
}

#[inline(always)]
pub(crate) fn procrustes_q7_slice_simd(
    src_slice: &[f32],
    tgt_slice: &[f32],
    head_dim: usize,
    num_heads: usize,
    phases_out: &mut [gcso_q7_t],
) {
    let limit_heads = num_heads.min(MAX_ATTENTION_HEADS).min(phases_out.len());
    let words = procrustes_q7_words_64_simd(src_slice, tgt_slice, head_dim, limit_heads);
    if limit_heads == MAX_ATTENTION_HEADS {
        // SAFETY: `phases_out.len() >= MAX_ATTENTION_HEADS` and `align_of::<gcso_q7_t>() == 1`.
        let out_arr =
            unsafe { &mut *phases_out.as_mut_ptr().cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        store_q7_words_64(out_arr, &words);
        return;
    }
    let full_cells = limit_heads >> 3;
    let mut c = 0usize;
    while c < full_cells {
        store_q7_x8_slice(phases_out, c << 3, words[c]);
        c += 1;
    }
    let rem_start = full_cells << 3;
    if rem_start < limit_heads {
        let mut w = words[full_cells];
        let mut h = rem_start;
        while h < limit_heads {
            phases_out[h] = (w & 0xFF) as gcso_q7_t;
            w >>= 8;
            h += 1;
        }
    }
}

/// Evaluates range-reduced $(\sin\theta, \cos\theta)$ across 8 `f32` angles simultaneously using AVX2.
#[inline(always)]
pub(crate) fn sincos_f32_x8_simd(
    angles: &[f32],
    sin_out: &mut [f32; 8],
    cos_out: &mut [f32; 8],
) {
    debug_assert!(angles.len() >= 8);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `target_feature = "avx2"` is enabled, and `angles`, `sin_out`, `cos_out` hold at least 8 `f32`s.
        unsafe {
            let v_in = _mm256_loadu_ps(angles.as_ptr());
            let v_abs = _mm256_and_ps(v_in, _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF)));
            let in_fast_range = _mm256_cmp_ps::<_CMP_LE_OQ>(v_abs, _mm256_set1_ps(65_536.0_f32));

            if _mm256_movemask_ps(in_fast_range) == 0xFF {
                let k =
                    _mm256_round_ps::<0x08>(_mm256_mul_ps(v_in, _mm256_set1_ps(0.159_154_94_f32)));
                let x = _mm256_sub_ps(
                    _mm256_sub_ps(v_in, _mm256_mul_ps(k, _mm256_set1_ps(6.283_185_f32))),
                    _mm256_mul_ps(k, _mm256_set1_ps(3.071_795_9e-7_f32)),
                );

                let gt_hp = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(FRAC_PI_2));
                let lt_nhp = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(-FRAC_PI_2));
                let x_fold_lo =
                    _mm256_blendv_ps(x, _mm256_sub_ps(_mm256_set1_ps(-PI), x), lt_nhp);
                let x_folded =
                    _mm256_blendv_ps(x_fold_lo, _mm256_sub_ps(_mm256_set1_ps(PI), x), gt_hp);
                let cos_neg_mask = _mm256_and_ps(
                    _mm256_or_ps(gt_hp, lt_nhp),
                    _mm256_set1_ps(f32::from_bits(0x8000_0000)),
                );

                let x2 = _mm256_mul_ps(x_folded, x_folded);
                let one_v = _mm256_set1_ps(1.0_f32);

                let mut sp = _mm256_add_ps(
                    _mm256_mul_ps(x2, _mm256_set1_ps(-0.000_000_025_052_108_f32)),
                    _mm256_set1_ps(0.000_002_755_732_f32),
                );
                sp = _mm256_add_ps(_mm256_mul_ps(x2, sp), _mm256_set1_ps(-0.000_198_412_7_f32));
                sp = _mm256_add_ps(_mm256_mul_ps(x2, sp), _mm256_set1_ps(0.008_333_334_f32));
                sp = _mm256_add_ps(_mm256_mul_ps(x2, sp), _mm256_set1_ps(-0.166_666_67_f32));
                let sin_v =
                    _mm256_mul_ps(x_folded, _mm256_add_ps(_mm256_mul_ps(x2, sp), one_v));

                let mut cp = _mm256_add_ps(
                    _mm256_mul_ps(x2, _mm256_set1_ps(-0.000_000_275_573_19_f32)),
                    _mm256_set1_ps(0.000_024_801_587_f32),
                );
                cp = _mm256_add_ps(_mm256_mul_ps(x2, cp), _mm256_set1_ps(-0.001_388_888_9_f32));
                cp = _mm256_add_ps(_mm256_mul_ps(x2, cp), _mm256_set1_ps(0.041_666_668_f32));
                cp = _mm256_add_ps(_mm256_mul_ps(x2, cp), _mm256_set1_ps(-0.5_f32));
                let cos_v =
                    _mm256_xor_ps(_mm256_add_ps(_mm256_mul_ps(x2, cp), one_v), cos_neg_mask);

                _mm256_storeu_ps(sin_out.as_mut_ptr(), sin_v);
                _mm256_storeu_ps(cos_out.as_mut_ptr(), cos_v);
                return;
            }
        }
    }
    for lane in 0..8 {
        let (s, c) = branchless_sincos_f32(angles[lane]);
        sin_out[lane] = s;
        cos_out[lane] = c;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_block32(ptr: *mut f32, v_cos: __m256, v_sin: __m256) {
    // SAFETY: Caller guarantees `ptr` is valid for 32 floats and AVX2 is enabled.
    unsafe {
        let p0 = ptr;
        let p1 = ptr.add(8);
        let v0 = _mm256_loadu_ps(p0);
        let v1 = _mm256_loadu_ps(p1);
        _mm256_storeu_ps(
            p0,
            _mm256_addsub_ps(
                _mm256_mul_ps(v0, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
            ),
        );
        _mm256_storeu_ps(
            p1,
            _mm256_addsub_ps(
                _mm256_mul_ps(v1, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
            ),
        );
        let p2 = ptr.add(16);
        let p3 = ptr.add(24);
        let v2 = _mm256_loadu_ps(p2);
        let v3 = _mm256_loadu_ps(p3);
        _mm256_storeu_ps(
            p2,
            _mm256_addsub_ps(
                _mm256_mul_ps(v2, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v2), v_sin),
            ),
        );
        _mm256_storeu_ps(
            p3,
            _mm256_addsub_ps(
                _mm256_mul_ps(v3, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v3), v_sin),
            ),
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_dispatch(
    ptr: *mut f32,
    len: usize,
    v_cos: __m256,
    v_sin: __m256,
    sin_t: f32,
    cos_t: f32,
) {
    // SAFETY: Caller guarantees `ptr` is valid for `len` floats and AVX2 is enabled.
    // Staged 2-YMM (16-float) sub-blocks keep peak live vector register count <= 6 (`ymm0..ymm5`),
    // eliminating callee-saved `xmm6..xmm15` stack spills on Windows x64 ABI while saturating both 256-bit FMA ports.
    unsafe {
        if len == 32 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            return;
        }
        if len == 64 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
            return;
        }
        if len == 128 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(64), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(96), v_cos, v_sin);
            return;
        }
        if len == 256 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(64), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(96), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(128), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(160), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(192), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(224), v_cos, v_sin);
            return;
        }
        let mut b = 0usize;
        if len >= 65_536 {
            // For large streaming buffers (`>= 256 KB`), step in 4KB-page chunks (`1024` floats)
            // issuing multi-page DTLB/L2 prefetches (`+8 KB`, `+16 KB`, `+32 KB`) outside a pure
            // 4-block (`128`-float, ~400B DSB-locked) inner loop with zero inner prefetch branches.
            while b + 1024 <= len {
                let page_ptr = ptr.add(b);
                if b + 2048 < len {
                    core::arch::x86_64::_mm_prefetch(
                        page_ptr.add(2048).cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }
                if b + 4096 < len {
                    core::arch::x86_64::_mm_prefetch(
                        page_ptr.add(4096).cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T1,
                    );
                }
                if b + 8192 < len {
                    core::arch::x86_64::_mm_prefetch(
                        page_ptr.add(8192).cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T1,
                    );
                }
                let page_end = b + 1024;
                while b < page_end {
                    let p = ptr.add(b);
                    avx2_rotate_so2_block32(p, v_cos, v_sin);
                    avx2_rotate_so2_block32(p.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(p.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(p.add(96), v_cos, v_sin);
                    b += 128;
                }
            }
        }
        while b + 128 <= len {
            let p = ptr.add(b);
            avx2_rotate_so2_block32(p, v_cos, v_sin);
            avx2_rotate_so2_block32(p.add(32), v_cos, v_sin);
            avx2_rotate_so2_block32(p.add(64), v_cos, v_sin);
            avx2_rotate_so2_block32(p.add(96), v_cos, v_sin);
            b += 128;
        }
        while b + 32 <= len {
            avx2_rotate_so2_block32(ptr.add(b), v_cos, v_sin);
            b += 32;
        }
        if b == len {
            return;
        }
        while b + 8 <= len {
            let p0 = ptr.add(b);
            let v0 = _mm256_loadu_ps(p0);
            _mm256_storeu_ps(
                p0,
                _mm256_addsub_ps(
                    _mm256_mul_ps(v0, v_cos),
                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                ),
            );
            b += 8;
        }
        while b + 2 <= len {
            let (r0, r1) = branchless_rotate_so2_pair(*ptr.add(b), *ptr.add(b + 1), sin_t, cos_t);
            *ptr.add(b) = r0;
            *ptr.add(b + 1) = r1;
            b += 2;
        }
    }
}

/// Bank-conflict-free 64B cache-line-staged 2-tensor (`Q`, `K`) AVX2 $SO(2)$ 32-float block rotation.
///
/// Rotates the two 64-byte cache lines of `qp` (`+0..+64B` and `+64..+128B`) via `avx2_rotate_so2_block32`
/// before rotating `kp` with the same register-resident `(v_cos, v_sin)`. Pairing `p0` (`+0B`, bit 5 = 0)
/// and `p1` (`+32B`, bit 5 = 1) inside each 64-byte cache line issues dual 256-bit loads to opposite L1D
/// 32B cache banks rather than colliding `qp` and `kp` at identical 4KB page offsets `(qp ^ kp) & 0xFFF == 0`,
/// while keeping peak live YMM usage `<= 6` (`ymm0..ymm5`, zero Windows x64 `xmm6..xmm15` stack spills).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_pair_block32(
    qp: *mut f32,
    kp: *mut f32,
    v_cos: __m256,
    v_sin: __m256,
) {
    // SAFETY: Caller guarantees `qp` and `kp` are disjoint, valid for 32 floats, and AVX2 is enabled.
    unsafe {
        avx2_rotate_so2_block32(qp, v_cos, v_sin);
        avx2_rotate_so2_block32(kp, v_cos, v_sin);
    }
}

/// Bank-conflict-free 64B cache-line-staged 3-tensor (`Q`, `K`, `V`) AVX2 $SO(2)$ 32-float block rotation.
///
/// Rotates `qp`, `kp`, and `vp` sequentially in 32-float (128-byte, 2-cache-line) blocks using shared
/// register-resident `(v_cos, v_sin)` (`<= 6` live YMM registers `ymm0..ymm5`, zero Windows x64
/// `xmm6..xmm15` callee-saved stack spills, zero L1D same-bank port conflicts, and zero 3-way 4KB
/// Load-Store Queue page-offset aliasing stalls).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_triple_block32(
    qp: *mut f32,
    kp: *mut f32,
    vp: *mut f32,
    v_cos: __m256,
    v_sin: __m256,
) {
    // SAFETY: Caller guarantees `qp`, `kp`, `vp` are pairwise disjoint, valid for 32 floats, and AVX2 is enabled.
    unsafe {
        avx2_rotate_so2_block32(qp, v_cos, v_sin);
        avx2_rotate_so2_block32(kp, v_cos, v_sin);
        avx2_rotate_so2_block32(vp, v_cos, v_sin);
    }
}

/// Rotates `buf` in-place by `(sin_t, cos_t)` in strictly constant time (`0` data-dependent
/// early-return branches on `sin_t == 0.0 && cos_t == 1.0`, guaranteeing `1.00x` timing parity
/// between zero phase `0` and max saturation `+127`/`-128`).
#[inline(always)]
pub(crate) fn rotate_so2_slice_unrolled16(buf: &mut [f32], sin_t: f32, cos_t: f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `target_feature = "avx2"` is enabled and `buf` is valid for `buf.len()` floats.
        unsafe {
            avx2_rotate_so2_dispatch(
                buf.as_mut_ptr(),
                buf.len(),
                _mm256_set1_ps(cos_t),
                _mm256_set1_ps(sin_t),
                sin_t,
                cos_t,
            );
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for pair in buf.chunks_exact_mut(2) {
            (pair[0], pair[1]) = branchless_rotate_so2_pair(pair[0], pair[1], sin_t, cos_t);
        }
    }
}

#[inline(always)]
pub(crate) fn rotate_so2_2d_xy_slice_simd(
    buf: &mut [f32],
    sin_x: f32,
    cos_x: f32,
    sin_y: f32,
    cos_y: f32,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let len = buf.len();
        let ptr = buf.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and `b + 32 <= len` / `b + 4 <= len` are bounds-checked.
        unsafe {
            let v_cos = _mm256_setr_ps(cos_x, cos_x, cos_y, cos_y, cos_x, cos_x, cos_y, cos_y);
            let v_sin = _mm256_setr_ps(sin_x, sin_x, sin_y, sin_y, sin_x, sin_x, sin_y, sin_y);
            let mut b = 0usize;
            while b + 32 <= len {
                for off in [0usize, 8, 16, 24] {
                    let p = ptr.add(b + off);
                    let v = _mm256_loadu_ps(p);
                    _mm256_storeu_ps(
                        p,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v), v_sin),
                        ),
                    );
                }
                b += 32;
            }
            while b + 8 <= len {
                let p = ptr.add(b);
                let v = _mm256_loadu_ps(p);
                _mm256_storeu_ps(
                    p,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v), v_sin),
                    ),
                );
                b += 8;
            }
            while b + 4 <= len {
                let p = ptr.add(b);
                let v = _mm_loadu_ps(p);
                let c_lo = _mm256_castps256_ps128(v_cos);
                let s_lo = _mm256_castps256_ps128(v_sin);
                _mm_storeu_ps(
                    p,
                    _mm_addsub_ps(
                        _mm_mul_ps(v, c_lo),
                        _mm_mul_ps(_mm_permute_ps::<0xB1>(v), s_lo),
                    ),
                );
                b += 4;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for quad in buf.chunks_exact_mut(4) {
            (quad[0], quad[1]) = branchless_rotate_so2_pair(quad[0], quad[1], sin_x, cos_x);
            (quad[2], quad[3]) = branchless_rotate_so2_pair(quad[2], quad[3], sin_y, cos_y);
        }
    }
}

/// Executes 3D spatiotemporal $(\Delta\theta_t, \Delta\theta_x, \Delta\theta_y)$ rotation across `buf`
/// using a 24-float (3-YMM = 4-hexagon) unrolled AVX2 kernel with 12-float (`1 YMM + 1 XMM`) and
/// 6-float (`1 XMM + 1 pair`) vector tail handling.
#[inline(always)]
pub(crate) fn rotate_so2_3d_txy_slice_simd(
    buf: &mut [f32],
    sin_t: f32,
    cos_t: f32,
    sin_x: f32,
    cos_x: f32,
    sin_y: f32,
    cos_y: f32,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let hex_len = (buf.len() / 6) * 6;
        let ptr = buf.as_mut_ptr();
        let mut b = 0usize;
        // SAFETY: `target_feature = "avx2"` is enabled, and `b + 24 / 12 / 6 <= hex_len` are bounds-checked.
        unsafe {
            let c0 = _mm256_setr_ps(cos_t, cos_t, cos_x, cos_x, cos_y, cos_y, cos_t, cos_t);
            let s0 = _mm256_setr_ps(sin_t, sin_t, sin_x, sin_x, sin_y, sin_y, sin_t, sin_t);
            let c1 = _mm256_setr_ps(cos_x, cos_x, cos_y, cos_y, cos_t, cos_t, cos_x, cos_x);
            let s1 = _mm256_setr_ps(sin_x, sin_x, sin_y, sin_y, sin_t, sin_t, sin_x, sin_x);
            let c2 = _mm256_setr_ps(cos_y, cos_y, cos_t, cos_t, cos_x, cos_x, cos_y, cos_y);
            let s2 = _mm256_setr_ps(sin_y, sin_y, sin_t, sin_t, sin_x, sin_x, sin_y, sin_y);

            while b + 24 <= hex_len {
                let p0 = ptr.add(b);
                let p1 = ptr.add(b + 8);
                let p2 = ptr.add(b + 16);
                let v0 = _mm256_loadu_ps(p0);
                let v1 = _mm256_loadu_ps(p1);
                let v2 = _mm256_loadu_ps(p2);
                _mm256_storeu_ps(
                    p0,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, c0),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), s0),
                    ),
                );
                _mm256_storeu_ps(
                    p1,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v1, c1),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), s1),
                    ),
                );
                _mm256_storeu_ps(
                    p2,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v2, c2),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v2), s2),
                    ),
                );
                b += 24;
            }
            if b + 12 <= hex_len {
                let p0 = ptr.add(b);
                let p1 = ptr.add(b + 8);
                let v0 = _mm256_loadu_ps(p0);
                let v1_lo = _mm_loadu_ps(p1);
                let c1_lo = _mm256_castps256_ps128(c1);
                let s1_lo = _mm256_castps256_ps128(s1);
                _mm256_storeu_ps(
                    p0,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, c0),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), s0),
                    ),
                );
                _mm_storeu_ps(
                    p1,
                    _mm_addsub_ps(
                        _mm_mul_ps(v1_lo, c1_lo),
                        _mm_mul_ps(_mm_permute_ps::<0xB1>(v1_lo), s1_lo),
                    ),
                );
                b += 12;
            }
            if b + 6 <= hex_len {
                let p0 = ptr.add(b);
                let v0_lo = _mm_loadu_ps(p0);
                let c0_lo = _mm256_castps256_ps128(c0);
                let s0_lo = _mm256_castps256_ps128(s0);
                _mm_storeu_ps(
                    p0,
                    _mm_addsub_ps(
                        _mm_mul_ps(v0_lo, c0_lo),
                        _mm_mul_ps(_mm_permute_ps::<0xB1>(v0_lo), s0_lo),
                    ),
                );
                let (ry0, ry1) =
                    branchless_rotate_so2_pair(*ptr.add(b + 4), *ptr.add(b + 5), sin_y, cos_y);
                *ptr.add(b + 4) = ry0;
                *ptr.add(b + 5) = ry1;
            }
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for hex in buf.chunks_exact_mut(6) {
            (hex[0], hex[1]) = branchless_rotate_so2_pair(hex[0], hex[1], sin_t, cos_t);
            (hex[2], hex[3]) = branchless_rotate_so2_pair(hex[2], hex[3], sin_x, cos_x);
            (hex[4], hex[5]) = branchless_rotate_so2_pair(hex[4], hex[5], sin_y, cos_y);
        }
    }
}

/// Evaluates SRL Rank-1 outer product write `y_slice[i] = scale * u_slice[i] * g_i` using AVX2.
#[inline(always)]
pub(crate) fn srl_write_outer_product_simd(
    y_slice: &mut [f32],
    u_slice: &[f32],
    g_slice_opt: Option<&[f32]>,
    scale: f32,
) {
    let clean_scale = branchless_flush_subnormal_f32(scale);
    let len = y_slice.len().min(u_slice.len());
    if let Some(g_slice) = g_slice_opt {
        let len_g = len.min(g_slice.len());
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let py = y_slice.as_mut_ptr();
            let pu = u_slice.as_ptr();
            let pg = g_slice.as_ptr();
            // SAFETY: `target_feature = "avx2"` is enabled, and `i + 32 <= len_g` is bounds-checked.
            unsafe {
                let vs = _mm256_set1_ps(clean_scale);
                let mut i = 0usize;
                while i + 32 <= len_g {
                    for off in [0usize, 8, 16, 24] {
                        let prod = _mm256_mul_ps(
                            _mm256_loadu_ps(pu.add(i + off)),
                            _mm256_loadu_ps(pg.add(i + off)),
                        );
                        _mm256_storeu_ps(py.add(i + off), _mm256_mul_ps(prod, vs));
                    }
                    i += 32;
                }
                while i < len_g {
                    *py.add(i) = clean_scale * *pu.add(i) * *pg.add(i);
                    i += 1;
                }
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for ((y_i, &u_i), &g_i) in y_slice[..len_g]
                .iter_mut()
                .zip(u_slice[..len_g].iter())
                .zip(g_slice[..len_g].iter())
            {
                *y_i = clean_scale * u_i * g_i;
            }
        }
    } else {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let py = y_slice.as_mut_ptr();
            let pu = u_slice.as_ptr();
            // SAFETY: `target_feature = "avx2"` is enabled, and `i + 32 <= len` is bounds-checked.
            unsafe {
                let vs = _mm256_set1_ps(clean_scale);
                let mut i = 0usize;
                while i + 32 <= len {
                    for off in [0usize, 8, 16, 24] {
                        _mm256_storeu_ps(
                            py.add(i + off),
                            _mm256_mul_ps(_mm256_loadu_ps(pu.add(i + off)), vs),
                        );
                    }
                    i += 32;
                }
                while i < len {
                    *py.add(i) = clean_scale * *pu.add(i);
                    i += 1;
                }
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for (y_i, &u_i) in y_slice[..len].iter_mut().zip(u_slice[..len].iter()) {
                *y_i = clean_scale * u_i;
            }
        }
    }
}

/// Accumulates SRL Rank-1 outer product `y_slice[i] += scale * u_slice[i] * g_i` using AVX2.
#[inline(always)]
pub(crate) fn srl_accum_outer_product_simd(
    y_slice: &mut [f32],
    u_slice: &[f32],
    g_slice_opt: Option<&[f32]>,
    scale: f32,
) {
    let clean_scale = branchless_flush_subnormal_f32(scale);
    if (clean_scale.to_bits() & 0x7FFF_FFFF) == 0 {
        return;
    }
    let len = y_slice.len().min(u_slice.len());
    if let Some(g_slice) = g_slice_opt {
        let len_g = len.min(g_slice.len());
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let py = y_slice.as_mut_ptr();
            let pu = u_slice.as_ptr();
            let pg = g_slice.as_ptr();
            // SAFETY: `target_feature = "avx2"` is enabled, and `i + 32 <= len_g` is bounds-checked.
            unsafe {
                let vs = _mm256_set1_ps(clean_scale);
                let mut i = 0usize;
                while i + 32 <= len_g {
                    for off in [0usize, 8, 16, 24] {
                        let d = _mm256_mul_ps(
                            _mm256_mul_ps(
                                _mm256_loadu_ps(pu.add(i + off)),
                                _mm256_loadu_ps(pg.add(i + off)),
                            ),
                            vs,
                        );
                        _mm256_storeu_ps(
                            py.add(i + off),
                            _mm256_add_ps(_mm256_loadu_ps(py.add(i + off)), d),
                        );
                    }
                    i += 32;
                }
                while i < len_g {
                    *py.add(i) += clean_scale * *pu.add(i) * *pg.add(i);
                    i += 1;
                }
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for ((y_i, &u_i), &g_i) in y_slice[..len_g]
                .iter_mut()
                .zip(u_slice[..len_g].iter())
                .zip(g_slice[..len_g].iter())
            {
                *y_i += clean_scale * u_i * g_i;
            }
        }
    } else {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let py = y_slice.as_mut_ptr();
            let pu = u_slice.as_ptr();
            // SAFETY: `target_feature = "avx2"` is enabled, and `i + 32 <= len` is bounds-checked.
            unsafe {
                let vs = _mm256_set1_ps(clean_scale);
                let mut i = 0usize;
                while i + 32 <= len {
                    for off in [0usize, 8, 16, 24] {
                        let d = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i + off)), vs);
                        _mm256_storeu_ps(
                            py.add(i + off),
                            _mm256_add_ps(_mm256_loadu_ps(py.add(i + off)), d),
                        );
                    }
                    i += 32;
                }
                while i < len {
                    *py.add(i) += clean_scale * *pu.add(i);
                    i += 1;
                }
            }
            return;
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for (y_i, &u_i) in y_slice[..len].iter_mut().zip(u_slice[..len].iter()) {
                *y_i += clean_scale * u_i;
            }
        }
    }
}


/// Computes $\sum_i (a_i - b_i)^2$ using 4-way unrolled AVX2 SIMD accumulation.
#[inline(always)]
#[must_use]
pub(crate) fn diff_norm_sq_f32_unrolled4(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled at compile time, and `i + 32 <= len` is in-bounds.
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 8)),
                    _mm256_loadu_ps(pb.add(i + 8)),
                );
                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 16)),
                    _mm256_loadu_ps(pb.add(i + 16)),
                );
                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 24)),
                    _mm256_loadu_ps(pb.add(i + 24)),
                );
                acc0 = _mm256_add_ps(
                    acc0,
                    _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1)),
                );
                acc1 = _mm256_add_ps(
                    acc1,
                    _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3)),
                );
                i += 32;
            }
            while i + 8 <= len {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
                i += 8;
            }
            let mut total = avx2_hsum_ps(_mm256_add_ps(acc0, acc1));
            while i < len {
                let d = *pa.add(i) - *pb.add(i);
                total += d * d;
                i += 1;
            }
            total
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut s = [0.0_f32; 4];
        let mut a_chunks = a[..len].chunks_exact(4);
        let mut b_chunks = b[..len].chunks_exact(4);
        for (ac, bc) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            for k in 0..4 {
                let d = ac[k] - bc[k];
                s[k] += d * d;
            }
        }
        let mut total = (s[0] + s[1]) + (s[2] + s[3]);
        for (&av, &bv) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
            let d = av - bv;
            total += d * d;
        }
        total
    }
}

/// Computes $(\mathbf{a} \cdot \mathbf{b}, \|\mathbf{a}\|_2^2, \|\mathbf{b}\|_2^2)$ in a single AVX2 pass
/// while keeping peak live YMM usage `<= 5` (`0` Windows x64 `xmm6..xmm15` stack saves/restores).
#[inline(always)]
#[must_use]
pub(crate) fn dot_and_norms_f32_unrolled4(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let len = a.len().min(b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled at compile time, and all offsets are bounds-checked against `len`.
        // Accumulating into 3 YMMs (`dot_all, na_all, nb_all`) with 2 scratch YMMs (`va, vb`) keeps peak live YMMs
        // at 5 (`ymm0..ymm4`), completely eliminating 8 `vmovaps` stack saves/restores of `xmm6..xmm13` on Windows x64.
        unsafe {
            let mut dot_all = _mm256_setzero_ps();
            let mut na_all = _mm256_setzero_ps();
            let mut nb_all = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                let va0 = _mm256_loadu_ps(pa.add(i));
                let vb0 = _mm256_loadu_ps(pb.add(i));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va0, vb0));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va0, va0));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb0, vb0));

                let va1 = _mm256_loadu_ps(pa.add(i + 8));
                let vb1 = _mm256_loadu_ps(pb.add(i + 8));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va1, vb1));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va1, va1));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb1, vb1));

                let va2 = _mm256_loadu_ps(pa.add(i + 16));
                let vb2 = _mm256_loadu_ps(pb.add(i + 16));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va2, vb2));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va2, va2));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb2, vb2));

                let va3 = _mm256_loadu_ps(pa.add(i + 24));
                let vb3 = _mm256_loadu_ps(pb.add(i + 24));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va3, vb3));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va3, va3));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb3, vb3));
                i += 32;
            }
            let rem = len - i;
            if (rem & 16) != 0 {
                let va0 = _mm256_loadu_ps(pa.add(i));
                let vb0 = _mm256_loadu_ps(pb.add(i));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va0, vb0));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va0, va0));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb0, vb0));

                let va1 = _mm256_loadu_ps(pa.add(i + 8));
                let vb1 = _mm256_loadu_ps(pb.add(i + 8));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va1, vb1));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va1, va1));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb1, vb1));
                i += 16;
            }
            if (rem & 8) != 0 {
                let va0 = _mm256_loadu_ps(pa.add(i));
                let vb0 = _mm256_loadu_ps(pb.add(i));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(va0, vb0));
                na_all = _mm256_add_ps(na_all, _mm256_mul_ps(va0, va0));
                nb_all = _mm256_add_ps(nb_all, _mm256_mul_ps(vb0, vb0));
                i += 8;
            }
            let (mut dot, mut norm_a_sq, mut norm_b_sq) = avx2_hsum3_ps(dot_all, na_all, nb_all);
            while i < len {
                let av = *pa.add(i);
                let bv = *pb.add(i);
                dot += av * bv;
                norm_a_sq += av * av;
                norm_b_sq += bv * bv;
                i += 1;
            }
            (dot, norm_a_sq, norm_b_sq)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let (mut dot, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
        for (&av, &bv) in a[..len].iter().zip(b[..len].iter()) {
            dot += av * bv;
            na += av * av;
            nb += bv * bv;
        }
        (dot, na, nb)
    }
}

/// Computes $(\|\mathbf{r} - \mathbf{p}\|_2^2, \langle \mathbf{r} - \mathbf{p}, \boldsymbol{\gamma} \rangle, \|\boldsymbol{\gamma}\|_2^2)$
/// in a single fused AVX2 memory pass while keeping peak live YMM usage `<= 5`
/// (`0` Windows x64 `xmm6..xmm15` stack saves/restores).
#[inline(always)]
#[must_use]
pub(crate) fn hodge_obstruction_and_dual_pairing_f32(
    r_slice: &[f32],
    p_slice: &[f32],
    g_slice: &[f32],
) -> (f32, f32, f32) {
    let n = r_slice.len().min(p_slice.len()).min(g_slice.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pr = r_slice.as_ptr();
        let pp = p_slice.as_ptr();
        let pg = g_slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled at compile time, and pointer offsets are bounds-checked against `n`.
        // Using 3 accumulator YMMs (`obs_all, dot_all, gam_all`) + 2 scratch YMMs (`d, g`) keeps peak live YMMs
        // at 5 (`ymm0..ymm4`), eliminating 8 `vmovaps` stack saves/restores of `xmm6..xmm13` on Windows x64 ABI
        // and cutting `hodge_obstruction_and_dual_homology_pairing_64` variance and latency.
        unsafe {
            let mut obs_all = _mm256_setzero_ps();
            let mut dot_all = _mm256_setzero_ps();
            let mut gam_all = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= n {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pp.add(i)));
                let g0 = _mm256_loadu_ps(pg.add(i));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d0, d0));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d0, g0));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g0, g0));

                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 8)),
                    _mm256_loadu_ps(pp.add(i + 8)),
                );
                let g1 = _mm256_loadu_ps(pg.add(i + 8));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d1, d1));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d1, g1));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g1, g1));

                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 16)),
                    _mm256_loadu_ps(pp.add(i + 16)),
                );
                let g2 = _mm256_loadu_ps(pg.add(i + 16));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d2, d2));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d2, g2));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g2, g2));

                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 24)),
                    _mm256_loadu_ps(pp.add(i + 24)),
                );
                let g3 = _mm256_loadu_ps(pg.add(i + 24));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d3, d3));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d3, g3));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g3, g3));
                i += 32;
            }
            let rem = n - i;
            if (rem & 16) != 0 {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pp.add(i)));
                let g0 = _mm256_loadu_ps(pg.add(i));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d0, d0));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d0, g0));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g0, g0));

                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 8)),
                    _mm256_loadu_ps(pp.add(i + 8)),
                );
                let g1 = _mm256_loadu_ps(pg.add(i + 8));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d1, d1));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d1, g1));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g1, g1));
                i += 16;
            }
            if (rem & 8) != 0 {
                let d = _mm256_sub_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pp.add(i)));
                let g = _mm256_loadu_ps(pg.add(i));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d, d));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d, g));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g, g));
                i += 8;
            }
            let (mut obs_sq, mut pair_dot, mut gam_sq) = avx2_hsum3_ps(obs_all, dot_all, gam_all);
            while i < n {
                let r_obs = *pr.add(i) - *pp.add(i);
                let gv = *pg.add(i);
                obs_sq += r_obs * r_obs;
                pair_dot += r_obs * gv;
                gam_sq += gv * gv;
                i += 1;
            }
            (obs_sq, pair_dot, gam_sq)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut s_obs = [0.0_f32; 4];
        let mut s_dot = [0.0_f32; 4];
        let mut s_gam = [0.0_f32; 4];

        let mut r_c = r_slice[..n].chunks_exact(4);
        let mut p_c = p_slice[..n].chunks_exact(4);
        let mut g_c = g_slice[..n].chunks_exact(4);

        for ((rc, pc), gc) in r_c.by_ref().zip(p_c.by_ref()).zip(g_c.by_ref()) {
            for k in 0..4 {
                let r_obs = rc[k] - pc[k];
                let gv = gc[k];
                s_obs[k] += r_obs * r_obs;
                s_dot[k] += r_obs * gv;
                s_gam[k] += gv * gv;
            }
        }
        let mut obs_sq = (s_obs[0] + s_obs[1]) + (s_obs[2] + s_obs[3]);
        let mut pair_dot = (s_dot[0] + s_dot[1]) + (s_dot[2] + s_dot[3]);
        let mut gam_sq = (s_gam[0] + s_gam[1]) + (s_gam[2] + s_gam[3]);
        for ((&rv, &pv), &gv) in r_c
            .remainder()
            .iter()
            .zip(p_c.remainder().iter())
            .zip(g_c.remainder().iter())
        {
            let r_obs = rv - pv;
            obs_sq += r_obs * r_obs;
            pair_dot += r_obs * gv;
            gam_sq += gv * gv;
        }
        (obs_sq, pair_dot, gam_sq)
    }
}

/// Blanket trait providing branchless bitwise multiplexers, `#![no_std]` transcendentals,
/// division-free (`0 fdiv`) reciprocals, and 4/8-lane SWAR Q7 & `u8` arithmetic to all engine types.
pub trait BranchlessKernel {
    /// Branchless bitwise selection for `u8`.
    #[inline(always)]
    fn select_u8(&self, cond: bool, if_true: u8, if_false: u8) -> u8 {
        branchless_select_u8(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `i8`.
    #[inline(always)]
    fn select_i8(&self, cond: bool, if_true: i8, if_false: i8) -> i8 {
        branchless_select_i8(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `u16`.
    #[inline(always)]
    fn select_u16(&self, cond: bool, if_true: u16, if_false: u16) -> u16 {
        branchless_select_u16(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `u32`.
    #[inline(always)]
    fn select_u32(&self, cond: bool, if_true: u32, if_false: u32) -> u32 {
        branchless_select_u32(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `u64`.
    #[inline(always)]
    fn select_u64(&self, cond: bool, if_true: u64, if_false: u64) -> u64 {
        branchless_select_u64(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `usize`.
    #[inline(always)]
    fn select_usize(&self, cond: bool, if_true: usize, if_false: usize) -> usize {
        branchless_select_usize(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `i32`.
    #[inline(always)]
    fn select_i32(&self, cond: bool, if_true: i32, if_false: i32) -> i32 {
        branchless_select_i32(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `i64`.
    #[inline(always)]
    fn select_i64(&self, cond: bool, if_true: i64, if_false: i64) -> i64 {
        branchless_select_i64(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `gcso_q7_t`.
    #[inline(always)]
    fn select_q7(&self, cond: bool, if_true: gcso_q7_t, if_false: gcso_q7_t) -> gcso_q7_t {
        branchless_select_q7(cond, if_true, if_false)
    }
    /// Branchless bitwise selection for `f32`.
    #[inline(always)]
    fn select_f32(&self, cond: bool, if_true: f32, if_false: f32) -> f32 {
        branchless_select_f32(cond, if_true, if_false)
    }
    /// Branchless finiteness check for `f32`.
    #[inline(always)]
    fn is_finite_f32(&self, x: f32) -> bool {
        branchless_is_finite_f32(x)
    }
    /// Branchlessly flushes IEEE-754 subnormals and `-0.0` to `+0.0`.
    #[inline(always)]
    fn flush_subnormal_f32(&self, x: f32) -> f32 {
        branchless_flush_subnormal_f32(x)
    }
    /// Branchless IEEE-754 absolute value (`|x|`).
    #[inline(always)]
    fn abs_f32(&self, x: f32) -> f32 {
        branchless_abs_f32(x)
    }
    /// Branchless ReLU ($\max(0.0, x)$).
    #[inline(always)]
    fn relu_f32(&self, x: f32) -> f32 {
        branchless_relu_f32(x)
    }
    /// Branchless interval clamp.
    #[inline(always)]
    fn clamp_f32(&self, x: f32, lo: f32, hi: f32) -> f32 {
        branchless_clamp_f32(x, lo, hi)
    }
    /// Branchless Z-score sanitizer.
    #[inline(always)]
    fn sanitize_z_score_f32(&self, z_score: f32, max_ceiling: f32) -> f32 {
        branchless_sanitize_z_score_f32(z_score, max_ceiling)
    }
    /// Division-free (`0 fdiv`) branchless positive reciprocal ($1 / x$).
    #[inline(always)]
    fn inv_pos_f32(&self, val: f32) -> f32 {
        branchless_inv_pos_f32(val)
    }
    /// Branchless hyperbolic tangent ($\tanh(x)$).
    #[inline(always)]
    fn tanh_f32(&self, x: f32) -> f32 {
        branchless_tanh_f32(x)
    }
    /// Branchless logistic sigmoid ($\sigma(x)$).
    #[inline(always)]
    fn sigmoid_f32(&self, x: f32) -> f32 {
        branchless_sigmoid_f32(x)
    }
    /// Branchless RIPA soft-bounded phase clamp.
    #[inline(always)]
    fn ripa_clamp_rad(&self, delta_rad: f32, max_rad: f32) -> f32 {
        branchless_ripa_clamp_rad(delta_rad, max_rad)
    }
    /// Branchless natural logarithm ($\ln(x)$).
    #[inline(always)]
    fn ln_f32(&self, val: f32) -> f32 {
        branchless_ln_f32(val)
    }
    /// Branchless fast exponential ($e^x$).
    #[inline(always)]
    fn exp_f32(&self, x: f32) -> f32 {
        branchless_exp_f32(x)
    }
    /// Division-free (`0 fdiv`) branchless square root ($\sqrt{x}$).
    #[inline(always)]
    fn sqrt_f32(&self, val: f32) -> f32 {
        branchless_sqrt_f32(val)
    }
    /// Division-free (`0 fdiv`) branchless reciprocal square root ($1 / \sqrt{x}$).
    #[inline(always)]
    fn rsqrt_f32(&self, val: f32) -> f32 {
        branchless_rsqrt_f32(val)
    }
    /// Branchless fused $(\sin\theta, \cos\theta)$ for arbitrary finite angles.
    #[inline(always)]
    fn sincos_f32(&self, theta_rad: f32) -> (f32, f32) {
        branchless_sincos_f32(theta_rad)
    }
    /// Branchless direct $(\sin\theta, \cos\theta)$ for angles bounded in $[-\pi/2, \pi/2]$.
    #[inline(always)]
    fn sincos_half_pi_f32(&self, theta_rad: f32) -> (f32, f32) {
        branchless_sincos_half_pi_f32(theta_rad)
    }
    /// Division-free RIPA soft-clamped $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ from Q7.
    #[inline(always)]
    fn sincos_q7_ripa(&self, phase_q7: gcso_q7_t, max_rad: f32) -> (f32, f32) {
        branchless_sincos_q7_ripa(phase_q7, max_rad)
    }
    /// $\mathcal{O}(1)$ 256-entry LUT RIPA $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ under `DEFAULT_RIPA_CLAMP_MAX_RAD`.
    #[inline(always)]
    fn sincos_q7_ripa_default_lut(&self, phase_q7: gcso_q7_t) -> (f32, f32) {
        branchless_sincos_q7_ripa_default_lut(phase_q7)
    }
    /// $\mathcal{O}(1)$ 256-entry LUT combining $S^1$ geodesic folding and RIPA clamping.
    #[inline(always)]
    fn sincos_q7_geodesic_ripa_default_lut(&self, phase_q7: gcso_q7_t) -> (f32, f32) {
        branchless_sincos_q7_geodesic_ripa_default_lut(phase_q7)
    }
    /// Pre-inverted RIPA $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ from Q7.
    #[inline(always)]
    fn sincos_q7_ripa_preinv(
        &self,
        phase_q7: gcso_q7_t,
        max_rad: f32,
        q7_to_norm_scale: f32,
    ) -> (f32, f32) {
        branchless_sincos_q7_ripa_preinv(phase_q7, max_rad, q7_to_norm_scale)
    }
    /// Direct 256-entry LUT $(\sin\theta, \cos\theta)$ from Q7.
    #[inline(always)]
    fn sincos_q7(&self, phase_q7: gcso_q7_t) -> (f32, f32) {
        branchless_sincos_q7(phase_q7)
    }
    /// Branchless four-quadrant $\mathrm{atan2}(y, x) \in [-\pi, \pi]$.
    #[inline(always)]
    fn atan2_f32(&self, y: f32, x: f32) -> f32 {
        branchless_atan2_f32(y, x)
    }
    /// Branchless four-quadrant $\mathrm{atan2}(y, x)$ directly quantized into `gcso_q7_t`.
    #[inline(always)]
    fn atan2_to_q7_fast(&self, y: f32, x: f32) -> gcso_q7_t {
        branchless_atan2_to_q7_fast(y, x)
    }
    /// Branchless von Mises concentration parameter $\kappa_c$ from $\bar{R}_c \in [0, 1)$.
    #[inline(always)]
    fn von_mises_kappa(&self, r_bar: f32) -> f32 {
        branchless_von_mises_kappa(r_bar)
    }
    /// Evaluates circular resultant length $\bar{R}_c \in [0, 1]$ from 8 `u64` SWAR words.
    #[inline(always)]
    fn circular_r_bar_words_64(&self, words: &[u64; 8]) -> f32 {
        circular_r_bar_from_words_64(words)
    }
    /// Evaluates both 64-head phase coherence and circular resultant statistics in a single register pass.
    #[inline(always)]
    fn coherence_and_resultant_64(
        &self,
        phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> (f32, f32, gcso_q7_t, f32) {
        branchless_q7_coherence_and_resultant_64(phases_a, phases_b)
    }
    /// Evaluates rounded arithmetic mean Q7 phase across 8 `u64` SWAR words.
    #[inline(always)]
    fn q7_mean_i8_words_64(&self, words: &[u64; 8]) -> gcso_q7_t {
        swar_q7_mean_i8_from_words_64(words)
    }
    /// Executes an Online Toroidal $k$-Means M-step centroid update in 8 `u64` SWAR words.
    #[inline(always)]
    fn toroidal_centroid_step_words_64(
        &self,
        centroid_words: &[u64; 8],
        prev_sample_count: u32,
        prev_base_energy: f32,
        prev_r_bar: f32,
        prev_kappa: f32,
        sample_words: &[u64; 8],
        sample_energy: f32,
    ) -> ([u64; 8], f32, f32, f32) {
        branchless_toroidal_centroid_step_words_64(
            centroid_words,
            prev_sample_count,
            prev_base_energy,
            prev_r_bar,
            prev_kappa,
            sample_words,
            sample_energy,
        )
    }
    /// Executes an Online Toroidal $k$-Means M-step centroid update in 8 `u64` SWAR words with an optional precomputed E-step coherence.
    #[inline(always)]
    fn toroidal_centroid_step_words_with_coh_64(
        &self,
        centroid_words: &[u64; 8],
        prev_sample_count: u32,
        prev_base_energy: f32,
        prev_r_bar: f32,
        prev_kappa: f32,
        sample_words: &[u64; 8],
        precomputed_coh_opt: Option<f32>,
        sample_energy: f32,
    ) -> ([u64; 8], f32, f32, f32) {
        branchless_toroidal_centroid_step_words_with_coh_64(
            centroid_words,
            prev_sample_count,
            prev_base_energy,
            prev_r_bar,
            prev_kappa,
            sample_words,
            precomputed_coh_opt,
            sample_energy,
        )
    }
    /// Evaluates Phase-Proximity Gravity $F_{\mathrm{grav}}$ on $SO(2)^{64}$.
    #[inline(always)]
    fn phase_proximity_gravity_f32(
        &self,
        gravity_mass: f32,
        phase_coherence: f32,
        softening_eps: f32,
    ) -> f32 {
        branchless_phase_proximity_gravity_f32(gravity_mass, phase_coherence, softening_eps)
    }
    /// Projects an `&[f32]` feature/embedding slice onto $SO(2)^{64}$ (`[gcso_q7_t; 64]`).
    #[inline(always)]
    fn project_f32_slice_to_q7_64(
        &self,
        values: &[f32],
        gain: f32,
    ) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        branchless_project_f32_slice_to_q7_64(values, gain)
    }
    /// Branchless scale-invariant Eyring-Kramers tunneling escape probability.
    #[inline(always)]
    fn eyring_kramers_escape(&self, cvoid: f32, tau_eff: f32, a: f32, b: f32, d0: f32) -> f32 {
        branchless_eyring_kramers_escape(cvoid, tau_eff, a, b, d0)
    }
    /// Branchless $SO(2)$ 2D rotation operator.
    #[inline(always)]
    fn rotate_so2_pair(&self, q0: f32, q1: f32, sin_t: f32, cos_t: f32) -> (f32, f32) {
        branchless_rotate_so2_pair(q0, q1, sin_t, cos_t)
    }
    /// Branchless composite $SO(2)$ rotation + isotropic scaling operator.
    #[inline(always)]
    fn rotate_so2_pair_scaled(
        &self,
        q0: f32,
        q1: f32,
        sin_t: f32,
        cos_t: f32,
        scale: f32,
    ) -> (f32, f32) {
        branchless_rotate_so2_pair_scaled(q0, q1, sin_t, cos_t, scale)
    }
    /// Converts Q7 phase to radians.
    #[inline(always)]
    fn q7_to_rad(&self, phase_q7: gcso_q7_t) -> f32 {
        branchless_q7_to_rad(phase_q7)
    }
    /// Quantizes radian angle to Q7 format.
    #[inline(always)]
    fn rad_to_q7(&self, rad: f32) -> gcso_q7_t {
        branchless_rad_to_q7(rad)
    }
    /// Quantizes radian QDPS threshold to ceiling `u8` step.
    #[inline(always)]
    fn rad_to_qdps_step_u8(&self, min_step_rad: f32) -> u8 {
        branchless_rad_to_qdps_step_u8(min_step_rad)
    }
    /// Branchless Q7 saturating addition.
    #[inline(always)]
    fn q7_add_sat(&self, a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_add_sat(a, b)
    }
    /// Branchless Q7 saturating subtraction.
    #[inline(always)]
    fn q7_sub_sat(&self, a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_sub_sat(a, b)
    }
    /// Branchless Q7 circular difference on $S^1$.
    #[inline(always)]
    fn q7_circular_diff(&self, a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_circular_diff(a, b)
    }
    /// Branchless Q7 $S^1$ geodesic folding (`[-128, 127] -> [-64, 64]`).
    #[inline(always)]
    fn q7_fold_s1_geodesic(&self, phase_q7: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_fold_s1_geodesic(phase_q7)
    }
    /// Branchless Q7 shortest-arc circular interpolation on $S^1$.
    #[inline(always)]
    fn q7_circular_lerp(&self, a: gcso_q7_t, b: gcso_q7_t, t_u7: u8) -> gcso_q7_t {
        branchless_q7_circular_lerp(a, b, t_u7)
    }
    /// Branchless Q7 saturating negation.
    #[inline(always)]
    fn q7_neg_sat(&self, x: gcso_q7_t) -> gcso_q7_t {
        branchless_q7_neg_sat(x)
    }
    /// Branchless Q7 fixed-point scaling.
    #[inline(always)]
    fn q7_scale_u7(&self, val: gcso_q7_t, scale_u7: u8) -> gcso_q7_t {
        branchless_q7_scale_u7(val, scale_u7)
    }
    /// SWAR 4-lane Q7 saturating addition.
    #[inline(always)]
    fn q7_add_sat_x4(&self, a_packed: u32, b_packed: u32) -> u32 {
        swar_q7_add_sat_x4(a_packed, b_packed)
    }
    /// SWAR 4-lane Q7 saturating subtraction.
    #[inline(always)]
    fn q7_sub_sat_x4(&self, a_packed: u32, b_packed: u32) -> u32 {
        swar_q7_sub_sat_x4(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 saturating addition.
    #[inline(always)]
    fn q7_add_sat_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_add_sat_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 saturating subtraction.
    #[inline(always)]
    fn q7_sub_sat_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_sub_sat_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 circular phase difference.
    #[inline(always)]
    fn q7_circular_diff_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_circular_diff_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 circular phase addition.
    #[inline(always)]
    fn q7_circular_add_x8(&self, a_packed: u64, b_packed: u64) -> u64 {
        swar_q7_circular_add_x8(a_packed, b_packed)
    }
    /// SWAR 8-lane Q7 $S^1$ geodesic folding.
    #[inline(always)]
    fn q7_fold_s1_geodesic_x8(&self, packed: u64) -> u64 {
        swar_q7_fold_s1_geodesic_x8(packed)
    }
    /// SWAR 8-lane Q7 saturating negation.
    #[inline(always)]
    fn q7_neg_sat_x8(&self, packed: u64) -> u64 {
        swar_q7_neg_sat_x8(packed)
    }
    /// SWAR 8-lane Q7 shortest-arc geodesic Slerp.
    #[inline(always)]
    fn q7_geodesic_slerp_x8(&self, a_packed: u64, b_packed: u64, t_u7: u8) -> u64 {
        swar_q7_geodesic_slerp_x8(a_packed, b_packed, t_u7)
    }
    /// SWAR 8-lane Q7 fixed-point gain scaling.
    #[inline(always)]
    fn q7_scale_u7_x8(&self, packed: u64, scale_u7: u8) -> u64 {
        swar_q7_scale_u7_x8(packed, scale_u7)
    }
    /// SWAR 8-lane QDPS step filter.
    #[inline(always)]
    fn q7_qdps_filter_x8(&self, packed: u64, min_step_q7: u8) -> u64 {
        swar_q7_qdps_filter_x8(packed, min_step_q7)
    }
    /// SWAR 8-lane `u8` halving decay.
    #[inline(always)]
    fn u8_halve_decay_x8(&self, packed_u8: u64) -> u64 {
        swar_u8_halve_decay_x8(packed_u8)
    }
    /// SWAR 8-lane `u8` saturating addition.
    #[inline(always)]
    fn u8_add_sat_x8(&self, a: u64, b: u64) -> u64 {
        swar_u8_add_sat_x8(a, b)
    }
    /// SWAR 8-lane `u8` saturating subtraction.
    #[inline(always)]
    fn u8_sub_sat_x8(&self, packed_u8: u64, sub_step: u8) -> u64 {
        swar_u8_sub_sat_x8(packed_u8, sub_step)
    }
    /// SWAR 8-lane `u8` elementwise maximum.
    #[inline(always)]
    fn u8_max_x8(&self, a: u64, b: u64) -> u64 {
        swar_u8_max_x8(a, b)
    }
    /// Scalar Q7 QDPS step filter.
    #[inline(always)]
    fn qdps_filter_q7(&self, delta: gcso_q7_t, min_step_q7: i32) -> gcso_q7_t {
        branchless_qdps_filter_q7(delta, min_step_q7)
    }
    /// SplitMix64 bijective hash mixer.
    #[inline(always)]
    fn splitmix64(&self, x: u64) -> u64 {
        branchless_splitmix64(x)
    }
    /// Computes IEEE 802.3 CRC32 checksum via the 64B-aligned 4KB Slice-by-4 lookup table.
    #[inline(always)]
    fn crc32_ieee(&self, bytes: &[u8]) -> u32 {
        branchless_crc32_ieee(bytes)
    }
}

impl<T: ?Sized> BranchlessKernel for T {}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyKernel;

    #[test]
    fn test_branchless_kernel_blanket_inv_pos_crc32_and_constant_time_swar() {
        let k = DummyKernel;
        assert_eq!(k.select_u32(true, 10, 20), 10);
        assert_eq!(k.select_f32(false, 1.5, -2.5), -2.5);
        assert!((k.inv_pos_f32(4.0) - 0.25).abs() < 1.0e-6);
        assert_eq!(k.inv_pos_f32(0.0), 0.0);
        assert_eq!(k.inv_pos_f32(-2.0), 0.0);
        assert_eq!(k.inv_pos_f32(f32::INFINITY), 0.0);

        // Standard IEEE 802.3 CRC32 test vector ("123456789" -> 0xCBF43926)
        assert_eq!(k.crc32_ieee(b"123456789"), 0xCBF4_3926);
        let chained = branchless_update_crc32_ieee(branchless_crc32_ieee(b"1234"), b"56789");
        assert_eq!(chained, 0xCBF4_3926);
        for b_val in [0u8, 1, 28, 127, 128, 255] {
            let unif_bytes = [b_val; 128];
            let w_unif = [u64::from_le_bytes([b_val; 8]); 16];
            assert_eq!(
                branchless_crc32_128b_words(&w_unif),
                branchless_crc32_ieee(&unif_bytes)
            );
        }
        let long_payload = [0x5Au8; 512];
        let crc_full = branchless_crc32_ieee(&long_payload);
        let crc_split = branchless_update_crc32_ieee(
            branchless_crc32_ieee(&long_payload[..256]),
            &long_payload[256..],
        );
        assert_eq!(crc_full, crc_split);

        let a = swar_q7_pack_x8(&[100, -100, 64, -64, 127, -128, 0, 12]);
        let b = swar_q7_pack_x8(&[50, -50, -64, 64, 10, -10, 0, -12]);
        let sum = swar_q7_unpack_x8(k.q7_add_sat_x8(a, b));
        assert_eq!(sum, [127, -128, 0, 0, 127, -128, 0, 0]);

        let slerp_mid = swar_q7_unpack_x8(k.q7_geodesic_slerp_x8(
            swar_q7_pack_x8(&[-120; 8]),
            swar_q7_pack_x8(&[120; 8]),
            64,
        ));
        for lane in slerp_mid {
            assert!(lane.unsigned_abs() >= 120);
        }

        for deg in -180..=180 {
            let rad = (deg as f32) * (PI / 180.0);
            let (s, c) = branchless_sincos_f32(rad);
            let q7_ref = branchless_rad_to_q7(branchless_atan2_f32(s, c));
            let q7_fast = branchless_atan2_to_q7_fast(s, c);
            assert!((i32::from(q7_ref) - i32::from(q7_fast)).abs() <= 1);
        }
    }

    #[test]
    fn test_unified_field_fold_gravity_fold_toroidal_centroid_step_and_limina_log_prob() {
        let trail = [32i8; MAX_ATTENTION_HEADS];
        let anchors = [
            [-96i8; MAX_ATTENTION_HEADS],
            [32i8; MAX_ATTENTION_HEADS],
            [28i8; MAX_ATTENTION_HEADS],
        ];
        let energies = [0.5_f32, 3.0_f32, 2.0_f32];
        let mut folded = [0i8; MAX_ATTENTION_HEADS];

        let (tot_pull, mean_coh, dom_idx, act_cnt) =
            branchless_q7_unified_field_fold_64(&trail, &anchors, &energies, &mut folded);
        assert_eq!(dom_idx, 1);
        assert_eq!(act_cnt, 3);
        assert!(tot_pull > 8.0 && mean_coh > 0.70);
        assert!((i32::from(folded[0]) - 30).abs() <= 4);

        let mut grav_folded = [0i8; MAX_ATTENTION_HEADS];
        let (grav_pull, grav_coh, grav_dom, grav_cnt) =
            branchless_q7_gravity_fold_64(&trail, &anchors, &energies, 0.25, &mut grav_folded);
        assert_eq!(grav_dom, 1);
        assert_eq!(grav_cnt, 3);
        assert!(grav_pull > 15.0 && grav_coh > 0.85);

        let (coh_ab, r_bar, mean_q7, kappa) =
            branchless_q7_coherence_and_resultant_64(&trail, &anchors[1]);
        assert!((coh_ab - 1.0).abs() < 1.0e-4);
        let mut nonunif_a = [0i8; MAX_ATTENTION_HEADS];
        let mut nonunif_b = [71i8; MAX_ATTENTION_HEADS];
        nonunif_a[0] = 1;
        nonunif_b[0] = 72;
        let coh_71 = branchless_q7_coherence_64(&nonunif_a, &nonunif_b);
        assert!((coh_71 - (-0.170_962)).abs() < 2.0e-3);
        assert!((r_bar - 1.0).abs() < 1.0e-4);
        assert_eq!(mean_q7, 32);
        assert!(kappa > 10.0);

        let w_init = load_q7_words_64(&trail);
        let (w_step1, e_step1, rb_step1, kp_step1) =
            branchless_toroidal_centroid_step_words_64(&w_init, 0, 0.0, 0.0, 0.0, &w_init, 2.5);
        assert_eq!(w_step1, w_init);
        assert!((e_step1 - 2.5).abs() < 1.0e-5 && (rb_step1 - 1.0).abs() < 1.0e-4 && kp_step1 > 10.0);
        let (w_stat, e_stat, rb_stat, kp_stat) = branchless_toroidal_centroid_step_words_64(
            &w_step1, 1, e_step1, rb_step1, kp_step1, &w_init, 2.0,
        );
        assert_eq!(w_stat, w_init);
        assert!((e_stat - 3.5).abs() < 1.0e-5 && rb_stat == rb_step1 && kp_stat == kp_step1);

        let mut scores = [0.0_f32, 0.0_f32, 0.0_f32];
        let densities = [5.0_f32, 2.0_f32, 1.0_f32];
        let (_mean_c, best_id) = modulate_attention_scores_by_bayesian_trails_64(
            &mut scores,
            &trail,
            &anchors,
            &densities,
            Some(1),
            128,
            2.0,
            0.2,
            0.5,
        );
        assert_eq!(best_id, 1);
        assert!(scores[1] > scores[2] && scores[2] > scores[0]);

        let (coh_same, lp_same) = branchless_q7_transition_log_prob_64(&trail, &anchors[1]);
        assert!((coh_same - 1.0).abs() < 1.0e-4 && lp_same.abs() < 1.0e-4);

        let emb = [0.5_f32; 128];
        let proj = branchless_project_f32_slice_to_q7_64(&emb, 1.0);
        assert_eq!(proj[0], proj[63]);
        assert!(proj[0] > 15);
    }

    #[test]
    fn test_avx2_sincos_x8_3d_txy_lockstep_qkv_and_subnormal_norm_rescue() {
        let angles = [-3.0_f32, -1.5, -0.5, 0.0, 0.5, 1.2, 2.4, 3.1];
        let mut sin_out = [0.0_f32; 8];
        let mut cos_out = [0.0_f32; 8];
        sincos_f32_x8_simd(&angles, &mut sin_out, &mut cos_out);
        for i in 0..8 {
            let (s_ref, c_ref) = branchless_sincos_f32(angles[i]);
            assert!((sin_out[i] - s_ref).abs() < 1.0e-4);
            assert!((cos_out[i] - c_ref).abs() < 1.0e-4);
        }

        let mut grid_3d = [1.0_f32; 48];
        let (st, ct) = branchless_sincos_q7(16);
        let (sx, cx) = branchless_sincos_q7(32);
        let (sy, cy) = branchless_sincos_q7(-16);
        rotate_so2_3d_txy_slice_simd(&mut grid_3d, st, ct, sx, cx, sy, cy);
        for hex in grid_3d.chunks_exact(6) {
            assert!(((hex[0] * hex[0] + hex[1] * hex[1]) - 2.0).abs() < 1.0e-5);
            assert!(((hex[2] * hex[2] + hex[3] * hex[3]) - 2.0).abs() < 1.0e-5);
            assert!(((hex[4] * hex[4] + hex[5] * hex[5]) - 2.0).abs() < 1.0e-5);
        }

        let mut q = [1.0_f32; 64];
        let mut k = [1.0_f32; 64];
        let mut v = [1.0_f32; 64];
        rotate_so2_slice_unrolled16(&mut q, st, ct);
        rotate_so2_slice_unrolled16(&mut k, st, ct);
        rotate_so2_slice_unrolled16(&mut v, st, ct);
        assert!((q[0] - k[0]).abs() < 1.0e-6 && (q[0] - v[0]).abs() < 1.0e-6);

        // Verify constant-time identity rotation preserves exact values
        let mut q_id = [1.5_f32; 64];
        rotate_so2_slice_unrolled16(&mut q_id, 0.0, 1.0);
        assert_eq!(q_id, [1.5_f32; 64]);

        // Verify AVX2 inline subnormal flushing in `scale_f32_slice_unrolled8`
        let mut sub_buf = [2.0e-38_f32; 64];
        scale_f32_slice_unrolled8(&mut sub_buf, 0.5);
        assert_eq!(sub_buf[0], 0.0);

        let tiny = [1.0e-22_f32; 64];
        let (norm, ok) = norm_l2_underflow_resilient_f32(&tiny);
        assert!(ok && (norm - 8.0e-22).abs() < 1.0e-26);

        let sample_vec = [-2.0_f32, 1.0, 3.0, -4.0];
        let (s_abs, s_sq, m_abs) = sum_abs_sq_and_max_f32_simd8(&sample_vec);
        assert!((s_abs - 10.0).abs() < 1.0e-5);
        assert!((s_sq - 30.0).abs() < 1.0e-5);
        assert!((m_abs - 4.0).abs() < 1.0e-5);

        // Verify single-evaluation fused RIPA + PSPM angle helper
        let scale_018 = Q7_PHASE_SCALE_RAD * branchless_inv_pos_f32(0.18);
        let (s1, c1) = eval_ripa_sincos_q7_fast(28, false, 0.18, scale_018);
        let (s2, c2) = eval_ripa_sincos_q7_fast(12, false, 0.18, scale_018);
        let (sf, cf) = eval_fused_ripa_pspm_sincos_q7_fast(
            28, false, false, 0.18, scale_018, 12, false, false, 0.18, scale_018,
        );
        assert!((sf - (s1 * c2 + c1 * s2)).abs() < 1.0e-5);
        assert!((cf - (c1 * c2 - s1 * s2)).abs() < 1.0e-5);

        // Verify 1KB GF(2) CRC32_UNIFORM56_PAD8_ZERO_SEED_REF and kappa CRC32 constants
        assert_eq!(
            CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB,
            crc32_step_l5(crc32_slice16_dep_lo0(VON_MISES_KAPPA_AT_MAX_R_BITS))
        );
        assert_eq!(
            CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB,
            crc32_step_l5(crc32_slice16_dep_lo0(0x4248_0000u32))
        );
        let w_trail_blk1 = [0x1818_1818_1818_1818u64, 0x1818_1818_1818_1818u64, 0x1818_1818_1818_1818u64,
            0x1818_1818_1818_1818u64, 0x1818_1818_1818_1818u64, 0x1818_1818_1818_1818u64,
            0x1818_1818_1818_1818u64, 0u64];
        let mut raw_blk1 = [0x18u8; 64];
        raw_blk1[56..64].fill(0);
        assert_eq!(
            crc32_block64_zero_seed_words(&w_trail_blk1),
            CRC32_UNIFORM56_PAD8_ZERO_SEED_REF[0x18]
        );
        assert_eq!(
            crc32_step_block64_words(0xFFFF_FFFFu32, &w_trail_blk1) ^ 0xFFFF_FFFFu32,
            branchless_crc32_ieee(&raw_blk1)
        );

        // Verify AVX2 / scalar RIPA angle clamp parity across finite, extreme, and non-finite inputs
        let mut angles_simd = [-12.0_f32, -0.5, -0.18, -1.0e-39, 0.0, 0.09, 0.18, 0.5, 12.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let mut angles_ref = angles_simd;
        let max_rad = 0.18_f32;
        let inv_max = branchless_inv_pos_f32(max_rad);
        for a in angles_ref.iter_mut() {
            let clean = branchless_select_f32(branchless_is_finite_f32(*a), *a, 0.0);
            *a = branchless_flush_subnormal_f32(max_rad * branchless_tanh_f32(clean * inv_max));
        }
        clamp_angles_ripa_in_place_simd8(&mut angles_simd, max_rad);
        for (got, exp) in angles_simd.iter().zip(angles_ref.iter()) {
            assert!((got - exp).abs() <= 1.0e-6, "got={} exp={}", got, exp);
        }
    }
}