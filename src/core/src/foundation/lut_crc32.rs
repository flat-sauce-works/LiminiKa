// name: src/core/src/foundation/lut_crc32.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Static 64B-aligned lookup tables (Q7, TANH, Slice-by-16 CRC32) and branchless IEEE 802.3 CRC32 updaters.

use crate::foundation::scalar_math::{
    DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE, VON_MISES_KAPPA_AT_MAX_R,
    branchless_flush_subnormal_f32, branchless_max_f32, branchless_q7_abs_u8,
    branchless_q7_fold_s1_geodesic, branchless_relu_f32, branchless_rsqrt_f32,
    fast_inv_pos_normal_f32,
};
use crate::liminal::{
    PhaseQ7, Q7_PHASE_SCALE_RAD, branchless_select_f32, branchless_select_u8,
    branchless_select_usize,
};

// ===================================================================
// 7. Static 64B-Aligned Lookup Tables, Branchless Math & 8-Lane SWAR Primitives
// ===================================================================

/// 64-bit SWAR sign-bit mask (`0x80` per byte lane across 8 lanes).
pub const SWAR_SIGN_MASK_64: u64 = 0x8080_8080_8080_8080;
/// 64-bit SWAR lower-7-bit payload mask (`0x7F` per byte lane across 8 lanes).
pub const SWAR_LOW7_MASK_64: u64 = 0x7F7F_7F7F_7F7F_7F7F;
/// 64-bit SWAR even-byte isolation mask (`0x00FF` per 16-bit lane across 4 lanes).
pub const SWAR_EVEN_BYTES_64: u64 = 0x00FF_00FF_00FF_00FF;
/// 64-bit SWAR odd-byte isolation mask (`0xFF00` per 16-bit lane across 4 lanes).
pub const SWAR_ODD_BYTES_64: u64 = 0xFF00_FF00_FF00_FF00;
/// 64-bit SWAR byte-splat multiplier (`0x01` per byte lane across 8 lanes).
pub const SWAR_SPLAT_BYTE_64: u64 = 0x0101_0101_0101_0101;
/// 64-bit SWAR 16-bit lane horizontal reduction multiplier (`0x0001` per 16-bit lane).
pub const SWAR_SUM_U16_64: u64 = 0x0001_0001_0001_0001;
/// 32-bit SWAR sign-bit mask (`0x80` per byte lane across 4 lanes).
pub const SWAR_SIGN_MASK_32: u32 = 0x8080_8080;
/// 32-bit SWAR lower-7-bit payload mask (`0x7F` per byte lane across 4 lanes).
pub const SWAR_LOW7_MASK_32: u32 = 0x7F7F_7F7F;
pub(crate) const INV_TWO_PI_F64: f64 = 1.0 / (2.0 * core::f64::consts::PI);
pub(crate) const TWO_PI_F64: f64 = 2.0 * core::f64::consts::PI;
pub(crate) const INV_LN2_F32: f32 = core::f32::consts::LOG2_E;
pub(crate) const LN2_HI_F32: f32 = 0.693_145_75;
pub(crate) const LN2_LO_F32: f32 = 1.428_606_8e-6;
#[cfg(test)]
pub(crate) const MAX_BAYESIAN_STIGMERGIC_DENSITY: f32 = 16.0;
pub(crate) const INV_64_Q14_F32: f32 = f32::from_bits((127 - 20) << 23);
pub(crate) const INV_8_Q14_F32: f32 = f32::from_bits((127 - 17) << 23);
pub(crate) const SUBNORMAL_SQ_RISK_FLOOR: f32 = 1.0e-18;
pub(crate) const Q14_BIAS_I32: i32 = 16_384;
pub(crate) const Q14_BIAS_X8_I32: i32 = 8 * Q14_BIAS_I32;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub(crate) const Q14_BIAS_X64_I32: i32 = 64 * Q14_BIAS_I32;

/// Minimum positive normal IEEE-754 `f32` value ($2^{-126} \approx 1.17549435 \times 10^{-38}$).
pub const F32_MIN_POSITIVE_NORMAL: f32 = f32::from_bits(0x0080_0000);
/// Default upper ceiling for token Z-score inputs ($10.0\sigma$).
pub const DEFAULT_MAX_VALID_Z_SCORE: f32 = 10.0;
/// Minimum clamped inverse-temperature scale for destructive phase interference ($0.25$).
pub const MIN_PHASE_INTERFERENCE_INV_TEMP: f32 = 0.25;
/// Maximum clamped inverse-temperature scale for constructive phase interference ($4.0$).
pub const MAX_PHASE_INTERFERENCE_INV_TEMP: f32 = 4.0;
/// Default softening epsilon $\epsilon_{\mathrm{grav}}$ for Phase-Proximity Gravity on $SO(2)^{64}$ ($0.25$).
pub const DEFAULT_PHASE_GRAVITY_EPS: f32 = 0.25;

pub(crate) const TANH_LUT_MAX_X: f32 = 4.97;
pub(crate) const TANH_LUT_STEP: f32 = TANH_LUT_MAX_X / 256.0;
pub(crate) const TANH_LUT_INV_STEP: f32 = 256.0 / TANH_LUT_MAX_X;

#[repr(C, align(64))]
pub(crate) struct AlignedLut256<T>(pub [T; 256]);

#[repr(C, align(64))]
pub(crate) struct AlignedLut257<T>(pub [T; 257]);

#[repr(C, align(64))]
pub(crate) struct AlignedLut2x257<T>(pub [[T; 257]; 2]);

pub(crate) const VON_MISES_KAPPA_SPLIT_R: f32 = 0.875;
pub(crate) const VON_MISES_KAPPA_MAX_R: f32 = 0.995;
pub(crate) const VON_MISES_KAPPA_LO_STEP: f32 = VON_MISES_KAPPA_SPLIT_R / 256.0;
pub(crate) const VON_MISES_KAPPA_HI_STEP: f32 =
    (VON_MISES_KAPPA_MAX_R - VON_MISES_KAPPA_SPLIT_R) / 256.0;
pub(crate) const VON_MISES_KAPPA_LO_SCALE: f32 = 256.0 / VON_MISES_KAPPA_SPLIT_R;
pub(crate) const VON_MISES_KAPPA_HI_SCALE: f32 =
    256.0 / (VON_MISES_KAPPA_MAX_R - VON_MISES_KAPPA_SPLIT_R);

#[repr(C, align(64))]
pub(crate) struct AlignedLut4x256<T>(pub [[T; 256]; 4]);

#[repr(C, align(64))]
pub(crate) struct AlignedLut16x256<T>(pub [[T; 256]; 16]);

/// 64B-aligned 16KB Slice-by-16 IEEE 802.3 CRC32 lookup table (`0xEDB8_8320` reflected polynomial)
/// enabling 16-byte (`2 x u64`) parallel XOR-table folding per step (`16x` fewer serial dependency
/// steps than byte-serial lookup and `2x` fewer than Slice-by-8 on `.lmnk` mapped-storage containers and
/// 128-byte `PhaseAnchorRecord` blocks).
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

pub(crate) const Q7_QUADRANT_SIN_LUT: [f32; 65] = {
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
        let phase_q7 = (b as u8) as PhaseQ7;
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
        let phase_q7 = (b as u8) as PhaseQ7;
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

pub(crate) const TANH_LUT_257: AlignedLut257<f32> = {
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
pub(crate) const fn sincos_phase_clamp_poly(safe_theta: f32) -> (f32, f32) {
    let x2 = safe_theta * safe_theta;
    let sin_t =
        safe_theta * (1.0 + x2 * (-0.166_666_67 + x2 * (0.008_333_334 - x2 * 0.000_198_412_7)));
    let cos_t =
        1.0 + x2 * (-0.5 + x2 * (0.041_666_668 + x2 * (-0.001_388_888_9 + x2 * 0.000_024_801_587)));
    (sin_t, cos_t)
}

pub(crate) const Q7_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256: AlignedLut256<(f32, f32)> = {
    let mut lut = [(0.0_f32, 1.0_f32); 256];
    let mut b = 0usize;
    while b < 256 {
        let phase_q7 = (b as u8) as PhaseQ7;
        let u = branchless_q7_abs_u8(phase_q7);
        let safe_theta = DEFAULT_PHASE_CLAMP_MAX_RAD
            * tanh_lut_eval_nonneg((u as f32) * DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE);
        let (sin_mag, cos_t) = sincos_phase_clamp_poly(safe_theta);
        let sin_neg = (phase_q7 < 0) & (u != 0);
        lut[b] = (
            f32::from_bits(sin_mag.to_bits() | ((sin_neg as u32) << 31)),
            cos_t,
        );
        b += 1;
    }
    AlignedLut256(lut)
};

pub(crate) const Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256: AlignedLut256<(f32, f32)> = {
    let mut table = [(0.0_f32, 1.0_f32); 256];
    let mut idx = 0usize;
    while idx < 256 {
        let folded = branchless_q7_fold_s1_geodesic(idx as i8);
        table[idx] = Q7_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256.0[(folded as u8) as usize];
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

/// 64B-aligned 2KB precomputed `(align_gain, inv_denom)` table for [`crate::foundation::scalar_math::branchless_phase_proximity_gravity_f32`]
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
/// Promoted `.rodata` reference to the 64B-aligned 256-entry default `PhaseClamp` `(sin, cos)` table.
pub(crate) const Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry geodesic default `PhaseClamp` `(sin, cos)` table.
pub(crate) const Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256.0;

#[inline(always)]
pub(crate) const fn build_q7_fused_self_phase_clamp_sincos_lut(
    fold_geodesic: bool,
) -> AlignedLut256<(f32, f32)> {
    let mut table = [(0.0_f32, 1.0_f32); 256];
    let mut i = 0usize;
    while i < 256 {
        let (sin_base, cos_base) = Q7_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256.0[i];
        let (sin_subhead, cos_subhead) = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256.0[i]
        } else {
            Q7_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256.0[i]
        };
        let sin_comb = sin_base * cos_subhead + cos_base * sin_subhead;
        let cos_comb = cos_base * cos_subhead - sin_base * sin_subhead;
        table[i] = (sin_comb, cos_comb);
        i += 1;
    }
    AlignedLut256(table)
}

/// Precomputed 64-byte-aligned 2KB LUT of fused $SO(2)$ double-rotation `(sin(2*theta), cos(2*theta))`
/// when `base_clamp_q7 == subhead_slot_q7` under `DEFAULT_PHASE_CLAMP_MAX_RAD` (non-geodesic sub-head router).
pub(crate) const Q7_FUSED_SELF_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256: AlignedLut256<(f32, f32)> =
    build_q7_fused_self_phase_clamp_sincos_lut(false);

/// Precomputed 64-byte-aligned 2KB LUT of fused $SO(2)$ double-rotation `(sin(theta_b + theta_p), cos(theta_b + theta_p))`
/// when `base_clamp_q7 == subhead_slot_q7` under `DEFAULT_PHASE_CLAMP_MAX_RAD` (geodesic sub-head router).
pub(crate) const Q7_FUSED_SELF_GEODESIC_PHASE_CLAMP_SINCOS_LUT_256: AlignedLut256<(f32, f32)> =
    build_q7_fused_self_phase_clamp_sincos_lut(true);

pub(crate) const Q7_FUSED_SELF_DEFAULT_PHASE_CLAMP_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_FUSED_SELF_DEFAULT_PHASE_CLAMP_SINCOS_LUT_256.0;

pub(crate) const Q7_FUSED_SELF_GEODESIC_PHASE_CLAMP_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_FUSED_SELF_GEODESIC_PHASE_CLAMP_SINCOS_LUT_256.0;
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
pub(crate) const fn crc32_slice4_step(crc: u32, b0: u8, b1: u8, b2: u8, b3: u8) -> u32 {
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
    (lut[15][(lo0 & 0xFF) as usize] ^ lut[14][((lo0 >> 8) & 0xFF) as usize])
        ^ (lut[13][((lo0 >> 16) & 0xFF) as usize] ^ lut[12][(lo0 >> 24) as usize])
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
/// enabling single-step 64B block folding (`128B` `PhaseAnchorRecord` in `2` parallel 64B halves).
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
/// `PhaseAnchorRecord` arrays and 256B ($4 \times 64\text{B}$) container blocks.
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

/// Promoted `.rodata` reference to the 4KB $L^8(x)$ 128-byte shift table.
pub(crate) const CRC32_IEEE_L8_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L8_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^{16}(x)$ 256-byte shift table.
pub(crate) const CRC32_IEEE_L16_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L16_LUT.0;

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

/// Exact IEEE-754 `f32::to_bits()` representation of `VON_MISES_KAPPA_AT_MAX_R` ($\bar{R}_c \ge 0.995$).
pub(crate) const VON_MISES_KAPPA_AT_MAX_R_BITS: u32 = VON_MISES_KAPPA_AT_MAX_R.to_bits();

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
pub(crate) const fn crc32_uniform_word_zero_seed_raw(w0: u64) -> u32 {
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
pub(crate) const fn crc32_uniform56_pad8_zero_seed_raw(w0: u64) -> u32 {
    let u64_crc = crc32_uniform_word_zero_seed_raw(w0);
    let lo7 = w0 as u32;
    let hi7 = (w0 >> 32) as u32;
    u64_crc
        ^ (if lo7 == 0 {
            0
        } else {
            crc32_slice16_indep_lo1_u32(lo7)
        })
        ^ (if hi7 == 0 {
            0
        } else {
            crc32_slice16_indep_hi1_u32(hi7)
        })
}

/// 64B-aligned 1KB GF(2)-linear lookup table mapping any uniform byte `b in 0..=255` to the exact
/// unseeded 64-byte IEEE 802.3 CRC32 operator over 56 uniform bytes followed by 8 zero bytes
/// (`[b; 56] ++ [0; 8]`).
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
/// and 56-byte uniform prefixes followed by 8 zero tail bytes.
/// Evaluates the zero-seed IEEE 802.3 CRC32 of a 64-byte block `[w0, w0, w0, w0, w0, w0, w0, w7]`
/// whose first 7 64-bit words are already known to equal `w0` (`0` duplicate uniformity checks).
#[inline(always)]
#[must_use]
pub(crate) const fn crc32_block64_uniform_prefix56_zero_seed(w0: u64, w7: u64) -> u32 {
    if w0 == 0 {
        if w7 == 0 {
            return 0;
        }
        let lo7 = w7 as u32;
        let hi7 = (w7 >> 32) as u32;
        return (if lo7 == 0 {
            0
        } else {
            crc32_slice16_indep_lo1_u32(lo7)
        }) ^ (if hi7 == 0 {
            0
        } else {
            crc32_slice16_indep_hi1_u32(hi7)
        });
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
        ^ (if lo7 == 0 {
            0
        } else {
            crc32_slice16_indep_lo1_u32(lo7)
        })
        ^ (if hi7 == 0 {
            0
        } else {
            crc32_slice16_indep_hi1_u32(hi7)
        })
}

/// Evaluates the raw IEEE 802.3 CRC32 operator over 8 native `u64` words (64 bytes)
/// starting from initial seed `0`, returning the unfinalized accumulator state.
#[inline(always)]
#[must_use]
pub const fn crc32_block64_zero_seed_words(w: &[u64; 8]) -> u32 {
    let w0 = w[0];
    if w0 == w[1] && (((w0 ^ w[2]) | (w0 ^ w[3])) | ((w0 ^ w[4]) | (w0 ^ w[5])) | (w0 ^ w[6])) == 0
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
    let s2 = if p2 == 0 {
        0
    } else {
        crc32_slice16_dep_lo0(p2)
    };
    (s0 ^ s1) ^ (s2 ^ p3)
}

#[inline(always)]
pub(crate) const fn crc32_step_block64_words(crc: u32, w: &[u64; 8]) -> u32 {
    if crc == 0 {
        return crc32_block64_zero_seed_words(w);
    }
    let w0 = w[0];
    if w0 == w[1] && (((w0 ^ w[2]) | (w0 ^ w[3])) | ((w0 ^ w[4]) | (w0 ^ w[5])) | (w0 ^ w[6])) == 0
    {
        let shifted_seed = if crc == 0xFFFF_FFFFu32 {
            const L4_INIT: u32 = crc32_step_l4(0xFFFF_FFFFu32);
            L4_INIT
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
    let s2 = if p2 == 0 {
        0
    } else {
        crc32_slice16_dep_lo0(p2)
    };
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
pub(crate) const fn load_u64_le_bytes_at(bytes: &[u8], i: usize) -> u64 {
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

/// Incrementally updates an IEEE 802.3 CRC32 checksum (`0xEDB88320` reflected polynomial) over `bytes`
/// starting from a previous finalized CRC value `crc_seed` (`0` for initial chunk) using the
/// 64B-aligned 256-entry lookup table (`CRC32_IEEE_LUT_256`) with 8-byte unrolled loop execution.
#[inline(always)]
#[must_use]
pub const fn update_crc32_ieee(crc_seed: u32, bytes: &[u8]) -> u32 {
    branchless_update_crc32_ieee(crc_seed, bytes)
}

/// Computes a standard IEEE 802.3 CRC32 checksum (`0xEDB88320` reflected polynomial) over `bytes`
/// using the 64B-aligned 256-entry lookup table (`CRC32_IEEE_LUT_256`) in `#![no_std]` `const fn` mode.
#[inline(always)]
#[must_use]
pub const fn compute_crc32_ieee(bytes: &[u8]) -> u32 {
    branchless_update_crc32_ieee(0, bytes)
}

#[inline(always)]
#[must_use]
fn read_block64_le_words(block: &[u8]) -> [u64; 8] {
    [
        u64::from_le_bytes([
            block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7],
        ]),
        u64::from_le_bytes([
            block[8], block[9], block[10], block[11], block[12], block[13], block[14], block[15],
        ]),
        u64::from_le_bytes([
            block[16], block[17], block[18], block[19], block[20], block[21], block[22], block[23],
        ]),
        u64::from_le_bytes([
            block[24], block[25], block[26], block[27], block[28], block[29], block[30], block[31],
        ]),
        u64::from_le_bytes([
            block[32], block[33], block[34], block[35], block[36], block[37], block[38], block[39],
        ]),
        u64::from_le_bytes([
            block[40], block[41], block[42], block[43], block[44], block[45], block[46], block[47],
        ]),
        u64::from_le_bytes([
            block[48], block[49], block[50], block[51], block[52], block[53], block[54], block[55],
        ]),
        u64::from_le_bytes([
            block[56], block[57], block[58], block[59], block[60], block[61], block[62], block[63],
        ]),
    ]
}

#[inline(always)]
#[must_use]
pub(crate) fn update_aligned64_blocks_crc32_raw(mut crc: u32, bytes: &[u8]) -> u32 {
    let len = bytes.len();
    let num_blocks = len >> 6;
    let mut b = 0usize;
    while b + 4 <= num_blocks {
        let base = b << 6;
        let w0 = read_block64_le_words(&bytes[base..base + 64]);
        let w1 = read_block64_le_words(&bytes[base + 64..base + 128]);
        let w2 = read_block64_le_words(&bytes[base + 128..base + 192]);
        let w3 = read_block64_le_words(&bytes[base + 192..base + 256]);
        let z0 = crc32_block64_zero_seed_words(&w0);
        let z1 = crc32_block64_zero_seed_words(&w1);
        let z01 = (if z0 == 0 { 0 } else { crc32_step_l4(z0) }) ^ z1;
        let z2 = crc32_block64_zero_seed_words(&w2);
        let z3 = crc32_block64_zero_seed_words(&w3);
        let c23 = (if z2 == 0 { 0 } else { crc32_step_l4(z2) }) ^ z3;
        crc = crc32_step_l16(crc) ^ (if z01 == 0 { 0 } else { crc32_step_l8(z01) }) ^ c23;
        b += 4;
    }
    while b + 2 <= num_blocks {
        let base = b << 6;
        let w0 = read_block64_le_words(&bytes[base..base + 64]);
        let w1 = read_block64_le_words(&bytes[base + 64..base + 128]);
        let z0 = crc32_block64_zero_seed_words(&w0);
        let z1 = crc32_block64_zero_seed_words(&w1);
        let z01 = (if z0 == 0 { 0 } else { crc32_step_l4(z0) }) ^ z1;
        crc = crc32_step_l8(crc) ^ z01;
        b += 2;
    }
    if b < num_blocks {
        let base = b << 6;
        let w0 = read_block64_le_words(&bytes[base..base + 64]);
        crc = crc32_step_block64_words(crc, &w0);
    }
    crc
}

#[inline(always)]
#[must_use]
pub(crate) fn compute_aligned64_payload_crc32_ieee(bytes: &[u8]) -> u32 {
    if (bytes.len() & 63) == 0 {
        return !update_aligned64_blocks_crc32_raw(0xFFFF_FFFFu32, bytes);
    }
    branchless_crc32_ieee(bytes)
}
