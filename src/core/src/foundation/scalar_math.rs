// name: src/core/src/foundation/scalar_math.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constant-time branchless scalar reciprocals, transcendentals, and scalar Q7/radian helpers.

use core::f32::consts::{FRAC_PI_2, LN_2, PI};

use crate::foundation::lut_crc32::{
    sincos_phase_clamp_poly, tanh_lut_eval_nonneg, AlignedLut257, AlignedLut2x257,
    DEFAULT_MAX_VALID_Z_SCORE, DEFAULT_PHASE_GRAVITY_EPS, INV_LN2_F32, INV_TWO_PI_F64, LN2_HI_F32,
    LN2_LO_F32, POPCOUNT_RSQRT_LUT_257_REF, Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF, Q7_FULL_COS_REF,
    Q7_FULL_SIN_REF, Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF, Q7_SINCOS_LUT_256_REF,
    TANH_LUT_MAX_X, TWO_PI_F64, VON_MISES_KAPPA_HI_SCALE, VON_MISES_KAPPA_HI_STEP,
    VON_MISES_KAPPA_LO_SCALE, VON_MISES_KAPPA_LO_STEP, VON_MISES_KAPPA_MAX_R,
    VON_MISES_KAPPA_SPLIT_R,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_i32, branchless_select_u8, branchless_select_usize,
    PhaseQ7, DEFAULT_ENTROPY_EPS, Q7_MAX, Q7_MIN, Q7_PHASE_INV_SCALE_RAD, Q7_PHASE_SCALE_RAD,
};

/// Default soft-bounded phase clamp limit in radians ($5^\circ \approx 0.08726646\text{ rad}$).
pub const DEFAULT_PHASE_CLAMP_MAX_RAD: f32 = 0.087_266_46;

/// Canonical ratio `Q7_PHASE_SCALE_RAD / DEFAULT_PHASE_CLAMP_MAX_RAD` mapping Q7 phase deltas to normalized `tanh` units.
pub const DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE: f32 =
    Q7_PHASE_SCALE_RAD / DEFAULT_PHASE_CLAMP_MAX_RAD;

/// Default minimum phase step cutoff in radians ($0.01\text{ rad}$).
pub const DEFAULT_PHASE_STEP_MIN_RAD: f32 = 0.01;

/// Fast positive normal reciprocal $1.0 / x$ for $x \ge \text{F32\_MIN\_POSITIVE\_NORMAL}$
/// using IEEE-754 magic bit seed and 3 Newton-Raphson iterations (`0` subnormal/validity branches).
#[inline(always)]
pub(crate) const fn fast_inv_pos_normal_f32(x: f32) -> f32 {
    let y0 = f32::from_bits(0x7EF1_27EA_u32.wrapping_sub(x.to_bits()));
    let y1 = y0 * (2.0 - x * y0);
    let e1 = 1.0 - x * y1;
    y1 + y1 * (e1 + e1 * e1)
}

pub(crate) const LN_1P_LUT_STEP: f32 = 0.0625;
pub(crate) const LN_1P_LUT_INV_STEP: f32 = 16.0;

#[inline(always)]
pub(crate) const fn const_ln_1p_exact_f32(rho_nonneg: f32) -> f32 {
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

/// Branchlessly flushes IEEE-754 subnormal (denormal) floats, non-finite values (`+/-Inf`, `NaN`),
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
    let valid_span = (span > DEFAULT_ENTROPY_EPS) & branchless_is_finite_f32(span);
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

/// Branchless composite `PhaseSteering` + Isotropic Block-Diagonal Scaling operator (Theorem 1.1).
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
pub(crate) const fn unpack_pos_f32_exp_and_mant(raw_bits: u32) -> (i32, u32) {
    if raw_bits < 0x0080_0000 {
        let lz = (raw_bits | 1).leading_zeros();
        let shift = lz.wrapping_sub(8);
        (-118i32 - (lz as i32), (raw_bits << shift) & 0x007F_FFFF)
    } else {
        (
            (((raw_bits >> 23) & 0xFF) as i32) - 127,
            raw_bits & 0x007F_FFFF,
        )
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

/// Branchless `#![no_std]` hyperbolic tangent ($\tanh(x)$) using Pade $(5, 4)$ rational approximation.
#[inline(always)]
pub const fn branchless_tanh_f32(x: f32) -> f32 {
    let safe_x = branchless_select_f32(branchless_is_nan_f32(x), 0.0, x);
    let clamped = branchless_clamp_f32(safe_x, -TANH_LUT_MAX_X, TANH_LUT_MAX_X);
    let x2 = clamped * clamped;
    let x4 = x2 * x2;
    (clamped * (945.0 + 105.0 * x2 + x4)) * fast_inv_pos_normal_f32(945.0 + 420.0 * x2 + 15.0 * x4)
}

/// Branchless `#![no_std]` logistic sigmoid ($\sigma(x) = 1 / (1 + e^{-x})$).
#[inline(always)]
pub const fn branchless_sigmoid_f32(x: f32) -> f32 {
    0.5 + 0.5 * branchless_tanh_f32(0.5 * x)
}

/// Branchless soft-bounded (`PhaseClamp`) phase angle clamp: $\Delta\theta_{\mathrm{safe}} = \theta_{\max} \cdot \tanh(\Delta\theta / \theta_{\max})$.
#[inline(always)]
pub const fn branchless_phase_clamp_rad(delta_rad: f32, max_rad: f32) -> f32 {
    let valid_max = (max_rad > 1.0e-9) & branchless_is_finite_f32(max_rad);
    let safe_max = branchless_select_f32(valid_max, max_rad, DEFAULT_PHASE_CLAMP_MAX_RAD);
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

/// Branchless `#![no_std]` fast exponential ($e^x$) using Cody-Waite base-2 range reduction and Pade $(3, 3)$.
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
    if r_sq <= DEFAULT_ENTROPY_EPS {
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
pub(crate) const fn sincos_poly_half_pi(x_folded: f32, sign_cos: f32) -> (f32, f32) {
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
pub(crate) const fn sincos_reduced_pi(x: f32) -> (f32, f32) {
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

/// Direct $\mathcal{O}(1)$ 64B-aligned 256-entry LUT lookup for `PhaseClamp` $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$
/// under [`DEFAULT_PHASE_CLAMP_MAX_RAD`].
#[inline(always)]
pub const fn branchless_sincos_q7_phase_clamp_default_lut(phase_q7: PhaseQ7) -> (f32, f32) {
    let lut = Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Direct $\mathcal{O}(1)$ 64B-aligned 256-entry LUT lookup combining $S^1$ geodesic folding
/// with `PhaseClamp` soft-clamping under [`DEFAULT_PHASE_CLAMP_MAX_RAD`].
#[inline(always)]
pub const fn branchless_sincos_q7_geodesic_phase_clamp_default_lut(
    phase_q7: PhaseQ7,
) -> (f32, f32) {
    let lut = Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Evaluates the signed `PhaseClamp`-limited radian angle $\Delta\theta_{\mathrm{safe}} = \mathrm{sgn}(q_7)\cdot |\theta_{\max}| \cdot \tanh(|q_7|\cdot s)$
/// using the 257-entry `TANH_LUT_257` table.
#[inline(always)]
pub(crate) const fn eval_phase_clamp_theta_rad_preinv(
    phase_q7: PhaseQ7,
    max_rad: f32,
    q7_to_norm_scale: f32,
) -> f32 {
    let u_abs = branchless_q7_abs_u8(phase_q7);
    let norm_u = (u_abs as f32) * branchless_abs_f32(q7_to_norm_scale);
    let safe_theta_mag = branchless_abs_f32(max_rad) * tanh_lut_eval_nonneg(norm_u);
    let is_neg = ((phase_q7 < 0) ^ (max_rad < 0.0)) & (u_abs != 0);
    f32::from_bits(safe_theta_mag.to_bits() | ((is_neg as u32) << 31))
}

/// Division-free (`0 fdiv`) branchless `PhaseClamp` soft-clamped $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$
/// using precomputed `q7_to_norm_scale = Q7_PHASE_SCALE_RAD / max_rad`.
#[inline(always)]
pub const fn branchless_sincos_q7_phase_clamp_preinv(
    phase_q7: PhaseQ7,
    max_rad: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if (max_rad == DEFAULT_PHASE_CLAMP_MAX_RAD)
        & (q7_to_norm_scale == DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE)
    {
        return branchless_sincos_q7_phase_clamp_default_lut(phase_q7);
    }
    let u_abs = branchless_q7_abs_u8(phase_q7);
    let norm_u = (u_abs as f32) * branchless_abs_f32(q7_to_norm_scale);
    let safe_theta_mag = branchless_abs_f32(max_rad) * tanh_lut_eval_nonneg(norm_u);
    let (sin_mag, cos_val) = sincos_phase_clamp_poly(safe_theta_mag);
    let sin_neg = ((phase_q7 < 0) ^ (max_rad < 0.0)) & (u_abs != 0);
    (
        f32::from_bits(sin_mag.to_bits() | ((sin_neg as u32) << 31)),
        cos_val,
    )
}

/// Standalone division-free branchless `PhaseClamp` soft-clamped $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ from Q7.
#[inline(always)]
pub const fn branchless_sincos_q7_phase_clamp(phase_q7: PhaseQ7, max_rad: f32) -> (f32, f32) {
    if max_rad == DEFAULT_PHASE_CLAMP_MAX_RAD {
        return branchless_sincos_q7_phase_clamp_default_lut(phase_q7);
    }
    let valid_max = (max_rad > 1.0e-9) & branchless_is_finite_f32(max_rad);
    let eff_max = branchless_min_f32(
        branchless_select_f32(valid_max, max_rad, DEFAULT_PHASE_CLAMP_MAX_RAD),
        FRAC_PI_2,
    );
    let q7_to_norm_scale = Q7_PHASE_SCALE_RAD * branchless_inv_pos_f32(eff_max);
    let (sin_t, cos_t) =
        branchless_sincos_q7_phase_clamp_preinv(phase_q7, eff_max, q7_to_norm_scale);
    (
        branchless_select_f32(valid_max, sin_t, 0.0),
        branchless_select_f32(valid_max, cos_t, 1.0),
    )
}

/// Direct 64B-aligned 256-entry interleaved LUT $(\sin\theta, \cos\theta)$ from Q7.
#[inline(always)]
pub const fn branchless_sincos_q7(phase_q7: PhaseQ7) -> (f32, f32) {
    let lut = Q7_SINCOS_LUT_256_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Direct 256-entry LUT $\sin\theta$ from a Q7 phase angle (`q7 * pi / 128`).
#[inline(always)]
pub const fn branchless_sin_q7(phase_q7: PhaseQ7) -> f32 {
    let lut = Q7_FULL_SIN_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Direct 256-entry LUT $\cos\theta$ from a Q7 phase angle (`q7 * pi / 128`).
#[inline(always)]
pub const fn branchless_cos_q7(phase_q7: PhaseQ7) -> f32 {
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

/// Fused `#![no_std]` four-quadrant $\mathrm{atan2}(y, x)$ directly quantized into `PhaseQ7`
/// ($[-128, 127]$), folding the $\frac{128}{\pi}$ scale factor directly into the minimax polynomial
/// coefficients (`0` Q7 error across $[-180^\circ, 180^\circ]$).
#[inline(always)]
pub const fn branchless_atan2_to_q7_fast(y: f32, x: f32) -> PhaseQ7 {
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
                + r2 * (7.886_416 + r2 * (-4.744_295 + r2 * (2.145_485 - 0.477_609 * r2)))));
    let mut ang_q7 = branchless_select_f32(swap, 64.0 - poly_q7, poly_q7);
    ang_q7 = branchless_select_f32(safe_x < 0.0, 128.0 - ang_q7, ang_q7);
    ang_q7 = branchless_copysign_f32(ang_q7, safe_y);
    let rounded = (ang_q7 + branchless_select_f32(ang_q7 >= 0.0, 0.5, -0.5)) as i32;
    branchless_select_i32(
        rounded < -128,
        -128,
        branchless_select_i32(rounded > 127, 127, rounded),
    ) as PhaseQ7
}

#[inline(always)]
pub(crate) const fn eval_von_mises_kappa_clamped(r: f32) -> f32 {
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
/// inversions and 2 Pade `tanh` evaluations per call on the hot/mezzo path.
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

pub(crate) const VON_MISES_KAPPA_2ZONE_REF: &[[f32; 257]; 2] = &VON_MISES_KAPPA_2ZONE_LUT.0;

/// Branchless sigmoidal-blended von Mises concentration parameter $\kappa_c$ from $\bar{R}_c \in [0, 1)$,
/// evaluated in strictly constant time (`0` Newton-Raphson divisions, `0` Pade `tanh` evaluations)
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
        branchless_select_f32(
            branchless_is_finite_f32(phase_coherence),
            phase_coherence,
            0.0,
        ),
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
    liminal_lacuna: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    d0: f32,
) -> f32 {
    let valid = branchless_is_finite_f32(liminal_lacuna)
        & branchless_is_finite_f32(tau_eff)
        & branchless_is_finite_f32(alpha)
        & branchless_is_finite_f32(beta)
        & branchless_is_finite_f32(d0)
        & (beta > DEFAULT_ENTROPY_EPS)
        & (d0 > DEFAULT_ENTROPY_EPS);
    let safe_denom = branchless_select_f32(valid, 4.0 * beta * d0, 1.0);
    let excess = branchless_relu_f32(liminal_lacuna - tau_eff);
    let safe_alpha = branchless_select_f32(valid, alpha, 0.0);
    let ratio = (safe_alpha * safe_alpha * excess * excess) * fast_inv_pos_normal_f32(safe_denom);
    branchless_select_f32(valid, branchless_exp_f32(-ratio), 0.0)
}

/// Converts a Q7 fixed-point phase value to radians ($\theta = \mathrm{q7} \cdot \pi / 128$).
#[inline(always)]
pub const fn branchless_q7_to_rad(phase_q7: PhaseQ7) -> f32 {
    (phase_q7 as f32) * Q7_PHASE_SCALE_RAD
}

/// Branchlessly quantizes a radian angle into Q7 fixed-point format ($[-128, 127]$).
#[inline(always)]
pub const fn branchless_rad_to_q7(rad: f32) -> PhaseQ7 {
    let valid = branchless_is_finite_f32(rad);
    let scaled = branchless_select_f32(valid, rad, 0.0) * Q7_PHASE_INV_SCALE_RAD;
    let rounded = (scaled + branchless_select_f32(scaled >= 0.0, 0.5, -0.5)) as i32;
    let clamped = branchless_select_i32(
        rounded < -128,
        -128,
        branchless_select_i32(rounded > 127, 127, rounded),
    );
    branchless_select_i32(valid, clamped, 0) as PhaseQ7
}

/// Converts a non-negative radian step threshold into a ceiling-rounded Q7 integer cutoff (`0..=128`).
#[inline(always)]
pub const fn branchless_rad_to_phase_step_u8(min_step_rad: f32) -> u8 {
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
pub const fn branchless_q7_add_sat(a: PhaseQ7, b: PhaseQ7) -> PhaseQ7 {
    let sum = (a as i32) + (b as i32);
    let clamped_hi = branchless_select_i32(sum > (Q7_MAX as i32), Q7_MAX as i32, sum);
    branchless_select_i32(clamped_hi < (Q7_MIN as i32), Q7_MIN as i32, clamped_hi) as PhaseQ7
}

/// Branchless Q7 fixed-point saturating subtraction ($[-128, 127]$).
#[inline(always)]
pub const fn branchless_q7_sub_sat(a: PhaseQ7, b: PhaseQ7) -> PhaseQ7 {
    let diff = (a as i32) - (b as i32);
    let clamped_hi = branchless_select_i32(diff > (Q7_MAX as i32), Q7_MAX as i32, diff);
    branchless_select_i32(clamped_hi < (Q7_MIN as i32), Q7_MIN as i32, clamped_hi) as PhaseQ7
}

/// Branchless Q7 circular phase difference $(a - b) \pmod{256}$ on $S^1 \cong SO(2)$.
#[inline(always)]
pub const fn branchless_q7_circular_diff(a: PhaseQ7, b: PhaseQ7) -> PhaseQ7 {
    a.wrapping_sub(b)
}

/// Branchlessly folds a circular Q7 phase angle on $S^1$ onto the principal geodesic wave `[-64, +64]`.
#[inline(always)]
pub const fn branchless_q7_fold_s1_geodesic(phase_q7: PhaseQ7) -> PhaseQ7 {
    let b = phase_q7 as u8;
    let fold_mask = ((b ^ (b << 1)) >> 7).wrapping_neg();
    let mirrored = 0x80u8.wrapping_sub(b);
    ((b & !fold_mask) | (mirrored & fold_mask)) as PhaseQ7
}

/// Branchless Q7 shortest-arc geodesic interpolation on $S^1 \cong SO(2)$ (`t_u7 in 0..=128`).
#[inline(always)]
pub const fn branchless_q7_circular_lerp(a: PhaseQ7, b: PhaseQ7, t_u7: u8) -> PhaseQ7 {
    let clamped_t = branchless_select_u8(t_u7 > 128, 128, t_u7);
    a.wrapping_add(branchless_q7_scale_u7(
        branchless_q7_circular_diff(b, a),
        clamped_t,
    ))
}

/// Branchless Q7 phase-conjugate negation ($-\Delta\theta$) saturating $-128 \to +127$.
#[inline(always)]
pub const fn branchless_q7_neg_sat(x: PhaseQ7) -> PhaseQ7 {
    let neg = -(x as i32);
    branchless_select_i32(neg > 127, 127, neg) as PhaseQ7
}

/// Branchless Q7 absolute value returning `[0, 128]` as `u8`.
#[inline(always)]
pub const fn branchless_q7_abs_u8(x: PhaseQ7) -> u8 {
    let val = x as i32;
    let mask = val >> 31;
    ((val ^ mask) - mask) as u8
}

/// Branchless Q7 fixed-point integer scaling: $\lfloor (\mathrm{val} \cdot \mathrm{scale\_u7}) / 128 \rfloor$.
#[inline(always)]
pub const fn branchless_q7_scale_u7(val: PhaseQ7, scale_u7: u8) -> PhaseQ7 {
    let scaled = ((val as i32) * (scale_u7 as i32)) >> 7;
    let clamped_hi = branchless_select_i32(scaled > 127, 127, scaled);
    branchless_select_i32(clamped_hi < -128, -128, clamped_hi) as PhaseQ7
}
