// name: src/core/src/foundation/swar_simd.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bit-parallel 4/8-lane SWAR Q7/u8 arithmetic, 64-head word codecs, and slice/phase-clamp LUT dispatchers.

use core::f32::consts::FRAC_PI_2;

use crate::foundation::lut_crc32::{
    sincos_phase_clamp_poly, Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF,
    Q7_FUSED_SELF_DEFAULT_PHASE_CLAMP_SINCOS_REF, Q7_FUSED_SELF_GEODESIC_PHASE_CLAMP_SINCOS_REF,
    SWAR_EVEN_BYTES_64, SWAR_LOW7_MASK_32, SWAR_LOW7_MASK_64, SWAR_ODD_BYTES_64, SWAR_SIGN_MASK_32,
    SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, SWAR_SUM_U16_64,
};
use crate::foundation::scalar_math::{
    branchless_abs_f32, branchless_is_finite_f32, branchless_is_nan_f32,
    branchless_q7_fold_s1_geodesic, branchless_relu_f32,
    branchless_sincos_q7_geodesic_phase_clamp_default_lut,
    branchless_sincos_q7_phase_clamp_default_lut, branchless_sincos_q7_phase_clamp_preinv,
    eval_phase_clamp_theta_rad_preinv, sincos_reduced_pi,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_i32, branchless_select_i8, branchless_select_u32,
    branchless_select_u64, branchless_select_u8, branchless_select_usize, PhaseQ7,
    MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
};

/// Packs 8 `PhaseQ7` lanes into a single 64-bit little-endian SWAR word.
#[inline(always)]
pub const fn swar_q7_pack_x8(lanes: &[PhaseQ7; 8]) -> u64 {
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

/// Unpacks a 64-bit little-endian SWAR word into 8 `PhaseQ7` lanes.
#[inline(always)]
pub const fn swar_q7_unpack_x8(packed: u64) -> [PhaseQ7; 8] {
    let b = packed.to_le_bytes();
    [
        b[0] as PhaseQ7,
        b[1] as PhaseQ7,
        b[2] as PhaseQ7,
        b[3] as PhaseQ7,
        b[4] as PhaseQ7,
        b[5] as PhaseQ7,
        b[6] as PhaseQ7,
        b[7] as PhaseQ7,
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

/// Saturating 64-head Q7 addition (`dst += src`) across 8 packed `u64` words.
///
/// Uses two AVX2 `_mm256_adds_epi8` instructions on `x86_64` + AVX2 targets and falls back
/// to 8-word bit-parallel SWAR [`swar_q7_add_sat_x8`] elsewhere.
#[inline(always)]
pub(crate) fn swar_or_avx2_q7_add_sat_words_64(dst: &mut [u64; 8], src: &[u64; 8]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::{
            __m256i, _mm256_adds_epi8, _mm256_loadu_si256, _mm256_storeu_si256,
        };
        // SAFETY: `dst` and `src` are valid 64-byte (`8 x u64`) arrays; unaligned 256-bit loads/stores
        // are valid for any 8-byte aligned `[u64; 8]` and guarded by `target_feature = "avx2"`.
        unsafe {
            let dp = dst.as_mut_ptr().cast::<__m256i>();
            let sp = src.as_ptr().cast::<__m256i>();
            let d0 = _mm256_loadu_si256(dp);
            let s0 = _mm256_loadu_si256(sp);
            let d1 = _mm256_loadu_si256(dp.add(1));
            let s1 = _mm256_loadu_si256(sp.add(1));
            _mm256_storeu_si256(dp, _mm256_adds_epi8(d0, s0));
            _mm256_storeu_si256(dp.add(1), _mm256_adds_epi8(d1, s1));
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        dst[0] = swar_q7_add_sat_x8(dst[0], src[0]);
        dst[1] = swar_q7_add_sat_x8(dst[1], src[1]);
        dst[2] = swar_q7_add_sat_x8(dst[2], src[2]);
        dst[3] = swar_q7_add_sat_x8(dst[3], src[3]);
        dst[4] = swar_q7_add_sat_x8(dst[4], src[4]);
        dst[5] = swar_q7_add_sat_x8(dst[5], src[5]);
        dst[6] = swar_q7_add_sat_x8(dst[6], src[6]);
        dst[7] = swar_q7_add_sat_x8(dst[7], src[7]);
    }
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
pub(crate) const fn swar_q7_phase_step_filter_with_thresh_x8(
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

/// True bit-parallel SWAR 8-lane discrete phase-step cutoff filter (`PhaseStepFilter`) in strictly constant time (`0` branches).
#[inline(always)]
pub const fn swar_q7_phase_step_filter_x8(packed: u64, min_step_q7: u8) -> u64 {
    let thresh_x8 = ((min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let valid_step_mask = ((min_step_q7 <= 127) as u64).wrapping_neg();
    swar_q7_phase_step_filter_with_thresh_x8(packed, thresh_x8, valid_step_mask)
}

/// Applies the 8-lane SWAR discrete phase-step cutoff filter (`PhaseStepFilter`) across 8 register-resident
/// `u64` SWAR words using a pre-evaluated word-uniformity flag (`w_is_uniform`), returning
/// `(filtered_words, filtered_is_uniform)`.
#[inline(always)]
#[must_use]
pub const fn swar_q7_phase_step_filter_words_with_uniform_flag_64(
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
        if w0 == w0.rotate_right(8) {
            let abs_b0 = (b0 as i8).unsigned_abs();
            return if abs_b0 >= min_step_q7 {
                (*w, true)
            } else {
                ([0u64; 8], true)
            };
        }
        return (
            [swar_q7_phase_step_filter_with_thresh_x8(w0, thresh_x8, u64::MAX); 8],
            true,
        );
    }
    let out = [
        swar_q7_phase_step_filter_with_thresh_x8(w[0], thresh_x8, u64::MAX),
        swar_q7_phase_step_filter_with_thresh_x8(w[1], thresh_x8, u64::MAX),
        swar_q7_phase_step_filter_with_thresh_x8(w[2], thresh_x8, u64::MAX),
        swar_q7_phase_step_filter_with_thresh_x8(w[3], thresh_x8, u64::MAX),
        swar_q7_phase_step_filter_with_thresh_x8(w[4], thresh_x8, u64::MAX),
        swar_q7_phase_step_filter_with_thresh_x8(w[5], thresh_x8, u64::MAX),
        swar_q7_phase_step_filter_with_thresh_x8(w[6], thresh_x8, u64::MAX),
        swar_q7_phase_step_filter_with_thresh_x8(w[7], thresh_x8, u64::MAX),
    ];
    (out, is_word_array_uniform_64(&out))
}

/// Applies the 8-lane SWAR discrete phase-step cutoff filter (`PhaseStepFilter`) across all 8 register-resident
/// `u64` SWAR words (`64` attention heads) with hoisted `thresh_x8` and `valid_step_mask` in
/// strictly constant time (`0` data-dependent branches).
#[inline(always)]
#[must_use]
pub const fn swar_q7_phase_step_filter_words_64(w: &[u64; 8], min_step_q7: u8) -> [u64; 8] {
    if min_step_q7 <= 1 {
        return *w;
    }
    if min_step_q7 > 127 {
        return [0u64; 8];
    }
    swar_q7_phase_step_filter_words_with_uniform_flag_64(
        w,
        is_word_array_uniform_64(w),
        min_step_q7,
    )
    .0
}

#[inline(always)]
pub(crate) const fn load_q7_word_at(arr: &[PhaseQ7; MAX_ATTENTION_HEADS], base: usize) -> u64 {
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
    arr: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
    base: usize,
    word: u64,
) {
    let b = word.to_le_bytes();
    arr[base] = b[0] as PhaseQ7;
    arr[base + 1] = b[1] as PhaseQ7;
    arr[base + 2] = b[2] as PhaseQ7;
    arr[base + 3] = b[3] as PhaseQ7;
    arr[base + 4] = b[4] as PhaseQ7;
    arr[base + 5] = b[5] as PhaseQ7;
    arr[base + 6] = b[6] as PhaseQ7;
    arr[base + 7] = b[7] as PhaseQ7;
}

/// Loads all 64 Q7 head phases into 8 `u64` SWAR registers using 8 direct 64-bit GPR word loads
/// (`mov r64, qword ptr [arr + 8*c]`), avoiding 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF)
/// stalls and 64-byte cache-line split penalties when reading `trail.accumulated_phase_delta` (offset 56).
#[inline(always)]
pub const fn load_q7_words_64(arr: &[PhaseQ7; MAX_ATTENTION_HEADS]) -> [u64; 8] {
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

/// Stores 8 `u64` SWAR registers into a 64-head Q7 phase array using 8 direct 64-bit GPR word stores
/// (`mov qword ptr [arr + 8*c], r64`), eliminating narrow-GPR-store-to-wide-YMM-load Store-to-Load
/// Forwarding (STLF) stalls (`~15-20 cycles`) and 64-byte cache-line split stores on `trail.accumulated_phase_delta`.
#[inline(always)]
pub const fn store_q7_words_64(arr: &mut [PhaseQ7; MAX_ATTENTION_HEADS], w: &[u64; 8]) {
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
    (((w0 ^ w[2]) | (w0 ^ w[3])) | ((w0 ^ w[4]) | (w0 ^ w[5])) | ((w0 ^ w[6]) | (w0 ^ w[7]))) == 0
}

#[inline(always)]
pub(crate) const fn swar_u8_pair_sum16(packed_u8: u64) -> u64 {
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
) -> PhaseQ7 {
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
        ) as PhaseQ7
    }
}

/// Computes the rounded arithmetic mean Q7 phase across 8 packed `u64` SWAR words (64 heads)
/// using a pre-evaluated word-uniformity flag (`words_is_uniform`).
#[inline(always)]
pub const fn swar_q7_mean_i8_from_words_with_uniform_flag_64(
    words: &[u64; 8],
    words_is_uniform: bool,
) -> PhaseQ7 {
    if words_is_uniform {
        let w0 = words[0];
        if w0 == w0.rotate_right(8) {
            return (w0 & 0xFF) as u8 as PhaseQ7;
        }
        let signed_sum8 = (swar_u8_sum_x8(w0 ^ SWAR_SIGN_MASK_64) as i32) - (8 * 128);
        let rounded8 = (signed_sum8 + branchless_select_i32(signed_sum8 >= 0, 4, -4)) / 8;
        return branchless_select_i32(
            rounded8 < -128,
            -128,
            branchless_select_i32(rounded8 > 127, 127, rounded8),
        ) as PhaseQ7;
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
    ) as PhaseQ7
}

/// Computes the rounded arithmetic mean Q7 phase across all 8 packed `u64` SWAR words (64 heads)
/// in strictly constant time (`0` data-dependent branches).
#[inline(always)]
pub const fn swar_q7_mean_i8_from_words_64(words: &[u64; 8]) -> PhaseQ7 {
    swar_q7_mean_i8_from_words_with_uniform_flag_64(words, is_word_array_uniform_64(words))
}

/// Returns `true` if any of the 64 Q7 head phases in `phases` is non-zero (`0` branches).
#[inline(always)]
pub const fn is_phase_profile_nonzero_64(phases: &[PhaseQ7; MAX_ATTENTION_HEADS]) -> bool {
    let w = load_q7_words_64(phases);
    (((w[0] | w[1]) | (w[2] | w[3])) | ((w[4] | w[5]) | (w[6] | w[7]))) != 0
}

/// Accumulates `deltas` into `acc` across all 64 attention heads using 8-word SWAR saturating addition.
#[inline(always)]
pub const fn branchless_q7_accumulate_64(
    acc: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
    deltas: &[PhaseQ7; MAX_ATTENTION_HEADS],
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
    a: &[PhaseQ7; MAX_ATTENTION_HEADS],
    b: &[PhaseQ7; MAX_ATTENTION_HEADS],
) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
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
    phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
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
pub const fn branchless_q7_scale_64(phases: &mut [PhaseQ7; MAX_ATTENTION_HEADS], scale_u7: u8) {
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

/// Applies the 8-lane SWAR discrete phase-step cutoff filter (`PhaseStepFilter`) across all 64 attention heads
/// in strictly constant time (`0` data-dependent branches) via [`swar_q7_phase_step_filter_words_64`].
#[inline(always)]
pub const fn branchless_q7_phase_step_filter_64(
    phases: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
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
    let out_w = swar_q7_phase_step_filter_words_64(&w, min_step_q7);
    store_q7_words_64(phases, &out_w);
}

/// Branchless discrete phase-step filter (`PhaseStepFilter`) in the Q7 integer domain (Section 2.3).
#[inline(always)]
pub const fn branchless_phase_step_filter_q7(delta: PhaseQ7, min_step_q7: i32) -> PhaseQ7 {
    let val = delta as i32;
    let sign_mask = val >> 31;
    let abs_val = (val ^ sign_mask) - sign_mask;
    let keep_mask = (((abs_val >= min_step_q7) & (min_step_q7 <= 127)) as i32).wrapping_neg();
    (val & keep_mask) as PhaseQ7
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
pub(crate) fn load_q7_x8_slice(slice: &[PhaseQ7], offset: usize) -> u64 {
    debug_assert!(offset + 8 <= slice.len());
    // SAFETY: Caller guarantees `offset + 8 <= slice.len()`.
    let raw = unsafe { core::ptr::read_unaligned(slice.as_ptr().add(offset).cast::<u64>()) };
    u64::from_le(raw)
}

#[inline(always)]
pub(crate) fn store_q7_x8_slice(slice: &mut [PhaseQ7], offset: usize, word: u64) {
    debug_assert!(offset + 8 <= slice.len());
    // SAFETY: Caller guarantees `offset + 8 <= slice.len()`.
    unsafe {
        core::ptr::write_unaligned(slice.as_mut_ptr().add(offset).cast::<u64>(), word.to_le());
    }
}

#[inline(always)]
pub(crate) fn swar_phase_step_filter_q7_slice_in_place(
    phases: &mut [PhaseQ7],
    limit: usize,
    min_step_q7: i32,
) {
    let clamped_step = branchless_select_i32(
        min_step_q7 > 128,
        128,
        branchless_select_i32(min_step_q7 < 0, 0, min_step_q7),
    ) as u8;
    // Since `|x| >= 1` for every non-zero `PhaseQ7` lane (`|0| = 0`), `clamped_step <= 1` is an exact identity.
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
                __m256i, _mm256_abs_epi8, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256,
                _mm256_max_epu8, _mm256_set1_epi8, _mm256_storeu_si256,
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
        // SAFETY: `end == MAX_ATTENTION_HEADS <= phases.len()` and `align_of::<PhaseQ7>() == 1`.
        let arr = unsafe { &mut *phases.as_mut_ptr().cast::<[PhaseQ7; MAX_ATTENTION_HEADS]>() };
        let w = load_q7_words_64(arr);
        store_q7_words_64(
            arr,
            &[
                swar_q7_phase_step_filter_with_thresh_x8(w[0], thresh_x8, u64::MAX),
                swar_q7_phase_step_filter_with_thresh_x8(w[1], thresh_x8, u64::MAX),
                swar_q7_phase_step_filter_with_thresh_x8(w[2], thresh_x8, u64::MAX),
                swar_q7_phase_step_filter_with_thresh_x8(w[3], thresh_x8, u64::MAX),
                swar_q7_phase_step_filter_with_thresh_x8(w[4], thresh_x8, u64::MAX),
                swar_q7_phase_step_filter_with_thresh_x8(w[5], thresh_x8, u64::MAX),
                swar_q7_phase_step_filter_with_thresh_x8(w[6], thresh_x8, u64::MAX),
                swar_q7_phase_step_filter_with_thresh_x8(w[7], thresh_x8, u64::MAX),
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
            swar_q7_phase_step_filter_with_thresh_x8(w0, thresh_x8, u64::MAX),
        );
        store_q7_x8_slice(
            phases,
            i + 8,
            swar_q7_phase_step_filter_with_thresh_x8(w1, thresh_x8, u64::MAX),
        );
        i += 16;
    }
    while i + 8 <= end {
        let w = load_q7_x8_slice(phases, i);
        store_q7_x8_slice(
            phases,
            i,
            swar_q7_phase_step_filter_with_thresh_x8(w, thresh_x8, u64::MAX),
        );
        i += 8;
    }
    while i < end {
        phases[i] = branchless_phase_step_filter_q7(phases[i], i32::from(clamped_step));
        i += 1;
    }
}

#[inline(always)]
pub(crate) const fn sanitize_inplace_gain_f32(raw_gain: f32) -> f32 {
    let valid_gain = branchless_is_finite_f32(raw_gain);
    let clean_gain = branchless_select_f32(valid_gain, raw_gain, 1.0);
    branchless_select_f32(
        branchless_abs_f32(clean_gain - 1.0) > 1.0e-7,
        clean_gain,
        1.0,
    )
}

#[inline(always)]
pub(crate) const fn eval_phase_clamp_sincos_q7_fast(
    phase_q7: PhaseQ7,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if is_default {
        branchless_sincos_q7_phase_clamp_default_lut(phase_q7)
    } else {
        branchless_sincos_q7_phase_clamp_preinv(phase_q7, eff_max, q7_to_norm_scale)
    }
}

#[inline(always)]
pub(crate) const fn eval_geodesic_phase_clamp_sincos_q7_fast(
    phase_q7: PhaseQ7,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if is_default {
        branchless_sincos_q7_geodesic_phase_clamp_default_lut(phase_q7)
    } else {
        let folded = branchless_q7_fold_s1_geodesic(phase_q7);
        branchless_sincos_q7_phase_clamp_preinv(folded, eff_max, q7_to_norm_scale)
    }
}

/// Evaluates the $SO(2)$-composed $(\sin(\theta_{\mathrm{base}} + \theta_{\mathrm{subhead}}), \cos(\theta_{\mathrm{base}} + \theta_{\mathrm{subhead}}))$
/// for a single attention head, using two 1-cycle LUT loads when both `PhaseClamp` limits equal
/// [`crate::foundation::scalar_math::DEFAULT_PHASE_CLAMP_MAX_RAD`], or summing the two `tanh`-clamped radian angles and evaluating
/// `sincos_phase_clamp_poly` **once** instead of twice when custom `PhaseClamp` limits are active.
#[inline(always)]
pub(crate) const fn eval_fused_phase_clamp_subhead_sincos_q7_fast(
    base_q7: PhaseQ7,
    geodesic_base: bool,
    is_default_base: bool,
    eff_base_max: f32,
    base_norm_scale: f32,
    subhead_q7: PhaseQ7,
    geodesic_subhead: bool,
    is_default_subhead: bool,
    eff_subhead_max: f32,
    subhead_norm_scale: f32,
) -> (f32, f32) {
    let b_q7 = if geodesic_base {
        branchless_q7_fold_s1_geodesic(base_q7)
    } else {
        base_q7
    };
    let p_q7 = if geodesic_subhead {
        branchless_q7_fold_s1_geodesic(subhead_q7)
    } else {
        subhead_q7
    };
    if is_default_base & is_default_subhead {
        if b_q7 == p_q7 {
            return Q7_FUSED_SELF_DEFAULT_PHASE_CLAMP_SINCOS_REF[(b_q7 as u8) as usize];
        }
        if (base_q7 == subhead_q7) & (!geodesic_base) & geodesic_subhead {
            return Q7_FUSED_SELF_GEODESIC_PHASE_CLAMP_SINCOS_REF[(base_q7 as u8) as usize];
        }
        let lut = Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF;
        let (s_b, c_b) = lut[(b_q7 as u8) as usize];
        let (s_p, c_p) = lut[(p_q7 as u8) as usize];
        return (s_b * c_p + c_b * s_p, c_b * c_p - s_b * s_p);
    }
    if (b_q7 == p_q7) & (eff_base_max.to_bits() == eff_subhead_max.to_bits()) {
        if b_q7 == 0 {
            return (0.0, 1.0);
        }
        let theta_comb =
            eval_phase_clamp_theta_rad_preinv(b_q7, eff_base_max + eff_base_max, base_norm_scale);
        let abs_comb = branchless_abs_f32(theta_comb);
        return if abs_comb <= FRAC_PI_2 {
            sincos_phase_clamp_poly(theta_comb)
        } else {
            sincos_reduced_pi(theta_comb)
        };
    }
    let theta_b = if b_q7 == 0 {
        0.0
    } else {
        eval_phase_clamp_theta_rad_preinv(b_q7, eff_base_max, base_norm_scale)
    };
    let theta_p = if p_q7 == 0 {
        0.0
    } else {
        eval_phase_clamp_theta_rad_preinv(p_q7, eff_subhead_max, subhead_norm_scale)
    };
    let theta_comb = theta_b + theta_p;
    let abs_comb = branchless_abs_f32(theta_comb);
    if abs_comb <= FRAC_PI_2 {
        sincos_phase_clamp_poly(theta_comb)
    } else {
        sincos_reduced_pi(theta_comb)
    }
}

#[must_use]
pub(crate) const fn words_to_q7_array_64(
    words: &[u64; NUM_STIGMERGIC_GRANULES],
) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
    // SAFETY: `[u64; 8]` and `[i8; 64]` have identical size (64 bytes) and Pod bit validity.
    unsafe {
        core::mem::transmute::<[u64; NUM_STIGMERGIC_GRANULES], [PhaseQ7; MAX_ATTENTION_HEADS]>(
            *words,
        )
    }
}

/// Selects the smaller cardinality side of an 8-bit partition (`inside` vs `outside`) branchlessly.
///
/// Given a `universe_mask` and a `cut_mask`, this evaluates `inside = universe_mask & cut_mask`
/// and `outside = universe_mask & !cut_mask` and returns `(smaller_mask, smaller_count, inside_is_smaller_or_tie)`.
/// When both sides have equal population count, `inside` is selected (`inside_is_smaller_or_tie = true`).
///
/// Complement polarity here is strictly minimal-representation / cardinality-selection arithmetic
/// and carries no authority or semantic judgment over either side of the partition.
#[inline(always)]
#[must_use]
pub const fn branchless_mask8_min_delta(universe_mask: u8, cut_mask: u8) -> (u8, u8, bool) {
    let inside = universe_mask & cut_mask;
    let outside = universe_mask & !cut_mask;
    let in_cnt = inside.count_ones() as u8;
    let out_cnt = outside.count_ones() as u8;
    let inside_is_smaller_or_tie = in_cnt <= out_cnt;
    let smaller_mask = branchless_select_u8(inside_is_smaller_or_tie, inside, outside);
    let smaller_count = branchless_select_u8(inside_is_smaller_or_tie, in_cnt, out_cnt);
    (smaller_mask, smaller_count, inside_is_smaller_or_tie)
}

/// Finds the lowest-index minimum value among masked `f32` lanes in constant time (`0` data-dependent branches).
///
/// Evaluates all 8 lanes in a fixed `const fn` loop. A lane `i` is selectable iff bit `i` of
/// `eligible_mask` is set and `values[i]` is not `NaN`. Finite values and `+infinity` are all
/// selectable (lowest index wins ties). Returns `(0xFF, f32::INFINITY)` only when `eligible_mask == 0`
/// or every eligible lane is `NaN`.
#[inline(always)]
#[must_use]
pub const fn branchless_argmin_f32_x8_masked(values: &[f32; 8], eligible_mask: u8) -> (u8, f32) {
    let mut best_lane: u8 = 0xFF;
    let mut best_val: f32 = f32::INFINITY;
    let mut has_valid: bool = false;
    let mut lane: u8 = 0;
    while lane < 8 {
        let v = values[lane as usize];
        let selectable = (((eligible_mask >> lane) & 1) != 0) & (!branchless_is_nan_f32(v));
        let take = selectable & ((!has_valid) | (v < best_val));
        best_lane = branchless_select_u8(take, lane, best_lane);
        best_val = branchless_select_f32(take, v, best_val);
        has_valid |= selectable;
        lane += 1;
    }
    (best_lane, best_val)
}

/// Finds the lowest-index minimum value among masked `u8` lanes in constant time (`0` data-dependent branches).
///
/// Evaluates all 8 lanes in a fixed `const fn` loop. A lane `i` is selectable iff bit `i` of
/// `eligible_mask` is set (lowest index wins ties). Returns `(0xFF, u8::MAX)` when
/// `eligible_mask == 0`.
#[inline(always)]
#[must_use]
pub const fn branchless_argmin_u8_x8_masked(values: &[u8; 8], eligible_mask: u8) -> (u8, u8) {
    let mut best_lane: u8 = 0xFF;
    let mut best_val: u8 = u8::MAX;
    let mut has_valid: bool = false;
    let mut lane: u8 = 0;
    while lane < 8 {
        let v = values[lane as usize];
        let selectable = ((eligible_mask >> lane) & 1) != 0;
        let take = selectable & ((!has_valid) | (v < best_val));
        best_lane = branchless_select_u8(take, lane, best_lane);
        best_val = branchless_select_u8(take, v, best_val);
        has_valid |= selectable;
        lane += 1;
    }
    (best_lane, best_val)
}

/// Integer Q0.7 reciprocal lookup table (`round(128 / k)`) for `0..=8` active parallel lanes.
pub const INV_PARALLEL_COUNT_Q7: [i32; 9] = [0, 128, 64, 43, 32, 26, 21, 18, 16];

#[inline(always)]
fn branchless_q7_weighted_ensemble_core<'a>(
    get_profile: impl Fn(usize) -> &'a [PhaseQ7; MAX_ATTENTION_HEADS],
    weights_q7: &[u8; 8],
    eligible_mask: u8,
    base_words: &[u64; NUM_STIGMERGIC_GRANULES],
) -> ([u64; NUM_STIGMERGIC_GRANULES], u8, u8) {
    if eligible_mask == 0 {
        return (*base_words, 0, 0);
    }

    let mut accum_norm_sum = 0u32;
    let mut head_sum = [0i32; MAX_ATTENTION_HEADS];
    let mut lane = 0usize;
    while lane < 8 {
        if ((eligible_mask >> lane) & 1) != 0 {
            let w_u32 = u32::from(weights_q7[lane]);
            let w_i32 = w_u32 as i32;
            for (acc, &d_q7) in head_sum.iter_mut().zip(get_profile(lane).iter()) {
                *acc += (i32::from(d_q7) * w_i32) >> 7;
            }
            accum_norm_sum = accum_norm_sum.saturating_add(w_u32);
        }
        lane += 1;
    }

    let inv = INV_PARALLEL_COUNT_Q7[(eligible_mask.count_ones() as usize).min(8)];
    let mut ensemble_q7 = [0 as PhaseQ7; MAX_ATTENTION_HEADS];
    for (out, &sum) in ensemble_q7.iter_mut().zip(head_sum.iter()) {
        let prod = sum * inv;
        let bias = if prod >= 0 { 64 } else { -64 };
        *out = ((prod + bias) / 128).clamp(-127, 127) as PhaseQ7;
    }

    let ensemble_words = load_q7_words_64(&ensemble_q7);
    let mut out_words = *base_words;
    for (acc_w, &ew) in out_words
        .iter_mut()
        .zip(ensemble_words.iter())
        .take(NUM_STIGMERGIC_GRANULES)
    {
        *acc_w = swar_q7_circular_add_x8(*acc_w, ew);
    }

    let avg_delta = (((accum_norm_sum as i32) * inv + 64) / 128).clamp(0, 127) as u8;
    let entry_idx = eligible_mask.trailing_zeros() as u8;
    (out_words, avg_delta, entry_idx)
}

/// Synthesizes a masked 8-lane weighted parallel ensemble over 64-head Q7 profiles and adds it
/// circularly onto `base_words` (`0` allocations, `0` floating-point divisions).
///
/// For each lane `i` whose bit is set in `eligible_mask`, accumulates `(profiles[i][h] * weights_q7[i]) >> 7`
/// across all 64 heads, divides by the active lane count via [`INV_PARALLEL_COUNT_Q7`] with symmetric
/// half-up rounding, clamps to `[-127, 127]`, and adds onto `base_words` with circular Q7 wraparound.
/// Returns `(out_words, avg_delta_norm_q7, lowest_active_lane_idx)`.
#[inline]
#[must_use]
pub fn branchless_q7_weighted_ensemble_words_64(
    profiles: [&[PhaseQ7; MAX_ATTENTION_HEADS]; 8],
    weights_q7: &[u8; 8],
    eligible_mask: u8,
    base_words: &[u64; NUM_STIGMERGIC_GRANULES],
) -> ([u64; NUM_STIGMERGIC_GRANULES], u8, u8) {
    branchless_q7_weighted_ensemble_core(
        move |i| profiles[i],
        weights_q7,
        eligible_mask,
        base_words,
    )
}

/// Zero-copy contiguous-matrix counterpart to [`branchless_q7_weighted_ensemble_words_64`],
/// reading directly from `&[[PhaseQ7; MAX_ATTENTION_HEADS]; 8]` without constructing a stack
/// pointer array.
#[inline]
#[must_use]
pub fn branchless_q7_weighted_ensemble_contiguous_words_64(
    profiles: &[[PhaseQ7; MAX_ATTENTION_HEADS]; 8],
    weights_q7: &[u8; 8],
    eligible_mask: u8,
    base_words: &[u64; NUM_STIGMERGIC_GRANULES],
) -> ([u64; NUM_STIGMERGIC_GRANULES], u8, u8) {
    branchless_q7_weighted_ensemble_core(|i| &profiles[i], weights_q7, eligible_mask, base_words)
}

#[inline(always)]
#[must_use]
pub(crate) const fn fast_mean_q7_tag_from_words_64(
    words: &[u64; NUM_STIGMERGIC_GRANULES],
) -> PhaseQ7 {
    let raw = swar_q7_mean_i8_from_words_64(words);
    branchless_select_i8(raw == -128, -127, raw)
}
