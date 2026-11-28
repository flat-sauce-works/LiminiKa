// name: src/core/src/foundation/simd_stats/procrustes.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Orthogonal Procrustes $SO(2)$, pairwise Q7 `atan2`, and vectorized 8-lane `sincos` kernels.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::f32::consts::{FRAC_PI_2, PI};

use crate::foundation::scalar_math::branchless_sincos_f32;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use crate::foundation::scalar_math::{branchless_atan2_to_q7_fast, branchless_is_finite_f32};
use crate::foundation::swar_simd::{store_q7_words_64, store_q7_x8_slice};
use crate::liminal::{DEFAULT_ENTROPY_EPS, MAX_ATTENTION_HEADS, PhaseQ7};

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
            .as_chunks::<2>()
            .0
            .iter()
            .zip(tgt_head[..len].as_chunks::<2>().0.iter())
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
            _mm_add_ps, _mm_movehl_ps, _mm_movelh_ps, _mm256_castps256_ps128, _mm256_extractf128_ps,
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
            _mm_addsub_ps, _mm_loadu_ps, _mm_shuffle_ps, _mm_storeu_ps, _mm256_castps128_ps256,
            _mm256_insertf128_ps,
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
            _mm256_insertf128_ps::<1>(
                _mm256_castps128_ps256(_mm_loadu_ps(sin_lo_buf.as_ptr())),
                sin_hi,
            ),
            _mm256_insertf128_ps::<1>(
                _mm256_castps128_ps256(_mm_loadu_ps(cos_lo_buf.as_ptr())),
                cos_hi,
            ),
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

        let sq_x = _mm256_mul_ps(vx, vx);
        let sq_y = _mm256_mul_ps(vy, vy);
        let swap_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(sq_x, sq_y);
        let norm_sq = _mm256_add_ps(sq_x, sq_y);
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
        ang_q7 = _mm256_or_ps(
            _mm256_and_ps(ang_q7, _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF))),
            y_sign,
        );
        let rounded_q7 = _mm256_add_ps(ang_q7, _mm256_or_ps(_mm256_set1_ps(0.5_f32), y_sign));
        let valid_mask = _mm256_and_ps(
            _mm256_cmp_ps::<_CMP_GT_OQ>(norm_sq, _mm256_set1_ps(DEFAULT_ENTROPY_EPS)),
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
            avx2_atan2_to_q7_word_x8_vec(vy, vx)
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
            let valid = (norm_sq > DEFAULT_ENTROPY_EPS) & branchless_is_finite_f32(norm_sq);
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

/// Evaluates 8-lane pairwise $\operatorname{atan2}(y_k, x_k)$ directly on 16 interleaved
/// `[x_0, y_0, x_1, y_1, ..., x_7, y_7]` floats (`interleaved_16.len() >= 16`), deinterleaving
/// even (`cos`) and odd (`sin`) lanes in AVX2 registers to eliminate 16 scalar stack stores and
/// 8-to-1 Store-to-Load Forwarding (STLF) pipeline stalls.
#[inline(always)]
pub(crate) fn atan2_interleaved_pairs_to_q7_word_x8_simd(interleaved_16: &[f32]) -> u64 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if interleaved_16.len() >= 16 {
            // SAFETY: `target_feature = "avx2"` is enabled and `interleaved_16.len() >= 16`.
            unsafe {
                let ptr = interleaved_16.as_ptr();
                let v_lo = _mm256_loadu_ps(ptr);
                let v_hi = _mm256_loadu_ps(ptr.add(8));
                let cos_shuf = _mm256_shuffle_ps::<0x88>(v_lo, v_hi);
                let sin_shuf = _mm256_shuffle_ps::<0xDD>(v_lo, v_hi);
                let vx = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(
                    _mm256_castps_si256(cos_shuf),
                ));
                let vy = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<0xD8>(
                    _mm256_castps_si256(sin_shuf),
                ));
                return avx2_atan2_to_q7_word_x8_vec(vy, vx);
            }
        }
    }
    let sin_arr = [
        interleaved_16[1],
        interleaved_16[3],
        interleaved_16[5],
        interleaved_16[7],
        interleaved_16[9],
        interleaved_16[11],
        interleaved_16[13],
        interleaved_16[15],
    ];
    let cos_arr = [
        interleaved_16[0],
        interleaved_16[2],
        interleaved_16[4],
        interleaved_16[6],
        interleaved_16[8],
        interleaved_16[10],
        interleaved_16[12],
        interleaved_16[14],
    ];
    atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr)
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
    phases_out: &mut [PhaseQ7],
) {
    let limit_heads = num_heads.min(MAX_ATTENTION_HEADS).min(phases_out.len());
    let words = procrustes_q7_words_64_simd(src_slice, tgt_slice, head_dim, limit_heads);
    if limit_heads == MAX_ATTENTION_HEADS {
        // SAFETY: `phases_out.len() >= MAX_ATTENTION_HEADS` and `align_of::<PhaseQ7>() == 1`.
        let out_arr = unsafe {
            &mut *phases_out
                .as_mut_ptr()
                .cast::<[PhaseQ7; MAX_ATTENTION_HEADS]>()
        };
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
            phases_out[h] = (w & 0xFF) as PhaseQ7;
            w >>= 8;
            h += 1;
        }
    }
}

/// Evaluates range-reduced $(\sin\theta, \cos\theta)$ across 8 `f32` angles simultaneously using AVX2.
#[inline(always)]
pub(crate) fn sincos_f32_x8_simd(angles: &[f32], sin_out: &mut [f32; 8], cos_out: &mut [f32; 8]) {
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
                // Cody-Waite double-split `2*pi = 6.283185 + 3.0717959e-7` requires the truncated
                // high part verbatim (`f32::consts::TAU` would degrade range-reduction accuracy).
                #[allow(clippy::approx_constant, clippy::excessive_precision)]
                let x = _mm256_sub_ps(
                    _mm256_sub_ps(v_in, _mm256_mul_ps(k, _mm256_set1_ps(6.283_185_f32))),
                    _mm256_mul_ps(k, _mm256_set1_ps(3.071_795_9e-7_f32)),
                );

                let gt_hp = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(FRAC_PI_2));
                let lt_nhp = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(-FRAC_PI_2));
                let x_fold_lo = _mm256_blendv_ps(x, _mm256_sub_ps(_mm256_set1_ps(-PI), x), lt_nhp);
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
                let sin_v = _mm256_mul_ps(x_folded, _mm256_add_ps(_mm256_mul_ps(x2, sp), one_v));

                // Exact minimax cosine polynomial coefficient `-1/10!` in f32 rounding.
                #[allow(clippy::excessive_precision)]
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
