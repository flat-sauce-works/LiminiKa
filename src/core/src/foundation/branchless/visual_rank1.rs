// name: src/core/src/foundation/branchless/visual_rank1.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2D/3D visual RoPE rotations, Rank-1 outer-product kernels, and shared P-Frame Zero-Forward Seek kernels.

use super::*;

// ===================================================================
// 2. AVX2 / Scalar SIMD Kernels for 2D/3D Visual RoPE & Rank-1 Rank-1 residual
// ===================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_build_so2_2d_xy_vecs(
    sin_x: f32,
    cos_x: f32,
    sin_y: f32,
    cos_y: f32,
) -> (core::arch::x86_64::__m256, core::arch::x86_64::__m256) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`. Builds `[cos_x, cos_x, cos_y, cos_y]^2`
    // and `[sin_x, sin_x, sin_y, sin_y]^2` in 3 shuffle/insert instructions without stack spills.
    unsafe {
        use core::arch::x86_64::{
            _mm256_castps128_ps256, _mm256_insertf128_ps, _mm_set_ss, _mm_unpacklo_ps,
        };
        let c_xy = _mm_unpacklo_ps(_mm_set_ss(cos_x), _mm_set_ss(cos_y));
        let c_128 = _mm_unpacklo_ps(c_xy, c_xy);
        let s_xy = _mm_unpacklo_ps(_mm_set_ss(sin_x), _mm_set_ss(sin_y));
        let s_128 = _mm_unpacklo_ps(s_xy, s_xy);
        (
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(c_128), c_128),
            _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(s_128), s_128),
        )
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_2d_xy_slice_with_vecs(
    buf: &mut [f32],
    v_cos: core::arch::x86_64::__m256,
    v_sin: core::arch::x86_64::__m256,
) {
    let len = buf.len();
    let ptr = buf.as_mut_ptr();
    // SAFETY: Caller guarantees `target_feature = "avx2"`, and all offsets are bounds-checked against `len`.
    // Delegates 32-float blocks to `avx2_rotate_so2_block32` (`2x` paired-YMM ILP across opposite 32B L1D banks).
    unsafe {
        let mut b = 0usize;
        while b + 32 <= len {
            avx2_rotate_so2_block32(ptr.add(b), v_cos, v_sin);
            b += 32;
        }
        let rem = len - b;
        if (rem & 16) != 0 {
            let p0 = ptr.add(b);
            let p1 = ptr.add(b + 8);
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
            b += 16;
        }
        if (rem & 8) != 0 {
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
        if (rem & 4) != 0 {
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
        // SAFETY: `target_feature = "avx2"` is enabled.
        unsafe {
            let (v_cos, v_sin) = avx2_build_so2_2d_xy_vecs(sin_x, cos_x, sin_y, cos_y);
            avx2_rotate_so2_2d_xy_slice_with_vecs(buf, v_cos, v_sin);
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
#[inline]
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

/// Evaluates Rank-1 residual Rank-1 outer product write `y_slice[i] = scale * u_slice[i] * g_i` using AVX2.
#[inline(always)]
pub(crate) fn rank1_write_outer_product_simd(
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
                    let p0 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i)), _mm256_loadu_ps(pg.add(i)));
                    let p1 = _mm256_mul_ps(
                        _mm256_loadu_ps(pu.add(i + 8)),
                        _mm256_loadu_ps(pg.add(i + 8)),
                    );
                    _mm256_storeu_ps(py.add(i), _mm256_mul_ps(p0, vs));
                    _mm256_storeu_ps(py.add(i + 8), _mm256_mul_ps(p1, vs));

                    let p2 = _mm256_mul_ps(
                        _mm256_loadu_ps(pu.add(i + 16)),
                        _mm256_loadu_ps(pg.add(i + 16)),
                    );
                    let p3 = _mm256_mul_ps(
                        _mm256_loadu_ps(pu.add(i + 24)),
                        _mm256_loadu_ps(pg.add(i + 24)),
                    );
                    _mm256_storeu_ps(py.add(i + 16), _mm256_mul_ps(p2, vs));
                    _mm256_storeu_ps(py.add(i + 24), _mm256_mul_ps(p3, vs));
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
                    let u0 = _mm256_loadu_ps(pu.add(i));
                    let u1 = _mm256_loadu_ps(pu.add(i + 8));
                    _mm256_storeu_ps(py.add(i), _mm256_mul_ps(u0, vs));
                    _mm256_storeu_ps(py.add(i + 8), _mm256_mul_ps(u1, vs));
                    let u2 = _mm256_loadu_ps(pu.add(i + 16));
                    let u3 = _mm256_loadu_ps(pu.add(i + 24));
                    _mm256_storeu_ps(py.add(i + 16), _mm256_mul_ps(u2, vs));
                    _mm256_storeu_ps(py.add(i + 24), _mm256_mul_ps(u3, vs));
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

/// Accumulates Rank-1 residual Rank-1 outer product `y_slice[i] += scale * u_slice[i] * g_i` using AVX2.
#[inline]
pub(crate) fn rank1_accum_outer_product_simd(
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
                    let d0 = _mm256_mul_ps(
                        _mm256_mul_ps(_mm256_loadu_ps(pu.add(i)), _mm256_loadu_ps(pg.add(i))),
                        vs,
                    );
                    let d1 = _mm256_mul_ps(
                        _mm256_mul_ps(
                            _mm256_loadu_ps(pu.add(i + 8)),
                            _mm256_loadu_ps(pg.add(i + 8)),
                        ),
                        vs,
                    );
                    _mm256_storeu_ps(py.add(i), _mm256_add_ps(_mm256_loadu_ps(py.add(i)), d0));
                    _mm256_storeu_ps(
                        py.add(i + 8),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 8)), d1),
                    );

                    let d2 = _mm256_mul_ps(
                        _mm256_mul_ps(
                            _mm256_loadu_ps(pu.add(i + 16)),
                            _mm256_loadu_ps(pg.add(i + 16)),
                        ),
                        vs,
                    );
                    let d3 = _mm256_mul_ps(
                        _mm256_mul_ps(
                            _mm256_loadu_ps(pu.add(i + 24)),
                            _mm256_loadu_ps(pg.add(i + 24)),
                        ),
                        vs,
                    );
                    _mm256_storeu_ps(
                        py.add(i + 16),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 16)), d2),
                    );
                    _mm256_storeu_ps(
                        py.add(i + 24),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 24)), d3),
                    );
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
                    let d0 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i)), vs);
                    let d1 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i + 8)), vs);
                    _mm256_storeu_ps(py.add(i), _mm256_add_ps(_mm256_loadu_ps(py.add(i)), d0));
                    _mm256_storeu_ps(
                        py.add(i + 8),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 8)), d1),
                    );
                    let d2 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i + 16)), vs);
                    let d3 = _mm256_mul_ps(_mm256_loadu_ps(pu.add(i + 24)), vs);
                    _mm256_storeu_ps(
                        py.add(i + 16),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 16)), d2),
                    );
                    _mm256_storeu_ps(
                        py.add(i + 24),
                        _mm256_add_ps(_mm256_loadu_ps(py.add(i + 24)), d3),
                    );
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

/// Encodes a 64-byte [`KvMotionDeltaPacket`] (P-Frame Predictive Phase-Motion Delta) between
/// `prev_phases` and `curr_phases` using 8-lane SWAR circular difference, $S^1$ geodesic folding,
/// and QDPS step filtering.
///
/// # Errors
/// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127`.
#[inline]
pub fn encode_pframe_motion_delta_64(
    prev_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    curr_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    phase_step_min_q7: u8,
) -> GcsoResult<KvMotionDeltaPacket> {
    if phase_step_min_q7 > 127 {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let pw = load_q7_words_64(prev_phases);
    let cw = load_q7_words_64(curr_phases);
    let diff_folded = [
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[0], pw[0])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[1], pw[1])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[2], pw[2])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[3], pw[3])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[4], pw[4])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[5], pw[5])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[6], pw[6])),
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_diff_x8(cw[7], pw[7])),
    ];
    let filt_w = swar_q7_phase_step_filter_words_64(&diff_folded, phase_step_min_q7);
    let mut motion_deltas = [0i8; MAX_ATTENTION_HEADS];
    store_q7_words_64(&mut motion_deltas, &filt_w);
    Ok(KvMotionDeltaPacket::new(motion_deltas))
}

/// Applies a single 64-byte [`KvMotionDeltaPacket`] onto `cur_w` during Zero-Forward Seek,
/// negating the delta when transition coherence drops below `-safe_thresh` (contradiction guard).
///
/// Returns `(step_log_prob, repulsed)`.
#[inline(always)]
#[must_use]
pub fn apply_pframe_motion_step_words_64(
    cur_w: &mut [u64; 8],
    pframe: &KvMotionDeltaPacket,
    safe_thresh: f32,
) -> (f32, bool) {
    let dw = load_q7_words_64(&pframe.motion_deltas);
    let coh = coherence_from_diff_words_64(dw[0], dw[1], dw[2], dw[3], dw[4], dw[5], dw[6], dw[7]);
    let log_p = branchless_fast_transition_log_prob(coh);
    let repulsed = coh < -safe_thresh;
    if repulsed {
        *cur_w = [
            swar_q7_circular_add_x8(cur_w[0], swar_q7_neg_sat_x8(dw[0])),
            swar_q7_circular_add_x8(cur_w[1], swar_q7_neg_sat_x8(dw[1])),
            swar_q7_circular_add_x8(cur_w[2], swar_q7_neg_sat_x8(dw[2])),
            swar_q7_circular_add_x8(cur_w[3], swar_q7_neg_sat_x8(dw[3])),
            swar_q7_circular_add_x8(cur_w[4], swar_q7_neg_sat_x8(dw[4])),
            swar_q7_circular_add_x8(cur_w[5], swar_q7_neg_sat_x8(dw[5])),
            swar_q7_circular_add_x8(cur_w[6], swar_q7_neg_sat_x8(dw[6])),
            swar_q7_circular_add_x8(cur_w[7], swar_q7_neg_sat_x8(dw[7])),
        ];
    } else {
        *cur_w = [
            swar_q7_circular_add_x8(cur_w[0], dw[0]),
            swar_q7_circular_add_x8(cur_w[1], dw[1]),
            swar_q7_circular_add_x8(cur_w[2], dw[2]),
            swar_q7_circular_add_x8(cur_w[3], dw[3]),
            swar_q7_circular_add_x8(cur_w[4], dw[4]),
            swar_q7_circular_add_x8(cur_w[5], dw[5]),
            swar_q7_circular_add_x8(cur_w[6], dw[6]),
            swar_q7_circular_add_x8(cur_w[7], dw[7]),
        ];
    }
    (log_p, repulsed)
}

/// Applies $S^1$ geodesic folding and QDPS step filtering to an accumulated 64-head phase word array.
#[inline(always)]
#[must_use]
pub fn finalize_pframe_chain_words_64(cur_w: &[u64; 8], phase_step_min_q7: u8) -> [u64; 8] {
    let folded_w = [
        swar_q7_fold_s1_geodesic_x8(cur_w[0]),
        swar_q7_fold_s1_geodesic_x8(cur_w[1]),
        swar_q7_fold_s1_geodesic_x8(cur_w[2]),
        swar_q7_fold_s1_geodesic_x8(cur_w[3]),
        swar_q7_fold_s1_geodesic_x8(cur_w[4]),
        swar_q7_fold_s1_geodesic_x8(cur_w[5]),
        swar_q7_fold_s1_geodesic_x8(cur_w[6]),
        swar_q7_fold_s1_geodesic_x8(cur_w[7]),
    ];
    swar_q7_phase_step_filter_words_64(&folded_w, phase_step_min_q7)
}

/// Reconstructs a 64-head phase state by replaying `pframes[..seek_steps]` onto `anchor_phases`
/// without any LLM forward pass (Zero-Forward Seek).
///
/// Returns `(reconstructed_words, joint_log_prob, repulsed_count)`.
///
/// # Errors
/// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `seek_steps > pframes.len()`,
/// `phase_step_min_q7 > 127`, or `contradiction_threshold` is non-finite.
#[inline]
pub fn seek_pframe_chain_words_64(
    anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    pframes: &[KvMotionDeltaPacket],
    seek_steps: usize,
    phase_step_min_q7: u8,
    contradiction_threshold: f32,
) -> GcsoResult<([u64; 8], f32, u32)> {
    if seek_steps > pframes.len()
        || phase_step_min_q7 > 127
        || !branchless_is_finite_f32(contradiction_threshold)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let safe_thresh = branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
    let mut cur_w = load_q7_words_64(anchor_phases);
    let mut joint_log_prob = 0.0_f32;
    let mut repulsed_count = 0u32;
    for pframe in &pframes[..seek_steps] {
        let (log_p, repulsed) = apply_pframe_motion_step_words_64(&mut cur_w, pframe, safe_thresh);
        joint_log_prob += log_p;
        if repulsed {
            repulsed_count = repulsed_count.wrapping_add(1);
        }
    }
    let filt_w = finalize_pframe_chain_words_64(&cur_w, phase_step_min_q7);
    Ok((
        filt_w,
        branchless_flush_subnormal_f32(joint_log_prob),
        repulsed_count,
    ))
}
