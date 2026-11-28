// name: src/core/src/foundation/rope/visual.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2D/3D spatiotemporal visual RoPE attention steering and frame-delta bitmask detection kernels.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

use crate::foundation::aligned::{AlignedSlice32, AlignedSliceMut32};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::foundation::phase_ops::avx2_hsum_ps;
use crate::foundation::pod::{PAGED_BITMASK_BITS, PagedBitmask};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use crate::foundation::scalar_math::branchless_rotate_so2_pair;
use crate::foundation::scalar_math::{
    branchless_is_finite_f32, branchless_sincos_q7, fast_inv_len_f32,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::foundation::simd_stats::avx2_rotate_so2_block32;
use crate::liminal::{LMNK_ERR_INVALID_ARGUMENT, LmnkResult, PhaseQ7};

// ===================================================================
// 1. AVX2 / Scalar SIMD Kernels for 2D/3D Visual RoPE
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

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
pub(crate) fn rotate_so2_2d_xy_slice_simd(
    buf: &mut [f32],
    sin_x: f32,
    cos_x: f32,
    sin_y: f32,
    cos_y: f32,
) {
    for quad in buf.as_chunks_mut::<4>().0 {
        (quad[0], quad[1]) = branchless_rotate_so2_pair(quad[0], quad[1], sin_x, cos_x);
        (quad[2], quad[3]) = branchless_rotate_so2_pair(quad[2], quad[3], sin_y, cos_y);
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
        use crate::foundation::scalar_math::branchless_rotate_so2_pair;

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
        for hex in buf.as_chunks_mut::<6>().0 {
            (hex[0], hex[1]) = branchless_rotate_so2_pair(hex[0], hex[1], sin_t, cos_t);
            (hex[2], hex[3]) = branchless_rotate_so2_pair(hex[2], hex[3], sin_x, cos_x);
            (hex[4], hex[5]) = branchless_rotate_so2_pair(hex[4], hex[5], sin_y, cos_y);
        }
    }
}

// ===================================================================
// 2. 2D/3D Spatiotemporal Visual Attention Steering & Bitmask Detection
// ===================================================================

#[inline(always)]
#[must_use]
fn patch_diff_sum_sq_unrolled4(cur_patch: &[f32], prv_patch: &[f32]) -> f32 {
    let n = cur_patch.len().min(prv_patch.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pc = cur_patch.as_ptr();
        let pp = prv_patch.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and pointer offsets are bounds-checked against `n`.
        unsafe {
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= n {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pc.add(i)), _mm256_loadu_ps(pp.add(i)));
                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pc.add(i + 8)),
                    _mm256_loadu_ps(pp.add(i + 8)),
                );
                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(pc.add(i + 16)),
                    _mm256_loadu_ps(pp.add(i + 16)),
                );
                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(pc.add(i + 24)),
                    _mm256_loadu_ps(pp.add(i + 24)),
                );
                sq0 = _mm256_add_ps(
                    sq0,
                    _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1)),
                );
                sq1 = _mm256_add_ps(
                    sq1,
                    _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3)),
                );
                i += 32;
            }
            while i + 8 <= n {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pc.add(i)), _mm256_loadu_ps(pp.add(i)));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(d0, d0));
                i += 8;
            }
            let mut sum_sq = avx2_hsum_ps(_mm256_add_ps(sq0, sq1));
            while i < n {
                let d = *pc.add(i) - *pp.add(i);
                sum_sq += d * d;
                i += 1;
            }
            sum_sq
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut acc0 = 0.0_f32;
        let mut acc1 = 0.0_f32;
        let mut acc2 = 0.0_f32;
        let mut acc3 = 0.0_f32;
        let mut i = 0usize;
        while i + 4 <= n {
            let d0 = cur_patch[i] - prv_patch[i];
            let d1 = cur_patch[i + 1] - prv_patch[i + 1];
            let d2 = cur_patch[i + 2] - prv_patch[i + 2];
            let d3 = cur_patch[i + 3] - prv_patch[i + 3];
            acc0 += d0 * d0;
            acc1 += d1 * d1;
            acc2 += d2 * d2;
            acc3 += d3 * d3;
            i += 4;
        }
        let mut sum_sq = (acc0 + acc1) + (acc2 + acc3);
        while i < n {
            let d = cur_patch[i] - prv_patch[i];
            sum_sq += d * d;
            i += 1;
        }
        sum_sq
    }
}

#[inline(always)]
fn steer_visual_grid_bitmask_core<F>(
    grid_tensor: &mut AlignedSliceMut32<'_, f32>,
    active_patch_mask: &PagedBitmask,
    patch_stride_elems: usize,
    num_patches: usize,
    min_dim_multiple: usize,
    mut rotate_patch_fn: F,
) -> LmnkResult<u32>
where
    F: FnMut(&mut [f32]),
{
    if num_patches == 0
        || num_patches > (PAGED_BITMASK_BITS as usize)
        || patch_stride_elems == 0
        || !patch_stride_elems.is_multiple_of(min_dim_multiple)
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let required_elems = num_patches
        .checked_mul(patch_stride_elems)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    if grid_tensor.len() < required_elems {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }

    let g_slice = grid_tensor.as_mut_slice();
    let mut steered_count = 0u32;

    for (w_idx, &word_val) in active_patch_mask.bits.iter().enumerate() {
        let patch_base = w_idx * 64;
        if patch_base >= num_patches {
            break;
        }
        let max_bits = (num_patches - patch_base).min(64);
        let valid_mask = if max_bits == 64 {
            u64::MAX
        } else {
            (1u64 << max_bits) - 1
        };
        let mut word = word_val & valid_mask;
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            let run_bits = (word >> bit).trailing_ones() as usize;
            let next_bit = bit + run_bits;
            word = if next_bit >= 64 {
                0
            } else {
                word & (!0u64 << next_bit)
            };
            let elem_start = (patch_base + bit) * patch_stride_elems;
            let elem_len = run_bits * patch_stride_elems;
            rotate_patch_fn(&mut g_slice[elem_start..elem_start + elem_len]);
            steered_count += run_bits as u32;
        }
    }
    Ok(steered_count)
}

/// Computes the 256-bit frame delta bitmask (`prev_mask ^ curr_mask`) and active changed patch
/// count (`0..=256`) in $\mathcal{O}(1)$ branchless POPCNT arithmetic.
#[inline(always)]
#[must_use]
pub fn detect_frame_delta_bitmask(
    prev_mask: &PagedBitmask,
    curr_mask: &PagedBitmask,
) -> (PagedBitmask, u32) {
    let w0 = prev_mask.bits[0] ^ curr_mask.bits[0];
    let w1 = prev_mask.bits[1] ^ curr_mask.bits[1];
    let w2 = prev_mask.bits[2] ^ curr_mask.bits[2];
    let w3 = prev_mask.bits[3] ^ curr_mask.bits[3];
    let changed = w0.count_ones() + w1.count_ones() + w2.count_ones() + w3.count_ones();
    (
        PagedBitmask {
            bits: [w0, w1, w2, w3],
        },
        changed,
    )
}

/// Detects active visual patches between raw `f32` frame buffers `current_frame` and `prev_frame`
/// whose mean squared difference exceeds `delta_threshold_sq`, populating a 256-bit
/// [`PagedBitmask`] (`mask_out`) and returning the active patch count (`0..=256`).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `num_patches == 0`, `num_patches > 256`,
/// `patch_stride_elems == 0`, `delta_threshold_sq < 0.0`, or frame buffers are too small.
#[inline]
pub fn detect_frame_mse_bitmask(
    current_frame: &AlignedSlice32<'_, f32>,
    prev_frame: &AlignedSlice32<'_, f32>,
    num_patches: usize,
    patch_stride_elems: usize,
    delta_threshold_sq: f32,
    mask_out: &mut PagedBitmask,
) -> LmnkResult<u32> {
    let total_elems = num_patches
        .checked_mul(patch_stride_elems)
        .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    if num_patches == 0
        || num_patches > (PAGED_BITMASK_BITS as usize)
        || patch_stride_elems == 0
        || delta_threshold_sq < 0.0
        || !branchless_is_finite_f32(delta_threshold_sq)
        || current_frame.len() < total_elems
        || prev_frame.len() < total_elems
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }

    *mask_out = PagedBitmask::default();
    let cur_s = &current_frame.as_slice()[..total_elems];
    let prv_s = &prev_frame.as_slice()[..total_elems];
    let inv_stride = fast_inv_len_f32(patch_stride_elems);
    let mut active_count = 0u32;

    for (p_idx, (c_patch, p_patch)) in cur_s
        .chunks_exact(patch_stride_elems)
        .zip(prv_s.chunks_exact(patch_stride_elems))
        .enumerate()
    {
        let sum_sq = patch_diff_sum_sq_unrolled4(c_patch, p_patch);
        let mse = sum_sq * inv_stride;
        let active = mse > delta_threshold_sq;
        let word_idx = p_idx >> 6;
        let bit_idx = p_idx & 63;
        mask_out.bits[word_idx] |= u64::from(active) << bit_idx;
        active_count += u32::from(active);
    }
    Ok(active_count)
}

/// Steers only the active 2D visual patches indicated by `active_patch_mask` (`bits[0..4]`)
/// using `ctz` bit-scanning (`word &= word - 1`) in $\mathcal{O}(\mathrm{popcount}(\mathbf{M}))$.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `num_patches == 0`, `num_patches > 256`,
/// `patch_stride_elems` is zero or not a multiple of `4`, or `grid_tensor` is too small.
#[inline]
pub fn steer_visual_grid_2d_bitmask_gated(
    grid_tensor: &mut AlignedSliceMut32<'_, f32>,
    active_patch_mask: &PagedBitmask,
    patch_stride_elems: usize,
    num_patches: usize,
    phase_x_q7: PhaseQ7,
    phase_y_q7: PhaseQ7,
) -> LmnkResult<u32> {
    let (sin_x, cos_x) = branchless_sincos_q7(phase_x_q7);
    let (sin_y, cos_y) = branchless_sincos_q7(phase_y_q7);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `target_feature = "avx2"` is enabled; hoists `(v_cos, v_sin)` construction
        // once outside the 256-patch bitmask scan.
        let (v_cos, v_sin) = unsafe { avx2_build_so2_2d_xy_vecs(sin_x, cos_x, sin_y, cos_y) };
        steer_visual_grid_bitmask_core(
            grid_tensor,
            active_patch_mask,
            patch_stride_elems,
            num_patches,
            4,
            |patch_slice| {
                // SAFETY: `target_feature = "avx2"` is enabled and `patch_slice` is valid.
                unsafe {
                    avx2_rotate_so2_2d_xy_slice_with_vecs(patch_slice, v_cos, v_sin);
                }
            },
        )
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    steer_visual_grid_bitmask_core(
        grid_tensor,
        active_patch_mask,
        patch_stride_elems,
        num_patches,
        4,
        |patch_slice| rotate_so2_2d_xy_slice_simd(patch_slice, sin_x, cos_x, sin_y, cos_y),
    )
}

/// Steers only the active 3D spatiotemporal patches indicated by `active_patch_mask` (`bits[0..4]`)
/// using `ctz` bit-scanning (`word &= word - 1`) in $\mathcal{O}(\mathrm{popcount}(\mathbf{M}))$.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `num_patches == 0`, `num_patches > 256`,
/// `patch_stride_elems` is zero or not a multiple of `6`, or `grid_tensor` is too small.
#[inline]
pub fn steer_visual_grid_3d_bitmask_gated(
    grid_tensor: &mut AlignedSliceMut32<'_, f32>,
    active_patch_mask: &PagedBitmask,
    patch_stride_elems: usize,
    num_patches: usize,
    phase_t_q7: PhaseQ7,
    phase_x_q7: PhaseQ7,
    phase_y_q7: PhaseQ7,
) -> LmnkResult<u32> {
    let (sin_t, cos_t) = branchless_sincos_q7(phase_t_q7);
    let (sin_x, cos_x) = branchless_sincos_q7(phase_x_q7);
    let (sin_y, cos_y) = branchless_sincos_q7(phase_y_q7);
    steer_visual_grid_bitmask_core(
        grid_tensor,
        active_patch_mask,
        patch_stride_elems,
        num_patches,
        6,
        |patch_slice| {
            rotate_so2_3d_txy_slice_simd(patch_slice, sin_t, cos_t, sin_x, cos_x, sin_y, cos_y);
        },
    )
}
