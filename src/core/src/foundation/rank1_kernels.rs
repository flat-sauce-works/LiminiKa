// name: src/core/src/foundation/rank1_kernels.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure synchronous Rank-1 outer-product residual kernels (`eval_rank1*` and AVX2/scalar SIMD leaves).

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

use crate::foundation::aligned::{AlignedSlice32, AlignedSliceMut32};
use crate::foundation::pod::{
    ACTION_MODE_RANK1_INJECT, ArenaSpan, RANK1_GAIN_UNUSED, Rank1Descriptor, TaggedPtr,
};
use crate::foundation::rope::kernels::ptr_has_action_mode;
use crate::foundation::scalar_math::{branchless_flush_subnormal_f32, branchless_is_finite_f32};
use crate::foundation::simd_stats::dot_f32_unrolled4;
use crate::liminal::{LMNK_ERR_ENTROPY_SINGULARITY, LMNK_ERR_INVALID_ARGUMENT, LmnkResult};

// ===================================================================
// 1. AVX2 / Scalar SIMD Leaves for Rank-1 Outer Product
// ===================================================================

/// Evaluates Rank-1 outer product write `y_slice[i] = scale * u_slice[i] * g_i` using AVX2.
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
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for (y_i, &u_i) in y_slice[..len].iter_mut().zip(u_slice[..len].iter()) {
                *y_i = clean_scale * u_i;
            }
        }
    }
}

/// Accumulates Rank-1 outer product `y_slice[i] += scale * u_slice[i] * g_i` using AVX2.
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
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            for (y_i, &u_i) in y_slice[..len].iter_mut().zip(u_slice[..len].iter()) {
                *y_i += clean_scale * u_i;
            }
        }
    }
}

// ===================================================================
// 2. Synchronous Rank-1 Residual Adapter Pure Functions
// ===================================================================

type ResolvedRank1Slices<'a> = (&'a [f32], &'a [f32], Option<&'a [f32]>);

/// Resolves and validates `desc`'s [`ArenaSpan`] offsets inside `arena` without raw pointer casts.
///
/// Enforces:
/// - `desc.rank == 1`, `d_in > 0`, `d_out > 0`, and finite `desc.scale_factor`
/// - `desc.u_span.len == d_out` and `desc.v_span.len == d_in`
/// - All spans (`u_span`, `v_span`, and optional `gain_offset..gain_offset+u_span.len`) lie
///   within `arena` without 32-bit integer overflow.
#[inline(always)]
fn resolve_rank1_arena_slices<'a>(
    desc: &Rank1Descriptor,
    arena: &'a [f32],
    d_in: usize,
    d_out: usize,
) -> LmnkResult<ResolvedRank1Slices<'a>> {
    if desc.rank != 1
        || d_in == 0
        || d_out == 0
        || (desc.u_span.len as usize) != d_out
        || (desc.v_span.len as usize) != d_in
        || !branchless_is_finite_f32(desc.scale_factor)
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let u_slice = desc.u_span.slice(arena).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    let v_slice = desc.v_span.slice(arena).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
    let g_slice_opt = if desc.gain_offset != RANK1_GAIN_UNUSED {
        let g_span = ArenaSpan::new(desc.gain_offset, desc.u_span.len);
        Some(g_span.slice(arena).ok_or(LMNK_ERR_INVALID_ARGUMENT)?)
    } else {
        None
    };
    Ok((u_slice, v_slice, g_slice_opt))
}

/// Evaluates the standalone Rank-1 outer product $\mathbf{y} = s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
/// into `y_out` from safe hardware-aligned slices using AVX2 (`dot_f32_unrolled4`, `rank1_write_outer_product_simd`).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `y_out`, `x_in`, `u_vec`, or `v_vec` is empty,
/// `x_in.len() != v_vec.len()`, `y_out.len() != u_vec.len()`, or `scale_factor` is non-finite.
#[inline]
pub fn eval_rank1_slices(
    y_out: &mut AlignedSliceMut32<'_, f32>,
    x_in: &AlignedSlice32<'_, f32>,
    u_vec: &AlignedSlice32<'_, f32>,
    v_vec: &AlignedSlice32<'_, f32>,
    gain_vec: Option<&AlignedSlice32<'_, f32>>,
    scale_factor: f32,
) -> LmnkResult<()> {
    let d_in = x_in.len();
    let d_out = y_out.len();
    if d_in == 0
        || d_out == 0
        || v_vec.len() < d_in
        || u_vec.len() < d_out
        || !branchless_is_finite_f32(scale_factor)
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    if let Some(g) = gain_vec
        && g.len() < d_out
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }

    let dot_vx = dot_f32_unrolled4(v_vec.as_slice(), x_in.as_slice(), d_in);
    if !branchless_is_finite_f32(dot_vx) {
        return Err(LMNK_ERR_ENTROPY_SINGULARITY);
    }
    let eff_scale = branchless_flush_subnormal_f32(scale_factor * dot_vx);
    let g_slice_opt = gain_vec.map(|g| &g.as_slice()[..d_out]);
    rank1_write_outer_product_simd(
        &mut y_out.as_mut_slice()[..d_out],
        &u_vec.as_slice()[..d_out],
        g_slice_opt,
        eff_scale,
    );
    Ok(())
}

/// Accumulates the fused Rank-1 outer product residual $\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
/// in-place into `y_inout` from safe hardware-aligned slices using AVX2 (`rank1_accum_outer_product_simd`).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any slice is empty, lengths mismatch, or `scale_factor` is non-finite.
#[inline]
pub fn eval_rank1_slices_fused(
    y_inout: &mut AlignedSliceMut32<'_, f32>,
    x_in: &AlignedSlice32<'_, f32>,
    u_vec: &AlignedSlice32<'_, f32>,
    v_vec: &AlignedSlice32<'_, f32>,
    gain_vec: Option<&AlignedSlice32<'_, f32>>,
    scale_factor: f32,
) -> LmnkResult<()> {
    let d_in = x_in.len();
    let d_out = y_inout.len();
    if d_in == 0
        || d_out == 0
        || v_vec.len() < d_in
        || u_vec.len() < d_out
        || !branchless_is_finite_f32(scale_factor)
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    if let Some(g) = gain_vec
        && g.len() < d_out
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }

    let dot_vx = dot_f32_unrolled4(v_vec.as_slice(), x_in.as_slice(), d_in);
    if !branchless_is_finite_f32(dot_vx) {
        return Err(LMNK_ERR_ENTROPY_SINGULARITY);
    }
    let eff_scale = branchless_flush_subnormal_f32(scale_factor * dot_vx);
    let g_slice_opt = gain_vec.map(|g| &g.as_slice()[..d_out]);
    rank1_accum_outer_product_simd(
        &mut y_inout.as_mut_slice()[..d_out],
        &u_vec.as_slice()[..d_out],
        g_slice_opt,
        eff_scale,
    );
    Ok(())
}

/// Evaluates the standalone Rank-1 outer product $\mathbf{y} = s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
/// from a portable 32-byte [`Rank1Descriptor`] and caller-owned `arena` slice (100% Strict Provenance safe).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `desc.rank != 1`, `desc.scale_factor` is non-finite,
/// `y_out` / `x_in` lengths disagree with `u_span` / `v_span`, or any arena span overflows/exceeds `arena`.
/// Returns [`LMNK_ERR_ENTROPY_SINGULARITY`] if $\mathbf{v}^T \mathbf{x}$ is non-finite.
#[inline]
pub fn eval_rank1(
    y_out: &mut AlignedSliceMut32<'_, f32>,
    x_in: &AlignedSlice32<'_, f32>,
    arena: &[f32],
    desc: &Rank1Descriptor,
) -> LmnkResult<()> {
    let d_in = x_in.len();
    let d_out = y_out.len();
    let (u_slice, v_slice, g_slice_opt) = resolve_rank1_arena_slices(desc, arena, d_in, d_out)?;
    let dot_vx = dot_f32_unrolled4(v_slice, x_in.as_slice(), d_in);
    if !branchless_is_finite_f32(dot_vx) {
        return Err(LMNK_ERR_ENTROPY_SINGULARITY);
    }
    let eff_scale = branchless_flush_subnormal_f32(desc.scale_factor * dot_vx);
    rank1_write_outer_product_simd(
        &mut y_out.as_mut_slice()[..d_out],
        u_slice,
        g_slice_opt,
        eff_scale,
    );
    Ok(())
}

/// Accumulates the fused Rank-1 outer product residual $\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
/// in-place into `y_inout` from a portable 32-byte [`Rank1Descriptor`] and caller-owned `arena` slice.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `desc.rank != 1`, `desc.scale_factor` is non-finite,
/// `y_inout` / `x_in` lengths disagree with `u_span` / `v_span`, or any arena span overflows/exceeds `arena`.
/// Returns [`LMNK_ERR_ENTROPY_SINGULARITY`] if $\mathbf{v}^T \mathbf{x}$ is non-finite.
#[inline]
pub fn eval_rank1_fused(
    y_inout: &mut AlignedSliceMut32<'_, f32>,
    x_in: &AlignedSlice32<'_, f32>,
    arena: &[f32],
    desc: &Rank1Descriptor,
) -> LmnkResult<()> {
    let d_in = x_in.len();
    let d_out = y_inout.len();
    let (u_slice, v_slice, g_slice_opt) = resolve_rank1_arena_slices(desc, arena, d_in, d_out)?;
    let dot_vx = dot_f32_unrolled4(v_slice, x_in.as_slice(), d_in);
    if !branchless_is_finite_f32(dot_vx) {
        return Err(LMNK_ERR_ENTROPY_SINGULARITY);
    }
    let eff_scale = branchless_flush_subnormal_f32(desc.scale_factor * dot_vx);
    rank1_accum_outer_product_simd(
        &mut y_inout.as_mut_slice()[..d_out],
        u_slice,
        g_slice_opt,
        eff_scale,
    );
    Ok(())
}

/// Accumulates the fused Rank-1 outer product residual gated by `tagged_ptr`'s `ACTION_MODE_RANK1_INJECT` bit (`0x10`).
///
/// # Errors
/// Propagates any descriptor or arena validation error from [`eval_rank1_fused`].
#[inline]
pub fn eval_rank1_pointer_gated_fused(
    tagged_ptr: TaggedPtr,
    y_inout: &mut AlignedSliceMut32<'_, f32>,
    x_in: &AlignedSlice32<'_, f32>,
    arena: &[f32],
    desc: &Rank1Descriptor,
) -> LmnkResult<bool> {
    let d_in = x_in.len();
    let d_out = y_inout.len();
    let (u_slice, v_slice, g_slice_opt) = resolve_rank1_arena_slices(desc, arena, d_in, d_out)?;
    if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_RANK1_INJECT) {
        return Ok(false);
    }
    let dot_vx = dot_f32_unrolled4(v_slice, x_in.as_slice(), d_in);
    if !branchless_is_finite_f32(dot_vx) {
        return Err(LMNK_ERR_ENTROPY_SINGULARITY);
    }
    let eff_scale = branchless_flush_subnormal_f32(desc.scale_factor * dot_vx);
    rank1_accum_outer_product_simd(
        &mut y_inout.as_mut_slice()[..d_out],
        u_slice,
        g_slice_opt,
        eff_scale,
    );
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::aligned::AlignedBuffer32;
    use crate::foundation::pod::ACTION_PRESET_FULL_ORCHESTRATION;

    #[test]
    fn test_eval_rank1_pointer_gated_and_gain_fused() {
        let mut y_rank1 = AlignedBuffer32::new([1.0_f32; 8]);
        let x_rank1 = AlignedBuffer32::new([1.0_f32; 8]);
        // Arena layout: [u (0..8), v (8..16), g (16..24)].
        let mut arena = [0.0_f32; 24];
        arena[0..8].fill(0.5);
        arena[8..16].fill(0.25);
        arena[16..24].fill(2.0);
        let desc = Rank1Descriptor {
            layer_idx: 0,
            rank: 1,
            u_span: ArenaSpan::new(0, 8),
            v_span: ArenaSpan::new(8, 8),
            gain_offset: RANK1_GAIN_UNUSED,
            scale_factor: 1.0,
        };
        let ptr = TaggedPtr::pack(0x1000, (1 << 5) | ACTION_PRESET_FULL_ORCHESTRATION, 0);
        let injected = eval_rank1_pointer_gated_fused(
            ptr,
            &mut y_rank1.as_aligned_slice_mut(),
            &x_rank1.as_aligned_slice(),
            &arena,
            &desc,
        )
        .unwrap();
        assert!(injected);
        assert!((y_rank1.data[0] - 2.0).abs() < 1.0e-5);

        let desc_with_gain = Rank1Descriptor {
            gain_offset: 16,
            ..desc
        };
        eval_rank1_fused(
            &mut y_rank1.as_aligned_slice_mut(),
            &x_rank1.as_aligned_slice(),
            &arena,
            &desc_with_gain,
        )
        .unwrap();
        assert!((y_rank1.data[0] - 4.0).abs() < 1.0e-5);

        // Overflowing gain_offset and out-of-bounds span must fail closed.
        let desc_overflow = Rank1Descriptor {
            gain_offset: u32::MAX - 2,
            ..desc
        };
        assert_eq!(
            eval_rank1(
                &mut y_rank1.as_aligned_slice_mut(),
                &x_rank1.as_aligned_slice(),
                &arena,
                &desc_overflow,
            ),
            Err(LMNK_ERR_INVALID_ARGUMENT)
        );
        let desc_oob = Rank1Descriptor {
            u_span: ArenaSpan::new(20, 8),
            ..desc
        };
        assert_eq!(
            eval_rank1(
                &mut y_rank1.as_aligned_slice_mut(),
                &x_rank1.as_aligned_slice(),
                &arena,
                &desc_oob,
            ),
            Err(LMNK_ERR_INVALID_ARGUMENT)
        );
    }
}
