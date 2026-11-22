// name: src/core/src/foundation/rank1_kernels.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure synchronous Rank-1 outer-product residual kernels (`eval_rank1*` and AVX2/scalar SIMD leaves).

use core::slice;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::{
    _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

use crate::foundation::aligned::{AlignedSlice32, AlignedSliceMut32};
use crate::foundation::pod::{
    raw_byte_ranges_overlap, validate_raw_slice_bounds, Rank1Descriptor, TaggedPtr,
    ACTION_MODE_RANK1_INJECT,
};
use crate::foundation::rope::kernels::ptr_has_action_mode;
use crate::foundation::scalar_math::{branchless_flush_subnormal_f32, branchless_is_finite_f32};
use crate::foundation::simd_stats::dot_f32_unrolled4;
use crate::liminal::{
    LmnkResult, LMNK_ERR_ALIGNMENT, LMNK_ERR_ENTROPY_SINGULARITY, LMNK_ERR_INVALID_ARGUMENT,
    LMNK_ERR_NULL_POINTER,
};

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

// ===================================================================
// 2. Synchronous Rank-1 Residual Adapter Pure Functions
// ===================================================================

/// # Safety
/// Caller must ensure `desc.u_vector_ptr` (`d_out` floats), `desc.v_vector_ptr` (`d_in` floats),
/// and optional `desc.gain_scalar_ptr` (`d_out` floats) point to valid, initialized, 32-byte aligned
/// memory readable for lifetime `'a` without concurrent mutation.
#[inline(always)]
unsafe fn validate_and_unpack_rank1_descriptor<'a>(
    desc: &'a Rank1Descriptor,
    d_in: usize,
    d_out: usize,
    y_out_ptr: *const f32,
) -> LmnkResult<(
    AlignedSlice32<'a, f32>,
    AlignedSlice32<'a, f32>,
    Option<AlignedSlice32<'a, f32>>,
)> {
    if desc.u_vector_ptr == 0 || desc.v_vector_ptr == 0 {
        return Err(LMNK_ERR_NULL_POINTER);
    }
    if desc.rank != 1 || d_in == 0 || d_out == 0 || !branchless_is_finite_f32(desc.scale_factor) {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let u_ptr = desc.u_vector_ptr as *const f32;
    let v_ptr = desc.v_vector_ptr as *const f32;
    if (u_ptr as usize) & 31 != 0 || (v_ptr as usize) & 31 != 0 {
        return Err(LMNK_ERR_ALIGNMENT);
    }
    let y_bytes = validate_raw_slice_bounds(y_out_ptr, d_out)?;
    let u_bytes = validate_raw_slice_bounds(u_ptr, d_out)?;
    let v_bytes = validate_raw_slice_bounds(v_ptr, d_in)?;
    if raw_byte_ranges_overlap(y_out_ptr as usize, y_bytes, u_ptr as usize, u_bytes)
        || raw_byte_ranges_overlap(y_out_ptr as usize, y_bytes, v_ptr as usize, v_bytes)
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }

    let g_ptr_opt = if desc.gain_scalar_ptr != 0 {
        let g_ptr = desc.gain_scalar_ptr as *const f32;
        if (g_ptr as usize) & 31 != 0 {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        let g_bytes = validate_raw_slice_bounds(g_ptr, d_out)?;
        if raw_byte_ranges_overlap(y_out_ptr as usize, y_bytes, g_ptr as usize, g_bytes) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        Some(g_ptr)
    } else {
        None
    };

    // SAFETY: `u_ptr`, `v_ptr`, and optional `g_ptr` are non-null, 32-byte aligned, bounds-checked,
    // disjoint from `y_out_ptr`, and caller guarantees valid immutable lifetimes for `'a`.
    let (u_slice, v_slice, g_slice_opt) = unsafe {
        (
            AlignedSlice32::new_unchecked(slice::from_raw_parts(u_ptr, d_out)),
            AlignedSlice32::new_unchecked(slice::from_raw_parts(v_ptr, d_in)),
            g_ptr_opt.map(|gp| AlignedSlice32::new_unchecked(slice::from_raw_parts(gp, d_out))),
        )
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
    if let Some(g) = gain_vec {
        if g.len() < d_out {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
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
    if let Some(g) = gain_vec {
        if g.len() < d_out {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
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
/// from a C-ABI [`Rank1Descriptor`] (including optional 32B-aligned `gain_scalar_ptr`).
///
/// # Safety
/// Caller must guarantee that `desc.u_vector_ptr` (`y_out.len()` `f32`s), `desc.v_vector_ptr`
/// (`x_in.len()` `f32`s), and any non-zero `desc.gain_scalar_ptr` (`y_out.len()` `f32`s) point to
/// valid, initialized, 32-byte aligned, immutable buffers for the duration of this call.
///
/// # Errors
/// Returns [`LMNK_ERR_NULL_POINTER`] if `u_vector_ptr == 0` or `v_vector_ptr == 0`,
/// [`LMNK_ERR_ALIGNMENT`] if pointers are not 32B-aligned, or
/// [`LMNK_ERR_INVALID_ARGUMENT`] if `desc.rank != 1`, `desc.scale_factor` is non-finite,
/// slices are empty, or input descriptors overlap `y_out`.
#[inline]
pub unsafe fn eval_rank1(
    y_out: &mut AlignedSliceMut32<'_, f32>,
    x_in: &AlignedSlice32<'_, f32>,
    desc: &Rank1Descriptor,
) -> LmnkResult<()> {
    let y_ptr = y_out.as_ptr();
    let y_len = y_out.len();
    // SAFETY: Caller guarantees valid immutable lifetimes for `desc`'s raw pointers.
    let (u_slice, v_slice, g_slice_opt) =
        unsafe { validate_and_unpack_rank1_descriptor(desc, x_in.len(), y_len, y_ptr)? };
    eval_rank1_slices(
        y_out,
        x_in,
        &u_slice,
        &v_slice,
        g_slice_opt.as_ref(),
        desc.scale_factor,
    )
}

/// Accumulates the fused Rank-1 outer product residual $\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
/// in-place into `y_inout` from a C-ABI [`Rank1Descriptor`] (including optional 32B-aligned `gain_scalar_ptr`).
///
/// # Safety
/// Caller must guarantee that `desc.u_vector_ptr` (`y_inout.len()` `f32`s), `desc.v_vector_ptr`
/// (`x_in.len()` `f32`s), and any non-zero `desc.gain_scalar_ptr` (`y_inout.len()` `f32`s) point to
/// valid, initialized, 32-byte aligned, immutable buffers for the duration of this call.
///
/// # Errors
/// Returns [`LMNK_ERR_NULL_POINTER`] if `u_vector_ptr == 0` or `v_vector_ptr == 0`,
/// [`LMNK_ERR_ALIGNMENT`] if pointers are not 32B-aligned, or
/// [`LMNK_ERR_INVALID_ARGUMENT`] if `desc.rank != 1`, `desc.scale_factor` is non-finite,
/// slices are empty, or input descriptors overlap `y_inout`.
#[inline]
pub unsafe fn eval_rank1_fused(
    y_inout: &mut AlignedSliceMut32<'_, f32>,
    x_in: &AlignedSlice32<'_, f32>,
    desc: &Rank1Descriptor,
) -> LmnkResult<()> {
    let y_ptr = y_inout.as_ptr();
    let y_len = y_inout.len();
    // SAFETY: Caller guarantees valid immutable lifetimes for `desc`'s raw pointers.
    let (u_slice, v_slice, g_slice_opt) =
        unsafe { validate_and_unpack_rank1_descriptor(desc, x_in.len(), y_len, y_ptr)? };
    eval_rank1_slices_fused(
        y_inout,
        x_in,
        &u_slice,
        &v_slice,
        g_slice_opt.as_ref(),
        desc.scale_factor,
    )
}

/// Accumulates the fused Rank-1 outer product residual gated by `tagged_ptr`'s `ACTION_MODE_RANK1_INJECT` bit (`0x10`).
///
/// # Safety
/// Caller must guarantee that `desc`'s raw pointers satisfy the safety contract of [`eval_rank1_fused`].
///
/// # Errors
/// Propagates any pointer/slice validation error from [`eval_rank1_fused`].
#[inline]
pub unsafe fn eval_rank1_pointer_gated_fused(
    tagged_ptr: TaggedPtr,
    y_inout: &mut AlignedSliceMut32<'_, f32>,
    x_in: &AlignedSlice32<'_, f32>,
    desc: &Rank1Descriptor,
) -> LmnkResult<bool> {
    let y_ptr = y_inout.as_ptr();
    let y_len = y_inout.len();
    // SAFETY: Caller guarantees valid immutable lifetimes for `desc`'s raw pointers.
    let (u_slice, v_slice, g_slice_opt) =
        unsafe { validate_and_unpack_rank1_descriptor(desc, x_in.len(), y_len, y_ptr)? };
    if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_RANK1_INJECT) {
        return Ok(false);
    }
    eval_rank1_slices_fused(
        y_inout,
        x_in,
        &u_slice,
        &v_slice,
        g_slice_opt.as_ref(),
        desc.scale_factor,
    )?;
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
        let u_rank1 = AlignedBuffer32::new([0.5_f32; 8]);
        let v_rank1 = AlignedBuffer32::new([0.25_f32; 8]);
        let g_rank1 = AlignedBuffer32::new([2.0_f32; 8]);
        let desc = Rank1Descriptor {
            layer_idx: 0,
            rank: 1,
            u_vector_ptr: u_rank1.as_aligned_slice().as_ptr() as u64,
            v_vector_ptr: v_rank1.as_aligned_slice().as_ptr() as u64,
            gain_scalar_ptr: 0,
            scale_factor: 1.0,
            flags: 0,
            reserved: [0; 24],
        };
        let ptr = TaggedPtr::pack(0x1000, (1 << 5) | ACTION_PRESET_FULL_ORCHESTRATION, 0);
        // SAFETY: `u_rank1`, `v_rank1`, and `g_rank1` are live, 32-byte aligned, and disjoint from `y_rank1`.
        let injected = unsafe {
            eval_rank1_pointer_gated_fused(
                ptr,
                &mut y_rank1.as_aligned_slice_mut(),
                &x_rank1.as_aligned_slice(),
                &desc,
            )
        }
        .unwrap();
        assert!(injected);
        assert!((y_rank1.data[0] - 2.0).abs() < 1.0e-5);

        let desc_with_gain = Rank1Descriptor {
            gain_scalar_ptr: g_rank1.as_aligned_slice().as_ptr() as u64,
            ..desc
        };
        // SAFETY: `u_rank1`, `v_rank1`, and `g_rank1` are live, 32-byte aligned, and disjoint from `y_rank1`.
        unsafe {
            eval_rank1_fused(
                &mut y_rank1.as_aligned_slice_mut(),
                &x_rank1.as_aligned_slice(),
                &desc_with_gain,
            )
        }
        .unwrap();
        assert!((y_rank1.data[0] - 4.0).abs() < 1.0e-5);
    }
}
