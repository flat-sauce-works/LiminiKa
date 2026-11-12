// name: src/core/src/traits/rope/adapters.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Synchronous Residual Adapter (SrlAdapter) and 2D/3D VisualAttentionSteering traits.

use super::*;

// ===================================================================
// 5. Synchronous Residual Adapter (`SrlAdapter`)
// ===================================================================

#[inline(always)]
pub(super) fn validate_and_unpack_srl_descriptor<'a>(
    desc: &'a gcso_srl_descriptor_t,
    d_in: usize,
    d_out: usize,
) -> GcsoResult<(
    AlignedSlice32<'a, f32>,
    AlignedSlice32<'a, f32>,
    Option<AlignedSlice32<'a, f32>>,
)> {
    if desc.u_vector_ptr == 0 || desc.v_vector_ptr == 0 {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    if desc.rank != 1 || d_in == 0 || d_out == 0 || !branchless_is_finite_f32(desc.scale_factor) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let u_ptr = desc.u_vector_ptr as *const f32;
    let v_ptr = desc.v_vector_ptr as *const f32;
    if (u_ptr as usize) & 31 != 0 || (v_ptr as usize) & 31 != 0 {
        return Err(GCSO_ERROR_MISALIGNED_POINTER);
    }
    let g_slice_opt = if desc.gain_scalar_ptr != 0 {
        let g_ptr = desc.gain_scalar_ptr as *const f32;
        if (g_ptr as usize) & 31 != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: `g_ptr` is non-null, 32-byte aligned, and sized to `d_out` elements per C-ABI contract.
        Some(unsafe { AlignedSlice32::new_unchecked(slice::from_raw_parts(g_ptr, d_out)) })
    } else {
        None
    };
    // SAFETY: `u_ptr` and `v_ptr` are non-null, 32-byte aligned, and sized to `d_out` and `d_in`.
    let u_slice = unsafe { AlignedSlice32::new_unchecked(slice::from_raw_parts(u_ptr, d_out)) };
    // SAFETY: `v_ptr` is non-null, 32-byte aligned, and sized to `d_in`.
    let v_slice = unsafe { AlignedSlice32::new_unchecked(slice::from_raw_parts(v_ptr, d_in)) };
    Ok((u_slice, v_slice, g_slice_opt))
}

/// Trait for Synchronous Residual Layer (SRL) Dynamic Rank-1 Outer Product Injection
/// ($\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$)
/// (**Micro Level**, Section 1.5 & 5.2).
pub trait SrlAdapter: BranchlessKernel {
    /// Evaluates the standalone Rank-1 outer product $\mathbf{y} = s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
    /// into `y_out` from safe hardware-aligned slices using AVX2 (`dot_f32_unrolled4`, `srl_write_outer_product_simd`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `y_out`, `x_in`, `u_vec`, or `v_vec` is empty,
    /// `x_in.len() != v_vec.len()`, `y_out.len() != u_vec.len()`, or `scale_factor` is non-finite.
    #[inline]
    fn eval_rank1_slices(
        &self,
        y_out: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        u_vec: &AlignedSlice32<'_, f32>,
        v_vec: &AlignedSlice32<'_, f32>,
        gain_vec: Option<&AlignedSlice32<'_, f32>>,
        scale_factor: f32,
    ) -> GcsoResult<()> {
        let d_in = x_in.len();
        let d_out = y_out.len();
        if d_in == 0
            || d_out == 0
            || v_vec.len() < d_in
            || u_vec.len() < d_out
            || !branchless_is_finite_f32(scale_factor)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(g) = gain_vec {
            if g.len() < d_out {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }

        let dot_vx = dot_f32_unrolled4(v_vec.as_slice(), x_in.as_slice(), d_in);
        if !branchless_is_finite_f32(dot_vx) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let eff_scale = branchless_flush_subnormal_f32(scale_factor * dot_vx);
        let g_slice_opt = gain_vec.map(|g| &g.as_slice()[..d_out]);
        srl_write_outer_product_simd(
            &mut y_out.as_mut_slice()[..d_out],
            &u_vec.as_slice()[..d_out],
            g_slice_opt,
            eff_scale,
        );
        Ok(())
    }

    /// Accumulates the fused Rank-1 outer product residual $\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
    /// in-place into `y_inout` from safe hardware-aligned slices using AVX2 (`srl_accum_outer_product_simd`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice is empty, lengths mismatch, or `scale_factor` is non-finite.
    #[inline]
    fn eval_rank1_slices_fused(
        &self,
        y_inout: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        u_vec: &AlignedSlice32<'_, f32>,
        v_vec: &AlignedSlice32<'_, f32>,
        gain_vec: Option<&AlignedSlice32<'_, f32>>,
        scale_factor: f32,
    ) -> GcsoResult<()> {
        let d_in = x_in.len();
        let d_out = y_inout.len();
        if d_in == 0
            || d_out == 0
            || v_vec.len() < d_in
            || u_vec.len() < d_out
            || !branchless_is_finite_f32(scale_factor)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(g) = gain_vec {
            if g.len() < d_out {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }

        let dot_vx = dot_f32_unrolled4(v_vec.as_slice(), x_in.as_slice(), d_in);
        if !branchless_is_finite_f32(dot_vx) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let eff_scale = branchless_flush_subnormal_f32(scale_factor * dot_vx);
        let g_slice_opt = gain_vec.map(|g| &g.as_slice()[..d_out]);
        srl_accum_outer_product_simd(
            &mut y_inout.as_mut_slice()[..d_out],
            &u_vec.as_slice()[..d_out],
            g_slice_opt,
            eff_scale,
        );
        Ok(())
    }

    /// Evaluates the standalone Rank-1 outer product $\mathbf{y} = s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
    /// from a C-ABI [`gcso_srl_descriptor_t`] (including optional 32B-aligned `gain_scalar_ptr`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `u_vector_ptr == 0` or `v_vector_ptr == 0`,
    /// [`GCSO_ERROR_MISALIGNED_POINTER`] if pointers are not 32B-aligned, or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if `desc.rank != 1`, `desc.scale_factor` is non-finite, or slices are empty.
    #[inline]
    fn eval_rank1(
        &self,
        y_out: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        desc: &gcso_srl_descriptor_t,
    ) -> GcsoResult<()> {
        let (u_slice, v_slice, g_slice_opt) =
            validate_and_unpack_srl_descriptor(desc, x_in.len(), y_out.len())?;
        self.eval_rank1_slices(
            y_out,
            x_in,
            &u_slice,
            &v_slice,
            g_slice_opt.as_ref(),
            desc.scale_factor,
        )
    }

    /// Accumulates the fused Rank-1 outer product residual $\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
    /// in-place into `y_inout` from a C-ABI [`gcso_srl_descriptor_t`] (including optional 32B-aligned `gain_scalar_ptr`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `u_vector_ptr == 0` or `v_vector_ptr == 0`,
    /// [`GCSO_ERROR_MISALIGNED_POINTER`] if pointers are not 32B-aligned, or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if `desc.rank != 1`, `desc.scale_factor` is non-finite, or slices are empty.
    #[inline]
    fn eval_rank1_fused(
        &self,
        y_inout: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        desc: &gcso_srl_descriptor_t,
    ) -> GcsoResult<()> {
        let (u_slice, v_slice, g_slice_opt) =
            validate_and_unpack_srl_descriptor(desc, x_in.len(), y_inout.len())?;
        self.eval_rank1_slices_fused(
            y_inout,
            x_in,
            &u_slice,
            &v_slice,
            g_slice_opt.as_ref(),
            desc.scale_factor,
        )
    }

    /// Accumulates the fused Rank-1 outer product residual gated by `tagged_ptr`'s `ACTION_MODE_SRL_INJECT` bit (`0x10`).
    ///
    /// # Errors
    /// Propagates any pointer/slice validation error from [`Self::eval_rank1_fused`].
    #[inline]
    fn eval_rank1_pointer_gated_fused(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        y_inout: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        desc: &gcso_srl_descriptor_t,
    ) -> GcsoResult<bool> {
        let (u_slice, v_slice, g_slice_opt) =
            validate_and_unpack_srl_descriptor(desc, x_in.len(), y_inout.len())?;
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_SRL_INJECT) {
            return Ok(false);
        }
        self.eval_rank1_slices_fused(
            y_inout,
            x_in,
            &u_slice,
            &v_slice,
            g_slice_opt.as_ref(),
            desc.scale_factor,
        )?;
        Ok(true)
    }
}

// ===================================================================
// 6. 2D/3D Spatiotemporal Visual Attention Steering (`VisualAttentionSteering`)
// ===================================================================

#[inline(always)]
#[must_use]
pub(super) fn patch_diff_sum_sq_unrolled4(cur_patch: &[f32], prv_patch: &[f32]) -> f32 {
    let n = cur_patch.len().min(prv_patch.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::{
            _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_sub_ps,
        };
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
pub(super) fn steer_visual_grid_bitmask_core<F>(
    grid_tensor: &mut AlignedSliceMut32<'_, f32>,
    active_patch_mask: &gcso_paged_bitmask_t,
    patch_stride_elems: usize,
    num_patches: usize,
    min_dim_multiple: usize,
    mut rotate_patch_fn: F,
) -> GcsoResult<u32>
where
    F: FnMut(&mut [f32]),
{
    if num_patches == 0
        || num_patches > (PAGED_BITMASK_BITS as usize)
        || patch_stride_elems == 0
        || (patch_stride_elems % min_dim_multiple) != 0
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let required_elems = num_patches
        .checked_mul(patch_stride_elems)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    if grid_tensor.len() < required_elems {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
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

/// Trait for 2D Spatial and 3D Spatiotemporal Visual Attention Phase Steering and
/// Bitmask-Gated Frame Delta Detection (**Micro / Multimodal Level**).
pub trait VisualAttentionSteering: BranchlessKernel {
    /// Applies 2D spatial RoPE phase steering $(\Delta\theta_x, \Delta\theta_y)$ across a visual
    /// patch tensor (`head_dim` must be a positive multiple of `4`) using AVX2 (`rotate_so2_2d_xy_slice_simd`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim == 0`, `(head_dim & 3) != 0`, `num_heads == 0`,
    /// or `patch_tensor.len() < head_dim * num_heads`.
    #[inline]
    fn steer_visual_patch_2d(
        &self,
        patch_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_x_q7: gcso_q7_t,
        phase_y_q7: gcso_q7_t,
        head_dim: usize,
        num_heads: usize,
    ) -> GcsoResult<()> {
        let expected_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if head_dim == 0
            || (head_dim & 3) != 0
            || num_heads == 0
            || patch_tensor.len() < expected_len
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (sin_x, cos_x) = branchless_sincos_q7(phase_x_q7);
        let (sin_y, cos_y) = branchless_sincos_q7(phase_y_q7);
        rotate_so2_2d_xy_slice_simd(
            &mut patch_tensor.as_mut_slice()[..expected_len],
            sin_x,
            cos_x,
            sin_y,
            cos_y,
        );
        Ok(())
    }

    /// Applies 3D spatiotemporal RoPE phase steering $(\Delta\theta_t, \Delta\theta_x, \Delta\theta_y)$
    /// across a video/3D patch tensor (`head_dim` must be a positive multiple of `6`) using AVX2
    /// (`rotate_so2_3d_txy_slice_simd`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim == 0`, `(head_dim % 6) != 0`, `num_heads == 0`,
    /// or `patch_tensor.len() < head_dim * num_heads`.
    #[inline]
    fn steer_visual_patch_3d(
        &self,
        patch_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_t_q7: gcso_q7_t,
        phase_x_q7: gcso_q7_t,
        phase_y_q7: gcso_q7_t,
        head_dim: usize,
        num_heads: usize,
    ) -> GcsoResult<()> {
        let expected_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if head_dim == 0
            || (head_dim % 6) != 0
            || num_heads == 0
            || patch_tensor.len() < expected_len
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (sin_t, cos_t) = branchless_sincos_q7(phase_t_q7);
        let (sin_x, cos_x) = branchless_sincos_q7(phase_x_q7);
        let (sin_y, cos_y) = branchless_sincos_q7(phase_y_q7);
        rotate_so2_3d_txy_slice_simd(
            &mut patch_tensor.as_mut_slice()[..expected_len],
            sin_t,
            cos_t,
            sin_x,
            cos_x,
            sin_y,
            cos_y,
        );
        Ok(())
    }

    /// Computes the 256-bit frame delta bitmask (`prev_mask ^ curr_mask`) and active changed patch
    /// count (`0..=256`) in $\mathcal{O}(1)$ branchless POPCNT arithmetic.
    #[inline(always)]
    #[must_use]
    fn detect_frame_delta_bitmask(
        &self,
        prev_mask: &gcso_paged_bitmask_t,
        curr_mask: &gcso_paged_bitmask_t,
    ) -> (gcso_paged_bitmask_t, u32) {
        let w0 = prev_mask.bits[0] ^ curr_mask.bits[0];
        let w1 = prev_mask.bits[1] ^ curr_mask.bits[1];
        let w2 = prev_mask.bits[2] ^ curr_mask.bits[2];
        let w3 = prev_mask.bits[3] ^ curr_mask.bits[3];
        let changed = w0.count_ones() + w1.count_ones() + w2.count_ones() + w3.count_ones();
        (
            gcso_paged_bitmask_t {
                bits: [w0, w1, w2, w3],
            },
            changed,
        )
    }

    /// Detects active visual patches between raw `f32` frame buffers `current_frame` and `prev_frame`
    /// whose mean squared difference exceeds `delta_threshold_sq`, populating a 256-bit
    /// [`gcso_paged_bitmask_t`] (`mask_out`) and returning the active patch count (`0..=256`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_patches == 0`, `num_patches > 256`,
    /// `patch_stride_elems == 0`, `delta_threshold_sq < 0.0`, or frame buffers are too small.
    #[inline]
    fn detect_frame_mse_bitmask(
        &self,
        current_frame: &AlignedSlice32<'_, f32>,
        prev_frame: &AlignedSlice32<'_, f32>,
        num_patches: usize,
        patch_stride_elems: usize,
        delta_threshold_sq: f32,
        mask_out: &mut gcso_paged_bitmask_t,
    ) -> GcsoResult<u32> {
        let total_elems = num_patches
            .checked_mul(patch_stride_elems)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if num_patches == 0
            || num_patches > (PAGED_BITMASK_BITS as usize)
            || patch_stride_elems == 0
            || delta_threshold_sq < 0.0
            || !branchless_is_finite_f32(delta_threshold_sq)
            || current_frame.len() < total_elems
            || prev_frame.len() < total_elems
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        *mask_out = gcso_paged_bitmask_t::default();
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
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_patches == 0`, `num_patches > 256`,
    /// `patch_stride_elems` is zero or not a multiple of `4`, or `grid_tensor` is too small.
    #[inline]
    fn steer_visual_grid_2d_bitmask_gated(
        &self,
        grid_tensor: &mut AlignedSliceMut32<'_, f32>,
        active_patch_mask: &gcso_paged_bitmask_t,
        patch_stride_elems: usize,
        num_patches: usize,
        phase_x_q7: gcso_q7_t,
        phase_y_q7: gcso_q7_t,
    ) -> GcsoResult<u32> {
        let (sin_x, cos_x) = branchless_sincos_q7(phase_x_q7);
        let (sin_y, cos_y) = branchless_sincos_q7(phase_y_q7);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY: `target_feature = "avx2"` is enabled; hoists `(v_cos, v_sin)` construction
            // once outside the 256-patch bitmask scan.
            let (v_cos, v_sin) = unsafe { avx2_build_so2_2d_xy_vecs(sin_x, cos_x, sin_y, cos_y) };
            return steer_visual_grid_bitmask_core(
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
            );
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
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_patches == 0`, `num_patches > 256`,
    /// `patch_stride_elems` is zero or not a multiple of `6`, or `grid_tensor` is too small.
    #[inline]
    fn steer_visual_grid_3d_bitmask_gated(
        &self,
        grid_tensor: &mut AlignedSliceMut32<'_, f32>,
        active_patch_mask: &gcso_paged_bitmask_t,
        patch_stride_elems: usize,
        num_patches: usize,
        phase_t_q7: gcso_q7_t,
        phase_x_q7: gcso_q7_t,
        phase_y_q7: gcso_q7_t,
    ) -> GcsoResult<u32> {
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
}

