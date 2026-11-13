// name: src/core/src/traits/rope/phase_steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PhaseSteering, EntropyGatedPhaseLookup, and DynamicLayerSkipper trait definitions.

use super::*;

// ===================================================================
// 1. Dynamic Phase-Shifted RoPE, RIPA, QDPS & Lazy Unwrapping (`PhaseSteering`)
// ===================================================================

/// Core trait for Dynamic Phase-Shifted RoPE (DPSR), Rotational Invariant Phase Anchors (RIPA),
/// Quantized Discrete Phase Stepping (QDPS), Seam-Free Circular $S^1$ Lazy Phase Unwrapping,
/// Single-Manifold Unified-Field Phase Steering, Attention $\times$ DPSR Phase-Interference Score
/// Modulation, Bayesian Pointer-Trail Attention Modulation, and Closed-Form Orthogonal Procrustes
/// Phase Estimation (**Nano / Micro Level**).
pub trait PhaseSteering: BranchlessKernel {
    /// Returns the configured attention head dimension ($d_{\mathrm{head}}$, positive even integer).
    fn head_dim(&self) -> u32;

    /// Returns the configured number of attention heads ($H > 0$).
    fn num_heads(&self) -> u32;

    /// Validates `(head_dim, num_heads)` invariants (`head_dim > 0`, `head_dim % 2 == 0`, `num_heads > 0`)
    /// and returns `(head_dim_usize, num_heads_usize, expected_tensor_len)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `num_heads` is zero,
    /// or `head_dim * num_heads` overflows `usize`.
    #[inline(always)]
    fn validate_head_layout(&self) -> GcsoResult<(usize, usize, usize)> {
        let head_dim = self.head_dim() as usize;
        let num_heads = self.num_heads() as usize;
        if head_dim == 0 || (head_dim & 1) != 0 || num_heads == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let expected_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        Ok((head_dim, num_heads, expected_len))
    }

    /// Computes the Lazy Phase Unwrapping relative phase differential
    /// $\Delta\theta_{\mathrm{effective}} = \Delta\theta_{\mathrm{query}} - \Delta\theta_{\mathrm{accum}}$
    /// across all configured attention heads using 8-lane SWAR saturating subtraction (Section 4.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice has fewer than `self.num_heads()` elements.
    #[inline(always)]
    fn compute_lazy_query_relative_phase_q7(
        &self,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        accumulated_context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        effective_phases_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if query_phases.len() < num_heads
            || accumulated_context_phases.len() < num_heads
            || effective_phases_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_sub_sat_q7_slices(
            query_phases.as_slice(),
            accumulated_context_phases.as_slice(),
            effective_phases_out.as_mut_slice(),
            num_heads,
        );
        Ok(())
    }

    /// Computes the **Seam-Free Circular Relative Phase Differential** on the circle group
    /// $S^1 \cong SO(2)$ using 8-lane SWAR modular subtraction (`swar_q7_circular_diff_x8`) (Section 4.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice has fewer than `self.num_heads()` elements.
    #[inline(always)]
    fn compute_lazy_query_circular_phase_q7(
        &self,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        accumulated_context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        effective_phases_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if query_phases.len() < num_heads
            || accumulated_context_phases.len() < num_heads
            || effective_phases_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_circular_diff_q7_slices(
            query_phases.as_slice(),
            accumulated_context_phases.as_slice(),
            effective_phases_out.as_mut_slice(),
            num_heads,
        );
        Ok(())
    }

    /// Computes the **Seam-Free $S^1$ Geodesic Relative Phase Differential** across all configured
    /// attention heads by combining 8-lane SWAR modular subtraction (`swar_q7_circular_diff_x8`)
    /// with 8-lane SWAR triangle-wave geodesic folding (`swar_q7_fold_s1_geodesic_x8` into `[-64, +64]`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice has fewer than `self.num_heads()` elements.
    #[inline(always)]
    fn compute_lazy_query_geodesic_phase_q7(
        &self,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        accumulated_context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        effective_phases_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if query_phases.len() < num_heads
            || accumulated_context_phases.len() < num_heads
            || effective_phases_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_circular_diff_and_fold_geodesic_q7_slices(
            query_phases.as_slice(),
            accumulated_context_phases.as_slice(),
            effective_phases_out.as_mut_slice(),
            num_heads,
        );
        Ok(())
    }

    /// Folds an existing slice of circular Q7 phases in-place onto the principal $S^1$ geodesic
    /// wave `[-64, +64]` ($\pm \pi/2$) using 8-lane SWAR (`swar_q7_fold_s1_geodesic_x8`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `phases.len() < self.num_heads()`.
    #[inline(always)]
    fn fold_circular_phases_s1_geodesic_q7(
        &self,
        phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if phases.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_fold_s1_geodesic_q7_slice_in_place(phases.as_mut_slice(), num_heads);
        Ok(())
    }

    /// Scales all configured attention head Q7 phase deltas in-place by `scale_u7` (`128 == 1.0`)
    /// using hoisted-bias 8-lane SWAR (`swar_q7_scale_le128_with_bias_x8`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `phases.len() < self.num_heads()`.
    #[inline(always)]
    fn scale_phases_swar_u7(
        &self,
        phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        scale_u7: u8,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if phases.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slice = phases.as_mut_slice();
        if scale_u7 == 128 {
            return Ok(());
        }
        if scale_u7 == 0 {
            slice[..num_heads].fill(0);
            return Ok(());
        }
        let gt_128 = scale_u7 > 128;
        let s_sub = u64::from(branchless_select_u8(
            gt_128,
            scale_u7.wrapping_sub(128),
            scale_u7,
        ));
        let bias_restore = (128u64.wrapping_sub(s_sub)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let extra_mask = (gt_128 as u64).wrapping_neg();

        let mut i = 0usize;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_adds_epi8, _mm256_and_si256, _mm256_loadu_si256,
                _mm256_mulhrs_epi16, _mm256_or_si256, _mm256_set1_epi16, _mm256_set1_epi8,
                _mm256_slli_epi16, _mm256_storeu_si256,
            };
            let v_s = _mm256_set1_epi16(s_sub as i16);
            let mask_ff00 = _mm256_set1_epi16(-256i16);
            let mask_00ff = _mm256_set1_epi16(0x00FFi16);
            let v_extra = _mm256_set1_epi8(-(gt_128 as i8));
            let ptr = slice.as_mut_ptr().cast::<__m256i>();
            while i + 32 <= num_heads {
                let off = i >> 5;
                let v = _mm256_loadu_si256(ptr.add(off));
                let ev = _mm256_mulhrs_epi16(_mm256_slli_epi16(v, 8), v_s);
                let od = _mm256_mulhrs_epi16(_mm256_and_si256(v, mask_ff00), v_s);
                let scaled = _mm256_or_si256(
                    _mm256_and_si256(ev, mask_00ff),
                    _mm256_slli_epi16(od, 8),
                );
                let out = _mm256_adds_epi8(scaled, _mm256_and_si256(v, v_extra));
                _mm256_storeu_si256(ptr.add(off), out);
                i += 32;
            }
        }
        while i + 16 <= num_heads {
            let w0 = load_q7_x8_slice(slice, i);
            let w1 = load_q7_x8_slice(slice, i + 8);
            let s0 = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(w0, s_sub, bias_restore),
                w0 & extra_mask,
            );
            let s1 = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(w1, s_sub, bias_restore),
                w1 & extra_mask,
            );
            store_q7_x8_slice(slice, i, s0);
            store_q7_x8_slice(slice, i + 8, s1);
            i += 16;
        }
        while i + 8 <= num_heads {
            let w = load_q7_x8_slice(slice, i);
            let s = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(w, s_sub, bias_restore),
                w & extra_mask,
            );
            store_q7_x8_slice(slice, i, s);
            i += 8;
        }
        while i < num_heads {
            slice[i] = branchless_q7_scale_u7(slice[i], scale_u7);
            i += 1;
        }
        Ok(())
    }

    /// Accumulates a per-step Q7 phase delta vector `step_deltas` into `accumulated_context_phases`
    /// on the circle group $S^1 \cong \mathbb{Z}/256\mathbb{Z}$ with 8-lane SWAR QDPS step filtering.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `min_step_rad < 0.0`, `min_step_rad` is non-finite,
    /// or slice lengths are smaller than `self.num_heads()`.
    #[inline]
    fn accumulate_context_phases_circular_q7(
        &self,
        accumulated_context_phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        step_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        min_step_rad: f32,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        let bits = min_step_rad.to_bits();
        let abs_bits = bits & 0x7FFF_FFFF;
        if ((bits >> 31) != 0 && abs_bits != 0)
            || (bits & 0x7F80_0000) == 0x7F80_0000
            || accumulated_context_phases.len() < num_heads
            || step_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let step_u8 = if abs_bits <= Q7_PHASE_SCALE_RAD.to_bits() {
            (abs_bits != 0) as u8
        } else {
            branchless_rad_to_qdps_step_u8(min_step_rad)
        };
        if step_u8 > 127 {
            return Ok(());
        }
        let acc_slice = accumulated_context_phases.as_mut_slice();
        let d_slice = step_deltas.as_slice();

        let mut i = 0usize;
        if step_u8 <= 1 {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                use core::arch::x86_64::{
                    __m256i, _mm256_add_epi8, _mm256_loadu_si256, _mm256_storeu_si256,
                };
                let acc_ptr = acc_slice.as_mut_ptr().cast::<__m256i>();
                let d_ptr = d_slice.as_ptr().cast::<__m256i>();
                if num_heads == MAX_ATTENTION_HEADS {
                    let va0 = _mm256_loadu_si256(acc_ptr);
                    let vd0 = _mm256_loadu_si256(d_ptr);
                    let va1 = _mm256_loadu_si256(acc_ptr.add(1));
                    let vd1 = _mm256_loadu_si256(d_ptr.add(1));
                    _mm256_storeu_si256(acc_ptr, _mm256_add_epi8(va0, vd0));
                    _mm256_storeu_si256(acc_ptr.add(1), _mm256_add_epi8(va1, vd1));
                    return Ok(());
                }
                if num_heads == 32 {
                    let va0 = _mm256_loadu_si256(acc_ptr);
                    let vd0 = _mm256_loadu_si256(d_ptr);
                    _mm256_storeu_si256(acc_ptr, _mm256_add_epi8(va0, vd0));
                    return Ok(());
                }
                while i + 32 <= num_heads {
                    let off = i >> 5;
                    let va = _mm256_loadu_si256(acc_ptr.add(off));
                    let vd = _mm256_loadu_si256(d_ptr.add(off));
                    _mm256_storeu_si256(acc_ptr.add(off), _mm256_add_epi8(va, vd));
                    i += 32;
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            while i + 32 <= num_heads {
                let a0 = load_q7_x8_slice(acc_slice, i);
                let a1 = load_q7_x8_slice(acc_slice, i + 8);
                let a2 = load_q7_x8_slice(acc_slice, i + 16);
                let a3 = load_q7_x8_slice(acc_slice, i + 24);
                let d0 = load_q7_x8_slice(d_slice, i);
                let d1 = load_q7_x8_slice(d_slice, i + 8);
                let d2 = load_q7_x8_slice(d_slice, i + 16);
                let d3 = load_q7_x8_slice(d_slice, i + 24);
                store_q7_x8_slice(acc_slice, i, swar_q7_circular_add_x8(a0, d0));
                store_q7_x8_slice(acc_slice, i + 8, swar_q7_circular_add_x8(a1, d1));
                store_q7_x8_slice(acc_slice, i + 16, swar_q7_circular_add_x8(a2, d2));
                store_q7_x8_slice(acc_slice, i + 24, swar_q7_circular_add_x8(a3, d3));
                i += 32;
            }
            while i + 8 <= num_heads {
                let acc_w = load_q7_x8_slice(acc_slice, i);
                let raw_d = load_q7_x8_slice(d_slice, i);
                store_q7_x8_slice(acc_slice, i, swar_q7_circular_add_x8(acc_w, raw_d));
                i += 8;
            }
        } else {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                use core::arch::x86_64::{
                    __m256i, _mm256_abs_epi8, _mm256_add_epi8, _mm256_and_si256,
                    _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_max_epu8, _mm256_set1_epi8,
                    _mm256_storeu_si256,
                };
                let acc_ptr = acc_slice.as_mut_ptr().cast::<__m256i>();
                let d_ptr = d_slice.as_ptr().cast::<__m256i>();
                let v_thresh = _mm256_set1_epi8(step_u8 as i8);
                if num_heads == MAX_ATTENTION_HEADS {
                    let va0 = _mm256_loadu_si256(acc_ptr);
                    let vd0 = _mm256_loadu_si256(d_ptr);
                    let va1 = _mm256_loadu_si256(acc_ptr.add(1));
                    let vd1 = _mm256_loadu_si256(d_ptr.add(1));
                    let vabs0 = _mm256_abs_epi8(vd0);
                    let vabs1 = _mm256_abs_epi8(vd1);
                    let keep0 = _mm256_cmpeq_epi8(_mm256_max_epu8(vabs0, v_thresh), vabs0);
                    let keep1 = _mm256_cmpeq_epi8(_mm256_max_epu8(vabs1, v_thresh), vabs1);
                    _mm256_storeu_si256(
                        acc_ptr,
                        _mm256_add_epi8(va0, _mm256_and_si256(vd0, keep0)),
                    );
                    _mm256_storeu_si256(
                        acc_ptr.add(1),
                        _mm256_add_epi8(va1, _mm256_and_si256(vd1, keep1)),
                    );
                    return Ok(());
                }
                if num_heads == 32 {
                    let va0 = _mm256_loadu_si256(acc_ptr);
                    let vd0 = _mm256_loadu_si256(d_ptr);
                    let vabs0 = _mm256_abs_epi8(vd0);
                    let keep0 = _mm256_cmpeq_epi8(_mm256_max_epu8(vabs0, v_thresh), vabs0);
                    _mm256_storeu_si256(
                        acc_ptr,
                        _mm256_add_epi8(va0, _mm256_and_si256(vd0, keep0)),
                    );
                    return Ok(());
                }
                while i + 32 <= num_heads {
                    let off = i >> 5;
                    let va = _mm256_loadu_si256(acc_ptr.add(off));
                    let vd = _mm256_loadu_si256(d_ptr.add(off));
                    let vabs = _mm256_abs_epi8(vd);
                    let keep = _mm256_cmpeq_epi8(_mm256_max_epu8(vabs, v_thresh), vabs);
                    let vfilt = _mm256_and_si256(vd, keep);
                    _mm256_storeu_si256(acc_ptr.add(off), _mm256_add_epi8(va, vfilt));
                    i += 32;
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            while i + 32 <= num_heads {
                let a0 = load_q7_x8_slice(acc_slice, i);
                let a1 = load_q7_x8_slice(acc_slice, i + 8);
                let a2 = load_q7_x8_slice(acc_slice, i + 16);
                let a3 = load_q7_x8_slice(acc_slice, i + 24);
                let d0 = swar_q7_qdps_filter_x8(load_q7_x8_slice(d_slice, i), step_u8);
                let d1 = swar_q7_qdps_filter_x8(load_q7_x8_slice(d_slice, i + 8), step_u8);
                let d2 = swar_q7_qdps_filter_x8(load_q7_x8_slice(d_slice, i + 16), step_u8);
                let d3 = swar_q7_qdps_filter_x8(load_q7_x8_slice(d_slice, i + 24), step_u8);
                store_q7_x8_slice(acc_slice, i, swar_q7_circular_add_x8(a0, d0));
                store_q7_x8_slice(acc_slice, i + 8, swar_q7_circular_add_x8(a1, d1));
                store_q7_x8_slice(acc_slice, i + 16, swar_q7_circular_add_x8(a2, d2));
                store_q7_x8_slice(acc_slice, i + 24, swar_q7_circular_add_x8(a3, d3));
                i += 32;
            }
            while i + 8 <= num_heads {
                let acc_w = load_q7_x8_slice(acc_slice, i);
                let raw_d = load_q7_x8_slice(d_slice, i);
                let filt_d = swar_q7_qdps_filter_x8(raw_d, step_u8);
                store_q7_x8_slice(acc_slice, i, swar_q7_circular_add_x8(acc_w, filt_d));
                i += 8;
            }
        }
        let min_step_q7 = i32::from(step_u8);
        while i < num_heads {
            let filt_d = branchless_qdps_filter_q7(d_slice[i], min_step_q7);
            acc_slice[i] = acc_slice[i].wrapping_add(filt_d);
            i += 1;
        }
        Ok(())
    }

    /// Executes the complete **Seam-Free Lazy Phase Unwrapping** hot-path pipeline (Section 4.1)
    /// in a **single fused 8-lane SWAR pass** (`q - ctx` + optional $S^1$ geodesic fold + QDPS
    /// step filter) without multi-pass memory reloads of `effective_phases`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `min_step_rad < 0.0`, `ripa_clamp_max_rad` is non-positive/non-finite, or tensor/phase bounds fail.
    #[inline]
    fn apply_seam_free_lazy_unwrap_q7(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        accumulated_context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        min_step_rad: f32,
        ripa_clamp_max_rad: Option<f32>,
    ) -> GcsoResult<AlignedBuffer16<gcso_q7_t, MAX_ATTENTION_HEADS>> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || query_tensor.len() < expected_len
            || query_phases.len() < num_heads
            || accumulated_context_phases.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref k) = key_tensor {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            query_tensor.ensure_disjoint_mut(k)?;
        }
        if let Some(max_rad) = ripa_clamp_max_rad {
            if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }

        let fold_geodesic = ripa_clamp_max_rad.is_some();
        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let q_p = &query_phases.as_slice()[..num_heads];
        let c_p = &accumulated_context_phases.as_slice()[..num_heads];
        let mut eff_words = [0u64; NUM_SWARM_CELLS];
        let mut h = 0usize;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if num_heads >= 32 {
            // SAFETY: `q_p` and `c_p` have length `num_heads >= 32`, and AVX2 is enabled.
            // Extracts `vo` directly into 64-bit GPR words (`eff_words[cell_base..cell_base + 4]`)
            // via `vextracti128` + `vmovq` + `vpextrq` to eliminate the 256-bit-store-to-64-bit-load
            // Store-to-Load Forwarding (STLF) stall in `store_q7_words_64` and `steer_from_swar_words_core_safe`.
            unsafe {
                use core::arch::x86_64::{
                    __m256i, _mm256_abs_epi8, _mm256_add_epi8, _mm256_and_si256,
                    _mm256_blendv_epi8, _mm256_castsi256_si128, _mm256_cmpeq_epi8,
                    _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_max_epu8,
                    _mm256_set1_epi8, _mm256_setzero_si256, _mm256_sub_epi8, _mm256_xor_si256,
                    _mm_cvtsi128_si64, _mm_extract_epi64,
                };
                let qp = q_p.as_ptr().cast::<__m256i>();
                let cp = c_p.as_ptr().cast::<__m256i>();
                let v_128 = _mm256_set1_epi8(-128);
                let v_thresh = _mm256_set1_epi8(step_u8 as i8);
                while h + 32 <= num_heads {
                    let off = h >> 5;
                    let vd = _mm256_sub_epi8(
                        _mm256_loadu_si256(qp.add(off)),
                        _mm256_loadu_si256(cp.add(off)),
                    );
                    let vf = if fold_geodesic {
                        let m = _mm256_xor_si256(vd, _mm256_add_epi8(vd, vd));
                        _mm256_blendv_epi8(vd, _mm256_sub_epi8(v_128, vd), m)
                    } else {
                        vd
                    };
                    let vo = if step_u8 <= 1 {
                        vf
                    } else if step_u8 > 127 {
                        _mm256_setzero_si256()
                    } else {
                        let vabs = _mm256_abs_epi8(vf);
                        let keep = _mm256_cmpeq_epi8(_mm256_max_epu8(vabs, v_thresh), vabs);
                        _mm256_and_si256(vf, keep)
                    };
                    let lo128 = _mm256_castsi256_si128(vo);
                    let hi128 = _mm256_extracti128_si256(vo, 1);
                    let cell_base = off << 2;
                    eff_words[cell_base] = _mm_cvtsi128_si64(lo128) as u64;
                    eff_words[cell_base + 1] = _mm_extract_epi64(lo128, 1) as u64;
                    eff_words[cell_base + 2] = _mm_cvtsi128_si64(hi128) as u64;
                    eff_words[cell_base + 3] = _mm_extract_epi64(hi128, 1) as u64;
                    h += 32;
                }
            }
        }
        let mut prev_diff_w = 0u64;
        let mut prev_filt_w = 0u64;
        let mut has_prev_w = false;

        while h + 8 <= num_heads {
            let diff_w =
                swar_q7_circular_diff_x8(load_q7_x8_slice(q_p, h), load_q7_x8_slice(c_p, h));
            let filt_w = if has_prev_w && (diff_w == prev_diff_w) {
                prev_filt_w
            } else {
                let folded_w = if fold_geodesic {
                    swar_q7_fold_s1_geodesic_x8(diff_w)
                } else {
                    diff_w
                };
                let fw = if step_u8 <= 1 {
                    folded_w
                } else if step_u8 > 127 {
                    0
                } else {
                    swar_q7_qdps_filter_x8(folded_w, step_u8)
                };
                prev_diff_w = diff_w;
                prev_filt_w = fw;
                has_prev_w = true;
                fw
            };
            eff_words[h >> 3] = filt_w;
            h += 8;
        }
        let min_step_q7 = i32::from(step_u8);
        while h < num_heads {
            let diff = branchless_q7_circular_diff(q_p[h], c_p[h]);
            let folded = if fold_geodesic {
                branchless_q7_fold_s1_geodesic(diff)
            } else {
                diff
            };
            let filt_b = branchless_qdps_filter_q7(folded, min_step_q7);
            eff_words[h >> 3] |= u64::from(filt_b as u8) << ((h & 7) << 3);
            h += 1;
        }
        let mut effective_phases =
            AlignedBuffer16::<gcso_q7_t, MAX_ATTENTION_HEADS>::new([0i8; MAX_ATTENTION_HEADS]);
        store_q7_words_64(&mut effective_phases.data, &eff_words);
        if let Some(max_rad) = ripa_clamp_max_rad {
            steer_from_swar_words_core_safe(
                &eff_words,
                0xFF,
                128,
                head_dim,
                num_heads,
                false,
                max_rad,
                query_tensor,
                key_tensor,
            );
        } else {
            let eff_slice = effective_phases.as_aligned_slice();
            if let Some(k) = key_tensor {
                self.apply_phase_steering_fused(query_tensor, k, &eff_slice)?;
            } else {
                self.apply_phase_steering(query_tensor, &eff_slice)?;
            }
        }
        Ok(effective_phases)
    }

    /// Applies Dynamic Phase-Shifted RoPE (DPSR) in-place over an arbitrary hardware-aligned
    /// tensor slice (`ALIGN` $\ge 32$, power of two).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32` or `ALIGN` is not a power of two,
    /// or [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_aligned<const ALIGN: usize>(
        &self,
        tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_phase_steering_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies Dynamic Phase-Shifted RoPE (DPSR) in-place to a single 32B-aligned Query or Key tensor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `tensor.len() < head_dim * num_heads` or `phase_deltas.len() < num_heads`.
    #[inline(always)]
    fn apply_phase_steering(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_aligned::<32>(tensor, phase_deltas)
    }

    /// Applies fused in-place phase steering to both Query and Key tensors simultaneously in lockstep.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` overlap (`GCSO_RESTRICT`)
    /// or have insufficient length.
    #[inline(always)]
    fn apply_phase_steering_fused(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_fused_aligned::<32>(query_tensor, key_tensor, phase_deltas)
    }

    /// Applies fused in-place phase steering to both Query and Key tensors over any alignment `ALIGN >= 32`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32` or not a power of two, or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if slices alias or are smaller than `head_dim * num_heads`.
    #[inline(always)]
    fn apply_phase_steering_fused_aligned<const ALIGN: usize>(
        &self,
        query_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        key_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_phase_steering_pair_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies fused in-place phase steering to a pre-verified [`DisjointMutPair`] of `(Q, K)` tensors.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor or phase slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_disjoint_pair<const ALIGN: usize>(
        &self,
        qk_pair: &mut DisjointMutPair<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        let (q_slice, k_slice) = qk_pair.as_mut_pair();
        if q_slice.len() < expected_len
            || k_slice.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_phase_steering_pair_slice_core(
            q_slice.as_mut_slice(),
            k_slice.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies 3-way fused in-place phase steering to Query, Key, and Value tensors simultaneously
    /// in lockstep (`rotate_so2_triple_slice_unrolled16`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor`, `key_tensor`, and `value_tensor`
    /// overlap or have lengths smaller than `head_dim * num_heads`.
    #[inline(always)]
    fn apply_phase_steering_fused_qkv(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        value_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        query_tensor.ensure_disjoint_mut(value_tensor)?;
        key_tensor.ensure_disjoint_mut(value_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || value_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_phase_steering_triple_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            value_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies 3-way fused in-place phase steering to a pre-verified [`DisjointMutTriple`] of `(Q, K, V)`
    /// using consecutive uniform-phase head-run coalescing and lockstep 16-float cacheline strides
    /// (`rotate_so2_triple_slice_unrolled16`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32`, or [`GCSO_ERROR_INVALID_ARGUMENT`]
    /// if tensor or phase lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_disjoint_triple<const ALIGN: usize>(
        &self,
        qkv_triple: &mut DisjointMutTriple<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        let (q_slice_ref, k_slice_ref, v_slice_ref) = qkv_triple.as_mut_triple();
        if q_slice_ref.len() < expected_len
            || k_slice_ref.len() < expected_len
            || v_slice_ref.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        apply_phase_steering_triple_slice_core(
            q_slice_ref.as_mut_slice(),
            k_slice_ref.as_mut_slice(),
            v_slice_ref.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies RIPA soft-bounded phase steering ($\Delta\theta_{\mathrm{safe}} = \theta_{\max} \cdot \tanh(\Delta\theta / \theta_{\max})$)
    /// to the upper low-frequency quarter (`rot_dim = ((head_dim >> 2) & !1).max(2)`) of an arbitrary
    /// hardware-aligned tensor (`ALIGN >= 32`, Section 2.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32`, or [`GCSO_ERROR_INVALID_ARGUMENT`]
    /// if `max_rad <= 0.0`, `max_rad` is non-finite, or slice bounds fail.
    #[inline(always)]
    fn apply_phase_steering_safe_aligned<const ALIGN: usize>(
        &self,
        tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_ripa_single_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            false,
            max_rad,
        );
        Ok(())
    }

    /// Resolves the **Swarm- & Blanket-Driven Autonomous Frequency Aperture** (`rot_dim`, even
    /// integer in `2..=head_dim`) for this engine's `head_dim()` without manual hyperparameters.
    #[inline(always)]
    #[must_use]
    fn resolve_autonomous_rot_dim(
        &self,
        provenance_bits: u32,
        quorum_mask: u8,
        moving_z_entropy: f32,
        von_mises_kappa: f32,
    ) -> usize {
        resolve_autonomous_swarm_rot_dim(
            self.head_dim() as usize,
            provenance_bits,
            quorum_mask,
            moving_z_entropy,
            von_mises_kappa,
        )
    }

    /// Applies **Swarm- & Blanket-Driven Autonomous Frequency Aperture Phase Steering**, dynamically
    /// selecting the upper low-frequency rotation window `rot_dim` ($\frac{1}{8}d_{\mathrm{head}}$,
    /// $\frac{1}{4}d_{\mathrm{head}}$, $\frac{3}{8}d_{\mathrm{head}}$, or $\frac{1}{2}d_{\mathrm{head}}$)
    /// from `provenance_bits`, `quorum_mask`, `moving_z_entropy`, and `von_mises_kappa`, and rotating
    /// `[base + head_dim - rot_dim .. base + head_dim]` of each head with RIPA soft clamping.
    ///
    /// Returns the resolved `rot_dim`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `max_rad` is non-finite,
    /// `query_tensor` and `key_tensor` alias, or slice lengths are insufficient.
    #[inline]
    fn apply_autonomous_swarm_phase_steering(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        mut key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
        provenance_bits: u32,
        quorum_mask: u8,
        moving_z_entropy: f32,
        von_mises_kappa: f32,
    ) -> GcsoResult<usize> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref mut k_ref) = key_tensor {
            query_tensor.ensure_disjoint_mut(k_ref)?;
            if k_ref.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }
        let rot_dim = resolve_autonomous_swarm_rot_dim(
            head_dim,
            provenance_bits,
            quorum_mask,
            moving_z_entropy,
            von_mises_kappa,
        );
        let upper_off = head_dim - rot_dim;
        let (is_default, eff_max, q7_to_norm_scale) = resolve_ripa_norm_params(max_rad);
        let p_slice = &phase_deltas.as_slice()[..num_heads];
        let q_slice = &mut query_tensor.as_mut_slice()[..expected_len];

        let mut h = 0usize;
        if let Some(k_ref) = key_tensor {
            let k_slice = &mut k_ref.as_mut_slice()[..expected_len];
            while h < num_heads {
                let raw_q7 = p_slice[h];
                if raw_q7 != 0 {
                    let (sin_t, cos_t) = eval_ripa_sincos_with_params(
                        raw_q7,
                        true,
                        is_default,
                        eff_max,
                        q7_to_norm_scale,
                    );
                    let base = h * head_dim + upper_off;
                    rotate_ripa_upper_slice(&mut q_slice[base..base + rot_dim], sin_t, cos_t);
                    rotate_ripa_upper_slice(&mut k_slice[base..base + rot_dim], sin_t, cos_t);
                }
                h += 1;
            }
        } else {
            while h < num_heads {
                let raw_q7 = p_slice[h];
                if raw_q7 != 0 {
                    let (sin_t, cos_t) = eval_ripa_sincos_with_params(
                        raw_q7,
                        true,
                        is_default,
                        eff_max,
                        q7_to_norm_scale,
                    );
                    let base = h * head_dim + upper_off;
                    rotate_ripa_upper_slice(&mut q_slice[base..base + rot_dim], sin_t, cos_t);
                }
                h += 1;
            }
        }
        Ok(rot_dim)
    }

    /// Applies RIPA soft-bounded phase steering to the upper $d_{\mathrm{head}}/4$ low-frequency
    /// channels of a single 32B-aligned tensor (Section 2.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `max_rad` is non-finite,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_safe(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_safe_aligned::<32>(tensor, phase_deltas, max_rad)
    }

    /// Applies **Seam-Free $S^1$ Geodesic-Folded RIPA Phase Steering** to the upper $d_{\mathrm{head}}/4$
    /// low-frequency channels of a single 32B-aligned tensor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `max_rad` is non-finite,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_geodesic_safe(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_ripa_single_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            true,
            max_rad,
        );
        Ok(())
    }

    /// Applies **Seam-Free $S^1$ Geodesic-Folded RIPA Phase Steering** to the upper $d_{\mathrm{head}}/4$
    /// low-frequency channels of both Query and Key tensors simultaneously in lockstep.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias, `max_rad <= 0.0`,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_geodesic_fused_safe(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_ripa_pair_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            true,
            max_rad,
        );
        Ok(())
    }

    /// Applies fused RIPA soft-bounded phase steering to the upper $d_{\mathrm{head}}/4$ low-frequency
    /// channels of both Query and Key tensors simultaneously.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias, `max_rad <= 0.0`,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_fused_safe(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_fused_safe_aligned::<32>(
            query_tensor,
            key_tensor,
            phase_deltas,
            max_rad,
        )
    }

    /// Applies fused RIPA soft-bounded phase steering to both Query and Key tensors over any alignment `ALIGN >= 32`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32`, or [`GCSO_ERROR_INVALID_ARGUMENT`]
    /// if slices alias, `max_rad <= 0.0`, or lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_fused_safe_aligned<const ALIGN: usize>(
        &self,
        query_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        key_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_ripa_pair_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            false,
            max_rad,
        );
        Ok(())
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
    /// **Single Unified Attractor Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) starting from
    /// `trail_phases` with initial weight `trail_init_weight`, keeping the 8 folded `u64` SWAR
    /// words directly in registers (eliminating intermediate stack buffer store/reload), filters
    /// via 8-lane SWAR QDPS (`min_step_rad`), and steers `query_tensor` (and optional `key_tensor`).
    ///
    /// Returns `(total_resonant_pull, mean_coherence, dominant_anchor_idx)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `anchor_phases.len() != anchor_base_energies.len()`, `min_step_rad < 0.0`, `max_rad <= 0.0`,
    /// or tensor bounds fail.
    #[inline]
    fn apply_unified_field_phase_steering_with_inertia_safe(
        &self,
        trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        trail_init_weight: f32,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, u32)> {
        if anchor_phases.len() != anchor_base_energies.len()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS || query_tensor.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref k) = key_tensor {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            query_tensor.ensure_disjoint_mut(k)?;
        }

        let trail_words = load_q7_words_64(trail_phases);
        let (raw_words, total_pull, mean_coh, dom_idx, _act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &trail_words,
                trail_init_weight,
                anchor_phases,
                anchor_base_energies,
            );

        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let swar_words = swar_q7_qdps_filter_words_64(&raw_words, step_u8);

        steer_from_swar_words_core_safe(
            &swar_words,
            0xFF,
            128,
            head_dim,
            num_heads,
            false,
            max_rad,
            query_tensor,
            key_tensor,
        );
        Ok((total_pull, mean_coh, dom_idx))
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
    /// **Single Unified Attractor Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) with zero
    /// initial inertia weight (`0.0`) and steers `query_tensor` (and optional `key_tensor`).
    ///
    /// Returns `(total_resonant_pull, mean_coherence, dominant_anchor_idx)`.
    ///
    /// # Errors
    /// Propagates any validation error from [`Self::apply_unified_field_phase_steering_with_inertia_safe`].
    #[inline(always)]
    fn apply_unified_field_phase_steering_safe(
        &self,
        trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, u32)> {
        self.apply_unified_field_phase_steering_with_inertia_safe(
            trail_phases,
            0.0,
            anchor_phases,
            anchor_base_energies,
            min_step_rad,
            max_rad,
            query_tensor,
            key_tensor,
        )
    }

    /// Applies RIPA soft-bounded phase steering gated by `tagged_ptr`'s `ACTION_MODE_DPSR_STEER` bit.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `max_rad <= 0.0` or slice lengths are insufficient.
    #[inline]
    fn apply_pointer_gated_phase_steering_safe(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<bool> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
            || tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_DPSR_STEER) {
            return Ok(false);
        }
        apply_ripa_single_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            false,
            max_rad,
        );
        Ok(true)
    }

    /// Applies fused RIPA soft-bounded phase steering to both Query and Key tensors gated by `tagged_ptr`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias, `max_rad <= 0.0`,
    /// or slice lengths are insufficient.
    #[inline]
    fn apply_pointer_gated_phase_steering_fused_safe(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<bool> {
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
            || query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_DPSR_STEER) {
            return Ok(false);
        }
        apply_ripa_pair_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            false,
            max_rad,
        );
        Ok(true)
    }

    /// Indexes directly into an 8-slot Attractor Phase Bank (`slot_phase_bank`, `>= 512` bytes)
    /// using `tagged_ptr`'s 3-bit Target Attractor Slot ID (`0..=7`) and steers `tensor` with RIPA clamping.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > 64`, `slot_phase_bank.len() < 512`,
    /// `max_rad <= 0.0`, or `tensor` is too small.
    #[inline]
    fn apply_slot_bank_phase_steering_by_ptr(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<bool> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slot_id = usize::from(tagged_ptr.unpack_tag() >> 5);
        let slot_slice = slot_phase_bank.slot_aligned::<MAX_ATTENTION_HEADS>(slot_id)?;
        self.apply_pointer_gated_phase_steering_safe(tagged_ptr, tensor, &slot_slice, max_rad)
    }

    /// Indexes directly into an 8-slot Attractor Phase Bank (`slot_phase_bank`, `>= 512` bytes)
    /// using `tagged_ptr` and steers both `query_tensor` and `key_tensor` in lockstep.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > 64`, `slot_phase_bank.len() < 512`,
    /// `max_rad <= 0.0`, or `query_tensor` and `key_tensor` alias or are too small.
    #[inline]
    fn apply_slot_bank_phase_steering_fused_by_ptr(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<bool> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slot_id = usize::from(tagged_ptr.unpack_tag() >> 5);
        let slot_slice = slot_phase_bank.slot_aligned::<MAX_ATTENTION_HEADS>(slot_id)?;
        self.apply_pointer_gated_phase_steering_fused_safe(
            tagged_ptr,
            query_tensor,
            key_tensor,
            &slot_slice,
            max_rad,
        )
    }

    /// Applies **8-Cell Micro-Swarm Quorum-Gated RIPA Phase Steering** directly from a 64-head Q7
    /// phase array (`phase_deltas`), rotating only the resolved 8-head cell groups (`swarm_quorum_mask`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > 64`, `max_rad <= 0.0`, or
    /// `tensor.len() < head_dim * num_heads`.
    #[inline]
    fn apply_swarm_quorum_gated_steering_safe(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        swarm_quorum_mask: u8,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS || tensor.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if swarm_quorum_mask == 0 {
            return Ok(());
        }
        let swar_words = load_q7_words_64(phase_deltas);
        steer_from_swar_words_core_safe(
            &swar_words,
            swarm_quorum_mask,
            0,
            head_dim,
            num_heads,
            true,
            max_rad,
            tensor,
            None,
        );
        Ok(())
    }

    /// Applies pointer-gated **8-Cell Micro-Swarm Quorum-Gated RIPA Phase Steering** with
    /// configurable `unresolved_damping_u7` directly from 8 register-resident `u64` SWAR words.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > 64`, `max_rad <= 0.0`,
    /// `phase_deltas.len() < num_heads`, or tensor lengths/non-aliasing invariants fail.
    #[inline]
    fn apply_pointer_swarm_quorum_gated_steering_safe(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        swarm_quorum_mask: u8,
        unresolved_damping_u7: u8,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS
            || query_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref k) = key_tensor {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            query_tensor.ensure_disjoint_mut(k)?;
        }
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_DPSR_STEER) {
            return Ok(());
        }

        let clean_damping =
            branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
        let swar_words = if let Ok(arr64) = phase_deltas.as_array::<MAX_ATTENTION_HEADS>() {
            load_q7_words_64(arr64)
        } else {
            let p_slice = phase_deltas.as_slice();
            let mut words = [0u64; NUM_SWARM_CELLS];
            let full_cells = num_heads / HEADS_PER_CELL;
            let mut c = 0usize;
            while c < full_cells {
                words[c] = load_q7_x8_slice(p_slice, c * HEADS_PER_CELL);
                c += 1;
            }
            let rem_start = full_cells * HEADS_PER_CELL;
            if rem_start < num_heads {
                let mut tail_w = 0u64;
                let mut h = rem_start;
                while h < num_heads {
                    tail_w |= u64::from(p_slice[h] as u8) << ((h - rem_start) * 8);
                    h += 1;
                }
                words[full_cells] = tail_w;
            }
            words
        };

        steer_from_swar_words_core_safe(
            &swar_words,
            swarm_quorum_mask,
            clean_damping,
            head_dim,
            num_heads,
            true,
            max_rad,
            query_tensor,
            key_tensor,
        );
        Ok(())
    }

    /// Interpolates two 64-head Q7 phase profiles (`phases_a`, `phases_b`) along the shortest-arc
    /// $S^1$ geodesic using hoisted-bias 8-word SWAR (`swar_slerp_words_with_bias_64`, fast-pathing
    /// endpoints and word-uniform profiles), applies register-fused QDPS step filtering
    /// (`min_step_rad`), and steers `query_tensor` (and optional `key_tensor`) via
    /// `steer_from_swar_words_core_safe`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `phases_a.len() != 64`, `phases_b.len() != 64`, `min_step_rad < 0.0`, `max_rad <= 0.0`,
    /// or tensor lengths/non-aliasing invariants fail.
    #[inline]
    fn slerp_and_apply_phase_steering_safe(
        &self,
        phases_a: &AlignedSlice16<'_, gcso_q7_t>,
        phases_b: &AlignedSlice16<'_, gcso_q7_t>,
        t_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<AlignedBuffer16<gcso_q7_t, MAX_ATTENTION_HEADS>> {
        if min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS || query_tensor.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref k) = key_tensor {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            query_tensor.ensure_disjoint_mut(k)?;
        }

        let a_arr = phases_a.as_array::<MAX_ATTENTION_HEADS>()?;
        let b_arr = phases_b.as_array::<MAX_ATTENTION_HEADS>()?;
        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let s = u64::from(branchless_select_u8(t_u7 > 128, 128, t_u7));
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);

        let wa = load_q7_words_64(a_arr);
        let wb = load_q7_words_64(b_arr);
        let swar_words = if s == 0 {
            swar_q7_qdps_filter_words_64(&wa, step_u8)
        } else if s >= 128 {
            swar_q7_qdps_filter_words_64(&wb, step_u8)
        } else {
            let a_uni = is_word_array_uniform_64(&wa);
            let b_uni = if a_uni {
                is_word_array_uniform_64(&wb)
            } else {
                false
            };
            if a_uni & b_uni {
                let w0 = crate::traits::branchless::swar_slerp_word_unchecked_x8(
                    wa[0],
                    wb[0],
                    s,
                    bias_restore,
                );
                let fw0 = if step_u8 <= 1 {
                    w0
                } else if step_u8 > 127 {
                    0
                } else {
                    swar_q7_qdps_filter_x8(w0, step_u8)
                };
                [fw0; NUM_SWARM_CELLS]
            } else {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    // SAFETY: `target_feature = "avx2"` is enabled; fuses 64-head geodesic SLERP and QDPS filter in 2 YMM registers.
                    unsafe {
                        let (va0, va1) =
                            crate::traits::branchless::avx2_words64_to_vec64_with_uniform(&wa, a_uni);
                        let (vb0, vb1) = crate::traits::branchless::avx2_words64_to_vec64(&wb);
                        let (vr0, vr1) =
                            crate::traits::branchless::avx2_slerp_vec64(va0, va1, vb0, vb1, s);
                        let (vf0, vf1) = if step_u8 <= 1 {
                            (vr0, vr1)
                        } else if step_u8 > 127 {
                            use core::arch::x86_64::_mm256_setzero_si256;
                            (_mm256_setzero_si256(), _mm256_setzero_si256())
                        } else {
                            crate::traits::branchless::avx2_qdps_filter_vec64(vr0, vr1, step_u8)
                        };
                        crate::traits::branchless::avx2_vec64_to_words64(vf0, vf1)
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    let slerped =
                        crate::traits::branchless::swar_slerp_words_with_bias_64(&wa, &wb, s, bias_restore);
                    swar_q7_qdps_filter_words_64(&slerped, step_u8)
                }
            }
        };

        let mut blended = AlignedBuffer16::<gcso_q7_t, MAX_ATTENTION_HEADS>::zeroed();
        store_q7_words_64(&mut blended.data, &swar_words);

        steer_from_swar_words_core_safe(
            &swar_words,
            0xFF,
            128,
            head_dim,
            num_heads,
            true,
            max_rad,
            query_tensor,
            key_tensor,
        );
        Ok(blended)
    }

    /// Applies **Isotropic Block-Diagonal Scaling** ($\mathbf{D} = \mathrm{diag}(s_{h,0}\mathbf{I}_2, \dots, s_{h,d/2-1}\mathbf{I}_2)$)
    /// across all 2D subspaces of `tensor` (Theorem 1.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `tensor.len() < head_dim * num_heads` or
    /// `block_scales.len() < (head_dim / 2) * num_heads`.
    #[inline]
    fn apply_isotropic_block_scaling(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        block_scales: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<()> {
        let (_head_dim, _num_heads, expected_len) = self.validate_head_layout()?;
        let num_pairs = expected_len / 2;
        if tensor.len() < expected_len || block_scales.len() < num_pairs {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let t_slice = &mut tensor.as_mut_slice()[..expected_len];
        let s_slice = &block_scales.as_slice()[..num_pairs];
        for (pair, &raw_scale) in t_slice.chunks_exact_mut(2).zip(s_slice.iter()) {
            let s = branchless_flush_subnormal_f32(sanitize_inplace_gain_f32(pair[0], raw_scale));
            pair[0] = branchless_flush_subnormal_f32(pair[0] * s);
            pair[1] = branchless_flush_subnormal_f32(pair[1] * s);
        }
        Ok(())
    }

    /// Applies the **Composite Modulation Operator** $\mathbf{M}_{\mathrm{GCSO}} = \mathbf{D} \cdot \mathbf{R}_{\mathrm{DPSR}}(\boldsymbol{\Delta\theta}) + \mathbf{u}\mathbf{v}^T$
    /// in a single fused pass over `tensor` (Theorem 1.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `max_rad` is non-finite,
    /// or slice lengths are insufficient.
    #[inline]
    fn apply_composite_dpsr_modulation(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        block_scales: &AlignedSlice32<'_, f32>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        let pairs_per_head = head_dim / 2;
        let total_pairs = expected_len / 2;
        if tensor.len() < expected_len
            || phase_deltas.len() < num_heads
            || block_scales.len() < total_pairs
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let (is_default, eff_max, q7_to_norm_scale) = resolve_ripa_norm_params(max_rad);
        let t_slice = tensor.as_mut_slice();
        let p_slice = phase_deltas.as_slice();
        let s_slice = block_scales.as_slice();

        for (h, &raw_q7) in p_slice.iter().enumerate().take(num_heads) {
            let (sin_t, cos_t) =
                eval_ripa_sincos_with_params(raw_q7, false, is_default, eff_max, q7_to_norm_scale);
            let h_base = h * head_dim;
            let s_base = h * pairs_per_head;
            let head_buf = &mut t_slice[h_base..h_base + head_dim];
            let head_scales = &s_slice[s_base..s_base + pairs_per_head];

            for (pair, &raw_s) in head_buf.chunks_exact_mut(2).zip(head_scales.iter()) {
                let s = branchless_flush_subnormal_f32(sanitize_inplace_gain_f32(pair[0], raw_s));
                let (r0, r1) =
                    branchless_rotate_so2_pair_scaled(pair[0], pair[1], sin_t, cos_t, s);
                pair[0] = r0;
                pair[1] = r1;
            }
        }
        Ok(())
    }

    /// Applies multi-token batched phase steering across `batch_tokens` contiguous tokens.
    ///
    /// Supports both **broadcast single-token phase slices** (`phase_deltas_batch.len() >= num_heads`,
    /// where consecutive uniform-phase head runs `(run_start, run_end, sin_t, cos_t)` are coalesced
    /// once outside the token loop and rotated in a single contiguous call per run per token)
    /// and **per-token phase slices** (`phase_deltas_batch.len() >= batch_tokens * num_heads`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `batch_tokens == 0`, `tensor_batch.len() < batch_tokens * head_dim * num_heads`,
    /// or `phase_deltas_batch.len() < num_heads`.
    #[inline(always)]
    fn apply_phase_steering_batch(
        &self,
        tensor_batch: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas_batch: &AlignedSlice16<'_, gcso_q7_t>,
        batch_tokens: usize,
    ) -> GcsoResult<()> {
        let (head_dim, num_heads, token_elems) = self.validate_head_layout()?;
        if batch_tokens == 1 {
            if tensor_batch.len() < token_elems || phase_deltas_batch.len() < num_heads {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            apply_phase_steering_slice_core(
                &mut tensor_batch.as_mut_slice()[..token_elems],
                &phase_deltas_batch.as_slice()[..num_heads],
                head_dim,
                num_heads,
            );
            return Ok(());
        }
        apply_phase_steering_multi_token_batch_core(
            tensor_batch.as_mut_slice(),
            phase_deltas_batch.as_slice(),
            batch_tokens,
            head_dim,
            num_heads,
            token_elems,
        )
    }

    /// Applies fused multi-token batched phase steering to both Query and Key batches in lockstep.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `query_batch` and `key_batch` alias/overlap,
    /// `batch_tokens == 0`, or slice bounds fail.
    #[inline(always)]
    fn apply_phase_steering_fused_batch(
        &self,
        query_batch: &mut AlignedSliceMut32<'_, f32>,
        key_batch: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas_batch: &AlignedSlice16<'_, gcso_q7_t>,
        batch_tokens: usize,
    ) -> GcsoResult<()> {
        query_batch.ensure_disjoint_mut(key_batch)?;
        let (head_dim, num_heads, token_elems) = self.validate_head_layout()?;
        if batch_tokens == 1 {
            if query_batch.len() < token_elems
                || key_batch.len() < token_elems
                || phase_deltas_batch.len() < num_heads
            {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            apply_phase_steering_pair_slice_core(
                &mut query_batch.as_mut_slice()[..token_elems],
                &mut key_batch.as_mut_slice()[..token_elems],
                &phase_deltas_batch.as_slice()[..num_heads],
                head_dim,
                num_heads,
            );
            return Ok(());
        }
        apply_phase_steering_fused_multi_token_batch_core(
            query_batch.as_mut_slice(),
            key_batch.as_mut_slice(),
            phase_deltas_batch.as_slice(),
            batch_tokens,
            head_dim,
            num_heads,
            token_elems,
        )
    }

    /// Applies fused RIPA soft-bounded multi-token batched phase steering to Query and Key batches,
    /// supporting both broadcast (`len >= num_heads`, hoisting RIPA `(sin_t, cos_t)` outside the
    /// token loop) and per-token (`len >= batch_tokens * num_heads`) phase slices.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `batch_tokens == 0`, `max_rad <= 0.0`,
    /// `query_batch` and `key_batch` overlap, or buffer lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_fused_safe_batch(
        &self,
        query_batch: &mut AlignedSliceMut32<'_, f32>,
        key_batch: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas_batch: &AlignedSlice16<'_, gcso_q7_t>,
        batch_tokens: usize,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        query_batch.ensure_disjoint_mut(key_batch)?;
        let (head_dim, num_heads, token_elems) = self.validate_head_layout()?;
        if batch_tokens == 1 {
            if query_batch.len() < token_elems
                || key_batch.len() < token_elems
                || phase_deltas_batch.len() < num_heads
            {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            apply_ripa_pair_slice_core(
                &mut query_batch.as_mut_slice()[..token_elems],
                &mut key_batch.as_mut_slice()[..token_elems],
                &phase_deltas_batch.as_slice()[..num_heads],
                head_dim,
                num_heads,
                false,
                max_rad,
            );
            return Ok(());
        }
        apply_phase_steering_fused_safe_multi_token_batch_core(
            query_batch.as_mut_slice(),
            key_batch.as_mut_slice(),
            phase_deltas_batch.as_slice(),
            batch_tokens,
            head_dim,
            num_heads,
            token_elems,
            max_rad,
        )
    }

    /// Applies subnormal-safe and near-identity-guarded multiplicative phase damping in-place across `tensor`
    /// using `sanitize_inplace_gain_f32` and unconditional 32-float unrolled AVX2 scaling
    /// (`scale_f32_slice_unrolled8`).
    ///
    /// Guarantees exact binary dyadic reversibility (e.g., alternating `0.5` and `2.0` produces zero
    /// mantissa ULP drift) and flushes subnormal products (`|x * s| < F32_MIN_POSITIVE_NORMAL`) to `+0.0`
    /// in YMM registers to prevent x86 microcode traps over million-token horizons.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `tensor` is empty, `damping_factor <= 0.0`,
    /// or `damping_factor` is non-finite.
    #[inline]
    fn apply_soft_phase_damping(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        damping_factor: f32,
    ) -> GcsoResult<()> {
        if tensor.is_empty()
            || damping_factor <= 0.0
            || !branchless_is_finite_f32(damping_factor)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        scale_f32_slice_unrolled8(tensor.as_mut_slice(), damping_factor);
        Ok(())
    }

    /// Applies RIPA hyperbolic tangent soft-clamping ($\theta_{\mathrm{safe}} = \theta_{\max}\tanh(\theta/\theta_{\max})$)
    /// in-place on floating-point radian angles.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `raw_angles` is empty, `max_rad <= 0.0`, or `max_rad` is non-finite.
    #[inline]
    fn clamp_angles_ripa_in_place(
        &self,
        raw_angles: &mut AlignedSliceMut32<'_, f32>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if raw_angles.is_empty() || max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        clamp_angles_ripa_in_place_simd8(raw_angles.as_mut_slice(), max_rad);
        Ok(())
    }

    /// Filters sub-grid Q7 phase noise below `min_step_rad` using 8-lane SWAR (`swar_q7_qdps_filter_x8`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `raw_phases.len() < self.num_heads()`, `min_step_rad < 0.0`,
    /// or `min_step_rad` is non-finite.
    #[inline(always)]
    fn qdps_filter_step(
        &self,
        raw_phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        min_step_rad: f32,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || raw_phases.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let min_step_q7 = i32::from(branchless_rad_to_qdps_step_u8(min_step_rad));
        swar_qdps_filter_q7_slice_in_place(raw_phases.as_mut_slice(), num_heads, min_step_q7);
        Ok(())
    }

    /// Applies Lazy Phase Unwrapping relative shift directly to a Query tensor in `f32` radians
    /// using 8-head batched AVX2 range-reduced `sincos_f32_x8_simd`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor.len() < head_dim * num_heads` or
    /// `accumulated_context_phases.len() < num_heads`.
    #[inline]
    fn lazy_unwrap_override(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        accumulated_context_phases: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<()> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len || accumulated_context_phases.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let q_slice = query_tensor.as_mut_slice();
        let c_slice = accumulated_context_phases.as_slice();

        let mut h = 0usize;
        while h + 8 <= num_heads {
            let mut sin_8 = [0.0_f32; 8];
            let mut cos_8 = [0.0_f32; 8];
            sincos_f32_x8_simd(&c_slice[h..h + 8], &mut sin_8, &mut cos_8);
            let mut lane = 0usize;
            while lane < 8 {
                let base = (h + lane) * head_dim;
                rotate_so2_slice_unrolled16(
                    &mut q_slice[base..base + head_dim],
                    sin_8[lane],
                    cos_8[lane],
                );
                lane += 1;
            }
            h += 8;
        }
        while h < num_heads {
            let (sin_t, cos_t) = branchless_sincos_f32(c_slice[h]);
            let base = h * head_dim;
            rotate_so2_slice_unrolled16(&mut q_slice[base..base + head_dim], sin_t, cos_t);
            h += 1;
        }
        Ok(())
    }

    /// Applies underflow-resilient SLERP norm stabilization across `tensor` using
    /// `norm_l2_underflow_resilient_f32` and division-free [`branchless_inv_pos_f32`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `tensor` is empty or `target_norm <= 0.0` /
    /// `damping_alpha < 0.0`, or [`GCSO_ERROR_EDBC_SINGULARITY`] if `tensor` contains `NaN`/`Inf`.
    #[inline]
    fn slerp_norm_guard_stable(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        target_norm: f32,
        damping_alpha: f32,
    ) -> GcsoResult<()> {
        if tensor.is_empty()
            || target_norm <= 0.0
            || damping_alpha < 0.0
            || !branchless_is_finite_f32(target_norm)
            || !branchless_is_finite_f32(damping_alpha)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slice = tensor.as_mut_slice();
        let (current_norm, is_finite) = norm_l2_underflow_resilient_f32(slice);
        if !is_finite {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let alpha = branchless_clamp_f32(damping_alpha, 0.0, 1.0);
        let valid_norm = current_norm > 0.0;
        let ratio = if current_norm >= F32_MIN_POSITIVE_NORMAL {
            target_norm * fast_inv_pos_normal_f32(current_norm)
        } else {
            target_norm
                * branchless_inv_pos_f32(branchless_select_f32(valid_norm, current_norm, 1.0))
        };
        let scale = branchless_select_f32(valid_norm, (1.0 - alpha) + alpha * ratio, 1.0);
        scale_f32_slice_unrolled8(slice, scale);
        Ok(())
    }

    /// Applies a fused phase-derived bias shift to `logits` in-place using 8-lane SWAR signed byte
    /// reduction and 32-float unrolled AVX2 scalar addition (`add_scalar_f32_slice_unrolled8`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty or `phase_deltas.len() < self.num_heads()`.
    #[inline]
    fn fused_logit_shift(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty() || phase_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let p_slice = phase_deltas.as_slice();
        let mut phase_sum_i32 = 0i32;
        let mut h = 0usize;
        while h + 8 <= num_heads {
            let u_w = load_q7_x8_slice(p_slice, h) ^ SWAR_SIGN_MASK_64;
            phase_sum_i32 += (swar_u8_sum_x8(u_w) as i32) - 1024;
            h += 8;
        }
        while h < num_heads {
            phase_sum_i32 += i32::from(p_slice[h]);
            h += 1;
        }
        let mean_phase_rad =
            (phase_sum_i32 as f32) * Q7_PHASE_SCALE_RAD * fast_inv_len_f32(num_heads);
        add_scalar_f32_slice_unrolled8(logits.as_mut_slice(), mean_phase_rad);
        Ok(())
    }

    /// Modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
    /// directly in-place by **Attention $\times$ DPSR Phase Interference**
    /// $\Delta z_{t,j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_j)$
    /// across `memory_anchor_phases` in zero-copy, zero-allocation ($C_{\mathrm{alloc}} = 0$) mode.
    ///
    /// Returns the mean phase coherence across the modulated candidate memories.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `attention_scores` or `memory_anchor_phases` is
    /// empty, `head_dim()` is invalid, or `interference_gain` is non-finite.
    #[inline]
    fn modulate_attention_scores_by_phase_interference(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        memory_anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        interference_gain: f32,
    ) -> GcsoResult<f32> {
        let (head_dim, _num_heads, _expected_len) = self.validate_head_layout()?;
        if attention_scores.is_empty()
            || memory_anchor_phases.is_empty()
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(modulate_attention_scores_by_phase_bank_64(
            attention_scores.as_mut_slice(),
            query_phases,
            memory_anchor_phases,
            head_dim as u32,
            interference_gain,
        ))
    }

    /// Modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
    /// directly in-place by the full **Bayes-Attention Stigmergic Transition Score**
    /// (combining sequential conditional phase interference and `ActionHub` pointer-chain link priors):
    /// $$\Delta z_{t \to j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_j) + \eta \ln(1 + \rho_j) + \lambda_{\mathrm{link}} \mathbb{I}(\mathrm{linked\_trail\_id} == j)$$
    ///
    /// Returns `(mean_phase_coherence, best_post_modulated_candidate_idx)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `attention_scores`, `candidate_key_phases`, or
    /// `candidate_stigmergic_densities` is empty, `head_dim()` is invalid, or any gain is non-finite.
    #[inline]
    fn modulate_attention_scores_by_bayesian_trails(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        candidate_key_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        candidate_stigmergic_densities: &[f32],
        linked_trail_id: Option<u32>,
        interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
    ) -> GcsoResult<(f32, u32)> {
        let (head_dim, _num_heads, _expected_len) = self.validate_head_layout()?;
        if attention_scores.is_empty()
            || candidate_key_phases.is_empty()
            || candidate_stigmergic_densities.is_empty()
            || !branchless_is_finite_f32(interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(modulate_attention_scores_by_bayesian_trails_64(
            attention_scores.as_mut_slice(),
            query_phases,
            candidate_key_phases,
            candidate_stigmergic_densities,
            linked_trail_id,
            head_dim as u32,
            interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
        ))
    }

    /// Applies **Multiplicative Inverse-Temperature Logit Scaling**
    /// $\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \cdot \mathcal{C}_{\mathrm{phase}}, 0.25, 4.0)$
    /// (`logits[i] *= beta_eff`), sharpening the Softmax distribution under constructive phase
    /// alignment ($\mathcal{C}_{\mathrm{phase}} > 0$) and flattening it under destructive interference ($\mathcal{C}_{\mathrm{phase}} < 0$).
    ///
    /// Returns `(phase_coherence, applied_inv_temp_scale)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty, `interference_gain` is non-finite,
    /// or phase slices have fewer than `self.num_heads()` elements.
    #[inline]
    fn apply_phase_interference_inv_temperature_scaling(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        memory_phases: &AlignedSlice16<'_, gcso_q7_t>,
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32)> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty()
            || !branchless_is_finite_f32(interference_gain)
            || query_phases.len() < num_heads
            || memory_phases.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let q_p = &query_phases.as_slice()[..num_heads];
        let m_p = &memory_phases.as_slice()[..num_heads];
        let coh = eval_multi_head_phase_coherence_unrolled4(q_p, m_p);
        let inv_temp = branchless_phase_coherence_inv_temp_scale(coh, interference_gain);
        scale_f32_slice_unrolled8(logits.as_mut_slice(), inv_temp);
        Ok((coh, inv_temp))
    }

    /// Applies **Multiplicative Inverse-Temperature Logit Scaling**
    /// $\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \cdot \mathcal{C}_{\mathrm{phase}}, 0.25, 4.0)$
    /// directly from 64-head Q7 phase arrays (`[gcso_q7_t; 64]`) in zero-copy mode.
    ///
    /// Returns `(phase_coherence, applied_inv_temp_scale)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty or `interference_gain` is non-finite.
    #[inline]
    fn apply_phase_interference_inv_temperature_scaling_64(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        memory_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32)> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty()
            || num_heads > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let coh = eval_multi_head_phase_coherence_unrolled4(
            &query_phases[..num_heads],
            &memory_phases[..num_heads],
        );
        let inv_temp = branchless_phase_coherence_inv_temp_scale(coh, interference_gain);
        scale_f32_slice_unrolled8(logits.as_mut_slice(), inv_temp);
        Ok((coh, inv_temp))
    }

    /// Modulates `logits` in-place by the **Attention $\times$ DPSR Phase-Interference Bias**
    /// $\Delta z = \gamma \cdot \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_q + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_m)$
    /// across the configured `num_heads` heads, returning `(phase_coherence, applied_logit_bias)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty, `interference_gain` is non-finite,
    /// or phase slices have fewer than `self.num_heads()` elements.
    #[inline]
    fn apply_phase_interference_logit_modulation(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        memory_phases: &AlignedSlice16<'_, gcso_q7_t>,
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32)> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty()
            || !branchless_is_finite_f32(interference_gain)
            || query_phases.len() < num_heads
            || memory_phases.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let q_p = &query_phases.as_slice()[..num_heads];
        let m_p = &memory_phases.as_slice()[..num_heads];
        let coh = eval_multi_head_phase_coherence_unrolled4(q_p, m_p);
        let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
        let bias = branchless_flush_subnormal_f32(clean_gain * coh);
        add_scalar_f32_slice_unrolled8(logits.as_mut_slice(), bias);
        Ok((coh, bias))
    }

    /// Modulates `logits` in-place by the **Attention $\times$ DPSR Phase-Interference Bias**
    /// directly from 64-head Q7 phase arrays (`[gcso_q7_t; 64]`) in zero-copy mode.
    ///
    /// Returns `(phase_coherence, applied_logit_bias)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty or `interference_gain` is non-finite.
    #[inline]
    fn apply_phase_interference_logit_modulation_64(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        memory_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32)> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty()
            || num_heads > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let coh = eval_multi_head_phase_coherence_unrolled4(
            &query_phases[..num_heads],
            &memory_phases[..num_heads],
        );
        let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
        let bias = branchless_flush_subnormal_f32(clean_gain * coh);
        add_scalar_f32_slice_unrolled8(logits.as_mut_slice(), bias);
        Ok((coh, bias))
    }

    /// Evaluates **Unified-Field Phase-Interference Logit Modulation** across an arbitrary slice
    /// of memory/attractor anchors (`anchor_phases`, `anchor_base_energies`), shifting `logits`
    /// by the energy-weighted phase interference $\Delta z = \gamma \cdot \bar{\mathcal{C}}_{\mathrm{unified}}$.
    ///
    /// Returns `(total_resonant_pull, mean_coherence, applied_logit_bias)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty, `anchor_phases.len() != anchor_base_energies.len()`,
    /// or `interference_gain` is non-finite.
    #[inline]
    fn apply_unified_field_logit_interference(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32, f32)> {
        if logits.is_empty()
            || anchor_phases.len() != anchor_base_energies.len()
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let trail_words = load_q7_words_64(trail_phases);
        let (_folded_words, total_pull, mean_coh, _dom_idx, _act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &trail_words,
                0.0,
                anchor_phases,
                anchor_base_energies,
            );
        let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
        let bias = branchless_flush_subnormal_f32(clean_gain * mean_coh);
        add_scalar_f32_slice_unrolled8(logits.as_mut_slice(), bias);
        Ok((total_pull, mean_coh, bias))
    }

    /// Computes the closed-form **Orthogonal Procrustes Phase Delta** $\Delta\theta^{(h)} \in [-128, 127]$
    /// that aligns each attention head of `source_tensor` with `target_tensor` in $SO(2)^{d_{\mathrm{head}}/2}$
    /// via AVX2 SIMD (`procrustes_q7_slice_simd`, Section 2.1 & 5.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `source_tensor` or `target_tensor` has fewer than
    /// `head_dim * num_heads` elements or `phase_deltas_out` has fewer than `num_heads` elements.
    #[inline]
    fn compute_procrustes_phase_delta(
        &self,
        source_tensor: &AlignedSlice32<'_, f32>,
        target_tensor: &AlignedSlice32<'_, f32>,
        phase_deltas_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if source_tensor.len() < expected_len
            || target_tensor.len() < expected_len
            || phase_deltas_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        procrustes_q7_slice_simd(
            source_tensor.as_slice(),
            target_tensor.as_slice(),
            head_dim,
            num_heads,
            phase_deltas_out.as_mut_slice(),
        );
        Ok(())
    }

    /// Evaluates circular phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$ and
    /// circular resultant statistics $(\bar{R}_a, \bar{\theta}_{a,\mathrm{Q7}}, \kappa_a)$ across 64 heads
    /// in a single register pass via [`branchless_q7_coherence_and_resultant_64`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `phases_a.len() != 64` or `phases_b.len() != 64`.
    #[inline(always)]
    fn evaluate_phase_coherence_and_resultant_64(
        &self,
        phases_a: &AlignedSlice16<'_, gcso_q7_t>,
        phases_b: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<(f32, f32, gcso_q7_t, f32)> {
        let a_arr = phases_a.as_array::<MAX_ATTENTION_HEADS>()?;
        let b_arr = phases_b.as_array::<MAX_ATTENTION_HEADS>()?;
        Ok(branchless_q7_coherence_and_resultant_64(a_arr, b_arr))
    }
}

// ===================================================================
// 2. Entropy-Gated Phase Lookup & Speculative Prefetch (`EntropyGatedPhaseLookup`)
// ===================================================================

/// Trait for Entropy-Gated Phase Lookup, AOT Geometric Phase Basis Synthesis, and
/// Speculative Paged-Bitmask Prefetching (**Nano / Micro Level**, Section 2.1).
pub trait EntropyGatedPhaseLookup: BranchlessKernel {
    /// Evaluates the smooth branchless entropy gate scale $g(\tilde{H}) = g_{\max} \cdot \mathrm{clamp}(\tilde{H} / \ln 2, 0, 1)$.
    #[inline(always)]
    #[must_use]
    fn eval_entropy_gate_scale(&self, h_tilde: f32, max_gain: f32) -> f32 {
        let valid = branchless_is_finite_f32(h_tilde) & branchless_is_finite_f32(max_gain);
        let norm_h = branchless_clamp_f32(branchless_relu_f32(h_tilde) * (1.0 / LN_2), 0.0, 1.0);
        let raw = branchless_relu_f32(max_gain) * norm_h;
        branchless_select_f32(valid, branchless_flush_subnormal_f32(raw), 0.0)
    }

    /// Synthesizes a 64-head Q7 phase profile from `k_bases` pre-computed AOT geometric phase bases
    /// (`basis_table`, length `>= k_bases * num_heads`) weighted by `alpha_weights` (`k_bases`)
    /// and gated by `h_tilde` (Section 2.1).
    ///
    /// Traverses `basis_table` in cache-friendly row-major (`b_idx` outer, `h` inner) order when
    /// `num_heads <= MAX_ATTENTION_HEADS`, hoisting `alpha` sanitization and `Q7_PHASE_SCALE_RAD * gate`
    /// scaling outside the head loop.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `k_bases == 0`, `num_heads == 0`,
    /// `basis_table.len() < k_bases * num_heads`, `alpha_weights.len() < k_bases`, or
    /// `synthesized_out.len() < num_heads`.
    #[inline]
    fn synthesize_aot_phase_bases(
        &self,
        basis_table: &AlignedSlice16<'_, gcso_q7_t>,
        alpha_weights: &AlignedSlice32<'_, f32>,
        k_bases: usize,
        num_heads: usize,
        h_tilde: f32,
        max_gain: f32,
        synthesized_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let required_table = k_bases
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if k_bases == 0
            || num_heads == 0
            || basis_table.len() < required_table
            || alpha_weights.len() < k_bases
            || synthesized_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let gate = self.eval_entropy_gate_scale(h_tilde, max_gain);
        let out_slice = &mut synthesized_out.as_mut_slice()[..num_heads];
        if gate == 0.0 {
            out_slice.fill(0);
            return Ok(());
        }

        let b_slice = basis_table.as_slice();
        let a_slice = alpha_weights.as_slice();
        let rad_gate_scale = Q7_PHASE_SCALE_RAD * gate;

        if num_heads <= MAX_ATTENTION_HEADS {
            let mut acc_rad_scaled = [0.0_f32; MAX_ATTENTION_HEADS];
            for (b_idx, &alpha) in a_slice.iter().enumerate().take(k_bases) {
                let clean_a = branchless_select_f32(branchless_is_finite_f32(alpha), alpha, 0.0);
                if clean_a == 0.0 {
                    continue;
                }
                let w_b = clean_a * rad_gate_scale;
                let row = &b_slice[b_idx * num_heads..(b_idx + 1) * num_heads];
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    // SAFETY: `target_feature = "avx2"` is enabled; `acc_rad_scaled` and `row` have `>= num_heads` elements.
                    unsafe {
                        use core::arch::x86_64::{
                            __m128i, _mm256_add_ps, _mm256_cvtepi32_ps, _mm256_cvtepi8_epi32,
                            _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
                            _mm_loadl_epi64,
                        };
                        let vw = _mm256_set1_ps(w_b);
                        let ap = acc_rad_scaled.as_mut_ptr();
                        let rp = row.as_ptr();
                        let mut h = 0usize;
                        while h + 8 <= num_heads {
                            let q8 = _mm_loadl_epi64(rp.add(h).cast::<__m128i>());
                            let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
                            let cur = _mm256_loadu_ps(ap.add(h));
                            _mm256_storeu_ps(ap.add(h), _mm256_add_ps(cur, _mm256_mul_ps(vw, qf)));
                            h += 8;
                        }
                        while h < num_heads {
                            *ap.add(h) += w_b * f32::from(*rp.add(h));
                            h += 1;
                        }
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    for (acc_h, &basis_q7) in acc_rad_scaled[..num_heads].iter_mut().zip(row.iter())
                    {
                        *acc_h += w_b * f32::from(basis_q7);
                    }
                }
            }
            for (out_q7, &rad) in out_slice.iter_mut().zip(acc_rad_scaled[..num_heads].iter()) {
                *out_q7 = branchless_rad_to_q7(rad);
            }
        } else {
            for (h, out_q7) in out_slice.iter_mut().enumerate() {
                let mut acc_rad = 0.0_f32;
                for (b_idx, &alpha) in a_slice.iter().enumerate().take(k_bases) {
                    let clean_a =
                        branchless_select_f32(branchless_is_finite_f32(alpha), alpha, 0.0);
                    let basis_q7 = b_slice[b_idx * num_heads + h];
                    acc_rad += clean_a * (f32::from(basis_q7) * Q7_PHASE_SCALE_RAD);
                }
                *out_q7 = branchless_rad_to_q7(acc_rad * gate);
            }
        }
        Ok(())
    }

    /// Computes a deterministic $\mathcal{O}(1)$ speculative prefetch index into a power-of-two
    /// phase table of size `table_capacity_pow2` from a 64-bit recent KV activity bitmask.
    #[inline(always)]
    #[must_use]
    fn speculative_prefetch_hash(&self, recent_kv_bitmask: u64, table_capacity_pow2: u32) -> u32 {
        let is_pow2 =
            (table_capacity_pow2 > 0) & ((table_capacity_pow2 & (table_capacity_pow2 - 1)) == 0);
        let mask = branchless_select_u32(is_pow2, table_capacity_pow2.wrapping_sub(1), 0);
        (branchless_splitmix64(recent_kv_bitmask) as u32) & mask
    }

    /// Folds a full 256-bit [`gcso_paged_bitmask_t`] and a [`gcso_tagged_ptr_t`] into a
    /// speculative prefetch table slot index in $\mathcal{O}(1)$ branchless operations.
    #[inline(always)]
    #[must_use]
    fn speculative_paged_bitmask_hash(
        &self,
        paged_mask: &gcso_paged_bitmask_t,
        tagged_ptr: gcso_tagged_ptr_t,
        table_capacity_pow2: u32,
    ) -> u32 {
        let w01 = paged_mask.bits[0] ^ paged_mask.bits[1].rotate_left(19);
        let w23 = paged_mask.bits[2].rotate_left(37) ^ paged_mask.bits[3].rotate_left(53);
        let combined = w01 ^ w23 ^ tagged_ptr.raw();
        self.speculative_prefetch_hash(combined, table_capacity_pow2)
    }

    /// Computes a speculative prefetch index directly from a [`gcso_pointer_trail_t`] and optional
    /// [`gcso_paged_bitmask_t`] in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    fn speculative_prefetch_from_trail(
        &self,
        trail: &gcso_pointer_trail_t,
        paged_mask: Option<&gcso_paged_bitmask_t>,
        table_capacity_pow2: u32,
    ) -> u32 {
        let ptr_mix = trail.current_ptr
            ^ trail.prev_ptr.rotate_left(23)
            ^ (u64::from(trail.cluster_id) << 32)
            ^ u64::from(trail.step_count);
        match paged_mask {
            Some(m) => self.speculative_paged_bitmask_hash(
                m,
                gcso_tagged_ptr_t(ptr_mix),
                table_capacity_pow2,
            ),
            None => self.speculative_prefetch_hash(ptr_mix, table_capacity_pow2),
        }
    }
}

// ===================================================================
// 3. Dynamic Layer Skipping with Phase Compensation (`DynamicLayerSkipper`)
// ===================================================================

/// Trait for Dynamic Layer Skipping and SWAR Phase Compensation during low-entropy inertial
/// decoding regimes (**Micro / Mezzo Level**, Section 3.2).
pub trait DynamicLayerSkipper: BranchlessKernel {
    /// Evaluates whether an intermediate Transformer layer can be skipped branchlessly:
    /// $\tilde{H} \le \tau_{\mathrm{skip}} \land \lvert\Phi_M(t)\rvert \le \tau_{\mathrm{flux}} \land \text{consecutive\_skips} < \text{max\_skips}$.
    #[inline(always)]
    #[must_use]
    fn should_skip_layer_branchless(
        &self,
        h_tilde: f32,
        flux_rate: f32,
        skip_entropy_thresh: f32,
        skip_flux_thresh: f32,
        consecutive_skips: u32,
        max_consecutive_skips: u32,
    ) -> bool {
        let valid = branchless_is_finite_f32(h_tilde)
            & branchless_is_finite_f32(flux_rate)
            & branchless_is_finite_f32(skip_entropy_thresh)
            & branchless_is_finite_f32(skip_flux_thresh);
        let low_h = h_tilde <= skip_entropy_thresh;
        let low_flux = self.abs_f32(flux_rate) <= skip_flux_thresh;
        let under_cap = consecutive_skips < max_consecutive_skips;
        valid & low_h & low_flux & under_cap
    }

    /// Evaluates whether an intermediate layer should be skipped by combining `tagged_ptr`'s
    /// `ACTION_MODE_LAYER_SKIP` bit (`0x02`) with the entropy/flux criteria.
    #[inline(always)]
    #[must_use]
    fn should_skip_layer_by_ptr_branchless(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        h_tilde: f32,
        flux_rate: f32,
        skip_entropy_thresh: f32,
        skip_flux_thresh: f32,
        consecutive_skips: u32,
        max_consecutive_skips: u32,
    ) -> bool {
        ptr_has_action_mode(tagged_ptr, ACTION_MODE_LAYER_SKIP)
            & self.should_skip_layer_branchless(
                h_tilde,
                flux_rate,
                skip_entropy_thresh,
                skip_flux_thresh,
                consecutive_skips,
                max_consecutive_skips,
            )
    }

    /// Applies 8-lane SWAR saturating phase compensation (`swar_q7_add_sat_x8`) across all
    /// `num_heads` heads when an intermediate Transformer layer is bypassed.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads == 0` or either slice has fewer than
    /// `num_heads` elements.
    #[inline]
    fn apply_skip_phase_compensation(
        &self,
        accumulated_phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        skip_compensation_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        num_heads: usize,
    ) -> GcsoResult<()> {
        if num_heads == 0
            || accumulated_phases.len() < num_heads
            || skip_compensation_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_add_sat_q7_slice_in_place(
            accumulated_phases.as_mut_slice(),
            skip_compensation_deltas.as_slice(),
            num_heads,
        );
        Ok(())
    }

    /// Evaluates pointer-gated layer skip eligibility and, if skipped, applies 8-lane SWAR
    /// phase compensation to `accumulated_phases` and increments `*consecutive_skips_inout`
    /// (or resets it to `0` when the layer is executed).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads == 0` or phase slices are smaller than `num_heads`.
    #[inline]
    fn evaluate_and_compensate_layer_skip_by_ptr(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        h_tilde: f32,
        flux_rate: f32,
        skip_entropy_thresh: f32,
        skip_flux_thresh: f32,
        max_consecutive_skips: u32,
        consecutive_skips_inout: &mut u32,
        accumulated_phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        skip_compensation_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        num_heads: usize,
    ) -> GcsoResult<bool> {
        if num_heads == 0
            || accumulated_phases.len() < num_heads
            || skip_compensation_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let skip = self.should_skip_layer_by_ptr_branchless(
            tagged_ptr,
            h_tilde,
            flux_rate,
            skip_entropy_thresh,
            skip_flux_thresh,
            *consecutive_skips_inout,
            max_consecutive_skips,
        );
        if skip {
            swar_add_sat_q7_slice_in_place(
                accumulated_phases.as_mut_slice(),
                skip_compensation_deltas.as_slice(),
                num_heads,
            );
            *consecutive_skips_inout = consecutive_skips_inout.saturating_add(1);
        } else {
            *consecutive_skips_inout = 0;
        }
        Ok(skip)
    }
}

