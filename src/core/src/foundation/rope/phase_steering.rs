// name: src/core/src/foundation/rope/phase_steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! PhaseSteering trait definitions and geometry-aware RoPE steering extensions.

use super::*;

// ===================================================================
// 1. Dynamic Phase-Shifted RoPE, `PhaseClamp`, `PhaseStepFilter` & Lazy Unwrapping (`PhaseSteering`)
// ===================================================================

/// Core trait for dynamic phase-shifted RoPE (`PhaseSteering`), soft-bounded phase clamping (`PhaseClamp`),
/// discrete phase-step filtering (`PhaseStepFilter`), Seam-Free Circular $S^1$ Lazy Phase Unwrapping,
/// Single-Manifold Unified-Field Phase Steering, Attention $\times$ Phase-Interference Score
/// Modulation, Bayesian Pointer-Trail Attention Modulation, and Closed-Form Orthogonal Procrustes
/// Phase Estimation (**Nano / Micro Level**).
pub trait PhaseSteering {
    /// Returns the configured attention head dimension ($d_{\mathrm{head}}$, positive even integer).
    fn head_dim(&self) -> u32;

    /// Returns the configured number of attention heads ($H > 0$).
    fn num_heads(&self) -> u32;

    /// Validates `(head_dim, num_heads)` invariants (`head_dim > 0`, `head_dim % 2 == 0`, `num_heads > 0`)
    /// and returns `(head_dim_usize, num_heads_usize, expected_tensor_len)`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `num_heads` is zero,
    /// or `head_dim * num_heads` overflows `usize`.
    #[inline(always)]
    fn validate_head_layout(&self) -> LmnkResult<(usize, usize, usize)> {
        let head_dim = self.head_dim() as usize;
        let num_heads = self.num_heads() as usize;
        if head_dim == 0 || (head_dim & 1) != 0 || num_heads == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let expected_len = head_dim
            .checked_mul(num_heads)
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        Ok((head_dim, num_heads, expected_len))
    }

    /// Applies dynamic phase-shifted RoPE (`PhaseSteering`) in-place over an arbitrary hardware-aligned
    /// tensor slice (`ALIGN` $\ge 32$, power of two).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `ALIGN < 32` or `ALIGN` is not a power of two,
    /// or [`LMNK_ERR_INVALID_ARGUMENT`] if slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_aligned<const ALIGN: usize>(
        &self,
        tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
    ) -> LmnkResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        apply_phase_steering_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies dynamic phase-shifted RoPE (`PhaseSteering`) in-place to a single 32B-aligned Query or Key tensor.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `tensor.len() < head_dim * num_heads` or `phase_deltas.len() < num_heads`.
    #[inline(always)]
    fn apply_phase_steering(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
    ) -> LmnkResult<()> {
        self.apply_phase_steering_aligned::<32>(tensor, phase_deltas)
    }

    /// Applies fused in-place phase steering to a pre-verified [`DisjointMutPair`] of `(Q, K)` tensors.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if tensor or phase slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_disjoint_pair<const ALIGN: usize>(
        &self,
        qk_pair: &mut DisjointMutPair<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
    ) -> LmnkResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        let (q_slice, k_slice) = qk_pair.as_mut_pair();
        if q_slice.len() < expected_len
            || k_slice.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
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

    /// Applies soft-bounded (`PhaseClamp`) phase steering ($\Delta\theta_{\mathrm{safe}} = \theta_{\max} \cdot \tanh(\Delta\theta / \theta_{\max})$)
    /// to the upper low-frequency quarter (`rot_dim = ((head_dim >> 2) & !1).max(2)`) of an arbitrary
    /// hardware-aligned tensor (`ALIGN >= 32`, Section 2.2).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `ALIGN < 32`, or [`LMNK_ERR_INVALID_ARGUMENT`]
    /// if `max_rad <= 0.0`, `max_rad` is non-finite, or slice bounds fail.
    #[inline(always)]
    fn apply_phase_steering_safe_aligned<const ALIGN: usize>(
        &self,
        tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        max_rad: f32,
    ) -> LmnkResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        apply_phase_clamp_single_slice_core(
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

    /// Applies soft-bounded (`PhaseClamp`) phase steering to the upper $d_{\mathrm{head}}/4$ low-frequency
    /// channels of a single 32B-aligned tensor (Section 2.2).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `max_rad` is non-finite,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_safe(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        max_rad: f32,
    ) -> LmnkResult<()> {
        self.apply_phase_steering_safe_aligned::<32>(tensor, phase_deltas, max_rad)
    }

    /// Applies soft-bounded (`PhaseClamp`) phase steering directly to a **HalfSplit (NeoX)** tensor
    /// (`(x[i], x[i + rotary_dim / 2])`) with optional partial-RoPE boundary `rotary_dim <= head_dim`,
    /// eliminating layout conversion copies on the hot path.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `rotary_dim` is invalid,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_half_split_safe(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        rotary_dim: usize,
        max_rad: f32,
    ) -> LmnkResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if rotary_dim < 2
            || (rotary_dim & 1) != 0
            || rotary_dim > head_dim
            || tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        apply_phase_clamp_half_split_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            rotary_dim,
            num_heads,
            max_rad,
        );
        Ok(())
    }

    /// Applies **Seam-Free $S^1$ Geodesic-Folded Soft-Bounded Phase Steering** to the upper $d_{\mathrm{head}}/4$
    /// low-frequency channels of a single 32B-aligned tensor.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `max_rad` is non-finite,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_geodesic_safe(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        max_rad: f32,
    ) -> LmnkResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        apply_phase_clamp_single_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            true,
            max_rad,
        );
        Ok(())
    }

    /// Applies fused soft-bounded (`PhaseClamp`) phase steering to the upper $d_{\mathrm{head}}/4$ low-frequency
    /// channels of both Query and Key tensors simultaneously.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias, `max_rad <= 0.0`,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_fused_safe(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        max_rad: f32,
    ) -> LmnkResult<()> {
        self.apply_phase_steering_fused_safe_aligned::<32>(
            query_tensor,
            key_tensor,
            phase_deltas,
            max_rad,
        )
    }

    /// Applies fused soft-bounded (`PhaseClamp`) phase steering to both Query and Key tensors over any alignment `ALIGN >= 32`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`] if `ALIGN < 32`, or [`LMNK_ERR_INVALID_ARGUMENT`]
    /// if slices alias, `max_rad <= 0.0`, or lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_fused_safe_aligned<const ALIGN: usize>(
        &self,
        query_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        key_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        max_rad: f32,
    ) -> LmnkResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        apply_phase_clamp_pair_slice_core(
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

    /// Applies soft-bounded (`PhaseClamp`) phase steering gated by `tagged_ptr`'s `ACTION_MODE_PHASE_STEER` bit.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `max_rad <= 0.0` or slice lengths are insufficient.
    #[inline]
    fn apply_pointer_gated_phase_steering_safe(
        &self,
        tagged_ptr: TaggedPtr,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        max_rad: f32,
    ) -> LmnkResult<bool> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
            || tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_PHASE_STEER) {
            return Ok(false);
        }
        apply_phase_clamp_single_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            false,
            max_rad,
        );
        Ok(true)
    }

    /// Applies fused soft-bounded (`PhaseClamp`) phase steering to both Query and Key tensors gated by `tagged_ptr`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias, `max_rad <= 0.0`,
    /// or slice lengths are insufficient.
    #[inline]
    fn apply_pointer_gated_phase_steering_fused_safe(
        &self,
        tagged_ptr: TaggedPtr,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        max_rad: f32,
    ) -> LmnkResult<bool> {
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
            || query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_PHASE_STEER) {
            return Ok(false);
        }
        apply_phase_clamp_pair_slice_core(
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

    /// Applies subnormal-safe and near-identity-guarded multiplicative phase damping in-place across `tensor`
    /// using `sanitize_inplace_gain_f32` and unconditional 32-float unrolled AVX2 scaling
    /// (`scale_f32_slice_unrolled8`).
    ///
    /// Guarantees exact binary dyadic reversibility (e.g., alternating `0.5` and `2.0` produces zero
    /// mantissa ULP drift) and flushes subnormal products (`|x * s| < F32_MIN_POSITIVE_NORMAL`) to `+0.0`
    /// in YMM registers to prevent x86 microcode traps over million-token horizons.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `tensor` is empty, `damping_factor <= 0.0`,
    /// or `damping_factor` is non-finite.
    #[inline]
    fn apply_soft_phase_damping(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        damping_factor: f32,
    ) -> LmnkResult<()> {
        if tensor.is_empty() || damping_factor <= 0.0 || !branchless_is_finite_f32(damping_factor) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        scale_f32_slice_unrolled8(tensor.as_mut_slice(), damping_factor);
        Ok(())
    }

    /// Filters sub-grid Q7 phase noise below `min_step_rad` using 8-lane SWAR (`swar_q7_phase_step_filter_x8`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `raw_phases.len() < self.num_heads()`, `min_step_rad < 0.0`,
    /// or `min_step_rad` is non-finite.
    #[inline(always)]
    fn phase_step_filter(
        &self,
        raw_phases: &mut AlignedSliceMut16<'_, PhaseQ7>,
        min_step_rad: f32,
    ) -> LmnkResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || raw_phases.len() < num_heads
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let min_step_q7 = i32::from(branchless_rad_to_phase_step_u8(min_step_rad));
        swar_phase_step_filter_q7_slice_in_place(raw_phases.as_mut_slice(), num_heads, min_step_q7);
        Ok(())
    }

    /// Applies Lazy Phase Unwrapping relative shift directly to a Query tensor in `f32` radians
    /// using 8-head batched AVX2 range-reduced `sincos_f32_x8_simd`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `query_tensor.len() < head_dim * num_heads` or
    /// `accumulated_context_phases.len() < num_heads`.
    #[inline]
    fn lazy_unwrap_override(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        accumulated_context_phases: &AlignedSlice32<'_, f32>,
    ) -> LmnkResult<()> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len || accumulated_context_phases.len() < num_heads {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
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
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `tensor` is empty or `target_norm <= 0.0` /
    /// `damping_alpha < 0.0`, or [`LMNK_ERR_ENTROPY_SINGULARITY`] if `tensor` contains `NaN`/`Inf`.
    #[inline]
    fn slerp_norm_guard_stable(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        target_norm: f32,
        damping_alpha: f32,
    ) -> LmnkResult<()> {
        if tensor.is_empty()
            || target_norm <= 0.0
            || damping_alpha < 0.0
            || !branchless_is_finite_f32(target_norm)
            || !branchless_is_finite_f32(damping_alpha)
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let slice = tensor.as_mut_slice();
        let (current_norm, is_finite) = norm_l2_underflow_resilient_f32(slice);
        if !is_finite {
            return Err(LMNK_ERR_ENTROPY_SINGULARITY);
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
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `logits` is empty or `phase_deltas.len() < self.num_heads()`.
    #[inline]
    fn fused_logit_shift(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
    ) -> LmnkResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty() || phase_deltas.len() < num_heads {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
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

    /// Computes the closed-form **Orthogonal Procrustes Phase Delta** $\Delta\theta^{(h)} \in [-128, 127]$
    /// that aligns each attention head of `source_tensor` with `target_tensor` in $SO(2)^{d_{\mathrm{head}}/2}$
    /// via AVX2 SIMD (`procrustes_q7_slice_simd`, Section 2.1 & 5.2).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `source_tensor` or `target_tensor` has fewer than
    /// `head_dim * num_heads` elements or `phase_deltas_out` has fewer than `num_heads` elements.
    #[inline]
    fn compute_procrustes_phase_delta(
        &self,
        source_tensor: &AlignedSlice32<'_, f32>,
        target_tensor: &AlignedSlice32<'_, f32>,
        phase_deltas_out: &mut AlignedSliceMut16<'_, PhaseQ7>,
    ) -> LmnkResult<()> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if source_tensor.len() < expected_len
            || target_tensor.len() < expected_len
            || phase_deltas_out.len() < num_heads
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
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

    /// Computes per-head Q7 Procrustes phase deltas directly on **HalfSplit (NeoX)** tensors
    /// (`(x[i], x[i + rotary_dim / 2])`) with partial-RoPE boundary `rotary_dim <= head_dim`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `rotary_dim` is invalid or slice lengths are insufficient.
    #[inline]
    fn compute_procrustes_phase_delta_half_split(
        &self,
        source_tensor: &AlignedSlice32<'_, f32>,
        target_tensor: &AlignedSlice32<'_, f32>,
        rotary_dim: usize,
        phase_deltas_out: &mut AlignedSliceMut16<'_, PhaseQ7>,
    ) -> LmnkResult<()> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if rotary_dim < 2
            || (rotary_dim & 1) != 0
            || rotary_dim > head_dim
            || source_tensor.len() < expected_len
            || target_tensor.len() < expected_len
            || phase_deltas_out.len() < num_heads
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        procrustes_q7_half_split_slice(
            source_tensor.as_slice(),
            target_tensor.as_slice(),
            head_dim,
            rotary_dim,
            num_heads,
            phase_deltas_out.as_mut_slice(),
        );
        Ok(())
    }
}

/// Memory pairing convention for Rotary Position Embedding (RoPE) channels.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopePairLayout {
    /// Adjacent interleaved pairs `(x[2i], x[2i + 1])`.
    #[default]
    Interleaved = 0,
    /// Split-half pairs `(x[i], x[i + rotary_dim / 2])` (e.g. NeoX / Hugging Face / Phi-4-mini).
    HalfSplit = 1,
}

impl RopePairLayout {
    /// Converts a C-ABI `u8` layout discriminant into [`RopePairLayout`].
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `raw > 1`.
    #[inline(always)]
    pub const fn from_u8(raw: u8) -> LmnkResult<Self> {
        match raw {
            0 => Ok(Self::Interleaved),
            1 => Ok(Self::HalfSplit),
            _ => Err(LMNK_ERR_INVALID_ARGUMENT),
        }
    }
}

/// Pure mathematical RoPE channel geometry descriptor for [`PhaseSteeringGeometryExt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RopeGeometryDescriptor {
    /// Pairwise rotation layout (`Interleaved` or `HalfSplit`).
    pub layout: RopePairLayout,
    /// Total head dimension (`d_head`, positive even integer).
    pub head_dim: usize,
    /// Active rotated dimensions (`<= head_dim`, positive even integer).
    pub rotary_dim: usize,
    /// Number of heads in the target tensor slice (`1..=MAX_ATTENTION_HEADS`).
    pub num_heads: usize,
}

impl RopeGeometryDescriptor {
    /// Validates RoPE channel geometry invariants.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `head_dim`, `rotary_dim`, or `num_heads`
    /// violate RoPE parity or bounds constraints.
    #[inline(always)]
    pub const fn validate(&self) -> LmnkResult<()> {
        if self.head_dim < 2
            || (self.head_dim & 1) != 0
            || self.rotary_dim < 2
            || (self.rotary_dim & 1) != 0
            || self.rotary_dim > self.head_dim
            || self.num_heads == 0
            || self.num_heads > MAX_ATTENTION_HEADS
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        Ok(())
    }
}

/// Trait implemented by any model or runtime descriptor that exposes a [`RopeGeometryDescriptor`].
pub trait RopeSteeringTarget {
    /// Returns the validated RoPE channel geometry for this target.
    fn rope_geometry(&self) -> RopeGeometryDescriptor;
}

impl RopeSteeringTarget for RopeGeometryDescriptor {
    #[inline(always)]
    fn rope_geometry(&self) -> RopeGeometryDescriptor {
        *self
    }
}

/// Blanket-implemented geometry-aware RoPE steering extension for any [`PhaseSteering`] engine.
///
/// Unifies `Interleaved` and `HalfSplit` (plus partial `rotary_dim`) dispatch behind a single
/// zero-allocation trait interface shared by both Rust adapters and the C-ABI boundary.
pub trait PhaseSteeringGeometryExt: PhaseSteering {
    /// Applies soft-bounded (`PhaseClamp`) phase steering according to `target.rope_geometry()`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `target` geometry, `max_rad`, or slice lengths
    /// are invalid.
    fn apply_phase_steering_for_geometry_safe<T: RopeSteeringTarget + ?Sized>(
        &self,
        target: &T,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        max_rad: f32,
    ) -> LmnkResult<()>;

    /// Recovers per-head Q7 Procrustes phase deltas according to `target.rope_geometry()`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `target` geometry or slice lengths are invalid.
    fn compute_procrustes_phase_delta_for_geometry<T: RopeSteeringTarget + ?Sized>(
        &self,
        target: &T,
        q_before: &AlignedSlice32<'_, f32>,
        q_after: &AlignedSlice32<'_, f32>,
        q7_phase_out: &mut AlignedSliceMut16<'_, PhaseQ7>,
    ) -> LmnkResult<()>;
}

impl<E: PhaseSteering + ?Sized> PhaseSteeringGeometryExt for E {
    #[inline]
    fn apply_phase_steering_for_geometry_safe<T: RopeSteeringTarget + ?Sized>(
        &self,
        target: &T,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, PhaseQ7>,
        max_rad: f32,
    ) -> LmnkResult<()> {
        let geom = target.rope_geometry();
        geom.validate()?;
        if geom.head_dim != self.head_dim() as usize || geom.num_heads != self.num_heads() as usize
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        match geom.layout {
            RopePairLayout::Interleaved => {
                self.apply_phase_steering_safe(query_tensor, phase_deltas, max_rad)
            }
            RopePairLayout::HalfSplit => self.apply_phase_steering_half_split_safe(
                query_tensor,
                phase_deltas,
                geom.rotary_dim,
                max_rad,
            ),
        }
    }

    #[inline]
    fn compute_procrustes_phase_delta_for_geometry<T: RopeSteeringTarget + ?Sized>(
        &self,
        target: &T,
        q_before: &AlignedSlice32<'_, f32>,
        q_after: &AlignedSlice32<'_, f32>,
        q7_phase_out: &mut AlignedSliceMut16<'_, PhaseQ7>,
    ) -> LmnkResult<()> {
        let geom = target.rope_geometry();
        geom.validate()?;
        if geom.head_dim != self.head_dim() as usize || geom.num_heads != self.num_heads() as usize
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        match geom.layout {
            RopePairLayout::Interleaved => {
                self.compute_procrustes_phase_delta(q_before, q_after, q7_phase_out)
            }
            RopePairLayout::HalfSplit => self.compute_procrustes_phase_delta_half_split(
                q_before,
                q_after,
                geom.rotary_dim,
                q7_phase_out,
            ),
        }
    }
}
