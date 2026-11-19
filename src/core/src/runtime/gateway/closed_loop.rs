// name: src/core/src/runtime/gateway/closed_loop.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Closed-loop steering ([`PhaseSteeringClosedLoop`]) and entropy gating
//! ([`EntropyClosedLoopCoordinator`]) trait extensions.

use super::*;
use crate::drivers::spine::entropy::kernels::{
    compute_moving_z_score_entropy, compute_sparse_surrogate_entropy,
};
use crate::drivers::DEFAULT_SINGULARITY_EPS;

// ===================================================================
// Internal Constant-Time Register-Resident SWAR & SO(2) Fusion Kernels
// ===================================================================

/// Validates `(head_dim, num_heads)`, `max_rad > 0.0`, and `query_tensor` / `key_tensor` slice
/// bounds and `GCSO_RESTRICT` non-aliasing invariants for closed-loop steering methods.
#[inline(always)]
pub(super) fn validate_closed_loop_tensors<T: PhaseSteering + ?Sized>(
    engine: &T,
    max_rad: f32,
    query_tensor: &AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(usize, usize)> {
    if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let (head_dim, num_heads, expected_len) = engine.validate_head_layout()?;
    if num_heads > MAX_ATTENTION_HEADS || query_tensor.len() < expected_len {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if let Some(k) = key_tensor {
        if k.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        query_tensor.ensure_disjoint_mut(k)?;
    }
    Ok((head_dim, num_heads))
}

/// Internal register-resident Gateway steering core returning
/// `(GatewayFrame, CausalProvenanceReport, frame_words_64)`.
#[inline(always)]
pub(super) fn steer_gateway_signal_core_safe<
    T: PhaseSteering + ?Sized,
    S: ManifoldGatewaySignal + ?Sized,
>(
    engine: &T,
    signal: &S,
    trail: &mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: &mut AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(
    GatewayFrame,
    CausalProvenanceReport,
    [u64; NUM_STIGMERGIC_GRANULES],
)> {
    let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
    let report = frame.to_causal_report();
    let (head_dim, num_heads) = validate_closed_loop_tensors(
        engine,
        frame.ripa_clamp_rad,
        query_tensor,
        key_tensor.as_deref(),
    )?;

    trail.prev_ptr = trail.current_ptr;
    trail.current_ptr = frame.tagged_ptr;
    let has_frame_anchor = frame.target_anchor_id != 0;
    trail.target_anchor_id = branchless_select_u32(
        has_frame_anchor,
        frame.target_anchor_id,
        trail.target_anchor_id,
    );
    trail.cluster_id = branchless_select_u32(
        has_frame_anchor,
        frame.target_anchor_id,
        frame.dominant_slot as u32,
    );

    trail.stigmergic_density =
        branchless_min_f32(trail.stigmergic_density + 0.25, MAX_STIGMERGIC_DENSITY);
    trail.step_count = trail.step_count.wrapping_add(1);

    let is_repulsed = frame.is_repulsed();
    let rep_flags = trail.flags | TRAIL_FLAG_REPULSION_ACTIVE;
    let coh_flags = (trail.flags | TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE)
        & !TRAIL_FLAG_REPULSION_ACTIVE;
    trail.flags = branchless_select_u32(is_repulsed, rep_flags, coh_flags);

    let _ = accumulate_trail_and_steer_gateway_words_core(
        &frame,
        &frm_w,
        trail,
        unresolved_damping_u7,
        head_dim,
        num_heads,
        Some(query_tensor),
        key_tensor,
        logits_opt,
    )?;

    Ok((frame, report, frm_w))
}

/// Blanket-implemented closed-loop steering trait extending [`PhaseSteering`]
/// for direct 128-byte [`GatewayFrame`] synthesis and $SO(2)$-fused RoPE steering.
///
/// Automatically implemented for all `T: PhaseSteering + ?Sized` (including
/// [`PhaseSteeringEngine`], [`ActionHub`], and [`crate::runtime::RuntimeContext`]).
pub trait PhaseSteeringClosedLoop: PhaseSteering {
    /// Synthesizes `signal` into a 128-byte [`GatewayFrame`] and [`CausalProvenanceReport`] via
    /// [`ManifoldGatewaySignal::synthesize_gateway_frame_with_words`], synchronizes `trail`,
    /// executes $SO(2)$-fused `PhaseClamp` + sub-head router steering on `query_tensor` (and optional
    /// `key_tensor`), and optionally scales `logits_opt` by `frame.effective_inv_temp`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor dimensions or `GCSO_RESTRICT` non-aliasing
    /// invariants are violated.
    #[inline(always)]
    fn steer_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
        let (frame, report, _frm_w) = steer_gateway_signal_core_safe(
            self,
            signal,
            trail,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            logits_opt,
        )?;
        Ok((frame, report))
    }
}

impl<T: PhaseSteering + ?Sized> PhaseSteeringClosedLoop for T {}

/// Blanket-implemented closed-loop coordinator over [`DecodingGate`].
///
/// Automatically implemented for all `T: DecodingGate + ?Sized`
/// (including [`EntropyController`] and [`crate::runtime::RuntimeContext`]).
pub trait EntropyClosedLoopCoordinator: DecodingGate {
    /// Evaluates Top-2 sparse surrogate entropy via the branchless `0-ln` Taylor-Horner
    /// polynomial, combines it with inter-layer cosine distance into Moving Z-Score Normalized
    /// Attention Entropy $\tilde{H}$, and steps the `EntropyController` via
    /// [`EntropyEvaluator::eval_stateful`].
    ///
    /// # Errors
    /// Propagates [`crate::liminal::GCSO_ERROR_EDBC_SINGULARITY`] if computed entropy is non-finite.
    #[inline(always)]
    fn eval_surrogate_and_step(
        &mut self,
        p_max: f32,
        p_2nd: f32,
        cosine_dist: f32,
        mu_dist: f32,
        sigma_dist: f32,
        alpha_z: f32,
        gamma_mod: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> GcsoResult<f32> {
        let eps = branchless_select_f32(
            (state_out.singularity_eps > 0.0) & branchless_is_finite_f32(state_out.singularity_eps),
            state_out.singularity_eps,
            DEFAULT_SINGULARITY_EPS,
        );
        let h_surr = compute_sparse_surrogate_entropy(p_max, p_2nd, eps);
        let h_tilde = compute_moving_z_score_entropy(
            h_surr,
            cosine_dist,
            mu_dist,
            sigma_dist,
            eps,
            alpha_z,
            gamma_mod,
        );
        self.eval_stateful(h_tilde, state_out)?;
        Ok(state_out.moving_z_entropy)
    }
}

impl<T: DecodingGate + ?Sized> EntropyClosedLoopCoordinator for T {}
