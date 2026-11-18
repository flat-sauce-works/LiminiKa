// name: src/core/src/phase_steering/closed_loop.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Blanket-implemented [`PhaseSteeringClosedLoop`] over [`PhaseSteering`].

use super::*;

/// Blanket-implemented closed-loop steering trait extending [`PhaseSteering`]
/// for direct 128-byte [`GatewayFrame`] synthesis and $SO(2)$-fused RoPE steering.
///
/// Automatically implemented for all `T: PhaseSteering + ?Sized` (including
/// [`PhaseSteeringEngine`], [`crate::traits::ActionHub`], and [`crate::RuntimeContext`]).
pub trait PhaseSteeringClosedLoop: PhaseSteering {
    /// Synthesizes `signal` into a 128-byte [`GatewayFrame`] and [`CausalProvenanceReport`] via
    /// [`ManifoldGatewaySignal::synthesize_gateway_frame_with_words`], synchronizes `trail`,
    /// executes $SO(2)$-fused RIPA + sub-head router steering on `query_tensor` (and optional
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
