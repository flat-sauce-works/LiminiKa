// name: src/core/src/entropy_control/closed_loop.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Blanket-implemented [`EntropyClosedLoopCoordinator`] over [`EdbcBranchController`].

use super::kernels::*;
use super::*;

/// Blanket-implemented closed-loop coordinator over [`EdbcBranchController`].
///
/// Automatically implemented for all `T: EdbcBranchController + ?Sized`
/// (including [`EntropyController`] and [`crate::RuntimeContext`]).
pub trait EntropyClosedLoopCoordinator: EdbcBranchController {
    /// Evaluates Top-2 sparse surrogate entropy via the branchless `0-ln` Taylor-Horner
    /// polynomial, combines it with inter-layer cosine distance into Moving Z-Score Normalized
    /// Attention Entropy $\tilde{H}$, and steps the EDBC controller via
    /// [`EntropyEvaluator::eval_stateful`].
    ///
    /// # Errors
    /// Propagates [`GCSO_ERROR_EDBC_SINGULARITY`] if computed entropy is non-finite.
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

impl<T: EdbcBranchController + ?Sized> EntropyClosedLoopCoordinator for T {}
