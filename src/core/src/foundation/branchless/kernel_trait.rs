// name: src/core/src/foundation/branchless/kernel_trait.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`BranchlessKernel`] blanket trait on top of [`Liminal`].

use super::phase_field::branchless_project_f32_slice_to_q7_64;
use crate::foundation::scalar_math::{
    branchless_phase_clamp_rad, branchless_sincos_half_pi_f32, branchless_von_mises_kappa,
};
use crate::liminal::{gcso_q7_t, Liminal, MAX_ATTENTION_HEADS};

/// Blanket trait exposing core soft phase clamping, sinusoidal, concentration, and $SO(2)^{64}$
/// projection operations on top of [`Liminal`].
pub trait BranchlessKernel: Liminal {
    /// Branchless soft-bounded phase clamp.
    #[inline(always)]
    fn phase_clamp_rad(&self, delta_rad: f32, max_rad: f32) -> f32 {
        branchless_phase_clamp_rad(delta_rad, max_rad)
    }

    /// Branchless direct $(\sin\theta, \cos\theta)$ for angles bounded in $[-\pi/2, \pi/2]$.
    #[inline(always)]
    fn sincos_half_pi_f32(&self, theta_rad: f32) -> (f32, f32) {
        branchless_sincos_half_pi_f32(theta_rad)
    }

    /// Branchless von Mises concentration parameter $\kappa_c$ from $\bar{R}_c \in [0, 1)$.
    #[inline(always)]
    fn von_mises_kappa(&self, r_bar: f32) -> f32 {
        branchless_von_mises_kappa(r_bar)
    }

    /// Projects an `&[f32]` feature/embedding slice onto $SO(2)^{64}$ (`[gcso_q7_t; 64]`).
    #[inline(always)]
    fn project_f32_slice_to_q7_64(
        &self,
        values: &[f32],
        gain: f32,
    ) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        branchless_project_f32_slice_to_q7_64(values, gain)
    }
}

impl<T: ?Sized> BranchlessKernel for T {}
