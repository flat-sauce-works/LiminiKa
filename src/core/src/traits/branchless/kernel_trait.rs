// name: src/core/src/traits/branchless/kernel_trait.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Root [`Liminal`] epistemic boundary supertrait and [`BranchlessKernel`] blanket trait.

use super::phase_field::branchless_project_f32_slice_to_q7_64;
use super::*;

/// Root epistemic supertrait representing the boundary separating cognizable state from the
/// unknowable exterior.
///
/// Every capability and signal trait in LiminiKa refines a specific cognitive aperture mask out of
/// `Liminal`. Any axis outside a trait's aperture (`!aperture_mask`) belongs strictly to the
/// unknowable exterior.
pub trait Liminal {
    /// Complete bitmask representing the unbounded unknowable exterior prior to any cognitive cut.
    const ROOT_UNKNOWABLE_VOID_MASK: u32 = u32::MAX;

    /// Partitions `world_bits` into `(cognized_bits, unknowable_void_bits)` strictly branchlessly
    /// for a trait's declared `aperture_mask`.
    #[inline(always)]
    fn partition_cognitive_and_unknowable_bits(
        &self,
        aperture_mask: u32,
        world_bits: u32,
    ) -> (u32, u32) {
        (world_bits & aperture_mask, world_bits & !aperture_mask)
    }
}

impl<T: ?Sized> Liminal for T {}

/// Blanket trait exposing core RIPA, sinusoidal, concentration, and $SO(2)^{64}$ projection
/// operations on top of [`Liminal`].
pub trait BranchlessKernel: Liminal {
    /// Branchless RIPA soft-bounded phase clamp.
    #[inline(always)]
    fn ripa_clamp_rad(&self, delta_rad: f32, max_rad: f32) -> f32 {
        branchless_ripa_clamp_rad(delta_rad, max_rad)
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
