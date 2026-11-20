// name: src/core/src/foundation/rope/engine.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compact 16B [Pod] [PhaseSteeringEngine] and core RoPE trait delegations.

use super::*;
use crate::liminal::{Active, ActiveScope, Scoped};

// ===================================================================
// 2. Primary Phase-Steering & Sub-Head Routing Engine (`16B` Pod)
// ===================================================================

/// Primary dynamic phase-shifted RoPE execution engine (`PhaseSteering`, `PhaseClamp`, `PhaseStepFilter`) (`16B` aligned, [`Pod`]).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSteeringEngine {
    /// Dimension per attention head ($d_{\mathrm{head}}$, positive even integer).
    pub(crate) head_dim: u32,
    /// Total number of attention heads ($H > 0$).
    pub(crate) num_heads: u32,
    /// Default soft-bounded (`PhaseClamp`) limit in radians ($\theta_{\max}$).
    pub(crate) phase_clamp_max_rad: f32,
    /// Default `PhaseStepFilter` minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    pub(crate) phase_step_min_rad: f32,
}

// SAFETY: `PhaseSteeringEngine` is `#[repr(C, align(16))]` containing four 32-bit primitive fields totaling 16B.
unsafe impl Pod for PhaseSteeringEngine {}

const _: () = {
    assert!(size_of::<PhaseSteeringEngine>() == 16);
    assert!(align_of::<PhaseSteeringEngine>() == 16);
    assert!(offset_of!(PhaseSteeringEngine, head_dim) == 0);
    assert!(offset_of!(PhaseSteeringEngine, num_heads) == 4);
    assert!(offset_of!(PhaseSteeringEngine, phase_clamp_max_rad) == 8);
    assert!(offset_of!(PhaseSteeringEngine, phase_step_min_rad) == 12);
    assert!(NUM_STIGMERGIC_GRANULES * HEADS_PER_STIGMERGIC_GRANULE == MAX_ATTENTION_HEADS);
};

impl Default for PhaseSteeringEngine {
    #[inline(always)]
    fn default() -> Self {
        Self {
            head_dim: DEFAULT_PHASE_STEER_HEAD_DIM,
            num_heads: DEFAULT_PHASE_STEER_NUM_HEADS,
            phase_clamp_max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
        }
    }
}

impl PhaseSteeringEngine {
    /// Creates a new [`PhaseSteeringEngine`] instance with validated `(head_dim, num_heads)` configuration.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `num_heads` is zero,
    /// or `head_dim * num_heads` overflows `usize`.
    #[inline(always)]
    pub const fn new(head_dim: u32, num_heads: u32) -> GcsoResult<Self> {
        let valid = (head_dim != 0) & ((head_dim & 1) == 0) & (num_heads != 0);
        if !valid {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if (head_dim as usize)
            .checked_mul(num_heads as usize)
            .is_none()
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(Self {
            head_dim,
            num_heads,
            phase_clamp_max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
        })
    }

    /// Constructs a [`PhaseSteeringEngine`] from a C-ABI [`gcso_config_t`] descriptor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `num_heads` is zero,
    /// `ripa_clamp_max_rad <= 0.0`, `qdps_min_step_rad < 0.0`, or thresholds are non-finite.
    #[inline(always)]
    pub const fn from_config(config: &gcso_config_t) -> GcsoResult<Self> {
        let mut engine = match Self::new(config.head_dim, config.num_heads) {
            Ok(e) => e,
            Err(err) => return Err(err),
        };
        let valid_ripa =
            branchless_is_finite_f32(config.ripa_clamp_max_rad) & (config.ripa_clamp_max_rad > 0.0);
        let valid_qdps =
            branchless_is_finite_f32(config.qdps_min_step_rad) & (config.qdps_min_step_rad >= 0.0);
        if !(valid_ripa & valid_qdps) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        engine.phase_clamp_max_rad = config.ripa_clamp_max_rad;
        engine.phase_step_min_rad = config.qdps_min_step_rad;
        Ok(engine)
    }

    /// Returns the configured soft-bounded (`PhaseClamp`) limit in radians ($\theta_{\max}$).
    #[inline(always)]
    pub const fn phase_clamp_max_rad(&self) -> f32 {
        self.phase_clamp_max_rad
    }

    /// Returns the configured `PhaseStepFilter` minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    #[inline(always)]
    pub const fn phase_step_min_rad(&self) -> f32 {
        self.phase_step_min_rad
    }

    /// Returns a copy of `self` with an updated `PhaseClamp` soft-clamping limit (`ripa_clamp_max_rad`).
    #[inline(always)]
    pub const fn with_phase_clamp_max_rad(mut self, max_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(max_rad) & (max_rad > 0.0);
        self.phase_clamp_max_rad = branchless_select_f32(valid, max_rad, self.phase_clamp_max_rad);
        self
    }

    /// Returns a copy of `self` with an updated `PhaseStepFilter` step cutoff threshold (`qdps_min_step_rad`).
    #[inline(always)]
    pub const fn with_phase_step_min_rad(mut self, min_step_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(min_step_rad) & (min_step_rad >= 0.0);
        self.phase_step_min_rad =
            branchless_select_f32(valid, min_step_rad, self.phase_step_min_rad);
        self
    }

    /// Returns the total number of `f32` elements expected per single-token Query/Key/Value tensor.
    #[inline(always)]
    pub const fn expected_tensor_len(&self) -> usize {
        (self.head_dim as usize) * (self.num_heads as usize)
    }

    /// Non-generic [`Active`] closure entry point for [`PhaseSteering::apply_phase_steering`].
    ///
    /// Keeps the underlying AVX2/scalar RoPE kernel strictly non-generic (`0 B` L1I duplication)
    /// while statically excluding [`crate::liminal::Unmarked`], [`crate::liminal::Bottom`], and
    /// unclassified origins at compile time.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths are insufficient.
    #[inline(always)]
    pub fn apply_phase_steering_active(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: Scoped<Active, &AlignedSlice16<'_, gcso_q7_t>>,
    ) -> GcsoResult<()> {
        self.apply_phase_steering(tensor, phase_deltas.active_ref())
    }

    /// Zero-cost closure shell folding any [`ActiveScope`] refinement into [`Active`] and
    /// delegating to [`Self::apply_phase_steering_active`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths are insufficient.
    #[inline(always)]
    pub fn apply_phase_steering_scoped<B: ActiveScope>(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: Scoped<B, &AlignedSlice16<'_, gcso_q7_t>>,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_active(tensor, phase_deltas.seal_active())
    }

    /// Non-generic [`Active`] closure entry point for
    /// [`PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if geometry or slice lengths are invalid.
    #[inline(always)]
    pub fn apply_phase_steering_for_geometry_safe_active<G: RopeSteeringTarget + ?Sized>(
        &self,
        geometry: &G,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: Scoped<Active, &AlignedSlice16<'_, gcso_q7_t>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_for_geometry_safe(
            geometry,
            tensor,
            phase_deltas.active_ref(),
            max_rad,
        )
    }

    /// Zero-cost closure shell folding any [`ActiveScope`] refinement into [`Active`] and
    /// delegating to [`Self::apply_phase_steering_for_geometry_safe_active`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if geometry or slice lengths are invalid.
    #[inline(always)]
    pub fn apply_phase_steering_for_geometry_safe_scoped<
        B: ActiveScope,
        G: RopeSteeringTarget + ?Sized,
    >(
        &self,
        geometry: &G,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: Scoped<B, &AlignedSlice16<'_, gcso_q7_t>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_for_geometry_safe_active(
            geometry,
            tensor,
            phase_deltas.seal_active(),
            max_rad,
        )
    }
}

// ===================================================================
// 3. Minimal Trait Delegations (Inheriting All `rope.rs` Default Implementations)
// ===================================================================

impl PhaseSteering for PhaseSteeringEngine {
    #[inline(always)]
    fn head_dim(&self) -> u32 {
        self.head_dim
    }

    #[inline(always)]
    fn num_heads(&self) -> u32 {
        self.num_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liminal::ScopedCut;

    #[repr(align(32))]
    struct Align32<T>(T);

    #[test]
    fn test_active_closure_gate_matches_non_generic_rope_kernel_bit_exact() {
        let engine = PhaseSteeringEngine::new(16, 4).unwrap();
        let mut raw_buf = Align32([0.0_f32; 64]);
        let mut gated_buf = Align32([0.0_f32; 64]);
        for (i, slot) in raw_buf.0.iter_mut().enumerate() {
            *slot = ((i as f32) * 0.13) - 2.0;
        }
        gated_buf.0 = raw_buf.0;

        let deltas_raw = Align32([12_i8, -18, 27, -9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let deltas_slice = AlignedSlice16::new(&deltas_raw.0[..4]).unwrap();

        {
            let mut raw_mut = AlignedSliceMut32::new(&mut raw_buf.0).unwrap();
            engine
                .apply_phase_steering(&mut raw_mut, &deltas_slice)
                .unwrap();
        }
        {
            let mut gated_mut = AlignedSliceMut32::new(&mut gated_buf.0).unwrap();
            let scoped = match Scoped::from_boundary(&deltas_slice).cut_by_lacuna_q7(true, 12, 4) {
                ScopedCut::Exterior(ext) => ext,
                _ => panic!("expected exterior active cut"),
            };
            engine
                .apply_phase_steering_scoped(&mut gated_mut, scoped)
                .unwrap();
        }

        for (lhs, rhs) in raw_buf.0.iter().zip(gated_buf.0.iter()) {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
    }
}
