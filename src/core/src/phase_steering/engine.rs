// name: src/core/src/phase_steering/engine.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compact 16B [Pod] [PhaseSteeringEngine] and core RoPE trait delegations.

#[allow(unused_imports)]
use super::closed_loop::*;
use super::*;

// ===================================================================
// 2. Primary DPSR Phase-Steering & Sub-Head Routing Engine (`16B` Pod)
// ===================================================================

/// Primary Dynamic Phase-Shifted RoPE (DPSR), RIPA, and QDPS execution engine (`16B` aligned, [`Pod`]).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSteeringEngine {
    /// Dimension per attention head ($d_{\mathrm{head}}$, positive even integer).
    pub(super) head_dim: u32,
    /// Total number of attention heads ($H > 0$).
    pub(super) num_heads: u32,
    /// Default RIPA soft-bounded phase clamp limit in radians ($\theta_{\max}$).
    pub(super) ripa_clamp_max_rad: f32,
    /// Default QDPS minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    pub(super) qdps_min_step_rad: f32,
}

// SAFETY: `PhaseSteeringEngine` is `#[repr(C, align(16))]` containing four 32-bit primitive fields totaling 16B.
unsafe impl Pod for PhaseSteeringEngine {}

const _: () = {
    assert!(size_of::<PhaseSteeringEngine>() == 16);
    assert!(align_of::<PhaseSteeringEngine>() == 16);
    assert!(offset_of!(PhaseSteeringEngine, head_dim) == 0);
    assert!(offset_of!(PhaseSteeringEngine, num_heads) == 4);
    assert!(offset_of!(PhaseSteeringEngine, ripa_clamp_max_rad) == 8);
    assert!(offset_of!(PhaseSteeringEngine, qdps_min_step_rad) == 12);
    assert!(NUM_STIGMERGIC_GRANULES * HEADS_PER_STIGMERGIC_GRANULE == MAX_ATTENTION_HEADS);
};

impl Default for PhaseSteeringEngine {
    #[inline(always)]
    fn default() -> Self {
        Self {
            head_dim: DEFAULT_DPSR_HEAD_DIM,
            num_heads: DEFAULT_DPSR_NUM_HEADS,
            ripa_clamp_max_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
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
            ripa_clamp_max_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
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
        engine.ripa_clamp_max_rad = config.ripa_clamp_max_rad;
        engine.qdps_min_step_rad = config.qdps_min_step_rad;
        Ok(engine)
    }

    /// Returns the configured RIPA soft-bounded phase clamp limit in radians ($\theta_{\max}$).
    #[inline(always)]
    pub const fn ripa_clamp_max_rad(&self) -> f32 {
        self.ripa_clamp_max_rad
    }

    /// Returns the configured QDPS minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    #[inline(always)]
    pub const fn qdps_min_step_rad(&self) -> f32 {
        self.qdps_min_step_rad
    }

    /// Returns a copy of `self` with an updated RIPA soft-clamping limit (`ripa_clamp_max_rad`).
    #[inline(always)]
    pub const fn with_ripa_clamp_max_rad(mut self, max_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(max_rad) & (max_rad > 0.0);
        self.ripa_clamp_max_rad = branchless_select_f32(valid, max_rad, self.ripa_clamp_max_rad);
        self
    }

    /// Returns a copy of `self` with an updated QDPS step cutoff threshold (`qdps_min_step_rad`).
    #[inline(always)]
    pub const fn with_qdps_min_step_rad(mut self, min_step_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(min_step_rad) & (min_step_rad >= 0.0);
        self.qdps_min_step_rad = branchless_select_f32(valid, min_step_rad, self.qdps_min_step_rad);
        self
    }

    /// Returns the total number of `f32` elements expected per single-token Query/Key/Value tensor.
    #[inline(always)]
    pub const fn expected_tensor_len(&self) -> usize {
        (self.head_dim as usize) * (self.num_heads as usize)
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

impl SubHeadRouter for PhaseSteeringEngine {}

impl VisualAttentionSteering for PhaseSteeringEngine {}
