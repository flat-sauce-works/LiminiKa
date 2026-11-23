// name: src/core/src/runtime/context/config.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safe high-level configuration wrapper ([`Config`]) and fluent builder.

use super::{
    Result, DEFAULT_BLOOM_REPULSION_SCALE_U7, DEFAULT_BLOOM_REPULSION_THRESHOLD,
    DEFAULT_ENTROPY_GATE_BIFURCATION_THRESHOLD, DEFAULT_ENTROPY_GATE_HYSTERESIS_BAND,
    DEFAULT_ENTROPY_GATE_TAU_FLUX, DEFAULT_PAGED_BLOCK_SIZE,
};
use crate::foundation::{
    branchless_is_finite_f32, DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEP_MIN_RAD,
};
use crate::liminal::{
    RuntimeConfig, DEFAULT_ENTROPY_EPS, LMNK_ERR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS,
    Q7_PHASE_SCALE_RAD,
};
use crate::stigmergy::{
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, DEFAULT_SUBSTRATE_CAPACITY, MAX_PHASE_ANCHORS,
};

// ===================================================================
// Safe High-Level Rust Configuration Wrapper & Fluent Builder
// ===================================================================

/// Safe high-level configuration wrapper for LiminiKa runtime initialization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Inner raw C-ABI configuration structure (64 bytes, 64-byte aligned).
    pub raw: RuntimeConfig,
    /// Hysteresis band ($\delta_{\mathrm{hyst}}$) for `EntropyController` branch control stabilization.
    pub hysteresis_band: f32,
    /// Tau flux threshold ($\tau_{\mathrm{flux}}$) for sliding-window entropy rate integration.
    pub tau_flux: f32,
    /// Z-score threshold ($\tau_{\mathrm{bifurcation}}$) triggering `EntropyController` pitchfork bifurcation.
    pub bifurcation_threshold: f32,
    /// Minimum Bloom filter counter density (`1..=15`) required to trigger anti-phase repulsion.
    pub repulsion_threshold: u8,
    /// Q7 unit scale (`0..=128`, where `128 == 1.0`) applied during phase-conjugate repulsion.
    pub repulsion_scale_u7: u8,
    /// 64-bit ($8 \times 8$) coupling bitmatrix governing cross-lane attractor synergy.
    pub slot_coupling_matrix: u64,
}

impl Default for Config {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Creates a new default configuration instance adhering to LiminiKa architectural invariants.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            raw: RuntimeConfig {
                head_dim: 128,
                num_heads: 32,
                paged_block_size: DEFAULT_PAGED_BLOCK_SIZE,
                q7_phase_scale: Q7_PHASE_SCALE_RAD,
                phase_clamp_max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
                phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
                entropy_singularity_eps: DEFAULT_ENTROPY_EPS,
                max_prompt_anchors: MAX_PHASE_ANCHORS as u32,
                substrate_capacity: DEFAULT_SUBSTRATE_CAPACITY,
                execution_flags: 0,
                enable_zero_alloc_strict: 1,
                ledger_mode: 0,
                rotary_dim: 0,
                rope_layout: 0,
                reserved: [0; 20],
            },
            hysteresis_band: DEFAULT_ENTROPY_GATE_HYSTERESIS_BAND,
            tau_flux: DEFAULT_ENTROPY_GATE_TAU_FLUX,
            bifurcation_threshold: DEFAULT_ENTROPY_GATE_BIFURCATION_THRESHOLD,
            repulsion_threshold: DEFAULT_BLOOM_REPULSION_THRESHOLD,
            repulsion_scale_u7: DEFAULT_BLOOM_REPULSION_SCALE_U7,
            slot_coupling_matrix: DEFAULT_SLOT_COUPLING_MATRIX_8X8,
        }
    }

    /// Constructs a high-level [`Config`] wrapper from a raw [`RuntimeConfig`] descriptor.
    #[inline(always)]
    #[must_use]
    pub const fn from_raw(raw: RuntimeConfig) -> Self {
        Self {
            raw,
            hysteresis_band: DEFAULT_ENTROPY_GATE_HYSTERESIS_BAND,
            tau_flux: DEFAULT_ENTROPY_GATE_TAU_FLUX,
            bifurcation_threshold: DEFAULT_ENTROPY_GATE_BIFURCATION_THRESHOLD,
            repulsion_threshold: DEFAULT_BLOOM_REPULSION_THRESHOLD,
            repulsion_scale_u7: DEFAULT_BLOOM_REPULSION_SCALE_U7,
            slot_coupling_matrix: DEFAULT_SLOT_COUPLING_MATRIX_8X8,
        }
    }

    /// Sets the attention head dimension ($d_{\mathrm{head}}$). Must be a positive even integer.
    #[inline(always)]
    #[must_use]
    pub const fn with_head_dim(mut self, head_dim: u32) -> Self {
        self.raw.head_dim = head_dim;
        self
    }

    /// Sets the total number of attention heads ($1 \le H \le 64$).
    #[inline(always)]
    #[must_use]
    pub const fn with_num_heads(mut self, num_heads: u32) -> Self {
        self.raw.num_heads = num_heads;
        self
    }

    /// Sets the PagedBlock token granularity (`16` or `32` tokens).
    #[inline(always)]
    #[must_use]
    pub const fn with_paged_block_size(mut self, paged_block_size: u32) -> Self {
        self.raw.paged_block_size = paged_block_size;
        self
    }

    /// Sets the Q7 phase scale factor ($\beta_{\mathrm{Q7}}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_q7_phase_scale(mut self, q7_phase_scale: f32) -> Self {
        self.raw.q7_phase_scale = q7_phase_scale;
        self
    }

    /// Sets the soft-bounded (`PhaseClamp`) limit in radians ($\theta_{\max}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_phase_clamp_max_rad(mut self, phase_clamp_max_rad: f32) -> Self {
        self.raw.phase_clamp_max_rad = phase_clamp_max_rad;
        self
    }

    /// Sets the `PhaseStepFilter` quantization threshold angle in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_phase_step_min_rad(mut self, phase_step_min_rad: f32) -> Self {
        self.raw.phase_step_min_rad = phase_step_min_rad;
        self
    }

    /// Sets the logarithmic singularity prevention epsilon ($\epsilon_{\mathrm{log}}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_entropy_singularity_eps(mut self, entropy_singularity_eps: f32) -> Self {
        self.raw.entropy_singularity_eps = entropy_singularity_eps;
        self
    }

    /// Sets the maximum supported prompt anchor points in the phase-anchor field.
    #[inline(always)]
    #[must_use]
    pub const fn with_max_prompt_anchors(mut self, max_prompt_anchors: u32) -> Self {
        self.raw.max_prompt_anchors = max_prompt_anchors;
        self
    }

    /// Sets the [`crate::stigmergy::StigmergicSubstrate`] pointer table slot capacity.
    #[inline(always)]
    #[must_use]
    pub const fn with_substrate_capacity(mut self, substrate_capacity: u32) -> Self {
        self.raw.substrate_capacity = substrate_capacity;
        self
    }

    /// Returns the configured [`crate::stigmergy::StigmergicSubstrate`] pointer table slot capacity.
    #[inline(always)]
    #[must_use]
    pub const fn substrate_capacity(&self) -> u32 {
        self.raw.substrate_capacity
    }

    /// Configures vendor-neutral runtime execution flags (`0` = default).
    #[inline(always)]
    #[must_use]
    pub const fn with_execution_flags(mut self, execution_flags: u8) -> Self {
        self.raw.execution_flags = execution_flags;
        self
    }

    /// Configures strict zero dynamic allocation enforcement on the Hot Path (`1` = strict).
    #[inline(always)]
    #[must_use]
    pub const fn with_enable_zero_alloc_strict(mut self, enable_zero_alloc_strict: u8) -> Self {
        self.raw.enable_zero_alloc_strict = enable_zero_alloc_strict;
        self
    }

    /// Configures FastPathLedger operating mode (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC).
    #[inline(always)]
    #[must_use]
    pub const fn with_ledger_mode(mut self, ledger_mode: u8) -> Self {
        self.raw.ledger_mode = ledger_mode;
        self
    }

    /// Sets the `EntropyController` hysteresis band width ($\delta_{\mathrm{hyst}}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_hysteresis_band(mut self, hysteresis_band: f32) -> Self {
        self.hysteresis_band = hysteresis_band;
        self
    }

    /// Sets the sliding-window entropy rate tau flux threshold ($\tau_{\mathrm{flux}}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_tau_flux(mut self, tau_flux: f32) -> Self {
        self.tau_flux = tau_flux;
        self
    }

    /// Sets the `EntropyController` pitchfork bifurcation threshold ($\tau_{\mathrm{bifurcation}}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_bifurcation_threshold(mut self, bifurcation_threshold: f32) -> Self {
        self.bifurcation_threshold = bifurcation_threshold;
        self
    }

    /// Sets the Contrastive Phase Bloom Filter density threshold (`1..=15`) for anti-phase repulsion.
    #[inline(always)]
    #[must_use]
    pub const fn with_repulsion_threshold(mut self, repulsion_threshold: u8) -> Self {
        self.repulsion_threshold = repulsion_threshold;
        self
    }

    /// Sets the Q7 unit scale (`0..=128`) applied during phase-conjugate repulsion.
    #[inline(always)]
    #[must_use]
    pub const fn with_repulsion_scale_u7(mut self, repulsion_scale_u7: u8) -> Self {
        self.repulsion_scale_u7 = repulsion_scale_u7;
        self
    }

    /// Sets the 64-bit ($8 \times 8$) coupling bitmatrix for cross-lane attractor synergy.
    #[inline(always)]
    #[must_use]
    pub const fn with_slot_coupling_matrix(mut self, slot_coupling_matrix: u64) -> Self {
        self.slot_coupling_matrix = slot_coupling_matrix;
        self
    }

    /// Validates configuration parameters branchlessly against LiminiKa architectural invariants.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `head_dim` is odd or zero,
    /// `num_heads` is zero or exceeds `64`, `paged_block_size` is neither `16` nor `32`,
    /// `ledger_mode > 2`, capacity bounds are zero, or any floating-point threshold is non-finite
    /// or outside its valid domain.
    #[inline]
    pub fn validate(&self) -> Result<()> {
        let valid_dims = (self.raw.head_dim != 0)
            & ((self.raw.head_dim & 1) == 0)
            & (self.raw.num_heads != 0)
            & ((self.raw.num_heads as usize) <= MAX_ATTENTION_HEADS)
            & ((self.raw.paged_block_size == 16) | (self.raw.paged_block_size == 32));

        let valid_scales = branchless_is_finite_f32(self.raw.q7_phase_scale)
            & (self.raw.q7_phase_scale > 0.0)
            & branchless_is_finite_f32(self.raw.phase_clamp_max_rad)
            & (self.raw.phase_clamp_max_rad > 0.0)
            & branchless_is_finite_f32(self.raw.phase_step_min_rad)
            & (self.raw.phase_step_min_rad >= 0.0)
            & branchless_is_finite_f32(self.raw.entropy_singularity_eps)
            & (self.raw.entropy_singularity_eps > 0.0);

        let valid_entropy_and_caps = (self.raw.max_prompt_anchors != 0)
            & (self.raw.substrate_capacity != 0)
            & (self.raw.ledger_mode <= 2)
            & branchless_is_finite_f32(self.hysteresis_band)
            & (self.hysteresis_band >= 0.0)
            & branchless_is_finite_f32(self.tau_flux)
            & (self.tau_flux >= 0.0)
            & branchless_is_finite_f32(self.bifurcation_threshold)
            & (self.bifurcation_threshold > 0.0);

        if valid_dims & valid_scales & valid_entropy_and_caps {
            Ok(())
        } else {
            Err(LMNK_ERR_INVALID_ARGUMENT)
        }
    }
}
