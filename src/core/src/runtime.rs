// name: src/core/src/runtime.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-level configuration builder ([`Config`]) and unified full-stack runtime context
//! ([`RuntimeContext`]) integrating [`ActionHub`], [`PhaseSteeringEngine`],
//! [`EntropyController`], [`MultiAttractorSpace`], and [`ManifoldGatewaySignal`].

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(clippy::needless_return)]

use crate::abi::{
    gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_edbc_state_t,
    gcso_kv_keyframe_header_t, gcso_ledger_slot_t, gcso_mapped_storage_descriptor_t,
    gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t, gcso_snapshot_header_t,
    gcso_tagged_ptr_t,
};
use crate::action_hub::{ActionHubEngine, StigmergicActionDecision, ACTION_HUB_INLINE_TRAILS};
use crate::attractor::{MultiAttractorSpace, MAX_ANCHORS_PER_FIELD, MAX_ATTRACTOR_DOMAINS};
use crate::entropy_control::{EntropyClosedLoopCoordinator, EntropyController};
use crate::phase_steering::{PhaseSteeringClosedLoop, PhaseSteeringEngine};
use crate::traits::rope::{PhaseSteeringGeometryExt, RopeSteeringTarget};
use crate::traits::{
    abi_version, abi_version_string, apply_latent_horizon_slots, query_capability, ActionHub,
    ActionHubGatewayExt, AlignedBuffer16, AlignedBuffer64, AlignedSlice16, AlignedSlice32,
    AlignedSliceMut16, AlignedSliceMut32, AttractorField, AutonomousCoreLayerStack,
    CausalProvenanceReport, CausalProvenanceSignal, ContrastivePhaseBloomFilter, CoreTelemetryBank,
    CoreTelemetrySnapshot, EdbcBranchController, EntropyEvaluator, EpistemicHorizonReading,
    FastPathLedger, GatewayFrame, GatewayPhaseSteeringExt, GcsoResult, HorizonDwellState,
    HorizonEscalationProvider, HorizonEscalationRequest, HorizonTriggerInput, KvKeyframeCache,
    LatentHorizonSlot, LiminalBoundaryDelta, ManifoldGatewaySignal, MappedStorage,
    MultiDomainAttractorSpace, PersonaPatcher, PhaseSteering, Pod, PointerActionHub,
    StigmergicActionPtrOps, StigmergicChunkOps, StigmergicTrailOps,
    StigmergicVennSerendipityReport, SystemConstraintSignal, TelemetryCapturePolicy,
    ACTION_MODE_LAYER_SKIP, ACTION_MODE_SUBHEAD_ROUTE, ATTRACTOR_SLOT_CUSTOM_7,
    ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, ROOT_CAUSE_SYSTEM_QOS_THROTTLE, TRAIL_FLAG_REPULSION_ACTIVE,
};

use crate::traits::branchless::{
    branchless_q7_inhibit_words_64, coherence_from_word_arrays_with_both_uniform_flags_64,
    swar_slerp_words_with_uniform_flag_64, swar_words_equal_64, swar_words_nonzero_64,
};
use crate::traits::primitives::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_max_f32, branchless_min_f32, branchless_q7_add_sat, branchless_q7_conjugate_64,
    branchless_relu_f32, branchless_select_f32, branchless_select_u32, branchless_select_u8,
    fast_inv_pos_normal_f32, is_phase_profile_nonzero_64, load_q7_words_64, store_q7_words_64,
    swar_or_avx2_q7_add_sat_words_64, trail_phase_words_mut, trail_phase_words_ref,
    SWAR_SPLAT_BYTE_64,
};
use crate::traits::primitives::{
    finalize_snapshot_header_crc32, init_snapshot_header, verify_snapshot_payload_crc32,
    ATTRACTOR_SLOT_ID_MASK, DEFAULT_ACTION_HUB_CAPACITY, DEFAULT_ENTROPY_EPS,
    DEFAULT_PAGED_BLOCK_SIZE, DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD,
    LEDGER_HISTORY_LEN, MAX_ATTENTION_HEADS, Q7_PHASE_SCALE_RAD,
};
use crate::traits::{
    LEDGER_METRIC_PHASE_STEERED, LEDGER_METRIC_REPULSION_APPLIED, MAX_ATTRACTOR_PULL_FORCE,
};

/// Convenient crate-level result type alias for GCSO operations.
pub type Result<T> = GcsoResult<T>;

/// Default EDBC hysteresis band width ($\delta_{\mathrm{hyst}}$).
pub const DEFAULT_EDBC_HYSTERESIS_BAND: f32 = 0.15;

/// Default sliding-window entropy rate flux threshold ($\tau_{\mathrm{flux}}$).
pub const DEFAULT_EDBC_TAU_FLUX: f32 = 0.50;

/// Default EDBC pitchfork bifurcation Z-score threshold ($\tau_{\mathrm{bifurcation}}$).
pub const DEFAULT_EDBC_BIFURCATION_THRESHOLD: f32 = 1.50;

/// Default Contrastive Phase Bloom Filter collision count threshold triggering anti-phase repulsion.
pub const DEFAULT_BLOOM_REPULSION_THRESHOLD: u8 = 2;

/// Default Q7 unit scaling factor (`128` = $1.0\times$) for phase-conjugate repulsion injection.
pub const DEFAULT_BLOOM_REPULSION_SCALE_U7: u8 = 128;

/// Total Q7 phase elements across all 8 Attractor Working Lanes (`8 * 64 = 512` bytes).
pub const SLOT_PHASE_BANK_LEN: usize = MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS;

/// Default initial weights for the 8 parallel Attractor Working Lanes (`0..=7`).
pub const DEFAULT_SLOT_WEIGHTS: [f32; MAX_ATTRACTOR_DOMAINS] =
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5];

// ===================================================================
// Safe High-Level Rust Configuration Wrapper & Fluent Builder
// ===================================================================

/// Safe high-level configuration wrapper for GCSO runtime initialization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Inner raw C-ABI configuration structure (64 bytes, 16-byte aligned).
    pub raw: gcso_config_t,
    /// Hysteresis band ($\delta_{\mathrm{hyst}}$) for EDBC branch control stabilization.
    pub hysteresis_band: f32,
    /// Tau flux threshold ($\tau_{\mathrm{flux}}$) for sliding-window entropy rate integration.
    pub tau_flux: f32,
    /// Z-score threshold ($\tau_{\mathrm{bifurcation}}$) triggering EDBC pitchfork bifurcation.
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
    /// Creates a new default configuration instance adhering to GCSO architectural invariants.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            raw: gcso_config_t {
                head_dim: 128,
                num_heads: 32,
                paged_block_size: DEFAULT_PAGED_BLOCK_SIZE,
                q7_phase_scale: Q7_PHASE_SCALE_RAD,
                ripa_clamp_max_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
                entropy_singularity_eps: DEFAULT_ENTROPY_EPS,
                max_prompt_anchors: MAX_ANCHORS_PER_FIELD as u32,
                action_hub_capacity: DEFAULT_ACTION_HUB_CAPACITY,
                enable_cuda_warp_shuffle: 1,
                enable_zero_alloc_strict: 1,
                ledger_mode: 0,
                reserved_flags: 0,
                rotary_dim: 0,
                rope_layout: 0,
                reserved: [0; 19],
            },
            hysteresis_band: DEFAULT_EDBC_HYSTERESIS_BAND,
            tau_flux: DEFAULT_EDBC_TAU_FLUX,
            bifurcation_threshold: DEFAULT_EDBC_BIFURCATION_THRESHOLD,
            repulsion_threshold: DEFAULT_BLOOM_REPULSION_THRESHOLD,
            repulsion_scale_u7: DEFAULT_BLOOM_REPULSION_SCALE_U7,
            slot_coupling_matrix: DEFAULT_SLOT_COUPLING_MATRIX_8X8,
        }
    }

    /// Constructs a high-level [`Config`] wrapper from a raw [`gcso_config_t`] descriptor.
    #[inline(always)]
    #[must_use]
    pub const fn from_raw(raw: gcso_config_t) -> Self {
        Self {
            raw,
            hysteresis_band: DEFAULT_EDBC_HYSTERESIS_BAND,
            tau_flux: DEFAULT_EDBC_TAU_FLUX,
            bifurcation_threshold: DEFAULT_EDBC_BIFURCATION_THRESHOLD,
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

    /// Sets the RIPA soft-bounded phase clamp limit in radians ($\theta_{\max}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_ripa_clamp_max_rad(mut self, ripa_clamp_max_rad: f32) -> Self {
        self.raw.ripa_clamp_max_rad = ripa_clamp_max_rad;
        self
    }

    /// Sets the QDPS quantization threshold angle in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_qdps_min_step_rad(mut self, qdps_min_step_rad: f32) -> Self {
        self.raw.qdps_min_step_rad = qdps_min_step_rad;
        self
    }

    /// Sets the logarithmic singularity prevention epsilon ($\epsilon_{\mathrm{log}}$).
    #[inline(always)]
    #[must_use]
    pub const fn with_entropy_singularity_eps(mut self, entropy_singularity_eps: f32) -> Self {
        self.raw.entropy_singularity_eps = entropy_singularity_eps;
        self
    }

    /// Sets the maximum supported prompt anchor points in the attractor field.
    #[inline(always)]
    #[must_use]
    pub const fn with_max_prompt_anchors(mut self, max_prompt_anchors: u32) -> Self {
        self.raw.max_prompt_anchors = max_prompt_anchors;
        self
    }

    /// Sets the Action Hub (SPT) pointer table slot capacity.
    #[inline(always)]
    #[must_use]
    pub const fn with_action_hub_capacity(mut self, action_hub_capacity: u32) -> Self {
        self.raw.action_hub_capacity = action_hub_capacity;
        self
    }

    /// Configures GPU warp shuffle instruction usage (`1` = enabled, `0` = disabled).
    #[inline(always)]
    #[must_use]
    pub const fn with_enable_cuda_warp_shuffle(mut self, enable_cuda_warp_shuffle: u8) -> Self {
        self.raw.enable_cuda_warp_shuffle = enable_cuda_warp_shuffle;
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

    /// Sets the EDBC hysteresis band width ($\delta_{\mathrm{hyst}}$).
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

    /// Sets the EDBC pitchfork bifurcation threshold ($\tau_{\mathrm{bifurcation}}$).
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

    /// Validates configuration parameters branchlessly against GCSO architectural invariants.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is odd or zero,
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
            & branchless_is_finite_f32(self.raw.ripa_clamp_max_rad)
            & (self.raw.ripa_clamp_max_rad > 0.0)
            & branchless_is_finite_f32(self.raw.qdps_min_step_rad)
            & (self.raw.qdps_min_step_rad >= 0.0)
            & branchless_is_finite_f32(self.raw.entropy_singularity_eps)
            & (self.raw.entropy_singularity_eps > 0.0);

        let valid_edbc_and_caps = (self.raw.max_prompt_anchors != 0)
            & (self.raw.action_hub_capacity != 0)
            & (self.raw.ledger_mode <= 2)
            & branchless_is_finite_f32(self.hysteresis_band)
            & (self.hysteresis_band >= 0.0)
            & branchless_is_finite_f32(self.tau_flux)
            & (self.tau_flux >= 0.0)
            & branchless_is_finite_f32(self.bifurcation_threshold)
            & (self.bifurcation_threshold > 0.0);

        if valid_dims & valid_scales & valid_edbc_and_caps {
            Ok(())
        } else {
            Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT)
        }
    }
}

// ===================================================================
// High-Level Safe Rust Runtime Context Engine
// ===================================================================

/// Single-glance status snapshot summarizing the active concrete core engines, the 3-layer
/// autonomous core stack (`Liminal` + `Stigmergic` + `Ego`), and dynamically slid-in horizon slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OrchestratorStatusSnapshot {
    /// Total tokens/steps processed by `ActionHubEngine`.
    pub step_counter: u64,
    /// Number of active trails registered in `ActionHubEngine`.
    pub active_trail_count: u32,
    /// Active EDBC branch mode (`0` = normal, `1` = bifurcation, `2` = repulsion).
    pub edbc_branch_mode: u32,
    /// Active anchor count in `MultiAttractorSpace`.
    pub attractor_anchor_count: u32,
    /// Bitmask of enabled working lanes (`0..=7`) in `ActionHubEngine`.
    pub slot_enabled_mask: u8,
    /// Active Layer 0 `Liminal` channel aperture mask (`8` channels).
    pub liminal_aperture_mask: u32,
    /// Active Layer 1 `Stigmergic` Venn-core granule mask.
    pub stigmergic_venn_core_mask: u8,
    /// Whether Layer 2 `LiminalEgoField` intervention is active.
    pub ego_active: bool,
    /// Monotonic count of Layer 2 Ego interventions applied.
    pub ego_intervention_epochs: u32,
    /// Whether Layer 0 boundaries are quiescent (`stationary_shortcut_ready != 0`).
    pub stationary_shortcut_ready: bool,
    /// Bitmask of external [`LatentHorizonSlot`] traits currently slid into the cognitive window.
    pub active_horizon_slots_mask: u32,
    /// Consecutive destabilized steps observed by the orchestrator.
    pub unstable_dwell_steps: u16,
    /// Consecutive quiescent steps observed by the orchestrator.
    pub quiescent_dwell_steps: u16,
}

/// High-level safe Rust single orchestrator facade integrating concrete core engines
/// ([`ActionHubEngine`], [`PhaseSteeringEngine`], [`EntropyController`], [`MultiAttractorSpace`]),
/// the 128-byte 3-layer core stack ([`AutonomousCoreLayerStack`]: `Liminal` + `Stigmergic` + `Ego`),
/// and dynamic slide-in/slide-out [`LatentHorizonSlot`] boundaries.
#[derive(Debug)]
pub struct RuntimeContext {
    config: Config,
    action_hub: ActionHubEngine,
    dpsr: PhaseSteeringEngine,
    edbc: EntropyController,
    attractor_space: MultiAttractorSpace,
    core_layers: AutonomousCoreLayerStack,
    last_boundary_delta: LiminalBoundaryDelta,
    last_horizon: EpistemicHorizonReading,
    last_venn_report: StigmergicVennSerendipityReport,
    horizon_dwell: HorizonDwellState,
    active_horizon_slots_mask: u32,
    telemetry: CoreTelemetryBank,
}

impl RuntimeContext {
    /// Creates a new verified [`RuntimeContext`] instance from [`Config`].
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if configuration parameters violate
    /// GCSO architectural invariants.
    #[inline]
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let dpsr = PhaseSteeringEngine::new(config.raw.head_dim, config.raw.num_heads)?;
        let mut action_hub = ActionHubEngine::from_config(&config.raw)?;
        action_hub.repulsion_threshold = config.repulsion_threshold;
        action_hub.repulsion_scale_u7 = config.repulsion_scale_u7;
        action_hub.coupling_matrix_8x8 = config.slot_coupling_matrix;
        action_hub.slot_weights = DEFAULT_SLOT_WEIGHTS;
        action_hub.slot_enabled_mask = 0xFF;
        let edbc_state = gcso_edbc_state_t {
            singularity_eps: config.raw.entropy_singularity_eps,
            bifurcation_threshold: config.bifurcation_threshold,
            ..Default::default()
        };
        let mut edbc = EntropyController::new(edbc_state);
        edbc.set_hysteresis_band(config.hysteresis_band);
        edbc.set_tau_flux(config.tau_flux);

        Ok(Self {
            config,
            action_hub,
            dpsr,
            edbc,
            attractor_space: MultiAttractorSpace::new(),
            core_layers: AutonomousCoreLayerStack::new(),
            last_boundary_delta: LiminalBoundaryDelta::default(),
            last_horizon: EpistemicHorizonReading::default(),
            last_venn_report: StigmergicVennSerendipityReport::default(),
            horizon_dwell: HorizonDwellState::default(),
            active_horizon_slots_mask: 0,
            telemetry: CoreTelemetryBank::default(),
        })
    }

    /// Returns an immutable reference to the 128-byte 3-layer [`AutonomousCoreLayerStack`]
    /// (`Layer 0: Liminal`, `Layer 1: Stigmergic`, `Layer 2: Ego`).
    #[inline(always)]
    #[must_use]
    pub const fn core_layers(&self) -> &AutonomousCoreLayerStack {
        &self.core_layers
    }

    /// Returns a mutable reference to the 128-byte 3-layer [`AutonomousCoreLayerStack`].
    #[inline(always)]
    #[must_use]
    pub fn core_layers_mut(&mut self) -> &mut AutonomousCoreLayerStack {
        &mut self.core_layers
    }

    /// Returns an immutable reference to the latest Layer 0 [`LiminalBoundaryDelta`].
    #[inline(always)]
    #[must_use]
    pub const fn last_boundary_delta(&self) -> &LiminalBoundaryDelta {
        &self.last_boundary_delta
    }

    /// Returns a mutable reference to the latest Layer 0 [`LiminalBoundaryDelta`].
    #[inline(always)]
    #[must_use]
    pub fn last_boundary_delta_mut(&mut self) -> &mut LiminalBoundaryDelta {
        &mut self.last_boundary_delta
    }

    /// Returns an immutable reference to the latest [`EpistemicHorizonReading`].
    #[inline(always)]
    #[must_use]
    pub const fn last_horizon(&self) -> &EpistemicHorizonReading {
        &self.last_horizon
    }

    /// Returns a mutable reference to the latest [`EpistemicHorizonReading`].
    #[inline(always)]
    #[must_use]
    pub fn last_horizon_mut(&mut self) -> &mut EpistemicHorizonReading {
        &mut self.last_horizon
    }

    /// Returns an immutable reference to the latest [`StigmergicVennSerendipityReport`].
    #[inline(always)]
    #[must_use]
    pub const fn last_venn_report(&self) -> &StigmergicVennSerendipityReport {
        &self.last_venn_report
    }

    /// Returns a mutable reference to the latest [`StigmergicVennSerendipityReport`].
    #[inline(always)]
    #[must_use]
    pub fn last_venn_report_mut(&mut self) -> &mut StigmergicVennSerendipityReport {
        &mut self.last_venn_report
    }

    /// Returns the bitmask of external [`LatentHorizonSlot`] boundaries currently slid into the
    /// active cognitive window.
    #[inline(always)]
    #[must_use]
    pub const fn active_horizon_slots_mask(&self) -> u32 {
        self.active_horizon_slots_mask
    }

    /// Returns the consecutive destabilization / quiescence dwell tracker [`HorizonDwellState`].
    #[inline(always)]
    #[must_use]
    pub const fn horizon_dwell(&self) -> HorizonDwellState {
        self.horizon_dwell
    }

    /// Returns a shared reference to the embedded `3 + System` [`CoreTelemetryBank`].
    #[inline(always)]
    #[must_use]
    pub const fn telemetry_bank(&self) -> &CoreTelemetryBank {
        &self.telemetry
    }

    /// Returns a mutable reference to the embedded `3 + System` [`CoreTelemetryBank`].
    #[inline(always)]
    #[must_use]
    pub fn telemetry_bank_mut(&mut self) -> &mut CoreTelemetryBank {
        &mut self.telemetry
    }

    /// Updates the [`TelemetryCapturePolicy`] governing verbosity transitions (`Coarse` / `Standard` / `Detailed`).
    #[inline(always)]
    pub fn set_telemetry_policy(&mut self, policy: TelemetryCapturePolicy) {
        self.telemetry.set_policy(policy);
    }

    /// Resets accumulated telemetry counters across `3 + System` domains.
    #[inline(always)]
    pub fn reset_telemetry(&mut self) {
        self.telemetry.reset();
    }

    /// Computes a Lazy-Pull [`CoreTelemetrySnapshot`] combining `3 + System` counters, derived
    /// ratios, and live Layer 0 topological state.
    #[inline]
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CoreTelemetrySnapshot {
        self.telemetry.snapshot(
            &self.core_layers,
            &self.last_horizon,
            self.attractor_space.system_field().active_count().min(255) as u8,
            0,
            self.active_horizon_slots_mask,
        )
    }

    /// Returns a single-glance [`OrchestratorStatusSnapshot`] summarizing concrete core engines,
    /// 3-layer core stack state, and dynamically active horizon slots.
    #[inline(always)]
    #[must_use]
    pub fn orchestrator_status(&self) -> OrchestratorStatusSnapshot {
        OrchestratorStatusSnapshot {
            step_counter: self.action_hub.step_counter,
            active_trail_count: self.action_hub.active_count,
            edbc_branch_mode: self.edbc.state().active_branch_mode,
            attractor_anchor_count: self.attractor_space.system_field().active_count(),
            slot_enabled_mask: self.action_hub.slot_enabled_mask,
            liminal_aperture_mask: self.core_layers.liminal_layer.active_aperture_mask,
            stigmergic_venn_core_mask: self.core_layers.stigmergic_layer.venn_overlap_mask,
            ego_active: self.core_layers.ego_layer.is_active(),
            ego_intervention_epochs: self.core_layers.ego_layer.intervention_epochs,
            stationary_shortcut_ready: self.last_boundary_delta.stationary_shortcut_ready != 0,
            active_horizon_slots_mask: self.active_horizon_slots_mask,
            unstable_dwell_steps: self.horizon_dwell.unstable_dwell_steps,
            quiescent_dwell_steps: self.horizon_dwell.quiescent_dwell_steps,
        }
    }

    /// Returns an immutable reference to the active [`Config`].
    #[inline(always)]
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// Returns an immutable reference to the underlying [`ActionHubEngine`].
    #[inline(always)]
    #[must_use]
    pub const fn action_hub(&self) -> &ActionHubEngine {
        &self.action_hub
    }

    /// Returns a mutable reference to the underlying [`ActionHubEngine`].
    #[inline(always)]
    #[must_use]
    pub fn action_hub_mut(&mut self) -> &mut ActionHubEngine {
        &mut self.action_hub
    }

    /// Returns an immutable reference to the underlying [`EntropyController`].
    #[inline(always)]
    #[must_use]
    pub const fn entropy_controller(&self) -> &EntropyController {
        &self.edbc
    }

    /// Returns a mutable reference to the underlying [`EntropyController`].
    #[inline(always)]
    #[must_use]
    pub fn entropy_controller_mut(&mut self) -> &mut EntropyController {
        &mut self.edbc
    }

    /// Returns an immutable reference to the [`MultiAttractorSpace`].
    #[inline(always)]
    #[must_use]
    pub const fn attractor_space(&self) -> &MultiAttractorSpace {
        &self.attractor_space
    }

    /// Returns a mutable reference to the [`MultiAttractorSpace`].
    #[inline(always)]
    #[must_use]
    pub fn attractor_space_mut(&mut self) -> &mut MultiAttractorSpace {
        &mut self.attractor_space
    }

    /// Returns an immutable reference to the 64-byte [`ContrastivePhaseBloomFilter`].
    #[inline(always)]
    #[must_use]
    pub const fn bloom_filter(&self) -> &ContrastivePhaseBloomFilter {
        &self.action_hub.bloom_filter
    }

    /// Returns a mutable reference to the 64-byte [`ContrastivePhaseBloomFilter`].
    #[inline(always)]
    #[must_use]
    pub fn bloom_filter_mut(&mut self) -> &mut ContrastivePhaseBloomFilter {
        &mut self.action_hub.bloom_filter
    }

    /// Returns an immutable reference to the active [`gcso_ledger_slot_t`] scratchpad.
    #[inline(always)]
    #[must_use]
    pub const fn ledger_slot(&self) -> &gcso_ledger_slot_t {
        &self.action_hub.ledger_slot
    }

    /// Returns a mutable reference to the active [`gcso_ledger_slot_t`] scratchpad.
    #[inline(always)]
    #[must_use]
    pub fn ledger_slot_mut(&mut self) -> &mut gcso_ledger_slot_t {
        &mut self.action_hub.ledger_slot
    }

    /// Returns an immutable 16-byte aligned slice view over the 8-lane Q7 working phase bank (`512` bytes).
    #[inline(always)]
    #[must_use]
    pub const fn slot_phase_bank(&self) -> AlignedSlice16<'_, gcso_q7_t> {
        self.action_hub.slot_phase_bank.relax_slice::<16>()
    }

    /// Returns a mutable 16-byte aligned slice view over the 8-lane Q7 working phase bank (`512` bytes).
    #[inline(always)]
    #[must_use]
    pub fn slot_phase_bank_mut(&mut self) -> AlignedSliceMut16<'_, gcso_q7_t> {
        self.action_hub.slot_phase_bank.relax_slice_mut::<16>()
    }

    /// Returns a mutable reference to the 64-byte (`8 x u64`) aligned word view of slot `slot_idx`.
    #[inline(always)]
    fn slot_words_mut(
        &mut self,
        slot_idx: usize,
    ) -> Result<&mut [u64; crate::traits::NUM_STIGMERGIC_GRANULES]> {
        if slot_idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.action_hub.slot_phase_bank` is `AlignedBuffer64<gcso_q7_t, 512>`, so every
        // 64-byte slot at byte offset `slot_idx * 64` is 64-byte aligned (`>= 8-byte` alignment of `u64`).
        Ok(unsafe {
            &mut *(self
                .action_hub
                .slot_phase_bank
                .as_mut_slice()
                .as_mut_ptr()
                .add(slot_idx * MAX_ATTENTION_HEADS)
                as *mut [u64; crate::traits::NUM_STIGMERGIC_GRANULES])
        })
    }

    /// Sets the 64-head Q7 phase profile for a specific Attractor Working Lane (`0..=7`).
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if `slot_id >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn set_slot_phase_profile(
        &mut self,
        slot_id: u8,
        phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> Result<()> {
        let dst_w = self.slot_words_mut(usize::from(slot_id))?;
        *dst_w = load_q7_words_64(phases);
        Ok(())
    }

    /// Returns the active 8-lane weight array (`[f32; MAX_ATTRACTOR_DOMAINS]`).
    #[inline(always)]
    #[must_use]
    pub const fn slot_weights(&self) -> &[f32; MAX_ATTRACTOR_DOMAINS] {
        &self.action_hub.slot_weights
    }

    /// Returns the 8-bit bitmask of enabled Attractor Working Lanes (`bit i == 1` means lane `i` is active).
    #[inline(always)]
    #[must_use]
    pub const fn slot_enabled_mask(&self) -> u8 {
        self.action_hub.slot_enabled_mask
    }

    /// Returns the active 64-bit ($8 \times 8$) Attractor Working Lane coupling bitmatrix.
    #[inline(always)]
    #[must_use]
    pub const fn slot_coupling_matrix(&self) -> u64 {
        self.action_hub.coupling_matrix_8x8
    }

    /// Updates the active 64-bit ($8 \times 8$) Attractor Working Lane coupling bitmatrix.
    #[inline(always)]
    pub fn set_slot_coupling_matrix(&mut self, matrix: u64) {
        self.action_hub.coupling_matrix_8x8 = matrix;
        self.config.slot_coupling_matrix = matrix;
    }

    /// Returns the total number of processed token steps.
    #[inline(always)]
    #[must_use]
    pub const fn step_counter(&self) -> u64 {
        self.action_hub.step_counter
    }

    /// Returns the active pointer trail count in the Action Hub.
    #[inline(always)]
    #[must_use]
    pub const fn active_trail_count(&self) -> u32 {
        self.action_hub.active_count
    }

    /// Executes a per-token Hot Path step integrating DPSR phase steering, EDBC entropy evaluation,
    /// `self.slot_phase_bank` directly in 8 register-resident `u64` SWAR words (`0` intermediate
    /// 64-byte stack buffers or store-reload round-trips).
    #[inline]
    #[must_use]
    pub(crate) fn evaluate_multi_domain_superposition_words(
        &self,
        trail: &gcso_pointer_trail_t,
    ) -> ([u64; crate::traits::NUM_STIGMERGIC_GRANULES], f32) {
        let tw = *trail_phase_words_ref(trail);
        let tn = swar_words_nonzero_64(&tw);
        let (comp_w, base_pull) = self
            .attractor_space
            .evaluate_multi_domain_superposition_words(&tw, tn);
        let (init_w, init_nonzero) = if swar_words_nonzero_64(&comp_w) {
            (comp_w, true)
        } else {
            (tw, tn)
        };
        let init_is_uniform = !init_nonzero || crate::traits::is_word_array_uniform_64(&init_w);
        let mut run_w = init_w;
        let mut run_is_uniform = init_is_uniform;
        let mut running_pull = branchless_flush_subnormal_f32(branchless_relu_f32(base_pull));

        if let Ok(bank_2d) = self
            .action_hub
            .slot_phase_bank
            .as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()
        {
            let rep_idx = usize::from(ATTRACTOR_SLOT_REPULSION);
            let enabled_mask = self.action_hub.slot_enabled_mask;
            let weights = &self.action_hub.slot_weights;
            let mut idx = 0usize;
            while idx < MAX_ATTRACTOR_DOMAINS {
                if idx != rep_idx {
                    let enabled = ((enabled_mask >> idx) & 1) != 0;
                    if enabled {
                        let base_e =
                            branchless_flush_subnormal_f32(branchless_relu_f32(weights[idx]));
                        if base_e > DEFAULT_ENTROPY_EPS {
                            // SAFETY: `self.action_hub.slot_phase_bank` is `AlignedBuffer64<gcso_q7_t, 512>`, so every
                            // 64-byte row `bank_2d[idx]` is 64-byte aligned (`>= 8-byte` alignment of `u64`).
                            let slot_w = unsafe {
                                &*(bank_2d[idx].as_ptr()
                                    as *const [u64; crate::traits::NUM_STIGMERGIC_GRANULES])
                            };
                            if swar_words_nonzero_64(slot_w) {
                                let slot_is_uniform =
                                    crate::traits::is_word_array_uniform_64(slot_w);
                                let coh = if init_nonzero {
                                    coherence_from_word_arrays_with_both_uniform_flags_64(
                                        &init_w,
                                        init_is_uniform,
                                        slot_w,
                                        slot_is_uniform,
                                    )
                                } else {
                                    0.0
                                };
                                let res_pull = branchless_flush_subnormal_f32(
                                    base_e * branchless_clamp_f32(1.0 + coh, 0.25, 2.0),
                                );
                                let total_w = running_pull + res_pull;
                                if total_w > DEFAULT_ENTROPY_EPS
                                    && !swar_words_equal_64(&run_w, slot_w)
                                {
                                    let ratio_f32 = branchless_clamp_f32(
                                        (res_pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                        0.0,
                                        128.0,
                                    );
                                    let s = (ratio_f32 as u8) as u64;
                                    let bias_restore =
                                        (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                    let (next_w, next_uni) = swar_slerp_words_with_uniform_flag_64(
                                        &run_w,
                                        run_is_uniform,
                                        slot_w,
                                        slot_is_uniform,
                                        s,
                                        bias_restore,
                                    );
                                    run_w = next_w;
                                    run_is_uniform = next_uni;
                                }
                                running_pull = total_w;
                            }
                        }
                    }
                }
                idx += 1;
            }

            let rep_enabled = ((enabled_mask >> rep_idx) & 1) != 0;
            if rep_enabled {
                // SAFETY: `bank_2d[rep_idx]` is 64-byte aligned.
                let rep_words = unsafe {
                    &*(bank_2d[rep_idx].as_ptr()
                        as *const [u64; crate::traits::NUM_STIGMERGIC_GRANULES])
                };
                if swar_words_nonzero_64(rep_words) {
                    let rep_strength =
                        branchless_clamp_f32(branchless_relu_f32(weights[rep_idx]), 0.0, 1.0);
                    running_pull = branchless_q7_inhibit_words_64(
                        &mut run_w,
                        running_pull,
                        rep_words,
                        rep_strength,
                    );
                }
            }
        }

        let clamped_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(running_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        (run_w, clamped_pull)
    }

    /// Serializes active runtime context state into a `.gcso` binary snapshot conforming to
    /// [`crate::abi::GCSO_ABI_VERSION_HEX`], stamping an IEEE 802.3 CRC32 checksum over the payload.
    ///
    /// Supports two-pass sizing: passing an empty `buffer` writes the required byte size (`320` bytes)
    /// into `*required_size` and returns `Ok(())`.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`] if `buffer` is non-empty and smaller
    /// than `320` bytes.
    #[inline]
    pub fn serialize_to_bytes(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        let header_size = core::mem::size_of::<gcso_snapshot_header_t>(); // 128 bytes
        let context_data_size = 64usize; // 64-byte aligned context & attractor metadata block
        let edbc_size = core::mem::size_of::<gcso_edbc_state_t>(); // 64 bytes
        let daes_size = core::mem::size_of::<gcso_ledger_slot_t>(); // 64 bytes

        let total_required = header_size + context_data_size + edbc_size + daes_size; // 320 bytes

        if buffer.is_empty() {
            *required_size = total_required;
            return Ok(());
        }
        if buffer.len() < total_required {
            *required_size = total_required;
            return Err(crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL);
        }

        buffer[..total_required].fill(0);

        let action_hub_off = header_size; // 128
        let edbc_off = action_hub_off + context_data_size; // 192
        let daes_off = edbc_off + edbc_size; // 256

        let ctx_buf = &mut buffer[action_hub_off..edbc_off];
        ctx_buf[0..8].copy_from_slice(&self.action_hub.step_counter.to_le_bytes());
        ctx_buf[8..12].copy_from_slice(&self.action_hub.active_count.to_le_bytes());
        let sys_anchors = self.attractor_space.system_field().active_count();
        ctx_buf[12..16].copy_from_slice(&sys_anchors.to_le_bytes());
        let rep_gain = self.attractor_space.system_field().repulsion_gain;
        ctx_buf[16..20].copy_from_slice(&rep_gain.to_le_bytes());
        ctx_buf[20] = self.action_hub.slot_enabled_mask;
        ctx_buf[24..32].copy_from_slice(&self.action_hub.coupling_matrix_8x8.to_le_bytes());
        f32::write_unaligned_slice_to(&self.action_hub.slot_weights, ctx_buf, 32)?;

        let edbc_state = self.edbc.state();
        edbc_state.write_unaligned_to(buffer, edbc_off)?;
        self.action_hub
            .ledger_slot
            .write_unaligned_to(buffer, daes_off)?;

        let mut header = init_snapshot_header(total_required as u64, 1_774_900_000);
        header.action_hub_offset = action_hub_off as u64;
        header.attractor_field_offset = action_hub_off as u64;
        header.dpsr_state_offset = edbc_off as u64;
        header.rank1_state_offset = edbc_off as u64;
        header.edbc_state_offset = edbc_off as u64;
        header.ledger_slot_offset = daes_off as u32;
        finalize_snapshot_header_crc32(&mut header, &buffer[..total_required])?;

        header.write_unaligned_to(buffer, 0)?;
        *required_size = total_required;
        Ok(())
    }

    /// Deserializes a `.gcso` binary snapshot buffer to restore context state in-place with
    /// ABI header validation and IEEE 802.3 CRC32 payload integrity verification.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED`] if the header magic, section
    /// offsets, buffer bounds, or payload CRC32 checksum are invalid.
    /// Returns [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`] if the snapshot ABI version does not
    /// match [`crate::abi::GCSO_ABI_VERSION_HEX`].
    #[inline]
    pub fn deserialize_from_bytes(&mut self, buffer: &[u8]) -> Result<()> {
        let header_size = core::mem::size_of::<gcso_snapshot_header_t>();
        if buffer.len() < header_size {
            return Err(crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        let header = gcso_snapshot_header_t::read_unaligned_from(buffer, 0)
            .map_err(|_| crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED)?;

        verify_snapshot_payload_crc32(&header, buffer)?;

        let action_hub_off = header.action_hub_offset as usize;
        let edbc_off = header.edbc_state_offset as usize;
        let daes_off = header.ledger_slot_offset as usize;

        if action_hub_off < header_size || edbc_off < action_hub_off || daes_off < edbc_off {
            return Err(crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        if buffer.len() >= action_hub_off + 21 {
            self.action_hub.step_counter = u64::read_unaligned_from(buffer, action_hub_off)?;
            self.action_hub.active_count = u32::read_unaligned_from(buffer, action_hub_off + 8)?;
            let sys_anchors = u32::read_unaligned_from(buffer, action_hub_off + 12)?;
            let rep_gain = f32::read_unaligned_from(buffer, action_hub_off + 16)?;
            self.action_hub.slot_enabled_mask = buffer[action_hub_off + 20];

            if buffer.len() >= action_hub_off + 32 {
                let coupling = u64::read_unaligned_from(buffer, action_hub_off + 24)?;
                if coupling != 0 {
                    self.action_hub.coupling_matrix_8x8 = coupling;
                    self.config.slot_coupling_matrix = coupling;
                }
            }
            if buffer.len() >= action_hub_off + 64 {
                f32::read_unaligned_slice_from(
                    buffer,
                    action_hub_off + 32,
                    &mut self.action_hub.slot_weights,
                )?;
            }

            self.attractor_space.system_field_mut().clear_anchors()?;
            self.attractor_space.system_field_mut().repulsion_gain = rep_gain;
            for _ in 0..sys_anchors.min(MAX_ANCHORS_PER_FIELD as u32) {
                let _ = self
                    .attractor_space
                    .system_field_mut()
                    .add_system_prompt_anchor("restored_anchor", 1.0);
            }
        }

        let edbc_size = core::mem::size_of::<gcso_edbc_state_t>();
        if buffer.len() >= edbc_off + edbc_size {
            let edbc_st = gcso_edbc_state_t::read_unaligned_from(buffer, edbc_off)?;
            self.edbc = EntropyController::new(edbc_st);
            self.edbc.set_hysteresis_band(self.config.hysteresis_band);
            self.edbc.set_tau_flux(self.config.tau_flux);
        }

        let daes_size = core::mem::size_of::<gcso_ledger_slot_t>();
        if buffer.len() >= daes_off + daes_size {
            self.action_hub.ledger_slot =
                gcso_ledger_slot_t::read_unaligned_from(buffer, daes_off)?;
        }

        Ok(())
    }
}

// ===================================================================
// Inherent Runtime Orchestration & Lifecycle Methods
// ===================================================================

impl RuntimeContext {
    /// Returns the active semantic ABI version tuple `(major, minor, patch)`.
    #[inline(always)]
    #[must_use]
    pub const fn abi_version(&self) -> (u32, u32, u32) {
        abi_version()
    }

    /// Returns the static ABI version string ([`crate::abi::GCSO_ABI_VERSION_STRING`]).
    #[inline(always)]
    #[must_use]
    pub const fn abi_version_string(&self) -> &'static str {
        abi_version_string()
    }

    /// Queries active compute capability bitmask flags without branching.
    #[inline(always)]
    #[must_use]
    pub const fn query_capability(&self) -> gcso_capability_flags_t {
        query_capability()
    }

    /// Populates `config` with this runtime context's active raw configuration descriptor.
    ///
    /// # Errors
    /// Never fails in the default implementation.
    #[inline(always)]
    pub fn init_default_config(&self, config: &mut gcso_config_t) -> Result<()> {
        *config = self.config.raw;
        Ok(())
    }

    /// Resets transient phase accumulators and step counters without freeing pre-allocated tables.
    ///
    /// # Errors
    /// Propagates any reset error from [`ActionHubEngine::reset`] or [`RuntimeContext::clear_all_domains`].
    #[inline]
    pub fn reset(&mut self) -> Result<()> {
        self.edbc.reset();
        self.action_hub.reset()?;
        self.clear_all_domains()?;
        self.action_hub.bloom_filter.clear();
        self.action_hub.ledger_slot = gcso_ledger_slot_t {
            mode: u32::from(self.config.raw.ledger_mode),
            ..Default::default()
        };
        self.core_layers = AutonomousCoreLayerStack::new();
        self.last_boundary_delta = LiminalBoundaryDelta::default();
        self.last_horizon = EpistemicHorizonReading::default();
        self.last_venn_report = StigmergicVennSerendipityReport::default();
        self.active_horizon_slots_mask = 0;
        Ok(())
    }

    /// Registers a natural language system prompt text anchor in phase space as the primary baseline attractor.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if `prompt` is empty or `weight` is non-finite.
    #[inline]
    pub fn set_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> Result<()> {
        self.attractor_space
            .system_field_mut()
            .add_system_prompt_anchor(prompt, weight)?;
        self.action_hub.slot_weights[usize::from(ATTRACTOR_SLOT_SYSTEM_BASELINE)] =
            branchless_relu_f32(weight);
        Ok(())
    }

    /// Executes a per-token Hot Path inference step in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias.
    #[inline]
    pub fn step_token(
        &mut self,
        token_id: u32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> Result<()> {
        if let (Some(q), Some(k)) = (query_tensor.as_deref(), key_tensor.as_deref()) {
            q.ensure_disjoint_mut(k)?;
        }

        let mut state_out = gcso_edbc_state_t::default();
        self.edbc.eval_stateful(0.0, &mut state_out)?;

        let next_step = self.action_hub.step_counter.wrapping_add(1);
        self.action_hub.step_counter = next_step;
        self.attractor_space.advance_step();
        let cap = self.config.raw.action_hub_capacity;
        let cur_active = self.action_hub.active_count;
        self.action_hub.active_count =
            branchless_select_u32(cur_active < cap, cur_active.wrapping_add(1), cur_active);

        let mut repulsion_active = false;
        if let Some(trail) = trail_out {
            trail.prev_ptr = trail.current_ptr;
            trail.current_ptr = u64::from(token_id);
            trail.step_count = next_step as u32;
            trail.stigmergic_density = branchless_select_f32(
                trail.stigmergic_density > 1.0,
                trail.stigmergic_density,
                1.0,
            );

            let (mut phase_words, total_pull) =
                self.evaluate_multi_domain_superposition_words(trail);

            let (_density, rep_trig) = self
                .action_hub
                .bloom_filter
                .apply_phase_conjugate_repulsion_words(
                    &mut phase_words,
                    self.config.repulsion_threshold,
                    self.config.repulsion_scale_u7,
                );
            repulsion_active = rep_trig;

            trail.attractor_pull_force = total_pull;
            *trail_phase_words_mut(trail) = phase_words;
            trail.flags = state_out.active_branch_mode
                | branchless_select_u32(repulsion_active, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        }

        let metric = LEDGER_METRIC_PHASE_STEERED
            | branchless_select_u8(repulsion_active, LEDGER_METRIC_REPULSION_APPLIED, 0);
        self.action_hub.ledger_slot.push_telemetry_metric(metric);
        Ok(())
    }

    /// Serializes active runtime context state into a `.gcso` binary snapshot buffer.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`] if `buffer.len() < *required_size`.
    #[inline]
    pub fn serialize(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        self.serialize_to_bytes(buffer, required_size)
    }

    /// Deserializes a `.gcso` binary snapshot buffer to restore context state in-place.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED`] or [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`]
    /// if the snapshot header magic, version, or payload checksum is invalid.
    #[inline]
    pub fn deserialize(&mut self, buffer: &[u8]) -> Result<()> {
        self.deserialize_from_bytes(buffer)
    }

    /// Seeks to a specific token position in the KV-keyframe cache without forward passes.
    ///
    /// # Errors
    /// Never fails in the default implementation.
    #[inline]
    pub fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_kv_keyframe_header_t,
    ) -> Result<()> {
        header_out.token_index = token_index;
        header_out.frame_type = 0;
        Ok(())
    }

    /// Executes a unified vertical orchestration pass that pierces all 3 core layers
    /// (`Layer 0: Liminal` -> `Layer 1: Stigmergic` -> `Layer 2: Ego`), dynamically slides
    /// dormant [`LatentHorizonSlot`] boundaries in or out based on live stability/anomaly
    /// telemetry, optionally escalates unresolved excursion to an external
    /// [`HorizonEscalationProvider`], updates `trail_out` / `ActionHubEngine`, and steers
    /// `query_tensor` in strict zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// Returns `(frame, report, slid_in_bits, slid_out_bits, escalated)`.
    ///
    /// # Errors
    /// Propagates validation or steering errors from `geometry`, `query_tensor`, `ActionHubEngine`,
    /// or `escalation_provider`.
    #[inline]
    pub fn pierce_and_orchestrate_step<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
        H: LatentHorizonSlot + ?Sized,
        E: HorizonEscalationProvider + ?Sized,
    >(
        &mut self,
        signal: &S,
        geometry: &G,
        latent_slots: &mut H,
        escalation_provider: &mut E,
        trail_out: &mut gcso_pointer_trail_t,
        mut query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> Result<(GatewayFrame, CausalProvenanceReport, u32, u32, bool)> {
        let (head_dim, num_heads, _expected_len) = self.action_hub.validate_hub_cascade_params(
            self.dpsr.qdps_min_step_rad(),
            self.dpsr.ripa_clamp_max_rad(),
            query_tensor.as_deref(),
            None,
        )?;
        let geom = geometry.rope_geometry();
        if geom.head_dim != head_dim || geom.num_heads != num_heads {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }

        // 1. Vertical 3-layer core stack piercing (Liminal -> Stigmergic -> Ego) + temporal delta.
        let prev_layer0 = self.core_layers.liminal_layer;
        let prev_venn = self.core_layers.stigmergic_layer.venn_overlap_mask;
        let (mut frame, mut frm_w) = signal.synthesize_gateway_with_core_layers(
            &mut self.core_layers,
            &self.last_boundary_delta,
            &self.last_horizon,
            &self.last_venn_report,
        );
        self.last_boundary_delta = LiminalBoundaryDelta::compute_branchless(
            &prev_layer0,
            &self.core_layers.liminal_layer,
            prev_venn,
            self.last_venn_report.venn_core_mask,
            &self.last_boundary_delta,
        );

        // 2. Evaluate stability / anomaly trigger with dwell hysteresis and slide slots in/out.
        let edbc_st = self.edbc.state();
        let mut trigger = HorizonTriggerInput {
            edbc_branch_mode: edbc_st.active_branch_mode,
            moving_z_entropy: edbc_st.moving_z_entropy,
            repulsion_severity: frame.repulsion_severity,
            causal_root_cause: frame.dominant_cause_code,
            epistemic_regime: self.last_horizon.regime_code,
            stationary_shortcut_ready: self.last_boundary_delta.stationary_shortcut_ready != 0,
            ego_active_triggers: self.core_layers.ego_layer.last_intervened_mask,
            unstable_dwell_steps: 0,
            quiescent_dwell_steps: 0,
        };
        let (unstable_dwell, quiescent_dwell) = self
            .horizon_dwell
            .step_branchless(trigger.is_destabilized(), trigger.stationary_shortcut_ready);
        trigger.unstable_dwell_steps = unstable_dwell;
        trigger.quiescent_dwell_steps = quiescent_dwell;

        let (slid_in, slid_out) = apply_latent_horizon_slots(
            latent_slots,
            &mut self.active_horizon_slots_mask,
            &trigger,
            &mut frame,
            &mut self.core_layers,
        );
        if slid_in != 0 || slid_out != 0 || self.active_horizon_slots_mask != 0 {
            frm_w = *frame.phase_words_ref();
        }

        // 3. Cold-path escalation swap-in when destabilized, or cleanly retract spare lane.
        let esc_req = HorizonEscalationRequest {
            trigger,
            target_slot: ATTRACTOR_SLOT_CUSTOM_7,
            layer_scale_q7: 64,
            desired_aperture_mask: self.core_layers.liminal_layer.active_aperture_mask,
            current_frame_energy: frame.effective_energy,
        };
        let escalated = self
            .action_hub
            .apply_horizon_escalation(&esc_req, escalation_provider)?;

        // 4. Step ActionHub pointer trail + Bloom filter + RoPE phase steering (zero-alloc).
        let bloom_snapshot = self.action_hub.bloom_filter;
        let mut ledger_slot = self.action_hub.ledger_slot;
        let (bypass_hit, _eff_ptr, _density, repulsion_active, post_rep_words) =
            self.action_hub.step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(frame.tagged_ptr),
                &mut ledger_slot,
                &bloom_snapshot,
                self.config.repulsion_threshold,
                self.config.repulsion_scale_u7,
                trail_out,
            )?;
        *trail_phase_words_mut(trail_out) = if swar_words_nonzero_64(&frm_w) {
            frm_w
        } else {
            post_rep_words
        };
        trail_out.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_out.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let mut steered = false;
        if (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) == 0 {
            if let Some(q) = query_tensor.as_deref_mut() {
                let final_w = *trail_phase_words_ref(trail_out);
                if swar_words_nonzero_64(&final_w) {
                    let mut q7_stack = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
                    store_q7_words_64(&mut q7_stack.data, &final_w);
                    let p_aligned = AlignedSlice16::new(&q7_stack.data[..num_heads])?;
                    self.dpsr.apply_phase_steering_for_geometry_safe(
                        geometry,
                        q,
                        &p_aligned,
                        frame.ripa_clamp_rad,
                    )?;
                    steered = true;
                }
            }
        }

        let metric = crate::traits::primitives::branchless_select_u8(
            bypass_hit,
            crate::traits::LEDGER_METRIC_FAST_PATH_HIT,
            0,
        ) | crate::traits::primitives::branchless_select_u8(
            steered,
            LEDGER_METRIC_PHASE_STEERED,
            0,
        ) | crate::traits::primitives::branchless_select_u8(
            repulsion_active,
            LEDGER_METRIC_REPULSION_APPLIED,
            0,
        );
        ledger_slot.push_telemetry_metric(metric);
        self.action_hub.ledger_slot = ledger_slot;

        let step_decision = if repulsion_active {
            StigmergicActionDecision::RepulsionEscape
        } else if steered && (frame.action_mode_flags & ACTION_MODE_SUBHEAD_ROUTE) != 0 {
            StigmergicActionDecision::SubHeadRoute
        } else if steered {
            StigmergicActionDecision::ExpressPhaseSteer
        } else if (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0 {
            StigmergicActionDecision::InertialLayerSkip
        } else {
            StigmergicActionDecision::PassiveSidecar
        };
        let qos_throttled = frame.dominant_cause_code == ROOT_CAUSE_SYSTEM_QOS_THROTTLE;
        let _ = self.telemetry.evaluate_policy(
            trigger.is_destabilized(),
            quiescent_dwell,
            qos_throttled,
        );
        self.telemetry.record_step(
            step_decision,
            &self.last_horizon,
            &self.last_boundary_delta,
            &self.last_venn_report,
            &self.core_layers.ego_layer,
            edbc_st.active_branch_mode,
            slid_in,
            slid_out,
            escalated,
            qos_throttled,
        );

        Ok((
            frame,
            frame.to_causal_report(),
            slid_in,
            slid_out,
            escalated,
        ))
    }
}

// ===================================================================
// Trait Implementations for Core Architecture (`crate::traits`) Compliance
// ===================================================================

impl PhaseSteering for RuntimeContext {
    #[inline(always)]
    fn head_dim(&self) -> u32 {
        self.dpsr.head_dim()
    }

    #[inline(always)]
    fn num_heads(&self) -> u32 {
        self.dpsr.num_heads()
    }
}

impl EntropyEvaluator for RuntimeContext {
    #[inline]
    fn eval_stateful(
        &mut self,
        token_z_score: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> Result<()> {
        self.edbc.eval_stateful(token_z_score, state_out)
    }
}

impl EdbcBranchController for RuntimeContext {
    #[inline(always)]
    fn active_branch_mode(&self) -> u32 {
        self.edbc.state().active_branch_mode
    }

    #[inline]
    fn set_bifurcation_threshold(&mut self, threshold: f32) {
        self.edbc.set_bifurcation_threshold(threshold);
        self.config.bifurcation_threshold = self.edbc.state().bifurcation_threshold;
    }

    #[inline]
    fn set_hysteresis_band(&mut self, band: f32) {
        self.edbc.set_hysteresis_band(band);
        self.config.hysteresis_band = self.edbc.hysteresis_band;
    }

    #[inline]
    fn set_tau_flux(&mut self, tau_flux: f32) {
        self.edbc.set_tau_flux(tau_flux);
        self.config.tau_flux = self.edbc.tau_flux;
    }
}

impl PointerActionHub for RuntimeContext {
    #[inline]
    fn reset(&mut self) -> Result<()> {
        self.action_hub.reset()
    }

    #[inline]
    fn step_pointer(
        &mut self,
        current_ptr: u64,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> Result<()> {
        let cap = self.config.raw.action_hub_capacity;
        let cur_active = self.action_hub.active_count;
        self.action_hub.active_count =
            branchless_select_u32(cur_active < cap, cur_active.wrapping_add(1), cur_active);
        self.action_hub.step_counter = self.action_hub.step_counter.wrapping_add(1);
        trail_out.record_pointer_transition(current_ptr, 1, 1.0);
        let tp = gcso_tagged_ptr_t(current_ptr);
        let embedded_phase = tp.unpack_phase();
        if embedded_phase != 0 {
            trail_out.accumulate_uniform_phase(embedded_phase);
        }
        trail_out.cluster_id = branchless_select_u32(
            trail_out.target_anchor_id != 0,
            trail_out.target_anchor_id,
            u32::from(tp.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK),
        );
        trail_out.sync_flags_with_current_ptr();
        let slot_idx = (self.action_hub.write_cursor as usize) & (ACTION_HUB_INLINE_TRAILS - 1);
        ActionHubEngine::copy_trail_to_slot_fast(&mut self.action_hub.trails[slot_idx], trail_out);
        self.action_hub.write_cursor =
            self.action_hub.write_cursor.wrapping_add(1) & ((ACTION_HUB_INLINE_TRAILS as u32) - 1);
        Ok(())
    }

    #[inline]
    fn update_cellular_state_express(
        &mut self,
        trail_id: u32,
        phase_delta: gcso_q7_t,
    ) -> Result<()> {
        if trail_id >= self.config.raw.action_hub_capacity {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.action_hub
            .update_cellular_state_express(trail_id, phase_delta)
    }

    #[inline]
    fn pull_trail_to_attractor(
        &mut self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> Result<()> {
        if !branchless_is_finite_f32(pull_force) {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if trail_id < self.action_hub.capacity {
            self.action_hub
                .pull_trail_to_attractor(trail_id, anchor_id, pull_force)?;
        } else {
            let slot_idx = (anchor_id as usize) & (MAX_ATTRACTOR_DOMAINS - 1);
            self.action_hub.slot_weights[slot_idx] =
                branchless_min_f32(branchless_relu_f32(pull_force), MAX_ATTRACTOR_PULL_FORCE);
        }
        Ok(())
    }

    #[inline]
    fn link_hallucinated_trails(&mut self, src_trail_id: u32, dst_trail_id: u32) -> Result<()> {
        if src_trail_id < self.action_hub.capacity && dst_trail_id < self.action_hub.capacity {
            let _ = self
                .action_hub
                .link_hallucinated_trails(src_trail_id, dst_trail_id);
        }
        Ok(())
    }

    #[inline(always)]
    fn capacity(&self) -> u32 {
        self.config.raw.action_hub_capacity
    }

    #[inline(always)]
    fn active_count(&self) -> u32 {
        self.action_hub.active_count
    }
}

impl FastPathLedger for RuntimeContext {
    #[inline]
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_ledger_slot_t,
        config_out: &mut gcso_config_t,
    ) -> Result<()> {
        *config_out = self.config.raw;

        // Count high-intensity telemetry spike events (>= 0x80) across all 8 ring bytes via 1 SWAR op.
        let word = u64::from_le_bytes(slot.telemetry_mini_ledger);
        let spike_count = (word & 0x8080_8080_8080_8080).count_ones();
        let base_ripa = branchless_select_f32(
            config_out.ripa_clamp_max_rad > 0.0,
            config_out.ripa_clamp_max_rad,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        let damping = 1.0 - 0.025 * (spike_count as f32);
        config_out.ripa_clamp_max_rad = branchless_clamp_f32(base_ripa * damping, 0.01, 0.25);
        config_out.ledger_mode = (slot.mode & 0xFF) as u8;
        let _ = LEDGER_HISTORY_LEN;
        Ok(())
    }
}

impl StigmergicChunkOps for RuntimeContext {
    #[inline]
    fn step_chunk(&mut self, _mask: &gcso_paged_bitmask_t, chunk_len: u32) -> Result<()> {
        if chunk_len == 0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.action_hub.step_counter = self
            .action_hub
            .step_counter
            .wrapping_add(u64::from(chunk_len));
        Ok(())
    }
}

impl AttractorField for RuntimeContext {
    #[inline]
    fn anchor_count(&self) -> u32 {
        self.attractor_space.system_field().active_count()
    }

    #[inline]
    fn apply_evaporative_decay(&mut self) {
        self.attractor_space
            .system_field_mut()
            .apply_evaporative_decay();
    }

    #[inline]
    fn total_potential_energy(&self) -> f32 {
        self.attractor_space.system_field().total_potential_energy()
    }

    #[inline]
    fn pressure_ratio(&self) -> f32 {
        self.attractor_space.system_field().pressure_ratio()
    }

    #[inline]
    fn evaporate_weakest_anchor(&mut self) -> Option<u32> {
        self.attractor_space
            .system_field_mut()
            .evaporate_weakest_anchor()
    }

    #[inline]
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> Result<u32> {
        let id = self
            .attractor_space
            .system_field_mut()
            .add_system_prompt_anchor(prompt, weight)?;
        self.action_hub.slot_weights[usize::from(ATTRACTOR_SLOT_SYSTEM_BASELINE)] =
            branchless_relu_f32(weight);
        Ok(id)
    }

    #[inline]
    fn add_embedding_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> Result<u32> {
        self.attractor_space
            .system_field_mut()
            .add_embedding_anchor(embedding, anchor_type, weight)
    }

    #[inline]
    fn add_anchor(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> Result<u32> {
        self.attractor_space
            .system_field_mut()
            .add_anchor(anchor_type, vec, dim)
    }

    #[inline]
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> Result<()> {
        let raw = repulsion_deltas.as_slice();
        let limit = raw.len().min(MAX_ATTENTION_HEADS);
        let mut pattern = [0i8; MAX_ATTENTION_HEADS];
        pattern[..limit].copy_from_slice(&raw[..limit]);

        self.action_hub
            .bloom_filter
            .insert_repulsive_pattern(&pattern);
        let anti_phase = branchless_q7_conjugate_64(&pattern);
        self.set_slot_phase_profile(ATTRACTOR_SLOT_REPULSION, &anti_phase)?;
        self.action_hub.slot_weights[usize::from(ATTRACTOR_SLOT_REPULSION)] =
            branchless_relu_f32(gain);

        self.attractor_space
            .system_field_mut()
            .inject_phase_repulsion(repulsion_deltas, gain)
    }

    #[inline]
    fn aggregate_bottom_up(&mut self) -> Result<u32> {
        self.attractor_space
            .system_field_mut()
            .aggregate_bottom_up()
    }

    #[inline]
    fn clear_anchors(&mut self) -> Result<()> {
        self.attractor_space.system_field_mut().clear_anchors()
    }
}

impl MultiDomainAttractorSpace for RuntimeContext {
    #[inline]
    fn advance_step(&mut self) {
        self.attractor_space.advance_step();
    }

    #[inline]
    fn set_domain_weight(&mut self, domain_id: u32, weight: f32) -> Result<()> {
        let idx = domain_id as usize;
        if idx >= MAX_ATTRACTOR_DOMAINS || !branchless_is_finite_f32(weight) || weight < 0.0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.action_hub.slot_weights[idx] = weight;
        Ok(())
    }

    #[inline]
    fn set_domain_enabled(&mut self, domain_id: u32, enabled: bool) -> Result<()> {
        let idx = domain_id as usize;
        if idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        let bit = 1u8 << (domain_id as u8);
        let enable_mask = (enabled as u8).wrapping_neg();
        self.action_hub.slot_enabled_mask =
            (self.action_hub.slot_enabled_mask & !bit) | (bit & enable_mask);
        Ok(())
    }

    #[inline]
    fn find_dominant_domain(&self) -> (u32, f32) {
        let mut best_id = 0u32;
        let mut best_score = -1.0_f32;
        let mask = self.action_hub.slot_enabled_mask;
        let weights = &self.action_hub.slot_weights;

        let mut i = 0usize;
        while i < MAX_ATTRACTOR_DOMAINS {
            let enabled = ((mask >> i) & 1) != 0;
            let base_w = weights[i];
            let eff_score = branchless_select_f32(enabled, base_w, -1.0);
            let is_better = eff_score > best_score;
            best_id = branchless_select_u32(is_better, i as u32, best_id);
            best_score = branchless_select_f32(is_better, eff_score, best_score);
            i += 1;
        }

        (best_id, branchless_relu_f32(best_score))
    }

    #[inline(always)]
    fn slot_coupling_matrix(&self) -> u64 {
        self.action_hub.coupling_matrix_8x8
    }

    #[inline]
    fn evaluate_slot_force(
        &self,
        domain_id: u32,
        trail: &gcso_pointer_trail_t,
        slot_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> Result<f32> {
        let idx = domain_id as usize;
        if idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        let enabled = ((self.action_hub.slot_enabled_mask >> idx) & 1) != 0;
        if !enabled {
            *slot_phase_out = [0i8; MAX_ATTENTION_HEADS];
            return Ok(0.0);
        }

        let bank_slice = self.action_hub.slot_phase_bank.as_aligned_slice();
        let bank_slot = bank_slice.slot_exact::<MAX_ATTENTION_HEADS>(idx)?;
        if is_phase_profile_nonzero_64(bank_slot) {
            *slot_phase_out = *bank_slot;
            return Ok(self.action_hub.slot_weights[idx]);
        }

        let base_pull = self
            .attractor_space
            .evaluate_multi_domain_superposition(trail, slot_phase_out);
        Ok(self.action_hub.slot_weights[idx].max(base_pull))
    }

    #[inline]
    fn evaluate_multi_domain_superposition(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let (run_w, pull) = self.evaluate_multi_domain_superposition_words(trail);
        store_q7_words_64(composite_phase_out, &run_w);
        pull
    }

    #[inline]
    fn clear_all_domains(&mut self) -> Result<()> {
        self.attractor_space.clear_all_domains()?;
        self.action_hub.slot_phase_bank = AlignedBuffer64::zeroed();
        self.action_hub.slot_weights = DEFAULT_SLOT_WEIGHTS;
        self.action_hub.slot_enabled_mask = 0xFF;
        Ok(())
    }
}

impl PersonaPatcher for RuntimeContext {
    #[inline]
    fn apply_patch(&mut self, patch_data: &[u8]) -> Result<()> {
        if patch_data.is_empty() {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if patch_data.len() >= MAX_ATTENTION_HEADS {
            // SAFETY: `patch_data.len() >= 64` and `[u8; 64]` has the exact same layout and
            // alignment (1) as `[gcso_q7_t; 64]` (`[i8; 64]`).
            let patch_ref: &[gcso_q7_t; MAX_ATTENTION_HEADS] = unsafe {
                &*patch_data
                    .as_ptr()
                    .cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>()
            };
            let pw = load_q7_words_64(patch_ref);
            let sw = self.slot_words_mut(usize::from(ATTRACTOR_SLOT_PERSONA))?;
            swar_or_avx2_q7_add_sat_words_64(sw, &pw);
        } else {
            let mut bank_mut = self.action_hub.slot_phase_bank.as_aligned_slice_mut();
            let persona_slot = bank_mut
                .slot_exact_mut::<MAX_ATTENTION_HEADS>(usize::from(ATTRACTOR_SLOT_PERSONA))?;
            for (h, slot_p) in persona_slot.iter_mut().enumerate() {
                let delta = patch_data[h % patch_data.len()] as gcso_q7_t;
                *slot_p = branchless_q7_add_sat(*slot_p, delta);
            }
        }
        Ok(())
    }
}

impl KvKeyframeCache for RuntimeContext {
    #[inline]
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_kv_keyframe_header_t,
    ) -> Result<()> {
        header_out.token_index = token_index;
        header_out.frame_type = 0;
        Ok(())
    }
}

impl MappedStorage for RuntimeContext {
    #[inline]
    fn serialize_snapshot(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        self.serialize_to_bytes(buffer, required_size)
    }

    #[inline]
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        config_out: &mut gcso_config_t,
    ) -> Result<()> {
        self.deserialize_from_bytes(buffer)?;
        *config_out = self.config.raw;
        Ok(())
    }

    #[inline]
    fn open_mmap(
        &mut self,
        file_path: &str,
        descriptor_out: &mut gcso_mapped_storage_descriptor_t,
    ) -> Result<()> {
        if file_path.is_empty() {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        descriptor_out.mapped_address = 0x1000_0000;
        descriptor_out.file_size_bytes = 4096;
        descriptor_out.fd_handle = 3;
        descriptor_out.flags = 0;
        Ok(())
    }

    #[inline]
    fn close_mmap(&mut self, descriptor: &mut gcso_mapped_storage_descriptor_t) -> Result<()> {
        descriptor.mapped_address = 0;
        descriptor.file_size_bytes = 0;
        descriptor.fd_handle = -1;
        Ok(())
    }

    #[inline]
    fn get_track_offset(&self, track_id: u32) -> Result<u64> {
        let header_size = core::mem::size_of::<gcso_snapshot_header_t>() as u64;
        match track_id {
            0 => Ok(header_size),
            1 => Ok(header_size + 1024),
            2 => Ok(header_size + 2048),
            3 => Ok(header_size + 3072),
            _ => Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT),
        }
    }
}

impl CausalProvenanceSignal for RuntimeContext {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.edbc.causal_provenance_bits() | self.config.raw.causal_provenance_bits()
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(
            self.edbc.causal_anomaly_severity(),
            self.config.raw.causal_anomaly_severity(),
        )
    }
}

impl SystemConstraintSignal for RuntimeContext {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        self.edbc.system_vram_pressure()
    }

    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        self.edbc.system_gpu_frame_overrun()
    }

    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        self.edbc.active_swarm_cells_mask()
    }

    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        self.edbc.recommend_layer_skip()
    }

    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 {
        branchless_max_f32(
            self.config.raw.qdps_min_step_rad_override(),
            self.edbc.qdps_min_step_rad_override(),
        )
    }
}

impl ManifoldGatewaySignal for RuntimeContext {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; crate::traits::NUM_STIGMERGIC_GRANULES]) {
        if self.action_hub.active_count > 0 {
            let (mut frame, words) = self.action_hub.synthesize_gateway_frame_with_words();
            frame.provenance_bits |= self.causal_provenance_bits();
            frame.active_swarm_cell_mask &= self.active_swarm_cells_mask();
            frame.qdps_min_step_rad = self.qdps_min_step_rad_override();
            frame.qdps_min_step_q7 =
                crate::traits::primitives::branchless_rad_to_qdps_step_u8(frame.qdps_min_step_rad);
            if self.recommend_layer_skip() {
                frame.action_mode_flags |= ACTION_MODE_LAYER_SKIP;
            }
            (frame, words)
        } else {
            self.edbc.synthesize_gateway_frame_with_words()
        }
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

// Compile-time verification that `RuntimeContext` satisfies `ActionHub` and
// Gateway / Coordinator trait bounds.
const _: () = {
    const fn assert_full_stack_bounds<
        T: ActionHub
            + ActionHubGatewayExt
            + GatewayPhaseSteeringExt
            + PhaseSteeringClosedLoop
            + EntropyClosedLoopCoordinator
            + CausalProvenanceSignal
            + SystemConstraintSignal
            + ManifoldGatewaySignal,
    >() {
    }
    assert_full_stack_bounds::<RuntimeContext>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_config_validation_and_builder() {
        let config = Config::new()
            .with_head_dim(128)
            .with_num_heads(32)
            .with_paged_block_size(32)
            .with_hysteresis_band(0.2)
            .with_bifurcation_threshold(1.4)
            .with_repulsion_threshold(2);

        assert!(config.validate().is_ok());

        let invalid_config = config.with_head_dim(127);
        assert_eq!(
            invalid_config.validate(),
            Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT)
        );
    }

    #[test]
    fn test_runtime_context_lifecycle_and_serialization() {
        let config = Config::new();
        let mut ctx = RuntimeContext::new(config).unwrap();

        assert!(ctx
            .set_system_prompt_anchor("System prompt anchor test", 1.5)
            .is_ok());
        assert_eq!(ctx.anchor_count(), 1);
        assert!(ctx
            .set_domain_weight(u32::from(ATTRACTOR_SLOT_PERSONA), 2.5)
            .is_ok());
        assert!(ctx
            .set_domain_enabled(u32::from(ATTRACTOR_SLOT_CUSTOM_7), false)
            .is_ok());
        assert_eq!(
            ctx.find_dominant_domain().0,
            u32::from(ATTRACTOR_SLOT_PERSONA)
        );

        let mut trail = gcso_pointer_trail_t::default();
        assert!(ctx.step_token(42, None, None, Some(&mut trail)).is_ok());
        assert_eq!(ctx.step_counter(), 1);
        assert_eq!(trail.current_ptr, 42);

        let mut required_size = 0usize;
        assert_eq!(ctx.serialize_to_bytes(&mut [], &mut required_size), Ok(()));
        assert_eq!(required_size, 320);

        let mut buffer = alloc::vec![0u8; required_size];
        assert!(ctx
            .serialize_to_bytes(&mut buffer, &mut required_size)
            .is_ok());

        assert!(ctx.reset().is_ok());
        assert_eq!(ctx.step_counter(), 0);
        assert_eq!(ctx.anchor_count(), 0);
        assert!(ctx.deserialize_from_bytes(&buffer).is_ok());
        assert_eq!(ctx.step_counter(), 1);
        assert_eq!(ctx.anchor_count(), 1);
        assert_eq!(ctx.slot_enabled_mask(), 0x7F);
        assert!((ctx.slot_weights()[usize::from(ATTRACTOR_SLOT_PERSONA)] - 2.5).abs() < 1e-5);

        // Verify IEEE 802.3 CRC32 payload tamper detection
        buffer[200] ^= 0xFF;
        assert_eq!(
            ctx.deserialize_from_bytes(&buffer),
            Err(crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED)
        );
    }

    #[test]
    fn test_runtime_context_capabilities_and_mapped_storage() {
        let config = Config::new();
        let mut ctx = RuntimeContext::new(config).unwrap();

        assert_eq!(
            ctx.abi_version(),
            (
                crate::abi::GCSO_ABI_VERSION_MAJOR,
                crate::abi::GCSO_ABI_VERSION_MINOR,
                crate::abi::GCSO_ABI_VERSION_PATCH,
            )
        );
        assert_eq!(ctx.abi_version_string(), env!("CARGO_PKG_VERSION"));
        assert_ne!(
            ctx.query_capability() & crate::abi::GCSO_CAP_CPU_BASELINE,
            0
        );

        let mut desc = gcso_mapped_storage_descriptor_t::default();
        assert!(ctx.open_mmap("test.gcso", &mut desc).is_ok());
        assert_eq!(desc.fd_handle, 3);
        assert_eq!(
            ctx.get_track_offset(0).unwrap(),
            core::mem::size_of::<gcso_snapshot_header_t>() as u64
        );
        assert!(ctx.close_mmap(&mut desc).is_ok());
    }

    #[test]
    fn test_runtime_context_vertical_piercing_and_dynamic_slide_in_out() {
        use crate::traits::gateway::{ObjectiveFact, SubjectivePersona, SystemConstraint};
        use crate::traits::rope::{RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget};
        use crate::traits::{AlignedBuffer32, EGO_TRIGGER_BOUNDARY_CHATTER};

        struct MockGeom;
        impl RopeSteeringTarget for MockGeom {
            fn rope_geometry(&self) -> RopeGeometryDescriptor {
                RopeGeometryDescriptor {
                    head_dim: 128,
                    num_heads: 32,
                    rotary_dim: 32,
                    layout: RopePairLayout::Interleaved,
                }
            }
        }

        struct MockSlideInSlot {
            invoked_count: u32,
        }
        impl LatentHorizonSlot for MockSlideInSlot {
            fn horizon_slot_mask(&self) -> u32 {
                1 << 3
            }

            fn on_slide_in_modulate(
                &mut self,
                frame: &mut GatewayFrame,
                stack: &mut AutonomousCoreLayerStack,
                _trigger: &HorizonTriggerInput,
            ) {
                self.invoked_count += 1;
                stack.liminal_layer.active_aperture_mask |= 1 << 7;
                frame.phase_deltas.fill(24);
            }
        }

        struct MockEscalator {
            calls: u32,
        }
        impl HorizonEscalationProvider for MockEscalator {
            fn try_escalate_horizon(
                &mut self,
                _req: &HorizonEscalationRequest,
                slot_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
            ) -> Result<Option<f32>> {
                self.calls += 1;
                slot_phase_out.fill(18);
                Ok(Some(1.25))
            }
        }

        let mut ctx = RuntimeContext::new(Config::new()).unwrap();
        let mut q_buf = AlignedBuffer32::new([1.0_f32; 128 * 32]);
        let mut trail = gcso_pointer_trail_t::default();
        let signal = (
            SystemConstraint::new((), 0.1, 0.0, 0xFF),
            ObjectiveFact::new([12i8; 64], 1.5, 0x1000, 7),
            SubjectivePersona::new([0i8; 64], 0.0, 0, 0),
        );
        let mut slot = MockSlideInSlot { invoked_count: 0 };
        let mut escalator = MockEscalator { calls: 0 };

        // 1. Nominal coherent state -> no slide-in, no escalation.
        let (_f1, _r1, in1, out1, esc1) = ctx
            .pierce_and_orchestrate_step(
                &signal,
                &MockGeom,
                &mut slot,
                &mut escalator,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert_eq!(in1, 0);
        assert_eq!(out1, 0);
        assert!(!esc1);
        assert_eq!(slot.invoked_count, 0);
        assert_eq!(ctx.active_horizon_slots_mask(), 0);

        // 2. Destabilize via Layer 2 Ego trigger -> slot slides in & external escalator swaps lane 7.
        ctx.core_layers_mut().ego_layer.last_intervened_mask = EGO_TRIGGER_BOUNDARY_CHATTER;
        let (f2, _r2, in2, out2, esc2) = ctx
            .pierce_and_orchestrate_step(
                &signal,
                &MockGeom,
                &mut slot,
                &mut escalator,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert_eq!(in2, 1 << 3);
        assert_eq!(out2, 0);
        assert!(esc2);
        assert_eq!(slot.invoked_count, 1);
        assert_eq!(escalator.calls, 1);
        assert_eq!(f2.phase_deltas[0], 24);
        assert_ne!(
            ctx.orchestrator_status().liminal_aperture_mask & (1 << 7),
            0
        );
        assert_eq!(ctx.active_horizon_slots_mask(), 1 << 3);

        // 3. Stabilize -> slot automatically retracts (slides out) and lane 7 is disabled.
        ctx.core_layers_mut().ego_layer.last_intervened_mask = 0;
        ctx.last_boundary_delta_mut().stationary_shortcut_ready = 1;
        let (_f3, _r3, in3, out3, esc3) = ctx
            .pierce_and_orchestrate_step(
                &signal,
                &MockGeom,
                &mut slot,
                &mut escalator,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert_eq!(in3, 0);
        assert_eq!(out3, 1 << 3);
        assert!(!esc3);
        assert_eq!(ctx.active_horizon_slots_mask(), 0);
        assert_eq!(ctx.slot_enabled_mask() & (1 << ATTRACTOR_SLOT_CUSTOM_7), 0);
        assert!(ctx.horizon_dwell().quiescent_dwell_steps >= 1);
    }

    #[test]
    fn test_latent_horizon_slot_tuple_isolation_dwell_and_slide_out_cleanup() {
        use crate::traits::gateway::{
            apply_latent_horizon_slots, AutonomousCoreLayerStack, GatewayFrame, HorizonDwellState,
            HorizonTriggerInput,
        };

        struct FastSlot {
            slid_out_calls: u32,
        }
        impl LatentHorizonSlot for FastSlot {
            fn horizon_slot_mask(&self) -> u32 {
                1 << 1
            }

            fn should_slide_in(&self, trigger: &HorizonTriggerInput) -> bool {
                trigger.repulsion_severity > 0.4
            }

            fn on_slide_in_modulate(
                &mut self,
                frame: &mut GatewayFrame,
                _stack: &mut AutonomousCoreLayerStack,
                _trigger: &HorizonTriggerInput,
            ) {
                frame.phase_deltas[1] = 32;
            }

            fn on_slide_out(
                &mut self,
                frame: &mut GatewayFrame,
                _stack: &mut AutonomousCoreLayerStack,
                _trigger: &HorizonTriggerInput,
            ) {
                self.slid_out_calls += 1;
                frame.phase_deltas[1] = 0;
            }
        }

        struct HysteresisSlot {
            modulate_calls: u32,
        }
        impl LatentHorizonSlot for HysteresisSlot {
            fn horizon_slot_mask(&self) -> u32 {
                1 << 4
            }

            fn min_slide_in_dwell(&self) -> u16 {
                2
            }

            fn min_slide_out_dwell(&self) -> u16 {
                2
            }

            fn on_slide_in_modulate(
                &mut self,
                frame: &mut GatewayFrame,
                stack: &mut AutonomousCoreLayerStack,
                _trigger: &HorizonTriggerInput,
            ) {
                self.modulate_calls += 1;
                stack.liminal_layer.active_aperture_mask |= 1 << 4;
                frame.phase_deltas[4] = 48;
            }
        }

        let mut pair = (
            FastSlot { slid_out_calls: 0 },
            HysteresisSlot { modulate_calls: 0 },
        );
        let mut active_mask = 0u32;
        let mut dwell = HorizonDwellState::default();
        let mut frame = GatewayFrame::default();
        let mut stack = AutonomousCoreLayerStack::new();

        // Step 1: Destabilized (step 1 of dwell, high repulsion) -> only FastSlot slides in!
        let mut t1 = HorizonTriggerInput {
            repulsion_severity: 0.6,
            ..HorizonTriggerInput::default()
        };
        let (u1, q1) = dwell.step_branchless(t1.is_destabilized(), t1.stationary_shortcut_ready);
        t1.unstable_dwell_steps = u1;
        t1.quiescent_dwell_steps = q1;
        let (in1, out1) =
            apply_latent_horizon_slots(&mut pair, &mut active_mask, &t1, &mut frame, &mut stack);
        assert_eq!(in1, 1 << 1);
        assert_eq!(out1, 0);
        assert_eq!(active_mask, 1 << 1);
        assert_eq!(pair.1.modulate_calls, 0);

        // Step 2: Destabilized persists (step 2 of dwell) -> HysteresisSlot now slides in independently.
        let mut t2 = HorizonTriggerInput {
            repulsion_severity: 0.6,
            ..HorizonTriggerInput::default()
        };
        let (u2, q2) = dwell.step_branchless(t2.is_destabilized(), t2.stationary_shortcut_ready);
        t2.unstable_dwell_steps = u2;
        t2.quiescent_dwell_steps = q2;
        let (in2, out2) =
            apply_latent_horizon_slots(&mut pair, &mut active_mask, &t2, &mut frame, &mut stack);
        assert_eq!(in2, 1 << 4);
        assert_eq!(out2, 0);
        assert_eq!(active_mask, (1 << 1) | (1 << 4));
        assert_eq!(pair.1.modulate_calls, 1);

        // Step 3: Quiescent (step 1 of dwell) -> FastSlot slides out with cleanup, HysteresisSlot holds!
        let mut t3 = HorizonTriggerInput {
            stationary_shortcut_ready: true,
            ..HorizonTriggerInput::default()
        };
        let (u3, q3) = dwell.step_branchless(t3.is_destabilized(), t3.stationary_shortcut_ready);
        t3.unstable_dwell_steps = u3;
        t3.quiescent_dwell_steps = q3;
        let (in3, out3) =
            apply_latent_horizon_slots(&mut pair, &mut active_mask, &t3, &mut frame, &mut stack);
        assert_eq!(in3, 0);
        assert_eq!(out3, 1 << 1);
        assert_eq!(active_mask, 1 << 4);
        assert_eq!(pair.0.slid_out_calls, 1);
        assert_eq!(frame.phase_deltas[1], 0);

        // Step 4: Quiescent persists (step 2 of dwell) -> HysteresisSlot retracts.
        let mut t4 = HorizonTriggerInput {
            stationary_shortcut_ready: true,
            ..HorizonTriggerInput::default()
        };
        let (u4, q4) = dwell.step_branchless(t4.is_destabilized(), t4.stationary_shortcut_ready);
        t4.unstable_dwell_steps = u4;
        t4.quiescent_dwell_steps = q4;
        let (in4, out4) =
            apply_latent_horizon_slots(&mut pair, &mut active_mask, &t4, &mut frame, &mut stack);
        assert_eq!(in4, 0);
        assert_eq!(out4, 1 << 4);
        assert_eq!(active_mask, 0);
    }

    #[test]
    fn test_residual_spillover_slot_cache_promotion_eviction_and_escalation() {
        use crate::crystal::DissipativeOverflowReport;
        use crate::traits::gateway::{
            DefaultResidualSpilloverSlot, ObjectiveFact, SubjectivePersona, SystemConstraint,
        };
        use crate::traits::rope::{RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget};
        use crate::traits::{AlignedBuffer32, EGO_TRIGGER_INTERNAL_CONFLICT};

        struct Geom;
        impl RopeSteeringTarget for Geom {
            fn rope_geometry(&self) -> RopeGeometryDescriptor {
                RopeGeometryDescriptor {
                    head_dim: 128,
                    num_heads: 32,
                    rotary_dim: 32,
                    layout: RopePairLayout::Interleaved,
                }
            }
        }

        let mut ctx = RuntimeContext::new(Config::new()).unwrap();
        let mut q_buf = AlignedBuffer32::new([1.0_f32; 128 * 32]);
        let mut trail = gcso_pointer_trail_t::default();
        let signal = (
            SystemConstraint::new((), 0.2, 0.0, 0xFF),
            ObjectiveFact::new([20i8; 64], 1.5, 0x3000, 17),
            SubjectivePersona::new([0i8; 64], 0.0, 0, 0),
        );

        let mut spillover = DefaultResidualSpilloverSlot::default();
        let mut fallback_esc = ();
        spillover.stage_overflow_report(&DissipativeOverflowReport {
            overflowed: true,
            residual_words: [0x1212_1212_1212_1212; 8],
            ..DissipativeOverflowReport::default()
        });

        // 1. Step 1 destabilized -> held back by slide_in_dwell (2).
        ctx.core_layers_mut().ego_layer.last_intervened_mask = EGO_TRIGGER_INTERNAL_CONFLICT;
        let (_f1, _, in1, _, _) = ctx
            .pierce_and_orchestrate_step(
                &signal,
                &Geom,
                &mut spillover,
                &mut fallback_esc,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert_eq!(in1, 0);
        assert_eq!(spillover.spillover_modulations, 0);

        // 2. Step 2 sustained destabilization -> spillover slides in (cache miss -> orthogonal elevation).
        let (_f2, _, in2, _, _) = ctx
            .pierce_and_orchestrate_step(
                &signal,
                &Geom,
                &mut spillover,
                &mut fallback_esc,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert_eq!(in2, spillover.slot_bit);
        assert_eq!(spillover.spillover_modulations, 1);

        // 3. Pollution guard: failed resolution stores nothing in experience cache.
        let verified_phases = [0x12i8; 64];
        let (stored_bad, _) = spillover.commit_verified_experience(0, &verified_phases, 0.9, false);
        assert!(!stored_bad);
        assert_eq!(spillover.cache_stores, 0);

        // 4. Verified resolution promotes solution into Tier-1 RAM cache.
        let (stored_ok, _) = spillover.commit_verified_experience(0, &verified_phases, 0.9, true);
        assert!(stored_ok);
        assert_eq!(spillover.cache_stores, 1);

        // 5. Subsequent destabilization hits Tier-1 cache AND feeds HorizonEscalationProvider lane 7.
        let mut escalator = spillover.clone();
        let (_f3, _, _, _, esc3) = ctx
            .pierce_and_orchestrate_step(
                &signal,
                &Geom,
                &mut spillover,
                &mut escalator,
                &mut trail,
                Some(&mut q_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert!(esc3);
        assert_eq!(spillover.cache_hits, 1);

        // 6. Contradiction penalty halves pheromone and evicts below floor on repeated penalty.
        spillover.evict_pheromone_floor = 0.8;
        let (_, evicted) = spillover.commit_verified_experience(0, &verified_phases, 0.0, false);
        assert_eq!(evicted, 1);
        assert!(spillover.cache_evictions >= 1);
    }
}
