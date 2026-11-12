// name: src/core/src/runtime.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-level configuration builder ([`Config`]) and unified full-stack runtime context
//! ([`RuntimeContext`]) implementing [`ActionHub`], [`HotPathObserver`], [`ColdPathObserver`],
//! [`ManifoldGatewaySignal`], and [`FullStackGcsoEngine`].

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
    gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_daes_slot_t,
    gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t,
    gcso_pspm_config_t, gcso_q7_t, gcso_snapshot_header_t, gcso_srl_descriptor_t,
    gcso_tagged_ptr_t, gcso_zimms_descriptor_t, GCSO_CAP_CPU_BASELINE, GCSO_CAP_CUDA_BACKEND,
    GCSO_CAP_METAL_BACKEND, GCSO_CAP_VULKAN_BACKEND,
};
use crate::action_hub::{
    ActionHubEngine, ActionOrchestrationReport, StigmergicThresholds, ACTION_HUB_INLINE_TRAILS,
};
use crate::attractor::{MultiAttractorSpace, MAX_ANCHORS_PER_FIELD, MAX_ATTRACTOR_DOMAINS};
use crate::entropy_control::{EntropyClosedLoopCoordinator, EntropyController};
use crate::phase_steering::{PhaseSteeringClosedLoop, PhaseSteeringEngine};
use crate::traits::{
    ActionHub, ActionHubGatewayExt, AlignedBuffer64, AlignedSlice16, AlignedSlice32,
    AlignedSliceMut16, AlignedSliceMut32, AttractorField, AttractorSlotSynergyCoordinator,
    BottomUpMacroCrystalCoordinator, CausalProvenanceSignal, ColdPathObserver,
    ContrastivePhaseBloomFilter, DaesScratchpad, DnpPacketDispatcher, DynamicLayerSkipper,
    EdbcBranchController, EntropyEvaluator, EntropyGatedPhaseLookup, FractalCascadeCoordinator,
    FullStackGcsoEngine, GatewayFrame, GatewayPhaseSteeringExt, GcsoResult, GcsoRuntimeContext,
    HierarchicalTreeTable, HotPathObserver, L2pSvdProjector, ManifoldGatewaySignal,
    MultiDomainAttractorSpace, PersonaPatcher, PhaseSteering, Pod, PointerActionHub, PprcCache,
    PspmRouter, SheafCohomologyEvaluator, SrlAdapter, StigmergicActionPtrOps,
    StigmergicCohomologyCoordinator, StigmergicTrailOps, SwarmCellChunk, SystemCapabilityQuery,
    SystemConstraintSignal, VisualAttentionSteering, ZimmsStorage, ACTION_MODE_LAYER_SKIP,
    ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};

use crate::traits::branchless::{
    branchless_fast_transition_log_prob, branchless_q7_inhibit_words_64,
    coherence_from_word_arrays_with_both_uniform_flags_64, swar_slerp_words_with_uniform_flag_64,
    swar_words_equal_64, swar_words_nonzero_64,
};
use crate::traits::primitives::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_max_f32, branchless_min_f32, branchless_q7_add_sat, branchless_q7_conjugate_64,
    branchless_relu_f32, branchless_select_f32, branchless_select_u32, branchless_select_u8,
    fast_inv_pos_normal_f32, is_phase_profile_nonzero_64, load_q7_words_64, store_q7_words_64,
    swar_q7_add_sat_x8, trail_phase_words_mut, trail_phase_words_ref, SWAR_SPLAT_BYTE_64,
};
use crate::traits::primitives::{
    finalize_snapshot_header_crc32, init_snapshot_header, verify_snapshot_payload_crc32,
    ATTRACTOR_SLOT_ID_MASK, DAES_MINI_LEDGER_LEN, DEFAULT_ACTION_HUB_CAPACITY, DEFAULT_ENTROPY_EPS,
    DEFAULT_PAGED_BLOCK_SIZE, DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD,
    MAX_ATTENTION_HEADS, Q7_PHASE_SCALE_RAD,
};
use crate::traits::{
    accumulate_q7_slice_into_words_64, select_rewire_bridge_and_modulate_with_preloaded_words_64,
    DAES_METRIC_FAST_PATH_HIT, DAES_METRIC_LAYER_SKIP_COMPENSATED, DAES_METRIC_PHASE_STEERED,
    DAES_METRIC_REPULSION_APPLIED, DAES_METRIC_SLOT_SYNERGY_ACTIVE, MAX_ATTRACTOR_PULL_FORCE,
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
    /// Creates a new default configuration instance strictly adhering to GCSO v0.1.1 invariants.
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
                daes_mode: 0,
                reserved_flags: 0,
                reserved: [0; 24],
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

    /// Configures DAES scratchpad operating mode (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC).
    #[inline(always)]
    #[must_use]
    pub const fn with_daes_mode(mut self, daes_mode: u8) -> Self {
        self.raw.daes_mode = daes_mode;
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
    /// `daes_mode > 2`, capacity bounds are zero, or any floating-point threshold is non-finite
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
            & (self.raw.daes_mode <= 2)
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

/// High-level safe Rust runtime context integrating [`ActionHub`],
/// [`HotPathObserver`] (Micro Observer: DPSR, DAES, PSPM, SRL, Contrastive Phase Bloom Filter,
/// Limina bridging, and Bayesian pointer-trail Attention modulation), and [`ColdPathObserver`]
/// (Macro Observer: EDBC, Single Unified Attractor Manifold, 8-Lane Working Cache Window,
/// Sheaf Cohomology $H^1$, L2P-SVD, PPRC, and ZIMMS Storage).
///
/// Because `RuntimeContext` implements all foundational Nano/Micro/Mezzo/Macro traits in
/// [`crate::traits`], it automatically receives the blanket implementations of:
/// - [`ActionHub`]
/// - [`crate::traits::HotPathExecutionEngine`] / [`HotPathObserver`] / [`FractalCascadeCoordinator`]
/// - [`crate::traits::ColdPathOrchestrator`] / [`ColdPathObserver`]
/// - [`AttractorSlotSynergyCoordinator`]
/// - [`StigmergicCohomologyCoordinator`]
/// - [`BottomUpMacroCrystalCoordinator`]
/// - [`FullStackGcsoEngine`]
#[derive(Debug)]
pub struct RuntimeContext {
    config: Config,
    action_hub: ActionHubEngine,
    dpsr: PhaseSteeringEngine,
    edbc: EntropyController,
    attractor_space: MultiAttractorSpace,
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
        })
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

    /// Returns an immutable reference to the active [`StigmergicThresholds`] ladder.
    #[inline(always)]
    #[must_use]
    pub const fn stigmergic_thresholds(&self) -> &StigmergicThresholds {
        &self.action_hub.thresholds
    }

    /// Validates and updates the active [`StigmergicThresholds`] ladder.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if any threshold is non-finite or out of order.
    #[inline]
    pub fn set_stigmergic_thresholds(&mut self, thresholds: StigmergicThresholds) -> Result<()> {
        thresholds.validate()?;
        self.action_hub.thresholds = thresholds;
        Ok(())
    }

    /// Executes a closed-loop **EDBC Entropy Evaluation + Stigmergic Threshold Action Selection**
    /// step via [`ActionHubEngine::orchestrate_next_action_step`] in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// # Errors
    /// Propagates any error from [`EntropyEvaluator::eval_stateful`] or
    /// [`ActionHubEngine::orchestrate_next_action_step`].
    #[inline]
    pub fn step_stigmergic_action_orchestration(
        &mut self,
        input_tagged_ptr: gcso_tagged_ptr_t,
        token_z_score: f32,
        trail_inout: &mut gcso_pointer_trail_t,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> Result<ActionOrchestrationReport> {
        let mut edbc_state = gcso_edbc_state_t::default();
        self.edbc.eval_stateful(token_z_score, &mut edbc_state)?;
        self.attractor_space.advance_step();
        let (dom_slot, _dom_w) = self.find_dominant_domain();
        let dom_slot_u8 = (dom_slot as u8) & ATTRACTOR_SLOT_ID_MASK;

        self.action_hub.orchestrate_next_action_step(
            input_tagged_ptr,
            edbc_state.moving_z_entropy,
            dom_slot_u8,
            trail_inout,
            attention_scores,
            query_tensor,
            key_tensor,
            logits_opt,
        )
    }

    /// Returns an immutable reference to the underlying [`PhaseSteeringEngine`].
    #[inline(always)]
    #[must_use]
    pub const fn phase_steering_engine(&self) -> &PhaseSteeringEngine {
        &self.dpsr
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

    /// Returns an immutable reference to the active [`gcso_daes_slot_t`] scratchpad.
    #[inline(always)]
    #[must_use]
    pub const fn daes_slot(&self) -> &gcso_daes_slot_t {
        &self.action_hub.daes_slot
    }

    /// Returns a mutable reference to the active [`gcso_daes_slot_t`] scratchpad.
    #[inline(always)]
    #[must_use]
    pub fn daes_slot_mut(&mut self) -> &mut gcso_daes_slot_t {
        &mut self.action_hub.daes_slot
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

    /// Returns an immutable reference to the 64-byte (`8 x u64`) aligned word view of slot `slot_idx`.
    #[allow(dead_code)]
    #[inline(always)]
    fn slot_words_ref(
        &self,
        slot_idx: usize,
    ) -> Result<&[u64; crate::traits::NUM_SWARM_MICRO_CELLS]> {
        if slot_idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.action_hub.slot_phase_bank` is `AlignedBuffer64<gcso_q7_t, 512>`, so every
        // 64-byte slot at byte offset `slot_idx * 64` is 64-byte aligned (`>= 8-byte` alignment of `u64`).
        Ok(unsafe {
            &*(self
                .action_hub
                .slot_phase_bank
                .as_slice()
                .as_ptr()
                .add(slot_idx * MAX_ATTENTION_HEADS)
                as *const [u64; crate::traits::NUM_SWARM_MICRO_CELLS])
        })
    }

    /// Returns a mutable reference to the 64-byte (`8 x u64`) aligned word view of slot `slot_idx`.
    #[inline(always)]
    fn slot_words_mut(
        &mut self,
        slot_idx: usize,
    ) -> Result<&mut [u64; crate::traits::NUM_SWARM_MICRO_CELLS]> {
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
                as *mut [u64; crate::traits::NUM_SWARM_MICRO_CELLS])
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
    /// Single Unified Attractor Manifold superposition, Contrastive Phase Bloom Filter repulsion,
    /// and DAES telemetry logging in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor`
    /// alias in memory or if tensor dimensions mismatch `head_dim * num_heads`.
    /// Returns [`crate::abi::GCSO_ERROR_EDBC_SINGULARITY`] if `token_z_score` is non-finite.
    #[inline]
    pub fn step_token_with_zscore(
        &mut self,
        token_id: u32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        phase_deltas: Option<&AlignedSlice16<'_, gcso_q7_t>>,
        token_z_score: f32,
        trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> Result<()> {
        if let (Some(q), Some(k)) = (query_tensor.as_deref(), key_tensor.as_deref()) {
            q.ensure_disjoint_mut(k)?;
        }

        match (query_tensor, key_tensor, phase_deltas) {
            (Some(q), Some(k), Some(p)) => {
                self.dpsr.apply_phase_steering_fused(q, k, p)?;
            }
            (Some(q), None, Some(p)) => {
                self.dpsr.apply_phase_steering(q, p)?;
            }
            (None, Some(k), Some(p)) => {
                self.dpsr.apply_phase_steering(k, p)?;
            }
            _ => {}
        }

        let mut state_out = gcso_edbc_state_t::default();
        self.edbc.eval_stateful(token_z_score, &mut state_out)?;

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

            if let Some(p) = phase_deltas {
                accumulate_q7_slice_into_words_64(&mut phase_words, p.as_slice());
            }
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
            *crate::traits::primitives::trail_phase_words_mut(trail) = phase_words;
            trail.flags = state_out.active_branch_mode
                | branchless_select_u32(repulsion_active, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        }

        let metric = DAES_METRIC_PHASE_STEERED
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0);
        self.action_hub.daes_slot.push_telemetry_metric(metric);

        Ok(())
    }

    /// Executes a **Single-Manifold Unified-Field** per-token cascade directly over an arbitrary
    /// slice of anchors (`anchor_phases`, `anchor_base_energies`) without requiring an 8-slot
    /// domain restriction:
    /// 1. Evaluates EDBC entropy on `token_z_score`.
    /// 2. Steps `input_tagged_ptr` with DAES fast-path shortcut lookup and `bloom_filter` repulsion.
    /// 3. Folds all unified-field anchors via [`PointerActionHub::orchestrate_unified_field_resonance`],
    ///    binding the 32-bit dominant resonant anchor ID into `trail_out.target_anchor_id`.
    /// 4. Optionally modulates candidate memory Attention scores (`attention_scores`) by
    ///    Attention $\times$ DPSR phase interference.
    /// 5. Applies 8-lane SWAR QDPS filtering and RIPA soft-clamped phase steering to `query_tensor`
    ///    and/or `key_tensor`.
    ///
    /// Returns `(repulsion_density, final_pull_force, mean_coherence, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Propagates any error from [`EntropyEvaluator::eval_stateful`] or
    /// [`FractalCascadeCoordinator::execute_unified_field_token_cascade`].
    #[inline]
    pub fn step_unified_field_token_cascade(
        &mut self,
        input_tagged_ptr: gcso_tagged_ptr_t,
        token_z_score: f32,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        attention_interference_gain: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> Result<(u8, f32, f32, u32)> {
        let mut edbc_state = gcso_edbc_state_t::default();
        self.edbc.eval_stateful(token_z_score, &mut edbc_state)?;
        self.attractor_space.advance_step();

        let rep_thresh = self.config.repulsion_threshold;
        let rep_scale = self.config.repulsion_scale_u7;
        let qdps_min = self.config.raw.qdps_min_step_rad;
        let ripa_max = self.config.raw.ripa_clamp_max_rad;

        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min,
            ripa_max,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, eff_ptr) = self
            .action_hub
            .daes_slot
            .resolve_or_pass_in_place(input_tagged_ptr);
        self.step_pointer(eff_ptr.raw(), trail_out)?;
        let (density, repulsion_active) = self
            .action_hub
            .bloom_filter
            .apply_phase_conjugate_repulsion_words(
                trail_phase_words_mut(trail_out),
                rep_thresh,
                rep_scale,
            );
        let init_w = *trail_phase_words_ref(trail_out);
        let (final_pull, mean_coh, active_cnt, raw_words) = self
            .orchestrate_unified_field_resonance_from_words(
                init_w,
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
                trail_out,
            )?;
        let steered = self.steer_qdps_words_prevalidated_core(
            &raw_words,
            head_dim,
            num_heads,
            qdps_min,
            ripa_max,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(active_cnt > 0, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        self.action_hub.daes_slot.push_telemetry_metric(metric);

        if let Some(scores) = attention_scores {
            if !anchor_phases.is_empty() && !scores.is_empty() {
                let _ = self.modulate_hub_attention_scores_by_unified_field(
                    scores,
                    trail_out,
                    anchor_phases,
                    attention_interference_gain,
                )?;
            }
        }

        Ok((density, final_pull, mean_coh, trail_out.target_anchor_id))
    }

    /// Executes the closed-loop **Autonomous Pointer-Trail Selection + Bayes-Attention
    /// Interference** cascade uniting **Hypothesis 1** (Persona as Joint Probability Chain Rule
    /// $\times$ Attention phase interference) and **Hypothesis 2** (Intelligence as Stigmergic
    /// Pointer-Trail rewiring in [`ActionHub`]) in zero-allocation ($C_{\mathrm{alloc}} = 0$) mode.
    ///
    /// Returns `(selected_trail_id, bayesian_score, transition_log_prob, dominant_anchor_id, found_coherent_transition)`.
    ///
    /// # Errors
    /// Propagates any error from [`EntropyEvaluator::eval_stateful`] or
    /// [`FractalCascadeCoordinator::execute_autonomous_pointer_trail_attention_cascade`].
    #[inline]
    pub fn step_autonomous_pointer_trail_attention_cascade(
        &mut self,
        src_trail_id: u32,
        input_tagged_ptr: gcso_tagged_ptr_t,
        token_z_score: f32,
        candidate_trails: &mut [gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> Result<(u32, f32, f32, u32, bool)> {
        let mut edbc_state = gcso_edbc_state_t::default();
        self.edbc.eval_stateful(token_z_score, &mut edbc_state)?;
        self.attractor_space.advance_step();

        let rep_thresh = self.config.repulsion_threshold;
        let rep_scale = self.config.repulsion_scale_u7;
        let qdps_min = self.config.raw.qdps_min_step_rad;
        let ripa_max = self.config.raw.ripa_clamp_max_rad;

        if candidate_trails.is_empty()
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min,
            ripa_max,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, eff_ptr) = self
            .action_hub
            .daes_slot
            .resolve_or_pass_in_place(input_tagged_ptr);
        self.step_pointer(eff_ptr.raw(), trail_out)?;
        let (_density, mut repulsion_active) = self
            .action_hub
            .bloom_filter
            .apply_phase_conjugate_repulsion_words(
                trail_phase_words_mut(trail_out),
                rep_thresh,
                rep_scale,
            );
        let init_w = *trail_phase_words_ref(trail_out);
        let (_final_pull, _mean_anc_coh, active_anchors, pre_bridge_words) = self
            .orchestrate_unified_field_resonance_from_words(
                init_w,
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
                trail_out,
            )?;

        let scores_slice_opt = attention_scores
            .filter(|s| !s.is_empty())
            .map(AlignedSliceMut32::as_mut_slice);

        let (selected_id, bayes_score, _mean_coh, _coh_cnt, found_coherent, final_words, best_coh) =
            select_rewire_bridge_and_modulate_with_preloaded_words_64(
                trail_out,
                pre_bridge_words,
                candidate_trails,
                scores_slice_opt,
                head_dim as u32,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                interpolation_u7,
                repulsion_penalty,
                0,
                None,
                0.0,
            );

        let mut log_prob = 0.0_f32;
        if found_coherent {
            self.link_hallucinated_trails(src_trail_id, selected_id)?;
            let dst_idx = (selected_id as usize).min(candidate_trails.len() - 1);
            let dst_trail = &mut candidate_trails[dst_idx];
            log_prob = branchless_fast_transition_log_prob(best_coh);
            *trail_phase_words_mut(dst_trail) = final_words;
            let dst_has_anchor = dst_trail.target_anchor_id != 0;
            let inherited_anchor = branchless_select_u32(
                dst_has_anchor,
                dst_trail.target_anchor_id,
                trail_out.target_anchor_id,
            );
            trail_out.target_anchor_id = inherited_anchor;
            dst_trail.target_anchor_id = inherited_anchor;
            if inherited_anchor != 0 {
                trail_out.cluster_id = inherited_anchor;
                dst_trail.cluster_id = inherited_anchor;
            }
            trail_out.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            dst_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            dst_trail.sync_flags_with_current_ptr();
        } else {
            repulsion_active = true;
        }
        trail_out.sync_flags_with_current_ptr();

        let steered = self.steer_qdps_words_prevalidated_core(
            &final_words,
            head_dim,
            num_heads,
            qdps_min,
            ripa_max,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(
                (active_anchors > 0) | found_coherent,
                DAES_METRIC_SLOT_SYNERGY_ACTIVE,
                0,
            );
        self.action_hub.daes_slot.push_telemetry_metric(metric);

        Ok((
            selected_id,
            bayes_score,
            log_prob,
            trail_out.target_anchor_id,
            found_coherent,
        ))
    }

    /// Executes the Pointer-Driven Stigmergic Mode Cascade on this context's internal DAES
    /// scratchpad and Contrastive Phase Bloom Filter in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`], [`crate::abi::GCSO_ERROR_NULL_POINTER`],
    /// or [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`] if tensor slices or descriptors are invalid.
    #[inline]
    pub fn step_pointer_mode_cascade(
        &mut self,
        input_tagged_ptr: gcso_tagged_ptr_t,
        token_z_score: f32,
        low_entropy_skip_thresh: f32,
        high_density_srl_thresh: f32,
        layer_skip_comp_profile: Option<&AlignedSlice16<'_, gcso_q7_t>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        pspm_config: Option<&gcso_pspm_config_t>,
        srl_activation_and_desc: Option<(&AlignedSlice32<'_, f32>, &gcso_srl_descriptor_t)>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> Result<(gcso_tagged_ptr_t, u8)> {
        let mut edbc_state = gcso_edbc_state_t::default();
        self.edbc.eval_stateful(token_z_score, &mut edbc_state)?;
        self.attractor_space.advance_step();

        let (dom_slot, _dom_weight) = self.find_dominant_domain();
        let dominant_slot_u8 = (dom_slot as u8) & ATTRACTOR_SLOT_ID_MASK;

        let rep_thresh = self.config.repulsion_threshold;
        let rep_scale = self.config.repulsion_scale_u7;
        let qdps_min = self.config.raw.qdps_min_step_rad;
        let ripa_max = self.config.raw.ripa_clamp_max_rad;

        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min,
            ripa_max,
            Some(&*query_tensor),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, eff_ptr) = self
            .action_hub
            .daes_slot
            .resolve_or_pass_in_place(input_tagged_ptr);
        self.step_pointer(eff_ptr.raw(), trail_out)?;
        self.finish_pointer_cascade_tail(
            edbc_state.moving_z_entropy,
            dominant_slot_u8,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
            rep_thresh,
            rep_scale,
            qdps_min,
            ripa_max,
            head_dim,
            num_heads,
            bypass_hit,
            false,
            0,
            layer_skip_comp_profile,
            query_tensor,
            key_tensor,
            pspm_config,
            srl_activation_and_desc,
            trail_out,
        )
    }

    /// Executes the full Slot-Bank Pointer Cascade integrating [`RuntimeContext`]'s internal
    /// 8-lane Q7 working phase bank (`slot_phase_bank`), cross-lane coupling matrix
    /// (`slot_coupling_matrix`), mutual inhibition from [`ATTRACTOR_SLOT_REPULSION`], EDBC
    /// entropy evaluation, and DAES telemetry logging in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// Returns `(evolved_tagged_ptr, bloom_repulsion_density, effective_pull_force, phase_coherence)`.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`], [`crate::abi::GCSO_ERROR_NULL_POINTER`],
    /// or [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`] if slot indices, tensor slices, or descriptors are invalid.
    #[inline]
    pub fn step_slot_bank_pointer_cascade(
        &mut self,
        input_tagged_ptr: gcso_tagged_ptr_t,
        secondary_slot: u8,
        synergy_gain: f32,
        inhibition_strength: f32,
        token_z_score: f32,
        low_entropy_skip_thresh: f32,
        high_density_srl_thresh: f32,
        layer_skip_comp_profile: Option<&AlignedSlice16<'_, gcso_q7_t>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        pspm_config: Option<&gcso_pspm_config_t>,
        srl_activation_and_desc: Option<(&AlignedSlice32<'_, f32>, &gcso_srl_descriptor_t)>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> Result<(gcso_tagged_ptr_t, u8, f32, f32)> {
        let mut edbc_state = gcso_edbc_state_t::default();
        self.edbc.eval_stateful(token_z_score, &mut edbc_state)?;
        self.attractor_space.advance_step();

        let (dom_slot, _dom_weight) = self.find_dominant_domain();
        let dominant_slot_u8 = (dom_slot as u8) & ATTRACTOR_SLOT_ID_MASK;

        let mut gated_weights = [0.0f32; MAX_ATTRACTOR_DOMAINS];
        let mask = self.action_hub.slot_enabled_mask;
        let weights = &self.action_hub.slot_weights;
        let mut i = 0usize;
        while i < MAX_ATTRACTOR_DOMAINS {
            let enabled = ((mask >> i) & 1) != 0;
            gated_weights[i] = branchless_select_f32(enabled, weights[i], 0.0);
            i += 1;
        }
        let coupling = self.action_hub.coupling_matrix_8x8;
        let rep_thresh = self.config.repulsion_threshold;
        let rep_scale = self.config.repulsion_scale_u7;
        let qdps_min = self.config.raw.qdps_min_step_rad;
        let ripa_max = self.config.raw.ripa_clamp_max_rad;

        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min,
            ripa_max,
            Some(&*query_tensor),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, eff_ptr) = self
            .action_hub
            .daes_slot
            .resolve_or_pass_in_place(input_tagged_ptr);
        self.step_pointer(eff_ptr.raw(), trail_out)?;

        let bank_slice = self.action_hub.slot_phase_bank.relax_slice::<16>();
        let (damped_pull, coherence) = self.orchestrate_cross_slot_synergy_from_bank(
            &bank_slice,
            &gated_weights,
            secondary_slot,
            coupling,
            synergy_gain,
            inhibition_strength,
            trail_out,
        )?;

        let (evolved_ptr, density) = self.finish_pointer_cascade_tail(
            edbc_state.moving_z_entropy,
            dominant_slot_u8,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
            rep_thresh,
            rep_scale,
            qdps_min,
            ripa_max,
            head_dim,
            num_heads,
            bypass_hit,
            input_tagged_ptr.has_action_mode(ACTION_MODE_LAYER_SKIP),
            DAES_METRIC_SLOT_SYNERGY_ACTIVE,
            layer_skip_comp_profile,
            query_tensor,
            key_tensor,
            pspm_config,
            srl_activation_and_desc,
            trail_out,
        )?;
        Ok((evolved_ptr, density, damped_pull, coherence))
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn finish_pointer_cascade_tail(
        &mut self,
        moving_z_entropy: f32,
        dominant_slot_u8: u8,
        low_entropy_skip_thresh: f32,
        high_density_srl_thresh: f32,
        rep_thresh: u8,
        rep_scale: u8,
        qdps_min: f32,
        ripa_max: f32,
        head_dim: usize,
        num_heads: usize,
        bypass_hit: bool,
        force_skip_flag: bool,
        base_metric_bits: u8,
        layer_skip_comp_profile: Option<&AlignedSlice16<'_, gcso_q7_t>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        pspm_config: Option<&gcso_pspm_config_t>,
        srl_activation_and_desc: Option<(&AlignedSlice32<'_, f32>, &gcso_srl_descriptor_t)>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> Result<(gcso_tagged_ptr_t, u8)> {
        let (density, repulsion_active) = self
            .action_hub
            .bloom_filter
            .apply_phase_conjugate_repulsion_words(
                trail_phase_words_mut(trail_out),
                rep_thresh,
                rep_scale,
            );
        let evolved_ptr = trail_out.evolve_trail_mode_stigmergically(
            moving_z_entropy,
            repulsion_active,
            dominant_slot_u8,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );

        let mut trail_w = *trail_phase_words_ref(trail_out);
        let skip_active = evolved_ptr.has_action_mode(ACTION_MODE_LAYER_SKIP) | force_skip_flag;
        let mut skip_compensated = false;
        if let Some(comp) = layer_skip_comp_profile {
            if skip_active {
                if comp.len() < num_heads {
                    return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
                }
                accumulate_q7_slice_into_words_64(&mut trail_w, &comp.as_slice()[..num_heads]);
                *trail_phase_words_mut(trail_out) = trail_w;
                trail_out.flags |= TRAIL_FLAG_INERTIAL_SKIP_ACTIVE;
                skip_compensated = true;
            }
        }

        let steered = self.steer_qdps_words_prevalidated_core(
            &trail_w,
            head_dim,
            num_heads,
            qdps_min,
            ripa_max,
            Some(&mut *query_tensor),
            key_tensor,
        );
        let extra_metrics = self.apply_optional_pspm_and_srl(
            query_tensor,
            head_dim,
            pspm_config,
            srl_activation_and_desc,
        )?;
        let metric = base_metric_bits
            | branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(skip_compensated, DAES_METRIC_LAYER_SKIP_COMPENSATED, 0)
            | extra_metrics;
        self.action_hub.daes_slot.push_telemetry_metric(metric);

        Ok((evolved_ptr, density))
    }

    /// Executes a sequence of token steps in zero-allocation mode (for prefill and batch decoding).
    ///
    /// # Errors
    /// Propagates any error returned by [`GcsoRuntimeContext::step_token`].
    #[inline]
    pub fn step_sequence(
        &mut self,
        token_ids: &[u32],
        mut trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> Result<()> {
        for &token_id in token_ids {
            let trail_ref = trail_out.as_deref_mut();
            self.step_token(token_id, None, None, trail_ref)?;
        }
        Ok(())
    }

    /// Evaluates unified multi-domain superposition across both `self.attractor_space` and
    /// `self.slot_phase_bank` directly in 8 register-resident `u64` SWAR words (`0` intermediate
    /// 64-byte stack buffers or store-reload round-trips).
    #[inline]
    #[must_use]
    pub(crate) fn evaluate_multi_domain_superposition_words(
        &self,
        trail: &gcso_pointer_trail_t,
    ) -> ([u64; crate::traits::NUM_SWARM_MICRO_CELLS], f32) {
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
                        if base_e > 1.0e-12 {
                            // SAFETY: `self.action_hub.slot_phase_bank` is `AlignedBuffer64<gcso_q7_t, 512>`, so every
                            // 64-byte row `bank_2d[idx]` is 64-byte aligned (`>= 8-byte` alignment of `u64`).
                            let slot_w = unsafe {
                                &*(bank_2d[idx].as_ptr()
                                    as *const [u64; crate::traits::NUM_SWARM_MICRO_CELLS])
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
                                if total_w > 1.0e-12 && !swar_words_equal_64(&run_w, slot_w) {
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
                        as *const [u64; crate::traits::NUM_SWARM_MICRO_CELLS])
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
    /// GCSO ABI v0.1.1 (`0x00000101`), stamping an IEEE 802.3 CRC32 checksum over the payload.
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
        let daes_size = core::mem::size_of::<gcso_daes_slot_t>(); // 64 bytes

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
            .daes_slot
            .write_unaligned_to(buffer, daes_off)?;

        let mut header = init_snapshot_header(total_required as u64, 1_774_900_000);
        header.action_hub_offset = action_hub_off as u64;
        header.attractor_field_offset = action_hub_off as u64;
        header.dpsr_state_offset = edbc_off as u64;
        header.srl_state_offset = edbc_off as u64;
        header.edbc_state_offset = edbc_off as u64;
        header.daes_slot_offset = daes_off as u32;
        finalize_snapshot_header_crc32(&mut header, &buffer[..total_required])?;

        header.write_unaligned_to(buffer, 0)?;
        *required_size = total_required;
        Ok(())
    }

    /// Deserializes a `.gcso` binary snapshot buffer to restore context state in-place with
    /// ABI v0.1.1 header validation and IEEE 802.3 CRC32 payload integrity verification.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED`] if the header magic, section
    /// offsets, buffer bounds, or payload CRC32 checksum are invalid.
    /// Returns [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`] if the snapshot ABI version does not
    /// match `0x00000101`.
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
        let daes_off = header.daes_slot_offset as usize;

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

        let daes_size = core::mem::size_of::<gcso_daes_slot_t>();
        if buffer.len() >= daes_off + daes_size {
            self.action_hub.daes_slot = gcso_daes_slot_t::read_unaligned_from(buffer, daes_off)?;
        }

        Ok(())
    }
}

// ===================================================================
// Trait Implementations for Core Architecture (`crate::traits`) Compliance
// ===================================================================

impl SystemCapabilityQuery for RuntimeContext {
    #[inline(always)]
    fn abi_version(&self) -> (u32, u32, u32) {
        crate::abi_version()
    }

    #[inline(always)]
    fn abi_version_string(&self) -> &'static str {
        crate::abi::GCSO_ABI_VERSION_STRING
    }

    #[inline(always)]
    fn query_capability(&self) -> gcso_capability_flags_t {
        GCSO_CAP_CPU_BASELINE
            | ((cfg!(feature = "cuda") as u64) * GCSO_CAP_CUDA_BACKEND)
            | ((cfg!(feature = "vulkan") as u64) * GCSO_CAP_VULKAN_BACKEND)
            | (((cfg!(target_os = "macos") || cfg!(feature = "metal")) as u64)
                * GCSO_CAP_METAL_BACKEND)
    }

    #[inline(always)]
    fn init_default_config(&self, config: &mut gcso_config_t) -> Result<()> {
        *config = self.config.raw;
        Ok(())
    }
}

impl GcsoRuntimeContext for RuntimeContext {
    #[inline]
    fn reset(&mut self) -> Result<()> {
        self.edbc.reset();
        self.action_hub.reset()?;
        self.clear_all_domains()?;
        self.action_hub.bloom_filter.clear();
        self.action_hub.daes_slot = gcso_daes_slot_t {
            mode: u32::from(self.config.raw.daes_mode),
            ..Default::default()
        };
        Ok(())
    }

    #[inline]
    fn set_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> Result<()> {
        self.attractor_space
            .system_field_mut()
            .add_system_prompt_anchor(prompt, weight)?;
        self.action_hub.slot_weights[usize::from(ATTRACTOR_SLOT_SYSTEM_BASELINE)] =
            branchless_relu_f32(weight);
        Ok(())
    }

    #[inline]
    fn step_token(
        &mut self,
        token_id: u32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> Result<()> {
        self.step_token_with_zscore(token_id, query_tensor, key_tensor, None, 0.0, trail_out)
    }

    #[inline]
    fn serialize(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        self.serialize_to_bytes(buffer, required_size)
    }

    #[inline]
    fn deserialize(&mut self, buffer: &[u8]) -> Result<()> {
        self.deserialize_from_bytes(buffer)
    }

    #[inline]
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> Result<()> {
        header_out.token_index = token_index;
        header_out.frame_type = 0;
        Ok(())
    }
}

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

impl DaesScratchpad for RuntimeContext {
    #[inline]
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_daes_slot_t,
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
        config_out.daes_mode = (slot.mode & 0xFF) as u8;
        let _ = DAES_MINI_LEDGER_LEN;
        Ok(())
    }
}

impl SwarmCellChunk for RuntimeContext {
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
            sw[0] = swar_q7_add_sat_x8(sw[0], pw[0]);
            sw[1] = swar_q7_add_sat_x8(sw[1], pw[1]);
            sw[2] = swar_q7_add_sat_x8(sw[2], pw[2]);
            sw[3] = swar_q7_add_sat_x8(sw[3], pw[3]);
            sw[4] = swar_q7_add_sat_x8(sw[4], pw[4]);
            sw[5] = swar_q7_add_sat_x8(sw[5], pw[5]);
            sw[6] = swar_q7_add_sat_x8(sw[6], pw[6]);
            sw[7] = swar_q7_add_sat_x8(sw[7], pw[7]);
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

impl PprcCache for RuntimeContext {
    #[inline]
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> Result<()> {
        header_out.token_index = token_index;
        header_out.frame_type = 0;
        Ok(())
    }
}

impl EntropyGatedPhaseLookup for RuntimeContext {}
impl DynamicLayerSkipper for RuntimeContext {}
impl VisualAttentionSteering for RuntimeContext {}
impl PspmRouter for RuntimeContext {}
impl SrlAdapter for RuntimeContext {}
impl SheafCohomologyEvaluator for RuntimeContext {}
impl HierarchicalTreeTable for RuntimeContext {}
impl L2pSvdProjector for RuntimeContext {}
impl DnpPacketDispatcher for RuntimeContext {}

impl ZimmsStorage for RuntimeContext {
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
        descriptor_out: &mut gcso_zimms_descriptor_t,
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
    fn close_mmap(&mut self, descriptor: &mut gcso_zimms_descriptor_t) -> Result<()> {
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
    ) -> (GatewayFrame, [u64; crate::traits::NUM_SWARM_MICRO_CELLS]) {
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

// Compile-time verification that `RuntimeContext` satisfies `ActionHub`, `HotPathObserver`,
// `ColdPathObserver`, and all Full-Stack & Coordinator trait bounds.
const _: () = {
    const fn assert_full_stack_bounds<
        T: ActionHub
            + ActionHubGatewayExt
            + GatewayPhaseSteeringExt
            + PhaseSteeringClosedLoop
            + EntropyClosedLoopCoordinator
            + CausalProvenanceSignal
            + SystemConstraintSignal
            + ManifoldGatewaySignal
            + HotPathObserver
            + ColdPathObserver
            + FullStackGcsoEngine
            + FractalCascadeCoordinator
            + AttractorSlotSynergyCoordinator
            + StigmergicCohomologyCoordinator
            + BottomUpMacroCrystalCoordinator
            + EntropyGatedPhaseLookup
            + DynamicLayerSkipper
            + VisualAttentionSteering
            + HierarchicalTreeTable
            + DnpPacketDispatcher,
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

        assert!(GcsoRuntimeContext::reset(&mut ctx).is_ok());
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
    fn test_runtime_context_capabilities_and_zimms() {
        let config = Config::new();
        let mut ctx = RuntimeContext::new(config).unwrap();

        assert_eq!(ctx.abi_version(), (0, 1, 1));
        assert_eq!(ctx.abi_version_string(), "0.1.1");
        assert_ne!(ctx.query_capability() & GCSO_CAP_CPU_BASELINE, 0);

        let mut desc = gcso_zimms_descriptor_t::default();
        assert!(ctx.open_mmap("test.gcso", &mut desc).is_ok());
        assert_eq!(desc.fd_handle, 3);
        assert_eq!(
            ctx.get_track_offset(0).unwrap(),
            core::mem::size_of::<gcso_snapshot_header_t>() as u64
        );
        assert!(ctx.close_mmap(&mut desc).is_ok());
    }

    #[test]
    fn test_runtime_context_slot_bank_unified_field_and_autonomous_pointer_cascades() {
        let config = Config::new().with_head_dim(8).with_num_heads(2);
        let mut ctx = RuntimeContext::new(config).unwrap();

        ctx.apply_patch(&[32u8; 64]).unwrap();
        ctx.set_slot_phase_profile(ATTRACTOR_SLOT_KNOWLEDGE, &[32i8; 64])
            .unwrap();
        ctx.set_domain_weight(u32::from(ATTRACTOR_SLOT_PERSONA), 2.0)
            .unwrap();
        ctx.set_domain_weight(u32::from(ATTRACTOR_SLOT_KNOWLEDGE), 2.0)
            .unwrap();

        let mut q_buf = AlignedBuffer32::new([1.0f32; 16]);
        let comp_phases = AlignedBuffer16::new([4i8; 64]);
        let mut trail = gcso_pointer_trail_t {
            stigmergic_density: 2.0,
            ..Default::default()
        };

        let input_ptr = gcso_tagged_ptr_t::pack_action(
            0x0400,
            ATTRACTOR_SLOT_KNOWLEDGE,
            ACTION_PRESET_INERTIAL_SKIP,
            16,
        );

        let (evolved_ptr, density, pull, coh) = ctx
            .step_slot_bank_pointer_cascade(
                input_ptr,
                ATTRACTOR_SLOT_PERSONA,
                0.5,
                0.0,
                0.5,
                0.15,
                1.50,
                Some(&comp_phases.as_aligned_slice()),
                &mut q_buf.as_aligned_slice_mut(),
                None,
                None,
                None,
                &mut trail,
            )
            .unwrap();

        assert_eq!(density, 0);
        assert!((coh - 1.0).abs() < 1e-4);
        assert!(pull >= 6.0);
        assert_ne!(trail.flags & TRAIL_FLAG_SYNERGY_RESONANCE, 0);
        assert!(evolved_ptr.has_action_mode(ACTION_MODE_DPSR_STEER | ACTION_MODE_SRL_INJECT));

        // Verify Single-Manifold Unified-Field cascade & Autonomous Pointer-Trail Attention cascade
        let unified_anchors = [[0i8; MAX_ATTENTION_HEADS], [32i8; MAX_ATTENTION_HEADS]];
        let unified_energies = [0.0f32, 3.0f32];
        let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
        let mut uf_trail = gcso_pointer_trail_t::default();

        let (uf_rep, uf_pull, uf_coh, uf_dom) = ctx
            .step_unified_field_token_cascade(
                input_ptr,
                0.3,
                &unified_anchors,
                &unified_energies,
                None,
                2.0,
                Some(&mut attn_scores.as_aligned_slice_mut()),
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
                &mut uf_trail,
            )
            .unwrap();
        assert_eq!(uf_rep, 0);
        assert_eq!(uf_dom, 1);
        assert!(uf_pull > 5.0 && uf_coh > 0.90);
        assert!(attn_scores.data[1] > 0.5);

        let cand0 = gcso_pointer_trail_t {
            stigmergic_density: 4.0,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let cand1 = gcso_pointer_trail_t {
            stigmergic_density: 3.0,
            accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut candidates = [cand0, cand1];
        let mut bayes_scores = AlignedBuffer32::new([0.0f32; 8]);
        let mut auto_trail = gcso_pointer_trail_t::default();

        let (sel_id, bayes_score, log_p, dom_id, found) = ctx
            .step_autonomous_pointer_trail_attention_cascade(
                1,
                input_ptr,
                0.3,
                &mut candidates,
                2.0,
                0.5,
                0.25,
                1.0,
                64,
                0.25,
                &unified_anchors,
                &unified_energies,
                None,
                Some(&mut bayes_scores.as_aligned_slice_mut()),
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
                &mut auto_trail,
            )
            .unwrap();
        assert!(found);
        assert_eq!(sel_id, 1);
        assert_eq!(dom_id, 1);
        assert!(bayes_score > 0.5 && log_p > -0.05);
        assert!(bayes_scores.data[1] > 1.0 && bayes_scores.data[0] < -0.5);
    }
}
