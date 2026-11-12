// name: src/core/src/action_hub.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stigmergic Threshold Action-Selecting Orchestrator (`ActionHubEngine`) and Sidecar Pointer
//! Table (SPT) implementation for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! # Architectural Role ("The Thing Itself" — Hypothesis 2 $\times$ Hypothesis 1)
//! In `LiminiKa` (`GCSO`), intelligence is realized as $\mathcal{O}(1)$ stigmergic pointer-trail
//! manipulation (`gcso_pointer_trail_t`), pheromone evaporation/reinforcement ($\rho_{\mathrm{stigmergy}}$),
//! inter-memory **Limina** geodesic bridging, and **threshold-driven next-action selection**:
//!
//! 1. **Explicit Stigmergic Threshold Ladder ([`StigmergicThresholds`] & [`StigmergicActionDecision`])**:
//!    Evaluates local trail pheromone density $\rho_{\mathrm{stigmergy}}$, EDBC moving Z-score
//!    entropy $\tilde{H} \in [0, \ln 2]$, directional phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1, 1]$,
//!    and Contrastive Phase Bloom Filter collision density in **strictly branchless constant time
//!    ($C_{\mathrm{branch}} = 0$)** to select one of 7 next-action execution regimes:
//!    - [`StigmergicActionDecision::PassiveSidecar`] (`0`): Zero-interference observation when
//!      $\rho < \tau_{\mathrm{steer}}$ and $\tilde{H} > \tau_{\mathrm{skip}}$.
//!    - [`StigmergicActionDecision::InertialLayerSkip`] (`1`): Laminar low-entropy bypass
//!      ($\tilde{H} \le \tau_{\mathrm{skip}}$) or Game-Coexistence QoS budget throttle
//!      ([`ACTION_PRESET_INERTIAL_SKIP`]).
//!    - [`StigmergicActionDecision::ExpressPhaseSteer`] (`2`): Single-pass RIPA soft-clamped
//!      RoPE phase steering when $\tau_{\mathrm{steer}} \le \rho < \tau_{\mathrm{pspm}}$
//!      ([`ACTION_PRESET_EXPRESS_PHASE`]).
//!    - [`StigmergicActionDecision::PspmSubHeadRoute`] (`3`): $SO(2)$-fused RIPA + PSPM
//!      (`Fact / Logic / Explore`) Sub-Head Pseudo-MoE routing when $\tau_{\mathrm{pspm}} \le \rho < \tau_{\mathrm{srl}}$
//!      or during EDBC pitchfork bifurcation (`ACTION_MODE_DPSR_STEER | ACTION_MODE_PSPM_ROUTE`).
//!    - [`StigmergicActionDecision::FullOrchestrationSrl`] (`4`): Full $SO(2)$-fused RIPA + PSPM +
//!      Dynamic Rank-1 SRL residual injection when $\tau_{\mathrm{srl}} \le \rho < \tau_{\mathrm{crystal}}$
//!      ([`ACTION_PRESET_FULL_ORCHESTRATION`]).
//!    - [`StigmergicActionDecision::CrystallizeMacroBasin`] (`5`): Promotes a super-saturated
//!      coherent trail ($\rho \ge \tau_{\mathrm{crystal}}$ and $\mathcal{C}_{\mathrm{phase}} \ge \tau_{\mathrm{merge}}$)
//!      to a `.gcso` Section-1 [`CrystallizedAttractorRecord`] macro centroid via Online Toroidal
//!      $k$-Means / DP-Means (`TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`).
//!    - [`StigmergicActionDecision::RepulsionEscape`] (`6`): Immediate anti-phase escape
//!      ($-\boldsymbol{\Delta\theta}$) onto [`ATTRACTOR_SLOT_REPULSION`] when sheaf cohomology
//!      obstruction ($\mathcal{C}_{\mathrm{phase}} < -\tau_{\mathrm{contradict}}$) or Bloom
//!      collision ($\text{density} \ge \tau_{\mathrm{bloom}}$) is detected ([`ACTION_PRESET_REPULSION_ESCAPE`]).
//!
//! 2. **Zero-Allocation L1D-Resident Sidecar Pointer Table ([`ActionHubEngine`])**:
//!    Packs a 16-slot inline trail table (`[gcso_pointer_trail_t; 16]` = 2,048 bytes), an 8-lane
//!    working phase cache window (`AlignedBuffer64<gcso_q7_t, 512>`), a 64-byte [`gcso_daes_slot_t`]
//!    scratchpad, a 64-byte [`ContrastivePhaseBloomFilter`], [`StigmergicThresholds`], and a
//!    [`DpsrEngine`] into an exact **2,816-byte (`22 x 128B`), 128-byte cacheline-aligned [`Pod`]
//!    structure** ($C_{\mathrm{alloc}} = 0$).
//!
//! 3. **Blanket Trait Closure**:
//!    Because [`ActionHubEngine`] implements [`PhaseSteering`], [`PointerActionHub`], [`PspmRouter`],
//!    [`SrlAdapter`], [`DaesScratchpad`], [`SwarmCellChunk`], [`CausalProvenanceSignal`],
//!    [`crate::traits::SystemConstraintSignal`], and [`ManifoldGatewaySignal`], it automatically
//!    receives the blanket implementations of [`ActionHub`], [`HotPathExecutionEngine`],
//!    [`HotPathObserver`], [`FractalCascadeCoordinator`], [`ActionHubGatewayExt`],
//!    [`GatewayPhaseSteeringExt`], and [`crate::dpsr::DpsrClosedLoopSteering`].

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
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

use core::mem::{align_of, offset_of, size_of};

use crate::abi::{
    gcso_config_t, gcso_daes_slot_t, gcso_paged_bitmask_t, gcso_pointer_trail_t,
    gcso_pspm_config_t, gcso_q7_t, gcso_tagged_ptr_t, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::dpsr::{DpsrEngine, DEFAULT_DPSR_HEAD_DIM, DEFAULT_DPSR_NUM_HEADS};
use crate::traits::aligned::{
    AlignedBuffer64, AlignedSlice16, AlignedSliceMut16, AlignedSliceMut32,
};
use crate::traits::branchless::{
    branchless_abs_f32, branchless_clamp_f32, branchless_fast_transition_log_prob,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_max_f32,
    branchless_min_f32, branchless_rad_to_qdps_step_u8, branchless_relu_f32, branchless_select_f32,
    branchless_select_u32, branchless_select_u64, branchless_select_u8, branchless_select_usize,
    coherence_from_word_arrays_64, load_q7_words_64, scale_f32_slice_unrolled8, swar_q7_add_sat_x8,
    swar_q7_qdps_filter_words_64, swar_words_nonzero_64, trail_phase_words_mut,
    trail_phase_words_ref, SWAR_SPLAT_BYTE_64,
};
use crate::traits::gateway::{
    CausalProvenanceSignal, CrystallizedAttractorRecord, GatewayFrame, ManifoldGatewaySignal,
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_AXIS_SYSTEM_CONSTRAINT, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
    DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE, ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_NOMINAL,
    ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};
use crate::traits::hot_path::{PointerActionHub, SwarmCellChunk};
use crate::traits::primitives::{
    pack_action_slot_tag, unpack_attractor_slot_from_tag, GcsoResult, Pod, ACTION_MODE_DPSR_STEER,
    ACTION_MODE_PSPM_ROUTE, ACTION_MODE_SRL_INJECT, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_REPULSION,
    DAES_MINI_LEDGER_LEN, DAES_SHORTCUT_SLOTS, DEFAULT_ACTION_HUB_CAPACITY,
    DEFAULT_RIPA_CLAMP_MAX_RAD, DEFAULT_SLOT_COUPLING_MATRIX_8X8, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS,
};
use crate::traits::rope::{
    dispatch_fused_ripa_and_pspm_words_core, resolve_autonomous_swarm_rot_dim, ripa_upper_rot_dim,
    steer_from_swar_words_core_safe, DynamicLayerSkipper, EntropyGatedPhaseLookup, PhaseSteering,
    PspmRouter, SrlAdapter, VisualAttentionSteering,
};
use crate::traits::telemetry::{
    select_rewire_bridge_and_modulate_with_preloaded_words_64, ContrastivePhaseBloomFilter,
    DaesScratchpad, StigmergicActionPtrOps, StigmergicTrailOps, DAES_METRIC_FAST_PATH_HIT,
    DAES_METRIC_LAYER_SKIP_COMPENSATED, DAES_METRIC_PHASE_STEERED, DAES_METRIC_PSPM_ROUTED,
    DAES_METRIC_REPULSION_APPLIED, DAES_METRIC_SLOT_SYNERGY_ACTIVE, DAES_METRIC_SRL_INJECTED,
    MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY, NUM_SWARM_MICRO_CELLS,
    TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_DPSR_ACTIVE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
use crate::traits::{
    ActionHub, ActionHubGatewayExt, FractalCascadeCoordinator, GatewayPhaseSteeringExt,
    HotPathExecutionEngine, HotPathObserver,
};

/// Number of inline 128-byte [`gcso_pointer_trail_t`] cells stored inside [`ActionHubEngine`] (`16` = 2,048 bytes).
pub const ACTION_HUB_INLINE_TRAILS: usize = 16;

/// Default per-step pheromone retention factor ($\lambda_{\mathrm{retain}} = 0.95$).
pub const DEFAULT_STIGMERGIC_EVAPORATION_RETAIN: f32 = 0.95;

/// Default pheromone reinforcement gain ($g_{\mathrm{reinforce}} = 0.35$) deposited on coherent transitions.
pub const DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN: f32 = 0.35;

/// Default low-entropy threshold ($\tau_{\mathrm{skip}} = 0.15\text{ nats}$) triggering [`StigmergicActionDecision::InertialLayerSkip`].
pub const DEFAULT_STIGMERGIC_INERTIAL_SKIP_ENTROPY: f32 = 0.15;

/// Default stigmergic trail density threshold ($\tau_{\mathrm{steer}} = 0.35$) triggering [`StigmergicActionDecision::ExpressPhaseSteer`].
pub const DEFAULT_STIGMERGIC_EXPRESS_STEER_THRESH: f32 = 0.35;

/// Default stigmergic trail density threshold ($\tau_{\mathrm{pspm}} = 0.85$) triggering [`StigmergicActionDecision::PspmSubHeadRoute`].
pub const DEFAULT_STIGMERGIC_PSPM_ROUTE_THRESH: f32 = 0.85;

/// Default stigmergic trail density threshold ($\tau_{\mathrm{srl}} = 1.50$) triggering [`StigmergicActionDecision::FullOrchestrationSrl`].
pub const DEFAULT_STIGMERGIC_SRL_INJECT_THRESH: f32 = 1.50;

/// Default stigmergic trail density threshold ($\tau_{\mathrm{crystal}} = 2.20$) triggering [`StigmergicActionDecision::CrystallizeMacroBasin`].
pub const DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH: f32 = 2.20;

/// Default directional contradiction threshold ($\tau_{\mathrm{contradict}} = 0.25$) where $\mathcal{C}_{\mathrm{phase}} < -\tau_{\mathrm{contradict}}$ triggers [`StigmergicActionDecision::RepulsionEscape`].
pub const DEFAULT_STIGMERGIC_CONTRADICTION_THRESH: f32 = 0.25;

/// Total Q7 phase elements across all 8 Attractor Working Lanes (`8 * 64 = 512` bytes).
const HUB_SLOT_BANK_LEN: usize = MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS;

/// Default initial weights for the 8 parallel Attractor Working Lanes (`0..=7`).
const DEFAULT_HUB_SLOT_WEIGHTS: [f32; MAX_ATTRACTOR_DOMAINS] =
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5];

// ===================================================================
// 1. Stigmergic Action Decision Enum & Threshold Ladder (`Pod`)
// ===================================================================

/// Discrete next-action execution regime selected branchlessly by [`StigmergicThresholds`]
/// inside [`ActionHubEngine`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum StigmergicActionDecision {
    /// Zero-interference observation (`ACTION_PRESET_PASSIVE_SIDECAR = 0x00`):
    /// trail density $\rho < \tau_{\mathrm{steer}}$ and entropy $\tilde{H} > \tau_{\mathrm{skip}}$.
    #[default]
    PassiveSidecar = 0,
    /// Low-entropy laminar bypass or QoS throttle (`ACTION_PRESET_INERTIAL_SKIP = 0x03`):
    /// $\tilde{H} \le \tau_{\mathrm{skip}}$ with no repulsion active.
    InertialLayerSkip = 1,
    /// Single-pass RIPA soft-clamped RoPE phase steering (`ACTION_PRESET_EXPRESS_PHASE = 0x01`):
    /// $\tau_{\mathrm{steer}} \le \rho < \tau_{\mathrm{pspm}}$.
    ExpressPhaseSteer = 2,
    /// $SO(2)$-fused RIPA + PSPM (`Fact / Logic / Explore`) Sub-Head Pseudo-MoE routing (`0x09`):
    /// $\tau_{\mathrm{pspm}} \le \rho < \tau_{\mathrm{srl}}$ or high bifurcation entropy.
    PspmSubHeadRoute = 3,
    /// Full $SO(2)$-fused RIPA + PSPM + Dynamic Rank-1 SRL residual injection
    /// (`ACTION_PRESET_FULL_ORCHESTRATION = 0x19`): $\tau_{\mathrm{srl}} \le \rho < \tau_{\mathrm{crystal}}$.
    FullOrchestrationSrl = 4,
    /// Super-saturated coherent basin promotion (`ACTION_PRESET_FULL_ORCHESTRATION` +
    /// `TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`): $\rho \ge \tau_{\mathrm{crystal}}$ and
    /// $\mathcal{C}_{\mathrm{phase}} \ge \tau_{\mathrm{merge}}$.
    CrystallizeMacroBasin = 5,
    /// Phase-conjugate anti-phase repulsion escape (`ACTION_PRESET_REPULSION_ESCAPE = 0x0D` on
    /// `ATTRACTOR_SLOT_REPULSION`): triggered when $\mathcal{C}_{\mathrm{phase}} < -\tau_{\mathrm{contradict}}$
    /// or Contrastive Phase Bloom Filter density $\ge \tau_{\mathrm{bloom}}$.
    RepulsionEscape = 6,
}

impl StigmergicActionDecision {
    /// Converts a raw `u8` discriminant (`0..=6`) into a [`StigmergicActionDecision`].
    #[inline(always)]
    #[must_use]
    pub const fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::PassiveSidecar,
            1 => Self::InertialLayerSkip,
            2 => Self::ExpressPhaseSteer,
            3 => Self::PspmSubHeadRoute,
            4 => Self::FullOrchestrationSrl,
            5 => Self::CrystallizeMacroBasin,
            _ => Self::RepulsionEscape,
        }
    }

    /// Returns the 5-bit `ACTION_MODE_*` bitmask corresponding to this decision.
    #[inline(always)]
    #[must_use]
    pub const fn action_mode_flags(self) -> u8 {
        match self {
            Self::PassiveSidecar => ACTION_PRESET_PASSIVE_SIDECAR,
            Self::InertialLayerSkip => ACTION_PRESET_INERTIAL_SKIP,
            Self::ExpressPhaseSteer => ACTION_PRESET_EXPRESS_PHASE,
            Self::PspmSubHeadRoute => ACTION_MODE_DPSR_STEER | ACTION_MODE_PSPM_ROUTE,
            Self::FullOrchestrationSrl | Self::CrystallizeMacroBasin => {
                ACTION_PRESET_FULL_ORCHESTRATION
            }
            Self::RepulsionEscape => ACTION_PRESET_REPULSION_ESCAPE,
        }
    }
}

/// 32-byte (`#[repr(C, align(32))]`) [`Pod`] threshold configuration governing autonomous
/// stigmergic pheromone evaporation, reinforcement, and 7-regime next-action selection.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StigmergicThresholds {
    /// Per-step pheromone retention multiplier $\lambda_{\mathrm{retain}} \in [0.0, 1.0]$ (default `0.95`).
    pub evaporation_retain: f32,
    /// Base pheromone reinforcement deposit $g_{\mathrm{reinforce}} \ge 0.0$ on coherent hops (default `0.35`).
    pub reinforcement_gain: f32,
    /// Low-entropy threshold $\tau_{\mathrm{skip}}$ in nats triggering [`StigmergicActionDecision::InertialLayerSkip`] (default `0.15`).
    pub inertial_skip_entropy_thresh: f32,
    /// Minimum stigmergic density $\tau_{\mathrm{steer}}$ triggering [`StigmergicActionDecision::ExpressPhaseSteer`] (default `0.35`).
    pub express_steer_density_thresh: f32,
    /// Minimum stigmergic density $\tau_{\mathrm{pspm}}$ triggering [`StigmergicActionDecision::PspmSubHeadRoute`] (default `0.85`).
    pub pspm_route_density_thresh: f32,
    /// Minimum stigmergic density $\tau_{\mathrm{srl}}$ triggering [`StigmergicActionDecision::FullOrchestrationSrl`] (default `1.50`).
    pub srl_inject_density_thresh: f32,
    /// Minimum stigmergic density $\tau_{\mathrm{crystal}}$ triggering [`StigmergicActionDecision::CrystallizeMacroBasin`] (default `2.20`).
    pub crystallize_density_thresh: f32,
    /// Contradiction coherence threshold $\tau_{\mathrm{contradict}} \in [0.0, 1.0]$ where $\mathcal{C}_{\mathrm{phase}} < -\tau_{\mathrm{contradict}}$ triggers [`StigmergicActionDecision::RepulsionEscape`] (default `0.25`).
    pub contradiction_coherence_thresh: f32,
}

// SAFETY: `StigmergicThresholds` is a 32-byte `#[repr(C, align(32))]` struct of 8 `f32` fields with zero padding.
unsafe impl Pod for StigmergicThresholds {}

impl Default for StigmergicThresholds {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl StigmergicThresholds {
    /// Creates the canonical GCSO v0.1.1 stigmergic threshold ladder.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            evaporation_retain: DEFAULT_STIGMERGIC_EVAPORATION_RETAIN,
            reinforcement_gain: DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN,
            inertial_skip_entropy_thresh: DEFAULT_STIGMERGIC_INERTIAL_SKIP_ENTROPY,
            express_steer_density_thresh: DEFAULT_STIGMERGIC_EXPRESS_STEER_THRESH,
            pspm_route_density_thresh: DEFAULT_STIGMERGIC_PSPM_ROUTE_THRESH,
            srl_inject_density_thresh: DEFAULT_STIGMERGIC_SRL_INJECT_THRESH,
            crystallize_density_thresh: DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH,
            contradiction_coherence_thresh: DEFAULT_STIGMERGIC_CONTRADICTION_THRESH,
        }
    }

    /// Sets the per-step pheromone retention factor $\lambda_{\mathrm{retain}} \in [0.0, 1.0]$.
    #[inline(always)]
    #[must_use]
    pub const fn with_evaporation_retain(mut self, retain: f32) -> Self {
        self.evaporation_retain = branchless_clamp_f32(branchless_relu_f32(retain), 0.0, 1.0);
        self
    }

    /// Sets the pheromone reinforcement deposit gain $g_{\mathrm{reinforce}} \ge 0.0$.
    #[inline(always)]
    #[must_use]
    pub const fn with_reinforcement_gain(mut self, gain: f32) -> Self {
        self.reinforcement_gain = branchless_relu_f32(gain);
        self
    }

    /// Sets the low-entropy inertial layer-skip threshold $\tau_{\mathrm{skip}}$.
    #[inline(always)]
    #[must_use]
    pub const fn with_inertial_skip_entropy_thresh(mut self, thresh: f32) -> Self {
        self.inertial_skip_entropy_thresh = branchless_relu_f32(thresh);
        self
    }

    /// Sets the express RIPA phase steering density threshold $\tau_{\mathrm{steer}}$.
    #[inline(always)]
    #[must_use]
    pub const fn with_express_steer_density_thresh(mut self, thresh: f32) -> Self {
        self.express_steer_density_thresh = branchless_relu_f32(thresh);
        self
    }

    /// Sets the PSPM Sub-Head Pseudo-MoE routing density threshold $\tau_{\mathrm{pspm}}$.
    #[inline(always)]
    #[must_use]
    pub const fn with_pspm_route_density_thresh(mut self, thresh: f32) -> Self {
        self.pspm_route_density_thresh = branchless_relu_f32(thresh);
        self
    }

    /// Sets the full orchestration + SRL injection density threshold $\tau_{\mathrm{srl}}$.
    #[inline(always)]
    #[must_use]
    pub const fn with_srl_inject_density_thresh(mut self, thresh: f32) -> Self {
        self.srl_inject_density_thresh = branchless_relu_f32(thresh);
        self
    }

    /// Sets the macro basin crystallization density threshold $\tau_{\mathrm{crystal}}$.
    #[inline(always)]
    #[must_use]
    pub const fn with_crystallize_density_thresh(mut self, thresh: f32) -> Self {
        self.crystallize_density_thresh = branchless_relu_f32(thresh);
        self
    }

    /// Sets the directional contradiction coherence threshold $\tau_{\mathrm{contradict}} \in [0.0, 1.0]$.
    #[inline(always)]
    #[must_use]
    pub const fn with_contradiction_coherence_thresh(mut self, thresh: f32) -> Self {
        self.contradiction_coherence_thresh =
            branchless_clamp_f32(branchless_abs_f32(thresh), 0.0, 1.0);
        self
    }

    /// Validates that all threshold parameters are finite and ordered cleanly.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any threshold is non-finite or negative.
    #[inline(always)]
    pub const fn validate(&self) -> GcsoResult<()> {
        let valid = branchless_is_finite_f32(self.evaporation_retain)
            & (self.evaporation_retain >= 0.0)
            & (self.evaporation_retain <= 1.0)
            & branchless_is_finite_f32(self.reinforcement_gain)
            & (self.reinforcement_gain >= 0.0)
            & branchless_is_finite_f32(self.inertial_skip_entropy_thresh)
            & (self.inertial_skip_entropy_thresh >= 0.0)
            & branchless_is_finite_f32(self.express_steer_density_thresh)
            & (self.express_steer_density_thresh >= 0.0)
            & branchless_is_finite_f32(self.pspm_route_density_thresh)
            & (self.pspm_route_density_thresh >= self.express_steer_density_thresh)
            & branchless_is_finite_f32(self.srl_inject_density_thresh)
            & (self.srl_inject_density_thresh >= self.pspm_route_density_thresh)
            & branchless_is_finite_f32(self.crystallize_density_thresh)
            & (self.crystallize_density_thresh >= self.srl_inject_density_thresh)
            & branchless_is_finite_f32(self.contradiction_coherence_thresh)
            & (self.contradiction_coherence_thresh >= 0.0)
            & (self.contradiction_coherence_thresh <= 1.0);
        if valid {
            Ok(())
        } else {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        }
    }

    /// Evaluates the 7-regime stigmergic threshold ladder in **strictly branchless constant time
    /// (`0` data-dependent branches)**, returning
    /// `(decision, action_mode_flags, target_slot, extra_trail_flags)`.
    #[inline(always)]
    #[must_use]
    pub const fn select_action_decision(
        &self,
        stigmergic_density: f32,
        entropy_h_tilde: f32,
        phase_coherence: f32,
        repulsion_triggered: bool,
        preferred_slot: u8,
    ) -> (StigmergicActionDecision, u8, u8, u32) {
        let rho = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_density));
        let h_tilde = branchless_select_f32(
            branchless_is_finite_f32(entropy_h_tilde),
            branchless_relu_f32(entropy_h_tilde),
            0.0,
        );
        let coh = branchless_select_f32(
            branchless_is_finite_f32(phase_coherence),
            branchless_clamp_f32(phase_coherence, -1.0, 1.0),
            0.0,
        );
        let contradict_thresh = branchless_clamp_f32(
            branchless_abs_f32(self.contradiction_coherence_thresh),
            0.0,
            1.0,
        );
        let is_repulsed = repulsion_triggered | (coh < -contradict_thresh);

        let is_low_entropy_skip = (!is_repulsed) & (h_tilde <= self.inertial_skip_entropy_thresh);
        let ge_steer = rho >= self.express_steer_density_thresh;
        let ge_pspm = (rho >= self.pspm_route_density_thresh) | (h_tilde >= 0.55);
        let ge_srl = rho >= self.srl_inject_density_thresh;
        let ge_crystal = (rho >= self.crystallize_density_thresh)
            & (coh >= DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE);

        // Build decision discriminant bottom-up along the density ladder (`0..=6`):
        let mut disc = 0u8;
        disc = branchless_select_u8(
            ge_steer,
            StigmergicActionDecision::ExpressPhaseSteer as u8,
            disc,
        );
        disc = branchless_select_u8(
            ge_pspm,
            StigmergicActionDecision::PspmSubHeadRoute as u8,
            disc,
        );
        disc = branchless_select_u8(
            ge_srl,
            StigmergicActionDecision::FullOrchestrationSrl as u8,
            disc,
        );
        disc = branchless_select_u8(
            ge_crystal,
            StigmergicActionDecision::CrystallizeMacroBasin as u8,
            disc,
        );
        // Low-entropy laminar regime overrides active steering unless super-saturated crystallization is ready:
        disc = branchless_select_u8(
            is_low_entropy_skip & (!ge_crystal),
            StigmergicActionDecision::InertialLayerSkip as u8,
            disc,
        );
        // Repulsion escape unconditionally overrides all other regimes:
        disc = branchless_select_u8(
            is_repulsed,
            StigmergicActionDecision::RepulsionEscape as u8,
            disc,
        );

        let decision = StigmergicActionDecision::from_u8(disc);
        let mode_flags = decision.action_mode_flags();
        let clean_pref_slot = preferred_slot & ATTRACTOR_SLOT_ID_MASK;
        let target_slot =
            branchless_select_u8(is_repulsed, ATTRACTOR_SLOT_REPULSION, clean_pref_slot);

        let extra_flags = branchless_select_u32(
            (mode_flags & ACTION_MODE_DPSR_STEER) != 0,
            TRAIL_FLAG_DPSR_ACTIVE,
            0,
        ) | branchless_select_u32(
            is_low_entropy_skip & (!ge_crystal),
            TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
            0,
        ) | branchless_select_u32(
            ge_crystal & (!is_repulsed),
            TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | TRAIL_FLAG_SYNERGY_RESONANCE,
            0,
        ) | branchless_select_u32(
            is_repulsed,
            TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
            0,
        );

        (decision, mode_flags, target_slot, extra_flags)
    }

    /// Evaluates the 7-regime stigmergic threshold ladder on `trail` and returns the 32-byte
    /// [`ActionOrchestrationReport`] directly from this 32-byte [`StigmergicThresholds`] descriptor
    /// (`0` bytes of [`ActionHubEngine`] stack allocation).
    #[inline(always)]
    #[must_use]
    pub const fn evaluate_report_on_trail(
        &self,
        trail: &gcso_pointer_trail_t,
        repulsion_threshold: u8,
        entropy_h_tilde: f32,
        phase_coherence: f32,
        transition_log_prob: f32,
        selected_trail_id: u32,
        bloom_repulsion_density: u8,
        dominant_slot: u8,
    ) -> ActionOrchestrationReport {
        let repulsion_triggered = ((bloom_repulsion_density >= repulsion_threshold)
            & (bloom_repulsion_density > 0))
            | ((trail.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0);
        let cur_ptr = gcso_tagged_ptr_t(trail.current_ptr);
        let embedded_slot = unpack_attractor_slot_from_tag(cur_ptr.unpack_tag());
        let pref_slot = branchless_select_u8(
            dominant_slot < (MAX_ATTRACTOR_DOMAINS as u8),
            dominant_slot,
            embedded_slot,
        );
        let (decision, mode_flags, target_slot, _extra_flags) = self.select_action_decision(
            trail.stigmergic_density,
            entropy_h_tilde,
            phase_coherence,
            repulsion_triggered,
            pref_slot,
        );
        let base_evolved = gcso_tagged_ptr_t::pack(
            cur_ptr.unpack_addr(),
            pack_action_slot_tag(target_slot, mode_flags),
            cur_ptr.unpack_phase(),
        );
        let evolved_ptr = branchless_select_u64(
            decision as u8 == StigmergicActionDecision::RepulsionEscape as u8,
            base_evolved.conjugate_phase().raw(),
            base_evolved.raw(),
        );

        ActionOrchestrationReport {
            evolved_ptr,
            updated_density: trail.stigmergic_density,
            phase_coherence,
            transition_log_prob,
            selected_trail_id,
            dominant_anchor_id: trail.target_anchor_id,
            decision: decision as u8,
            action_mode_flags: mode_flags,
            target_slot,
            bloom_repulsion_density,
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn execute_decision_steering_and_logit_scaling(
        decision: StigmergicActionDecision,
        steer_words: &[u64; NUM_SWARM_MICRO_CELLS],
        qdps_min_rad: f32,
        ripa_clamp: f32,
        head_dim: usize,
        num_heads: usize,
        best_coh: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(bool, bool, bool)> {
        let mut steered = false;
        let mut pspm_routed = false;
        let mut skip_compensated = false;
        let min_step_q7 = branchless_rad_to_qdps_step_u8(qdps_min_rad);

        match decision {
            StigmergicActionDecision::PassiveSidecar => {}
            StigmergicActionDecision::InertialLayerSkip => {
                skip_compensated = true;
            }
            StigmergicActionDecision::ExpressPhaseSteer
            | StigmergicActionDecision::RepulsionEscape => {
                let swar_w = if min_step_q7 <= 1 {
                    *steer_words
                } else {
                    swar_q7_qdps_filter_words_64(steer_words, min_step_q7)
                };
                if swar_words_nonzero_64(&swar_w) {
                    if let Some(q) = query_tensor {
                        steer_from_swar_words_core_safe(
                            &swar_w, 0xFF, 128, head_dim, num_heads, false, ripa_clamp, q,
                            key_tensor,
                        );
                        steered = true;
                    } else if let Some(k) = key_tensor {
                        steer_from_swar_words_core_safe(
                            &swar_w, 0xFF, 128, head_dim, num_heads, false, ripa_clamp, k, None,
                        );
                        steered = true;
                    }
                }
            }
            StigmergicActionDecision::PspmSubHeadRoute
            | StigmergicActionDecision::FullOrchestrationSrl
            | StigmergicActionDecision::CrystallizeMacroBasin => {
                let swar_w = if min_step_q7 <= 1 {
                    *steer_words
                } else {
                    swar_q7_qdps_filter_words_64(steer_words, min_step_q7)
                };
                if swar_words_nonzero_64(&swar_w) {
                    if let Some(q) = query_tensor {
                        let is_default_ripa = (ripa_clamp.to_bits()
                            == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
                            & (num_heads > 0)
                            & (num_heads <= MAX_ATTENTION_HEADS);
                        if is_default_ripa {
                            let rot_dim = ripa_upper_rot_dim(head_dim);
                            if let Some(k) = key_tensor {
                                crate::traits::rope::rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
                                    q.as_mut_slice(),
                                    k.as_mut_slice(),
                                    &swar_w,
                                    num_heads,
                                    head_dim,
                                    rot_dim,
                                );
                            } else {
                                crate::traits::rope::rotate_fused_self_ripa_and_pspm_default_words(
                                    q.as_mut_slice(),
                                    &swar_w,
                                    num_heads,
                                    head_dim,
                                    rot_dim,
                                    true,
                                );
                            }
                        } else {
                            let f_heads = (num_heads >> 1) as u16;
                            let l_heads = (num_heads >> 2) as u16;
                            let e_heads = (num_heads as u16).saturating_sub(f_heads + l_heads);
                            let pspm_cfg = gcso_pspm_config_t {
                                num_fact_heads: f_heads,
                                num_logic_heads: l_heads,
                                num_explore_heads: e_heads,
                                flags: 0,
                                fact_phase_gain: 1.0,
                                logic_phase_gain: 1.0,
                                explore_phase_gain: 1.15,
                                reserved: [0; 12],
                            };
                            dispatch_fused_ripa_and_pspm_words_core(
                                q, &swar_w, ripa_clamp, &pspm_cfg, &swar_w, &swar_w, &swar_w,
                                ripa_clamp, 0, head_dim, true,
                            )?;
                            if let Some(k) = key_tensor {
                                steer_from_swar_words_core_safe(
                                    &swar_w, 0xFF, 128, head_dim, num_heads, false, ripa_clamp, k,
                                    None,
                                );
                            }
                        }
                        steered = true;
                        pspm_routed = true;
                    }
                }
            }
        }

        if let Some(logits) = logits_opt {
            if !logits.is_empty() {
                let inv_temp = branchless_select_f32(
                    decision as u8 == StigmergicActionDecision::RepulsionEscape as u8,
                    0.80,
                    branchless_clamp_f32(1.0 + 0.15 * branchless_relu_f32(best_coh), 0.75, 1.35),
                );
                scale_f32_slice_unrolled8(logits.as_mut_slice(), inv_temp);
            }
        }
        Ok((steered, pspm_routed, skip_compensated))
    }

    /// Executes a stateless one-shot stigmergic orchestration step directly from `self` and `dpsr`
    /// without allocating a 2,816-byte [`ActionHubEngine`] on the stack (`0` stack table zero-init).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor`, `key_tensor`, or `logits_opt`
    /// alias or have invalid dimensions.
    #[inline]
    pub fn orchestrate_stateless_action_step(
        &self,
        dpsr: &DpsrEngine,
        input_ptr: gcso_tagged_ptr_t,
        entropy_h_tilde: f32,
        dominant_slot: u8,
        trail_inout: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<ActionOrchestrationReport> {
        let qdps_min_rad = dpsr.qdps_min_step_rad();
        let ripa_clamp = dpsr.ripa_clamp_max_rad();
        let head_dim = dpsr.head_dim() as usize;
        let num_heads = dpsr.num_heads() as usize;
        let expected_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if head_dim == 0
            || (head_dim & 1) != 0
            || num_heads == 0
            || num_heads > MAX_ATTENTION_HEADS
            || qdps_min_rad < 0.0
            || !branchless_is_finite_f32(qdps_min_rad)
            || ripa_clamp <= 0.0
            || !branchless_is_finite_f32(ripa_clamp)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref q_ref) = query_tensor {
            if q_ref.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            if let Some(ref k_ref) = key_tensor {
                if k_ref.len() < expected_len {
                    return Err(GCSO_ERROR_INVALID_ARGUMENT);
                }
                q_ref.ensure_disjoint_mut(k_ref)?;
            }
            if let Some(ref l_ref) = logits_opt {
                q_ref.ensure_disjoint_mut(l_ref)?;
            }
        } else if let Some(ref k_ref) = key_tensor {
            if k_ref.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            if let Some(ref l_ref) = logits_opt {
                k_ref.ensure_disjoint_mut(l_ref)?;
            }
        }

        trail_inout.record_pointer_transition(input_ptr.raw(), 1, 0.0);
        let embedded_phase = input_ptr.unpack_phase();
        if embedded_phase != 0 {
            trail_inout.accumulate_uniform_phase(embedded_phase);
        }

        let pref_slot = branchless_select_u8(
            dominant_slot < (MAX_ATTRACTOR_DOMAINS as u8),
            dominant_slot,
            input_ptr.target_attractor_slot(),
        ) & ATTRACTOR_SLOT_ID_MASK;

        let steer_words = *trail_phase_words_ref(trail_inout);
        let best_coh = 1.0_f32;
        let log_prob = 0.0_f32;
        let selected_id = trail_inout.linked_trail_id;

        let retain = branchless_clamp_f32(branchless_relu_f32(self.evaporation_retain), 0.0, 1.0);
        let reinf_gain = branchless_relu_f32(self.reinforcement_gain);
        let base_rho = branchless_flush_subnormal_f32(
            branchless_relu_f32(trail_inout.stigmergic_density) * retain,
        );
        let updated_rho = branchless_min_f32(
            branchless_flush_subnormal_f32(base_rho + reinf_gain * 2.0),
            MAX_STIGMERGIC_DENSITY,
        );
        trail_inout.stigmergic_density = updated_rho;

        let (decision, mode_flags, target_slot, extra_flags) =
            self.select_action_decision(updated_rho, entropy_h_tilde, best_coh, false, pref_slot);

        let base_evolved = gcso_tagged_ptr_t::pack(
            input_ptr.unpack_addr(),
            pack_action_slot_tag(target_slot, mode_flags),
            input_ptr.unpack_phase(),
        );
        let evolved_ptr = gcso_tagged_ptr_t(branchless_select_u64(
            decision as u8 == StigmergicActionDecision::RepulsionEscape as u8,
            base_evolved.conjugate_phase().raw(),
            base_evolved.raw(),
        ));

        trail_inout.current_ptr = evolved_ptr.raw();
        trail_inout.cluster_id = branchless_select_u32(
            trail_inout.target_anchor_id != 0,
            trail_inout.target_anchor_id,
            u32::from(target_slot),
        );
        trail_inout.flags = (trail_inout.flags
            & !(TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | extra_flags;
        trail_inout.sync_flags_with_current_ptr();

        let _ = Self::execute_decision_steering_and_logit_scaling(
            decision,
            &steer_words,
            qdps_min_rad,
            ripa_clamp,
            head_dim,
            num_heads,
            best_coh,
            query_tensor,
            key_tensor,
            logits_opt,
        )?;

        Ok(ActionOrchestrationReport {
            evolved_ptr: evolved_ptr.raw(),
            updated_density: updated_rho,
            phase_coherence: best_coh,
            transition_log_prob: log_prob,
            selected_trail_id: selected_id,
            dominant_anchor_id: trail_inout.target_anchor_id,
            decision: decision as u8,
            action_mode_flags: mode_flags,
            target_slot,
            bloom_repulsion_density: 0,
        })
    }
}

/// 32-byte (`#[repr(C, align(32))]`) [`Pod`] report summarizing the outcome of a single
/// stigmergic threshold action-selection step in [`ActionHubEngine`].
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ActionOrchestrationReport {
    /// Evolved 64-bit tagged pointer (`gcso_tagged_ptr_t::raw()`) carrying the selected Action Mode and Target Slot.
    pub evolved_ptr: u64,
    /// Updated local trail pheromone density $\rho_{\mathrm{stigmergy}}$ after evaporation & reinforcement.
    pub updated_density: f32,
    /// Directional cosine phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1, 1]$ with the selected trail/centroid.
    pub phase_coherence: f32,
    /// Conditional transition log-probability $\ln P(M_t \mid M_{t-1}) \le 0$ under Hypothesis 1.
    pub transition_log_prob: f32,
    /// Winning candidate trail index $j^*$ selected in the Sidecar Pointer Table.
    pub selected_trail_id: u32,
    /// Dominant 32-bit macro attractor anchor or centroid ID (`0` if unbound).
    pub dominant_anchor_id: u32,
    /// Selected next-action regime discriminant ([`StigmergicActionDecision`] as `u8`).
    pub decision: u8,
    /// 5-bit `ACTION_MODE_*` execution flags embedded in `evolved_ptr`.
    pub action_mode_flags: u8,
    /// 3-bit Target Attractor Working Lane ID (`0..=7`) embedded in `evolved_ptr`.
    pub target_slot: u8,
    /// 4-bit Contrastive Phase Bloom Filter collision density (`0..=15`).
    pub bloom_repulsion_density: u8,
}

// SAFETY: `ActionOrchestrationReport` is a 32-byte `#[repr(C, align(32))]` struct of primitive fields with zero implicit padding.
unsafe impl Pod for ActionOrchestrationReport {}

impl ActionOrchestrationReport {
    /// Returns the typed [`StigmergicActionDecision`] enum for `self.decision`.
    #[inline(always)]
    #[must_use]
    pub const fn action_decision(&self) -> StigmergicActionDecision {
        StigmergicActionDecision::from_u8(self.decision)
    }

    /// Returns `self.evolved_ptr` wrapped as a [`gcso_tagged_ptr_t`].
    #[inline(always)]
    #[must_use]
    pub const fn tagged_ptr(&self) -> gcso_tagged_ptr_t {
        gcso_tagged_ptr_t(self.evolved_ptr)
    }
}

// ===================================================================
// 2. `ActionHubEngine` — 2,816-Byte L1D-Resident Sidecar Pointer Table
// ===================================================================

impl gcso_daes_slot_t {
    /// Executes strictly branchless $\mathcal{O}(1)$ fast-path shortcut lookup directly in-place
    /// on `self` without copying the 64-byte `gcso_daes_slot_t` onto the stack.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn lookup_fast_path_branchless(&self, input_key: u64) -> (bool, u64) {
        let mask_hit = ((self.fast_path_bypass_mask & input_key) != 0) & (self.mode == 0);
        let idx = ((input_key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
        let candidate = self.fast_path_shortcuts[idx & (DAES_SHORTCUT_SLOTS - 1)];
        let fallback = self.fast_path_shortcuts[0];
        let resolved = branchless_select_u64(candidate != 0, candidate, fallback);
        let valid_hit = mask_hit & (resolved != 0);
        (valid_hit, branchless_select_u64(valid_hit, resolved, 0))
    }

    /// Branchlessly resolves `input_ptr` against `self.fast_path_shortcuts` in-place (`0` stack
    /// copy, `0` 256-bit-to-64-bit Store-to-Load Forwarding pipeline stall).
    #[inline(always)]
    pub(crate) fn resolve_or_pass_in_place(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
    ) -> (bool, gcso_tagged_ptr_t) {
        let (hit, shortcut_raw) = self.lookup_fast_path_branchless(input_ptr.raw());
        self.cache_hit_count = self.cache_hit_count.wrapping_add(u32::from(hit));
        let eff_raw = branchless_select_u64(hit, shortcut_raw, input_ptr.raw());
        (hit, gcso_tagged_ptr_t(eff_raw))
    }

    /// Pushes a profiling metric byte code (`DAES_METRIC_*`) into `self.telemetry_mini_ledger` in-place.
    #[inline(always)]
    pub(crate) fn push_telemetry_metric(&mut self, metric_code: u8) {
        let head = (self.telemetry_ring_head as usize) & (DAES_MINI_LEDGER_LEN - 1);
        self.telemetry_mini_ledger[head] = metric_code;
        self.telemetry_ring_head = self.telemetry_ring_head.wrapping_add(1);
    }
}

/// Concrete zero-allocation (`#[repr(C, align(128))]`, 2,816-byte [`Pod`]) Sidecar Pointer Table
/// and **Stigmergic Threshold Action-Selecting Orchestrator** implementing [`ActionHub`] ("The Thing Itself").
///
/// # Memory Layout (2,816 Bytes = `22 x 128B` Cache-Line Pairs, Zero Implicit Padding)
/// - `0..2048` (`2048B`): `trails: [gcso_pointer_trail_t; 16]` (16 inline 128B pointer trails)
/// - `2048..2560` (`512B`): `slot_phase_bank: AlignedBuffer64<gcso_q7_t, 512>` (8 working lanes $\times$ 64 heads)
/// - `2560..2624` (`64B`): `daes_slot: gcso_daes_slot_t` (DAES fast-path shortcuts & telemetry ring)
/// - `2624..2688` (`64B`): `bloom_filter: ContrastivePhaseBloomFilter` (128-bucket 4-bit Bloom filter)
/// - `2688..2816` (`128B`): `thresholds` (`32B`) + `slot_weights` (`32B`) + `dpsr` (`16B`) +
///   control & telemetry registers (`48B`).
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionHubEngine {
    /// Inline L1D-resident Sidecar Pointer Table (`16 x 128B = 2,048` bytes).
    pub trails: [gcso_pointer_trail_t; ACTION_HUB_INLINE_TRAILS],
    /// 8-lane working phase cache window (`8 x 64B = 512` bytes, 64-byte aligned).
    pub slot_phase_bank: AlignedBuffer64<gcso_q7_t, HUB_SLOT_BANK_LEN>,
    /// Dynamic Adaptive Extension Scratchpad (64 bytes, 64-byte aligned).
    pub daes_slot: gcso_daes_slot_t,
    /// Contrastive Phase Bloom Filter for $\mathcal{O}(1)$ anti-phase repulsion (64 bytes, 64-byte aligned).
    pub bloom_filter: ContrastivePhaseBloomFilter,
    /// Stigmergic threshold regime governing pheromone dynamics and next-action selection (32 bytes).
    pub thresholds: StigmergicThresholds,
    /// Active gravitational weights for the 8 Attractor Working Lanes (`0..=7`, 32 bytes).
    pub slot_weights: [f32; MAX_ATTRACTOR_DOMAINS],
    /// Embedded 16-byte DPSR/RIPA kernel configuration (`head_dim`, `num_heads`, `ripa_clamp_max_rad`, `qdps_min_step_rad`).
    pub dpsr: DpsrEngine,
    /// 64-bit ($8 \times 8$) cross-lane coupling bitmatrix.
    pub coupling_matrix_8x8: u64,
    /// Total pointer transitions executed by this Action Hub.
    pub step_counter: u64,
    /// Configured logical trail slot capacity (`>= 1`).
    pub capacity: u32,
    /// Number of active trail slots currently populated (`0..=capacity`).
    pub active_count: u32,
    /// Circular write cursor (`0..ACTION_HUB_INLINE_TRAILS`) into `self.trails`.
    pub write_cursor: u32,
    /// Most recently selected candidate trail ID (`0..ACTION_HUB_INLINE_TRAILS`).
    pub last_selected_trail_id: u32,
    /// Most recently observed EDBC moving Z-score entropy $\tilde{H} \in [0, \ln 2]$.
    pub last_entropy_h_tilde: f32,
    /// Most recently observed causal anomaly severity $\in [0.0, 1.0]$.
    pub last_anomaly_severity: f32,
    /// Minimum Bloom filter counter density (`1..=15`) triggering anti-phase repulsion.
    pub repulsion_threshold: u8,
    /// Q7 unit scale (`0..=128`, where `128 == 1.0`) applied during phase-conjugate repulsion.
    pub repulsion_scale_u7: u8,
    /// 8-bit bitmask of enabled Attractor Working Lanes (`0xFF` = all 8 lanes enabled).
    pub slot_enabled_mask: u8,
    /// Most recently selected [`StigmergicActionDecision`] discriminant (`0..=6`).
    pub last_decision: u8,
    /// Explicit padding to guarantee exact 2,816-byte (`22 x 128B`) size and zero implicit padding.
    pub reserved_align128: [u8; 4],
}

// SAFETY: `ActionHubEngine` is a `#[repr(C, align(128))]` composition of `Pod` structs and primitive
// arrays totaling exactly 2,816 bytes with zero implicit padding.
unsafe impl Pod for ActionHubEngine {}

impl Default for ActionHubEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl ActionHubEngine {
    /// Creates a new default [`ActionHubEngine`] with `head_dim = 128`, `num_heads = 32`,
    /// `capacity = DEFAULT_ACTION_HUB_CAPACITY`, and canonical [`StigmergicThresholds`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        let dpsr = match DpsrEngine::new(DEFAULT_DPSR_HEAD_DIM, DEFAULT_DPSR_NUM_HEADS) {
            Ok(d) => d,
            Err(_) => unreachable!(),
        };
        Self {
            trails: [gcso_pointer_trail_t {
                current_ptr: 0,
                prev_ptr: 0,
                user_data: 0,
                transition_cost: 0,
                step_count: 0,
                stigmergic_density: 0.0,
                target_anchor_id: 0,
                cluster_id: 0,
                linked_trail_id: 0,
                attractor_pull_force: 0.0,
                flags: 0,
                accumulated_phase_delta: [0; MAX_ATTENTION_HEADS],
                reserved_padding: [0; 8],
            }; ACTION_HUB_INLINE_TRAILS],
            slot_phase_bank: AlignedBuffer64::new([0; HUB_SLOT_BANK_LEN]),
            daes_slot: gcso_daes_slot_t {
                mode: 0,
                telemetry_ring_head: 0,
                telemetry_ring_tail: 0,
                cache_hit_count: 0,
                auto_tune_flags: 0,
                fast_path_bypass_mask: 0,
                fast_path_shortcuts: [0; 4],
                telemetry_mini_ledger: [0; 8],
            },
            bloom_filter: ContrastivePhaseBloomFilter::new(),
            thresholds: StigmergicThresholds::new(),
            slot_weights: DEFAULT_HUB_SLOT_WEIGHTS,
            dpsr,
            coupling_matrix_8x8: DEFAULT_SLOT_COUPLING_MATRIX_8X8,
            step_counter: 0,
            capacity: DEFAULT_ACTION_HUB_CAPACITY,
            active_count: 0,
            write_cursor: 0,
            last_selected_trail_id: 0,
            last_entropy_h_tilde: 0.0,
            last_anomaly_severity: 0.0,
            repulsion_threshold: 2,
            repulsion_scale_u7: 128,
            slot_enabled_mask: 0xFF,
            last_decision: StigmergicActionDecision::PassiveSidecar as u8,
            reserved_align128: [0; 4],
        }
    }

    /// Creates a new [`ActionHubEngine`] configured with the specified logical `capacity`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `capacity == 0`.
    #[inline]
    pub const fn with_capacity(capacity: u32) -> GcsoResult<Self> {
        if capacity == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut hub = Self::new();
        hub.capacity = capacity;
        Ok(hub)
    }

    /// Creates a new [`ActionHubEngine`] configured with `(head_dim, num_heads, capacity)` and
    /// custom [`StigmergicThresholds`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `(head_dim, num_heads)` or `thresholds` are invalid,
    /// or if `capacity == 0`.
    #[inline]
    pub fn with_dimensions_and_thresholds(
        head_dim: u32,
        num_heads: u32,
        capacity: u32,
        thresholds: StigmergicThresholds,
    ) -> GcsoResult<Self> {
        if capacity == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        thresholds.validate()?;
        let dpsr = DpsrEngine::new(head_dim, num_heads)?;
        let mut hub = Self::new();
        hub.dpsr = dpsr;
        hub.capacity = capacity;
        hub.thresholds = thresholds;
        Ok(hub)
    }

    /// Creates a new [`ActionHubEngine`] initialized from a raw [`gcso_config_t`] descriptor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `config` dimensions or capacities are invalid.
    #[inline]
    pub fn from_config(config: &gcso_config_t) -> GcsoResult<Self> {
        if config.action_hub_capacity == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let dpsr = DpsrEngine::from_config(config)?;
        let mut hub = Self::new();
        hub.dpsr = dpsr;
        hub.capacity = config.action_hub_capacity;
        hub.daes_slot.mode = u32::from(config.daes_mode);
        Ok(hub)
    }

    /// Sets the active [`StigmergicThresholds`] ladder.
    #[inline(always)]
    #[must_use]
    pub const fn with_thresholds(mut self, thresholds: StigmergicThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Returns the number of populated inline trails in `self.trails` (`0..=ACTION_HUB_INLINE_TRAILS`).
    #[inline(always)]
    #[must_use]
    pub const fn inline_active_len(&self) -> usize {
        branchless_select_usize(
            (self.active_count as usize) < ACTION_HUB_INLINE_TRAILS,
            self.active_count as usize,
            ACTION_HUB_INLINE_TRAILS,
        )
    }

    /// Returns an immutable slice over the currently populated inline pointer trails.
    #[inline(always)]
    #[must_use]
    pub fn active_trails(&self) -> &[gcso_pointer_trail_t] {
        &self.trails[..self.inline_active_len()]
    }

    /// Returns a mutable slice over the currently populated inline pointer trails.
    #[inline(always)]
    #[must_use]
    pub fn active_trails_mut(&mut self) -> &mut [gcso_pointer_trail_t] {
        let len = self.inline_active_len();
        &mut self.trails[..len]
    }

    /// Returns an immutable reference to inline trail slot `trail_id & 15`.
    #[inline(always)]
    #[must_use]
    pub const fn trail_slot(&self, trail_id: u32) -> &gcso_pointer_trail_t {
        &self.trails[(trail_id as usize) & (ACTION_HUB_INLINE_TRAILS - 1)]
    }

    /// Returns a mutable reference to inline trail slot `trail_id & 15`.
    #[inline(always)]
    #[must_use]
    pub fn trail_slot_mut(&mut self, trail_id: u32) -> &mut gcso_pointer_trail_t {
        &mut self.trails[(trail_id as usize) & (ACTION_HUB_INLINE_TRAILS - 1)]
    }

    /// Returns an immutable 16-byte aligned slice view over the 8-lane Q7 working phase bank (`512` bytes).
    #[inline(always)]
    #[must_use]
    pub const fn slot_phase_bank(&self) -> AlignedSlice16<'_, gcso_q7_t> {
        self.slot_phase_bank.relax_slice::<16>()
    }

    /// Returns a mutable 16-byte aligned slice view over the 8-lane Q7 working phase bank (`512` bytes).
    #[inline(always)]
    #[must_use]
    pub fn slot_phase_bank_mut(&mut self) -> AlignedSliceMut16<'_, gcso_q7_t> {
        self.slot_phase_bank.relax_slice_mut::<16>()
    }

    /// Sets the 64-head Q7 phase profile for a specific Attractor Working Lane (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_id >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn set_slot_phase_profile(
        &mut self,
        slot_id: u8,
        phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<()> {
        let idx = usize::from(slot_id);
        if idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: `self.slot_phase_bank` is `AlignedBuffer64<gcso_q7_t, 512>`, so every 64-byte slot
        // at byte offset `idx * 64` is 64-byte aligned (`>= 8-byte` alignment of `u64`).
        let dst_w = unsafe {
            &mut *(self
                .slot_phase_bank
                .as_mut_slice()
                .as_mut_ptr()
                .add(idx * MAX_ATTENTION_HEADS)
                as *mut [u64; NUM_SWARM_MICRO_CELLS])
        };
        *dst_w = load_q7_words_64(phases);
        Ok(())
    }

    /// Evaluates the [`StigmergicThresholds`] ladder on `trail` and returns a 32-byte
    /// [`ActionOrchestrationReport`] without mutating `self`.
    #[inline(always)]
    #[must_use]
    pub const fn evaluate_stigmergic_thresholds(
        &self,
        trail: &gcso_pointer_trail_t,
        entropy_h_tilde: f32,
        phase_coherence: f32,
        transition_log_prob: f32,
        selected_trail_id: u32,
        bloom_repulsion_density: u8,
        dominant_slot: u8,
    ) -> ActionOrchestrationReport {
        self.thresholds.evaluate_report_on_trail(
            trail,
            self.repulsion_threshold,
            entropy_h_tilde,
            phase_coherence,
            transition_log_prob,
            selected_trail_id,
            bloom_repulsion_density,
            dominant_slot,
        )
    }

    /// Applies per-step pheromone evaporation ($\rho_j \gets \rho_j \cdot \lambda_{\mathrm{retain}}$)
    /// across all active inline trails in `self.trails`, deposits pheromone reinforcement
    /// $\Delta\rho = g_{\mathrm{reinforce}} \cdot (1 + \max(0, \mathcal{C}_{\mathrm{phase}}))$ onto
    /// `trail_inout` (or halves $\rho$ on contradiction), and synchronizes `trail_inout` into its
    /// inline table slot in $\mathcal{O}(1)$.
    ///
    /// Returns `(slot_idx, updated_density)`.
    #[inline]
    pub fn evaporate_and_deposit_pheromone(
        &mut self,
        trail_inout: &mut gcso_pointer_trail_t,
        phase_coherence: f32,
        contradiction_repulsed: bool,
    ) -> (u32, f32) {
        let retain = branchless_clamp_f32(
            branchless_relu_f32(self.thresholds.evaporation_retain),
            0.0,
            1.0,
        );
        let active_len = self.inline_active_len();
        let mut i = 0usize;
        while i < active_len {
            let decayed = branchless_flush_subnormal_f32(
                branchless_relu_f32(self.trails[i].stigmergic_density) * retain,
            );
            self.trails[i].stigmergic_density = decayed;
            i += 1;
        }

        let cur_rho = branchless_flush_subnormal_f32(
            branchless_relu_f32(trail_inout.stigmergic_density) * retain,
        );
        let pos_coh = branchless_clamp_f32(branchless_relu_f32(phase_coherence), 0.0, 1.0);
        let deposit = branchless_relu_f32(self.thresholds.reinforcement_gain) * (1.0 + pos_coh);
        let reinforced_rho = branchless_min_f32(cur_rho + deposit, MAX_STIGMERGIC_DENSITY);
        let damped_rho = branchless_flush_subnormal_f32(cur_rho * 0.5);
        let next_rho = branchless_select_f32(contradiction_repulsed, damped_rho, reinforced_rho);
        trail_inout.stigmergic_density = next_rho;

        let slot_idx = self.write_cursor & ((ACTION_HUB_INLINE_TRAILS as u32) - 1);
        Self::copy_trail_to_slot_fast(&mut self.trails[slot_idx as usize], trail_inout);
        self.write_cursor =
            self.write_cursor.wrapping_add(1) & ((ACTION_HUB_INLINE_TRAILS as u32) - 1);
        let cap = self.capacity;
        self.active_count = branchless_select_u32(
            self.active_count < cap,
            self.active_count.wrapping_add(1),
            self.active_count,
        );
        (slot_idx, next_rho)
    }

    /// Copies `src` and pre-loaded `phase_words` into `dst` using width-matched scalar writes and
    /// 8-byte aligned 64-bit GPR stores (`*trail_phase_words_mut(dst) = *phase_words`) at byte
    /// offset `56..120`, avoiding unaligned 256-bit `vmovdqu` cache-line splits, 64-bit-to-256-bit
    /// STLF stalls, and redundant 64-byte reloads from `src`.
    #[inline(always)]
    pub(crate) fn copy_trail_and_words_to_slot_fast(
        dst: &mut gcso_pointer_trail_t,
        src: &gcso_pointer_trail_t,
        phase_words: &[u64; NUM_SWARM_MICRO_CELLS],
    ) {
        dst.copy_from_with_words_fast(src, phase_words);
    }

    #[inline(always)]
    pub(crate) fn copy_trail_to_slot_fast(
        dst: &mut gcso_pointer_trail_t,
        src: &gcso_pointer_trail_t,
    ) {
        dst.copy_from_fast(src);
    }

    /// Clears a single inline trail slot in-place using width-matched 64-bit GPR stores (`0` YMM
    /// stores, `0` 256-bit-to-64-bit STLF stalls on subsequent trail reads/writes).
    #[inline(always)]
    pub(crate) fn clear_trail_slot_fast(dst: &mut gcso_pointer_trail_t) {
        dst.clear_fast();
    }

    /// Applies pheromone evaporation across active inline trails (skipping `slot_idx`, which is
    /// about to be overwritten, and skipping the loop entirely when `retain >= 1.0`) and updates
    /// `trail_inout.stigmergic_density` in registers, returning `(slot_idx, updated_density)`.
    #[inline(always)]
    fn evaporate_and_prepare_slot_without_copy(
        &mut self,
        trail_inout: &mut gcso_pointer_trail_t,
        phase_coherence: f32,
        contradiction_repulsed: bool,
    ) -> (u32, f32) {
        let retain = branchless_clamp_f32(
            branchless_relu_f32(self.thresholds.evaporation_retain),
            0.0,
            1.0,
        );
        let slot_idx = self.write_cursor & ((ACTION_HUB_INLINE_TRAILS as u32) - 1);
        let slot_usize = slot_idx as usize;
        let active_len = self.inline_active_len();
        if retain < 1.0 - 1.0e-7 {
            let mut i = 0usize;
            while i < active_len {
                if i != slot_usize {
                    let decayed = branchless_flush_subnormal_f32(
                        branchless_relu_f32(self.trails[i].stigmergic_density) * retain,
                    );
                    self.trails[i].stigmergic_density = decayed;
                }
                i += 1;
            }
        }

        let cur_rho = branchless_flush_subnormal_f32(
            branchless_relu_f32(trail_inout.stigmergic_density) * retain,
        );
        let pos_coh = branchless_clamp_f32(branchless_relu_f32(phase_coherence), 0.0, 1.0);
        let deposit = branchless_relu_f32(self.thresholds.reinforcement_gain) * (1.0 + pos_coh);
        let reinforced_rho = branchless_min_f32(cur_rho + deposit, MAX_STIGMERGIC_DENSITY);
        let damped_rho = branchless_flush_subnormal_f32(cur_rho * 0.5);
        let next_rho = branchless_select_f32(contradiction_repulsed, damped_rho, reinforced_rho);
        trail_inout.stigmergic_density = next_rho;

        self.write_cursor =
            self.write_cursor.wrapping_add(1) & ((ACTION_HUB_INLINE_TRAILS as u32) - 1);
        let cap = self.capacity;
        self.active_count = branchless_select_u32(
            self.active_count < cap,
            self.active_count.wrapping_add(1),
            self.active_count,
        );
        (slot_idx, next_rho)
    }

    /// Executes the primary **Stigmergic Threshold Action-Selecting Orchestration Step** in
    /// zero-allocation constant-time mode ($C_{\mathrm{alloc}} = 0$):
    /// 1. Resolves DAES fast-path shortcut (`resolve_or_pass_tagged_ptr`).
    /// 2. Advances `trail_inout` pointer & accumulates embedded Q7 phase.
    /// 3. Evaluates `self.bloom_filter` contrastive anti-phase repulsion (`apply_phase_conjugate_repulsion_words`).
    /// 4. Scans populated inline trails `self.trails[..active_len]` via
    ///    `select_rewire_bridge_and_modulate_with_preloaded_words_64` to select the highest-scoring
    ///    non-contradictory trail $j^*$ under the Bayes-Attention transition score, rewires
    ///    `trail_inout.linked_trail_id = j^*`, bridges the inter-memory **Limina** gap via $S^1$
    ///    geodesic SLERP, and optionally modulates `attention_scores` in-place.
    /// 5. Applies global pheromone evaporation & reinforcement (`evaporate_and_deposit_pheromone`).
    /// 6. Evaluates `self.thresholds.select_action_decision` to choose the next
    ///    [`StigmergicActionDecision`] (`PassiveSidecar`, `InertialLayerSkip`, `ExpressPhaseSteer`,
    ///    `PspmSubHeadRoute`, `FullOrchestrationSrl`, `CrystallizeMacroBasin`, or `RepulsionEscape`).
    /// 7. Executes the selected action on `query_tensor` / `key_tensor` / `logits_opt` and logs
    ///    telemetry into `self.daes_slot`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias or have
    /// fewer than `head_dim * num_heads` elements.
    #[inline]
    pub fn orchestrate_next_action_step(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        entropy_h_tilde: f32,
        dominant_slot: u8,
        trail_inout: &mut gcso_pointer_trail_t,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<ActionOrchestrationReport> {
        self.orchestrate_next_action_step_core(
            input_ptr,
            entropy_h_tilde,
            dominant_slot,
            trail_inout,
            attention_scores,
            query_tensor,
            key_tensor,
            logits_opt,
            true,
            None,
        )
    }

    #[inline(always)]
    fn orchestrate_next_action_step_core(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        entropy_h_tilde: f32,
        dominant_slot: u8,
        trail_inout: &mut gcso_pointer_trail_t,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        accumulate_ptr_phase: bool,
        macro_centroid_words: Option<&[u64; NUM_SWARM_MICRO_CELLS]>,
    ) -> GcsoResult<ActionOrchestrationReport> {
        let qdps_min_rad = self.dpsr.qdps_min_step_rad();
        let ripa_clamp = self.dpsr.ripa_clamp_max_rad();
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_rad,
            ripa_clamp,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        if let Some(ref l_ref) = logits_opt {
            if let Some(ref q_ref) = query_tensor {
                q_ref.ensure_disjoint_mut(l_ref)?;
            }
            if let Some(ref k_ref) = key_tensor {
                k_ref.ensure_disjoint_mut(l_ref)?;
            }
        }

        let (bypass_hit, eff_ptr) = self.daes_slot.resolve_or_pass_in_place(input_ptr);

        self.step_counter = self.step_counter.wrapping_add(1);
        trail_inout.record_pointer_transition(eff_ptr.raw(), 1, 0.0);
        if accumulate_ptr_phase {
            let embedded_phase = eff_ptr.unpack_phase();
            if embedded_phase != 0 {
                trail_inout.accumulate_uniform_phase(embedded_phase);
            }
        }

        let (bloom_density, bloom_repulsed) =
            self.bloom_filter.apply_phase_conjugate_repulsion_words(
                trail_phase_words_mut(trail_inout),
                self.repulsion_threshold,
                self.repulsion_scale_u7,
            );

        let pref_slot = branchless_select_u8(
            dominant_slot < (MAX_ATTRACTOR_DOMAINS as u8),
            dominant_slot,
            eff_ptr.target_attractor_slot(),
        ) & ATTRACTOR_SLOT_ID_MASK;

        let active_len = self.inline_active_len();
        let init_w = *trail_phase_words_ref(trail_inout);
        let (selected_id, best_coh, log_prob, found_coherent, mut steer_words) = if active_len > 0 {
            let scores_slice_opt = attention_scores.map(AlignedSliceMut32::as_mut_slice);
            let (sel_id, _bayes_score, _mean_coh, _coh_cnt, found, bridged_w, raw_best_coh) =
                select_rewire_bridge_and_modulate_with_preloaded_words_64(
                    trail_inout,
                    init_w,
                    &self.trails[..active_len],
                    scores_slice_opt,
                    head_dim as u32,
                    2.0,
                    0.5,
                    0.25,
                    self.thresholds.contradiction_coherence_thresh,
                    64,
                    1.0,
                    trail_inout.target_anchor_id,
                    macro_centroid_words,
                    0.25,
                );
            let lp = branchless_fast_transition_log_prob(raw_best_coh);
            (sel_id, raw_best_coh, lp, found, bridged_w)
        } else {
            // SAFETY: `self.slot_phase_bank` is 64B-aligned and `pref_slot < 8`.
            let pref_slot_w = unsafe {
                &*(self
                    .slot_phase_bank
                    .as_slice()
                    .as_ptr()
                    .add(usize::from(pref_slot) * MAX_ATTENTION_HEADS)
                    as *const [u64; NUM_SWARM_MICRO_CELLS])
            };
            let pref_enabled = ((self.slot_enabled_mask >> pref_slot) & 1) != 0;
            let slot_coh = if pref_enabled && swar_words_nonzero_64(pref_slot_w) {
                coherence_from_word_arrays_64(&init_w, pref_slot_w)
            } else {
                1.0_f32
            };
            let lp = branchless_fast_transition_log_prob(slot_coh);
            (
                trail_inout.linked_trail_id,
                slot_coh,
                lp,
                slot_coh >= -self.thresholds.contradiction_coherence_thresh,
                init_w,
            )
        };

        let contradiction_repulsed = bloom_repulsed
            | ((!found_coherent) & (best_coh < -self.thresholds.contradiction_coherence_thresh));
        if contradiction_repulsed && !bloom_repulsed {
            self.bloom_filter
                .insert_repulsive_pattern_words_weighted(&steer_words, 1);
        }

        let (stored_slot_idx, updated_rho) = self.evaporate_and_prepare_slot_without_copy(
            trail_inout,
            best_coh,
            contradiction_repulsed,
        );
        let (decision, mode_flags, target_slot, extra_flags) =
            self.thresholds.select_action_decision(
                updated_rho,
                entropy_h_tilde,
                best_coh,
                contradiction_repulsed,
                pref_slot,
            );

        let base_evolved = gcso_tagged_ptr_t::pack(
            eff_ptr.unpack_addr(),
            pack_action_slot_tag(target_slot, mode_flags),
            eff_ptr.unpack_phase(),
        );
        let evolved_ptr = gcso_tagged_ptr_t(branchless_select_u64(
            decision as u8 == StigmergicActionDecision::RepulsionEscape as u8,
            base_evolved.conjugate_phase().raw(),
            base_evolved.raw(),
        ));

        trail_inout.current_ptr = evolved_ptr.raw();
        trail_inout.cluster_id = branchless_select_u32(
            (trail_inout.target_anchor_id != 0) & (!contradiction_repulsed),
            trail_inout.target_anchor_id,
            u32::from(target_slot),
        );
        trail_inout.flags = (trail_inout.flags
            & !(TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | extra_flags;
        trail_inout.sync_flags_with_current_ptr();

        // Fold coupled working-lane phase profile from `slot_phase_bank[target_slot]` if enabled and non-zero:
        let masked_target_slot = target_slot & ATTRACTOR_SLOT_ID_MASK;
        if ((self.slot_enabled_mask >> masked_target_slot) & 1) != 0 {
            // SAFETY: `self.slot_phase_bank` is 64B-aligned and `masked_target_slot < 8`.
            let slot_w = unsafe {
                &*(self
                    .slot_phase_bank
                    .as_slice()
                    .as_ptr()
                    .add(usize::from(masked_target_slot) * MAX_ATTENTION_HEADS)
                    as *const [u64; NUM_SWARM_MICRO_CELLS])
            };
            if swar_words_nonzero_64(slot_w) {
                steer_words[0] = swar_q7_add_sat_x8(steer_words[0], slot_w[0]);
                steer_words[1] = swar_q7_add_sat_x8(steer_words[1], slot_w[1]);
                steer_words[2] = swar_q7_add_sat_x8(steer_words[2], slot_w[2]);
                steer_words[3] = swar_q7_add_sat_x8(steer_words[3], slot_w[3]);
                steer_words[4] = swar_q7_add_sat_x8(steer_words[4], slot_w[4]);
                steer_words[5] = swar_q7_add_sat_x8(steer_words[5], slot_w[5]);
                steer_words[6] = swar_q7_add_sat_x8(steer_words[6], slot_w[6]);
                steer_words[7] = swar_q7_add_sat_x8(steer_words[7], slot_w[7]);
                *trail_phase_words_mut(trail_inout) = steer_words;
            }
        }
        Self::copy_trail_and_words_to_slot_fast(
            &mut self.trails[stored_slot_idx as usize],
            trail_inout,
            &steer_words,
        );

        let (steered, pspm_routed, skip_compensated) =
            StigmergicThresholds::execute_decision_steering_and_logit_scaling(
                decision,
                &steer_words,
                qdps_min_rad,
                ripa_clamp,
                head_dim,
                num_heads,
                best_coh,
                query_tensor,
                key_tensor,
                logits_opt,
            )?;

        let srl_flag = (mode_flags & ACTION_MODE_SRL_INJECT) != 0;
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(pspm_routed, DAES_METRIC_PSPM_ROUTED, 0)
            | branchless_select_u8(srl_flag, DAES_METRIC_SRL_INJECTED, 0)
            | branchless_select_u8(skip_compensated, DAES_METRIC_LAYER_SKIP_COMPENSATED, 0)
            | branchless_select_u8(contradiction_repulsed, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(found_coherent, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        self.daes_slot.push_telemetry_metric(metric);

        self.last_selected_trail_id = selected_id;
        self.last_entropy_h_tilde = branchless_select_f32(
            branchless_is_finite_f32(entropy_h_tilde),
            branchless_relu_f32(entropy_h_tilde),
            0.0,
        );
        self.last_anomaly_severity = branchless_select_f32(
            contradiction_repulsed,
            0.85,
            branchless_clamp_f32(0.5 * (1.0 - best_coh), 0.0, 1.0),
        );
        self.last_decision = decision as u8;

        Ok(ActionOrchestrationReport {
            evolved_ptr: evolved_ptr.raw(),
            updated_density: updated_rho,
            phase_coherence: best_coh,
            transition_log_prob: log_prob,
            selected_trail_id: selected_id,
            dominant_anchor_id: trail_inout.target_anchor_id,
            decision: decision as u8,
            action_mode_flags: mode_flags,
            target_slot,
            bloom_repulsion_density: bloom_density,
        })
    }

    /// Executes a complete closed-loop **2-Origin Composite Manifold Gateway $\to$ Stigmergic
    /// Threshold Action Selection $\to$ Online Toroidal $k$-Means / DP-Means Macro Clustering**
    /// step in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// Returns `(GatewayFrame, ActionOrchestrationReport, assigned_centroid_idx, assigned_anchor_id, assigned_coherence, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `centroids` is empty, `split_coherence_thresh`
    /// is non-finite, or tensor slices are invalid.
    #[inline]
    pub fn orchestrate_gateway_kmeans_and_select_action<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        entropy_h_tilde: f32,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        trail_inout: &mut gcso_pointer_trail_t,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        GatewayFrame,
        ActionOrchestrationReport,
        usize,
        u32,
        f32,
        bool,
    )> {
        if centroids.is_empty() || !branchless_is_finite_f32(split_coherence_thresh) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        if swar_words_nonzero_64(&frm_w) {
            *trail_phase_words_mut(trail_inout) = frm_w;
        }
        if frame.target_anchor_id != 0 {
            trail_inout.target_anchor_id = frame.target_anchor_id;
        }
        trail_inout.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_inout.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let (assigned_idx, assigned_anchor_id, assigned_coh, _updated_mass, spawned_new) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                &frame,
                &frm_w,
                split_coherence_thresh,
                Some(trail_inout),
            );

        let win_w = *centroids[assigned_idx].phase_words_ref();
        let report = self.orchestrate_next_action_step_core(
            gcso_tagged_ptr_t(frame.tagged_ptr),
            entropy_h_tilde,
            frame.dominant_slot,
            trail_inout,
            attention_scores,
            query_tensor,
            key_tensor,
            logits_opt,
            false,
            Some(&win_w),
        )?;

        Ok((
            frame,
            report,
            assigned_idx,
            assigned_anchor_id,
            assigned_coh,
            spawned_new,
        ))
    }

    /// Resolves the **Swarm- & Blanket-Driven Autonomous Frequency Aperture** (`rot_dim`, even
    /// integer in `2..=head_dim`) for this engine's `head_dim` and last observed entropy $\tilde{H}$.
    #[inline(always)]
    #[must_use]
    pub fn resolve_autonomous_rot_dim(
        &self,
        provenance_bits: u32,
        quorum_mask: u8,
        von_mises_kappa: f32,
    ) -> usize {
        resolve_autonomous_swarm_rot_dim(
            self.dpsr.head_dim() as usize,
            provenance_bits,
            quorum_mask,
            self.last_entropy_h_tilde,
            von_mises_kappa,
        )
    }
}

// ===================================================================
// 3. Trait Implementations for `ActionHubEngine` ("The Thing Itself")
// ===================================================================

impl PhaseSteering for ActionHubEngine {
    #[inline(always)]
    fn head_dim(&self) -> u32 {
        self.dpsr.head_dim()
    }

    #[inline(always)]
    fn num_heads(&self) -> u32 {
        self.dpsr.num_heads()
    }
}

impl EntropyGatedPhaseLookup for ActionHubEngine {}
impl DynamicLayerSkipper for ActionHubEngine {}
impl VisualAttentionSteering for ActionHubEngine {}
impl PspmRouter for ActionHubEngine {}
impl SrlAdapter for ActionHubEngine {}

impl DaesScratchpad for ActionHubEngine {
    #[inline]
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_daes_slot_t,
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()> {
        let word = u64::from_le_bytes(slot.telemetry_mini_ledger);
        let spike_count = (word & 0x8080_8080_8080_8080).count_ones();
        let ripa_max = self.dpsr.ripa_clamp_max_rad();
        let base_ripa = branchless_select_f32(ripa_max > 0.0, ripa_max, DEFAULT_RIPA_CLAMP_MAX_RAD);
        let damping = 1.0 - 0.025 * (spike_count as f32);
        config_out.head_dim = self.dpsr.head_dim();
        config_out.num_heads = self.dpsr.num_heads();
        config_out.ripa_clamp_max_rad = branchless_clamp_f32(base_ripa * damping, 0.01, 0.25);
        config_out.qdps_min_step_rad = self.dpsr.qdps_min_step_rad();
        config_out.action_hub_capacity = self.capacity;
        config_out.daes_mode = (slot.mode & 0xFF) as u8;
        let _ = DAES_MINI_LEDGER_LEN;
        Ok(())
    }
}

impl PointerActionHub for ActionHubEngine {
    #[inline]
    fn reset(&mut self) -> GcsoResult<()> {
        let active_len = self.inline_active_len();
        let mut i = 0usize;
        while i < active_len {
            Self::clear_trail_slot_fast(&mut self.trails[i]);
            i += 1;
        }
        self.active_count = 0;
        self.write_cursor = 0;
        self.step_counter = 0;
        self.last_selected_trail_id = 0;
        self.last_entropy_h_tilde = 0.0;
        self.last_anomaly_severity = 0.0;
        self.last_decision = StigmergicActionDecision::PassiveSidecar as u8;
        Ok(())
    }

    #[inline]
    fn step_pointer(
        &mut self,
        current_ptr: u64,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<()> {
        let tp = gcso_tagged_ptr_t(current_ptr);
        self.step_counter = self.step_counter.wrapping_add(1);
        trail_out.record_pointer_transition(current_ptr, 1, self.thresholds.reinforcement_gain);
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

        let slot_idx = (self.write_cursor as usize) & (ACTION_HUB_INLINE_TRAILS - 1);
        Self::copy_trail_to_slot_fast(&mut self.trails[slot_idx], trail_out);
        self.write_cursor =
            self.write_cursor.wrapping_add(1) & ((ACTION_HUB_INLINE_TRAILS as u32) - 1);
        let cap = self.capacity;
        self.active_count = branchless_select_u32(
            self.active_count < cap,
            self.active_count.wrapping_add(1),
            self.active_count,
        );
        Ok(())
    }

    #[inline]
    fn update_cellular_state_express(
        &mut self,
        trail_id: u32,
        phase_delta: gcso_q7_t,
    ) -> GcsoResult<()> {
        if trail_id >= self.capacity {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let inline_idx = (trail_id as usize) & (ACTION_HUB_INLINE_TRAILS - 1);
        self.trails[inline_idx].accumulate_uniform_phase(phase_delta);
        let slot_idx = (trail_id as usize) & (MAX_ATTRACTOR_DOMAINS - 1);
        if phase_delta != 0 {
            let splat = ((phase_delta as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
            // SAFETY: `self.slot_phase_bank` is 64B-aligned and `slot_idx < 8`.
            let w = unsafe {
                &mut *(self
                    .slot_phase_bank
                    .as_mut_slice()
                    .as_mut_ptr()
                    .add(slot_idx * MAX_ATTENTION_HEADS)
                    as *mut [u64; NUM_SWARM_MICRO_CELLS])
            };
            w[0] = swar_q7_add_sat_x8(w[0], splat);
            w[1] = swar_q7_add_sat_x8(w[1], splat);
            w[2] = swar_q7_add_sat_x8(w[2], splat);
            w[3] = swar_q7_add_sat_x8(w[3], splat);
            w[4] = swar_q7_add_sat_x8(w[4], splat);
            w[5] = swar_q7_add_sat_x8(w[5], splat);
            w[6] = swar_q7_add_sat_x8(w[6], splat);
            w[7] = swar_q7_add_sat_x8(w[7], splat);
        }
        Ok(())
    }

    #[inline]
    fn pull_trail_to_attractor(
        &mut self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> GcsoResult<()> {
        if trail_id >= self.capacity || !branchless_is_finite_f32(pull_force) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let clean_pull =
            branchless_min_f32(branchless_relu_f32(pull_force), MAX_ATTRACTOR_PULL_FORCE);
        let inline_idx = (trail_id as usize) & (ACTION_HUB_INLINE_TRAILS - 1);
        let trail = &mut self.trails[inline_idx];
        trail.target_anchor_id = anchor_id;
        trail.cluster_id = branchless_select_u32(anchor_id != 0, anchor_id, trail.cluster_id);
        trail.attractor_pull_force = clean_pull;
        let slot_idx = (anchor_id as usize) & (MAX_ATTRACTOR_DOMAINS - 1);
        self.slot_weights[slot_idx] = clean_pull;
        Ok(())
    }

    #[inline]
    fn link_hallucinated_trails(&mut self, src_trail_id: u32, dst_trail_id: u32) -> GcsoResult<()> {
        if src_trail_id >= self.capacity || dst_trail_id >= self.capacity {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let src_idx = (src_trail_id as usize) & (ACTION_HUB_INLINE_TRAILS - 1);
        self.trails[src_idx].linked_trail_id = dst_trail_id;
        self.last_selected_trail_id = dst_trail_id;
        Ok(())
    }

    #[inline(always)]
    fn capacity(&self) -> u32 {
        self.capacity
    }

    #[inline(always)]
    fn active_count(&self) -> u32 {
        self.active_count
    }
}

impl SwarmCellChunk for ActionHubEngine {
    #[inline]
    fn step_chunk(&mut self, _mask: &gcso_paged_bitmask_t, chunk_len: u32) -> GcsoResult<()> {
        if chunk_len == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.step_counter = self.step_counter.wrapping_add(u64::from(chunk_len));
        Ok(())
    }
}

impl CausalProvenanceSignal for ActionHubEngine {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let dec = StigmergicActionDecision::from_u8(self.last_decision);
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | branchless_select_u32(
                dec == StigmergicActionDecision::InertialLayerSkip,
                CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
                0,
            )
            | branchless_select_u32(
                dec == StigmergicActionDecision::CrystallizeMacroBasin,
                CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED | CAUSAL_AXIS_SUBJECTIVE_PERSONA,
                0,
            )
            | branchless_select_u32(
                dec == StigmergicActionDecision::RepulsionEscape,
                CAUSAL_AXIS_CONTRADICTION_GUARD | CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
                CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA,
            )
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.last_anomaly_severity
    }
}

impl crate::traits::SystemConstraintSignal for ActionHubEngine {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        self.slot_enabled_mask
    }

    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        self.last_decision == (StigmergicActionDecision::InertialLayerSkip as u8)
    }

    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 {
        self.dpsr.qdps_min_step_rad()
    }
}

impl ManifoldGatewaySignal for ActionHubEngine {
    #[inline]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let idx = (self.last_selected_trail_id as usize) & (ACTION_HUB_INLINE_TRAILS - 1);
        let active_trail = &self.trails[idx];
        let words = *trail_phase_words_ref(active_trail);
        let dec = StigmergicActionDecision::from_u8(self.last_decision);
        let mode_flags = dec.action_mode_flags();
        let cur_tp = gcso_tagged_ptr_t(active_trail.current_ptr);
        let dom_slot = branchless_select_u8(
            dec == StigmergicActionDecision::RepulsionEscape,
            ATTRACTOR_SLOT_REPULSION,
            cur_tp.target_attractor_slot(),
        );
        let cause_code = branchless_select_u8(
            dec == StigmergicActionDecision::RepulsionEscape,
            ROOT_CAUSE_COHOMOLOGY_REPULSION,
            branchless_select_u8(
                dec == StigmergicActionDecision::InertialLayerSkip,
                ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
                ROOT_CAUSE_NOMINAL,
            ),
        );
        let qdps_min_rad = self.dpsr.qdps_min_step_rad();
        let num_heads_u8 = (self.dpsr.num_heads().min(MAX_ATTENTION_HEADS as u32)) as u8;
        let fact_h = num_heads_u8 >> 1;
        let logic_h = num_heads_u8 >> 2;
        let explore_h = num_heads_u8.saturating_sub(fact_h + logic_h);
        let mut frame = GatewayFrame {
            tagged_ptr: active_trail.current_ptr,
            zimms_section_offset48: cur_tp.unpack_addr(),
            target_anchor_id: active_trail.target_anchor_id,
            effective_energy: branchless_max_f32(
                active_trail.stigmergic_density,
                active_trail.attractor_pull_force,
            ),
            ripa_clamp_rad: self.dpsr.ripa_clamp_max_rad(),
            qdps_min_step_rad: qdps_min_rad,
            effective_inv_temp: 1.0,
            fact_persona_coherence: 1.0 - 2.0 * self.last_anomaly_severity,
            repulsion_severity: self.last_anomaly_severity,
            provenance_bits: self.causal_provenance_bits(),
            dominant_cause_code: cause_code,
            dominant_slot: dom_slot,
            action_mode_flags: mode_flags,
            active_swarm_cell_mask: self.slot_enabled_mask,
            qdps_min_step_q7: branchless_rad_to_qdps_step_u8(qdps_min_rad),
            pspm_fact_heads: fact_h,
            pspm_logic_heads: logic_h,
            pspm_explore_heads: explore_h,
            pspm_fact_gain_q7: 128,
            pspm_logic_gain_q7: 128,
            pspm_explore_gain_q7: 128,
            mean_phase_tag_q7: cur_tp.unpack_phase(),
            active_head_count: u16::from(num_heads_u8),
            frame_flags: u16::from(dec == StigmergicActionDecision::RepulsionEscape),
            phase_deltas: [0; MAX_ATTENTION_HEADS],
        };
        *frame.phase_words_mut() = words;
        (frame, words)
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

// ===================================================================
// 4. Compile-Time Layout & Trait-Bound Verification
// ===================================================================

const _: () = {
    assert!(size_of::<StigmergicActionDecision>() == 1);
    assert!(size_of::<StigmergicThresholds>() == 32 && align_of::<StigmergicThresholds>() == 32);
    assert!(
        size_of::<ActionOrchestrationReport>() == 32
            && align_of::<ActionOrchestrationReport>() == 32
    );
    assert!(offset_of!(ActionOrchestrationReport, evolved_ptr) == 0);
    assert!(offset_of!(ActionOrchestrationReport, updated_density) == 8);
    assert!(offset_of!(ActionOrchestrationReport, phase_coherence) == 12);
    assert!(offset_of!(ActionOrchestrationReport, transition_log_prob) == 16);
    assert!(offset_of!(ActionOrchestrationReport, selected_trail_id) == 20);
    assert!(offset_of!(ActionOrchestrationReport, dominant_anchor_id) == 24);
    assert!(offset_of!(ActionOrchestrationReport, decision) == 28);
    assert!(offset_of!(ActionOrchestrationReport, action_mode_flags) == 29);
    assert!(offset_of!(ActionOrchestrationReport, target_slot) == 30);
    assert!(offset_of!(ActionOrchestrationReport, bloom_repulsion_density) == 31);

    assert!(size_of::<ActionHubEngine>() == 2816 && align_of::<ActionHubEngine>() == 128);
    assert!(offset_of!(ActionHubEngine, trails) == 0);
    assert!(offset_of!(ActionHubEngine, slot_phase_bank) == 2048);
    assert!(offset_of!(ActionHubEngine, daes_slot) == 2560);
    assert!(offset_of!(ActionHubEngine, bloom_filter) == 2624);
    assert!(offset_of!(ActionHubEngine, thresholds) == 2688);

    const fn assert_action_hub_bounds<
        T: ActionHub
            + HotPathExecutionEngine
            + HotPathObserver
            + FractalCascadeCoordinator
            + ActionHubGatewayExt
            + GatewayPhaseSteeringExt
            + crate::dpsr::DpsrClosedLoopSteering
            + CausalProvenanceSignal
            + crate::traits::SystemConstraintSignal
            + ManifoldGatewaySignal
            + Pod,
    >() {
    }
    assert_action_hub_bounds::<ActionHubEngine>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::aligned::AlignedBuffer32;
    use crate::traits::gateway::{ObjectiveFact, SubjectivePersona, SystemConstraint};
    use crate::traits::primitives::{ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA};

    #[test]
    fn test_stigmergic_thresholds_ladder_all_regimes() {
        let thresh = StigmergicThresholds::new();
        assert!(thresh.validate().is_ok());

        // 0. PassiveSidecar: low density (0.10 < 0.35), normal entropy (0.30 > 0.15)
        let (dec0, flags0, slot0, _) =
            thresh.select_action_decision(0.10, 0.30, 0.80, false, ATTRACTOR_SLOT_PERSONA);
        assert_eq!(dec0, StigmergicActionDecision::PassiveSidecar);
        assert_eq!(flags0, ACTION_PRESET_PASSIVE_SIDECAR);
        assert_eq!(slot0, ATTRACTOR_SLOT_PERSONA);

        // 1. InertialLayerSkip: low entropy (0.08 <= 0.15)
        let (dec1, flags1, _, extra1) =
            thresh.select_action_decision(0.50, 0.08, 0.90, false, ATTRACTOR_SLOT_PERSONA);
        assert_eq!(dec1, StigmergicActionDecision::InertialLayerSkip);
        assert_eq!(flags1, ACTION_PRESET_INERTIAL_SKIP);
        assert_ne!(extra1 & TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);

        // 2. ExpressPhaseSteer: 0.35 <= density < 0.85, normal entropy
        let (dec2, flags2, _, _) =
            thresh.select_action_decision(0.55, 0.30, 0.85, false, ATTRACTOR_SLOT_KNOWLEDGE);
        assert_eq!(dec2, StigmergicActionDecision::ExpressPhaseSteer);
        assert_eq!(flags2, ACTION_PRESET_EXPRESS_PHASE);

        // 3. PspmSubHeadRoute: 0.85 <= density < 1.50
        let (dec3, flags3, _, _) =
            thresh.select_action_decision(1.10, 0.30, 0.85, false, ATTRACTOR_SLOT_KNOWLEDGE);
        assert_eq!(dec3, StigmergicActionDecision::PspmSubHeadRoute);
        assert_eq!(flags3, ACTION_MODE_DPSR_STEER | ACTION_MODE_PSPM_ROUTE);

        // 4. FullOrchestrationSrl: 1.50 <= density < 2.20
        let (dec4, flags4, _, _) =
            thresh.select_action_decision(1.80, 0.30, 0.85, false, ATTRACTOR_SLOT_PERSONA);
        assert_eq!(dec4, StigmergicActionDecision::FullOrchestrationSrl);
        assert_eq!(flags4, ACTION_PRESET_FULL_ORCHESTRATION);

        // 5. CrystallizeMacroBasin: density >= 2.20 and coherence >= 0.85
        let (dec5, flags5, _, extra5) =
            thresh.select_action_decision(2.50, 0.10, 0.92, false, ATTRACTOR_SLOT_PERSONA);
        assert_eq!(dec5, StigmergicActionDecision::CrystallizeMacroBasin);
        assert_eq!(flags5, ACTION_PRESET_FULL_ORCHESTRATION);
        assert_ne!(extra5 & TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, 0);

        // 6. RepulsionEscape: coherence < -0.25 or repulsion_triggered
        let (dec6, flags6, slot6, extra6) =
            thresh.select_action_decision(2.50, 0.30, -0.60, false, ATTRACTOR_SLOT_PERSONA);
        assert_eq!(dec6, StigmergicActionDecision::RepulsionEscape);
        assert_eq!(flags6, ACTION_PRESET_REPULSION_ESCAPE);
        assert_eq!(slot6, ATTRACTOR_SLOT_REPULSION);
        assert_ne!(extra6 & TRAIL_FLAG_REPULSION_ACTIVE, 0);
    }

    #[test]
    fn test_action_hub_engine_stigmergic_orchestration_and_gateway_kmeans() {
        let mut hub =
            ActionHubEngine::with_dimensions_and_thresholds(8, 2, 16, StigmergicThresholds::new())
                .unwrap();

        // Populate an inline candidate trail
        let mut t0 = gcso_pointer_trail_t {
            stigmergic_density: 1.2,
            target_anchor_id: 42,
            accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let ptr0 = gcso_tagged_ptr_t::pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            24,
        );
        hub.step_pointer(ptr0.raw(), &mut t0).unwrap();
        assert_eq!(hub.active_count(), 1);

        let mut q_buf = AlignedBuffer32::new([1.0f32; 16]);
        let mut k_buf = AlignedBuffer32::new([1.0f32; 16]);
        let mut logits = AlignedBuffer32::new([2.0f32; 8]);
        let mut scores = AlignedBuffer32::new([0.0f32; 8]);
        let mut active_trail = gcso_pointer_trail_t {
            stigmergic_density: 1.2,
            accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let report = hub
            .orchestrate_next_action_step(
                ptr0,
                0.35,
                ATTRACTOR_SLOT_PERSONA,
                &mut active_trail,
                Some(&mut scores.as_aligned_slice_mut()),
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
                Some(&mut logits.as_aligned_slice_mut()),
            )
            .unwrap();

        assert_eq!(
            report.action_decision(),
            StigmergicActionDecision::FullOrchestrationSrl
        );
        assert!(report.phase_coherence > 0.95);
        assert_eq!(active_trail.target_anchor_id, 42);

        // Verify Gateway + Online Toroidal k-Means closed-loop orchestration
        let sys_cfg = gcso_config_t {
            head_dim: 8,
            num_heads: 2,
            ripa_clamp_max_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: 0.01,
            ..Default::default()
        };
        let sys = SystemConstraint::new(&sys_cfg, 0.10, 0.05, 0xFF);
        let fact = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 2.0, 0x0010_0000, 77);
        let persona = SubjectivePersona::new([24i8; MAX_ATTENTION_HEADS], 2.5, 0x0020_0000, 42);
        let composite = (sys, fact, persona);
        let mut centroids = [CrystallizedAttractorRecord::default(); 2];

        let (gw_frame, gw_rep, c_idx, c_aid, c_coh, _spawned) = hub
            .orchestrate_gateway_kmeans_and_select_action(
                &composite,
                0.30,
                &mut centroids,
                0.55,
                &mut active_trail,
                None,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
                None,
            )
            .unwrap();

        assert_eq!(c_idx, 0);
        assert_eq!(c_aid, 42);
        assert!(c_coh > 0.90);
        assert_eq!(gw_frame.target_anchor_id, 42);
        assert_eq!(gw_frame.resolve_autonomous_rot_dim(16, 0.20, 1.5), 4);
        assert_eq!(hub.resolve_autonomous_rot_dim(0x04, 0xFF, 1.5), 2);
        assert!(gw_rep.updated_density >= 1.8);
        assert!(centroids[0].verify_integrity());
    }
}
