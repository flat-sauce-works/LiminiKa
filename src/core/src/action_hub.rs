// name: src/core/src/action_hub.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stigmergic Threshold Action-Selecting Orchestrator (`ActionHubEngine`) and Sidecar Pointer
//! Table (SPT) implementation for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! # Architectural Role (Indirect Pointer-Chain Graph Routing & Stigmergic Action Selection)
//! Rather than holding all long-term context inside active model weights, `LiminiKa` (`GCSO`)
//! represents active memory as a lightweight graph of 64-bit tagged pointers (`gcso_tagged_ptr_t`)
//! and 128-byte pointer trails (`gcso_pointer_trail_t`). The CPU only needs to hold the entry
//! pointer to a trail memo; each memo stores a 48-bit external mapped-storage storage offset, a 64-head
//! phase signature, pheromone density ($\rho_{\mathrm{stigmergy}}$), and a `linked_trail_id`
//! pointing to the next memo in the chain. Traversing these linked memos forms a sparse,
//! neural-like associative pathway that bridges inter-memory **Limina** gaps and drives
//! **threshold-driven next-action selection**:
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
//!    - [`StigmergicActionDecision::SubHeadRoute`] (`3`): $SO(2)$-fused RIPA + sub-head router
//!      (`Fact / Logic / Explore`) Sub-Head Pseudo-MoE routing when $\tau_{\mathrm{pspm}} \le \rho < \tau_{\mathrm{srl}}$
//!      or during EDBC pitchfork bifurcation (`ACTION_MODE_DPSR_STEER | ACTION_MODE_SUBHEAD_ROUTE`).
//!    - [`StigmergicActionDecision::FullOrchestration`] (`4`): Full $SO(2)$-fused RIPA + sub-head router +
//!      Dynamic Rank-1 Rank-1 residual residual injection when $\tau_{\mathrm{srl}} \le \rho < \tau_{\mathrm{crystal}}$
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
//!    working phase cache window (`AlignedBuffer64<gcso_q7_t, 512>`), a 64-byte [`gcso_ledger_slot_t`]
//!    scratchpad, a 64-byte [`ContrastivePhaseBloomFilter`], [`StigmergicThresholds`], and a
//!    [`PhaseSteeringEngine`] into an exact **2,816-byte (`22 x 128B`), 128-byte cacheline-aligned [`Pod`]
//!    structure** ($C_{\mathrm{alloc}} = 0$).
//!
//! 3. **Blanket Trait Closure**:
//!    Because [`ActionHubEngine`] implements [`PhaseSteering`], [`PointerActionHub`],
//!    [`FastPathLedger`], [`StigmergicChunkOps`], [`CausalProvenanceSignal`],
//!    [`crate::traits::SystemConstraintSignal`], and [`ManifoldGatewaySignal`], it automatically
//!    receives the blanket implementations of [`ActionHub`], [`ActionHubGatewayExt`],
//!    [`GatewayPhaseSteeringExt`], and [`crate::phase_steering::PhaseSteeringClosedLoop`].

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
    gcso_config_t, gcso_ledger_slot_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t,
    gcso_subhead_config_t, gcso_tagged_ptr_t, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::phase_steering::{PhaseSteeringEngine, DEFAULT_DPSR_HEAD_DIM, DEFAULT_DPSR_NUM_HEADS};
use crate::traits::aligned::{AlignedBuffer64, AlignedSlice16, AlignedSliceMut32};
use crate::traits::branchless::{
    branchless_fast_transition_log_prob, coherence_from_word_arrays_64, scale_f32_slice_unrolled8,
    swar_words_nonzero_64,
};
use crate::traits::gateway::{
    CausalProvenanceSignal, CrystallizedAttractorRecord, GatewayFrame, ManifoldGatewaySignal,
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_AXIS_SYSTEM_CONSTRAINT, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
    DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE, ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_NOMINAL,
    ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};
use crate::traits::hot_path::{PointerActionHub, StigmergicChunkOps};
use crate::traits::primitives::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_max_f32, branchless_min_f32,
    branchless_rad_to_qdps_step_u8, branchless_relu_f32, branchless_select_f32,
    branchless_select_u32, branchless_select_u64, branchless_select_u8, branchless_select_usize,
    fast_inv_len_f32, swar_or_avx2_q7_add_sat_words_64, swar_q7_qdps_filter_words_64,
    trail_phase_words_mut, trail_phase_words_ref, SWAR_SPLAT_BYTE_64,
};
use crate::traits::primitives::{
    pack_action_slot_tag, unpack_attractor_slot_from_tag, GcsoResult, Pod, ACTION_MODE_DPSR_STEER,
    ACTION_MODE_RANK1_INJECT, ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_CUSTOM_5, ATTRACTOR_SLOT_CUSTOM_6,
    ATTRACTOR_SLOT_CUSTOM_7, ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_KNOWLEDGE,
    ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE,
    ATTRACTOR_SLOT_USER_CONTEXT, DEFAULT_ACTION_HUB_CAPACITY, DEFAULT_RIPA_CLAMP_MAX_RAD,
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, FAST_PATH_SHORTCUT_SLOTS, LEDGER_HISTORY_LEN,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
};
use crate::traits::rope::{
    dispatch_fused_ripa_and_pspm_words_core, resolve_autonomous_swarm_rot_dim, ripa_upper_rot_dim,
    steer_from_swar_words_core_safe, PhaseSteering,
};
use crate::traits::telemetry::PheromoneDensityOps;
use crate::traits::telemetry::{
    select_rewire_bridge_and_modulate_with_preloaded_words_64, ContrastivePhaseBloomFilter,
    FastPathLedger, StigmergicActionPtrOps, StigmergicReconstitutionOps, StigmergicTrailOps,
    LEDGER_METRIC_FAST_PATH_HIT, LEDGER_METRIC_LAYER_SKIP_COMPENSATED, LEDGER_METRIC_PHASE_STEERED,
    LEDGER_METRIC_RANK1_INJECTED, LEDGER_METRIC_REPULSION_APPLIED,
    LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, LEDGER_METRIC_SUBHEAD_ROUTED, MAX_ATTRACTOR_PULL_FORCE,
    MAX_STIGMERGIC_DENSITY, NUM_STIGMERGIC_GRANULES, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_DPSR_ACTIVE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
use crate::traits::{ActionHub, ActionHubGatewayExt, GatewayPhaseSteeringExt};

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

/// Default stigmergic trail density threshold ($\tau_{\mathrm{pspm}} = 0.85$) triggering [`StigmergicActionDecision::SubHeadRoute`].
pub const DEFAULT_STIGMERGIC_SUBHEAD_ROUTE_THRESH: f32 = 0.85;

/// Default stigmergic trail density threshold ($\tau_{\mathrm{srl}} = 1.50$) triggering [`StigmergicActionDecision::FullOrchestration`].
pub const DEFAULT_STIGMERGIC_RANK1_INJECT_THRESH: f32 = 1.50;

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
    /// $SO(2)$-fused RIPA + sub-head router (`Fact / Logic / Explore`) Sub-Head Pseudo-MoE routing (`0x09`):
    /// $\tau_{\mathrm{pspm}} \le \rho < \tau_{\mathrm{srl}}$ or high bifurcation entropy.
    SubHeadRoute = 3,
    /// Full $SO(2)$-fused RIPA + sub-head router + Dynamic Rank-1 Rank-1 residual residual injection
    /// (`ACTION_PRESET_FULL_ORCHESTRATION = 0x19`): $\tau_{\mathrm{srl}} \le \rho < \tau_{\mathrm{crystal}}$.
    FullOrchestration = 4,
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
            3 => Self::SubHeadRoute,
            4 => Self::FullOrchestration,
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
            Self::SubHeadRoute => ACTION_MODE_DPSR_STEER | ACTION_MODE_SUBHEAD_ROUTE,
            Self::FullOrchestration | Self::CrystallizeMacroBasin => {
                ACTION_PRESET_FULL_ORCHESTRATION
            }
            Self::RepulsionEscape => ACTION_PRESET_REPULSION_ESCAPE,
        }
    }
}

/// Preset regime profile for configuring [`ActionHubEngine`]'s 8-lane phase workbench
/// at prompt or epistemic regime boundaries.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkbenchPreset {
    /// Balanced dialogue profile enabling baseline, persona, fact, and user-context lanes (`0`).
    #[default]
    BalancedDialog = 0,
    /// Factual grounding profile emphasizing knowledge, baseline, and custom specialist lanes (`1`).
    FactualGrounding = 1,
    /// Conflict/guard defense profile emphasizing repulsion, knowledge counter-anchor, and spare lane 7 (`2`).
    GuardDefense = 2,
    /// Exploratory liminal profile emphasizing persona, context, and custom lanes 5..=7 (`3`).
    CreativeLiminal = 3,
}

impl WorkbenchPreset {
    /// Maps an epistemic regime code (`0=KnownCoherent`, `1=InternalConflict`, `2=Liminal`) to
    /// its default [`WorkbenchPreset`].
    #[inline(always)]
    #[must_use]
    pub const fn from_regime_code(regime_code: u8) -> Self {
        match regime_code {
            1 => Self::GuardDefense,
            2 => Self::CreativeLiminal,
            _ => Self::BalancedDialog,
        }
    }

    /// Returns the 8-bit lane enable mask (`0..=7`) recommended for this preset.
    #[inline(always)]
    #[must_use]
    pub const fn recommended_lane_mask(self) -> u8 {
        match self {
            Self::BalancedDialog => {
                (1u8 << ATTRACTOR_SLOT_SYSTEM_BASELINE)
                    | (1u8 << ATTRACTOR_SLOT_PERSONA)
                    | (1u8 << ATTRACTOR_SLOT_KNOWLEDGE)
                    | (1u8 << ATTRACTOR_SLOT_USER_CONTEXT)
            }
            Self::FactualGrounding => {
                (1u8 << ATTRACTOR_SLOT_SYSTEM_BASELINE)
                    | (1u8 << ATTRACTOR_SLOT_KNOWLEDGE)
                    | (1u8 << ATTRACTOR_SLOT_CUSTOM_5)
                    | (1u8 << ATTRACTOR_SLOT_CUSTOM_6)
            }
            Self::GuardDefense => {
                (1u8 << ATTRACTOR_SLOT_SYSTEM_BASELINE)
                    | (1u8 << ATTRACTOR_SLOT_KNOWLEDGE)
                    | (1u8 << ATTRACTOR_SLOT_REPULSION)
                    | (1u8 << ATTRACTOR_SLOT_CUSTOM_7)
            }
            Self::CreativeLiminal => {
                (1u8 << ATTRACTOR_SLOT_PERSONA)
                    | (1u8 << ATTRACTOR_SLOT_USER_CONTEXT)
                    | (1u8 << ATTRACTOR_SLOT_CUSTOM_5)
                    | (1u8 << ATTRACTOR_SLOT_CUSTOM_6)
                    | (1u8 << ATTRACTOR_SLOT_CUSTOM_7)
            }
        }
    }
}

/// Zero-alloc POD descriptor for swapping a single specialist lane (`0..=7`) on
/// [`ActionHubEngine`]'s 8-lane working phase workbench.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkbenchSlotSpec {
    /// Target lane index (`0..=7`).
    pub lane: u8,
    /// Non-negative lane weight.
    pub weight: f32,
    /// 64-head Q0.7 phase profile loaded into the lane.
    pub phases: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

impl Default for WorkbenchSlotSpec {
    #[inline]
    fn default() -> Self {
        Self {
            lane: 0,
            weight: 0.0,
            phases: [0; MAX_ATTENTION_HEADS],
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
    /// Minimum stigmergic density $\tau_{\mathrm{pspm}}$ triggering [`StigmergicActionDecision::SubHeadRoute`] (default `0.85`).
    pub subhead_route_density_thresh: f32,
    /// Minimum stigmergic density $\tau_{\mathrm{srl}}$ triggering [`StigmergicActionDecision::FullOrchestration`] (default `1.50`).
    pub rank1_inject_density_thresh: f32,
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
    /// Creates the canonical GCSO stigmergic threshold ladder.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            evaporation_retain: DEFAULT_STIGMERGIC_EVAPORATION_RETAIN,
            reinforcement_gain: DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN,
            inertial_skip_entropy_thresh: DEFAULT_STIGMERGIC_INERTIAL_SKIP_ENTROPY,
            express_steer_density_thresh: DEFAULT_STIGMERGIC_EXPRESS_STEER_THRESH,
            subhead_route_density_thresh: DEFAULT_STIGMERGIC_SUBHEAD_ROUTE_THRESH,
            rank1_inject_density_thresh: DEFAULT_STIGMERGIC_RANK1_INJECT_THRESH,
            crystallize_density_thresh: DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH,
            contradiction_coherence_thresh: DEFAULT_STIGMERGIC_CONTRADICTION_THRESH,
        }
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
            & branchless_is_finite_f32(self.subhead_route_density_thresh)
            & (self.subhead_route_density_thresh >= self.express_steer_density_thresh)
            & branchless_is_finite_f32(self.rank1_inject_density_thresh)
            & (self.rank1_inject_density_thresh >= self.subhead_route_density_thresh)
            & branchless_is_finite_f32(self.crystallize_density_thresh)
            & (self.crystallize_density_thresh >= self.rank1_inject_density_thresh)
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
    #[inline]
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
        let srl_entropy_min = self.inertial_skip_entropy_thresh * 1.6_f32;
        let pspm_entropy_fallback = self.inertial_skip_entropy_thresh * 3.6_f32;
        let ge_steer = rho >= self.express_steer_density_thresh;
        let ge_pspm = (rho >= self.subhead_route_density_thresh)
            | (ge_steer & (h_tilde >= pspm_entropy_fallback));
        let ge_srl = (rho >= self.rank1_inject_density_thresh) & (h_tilde > srl_entropy_min);
        let ge_crystal = (rho >= self.crystallize_density_thresh)
            & (coh >= DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE);

        // Build decision discriminant bottom-up along the density ladder (`0..=6`):
        let mut disc = 0u8;
        disc = branchless_select_u8(
            ge_steer,
            StigmergicActionDecision::ExpressPhaseSteer as u8,
            disc,
        );
        disc = branchless_select_u8(ge_pspm, StigmergicActionDecision::SubHeadRoute as u8, disc);
        disc = branchless_select_u8(
            ge_srl,
            StigmergicActionDecision::FullOrchestration as u8,
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

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn execute_decision_steering_and_logit_scaling(
        decision: StigmergicActionDecision,
        steer_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
        let mut subhead_routed = false;
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
            StigmergicActionDecision::SubHeadRoute
            | StigmergicActionDecision::FullOrchestration
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
                            let subhead_cfg = gcso_subhead_config_t {
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
                                q,
                                &swar_w,
                                ripa_clamp,
                                &subhead_cfg,
                                &swar_w,
                                &swar_w,
                                &swar_w,
                                ripa_clamp,
                                0,
                                head_dim,
                                true,
                            )?;
                            if let Some(k) = key_tensor {
                                steer_from_swar_words_core_safe(
                                    &swar_w, 0xFF, 128, head_dim, num_heads, false, ripa_clamp, k,
                                    None,
                                );
                            }
                        }
                        steered = true;
                        subhead_routed = true;
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
        Ok((steered, subhead_routed, skip_compensated))
    }

    /// Executes a stateless one-shot stigmergic orchestration step directly from `self` and
    /// `phase_steering` without allocating a 2,816-byte [`ActionHubEngine`] on the stack
    /// (`0` stack table zero-init).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor`, `key_tensor`, or `logits_opt`
    /// alias or have invalid dimensions.
    #[inline]
    pub fn orchestrate_stateless_action_step(
        &self,
        phase_steering: &PhaseSteeringEngine,
        input_ptr: gcso_tagged_ptr_t,
        entropy_h_tilde: f32,
        dominant_slot: u8,
        trail_inout: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<ActionOrchestrationReport> {
        let qdps_min_rad = phase_steering.qdps_min_step_rad();
        let ripa_clamp = phase_steering.ripa_clamp_max_rad();
        let head_dim = phase_steering.head_dim() as usize;
        let num_heads = phase_steering.num_heads() as usize;
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
    /// Conditional transition log-probability $\ln P(M_t \mid M_{t-1}) \le 0$ between linked pointer trails.
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

    /// Computes the branchless stigmergic emergence potential
    /// $\Psi_{\mathrm{emerge}} = \rho_{\mathrm{stigmergy}} \cdot \max(0.1, \mathcal{C}_{\mathrm{phase}})$
    /// quantifying how strongly the Swarm and mapped-storage Attractors have reinforced this action trail.
    #[inline(always)]
    #[must_use]
    pub const fn emergence_potential(&self) -> f32 {
        let coh_weight = branchless_max_f32(0.1, self.phase_coherence);
        branchless_flush_subnormal_f32(branchless_relu_f32(self.updated_density) * coh_weight)
    }

    /// Returns `true` when this orchestration step crossed into an active non-passive regime and
    /// its [`Self::emergence_potential`] meets or exceeds `min_potential`.
    #[inline(always)]
    #[must_use]
    pub const fn is_action_emergent(&self, min_potential: f32) -> bool {
        (self.decision != (StigmergicActionDecision::PassiveSidecar as u8))
            & (self.emergence_potential() >= min_potential)
    }
}

/// Composable receptor trait invoked by [`ActionHubEngine`] when Swarm consensus and mapped-storage
/// attractor pull elevate a pointer trail's stigmergic pheromone density across an action regime.
///
/// Instead of writing ad-hoc `if/else` semantic branches (e.g., `if needs_web_search { ... }`),
/// external or Cold-Path capabilities (such as asynchronous retrieval, mapped-storage RAG prefetch, or
/// macro-basin consolidation) implement [`StigmergicActionReceptor`] and are composed into tuples
/// `(ReceptorA, ReceptorB)` that react when `report.target_slot`, `report.dominant_anchor_id`, and
/// `report.action_decision()` cross the physical stigmergic threshold.
pub trait StigmergicActionReceptor {
    /// Invoked when an [`ActionOrchestrationReport`] emerges from [`ActionHubEngine`] with a
    /// non-passive [`StigmergicActionDecision`].
    ///
    /// # Errors
    /// Returns a [`GcsoResult`] if the downstream receptor encounters an execution or capacity error.
    fn on_stigmergic_action_emerged(
        &mut self,
        frame: &GatewayFrame,
        report: &ActionOrchestrationReport,
        trail: &gcso_pointer_trail_t,
    ) -> GcsoResult<()>;
}

impl StigmergicActionReceptor for () {
    #[inline(always)]
    fn on_stigmergic_action_emerged(
        &mut self,
        _frame: &GatewayFrame,
        _report: &ActionOrchestrationReport,
        _trail: &gcso_pointer_trail_t,
    ) -> GcsoResult<()> {
        Ok(())
    }
}

impl<A: StigmergicActionReceptor, B: StigmergicActionReceptor> StigmergicActionReceptor for (A, B) {
    #[inline(always)]
    fn on_stigmergic_action_emerged(
        &mut self,
        frame: &GatewayFrame,
        report: &ActionOrchestrationReport,
        trail: &gcso_pointer_trail_t,
    ) -> GcsoResult<()> {
        self.0.on_stigmergic_action_emerged(frame, report, trail)?;
        self.1.on_stigmergic_action_emerged(frame, report, trail)
    }
}

// ===================================================================
// 2. `ActionHubEngine` — 2,816-Byte L1D-Resident Sidecar Pointer Table
// ===================================================================

impl gcso_ledger_slot_t {
    /// Executes strictly branchless $\mathcal{O}(1)$ fast-path shortcut lookup directly in-place
    /// on `self` without copying the 64-byte `gcso_ledger_slot_t` onto the stack.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn lookup_fast_path_branchless(&self, input_key: u64) -> (bool, u64) {
        let mask_hit = ((self.fast_path_bypass_mask & input_key) != 0) & (self.mode == 0);
        let idx = ((input_key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
        let candidate = self.fast_path_shortcuts[idx & (FAST_PATH_SHORTCUT_SLOTS - 1)];
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

    /// Pushes a profiling metric byte code (`LEDGER_METRIC_*`) into `self.telemetry_mini_ledger` in-place.
    #[inline(always)]
    pub(crate) fn push_telemetry_metric(&mut self, metric_code: u8) {
        let head = (self.telemetry_ring_head as usize) & (LEDGER_HISTORY_LEN - 1);
        self.telemetry_mini_ledger[head] = metric_code;
        self.telemetry_ring_head = self.telemetry_ring_head.wrapping_add(1);
    }
}

/// Concrete zero-allocation (`#[repr(C, align(128))]`, 2,816-byte [`Pod`]) Sidecar Pointer Table
/// and **Stigmergic Threshold Action-Selecting Orchestrator** implementing [`ActionHub`].
///
/// # Memory Layout (2,816 Bytes = `22 x 128B` Cache-Line Pairs, Zero Implicit Padding)
/// - `0..2048` (`2048B`): `trails: [gcso_pointer_trail_t; 16]` (16 inline 128B pointer trails)
/// - `2048..2560` (`512B`): `slot_phase_bank: AlignedBuffer64<gcso_q7_t, 512>` (8 working lanes $\times$ 64 heads)
/// - `2560..2624` (`64B`): `ledger_slot: gcso_ledger_slot_t` (FastPathLedger fast-path shortcuts & telemetry ring)
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
    pub ledger_slot: gcso_ledger_slot_t,
    /// Contrastive Phase Bloom Filter for $\mathcal{O}(1)$ anti-phase repulsion (64 bytes, 64-byte aligned).
    pub bloom_filter: ContrastivePhaseBloomFilter,
    /// Stigmergic threshold regime governing pheromone dynamics and next-action selection (32 bytes).
    pub thresholds: StigmergicThresholds,
    /// Active gravitational weights for the 8 Attractor Working Lanes (`0..=7`, 32 bytes).
    pub slot_weights: [f32; MAX_ATTRACTOR_DOMAINS],
    /// Embedded 16-byte DPSR/RIPA kernel configuration (`head_dim`, `num_heads`, `ripa_clamp_max_rad`, `qdps_min_step_rad`).
    pub dpsr: PhaseSteeringEngine,
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
        let dpsr = match PhaseSteeringEngine::new(DEFAULT_DPSR_HEAD_DIM, DEFAULT_DPSR_NUM_HEADS) {
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
            ledger_slot: gcso_ledger_slot_t {
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
        let dpsr = PhaseSteeringEngine::new(head_dim, num_heads)?;
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
        let dpsr = PhaseSteeringEngine::from_config(config)?;
        let mut hub = Self::new();
        hub.dpsr = dpsr;
        hub.capacity = config.action_hub_capacity;
        hub.ledger_slot.mode = u32::from(config.ledger_mode);
        Ok(hub)
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

    /// Returns an immutable 16-byte aligned slice view over the 8-lane Q7 working phase bank (`512` bytes).
    #[inline(always)]
    #[must_use]
    pub const fn slot_phase_bank(&self) -> AlignedSlice16<'_, gcso_q7_t> {
        self.slot_phase_bank.relax_slice::<16>()
    }

    /// Evaluates external cold-path escalation (`escalation_provider`) against `req.trigger` and
    /// swaps phases into `ATTRACTOR_SLOT_CUSTOM_7`, or cleanly clears and retracts that spare lane
    /// when `req.trigger.can_retract()` holds.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `req.target_slot >= MAX_ATTRACTOR_DOMAINS` or
    /// propagates any error returned by `escalation_provider`.
    #[inline]
    pub fn apply_horizon_escalation<
        E: crate::traits::gateway::HorizonEscalationProvider + ?Sized,
    >(
        &mut self,
        req: &crate::traits::gateway::HorizonEscalationRequest,
        escalation_provider: &mut E,
    ) -> GcsoResult<bool> {
        if usize::from(req.target_slot) >= crate::traits::primitives::MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let lane_idx = usize::from(req.target_slot);
        let lane_bit = 1u8 << req.target_slot;
        let start = lane_idx * crate::traits::primitives::MAX_ATTENTION_HEADS;
        let end = start + crate::traits::primitives::MAX_ATTENTION_HEADS;
        if req.trigger.is_destabilized() {
            let mut escalated_phases = [0i8; crate::traits::primitives::MAX_ATTENTION_HEADS];
            if let Some(weight) =
                escalation_provider.try_escalate_horizon(req, &mut escalated_phases)?
            {
                self.slot_phase_bank.data[start..end].copy_from_slice(&escalated_phases);
                self.slot_weights[lane_idx] = branchless_relu_f32(weight);
                self.slot_enabled_mask |= lane_bit;
                return Ok(true);
            }
        } else if req.trigger.can_retract() {
            if (self.slot_enabled_mask & lane_bit) != 0 {
                self.slot_phase_bank.data[start..end].fill(0);
            }
            self.slot_enabled_mask &= !lane_bit;
            self.slot_weights[lane_idx] = 0.0;
        }
        Ok(false)
    }

    /// Returns the bitmask of currently enabled lanes (`0..=7`) on the 8-lane working phase bank.
    #[inline(always)]
    #[must_use]
    pub const fn active_workbench_lane_mask(&self) -> u8 {
        self.slot_enabled_mask
    }

    /// Atomically swaps a preset batch of specialist lanes into the 8-lane working phase bank
    /// (`slot_phase_bank` + `slot_weights`) and updates `slot_enabled_mask` in zero-alloc
    /// constant space, returning the count of updated lanes.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any `spec.lane >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn apply_workbench_preset_lanes(
        &mut self,
        enable_mask: u8,
        specs: &[WorkbenchSlotSpec],
    ) -> GcsoResult<u8> {
        let mut updated = 0u8;
        let mut idx = 0usize;
        while idx < specs.len() {
            let spec = &specs[idx];
            if usize::from(spec.lane) >= crate::traits::primitives::MAX_ATTRACTOR_DOMAINS {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            let lane_idx = usize::from(spec.lane);
            let start = lane_idx * crate::traits::primitives::MAX_ATTENTION_HEADS;
            let end = start + crate::traits::primitives::MAX_ATTENTION_HEADS;
            self.slot_phase_bank.data[start..end].copy_from_slice(&spec.phases);
            self.slot_weights[lane_idx] = branchless_relu_f32(spec.weight);
            updated = updated.saturating_add(1);
            idx += 1;
        }
        self.slot_enabled_mask = enable_mask;
        Ok(updated)
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
            let _ = self.trails[i].evaporate_pheromone(retain);
            i += 1;
        }

        let next_rho = trail_inout.step_pheromone(
            retain,
            self.thresholds.reinforcement_gain,
            phase_coherence,
            contradiction_repulsed,
        );

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
        phase_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
    /// 1. Resolves FastPathLedger fast-path shortcut (`resolve_or_pass_tagged_ptr`).
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
    ///    `SubHeadRoute`, `FullOrchestration`, `CrystallizeMacroBasin`, or `RepulsionEscape`).
    /// 7. Executes the selected action on `query_tensor` / `key_tensor` / `logits_opt` and logs
    ///    telemetry into `self.ledger_slot`.
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

    #[inline]
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
        macro_centroid_words: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
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

        let (bypass_hit, eff_ptr) = self.ledger_slot.resolve_or_pass_in_place(input_ptr);

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
                    as *const [u64; NUM_STIGMERGIC_GRANULES])
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
                    as *const [u64; NUM_STIGMERGIC_GRANULES])
            };
            if swar_words_nonzero_64(slot_w) {
                swar_or_avx2_q7_add_sat_words_64(&mut steer_words, slot_w);
                *trail_phase_words_mut(trail_inout) = steer_words;
            }
        }
        Self::copy_trail_and_words_to_slot_fast(
            &mut self.trails[stored_slot_idx as usize],
            trail_inout,
            &steer_words,
        );

        let (steered, subhead_routed, skip_compensated) =
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

        let srl_flag = (mode_flags & ACTION_MODE_RANK1_INJECT) != 0;
        let metric = branchless_select_u8(bypass_hit, LEDGER_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, LEDGER_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(subhead_routed, LEDGER_METRIC_SUBHEAD_ROUTED, 0)
            | branchless_select_u8(srl_flag, LEDGER_METRIC_RANK1_INJECTED, 0)
            | branchless_select_u8(skip_compensated, LEDGER_METRIC_LAYER_SKIP_COMPENSATED, 0)
            | branchless_select_u8(contradiction_repulsed, LEDGER_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(found_coherent, LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        self.ledger_slot.push_telemetry_metric(metric);

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
    #[allow(clippy::too_many_arguments)]
    fn commit_gateway_kmeans_and_select_action(
        &mut self,
        frame: &GatewayFrame,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        entropy_h_tilde: f32,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        trail_inout: &mut gcso_pointer_trail_t,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(ActionOrchestrationReport, usize, u32, f32, bool)> {
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
                frame,
                input_words,
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
            report,
            assigned_idx,
            assigned_anchor_id,
            assigned_coh,
            spawned_new,
        ))
    }

    /// Executes Gateway synthesis, Online Toroidal $k$-Means / DP-Means clustering, and
    /// Stigmergic action selection in a single `#![no_std]` zero-allocation pass.
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
        let (report, assigned_idx, assigned_anchor_id, assigned_coh, spawned_new) = self
            .commit_gateway_kmeans_and_select_action(
                &frame,
                &frm_w,
                entropy_h_tilde,
                centroids,
                split_coherence_thresh,
                trail_inout,
                attention_scores,
                query_tensor,
                key_tensor,
                logits_opt,
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

    /// Executes the full **Gateway-Origin Trait Composition $\to$ 8-Cell Micro-Swarm Consensus $\to$
    /// mapped-storage Attractor Basin Pull $\to$ Stigmergic Pheromone Action Selection & Receptor Dispatch**
    /// closed loop without manual hyperparameter arguments or semantic `if/else` task branching.
    ///
    /// 1. Synthesizes `(GatewayFrame, [u64; 8])` from `signal: &S` ([`ManifoldGatewaySignal`]).
    /// 2. Resolves the 8-cell SWAR Micro-Swarm (`try_resolve_action_in_swarm_from_words`) using
    ///    autonomous physical parameters derived directly from the [`GatewayFrame`].
    /// 3. Steps the mapped-storage Online Toroidal $k$-Means / DP-Means attractor basins (`centroids`) and
    ///    evolves `trail_inout`'s stigmergic pheromone density (`orchestrate_next_action_step_core`).
    /// 4. Dispatches `receptor.on_stigmergic_action_emerged(&frame, &report, trail_inout)` whenever
    ///    a non-passive [`StigmergicActionDecision`] emerges from the trail's accumulated density.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `centroids` is empty or tensor slices are invalid.
    #[inline]
    pub fn orchestrate_gateway_attractor_swarm_step<S, R>(
        &mut self,
        signal: &S,
        entropy_h_tilde: f32,
        centroids: &mut [CrystallizedAttractorRecord],
        trail_inout: &mut gcso_pointer_trail_t,
        receptor: &mut R,
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
    )>
    where
        S: ManifoldGatewaySignal + ?Sized,
        R: StigmergicActionReceptor + ?Sized,
    {
        if centroids.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (frame, init_w) = signal.synthesize_gateway_frame_with_words();
        let (_swarm_rep, swarm_w) = trail_inout.try_resolve_action_in_swarm_from_words(
            gcso_tagged_ptr_t(frame.tagged_ptr),
            &init_w,
            frame.active_swarm_cell_mask,
            frame.autonomous_swarm_step_gain_u7(),
            frame.qdps_min_step_q7,
            frame.autonomous_swarm_tolerance_q7(),
            frame.autonomous_swarm_quorum_threshold(),
            entropy_h_tilde,
            self.thresholds.inertial_skip_entropy_thresh,
            Some(&self.bloom_filter),
            3,
            96,
        );
        let (report, assigned_idx, assigned_anchor_id, assigned_coh, spawned_new) = self
            .commit_gateway_kmeans_and_select_action(
                &frame,
                &swarm_w,
                entropy_h_tilde,
                centroids,
                DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
                trail_inout,
                attention_scores,
                query_tensor,
                key_tensor,
                logits_opt,
            )?;

        if report.action_decision() != StigmergicActionDecision::PassiveSidecar {
            receptor.on_stigmergic_action_emerged(&frame, &report, trail_inout)?;
        }

        Ok((
            frame,
            report,
            assigned_idx,
            assigned_anchor_id,
            assigned_coh,
            spawned_new,
        ))
    }

    /// Traverses a multi-hop **Indirect Pointer Chain** (`start_trail_idx -> linked_trail_id -> ...`)
    /// across the inline L1D Sidecar Pointer Table (`self.trails`), accumulating coherent 64-head
    /// SWAR phase signatures along the pathway, reinforcing traversed memo links, and executing
    /// stigmergic threshold action selection & RoPE steering from the terminal memo.
    ///
    /// Starting from a single entry memo index (`start_trail_idx`), the CPU follows up to
    /// `max_hops` (`1..=ACTION_HUB_INLINE_TRAILS`) linked memos using a 16-bit visited bitmask
    /// for $\mathcal{O}(1)$ zero-allocation cycle prevention. Each coherent hop ($\mathcal{C}_{\mathrm{hop}} > 0$)
    /// contributes its phase delta to `trail_inout` and receives Hebbian/stigmergic pheromone
    /// reinforcement along the traversed pathway.
    ///
    /// Returns `(ActionOrchestrationReport, hops_traversed, terminal_trail_idx, terminal_anchor_id, mean_chain_coherence)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `start_trail_idx >= self.inline_active_len()` or
    /// `max_hops == 0`.
    #[inline]
    pub fn traverse_pointer_chain_and_orchestrate(
        &mut self,
        start_trail_idx: usize,
        max_hops: usize,
        entropy_h_tilde: f32,
        trail_inout: &mut gcso_pointer_trail_t,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(ActionOrchestrationReport, u32, usize, u32, f32)> {
        let active_len = self.inline_active_len();
        if start_trail_idx >= active_len || max_hops == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let bounded_hops = branchless_select_usize(
            max_hops > ACTION_HUB_INLINE_TRAILS,
            ACTION_HUB_INLINE_TRAILS,
            max_hops,
        );

        let mut cur_idx = start_trail_idx;
        let mut visited_mask: u16 = 0;
        let mut hops: u32 = 0;
        let mut coh_sum = 0.0_f32;
        let mut chain_w = *trail_phase_words_ref(&self.trails[start_trail_idx]);
        let mut terminal_ptr = gcso_tagged_ptr_t(self.trails[start_trail_idx].current_ptr);
        let mut terminal_anchor_id = self.trails[start_trail_idx].target_anchor_id;
        let mut max_chain_rho = self.trails[start_trail_idx].stigmergic_density;
        let mut max_chain_pull = self.trails[start_trail_idx].attractor_pull_force;
        let reinf = branchless_relu_f32(self.thresholds.reinforcement_gain) * 0.5;

        let mut step = 0usize;
        while step < bounded_hops {
            let bit = 1u16 << (cur_idx & (ACTION_HUB_INLINE_TRAILS - 1));
            if (visited_mask & bit) != 0 {
                break;
            }
            visited_mask |= bit;
            hops = hops.wrapping_add(1);

            let node_w = *trail_phase_words_ref(&self.trails[cur_idx]);
            let hop_coh = if step == 0 {
                1.0_f32
            } else {
                coherence_from_word_arrays_64(&chain_w, &node_w)
            };

            if hop_coh < -self.thresholds.contradiction_coherence_thresh {
                break;
            }

            if step > 0 {
                swar_or_avx2_q7_add_sat_words_64(&mut chain_w, &node_w);
            }

            coh_sum += hop_coh;
            let node = &mut self.trails[cur_idx];
            let boosted_rho = branchless_min_f32(
                branchless_flush_subnormal_f32(
                    branchless_relu_f32(node.stigmergic_density)
                        + reinf * branchless_relu_f32(hop_coh),
                ),
                MAX_STIGMERGIC_DENSITY,
            );
            node.stigmergic_density = boosted_rho;
            max_chain_rho = branchless_max_f32(max_chain_rho, boosted_rho);
            max_chain_pull = branchless_max_f32(max_chain_pull, node.attractor_pull_force);
            if node.current_ptr != 0 {
                terminal_ptr = gcso_tagged_ptr_t(node.current_ptr);
            }
            if node.target_anchor_id != 0 {
                terminal_anchor_id = node.target_anchor_id;
            }

            let next_id = node.linked_trail_id as usize;
            if next_id >= active_len || next_id == cur_idx {
                break;
            }
            cur_idx = next_id;
            step += 1;
        }

        let mean_coh = branchless_select_f32(
            hops > 0,
            branchless_clamp_f32(coh_sum * fast_inv_len_f32(hops as usize), -1.0, 1.0),
            0.0,
        );
        *trail_phase_words_mut(trail_inout) = chain_w;
        trail_inout.stigmergic_density =
            branchless_max_f32(trail_inout.stigmergic_density, max_chain_rho);
        trail_inout.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_inout.attractor_pull_force, max_chain_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        if terminal_anchor_id != 0 {
            trail_inout.target_anchor_id = terminal_anchor_id;
        }
        trail_inout.linked_trail_id = cur_idx as u32;

        let dom_slot = terminal_ptr.target_attractor_slot();
        let report = self.orchestrate_next_action_step_core(
            terminal_ptr,
            entropy_h_tilde,
            dom_slot,
            trail_inout,
            attention_scores,
            query_tensor,
            key_tensor,
            logits_opt,
            false,
            Some(&chain_w),
        )?;

        Ok((report, hops, cur_idx, terminal_anchor_id, mean_coh))
    }
}

// ===================================================================
// 3. Trait Implementations for `ActionHubEngine`
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

impl FastPathLedger for ActionHubEngine {
    #[inline]
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_ledger_slot_t,
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
        config_out.ledger_mode = (slot.mode & 0xFF) as u8;
        let _ = LEDGER_HISTORY_LEN;
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
        let min_active = ((inline_idx as u32) + 1).min(self.capacity);
        self.active_count = branchless_select_u32(
            self.active_count < min_active,
            min_active,
            self.active_count,
        );
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
                    as *mut [u64; NUM_STIGMERGIC_GRANULES])
            };
            swar_or_avx2_q7_add_sat_words_64(w, &[splat; NUM_STIGMERGIC_GRANULES]);
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
        let min_active = ((inline_idx as u32) + 1).min(self.capacity);
        self.active_count = branchless_select_u32(
            self.active_count < min_active,
            min_active,
            self.active_count,
        );
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
        let dst_idx = (dst_trail_id as usize) & (ACTION_HUB_INLINE_TRAILS - 1);
        let min_active = ((src_idx.max(dst_idx) as u32) + 1).min(self.capacity);
        self.active_count = branchless_select_u32(
            self.active_count < min_active,
            min_active,
            self.active_count,
        );
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

impl StigmergicChunkOps for ActionHubEngine {
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
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let sel_idx = (self.last_selected_trail_id as usize) & (ACTION_HUB_INLINE_TRAILS - 1);
        let latest_idx =
            (self.write_cursor.wrapping_sub(1) as usize) & (ACTION_HUB_INLINE_TRAILS - 1);
        let idx = if (self.active_count > 0) & (self.trails[sel_idx].current_ptr == 0) {
            latest_idx
        } else {
            sel_idx
        };
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
            mapped_section_offset48: cur_tp.unpack_addr(),
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
            subhead_fact_heads: fact_h,
            subhead_logic_heads: logic_h,
            subhead_explore_heads: explore_h,
            subhead_fact_gain_q7: 128,
            subhead_logic_gain_q7: 128,
            subhead_explore_gain_q7: 128,
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
    assert!(offset_of!(ActionHubEngine, ledger_slot) == 2560);
    assert!(offset_of!(ActionHubEngine, bloom_filter) == 2624);
    assert!(offset_of!(ActionHubEngine, thresholds) == 2688);

    const fn assert_action_hub_bounds<
        T: ActionHub
            + ActionHubGatewayExt
            + GatewayPhaseSteeringExt
            + crate::phase_steering::PhaseSteeringClosedLoop
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

        // 2b. High-entropy fallback requires at least express_steer_density_thresh; sub-express stays PassiveSidecar.
        let (dec2_low, _, _, _) =
            thresh.select_action_decision(0.10, 0.60, 0.85, false, ATTRACTOR_SLOT_KNOWLEDGE);
        assert_eq!(dec2_low, StigmergicActionDecision::PassiveSidecar);
        let (dec2_fb, _, _, _) =
            thresh.select_action_decision(0.55, 0.60, 0.85, false, ATTRACTOR_SLOT_KNOWLEDGE);
        assert_eq!(dec2_fb, StigmergicActionDecision::SubHeadRoute);

        // 3. SubHeadRoute: 0.85 <= density < 1.50
        let (dec3, flags3, _, _) =
            thresh.select_action_decision(1.10, 0.30, 0.85, false, ATTRACTOR_SLOT_KNOWLEDGE);
        assert_eq!(dec3, StigmergicActionDecision::SubHeadRoute);
        assert_eq!(flags3, ACTION_MODE_DPSR_STEER | ACTION_MODE_SUBHEAD_ROUTE);

        // 4. FullOrchestration: 1.50 <= density < 2.20
        let (dec4, flags4, _, _) =
            thresh.select_action_decision(1.80, 0.30, 0.85, false, ATTRACTOR_SLOT_PERSONA);
        assert_eq!(dec4, StigmergicActionDecision::FullOrchestration);
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
            StigmergicActionDecision::FullOrchestration
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

        struct MockExternalSearchReceptor {
            fired_count: u32,
            last_anchor_id: u32,
            last_slot: u8,
        }
        impl StigmergicActionReceptor for MockExternalSearchReceptor {
            fn on_stigmergic_action_emerged(
                &mut self,
                _frame: &GatewayFrame,
                report: &ActionOrchestrationReport,
                _trail: &gcso_pointer_trail_t,
            ) -> GcsoResult<()> {
                if report.is_action_emergent(1.0) {
                    self.fired_count += 1;
                    self.last_anchor_id = report.dominant_anchor_id;
                    self.last_slot = report.target_slot;
                }
                Ok(())
            }
        }

        let mut search_receptor = MockExternalSearchReceptor {
            fired_count: 0,
            last_anchor_id: 0,
            last_slot: 0,
        };
        let (_f2, rep2, _idx2, aid2, _coh2, _sp2) = hub
            .orchestrate_gateway_attractor_swarm_step(
                &composite,
                0.30,
                &mut centroids,
                &mut active_trail,
                &mut search_receptor,
                None,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
                None,
            )
            .unwrap();
        assert_eq!(search_receptor.fired_count, 1);
        assert_eq!(search_receptor.last_anchor_id, aid2);
        assert!(rep2.emergence_potential() > 1.5);

        // Verify multi-hop pointer-chain traversal (trail[0] -> trail[1] -> trail[2])
        hub.trails[0].linked_trail_id = 1;
        hub.trails[0].stigmergic_density = 0.8;
        hub.trails[0].accumulated_phase_delta = [8i8; MAX_ATTENTION_HEADS];
        hub.trails[1].linked_trail_id = 2;
        hub.trails[1].stigmergic_density = 1.1;
        hub.trails[1].accumulated_phase_delta = [10i8; MAX_ATTENTION_HEADS];
        hub.trails[2].linked_trail_id = 2; // terminal self-link stops traversal cleanly
        hub.trails[2].stigmergic_density = 1.6;
        hub.trails[2].target_anchor_id = 777;
        hub.trails[2].current_ptr = gcso_tagged_ptr_t::pack(
            0x2000,
            pack_action_slot_tag(ATTRACTOR_SLOT_PERSONA, ACTION_PRESET_FULL_ORCHESTRATION),
            12,
        )
        .raw();
        hub.trails[2].accumulated_phase_delta = [12i8; MAX_ATTENTION_HEADS];
        hub.active_count = 3;

        let mut chain_trail = gcso_pointer_trail_t::default();
        let (chain_rep, hops, term_idx, term_aid, chain_coh) = hub
            .traverse_pointer_chain_and_orchestrate(
                0,
                8,
                0.25,
                &mut chain_trail,
                None,
                Some(&mut q_buf.as_aligned_slice_mut()),
                Some(&mut k_buf.as_aligned_slice_mut()),
                None,
            )
            .unwrap();
        assert_eq!(hops, 3);
        assert_eq!(term_idx, 2);
        assert_eq!(term_aid, 777);
        assert!(chain_coh > 0.95);
        assert_eq!(chain_rep.dominant_anchor_id, 777);
        assert!(hub.trails[1].stigmergic_density > 1.1);
    }
}
