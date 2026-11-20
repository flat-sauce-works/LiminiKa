// name: src/core/src/stigmergy/action_hub.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stigmergic Threshold Action-Selecting Orchestrator (`ActionHubEngine`) and Sidecar
//! Pointer Table implementation.

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

use crate::foundation::{
    branchless_abs_f32, branchless_argmin_f32_x8_masked, branchless_clamp_f32,
    branchless_fast_transition_log_prob, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_max_f32, branchless_min_f32, branchless_rad_to_phase_step_u8, branchless_relu_f32,
    coherence_from_word_arrays_64, dispatch_fused_phase_clamp_and_subhead_words_core,
    fast_inv_len_f32, gcso_paged_bitmask_t, gcso_subhead_config_t, gcso_tagged_ptr_t,
    load_q7_words_64, pack_action_slot_tag, phase_clamp_upper_rot_dim,
    resolve_autonomous_swarm_rot_dim, scale_f32_slice_unrolled8, steer_from_swar_words_core_safe,
    swar_or_avx2_q7_add_sat_words_64, swar_q7_phase_step_filter_words_64, swar_words_nonzero_64,
    unpack_attractor_slot_from_tag, AlignedBuffer64, AlignedSlice16, AlignedSliceMut32,
    PhaseSteering, Pod, ACTION_MODE_PHASE_STEER, ACTION_MODE_RANK1_INJECT,
    ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION,
    ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE,
    ATTRACTOR_SLOT_ID_MASK, DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEER_HEAD_DIM,
    DEFAULT_PHASE_STEER_NUM_HEADS, SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, branchless_select_u64, branchless_select_u8,
    branchless_select_usize, gcso_config_t, gcso_q7_t, BitCut8, GcsoResult,
    GCSO_ERROR_ACTION_HUB_FULL, GCSO_ERROR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, NUM_STIGMERGIC_GRANULES,
};
use crate::stigmergy::crystal::DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE;
use crate::stigmergy::gate::PhaseSteeringEngine;
use crate::stigmergy::granule::StigmergicGranuleField;
use crate::stigmergy::ledger::{gcso_ledger_slot_t, ContrastivePhaseBloomFilter, FastPathLedger};
use crate::stigmergy::pheromone::PheromoneDensityOps;
use crate::stigmergy::trail::{
    gcso_pointer_trail_t, select_rewire_bridge_and_modulate_with_preloaded_words_64,
    trail_phase_words_mut, trail_phase_words_ref, ActionHub, PointerActionHub,
};
use crate::stigmergy::{
    ATTRACTOR_SLOT_REPULSION, DEFAULT_ACTION_HUB_CAPACITY, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    LEDGER_HISTORY_LEN, WORKBENCH_LANE_FLAG_PINNED, WORKBENCH_LANE_FLAG_REPULSION,
    WORKBENCH_LANE_FLAG_SANDBOX,
};
use crate::stigmergy::{
    LEDGER_METRIC_FAST_PATH_HIT, LEDGER_METRIC_LAYER_SKIP_COMPENSATED, LEDGER_METRIC_PHASE_STEERED,
    LEDGER_METRIC_RANK1_INJECTED, LEDGER_METRIC_REPULSION_APPLIED,
    LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, LEDGER_METRIC_SUBHEAD_ROUTED, MAX_ATTRACTOR_PULL_FORCE,
    MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_PHASE_STEER_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SYNERGY_RESONANCE,
};

/// Number of inline 128-byte [`gcso_pointer_trail_t`] cells stored inside [`ActionHubEngine`] (`16` = 2,048 bytes).
pub const ACTION_HUB_INLINE_TRAILS: usize = 16;

/// Default per-step pheromone retention factor ($\lambda_{\mathrm{retain}} = 0.95$).
pub const DEFAULT_STIGMERGIC_EVAPORATION_RETAIN: f32 = 0.95;

/// Default pheromone reinforcement gain ($g_{\mathrm{reinforce}} = 0.35$) deposited on coherent transitions.
pub const DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN: f32 = 0.35;
/// Default Q0.7 derivative weight $\alpha \in [0, 127]$ for convex integral/derivative pheromone synthesis.
pub const DEFAULT_DERIVATIVE_ALPHA_Q7: u8 = 48;
/// Minimum Q0.7 derivative weight permitted by Slow-Loop self-tuning.
pub const MIN_DERIVATIVE_ALPHA_Q7: u8 = 16;
/// Maximum Q0.7 derivative weight permitted by Slow-Loop self-tuning.
pub const MAX_DERIVATIVE_ALPHA_Q7: u8 = 96;

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
    /// Single-pass `PhaseClamp` soft-clamped RoPE phase steering (`ACTION_PRESET_EXPRESS_PHASE = 0x01`):
    /// $\tau_{\mathrm{steer}} \le \rho < \tau_{\mathrm{pspm}}$.
    ExpressPhaseSteer = 2,
    /// $SO(2)$-fused `PhaseClamp` + sub-head router (`Fact / Logic / Explore`) Sub-Head Pseudo-MoE routing (`0x09`):
    /// $\tau_{\mathrm{pspm}} \le \rho < \tau_{\mathrm{srl}}$ or high bifurcation entropy.
    SubHeadRoute = 3,
    /// Full $SO(2)$-fused `PhaseClamp` + sub-head router + Dynamic Rank-1 residual injection
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
            Self::SubHeadRoute => ACTION_MODE_PHASE_STEER | ACTION_MODE_SUBHEAD_ROUTE,
            Self::FullOrchestration | Self::CrystallizeMacroBasin => {
                ACTION_PRESET_FULL_ORCHESTRATION
            }
            Self::RepulsionEscape => ACTION_PRESET_REPULSION_ESCAPE,
        }
    }
}

/// Preset regime profile guiding lane budgets and attribute policies for [`ActionHubEngine`]'s
/// anonymous 8-lane working phase workbench at prompt or epistemic regime boundaries.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkbenchPreset {
    /// Balanced dialogue profile balancing grounded memory and conversational persona (`0`).
    #[default]
    BalancedDialog = 0,
    /// Factual grounding profile prioritizing multiple objective knowledge attractors (`1`).
    FactualGrounding = 1,
    /// Conflict/guard defense profile prioritizing repulsion escape lanes alongside counter-anchors (`2`).
    GuardDefense = 2,
    /// Exploratory liminal profile permitting isolated sandbox hypothesis lanes (`3`).
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

    /// Returns the recommended active lane budget (`1..=8`) for this regime profile.
    #[inline(always)]
    #[must_use]
    pub const fn recommended_lane_budget(self) -> u8 {
        match self {
            Self::BalancedDialog => 4,
            Self::FactualGrounding | Self::GuardDefense => 5,
            Self::CreativeLiminal => 6,
        }
    }

    /// Returns whether this preset permits isolated exterior-only sandbox hypothesis lanes.
    #[inline(always)]
    #[must_use]
    pub const fn allows_sandbox_lanes(self) -> bool {
        matches!(self, Self::CreativeLiminal)
    }

    /// Returns whether this preset prioritizes anti-phase repulsion lanes during recall loading.
    #[inline(always)]
    #[must_use]
    pub const fn prioritizes_repulsion(self) -> bool {
        matches!(self, Self::GuardDefense)
    }
}

/// Placement policy specifying how a phase profile is assigned to one of the 8 anonymous
/// physical lanes (`0..=7`) inside [`ActionHubEngine::slot_phase_bank`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkbenchLanePlacement {
    /// Claims the lowest free unpinned lane (`(!enabled & !pinned).trailing_zeros()`), or evicts
    /// the lowest-weight unpinned lane when all unpinned lanes are occupied.
    #[default]
    AutoFree,
    /// Targets an explicit physical workbench lane (`0..=7`) and optionally pins it.
    Explicit {
        /// Physical workbench lane index (`0..=7`).
        lane: u8,
        /// When `true`, marks `lane` in `slot_pinned_mask` to prevent eviction or bulk unpinned release.
        pin: bool,
    },
}

impl WorkbenchLanePlacement {
    /// Creates an automatic free-pool placement policy ([`Self::AutoFree`]).
    #[inline(always)]
    #[must_use]
    pub const fn auto_free() -> Self {
        Self::AutoFree
    }

    /// Creates an explicit physical lane placement policy (`0..=7`) with configurable pinning.
    #[inline(always)]
    #[must_use]
    pub const fn explicit(lane: u8, pin: bool) -> Self {
        Self::Explicit { lane, pin }
    }

    /// Creates a pinned explicit physical lane placement policy (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn pinned(lane: u8) -> Self {
        Self::Explicit { lane, pin: true }
    }
}

/// Zero-alloc descriptor for loading a phase profile onto [`ActionHubEngine`]'s anonymous 8-lane
/// working phase workbench.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkbenchSlotSpec {
    /// Physical lane placement policy (`AutoFree` or `Explicit { lane, pin }`).
    pub placement: WorkbenchLanePlacement,
    /// Bitmask of `WORKBENCH_LANE_FLAG_*` behavior attributes (`PINNED`, `REPULSION`, `SANDBOX`).
    pub flags: u8,
    /// Non-negative lane weight.
    pub weight: f32,
    /// 64-head Q0.7 phase profile loaded into the lane.
    pub phases: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

impl Default for WorkbenchSlotSpec {
    #[inline]
    fn default() -> Self {
        Self {
            placement: WorkbenchLanePlacement::AutoFree,
            flags: 0,
            weight: 0.0,
            phases: [0; MAX_ATTENTION_HEADS],
        }
    }
}

impl WorkbenchSlotSpec {
    /// Constructs a standard unpinned [`WorkbenchSlotSpec`] allocated from the free pool.
    #[inline(always)]
    #[must_use]
    pub const fn auto_free(weight: f32, phases: [gcso_q7_t; MAX_ATTENTION_HEADS]) -> Self {
        Self {
            placement: WorkbenchLanePlacement::AutoFree,
            flags: 0,
            weight,
            phases,
        }
    }

    /// Constructs a repulsion-flagged [`WorkbenchSlotSpec`] allocated from the free pool.
    #[inline(always)]
    #[must_use]
    pub const fn auto_repulsion(weight: f32, phases: [gcso_q7_t; MAX_ATTENTION_HEADS]) -> Self {
        Self {
            placement: WorkbenchLanePlacement::AutoFree,
            flags: WORKBENCH_LANE_FLAG_REPULSION,
            weight,
            phases,
        }
    }

    /// Constructs an isolated exterior-only sandbox [`WorkbenchSlotSpec`] allocated from the free pool.
    #[inline(always)]
    #[must_use]
    pub const fn auto_sandbox(weight: f32, phases: [gcso_q7_t; MAX_ATTENTION_HEADS]) -> Self {
        Self {
            placement: WorkbenchLanePlacement::AutoFree,
            flags: WORKBENCH_LANE_FLAG_SANDBOX,
            weight,
            phases,
        }
    }

    /// Constructs an explicitly targeted [`WorkbenchSlotSpec`] for physical lane `lane` (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn explicit(
        lane: u8,
        pin: bool,
        flags: u8,
        weight: f32,
        phases: [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> Self {
        Self {
            placement: WorkbenchLanePlacement::Explicit { lane, pin },
            flags: if pin {
                flags | WORKBENCH_LANE_FLAG_PINNED
            } else {
                flags
            },
            weight,
            phases,
        }
    }

    /// Returns `true` if this spec requests pinning in `slot_pinned_mask`.
    #[inline(always)]
    #[must_use]
    pub const fn is_pinned(&self) -> bool {
        (self.flags & WORKBENCH_LANE_FLAG_PINNED) != 0
            || matches!(
                self.placement,
                WorkbenchLanePlacement::Explicit { pin: true, .. }
            )
    }

    /// Returns `true` if this spec marks the allocated lane as an anti-phase repulsion lane.
    #[inline(always)]
    #[must_use]
    pub const fn is_repulsion(&self) -> bool {
        (self.flags & WORKBENCH_LANE_FLAG_REPULSION) != 0
    }

    /// Returns `true` if this spec marks the allocated lane as an exterior-only sandbox lane.
    #[inline(always)]
    #[must_use]
    pub const fn is_sandbox(&self) -> bool {
        (self.flags & WORKBENCH_LANE_FLAG_SANDBOX) != 0
    }
}

/// Outcome emitted by `ActionHubEngine::step_sandbox_hypothesis_lifecycle`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxLifecycleEvent {
    /// No sandbox state transition occurred on this step.
    Idle,
    /// A new isolated sandbox lane was leased from the free pool.
    Nucleated(u8),
    /// An active sandbox lane was promoted in-place to standard full-aperture participation.
    Promoted(u8),
    /// Unpinned active sandbox lanes were released immediately due to repulsion or overdrive.
    Released(u8),
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

    /// Combines integrated pheromone density $\rho_{\mathrm{int}}$ with a deadband-guarded positive
    /// rising derivative $\Delta\rho_+$ via a Q0.7 convex combination weighted by the branchless
    /// $4p(1-p)$ boundary sensitivity gate (`0 fdiv`, `0 alloc`).
    ///
    /// Returns `(effective_rho, exceeds_deadband)` where `effective_rho` stays strictly within
    /// `[0.0, MAX_STIGMERGIC_DENSITY]`, preserving the monotone threshold ladder.
    #[inline]
    #[must_use]
    pub const fn compute_effective_density_calculus(
        prev_rho: f32,
        integral_rho: f32,
        energy_gate: f32,
        alpha_q7: u8,
        noise_tolerance_q7: u8,
        prev_rising: bool,
    ) -> (f32, bool) {
        let p_rho = branchless_flush_subnormal_f32(branchless_relu_f32(prev_rho));
        let i_rho = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(integral_rho)),
            MAX_STIGMERGIC_DENSITY,
        );
        let e_gate = branchless_clamp_f32(energy_gate, 0.0, 1.0);

        let raw_delta = branchless_relu_f32(i_rho - p_rho);
        let deadband = (noise_tolerance_q7 as f32) * (1.0_f32 / 256.0_f32);
        let exceeds_deadband = raw_delta > deadband;
        let strong_rise = raw_delta > (deadband * 2.0_f32);
        // Guard against single-step quantized noise spikes: require either 2 consecutive rising
        // steps above deadband or a strong rise exceeding 2x deadband.
        let confirmed_rise = exceeds_deadband & (prev_rising | strong_rise);

        let p_norm = branchless_clamp_f32(i_rho * 0.25_f32 + e_gate * 0.5_f32, 0.0, 1.0);
        let p_q7 = (p_norm * 127.0_f32) as u8;
        let gate_q7 = StigmergicGranuleField::boundary_sensitivity_gate_q7(p_q7);
        let gate_f = (gate_q7 as f32) * (1.0_f32 / 128.0_f32);

        let net_delta = branchless_relu_f32(raw_delta - deadband);
        let deriv_scaled = branchless_min_f32(net_delta * 8.0_f32 * gate_f, MAX_STIGMERGIC_DENSITY);
        let deriv_term = branchless_select_f32(confirmed_rise, deriv_scaled, 0.0_f32);

        let clamped_alpha = if alpha_q7 > 127 { 127 } else { alpha_q7 };
        let alpha_f = (clamped_alpha as f32) * (1.0_f32 / 128.0_f32);
        let one_minus_alpha_f = ((128 - clamped_alpha) as f32) * (1.0_f32 / 128.0_f32);
        let eff = branchless_min_f32(
            branchless_flush_subnormal_f32(one_minus_alpha_f * i_rho + alpha_f * deriv_term),
            MAX_STIGMERGIC_DENSITY,
        );
        (eff, exceeds_deadband)
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
            (mode_flags & ACTION_MODE_PHASE_STEER) != 0,
            TRAIL_FLAG_PHASE_STEER_ACTIVE,
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
        let min_step_q7 = branchless_rad_to_phase_step_u8(qdps_min_rad);

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
                    swar_q7_phase_step_filter_words_64(steer_words, min_step_q7)
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
                    swar_q7_phase_step_filter_words_64(steer_words, min_step_q7)
                };
                if swar_words_nonzero_64(&swar_w) {
                    if let Some(q) = query_tensor {
                        let is_default_ripa = (ripa_clamp.to_bits()
                            == DEFAULT_PHASE_CLAMP_MAX_RAD.to_bits())
                            & (num_heads > 0)
                            & (num_heads <= MAX_ATTENTION_HEADS);
                        if is_default_ripa {
                            let rot_dim = phase_clamp_upper_rot_dim(head_dim);
                            if let Some(k) = key_tensor {
                                crate::foundation::rope::rotate_fused_self_phase_clamp_subhead_q_and_clamp_k_default_words(
                                    q.as_mut_slice(),
                                    k.as_mut_slice(),
                                    &swar_w,
                                    num_heads,
                                    head_dim,
                                    rot_dim,
                                );
                            } else {
                                crate::foundation::rope::rotate_fused_self_phase_clamp_and_subhead_default_words(
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
                            dispatch_fused_phase_clamp_and_subhead_words_core(
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
        let qdps_min_rad = phase_steering.phase_step_min_rad();
        let ripa_clamp = phase_steering.phase_clamp_max_rad();
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

/// Concrete zero-allocation (`#[repr(C, align(128))]`, 2,816-byte [`Pod`]) Sidecar Pointer Table
/// and **Stigmergic Threshold Action-Selecting Orchestrator** implementing [`ActionHub`].
///
/// # Memory Layout (2,816 Bytes = `22 x 128B` Cache-Line Pairs, Zero Implicit Padding)
/// - `0..2048` (`2048B`): `trails: [gcso_pointer_trail_t; 16]` (16 inline 128B pointer trails)
/// - `2048..2560` (`512B`): `slot_phase_bank: AlignedBuffer64<gcso_q7_t, 512>` (8 working lanes $\times$ 64 heads)
/// - `2560..2624` (`64B`): `ledger_slot: gcso_ledger_slot_t` (FastPathLedger fast-path shortcuts & telemetry ring)
/// - `2624..2688` (`64B`): `bloom_filter: ContrastivePhaseBloomFilter` (128-bucket 4-bit Bloom filter)
/// - `2688..2816` (`128B`): `thresholds` (`32B`) + `slot_weights` (`32B`) + `phase_steering` (`16B`) +
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
    /// Embedded 16-byte phase-steering kernel configuration (`head_dim`, `num_heads`, `ripa_clamp_max_rad`, `qdps_min_step_rad`).
    pub phase_steering: PhaseSteeringEngine,
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
    /// Most recently observed `EntropyController` moving Z-score entropy $\tilde{H} \in [0, \ln 2]$.
    pub last_entropy_h_tilde: f32,
    /// Most recently observed causal anomaly severity $\in [0.0, 1.0]$.
    pub last_anomaly_severity: f32,
    /// Minimum Bloom filter counter density (`1..=15`) triggering anti-phase repulsion.
    pub repulsion_threshold: u8,
    /// Q7 unit scale (`0..=128`, where `128 == 1.0`) applied during phase-conjugate repulsion.
    pub repulsion_scale_u7: u8,
    /// 8-bit bitmask of enabled anonymous Workbench Lanes (`0x00` = all 8 lanes free by default).
    pub slot_enabled_mask: u8,
    /// Most recently selected [`StigmergicActionDecision`] discriminant (`0..=6`).
    pub last_decision: u8,
    /// 8-bit bitmask of explicitly pinned Workbench Lanes exempt from automatic release or eviction.
    pub slot_pinned_mask: u8,
    /// 8-bit bitmask of Workbench Lanes carrying phase-conjugate anti-hallucination repulsion profiles.
    pub slot_repulsion_mask: u8,
    /// 8-bit bitmask of Workbench Lanes acting as isolated exterior-only hypothesis sandboxes.
    pub slot_sandbox_mask: u8,
    /// Explicit 1-byte padding preserving exact 2,816-byte (`22 x 128B`) `Pod` layout.
    pub reserved_align128: u8,
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
        let phase_steering = match PhaseSteeringEngine::new(
            DEFAULT_PHASE_STEER_HEAD_DIM,
            DEFAULT_PHASE_STEER_NUM_HEADS,
        ) {
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
            slot_weights: [0.0; MAX_ATTRACTOR_DOMAINS],
            phase_steering,
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
            slot_enabled_mask: 0x00,
            last_decision: StigmergicActionDecision::PassiveSidecar as u8,
            slot_pinned_mask: 0x00,
            slot_repulsion_mask: 0x00,
            slot_sandbox_mask: 0x00,
            reserved_align128: 0,
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
        let phase_steering = PhaseSteeringEngine::new(head_dim, num_heads)?;
        let mut hub = Self::new();
        hub.phase_steering = phase_steering;
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
        let phase_steering = PhaseSteeringEngine::from_config(config)?;
        let mut hub = Self::new();
        hub.phase_steering = phase_steering;
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

    /// Allocates an anonymous physical workbench lane (`0..=7`) according to `spec.placement`,
    /// populating `slot_phase_bank`, `slot_weights`, and the 4 lane attribute bitmasks in $\mathcal{O}(1)$
    /// zero-alloc time.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `spec.weight` is non-finite or an explicit lane
    /// index `>= MAX_ATTRACTOR_DOMAINS`, or [`GCSO_ERROR_ACTION_HUB_FULL`] if `AutoFree` is
    /// requested while all 8 lanes are pinned.
    #[inline]
    pub fn allocate_workbench_lane(&mut self, spec: &WorkbenchSlotSpec) -> GcsoResult<u8> {
        if !branchless_is_finite_f32(spec.weight) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let lane = match spec.placement {
            WorkbenchLanePlacement::AutoFree => {
                let (active_cut, free_cut) = self.workbench_lifecycle_quadrants();
                if free_cut.outside != 0 {
                    free_cut.outside.trailing_zeros() as u8
                } else {
                    let evictable = active_cut.outside;
                    if evictable == 0 {
                        return Err(GCSO_ERROR_ACTION_HUB_FULL);
                    }
                    let mut clean_weights = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
                    for (dst, &w) in clean_weights.iter_mut().zip(self.slot_weights.iter()) {
                        *dst = branchless_relu_f32(w);
                    }
                    let (best_lane, _) = branchless_argmin_f32_x8_masked(&clean_weights, evictable);
                    if usize::from(best_lane) >= MAX_ATTRACTOR_DOMAINS {
                        return Err(GCSO_ERROR_ACTION_HUB_FULL);
                    }
                    best_lane
                }
            }
            WorkbenchLanePlacement::Explicit { lane, .. } => {
                if usize::from(lane) >= MAX_ATTRACTOR_DOMAINS {
                    return Err(GCSO_ERROR_INVALID_ARGUMENT);
                }
                lane
            }
        };

        let lane_idx = usize::from(lane);
        let lane_bit = 1u8 << lane;
        let start = lane_idx * MAX_ATTENTION_HEADS;
        let end = start + MAX_ATTENTION_HEADS;
        self.slot_phase_bank.data[start..end].copy_from_slice(&spec.phases);
        self.slot_weights[lane_idx] = branchless_relu_f32(spec.weight);
        self.slot_enabled_mask |= lane_bit;

        let pin_bit = u8::from(spec.is_pinned()).wrapping_neg() & lane_bit;
        let rep_bit = u8::from(spec.is_repulsion()).wrapping_neg() & lane_bit;
        let sb_bit = u8::from(spec.is_sandbox()).wrapping_neg() & lane_bit;
        self.slot_pinned_mask = (self.slot_pinned_mask & !lane_bit) | pin_bit;
        self.slot_repulsion_mask = (self.slot_repulsion_mask & !lane_bit) | rep_bit;
        self.slot_sandbox_mask = (self.slot_sandbox_mask & !lane_bit) | sb_bit;
        Ok(lane)
    }

    /// Releases physical workbench `lane` (`0..=7`), zeroing its phase row and clearing its bits
    /// across `slot_enabled_mask`, `slot_pinned_mask`, `slot_repulsion_mask`, and `slot_sandbox_mask`.
    ///
    /// Returns `Ok(true)` if the lane had been enabled prior to release.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `lane >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn release_workbench_lane(&mut self, lane: u8) -> GcsoResult<bool> {
        let lane_idx = usize::from(lane);
        if lane_idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let lane_bit = 1u8 << lane;
        let was_active = (self.slot_enabled_mask & lane_bit) != 0;
        let start = lane_idx * MAX_ATTENTION_HEADS;
        let end = start + MAX_ATTENTION_HEADS;
        self.slot_phase_bank.data[start..end].fill(0);
        self.slot_weights[lane_idx] = 0.0;
        self.slot_enabled_mask &= !lane_bit;
        self.slot_pinned_mask &= !lane_bit;
        self.slot_repulsion_mask &= !lane_bit;
        self.slot_sandbox_mask &= !lane_bit;
        Ok(was_active)
    }

    /// Releases all unpinned active workbench lanes while leaving pinned lanes untouched, returning
    /// the count of lanes cleared.
    #[inline]
    pub fn release_unpinned_lanes(&mut self) -> u8 {
        let mut to_clear = self.slot_enabled_mask & !self.slot_pinned_mask;
        let released = to_clear.count_ones() as u8;
        while to_clear != 0 {
            let lane_idx = to_clear.trailing_zeros() as usize;
            to_clear &= to_clear - 1;
            let start = lane_idx * MAX_ATTENTION_HEADS;
            let end = start + MAX_ATTENTION_HEADS;
            self.slot_phase_bank.data[start..end].fill(0);
            self.slot_weights[lane_idx] = 0.0;
        }
        let keep = self.slot_pinned_mask;
        self.slot_enabled_mask &= keep;
        self.slot_repulsion_mask &= keep;
        self.slot_sandbox_mask &= keep;
        released
    }

    /// Releases every workbench lane (including pinned lanes), resetting all 8 lanes to the free
    /// pool and returning the count of previously enabled lanes.
    #[inline]
    pub fn release_all_workbench_lanes(&mut self) -> u8 {
        let released = self.slot_enabled_mask.count_ones() as u8;
        self.slot_phase_bank.data.fill(0);
        self.slot_weights = [0.0; MAX_ATTRACTOR_DOMAINS];
        self.slot_enabled_mask = 0;
        self.slot_pinned_mask = 0;
        self.slot_repulsion_mask = 0;
        self.slot_sandbox_mask = 0;
        released
    }

    /// Decays unpinned active workbench lane weights by `retain_factor` (`[0.0, 1.0]`) and
    /// automatically reclaims any lane whose decayed weight drops below `min_weight_floor` (or
    /// reaches `0.0`). Pinned lanes remain untouched. Returns the number of reclaimed lanes.
    #[inline]
    pub fn decay_and_reclaim_workbench_lanes(
        &mut self,
        retain_factor: f32,
        min_weight_floor: f32,
    ) -> u8 {
        let retain = if branchless_is_finite_f32(retain_factor) {
            branchless_clamp_f32(retain_factor, 0.0, 1.0)
        } else {
            0.0
        };
        let floor = if branchless_is_finite_f32(min_weight_floor) {
            branchless_relu_f32(min_weight_floor)
        } else {
            0.0
        };
        let mut active_unpinned = self.slot_enabled_mask & !self.slot_pinned_mask;
        let mut reclaimed = 0u8;
        while active_unpinned != 0 {
            let lane = active_unpinned.trailing_zeros() as u8;
            active_unpinned &= active_unpinned - 1;
            let lane_idx = usize::from(lane);
            let next_w = branchless_flush_subnormal_f32(
                branchless_relu_f32(self.slot_weights[lane_idx]) * retain,
            );
            if next_w <= 0.0 || next_w < floor {
                let lane_bit = 1u8 << lane;
                let start = lane_idx * MAX_ATTENTION_HEADS;
                let end = start + MAX_ATTENTION_HEADS;
                self.slot_phase_bank.data[start..end].fill(0);
                self.slot_weights[lane_idx] = 0.0;
                self.slot_enabled_mask &= !lane_bit;
                self.slot_repulsion_mask &= !lane_bit;
                self.slot_sandbox_mask &= !lane_bit;
                reclaimed = reclaimed.saturating_add(1);
            } else {
                self.slot_weights[lane_idx] = next_w;
            }
        }
        reclaimed
    }

    /// Sets or clears the explicit pin bit (`slot_pinned_mask`) for physical `lane` (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `lane >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn pin_workbench_lane(&mut self, lane: u8, pin: bool) -> GcsoResult<()> {
        if usize::from(lane) >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let lane_bit = 1u8 << lane;
        let bit_val = u8::from(pin).wrapping_neg() & lane_bit;
        self.slot_pinned_mask = (self.slot_pinned_mask & !lane_bit) | bit_val;
        Ok(())
    }

    /// Promotes an active sandbox workbench lane (`slot_sandbox_mask`) to standard full-aperture
    /// participation in-place without relocating its physical register slot.
    ///
    /// Returns `Ok(true)` if `lane` was an active sandbox lane and is now promoted.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `lane >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn promote_sandbox_lane(&mut self, lane: u8) -> GcsoResult<bool> {
        if usize::from(lane) >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let lane_bit = 1u8 << lane;
        let active_sandbox = ((self.slot_enabled_mask & self.slot_sandbox_mask) & lane_bit) != 0;
        if active_sandbox {
            self.slot_sandbox_mask &= !lane_bit;
        }
        Ok(active_sandbox)
    }

    /// Releases all unpinned active sandbox lanes (`slot_enabled_mask & slot_sandbox_mask & !slot_pinned_mask`),
    /// returning the number of sandbox lanes reclaimed.
    #[inline]
    pub fn release_sandbox_lanes(&mut self) -> u8 {
        let mut targets = self.slot_enabled_mask & self.slot_sandbox_mask & !self.slot_pinned_mask;
        let mut released = 0u8;
        while targets != 0 {
            let lane = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            let lane_idx = usize::from(lane);
            let lane_bit = 1u8 << lane;
            let start = lane_idx * MAX_ATTENTION_HEADS;
            let end = start + MAX_ATTENTION_HEADS;
            self.slot_phase_bank.data[start..end].fill(0);
            self.slot_weights[lane_idx] = 0.0;
            self.slot_enabled_mask &= !lane_bit;
            self.slot_repulsion_mask &= !lane_bit;
            self.slot_sandbox_mask &= !lane_bit;
            released = released.saturating_add(1);
        }
        released
    }

    /// Returns the bitmask of currently enabled lanes (`0..=7`) on the 8-lane working phase bank.
    #[inline(always)]
    #[must_use]
    pub const fn active_workbench_lane_mask(&self) -> u8 {
        self.slot_enabled_mask
    }

    /// Returns the bitmask of explicitly pinned workbench lanes (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn workbench_pinned_mask(&self) -> u8 {
        self.slot_pinned_mask
    }

    /// Returns the bitmask of anti-phase repulsion workbench lanes (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn workbench_repulsion_mask(&self) -> u8 {
        self.slot_repulsion_mask
    }

    /// Returns the bitmask of isolated exterior-only sandbox workbench lanes (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn workbench_sandbox_mask(&self) -> u8 {
        self.slot_sandbox_mask
    }

    /// Returns the 4-quadrant lifecycle partition `(active_cut, free_cut)` over all 8 workbench lanes:
    /// - `active_cut.inside`: enabled & pinned
    /// - `active_cut.outside`: enabled & evictable
    /// - `free_cut.inside`: disabled & pinned (reserved)
    /// - `free_cut.outside`: disabled & unpinned (allocatable free)
    #[inline(always)]
    #[must_use]
    pub const fn workbench_lifecycle_quadrants(&self) -> (BitCut8, BitCut8) {
        BitCut8::split(0xFF, self.slot_enabled_mask).refine(self.slot_pinned_mask)
    }

    /// Returns the 4-quadrant role partition `(rep_cut, attract_cut)` over enabled workbench lanes:
    /// - `rep_cut.outside`: global repulsion (`enabled & repulsion & !sandbox`)
    /// - `rep_cut.inside`: sandbox-scoped exterior repulsion (`enabled & repulsion & sandbox`)
    /// - `attract_cut.outside`: standard attractive (`enabled & !repulsion & !sandbox`)
    /// - `attract_cut.inside`: pure sandbox hypothesis (`enabled & !repulsion & sandbox`)
    #[inline(always)]
    #[must_use]
    pub const fn workbench_role_quadrants(&self) -> (BitCut8, BitCut8) {
        BitCut8::split(self.slot_enabled_mask, self.slot_repulsion_mask)
            .refine(self.slot_sandbox_mask)
    }

    /// Returns the bitmask of workbench lanes currently free for unpinned allocation (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn free_workbench_lane_mask(&self) -> u8 {
        !self.slot_enabled_mask & !self.slot_pinned_mask
    }

    /// Returns the number of currently active workbench lanes (`0..=8`).
    #[inline(always)]
    #[must_use]
    pub const fn workbench_utilization_count(&self) -> u8 {
        self.slot_enabled_mask.count_ones() as u8
    }

    /// Atomically refreshes unpinned lanes on the 8-lane working phase workbench with `specs`,
    /// preserving any explicitly pinned lanes while allocating each [`WorkbenchSlotSpec`] via
    /// [`Self::allocate_workbench_lane`]. Returns the count of lanes loaded.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any spec has non-finite weight or an explicit
    /// lane `>= MAX_ATTRACTOR_DOMAINS`, or [`GCSO_ERROR_ACTION_HUB_FULL`] if an `AutoFree` spec
    /// cannot be satisfied because all 8 lanes are pinned.
    #[inline]
    pub fn apply_workbench_preset_lanes(&mut self, specs: &[WorkbenchSlotSpec]) -> GcsoResult<u8> {
        let mut sim_pinned = self.slot_pinned_mask;
        let mut idx = 0usize;
        while idx < specs.len() {
            let spec = &specs[idx];
            if !branchless_is_finite_f32(spec.weight) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            match spec.placement {
                WorkbenchLanePlacement::Explicit { lane, .. } => {
                    if usize::from(lane) >= MAX_ATTRACTOR_DOMAINS {
                        return Err(GCSO_ERROR_INVALID_ARGUMENT);
                    }
                    let bit = 1u8 << lane;
                    if spec.is_pinned() {
                        sim_pinned |= bit;
                    } else {
                        sim_pinned &= !bit;
                    }
                }
                WorkbenchLanePlacement::AutoFree => {
                    if sim_pinned == 0xFF {
                        return Err(GCSO_ERROR_ACTION_HUB_FULL);
                    }
                    if spec.is_pinned() {
                        sim_pinned |= 1u8 << (!sim_pinned).trailing_zeros();
                    }
                }
            }
            idx += 1;
        }

        let _ = self.release_unpinned_lanes();
        let mut updated = 0u8;
        let mut load_idx = 0usize;
        while load_idx < specs.len() {
            let _lane = self.allocate_workbench_lane(&specs[load_idx])?;
            updated = updated.saturating_add(1);
            load_idx += 1;
        }
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
    pub(crate) fn orchestrate_next_action_step_core(
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
        let qdps_min_rad = self.phase_steering.phase_step_min_rad();
        let ripa_clamp = self.phase_steering.phase_clamp_max_rad();
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

        // Fold coupled working-lane phase profile from `slot_phase_bank[target_slot]` strictly
        // through the 4-quadrant role partition (global repulsion on contradiction, standard
        // attractive otherwise; sandboxed and sandbox-repulsion lanes never leak here):
        let masked_target_slot = target_slot & ATTRACTOR_SLOT_ID_MASK;
        let (rep_cut, attract_cut) = self.workbench_role_quadrants();
        let allowed_mask =
            branchless_select_u8(contradiction_repulsed, rep_cut.outside, attract_cut.outside);
        if ((allowed_mask >> masked_target_slot) & 1) != 0 {
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
            self.phase_steering.head_dim() as usize,
            provenance_bits,
            quorum_mask,
            self.last_entropy_h_tilde,
            von_mises_kappa,
        )
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
        self.phase_steering.head_dim()
    }

    #[inline(always)]
    fn num_heads(&self) -> u32 {
        self.phase_steering.num_heads()
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
        let ripa_max = self.phase_steering.phase_clamp_max_rad();
        let base_ripa =
            branchless_select_f32(ripa_max > 0.0, ripa_max, DEFAULT_PHASE_CLAMP_MAX_RAD);
        let damping = 1.0 - 0.025 * (spike_count as f32);
        config_out.head_dim = self.phase_steering.head_dim();
        config_out.num_heads = self.phase_steering.num_heads();
        config_out.ripa_clamp_max_rad = branchless_clamp_f32(base_ripa * damping, 0.01, 0.25);
        config_out.qdps_min_step_rad = self.phase_steering.phase_step_min_rad();
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
        let _ = self.release_unpinned_lanes();
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
            self.slot_enabled_mask |= 1u8 << slot_idx;
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

    #[inline(always)]
    fn workbench_lane_mask(&self) -> u8 {
        self.slot_enabled_mask
    }

    #[inline(always)]
    fn workbench_repulsion_mask(&self) -> u8 {
        self.slot_repulsion_mask
    }

    #[inline(always)]
    fn workbench_sandbox_mask(&self) -> u8 {
        self.slot_sandbox_mask
    }

    #[inline(always)]
    fn workbench_pinned_mask(&self) -> u8 {
        self.slot_pinned_mask
    }

    #[inline]
    fn workbench_lane_words_and_weight(
        &self,
        lane: u8,
    ) -> Option<([u64; NUM_STIGMERGIC_GRANULES], f32)> {
        let idx = usize::from(lane);
        if idx >= MAX_ATTRACTOR_DOMAINS || ((self.slot_enabled_mask >> idx) & 1) == 0 {
            return None;
        }
        let start = idx * MAX_ATTENTION_HEADS;
        let end = start + MAX_ATTENTION_HEADS;
        let mut arr = [0_i8; MAX_ATTENTION_HEADS];
        arr.copy_from_slice(&self.slot_phase_bank.data[start..end]);
        let words = load_q7_words_64(&arr);
        if swar_words_nonzero_64(&words) {
            Some((words, self.slot_weights[idx]))
        } else {
            None
        }
    }
}

impl ActionHubEngine {
    /// Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `chunk_len == 0`.
    #[inline]
    pub fn step_chunk(&mut self, _mask: &gcso_paged_bitmask_t, chunk_len: u32) -> GcsoResult<()> {
        if chunk_len == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.step_counter = self.step_counter.wrapping_add(u64::from(chunk_len));
        Ok(())
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
    assert!(offset_of!(ActionHubEngine, slot_enabled_mask) == 2810);
    assert!(offset_of!(ActionHubEngine, last_decision) == 2811);
    assert!(offset_of!(ActionHubEngine, slot_pinned_mask) == 2812);
    assert!(offset_of!(ActionHubEngine, slot_repulsion_mask) == 2813);
    assert!(offset_of!(ActionHubEngine, slot_sandbox_mask) == 2814);
    assert!(offset_of!(ActionHubEngine, reserved_align128) == 2815);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::AlignedBuffer32;
    use crate::stigmergy::{ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA};

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
        assert_eq!(flags3, ACTION_MODE_PHASE_STEER | ACTION_MODE_SUBHEAD_ROUTE);

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

        // Multi-hop chain test:
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

    #[test]
    fn test_workbench_free_by_default_allocator_pin_evict_decay_and_sandbox() {
        let mut hub = ActionHubEngine::new();
        assert_eq!(hub.active_workbench_lane_mask(), 0x00);
        assert_eq!(hub.workbench_pinned_mask(), 0x00);
        assert_eq!(hub.workbench_repulsion_mask(), 0x00);
        assert_eq!(hub.workbench_sandbox_mask(), 0x00);
        assert_eq!(hub.free_workbench_lane_mask(), 0xFF);
        assert_eq!(hub.workbench_utilization_count(), 0);
        assert_eq!(hub.slot_weights, [0.0; MAX_ATTRACTOR_DOMAINS]);

        // Deterministic AutoFree allocation fills physical lanes 0..=7 in ascending order.
        let mut allocated = [0u8; MAX_ATTRACTOR_DOMAINS];
        for (i, slot) in allocated.iter_mut().enumerate() {
            let weight = 0.25 + (i as f32) * 0.25;
            *slot = hub
                .allocate_workbench_lane(&WorkbenchSlotSpec::auto_free(
                    weight,
                    [(i as i8) + 1; MAX_ATTENTION_HEADS],
                ))
                .unwrap();
            assert_eq!(*slot, i as u8);
        }
        assert_eq!(hub.active_workbench_lane_mask(), 0xFF);
        assert_eq!(hub.workbench_utilization_count(), 8);
        assert_eq!(hub.free_workbench_lane_mask(), 0x00);

        // Pin lowest-weight lane 0: overflow AutoFree skips pinned lane 0 and evicts lane 1 (0.50).
        hub.pin_workbench_lane(0, true).unwrap();
        assert_eq!(hub.workbench_pinned_mask(), 0x01);
        let evicted_lane = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec::auto_repulsion(
                1.25,
                [-12; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        assert_eq!(evicted_lane, 1);
        assert_eq!(hub.workbench_repulsion_mask(), 1 << 1);
        assert_eq!(hub.slot_phase_bank.data[0], 1);
        assert_eq!(hub.slot_phase_bank.data[MAX_ATTENTION_HEADS], -12);

        // Pin all 8 lanes -> AutoFree returns GCSO_ERROR_ACTION_HUB_FULL without mutating.
        for lane in 0..MAX_ATTRACTOR_DOMAINS as u8 {
            hub.pin_workbench_lane(lane, true).unwrap();
        }
        assert_eq!(
            hub.allocate_workbench_lane(&WorkbenchSlotSpec::auto_free(
                1.0,
                [4; MAX_ATTENTION_HEADS]
            )),
            Err(GCSO_ERROR_ACTION_HUB_FULL)
        );

        // Unpin lanes 1..=7, keep lane 0 pinned, then release_unpinned_lanes keeps only lane 0.
        for lane in 1..MAX_ATTRACTOR_DOMAINS as u8 {
            hub.pin_workbench_lane(lane, false).unwrap();
        }
        let released = hub.release_unpinned_lanes();
        assert_eq!(released, 7);
        assert_eq!(hub.active_workbench_lane_mask(), 0x01);
        assert_eq!(hub.workbench_pinned_mask(), 0x01);
        assert_eq!(hub.workbench_repulsion_mask(), 0x00);
        assert_eq!(hub.slot_phase_bank.data[MAX_ATTENTION_HEADS], 0);

        // Allocate sandbox & standard unpinned lanes, verify decay_and_reclaim preserves pinned lane 0.
        let std_lane = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec::auto_free(
                0.80,
                [16; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        let sb_lane = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec::auto_sandbox(
                0.20,
                [32; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        assert_eq!(std_lane, 1);
        assert_eq!(sb_lane, 2);
        assert_eq!(hub.workbench_sandbox_mask(), 1 << 2);

        let reclaimed = hub.decay_and_reclaim_workbench_lanes(0.50, 0.25);
        assert_eq!(reclaimed, 1);
        assert_eq!(hub.active_workbench_lane_mask(), (1 << 0) | (1 << 1));
        assert_eq!(hub.workbench_sandbox_mask(), 0);
        assert!((hub.slot_weights[0] - 0.25).abs() < 1e-6);
        assert!((hub.slot_weights[1] - 0.40).abs() < 1e-6);

        // Allocate sandbox lane again -> promote_sandbox_lane clears sandbox bit in-place;
        // release_sandbox_lanes prunes unpinned sandbox lanes.
        let sb2 = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec::auto_sandbox(
                0.60,
                [24; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        assert_eq!(sb2, 2);
        assert!(hub.promote_sandbox_lane(sb2).unwrap());
        assert_eq!(hub.workbench_sandbox_mask(), 0);
        assert_ne!(hub.active_workbench_lane_mask() & (1 << sb2), 0);

        let sb3 = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec::auto_sandbox(
                0.60,
                [28; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        assert_eq!(sb3, 3);
        assert_eq!(hub.release_sandbox_lanes(), 1);
        assert_eq!(hub.active_workbench_lane_mask() & (1 << sb3), 0);
    }

    #[test]
    fn test_stigmergic_calculus_convex_combination_and_boundary_gate() {
        use StigmergicGranuleField;

        // 1. 4p(1-p) boundary sensitivity gate vanishes at ignorance (0) and saturation (127), peaks at 64.
        assert_eq!(StigmergicGranuleField::boundary_sensitivity_gate_q7(0), 0);
        assert_eq!(StigmergicGranuleField::boundary_sensitivity_gate_q7(127), 0);
        assert!(StigmergicGranuleField::boundary_sensitivity_gate_q7(64) >= 120);

        // 2. alpha = 0 reduces identically to pure integral density.
        let (eff_zero, _) =
            StigmergicThresholds::compute_effective_density_calculus(0.10, 0.30, 0.80, 0, 4, true);
        assert!((eff_zero - 0.30).abs() < 1e-6);

        // 3. Single-step small noise just above deadband (0.015625) without prev_rising is suppressed.
        let (eff_noise, rising_flag) = StigmergicThresholds::compute_effective_density_calculus(
            0.10, 0.12, 0.80, 64, 4, false,
        );
        assert!(rising_flag);
        assert!(eff_noise < 0.12);

        // 4. Two consecutive rising steps (or strong rising edge) boost effective density into Express range.
        let (eff_boost, _) =
            StigmergicThresholds::compute_effective_density_calculus(0.10, 0.30, 0.80, 64, 4, true);
        assert!(eff_boost > 0.45);
        assert!(eff_boost <= MAX_STIGMERGIC_DENSITY);

        // 5. Saturated steady state (delta == 0) is pulled down by (1 - alpha) to relieve PassiveSaturation.
        let (eff_sat, rising_sat) =
            StigmergicThresholds::compute_effective_density_calculus(2.00, 2.00, 0.90, 64, 4, true);
        assert!(!rising_sat);
        assert!(eff_sat < 1.10);
    }

    #[test]
    fn test_action_hub_bitcut8_quadrants_layout_and_eviction() {
        assert_eq!(size_of::<ActionHubEngine>(), 2816);
        assert_eq!(align_of::<ActionHubEngine>(), 128);

        let mut hub = ActionHubEngine::new();
        // Lane 0: standard attractive (unpinned)
        let l0 = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec::auto_free(
                1.0,
                [12; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        // Lane 1: pinned attractive
        let l1 = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec::explicit(
                1,
                true,
                0,
                1.5,
                [24; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        // Lane 2: global repulsion
        let l2 = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec::auto_repulsion(
                0.8,
                [-16; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        // Lane 3: pure sandbox hypothesis
        let l3 = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec::auto_sandbox(
                0.6,
                [32; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        // Lane 4: REPULSION | SANDBOX | PINNED
        let l4 = hub
            .allocate_workbench_lane(&WorkbenchSlotSpec {
                placement: WorkbenchLanePlacement::Explicit { lane: 4, pin: true },
                weight: 0.9,
                flags: WORKBENCH_LANE_FLAG_PINNED
                    | WORKBENCH_LANE_FLAG_REPULSION
                    | WORKBENCH_LANE_FLAG_SANDBOX,
                phases: [-32; MAX_ATTENTION_HEADS],
            })
            .unwrap();

        let (active_cut, free_cut) = hub.workbench_lifecycle_quadrants();
        assert_eq!(active_cut.inside, (1 << l1) | (1 << l4));
        assert_eq!(active_cut.outside, (1 << l0) | (1 << l2) | (1 << l3));
        assert_eq!(free_cut.inside, 0);
        assert_eq!(
            active_cut.inside | active_cut.outside | free_cut.inside | free_cut.outside,
            0xFF
        );

        let (rep_cut, attract_cut) = hub.workbench_role_quadrants();
        assert_eq!(rep_cut.outside, 1 << l2);
        assert_eq!(rep_cut.inside, 1 << l4);
        assert_eq!(attract_cut.outside, (1 << l0) | (1 << l1));
        assert_eq!(attract_cut.inside, 1 << l3);
        assert_eq!(
            rep_cut.outside | rep_cut.inside | attract_cut.outside | attract_cut.inside,
            hub.active_workbench_lane_mask()
        );
    }
}
