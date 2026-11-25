// name: src/core/src/stigmergy/substrate/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Stigmergic Threshold Action-Selecting Orchestrator (`StigmergicSubstrate`) and Sidecar
//! Pointer Table implementation.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::mem::{align_of, offset_of, size_of};

use crate::foundation::{
    branchless_flush_subnormal_f32, branchless_max_f32, branchless_relu_f32, AlignedBuffer64,
    AlignedSlice16, Pod, TaggedPtr, ACTION_MODE_PHASE_STEER, ACTION_MODE_SUBHEAD_ROUTE,
    ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP,
    ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE, DEFAULT_PHASE_STEER_HEAD_DIM,
    DEFAULT_PHASE_STEER_NUM_HEADS,
};
use crate::liminal::{
    branchless_select_usize, LmnkResult, PhaseQ7, RuntimeConfig, LMNK_ERR_INVALID_ARGUMENT,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
};
use crate::stigmergy::gate::PhaseSteeringEngine;
use crate::stigmergy::ledger::{ContrastivePhaseBloomFilter, LedgerSlot};
use crate::stigmergy::trail::PointerTrail;
use crate::stigmergy::{
    DEFAULT_SLOT_COUPLING_MATRIX_8X8, DEFAULT_SUBSTRATE_CAPACITY, WORKBENCH_LANE_FLAG_PINNED,
    WORKBENCH_LANE_FLAG_REPULSION, WORKBENCH_LANE_FLAG_SANDBOX,
};

mod cascade;
mod lanes;
mod thresholds;

pub use thresholds::StigmergicThresholds;

/// Number of inline 128-byte [`PointerTrail`] cells stored inside [`StigmergicSubstrate`] (`16` = 2,048 bytes).
pub const SUBSTRATE_INLINE_TRAILS: usize = 16;

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

/// Default stigmergic trail density threshold ($\tau_{\mathrm{subhead}} = 0.85$) triggering [`StigmergicActionDecision::SubHeadRoute`].
pub const DEFAULT_STIGMERGIC_SUBHEAD_ROUTE_THRESH: f32 = 0.85;

/// Default stigmergic trail density threshold ($\tau_{\mathrm{rank1}} = 1.50$) triggering [`StigmergicActionDecision::FullOrchestration`].
pub const DEFAULT_STIGMERGIC_RANK1_INJECT_THRESH: f32 = 1.50;

/// Default stigmergic trail density threshold ($\tau_{\mathrm{anchor}} = 2.20$) triggering [`StigmergicActionDecision::CrystallizeMacroBasin`].
pub const DEFAULT_STIGMERGIC_CRYSTALLIZE_THRESH: f32 = 2.20;

/// Default directional contradiction threshold ($\tau_{\mathrm{contradict}} = 0.25$) where $\mathcal{C}_{\mathrm{phase}} < -\tau_{\mathrm{contradict}}$ triggers [`StigmergicActionDecision::RepulsionEscape`].
pub const DEFAULT_STIGMERGIC_CONTRADICTION_THRESH: f32 = 0.25;

/// Total Q7 phase elements across all 8 Attractor Working Lanes (`8 * 64 = 512` bytes).
const SUBSTRATE_SLOT_BANK_LEN: usize = MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS;

// ===================================================================
// 1. Stigmergic Action Decision Enum & Threshold Ladder (`Pod`)
// ===================================================================

/// Discrete next-action execution regime selected branchlessly by [`StigmergicThresholds`]
/// inside [`StigmergicSubstrate`].
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
    /// $\tau_{\mathrm{steer}} \le \rho < \tau_{\mathrm{subhead}}$.
    ExpressPhaseSteer = 2,
    /// $SO(2)$-fused `PhaseClamp` + sub-head router (`Fact / Logic / Explore`) Sub-Head Pseudo-MoE routing (`0x09`):
    /// $\tau_{\mathrm{subhead}} \le \rho < \tau_{\mathrm{rank1}}$ or high bifurcation entropy.
    SubHeadRoute = 3,
    /// Full $SO(2)$-fused `PhaseClamp` + sub-head router + Dynamic Rank-1 residual injection
    /// (`ACTION_PRESET_FULL_ORCHESTRATION = 0x19`): $\tau_{\mathrm{rank1}} \le \rho < \tau_{\mathrm{anchor}}$.
    FullOrchestration = 4,
    /// Super-saturated coherent basin promotion (`ACTION_PRESET_FULL_ORCHESTRATION` +
    /// `TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`): $\rho \ge \tau_{\mathrm{anchor}}$ and
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

/// Preset regime profile guiding lane budgets and attribute policies for [`StigmergicSubstrate`]'s
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
/// physical lanes (`0..=7`) inside [`StigmergicSubstrate::slot_phase_bank`].
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

/// Zero-alloc descriptor for loading a phase profile onto [`StigmergicSubstrate`]'s anonymous 8-lane
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
    pub phases: [PhaseQ7; MAX_ATTENTION_HEADS],
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
    pub const fn auto_free(weight: f32, phases: [PhaseQ7; MAX_ATTENTION_HEADS]) -> Self {
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
    pub const fn auto_repulsion(weight: f32, phases: [PhaseQ7; MAX_ATTENTION_HEADS]) -> Self {
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
    pub const fn auto_sandbox(weight: f32, phases: [PhaseQ7; MAX_ATTENTION_HEADS]) -> Self {
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
        phases: [PhaseQ7; MAX_ATTENTION_HEADS],
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

/// Outcome emitted by `StigmergicSubstrate::step_sandbox_hypothesis_lifecycle`.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxLifecycleEvent {
    /// No sandbox state transition occurred on this step.
    Idle = 0,
    /// A new isolated sandbox lane was leased from the free pool.
    Nucleated(u8) = 1,
    /// An active sandbox lane settled in-place into standard full-aperture participation.
    Settled(u8) = 2,
    /// Unpinned active sandbox lanes were released immediately due to repulsion or overdrive.
    Released(u8) = 3,
}

/// 32-byte (`#[repr(C, align(32))]`) [`Pod`] report summarizing the outcome of a single
/// stigmergic threshold action-selection step in [`StigmergicSubstrate`].
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SubstrateStepReport {
    /// Evolved 64-bit tagged pointer (`TaggedPtr::raw()`) carrying the selected Action Mode and Target Slot.
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

// SAFETY: `SubstrateStepReport` is a 32-byte `#[repr(C, align(32))]` struct of primitive fields with zero implicit padding.
unsafe impl Pod for SubstrateStepReport {}

impl SubstrateStepReport {
    /// Returns the typed [`StigmergicActionDecision`] enum for `self.decision`.
    #[inline(always)]
    #[must_use]
    pub const fn action_decision(&self) -> StigmergicActionDecision {
        StigmergicActionDecision::from_u8(self.decision)
    }

    /// Returns `self.evolved_ptr` wrapped as a [`TaggedPtr`].
    #[inline(always)]
    #[must_use]
    pub const fn tagged_ptr(&self) -> TaggedPtr {
        TaggedPtr(self.evolved_ptr)
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

/// Concrete zero-allocation (`#[repr(C, align(64))]`, 2,816-byte [`Pod`]) micro passive carrier
/// implementing [`crate::stigmergy::StigmergicSubstrateOps`]. Accumulates pointer trails, pheromones, and rejection
/// traces whose grounded clusters cross the `liminal` boundary to form [`crate::stigmergy::anchor::PhaseAnchorRecord`] centroids.
///
/// # Memory Layout (2,816 Bytes = `44 x 64B` Cache Lines, Header-First, Zero Implicit Padding)
/// - `CL0 (0..64)`: `thresholds` (`32B`) + `slot_weights` (`32B`)
/// - `CL1 (64..128)`: control, dimension (`head_dim`, `num_heads`), & mask header (`64B`)
/// - `CL2 (128..192)`: `ledger_slot: LedgerSlot` (`64B`)
/// - `CL3 (192..256)`: `bloom_filter: ContrastivePhaseBloomFilter` (`64B`)
/// - `CL4..11 (256..768)`: `slot_phase_bank: AlignedBuffer64<PhaseQ7, 512>` (`512B`)
/// - `CL12..43 (768..2816)`: `trails: [PointerTrail; 16]` (`2,048B`)
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StigmergicSubstrate {
    /// Stigmergic threshold regime governing pheromone dynamics and next-action selection (`offset 0..32`).
    pub thresholds: StigmergicThresholds,
    /// Active gravitational weights for the 8 Attractor Working Lanes (`0..=7`, `offset 32..64`).
    pub slot_weights: [f32; MAX_ATTRACTOR_DOMAINS],
    /// 64-bit ($8 \times 8$) cross-lane coupling bitmatrix (`offset 64..72`).
    pub coupling_matrix_8x8: u64,
    /// Total pointer transitions executed on this substrate (`offset 72..80`).
    pub step_counter: u64,
    /// Configured logical trail slot capacity (`>= 1`, `offset 80..84`).
    pub capacity: u32,
    /// Number of active trail slots currently populated (`0..=capacity`, `offset 84..88`).
    pub active_count: u32,
    /// Circular write cursor (`0..SUBSTRATE_INLINE_TRAILS`) into `self.trails` (`offset 88..92`).
    pub write_cursor: u32,
    /// Most recently selected candidate trail ID (`0..SUBSTRATE_INLINE_TRAILS`, `offset 92..96`).
    pub last_selected_trail_id: u32,
    /// Most recently observed `EntropyController` moving Z-score entropy $\tilde{H} \in [0, \ln 2]$ (`offset 96..100`).
    pub last_entropy_h_tilde: f32,
    /// Most recently observed causal anomaly severity $\in [0.0, 1.0]$ (`offset 100..104`).
    pub last_anomaly_severity: f32,
    /// Configured Transformer head dimension (`offset 104..106`).
    pub head_dim: u16,
    /// Configured attention head count (`offset 106..108`).
    pub num_heads: u16,
    /// Minimum Bloom filter counter density (`1..=15`) triggering anti-phase repulsion (`offset 108`).
    pub repulsion_threshold: u8,
    /// Q7 unit scale (`0..=128`, where `128 == 1.0`) applied during phase-conjugate repulsion (`offset 109`).
    pub repulsion_scale_u7: u8,
    /// 8-bit bitmask of enabled anonymous Workbench Lanes (`0x00` = all 8 lanes free by default, `offset 110`).
    pub slot_enabled_mask: u8,
    /// Most recently selected [`StigmergicActionDecision`] discriminant (`0..=6`, `offset 111`).
    pub last_decision: u8,
    /// 8-bit bitmask of explicitly pinned Workbench Lanes exempt from automatic release or eviction (`offset 112`).
    pub slot_pinned_mask: u8,
    /// 8-bit bitmask of Workbench Lanes carrying phase-conjugate anti-hallucination repulsion profiles (`offset 113`).
    pub slot_repulsion_mask: u8,
    /// 8-bit bitmask of Workbench Lanes acting as isolated exterior-only hypothesis sandboxes (`offset 114`).
    pub slot_sandbox_mask: u8,
    /// Explicit header padding aligning `ledger_slot` to exact CL2 boundary (`offset 115..128`).
    pub reserved_header: [u8; 13],
    /// Dynamic Adaptive Extension Scratchpad (`offset 128..192`, 1 cache line).
    pub ledger_slot: LedgerSlot,
    /// Contrastive Phase Bloom Filter for $\mathcal{O}(1)$ anti-phase repulsion (`offset 192..256`, 1 cache line).
    pub bloom_filter: ContrastivePhaseBloomFilter,
    /// 8-lane working phase cache window (`offset 256..768`, 8 cache lines).
    pub slot_phase_bank: AlignedBuffer64<PhaseQ7, SUBSTRATE_SLOT_BANK_LEN>,
    /// Inline L1D-resident Sidecar Pointer Table (`offset 768..2816`, `16 x 128B = 2,048B`).
    pub trails: [PointerTrail; SUBSTRATE_INLINE_TRAILS],
}

// SAFETY: `StigmergicSubstrate` is a `#[repr(C, align(64))]` composition of `Pod` structs and primitive
// arrays totaling exactly 2,816 bytes with zero implicit padding.
unsafe impl Pod for StigmergicSubstrate {}

const _: () = {
    assert!(core::mem::size_of::<StigmergicSubstrate>() == 2816);
    assert!(core::mem::align_of::<StigmergicSubstrate>() == 64);
    assert!(core::mem::offset_of!(StigmergicSubstrate, coupling_matrix_8x8) == 64);
    assert!(core::mem::offset_of!(StigmergicSubstrate, step_counter) == 72);
    assert!(core::mem::offset_of!(StigmergicSubstrate, capacity) == 80);
    assert!(core::mem::offset_of!(StigmergicSubstrate, head_dim) == 104);
    assert!(core::mem::offset_of!(StigmergicSubstrate, num_heads) == 106);
    assert!(core::mem::offset_of!(StigmergicSubstrate, reserved_header) == 115);
    assert!(core::mem::offset_of!(StigmergicSubstrate, ledger_slot) == 128);
    assert!(core::mem::offset_of!(StigmergicSubstrate, trails) == 768);
};

impl Default for StigmergicSubstrate {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl StigmergicSubstrate {
    /// Creates a new default [`StigmergicSubstrate`] with `head_dim = 128`, `num_heads = 32`,
    /// `capacity = DEFAULT_SUBSTRATE_CAPACITY`, and canonical [`StigmergicThresholds`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            thresholds: StigmergicThresholds::new(),
            slot_weights: [0.0; MAX_ATTRACTOR_DOMAINS],
            coupling_matrix_8x8: DEFAULT_SLOT_COUPLING_MATRIX_8X8,
            step_counter: 0,
            capacity: DEFAULT_SUBSTRATE_CAPACITY,
            active_count: 0,
            write_cursor: 0,
            last_selected_trail_id: 0,
            last_entropy_h_tilde: 0.0,
            last_anomaly_severity: 0.0,
            head_dim: DEFAULT_PHASE_STEER_HEAD_DIM as u16,
            num_heads: DEFAULT_PHASE_STEER_NUM_HEADS as u16,
            repulsion_threshold: 2,
            repulsion_scale_u7: 128,
            slot_enabled_mask: 0x00,
            last_decision: StigmergicActionDecision::PassiveSidecar as u8,
            slot_pinned_mask: 0x00,
            slot_repulsion_mask: 0x00,
            slot_sandbox_mask: 0x00,
            reserved_header: [0; 13],
            ledger_slot: LedgerSlot {
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
            slot_phase_bank: AlignedBuffer64::new([0; SUBSTRATE_SLOT_BANK_LEN]),
            trails: [PointerTrail {
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
                reserved_header: [0; 8],
                accumulated_phase_delta: [0; MAX_ATTENTION_HEADS],
            }; SUBSTRATE_INLINE_TRAILS],
        }
    }

    /// Creates a new [`StigmergicSubstrate`] configured with the specified logical `capacity`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `capacity == 0`.
    #[inline]
    pub const fn with_capacity(capacity: u32) -> LmnkResult<Self> {
        if capacity == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let mut substrate = Self::new();
        substrate.capacity = capacity;
        Ok(substrate)
    }

    /// Creates a new [`StigmergicSubstrate`] configured with `(head_dim, num_heads, capacity)` and
    /// custom [`StigmergicThresholds`].
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `(head_dim, num_heads)` or `thresholds` are invalid,
    /// exceed `u16::MAX`, or if `capacity == 0`.
    #[inline]
    pub fn with_dimensions_and_thresholds(
        head_dim: u32,
        num_heads: u32,
        capacity: u32,
        thresholds: StigmergicThresholds,
    ) -> LmnkResult<Self> {
        if capacity == 0 || head_dim > u16::MAX as u32 || num_heads > u16::MAX as u32 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        thresholds.validate()?;
        let _phase_steering = PhaseSteeringEngine::new(head_dim, num_heads)?;
        let mut substrate = Self::new();
        substrate.head_dim = head_dim as u16;
        substrate.num_heads = num_heads as u16;
        substrate.capacity = capacity;
        substrate.thresholds = thresholds;
        Ok(substrate)
    }

    /// Creates a new [`StigmergicSubstrate`] initialized from a raw [`RuntimeConfig`] descriptor.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `config` dimensions or capacities are invalid.
    #[inline]
    pub fn from_config(config: &RuntimeConfig) -> LmnkResult<Self> {
        if config.substrate_capacity == 0
            || config.head_dim > u16::MAX as u32
            || config.num_heads > u16::MAX as u32
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let _phase_steering = PhaseSteeringEngine::from_config(config)?;
        let mut substrate = Self::new();
        substrate.head_dim = config.head_dim as u16;
        substrate.num_heads = config.num_heads as u16;
        substrate.capacity = config.substrate_capacity;
        substrate.ledger_slot.mode = u32::from(config.ledger_mode);
        Ok(substrate)
    }

    /// Initializes a zeroed `dst` allocation of [`StigmergicSubstrate`] in-place without
    /// materializing the full 2,816-byte struct on the stack.
    ///
    /// # Safety
    /// Caller must ensure `dst` points to writable, zero-initialized, properly aligned memory
    /// for one `StigmergicSubstrate` matching the pre-validated config fields.
    #[inline]
    pub unsafe fn init_in_place_on_zeroed(
        dst: *mut Self,
        config: &RuntimeConfig,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        coupling_matrix_8x8: u64,
    ) {
        // SAFETY: Caller guarantees `dst` is valid, writable, aligned, and zero-initialized.
        // Zeroed bytes match empty `PointerTrail`s, zeroed `slot_phase_bank`, zeroed
        // `ContrastivePhaseBloomFilter`, and zeroed `LedgerSlot` fields except `mode`.
        unsafe {
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).thresholds),
                StigmergicThresholds::new(),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).coupling_matrix_8x8),
                coupling_matrix_8x8,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).capacity),
                config.substrate_capacity,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).head_dim),
                config.head_dim as u16,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).num_heads),
                config.num_heads as u16,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).repulsion_threshold),
                repulsion_threshold,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).repulsion_scale_u7),
                repulsion_scale_u7,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).last_decision),
                StigmergicActionDecision::PassiveSidecar as u8,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).ledger_slot.mode),
                u32::from(config.ledger_mode),
            );
        }
    }

    /// Returns the number of populated inline trails in `self.trails` (`0..=SUBSTRATE_INLINE_TRAILS`).
    #[inline(always)]
    #[must_use]
    pub const fn inline_active_len(&self) -> usize {
        branchless_select_usize(
            (self.active_count as usize) < SUBSTRATE_INLINE_TRAILS,
            self.active_count as usize,
            SUBSTRATE_INLINE_TRAILS,
        )
    }

    /// Returns an immutable 16-byte aligned slice view over the 8-lane Q7 working phase bank (`512` bytes).
    #[inline(always)]
    #[must_use]
    pub const fn slot_phase_bank(&self) -> AlignedSlice16<'_, PhaseQ7> {
        self.slot_phase_bank.relax_slice::<16>()
    }
}

// ===================================================================
// 4. Compile-Time Layout & Trait-Bound Verification
// ===================================================================

const _: () = {
    assert!(size_of::<StigmergicActionDecision>() == 1);
    assert!(size_of::<StigmergicThresholds>() == 32 && align_of::<StigmergicThresholds>() == 32);
    assert!(size_of::<SubstrateStepReport>() == 32 && align_of::<SubstrateStepReport>() == 32);
    assert!(offset_of!(SubstrateStepReport, evolved_ptr) == 0);
    assert!(offset_of!(SubstrateStepReport, updated_density) == 8);
    assert!(offset_of!(SubstrateStepReport, phase_coherence) == 12);
    assert!(offset_of!(SubstrateStepReport, transition_log_prob) == 16);
    assert!(offset_of!(SubstrateStepReport, selected_trail_id) == 20);
    assert!(offset_of!(SubstrateStepReport, dominant_anchor_id) == 24);
    assert!(offset_of!(SubstrateStepReport, decision) == 28);
    assert!(offset_of!(SubstrateStepReport, action_mode_flags) == 29);
    assert!(offset_of!(SubstrateStepReport, target_slot) == 30);
    assert!(offset_of!(SubstrateStepReport, bloom_repulsion_density) == 31);

    assert!(size_of::<StigmergicSubstrate>() == 2816 && align_of::<StigmergicSubstrate>() == 64);
    assert!(offset_of!(StigmergicSubstrate, thresholds) == 0);
    assert!(offset_of!(StigmergicSubstrate, slot_weights) == 32);
    assert!(offset_of!(StigmergicSubstrate, coupling_matrix_8x8) == 64);
    assert!(offset_of!(StigmergicSubstrate, head_dim) == 104);
    assert!(offset_of!(StigmergicSubstrate, num_heads) == 106);
    assert!(offset_of!(StigmergicSubstrate, slot_enabled_mask) == 110);
    assert!(offset_of!(StigmergicSubstrate, last_decision) == 111);
    assert!(offset_of!(StigmergicSubstrate, slot_pinned_mask) == 112);
    assert!(offset_of!(StigmergicSubstrate, slot_repulsion_mask) == 113);
    assert!(offset_of!(StigmergicSubstrate, slot_sandbox_mask) == 114);
    assert!(offset_of!(StigmergicSubstrate, reserved_header) == 115);
    assert!(offset_of!(StigmergicSubstrate, ledger_slot) == 128);
    assert!(offset_of!(StigmergicSubstrate, bloom_filter) == 192);
    assert!(offset_of!(StigmergicSubstrate, slot_phase_bank) == 256);
    assert!(offset_of!(StigmergicSubstrate, trails) == 768);
};

#[cfg(test)]
mod tests;
