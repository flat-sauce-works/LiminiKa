// name: src/core/src/stigmergy/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 2 (`Stigmergy`): 8-granule SWAR stigmergic trails, pheromone density laws,
//! 256-bit paged bitmasks, contrastive Bloom filters, fast-path ledgers, 8-lane
//! [`StigmergicSubstrate`], multi-scale cognitive circuits, `.lmnk` crystallization, and macro
//! attractor fields.
//!
//! Depends strictly on `core`, [`crate::liminal`], and [`crate::foundation`].

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

use core::mem::{align_of, size_of};

pub use crate::foundation::swar_q7_circular_add_x8;
pub use crate::liminal::{HEADS_PER_STIGMERGIC_GRANULE, NUM_STIGMERGIC_GRANULES};
use crate::liminal::{MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS};

// ===================================================================
// 1. Semantic Attractor Provenance IDs, Workbench Pool & 8x8 Coupling
// ===================================================================

/// Default logical trail slot capacity of [`StigmergicSubstrate`] (`256`).
pub const DEFAULT_SUBSTRATE_CAPACITY: u32 = 256;

/// Number of fast-path shortcut slots in a `LedgerSlot` (`4`).
pub const FAST_PATH_SHORTCUT_SLOTS: usize = 4;

/// Byte length of the circular mini telemetry ledger in a `LedgerSlot` (`8`).
pub const LEDGER_HISTORY_LEN: usize = 8;

/// Semantic Provenance Label 0: Primary System Baseline & Core Identity Anchor (`0`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_SYSTEM_BASELINE: u8 = 0;

/// Semantic Provenance Label 1: Character Persona & Emotional Style Attractor (`1`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_PERSONA: u8 = 1;

/// Semantic Provenance Label 2: Factual Knowledge & World Lore Anchor (`2`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_KNOWLEDGE: u8 = 2;

/// Semantic Provenance Label 3: Phase-Conjugate Anti-Hallucination Repulsion Field (`3`, $-\boldsymbol{\Delta\theta}$).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_REPULSION: u8 = 3;

/// Semantic Provenance Label 4: Dynamic User Context & Episodic Dialogue Memory (`4`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_USER_CONTEXT: u8 = 4;

/// Semantic Provenance Label 5: Temporal / Visual Co-Watching Horizon Attractor (`5`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_TEMPORAL: u8 = 5;

/// Semantic Provenance Label 6: User Intent / Circadian Drift Attractor (`6`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_USER_INTENT: u8 = 6;

/// Semantic Provenance Label 7: Crystallized Macro-Basin / Dynamic Extension Provenance (`7`).
///
/// Decoupled from physical workbench register lane indices (`0..=7`).
pub const ATTRACTOR_SLOT_CRYSTALLIZED: u8 = 7;

/// Number of anonymous physical lanes in the 8-lane working phase workbench (`8`).
pub const WORKBENCH_LANE_COUNT: usize = MAX_ATTRACTOR_DOMAINS;

/// Sentinel lane selector requesting deterministic automatic allocation from the free workbench pool (`0xFF`).
pub const WORKBENCH_LANE_AUTO: u8 = 0xFF;

/// Workbench lane attribute bit: lane is explicitly pinned and exempt from automatic release or eviction (`bit 0`).
pub const WORKBENCH_LANE_FLAG_PINNED: u8 = 1 << 0;

/// Workbench lane attribute bit: lane holds a phase-conjugate anti-hallucination repulsion profile (`bit 1`).
pub const WORKBENCH_LANE_FLAG_REPULSION: u8 = 1 << 1;

/// Workbench lane attribute bit: lane holds an isolated hypothesis acting only on exterior/discovery granules (`bit 2`).
pub const WORKBENCH_LANE_FLAG_SANDBOX: u8 = 1 << 2;

/// Default $8 \times 8$ Attractor Slot coupling bitmatrix (`u64`) enabling symmetric cross-slot
/// synergy across all 8 parallel attractor domain slots.
pub const DEFAULT_SLOT_COUPLING_MATRIX_8X8: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Computes the bit position (`0..=63`) in an $8 \times 8$ `u64` coupling bitmatrix for `(slot_a, slot_b)`.
#[inline(always)]
#[must_use]
pub const fn slot_coupling_bit_index(slot_a: u32, slot_b: u32) -> u32 {
    ((slot_a & 7) << 3) | (slot_b & 7)
}

/// Branchlessly tests whether `slot_a` and `slot_b` are coupled in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn is_slot_pair_coupled(coupling_matrix_8x8: u64, slot_a: u32, slot_b: u32) -> bool {
    ((coupling_matrix_8x8 >> slot_coupling_bit_index(slot_a, slot_b)) & 1) != 0
}

/// Extracts the 8-bit row mask (`bits 0..=7`) representing all target slots coupled with `slot_a`
/// in $\mathcal{O}(1)$ via a single register shift.
#[inline(always)]
#[must_use]
pub const fn coupled_slot_row_mask(coupling_matrix_8x8: u64, slot_a: u32) -> u8 {
    ((coupling_matrix_8x8 >> ((slot_a & 7) << 3)) & 0xFF) as u8
}

/// Returns the number of target slots (`0..=8`) coupled with `slot_a` in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn coupled_slot_count(coupling_matrix_8x8: u64, slot_a: u32) -> u32 {
    coupled_slot_row_mask(coupling_matrix_8x8, slot_a).count_ones()
}

/// Branchlessly sets or clears the directed coupling bit `(slot_a, slot_b)` in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn with_slot_pair_coupled(
    coupling_matrix_8x8: u64,
    slot_a: u32,
    slot_b: u32,
    enabled: bool,
) -> u64 {
    let bit_mask = 1u64 << slot_coupling_bit_index(slot_a, slot_b);
    let enable_mask = (enabled as u64).wrapping_neg();
    (coupling_matrix_8x8 & !bit_mask) | (bit_mask & enable_mask)
}

/// Branchlessly sets or clears symmetric bidirectional coupling between `slot_a` and `slot_b`
/// (`(slot_a, slot_b)` and `(slot_b, slot_a)`) in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn with_slot_pair_coupled_symmetric(
    coupling_matrix_8x8: u64,
    slot_a: u32,
    slot_b: u32,
    enabled: bool,
) -> u64 {
    let m1 = with_slot_pair_coupled(coupling_matrix_8x8, slot_a, slot_b, enabled);
    with_slot_pair_coupled(m1, slot_b, slot_a, enabled)
}

// ===================================================================
// 1. Telemetry, Trail, Anti-Saturation & Swarm Quorum Constants
// ===================================================================

/// Default Stigmergic Reconstitution Quorum threshold (`6` out of `8` granules = 75% consensus).
pub const DEFAULT_STIGMERGIC_QUORUM_THRESHOLD: u8 = 6;
/// Default Q7 circular phase residual tolerance (`4` Q7 steps $\approx 0.098\text{ rad}$).
pub const DEFAULT_STIGMERGIC_TOLERANCE_Q7: u8 = 4;

/// Default total counter mass threshold (`128`) triggering branchless SWAR halving in [`ContrastivePhaseBloomFilter`].
pub const DEFAULT_BLOOM_MAX_MASS_THRESHOLD: u32 = 128;
/// Default active bucket count ceiling (`48` out of `64` buckets = 75% occupancy).
pub const DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS: u32 = 48;
/// Maximum stigmergic trail density ceiling ($\rho_{\max} = 16.0$).
pub const MAX_STIGMERGIC_DENSITY: f32 = 16.0;
/// Maximum macro attractor pull force ceiling ($F_{\mathrm{pull},\max} = 64.0$).
pub const MAX_ATTRACTOR_PULL_FORCE: f32 = 64.0;

/// Trail flag bit (`0x01`): Inline phase steering is active on this trail.
pub const TRAIL_FLAG_PHASE_STEER_ACTIVE: u32 = 1 << 0;
/// Trail flag bit (`0x02`): Phase-Conjugate Attractor Repulsion ($-\boldsymbol{\Delta\theta}$) was triggered.
pub const TRAIL_FLAG_REPULSION_ACTIVE: u32 = 1 << 1;
/// Trail flag bit (`0x04`): Trail crossed the crystallization density threshold ($\rho_{\mathrm{stigmergy}} \ge \tau_{\mathrm{crystal}}$).
pub const TRAIL_FLAG_CRYSTALLIZED_CANDIDATE: u32 = 1 << 2;
/// Trail flag bit (`0x08`): Non-linear unified-field or cross-slot attractor resonance boosted the trail's phase trajectory.
pub const TRAIL_FLAG_SYNERGY_RESONANCE: u32 = 1 << 3;
/// Trail flag bit (`0x10`): Liminal beat perturbation was injected across competing attractor boundaries.
pub const TRAIL_FLAG_LIMINAL_BEAT_ACTIVE: u32 = 1 << 4;
/// Trail flag bit (`0x20`): Low-entropy inertial layer skipping is active along this pointer chain.
pub const TRAIL_FLAG_INERTIAL_SKIP_ACTIVE: u32 = 1 << 5;
/// Trail flag bit (`0x40`): Subdivided 8-cell Micro-Swarm resolved the Action Hub command on the Hot Path.
pub const TRAIL_FLAG_SWARM_RESOLVED: u32 = 1 << 6;
/// Trail flag bit (`0x80`): Micro-Swarm encountered quorum collapse and escalated the trail.
pub const TRAIL_FLAG_SWARM_ESCALATED: u32 = 1 << 7;
/// Bit shift offset (`8`) embedding the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) inside `PointerTrail::flags`.
pub const TRAIL_SWARM_QUORUM_SHIFT: u32 = 8;
/// Bitmask (`0x0000_FF00`) isolating the 8-bit Micro-Swarm Quorum bitmask inside `PointerTrail::flags`.
pub const TRAIL_SWARM_QUORUM_MASK: u32 = 0x0000_FF00;
/// Trail flag bit (`0x0001_0000`): Previous step observed a positive pheromone rise above the granule noise deadband.
pub const TRAIL_FLAG_PHEROMONE_RISING: u32 = 1 << 16;

/// FastPathLedger telemetry metric bit (`0x01`): Fast-path bypass shortcut hit in `LedgerSlot`.
pub const LEDGER_METRIC_FAST_PATH_HIT: u8 = 1 << 0;
/// FastPathLedger telemetry metric bit (`0x02`): Inline `PhaseSteering`/`PhaseClamp` phase steering applied.
pub const LEDGER_METRIC_PHASE_STEERED: u8 = 1 << 1;
/// FastPathLedger telemetry metric bit (`0x04`): Contrastive Phase Bloom Filter repulsion triggered.
pub const LEDGER_METRIC_REPULSION_APPLIED: u8 = 1 << 2;
/// FastPathLedger telemetry metric bit (`0x08`): Intermediate layer skipped with SWAR phase compensation.
pub const LEDGER_METRIC_LAYER_SKIP_COMPENSATED: u8 = 1 << 3;
/// FastPathLedger telemetry metric bit (`0x10`): sub-head router single-pass sub-head routing executed.
pub const LEDGER_METRIC_SUBHEAD_ROUTED: u8 = 1 << 4;
/// FastPathLedger telemetry metric bit (`0x20`): Rank-1 residual Dynamic Rank-1 residual adapter injected.
pub const LEDGER_METRIC_RANK1_INJECTED: u8 = 1 << 5;
/// FastPathLedger telemetry metric bit (`0x40`): Lazy Phase Unwrapping relative shift applied.
pub const LEDGER_METRIC_LAZY_UNWRAP_APPLIED: u8 = 1 << 6;
/// FastPathLedger telemetry metric bit (`0x80`): Unified-field or cross-slot attractor synergy / mutual inhibition active.
pub const LEDGER_METRIC_SLOT_SYNERGY_ACTIVE: u8 = 1 << 7;
/// `LedgerSlot::auto_tune_flags` bit (`0x01`): `telemetry_mini_ledger` stores chronological
/// [`CognitiveStepOp`] records (`[Scale | Op | Intensity]`) rather than legacy metric masks.
pub const LEDGER_FLAG_COGNITIVE_TRAJECTORY: u32 = 1 << 0;

pub mod anchor;
pub mod circuit;
pub mod gate;
pub mod granule;
pub mod ledger;
pub mod pheromone;
pub mod substrate;
pub mod trail;

pub use self::anchor::*;
pub use self::circuit::*;
pub use self::gate::*;
pub use self::granule::*;
pub use self::ledger::*;
pub use self::pheromone::*;
pub use self::substrate::*;
pub use self::trail::*;

const _: () = {
    assert!(size_of::<PagedBitmask>() == 32);
    assert!(align_of::<PagedBitmask>() == 32);
    assert!(size_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(align_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(size_of::<PointerTrail>() == 128);
    assert!(align_of::<PointerTrail>() == 64);
    assert!(size_of::<OverlapDiscoveryReport>() == 32);
    assert!(align_of::<OverlapDiscoveryReport>() == 16);
    assert!(size_of::<StigmergicSubstrate>() == 2816);
    assert!(align_of::<StigmergicSubstrate>() == 64);
    assert!(size_of::<StigmergicReconstitutionReport>() == 16);
    assert!(align_of::<StigmergicReconstitutionReport>() == 16);
    assert!(NUM_STIGMERGIC_GRANULES * HEADS_PER_STIGMERGIC_GRANULE == MAX_ATTENTION_HEADS);
    assert!(slot_coupling_bit_index(7, 7) == 63);
    assert!(coupled_slot_row_mask(DEFAULT_SLOT_COUPLING_MATRIX_8X8, 3) == 0xFF);
    assert!(coupled_slot_count(DEFAULT_SLOT_COUPLING_MATRIX_8X8, 3) == 8);

    let (next_w, res_q7, ok) =
        swar_subswarm_resolve_x8(0x1010_1010_1010_1010, 0x2020_2020_2020_2020, 128, 1, 2);
    assert!(next_w == 0x2020_2020_2020_2020);
    assert!(res_q7 == 0 && ok);
};

#[cfg(test)]
mod tests;
