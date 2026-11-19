// name: src/core/src/stigmergy/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 2 (`Stigmergy`): 8-granule SWAR stigmergic trails, pheromone density laws,
//! 256-bit paged bitmasks, contrastive Bloom filters, fast-path ledgers, 8-lane [`ActionHubEngine`],
//! multi-scale cognitive circuits, `.gcso` crystallization, and macro attractor fields.
//!
//! Depends strictly on `core`, [`crate::liminal`], and [`crate::foundation`].

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

use core::mem::{align_of, size_of};

pub use crate::foundation::swar_q7_circular_add_x8;
use crate::liminal::MAX_ATTENTION_HEADS;
pub use crate::liminal::{HEADS_PER_STIGMERGIC_GRANULE, NUM_STIGMERGIC_GRANULES};

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

/// Trail flag bit (`0x01`): Inline DPSR phase steering is active on this trail.
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
/// Bit shift offset (`8`) embedding the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) inside `gcso_pointer_trail_t::flags`.
pub const TRAIL_SWARM_QUORUM_SHIFT: u32 = 8;
/// Bitmask (`0x0000_FF00`) isolating the 8-bit Micro-Swarm Quorum bitmask inside `gcso_pointer_trail_t::flags`.
pub const TRAIL_SWARM_QUORUM_MASK: u32 = 0x0000_FF00;
/// Trail flag bit (`0x0001_0000`): Previous step observed a positive pheromone rise above the granule noise deadband.
pub const TRAIL_FLAG_PHEROMONE_RISING: u32 = 1 << 16;

/// FastPathLedger telemetry metric bit (`0x01`): Fast-path bypass shortcut hit in `gcso_ledger_slot_t`.
pub const LEDGER_METRIC_FAST_PATH_HIT: u8 = 1 << 0;
/// FastPathLedger telemetry metric bit (`0x02`): Inline DPSR/RIPA phase steering applied.
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
/// `gcso_ledger_slot_t::auto_tune_flags` bit (`0x01`): `telemetry_mini_ledger` stores chronological
/// [`CognitiveStepOp`] records (`[Scale | Op | Intensity]`) rather than legacy metric masks.
pub const LEDGER_FLAG_COGNITIVE_TRAJECTORY: u32 = 1 << 0;

pub mod action_hub;
pub mod attractor;
pub mod circuit;
pub mod crystal;
pub mod granule;
pub mod ledger;
pub mod pheromone;
pub mod substrate;
pub mod trail;

pub use self::action_hub::*;
pub use self::attractor::*;
pub use self::circuit::*;
pub use self::crystal::*;
pub use self::granule::*;
pub use self::ledger::*;
pub use self::pheromone::*;
pub use self::substrate::*;
pub use self::trail::*;

const _: () = {
    assert!(size_of::<gcso_paged_bitmask_t>() == 32);
    assert!(align_of::<gcso_paged_bitmask_t>() == 32);
    assert!(size_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(align_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(size_of::<gcso_pointer_trail_t>() == 128);
    assert!(align_of::<gcso_pointer_trail_t>() == 128);
    assert!(size_of::<StigmergicReconstitutionReport>() == 16);
    assert!(align_of::<StigmergicReconstitutionReport>() == 16);
    assert!(NUM_STIGMERGIC_GRANULES * HEADS_PER_STIGMERGIC_GRANULE == MAX_ATTENTION_HEADS);

    let (next_w, res_q7, ok) =
        swar_subswarm_resolve_x8(0x1010_1010_1010_1010, 0x2020_2020_2020_2020, 128, 1, 2);
    assert!(next_w == 0x2020_2020_2020_2020);
    assert!(res_q7 == 0 && ok);
};

#[cfg(test)]
mod tests;
