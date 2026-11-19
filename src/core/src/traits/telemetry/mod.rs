// name: src/core/src/traits/telemetry/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.1 Capability 2: Stigmergic trails, internal telemetry, inter-memory Limina bridging,
//! Bayesian pointer-trail transition scoring & rewiring, single-manifold unified-field resonance
//! with 32-bit `target_anchor_id` preservation, and stigmergic granule reconstitution (`PagedBitmaskOps`,
//! `ContrastivePhaseBloomFilter`, `StigmergicActionPtrOps`, `StigmergicTrailOps`,
//! `StigmergicReconstitutionOps`, `FastPathLedger`, and `StigmergicReconstitutionReport`) registered into
//! [`crate::traits::ActionHub`] for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! Core inner-loop reductions execute via branchless SWAR bit-parallel arithmetic (`0 fdiv`
//! in similarity and synergy reductions) with bounded register-resident footprints.

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

// Re-export canonical 8-lane SWAR circular phase addition used by Stigmergic & Swarm modules.
pub use crate::traits::primitives::swar_q7_circular_add_x8;
use crate::traits::branchless::{
    BranchlessKernel, branchless_fast_transition_log_prob, branchless_q7_inhibit_words_64,
    branchless_q7_synergy_fuse_words_64, branchless_q7_unified_field_fold_words_with_flags_64,
    circular_resultant_from_words_64,
    coherence_from_uniform_diff_word_x8, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64, swar_slerp_words_with_bias_64,
    swar_slerp_words_with_uniform_flag_64, swar_words_nonzero_64,
};
use crate::traits::primitives::{
    DEFAULT_PHASE_GRAVITY_EPS, Q7_FULL_COS_REF, SWAR_EVEN_BYTES_64, SWAR_LOW7_MASK_64,
    SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, SWAR_SUM_U16_64, branchless_abs_f32,
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_min_f32, branchless_phase_proximity_gravity_f32, branchless_popcount_rsqrt_u32,
    branchless_relu_f32, branchless_rsqrt_f32, branchless_select_f32, branchless_select_u32,
    branchless_select_u64, branchless_select_u8, branchless_splitmix64,
    branchless_tanh_f32, fast_inv_len_f32, fast_ln_1p_bounded_f32, is_word_array_uniform_64,
    load_q7_words_64, store_q7_words_64, swar_or_avx2_q7_add_sat_words_64,
    swar_q7_abs_u8_x8, swar_q7_add_sat_x8,
    swar_q7_avg_round_x8, swar_q7_circular_diff_x8, swar_q7_neg_sat_x8,
    swar_q7_scale_le128_with_bias_x8, swar_u8_add_sat_x8, swar_u8_halve_decay_x8, swar_u8_max_x8,
    swar_u8_mean_round_x8, swar_u8_nonzero_count_x8, swar_u8_nonzero_msb_mask, swar_u8_sub_sat_x8,
    trail_phase_words_mut, trail_phase_words_ref,
};
use crate::traits::primitives::{
    ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE,
    ACTION_MODE_SUBHEAD_ROUTE, ACTION_MODE_RANK1_INJECT, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_REPULSION_ESCAPE,
    ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_REPULSION, LEDGER_HISTORY_LEN, FAST_PATH_SHORTCUT_SLOTS,
    DEFAULT_ENTROPY_EPS, GcsoResult, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, PAGED_BITMASK_BITS,
    Pod,
    coupled_slot_row_mask, is_slot_pair_coupled, pack_action_slot_tag, unpack_action_mode_from_tag,
    unpack_attractor_slot_from_tag,
};
use crate::abi::{
    GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_INVALID_STATE, gcso_config_t, gcso_ledger_slot_t,
    gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t, gcso_tagged_ptr_t,
};
use core::mem::{align_of, size_of};

// ===================================================================
// 1. Telemetry, Trail, Anti-Saturation & Swarm Quorum Constants
// ===================================================================

/// Number of parallel 8-head SWAR Stigmergic Granules partitioning the 64 attention heads (`8`).
pub const NUM_STIGMERGIC_GRANULES: usize = 8;
/// Number of attention heads per SWAR Stigmergic Granule (`8` heads = 1 `u64` SWAR register).
pub const HEADS_PER_STIGMERGIC_GRANULE: usize = MAX_ATTENTION_HEADS / NUM_STIGMERGIC_GRANULES;
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
pub const TRAIL_FLAG_DPSR_ACTIVE: u32 = 1 << 0;
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
/// [`crate::traits::gateway::CognitiveStepOp`] records (`[Scale | Op | Intensity]`) rather than legacy metric masks.
pub const LEDGER_FLAG_COGNITIVE_TRAJECTORY: u32 = 1 << 0;


pub mod bitmask;
pub mod bloom;
pub mod ledger;
pub mod stigmergy;
pub mod trail;

pub use self::bitmask::*;
pub use self::bloom::*;
pub use self::ledger::*;
pub use self::stigmergy::*;
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
