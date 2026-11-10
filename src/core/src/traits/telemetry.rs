// name: src/core/src/traits/telemetry.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.1 Capability 2: Stigmergic trails, internal telemetry, inter-memory Limina bridging,
//! Bayesian pointer-trail transition scoring & rewiring, single-manifold unified-field resonance
//! with 32-bit `target_anchor_id` preservation, and micro-swarm intelligence (`PagedBitmaskOps`,
//! `ContrastivePhaseBloomFilter`, `StigmergicActionPtrOps`, `StigmergicTrailOps`,
//! `StigmergicSwarmOps`, `DaesScratchpad`, and `SwarmResolutionReport`) registered into
//! [`super::ActionHub`] for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! All operations execute in strictly constant time (`0` data-dependent early-return branches
//! on convergence, anchor weights, or filter emptiness, and `0 fdiv` in similarity/synergy
//! reductions) to guarantee linear Criterion benchmark scaling.

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

use core::mem::{align_of, size_of};
use core::slice;

pub use super::branchless::swar_q7_circular_add_x8;
use super::branchless::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_is_finite_f32,
    branchless_ln_f32, branchless_max_f32, branchless_min_f32,
    branchless_phase_proximity_gravity_f32, branchless_popcount_rsqrt_u32,
    branchless_q7_accumulate_64, branchless_q7_bayesian_attention_score_delta_64,
    branchless_q7_circular_resultant_64, branchless_q7_circular_weighted_step_64,
    branchless_q7_coherence_64, branchless_q7_inhibit_64,
    branchless_q7_inhibit_words_64, branchless_q7_resonant_pull_64,
    branchless_q7_synergy_fuse_words_64,
    branchless_q7_unified_field_fold_words_with_flags_64,
    branchless_q7_unified_field_fold_words_with_init_weight_64, branchless_relu_f32,
    branchless_rsqrt_f32, branchless_select_f32, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, branchless_select_usize, branchless_splitmix64, branchless_tanh_f32,
    coherence_from_diff_words_64, coherence_from_uniform_diff_word_x8,
    coherence_from_word_arrays_with_both_uniform_flags_64,
    fast_inv_len_f32,
    fast_ln_1p_bounded_f32, is_word_array_uniform_64, load_q7_word_at, load_q7_words_64,
    trail_phase_words_mut, trail_phase_words_ref,
    store_q7_word_at,
    store_q7_words_64, swar_q7_abs_u8_x8, swar_q7_add_sat_x8, swar_q7_avg_round_x8,
    swar_q7_circular_diff_x8, swar_q7_neg_sat_x8, swar_q7_scale_le128_with_bias_x8,
    swar_slerp_words_with_bias_64, swar_slerp_words_with_uniform_flag_64, swar_u8_add_sat_x8,
    swar_u8_halve_decay_x8, swar_u8_max_x8,
    swar_u8_mean_round_x8, swar_u8_nonzero_count_x8, swar_u8_nonzero_msb_mask,
    swar_u8_sub_sat_x8, BranchlessKernel, DEFAULT_PHASE_GRAVITY_EPS, Q7_FULL_COS_REF,
};
use super::primitives::{
    coupled_slot_row_mask, is_slot_pair_coupled, pack_action_slot_tag, unpack_action_mode_from_tag,
    unpack_attractor_slot_from_tag, GcsoResult, Pod, ACTION_MODE_DPSR_STEER,
    ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE, ACTION_MODE_PSPM_ROUTE,
    ACTION_MODE_SRL_INJECT, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION,
    ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK,
    ATTRACTOR_SLOT_REPULSION, DAES_MINI_LEDGER_LEN, DAES_SHORTCUT_SLOTS, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, PAGED_BITMASK_BITS,
};
use crate::abi::{
    gcso_config_t, gcso_daes_slot_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t,
    gcso_tagged_ptr_t, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_INVALID_STATE,
};

const SWAR_SIGN_MASK_64: u64 = 0x8080_8080_8080_8080;
const SWAR_LOW7_MASK_64: u64 = 0x7F7F_7F7F_7F7F_7F7F;
const SWAR_EVEN_BYTES_64: u64 = 0x00FF_00FF_00FF_00FF;
const SWAR_SUM_U16_64: u64 = 0x0001_0001_0001_0001;
const SWAR_SPLAT_BYTE_64: u64 = 0x0101_0101_0101_0101;

// ===================================================================
// 1. Telemetry, Trail, Anti-Saturation & Swarm Quorum Constants
// ===================================================================

/// Number of parallel 8-head SWAR Micro-Swarm cells partitioning the 64 attention heads (`8`).
pub const NUM_SWARM_MICRO_CELLS: usize = 8;
/// Number of attention heads per SWAR Micro-Swarm cell (`8` heads = 1 `u64` SWAR register).
pub const HEADS_PER_SWARM_CELL: usize = MAX_ATTENTION_HEADS / NUM_SWARM_MICRO_CELLS;
/// Default Swarm Quorum threshold (`6` out of `8` Micro-Swarm cells = 75% consensus).
pub const DEFAULT_SWARM_QUORUM_THRESHOLD: u8 = 6;
/// Default Q7 circular phase residual tolerance (`4` Q7 steps $\approx 0.098\text{ rad}$).
pub const DEFAULT_SWARM_TOLERANCE_Q7: u8 = 4;

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

/// DAES telemetry metric bit (`0x01`): Fast-path bypass shortcut hit in `gcso_daes_slot_t`.
pub const DAES_METRIC_FAST_PATH_HIT: u8 = 1 << 0;
/// DAES telemetry metric bit (`0x02`): Inline DPSR/RIPA phase steering applied.
pub const DAES_METRIC_PHASE_STEERED: u8 = 1 << 1;
/// DAES telemetry metric bit (`0x04`): Contrastive Phase Bloom Filter repulsion triggered.
pub const DAES_METRIC_REPULSION_APPLIED: u8 = 1 << 2;
/// DAES telemetry metric bit (`0x08`): Intermediate layer skipped with SWAR phase compensation.
pub const DAES_METRIC_LAYER_SKIP_COMPENSATED: u8 = 1 << 3;
/// DAES telemetry metric bit (`0x10`): PSPM single-pass sub-head routing executed.
pub const DAES_METRIC_PSPM_ROUTED: u8 = 1 << 4;
/// DAES telemetry metric bit (`0x20`): SRL Dynamic Rank-1 residual adapter injected.
pub const DAES_METRIC_SRL_INJECTED: u8 = 1 << 5;
/// DAES telemetry metric bit (`0x40`): Lazy Phase Unwrapping relative shift applied.
pub const DAES_METRIC_LAZY_UNWRAP_APPLIED: u8 = 1 << 6;
/// DAES telemetry metric bit (`0x80`): Unified-field or cross-slot attractor synergy / mutual inhibition active.
pub const DAES_METRIC_SLOT_SYNERGY_ACTIVE: u8 = 1 << 7;

// ===================================================================
// Internal Zero-Cost SWAR 64-Byte Word Helpers
// ===================================================================

#[inline(always)]
#[must_use]
const fn swar_u8_pair_sum_u16(w: u64) -> u64 {
    (w & SWAR_EVEN_BYTES_64).wrapping_add((w >> 8) & SWAR_EVEN_BYTES_64)
}

#[inline(always)]
#[must_use]
const fn swar_total_mass_from_words(w: &[u64; 8]) -> u32 {
    let s01 = swar_u8_pair_sum_u16(w[0]).wrapping_add(swar_u8_pair_sum_u16(w[1]));
    let s23 = swar_u8_pair_sum_u16(w[2]).wrapping_add(swar_u8_pair_sum_u16(w[3]));
    let s45 = swar_u8_pair_sum_u16(w[4]).wrapping_add(swar_u8_pair_sum_u16(w[5]));
    let s67 = swar_u8_pair_sum_u16(w[6]).wrapping_add(swar_u8_pair_sum_u16(w[7]));
    let s_all = s01.wrapping_add(s23).wrapping_add(s45.wrapping_add(s67));
    (s_all.wrapping_mul(SWAR_SUM_U16_64) >> 48) as u32
}

#[inline(always)]
#[must_use]
const fn swar_active_buckets_from_words(w: &[u64; 8]) -> u32 {
    let nz01 = swar_u8_nonzero_msb_mask(w[0]) | (swar_u8_nonzero_msb_mask(w[1]) >> 1);
    let nz23 = (swar_u8_nonzero_msb_mask(w[2]) >> 2) | (swar_u8_nonzero_msb_mask(w[3]) >> 3);
    let nz45 = (swar_u8_nonzero_msb_mask(w[4]) >> 4) | (swar_u8_nonzero_msb_mask(w[5]) >> 5);
    let nz67 = (swar_u8_nonzero_msb_mask(w[6]) >> 6) | (swar_u8_nonzero_msb_mask(w[7]) >> 7);
    ((nz01 | nz23) | (nz45 | nz67)).count_ones()
}

#[inline(always)]
#[must_use]
const fn words_any_nonzero_64(w: &[u64; 8]) -> bool {
    if w[0] != 0 {
        return true;
    }
    (((w[1] | w[2]) | (w[3] | w[4])) | ((w[5] | w[6]) | w[7])) != 0
}

#[inline(always)]
#[must_use]
fn eval_unified_anchor_resonance_words_with_uniform(
    init_w: &[u64; 8],
    init_nonzero: bool,
    init_is_uniform: bool,
    anc_w: &[u64; 8],
    anc_is_uniform: bool,
    base_energy: f32,
) -> (f32, f32) {
    let clean_base = branchless_flush_subnormal_f32(branchless_relu_f32(base_energy));
    if clean_base <= 1.0e-12 {
        return (0.0, 0.0);
    }
    let anc_nonzero = if anc_is_uniform {
        anc_w[0] != 0
    } else {
        words_any_nonzero_64(anc_w)
    };
    let raw_coh = if init_nonzero & anc_nonzero {
        coherence_from_word_arrays_with_both_uniform_flags_64(
            init_w,
            init_is_uniform,
            anc_w,
            anc_is_uniform,
        )
    } else {
        0.0
    };
    let proximity_gain = branchless_clamp_f32(1.0 + raw_coh, 0.25, 2.0);
    (
        branchless_flush_subnormal_f32(clean_base * proximity_gain),
        raw_coh,
    )
}

#[inline(always)]
#[must_use]
const fn precompute_bayesian_trail_scales(
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
) -> (f32, f32, f32, f32) {
    let safe_d = branchless_select_u32(head_dim > 0, head_dim, 128);
    let inv_sqrt_d = if safe_d <= 256 {
        branchless_popcount_rsqrt_u32(safe_d)
    } else {
        branchless_rsqrt_f32(safe_d as f32)
    };
    let phase_scale =
        branchless_flush_subnormal_f32(branchless_relu_f32(phase_interference_gain)) * inv_sqrt_d;
    let clean_eta = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_prior_gain));
    let clean_link = branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));
    let safe_thresh =
        branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
    (phase_scale, clean_eta, clean_link, safe_thresh)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
#[must_use]
fn eval_candidate_trail_words_with_macro_fast(
    self_w: &[u64; 8],
    self_is_uniform: bool,
    self_is_byte_uniform: bool,
    self_b0: u8,
    cur_linked_id: u32,
    cand_id: u32,
    cand: &gcso_pointer_trail_t,
    phase_scale: f32,
    clean_eta: f32,
    clean_link: f32,
    safe_thresh: f32,
    macro_anchor_id: u32,
    win_w_opt: Option<&[u64; 8]>,
    win_meta_evaluated: &mut bool,
    win_is_uniform: &mut bool,
    win_eq_self: &mut bool,
    clean_cluster_bonus: f32,
) -> (f32, f32, bool, bool) {
    let cand_w = trail_phase_words_ref(cand);
    let cand_is_uniform = if self_is_uniform {
        is_word_array_uniform_64(cand_w)
    } else {
        false
    };
    let coh = if cand_is_uniform {
        let cw0 = cand_w[0];
        if cw0 == self_w[0] {
            1.0
        } else if self_is_byte_uniform
            && (cw0 == cw0.rotate_right(8))
        {
            Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
        } else {
            coherence_from_word_arrays_with_both_uniform_flags_64(self_w, true, cand_w, true)
        }
    } else {
        coherence_from_word_arrays_with_both_uniform_flags_64(
            self_w,
            self_is_uniform,
            cand_w,
            false,
        )
    };
    let cand_not_repulsed = (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
    let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
    let eff_prior_link = if is_coherent {
        let prior_term = clean_eta * fast_ln_1p_bounded_f32(branchless_relu_f32(cand.stigmergic_density));
        let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
        let link_term = branchless_select_f32(is_linked, clean_link, 0.0);
        let macro_affinity = if (macro_anchor_id != 0)
            & ((cand.cluster_id == macro_anchor_id) | (cand.target_anchor_id == macro_anchor_id))
        {
            1.0
        } else if (macro_anchor_id != 0)
            & (clean_cluster_bonus > 0.0)
            & (cand.cluster_id == 0)
            & (cand.target_anchor_id == 0)
        {
            match win_w_opt {
                Some(win_w) => {
                    if !*win_meta_evaluated {
                        let eq = (((self_w[0] ^ win_w[0]) | (self_w[1] ^ win_w[1]))
                            | ((self_w[2] ^ win_w[2]) | (self_w[3] ^ win_w[3]))
                            | ((self_w[4] ^ win_w[4]) | (self_w[5] ^ win_w[5]))
                            | ((self_w[6] ^ win_w[6]) | (self_w[7] ^ win_w[7])))
                            == 0;
                        *win_eq_self = eq;
                        *win_is_uniform = if eq {
                            self_is_uniform
                        } else {
                            is_word_array_uniform_64(win_w)
                        };
                        *win_meta_evaluated = true;
                    }
                    if *win_eq_self {
                        branchless_relu_f32(coh)
                    } else {
                        let cand_uni_for_win = if self_is_uniform {
                            cand_is_uniform
                        } else if *win_is_uniform {
                            is_word_array_uniform_64(cand_w)
                        } else {
                            false
                        };
                        branchless_relu_f32(
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                win_w,
                                *win_is_uniform,
                                cand_w,
                                cand_uni_for_win,
                            ),
                        )
                    }
                }
                None => 0.0,
            }
        } else {
            0.0
        };
        prior_term + link_term + (clean_cluster_bonus * macro_affinity)
    } else {
        0.0
    };
    let bayes_delta = branchless_flush_subnormal_f32((phase_scale * coh) + eff_prior_link);
    (bayes_delta, coh, is_coherent, cand_is_uniform)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
#[must_use]
unsafe fn eval_candidate_trail_avx2_with_macro_fast(
    self_w: &[u64; 8],
    self_is_uniform: bool,
    self_is_byte_uniform: bool,
    self_b0: u8,
    self_v0: core::arch::x86_64::__m256i,
    self_v1: core::arch::x86_64::__m256i,
    cur_linked_id: u32,
    cand_id: u32,
    cand: &gcso_pointer_trail_t,
    phase_scale: f32,
    clean_eta: f32,
    clean_link: f32,
    safe_thresh: f32,
    macro_anchor_id: u32,
    win_w_opt: Option<&[u64; 8]>,
    win_meta_evaluated: &mut bool,
    win_is_uniform: &mut bool,
    win_eq_self: &mut bool,
    win_v0: &mut core::arch::x86_64::__m256i,
    win_v1: &mut core::arch::x86_64::__m256i,
    clean_cluster_bonus: f32,
) -> (f32, f32, bool, bool) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`. Keeps `(self_v0, self_v1)` and
    // `(win_v0, win_v1)` hoisted in YMM registers across the candidate loop and loads each
    // candidate phase profile directly via two 256-bit `vmovdqu` instructions (`0` per-candidate
    // GPR-to-YMM Port-5 inserts).
    unsafe {
        use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
        let cw = trail_phase_words_ref(cand);
        let c_uni = if self_is_uniform {
            is_word_array_uniform_64(cw)
        } else {
            false
        };
        let coh = if c_uni {
            let cw0 = cw[0];
            if cw0 == self_w[0] {
                1.0
            } else if self_is_byte_uniform
                && (cw0 == cw0.rotate_right(8))
            {
                Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(self_w, true, cw, true)
            }
        } else {
            let cp = cand.accumulated_phase_delta.as_ptr().cast::<__m256i>();
            let cv0 = _mm256_loadu_si256(cp);
            let cv1 = _mm256_loadu_si256(cp.add(1));
            super::branchless::avx2_coherence_from_diff_vec64(
                _mm256_sub_epi8(self_v0, cv0),
                _mm256_sub_epi8(self_v1, cv1),
            )
        };
        let cand_not_repulsed = (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
        let eff_prior_link = if is_coherent {
            let prior_term =
                clean_eta * fast_ln_1p_bounded_f32(branchless_relu_f32(cand.stigmergic_density));
            let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
            let link_term = branchless_select_f32(is_linked, clean_link, 0.0);
            let macro_affinity = if (macro_anchor_id != 0)
                & ((cand.cluster_id == macro_anchor_id)
                    | (cand.target_anchor_id == macro_anchor_id))
            {
                1.0
            } else if (macro_anchor_id != 0)
                & (clean_cluster_bonus > 0.0)
                & (cand.cluster_id == 0)
                & (cand.target_anchor_id == 0)
            {
                match win_w_opt {
                    Some(win_w) => {
                        if !*win_meta_evaluated {
                            let eq = (((self_w[0] ^ win_w[0]) | (self_w[1] ^ win_w[1]))
                                | ((self_w[2] ^ win_w[2]) | (self_w[3] ^ win_w[3]))
                                | ((self_w[4] ^ win_w[4]) | (self_w[5] ^ win_w[5]))
                                | ((self_w[6] ^ win_w[6]) | (self_w[7] ^ win_w[7])))
                                == 0;
                            *win_eq_self = eq;
                            if eq {
                                *win_is_uniform = self_is_uniform;
                            } else {
                                let w_uni = is_word_array_uniform_64(win_w);
                                *win_is_uniform = w_uni;
                                (*win_v0, *win_v1) =
                                    super::branchless::avx2_words64_to_vec64_with_uniform(
                                        win_w, w_uni,
                                    );
                            }
                            *win_meta_evaluated = true;
                        }
                        if *win_eq_self {
                            branchless_relu_f32(coh)
                        } else if *win_is_uniform & c_uni {
                            branchless_relu_f32(
                                coherence_from_word_arrays_with_both_uniform_flags_64(
                                    win_w, true, cw, true,
                                ),
                            )
                        } else {
                            let cp = cand.accumulated_phase_delta.as_ptr().cast::<__m256i>();
                            let cv0 = _mm256_loadu_si256(cp);
                            let cv1 = _mm256_loadu_si256(cp.add(1));
                            branchless_relu_f32(super::branchless::avx2_coherence_from_diff_vec64(
                                _mm256_sub_epi8(*win_v0, cv0),
                                _mm256_sub_epi8(*win_v1, cv1),
                            ))
                        }
                    }
                    None => 0.0,
                }
            } else {
                0.0
            };
            prior_term + link_term + (clean_cluster_bonus * macro_affinity)
        } else {
            0.0
        };
        let bayes_delta = branchless_flush_subnormal_f32((phase_scale * coh) + eff_prior_link);
        (bayes_delta, coh, is_coherent, c_uni)
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline(always)]
#[must_use]
fn eval_candidate_trail_words_fast(
    self_w: &[u64; 8],
    self_is_uniform: bool,
    self_is_byte_uniform: bool,
    self_b0: u8,
    cur_linked_id: u32,
    cand_id: u32,
    cand: &gcso_pointer_trail_t,
    phase_scale: f32,
    clean_eta: f32,
    clean_link: f32,
    safe_thresh: f32,
) -> (f32, f32, bool) {
    let mut win_meta_evaluated = true;
    let mut win_is_uniform = false;
    let mut win_eq_self = false;
    let (bayes_delta, coh, is_coherent, _) = eval_candidate_trail_words_with_macro_fast(
        self_w,
        self_is_uniform,
        self_is_byte_uniform,
        self_b0,
        cur_linked_id,
        cand_id,
        cand,
        phase_scale,
        clean_eta,
        clean_link,
        safe_thresh,
        0,
        None,
        &mut win_meta_evaluated,
        &mut win_is_uniform,
        &mut win_eq_self,
        0.0,
    );
    (bayes_delta, coh, is_coherent)
}

/// Hoisted-invariant 8-head SWAR Micro-Swarm cell resolver using precomputed `gain_s <= 128`,
/// `gain_bias = (128 - gain_s) * 0x0101_0101_0101_0101`, `qdps_thresh_x8`, and `qdps_valid_mask`.
/// Eliminates per-cell 64-bit bias/threshold multiplications and bypasses `swar_q7_add_sat_x8`
/// when `step_gain_u7 <= 128`.
#[inline(always)]
#[must_use]
const fn swar_subswarm_resolve_with_bias_x8(
    current_word: u64,
    target_word: u64,
    gain_s: u64,
    gain_bias: u64,
    gain_gt_128: bool,
    qdps_thresh_x8: u64,
    qdps_valid_mask: u64,
    tolerance_q7: u8,
) -> (u64, u8, bool) {
    if current_word == target_word {
        return (current_word, 0, true);
    }
    let raw_diff = swar_q7_circular_diff_x8(target_word, current_word);
    let filtered_diff = if qdps_thresh_x8 <= SWAR_SPLAT_BYTE_64 {
        raw_diff & qdps_valid_mask
    } else {
        let abs_lanes = swar_q7_abs_u8_x8(raw_diff);
        let ge_bit7 = (((abs_lanes | SWAR_SIGN_MASK_64).wrapping_sub(qdps_thresh_x8))
            | (abs_lanes & SWAR_SIGN_MASK_64))
            & SWAR_SIGN_MASK_64;
        let keep_mask = (ge_bit7 >> 7).wrapping_mul(0xFF);
        raw_diff & keep_mask & qdps_valid_mask
    };

    let step_word = if (!gain_gt_128) & (gain_s == 128) {
        filtered_diff
    } else {
        let base_scaled = swar_q7_scale_le128_with_bias_x8(filtered_diff, gain_s, gain_bias);
        if gain_gt_128 {
            swar_q7_add_sat_x8(base_scaled, filtered_diff)
        } else {
            base_scaled
        }
    };
    let next_word = swar_q7_circular_add_x8(current_word, step_word);
    if step_word == raw_diff {
        return (next_word, 0, true);
    }
    let rem_diff = swar_q7_circular_diff_x8(raw_diff, step_word);
    let rem_abs = swar_q7_abs_u8_x8(rem_diff);
    let mean_res = swar_u8_mean_round_x8(rem_abs);
    (next_word, mean_res, mean_res <= tolerance_q7)
}

#[inline(always)]
#[must_use]
const fn precompute_swarm_step_params(
    step_gain_u7: u8,
    qdps_min_step_q7: u8,
) -> (u64, u64, bool, u64, u64) {
    let gain_gt_128 = step_gain_u7 > 128;
    let gain_s =
        branchless_select_u8(gain_gt_128, step_gain_u7.wrapping_sub(128), step_gain_u7) as u64;
    let gain_bias = (128u64.wrapping_sub(gain_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let qdps_thresh_x8 = ((qdps_min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let qdps_valid_mask = ((qdps_min_step_q7 <= 127) as u64).wrapping_neg();
    (gain_s, gain_bias, gain_gt_128, qdps_thresh_x8, qdps_valid_mask)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
#[must_use]
unsafe fn avx2_subswarm_resolve_words_64(
    cur_w: &[u64; NUM_SWARM_MICRO_CELLS],
    tgt_base_w: &[u64; NUM_SWARM_MICRO_CELLS],
    use_explicit_tgt: bool,
    ptr_phase_splat: u64,
    active_cell_mask: u8,
    step_gain_u7: u8,
    qdps_min_step_q7: u8,
    tolerance_q7: u8,
) -> ([u64; NUM_SWARM_MICRO_CELLS], u8, u32) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`. Resolves all 8 Micro-Swarm cells (64 heads)
    // in parallel inside two 256-bit YMM registers (`_mm256_mulhi_epi16(d << 8, s << 1)` and `_mm256_sad_epu8`),
    // preserving 100% bit-for-bit identity with `swar_subswarm_resolve_with_bias_x8`.
    unsafe {
        use core::arch::x86_64::{
            __m256i, _mm256_abs_epi8, _mm256_add_epi64, _mm256_add_epi8, _mm256_adds_epi8,
            _mm256_and_si256, _mm256_castsi256_pd, _mm256_castsi256_si128, _mm256_cmpeq_epi8,
            _mm256_cmpgt_epi64, _mm256_extracti128_si256, _mm256_max_epu8, _mm256_movemask_pd,
            _mm256_mulhi_epi16, _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi16,
            _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_slli_epi16,
            _mm256_srli_epi64, _mm256_sub_epi8, _mm_add_epi64, _mm_cvtsi128_si64,
            _mm_unpackhi_epi64,
        };
        use core::arch::x86_64::_mm256_testz_si256;
        let _ = core::mem::size_of::<__m256i>();
        let (vc0, vc1) = crate::traits::branchless::avx2_words64_to_vec64(cur_w);
        let (vt0, vt1) = if use_explicit_tgt {
            crate::traits::branchless::avx2_words64_to_vec64(tgt_base_w)
        } else {
            let v_splat = _mm256_set1_epi64x(ptr_phase_splat as i64);
            (v_splat, v_splat)
        };

        let vd0 = _mm256_sub_epi8(vt0, vc0);
        let vd1 = _mm256_sub_epi8(vt1, vc1);
        let vor = _mm256_or_si256(vd0, vd1);
        if _mm256_testz_si256(vor, vor) != 0 {
            return (*cur_w, 0xFF, 0);
        }
        let zero = _mm256_setzero_si256();

        let (vf0, vf1) = if qdps_min_step_q7 <= 1 {
            (vd0, vd1)
        } else if qdps_min_step_q7 > 127 {
            (zero, zero)
        } else {
            let v_thresh = _mm256_set1_epi8(qdps_min_step_q7 as i8);
            let a0 = _mm256_abs_epi8(vd0);
            let a1 = _mm256_abs_epi8(vd1);
            let k0 = _mm256_cmpeq_epi8(_mm256_max_epu8(a0, v_thresh), a0);
            let k1 = _mm256_cmpeq_epi8(_mm256_max_epu8(a1, v_thresh), a1);
            (_mm256_and_si256(vd0, k0), _mm256_and_si256(vd1, k1))
        };

        let (vs0, vs1) = if step_gain_u7 == 128 {
            (vf0, vf1)
        } else if step_gain_u7 == 0 {
            (zero, zero)
        } else {
            let gt_128 = step_gain_u7 > 128;
            let s_sub = branchless_select_u8(gt_128, step_gain_u7.wrapping_sub(128), step_gain_u7);
            let v_s2 = _mm256_set1_epi16(i16::from(s_sub) << 1);
            let mask_ff00 = _mm256_set1_epi16(-256i16);
            let mask_00ff = _mm256_set1_epi16(0x00FFi16);
            let ev0 = _mm256_mulhi_epi16(_mm256_slli_epi16(vf0, 8), v_s2);
            let od0 = _mm256_mulhi_epi16(_mm256_and_si256(vf0, mask_ff00), v_s2);
            let b0 = _mm256_or_si256(_mm256_and_si256(ev0, mask_00ff), _mm256_slli_epi16(od0, 8));
            let ev1 = _mm256_mulhi_epi16(_mm256_slli_epi16(vf1, 8), v_s2);
            let od1 = _mm256_mulhi_epi16(_mm256_and_si256(vf1, mask_ff00), v_s2);
            let b1 = _mm256_or_si256(_mm256_and_si256(ev1, mask_00ff), _mm256_slli_epi16(od1, 8));
            if gt_128 {
                (_mm256_adds_epi8(b0, vf0), _mm256_adds_epi8(b1, vf1))
            } else {
                (b0, b1)
            }
        };

        let vn0 = _mm256_add_epi8(vc0, vs0);
        let vn1 = _mm256_add_epi8(vc1, vs1);
        let rem0 = _mm256_abs_epi8(_mm256_sub_epi8(vd0, vs0));
        let rem1 = _mm256_abs_epi8(_mm256_sub_epi8(vd1, vs1));
        let four64 = _mm256_set1_epi64x(4);
        let res0_vec = _mm256_srli_epi64(_mm256_add_epi64(_mm256_sad_epu8(rem0, zero), four64), 3);
        let res1_vec = _mm256_srli_epi64(_mm256_add_epi64(_mm256_sad_epu8(rem1, zero), four64), 3);
        let v_tol_plus1 = _mm256_set1_epi64x(i64::from(tolerance_q7) + 1);
        let ok0 = _mm256_cmpgt_epi64(v_tol_plus1, res0_vec);
        let ok1 = _mm256_cmpgt_epi64(v_tol_plus1, res1_vec);
        let raw_quorum = (_mm256_movemask_pd(_mm256_castsi256_pd(ok0)) as u8)
            | ((_mm256_movemask_pd(_mm256_castsi256_pd(ok1)) as u8) << 4);
        let mut next_words = crate::traits::branchless::avx2_vec64_to_words64(vn0, vn1);

        if active_cell_mask == 0xFF {
            let r_sum4 = _mm256_add_epi64(res0_vec, res1_vec);
            let r_sum2 = _mm_add_epi64(
                _mm256_castsi256_si128(r_sum4),
                _mm256_extracti128_si256(r_sum4, 1),
            );
            let residual_sum = _mm_cvtsi128_si64(_mm_add_epi64(
                r_sum2,
                _mm_unpackhi_epi64(r_sum2, r_sum2),
            )) as u32;
            return (next_words, raw_quorum, residual_sum);
        }

        let res_arr = crate::traits::branchless::avx2_vec64_to_words64(res0_vec, res1_vec);
        let mut residual_sum = 0u32;
        let mut cell = 0usize;
        while cell < NUM_SWARM_MICRO_CELLS {
            let cell_active = ((active_cell_mask >> cell) & 1) != 0;
            next_words[cell] = branchless_select_u64(cell_active, next_words[cell], cur_w[cell]);
            residual_sum += branchless_select_u32(cell_active, res_arr[cell] as u32, 0);
            cell += 1;
        }
        (next_words, raw_quorum | (!active_cell_mask), residual_sum)
    }
}

#[inline(always)]
#[must_use]
fn swar_subswarm_resolve_words_64(
    cur_w: &[u64; NUM_SWARM_MICRO_CELLS],
    tgt_base_w: &[u64; NUM_SWARM_MICRO_CELLS],
    use_explicit_tgt: bool,
    ptr_phase_splat: u64,
    active_cell_mask: u8,
    step_gain_u7: u8,
    qdps_min_step_q7: u8,
    tolerance_q7: u8,
) -> ([u64; NUM_SWARM_MICRO_CELLS], u8, u32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `cfg(target_feature = "avx2")`.
        return unsafe {
            avx2_subswarm_resolve_words_64(
                cur_w,
                tgt_base_w,
                use_explicit_tgt,
                ptr_phase_splat,
                active_cell_mask,
                step_gain_u7,
                qdps_min_step_q7,
                tolerance_q7,
            )
        };
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let (gain_s, gain_bias, gain_gt_128, qdps_thresh_x8, qdps_valid_mask) =
            precompute_swarm_step_params(step_gain_u7, qdps_min_step_q7);
        let mut next_words = *cur_w;
        let mut quorum_mask = 0u8;
        let mut residual_sum = 0u32;
        let mut has_prev = false;
        let mut prev_cw = 0u64;
        let mut prev_tw = 0u64;
        let mut prev_stepped = 0u64;
        let mut prev_res = 0u8;
        let mut prev_ok = false;
        let mut cell = 0usize;
        while cell < NUM_SWARM_MICRO_CELLS {
            let cell_active = ((active_cell_mask >> cell) & 1) != 0;
            let cw = cur_w[cell];
            let tw = branchless_select_u64(use_explicit_tgt, tgt_base_w[cell], ptr_phase_splat);
            let (stepped_w, cell_res, cell_ok) = if has_prev & (cw == prev_cw) & (tw == prev_tw) {
                (prev_stepped, prev_res, prev_ok)
            } else {
                let (sw, cr, co) = swar_subswarm_resolve_with_bias_x8(
                    cw,
                    tw,
                    gain_s,
                    gain_bias,
                    gain_gt_128,
                    qdps_thresh_x8,
                    qdps_valid_mask,
                    tolerance_q7,
                );
                has_prev = true;
                prev_cw = cw;
                prev_tw = tw;
                prev_stepped = sw;
                prev_res = cr;
                prev_ok = co;
                (sw, cr, co)
            };
            next_words[cell] = branchless_select_u64(cell_active, stepped_w, cw);
            let eff_res = branchless_select_u8(cell_active, cell_res, 0);
            let eff_resolved = (!cell_active) | cell_ok;
            quorum_mask |= (eff_resolved as u8) << cell;
            residual_sum += u32::from(eff_res);
            cell += 1;
        }
        (next_words, quorum_mask, residual_sum)
    }
}

// ===================================================================
// 2. 256-Bit PagedBlock Bitmask Algebra (`PagedBitmaskOps`)
// ===================================================================

/// Branchless $\mathcal{O}(1)$ bitwise, stigmergic graph adjacency, and 8-cell sub-swarm partitioning
/// operations on 256-bit [`gcso_paged_bitmask_t`].
pub trait PagedBitmaskOps: Sized {
    /// Returns `true` if all 256 bits across the four 64-bit words are zero.
    fn is_all_zero(&self) -> bool;
    /// Returns the total Hamming weight (`popcount`) across all 256 bits in $\mathcal{O}(1)$.
    fn popcount(&self) -> u32;
    /// Sets bit `bit_idx` ($0 \le \text{bit\_idx} < 256$) branchlessly. Out-of-range indices safely no-op.
    fn set_bit_branchless(&mut self, bit_idx: u32);
    /// Clears bit `bit_idx` ($0 \le \text{bit\_idx} < 256$) branchlessly. Out-of-range indices safely no-op.
    fn clear_bit_branchless(&mut self, bit_idx: u32);
    /// Tests whether bit `bit_idx` ($0 \le \text{bit\_idx} < 256$) is set branchlessly.
    fn test_bit_branchless(&self, bit_idx: u32) -> bool;
    /// Returns the lowest set bit index (`0..=255`), or `256` (`PAGED_BITMASK_BITS`) if all bits are zero.
    fn first_set_bit(&self) -> u32;
    /// Returns the highest set bit index (`0..=255`), or `256` (`PAGED_BITMASK_BITS`) if all bits are zero.
    fn last_set_bit(&self) -> u32;
    /// Compresses the 256-bit PagedBlock bitmask into an **8-bit Micro-Swarm Activity Mask**.
    fn subswarm_activity_mask_8(&self) -> u8;
    /// Returns the active token count (`0..=32`) for each of the 8 Micro-Swarm 32-bit sub-regions.
    fn subswarm_popcounts_8(&self) -> [u8; NUM_SWARM_MICRO_CELLS];
    /// Computes bitwise NOT ($\neg \mathbf{M}_i$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitnot_mask(&self) -> Self;
    /// Computes bitwise AND ($\mathbf{M}_i \land \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitand_mask(&self, other: &Self) -> Self;
    /// Computes bitwise OR ($\mathbf{M}_i \lor \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitor_mask(&self, other: &Self) -> Self;
    /// Computes bitwise XOR ($\mathbf{M}_i \oplus \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitxor_mask(&self, other: &Self) -> Self;
    /// Computes bitwise relative complement ($\mathbf{M}_i \land \neg \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitandnot_mask(&self, other: &Self) -> Self;
    /// Accumulates `other` into `self` via bitwise OR in-place ($\mathbf{M}_i \gets \mathbf{M}_i \lor \mathbf{M}_j$).
    fn merge_or_assign(&mut self, other: &Self);
    /// Computes Hamming distance $\mathrm{popcount}(\mathbf{M}_i \oplus \mathbf{M}_j)$ in $\mathcal{O}(1)$.
    fn hamming_distance(&self, other: &Self) -> u32;
    /// Evaluates discrete Stigmergic Graph Laplacian adjacency weight $A_{ij} = \mathbf{M}_i^T \mathbf{M}_j$.
    fn stigmergic_adjacency_overlap(&self, other: &Self) -> u32;
    /// Evaluates normalized Jaccard similarity $\frac{\vert{}\mathbf{M}_i \cap \mathbf{M}_j\vert{}}{\vert{}\mathbf{M}_i \cup \mathbf{M}_j\vert{}} \in [0, 1]$
    /// in division-free (`0 fdiv`) constant time via `POPCOUNT_RSQRT_LUT_257`.
    fn stigmergic_jaccard_similarity(&self, other: &Self) -> f32;
    /// Evaluates non-negative binary cosine similarity $\frac{\mathbf{M}_i^T \mathbf{M}_j}{\sqrt{\Vert{}\mathbf{M}_i\Vert{}_1 \Vert{}\mathbf{M}_j\Vert{}_1}} \in [0, 1]$.
    fn stigmergic_cosine_similarity(&self, other: &Self) -> f32;

    /// Evaluates binary cosine similarity using a precomputed reciprocal square root `inv_sqrt_norm_self`
    /// and the 64B-aligned `POPCOUNT_RSQRT_LUT_257` table in strictly constant time.
    #[inline(always)]
    fn stigmergic_cosine_similarity_with_inv_norm(
        &self,
        inv_sqrt_norm_self: f32,
        other: &Self,
    ) -> f32 {
        let inter = self.stigmergic_adjacency_overlap(other);
        let inv_sqrt_other = branchless_popcount_rsqrt_u32(other.popcount());
        let clean_self = branchless_relu_f32(inv_sqrt_norm_self);
        (inter as f32) * clean_self * inv_sqrt_other
    }

    /// Evaluates the Lowest Common Ancestor (LCA) prefix depth $l_{\mathrm{lca}}(x, y) \in [0, 256]$
    /// between two 256-bit dyadic keys in $\mathcal{O}(1)$ branchless operations (Lemma 1.1).
    fn dyadic_lca_depth(&self, other: &Self) -> u32;
}

impl PagedBitmaskOps for gcso_paged_bitmask_t {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        ((self.bits[0] | self.bits[1]) | (self.bits[2] | self.bits[3])) == 0
    }

    #[inline(always)]
    fn popcount(&self) -> u32 {
        (self.bits[0].count_ones() + self.bits[1].count_ones())
            + (self.bits[2].count_ones() + self.bits[3].count_ones())
    }

    #[inline(always)]
    fn set_bit_branchless(&mut self, bit_idx: u32) {
        let in_bounds_mask = ((bit_idx < PAGED_BITMASK_BITS) as u64).wrapping_neg();
        let bit_val = (1u64 << (bit_idx & 63)) & in_bounds_mask;
        self.bits[((bit_idx >> 6) as usize) & 3] |= bit_val;
    }

    #[inline(always)]
    fn clear_bit_branchless(&mut self, bit_idx: u32) {
        let in_bounds_mask = ((bit_idx < PAGED_BITMASK_BITS) as u64).wrapping_neg();
        let clear_mask = !((1u64 << (bit_idx & 63)) & in_bounds_mask);
        self.bits[((bit_idx >> 6) as usize) & 3] &= clear_mask;
    }

    #[inline(always)]
    fn test_bit_branchless(&self, bit_idx: u32) -> bool {
        let in_bounds = bit_idx < PAGED_BITMASK_BITS;
        let word = self.bits[((bit_idx >> 6) as usize) & 3];
        in_bounds & (((word >> (bit_idx & 63)) & 1) != 0)
    }

    #[inline(always)]
    fn first_set_bit(&self) -> u32 {
        let z0 = self.bits[0] == 0;
        let z01 = z0 & (self.bits[1] == 0);
        let z012 = z01 & (self.bits[2] == 0);
        let all_zero = z012 & (self.bits[3] == 0);
        let active_w = branchless_select_u64(
            z0,
            branchless_select_u64(z01, branchless_select_u64(z012, self.bits[3], self.bits[2]), self.bits[1]),
            self.bits[0],
        );
        let base_tz = ((z0 as u32) + (z01 as u32) + (z012 as u32)) << 6;
        branchless_select_u32(
            all_zero,
            PAGED_BITMASK_BITS,
            base_tz + (active_w | (1u64 << 63)).trailing_zeros(),
        )
    }

    #[inline(always)]
    fn last_set_bit(&self) -> u32 {
        let e3 = self.bits[3] == 0;
        let e32 = e3 & (self.bits[2] == 0);
        let e321 = e32 & (self.bits[1] == 0);
        let all_zero = e321 & (self.bits[0] == 0);
        let active_w = branchless_select_u64(
            e3,
            branchless_select_u64(e32, branchless_select_u64(e321, self.bits[0], self.bits[1]), self.bits[2]),
            self.bits[3],
        );
        let base_lz = ((e3 as u32) + (e32 as u32) + (e321 as u32)) << 6;
        branchless_select_u32(
            all_zero,
            PAGED_BITMASK_BITS,
            255u32.wrapping_sub(base_lz + (active_w | 1).leading_zeros()),
        )
    }

    #[inline(always)]
    fn subswarm_activity_mask_8(&self) -> u8 {
        (((self.bits[0] & 0xFFFF_FFFF) != 0) as u8)
            | ((((self.bits[0] >> 32) != 0) as u8) << 1)
            | ((((self.bits[1] & 0xFFFF_FFFF) != 0) as u8) << 2)
            | ((((self.bits[1] >> 32) != 0) as u8) << 3)
            | ((((self.bits[2] & 0xFFFF_FFFF) != 0) as u8) << 4)
            | ((((self.bits[2] >> 32) != 0) as u8) << 5)
            | ((((self.bits[3] & 0xFFFF_FFFF) != 0) as u8) << 6)
            | ((((self.bits[3] >> 32) != 0) as u8) << 7)
    }

    #[inline(always)]
    fn subswarm_popcounts_8(&self) -> [u8; NUM_SWARM_MICRO_CELLS] {
        [
            (self.bits[0] as u32).count_ones() as u8,
            ((self.bits[0] >> 32) as u32).count_ones() as u8,
            (self.bits[1] as u32).count_ones() as u8,
            ((self.bits[1] >> 32) as u32).count_ones() as u8,
            (self.bits[2] as u32).count_ones() as u8,
            ((self.bits[2] >> 32) as u32).count_ones() as u8,
            (self.bits[3] as u32).count_ones() as u8,
            ((self.bits[3] >> 32) as u32).count_ones() as u8,
        ]
    }

    #[inline(always)]
    fn bitnot_mask(&self) -> Self {
        Self {
            bits: [!self.bits[0], !self.bits[1], !self.bits[2], !self.bits[3]],
        }
    }

    #[inline(always)]
    fn bitand_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] & other.bits[0],
                self.bits[1] & other.bits[1],
                self.bits[2] & other.bits[2],
                self.bits[3] & other.bits[3],
            ],
        }
    }

    #[inline(always)]
    fn bitor_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] | other.bits[0],
                self.bits[1] | other.bits[1],
                self.bits[2] | other.bits[2],
                self.bits[3] | other.bits[3],
            ],
        }
    }

    #[inline(always)]
    fn bitxor_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] ^ other.bits[0],
                self.bits[1] ^ other.bits[1],
                self.bits[2] ^ other.bits[2],
                self.bits[3] ^ other.bits[3],
            ],
        }
    }

    #[inline(always)]
    fn bitandnot_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] & !other.bits[0],
                self.bits[1] & !other.bits[1],
                self.bits[2] & !other.bits[2],
                self.bits[3] & !other.bits[3],
            ],
        }
    }

    #[inline(always)]
    fn merge_or_assign(&mut self, other: &Self) {
        self.bits[0] |= other.bits[0];
        self.bits[1] |= other.bits[1];
        self.bits[2] |= other.bits[2];
        self.bits[3] |= other.bits[3];
    }

    #[inline(always)]
    fn hamming_distance(&self, other: &Self) -> u32 {
        ((self.bits[0] ^ other.bits[0]).count_ones() + (self.bits[1] ^ other.bits[1]).count_ones())
            + ((self.bits[2] ^ other.bits[2]).count_ones()
                + (self.bits[3] ^ other.bits[3]).count_ones())
    }

    #[inline(always)]
    fn stigmergic_adjacency_overlap(&self, other: &Self) -> u32 {
        ((self.bits[0] & other.bits[0]).count_ones() + (self.bits[1] & other.bits[1]).count_ones())
            + ((self.bits[2] & other.bits[2]).count_ones()
                + (self.bits[3] & other.bits[3]).count_ones())
    }

    #[inline(always)]
    fn stigmergic_jaccard_similarity(&self, other: &Self) -> f32 {
        let inter = self.stigmergic_adjacency_overlap(other);
        let union = ((self.bits[0] | other.bits[0]).count_ones()
            + (self.bits[1] | other.bits[1]).count_ones())
            + ((self.bits[2] | other.bits[2]).count_ones()
                + (self.bits[3] | other.bits[3]).count_ones());
        let rsqrt_u = branchless_popcount_rsqrt_u32(union);
        branchless_clamp_f32((inter as f32) * (rsqrt_u * rsqrt_u), 0.0, 1.0)
    }

    #[inline(always)]
    fn stigmergic_cosine_similarity(&self, other: &Self) -> f32 {
        let inter = self.stigmergic_adjacency_overlap(other);
        let rsqrt_a = branchless_popcount_rsqrt_u32(self.popcount());
        let rsqrt_b = branchless_popcount_rsqrt_u32(other.popcount());
        (inter as f32) * (rsqrt_a * rsqrt_b)
    }

    #[inline(always)]
    fn dyadic_lca_depth(&self, other: &Self) -> u32 {
        let d0 = self.bits[0] ^ other.bits[0];
        let d1 = self.bits[1] ^ other.bits[1];
        let d2 = self.bits[2] ^ other.bits[2];
        let d3 = self.bits[3] ^ other.bits[3];
        let eq0 = d0 == 0;
        let eq01 = eq0 & (d1 == 0);
        let eq012 = eq01 & (d2 == 0);
        let all_zero = eq012 & (d3 == 0);
        let active_w = branchless_select_u64(
            eq0,
            branchless_select_u64(eq01, branchless_select_u64(eq012, d3, d2), d1),
            d0,
        );
        let base_lz = ((eq0 as u32) + (eq01 as u32) + (eq012 as u32)) << 6;
        branchless_select_u32(
            all_zero,
            PAGED_BITMASK_BITS,
            base_lz + (active_w | 1).leading_zeros(),
        )
    }
}

/// Evaluates the normalized pseudo-ultrametric tree distance $d_{\mathcal{V}}(x, y) \in [0, 1)$
/// directly from two 256-bit dyadic bitmasks in $\mathcal{O}(1)$ (Lemma 1.1).
#[inline(always)]
#[must_use]
pub const fn eval_bitmask_dyadic_ultrametric(
    mask_x: &gcso_paged_bitmask_t,
    mask_y: &gcso_paged_bitmask_t,
    w_key: f32,
) -> f32 {
    let d0 = mask_x.bits[0] ^ mask_y.bits[0];
    let d1 = mask_x.bits[1] ^ mask_y.bits[1];
    let d2 = mask_x.bits[2] ^ mask_y.bits[2];
    let d3 = mask_x.bits[3] ^ mask_y.bits[3];
    let eq0 = d0 == 0;
    let eq01 = eq0 & (d1 == 0);
    let eq012 = eq01 & (d2 == 0);
    let all_zero = eq012 & (d3 == 0);
    let active_w = branchless_select_u64(
        eq0,
        branchless_select_u64(eq01, branchless_select_u64(eq012, d3, d2), d1),
        d0,
    );
    let base_lz = ((eq0 as u32) + (eq01 as u32) + (eq012 as u32)) << 6;
    let l_lca = branchless_select_u32(
        all_zero,
        PAGED_BITMASK_BITS,
        base_lz + (active_w | 1).leading_zeros(),
    );
    let diff = PAGED_BITMASK_BITS.saturating_sub(l_lca) as f32;
    branchless_tanh_f32(branchless_relu_f32(w_key) * diff)
}

// ===================================================================
// 3. Pointer-Driven Action Mode Orchestration (`StigmergicActionPtrOps`)
// ===================================================================

/// Branchless helper trait extending [`gcso_tagged_ptr_t`] into a self-contained
/// orchestrator micro-instruction word (48-bit Address + 5-bit Action Mode + 3-bit Working Lane ID + 8-bit Q7 Phase).
pub trait StigmergicActionPtrOps: Copy {
    /// Packs a 48-bit address, 3-bit target working lane (`0..=7`), 5-bit action mode mask, and Q7 phase.
    #[must_use]
    fn pack_action(
        raw_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: gcso_q7_t,
    ) -> Self;
    /// Extracts the 5-bit Action Mode execution bitmask (`ACTION_MODE_*`).
    #[must_use]
    fn action_mode_flags(self) -> u8;
    /// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`).
    #[must_use]
    fn target_attractor_slot(self) -> u8;
    /// Returns `true` if all specified `mode_mask` bits are enabled in the pointer's Action Mode.
    #[must_use]
    fn has_action_mode(self, mode_mask: u8) -> bool;
    /// Returns a new tagged pointer with updated Target Attractor Working Lane ID and Action Mode flags.
    #[must_use]
    fn with_action_and_slot(self, attractor_slot_id: u8, action_mode_flags: u8) -> Self;
    /// Branchlessly mutates the pointer's Action Mode flags, Target Working Lane ID, and embedded
    /// Q7 phase based on bottom-up stigmergic environmental traces.
    #[must_use]
    fn evolve_mode_stigmergically(
        self,
        trail_density: f32,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_srl_thresh: f32,
    ) -> Self;
}

impl StigmergicActionPtrOps for gcso_tagged_ptr_t {
    #[inline(always)]
    fn pack_action(
        raw_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: gcso_q7_t,
    ) -> Self {
        Self::pack(
            raw_addr,
            pack_action_slot_tag(attractor_slot_id, action_mode_flags),
            phase_q7,
        )
    }

    #[inline(always)]
    fn action_mode_flags(self) -> u8 {
        unpack_action_mode_from_tag(self.unpack_tag())
    }

    #[inline(always)]
    fn target_attractor_slot(self) -> u8 {
        unpack_attractor_slot_from_tag(self.unpack_tag())
    }

    #[inline(always)]
    fn has_action_mode(self, mode_mask: u8) -> bool {
        (self.action_mode_flags() & mode_mask) == mode_mask
    }

    #[inline(always)]
    fn with_action_and_slot(self, attractor_slot_id: u8, action_mode_flags: u8) -> Self {
        self.with_tag(pack_action_slot_tag(attractor_slot_id, action_mode_flags))
    }

    #[inline(always)]
    fn evolve_mode_stigmergically(
        self,
        trail_density: f32,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_srl_thresh: f32,
    ) -> Self {
        let finite_h = branchless_is_finite_f32(entropy_h_tilde);
        let clean_h = branchless_select_f32(finite_h, entropy_h_tilde, 0.0);
        let clean_rho = branchless_relu_f32(trail_density);

        let active_signal = (!self.is_null()) | (self.unpack_phase() != 0) | (clean_rho > 0.0);
        let m_dpsr = branchless_select_u8(active_signal, ACTION_MODE_DPSR_STEER, 0);
        let can_skip = active_signal & (clean_h <= low_entropy_skip_thresh) & (!repulsion_triggered);
        let m_skip = branchless_select_u8(can_skip, ACTION_MODE_LAYER_SKIP, 0);
        let m_rep = branchless_select_u8(repulsion_triggered, ACTION_MODE_PHASE_REPULSE, 0);
        let need_pspm = (clean_h > low_entropy_skip_thresh) | repulsion_triggered;
        let m_pspm = branchless_select_u8(need_pspm, ACTION_MODE_PSPM_ROUTE, 0);
        let need_srl = (clean_rho >= high_density_srl_thresh) & (high_density_srl_thresh > 0.0);
        let m_srl = branchless_select_u8(need_srl, ACTION_MODE_SRL_INJECT, 0);

        let next_mode = m_dpsr | m_skip | m_rep | m_pspm | m_srl;
        let next_slot = branchless_select_u8(
            repulsion_triggered,
            ATTRACTOR_SLOT_REPULSION,
            dominant_attractor_slot & ATTRACTOR_SLOT_ID_MASK,
        );

        let updated = self.with_action_and_slot(next_slot, next_mode);
        let conjugated = updated.conjugate_phase();
        Self(branchless_select_u64(
            repulsion_triggered,
            conjugated.raw(),
            updated.raw(),
        ))
    }
}

// ===================================================================
// 4. Constant-Time Contrastive Phase Memory Counting Bloom Filter
// ===================================================================

/// Cacheline-aligned (64-byte) Counting Bloom Filter for Contrastive Phase Memory (Section 3.3).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContrastivePhaseBloomFilter {
    /// 64 saturating 8-bit frequency counters occupying exactly 1 CPU/GPU cacheline.
    pub counters: [u8; 64],
}

impl Default for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl ContrastivePhaseBloomFilter {
    /// Constructs a zeroed `ContrastivePhaseBloomFilter`.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counters: [0u8; 64],
        }
    }

    /// Borrows `self.counters` (`#[repr(C, align(64))]`) as a zero-copy 64-byte aligned `&[u64; 8]`
    /// SWAR word array on little-endian targets.
    #[inline(always)]
    #[must_use]
    pub const fn counter_words_ref(&self) -> &[u64; 8] {
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: `ContrastivePhaseBloomFilter` is `#[repr(C, align(64))]` (`64 % 8 == 0`),
        // and `counters: [u8; 64]` has identical size and bit validity as `[u64; 8]`.
        unsafe { &*(self.counters.as_ptr().cast::<[u64; 8]>()) }
    }

    /// Mutably borrows `self.counters` (`#[repr(C, align(64))]`) as a zero-copy 64-byte aligned
    /// `&mut [u64; 8]` SWAR word array on little-endian targets.
    #[inline(always)]
    #[must_use]
    pub const fn counter_words_mut(&mut self) -> &mut [u64; 8] {
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: `ContrastivePhaseBloomFilter` is `#[repr(C, align(64))]` (`64 % 8 == 0`),
        // and `counters: [u8; 64]` has identical size and bit validity as `[u64; 8]`.
        unsafe { &mut *(self.counters.as_mut_ptr().cast::<[u64; 8]>()) }
    }

    /// Resets all 64 counters to zero in-place.
    #[inline(always)]
    pub const fn clear(&mut self) {
        *self.counter_words_mut() = [0u64; 8];
    }

    /// Returns `true` in $\mathcal{O}(1)$ if all 64 counter buckets are zero.
    #[inline(always)]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        !words_any_nonzero_64(self.counter_words_ref())
    }

    /// Halves all 64 8-bit counters in-place (`c_i >>= 1`) using unrolled 8-lane SWAR (`swar_u8_halve_decay_x8`).
    #[inline(always)]
    pub const fn halve_counters_swar(&mut self) {
        let w = *self.counter_words_ref();
        *self.counter_words_mut() = [
            swar_u8_halve_decay_x8(w[0]),
            swar_u8_halve_decay_x8(w[1]),
            swar_u8_halve_decay_x8(w[2]),
            swar_u8_halve_decay_x8(w[3]),
            swar_u8_halve_decay_x8(w[4]),
            swar_u8_halve_decay_x8(w[5]),
            swar_u8_halve_decay_x8(w[6]),
            swar_u8_halve_decay_x8(w[7]),
        ];
    }

    /// Evaluates whether the Counting Bloom Filter has crossed `max_mass_threshold` or
    /// [`DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS`] (`48` buckets) and halves all 8 SWAR words in-place
    /// when saturated, skipping redundant 64-byte stores when below threshold (`0` narrow-to-wide STLF stalls).
    #[inline(always)]
    pub const fn auto_halve_if_saturated(&mut self, max_mass_threshold: u32) -> bool {
        let w = *self.counter_words_ref();
        let mass = swar_total_mass_from_words(&w);
        let active_buckets = swar_active_buckets_from_words(&w);

        let saturated =
            (mass > max_mass_threshold) | (active_buckets > DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS);
        if saturated {
            *self.counter_words_mut() = [
                (w[0] >> 1) & SWAR_LOW7_MASK_64,
                (w[1] >> 1) & SWAR_LOW7_MASK_64,
                (w[2] >> 1) & SWAR_LOW7_MASK_64,
                (w[3] >> 1) & SWAR_LOW7_MASK_64,
                (w[4] >> 1) & SWAR_LOW7_MASK_64,
                (w[5] >> 1) & SWAR_LOW7_MASK_64,
                (w[6] >> 1) & SWAR_LOW7_MASK_64,
                (w[7] >> 1) & SWAR_LOW7_MASK_64,
            ];
        }
        saturated
    }

    /// Computes 4 independent 6-bit slot indices ($0..64$) by folding all 64 Q7 head phases
    /// across 4 parallel superscalar multiply lanes in $\mathcal{O}(1)$ directly from 8 SWAR words.
    #[inline(always)]
    #[must_use]
    pub const fn compute_hash_slots_from_words(w: &[u64; 8]) -> [usize; 4] {
        const K0: u64 = 0x9E37_79B9_7F4A_7C15;
        const K1: u64 = 0xBF58_476D_1CE4_E5B9;
        const K2: u64 = 0x94D0_49BB_1331_11EB;
        const K3: u64 = 0xD6E8_FEB8_6659_FD93;

        let m0 = (w[0] ^ K0).wrapping_mul(K1) ^ w[4].rotate_left(17);
        let m1 = (w[1] ^ K1).wrapping_mul(K2) ^ w[5].rotate_left(29);
        let m2 = (w[2] ^ K2).wrapping_mul(K3) ^ w[6].rotate_left(41);
        let m3 = (w[3] ^ K3).wrapping_mul(K0) ^ w[7].rotate_left(53);

        let f01 = (m0 ^ m1.rotate_left(13)).wrapping_mul(K1);
        let f23 = (m2 ^ m3.rotate_left(37)).wrapping_mul(K2);
        let mix = branchless_splitmix64(f01.wrapping_add(f23));
        [
            (mix & 0x3F) as usize,
            ((mix >> 16) & 0x3F) as usize,
            ((mix >> 32) & 0x3F) as usize,
            ((mix >> 48) & 0x3F) as usize,
        ]
    }

    /// Computes 4 independent 6-bit slot indices ($0..64$) by folding all 64 Q7 head phases
    /// across 4 parallel superscalar multiply lanes in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn compute_hash_slots(phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> [usize; 4] {
        Self::compute_hash_slots_from_words(&load_q7_words_64(phase_pattern))
    }

    /// Queries the minimum collision count (repulsive potential intensity) directly from 8 SWAR words,
    /// short-circuiting in $\mathcal{O}(1)$ when the 64-byte Bloom filter is empty (`0` 64-bit multiplies).
    #[inline(always)]
    #[must_use]
    pub const fn query_repulsive_density_from_words(&self, w: &[u64; 8]) -> u8 {
        if self.counters[0] == 0 && self.is_empty() {
            return 0;
        }
        let slots = Self::compute_hash_slots_from_words(w);
        let c0 = self.counters[slots[0]];
        let c1 = self.counters[slots[1]];
        let c2 = self.counters[slots[2]];
        let c3 = self.counters[slots[3]];
        let m01 = branchless_select_u8(c0 < c1, c0, c1);
        let m23 = branchless_select_u8(c2 < c3, c2, c3);
        branchless_select_u8(m01 < m23, m01, m23)
    }

    /// Applies Phase-Conjugate Attractor Repulsion ($-\boldsymbol{\Delta\theta}$) in-place directly
    /// on 8 register-resident SWAR words `words: &mut [u64; 8]` in strictly constant time (`0` branches),
    /// fast-pathing word-uniform 64-head profiles (`1` SWAR evaluation instead of `8`) and
    /// returning `(density, repulsion_triggered)`.
    #[inline(always)]
    pub const fn apply_phase_conjugate_repulsion_words(
        &self,
        words: &mut [u64; 8],
        density_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (u8, bool) {
        let density = self.query_repulsive_density_from_words(words);
        let active = (density >= density_threshold) & (density > 0);
        if !active {
            return (density, false);
        }
        let gt_128 = repulsion_scale_u7 > 128;
        let s_sub = branchless_select_u8(
            gt_128,
            repulsion_scale_u7.wrapping_sub(128),
            repulsion_scale_u7,
        ) as u64;
        let bias_restore = (128u64.wrapping_sub(s_sub)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let extra_mask = (gt_128 as u64).wrapping_neg();

        if is_word_array_uniform_64(words) {
            let neg0 = swar_q7_neg_sat_x8(words[0]);
            let rep0 = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(neg0, s_sub, bias_restore),
                neg0 & extra_mask,
            );
            *words = [rep0; 8];
            return (density, true);
        }

        let mut c = 0usize;
        while c < 8 {
            let cur = words[c];
            let neg = swar_q7_neg_sat_x8(cur);
            words[c] = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(neg, s_sub, bias_restore),
                neg & extra_mask,
            );
            c += 1;
        }
        (density, true)
    }

    /// Records a spurious local minimum phase pattern directly from 8 register-resident `u64` SWAR
    /// words with a custom saturating weight step (`0` stack reload, `0` branches).
    #[inline(always)]
    pub const fn insert_repulsive_pattern_words_weighted(
        &mut self,
        words: &[u64; 8],
        weight: u8,
    ) {
        let _ = self.auto_halve_if_saturated(DEFAULT_BLOOM_MAX_MASS_THRESHOLD);
        let slots = Self::compute_hash_slots_from_words(words);
        self.counters[slots[0]] = self.counters[slots[0]].saturating_add(weight);
        self.counters[slots[1]] = self.counters[slots[1]].saturating_add(weight);
        self.counters[slots[2]] = self.counters[slots[2]].saturating_add(weight);
        self.counters[slots[3]] = self.counters[slots[3]].saturating_add(weight);
    }

    /// Records a spurious local minimum phase pattern into the Counting Bloom Filter in strictly
    /// constant time (`0` branches).
    #[inline(always)]
    pub const fn insert_repulsive_pattern(
        &mut self,
        phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) {
        self.insert_repulsive_pattern_weighted(phase_pattern, 1);
    }

    /// Records a spurious local minimum phase pattern with a custom saturating weight step,
    /// performing branchless constant-time SWAR self-normalization (`auto_halve_if_saturated`).
    #[inline(always)]
    pub const fn insert_repulsive_pattern_weighted(
        &mut self,
        phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight: u8,
    ) {
        self.insert_repulsive_pattern_words_weighted(&load_q7_words_64(phase_pattern), weight);
    }

    /// Queries the minimum collision count (repulsive potential intensity) in strictly constant time (`0` branches).
    #[inline(always)]
    #[must_use]
    pub const fn query_repulsive_density(
        &self,
        phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> u8 {
        self.query_repulsive_density_from_words(&load_q7_words_64(phase_pattern))
    }

    /// Evaluates normalized repulsive potential energy in floating-point domain ($\in [0, \mathrm{gain}]$).
    #[inline(always)]
    #[must_use]
    pub const fn query_repulsion_potential(
        &self,
        phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        gain: f32,
    ) -> f32 {
        let density = self.query_repulsive_density(phase_pattern) as f32;
        branchless_relu_f32(gain) * (density * (1.0 / 255.0))
    }

    /// Applies Phase-Conjugate Attractor Repulsion ($-\boldsymbol{\Delta\theta}$) in-place across
    /// `candidate_phases` and returns `(density, repulsion_triggered, post_repulsion_words)` directly
    /// in 8 `u64` SWAR registers (`0` stack reload for downstream QDPS/RIPA or unified-field kernels).
    #[inline(always)]
    pub const fn apply_phase_conjugate_repulsion_to_words(
        &self,
        candidate_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        density_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (u8, bool, [u64; 8]) {
        let mut words = load_q7_words_64(candidate_phases);
        let (density, active) = self.apply_phase_conjugate_repulsion_words(
            &mut words,
            density_threshold,
            repulsion_scale_u7,
        );
        if active {
            store_q7_words_64(candidate_phases, &words);
        }
        (density, active, words)
    }

    /// Applies Phase-Conjugate Attractor Repulsion ($-\boldsymbol{\Delta\theta}$) in-place across
    /// all 64 attention heads in strictly constant time (`0` branches) using `branchless_select_u64`
    /// and hoisted `bias_restore`.
    #[inline(always)]
    pub const fn apply_phase_conjugate_repulsion(
        &self,
        candidate_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        density_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> u8 {
        self.apply_phase_conjugate_repulsion_to_words(
            candidate_phases,
            density_threshold,
            repulsion_scale_u7,
        )
        .0
    }

    /// Evaluates `apply_phase_conjugate_repulsion` on `candidate_phases` and simultaneously evolves
    /// `tagged_ptr` into [`ACTION_PRESET_REPULSION_ESCAPE`] targeting [`ATTRACTOR_SLOT_REPULSION`] (`lane 3`).
    #[inline(always)]
    pub fn apply_repulsion_and_evolve_ptr(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        candidate_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        density_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (gcso_tagged_ptr_t, u8, bool) {
        let density = self.apply_phase_conjugate_repulsion(
            candidate_phases,
            density_threshold,
            repulsion_scale_u7,
        );
        let triggered = (density >= density_threshold) & (density > 0);
        let rep_mode = tagged_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
        let rep_ptr = tagged_ptr
            .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
            .conjugate_phase();
        let out_ptr = gcso_tagged_ptr_t(branchless_select_u64(
            triggered,
            rep_ptr.raw(),
            tagged_ptr.raw(),
        ));
        (out_ptr, density, triggered)
    }

    /// Applies evaporative decay across all 64 counters in-place in strictly constant time (`0` branches).
    #[inline(always)]
    pub const fn decay_counters(&mut self, decay_step: u8) {
        let w = *self.counter_words_ref();
        *self.counter_words_mut() = [
            swar_u8_sub_sat_x8(w[0], decay_step),
            swar_u8_sub_sat_x8(w[1], decay_step),
            swar_u8_sub_sat_x8(w[2], decay_step),
            swar_u8_sub_sat_x8(w[3], decay_step),
            swar_u8_sub_sat_x8(w[4], decay_step),
            swar_u8_sub_sat_x8(w[5], decay_step),
            swar_u8_sub_sat_x8(w[6], decay_step),
            swar_u8_sub_sat_x8(w[7], decay_step),
        ];
    }

    /// Merges another `ContrastivePhaseBloomFilter` into `self` via 8-lane SWAR saturating addition
    /// in strictly constant time (`0` branches).
    #[inline(always)]
    pub const fn merge_counters_sat(&mut self, other: &Self) {
        let a = *self.counter_words_ref();
        let b = *other.counter_words_ref();
        *self.counter_words_mut() = [
            swar_u8_add_sat_x8(a[0], b[0]),
            swar_u8_add_sat_x8(a[1], b[1]),
            swar_u8_add_sat_x8(a[2], b[2]),
            swar_u8_add_sat_x8(a[3], b[3]),
            swar_u8_add_sat_x8(a[4], b[4]),
            swar_u8_add_sat_x8(a[5], b[5]),
            swar_u8_add_sat_x8(a[6], b[6]),
            swar_u8_add_sat_x8(a[7], b[7]),
        ];
        let _ = self.auto_halve_if_saturated(DEFAULT_BLOOM_MAX_MASS_THRESHOLD);
    }

    /// Merges another `ContrastivePhaseBloomFilter` into `self` via 8-lane SWAR elementwise maximum
    /// in strictly constant time (`0` branches).
    #[inline(always)]
    pub const fn merge_counters_max(&mut self, other: &Self) {
        let a = *self.counter_words_ref();
        let b = *other.counter_words_ref();
        *self.counter_words_mut() = [
            swar_u8_max_x8(a[0], b[0]),
            swar_u8_max_x8(a[1], b[1]),
            swar_u8_max_x8(a[2], b[2]),
            swar_u8_max_x8(a[3], b[3]),
            swar_u8_max_x8(a[4], b[4]),
            swar_u8_max_x8(a[5], b[5]),
            swar_u8_max_x8(a[6], b[6]),
            swar_u8_max_x8(a[7], b[7]),
        ];
    }

    /// Returns the total accumulated counter mass across all 64 slots in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn total_mass(&self) -> u32 {
        swar_total_mass_from_words(self.counter_words_ref())
    }

    /// Returns the number of non-zero counter buckets (`0..=64`) currently active in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn active_bucket_count(&self) -> u32 {
        swar_active_buckets_from_words(self.counter_words_ref())
    }
}

// SAFETY: `ContrastivePhaseBloomFilter` is a 64-byte aligned array of `u8` with no padding traps.
unsafe impl Pod for ContrastivePhaseBloomFilter {}

// ===================================================================
// 5. Stigmergic Pointer Trail, Limina Bridging & Unified Resonance (`StigmergicTrailOps`)
// ===================================================================

/// Branchless $\mathcal{O}(1)$ stigmergic trail, pointer-mode orchestration, inter-memory Limina
/// bridging, Bayesian pointer-trail transition scoring & rewiring, single-manifold unified-field
/// anchor resonance (preserving 32-bit `target_anchor_id`), and 8-lane working neighborhood
/// synergy operations on 128-byte aligned [`gcso_pointer_trail_t`].
pub trait StigmergicTrailOps {
    /// Records an $\mathcal{O}(1)$ tagged pointer transition and reinforces stigmergic trail density.
    fn record_pointer_transition(&mut self, next_ptr: u64, edge_cost: i32, density_boost: f32);
    /// Accumulates a 64-head Q7 phase delta vector into the trail's phase state using branchless saturation.
    fn accumulate_phase_vector(&mut self, delta_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]);
    /// Accumulates a uniform Q7 phase delta across all 64 heads in-place using 8-lane SWAR saturation.
    fn accumulate_uniform_phase(&mut self, delta_q7: gcso_q7_t);
    /// Smoothly blends the trail's 64-head phase trajectory toward `target_phases` along $S^1$.
    fn geodesic_blend_phase_with(
        &mut self,
        target_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        t_u7: u8,
    );
    /// Stamps a phase-conjugate repulsion marker ($-\boldsymbol{\Delta\theta}$) onto the pointer trail.
    fn stamp_phase_conjugate_repulsion(&mut self);
    /// Applies subnormal-safe temporal evaporative decay (`decay_factor` $\in [0, 1]$) to trail density.
    fn apply_trail_evaporation(&mut self, decay_factor: f32);
    /// Evaluates whether the trail's stigmergic density exceeds `crystal_threshold`.
    fn should_crystallize_branchless(&self, crystal_threshold: f32) -> bool;
    /// Returns `current_ptr` wrapped as a strongly-typed [`gcso_tagged_ptr_t`].
    fn current_tagged_ptr(&self) -> gcso_tagged_ptr_t;
    /// Returns `prev_ptr` wrapped as a strongly-typed [`gcso_tagged_ptr_t`].
    fn prev_tagged_ptr(&self) -> gcso_tagged_ptr_t;
    /// Extracts the active 5-bit Action Mode bitmask (`ACTION_MODE_*`) embedded in `current_ptr`.
    fn active_action_mode(&self) -> u8;
    /// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`) embedded in `current_ptr`.
    fn target_attractor_slot(&self) -> u8;
    /// Extracts the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) embedded in `self.flags`.
    fn swarm_quorum_mask(&self) -> u8;
    /// Updates the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) and resolution/escalation flags.
    fn set_swarm_quorum_state(&mut self, quorum_mask: u8, hot_path_resolved: bool);
    /// Packs `(next_addr, attractor_slot_id, action_mode_flags, phase_q7)` and advances the trail.
    fn record_action_transition(
        &mut self,
        next_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: gcso_q7_t,
        edge_cost: i32,
        density_boost: f32,
    );
    /// Evolves the trail's `current_ptr` Action Mode flags and Target Working Lane ID bottom-up,
    /// preserving 32-bit `target_anchor_id` in `cluster_id` when `target_anchor_id != 0` and
    /// `!repulsion_triggered`.
    fn evolve_trail_mode_stigmergically(
        &mut self,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_srl_thresh: f32,
    ) -> gcso_tagged_ptr_t;
    /// Evaluates circular phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1.0, 1.0]$ with `slot_phases`.
    fn phase_coherence_with(&self, slot_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32;
    /// Evaluates the trail's 64-head circular resultant statistics $(\bar{R}_c, \bar{\theta}_{\mathrm{Q7}}, \kappa_c)$
    /// alongside its Phase-Proximity Gravity mass $M_{\mathrm{grav}} = \rho_{\mathrm{stigmergy}} / (1 - \bar{R}_c + \epsilon)$
    /// in $\mathcal{O}(1)$ division-free (`0 fdiv`) mode.
    fn evaluate_trail_circular_stats_and_gravity(&self) -> (f32, gcso_q7_t, f32, f32);
    /// Applies **Zero-Hyperparameter Phase-Proximity Resonance** from a single anchor on the
    /// unified phase manifold (`anchor_phases`, `anchor_base_energy`) in strictly constant time (`0` branches).
    fn apply_unified_field_anchor_resonance(
        &mut self,
        anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        anchor_base_energy: f32,
    ) -> (f32, f32);
    /// Applies **Order-Invariant (Commutative) Unified-Field Multi-Anchor Resonance** across an
    /// arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional repulsive
    /// phase inhibitor (`repulsion_phases_and_strength`) on a **single unified phase manifold**
    /// using register-resident 8-word SWAR accumulators (`0` branches, `0 fdiv`).
    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional
    /// repulsive phase inhibitor (`repulsion_phases_and_strength`) starting from pre-loaded 8-word
    /// SWAR registers `init_w` (`0` entry reload) and returning `(final_pull_force, mean_coherence, active_anchor_count, post_words)`.
    fn apply_unified_field_anchor_slice_resonance_from_words(
        &mut self,
        init_w: [u64; NUM_SWARM_MICRO_CELLS],
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS]);

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the single
    /// unified phase manifold and returns `(final_pull_force, mean_coherence, active_anchor_count, post_words)`
    /// directly in 8 `u64` SWAR registers (`0` exit reload).
    fn apply_unified_field_anchor_slice_resonance_to_words(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS]);

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional
    /// repulsive phase inhibitor (`repulsion_phases_and_strength`) on a **single unified phase manifold**
    /// using register-resident 8-word SWAR accumulators (`0` branches, `0 fdiv`).
    fn apply_unified_field_anchor_slice_resonance(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32);
    /// Bridges the "unwritten space between the lines" (**Limina / Inter-Memory Hallucination**)
    /// from `prev_memory_trail` ($M_{t-1}$) to `self` ($M_t$) in a **single fused 8-word SWAR pass**
    /// (`0` branches, zero temporary array stack spills).
    fn bridge_liminal_hallucinated_trail(
        &mut self,
        prev_memory_trail: &gcso_pointer_trail_t,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool);
    /// Links `self` ($M_{t-1}$) to `next_trail` ($M_t$, identified by `next_trail_id`) by setting
    /// `self.linked_trail_id = next_trail_id` and bridging their inter-memory **Limina** gap.
    fn link_with_next_hallucinated_trail(
        &mut self,
        next_trail_id: u32,
        next_trail: &mut gcso_pointer_trail_t,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool);
    /// Evaluates the **Bayes-Attention Stigmergic Transition Score** from `self` ($M_t$) to a
    /// candidate memory trail `candidate_trail` ($M_j$, identified by `candidate_trail_id`) in
    /// strictly constant time (`0` branches, `0 fdiv`).
    fn evaluate_bayesian_transition_score_with(
        &self,
        candidate_trail_id: u32,
        candidate_trail: &gcso_pointer_trail_t,
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool);
    /// Scans `candidate_trails` in constant time per candidate (`0` data-dependent branches) to
    /// select the optimal non-contradictory ($H^1(K; \mathcal{F}) = 0$) next memory trail $j^*$
    /// and autonomously rewires `self.linked_trail_id = j^*`.
    fn select_and_rewire_best_next_trail(
        &mut self,
        candidate_trails: &[gcso_pointer_trail_t],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
    ) -> (u32, f32, f32, bool);
    /// Synchronizes `self.flags` (`TRAIL_FLAG_DPSR_ACTIVE` and `TRAIL_FLAG_INERTIAL_SKIP_ACTIVE`)
    /// with the 5-bit `ACTION_MODE_*` flags embedded in `self.current_ptr` in branchless constant time.
    fn sync_flags_with_current_ptr(&mut self);
    /// Directly modulates pre-Softmax candidate memory Attention scores (`attention_scores[..M]`)
    /// in-place in zero-copy, zero-allocation ($C_{\mathrm{alloc}} = 0$) constant time per candidate
    /// with loop-hoisted invariants and register-resident SWAR words.
    fn modulate_candidate_trail_scores_in_place(
        &self,
        attention_scores: &mut [f32],
        candidate_trails: &[gcso_pointer_trail_t],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        repulsion_penalty: f32,
    ) -> (f32, u32, u32);
    /// Single-pass fused evaluator for **Bayesian Candidate Selection, Pointer Rewiring, Limina
    /// Geodesic Bridging, and Optional In-Place Attention Score Modulation**:
    /// eliminates duplicate candidate-trail scans across `dpsr.rs` and `edbc.rs`, tracking only
    /// scalar `best_idx` inside the loop to eliminate 8-word `best_w` register pressure.
    ///
    /// Returns `(selected_id, best_bayesian_score, mean_coherence, coherent_count, found_coherent)`.
    fn select_rewire_bridge_and_modulate_bayesian_trails(
        &mut self,
        candidate_trails: &[gcso_pointer_trail_t],
        attention_scores_opt: Option<&mut [f32]>,
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        liminal_interpolation_u7: u8,
        repulsion_penalty: f32,
    ) -> (u32, f32, f32, u32, bool);
    /// Single-pass Coarse-to-Fine fused evaluator uniting **Macro-Centroid Affinity
    /// ($\mathcal{A}_{\mathrm{macro}}(c^*, j)$), Bayesian Candidate Selection, Pointer Rewiring,
    /// Limina Geodesic Bridging, and Optional In-Place Attention Score Modulation** in a single
    /// pass over `candidate_trails` (`0` duplicate candidate scans).
    ///
    /// Returns `(selected_id, best_bayesian_score, mean_coherence, coherent_count, found_coherent)`.
    #[inline]
    fn select_rewire_bridge_and_modulate_with_macro_centroid_64(
        &mut self,
        candidate_trails: &[gcso_pointer_trail_t],
        attention_scores_opt: Option<&mut [f32]>,
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        liminal_interpolation_u7: u8,
        repulsion_penalty: f32,
        _macro_anchor_id: u32,
        _win_w_opt: Option<&[u64; NUM_SWARM_MICRO_CELLS]>,
        _cluster_bonus_gain: f32,
    ) -> (u32, f32, f32, u32, bool) {
        self.select_rewire_bridge_and_modulate_bayesian_trails(
            candidate_trails,
            attention_scores_opt,
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
            liminal_interpolation_u7,
            repulsion_penalty,
        )
    }
    /// Evaluates non-linear cross-slot synergy and liminal beat interference between two working lanes.
    fn apply_cross_slot_synergy(
        &mut self,
        slot_a: u32,
        phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight_a: f32,
        slot_b: u32,
        phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight_b: f32,
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
    ) -> (f32, f32);
    /// Applies branchless mutual inhibition gating to the trail's accumulated phase vector and pull force.
    fn apply_mutual_slot_inhibition(
        &mut self,
        inhibitor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        inhibition_strength: f32,
    ) -> f32;
    /// Register-resident evaluator for **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition**
    /// starting from pre-loaded 8-word SWAR registers `acc_w` and returning
    /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)`.
    fn apply_neighborhood_synergy_from_bank_from_words(
        &mut self,
        acc_w: [u64; NUM_SWARM_MICRO_CELLS],
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS]);

    /// Evaluates **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition** and returns both
    /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)` in registers (`0` exit reload).
    fn apply_neighborhood_synergy_from_bank_to_words(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS]);

    /// Unified single-source-of-truth evaluator for **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition**
    /// across `slot_phase_bank` (`[[gcso_q7_t; 64]; 8]`) and `slot_weights` (`[f32; 8]`).
    fn apply_neighborhood_synergy_from_bank(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32);
}

/// Register-resident Coarse-to-Fine Bayesian candidate trail selector, rewirer, Limina geodesic
/// bridger, and Attention score modulator accepting `self_w` directly in 8 `u64` SWAR registers
/// and returning the post-bridged 8 `u64` SWAR words (`0` redundant stack/memory reloads).
#[inline]
pub(crate) fn select_rewire_bridge_and_modulate_with_preloaded_words_64(
    trail: &mut gcso_pointer_trail_t,
    mut self_w: [u64; NUM_SWARM_MICRO_CELLS],
    candidate_trails: &[gcso_pointer_trail_t],
    mut attention_scores_opt: Option<&mut [f32]>,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
    liminal_interpolation_u7: u8,
    repulsion_penalty: f32,
    macro_anchor_id: u32,
    win_w_opt: Option<&[u64; NUM_SWARM_MICRO_CELLS]>,
    cluster_bonus_gain: f32,
) -> (u32, f32, f32, u32, bool, [u64; NUM_SWARM_MICRO_CELLS], f32) {
    let cand_len = candidate_trails.len();
    if cand_len == 0 {
        return (trail.linked_trail_id, 0.0, 0.0, 0, false, self_w, 0.0);
    }

    let (phase_scale, clean_eta, clean_link, safe_thresh) = precompute_bayesian_trail_scales(
        head_dim,
        phase_interference_gain,
        stigmergic_prior_gain,
        pointer_link_bonus,
        contradiction_threshold,
    );
    let clean_penalty = branchless_flush_subnormal_f32(branchless_relu_f32(repulsion_penalty));
    let clean_cluster_bonus =
        branchless_flush_subnormal_f32(branchless_relu_f32(cluster_bonus_gain));
    let self_is_uniform = is_word_array_uniform_64(&self_w);
    let self_b0 = (self_w[0] & 0xFF) as u8;
    let self_is_byte_uniform =
        self_is_uniform & (self_w[0] == self_w[0].rotate_right(8));
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (self_v0, self_v1) = unsafe {
        super::branchless::avx2_words64_to_vec64_with_uniform(&self_w, self_is_uniform)
    };
    let mut win_meta_evaluated = false;
    let mut win_eq_self = false;
    let mut win_is_uniform = false;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (mut win_v0, mut win_v1) = (self_v0, self_v1);
    let cur_linked_id = trail.linked_trail_id;

    let mut best_id = cur_linked_id;
    let mut best_idx = 0usize;
    let mut best_score = -1.0e30_f32;
    let mut best_coh = 0.0_f32;
    let mut best_is_uniform = self_is_uniform;
    let mut found_valid = false;
    let mut coh_sum = 0.0_f32;
    let mut coherent_count = 0u32;

    let mut idx = 0usize;
    while idx < cand_len {
        let cand = &candidate_trails[idx];
        let cand_id = idx as u32;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (bayes_delta, coh, is_coherent, cand_is_uniform) = unsafe {
            eval_candidate_trail_avx2_with_macro_fast(
                &self_w,
                self_is_uniform,
                self_is_byte_uniform,
                self_b0,
                self_v0,
                self_v1,
                cur_linked_id,
                cand_id,
                cand,
                phase_scale,
                clean_eta,
                clean_link,
                safe_thresh,
                macro_anchor_id,
                win_w_opt,
                &mut win_meta_evaluated,
                &mut win_is_uniform,
                &mut win_eq_self,
                &mut win_v0,
                &mut win_v1,
                clean_cluster_bonus,
            )
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let (bayes_delta, coh, is_coherent, cand_is_uniform) =
            eval_candidate_trail_words_with_macro_fast(
                &self_w,
                self_is_uniform,
                self_is_byte_uniform,
                self_b0,
                cur_linked_id,
                cand_id,
                cand,
                phase_scale,
                clean_eta,
                clean_link,
                safe_thresh,
                macro_anchor_id,
                win_w_opt,
                &mut win_meta_evaluated,
                &mut win_is_uniform,
                &mut win_eq_self,
                clean_cluster_bonus,
            );

        if let Some(ref mut scores) = attention_scores_opt {
            if idx < scores.len() {
                let penalty = branchless_select_f32(is_coherent, 0.0, clean_penalty);
                scores[idx] =
                    branchless_flush_subnormal_f32(scores[idx] + bayes_delta - penalty);
            }
        }

        let take = is_coherent & ((!found_valid) | (bayes_delta > best_score));
        if take {
            best_id = cand_id;
            best_idx = idx;
            best_score = bayes_delta;
            best_coh = coh;
            best_is_uniform = cand_is_uniform;
        }
        found_valid |= is_coherent;
        coh_sum += coh;
        coherent_count = coherent_count.wrapping_add(u32::from(is_coherent));
        idx += 1;
    }

    if found_valid {
        trail.linked_trail_id = best_id;
        let best_cand = &candidate_trails[best_idx];
        let eff_s = branchless_select_u8(
            liminal_interpolation_u7 > 128,
            128,
            liminal_interpolation_u7,
        ) as u64;
        if eff_s > 0 {
            let best_w = trail_phase_words_ref(best_cand);
            if eff_s >= 128 {
                self_w = *best_w;
            } else {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    if self_is_uniform & best_is_uniform {
                        let bias_restore =
                            (128u64.wrapping_sub(eff_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        self_w = swar_slerp_words_with_uniform_flag_64(
                            &self_w,
                            true,
                            best_w,
                            true,
                            eff_s,
                            bias_restore,
                        )
                        .0;
                    } else {
                        // SAFETY: `(self_v0, self_v1)` are already hoisted in YMM registers and AVX2 is enabled.
                        unsafe {
                            use core::arch::x86_64::{__m256i, _mm256_loadu_si256};
                            let b_ptr = best_cand.accumulated_phase_delta.as_ptr().cast::<__m256i>();
                            let bv0 = _mm256_loadu_si256(b_ptr);
                            let bv1 = _mm256_loadu_si256(b_ptr.add(1));
                            let (rv0, rv1) = super::branchless::avx2_slerp_vec64(
                                self_v0, self_v1, bv0, bv1, eff_s,
                            );
                            self_w = super::branchless::avx2_vec64_to_words64(rv0, rv1);
                        }
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    let bias_restore =
                        (128u64.wrapping_sub(eff_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    self_w = swar_slerp_words_with_uniform_flag_64(
                        &self_w,
                        self_is_uniform,
                        best_w,
                        best_is_uniform,
                        eff_s,
                        bias_restore,
                    )
                    .0;
                }
            }
            *trail_phase_words_mut(trail) = self_w;
        }

        let best_anchor_id = best_cand.target_anchor_id;
        let adopt_anchor = (trail.target_anchor_id == 0) & (best_anchor_id != 0);
        trail.target_anchor_id =
            branchless_select_u32(adopt_anchor, best_anchor_id, trail.target_anchor_id);
        trail.cluster_id = branchless_select_u32(
            trail.target_anchor_id != 0,
            trail.target_anchor_id,
            trail.cluster_id,
        );

        if let Some(scores) = attention_scores_opt {
            if best_idx < scores.len() && cur_linked_id != best_id {
                scores[best_idx] =
                    branchless_flush_subnormal_f32(scores[best_idx] + clean_link);
            }
        }
    }

    let mean_coh = branchless_clamp_f32(coh_sum * fast_inv_len_f32(cand_len), -1.0, 1.0);
    let out_score = branchless_select_f32(found_valid, best_score, 0.0);
    let out_best_coh = branchless_select_f32(found_valid, best_coh, 0.0);
    (
        best_id,
        out_score,
        mean_coh,
        coherent_count,
        found_valid,
        self_w,
        out_best_coh,
    )
}

impl StigmergicTrailOps for gcso_pointer_trail_t {
    #[inline(always)]
    fn record_pointer_transition(&mut self, next_ptr: u64, edge_cost: i32, density_boost: f32) {
        self.prev_ptr = self.current_ptr;
        self.current_ptr = next_ptr;
        self.transition_cost = self.transition_cost.saturating_add(edge_cost);
        self.step_count = self.step_count.wrapping_add(1);
        let raw_rho = branchless_flush_subnormal_f32(branchless_relu_f32(self.stigmergic_density))
            + branchless_flush_subnormal_f32(branchless_relu_f32(density_boost));
        self.stigmergic_density = branchless_min_f32(raw_rho, MAX_STIGMERGIC_DENSITY);
    }

    #[inline(always)]
    fn accumulate_phase_vector(&mut self, delta_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) {
        branchless_q7_accumulate_64(&mut self.accumulated_phase_delta, delta_phases);
    }

    #[inline(always)]
    fn accumulate_uniform_phase(&mut self, delta_q7: gcso_q7_t) {
        if delta_q7 == 0 {
            return;
        }
        let splat_u64 = ((delta_q7 as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let w = trail_phase_words_mut(self);
        w[0] = swar_q7_add_sat_x8(w[0], splat_u64);
        w[1] = swar_q7_add_sat_x8(w[1], splat_u64);
        w[2] = swar_q7_add_sat_x8(w[2], splat_u64);
        w[3] = swar_q7_add_sat_x8(w[3], splat_u64);
        w[4] = swar_q7_add_sat_x8(w[4], splat_u64);
        w[5] = swar_q7_add_sat_x8(w[5], splat_u64);
        w[6] = swar_q7_add_sat_x8(w[6], splat_u64);
        w[7] = swar_q7_add_sat_x8(w[7], splat_u64);
    }

    #[inline(always)]
    fn geodesic_blend_phase_with(
        &mut self,
        target_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        t_u7: u8,
    ) {
        if t_u7 == 0 {
            return;
        }
        let b = load_q7_words_64(target_phases);
        if t_u7 >= 128 {
            *trail_phase_words_mut(self) = b;
            return;
        }
        let s = u64::from(t_u7);
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let a = *trail_phase_words_ref(self);
        *trail_phase_words_mut(self) = swar_slerp_words_with_bias_64(&a, &b, s, bias_restore);
    }

    #[inline(always)]
    fn stamp_phase_conjugate_repulsion(&mut self) {
        let w = trail_phase_words_mut(self);
        w[0] = swar_q7_neg_sat_x8(w[0]);
        w[1] = swar_q7_neg_sat_x8(w[1]);
        w[2] = swar_q7_neg_sat_x8(w[2]);
        w[3] = swar_q7_neg_sat_x8(w[3]);
        w[4] = swar_q7_neg_sat_x8(w[4]);
        w[5] = swar_q7_neg_sat_x8(w[5]);
        w[6] = swar_q7_neg_sat_x8(w[6]);
        w[7] = swar_q7_neg_sat_x8(w[7]);
        self.current_ptr = gcso_tagged_ptr_t(self.current_ptr).conjugate_phase().raw();
        self.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
    }

    #[inline(always)]
    fn apply_trail_evaporation(&mut self, decay_factor: f32) {
        let clean_decay =
            branchless_select_f32(decay_factor > 1.0, 1.0, branchless_relu_f32(decay_factor));
        let decayed = branchless_relu_f32(self.stigmergic_density) * clean_decay;
        self.stigmergic_density =
            branchless_min_f32(branchless_flush_subnormal_f32(decayed), MAX_STIGMERGIC_DENSITY);
    }

    #[inline(always)]
    fn should_crystallize_branchless(&self, crystal_threshold: f32) -> bool {
        (self.stigmergic_density >= crystal_threshold) & (crystal_threshold > 0.0)
    }

    #[inline(always)]
    fn current_tagged_ptr(&self) -> gcso_tagged_ptr_t {
        gcso_tagged_ptr_t(self.current_ptr)
    }
    #[inline(always)]
    fn prev_tagged_ptr(&self) -> gcso_tagged_ptr_t {
        gcso_tagged_ptr_t(self.prev_ptr)
    }
    #[inline(always)]
    fn active_action_mode(&self) -> u8 {
        gcso_tagged_ptr_t(self.current_ptr).action_mode_flags()
    }
    #[inline(always)]
    fn target_attractor_slot(&self) -> u8 {
        gcso_tagged_ptr_t(self.current_ptr).target_attractor_slot()
    }
    #[inline(always)]
    fn swarm_quorum_mask(&self) -> u8 {
        ((self.flags & TRAIL_SWARM_QUORUM_MASK) >> TRAIL_SWARM_QUORUM_SHIFT) as u8
    }

    #[inline(always)]
    fn set_swarm_quorum_state(&mut self, quorum_mask: u8, hot_path_resolved: bool) {
        let cleared = self.flags
            & !(TRAIL_SWARM_QUORUM_MASK | TRAIL_FLAG_SWARM_RESOLVED | TRAIL_FLAG_SWARM_ESCALATED);
        let status_bit = branchless_select_u32(
            hot_path_resolved,
            TRAIL_FLAG_SWARM_RESOLVED,
            TRAIL_FLAG_SWARM_ESCALATED,
        );
        self.flags = cleared | (u32::from(quorum_mask) << TRAIL_SWARM_QUORUM_SHIFT) | status_bit;
    }

    #[inline(always)]
    fn record_action_transition(
        &mut self,
        next_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: gcso_q7_t,
        edge_cost: i32,
        density_boost: f32,
    ) {
        let packed = gcso_tagged_ptr_t::pack_action(
            next_addr,
            attractor_slot_id,
            action_mode_flags,
            phase_q7,
        );
        self.record_pointer_transition(packed.raw(), edge_cost, density_boost);
        self.cluster_id = u32::from(attractor_slot_id & ATTRACTOR_SLOT_ID_MASK);
        let dpsr_on = (action_mode_flags & ACTION_MODE_DPSR_STEER) != 0;
        let skip_on = (action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags |= branchless_select_u32(dpsr_on, TRAIL_FLAG_DPSR_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
    }

    #[inline(always)]
    fn evolve_trail_mode_stigmergically(
        &mut self,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_srl_thresh: f32,
    ) -> gcso_tagged_ptr_t {
        let evolved = gcso_tagged_ptr_t(self.current_ptr).evolve_mode_stigmergically(
            self.stigmergic_density,
            entropy_h_tilde,
            repulsion_triggered,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );
        self.current_ptr = evolved.raw();
        let preserve_unified_anchor = (self.target_anchor_id != 0) & (!repulsion_triggered);
        self.cluster_id = branchless_select_u32(
            preserve_unified_anchor,
            self.target_anchor_id,
            u32::from(evolved.target_attractor_slot()),
        );

        let dpsr_active = (evolved.action_mode_flags() & ACTION_MODE_DPSR_STEER) != 0;
        let skip_active = (evolved.action_mode_flags() & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags = (self.flags & !(TRAIL_FLAG_DPSR_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | branchless_select_u32(dpsr_active, TRAIL_FLAG_DPSR_ACTIVE, 0)
            | branchless_select_u32(skip_active, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0)
            | branchless_select_u32(repulsion_triggered, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        evolved
    }

    #[inline(always)]
    fn phase_coherence_with(&self, slot_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        branchless_q7_coherence_64(&self.accumulated_phase_delta, slot_phases)
    }

    #[inline(always)]
    fn evaluate_trail_circular_stats_and_gravity(&self) -> (f32, gcso_q7_t, f32, f32) {
        let (r_bar, mean_q7, kappa) =
            branchless_q7_circular_resultant_64(&self.accumulated_phase_delta);
        let gravity = branchless_phase_proximity_gravity_f32(
            self.stigmergic_density,
            r_bar,
            DEFAULT_PHASE_GRAVITY_EPS,
        );
        (r_bar, mean_q7, kappa, gravity)
    }

    #[inline]
    fn apply_unified_field_anchor_resonance(
        &mut self,
        anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        anchor_base_energy: f32,
    ) -> (f32, f32) {
        let (resonant_pull, coherence) = branchless_q7_resonant_pull_64(
            &self.accumulated_phase_delta,
            anchor_phases,
            anchor_base_energy,
        );
        let cur_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(self.attractor_pull_force));
        let new_pull = branchless_q7_circular_weighted_step_64(
            &mut self.accumulated_phase_delta,
            cur_pull,
            anchor_phases,
            resonant_pull,
        );
        self.attractor_pull_force = branchless_min_f32(new_pull, MAX_ATTRACTOR_PULL_FORCE);
        let is_resonant = (coherence > 0.25) & (resonant_pull > 0.0);
        let is_liminal = (coherence < -0.10) & (resonant_pull > 0.0);
        self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(is_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);
        (resonant_pull, coherence)
    }

    #[inline]
    fn apply_unified_field_anchor_slice_resonance_from_words(
        &mut self,
        init_w: [u64; NUM_SWARM_MICRO_CELLS],
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS]) {
        let (
            mut run_w,
            running_pull,
            mean_coh,
            dominant_anchor_id,
            active_anchors,
            any_resonant,
            any_liminal,
        ) = branchless_q7_unified_field_fold_words_with_flags_64(
            &init_w,
            self.attractor_pull_force,
            self.target_anchor_id,
            anchor_phases,
            anchor_base_energies,
        );
        let mut final_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(running_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        let has_active = active_anchors > 0;

        self.target_anchor_id = dominant_anchor_id;
        let lane_tag = (dominant_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK;
        let cur_ptr = gcso_tagged_ptr_t(self.current_ptr);
        let updated_ptr = cur_ptr.with_action_and_slot(
            branchless_select_u8(has_active, lane_tag, cur_ptr.target_attractor_slot()),
            cur_ptr.action_mode_flags()
                | branchless_select_u8(has_active, ACTION_MODE_DPSR_STEER, 0),
        );
        self.current_ptr = updated_ptr.raw();
        self.cluster_id = branchless_select_u32(has_active, dominant_anchor_id, self.cluster_id);
        self.flags |= branchless_select_u32(any_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(any_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        if let Some((rep_phases, inh_strength)) = repulsion_phases_and_strength {
            let rep_w = load_q7_words_64(rep_phases);
            let raw_damped =
                branchless_q7_inhibit_words_64(&mut run_w, final_pull, &rep_w, inh_strength);
            final_pull = branchless_min_f32(
                branchless_flush_subnormal_f32(raw_damped),
                MAX_ATTRACTOR_PULL_FORCE,
            );
        }

        *trail_phase_words_mut(self) = run_w;
        self.attractor_pull_force = final_pull;
        (final_pull, mean_coh, active_anchors, run_w)
    }

    #[inline(always)]
    fn apply_unified_field_anchor_slice_resonance_to_words(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS]) {
        let init_w = *trail_phase_words_ref(self);
        self.apply_unified_field_anchor_slice_resonance_from_words(
            init_w,
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
        )
    }

    #[inline(always)]
    fn apply_unified_field_anchor_slice_resonance(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32) {
        let (final_pull, mean_coh, active_anchors, _words) = self
            .apply_unified_field_anchor_slice_resonance_to_words(
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
            );
        (final_pull, mean_coh, active_anchors)
    }

    #[inline]
    fn bridge_liminal_hallucinated_trail(
        &mut self,
        prev_memory_trail: &gcso_pointer_trail_t,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool) {
        let p = trail_phase_words_ref(prev_memory_trail);
        let s = trail_phase_words_ref(self);
        let both_uni = is_word_array_uniform_64(p) && is_word_array_uniform_64(s);
        let d0 = swar_q7_circular_diff_x8(s[0], p[0]);
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let t_s = branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;
        let bias_restore = (128u64.wrapping_sub(t_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);

        let (coh, log_prob, contradiction, out_w, store_words) = if both_uni {
            if d0 == 0 {
                (1.0_f32, 0.0_f32, false, *s, false)
            } else {
                let c_val = coherence_from_uniform_diff_word_x8(d0);
                let prob = branchless_max_f32(0.5 * (1.0 + c_val), 1.0e-6);
                let lp = branchless_ln_f32(prob);
                let contr = c_val < -safe_thresh;
                let w0 = if contr {
                    swar_q7_neg_sat_x8(s[0])
                } else if t_s >= 128 {
                    s[0]
                } else if t_s == 0 {
                    p[0]
                } else {
                    swar_q7_circular_add_x8(
                        p[0],
                        swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore),
                    )
                };
                (c_val, lp, contr, [w0; 8], w0 != s[0])
            }
        } else {
            // Single-pass circular differences (M_t - M_{t-1}) across all 8 SWAR words:
            let d1 = swar_q7_circular_diff_x8(s[1], p[1]);
            let d2 = swar_q7_circular_diff_x8(s[2], p[2]);
            let d3 = swar_q7_circular_diff_x8(s[3], p[3]);
            let d4 = swar_q7_circular_diff_x8(s[4], p[4]);
            let d5 = swar_q7_circular_diff_x8(s[5], p[5]);
            let d6 = swar_q7_circular_diff_x8(s[6], p[6]);
            let d7 = swar_q7_circular_diff_x8(s[7], p[7]);

            if ((d0 | d1) | (d2 | d3) | (d4 | d5) | (d6 | d7)) == 0 {
                (1.0_f32, 0.0_f32, false, *s, false)
            } else {
                let c_val = coherence_from_diff_words_64(d0, d1, d2, d3, d4, d5, d6, d7);
                let prob = branchless_max_f32(0.5 * (1.0 + c_val), 1.0e-6);
                let lp = branchless_ln_f32(prob);
                let contr = c_val < -safe_thresh;

                if contr {
                    let w_arr = [
                        swar_q7_neg_sat_x8(s[0]),
                        swar_q7_neg_sat_x8(s[1]),
                        swar_q7_neg_sat_x8(s[2]),
                        swar_q7_neg_sat_x8(s[3]),
                        swar_q7_neg_sat_x8(s[4]),
                        swar_q7_neg_sat_x8(s[5]),
                        swar_q7_neg_sat_x8(s[6]),
                        swar_q7_neg_sat_x8(s[7]),
                    ];
                    (c_val, lp, true, w_arr, true)
                } else if t_s >= 128 {
                    (c_val, lp, false, *s, false)
                } else if t_s == 0 {
                    (c_val, lp, false, *p, true)
                } else {
                    let w_arr = [
                        swar_q7_circular_add_x8(
                            p[0],
                            swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[1],
                            swar_q7_scale_le128_with_bias_x8(d1, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[2],
                            swar_q7_scale_le128_with_bias_x8(d2, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[3],
                            swar_q7_scale_le128_with_bias_x8(d3, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[4],
                            swar_q7_scale_le128_with_bias_x8(d4, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[5],
                            swar_q7_scale_le128_with_bias_x8(d5, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[6],
                            swar_q7_scale_le128_with_bias_x8(d6, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[7],
                            swar_q7_scale_le128_with_bias_x8(d7, t_s, bias_restore),
                        ),
                    ];
                    (c_val, lp, false, w_arr, true)
                }
            }
        };
        if store_words {
            *trail_phase_words_mut(self) = out_w;
        }

        self.prev_ptr = prev_memory_trail.current_ptr;
        self.step_count = prev_memory_trail.step_count.wrapping_add(1);
        let inherit_anchor = (self.target_anchor_id == 0) & (!contradiction);
        self.target_anchor_id = branchless_select_u32(
            inherit_anchor,
            prev_memory_trail.target_anchor_id,
            self.target_anchor_id,
        );
        self.cluster_id = branchless_select_u32(
            inherit_anchor & (prev_memory_trail.target_anchor_id != 0),
            prev_memory_trail.target_anchor_id,
            self.cluster_id,
        );

        let cur_ptr = gcso_tagged_ptr_t(self.current_ptr);
        let rep_ptr = cur_ptr
            .with_action_and_slot(
                ATTRACTOR_SLOT_REPULSION,
                cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE,
            )
            .conjugate_phase();
        self.current_ptr = branchless_select_u64(contradiction, rep_ptr.raw(), cur_ptr.raw());

        let cur_rho = branchless_flush_subnormal_f32(branchless_relu_f32(self.stigmergic_density));
        let pos_rho = cur_rho + 0.25 * (1.0 + branchless_relu_f32(coh));
        let neg_rho = cur_rho * 0.5;
        self.stigmergic_density = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_select_f32(contradiction, neg_rho, pos_rho)),
            MAX_STIGMERGIC_DENSITY,
        );

        let prev_pull = branchless_flush_subnormal_f32(branchless_relu_f32(
            prev_memory_trail.attractor_pull_force,
        ));
        let cur_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(self.attractor_pull_force));
        let bridged_pull = cur_pull + 0.5 * prev_pull * branchless_clamp_f32(1.0 + coh, 0.25, 2.0);
        self.attractor_pull_force = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_select_f32(
                contradiction,
                cur_pull * 0.5,
                bridged_pull,
            )),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let is_resonant = (!contradiction) & (coh > 0.25);
        self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(
                contradiction,
                TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
                0,
            );
        (coh, log_prob, contradiction)
    }

    #[inline(always)]
    fn link_with_next_hallucinated_trail(
        &mut self,
        next_trail_id: u32,
        next_trail: &mut gcso_pointer_trail_t,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool) {
        self.linked_trail_id = next_trail_id;
        next_trail.bridge_liminal_hallucinated_trail(
            self,
            interpolation_u7,
            contradiction_threshold,
        )
    }

    #[inline(always)]
    fn evaluate_bayesian_transition_score_with(
        &self,
        candidate_trail_id: u32,
        candidate_trail: &gcso_pointer_trail_t,
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool) {
        let is_linked = (self.linked_trail_id != 0) & (self.linked_trail_id == candidate_trail_id);
        let (coh, total_score) = branchless_q7_bayesian_attention_score_delta_64(
            &self.accumulated_phase_delta,
            &candidate_trail.accumulated_phase_delta,
            candidate_trail.stigmergic_density,
            is_linked,
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
        );
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let cand_not_repulsed = (candidate_trail.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
        (total_score, coh, is_coherent)
    }

    #[inline]
    fn select_and_rewire_best_next_trail(
        &mut self,
        candidate_trails: &[gcso_pointer_trail_t],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
    ) -> (u32, f32, f32, bool) {
        let mut best_id = self.linked_trail_id;
        let mut best_score = -1.0e30_f32;
        let mut best_coh = 0.0_f32;
        let mut found_valid = false;

        let (phase_scale, clean_eta, clean_link, safe_thresh) = precompute_bayesian_trail_scales(
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
        );
        let self_w = trail_phase_words_ref(self);
        let self_is_uniform = is_word_array_uniform_64(self_w);
        let self_b0 = (self_w[0] & 0xFF) as u8;
        let self_is_byte_uniform =
            self_is_uniform & (self_w[0] == self_w[0].rotate_right(8));
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (self_v0, self_v1) = unsafe {
            super::branchless::avx2_words64_to_vec64_with_uniform(self_w, self_is_uniform)
        };
        let cur_linked_id = self.linked_trail_id;

        let mut idx = 0usize;
        while idx < candidate_trails.len() {
            if (candidate_trails[idx].flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0 {
                idx += 1;
                continue;
            }
            let cand_id = idx as u32;
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            let (score, coh, is_coherent) = unsafe {
                let mut win_meta_evaluated = true;
                let mut win_is_uniform = false;
                let mut win_eq_self = false;
                let (mut wv0, mut wv1) = (self_v0, self_v1);
                let (sc, ch, coh_ok, _) = eval_candidate_trail_avx2_with_macro_fast(
                    self_w,
                    self_is_uniform,
                    self_is_byte_uniform,
                    self_b0,
                    self_v0,
                    self_v1,
                    cur_linked_id,
                    cand_id,
                    &candidate_trails[idx],
                    phase_scale,
                    clean_eta,
                    clean_link,
                    safe_thresh,
                    0,
                    None,
                    &mut win_meta_evaluated,
                    &mut win_is_uniform,
                    &mut win_eq_self,
                    &mut wv0,
                    &mut wv1,
                    0.0,
                );
                (sc, ch, coh_ok)
            };
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            let (score, coh, is_coherent) = eval_candidate_trail_words_fast(
                self_w,
                self_is_uniform,
                self_is_byte_uniform,
                self_b0,
                cur_linked_id,
                cand_id,
                &candidate_trails[idx],
                phase_scale,
                clean_eta,
                clean_link,
                safe_thresh,
            );
            let take = is_coherent & ((!found_valid) | (score > best_score));
            best_id = branchless_select_u32(take, cand_id, best_id);
            best_score = branchless_select_f32(take, score, best_score);
            best_coh = branchless_select_f32(take, coh, best_coh);
            found_valid |= is_coherent;
            idx += 1;
        }

        self.linked_trail_id = branchless_select_u32(found_valid, best_id, self.linked_trail_id);
        let out_score = branchless_select_f32(found_valid, best_score, 0.0);
        (best_id, out_score, best_coh, found_valid)
    }

    #[inline(always)]
    fn sync_flags_with_current_ptr(&mut self) {
        let mode = unpack_action_mode_from_tag(gcso_tagged_ptr_t(self.current_ptr).unpack_tag());
        let dpsr_on = (mode & ACTION_MODE_DPSR_STEER) != 0;
        let skip_on = (mode & ACTION_MODE_LAYER_SKIP) != 0;
        let cleared = self.flags & !(TRAIL_FLAG_DPSR_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE);
        self.flags = cleared
            | branchless_select_u32(dpsr_on, TRAIL_FLAG_DPSR_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
    }

    #[inline]
    fn modulate_candidate_trail_scores_in_place(
        &self,
        attention_scores: &mut [f32],
        candidate_trails: &[gcso_pointer_trail_t],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        repulsion_penalty: f32,
    ) -> (f32, u32, u32) {
        let count = attention_scores.len().min(candidate_trails.len());
        if count == 0 {
            return (0.0, 0, 0);
        }
        let (phase_scale, clean_eta, clean_link, safe_thresh) = precompute_bayesian_trail_scales(
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
        );
        let clean_penalty = branchless_flush_subnormal_f32(branchless_relu_f32(repulsion_penalty));
        let self_w = trail_phase_words_ref(self);
        let self_is_uniform = is_word_array_uniform_64(self_w);
        let self_b0 = (self_w[0] & 0xFF) as u8;
        let self_is_byte_uniform =
            self_is_uniform & (self_w[0] == self_w[0].rotate_right(8));
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (self_v0, self_v1) = unsafe {
            super::branchless::avx2_words64_to_vec64_with_uniform(self_w, self_is_uniform)
        };
        let cur_linked_id = self.linked_trail_id;

        let mut coh_sum = 0.0_f32;
        let mut best_score = -1.0e30_f32;
        let mut best_id = 0u32;
        let mut coherent_count = 0u32;

        let mut idx = 0usize;
        while idx < count {
            let cand_id = idx as u32;
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            let (bayes_delta, coh, is_coherent) = unsafe {
                let mut win_meta_evaluated = true;
                let mut win_is_uniform = false;
                let mut win_eq_self = false;
                let (mut wv0, mut wv1) = (self_v0, self_v1);
                let (bd, ch, coh_ok, _) = eval_candidate_trail_avx2_with_macro_fast(
                    self_w,
                    self_is_uniform,
                    self_is_byte_uniform,
                    self_b0,
                    self_v0,
                    self_v1,
                    cur_linked_id,
                    cand_id,
                    &candidate_trails[idx],
                    phase_scale,
                    clean_eta,
                    clean_link,
                    safe_thresh,
                    0,
                    None,
                    &mut win_meta_evaluated,
                    &mut win_is_uniform,
                    &mut win_eq_self,
                    &mut wv0,
                    &mut wv1,
                    0.0,
                );
                (bd, ch, coh_ok)
            };
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            let (bayes_delta, coh, is_coherent) = eval_candidate_trail_words_fast(
                self_w,
                self_is_uniform,
                self_is_byte_uniform,
                self_b0,
                cur_linked_id,
                cand_id,
                &candidate_trails[idx],
                phase_scale,
                clean_eta,
                clean_link,
                safe_thresh,
            );
            let penalty = branchless_select_f32(is_coherent, 0.0, clean_penalty);
            let new_score =
                branchless_flush_subnormal_f32(attention_scores[idx] + bayes_delta - penalty);
            attention_scores[idx] = new_score;
            coh_sum += coh;
            let better = (idx == 0) | (new_score > best_score);
            best_score = branchless_select_f32(better, new_score, best_score);
            best_id = branchless_select_u32(better, cand_id, best_id);
            coherent_count = coherent_count.wrapping_add(u32::from(is_coherent));
            idx += 1;
        }
        let mean_coh = branchless_clamp_f32(coh_sum * fast_inv_len_f32(count), -1.0, 1.0);
        (mean_coh, best_id, coherent_count)
    }

    #[inline(always)]
    fn select_rewire_bridge_and_modulate_bayesian_trails(
        &mut self,
        candidate_trails: &[gcso_pointer_trail_t],
        attention_scores_opt: Option<&mut [f32]>,
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        liminal_interpolation_u7: u8,
        repulsion_penalty: f32,
    ) -> (u32, f32, f32, u32, bool) {
        self.select_rewire_bridge_and_modulate_with_macro_centroid_64(
            candidate_trails,
            attention_scores_opt,
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
            liminal_interpolation_u7,
            repulsion_penalty,
            0,
            None,
            0.0,
        )
    }

    #[inline(always)]
    fn select_rewire_bridge_and_modulate_with_macro_centroid_64(
        &mut self,
        candidate_trails: &[gcso_pointer_trail_t],
        attention_scores_opt: Option<&mut [f32]>,
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        liminal_interpolation_u7: u8,
        repulsion_penalty: f32,
        macro_anchor_id: u32,
        win_w_opt: Option<&[u64; NUM_SWARM_MICRO_CELLS]>,
        cluster_bonus_gain: f32,
    ) -> (u32, f32, f32, u32, bool) {
        let init_w = *trail_phase_words_ref(self);
        let (best_id, out_score, mean_coh, coherent_count, found_valid, _post_w, _best_coh) =
            select_rewire_bridge_and_modulate_with_preloaded_words_64(
                self,
                init_w,
                candidate_trails,
                attention_scores_opt,
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
                liminal_interpolation_u7,
                repulsion_penalty,
                macro_anchor_id,
                win_w_opt,
                cluster_bonus_gain,
            );
        (best_id, out_score, mean_coh, coherent_count, found_valid)
    }

    #[inline]
    fn apply_cross_slot_synergy(
        &mut self,
        slot_a: u32,
        phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight_a: f32,
        slot_b: u32,
        phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight_b: f32,
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
    ) -> (f32, f32) {
        let coupled = is_slot_pair_coupled(coupling_matrix_8x8, slot_a, slot_b);
        let effective_gain = branchless_select_f32(coupled, synergy_gain, 0.0);

        let wa = load_q7_words_64(phases_a);
        let wb = load_q7_words_64(phases_b);
        let (fw, synergy_pull, coherence) =
            branchless_q7_synergy_fuse_words_64(&wa, weight_a, &wb, weight_b, effective_gain);
        let cur_w = trail_phase_words_mut(self);
        cur_w[0] = swar_q7_add_sat_x8(cur_w[0], fw[0]);
        cur_w[1] = swar_q7_add_sat_x8(cur_w[1], fw[1]);
        cur_w[2] = swar_q7_add_sat_x8(cur_w[2], fw[2]);
        cur_w[3] = swar_q7_add_sat_x8(cur_w[3], fw[3]);
        cur_w[4] = swar_q7_add_sat_x8(cur_w[4], fw[4]);
        cur_w[5] = swar_q7_add_sat_x8(cur_w[5], fw[5]);
        cur_w[6] = swar_q7_add_sat_x8(cur_w[6], fw[6]);
        cur_w[7] = swar_q7_add_sat_x8(cur_w[7], fw[7]);
        let new_pull = branchless_flush_subnormal_f32(
            branchless_relu_f32(self.attractor_pull_force) + synergy_pull,
        );
        self.attractor_pull_force = branchless_min_f32(new_pull, MAX_ATTRACTOR_PULL_FORCE);

        let dom_slot = branchless_select_u32(weight_a >= weight_b, slot_a, slot_b) as u8;
        let cur_ptr = gcso_tagged_ptr_t(self.current_ptr);
        self.current_ptr = cur_ptr
            .with_action_and_slot(dom_slot, cur_ptr.action_mode_flags() | ACTION_MODE_DPSR_STEER)
            .raw();
        self.cluster_id = u32::from(dom_slot & ATTRACTOR_SLOT_ID_MASK);

        let is_resonant = coupled & (coherence > 0.25) & (synergy_pull > 0.0);
        let is_liminal_beat = coupled & (coherence < -0.10) & (synergy_pull > 0.0);
        self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(is_liminal_beat, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        (synergy_pull, coherence)
    }

    #[inline]
    fn apply_mutual_slot_inhibition(
        &mut self,
        inhibitor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        inhibition_strength: f32,
    ) -> f32 {
        let raw_damped = branchless_q7_inhibit_64(
            &mut self.accumulated_phase_delta,
            self.attractor_pull_force,
            inhibitor_phases,
            inhibition_strength,
        );
        let damped_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(raw_damped),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        self.attractor_pull_force = damped_pull;
        damped_pull
    }

    #[inline]
    fn apply_neighborhood_synergy_from_bank_from_words(
        &mut self,
        mut acc_w: [u64; NUM_SWARM_MICRO_CELLS],
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS]) {
        let primary_slot = u32::from(self.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK);
        let p_idx = primary_slot as usize;
        let primary_phase = &slot_phase_bank[p_idx];
        let primary_w = load_q7_words_64(primary_phase);
        let primary_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(slot_weights[p_idx]));

        let exclude_mask = (1u8 << p_idx) | (1u8 << ATTRACTOR_SLOT_REPULSION);
        let mut neighbor_mask =
            coupled_slot_row_mask(coupling_matrix_8x8, primary_slot) & !exclude_mask;

        // Keep `acc_w` and `cur_w` in 8 64-bit registers across all coupled neighbors (`0` intermediate stack arrays)
        let mut cur_w = primary_w;
        let mut running_pull = primary_pull;
        let mut coh_sum = 0.0_f32;
        let mut active_neighbors = 0u32;
        let mut any_liminal = false;

        while neighbor_mask != 0 {
            let sec_idx = neighbor_mask.trailing_zeros() as usize;
            neighbor_mask &= neighbor_mask - 1;

            let sec_pull =
                branchless_flush_subnormal_f32(branchless_relu_f32(slot_weights[sec_idx]));
            if sec_pull > 1.0e-9 {
                let sec_w = load_q7_words_64(&slot_phase_bank[sec_idx]);
                let (fw, syn_pull, coh) = branchless_q7_synergy_fuse_words_64(
                    &cur_w,
                    running_pull,
                    &sec_w,
                    sec_pull,
                    synergy_gain,
                );
                cur_w = fw;
                running_pull = syn_pull;
                coh_sum += coh;
                active_neighbors += 1;
                any_liminal |= coh < -0.10;
            }
        }

        if (active_neighbors > 0) | (primary_pull > 1.0e-9) {
            acc_w[0] = swar_q7_add_sat_x8(acc_w[0], cur_w[0]);
            acc_w[1] = swar_q7_add_sat_x8(acc_w[1], cur_w[1]);
            acc_w[2] = swar_q7_add_sat_x8(acc_w[2], cur_w[2]);
            acc_w[3] = swar_q7_add_sat_x8(acc_w[3], cur_w[3]);
            acc_w[4] = swar_q7_add_sat_x8(acc_w[4], cur_w[4]);
            acc_w[5] = swar_q7_add_sat_x8(acc_w[5], cur_w[5]);
            acc_w[6] = swar_q7_add_sat_x8(acc_w[6], cur_w[6]);
            acc_w[7] = swar_q7_add_sat_x8(acc_w[7], cur_w[7]);
        }
        let mut total_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(self.attractor_pull_force))
                + running_pull;

        total_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(total_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        let safe_neighbors = branchless_select_u32(active_neighbors > 0, active_neighbors, 1);
        let rsqrt_n = branchless_popcount_rsqrt_u32(safe_neighbors);
        let mean_coh = branchless_select_f32(
            active_neighbors > 0,
            coh_sum * (rsqrt_n * rsqrt_n),
            branchless_select_f32(primary_pull > 1.0e-9, 1.0, 0.0),
        );
        let is_resonant = (active_neighbors > 0) & (mean_coh > 0.25);
        self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(any_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        if inhibition_strength > 0.0 {
            let rep_idx = usize::from(ATTRACTOR_SLOT_REPULSION);
            let rep_weight = branchless_relu_f32(slot_weights[rep_idx]);
            let eff_inh = branchless_select_f32(rep_weight > 1.0e-9, inhibition_strength, 0.0);
            let rep_words = load_q7_words_64(&slot_phase_bank[rep_idx]);
            let raw_damped =
                branchless_q7_inhibit_words_64(&mut acc_w, total_pull, &rep_words, eff_inh);
            total_pull = branchless_min_f32(
                branchless_flush_subnormal_f32(raw_damped),
                MAX_ATTRACTOR_PULL_FORCE,
            );
        }

        *trail_phase_words_mut(self) = acc_w;
        self.attractor_pull_force = total_pull;
        (total_pull, mean_coh, active_neighbors, acc_w)
    }

    #[inline(always)]
    fn apply_neighborhood_synergy_from_bank_to_words(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS]) {
        let acc_w = *trail_phase_words_ref(self);
        self.apply_neighborhood_synergy_from_bank_from_words(
            acc_w,
            slot_phase_bank,
            slot_weights,
            coupling_matrix_8x8,
            synergy_gain,
            inhibition_strength,
        )
    }

    #[inline(always)]
    fn apply_neighborhood_synergy_from_bank(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32) {
        let (total_pull, mean_coh, active_neighbors, _words) = self
            .apply_neighborhood_synergy_from_bank_to_words(
                slot_phase_bank,
                slot_weights,
                coupling_matrix_8x8,
                synergy_gain,
                inhibition_strength,
            );
        (total_pull, mean_coh, active_neighbors)
    }
}

// ===================================================================
// 6. Constant-Time Closed-Loop Micro-Swarm Resolution (`StigmergicSwarmOps`)
// ===================================================================

/// 16-byte register-aligned summary returned when the 8-cell Micro-Swarm attempts to resolve
/// an Action Hub command on the Hot Path and feeds back into the Stigmergic environment.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwarmResolutionReport {
    /// Feedback-evolved 64-bit tagged pointer ready for the Action Hub's next instruction step.
    pub evolved_ptr: gcso_tagged_ptr_t,
    /// Signed stigmergic density feedback ($\Delta\rho_{\mathrm{stigmergy}}$) applied to the trail.
    pub stigmergic_delta_density: f32,
    /// 8-bit Micro-Swarm Quorum bitmask (`bit c == 1` iff 8-head cell $c \in 0..8$ converged).
    pub quorum_mask: u8,
    /// Number of Micro-Swarm cells (`0..=8`) that resolved their sub-task within `tolerance_q7`.
    pub resolved_cell_count: u8,
    /// Mean circular Q7 phase residual (`0..=128`) across all 64 heads after the Micro-Swarm step.
    pub mean_residual_q7: u8,
    /// `1` if the Swarm achieved quorum on the Hot Path without Bloom repulsion, `0` if escalated.
    pub hot_path_resolved: u8,
}

impl Default for SwarmResolutionReport {
    #[inline(always)]
    fn default() -> Self {
        Self {
            evolved_ptr: gcso_tagged_ptr_t::NULL,
            stigmergic_delta_density: 0.0,
            quorum_mask: 0,
            resolved_cell_count: 0,
            mean_residual_q7: 0,
            hot_path_resolved: 0,
        }
    }
}

// SAFETY: `SwarmResolutionReport` is a 16-byte `#[repr(C, align(16))]` structure with zero padding.
unsafe impl Pod for SwarmResolutionReport {}

/// Evaluates a single 8-head SWAR Micro-Swarm cell step on the circle group $S^1 \cong SO(2)$
/// in **strictly constant time (`0` branches)** across all inputs.
#[inline(always)]
#[must_use]
pub const fn swar_subswarm_resolve_x8(
    current_word: u64,
    target_word: u64,
    step_gain_u7: u8,
    qdps_min_step_q7: u8,
    tolerance_q7: u8,
) -> (u64, u8, bool) {
    let (gain_s, gain_bias, gain_gt_128, qdps_thresh_x8, qdps_valid_mask) =
        precompute_swarm_step_params(step_gain_u7, qdps_min_step_q7);
    swar_subswarm_resolve_with_bias_x8(
        current_word,
        target_word,
        gain_s,
        gain_bias,
        gain_gt_128,
        qdps_thresh_x8,
        qdps_valid_mask,
        tolerance_q7,
    )
}

/// Closed-loop trait uniting **Micro-Swarm Subdivision** and **Stigmergic Feedback**.
pub trait StigmergicSwarmOps: StigmergicTrailOps {
    /// Extracts the 8-head Q7 phase word (`u64`) belonging to Micro-Swarm cell `cell_idx` (`0..=7`).
    #[must_use]
    fn extract_subswarm_word(&self, cell_idx: usize) -> u64;
    /// Writes an 8-head Q7 phase word (`u64`) into Micro-Swarm cell `cell_idx` (`0..=7`).
    fn write_subswarm_word(&mut self, cell_idx: usize, word: u64);

    /// Resolves an Action Hub command across all 8 SWAR Micro-Swarm cells directly from
    /// pre-loaded target SWAR words (`target_words`), avoiding any 64-byte stack reload when
    /// chained after `ManifoldGatewaySignal::synthesize_gateway_frame_with_words`.
    fn try_resolve_action_in_swarm_from_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        target_words: &[u64; NUM_SWARM_MICRO_CELLS],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS]);

    /// Subdivides the Action Hub's target 64-head phase command into **8 parallel 8-head SWAR
    /// Micro-Swarm cells** gated by `active_cell_mask` in strictly constant time (`0` branches),
    /// returning both `(SwarmResolutionReport, resolved_swar_words)` so downstream RoPE/RIPA kernels
    /// can steer directly from registers without re-loading `accumulated_phase_delta`.
    #[inline(always)]
    fn try_resolve_action_in_swarm_to_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        target_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS]) {
        let tgt_w = load_q7_words_64(target_phases);
        self.try_resolve_action_in_swarm_from_words(
            action_ptr,
            &tgt_w,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        )
    }

    /// Subdivides the Action Hub's target 64-head phase command into **8 parallel 8-head SWAR
    /// Micro-Swarm cells** gated by `active_cell_mask` in strictly constant time (`0` branches).
    #[inline(always)]
    fn try_resolve_action_in_swarm_masked(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        target_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> SwarmResolutionReport {
        self.try_resolve_action_in_swarm_to_words(
            action_ptr,
            target_phases,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        )
        .0
    }

    /// Subdivides the Action Hub's target 64-head phase command across all 8 Micro-Swarm cells (`0xFF`).
    #[inline(always)]
    fn try_resolve_action_in_swarm(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        target_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> SwarmResolutionReport {
        self.try_resolve_action_in_swarm_masked(
            action_ptr,
            target_phases,
            0xFF,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        )
    }

    /// Recruits a pre-identified dominant unified-field donor anchor (`donor_phases`, `donor_anchor_id`,
    /// `donor_pull`) across any unresolved 8-head Micro-Swarm cells (`!prev_quorum`) in $\mathcal{O}(\mathrm{popcount}(\neg\mathrm{quorum}))$,
    /// avoiding a redundant second scan over the anchor slice and skipping already-resolved cells.
    fn recruit_unified_field_donor_subswarms(
        &mut self,
        donor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        donor_anchor_id: u32,
        donor_pull: f32,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> u8;

    /// Performs **Zero-Hyperparameter Lateral Recruitment on the Single Unified Phase Manifold**:
    /// scans `anchor_phases` and `anchor_base_energies` for the highest-resonance anchor and
    /// recruits it across any unresolved 8-head Micro-Swarm cell $c \in 0..8$.
    fn recruit_unified_field_anchor_subswarms(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> u8;

    /// Performs **Lateral Stigmergic Recruitment** across the 8 Working Lanes (`0..=7`)
    /// directly from a flat `[gcso_q7_t; 512]` phase bank and `[f32; 8]` weight slice (`0` copy),
    /// iterating only over unresolved cells (`!prev_quorum`).
    fn recruit_coupled_slot_subswarms_flat(
        &mut self,
        flat_slot_phases_512: &[gcso_q7_t],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> u8;

    /// Performs **Lateral Stigmergic Recruitment** across the 8 Working Lanes (`0..=7`)
    /// for any straggler Micro-Swarm cell $c \in 0..8$ that has not yet achieved quorum.
    fn recruit_coupled_slot_subswarms(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> u8;

    /// Fuses resolved 8-head Micro-Swarm cells from a peer pointer trail (`peer_trail`) into `self`
    /// in 8 branchless SWAR register operations.
    fn fuse_with_peer_swarm_trail(
        &mut self,
        peer_trail: &gcso_pointer_trail_t,
        quorum_threshold: u8,
    ) -> u8;

    /// Folds `anchor_phases` and `anchor_base_energies` on the Single Unified Attractor Manifold
    /// via [`branchless_q7_unified_field_fold_words_with_init_weight_64`], resolves the composite
    /// phase target across the 8 Micro-Swarm cells (`active_cell_mask`), and recruits the dominant
    /// unified-field anchor on partial quorum in a single register-resident pass, returning both
    /// `(SwarmResolutionReport, resolved_swar_words)`.
    fn resolve_unified_field_in_swarm_to_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS]);

    /// Folds `anchor_phases` and `anchor_base_energies` on the Single Unified Attractor Manifold,
    /// resolves the composite phase target across the 8 Micro-Swarm cells (`active_cell_mask`),
    /// and recruits the dominant unified-field anchor on partial quorum in a single register-resident
    /// pass without intermediate stack buffers or redundant load/store round-trips.
    #[inline(always)]
    fn resolve_unified_field_in_swarm_with_recruitment(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> SwarmResolutionReport {
        self.resolve_unified_field_in_swarm_to_words(
            action_ptr,
            anchor_phases,
            anchor_base_energies,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        )
        .0
    }

    /// Resolves `action_ptr`'s target working lane from `slot_phase_bank` (`[[gcso_q7_t; 64]; 8]`)
    /// across the 8 Micro-Swarm cells (`active_cell_mask`), recruits coupled neighbor lanes on
    /// partial quorum entirely in registers, and returns `(SwarmResolutionReport, resolved_swar_words)`.
    fn resolve_slot_bank_in_swarm_to_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS]);

    /// Resolves `action_ptr`'s target working lane from `slot_phase_bank` (`[[gcso_q7_t; 64]; 8]`)
    /// across the 8 Micro-Swarm cells (`active_cell_mask`) and recruits coupled neighbor lanes on partial quorum.
    #[inline(always)]
    fn resolve_slot_bank_in_swarm_with_recruitment(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> SwarmResolutionReport {
        self.resolve_slot_bank_in_swarm_to_words(
            action_ptr,
            slot_phase_bank,
            slot_weights,
            coupling_matrix_8x8,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        )
        .0
    }
}

impl gcso_pointer_trail_t {
    #[inline(always)]
    pub(crate) fn try_resolve_action_in_swarm_with_cur_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS],
        target_words: &[u64; NUM_SWARM_MICRO_CELLS],
        has_active_anchors: bool,
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS]) {
        let ptr_phase = action_ptr.unpack_phase();
        let ptr_phase_splat = ((ptr_phase as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let cur_w = *cur_w;
        let tgt_base_w = *target_words;
        let tgt_has_explicit = words_any_nonzero_64(&tgt_base_w);
        let use_explicit_tgt = tgt_has_explicit | (ptr_phase_splat == 0);
        let mut next_words = cur_w;
        let quorum_mask: u8;
        let residual_sum: u32;
        if use_explicit_tgt
            && (cur_w[0] == tgt_base_w[0])
            && ((((cur_w[1] ^ tgt_base_w[1]) | (cur_w[2] ^ tgt_base_w[2]))
                | ((cur_w[3] ^ tgt_base_w[3]) | (cur_w[4] ^ tgt_base_w[4]))
                | ((cur_w[5] ^ tgt_base_w[5]) | (cur_w[6] ^ tgt_base_w[6]))
                | (cur_w[7] ^ tgt_base_w[7]))
                == 0)
        {
            quorum_mask = 0xFF;
            residual_sum = 0;
        } else {
            let cur_is_uniform = is_word_array_uniform_64(&cur_w);
            let tgt_is_uniform = (!use_explicit_tgt) || is_word_array_uniform_64(&tgt_base_w);
            if (active_cell_mask == 0xFF) & cur_is_uniform & tgt_is_uniform {
            let cw0 = cur_w[0];
            let tw0 = branchless_select_u64(use_explicit_tgt, tgt_base_w[0], ptr_phase_splat);
            if cw0 == tw0 {
                quorum_mask = 0xFF;
                residual_sum = 0;
            } else {
                let (gain_s, gain_bias, gain_gt_128, qdps_thresh_x8, qdps_valid_mask) =
                    precompute_swarm_step_params(step_gain_u7, qdps_min_step_q7);
                let (stepped_w0, cell_res, cell_ok) = swar_subswarm_resolve_with_bias_x8(
                    cw0,
                    tw0,
                    gain_s,
                    gain_bias,
                    gain_gt_128,
                    qdps_thresh_x8,
                    qdps_valid_mask,
                    tolerance_q7,
                );
                next_words = [stepped_w0; NUM_SWARM_MICRO_CELLS];
                quorum_mask = branchless_select_u8(cell_ok, 0xFF, 0x00);
                residual_sum = u32::from(cell_res) << 3;
            }
        } else {
            let (nw, qm, rsum) = swar_subswarm_resolve_words_64(
                &cur_w,
                &tgt_base_w,
                use_explicit_tgt,
                ptr_phase_splat,
                active_cell_mask,
                step_gain_u7,
                qdps_min_step_q7,
                tolerance_q7,
            );
                next_words = nw;
                quorum_mask = qm;
                residual_sum = rsum;
            }
        }

        let mut repulsion_triggered = false;
        if let Some(bf) = bloom_filter {
            let (_density, rep_active) = bf.apply_phase_conjugate_repulsion_words(
                &mut next_words,
                repulsion_threshold,
                repulsion_scale_u7,
            );
            repulsion_triggered = rep_active;
        }

        if !(quorum_mask == 0xFF && residual_sum == 0 && !repulsion_triggered && cur_w[0] == next_words[0])
            && ((cur_w[0] != next_words[0])
                || (((cur_w[1] ^ next_words[1])
                    | (cur_w[2] ^ next_words[2])
                    | (cur_w[3] ^ next_words[3])
                    | (cur_w[4] ^ next_words[4])
                    | (cur_w[5] ^ next_words[5])
                    | (cur_w[6] ^ next_words[6])
                    | (cur_w[7] ^ next_words[7]))
                    != 0))
        {
            *trail_phase_words_mut(self) = next_words;
        }

        let resolved_count = quorum_mask.count_ones() as u8;
        let mean_residual = ((residual_sum + 4) >> 3) as u8;
        let safe_quorum_req = branchless_select_u8(
            quorum_threshold == 0,
            1,
            branchless_select_u8(quorum_threshold > 8, 8, quorum_threshold),
        );
        let hot_path_resolved = (resolved_count >= safe_quorum_req) & (!repulsion_triggered);

        let prev_rho = branchless_flush_subnormal_f32(branchless_relu_f32(self.stigmergic_density));
        let quorum_ratio = f32::from(resolved_count) * 0.125;
        let convergence_bonus =
            branchless_clamp_f32(1.0 - (f32::from(mean_residual) * (1.0 / 128.0)), 0.0, 1.0);
        let pos_boost = 0.25 * quorum_ratio * (1.0 + convergence_bonus);
        let neg_damping = -0.20 * prev_rho;
        let raw_delta_rho = branchless_select_f32(hot_path_resolved, pos_boost, neg_damping);
        let next_rho = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(prev_rho + raw_delta_rho)),
            MAX_STIGMERGIC_DENSITY,
        );
        let delta_rho = next_rho - prev_rho;
        self.stigmergic_density = next_rho;

        self.prev_ptr = self.current_ptr;
        self.step_count = self.step_count.wrapping_add(1);
        self.set_swarm_quorum_state(quorum_mask, hot_path_resolved);
        self.flags |= branchless_select_u32(repulsion_triggered, TRAIL_FLAG_REPULSION_ACTIVE, 0);

        let clean_h = branchless_select_f32(
            branchless_is_finite_f32(entropy_h_tilde),
            entropy_h_tilde,
            0.0,
        );
        let can_skip = hot_path_resolved & (clean_h <= low_entropy_skip_thresh);
        let resolved_mode = branchless_select_u8(
            can_skip,
            ACTION_PRESET_INERTIAL_SKIP,
            ACTION_PRESET_EXPRESS_PHASE,
        );
        let escalated_mode = branchless_select_u8(
            repulsion_triggered,
            ACTION_PRESET_REPULSION_ESCAPE,
            ACTION_PRESET_FULL_ORCHESTRATION,
        );
        let next_mode = branchless_select_u8(hot_path_resolved, resolved_mode, escalated_mode);
        let has_unified_anchor = has_active_anchors | (self.target_anchor_id != 0);
        let primary_slot = branchless_select_u8(
            has_unified_anchor,
            (self.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK,
            action_ptr.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK,
        );
        let next_slot =
            branchless_select_u8(repulsion_triggered, ATTRACTOR_SLOT_REPULSION, primary_slot);

        let base_evolved = action_ptr.with_action_and_slot(next_slot, next_mode);
        let conjugated_evolved = base_evolved.conjugate_phase();
        let evolved_ptr = gcso_tagged_ptr_t(branchless_select_u64(
            repulsion_triggered,
            conjugated_evolved.raw(),
            base_evolved.raw(),
        ));

        self.current_ptr = evolved_ptr.raw();
        self.cluster_id = branchless_select_u32(
            has_unified_anchor & (!repulsion_triggered),
            self.target_anchor_id,
            u32::from(next_slot),
        );
        let dpsr_on = (next_mode & ACTION_MODE_DPSR_STEER) != 0;
        let skip_on = (next_mode & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags = (self.flags & !(TRAIL_FLAG_DPSR_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | branchless_select_u32(dpsr_on, TRAIL_FLAG_DPSR_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);

        (
            SwarmResolutionReport {
                evolved_ptr,
                stigmergic_delta_density: delta_rho,
                quorum_mask,
                resolved_cell_count: resolved_count,
                mean_residual_q7: mean_residual,
                hot_path_resolved: u8::from(hot_path_resolved),
            },
            next_words,
        )
    }


    #[inline(always)]
    fn recruit_single_anchor_stragglers_cold(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        report: &mut SwarmResolutionReport,
        next_words: &mut [u64; NUM_SWARM_MICRO_CELLS],
        tgt_w: &[u64; NUM_SWARM_MICRO_CELLS],
        total_pull: f32,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
    ) {
        let eff_gain =
            ((f32::from(step_gain_u7) * branchless_clamp_f32(total_pull, 0.25, 1.0)) + 0.5) as u8;
        let (rg_s, rg_bias, rg_gt_128, qdps_thresh_x8, qdps_valid_mask) =
            precompute_swarm_step_params(eff_gain, qdps_min_step_q7);
        let mut rec_quorum = report.quorum_mask;
        let mut unresolved = !rec_quorum;
        let mut has_rec_prev = false;
        let mut rec_prev_nw = 0u64;
        let mut rec_prev_dw = 0u64;
        let mut rec_prev_cand = 0u64;
        let mut rec_prev_ok = false;
        while unresolved != 0 {
            let c = unresolved.trailing_zeros() as usize;
            unresolved &= unresolved - 1;
            let nw = next_words[c];
            let donor_w = tgt_w[c];
            let (cand_w, cell_ok) = if has_rec_prev & (nw == rec_prev_nw) & (donor_w == rec_prev_dw)
            {
                (rec_prev_cand, rec_prev_ok)
            } else {
                let (cw_out, _r, ok_out) = swar_subswarm_resolve_with_bias_x8(
                    nw,
                    donor_w,
                    rg_s,
                    rg_bias,
                    rg_gt_128,
                    qdps_thresh_x8,
                    qdps_valid_mask,
                    tolerance_q7,
                );
                has_rec_prev = true;
                rec_prev_nw = nw;
                rec_prev_dw = donor_w;
                rec_prev_cand = cw_out;
                rec_prev_ok = ok_out;
                (cw_out, ok_out)
            };
            next_words[c] = cand_w;
            rec_quorum |= (cell_ok as u8) << c;
        }
        *trail_phase_words_mut(self) = *next_words;
        let safe_req = branchless_select_u8(
            quorum_threshold == 0,
            1,
            branchless_select_u8(quorum_threshold > 8, 8, quorum_threshold),
        );
        let resolved_count = rec_quorum.count_ones() as u8;
        let now_resolved = resolved_count >= safe_req;
        self.set_swarm_quorum_state(rec_quorum, now_resolved);
        report.quorum_mask = rec_quorum;
        report.resolved_cell_count = resolved_count;
        report.hot_path_resolved = u8::from(now_resolved);
        if now_resolved {
            let clean_h = branchless_select_f32(
                branchless_is_finite_f32(entropy_h_tilde),
                entropy_h_tilde,
                0.0,
            );
            let can_skip = clean_h <= low_entropy_skip_thresh;
            let resolved_mode = branchless_select_u8(
                can_skip,
                ACTION_PRESET_INERTIAL_SKIP,
                ACTION_PRESET_EXPRESS_PHASE,
            );
            let next_slot = (self.target_anchor_id as u8) & ATTRACTOR_SLOT_ID_MASK;
            report.evolved_ptr = action_ptr.with_action_and_slot(next_slot, resolved_mode);
            self.current_ptr = report.evolved_ptr.raw();
            let dpsr_on = (resolved_mode & ACTION_MODE_DPSR_STEER) != 0;
            let skip_on = (resolved_mode & ACTION_MODE_LAYER_SKIP) != 0;
            self.flags = (self.flags & !(TRAIL_FLAG_DPSR_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
                | branchless_select_u32(dpsr_on, TRAIL_FLAG_DPSR_ACTIVE, 0)
                | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
        }
    }

    #[inline(always)]
    fn recruit_multi_anchor_stragglers_cold(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        report: &mut SwarmResolutionReport,
        next_words: &mut [u64; NUM_SWARM_MICRO_CELLS],
        donor_anchor: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        total_pull: f32,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
    ) {
        let donor_words = load_q7_words_64(donor_anchor);
        self.recruit_single_anchor_stragglers_cold(
            action_ptr,
            report,
            next_words,
            &donor_words,
            total_pull,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
        );
    }

    #[inline(always)]
    fn recruit_slot_bank_stragglers_cold(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        slot_id: usize,
        orig_words: &[u64; NUM_SWARM_MICRO_CELLS],
        target_words: &[u64; NUM_SWARM_MICRO_CELLS],
        report: &mut SwarmResolutionReport,
        next_words: &mut [u64; NUM_SWARM_MICRO_CELLS],
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
    ) {
        let coupled_row = coupled_slot_row_mask(coupling_matrix_8x8, slot_id as u32);
        let gain_f32 = f32::from(step_gain_u7);
        let qdps_thresh_x8 = ((qdps_min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let qdps_valid_mask = ((qdps_min_step_q7 <= 127) as u64).wrapping_neg();
        let mut new_quorum = report.quorum_mask;
        let mut unresolved = !new_quorum;

        while unresolved != 0 {
            let cell = unresolved.trailing_zeros() as usize;
            unresolved &= unresolved - 1;
            let base = cell * HEADS_PER_SWARM_CELL;
            let slot_coupled = ((coupled_row >> cell) & 1) != 0;
            let slot_w = branchless_relu_f32(slot_weights[cell]);
            let use_neighbor = slot_coupled & (slot_w > 1.0e-6);
            let donor_slot = branchless_select_usize(use_neighbor, cell, slot_id);
            let donor_w = if use_neighbor {
                load_q7_word_at(&slot_phase_bank[cell], base)
            } else {
                target_words[cell]
            };
            let donor_weight = branchless_relu_f32(slot_weights[donor_slot]);
            let eff_gain =
                ((gain_f32 * branchless_clamp_f32(donor_weight, 0.25, 1.0)) + 0.5) as u8;
            let gain_gt_128 = eff_gain > 128;
            let gain_s = branchless_select_u8(
                gain_gt_128,
                eff_gain.wrapping_sub(128),
                eff_gain,
            ) as u64;
            let gain_bias = (128u64.wrapping_sub(gain_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);

            let (cand_next, res_next, ok_next) = swar_subswarm_resolve_with_bias_x8(
                next_words[cell],
                donor_w,
                gain_s,
                gain_bias,
                gain_gt_128,
                qdps_thresh_x8,
                qdps_valid_mask,
                tolerance_q7,
            );
            let (cand_orig, res_orig, ok_orig) = swar_subswarm_resolve_with_bias_x8(
                orig_words[cell],
                donor_w,
                gain_s,
                gain_bias,
                gain_gt_128,
                qdps_thresh_x8,
                qdps_valid_mask,
                tolerance_q7,
            );
            let use_orig = res_orig < res_next;
            let candidate_w = branchless_select_u64(use_orig, cand_orig, cand_next);
            let cell_resolved = ok_next | ok_orig;
            next_words[cell] = candidate_w;
            new_quorum |= (cell_resolved as u8) << cell;
        }

        *trail_phase_words_mut(self) = *next_words;
        let safe_req = branchless_select_u8(
            quorum_threshold == 0,
            1,
            branchless_select_u8(quorum_threshold > 8, 8, quorum_threshold),
        );
        let now_resolved = (new_quorum.count_ones() as u8) >= safe_req;
        self.set_swarm_quorum_state(new_quorum, now_resolved);
        report.quorum_mask = new_quorum;
        report.resolved_cell_count = new_quorum.count_ones() as u8;
        report.hot_path_resolved = u8::from(now_resolved);
        if now_resolved {
            let clean_h = branchless_select_f32(
                branchless_is_finite_f32(entropy_h_tilde),
                entropy_h_tilde,
                0.0,
            );
            let resolved_mode = branchless_select_u8(
                clean_h <= low_entropy_skip_thresh,
                ACTION_PRESET_INERTIAL_SKIP,
                ACTION_PRESET_EXPRESS_PHASE,
            );
            report.evolved_ptr =
                action_ptr.with_action_and_slot(action_ptr.target_attractor_slot(), resolved_mode);
            self.current_ptr = report.evolved_ptr.raw();
            let dpsr_on = (resolved_mode & ACTION_MODE_DPSR_STEER) != 0;
            let skip_on = (resolved_mode & ACTION_MODE_LAYER_SKIP) != 0;
            self.flags = (self.flags & !(TRAIL_FLAG_DPSR_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
                | branchless_select_u32(dpsr_on, TRAIL_FLAG_DPSR_ACTIVE, 0)
                | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
        }
    }

}

impl StigmergicSwarmOps for gcso_pointer_trail_t {
    #[inline(always)]
    fn extract_subswarm_word(&self, cell_idx: usize) -> u64 {
        let safe_cell = cell_idx & (NUM_SWARM_MICRO_CELLS - 1);
        load_q7_word_at(&self.accumulated_phase_delta, safe_cell * HEADS_PER_SWARM_CELL)
    }

    #[inline(always)]
    fn write_subswarm_word(&mut self, cell_idx: usize, word: u64) {
        let safe_cell = cell_idx & (NUM_SWARM_MICRO_CELLS - 1);
        store_q7_word_at(
            &mut self.accumulated_phase_delta,
            safe_cell * HEADS_PER_SWARM_CELL,
            word,
        );
    }

    #[inline(always)]
    fn try_resolve_action_in_swarm_from_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        target_words: &[u64; NUM_SWARM_MICRO_CELLS],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS]) {
        let cur_w = *trail_phase_words_ref(self);
        self.try_resolve_action_in_swarm_with_cur_words(
            action_ptr,
            &cur_w,
            target_words,
            false,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        )
    }

    #[inline(always)]
    fn recruit_unified_field_donor_subswarms(
        &mut self,
        donor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        donor_anchor_id: u32,
        donor_pull: f32,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> u8 {
        let prev_quorum = self.swarm_quorum_mask();
        let has_best = donor_pull > 1.0e-12;
        self.target_anchor_id =
            branchless_select_u32(has_best, donor_anchor_id, self.target_anchor_id);
        self.cluster_id = branchless_select_u32(has_best, donor_anchor_id, self.cluster_id);
        let mut quorum_mask = prev_quorum;
        let mut unresolved = !prev_quorum;

        if unresolved != 0 {
            let eff_gain =
                ((f32::from(step_gain_u7) * branchless_clamp_f32(donor_pull, 0.25, 1.0)) + 0.5)
                    as u8;
            let (gain_s, gain_bias, gain_gt_128, qdps_thresh_x8, qdps_valid_mask) =
                precompute_swarm_step_params(eff_gain, qdps_min_step_q7);

            while unresolved != 0 {
                let cell = unresolved.trailing_zeros() as usize;
                unresolved &= unresolved - 1;
                let base = cell * HEADS_PER_SWARM_CELL;
                let cur_w = load_q7_word_at(&self.accumulated_phase_delta, base);
                let donor_w = load_q7_word_at(donor_phases, base);
                let (candidate_w, _res, cell_resolved) = swar_subswarm_resolve_with_bias_x8(
                    cur_w,
                    donor_w,
                    gain_s,
                    gain_bias,
                    gain_gt_128,
                    qdps_thresh_x8,
                    qdps_valid_mask,
                    tolerance_q7,
                );
                store_q7_word_at(&mut self.accumulated_phase_delta, base, candidate_w);
                quorum_mask |= (cell_resolved as u8) << cell;
            }
        }

        let safe_req = branchless_select_u8(
            quorum_threshold == 0,
            1,
            branchless_select_u8(quorum_threshold > 8, 8, quorum_threshold),
        );
        self.set_swarm_quorum_state(quorum_mask, (quorum_mask.count_ones() as u8) >= safe_req);
        quorum_mask
    }

    #[inline]
    fn recruit_unified_field_anchor_subswarms(
        &mut self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> u8 {
        let count = anchor_phases.len().min(anchor_base_energies.len());
        let prev_quorum = self.swarm_quorum_mask();
        if count == 0 {
            return prev_quorum;
        }

        let self_w = *trail_phase_words_ref(self);
        let self_is_uniform = is_word_array_uniform_64(&self_w);
        let self_nonzero = if self_is_uniform {
            self_w[0] != 0
        } else {
            words_any_nonzero_64(&self_w)
        };
        let mut best_idx = 0usize;
        let mut best_pull = 0.0_f32;
        let mut idx = 0usize;
        while idx < count {
            let base_e = anchor_base_energies[idx];
            if base_e > 1.0e-12 && branchless_is_finite_f32(base_e) {
                let anc_w = load_q7_words_64(&anchor_phases[idx]);
                let anc_is_uniform = if self_is_uniform {
                    is_word_array_uniform_64(&anc_w)
                } else {
                    false
                };
                let (res_pull, _coh) = eval_unified_anchor_resonance_words_with_uniform(
                    &self_w,
                    self_nonzero,
                    self_is_uniform,
                    &anc_w,
                    anc_is_uniform,
                    base_e,
                );
                let better = res_pull > best_pull;
                best_idx = branchless_select_usize(better, idx, best_idx);
                best_pull = branchless_select_f32(better, res_pull, best_pull);
            }
            idx += 1;
        }

        self.recruit_unified_field_donor_subswarms(
            &anchor_phases[best_idx],
            best_idx as u32,
            best_pull,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
        )
    }

    #[inline]
    fn recruit_coupled_slot_subswarms_flat(
        &mut self,
        flat_slot_phases_512: &[gcso_q7_t],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> u8 {
        let prev_quorum = self.swarm_quorum_mask();
        if flat_slot_phases_512.len() < MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS {
            return prev_quorum;
        }
        let mut quorum_mask = prev_quorum;
        let mut unresolved = !prev_quorum;

        if unresolved != 0 {
            let primary_slot = usize::from(self.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK);
            let coupled_row = coupled_slot_row_mask(coupling_matrix_8x8, primary_slot as u32);
            let gain_f32 = f32::from(step_gain_u7);
            let qdps_thresh_x8 = ((qdps_min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let qdps_valid_mask = ((qdps_min_step_q7 <= 127) as u64).wrapping_neg();

            while unresolved != 0 {
                let cell = unresolved.trailing_zeros() as usize;
                unresolved &= unresolved - 1;
                let base = cell * HEADS_PER_SWARM_CELL;
                let cur_w = load_q7_word_at(&self.accumulated_phase_delta, base);
                let slot_coupled = ((coupled_row >> cell) & 1) != 0;
                let slot_w = branchless_relu_f32(slot_weights[cell]);
                let use_neighbor = slot_coupled & (slot_w > 1.0e-6);
                let donor_slot = branchless_select_usize(use_neighbor, cell, primary_slot);
                let off = donor_slot * MAX_ATTENTION_HEADS + base;
                let donor_w = u64::from_le_bytes([
                    flat_slot_phases_512[off] as u8,
                    flat_slot_phases_512[off + 1] as u8,
                    flat_slot_phases_512[off + 2] as u8,
                    flat_slot_phases_512[off + 3] as u8,
                    flat_slot_phases_512[off + 4] as u8,
                    flat_slot_phases_512[off + 5] as u8,
                    flat_slot_phases_512[off + 6] as u8,
                    flat_slot_phases_512[off + 7] as u8,
                ]);
                let donor_weight = branchless_relu_f32(slot_weights[donor_slot]);
                let eff_gain =
                    ((gain_f32 * branchless_clamp_f32(donor_weight, 0.25, 1.0)) + 0.5) as u8;
                let gain_gt_128 = eff_gain > 128;
                let gain_s = branchless_select_u8(gain_gt_128, eff_gain.wrapping_sub(128), eff_gain)
                    as u64;
                let gain_bias = (128u64.wrapping_sub(gain_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);

                let (candidate_w, _res, cell_resolved) = swar_subswarm_resolve_with_bias_x8(
                    cur_w,
                    donor_w,
                    gain_s,
                    gain_bias,
                    gain_gt_128,
                    qdps_thresh_x8,
                    qdps_valid_mask,
                    tolerance_q7,
                );
                store_q7_word_at(&mut self.accumulated_phase_delta, base, candidate_w);
                quorum_mask |= (cell_resolved as u8) << cell;
            }
        }

        let safe_req = branchless_select_u8(
            quorum_threshold == 0,
            1,
            branchless_select_u8(quorum_threshold > 8, 8, quorum_threshold),
        );
        self.set_swarm_quorum_state(quorum_mask, (quorum_mask.count_ones() as u8) >= safe_req);
        quorum_mask
    }

    #[inline(always)]
    fn recruit_coupled_slot_subswarms(
        &mut self,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> u8 {
        // SAFETY: `[[gcso_q7_t; 64]; 8]` has a contiguous 512-byte layout identical to `[gcso_q7_t; 512]`.
        let flat = unsafe {
            slice::from_raw_parts(
                slot_phase_bank.as_ptr().cast::<gcso_q7_t>(),
                MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS,
            )
        };
        self.recruit_coupled_slot_subswarms_flat(
            flat,
            slot_weights,
            coupling_matrix_8x8,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
        )
    }

    #[inline(always)]
    fn resolve_unified_field_in_swarm_to_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS]) {
        let count = anchor_phases.len().min(anchor_base_energies.len());
        let cur_w = *trail_phase_words_ref(self);

        let (tgt_w, total_pull, dom_id, has_active_anchors) = if count == 1 {
            let clean_e =
                branchless_flush_subnormal_f32(branchless_relu_f32(anchor_base_energies[0]));
            let has_active = clean_e > 1.0e-12;
            let tw = if has_active {
                load_q7_words_64(&anchor_phases[0])
            } else {
                [0u64; NUM_SWARM_MICRO_CELLS]
            };
            (tw, clean_e, 0u32, has_active)
        } else if count == 0 {
            ([0u64; NUM_SWARM_MICRO_CELLS], 0.0, 0u32, false)
        } else {
            let (tw, tp, _mc, did, ac) =
                branchless_q7_unified_field_fold_words_with_init_weight_64(
                    &cur_w,
                    0.0,
                    anchor_phases,
                    anchor_base_energies,
                );
            (tw, tp, did, ac > 0)
        };

        if has_active_anchors {
            self.target_anchor_id = dom_id;
            self.cluster_id = dom_id;
            let raw_pull = branchless_flush_subnormal_f32(
                branchless_relu_f32(self.attractor_pull_force) + total_pull,
            );
            self.attractor_pull_force = branchless_min_f32(raw_pull, MAX_ATTRACTOR_PULL_FORCE);
        }

        let (mut report, mut next_words) = self.try_resolve_action_in_swarm_with_cur_words(
            action_ptr,
            &cur_w,
            &tgt_w,
            has_active_anchors,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        if (count > 1)
            & (report.hot_path_resolved == 0)
            & ((self.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0)
            & has_active_anchors
        {
            self.recruit_multi_anchor_stragglers_cold(
                action_ptr,
                &mut report,
                &mut next_words,
                &anchor_phases[dom_id as usize],
                total_pull,
                step_gain_u7,
                qdps_min_step_q7,
                tolerance_q7,
                quorum_threshold,
                entropy_h_tilde,
                low_entropy_skip_thresh,
            );
        }
        (report, next_words)
    }

    #[inline(always)]
    fn resolve_slot_bank_in_swarm_to_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        slot_phase_bank: &[[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS]) {
        let orig_words = *trail_phase_words_ref(self);
        let slot_id = usize::from(action_ptr.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK);
        let target_words = load_q7_words_64(&slot_phase_bank[slot_id]);
        let (mut report, mut next_words) = self.try_resolve_action_in_swarm_with_cur_words(
            action_ptr,
            &orig_words,
            &target_words,
            false,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        if (report.hot_path_resolved == 0) & ((self.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0) {
            self.recruit_slot_bank_stragglers_cold(
                action_ptr,
                slot_id,
                &orig_words,
                &target_words,
                &mut report,
                &mut next_words,
                slot_phase_bank,
                slot_weights,
                coupling_matrix_8x8,
                step_gain_u7,
                qdps_min_step_q7,
                tolerance_q7,
                quorum_threshold,
                entropy_h_tilde,
                low_entropy_skip_thresh,
            );
        }
        (report, next_words)
    }

    #[inline]
    fn fuse_with_peer_swarm_trail(
        &mut self,
        peer_trail: &gcso_pointer_trail_t,
        quorum_threshold: u8,
    ) -> u8 {
        let self_quorum = self.swarm_quorum_mask();
        let peer_quorum = peer_trail.swarm_quorum_mask();
        let combined_quorum = self_quorum | peer_quorum;

        let mut cell = 0usize;
        while cell < NUM_SWARM_MICRO_CELLS {
            let base = cell * HEADS_PER_SWARM_CELL;
            let s_ok = ((self_quorum >> cell) & 1) != 0;
            let p_ok = ((peer_quorum >> cell) & 1) != 0;
            let s_w = load_q7_word_at(&self.accumulated_phase_delta, base);
            let p_w = load_q7_word_at(&peer_trail.accumulated_phase_delta, base);
            let avg_w = swar_q7_avg_round_x8(s_w, p_w);
            let adopt_w = branchless_select_u64(p_ok, p_w, s_w);
            let fused_w = branchless_select_u64(
                s_ok & p_ok,
                avg_w,
                branchless_select_u64(s_ok, s_w, adopt_w),
            );
            store_q7_word_at(&mut self.accumulated_phase_delta, base, fused_w);
            cell += 1;
        }

        let newly_resolved = (combined_quorum & !self_quorum).count_ones();
        let boosted_rho = branchless_flush_subnormal_f32(branchless_relu_f32(
            self.stigmergic_density,
        )) + 0.05 * (newly_resolved as f32);
        self.stigmergic_density = branchless_min_f32(boosted_rho, MAX_STIGMERGIC_DENSITY);

        let safe_req = branchless_select_u8(
            quorum_threshold == 0,
            1,
            branchless_select_u8(quorum_threshold > 8, 8, quorum_threshold),
        );
        self.set_swarm_quorum_state(
            combined_quorum,
            (combined_quorum.count_ones() as u8) >= safe_req,
        );
        combined_quorum
    }
}

// ===================================================================
// 7. Dynamic Adaptive Extension Scratchpad (`DaesScratchpad`)
// ===================================================================

/// Lv.1 Capability Trait for the 64-byte Dynamic Adaptive Extension Scratchpad (DAES):
/// $\mathcal{O}(1)$ fast-path tagged pointer shortcuts, 8-byte circular telemetry mini-ledger,
/// and dynamic parameter auto-tuning.
pub trait DaesScratchpad: BranchlessKernel {
    /// Resets active telemetry mini ledger and cache counters in-place.
    ///
    /// # Errors
    /// Returns an error status only if a custom implementor rejects resetting a locked slot.
    #[inline(always)]
    fn reset_telemetry(&mut self, slot: &mut gcso_daes_slot_t) -> GcsoResult<()> {
        slot.telemetry_ring_head = 0;
        slot.telemetry_ring_tail = 0;
        slot.cache_hit_count = 0;
        slot.telemetry_mini_ledger = [0u8; DAES_MINI_LEDGER_LEN];
        Ok(())
    }

    /// Executes fast-path shortcut lookup in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `slot.mode != 0` or no matching non-zero shortcut is registered.
    #[inline(always)]
    fn fast_path_lookup(&self, slot: &gcso_daes_slot_t, input_key: u64) -> GcsoResult<u64> {
        let (hit, val) = self.fast_path_lookup_branchless(slot, input_key);
        if hit {
            Ok(val)
        } else {
            Err(GCSO_ERROR_INVALID_STATE)
        }
    }

    /// Executes strictly branchless $\mathcal{O}(1)$ fast-path lookup across all 4 shortcut registers.
    #[inline(always)]
    #[must_use]
    fn fast_path_lookup_branchless(&self, slot: &gcso_daes_slot_t, input_key: u64) -> (bool, u64) {
        let mask_hit = ((slot.fast_path_bypass_mask & input_key) != 0) & (slot.mode == 0);
        let idx = ((input_key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
        let candidate = slot.fast_path_shortcuts[idx & (DAES_SHORTCUT_SLOTS - 1)];
        let fallback = slot.fast_path_shortcuts[0];
        let resolved = branchless_select_u64(candidate != 0, candidate, fallback);
        let valid_hit = mask_hit & (resolved != 0);
        (valid_hit, branchless_select_u64(valid_hit, resolved, 0))
    }

    /// Branchlessly resolves `input_ptr` against the DAES fast-path shortcut table in $\mathcal{O}(1)$.
    #[inline(always)]
    fn resolve_or_pass_tagged_ptr(
        &self,
        slot: &mut gcso_daes_slot_t,
        input_ptr: gcso_tagged_ptr_t,
    ) -> (bool, gcso_tagged_ptr_t) {
        let (hit, shortcut_raw) = self.fast_path_lookup_branchless(slot, input_ptr.raw());
        slot.cache_hit_count = slot.cache_hit_count.wrapping_add(u32::from(hit));
        let eff_raw = branchless_select_u64(hit, shortcut_raw, input_ptr.raw());
        (hit, gcso_tagged_ptr_t(eff_raw))
    }

    /// Registers a 64-bit tagged pointer shortcut into the DAES fast-path table in $\mathcal{O}(1)$.
    #[inline(always)]
    fn register_fast_path_shortcut(
        &mut self,
        slot: &mut gcso_daes_slot_t,
        input_key: u64,
        target_shortcut: u64,
    ) {
        let idx = ((input_key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
        slot.fast_path_shortcuts[idx & (DAES_SHORTCUT_SLOTS - 1)] = target_shortcut;
        slot.fast_path_bypass_mask |= branchless_select_u64(target_shortcut != 0, input_key, 0);
    }

    /// Pushes a profiling metric byte code (`DAES_METRIC_*`) into the DAES telemetry ring ledger in-place.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_DAES_SCRATCHPAD_FULL`] only if non-overwriting ring semantics are enforced.
    #[inline(always)]
    fn telemetry_push(&mut self, slot: &mut gcso_daes_slot_t, metric_code: u8) -> GcsoResult<()> {
        let head = (slot.telemetry_ring_head as usize) & (DAES_MINI_LEDGER_LEN - 1);
        slot.telemetry_mini_ledger[head] = metric_code;
        slot.telemetry_ring_head = slot.telemetry_ring_head.wrapping_add(1);
        Ok(())
    }

    /// Summarizes the 8-byte `telemetry_mini_ledger` in $\mathcal{O}(1)$ using 8-lane SWAR,
    /// returning `(combined_metric_or_mask, active_entry_count)`.
    #[inline(always)]
    #[must_use]
    fn telemetry_snapshot_metrics(&self, slot: &gcso_daes_slot_t) -> (u8, u32) {
        let word = u64::from_le_bytes(slot.telemetry_mini_ledger);
        let fold32 = (word as u32) | ((word >> 32) as u32);
        let fold16 = (fold32 & 0xFFFF) | (fold32 >> 16);
        let combined_mask = ((fold16 & 0xFF) | (fold16 >> 8)) as u8;
        (combined_mask, swar_u8_nonzero_count_x8(word))
    }

    /// Sets the operating mode of the DAES slot (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC Buffer).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `mode > 2`.
    #[inline(always)]
    fn set_mode(&mut self, slot: &mut gcso_daes_slot_t, mode: u32) -> GcsoResult<()> {
        if mode > 2 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        slot.mode = mode;
        Ok(())
    }

    /// Evaluates telemetry ledger to auto-tune PSPM ratios, RIPA clamps, and EDBC thresholds.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `slot` is in an incompatible mode.
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_daes_slot_t,
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()>;

    /// Returns the cumulative cache hit count recorded in the slot.
    #[inline(always)]
    fn cache_hit_count(&self, slot: &gcso_daes_slot_t) -> u32 {
        slot.cache_hit_count
    }
}

const _: () = {
    assert!(size_of::<gcso_paged_bitmask_t>() == 32);
    assert!(align_of::<gcso_paged_bitmask_t>() == 32);
    assert!(size_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(align_of::<ContrastivePhaseBloomFilter>() == 64);
    assert!(size_of::<gcso_pointer_trail_t>() == 128);
    assert!(align_of::<gcso_pointer_trail_t>() == 128);
    assert!(size_of::<SwarmResolutionReport>() == 16);
    assert!(align_of::<SwarmResolutionReport>() == 16);
    assert!(NUM_SWARM_MICRO_CELLS * HEADS_PER_SWARM_CELL == MAX_ATTENTION_HEADS);

    let (next_w, res_q7, ok) =
        swar_subswarm_resolve_x8(0x1010_1010_1010_1010, 0x2020_2020_2020_2020, 128, 1, 2);
    assert!(next_w == 0x2020_2020_2020_2020);
    assert!(res_q7 == 0 && ok);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::primitives::ATTRACTOR_SLOT_PERSONA;

    struct DummyDaes;
    impl DaesScratchpad for DummyDaes {
        fn evaluate_auto_tune(
            &self,
            _slot: &gcso_daes_slot_t,
            _config_out: &mut gcso_config_t,
        ) -> GcsoResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_paged_bitmask_bloom_trail_swarm_and_daes_constant_time() {
        let mut m1 = gcso_paged_bitmask_t::default();
        let mut m2 = gcso_paged_bitmask_t::default();
        m1.set_bit_branchless(0);
        m1.set_bit_branchless(65);
        m1.set_bit_branchless(255);
        m2.set_bit_branchless(65);
        m2.set_bit_branchless(128);
        m2.set_bit_branchless(255);

        assert_eq!(m1.popcount(), 3);
        assert_eq!(m1.stigmergic_adjacency_overlap(&m2), 2);
        assert!((m1.stigmergic_cosine_similarity(&m2) - (2.0 / 3.0)).abs() < 1e-4);
        assert!((m1.stigmergic_jaccard_similarity(&m2) - 0.5).abs() < 1e-4);

        let mut filter = ContrastivePhaseBloomFilter::new();
        let mut pattern = [12i8; MAX_ATTENTION_HEADS];
        pattern[63] = -40;
        filter.insert_repulsive_pattern(&pattern);
        filter.insert_repulsive_pattern(&pattern);
        assert_eq!(filter.query_repulsive_density(&pattern), 2);

        let init_ptr = gcso_tagged_ptr_t::pack_action(
            0x4000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            24,
        );
        let (evolved_ptr, density, triggered) =
            filter.apply_repulsion_and_evolve_ptr(init_ptr, &mut pattern, 2, 128);
        assert!(triggered && density == 2 && pattern[0] == -12 && pattern[63] == 40);
        assert_eq!(evolved_ptr.target_attractor_slot(), ATTRACTOR_SLOT_REPULSION);

        for step in 0..100u8 {
            let p = [(step as i8).wrapping_mul(3); MAX_ATTENTION_HEADS];
            filter.insert_repulsive_pattern_weighted(&p, 8);
            assert!(filter.total_mass() <= DEFAULT_BLOOM_MAX_MASS_THRESHOLD + 32);
        }

        let mut trail = gcso_pointer_trail_t {
            stigmergic_density: 1.0,
            target_anchor_id: 42,
            ..Default::default()
        };
        trail.accumulate_uniform_phase(16);
        let (r_bar, mean_q7, kappa, grav) = trail.evaluate_trail_circular_stats_and_gravity();
        assert!(r_bar > 0.99 && mean_q7 == 16 && kappa > 10.0 && grav > 1.0);

        let mut target_phases = [24i8; MAX_ATTENTION_HEADS];
        target_phases[48..64].fill(112);

        let report = trail.try_resolve_action_in_swarm(
            init_ptr,
            &target_phases,
            64,
            1,
            DEFAULT_SWARM_TOLERANCE_Q7,
            DEFAULT_SWARM_QUORUM_THRESHOLD,
            0.10,
            0.15,
            None,
            2,
            128,
        );
        assert_eq!(report.hot_path_resolved, 1);
        assert_eq!(report.resolved_cell_count, 6);
        assert_eq!(report.quorum_mask, 0b0011_1111);
        assert_eq!(trail.cluster_id, 42);
        assert_eq!(report.evolved_ptr.target_attractor_slot(), 42 & 7);

        let unified_anchors = [[0i8; MAX_ATTENTION_HEADS], [112i8; MAX_ATTENTION_HEADS]];
        let unified_energies = [0.1f32, 2.5f32];
        let unified_quorum = trail.recruit_unified_field_anchor_subswarms(
            &unified_anchors,
            &unified_energies,
            128,
            1,
            DEFAULT_SWARM_TOLERANCE_Q7,
            DEFAULT_SWARM_QUORUM_THRESHOLD,
        );
        assert_eq!(unified_quorum, 0xFF);
        assert_eq!(trail.target_anchor_id, 1);
        assert_eq!(trail.cluster_id, 1);

        let mut daes_ctrl = DummyDaes;
        let mut daes_slot = gcso_daes_slot_t::default();
        daes_ctrl.register_fast_path_shortcut(&mut daes_slot, init_ptr.raw(), 0x9999);
        let (hit, resolved_ptr) = daes_ctrl.resolve_or_pass_tagged_ptr(&mut daes_slot, init_ptr);
        assert!(hit && resolved_ptr.raw() == 0x9999);
        daes_ctrl
            .telemetry_push(
                &mut daes_slot,
                DAES_METRIC_FAST_PATH_HIT | DAES_METRIC_PHASE_STEERED,
            )
            .unwrap();
        let (mask, count) = daes_ctrl.telemetry_snapshot_metrics(&daes_slot);
        assert_eq!(
            (mask, count),
            (DAES_METRIC_FAST_PATH_HIT | DAES_METRIC_PHASE_STEERED, 1)
        );
    }

    #[test]
    fn test_liminal_bridge_order_invariant_unified_field_and_bayesian_pointer_rewire() {
        let mut mem_a = gcso_pointer_trail_t {
            current_ptr: 0x1000,
            step_count: 5,
            target_anchor_id: 42,
            attractor_pull_force: 2.0,
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let mut mem_b = gcso_pointer_trail_t {
            current_ptr: 0x2000,
            stigmergic_density: 1.0,
            attractor_pull_force: 1.0,
            accumulated_phase_delta: [48i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let (coh, log_prob, repulsed) =
            mem_a.link_with_next_hallucinated_trail(9, &mut mem_b, 64, 0.25);
        assert!(!repulsed && coh > 0.90 && log_prob > -0.10);
        assert_eq!(mem_a.linked_trail_id, 9);
        assert_eq!(mem_b.prev_ptr, 0x1000);
        assert_eq!(mem_b.step_count, 6);
        assert_eq!(mem_b.target_anchor_id, 42);
        assert_eq!(mem_b.cluster_id, 42);
        assert_eq!(mem_b.accumulated_phase_delta[0], 40);
        assert_ne!(mem_b.flags & TRAIL_FLAG_SYNERGY_RESONANCE, 0);

        let anchor_phases = [40i8; MAX_ATTENTION_HEADS];
        let (res_pull, res_coh) = mem_b.apply_unified_field_anchor_resonance(&anchor_phases, 1.5);
        assert!((res_coh - 1.0).abs() < 1e-4 && (res_pull - 3.0).abs() < 1e-4);

        let anchors_fwd = [
            [16i8; MAX_ATTENTION_HEADS],
            [40i8; MAX_ATTENTION_HEADS],
            [64i8; MAX_ATTENTION_HEADS],
        ];
        let energies_fwd = [1.0f32, 2.5, 1.0];
        let anchors_rev = [
            [64i8; MAX_ATTENTION_HEADS],
            [40i8; MAX_ATTENTION_HEADS],
            [16i8; MAX_ATTENTION_HEADS],
        ];
        let energies_rev = [1.0f32, 2.5, 1.0];

        let mut trail_fwd = mem_b;
        let mut trail_rev = mem_b;
        let (pull_f, coh_f, cnt_f) =
            trail_fwd.apply_unified_field_anchor_slice_resonance(&anchors_fwd, &energies_fwd, None);
        let (pull_r, coh_r, cnt_r) =
            trail_rev.apply_unified_field_anchor_slice_resonance(&anchors_rev, &energies_rev, None);
        assert_eq!((cnt_f, cnt_r), (3, 3));
        assert!((pull_f - pull_r).abs() < 1e-5 && (coh_f - coh_r).abs() < 1e-5);
        assert_eq!((trail_fwd.target_anchor_id, trail_rev.target_anchor_id), (1, 1));

        let cand_bad = gcso_pointer_trail_t {
            stigmergic_density: 5.0,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let cand_good = gcso_pointer_trail_t {
            stigmergic_density: 3.0,
            accumulated_phase_delta: [42i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let candidates = [cand_bad, cand_good];
        let mut scores = [0.5f32, 0.5f32];
        let (selected_id, best_score, mean_coh, coh_cnt, found) = mem_b
            .select_rewire_bridge_and_modulate_bayesian_trails(
                &candidates,
                Some(&mut scores),
                128,
                2.0,
                0.5,
                0.25,
                0.25,
                64,
                8.0,
            );
        assert!(found && selected_id == 1 && coh_cnt == 1);
        assert_eq!(mem_b.linked_trail_id, 1);
        assert!(best_score > 0.5 && mean_coh > -0.1 && scores[1] > scores[0]);

        let mut mem_contradict = gcso_pointer_trail_t {
            current_ptr: 0x3000,
            stigmergic_density: 2.0,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let (coh_bad, _lp_bad, repulsed_bad) =
            mem_contradict.bridge_liminal_hallucinated_trail(&mem_a, 64, 0.25);
        assert!(repulsed_bad && coh_bad < -0.90);
        assert_eq!(mem_contradict.accumulated_phase_delta[0], 96);
        assert_ne!(mem_contradict.flags & TRAIL_FLAG_REPULSION_ACTIVE, 0);
    }
}