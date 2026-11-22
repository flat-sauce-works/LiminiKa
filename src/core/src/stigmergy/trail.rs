// name: src/core/src/stigmergy/trail.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical Layer 2 [`PointerTrail`] POD descriptor, word accessors,
//! tagged-pointer action mode orchestration, pointer-trail operations, Bayesian trail attention modulation,
//! trail chain cohomology, [`PointerSubstrate`], and [`StigmergicSubstrateOps`].

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use crate::foundation::Q7_FULL_COS_REF;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::foundation::{
    avx2_coherence_from_diff_vec64, avx2_words64_to_vec64_with_uniform, Q7_FULL_COS_REF,
};
use crate::foundation::{
    branchless_abs_f32, branchless_clamp_f32, branchless_fast_transition_log_prob,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_max_f32,
    branchless_min_f32, branchless_phase_proximity_gravity_f32, branchless_popcount_rsqrt_u32,
    branchless_q7_conjugate_64, branchless_q7_inhibit_words_64,
    branchless_q7_synergy_fuse_words_64, branchless_q7_transition_log_prob_words_64,
    branchless_q7_unified_manifold_fold_words_with_flags_64, branchless_rad_to_phase_step_u8,
    branchless_relu_f32, branchless_rsqrt_f32, circular_resultant_from_words_64,
    coherence_from_uniform_diff_word_x8, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64, evaluate_cohomological_obstruction,
    fast_inv_len_f32, fast_ln_1p_bounded_f32, is_word_array_uniform_64, load_q7_words_64,
    pack_action_slot_tag, steer_from_swar_words_core_safe, swar_or_avx2_q7_add_sat_words_64,
    swar_q7_add_sat_x8, swar_q7_circular_add_x8, swar_q7_circular_diff_x8, swar_q7_neg_sat_x8,
    swar_q7_phase_step_filter_words_64, swar_q7_scale_le128_with_bias_x8,
    swar_slerp_words_with_bias_64, swar_slerp_words_with_uniform_flag_64, swar_words_nonzero_64,
    unpack_action_mode_from_tag, unpack_attractor_slot_from_tag, AlignedSlice32, AlignedSliceMut32,
    Pod, RawPhaseProjectable, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE,
    ACTION_MODE_PHASE_STEER, ACTION_MODE_RANK1_INJECT, ACTION_MODE_SUBHEAD_ROUTE,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, DEFAULT_PHASE_GRAVITY_EPS,
    MAX_BAYESIAN_STIGMERGIC_DENSITY, POPCOUNT_RSQRT_LUT_257_REF, SWAR_SPLAT_BYTE_64,
};
pub use crate::foundation::{PagedBitmask, TaggedPtr};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, branchless_select_u64, branchless_select_u8,
    BitCut32, BitCut8, LmnkResult, PhaseQ7, DEFAULT_ENTROPY_EPS, LMNK_ERR_INVALID_ARGUMENT,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, NUM_STIGMERGIC_GRANULES, THRESHOLD_STIGMERGY_WIDE,
};
pub use crate::stigmergy::ledger::LedgerSlot;
use crate::stigmergy::ledger::{
    eval_bitmask_dyadic_ultrametric, ContrastivePhaseBloomFilter, FastPathLedger,
};
use crate::stigmergy::pheromone::PheromoneDensityOps;
use crate::stigmergy::{
    coupled_slot_row_mask, is_slot_pair_coupled, ATTRACTOR_SLOT_REPULSION,
    LEDGER_METRIC_FAST_PATH_HIT, LEDGER_METRIC_PHASE_STEERED, LEDGER_METRIC_REPULSION_APPLIED,
    MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_PHASE_STEER_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SWARM_ESCALATED, TRAIL_FLAG_SWARM_RESOLVED,
    TRAIL_FLAG_SYNERGY_RESONANCE, TRAIL_SWARM_QUORUM_MASK, TRAIL_SWARM_QUORUM_SHIFT,
};
/// Stigmergic pointer trail structure aligned to 64 bytes (2 cache lines, 128 bytes total).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerTrail {
    /// Current 64-bit tagged memory pointer address (`offset 0..8`).
    pub current_ptr: u64,
    /// Previous 64-bit tagged memory pointer address (`offset 8..16`).
    pub prev_ptr: u64,
    /// Extensible caller payload data (`offset 16..24`).
    pub user_data: u64,
    /// Accumulated graph edge traversal cost (`offset 24..28`).
    pub transition_cost: i32,
    /// Total path execution steps taken along this trail (`offset 28..32`).
    pub step_count: u32,
    /// Local stigmergic trail density ($\rho_{\mathrm{stigmergy}}$) for cellular swarm routing (`offset 32..36`).
    pub stigmergic_density: f32,
    /// Full 32-bit identifier of the dominant macro attractor anchor on the unified phase manifold (`offset 36..40`).
    pub target_anchor_id: u32,
    /// Active topological cluster / macro centroid identifier (or 3-bit working lane ID `0..=7`, `offset 40..44`).
    pub cluster_id: u32,
    /// Linked adjacent cellular trail identifier along the pointer chain (`offset 44..48`).
    pub linked_trail_id: u32,
    /// Gravitational pull force ($F_{\mathrm{pull}}$) exerted by nearby attractor(s) (`offset 48..52`).
    pub attractor_pull_force: f32,
    /// Execution status, synergy resonance, Limina beat, and repulsion bitmask flags (`offset 52..56`).
    pub flags: u32,
    /// Explicit header padding aligning `accumulated_phase_delta` to CL1 (`offset 56..64`).
    pub reserved_header: [u8; 8],
    /// Accumulated Q7 phase rotation state across 64 attention heads (`offset 64..128`, 1 cache line).
    pub accumulated_phase_delta: [PhaseQ7; 64],
}

impl Default for PointerTrail {
    #[inline(always)]
    fn default() -> Self {
        // SAFETY: `PointerTrail` is a Canonical Little-Endian `Pod` struct where all-zero bytes are valid.
        unsafe { core::mem::zeroed() }
    }
}

// SAFETY: `PointerTrail` is `#[repr(C, align(64))]` with explicit alignment padding and zero traps.
unsafe impl Pod for PointerTrail {}

const _: () = {
    assert!(core::mem::size_of::<PointerTrail>() == 128);
    assert!(core::mem::align_of::<PointerTrail>() == 64);
    assert!(core::mem::offset_of!(PointerTrail, reserved_header) == 56);
    assert!(core::mem::offset_of!(PointerTrail, accumulated_phase_delta) == 64);
};

/// Borrows `trail.accumulated_phase_delta` (`offset 64` of `#[repr(C, align(64))]` [`PointerTrail`],
/// guaranteed 64-byte aligned) as a zero-copy `&[u64; 8]` SWAR reference on little-endian targets,
/// eliminating 64-byte stack copies and 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF) stalls.
#[inline(always)]
#[must_use]
pub const fn trail_phase_words_ref(trail: &PointerTrail) -> &[u64; 8] {
    const _: () = assert!(cfg!(target_endian = "little"));
    // SAFETY: `PointerTrail` is `#[repr(C, align(64))]` and `accumulated_phase_delta` is at
    // byte offset 64 (64-byte cache-line aligned), with identical 64-byte layout to `[u64; 8]`.
    unsafe { &*(trail.accumulated_phase_delta.as_ptr().cast::<[u64; 8]>()) }
}

/// Mutably borrows `trail.accumulated_phase_delta` (`offset 64` of `#[repr(C, align(64))]`
/// [`PointerTrail`], guaranteed 64-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
#[inline(always)]
#[must_use]
pub const fn trail_phase_words_mut(trail: &mut PointerTrail) -> &mut [u64; 8] {
    const _: () = assert!(cfg!(target_endian = "little"));
    // SAFETY: `PointerTrail` is `#[repr(C, align(64))]` and `accumulated_phase_delta` is at
    // byte offset 64 (64-byte cache-line aligned), with identical 64-byte layout to `[u64; 8]`.
    unsafe {
        &mut *(trail
            .accumulated_phase_delta
            .as_mut_ptr()
            .cast::<[u64; 8]>())
    }
}

impl RawPhaseProjectable for PointerTrail {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.accumulated_phase_delta
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *trail_phase_words_ref(self)
    }
}

#[inline(always)]
#[must_use]
pub(super) const fn precompute_bayesian_trail_scales(
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
        branchless_flush_subnormal_f32(branchless_relu_f32(phase_interference_gain) * inv_sqrt_d);
    let clean_eta = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_prior_gain));
    let clean_link = branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));
    let safe_thresh = branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
    (phase_scale, clean_eta, clean_link, safe_thresh)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
#[inline]
#[must_use]
pub(super) fn eval_candidate_trail_words_with_macro_fast(
    self_w: &[u64; 8],
    self_is_uniform: bool,
    self_is_byte_uniform: bool,
    self_b0: u8,
    cur_linked_id: u32,
    cand_id: u32,
    cand: &PointerTrail,
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
        } else if self_is_byte_uniform && (cw0 == cw0.rotate_right(8)) {
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
        let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
        let link_term = branchless_select_f32(is_linked, clean_link, 0.0);
        let prior_term = if clean_eta > 0.0 {
            clean_eta * fast_ln_1p_bounded_f32(branchless_relu_f32(cand.stigmergic_density))
        } else {
            0.0
        };
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
                        branchless_relu_f32(coherence_from_word_arrays_with_both_uniform_flags_64(
                            win_w,
                            *win_is_uniform,
                            cand_w,
                            cand_uni_for_win,
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
    (bayes_delta, coh, is_coherent, cand_is_uniform)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
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
    cand: &PointerTrail,
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
) -> (
    f32,
    f32,
    bool,
    bool,
    Option<(core::arch::x86_64::__m256i, core::arch::x86_64::__m256i)>,
) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`. Keeps `(self_v0, self_v1)` and
    // `(win_v0, win_v1)` hoisted in YMM registers across the candidate loop and loads each
    // candidate's offset-64 phase profile via 64-byte aligned GPR loads (`trail_phase_words_ref`)
    // to eliminate 64B cache-line splits and 4-to-1 GPR-store-to-YMM-load STLF stalls.
    unsafe {
        use core::arch::x86_64::_mm256_sub_epi8;
        let cw = trail_phase_words_ref(cand);
        let c_uni = if self_is_uniform {
            is_word_array_uniform_64(cw)
        } else {
            false
        };
        let mut cv_opt: Option<(core::arch::x86_64::__m256i, core::arch::x86_64::__m256i)> = None;
        let coh = if c_uni {
            let cw0 = cw[0];
            if cw0 == self_w[0] {
                1.0
            } else if self_is_byte_uniform && (cw0 == cw0.rotate_right(8)) {
                Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(self_w, true, cw, true)
            }
        } else {
            let (cv0, cv1) = crate::foundation::avx2_words64_to_vec64(cw);
            cv_opt = Some((cv0, cv1));
            crate::foundation::avx2_coherence_from_diff_vec64(
                _mm256_sub_epi8(self_v0, cv0),
                _mm256_sub_epi8(self_v1, cv1),
            )
        };
        let cand_not_repulsed = (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
        let eff_prior_link = if is_coherent {
            let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
            let link_term = branchless_select_f32(is_linked, clean_link, 0.0);
            let prior_term = if clean_eta > 0.0 {
                clean_eta * fast_ln_1p_bounded_f32(branchless_relu_f32(cand.stigmergic_density))
            } else {
                0.0
            };
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
                                    crate::foundation::avx2_words64_to_vec64_with_uniform(
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
                            let (cv0, cv1) = match cv_opt {
                                Some(v) => v,
                                None => {
                                    let v = crate::foundation::avx2_words64_to_vec64_with_uniform(
                                        cw, c_uni,
                                    );
                                    cv_opt = Some(v);
                                    v
                                }
                            };
                            branchless_relu_f32(crate::foundation::avx2_coherence_from_diff_vec64(
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
        (bayes_delta, coh, is_coherent, c_uni, cv_opt)
    }
}

// ===================================================================
// 3. Pointer-Driven Action Mode Orchestration (`TaggedPtr`)
// ===================================================================

impl TaggedPtr {
    /// Packs a 48-bit address, 3-bit target working lane (`0..=7`), 5-bit action mode mask, and Q7 phase.
    #[inline(always)]
    #[must_use]
    pub fn pack_action(
        raw_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: PhaseQ7,
    ) -> Self {
        Self::pack(
            raw_addr,
            pack_action_slot_tag(attractor_slot_id, action_mode_flags),
            phase_q7,
        )
    }

    /// Extracts the 5-bit Action Mode execution bitmask (`ACTION_MODE_*`).
    #[inline(always)]
    #[must_use]
    pub fn action_mode_flags(self) -> u8 {
        unpack_action_mode_from_tag(self.unpack_tag())
    }

    /// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub fn target_attractor_slot(self) -> u8 {
        unpack_attractor_slot_from_tag(self.unpack_tag())
    }

    /// Returns `true` if all specified `mode_mask` bits are enabled in the pointer's Action Mode.
    #[inline(always)]
    #[must_use]
    pub fn has_action_mode(self, mode_mask: u8) -> bool {
        (self.action_mode_flags() & mode_mask) == mode_mask
    }

    /// Returns a new tagged pointer with updated Target Attractor Working Lane ID and Action Mode flags.
    #[inline(always)]
    #[must_use]
    pub fn with_action_and_slot(self, attractor_slot_id: u8, action_mode_flags: u8) -> Self {
        self.with_tag(pack_action_slot_tag(attractor_slot_id, action_mode_flags))
    }

    /// Branchlessly mutates the pointer's Action Mode flags, Target Working Lane ID, and embedded
    /// Q7 phase based on bottom-up stigmergic environmental traces.
    #[inline(always)]
    #[must_use]
    pub fn evolve_mode_stigmergically(
        self,
        trail_density: f32,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_rank1_thresh: f32,
    ) -> Self {
        let finite_h = branchless_is_finite_f32(entropy_h_tilde);
        let clean_h = branchless_select_f32(finite_h, entropy_h_tilde, 0.0);
        let clean_rho = branchless_relu_f32(trail_density);

        let active_signal = (!self.is_null()) | (self.unpack_phase() != 0) | (clean_rho > 0.0);
        let m_phase_steer = branchless_select_u8(active_signal, ACTION_MODE_PHASE_STEER, 0);
        let can_skip =
            active_signal & (clean_h <= low_entropy_skip_thresh) & (!repulsion_triggered);
        let m_skip = branchless_select_u8(can_skip, ACTION_MODE_LAYER_SKIP, 0);
        let m_rep = branchless_select_u8(repulsion_triggered, ACTION_MODE_PHASE_REPULSE, 0);
        let need_subhead = (clean_h > low_entropy_skip_thresh) | repulsion_triggered;
        let m_subhead = branchless_select_u8(need_subhead, ACTION_MODE_SUBHEAD_ROUTE, 0);
        let need_rank1 =
            (clean_rho >= high_density_rank1_thresh) & (high_density_rank1_thresh > 0.0);
        let m_rank1 = branchless_select_u8(need_rank1, ACTION_MODE_RANK1_INJECT, 0);

        let next_mode = m_phase_steer | m_skip | m_rep | m_subhead | m_rank1;
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

/// Register-resident Coarse-to-Fine Bayesian candidate trail selector, rewirer, Limina geodesic
/// bridger, and Attention score modulator accepting `self_w` directly in 8 `u64` SWAR registers
/// and returning the post-bridged 8 `u64` SWAR words (`0` redundant stack/memory reloads).
#[inline]
pub(crate) fn select_rewire_bridge_and_modulate_with_preloaded_words_64(
    trail: &mut PointerTrail,
    mut self_w: [u64; NUM_STIGMERGIC_GRANULES],
    candidate_trails: &[PointerTrail],
    mut attention_scores_opt: Option<&mut [f32]>,
    head_dim: u32,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    contradiction_threshold: f32,
    liminal_interpolation_u7: u8,
    repulsion_penalty: f32,
    macro_anchor_id: u32,
    win_w_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
    cluster_bonus_gain: f32,
) -> (
    u32,
    f32,
    f32,
    u32,
    bool,
    [u64; NUM_STIGMERGIC_GRANULES],
    f32,
) {
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
    let self_is_byte_uniform = self_is_uniform & (self_w[0] == self_w[0].rotate_right(8));
    // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and `self_w` has 8 `u64` words.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (self_v0, self_v1) =
        unsafe { crate::foundation::avx2_words64_to_vec64_with_uniform(&self_w, self_is_uniform) };
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let mut best_vec_opt: Option<(core::arch::x86_64::__m256i, core::arch::x86_64::__m256i)> = None;
    let mut found_valid = false;
    let mut coh_sum = 0.0_f32;
    let mut coherent_count = 0u32;

    let mut idx = 0usize;
    while idx < cand_len {
        let cand = &candidate_trails[idx];
        let cand_id = idx as u32;
        // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and all trail arguments are valid references.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (bayes_delta, coh, is_coherent, cand_is_uniform, cand_vec_opt) = unsafe {
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
                scores[idx] = branchless_flush_subnormal_f32(scores[idx] + bayes_delta - penalty);
            }
        }

        let take = is_coherent & ((!found_valid) | (bayes_delta > best_score));
        if take {
            best_id = cand_id;
            best_idx = idx;
            best_score = bayes_delta;
            best_coh = coh;
            best_is_uniform = cand_is_uniform;
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                best_vec_opt = cand_vec_opt;
            }
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
            if !(self_is_uniform & best_is_uniform & (self_w[0] == best_w[0])) {
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
                            // SAFETY: `(self_v0, self_v1)` and `best_vec_opt` are already hoisted in YMM
                            // registers across the candidate loop, eliminating post-loop memory reloads.
                            unsafe {
                                let (bv0, bv1) = match best_vec_opt {
                                    Some(v) => v,
                                    None => crate::foundation::avx2_words64_to_vec64_with_uniform(
                                        best_w,
                                        best_is_uniform,
                                    ),
                                };
                                let (rv0, rv1) = crate::foundation::avx2_slerp_vec64(
                                    self_v0, self_v1, bv0, bv1, eff_s,
                                );
                                self_w = crate::foundation::avx2_vec64_to_words64(rv0, rv1);
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
                scores[best_idx] = branchless_flush_subnormal_f32(scores[best_idx] + clean_link);
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

impl PointerTrail {
    /// Copies `src` and pre-loaded `phase_words` into `self` using width-matched scalar writes and
    /// 64-byte aligned 64-bit GPR stores (`*trail_phase_words_mut(self) = *phase_words`) at byte
    /// offset `64..128`, avoiding unaligned 256-bit `vmovdqu` cache-line splits, 256-bit-to-64-bit
    /// Store-to-Load Forwarding (STLF) pipeline stalls, and redundant 64-byte reloads from `src`.
    #[inline(always)]
    pub fn copy_from_with_words_fast(
        &mut self,
        src: &Self,
        phase_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ) {
        self.current_ptr = src.current_ptr;
        self.prev_ptr = src.prev_ptr;
        self.user_data = src.user_data;
        self.transition_cost = src.transition_cost;
        self.step_count = src.step_count;
        self.stigmergic_density = src.stigmergic_density;
        self.target_anchor_id = src.target_anchor_id;
        self.cluster_id = src.cluster_id;
        self.linked_trail_id = src.linked_trail_id;
        self.attractor_pull_force = src.attractor_pull_force;
        self.flags = src.flags;
        self.reserved_header = src.reserved_header;
        *trail_phase_words_mut(self) = *phase_words;
    }

    /// Restores `self` from `src` using width-matched scalar writes and 64-bit GPR stores
    /// (`*trail_phase_words_mut(self) = *trail_phase_words_ref(src)`), preventing 256-bit YMM-to-GPR
    /// Store-to-Load Forwarding (STLF) pipeline stalls (`~15-20 cycles`).
    #[inline(always)]
    pub fn copy_from_fast(&mut self, src: &Self) {
        self.copy_from_with_words_fast(src, trail_phase_words_ref(src));
    }

    /// Clears `self` in-place using width-matched 64-bit GPR stores (`0` YMM stores, `0` STLF stalls).
    #[inline(always)]
    pub fn clear_fast(&mut self) {
        self.current_ptr = 0;
        self.prev_ptr = 0;
        self.user_data = 0;
        self.transition_cost = 0;
        self.step_count = 0;
        self.stigmergic_density = 0.0;
        self.target_anchor_id = 0;
        self.cluster_id = 0;
        self.linked_trail_id = 0;
        self.attractor_pull_force = 0.0;
        self.flags = 0;
        self.reserved_header = [0; 8];
        *trail_phase_words_mut(self) = [0u64; NUM_STIGMERGIC_GRANULES];
    }
}

impl PointerTrail {
    /// Records an $\mathcal{O}(1)$ tagged pointer transition and reinforces stigmergic trail density.
    #[inline(always)]
    pub fn record_pointer_transition(&mut self, next_ptr: u64, edge_cost: i32, density_boost: f32) {
        self.prev_ptr = self.current_ptr;
        self.current_ptr = next_ptr;
        self.transition_cost = self.transition_cost.saturating_add(edge_cost);
        self.step_count = self.step_count.wrapping_add(1);
        let raw_rho = branchless_flush_subnormal_f32(branchless_relu_f32(self.stigmergic_density))
            + branchless_flush_subnormal_f32(branchless_relu_f32(density_boost));
        self.stigmergic_density = branchless_min_f32(raw_rho, MAX_STIGMERGIC_DENSITY);
    }

    /// Accumulates a 64-head Q7 phase delta vector into the trail's phase state using branchless saturation.
    #[inline(always)]
    pub fn accumulate_phase_vector(&mut self, delta_phases: &[PhaseQ7; MAX_ATTENTION_HEADS]) {
        let d = load_q7_words_64(delta_phases);
        let w = trail_phase_words_mut(self);
        swar_or_avx2_q7_add_sat_words_64(w, &d);
    }

    /// Accumulates a uniform Q7 phase delta across all 64 heads in-place using 8-lane SWAR saturation.
    #[inline(always)]
    pub fn accumulate_uniform_phase(&mut self, delta_q7: PhaseQ7) {
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

    /// Smoothly blends the trail's 64-head phase trajectory toward `target_phases` along $S^1$.
    #[inline(always)]
    pub fn geodesic_blend_phase_with(
        &mut self,
        target_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
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

    /// Stamps a phase-conjugate repulsion marker ($-\boldsymbol{\Delta\theta}$) onto the pointer trail.
    #[inline(always)]
    pub fn stamp_phase_conjugate_repulsion(&mut self) {
        let w = trail_phase_words_mut(self);
        w[0] = swar_q7_neg_sat_x8(w[0]);
        w[1] = swar_q7_neg_sat_x8(w[1]);
        w[2] = swar_q7_neg_sat_x8(w[2]);
        w[3] = swar_q7_neg_sat_x8(w[3]);
        w[4] = swar_q7_neg_sat_x8(w[4]);
        w[5] = swar_q7_neg_sat_x8(w[5]);
        w[6] = swar_q7_neg_sat_x8(w[6]);
        w[7] = swar_q7_neg_sat_x8(w[7]);
        self.current_ptr = TaggedPtr(self.current_ptr).conjugate_phase().raw();
        self.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
    }

    /// Applies subnormal-safe temporal evaporative decay (`decay_factor` $\in [0, 1]$) to trail density.
    #[inline(always)]
    pub fn apply_trail_evaporation(&mut self, decay_factor: f32) {
        let _ = self.evaporate_pheromone(decay_factor);
    }

    /// Evaluates whether the trail's stigmergic density exceeds `anchor_threshold`.
    #[inline(always)]
    pub fn should_crystallize_branchless(&self, anchor_threshold: f32) -> bool {
        (self.stigmergic_density >= anchor_threshold) & (anchor_threshold > 0.0)
    }

    /// Returns `current_ptr` wrapped as a strongly-typed [`TaggedPtr`].
    #[inline(always)]
    pub fn current_tagged_ptr(&self) -> TaggedPtr {
        TaggedPtr(self.current_ptr)
    }
    /// Returns `prev_ptr` wrapped as a strongly-typed [`TaggedPtr`].
    #[inline(always)]
    pub fn prev_tagged_ptr(&self) -> TaggedPtr {
        TaggedPtr(self.prev_ptr)
    }
    /// Extracts the active 5-bit Action Mode bitmask (`ACTION_MODE_*`) embedded in `current_ptr`.
    #[inline(always)]
    pub fn active_action_mode(&self) -> u8 {
        TaggedPtr(self.current_ptr).action_mode_flags()
    }
    /// Extracts the 3-bit Target Attractor Working Lane ID (`0..=7`) embedded in `current_ptr`.
    #[inline(always)]
    pub fn target_attractor_slot(&self) -> u8 {
        TaggedPtr(self.current_ptr).target_attractor_slot()
    }
    /// Extracts the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) embedded in `self.flags`.
    #[inline(always)]
    pub fn swarm_quorum_mask(&self) -> u8 {
        ((self.flags & TRAIL_SWARM_QUORUM_MASK) >> TRAIL_SWARM_QUORUM_SHIFT) as u8
    }

    /// Updates the 8-bit Micro-Swarm Quorum bitmask (`bits 8..=15`) and resolution/escalation flags.
    #[inline(always)]
    pub fn set_swarm_quorum_state(&mut self, quorum_mask: u8, hot_path_resolved: bool) {
        let cleared = self.flags
            & !(TRAIL_SWARM_QUORUM_MASK | TRAIL_FLAG_SWARM_RESOLVED | TRAIL_FLAG_SWARM_ESCALATED);
        let status_bit = branchless_select_u32(
            hot_path_resolved,
            TRAIL_FLAG_SWARM_RESOLVED,
            TRAIL_FLAG_SWARM_ESCALATED,
        );
        self.flags = cleared | (u32::from(quorum_mask) << TRAIL_SWARM_QUORUM_SHIFT) | status_bit;
    }

    /// Packs `(next_addr, attractor_slot_id, action_mode_flags, phase_q7)` and advances the trail.
    #[inline(always)]
    pub fn record_action_transition(
        &mut self,
        next_addr: u64,
        attractor_slot_id: u8,
        action_mode_flags: u8,
        phase_q7: PhaseQ7,
        edge_cost: i32,
        density_boost: f32,
    ) {
        let packed =
            TaggedPtr::pack_action(next_addr, attractor_slot_id, action_mode_flags, phase_q7);
        self.record_pointer_transition(packed.raw(), edge_cost, density_boost);
        self.cluster_id = u32::from(attractor_slot_id & ATTRACTOR_SLOT_ID_MASK);
        let phase_steer_on = (action_mode_flags & ACTION_MODE_PHASE_STEER) != 0;
        let skip_on = (action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags |= branchless_select_u32(phase_steer_on, TRAIL_FLAG_PHASE_STEER_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
    }

    /// Evolves the trail's `current_ptr` Action Mode flags and Target Working Lane ID bottom-up,
    /// preserving 32-bit `target_anchor_id` in `cluster_id` when `target_anchor_id != 0` and
    /// `!repulsion_triggered`.
    #[inline(always)]
    pub fn evolve_trail_mode_stigmergically(
        &mut self,
        entropy_h_tilde: f32,
        repulsion_triggered: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_rank1_thresh: f32,
    ) -> TaggedPtr {
        let evolved = TaggedPtr(self.current_ptr).evolve_mode_stigmergically(
            self.stigmergic_density,
            entropy_h_tilde,
            repulsion_triggered,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_rank1_thresh,
        );
        self.current_ptr = evolved.raw();
        let preserve_unified_anchor = (self.target_anchor_id != 0) & (!repulsion_triggered);
        self.cluster_id = branchless_select_u32(
            preserve_unified_anchor,
            self.target_anchor_id,
            u32::from(evolved.target_attractor_slot()),
        );

        let phase_steer_active = (evolved.action_mode_flags() & ACTION_MODE_PHASE_STEER) != 0;
        let skip_active = (evolved.action_mode_flags() & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags = (self.flags
            & !(TRAIL_FLAG_PHASE_STEER_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | branchless_select_u32(phase_steer_active, TRAIL_FLAG_PHASE_STEER_ACTIVE, 0)
            | branchless_select_u32(skip_active, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0)
            | branchless_select_u32(repulsion_triggered, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        evolved
    }

    /// Evaluates circular phase coherence $\mathcal{C}_{\mathrm{phase}} \in [-1.0, 1.0]$ with `slot_phases`.
    #[inline(always)]
    pub fn phase_coherence_with(&self, slot_phases: &[PhaseQ7; MAX_ATTENTION_HEADS]) -> f32 {
        coherence_from_word_arrays_64(trail_phase_words_ref(self), &load_q7_words_64(slot_phases))
    }

    /// Evaluates the trail's 64-head circular resultant statistics $(\bar{R}_c, \bar{\theta}_{\mathrm{Q7}}, \kappa_c)$
    /// alongside its Phase-Proximity Gravity mass $M_{\mathrm{grav}} = \rho_{\mathrm{stigmergy}} / (1 - \bar{R}_c + \epsilon)$
    /// in $\mathcal{O}(1)$ division-free (`0 fdiv`) mode.
    #[inline(always)]
    pub fn evaluate_trail_circular_stats_and_gravity(&self) -> (f32, PhaseQ7, f32, f32) {
        let (r_bar, mean_q7, kappa) = circular_resultant_from_words_64(trail_phase_words_ref(self));
        let gravity = branchless_phase_proximity_gravity_f32(
            self.stigmergic_density,
            r_bar,
            DEFAULT_PHASE_GRAVITY_EPS,
        );
        (r_bar, mean_q7, kappa, gravity)
    }

    /// Applies **Zero-Hyperparameter Phase-Proximity Resonance** from a single anchor on the
    /// unified phase manifold (`anchor_phases`, `anchor_base_energy`) in strictly constant time (`0` branches).
    #[inline]
    pub fn apply_unified_manifold_anchor_resonance(
        &mut self,
        anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        anchor_base_energy: f32,
    ) -> (f32, f32) {
        let clean_energy = branchless_flush_subnormal_f32(branchless_relu_f32(anchor_base_energy));
        let cur_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(self.attractor_pull_force));
        if clean_energy <= DEFAULT_ENTROPY_EPS {
            self.attractor_pull_force = branchless_min_f32(cur_pull, MAX_ATTRACTOR_PULL_FORCE);
            return (0.0, 0.0);
        }
        let t_w = *trail_phase_words_ref(self);
        let b_w = load_q7_words_64(anchor_phases);
        let t_uni = is_word_array_uniform_64(&t_w);
        let b_uni = is_word_array_uniform_64(&b_w);
        let t_nz = if t_uni {
            t_w[0] != 0
        } else {
            (((t_w[0] | t_w[1]) | (t_w[2] | t_w[3])) | ((t_w[4] | t_w[5]) | (t_w[6] | t_w[7]))) != 0
        };
        let b_nz = if b_uni {
            b_w[0] != 0
        } else {
            (((b_w[0] | b_w[1]) | (b_w[2] | b_w[3])) | ((b_w[4] | b_w[5]) | (b_w[6] | b_w[7]))) != 0
        };
        let coherence = if t_nz & b_nz {
            coherence_from_word_arrays_with_both_uniform_flags_64(&t_w, t_uni, &b_w, b_uni)
        } else {
            0.0
        };
        let proximity_gain = branchless_clamp_f32(1.0 + coherence, 0.25, 2.0);
        let resonant_pull = branchless_flush_subnormal_f32(clean_energy * proximity_gain);
        let new_pull = cur_pull + resonant_pull;
        if resonant_pull > DEFAULT_ENTROPY_EPS {
            if cur_pull <= DEFAULT_ENTROPY_EPS {
                *trail_phase_words_mut(self) = b_w;
            } else {
                let ratio_f32 = branchless_clamp_f32(
                    (resonant_pull * crate::foundation::fast_inv_pos_normal_f32(new_pull)) * 128.0
                        + 0.5,
                    0.0,
                    128.0,
                );
                let s = (ratio_f32 as u8) as u64;
                if s >= 128 {
                    *trail_phase_words_mut(self) = b_w;
                } else if s > 0 {
                    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let (out_w, _) = swar_slerp_words_with_uniform_flag_64(
                        &t_w,
                        t_uni,
                        &b_w,
                        b_uni,
                        s,
                        bias_restore,
                    );
                    *trail_phase_words_mut(self) = out_w;
                }
            }
        }
        self.attractor_pull_force = branchless_min_f32(new_pull, MAX_ATTRACTOR_PULL_FORCE);
        let is_resonant = (coherence > 0.25) & (resonant_pull > 0.0);
        let is_liminal = (coherence < -0.10) & (resonant_pull > 0.0);
        self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(is_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);
        (resonant_pull, coherence)
    }

    /// Applies **Order-Invariant (Commutative) Unified-Manifold Multi-Anchor Resonance** across an
    /// arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional repulsive
    /// phase inhibitor (`repulsion_phases_and_strength`) on a **single unified phase manifold**
    /// using register-resident 8-word SWAR accumulators (`0` branches, `0 fdiv`).
    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional
    /// repulsive phase inhibitor (`repulsion_phases_and_strength`) starting from pre-loaded 8-word
    /// SWAR registers `init_w` (`0` entry reload) and returning `(final_pull_force, mean_coherence, active_anchor_count, post_words)`.
    #[inline]
    pub fn apply_unified_manifold_anchor_slice_resonance_from_words(
        &mut self,
        init_w: [u64; NUM_STIGMERGIC_GRANULES],
        anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[PhaseQ7; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
        let (
            mut run_w,
            running_pull,
            mean_coh,
            dominant_anchor_id,
            active_anchors,
            any_resonant,
            any_liminal,
        ) = branchless_q7_unified_manifold_fold_words_with_flags_64(
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
        let cur_ptr = TaggedPtr(self.current_ptr);
        let updated_ptr = cur_ptr.with_action_and_slot(
            branchless_select_u8(has_active, lane_tag, cur_ptr.target_attractor_slot()),
            cur_ptr.action_mode_flags()
                | branchless_select_u8(has_active, ACTION_MODE_PHASE_STEER, 0),
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

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the single
    /// unified phase manifold and returns `(final_pull_force, mean_coherence, active_anchor_count, post_words)`
    /// directly in 8 `u64` SWAR registers (`0` exit reload).
    #[inline(always)]
    pub fn apply_unified_manifold_anchor_slice_resonance_to_words(
        &mut self,
        anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[PhaseQ7; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
        let init_w = *trail_phase_words_ref(self);
        self.apply_unified_manifold_anchor_slice_resonance_from_words(
            init_w,
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
        )
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional
    /// repulsive phase inhibitor (`repulsion_phases_and_strength`) on a **single unified phase manifold**
    /// using register-resident 8-word SWAR accumulators (`0` branches, `0 fdiv`).
    #[inline(always)]
    pub fn apply_unified_manifold_anchor_slice_resonance(
        &mut self,
        anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[PhaseQ7; MAX_ATTENTION_HEADS], f32)>,
    ) -> (f32, f32, u32) {
        let (final_pull, mean_coh, active_anchors, _words) = self
            .apply_unified_manifold_anchor_slice_resonance_to_words(
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
            );
        (final_pull, mean_coh, active_anchors)
    }

    /// Bridges the "unwritten space between the lines" (**Limina / Inter-Memory Hallucination**)
    /// from `prev_memory_trail` ($M_{t-1}$) to `self` ($M_t$) in a **single fused 8-word SWAR pass**
    /// (`0` branches, zero temporary array stack spills).
    #[inline]
    pub fn bridge_liminal_hallucinated_trail(
        &mut self,
        prev_memory_trail: &PointerTrail,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool) {
        let p = trail_phase_words_ref(prev_memory_trail);
        let s = trail_phase_words_ref(self);
        let p_uni = is_word_array_uniform_64(p);
        let both_uni = p_uni && is_word_array_uniform_64(s);
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
                let lp = branchless_fast_transition_log_prob(c_val);
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
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                // SAFETY: `target_feature = "avx2"` is enabled; evaluates 64-head circular difference,
                // cosine coherence, and geodesic SLERP in two 256-bit YMM registers.
                unsafe {
                    use core::arch::x86_64::{
                        _mm256_max_epi8, _mm256_or_si256, _mm256_set1_epi8, _mm256_setzero_si256,
                        _mm256_sub_epi8, _mm256_testz_si256,
                    };
                    let (vp0, vp1) =
                        crate::foundation::avx2_words64_to_vec64_with_uniform(p, p_uni);
                    let (vs0, vs1) = crate::foundation::avx2_words64_to_vec64(s);
                    let vd0 = _mm256_sub_epi8(vs0, vp0);
                    let vd1 = _mm256_sub_epi8(vs1, vp1);
                    let vd_or = _mm256_or_si256(vd0, vd1);
                    if _mm256_testz_si256(vd_or, vd_or) != 0 {
                        (1.0_f32, 0.0_f32, false, *s, false)
                    } else {
                        let c_val = crate::foundation::avx2_coherence_from_diff_vec64(vd0, vd1);
                        let lp = branchless_fast_transition_log_prob(c_val);
                        let contr = c_val < -safe_thresh;
                        if contr {
                            let zero = _mm256_setzero_si256();
                            let min_m127 = _mm256_set1_epi8(-127);
                            let r0 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs0, min_m127));
                            let r1 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs1, min_m127));
                            (
                                c_val,
                                lp,
                                true,
                                crate::foundation::avx2_vec64_to_words64(r0, r1),
                                true,
                            )
                        } else if t_s >= 128 {
                            (c_val, lp, false, *s, false)
                        } else if t_s == 0 {
                            (c_val, lp, false, *p, true)
                        } else {
                            let (r0, r1) = crate::foundation::avx2_slerp_from_diff_vec64(
                                vp0, vp1, vd0, vd1, t_s,
                            );
                            (
                                c_val,
                                lp,
                                false,
                                crate::foundation::avx2_vec64_to_words64(r0, r1),
                                true,
                            )
                        }
                    }
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
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
                    let c_val = crate::foundation::coherence_from_diff_words_64(
                        d0, d1, d2, d3, d4, d5, d6, d7,
                    );
                    let lp = branchless_fast_transition_log_prob(c_val);
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

        let cur_ptr = TaggedPtr(self.current_ptr);
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

    /// Links `self` ($M_{t-1}$) to `next_trail` ($M_t$, identified by `next_trail_id`) by setting
    /// `self.linked_trail_id = next_trail_id` and bridging their inter-memory **Limina** gap.
    #[inline(always)]
    pub fn link_with_next_hallucinated_trail(
        &mut self,
        next_trail_id: u32,
        next_trail: &mut PointerTrail,
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

    /// Evaluates the **Bayes-Attention Stigmergic Transition Score** from `self` ($M_t$) to a
    /// candidate memory trail `candidate_trail` ($M_j$, identified by `candidate_trail_id`) in
    /// strictly constant time (`0` branches, `0 fdiv`).
    #[inline(always)]
    pub fn evaluate_bayesian_transition_score_with(
        &self,
        candidate_trail_id: u32,
        candidate_trail: &PointerTrail,
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
    ) -> (f32, f32, bool) {
        let is_linked = (self.linked_trail_id != 0) & (self.linked_trail_id == candidate_trail_id);
        let (phase_scale, clean_eta, clean_link, safe_thresh) = precompute_bayesian_trail_scales(
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
        );
        let coh = coherence_from_word_arrays_64(
            trail_phase_words_ref(self),
            trail_phase_words_ref(candidate_trail),
        );
        let attn_delta = branchless_flush_subnormal_f32(phase_scale * coh);
        let clean_rho = branchless_clamp_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(candidate_trail.stigmergic_density)),
            0.0,
            MAX_STIGMERGIC_DENSITY,
        );
        let prior_delta =
            branchless_flush_subnormal_f32(clean_eta * fast_ln_1p_bounded_f32(clean_rho));
        let link_delta = branchless_select_f32(is_linked, clean_link, 0.0);
        let total_score = branchless_flush_subnormal_f32(attn_delta + prior_delta + link_delta);
        let cand_not_repulsed = (candidate_trail.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
        (total_score, coh, is_coherent)
    }

    /// Synchronizes `self.flags` (`TRAIL_FLAG_PHASE_STEER_ACTIVE` and `TRAIL_FLAG_INERTIAL_SKIP_ACTIVE`)
    /// with the 5-bit `ACTION_MODE_*` flags embedded in `self.current_ptr` in branchless constant time.
    #[inline(always)]
    pub fn sync_flags_with_current_ptr(&mut self) {
        let mode = unpack_action_mode_from_tag(TaggedPtr(self.current_ptr).unpack_tag());
        let phase_steer_on = (mode & ACTION_MODE_PHASE_STEER) != 0;
        let skip_on = (mode & ACTION_MODE_LAYER_SKIP) != 0;
        let cleared =
            self.flags & !(TRAIL_FLAG_PHASE_STEER_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE);
        self.flags = cleared
            | branchless_select_u32(phase_steer_on, TRAIL_FLAG_PHASE_STEER_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
    }

    /// Evaluates non-linear cross-slot synergy and liminal beat interference between two working lanes.
    #[inline]
    pub fn apply_cross_slot_synergy(
        &mut self,
        slot_a: u32,
        phases_a: &[PhaseQ7; MAX_ATTENTION_HEADS],
        weight_a: f32,
        slot_b: u32,
        phases_b: &[PhaseQ7; MAX_ATTENTION_HEADS],
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
        swar_or_avx2_q7_add_sat_words_64(cur_w, &fw);
        let new_pull = branchless_flush_subnormal_f32(
            branchless_relu_f32(self.attractor_pull_force) + synergy_pull,
        );
        self.attractor_pull_force = branchless_min_f32(new_pull, MAX_ATTRACTOR_PULL_FORCE);

        let dom_slot = branchless_select_u32(weight_a >= weight_b, slot_a, slot_b) as u8;
        let cur_ptr = TaggedPtr(self.current_ptr);
        self.current_ptr = cur_ptr
            .with_action_and_slot(
                dom_slot,
                cur_ptr.action_mode_flags() | ACTION_MODE_PHASE_STEER,
            )
            .raw();
        self.cluster_id = u32::from(dom_slot & ATTRACTOR_SLOT_ID_MASK);

        let is_resonant = coupled & (coherence > 0.25) & (synergy_pull > 0.0);
        let is_liminal_beat = coupled & (coherence < -0.10) & (synergy_pull > 0.0);
        self.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(is_liminal_beat, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        (synergy_pull, coherence)
    }

    /// Applies branchless mutual inhibition gating to the trail's accumulated phase vector and pull force.
    #[inline]
    pub fn apply_mutual_slot_inhibition(
        &mut self,
        inhibitor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        inhibition_strength: f32,
    ) -> f32 {
        let cur_pull = self.attractor_pull_force;
        let inh_w = load_q7_words_64(inhibitor_phases);
        let raw_damped = branchless_q7_inhibit_words_64(
            trail_phase_words_mut(self),
            cur_pull,
            &inh_w,
            inhibition_strength,
        );
        let damped_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(raw_damped),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        self.attractor_pull_force = damped_pull;
        damped_pull
    }

    /// Register-resident evaluator for **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition**
    /// starting from pre-loaded 8-word SWAR registers `acc_w` and returning
    /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)`.
    #[inline]
    pub fn apply_neighborhood_synergy_from_bank_from_words(
        &mut self,
        mut acc_w: [u64; NUM_STIGMERGIC_GRANULES],
        slot_phase_bank: &[[PhaseQ7; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
        let primary_slot = u32::from(self.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK);
        let p_idx = primary_slot as usize;
        let primary_phase = &slot_phase_bank[p_idx];
        let primary_w = load_q7_words_64(primary_phase);
        let primary_pull = branchless_flush_subnormal_f32(branchless_relu_f32(slot_weights[p_idx]));

        let exclude_mask = (1u8 << p_idx) | (1u8 << ATTRACTOR_SLOT_REPULSION);
        let mut neighbor_mask = BitCut8::split(
            coupled_slot_row_mask(coupling_matrix_8x8, primary_slot),
            exclude_mask,
        )
        .outside;

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
            swar_or_avx2_q7_add_sat_words_64(&mut acc_w, &cur_w);
        }
        let mut total_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(self.attractor_pull_force))
                + running_pull;

        total_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(total_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        let safe_neighbors = branchless_select_u32(active_neighbors > 0, active_neighbors, 1);
        let mean_coh = branchless_select_f32(
            active_neighbors > 0,
            coh_sum * fast_inv_len_f32(safe_neighbors as usize),
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

    /// Evaluates **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition** and returns both
    /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)` in registers (`0` exit reload).
    #[inline(always)]
    pub fn apply_neighborhood_synergy_from_bank_to_words(
        &mut self,
        slot_phase_bank: &[[PhaseQ7; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
    ) -> (f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES]) {
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

    /// Unified single-source-of-truth evaluator for **8-Lane Coupled Neighborhood Synergy & Mutual Inhibition**
    /// across `slot_phase_bank` (`[[PhaseQ7; 64]; 8]`) and `slot_weights` (`[f32; 8]`).
    #[inline(always)]
    pub fn apply_neighborhood_synergy_from_bank(
        &mut self,
        slot_phase_bank: &[[PhaseQ7; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
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

/// Directly modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
/// in-place by the full **Bayes-Attention Stigmergic Transition Score** (`0` branches, `0 fdiv`),
/// preloading `query_phases` into registers and using `fast_ln_1p_bounded_f32`.
#[inline]
pub fn modulate_attention_scores_by_bayesian_trails_64(
    attention_scores: &mut [f32],
    query_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    key_anchor_phases: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
    key_stigmergic_densities: &[f32],
    linked_trail_id: Option<u32>,
    head_dim: u32,
    interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
) -> (f32, u32) {
    let count = attention_scores
        .len()
        .min(key_anchor_phases.len())
        .min(key_stigmergic_densities.len());
    if count == 0 {
        return (0.0, 0);
    }

    let q = load_q7_words_64(query_phases);
    let q_is_uniform = is_word_array_uniform_64(&q);
    let q_w0 = q[0];
    let q_b0 = (q_w0 & 0xFF) as u8;
    let q_is_byte_uniform = q_is_uniform & (q_w0 == q_w0.rotate_right(8));
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let mut vq_vecs: Option<(core::arch::x86_64::__m256i, core::arch::x86_64::__m256i)> =
        if q_is_uniform {
            None
        } else {
            // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and `q` has 8 `u64` words.
            Some(unsafe { avx2_words64_to_vec64_with_uniform(&q, false) })
        };
    let safe_dim = branchless_select_u32(head_dim > 0, head_dim, 128);
    let rsqrt_d = if safe_dim <= 256 {
        POPCOUNT_RSQRT_LUT_257_REF[safe_dim as usize]
    } else {
        branchless_rsqrt_f32(safe_dim as f32)
    };
    let eff_interf_scale =
        branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain) * rsqrt_d);
    let clean_prior_gain = branchless_relu_f32(stigmergic_prior_gain);
    let clean_link_bonus = branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));

    let has_link = linked_trail_id.is_some();
    let link_target = linked_trail_id.unwrap_or(0);
    let mut coh_sum = 0.0_f32;
    let mut best_score = -1.0e30_f32;
    let mut best_idx = 0u32;

    for idx in 0..count {
        let is_linked = has_link & ((idx as u32) == link_target);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let coh = {
            if q_is_uniform {
                let kw = load_q7_words_64(&key_anchor_phases[idx]);
                if is_word_array_uniform_64(&kw) {
                    let kw0 = kw[0];
                    if kw0 == q_w0 {
                        1.0
                    } else if q_is_byte_uniform && (kw0 == kw0.rotate_right(8)) {
                        Q7_FULL_COS_REF[q_b0.wrapping_sub(kw0 as u8) as usize]
                    } else {
                        coherence_from_word_arrays_with_both_uniform_flags_64(&q, true, &kw, true)
                    }
                } else {
                    // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and `key_anchor_phases[idx]` is a 64-byte slice (`_mm256_loadu_si256` allows unaligned loads).
                    unsafe {
                        use core::arch::x86_64::{
                            __m256i, _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_sub_epi8,
                        };
                        let (vq0, vq1) = match vq_vecs {
                            Some(v) => v,
                            None => {
                                let z = _mm256_set1_epi64x(q_w0 as i64);
                                let v = (z, z);
                                vq_vecs = Some(v);
                                v
                            }
                        };
                        let kp = key_anchor_phases[idx].as_ptr().cast::<__m256i>();
                        let vk0 = _mm256_loadu_si256(kp);
                        let vk1 = _mm256_loadu_si256(kp.add(1));
                        avx2_coherence_from_diff_vec64(
                            _mm256_sub_epi8(vq0, vk0),
                            _mm256_sub_epi8(vq1, vk1),
                        )
                    }
                }
            } else {
                // SAFETY: `target_feature = "avx2"` is checked by `#[cfg]`, and `key_anchor_phases[idx]` is 64 bytes (`2 * __m256i`).
                unsafe {
                    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
                    let (vq0, vq1) =
                        vq_vecs.unwrap_or_else(|| avx2_words64_to_vec64_with_uniform(&q, false));
                    let kp = key_anchor_phases[idx].as_ptr().cast::<__m256i>();
                    let vk0 = _mm256_loadu_si256(kp);
                    let vk1 = _mm256_loadu_si256(kp.add(1));
                    avx2_coherence_from_diff_vec64(
                        _mm256_sub_epi8(vq0, vk0),
                        _mm256_sub_epi8(vq1, vk1),
                    )
                }
            }
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let coh = {
            let kw = load_q7_words_64(&key_anchor_phases[idx]);
            if q_is_uniform && is_word_array_uniform_64(&kw) {
                let kw0 = kw[0];
                if kw0 == q_w0 {
                    1.0
                } else if q_is_byte_uniform && (kw0 == kw0.rotate_right(8)) {
                    Q7_FULL_COS_REF[q_b0.wrapping_sub(kw0 as u8) as usize]
                } else {
                    coherence_from_word_arrays_with_both_uniform_flags_64(&q, true, &kw, true)
                }
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(&q, q_is_uniform, &kw, false)
            }
        };
        let attn_delta = branchless_flush_subnormal_f32(eff_interf_scale * coh);
        let clean_rho = branchless_clamp_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(key_stigmergic_densities[idx])),
            0.0,
            MAX_BAYESIAN_STIGMERGIC_DENSITY,
        );
        let prior_delta =
            branchless_flush_subnormal_f32(clean_prior_gain * fast_ln_1p_bounded_f32(clean_rho));
        let link_delta = branchless_select_f32(is_linked, clean_link_bonus, 0.0);
        let delta_z = branchless_flush_subnormal_f32(attn_delta + prior_delta + link_delta);

        let new_score = branchless_flush_subnormal_f32(attention_scores[idx] + delta_z);
        attention_scores[idx] = new_score;
        coh_sum += coh;
        let better = (idx == 0) | (new_score > best_score);
        best_score = branchless_select_f32(better, new_score, best_score);
        best_idx = branchless_select_u32(better, idx as u32, best_idx);
    }

    (
        branchless_clamp_f32(coh_sum * fast_inv_len_f32(count), -1.0, 1.0),
        best_idx,
    )
}

pub(crate) const BLOCK_NORM_CACHE_CAP: usize = 64;

/// Evaluates coarse supernode degrees $D_{ii} = \sum_{j \neq i} A_{ij}$ and returns
/// $\hat{\lambda}_{\max} = 2 \max_i D_{ii}$ directly from 256-bit PagedBlock bitmasks (Section 1.2 & Lemma 1.2).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `block_masks` is empty or `degrees_out.len() < block_masks.len()`.
#[inline]
pub fn eval_stigmergic_block_degrees(
    block_masks: &[PagedBitmask],
    degrees_out: &mut AlignedSliceMut32<'_, f32>,
) -> LmnkResult<f32> {
    let n = block_masks.len();
    if n == 0 || degrees_out.len() < n {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let deg_slice = &mut degrees_out.as_mut_slice()[..n];
    deg_slice.fill(0.0);

    let mut inv_norm_cache = [0.0_f32; BLOCK_NORM_CACHE_CAP];
    let cached_len = n.min(BLOCK_NORM_CACHE_CAP);
    for (slot, mask) in inv_norm_cache[..cached_len]
        .iter_mut()
        .zip(block_masks[..cached_len].iter())
    {
        *slot = branchless_popcount_rsqrt_u32(mask.popcount());
    }

    let mut max_deg = 0.0_f32;
    let cached_upper = cached_len;
    for i in 0..n {
        let mask_i = &block_masks[i];
        let inv_norm_i = if i < BLOCK_NORM_CACHE_CAP {
            inv_norm_cache[i]
        } else {
            branchless_popcount_rsqrt_u32(mask_i.popcount())
        };

        if inv_norm_i > 0.0 {
            let [m0, m1, m2, m3] = mask_i.bits;
            let mut acc_i0 = deg_slice[i];
            let mut acc_i1 = 0.0_f32;
            let mut j = i + 1;

            while j + 4 <= cached_upper {
                let [w0_a, w1_a, w2_a, w3_a] = block_masks[j].bits;
                let [w0_b, w1_b, w2_b, w3_b] = block_masks[j + 1].bits;
                let [w0_c, w1_c, w2_c, w3_c] = block_masks[j + 2].bits;
                let [w0_d, w1_d, w2_d, w3_d] = block_masks[j + 3].bits;

                let inter_a = ((m0 & w0_a).count_ones() + (m1 & w1_a).count_ones())
                    + ((m2 & w2_a).count_ones() + (m3 & w3_a).count_ones());
                let inter_b = ((m0 & w0_b).count_ones() + (m1 & w1_b).count_ones())
                    + ((m2 & w2_b).count_ones() + (m3 & w3_b).count_ones());
                let inter_c = ((m0 & w0_c).count_ones() + (m1 & w1_c).count_ones())
                    + ((m2 & w2_c).count_ones() + (m3 & w3_c).count_ones());
                let inter_d = ((m0 & w0_d).count_ones() + (m1 & w1_d).count_ones())
                    + ((m2 & w2_d).count_ones() + (m3 & w3_d).count_ones());

                let a_ij_a = (inter_a as f32) * (inv_norm_i * inv_norm_cache[j]);
                let a_ij_b = (inter_b as f32) * (inv_norm_i * inv_norm_cache[j + 1]);
                let a_ij_c = (inter_c as f32) * (inv_norm_i * inv_norm_cache[j + 2]);
                let a_ij_d = (inter_d as f32) * (inv_norm_i * inv_norm_cache[j + 3]);

                deg_slice[j] += a_ij_a;
                deg_slice[j + 1] += a_ij_b;
                deg_slice[j + 2] += a_ij_c;
                deg_slice[j + 3] += a_ij_d;
                acc_i0 += a_ij_a + a_ij_b;
                acc_i1 += a_ij_c + a_ij_d;
                j += 4;
            }
            while j < cached_upper {
                let [w0, w1, w2, w3] = block_masks[j].bits;
                let inter = ((m0 & w0).count_ones() + (m1 & w1).count_ones())
                    + ((m2 & w2).count_ones() + (m3 & w3).count_ones());
                let a_ij = (inter as f32) * (inv_norm_i * inv_norm_cache[j]);
                deg_slice[j] += a_ij;
                acc_i0 += a_ij;
                j += 1;
            }
            for j_tail in cached_upper.max(i + 1)..n {
                let a_ij = mask_i
                    .stigmergic_cosine_similarity_with_inv_norm(inv_norm_i, &block_masks[j_tail]);
                acc_i0 += a_ij;
                deg_slice[j_tail] += a_ij;
            }
            deg_slice[i] = acc_i0 + acc_i1;
        }
        max_deg = branchless_max_f32(max_deg, deg_slice[i]);
    }
    Ok(branchless_flush_subnormal_f32(2.0 * max_deg))
}

/// Detects cohomological obstruction $H^1(K; \mathcal{F})$ and records the offending phase
/// pattern into `bloom_filter` while returning the phase-conjugate anti-phase vector ($-\boldsymbol{\Delta\theta}$).
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if slice lengths mismatch or are empty.
/// Returns [`crate::liminal::LMNK_ERR_ENTROPY_SINGULARITY`] if residual norms are non-finite.
#[inline]
pub fn detect_and_record_obstruction_repulsion(
    cochain_residual_r: &AlignedSlice32<'_, f32>,
    exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
    obstruction_threshold: f32,
    active_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    bloom_filter: &mut ContrastivePhaseBloomFilter,
) -> LmnkResult<(f32, bool, [PhaseQ7; MAX_ATTENTION_HEADS])> {
    let (mean_obs, is_obstructed) = evaluate_cohomological_obstruction(
        cochain_residual_r,
        exact_projection_delta0_phi,
        obstruction_threshold,
    )?;
    if is_obstructed {
        bloom_filter.insert_repulsive_pattern(active_phases);
        Ok((mean_obs, true, branchless_q7_conjugate_64(active_phases)))
    } else {
        Ok((mean_obs, false, *active_phases))
    }
}

/// Evaluates the **Joint Probability Chain Rule Log-Likelihood**
/// $\ln P(M_1, \dots, M_T) = \sum_{t=1}^{T-1} \ln P(M_t \mid M_{t-1})$ and **$S^1$ Sheaf
/// Holonomy Obstruction Energy** ($H^1(K; \mathcal{F})$) across `trail_chain` in $\mathcal{O}(T)$ time.
///
/// Returns `(joint_log_prob_sum, holonomy_obstruction_energy, is_obstructed)`.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `trail_chain` is empty or `obstruction_threshold` is non-finite.
#[inline]
pub fn evaluate_trail_chain_cohomology_and_joint_log_prob(
    trail_chain: &[PointerTrail],
    obstruction_threshold: f32,
) -> LmnkResult<(f32, f32, bool)> {
    let len = trail_chain.len();
    if len == 0 || !branchless_is_finite_f32(obstruction_threshold) {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    if len == 1 {
        return Ok((0.0, 0.0, false));
    }

    let mut joint_log_prob = 0.0_f32;
    let mut coboundary_defect_sum = 0.0_f32;
    let first_w = *trail_phase_words_ref(&trail_chain[0]);
    let mut prev_w = first_w;
    for trail in &trail_chain[1..len] {
        let cur_w = *trail_phase_words_ref(trail);
        let (coh, log_p) = branchless_q7_transition_log_prob_words_64(&prev_w, &cur_w);
        joint_log_prob += log_p;
        coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - coh));
        prev_w = cur_w;
    }

    let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
    let loop_coh = coherence_from_word_arrays_64(&first_w, &prev_w);
    let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
    let holonomy_energy = branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
    let is_obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);

    Ok((
        branchless_flush_subnormal_f32(joint_log_prob),
        holonomy_energy,
        is_obstructed,
    ))
}

/// Bridges all consecutive inter-memory gaps (**Limina**) across `trail_chain` via
/// [`PointerTrail::bridge_liminal_hallucinated_trail`], recording any self-contradictory
/// transitions ($H^1(K; \mathcal{F}) \neq 0$) into `bloom_filter`.
///
/// Returns `(total_joint_log_prob, repulsed_transition_count)`.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `trail_chain` is empty or `contradiction_threshold` is non-finite.
#[inline]
pub fn bridge_and_filter_liminal_trail_chain(
    trail_chain: &mut [PointerTrail],
    interpolation_u7: u8,
    contradiction_threshold: f32,
    bloom_filter: &mut ContrastivePhaseBloomFilter,
) -> LmnkResult<(f32, u32)> {
    let len = trail_chain.len();
    if len == 0 || !branchless_is_finite_f32(contradiction_threshold) {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let mut total_log_prob = 0.0_f32;
    let mut repulsed_count = 0u32;

    for idx in 1..len {
        let (head, tail) = trail_chain.split_at_mut(idx);
        let prev_ref = &head[idx - 1];
        let cur_mut = &mut tail[0];
        let (_coh, log_p, repulsed) = cur_mut.bridge_liminal_hallucinated_trail(
            prev_ref,
            interpolation_u7,
            contradiction_threshold,
        );
        total_log_prob += log_p;
        if repulsed {
            bloom_filter.insert_repulsive_pattern_words_weighted(trail_phase_words_ref(cur_mut), 1);
            repulsed_count = repulsed_count.wrapping_add(1);
        }
    }
    Ok((
        branchless_flush_subnormal_f32(total_log_prob),
        repulsed_count,
    ))
}

/// Evaluates 256-bit dyadic distance $d_{\mathcal{V}}(x, y) \in [0, 1)$, quantizes it into a
/// discrete QDPT level, and cascades `parent_ptr` in $\mathcal{O}(1)$.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `qdpt_addr_offsets` is empty or its length does
/// not match `qdpt_phase_deltas.len()`.
#[inline(always)]
pub fn resolve_dyadic_qdpt_and_cascade_ptr(
    parent_ptr: TaggedPtr,
    current_mask: &PagedBitmask,
    anchor_mask: &PagedBitmask,
    w_key: f32,
    current_level: u32,
    hysteresis_band: f32,
    qdpt_addr_offsets: &[u64],
    qdpt_phase_deltas: &[PhaseQ7],
) -> LmnkResult<(u32, f32, TaggedPtr)> {
    let table_len = qdpt_addr_offsets.len();
    if table_len == 0 || table_len != qdpt_phase_deltas.len() {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let max_level = (table_len - 1) as u32;
    let d_v = eval_bitmask_dyadic_ultrametric(current_mask, anchor_mask, w_key);
    let clamped_cur = branchless_select_u32(current_level <= max_level, current_level, max_level);
    let level =
        crate::foundation::quantize_depth_hysteresis(d_v, clamped_cur, max_level, hysteresis_band);
    let idx = (level as usize).min(table_len - 1);
    let cascaded = crate::foundation::cascade_tagged_ptr(
        parent_ptr,
        qdpt_addr_offsets[idx],
        qdpt_phase_deltas[idx],
    );
    Ok((level, d_v, cascaded))
}

/// Hot-path capability trait for $\mathcal{O}(1)$ tagged pointer transitions and trail updates.
pub trait PointerSubstrate: Send + Sync {
    /// Resets transient pointer trails and slot counters without freeing pre-allocated tables.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_INVALID_STATE`] if the substrate is locked or uninitialized.
    fn reset(&mut self) -> LmnkResult<()>;

    /// Advances a 64-bit tagged pointer transition in zero-allocation $\mathcal{O}(1)$ mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] or [`crate::liminal::LMNK_ERR_SUBSTRATE_FULL`] on failure.
    fn step_pointer(&mut self, current_ptr: u64, trail_out: &mut PointerTrail) -> LmnkResult<()>;

    /// Updates cellular state for a trail directly with a Q7 phase delta in fast-path express mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `trail_id` exceeds substrate capacity.
    fn update_cellular_state_express(
        &mut self,
        trail_id: u32,
        phase_delta: PhaseQ7,
    ) -> LmnkResult<()>;

    /// Pulls active pointer trails toward a macro target anchor via pull force $F_{\mathrm{pull}}$.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `trail_id` is out of bounds or `pull_force` is non-finite.
    fn pull_trail_to_attractor(
        &mut self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> LmnkResult<()>;

    /// Links adjacent cellular hallucinated trails into contiguous stigmergic trace graphs.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if either trail ID is out of bounds.
    fn link_hallucinated_trails(&mut self, src_trail_id: u32, dst_trail_id: u32) -> LmnkResult<()>;

    /// Returns the maximum slot capacity of the substrate table.
    fn capacity(&self) -> u32;

    /// Returns the active pointer trail count currently registered.
    fn active_count(&self) -> u32;

    /// Returns the active 8-bit lane enable mask (`0..=7`) on the working phase bank (`0` = none).
    #[inline(always)]
    fn lane_mask(&self) -> u8 {
        0
    }

    /// Returns the 8-bit bitmask of lanes (`0..=7`) tagged with anti-phase repulsion behavior.
    #[inline(always)]
    fn repulsion_lane_mask(&self) -> u8 {
        0
    }

    /// Returns the 8-bit bitmask of lanes (`0..=7`) tagged as isolated exterior-only sandbox hypotheses.
    #[inline(always)]
    fn sandbox_lane_mask(&self) -> u8 {
        0
    }

    /// Returns the 8-bit bitmask of lanes (`0..=7`) explicitly pinned against auto-eviction or release.
    #[inline(always)]
    fn pinned_lane_mask(&self) -> u8 {
        0
    }

    /// Returns the 64-head SWAR words and non-negative lane weight for working-bank `lane` (`0..=7`)
    /// when enabled and non-zero.
    #[inline(always)]
    fn lane_words_and_weight(&self, _lane: u8) -> Option<([u64; NUM_STIGMERGIC_GRANULES], f32)> {
        None
    }
}

/// Validates optional Query/Key tensor slice lengths and `LMNK_RESTRICT` pairwise disjointness.
#[inline(always)]
pub(crate) fn validate_optional_qk_tensors(
    query_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    expected_len: usize,
) -> LmnkResult<()> {
    match (query_tensor, key_tensor) {
        (Some(q), Some(k)) => {
            if q.len() < expected_len || k.len() < expected_len {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            q.ensure_disjoint_mut(k)
        }
        (Some(q), None) => {
            if q.len() < expected_len {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            Ok(())
        }
        (None, Some(k)) => {
            if k.len() < expected_len {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            Ok(())
        }
        (None, None) => Ok(()),
    }
}

/// Core Composite Ledger (**Central Trait Registry / Super-Registry**) uniting the branchless
/// foundation, RoPE steering kernels, $\mathcal{O}(1)$ pointer-trail transitions, and FastPathLedger
/// telemetry under static dispatch.
///
/// Blanket extensions (`SubstrateGatewayExt`) observe this composite directly.
pub trait StigmergicSubstrateOps: PointerSubstrate + FastPathLedger {
    /// Broadest cognitive aperture spanned across all gateway, steering, swarm, cohomology, and
    /// mapped-storage capabilities (`0x0000_FFFF`).
    const HUB_APERTURE_MASK: u32 = THRESHOLD_STIGMERGY_WIDE;

    /// Type-level complement identifying signals outside `StigmergicSubstrateOps`'s horizon as unobserved (`!0xFFFF`).
    const HUB_UNOBSERVED_LACUNA_MASK: u32 = !Self::HUB_APERTURE_MASK;

    /// Branchlessly partitions `signal_bits` into `(settled_bits, unobserved_lacuna_bits)` for
    /// [`Self::HUB_APERTURE_MASK`].
    #[inline(always)]
    fn inspect_substrate_horizon_partition(&self, signal_bits: u32) -> (u32, u32) {
        let cut = BitCut32::split(signal_bits, Self::HUB_APERTURE_MASK);
        (cut.inside, cut.outside)
    }

    /// Validates `head_dim`, `num_heads <= MAX_ATTENTION_HEADS`, `phase_step_min_rad >= 0.0`,
    /// `max_rad > 0.0`, and optional Query/Key tensor lengths and `LMNK_RESTRICT` disjointness.
    ///
    /// Returns `(head_dim, num_heads, expected_tensor_len)` on success.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any dimension, radian threshold, or tensor slice
    /// invariant is violated.
    #[inline(always)]
    fn validate_substrate_cascade_params(
        &self,
        head_dim: u32,
        num_heads: u32,
        phase_step_min_rad: f32,
        max_rad: f32,
        query_tensor: Option<&AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(usize, usize, usize)> {
        let valid_dims = (head_dim != 0)
            & ((head_dim & 1) == 0)
            & (num_heads != 0)
            & ((num_heads as usize) <= MAX_ATTENTION_HEADS);
        if !valid_dims
            || phase_step_min_rad < 0.0
            || !branchless_is_finite_f32(phase_step_min_rad)
            || max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let hd = head_dim as usize;
        let nh = num_heads as usize;
        let expected_len = hd.checked_mul(nh).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        validate_optional_qk_tensors(query_tensor, key_tensor, expected_len)?;
        Ok((hd, nh, expected_len))
    }

    /// Resolves `input_ptr` against `ledger_slot`'s $\mathcal{O}(1)$ shortcut table, advances the
    /// pointer transition in `trail_out`, evaluates `bloom_filter` anti-phase repulsion in-place,
    /// and returns `(bypass_hit, effective_ptr, repulsion_density, repulsion_active, post_repulsion_words)`
    /// directly in 8 `u64` SWAR registers.
    ///
    /// # Errors
    /// Propagates any error returned by [`PointerSubstrate::step_pointer`].
    #[inline(always)]
    fn step_substrate_with_bloom_repulsion_to_words(
        &mut self,
        input_ptr: TaggedPtr,
        ledger_slot: &mut LedgerSlot,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<(bool, TaggedPtr, u8, bool, [u64; NUM_STIGMERGIC_GRANULES])> {
        let (bypass_hit, effective_ptr) = self.resolve_or_pass_tagged_ptr(ledger_slot, input_ptr);
        self.step_pointer(effective_ptr.raw(), trail_out)?;

        let (density, repulsion_active, words) = bloom_filter
            .apply_phase_conjugate_repulsion_to_words(
                &mut trail_out.accumulated_phase_delta,
                repulsion_threshold,
                repulsion_scale_u7,
            );
        trail_out.flags |= branchless_select_u32(repulsion_active, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        Ok((bypass_hit, effective_ptr, density, repulsion_active, words))
    }

    /// Filters `raw_words` via 8-lane SWAR `PhaseStepFilter` and steers `query_tensor` / `key_tensor` using
    /// caller-validated `(head_dim, num_heads)`.
    #[inline(always)]
    fn steer_phase_step_words_prevalidated_core(
        &self,
        raw_words: &[u64; NUM_STIGMERGIC_GRANULES],
        head_dim: usize,
        num_heads: usize,
        phase_step_min_rad: f32,
        max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> bool {
        let steered = query_tensor.is_some() | key_tensor.is_some();
        if !steered {
            return false;
        }
        let min_step_q7 = if phase_step_min_rad.to_bits() == 0 {
            0
        } else {
            branchless_rad_to_phase_step_u8(phase_step_min_rad)
        };
        let filtered_buf;
        let swar_words = if min_step_q7 <= 1 {
            raw_words
        } else {
            filtered_buf = swar_q7_phase_step_filter_words_64(raw_words, min_step_q7);
            &filtered_buf
        };
        if !swar_words_nonzero_64(swar_words) {
            return true;
        }

        match (query_tensor, key_tensor) {
            (Some(q), k_opt) => {
                steer_from_swar_words_core_safe(
                    swar_words, 0xFF, 128, head_dim, num_heads, false, max_rad, q, k_opt,
                );
            }
            (None, Some(k)) => {
                steer_from_swar_words_core_safe(
                    swar_words, 0xFF, 128, head_dim, num_heads, false, max_rad, k, None,
                );
            }
            (None, None) => {}
        }
        true
    }

    /// Executes a constant-time per-token stigmergic cascade over [`StigmergicSubstrateOps`]:
    /// 1. Validates tensor dimensions, `LMNK_RESTRICT` disjointness, and radian thresholds.
    /// 2. Resolves `input_ptr` against `ledger_slot`'s $\mathcal{O}(1)$ shortcut table, steps the
    ///    pointer transition in `trail_out`, and applies [`ContrastivePhaseBloomFilter`] repulsion.
    /// 3. Filters `trail_out.accumulated_phase_delta` via 8-lane SWAR `PhaseStepFilter` and steers
    ///    `query_tensor` and/or `key_tensor` with `PhaseClamp` soft-clamping (`max_rad`).
    /// 4. Pushes the combined telemetry metric code (`LEDGER_METRIC_*`) into `ledger_slot`.
    ///
    /// Returns the observed Bloom repulsion bucket density (`0..=255`).
    ///
    /// # Errors
    /// Propagates any error from [`StigmergicSubstrateOps::validate_substrate_cascade_params`],
    /// [`StigmergicSubstrateOps::step_substrate_with_bloom_repulsion_to_words`], or
    /// [`FastPathLedger::telemetry_push`].
    #[inline]
    fn execute_stigmergic_token_cascade(
        &mut self,
        input_ptr: u64,
        ledger_slot: &mut LedgerSlot,
        bloom_filter: &ContrastivePhaseBloomFilter,
        head_dim: u32,
        num_heads: u32,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        phase_step_min_rad: f32,
        max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<u8> {
        let (head_dim, num_heads, _expected_len) = self.validate_substrate_cascade_params(
            head_dim,
            num_heads,
            phase_step_min_rad,
            max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, post_rep_words) = self
            .step_substrate_with_bloom_repulsion_to_words(
                TaggedPtr(input_ptr),
                ledger_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let steered = self.steer_phase_step_words_prevalidated_core(
            &post_rep_words,
            head_dim,
            num_heads,
            phase_step_min_rad,
            max_rad,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, LEDGER_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, LEDGER_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, LEDGER_METRIC_REPULSION_APPLIED, 0);
        self.telemetry_push(ledger_slot, metric)?;
        Ok(density)
    }
}

impl<T> StigmergicSubstrateOps for T where T: PointerSubstrate + FastPathLedger + ?Sized {}

impl PointerTrail {
    /// Branchlessly evaluates whether a pointer trail qualifies for bottom-up macro
    /// crystallization (`stigmergic_density >= density_thresh && r_bar_t >= coherence_thresh && !repulsion_active`)
    /// and stamps or clears [`TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`] on `self.flags` in $\mathcal{O}(1)$.
    #[inline(always)]
    pub fn evaluate_crystallization_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
    ) -> bool {
        let density_ok = self.should_crystallize_branchless(density_crystallization_thresh)
            & branchless_is_finite_f32(density_crystallization_thresh);
        let coherence_ok = (r_bar_t >= coherence_crystallization_thresh)
            & (coherence_crystallization_thresh >= 0.0)
            & branchless_is_finite_f32(coherence_crystallization_thresh)
            & branchless_is_finite_f32(r_bar_t);
        let not_repulsed = (self.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_candidate = density_ok & coherence_ok & not_repulsed;
        self.flags = (self.flags & !TRAIL_FLAG_CRYSTALLIZED_CANDIDATE)
            | branchless_select_u32(is_candidate, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, 0);
        is_candidate
    }
}
