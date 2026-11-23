// name: src/core/src/stigmergy/trail/mod.rs
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
use crate::foundation::Q7_FULL_COS_REF;
use crate::foundation::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_popcount_rsqrt_u32, branchless_relu_f32, branchless_rsqrt_f32,
    coherence_from_word_arrays_with_both_uniform_flags_64, fast_ln_1p_bounded_f32,
    is_word_array_uniform_64, Pod, RawPhaseProjectable,
};
pub use crate::foundation::{PagedBitmask, TaggedPtr};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, PhaseQ7, MAX_ATTENTION_HEADS,
    NUM_STIGMERGIC_GRANULES,
};
pub use crate::stigmergy::ledger::LedgerSlot;
use crate::stigmergy::TRAIL_FLAG_REPULSION_ACTIVE;

mod ops;
mod resonance;
mod traits;

pub(crate) use ops::select_rewire_bridge_and_modulate_with_preloaded_words_64;
pub use resonance::{
    bridge_and_filter_liminal_trail_chain, detect_and_record_obstruction_repulsion,
    eval_stigmergic_block_degrees, evaluate_trail_chain_cohomology_and_joint_log_prob,
    modulate_attention_scores_by_bayesian_trails_64, resolve_dyadic_qdpt_and_cascade_ptr,
};
pub(crate) use traits::validate_optional_qk_tensors;
pub use traits::{PointerSubstrate, StigmergicSubstrateOps};

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
pub(super) unsafe fn eval_candidate_trail_avx2_with_macro_fast(
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
