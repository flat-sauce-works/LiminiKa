// name: src/core/src/stigmergy/pheromone.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared pheromone density laws (`PheromoneDensityOps`), 8-granule SWAR stigmergic
//! reconstitution (`StigmergicReconstitutionOps`), and scale-aware swarm resolution
//! (`StigmergicLayeredSwarmExt`).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

#[cfg(test)]
use crate::foundation::load_q7_words_64;
use crate::foundation::{
    ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_STEER, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_REPULSION_ESCAPE,
    ATTRACTOR_SLOT_ID_MASK, Pod, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, TaggedPtr,
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_min_f32, branchless_relu_f32, is_word_array_uniform_64, swar_q7_abs_u8_x8,
    swar_q7_add_sat_x8, swar_q7_avg_round_x8, swar_q7_circular_add_x8, swar_q7_circular_diff_x8,
    swar_q7_scale_le128_with_bias_x8, swar_u8_mean_round_x8, swar_words_nonzero_64,
};
#[cfg(test)]
use crate::liminal::{MAX_ATTENTION_HEADS, PhaseQ7};
use crate::liminal::{
    NUM_STIGMERGIC_GRANULES, branchless_select_f32, branchless_select_u8, branchless_select_u32,
    branchless_select_u64,
};
use crate::stigmergy::circuit::StigmergicScaleGranularity;
use crate::stigmergy::ledger::ContrastivePhaseBloomFilter;
use crate::stigmergy::trail::{PointerTrail, trail_phase_words_mut, trail_phase_words_ref};
use crate::stigmergy::{
    ATTRACTOR_SLOT_REPULSION, DEFAULT_STIGMERGIC_QUORUM_THRESHOLD, DEFAULT_STIGMERGIC_TOLERANCE_Q7,
    MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_PHASE_STEER_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE,
};
// ===================================================================
// 5. Shared Pheromone Density Laws (`PheromoneDensityOps`) & Pointer Trail
// ===================================================================

/// Shared branchless pheromone decay and coherence-weighted reinforcement law implemented by both
/// hot pointer trails ([`PointerTrail`]) and RAM experience cache slots (`RamPhaseKvEntry`).
pub trait PheromoneDensityOps {
    /// Returns the raw non-negative pheromone density scalar for this record.
    fn pheromone_density(&self) -> f32;

    /// Updates the stored pheromone density scalar for this record.
    fn set_pheromone_density(&mut self, density: f32);

    /// Applies subnormal-safe temporal evaporative decay ($\rho \gets \min(\rho \cdot
    /// \lambda_{\mathrm{retain}}, \rho_{\max})$).
    #[inline(always)]
    fn evaporate_pheromone(&mut self, retain_factor: f32) -> f32 {
        let clean_decay =
            branchless_select_f32(retain_factor > 1.0, 1.0, branchless_relu_f32(retain_factor));
        let decayed = branchless_relu_f32(self.pheromone_density()) * clean_decay;
        let next = branchless_min_f32(
            branchless_flush_subnormal_f32(decayed),
            MAX_STIGMERGIC_DENSITY,
        );
        self.set_pheromone_density(next);
        next
    }

    /// Executes a single closed-loop pheromone update step: temporal evaporation, followed by
    /// coherence-weighted reinforcement (or immediate halving on contradiction repulsion).
    #[inline(always)]
    fn step_pheromone(
        &mut self,
        retain_factor: f32,
        reinforcement_gain: f32,
        phase_coherence: f32,
        contradiction_repulsed: bool,
    ) -> f32 {
        let cur_rho = self.evaporate_pheromone(retain_factor);
        let pos_coh = branchless_clamp_f32(branchless_relu_f32(phase_coherence), 0.0, 1.0);
        let deposit = branchless_relu_f32(reinforcement_gain) * (1.0 + pos_coh);
        let reinforced_rho = branchless_min_f32(cur_rho + deposit, MAX_STIGMERGIC_DENSITY);
        let damped_rho = branchless_flush_subnormal_f32(cur_rho * 0.5);
        let next_rho = branchless_select_f32(contradiction_repulsed, damped_rho, reinforced_rho);
        self.set_pheromone_density(next_rho);
        next_rho
    }
}

impl PheromoneDensityOps for PointerTrail {
    #[inline(always)]
    fn pheromone_density(&self) -> f32 {
        self.stigmergic_density
    }

    #[inline(always)]
    fn set_pheromone_density(&mut self, density: f32) {
        self.stigmergic_density = density;
    }
}

/// Hoisted-invariant 8-head SWAR Micro-Swarm cell resolver using precomputed `gain_s <= 128`,
/// `gain_bias = (128 - gain_s) * 0x0101_0101_0101_0101`, `min_step_thresh_x8`, and `min_step_valid_mask`.
/// Eliminates per-cell 64-bit bias/threshold multiplications and bypasses `swar_q7_add_sat_x8`
/// when `step_gain_u7 <= 128`.
#[inline(always)]
#[must_use]
// Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
#[allow(clippy::too_many_arguments)]
pub(crate) const fn swar_subswarm_resolve_with_bias_x8(
    current_word: u64,
    target_word: u64,
    gain_s: u64,
    gain_bias: u64,
    gain_gt_128: bool,
    min_step_thresh_x8: u64,
    min_step_valid_mask: u64,
    tolerance_q7: u8,
) -> (u64, u8, bool) {
    if current_word == target_word {
        return (current_word, 0, true);
    }
    let raw_diff = swar_q7_circular_diff_x8(target_word, current_word);
    let filtered_diff = if min_step_thresh_x8 <= SWAR_SPLAT_BYTE_64 {
        raw_diff & min_step_valid_mask
    } else {
        let abs_lanes = swar_q7_abs_u8_x8(raw_diff);
        let ge_bit7 = (((abs_lanes | SWAR_SIGN_MASK_64).wrapping_sub(min_step_thresh_x8))
            | (abs_lanes & SWAR_SIGN_MASK_64))
            & SWAR_SIGN_MASK_64;
        let keep_mask = (ge_bit7 >> 7).wrapping_mul(0xFF);
        raw_diff & keep_mask & min_step_valid_mask
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
pub(crate) const fn precompute_swarm_step_params(
    step_gain_u7: u8,
    min_step_q7: u8,
) -> (u64, u64, bool, u64, u64) {
    let gain_gt_128 = step_gain_u7 > 128;
    let gain_s =
        branchless_select_u8(gain_gt_128, step_gain_u7.wrapping_sub(128), step_gain_u7) as u64;
    let gain_bias = (128u64.wrapping_sub(gain_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let min_step_thresh_x8 = ((min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let min_step_valid_mask = ((min_step_q7 <= 127) as u64).wrapping_neg();
    (
        gain_s,
        gain_bias,
        gain_gt_128,
        min_step_thresh_x8,
        min_step_valid_mask,
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[must_use]
// Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
#[allow(clippy::too_many_arguments)]
unsafe fn avx2_subswarm_resolve_words_64(
    cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
    cur_is_uniform: bool,
    tgt_base_w: &[u64; NUM_STIGMERGIC_GRANULES],
    tgt_is_uniform: bool,
    use_explicit_tgt: bool,
    ptr_phase_splat: u64,
    active_cell_mask: u8,
    step_gain_u7: u8,
    min_step_q7: u8,
    tolerance_q7: u8,
) -> ([u64; NUM_STIGMERGIC_GRANULES], u8, u32) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`. Resolves all 8 Micro-Swarm cells (64 heads)
    // in parallel inside two 256-bit YMM registers (`_mm256_mulhi_epi16(d << 8, s << 1)` and `_mm256_sad_epu8`),
    // preserving 100% bit-for-bit identity with `swar_subswarm_resolve_with_bias_x8`.
    unsafe {
        use core::arch::x86_64::_mm256_testz_si256;
        use core::arch::x86_64::{
            __m256i, _mm_add_epi64, _mm_cvtsi128_si64, _mm_unpackhi_epi64, _mm256_abs_epi8,
            _mm256_add_epi8, _mm256_add_epi64, _mm256_adds_epi8, _mm256_and_si256,
            _mm256_castsi256_pd, _mm256_castsi256_si128, _mm256_cmpeq_epi8, _mm256_cmpgt_epi64,
            _mm256_extracti128_si256, _mm256_max_epu8, _mm256_movemask_pd, _mm256_mulhi_epi16,
            _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi8, _mm256_set1_epi16,
            _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_slli_epi16, _mm256_srli_epi64,
            _mm256_sub_epi8,
        };
        let _ = core::mem::size_of::<__m256i>();
        let (vc0, vc1) =
            crate::foundation::phase_ops::avx2_words64_to_vec64_with_uniform(cur_w, cur_is_uniform);
        let (vt0, vt1) = if use_explicit_tgt {
            crate::foundation::phase_ops::avx2_words64_to_vec64_with_uniform(
                tgt_base_w,
                tgt_is_uniform,
            )
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

        let (vf0, vf1) = if min_step_q7 <= 1 {
            (vd0, vd1)
        } else if min_step_q7 > 127 {
            (zero, zero)
        } else {
            let v_thresh = _mm256_set1_epi8(min_step_q7 as i8);
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
        let mut next_words = crate::foundation::phase_ops::avx2_vec64_to_words64(vn0, vn1);

        if active_cell_mask == 0xFF {
            let r_sum4 = _mm256_add_epi64(res0_vec, res1_vec);
            let r_sum2 = _mm_add_epi64(
                _mm256_castsi256_si128(r_sum4),
                _mm256_extracti128_si256(r_sum4, 1),
            );
            let residual_sum =
                _mm_cvtsi128_si64(_mm_add_epi64(r_sum2, _mm_unpackhi_epi64(r_sum2, r_sum2))) as u32;
            return (next_words, raw_quorum, residual_sum);
        }

        let res_arr = crate::foundation::phase_ops::avx2_vec64_to_words64(res0_vec, res1_vec);
        let mut residual_sum = 0u32;
        let mut cell = 0usize;
        while cell < NUM_STIGMERGIC_GRANULES {
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
// Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
#[allow(clippy::too_many_arguments)]
pub(super) fn swar_subswarm_resolve_words_64(
    cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
    cur_is_uniform: bool,
    tgt_base_w: &[u64; NUM_STIGMERGIC_GRANULES],
    tgt_is_uniform: bool,
    use_explicit_tgt: bool,
    ptr_phase_splat: u64,
    active_cell_mask: u8,
    step_gain_u7: u8,
    min_step_q7: u8,
    tolerance_q7: u8,
) -> ([u64; NUM_STIGMERGIC_GRANULES], u8, u32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `cfg(target_feature = "avx2")`.
        unsafe {
            avx2_subswarm_resolve_words_64(
                cur_w,
                cur_is_uniform,
                tgt_base_w,
                tgt_is_uniform,
                use_explicit_tgt,
                ptr_phase_splat,
                active_cell_mask,
                step_gain_u7,
                min_step_q7,
                tolerance_q7,
            )
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    let _ = (cur_is_uniform, tgt_is_uniform);
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let (gain_s, gain_bias, gain_gt_128, min_step_thresh_x8, min_step_valid_mask) =
            precompute_swarm_step_params(step_gain_u7, min_step_q7);
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
        while cell < NUM_STIGMERGIC_GRANULES {
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
                    min_step_thresh_x8,
                    min_step_valid_mask,
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
// 6. Constant-Time Closed-Loop Micro-Swarm Resolution (`StigmergicReconstitutionOps`)
// ===================================================================

/// 16-byte register-aligned summary returned when the 8-granule stigmergic field attempts to
/// reconstitute a Stigmergic Substrate command on the Hot Path and feeds back into pointer trails.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StigmergicReconstitutionReport {
    /// Feedback-evolved 64-bit tagged pointer ready for the Stigmergic Substrate's next instruction step.
    pub evolved_ptr: TaggedPtr,
    /// Signed stigmergic density feedback ($\Delta\rho_{\mathrm{stigmergy}}$) applied to the trail.
    pub stigmergic_delta_density: f32,
    /// 8-bit Stigmergic Quorum bitmask (`bit c == 1` iff 8-head granule $c \in 0..8$ converged).
    pub quorum_mask: u8,
    /// Number of Stigmergic granules (`0..=8`) that resolved their sub-task within `tolerance_q7`.
    pub resolved_cell_count: u8,
    /// Mean circular Q7 phase residual (`0..=128`) across all 64 heads after the reconstitution step.
    pub mean_residual_q7: u8,
    /// `1` if stigmergic reconstitution achieved quorum on the Hot Path without Bloom repulsion, `0` if escalated.
    pub hot_path_resolved: u8,
}

impl Default for StigmergicReconstitutionReport {
    #[inline(always)]
    fn default() -> Self {
        Self {
            evolved_ptr: TaggedPtr::NULL,
            stigmergic_delta_density: 0.0,
            quorum_mask: 0,
            resolved_cell_count: 0,
            mean_residual_q7: 0,
            hot_path_resolved: 0,
        }
    }
}

// SAFETY: `StigmergicReconstitutionReport` is a 16-byte `#[repr(C, align(16))]` structure with zero padding.
unsafe impl Pod for StigmergicReconstitutionReport {}

/// Evaluates a single 8-head SWAR Stigmergic Granule step on the circle group $S^1 \cong SO(2)$
/// in **strictly constant time (`0` branches)** across all inputs.
#[inline(always)]
#[must_use]
pub const fn swar_subswarm_resolve_x8(
    current_word: u64,
    target_word: u64,
    step_gain_u7: u8,
    min_step_q7: u8,
    tolerance_q7: u8,
) -> (u64, u8, bool) {
    let (gain_s, gain_bias, gain_gt_128, min_step_thresh_x8, min_step_valid_mask) =
        precompute_swarm_step_params(step_gain_u7, min_step_q7);
    swar_subswarm_resolve_with_bias_x8(
        current_word,
        target_word,
        gain_s,
        gain_bias,
        gain_gt_128,
        min_step_thresh_x8,
        min_step_valid_mask,
        tolerance_q7,
    )
}

/// Parameters for [`PointerTrail::try_resolve_action_in_swarm_from_words`].
#[derive(Debug, Clone, Copy)]
pub struct SwarmResolveParams<'a> {
    /// Incoming tagged pointer describing target slot and action mode.
    pub action_ptr: TaggedPtr,
    /// Bitmask of active 8-head micro-swarm cells (`0xFF` for all 8 cells).
    pub active_cell_mask: u8,
    /// Per-cell phase step gain (`0..=128`).
    pub step_gain_u7: u8,
    /// Minimum Q7 phase step filter threshold.
    pub min_step_q7: u8,
    /// Maximum residual Q7 tolerance for cell convergence.
    pub tolerance_q7: u8,
    /// Minimum number of converged cells required for hot-path quorum (`1..=8`).
    pub quorum_threshold: u8,
    /// Current normalized entropy observation $\tilde{H}$.
    pub entropy_h_tilde: f32,
    /// Low-entropy inertial skip threshold.
    pub low_entropy_skip_thresh: f32,
    /// Optional contrastive phase Bloom filter for repulsion gating.
    pub bloom_filter: Option<&'a ContrastivePhaseBloomFilter>,
    /// Bloom repulsion density threshold (`0..=255`).
    pub repulsion_threshold: u8,
    /// Repulsion conjugate blend scale (`0..=128`).
    pub repulsion_scale_u7: u8,
}

/// Approximation-first trait framing the 3-stage stigmergic compute pipeline:
/// 1. **Fragmentation (Granulate)**: decompose a 64-head phase trajectory into 8 myopic `u64` SWAR granules.
impl PointerTrail {
    /// Subdivides the Stigmergic Substrate's target 64-head phase command into **8 parallel 8-head SWAR
    /// Micro-Swarm cells** gated by `active_cell_mask` in strictly constant time (`0` branches),
    /// returning both `(StigmergicReconstitutionReport, resolved_swar_words)` so downstream RoPE/`PhaseClamp` kernels
    /// can steer directly from registers without re-loading `accumulated_phase_delta`.
    #[cfg(test)]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_resolve_action_in_swarm_to_words(
        &mut self,
        action_ptr: TaggedPtr,
        target_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        active_cell_mask: u8,
        step_gain_u7: u8,
        min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (
        StigmergicReconstitutionReport,
        [u64; NUM_STIGMERGIC_GRANULES],
    ) {
        let tgt_w = load_q7_words_64(target_phases);
        self.try_resolve_action_in_swarm_from_words(
            &tgt_w,
            SwarmResolveParams {
                action_ptr,
                active_cell_mask,
                step_gain_u7,
                min_step_q7,
                tolerance_q7,
                quorum_threshold,
                entropy_h_tilde,
                low_entropy_skip_thresh,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
            },
        )
    }

    /// Subdivides the Stigmergic Substrate's target 64-head phase command into **8 parallel 8-head SWAR
    /// Micro-Swarm cells** gated by `active_cell_mask` in strictly constant time (`0` branches).
    #[cfg(test)]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_resolve_action_in_swarm_masked(
        &mut self,
        action_ptr: TaggedPtr,
        target_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        active_cell_mask: u8,
        step_gain_u7: u8,
        min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> StigmergicReconstitutionReport {
        self.try_resolve_action_in_swarm_to_words(
            action_ptr,
            target_phases,
            active_cell_mask,
            step_gain_u7,
            min_step_q7,
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

    /// Fragments the trail across 8 myopic granules, drifts each along `target_words` gated by
    /// `active_granule_mask`, and reconstitutes the resulting state transition into
    /// `(StigmergicReconstitutionReport, reconstituted_words)` in branchless constant time.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reconstitute_from_stigmergy_words(
        &mut self,
        action_ptr: TaggedPtr,
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
        active_granule_mask: u8,
        step_gain_u7: u8,
        min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (
        StigmergicReconstitutionReport,
        [u64; NUM_STIGMERGIC_GRANULES],
    ) {
        self.try_resolve_action_in_swarm_from_words(
            target_words,
            SwarmResolveParams {
                action_ptr,
                active_cell_mask: active_granule_mask,
                step_gain_u7,
                min_step_q7,
                tolerance_q7,
                quorum_threshold,
                entropy_h_tilde,
                low_entropy_skip_thresh,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
            },
        )
    }

    #[inline]
    // Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_resolve_action_in_swarm_with_cur_words(
        &mut self,
        action_ptr: TaggedPtr,
        cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
        has_active_anchors: bool,
        active_cell_mask: u8,
        step_gain_u7: u8,
        min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (
        StigmergicReconstitutionReport,
        [u64; NUM_STIGMERGIC_GRANULES],
    ) {
        let ptr_phase = action_ptr.unpack_phase();
        let ptr_phase_splat = ((ptr_phase as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let cur_w = *cur_w;
        let tgt_base_w = *target_words;
        let tgt_has_explicit = swar_words_nonzero_64(&tgt_base_w);
        let use_explicit_tgt = tgt_has_explicit | (ptr_phase_splat == 0);
        let mut next_words = cur_w;
        let quorum_mask: u8;
        let residual_sum: u32;
        if (use_explicit_tgt
            && (cur_w[0] == tgt_base_w[0])
            && ((((cur_w[1] ^ tgt_base_w[1]) | (cur_w[2] ^ tgt_base_w[2]))
                | ((cur_w[3] ^ tgt_base_w[3]) | (cur_w[4] ^ tgt_base_w[4]))
                | ((cur_w[5] ^ tgt_base_w[5]) | (cur_w[6] ^ tgt_base_w[6]))
                | (cur_w[7] ^ tgt_base_w[7]))
                == 0))
            || active_cell_mask == 0
        {
            quorum_mask = 0xFF;
            residual_sum = 0;
        } else {
            let cur_is_uniform = is_word_array_uniform_64(&cur_w);
            let tgt_is_uniform = (!use_explicit_tgt) || is_word_array_uniform_64(&tgt_base_w);
            if cur_is_uniform & tgt_is_uniform {
                let cw0 = cur_w[0];
                let tw0 = branchless_select_u64(use_explicit_tgt, tgt_base_w[0], ptr_phase_splat);
                if cw0 == tw0 {
                    quorum_mask = 0xFF;
                    residual_sum = 0;
                } else {
                    let (gain_s, gain_bias, gain_gt_128, min_step_thresh_x8, min_step_valid_mask) =
                        precompute_swarm_step_params(step_gain_u7, min_step_q7);
                    let (stepped_w0, cell_res, cell_ok) = swar_subswarm_resolve_with_bias_x8(
                        cw0,
                        tw0,
                        gain_s,
                        gain_bias,
                        gain_gt_128,
                        min_step_thresh_x8,
                        min_step_valid_mask,
                        tolerance_q7,
                    );
                    if active_cell_mask == 0xFF {
                        next_words = [stepped_w0; NUM_STIGMERGIC_GRANULES];
                        quorum_mask = branchless_select_u8(cell_ok, 0xFF, 0x00);
                        residual_sum = u32::from(cell_res) << 3;
                    } else {
                        let mut c = 0usize;
                        while c < NUM_STIGMERGIC_GRANULES {
                            next_words[c] = branchless_select_u64(
                                ((active_cell_mask >> c) & 1) != 0,
                                stepped_w0,
                                cw0,
                            );
                            c += 1;
                        }
                        quorum_mask = branchless_select_u8(cell_ok, 0xFF, !active_cell_mask);
                        residual_sum = u32::from(cell_res) * active_cell_mask.count_ones();
                    }
                }
            } else {
                let (nw, qm, rsum) = swar_subswarm_resolve_words_64(
                    &cur_w,
                    cur_is_uniform,
                    &tgt_base_w,
                    tgt_is_uniform,
                    use_explicit_tgt,
                    ptr_phase_splat,
                    active_cell_mask,
                    step_gain_u7,
                    min_step_q7,
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

        if !(quorum_mask == 0xFF
            && residual_sum == 0
            && !repulsion_triggered
            && cur_w[0] == next_words[0])
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
        let evolved_ptr = TaggedPtr(branchless_select_u64(
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
        let phase_steer_on = (next_mode & ACTION_MODE_PHASE_STEER) != 0;
        let skip_on = (next_mode & ACTION_MODE_LAYER_SKIP) != 0;
        self.flags = (self.flags
            & !(TRAIL_FLAG_PHASE_STEER_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | branchless_select_u32(phase_steer_on, TRAIL_FLAG_PHASE_STEER_ACTIVE, 0)
            | branchless_select_u32(skip_on, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);

        (
            StigmergicReconstitutionReport {
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
}

impl PointerTrail {
    /// Extracts the 8-head Q7 phase word (`u64`) belonging to Micro-Swarm cell `cell_idx` (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub fn extract_subswarm_word(&self, cell_idx: usize) -> u64 {
        let safe_cell = cell_idx & (NUM_STIGMERGIC_GRANULES - 1);
        trail_phase_words_ref(self)[safe_cell]
    }

    /// Writes an 8-head Q7 phase word (`u64`) into Micro-Swarm cell `cell_idx` (`0..=7`).
    #[inline(always)]
    pub fn write_subswarm_word(&mut self, cell_idx: usize, word: u64) {
        let safe_cell = cell_idx & (NUM_STIGMERGIC_GRANULES - 1);
        trail_phase_words_mut(self)[safe_cell] = word;
    }

    /// Resolves a Stigmergic Substrate command across all 8 SWAR Micro-Swarm cells directly from
    /// Subdivides the Stigmergic Substrate's target 64-head phase command directly from
    /// pre-loaded target SWAR words (`target_words`), avoiding any 64-byte stack reload when
    /// chained after `ManifoldGatewaySignal::synthesize_gateway_frame_with_words`.
    #[inline(always)]
    pub fn try_resolve_action_in_swarm_from_words(
        &mut self,
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
        params: SwarmResolveParams<'_>,
    ) -> (
        StigmergicReconstitutionReport,
        [u64; NUM_STIGMERGIC_GRANULES],
    ) {
        let cur_w = *trail_phase_words_ref(self);
        self.try_resolve_action_in_swarm_with_cur_words(
            params.action_ptr,
            &cur_w,
            target_words,
            false,
            params.active_cell_mask,
            params.step_gain_u7,
            params.min_step_q7,
            params.tolerance_q7,
            params.quorum_threshold,
            params.entropy_h_tilde,
            params.low_entropy_skip_thresh,
            params.bloom_filter,
            params.repulsion_threshold,
            params.repulsion_scale_u7,
        )
    }

    /// Fuses resolved 8-head Micro-Swarm cells from a peer pointer trail (`peer_trail`) into `self`
    /// in 8 branchless SWAR register operations.
    #[inline]
    pub fn fuse_with_peer_swarm_trail(
        &mut self,
        peer_trail: &PointerTrail,
        quorum_threshold: u8,
    ) -> u8 {
        let self_quorum = self.swarm_quorum_mask();
        let peer_quorum = peer_trail.swarm_quorum_mask();
        let combined_quorum = self_quorum | peer_quorum;

        let peer_w = trail_phase_words_ref(peer_trail);
        let self_w = trail_phase_words_mut(self);
        let mut cell = 0usize;
        while cell < NUM_STIGMERGIC_GRANULES {
            let s_ok = ((self_quorum >> cell) & 1) != 0;
            let p_ok = ((peer_quorum >> cell) & 1) != 0;
            let s_w = self_w[cell];
            let p_w = peer_w[cell];
            let avg_w = swar_q7_avg_round_x8(s_w, p_w);
            let adopt_w = branchless_select_u64(p_ok, p_w, s_w);
            self_w[cell] = branchless_select_u64(
                s_ok & p_ok,
                avg_w,
                branchless_select_u64(s_ok, s_w, adopt_w),
            );
            cell += 1;
        }

        let newly_resolved = (combined_quorum & !self_quorum).count_ones();
        let boosted_rho =
            branchless_flush_subnormal_f32(branchless_relu_f32(self.stigmergic_density))
                + 0.05 * (newly_resolved as f32);
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

    /// Resolves `target_words` inside the 8-cell SWAR Micro-Swarm using parameters derived
    /// structurally from `(granularity, layer_index, total_layers)` and optional Bloom guards.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn resolve_swarm_with_scale_layer(
        &mut self,
        action_ptr: TaggedPtr,
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
        granularity: StigmergicScaleGranularity,
        layer_index: usize,
        total_layers: usize,
        active_cell_mask: u8,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
    ) -> (
        StigmergicReconstitutionReport,
        [u64; NUM_STIGMERGIC_GRANULES],
    ) {
        let gain_q7 = granularity.layer_peel_gain_q7(layer_index, total_layers);
        let tol_q7 = granularity.swarm_tolerance_q7(DEFAULT_STIGMERGIC_TOLERANCE_Q7);
        let quorum = granularity.swarm_quorum_requirement(DEFAULT_STIGMERGIC_QUORUM_THRESHOLD);
        let mask = if active_cell_mask == 0 {
            0xFF
        } else {
            active_cell_mask
        };
        self.try_resolve_action_in_swarm_from_words(
            target_words,
            SwarmResolveParams {
                action_ptr,
                active_cell_mask: mask,
                step_gain_u7: gain_q7,
                min_step_q7: 2,
                tolerance_q7: tol_q7,
                quorum_threshold: quorum,
                entropy_h_tilde: 0.40,
                low_entropy_skip_thresh: 0.15,
                bloom_filter,
                repulsion_threshold: 3,
                repulsion_scale_u7: 96,
            },
        )
    }
}
