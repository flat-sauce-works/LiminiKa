// name: src/core/src/traits/telemetry/stigmergy.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 8-granule x 8-head SWAR stigmergic reconstitution and lateral phase consensus (`StigmergicReconstitutionOps`).

use super::trail::*;
use super::*;

#[inline(always)]
#[must_use]
pub(super) const fn swar_u8_pair_sum_u16(w: u64) -> u64 {
    (w & SWAR_EVEN_BYTES_64).wrapping_add((w >> 8) & SWAR_EVEN_BYTES_64)
}

#[inline(always)]
#[must_use]
pub(super) const fn swar_total_mass_from_words(w: &[u64; 8]) -> u32 {
    let s01 = swar_u8_pair_sum_u16(w[0]).wrapping_add(swar_u8_pair_sum_u16(w[1]));
    let s23 = swar_u8_pair_sum_u16(w[2]).wrapping_add(swar_u8_pair_sum_u16(w[3]));
    let s45 = swar_u8_pair_sum_u16(w[4]).wrapping_add(swar_u8_pair_sum_u16(w[5]));
    let s67 = swar_u8_pair_sum_u16(w[6]).wrapping_add(swar_u8_pair_sum_u16(w[7]));
    let s_all = s01.wrapping_add(s23).wrapping_add(s45.wrapping_add(s67));
    (s_all.wrapping_mul(SWAR_SUM_U16_64) >> 48) as u32
}

#[inline(always)]
#[must_use]
pub(super) const fn swar_active_buckets_from_words(w: &[u64; 8]) -> u32 {
    let nz01 = swar_u8_nonzero_msb_mask(w[0]) | (swar_u8_nonzero_msb_mask(w[1]) >> 1);
    let nz23 = (swar_u8_nonzero_msb_mask(w[2]) >> 2) | (swar_u8_nonzero_msb_mask(w[3]) >> 3);
    let nz45 = (swar_u8_nonzero_msb_mask(w[4]) >> 4) | (swar_u8_nonzero_msb_mask(w[5]) >> 5);
    let nz67 = (swar_u8_nonzero_msb_mask(w[6]) >> 6) | (swar_u8_nonzero_msb_mask(w[7]) >> 7);
    ((nz01 | nz23) | (nz45 | nz67)).count_ones()
}

/// Hoisted-invariant 8-head SWAR Micro-Swarm cell resolver using precomputed `gain_s <= 128`,
/// `gain_bias = (128 - gain_s) * 0x0101_0101_0101_0101`, `qdps_thresh_x8`, and `qdps_valid_mask`.
/// Eliminates per-cell 64-bit bias/threshold multiplications and bypasses `swar_q7_add_sat_x8`
/// when `step_gain_u7 <= 128`.
#[inline(always)]
#[must_use]
pub(super) const fn swar_subswarm_resolve_with_bias_x8(
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
pub(super) const fn precompute_swarm_step_params(
    step_gain_u7: u8,
    qdps_min_step_q7: u8,
) -> (u64, u64, bool, u64, u64) {
    let gain_gt_128 = step_gain_u7 > 128;
    let gain_s =
        branchless_select_u8(gain_gt_128, step_gain_u7.wrapping_sub(128), step_gain_u7) as u64;
    let gain_bias = (128u64.wrapping_sub(gain_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let qdps_thresh_x8 = ((qdps_min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let qdps_valid_mask = ((qdps_min_step_q7 <= 127) as u64).wrapping_neg();
    (
        gain_s,
        gain_bias,
        gain_gt_128,
        qdps_thresh_x8,
        qdps_valid_mask,
    )
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
#[must_use]
unsafe fn avx2_subswarm_resolve_words_64(
    cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
    cur_is_uniform: bool,
    tgt_base_w: &[u64; NUM_STIGMERGIC_GRANULES],
    tgt_is_uniform: bool,
    use_explicit_tgt: bool,
    ptr_phase_splat: u64,
    active_cell_mask: u8,
    step_gain_u7: u8,
    qdps_min_step_q7: u8,
    tolerance_q7: u8,
) -> ([u64; NUM_STIGMERGIC_GRANULES], u8, u32) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`. Resolves all 8 Micro-Swarm cells (64 heads)
    // in parallel inside two 256-bit YMM registers (`_mm256_mulhi_epi16(d << 8, s << 1)` and `_mm256_sad_epu8`),
    // preserving 100% bit-for-bit identity with `swar_subswarm_resolve_with_bias_x8`.
    unsafe {
        use core::arch::x86_64::_mm256_testz_si256;
        use core::arch::x86_64::{
            __m256i, _mm256_abs_epi8, _mm256_add_epi64, _mm256_add_epi8, _mm256_adds_epi8,
            _mm256_and_si256, _mm256_castsi256_pd, _mm256_castsi256_si128, _mm256_cmpeq_epi8,
            _mm256_cmpgt_epi64, _mm256_extracti128_si256, _mm256_max_epu8, _mm256_movemask_pd,
            _mm256_mulhi_epi16, _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi16,
            _mm256_set1_epi64x, _mm256_set1_epi8, _mm256_setzero_si256, _mm256_slli_epi16,
            _mm256_srli_epi64, _mm256_sub_epi8, _mm_add_epi64, _mm_cvtsi128_si64,
            _mm_unpackhi_epi64,
        };
        let _ = core::mem::size_of::<__m256i>();
        let (vc0, vc1) =
            crate::traits::branchless::avx2_words64_to_vec64_with_uniform(cur_w, cur_is_uniform);
        let (vt0, vt1) = if use_explicit_tgt {
            crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
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
            let residual_sum =
                _mm_cvtsi128_si64(_mm_add_epi64(r_sum2, _mm_unpackhi_epi64(r_sum2, r_sum2))) as u32;
            return (next_words, raw_quorum, residual_sum);
        }

        let res_arr = crate::traits::branchless::avx2_vec64_to_words64(res0_vec, res1_vec);
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
pub(super) fn swar_subswarm_resolve_words_64(
    cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
    cur_is_uniform: bool,
    tgt_base_w: &[u64; NUM_STIGMERGIC_GRANULES],
    tgt_is_uniform: bool,
    use_explicit_tgt: bool,
    ptr_phase_splat: u64,
    active_cell_mask: u8,
    step_gain_u7: u8,
    qdps_min_step_q7: u8,
    tolerance_q7: u8,
) -> ([u64; NUM_STIGMERGIC_GRANULES], u8, u32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Guarded by `cfg(target_feature = "avx2")`.
        return unsafe {
            avx2_subswarm_resolve_words_64(
                cur_w,
                cur_is_uniform,
                tgt_base_w,
                tgt_is_uniform,
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
    let _ = (cur_is_uniform, tgt_is_uniform);
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
// 6. Constant-Time Closed-Loop Micro-Swarm Resolution (`StigmergicReconstitutionOps`)
// ===================================================================

/// 16-byte register-aligned summary returned when the 8-granule stigmergic field attempts to
/// reconstitute an Action Hub command on the Hot Path and feeds back into pointer trails.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StigmergicReconstitutionReport {
    /// Feedback-evolved 64-bit tagged pointer ready for the Action Hub's next instruction step.
    pub evolved_ptr: gcso_tagged_ptr_t,
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
            evolved_ptr: gcso_tagged_ptr_t::NULL,
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

/// Approximation-first trait framing the 3-stage stigmergic compute pipeline:
/// 1. **Fragmentation (Granulate)**: decompose a 64-head phase trajectory into 8 myopic `u64` SWAR granules.
/// 2. **Stigmergic Drift**: steer each granule along local terrain pheromone trails without global planning.
/// 3. **Reconstitution**: reassemble converged granules via quorum consensus into an approximate state transition.
pub trait StigmergicReconstitutionOps: StigmergicTrailOps {
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
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]);

    /// Subdivides the Action Hub's target 64-head phase command into **8 parallel 8-head SWAR
    /// Micro-Swarm cells** gated by `active_cell_mask` in strictly constant time (`0` branches),
    /// returning both `(StigmergicReconstitutionReport, resolved_swar_words)` so downstream RoPE/RIPA kernels
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
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    ) -> StigmergicReconstitutionReport {
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
    ) -> StigmergicReconstitutionReport {
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
    /// `(StigmergicReconstitutionReport, resolved_swar_words)`.
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
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]);

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
    ) -> StigmergicReconstitutionReport {
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
    /// partial quorum entirely in registers, and returns `(StigmergicReconstitutionReport, resolved_swar_words)`.
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
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]);

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
    ) -> StigmergicReconstitutionReport {
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

    /// Extracts the myopic 8-head `u64` SWAR word (`granule_idx` in `0..8`) during fragmentation.
    #[inline(always)]
    #[must_use]
    fn granulate_phase_words(&self, granule_idx: usize) -> u64 {
        self.extract_subswarm_word(granule_idx)
    }

    /// Writes a myopic 8-head `u64` SWAR granule word back into the trail buffer.
    #[inline(always)]
    fn write_granule_word(&mut self, granule_idx: usize, word: u64) {
        self.write_subswarm_word(granule_idx, word);
    }

    /// Fragments the trail across 8 myopic granules, drifts each along `target_words` gated by
    /// `active_granule_mask`, and reconstitutes the resulting state transition into
    /// `(StigmergicReconstitutionReport, reconstituted_words)` in branchless constant time.
    #[inline(always)]
    fn reconstitute_from_stigmergy_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
        active_granule_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]) {
        self.try_resolve_action_in_swarm_from_words(
            action_ptr,
            target_words,
            active_granule_mask,
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
}

impl gcso_pointer_trail_t {
    #[inline]
    pub(crate) fn try_resolve_action_in_swarm_with_cur_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]) {
        let ptr_phase = action_ptr.unpack_phase();
        let ptr_phase_splat = ((ptr_phase as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let cur_w = *cur_w;
        let tgt_base_w = *target_words;
        let tgt_has_explicit = swar_words_nonzero_64(&tgt_base_w);
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
        } else if active_cell_mask == 0 {
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

    #[cold]
    fn recruit_single_anchor_stragglers_cold(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        report: &mut StigmergicReconstitutionReport,
        next_words: &mut [u64; NUM_STIGMERGIC_GRANULES],
        tgt_w: &[u64; NUM_STIGMERGIC_GRANULES],
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

    #[cold]
    fn recruit_multi_anchor_stragglers_cold(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        report: &mut StigmergicReconstitutionReport,
        next_words: &mut [u64; NUM_STIGMERGIC_GRANULES],
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

    #[cold]
    fn recruit_slot_bank_stragglers_cold(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        slot_id: usize,
        orig_words: &[u64; NUM_STIGMERGIC_GRANULES],
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
        report: &mut StigmergicReconstitutionReport,
        next_words: &mut [u64; NUM_STIGMERGIC_GRANULES],
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
            let base = cell * HEADS_PER_STIGMERGIC_GRANULE;
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
            let eff_gain = ((gain_f32 * branchless_clamp_f32(donor_weight, 0.25, 1.0)) + 0.5) as u8;
            let gain_gt_128 = eff_gain > 128;
            let gain_s =
                branchless_select_u8(gain_gt_128, eff_gain.wrapping_sub(128), eff_gain) as u64;
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

impl StigmergicReconstitutionOps for gcso_pointer_trail_t {
    #[inline(always)]
    fn extract_subswarm_word(&self, cell_idx: usize) -> u64 {
        let safe_cell = cell_idx & (NUM_STIGMERGIC_GRANULES - 1);
        trail_phase_words_ref(self)[safe_cell]
    }

    #[inline(always)]
    fn write_subswarm_word(&mut self, cell_idx: usize, word: u64) {
        let safe_cell = cell_idx & (NUM_STIGMERGIC_GRANULES - 1);
        trail_phase_words_mut(self)[safe_cell] = word;
    }

    #[inline(always)]
    fn try_resolve_action_in_swarm_from_words(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]) {
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
        let has_best = donor_pull > DEFAULT_ENTROPY_EPS;
        self.target_anchor_id =
            branchless_select_u32(has_best, donor_anchor_id, self.target_anchor_id);
        self.cluster_id = branchless_select_u32(has_best, donor_anchor_id, self.cluster_id);
        let mut quorum_mask = prev_quorum;
        let mut unresolved = !prev_quorum;

        if unresolved != 0 {
            let eff_gain = ((f32::from(step_gain_u7) * branchless_clamp_f32(donor_pull, 0.25, 1.0))
                + 0.5) as u8;
            let (gain_s, gain_bias, gain_gt_128, qdps_thresh_x8, qdps_valid_mask) =
                precompute_swarm_step_params(eff_gain, qdps_min_step_q7);

            let self_w = trail_phase_words_mut(self);
            while unresolved != 0 {
                let cell = unresolved.trailing_zeros() as usize;
                unresolved &= unresolved - 1;
                let base = cell * HEADS_PER_STIGMERGIC_GRANULE;
                let cur_w = self_w[cell];
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
                self_w[cell] = candidate_w;
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
            swar_words_nonzero_64(&self_w)
        };
        let mut best_idx = 0usize;
        let mut best_pull = 0.0_f32;
        let mut idx = 0usize;
        while idx < count {
            let base_e = anchor_base_energies[idx];
            if base_e > DEFAULT_ENTROPY_EPS && branchless_is_finite_f32(base_e) {
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
            let qdps_thresh_x8 =
                ((qdps_min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let qdps_valid_mask = ((qdps_min_step_q7 <= 127) as u64).wrapping_neg();

            let self_w = trail_phase_words_mut(self);
            while unresolved != 0 {
                let cell = unresolved.trailing_zeros() as usize;
                unresolved &= unresolved - 1;
                let base = cell * HEADS_PER_STIGMERGIC_GRANULE;
                let cur_w = self_w[cell];
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
                let gain_s =
                    branchless_select_u8(gain_gt_128, eff_gain.wrapping_sub(128), eff_gain) as u64;
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
                self_w[cell] = candidate_w;
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
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]) {
        let count = anchor_phases.len().min(anchor_base_energies.len());
        let cur_w = *trail_phase_words_ref(self);

        let (tgt_w, total_pull, dom_id, has_active_anchors) = if count == 1 {
            let clean_e =
                branchless_flush_subnormal_f32(branchless_relu_f32(anchor_base_energies[0]));
            let has_active = clean_e > DEFAULT_ENTROPY_EPS;
            let tw = if has_active {
                load_q7_words_64(&anchor_phases[0])
            } else {
                [0u64; NUM_STIGMERGIC_GRANULES]
            };
            (tw, clean_e, 0u32, has_active)
        } else if count == 0 {
            ([0u64; NUM_STIGMERGIC_GRANULES], 0.0, 0u32, false)
        } else {
            let (tw, tp, _mc, did, ac) = branchless_q7_unified_field_fold_words_with_init_weight_64(
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
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]) {
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
}

