// name: src/core/src/runtime/storage/keyframe.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! KV-keyframe I/P-frame packet codecs and [`KvKeyframeCache`].

use super::KvKeyframeHeader;
use crate::foundation::{
    ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_ID_MASK, AlignedSlice16, AlignedSlice32,
    AlignedSlice64, AlignedSliceMut64, KvMotionDeltaPacket, Pod, TaggedPtr,
    apply_pframe_motion_step_words_64, branchless_abs_f32, branchless_atan2_f32,
    branchless_clamp_f32, branchless_copysign_f32, branchless_fast_transition_log_prob,
    branchless_flush_subnormal_f32, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_min_f32, branchless_q7_accumulate_64, branchless_q7_inhibit_words_64,
    branchless_q7_phase_step_filter_64, branchless_relu_f32, branchless_sincos_q7,
    branchless_sqrt_f32, branchless_von_mises_kappa, circular_r_bar_from_words_64,
    circular_resultant_from_words_64, coherence_from_diff_words_64, compute_crc32_ieee,
    crc32_block64_zero_seed_words, encode_pframe_motion_delta_64, finalize_pframe_chain_words_64,
    load_q7_words_64, pack_action_slot_tag, seek_pframe_chain_words_64, store_q7_words_64,
    swar_q7_add_sat_x8, swar_q7_circular_add_x8, swar_q7_fold_s1_geodesic_x8, swar_q7_neg_sat_x8,
    swar_q7_phase_step_filter_words_64, unpack_action_mode_from_tag,
    unpack_attractor_slot_from_tag,
};
use crate::liminal::{
    DEFAULT_ENTROPY_EPS, LMNK_ERR_BUFFER_TOO_SMALL, LMNK_ERR_CONTAINER_CORRUPTED,
    LMNK_ERR_INVALID_ARGUMENT, LMNK_ERR_INVALID_STATE, LmnkResult, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, PhaseQ7, Q7_PHASE_SCALE_RAD, branchless_select_f32,
    branchless_select_u32,
};
use crate::stigmergy::{
    ATTRACTOR_SLOT_REPULSION, ContrastivePhaseBloomFilter, LedgerSlot, MAX_ATTRACTOR_PULL_FORCE,
    PointerTrail, TRAIL_FLAG_REPULSION_ACTIVE, trail_phase_words_mut,
};

/// 128-byte cacheline-aligned Plain Old Data (`Pod`) **KV-keyframe Keyframe (I-Frame) Packet** (Section 4.3).
///
/// Packs the 64-byte [`KvKeyframeHeader`] metadata header (`offset 0..64`) and the
/// uncompressed 64-head Q7 phase anchor snapshot (`offset 64..128`) for lightweight video-codec-style
/// `.lmnk` memory & persona streams (portable container distribution ready).
///
/// The 24-byte `header.reserved` field encodes:
/// - `reserved[0..4]`: Little-endian IEEE 802.3 CRC32 checksum of `anchor_phases`.
/// - `reserved[4..8]`: Little-endian 32-bit `lineage_flags` causal mask.
/// - `reserved[8..12]`: Little-endian 32-bit macro Toroidal $k$-Means `cluster_id`.
/// - `reserved[12]`: 3-bit working-lane `slot_id` (`0..=7`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvKeyframePacket {
    /// 64-byte C-ABI KV-keyframe index header (`offset 0..64`, `frame_type == 0`).
    pub header: KvKeyframeHeader,
    /// 64-head Q7 phase anchor snapshot on $SO(2)^{64}$ (`offset 64..128`).
    pub anchor_phases: [PhaseQ7; MAX_ATTENTION_HEADS],
}

// SAFETY: `KvKeyframePacket` is `#[repr(C, align(64))]` of 128 bytes with zero padding traps.
unsafe impl Pod for KvKeyframePacket {}

impl Default for KvKeyframePacket {
    #[inline(always)]
    fn default() -> Self {
        Self {
            header: KvKeyframeHeader {
                frame_type: 0,
                token_index: 0,
                gop_length: 1,
                composite_vector_length: 0.0,
                von_mises_kappa: 0.0,
                sparse_scalar_residual: 0.0,
                icache_payload_offset: 0,
                pcache_payload_offset: 0,
                reserved: [0; 24],
            },
            anchor_phases: [0; MAX_ATTENTION_HEADS],
        }
    }
}

impl KvKeyframePacket {
    /// Computes the IEEE 802.3 CRC32 checksum over the 64-head `anchor_phases` payload in `const fn` mode.
    #[inline(always)]
    #[must_use]
    pub const fn compute_anchor_crc32(anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS]) -> u32 {
        let mut raw_bytes = [0u8; MAX_ATTENTION_HEADS];
        let mut idx = 0usize;
        while idx < MAX_ATTENTION_HEADS {
            raw_bytes[idx] = anchor_phases[idx] as u8;
            idx += 1;
        }
        compute_crc32_ieee(&raw_bytes)
    }

    /// Constructs a 128-byte [`KvKeyframePacket`] (I-Frame Key Anchor) stamped with IEEE 802.3
    /// CRC32 integrity protection and causal provenance metadata.
    #[inline(always)]
    #[must_use]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        token_index: u32,
        gop_length: u32,
        composite_vector_length: f32,
        von_mises_kappa: f32,
        sparse_scalar_residual: f32,
        lineage_flags: u32,
        cluster_id: u32,
        slot_id: u8,
        anchor_phases: [PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> Self {
        let crc = Self::compute_anchor_crc32(&anchor_phases);
        let crc_le = crc.to_le_bytes();
        let prov_le = lineage_flags.to_le_bytes();
        let cid_le = cluster_id.to_le_bytes();
        let mut reserved = [0u8; 24];
        reserved[0] = crc_le[0];
        reserved[1] = crc_le[1];
        reserved[2] = crc_le[2];
        reserved[3] = crc_le[3];
        reserved[4] = prov_le[0];
        reserved[5] = prov_le[1];
        reserved[6] = prov_le[2];
        reserved[7] = prov_le[3];
        reserved[8] = cid_le[0];
        reserved[9] = cid_le[1];
        reserved[10] = cid_le[2];
        reserved[11] = cid_le[3];
        reserved[12] = slot_id & ATTRACTOR_SLOT_ID_MASK;

        Self {
            header: KvKeyframeHeader {
                frame_type: 0,
                token_index,
                gop_length: if gop_length > 0 { gop_length } else { 1 },
                composite_vector_length,
                von_mises_kappa,
                sparse_scalar_residual,
                icache_payload_offset: 64,
                pcache_payload_offset: 128,
                reserved,
            },
            anchor_phases,
        }
    }

    /// Extracts the stamped IEEE 802.3 CRC32 checksum of `anchor_phases` from `header.reserved[0..4]`.
    #[inline(always)]
    #[must_use]
    pub const fn payload_crc32(&self) -> u32 {
        u32::from_le_bytes([
            self.header.reserved[0],
            self.header.reserved[1],
            self.header.reserved[2],
            self.header.reserved[3],
        ])
    }

    /// Extracts the 32-bit `lineage_flags` causal mask from `header.reserved[4..8]`.
    #[inline(always)]
    #[must_use]
    pub const fn lineage_flags(&self) -> u32 {
        u32::from_le_bytes([
            self.header.reserved[4],
            self.header.reserved[5],
            self.header.reserved[6],
            self.header.reserved[7],
        ])
    }

    /// Extracts the 32-bit Toroidal $k$-Means `cluster_id` from `header.reserved[8..12]`.
    #[inline(always)]
    #[must_use]
    pub const fn cluster_id(&self) -> u32 {
        u32::from_le_bytes([
            self.header.reserved[8],
            self.header.reserved[9],
            self.header.reserved[10],
            self.header.reserved[11],
        ])
    }

    /// Extracts the 3-bit working-lane `slot_id` (`0..=7`) from `header.reserved[12]`.
    #[inline(always)]
    #[must_use]
    pub const fn slot_id(&self) -> u8 {
        self.header.reserved[12] & ATTRACTOR_SLOT_ID_MASK
    }

    /// Verifies that `header.frame_type == 0` (I-Frame) and that the IEEE 802.3 CRC32 checksum of
    /// `anchor_phases` matches `self.payload_crc32()`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if `frame_type != 0` or CRC32 verification fails.
    #[inline(always)]
    pub const fn verify_integrity(&self) -> LmnkResult<()> {
        if self.header.frame_type != 0 {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        let expected_crc = self.payload_crc32();
        if expected_crc != 0 && Self::compute_anchor_crc32(&self.anchor_phases) != expected_crc {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        Ok(())
    }
}

/// Mezzo/Cold-path trait for Predictive Phase-Motion & Residual Compensation (KV-keyframe) (**Mezzo / Storage Level**).
pub trait KvKeyframeCache: Send + Sync {
    /// Seeks to `token_index` in the KV-keyframe keyframe cache and populates `header_out`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `token_index` is out of bounds.
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut KvKeyframeHeader,
    ) -> LmnkResult<()>;

    /// Computes Circular Statistics resultant vector length $\bar{R}_c \in [0, 1]$ and von Mises
    /// concentration $\kappa_c$ using $\mathcal{O}(1)$ 256-entry Q7 lookup (`branchless_sincos_q7`)
    /// and division-free (`0 fdiv`) `branchless_inv_pos_f32` (Section 4.2).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `pruned_phases` is empty or its length does not
    /// match `token_weights.len()`.
    #[inline]
    fn aggregate_pruned_kv_circular_stats(
        &self,
        pruned_phases: &AlignedSlice16<'_, PhaseQ7>,
        token_weights: &AlignedSlice32<'_, f32>,
        header_out: &mut KvKeyframeHeader,
    ) -> LmnkResult<(f32, f32)> {
        if pruned_phases.is_empty() || pruned_phases.len() != token_weights.len() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let phases = pruned_phases.as_slice();
        let weights = token_weights.as_slice();
        let n = phases.len();
        let mut sum_c0 = 0.0_f32;
        let mut sum_s0 = 0.0_f32;
        let mut sum_w0 = 0.0_f32;
        let mut sum_c1 = 0.0_f32;
        let mut sum_s1 = 0.0_f32;
        let mut sum_w1 = 0.0_f32;

        let mut idx = 0usize;
        while idx + 2 <= n {
            let w0 = branchless_relu_f32(weights[idx]);
            let (s0, c0) = branchless_sincos_q7(phases[idx]);
            sum_c0 += w0 * c0;
            sum_s0 += w0 * s0;
            sum_w0 += w0;

            let w1 = branchless_relu_f32(weights[idx + 1]);
            let (s1, c1) = branchless_sincos_q7(phases[idx + 1]);
            sum_c1 += w1 * c1;
            sum_s1 += w1 * s1;
            sum_w1 += w1;

            idx += 2;
        }
        if idx < n {
            let w0 = branchless_relu_f32(weights[idx]);
            let (s0, c0) = branchless_sincos_q7(phases[idx]);
            sum_c0 += w0 * c0;
            sum_s0 += w0 * s0;
            sum_w0 += w0;
        }
        let sum_c = sum_c0 + sum_c1;
        let sum_s = sum_s0 + sum_s1;
        let sum_w = sum_w0 + sum_w1;

        let norm = branchless_sqrt_f32(sum_c * sum_c + sum_s * sum_s);
        let r_bar = branchless_flush_subnormal_f32(branchless_select_f32(
            sum_w > DEFAULT_ENTROPY_EPS,
            norm * branchless_inv_pos_f32(sum_w),
            0.0,
        ));
        let r_bar_clamped = branchless_clamp_f32(r_bar, 0.0, 1.0);
        let kappa_c = branchless_von_mises_kappa(r_bar_clamped);
        let theta_bar = branchless_atan2_f32(sum_s, sum_c);

        header_out.composite_vector_length = r_bar_clamped;
        header_out.von_mises_kappa = kappa_c;
        header_out.sparse_scalar_residual = theta_bar;
        Ok((r_bar_clamped, kappa_c))
    }

    /// Computes Multi-Head Virtual Position Shift $\Delta m^{(h)}$ in division-free (`0 fdiv`)
    /// branchless mode (Section 4.3).
    #[inline(always)]
    #[must_use]
    fn compute_virtual_position_shift(
        &self,
        accumulated_phase_rad: f32,
        mean_base_rope_freq: f32,
    ) -> f32 {
        let abs_freq = branchless_abs_f32(mean_base_rope_freq);
        let valid_freq = (abs_freq > 1.0e-9) & branchless_is_finite_f32(mean_base_rope_freq);
        let inv_freq = branchless_copysign_f32(
            branchless_inv_pos_f32(branchless_select_f32(valid_freq, abs_freq, 1.0)),
            mean_base_rope_freq,
        );
        branchless_select_f32(
            valid_freq,
            branchless_flush_subnormal_f32(accumulated_phase_rad * inv_freq),
            0.0,
        )
    }

    /// Evaluates the **Variable Group of Pictures (GOP) Structure** keyframe classification
    /// (`0` = I-Frame Keyframe Anchor, `1` = P-Frame Phase Motion Delta) and effective GOP length (Section 4.3).
    #[inline(always)]
    #[must_use]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_variable_gop_trigger(
        &self,
        moving_z_entropy: f32,
        sliding_flux: f32,
        steps_since_keyframe: u32,
        min_gop: u32,
        max_gop: u32,
        entropy_surge_thresh: f32,
        flux_surge_thresh: f32,
    ) -> (u32, u32) {
        let safe_min = branchless_select_u32(min_gop > 0, min_gop, 1);
        let safe_max = branchless_select_u32(max_gop >= safe_min, max_gop, safe_min);
        let clean_h = branchless_select_f32(
            branchless_is_finite_f32(moving_z_entropy),
            moving_z_entropy,
            0.0,
        );
        let clean_flux =
            branchless_select_f32(branchless_is_finite_f32(sliding_flux), sliding_flux, 0.0);

        let surge = (clean_h > entropy_surge_thresh) | (clean_flux > flux_surge_thresh);
        let is_iframe =
            (steps_since_keyframe >= safe_max) | ((steps_since_keyframe >= safe_min) & surge);
        let frame_type = branchless_select_u32(is_iframe, 0, 1);
        let gop_len = branchless_select_u32(is_iframe, steps_since_keyframe.max(1), safe_max);
        (frame_type, gop_len)
    }

    /// Executes **Zero-Forward Latency Seek (Instantaneous Replay)** phase reconstruction (Section 4.3)
    /// using 8-lane SWAR saturating addition and SWAR `PhaseStepFilter` filtering without forward passes.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127`.
    #[inline]
    fn reconstruct_pframe_phase_trajectory(
        &self,
        icache_anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
        phase_step_min_q7: u8,
        reconstructed_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> LmnkResult<()> {
        if phase_step_min_q7 > 127 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        *reconstructed_out = *icache_anchor_phases;
        for delta in pcache_motion_deltas {
            branchless_q7_accumulate_64(reconstructed_out, delta);
        }
        branchless_q7_phase_step_filter_64(reconstructed_out, phase_step_min_q7);
        Ok(())
    }

    /// Executes **Seam-Free $S^1$ Geodesic P-Frame Trajectory Replay** (Section 4.3):
    /// accumulates P-Frame motion deltas on the circle group $S^1 \cong \mathbb{Z}/256\mathbb{Z}$,
    /// folds onto `[-64, +64]`, and applies 8-lane SWAR step filtering (`PhaseStepFilter`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127`.
    #[inline]
    fn reconstruct_pframe_geodesic_trajectory(
        &self,
        icache_anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
        phase_step_min_q7: u8,
        reconstructed_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> LmnkResult<()> {
        if phase_step_min_q7 > 127 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let mut cur_w = load_q7_words_64(icache_anchor_phases);
        for delta in pcache_motion_deltas {
            let dw = load_q7_words_64(delta);
            cur_w[0] = swar_q7_circular_add_x8(cur_w[0], dw[0]);
            cur_w[1] = swar_q7_circular_add_x8(cur_w[1], dw[1]);
            cur_w[2] = swar_q7_circular_add_x8(cur_w[2], dw[2]);
            cur_w[3] = swar_q7_circular_add_x8(cur_w[3], dw[3]);
            cur_w[4] = swar_q7_circular_add_x8(cur_w[4], dw[4]);
            cur_w[5] = swar_q7_circular_add_x8(cur_w[5], dw[5]);
            cur_w[6] = swar_q7_circular_add_x8(cur_w[6], dw[6]);
            cur_w[7] = swar_q7_circular_add_x8(cur_w[7], dw[7]);
        }
        let folded_w = [
            swar_q7_fold_s1_geodesic_x8(cur_w[0]),
            swar_q7_fold_s1_geodesic_x8(cur_w[1]),
            swar_q7_fold_s1_geodesic_x8(cur_w[2]),
            swar_q7_fold_s1_geodesic_x8(cur_w[3]),
            swar_q7_fold_s1_geodesic_x8(cur_w[4]),
            swar_q7_fold_s1_geodesic_x8(cur_w[5]),
            swar_q7_fold_s1_geodesic_x8(cur_w[6]),
            swar_q7_fold_s1_geodesic_x8(cur_w[7]),
        ];
        let filt_w = swar_q7_phase_step_filter_words_64(&folded_w, phase_step_min_q7);
        store_q7_words_64(reconstructed_out, &filt_w);
        Ok(())
    }

    /// Replays a **Limina P-Frame Motion Chain** ($\boldsymbol{\Delta\theta}_1, \dots, \boldsymbol{\Delta\theta}_T$)
    /// from an **I-Frame Key Anchor** (`icache_anchor_phases`), evaluating the **Joint Probability
    /// Chain Rule Log-Likelihood** $\ln P(M_1, \dots, M_T) = \sum_{t=1}^T \ln P(M_t \mid M_{t-1})$
    /// and inverting any self-contradictory ($H^1(K; \mathcal{F}) \neq 0$) P-Frame step via
    /// Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}_t$).
    ///
    /// Populates `header_out` (`gop_length`, `composite_vector_length = r_bar`,
    /// `von_mises_kappa = kappa_c`, `sparse_scalar_residual = joint_log_prob`) and returns
    /// `(joint_log_prob, r_bar, kappa_c, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127` or `contradiction_threshold` is non-finite.
    #[inline]
    fn replay_liminal_pframe_chain_with_joint_log_prob(
        &self,
        icache_anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
        phase_step_min_q7: u8,
        contradiction_threshold: f32,
        reconstructed_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
        header_out: &mut KvKeyframeHeader,
    ) -> LmnkResult<(f32, f32, f32, u32)> {
        if phase_step_min_q7 > 127 || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let mut cur_w = load_q7_words_64(icache_anchor_phases);
        let mut joint_log_prob = 0.0_f32;
        let mut repulsed_count = 0u32;

        for delta in pcache_motion_deltas {
            let dw = load_q7_words_64(delta);
            // Because `next_w = cur_w (+) dw` in Z_256^64 and cos(-d) = cos(d), the transition
            // coherence C(cur_w, next_w) depends purely on `dw` against zero, avoiding a
            // tentative 8-word addition + 8-word subtraction before checking repulsion.
            let coh = coherence_from_diff_words_64(
                dw[0], dw[1], dw[2], dw[3], dw[4], dw[5], dw[6], dw[7],
            );
            let log_p = branchless_fast_transition_log_prob(coh);
            if coh < -safe_thresh {
                cur_w = [
                    swar_q7_circular_add_x8(cur_w[0], swar_q7_neg_sat_x8(dw[0])),
                    swar_q7_circular_add_x8(cur_w[1], swar_q7_neg_sat_x8(dw[1])),
                    swar_q7_circular_add_x8(cur_w[2], swar_q7_neg_sat_x8(dw[2])),
                    swar_q7_circular_add_x8(cur_w[3], swar_q7_neg_sat_x8(dw[3])),
                    swar_q7_circular_add_x8(cur_w[4], swar_q7_neg_sat_x8(dw[4])),
                    swar_q7_circular_add_x8(cur_w[5], swar_q7_neg_sat_x8(dw[5])),
                    swar_q7_circular_add_x8(cur_w[6], swar_q7_neg_sat_x8(dw[6])),
                    swar_q7_circular_add_x8(cur_w[7], swar_q7_neg_sat_x8(dw[7])),
                ];
                repulsed_count = repulsed_count.wrapping_add(1);
            } else {
                cur_w = [
                    swar_q7_circular_add_x8(cur_w[0], dw[0]),
                    swar_q7_circular_add_x8(cur_w[1], dw[1]),
                    swar_q7_circular_add_x8(cur_w[2], dw[2]),
                    swar_q7_circular_add_x8(cur_w[3], dw[3]),
                    swar_q7_circular_add_x8(cur_w[4], dw[4]),
                    swar_q7_circular_add_x8(cur_w[5], dw[5]),
                    swar_q7_circular_add_x8(cur_w[6], dw[6]),
                    swar_q7_circular_add_x8(cur_w[7], dw[7]),
                ];
            }
            joint_log_prob += log_p;
        }

        let folded_w = [
            swar_q7_fold_s1_geodesic_x8(cur_w[0]),
            swar_q7_fold_s1_geodesic_x8(cur_w[1]),
            swar_q7_fold_s1_geodesic_x8(cur_w[2]),
            swar_q7_fold_s1_geodesic_x8(cur_w[3]),
            swar_q7_fold_s1_geodesic_x8(cur_w[4]),
            swar_q7_fold_s1_geodesic_x8(cur_w[5]),
            swar_q7_fold_s1_geodesic_x8(cur_w[6]),
            swar_q7_fold_s1_geodesic_x8(cur_w[7]),
        ];
        let filt_w = swar_q7_phase_step_filter_words_64(&folded_w, phase_step_min_q7);
        store_q7_words_64(reconstructed_out, &filt_w);

        let r_bar = circular_r_bar_from_words_64(&filt_w);
        let kappa_c = branchless_von_mises_kappa(r_bar);
        let clean_lp = branchless_flush_subnormal_f32(joint_log_prob);
        header_out.gop_length = (pcache_motion_deltas.len() as u32).max(1);
        header_out.composite_vector_length = r_bar;
        header_out.von_mises_kappa = kappa_c;
        header_out.sparse_scalar_residual = clean_lp;

        Ok((clean_lp, r_bar, kappa_c, repulsed_count))
    }

    /// Encodes a 128-byte [`KvKeyframePacket`] (**I-Frame Key Anchor**, Section 4.3) from a
    /// 64-head Q7 phase profile (`anchor_phases`), computing Circular Statistics resultant vector
    /// length $\bar{R}_c \in [0, 1]$, von Mises concentration $\kappa_c$, resultant mean phase
    /// angle $\bar{\theta}_c$, and stamping the IEEE 802.3 CRC32 integrity checksum plus causal
    /// provenance metadata (`lineage_flags`, `cluster_id`, `slot_id`).
    #[inline]
    #[must_use]
    fn encode_kv_keyframe_iframe_packet(
        &self,
        token_index: u32,
        gop_length: u32,
        anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        lineage_flags: u32,
        cluster_id: u32,
        slot_id: u8,
    ) -> KvKeyframePacket {
        let words = load_q7_words_64(anchor_phases);
        let (r_bar, mean_q7, kappa_c) = circular_resultant_from_words_64(&words);
        let theta_bar = branchless_flush_subnormal_f32((mean_q7 as f32) * Q7_PHASE_SCALE_RAD);
        KvKeyframePacket::new(
            token_index,
            gop_length.max(1),
            r_bar,
            kappa_c,
            theta_bar,
            lineage_flags,
            cluster_id,
            slot_id,
            *anchor_phases,
        )
    }

    /// Encodes a 64-byte [`KvMotionDeltaPacket`] (**P-Frame Predictive Phase-Motion Delta**
    /// $\boldsymbol{\Delta\theta}_t$, Section 4.3) between `prev_phases` ($\boldsymbol{\theta}_{t-1}$)
    /// and `curr_phases` ($\boldsymbol{\theta}_t$) using 8-lane SWAR circular difference
    /// (`swar_q7_circular_diff_x8`), $S^1$ geodesic folding (`swar_q7_fold_s1_geodesic_x8`), and
    /// step filtering (`PhaseStepFilter`) (`swar_q7_phase_step_filter_words_64`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127`.
    #[inline]
    fn encode_kv_keyframe_pframe_delta_packet(
        &self,
        prev_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        curr_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        phase_step_min_q7: u8,
    ) -> LmnkResult<KvMotionDeltaPacket> {
        encode_pframe_motion_delta_64(prev_phases, curr_phases, phase_step_min_q7)
    }

    /// Executes **Zero-Forward Latency Seek (Instantaneous Persona & Memory Replay)** over a
    /// lightweight KV-keyframe Variable GOP stream (`keyframe` I-Frame + `pframes[..seek_pframe_steps]`
    /// P-Frame deltas), reconstructing the historical 64-head persona & episodic memory state
    /// directly onto `trail_out` without running any LLM forward pass (Section 4.3).
    ///
    /// Returns `(joint_log_prob, r_bar, kappa_c, virtual_pos_shift, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if `keyframe.verify_integrity()` fails, or
    /// [`LMNK_ERR_INVALID_ARGUMENT`] if `seek_pframe_steps > pframes.len()`,
    /// `phase_step_min_q7 > 127`, or `contradiction_threshold` is non-finite.
    #[inline]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    fn seek_and_reconstruct_persona_from_gop(
        &self,
        keyframe: &KvKeyframePacket,
        pframes: &[KvMotionDeltaPacket],
        seek_pframe_steps: usize,
        phase_step_min_q7: u8,
        contradiction_threshold: f32,
        mean_base_rope_freq: f32,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<(f32, f32, f32, f32, u32)> {
        keyframe.verify_integrity()?;
        let (filt_w, clean_lp, repulsed_count) = seek_pframe_chain_words_64(
            &keyframe.anchor_phases,
            pframes,
            seek_pframe_steps,
            phase_step_min_q7,
            contradiction_threshold,
        )?;
        store_q7_words_64(&mut trail_out.accumulated_phase_delta, &filt_w);

        let (r_bar, mean_q7, kappa_c) = circular_resultant_from_words_64(&filt_w);
        let mean_rad = (mean_q7 as f32) * Q7_PHASE_SCALE_RAD;
        let virtual_pos_shift = self.compute_virtual_position_shift(mean_rad, mean_base_rope_freq);

        let cur_mode = unpack_action_mode_from_tag(TaggedPtr(trail_out.current_ptr).unpack_tag());
        let target_token = keyframe
            .header
            .token_index
            .wrapping_add(seek_pframe_steps as u32);
        trail_out.current_ptr = TaggedPtr::pack_action(
            u64::from(target_token),
            keyframe.slot_id(),
            cur_mode,
            mean_q7,
        )
        .raw();
        trail_out.user_data |= u64::from(keyframe.lineage_flags());
        trail_out.target_anchor_id = keyframe.cluster_id();
        trail_out.cluster_id = keyframe.cluster_id();
        if trail_out.stigmergic_density < r_bar {
            trail_out.stigmergic_density = r_bar;
        }
        Ok((clean_lp, r_bar, kappa_c, virtual_pos_shift, repulsed_count))
    }

    /// Serializes a lightweight KV-keyframe Variable GOP stream (`1` 128-byte [`KvKeyframePacket`] +
    /// `pframes.len()` 64-byte [`KvMotionDeltaPacket`]s) into `dst_bytes` (`128 + N * 64` bytes),
    /// returning the total byte length written.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if `keyframe.verify_integrity()` fails, or
    /// [`LMNK_ERR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < 128 + pframes.len() * 64`.
    #[inline]
    fn pack_kv_gop_stream_bytes(
        &self,
        keyframe: &KvKeyframePacket,
        pframes: &[KvMotionDeltaPacket],
        dst_bytes: &mut [u8],
    ) -> LmnkResult<usize> {
        keyframe.verify_integrity()?;
        let p_bytes = pframes
            .len()
            .checked_mul(core::mem::size_of::<KvMotionDeltaPacket>())
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        let total_bytes = core::mem::size_of::<KvKeyframePacket>()
            .checked_add(p_bytes)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if dst_bytes.len() < total_bytes {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        keyframe.write_unaligned_to(dst_bytes, 0)?;
        KvMotionDeltaPacket::write_unaligned_slice_to(
            pframes,
            dst_bytes,
            core::mem::size_of::<KvKeyframePacket>(),
        )?;
        Ok(total_bytes)
    }

    /// Unpacks a serialized KV-keyframe Variable GOP byte stream (`stream_bytes`) in $\mathcal{O}(1)$
    /// stack space ($C_{\mathrm{alloc}} = 0$) and executes **Zero-Forward Latency Seek** up to
    /// `seek_pframe_steps`, reconstructing the historical persona and episodic memory state onto
    /// `trail_out` (Section 4.3).
    ///
    /// Returns `(joint_log_prob, r_bar, kappa_c, virtual_pos_shift, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `stream_bytes` is truncated,
    /// [`LMNK_ERR_CONTAINER_CORRUPTED`] if the I-Frame header or CRC32 is invalid, or
    /// [`LMNK_ERR_INVALID_ARGUMENT`] if parameters are out of range.
    #[inline]
    fn unpack_and_seek_kv_gop_stream_bytes(
        &self,
        stream_bytes: &[u8],
        seek_pframe_steps: usize,
        phase_step_min_q7: u8,
        contradiction_threshold: f32,
        mean_base_rope_freq: f32,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<(f32, f32, f32, f32, u32)> {
        if phase_step_min_q7 > 127 || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let iframe_size = core::mem::size_of::<KvKeyframePacket>();
        let pframe_size = core::mem::size_of::<KvMotionDeltaPacket>();
        let needed_bytes = seek_pframe_steps
            .checked_mul(pframe_size)
            .and_then(|b| b.checked_add(iframe_size))
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if stream_bytes.len() < needed_bytes {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        let keyframe = KvKeyframePacket::read_unaligned_from(stream_bytes, 0)?;
        keyframe.verify_integrity()?;

        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let mut cur_w = load_q7_words_64(&keyframe.anchor_phases);
        let mut joint_log_prob = 0.0_f32;
        let mut repulsed_count = 0u32;

        let mut step = 0usize;
        while step < seek_pframe_steps {
            let off = iframe_size + step * pframe_size;
            let pframe = KvMotionDeltaPacket::read_unaligned_from(stream_bytes, off)?;
            let (log_p, repulsed) =
                apply_pframe_motion_step_words_64(&mut cur_w, &pframe, safe_thresh);
            joint_log_prob += log_p;
            if repulsed {
                repulsed_count = repulsed_count.wrapping_add(1);
            }
            step += 1;
        }

        let filt_w = finalize_pframe_chain_words_64(&cur_w, phase_step_min_q7);
        store_q7_words_64(&mut trail_out.accumulated_phase_delta, &filt_w);

        let (r_bar, mean_q7, kappa_c) = circular_resultant_from_words_64(&filt_w);
        let clean_lp = branchless_flush_subnormal_f32(joint_log_prob);
        let mean_rad = (mean_q7 as f32) * Q7_PHASE_SCALE_RAD;
        let virtual_pos_shift = self.compute_virtual_position_shift(mean_rad, mean_base_rope_freq);

        let cur_mode = unpack_action_mode_from_tag(TaggedPtr(trail_out.current_ptr).unpack_tag());
        let target_token = keyframe
            .header
            .token_index
            .wrapping_add(seek_pframe_steps as u32);
        trail_out.current_ptr = TaggedPtr::pack_action(
            u64::from(target_token),
            keyframe.slot_id(),
            cur_mode,
            mean_q7,
        )
        .raw();
        trail_out.user_data |= u64::from(keyframe.lineage_flags());
        trail_out.target_anchor_id = keyframe.cluster_id();
        trail_out.cluster_id = keyframe.cluster_id();
        if trail_out.stigmergic_density < r_bar {
            trail_out.stigmergic_density = r_bar;
        }
        Ok((clean_lp, r_bar, kappa_c, virtual_pos_shift, repulsed_count))
    }
}

// ===================================================================
// Compact 64-Byte Phase Synchronization Packet Helpers
// ===================================================================

/// Dispatches a compact 64-byte phase/repulsion synchronization packet in zero-allocation mode.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_STATE`] if `ledger_slot.mode != 1` (Network Plugin Mode).
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
#[inline]
pub fn dispatch_phase_packet(
    ledger_slot: &LedgerSlot,
    packet_payload: &AlignedSlice64<'_, u8>,
) -> LmnkResult<()> {
    if ledger_slot.mode != 1 {
        return Err(LMNK_ERR_INVALID_STATE);
    }
    if packet_payload.len() < 64 {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    Ok(())
}

/// Encodes a 64-head Q7 phase/repulsion vector into a 64-byte cacheline-aligned phase
/// packet (`packet_out`) using bulk `Pod` copy and returns its branchless IEEE 802.3 CRC32 integrity tag.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_STATE`] if `ledger_slot.mode != 1`.
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `packet_out.len() < 64` or `source_slot_id >= 8`.
#[inline]
pub fn encode_phase_packet(
    ledger_slot: &LedgerSlot,
    source_slot_id: u8,
    action_mode_flags: u8,
    phase_deltas: &[PhaseQ7; MAX_ATTENTION_HEADS],
    packet_out: &mut AlignedSliceMut64<'_, u8>,
) -> LmnkResult<u32> {
    if ledger_slot.mode != 1 {
        return Err(LMNK_ERR_INVALID_STATE);
    }
    if packet_out.len() < MAX_ATTENTION_HEADS || (source_slot_id as usize) >= MAX_ATTRACTOR_DOMAINS
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let phase_w = load_q7_words_64(phase_deltas);
    let dst = &mut packet_out.as_mut_slice()[..MAX_ATTENTION_HEADS];
    let mut cell = 0usize;
    while cell < 8 {
        let base = cell << 3;
        dst[base..base + 8].copy_from_slice(&phase_w[cell].to_le_bytes());
        cell += 1;
    }
    let tag_byte = pack_action_slot_tag(source_slot_id, action_mode_flags);
    let crc = crc32_block64_zero_seed_words(&phase_w) ^ u32::from(tag_byte);
    Ok(crc)
}

/// Verifies and decodes a 64-byte cacheline-aligned phase packet (`packet_payload`) into
/// 8 register-resident `u64` SWAR words and the recovered 8-bit tag byte.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_STATE`] if `ledger_slot.mode != 1`.
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
/// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if the upper 24 bits of the IEEE 802.3 CRC32
/// checksum do not match `expected_crc32`.
#[inline(always)]
pub fn decode_and_verify_phase_packet_words(
    ledger_slot: &LedgerSlot,
    packet_payload: &AlignedSlice64<'_, u8>,
    expected_crc32: u32,
) -> LmnkResult<([u64; 8], u8)> {
    if ledger_slot.mode != 1 {
        return Err(LMNK_ERR_INVALID_STATE);
    }
    if packet_payload.len() < MAX_ATTENTION_HEADS {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let src = &packet_payload.as_slice()[..MAX_ATTENTION_HEADS];
    let payload_words = [
        u64::from_le_bytes([
            src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
        ]),
        u64::from_le_bytes([
            src[8], src[9], src[10], src[11], src[12], src[13], src[14], src[15],
        ]),
        u64::from_le_bytes([
            src[16], src[17], src[18], src[19], src[20], src[21], src[22], src[23],
        ]),
        u64::from_le_bytes([
            src[24], src[25], src[26], src[27], src[28], src[29], src[30], src[31],
        ]),
        u64::from_le_bytes([
            src[32], src[33], src[34], src[35], src[36], src[37], src[38], src[39],
        ]),
        u64::from_le_bytes([
            src[40], src[41], src[42], src[43], src[44], src[45], src[46], src[47],
        ]),
        u64::from_le_bytes([
            src[48], src[49], src[50], src[51], src[52], src[53], src[54], src[55],
        ]),
        u64::from_le_bytes([
            src[56], src[57], src[58], src[59], src[60], src[61], src[62], src[63],
        ]),
    ];
    let raw_crc = crc32_block64_zero_seed_words(&payload_words);
    let diff = raw_crc ^ expected_crc32;
    if (diff & 0xFFFF_FF00) != 0 {
        return Err(LMNK_ERR_CONTAINER_CORRUPTED);
    }
    Ok((payload_words, (diff & 0xFF) as u8))
}

/// Verifies and decodes a 64-byte cacheline-aligned LiminiKa Phase Packet (`packet_payload`) into
/// a 64-head Q7 phase/repulsion vector (`phases_out`), returning the recovered 8-bit tag byte.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_STATE`] if `ledger_slot.mode != 1`.
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
/// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if the upper 24 bits of the IEEE 802.3 CRC32
/// checksum do not match `expected_crc32`.
#[inline]
pub fn decode_and_verify_phase_packet(
    ledger_slot: &LedgerSlot,
    packet_payload: &AlignedSlice64<'_, u8>,
    expected_crc32: u32,
    phases_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
) -> LmnkResult<u8> {
    match decode_and_verify_phase_packet_words(ledger_slot, packet_payload, expected_crc32) {
        Ok((payload_words, tag_byte)) => {
            store_q7_words_64(phases_out, &payload_words);
            Ok(tag_byte)
        }
        Err(err) => {
            if err == LMNK_ERR_CONTAINER_CORRUPTED {
                *phases_out = [0i8; MAX_ATTENTION_HEADS];
            }
            Err(err)
        }
    }
}

/// Verifies a 64-byte LiminiKa Phase Packet and applies its decoded 64-head phase/repulsion vector
/// directly to `trail` (and `bloom_filter` if repulsive).
///
/// # Errors
/// Propagates any error returned by [`decode_and_verify_phase_packet`].
#[inline]
pub fn decode_and_apply_phase_packet_to_trail(
    ledger_slot: &LedgerSlot,
    packet_payload: &AlignedSlice64<'_, u8>,
    expected_crc32: u32,
    inhibition_strength: f32,
    bloom_filter: Option<&mut ContrastivePhaseBloomFilter>,
    trail: &mut PointerTrail,
) -> LmnkResult<(u8, u8, bool)> {
    let (payload_words, tag_byte) =
        decode_and_verify_phase_packet_words(ledger_slot, packet_payload, expected_crc32)?;
    let slot_id = unpack_attractor_slot_from_tag(tag_byte);
    let mode_flags = unpack_action_mode_from_tag(tag_byte);
    let is_repulsion =
        (slot_id == ATTRACTOR_SLOT_REPULSION) | ((mode_flags & ACTION_MODE_PHASE_REPULSE) != 0);

    if is_repulsion {
        if let Some(bf) = bloom_filter {
            bf.insert_repulsive_pattern_words_weighted(&payload_words, 1);
        }
        let cur_pull = trail.attractor_pull_force;
        let damped = branchless_q7_inhibit_words_64(
            trail_phase_words_mut(trail),
            cur_pull,
            &payload_words,
            inhibition_strength,
        );
        trail.attractor_pull_force = branchless_min_f32(
            branchless_flush_subnormal_f32(damped),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
    } else {
        let tw = trail_phase_words_mut(trail);
        tw[0] = swar_q7_add_sat_x8(tw[0], payload_words[0]);
        tw[1] = swar_q7_add_sat_x8(tw[1], payload_words[1]);
        tw[2] = swar_q7_add_sat_x8(tw[2], payload_words[2]);
        tw[3] = swar_q7_add_sat_x8(tw[3], payload_words[3]);
        tw[4] = swar_q7_add_sat_x8(tw[4], payload_words[4]);
        tw[5] = swar_q7_add_sat_x8(tw[5], payload_words[5]);
        tw[6] = swar_q7_add_sat_x8(tw[6], payload_words[6]);
        tw[7] = swar_q7_add_sat_x8(tw[7], payload_words[7]);
    }
    Ok((slot_id, mode_flags, is_repulsion))
}
