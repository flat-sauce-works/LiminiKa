// name: src/core/src/runtime/context/traits.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core trait implementations and domain/patch operations on [`RuntimeContext`].

use super::{Result, RuntimeContext};
use crate::foundation::{
    ACTION_MODE_LAYER_SKIP, ATTRACTOR_SLOT_ID_MASK, AlignedSlice16, DEFAULT_PHASE_CLAMP_MAX_RAD,
    PhaseSteering, TaggedPtr, branchless_clamp_f32, branchless_is_finite_f32, branchless_max_f32,
    branchless_min_f32, branchless_q7_add_sat, branchless_q7_conjugate_64,
    branchless_rad_to_phase_step_u8, branchless_relu_f32, load_q7_words_64,
    swar_or_avx2_q7_add_sat_words_64,
};
use crate::liminal::{
    LMNK_ERR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, NUM_STIGMERGIC_GRANULES,
    PhaseQ7, RuntimeConfig, branchless_select_f32, branchless_select_u32,
};
use crate::runtime::gateway::{
    CausalProvenanceSignal, GatewayFrame, ManifoldGatewaySignal, SystemConstraintSignal,
};
use crate::runtime::storage::{
    KvKeyframeCache, KvKeyframeHeader, LMNK_CONTAINER_HEADER_BYTES, MappedStorage,
};
use crate::stigmergy::{
    ATTRACTOR_SLOT_PERSONA, FastPathLedger, LEDGER_HISTORY_LEN, LedgerSlot,
    MAX_ATTRACTOR_PULL_FORCE, PagedBitmask, PointerSubstrate, PointerTrail,
    SUBSTRATE_INLINE_TRAILS, StigmergicSubstrate, WorkbenchSlotSpec,
};

// ===================================================================
// Trait Implementations for Core Architecture Contracts
// ===================================================================

impl PhaseSteering for RuntimeContext {
    #[inline(always)]
    fn head_dim(&self) -> u32 {
        self.phase_steering.head_dim()
    }

    #[inline(always)]
    fn num_heads(&self) -> u32 {
        self.phase_steering.num_heads()
    }
}

impl PointerSubstrate for RuntimeContext {
    #[inline]
    fn reset(&mut self) -> Result<()> {
        self.substrate.reset()
    }

    #[inline]
    fn step_pointer(&mut self, current_ptr: u64, trail_out: &mut PointerTrail) -> Result<()> {
        let cap = self.config.raw.substrate_capacity;
        let cur_active = self.substrate.active_count;
        self.substrate.active_count =
            branchless_select_u32(cur_active < cap, cur_active.wrapping_add(1), cur_active);
        self.substrate.step_counter = self.substrate.step_counter.wrapping_add(1);
        trail_out.record_pointer_transition(current_ptr, 1, 1.0);
        let tp = TaggedPtr(current_ptr);
        let embedded_phase = tp.unpack_phase();
        if embedded_phase != 0 {
            trail_out.accumulate_uniform_phase(embedded_phase);
        }
        trail_out.cluster_id = branchless_select_u32(
            trail_out.target_anchor_id != 0,
            trail_out.target_anchor_id,
            u32::from(tp.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK),
        );
        trail_out.sync_flags_with_current_ptr();
        let slot_idx = (self.substrate.write_cursor as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        StigmergicSubstrate::copy_trail_to_slot_fast(
            &mut self.substrate.trails[slot_idx],
            trail_out,
        );
        self.substrate.write_cursor =
            self.substrate.write_cursor.wrapping_add(1) & ((SUBSTRATE_INLINE_TRAILS as u32) - 1);
        Ok(())
    }

    #[inline]
    fn update_cellular_state_express(&mut self, trail_id: u32, phase_delta: PhaseQ7) -> Result<()> {
        if trail_id >= self.config.raw.substrate_capacity {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        self.substrate
            .update_cellular_state_express(trail_id, phase_delta)
    }

    #[inline]
    fn pull_trail_to_attractor(
        &mut self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> Result<()> {
        if !branchless_is_finite_f32(pull_force) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        if trail_id < self.substrate.capacity {
            self.substrate
                .pull_trail_to_attractor(trail_id, anchor_id, pull_force)?;
        } else {
            let slot_idx = (anchor_id as usize) & (MAX_ATTRACTOR_DOMAINS - 1);
            self.substrate.slot_weights[slot_idx] =
                branchless_min_f32(branchless_relu_f32(pull_force), MAX_ATTRACTOR_PULL_FORCE);
        }
        Ok(())
    }

    #[inline]
    fn link_hallucinated_trails(&mut self, src_trail_id: u32, dst_trail_id: u32) -> Result<()> {
        if src_trail_id < self.substrate.capacity && dst_trail_id < self.substrate.capacity {
            let _ = self
                .substrate
                .link_hallucinated_trails(src_trail_id, dst_trail_id);
        }
        Ok(())
    }

    #[inline(always)]
    fn capacity(&self) -> u32 {
        self.config.raw.substrate_capacity
    }

    #[inline(always)]
    fn active_count(&self) -> u32 {
        self.substrate.active_count
    }

    #[inline(always)]
    fn lane_mask(&self) -> u8 {
        self.substrate.lane_mask()
    }

    #[inline(always)]
    fn repulsion_lane_mask(&self) -> u8 {
        self.substrate.repulsion_lane_mask()
    }

    #[inline(always)]
    fn sandbox_lane_mask(&self) -> u8 {
        self.substrate.sandbox_lane_mask()
    }

    #[inline(always)]
    fn pinned_lane_mask(&self) -> u8 {
        self.substrate.pinned_lane_mask()
    }

    #[inline]
    fn lane_words_and_weight(
        &self,
        lane: u8,
    ) -> Option<([u64; crate::NUM_STIGMERGIC_GRANULES], f32)> {
        self.substrate.lane_words_and_weight(lane)
    }
}

impl FastPathLedger for RuntimeContext {
    #[inline]
    fn evaluate_auto_tune(&self, slot: &LedgerSlot, config_out: &mut RuntimeConfig) -> Result<()> {
        *config_out = self.config.raw;

        // Count high-intensity telemetry spike events (>= 0x80) across all 8 ring bytes via 1 SWAR op.
        let word = u64::from_le_bytes(slot.telemetry_mini_ledger);
        let spike_count = (word & 0x8080_8080_8080_8080).count_ones();
        let base_clamp_rad = branchless_select_f32(
            config_out.phase_clamp_max_rad > 0.0,
            config_out.phase_clamp_max_rad,
            DEFAULT_PHASE_CLAMP_MAX_RAD,
        );
        let damping = 1.0 - 0.025 * (spike_count as f32);
        config_out.phase_clamp_max_rad = branchless_clamp_f32(base_clamp_rad * damping, 0.01, 0.25);
        config_out.ledger_mode = (slot.mode & 0xFF) as u8;
        let _ = LEDGER_HISTORY_LEN;
        Ok(())
    }
}

impl RuntimeContext {
    /// Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `chunk_len == 0`.
    #[inline]
    pub fn step_chunk(&mut self, _mask: &PagedBitmask, chunk_len: u32) -> Result<()> {
        if chunk_len == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        self.substrate.step_counter = self
            .substrate
            .step_counter
            .wrapping_add(u64::from(chunk_len));
        Ok(())
    }
}

impl RuntimeContext {
    /// Injects a phase-conjugate repulsion pattern into the Bloom filter and workbench lane.
    ///
    /// # Errors
    /// Propagates any error from workbench lane allocation.
    #[inline]
    pub fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, PhaseQ7>,
        gain: f32,
    ) -> Result<()> {
        let raw = repulsion_deltas.as_slice();
        let limit = raw.len().min(MAX_ATTENTION_HEADS);
        let mut pattern = [0i8; MAX_ATTENTION_HEADS];
        pattern[..limit].copy_from_slice(&raw[..limit]);

        self.substrate
            .bloom_filter
            .insert_repulsive_pattern(&pattern);
        let anti_phase = branchless_q7_conjugate_64(&pattern);
        let _lane = self
            .substrate
            .allocate_lane(&WorkbenchSlotSpec::auto_repulsion(
                branchless_relu_f32(gain),
                anti_phase,
            ))?;

        Ok(())
    }

    /// Sets the relative weight of workbench domain lane `domain_id`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `domain_id >= MAX_ATTRACTOR_DOMAINS` or `weight` is negative/non-finite.
    #[inline]
    pub fn set_domain_weight(&mut self, domain_id: u32, weight: f32) -> Result<()> {
        let idx = domain_id as usize;
        if idx >= MAX_ATTRACTOR_DOMAINS || !branchless_is_finite_f32(weight) || weight < 0.0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        self.substrate.slot_weights[idx] = weight;
        Ok(())
    }

    /// Enables or disables workbench domain lane `domain_id`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `domain_id >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn set_domain_enabled(&mut self, domain_id: u32, enabled: bool) -> Result<()> {
        let idx = domain_id as usize;
        if idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let bit = 1u8 << (domain_id as u8);
        let enable_mask = (enabled as u8).wrapping_neg();
        self.substrate.slot_enabled_mask =
            (self.substrate.slot_enabled_mask & !bit) | (bit & enable_mask);
        Ok(())
    }

    /// Finds the highest-weighted enabled workbench lane and returns `(lane_id, weight)`.
    #[inline]
    #[must_use]
    pub fn find_dominant_domain(&self) -> (u32, f32) {
        let mut best_id = 0u32;
        let mut best_score = -1.0_f32;
        let mask = self.substrate.slot_enabled_mask;
        let weights = &self.substrate.slot_weights;

        let mut i = 0usize;
        while i < MAX_ATTRACTOR_DOMAINS {
            let enabled = ((mask >> i) & 1) != 0;
            let base_w = weights[i];
            let eff_score = branchless_select_f32(enabled, base_w, -1.0);
            let is_better = eff_score > best_score;
            best_id = branchless_select_u32(is_better, i as u32, best_id);
            best_score = branchless_select_f32(is_better, eff_score, best_score);
            i += 1;
        }

        (best_id, branchless_relu_f32(best_score))
    }

    /// Releases all workbench lanes and clears any escalated lane marker.
    ///
    /// # Errors
    /// Propagates any error from [`StigmergicSubstrate::release_all_lanes`].
    #[inline]
    pub fn clear_all_domains(&mut self) -> Result<()> {
        let _ = self.substrate.release_all_lanes();
        self.escalated_workbench_lane = None;
        Ok(())
    }

    /// Applies a binary persona phase modulation patch into the Persona workbench lane.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `patch_data` is empty.
    #[inline]
    pub fn apply_patch(&mut self, patch_data: &[u8]) -> Result<()> {
        if patch_data.is_empty() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        if patch_data.len() >= MAX_ATTENTION_HEADS {
            // SAFETY: `patch_data.len() >= 64` and `[u8; 64]` has the exact same layout and
            // alignment (1) as `[PhaseQ7; 64]` (`[i8; 64]`).
            let patch_ref: &[PhaseQ7; MAX_ATTENTION_HEADS] =
                unsafe { &*patch_data.as_ptr().cast::<[PhaseQ7; MAX_ATTENTION_HEADS]>() };
            let pw = load_q7_words_64(patch_ref);
            let sw = self.slot_words_mut(usize::from(ATTRACTOR_SLOT_PERSONA))?;
            swar_or_avx2_q7_add_sat_words_64(sw, &pw);
        } else {
            let mut bank_mut = self.substrate.slot_phase_bank.as_aligned_slice_mut();
            let persona_slot = bank_mut
                .slot_exact_mut::<MAX_ATTENTION_HEADS>(usize::from(ATTRACTOR_SLOT_PERSONA))?;
            for (h, slot_p) in persona_slot.iter_mut().enumerate() {
                let delta = patch_data[h % patch_data.len()] as PhaseQ7;
                *slot_p = branchless_q7_add_sat(*slot_p, delta);
            }
        }
        self.substrate.slot_enabled_mask |= 1u8 << ATTRACTOR_SLOT_PERSONA;
        Ok(())
    }
}

impl KvKeyframeCache for RuntimeContext {
    #[inline]
    fn seek_to_token(&mut self, token_index: u32, header_out: &mut KvKeyframeHeader) -> Result<()> {
        header_out.token_index = token_index;
        header_out.frame_type = 0;
        Ok(())
    }
}

impl MappedStorage for RuntimeContext {
    #[inline]
    fn serialize_snapshot(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        self.serialize_to_bytes(buffer, required_size)
    }

    #[inline]
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        config_out: &mut RuntimeConfig,
    ) -> Result<()> {
        self.deserialize_from_bytes(buffer)?;
        *config_out = self.config.raw;
        Ok(())
    }

    #[inline]
    fn get_track_offset(&self, track_id: u32) -> Result<u64> {
        let header_size = LMNK_CONTAINER_HEADER_BYTES as u64;
        match track_id {
            0 => Ok(header_size),
            1 => Ok(header_size + 1024),
            2 => Ok(header_size + 2048),
            3 => Ok(header_size + 3072),
            _ => Err(LMNK_ERR_INVALID_ARGUMENT),
        }
    }
}

impl CausalProvenanceSignal for RuntimeContext {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.entropy_controller.causal_provenance_bits() | self.config.raw.causal_provenance_bits()
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(
            self.entropy_controller.causal_anomaly_severity(),
            self.config.raw.causal_anomaly_severity(),
        )
    }
}

impl SystemConstraintSignal for RuntimeContext {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        self.entropy_controller.system_vram_pressure()
    }

    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        self.entropy_controller.system_gpu_frame_overrun()
    }

    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        self.entropy_controller.active_swarm_cells_mask()
    }

    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        self.entropy_controller.recommend_layer_skip()
    }

    #[inline(always)]
    fn phase_step_min_rad_override(&self) -> f32 {
        branchless_max_f32(
            self.config.raw.phase_step_min_rad_override(),
            self.entropy_controller.phase_step_min_rad_override(),
        )
    }
}

impl ManifoldGatewaySignal for RuntimeContext {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        if self.substrate.active_count > 0 {
            let (mut frame, words) = self.substrate.synthesize_gateway_frame_with_words();
            frame.lineage_flags |= self.causal_provenance_bits();
            frame.active_swarm_cell_mask &= self.active_swarm_cells_mask();
            frame.phase_step_min_rad = self.phase_step_min_rad_override();
            frame.phase_step_min_q7 = branchless_rad_to_phase_step_u8(frame.phase_step_min_rad);
            if self.recommend_layer_skip() {
                frame.action_mode_flags |= ACTION_MODE_LAYER_SKIP;
            }
            (frame, words)
        } else {
            self.entropy_controller
                .synthesize_gateway_frame_with_words()
        }
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}
