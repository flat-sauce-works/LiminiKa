// name: src/core/src/traits/gateway/kv_cache.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-RAM compressed KV-phase context caches (`RamPhaseKvCache` / `RamPhaseKvGopCache`)
//! supporting P-Frame Zero-Forward Seek and autonomous $S^{H-1}$ resonance gating.

use super::signals::*;
use super::circuit::*;
use super::*;

/// 192-byte (`#[repr(C, align(64))]`) host-RAM entry pairing a compressed 128-byte
/// [`KvKeyframePacket`] (64-head Q7 KV-phase anchor) with an 8-byte [`CognitiveTrajectoryLedger`]
/// and [`CognitiveCircuitActivation`], acting as a zero-VRAM associative context KV cache slot.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamPhaseKvEntry {
    /// Compressed 128-byte I-Frame KV-phase anchor (`offset 0..128`).
    pub keyframe: KvKeyframePacket,
    /// Cognitive trajectory active when this context chunk was crystallized (`offset 128..136`).
    pub circuit: CognitiveTrajectoryLedger,
    /// Stigmergic cognitive circuit activation recorded at storage time (`offset 136..144`).
    pub last_activation: CognitiveCircuitActivation,
    /// Deterministic 64-bit context chunk key (`0` indicates an empty slot, `offset 144..152`).
    pub context_hash: u64,
    /// Number of cache hits / associative recalls (`offset 152..156`).
    pub reuse_count: u32,
    /// Monotonic logical epoch of the most recent store or hit (`offset 156..160`).
    pub last_used_epoch: u32,
    /// Reserved zero padding aligning the entry to 3 cachelines (`192` bytes, `offset 160..192`).
    pub reserved_pad: [u8; 32],
}

impl Default for RamPhaseKvEntry {
    #[inline(always)]
    fn default() -> Self {
        Self {
            keyframe: KvKeyframePacket::default(),
            circuit: CognitiveTrajectoryLedger::new(),
            last_activation: CognitiveCircuitActivation::default(),
            context_hash: 0,
            reuse_count: 0,
            last_used_epoch: 0,
            reserved_pad: [0u8; 32],
        }
    }
}

impl RamPhaseKvEntry {
    /// Returns `true` when this slot holds a populated context keyframe (`context_hash != 0`).
    #[inline(always)]
    #[must_use]
    pub const fn is_occupied(&self) -> bool {
        self.context_hash != 0
    }

    /// Returns a copy of this entry whose keyframe phases and token index reflect a Zero-Forward
    /// Seek reconstruction.
    #[inline]
    #[must_use]
    pub fn with_reconstructed_phases(
        &self,
        restored_phases: [gcso_q7_t; MAX_ATTENTION_HEADS],
        token_shift: u32,
        r_bar: f32,
    ) -> Self {
        Self {
            keyframe: KvKeyframePacket::new(
                self.keyframe.header.token_index.wrapping_add(token_shift),
                self.keyframe.header.gop_length,
                r_bar,
                1.0 + r_bar * 4.0,
                0.0,
                self.keyframe.provenance_bits(),
                self.keyframe.cluster_id(),
                self.keyframe.slot_id(),
                restored_phases,
            ),
            ..*self
        }
    }
}

impl RawPhaseProjectable for RamPhaseKvEntry {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.keyframe.anchor_phases
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        load_q7_words_64(&self.keyframe.anchor_phases)
    }
}

impl CausalProvenanceSignal for RamPhaseKvEntry {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.keyframe.provenance_bits() | self.circuit.causal_provenance_bits()
    }
}

impl CognitiveGroundingSignal for RamPhaseKvEntry {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.keyframe.anchor_phases
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        (1.0 + self.keyframe.header.composite_vector_length).clamp(0.5, 4.0)
    }

    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.keyframe.cluster_id()
    }
}

impl ObjectiveFactSignal for RamPhaseKvEntry {}

impl ManifoldGatewaySignal for RamPhaseKvEntry {
    #[inline]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        (self, &self.circuit).synthesize_gateway_frame()
    }
}

/// Trait governing host-RAM-resident compressed KV-phase caching (`KvKeyframePacket` +
/// `CognitiveTrajectoryLedger`), allowing CPU/RAM capacity to absorb long-context memory
/// without expanding GPU VRAM or triggering disk I/O.
pub trait RamPhaseKvCacheOps {
    /// Returns an immutable slice over all [`RamPhaseKvEntry`] cache slots.
    fn entries(&self) -> &[RamPhaseKvEntry];

    /// Returns a mutable slice over all [`RamPhaseKvEntry`] cache slots.
    fn entries_mut(&mut self) -> &mut [RamPhaseKvEntry];

    /// Stores or updates a compressed 64-head KV-phase keyframe and its cognitive circuit in RAM,
    /// evicting the least-recently-used (`last_used_epoch`) slot when full.
    #[inline]
    fn store_context_keyframe(
        &mut self,
        context_hash: u64,
        token_index: u32,
        anchor_id: u32,
        phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        circuit: CognitiveTrajectoryLedger,
        activation: CognitiveCircuitActivation,
        epoch: u32,
    ) {
        if context_hash == 0 {
            return;
        }
        let entries = self.entries_mut();
        if entries.is_empty() {
            return;
        }
        let words = load_q7_words_64(phases);
        let mut norm_approx = 0u16;
        let mut c = 0usize;
        while c < NUM_STIGMERGIC_GRANULES {
            norm_approx += u16::from(swar_mean_u8_le128_x8(swar_q7_abs_u8_x8(words[c])));
            c += 1;
        }
        let r_bar_approx = ((norm_approx as f32) / (8.0 * 127.0)).clamp(0.0, 1.0);
        let keyframe = KvKeyframePacket::new(
            token_index,
            1,
            r_bar_approx,
            1.0 + r_bar_approx * 4.0,
            0.0,
            CAUSAL_AXIS_OBJECTIVE_FACT | circuit.causal_provenance_bits(),
            anchor_id,
            ATTRACTOR_SLOT_KNOWLEDGE,
            *phases,
        );

        let mut target_idx = 0usize;
        let mut oldest_epoch = u32::MAX;
        let mut found_exact_or_empty = false;
        let mut i = 0usize;
        while i < entries.len() {
            if entries[i].context_hash == context_hash {
                target_idx = i;
                found_exact_or_empty = true;
                break;
            }
            if !entries[i].is_occupied() {
                target_idx = i;
                found_exact_or_empty = true;
                break;
            }
            if entries[i].last_used_epoch < oldest_epoch {
                oldest_epoch = entries[i].last_used_epoch;
                target_idx = i;
            }
            i += 1;
        }
        let prev_reuse = if found_exact_or_empty && entries[target_idx].context_hash == context_hash
        {
            entries[target_idx].reuse_count.saturating_add(1)
        } else {
            0
        };
        entries[target_idx] = RamPhaseKvEntry {
            keyframe,
            circuit,
            last_activation: activation,
            context_hash,
            reuse_count: prev_reuse,
            last_used_epoch: epoch,
            reserved_pad: [0u8; 32],
        };
    }

    /// Looks up an exact `context_hash` match (`O(N)` cacheline scan) or falls back to the highest
    /// active-phase resonance slot when `min_resonance_q7` is met or exceeded, updating LRU metadata
    /// on hit.
    ///
    /// Resonance combines scale-invariant tangent cosine similarity
    /// (`tangent_cosine_active_heads_64`) with un-truncated SWAR circular proximity across all 64
    /// lanes, mapped into `[-127, 127]`. All-zero query or entry phase vectors yield `0` resonance
    /// so ungrounded queries do not falsely alias occupied slots.
    ///
    /// Returns `Some((slot_index, exact_hit, resonance_q7))`.
    #[inline]
    fn lookup_context_or_resonate(
        &mut self,
        context_hash: u64,
        query_words: &[u64; NUM_STIGMERGIC_GRANULES],
        min_resonance_q7: gcso_q7_t,
        epoch: u32,
    ) -> Option<(usize, bool, gcso_q7_t)> {
        let entries = self.entries_mut();
        if entries.is_empty() {
            return None;
        }
        let query_nonzero = swar_words_nonzero_64(query_words);
        let mut best_idx: Option<usize> = None;
        let mut best_res: gcso_q7_t = i8::MIN;
        let mut i = 0usize;
        while i < entries.len() {
            if !entries[i].is_occupied() {
                i += 1;
                continue;
            }
            if context_hash != 0 && entries[i].context_hash == context_hash {
                entries[i].reuse_count = entries[i].reuse_count.saturating_add(1);
                entries[i].last_used_epoch = epoch;
                return Some((i, true, 127));
            }
            if query_nonzero {
                let e_words = load_q7_words_64(&entries[i].keyframe.anchor_phases);
                let res_q7 = if swar_words_nonzero_64(&e_words) {
                    let cos = tangent_cosine_active_heads_64(
                        query_words,
                        &e_words,
                        MAX_ATTENTION_HEADS as u16,
                    );
                    let mut sum_diff = 0u32;
                    let mut c = 0usize;
                    while c < NUM_STIGMERGIC_GRANULES {
                        let abs_w =
                            swar_q7_abs_u8_x8(swar_q7_circular_diff_x8(query_words[c], e_words[c]));
                        let even = abs_w & 0x00FF_00FF_00FF_00FF;
                        let odd = (abs_w >> 8) & 0x00FF_00FF_00FF_00FF;
                        let cell_sum = ((even.wrapping_add(odd))
                            .wrapping_mul(0x0001_0001_0001_0001)
                            >> 48) as u32;
                        sum_diff += cell_sum;
                        c += 1;
                    }
                    let prox =
                        branchless_clamp_f32(1.0 - (sum_diff as f32) * (1.0 / 8192.0), 0.0, 1.0);
                    let scaled = branchless_select_f32(cos > 0.0, cos * prox, cos) * 127.0;
                    let rounded = if scaled >= 0.0 {
                        scaled + 0.5
                    } else {
                        scaled - 0.5
                    };
                    (rounded as i32).clamp(-127, 127) as gcso_q7_t
                } else {
                    0
                };
                if res_q7 >= min_resonance_q7 && res_q7 > best_res {
                    best_res = res_q7;
                    best_idx = Some(i);
                }
            }
            i += 1;
        }
        if let Some(idx) = best_idx {
            entries[idx].reuse_count = entries[idx].reuse_count.saturating_add(1);
            entries[idx].last_used_epoch = epoch;
            return Some((idx, false, best_res));
        }
        None
    }

    /// Encodes a 64-byte P-Frame [`KvMotionDeltaPacket`] relative to the keyframe stored in
    /// `slot_idx` and `curr_phases`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_idx` is out of bounds or unoccupied, or
    /// `qdps_min_step_q7 > 127`. Also propagates [`KvKeyframePacket::verify_integrity`] errors.
    #[inline]
    fn encode_slot_pframe_delta(
        &self,
        slot_idx: usize,
        curr_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        qdps_min_step_q7: u8,
    ) -> GcsoResult<KvMotionDeltaPacket> {
        let entries = self.entries();
        let entry = entries.get(slot_idx).ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if !entry.is_occupied() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        entry.keyframe.verify_integrity()?;
        encode_pframe_motion_delta_64(&entry.keyframe.anchor_phases, curr_phases, qdps_min_step_q7)
    }

    /// Executes Zero-Forward Seek on `slot_idx` by replaying `pframes[..seek_steps]` onto the
    /// slot's verified I-Frame [`KvKeyframePacket`], updating LRU metadata and returning
    /// `(restored_phases, joint_log_prob, r_bar, repulsed_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_idx` is out of bounds / unoccupied,
    /// `seek_steps > pframes.len()`, `qdps_min_step_q7 > 127`, or `contradiction_threshold` is
    /// non-finite; propagates [`KvKeyframePacket::verify_integrity`] errors on corrupted frames.
    #[inline]
    fn seek_slot_with_pframes(
        &mut self,
        slot_idx: usize,
        pframes: &[KvMotionDeltaPacket],
        seek_steps: usize,
        qdps_min_step_q7: u8,
        contradiction_threshold: f32,
        epoch: u32,
    ) -> GcsoResult<([gcso_q7_t; MAX_ATTENTION_HEADS], f32, f32, u32)> {
        let entries = self.entries_mut();
        let entry = entries
            .get_mut(slot_idx)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if !entry.is_occupied() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        entry.keyframe.verify_integrity()?;
        let (filt_w, clean_lp, repulsed_count) = seek_pframe_chain_words_64(
            &entry.keyframe.anchor_phases,
            pframes,
            seek_steps,
            qdps_min_step_q7,
            contradiction_threshold,
        )?;
        let (r_bar, _mean_q7, _kappa_c) = circular_resultant_from_words_64(&filt_w);
        let mut restored_phases = [0i8; MAX_ATTENTION_HEADS];
        store_q7_words_64(&mut restored_phases, &filt_w);
        entry.reuse_count = entry.reuse_count.saturating_add(1);
        entry.last_used_epoch = epoch;
        Ok((restored_phases, clean_lp, r_bar, repulsed_count))
    }

    /// Reconstructs a 128-byte [`GatewayFrame`] by Zero-Forward Seeking `slot_idx` through
    /// `pframes[..seek_steps]` and composing the reconstructed KV-phase state with the slot's
    /// stored [`CognitiveTrajectoryLedger`].
    ///
    /// # Errors
    /// Propagates errors from [`RamPhaseKvCacheOps::seek_slot_with_pframes`].
    #[inline]
    fn synthesize_seeked_gateway_frame(
        &mut self,
        slot_idx: usize,
        pframes: &[KvMotionDeltaPacket],
        seek_steps: usize,
        qdps_min_step_q7: u8,
        contradiction_threshold: f32,
        epoch: u32,
    ) -> GcsoResult<GatewayFrame> {
        let (restored_phases, _clean_lp, r_bar, _repulsed_count) = self.seek_slot_with_pframes(
            slot_idx,
            pframes,
            seek_steps,
            qdps_min_step_q7,
            contradiction_threshold,
            epoch,
        )?;
        let entry = &self.entries()[slot_idx];
        Ok(entry
            .with_reconstructed_phases(restored_phases, seek_steps as u32, r_bar)
            .synthesize_gateway_frame())
    }
}

impl RamPhaseKvCacheOps for [RamPhaseKvEntry] {
    #[inline(always)]
    fn entries(&self) -> &[RamPhaseKvEntry] {
        self
    }

    #[inline(always)]
    fn entries_mut(&mut self) -> &mut [RamPhaseKvEntry] {
        self
    }
}

/// Fixed-capacity, cacheline-aligned host-RAM compressed KV-phase cache (`192 * N` bytes).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamPhaseKvCache<const N: usize> {
    /// Contiguous array of 192-byte compressed KV-phase + cognitive circuit slots.
    pub slots: [RamPhaseKvEntry; N],
}

impl<const N: usize> Default for RamPhaseKvCache<N> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> RamPhaseKvCache<N> {
    /// Creates an empty [`RamPhaseKvCache`] of `N` slots (`192 * N` bytes).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: [RamPhaseKvEntry {
                keyframe: KvKeyframePacket {
                    header: crate::abi::gcso_kv_keyframe_header_t {
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
                },
                circuit: CognitiveTrajectoryLedger::new(),
                last_activation: CognitiveCircuitActivation {
                    peel_gate_q7: 0,
                    fuse_gate_q7: 0,
                    repulse_sensitivity_q7: 0,
                    selected_op: 0,
                    preferred_next_scale: 0,
                    intensity_q7: 0,
                    circuit_resonance_q7: 0,
                    trajectory_depth: 0,
                },
                context_hash: 0,
                reuse_count: 0,
                last_used_epoch: 0,
                reserved_pad: [0u8; 32],
            }; N],
        }
    }
}

impl<const N: usize> RamPhaseKvCacheOps for RamPhaseKvCache<N> {
    #[inline(always)]
    fn entries(&self) -> &[RamPhaseKvEntry] {
        &self.slots
    }

    #[inline(always)]
    fn entries_mut(&mut self) -> &mut [RamPhaseKvEntry] {
        &mut self.slots
    }
}

/// Low-spec PC preset: 16 slots = 3,072 bytes (~3 KB, fits inside L1D alongside LUTs).
pub type LowSpecPhaseKvCache = RamPhaseKvCache<16>;

/// Standard PC preset: 64 slots = 12,288 bytes (12 KB, comfortably inside L1D/L2).
pub type StandardPhaseKvCache = RamPhaseKvCache<64>;

/// High-capacity RAM / Zen 4 L2 preset: 256 slots = 49,152 bytes (48 KB, inside L2 1 MB).
pub type HighCapPhaseKvCache = RamPhaseKvCache<256>;

/// Host-RAM compressed KV-phase GOP cache pairing `N` 192-byte I-Frame slots
/// ([`RamPhaseKvCache<N>`]) with per-slot fixed-capacity rings of `P` 64-byte P-Frame motion
/// delta packets ([`KvMotionDeltaPacket`]), enabling Zero-Forward Seek context replay across
/// extended multi-turn histories with zero heap allocation.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamPhaseKvGopCache<const N: usize, const P: usize> {
    /// Base I-Frame + cognitive circuit cache slots.
    pub base: RamPhaseKvCache<N>,
    /// Per-slot ring buffer of 64-byte P-Frame phase-motion deltas.
    pub pframes: [[KvMotionDeltaPacket; P]; N],
    /// Last reconstructed/stored 64-head phase per slot (used to encode incremental P-Frame steps).
    pub last_phases: [[gcso_q7_t; MAX_ATTENTION_HEADS]; N],
    /// Number of valid P-Frames currently stored in each slot's ring (`<= P`).
    pub pframe_counts: [u8; N],
    /// Next write index in each slot's P-Frame ring (`< P`).
    pub pframe_heads: [u8; N],
}

impl<const N: usize, const P: usize> Default for RamPhaseKvGopCache<N, P> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize, const P: usize> RamPhaseKvGopCache<N, P> {
    /// Creates an empty [`RamPhaseKvGopCache`] with `N` I-Frame slots and `P` P-Frames per slot.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            base: RamPhaseKvCache::new(),
            pframes: [[KvMotionDeltaPacket {
                motion_deltas: [0; MAX_ATTENTION_HEADS],
            }; P]; N],
            last_phases: [[0; MAX_ATTENTION_HEADS]; N],
            pframe_counts: [0; N],
            pframe_heads: [0; N],
        }
    }

    /// Stores or updates a context I-Frame keyframe, resets that slot's P-Frame ring, and returns
    /// the slot index when `context_hash != 0` and `N > 0`.
    #[inline]
    pub fn store_keyframe_and_reset_gop(
        &mut self,
        context_hash: u64,
        token_index: u32,
        anchor_id: u32,
        phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        circuit: CognitiveTrajectoryLedger,
        activation: CognitiveCircuitActivation,
        epoch: u32,
    ) -> Option<usize> {
        if context_hash == 0 || N == 0 {
            return None;
        }
        self.base.store_context_keyframe(
            context_hash,
            token_index,
            anchor_id,
            phases,
            circuit,
            activation,
            epoch,
        );
        let mut idx = 0usize;
        while idx < N {
            if self.base.slots[idx].context_hash == context_hash {
                self.last_phases[idx] = *phases;
                self.pframe_counts[idx] = 0;
                self.pframe_heads[idx] = 0;
                return Some(idx);
            }
            idx += 1;
        }
        None
    }

    /// Encodes and appends a 64-byte P-Frame [`KvMotionDeltaPacket`] from the slot's previous
    /// phase state to `curr_phases`, updating the slot's LRU epoch.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_idx >= N`, the slot is unoccupied,
    /// `P == 0`, or `qdps_min_step_q7 > 127`.
    #[inline]
    pub fn append_context_pframe(
        &mut self,
        slot_idx: usize,
        curr_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        qdps_min_step_q7: u8,
        epoch: u32,
    ) -> GcsoResult<()> {
        if slot_idx >= N || P == 0 || !self.base.slots[slot_idx].is_occupied() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let delta = encode_pframe_motion_delta_64(
            &self.last_phases[slot_idx],
            curr_phases,
            qdps_min_step_q7,
        )?;
        let head = usize::from(self.pframe_heads[slot_idx]) % P;
        self.pframes[slot_idx][head] = delta;
        self.pframe_heads[slot_idx] = ((head + 1) % P) as u8;
        let count = usize::from(self.pframe_counts[slot_idx]);
        if count < P {
            self.pframe_counts[slot_idx] = (count + 1) as u8;
        }
        self.last_phases[slot_idx] = *curr_phases;
        self.base.slots[slot_idx].last_used_epoch = epoch;
        Ok(())
    }

    /// Returns the number of valid P-Frames stored for `slot_idx`.
    #[inline(always)]
    #[must_use]
    pub fn slot_pframe_count(&self, slot_idx: usize) -> usize {
        if slot_idx >= N {
            0
        } else {
            usize::from(self.pframe_counts[slot_idx])
        }
    }

    /// Returns the total number of valid P-Frames across all occupied slots.
    #[inline]
    #[must_use]
    pub fn total_stored_pframes(&self) -> usize {
        let mut total = 0usize;
        let mut i = 0usize;
        while i < N {
            if self.base.slots[i].is_occupied() {
                total += usize::from(self.pframe_counts[i]);
            }
            i += 1;
        }
        total
    }

    /// Collects the slot's P-Frame ring in chronological order and runs Zero-Forward Seek
    /// reconstruction onto `(restored_phases, joint_log_prob, r_bar, repulsed_count)`.
    ///
    /// # Errors
    /// Propagates errors from [`RamPhaseKvCacheOps::seek_slot_with_pframes`].
    #[inline]
    pub fn seek_slot_history(
        &mut self,
        slot_idx: usize,
        qdps_min_step_q7: u8,
        contradiction_threshold: f32,
        epoch: u32,
    ) -> GcsoResult<([gcso_q7_t; MAX_ATTENTION_HEADS], f32, f32, u32)> {
        if slot_idx >= N || !self.base.slots[slot_idx].is_occupied() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let count = usize::from(self.pframe_counts[slot_idx]).min(P);
        let mut ordered = [KvMotionDeltaPacket {
            motion_deltas: [0; MAX_ATTENTION_HEADS],
        }; P];
        if count > 0 && P > 0 {
            let head = usize::from(self.pframe_heads[slot_idx]) % P;
            let start = (head + P - count) % P;
            let mut k = 0usize;
            while k < count {
                ordered[k] = self.pframes[slot_idx][(start + k) % P];
                k += 1;
            }
        }
        self.seek_slot_with_pframes(
            slot_idx,
            &ordered[..count],
            count,
            qdps_min_step_q7,
            contradiction_threshold,
            epoch,
        )
    }

    /// Reconstructs a 128-byte [`GatewayFrame`] from `slot_idx`'s I-Frame and accumulated P-Frame
    /// ring history without running an LLM forward pass.
    ///
    /// # Errors
    /// Propagates errors from [`RamPhaseKvGopCache::seek_slot_history`].
    #[inline]
    pub fn synthesize_slot_history_gateway_frame(
        &mut self,
        slot_idx: usize,
        qdps_min_step_q7: u8,
        contradiction_threshold: f32,
        epoch: u32,
    ) -> GcsoResult<GatewayFrame> {
        let (restored_phases, _clean_lp, r_bar, _repulsed) = self.seek_slot_history(
            slot_idx,
            qdps_min_step_q7,
            contradiction_threshold,
            epoch,
        )?;
        let count = self.slot_pframe_count(slot_idx);
        let entry = &self.base.slots[slot_idx];
        Ok(entry
            .with_reconstructed_phases(restored_phases, count as u32, r_bar)
            .synthesize_gateway_frame())
    }

    /// Returns the number of currently occupied I-Frame slots (`<= N`).
    #[inline]
    #[must_use]
    pub fn occupied_slot_count(&self) -> usize {
        let mut count = 0usize;
        let mut i = 0usize;
        while i < N {
            if self.base.slots[i].is_occupied() {
                count += 1;
            }
            i += 1;
        }
        count
    }

    /// Executes one full autonomous GOP KV-cache step (void gate, $S^{H-1}$ resonance lookup,
    /// Zero-Forward Seek replay, cognitive circuit composition, and P-Frame/I-Frame update)
    /// deriving all gating thresholds from `policy` and `frame` invariants.
    ///
    /// # Errors
    /// Propagates [`GCSO_ERROR_INVALID_ARGUMENT`] or keyframe CRC errors from slot seek/append.
    #[inline]
    pub fn step_with_gateway_autonomous<Pol: CognitiveCircuitThresholdPolicy + ?Sized>(
        &mut self,
        context_hash: u64,
        token_index: u32,
        frame: &GatewayFrame,
        circuit: &mut CognitiveTrajectoryLedger,
        activation: CognitiveCircuitActivation,
        policy: &Pol,
        epoch: u32,
    ) -> GcsoResult<RamPhaseKvStepOutcome> {
        self.step_with_gateway_and_override(
            context_hash,
            token_index,
            frame,
            circuit,
            activation,
            policy,
            None,
            epoch,
        )
    }

    /// Executes [`Self::step_with_gateway_autonomous`] with an optional diagnostic
    /// `override_min_resonance_q7` (when `None`, derives the $2 / \sqrt{H}$ floor from `policy`).
    ///
    /// # Errors
    /// Propagates [`GCSO_ERROR_INVALID_ARGUMENT`] or keyframe CRC errors from slot seek/append.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_gateway_and_override<Pol: CognitiveCircuitThresholdPolicy + ?Sized>(
        &mut self,
        context_hash: u64,
        token_index: u32,
        frame: &GatewayFrame,
        circuit: &mut CognitiveTrajectoryLedger,
        activation: CognitiveCircuitActivation,
        policy: &Pol,
        override_min_resonance_q7: Option<gcso_q7_t>,
        epoch: u32,
    ) -> GcsoResult<RamPhaseKvStepOutcome> {
        let ungrounded_void =
            context_hash == 0 && (frame.provenance_bits & CAUSAL_AXIS_OBJECTIVE_FACT) == 0;
        if ungrounded_void {
            return Ok(RamPhaseKvStepOutcome::SkippedVoid);
        }
        let occupied = self.occupied_slot_count();
        let min_resonance_q7 = override_min_resonance_q7.unwrap_or_else(|| {
            policy.kv_resonance_floor_for_heads_q7(frame.active_head_count, occupied, N)
        });
        if let Some((slot_idx, exact_hit, resonance_q7)) = self.lookup_context_or_resonate(
            context_hash,
            frame.phase_words_ref(),
            min_resonance_q7,
            epoch,
        ) {
            let qdps_step = frame.qdps_min_step_q7.max(1);
            let mut seek_r_bar = 1.0_f32;
            let mut seek_repulsed = 0_u32;
            let mut seek_executed = false;
            if self.slot_pframe_count(slot_idx) > 0 {
                let (_restored, _lp, r_bar, repulsed) = self.seek_slot_history(
                    slot_idx,
                    qdps_step,
                    policy.kv_contradiction_threshold(),
                    epoch,
                )?;
                seek_r_bar = r_bar;
                seek_repulsed = repulsed;
                seek_executed = true;
            }
            let cached_circuit = self.entries()[slot_idx].circuit;
            if cached_circuit.latest_step().is_some() {
                let blend_q7 = policy.kv_circuit_blend_q7(frame);
                *circuit = cached_circuit.compose_persona_circuit(circuit, blend_q7);
            }
            self.append_context_pframe(slot_idx, &frame.phase_deltas, qdps_step, epoch)?;
            let pframe_count = self.slot_pframe_count(slot_idx);
            return Ok(RamPhaseKvStepOutcome::ResonatedSeeked {
                slot_idx,
                exact_hit,
                resonance_q7,
                min_resonance_q7,
                pframe_count,
                seek_r_bar,
                seek_repulsed,
                seek_executed,
            });
        }
        if context_hash != 0 {
            if let Some(slot_idx) = self.store_keyframe_and_reset_gop(
                context_hash,
                token_index,
                frame.target_anchor_id,
                &frame.phase_deltas,
                *circuit,
                activation,
                epoch,
            ) {
                return Ok(RamPhaseKvStepOutcome::StoredKeyframe {
                    slot_idx,
                    min_resonance_q7,
                });
            }
        }
        Ok(RamPhaseKvStepOutcome::Unbacked)
    }
}

/// Outcome returned by [`RamPhaseKvGopCache::step_with_gateway_autonomous`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RamPhaseKvStepOutcome {
    /// Query carried neither a context fingerprint nor factual grounding (`Liminal`).
    SkippedVoid,
    /// No existing slot resonated above the autonomous threshold; stored a new I-Frame slot.
    StoredKeyframe {
        /// Allocated slot index.
        slot_idx: usize,
        /// Autonomous or overridden resonance floor applied during lookup.
        min_resonance_q7: gcso_q7_t,
    },
    /// Matched an existing I-Frame slot by exact hash or phase resonance, replayed P-Frames, and appended a delta.
    ResonatedSeeked {
        /// Matched slot index.
        slot_idx: usize,
        /// True on exact `context_hash` match.
        exact_hit: bool,
        /// Resonance score (`127` on exact match).
        resonance_q7: gcso_q7_t,
        /// Autonomous or overridden resonance floor applied during lookup.
        min_resonance_q7: gcso_q7_t,
        /// Total valid P-Frames now in `slot_idx`.
        pframe_count: usize,
        /// Circular resultant concentration $\bar{R}$ from Zero-Forward Seek (`1.0` if no prior P-Frames).
        seek_r_bar: f32,
        /// Count of repulsed P-Frame transitions during Zero-Forward Seek.
        seek_repulsed: u32,
        /// True when prior P-Frames were replayed via Zero-Forward Seek.
        seek_executed: bool,
    },
    /// Cache capacity is zero or `context_hash == 0` prevented keyframe storage.
    Unbacked,
}

impl<const N: usize, const P: usize> RamPhaseKvCacheOps for RamPhaseKvGopCache<N, P> {
    #[inline(always)]
    fn entries(&self) -> &[RamPhaseKvEntry] {
        self.base.entries()
    }

    #[inline(always)]
    fn entries_mut(&mut self) -> &mut [RamPhaseKvEntry] {
        self.base.entries_mut()
    }
}

/// Low-spec PC GOP cache preset: 16 I-Frame slots × 4 P-Frames/slot (~8 KB, fits in L1D).
pub type LowSpecPhaseKvGopCache = RamPhaseKvGopCache<16, 4>;

/// Standard PC GOP cache preset: 64 I-Frame slots × 8 P-Frames/slot (~48 KB, fits in L2).
pub type StandardPhaseKvGopCache = RamPhaseKvGopCache<64, 8>;

/// High-capacity RAM / Zen 4 L2 GOP cache preset: 256 I-Frame slots × 8 P-Frames/slot (~192 KB).
pub type HighCapPhaseKvGopCache = RamPhaseKvGopCache<256, 8>;

impl RawPhaseProjectable for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        words_to_q7_array_64(&self.to_phase_words_64())
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.to_phase_words_64()
    }
}

impl CausalProvenanceSignal for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let (hist, depth) = self.op_counts();
        if depth == 0 {
            return 0;
        }
        let repulse_active = hist[CognitiveOpKind::RepulseAudit as usize]
            > hist[CognitiveOpKind::SynthesizeFuse as usize];
        CAUSAL_AXIS_SUBJECTIVE_PERSONA
            | branchless_select_u32(repulse_active, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
    }
}

impl CognitiveGroundingSignal for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.project_to_q7_64()
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.to_phase_words_64()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        let (_hist, depth) = self.op_counts();
        (depth as f32) * 0.25
    }
}

impl SubjectivePersonaSignal for CognitiveTrajectoryLedger {}

impl ManifoldGatewaySignal for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let words = self.to_phase_words_64();
        let (hist, depth) = self.op_counts();
        if depth == 0 {
            return (GatewayFrame::default(), [0u64; NUM_STIGMERGIC_GRANULES]);
        }
        let tag_q7 = fast_mean_q7_tag_from_words_64(&words);
        let peel_dominates =
            hist[CognitiveOpKind::DecomposePeel as usize] > hist[CognitiveOpKind::SynthesizeFuse as usize];
        let pspm_fact = branchless_select_u8(peel_dominates, 32, 20);
        let pspm_logic = branchless_select_u8(peel_dominates, 20, 24);
        let pspm_explore = MAX_ATTENTION_HEADS as u8 - pspm_fact - pspm_logic;
        let action_mode = if peel_dominates {
            ACTION_MODE_DPSR_STEER | ACTION_MODE_SUBHEAD_ROUTE
        } else {
            ACTION_PRESET_EXPRESS_PHASE
        };
        let packed_tag = pack_action_slot_tag(ATTRACTOR_SLOT_PERSONA, action_mode);
        let prov = self.causal_provenance_bits();
        (
            GatewayFrame {
                tagged_ptr: gcso_tagged_ptr_t::pack(0, packed_tag, tag_q7).raw(),
                effective_energy: (depth as f32) * 0.25,
                provenance_bits: prov,
                dominant_slot: ATTRACTOR_SLOT_PERSONA,
                action_mode_flags: action_mode,
                active_swarm_cell_mask: 0xFF,
                subhead_fact_heads: pspm_fact,
                subhead_logic_heads: pspm_logic,
                subhead_explore_heads: pspm_explore,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas: words_to_q7_array_64(&words),
                ..GatewayFrame::default()
            },
            words,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

