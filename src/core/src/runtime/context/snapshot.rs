// name: src/core/src/runtime/context/snapshot.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified `.lmnk` binary snapshot serialization and deserialization for [`RuntimeContext`].

use super::{
    CoreStateStack, LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES, Result,
    RuntimeContext,
};
use crate::drivers::{
    EntropyController, EntropyState, HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
    SPINE_BYPASS_VETO_KNOWN_MASK, SPINE_CONTROL_KNOWN_FLAGS, SPINE_ENTROPY_Q15_ONE,
    SystemSpineState,
};
use crate::foundation::{Pod, RopePairLayout, branchless_is_finite_f32};
use crate::liminal::{
    LMNK_ERR_BUFFER_TOO_SMALL, LMNK_ERR_CONTAINER_CORRUPTED, MAX_ATTENTION_HEADS,
};
use crate::runtime::storage::{
    LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT, LMNK_SECTION_RUNTIME_SNAPSHOT, LmnkContainerView,
    LmnkSectionInput, write_container_sections_to_bytes,
};
use crate::stigmergy::{DEFAULT_SUBSTRATE_CAPACITY, LedgerSlot, PhaseSteeringEngine};

impl RuntimeContext {
    /// Serializes active runtime context state into a unified `.lmnk` binary snapshot container
    /// conforming to [`crate::liminal::LMNK_ABI_VERSION_HEX`] (`512` bytes total).
    ///
    /// Supports two-pass sizing: passing an empty `buffer` writes the required byte size
    /// ([`LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES`] = `512` bytes) into `*required_size` and returns `Ok(())`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `buffer` is non-empty and smaller than `512` bytes.
    #[inline]
    pub fn serialize_to_bytes(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        let total_required = LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES;
        *required_size = total_required;
        if buffer.is_empty() {
            return Ok(());
        }
        if buffer.len() < total_required {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }

        let mut payload = [0u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES];
        payload[0..8].copy_from_slice(&self.substrate.step_counter.to_le_bytes());
        payload[8..12].copy_from_slice(&self.substrate.active_count.to_le_bytes());
        payload[12..16]
            .copy_from_slice(&u32::from(self.substrate.repulsion_threshold).to_le_bytes());
        payload[16..20].copy_from_slice(&self.substrate.last_entropy_h_tilde.to_le_bytes());
        payload[20] = self.substrate.slot_enabled_mask;
        payload[21] = self.substrate.slot_pinned_mask;
        payload[22] = self.substrate.slot_repulsion_mask;
        payload[23] = self.substrate.slot_sandbox_mask;
        payload[24..32].copy_from_slice(&self.substrate.coupling_matrix_8x8.to_le_bytes());
        for (idx, &w) in self.substrate.slot_weights.iter().enumerate() {
            let off = 32 + idx * 4;
            payload[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }

        self.entropy_controller
            .state()
            .write_unaligned_to(&mut payload, 64)?;

        payload[96..100].copy_from_slice(&self.substrate.last_selected_trail_id.to_le_bytes());
        payload[100..104].copy_from_slice(&self.config.raw.head_dim.to_le_bytes());
        payload[104..108].copy_from_slice(&self.config.raw.num_heads.to_le_bytes());
        payload[108..112].copy_from_slice(&self.config.raw.rotary_dim.to_le_bytes());
        payload[112] = self.config.raw.rope_layout;

        self.substrate
            .ledger_slot
            .write_unaligned_to(&mut payload, 128)?;
        self.core_layers.write_unaligned_to(&mut payload, 192)?;
        self.system_spine.write_unaligned_to(&mut payload, 320)?;

        let sections = [LmnkSectionInput {
            kind: LMNK_SECTION_RUNTIME_SNAPSHOT,
            flags: 0,
            payload: &payload,
        }];
        let written = write_container_sections_to_bytes(
            LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT,
            1_774_900_000,
            &sections,
            buffer,
        )?;
        *required_size = written;
        Ok(())
    }

    /// Enforces a minimum host-pinned compute mask floor that external containers cannot clear.
    #[inline(always)]
    pub const fn apply_host_spine_pin_floor(&mut self, required_pinned_mask: u64) {
        self.system_spine.apply_host_pin_floor(required_pinned_mask);
    }

    /// Deserializes a unified `.lmnk` binary snapshot container to restore context state in-place
    /// with full header, topology, and IEEE 802.3 CRC32 verification while preserving host-locked
    /// `SystemSpineState` privilege floors.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `buffer` is shorter than `64` bytes,
    /// [`LMNK_ERR_CONTAINER_CORRUPTED`] if container structure or CRC32 checks fail,
    /// [`crate::liminal::LMNK_ERR_VERSION_MISMATCH`] if the ABI version mismatches, or
    /// [`crate::liminal::LMNK_ERR_TOPOLOGY_MISMATCH`] if the topology CRC32 mismatches.
    #[inline]
    pub fn deserialize_from_bytes(&mut self, buffer: &[u8]) -> Result<()> {
        let view = LmnkContainerView::parse(buffer)?;
        if (view.header().container_flags & LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT) == 0 {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        let payload = view
            .section_bytes(LMNK_SECTION_RUNTIME_SNAPSHOT)
            .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
        if payload.len() < 384 {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        let read_u32 = |off: usize| -> Result<u32> {
            let s = payload
                .get(off..off + 4)
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            let arr = <[u8; 4]>::try_from(s).map_err(|_| LMNK_ERR_CONTAINER_CORRUPTED)?;
            Ok(u32::from_le_bytes(arr))
        };
        let read_u64 = |off: usize| -> Result<u64> {
            let s = payload
                .get(off..off + 8)
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            let arr = <[u8; 8]>::try_from(s).map_err(|_| LMNK_ERR_CONTAINER_CORRUPTED)?;
            Ok(u64::from_le_bytes(arr))
        };
        let read_u8 = |off: usize| -> Result<u8> {
            payload
                .get(off)
                .copied()
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)
        };

        let host_baseline_compute_pin_floor = self.system_spine.baseline_pinned_compute_mask;
        let host_aperture_pin_floor = self.system_spine.pinned_aperture_mask;
        let host_slot_pin_floor = self.system_spine.pinned_slot_mask;
        let host_mode_pin_floor = self.system_spine.pinned_action_mode_mask;
        let host_substrate_slot_pin_floor = self.substrate.slot_pinned_mask;

        // ------------------------------------------------------------------
        // Step 1: Transactional parsing and validation (zero mutation on Err)
        // ------------------------------------------------------------------
        let saved_step_counter = read_u64(0)?;
        let saved_active_count = read_u32(8)?;
        let saved_repulsion_threshold = read_u32(12)?;
        let restored_entropy = f32::from_bits(read_u32(16)?);
        let saved_slot_enabled_mask = read_u8(20)?;
        let saved_slot_pinned_mask = read_u8(21)?;
        let saved_slot_repulsion_mask = read_u8(22)?;
        let saved_slot_sandbox_mask = read_u8(23)?;
        let saved_coupling = read_u64(24)?;

        let mut saved_slot_weights = [0.0f32; 8];
        for (idx, dst) in saved_slot_weights.iter_mut().enumerate() {
            let w = f32::from_bits(read_u32(32 + idx * 4)?);
            if !branchless_is_finite_f32(w) {
                return Err(LMNK_ERR_CONTAINER_CORRUPTED);
            }
            *dst = w;
        }

        if !branchless_is_finite_f32(restored_entropy) {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        if saved_active_count > self.substrate.capacity.max(DEFAULT_SUBSTRATE_CAPACITY)
            || saved_repulsion_threshold > u32::from(u8::MAX)
        {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        let entropy_st = EntropyState::read_unaligned_from(payload, 64)?;
        if !branchless_is_finite_f32(entropy_st.moving_z_entropy)
            || !branchless_is_finite_f32(entropy_st.bifurcation_threshold)
            || !branchless_is_finite_f32(entropy_st.singularity_eps)
            || !branchless_is_finite_f32(entropy_st.sliding_entropy_rate)
            || !branchless_is_finite_f32(entropy_st.repulsion_gain)
            || !branchless_is_finite_f32(entropy_st.sample_temperature)
            || entropy_st.active_branch_mode > 2
        {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        let saved_last_selected_trail_id = read_u32(96)?;
        let saved_head_dim = read_u32(100)?;
        let saved_num_heads = read_u32(104)?;
        let saved_rotary_dim = read_u32(108)?;
        let saved_rope_layout = read_u8(112)?;

        if saved_head_dim == 0
            || (saved_head_dim & 1) != 0
            || saved_num_heads == 0
            || (saved_num_heads as usize) > MAX_ATTENTION_HEADS
            || saved_head_dim > u32::from(u16::MAX)
            || saved_num_heads > u32::from(u16::MAX)
            || saved_rotary_dim > saved_head_dim
            || (saved_rotary_dim & 1) != 0
            || RopePairLayout::from_u8(saved_rope_layout).is_err()
        {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        let new_phase_steering = PhaseSteeringEngine::new(saved_head_dim, saved_num_heads)
            .map_err(|_| LMNK_ERR_CONTAINER_CORRUPTED)?;

        let saved_ledger_slot = LedgerSlot::read_unaligned_from(payload, 128)?;
        let saved_core_layers = CoreStateStack::read_unaligned_from(payload, 192)?;
        let mut restored_spine = SystemSpineState::read_unaligned_from(payload, 320)?;

        if (restored_spine.control_flags & !SPINE_CONTROL_KNOWN_FLAGS) != 0
            || restored_spine.last_observed_entropy_q15 > SPINE_ENTROPY_Q15_ONE
            || (restored_spine.last_veto_mask & !SPINE_BYPASS_VETO_KNOWN_MASK) != 0
        {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        if restored_spine.max_observe_stride == 0 {
            restored_spine.max_observe_stride = 1;
        }

        // Host-locked Spine & slot privilege invariants cannot be stripped by untrusted snapshot data,
        // and transient turn-guard compute pins never bleed into permanent baseline pin floors.
        let default_spine = SystemSpineState::new();
        restored_spine.baseline_pinned_compute_mask |=
            host_baseline_compute_pin_floor | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
        restored_spine.pinned_compute_mask |= restored_spine.baseline_pinned_compute_mask;
        restored_spine.pinned_aperture_mask |=
            host_aperture_pin_floor | default_spine.pinned_aperture_mask;
        restored_spine.pinned_slot_mask |= host_slot_pin_floor | default_spine.pinned_slot_mask;
        restored_spine.pinned_action_mode_mask |=
            host_mode_pin_floor | default_spine.pinned_action_mode_mask;

        // ------------------------------------------------------------------
        // Step 2: Atomic commit after all validation succeeded
        // ------------------------------------------------------------------
        self.config.raw.head_dim = saved_head_dim;
        self.config.raw.num_heads = saved_num_heads;
        self.config.raw.rotary_dim = saved_rotary_dim;
        self.config.raw.rope_layout = saved_rope_layout;
        self.config.raw.guard_hold_tokens = restored_spine.guard_hold_tokens;
        self.config.raw.max_observe_stride = restored_spine.max_observe_stride;
        self.phase_steering = new_phase_steering;
        self.substrate.head_dim = saved_head_dim as u16;
        self.substrate.num_heads = saved_num_heads as u16;

        self.substrate.step_counter = saved_step_counter;
        self.substrate.active_count = saved_active_count;
        self.substrate.repulsion_threshold = saved_repulsion_threshold as u8;
        self.substrate.last_entropy_h_tilde = restored_entropy;
        self.substrate.slot_enabled_mask = saved_slot_enabled_mask;
        self.substrate.slot_pinned_mask = saved_slot_pinned_mask | host_substrate_slot_pin_floor;
        self.substrate.slot_repulsion_mask = saved_slot_repulsion_mask;
        self.substrate.slot_sandbox_mask = saved_slot_sandbox_mask;
        if saved_coupling != 0 {
            self.substrate.coupling_matrix_8x8 = saved_coupling;
            self.config.slot_coupling_matrix = saved_coupling;
        }
        self.substrate.slot_weights = saved_slot_weights;

        self.entropy_controller = EntropyController::new(entropy_st);
        self.entropy_controller
            .set_hysteresis_band(self.config.hysteresis_band);
        self.entropy_controller.set_tau_flux(self.config.tau_flux);

        self.substrate.last_selected_trail_id = saved_last_selected_trail_id;
        self.substrate.ledger_slot = saved_ledger_slot;
        self.core_layers = saved_core_layers;
        self.system_spine = restored_spine;

        Ok(())
    }
}
