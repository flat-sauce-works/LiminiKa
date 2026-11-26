// name: src/core/src/runtime/boundary_delta.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Temporal boundary velocity and chattering analyzer ([`LiminalBoundaryDelta`], 16 B)
//! comparing consecutive Core Layer 0 [`LiminalThresholdState`] epochs.

use crate::foundation::Pod;
use crate::liminal::{
    BitCut8, LiminalThresholdState, NUM_LIMINAL_THRESHOLD_CHANNELS, branchless_select_u8,
};

/// Threshold for leaky integrate-and-fire phase transition pulse on [`LiminalBoundaryDelta`].
pub const CHATTER_FIRE_THRESHOLD_U8: u8 = 64;

/// Branchless 16-byte (`#[repr(C, align(16))]`) temporal differential analyzer measuring boundary
/// velocity (`channel_floor_velocity_q7`), granule transition churn (`granule_transition_mask`),
/// chattering oscillation (`chattering_channel_mask`), leaky integrate-and-fire phase-transition
/// pulses (`chatter_potential_u8`, `phase_transition_pulse`), and stationary fast-skip readiness
/// between consecutive [`LiminalThresholdState`] epochs.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiminalBoundaryDelta {
    /// Signed per-channel boundary velocity (`cur.threshold_floor_q7 - prev.threshold_floor_q7`, `offset 0..8`).
    /// Negative velocity indicates outward threshold expansion; positive indicates contraction.
    pub channel_floor_velocity_q7: [i8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Bitmask of SWAR granules whose overlap-core membership flipped since the previous step (`offset 8`).
    pub granule_transition_mask: u8,
    /// Bitmask of channels whose boundary velocity reversed sign relative to `prev_delta` (`offset 9`).
    pub chattering_channel_mask: u8,
    /// Aggregate boundary kinetic energy in `[0..=127]` (`offset 10`).
    pub kinetic_energy_q7: u8,
    /// `1` when boundaries are quiescent without chattering or pulse fire (`offset 11`).
    pub stationary_shortcut_ready: u8,
    /// `1` when net boundary displacement expands thresholds outward into new territory, `0` otherwise (`offset 12`).
    pub expansion_dominant: u8,
    /// Leaky integrate-and-fire chattering membrane potential (`offset 13`).
    pub chatter_potential_u8: u8,
    /// `1` when `chatter_potential_u8` reaches [`CHATTER_FIRE_THRESHOLD_U8`] and fires a phase-transition pulse (`offset 14`).
    pub phase_transition_pulse: u8,
    /// Explicit alignment padding (`offset 15`).
    pub reserved: u8,
}

impl LiminalBoundaryDelta {
    /// Computes the temporal boundary delta strictly branchlessly from consecutive Layer 0
    /// threshold fields (`prev_layer0`, `cur_layer0`), consecutive overlap masks
    /// (`prev_overlap_mask`, `cur_overlap_mask`), and the prior velocity state `prev_delta`.
    #[inline]
    #[must_use]
    pub fn compute_branchless(
        prev_layer0: &LiminalThresholdState,
        cur_layer0: &LiminalThresholdState,
        prev_overlap_mask: u8,
        cur_overlap_mask: u8,
        prev_delta: &Self,
    ) -> Self {
        let mut channel_floor_velocity_q7 = [0i8; NUM_LIMINAL_THRESHOLD_CHANNELS];
        let mut chattering_channel_mask = 0u8;
        let mut sum_abs_v = 0u32;
        let mut signed_sum = 0i32;

        let mut ch = 0usize;
        while ch < NUM_LIMINAL_THRESHOLD_CHANNELS {
            let cur_f = i32::from(cur_layer0.threshold_floor_q7[ch]);
            let prev_f = i32::from(prev_layer0.threshold_floor_q7[ch]);
            let vel = (cur_f - prev_f).clamp(-127, 127) as i8;
            channel_floor_velocity_q7[ch] = vel;

            let prev_v = i32::from(prev_delta.channel_floor_velocity_q7[ch]);
            let prod = i32::from(vel) * prev_v;
            let flipped = prod < 0;
            chattering_channel_mask |= branchless_select_u8(flipped, 1u8 << ch, 0);

            sum_abs_v = sum_abs_v.saturating_add(u32::from(vel.unsigned_abs()));
            signed_sum += i32::from(vel);
            ch += 1;
        }

        let granule_transition_mask = prev_overlap_mask ^ cur_overlap_mask;
        let overlap_churn = granule_transition_mask.count_ones() * 6;
        let kinetic_energy_q7 = sum_abs_v.saturating_add(overlap_churn).min(127) as u8;

        let prev_v = branchless_select_u8(
            prev_delta.phase_transition_pulse != 0,
            0,
            prev_delta.chatter_potential_u8,
        );
        let leak_min = u8::from(prev_v > 0);
        let leak = (prev_v >> 2).max(leak_min);
        let decayed_v = prev_v.saturating_sub(leak);
        let chatter_drive = (chattering_channel_mask.count_ones() as u8).saturating_mul(16);
        let chatter_potential_u8 = decayed_v.saturating_add(chatter_drive);
        let phase_transition_pulse = u8::from(chatter_potential_u8 >= CHATTER_FIRE_THRESHOLD_U8);

        let stationary_shortcut_ready = u8::from(
            (kinetic_energy_q7 <= 2)
                & (chattering_channel_mask == 0)
                & (phase_transition_pulse == 0),
        );
        let expanded_overlap = BitCut8::split(cur_overlap_mask, prev_overlap_mask).outside != 0;
        let expansion_dominant =
            u8::from((signed_sum < 0) | ((signed_sum == 0) & expanded_overlap));

        Self {
            channel_floor_velocity_q7,
            granule_transition_mask,
            chattering_channel_mask,
            kinetic_energy_q7,
            stationary_shortcut_ready,
            expansion_dominant,
            chatter_potential_u8,
            phase_transition_pulse,
            reserved: 0,
        }
    }
}

// SAFETY: `LiminalBoundaryDelta` is `#[repr(C, align(16))]` with 16 bytes of primitive
// integer fields only and no uninitialized padding bytes or drop glue.
unsafe impl Pod for LiminalBoundaryDelta {}
