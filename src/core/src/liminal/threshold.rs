// name: src/core/src/liminal/threshold.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Root [`Liminal`] boundary supertrait, $4p(1-p)$ threshold sensitivity functions,
//! [`LiminalThresholdField`] (32 B), [`LiminalBoundaryDelta`] (16 B), and
//! [`GCSO_LIMINAL_HORIZON_TOPOLOGY`].

#![forbid(unsafe_code)]

use crate::liminal::constants::{
    NUM_LIMINAL_THRESHOLD_CHANNELS, THRESHOLD_ACTION_HUB_WIDE, THRESHOLD_COGNITIVE_CIRCUIT,
    THRESHOLD_CONTRADICTION_GUARD, THRESHOLD_KV_PHASE_CACHE, THRESHOLD_OBJECTIVE_FACT,
    THRESHOLD_RELATIVE_COUPLING, THRESHOLD_SUBJECTIVE_PERSONA, THRESHOLD_SYSTEM_CONSTRAINT,
};
use crate::liminal::select::branchless_select_u8;

/// Root epistemic supertrait representing the boundary separating cognizable state from the
/// unknowable exterior.
///
/// Every capability and signal trait in LiminiKa refines a specific cognitive threshold mask out of
/// `Liminal`. Any axis outside a trait's active threshold (`!threshold_mask`) belongs strictly to the
/// unknowable exterior.
pub trait Liminal {
    /// Complete bitmask representing the unbounded unknowable exterior prior to any cognitive cut.
    const ROOT_UNKNOWABLE_VOID_MASK: u32 = u32::MAX;

    /// Partitions `world_bits` into `(cognized_bits, unknowable_void_bits)` strictly branchlessly
    /// for a trait's declared `threshold_mask`.
    #[inline(always)]
    fn partition_cognitive_and_unknowable_bits(
        &self,
        threshold_mask: u32,
        world_bits: u32,
    ) -> (u32, u32) {
        (world_bits & threshold_mask, world_bits & !threshold_mask)
    }
}

impl<T: ?Sized> Liminal for T {}

/// Evaluates the branchless self-differentiable boundary sensitivity gate $4p(1-p)$ in Q0.7 (`0 fdiv`).
///
/// Vanishes at both ignorance (`p_q7 == 0`) and saturation (`p_q7 >= 127`) while peaking at the
/// transition boundary (`p_q7 == 64`, returning `126`).
#[inline(always)]
#[must_use]
pub const fn sensitivity_4p1mp_q7(p_q7: u8) -> u8 {
    let p = if p_q7 > 127 { 127 } else { p_q7 };
    let prod = (p as u32) * ((127 - p) as u32);
    let scaled = prod >> 5;
    if scaled > 127 {
        127
    } else {
        scaled as u8
    }
}

/// Evaluates the normalized scalar boundary sensitivity $4p(1-p)$ clamped to $[0.0, 1.0]$.
#[inline(always)]
#[must_use]
pub fn sensitivity_4p1mp_f32(p: f32) -> f32 {
    if !p.is_finite() {
        return 0.0;
    }
    let pc = p.clamp(0.0, 1.0);
    (4.0 * pc * (1.0 - pc)).clamp(0.0, 1.0)
}

/// Dynamic 32-byte (`#[repr(C, align(16))]`) boundary threshold field governing per-channel
/// cognitive thresholds across Core Layer 0 (`Liminal`):
/// - `0`: `SystemConstraintSignal` (`System`)
/// - `1`: `ObjectiveFactSignal` (`Fact`)
/// - `2`: `SubjectivePersonaSignal` (`Persona`)
/// - `3`: `ContradictionGuardSignal` (`Guard`)
/// - `4`: `CognitiveTrajectoryLedger` (`Circuit`)
/// - `5`: `RamPhaseKvCacheOps` (`KvCache`)
/// - `6`: `RelativeCouplingMatrix` (`RelativeCoupling`)
/// - `7`: `ActionHub` (Wide Execution Threshold)
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiminalThresholdField {
    /// Per-channel `Liminal` threshold floor in Q0.7 (`[8..=120]`): higher values expand the
    /// unknowable exterior for that channel, filtering low-support signals cleanly to `0`.
    pub void_floor_q7: [u8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Per-channel cognitive window opening gain in Q0.7 (`[16..=127]`).
    pub aperture_gain_q7: [u8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Per-channel signed Q7 focal phase offset (`[-127..=127]`).
    pub phase_shift_q7: [i8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Active cognitive threshold bitmask reflecting dynamically open vs. collapsed channels.
    pub active_aperture_mask: u32,
    /// Monotonic count of autonomous threshold adaptation cycles applied.
    pub adaptation_epochs: u32,
}

impl Default for LiminalThresholdField {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl LiminalThresholdField {
    /// Neutral baseline cognitive threshold configuration (`32 B`, `#[repr(C, align(16))]`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            void_floor_q7: [20, 28, 32, 24, 24, 24, 24, 16],
            aperture_gain_q7: [96; NUM_LIMINAL_THRESHOLD_CHANNELS],
            phase_shift_q7: [0; NUM_LIMINAL_THRESHOLD_CHANNELS],
            active_aperture_mask: THRESHOLD_ACTION_HUB_WIDE,
            adaptation_epochs: 0,
        }
    }

    /// Returns channel `ch`'s normalized `Liminal` threshold floor in `[0.0, 1.0]`.
    #[inline(always)]
    #[must_use]
    pub const fn channel_void_floor_f32(&self, ch: usize) -> f32 {
        (self.void_floor_q7[ch & (NUM_LIMINAL_THRESHOLD_CHANNELS - 1)] as f32) * (1.0 / 127.0)
    }

    /// Returns channel `ch`'s normalized cognitive opening gain in `[0.0, 1.0]`.
    #[inline(always)]
    #[must_use]
    pub const fn channel_gain_f32(&self, ch: usize) -> f32 {
        (self.aperture_gain_q7[ch & (NUM_LIMINAL_THRESHOLD_CHANNELS - 1)] as f32) * (1.0 / 127.0)
    }
}

/// Threshold for leaky integrate-and-fire phase transition pulse on [`LiminalBoundaryDelta`].
pub const CHATTER_FIRE_THRESHOLD_U8: u8 = 64;

/// Branchless 16-byte (`#[repr(C, align(16))]`) temporal differential analyzer measuring boundary
/// velocity (`channel_floor_velocity_q7`), granule transition churn (`granule_transition_mask`),
/// chattering oscillation (`chattering_channel_mask`), leaky integrate-and-fire phase-transition
/// pulses (`chatter_potential_u8`, `phase_transition_pulse`), and stationary fast-skip readiness
/// between consecutive [`LiminalThresholdField`] epochs.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiminalBoundaryDelta {
    /// Signed per-channel boundary velocity (`cur.void_floor_q7 - prev.void_floor_q7`, `offset 0..8`).
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
        prev_layer0: &LiminalThresholdField,
        cur_layer0: &LiminalThresholdField,
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
            let cur_f = i32::from(cur_layer0.void_floor_q7[ch]);
            let prev_f = i32::from(prev_layer0.void_floor_q7[ch]);
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
        let expanded_overlap = (cur_overlap_mask & !prev_overlap_mask) != 0;
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

/// Static compile-time descriptor binding a trait horizon node to its static cognitive threshold
/// and unknowable exterior complement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiminalHorizonNodeDescriptor {
    /// Mermaid graph node identifier (`"N0"`..`"N7"`).
    pub node_id: &'static str,
    /// Human-readable Rust trait name refining [`Liminal`].
    pub trait_name: &'static str,
    /// Channel index (`0..8`) inside [`LiminalThresholdField`].
    pub channel_idx: usize,
    /// Static cognitive threshold mask owned by this trait node.
    pub static_aperture_mask: u32,
    /// Type-level unknowable exterior complement (`!static_aperture_mask`).
    pub unknowable_void_mask: u32,
}

/// Canonical 8-channel trait topology table derived from type-level [`Liminal`] refinements.
pub const GCSO_LIMINAL_HORIZON_TOPOLOGY: [LiminalHorizonNodeDescriptor;
    NUM_LIMINAL_THRESHOLD_CHANNELS] = [
    LiminalHorizonNodeDescriptor {
        node_id: "N0",
        trait_name: "SystemConstraintSignal",
        channel_idx: 0,
        static_aperture_mask: THRESHOLD_SYSTEM_CONSTRAINT,
        unknowable_void_mask: !THRESHOLD_SYSTEM_CONSTRAINT,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N1",
        trait_name: "ObjectiveFactSignal",
        channel_idx: 1,
        static_aperture_mask: THRESHOLD_OBJECTIVE_FACT,
        unknowable_void_mask: !THRESHOLD_OBJECTIVE_FACT,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N2",
        trait_name: "SubjectivePersonaSignal",
        channel_idx: 2,
        static_aperture_mask: THRESHOLD_SUBJECTIVE_PERSONA,
        unknowable_void_mask: !THRESHOLD_SUBJECTIVE_PERSONA,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N3",
        trait_name: "ContradictionGuardSignal",
        channel_idx: 3,
        static_aperture_mask: THRESHOLD_CONTRADICTION_GUARD,
        unknowable_void_mask: !THRESHOLD_CONTRADICTION_GUARD,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N4",
        trait_name: "CognitiveTrajectoryLedger",
        channel_idx: 4,
        static_aperture_mask: THRESHOLD_COGNITIVE_CIRCUIT,
        unknowable_void_mask: !THRESHOLD_COGNITIVE_CIRCUIT,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N5",
        trait_name: "RamPhaseKvCacheOps",
        channel_idx: 5,
        static_aperture_mask: THRESHOLD_KV_PHASE_CACHE,
        unknowable_void_mask: !THRESHOLD_KV_PHASE_CACHE,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N6",
        trait_name: "RelativeCouplingMatrix",
        channel_idx: 6,
        static_aperture_mask: THRESHOLD_RELATIVE_COUPLING,
        unknowable_void_mask: !THRESHOLD_RELATIVE_COUPLING,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N7",
        trait_name: "ActionHub",
        channel_idx: 7,
        static_aperture_mask: THRESHOLD_ACTION_HUB_WIDE,
        unknowable_void_mask: !THRESHOLD_ACTION_HUB_WIDE,
    },
];
