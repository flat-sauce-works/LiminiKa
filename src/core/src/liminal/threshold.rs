// name: src/core/src/liminal/threshold.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Root [`Liminal`] boundary supertrait, $4p(1-p)$ threshold sensitivity functions, and
//! [`LiminalThresholdField`] (32 B).

#![forbid(unsafe_code)]

use crate::liminal::axiom::BitCut32;
use crate::liminal::constants::{NUM_LIMINAL_THRESHOLD_CHANNELS, THRESHOLD_ACTION_HUB_WIDE};

/// Root epistemic supertrait representing the boundary separating cognizable state from the
/// unknowable exterior.
///
/// Every capability and signal trait in LiminiKa refines a specific cognitive threshold mask out of
/// `Liminal`. Any axis outside a trait's active threshold (`!threshold_mask`) belongs strictly to the
/// unknowable exterior.
pub trait Liminal {
    /// Splits `world_bits` branchlessly along `threshold_mask` via [`BitCut32::split`].
    #[inline(always)]
    fn cut_bits(&self, world_bits: u32, threshold_mask: u32) -> BitCut32 {
        BitCut32::split(world_bits, threshold_mask)
    }

    /// Partitions `world_bits` into `(cognized_bits, unknowable_void_bits)` strictly branchlessly
    /// for a trait's declared `threshold_mask` via [`Self::cut_bits`].
    #[inline(always)]
    fn partition_cognitive_and_unknowable_bits(
        &self,
        threshold_mask: u32,
        world_bits: u32,
    ) -> (u32, u32) {
        let cut = self.cut_bits(world_bits, threshold_mask);
        (cut.inside, cut.outside)
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
