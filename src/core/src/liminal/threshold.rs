// name: src/core/src/liminal/threshold.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Threshold sensitivity functions ($4p(1-p)$) and [`LiminalThresholdState`] (32 B).

#![forbid(unsafe_code)]

use crate::liminal::constants::{NUM_LIMINAL_THRESHOLD_CHANNELS, THRESHOLD_STIGMERGY_WIDE};

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
    if scaled > 127 { 127 } else { scaled as u8 }
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

/// Dynamic 32-byte (`#[repr(C, align(16))]`) boundary threshold state governing per-channel
/// cognitive thresholds across Core Stack Slot 0 (`Liminal`):
/// - `0`: `SystemConstraintSignal` (`System`)
/// - `1`: `ObjectiveFactSignal` (`Fact`)
/// - `2`: `SubjectivePersonaSignal` (`Persona`)
/// - `3`: `ContradictionGuardSignal` (`Guard`)
/// - `4`: `CognitiveTrajectoryLedger` (`Circuit`)
/// - `5`: `RamPhaseKvCacheOps` (`KvCache`)
/// - `6`: `RelativeCouplingMatrix` (`RelativeCoupling`)
/// - `7`: `StigmergicSubstrateOps` (Wide Execution Threshold)
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiminalThresholdState {
    /// Per-channel `Liminal` threshold floor in Q0.7 (`[8..=120]`): higher values expand the
    /// unobserved exterior for that channel, filtering low-support signals cleanly to `0`.
    pub threshold_floor_q7: [u8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Per-channel cognitive window opening gain in Q0.7 (`[16..=127]`).
    pub aperture_gain_q7: [u8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Per-channel signed Q7 focal phase offset (`[-127..=127]`).
    pub phase_shift_q7: [i8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Active cognitive threshold bitmask reflecting dynamically open vs. collapsed channels.
    pub active_aperture_mask: u32,
    /// Monotonic count of autonomous threshold adaptation cycles applied.
    pub adaptation_epochs: u32,
}

impl Default for LiminalThresholdState {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl LiminalThresholdState {
    /// Neutral baseline cognitive threshold configuration (`32 B`, `#[repr(C, align(16))]`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold_floor_q7: [20, 28, 32, 24, 24, 24, 24, 16],
            aperture_gain_q7: [96; NUM_LIMINAL_THRESHOLD_CHANNELS],
            phase_shift_q7: [0; NUM_LIMINAL_THRESHOLD_CHANNELS],
            active_aperture_mask: THRESHOLD_STIGMERGY_WIDE,
            adaptation_epochs: 0,
        }
    }

    /// Returns channel `ch`'s normalized `Liminal` threshold floor in `[0.0, 1.0]`.
    #[inline(always)]
    #[must_use]
    pub const fn channel_threshold_floor_f32(&self, ch: usize) -> f32 {
        (self.threshold_floor_q7[ch & (NUM_LIMINAL_THRESHOLD_CHANNELS - 1)] as f32) * (1.0 / 127.0)
    }

    /// Returns channel `ch`'s normalized cognitive opening gain in `[0.0, 1.0]`.
    #[inline(always)]
    #[must_use]
    pub const fn channel_gain_f32(&self, ch: usize) -> f32 {
        (self.aperture_gain_q7[ch & (NUM_LIMINAL_THRESHOLD_CHANNELS - 1)] as f32) * (1.0 / 127.0)
    }
}
