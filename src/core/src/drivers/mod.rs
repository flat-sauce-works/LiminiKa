// name: src/core/src/drivers/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 3 (`Drivers`): Dual execution axes (`spine` vs. `ego`) mediating homeostatic
//! convergence vs. intentional deviation through [`DistortionTension`] (`32 B`).
//! `spine` and `ego` never import one another.
//!
//! Depends strictly on `core`, [`crate::liminal`], [`crate::foundation`], and [`crate::stigmergy`].

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

pub mod ego;
pub mod spine;

pub use self::ego::*;
pub use self::spine::*;

use crate::foundation::Pod;
use crate::liminal::{branchless_select_u8, AttributionTag};

// ===================================================================
// Asymmetric Homeostatic Tension (`DistortionTension`, 32 B)
// ===================================================================

/// Classification states of [`DistortionTension`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistortionTensionStateCode {
    /// Healthy differential torsion inside Deadband window `[deadband_low_q7, deadband_high_q7]` (`0`).
    #[default]
    BalancedDeadband = 0,
    /// Slow-Loop accumulation exciting Ego out of passive stagnation (`1`).
    PassiveExcitation = 1,
    /// Immediate 1-step Hot-Path damping on ActiveOverdrive / entropy spike / guard proximity (`2`).
    ActiveImmediateDamping = 2,
}

/// Asymmetric homeostatic restoring tension (`32 B`, `#[repr(C, align(16))]`) mediating the tug-of-war
/// between [`SystemSpineState`] convergence and [`EgoDifferentialNet`] deviation.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistortionTension {
    /// Instantaneous tension magnitude in Q0.7 (`0..=127`, `offset 0`).
    pub tension_q7: u8,
    /// Lower Deadband torsion floor in Q0.7 below which stagnation builds (`offset 1`).
    pub deadband_low_q7: u8,
    /// Upper Deadband torsion ceiling in Q0.7 above which immediate damping fires (`offset 2`).
    pub deadband_high_q7: u8,
    /// Effective multiplicative Q0.7 scale applied to [`DistortionDelta`] amplitudes (`offset 3`).
    pub effective_ego_scale_q7: u8,
    /// Slow-Loop accumulator (`0..=255`) driving gradual Ego excitation (`offset 4`).
    pub excitation_potential_u8: u8,
    /// Remaining Hot-Path cooldown steps after immediate damping (`offset 5`).
    pub suppression_cooldown_steps: u8,
    /// Latest $4p(1-p)$ boundary sensitivity in Q0.7 (`offset 6`).
    pub boundary_sensitivity_q7: u8,
    /// Active [`DistortionTensionStateCode`] byte (`offset 7`).
    pub state_code: u8,
    /// Cumulative count of 1-step Hot-Path immediate damping interventions (`offset 8..12`).
    pub immediate_damping_events: u32,
    /// Cumulative count of Slow-Loop stagnation excitations (`offset 12..16`).
    pub slow_excitation_events: u32,
    /// L-infinity / mean magnitude of the latest differential $\delta_{\mathrm{ego}}$ (`offset 16`).
    pub last_ego_delta_norm_q7: u8,
    /// 2-bit [`AttributionTag`] (`Liminal`, `Ego`, `offset 17`).
    pub attribution_tag: u8,
    /// Observed non-pinned guard proximity in Q0.7 (`offset 18`).
    pub guard_proximity_q7: u8,
    /// Observed exit entropy in Q0.7 (`offset 19`).
    pub exit_entropy_q7: u8,
    /// Cumulative L1 deviation norm tracked across steps (`offset 20..24`).
    pub cumulative_delta_l1: u32,
    /// Monotonic Hot-Path evaluation count (`offset 24..28`).
    pub step_epochs: u32,
    /// Explicit C alignment padding (`offset 28..32`).
    pub reserved_align: [u8; 4],
}

impl Default for DistortionTension {
    #[inline(always)]
    fn default() -> Self {
        Self::new(false)
    }
}

impl DistortionTension {
    /// Constructs a neutral [`DistortionTension`] controller (`effective_ego_scale_q7 = 0` when `!ego_enabled`,
    /// or `96` with Deadband `[16, 84]` when `ego_enabled`).
    #[inline(always)]
    #[must_use]
    pub const fn new(ego_enabled: bool) -> Self {
        Self {
            tension_q7: 0,
            deadband_low_q7: 16,
            deadband_high_q7: 84,
            effective_ego_scale_q7: if ego_enabled { 96 } else { 0 },
            excitation_potential_u8: 0,
            suppression_cooldown_steps: 0,
            boundary_sensitivity_q7: 0,
            state_code: DistortionTensionStateCode::BalancedDeadband as u8,
            immediate_damping_events: 0,
            slow_excitation_events: 0,
            last_ego_delta_norm_q7: 0,
            attribution_tag: AttributionTag::LIMINAL_EGO.raw(),
            guard_proximity_q7: 0,
            exit_entropy_q7: 0,
            cumulative_delta_l1: 0,
            step_epochs: 0,
            reserved_align: [0; 4],
        }
    }

    /// Executes 1 Hot-Path tension step (`0 alloc`, `0 fdiv`, `0` data-dependent branches).
    ///
    /// Asymmetric fast safety response: when `active_overdrive`, `exit_entropy_q7 >= 84`,
    /// `guard_proximity_q7 >= 72`, or `tension_q7 > deadband_high_q7` occurs, suppresses
    /// `effective_ego_scale_q7` in 1 step and arms `suppression_cooldown_steps = 3`.
    #[inline]
    pub fn step_hot_path_branchless(
        &mut self,
        ego_delta_norm_q7: u8,
        boundary_sensitivity_q7: u8,
        exit_entropy_q7: u8,
        guard_proximity_q7: u8,
        active_overdrive: bool,
    ) {
        self.last_ego_delta_norm_q7 = ego_delta_norm_q7.min(127);
        self.boundary_sensitivity_q7 = boundary_sensitivity_q7.min(127);
        self.exit_entropy_q7 = exit_entropy_q7.min(127);
        self.guard_proximity_q7 = guard_proximity_q7.min(127);
        self.cumulative_delta_l1 = self
            .cumulative_delta_l1
            .saturating_add(u32::from(self.last_ego_delta_norm_q7));
        self.step_epochs = self.step_epochs.saturating_add(1);

        // Compute instantaneous torsion mitigated inside the Deadband at high boundary sensitivity.
        let sens_relief = boundary_sensitivity_q7 >> 3;
        let raw_tension = self
            .last_ego_delta_norm_q7
            .saturating_add(self.guard_proximity_q7 >> 1)
            .saturating_add(self.exit_entropy_q7 >> 2)
            .saturating_sub(sens_relief)
            .min(127);
        self.tension_q7 = raw_tension;

        let spike = active_overdrive
            | (self.exit_entropy_q7 >= 84)
            | (self.guard_proximity_q7 >= 72)
            | (raw_tension > self.deadband_high_q7);
        let severe_guard = self.guard_proximity_q7 >= 96;

        let quarter = self.effective_ego_scale_q7 >> 2;
        let mild_floor = branchless_select_u8(self.effective_ego_scale_q7 > 0, quarter.max(8), 0);
        let damped_scale = branchless_select_u8(severe_guard, 0, mild_floor);
        let cooled = self.suppression_cooldown_steps.saturating_sub(1);
        let healthy_recover = branchless_select_u8(
            (cooled == 0) & (self.effective_ego_scale_q7 > 0),
            self.effective_ego_scale_q7.saturating_add(8).min(96),
            self.effective_ego_scale_q7,
        );

        self.effective_ego_scale_q7 = branchless_select_u8(spike, damped_scale, healthy_recover);
        self.excitation_potential_u8 = branchless_select_u8(spike, 0, self.excitation_potential_u8);
        self.suppression_cooldown_steps = branchless_select_u8(spike, 3, cooled);
        self.state_code = branchless_select_u8(
            spike,
            DistortionTensionStateCode::ActiveImmediateDamping as u8,
            DistortionTensionStateCode::BalancedDeadband as u8,
        );
        self.immediate_damping_events = self
            .immediate_damping_events
            .saturating_add(u32::from(spike));
    }

    /// Executes Slow-Loop tension excitation on `PassiveSaturation` / chatter stagnation (`0 fdiv`).
    ///
    /// Accumulates `excitation_potential_u8` gradually and raises `effective_ego_scale_q7` only
    /// when Hot-Path suppression cooldown has cleared.
    #[inline]
    pub fn step_slow_loop_excitation(
        &mut self,
        passive_saturation: bool,
        chatter_potential_u8: u8,
        ego_enabled: bool,
    ) {
        let can_excite = ego_enabled
            & (self.suppression_cooldown_steps == 0)
            & (passive_saturation | (chatter_potential_u8 >= 32));
        if !can_excite {
            return;
        }
        let next_pot = self
            .excitation_potential_u8
            .saturating_add(16u8.saturating_add(chatter_potential_u8 >> 3));
        if next_pot >= 96 {
            self.excitation_potential_u8 = next_pot.saturating_sub(64);
            self.effective_ego_scale_q7 = self
                .effective_ego_scale_q7
                .max(32)
                .saturating_add(12)
                .min(112);
            self.state_code = DistortionTensionStateCode::PassiveExcitation as u8;
            self.slow_excitation_events = self.slow_excitation_events.saturating_add(1);
        } else {
            self.excitation_potential_u8 = next_pot;
        }
    }
}

// SAFETY: `DistortionTension` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for DistortionTension {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distortion_tension_asymmetric_homeostasis() {
        let mut t = DistortionTension::new(true);
        // Immediate 1-step Hot-Path damping under overdrive / guard proximity.
        t.step_hot_path_branchless(80, 64, 90, 98, true);
        assert_eq!(t.effective_ego_scale_q7, 0);
        assert_eq!(t.immediate_damping_events, 1);
        assert_eq!(t.suppression_cooldown_steps, 3);
        assert_eq!(
            t.state_code,
            DistortionTensionStateCode::ActiveImmediateDamping as u8
        );

        // While cooldown > 0, Slow-Loop excitation does not override safety interlock.
        t.step_slow_loop_excitation(true, 96, true);
        assert_eq!(t.slow_excitation_events, 0);

        // After cooldown drains, Slow-Loop excitation awakens Ego out of passive stagnation.
        for _ in 0..3 {
            t.step_hot_path_branchless(10, 48, 20, 0, false);
        }
        for _ in 0..6 {
            t.step_slow_loop_excitation(true, 96, true);
        }
        assert!(t.slow_excitation_events >= 1);
        assert!(t.effective_ego_scale_q7 >= 32);
    }
}
