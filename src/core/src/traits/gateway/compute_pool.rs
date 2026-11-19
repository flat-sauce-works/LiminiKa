// name: src/core/src/traits/gateway/compute_pool.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Step 4: Label-Free Anonymous Compute Pool (`#![no_std]`, zero-alloc, zero-`fdiv`).
//!
//! Dual of the Phase A 8-lane memory workbench pool: manages up to 64 anonymous compute slots
//! (`u64` masks + Q0.7 contribution pheromones) and 8 anonymous execution lanes (externally
//! bound to dGPU/CPU/iGPU) as a pure state machine that emits a pre-execution
//! [`ComputeDispatchMask`].
//!
//! Physical guardrails enforced by construction:
//! 1. **Asymmetric transition cost**: Fast-Loop reversible bypass/throttling is 0-cost and
//!    immediate, whereas Slow-Loop lane migration requires `lane_switch_dwell >= min_lane_dwell`.
//! 2. **No mid-layer D2H sync**: Pheromones update strictly via exit scalar reward
//!    ([`AnonymousComputePool::commit_exit_reward`]).
//! 3. **Zero-`fdiv` EMA + asymmetric hysteresis**: External `+ System` Gauge-Zero pressure surges
//!    retreat rapidly (`alpha = 1/2`), while recovery decays slowly (`alpha = 1/16`) with dwell
//!    gating to protect 1% Low frame times.
//! 4. **Pinned protection & anti-paralysis**: Irreversible slots (`pinned_mask`) are never
//!    bypassed or migrated, and bypassed slots are excluded from `stationary_evidence_mask`.

use crate::traits::primitives::{Pod, branchless_select_u8, branchless_select_u16, branchless_select_u64};

/// Maximum number of anonymous compute slots tracked via `u64` bitmasks.
pub const MAX_ANONYMOUS_COMPUTE_SLOTS: usize = 64;

/// Maximum number of anonymous execution lanes (symmetric with the 8-lane workbench pool).
pub const MAX_ANONYMOUS_EXECUTION_LANES: usize = 8;

/// Sentinel lane index indicating a slot is currently bypassed in the Fast-Loop.
pub const COMPUTE_LANE_BYPASSED: u8 = 0xFF;

/// Pressure regime derived from Gauge-Zero EMA + asymmetric hysteresis.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputePressureRegime {
    /// External pressure is within nominal Gauge-Zero bounds.
    #[default]
    Quiescent = 0,
    /// External pressure is elevated; Fast-Loop throttling/bypass may engage on low-pheromone reversible slots.
    Elevated = 1,
    /// External pressure surged; immediate Fast-Loop retreat and Slow-Loop migration countdown engage.
    Surge = 2,
}

/// External `+ System` Gauge-Zero pressure reading (frame time, memory/PCIe bandwidth, L3 load).
///
/// All pressures are Q0.15 fixed-point (`0..=32767`, where `16384` represents `0.50` normalized load).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SystemPressureReading {
    /// Frame-time pressure relative to baseline (`0..=32767`).
    pub frame_time_pressure_q15: u16,
    /// Memory / interconnect bandwidth pressure (`0..=32767`).
    pub bandwidth_pressure_q15: u16,
    /// Shared L3 / host CPU load pressure (`0..=32767`).
    pub l3_load_pressure_q15: u16,
    /// Non-zero when the current step is an irreversible phase (e.g., Prefill) protecting pinned slots.
    pub irreversible_step: u8,
    /// Explicit alignment padding.
    pub _pad: u8,
}

// SAFETY: `SystemPressureReading` is `#[repr(C)]` with primitive integer fields only and no padding invariants.
unsafe impl Pod for SystemPressureReading {}

/// Configuration thresholds for [`AnonymousComputePool`] (all `0 fdiv` integer Q0.15 / Q0.7).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputePoolConfig {
    /// Gauge-Zero pressure threshold (above baseline) entering [`ComputePressureRegime::Elevated`].
    pub elevated_threshold_q15: u16,
    /// Gauge-Zero pressure threshold (above baseline) entering [`ComputePressureRegime::Surge`].
    pub surge_threshold_q15: u16,
    /// Hysteresis margin subtracted from thresholds when recovering downward.
    pub recovery_hysteresis_q15: u16,
    /// Minimum consecutive low-pressure steps required before stepping down from `Surge`/`Elevated`.
    pub min_recovery_dwell: u16,
    /// Minimum sustained dwell steps before Slow-Loop lane migration is permitted.
    pub min_lane_switch_dwell: u16,
    /// Minimum Q0.7 contribution pheromone (`0..=127`) required for a reversible slot to resist Fast-Loop bypass under pressure.
    pub bypass_pheromone_floor_q7: u8,
    /// Pheromone floor under `Surge` pressure (higher than `Elevated` to shed more reversible work).
    pub surge_bypass_pheromone_floor_q7: u8,
    /// Maximum non-consecutive reversible slots bypassed per step under `Elevated` pressure.
    pub max_elevated_bypass_slots: u8,
    /// Maximum non-consecutive reversible slots bypassed per step under `Surge` pressure.
    pub max_surge_bypass_slots: u8,
    /// Minimum calm steps required before re-entering bypass after an exit instability recovery.
    pub min_re_bypass_dwell: u8,
    /// Explicit alignment padding.
    pub _pad_cfg: u8,
}

impl Default for ComputePoolConfig {
    fn default() -> Self {
        Self {
            elevated_threshold_q15: 4096, // +0.125 above Gauge-Zero baseline
            surge_threshold_q15: 9830,    // +0.300 above Gauge-Zero baseline
            recovery_hysteresis_q15: 1024,
            min_recovery_dwell: 8,
            min_lane_switch_dwell: 24,
            bypass_pheromone_floor_q7: 40,
            surge_bypass_pheromone_floor_q7: 72,
            max_elevated_bypass_slots: 4,
            max_surge_bypass_slots: 8,
            min_re_bypass_dwell: 3,
            _pad_cfg: 0,
        }
    }
}

/// Pre-execution dispatch plan emitted by [`AnonymousComputePool::step_dispatch`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComputeDispatchMask {
    /// Slots scheduled to execute normally on their assigned lane.
    pub execute_mask: u64,
    /// Reversible, unpinned slots bypassed at zero cost in the Fast-Loop.
    pub bypass_mask: u64,
    /// Reversible, unpinned slots throttled (e.g., reduced precision / draft mode) under pressure.
    pub throttle_mask: u64,
    /// Subset of `execute_mask` eligible to count as Layer 0 stationary evidence (excludes bypassed slots).
    pub stationary_evidence_mask: u64,
    /// Primary anonymous execution lane selected for non-migrated work (`0..=7`).
    pub primary_lane: u8,
    /// Fallback anonymous execution lane recommended if Slow-Loop migration is ready (`0..=7`).
    pub fallback_lane: u8,
    /// Active [`ComputePressureRegime`] as `u8`.
    pub regime_u8: u8,
    /// `1` when `lane_switch_dwell >= min_lane_switch_dwell` permits Slow-Loop lane migration, else `0`.
    pub slow_loop_migration_ready: u8,
    /// Smoothed Gauge-Zero relative pressure delta in Q0.15.
    pub relative_pressure_q15: u16,
    /// Active Fast-Loop bypass pheromone floor in Q0.7.
    pub active_pheromone_floor_q7: u8,
    /// Explicit alignment padding.
    pub _pad: u8,
}

// SAFETY: `ComputeDispatchMask` is `#[repr(C)]` with primitive integer fields only.
unsafe impl Pod for ComputeDispatchMask {}

/// Label-free anonymous compute pool state machine (`#![no_std]`, zero-alloc, zero-`fdiv`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnonymousComputePool {
    /// Bitmask of registered anonymous compute slots (`0..64`).
    pub active_mask: u64,
    /// Irreversible slots protected from Fast-Loop bypass and Slow-Loop migration.
    pub pinned_mask: u64,
    /// Repulsed/quarantined slots excluded from primary execution.
    pub repulsion_mask: u64,
    /// Reversible slots eligible for Fast-Loop bypass or throttling under pressure.
    pub reversible_mask: u64,
    /// Per-slot contribution pheromone in Q0.7 (`0..=127`), updated only from exit scalar reward.
    pub slot_pheromone_q7: [u8; MAX_ANONYMOUS_COMPUTE_SLOTS],
    /// Current anonymous execution lane assignment (`0..=7`) per slot.
    pub slot_lane: [u8; MAX_ANONYMOUS_COMPUTE_SLOTS],
    /// Per-lane relative capacity/headroom in Q0.7 (`0..=127`).
    pub lane_capacity_q7: [u8; MAX_ANONYMOUS_EXECUTION_LANES],
    /// Bitmask of enabled anonymous execution lanes (`0..8`).
    pub lane_enabled_mask: u8,
    /// Bitmask of pinned execution lanes.
    pub lane_pinned_mask: u8,
    /// Current primary execution lane index (`0..=7`).
    pub primary_lane: u8,
    /// Current pressure regime.
    pub regime: ComputePressureRegime,
    /// Gauge-Zero baseline pressure in Q0.15 (re-anchored only by `+ System` checkpoints).
    pub baseline_pressure_q15: u16,
    /// Asymmetric-EMA smoothed raw pressure in Q0.15 (`>> 1` on surge, `>> 4` on recovery).
    pub ema_pressure_q15: u16,
    /// Consecutive steps spent below recovery hysteresis threshold.
    pub recovery_dwell: u16,
    /// Consecutive steps spent under sustained `Elevated`/`Surge` pressure for Slow-Loop migration.
    pub lane_switch_dwell: u16,
    /// Remaining calm steps before Fast-Loop bypass may re-engage after an exit instability recovery.
    pub re_bypass_cooldown: u8,
    /// Pool configuration thresholds.
    pub config: ComputePoolConfig,
}

impl Default for AnonymousComputePool {
    fn default() -> Self {
        Self::new(ComputePoolConfig::default())
    }
}

impl AnonymousComputePool {
    /// Creates a new [`AnonymousComputePool`] with lane 0 enabled by default and neutral Q0.7 pheromones.
    #[must_use]
    pub const fn new(config: ComputePoolConfig) -> Self {
        Self {
            active_mask: 0,
            pinned_mask: 0,
            repulsion_mask: 0,
            reversible_mask: 0,
            slot_pheromone_q7: [64; MAX_ANONYMOUS_COMPUTE_SLOTS],
            slot_lane: [0; MAX_ANONYMOUS_COMPUTE_SLOTS],
            lane_capacity_q7: [127, 96, 64, 64, 0, 0, 0, 0],
            lane_enabled_mask: 0b0000_0001,
            lane_pinned_mask: 0,
            primary_lane: 0,
            regime: ComputePressureRegime::Quiescent,
            baseline_pressure_q15: 8192,
            ema_pressure_q15: 8192,
            recovery_dwell: 0,
            lane_switch_dwell: 0,
            re_bypass_cooldown: 0,
            config,
        }
    }

    /// Triggers immediate 1-step full-layer recovery and arms asymmetric re-bypass dwell hysteresis.
    ///
    /// Call on exit entropy spikes, `InternalConflict`, or `LiminalExterior` transitions so the
    /// next step executes all active slots and requires `min_re_bypass_dwell` calm steps before
    /// re-entering Fast-Loop bypass.
    #[inline(always)]
    pub fn signal_exit_instability(&mut self) {
        self.re_bypass_cooldown = self.config.min_re_bypass_dwell.max(1);
    }

    /// Selects up to `max_slots` lowest-pheromone slots from `candidate_mask` while strictly
    /// avoiding consecutive adjacent slot indices (preserving Gauss-Legendre-like interior
    /// representative layers; `0 alloc`, `0 fdiv`).
    #[inline]
    #[must_use]
    pub fn select_non_consecutive_bypass_mask(
        candidate_mask: u64,
        slot_pheromone_q7: &[u8; MAX_ANONYMOUS_COMPUTE_SLOTS],
        max_slots: u8,
    ) -> u64 {
        let mut remaining = candidate_mask;
        let mut selected = 0_u64;
        let mut count = 0_u8;
        while remaining != 0 && count < max_slots {
            // Find the lowest-pheromone eligible slot in `remaining`.
            let mut scan = remaining;
            let mut best_idx = 64_usize;
            let mut best_ph = u8::MAX;
            while scan != 0 {
                let idx = scan.trailing_zeros() as usize;
                scan &= scan - 1;
                let ph = slot_pheromone_q7[idx];
                if ph < best_ph {
                    best_ph = ph;
                    best_idx = idx;
                }
            }
            if best_idx >= MAX_ANONYMOUS_COMPUTE_SLOTS {
                break;
            }
            let bit = 1_u64 << best_idx;
            let neighbors = (bit << 1) | (bit >> 1);
            selected |= bit;
            // Exclude the selected slot and both immediate neighbors to forbid consecutive skips.
            remaining &= !(bit | neighbors);
            count = count.saturating_add(1);
        }
        selected
    }

    /// Re-anchors the Gauge-Zero pressure baseline at a `+ System` / Cold-Path checkpoint.
    pub fn reanchor_baseline_pressure(&mut self, baseline_q15: u16) {
        let clamped = baseline_q15.min(32767);
        self.baseline_pressure_q15 = clamped;
        self.ema_pressure_q15 = clamped;
        self.recovery_dwell = 0;
        self.lane_switch_dwell = 0;
        self.regime = ComputePressureRegime::Quiescent;
    }

    /// Configures anonymous slot masks (`active`, `pinned`, `repulsion`, `reversible`).
    pub fn set_slot_masks(
        &mut self,
        active_mask: u64,
        pinned_mask: u64,
        repulsion_mask: u64,
        reversible_mask: u64,
    ) {
        self.active_mask = active_mask;
        self.pinned_mask = pinned_mask & active_mask;
        self.repulsion_mask = repulsion_mask & active_mask & !self.pinned_mask;
        self.reversible_mask = reversible_mask & active_mask & !self.pinned_mask;
    }

    /// Configures available anonymous execution lanes (`0..8`) and their Q0.7 headroom.
    pub fn set_lanes(
        &mut self,
        enabled_mask: u8,
        pinned_mask: u8,
        capacity_q7: [u8; MAX_ANONYMOUS_EXECUTION_LANES],
    ) {
        self.lane_enabled_mask = enabled_mask.max(1);
        self.lane_pinned_mask = pinned_mask & self.lane_enabled_mask;
        self.lane_capacity_q7 = capacity_q7;
        if (self.lane_enabled_mask & (1_u8 << (self.primary_lane & 7))) == 0 {
            self.primary_lane = self.lane_enabled_mask.trailing_zeros() as u8;
        }
    }

    /// Computes instantaneous composite pressure (max of frame/bandwidth/L3) in Q0.15 (`0 fdiv`).
    #[inline]
    #[must_use]
    pub const fn composite_pressure_q15(reading: SystemPressureReading) -> u16 {
        let fb = if reading.frame_time_pressure_q15 > reading.bandwidth_pressure_q15 {
            reading.frame_time_pressure_q15
        } else {
            reading.bandwidth_pressure_q15
        };
        let raw = if fb > reading.l3_load_pressure_q15 {
            fb
        } else {
            reading.l3_load_pressure_q15
        };
        if raw > 32767 { 32767 } else { raw }
    }

    /// Advances the Gauge-Zero pressure EMA + asymmetric hysteresis state machine and emits a
    /// pre-execution [`ComputeDispatchMask`] (`0 alloc`, `0 fdiv`, branchless mask folding).
    pub fn step_dispatch(&mut self, reading: SystemPressureReading) -> ComputeDispatchMask {
        let raw_q15 = Self::composite_pressure_q15(reading);
        let prev = self.ema_pressure_q15;

        // Asymmetric zero-fdiv EMA: fast retreat on surge (alpha = 1/2), slow recovery (alpha = 1/16).
        let surge_step = ((u32::from(prev) + u32::from(raw_q15)) >> 1) as u16;
        let recov_delta = (u32::from(prev.saturating_sub(raw_q15))) >> 4;
        let recov_step = prev.saturating_sub(recov_delta.max(u32::from(prev > raw_q15)) as u16);
        let is_rising = raw_q15 >= prev;
        self.ema_pressure_q15 = branchless_select_u16(is_rising, surge_step, recov_step);

        let rel_q15 = self
            .ema_pressure_q15
            .saturating_sub(self.baseline_pressure_q15);

        let next_regime = self.update_pressure_regime(rel_q15);
        self.regime = next_regime;

        let in_pressure = !matches!(next_regime, ComputePressureRegime::Quiescent);
        let next_dwell = self.lane_switch_dwell.saturating_add(1);
        self.lane_switch_dwell = branchless_select_u16(in_pressure, next_dwell, 0);

        let slow_loop_migration_ready =
            u8::from(in_pressure && self.lane_switch_dwell >= self.config.min_lane_switch_dwell);

        let (pheromone_floor_q7, max_bypass_slots) = match next_regime {
            ComputePressureRegime::Quiescent => (0, 0),
            ComputePressureRegime::Elevated => (
                self.config.bypass_pheromone_floor_q7,
                self.config.max_elevated_bypass_slots,
            ),
            ComputePressureRegime::Surge => (
                self.config.surge_bypass_pheromone_floor_q7,
                self.config.max_surge_bypass_slots,
            ),
        };

        // Immediate 1-step full-layer recovery with asymmetric re-bypass dwell hysteresis.
        let in_instability_cooldown = self.re_bypass_cooldown > 0;
        self.re_bypass_cooldown = self.re_bypass_cooldown.saturating_sub(1);

        // Irreversible steps (e.g. Prefill) or instability cooldown force-protect all active slots.
        let effective_reversible = branchless_select_u64(
            reading.irreversible_step != 0 || in_instability_cooldown,
            0,
            self.reversible_mask & !self.pinned_mask,
        );

        let mut low_pheromone_mask = 0_u64;
        let mut throttle_candidate_mask = 0_u64;
        for (idx, &pheromone) in self.slot_pheromone_q7.iter().enumerate() {
            let bit = 1_u64 << idx;
            let below_bypass = pheromone < pheromone_floor_q7;
            let below_throttle = pheromone < pheromone_floor_q7.saturating_add(16).min(127);
            low_pheromone_mask |= branchless_select_u64(below_bypass, bit, 0);
            throttle_candidate_mask |= branchless_select_u64(below_throttle, bit, 0);
        }

        let candidate_bypass = effective_reversible & low_pheromone_mask;
        let bypass_mask = Self::select_non_consecutive_bypass_mask(
            candidate_bypass,
            &self.slot_pheromone_q7,
            max_bypass_slots,
        );
        let eligible_exec = self.active_mask & !self.repulsion_mask;
        let execute_mask = (eligible_exec & !bypass_mask) | self.pinned_mask;
        let throttle_mask = effective_reversible & throttle_candidate_mask & !bypass_mask;
        // Anti-paralysis guardrail: bypassed slots never count as Layer 0 stationary evidence.
        let stationary_evidence_mask = execute_mask & !bypass_mask;

        let fallback_lane = self.select_fallback_lane();

        ComputeDispatchMask {
            execute_mask,
            bypass_mask,
            throttle_mask,
            stationary_evidence_mask,
            primary_lane: self.primary_lane,
            fallback_lane,
            regime_u8: next_regime as u8,
            slow_loop_migration_ready,
            relative_pressure_q15: rel_q15,
            active_pheromone_floor_q7: pheromone_floor_q7,
            _pad: 0,
        }
    }

    /// Updates per-slot contribution pheromones strictly from an exit scalar reward (`0..=127`)
    /// without mid-layer D2H synchronization (`0 alloc`, `0 fdiv`).
    ///
    /// Only slots present in `executed_mask` are reinforced or decayed; bypassed slots decay gently
    /// (`>> 5`) so unused paths do not freeze permanently.
    pub fn commit_exit_reward(&mut self, executed_mask: u64, reward_q7: u8) {
        let clamped_reward = reward_q7.min(127);
        for (idx, pheromone) in self.slot_pheromone_q7.iter_mut().enumerate() {
            let bit = 1_u64 << idx;
            if (self.active_mask & bit) == 0 {
                continue;
            }
            let cur = i32::from(*pheromone);
            if (executed_mask & bit) != 0 {
                // Q0.7 EMA with alpha = 1/8 (`>> 3`): zero fdiv.
                let next = cur + ((i32::from(clamped_reward) - cur) >> 3);
                *pheromone = next.clamp(0, 127) as u8;
            } else {
                // Gentle decay (`>> 5`) for bypassed slots.
                let next = cur - (cur >> 5);
                *pheromone = next.clamp(0, 127) as u8;
            }
        }
    }

    /// Applies Slow-Loop lane migration for `migrate_mask` only when `slow_loop_migration_ready` is true.
    ///
    /// Pinned slots are never migrated. Returns `true` if any slot lane assignment changed.
    pub fn try_migrate_lanes(&mut self, migrate_mask: u64, target_lane: u8) -> bool {
        let target = target_lane & 7;
        if self.lane_switch_dwell < self.config.min_lane_switch_dwell
            || (self.lane_enabled_mask & (1_u8 << target)) == 0
        {
            return false;
        }
        let movable = migrate_mask & self.active_mask & !self.pinned_mask;
        if movable == 0 {
            return false;
        }
        for (idx, lane) in self.slot_lane.iter_mut().enumerate() {
            let bit = 1_u64 << idx;
            let move_this = (movable & bit) != 0;
            *lane = branchless_select_u8(move_this, target, *lane);
        }
        self.lane_switch_dwell = 0;
        true
    }

    #[inline]
    fn update_pressure_regime(&mut self, rel_q15: u16) -> ComputePressureRegime {
        let hyst = self.config.recovery_hysteresis_q15;
        let elev = self.config.elevated_threshold_q15;
        let surge = self.config.surge_threshold_q15;

        if rel_q15 >= surge {
            self.recovery_dwell = 0;
            return ComputePressureRegime::Surge;
        }

        match self.regime {
            ComputePressureRegime::Quiescent => {
                self.recovery_dwell = 0;
                if rel_q15 >= elev {
                    ComputePressureRegime::Elevated
                } else {
                    ComputePressureRegime::Quiescent
                }
            }
            ComputePressureRegime::Elevated => {
                if rel_q15 < elev.saturating_sub(hyst) {
                    self.recovery_dwell = self.recovery_dwell.saturating_add(1);
                    if self.recovery_dwell >= self.config.min_recovery_dwell {
                        self.recovery_dwell = 0;
                        ComputePressureRegime::Quiescent
                    } else {
                        ComputePressureRegime::Elevated
                    }
                } else {
                    self.recovery_dwell = 0;
                    ComputePressureRegime::Elevated
                }
            }
            ComputePressureRegime::Surge => {
                if rel_q15 < surge.saturating_sub(hyst) {
                    self.recovery_dwell = self.recovery_dwell.saturating_add(1);
                    if self.recovery_dwell >= self.config.min_recovery_dwell {
                        self.recovery_dwell = 0;
                        if rel_q15 >= elev {
                            ComputePressureRegime::Elevated
                        } else {
                            ComputePressureRegime::Quiescent
                        }
                    } else {
                        ComputePressureRegime::Surge
                    }
                } else {
                    self.recovery_dwell = 0;
                    ComputePressureRegime::Surge
                }
            }
        }
    }

    #[inline]
    fn select_fallback_lane(&self) -> u8 {
        let other_lanes = self.lane_enabled_mask & !(1_u8 << (self.primary_lane & 7));
        if other_lanes == 0 {
            return self.primary_lane;
        }
        let mut best_lane = other_lanes.trailing_zeros() as u8;
        let mut best_cap = 0_u8;
        for (lane_idx, &cap) in self.lane_capacity_q7.iter().enumerate() {
            let bit = 1_u8 << lane_idx;
            let eligible = (other_lanes & bit) != 0 && cap >= best_cap;
            best_lane = branchless_select_u8(eligible, lane_idx as u8, best_lane);
            best_cap = branchless_select_u8(eligible, cap, best_cap);
        }
        best_lane
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    const _: () = {
        assert!(size_of::<SystemPressureReading>() == 8);
        assert!(size_of::<ComputeDispatchMask>() == 40);
        assert!(align_of::<AnonymousComputePool>() == 64);
    };

    #[test]
    fn compute_pool_enforces_asymmetric_hysteresis_and_slow_loop_dwell() {
        let mut pool = AnonymousComputePool::new(ComputePoolConfig {
            elevated_threshold_q15: 4000,
            surge_threshold_q15: 10000,
            recovery_hysteresis_q15: 1000,
            min_recovery_dwell: 4,
            min_lane_switch_dwell: 6,
            bypass_pheromone_floor_q7: 48,
            surge_bypass_pheromone_floor_q7: 80,
            ..ComputePoolConfig::default()
        });
        pool.reanchor_baseline_pressure(8000);
        pool.set_lanes(0b0011, 0b0001, [127, 96, 0, 0, 0, 0, 0, 0]);
        // Slots 0..=1 pinned, slots 2..=3 reversible, slot 4 repulsed.
        pool.set_slot_masks(0b1_1111, 0b0_0011, 0b1_0000, 0b0_1100);
        // Slot 2 has low pheromone (32), slot 3 has high pheromone (96).
        pool.slot_pheromone_q7[2] = 32;
        pool.slot_pheromone_q7[3] = 96;

        // 1. Sudden pressure surge triggers fast EMA retreat (alpha = 1/2) -> Surge quickly.
        let surge_reading = SystemPressureReading {
            frame_time_pressure_q15: 30000,
            bandwidth_pressure_q15: 12000,
            l3_load_pressure_q15: 10000,
            irreversible_step: 0,
            _pad: 0,
        };
        let d1 = pool.step_dispatch(surge_reading);
        assert_eq!(d1.regime_u8, ComputePressureRegime::Surge as u8);
        // Slot 2 is reversible & low pheromone -> bypassed; slot 3 resists bypass; slot 4 repulsed.
        assert_eq!(d1.bypass_mask, 0b0_0100);
        assert_eq!(d1.execute_mask, 0b0_1011);
        // Anti-paralysis guardrail: bypassed slot 2 must NOT appear in stationary_evidence_mask.
        assert_eq!(d1.stationary_evidence_mask & d1.bypass_mask, 0);
        assert_eq!(d1.stationary_evidence_mask, 0b0_1011);
        // Slow-Loop migration must be blocked until min_lane_switch_dwell elapses.
        assert_eq!(d1.slow_loop_migration_ready, 0);
        assert!(!pool.try_migrate_lanes(0b0_1100, 1));

        // 2. Sudden drop to baseline must NOT immediately drop Surge (slow alpha = 1/16 + recovery dwell).
        let calm_reading = SystemPressureReading {
            frame_time_pressure_q15: 8000,
            bandwidth_pressure_q15: 8000,
            l3_load_pressure_q15: 8000,
            irreversible_step: 0,
            _pad: 0,
        };
        let d2 = pool.step_dispatch(calm_reading);
        assert_eq!(d2.regime_u8, ComputePressureRegime::Surge as u8);

        // 3. Sustained pressure unlocks Slow-Loop migration, while pinned slots remain immovable.
        for _ in 0..6 {
            let _ = pool.step_dispatch(surge_reading);
        }
        let d_ready = pool.step_dispatch(surge_reading);
        assert_eq!(d_ready.slow_loop_migration_ready, 1);
        assert_eq!(d_ready.fallback_lane, 1);
        assert!(pool.try_migrate_lanes(0b0_1111, 1));
        assert_eq!(pool.slot_lane[0], 0); // Pinned slot 0 stays on lane 0.
        assert_eq!(pool.slot_lane[1], 0); // Pinned slot 1 stays on lane 0.
        assert_eq!(pool.slot_lane[2], 1); // Unpinned slot 2 migrated to lane 1.
        assert_eq!(pool.slot_lane[3], 1); // Unpinned slot 3 migrated to lane 1.
    }

    #[test]
    fn compute_pool_protects_irreversible_steps_and_updates_exit_reward() {
        let mut pool = AnonymousComputePool::default();
        pool.reanchor_baseline_pressure(8000);
        pool.set_slot_masks(0b1111, 0b0001, 0, 0b1110);
        pool.slot_pheromone_q7[1] = 10;
        pool.slot_pheromone_q7[2] = 10;

        // Irreversible step (e.g., Prefill) forbids Fast-Loop bypass even under Surge.
        let prefill_surge = SystemPressureReading {
            frame_time_pressure_q15: 32000,
            bandwidth_pressure_q15: 32000,
            l3_load_pressure_q15: 32000,
            irreversible_step: 1,
            _pad: 0,
        };
        let d = pool.step_dispatch(prefill_surge);
        assert_eq!(d.bypass_mask, 0);
        assert_eq!(d.execute_mask, 0b1111);

        // Exit scalar reward reinforces executed slots and gently decays unexecuted ones.
        let before_1 = pool.slot_pheromone_q7[1];
        let before_3 = pool.slot_pheromone_q7[3];
        pool.commit_exit_reward(0b0011, 120);
        assert!(pool.slot_pheromone_q7[1] > before_1);
        assert!(pool.slot_pheromone_q7[3] < before_3);
    }

    #[test]
    fn compute_pool_enforces_non_consecutive_bypass_and_instability_recovery() {
        let mut pool = AnonymousComputePool::default();
        pool.reanchor_baseline_pressure(8000);
        // 32-layer model: head 0..4 and tail 28..32 pinned, middle 4..28 reversible.
        let active = (1_u64 << 32) - 1;
        let pinned = 0x0000_000F_u64 | (0x0000_000F_u64 << 28);
        let rev = active & !pinned;
        pool.set_slot_masks(active, pinned, 0, rev);

        let surge = SystemPressureReading {
            frame_time_pressure_q15: 30000,
            bandwidth_pressure_q15: 30000,
            l3_load_pressure_q15: 15000,
            irreversible_step: 0,
            _pad: 0,
        };
        let d = pool.step_dispatch(surge);
        assert_eq!(d.regime_u8, ComputePressureRegime::Surge as u8);
        // Must never bypass pinned layers.
        assert_eq!(d.bypass_mask & pinned, 0);
        // Must cap at max_surge_bypass_slots (8) and never bypass two adjacent layers.
        assert!(d.bypass_mask.count_ones() <= 8);
        assert!(d.bypass_mask.count_ones() >= 4);
        assert_eq!(d.bypass_mask & (d.bypass_mask << 1), 0);
        // Bypassed layers never appear in stationary_evidence_mask.
        assert_eq!(d.stationary_evidence_mask & d.bypass_mask, 0);

        // Exit instability immediately forces full-layer recovery for min_re_bypass_dwell steps.
        pool.signal_exit_instability();
        for _ in 0..pool.config.min_re_bypass_dwell {
            let d_rec = pool.step_dispatch(surge);
            assert_eq!(d_rec.bypass_mask, 0);
            assert_eq!(d_rec.execute_mask, active);
        }
        // After dwell expires under sustained surge, bypass may re-engage.
        let d_after = pool.step_dispatch(surge);
        assert!(d_after.bypass_mask.count_ones() > 0);
    }
}
