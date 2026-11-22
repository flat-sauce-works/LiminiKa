// name: src/core/src/stigmergy/substrate/lanes.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Anonymous 8-lane workbench pool allocation, `BitCut8` quadrants, and sandbox settlement.

use super::{StigmergicSubstrate, WorkbenchLanePlacement, WorkbenchSlotSpec};
use crate::foundation::{
    branchless_argmin_f32_x8_masked, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_relu_f32,
};
use crate::liminal::{
    BitCut8, LmnkResult, LMNK_ERR_INVALID_ARGUMENT, LMNK_ERR_SUBSTRATE_FULL, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS,
};

impl StigmergicSubstrate {
    /// Allocates an anonymous physical workbench lane (`0..=7`) according to `spec.placement`,
    /// populating `slot_phase_bank`, `slot_weights`, and the 4 lane attribute bitmasks in $\mathcal{O}(1)$
    /// zero-alloc time.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `spec.weight` is non-finite or an explicit lane
    /// index `>= MAX_ATTRACTOR_DOMAINS`, or [`LMNK_ERR_SUBSTRATE_FULL`] if `AutoFree` is
    /// requested while all 8 lanes are pinned.
    #[inline]
    pub fn allocate_lane(&mut self, spec: &WorkbenchSlotSpec) -> LmnkResult<u8> {
        if !branchless_is_finite_f32(spec.weight) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let lane = match spec.placement {
            WorkbenchLanePlacement::AutoFree => {
                let (active_cut, free_cut) = self.lane_lifecycle_quadrants();
                if free_cut.outside != 0 {
                    free_cut.outside.trailing_zeros() as u8
                } else {
                    let evictable = active_cut.outside;
                    if evictable == 0 {
                        return Err(LMNK_ERR_SUBSTRATE_FULL);
                    }
                    let mut clean_weights = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
                    for (dst, &w) in clean_weights.iter_mut().zip(self.slot_weights.iter()) {
                        *dst = branchless_relu_f32(w);
                    }
                    let (best_lane, _) = branchless_argmin_f32_x8_masked(&clean_weights, evictable);
                    if usize::from(best_lane) >= MAX_ATTRACTOR_DOMAINS {
                        return Err(LMNK_ERR_SUBSTRATE_FULL);
                    }
                    best_lane
                }
            }
            WorkbenchLanePlacement::Explicit { lane, .. } => {
                if usize::from(lane) >= MAX_ATTRACTOR_DOMAINS {
                    return Err(LMNK_ERR_INVALID_ARGUMENT);
                }
                lane
            }
        };

        let lane_idx = usize::from(lane);
        let lane_bit = 1u8 << lane;
        let start = lane_idx * MAX_ATTENTION_HEADS;
        let end = start + MAX_ATTENTION_HEADS;
        self.slot_phase_bank.data[start..end].copy_from_slice(&spec.phases);
        self.slot_weights[lane_idx] = branchless_relu_f32(spec.weight);
        self.slot_enabled_mask |= lane_bit;

        let pin_bit = u8::from(spec.is_pinned()).wrapping_neg() & lane_bit;
        let rep_bit = u8::from(spec.is_repulsion()).wrapping_neg() & lane_bit;
        let sb_bit = u8::from(spec.is_sandbox()).wrapping_neg() & lane_bit;
        self.slot_pinned_mask = (self.slot_pinned_mask & !lane_bit) | pin_bit;
        self.slot_repulsion_mask = (self.slot_repulsion_mask & !lane_bit) | rep_bit;
        self.slot_sandbox_mask = (self.slot_sandbox_mask & !lane_bit) | sb_bit;
        Ok(lane)
    }

    /// Releases physical workbench `lane` (`0..=7`), zeroing its phase row and clearing its bits
    /// across `slot_enabled_mask`, `slot_pinned_mask`, `slot_repulsion_mask`, and `slot_sandbox_mask`.
    ///
    /// Returns `Ok(true)` if the lane had been enabled prior to release.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `lane >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn release_lane(&mut self, lane: u8) -> LmnkResult<bool> {
        let lane_idx = usize::from(lane);
        if lane_idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let lane_bit = 1u8 << lane;
        let was_active = (self.slot_enabled_mask & lane_bit) != 0;
        let start = lane_idx * MAX_ATTENTION_HEADS;
        let end = start + MAX_ATTENTION_HEADS;
        self.slot_phase_bank.data[start..end].fill(0);
        self.slot_weights[lane_idx] = 0.0;
        self.slot_enabled_mask &= !lane_bit;
        self.slot_pinned_mask &= !lane_bit;
        self.slot_repulsion_mask &= !lane_bit;
        self.slot_sandbox_mask &= !lane_bit;
        Ok(was_active)
    }

    /// Releases all unpinned active workbench lanes while leaving pinned lanes untouched, returning
    /// the count of lanes cleared.
    #[inline]
    pub fn release_unpinned_lanes(&mut self) -> u8 {
        let (active_cut, _) = self.lane_lifecycle_quadrants();
        let mut to_clear = active_cut.outside;
        let released = to_clear.count_ones() as u8;
        while to_clear != 0 {
            let lane_idx = to_clear.trailing_zeros() as usize;
            to_clear &= to_clear - 1;
            let start = lane_idx * MAX_ATTENTION_HEADS;
            let end = start + MAX_ATTENTION_HEADS;
            self.slot_phase_bank.data[start..end].fill(0);
            self.slot_weights[lane_idx] = 0.0;
        }
        let keep = self.slot_pinned_mask;
        self.slot_enabled_mask &= keep;
        self.slot_repulsion_mask &= keep;
        self.slot_sandbox_mask &= keep;
        released
    }

    /// Releases every workbench lane (including pinned lanes), resetting all 8 lanes to the free
    /// pool and returning the count of previously enabled lanes.
    #[inline]
    pub fn release_all_lanes(&mut self) -> u8 {
        let released = self.slot_enabled_mask.count_ones() as u8;
        self.slot_phase_bank.data.fill(0);
        self.slot_weights = [0.0; MAX_ATTRACTOR_DOMAINS];
        self.slot_enabled_mask = 0;
        self.slot_pinned_mask = 0;
        self.slot_repulsion_mask = 0;
        self.slot_sandbox_mask = 0;
        released
    }

    /// Decays unpinned active workbench lane weights by `retain_factor` (`[0.0, 1.0]`) and
    /// automatically reclaims any lane whose decayed weight drops below `min_weight_floor` (or
    /// reaches `0.0`). Pinned lanes remain untouched. Returns the number of reclaimed lanes.
    #[inline]
    pub fn decay_and_reclaim_lanes(&mut self, retain_factor: f32, min_weight_floor: f32) -> u8 {
        let retain = if branchless_is_finite_f32(retain_factor) {
            branchless_clamp_f32(retain_factor, 0.0, 1.0)
        } else {
            0.0
        };
        let floor = if branchless_is_finite_f32(min_weight_floor) {
            branchless_relu_f32(min_weight_floor)
        } else {
            0.0
        };
        let (active_cut, _) = self.lane_lifecycle_quadrants();
        let mut active_unpinned = active_cut.outside;
        let mut reclaimed = 0u8;
        while active_unpinned != 0 {
            let lane = active_unpinned.trailing_zeros() as u8;
            active_unpinned &= active_unpinned - 1;
            let lane_idx = usize::from(lane);
            let next_w = branchless_flush_subnormal_f32(
                branchless_relu_f32(self.slot_weights[lane_idx]) * retain,
            );
            if next_w <= 0.0 || next_w < floor {
                let lane_bit = 1u8 << lane;
                let start = lane_idx * MAX_ATTENTION_HEADS;
                let end = start + MAX_ATTENTION_HEADS;
                self.slot_phase_bank.data[start..end].fill(0);
                self.slot_weights[lane_idx] = 0.0;
                self.slot_enabled_mask &= !lane_bit;
                self.slot_repulsion_mask &= !lane_bit;
                self.slot_sandbox_mask &= !lane_bit;
                reclaimed = reclaimed.saturating_add(1);
            } else {
                self.slot_weights[lane_idx] = next_w;
            }
        }
        reclaimed
    }

    /// Sets or clears the explicit pin bit (`slot_pinned_mask`) for physical `lane` (`0..=7`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `lane >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn pin_lane(&mut self, lane: u8, pin: bool) -> LmnkResult<()> {
        if usize::from(lane) >= MAX_ATTRACTOR_DOMAINS {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let lane_bit = 1u8 << lane;
        let bit_val = u8::from(pin).wrapping_neg() & lane_bit;
        self.slot_pinned_mask = (self.slot_pinned_mask & !lane_bit) | bit_val;
        Ok(())
    }

    /// Settles an active sandbox workbench lane (`slot_sandbox_mask`) into standard full-aperture
    /// participation in-place without relocating its physical register slot.
    ///
    /// Returns `Ok(true)` if `lane` was an active sandbox lane and is now settled.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `lane >= MAX_ATTRACTOR_DOMAINS`.
    #[inline]
    pub fn settle_sandbox_lane(&mut self, lane: u8) -> LmnkResult<bool> {
        if usize::from(lane) >= MAX_ATTRACTOR_DOMAINS {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let lane_bit = 1u8 << lane;
        let active_sandbox = ((self.slot_enabled_mask & self.slot_sandbox_mask) & lane_bit) != 0;
        if active_sandbox {
            self.slot_sandbox_mask &= !lane_bit;
        }
        Ok(active_sandbox)
    }

    /// Releases all unpinned active sandbox lanes (`slot_enabled_mask & slot_sandbox_mask & !slot_pinned_mask`),
    /// returning the number of sandbox lanes reclaimed.
    #[inline]
    pub fn release_sandbox_lanes(&mut self) -> u8 {
        let (active_cut, _) = self.lane_lifecycle_quadrants();
        let mut targets = BitCut8::split(active_cut.outside, self.slot_sandbox_mask).inside;
        let mut released = 0u8;
        while targets != 0 {
            let lane = targets.trailing_zeros() as u8;
            targets &= targets - 1;
            let lane_idx = usize::from(lane);
            let lane_bit = 1u8 << lane;
            let start = lane_idx * MAX_ATTENTION_HEADS;
            let end = start + MAX_ATTENTION_HEADS;
            self.slot_phase_bank.data[start..end].fill(0);
            self.slot_weights[lane_idx] = 0.0;
            self.slot_enabled_mask &= !lane_bit;
            self.slot_repulsion_mask &= !lane_bit;
            self.slot_sandbox_mask &= !lane_bit;
            released = released.saturating_add(1);
        }
        released
    }

    /// Returns the bitmask of currently enabled lanes (`0..=7`) on the 8-lane working phase bank.
    #[inline(always)]
    #[must_use]
    pub const fn active_lane_mask(&self) -> u8 {
        self.slot_enabled_mask
    }

    /// Returns the bitmask of explicitly pinned workbench lanes (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn pinned_lane_mask(&self) -> u8 {
        self.slot_pinned_mask
    }

    /// Returns the bitmask of anti-phase repulsion workbench lanes (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn repulsion_lane_mask(&self) -> u8 {
        self.slot_repulsion_mask
    }

    /// Returns the bitmask of isolated exterior-only sandbox workbench lanes (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn sandbox_lane_mask(&self) -> u8 {
        self.slot_sandbox_mask
    }

    /// Returns the 4-quadrant lifecycle partition `(active_cut, free_cut)` over all 8 workbench lanes:
    /// - `active_cut.inside`: enabled & pinned
    /// - `active_cut.outside`: enabled & evictable
    /// - `free_cut.inside`: disabled & pinned (reserved)
    /// - `free_cut.outside`: disabled & unpinned (allocatable free)
    #[inline(always)]
    #[must_use]
    pub const fn lane_lifecycle_quadrants(&self) -> (BitCut8, BitCut8) {
        BitCut8::split(0xFF, self.slot_enabled_mask).refine(self.slot_pinned_mask)
    }

    /// Returns the 4-quadrant role partition `(rep_cut, attract_cut)` over enabled workbench lanes:
    /// - `rep_cut.outside`: global repulsion (`enabled & repulsion & !sandbox`)
    /// - `rep_cut.inside`: sandbox-scoped exterior repulsion (`enabled & repulsion & sandbox`)
    /// - `attract_cut.outside`: standard attractive (`enabled & !repulsion & !sandbox`)
    /// - `attract_cut.inside`: pure sandbox hypothesis (`enabled & !repulsion & sandbox`)
    #[inline(always)]
    #[must_use]
    pub const fn lane_role_quadrants(&self) -> (BitCut8, BitCut8) {
        BitCut8::split(self.slot_enabled_mask, self.slot_repulsion_mask)
            .refine(self.slot_sandbox_mask)
    }

    /// Returns the bitmask of workbench lanes currently free for unpinned allocation (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn free_lane_mask(&self) -> u8 {
        self.lane_lifecycle_quadrants().1.outside
    }

    /// Returns the number of currently active workbench lanes (`0..=8`).
    #[inline(always)]
    #[must_use]
    pub const fn lane_utilization_count(&self) -> u8 {
        self.slot_enabled_mask.count_ones() as u8
    }

    /// Atomically refreshes unpinned lanes on the 8-lane working phase workbench with `specs`,
    /// preserving any explicitly pinned lanes while allocating each [`WorkbenchSlotSpec`] via
    /// [`Self::allocate_lane`]. Returns the count of lanes loaded.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any spec has non-finite weight or an explicit
    /// lane `>= MAX_ATTRACTOR_DOMAINS`, or [`LMNK_ERR_SUBSTRATE_FULL`] if an `AutoFree` spec
    /// cannot be satisfied because all 8 lanes are pinned.
    #[inline]
    pub fn apply_preset_lanes(&mut self, specs: &[WorkbenchSlotSpec]) -> LmnkResult<u8> {
        let mut sim_pinned = self.slot_pinned_mask;
        let mut idx = 0usize;
        while idx < specs.len() {
            let spec = &specs[idx];
            if !branchless_is_finite_f32(spec.weight) {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            match spec.placement {
                WorkbenchLanePlacement::Explicit { lane, .. } => {
                    if usize::from(lane) >= MAX_ATTRACTOR_DOMAINS {
                        return Err(LMNK_ERR_INVALID_ARGUMENT);
                    }
                    let bit = 1u8 << lane;
                    if spec.is_pinned() {
                        sim_pinned |= bit;
                    } else {
                        sim_pinned &= !bit;
                    }
                }
                WorkbenchLanePlacement::AutoFree => {
                    if sim_pinned == 0xFF {
                        return Err(LMNK_ERR_SUBSTRATE_FULL);
                    }
                    if spec.is_pinned() {
                        sim_pinned |= 1u8 << (!sim_pinned).trailing_zeros();
                    }
                }
            }
            idx += 1;
        }

        let _ = self.release_unpinned_lanes();
        let mut updated = 0u8;
        let mut load_idx = 0usize;
        while load_idx < specs.len() {
            let _lane = self.allocate_lane(&specs[load_idx])?;
            updated = updated.saturating_add(1);
            load_idx += 1;
        }
        Ok(updated)
    }
}
