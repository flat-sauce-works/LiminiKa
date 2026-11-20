// name: src/core/src/drivers/spine/state.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mechanical execution spine state (`SystemSpineState`, 32 B) and primitive `pinned`
//! isolation logic (`isolate_slot_and_modes_branchless`).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use crate::foundation::{Pod, ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_ID_MASK};
use crate::liminal::{
    branchless_select_u8, AttributionTag, THRESHOLD_CONTRADICTION_GUARD,
    THRESHOLD_SYSTEM_CONSTRAINT,
};
use crate::stigmergy::{ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE};

// 3. System Spine State & Physical `pinned` Isolation (32 B)
// ===================================================================

/// No spine bypass veto active (`0x00`).
pub const SPINE_BYPASS_VETO_NONE: u8 = 0x00;
/// Bypass vetoed because a contradiction guard boundary is active (`1 << 0`).
pub const SPINE_BYPASS_VETO_GUARD_ACTIVE: u8 = 1 << 0;
/// Bypass vetoed because phase-conjugate repulsion is active (`1 << 1`).
pub const SPINE_BYPASS_VETO_REPULSION_ACTIVE: u8 = 1 << 1;
/// Bypass vetoed because an irreversible step lock or pinned compute mask is active (`1 << 2`).
pub const SPINE_BYPASS_VETO_PINNED_LOCK: u8 = 1 << 2;

/// Mechanical execution spine (`32 B`, `#[repr(C, align(16))]`, `DriverKind::System` = I-Frame).
///
/// Owns physical `pinned` masks isolating irreversible steps, memory safety, OOD aperture-0 locks,
/// and explicit contradiction guards so `EgoDifferentialNet` masks/XORs cannot reach them.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemSpineState {
    /// Physical `Liminal` aperture bits protected from Ego override (`offset 0..4`).
    pub pinned_aperture_mask: u32,
    /// Physical workbench slot bits protected from Ego deflection (`offset 4`).
    pub pinned_slot_mask: u8,
    /// Physical `ACTION_MODE_*` bits protected from Ego suppression (`offset 5`).
    pub pinned_action_mode_mask: u8,
    /// Non-zero when True Void ($\overline{S \cup O}$) forces aperture-0 lock (`offset 6`).
    pub ood_void_locked: u8,
    /// Non-zero during irreversible Prefill / safety checkpoints (`offset 7`).
    pub irreversible_step_lock: u8,
    /// Physical compute slot bitmask protected from Fast-Loop bypass (`offset 8..16`).
    pub pinned_compute_mask: u64,
    /// Monotonic gauge re-anchoring I-Frame epoch counter (`offset 16..20`).
    pub gauge_anchor_epoch: u32,
    /// Monotonic Slow-Loop epoch counter (`offset 20..24`).
    pub slow_loop_epoch: u32,
    /// Non-zero when token-level whole-steering bypass is active (`0` D2H syncs, `offset 24`).
    pub whole_token_bypass_active: u8,
    /// Consecutive calm steps accumulated toward pre-inference bypass (`offset 25`).
    pub bypass_dwell_steps: u8,
    /// Remaining recovery cooldown steps before bypass may re-engage (`offset 26`).
    pub bypass_cooldown_steps: u8,
    /// 2-bit [`AttributionTag`] (`Liminal`/`System`, `offset 27`).
    pub attribution_tag: u8,
    /// Explicit C alignment padding (`offset 28..32`).
    pub reserved_align: [u8; 4],
}

impl Default for SystemSpineState {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl SystemSpineState {
    /// Constructs a canonical [`SystemSpineState`] locking SystemConstraint, ContradictionGuard,
    /// and `ACTION_MODE_PHASE_REPULSE` in physical `pinned` storage.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pinned_aperture_mask: THRESHOLD_SYSTEM_CONSTRAINT | THRESHOLD_CONTRADICTION_GUARD,
            pinned_slot_mask: (1u8 << ATTRACTOR_SLOT_SYSTEM_BASELINE)
                | (1u8 << ATTRACTOR_SLOT_REPULSION),
            pinned_action_mode_mask: ACTION_MODE_PHASE_REPULSE,
            ood_void_locked: 0,
            irreversible_step_lock: 0,
            pinned_compute_mask: 0,
            gauge_anchor_epoch: 0,
            slow_loop_epoch: 0,
            whole_token_bypass_active: 0,
            bypass_dwell_steps: 0,
            bypass_cooldown_steps: 0,
            attribution_tag: AttributionTag::LIMINAL_SYSTEM.raw(),
            reserved_align: [0; 4],
        }
    }

    /// Physical `pinned` isolation barrier ensuring Ego modulation never mutates pinned apertures.
    #[inline(always)]
    #[must_use]
    pub const fn mask_ego_aperture_edits(
        &self,
        prev_aperture_mask: u32,
        next_aperture_mask: u32,
    ) -> u32 {
        (next_aperture_mask & !self.pinned_aperture_mask)
            | (prev_aperture_mask & self.pinned_aperture_mask)
    }

    /// Primitive `pinned` isolation helper returning `(full_lock, effective_slot, effective_modes)`
    /// without depending on upper-layer `GatewayFrame`.
    #[inline(always)]
    #[must_use]
    pub fn isolate_slot_and_modes_branchless(
        &self,
        base_slot: u8,
        base_modes: u8,
        ego_slot: u8,
        ego_modes: u8,
    ) -> (bool, u8, u8) {
        let masked_base = base_slot & ATTRACTOR_SLOT_ID_MASK;
        let slot_is_pinned = (self.pinned_slot_mask & (1u8 << masked_base)) != 0;
        let eff_slot = branchless_select_u8(slot_is_pinned, base_slot, ego_slot);
        let preserved_modes = base_modes & self.pinned_action_mode_mask;
        let eff_modes = ego_modes | preserved_modes;
        let full_lock = (self.ood_void_locked != 0) | (self.irreversible_step_lock != 0);
        (full_lock, eff_slot, eff_modes)
    }

    /// Pins compute slots in `pinned_compute_mask` so Fast-Loop bypass cannot skip them.
    #[inline(always)]
    pub const fn pin_compute_slots(&mut self, slot_mask: u64) {
        self.pinned_compute_mask |= slot_mask;
    }

    /// Unpins compute slots from `pinned_compute_mask`.
    #[inline(always)]
    pub const fn unpin_compute_slots(&mut self, slot_mask: u64) {
        self.pinned_compute_mask &= !slot_mask;
    }

    /// Clears all bits in `pinned_compute_mask`.
    #[inline(always)]
    pub const fn clear_pinned_compute_slots(&mut self) {
        self.pinned_compute_mask = 0;
    }

    /// Evaluates branchless whole-token bypass veto flags (`SPINE_BYPASS_VETO_*`).
    ///
    /// Explicitly excludes `ood_void_locked` so OOD (`Lacuna::Void`) does not trigger
    /// guard/repulsion bypass vetoes.
    #[inline(always)]
    #[must_use]
    pub const fn evaluate_whole_token_bypass_veto_branchless(
        &self,
        guard_active: bool,
        repulsion_active: bool,
    ) -> u8 {
        let g = (guard_active as u8) * SPINE_BYPASS_VETO_GUARD_ACTIVE;
        let r = (repulsion_active as u8) * SPINE_BYPASS_VETO_REPULSION_ACTIVE;
        let pinned = (self.irreversible_step_lock != 0) | (self.pinned_compute_mask != 0);
        let p = (pinned as u8) * SPINE_BYPASS_VETO_PINNED_LOCK;
        g | r | p
    }

    /// Returns `true` only when `candidate_bypass` is requested and no spine bypass veto is set.
    #[inline(always)]
    #[must_use]
    pub const fn should_bypass_whole_token_branchless(
        &self,
        candidate_bypass: bool,
        guard_active: bool,
        repulsion_active: bool,
    ) -> bool {
        candidate_bypass
            & (self.evaluate_whole_token_bypass_veto_branchless(guard_active, repulsion_active)
                == SPINE_BYPASS_VETO_NONE)
    }

    /// Masks a candidate layer bypass bitmask against `pinned_compute_mask` and `veto_mask`
    /// without branches: returns `0` when `veto_mask != 0`, else `raw_bypass_mask & !pinned`.
    #[inline(always)]
    #[must_use]
    pub const fn mask_layer_bypass_with_pinned(&self, raw_bypass_mask: u64, veto_mask: u8) -> u64 {
        let allow_mask = ((veto_mask == SPINE_BYPASS_VETO_NONE) as u64).wrapping_neg();
        (raw_bypass_mask & !self.pinned_compute_mask) & allow_mask
    }
}

// SAFETY: `SystemSpineState` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for SystemSpineState {}
