// name: src/core/src/drivers/spine/state.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mechanical execution spine state (`SystemSpineState`, 32 B), primitive `pinned`
//! isolation logic (`isolate_slot_and_modes_branchless`), and branchless I-Frame / P-Frame
//! transition evaluation (`evaluate_frame_transition_branchless`).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::mem::{align_of, size_of};

use crate::foundation::{Pod, ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_ID_MASK};
use crate::liminal::{
    branchless_select_u8, AttributionTag, BitCut32, BitCut64, BitCut8, Lacuna,
    THRESHOLD_CONTRADICTION_GUARD, THRESHOLD_SYSTEM_CONSTRAINT,
};
use crate::stigmergy::{ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE};

/// Frame representation kind in I-Frame / P-Frame steering and timeline encoding.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameKind {
    /// Complete keyframe anchor (I-Frame) establishing a reference gauge origin (`0`).
    #[default]
    Keyframe = 0,
    /// Incremental differential residual (P-Frame) relative to a prior keyframe (`1`).
    Delta = 1,
}

/// Action directive emitted by [`SystemSpineState::evaluate_frame_transition_branchless`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameDirective {
    /// Zero-aperture or stationary-ready skip; emit neither keyframe nor delta (`0`).
    #[default]
    Skip = 0,
    /// Mechanical pass-through locked by irreversible spine guards or OOD aperture locks (`1`).
    PassThrough = 1,
    /// Emit an incremental P-Frame differential update on allowed non-pinned granules (`2`).
    EmitDelta = 2,
    /// Re-anchor a fresh I-Frame keyframe on GOP expiry, exterior shift, new basin, or guard repulsion (`3`).
    ReAnchorKeyframe = 3,
    /// Reject non-finite [`Lacuna::Undefined`] measurement (`4`).
    RejectUndefined = 4,
}

/// Spine transition input flag (`bit 0`): current step is irreversible (e.g., prompt Prefill).
pub const SPINE_TRANSITION_FLAG_IRREVERSIBLE: u8 = 1 << 0;
/// Spine transition input flag (`bit 1`): contradiction guard or repulsion escape fired.
pub const SPINE_TRANSITION_FLAG_GUARD_REPULSION: u8 = 1 << 1;
/// Spine transition input flag (`bit 2`): a new attractor basin crystallized on this step.
pub const SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED: u8 = 1 << 2;
/// Spine transition input flag (`bit 3`): stationary self-reference shortcut is ready.
pub const SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY: u8 = 1 << 3;

/// Compact 8-byte (`#[repr(C, align(8))]`) transition input evaluated branchlessly by
/// [`SystemSpineState::evaluate_frame_transition_branchless`].
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpineFrameTransitionInput {
    /// Encoded [`Lacuna`] discriminant (`0=Unobserved`, `1=SelfRef`, `2=Exterior`, `3=Undefined`, `offset 0`).
    pub lacuna_code: u8,
    /// Quantized L1 phase shift relative to the reference keyframe in Q0.7 (`0..=127`, `offset 1`).
    pub phase_delta_q7: u8,
    /// Re-anchor phase delta threshold in Q0.7 (`0` disables delta-based re-anchoring, `offset 2`).
    pub reanchor_delta_thresh_q7: u8,
    /// Number of delta steps elapsed since the last I-Frame keyframe (`offset 3`).
    pub steps_since_keyframe: u8,
    /// Maximum GOP length before re-anchoring (`0` disables GOP-length expiry, `offset 4`).
    pub max_gop_steps: u8,
    /// Candidate 8-granule modification mask prior to `pinned_slot_mask` isolation (`offset 5`).
    pub candidate_granule_mask: u8,
    /// Bitwise OR of `SPINE_TRANSITION_FLAG_*` conditions (`offset 6`).
    pub flags: u8,
    /// Explicit C alignment padding (`offset 7`).
    pub reserved: u8,
}

const _: () = {
    assert!(size_of::<SpineFrameTransitionInput>() == 8);
    assert!(align_of::<SpineFrameTransitionInput>() == 8);
};

impl SpineFrameTransitionInput {
    /// Builds a [`SpineFrameTransitionInput`] branchlessly from a 16-byte [`crate::liminal::LacunaPartition8`].
    #[inline(always)]
    #[must_use]
    pub const fn from_lacuna_partition(
        partition: &crate::liminal::LacunaPartition8,
        phase_delta_q7: u8,
        reanchor_delta_thresh_q7: u8,
        steps_since_keyframe: u8,
        max_gop_steps: u8,
        flags: u8,
    ) -> Self {
        let mut lacuna_code = Lacuna::Unobserved.code();
        if partition.self_ref_mask != 0 {
            lacuna_code = Lacuna::SelfRef.code();
        }
        if partition.exterior_mask != 0 {
            lacuna_code = Lacuna::Exterior.code();
        }
        if partition.undefined_mask != 0 {
            lacuna_code = Lacuna::Undefined.code();
        }
        Self {
            lacuna_code,
            phase_delta_q7,
            reanchor_delta_thresh_q7,
            steps_since_keyframe,
            max_gop_steps,
            candidate_granule_mask: partition.self_ref_mask | partition.exterior_mask,
            flags,
            reserved: 0,
        }
    }
}

// SAFETY: `SpineFrameTransitionInput` is `#[repr(C, align(8))]` of exact size 8 bytes with primitive `u8` fields only.
unsafe impl Pod for SpineFrameTransitionInput {}

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
    /// Non-zero when Unobserved boundary ($\overline{S \cup O}$) forces aperture-0 lock (`offset 6`).
    pub unobserved_locked: u8,
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
            unobserved_locked: 0,
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
        let prev_cut = BitCut32::split(prev_aperture_mask, self.pinned_aperture_mask);
        let next_cut = BitCut32::split(next_aperture_mask, self.pinned_aperture_mask);
        next_cut.outside | prev_cut.inside
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
        let slot_is_pinned = BitCut8::split(self.pinned_slot_mask, 1u8 << masked_base).inside != 0;
        let eff_slot = branchless_select_u8(slot_is_pinned, base_slot, ego_slot);
        let preserved_modes = BitCut8::split(base_modes, self.pinned_action_mode_mask).inside;
        let eff_modes = ego_modes | preserved_modes;
        let full_lock = (self.unobserved_locked != 0) | (self.irreversible_step_lock != 0);
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
    /// Explicitly excludes `unobserved_locked` so OOD (`Lacuna::Unobserved`) does not trigger
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

    /// Evaluates branchless whole-token bypass veto flags (`SPINE_BYPASS_VETO_*`) against an
    /// anonymous compute pool's `reversible_mask`.
    ///
    /// Unlike [`Self::evaluate_whole_token_bypass_veto_branchless`], stationary boundary pins
    /// outside `reversible_mask` (such as permanent head/tail anchors) protect their own slots
    /// from layer bypass without vetoing calm whole-token bypass; only `irreversible_step_lock`
    /// or dynamic pins overlapping `reversible_mask` raise `SPINE_BYPASS_VETO_PINNED_LOCK`.
    #[inline(always)]
    #[must_use]
    pub const fn evaluate_pool_bypass_veto_branchless(
        &self,
        reversible_mask: u64,
        guard_active: bool,
        repulsion_active: bool,
    ) -> u8 {
        let g = (guard_active as u8) * SPINE_BYPASS_VETO_GUARD_ACTIVE;
        let r = (repulsion_active as u8) * SPINE_BYPASS_VETO_REPULSION_ACTIVE;
        let dynamic_pin_overlap =
            BitCut64::split(self.pinned_compute_mask, reversible_mask).inside != 0;
        let pinned = (self.irreversible_step_lock != 0) | dynamic_pin_overlap;
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
        BitCut64::split(raw_bypass_mask, self.pinned_compute_mask).outside & allow_mask
    }

    /// Evaluates an I-Frame / P-Frame transition branchlessly (`0 alloc`, `0 fdiv`), returning
    /// `(directive, allowed_granule_mask)`.
    ///
    /// Priority order (highest to lowest):
    /// 1. [`Lacuna::Undefined`] (`>= 3`) -> [`FrameDirective::RejectUndefined`] (`allowed = 0`).
    /// 2. Spine lock (`irreversible_step_lock`, `SPINE_TRANSITION_FLAG_IRREVERSIBLE`, or `unobserved_locked`)
    ///    -> [`FrameDirective::PassThrough`] (`allowed = 0`).
    /// 3. Defined observed re-anchor (`!Unobserved` and (`max_gop_steps` expired, `NEW_BASIN_SPAWNED`,
    ///    `GUARD_REPULSION`, [`Lacuna::Exterior`], or `phase_delta_q7 >= reanchor_delta_thresh_q7`))
    ///    -> [`FrameDirective::ReAnchorKeyframe`] (`allowed = non_pinned`).
    /// 4. [`Lacuna::Unobserved`] (`0`) or (`SelfRef` with `STATIONARY_BYPASS_READY`)
    ///    -> [`FrameDirective::Skip`] (`allowed = 0`; prevents OOD queries from spawning spurious keyframes).
    /// 5. Otherwise -> [`FrameDirective::EmitDelta`] (`allowed = non_pinned`).
    #[inline]
    #[must_use]
    pub const fn evaluate_frame_transition_branchless(
        &self,
        input: SpineFrameTransitionInput,
    ) -> (FrameDirective, u8) {
        let is_unobserved = input.lacuna_code == Lacuna::Unobserved.code();
        let is_self_ref = input.lacuna_code == Lacuna::SelfRef.code();
        let is_exterior = input.lacuna_code == Lacuna::Exterior.code();
        let is_undefined = input.lacuna_code >= Lacuna::Undefined.code();

        let irreversible = (self.irreversible_step_lock != 0)
            | ((input.flags & SPINE_TRANSITION_FLAG_IRREVERSIBLE) != 0);
        let ood_locked = self.unobserved_locked != 0;
        let guard_repulsion = (input.flags & SPINE_TRANSITION_FLAG_GUARD_REPULSION) != 0;
        let new_basin = (input.flags & SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED) != 0;
        let stationary_ready = (input.flags & SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY) != 0;

        let gop_expired =
            (input.max_gop_steps > 0) & (input.steps_since_keyframe >= input.max_gop_steps);
        let delta_exceeded = (!is_unobserved)
            & (input.reanchor_delta_thresh_q7 > 0)
            & (input.phase_delta_q7 >= input.reanchor_delta_thresh_q7);

        let want_reanchor = (!is_unobserved)
            & (gop_expired | new_basin | guard_repulsion | is_exterior | delta_exceeded);
        let want_skip = is_unobserved | (is_self_ref & stationary_ready);

        let mut dir = FrameDirective::EmitDelta as u8;
        dir = branchless_select_u8(want_skip, FrameDirective::Skip as u8, dir);
        dir = branchless_select_u8(want_reanchor, FrameDirective::ReAnchorKeyframe as u8, dir);
        dir = branchless_select_u8(
            irreversible | ood_locked,
            FrameDirective::PassThrough as u8,
            dir,
        );
        dir = branchless_select_u8(is_undefined, FrameDirective::RejectUndefined as u8, dir);

        let non_pinned =
            BitCut8::split(input.candidate_granule_mask, self.pinned_slot_mask).outside;
        let allow_mod = (dir == FrameDirective::EmitDelta as u8)
            | (dir == FrameDirective::ReAnchorKeyframe as u8);
        let allowed_granule_mask = branchless_select_u8(allow_mod, non_pinned, 0);

        let directive = match dir {
            0 => FrameDirective::Skip,
            1 => FrameDirective::PassThrough,
            2 => FrameDirective::EmitDelta,
            3 => FrameDirective::ReAnchorKeyframe,
            _ => FrameDirective::RejectUndefined,
        };
        (directive, allowed_granule_mask)
    }
}

// SAFETY: `SystemSpineState` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for SystemSpineState {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_frame_transition_branchless_enforces_priority_and_ood_isolation() {
        let mut spine = SystemSpineState::new();
        let baseline_mask = 0xFF_u8;
        let expected_non_pinned = baseline_mask & !spine.pinned_slot_mask;

        // 1. Nominal SelfRef -> EmitDelta with pinned granules masked out.
        let nominal = SpineFrameTransitionInput {
            lacuna_code: Lacuna::SelfRef.code(),
            phase_delta_q7: 8,
            reanchor_delta_thresh_q7: 32,
            steps_since_keyframe: 2,
            max_gop_steps: 16,
            candidate_granule_mask: baseline_mask,
            flags: 0,
            reserved: 0,
        };
        assert_eq!(
            spine.evaluate_frame_transition_branchless(nominal),
            (FrameDirective::EmitDelta, expected_non_pinned)
        );
        assert_eq!(expected_non_pinned & spine.pinned_slot_mask, 0);

        // 2. Stationary shortcut on SelfRef -> Skip with 0 mask.
        let stationary = SpineFrameTransitionInput {
            flags: SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY,
            ..nominal
        };
        assert_eq!(
            spine.evaluate_frame_transition_branchless(stationary),
            (FrameDirective::Skip, 0)
        );

        // 3. ReAnchorKeyframe overrides stationary skip on GOP expiry, phase delta, new basin,
        //    guard repulsion, or Lacuna::Exterior.
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                steps_since_keyframe: 16,
                ..stationary
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                phase_delta_q7: 32,
                ..stationary
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                flags: SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED
                    | SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY,
                ..nominal
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                flags: SPINE_TRANSITION_FLAG_GUARD_REPULSION,
                ..nominal
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                lacuna_code: Lacuna::Exterior.code(),
                ..nominal
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );

        // 4. Unobserved OOD zero-aperture MUST return Skip and never spawn spurious keyframes
        //    even if delta/GOP/basin flags are set.
        let ood_noisy = SpineFrameTransitionInput {
            lacuna_code: Lacuna::Unobserved.code(),
            phase_delta_q7: 120,
            steps_since_keyframe: 20,
            flags: SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED | SPINE_TRANSITION_FLAG_GUARD_REPULSION,
            ..nominal
        };
        assert_eq!(
            spine.evaluate_frame_transition_branchless(ood_noisy),
            (FrameDirective::Skip, 0)
        );

        // 5. Irreversible flag or spine lock forces PassThrough above ReAnchorKeyframe/Skip.
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                lacuna_code: Lacuna::Exterior.code(),
                flags: SPINE_TRANSITION_FLAG_IRREVERSIBLE,
                ..nominal
            }),
            (FrameDirective::PassThrough, 0)
        );
        spine.unobserved_locked = 1;
        assert_eq!(
            spine.evaluate_frame_transition_branchless(nominal),
            (FrameDirective::PassThrough, 0)
        );

        // 6. Lacuna::Undefined dominates everything -> RejectUndefined.
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                lacuna_code: Lacuna::Undefined.code(),
                ..nominal
            }),
            (FrameDirective::RejectUndefined, 0)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                lacuna_code: 255,
                ..nominal
            }),
            (FrameDirective::RejectUndefined, 0)
        );
    }
}
