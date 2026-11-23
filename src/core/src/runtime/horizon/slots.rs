// name: src/core/src/runtime/horizon/slots.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic slide-in/slide-out [`LatentHorizonSlot`] boundary contracts,
//! [`HorizonEscalationProvider`], dwell anti-chattering state, and [`StigmergicSubstrate`] escalation.

use crate::liminal::{
    BitCut8, LmnkResult, PhaseQ7, LMNK_ERR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS,
};
use crate::runtime::context::CoreStateStack;
use crate::runtime::gateway::{GatewayFrame, ROOT_CAUSE_NOMINAL};
use crate::runtime::topology::CORE_TOPOLOGY;
use crate::stigmergy::{
    StigmergicSubstrate, WorkbenchLanePlacement, WorkbenchSlotSpec, WORKBENCH_LANE_FLAG_PINNED,
};

/// Consecutive-step hysteresis tracker for dynamic [`LatentHorizonSlot`] slide-in / slide-out.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HorizonDwellState {
    /// Consecutive steps during which horizon destabilization persisted.
    pub unstable_dwell_steps: u16,
    /// Consecutive steps during which boundaries stayed quiescent and ready for retraction.
    pub quiescent_dwell_steps: u16,
}

impl HorizonDwellState {
    /// Advances dwell hysteresis counters branchlessly and returns `(unstable, quiescent)`.
    #[inline(always)]
    pub fn step_branchless(&mut self, destabilized: bool, _stationary_ready: bool) -> (u16, u16) {
        let next_unstable = self.unstable_dwell_steps.saturating_add(1);
        let next_quiescent = self.quiescent_dwell_steps.saturating_add(1);
        let unstable_out = if destabilized { next_unstable } else { 0 };
        let quiescent_out = if !destabilized { next_quiescent } else { 0 };
        self.unstable_dwell_steps = unstable_out;
        self.quiescent_dwell_steps = quiescent_out;
        (unstable_out, quiescent_out)
    }
}

/// Branchless diagnostic snapshot evaluated by the single orchestrator to decide whether
/// dormant [`LatentHorizonSlot`] boundaries should slide into or retract from the active
/// cognitive window.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HorizonTriggerInput {
    /// Active `EntropyController` branch mode (`0` = normal, `1` = bifurcation, `2` = repulsion).
    pub entropy_gate_mode: u32,
    /// Latest moving Z-score normalized attention entropy ($\tilde{H}$).
    pub moving_z_entropy: f32,
    /// Latest repulsion severity observed on the synthesized [`GatewayFrame`] in $[0.0, 1.0]$.
    pub repulsion_severity: f32,
    /// Dominant causal root-cause code (`ROOT_CAUSE_*`).
    pub causal_root_cause: u8,
    /// Active [`crate::liminal::EpistemicRegime`] byte (`0` = Coherent, `1` = InternalConflict, `2` = Liminal).
    pub epistemic_regime: u8,
    /// `true` when Layer 0 boundaries are quiescent without chattering (`stationary_shortcut_ready != 0`).
    pub stationary_shortcut_ready: bool,
    /// Bitmask of `EGO_TRIGGER_*` flags that fired on the latest Layer 2 Ego evaluation.
    pub ego_active_triggers: u8,
    /// Consecutive destabilized steps observed by the orchestrator (`0` = single-shot default).
    pub unstable_dwell_steps: u16,
    /// Consecutive quiescent steps observed by the orchestrator (`0` = single-shot default).
    pub quiescent_dwell_steps: u16,
}

impl HorizonTriggerInput {
    /// Returns `true` when current concrete state indicates instability, conflict, or horizon
    /// excursion requiring dormant latent horizon boundaries to slide in.
    #[inline(always)]
    #[must_use]
    pub const fn is_destabilized(&self) -> bool {
        (self.entropy_gate_mode != 0)
            | (self.repulsion_severity > 0.25)
            | (self.epistemic_regime != 0)
            | (self.ego_active_triggers != 0)
            | (self.causal_root_cause != ROOT_CAUSE_NOMINAL)
    }

    /// Returns `true` when boundaries are quiescent and coherent enough to retract optional
    /// slide-in slots back to zero runtime overhead.
    #[inline(always)]
    #[must_use]
    pub const fn can_retract(&self) -> bool {
        !self.is_destabilized()
    }

    /// Returns effective destabilization dwell steps (promoting explicit destabilization to `1`
    /// when single-shot callers leave `unstable_dwell_steps == 0`).
    #[inline(always)]
    #[must_use]
    pub const fn effective_unstable_dwell(&self) -> u16 {
        if self.unstable_dwell_steps > 0 {
            self.unstable_dwell_steps
        } else if self.is_destabilized() {
            1
        } else {
            0
        }
    }

    /// Returns effective quiescent dwell steps (promoting explicit `can_retract()` to `1`
    /// when single-shot callers leave `quiescent_dwell_steps == 0`).
    #[inline(always)]
    #[must_use]
    pub const fn effective_quiescent_dwell(&self) -> u16 {
        if self.quiescent_dwell_steps > 0 {
            self.quiescent_dwell_steps
        } else if self.can_retract() {
            1
        } else {
            0
        }
    }
}

/// Thin latent boundary trait for outer capabilities (auxiliary attractors, contradiction guards,
/// sensory channels, or Ego overrides) that remain dormant until the orchestrator dynamically
/// slides them into the active cognitive aperture on instability or horizon excursion.
pub trait LatentHorizonSlot {
    /// Unique bitmask (`1 << k`) identifying this slot inside `active_horizon_slots_mask`.
    fn horizon_slot_mask(&self) -> u32;

    /// Returns the 8-bit `BitCut8` admission token evaluated against [`crate::runtime::CORE_TOPOLOGY`].
    #[inline(always)]
    fn horizon_admission_cut(&self) -> BitCut8 {
        BitCut8::split((self.horizon_slot_mask() & 0xFF) as u8, 0xFF)
    }

    /// Minimum consecutive destabilized steps required before sliding this slot in (default `1`).
    #[inline(always)]
    fn min_slide_in_dwell(&self) -> u16 {
        1
    }

    /// Minimum consecutive quiescent steps required before retracting this slot out (default `1`).
    #[inline(always)]
    fn min_slide_out_dwell(&self) -> u16 {
        1
    }

    /// Returns `true` when this boundary should slide into the active cognitive window.
    #[inline(always)]
    fn should_slide_in(&self, trigger: &HorizonTriggerInput) -> bool {
        trigger.is_destabilized()
    }

    /// Returns `true` when this boundary can safely retract out of the active cognitive window.
    #[inline(always)]
    fn should_slide_out(&self, trigger: &HorizonTriggerInput) -> bool {
        trigger.can_retract()
    }

    /// Applies zero-allocation modulation to `frame` and the 3-layer core `stack` while active.
    fn on_slide_in_modulate(
        &mut self,
        frame: &mut GatewayFrame,
        stack: &mut CoreStateStack,
        trigger: &HorizonTriggerInput,
    );

    /// Performs optional zero-allocation cleanup on `frame` and `stack` when sliding back out.
    #[inline(always)]
    fn on_slide_out(
        &mut self,
        _frame: &mut GatewayFrame,
        _stack: &mut CoreStateStack,
        _trigger: &HorizonTriggerInput,
    ) {
    }

    /// Evaluates this slot (or nested tuple of slots) against `trigger` and `active_mask`,
    /// returning per-slot `(slid_in_bits, slid_out_bits)`.
    #[inline]
    fn apply_horizon_slot(
        &mut self,
        active_mask: &mut u32,
        trigger: &HorizonTriggerInput,
        frame: &mut GatewayFrame,
        stack: &mut CoreStateStack,
    ) -> (u32, u32) {
        let slot_bit = self.horizon_slot_mask();
        if slot_bit == 0 {
            return (0, 0);
        }
        let was_active = (*active_mask & slot_bit) == slot_bit;
        let channel_idx = slot_bit.trailing_zeros() as usize;
        let admitted = (slot_bit <= 0xFF)
            & CORE_TOPOLOGY.admits_channel_branchless(channel_idx, self.horizon_admission_cut());
        if !admitted {
            if was_active {
                *active_mask &= !slot_bit;
                self.on_slide_out(frame, stack, trigger);
                return (0, slot_bit);
            }
            return (0, 0);
        }
        let enter_ready = self.should_slide_in(trigger)
            && trigger.effective_unstable_dwell() >= self.min_slide_in_dwell();
        if enter_ready {
            *active_mask |= slot_bit;
            self.on_slide_in_modulate(frame, stack, trigger);
            let slid_in = if was_active { 0 } else { slot_bit };
            return (slid_in, 0);
        }
        let exit_ready = self.should_slide_out(trigger)
            && trigger.effective_quiescent_dwell() >= self.min_slide_out_dwell();
        if exit_ready {
            *active_mask &= !slot_bit;
            if was_active {
                self.on_slide_out(frame, stack, trigger);
                return (0, slot_bit);
            }
            return (0, 0);
        }
        if (*active_mask & slot_bit) != 0 {
            self.on_slide_in_modulate(frame, stack, trigger);
        }
        (0, 0)
    }
}

impl<T: LatentHorizonSlot + ?Sized> LatentHorizonSlot for &mut T {
    #[inline(always)]
    fn horizon_slot_mask(&self) -> u32 {
        (**self).horizon_slot_mask()
    }

    #[inline(always)]
    fn horizon_admission_cut(&self) -> BitCut8 {
        (**self).horizon_admission_cut()
    }

    #[inline(always)]
    fn min_slide_in_dwell(&self) -> u16 {
        (**self).min_slide_in_dwell()
    }

    #[inline(always)]
    fn min_slide_out_dwell(&self) -> u16 {
        (**self).min_slide_out_dwell()
    }

    #[inline(always)]
    fn should_slide_in(&self, trigger: &HorizonTriggerInput) -> bool {
        (**self).should_slide_in(trigger)
    }

    #[inline(always)]
    fn should_slide_out(&self, trigger: &HorizonTriggerInput) -> bool {
        (**self).should_slide_out(trigger)
    }

    #[inline(always)]
    fn on_slide_in_modulate(
        &mut self,
        frame: &mut GatewayFrame,
        stack: &mut CoreStateStack,
        trigger: &HorizonTriggerInput,
    ) {
        (**self).on_slide_in_modulate(frame, stack, trigger);
    }

    #[inline(always)]
    fn on_slide_out(
        &mut self,
        frame: &mut GatewayFrame,
        stack: &mut CoreStateStack,
        trigger: &HorizonTriggerInput,
    ) {
        (**self).on_slide_out(frame, stack, trigger);
    }

    #[inline]
    fn apply_horizon_slot(
        &mut self,
        active_mask: &mut u32,
        trigger: &HorizonTriggerInput,
        frame: &mut GatewayFrame,
        stack: &mut CoreStateStack,
    ) -> (u32, u32) {
        (**self).apply_horizon_slot(active_mask, trigger, frame, stack)
    }
}

impl LatentHorizonSlot for () {
    #[inline(always)]
    fn horizon_slot_mask(&self) -> u32 {
        0
    }

    #[inline(always)]
    fn should_slide_in(&self, _trigger: &HorizonTriggerInput) -> bool {
        false
    }

    #[inline(always)]
    fn should_slide_out(&self, _trigger: &HorizonTriggerInput) -> bool {
        false
    }

    #[inline(always)]
    fn on_slide_in_modulate(
        &mut self,
        _frame: &mut GatewayFrame,
        _stack: &mut CoreStateStack,
        _trigger: &HorizonTriggerInput,
    ) {
    }
}

impl<A: LatentHorizonSlot, B: LatentHorizonSlot> LatentHorizonSlot for (A, B) {
    #[inline(always)]
    fn horizon_slot_mask(&self) -> u32 {
        self.0.horizon_slot_mask() | self.1.horizon_slot_mask()
    }

    #[inline(always)]
    fn should_slide_in(&self, trigger: &HorizonTriggerInput) -> bool {
        self.0.should_slide_in(trigger) | self.1.should_slide_in(trigger)
    }

    #[inline(always)]
    fn should_slide_out(&self, trigger: &HorizonTriggerInput) -> bool {
        self.0.should_slide_out(trigger) & self.1.should_slide_out(trigger)
    }

    #[inline(always)]
    fn on_slide_in_modulate(
        &mut self,
        frame: &mut GatewayFrame,
        stack: &mut CoreStateStack,
        trigger: &HorizonTriggerInput,
    ) {
        if self.0.should_slide_in(trigger) {
            self.0.on_slide_in_modulate(frame, stack, trigger);
        }
        if self.1.should_slide_in(trigger) {
            self.1.on_slide_in_modulate(frame, stack, trigger);
        }
    }

    #[inline(always)]
    fn on_slide_out(
        &mut self,
        frame: &mut GatewayFrame,
        stack: &mut CoreStateStack,
        trigger: &HorizonTriggerInput,
    ) {
        self.0.on_slide_out(frame, stack, trigger);
        self.1.on_slide_out(frame, stack, trigger);
    }

    #[inline]
    fn apply_horizon_slot(
        &mut self,
        active_mask: &mut u32,
        trigger: &HorizonTriggerInput,
        frame: &mut GatewayFrame,
        stack: &mut CoreStateStack,
    ) -> (u32, u32) {
        let (in_a, out_a) = self
            .0
            .apply_horizon_slot(active_mask, trigger, frame, stack);
        let (in_b, out_b) = self
            .1
            .apply_horizon_slot(active_mask, trigger, frame, stack);
        (in_a | in_b, out_a | out_b)
    }
}

/// Evaluates `slots` against `trigger`, updating `active_mask` (`slide-in` / `slide-out`) per
/// individual slot and modulating `frame` + `stack` in zero-allocation mode. Returns
/// `(slid_in_bits, slid_out_bits)`.
#[inline]
pub fn apply_latent_horizon_slots<H: LatentHorizonSlot + ?Sized>(
    slots: &mut H,
    active_mask: &mut u32,
    trigger: &HorizonTriggerInput,
    frame: &mut GatewayFrame,
    stack: &mut CoreStateStack,
) -> (u32, u32) {
    slots.apply_horizon_slot(active_mask, trigger, frame, stack)
}

/// Request descriptor passed to an external [`HorizonEscalationProvider`] when Hot-Path
/// concrete engines and latent slots detect unresolved instability or epistemic excursion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HorizonEscalationRequest {
    /// Stability / anomaly snapshot that triggered escalation.
    pub trigger: HorizonTriggerInput,
    /// Placement policy (`AutoFree` or `Explicit { lane, pin }`) on the 8-lane workbench pool.
    pub placement: WorkbenchLanePlacement,
    /// Optional `WORKBENCH_LANE_FLAG_*` bitmask applied when the escalated lane is allocated.
    pub lane_flags: u8,
    /// Normalized depth scale along the model hierarchy (`0..=127`, where `0` = shallowest, `127` = deepest).
    pub layer_scale_q7: u8,
    /// Active Layer 0 cognitive aperture mask at the time of escalation.
    pub desired_aperture_mask: u32,
    /// Current effective energy on the synthesized [`GatewayFrame`].
    pub current_frame_energy: f32,
}

impl Default for HorizonEscalationRequest {
    #[inline(always)]
    fn default() -> Self {
        Self {
            trigger: HorizonTriggerInput::default(),
            placement: WorkbenchLanePlacement::AutoFree,
            lane_flags: 0,
            layer_scale_q7: 0,
            desired_aperture_mask: 0,
            current_frame_energy: 0.0,
        }
    }
}

/// Cold-path boundary trait allowing the orchestrator to pull external phases (e.g. from NVMe
/// `.lmnk` `MappedStorage` containers or `std` corpus recall) into an anonymous workbench lane when local
/// Hot-Path resources cannot resolve destabilization.
pub trait HorizonEscalationProvider {
    /// Attempts to populate `slot_phase_out` with external phases matching `req`, returning
    /// `Ok(Some(lane_weight))` when new phases were pulled in or `Ok(None)` when no swap-in occurred.
    ///
    /// # Errors
    /// Returns a [`LmnkResult`] error code if external storage or recall fails.
    fn try_escalate_horizon(
        &mut self,
        req: &HorizonEscalationRequest,
        slot_phase_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> LmnkResult<Option<f32>>;
}

impl HorizonEscalationProvider for () {
    #[inline(always)]
    fn try_escalate_horizon(
        &mut self,
        _req: &HorizonEscalationRequest,
        _slot_phase_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> LmnkResult<Option<f32>> {
        Ok(None)
    }
}

impl StigmergicSubstrate {
    /// Evaluates external cold-path escalation (`escalation_provider`) against `req.trigger` and
    /// swaps phases into an anonymous free (or explicitly requested) workbench lane tracked in
    /// `active_lane_inout`, or cleanly releases that lane when `req.trigger.can_retract()` holds.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `req.placement` specifies an explicit lane
    /// `>= MAX_ATTRACTOR_DOMAINS`, [`crate::liminal::LMNK_ERR_SUBSTRATE_FULL`] if all lanes are pinned during
    /// `AutoFree` allocation, or propagates any error returned by `escalation_provider`.
    #[inline]
    pub fn apply_horizon_escalation<E: HorizonEscalationProvider + ?Sized>(
        &mut self,
        req: &HorizonEscalationRequest,
        active_lane_inout: &mut Option<u8>,
        escalation_provider: &mut E,
    ) -> LmnkResult<bool> {
        if let WorkbenchLanePlacement::Explicit { lane, .. } = req.placement {
            if usize::from(lane) >= MAX_ATTRACTOR_DOMAINS {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
        }
        if req.trigger.is_destabilized() {
            let mut escalated_phases = [0i8; MAX_ATTENTION_HEADS];
            if let Some(weight) =
                escalation_provider.try_escalate_horizon(req, &mut escalated_phases)?
            {
                let placement = match req.placement {
                    WorkbenchLanePlacement::AutoFree => {
                        if let Some(prev) = *active_lane_inout {
                            if usize::from(prev) < MAX_ATTRACTOR_DOMAINS
                                && (self.lane_lifecycle_quadrants().0.outside & (1u8 << prev)) != 0
                            {
                                WorkbenchLanePlacement::Explicit {
                                    lane: prev,
                                    pin: (req.lane_flags & WORKBENCH_LANE_FLAG_PINNED) != 0,
                                }
                            } else {
                                WorkbenchLanePlacement::AutoFree
                            }
                        } else {
                            WorkbenchLanePlacement::AutoFree
                        }
                    }
                    WorkbenchLanePlacement::Explicit { lane, pin } => {
                        if let Some(prev) = *active_lane_inout {
                            if prev != lane
                                && usize::from(prev) < MAX_ATTRACTOR_DOMAINS
                                && (self.slot_pinned_mask & (1u8 << prev)) == 0
                            {
                                let _ = self.release_lane(prev)?;
                            }
                        }
                        WorkbenchLanePlacement::Explicit { lane, pin }
                    }
                };
                let allocated = self.allocate_lane(&WorkbenchSlotSpec {
                    placement,
                    flags: req.lane_flags,
                    weight,
                    phases: escalated_phases,
                })?;
                *active_lane_inout = Some(allocated);
                return Ok(true);
            }
        } else if req.trigger.can_retract() {
            if let Some(lane) = active_lane_inout.take() {
                if usize::from(lane) < MAX_ATTRACTOR_DOMAINS {
                    let _ = self.release_lane(lane)?;
                }
            } else if let WorkbenchLanePlacement::Explicit { lane, .. } = req.placement {
                let _ = self.release_lane(lane)?;
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CustomCutSlot {
        mask: u32,
        cut: BitCut8,
        modulated: u32,
    }

    impl LatentHorizonSlot for CustomCutSlot {
        fn horizon_slot_mask(&self) -> u32 {
            self.mask
        }

        fn horizon_admission_cut(&self) -> BitCut8 {
            self.cut
        }

        fn on_slide_in_modulate(
            &mut self,
            _frame: &mut GatewayFrame,
            _stack: &mut CoreStateStack,
            _trigger: &HorizonTriggerInput,
        ) {
            self.modulated += 1;
        }
    }

    #[test]
    fn test_horizon_slot_bitcut8_admission_control() {
        let mut frame = GatewayFrame::default();
        let mut stack = CoreStateStack::new();
        let trigger = HorizonTriggerInput {
            repulsion_severity: 0.8,
            ..Default::default()
        };

        // 1. Authorized slot slides in cleanly.
        let mut ok_slot = CustomCutSlot {
            mask: 1 << 2,
            cut: BitCut8::split(1 << 2, 0xFF),
            modulated: 0,
        };
        let mut active = 0u32;
        let (in_b, out_b) =
            ok_slot.apply_horizon_slot(&mut active, &trigger, &mut frame, &mut stack);
        assert_eq!((in_b, out_b), (1 << 2, 0));
        assert_eq!(active, 1 << 2);
        assert_eq!(ok_slot.modulated, 1);

        // 2. Out-of-bounds channel (>= 8) is rejected branchlessly.
        let mut oob_slot = CustomCutSlot {
            mask: 1 << 8,
            cut: BitCut8::split(0xFF, 0xFF),
            modulated: 0,
        };
        let mut oob_active = 0u32;
        assert_eq!(
            oob_slot.apply_horizon_slot(&mut oob_active, &trigger, &mut frame, &mut stack),
            (0, 0)
        );
        assert_eq!(oob_active, 0);
        assert_eq!(oob_slot.modulated, 0);

        // 3. Empty BitCut8 cut is rejected.
        let mut empty_cut_slot = CustomCutSlot {
            mask: 1 << 1,
            cut: BitCut8::split(0, 0xFF),
            modulated: 0,
        };
        let mut empty_active = 0u32;
        assert_eq!(
            empty_cut_slot.apply_horizon_slot(&mut empty_active, &trigger, &mut frame, &mut stack),
            (0, 0)
        );
        assert_eq!(empty_cut_slot.modulated, 0);

        // 4. Unauthorized pinned-channel cut is rejected.
        let mut pinned_bad_slot = CustomCutSlot {
            mask: 1 << 0,
            cut: BitCut8::split(0x02, 0xFF),
            modulated: 0,
        };
        let mut pinned_active = 0u32;
        assert_eq!(
            pinned_bad_slot.apply_horizon_slot(
                &mut pinned_active,
                &trigger,
                &mut frame,
                &mut stack
            ),
            (0, 0)
        );
        assert_eq!(pinned_bad_slot.modulated, 0);

        // 5. Pinned-only cut on sandbox channel 7 is rejected.
        let mut sandbox_bad_slot = CustomCutSlot {
            mask: 1 << 7,
            cut: BitCut8::split(0x09, 0xFF),
            modulated: 0,
        };
        let mut sb_active = 0u32;
        assert_eq!(
            sandbox_bad_slot.apply_horizon_slot(&mut sb_active, &trigger, &mut frame, &mut stack),
            (0, 0)
        );
        assert_eq!(sandbox_bad_slot.modulated, 0);
    }
}
