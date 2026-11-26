// name: src/core/src/runtime/gateway/bridges/substrate_bridge.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`StigmergicSubstrate`] sandbox lifecycle, online k-Means, and Gateway signal bridges.

use super::super::*;
use crate::foundation::{
    DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEP_MIN_RAD, Pod, TaggedPtr, branchless_clamp_f32,
    branchless_max_f32, branchless_rad_to_phase_step_u8, branchless_relu_f32, store_q7_words_64,
    swar_words_nonzero_64,
};
use crate::liminal::{
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, NUM_STIGMERGIC_GRANULES, PhaseQ7,
    branchless_select_u8, branchless_select_u32,
};
use crate::runtime::telemetry::{CoreTelemetryBank, LayerDistortionKind};
use crate::stigmergy::{
    ATTRACTOR_SLOT_REPULSION, SUBSTRATE_INLINE_TRAILS, SandboxLifecycleEvent,
    StigmergicActionDecision, StigmergicSubstrate, StigmergicSubstrateOps, WorkbenchSlotSpec,
    trail_phase_words_ref,
};

// ===================================================================
// 2. `StigmergicSubstrate` -- 2,816-Byte L1D-Resident Micro Passive Carrier
// ===================================================================

/// Inputs for advancing the isolated exterior sandbox hypothesis lifecycle via
/// [`StigmergicSubstrate::step_sandbox_hypothesis_lifecycle`] or
/// [`crate::runtime::RuntimeContext::step_sandbox_hypothesis_lifecycle`].
#[derive(Debug, Clone, Copy)]
pub struct SandboxLifecycleInput<'a> {
    /// Current stigmergic action decision for this step.
    pub decision: StigmergicActionDecision,
    /// Whether subjective Ego is active and allowed to nucleate sandbox lanes.
    pub ego_active: bool,
    /// Granule bitmask permitted inside the isolated exterior sandbox lane.
    pub exterior_granule_mask: u8,
    /// Candidate 64-head Q7 SWAR phase words for potential sandbox nucleation.
    pub candidate_words: &'a [u64; NUM_STIGMERGIC_GRANULES],
    /// Candidate lane energy/weight before `[0.125, 1.0]` clamping.
    pub candidate_weight: f32,
    /// Sustained assimilation drift dwell threshold before nucleation.
    pub drift_dwell_threshold: u16,
    /// Consecutive coherence-improvement dwell threshold before settlement.
    pub settle_dwell_threshold: u16,
}

impl StigmergicSubstrate {
    /// Drives the isolated sandbox hypothesis lifecycle in zero-alloc mode:
    /// 1. Immediately releases unpinned sandbox lanes on [`StigmergicActionDecision::RepulsionEscape`]
    ///    or [`LayerDistortionKind::ActiveOverdrive`].
    /// 2. Settles an active sandbox lane via [`Self::settle_sandbox_lane`] once consecutive
    ///    coherence improvement reaches `input.settle_dwell_threshold`.
    /// 3. On sustained assimilation drift (`drift_dwell >= input.drift_dwell_threshold`), when `input.ego_active`
    ///    and `input.exterior_granule_mask != 0` (`(lacuna_exterior_mask | discovery_excursion_mask) & !overlap_core_mask`),
    ///    leases a free lane via [`WorkbenchSlotSpec::auto_sandbox`] masked strictly to `input.exterior_granule_mask`.
    #[inline]
    pub fn step_sandbox_hypothesis_lifecycle(
        &mut self,
        telemetry: &mut CoreTelemetryBank,
        active_sandbox_lane: &mut Option<u8>,
        input: SandboxLifecycleInput<'_>,
    ) -> SandboxLifecycleEvent {
        let SandboxLifecycleInput {
            decision,
            ego_active,
            exterior_granule_mask,
            candidate_words,
            candidate_weight,
            drift_dwell_threshold,
            settle_dwell_threshold,
        } = input;
        let distortion = telemetry.compute_distortion_report();
        if matches!(decision, StigmergicActionDecision::RepulsionEscape)
            || matches!(distortion.kind, LayerDistortionKind::ActiveOverdrive)
        {
            let released = self.release_sandbox_lanes();
            if let Some(lane) = *active_sandbox_lane {
                let lane_bit = if usize::from(lane) < MAX_ATTRACTOR_DOMAINS {
                    1u8 << lane
                } else {
                    0
                };
                let still_sandbox = lane_bit != 0
                    && ((self.slot_enabled_mask & self.slot_sandbox_mask) & lane_bit) != 0;
                if !still_sandbox {
                    *active_sandbox_lane = None;
                }
            }
            telemetry.reset_assimilation_dwell();
            telemetry.reset_sandbox_improve_dwell();
            return if released > 0 {
                SandboxLifecycleEvent::Released(released)
            } else {
                SandboxLifecycleEvent::Idle
            };
        }

        if let Some(lane) = *active_sandbox_lane {
            let lane_bit = if usize::from(lane) < MAX_ATTRACTOR_DOMAINS {
                1u8 << lane
            } else {
                0
            };
            let still_sandbox = lane_bit != 0
                && ((self.slot_enabled_mask & self.slot_sandbox_mask) & lane_bit) != 0;
            if !still_sandbox {
                *active_sandbox_lane = None;
            } else if telemetry.sandbox_coherence_improve_dwell() >= settle_dwell_threshold.max(1) {
                if self.settle_sandbox_lane(lane).unwrap_or(false) {
                    *active_sandbox_lane = None;
                    telemetry.reset_assimilation_dwell();
                    telemetry.reset_sandbox_improve_dwell();
                    return SandboxLifecycleEvent::Settled(lane);
                }
            } else {
                return SandboxLifecycleEvent::Idle;
            }
        }

        if active_sandbox_lane.is_none()
            && ego_active
            && exterior_granule_mask != 0
            && telemetry.assimilation_drift_dwell() >= drift_dwell_threshold.max(1)
        {
            let mut masked_words = [0u64; NUM_STIGMERGIC_GRANULES];
            for (g, (dst, &src)) in masked_words
                .iter_mut()
                .zip(candidate_words.iter())
                .enumerate()
            {
                let active_mask = (((exterior_granule_mask >> g) & 1) as u64).wrapping_neg();
                *dst = src & active_mask;
            }
            if swar_words_nonzero_64(&masked_words) {
                let mut phases = [0 as PhaseQ7; MAX_ATTENTION_HEADS];
                store_q7_words_64(&mut phases, &masked_words);
                let weight =
                    branchless_clamp_f32(branchless_relu_f32(candidate_weight), 0.125, 1.0);
                if let Ok(lane) =
                    self.allocate_lane(&WorkbenchSlotSpec::auto_sandbox(weight, phases))
                {
                    *active_sandbox_lane = Some(lane);
                    telemetry.reset_sandbox_improve_dwell();
                    return SandboxLifecycleEvent::Nucleated(lane);
                }
            }
        }

        SandboxLifecycleEvent::Idle
    }
}

impl CausalProvenanceSignal for StigmergicSubstrate {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let dec = StigmergicActionDecision::from_u8(self.last_decision);
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | branchless_select_u32(
                dec == StigmergicActionDecision::InertialLayerSkip,
                CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
                0,
            )
            | branchless_select_u32(
                dec == StigmergicActionDecision::CrystallizeMacroBasin,
                CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED | CAUSAL_AXIS_SUBJECTIVE_PERSONA,
                0,
            )
            | branchless_select_u32(
                dec == StigmergicActionDecision::RepulsionEscape,
                CAUSAL_AXIS_CONTRADICTION_GUARD | CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
                CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA,
            )
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.last_anomaly_severity
    }
}

impl SystemConstraintSignal for StigmergicSubstrate {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        0.0
    }

    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        self.slot_enabled_mask
    }

    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        self.last_decision == (StigmergicActionDecision::InertialLayerSkip as u8)
    }

    #[inline(always)]
    fn phase_step_min_rad_override(&self) -> f32 {
        DEFAULT_PHASE_STEP_MIN_RAD
    }
}

impl ManifoldGatewaySignal for StigmergicSubstrate {
    #[inline]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let sel_idx = (self.last_selected_trail_id as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        let latest_idx =
            (self.write_cursor.wrapping_sub(1) as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        let idx = if (self.active_count > 0) & (self.trails[sel_idx].current_ptr == 0) {
            latest_idx
        } else {
            sel_idx
        };
        let active_trail = &self.trails[idx];
        let words = *trail_phase_words_ref(active_trail);
        let dec = StigmergicActionDecision::from_u8(self.last_decision);
        let mode_flags = dec.action_mode_flags();
        let cur_tp = TaggedPtr(active_trail.current_ptr);
        let dom_slot = branchless_select_u8(
            dec == StigmergicActionDecision::RepulsionEscape,
            ATTRACTOR_SLOT_REPULSION,
            cur_tp.target_attractor_slot(),
        );
        let cause_code = branchless_select_u8(
            dec == StigmergicActionDecision::RepulsionEscape,
            ROOT_CAUSE_COHOMOLOGY_REPULSION,
            branchless_select_u8(
                dec == StigmergicActionDecision::InertialLayerSkip,
                ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
                ROOT_CAUSE_NOMINAL,
            ),
        );
        let phase_step_min_rad = DEFAULT_PHASE_STEP_MIN_RAD;
        let num_heads_u8 = (u32::from(self.num_heads).min(MAX_ATTENTION_HEADS as u32)) as u8;
        let fact_h = num_heads_u8 >> 1;
        let logic_h = num_heads_u8 >> 2;
        let explore_h = num_heads_u8.saturating_sub(fact_h + logic_h);
        let mut frame = GatewayFrame {
            tagged_ptr: active_trail.current_ptr,
            mapped_section_offset48: cur_tp.unpack_addr(),
            target_anchor_id: active_trail.target_anchor_id,
            effective_energy: branchless_max_f32(
                active_trail.stigmergic_density,
                active_trail.attractor_pull_force,
            ),
            phase_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            phase_step_min_rad,
            effective_inv_temp: 1.0,
            fact_persona_coherence: 1.0 - 2.0 * self.last_anomaly_severity,
            repulsion_severity: self.last_anomaly_severity,
            lineage_flags: self.causal_provenance_bits(),
            root_cause_code: cause_code,
            dominant_slot: dom_slot,
            action_mode_flags: mode_flags,
            active_swarm_cell_mask: self.slot_enabled_mask,
            phase_step_min_q7: branchless_rad_to_phase_step_u8(phase_step_min_rad),
            subhead_fact_heads: fact_h,
            subhead_logic_heads: logic_h,
            subhead_explore_heads: explore_h,
            subhead_fact_gain_q7: 128,
            subhead_logic_gain_q7: 128,
            subhead_explore_gain_q7: 128,
            mean_phase_tag_q7: cur_tp.unpack_phase(),
            active_head_count: u16::from(num_heads_u8),
            frame_flags: u16::from(dec == StigmergicActionDecision::RepulsionEscape),
            phase_deltas: [0; MAX_ATTENTION_HEADS],
        };
        *frame.phase_words_mut() = words;
        (frame, words)
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

const _: () = {
    const fn assert_substrate_bounds<
        T: StigmergicSubstrateOps
            + SubstrateGatewayExt
            + CausalProvenanceSignal
            + SystemConstraintSignal
            + ManifoldGatewaySignal
            + Pod,
    >() {
    }
    assert_substrate_bounds::<StigmergicSubstrate>();
};
