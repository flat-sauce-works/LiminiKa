// name: src/core/src/runtime/gateway/bridges/substrate_bridge.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`StigmergicSubstrate`] sandbox lifecycle, online k-Means, and Gateway signal bridges.

use super::super::*;
use super::StigmergicReceptor;
use crate::foundation::{
    branchless_clamp_f32, branchless_is_finite_f32, branchless_max_f32, branchless_min_f32,
    branchless_rad_to_phase_step_u8, branchless_relu_f32, store_q7_words_64, swar_words_nonzero_64,
    AlignedSliceMut32, Pod, TaggedPtr, DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEP_MIN_RAD,
};
use crate::liminal::{
    branchless_select_u32, branchless_select_u8, LmnkResult, PhaseQ7, LMNK_ERR_INVALID_ARGUMENT,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, NUM_STIGMERGIC_GRANULES,
};
use crate::runtime::telemetry::{CoreTelemetryBank, LayerDistortionKind};
use crate::stigmergy::{
    trail_phase_words_mut, trail_phase_words_ref, PhaseAnchorRecord, PointerTrail,
    SandboxLifecycleEvent, StigmergicActionDecision, StigmergicSubstrate, StigmergicSubstrateOps,
    SubstrateStepReport, WorkbenchSlotSpec, ATTRACTOR_SLOT_REPULSION,
    DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE, MAX_ATTRACTOR_PULL_FORCE, SUBSTRATE_INLINE_TRAILS,
};

// ===================================================================
// 2. `StigmergicSubstrate` -- 2,816-Byte L1D-Resident Micro Passive Carrier
// ===================================================================

impl StigmergicSubstrate {
    /// Drives the isolated sandbox hypothesis lifecycle in zero-alloc mode:
    /// 1. Immediately releases unpinned sandbox lanes on [`StigmergicActionDecision::RepulsionEscape`]
    ///    or [`LayerDistortionKind::ActiveOverdrive`].
    /// 2. Settles an active sandbox lane via [`Self::settle_sandbox_lane`] once consecutive
    ///    coherence improvement reaches `settle_dwell_threshold`.
    /// 3. On sustained assimilation drift (`drift_dwell >= drift_dwell_threshold`), when `ego_active`
    ///    and `exterior_granule_mask != 0` (`(lacuna_exterior_mask | discovery_excursion_mask) & !overlap_core_mask`),
    ///    leases a free lane via [`WorkbenchSlotSpec::auto_sandbox`] masked strictly to `exterior_granule_mask`.
    #[inline]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn step_sandbox_hypothesis_lifecycle(
        &mut self,
        telemetry: &mut CoreTelemetryBank,
        active_sandbox_lane: &mut Option<u8>,
        decision: StigmergicActionDecision,
        ego_active: bool,
        exterior_granule_mask: u8,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        candidate_weight: f32,
        drift_dwell_threshold: u16,
        settle_dwell_threshold: u16,
    ) -> SandboxLifecycleEvent {
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

    /// Executes a complete closed-loop **2-Origin Composite Manifold Gateway $\to$ Stigmergic
    /// Threshold Action Selection $\to$ Online Toroidal $k$-Means / DP-Means Macro Clustering**
    /// step in zero-allocation mode ($C_{\mathrm{alloc}} = 0$).
    ///
    /// Returns `(GatewayFrame, SubstrateStepReport, assigned_centroid_idx, assigned_anchor_id, assigned_coherence, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `centroids` is empty, `split_coherence_thresh`
    /// is non-finite, or tensor slices are invalid.
    #[inline]
    // Internal helper keeps a flat parameter list; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    fn commit_gateway_kmeans_and_select_action(
        &mut self,
        frame: &GatewayFrame,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        entropy_h_tilde: f32,
        centroids: &mut [PhaseAnchorRecord],
        split_coherence_thresh: f32,
        trail_inout: &mut PointerTrail,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(SubstrateStepReport, usize, u32, f32, bool)> {
        if frame.target_anchor_id != 0 {
            trail_inout.target_anchor_id = frame.target_anchor_id;
        }
        trail_inout.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_inout.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let (assigned_idx, assigned_anchor_id, assigned_coh, _updated_mass, spawned_new) =
            PhaseAnchorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                frame,
                input_words,
                split_coherence_thresh,
                Some(trail_inout),
            );

        let win_w = *centroids[assigned_idx].phase_words_ref();
        let report = self.orchestrate_next_action_step_core(
            TaggedPtr(frame.tagged_ptr),
            entropy_h_tilde,
            frame.dominant_slot,
            trail_inout,
            attention_scores,
            query_tensor,
            key_tensor,
            logits_opt,
            false,
            Some(&win_w),
            frame.phase_step_min_rad,
            frame.phase_clamp_rad,
        )?;

        Ok((
            report,
            assigned_idx,
            assigned_anchor_id,
            assigned_coh,
            spawned_new,
        ))
    }

    /// Executes Gateway synthesis, Online Toroidal $k$-Means / DP-Means clustering, and
    /// Stigmergic action selection in a single `#![no_std]` zero-allocation pass.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `centroids` is empty, `split_coherence_thresh`
    /// is non-finite, or tensor slices are invalid.
    #[inline]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_gateway_kmeans_and_select_action<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        entropy_h_tilde: f32,
        centroids: &mut [PhaseAnchorRecord],
        split_coherence_thresh: f32,
        trail_inout: &mut PointerTrail,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(GatewayFrame, SubstrateStepReport, usize, u32, f32, bool)> {
        if centroids.is_empty() || !branchless_is_finite_f32(split_coherence_thresh) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        if swar_words_nonzero_64(&frm_w) {
            *trail_phase_words_mut(trail_inout) = frm_w;
        }
        let (report, assigned_idx, assigned_anchor_id, assigned_coh, spawned_new) = self
            .commit_gateway_kmeans_and_select_action(
                &frame,
                &frm_w,
                entropy_h_tilde,
                centroids,
                split_coherence_thresh,
                trail_inout,
                attention_scores,
                query_tensor,
                key_tensor,
                logits_opt,
            )?;

        Ok((
            frame,
            report,
            assigned_idx,
            assigned_anchor_id,
            assigned_coh,
            spawned_new,
        ))
    }

    /// Executes the full **Gateway-Origin Trait Composition $\to$ 8-Cell Micro-Swarm Consensus $\to$
    /// mapped-storage Attractor Basin Pull $\to$ Stigmergic Pheromone Action Selection & Receptor Dispatch**
    /// closed loop without manual hyperparameter arguments or semantic `if/else` task branching.
    ///
    /// 1. Synthesizes `(GatewayFrame, [u64; 8])` from `signal: &S` ([`ManifoldGatewaySignal`]).
    /// 2. Resolves the 8-cell SWAR Micro-Swarm (`try_resolve_action_in_swarm_from_words`) using
    ///    autonomous physical parameters derived directly from the [`GatewayFrame`].
    /// 3. Steps the mapped-storage Online Toroidal $k$-Means / DP-Means attractor basins (`centroids`) and
    ///    evolves `trail_inout`'s stigmergic pheromone density (`orchestrate_next_action_step_core`).
    /// 4. Dispatches `receptor.on_stigmergic_emergence(&frame, &report, trail_inout)` whenever
    ///    a non-passive [`StigmergicActionDecision`] emerges from the trail's accumulated density.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `centroids` is empty or tensor slices are invalid.
    #[inline]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_gateway_attractor_swarm_step<S, R>(
        &mut self,
        signal: &S,
        entropy_h_tilde: f32,
        centroids: &mut [PhaseAnchorRecord],
        trail_inout: &mut PointerTrail,
        receptor: &mut R,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(GatewayFrame, SubstrateStepReport, usize, u32, f32, bool)>
    where
        S: ManifoldGatewaySignal + ?Sized,
        R: StigmergicReceptor + ?Sized,
    {
        if centroids.is_empty() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let (frame, init_w) = signal.synthesize_gateway_frame_with_words();
        let (_swarm_rep, swarm_w) = trail_inout.try_resolve_action_in_swarm_from_words(
            TaggedPtr(frame.tagged_ptr),
            &init_w,
            frame.active_swarm_cell_mask,
            frame.autonomous_swarm_step_gain_u7(),
            frame.phase_step_min_q7,
            frame.autonomous_swarm_tolerance_q7(),
            frame.autonomous_swarm_quorum_threshold(),
            entropy_h_tilde,
            self.thresholds.inertial_skip_entropy_thresh,
            Some(&self.bloom_filter),
            3,
            96,
        );
        let (report, assigned_idx, assigned_anchor_id, assigned_coh, spawned_new) = self
            .commit_gateway_kmeans_and_select_action(
                &frame,
                &swarm_w,
                entropy_h_tilde,
                centroids,
                DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
                trail_inout,
                attention_scores,
                query_tensor,
                key_tensor,
                logits_opt,
            )?;

        if report.action_decision() != StigmergicActionDecision::PassiveSidecar {
            receptor.on_stigmergic_emergence(&frame, &report, trail_inout)?;
        }

        Ok((
            frame,
            report,
            assigned_idx,
            assigned_anchor_id,
            assigned_coh,
            spawned_new,
        ))
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
            attribution_tag: cause_code,
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
