// name: src/core/src/runtime/gateway/orchestrator.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gateway Orchestrator extensions (`CoreLayerEvalContext`, [`GatewayPhaseSteeringExt`],
//! [`SubstrateGatewayExt`], [`ScaleLayerOrchestrator`], [`HybridScaleCircuit`], and [`StigmergicScaleLayer`]).

use super::*;
use crate::drivers::{DistortionTension, EgoDifferentialNet, SystemSpineState};
use crate::foundation::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_max_f32, branchless_min_f32, branchless_relu_f32,
    dispatch_fused_phase_clamp_and_subhead_words_core,
    rotate_fused_self_phase_clamp_and_subhead_default_words,
    rotate_fused_self_phase_clamp_subhead_q_and_clamp_k_default_words, scale_f32_slice_unrolled8,
    steer_from_swar_words_core_safe, store_q7_words_64, swar_or_avx2_q7_add_sat_words_64,
    swar_q7_abs_u8_x8, swar_q7_circular_add_x8, swar_q7_circular_diff_x8,
    swar_q7_phase_step_filter_words_64, swar_q7_scale_le128_with_bias_x8, swar_words_nonzero_64,
    AlignedBuffer16, AlignedSlice16, AlignedSlice32, AlignedSliceMut32, PhaseSteering,
    RawPhaseProjectable, RopePairLayout, RopeSteeringTarget, TaggedPtr, ACTION_MODE_LAYER_SKIP,
    ACTION_MODE_PHASE_REPULSE, ACTION_MODE_SUBHEAD_ROUTE, ATTRACTOR_SLOT_ID_MASK,
    DEFAULT_PHASE_CLAMP_MAX_RAD, SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{
    branchless_select_u32, branchless_select_u8, LacunaPartition8, LmnkResult, PhaseQ7, Scoped,
    LMNK_ERR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::context::CoreStateStack;
use crate::runtime::horizon::EpistemicHorizonReading;
use crate::stigmergy::{
    swar_mean_u8_le128_x8, trail_phase_words_mut, trail_phase_words_ref,
    validate_optional_qk_tensors, CognitiveCircuitActivation, CognitiveOpKind, CognitiveStepOp,
    CognitiveTrajectoryLedger, ContrastivePhaseBloomFilter, LedgerSlot, OverlapDiscoveryReport,
    PhaseAnchorRecord, PhaseDeltaSlice, PhaseSteeringActiveExt, PointerTrail,
    ResidualAdapterEngine, ResidualScale, StigmergicActionDecision, StigmergicReconstitutionReport,
    StigmergicScaleGranularity, StigmergicSubstrateOps, StigmergicThresholds,
    LEDGER_FLAG_COGNITIVE_TRAJECTORY, LEDGER_METRIC_FAST_PATH_HIT,
    LEDGER_METRIC_LAYER_SKIP_COMPENSATED, LEDGER_METRIC_PHASE_STEERED,
    LEDGER_METRIC_RANK1_INJECTED, LEDGER_METRIC_REPULSION_APPLIED,
    LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_PHEROMONE_RISING, TRAIL_FLAG_REPULSION_ACTIVE,
};

#[inline]
pub(crate) fn accumulate_trail_and_steer_gateway_words_core(
    frame: &GatewayFrame,
    frm_w: &[u64; NUM_STIGMERGIC_GRANULES],
    trail_inout: &mut PointerTrail,
    unresolved_damping_u7: u8,
    head_dim: usize,
    num_heads: usize,
    query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> LmnkResult<(bool, bool, bool)> {
    let frm_nonzero = swar_words_nonzero_64(frm_w);
    let next_w = if frm_nonzero {
        *trail_phase_words_mut(trail_inout) = *frm_w;
        *frm_w
    } else {
        *trail_phase_words_ref(trail_inout)
    };
    trail_inout.attractor_pull_force = branchless_min_f32(
        branchless_max_f32(trail_inout.attractor_pull_force, frame.effective_energy),
        MAX_ATTRACTOR_PULL_FORCE,
    );

    let layer_skipped = (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0;
    let mut steered = false;
    let mut subhead_routed = false;

    if layer_skipped {
        trail_inout.flags |= TRAIL_FLAG_INERTIAL_SKIP_ACTIVE;
    } else if query_tensor.is_some() | key_tensor.is_some() {
        steered = true;
        let swar_words = if frame.phase_step_min_q7 <= 1 {
            next_w
        } else if frame.phase_step_min_q7 > 127 {
            [0u64; NUM_STIGMERGIC_GRANULES]
        } else {
            swar_q7_phase_step_filter_words_64(&next_w, frame.phase_step_min_q7)
        };
        let clean_damping =
            branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);

        let has_nonzero_words = if frm_nonzero & (frame.phase_step_min_q7 <= 1) {
            true
        } else {
            swar_words_nonzero_64(&swar_words)
        };
        if (frame.action_mode_flags & ACTION_MODE_SUBHEAD_ROUTE) != 0
            && frame.active_swarm_cell_mask == 0xFF
        {
            let is_def_clamp = (frame.phase_clamp_rad.to_bits()
                == DEFAULT_PHASE_CLAMP_MAX_RAD.to_bits())
                & frame.has_unit_subhead_gains()
                & (num_heads > 0)
                & (num_heads <= MAX_ATTENTION_HEADS);
            match (query_tensor, key_tensor) {
                (Some(q), Some(k)) => {
                    if has_nonzero_words {
                        if is_def_clamp {
                            let rot_dim = frame.resolve_autonomous_rot_dim(head_dim, 0.0, 0.0);
                            rotate_fused_self_phase_clamp_subhead_q_and_clamp_k_default_words(
                                q.as_mut_slice(),
                                k.as_mut_slice(),
                                &swar_words,
                                num_heads,
                                head_dim,
                                rot_dim,
                            );
                        } else {
                            let subhead_cfg = frame.to_subhead_config_for_heads(num_heads);
                            dispatch_fused_phase_clamp_and_subhead_words_core(
                                q,
                                &swar_words,
                                frame.phase_clamp_rad,
                                &subhead_cfg,
                                &swar_words,
                                &swar_words,
                                &swar_words,
                                frame.phase_clamp_rad,
                                0,
                                head_dim,
                                true,
                            )?;
                            steer_from_swar_words_core_safe(
                                &swar_words,
                                0xFF,
                                clean_damping,
                                head_dim,
                                num_heads,
                                false,
                                frame.phase_clamp_rad,
                                k,
                                None,
                            );
                        }
                    }
                    subhead_routed = true;
                }
                (Some(q), None) => {
                    if has_nonzero_words {
                        if is_def_clamp {
                            let rot_dim = frame.resolve_autonomous_rot_dim(head_dim, 0.0, 0.0);
                            rotate_fused_self_phase_clamp_and_subhead_default_words(
                                q.as_mut_slice(),
                                &swar_words,
                                num_heads,
                                head_dim,
                                rot_dim,
                                true,
                            );
                        } else {
                            let subhead_cfg = frame.to_subhead_config_for_heads(num_heads);
                            dispatch_fused_phase_clamp_and_subhead_words_core(
                                q,
                                &swar_words,
                                frame.phase_clamp_rad,
                                &subhead_cfg,
                                &swar_words,
                                &swar_words,
                                &swar_words,
                                frame.phase_clamp_rad,
                                0,
                                head_dim,
                                true,
                            )?;
                        }
                    }
                    subhead_routed = true;
                }
                (None, Some(k)) => {
                    if has_nonzero_words {
                        steer_from_swar_words_core_safe(
                            &swar_words,
                            0xFF,
                            clean_damping,
                            head_dim,
                            num_heads,
                            false,
                            frame.phase_clamp_rad,
                            k,
                            None,
                        );
                    }
                }
                (None, None) => {}
            }
        } else if has_nonzero_words {
            match (query_tensor, key_tensor) {
                (Some(q), k_opt) => {
                    steer_from_swar_words_core_safe(
                        &swar_words,
                        frame.active_swarm_cell_mask,
                        clean_damping,
                        head_dim,
                        num_heads,
                        false,
                        frame.phase_clamp_rad,
                        q,
                        k_opt,
                    );
                }
                (None, Some(k)) => {
                    steer_from_swar_words_core_safe(
                        &swar_words,
                        frame.active_swarm_cell_mask,
                        clean_damping,
                        head_dim,
                        num_heads,
                        false,
                        frame.phase_clamp_rad,
                        k,
                        None,
                    );
                }
                (None, None) => {}
            }
        }
    }

    if let Some(logits) = logits_opt {
        if !logits.is_empty() {
            scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
        }
    }

    Ok((layer_skipped, steered, subhead_routed))
}

#[inline(always)]
pub(super) fn filter_and_damp_gateway_words(
    next_w: &[u64; NUM_STIGMERGIC_GRANULES],
    phase_step_min_q7: u8,
    unresolved_damping_u7: u8,
) -> [u64; NUM_STIGMERGIC_GRANULES] {
    let phase_step_words = if phase_step_min_q7 <= 1 {
        *next_w
    } else if phase_step_min_q7 > 127 {
        [0u64; NUM_STIGMERGIC_GRANULES]
    } else {
        swar_q7_phase_step_filter_words_64(next_w, phase_step_min_q7)
    };
    let clean_damping =
        branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
    if clean_damping >= 128 {
        phase_step_words
    } else {
        let s_u64 = u64::from(clean_damping);
        let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        [
            swar_q7_scale_le128_with_bias_x8(phase_step_words[0], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(phase_step_words[1], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(phase_step_words[2], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(phase_step_words[3], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(phase_step_words[4], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(phase_step_words[5], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(phase_step_words[6], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(phase_step_words[7], s_u64, bias_restore),
        ]
    }
}

/// Borrowed read-only core context (`2 Primitives x 2 Drivers`) passed into
/// per-layer Hot-Path RoPE steering and substrate orchestration.
#[derive(Clone, Copy)]
pub struct CoreLayerEvalContext<'a> {
    /// Unified 3-layer autonomous core stack (`128 B`).
    pub stack: &'a CoreStateStack,
    /// Temporal boundary velocity between consecutive Layer 0 epochs.
    pub delta: &'a LiminalBoundaryDelta,
    /// Latest epistemic horizon reading (`KnownCoherent`, `InternalConflict`, or `Liminal`).
    pub horizon: &'a EpistemicHorizonReading,
    /// Latest overlap and discovery telemetry report across the 8 SWAR granules.
    pub overlap: &'a OverlapDiscoveryReport,
    /// Read-only Slow-Loop Q0.7 derivative weight $\alpha \in [0, 127]$ for convex pheromone synthesis.
    pub derivative_alpha_q7: u8,
    /// Optional 4-region branchless [`LacunaPartition8`].
    pub lacuna_partition: Option<&'a LacunaPartition8>,
    /// Optional [`SystemSpineState`] physical `pinned` isolation barrier.
    pub spine: Option<&'a SystemSpineState>,
    /// Optional P-Frame differential chain [`EgoDifferentialNet`].
    pub ego_net: Option<&'a EgoDifferentialNet>,
    /// Optional [`DistortionTension`] scaling the P-Frame delta amplitude.
    pub tension: Option<&'a DistortionTension>,
}

#[inline(always)]
fn synthesize_signal_with_optional_core_layers<S: ManifoldGatewaySignal + ?Sized>(
    signal: &S,
    core_ctx: Option<CoreLayerEvalContext<'_>>,
) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
    if let Some(ctx) = core_ctx {
        if ctx.stack.liminal.adaptation_epochs > 0 || ctx.stack.ego.is_active() {
            let mut frame = signal.synthesize_gateway_with_aperture_from_stack(ctx.stack);
            if ctx.stack.ego.is_active() {
                // Compute the pure `SystemSpine` I-Frame baseline BEFORE `LiminalEgoState` and
                // `EgoDifferentialNet` mutate the frame so `pinned` isolation always anchors to
                // pre-Ego state.
                let mut sys_baseline = frame;
                ctx.stack.modulate_gateway_frame_branchless(
                    &mut sys_baseline,
                    ctx.horizon,
                    ctx.overlap,
                );
                ctx.stack.modulate_gateway_frame_with_ego_readonly(
                    ctx.delta,
                    ctx.horizon,
                    ctx.overlap,
                    &mut frame,
                );
                if let (Some(spine), Some(vp), Some(net), Some(tension)) =
                    (ctx.spine, ctx.lacuna_partition, ctx.ego_net, ctx.tension)
                {
                    let _ = net.evaluate_pframe_chain_branchless(
                        &ctx.stack.ego,
                        spine,
                        vp,
                        &ctx.stack.stigmergy,
                        tension,
                        &mut frame,
                    );
                    spine.enforce_pinned_isolation_branchless(&sys_baseline, &mut frame);
                    let w = *frame.phase_words_ref();
                    return (frame, w);
                }
                if let Some(spine) = ctx.spine {
                    spine.enforce_pinned_isolation_branchless(&sys_baseline, &mut frame);
                    let w = *frame.phase_words_ref();
                    return (frame, w);
                }
            } else {
                // Non-Ego fast path: modulate in-place without cloning `sys_baseline` or
                // enforcing pinned isolation when `ego` is inactive.
                ctx.stack
                    .modulate_gateway_frame_branchless(&mut frame, ctx.horizon, ctx.overlap);
            }
            let w = *frame.phase_words_ref();
            return (frame, w);
        }
    }
    signal.synthesize_gateway_frame_with_words()
}

#[inline(always)]
pub(super) fn prepare_gateway_steering_state<
    T: PhaseSteering + ?Sized,
    S: ManifoldGatewaySignal + ?Sized,
>(
    engine: &T,
    signal: &S,
    core_ctx: Option<CoreLayerEvalContext<'_>>,
    trail_inout: &mut PointerTrail,
    query_tensor: &AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
) -> LmnkResult<(
    GatewayFrame,
    [u64; NUM_STIGMERGIC_GRANULES],
    CausalProvenanceReport,
    usize,
    usize,
)> {
    let (frame, frm_w) = synthesize_signal_with_optional_core_layers(signal, core_ctx);
    let report = frame.to_causal_report();
    let (head_dim, num_heads, expected_len) = engine.validate_head_layout()?;
    if num_heads > MAX_ATTENTION_HEADS
        || frame.phase_step_min_rad < 0.0
        || !branchless_is_finite_f32(frame.phase_step_min_rad)
        || frame.phase_clamp_rad <= 0.0
        || !branchless_is_finite_f32(frame.phase_clamp_rad)
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    validate_optional_qk_tensors(Some(query_tensor), key_tensor, expected_len)?;

    trail_inout.prev_ptr = trail_inout.current_ptr;
    trail_inout.current_ptr = frame.tagged_ptr;
    trail_inout.step_count = trail_inout.step_count.wrapping_add(1);
    if frame.target_anchor_id != 0 {
        trail_inout.target_anchor_id = frame.target_anchor_id;
        trail_inout.cluster_id = frame.target_anchor_id;
    } else {
        trail_inout.cluster_id = u32::from(frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK);
    }
    Ok((frame, frm_w, report, head_dim, num_heads))
}

/// Blanket-implemented closed-loop Gateway & Crystallized Ensemble steering extension trait for
/// any type implementing [`PhaseSteering`].
pub trait GatewayPhaseSteeringExt: PhaseSteering {
    /// Synthesizes any [`ManifoldGatewaySignal`] into a 128-byte [`GatewayFrame`] and register-resident
    /// phase words, updates `trail_inout`, and executes $SO(2)$-fused `PhaseClamp` + sub-head router sub-head steering.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if head dimensions, radian clamps, or tensor slices
    /// violate layout or `LMNK_RESTRICT` disjointness invariants.
    #[inline(always)]
    fn steer_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail_inout: &mut PointerTrail,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(GatewayFrame, CausalProvenanceReport)> {
        let (frame, frm_w, report, head_dim, num_heads) = prepare_gateway_steering_state(
            self,
            signal,
            None,
            trail_inout,
            query_tensor,
            key_tensor.as_deref(),
        )?;

        let _ = accumulate_trail_and_steer_gateway_words_core(
            &frame,
            &frm_w,
            trail_inout,
            unresolved_damping_u7,
            head_dim,
            num_heads,
            Some(query_tensor),
            key_tensor,
            logits_opt,
        )?;
        Ok((frame, report))
    }

    /// Synthesizes any [`ManifoldGatewaySignal`] into structural hyperparameters (`phase_clamp_rad`,
    /// `phase_step_min_rad`, `effective_inv_temp`, SWAR phase words) and steers `query_tensor`
    /// (and optional `key_tensor`) according to [`RopeSteeringTarget`] (`Interleaved` or `HalfSplit`,
    /// including partial `rotary_dim`) with zero heap allocation.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `geometry`, radian clamps, or tensor slices
    /// violate layout or `LMNK_RESTRICT` disjointness invariants.
    #[inline]
    fn steer_gateway_signal_for_geometry_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &self,
        signal: &S,
        geometry: &G,
        trail_inout: &mut PointerTrail,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(GatewayFrame, CausalProvenanceReport)> {
        self.steer_gateway_signal_with_core_layers_for_geometry_safe(
            signal,
            geometry,
            None,
            trail_inout,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            logits_opt,
        )
    }

    /// Core-layer-aware counterpart to [`Self::steer_gateway_signal_for_geometry_safe`] applying
    /// dynamic trait cognitive apertures and read-only 3-layer modulation (`Liminal` + `Stigmergic` + `Ego`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `geometry`, radian clamps, or tensor slices
    /// violate layout or `LMNK_RESTRICT` disjointness invariants.
    #[inline]
    fn steer_gateway_signal_with_core_layers_for_geometry_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &self,
        signal: &S,
        geometry: &G,
        core_ctx: Option<CoreLayerEvalContext<'_>>,
        trail_inout: &mut PointerTrail,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(GatewayFrame, CausalProvenanceReport)> {
        let geom = geometry.rope_geometry();
        geom.validate()?;
        if core_ctx.is_none()
            && geom.layout == RopePairLayout::Interleaved
            && geom.rotary_dim == geom.head_dim
        {
            return self.steer_gateway_signal_safe(
                signal,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                logits_opt,
            );
        }

        let (frame, frm_w, report, head_dim, num_heads) = prepare_gateway_steering_state(
            self,
            signal,
            core_ctx,
            trail_inout,
            query_tensor,
            key_tensor.as_deref(),
        )?;
        if geom.head_dim != head_dim || geom.num_heads != num_heads {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        let frm_nonzero = swar_words_nonzero_64(&frm_w);
        let next_w = if frm_nonzero {
            *trail_phase_words_mut(trail_inout) = frm_w;
            frm_w
        } else {
            *trail_phase_words_ref(trail_inout)
        };
        trail_inout.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_inout.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let layer_skipped = (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0;
        if layer_skipped {
            trail_inout.flags |= TRAIL_FLAG_INERTIAL_SKIP_ACTIVE;
        } else {
            let swar_words = filter_and_damp_gateway_words(
                &next_w,
                frame.phase_step_min_q7,
                unresolved_damping_u7,
            );

            if swar_words_nonzero_64(&swar_words) {
                let mut q7_stack = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
                store_q7_words_64(&mut q7_stack.data, &swar_words);
                let p_aligned = AlignedSlice16::new(&q7_stack.data[..num_heads])?;
                if let Some(active_deltas) = Scoped::from_boundary(PhaseDeltaSlice(p_aligned))
                    .cut_by_lacuna(frame.phase_clamp_rad > 0.0, frame.phase_clamp_rad, 0.0)
                    .into_active()
                {
                    PhaseSteeringActiveExt::apply_phase_steering_for_geometry_safe_active(
                        self,
                        geometry,
                        query_tensor,
                        active_deltas,
                        frame.phase_clamp_rad,
                    )?;
                    if let Some(k) = key_tensor {
                        PhaseSteeringActiveExt::apply_phase_steering_for_geometry_safe_active(
                            self,
                            geometry,
                            k,
                            active_deltas,
                            frame.phase_clamp_rad,
                        )?;
                    }
                }
            }
        }

        if let Some(logits) = logits_opt {
            if !logits.is_empty() {
                scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
            }
        }
        Ok((frame, report))
    }
}

impl<T: PhaseSteering + ?Sized> GatewayPhaseSteeringExt for T {}

/// Applies norm-adaptive fused Rank-1 residual injection from an already synthesized or
/// circuit-modulated [`GatewayFrame`].
///
/// Derives a scale-invariant residual injection gain autonomously from the hidden state's
/// L2 norm and [`GatewayFrame`] structural quantities (`effective_energy`, `phase_clamp_rad`,
/// `fact_persona_coherence`, `repulsion_severity`), eliminating manual scalar tuning.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if slice lengths mismatch or Rank-1 residual evaluation rejects inputs.
#[inline]
pub fn apply_gateway_frame_rank1_hidden_fused(
    frame: &GatewayFrame,
    y_out: &mut AlignedSliceMut32<'_, f32>,
    x_in: &AlignedSlice32<'_, f32>,
    u_vec: &AlignedSlice32<'_, f32>,
    v_vec: &AlignedSlice32<'_, f32>,
) -> LmnkResult<f32> {
    if (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0 {
        return Ok(0.0);
    }
    // Rank-1 outer product `(v^T x) u` already scales linearly with `||x||_2`,
    // so `structural_gain` must be a dimensionless ratio derived from Gateway physics.
    let energy_gate = branchless_clamp_f32(
        frame.effective_energy * (1.0 - 0.5 * frame.repulsion_severity),
        0.0,
        4.0,
    );
    let coh_boost = branchless_clamp_f32(0.5 + 0.5 * frame.fact_persona_coherence, 0.25, 1.0);
    let structural_gain = branchless_flush_subnormal_f32(branchless_clamp_f32(
        frame.phase_clamp_rad * energy_gate * coh_boost * 0.125,
        0.0,
        0.05,
    ));
    if let Some(active_scale) = Scoped::from_boundary(ResidualScale(structural_gain))
        .cut_by_lacuna(structural_gain > 0.0, structural_gain, 0.0)
        .into_active()
    {
        ResidualAdapterEngine::default().eval_rank1_slices_fused_active(
            y_out,
            x_in,
            u_vec,
            v_vec,
            None,
            active_scale,
        )?;
    }
    Ok(structural_gain)
}

/// Blanket-implemented extension trait equipping any [`StigmergicSubstrateOps`] with single-call
/// [`ManifoldGatewaySignal`] ingestion, Online Toroidal $k$-Means / DP-Means macro clustering,
/// online basin consolidation, and `.lmnk` crystallization.
pub trait SubstrateGatewayExt: StigmergicSubstrateOps {
    /// Synthesizes `signal` with dynamic trait cognitive apertures and optional 3-layer core
    /// modulation (`core_ctx`), then steers `query_tensor` and optional `logits_opt` for `geometry`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`StigmergicSubstrateOps`] or [`PhaseSteeringActiveExt`].
    #[inline]
    fn ingest_gateway_and_steer_with_core_layers_for_geometry_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &mut self,
        signal: &S,
        geometry: &G,
        core_ctx: Option<CoreLayerEvalContext<'_>>,
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        unresolved_damping_u7: u8,
        mut query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(GatewayFrame, CausalProvenanceReport)> {
        let geom = geometry.rope_geometry();
        let (frame, frm_w) = synthesize_signal_with_optional_core_layers(signal, core_ctx);
        let report = frame.to_causal_report();
        let (head_dim, num_heads, _expected_len) = self.validate_substrate_cascade_params(
            geom.head_dim as u32,
            geom.num_heads as u32,
            frame.phase_step_min_rad,
            frame.phase_clamp_rad,
            query_tensor.as_deref(),
            None,
        )?;
        if geom.head_dim != head_dim || geom.num_heads != num_heads {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        let (bypass_hit, effective_ptr) =
            self.resolve_or_pass_tagged_ptr(ledger_slot, TaggedPtr(frame.tagged_ptr));
        let frm_nonzero = swar_words_nonzero_64(&frm_w);
        let next_w = if frm_nonzero {
            *trail_phase_words_mut(trail_out) = frm_w;
            frm_w
        } else {
            *trail_phase_words_ref(trail_out)
        };
        if frame.target_anchor_id != 0 {
            trail_out.target_anchor_id = frame.target_anchor_id;
            trail_out.cluster_id = frame.target_anchor_id;
        } else {
            trail_out.cluster_id = frame.dominant_slot as u32;
        }
        trail_out.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_out.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        let step_ptr = if frm_nonzero {
            effective_ptr.with_phase(0)
        } else {
            effective_ptr
        };
        self.step_pointer(step_ptr.raw(), trail_out)?;
        trail_out.current_ptr = effective_ptr.raw();

        let is_repulsed = report.is_cohomology_repulsion()
            | ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0);
        let mut metrics = branchless_select_u8(bypass_hit, LEDGER_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(is_repulsed, LEDGER_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(
                (frame.lineage_flags
                    & (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA))
                    == (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA),
                LEDGER_METRIC_SLOT_SYNERGY_ACTIVE,
                0,
            );

        let layer_skipped = (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0;
        if layer_skipped {
            trail_out.flags |= TRAIL_FLAG_INERTIAL_SKIP_ACTIVE;
            metrics |= LEDGER_METRIC_LAYER_SKIP_COMPENSATED;
        } else if let Some(q) = query_tensor.as_deref_mut() {
            let clean_damping =
                branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
            let swar_words =
                filter_and_damp_gateway_words(&next_w, frame.phase_step_min_q7, clean_damping);
            if swar_words_nonzero_64(&swar_words) {
                let mut q7_stack = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
                store_q7_words_64(&mut q7_stack.data, &swar_words);
                let p_aligned = AlignedSlice16::new(&q7_stack.data[..num_heads])?;
                if let Some(active_deltas) = Scoped::from_boundary(PhaseDeltaSlice(p_aligned))
                    .cut_by_lacuna(frame.phase_clamp_rad > 0.0, frame.phase_clamp_rad, 0.0)
                    .into_active()
                {
                    let steering = crate::stigmergy::PhaseSteeringEngine::new(
                        head_dim as u32,
                        num_heads as u32,
                    )?;
                    PhaseSteeringActiveExt::apply_phase_steering_for_geometry_safe_active(
                        &steering,
                        geometry,
                        q,
                        active_deltas,
                        frame.phase_clamp_rad,
                    )?;
                    metrics |= LEDGER_METRIC_PHASE_STEERED;
                }
            }
        }

        if let Some(logits) = logits_opt {
            if !logits.is_empty() && frame.effective_inv_temp > 0.0 {
                scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
            }
        }

        let _ = self.telemetry_push(ledger_slot, metrics);
        Ok((frame, report))
    }

    /// Orchestrates scale-aware `signal` synthesis, 3-layer core modulation, and [`StigmergicSubstrateOps`]
    /// decision routing for `geometry`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`StigmergicSubstrateOps`].
    #[inline]
    fn orchestrate_gateway_layer_with_scale_and_core_layers_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &mut self,
        signal: &S,
        geometry: &G,
        core_ctx: Option<CoreLayerEvalContext<'_>>,
        thresholds: &StigmergicThresholds,
        scale: StigmergicScaleGranularity,
        entropy_h_tilde: f32,
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        unresolved_damping_u7: u8,
        mut query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        rank1_hidden: Option<(
            &mut AlignedSliceMut32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
        )>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(
        GatewayFrame,
        CausalProvenanceReport,
        StigmergicActionDecision,
    )> {
        let geom = geometry.rope_geometry();
        let (mut frame, mut frm_w) = synthesize_signal_with_optional_core_layers(signal, core_ctx);
        let lane_mask = self.lane_mask();
        if lane_mask != 0 {
            let (rep_cut, attract_cut) =
                crate::liminal::BitCut8::split(lane_mask, self.repulsion_lane_mask())
                    .refine(self.sandbox_lane_mask());
            let is_pre_repulsed = ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0)
                || frame.to_causal_report().is_cohomology_repulsion();
            let active_cut = if is_pre_repulsed {
                rep_cut
            } else {
                attract_cut
            };
            let mut eligible_lanes = active_cut.outside | active_cut.inside;
            if eligible_lanes != 0 {
                let standard_granule_mask = core_ctx.map_or(0xFF_u8, |c| {
                    c.overlap.overlap_core_mask | c.overlap.overlap_fringe_mask
                });
                let sandbox_granule_mask = core_ctx.map_or(0x00_u8, |c| {
                    crate::liminal::BitCut8::split(
                        c.overlap.lacuna_exterior_mask | c.overlap.discovery_excursion_mask,
                        c.overlap.overlap_core_mask,
                    )
                    .outside
                });
                while eligible_lanes != 0 {
                    let lane = eligible_lanes.trailing_zeros() as u8;
                    eligible_lanes &= eligible_lanes - 1;
                    let is_sandbox = ((active_cut.inside >> lane) & 1) != 0;
                    let granule_mask = branchless_select_u8(
                        is_sandbox,
                        sandbox_granule_mask,
                        standard_granule_mask,
                    );
                    if granule_mask != 0 {
                        if let Some((slot_w, weight)) = self.lane_words_and_weight(lane) {
                            if weight > 0.0 {
                                let s = branchless_clamp_f32(weight * 64.0, 0.0, 64.0) as u64;
                                if s != 0 {
                                    let bias_restore =
                                        (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                    let mut scaled_w = [0u64; NUM_STIGMERGIC_GRANULES];
                                    let mut g = 0usize;
                                    while g < NUM_STIGMERGIC_GRANULES {
                                        let g_active =
                                            (((granule_mask >> g) & 1) as u64).wrapping_neg();
                                        scaled_w[g] = swar_q7_scale_le128_with_bias_x8(
                                            slot_w[g],
                                            s,
                                            bias_restore,
                                        ) & g_active;
                                        g += 1;
                                    }
                                    swar_or_avx2_q7_add_sat_words_64(&mut frm_w, &scaled_w);
                                }
                            }
                        }
                    }
                }
            }
        }
        if (ledger_slot.auto_tune_flags & LEDGER_FLAG_COGNITIVE_TRAJECTORY) != 0 {
            let circuit = CognitiveTrajectoryLedger::from_ledger_slot(ledger_slot);
            let ref_w = if trail_out.step_count > 0 {
                trail_phase_words_ref(trail_out)
            } else {
                &frm_w
            };
            let _ = circuit.modulate_gateway_frame_at_scale(&mut frame, &frm_w, ref_w, scale);
        }
        let report = frame.to_causal_report();
        let (head_dim, num_heads, _expected_len) = self.validate_substrate_cascade_params(
            geom.head_dim as u32,
            geom.num_heads as u32,
            frame.phase_step_min_rad,
            frame.phase_clamp_rad,
            query_tensor.as_deref(),
            None,
        )?;
        if geom.head_dim != head_dim || geom.num_heads != num_heads {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        // Evaporate stale pheromone and deposit new pheromone weighted by resonance energy and
        // active-head trajectory self-coherence (`prev_w` vs `frm_w`) so multi-layer traversal
        // dynamically transitions across Express, sub-head router, Rank-1 residual, and Anchor instead of saturating.
        let prev_rho =
            branchless_flush_subnormal_f32(branchless_relu_f32(trail_out.stigmergic_density));
        let energy_gate = branchless_clamp_f32(frame.effective_energy, 0.0, 1.0);
        let prev_w = *trail_phase_words_ref(trail_out);
        let frm_nonzero = swar_words_nonzero_64(&frm_w);
        let trail_coh = if trail_out.step_count == 0 || !frm_nonzero {
            1.0_f32
        } else {
            tangent_cosine_active_heads_64(&prev_w, &frm_w, frame.active_head_count)
        };
        let traj_gate = if trail_out.step_count == 0 {
            1.0_f32
        } else if frame.active_head_count < MAX_ATTENTION_HEADS as u16 {
            0.38_f32 + 0.52_f32 * branchless_relu_f32(trail_coh)
        } else {
            0.50_f32 + 0.50_f32 * branchless_relu_f32(trail_coh)
        };
        let should_step_trail =
            query_tensor.is_some() || rank1_hidden.is_none() || (trail_out.step_count == 0);
        let next_rho = if should_step_trail {
            let base_retain =
                branchless_clamp_f32(branchless_relu_f32(thresholds.evaporation_retain), 0.0, 1.0);
            let retain = if trail_out.step_count > 0
                && frame.active_head_count < MAX_ATTENTION_HEADS as u16
            {
                base_retain * (0.85_f32 + 0.15_f32 * branchless_relu_f32(trail_coh))
            } else {
                base_retain
            };
            let deposit = branchless_relu_f32(thresholds.reinforcement_gain)
                * (energy_gate * energy_gate)
                * (traj_gate * traj_gate);
            branchless_min_f32(
                branchless_flush_subnormal_f32(prev_rho * retain + deposit),
                MAX_STIGMERGIC_DENSITY,
            )
        } else {
            prev_rho
        };

        let (bypass_hit, effective_ptr) =
            self.resolve_or_pass_tagged_ptr(ledger_slot, TaggedPtr(frame.tagged_ptr));
        let next_w = if frm_nonzero {
            *trail_phase_words_mut(trail_out) = frm_w;
            frm_w
        } else {
            prev_w
        };
        if frame.target_anchor_id != 0 {
            trail_out.target_anchor_id = frame.target_anchor_id;
            trail_out.cluster_id = frame.target_anchor_id;
        } else {
            trail_out.cluster_id = frame.dominant_slot as u32;
        }
        trail_out.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_out.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        if should_step_trail {
            let step_ptr = if frm_nonzero {
                effective_ptr.with_phase(0)
            } else {
                effective_ptr
            };
            self.step_pointer(step_ptr.raw(), trail_out)?;
        }
        trail_out.stigmergic_density = next_rho;
        trail_out.current_ptr = effective_ptr.raw();

        let is_repulsed = report.is_cohomology_repulsion()
            | ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0);
        let coh = if is_repulsed {
            -1.0
        } else {
            branchless_clamp_f32(
                branchless_max_f32(frame.fact_persona_coherence, 0.25) * energy_gate,
                0.0,
                1.0,
            )
        };
        let (alpha_q7, noise_tol_q7) = match core_ctx {
            Some(c) => (
                c.derivative_alpha_q7,
                c.stack.stigmergy.mean_noise_tolerance_q7(),
            ),
            None => (0, 4),
        };
        let prev_rising = (trail_out.flags & TRAIL_FLAG_PHEROMONE_RISING) != 0;
        let (effective_rho, now_rising) = StigmergicThresholds::compute_effective_density_calculus(
            prev_rho,
            next_rho,
            energy_gate,
            alpha_q7,
            noise_tol_q7,
            prev_rising,
        );
        let (decision, _mode_flags, _target_slot, extra_flags) = thresholds.select_action_decision(
            effective_rho,
            entropy_h_tilde,
            coh,
            is_repulsed,
            frame.dominant_slot,
        );
        let rising_flag = branchless_select_u32(now_rising, TRAIL_FLAG_PHEROMONE_RISING, 0);
        trail_out.flags = (trail_out.flags
            & !(TRAIL_FLAG_REPULSION_ACTIVE
                | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE
                | TRAIL_FLAG_PHEROMONE_RISING))
            | extra_flags
            | rising_flag;

        let mut metrics = branchless_select_u8(bypass_hit, LEDGER_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(is_repulsed, LEDGER_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(
                (frame.lineage_flags
                    & (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA))
                    == (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA),
                LEDGER_METRIC_SLOT_SYNERGY_ACTIVE,
                0,
            );

        let is_passive_or_skip = matches!(
            decision,
            StigmergicActionDecision::PassiveSidecar | StigmergicActionDecision::InertialLayerSkip
        ) || ((frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0);

        if is_passive_or_skip {
            if matches!(decision, StigmergicActionDecision::InertialLayerSkip) {
                metrics |= LEDGER_METRIC_LAYER_SKIP_COMPENSATED;
            }
        } else {
            if let Some(q) = query_tensor.as_deref_mut() {
                let clean_damping =
                    branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
                let hub_damping_u7 =
                    if matches!(decision, StigmergicActionDecision::RepulsionEscape) {
                        (clean_damping >> 1).max(1)
                    } else {
                        clean_damping
                    };
                let swar_words =
                    filter_and_damp_gateway_words(&next_w, frame.phase_step_min_q7, hub_damping_u7);
                if swar_words_nonzero_64(&swar_words) {
                    let mut q7_stack = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
                    store_q7_words_64(&mut q7_stack.data, &swar_words);
                    let p_aligned = AlignedSlice16::new(&q7_stack.data[..num_heads])?;
                    if let Some(active_deltas) = Scoped::from_boundary(PhaseDeltaSlice(p_aligned))
                        .cut_by_lacuna(frame.phase_clamp_rad > 0.0, frame.phase_clamp_rad, 0.0)
                        .into_active()
                    {
                        let steering = crate::stigmergy::PhaseSteeringEngine::new(
                            head_dim as u32,
                            num_heads as u32,
                        )?;
                        PhaseSteeringActiveExt::apply_phase_steering_for_geometry_safe_active(
                            &steering,
                            geometry,
                            q,
                            active_deltas,
                            frame.phase_clamp_rad,
                        )?;
                        metrics |= LEDGER_METRIC_PHASE_STEERED;
                    }
                }
            }

            if matches!(
                decision,
                StigmergicActionDecision::FullOrchestration
                    | StigmergicActionDecision::CrystallizeMacroBasin
            ) {
                if let Some((y_mut, x_in, u_vec, v_vec)) = rank1_hidden {
                    let _ =
                        apply_gateway_frame_rank1_hidden_fused(&frame, y_mut, x_in, u_vec, v_vec)?;
                    metrics |= LEDGER_METRIC_RANK1_INJECTED;
                }
            }

            if let Some(logits) = logits_opt {
                if !logits.is_empty() && frame.effective_inv_temp > 0.0 {
                    scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
                }
            }
        }

        let _ = self.telemetry_push(ledger_slot, metrics);
        Ok((frame, report, decision))
    }

    /// Scale-, policy-, and core-layer-aware layer orchestrator allowing per-layer traversal across
    /// `MacroCoalition -> TypeGateway -> CellTrail -> PointerBit` while connecting dynamic trait
    /// cognitive apertures and the 3-layer core stack (`Liminal` + `Stigmergic` + `Ego`) directly into
    /// per-layer substrate RoPE/Rank-1 orchestration.
    ///
    /// # Errors
    /// Propagates any error from [`Self::orchestrate_gateway_layer_with_scale_and_core_layers_safe`].
    #[inline]
    fn orchestrate_gateway_layer_with_core_layers_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
        P: StigmergicThresholdPolicy + ?Sized,
        C: SystemConstraintSignal + ?Sized,
    >(
        &mut self,
        signal: &S,
        geometry: &G,
        core_ctx: Option<CoreLayerEvalContext<'_>>,
        policy: &P,
        system: &C,
        probe: &ManifoldProbeBaseline,
        scale: StigmergicScaleGranularity,
        entropy_h_tilde: f32,
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        unresolved_damping_u7: u8,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        rank1_hidden: Option<(
            &mut AlignedSliceMut32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
        )>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(
        GatewayFrame,
        CausalProvenanceReport,
        StigmergicActionDecision,
    )> {
        let circuit_opt = ((ledger_slot.auto_tune_flags & LEDGER_FLAG_COGNITIVE_TRAJECTORY) != 0)
            .then(|| CognitiveTrajectoryLedger::from_ledger_slot(ledger_slot));
        let dyn_thresholds = policy.resolve_thresholds_with_circuit(
            system,
            probe,
            trail_out,
            entropy_h_tilde,
            circuit_opt.as_ref(),
            scale,
        );
        self.orchestrate_gateway_layer_with_scale_and_core_layers_safe(
            signal,
            geometry,
            core_ctx,
            &dyn_thresholds,
            scale,
            entropy_h_tilde,
            ledger_slot,
            trail_out,
            unresolved_damping_u7,
            query_tensor,
            rank1_hidden,
            logits_opt,
        )
    }
}

impl<T: StigmergicSubstrateOps + ?Sized> SubstrateGatewayExt for T {}

/// Blanket-implemented extension trait orchestrating [`StigmergicScaleLayer`] projection,
/// [`CognitiveTrajectoryLedger`], and scale-aware swarm resolution in zero-allocation mode.
pub trait ScaleLayerOrchestrator: StigmergicSubstrateOps + SubstrateGatewayExt {
    /// Projects `layer` across `input_words`, resolves the active pointer trail in the Micro-Swarm
    /// at the activated granularity tier, and records the emitted [`CognitiveStepOp`] into
    /// `ledger_slot.telemetry_mini_ledger`.
    ///
    /// # Errors
    /// Propagates any error from [`crate::stigmergy::PointerSubstrate::step_pointer`] or
    /// [`crate::stigmergy::FastPathLedger::telemetry_push`].
    #[inline]
    fn orchestrate_scale_layer(
        &mut self,
        layer: &(impl StigmergicScaleLayer + ?Sized),
        input_ptr: TaggedPtr,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        layer_index: usize,
        total_layers: usize,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
    ) -> LmnkResult<(
        ScaleLayerProjection,
        CognitiveTrajectoryLedger,
        StigmergicReconstitutionReport,
    )> {
        let mut ledger = CognitiveTrajectoryLedger::from_ledger_slot(ledger_slot);
        let proj = layer.project_scale_layer(input_words, &ledger, layer_index, total_layers);
        let eff_ptr = TaggedPtr(if input_ptr.is_null() {
            proj.frame.tagged_ptr
        } else {
            input_ptr.raw()
        });
        let (report, _resolved_w) = trail_out.resolve_swarm_with_scale_layer(
            eff_ptr,
            &proj.peeled_words,
            proj.granularity,
            layer_index,
            total_layers,
            proj.frame.active_swarm_cell_mask,
            bloom_filter,
        );
        self.step_pointer(report.evolved_ptr.raw(), trail_out)?;
        ledger.push_step(proj.emitted_step);
        ledger.write_to_ledger_slot(ledger_slot);
        ledger_slot.telemetry_ring_head = ledger_slot.telemetry_ring_head.wrapping_add(1);
        Ok((proj, ledger, report))
    }

    /// Executes an elastic `N`-layer cascade (`depth_n` in `1..=layers.len()`) across a slice of
    /// [`StigmergicScaleLayer`] stages using `AdaptiveThresholdPolicy::default()`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `layers` is empty or `depth_n == 0`, or
    /// propagates pointer/telemetry errors.
    #[inline]
    fn orchestrate_adaptive_depth_n<L: StigmergicScaleLayer>(
        &mut self,
        layers: &[L],
        depth_n: usize,
        input_ptr: TaggedPtr,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
    ) -> LmnkResult<(
        ScaleLayerProjection,
        CognitiveTrajectoryLedger,
        StigmergicReconstitutionReport,
    )> {
        let default_policy = AdaptiveThresholdPolicy::default();
        self.orchestrate_adaptive_depth_n_with_policy(
            layers,
            depth_n,
            input_ptr,
            input_words,
            ledger_slot,
            trail_out,
            bloom_filter,
            &default_policy,
        )
    }

    /// Executes an elastic `N`-layer cascade (`depth_n` in `1..=layers.len()`) across a slice of
    /// [`StigmergicScaleLayer`] stages, deriving early-exit and recurrent-refinement thresholds
    /// from `circuit_policy`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `layers` is empty or `depth_n == 0`, or
    /// propagates pointer/telemetry errors.
    #[inline]
    fn orchestrate_adaptive_depth_n_with_policy<
        L: StigmergicScaleLayer,
        P: CognitiveCircuitThresholdPolicy + ?Sized,
    >(
        &mut self,
        layers: &[L],
        depth_n: usize,
        input_ptr: TaggedPtr,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        circuit_policy: &P,
    ) -> LmnkResult<(
        ScaleLayerProjection,
        CognitiveTrajectoryLedger,
        StigmergicReconstitutionReport,
    )> {
        #[inline(always)]
        const fn residual_mean_abs_q7(words: &[u64; NUM_STIGMERGIC_GRANULES]) -> u8 {
            let mut sum = 0u16;
            let mut c = 0usize;
            while c < NUM_STIGMERGIC_GRANULES {
                sum += swar_mean_u8_le128_x8(swar_q7_abs_u8_x8(words[c])) as u16;
                c += 1;
            }
            ((sum + 4) >> 3) as u8
        }

        let active_n = depth_n.min(layers.len());
        if active_n == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let fuse_floor = circuit_policy.early_exit_fuse_floor_q7();
        let repulse_ceil = circuit_policy.early_exit_repulse_ceiling_q7();
        let res_max = circuit_policy.early_exit_residual_max_q7();
        let refine_floor = circuit_policy.recurrent_refine_intensity_floor_q7();

        let mut last_proj = self.orchestrate_scale_layer(
            &layers[0],
            input_ptr,
            input_words,
            ledger_slot,
            trail_out,
            0,
            active_n,
            bloom_filter,
        )?;
        let mut combined_peeled = last_proj.0.peeled_words;
        let mut residual = last_proj.0.residual_words;
        let mut ledger = last_proj.1;

        let mut i = 1usize;
        while i < active_n {
            let act = last_proj.0.activation;
            let converged_early = matches!(
                act.op_kind(),
                CognitiveOpKind::SynthesizeFuse | CognitiveOpKind::CrystallizeCommit
            ) && act.fuse_gate_q7 >= fuse_floor
                && act.repulse_sensitivity_q7 < repulse_ceil
                && residual_mean_abs_q7(&residual) <= res_max;
            if converged_early {
                break;
            }

            let step_res = self.orchestrate_scale_layer(
                &layers[i],
                last_proj.2.evolved_ptr,
                &residual,
                ledger_slot,
                trail_out,
                i,
                active_n,
                bloom_filter,
            )?;
            let mut c = 0usize;
            while c < NUM_STIGMERGIC_GRANULES {
                combined_peeled[c] =
                    swar_q7_circular_add_x8(combined_peeled[c], step_res.0.peeled_words[c]);
                c += 1;
            }
            residual = step_res.0.residual_words;
            ledger = step_res.1;
            last_proj = step_res;
            i += 1;
        }

        // Bounded single-step recurrent residual refinement when high-intensity peeling or guard
        // repulsion persists at the end of a multi-layer cascade.
        let tail_act = last_proj.0.activation;
        if active_n >= 2
            && matches!(
                tail_act.op_kind(),
                CognitiveOpKind::DecomposePeel | CognitiveOpKind::RepulseAudit
            )
            && tail_act.intensity_q7 >= refine_floor
            && residual_mean_abs_q7(&residual) > res_max
        {
            let refine_res = self.orchestrate_scale_layer(
                &layers[active_n - 1],
                last_proj.2.evolved_ptr,
                &residual,
                ledger_slot,
                trail_out,
                active_n - 1,
                active_n,
                bloom_filter,
            )?;
            let mut c = 0usize;
            while c < NUM_STIGMERGIC_GRANULES {
                combined_peeled[c] =
                    swar_q7_circular_add_x8(combined_peeled[c], refine_res.0.peeled_words[c]);
                c += 1;
            }
            residual = refine_res.0.residual_words;
            ledger = refine_res.1;
            last_proj = refine_res;
        }

        let mut final_proj = last_proj.0;
        final_proj.peeled_words = combined_peeled;
        final_proj.residual_words = residual;
        final_proj.total_layers = active_n;
        Ok((final_proj, ledger, last_proj.2))
    }
}

impl<T: StigmergicSubstrateOps + SubstrateGatewayExt + ?Sized> ScaleLayerOrchestrator for T {}

/// Hierarchical scale-specialized hybrid circuit pairing a micro-scale (`PointerBit`/`CellTrail`)
/// [`CognitiveTrajectoryLedger`] with a macro-scale (`TypeGateway`/`MacroCoalition`) ledger.
///
/// Implements [`ManifoldGatewaySignal`] and [`StigmergicScaleLayer`] directly so callers can compose
/// analytical micro-peeling with integrative macro-fusion in zero-allocation cascades.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HybridScaleCircuit {
    /// Micro-scale circuit governing fine-grained decomposition and sub-head routing.
    pub micro_circuit: CognitiveTrajectoryLedger,
    /// Macro-scale circuit governing basin fusion and guard repulsion.
    pub macro_circuit: CognitiveTrajectoryLedger,
}

impl HybridScaleCircuit {
    /// Creates a new [`HybridScaleCircuit`] from distinct micro and macro trajectory ledgers.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        micro_circuit: CognitiveTrajectoryLedger,
        macro_circuit: CognitiveTrajectoryLedger,
    ) -> Self {
        Self {
            micro_circuit,
            macro_circuit,
        }
    }

    /// Selects the active ledger corresponding to `granularity`.
    #[inline(always)]
    #[must_use]
    pub const fn circuit_for_scale(
        &self,
        granularity: StigmergicScaleGranularity,
    ) -> &CognitiveTrajectoryLedger {
        match granularity {
            StigmergicScaleGranularity::PointerBit | StigmergicScaleGranularity::CellTrail => {
                &self.micro_circuit
            }
            StigmergicScaleGranularity::TypeGateway
            | StigmergicScaleGranularity::MacroCoalition => &self.macro_circuit,
        }
    }

    /// Flattens this hierarchical hybrid into a single blended 8-byte [`CognitiveTrajectoryLedger`].
    #[inline]
    #[must_use]
    pub fn to_blended_ledger(&self, macro_weight_q7: u8) -> CognitiveTrajectoryLedger {
        self.micro_circuit
            .compose_persona_circuit(&self.macro_circuit, macro_weight_q7)
    }
}

impl RawPhaseProjectable for HybridScaleCircuit {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.to_blended_ledger(64).project_to_q7_64()
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.to_blended_ledger(64).to_phase_words_64()
    }
}

impl CausalProvenanceSignal for HybridScaleCircuit {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.micro_circuit.causal_provenance_bits() | self.macro_circuit.causal_provenance_bits()
    }
}

impl CognitiveGroundingSignal for HybridScaleCircuit {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.project_to_q7_64()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        branchless_max_f32(
            self.micro_circuit.grounding_base_energy(),
            self.macro_circuit.grounding_base_energy(),
        )
    }
}

impl SubjectivePersonaSignal for HybridScaleCircuit {}

impl ManifoldGatewaySignal for HybridScaleCircuit {
    #[inline]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.to_blended_ledger(64).synthesize_gateway_frame()
    }
}

/// Output of projecting an input SWAR phase field through one or more [`StigmergicScaleLayer`] tiers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleLayerProjection {
    /// Composite 128-byte [`GatewayFrame`] synthesized at this layer boundary.
    pub frame: GatewayFrame,
    /// Orthogonal contribution explained/peeled by this layer across 64 heads.
    pub peeled_words: [u64; NUM_STIGMERGIC_GRANULES],
    /// Remaining unexplained Q7 phase residual forwarded to layer `layer_index + 1`.
    pub residual_words: [u64; NUM_STIGMERGIC_GRANULES],
    /// Granularity tier executed at this projection step.
    pub granularity: StigmergicScaleGranularity,
    /// Zero-based starting layer index across the active stack (`0..total_layers`).
    pub layer_index: usize,
    /// Total effective layer count $N$ across the active cascade.
    pub total_layers: usize,
    /// Stigmergic cognitive gating activation computed from trajectory history and phase resonance.
    pub activation: CognitiveCircuitActivation,
    /// Compact 1-byte cognitive step emitted for recording into [`CognitiveTrajectoryLedger`].
    pub emitted_step: CognitiveStepOp,
}

/// Branchless core helper projecting `signal` at a specific [`StigmergicScaleGranularity`] and layer
/// depth ratio `layer_index / total_layers` with zero dynamic allocations.
#[inline]
pub fn project_gateway_signal_at_scale<S: ManifoldGatewaySignal + ?Sized>(
    signal: &S,
    granularity: StigmergicScaleGranularity,
    input_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ledger: &CognitiveTrajectoryLedger,
    layer_index: usize,
    total_layers: usize,
) -> ScaleLayerProjection {
    let safe_total = if total_layers == 0 { 1 } else { total_layers };
    let (mut frame, frm_w) = signal.synthesize_gateway_frame_with_words();
    let activation = ledger.evaluate_circuit_activation(&frm_w, input_words, granularity);
    let base_gain_q7 = granularity.layer_peel_gain_q7(layer_index, safe_total);
    let gate_mod = if activation.op_kind() == CognitiveOpKind::RepulseAudit {
        activation.repulse_sensitivity_q7
    } else {
        ((u16::from(base_gain_q7) + u16::from(activation.intensity_q7)) >> 1) as u8
    };
    let s_u64 = u64::from(gate_mod.min(128));
    let bias = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let active_mask = match granularity {
        StigmergicScaleGranularity::PointerBit => 1u8 << (layer_index & 7),
        StigmergicScaleGranularity::CellTrail => {
            if frame.active_swarm_cell_mask == 0 {
                0x0F
            } else {
                frame.active_swarm_cell_mask
            }
        }
        StigmergicScaleGranularity::TypeGateway | StigmergicScaleGranularity::MacroCoalition => {
            0xFF
        }
    };

    let mut peeled_words = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut residual_words = *input_words;
    let mut c = 0usize;
    while c < NUM_STIGMERGIC_GRANULES {
        if ((active_mask >> c) & 1) != 0 {
            let scaled = swar_q7_scale_le128_with_bias_x8(frm_w[c], s_u64, bias);
            peeled_words[c] = scaled;
            residual_words[c] = swar_q7_circular_diff_x8(input_words[c], scaled);
        }
        c += 1;
    }

    let eff_op = if frame.is_repulsed() {
        CognitiveOpKind::RepulseAudit
    } else {
        activation.op_kind()
    };
    let intensity_0_to_7 = (activation.intensity_q7 >> 4).min(7);
    let emitted_step = CognitiveStepOp::pack(granularity, eff_op, intensity_0_to_7);
    frame.active_swarm_cell_mask = active_mask;

    ScaleLayerProjection {
        frame,
        peeled_words,
        residual_words,
        granularity,
        layer_index,
        total_layers: safe_total,
        activation,
        emitted_step,
    }
}

/// Core Layer Management Trait rooted directly in [`ManifoldGatewaySignal`].
///
/// Governs multi-scale granularity (`PointerBit` -> `CellTrail` -> `TypeGateway` -> `MacroCoalition`),
/// structural depth composition (`N +/- 1` via tuples `(A, B)`, arrays `[T; N]`, or slices), and
/// orthogonal residual peeling guided by [`CognitiveTrajectoryLedger`].
pub trait StigmergicScaleLayer: ManifoldGatewaySignal {
    /// Returns the primary [`StigmergicScaleGranularity`] owned by this layer.
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::TypeGateway
    }

    /// Returns the structural layer count `N >= 1` composed within this value.
    #[inline(always)]
    fn layer_count(&self) -> usize {
        1
    }

    /// Projects `input_words` through this layer at `(layer_index, total_layers)` using
    /// `ledger`, returning the peeled field, orthogonal residual, and emitted cognitive step.
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        project_gateway_signal_at_scale(
            self,
            self.scale_granularity(),
            input_words,
            ledger,
            layer_index,
            total_layers.max(self.layer_count()),
        )
    }
}

/// Zero-cost explicit-scale wrapper promoting any [`ManifoldGatewaySignal`] into a
/// [`StigmergicScaleLayer`] pinned to `granularity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleLayerNode<T> {
    /// Underlying signal implementing [`ManifoldGatewaySignal`].
    pub signal: T,
    /// Assigned [`StigmergicScaleGranularity`] tier inside a multi-layer stack.
    pub granularity: StigmergicScaleGranularity,
}

impl<T> ScaleLayerNode<T> {
    /// Wraps `signal` at `granularity`.
    #[inline(always)]
    #[must_use]
    pub const fn new(signal: T, granularity: StigmergicScaleGranularity) -> Self {
        Self {
            signal,
            granularity,
        }
    }
}

impl<T: CausalProvenanceSignal> CausalProvenanceSignal for ScaleLayerNode<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.signal.causal_provenance_bits()
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.signal.causal_anomaly_severity()
    }
}

impl<T: ManifoldGatewaySignal> ManifoldGatewaySignal for ScaleLayerNode<T> {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.signal.synthesize_gateway_frame()
    }

    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        self.signal.synthesize_gateway_frame_with_words()
    }
}

impl<T: ManifoldGatewaySignal> StigmergicScaleLayer for ScaleLayerNode<T> {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        self.granularity
    }
}

impl<T: StigmergicScaleLayer + ?Sized> StigmergicScaleLayer for &T {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        (**self).scale_granularity()
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        (**self).layer_count()
    }

    #[inline(always)]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        (**self).project_scale_layer(input_words, ledger, layer_index, total_layers)
    }
}

impl<
        A: StigmergicScaleLayer + ObjectiveFactSignal,
        B: StigmergicScaleLayer + SubjectivePersonaSignal,
    > StigmergicScaleLayer for (A, B)
{
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        let ga = self.0.scale_granularity();
        let gb = self.1.scale_granularity();
        if gb > ga {
            gb
        } else {
            ga
        }
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        self.0.layer_count() + self.1.layer_count()
    }

    #[inline]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        let a_layers = self.0.layer_count();
        let total = total_layers.max(self.layer_count());
        let proj_a = self
            .0
            .project_scale_layer(input_words, ledger, layer_index, total);
        let mut mid_ledger = *ledger;
        mid_ledger.push_step(proj_a.emitted_step);
        let proj_b = self.1.project_scale_layer(
            &proj_a.residual_words,
            &mid_ledger,
            layer_index + a_layers,
            total,
        );
        let (frame, _w) = self.synthesize_gateway_frame_with_words();
        let mut combined_peeled = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut c = 0usize;
        while c < NUM_STIGMERGIC_GRANULES {
            combined_peeled[c] =
                swar_q7_circular_add_x8(proj_a.peeled_words[c], proj_b.peeled_words[c]);
            c += 1;
        }
        ScaleLayerProjection {
            frame,
            peeled_words: combined_peeled,
            residual_words: proj_b.residual_words,
            granularity: proj_b.granularity,
            layer_index,
            total_layers: total,
            activation: proj_b.activation,
            emitted_step: proj_b.emitted_step,
        }
    }
}

impl<T: StigmergicScaleLayer> StigmergicScaleLayer for [T] {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        if self.is_empty() {
            StigmergicScaleGranularity::TypeGateway
        } else {
            self[0].scale_granularity()
        }
    }

    #[inline]
    fn layer_count(&self) -> usize {
        let mut total = 0usize;
        let mut i = 0usize;
        while i < self.len() {
            total += self[i].layer_count();
            i += 1;
        }
        total.max(1)
    }

    #[inline]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        if self.is_empty() {
            return ScaleLayerProjection {
                frame: GatewayFrame::default(),
                peeled_words: [0u64; NUM_STIGMERGIC_GRANULES],
                residual_words: *input_words,
                granularity: StigmergicScaleGranularity::TypeGateway,
                layer_index,
                total_layers: total_layers.max(1),
                activation: CognitiveCircuitActivation::default(),
                emitted_step: CognitiveStepOp::default(),
            };
        }
        let own_layers = self.layer_count();
        let total = total_layers.max(own_layers);
        let mut cur_residual = *input_words;
        let mut cur_ledger = *ledger;
        let mut combined_peeled = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut cursor = layer_index;
        let mut last_proj = self[0].project_scale_layer(&cur_residual, &cur_ledger, cursor, total);
        let mut c = 0usize;
        while c < NUM_STIGMERGIC_GRANULES {
            combined_peeled[c] = last_proj.peeled_words[c];
            c += 1;
        }
        cur_residual = last_proj.residual_words;
        cur_ledger.push_step(last_proj.emitted_step);
        cursor += self[0].layer_count();
        let mut i = 1usize;
        while i < self.len() {
            let p = self[i].project_scale_layer(&cur_residual, &cur_ledger, cursor, total);
            let mut k = 0usize;
            while k < NUM_STIGMERGIC_GRANULES {
                combined_peeled[k] = swar_q7_circular_add_x8(combined_peeled[k], p.peeled_words[k]);
                k += 1;
            }
            cur_residual = p.residual_words;
            cur_ledger.push_step(p.emitted_step);
            cursor += self[i].layer_count();
            last_proj = p;
            i += 1;
        }
        let (frame, _w) = self.synthesize_gateway_frame_with_words();
        ScaleLayerProjection {
            frame,
            peeled_words: combined_peeled,
            residual_words: cur_residual,
            granularity: last_proj.granularity,
            layer_index,
            total_layers: total,
            activation: last_proj.activation,
            emitted_step: last_proj.emitted_step,
        }
    }
}

impl<T: StigmergicScaleLayer, const N: usize> StigmergicScaleLayer for [T; N] {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        self.as_slice().scale_granularity()
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        self.as_slice().layer_count()
    }

    #[inline(always)]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        self.as_slice()
            .project_scale_layer(input_words, ledger, layer_index, total_layers)
    }
}

impl StigmergicScaleLayer for GatewayFrame {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::TypeGateway
    }
}

impl<P: RawPhaseProjectable + Clone> StigmergicScaleLayer for ObjectiveFact<P> {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::CellTrail
    }
}

impl<P: RawPhaseProjectable + Clone> StigmergicScaleLayer for SubjectivePersona<P> {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::MacroCoalition
    }
}

impl StigmergicScaleLayer for GuardPhaseAnchor {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::TypeGateway
    }
}

impl StigmergicScaleLayer for PhaseAnchorRecord {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::MacroCoalition
    }
}

impl StigmergicScaleLayer for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        self.latest_step()
            .map(CognitiveStepOp::granularity)
            .unwrap_or(StigmergicScaleGranularity::TypeGateway)
    }
}

impl StigmergicScaleLayer for HybridScaleCircuit {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::CellTrail
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        2
    }

    #[inline]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        let g = StigmergicScaleGranularity::from_layer_ratio(layer_index, total_layers);
        let active_circuit = self.circuit_for_scale(g);
        let merged = active_circuit.compose_persona_circuit(ledger, 64);
        project_gateway_signal_at_scale(
            active_circuit,
            g,
            input_words,
            &merged,
            layer_index,
            total_layers,
        )
    }
}
