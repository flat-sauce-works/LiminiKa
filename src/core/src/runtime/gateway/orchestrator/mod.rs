// name: src/core/src/runtime/gateway/orchestrator/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gateway Orchestrator extensions (`CoreLayerEvalContext`, [`GatewayPhaseSteeringExt`],
//! [`SubstrateGatewayExt`], [`ScaleLayerOrchestrator`], [`HybridScaleCircuit`], and [`StigmergicScaleLayer`]).

use super::*;
use crate::drivers::{DistortionTension, EgoDifferentialNet, EnvelopeState};
use crate::foundation::{
    ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE, ACTION_MODE_SUBHEAD_ROUTE,
    ATTRACTOR_SLOT_ID_MASK, AlignedBuffer16, AlignedSlice16, AlignedSlice32, AlignedSliceMut32,
    DEFAULT_PHASE_CLAMP_MAX_RAD, PhaseSteering, RopePairLayout, RopeSteeringTarget,
    SWAR_SPLAT_BYTE_64, TaggedPtr, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_max_f32, branchless_min_f32, branchless_relu_f32,
    dispatch_fused_phase_clamp_and_subhead_words_core,
    rotate_fused_self_phase_clamp_and_subhead_default_words,
    rotate_fused_self_phase_clamp_subhead_q_and_clamp_k_default_words, scale_f32_slice_unrolled8,
    steer_from_swar_words_core_safe, store_q7_words_64, swar_or_avx2_q7_add_sat_words_64,
    swar_q7_phase_step_filter_words_64, swar_q7_scale_le128_with_bias_x8, swar_words_nonzero_64,
};
use crate::liminal::{
    LMNK_ERR_INVALID_ARGUMENT, LacunaPartition8, LmnkResult, MAX_ATTENTION_HEADS,
    NUM_STIGMERGIC_GRANULES, Scoped, branchless_select_u8, branchless_select_u32,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::context::CoreStateStack;
use crate::runtime::horizon::EpistemicHorizonReading;
use crate::stigmergy::{
    CognitiveTrajectoryLedger, ContrastivePhaseBloomFilter, LEDGER_FLAG_COGNITIVE_TRAJECTORY,
    LEDGER_METRIC_FAST_PATH_HIT, LEDGER_METRIC_LAYER_SKIP_COMPENSATED, LEDGER_METRIC_PHASE_STEERED,
    LEDGER_METRIC_RANK1_INJECTED, LEDGER_METRIC_REPULSION_APPLIED,
    LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, LedgerSlot, MAX_ATTRACTOR_PULL_FORCE,
    MAX_STIGMERGIC_DENSITY, OverlapDiscoveryReport, PhaseDeltaSlice, PhaseSteeringActiveExt,
    PointerTrail, ResidualAdapterEngine, ResidualScale, StigmergicActionDecision,
    StigmergicScaleGranularity, StigmergicSubstrateOps, StigmergicThresholds,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_PHEROMONE_RISING, TRAIL_FLAG_REPULSION_ACTIVE,
    trail_phase_words_mut, trail_phase_words_ref, validate_optional_qk_tensors,
};

/// Shape and damping parameters for [`accumulate_trail_and_steer_gateway_words_core`].
#[derive(Clone, Copy)]
pub(crate) struct GatewaySteerShape {
    pub unresolved_damping_u7: u8,
    pub head_dim: usize,
    pub num_heads: usize,
}

#[inline]
pub(crate) fn accumulate_trail_and_steer_gateway_words_core(
    frame: &GatewayFrame,
    frm_w: &[u64; NUM_STIGMERGIC_GRANULES],
    trail_inout: &mut PointerTrail,
    shape: GatewaySteerShape,
    query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> LmnkResult<(bool, bool, bool)> {
    let GatewaySteerShape {
        unresolved_damping_u7,
        head_dim,
        num_heads,
    } = shape;
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

    if let Some(logits) = logits_opt
        && !logits.is_empty()
    {
        scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
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

/// Borrowed read-only core context (`2 Primitives x 3 Drivers`) passed into
/// per-layer Hot-Path RoPE steering and substrate orchestration.
#[derive(Debug, Clone, Copy)]
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
    /// Optional [`EnvelopeState`] physical `pinned` isolation barrier.
    pub envelope: Option<&'a EnvelopeState>,
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
    if let Some(ctx) = core_ctx
        && (ctx.stack.liminal.adaptation_epochs > 0 || ctx.stack.ego.is_active())
    {
        let mut frame = signal.synthesize_gateway_with_aperture_from_stack(ctx.stack);
        if ctx.stack.ego.is_active() {
            // Compute the pure I-Frame baseline BEFORE `LiminalEgoState` and `EgoDifferentialNet`
            // mutate the frame so `pinned` isolation always anchors to pre-Ego state.
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
            if let (Some(envelope), Some(vp), Some(net), Some(tension)) =
                (ctx.envelope, ctx.lacuna_partition, ctx.ego_net, ctx.tension)
            {
                let _ = net.evaluate_pframe_chain_branchless(
                    &ctx.stack.ego,
                    envelope,
                    vp,
                    &ctx.stack.stigmergy,
                    tension,
                    &mut frame,
                );
                envelope.enforce_pinned_isolation_branchless(&sys_baseline, &mut frame);
                let w = *frame.phase_words_ref();
                return (frame, w);
            }
            if let Some(envelope) = ctx.envelope {
                envelope.enforce_pinned_isolation_branchless(&sys_baseline, &mut frame);
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
            GatewaySteerShape {
                unresolved_damping_u7,
                head_dim,
                num_heads,
            },
            Some(query_tensor),
            key_tensor,
            logits_opt,
        )?;
        Ok((frame, report))
    }

    /// Synthesizes any [`ManifoldGatewaySignal`] with dynamic trait cognitive apertures and optional
    /// read-only 3-layer modulation (`Liminal` + `Stigmergic` + `Ego`), then steers `query_tensor`
    /// (and optional `key_tensor`) according to [`RopeSteeringTarget`] (`Interleaved` or `HalfSplit`,
    /// including partial `rotary_dim`) with zero heap allocation.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `input.geometry`, radian clamps, or tensor slices
    /// violate layout or `LMNK_RESTRICT` disjointness invariants.
    #[inline]
    fn steer_gateway_signal_with_core_layers_for_geometry_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &self,
        input: GatewaySteerInput<'_, S, G>,
        trail_inout: &mut PointerTrail,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(GatewayFrame, CausalProvenanceReport)> {
        let GatewaySteerInput {
            signal,
            geometry,
            core_ctx,
            unresolved_damping_u7,
        } = input;
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

        if let Some(logits) = logits_opt
            && !logits.is_empty()
        {
            scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
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

/// Read-only signal, geometry, and core-layer inputs for gateway phase steering.
#[derive(Debug)]
pub struct GatewaySteerInput<'a, S: ?Sized, G: ?Sized> {
    /// Manifold gateway signal to synthesize.
    pub signal: &'a S,
    /// RoPE steering target geometry descriptor.
    pub geometry: &'a G,
    /// Optional read-only 3-layer core stack evaluation context.
    pub core_ctx: Option<CoreLayerEvalContext<'a>>,
    /// Damping weight (`0..=128`) applied to unresolved heads.
    pub unresolved_damping_u7: u8,
}

impl<S: ?Sized, G: ?Sized> Clone for GatewaySteerInput<'_, S, G> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: ?Sized, G: ?Sized> Copy for GatewaySteerInput<'_, S, G> {}

/// Bundled hidden-state slices for fused Rank-1 residual injection during gateway layer orchestration.
pub struct Rank1HiddenSlices<'a, 'b> {
    /// Mutable output residual tensor (`y_out`).
    pub y_out: &'a mut AlignedSliceMut32<'b, f32>,
    /// Input hidden-state tensor (`x_in`).
    pub x_in: &'a AlignedSlice32<'b, f32>,
    /// Left Rank-1 factor vector (`u`).
    pub u_vec: &'a AlignedSlice32<'b, f32>,
    /// Right Rank-1 factor vector (`v`).
    pub v_vec: &'a AlignedSlice32<'b, f32>,
}

/// Mutable tensor targets for [`SubstrateGatewayExt`] layer orchestration passes.
pub struct GatewayLayerTargets<'a, 'q, 'r, 'l> {
    /// Optional mutable query tensor steered in-place.
    pub query_tensor: Option<&'a mut AlignedSliceMut32<'q, f32>>,
    /// Optional fused Rank-1 residual injection buffers.
    pub rank1_hidden: Option<Rank1HiddenSlices<'a, 'r>>,
    /// Optional mutable logits slice scaled by effective inverse temperature.
    pub logits_opt: Option<&'a mut AlignedSliceMut32<'l, f32>>,
}

/// Signal, geometry, scale, and entropy inputs for [`SubstrateGatewayExt`] layer orchestration.
#[derive(Debug)]
pub struct GatewayLayerInput<'a, S: ?Sized, G: ?Sized> {
    /// Manifold gateway signal to synthesize.
    pub signal: &'a S,
    /// RoPE steering target geometry descriptor.
    pub geometry: &'a G,
    /// Optional read-only 3-layer core stack evaluation context.
    pub core_ctx: Option<CoreLayerEvalContext<'a>>,
    /// Active stigmergic scale granularity tier.
    pub scale: StigmergicScaleGranularity,
    /// Current normalized entropy observation $\tilde{H}$.
    pub entropy_h_tilde: f32,
    /// Damping weight (`0..=128`) applied to unresolved heads.
    pub unresolved_damping_u7: u8,
}

impl<S: ?Sized, G: ?Sized> Clone for GatewayLayerInput<'_, S, G> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: ?Sized, G: ?Sized> Copy for GatewayLayerInput<'_, S, G> {}

/// Policy and system-constraint context for [`SubstrateGatewayExt::orchestrate_gateway_layer_with_core_layers_safe`].
#[derive(Debug)]
pub struct GatewayPolicyContext<'a, P: ?Sized, C: ?Sized> {
    /// Dynamic stigmergic threshold policy.
    pub policy: &'a P,
    /// System constraint telemetry signal.
    pub system: &'a C,
    /// Calibrated manifold probe baseline.
    pub probe: &'a ManifoldProbeBaseline,
}

impl<P: ?Sized, C: ?Sized> Clone for GatewayPolicyContext<'_, P, C> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P: ?Sized, C: ?Sized> Copy for GatewayPolicyContext<'_, P, C> {}

/// Shared input payload for [`ScaleLayerOrchestrator`] multi-scale cascades.
#[derive(Debug, Clone, Copy)]
pub struct ScaleCascadeInput<'a> {
    /// Incoming 64-bit tagged pointer (falls back to projected frame pointer when null).
    pub input_ptr: TaggedPtr,
    /// Incoming 64-head Q7 SWAR phase words.
    pub input_words: &'a [u64; NUM_STIGMERGIC_GRANULES],
    /// Optional contrastive phase Bloom filter for repulsion gating.
    pub bloom_filter: Option<&'a ContrastivePhaseBloomFilter>,
}

/// Blanket-implemented extension trait equipping any [`StigmergicSubstrateOps`] with single-call
/// [`ManifoldGatewaySignal`] ingestion, Online Toroidal $k$-Means / DP-Means macro clustering,
/// online basin consolidation, and `.lmnk` crystallization.
pub trait SubstrateGatewayExt: StigmergicSubstrateOps {
    /// Synthesizes `input.signal` with dynamic trait cognitive apertures and optional 3-layer core
    /// modulation (`input.core_ctx`), then steers `query_tensor` and optional `logits_opt` for `input.geometry`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`StigmergicSubstrateOps`] or [`PhaseSteeringActiveExt`].
    #[inline]
    fn ingest_gateway_and_steer_with_core_layers_for_geometry_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &mut self,
        input: GatewaySteerInput<'_, S, G>,
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(GatewayFrame, CausalProvenanceReport)> {
        let GatewaySteerInput {
            signal,
            geometry,
            core_ctx,
            unresolved_damping_u7,
        } = input;
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
        } else if let Some(q) = query_tensor {
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

        if let Some(logits) = logits_opt
            && !logits.is_empty()
            && frame.effective_inv_temp > 0.0
        {
            scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
        }

        let _ = self.telemetry_push(ledger_slot, metrics);
        Ok((frame, report))
    }

    /// Orchestrates scale-aware `input.signal` synthesis, 3-layer core modulation, and [`StigmergicSubstrateOps`]
    /// decision routing for `input.geometry`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`StigmergicSubstrateOps`].
    #[inline]
    fn orchestrate_gateway_layer_with_scale_and_core_layers_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &mut self,
        input: GatewayLayerInput<'_, S, G>,
        thresholds: &StigmergicThresholds,
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        targets: GatewayLayerTargets<'_, '_, '_, '_>,
    ) -> LmnkResult<(
        GatewayFrame,
        CausalProvenanceReport,
        StigmergicActionDecision,
    )> {
        let GatewayLayerInput {
            signal,
            geometry,
            core_ctx,
            scale,
            entropy_h_tilde,
            unresolved_damping_u7,
        } = input;
        let GatewayLayerTargets {
            query_tensor,
            rank1_hidden,
            logits_opt,
        } = targets;
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
                    if granule_mask != 0
                        && let Some((slot_w, weight)) = self.lane_words_and_weight(lane)
                        && weight > 0.0
                    {
                        let s = branchless_clamp_f32(weight * 64.0, 0.0, 64.0) as u64;
                        if s != 0 {
                            let bias_restore =
                                (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                            let mut scaled_w = [0u64; NUM_STIGMERGIC_GRANULES];
                            let mut g = 0usize;
                            while g < NUM_STIGMERGIC_GRANULES {
                                let g_active = (((granule_mask >> g) & 1) as u64).wrapping_neg();
                                scaled_w[g] =
                                    swar_q7_scale_le128_with_bias_x8(slot_w[g], s, bias_restore)
                                        & g_active;
                                g += 1;
                            }
                            swar_or_avx2_q7_add_sat_words_64(&mut frm_w, &scaled_w);
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
            if let Some(q) = query_tensor {
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
            ) && let Some(Rank1HiddenSlices {
                y_out: y_mut,
                x_in,
                u_vec,
                v_vec,
            }) = rank1_hidden
            {
                let _ = apply_gateway_frame_rank1_hidden_fused(&frame, y_mut, x_in, u_vec, v_vec)?;
                metrics |= LEDGER_METRIC_RANK1_INJECTED;
            }

            if let Some(logits) = logits_opt
                && !logits.is_empty()
                && frame.effective_inv_temp > 0.0
            {
                scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
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
        input: GatewayLayerInput<'_, S, G>,
        policy_ctx: GatewayPolicyContext<'_, P, C>,
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        targets: GatewayLayerTargets<'_, '_, '_, '_>,
    ) -> LmnkResult<(
        GatewayFrame,
        CausalProvenanceReport,
        StigmergicActionDecision,
    )> {
        let circuit_opt = ((ledger_slot.auto_tune_flags & LEDGER_FLAG_COGNITIVE_TRAJECTORY) != 0)
            .then(|| CognitiveTrajectoryLedger::from_ledger_slot(ledger_slot));
        let dyn_thresholds = policy_ctx.policy.resolve_thresholds_with_circuit(
            policy_ctx.system,
            policy_ctx.probe,
            trail_out,
            input.entropy_h_tilde,
            circuit_opt.as_ref(),
            input.scale,
        );
        self.orchestrate_gateway_layer_with_scale_and_core_layers_safe(
            input,
            &dyn_thresholds,
            ledger_slot,
            trail_out,
            targets,
        )
    }
}

impl<T: StigmergicSubstrateOps + ?Sized> SubstrateGatewayExt for T {}

mod scale_layer;
pub use scale_layer::*;
