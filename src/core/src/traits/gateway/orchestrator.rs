// name: src/core/src/traits/gateway/orchestrator.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Blanket extension traits wiring [`ManifoldGatewaySignal`] into [`PhaseSteering`],
//! [`Rank1ResidualAdapter`], and [`ActionHub`] orchestration.

use super::signals::*;
use super::circuit::*;
use super::*;

// ===================================================================
// 9. Blanket Extension Traits on `PhaseSteering + SubHeadRouter` & `ActionHub`
// ===================================================================

#[inline]
pub(crate) fn accumulate_trail_and_steer_gateway_words_core(
    frame: &GatewayFrame,
    frm_w: &[u64; NUM_STIGMERGIC_GRANULES],
    trail_inout: &mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    head_dim: usize,
    num_heads: usize,
    query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(bool, bool, bool)> {
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
        let swar_words = if frame.qdps_min_step_q7 <= 1 {
            next_w
        } else if frame.qdps_min_step_q7 > 127 {
            [0u64; NUM_STIGMERGIC_GRANULES]
        } else {
            swar_q7_qdps_filter_words_64(&next_w, frame.qdps_min_step_q7)
        };
        let clean_damping =
            branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);

        let has_nonzero_words = if frm_nonzero & (frame.qdps_min_step_q7 <= 1) {
            true
        } else {
            swar_words_nonzero_64(&swar_words)
        };
        if (frame.action_mode_flags & ACTION_MODE_SUBHEAD_ROUTE) != 0
            && frame.active_swarm_cell_mask == 0xFF
        {
            let is_def_clamp = (frame.ripa_clamp_rad.to_bits()
                == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
                & frame.has_unit_pspm_gains()
                & (num_heads > 0)
                & (num_heads <= MAX_ATTENTION_HEADS);
            match (query_tensor, key_tensor) {
                (Some(q), Some(k)) => {
                    if has_nonzero_words {
                        if is_def_clamp {
                            let rot_dim = frame.resolve_autonomous_rot_dim(head_dim, 0.0, 0.0);
                            rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
                                q.as_mut_slice(),
                                k.as_mut_slice(),
                                &swar_words,
                                num_heads,
                                head_dim,
                                rot_dim,
                            );
                        } else {
                            let subhead_cfg = frame.to_subhead_config_for_heads(num_heads);
                            dispatch_fused_ripa_and_pspm_words_core(
                                q,
                                &swar_words,
                                frame.ripa_clamp_rad,
                                &subhead_cfg,
                                &swar_words,
                                &swar_words,
                                &swar_words,
                                frame.ripa_clamp_rad,
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
                                frame.ripa_clamp_rad,
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
                            rotate_fused_self_ripa_and_pspm_default_words(
                                q.as_mut_slice(),
                                &swar_words,
                                num_heads,
                                head_dim,
                                rot_dim,
                                true,
                            );
                        } else {
                            let subhead_cfg = frame.to_subhead_config_for_heads(num_heads);
                            dispatch_fused_ripa_and_pspm_words_core(
                                q,
                                &swar_words,
                                frame.ripa_clamp_rad,
                                &subhead_cfg,
                                &swar_words,
                                &swar_words,
                                &swar_words,
                                frame.ripa_clamp_rad,
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
                            frame.ripa_clamp_rad,
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
                        frame.ripa_clamp_rad,
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
                        frame.ripa_clamp_rad,
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
    qdps_min_step_q7: u8,
    unresolved_damping_u7: u8,
) -> [u64; NUM_STIGMERGIC_GRANULES] {
    let qdps_words = if qdps_min_step_q7 <= 1 {
        *next_w
    } else if qdps_min_step_q7 > 127 {
        [0u64; NUM_STIGMERGIC_GRANULES]
    } else {
        swar_q7_qdps_filter_words_64(next_w, qdps_min_step_q7)
    };
    let clean_damping =
        branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
    if clean_damping >= 128 {
        qdps_words
    } else {
        let s_u64 = u64::from(clean_damping);
        let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        [
            swar_q7_scale_le128_with_bias_x8(qdps_words[0], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(qdps_words[1], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(qdps_words[2], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(qdps_words[3], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(qdps_words[4], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(qdps_words[5], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(qdps_words[6], s_u64, bias_restore),
            swar_q7_scale_le128_with_bias_x8(qdps_words[7], s_u64, bias_restore),
        ]
    }
}

/// Borrowed read-only 3-layer core context (`Liminal` + `Stigmergic` + `Ego`) passed into
/// per-layer Hot-Path RoPE steering and ActionHub orchestration.
#[derive(Clone, Copy)]
pub struct CoreLayerEvalContext<'a> {
    /// Unified 3-layer autonomous core stack (`128 B`).
    pub stack: &'a AutonomousCoreLayerStack,
    /// Temporal boundary velocity between consecutive Layer 0 epochs.
    pub delta: &'a LiminalBoundaryDelta,
    /// Latest epistemic horizon reading (`KnownCoherent`, `InternalConflict`, or `Liminal`).
    pub horizon: &'a EpistemicHorizonReading,
    /// Latest Venn overlap & serendipity telemetry report across the 8 SWAR granules.
    pub venn: &'a StigmergicVennSerendipityReport,
}

#[inline(always)]
fn synthesize_signal_with_optional_core_layers<S: ManifoldGatewaySignal + ?Sized>(
    signal: &S,
    core_ctx: Option<CoreLayerEvalContext<'_>>,
) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
    if let Some(ctx) = core_ctx {
        if ctx.stack.liminal_layer.adaptation_epochs > 0 || ctx.stack.ego_layer.is_active() {
            return signal.synthesize_gateway_with_core_layers_readonly(
                ctx.stack,
                ctx.delta,
                ctx.horizon,
                ctx.venn,
            );
        }
    }
    signal.synthesize_gateway_frame_with_words()
}

#[inline(always)]
pub(super) fn prepare_gateway_steering_state<T: PhaseSteering + ?Sized, S: ManifoldGatewaySignal + ?Sized>(
    engine: &T,
    signal: &S,
    core_ctx: Option<CoreLayerEvalContext<'_>>,
    trail_inout: &mut gcso_pointer_trail_t,
    query_tensor: &AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(
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
        || frame.qdps_min_step_rad < 0.0
        || !branchless_is_finite_f32(frame.qdps_min_step_rad)
        || frame.ripa_clamp_rad <= 0.0
        || !branchless_is_finite_f32(frame.ripa_clamp_rad)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
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
/// any type implementing [`PhaseSteering`] + [`SubHeadRouter`].
pub trait GatewayPhaseSteeringExt: PhaseSteering + SubHeadRouter {
    /// Synthesizes any [`ManifoldGatewaySignal`] into a 128-byte [`GatewayFrame`] and register-resident
    /// phase words, updates `trail_inout`, and executes $SO(2)$-fused RIPA + sub-head router sub-head steering.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if head dimensions, radian clamps, or tensor slices
    /// violate layout or `GCSO_RESTRICT` disjointness invariants.
    #[inline(always)]
    fn steer_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail_inout: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
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

    /// Synthesizes any [`ManifoldGatewaySignal`] into structural hyperparameters (`ripa_clamp_rad`,
    /// `qdps_min_step_rad`, `effective_inv_temp`, SWAR phase words) and steers `query_tensor`
    /// (and optional `key_tensor`) according to [`RopeSteeringTarget`] (`Interleaved` or `HalfSplit`,
    /// including partial `rotary_dim`) with zero heap allocation.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `geometry`, radian clamps, or tensor slices
    /// violate layout or `GCSO_RESTRICT` disjointness invariants.
    #[inline]
    fn steer_gateway_signal_for_geometry_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &self,
        signal: &S,
        geometry: &G,
        trail_inout: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
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
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `geometry`, radian clamps, or tensor slices
    /// violate layout or `GCSO_RESTRICT` disjointness invariants.
    #[inline]
    fn steer_gateway_signal_with_core_layers_for_geometry_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &self,
        signal: &S,
        geometry: &G,
        core_ctx: Option<CoreLayerEvalContext<'_>>,
        trail_inout: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
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
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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
                frame.qdps_min_step_q7,
                unresolved_damping_u7,
            );

            if swar_words_nonzero_64(&swar_words) {
                let mut q7_stack = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
                store_q7_words_64(&mut q7_stack.data, &swar_words);
                let p_aligned = AlignedSlice16::new(&q7_stack.data[..num_heads])?;
                PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe(
                    self,
                    geometry,
                    query_tensor,
                    &p_aligned,
                    frame.ripa_clamp_rad,
                )?;
                if let Some(k) = key_tensor {
                    PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe(
                        self,
                        geometry,
                        k,
                        &p_aligned,
                        frame.ripa_clamp_rad,
                    )?;
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

impl<T: PhaseSteering + SubHeadRouter + ?Sized> GatewayPhaseSteeringExt for T {}

/// Blanket-implemented Gateway-driven Semantic Residual Layer (Rank-1 residual) extension for any
/// [`Rank1ResidualAdapter`] engine.
///
/// Derives a scale-invariant residual injection gain autonomously from the hidden state's
/// L2 norm and [`GatewayFrame`] structural quantities (`effective_energy`, `ripa_clamp_rad`,
/// `fact_persona_coherence`, `repulsion_severity`), eliminating manual scalar tuning.
pub trait GatewayRank1SteeringExt: Rank1ResidualAdapter {

    /// Applies norm-adaptive fused Rank-1 residual injection from an already synthesized or
    /// circuit-modulated [`GatewayFrame`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths mismatch or `Rank1ResidualAdapter` rejects inputs.
    #[inline]
    fn apply_gateway_frame_rank1_hidden_fused(
        &self,
        frame: &GatewayFrame,
        y_out: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        u_vec: &AlignedSlice32<'_, f32>,
        v_vec: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<f32> {
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
        let coh_boost =
            branchless_clamp_f32(0.5 + 0.5 * frame.fact_persona_coherence, 0.25, 1.0);
        let structural_gain = branchless_flush_subnormal_f32(branchless_clamp_f32(
            frame.ripa_clamp_rad * energy_gate * coh_boost * 0.125,
            0.0,
            0.05,
        ));
        if structural_gain > 0.0 {
            self.eval_rank1_slices_fused(y_out, x_in, u_vec, v_vec, None, structural_gain)?;
        }
        Ok(structural_gain)
    }
}

impl<T: Rank1ResidualAdapter + ?Sized> GatewayRank1SteeringExt for T {}

/// Blanket-implemented extension trait equipping any [`ActionHub`] with single-call
/// [`ManifoldGatewaySignal`] ingestion, Online Toroidal $k$-Means / DP-Means macro clustering,
/// online basin consolidation, and `.gcso` crystallization.
pub trait ActionHubGatewayExt: ActionHub {

    /// Synthesizes `signal` with dynamic trait cognitive apertures and optional 3-layer core
    /// modulation (`core_ctx`), then steers `query_tensor` and optional `logits_opt` for `geometry`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`ActionHub`] or [`PhaseSteeringGeometryExt`].
    #[inline]
    fn ingest_gateway_and_steer_with_core_layers_for_geometry_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
    >(
        &mut self,
        signal: &S,
        geometry: &G,
        core_ctx: Option<CoreLayerEvalContext<'_>>,
        ledger_slot: &mut gcso_ledger_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        mut query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
        let geom = geometry.rope_geometry();
        let (frame, frm_w) = synthesize_signal_with_optional_core_layers(signal, core_ctx);
        let report = frame.to_causal_report();
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            frame.qdps_min_step_rad,
            frame.ripa_clamp_rad,
            query_tensor.as_deref(),
            None,
        )?;
        if geom.head_dim != head_dim || geom.num_heads != num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let (bypass_hit, effective_ptr) =
            self.resolve_or_pass_tagged_ptr(ledger_slot, gcso_tagged_ptr_t(frame.tagged_ptr));
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
                (frame.provenance_bits
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
                filter_and_damp_gateway_words(&next_w, frame.qdps_min_step_q7, clean_damping);
            if swar_words_nonzero_64(&swar_words) {
                let mut q7_stack = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
                store_q7_words_64(&mut q7_stack.data, &swar_words);
                let p_aligned = AlignedSlice16::new(&q7_stack.data[..num_heads])?;
                PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe(
                    self,
                    geometry,
                    q,
                    &p_aligned,
                    frame.ripa_clamp_rad,
                )?;
                metrics |= LEDGER_METRIC_PHASE_STEERED;
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

    /// Orchestrates scale-aware `signal` synthesis, 3-layer core modulation, and [`ActionHub`]
    /// decision routing for `geometry`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`ActionHub`].
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
        ledger_slot: &mut gcso_ledger_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        mut query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        srl_hidden: Option<(
            &mut AlignedSliceMut32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
        )>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport, StigmergicActionDecision)> {
        let geom = geometry.rope_geometry();
        let (mut frame, frm_w) = synthesize_signal_with_optional_core_layers(signal, core_ctx);
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
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            frame.qdps_min_step_rad,
            frame.ripa_clamp_rad,
            query_tensor.as_deref(),
            None,
        )?;
        if geom.head_dim != head_dim || geom.num_heads != num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // Evaporate stale pheromone and deposit new pheromone weighted by resonance energy and
        // active-head trajectory self-coherence (`prev_w` vs `frm_w`) so multi-layer traversal
        // dynamically transitions across Express, sub-head router, Rank-1 residual, and Crystal instead of saturating.
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
            query_tensor.is_some() || srl_hidden.is_none() || (trail_out.step_count == 0);
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
            self.resolve_or_pass_tagged_ptr(ledger_slot, gcso_tagged_ptr_t(frame.tagged_ptr));
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
        let (decision, _mode_flags, _target_slot, extra_flags) = thresholds
            .select_action_decision(
                next_rho,
                entropy_h_tilde,
                coh,
                is_repulsed,
                frame.dominant_slot,
            );
        trail_out.flags = (trail_out.flags
            & !(TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | extra_flags;

        let mut metrics = branchless_select_u8(bypass_hit, LEDGER_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(is_repulsed, LEDGER_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(
                (frame.provenance_bits
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
                let hub_damping_u7 = if matches!(decision, StigmergicActionDecision::RepulsionEscape)
                {
                    (clean_damping >> 1).max(1)
                } else {
                    clean_damping
                };
                let swar_words =
                    filter_and_damp_gateway_words(&next_w, frame.qdps_min_step_q7, hub_damping_u7);
                if swar_words_nonzero_64(&swar_words) {
                    let mut q7_stack = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
                    store_q7_words_64(&mut q7_stack.data, &swar_words);
                    let p_aligned = AlignedSlice16::new(&q7_stack.data[..num_heads])?;
                    PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe(
                        self,
                        geometry,
                        q,
                        &p_aligned,
                        frame.ripa_clamp_rad,
                    )?;
                    metrics |= LEDGER_METRIC_PHASE_STEERED;
                }
            }

            if matches!(
                decision,
                StigmergicActionDecision::FullOrchestration
                    | StigmergicActionDecision::CrystallizeMacroBasin
            ) {
                if let Some((y_mut, x_in, u_vec, v_vec)) = srl_hidden {
                    let _ = GatewayRank1SteeringExt::apply_gateway_frame_rank1_hidden_fused(
                        self, &frame, y_mut, x_in, u_vec, v_vec,
                    )?;
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

    /// Scale- and policy-aware layer orchestrator allowing per-layer traversal across
    /// `MacroCoalition -> TypeGateway -> CellTrail -> PointerBit`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub`] or [`PhaseSteeringGeometryExt`].
    #[inline]
    fn orchestrate_gateway_layer_with_policy_and_scale_safe<
        S: ManifoldGatewaySignal + ?Sized,
        G: RopeSteeringTarget + ?Sized,
        P: StigmergicThresholdPolicy + ?Sized,
        C: SystemConstraintSignal + ?Sized,
    >(
        &mut self,
        signal: &S,
        geometry: &G,
        policy: &P,
        system: &C,
        probe: &ManifoldProbeBaseline,
        scale: StigmergicScaleGranularity,
        entropy_h_tilde: f32,
        ledger_slot: &mut gcso_ledger_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        srl_hidden: Option<(
            &mut AlignedSliceMut32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
        )>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport, StigmergicActionDecision)> {
        self.orchestrate_gateway_layer_with_core_layers_safe(
            signal,
            geometry,
            None,
            policy,
            system,
            probe,
            scale,
            entropy_h_tilde,
            ledger_slot,
            trail_out,
            unresolved_damping_u7,
            query_tensor,
            srl_hidden,
            logits_opt,
        )
    }

    /// Core-layer-aware counterpart to [`Self::orchestrate_gateway_layer_with_policy_and_scale_safe`]
    /// connecting dynamic trait cognitive apertures and the 3-layer core stack (`Liminal` +
    /// `Stigmergic` + `Ego`) directly into per-layer ActionHub RoPE/Rank-1 orchestration.
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
        ledger_slot: &mut gcso_ledger_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        srl_hidden: Option<(
            &mut AlignedSliceMut32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
            &AlignedSlice32<'_, f32>,
        )>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport, StigmergicActionDecision)> {
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
            srl_hidden,
            logits_opt,
        )
    }

}

impl<T: ActionHub + ?Sized> ActionHubGatewayExt for T {}

impl CausalProvenanceSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0)
    }
}

impl CognitiveGroundingSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_profile
    }
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        Some(&self.phase_profile)
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *self.phase_words_ref()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.phase_gravity_mass
    }
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        self.mapped_section_offset48
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl ObjectiveFactSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 {
        self.ripa_clamp_rad
    }
}

impl SubjectivePersonaSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 {
        self.ripa_clamp_rad
    }
}

impl ManifoldGatewaySignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > DEFAULT_ENTROPY_EPS);
        if !is_active {
            let packed_tag =
                pack_action_slot_tag(self.dominant_slot, ACTION_PRESET_PASSIVE_SIDECAR);
            return (
                GatewayFrame {
                    tagged_ptr: gcso_tagged_ptr_t::pack(self.mapped_section_offset48, packed_tag, 0)
                        .raw(),
                    mapped_section_offset48: self.mapped_section_offset48,
                    target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot,
                    action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask,
                    dominant_cause_code: self.dominant_cause_code,
                    frame_flags: packed_tag as u16,
                    ..GatewayFrame::default()
                },
                [0u64; NUM_STIGMERGIC_GRANULES],
            );
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let tag_q7 = fast_mean_q7_tag_from_words_64(&w);
        let eff_action = self.action_mode_flags;
        let packed_tag = pack_action_slot_tag(self.dominant_slot, eff_action);
        let tagged_ptr =
            gcso_tagged_ptr_t::pack(self.mapped_section_offset48, packed_tag, tag_q7).raw();
        let prov_bits = self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let pspm_fact =
            branchless_select_u8(is_fact_slot, 36, branchless_select_u8(is_persona_slot, 16, 24));
        let pspm_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let pspm_explore =
            branchless_select_u8(is_fact_slot, 8, branchless_select_u8(is_persona_slot, 24, 16));
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (
            GatewayFrame {
                tagged_ptr,
                mapped_section_offset48: self.mapped_section_offset48,
                target_anchor_id: self.anchor_id,
                effective_energy: branchless_min_f32(
                    self.phase_gravity_mass,
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
                effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar,
                repulsion_severity: 0.0,
                provenance_bits: prov_bits,
                dominant_cause_code: self.dominant_cause_code,
                dominant_slot: self.dominant_slot,
                action_mode_flags: eff_action,
                active_swarm_cell_mask: self.active_swarm_mask,
                qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
                subhead_fact_heads: pspm_fact,
                subhead_logic_heads: pspm_logic,
                subhead_explore_heads: pspm_explore,
                subhead_fact_gain_q7: fact_gain_q7,
                subhead_logic_gain_q7: logic_gain_q7,
                subhead_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas: self.phase_profile,
            },
            w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > DEFAULT_ENTROPY_EPS);
        if !is_active {
            return (
                GatewayFrame {
                    mapped_section_offset48: self.mapped_section_offset48,
                    target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot,
                    action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask,
                    dominant_cause_code: self.dominant_cause_code,
                    ..GatewayFrame::default()
                },
                [0u64; NUM_STIGMERGIC_GRANULES],
            );
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let prov_bits = self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);
        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let pspm_fact =
            branchless_select_u8(is_fact_slot, 36, branchless_select_u8(is_persona_slot, 16, 24));
        let pspm_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let pspm_explore =
            branchless_select_u8(is_fact_slot, 8, branchless_select_u8(is_persona_slot, 24, 16));
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (
            GatewayFrame {
                tagged_ptr: 0,
                mapped_section_offset48: self.mapped_section_offset48,
                target_anchor_id: self.anchor_id,
                effective_energy: branchless_min_f32(
                    self.phase_gravity_mass,
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
                effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar,
                repulsion_severity: 0.0,
                provenance_bits: prov_bits,
                dominant_cause_code: self.dominant_cause_code,
                dominant_slot: self.dominant_slot,
                action_mode_flags: self.action_mode_flags,
                active_swarm_cell_mask: self.active_swarm_mask,
                qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
                subhead_fact_heads: pspm_fact,
                subhead_logic_heads: pspm_logic,
                subhead_explore_heads: pspm_explore,
                subhead_fact_gain_q7: fact_gain_q7,
                subhead_logic_gain_q7: logic_gain_q7,
                subhead_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: 0,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: 0,
                phase_deltas: [0i8; MAX_ATTENTION_HEADS],
            },
            w,
        )
    }

    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        (self.sample_count > 0) & (self.phase_gravity_mass > DEFAULT_ENTROPY_EPS)
    }

    #[inline]
    fn synthesize_gateway_slice_with_words(
        records: &[Self],
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let len = records.len();
        if len == 0 {
            return (GatewayFrame::default(), [0u64; NUM_STIGMERGIC_GRANULES]);
        }
        if len == 1 {
            return records[0].synthesize_gateway_frame_with_words();
        }

        let mut acc_w = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut acc_is_uniform = true;
        let mut tot_e = 0.0_f32;
        let mut best_e = 0.0_f32;
        let mut weighted_inv_temp = 0.0_f32;
        let mut weighted_coh = 0.0_f32;
        let mut dom_idx = 0usize;
        let mut dom_storage_addr = 0u64;
        let mut action_flags = ACTION_PRESET_PASSIVE_SIDECAR;
        let mut swarm_mask = 0xFFu8;
        let mut prov_bits = 0u32;
        let mut dom_cause = ROOT_CAUSE_NOMINAL;
        let mut has_active = false;

        let mut i = 0usize;
        while i < len {
            let rec = &records[i];
            if (rec.sample_count > 0) & (rec.phase_gravity_mass > DEFAULT_ENTROPY_EPS) {
                let rec_w = rec.phase_words_ref();
                let rec_is_uniform = if !has_active || acc_is_uniform {
                    is_word_array_uniform_64(rec_w)
                } else {
                    false
                };
                let f_e = branchless_min_f32(
                    branchless_flush_subnormal_f32(branchless_relu_f32(rec.phase_gravity_mass)),
                    MAX_ATTRACTOR_PULL_FORCE,
                );
                let is_mature = (rec.sample_count >= 2) & (rec.resultant_r_bar >= 0.45);
                let rec_prov = rec.provenance_bits
                    | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

                if !has_active {
                    acc_w = *rec_w;
                    acc_is_uniform = rec_is_uniform;
                    tot_e = f_e;
                    best_e = f_e;
                    weighted_inv_temp = f_e * rec.effective_inv_temp;
                    weighted_coh = f_e * rec.resultant_r_bar;
                    dom_idx = i;
                    dom_storage_addr = rec.mapped_section_offset48;
                    action_flags = rec.action_mode_flags;
                    swarm_mask = rec.active_swarm_mask;
                    prov_bits = rec_prov;
                    dom_cause = rec.dominant_cause_code;
                    has_active = true;
                } else if f_e > DEFAULT_ENTROPY_EPS {
                    let new_tot = tot_e + f_e;
                    if !swar_words_equal_64(&acc_w, rec_w) {
                        let safe_tot = branchless_select_f32(new_tot > DEFAULT_ENTROPY_EPS, new_tot, 1.0);
                        let w_ratio = branchless_clamp_f32(
                            f_e * fast_inv_pos_normal_f32(safe_tot),
                            0.0,
                            1.0,
                        );
                        let s_u8 = ((w_ratio * 128.0 + 0.5) as u8).min(128);
                        let s_u64 = s_u8 as u64;
                        let bias_restore =
                            (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let (next_acc, next_uni) = swar_slerp_words_with_uniform_flag_64(
                            &acc_w,
                            acc_is_uniform,
                            rec_w,
                            rec_is_uniform,
                            s_u64,
                            bias_restore,
                        );
                        acc_w = next_acc;
                        acc_is_uniform = next_uni;
                    }
                    weighted_inv_temp += f_e * rec.effective_inv_temp;
                    weighted_coh += f_e * rec.resultant_r_bar;
                    let take_dom = f_e > best_e;
                    best_e = branchless_select_f32(take_dom, f_e, best_e);
                    dom_idx = branchless_select_usize(take_dom, i, dom_idx);
                    dom_storage_addr = branchless_select_u64(
                        take_dom & (rec.mapped_section_offset48 != 0),
                        rec.mapped_section_offset48,
                        branchless_select_u64(
                            dom_storage_addr != 0,
                            dom_storage_addr,
                            rec.mapped_section_offset48,
                        ),
                    );
                    action_flags |= rec.action_mode_flags;
                    swarm_mask &= rec.active_swarm_mask;
                    prov_bits |= rec_prov;
                    dom_cause =
                        branchless_select_u8(
                            rec.dominant_cause_code > dom_cause,
                            rec.dominant_cause_code,
                            dom_cause,
                        );
                    tot_e = new_tot;
                }
            }
            i += 1;
        }

        let safe_tot_e = branchless_select_f32(tot_e > DEFAULT_ENTROPY_EPS, tot_e, 1.0);
        let inv_tot_e = fast_inv_pos_normal_f32(safe_tot_e);
        let eff_inv_temp = branchless_select_f32(
            tot_e > DEFAULT_ENTROPY_EPS,
            branchless_clamp_f32(weighted_inv_temp * inv_tot_e, 0.25, 4.0),
            records[dom_idx].effective_inv_temp,
        );
        let fp_coh = branchless_select_f32(
            tot_e > DEFAULT_ENTROPY_EPS,
            branchless_clamp_f32(weighted_coh * inv_tot_e, 0.0, 1.0),
            records[dom_idx].resultant_r_bar,
        );
        let dom = &records[dom_idx];
        let dom_slot = dom.dominant_slot;
        let is_fact_slot = dom_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = dom_slot == ATTRACTOR_SLOT_PERSONA;
        let pspm_fact =
            branchless_select_u8(is_fact_slot, 36, branchless_select_u8(is_persona_slot, 16, 24));
        let pspm_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let pspm_explore =
            branchless_select_u8(is_fact_slot, 8, branchless_select_u8(is_persona_slot, 24, 16));
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        let phase_deltas = words_to_q7_array_64(&acc_w);
        let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&acc_w, acc_is_uniform);
        let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
        let tagged_ptr = gcso_tagged_ptr_t::pack(dom_storage_addr, packed_tag, tag_q7).raw();

        (
            GatewayFrame {
                tagged_ptr,
                mapped_section_offset48: dom_storage_addr,
                target_anchor_id: dom.anchor_id,
                effective_energy: branchless_min_f32(tot_e, MAX_ATTRACTOR_PULL_FORCE),
                ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
                effective_inv_temp: eff_inv_temp,
                fact_persona_coherence: fp_coh,
                repulsion_severity: 0.0,
                provenance_bits: prov_bits,
                dominant_cause_code: dom_cause,
                dominant_slot: dom_slot,
                action_mode_flags: action_flags,
                active_swarm_cell_mask: swarm_mask,
                qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
                subhead_fact_heads: pspm_fact,
                subhead_logic_heads: pspm_logic,
                subhead_explore_heads: pspm_explore,
                subhead_fact_gain_q7: fact_gain_q7,
                subhead_logic_gain_q7: logic_gain_q7,
                subhead_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas,
            },
            acc_w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

/// Blanket-implemented bridge coupling [`StigmergicReconstitutionOps`] with [`StigmergicScaleLayer`]
/// so Micro-Swarm quorum thresholds, tolerances, and active cell masks adapt automatically
/// to the active granularity tier and layer depth `(layer_index, total_layers)`.
pub trait StigmergicLayeredSwarmExt: StigmergicReconstitutionOps {
    /// Resolves `target_words` inside the 8-cell SWAR Micro-Swarm using parameters derived
    /// structurally from `(granularity, layer_index, total_layers)` and optional Bloom guards.
    #[inline(always)]
    fn resolve_swarm_with_scale_layer(
        &mut self,
        action_ptr: gcso_tagged_ptr_t,
        target_words: &[u64; NUM_STIGMERGIC_GRANULES],
        granularity: StigmergicScaleGranularity,
        layer_index: usize,
        total_layers: usize,
        active_cell_mask: u8,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
    ) -> (StigmergicReconstitutionReport, [u64; NUM_STIGMERGIC_GRANULES]) {
        let gain_q7 = granularity.layer_peel_gain_q7(layer_index, total_layers);
        let tol_q7 = granularity.swarm_tolerance_q7(DEFAULT_STIGMERGIC_TOLERANCE_Q7);
        let quorum = granularity.swarm_quorum_requirement(DEFAULT_STIGMERGIC_QUORUM_THRESHOLD);
        let mask = if active_cell_mask == 0 { 0xFF } else { active_cell_mask };
        self.try_resolve_action_in_swarm_from_words(
            action_ptr,
            target_words,
            mask,
            gain_q7,
            2,
            tol_q7,
            quorum,
            0.40,
            0.15,
            bloom_filter,
            3,
            96,
        )
    }
}

impl<T: StigmergicReconstitutionOps + ?Sized> StigmergicLayeredSwarmExt for T {}

/// Cross-scale [`ActionHub`] orchestrator uniting [`StigmergicScaleLayer`],
/// [`CognitiveTrajectoryLedger`], and [`StigmergicLayeredSwarmExt`] in zero-allocation mode.
pub trait ActionHubScaleOrchestrator: ActionHub + ActionHubGatewayExt {
    /// Projects `layer` across `input_words`, resolves the active pointer trail in the Micro-Swarm
    /// at the activated granularity tier, and records the emitted [`CognitiveStepOp`] into
    /// `ledger_slot.telemetry_mini_ledger`.
    ///
    /// # Errors
    /// Propagates any error from [`crate::traits::hot_path::PointerActionHub::step_pointer`] or
    /// [`crate::traits::telemetry::FastPathLedger::telemetry_push`].
    #[inline]
    fn orchestrate_scale_layer(
        &mut self,
        layer: &(impl StigmergicScaleLayer + ?Sized),
        input_ptr: gcso_tagged_ptr_t,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger_slot: &mut gcso_ledger_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        layer_index: usize,
        total_layers: usize,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
    ) -> GcsoResult<(
        ScaleLayerProjection,
        CognitiveTrajectoryLedger,
        StigmergicReconstitutionReport,
    )> {
        let mut ledger = CognitiveTrajectoryLedger::from_ledger_slot(ledger_slot);
        let proj = layer.project_scale_layer(input_words, &ledger, layer_index, total_layers);
        let eff_ptr = gcso_tagged_ptr_t(if input_ptr.is_null() {
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
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `layers` is empty or `depth_n == 0`, or
    /// propagates pointer/telemetry errors.
    #[inline]
    fn orchestrate_adaptive_depth_n<L: StigmergicScaleLayer>(
        &mut self,
        layers: &[L],
        depth_n: usize,
        input_ptr: gcso_tagged_ptr_t,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger_slot: &mut gcso_ledger_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
    ) -> GcsoResult<(
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
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `layers` is empty or `depth_n == 0`, or
    /// propagates pointer/telemetry errors.
    #[inline]
    fn orchestrate_adaptive_depth_n_with_policy<
        L: StigmergicScaleLayer,
        P: CognitiveCircuitThresholdPolicy + ?Sized,
    >(
        &mut self,
        layers: &[L],
        depth_n: usize,
        input_ptr: gcso_tagged_ptr_t,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger_slot: &mut gcso_ledger_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        circuit_policy: &P,
    ) -> GcsoResult<(
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
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let fuse_floor = circuit_policy.early_exit_fuse_floor_q7();
        let repulse_ceil = circuit_policy.early_exit_repulse_ceiling_q7();
        let res_max = circuit_policy.early_exit_residual_max_q7();
        let refine_floor = circuit_policy.recurrent_refine_intensity_floor_q7();

        let mut last_proj = self
            .orchestrate_scale_layer(
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

impl<T: ActionHub + ActionHubGatewayExt + ?Sized> ActionHubScaleOrchestrator for T {}

