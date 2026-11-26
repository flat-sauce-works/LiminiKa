// name: src/adapter/src/model_adapter/steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Per-layer RoPE phase baking, gateway/horizon steering, Rank-1 residual injection, and entropy evaluation on [`ModelSteeringAdapter`].

use super::super::*;
use super::{AlignedF32Buffer32, AlignedQ7Buffer16, ModelSteeringAdapter};

impl ModelSteeringAdapter {
    /// Computes per-head Q7 phase deltas from negative (`source`) and positive (`target`)
    /// pre-RoPE query centroids via [`PhaseSteeringGeometryExt::compute_procrustes_phase_delta_for_geometry`].
    ///
    /// # Errors
    ///
    /// Returns an error if either slice length does not match `num_q_heads * rotary_dim`
    /// or if `liminika-core` returns an argument error.
    pub fn bake_layer_procrustes_q7(
        &self,
        neg_q_rot: &[f32],
        pos_q_rot: &[f32],
    ) -> Result<Vec<lmnk_q7_t>, String> {
        let expected = self.geometry.q_rot_len();
        if neg_q_rot.len() != expected || pos_q_rot.len() != expected {
            return Err(format!(
                "expected q_rot length {}, got neg={} pos={}",
                expected,
                neg_q_rot.len(),
                pos_q_rot.len()
            ));
        }

        let mut neg_buf = AlignedF32Buffer32::new_zeroed(expected);
        let mut pos_buf = AlignedF32Buffer32::new_zeroed(expected);
        neg_buf.as_mut_slice().copy_from_slice(neg_q_rot);
        pos_buf.as_mut_slice().copy_from_slice(pos_q_rot);

        let src_aligned = AlignedSlice32::new(neg_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 src error: {e}"))?;
        let dst_aligned = AlignedSlice32::new(pos_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 dst error: {e}"))?;
        let mut q7_buf = AlignedQ7Buffer16::new_zeroed(self.geometry.num_q_heads);
        {
            let mut q7_mut = AlignedSliceMut16::new(q7_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut16 error: {e}"))?;
            self.ctx
                .phase_steering()
                .compute_procrustes_phase_delta_for_geometry(
                    &self.geometry,
                    &src_aligned,
                    &dst_aligned,
                    &mut q7_mut,
                )
                .map_err(|e| format!("compute_procrustes_phase_delta_for_geometry error: {e}"))?;
        }

        Ok(q7_buf.as_slice().to_vec())
    }

    /// Applies soft-bounded (`PhaseClamp`) phase steering to a single token's rotated query slice
    /// (`num_q_heads * rotary_dim`) using [`PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe`].
    ///
    /// Layers disabled in `geometry.layer_rope_mask` (NoPE layers) are skipped in O(1).
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `liminika-core` rejects the inputs.
    pub fn steer_layer_q_rot(
        &self,
        layer_idx: usize,
        q_rot_inout: &mut [f32],
        q7_deltas: &[lmnk_q7_t],
        max_rad: f32,
    ) -> Result<(), String> {
        if max_rad <= 0.0 || !self.geometry.is_rope_layer(layer_idx) {
            return Ok(());
        }
        let expected = self.geometry.q_rot_len();
        if q_rot_inout.len() != expected {
            return Err(format!(
                "expected q_rot_inout length {}, got {}",
                expected,
                q_rot_inout.len()
            ));
        }
        if q7_deltas.len() != self.geometry.num_q_heads {
            return Err(format!(
                "expected {} q7 deltas, got {}",
                self.geometry.num_q_heads,
                q7_deltas.len()
            ));
        }

        let mut work_buf = AlignedF32Buffer32::new_zeroed(expected);
        work_buf.as_mut_slice().copy_from_slice(q_rot_inout);

        let q7_buf = AlignedQ7Buffer16::from_slice(q7_deltas);
        let q7_aligned = AlignedSlice16::new(q7_buf.as_slice())
            .map_err(|e| format!("AlignedSlice16 error: {e}"))?;
        let Some(active_q7) =
            liminika_core::liminal::Scoped::from_boundary(PhaseDeltaSlice(q7_aligned))
                .cut_by_lacuna(max_rad.is_finite() && max_rad > 0.0, max_rad, 0.0)
                .into_active()
        else {
            return Ok(());
        };
        {
            let mut tensor_mut = AlignedSliceMut32::new(work_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut32 error: {e}"))?;
            self.ctx
                .phase_steering()
                .apply_phase_steering_for_geometry_safe_active(
                    &self.geometry,
                    &mut tensor_mut,
                    active_q7,
                    max_rad,
                )
                .map_err(|e| format!("apply_phase_steering_for_geometry_safe_active error: {e}"))?;
        }

        q_rot_inout.copy_from_slice(work_buf.as_slice());

        Ok(())
    }

    fn marshal_q_rot_and_logits<R, F>(
        &mut self,
        layer_idx: usize,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
        f: F,
    ) -> Result<Option<R>, String>
    where
        F: FnOnce(
            &mut Self,
            &mut AlignedSliceMut32<'_, f32>,
            Option<&mut AlignedSliceMut32<'_, f32>>,
        ) -> Result<R, String>,
    {
        if !self.geometry.is_rope_layer(layer_idx) {
            return Ok(None);
        }
        let expected = self.geometry.q_rot_len();
        if q_rot_inout.len() != expected {
            return Err(format!(
                "expected q_rot_inout length {}, got {}",
                expected,
                q_rot_inout.len()
            ));
        }

        let mut work_buf = AlignedF32Buffer32::new_zeroed(expected);
        work_buf.as_mut_slice().copy_from_slice(q_rot_inout);

        let mut logits_buf = logits_opt.as_ref().map(|l| {
            let mut b = AlignedF32Buffer32::new_zeroed(l.len());
            b.as_mut_slice().copy_from_slice(l);
            b
        });

        let out = {
            let mut tensor_mut = AlignedSliceMut32::new(work_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut32 error: {e}"))?;
            let mut logits_mut_opt = match logits_buf.as_mut() {
                Some(b) => Some(
                    AlignedSliceMut32::new(b.as_mut_slice())
                        .map_err(|e| format!("AlignedSliceMut32 logits error: {e}"))?,
                ),
                None => None,
            };
            f(self, &mut tensor_mut, logits_mut_opt.as_mut())?
        };

        q_rot_inout.copy_from_slice(work_buf.as_slice());
        if let (Some(dst), Some(src)) = (logits_opt, logits_buf) {
            dst.copy_from_slice(src.as_slice());
        }

        Ok(Some(out))
    }

    /// Steers a single token's rotated query slice (`num_q_heads * rotary_dim`) using a
    /// [`ManifoldGatewaySignal`] composite so structural hyperparameters (`phase_clamp_rad`,
    /// `phase_step_min_rad`, `effective_inv_temp`) are derived autonomously from Gateway physics
    /// rather than manual scalar tuning.
    ///
    /// Layers disabled in `geometry.layer_rope_mask` (NoPE layers) are skipped in O(1),
    /// returning `Ok(None)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches or `liminika-core` rejects the inputs.
    pub fn steer_layer_with_gateway<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        layer_idx: usize,
        signal: &S,
        trail_inout: &mut lmnk_pointer_trail_t,
        unresolved_damping_u7: u8,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
    ) -> Result<Option<(GatewayFrame, CausalProvenanceReport)>, String> {
        self.marshal_q_rot_and_logits(layer_idx, q_rot_inout, logits_opt, |this, q, logits| {
            let core_ctx = this.ctx.build_core_layer_eval_context();
            this.ctx
                .phase_steering()
                .steer_gateway_signal_with_core_layers_for_geometry_safe(
                    signal,
                    &this.geometry,
                    core_ctx,
                    trail_inout,
                    unresolved_damping_u7,
                    q,
                    None,
                    logits,
                )
                .map_err(|e| {
                    format!("steer_gateway_signal_with_core_layers_for_geometry_safe error: {e}")
                })
        })
    }

    /// Routes [`ManifoldGatewaySignal`] through the embedded [`StigmergicSubstrate`] via
    /// [`SubstrateGatewayExt::ingest_gateway_and_steer_with_core_layers_for_geometry_safe`],
    /// allowing the substrate's stigmergic trail density, FastPathLedger fast-path state, and live
    /// `2 Primitives x 2 Drivers` stack (`Liminal`/`Stigmergy` x `System`/`Ego`) to autonomously modulate steering.
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches or `liminika-core` rejects the inputs.
    pub fn steer_layer_with_substrate<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        layer_idx: usize,
        signal: &S,
        trail_inout: &mut lmnk_pointer_trail_t,
        unresolved_damping_u7: u8,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
    ) -> Result<Option<(GatewayFrame, CausalProvenanceReport)>, String> {
        self.marshal_q_rot_and_logits(layer_idx, q_rot_inout, logits_opt, |this, q, logits| {
            let (core_ctx, _probe, substrate) = this.ctx.split_for_substrate_gateway();
            let mut ledger_slot = substrate.ledger_slot;
            let res = substrate
                .ingest_gateway_and_steer_with_core_layers_for_geometry_safe(
                    signal,
                    &this.geometry,
                    core_ctx,
                    &mut ledger_slot,
                    trail_inout,
                    unresolved_damping_u7,
                    Some(q),
                    logits,
                )
                .map_err(|e| {
                    format!(
                        "ingest_gateway_and_steer_with_core_layers_for_geometry_safe error: {e}"
                    )
                });
            substrate.ledger_slot = ledger_slot;
            res
        })
    }

    /// Orchestrates a layer's RoPE steering via [`StigmergicSubstrate`] using stigmergic
    /// evaporation + resonance-weighted reinforcement, the live 3-layer core stack, and the
    /// built-in dormant-until-destabilized [`CoherenceHorizonGuardSlot`].
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches or `liminika-core` rejects the inputs.
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_layer_rope_with_substrate<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        layer_idx: usize,
        signal: &S,
        trail_inout: &mut lmnk_pointer_trail_t,
        entropy_h_tilde_opt: Option<f32>,
        unresolved_damping_u7: u8,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
    ) -> Result<
        Option<(
            GatewayFrame,
            CausalProvenanceReport,
            StigmergicActionDecision,
        )>,
        String,
    > {
        let mut guard_slot = self.coherence_guard_slot;
        let mut no_escalator = ();
        let res = self.orchestrate_layer_rope_with_horizon_slots(
            layer_idx,
            signal,
            &mut guard_slot,
            &mut no_escalator,
            trail_inout,
            entropy_h_tilde_opt,
            unresolved_damping_u7,
            q_rot_inout,
            logits_opt,
        );
        self.coherence_guard_slot = guard_slot;
        res
    }

    /// Extended layer-wise orchestration pass pairing [`StigmergicSubstrate`] + 3-layer core stack
    /// piercing with caller-supplied [`LatentHorizonSlot`] boundaries (dynamic slide-in/out with
    /// dwell hysteresis) and an optional cold-path [`HorizonEscalationProvider`].
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches, `escalation_provider` fails, or
    /// `liminika-core` rejects the inputs.
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_layer_rope_with_horizon_slots<
        S: ManifoldGatewaySignal + ?Sized,
        H: LatentHorizonSlot + ?Sized,
        E: HorizonEscalationProvider + ?Sized,
    >(
        &mut self,
        layer_idx: usize,
        signal: &S,
        horizon_slots: &mut H,
        escalation_provider: &mut E,
        trail_inout: &mut lmnk_pointer_trail_t,
        entropy_h_tilde_opt: Option<f32>,
        unresolved_damping_u7: u8,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
    ) -> Result<
        Option<(
            GatewayFrame,
            CausalProvenanceReport,
            StigmergicActionDecision,
        )>,
        String,
    > {
        self.marshal_q_rot_and_logits(layer_idx, q_rot_inout, logits_opt, |this, q, logits| {
            let entropy_h_tilde =
                entropy_h_tilde_opt.unwrap_or(this.ctx.substrate().last_entropy_h_tilde);
            let scale = StigmergicScaleGranularity::from_layer_ratio(
                layer_idx,
                this.geometry.num_layers.max(1),
            );
            let pre_trigger = HorizonTriggerInput {
                entropy_gate_mode: this.ctx.entropy_controller().state().active_branch_mode,
                moving_z_entropy: entropy_h_tilde,
                repulsion_severity: this.ctx.last_horizon().conflict_severity,
                causal_root_cause: ROOT_CAUSE_NOMINAL,
                epistemic_regime: this.ctx.last_horizon().regime_code,
                stationary_shortcut_ready: this.ctx.last_boundary_delta().stationary_shortcut_ready
                    != 0,
                ego_active_triggers: this.ctx.core_layers().ego.last_intervened_mask,
                unstable_dwell_steps: this.ctx.horizon_dwell().unstable_dwell_steps,
                quiescent_dwell_steps: this.ctx.horizon_dwell().quiescent_dwell_steps,
            };
            let num_layers_denom = this.geometry.num_layers.saturating_sub(1).max(1);
            let layer_scale_q7 = ((layer_idx * 127) / num_layers_denom).min(127) as u8;
            let conflict_active = pre_trigger.repulsion_severity > 0.25;
            let repulsion_lane_already_active = (this.ctx.substrate().slot_repulsion_mask
                & this.ctx.substrate().slot_enabled_mask)
                != 0;
            let suppress_duplicate_repulsion_escalation =
                conflict_active && repulsion_lane_already_active;
            let esc_req = HorizonEscalationRequest {
                trigger: pre_trigger,
                placement: WorkbenchLanePlacement::AutoFree,
                lane_flags: if conflict_active {
                    WORKBENCH_LANE_FLAG_REPULSION
                } else {
                    0
                },
                layer_scale_q7,
                desired_aperture_mask: this.ctx.core_layers().liminal.active_aperture_mask,
                current_frame_energy: this.ctx.last_horizon().total_memory_support,
            };
            let lane_active = this.ctx.escalated_workbench_lane().is_some_and(|l| {
                l < 8 && (this.ctx.substrate().slot_enabled_mask & (1u8 << l)) != 0
            });
            let should_check_escalation = !suppress_duplicate_repulsion_escalation
                && (this.escalation_boundary_dirty
                    || pre_trigger.epistemic_regime != this.last_escalated_regime
                    || (lane_active && pre_trigger.can_retract())
                    || (!lane_active && pre_trigger.is_destabilized()));
            let escalated = if should_check_escalation {
                let mut esc_lane = this.ctx.escalated_workbench_lane();
                let sw = this
                    .ctx
                    .substrate_mut()
                    .apply_horizon_escalation(&esc_req, &mut esc_lane, escalation_provider)
                    .map_err(|e| format!("apply_horizon_escalation error: {e}"))?;
                *this.ctx.escalated_workbench_lane_mut() = esc_lane;
                this.escalation_boundary_dirty = false;
                this.last_escalated_regime = pre_trigger.epistemic_regime;
                sw
            } else {
                false
            };

            let (mut frame, report, decision) = {
                let (core_ctx, probe_baseline, substrate) = this.ctx.split_for_substrate_gateway();
                let mut ledger_slot = substrate.ledger_slot;
                let res = substrate
                    .orchestrate_gateway_layer_with_core_layers_safe(
                        signal,
                        &this.geometry,
                        core_ctx,
                        &this.threshold_policy,
                        &this.system_constraint,
                        probe_baseline,
                        scale,
                        entropy_h_tilde,
                        &mut ledger_slot,
                        trail_inout,
                        unresolved_damping_u7,
                        Some(q),
                        None,
                        logits,
                    )
                    .map_err(|e| {
                        format!("orchestrate_gateway_layer_with_core_layers_safe error: {e}")
                    });
                substrate.ledger_slot = ledger_slot;
                res?
            };

            let mut trigger = HorizonTriggerInput {
                entropy_gate_mode: this.ctx.entropy_controller().state().active_branch_mode,
                moving_z_entropy: entropy_h_tilde,
                repulsion_severity: frame.repulsion_severity,
                causal_root_cause: frame.attribution_tag,
                epistemic_regime: this.ctx.last_horizon().regime_code,
                stationary_shortcut_ready: this.ctx.last_boundary_delta().stationary_shortcut_ready
                    != 0,
                ego_active_triggers: this.ctx.core_layers().ego.last_intervened_mask,
                unstable_dwell_steps: 0,
                quiescent_dwell_steps: 0,
            };
            let (u_dwell, q_dwell) = this
                .ctx
                .horizon_dwell_mut()
                .step_branchless(trigger.is_destabilized(), trigger.stationary_shortcut_ready);
            trigger.unstable_dwell_steps = u_dwell;
            trigger.quiescent_dwell_steps = q_dwell;
            let (slid_in, slid_out) = {
                let mut active_mask = this.ctx.active_horizon_slots_mask();
                let mut stack = *this.ctx.core_layers();
                let res = apply_latent_horizon_slots(
                    horizon_slots,
                    &mut active_mask,
                    &trigger,
                    &mut frame,
                    &mut stack,
                );
                *this.ctx.active_horizon_slots_mask_mut() = active_mask;
                *this.ctx.core_layers_mut() = stack;
                res
            };

            let qos_throttled = frame.attribution_tag == ROOT_CAUSE_SYSTEM_QOS_THROTTLE;
            let last_horizon = *this.ctx.last_horizon();
            let last_delta = *this.ctx.last_boundary_delta();
            let last_overlap = *this.ctx.last_overlap_report();
            let ego_snapshot = this.ctx.core_layers().ego;
            let probe_baseline = *this.ctx.probe_baseline();
            {
                let bank = this.ctx.telemetry_bank_mut();
                bank.evaluate_policy(trigger.is_destabilized(), q_dwell, qos_throttled);
                bank.record_step(
                    decision,
                    &last_horizon,
                    &last_delta,
                    &last_overlap,
                    &ego_snapshot,
                    trigger.entropy_gate_mode,
                    slid_in,
                    slid_out,
                    escalated,
                    qos_throttled,
                );
                let _ = bank.update_assimilation_ema(
                    entropy_h_tilde,
                    frame.fact_persona_coherence,
                    frame.repulsion_severity,
                    &probe_baseline,
                    0.15_f32,
                );
            }
            let exterior_granule_mask = BitCut8::split(
                last_overlap.lacuna_exterior_mask | last_overlap.discovery_excursion_mask,
                last_overlap.overlap_core_mask,
            )
            .outside;
            let sb_event = this.ctx.step_sandbox_hypothesis_lifecycle(
                decision,
                this.configured_ego.is_active(),
                exterior_granule_mask,
                liminika_core::stigmergy::trail_phase_words_ref(trail_inout),
                frame.effective_energy,
                3,
                3,
            );
            match sb_event {
                SandboxLifecycleEvent::Settled(_) | SandboxLifecycleEvent::Released(_) => {
                    this.sandbox_settlement_streak = 0;
                }
                SandboxLifecycleEvent::Idle | SandboxLifecycleEvent::Nucleated(_) => {}
            }
            this.ctx.substrate_mut().last_decision = decision as u8;
            Ok((frame, report, decision))
        })
    }

    /// Orchestrates a layer's RoPE steering while pairing [`CoherenceHorizonGuardSlot`] and the
    /// auxiliary [`DefaultResidualSpilloverSlot`] experience cache (with an optional external
    /// [`HorizonEscalationProvider`]), syncing hits/stores/evictions to [`CoreTelemetryBank`].
    ///
    /// # Errors
    ///
    /// Returns an error if `q_rot_inout` length mismatches, `escalation_provider` fails, or
    /// `liminika-core` rejects the inputs.
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_layer_rope_with_spillover<
        S: ManifoldGatewaySignal + ?Sized,
        E: HorizonEscalationProvider + ?Sized,
    >(
        &mut self,
        layer_idx: usize,
        signal: &S,
        escalation_provider: &mut E,
        trail_inout: &mut lmnk_pointer_trail_t,
        entropy_h_tilde_opt: Option<f32>,
        unresolved_damping_u7: u8,
        q_rot_inout: &mut [f32],
        logits_opt: Option<&mut [f32]>,
    ) -> Result<
        Option<(
            GatewayFrame,
            CausalProvenanceReport,
            StigmergicActionDecision,
        )>,
        String,
    > {
        let mut guard_slot = self.coherence_guard_slot;
        let mut spillover_box = std::mem::take(&mut self.residual_spillover);
        let res = {
            let mut combined = (&mut guard_slot, &mut *spillover_box);
            self.orchestrate_layer_rope_with_horizon_slots(
                layer_idx,
                signal,
                &mut combined,
                escalation_provider,
                trail_inout,
                entropy_h_tilde_opt,
                unresolved_damping_u7,
                q_rot_inout,
                logits_opt,
            )
        };
        self.coherence_guard_slot = guard_slot;
        self.ctx.telemetry_bank_mut().sync_experience_metrics(
            spillover_box.cache_hits,
            spillover_box.cache_stores,
            spillover_box.cache_evictions,
        );
        self.residual_spillover = spillover_box;
        res
    }

    /// Executes norm-adaptive Rank-1 residual injection only when [`StigmergicSubstrate`]'s latest
    /// [`StigmergicActionDecision`] authorizes full orchestration (`FullOrchestration` or
    /// `CrystallizeMacroBasin`), persisting Core Layer 0/1/2 Ego interventions onto `self.ctx`.
    /// Returns `Ok(None)` when bypassed by the substrate.
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `liminika-core` rejects the inputs.
    pub fn orchestrate_layer_rank1_with_substrate<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<Option<(GatewayFrame, f32)>, String> {
        let decision = StigmergicActionDecision::from_u8(self.ctx.substrate().last_decision);
        if matches!(
            decision,
            StigmergicActionDecision::FullOrchestration
                | StigmergicActionDecision::CrystallizeMacroBasin
        ) {
            let res = self.apply_rank1_with_gateway_mut(signal, hidden_inout, u_vec, v_vec)?;
            Ok(Some(res))
        } else {
            Ok(None)
        }
    }

    /// Evaluates a slice of fragmented signals (`fragments`) as a subspace micro-ensemble over
    /// the live 3-layer [`CoreStateStack`], updating Ego intervention state in place.
    pub fn synthesize_fragment_ensemble<S: ManifoldGatewaySignal>(
        &mut self,
        fragments: &[S],
    ) -> (
        GatewayFrame,
        [u64; NUM_STIGMERGIC_GRANULES],
        StigmergicFragmentEnsembleReport,
    ) {
        let boundary_delta = *self.ctx.last_boundary_delta();
        let last_horizon = *self.ctx.last_horizon();
        let last_overlap = *self.ctx.last_overlap_report();
        S::synthesize_fragment_ensemble_with_core_layers(
            fragments,
            self.ctx.core_layers_mut(),
            &boundary_delta,
            &last_horizon,
            &last_overlap,
        )
    }

    fn modulate_rank1_circuit_frame(
        circuit: Option<CognitiveTrajectoryLedger>,
        frame: &mut GatewayFrame,
        frm_w: &[u64; NUM_STIGMERGIC_GRANULES],
    ) {
        if let Some(c) = circuit {
            c.modulate_gateway_frame_at_scale(
                frame,
                frm_w,
                frm_w,
                StigmergicScaleGranularity::MacroCoalition,
            );
        }
    }

    fn apply_rank1_frame_internal(
        &self,
        frame: &GatewayFrame,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<f32, String> {
        let d = hidden_inout.len();
        if d == 0 || u_vec.len() != d || v_vec.len() != d {
            return Err(format!(
                "Rank-1 residual hidden/u/v length mismatch: hidden={}, u={}, v={}",
                d,
                u_vec.len(),
                v_vec.len()
            ));
        }

        let mut y_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut x_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut u_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut v_buf = AlignedF32Buffer32::new_zeroed(d);
        y_buf.as_mut_slice().copy_from_slice(hidden_inout);
        x_buf.as_mut_slice().copy_from_slice(hidden_inout);
        u_buf.as_mut_slice().copy_from_slice(u_vec);
        v_buf.as_mut_slice().copy_from_slice(v_vec);

        let gain = {
            let mut y_mut = AlignedSliceMut32::new(y_buf.as_mut_slice())
                .map_err(|e| format!("AlignedSliceMut32 y error: {e}"))?;
            let x_aligned = AlignedSlice32::new(x_buf.as_slice())
                .map_err(|e| format!("AlignedSlice32 x error: {e}"))?;
            let u_aligned = AlignedSlice32::new(u_buf.as_slice())
                .map_err(|e| format!("AlignedSlice32 u error: {e}"))?;
            let v_aligned = AlignedSlice32::new(v_buf.as_slice())
                .map_err(|e| format!("AlignedSlice32 v error: {e}"))?;
            self.residual_adapter
                .apply_gateway_frame_rank1_hidden_fused(
                    frame, &mut y_mut, &x_aligned, &u_aligned, &v_aligned,
                )
                .map_err(|e| format!("apply_gateway_frame_rank1_hidden_fused error: {e}"))?
        };

        hidden_inout.copy_from_slice(y_buf.as_slice());
        Ok(gain)
    }

    /// Applies norm-adaptive Rank-1 residual injection driven purely by a [`ManifoldGatewaySignal`]
    /// via [`ResidualAdapterEngine::apply_gateway_frame_rank1_hidden_fused`] without manual
    /// scale tuning (shared `&self` snapshot path for read-only forward passes).
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `liminika-core` rejects the inputs.
    pub fn apply_rank1_with_gateway<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<(GatewayFrame, f32), String> {
        let core_layers = self.ctx.core_layers();
        let last_horizon = self.ctx.last_horizon();
        let last_overlap = self.ctx.last_overlap_report();
        let frame = if core_layers.ego.is_active() {
            let mut sys_baseline = signal.synthesize_gateway_with_aperture_from_stack(core_layers);
            let mut frame = sys_baseline;
            core_layers.modulate_gateway_frame_branchless(
                &mut sys_baseline,
                last_horizon,
                last_overlap,
            );
            core_layers.modulate_gateway_frame_with_ego_readonly(
                self.ctx.last_boundary_delta(),
                last_horizon,
                last_overlap,
                &mut frame,
            );
            let _ = self
                .ctx
                .ego_differential_net()
                .evaluate_pframe_chain_branchless(
                    &core_layers.ego,
                    self.ctx.system_spine(),
                    self.ctx.last_lacuna_partition(),
                    &core_layers.stigmergy,
                    self.ctx.distortion_tension(),
                    &mut frame,
                );
            let cand_w = *frame.phase_words_ref();
            Self::modulate_rank1_circuit_frame(self.active_circuit(), &mut frame, &cand_w);
            self.ctx
                .system_spine()
                .enforce_pinned_isolation_branchless(&sys_baseline, &mut frame);
            frame
        } else if core_layers.liminal.adaptation_epochs > 0 {
            let mut frame = signal.synthesize_gateway_with_aperture_from_stack(core_layers);
            core_layers.modulate_gateway_frame_branchless(&mut frame, last_horizon, last_overlap);
            let cand_w = *frame.phase_words_ref();
            Self::modulate_rank1_circuit_frame(self.active_circuit(), &mut frame, &cand_w);
            frame
        } else {
            let (mut raw_frame, frm_w) = signal.synthesize_gateway_frame_with_words();
            Self::modulate_rank1_circuit_frame(self.active_circuit(), &mut raw_frame, &frm_w);
            raw_frame
        };
        let gain = self.apply_rank1_frame_internal(&frame, hidden_inout, u_vec, v_vec)?;
        Ok((frame, gain))
    }

    /// Mutable counterpart to [`Self::apply_rank1_with_gateway`] that evaluates dynamic cognitive
    /// apertures and persists Core Layer 0/1/2 Ego interventions onto `self.ctx`.
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `liminika-core` rejects the inputs.
    pub fn apply_rank1_with_gateway_mut<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
    ) -> Result<(GatewayFrame, f32), String> {
        let circuit = self.active_circuit();
        let ego_active = self.ctx.core_layers().ego.is_active();
        let adapted = self.ctx.core_layers().liminal.adaptation_epochs > 0;
        let last_boundary_delta = *self.ctx.last_boundary_delta();
        let last_horizon = *self.ctx.last_horizon();
        let last_overlap = *self.ctx.last_overlap_report();
        let spine = *self.ctx.system_spine();
        let frame = if ego_active {
            let mut sys_baseline =
                signal.synthesize_gateway_with_aperture_from_stack(self.ctx.core_layers());
            self.ctx.core_layers().modulate_gateway_frame_branchless(
                &mut sys_baseline,
                &last_horizon,
                &last_overlap,
            );
            let (mut frame, _ego_w) = signal.synthesize_gateway_with_core_layers(
                self.ctx.core_layers_mut(),
                &last_boundary_delta,
                &last_horizon,
                &last_overlap,
            );
            self.ctx.evaluate_ego_pframe_chain_mut(&mut frame);
            let cand_w = *frame.phase_words_ref();
            Self::modulate_rank1_circuit_frame(circuit, &mut frame, &cand_w);
            spine.enforce_pinned_isolation_branchless(&sys_baseline, &mut frame);
            frame
        } else if adapted {
            let mut frame =
                signal.synthesize_gateway_with_aperture_from_stack(self.ctx.core_layers());
            self.ctx.core_layers().modulate_gateway_frame_branchless(
                &mut frame,
                &last_horizon,
                &last_overlap,
            );
            let cand_w = *frame.phase_words_ref();
            Self::modulate_rank1_circuit_frame(circuit, &mut frame, &cand_w);
            frame
        } else {
            let (mut raw_frame, frm_w) = signal.synthesize_gateway_frame_with_words();
            Self::modulate_rank1_circuit_frame(circuit, &mut raw_frame, &frm_w);
            raw_frame
        };
        let gain = self.apply_rank1_frame_internal(&frame, hidden_inout, u_vec, v_vec)?;
        Ok((frame, gain))
    }

    /// Evaluates Top-2 sparse surrogate entropy via [`EntropyController::eval_surrogate_and_step`]
    /// and updates `substrate.last_entropy_h_tilde` without rescaling `logits`.
    ///
    /// # Errors
    ///
    /// Returns an error if `logits` is empty or `EntropyController` reports an error.
    pub fn measure_entropy_state(
        &mut self,
        logits: &[f32],
    ) -> Result<lmnk_entropy_state_t, String> {
        if logits.is_empty() {
            return Err("logits slice must not be empty".to_string());
        }
        let mut max1 = f32::NEG_INFINITY;
        let mut max2 = f32::NEG_INFINITY;
        for &v in logits.iter() {
            if v > max1 {
                max2 = max1;
                max1 = v;
            } else if v > max2 {
                max2 = v;
            }
        }
        let sum_exp: f32 = logits
            .iter()
            .map(|&v| (v - max1).exp())
            .sum::<f32>()
            .max(1e-12);
        let p_max = 1.0 / sum_exp;
        let p_2nd = if max2.is_finite() {
            (max2 - max1).exp() / sum_exp
        } else {
            0.0
        };

        let mut state_out = lmnk_entropy_state_t::default();
        let _temp = self
            .ctx
            .entropy_controller_mut()
            .eval_surrogate_and_step(p_max, p_2nd, 0.0, 0.0, 1.0, 0.1, 0.1, &mut state_out)
            .map_err(|e| format!("eval_surrogate_and_step error: {e}"))?;
        self.ctx.substrate_mut().last_entropy_h_tilde = state_out.moving_z_entropy;
        Ok(state_out)
    }

    /// Evaluates Top-2 sparse surrogate entropy via [`Self::measure_entropy_state`]
    /// and scales `logits` by the resulting adaptive `sample_temperature`.
    ///
    /// # Errors
    ///
    /// Returns an error if `logits` is empty or `EntropyController` reports an error.
    pub fn step_entropy_logits(
        &mut self,
        logits: &mut [f32],
    ) -> Result<lmnk_entropy_state_t, String> {
        let state_out = self.measure_entropy_state(logits)?;
        let temp = state_out.sample_temperature;
        if temp.is_finite() && temp > 1e-4 && (temp - 1.0).abs() > 1e-4 {
            let inv_temp = 1.0 / temp;
            for v in logits.iter_mut() {
                *v *= inv_temp;
            }
        }
        Ok(state_out)
    }

    /// Applies Rank-1 residual injection (`y <- y + s * u * (v^T * y)`) to a hidden-state
    /// vector via [`ResidualAdapterEngine::eval_rank1_slices_fused_active`].
    ///
    /// # Errors
    ///
    /// Returns an error if slice lengths mismatch or `ResidualAdapterEngine` rejects inputs.
    pub fn apply_rank1_hidden_fused(
        &self,
        hidden_inout: &mut [f32],
        u_vec: &[f32],
        v_vec: &[f32],
        scale_factor: f32,
    ) -> Result<(), String> {
        let active_scale =
            match liminika_core::liminal::Scoped::from_boundary(ResidualScale(scale_factor))
                .cut_by_lacuna(scale_factor != 0.0, scale_factor, 0.0)
            {
                liminika_core::liminal::ScopedCut::Bottom(_) => {
                    return Err(format!(
                        "eval_rank1_slices_fused_active error: {}",
                        liminika_core::liminal::LMNK_ERR_INVALID_ARGUMENT
                    ));
                }
                cut => match cut.into_active() {
                    Some(active) => active,
                    None => return Ok(()),
                },
            };
        let d = hidden_inout.len();
        if d == 0 || u_vec.len() != d || v_vec.len() != d {
            return Err(format!(
                "Rank-1 residual hidden/u/v length mismatch: hidden={}, u={}, v={}",
                d,
                u_vec.len(),
                v_vec.len()
            ));
        }

        let mut y_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut x_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut u_buf = AlignedF32Buffer32::new_zeroed(d);
        let mut v_buf = AlignedF32Buffer32::new_zeroed(d);
        y_buf.as_mut_slice().copy_from_slice(hidden_inout);
        x_buf.as_mut_slice().copy_from_slice(hidden_inout);
        u_buf.as_mut_slice().copy_from_slice(u_vec);
        v_buf.as_mut_slice().copy_from_slice(v_vec);

        let mut y_mut = AlignedSliceMut32::new(y_buf.as_mut_slice())
            .map_err(|e| format!("AlignedSliceMut32 y error: {e}"))?;
        let x_aligned = AlignedSlice32::new(x_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 x error: {e}"))?;
        let u_aligned = AlignedSlice32::new(u_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 u error: {e}"))?;
        let v_aligned = AlignedSlice32::new(v_buf.as_slice())
            .map_err(|e| format!("AlignedSlice32 v error: {e}"))?;

        self.residual_adapter
            .eval_rank1_slices_fused_active(
                &mut y_mut,
                &x_aligned,
                &u_aligned,
                &v_aligned,
                None,
                active_scale,
            )
            .map_err(|e| format!("eval_rank1_slices_fused_active error: {e}"))?;

        hidden_inout.copy_from_slice(y_buf.as_slice());
        Ok(())
    }
}
