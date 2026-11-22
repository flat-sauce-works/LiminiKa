// name: src/adapter/src/model_adapter/recall_sync.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recall synchronization, workbench presets, probe calibration, and Mermaid diagnostics on [`ModelSteeringAdapter`].

use super::super::*;
use super::{ModelSteeringAdapter, RopeGeometry};

impl ModelSteeringAdapter {
    /// Updates Core Layer 0 (`Liminal`), Layer 1 (`Stigmergic`), and Layer 2 (`Ego`)
    /// branchlessly, leveraging `LiminalBoundaryDelta::stationary_shortcut_ready` to skip
    /// redundant adaptation churn when boundaries are quiescent without stagnation triggers.
    pub fn adapt_core_layers(
        &mut self,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
        overflow_opt: Option<&DissipativeOverflowReport>,
    ) {
        let stagnation_drive =
            (self.ctx.core_layers().ego.trigger_mask & EGO_TRIGGER_BOUNDARY_STAGNATION) != 0;
        let can_shortcut = self.ctx.core_layers().liminal.adaptation_epochs > 0
            && self.ctx.last_boundary_delta().stationary_shortcut_ready == 1
            && self.ctx.last_horizon().regime_code == horizon.regime_code
            && overlap_report.discovery_triggered == 0
            && self.ctx.last_overlap_report().overlap_core_mask == overlap_report.overlap_core_mask
            && overflow_opt.is_none()
            && !stagnation_drive;
        *self.ctx.last_horizon_mut() = *horizon;
        *self.ctx.last_overlap_report_mut() = *overlap_report;
        if can_shortcut {
            return;
        }

        let prev_layer0 = self.ctx.core_layers().liminal;
        let prev_overlap = self.ctx.core_layers().stigmergy.overlap_mask;
        let ledger = self.active_circuit().unwrap_or_default();
        self.ctx.core_layers_mut().adapt_layers_branchless(
            &ledger,
            horizon,
            overlap_report,
            overflow_opt,
        );
        let next_layer0 = self.ctx.core_layers().liminal;
        let prev_delta = *self.ctx.last_boundary_delta();
        *self.ctx.last_boundary_delta_mut() = LiminalBoundaryDelta::compute_branchless(
            &prev_layer0,
            &next_layer0,
            prev_overlap,
            overlap_report.overlap_core_mask,
            &prev_delta,
        );
        if self.ctx.core_layers().ego.is_active() {
            let boundary_delta = *self.ctx.last_boundary_delta();
            let mut preview = GatewayFrame {
                effective_energy: horizon.total_memory_support,
                ..GatewayFrame::default()
            };
            self.ctx
                .core_layers_mut()
                .intervene_ego_and_modulate_frame_branchless(
                    &boundary_delta,
                    horizon,
                    overlap_report,
                    &mut preview,
                );
        }
    }

    /// Synchronizes all three core layers (`Liminal`, `Stigmergic`, `Ego`), temporal boundary
    /// delta (`LiminalBoundaryDelta`), and fragmented micro-ensemble consensus directly from a
    /// live [`DynamicAttractorRecall`] outcome.
    pub fn sync_core_layers_from_recall(
        &mut self,
        recall: &DynamicAttractorRecall<'_>,
        query_words_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
    ) -> (
        EpistemicHorizonReading,
        OverlapDiscoveryReport,
        Option<StigmergicFragmentEnsembleReport>,
    ) {
        let has_guard_conflict = recall.guard_conflict_margin > 0.0;
        let guard_turn = has_guard_conflict
            || matches!(
                recall.action_decision,
                liminika_core::stigmergy::StigmergicActionDecision::RepulsionEscape
            );
        self.turn_guard_active = guard_turn;
        if guard_turn {
            let active_mask = self.ctx.compute_pool().active_mask;
            self.ctx.apply_host_spine_pin_floor(active_mask);
        } else {
            self.ctx.system_spine_mut().pinned_compute_mask = self.baseline_pinned_compute_mask;
        }
        self.ctx
            .apply_host_spine_pin_floor(self.baseline_pinned_compute_mask);
        *self.ctx.last_lacuna_partition_mut() = recall.lacuna_partition;
        self.ctx.system_spine_mut().unobserved_locked = u8::from(
            matches!(recall.lacuna_state, LacunaBoundaryState::LiminalExterior)
                && !recall.is_known_unknown_exterior
                && recall.guard_conflict_margin <= 0.05,
        );
        if !has_guard_conflict && self.configured_ego.is_active() {
            let tension = self.ctx.distortion_tension_mut();
            tension.effective_ego_scale_q7 = tension.effective_ego_scale_q7.max(48);
            tension.suppression_cooldown_steps = 0;
        }
        let regime_code = if has_guard_conflict
            || matches!(recall.lacuna_state, LacunaBoundaryState::LiminalBoundary)
        {
            EpistemicRegime::InternalConflict as u8
        } else if matches!(recall.lacuna_state, LacunaBoundaryState::LiminalExterior) {
            EpistemicRegime::Liminal as u8
        } else {
            EpistemicRegime::KnownCoherent as u8
        };
        let is_liminal = regime_code == EpistemicRegime::Liminal as u8;
        let horizon = EpistemicHorizonReading {
            total_memory_support: (recall.top1_fact_score.max(0.0) * 2.0).max(if is_liminal {
                0.0
            } else {
                0.35
            }),
            peak_positive_affinity: recall.top1_fact_score.max(0.0),
            internal_conflict_energy: if has_guard_conflict {
                recall.guard_conflict_margin.max(0.0)
            } else {
                (recall.top2_fact_score.max(0.0) * 0.5).max(0.0)
            },
            horizon_gate: if is_liminal { 0.0 } else { 1.0 },
            conflict_severity: if has_guard_conflict {
                recall.guard_conflict_margin.clamp(0.15, 1.0)
            } else if regime_code == EpistemicRegime::InternalConflict as u8 {
                0.25
            } else {
                0.0
            },
            dominant_anchor_id: recall
                .recalled_facts
                .first()
                .map_or(0, |(e, _)| e.anchor_id),
            opposing_anchor_id: recall.recalled_facts.get(1).map_or(0, |(e, _)| e.anchor_id),
            regime_code,
            reserved_align: [0; 3],
        };

        let q_words = query_words_opt
            .copied()
            .unwrap_or(*recall.gateway_frame.phase_words_ref());
        let mut drifted_words = q_words;
        for (g, w) in drifted_words.iter_mut().enumerate() {
            let tol = self.ctx.core_layers().stigmergy.granule_noise_tolerance_q7[g];
            let nudge = u64::from(tol.min(16)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            *w = swar_q7_circular_add_x8(*w, nudge);
        }

        let mut fact_fragments: Vec<ObjectiveFact<[i8; MAX_ATTENTION_HEADS]>> =
            Vec::with_capacity(recall.recalled_facts.len());
        for &(entry, score) in &recall.recalled_facts {
            let q7 = entry
                .per_layer_q7
                .last()
                .copied()
                .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
            let scaled_e = (entry.energy * score.clamp(0.25, 1.25)).max(0.25);
            fact_fragments.push(ObjectiveFact::new(q7, scaled_e, 0, entry.anchor_id));
        }

        let aperture_floor =
            f32::from(self.ctx.core_layers().liminal.threshold_floor_q7[1]) * (1.0 / 127.0);
        let overlap_report = evaluate_stigmergic_overlap_and_discovery_words_64(
            &fact_fragments,
            &q_words,
            &drifted_words,
            aperture_floor,
            self.ctx.core_layers().stigmergy.discovery_floor_q7,
        );

        self.adapt_core_layers(&horizon, &overlap_report, None);
        let ensemble_report = if fact_fragments.len() >= 2 {
            let (_frame, _words, rep) = self.synthesize_fragment_ensemble(&fact_fragments);
            let agreed_or_tie = rep.quorum_agreed_granule_mask | rep.ego_tiebreak_granule_mask;
            for (g, ph) in self
                .ctx
                .core_layers_mut()
                .stigmergy
                .granule_pheromone_q7
                .iter_mut()
                .enumerate()
                .take(NUM_STIGMERGIC_GRANULES)
            {
                let bit = 1u8 << g;
                if (agreed_or_tie & bit) != 0 {
                    *ph = ph.saturating_add(4).min(127);
                }
            }
            Some(rep)
        } else {
            None
        };
        self.last_ensemble_report = ensemble_report;
        self.escalation_boundary_dirty = true;
        let preset = if horizon.regime_code == EpistemicRegime::InternalConflict as u8 {
            WorkbenchPreset::GuardDefense
        } else if horizon.regime_code == EpistemicRegime::Liminal as u8 {
            WorkbenchPreset::CreativeLiminal
        } else if !recall.recalled_facts.is_empty() && recall.top1_fact_score >= 0.20 {
            WorkbenchPreset::FactualGrounding
        } else {
            WorkbenchPreset::BalancedDialog
        };

        let lane_budget = usize::from(preset.recommended_lane_budget());
        let mut specs: Vec<WorkbenchSlotSpec> = Vec::with_capacity(lane_budget);
        let repulsion_spec = has_guard_conflict.then(|| {
            let mut rep_phases = [0_i8; MAX_ATTENTION_HEADS];
            for (dst, &src) in rep_phases
                .iter_mut()
                .zip(recall.gateway_frame.phase_deltas.iter())
            {
                *dst = src.saturating_neg();
            }
            WorkbenchSlotSpec::auto_repulsion(
                recall.guard_conflict_margin.clamp(0.35, 1.0),
                rep_phases,
            )
        });
        if preset.prioritizes_repulsion() {
            if let Some(spec) = repulsion_spec {
                if specs.len() < lane_budget {
                    specs.push(spec);
                }
            }
        }
        for (fact_rank, &(entry, score)) in recall.recalled_facts.iter().take(3).enumerate() {
            if specs.len() >= lane_budget {
                break;
            }
            if let Some(&phases) = entry.per_layer_q7.last() {
                let (min_s, max_s, min_w, max_w) = match fact_rank {
                    0 => (0.25_f32, 1.0_f32, 0.15_f32, 1.0_f32),
                    1 => (0.20_f32, 0.90_f32, 0.10_f32, 0.90_f32),
                    _ => (0.15_f32, 0.80_f32, 0.10_f32, 0.80_f32),
                };
                specs.push(WorkbenchSlotSpec::auto_free(
                    (entry.energy * score.clamp(min_s, max_s)).clamp(min_w, max_w),
                    phases,
                ));
            }
        }
        if specs.len() < lane_budget {
            if let Some(&(entry, score)) = recall.recalled_personas.first() {
                if let Some(&phases) = entry.per_layer_q7.last() {
                    specs.push(WorkbenchSlotSpec::auto_free(
                        (entry.energy * score.clamp(0.20, 1.0)).clamp(0.15, 1.0),
                        phases,
                    ));
                }
            }
        }
        if !preset.prioritizes_repulsion() {
            if let Some(spec) = repulsion_spec {
                if specs.len() < lane_budget {
                    specs.push(spec);
                }
            }
        }
        let prev_was_sandbox = self.ctx.sandbox_workbench_lane().is_some();
        let _ = self.ctx.substrate_mut().apply_preset_lanes(&specs);
        *self.ctx.sandbox_workbench_lane_mut() = None;
        let assim_drift = self.ctx.telemetry_bank().assimilation_drift();
        if prev_was_sandbox || recall.is_known_unknown_exterior {
            let improved = prev_was_sandbox
                && recall.top1_fact_score > self.prev_sandbox_coherence
                && assim_drift <= self.prev_sandbox_assim_drift + 0.02;
            self.sandbox_settlement_streak = if improved {
                self.sandbox_settlement_streak.saturating_add(1)
            } else {
                0
            };
            self.prev_sandbox_coherence = recall.top1_fact_score;
            self.prev_sandbox_assim_drift = assim_drift;
        }
        if !has_guard_conflict && recall.is_known_unknown_exterior {
            if let Some(&(ext_entry, ext_score)) = recall.exterior_sandbox_facts.first() {
                if let Some(&ext_phases) = ext_entry.per_layer_q7.last() {
                    let ext_weight =
                        (ext_entry.energy * ext_score.clamp(0.20, 0.90)).clamp(0.15, 0.95);
                    if let Ok(lane) = self
                        .ctx
                        .substrate_mut()
                        .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(ext_weight, ext_phases))
                    {
                        *self.ctx.sandbox_workbench_lane_mut() = Some(lane);
                        if self.sandbox_settlement_streak >= 2
                            && self
                                .ctx
                                .substrate_mut()
                                .settle_sandbox_lane(lane)
                                .unwrap_or(false)
                        {
                            self.ctx.increment_sandbox_settlements();
                            *self.ctx.sandbox_workbench_lane_mut() = None;
                            self.sandbox_settlement_streak = 0;
                        }
                    }
                }
            }
        }
        if let Some(esc_lane) = self.ctx.escalated_workbench_lane() {
            if (self.ctx.substrate().slot_enabled_mask & (1u8 << esc_lane)) == 0 {
                *self.ctx.escalated_workbench_lane_mut() = None;
            }
        }
        if self.ctx.core_layers().ego.is_active() {
            let mut p_frame = recall.gateway_frame;
            let _ = self.ctx.evaluate_ego_pframe_chain_mut(&mut p_frame);
        }
        self.active_workbench_preset = preset;
        (horizon, overlap_report, ensemble_report)
    }

    /// Returns the anonymous workbench lane (`0..=7`) currently leased by cold-path escalation, if any.
    #[must_use]
    pub fn escalated_workbench_lane(&self) -> Option<u8> {
        self.ctx.escalated_workbench_lane()
    }

    /// Returns the currently active [`WorkbenchPreset`] selected for the 8-lane working bank.
    #[must_use]
    pub const fn active_workbench_preset(&self) -> WorkbenchPreset {
        self.active_workbench_preset
    }

    /// Switches the 8-lane working phase workbench to `preset` and atomically loads `specs`
    /// into [`StigmergicSubstrate`] (preserving pinned lanes), returning the number of lanes updated.
    ///
    /// # Errors
    ///
    /// Returns an error if any `spec` has invalid weight, out-of-range explicit lane (`>= 8`), or
    /// if an `AutoFree` allocation fails because all 8 lanes are pinned.
    pub fn switch_workbench_preset(
        &mut self,
        preset: WorkbenchPreset,
        specs: &[WorkbenchSlotSpec],
    ) -> Result<u8, String> {
        let updated = self
            .ctx
            .substrate_mut()
            .apply_preset_lanes(specs)
            .map_err(|e| format!("apply_preset_lanes error: {e}"))?;
        if let Some(esc_lane) = self.ctx.escalated_workbench_lane() {
            if (self.ctx.substrate().slot_enabled_mask & (1u8 << esc_lane)) == 0 {
                *self.ctx.escalated_workbench_lane_mut() = None;
            }
        }
        self.active_workbench_preset = preset;
        Ok(updated)
    }

    /// Captures the live [`LiminalTopologyTelemetrySnapshot`] and renders a self-introspecting
    /// 3-layer Mermaid `flowchart TD` diagram covering Core Layer 0 (`Liminal`), Core Layer 1
    /// (`StigmergicGranuleState`), Core Layer 2 (`LiminalEgoState`), cognitive aperture masks,
    /// and anchor populations.
    #[must_use]
    pub fn render_liminal_topology_mermaid(
        &self,
        active_anchors: u8,
        spawned_daughters: u8,
    ) -> String {
        let distortion = self.telemetry_snapshot().distortion;
        self.render_liminal_topology_mermaid_with_distortion(
            active_anchors,
            spawned_daughters,
            &distortion,
        )
    }

    /// Renders the live 3-layer Mermaid `flowchart TD` diagram using an explicit
    /// [`LayerDistortionReport`] (such as a multi-probe cumulative report).
    #[must_use]
    pub fn render_liminal_topology_mermaid_with_distortion(
        &self,
        active_anchors: u8,
        spawned_daughters: u8,
        distortion: &LayerDistortionReport,
    ) -> String {
        let core_layers = self.ctx.core_layers();
        let snap = LiminalTopologyTelemetrySnapshot::capture(
            &core_layers.liminal,
            self.ctx.last_horizon(),
            active_anchors,
            spawned_daughters,
        );
        let mut buf = [0u8; 2048];
        let written = snap.format_core_topology_mermaid_with_distortion(
            core_layers,
            self.ctx.last_overlap_report(),
            Some(distortion),
            &mut buf,
        );
        String::from_utf8_lossy(&buf[..written]).into_owned()
    }

    /// Returns a mutable reference to the dynamic [`AdaptiveThresholdPolicy`].
    pub fn threshold_policy_mut(&mut self) -> &mut AdaptiveThresholdPolicy {
        &mut self.threshold_policy
    }

    /// Updates hardware/runtime telemetry (`vram_pressure`, `gpu_frame_overrun`, active swarm mask)
    /// from an external environment probe and immediately refreshes cached thresholds.
    pub fn update_environment_telemetry(
        &mut self,
        vram_pressure: f32,
        gpu_frame_overrun: f32,
        active_swarm_mask: u8,
    ) {
        self.system_constraint =
            SystemConstraint::new((), vram_pressure, gpu_frame_overrun, active_swarm_mask);
        let neutral_trail = lmnk_pointer_trail_t::default();
        let probe = *self.ctx.probe_baseline();
        self.ctx.substrate_mut().thresholds = self.threshold_policy.resolve_thresholds(
            &self.system_constraint,
            &probe,
            &neutral_trail,
            probe.baseline_entropy,
        );
    }

    /// Cold-Path model calibration probe run before startup or periodically when drift / model
    /// changes occur. Measures the model's empirical inter-layer phase trajectory coherence and
    /// neutral entropy distribution, updates [`ManifoldProbeBaseline`], and equilibrates
    /// [`StigmergicThresholds`] without manual constant tuning.
    ///
    /// # Errors
    ///
    /// Returns an error if any probe Q-slice length mismatches `self.geometry().q_rot_len()`.
    pub fn calibrate_manifold_probe(
        &mut self,
        per_layer_sample_q: &[&[f32]],
        baseline_ref_per_layer: &[&[f32]],
        observed_entropies: &[f32],
    ) -> Result<ManifoldProbeBaseline, String> {
        let mut prev_q7: Option<Vec<lmnk_q7_t>> = None;
        let mut coh_sum = 0.0_f64;
        let mut coh_count = 0_usize;

        for (layer_idx, (q_sample, q_ref)) in per_layer_sample_q
            .iter()
            .zip(baseline_ref_per_layer.iter())
            .enumerate()
        {
            if !self.geometry.is_rope_layer(layer_idx) {
                continue;
            }
            let cur_q7 = self.bake_layer_procrustes_q7(q_ref, q_sample)?;
            if let Some(prev) = prev_q7.as_ref() {
                let mut dot = 0.0_f64;
                let mut n_a = 0.0_f64;
                let mut n_b = 0.0_f64;
                for (&a, &b) in prev.iter().zip(cur_q7.iter()) {
                    let fa = f64::from(a);
                    let fb = f64::from(b);
                    dot += fa * fb;
                    n_a += fa * fa;
                    n_b += fb * fb;
                }
                let denom = (n_a * n_b).sqrt();
                if denom > 1e-12 {
                    let cos = (dot / denom).clamp(-1.0, 1.0);
                    if cos > 0.0 {
                        coh_sum += cos;
                    }
                    coh_count += 1;
                }
            }
            prev_q7 = Some(cur_q7);
        }

        let active_heads = self.geometry.num_q_heads.min(64) as u16;
        let steered_layers = Self::compute_steered_layer_count(&self.geometry);
        {
            let baseline = self.ctx.probe_baseline_mut();
            if coh_count > 0 {
                baseline.baseline_inter_layer_coherence =
                    (coh_sum / coh_count as f64).clamp(0.05, 1.0) as f32;
            }
            let mut ent_sum = 0.0_f64;
            let mut ent_count = 0_usize;
            for &h in observed_entropies {
                if h.is_finite() && h >= 0.0 {
                    ent_sum += f64::from(h.clamp(0.0, 1.0));
                    ent_count += 1;
                }
            }
            if ent_count > 0 {
                baseline.baseline_entropy = (ent_sum / ent_count as f64) as f32;
            }
            baseline.active_heads = active_heads;
            baseline.steered_layers = steered_layers;
            baseline.probe_epoch = baseline.probe_epoch.saturating_add(1);
        }

        let probe = *self.ctx.probe_baseline();
        let neutral_trail = lmnk_pointer_trail_t::default();
        self.ctx.substrate_mut().thresholds = self.threshold_policy.resolve_thresholds(
            &self.system_constraint,
            &probe,
            &neutral_trail,
            probe.baseline_entropy,
        );
        Ok(probe)
    }

    /// Safely re-initializes the Cold-Path engines (`PhaseSteeringEngine`, `StigmergicSubstrate`,
    /// and `ledger_slot`) in-place when the underlying LLM or RoPE geometry changes during a
    /// periodic environment probe, returning the new `probe_epoch`.
    ///
    /// # Errors
    ///
    /// Returns an error if `new_geometry.validate()` fails or `liminika-core` rejects dimensions.
    pub fn rebind_geometry_cold_path(&mut self, new_geometry: RopeGeometry) -> Result<u32, String> {
        new_geometry.validate()?;
        let mut next_probe = *self.ctx.probe_baseline();
        next_probe.active_heads = new_geometry.num_q_heads.min(64) as u16;
        next_probe.steered_layers = Self::compute_steered_layer_count(&new_geometry);
        next_probe.probe_epoch = next_probe.probe_epoch.saturating_add(1);

        let neutral_trail = lmnk_pointer_trail_t::default();
        let calibrated = self.threshold_policy.resolve_thresholds(
            &self.system_constraint,
            &next_probe,
            &neutral_trail,
            next_probe.baseline_entropy,
        );
        let layout_byte = match new_geometry.layout {
            RopeLayout::Interleaved => 0,
            RopeLayout::HalfSplit => 1,
        };
        self.ctx
            .rebind_phase_and_substrate(
                new_geometry.rotary_dim as u32,
                new_geometry.num_q_heads as u32,
                new_geometry.rotary_dim as u32,
                layout_byte,
                16,
                calibrated,
                next_probe.baseline_entropy,
            )
            .map_err(|e| format!("rebind_phase_and_substrate failed: {e}"))?;

        let (active_slots, pinned_slots, reversible_slots) =
            Self::compute_slot_masks(new_geometry.num_layers);
        self.geometry = new_geometry;
        *self.ctx.sandbox_workbench_lane_mut() = None;
        *self.ctx.escalated_workbench_lane_mut() = None;
        self.escalation_boundary_dirty = true;
        self.last_escalated_regime = EpistemicRegime::KnownCoherent as u8;
        self.baseline_pinned_compute_mask = pinned_slots;
        self.ctx
            .compute_pool_mut()
            .set_slot_masks(active_slots, pinned_slots, 0, reversible_slots);
        self.ctx.system_spine_mut().pinned_compute_mask = pinned_slots;
        self.ctx.apply_host_spine_pin_floor(pinned_slots);
        self.ctx.reset_telemetry();
        self.ctx.set_probe_baseline(next_probe);
        Ok(next_probe.probe_epoch)
    }

    /// Returns a mutable reference to the embedded [`EntropyController`].
    pub fn entropy_controller_mut(&mut self) -> &mut EntropyController {
        self.ctx.entropy_controller_mut()
    }

    /// Returns a mutable reference to the embedded [`ResidualAdapterEngine`] (Rank-1 residual).
    pub fn residual_adapter_mut(&mut self) -> &mut ResidualAdapterEngine {
        &mut self.residual_adapter
    }
}
