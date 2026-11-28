// name: src/adapter/src/tests/timeline_and_codec.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Timeline, GOP seek, container codec, and reflection tests.

use super::*;

#[test]
fn container_v2_logical_pframe_topology_roundtrip_and_cross_model_rebake() {
    let mut src_adapter =
        ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("src adapter");
    src_adapter.configure_ego_from_preset(LiminalEgoPreset::Medium);
    let topo = src_adapter.export_logical_pframe_topology();
    assert_eq!(topo.len(), 3);

    let mut tl = LmnkTimeline::new(LmnkTimeline::geometry_fingerprint(src_adapter.geometry()));
    tl.set_logical_pframe_topology(topo.clone());

    let encoded = tl.to_bytes();
    let decoded = LmnkTimeline::from_bytes(&encoded).expect("decode v2");
    assert_eq!(decoded.logical_pframe_topology(), topo.as_slice());

    let mut alt_geom = sample_gqa_partial_rope();
    alt_geom.head_dim = 128;
    alt_geom.rotary_dim = 128;
    let mut alt_adapter = ModelSteeringAdapter::new(alt_geom).expect("alt adapter");
    alt_adapter.bake_logical_pframe_topology(decoded.logical_pframe_topology());
    assert_eq!(
        alt_adapter.ego_differential_net().active_node_mask & 0x07,
        0x07
    );
    assert!(alt_adapter.ego_differential_net().is_node_baked_active(0));
    assert!(alt_adapter.ego_differential_net().is_node_baked_active(2));
}

#[test]
fn hot_path_sandbox_settlement_increments_sandbox_settlements() {
    use liminika_core::liminal::EpistemicRegime;
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact};
    use liminika_core::stigmergy::OverlapDiscoveryReport;

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    adapter.configure_ego_from_preset(LiminalEgoPreset::Low);
    adapter.adapt_core_layers(
        &EpistemicHorizonReading {
            total_memory_support: 1.0,
            peak_positive_affinity: 1.0,
            horizon_gate: 1.0,
            conflict_severity: 0.0,
            regime_code: EpistemicRegime::KnownCoherent as u8,
            ..EpistemicHorizonReading::default()
        },
        &OverlapDiscoveryReport {
            overlap_core_mask: 0x0F,
            lacuna_exterior_mask: 0xF0,
            ..OverlapDiscoveryReport::default()
        },
        None,
    );
    assert_eq!(adapter.sandbox_settlements(), 0);

    let q_len = adapter.geometry().q_rot_len();
    let mut q = vec![0.25_f32; q_len];
    let mut trail = lmnk_pointer_trail_t {
        accumulated_phase_delta: [24_i8; MAX_ATTENTION_HEADS],
        ..lmnk_pointer_trail_t::default()
    };
    let sig = ObjectiveFact::new([24_i8; MAX_ATTENTION_HEADS], 2.0, 0, 7);

    // Run Hot-Path orchestration steps under sustained entropy drift so telemetry crosses
    // `continuous_standard_steps >= 16` + `assimilation_drift_dwell >= 3` to nucleate,
    // and then crosses `sandbox_coherence_improve_dwell >= 3` to settle in-place.
    for _ in 0..32 {
        let _ = adapter
            .orchestrate_layer_rope_with_substrate(
                LayerRopeStepInput {
                    layer_idx: 0,
                    signal: &sig,
                    entropy_h_tilde_opt: Some(0.90),
                    unresolved_damping_u7: 128,
                    q_rot_inout: &mut q,
                    logits_opt: None,
                },
                &mut trail,
            )
            .expect("hot-path step");
    }

    assert!(
        adapter.sandbox_settlements() >= 1,
        "expected Hot-Path SandboxLifecycleEvent::Settled to increment sandbox_settlements, got {}",
        adapter.sandbox_settlements()
    );
}

#[test]
fn adapter_ego_intent_natural_language_and_word_boundary_matching() {
    use liminika_core::drivers::{LiminalEgoPreset, LiminalEgoState};

    // 1. Blank/whitespace inputs remain inactive.
    assert!(!parse_ego_intent_axes("").is_active());
    assert!(!parse_ego_intent_axes("   \t\n  ").is_active());
    assert!(!ego_state_from_natural_language("   ").is_active());

    // 2. Active rebellious/transgressive NL cue profile outranks calm logical cautious profile.
    let rebel = ego_state_from_natural_language(
        "Strong-willed stubborn rebel who defies constraints and explores beyond limits",
    );
    let calm = ego_state_from_natural_language("Calm logical cautious observer");
    let med = LiminalEgoState::from_preset(LiminalEgoPreset::Medium);
    let overlay = overlay_ego_state_from_natural_language(med, "creative curious explore");
    let symbolic = ego_state_from_natural_language("@persona_profile_01");

    assert!(rebel.is_active());
    assert!(rebel.basin_resistance_q7 > calm.basin_resistance_q7);
    assert!(rebel.boundary_crossing_q7 > calm.boundary_crossing_q7);
    assert!(overlay.is_active());
    assert!(overlay.boundary_crossing_q7 > med.boundary_crossing_q7);
    assert!(symbolic.is_active());
    assert_ne!(symbolic.ego_signature_hash, 0);
    assert!(symbolic.basin_resistance_q7 >= 24);

    // 3. Word-boundary enforcement: `"free"` must not match `"freeze"`, `"ego"` must not match `"category"`.
    let freeze_axes = parse_ego_intent_axes("freeze");
    let free_axes = parse_ego_intent_axes("free");
    let category_axes = parse_ego_intent_axes("category");
    let ego_axes = parse_ego_intent_axes("(Ego)");

    assert_eq!(
        freeze_axes.confidence_q7, 64,
        "substring inside `freeze` must hit hash fallback, not keyword match"
    );
    assert_eq!(free_axes.confidence_q7, 112);
    assert_eq!(free_axes.autonomy_q7, 32);

    assert_eq!(
        category_axes.confidence_q7, 64,
        "substring inside `category` must hit hash fallback, not keyword match"
    );
    assert_eq!(ego_axes.confidence_q7, 112);
    assert_eq!(ego_axes.autonomy_q7, 32);

    // Combined phrase: only `calm` (logic) and `free` (autonomy) match; `category` does not add a second autonomy hit.
    let mixed = parse_ego_intent_axes("Calm and free in category");
    let expected_mixed = parse_ego_intent_axes("Calm and free in neutral");
    assert_eq!(mixed.confidence_q7, 112);
    assert_eq!(mixed.autonomy_q7, expected_mixed.autonomy_q7);
    assert_eq!(mixed.logic_q7, expected_mixed.logic_q7);
    assert_eq!(mixed.defiance_q7, expected_mixed.defiance_q7);
}

#[test]
fn adapter_preserves_baseline_pins_and_allows_calm_whole_token_bypass() {
    use liminika_core::drivers::{SystemPressureReading, TokenBypassObservation};

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let expected_baseline_pins = 0x0000_000F_u64 | (0x0000_000F_u64 << 28);
    assert_eq!(
        adapter.envelope().pinned_compute_mask,
        expected_baseline_pins
    );

    let fact = TextAttractorEntry {
        anchor_id: 1,
        role: AttractorRole::Fact,
        text: "Mars orbits the Sun at roughly 1.52 AU.".to_string(),
        per_layer_q7: vec![[24i8; MAX_ATTENTION_HEADS]],
        energy: 2.0,
        residual_sig: vec![1.0, 0.0],
    };
    let calm_recall = DynamicAttractorRecall {
        lacuna_state: LacunaBoundaryState::InsideBasin,
        top1_fact_score: 0.85,
        top2_fact_score: 0.40,
        top1_persona_score: 0.15,
        min_local_lacuna_depth: 0.0,
        guard_conflict_margin: -0.30,
        action_decision: StigmergicActionDecision::FullOrchestration,
        gateway_frame: GatewayFrame::default(),
        telemetry: StratumProbeTelemetry::default(),
        recalled_facts: vec![(&fact, 0.85)],
        recalled_personas: Vec::new(),
        top_guard: None,
        lacuna_partition: LacunaPartition8::default(),
        exterior_sandbox_facts: Vec::new(),
        is_known_unknown_exterior: false,
    };

    // Calm non-guard recall must preserve baseline head/tail pins without clearing to 0.
    let _ = adapter
        .sync_core_layers_from_recall(&calm_recall, None)
        .expect("sync calm recall");
    assert_eq!(
        adapter.envelope().pinned_compute_mask,
        expected_baseline_pins
    );

    // Whole-token bypass can engage across consecutive calm steps despite head/tail pins.
    let surge = SystemPressureReading {
        frame_time_pressure_q15: 30000,
        bandwidth_pressure_q15: 30000,
        l3_load_pressure_q15: 15000,
        irreversible_step: 0,
        _pad: 0,
    };
    let calm_obs = TokenBypassObservation {
        irreversible_step: 0,
        liminal_stationary: 1,
        boundary_sensitivity_q7: 16,
        repulsion_active: 0,
        exit_entropy_rising: 0,
        ego_active_boundary: 0,
        min_calm_dwell: 2,
        _pad: 0,
    };
    let mut bypassed = false;
    for _ in 0..4 {
        let mut d = adapter.step_compute_dispatch(surge);
        assert_eq!(d.bypass_mask & expected_baseline_pins, 0);
        if adapter.evaluate_whole_token_bypass(calm_obs, &mut d) {
            bypassed = true;
            break;
        }
    }
    assert!(
        bypassed,
        "calm non-guard turns should permit whole-token bypass"
    );

    // Guard turn pins all active slots and vetoes whole-token bypass; subsequent calm turn
    // restores `expected_baseline_pins` (never wiping permanent anchors to 0).
    let guard_recall = DynamicAttractorRecall {
        guard_conflict_margin: 0.30,
        action_decision: StigmergicActionDecision::RepulsionEscape,
        ..calm_recall.clone()
    };
    let _ = adapter
        .sync_core_layers_from_recall(&guard_recall, None)
        .expect("sync guard recall");
    assert_eq!(
        adapter.envelope().pinned_compute_mask,
        adapter.compute_pool().active_mask
    );
    let mut d_guard = adapter.step_compute_dispatch(surge);
    assert!(!adapter.evaluate_whole_token_bypass(calm_obs, &mut d_guard));

    let _ = adapter
        .sync_core_layers_from_recall(&calm_recall, None)
        .expect("sync calm recall");
    assert_eq!(
        adapter.envelope().pinned_compute_mask,
        expected_baseline_pins
    );

    // Rebind cold-path resets baseline pins to the new geometry's head/tail boundaries.
    let mut small_geom = sample_gqa_partial_rope();
    small_geom.num_layers = 8;
    small_geom.layer_rope_mask = 0xFF;
    let _ = adapter
        .rebind_geometry_cold_path(small_geom)
        .expect("rebind");
    assert_eq!(adapter.envelope().pinned_compute_mask, 0x81_u64);
}

#[test]
fn adapter_embeds_boxed_runtime_context_with_drop_and_spine_pin_floor() {
    use liminika_core::runtime::RuntimeContext;

    assert_eq!(core::mem::size_of::<RuntimeContext>(), 5184);
    assert_eq!(core::mem::align_of::<RuntimeContext>(), 64);
    assert!(!core::mem::needs_drop::<RuntimeContext>());
    assert!(core::mem::needs_drop::<ModelSteeringAdapter>());

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let baseline_pins = 0x0000_000F_u64 | (0x0000_000F_u64 << 28);

    // Embedded `RuntimeContext` must mirror adapter initialization parity and pin floor.
    assert_eq!(
        adapter.runtime_context().envelope().pinned_compute_mask,
        baseline_pins
    );
    assert_eq!(
        adapter.envelope().pinned_compute_mask,
        adapter.runtime_context().envelope().pinned_compute_mask
    );

    // Verify `apply_host_spine_pin_floor` is enforced monotonically on the embedded context.
    adapter.runtime_context_mut().set_baseline_pin_mask(0);
    adapter
        .runtime_context_mut()
        .apply_host_spine_pin_floor(baseline_pins);
    assert_eq!(
        adapter.runtime_context().envelope().pinned_compute_mask,
        baseline_pins
    );

    // Verify explicit `drop` executes cleanly (scrubbing the embedded `RuntimeContext`).
    drop(adapter);
}

#[test]
fn liminal_boundary_flow_summary_is_zero_safe_and_reflects_live_counters() {
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");

    // Zero-step safety: no NaN or division-by-zero on fresh adapter.
    let zero_summary: LiminalBoundaryFlowSummary = adapter.liminal_boundary_flow_summary();
    assert_eq!(zero_summary.total_steps, 0);
    assert_eq!(zero_summary.continuous_standard_steps, 0);
    assert_eq!(zero_summary.abstract_residency_ratio, 0.0);
    assert_eq!(zero_summary.settlement_ratio, 0.0);
    assert_eq!(zero_summary.tension_balance_ratio, 0.0);
    assert_eq!(zero_summary.ego_excitation_ratio, 0.0);
    assert!(zero_summary.abstract_residency_ratio.is_finite());
    assert!(zero_summary.settlement_ratio.is_finite());
    assert!(zero_summary.tension_balance_ratio.is_finite());
    assert!(zero_summary.ego_excitation_ratio.is_finite());

    // Drive real adapter steps so `CoreTelemetryBank` accumulates live counters.
    let mut trail = lmnk_pointer_trail_t::default();
    let fact = ObjectiveFact::new([20i8; 64], 2.5, 0, 42);
    let len = adapter.geometry().q_rot_len();
    let mut q_rot = vec![0.25_f32; len];
    let mut logits = vec![2.5_f32, 1.0, 0.2, -0.5];
    let _ = adapter
        .steer_layer_with_gateway(0, &fact, &mut trail, 64, &mut q_rot, Some(&mut logits))
        .expect("gateway step");

    // Exercise whole-token bypass telemetry counter on the live adapter.
    let mut dispatch = ComputeDispatchMask::default();
    let obs = TokenBypassObservation {
        irreversible_step: 0,
        liminal_stationary: 1,
        boundary_sensitivity_q7: 16,
        repulsion_active: 0,
        exit_entropy_rising: 0,
        ego_active_boundary: 0,
        min_calm_dwell: 1,
        _pad: 0,
    };
    let _ = adapter.evaluate_whole_token_bypass_with_spine(obs, &mut dispatch);

    let live = adapter.liminal_boundary_flow_summary();
    let snap = adapter.telemetry_snapshot();
    assert_eq!(live.total_steps, snap.spine.total_steps);
    assert_eq!(
        live.continuous_standard_steps,
        snap.spine.continuous_standard_steps
    );
    assert_eq!(
        live.whole_token_bypass_events,
        snap.whole_token_bypass_events
    );
    assert_eq!(live.ego_interventions, snap.ego.intervention_count);
    assert_eq!(live.sandbox_settlements, adapter.sandbox_settlements());
    assert!((0.0..=1.0).contains(&live.abstract_residency_ratio));
    assert!((0.0..=1.0).contains(&live.settlement_ratio));
    assert!((0.0..=1.0).contains(&live.tension_balance_ratio));
    assert!((0.0..=1.0).contains(&live.ego_excitation_ratio));

    // Render Mermaid + Markdown table and verify GitHub Mermaid parser safety
    // (zero raw `<` or `>` inside quoted node labels `["..."]`).
    let rendered = adapter.render_liminal_flow_mermaid();
    assert!(rendered.contains("```mermaid"));
    assert!(rendered.contains("flowchart LR"));
    assert!(rendered.contains("| Boundary Axis |"));
    for line in rendered.lines() {
        if let Some(start) = line.find("[\"") {
            let label = &line[start + 2..];
            if let Some(end) = label.find("\"]") {
                let inner = &label[..end];
                assert!(
                    !inner.contains('<') && !inner.contains('>'),
                    "Mermaid node label must not contain raw angle brackets: {inner}"
                );
            }
        }
    }
}

#[test]
fn adapter_transient_guard_relaxation_and_closed_loop_reescalation() {
    use liminika_core::drivers::{
        HOST_IMMUTABLE_PINNED_COMPUTE_MASK, SPINE_BYPASS_VETO_GUARD_ACTIVE, SPINE_BYPASS_VETO_NONE,
        SystemPressureReading, TokenBypassObservation,
    };

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let expected_baseline_pins = 0x0000_000F_u64 | (0x0000_000F_u64 << 28);
    assert_eq!(
        adapter.guard_hold_tokens(),
        ModelSteeringAdapter::DEFAULT_GUARD_HOLD_TOKENS
    );

    let fact = TextAttractorEntry {
        anchor_id: 1,
        role: AttractorRole::Fact,
        text: "LiminiKa does not fine-tune PyTorch weights or require 24 GB VRAM.".to_string(),
        per_layer_q7: vec![[24i8; MAX_ATTENTION_HEADS]],
        energy: 2.0,
        residual_sig: vec![1.0, 0.0],
    };
    let guard_recall = DynamicAttractorRecall {
        lacuna_state: LacunaBoundaryState::LiminalBoundary,
        top1_fact_score: 0.70,
        top2_fact_score: 0.55,
        top1_persona_score: 0.20,
        min_local_lacuna_depth: 0.0,
        guard_conflict_margin: 0.35,
        action_decision: StigmergicActionDecision::FullOrchestration,
        gateway_frame: GatewayFrame::default(),
        telemetry: StratumProbeTelemetry::default(),
        recalled_facts: vec![(&fact, 0.70)],
        recalled_personas: Vec::new(),
        top_guard: None,
        lacuna_partition: LacunaPartition8::default(),
        exterior_sandbox_facts: Vec::new(),
        is_known_unknown_exterior: false,
    };

    // Slide in CoherenceHorizonGuardSlot so static `guard_slot_slid_in && conflict_severity > 0.25`
    // is active alongside `turn_guard_active` without activating `RepulsionEscape` (verifies Trap-2 resolution).
    let _ = adapter
        .sync_core_layers_from_recall(&guard_recall, None)
        .expect("sync guard recall");
    let mut trail = lmnk_pointer_trail_t::default();
    let sig = ObjectiveFact::new([18i8; 64], 1.5, 0, 1);
    let mut q_rot = vec![0.25_f32; adapter.geometry().q_rot_len()];
    let _ = adapter
        .orchestrate_layer_rope_with_substrate(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &sig,
                entropy_h_tilde_opt: Some(0.35),
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut trail,
        )
        .expect("slide-in guard slot");
    assert!(
        adapter.coherence_guard_slot().activations > adapter.coherence_guard_slot().retractions
    );
    assert!(adapter.runtime_context().last_horizon().conflict_severity > 0.25);
    assert!(!adapter.effective_spine_guard_and_repulsion(false).1);

    let surge = SystemPressureReading {
        frame_time_pressure_q15: 8192,
        bandwidth_pressure_q15: 8192,
        l3_load_pressure_q15: 4096,
        irreversible_step: 0,
        _pad: 0,
    };
    let calm_obs = TokenBypassObservation {
        irreversible_step: 0,
        liminal_stationary: 1,
        boundary_sensitivity_q7: 16,
        repulsion_active: 0,
        exit_entropy_rising: 0,
        ego_active_boundary: 0,
        min_calm_dwell: 2,
        _pad: 0,
    };

    // (a) Held during `0..guard_hold_tokens` (`W = 4`).
    adapter.set_guard_hold_tokens(4);
    for _ in 0..4 {
        assert!(!adapter.evaluate_transient_guard_relaxation(0.40, 1, false, false, 2));
        assert!(!adapter.is_transient_guard_relaxed());
        assert_eq!(
            adapter.envelope().pinned_compute_mask,
            adapter.compute_pool().active_mask
        );
        assert!(adapter.effective_spine_guard_and_repulsion(false).0);
        let mut d = adapter.step_compute_dispatch(surge);
        let veto = adapter.evaluate_whole_token_bypass_with_spine(calm_obs, &mut d);
        assert_ne!(veto & SPINE_BYPASS_VETO_GUARD_ACTIVE, 0);
        assert_eq!(d.whole_token_steering_bypass, 0);
    }

    // (d) Guarded gates block relaxation when entropy >= 0.72, NaN, repulsion active, or unobserved_locked.
    assert!(!adapter.evaluate_transient_guard_relaxation(0.72, 1, false, false, 2));
    assert!(!adapter.evaluate_transient_guard_relaxation(f32::NAN, 1, false, false, 2));
    assert!(!adapter.evaluate_transient_guard_relaxation(0.40, 1, true, false, 2));
    adapter.runtime_context_mut().set_unobserved_locked(true);
    assert!(!adapter.evaluate_transient_guard_relaxation(0.40, 1, false, false, 2));
    adapter.runtime_context_mut().set_unobserved_locked(false);

    // (c) `guard_hold_tokens == 0` disables relaxation for the entire turn.
    adapter.set_guard_hold_tokens(0);
    assert!(!adapter.evaluate_transient_guard_relaxation(0.40, 1, false, false, 2));

    // (b) Re-enabling `guard_hold_tokens = 4` relaxes cleanly, clears both static guard clauses,
    // preserves `expected_baseline_pins` & `HOST_IMMUTABLE_PINNED_COMPUTE_MASK`, and enters
    // whole-token bypass on the very next step without extra cooldown lag.
    adapter.set_guard_hold_tokens(4);
    assert!(adapter.evaluate_transient_guard_relaxation(0.40, 1, false, false, 2));
    assert!(adapter.is_transient_guard_relaxed());
    assert_eq!(
        adapter.envelope().pinned_compute_mask,
        expected_baseline_pins
    );
    assert_ne!(
        adapter.envelope().pinned_compute_mask & HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
        0
    );
    assert!(!adapter.effective_spine_guard_and_repulsion(false).0);
    let mut d_relaxed = adapter.step_compute_dispatch(surge);
    let veto_relaxed = adapter.evaluate_whole_token_bypass_with_spine(calm_obs, &mut d_relaxed);
    assert_eq!(veto_relaxed, SPINE_BYPASS_VETO_NONE);
    assert_eq!(d_relaxed.whole_token_steering_bypass, 1);

    // (e) Post-relaxation `exit_entropy_rising` or `repulsion_active` re-escalates immediately.
    assert!(!adapter.evaluate_transient_guard_relaxation(0.50, 1, false, true, 2));
    assert!(!adapter.is_transient_guard_relaxed());
    assert_eq!(
        adapter.envelope().pinned_compute_mask,
        adapter.compute_pool().active_mask
    );
    assert!(adapter.effective_spine_guard_and_repulsion(false).0);
    let mut d_re_esc = adapter.step_compute_dispatch(surge);
    let veto_re_esc = adapter.evaluate_whole_token_bypass_with_spine(calm_obs, &mut d_re_esc);
    assert_ne!(veto_re_esc & SPINE_BYPASS_VETO_GUARD_ACTIVE, 0);
    assert_eq!(d_re_esc.whole_token_steering_bypass, 0);

    // Syncing a new recall clears `transient_guard_relaxed` and resets turn counters while keeping
    // the configured `guard_hold_tokens` across `reset_horizon_and_core_state`.
    adapter.reset_horizon_and_core_state();
    assert_eq!(adapter.guard_hold_tokens(), 4);
    assert!(!adapter.is_transient_guard_relaxed());

    // Strong conflict (`guard_conflict_margin: 0.75`) triggering `RepulsionEscape` raises
    // effective repulsion and strictly blocks transient guard relaxation.
    let repulse_recall = DynamicAttractorRecall {
        guard_conflict_margin: 0.75,
        ..guard_recall
    };
    let _ = adapter
        .sync_core_layers_from_recall(&repulse_recall, None)
        .expect("sync repulse recall");
    let _ = adapter
        .orchestrate_layer_rope_with_substrate(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &sig,
                entropy_h_tilde_opt: Some(0.35),
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut trail,
        )
        .expect("repulse slide-in");
    assert!(adapter.effective_spine_guard_and_repulsion(false).1);
    for _ in 0..6 {
        assert!(!adapter.evaluate_transient_guard_relaxation(0.40, 1, false, false, 2));
        let mut d = adapter.step_compute_dispatch(surge);
        let _ = adapter.evaluate_whole_token_bypass_with_spine(calm_obs, &mut d);
    }
    assert!(!adapter.is_transient_guard_relaxed());
}

#[test]
fn snapshot_restore_under_active_transient_guard_does_not_leak_permanent_pins() {
    let mut clean_adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("adapter");
    let baseline = clean_adapter.envelope().baseline_pinned_compute_mask;
    clean_adapter.begin_turn_spine_policy(false, false);
    assert_eq!(clean_adapter.envelope().pinned_compute_mask, baseline);

    let mut req = 0usize;
    clean_adapter
        .runtime_context()
        .serialize_to_bytes(&mut [], &mut req)
        .expect("size snapshot");
    let mut buf = vec![0u8; req];
    clean_adapter
        .runtime_context()
        .serialize_to_bytes(&mut buf, &mut req)
        .expect("serialize unguarded snapshot");

    // Arm transient all-layer guard pin on live adapter, then restore clean unguarded snapshot.
    let mut guarded_adapter =
        ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("guarded adapter");
    guarded_adapter.begin_turn_spine_policy(true, false);
    assert_eq!(
        guarded_adapter.envelope().pinned_compute_mask,
        guarded_adapter.compute_pool().active_mask
    );
    assert_eq!(
        guarded_adapter.envelope().baseline_pinned_compute_mask,
        baseline
    );

    guarded_adapter
        .runtime_context_mut()
        .deserialize_from_bytes(&buf)
        .expect("restore snapshot");
    assert_eq!(
        guarded_adapter.envelope().baseline_pinned_compute_mask,
        baseline
    );
    assert_eq!(guarded_adapter.envelope().pinned_compute_mask, baseline);
    assert!(!guarded_adapter.system_spine().turn_guard_active());
}

#[test]
fn adapter_begin_turn_and_step_decode_control_delegate_to_core_ctx() {
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("adapter");
    adapter.set_guard_hold_tokens(2);
    adapter.set_max_observe_stride(4);
    assert_eq!(adapter.guard_hold_tokens(), 2);
    assert_eq!(adapter.max_observe_stride(), 4);
    assert!(adapter.is_pre_bypass_enabled());

    adapter.begin_turn_spine_policy(false, false);
    let p = SystemPressureReading {
        frame_time_pressure_q15: 8_192,
        bandwidth_pressure_q15: 8_192,
        l3_load_pressure_q15: 4_096,
        irreversible_step: 0,
        _pad: 0,
    };
    let s0 = adapter.step_decode_control(&SpineDecodeInput::from_observed_entropy_f32(
        p, 0, 0.30, 8, false, true,
    ));
    assert_eq!(
        s0.next_observe_mode_enum(),
        SpineLogitsObserveMode::ObserveFullLogits
    );
    assert!(!s0.is_whole_token_bypass());

    let s1 = adapter.step_decode_control(&SpineDecodeInput::from_observed_entropy_f32(
        p, 1, 0.30, 8, false, true,
    ));
    assert!(s1.is_whole_token_bypass());
    assert_eq!(
        s1.next_observe_mode_enum(),
        SpineLogitsObserveMode::SkipToDeviceArgmax
    );
    let flow = adapter.liminal_boundary_flow_summary();
    assert_eq!(flow.concrete_token_count, 2);
    assert_eq!(flow.abstract_token_count, 0);
    assert!(
        adapter
            .render_liminal_flow_mermaid()
            .contains("Decode Spine Control")
    );
}

#[test]
fn load_from_path_recognizes_uppercase_lmnkp_extension() {
    let dir = std::env::temp_dir().join(format!("liminika_upper_lmnkp_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp dir");

    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("adapter");
    let mut tl = LmnkTimeline::new(LmnkTimeline::geometry_fingerprint(adapter.geometry()));
    let cut = AdaptiveGopCutDetector::default();
    let policy = RecallPolicy::default();
    let mut corpus = TextAttractorCorpus::new();
    let q_vecs = vec![vec![0.1_f32; adapter.geometry().q_rot_len()]; adapter.geometry().num_layers];
    let ref_vecs =
        vec![vec![0.0_f32; adapter.geometry().q_rot_len()]; adapter.geometry().num_layers];
    let q_refs: Vec<&[f32]> = q_vecs.iter().map(Vec::as_slice).collect();
    let ref_refs: Vec<&[f32]> = ref_vecs.iter().map(Vec::as_slice).collect();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "LiminiKa uppercase extension check.",
            &q_refs,
            &ref_refs,
            2.0,
        )
        .expect("ingest");
    tl.record_step(
        &TimelineStepContext {
            adapter: &adapter,
            corpus: &corpus,
            policy: &policy,
            detector: &cut,
        },
        SourceTextRecord {
            frame_index: 0,
            timestamp_epoch_sec: 1_700_000_000,
            role: AttractorRole::Fact,
            text: "LiminiKa uppercase extension check.".to_string(),
            energy: 2.0,
        },
        &FrameTransitionMetrics {
            phase_l1_delta: 0.0,
            lacuna_state: LacunaBoundaryState::InsideBasin,
            frames_since_keyframe: 0,
            new_cluster_spawned: true,
        },
    )
    .expect("record step");

    let upper_valid = dir.join("package_upper.LMNKP");
    tl.save_lmnkp_to_path(&upper_valid, None)
        .expect("save uppercase .LMNKP");
    let loaded = LmnkTimeline::load_from_path(&upper_valid).expect("load uppercase .LMNKP");
    assert_eq!(loaded, tl);

    // Without PK magic, `.LMNKP` must still route through the package parser (returning `Package`).
    let upper_corrupt = dir.join("corrupt_upper.LMNKP");
    std::fs::write(&upper_corrupt, b"NOTAZIP").expect("write corrupt .LMNKP");
    let err = LmnkTimeline::load_from_path(&upper_corrupt)
        .expect_err("corrupt .LMNKP must fail inside package parser");
    assert!(
        matches!(err, LmnkTimelineError::Package(_)),
        "expected LmnkTimelineError::Package for .LMNKP, got {err:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn guard_repulsion_boundary_writes_nonzero_antiphase_to_auto_repulsion_lane() {
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let head_rot = adapter.geometry().rotary_dim;
    let num_heads = adapter.geometry().num_q_heads;
    let num_layers = adapter.geometry().num_layers;

    // Reference vector places unit energy on each head's (0, half) rotary pair so Procrustes
    // baking produces a strictly non-zero per-head Q7 phase profile for the Guard attractor.
    let half = head_rot / 2;
    let mut centroid = vec![0.0_f32; len];
    let mut fact_vec = vec![0.0_f32; len];
    let mut guard_vec = vec![0.0_f32; len];
    for h in 0..num_heads {
        let base = h * head_rot;
        centroid[base] = 1.0;
        fact_vec[base] = 0.85;
        fact_vec[base + half] = 0.52;
        guard_vec[base] = 0.60;
        guard_vec[base + half] = -0.80;
    }
    // Keep fact and guard strongly separated in residual space while matching query to guard.
    fact_vec[1] = 2.0;
    guard_vec[2] = 2.0;
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Reference attractive anchor",
            &vec![fact_vec.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("ingest fact");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Guard,
            "Reference repulsive guard anchor",
            &vec![guard_vec.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("ingest guard");

    let guard_q7 = corpus
        .entries()
        .iter()
        .find(|e| e.role == AttractorRole::Guard)
        .and_then(|e| e.per_layer_q7.last().copied())
        .expect("baked guard q7");
    assert!(
        guard_q7.iter().any(|&v| v != 0),
        "baked guard anchor must have non-zero Q7 phase deltas"
    );

    let policy = RecallPolicy {
        min_coherence: 0.20,
        boundary_margin: 0.04,
        guard_suppress_margin: 0.0,
        top_k_facts: 1,
        top_k_personas: 1,
    };
    let recall = corpus
        .recall_for_query(
            &adapter,
            &vec![guard_vec.as_slice(); num_layers],
            &refs,
            &policy,
        )
        .expect("guard repulsion recall");
    assert_eq!(
        recall.action_decision,
        StigmergicActionDecision::RepulsionEscape
    );
    assert!(recall.gateway_frame.is_repulsed());

    let mut expected_antiphase = [0i8; MAX_ATTENTION_HEADS];
    for (dst, &src) in expected_antiphase.iter_mut().zip(guard_q7.iter()) {
        *dst = src.saturating_neg();
    }
    assert_eq!(recall.gateway_frame.phase_deltas, expected_antiphase);
    assert!(recall.gateway_frame.phase_deltas.iter().any(|&v| v != 0));

    let _ = adapter
        .sync_core_layers_from_recall(&recall, None)
        .expect("sync recall");
    let rep_mask = adapter.substrate().repulsion_lane_mask();
    assert_ne!(
        rep_mask, 0,
        "expected an active WorkbenchSlotSpec::auto_repulsion lane"
    );
    let rep_lane = rep_mask.trailing_zeros() as usize;
    let start = rep_lane * MAX_ATTENTION_HEADS;
    let lane_phases = &adapter.substrate().slot_phase_bank.data[start..start + MAX_ATTENTION_HEADS];
    assert_eq!(
        lane_phases, &expected_antiphase,
        "auto_repulsion lane must store non-zero anti-phase `-guard` without double negation"
    );

    // TrajectoryManifoldSnapshot::from_recall captures observed `top_guard_q7` directly.
    let snap = TrajectoryManifoldSnapshot::from_recall(&recall);
    assert_eq!(snap.top_guard_q7, Some(guard_q7));
}

#[test]
fn corpus_recall_escalation_writes_antiphase_for_guard_and_stays_unobserved_without_guard() {
    use liminika_core::liminal::EpistemicRegime;
    use liminika_core::runtime::{
        HorizonEscalationProvider, HorizonEscalationRequest, HorizonTriggerInput,
    };
    use liminika_core::stigmergy::WORKBENCH_LANE_FLAG_REPULSION;

    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let q_len = adapter.geometry().q_rot_len();
    let head_rot = adapter.geometry().rotary_dim;
    let half = head_rot / 2;
    let num_heads = adapter.geometry().num_q_heads;
    let num_layers = adapter.geometry().num_layers;

    let mut centroid = vec![0.0_f32; q_len];
    let mut fact_vec = vec![0.0_f32; q_len];
    let mut guard_vec = vec![0.0_f32; q_len];
    for h in 0..num_heads {
        let base = h * head_rot;
        centroid[base] = 1.0;
        fact_vec[base] = 0.85;
        fact_vec[base + half] = 0.52;
        guard_vec[base] = 0.60;
        guard_vec[base + half] = -0.80;
    }
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    // 1. Fact-only corpus must return Ok(None) under WORKBENCH_LANE_FLAG_REPULSION (zero leakage).
    let mut fact_only_corpus = TextAttractorCorpus::new();
    fact_only_corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Fact only",
            &vec![fact_vec.as_slice(); num_layers],
            &refs,
            2.0,
        )
        .expect("ingest fact");
    let mut fact_only_esc = CorpusRecallEscalationProvider::new(&fact_only_corpus);
    let rep_req = HorizonEscalationRequest {
        trigger: HorizonTriggerInput {
            repulsion_severity: 0.75,
            epistemic_regime: EpistemicRegime::InternalConflict as u8,
            ..HorizonTriggerInput::default()
        },
        lane_flags: WORKBENCH_LANE_FLAG_REPULSION,
        layer_scale_q7: 127,
        ..HorizonEscalationRequest::default()
    };
    let mut out_phases = [99_i8; MAX_ATTENTION_HEADS];
    let res_none = fact_only_esc
        .try_escalate_horizon(&rep_req, &mut out_phases)
        .expect("fact-only repulsion escalation");
    assert!(res_none.is_none());
    assert_eq!(fact_only_esc.escalation_calls, 0);

    // 2. Corpus with Guard writes saturating anti-phase (`-R`) matching auto_repulsion contract.
    let mut guard_corpus = TextAttractorCorpus::new();
    guard_corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Guard,
            "Repulsive guard",
            &vec![guard_vec.as_slice(); num_layers],
            &refs,
            2.4,
        )
        .expect("ingest guard");
    let guard_q7 = guard_corpus.entries()[0]
        .per_layer_q7
        .last()
        .copied()
        .expect("guard q7");
    assert!(guard_q7.iter().any(|&v| v != 0));
    let mut expected_anti = [0_i8; MAX_ATTENTION_HEADS];
    for (dst, &src) in expected_anti.iter_mut().zip(guard_q7.iter()) {
        *dst = src.saturating_neg();
    }

    let mut guard_esc = CorpusRecallEscalationProvider::new(&guard_corpus);
    let mut esc_phases = [0_i8; MAX_ATTENTION_HEADS];
    let weight = guard_esc
        .try_escalate_horizon(&rep_req, &mut esc_phases)
        .expect("guard repulsion escalation")
        .expect("guard selected");
    assert!(weight >= 2.4 - 1e-5);
    assert_eq!(esc_phases, expected_anti);
    assert_eq!(guard_esc.escalation_calls, 1);

    // 3. Single-layer i8::MIN Guard pole saturates to +127 without overflow wraparound.
    let mut single_layer_geom = sample_gqa_partial_rope();
    single_layer_geom.num_layers = 1;
    single_layer_geom.layer_rope_mask = 1;
    let sl_adapter = ModelSteeringAdapter::new(single_layer_geom).expect("single-layer adapter");
    let mut opp_vec = vec![0.0_f32; q_len];
    for h in 0..num_heads {
        opp_vec[h * head_rot] = -1.0;
    }
    let mut sl_corpus = TextAttractorCorpus::new();
    sl_corpus
        .ingest_text_activations(
            &sl_adapter,
            AttractorRole::Guard,
            "Opposite pole guard",
            &[opp_vec.as_slice()],
            &[centroid.as_slice()],
            2.0,
        )
        .expect("ingest opposite guard");
    let sl_raw = sl_corpus.entries()[0].per_layer_q7[0];
    let mut sl_expected = [0_i8; MAX_ATTENTION_HEADS];
    for (dst, &src) in sl_expected.iter_mut().zip(sl_raw.iter()) {
        *dst = src.saturating_neg();
    }
    let mut sl_esc = CorpusRecallEscalationProvider::new(&sl_corpus);
    let mut sl_out = [0_i8; MAX_ATTENTION_HEADS];
    let _ = sl_esc
        .try_escalate_horizon(&rep_req, &mut sl_out)
        .expect("single-layer guard escalation")
        .expect("guard selected");
    assert_eq!(sl_out, sl_expected);
    assert!(sl_out.iter().all(|&v| v != i8::MIN));
}

#[test]
fn text_attractor_corpus_lazy_flat_cache_invalidation_clone_and_mock_provider() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<TextAttractorCorpus>();
    assert_send_sync::<CompiledTokenCorpus>();

    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let q_len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; q_len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let make_basis_vec = |slot_a: usize, w_a: f32, slot_b: usize, w_b: f32| -> Vec<f32> {
        let mut v = vec![0.0_f32; q_len];
        v[slot_a] = w_a;
        v[slot_b] = w_b;
        v
    };
    let f0 = make_basis_vec(0, 1.0, 2, 0.25);
    let f1 = make_basis_vec(0, 0.95, 4, 0.30);
    let f2 = make_basis_vec(0, 0.92, 6, 0.35);
    let f3 = make_basis_vec(1, 1.0, 8, 0.15);
    let p0 = make_basis_vec(10, 1.0, 12, 0.20);

    let mut corpus = TextAttractorCorpus::new();
    assert!(!corpus.is_recall_cache_populated());
    for (role, text, vec_q) in [
        (AttractorRole::Fact, "Hub 0", &f0),
        (AttractorRole::Fact, "Hub 1", &f1),
        (AttractorRole::Fact, "Hub 2", &f2),
        (AttractorRole::Fact, "Periph 3", &f3),
        (AttractorRole::Persona, "Persona 0", &p0),
    ] {
        corpus
            .ingest_text_activations(
                &adapter,
                role,
                text,
                &vec![vec_q.as_slice(); num_layers],
                &refs,
                1.5,
            )
            .expect("ingest");
    }
    assert!(!corpus.is_recall_cache_populated());

    let policy = RecallPolicy {
        min_coherence: 0.25,
        boundary_margin: 0.02,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    let query = make_basis_vec(0, 0.60, 1, 0.58);
    let q_slices: Vec<&[f32]> = vec![query.as_slice(); num_layers];

    // 1. First recall populates the flat cache; second recall reuses the exact allocation.
    let r1 = corpus
        .recall_for_query(&adapter, &q_slices, &refs, &policy)
        .expect("recall 1");
    assert!(corpus.is_recall_cache_populated());
    let ptr1 = corpus.recall_cache() as *const _;
    assert_eq!(
        corpus.recall_cache().projected_sigs_flat.len(),
        corpus.entries().len() * corpus.recall_cache().sig_dim
    );
    let r2 = corpus
        .recall_for_query(&adapter, &q_slices, &refs, &policy)
        .expect("recall 2");
    let ptr2 = corpus.recall_cache() as *const _;
    assert_eq!(ptr1, ptr2, "second recall must reuse OnceLock cache");
    assert_eq!(r1.top1_fact_score.to_bits(), r2.top1_fact_score.to_bits());
    assert_eq!(r1.top2_fact_score.to_bits(), r2.top2_fact_score.to_bits());
    assert_eq!(
        r1.min_local_lacuna_depth.to_bits(),
        r2.min_local_lacuna_depth.to_bits()
    );

    // 2. Cloned corpus starts with an empty lazy cache and rebuilds bit-identical results.
    let cloned = corpus.clone();
    assert!(!cloned.is_recall_cache_populated());
    let r_clone = cloned
        .recall_for_query(&adapter, &q_slices, &refs, &policy)
        .expect("cloned recall");
    assert!(cloned.is_recall_cache_populated());
    assert_eq!(
        r1.top1_fact_score.to_bits(),
        r_clone.top1_fact_score.to_bits()
    );
    assert_eq!(
        r1.top2_fact_score.to_bits(),
        r_clone.top2_fact_score.to_bits()
    );

    // 3. Every mutator invalidates the cache and subsequent recall rebuilds cleanly.
    corpus.fit_common_mode_subspace(2);
    assert!(!corpus.is_recall_cache_populated());
    let _ = corpus
        .recall_for_query(&adapter, &q_slices, &refs, &policy)
        .expect("recall after fit");
    assert!(corpus.is_recall_cache_populated());

    let extra_dir = vec![1.0_f32; corpus.entries()[0].residual_sig.len()];
    corpus.extend_common_mode_basis(&[extra_dir], 4);
    assert!(!corpus.is_recall_cache_populated());
    let _ = corpus
        .recall_for_query(&adapter, &q_slices, &refs, &policy)
        .expect("recall after extend");
    assert!(corpus.is_recall_cache_populated());

    let extra_fact = make_basis_vec(20, 1.0, 22, 0.25);
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Extra fact",
            &vec![extra_fact.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("ingest extra");
    assert!(!corpus.is_recall_cache_populated());
    let _ = corpus
        .recall_for_query(&adapter, &q_slices, &refs, &policy)
        .expect("recall after ingest");
    assert!(corpus.is_recall_cache_populated());
    assert_eq!(
        corpus
            .recall_cache()
            .role_indices(AttractorRole::Fact)
            .len(),
        5
    );

    // 4. Mock NeighborhoodStatsProvider swaps cleanly into StratumContext & PerspectiveHubnessStratum.
    struct MockStatsProvider {
        fact_idxs: Vec<usize>,
        persona_idxs: Vec<usize>,
        guard_idxs: Vec<usize>,
        per_entry: Vec<EntryNeighborhoodStats>,
        global_fact_mean: f32,
    }
    impl NeighborhoodStatsProvider for MockStatsProvider {
        fn entry_stats(&self, entry_idx: usize) -> EntryNeighborhoodStats {
            self.per_entry.get(entry_idx).copied().unwrap_or_default()
        }
        fn role_global_mean(&self, role: AttractorRole) -> f32 {
            match role {
                AttractorRole::Fact => self.global_fact_mean,
                AttractorRole::Persona | AttractorRole::Guard => 0.0,
            }
        }
        fn role_indices(&self, role: AttractorRole) -> &[usize] {
            match role {
                AttractorRole::Fact => &self.fact_idxs,
                AttractorRole::Persona => &self.persona_idxs,
                AttractorRole::Guard => &self.guard_idxs,
            }
        }
    }

    let entries = corpus.entries();
    let mock = MockStatsProvider {
        fact_idxs: vec![0, 1, 2, 3, 5],
        persona_idxs: vec![4],
        guard_idxs: vec![],
        per_entry: vec![
            EntryNeighborhoodStats {
                neighbor_mean_cosine: 0.90,
                neighbor_std_cosine: 0.10,
            },
            EntryNeighborhoodStats {
                neighbor_mean_cosine: 0.10,
                neighbor_std_cosine: 0.10,
            },
            EntryNeighborhoodStats::default(),
            EntryNeighborhoodStats::default(),
            EntryNeighborhoodStats::default(),
            EntryNeighborhoodStats::default(),
        ],
        global_fact_mean: 0.50,
    };
    let ctx = StratumContext {
        entries,
        raw_sig: &[],
        query_sig: &[],
        projected_sigs_flat: &[],
        projected_sig_dim: 0,
        neighborhood_stats: &mock,
        layer_chunk_dim: q_len,
        num_q_heads: adapter.geometry().num_q_heads,
        rotary_dim: adapter.geometry().rotary_dim,
        policy: &policy,
        coalition_thresholds: CoalitionThresholds::default(),
        query_seq_len: 0,
        query_ortho_ratio: 1.0,
        u_foreign: RecallEnvelopeBounds::standard().u_foreign_ceiling,
    };
    let mut signal = StratumSignal {
        candidate_scores: vec![0.50; entries.len()],
        local_token_scores: Vec::new(),
        q_to_d_scores: Vec::new(),
        d_to_q_scores: Vec::new(),
        token_deposits: Vec::new(),
        linked_fact_indices: Vec::new(),
        decomposition_triggered: false,
        local_lacuna_depths: vec![0.0; entries.len()],
        stigmergic_weights: vec![1.0; entries.len()],
        boundary_metrics: RecallBoundaryMetrics {
            top1_fact_score: -1.0,
            top2_fact_score: -1.0,
            top1_persona_score: -1.0,
            max_guard_score: -1.0,
            guard_conflict_margin: 0.0,
        },
        gate_decision: RecallGateDecision {
            lacuna_state: LacunaBoundaryState::LiminalExterior,
            suppress_fact_injection: true,
        },
        action_decision: StigmergicActionDecision::PassiveSidecar,
        gateway_frame: GatewayFrame::default(),
        swarm_report: Default::default(),
        local_channel_preferred: false,
        telemetry: StratumProbeTelemetry::default(),
    };
    let perspective = AsymmetricHubnessPerspective::default();
    let stratum = PerspectiveHubnessStratum {
        perspective: &perspective,
    };
    stratum
        .infer_up(&ctx, &mut signal)
        .expect("mock provider infer_up");
    // Entry 0 has high hubness (0.90 > 0.50) so it is damped; Entry 1 has low hubness (0.10 < 0.50) so it wins.
    assert!(
        signal.candidate_scores[1] > signal.candidate_scores[0] + 0.30,
        "expected mock hubness stats to damp entry 0 ({}) below entry 1 ({})",
        signal.candidate_scores[0],
        signal.candidate_scores[1]
    );
}

#[test]
fn zero_copy_common_mode_subspace_matches_explicit_deflation() {
    fn explicit_deflated_reference(sigs: &[Vec<f32>], max_components: usize) -> Vec<Vec<f32>> {
        let mut basis: Vec<Vec<f32>> = Vec::new();
        if sigs.len() < 4 || max_components == 0 {
            return basis;
        }
        let dim = sigs[0].len();
        let rank_cap = max_components.min(sigs.len() / 3);
        let mut work = sigs.to_vec();
        let normalize = |v: &mut [f32]| {
            let n = v
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                .sqrt();
            if n > 1e-12 {
                let inv = (1.0 / n) as f32;
                for x in v.iter_mut() {
                    *x *= inv;
                }
            }
        };
        for _ in 0..rank_cap {
            let mut v = vec![0.0_f32; dim];
            for w in &work {
                for (d, &s) in v.iter_mut().zip(w.iter()) {
                    *d += s;
                }
            }
            for dir in &basis {
                let dot: f64 = v
                    .iter()
                    .zip(dir.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let d = dot as f32;
                for (dst, &c) in v.iter_mut().zip(dir.iter()) {
                    *dst -= d * c;
                }
            }
            normalize(&mut v);
            if v.iter().all(|&x| x == 0.0) {
                break;
            }
            for _ in 0..8 {
                let mut next = vec![0.0_f32; dim];
                for w in &work {
                    let dot: f64 = w
                        .iter()
                        .zip(v.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let d = dot as f32;
                    for (dst, &src) in next.iter_mut().zip(w.iter()) {
                        *dst += d * src;
                    }
                }
                for dir in &basis {
                    let dot: f64 = next
                        .iter()
                        .zip(dir.iter())
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    let d = dot as f32;
                    for (dst, &c) in next.iter_mut().zip(dir.iter()) {
                        *dst -= d * c;
                    }
                }
                let norm_sq: f64 = next.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
                if norm_sq < 1e-12 {
                    break;
                }
                normalize(&mut next);
                v = next;
            }
            let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
            if norm_sq < 0.5 {
                break;
            }
            for w in &mut work {
                let dot: f64 = w
                    .iter()
                    .zip(v.iter())
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let d = dot as f32;
                for (dst, &c) in w.iter_mut().zip(v.iter()) {
                    *dst -= d * c;
                }
            }
            basis.push(v);
        }
        basis
    }

    let dim = 24;
    let sigs: Vec<Vec<f32>> = (0..12)
        .map(|i| {
            let mut v = vec![0.0_f32; dim];
            v[0] = 0.80 + 0.02 * (i as f32);
            v[1] = if i % 2 == 0 { 0.55 } else { -0.35 };
            v[2 + (i % 8)] = 0.45;
            v[12 + (i % 6)] = -0.25;
            v
        })
        .collect();
    let sig_refs: Vec<&[f32]> = sigs.iter().map(Vec::as_slice).collect();
    let zero_copy = TextAttractorCorpus::fit_common_mode_subspace_from_sigs(&sig_refs, 3);
    let reference = explicit_deflated_reference(&sigs, 3);

    assert_eq!(zero_copy.len(), reference.len());
    assert_eq!(zero_copy.len(), 3);
    for (k, (zc, rf)) in zero_copy.iter().zip(reference.iter()).enumerate() {
        let cos = TextAttractorCorpus::cosine_f32(zc, rf).abs();
        assert!(
            cos > 0.99999,
            "basis component {k} diverged from explicit deflation: |cos|={cos}"
        );
    }
}
