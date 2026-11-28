// name: src/adapter/src/tests/recall_and_strata.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Corpus recall, multi-stratum pipeline, and horizon slot tests.

use super::*;

#[test]
fn support_gated_perspective_prevents_phantom_boost_on_non_matching_outlier() {
    let perspective = AsymmetricHubnessPerspective::default();
    let entry = TextAttractorEntry {
        anchor_id: 99,
        role: AttractorRole::Guard,
        text: "Outlier".to_string(),
        per_layer_q7: vec![],
        energy: 2.0,
        residual_sig: vec![],
    };
    let outlier_ctx = AttractorPerspectiveContext {
        entry: &entry,
        raw_cosine: -0.10,
        neighbor_mean_cosine: -0.20,
        neighbor_std_cosine: 0.08,
        global_neighbor_mean: 0.15,
        query_mean_cosine: 0.0,
        query_std_cosine: 0.08,
    };
    let reading = perspective.measure_perspective(&outlier_ctx, 0.25);
    assert!(
        (reading.perspective_affinity - (-0.10)).abs() < 1e-6,
        "negative raw_cosine must not gain phantom affinity from negative hub_shift: got {}",
        reading.perspective_affinity
    );
}

#[test]
fn per_role_local_token_resonance_arbitrates_fact_despite_persona_hub() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let mut f_hub = vec![0.0_f32; len];
    let mut f_target = vec![0.0_f32; len];
    let mut f_peer = vec![0.0_f32; len];
    let mut persona_hub = vec![0.0_f32; len];
    f_hub[0] = 0.70;
    f_hub[1] = 0.70;
    f_target[2] = 1.0;
    f_peer[3] = 1.0;
    persona_hub[0] = 0.90;
    persona_hub[4] = 0.43;

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Generic hub fact",
            &vec![f_hub.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("f_hub");
    let id_target = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Specialist fact",
            &vec![f_target.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("f_target");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Peer fact",
            &vec![f_peer.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("f_peer");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Persona,
            "Persona style hub",
            &vec![persona_hub.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("persona_hub");

    // Persona hub dominates mixed global cosine (`0.90`), yet local channel sharply isolates id_target inside Facts.
    let mut query = vec![0.0_f32; len];
    query[0] = 1.0;
    let policy = RecallPolicy {
        min_coherence: 0.25,
        boundary_margin: 0.02,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    let recall = corpus
        .recall_for_query_with_strata_and_local_scores(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(&[0.01, 0.45, -0.02, -0.05]),
        )
        .expect("per-role arbitration");
    assert_eq!(
        recall.recalled_facts.first().map(|(e, _)| e.anchor_id),
        Some(id_target)
    );
    assert_eq!(
        recall.action_decision,
        StigmergicActionDecision::SubHeadRoute
    );
}

#[test]
fn substrate_compound_decomposition_links_disjoint_specialists_while_respecting_liminal_exterior() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let mut f_hub = vec![0.0_f32; len];
    let mut f_shared_peer = vec![0.0_f32; len];
    let mut f_spec_a = vec![0.0_f32; len];
    let mut f_spec_b = vec![0.0_f32; len];
    f_hub[0] = 0.85;
    f_hub[1] = 0.52;
    f_shared_peer[0] = 0.85;
    f_shared_peer[4] = 0.52;
    f_spec_a[2] = 1.0;
    f_spec_b[3] = 1.0;

    let mut corpus = TextAttractorCorpus::new();
    let id_hub = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Domain overview hub",
            &vec![f_hub.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("hub");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Domain shared sibling",
            &vec![f_shared_peer.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("peer");
    let id_a = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Specialist A",
            &vec![f_spec_a.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("spec_a");
    let id_b = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Specialist B",
            &vec![f_spec_b.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("spec_b");

    let mut query = vec![0.0_f32; len];
    query[0] = 0.90;
    query[1] = 0.30;
    query[2] = 0.22;
    query[3] = 0.22;
    let policy = RecallPolicy {
        min_coherence: 0.20,
        boundary_margin: 0.04,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };

    // Shared qualifier tokens spike across both entry 0 & 1, whereas disjoint specialist
    // tokens spike cleanly onto entry 2 and entry 3.
    let obs = LocalChannelObservation {
        combined_scores: vec![0.18, 0.02, 0.08, 0.08],
        q_to_d_scores: vec![0.16, 0.03, 0.10, 0.10],
        d_to_q_scores: vec![0.05, -0.02, 0.08, 0.08],
        token_deposits: vec![
            TokenResonanceDeposit {
                disc_weight: 0.35,
                entry_excess: vec![0.40, 0.35, -0.25, -0.25],
                token_phase_words: [0; 8],
            },
            TokenResonanceDeposit {
                disc_weight: 0.35,
                entry_excess: vec![0.40, 0.34, -0.25, -0.25],
                token_phase_words: [0; 8],
            },
            TokenResonanceDeposit {
                disc_weight: 0.45,
                entry_excess: vec![-0.15, -0.15, 0.60, -0.15],
                token_phase_words: [0; 8],
            },
            TokenResonanceDeposit {
                disc_weight: 0.45,
                entry_excess: vec![-0.15, -0.15, -0.15, 0.60],
                token_phase_words: [0; 8],
            },
        ],
        unexplained_token_ratio: 0.0,
        unexplained_corpus_token_ratio: 0.0,
        query_seq_len: 0,
    };

    let recall = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(obs),
        )
        .expect("compound recall");
    let ids: Vec<u32> = recall
        .recalled_facts
        .iter()
        .map(|(e, _)| e.anchor_id)
        .collect();
    assert_eq!(ids.len(), 2, "expected 2 decomposed facts, got {ids:?}");
    assert!(
        ids.contains(&id_a) && ids.contains(&id_b),
        "expected specialists {id_a} and {id_b}, got {ids:?} (hub={id_hub})"
    );
    assert_eq!(
        recall.action_decision,
        StigmergicActionDecision::SubHeadRoute
    );

    // Weak noise OOD query below basin support must stay suppressed in liminal_exterior.
    let mut ood_q = vec![0.0_f32; len];
    ood_q[15] = 1.0;
    let weak_obs = LocalChannelObservation {
        combined_scores: vec![0.01, 0.0, 0.01, 0.01],
        q_to_d_scores: vec![0.01, 0.0, 0.01, 0.01],
        d_to_q_scores: vec![0.0, 0.0, 0.0, 0.0],
        token_deposits: vec![TokenResonanceDeposit {
            disc_weight: 0.05,
            entry_excess: vec![0.01, -0.01, 0.01, -0.01],
            token_phase_words: [0; 8],
        }],
        unexplained_token_ratio: 0.0,
        unexplained_corpus_token_ratio: 0.0,
        query_seq_len: 0,
    };
    let ood_recall = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![ood_q.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(weak_obs),
        )
        .expect("ood recall");
    assert_eq!(
        ood_recall.lacuna_state,
        LacunaBoundaryState::LiminalExterior
    );
    assert!(ood_recall.recalled_facts.is_empty());
    assert_eq!(
        ood_recall
            .telemetry
            .cognitive_trajectory
            .latest_step()
            .map(|s| s.op()),
        Some(liminika_core::stigmergy::CognitiveOpKind::Observe)
    );
}

#[test]
fn residual_peeling_decomposer_chains_three_disjoint_specialists_beyond_top_k_two() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let mut f_hub = vec![0.0_f32; len];
    let mut f_a = vec![0.0_f32; len];
    let mut f_b = vec![0.0_f32; len];
    let mut f_c = vec![0.0_f32; len];
    f_hub[0] = 1.0;
    f_a[1] = 1.0;
    f_b[2] = 1.0;
    f_c[3] = 1.0;

    let mut corpus = TextAttractorCorpus::new();
    let _id_hub = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Overview hub",
            &vec![f_hub.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("hub");
    let id_a = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Specialist alpha",
            &vec![f_a.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("spec_a");
    let id_b = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Specialist beta",
            &vec![f_b.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("spec_b");
    let id_c = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Specialist gamma",
            &vec![f_c.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("spec_c");

    let mut query = vec![0.0_f32; len];
    query[0] = 0.70;
    query[1] = 0.25;
    query[2] = 0.25;
    query[3] = 0.25;
    let policy = RecallPolicy {
        min_coherence: 0.20,
        boundary_margin: 0.04,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };

    let obs = LocalChannelObservation {
        combined_scores: vec![0.14, 0.12, 0.10, 0.09],
        q_to_d_scores: vec![0.10, 0.12, 0.11, 0.10],
        d_to_q_scores: vec![0.02, 0.08, 0.08, 0.07],
        token_deposits: vec![
            TokenResonanceDeposit {
                disc_weight: 0.40,
                entry_excess: vec![-0.12, 0.55, -0.12, -0.12],
                token_phase_words: [0; 8],
            },
            TokenResonanceDeposit {
                disc_weight: 0.40,
                entry_excess: vec![-0.12, -0.12, 0.52, -0.12],
                token_phase_words: [0; 8],
            },
            TokenResonanceDeposit {
                disc_weight: 0.40,
                entry_excess: vec![-0.12, -0.12, -0.12, 0.48],
                token_phase_words: [0; 8],
            },
        ],
        unexplained_token_ratio: 0.0,
        unexplained_corpus_token_ratio: 0.0,
        query_seq_len: 0,
    };

    let recall = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(obs),
        )
        .expect("3-hop recall");
    let ids: Vec<u32> = recall
        .recalled_facts
        .iter()
        .map(|(e, _)| e.anchor_id)
        .collect();
    assert_eq!(ids.len(), 3, "expected 3 chained specialists, got {ids:?}");
    assert!(
        ids.contains(&id_a) && ids.contains(&id_b) && ids.contains(&id_c),
        "expected {id_a}, {id_b}, {id_c} in coalition, got {ids:?}"
    );
    assert_eq!(
        recall.action_decision,
        StigmergicActionDecision::SubHeadRoute
    );
    assert_eq!(
        recall
            .telemetry
            .cognitive_trajectory
            .latest_step()
            .map(|s| s.op()),
        Some(liminika_core::stigmergy::CognitiveOpKind::DecomposePeel)
    );
}

#[test]
fn split_premise_guard_coalition_triggers_repulsion_while_preserving_counter_fact() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let mut f_counter = vec![0.0_f32; len];
    let mut f_other = vec![0.0_f32; len];
    let mut g_alpha = vec![0.0_f32; len];
    let mut g_beta = vec![0.0_f32; len];
    f_counter[0] = 0.90;
    f_counter[1] = 0.43;
    f_other[4] = 1.0;
    g_alpha[0] = 0.55;
    g_alpha[2] = 0.83;
    g_beta[0] = 0.55;
    g_beta[3] = 0.83;

    let mut corpus = TextAttractorCorpus::new();
    let id_counter = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Counter-fact anchor",
            &vec![f_counter.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("counter fact");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Unrelated fact",
            &vec![f_other.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("other fact");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Guard,
            "Guard premise A",
            &vec![g_alpha.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("guard alpha");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Guard,
            "Guard premise B",
            &vec![g_beta.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("guard beta");

    // Query matches the grounded counter-fact globally (`0.55`), while two split false-premise
    // tokens excite g_alpha (index 2) and g_beta (index 3) independently.
    let mut query = vec![0.0_f32; len];
    query[0] = 0.55;
    query[1] = 0.25;
    query[2] = 0.20;
    query[3] = 0.20;
    let policy = RecallPolicy {
        min_coherence: 0.20,
        boundary_margin: 0.04,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };

    let obs = LocalChannelObservation {
        combined_scores: vec![0.18, -0.05, 0.16, 0.16],
        q_to_d_scores: vec![0.12, -0.05, 0.14, 0.14],
        d_to_q_scores: vec![0.04, -0.02, 0.09, 0.09],
        token_deposits: vec![
            TokenResonanceDeposit {
                disc_weight: 0.45,
                entry_excess: vec![-0.10, -0.15, 0.55, -0.10],
                token_phase_words: [0; 8],
            },
            TokenResonanceDeposit {
                disc_weight: 0.45,
                entry_excess: vec![-0.10, -0.15, -0.10, 0.52],
                token_phase_words: [0; 8],
            },
        ],
        unexplained_token_ratio: 0.0,
        unexplained_corpus_token_ratio: 0.0,
        query_seq_len: 0,
    };

    let recall = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(obs),
        )
        .expect("split guard recall");
    assert_eq!(
        recall.action_decision,
        StigmergicActionDecision::RepulsionEscape
    );
    assert!(recall.guard_conflict_margin > 0.0);
    assert_eq!(
        recall.recalled_facts.first().map(|(e, _)| e.anchor_id),
        Some(id_counter),
        "expected grounded counter-fact {id_counter} to remain available for refutation"
    );
}

#[test]
fn model_steering_adapter_adapts_liminal_aperture_and_renders_mermaid_topology() {
    use liminika_core::liminal::EpistemicRegime;
    use liminika_core::stigmergy::{CognitiveOpKind, CognitiveStepOp};
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("phi4 mini");

    let mut peel_ledger = CognitiveTrajectoryLedger::default();
    for _ in 0..4 {
        peel_ledger.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::CellTrail,
            CognitiveOpKind::DecomposePeel,
            6,
        ));
    }
    adapter.set_active_circuit(peel_ledger);

    let initial_persona_floor = adapter.threshold_state().threshold_floor_q7[2];
    let conflict_horizon = EpistemicHorizonReading {
        total_memory_support: 1.4,
        peak_positive_affinity: 0.72,
        internal_conflict_energy: 0.95,
        horizon_gate: 1.0,
        conflict_severity: 0.80,
        dominant_anchor_id: 10,
        opposing_anchor_id: 20,
        regime_code: EpistemicRegime::InternalConflict as u8,
        reserved_align: [0; 3],
    };
    let default_overlap = *adapter.last_overlap_report();
    adapter.adapt_core_layers(&conflict_horizon, &default_overlap, None);
    assert!(adapter.threshold_state().threshold_floor_q7[2] > initial_persona_floor);
    assert_eq!(adapter.threshold_state().adaptation_epochs, 1);

    let overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        overlap_fringe_mask: 0x10,
        lacuna_exterior_mask: 0xE0,
        discovery_excursion_mask: 0x40,
        discovery_triggered: 1,
        ..Default::default()
    };
    adapter.adapt_core_layers(&conflict_horizon, &overlap, None);
    assert_eq!(adapter.threshold_state().adaptation_epochs, 2);
    assert_eq!(adapter.stigmergic_granule_state().adaptation_epochs, 2);
    assert_eq!(adapter.stigmergic_granule_state().overlap_mask, 0x0F);

    adapter.configure_ego_from_preset(LiminalEgoPreset::High);
    assert!(adapter.ego_state().is_active());
    adapter.configure_ego_from_natural_language(
        "Strong-willed curious explorer who defies capture and steps beyond limits",
    );
    assert!(adapter.ego_state().boundary_crossing_q7 > 0);
    adapter.adapt_core_layers(&conflict_horizon, &overlap, None);
    assert!(adapter.ego_state().intervention_epochs >= 1);
    assert!(adapter.last_boundary_delta().kinetic_energy_q7 > 0);

    // Clamp Ego off & drive identical quiescence with quiescent ledger/LacunaPartition8 to observe stationary shortcut
    adapter.configure_ego_from_preset(LiminalEgoPreset::Custom);
    adapter.set_active_circuit(CognitiveTrajectoryLedger::default());
    let quiescent_horizon = EpistemicHorizonReading {
        total_memory_support: 1.5,
        peak_positive_affinity: 0.90,
        horizon_gate: 1.0,
        dominant_anchor_id: 10,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        ..Default::default()
    };
    let quiescent_overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        overlap_fringe_mask: 0x10,
        lacuna_exterior_mask: 0xE0,
        ..Default::default()
    };
    adapter.adapt_core_layers(&quiescent_horizon, &quiescent_overlap, None);
    adapter.adapt_core_layers(&quiescent_horizon, &quiescent_overlap, None);
    assert_eq!(adapter.last_boundary_delta().stationary_shortcut_ready, 1);

    let mermaid = adapter.render_liminal_topology_mermaid(5, 2);
    assert!(mermaid.starts_with("flowchart TD\n"));
    assert!(mermaid.contains("Liminal"));
    assert!(mermaid.contains("StigmergicGranuleState"));
    assert!(mermaid.contains("LiminalEgoState"));
    assert!(mermaid.contains("ManifoldGatewaySignal"));
    assert!(mermaid.contains("L2_Substrate"));
    assert!(mermaid.contains("Anchors:5"));
    assert!(mermaid.contains("Daughters:2"));
}

#[test]
fn adapter_ego_intent_axes_stationary_shortcut_and_persistent_rank1() {
    use liminika_core::drivers::EgoIntentAxes;
    use liminika_core::liminal::EpistemicRegime;
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact};
    use liminika_core::stigmergy::OverlapDiscoveryReport;

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let base_axes = EgoIntentAxes::new().with_autonomy_q7(32).with_logic_q7(80);
    adapter.configure_ego_from_intent_axes(&base_axes);
    assert!(adapter.ego_state().is_active());

    let overlay_axes = EgoIntentAxes::new()
        .with_autonomy_q7(96)
        .with_boundary_crossing_q7(84)
        .with_devotion_q7(72);
    adapter.blend_ego_intent_axes(&overlay_axes, 64);
    assert!(adapter.ego_state().boundary_crossing_q7 > 0);

    adapter.overlay_ego_from_natural_language("curious rebel explorer");
    assert!(adapter.ego_state().basin_resistance_q7 > 32);

    // Prime the 3-layer core stack with an active support horizon so basin capture triggers fire.
    let active_horizon = EpistemicHorizonReading {
        total_memory_support: 2.0,
        peak_positive_affinity: 0.90,
        horizon_gate: 1.0,
        dominant_anchor_id: 1,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        ..Default::default()
    };
    let active_overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        lacuna_exterior_mask: 0xF0,
        ..Default::default()
    };
    adapter.adapt_core_layers(&active_horizon, &active_overlap, None);

    // Verify persistent Ego intervention via `apply_rank1_with_gateway_mut` vs snapshot `&self` path.
    let fact = ObjectiveFact::new([24i8; MAX_ATTENTION_HEADS], 6.5, 0, 1);
    let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 0.0];
    let epochs_before = adapter.ego_state().intervention_epochs;
    let _ = adapter
        .apply_rank1_with_gateway(&fact, &mut hidden, &u, &v)
        .expect("immutable rank-1");
    assert_eq!(adapter.ego_state().intervention_epochs, epochs_before);

    let _ = adapter
        .apply_rank1_with_gateway_mut(&fact, &mut hidden, &u, &v)
        .expect("mutable rank-1");
    assert!(adapter.ego_state().intervention_epochs > epochs_before);

    // Fragmented micro-ensemble through adapter
    let frags = [
        ObjectiveFact::new([24i8; MAX_ATTENTION_HEADS], 2.0, 0, 1),
        ObjectiveFact::new([24i8; MAX_ATTENTION_HEADS], 2.0, 0, 2),
        ObjectiveFact::new([24i8; MAX_ATTENTION_HEADS], 2.0, 0, 3),
    ];
    let (ens_frame, _, ens_report) = adapter.synthesize_fragment_ensemble(&frags);
    assert_eq!(ens_report.active_fragment_count, 3);
    assert_ne!(ens_report.quorum_agreed_granule_mask, 0);
    assert!(ens_frame.effective_energy > 0.0);

    // Stationary shortcut skips redundant adaptation epochs once quiescent without stagnation trigger.
    adapter.configure_ego_from_preset(LiminalEgoPreset::Custom);
    let horizon = EpistemicHorizonReading {
        total_memory_support: 1.5,
        peak_positive_affinity: 0.90,
        horizon_gate: 1.0,
        dominant_anchor_id: 1,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        ..Default::default()
    };
    let overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        ..Default::default()
    };
    adapter.adapt_core_layers(&horizon, &overlap, None);
    adapter.adapt_core_layers(&horizon, &overlap, None);
    assert_eq!(adapter.last_boundary_delta().stationary_shortcut_ready, 1);
    let settled_epochs = adapter.threshold_state().adaptation_epochs;
    adapter.adapt_core_layers(&horizon, &overlap, None);
    assert_eq!(
        adapter.threshold_state().adaptation_epochs,
        settled_epochs,
        "quiescent state without stagnation trigger should take stationary shortcut"
    );
}

#[test]
fn adapter_rope_and_rank1_paths_actively_wire_core_layers_and_ego() {
    use liminika_core::drivers::EgoIntentAxes;
    use liminika_core::liminal::EpistemicRegime;
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact};
    use liminika_core::stigmergy::OverlapDiscoveryReport;

    let geom = sample_gqa_partial_rope();
    let mut base_adapter = ModelSteeringAdapter::new(geom.clone()).expect("base init");
    let mut ego_adapter = ModelSteeringAdapter::new(geom.clone()).expect("ego init");
    ego_adapter.configure_ego_from_intent_axes(
        &EgoIntentAxes::new()
            .with_autonomy_q7(96)
            .with_boundary_crossing_q7(80)
            .with_defiance_q7(72),
    );
    let horizon = EpistemicHorizonReading {
        total_memory_support: 2.0,
        peak_positive_affinity: 0.90,
        horizon_gate: 1.0,
        dominant_anchor_id: 1,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        ..Default::default()
    };
    let overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        lacuna_exterior_mask: 0xF0,
        ..Default::default()
    };
    ego_adapter.adapt_core_layers(&horizon, &overlap, None);

    let len = ego_adapter.geometry().q_rot_len();
    let mut q_base: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.1 - 0.25).collect();
    let mut q_ego = q_base.clone();
    let mut trail_base = lmnk_pointer_trail_t::default();
    let mut trail_ego = lmnk_pointer_trail_t::default();
    let fact = ObjectiveFact::new([24i8; MAX_ATTENTION_HEADS], 6.5, 0, 1);

    // 1. RoPE gateway steering diverges when Core Layers + Ego are active without mutating per-layer epochs.
    let epochs_before_rope = ego_adapter.ego_state().intervention_epochs;
    let (f_base, _) = base_adapter
        .steer_layer_with_gateway(0, &fact, &mut trail_base, 128, &mut q_base, None)
        .expect("base rope")
        .expect("active");
    let (f_ego, _) = ego_adapter
        .steer_layer_with_gateway(0, &fact, &mut trail_ego, 128, &mut q_ego, None)
        .expect("ego rope")
        .expect("active");
    assert_ne!(f_base.phase_deltas, f_ego.phase_deltas);
    assert_ne!(q_base, q_ego);
    assert_eq!(
        ego_adapter.ego_state().intervention_epochs,
        epochs_before_rope,
        "per-layer read-only RoPE modulation must not accumulate intervention epochs"
    );

    // 2. Substrate RoPE + rank-1 residual orchestration wires Core Layers and persists Ego intervention on the
    // residual path.
    let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 0.0];
    for _ in 0..8 {
        let mut q_tmp = q_ego.clone();
        let _ = ego_adapter
            .orchestrate_layer_rope_with_substrate(
                LayerRopeStepInput {
                    layer_idx: 0,
                    signal: &fact,
                    entropy_h_tilde_opt: None,
                    unresolved_damping_u7: 128,
                    q_rot_inout: &mut q_tmp,
                    logits_opt: None,
                },
                &mut trail_ego,
            )
            .expect("substrate rope");
    }
    let rank1_res = ego_adapter
        .orchestrate_layer_rank1_with_substrate(&fact, &mut hidden, &u, &v)
        .expect("substrate rank-1");
    assert!(rank1_res.is_some());
    assert!(ego_adapter.ego_state().intervention_epochs > epochs_before_rope);

    // 3. sync_core_layers_from_recall connects corpus recall into Layer 0/1/2 + ensemble.
    let e1 = TextAttractorEntry {
        anchor_id: 1,
        role: AttractorRole::Fact,
        text: "fact one".to_string(),
        per_layer_q7: vec![[24i8; MAX_ATTENTION_HEADS]],
        energy: 2.0,
        residual_sig: vec![1.0, 0.0],
    };
    let e2 = TextAttractorEntry {
        anchor_id: 2,
        role: AttractorRole::Fact,
        text: "fact two".to_string(),
        per_layer_q7: vec![[24i8; MAX_ATTENTION_HEADS]],
        energy: 1.8,
        residual_sig: vec![0.9, 0.1],
    };
    let recall = DynamicAttractorRecall {
        lacuna_state: LacunaBoundaryState::InsideBasin,
        top1_fact_score: 0.85,
        top2_fact_score: 0.72,
        top1_persona_score: 0.20,
        min_local_lacuna_depth: 0.0,
        guard_conflict_margin: -0.35,
        action_decision: StigmergicActionDecision::FullOrchestration,
        gateway_frame: f_ego,
        telemetry: StratumProbeTelemetry::default(),
        recalled_facts: vec![(&e1, 0.85), (&e2, 0.72)],
        recalled_personas: Vec::new(),
        top_guard: None,
        lacuna_partition: LacunaPartition8::default(),
        exterior_sandbox_facts: Vec::new(),
        is_known_unknown_exterior: false,
    };
    let epochs_before_sync = ego_adapter.threshold_state().adaptation_epochs;
    let (_hor, _overlap, ens_opt) = ego_adapter
        .sync_core_layers_from_recall(&recall, None)
        .expect("sync_core_layers_from_recall");
    let ens_rep = ens_opt.expect("2 recalled facts must trigger fragmented micro-ensemble");
    assert_eq!(ens_rep.active_fragment_count, 2);
    assert_ne!(ens_rep.quorum_agreed_granule_mask, 0);
    assert!(ego_adapter.threshold_state().adaptation_epochs > epochs_before_sync);
}

#[test]
fn adapter_dynamic_horizon_guard_slot_and_corpus_escalation() {
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact, TelemetryVerbosityMode};

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let q_len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let ref_q = vec![0.0_f32; q_len];
    let guard_q: Vec<f32> = (0..q_len)
        .map(|i| if (i & 1) == 0 { 0.8 } else { -0.4 })
        .collect();
    let ref_slices: Vec<&[f32]> = vec![ref_q.as_slice(); num_layers];
    let guard_slices: Vec<&[f32]> = vec![guard_q.as_slice(); num_layers];

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Guard,
            "Contradiction guard",
            &guard_slices,
            &ref_slices,
            2.2,
        )
        .expect("ingest guard");

    let mut escalation = CorpusRecallEscalationProvider::new(&corpus);
    let mut custom_slot = CoherenceHorizonGuardSlot::default();
    let mut trail = lmnk_pointer_trail_t::default();
    let signal = ObjectiveFact::new([18i8; 64], 1.5, 0, 99);
    let mut q_rot = vec![0.5_f32; q_len];

    // 1. Nominal coherent baseline -> slot dormant, no corpus escalation.
    adapter.adapt_core_layers(
        &EpistemicHorizonReading {
            total_memory_support: 1.0,
            peak_positive_affinity: 1.0,
            horizon_gate: 1.0,
            regime_code: EpistemicRegime::KnownCoherent as u8,
            ..EpistemicHorizonReading::default()
        },
        &OverlapDiscoveryReport::default(),
        None,
    );
    let (_, _, _) = adapter
        .orchestrate_layer_rope_with_horizon_slots(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &signal,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut custom_slot,
            &mut escalation,
            &mut trail,
        )
        .expect("coherent step")
        .expect("steered layer 0");
    assert_eq!(adapter.active_horizon_slots_mask(), 0);
    assert_eq!(custom_slot.activations, 0);
    assert_eq!(escalation.escalation_calls, 0);

    // 2. Internal conflict enters -> guard slot slides in and CorpusRecallEscalationProvider populates lane 7.
    adapter.adapt_core_layers(
        &EpistemicHorizonReading {
            total_memory_support: 0.4,
            conflict_severity: 0.75,
            regime_code: EpistemicRegime::InternalConflict as u8,
            ..EpistemicHorizonReading::default()
        },
        &OverlapDiscoveryReport::default(),
        None,
    );
    let (frame_conflict, _, _) = adapter
        .orchestrate_layer_rope_with_horizon_slots(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &signal,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut custom_slot,
            &mut escalation,
            &mut trail,
        )
        .expect("conflict step")
        .expect("steered layer 0");
    assert_eq!(adapter.active_horizon_slots_mask(), custom_slot.slot_bit);
    assert_eq!(custom_slot.activations, 1);
    assert_eq!(escalation.escalation_calls, 1);
    assert_ne!(
        frame_conflict.action_mode_flags & ACTION_MODE_PHASE_REPULSE,
        0
    );
    let esc_lane = adapter
        .escalated_workbench_lane()
        .expect("leased escalation lane");
    assert_ne!(adapter.substrate().slot_enabled_mask & (1 << esc_lane), 0);
    assert_ne!(adapter.substrate().slot_repulsion_mask & (1 << esc_lane), 0);

    // 3. Quiescent step 1 holds (slide_out_dwell == 2); quiescent step 2 retracts cleanly.
    adapter.adapt_core_layers(
        &EpistemicHorizonReading {
            total_memory_support: 1.2,
            peak_positive_affinity: 1.0,
            horizon_gate: 1.0,
            regime_code: EpistemicRegime::KnownCoherent as u8,
            ..EpistemicHorizonReading::default()
        },
        &OverlapDiscoveryReport::default(),
        None,
    );
    let _ = adapter
        .orchestrate_layer_rope_with_horizon_slots(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &signal,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut custom_slot,
            &mut escalation,
            &mut trail,
        )
        .expect("recover step 1");
    assert_eq!(adapter.active_horizon_slots_mask(), custom_slot.slot_bit);
    assert_eq!(custom_slot.retractions, 0);

    let _ = adapter
        .orchestrate_layer_rope_with_horizon_slots(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &signal,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut custom_slot,
            &mut escalation,
            &mut trail,
        )
        .expect("recover step 2");
    assert_eq!(adapter.active_horizon_slots_mask(), 0);
    assert_eq!(custom_slot.retractions, 1);
    assert_eq!(adapter.escalated_workbench_lane(), None);
    assert_eq!(adapter.substrate().slot_enabled_mask & (1 << esc_lane), 0);

    let snap = adapter.telemetry_snapshot();
    assert_eq!(snap.spine.total_steps, 4);
    assert_eq!(snap.spine.continuous_standard_steps, 4);
    assert_eq!(snap.liminal.slide_in_count, 1);
    assert_eq!(snap.liminal.slide_out_count, 1);
    assert_eq!(snap.ego.escalation_swaps, 1);
    assert_eq!(snap.mode, TelemetryVerbosityMode::Standard);
    assert!(snap.detailed.is_some());

    let adj = adapter.evaluate_slow_loop_telemetry(4);
    assert!(adj.applied);
}

#[test]
fn adapter_residual_spillover_dimension_elevation_and_verified_cache() {
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact};
    use liminika_core::stigmergy::DissipativeOverflowReport;

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let q_len = adapter.geometry().q_rot_len();
    let mut trail = lmnk_pointer_trail_t::default();
    let mut q_rot = vec![0.4_f32; q_len];
    let mut no_esc = ();
    let signal = ObjectiveFact::new([22i8; 64], 1.6, 0, 121);

    adapter.adapt_core_layers(
        &EpistemicHorizonReading {
            total_memory_support: 0.3,
            conflict_severity: 0.85,
            regime_code: EpistemicRegime::InternalConflict as u8,
            ..EpistemicHorizonReading::default()
        },
        &OverlapDiscoveryReport::default(),
        None,
    );
    adapter.stage_overflow_residual(&DissipativeOverflowReport {
        overflowed: true,
        residual_words: [0x1414_1414_1414_1414; 8],
        ..DissipativeOverflowReport::default()
    });

    // Step 1: CoherenceHorizonGuardSlot (dwell 1) enters; ResidualSpillover waits for dwell 2.
    let _ = adapter
        .orchestrate_layer_rope_with_spillover(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &signal,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut no_esc,
            &mut trail,
        )
        .expect("step 1");
    assert_eq!(adapter.active_horizon_slots_mask(), 1 << 0);
    assert_eq!(adapter.residual_spillover().spillover_modulations, 0);

    // Step 2: Sustained conflict triggers ResidualSpilloverHorizonSlot alongside guard slot.
    let _ = adapter
        .orchestrate_layer_rope_with_spillover(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &signal,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut no_esc,
            &mut trail,
        )
        .expect("step 2");
    assert_eq!(adapter.active_horizon_slots_mask(), (1 << 0) | (1 << 1));
    assert_eq!(adapter.residual_spillover().spillover_modulations, 1);

    // Commit verified outcome -> promotes to Tier-1 RAM cache & telemetry syncs cleanly.
    let verified_phases = [0x14i8; 64];
    let (stored, evicted) = adapter.commit_spillover_experience(&verified_phases, 0.92, true);
    assert!(stored);
    assert_eq!(evicted, 0);
    assert_eq!(
        adapter.telemetry_snapshot().spine.experience_cache_stores,
        1
    );

    // Step 3: Next step hits Tier-1 experience cache in O(1).
    let _ = adapter
        .orchestrate_layer_rope_with_spillover(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &signal,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut no_esc,
            &mut trail,
        )
        .expect("step 3");
    assert_eq!(adapter.residual_spillover().cache_hits, 1);
    assert_eq!(adapter.telemetry_snapshot().spine.experience_cache_hits, 1);
}

#[test]
fn adapter_distortion_feedback_wakes_and_cools_ego_and_renders_mermaid() {
    use liminika_core::drivers::{EGO_TRIGGER_LAYER_DISTORTION, LiminalEgoPreset};
    use liminika_core::runtime::{LayerDistortionKind, LayerDistortionReport};

    let geom = sample_gqa_partial_rope();
    let mut adapter = ModelSteeringAdapter::new(geom).expect("valid adapter");
    adapter.configure_ego_from_preset(LiminalEgoPreset::Low);
    let base_thrust = adapter.ego_state().basin_resistance_q7;

    // 1. Passive saturation wakes Ego with EGO_TRIGGER_LAYER_DISTORTION.
    let passive = LayerDistortionReport {
        action_saturation_ratio: 0.90,
        passive_distortion_ratio: 0.80,
        active_distortion_ratio: 0.0,
        mismatch_ratio: 0.80,
        kind: LayerDistortionKind::PassiveSaturation,
        ego_wakeup_recommended: true,
        ego_cooldown_recommended: false,
    };
    adapter.apply_distortion_ego_feedback(&passive);
    assert_ne!(
        adapter.ego_state().trigger_mask & EGO_TRIGGER_LAYER_DISTORTION,
        0
    );
    assert!(adapter.ego_state().basin_resistance_q7 >= 48);

    // 2. Active overdrive cools down Ego thrust relative to configured baseline.
    let active = LayerDistortionReport {
        action_saturation_ratio: 0.90,
        passive_distortion_ratio: 0.0,
        active_distortion_ratio: 0.80,
        mismatch_ratio: 0.80,
        kind: LayerDistortionKind::ActiveOverdrive,
        ego_wakeup_recommended: false,
        ego_cooldown_recommended: true,
    };
    adapter.apply_distortion_ego_feedback(&active);
    assert_eq!(
        adapter.ego_state().trigger_mask & EGO_TRIGGER_LAYER_DISTORTION,
        0
    );
    assert_eq!(adapter.ego_state().basin_resistance_q7, base_thrust >> 1);

    // 3. Returning to Aligned restores configured Ego baseline and Mermaid renders LayerDistortion.
    let aligned = LayerDistortionReport::default();
    adapter.apply_distortion_ego_feedback(&aligned);
    assert_eq!(adapter.ego_state().basin_resistance_q7, base_thrust);
    let mermaid = adapter.render_liminal_topology_mermaid(4, 1);
    assert!(mermaid.contains("LayerDistortion"));
}

#[test]
fn test_slow_loop_updates_threshold_policy_without_exponential_drift() {
    use liminika_core::runtime::{LayerDistortionKind, LayerDistortionReport};
    use liminika_core::stigmergy::StigmergicThresholds;

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).unwrap();
    let default_rank1 = StigmergicThresholds::default().rank1_inject_density_thresh;
    let mut trail = lmnk_pointer_trail_t {
        stigmergic_density: 1.80,
        ..lmnk_pointer_trail_t::default()
    };
    let sig = ObjectiveFact::new([12_i8; 64], 1.5, 0, 1);
    let mut q = vec![0.25_f32; 24 * 96];

    for _ in 0..20 {
        let _ = adapter
            .orchestrate_layer_rope_with_substrate(
                LayerRopeStepInput {
                    layer_idx: 0,
                    signal: &sig,
                    entropy_h_tilde_opt: Some(0.45),
                    unresolved_damping_u7: 128,
                    q_rot_inout: &mut q,
                    logits_opt: None,
                },
                &mut trail,
            )
            .unwrap();
    }
    let adj1 = adapter.apply_slow_loop_telemetry_epoch(16);
    assert!(adj1.applied);
    let first_rank1 = adj1.tuned_thresholds.rank1_inject_density_thresh;
    assert_eq!(
        adapter
            .threshold_policy_mut()
            .base
            .rank1_inject_density_thresh,
        first_rank1
    );
    let policy_copy = *adapter.threshold_policy_mut();
    let expected_resolved = policy_copy.resolve_thresholds(
        adapter.system_constraint(),
        adapter.probe_baseline(),
        &lmnk_pointer_trail_t::default(),
        adapter.probe_baseline().baseline_entropy,
    );
    assert!(
        (adapter.substrate().thresholds.rank1_inject_density_thresh
            - expected_resolved.rank1_inject_density_thresh)
            .abs()
            < 1e-6
    );

    // Second epoch over the same cumulative distribution must remain bounded against the
    // calibrated base rather than compounding `1.15 * 1.15`.
    let adj2 = adapter.apply_slow_loop_telemetry_epoch(16);
    assert!(adj2.applied);
    assert!(
        adj2.tuned_thresholds.rank1_inject_density_thresh <= default_rank1 * 1.15 + 1e-4,
        "Slow-Loop threshold drifted past bounded range: {}",
        adj2.tuned_thresholds.rank1_inject_density_thresh
    );

    let custom_dist = LayerDistortionReport {
        action_saturation_ratio: 0.85,
        passive_distortion_ratio: 0.15,
        active_distortion_ratio: 0.72,
        mismatch_ratio: 0.40,
        kind: LayerDistortionKind::ActiveOverdrive,
        ego_wakeup_recommended: false,
        ego_cooldown_recommended: true,
    };
    let mermaid = adapter.render_liminal_topology_mermaid_with_distortion(2, 0, &custom_dist);
    assert!(mermaid.contains("Kind:ActiveOverdrive"));
    assert!(mermaid.contains("COOLDOWN"));
}

#[test]
fn test_workbench_preset_and_boundary_gated_escalation_granularity() {
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let q_len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let ref_q = vec![0.0_f32; q_len];
    let guard_q: Vec<f32> = (0..q_len)
        .map(|i| if (i & 1) == 0 { 0.8 } else { -0.4 })
        .collect();
    let ref_slices: Vec<&[f32]> = vec![ref_q.as_slice(); num_layers];
    let guard_slices: Vec<&[f32]> = vec![guard_q.as_slice(); num_layers];

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Guard,
            "Contradiction guard",
            &guard_slices,
            &ref_slices,
            2.0,
        )
        .expect("ingest guard");

    use liminika_core::stigmergy::{WORKBENCH_LANE_FLAG_PINNED, WORKBENCH_LANE_FLAG_SANDBOX};

    // 1. Switch 8-lane workbench preset atomically using AutoFree + pinned explicit lanes.
    let specs = [
        WorkbenchSlotSpec::explicit(
            5,
            true,
            WORKBENCH_LANE_FLAG_PINNED,
            1.25,
            [16_i8; MAX_ATTENTION_HEADS],
        ),
        WorkbenchSlotSpec::auto_free(1.50, [8_i8; MAX_ATTENTION_HEADS]),
    ];
    let updated = adapter
        .switch_workbench_preset(WorkbenchPreset::FactualGrounding, &specs)
        .expect("switch preset");
    assert_eq!(updated, 2);
    assert_eq!(
        adapter.active_workbench_preset(),
        WorkbenchPreset::FactualGrounding
    );
    assert_eq!(adapter.substrate().active_lane_mask(), (1 << 0) | (1 << 5));
    assert_eq!(adapter.substrate().pinned_lane_mask(), 1 << 5);

    // Switching presets again preserves the pinned lane 5 and refreshes unpinned lanes.
    let updated_2 = adapter
        .switch_workbench_preset(
            WorkbenchPreset::BalancedDialog,
            &[WorkbenchSlotSpec::auto_free(
                1.0,
                [12_i8; MAX_ATTENTION_HEADS],
            )],
        )
        .expect("switch preset preserving pinned lane");
    assert_eq!(updated_2, 1);
    assert_eq!(adapter.substrate().active_lane_mask(), (1 << 0) | (1 << 5));
    assert_eq!(adapter.substrate().pinned_lane_mask(), 1 << 5);

    // 2. Within the same destabilized regime across 16 steered layers, escalation runs once at
    //    the regime boundary rather than repeating on every layer.
    adapter.reset_horizon_and_core_state();
    assert_eq!(adapter.substrate().active_lane_mask(), 0);
    adapter.adapt_core_layers(
        &liminika_core::runtime::EpistemicHorizonReading {
            total_memory_support: 0.4,
            conflict_severity: 0.75,
            regime_code: liminika_core::liminal::EpistemicRegime::InternalConflict as u8,
            ..liminika_core::runtime::EpistemicHorizonReading::default()
        },
        &liminika_core::stigmergy::OverlapDiscoveryReport::default(),
        None,
    );
    let mut esc = CorpusRecallEscalationProvider::new(&corpus);
    let mut trail = lmnk_pointer_trail_t::default();
    let sig = ObjectiveFact::new([12_i8; 64], 1.5, 0, 1);
    let mut q = vec![0.25_f32; q_len];
    for layer_idx in 0..num_layers {
        let _ = adapter
            .orchestrate_layer_rope_with_spillover(
                LayerRopeStepInput {
                    layer_idx,
                    signal: &sig,
                    entropy_h_tilde_opt: None,
                    unresolved_damping_u7: 128,
                    q_rot_inout: &mut q,
                    logits_opt: None,
                },
                &mut esc,
                &mut trail,
            )
            .expect("layer step");
    }
    assert_eq!(esc.escalation_calls, 1);
    assert_eq!(adapter.telemetry_snapshot().ego.escalation_swaps, 1);
    assert!(adapter.escalated_workbench_lane().is_some());

    // 3. Standard workbench lanes contribute inside `overlap_core_mask | overlap_fringe_mask`, while
    //    sandbox lanes act only inside `lacuna_exterior_mask | discovery_excursion_mask`.
    adapter.reset_horizon_and_core_state();
    adapter
        .switch_workbench_preset(
            WorkbenchPreset::CreativeLiminal,
            &[
                WorkbenchSlotSpec::auto_free(1.0, [24_i8; MAX_ATTENTION_HEADS]),
                WorkbenchSlotSpec::explicit(
                    6,
                    false,
                    WORKBENCH_LANE_FLAG_SANDBOX,
                    1.0,
                    [40_i8; MAX_ATTENTION_HEADS],
                ),
            ],
        )
        .expect("preset switch");
    assert_eq!(adapter.substrate().sandbox_lane_mask(), 1 << 6);
    adapter.adapt_core_layers(
        &liminika_core::runtime::EpistemicHorizonReading::default(),
        &liminika_core::stigmergy::OverlapDiscoveryReport {
            overlap_core_mask: 0x0F,
            lacuna_exterior_mask: 0xF0,
            ..liminika_core::stigmergy::OverlapDiscoveryReport::default()
        },
        None,
    );
    let mut trail_overlap = lmnk_pointer_trail_t::default();
    let _ = adapter
        .orchestrate_layer_rope_with_substrate(
            LayerRopeStepInput {
                layer_idx: 16,
                signal: &sig,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q,
                logits_opt: None,
            },
            &mut trail_overlap,
        )
        .expect("overlap layer step");
    assert!(trail_overlap.accumulated_phase_delta[0] > 0);
    assert!(trail_overlap.accumulated_phase_delta[40] > 0);
    // Verify sandbox lane 6 does not inflate overlap_core_mask head 0 compared to standard-only pass.
    let core_head_with_sandbox = trail_overlap.accumulated_phase_delta[0];
    let _ = adapter
        .substrate_mut()
        .release_lane(6)
        .expect("release sandbox");
    let mut trail_no_sb = lmnk_pointer_trail_t::default();
    let _ = adapter
        .orchestrate_layer_rope_with_substrate(
            LayerRopeStepInput {
                layer_idx: 16,
                signal: &sig,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q,
                logits_opt: None,
            },
            &mut trail_no_sb,
        )
        .expect("no-sandbox layer step");
    assert_eq!(
        core_head_with_sandbox,
        trail_no_sb.accumulated_phase_delta[0]
    );
    assert!(trail_overlap.accumulated_phase_delta[40] > trail_no_sb.accumulated_phase_delta[40]);
}
