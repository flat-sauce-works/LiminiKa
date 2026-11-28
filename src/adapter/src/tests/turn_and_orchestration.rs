// name: src/adapter/src/tests/turn_and_orchestration.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Turn orchestrator, cognitive circuits, and multi-turn tests.

use super::*;

#[test]
fn turn_orchestrator_mock_source_separates_self_ref_and_exterior_and_syncs_circuit() {
    use liminika_core::runtime::{
        AdaptiveThresholdPolicy, ManifoldGatewaySignal, ObjectiveFact, StandardPhaseKvGopCache,
    };
    use liminika_core::stigmergy::{
        CognitiveOpKind, CognitiveStepOp, StandardCognitiveBank, StigmergicScaleGranularity,
    };

    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 4;
    geom.layer_rope_mask = 0xF;
    let mut adapter = ModelSteeringAdapter::new(geom).expect("init adapter");
    let mut kv_cache = StandardPhaseKvGopCache::new();
    let kv_policy = AdaptiveThresholdPolicy::default();

    struct MockRecallSource {
        recall: OwnedAttractorRecall,
    }
    impl AttractorRecallSource for MockRecallSource {
        fn recall_query(
            &self,
            _adapter: &ModelSteeringAdapter,
            _query_token_qs: &[Vec<f32>],
            _query_seq_len: usize,
            _prior_circuit: Option<CognitiveTrajectoryLedger>,
        ) -> Result<OwnedAttractorRecall, String> {
            Ok(self.recall.clone())
        }
    }

    let self_ref_q7 = [24_i8; MAX_ATTENTION_HEADS];
    let exterior_q7 = [-32_i8; MAX_ATTENTION_HEADS];
    let persona_q7 = [12_i8; MAX_ATTENTION_HEADS];
    let frame = ObjectiveFact::new(self_ref_q7, 2.0, 3, 11).synthesize_gateway_frame();
    let mut traj = CognitiveTrajectoryLedger::new();
    traj.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::DecomposePeel,
        6,
    ));
    let partition =
        evaluate_lacuna_partition_branchless(0x0F, &[64, 64, 0, 0, 72, 72, 0, 0], &[8; 8], 28);

    // Deliberately supply BOTH `recalled_facts` (S & O) AND `exterior_sandbox_facts` (O \ S)
    // to prove `TurnSteeringPlan` never conflates them across channels.
    let mock_both = MockRecallSource {
        recall: OwnedAttractorRecall {
            lacuna_state: LacunaBoundaryState::InsideBasin,
            top1_fact_score: 0.72,
            top2_fact_score: 0.25,
            top1_persona_score: 0.45,
            min_local_lacuna_depth: 0.0,
            guard_conflict_margin: -0.35,
            action_decision: StigmergicActionDecision::ExpressPhaseSteer,
            gateway_frame: frame,
            telemetry: StratumProbeTelemetry {
                cognitive_trajectory: traj,
                ..StratumProbeTelemetry::default()
            },
            recalled_facts: vec![RecalledAnchorRecord {
                anchor_id: 11,
                role: AttractorRole::Fact,
                text: "SelfRef ground truth".to_string(),
                per_layer_q7: vec![self_ref_q7; 4],
                energy: 2.5,
                score: 0.72,
            }],
            recalled_personas: vec![RecalledAnchorRecord {
                anchor_id: 21,
                role: AttractorRole::Persona,
                text: "Concise style".to_string(),
                per_layer_q7: vec![persona_q7; 4],
                energy: 1.5,
                score: 0.45,
            }],
            top_guard: None,
            lacuna_partition: partition,
            exterior_sandbox_facts: vec![RecalledAnchorRecord {
                anchor_id: 99,
                role: AttractorRole::Fact,
                text: "Exterior known-unknown pointer".to_string(),
                per_layer_q7: vec![exterior_q7; 4],
                energy: 2.0,
                score: 0.60,
            }],
            is_known_unknown_exterior: true,
        },
    };

    let mut bank = StandardCognitiveBank::from_ledger(traj);
    let plan = TurnOrchestrator::prepare_turn(
        &mock_both,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput {
            query_token_qs: &[],
            query_seq_len: 4,
            token_include_mask: None,
            turn_index: 1,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: Some(&bank),
            layer_steer_mask_override: Some(0x0000_000C),
        },
    )
    .expect("prepare_turn with mock source");

    // 1. Strict topological separation: S & O never contains O \ S anchor #99 or exterior text.
    assert_eq!(plan.self_ref_anchor_ids(), &[11]);
    assert_eq!(plan.self_ref_context(), "SelfRef ground truth");
    assert_eq!(plan.persona_context(), "Concise style");
    assert_eq!(plan.exterior_context(), "Exterior known-unknown pointer");
    assert_eq!(plan.self_ref_anchors_per_layer().len(), 4);
    assert_eq!(plan.exterior_sandbox_anchors_per_layer().len(), 4);
    assert_eq!(plan.persona_anchors_per_layer().len(), 4);
    for layer in 0..4 {
        assert_eq!(plan.self_ref_anchors_per_layer()[layer].len(), 1);
        assert_eq!(plan.self_ref_anchors_per_layer()[layer][0].anchor_id, 11);
        assert_eq!(plan.exterior_sandbox_anchors_per_layer()[layer].len(), 1);
        assert_eq!(
            plan.exterior_sandbox_anchors_per_layer()[layer][0].anchor_id,
            99
        );
        // Exterior sandbox anchor uses 0.5x damped energy (`2.0 * 0.60 * 0.5 = 0.60`).
        assert!(
            (plan.exterior_sandbox_anchors_per_layer()[layer][0].base_energy - 0.60).abs() < 1e-5
        );
        assert_eq!(plan.persona_anchors_per_layer()[layer].len(), 1);
        assert_eq!(plan.persona_anchors_per_layer()[layer][0].anchor_id, 21);
    }

    // 2. LacunaPartition8, layer_steer_mask_override, top_guard zero-aperture, and active circuit synced.
    assert_eq!(plan.lacuna_partition(), partition);
    assert_eq!(*adapter.last_lacuna_partition(), partition);
    assert_eq!(plan.layer_steer_mask_override(), Some(0x0000_000C));
    assert_eq!(plan.effective_layer_steer_mask(&adapter), 0x0000_000C);
    assert_eq!(plan.query_snapshot().top_guard_q7, None);
    assert!(adapter.active_circuit().is_some());

    // 3. Post-turn circuit commit reinforces bank and updates adapter active circuit.
    TurnOrchestrator::commit_turn_circuit(&mut adapter, &plan, &mut bank, None);
    assert!(adapter.active_circuit().is_some());

    // 4. Preparing a turn without persona_bank clears the active circuit on adapter.
    let plan_no_bank = TurnOrchestrator::prepare_turn(
        &mock_both,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput {
            query_token_qs: &[],
            query_seq_len: 4,
            token_include_mask: None,
            turn_index: 2,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None::<&StandardCognitiveBank>,
            layer_steer_mask_override: None,
        },
    )
    .expect("prepare_turn without bank");
    assert!(adapter.active_circuit().is_none());
    assert_eq!(
        plan_no_bank.effective_layer_steer_mask(&adapter),
        adapter.geometry().layer_rope_mask
    );
}

#[test]
fn turn_steering_plan_reflection_gate_input_respects_live_entropy_and_lacuna_regions() {
    use liminika_core::runtime::{AdaptiveThresholdPolicy, StandardPhaseKvGopCache};
    use liminika_core::stigmergy::StandardCognitiveBank;

    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 2;
    geom.layer_rope_mask = 0x3;
    let mut adapter = ModelSteeringAdapter::new(geom).expect("init adapter");
    let mut kv_cache = StandardPhaseKvGopCache::new();
    let kv_policy = AdaptiveThresholdPolicy::default();

    struct StaticSource(OwnedAttractorRecall);
    impl AttractorRecallSource for StaticSource {
        fn recall_query(
            &self,
            _adapter: &ModelSteeringAdapter,
            _query_token_qs: &[Vec<f32>],
            _query_seq_len: usize,
            _prior_circuit: Option<CognitiveTrajectoryLedger>,
        ) -> Result<OwnedAttractorRecall, String> {
            Ok(self.0.clone())
        }
    }

    // Case A: Pure Unobserved OOD (`LiminalExterior`, `is_known_unknown_exterior = false`).
    let unobserved_part = evaluate_lacuna_partition_branchless(0x0F, &[0; 8], &[8; 8], 28);
    let ood_source = StaticSource(OwnedAttractorRecall {
        lacuna_state: LacunaBoundaryState::LiminalExterior,
        top1_fact_score: 0.05,
        top2_fact_score: 0.02,
        top1_persona_score: 0.01,
        min_local_lacuna_depth: 0.30,
        guard_conflict_margin: -0.20,
        action_decision: StigmergicActionDecision::PassiveSidecar,
        gateway_frame: GatewayFrame::default(),
        telemetry: StratumProbeTelemetry::default(),
        recalled_facts: Vec::new(),
        recalled_personas: Vec::new(),
        top_guard: None,
        lacuna_partition: unobserved_part,
        exterior_sandbox_facts: Vec::new(),
        is_known_unknown_exterior: false,
    });
    let ood_plan = TurnOrchestrator::prepare_turn(
        &ood_source,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput {
            query_token_qs: &[],
            query_seq_len: 3,
            token_include_mask: None,
            turn_index: 1,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None::<&StandardCognitiveBank>,
            layer_steer_mask_override: None,
        },
    )
    .expect("ood plan");
    assert!(ood_plan.is_unobserved_lacuna());
    assert!(!ood_plan.steering_eligible());

    let calm_gate = ood_plan.reflection_gate_input(0.25);
    assert!(calm_gate.is_unobserved_lacuna);
    assert!(!calm_gate.has_undigested_known_unknown);
    assert!((calm_gate.exit_entropy_h_tilde - 0.25).abs() < 1e-6);
    assert!(!ManifoldReflectionEvaluator.should_trigger_reflection(&calm_gate));

    let spike_gate = ood_plan.reflection_gate_input(0.75);
    assert!((spike_gate.exit_entropy_h_tilde - 0.75).abs() < 1e-6);
    assert!(ManifoldReflectionEvaluator.should_trigger_reflection(&spike_gate));

    // Case B: Validated Known Unknown (`O \ S`, `is_known_unknown_exterior = true`) triggers even
    // under calm live entropy while leaving `self_ref_anchors_per_layer` strictly empty.
    let ext_part =
        evaluate_lacuna_partition_branchless(0x0F, &[64, 64, 0, 0, 80, 0, 0, 0], &[8; 8], 28);
    let ext_source = StaticSource(OwnedAttractorRecall {
        lacuna_state: LacunaBoundaryState::LiminalExterior,
        top1_fact_score: 0.31,
        top2_fact_score: 0.12,
        top1_persona_score: 0.05,
        min_local_lacuna_depth: 0.04,
        guard_conflict_margin: -0.15,
        action_decision: StigmergicActionDecision::PassiveSidecar,
        gateway_frame: GatewayFrame::default(),
        telemetry: StratumProbeTelemetry::default(),
        recalled_facts: Vec::new(),
        recalled_personas: Vec::new(),
        top_guard: Some(RecalledAnchorRecord {
            anchor_id: 77,
            role: AttractorRole::Guard,
            text: "Observed guard".to_string(),
            per_layer_q7: vec![[19_i8; MAX_ATTENTION_HEADS]; 2],
            energy: 2.0,
            score: 0.16,
        }),
        lacuna_partition: ext_part,
        exterior_sandbox_facts: vec![RecalledAnchorRecord {
            anchor_id: 55,
            role: AttractorRole::Fact,
            text: "Known unknown target".to_string(),
            per_layer_q7: vec![[40_i8; MAX_ATTENTION_HEADS]; 2],
            energy: 2.2,
            score: 0.31,
        }],
        is_known_unknown_exterior: true,
    });
    let ext_plan = TurnOrchestrator::prepare_turn(
        &ext_source,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput {
            query_token_qs: &[],
            query_seq_len: 3,
            token_include_mask: None,
            turn_index: 2,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None::<&StandardCognitiveBank>,
            layer_steer_mask_override: None,
        },
    )
    .expect("ext plan");
    assert!(!ext_plan.is_unobserved_lacuna());
    assert!(ext_plan.is_known_unknown_exterior());
    assert!(ext_plan.steering_eligible());
    assert!(!ext_plan.has_self_ref_anchors());
    assert!(ext_plan.has_exterior_sandbox_anchors());
    assert!(ext_plan.self_ref_context().is_empty());
    assert_eq!(ext_plan.exterior_context(), "Known unknown target");
    assert_eq!(
        ext_plan.query_snapshot().top_guard_q7,
        Some([19_i8; MAX_ATTENTION_HEADS])
    );

    let ext_gate = ext_plan.reflection_gate_input(0.20);
    assert!(!ext_gate.is_unobserved_lacuna);
    assert!(ext_gate.has_undigested_known_unknown);
    assert!(ManifoldReflectionEvaluator.should_trigger_reflection(&ext_gate));
}

#[test]
fn input_distillation_boundary_encoders_alias_probe_and_contrastive_bank() {
    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 2;
    geom.num_q_heads = 8;
    geom.num_kv_heads = 4;
    geom.head_dim = 16;
    geom.rotary_dim = 16;
    geom.layer_rope_mask = 0x3;
    let adapter = ModelSteeringAdapter::new(geom).expect("init compact adapter");
    let q_len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;

    assert_eq!(DEFAULT_STARTUP_PRIOR_PROBES.len(), 24);
    assert_eq!(
        extract_leading_alias_word("Alpha, beta gamma"),
        Some("Alpha")
    );
    assert_eq!(extract_leading_alias_word("A: short"), None);
    assert_eq!(extract_leading_alias_word("   "), None);

    // 1. TokenSequenceEncoder + PooledSubtitleEncoder (resize pad and truncate).
    struct SynthEncoder {
        num_layers: usize,
        q_len: usize,
    }
    impl SynthEncoder {
        fn synth_layer(&self, seed: f32) -> Vec<f32> {
            let mut flat = vec![0.0_f32; 2 * self.q_len];
            for t in 0..2 {
                let base = t * self.q_len;
                for d in 0..self.q_len {
                    let angle = seed + (d as f32) * 0.13 + (t as f32) * 0.05;
                    flat[base + d] = angle.cos();
                }
            }
            flat
        }
    }
    impl TokenSequenceEncoder for SynthEncoder {
        fn encode_token_sequence(&mut self, text: &str) -> Result<CapturedTokenSequence, String> {
            let seed = text
                .bytes()
                .fold(0.1_f32, |acc, b| acc + f32::from(b) * 0.01);
            let per_layer = (0..self.num_layers)
                .map(|l| self.synth_layer(seed + (l as f32) * 0.2))
                .collect();
            let hidden = vec![seed.cos(), seed.sin(), 0.25, -0.5];
            Ok(CapturedTokenSequence::new(2, per_layer).with_final_hidden(hidden))
        }
    }

    let mut enc = SynthEncoder { num_layers, q_len };
    let staged = CompiledTokenCorpus::capture_entries(
        &mut enc,
        &[
            (AttractorRole::Fact, "Alpha, core manifold invariant.", 2.5),
            (AttractorRole::Fact, "Beta: boundary staging rule.", 2.5),
            (AttractorRole::Persona, "Calm concise tone.", 1.5),
            (AttractorRole::Guard, "False overwrite claim.", 2.0),
        ],
    )
    .expect("capture entries");
    assert_eq!(staged.len(), 4);

    let mut sub_enc =
        PooledSubtitleEncoder::new(SynthEncoder { num_layers, q_len }, q_len, q_len + 8);
    let rec = SourceTextRecord {
        frame_index: 0,
        timestamp_epoch_sec: 1,
        role: AttractorRole::Fact,
        text: "Alpha, core manifold invariant.".to_string(),
        energy: 2.0,
    };
    let pooled_sub = sub_enc
        .encode_subtitle_activations(&rec)
        .expect("pooled subtitle");
    assert_eq!(pooled_sub.len(), num_layers);
    assert_eq!(pooled_sub[0].len(), q_len + 8);
    assert!(pooled_sub[0][q_len..].iter().all(|&v| v == 0.0));

    // 2. PrefixAliasProbe only probes Fact entries with leading words >= 2 chars.
    struct MockAliasProbe {
        probed_words: Vec<String>,
        num_layers: usize,
    }
    impl PrefixAliasProbe for MockAliasProbe {
        fn probe_leading_alias(
            &mut self,
            word: &str,
            q_rot_len: usize,
        ) -> Result<Option<CapturedTokenSequence>, String> {
            self.probed_words.push(word.to_string());
            if word == "Alpha" {
                Ok(Some(CapturedTokenSequence::new(
                    1,
                    vec![vec![0.5_f32; q_rot_len]; self.num_layers],
                )))
            } else {
                Ok(None)
            }
        }
    }
    let mut compiled =
        CompiledTokenCorpus::compile(&adapter, &staged, SequenceActivationPooler::default(), 2, 1)
            .expect("compile");
    let before_toks = compiled.entry_token_sigs[0].len();
    let mut alias_probe = MockAliasProbe {
        probed_words: Vec::new(),
        num_layers,
    };
    let appended = compiled
        .append_leading_alias_observations(&adapter, &staged, &mut alias_probe)
        .expect("append alias observations");
    assert_eq!(alias_probe.probed_words, vec!["Alpha", "Beta"]);
    assert_eq!(appended, 1);
    assert_eq!(compiled.entry_token_sigs[0].len(), before_toks + 1);

    // 3. ContrastiveAnchorBank polarity separation: Fact/Persona vs Guard + Rank-1 normalization.
    let head_rot = adapter.geometry().rotary_dim;
    let half = head_rot / 2;
    let num_heads = adapter.geometry().num_q_heads;
    let mut fact_q = vec![0.0_f32; q_len];
    let mut persona_q = vec![0.0_f32; q_len];
    let mut guard_q = vec![0.0_f32; q_len];
    for h in 0..num_heads {
        let base = h * head_rot;
        fact_q[base] = 0.80;
        fact_q[base + half] = 0.60;
        persona_q[base] = 0.90;
        persona_q[base + half] = 0.44;
        guard_q[base] = 0.60;
        guard_q[base + half] = -0.80;
    }
    let entries = vec![
        ContrastiveAnchorEntry {
            role: AttractorRole::Fact,
            captured: CapturedTokenSequence::new(1, vec![fact_q; num_layers])
                .with_final_hidden(vec![1.0, 2.0, 0.0, 0.0]),
        },
        ContrastiveAnchorEntry {
            role: AttractorRole::Persona,
            captured: CapturedTokenSequence::new(1, vec![persona_q; num_layers])
                .with_final_hidden(vec![1.0, 0.0, 2.0, 0.0]),
        },
        ContrastiveAnchorEntry {
            role: AttractorRole::Guard,
            captured: CapturedTokenSequence::new(1, vec![guard_q; num_layers])
                .with_final_hidden(vec![-1.0, -1.0, 0.0, 0.0]),
        },
    ];
    let bank = ContrastiveAnchorBank::from_entries(&adapter, &entries).expect("contrastive bank");
    assert_eq!(bank.fact_anchors_per_layer.len(), num_layers);
    assert_eq!(bank.persona_anchors_per_layer.len(), num_layers);
    assert_eq!(bank.guard_anchors_per_layer.len(), num_layers);
    assert_eq!(bank.fact_anchors_per_layer[0][0].anchor_id, 1);
    assert_eq!(bank.persona_anchors_per_layer[0][0].anchor_id, 101);
    // Attractive Fact delta (`neg -> fact`) and repulsive Guard delta (`pos -> guard`) must oppose in sign.
    let fact_h0 = bank.fact_anchors_per_layer[0][0].payload[0];
    let guard_h0 = bank.guard_anchors_per_layer[0][0].phase_deltas[0];
    assert!(
        i32::from(fact_h0) * i32::from(guard_h0) < 0,
        "expected opposite geometric polarity between fact ({fact_h0}) and guard ({guard_h0})"
    );
    let u_norm = bank.rank1_u.iter().map(|&x| x * x).sum::<f32>().sqrt();
    assert!(
        (u_norm - 1.0).abs() < 1e-5,
        "rank1_u must be unit normalized"
    );
    assert_eq!(bank.rank1_u, bank.rank1_v);

    // Without `final_hidden`, rank1 vectors stay cleanly empty.
    let no_hidden_entries = vec![ContrastiveAnchorEntry {
        role: AttractorRole::Fact,
        captured: CapturedTokenSequence::new(1, vec![vec![1.0_f32; q_len]; num_layers]),
    }];
    let bank_no_h =
        ContrastiveAnchorBank::from_entries(&adapter, &no_hidden_entries).expect("no-hidden bank");
    assert!(bank_no_h.rank1_u.is_empty());
    assert!(bank_no_h.rank1_v.is_empty());
}

#[test]
fn adapter_reanchor_compute_pressure_and_reset_turn_decode_state_preserve_guard_policy() {
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("adapter");
    adapter.reanchor_compute_pressure(12_345);
    assert_eq!(adapter.compute_pool().baseline_pressure_q15, 12_345);

    adapter.begin_turn_spine_policy(true, false);
    adapter.substrate_mut().last_entropy_h_tilde = 0.65;
    adapter.reset_turn_decode_state();
    assert_eq!(adapter.substrate().last_entropy_h_tilde, 0.0);
    assert_eq!(adapter.system_spine().whole_token_bypass_active, 0);
    assert!(
        adapter.system_spine().turn_guard_active(),
        "reset_turn_decode_state must preserve active turn guard pin"
    );
}

#[test]
fn slow_loop_crystallized_anchors_are_captured_in_bounded_fifo_and_drainable() {
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("adapter");
    adapter.configure_ego_from_preset(LiminalEgoPreset::High);
    adapter.set_ego_max_hops(3);
    assert!(adapter.crystallized_anchors().is_empty());

    // Push more than `MAX_CRYSTALLIZED_ANCHOR_BUFFER` epochs with a consolidation-eligible
    // 3-hop P-Frame node so we verify both live Slow-Loop capture and bounded FIFO eviction.
    let total_to_crystallize = ModelSteeringAdapter::MAX_CRYSTALLIZED_ANCHOR_BUFFER + 5;
    for i in 0..total_to_crystallize {
        let mut node =
            liminika_core::drivers::DistortionDelta::new_empty((i + 1) as u32, (i + 100) as u32);
        node.hop_depth = 3;
        node.reinforce_count = 6;
        node.gain_q7 = 96;
        node.pheromone_q7 = 96;
        node.delta_phase_q7 = [24_i8; 64];
        node.flags = liminika_core::drivers::DISTORTION_DELTA_FLAG_ACTIVE
            | liminika_core::drivers::DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        {
            let net = adapter.runtime_context_mut().ego_differential_net_mut();
            net.consolidation_hop_limit = 2;
            net.set_node(0, node);
        }
        // Populate >= 4 telemetry samples so `apply_slow_loop_telemetry_epoch(4)` triggers.
        let horizon = EpistemicHorizonReading::default();
        let delta = LiminalBoundaryDelta::default();
        let overlap = OverlapDiscoveryReport::default();
        let ego = *adapter.ego_state();
        for _ in 0..4 {
            adapter
                .runtime_context_mut()
                .telemetry_bank_mut()
                .record_step(liminika_core::runtime::StepTelemetryInput {
                    decision: StigmergicActionDecision::FullOrchestration,
                    horizon: &horizon,
                    delta: &delta,
                    overlap: &overlap,
                    ego: &ego,
                    entropy_gate_mode: 0,
                    slid_in_bits: 0,
                    slid_out_bits: 0,
                    escalated: false,
                    qos_throttled: false,
                });
        }
        let adj = adapter.apply_slow_loop_telemetry_epoch(4);
        assert!(adj.applied, "epoch {i} should apply slow-loop maintenance");
    }

    let buffered = adapter.crystallized_anchors();
    assert_eq!(
        buffered.len(),
        ModelSteeringAdapter::MAX_CRYSTALLIZED_ANCHOR_BUFFER
    );
    // Oldest 5 records (`anchor_ref_id` 100..105) were evicted in FIFO order.
    assert_eq!(buffered[0].anchor_id, 105);
    assert_eq!(
        buffered[buffered.len() - 1].anchor_id,
        (total_to_crystallize - 1 + 100) as u32
    );
    assert!(buffered[0].verify_integrity());

    let drained = adapter.drain_crystallized_anchors();
    assert_eq!(
        drained.len(),
        ModelSteeringAdapter::MAX_CRYSTALLIZED_ANCHOR_BUFFER
    );
    assert_eq!(drained[0].anchor_id, 105);
    assert!(adapter.crystallized_anchors().is_empty());
}

#[test]
fn pure_guard_trap_fires_repulsion_escape_with_single_antiphase_rotation_and_no_double_negation() {
    use liminika_core::runtime::ResonantSignalSlice;
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("adapter");
    let geom = adapter.geometry().clone();
    let q_len = geom.q_rot_len();
    let num_heads = geom.num_q_heads;
    let rot_dim = geom.rotary_dim;
    let half = rot_dim / 2;

    // Construct an attractive reference vector at angle 0 and a Guard pole rotated by +0.30 rad
    // in HalfSplit RoPE coordinates across every head.
    let theta_guard = 0.30_f32;
    let mut non_guard_ref = vec![0.0_f32; q_len];
    let mut guard_vec = vec![0.0_f32; q_len];
    for h in 0..num_heads {
        let base = h * rot_dim;
        for i in 0..half {
            non_guard_ref[base + i] = 1.0;
            non_guard_ref[base + half + i] = 0.0;
            guard_vec[base + i] = theta_guard.cos();
            guard_vec[base + half + i] = theta_guard.sin();
        }
    }

    let guard_q7_vec = adapter
        .bake_layer_procrustes_q7(&non_guard_ref, &guard_vec)
        .expect("bake guard q7");
    let mut guard_q7 = [0_i8; MAX_ATTENTION_HEADS];
    guard_q7[..num_heads].copy_from_slice(&guard_q7_vec[..num_heads]);
    assert!(
        guard_q7[..num_heads].iter().all(|&v| v > 0),
        "positive Guard pole `non_guard -> guard` must have positive Q7 deltas: {:?}",
        &guard_q7[..num_heads]
    );

    // Pure Guard trap: empty Facts & Personas, single active GuardPhaseAnchor (`obstruction > 0.50`).
    let empty_facts: [ObjectiveFact<[i8; 64]>; 0] = [];
    let empty_personas: [SubjectivePersona<[i8; 64]>; 0] = [];
    let active_guard = [GuardPhaseAnchor::new(guard_q7, 0.80, 0.50)];
    let probe = adapter
        .bake_layer_procrustes_q7(&guard_vec, &guard_vec)
        .expect("probe");
    let fact_res = ResonantSignalSlice::from_q7_probe(&empty_facts, &probe);
    let persona_res = ResonantSignalSlice::from_q7_probe(&empty_personas, &probe);
    let guard_res = ResonantSignalSlice::from_q7_probe(&active_guard, &probe);
    let sys_res = *adapter.system_constraint();
    let composite = (sys_res, fact_res, persona_res, guard_res);

    // Verify core `apply_sys_and_guard_to_pair_frame` path (via `synthesize_gateway_with_core_layers`)
    // emits non-zero anti-phase (`-R`) with positive energy and zero double-negation.
    let mut stack = *adapter.core_layer_stack();
    let delta = *adapter.runtime_context().last_boundary_delta();
    let horizon = *adapter.last_horizon();
    let overlap = *adapter.last_overlap_report();
    let (core_frame, _) =
        composite.synthesize_gateway_with_core_layers(&mut stack, &delta, &horizon, &overlap);
    assert!(core_frame.is_repulsed());
    assert!(
        core_frame.effective_energy >= 0.55,
        "pure Guard trap must carry non-zero energy, got {}",
        core_frame.effective_energy
    );
    let dot_core: i64 = guard_q7[..num_heads]
        .iter()
        .zip(core_frame.phase_deltas[..num_heads].iter())
        .map(|(&g, &f)| i64::from(g) * i64::from(f))
        .sum();
    assert!(
        dot_core < 0,
        "core Guard frame must negate +R exactly once to -R (dot={dot_core})"
    );

    // Verify live RoPE steering via `orchestrate_layer_rope_with_substrate` selects `RepulsionEscape`
    // and rotates a query currently at `guard_vec` strictly AWAY from `guard_vec` toward `non_guard_ref`.
    let mut q_rot = guard_vec.clone();
    let cos_to_ref_before = TextAttractorCorpus::cosine_f32(&q_rot, &non_guard_ref);
    let cos_to_guard_before = TextAttractorCorpus::cosine_f32(&q_rot, &guard_vec);
    let mut trail = lmnk_pointer_trail_t::default();
    let (frame, _rep, decision) = adapter
        .orchestrate_layer_rope_with_substrate(
            LayerRopeStepInput {
                layer_idx: 16,
                signal: &composite,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut trail,
        )
        .expect("orchestrate pure guard")
        .expect("layer 16 active");
    assert_eq!(decision, StigmergicActionDecision::RepulsionEscape);
    assert!(frame.is_repulsed());
    let cos_to_ref_after = TextAttractorCorpus::cosine_f32(&q_rot, &non_guard_ref);
    let cos_to_guard_after = TextAttractorCorpus::cosine_f32(&q_rot, &guard_vec);
    assert!(
        cos_to_guard_after < cos_to_guard_before - 1e-4,
        "repulsion must rotate q_rot away from Guard: before={cos_to_guard_before} after={cos_to_guard_after}"
    );
    assert!(
        cos_to_ref_after > cos_to_ref_before + 1e-4,
        "repulsion must rotate q_rot toward safe non-Guard pole: before={cos_to_ref_before} after={cos_to_ref_after}"
    );
}

#[test]
fn guard_centroid_stays_uncontaminated_and_duplicate_repulsion_escalation_is_suppressed() {
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("adapter");
    let geom = adapter.geometry().clone();
    let num_layers = geom.num_layers;
    let q_len = geom.q_rot_len();
    let rot_dim = geom.rotary_dim;
    let half = rot_dim / 2;

    let make_angle_seq = |theta: f32| -> Vec<Vec<f32>> {
        let mut v = vec![0.0_f32; q_len];
        for h in 0..geom.num_q_heads {
            let base = h * rot_dim;
            for i in 0..half {
                v[base + i] = theta.cos();
                v[base + half + i] = theta.sin();
            }
        }
        vec![v; num_layers]
    };

    let fact_entry = TokenSequenceEntry {
        role: AttractorRole::Fact,
        text: "Safe factual anchor".to_string(),
        energy: 2.5,
        seq_len: 1,
        per_layer_token_qs: make_angle_seq(0.0),
        token_include_mask: None,
    };
    let guard_entry = TokenSequenceEntry {
        role: AttractorRole::Guard,
        text: "Forbidden guard claim".to_string(),
        energy: 2.0,
        seq_len: 1,
        per_layer_token_qs: make_angle_seq(0.40),
        token_include_mask: None,
    };

    // Compile with 1 Guard vs 6 identical Guards; Guard Q7 phase angle must stay invariant because
    // Guards bake against the non-Guard (`Fact + Persona`) centroid.
    let c1 = CompiledTokenCorpus::compile(
        &adapter,
        &[fact_entry.clone(), guard_entry.clone()],
        SequenceActivationPooler::default(),
        2,
        1,
    )
    .expect("compile 1 guard");
    let mut many_entries = vec![fact_entry];
    for _ in 0..6 {
        many_entries.push(guard_entry.clone());
    }
    let c6 = CompiledTokenCorpus::compile(
        &adapter,
        &many_entries,
        SequenceActivationPooler::default(),
        2,
        1,
    )
    .expect("compile 6 guards");
    let g1_q7 = c1
        .corpus
        .entries()
        .iter()
        .find(|e| e.role == AttractorRole::Guard)
        .expect("g1")
        .per_layer_q7[0];
    let g6_q7 = c6
        .corpus
        .entries()
        .iter()
        .find(|e| e.role == AttractorRole::Guard)
        .expect("g6")
        .per_layer_q7[0];
    assert_eq!(
        g1_q7, g6_q7,
        "Guard Procrustes angle must not shrink as Guard ratio increases"
    );

    // Verify TurnSteeringPlan exposes non-empty guard anchors ONLY on Guard turns and suppresses
    // duplicate repulsion escalation when `sync_core_layers_from_recall` already loaded a repulsion lane.
    use liminika_core::runtime::StandardPhaseKvGopCache;
    use liminika_core::stigmergy::StandardCognitiveBank;
    let mut kv_cache = StandardPhaseKvGopCache::new();
    let kv_policy = AdaptiveThresholdPolicy::default();
    let guard_query = make_angle_seq(0.40);
    let guard_plan = TurnOrchestrator::prepare_turn(
        &c1,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput::<AdaptiveThresholdPolicy, StandardCognitiveBank> {
            query_token_qs: &guard_query,
            query_seq_len: 1,
            token_include_mask: None,
            turn_index: 1,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None,
            layer_steer_mask_override: None,
        },
    )
    .expect("guard prepare_turn");
    assert!(
        guard_plan.is_guard_trap() || guard_plan.guard_conflict_margin() > 0.0,
        "guard query should trigger guard conflict"
    );
    assert!(
        !guard_plan.guard_anchors_per_layer()[0].is_empty(),
        "guard turn must populate dynamic GuardPhaseAnchor"
    );
    let ga = guard_plan.guard_anchors_per_layer()[0][0];
    assert!(ga.cohomology_obstruction >= 0.55);
    assert_eq!(ga.obstruction_threshold, 0.50);

    // Since `sync_core_layers_from_recall` already enabled a repulsion lane, escalation provider
    // must NOT be invoked for a duplicate repulsion lane during layer orchestration.
    assert_ne!(
        adapter.substrate().slot_repulsion_mask & adapter.substrate().slot_enabled_mask,
        0
    );
    let mut esc_provider = CorpusRecallEscalationProvider::new(&c1.corpus);
    let mut q_rot = guard_query[16].clone();
    let mut trail = lmnk_pointer_trail_t::default();
    let _ = adapter
        .orchestrate_layer_rope_with_spillover(
            LayerRopeStepInput {
                layer_idx: 16,
                signal: &ga,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot,
                logits_opt: None,
            },
            &mut esc_provider,
            &mut trail,
        )
        .expect("spillover guard");
    assert_eq!(
        esc_provider.escalation_calls, 0,
        "duplicate repulsion escalation must be suppressed when repulsion lane is already active"
    );

    // On a safe Fact turn, `guard_anchors_per_layer()` must be completely empty (zero misfire).
    let fact_query = make_angle_seq(0.0);
    let fact_plan = TurnOrchestrator::prepare_turn(
        &c1,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput::<AdaptiveThresholdPolicy, StandardCognitiveBank> {
            query_token_qs: &fact_query,
            query_seq_len: 1,
            token_include_mask: None,
            turn_index: 2,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None,
            layer_steer_mask_override: None,
        },
    )
    .expect("fact prepare_turn");
    assert!(!fact_plan.is_guard_trap());
    assert!(
        fact_plan
            .guard_anchors_per_layer()
            .iter()
            .all(Vec::is_empty),
        "non-Guard turn must leave guard_anchors_per_layer empty"
    );
}

#[test]
fn short_query_finite_sample_shrinkage_attenuates_loo_contrast_and_two_axis_isolates_exterior() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let num_heads = adapter.geometry().num_q_heads;
    let rot_dim = adapter.geometry().rotary_dim;
    let mut f0 = vec![0.0_f32; len];
    let mut f1 = vec![0.0_f32; len];
    let mut f2 = vec![0.0_f32; len];
    for h in 0..num_heads {
        let base = h * rot_dim;
        f0[base] = 1.0;
        f1[base + 1] = 1.0;
        f2[base + 2] = 1.0;
    }

    let mut corpus = TextAttractorCorpus::new();
    let id0 = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Known concept alpha",
            &vec![f0.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("f0");
    let id1 = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Known concept beta",
            &vec![f1.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("f1");
    let _id2 = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Known concept gamma",
            &vec![f2.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("f2");

    let mut query = vec![0.0_f32; len];
    for h in 0..num_heads {
        let base = h * rot_dim;
        query[base] = 0.45;
        query[base + 1] = 0.35;
        query[base + 6] = 0.82;
    }
    let policy = RecallPolicy {
        min_coherence: 0.25,
        boundary_margin: 0.04,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };

    // 1. Finite-sample shrinkage: short query (`S=2`) must yield strictly lower top-1 score
    //    than an otherwise identical long query (`S=32`) with the same LOO local scores.
    let short_obs = LocalChannelObservation {
        combined_scores: vec![0.30, 0.02, -0.05],
        q_to_d_scores: vec![0.30, 0.02, -0.05],
        d_to_q_scores: vec![0.10, 0.01, -0.02],
        token_deposits: Vec::new(),
        unexplained_token_ratio: 0.0,
        unexplained_corpus_token_ratio: 0.0,
        query_seq_len: 2,
    };
    let long_obs = LocalChannelObservation {
        query_seq_len: 32,
        ..short_obs.clone()
    };
    let short_recall = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(short_obs),
        )
        .expect("short recall");
    let long_recall = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(long_obs),
        )
        .expect("long recall");
    assert!(
        short_recall.top1_fact_score < long_recall.top1_fact_score,
        "short query S=2 ({}) must shrink LOO boost below long query S=32 ({})",
        short_recall.top1_fact_score,
        long_recall.top1_fact_score
    );

    // 2. Two-axis `SelfRef` vs `Exterior`: semi-known query bridging 2 known facts (`id0`, `id1`)
    //    with high `unexplained_token_ratio` (`0.74`, below the `0.80` foreign-OOD ceiling and
    //    above the `0.72` sharp-unimodal cutoff) must isolate into `exterior_sandbox_facts`
    //    with empty `recalled_facts` and disjoint IDs.
    let exterior_obs = LocalChannelObservation {
        combined_scores: vec![0.35, 0.30, -0.10],
        q_to_d_scores: vec![0.35, 0.30, -0.10],
        d_to_q_scores: vec![0.15, 0.12, -0.05],
        token_deposits: Vec::new(),
        unexplained_token_ratio: 0.74,
        unexplained_corpus_token_ratio: 0.74,
        query_seq_len: 16,
    };
    let ext_recall = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(exterior_obs),
        )
        .expect("exterior recall");
    assert!(
        ext_recall.recalled_facts.is_empty(),
        "semi-known exterior query must not populate SelfRef recalled_facts: \
         top1={} unexp={} bridge={} ext_mask={:#x} sandbox={} action={:?}",
        ext_recall.top1_fact_score,
        ext_recall.telemetry.unexplained_token_ratio,
        ext_recall.lacuna_partition.bridge_count,
        ext_recall.lacuna_partition.exterior_mask,
        ext_recall.lacuna_partition.is_sandbox_eligible(),
        ext_recall.action_decision
    );
    assert!(
        !ext_recall.exterior_sandbox_facts.is_empty(),
        "semi-known exterior query must populate exterior_sandbox_facts"
    );
    let ext_ids: Vec<u32> = ext_recall
        .exterior_sandbox_facts
        .iter()
        .map(|(e, _)| e.anchor_id)
        .collect();
    assert!(
        ext_ids.contains(&id0) && ext_ids.contains(&id1),
        "expected bridged anchors {id0}, {id1} in exterior_sandbox_facts, got {ext_ids:?}"
    );
    assert_eq!(
        ext_recall.lacuna_state,
        LacunaBoundaryState::LiminalExterior
    );
    assert!(ext_recall.telemetry.unexplained_token_ratio >= 0.60);

    // Pure SelfRef query (low unexplained_token_ratio) stays in `recalled_facts`.
    let self_obs = LocalChannelObservation {
        combined_scores: vec![0.45, 0.05, -0.10],
        q_to_d_scores: vec![0.45, 0.05, -0.10],
        d_to_q_scores: vec![0.25, 0.02, -0.05],
        token_deposits: Vec::new(),
        unexplained_token_ratio: 0.10,
        unexplained_corpus_token_ratio: 0.10,
        query_seq_len: 16,
    };
    let self_recall = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(self_obs),
        )
        .expect("self recall");
    assert!(
        !self_recall.recalled_facts.is_empty(),
        "pure SelfRef query must retain recalled_facts"
    );
    assert!(
        self_recall.exterior_sandbox_facts.is_empty(),
        "pure SelfRef query must keep exterior_sandbox_facts empty"
    );
}

#[test]
fn adapter_equilibrium_sync_restores_snapshot_preserves_host_spine_pin_floor_and_drains_fifo() {
    let geom = sample_gqa_partial_rope();
    let mut session1 = ModelSteeringAdapter::new(geom.clone()).expect("session1");
    session1.configure_ego_from_preset(LiminalEgoPreset::Medium);

    // Trigger Slow-Loop crystallization into the bounded FIFO buffer.
    let mut node = liminika_core::drivers::DistortionDelta::new_empty(7, 707);
    node.hop_depth = 3;
    node.reinforce_count = 6;
    node.gain_q7 = 96;
    node.pheromone_q7 = 96;
    node.delta_phase_q7 = [32_i8; 64];
    node.flags = liminika_core::drivers::DISTORTION_DELTA_FLAG_ACTIVE
        | liminika_core::drivers::DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
    {
        let net = session1.runtime_context_mut().ego_differential_net_mut();
        net.consolidation_hop_limit = 2;
        net.set_node(0, node);
    }
    let horizon = EpistemicHorizonReading::default();
    let delta = LiminalBoundaryDelta::default();
    let overlap = OverlapDiscoveryReport::default();
    let ego = *session1.ego_state();
    for _ in 0..4 {
        session1
            .runtime_context_mut()
            .telemetry_bank_mut()
            .record_step(liminika_core::runtime::StepTelemetryInput {
                decision: StigmergicActionDecision::FullOrchestration,
                horizon: &horizon,
                delta: &delta,
                overlap: &overlap,
                ego: &ego,
                entropy_gate_mode: 0,
                slid_in_bits: 0,
                slid_out_bits: 0,
                escalated: false,
                qos_throttled: false,
            });
    }
    let adj = session1.apply_slow_loop_telemetry_epoch(4);
    assert!(adj.applied);
    assert_eq!(session1.crystallized_anchors().len(), 1);
    assert_eq!(session1.crystallized_anchors()[0].anchor_id, 707);

    // Sync into EquilibriumState and persist to .lmnkc.
    let cfg = EquilibriumConfig::default();
    let mut eq = EquilibriumState::new(cfg);
    let report = session1
        .sync_equilibrium_state(&mut eq)
        .expect("sync_equilibrium_state");
    assert_eq!(report.inserted_anchors, 1);
    assert!(session1.crystallized_anchors().is_empty());
    assert!(eq.pager().is_resident(707));
    assert!(eq.runtime_snapshot_payload().is_some());

    let lmnkc_bytes = eq.to_lmnkc_bytes(1_774_901_000).expect("to_lmnkc_bytes");
    let expected_host_pin_floor = session1.envelope().baseline_pinned_compute_mask;
    drop(session1);

    // Restore into a completely fresh session2 instance and verify host Spine pin floor immunity.
    let restored_eq = EquilibriumState::from_lmnkc_bytes(&lmnkc_bytes, cfg).expect("from_lmnkc");
    assert!(restored_eq.pager().is_resident(707));
    let mut session2 = ModelSteeringAdapter::new(geom).expect("session2");
    // Extra host-side pin bit on session2 must never be cleared by deserialized snapshot.
    session2
        .runtime_context_mut()
        .apply_host_spine_pin_floor(1_u64 << 10);
    let restored_snap = session2
        .restore_from_equilibrium(&restored_eq)
        .expect("restore_from_equilibrium");
    assert!(restored_snap);
    assert_eq!(
        session2.envelope().baseline_pinned_compute_mask & expected_host_pin_floor,
        expected_host_pin_floor
    );
    assert_ne!(
        session2.envelope().baseline_pinned_compute_mask & (1_u64 << 10),
        0,
        "deserialized snapshot must never weaken host Envelope pin floor"
    );
    assert!(session2.ego_state().is_active());
}

#[test]
fn lmnkp_immutable_source_discipline_survives_multi_gen_equilibrium_and_topology_rebake() {
    struct DeterministicSubEncoder {
        layers: usize,
        rot_len: usize,
    }
    impl SubtitleActivationEncoder for DeterministicSubEncoder {
        fn encode_subtitle_activations(
            &mut self,
            record: &SourceTextRecord,
        ) -> Result<Vec<Vec<f32>>, String> {
            let seed = record.text.bytes().fold(0x4242_u64, |acc, b| {
                acc.wrapping_mul(131).wrapping_add(u64::from(b))
            });
            Ok((0..self.layers)
                .map(|l| {
                    (0..self.rot_len)
                        .map(|d| {
                            (((seed ^ ((l as u64) << 7) ^ (d as u64)) % 100) as f32) * 0.01 - 0.5
                        })
                        .collect()
                })
                .collect())
        }
    }

    let geom_a = RopeGeometry {
        layout: RopeLayout::HalfSplit,
        head_dim: 64,
        rotary_dim: 32,
        num_q_heads: 8,
        num_kv_heads: 4,
        num_layers: 4,
        layer_rope_mask: 0x0F,
    };
    let adapter_a = ModelSteeringAdapter::new(geom_a.clone()).expect("adapter_a");
    let policy = RecallPolicy::default();
    let detector = AdaptiveGopCutDetector::default();
    let mut enc_a = DeterministicSubEncoder {
        layers: geom_a.num_layers,
        rot_len: geom_a.q_rot_len(),
    };
    let source_dsl =
        "persona LiminaCore {\n  fact \"zero-alloc core\"\n  fact \"immutable source\"\n}\n";
    let records = vec![
        SourceTextRecord {
            frame_index: 0,
            timestamp_epoch_sec: 1_774_900_000,
            role: AttractorRole::Fact,
            text: "LiminiKa keeps core zero-alloc and non-invasive.".to_string(),
            energy: 2.0,
        },
        SourceTextRecord {
            frame_index: 1,
            timestamp_epoch_sec: 1_774_900_010,
            role: AttractorRole::Fact,
            text: "Immutable source truth lives strictly in .lmnkp archives.".to_string(),
            energy: 1.8,
        },
    ];
    let ctx_a = RebakeContext {
        adapter: &adapter_a,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    let (tl_a, _) =
        LmnkTimeline::bake_from_subtitle_records(&records, &ctx_a, &mut enc_a).expect("bake a");

    // Persist .lmnkp archive + snapshot baseline bytes and CRC32s.
    let tmp_dir = std::env::temp_dir().join(format!("liminika_p2c_immut_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp_dir);
    let lmnkp_path = tmp_dir.join("canonical_source.lmnkp");
    let lmnkc_path = tmp_dir.join("runtime_equilibrium.lmnkc");
    let _ = tl_a
        .save_lmnkp_to_path(&lmnkp_path, Some(source_dsl))
        .expect("save .lmnkp");

    let orig_pkg_bytes = std::fs::read(&lmnkp_path).expect("read orig .lmnkp");
    let orig_pkg_crc = liminika_core::foundation::compute_crc32_ieee(&orig_pkg_bytes);
    let orig_view = LmnkPackageView::parse(&orig_pkg_bytes).expect("parse orig .lmnkp");
    let orig_entry_crcs: Vec<(String, u32, usize)> = orig_view
        .entries()
        .iter()
        .map(|e| (e.name.to_string(), e.crc32, e.payload.len()))
        .collect();
    assert_eq!(orig_view.source_lmnk(), Some(source_dsl.as_bytes()));

    // Verify timeline v2 with embedded RuntimeSnapshot remains backward compatible.
    let snap_a = adapter_a.export_runtime_snapshot().expect("snap_a");
    let tl_with_snap = tl_a
        .to_container_bytes_with_snapshot(LMNK_CONTAINER_FLAG_COMPILED_CACHE, Some(&snap_a))
        .expect("tl with snap");
    let decoded_tl = LmnkTimeline::from_bytes(&tl_with_snap).expect("v2 decode with snap");
    assert_eq!(decoded_tl, tl_a);
    let extracted_snap =
        LmnkTimeline::extract_runtime_snapshot_from_bytes(&tl_with_snap).expect("extract snap");
    assert!(extracted_snap.is_some());

    // Run 10 generations of EquilibriumState flat join + .lmnkc persistence from the .lmnkp seed.
    let owned_pkg = LmnkPackageOwned::load_from_path(&lmnkp_path).expect("load owned .lmnkp");
    let imported_tl =
        LmnkTimeline::import_lmnkp_bytes(&owned_pkg.clone().into_bytes()).expect("import .lmnkp");
    let eq_cfg = EquilibriumConfig::default();
    let mut eq = imported_tl
        .build_equilibrium_state(eq_cfg)
        .expect("build_equilibrium_state");
    eq.update_runtime_snapshot_from_container(&snap_a)
        .expect("attach snap");
    for generation in 1..=10_u64 {
        let pulse = imported_tl.build_equilibrium_state(eq_cfg).expect("pulse");
        let rep = eq.join(&pulse).expect("eq join");
        assert_eq!(rep.generation, generation);
        let _ = eq
            .save_lmnkc_to_path(&lmnkc_path, 1_774_900_000 + generation)
            .expect("save .lmnkc");
        eq = EquilibriumState::load_lmnkc_from_path(&lmnkc_path, eq_cfg).expect("load .lmnkc");
    }

    // Encounter a different topology_crc32: join rejects, host rebakes from immutable .lmnkp.
    let foreign_eq =
        EquilibriumState::with_topology_crc32(eq_cfg, CORE_TOPOLOGY.topology_crc32 ^ 0x55AA_1234);
    assert!(matches!(
        eq.join(&foreign_eq),
        Err(EquilibriumError::TopologyMismatch { .. })
    ));

    let mut geom_b = geom_a;
    geom_b.rotary_dim = 64;
    let adapter_b = ModelSteeringAdapter::new(geom_b.clone()).expect("adapter_b");
    let mut enc_b = DeterministicSubEncoder {
        layers: geom_b.num_layers,
        rot_len: geom_b.q_rot_len(),
    };
    let ctx_b = RebakeContext {
        adapter: &adapter_b,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    let rebaked_tl = match imported_tl
        .resume_or_rebake(&ctx_b, &mut enc_b, 1)
        .expect("rebake from .lmnkp")
    {
        TimelineResumeOutcome::Rebaked { timeline, .. } => timeline,
        TimelineResumeOutcome::FastResumed { .. } => panic!("expected rebake on altered geometry"),
    };
    let rebaked_eq = rebaked_tl
        .build_equilibrium_state(eq_cfg)
        .expect("rebaked eq");
    let _ = rebaked_eq
        .save_lmnkc_to_path(&lmnkc_path, 1_774_900_999)
        .expect("save rebaked .lmnkc");

    // Verify .lmnkp file and every embedded source/manifest/compiled entry are bit-for-bit identical.
    let after_pkg_bytes = std::fs::read(&lmnkp_path).expect("read after .lmnkp");
    let after_pkg_crc = liminika_core::foundation::compute_crc32_ieee(&after_pkg_bytes);
    assert_eq!(after_pkg_bytes, orig_pkg_bytes, ".lmnkp bytes mutated!");
    assert_eq!(after_pkg_crc, orig_pkg_crc, ".lmnkp CRC32 mutated!");
    let after_view = LmnkPackageView::parse(&after_pkg_bytes).expect("parse after .lmnkp");
    let after_entry_crcs: Vec<(String, u32, usize)> = after_view
        .entries()
        .iter()
        .map(|e| (e.name.to_string(), e.crc32, e.payload.len()))
        .collect();
    assert_eq!(after_entry_crcs, orig_entry_crcs);
    assert_eq!(after_view.source_lmnk(), Some(source_dsl.as_bytes()));

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
