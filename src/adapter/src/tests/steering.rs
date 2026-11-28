// name: src/adapter/src/tests/steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RoPE steering, gateway orchestration, and PhaseClamp tests.

use super::*;

#[test]
fn gqa_partial_rope_geometry_validates_and_skips_nope_layers() {
    let mut geom = sample_gqa_partial_rope();
    assert!(geom.validate().is_ok());
    assert_eq!(geom.q_rot_len(), 24 * 96);

    geom.layer_rope_mask &= !(1 << 5);
    assert!(geom.is_rope_layer(4));
    assert!(!geom.is_rope_layer(5));
}

#[test]
fn adapter_bakes_and_steers_half_split_and_interleaved() {
    for layout in [RopeLayout::HalfSplit, RopeLayout::Interleaved] {
        let mut geom = sample_gqa_partial_rope();
        geom.layout = layout;
        geom.layer_rope_mask &= !(1 << 1); // Layer 1 is NoPE

        let mut adapter = ModelSteeringAdapter::new(geom).expect("adapter init");
        assert!(adapter.residual_adapter_mut().validate().is_ok());
        let _ = adapter.entropy_controller_mut();

        let len = adapter.geometry().q_rot_len();
        let neg: Vec<f32> = (0..len).map(|i| ((i % 13) as f32) * 0.1 - 0.5).collect();
        let pos: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.15 - 0.3).collect();

        let q7 = adapter
            .bake_layer_procrustes_q7(&neg, &pos)
            .expect("bake should succeed");
        assert_eq!(q7.len(), 24);

        let mut q_layer0 = neg.clone();
        adapter
            .steer_layer_q_rot(0, &mut q_layer0, &q7, 0.10)
            .expect("steer layer 0");

        // NoPE layer 1 must remain untouched.
        let mut q_layer1 = neg.clone();
        adapter
            .steer_layer_q_rot(1, &mut q_layer1, &q7, 0.10)
            .expect("skip layer 1");
        assert_eq!(q_layer1, neg);
    }
}

#[test]
fn adapter_steps_entropy_gate_and_applies_rank1_residual() {
    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let mut logits = vec![4.0, 2.0, 1.0, 0.5];
    let state = adapter
        .step_entropy_logits(&mut logits)
        .expect("entropy gate step should succeed");
    assert!(state.sample_temperature > 0.0);

    let mut hidden = vec![1.0, 0.5, -0.5, 0.25];
    let u = vec![0.5, 0.5, 0.5, 0.5];
    let v = vec![1.0, 0.0, 0.0, 0.0];
    adapter
        .apply_rank1_hidden_fused(&mut hidden, &u, &v, 0.5)
        .expect("rank-1 fused should succeed");
    assert!((hidden[0] - 1.25).abs() < 1e-5);
}

#[test]
fn adapter_steers_via_gateway_trait_composition() {
    use liminika_core::runtime::{
        CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA, ObjectiveFact,
        SubjectivePersona,
    };

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let orig: Vec<f32> = (0..len).map(|i| ((i % 11) as f32) * 0.1 - 0.4).collect();
    let mut q_rot = orig.clone();
    let mut logits = vec![2.0, 1.0, 0.5, -0.5];
    let mut trail = lmnk_pointer_trail_t::default();

    let fact = ObjectiveFact::new([20i8; 64], 2.5, 0, 42);
    let persona = SubjectivePersona::new([-12i8; 64], 1.8, 0, 7);
    let composite_signal = (fact, persona);

    let (frame, report) = adapter
        .steer_layer_with_gateway(
            0,
            &composite_signal,
            &mut trail,
            128,
            &mut q_rot,
            Some(&mut logits),
        )
        .expect("gateway steering should succeed")
        .expect("layer 0 is active");

    assert!(frame.phase_clamp_rad > 0.0);
    assert_ne!(report.lineage_flags & CAUSAL_AXIS_OBJECTIVE_FACT, 0);
    assert_ne!(report.lineage_flags & CAUSAL_AXIS_SUBJECTIVE_PERSONA, 0);
    assert_ne!(q_rot, orig);
    assert_eq!(trail.step_count, 1);

    let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 0.0];
    let (_, gain) = adapter
        .apply_rank1_with_gateway(&composite_signal, &mut hidden, &u, &v)
        .expect("gateway rank-1 should succeed");
    assert!(gain > 0.0);
    assert!(hidden[0] > 10.0);
}

#[test]
fn adapter_resonates_arbitrary_slice_without_hardcoded_count() {
    use liminika_core::runtime::{ObjectiveFact, ResonantSignalSlice};

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let mut q_rot: Vec<f32> = (0..len).map(|i| ((i % 9) as f32) * 0.1 - 0.3).collect();
    let mut trail = lmnk_pointer_trail_t::default();

    // Arbitrary number of candidates (e.g. 3, not tied to 8) matching 24 active Q heads.
    let mut a1 = [0i8; 64];
    a1[..24].fill(60);
    let mut a2 = [0i8; 64];
    a2[..24].fill(-60);
    let signals = [
        ObjectiveFact::new(a1, 2.0, 0, 10),
        ObjectiveFact::new(a2, 2.0, 0, 20),
        ObjectiveFact::new([0i8; 64], 2.0, 0, 30),
    ];
    let probe = [60i8; 24];
    let resonant = ResonantSignalSlice::from_q7_probe(&signals, &probe);

    let (frame, _) = adapter
        .steer_layer_with_gateway(0, &resonant, &mut trail, 128, &mut q_rot, None)
        .expect("resonant steering should succeed")
        .expect("layer 0 active");

    assert_eq!(frame.target_anchor_id, 10);
    assert!(frame.phase_deltas[0] > 30);
}

#[test]
fn adapter_steers_via_substrate_and_builds_stigmergic_density() {
    use liminika_core::runtime::{ObjectiveFact, ResonantSignalSlice};

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let mut q_rot: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.1 - 0.25).collect();
    let orig = q_rot.clone();
    let mut trail = lmnk_pointer_trail_t::default();

    let mut a1 = [0i8; 64];
    a1[..24].fill(55);
    let signals = [ObjectiveFact::new(a1, 2.5, 0, 77)];
    let probe = [55i8; 24];
    let resonant = ResonantSignalSlice::from_q7_probe(&signals, &probe);

    let (frame, _) = adapter
        .steer_layer_with_substrate(0, &resonant, &mut trail, 128, &mut q_rot, None)
        .expect("substrate steering should succeed")
        .expect("layer 0 active");

    assert_eq!(frame.target_anchor_id, 77);
    assert_ne!(q_rot, orig);
    assert_eq!(adapter.substrate().step_counter, 1);
    assert!(trail.stigmergic_density > 0.0);
}

#[test]
fn substrate_orchestrator_gates_rope_and_rank1_by_stigmergic_density() {
    use liminika_core::runtime::{ObjectiveFact, ResonantSignalSlice};

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let mut trail = lmnk_pointer_trail_t::default();

    let mut a1 = [0i8; 64];
    a1[..24].fill(55);
    let signals = [ObjectiveFact::new(a1, 2.5, 0, 77)];
    let probe = [55i8; 24];
    let resonant = ResonantSignalSlice::from_q7_probe(&signals, &probe);

    let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 0.0];

    // First step from zero density crosses express_steer (0.35) but not rank1_inject (1.50).
    let mut q_rot1: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.1 - 0.25).collect();
    let orig1 = q_rot1.clone();
    let (_, _, d1) = adapter
        .orchestrate_layer_rope_with_substrate(
            LayerRopeStepInput {
                layer_idx: 0,
                signal: &resonant,
                entropy_h_tilde_opt: None,
                unresolved_damping_u7: 128,
                q_rot_inout: &mut q_rot1,
                logits_opt: None,
            },
            &mut trail,
        )
        .expect("step 1")
        .expect("active");
    assert_eq!(d1, StigmergicActionDecision::ExpressPhaseSteer);
    assert_ne!(q_rot1, orig1);
    assert!(
        adapter
            .orchestrate_layer_rank1_with_substrate(&resonant, &mut hidden, &u, &v)
            .expect("rank-1 gate")
            .is_none()
    );

    // Repeated resonant steps accumulate density above rank1_inject_density_thresh (1.50).
    let mut final_decision = d1;
    for _ in 0..8 {
        let mut q_rot_k: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.1 - 0.25).collect();
        let (_, _, dk) = adapter
            .orchestrate_layer_rope_with_substrate(
                LayerRopeStepInput {
                    layer_idx: 0,
                    signal: &resonant,
                    entropy_h_tilde_opt: None,
                    unresolved_damping_u7: 128,
                    q_rot_inout: &mut q_rot_k,
                    logits_opt: None,
                },
                &mut trail,
            )
            .expect("step k")
            .expect("active");
        final_decision = dk;
    }
    assert!(matches!(
        final_decision,
        StigmergicActionDecision::FullOrchestration
            | StigmergicActionDecision::CrystallizeMacroBasin
    ));
    let rank1_res = adapter
        .orchestrate_layer_rank1_with_substrate(&resonant, &mut hidden, &u, &v)
        .expect("rank-1 gate");
    assert!(rank1_res.is_some());
    assert!(hidden[0] > 10.0);
}

#[test]
fn cold_path_probe_calibrates_thresholds_and_rebinds_geometry() {
    let geom = sample_gqa_partial_rope();
    let mut adapter = ModelSteeringAdapter::new(geom).expect("phi4 geometry valid");
    adapter.set_layer_rope_mask(0xFFFF_0000);
    let q_len = adapter.geometry().q_rot_len();

    let ref_vec = vec![0.0_f32; q_len];
    let probe_vec: Vec<f32> = (0..q_len)
        .map(|i| if (i & 1) == 0 { 0.85 } else { 0.45 })
        .collect();
    let samples: Vec<&[f32]> = vec![probe_vec.as_slice(); 32];
    let refs: Vec<&[f32]> = vec![ref_vec.as_slice(); 32];

    let baseline = adapter
        .calibrate_manifold_probe(&samples, &refs, &[0.42, 0.46])
        .expect("calibration probe should succeed");
    assert_eq!(baseline.probe_epoch, 1);
    assert_eq!(baseline.active_heads, 24);
    assert_eq!(baseline.steered_layers, 16);
    assert!(baseline.baseline_entropy > 0.40);

    let low_load_rank1 = adapter.substrate().thresholds.rank1_inject_density_thresh;
    adapter.update_environment_telemetry(0.85, 0.70, 0x0F);
    let high_load_rank1 = adapter.substrate().thresholds.rank1_inject_density_thresh;
    assert!(high_load_rank1 > low_load_rank1);

    let mut alt_geom = sample_gqa_partial_rope();
    alt_geom.num_q_heads = 32;
    alt_geom.head_dim = 128;
    alt_geom.rotary_dim = 128;
    alt_geom.layer_rope_mask = 0xFFFF_FFFF;
    let new_epoch = adapter
        .rebind_geometry_cold_path(alt_geom)
        .expect("cold-path geometry rebind should succeed");
    assert_eq!(new_epoch, 2);
    assert_eq!(adapter.probe_baseline().active_heads, 32);
    assert_eq!(adapter.probe_baseline().steered_layers, 32);
}

#[test]
fn text_attractor_corpus_recalls_basin_and_detects_liminal_exterior_boundary() {
    let geom = sample_gqa_partial_rope();
    let mut adapter = ModelSteeringAdapter::new(geom).expect("phi4 geometry valid");
    adapter.set_layer_rope_mask(0xFFFF_0000);
    let q_len = adapter.geometry().q_rot_len();

    let ref_vec = vec![0.0_f32; q_len];
    let half = adapter.geometry().rotary_dim / 2;
    let rot = adapter.geometry().rotary_dim;
    let mut fact_a = vec![0.0_f32; q_len];
    let mut fact_b = vec![0.0_f32; q_len];
    let mut exterior_q = vec![0.0_f32; q_len];
    for h in 0..adapter.geometry().num_q_heads {
        let base = h * rot;
        for i in 0..half {
            let (sa, ca) = (0.35 + 0.02 * h as f32).sin_cos();
            let (sb, cb) = (-0.45 - 0.01 * h as f32).sin_cos();
            let (sv, cv) = (2.80 - 0.02 * h as f32).sin_cos();
            fact_a[base + i] = ca;
            fact_a[base + half + i] = sa;
            fact_b[base + i] = cb;
            fact_b[base + half + i] = sb;
            exterior_q[base + i] = cv;
            exterior_q[base + half + i] = sv;
        }
    }

    let refs: Vec<&[f32]> = vec![ref_vec.as_slice(); 32];
    let qa_slices: Vec<&[f32]> = vec![fact_a.as_slice(); 32];
    let qb_slices: Vec<&[f32]> = vec![fact_b.as_slice(); 32];
    let qv_slices: Vec<&[f32]> = vec![exterior_q.as_slice(); 32];

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Fact A",
            &qa_slices,
            &refs,
            2.5,
        )
        .expect("ingest A");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Fact B",
            &qb_slices,
            &refs,
            2.5,
        )
        .expect("ingest B");

    let policy = RecallPolicy {
        top_k_facts: 1,
        top_k_personas: 1,
        min_coherence: 0.50,
        boundary_margin: 0.05,
        ..RecallPolicy::default()
    };
    let recall_a = corpus
        .recall_for_query(&adapter, &qa_slices, &refs, &policy)
        .expect("recall A");
    assert_eq!(recall_a.lacuna_state, LacunaBoundaryState::InsideBasin);
    assert_eq!(recall_a.recalled_facts[0].0.text, "Fact A");

    let recall_lacuna = corpus
        .recall_for_query(&adapter, &qv_slices, &refs, &policy)
        .expect("recall lacuna");
    assert_eq!(
        recall_lacuna.lacuna_state,
        LacunaBoundaryState::LiminalExterior
    );
    assert!(recall_lacuna.recalled_facts.is_empty());
}

#[test]
fn dual_track_timeline_records_gop_seeks_zero_forward_and_roundtrips() {
    let geom = sample_gqa_partial_rope();
    let adapter = ModelSteeringAdapter::new(geom).expect("phi4 geometry");
    let q_len = adapter.geometry().q_rot_len();
    let ref_vec = vec![0.0_f32; q_len];
    let mut q_a = vec![0.0_f32; q_len];
    let mut q_b = vec![0.0_f32; q_len];
    let rot = adapter.geometry().rotary_dim;
    let half = rot / 2;
    for h in 0..adapter.geometry().num_q_heads {
        let base = h * rot;
        for i in 0..half {
            let (sa, ca) = (0.30 + 0.01 * h as f32).sin_cos();
            let (sb, cb) = (0.42 + 0.01 * h as f32).sin_cos();
            q_a[base + i] = ca;
            q_a[base + half + i] = sa;
            q_b[base + i] = cb;
            q_b[base + half + i] = sb;
        }
    }
    let refs: Vec<&[f32]> = vec![ref_vec.as_slice(); 32];
    let qa_slices: Vec<&[f32]> = vec![q_a.as_slice(); 32];
    let qb_slices: Vec<&[f32]> = vec![q_b.as_slice(); 32];

    let mut corpus = TextAttractorCorpus::new();
    let id1 = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Turn 0 fact",
            &qa_slices,
            &refs,
            2.0,
        )
        .expect("ingest 0");
    assert_eq!(id1, 1);

    let detector = AdaptiveGopCutDetector::default();
    let policy = RecallPolicy::default();
    let mut timeline = LmnkTimeline::new(LmnkTimeline::geometry_fingerprint(adapter.geometry()));

    // Frame 0: initial frame -> always emits I-Frame keyframe.
    let is_i0 = timeline
        .record_step(
            &TimelineStepContext {
                adapter: &adapter,
                corpus: &corpus,
                policy: &policy,
                detector: &detector,
            },
            SourceTextRecord {
                frame_index: 0,
                timestamp_epoch_sec: 1_700_000_000,
                role: AttractorRole::Fact,
                text: "Turn 0 fact".to_string(),
                energy: 2.0,
            },
            &FrameTransitionMetrics {
                phase_l1_delta: 0.0,
                lacuna_state: LacunaBoundaryState::InsideBasin,
                frames_since_keyframe: 0,
                new_cluster_spawned: false,
            },
        )
        .expect("frame 0");
    assert!(is_i0);

    // Frame 1: in-basin small phase delta -> emits P-Frame delta.
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Persona,
            "Turn 1 persona",
            &qb_slices,
            &refs,
            1.5,
        )
        .expect("ingest 1");
    let is_i1 = timeline
        .record_step(
            &TimelineStepContext {
                adapter: &adapter,
                corpus: &corpus,
                policy: &policy,
                detector: &detector,
            },
            SourceTextRecord {
                frame_index: 1,
                timestamp_epoch_sec: 1_700_000_010,
                role: AttractorRole::Persona,
                text: "Turn 1 persona".to_string(),
                energy: 1.5,
            },
            &FrameTransitionMetrics {
                phase_l1_delta: 0.08,
                lacuna_state: LacunaBoundaryState::InsideBasin,
                frames_since_keyframe: 1,
                new_cluster_spawned: false,
            },
        )
        .expect("frame 1");
    assert!(!is_i1);

    // Frame 2: liminal_exterior scene cut -> automatically emits a new I-Frame keyframe.
    let is_i2 = timeline
        .record_step(
            &TimelineStepContext {
                adapter: &adapter,
                corpus: &corpus,
                policy: &policy,
                detector: &detector,
            },
            SourceTextRecord {
                frame_index: 2,
                timestamp_epoch_sec: 1_700_000_020,
                role: AttractorRole::Guard,
                text: "Turn 2 scene cut".to_string(),
                energy: 2.0,
            },
            &FrameTransitionMetrics {
                phase_l1_delta: 0.10,
                lacuna_state: LacunaBoundaryState::LiminalExterior,
                frames_since_keyframe: 2,
                new_cluster_spawned: true,
            },
        )
        .expect("frame 2");
    assert!(is_i2);
    assert_eq!(timeline.keyframe_positions(), &[0, 2]);

    // Zero-forward seek to frame 1 reconstructs from keyframe 0 + 1 P-Frame.
    let seek1 = timeline.seek_zero_forward(1).expect("seek frame 1");
    assert_eq!(seek1.keyframe_index, 0);
    assert_eq!(seek1.p_frames_applied, 1);
    assert!((seek1.total_energy - 3.5).abs() < 1e-5);

    // Portable binary serialization roundtrip & subtitle track export.
    let bytes = timeline.to_bytes();
    let decoded = LmnkTimeline::from_bytes(&bytes).expect("decode dual-track");
    assert_eq!(decoded, timeline);
    let subtitles = decoded.export_subtitle_track(Some(1));
    assert_eq!(subtitles.len(), 2);
    assert_eq!(subtitles[0].text, "Turn 0 fact");
    assert_eq!(subtitles[1].text, "Turn 1 persona");

    // Disk persistence roundtrip inside repository-contained `containers/`.
    let repo_container = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("containers")
        .join("dual_track_test.lmnkc");
    let written = timeline
        .save_to_path(&repo_container)
        .expect("save .lmnkc to repository containers/");
    assert_eq!(written, bytes.len());
    let loaded = LmnkTimeline::load_from_path(&repo_container)
        .expect("load .lmnkc from repository containers/");
    assert_eq!(loaded, timeline);
    let _ = std::fs::remove_file(&repo_container);

    // Also verify automatic creation of missing nested parent directories when `.gitkeep` is absent.
    let nested_base = std::env::temp_dir().join(format!(
        "liminika_adapter_autocreated_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&nested_base);
    let nested_path = nested_base.join("containers").join("nested_test.lmnkc");
    assert!(!nested_base.exists());
    let nested_written = timeline
        .save_to_path(&nested_path)
        .expect("save .lmnkc should auto-create missing parent directories");
    assert_eq!(nested_written, bytes.len());
    let nested_loaded =
        LmnkTimeline::load_from_path(&nested_path).expect("load auto-created .lmnkc");
    assert_eq!(nested_loaded, timeline);
    let _ = std::fs::remove_dir_all(&nested_base);
}

#[test]
fn dual_track_timeline_persists_composes_and_updates_persona_circuit() {
    use crate::reflection::{
        AttractorTrajectoryEvaluator, ManifoldReflectionEvaluator, ReflectionPolicy,
        TrajectoryManifoldSnapshot,
    };
    use liminika_core::stigmergy::{
        CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger, StigmergicScaleGranularity,
    };

    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("geometry");
    let policy = RecallPolicy::default();
    let detector = AdaptiveGopCutDetector::default();
    let mut corpus = TextAttractorCorpus::new();
    let q_rot_len = adapter.geometry().num_q_heads * adapter.geometry().rotary_dim;
    let q_vec = vec![0.15_f32; q_rot_len];
    let r_vec = vec![0.0_f32; q_rot_len];
    let qb_slices = vec![q_vec.as_slice(); adapter.geometry().num_layers];
    let refs = vec![r_vec.as_slice(); adapter.geometry().num_layers];
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Persona,
            "Calm analytical persona",
            &qb_slices,
            &refs,
            1.8,
        )
        .expect("ingest persona");

    let mut base_circuit = CognitiveTrajectoryLedger::new();
    base_circuit.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::DecomposePeel,
        6,
    ));
    base_circuit.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::SynthesizeFuse,
        7,
    ));

    let mut tl_base = LmnkTimeline::new(LmnkTimeline::geometry_fingerprint(adapter.geometry()));
    tl_base.set_persona_circuit(base_circuit);
    tl_base
        .record_step(
            &TimelineStepContext {
                adapter: &adapter,
                corpus: &corpus,
                policy: &policy,
                detector: &detector,
            },
            SourceTextRecord {
                frame_index: 0,
                timestamp_epoch_sec: 1_700_000_000,
                role: AttractorRole::Persona,
                text: "Calm analytical persona".to_string(),
                energy: 1.8,
            },
            &FrameTransitionMetrics {
                phase_l1_delta: 0.0,
                lacuna_state: LacunaBoundaryState::InsideBasin,
                frames_since_keyframe: 0,
                new_cluster_spawned: true,
            },
        )
        .expect("record I-frame");

    let encoded = tl_base.to_bytes();
    let decoded = LmnkTimeline::from_bytes(&encoded).expect("decode .lmnkc");
    assert_eq!(decoded.persona_circuit(), Some(base_circuit));

    // Compose with a second timeline's overlay circuit and apply reflection plasticity.
    let mut overlay_circuit = CognitiveTrajectoryLedger::new();
    overlay_circuit.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::PointerBit,
        CognitiveOpKind::RouteSubHead,
        5,
    ));
    let mut tl_overlay = LmnkTimeline::new(tl_base.model_fingerprint());
    tl_overlay.set_persona_circuit(overlay_circuit);

    let mut composed = decoded
        .compose_persona_circuits(&tl_overlay, 64)
        .expect("composed circuit");
    assert_ne!(composed, base_circuit);

    let recall = corpus
        .recall_for_query(&adapter, &qb_slices, &refs, &policy)
        .expect("recall");
    let query_snap = TrajectoryManifoldSnapshot::from_recall(&recall);
    let draft_snap = query_snap.clone();
    let critique = ManifoldReflectionEvaluator.assess_trajectory(
        &query_snap,
        &draft_snap,
        &policy,
        &ReflectionPolicy::default(),
    );
    ManifoldReflectionEvaluator.update_persona_circuit(&mut composed, &critique);
    assert!(composed.latest_step().is_some());
}

#[test]
fn dual_track_timeline_fast_resumes_and_rebakes_across_models() {
    struct DeterministicHashEncoder {
        num_layers: usize,
        rotary_dim: usize,
    }

    impl SubtitleActivationEncoder for DeterministicHashEncoder {
        fn encode_subtitle_activations(
            &mut self,
            record: &SourceTextRecord,
        ) -> Result<Vec<Vec<f32>>, String> {
            let seed = record
                .text
                .bytes()
                .fold(0x9E37_79B9_7F4A_7C15_u64, |acc, b| {
                    acc.wrapping_mul(131).wrapping_add(u64::from(b))
                });
            let mut layers = Vec::with_capacity(self.num_layers);
            for l in 0..self.num_layers {
                let mut v = Vec::with_capacity(self.rotary_dim);
                for d in 0..self.rotary_dim {
                    let phase = ((seed ^ ((l as u64) << 8) ^ (d as u64)) % 100) as f32 * 0.01;
                    v.push((phase - 0.5) * 2.0);
                }
                layers.push(v);
            }
            Ok(layers)
        }
    }

    let src_adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("src init");
    let len = src_adapter.geometry().q_rot_len();
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); 32];
    let q_a: Vec<f32> = (0..len)
        .map(|i| if i % 2 == 0 { 0.7 } else { -0.3 })
        .collect();
    let qa_slices: Vec<&[f32]> = vec![q_a.as_slice(); 32];

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &src_adapter,
            AttractorRole::Fact,
            "Fact zero",
            &qa_slices,
            &refs,
            1.5,
        )
        .expect("ingest 0");
    corpus
        .ingest_text_activations(
            &src_adapter,
            AttractorRole::Persona,
            "Persona one",
            &qa_slices,
            &refs,
            1.2,
        )
        .expect("ingest 1");

    let detector = AdaptiveGopCutDetector::default();
    let policy = RecallPolicy::default();
    let mut timeline =
        LmnkTimeline::new(LmnkTimeline::geometry_fingerprint(src_adapter.geometry()));
    let step_ctx = TimelineStepContext {
        adapter: &src_adapter,
        corpus: &corpus,
        policy: &policy,
        detector: &detector,
    };
    timeline
        .record_step(
            &step_ctx,
            SourceTextRecord {
                frame_index: 0,
                timestamp_epoch_sec: 1_700_000_000,
                role: AttractorRole::Fact,
                text: "Fact zero".to_string(),
                energy: 1.5,
            },
            &FrameTransitionMetrics {
                phase_l1_delta: 0.0,
                lacuna_state: LacunaBoundaryState::InsideBasin,
                frames_since_keyframe: 0,
                new_cluster_spawned: false,
            },
        )
        .expect("step 0");
    timeline
        .record_step(
            &step_ctx,
            SourceTextRecord {
                frame_index: 1,
                timestamp_epoch_sec: 1_700_000_010,
                role: AttractorRole::Persona,
                text: "Persona one".to_string(),
                energy: 1.2,
            },
            &FrameTransitionMetrics {
                phase_l1_delta: 0.05,
                lacuna_state: LacunaBoundaryState::InsideBasin,
                frames_since_keyframe: 1,
                new_cluster_spawned: false,
            },
        )
        .expect("step 1");

    // 1. Same model geometry -> FastResumed (zero-forward seek).
    let mut same_encoder = DeterministicHashEncoder {
        num_layers: src_adapter.geometry().num_layers,
        rotary_dim: len,
    };
    let same_ctx = RebakeContext {
        adapter: &src_adapter,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    match timeline
        .resume_or_rebake(&same_ctx, &mut same_encoder, 1)
        .expect("fast resume")
    {
        TimelineResumeOutcome::FastResumed { seek_state } => {
            assert_eq!(seek_state.target_frame_index, 1);
        }
        TimelineResumeOutcome::Rebaked { .. } => {
            panic!("expected FastResumed for matching geometry");
        }
    }

    // 2. Different model geometry -> Rebaked automatically from Subtitle Track.
    let mut alt_geom = sample_gqa_partial_rope();
    alt_geom.head_dim = 128;
    alt_geom.rotary_dim = 128;
    let alt_adapter = ModelSteeringAdapter::new(alt_geom).expect("alt init");
    let alt_len = alt_adapter.geometry().q_rot_len();
    let mut alt_encoder = DeterministicHashEncoder {
        num_layers: alt_adapter.geometry().num_layers,
        rotary_dim: alt_len,
    };
    let alt_ctx = RebakeContext {
        adapter: &alt_adapter,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    match timeline
        .resume_or_rebake(&alt_ctx, &mut alt_encoder, 1)
        .expect("cross-model rebake")
    {
        TimelineResumeOutcome::Rebaked {
            timeline: rebaked_tl,
            corpus: rebaked_corpus,
            seek_state,
            source_fingerprint,
            target_fingerprint,
            rebaked_frames,
        } => {
            assert_ne!(source_fingerprint, target_fingerprint);
            assert_eq!(rebaked_frames, 2);
            assert_eq!(rebaked_corpus.entries().len(), 2);
            assert_eq!(rebaked_tl.model_fingerprint(), target_fingerprint);
            assert_eq!(seek_state.target_frame_index, 1);
        }
        TimelineResumeOutcome::FastResumed { .. } => {
            panic!("expected Rebaked for mismatched geometry");
        }
    }
}

#[test]
fn coherence_guard_gate_flags_low_fact_coherence_as_liminal_exterior() {
    let gate = CoherenceGuardGate;
    let policy = RecallPolicy {
        min_coherence: 0.16,
        boundary_margin: 0.05,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    let decision = gate.evaluate(
        &RecallBoundaryMetrics {
            top1_fact_score: 0.077,
            top2_fact_score: 0.043,
            top1_persona_score: 0.240,
            max_guard_score: 0.100,
            guard_conflict_margin: -0.140,
        },
        &policy,
    );
    assert!(decision.suppress_fact_injection);
    assert_eq!(decision.lacuna_state, LacunaBoundaryState::LiminalExterior);
}

#[test]
fn coherence_guard_gate_detects_sharp_winner_liminal_without_weakening_ood_liminal_lacuna() {
    let gate = CoherenceGuardGate;
    let policy = RecallPolicy {
        min_coherence: 0.19,
        boundary_margin: 0.06,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    // Sub-threshold winner (`0.17 < min_coherence 0.19`) must NOT bypass `min_coherence`
    // (the legacy 0.85x inversion is abolished; OOD noise below floor stays LiminalExterior).
    let sub_floor = gate.evaluate(
        &RecallBoundaryMetrics {
            top1_fact_score: 0.17,
            top2_fact_score: 0.06,
            top1_persona_score: 0.05,
            max_guard_score: 0.02,
            guard_conflict_margin: -0.15,
        },
        &policy,
    );
    assert!(sub_floor.suppress_fact_injection);
    assert_eq!(sub_floor.lacuna_state, LacunaBoundaryState::LiminalExterior);

    // Above-threshold winner with narrow margin (`0.20 - 0.16 < boundary_margin 0.06`) enters
    // `LiminalBoundary` while allowing fact injection.
    let liminal = gate.evaluate(
        &RecallBoundaryMetrics {
            top1_fact_score: 0.20,
            top2_fact_score: 0.16,
            top1_persona_score: 0.05,
            max_guard_score: 0.02,
            guard_conflict_margin: -0.18,
        },
        &policy,
    );
    assert!(!liminal.suppress_fact_injection);
    assert_eq!(liminal.lacuna_state, LacunaBoundaryState::LiminalBoundary);

    // Flat out-of-domain noise remains strictly Liminal.
    let flat_ood = gate.evaluate(
        &RecallBoundaryMetrics {
            top1_fact_score: 0.15,
            top2_fact_score: 0.12,
            top1_persona_score: 0.05,
            max_guard_score: 0.02,
            guard_conflict_margin: -0.13,
        },
        &policy,
    );
    assert!(flat_ood.suppress_fact_injection);
    assert_eq!(flat_ood.lacuna_state, LacunaBoundaryState::LiminalExterior);

    // Single-token OOD outlier noise below 0.85 * min_coherence must not bleed into LiminalBoundary.
    let ood_outlier = gate.evaluate(
        &RecallBoundaryMetrics {
            top1_fact_score: 0.141,
            top2_fact_score: 0.037,
            top1_persona_score: 0.311,
            max_guard_score: 0.01,
            guard_conflict_margin: -0.301,
        },
        &policy,
    );
    assert!(ood_outlier.suppress_fact_injection);
    assert_eq!(
        ood_outlier.lacuna_state,
        LacunaBoundaryState::LiminalExterior
    );
}

#[test]
fn perspective_hubness_damps_dense_hub_and_preserves_peripheral_attractor() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); 32];

    // Construct a crowded hub cluster (#1..#3) near axis 0 and a distinct peripheral
    // attractor (#4) along orthogonal axis 1.
    let mut hub_a = vec![0.0_f32; len];
    let mut hub_b = vec![0.0_f32; len];
    let mut hub_c = vec![0.0_f32; len];
    let mut periph = vec![0.0_f32; len];
    hub_a[0] = 1.0;
    hub_a[2] = 0.25;
    hub_b[0] = 0.95;
    hub_b[4] = 0.30;
    hub_c[0] = 0.95;
    hub_c[6] = 0.30;
    periph[1] = 1.0;

    let mut corpus = TextAttractorCorpus::new();
    let id_hub = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Hub attractor A",
            &vec![hub_a.as_slice(); 32],
            &refs,
            1.5,
        )
        .expect("hub a");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Hub attractor B",
            &vec![hub_b.as_slice(); 32],
            &refs,
            1.5,
        )
        .expect("hub b");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Hub attractor C",
            &vec![hub_c.as_slice(); 32],
            &refs,
            1.5,
        )
        .expect("hub c");
    let id_periph = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Peripheral attractor",
            &vec![periph.as_slice(); 32],
            &refs,
            1.5,
        )
        .expect("periph");

    // Query has equal raw projection on hub_a's axis 0 and peripheral's axis 1 (slightly
    // higher raw weight toward hub), yet from the peripheral attractor's perspective the query
    // stands out sharply whereas hub_a sits inside a crowded neighborhood.
    let mut query = vec![0.0_f32; len];
    query[0] = 0.62;
    query[1] = 0.58;
    let q_slices = vec![query.as_slice(); 32];
    let policy = RecallPolicy {
        min_coherence: 0.30,
        boundary_margin: 0.02,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    let recall = corpus
        .recall_for_query(&adapter, &q_slices, &refs, &policy)
        .expect("recall");
    assert_ne!(id_hub, id_periph);
    assert_eq!(recall.recalled_facts[0].0.anchor_id, id_periph);
    assert!(recall.min_local_lacuna_depth < 0.05);
}

#[test]
fn layer_angle_perspective_separates_specialized_late_layer_attractor() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    // Hub shares strong early-layer alignment (layers 0..24), while specialized attractor B
    // dominates the final 8 semantic layers.
    let mut hub_layers = vec![vec![0.0_f32; len]; num_layers];
    let mut spec_layers = vec![vec![0.0_f32; len]; num_layers];
    let mut query_layers = vec![vec![0.0_f32; len]; num_layers];
    for l in 0..num_layers {
        if l < 24 {
            hub_layers[l][0] = 1.0;
            spec_layers[l][1] = 1.0;
            query_layers[l][0] = 0.65;
            query_layers[l][1] = 0.55;
        } else {
            hub_layers[l][0] = 0.2;
            hub_layers[l][2] = 0.98;
            spec_layers[l][3] = 1.0;
            query_layers[l][3] = 1.0;
        }
    }
    let hub_refs: Vec<&[f32]> = hub_layers.iter().map(Vec::as_slice).collect();
    let spec_refs: Vec<&[f32]> = spec_layers.iter().map(Vec::as_slice).collect();
    let q_refs: Vec<&[f32]> = query_layers.iter().map(Vec::as_slice).collect();

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Early hub",
            &hub_refs,
            &refs,
            1.5,
        )
        .expect("ingest hub");
    let id_spec = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Late specialist",
            &spec_refs,
            &refs,
            1.5,
        )
        .expect("ingest spec");

    let policy = RecallPolicy {
        min_coherence: 0.20,
        boundary_margin: 0.01,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    let recall = corpus
        .recall_for_query(&adapter, &q_refs, &refs, &policy)
        .expect("recall");
    assert_eq!(recall.recalled_facts[0].0.anchor_id, id_spec);
}

#[test]
fn coalition_stratum_pipeline_supports_dynamic_add_remove_and_substrate_arbitration() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); 32];

    let mut fact_vec = vec![0.0_f32; len];
    let mut guard_vec = vec![0.0_f32; len];
    fact_vec[0] = 1.0;
    guard_vec[1] = 1.0;

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Coherent fact",
            &vec![fact_vec.as_slice(); 32],
            &refs,
            1.5,
        )
        .expect("fact");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Guard,
            "Contrastive guard",
            &vec![guard_vec.as_slice(); 32],
            &refs,
            1.5,
        )
        .expect("guard");

    let policy = RecallPolicy {
        min_coherence: 0.30,
        boundary_margin: 0.05,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };

    // 1. Default 5-stratum pipeline can dynamically remove & re-insert intermediate strata.
    let mut pipeline = StratumPipeline::default();
    assert_eq!(
        pipeline.stratum_names(),
        vec![
            "local_feature_worker",
            "head_angle_worker",
            "local_token_resonance",
            "perspective_hubness_liminal_lacuna",
            "substrate_boundary",
        ]
    );
    let removed = pipeline.remove_stratum("perspective_hubness_liminal_lacuna");
    assert!(removed.is_some());
    assert_eq!(
        pipeline.stratum_names(),
        vec![
            "local_feature_worker",
            "head_angle_worker",
            "local_token_resonance",
            "substrate_boundary",
        ]
    );

    // Even with the perspective stratum removed, StigmergicSubstrate still arbitrates cleanly.
    let recall_fact = corpus
        .recall_for_query_with_strata(
            &adapter,
            &vec![fact_vec.as_slice(); 32],
            &refs,
            &policy,
            &pipeline,
        )
        .expect("fact recall");
    assert_eq!(
        recall_fact.action_decision,
        StigmergicActionDecision::CrystallizeMacroBasin
    );

    // Re-insert the perspective stratum at index 3 and verify Guard dominance triggers RepulsionEscape.
    if let Some(mid) = removed {
        pipeline.insert_stratum(3, mid);
    }
    assert_eq!(pipeline.stratum_names().len(), 5);
    let recall_guard = corpus
        .recall_for_query_with_strata(
            &adapter,
            &vec![guard_vec.as_slice(); 32],
            &refs,
            &policy,
            &pipeline,
        )
        .expect("guard recall");
    assert_eq!(
        recall_guard.action_decision,
        StigmergicActionDecision::RepulsionEscape
    );
    assert_eq!(
        recall_guard.lacuna_state,
        LacunaBoundaryState::LiminalExterior
    );
}

#[test]
fn head_angle_worker_stratum_separates_specialized_head_attractor() {
    // Use sample GQA partial-RoPE geometry where head 0 is shared noise and head 1 holds the specialized signal.
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let rotary_dim = len / 24;
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let mut spec_a = vec![0.0_f32; len];
    let mut spec_b = vec![0.0_f32; len];
    for i in 0..4 {
        spec_a[i] = 0.25;
        spec_b[i] = 0.25;
    }
    spec_a[rotary_dim] = 1.0;
    spec_b[rotary_dim + 1] = 1.0;

    let mut corpus = TextAttractorCorpus::new();
    let id_a = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "specialized head A",
            &vec![spec_a.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("ingest spec_a");
    let _id_b = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "specialized head B",
            &vec![spec_b.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("ingest spec_b");

    let policy = RecallPolicy {
        min_coherence: 0.20,
        boundary_margin: 0.01,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    let recall = corpus
        .recall_for_query_with_strata(
            &adapter,
            &vec![spec_a.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
        )
        .expect("head-angle recall");
    assert_eq!(
        recall.recalled_facts.first().map(|(e, _)| e.anchor_id),
        Some(id_a)
    );
    assert!(recall.top1_fact_score > recall.top2_fact_score);
}

#[test]
fn persona_recall_survives_factual_liminal_exterior_and_routes_via_gateway_frame() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let mut fact_vec = vec![0.0_f32; len];
    let mut persona_vec = vec![0.0_f32; len];
    fact_vec[0] = 1.0;
    persona_vec[1] = 1.0;

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Orthogonal fact",
            &vec![fact_vec.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("fact");
    let id_persona = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Persona,
            "Target persona",
            &vec![persona_vec.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("persona");

    let policy = RecallPolicy {
        min_coherence: 0.30,
        boundary_margin: 0.02,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    let recall = corpus
        .recall_for_query_with_strata(
            &adapter,
            &vec![persona_vec.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
        )
        .expect("persona recall");

    assert!(recall.recalled_facts.is_empty());
    assert_eq!(
        recall.recalled_personas.first().map(|(e, _)| e.anchor_id),
        Some(id_persona)
    );
    assert_eq!(recall.gateway_frame.dominant_slot, ATTRACTOR_SLOT_PERSONA);
    assert_eq!(
        recall.action_decision,
        StigmergicActionDecision::ExpressPhaseSteer
    );
}

#[test]
fn local_token_resonance_arbitrates_channel_disagreement_via_subhead_route() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let mut f1 = vec![0.0_f32; len];
    let mut f2 = vec![0.0_f32; len];
    let mut f3 = vec![0.0_f32; len];
    f1[0] = 0.9;
    f1[1] = 0.4;
    f2[0] = 0.85;
    f2[2] = 0.5;
    f3[0] = 0.85;
    f3[3] = 0.5;

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Ambiguous hub fact",
            &vec![f1.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("f1");
    let id_target = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Target specialist fact",
            &vec![f2.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("f2");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Peer fact",
            &vec![f3.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("f3");

    let mut query = vec![0.0_f32; len];
    query[0] = 1.0;
    let policy = RecallPolicy {
        min_coherence: 0.30,
        boundary_margin: 0.02,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    // Local channel sharply selects id_target while global channel prefers f1 slightly -> disagreement triggers
    // SubHeadRoute.
    let recall = corpus
        .recall_for_query_with_strata_and_local_scores(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(&[0.10, 0.85, 0.12]),
        )
        .expect("disagreement recall");
    assert!(recall.telemetry.channel_disagreement);
    assert_eq!(
        recall.action_decision,
        StigmergicActionDecision::SubHeadRoute
    );
    assert_eq!(
        recall.recalled_facts.first().map(|(e, _)| e.anchor_id),
        Some(id_target)
    );
}
