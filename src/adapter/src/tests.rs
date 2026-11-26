// name: src/adapter/src/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

const fn sample_gqa_partial_rope() -> RopeGeometry {
    RopeGeometry::new(RopeLayout::HalfSplit, 128, 96, 24, 8, 32, 0xFFFF_FFFF)
}

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
            0,
            &resonant,
            &mut trail,
            None,
            128,
            &mut q_rot1,
            None,
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
                0,
                &resonant,
                &mut trail,
                None,
                128,
                &mut q_rot_k,
                None,
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
    // Local channel sharply selects id_target while global channel prefers f1 slightly -> disagreement triggers SubHeadRoute.
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
    assert!(adapter.ego_state().transgression_bias_q7 > 0);
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
        .with_transgression_q7(84)
        .with_devotion_q7(72);
    adapter.blend_ego_intent_axes(&overlay_axes, 64);
    assert!(adapter.ego_state().transgression_bias_q7 > 0);

    adapter.overlay_ego_from_natural_language("curious rebel explorer");
    assert!(adapter.ego_state().counter_basin_thrust_q7 > 32);

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
            .with_transgression_q7(80)
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

    // 2. Substrate RoPE + rank-1 residual orchestration wires Core Layers and persists Ego intervention on the residual path.
    let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 0.0];
    for _ in 0..8 {
        let mut q_tmp = q_ego.clone();
        let _ = ego_adapter
            .orchestrate_layer_rope_with_substrate(
                0,
                &fact,
                &mut trail_ego,
                None,
                128,
                &mut q_tmp,
                None,
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
            0,
            &signal,
            &mut custom_slot,
            &mut escalation,
            &mut trail,
            None,
            128,
            &mut q_rot,
            None,
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
            0,
            &signal,
            &mut custom_slot,
            &mut escalation,
            &mut trail,
            None,
            128,
            &mut q_rot,
            None,
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
            0,
            &signal,
            &mut custom_slot,
            &mut escalation,
            &mut trail,
            None,
            128,
            &mut q_rot,
            None,
        )
        .expect("recover step 1");
    assert_eq!(adapter.active_horizon_slots_mask(), custom_slot.slot_bit);
    assert_eq!(custom_slot.retractions, 0);

    let _ = adapter
        .orchestrate_layer_rope_with_horizon_slots(
            0,
            &signal,
            &mut custom_slot,
            &mut escalation,
            &mut trail,
            None,
            128,
            &mut q_rot,
            None,
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
            0,
            &signal,
            &mut no_esc,
            &mut trail,
            None,
            128,
            &mut q_rot,
            None,
        )
        .expect("step 1");
    assert_eq!(adapter.active_horizon_slots_mask(), 1 << 0);
    assert_eq!(adapter.residual_spillover().spillover_modulations, 0);

    // Step 2: Sustained conflict triggers ResidualSpilloverHorizonSlot alongside guard slot.
    let _ = adapter
        .orchestrate_layer_rope_with_spillover(
            0,
            &signal,
            &mut no_esc,
            &mut trail,
            None,
            128,
            &mut q_rot,
            None,
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
            0,
            &signal,
            &mut no_esc,
            &mut trail,
            None,
            128,
            &mut q_rot,
            None,
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
    let base_thrust = adapter.ego_state().counter_basin_thrust_q7;

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
    assert!(adapter.ego_state().counter_basin_thrust_q7 >= 48);

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
    assert_eq!(
        adapter.ego_state().counter_basin_thrust_q7,
        base_thrust >> 1
    );

    // 3. Returning to Aligned restores configured Ego baseline and Mermaid renders LayerDistortion.
    let aligned = LayerDistortionReport::default();
    adapter.apply_distortion_ego_feedback(&aligned);
    assert_eq!(adapter.ego_state().counter_basin_thrust_q7, base_thrust);
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
                0,
                &sig,
                &mut trail,
                Some(0.45),
                128,
                &mut q,
                None,
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
                layer_idx, &sig, &mut esc, &mut trail, None, 128, &mut q, None,
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
            16,
            &sig,
            &mut trail_overlap,
            None,
            128,
            &mut q,
            None,
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
        .orchestrate_layer_rope_with_substrate(16, &sig, &mut trail_no_sb, None, 128, &mut q, None)
        .expect("no-sandbox layer step");
    assert_eq!(
        core_head_with_sandbox,
        trail_no_sb.accumulated_phase_delta[0]
    );
    assert!(trail_overlap.accumulated_phase_delta[40] > trail_no_sb.accumulated_phase_delta[40]);
}

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
                0,
                &sig,
                &mut trail,
                Some(0.90),
                128,
                &mut q,
                None,
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
    assert!(rebel.counter_basin_thrust_q7 > calm.counter_basin_thrust_q7);
    assert!(rebel.transgression_bias_q7 > calm.transgression_bias_q7);
    assert!(overlay.is_active());
    assert!(overlay.transgression_bias_q7 > med.transgression_bias_q7);
    assert!(symbolic.is_active());
    assert_ne!(symbolic.ego_signature_hash, 0);
    assert!(symbolic.counter_basin_thrust_q7 >= 24);

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
        adapter.system_spine().pinned_compute_mask,
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
        adapter.system_spine().pinned_compute_mask,
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
        adapter.system_spine().pinned_compute_mask,
        adapter.compute_pool().active_mask
    );
    let mut d_guard = adapter.step_compute_dispatch(surge);
    assert!(!adapter.evaluate_whole_token_bypass(calm_obs, &mut d_guard));

    let _ = adapter
        .sync_core_layers_from_recall(&calm_recall, None)
        .expect("sync calm recall");
    assert_eq!(
        adapter.system_spine().pinned_compute_mask,
        expected_baseline_pins
    );

    // Rebind cold-path resets baseline pins to the new geometry's head/tail boundaries.
    let mut small_geom = sample_gqa_partial_rope();
    small_geom.num_layers = 8;
    small_geom.layer_rope_mask = 0xFF;
    let _ = adapter
        .rebind_geometry_cold_path(small_geom)
        .expect("rebind");
    assert_eq!(adapter.system_spine().pinned_compute_mask, 0x81_u64);
}

#[test]
fn adapter_embeds_boxed_runtime_context_with_drop_and_spine_pin_floor() {
    use liminika_core::runtime::RuntimeContext;

    assert_eq!(core::mem::size_of::<RuntimeContext>(), 5120);
    assert_eq!(core::mem::align_of::<RuntimeContext>(), 64);
    assert!(!core::mem::needs_drop::<RuntimeContext>());
    assert!(core::mem::needs_drop::<ModelSteeringAdapter>());

    let mut adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let baseline_pins = 0x0000_000F_u64 | (0x0000_000F_u64 << 28);

    // Embedded `RuntimeContext` must mirror adapter initialization parity and pin floor.
    assert_eq!(
        adapter.runtime_context().system_spine().pinned_compute_mask,
        baseline_pins
    );
    assert_eq!(
        adapter.system_spine().pinned_compute_mask,
        adapter.runtime_context().system_spine().pinned_compute_mask
    );

    // Verify `apply_host_spine_pin_floor` is enforced on the embedded context.
    adapter
        .runtime_context_mut()
        .system_spine_mut()
        .pinned_compute_mask = 0;
    adapter
        .runtime_context_mut()
        .apply_host_spine_pin_floor(baseline_pins);
    assert_eq!(
        adapter.runtime_context().system_spine().pinned_compute_mask,
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
            0,
            &sig,
            &mut trail,
            Some(0.35),
            128,
            &mut q_rot,
            None,
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
            adapter.system_spine().pinned_compute_mask,
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
    adapter
        .runtime_context_mut()
        .system_spine_mut()
        .unobserved_locked = 1;
    assert!(!adapter.evaluate_transient_guard_relaxation(0.40, 1, false, false, 2));
    adapter
        .runtime_context_mut()
        .system_spine_mut()
        .unobserved_locked = 0;

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
        adapter.system_spine().pinned_compute_mask,
        expected_baseline_pins
    );
    assert_ne!(
        adapter.system_spine().pinned_compute_mask & HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
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
        adapter.system_spine().pinned_compute_mask,
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
            0,
            &sig,
            &mut trail,
            Some(0.35),
            128,
            &mut q_rot,
            None,
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
    let baseline = clean_adapter.system_spine().baseline_pinned_compute_mask;
    clean_adapter.begin_turn_spine_policy(false, false);
    assert_eq!(clean_adapter.system_spine().pinned_compute_mask, baseline);

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
        guarded_adapter.system_spine().pinned_compute_mask,
        guarded_adapter.compute_pool().active_mask
    );
    assert_eq!(
        guarded_adapter.system_spine().baseline_pinned_compute_mask,
        baseline
    );

    guarded_adapter
        .runtime_context_mut()
        .deserialize_from_bytes(&buf)
        .expect("restore snapshot");
    assert_eq!(
        guarded_adapter.system_spine().baseline_pinned_compute_mask,
        baseline
    );
    assert_eq!(guarded_adapter.system_spine().pinned_compute_mask, baseline);
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
        query_seq_len: 0,
        query_ortho_ratio: 1.0,
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

    assert_eq!(DEFAULT_STARTUP_PRIOR_PROBES.len(), 12);
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
            16, &composite, &mut trail, None, 128, &mut q_rot, None,
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
    };
    let guard_entry = TokenSequenceEntry {
        role: AttractorRole::Guard,
        text: "Forbidden guard claim".to_string(),
        energy: 2.0,
        seq_len: 1,
        per_layer_token_qs: make_angle_seq(0.40),
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
            16,
            &ga,
            &mut esc_provider,
            &mut trail,
            None,
            128,
            &mut q_rot,
            None,
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
    let expected_host_pin_floor = session1.system_spine().baseline_pinned_compute_mask;
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
        session2.system_spine().baseline_pinned_compute_mask & expected_host_pin_floor,
        expected_host_pin_floor
    );
    assert_ne!(
        session2.system_spine().baseline_pinned_compute_mask & (1_u64 << 10),
        0,
        "deserialized snapshot must never weaken host Spine pin floor"
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
