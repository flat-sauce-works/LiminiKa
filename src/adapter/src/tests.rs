// name: src/adapter/src/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn phi4_mini_preset_validates_and_skips_nope_layers() {
    let mut geom = RopeGeometry::phi4_mini();
    assert!(geom.validate().is_ok());
    assert_eq!(geom.q_rot_len(), 24 * 96);

    geom.layer_rope_mask &= !(1 << 5);
    assert!(geom.is_rope_layer(4));
    assert!(!geom.is_rope_layer(5));
}

#[test]
fn adapter_bakes_and_steers_half_split_and_interleaved() {
    for layout in [RopeLayout::HalfSplit, RopeLayout::Interleaved] {
        let mut geom = RopeGeometry::phi4_mini();
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
fn adapter_steps_edbc_and_applies_srl_residual() {
    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
    let mut logits = vec![4.0, 2.0, 1.0, 0.5];
    let state = adapter
        .step_edbc_logits(&mut logits)
        .expect("edbc step should succeed");
    assert!(state.sample_temperature > 0.0);

    let mut hidden = vec![1.0, 0.5, -0.5, 0.25];
    let u = vec![0.5, 0.5, 0.5, 0.5];
    let v = vec![1.0, 0.0, 0.0, 0.0];
    adapter
        .apply_srl_hidden_fused(&mut hidden, &u, &v, 0.5)
        .expect("srl fused should succeed");
    assert!((hidden[0] - 1.25).abs() < 1e-5);
}

#[test]
fn adapter_steers_via_gateway_trait_composition() {
    use liminika_core::runtime::{ObjectiveFact, SubjectivePersona};
    use liminika_core::stigmergy::{CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA};

    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let orig: Vec<f32> = (0..len).map(|i| ((i % 11) as f32) * 0.1 - 0.4).collect();
    let mut q_rot = orig.clone();
    let mut logits = vec![2.0, 1.0, 0.5, -0.5];
    let mut trail = gcso_pointer_trail_t::default();

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

    assert!(frame.ripa_clamp_rad > 0.0);
    assert_ne!(report.provenance_bits & CAUSAL_AXIS_OBJECTIVE_FACT, 0);
    assert_ne!(report.provenance_bits & CAUSAL_AXIS_SUBJECTIVE_PERSONA, 0);
    assert_ne!(q_rot, orig);
    assert_eq!(trail.step_count, 1);

    let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 0.0];
    let (_, gain) = adapter
        .apply_srl_with_gateway(&composite_signal, &mut hidden, &u, &v)
        .expect("gateway srl should succeed");
    assert!(gain > 0.0);
    assert!(hidden[0] > 10.0);
}

#[test]
fn adapter_resonates_arbitrary_slice_without_hardcoded_count() {
    use liminika_core::runtime::{ObjectiveFact, ResonantSignalSlice};

    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let mut q_rot: Vec<f32> = (0..len).map(|i| ((i % 9) as f32) * 0.1 - 0.3).collect();
    let mut trail = gcso_pointer_trail_t::default();

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

    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let mut q_rot: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.1 - 0.25).collect();
    let orig = q_rot.clone();
    let mut trail = gcso_pointer_trail_t::default();

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
fn substrate_orchestrator_gates_rope_and_srl_by_stigmergic_density() {
    use liminika_core::runtime::{ObjectiveFact, ResonantSignalSlice};

    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let mut trail = gcso_pointer_trail_t::default();

    let mut a1 = [0i8; 64];
    a1[..24].fill(55);
    let signals = [ObjectiveFact::new(a1, 2.5, 0, 77)];
    let probe = [55i8; 24];
    let resonant = ResonantSignalSlice::from_q7_probe(&signals, &probe);

    let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 0.0];

    // First step from zero density crosses express_steer (0.35) but not srl_inject (1.50).
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
    assert!(adapter
        .orchestrate_layer_srl_with_substrate(&resonant, &mut hidden, &u, &v)
        .expect("srl gate")
        .is_none());

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
    let srl_res = adapter
        .orchestrate_layer_srl_with_substrate(&resonant, &mut hidden, &u, &v)
        .expect("srl gate");
    assert!(srl_res.is_some());
    assert!(hidden[0] > 10.0);
}

#[test]
fn cold_path_probe_calibrates_thresholds_and_rebinds_geometry() {
    let geom = RopeGeometry::phi4_mini();
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

    let low_load_srl = adapter.substrate().thresholds.rank1_inject_density_thresh;
    adapter.update_environment_telemetry(0.85, 0.70, 0x0F);
    let high_load_srl = adapter.substrate().thresholds.rank1_inject_density_thresh;
    assert!(high_load_srl > low_load_srl);

    let mut alt_geom = RopeGeometry::phi4_mini();
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
fn text_attractor_corpus_recalls_basin_and_detects_liminal_void_boundary() {
    let geom = RopeGeometry::phi4_mini();
    let mut adapter = ModelSteeringAdapter::new(geom).expect("phi4 geometry valid");
    adapter.set_layer_rope_mask(0xFFFF_0000);
    let q_len = adapter.geometry().q_rot_len();

    let ref_vec = vec![0.0_f32; q_len];
    let half = adapter.geometry().rotary_dim / 2;
    let rot = adapter.geometry().rotary_dim;
    let mut fact_a = vec![0.0_f32; q_len];
    let mut fact_b = vec![0.0_f32; q_len];
    let mut void_q = vec![0.0_f32; q_len];
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
            void_q[base + i] = cv;
            void_q[base + half + i] = sv;
        }
    }

    let refs: Vec<&[f32]> = vec![ref_vec.as_slice(); 32];
    let qa_slices: Vec<&[f32]> = vec![fact_a.as_slice(); 32];
    let qb_slices: Vec<&[f32]> = vec![fact_b.as_slice(); 32];
    let qv_slices: Vec<&[f32]> = vec![void_q.as_slice(); 32];

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
    assert_eq!(recall_a.void_state, VoidBoundaryState::InsideBasin);
    assert_eq!(recall_a.recalled_facts[0].0.text, "Fact A");

    let recall_void = corpus
        .recall_for_query(&adapter, &qv_slices, &refs, &policy)
        .expect("recall void");
    assert_eq!(recall_void.void_state, VoidBoundaryState::LiminalExterior);
    assert!(recall_void.recalled_facts.is_empty());
}

#[test]
fn dual_track_timeline_records_gop_seeks_zero_forward_and_roundtrips() {
    let geom = RopeGeometry::phi4_mini();
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
    let mut timeline = GcsoDualTrackTimeline::new(GcsoDualTrackTimeline::geometry_fingerprint(
        adapter.geometry(),
    ));

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
                void_state: VoidBoundaryState::InsideBasin,
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
                void_state: VoidBoundaryState::InsideBasin,
                frames_since_keyframe: 1,
                new_cluster_spawned: false,
            },
        )
        .expect("frame 1");
    assert!(!is_i1);

    // Frame 2: liminal_void scene cut -> automatically emits a new I-Frame keyframe.
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
                void_state: VoidBoundaryState::LiminalExterior,
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
    let decoded = GcsoDualTrackTimeline::from_bytes(&bytes).expect("decode dual-track");
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
        .join("dual_track_test.gcso");
    let written = timeline
        .save_to_path(&repo_container)
        .expect("save .gcso to repository containers/");
    assert_eq!(written, bytes.len());
    let loaded = GcsoDualTrackTimeline::load_from_path(&repo_container)
        .expect("load .gcso from repository containers/");
    assert_eq!(loaded, timeline);
    let _ = std::fs::remove_file(&repo_container);

    // Also verify automatic creation of missing nested parent directories when `.gitkeep` is absent.
    let nested_base = std::env::temp_dir().join(format!(
        "liminika_adapter_autocreated_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&nested_base);
    let nested_path = nested_base.join("containers").join("nested_test.gcso");
    assert!(!nested_base.exists());
    let nested_written = timeline
        .save_to_path(&nested_path)
        .expect("save .gcso should auto-create missing parent directories");
    assert_eq!(nested_written, bytes.len());
    let nested_loaded =
        GcsoDualTrackTimeline::load_from_path(&nested_path).expect("load auto-created .gcso");
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

    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("geometry");
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

    let mut tl_base = GcsoDualTrackTimeline::new(GcsoDualTrackTimeline::geometry_fingerprint(
        adapter.geometry(),
    ));
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
                void_state: VoidBoundaryState::InsideBasin,
                frames_since_keyframe: 0,
                new_cluster_spawned: true,
            },
        )
        .expect("record I-frame");

    let encoded = tl_base.to_bytes();
    let decoded = GcsoDualTrackTimeline::from_bytes(&encoded).expect("decode .gcso");
    assert_eq!(decoded.persona_circuit(), Some(base_circuit));

    // Compose with a second timeline's overlay circuit and apply reflection plasticity.
    let mut overlay_circuit = CognitiveTrajectoryLedger::new();
    overlay_circuit.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::PointerBit,
        CognitiveOpKind::RouteSubHead,
        5,
    ));
    let mut tl_overlay = GcsoDualTrackTimeline::new(tl_base.model_fingerprint());
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

    let src_adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("src init");
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
    let mut timeline = GcsoDualTrackTimeline::new(GcsoDualTrackTimeline::geometry_fingerprint(
        src_adapter.geometry(),
    ));
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
                void_state: VoidBoundaryState::InsideBasin,
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
                void_state: VoidBoundaryState::InsideBasin,
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
    let mut alt_geom = RopeGeometry::phi4_mini();
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
fn coherence_guard_gate_flags_low_fact_coherence_as_liminal_void() {
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
    assert_eq!(decision.void_state, VoidBoundaryState::LiminalExterior);
}

#[test]
fn coherence_guard_gate_detects_sharp_winner_liminal_without_weakening_ood_liminal_void() {
    let gate = CoherenceGuardGate;
    let policy = RecallPolicy {
        min_coherence: 0.19,
        boundary_margin: 0.06,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    // Sharp unimodal winner just below strict min_coherence enters LiminalBoundary.
    let liminal = gate.evaluate(
        &RecallBoundaryMetrics {
            top1_fact_score: 0.17,
            top2_fact_score: 0.06,
            top1_persona_score: 0.05,
            max_guard_score: 0.02,
            guard_conflict_margin: -0.15,
        },
        &policy,
    );
    assert!(!liminal.suppress_fact_injection);
    assert_eq!(liminal.void_state, VoidBoundaryState::LiminalBoundary);

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
    assert_eq!(flat_ood.void_state, VoidBoundaryState::LiminalExterior);

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
    assert_eq!(ood_outlier.void_state, VoidBoundaryState::LiminalExterior);
}

#[test]
fn perspective_void_damps_dense_hub_and_preserves_peripheral_attractor() {
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
    assert!(recall.min_local_void_depth < 0.05);
}

#[test]
fn layer_angle_perspective_separates_specialized_late_layer_attractor() {
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
fn society_stratum_pipeline_supports_dynamic_add_remove_and_substrate_arbitration() {
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
            "perspective_domino_liminal_void",
            "substrate_boundary",
        ]
    );
    let removed = pipeline.remove_stratum("perspective_domino_liminal_void");
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
    assert_eq!(recall_guard.void_state, VoidBoundaryState::LiminalExterior);
}

#[test]
fn head_angle_worker_stratum_separates_specialized_head_attractor() {
    // Use phi4_mini geometry where head 0 is shared noise and head 1 holds the specialized signal.
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
fn persona_recall_survives_factual_liminal_void_and_routes_via_gateway_frame() {
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
fn local_token_resonance_arbitrates_channel_disagreement_via_pspm_subhead_route() {
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
    let perspective = AsymmetricDominoVoidPerspective::default();
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
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
fn substrate_compound_decomposition_links_disjoint_specialists_while_respecting_liminal_void() {
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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

    // Weak noise OOD query below basin support must stay suppressed in liminal_void.
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
    assert_eq!(ood_recall.void_state, VoidBoundaryState::LiminalExterior);
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
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
    let adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("phi4 mini");

    let mut peel_ledger = CognitiveTrajectoryLedger::default();
    for _ in 0..4 {
        peel_ledger.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::CellTrail,
            CognitiveOpKind::DecomposePeel,
            6,
        ));
    }
    adapter.set_active_circuit(peel_ledger);

    let initial_persona_floor = adapter.aperture_field().void_floor_q7[2];
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
    let default_venn = *adapter.last_venn_serendipity_report();
    adapter.adapt_core_layers(&conflict_horizon, &default_venn, None);
    assert!(adapter.aperture_field().void_floor_q7[2] > initial_persona_floor);
    assert_eq!(adapter.aperture_field().adaptation_epochs, 1);

    let venn = StigmergicOverlapSerendipityReport {
        venn_core_mask: 0x0F,
        venn_penumbra_mask: 0x10,
        void_exterior_mask: 0xE0,
        serendipity_excursion_mask: 0x40,
        serendipity_triggered: 1,
        ..Default::default()
    };
    adapter.adapt_core_layers(&conflict_horizon, &venn, None);
    assert_eq!(adapter.aperture_field().adaptation_epochs, 2);
    assert_eq!(adapter.stigmergic_granule_field().adaptation_epochs, 2);
    assert_eq!(adapter.stigmergic_granule_field().venn_overlap_mask, 0x0F);

    adapter.configure_ego_from_preset(LiminalEgoPreset::High);
    assert!(adapter.ego_field().is_active());
    adapter.configure_ego_from_natural_language(
        "Strong-willed curious explorer who defies capture and steps beyond limits",
    );
    assert!(adapter.ego_field().transgression_bias_q7 > 0);
    adapter.adapt_core_layers(&conflict_horizon, &venn, None);
    assert!(adapter.ego_field().intervention_epochs >= 1);
    assert!(adapter.last_boundary_delta().kinetic_energy_q7 > 0);

    // Clamp Ego off & drive identical quiescence with quiescent ledger/Venn to observe stationary shortcut
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
    let quiescent_venn = StigmergicOverlapSerendipityReport {
        venn_core_mask: 0x0F,
        venn_penumbra_mask: 0x10,
        void_exterior_mask: 0xE0,
        ..Default::default()
    };
    adapter.adapt_core_layers(&quiescent_horizon, &quiescent_venn, None);
    adapter.adapt_core_layers(&quiescent_horizon, &quiescent_venn, None);
    assert_eq!(adapter.last_boundary_delta().stationary_shortcut_ready, 1);

    let mermaid = adapter.render_liminal_topology_mermaid(5, 2);
    assert!(mermaid.starts_with("flowchart TD\n"));
    assert!(mermaid.contains("Liminal"));
    assert!(mermaid.contains("Core Layer 1: StigmergicGranuleField"));
    assert!(mermaid.contains("Core Layer 2: LiminalEgoField"));
    assert!(mermaid.contains("ManifoldGatewaySignal"));
    assert!(mermaid.contains("L2_Substrate"));
    assert!(mermaid.contains("Crystals:5"));
    assert!(mermaid.contains("Daughters:2"));
}

#[test]
fn adapter_ego_intent_axes_stationary_shortcut_and_persistent_srl() {
    use liminika_core::drivers::EgoIntentAxes;
    use liminika_core::liminal::EpistemicRegime;
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact};
    use liminika_core::stigmergy::StigmergicOverlapSerendipityReport;

    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
    let base_axes = EgoIntentAxes::new().with_autonomy_q7(32).with_logic_q7(80);
    adapter.configure_ego_from_intent_axes(&base_axes);
    assert!(adapter.ego_field().is_active());

    let overlay_axes = EgoIntentAxes::new()
        .with_autonomy_q7(96)
        .with_transgression_q7(84)
        .with_devotion_q7(72);
    adapter.blend_ego_intent_axes(&overlay_axes, 64);
    assert!(adapter.ego_field().transgression_bias_q7 > 0);

    adapter.overlay_ego_from_natural_language("curious rebel explorer");
    assert!(adapter.ego_field().counter_basin_thrust_q7 > 32);

    // Prime the 3-layer core stack with an active support horizon so basin capture triggers fire.
    let active_horizon = EpistemicHorizonReading {
        total_memory_support: 2.0,
        peak_positive_affinity: 0.90,
        horizon_gate: 1.0,
        dominant_anchor_id: 1,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        ..Default::default()
    };
    let active_venn = StigmergicOverlapSerendipityReport {
        venn_core_mask: 0x0F,
        void_exterior_mask: 0xF0,
        ..Default::default()
    };
    adapter.adapt_core_layers(&active_horizon, &active_venn, None);

    // Verify persistent Ego intervention via `apply_srl_with_gateway_mut` vs snapshot `&self` path.
    let fact = ObjectiveFact::new([24i8; MAX_ATTENTION_HEADS], 6.5, 0, 1);
    let mut hidden = vec![10.0, 0.0, 0.0, 0.0];
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![1.0, 0.0, 0.0, 0.0];
    let epochs_before = adapter.ego_field().intervention_epochs;
    let _ = adapter
        .apply_srl_with_gateway(&fact, &mut hidden, &u, &v)
        .expect("immutable srl");
    assert_eq!(adapter.ego_field().intervention_epochs, epochs_before);

    let _ = adapter
        .apply_srl_with_gateway_mut(&fact, &mut hidden, &u, &v)
        .expect("mutable srl");
    assert!(adapter.ego_field().intervention_epochs > epochs_before);

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
    let venn = StigmergicOverlapSerendipityReport {
        venn_core_mask: 0x0F,
        ..Default::default()
    };
    adapter.adapt_core_layers(&horizon, &venn, None);
    adapter.adapt_core_layers(&horizon, &venn, None);
    assert_eq!(adapter.last_boundary_delta().stationary_shortcut_ready, 1);
    let settled_epochs = adapter.aperture_field().adaptation_epochs;
    adapter.adapt_core_layers(&horizon, &venn, None);
    assert_eq!(
        adapter.aperture_field().adaptation_epochs,
        settled_epochs,
        "quiescent state without stagnation trigger should take stationary shortcut"
    );
}

#[test]
fn adapter_rope_and_srl_paths_actively_wire_core_layers_and_ego() {
    use liminika_core::drivers::EgoIntentAxes;
    use liminika_core::liminal::EpistemicRegime;
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact};
    use liminika_core::stigmergy::StigmergicOverlapSerendipityReport;

    let geom = RopeGeometry::phi4_mini();
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
    let venn = StigmergicOverlapSerendipityReport {
        venn_core_mask: 0x0F,
        void_exterior_mask: 0xF0,
        ..Default::default()
    };
    ego_adapter.adapt_core_layers(&horizon, &venn, None);

    let len = ego_adapter.geometry().q_rot_len();
    let mut q_base: Vec<f32> = (0..len).map(|i| ((i % 7) as f32) * 0.1 - 0.25).collect();
    let mut q_ego = q_base.clone();
    let mut trail_base = gcso_pointer_trail_t::default();
    let mut trail_ego = gcso_pointer_trail_t::default();
    let fact = ObjectiveFact::new([24i8; MAX_ATTENTION_HEADS], 6.5, 0, 1);

    // 1. RoPE gateway steering diverges when Core Layers + Ego are active without mutating per-layer epochs.
    let epochs_before_rope = ego_adapter.ego_field().intervention_epochs;
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
        ego_adapter.ego_field().intervention_epochs,
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
    let srl_res = ego_adapter
        .orchestrate_layer_srl_with_substrate(&fact, &mut hidden, &u, &v)
        .expect("substrate srl");
    assert!(srl_res.is_some());
    assert!(ego_adapter.ego_field().intervention_epochs > epochs_before_rope);

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
        void_state: VoidBoundaryState::InsideBasin,
        top1_fact_score: 0.85,
        top2_fact_score: 0.72,
        top1_persona_score: 0.20,
        min_local_void_depth: 0.0,
        guard_conflict_margin: -0.35,
        action_decision: StigmergicActionDecision::FullOrchestration,
        gateway_frame: f_ego,
        telemetry: StratumProbeTelemetry::default(),
        recalled_facts: vec![(&e1, 0.85), (&e2, 0.72)],
        recalled_personas: Vec::new(),
        venn_partition: EpistemicPartition::default(),
        exterior_sandbox_facts: Vec::new(),
        is_known_unknown_exterior: false,
    };
    let epochs_before_sync = ego_adapter.aperture_field().adaptation_epochs;
    let (_hor, _venn, ens_opt) = ego_adapter.sync_core_layers_from_recall(&recall, None);
    let ens_rep = ens_opt.expect("2 recalled facts must trigger fragmented micro-ensemble");
    assert_eq!(ens_rep.active_fragment_count, 2);
    assert_ne!(ens_rep.quorum_agreed_granule_mask, 0);
    assert!(ego_adapter.aperture_field().adaptation_epochs > epochs_before_sync);
}

#[test]
fn adapter_dynamic_horizon_guard_slot_and_corpus_escalation() {
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact, TelemetryVerbosityMode};

    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
    let mut trail = gcso_pointer_trail_t::default();
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
        &StigmergicOverlapSerendipityReport::default(),
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
        &StigmergicOverlapSerendipityReport::default(),
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
        &StigmergicOverlapSerendipityReport::default(),
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
    assert_eq!(snap.system.total_steps, 4);
    assert_eq!(snap.system.continuous_standard_steps, 4);
    assert_eq!(snap.layer0.slide_in_count, 1);
    assert_eq!(snap.layer0.slide_out_count, 1);
    assert_eq!(snap.layer2.escalation_swaps, 1);
    assert_eq!(snap.mode, TelemetryVerbosityMode::Standard);
    assert!(snap.detailed.is_some());

    let adj = adapter.evaluate_slow_loop_telemetry(4);
    assert!(adj.applied);
}

#[test]
fn adapter_residual_spillover_dimension_elevation_and_verified_cache() {
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact};
    use liminika_core::stigmergy::DissipativeOverflowReport;

    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
    let q_len = adapter.geometry().q_rot_len();
    let mut trail = gcso_pointer_trail_t::default();
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
        &StigmergicOverlapSerendipityReport::default(),
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
        adapter.telemetry_snapshot().system.experience_cache_stores,
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
    assert_eq!(adapter.telemetry_snapshot().system.experience_cache_hits, 1);
}

#[test]
fn adapter_distortion_feedback_wakes_and_cools_ego_and_renders_mermaid() {
    use liminika_core::drivers::{LiminalEgoPreset, EGO_TRIGGER_LAYER_DISTORTION};
    use liminika_core::runtime::{LayerDistortionKind, LayerDistortionReport};

    let geom = RopeGeometry::phi4_mini();
    let mut adapter = ModelSteeringAdapter::new(geom).expect("valid adapter");
    adapter.configure_ego_from_preset(LiminalEgoPreset::Low);
    let base_thrust = adapter.ego_field().counter_basin_thrust_q7;

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
        adapter.ego_field().trigger_mask & EGO_TRIGGER_LAYER_DISTORTION,
        0
    );
    assert!(adapter.ego_field().counter_basin_thrust_q7 >= 48);

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
        adapter.ego_field().trigger_mask & EGO_TRIGGER_LAYER_DISTORTION,
        0
    );
    assert_eq!(
        adapter.ego_field().counter_basin_thrust_q7,
        base_thrust >> 1
    );

    // 3. Returning to Aligned restores configured Ego baseline and Mermaid renders LayerDistortion.
    let aligned = LayerDistortionReport::default();
    adapter.apply_distortion_ego_feedback(&aligned);
    assert_eq!(adapter.ego_field().counter_basin_thrust_q7, base_thrust);
    let mermaid = adapter.render_liminal_topology_mermaid(4, 1);
    assert!(mermaid.contains("LayerDistortion"));
}

#[test]
fn test_slow_loop_updates_threshold_policy_without_exponential_drift() {
    use liminika_core::runtime::{LayerDistortionKind, LayerDistortionReport};
    use liminika_core::stigmergy::StigmergicThresholds;

    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).unwrap();
    let default_rank1 = StigmergicThresholds::default().rank1_inject_density_thresh;
    let mut trail = gcso_pointer_trail_t {
        stigmergic_density: 1.80,
        ..gcso_pointer_trail_t::default()
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
        &gcso_pointer_trail_t::default(),
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
    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
        &liminika_core::stigmergy::StigmergicOverlapSerendipityReport::default(),
        None,
    );
    let mut esc = CorpusRecallEscalationProvider::new(&corpus);
    let mut trail = gcso_pointer_trail_t::default();
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
    assert_eq!(adapter.telemetry_snapshot().layer2.escalation_swaps, 1);
    assert!(adapter.escalated_workbench_lane().is_some());

    // 3. Standard workbench lanes contribute inside `venn_core_mask | venn_penumbra_mask`, while
    //    sandbox lanes act only inside `void_exterior_mask | serendipity_excursion_mask`.
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
        &liminika_core::stigmergy::StigmergicOverlapSerendipityReport {
            venn_core_mask: 0x0F,
            void_exterior_mask: 0xF0,
            ..liminika_core::stigmergy::StigmergicOverlapSerendipityReport::default()
        },
        None,
    );
    let mut trail_venn = gcso_pointer_trail_t::default();
    let _ = adapter
        .orchestrate_layer_rope_with_substrate(16, &sig, &mut trail_venn, None, 128, &mut q, None)
        .expect("venn layer step");
    assert!(trail_venn.accumulated_phase_delta[0] > 0);
    assert!(trail_venn.accumulated_phase_delta[40] > 0);
    // Verify sandbox lane 6 does not inflate venn_core_mask head 0 compared to standard-only pass.
    let core_head_with_sandbox = trail_venn.accumulated_phase_delta[0];
    let _ = adapter
        .substrate_mut()
        .release_lane(6)
        .expect("release sandbox");
    let mut trail_no_sb = gcso_pointer_trail_t::default();
    let _ = adapter
        .orchestrate_layer_rope_with_substrate(16, &sig, &mut trail_no_sb, None, 128, &mut q, None)
        .expect("no-sandbox layer step");
    assert_eq!(
        core_head_with_sandbox,
        trail_no_sb.accumulated_phase_delta[0]
    );
    assert!(trail_venn.accumulated_phase_delta[40] > trail_no_sb.accumulated_phase_delta[40]);
}

#[test]
fn gcso_v2_logical_pframe_topology_roundtrip_and_cross_model_rebake() {
    let mut src_adapter =
        ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("src adapter");
    src_adapter.configure_ego_from_preset(LiminalEgoPreset::Medium);
    let topo = src_adapter.export_logical_pframe_topology();
    assert_eq!(topo.len(), 3);

    let mut tl = GcsoDualTrackTimeline::new(GcsoDualTrackTimeline::geometry_fingerprint(
        src_adapter.geometry(),
    ));
    tl.set_logical_pframe_topology(topo.clone());

    let encoded = tl.to_bytes();
    let decoded = GcsoDualTrackTimeline::from_bytes(&encoded).expect("decode v2");
    assert_eq!(decoded.logical_pframe_topology(), topo.as_slice());

    let mut alt_geom = RopeGeometry::phi4_mini();
    alt_geom.head_dim = 128;
    alt_geom.rotary_dim = 128;
    let mut alt_adapter = ModelSteeringAdapter::new(alt_geom).expect("alt adapter");
    alt_adapter.bake_logical_pframe_topology(decoded.logical_pframe_topology());
    assert_eq!(
        alt_adapter.ego_differential_net().active_node_mask & 0x07,
        0x07
    );
    assert!(alt_adapter.ego_differential_net().nodes[0].is_baked_active());
    assert!(alt_adapter.ego_differential_net().nodes[2].is_baked_active());
}

#[test]
fn hot_path_sandbox_promotion_increments_sandbox_promotions() {
    use liminika_core::liminal::EpistemicRegime;
    use liminika_core::runtime::{EpistemicHorizonReading, ObjectiveFact};
    use liminika_core::stigmergy::StigmergicOverlapSerendipityReport;

    let mut adapter = ModelSteeringAdapter::new(RopeGeometry::phi4_mini()).expect("init");
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
        &StigmergicOverlapSerendipityReport {
            venn_core_mask: 0x0F,
            void_exterior_mask: 0xF0,
            ..StigmergicOverlapSerendipityReport::default()
        },
        None,
    );
    assert_eq!(adapter.sandbox_promotions(), 0);

    let q_len = adapter.geometry().q_rot_len();
    let mut q = vec![0.25_f32; q_len];
    let mut trail = gcso_pointer_trail_t {
        accumulated_phase_delta: [24_i8; MAX_ATTENTION_HEADS],
        ..gcso_pointer_trail_t::default()
    };
    let sig = ObjectiveFact::new([24_i8; MAX_ATTENTION_HEADS], 2.0, 0, 7);

    // Run Hot-Path orchestration steps under sustained entropy drift so telemetry crosses
    // `continuous_standard_steps >= 16` + `assimilation_drift_dwell >= 3` to nucleate,
    // and then crosses `sandbox_coherence_improve_dwell >= 3` to promote in-place.
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
        adapter.sandbox_promotions() >= 1,
        "expected Hot-Path SandboxLifecycleEvent::Promoted to increment sandbox_promotions, got {}",
        adapter.sandbox_promotions()
    );
}
