// name: src/core/src/runtime/tests/lifecycle_and_abi.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RuntimeContext lifecycle, config, snapshot, and C-ABI tests.

use super::*;

#[test]
fn test_two_origin_supertrait_synthesis_and_causal_provenance() {
    let sys = SystemConstraint::new((), 0.20, 0.10, 0xFF);
    let fact = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 3.0, 0x0012_3400, 77);
    let persona_aligned = SubjectivePersona::new([30i8; MAX_ATTENTION_HEADS], 2.0, 0x0020_0000, 12);

    let (frame, words) = (&sys, &fact, &persona_aligned).synthesize_gateway_frame_with_words();
    assert_eq!(words, load_q7_words_64(&frame.phase_deltas));
    let report = frame.to_causal_report();
    assert_eq!(frame.mapped_section_offset48, 0x0012_3400);
    assert_eq!(frame.target_anchor_id, 77);
    assert_eq!(frame.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(report.root_cause_code, ROOT_CAUSE_NOMINAL);
    assert!(report.fact_persona_coherence > 0.95);
    assert!((frame.phase_clamp_rad - DEFAULT_PHASE_CLAMP_MAX_RAD).abs() < 1e-6);

    let f32_emb = [0.5_f32, 0.866_025_4, -0.5, 0.866_025_4];
    let fact_f32 = ObjectiveFact::new(&f32_emb[..], 2.5, 0x0015_0000, 88);
    let (f_f32, w_f32) = fact_f32.synthesize_gateway_frame_with_words();
    assert_eq!(w_f32, load_q7_words_64(&f_f32.phase_deltas));

    let persona_opposed =
        SubjectivePersona::new([-96i8; MAX_ATTENTION_HEADS], 2.5, 0x0020_0000, 12);
    let col_frame = (&sys, &fact, &persona_opposed).synthesize_gateway_frame();
    let col_report = col_frame.to_causal_report();
    assert!(col_report.is_persona_fact_collision());
    assert_eq!(
        col_report.root_cause_code,
        ROOT_CAUSE_PERSONA_FACT_COLLISION
    );
    assert_eq!(col_frame.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(col_frame.target_anchor_id, 77);

    let guard = ContradictionGuard::new(ContrastivePhaseBloomFilter::new(), 0.65, 3);
    let rep_frame = (&sys, &fact, &persona_aligned, &guard).synthesize_gateway_frame();
    let rep_report = rep_frame.to_causal_report();
    assert!(rep_report.is_cohomology_repulsion());
    assert_eq!(rep_report.root_cause_code, ROOT_CAUSE_COHOMOLOGY_REPULSION);
    assert_eq!(rep_frame.dominant_slot, ATTRACTOR_SLOT_REPULSION);
}

#[test]
fn test_online_toroidal_kmeans_dp_means_consolidation_and_unified_container_view() {
    let mut centroids = [
        PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0),
        PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
        PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
    ];
    assert_eq!(centroids[0].causal_provenance_bits(), 0);
    assert_eq!(
        centroids[0].synthesize_gateway_frame().action_mode_flags,
        ACTION_PRESET_PASSIVE_SIDECAR
    );

    let mut trail = PointerTrail::default();

    let fact_a = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 2.5, 0x0010_0000, 101);
    let (f1, w1) = fact_a.synthesize_gateway_frame_with_words();
    let (idx1, id1, _coh1, mass1, spawned1) =
        PhaseAnchorRecord::step_online_toroidal_kmeans_frame_with_words_64(
            &mut centroids,
            &f1,
            &w1,
            DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
            Some(&mut trail),
        );
    assert!(spawned1 && idx1 == 0 && id1 == 101 && trail.cluster_id == 101 && mass1 > 0.0);

    let fact_a2 = ObjectiveFact::new([30i8; MAX_ATTENTION_HEADS], 3.0, 0x0010_0000, 101);
    let (f2, w2) = fact_a2.synthesize_gateway_frame_with_words();
    let (idx2, id2, coh2, mass2, spawned2) =
        PhaseAnchorRecord::step_online_toroidal_kmeans_frame_with_words_64(
            &mut centroids,
            &f2,
            &w2,
            DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
            Some(&mut trail),
        );
    assert!(!spawned2 && idx2 == 0 && id2 == 101 && coh2 > 0.95 && mass2 > mass1);
    assert_eq!(centroids[0].sample_count, 2);

    let persona_b = SubjectivePersona::new([-80i8; MAX_ATTENTION_HEADS], 2.8, 0x0024_0000, 202);
    let (f3, w3) = persona_b.synthesize_gateway_frame_with_words();
    let (idx3, id3, _coh3, mass3, spawned3) =
        PhaseAnchorRecord::step_online_toroidal_kmeans_frame_with_words_64(
            &mut centroids,
            &f3,
            &w3,
            DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
            Some(&mut trail),
        );
    assert!(spawned3 && idx3 == 1 && id3 == 202 && trail.cluster_id == 202 && mass3 > 0.0);

    let (act_cnt, tot_smp, mean_r, mean_k, _max_coh) =
        PhaseAnchorRecord::diagnose_macro_distribution_64(&centroids);
    assert_eq!(act_cnt, 2);
    assert_eq!(tot_smp, 3);
    assert!(mean_r > 0.85 && mean_k > 2.0);

    let closed_loop_frame = centroids.synthesize_gateway_frame();
    assert!(closed_loop_frame.effective_energy > 2.0);
    assert_ne!(
        closed_loop_frame.lineage_flags & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
        0
    );

    let batch_frames = [f1, f2];
    let (_b_idx, b_id, b_coh, b_spawned) =
        PhaseAnchorRecord::step_online_toroidal_kmeans_batch_frames_64(
            &mut centroids,
            &batch_frames,
            DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
            Some(&mut trail),
        );
    assert_eq!(b_id, 101);
    assert_eq!(b_spawned, 0);
    assert!(b_coh > 0.95 && centroids[0].verify_integrity());

    // Populate slot 2 with a near-duplicate of slot 0 and verify online consolidation
    centroids[2] = PhaseAnchorRecord::new_empty(303, ATTRACTOR_SLOT_KNOWLEDGE, 0);
    let sample_c2 = [29i8; MAX_ATTENTION_HEADS];
    let _ = centroids[2].absorb_sample_point_64(
        crate::liminal::Scoped::interior(PhaseProfileRef64(&sample_c2)).into_grounded(),
        1.2,
    );
    let (merged_cnt, pruned_cnt) = PhaseAnchorRecord::merge_and_prune_centroids_64(
        &mut centroids,
        DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
        0.10,
        0.98,
    );
    assert_eq!(merged_cnt, 1);
    assert_eq!(pruned_cnt, 0);
    assert_eq!(centroids[2].sample_count, 0);
    assert!(centroids[0].verify_integrity() && centroids[2].verify_integrity());

    let mut query_trail = PointerTrail {
        accumulated_phase_delta: [-76i8; MAX_ATTENTION_HEADS],
        stigmergic_density: 1.0,
        attractor_pull_force: 1.0,
        ..Default::default()
    };
    let (routed_idx, routed_id, routed_coh, routed_grav) =
        PhaseAnchorRecord::route_coarse_to_fine_trail_64(&centroids, &mut query_trail, 64);
    assert_eq!(routed_idx, 1);
    assert_eq!(routed_id, 202);
    assert_eq!(query_trail.cluster_id, 202);
    assert_eq!(
        TaggedPtr(query_trail.current_ptr).unpack_addr(),
        0x0024_0000
    );
    assert!(routed_coh > 0.95 && routed_grav > 1.0);

    let mut storage_buf = AlignedBuffer64::<u8, 384>::new([0u8; 384]);
    let written =
        PhaseAnchorRecord::write_records_to_container_bytes(&centroids, &mut storage_buf.data)
            .unwrap();
    assert_eq!(written, 384);
    let restored =
        PhaseAnchorRecord::slice_from_container_bytes_zero_copy(&storage_buf.data, 3).unwrap();
    assert_eq!(restored[0].anchor_id, 101);
    assert_eq!(restored[1].anchor_id, 202);

    let bloom = ContrastivePhaseBloomFilter::new();
    let ledger = LedgerSlot::default();
    let reports_arr = [f1.to_causal_report(), f2.to_causal_report()];
    let trails_arr = [query_trail];
    let mut container_buf = AlignedBuffer128::<u8, 1024>::new([0u8; 1024]);
    let no_rep_len = LmnkContainerView::write_standard_sections_to_bytes(
        StandardSectionsInput {
            container_flags: LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            anchors: &centroids[..2],
            trails: &trails_arr,
            bloom: Some(&bloom),
            ledger: Some(&ledger),
            provenance_reports: &[],
            timestamp_epoch_sec: 1_774_900_000,
        },
        &mut container_buf.data,
    )
    .unwrap();
    let no_rep_view = LmnkContainerView::parse(&container_buf.data[..no_rep_len]).unwrap();
    assert_eq!(no_rep_view.phase_anchor_count(), 2);

    let total_len = LmnkContainerView::write_standard_sections_to_bytes(
        StandardSectionsInput {
            container_flags: LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            anchors: &centroids,
            trails: &trails_arr,
            bloom: Some(&bloom),
            ledger: Some(&ledger),
            provenance_reports: &reports_arr,
            timestamp_epoch_sec: 1_774_900_000,
        },
        &mut container_buf.data,
    )
    .unwrap();

    let view = LmnkContainerView::parse(&container_buf.data[..total_len]).unwrap();
    assert_eq!(view.header().total_size, total_len as u64);
    assert_eq!(
        view.header().container_flags,
        LMNK_CONTAINER_FLAG_COMPILED_CACHE
    );
    assert_eq!(view.phase_anchors_zero_copy().unwrap().len(), 3);
    assert_eq!(view.pointer_trails_zero_copy().unwrap().len(), 1);
    assert_eq!(view.causal_provenance_zero_copy().unwrap().len(), 2);
    assert!(
        view.bloom_filter_zero_copy().unwrap().is_some()
            && view.ledger_slot_zero_copy().unwrap().is_some()
    );

    // Verify Top-2 Liminal Bifurcation Routing between two balanced attractors
    let mut lim_centroids = [
        PhaseAnchorRecord::new_empty(101, ATTRACTOR_SLOT_KNOWLEDGE, 0x0010_0000),
        PhaseAnchorRecord::new_empty(202, ATTRACTOR_SLOT_PERSONA, 0x0024_0000),
    ];
    let lim_p0 = [16i8; MAX_ATTENTION_HEADS];
    let lim_p1 = [48i8; MAX_ATTENTION_HEADS];
    let _ = lim_centroids[0].absorb_sample_point_64(
        crate::liminal::Scoped::interior(PhaseProfileRef64(&lim_p0)).into_grounded(),
        2.5,
    );
    let _ = lim_centroids[1].absorb_sample_point_64(
        crate::liminal::Scoped::interior(PhaseProfileRef64(&lim_p1)).into_grounded(),
        2.5,
    );
    let mut lim_trail = PointerTrail {
        accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
        stigmergic_density: 1.0,
        attractor_pull_force: 1.0,
        ..Default::default()
    };
    let (_k1, _k2, aid1, aid2, lim_coh, lim_grav, is_liminal) =
        PhaseAnchorRecord::route_top2_liminal_centroids_64(
            &lim_centroids,
            &mut lim_trail,
            64,
            DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO,
        );
    assert!(is_liminal && aid1 != 0 && aid2 != 0 && aid1 != aid2);
    assert!(lim_coh > 0.70 && lim_grav > 1.0);
    assert_ne!(lim_trail.flags & TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

    // Verify Continuous Geodesic Affinity Prior on unstamped candidate trails (cluster_id == 0)
    let cand_aligned = PointerTrail {
        accumulated_phase_delta: [18i8; MAX_ATTENTION_HEADS],
        stigmergic_density: 1.0,
        cluster_id: 0,
        ..Default::default()
    };
    let cand_orthogonal = PointerTrail {
        accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
        stigmergic_density: 1.0,
        cluster_id: 0,
        ..Default::default()
    };
    let mut src_trail = PointerTrail {
        accumulated_phase_delta: [16i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let (win_c, sel_tid, best_sc, best_c, found_ok) =
        PhaseAnchorRecord::route_coarse_to_fine_candidates_64(
            &lim_centroids,
            &mut src_trail,
            &[cand_orthogonal, cand_aligned],
            crate::stigmergy::CoarseToFineParams {
                head_dim: 128,
                phase_interference_gain: 1.0,
                stigmergic_prior_gain: 0.25,
                pointer_link_bonus: 0.10,
                cluster_bonus_gain: 0.50,
                contradiction_threshold: 0.35,
            },
        );
    assert!(found_ok && win_c == 0 && sel_tid == 1 && best_sc > 0.5 && best_c > 0.95);
}

#[test]
fn test_adaptive_threshold_policy_probe_and_telemetry_scaling() {
    let policy = AdaptiveThresholdPolicy::default();
    let low_pressure = SystemConstraint::new((), 0.0, 0.0, 0xFF);
    let high_pressure = SystemConstraint::new((), 0.85, 0.70, 0x0F);

    let mut probe = ManifoldProbeBaseline {
        active_heads: 24,
        steered_layers: 16,
        baseline_entropy: 0.40,
        baseline_inter_layer_coherence: 0.35,
        baseline_guard_noise_floor: 0.45,
        probe_epoch: 1,
    };
    let trail = PointerTrail {
        step_count: 32,
        attractor_pull_force: 1.0,
        ..Default::default()
    };

    let low_coh_t = policy.resolve_thresholds(&low_pressure, &probe, &trail, 0.40);
    probe.baseline_inter_layer_coherence = 0.90;
    let high_coh_t = policy.resolve_thresholds(&low_pressure, &probe, &trail, 0.40);
    // Models with weaker natural inter-layer coherence receive higher calibrated deposit gain
    // so equilibrium pheromone density stays within the Express..Rank-1 residual band without hand tuning.
    assert!(low_coh_t.reinforcement_gain > high_coh_t.reinforcement_gain);

    let stressed_t = policy.resolve_thresholds(&high_pressure, &probe, &trail, 0.40);
    assert!(stressed_t.inertial_skip_entropy_thresh > high_coh_t.inertial_skip_entropy_thresh);
    assert!(stressed_t.rank1_inject_density_thresh > high_coh_t.rank1_inject_density_thresh);
    assert!(stressed_t.crystallize_density_thresh > high_coh_t.crystallize_density_thresh);

    // Higher baseline_entropy gauge origin proportionally lifts inertial_skip_entropy_thresh.
    let mut high_h0_probe = probe;
    high_h0_probe.baseline_entropy = 0.50;
    let high_h0_t = policy.resolve_thresholds(&low_pressure, &high_h0_probe, &trail, 0.50);
    assert!(high_h0_t.inertial_skip_entropy_thresh > high_coh_t.inertial_skip_entropy_thresh);
}

#[test]
fn test_multi_scale_layer_cascade_and_cognitive_trajectory_circuit() {
    // 1. Compare two distinct cognitive circuits encoded solely via CognitiveTrajectoryLedger.
    let mut peel_heavy = CognitiveTrajectoryLedger::new();
    peel_heavy.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::DecomposePeel,
        6,
    ));
    peel_heavy.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::DecomposePeel,
        7,
    ));
    peel_heavy.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::PointerBit,
        CognitiveOpKind::RouteSubHead,
        5,
    ));

    let mut fuse_heavy = CognitiveTrajectoryLedger::new();
    fuse_heavy.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::PointerBit,
        CognitiveOpKind::SynthesizeFuse,
        5,
    ));
    fuse_heavy.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::SynthesizeFuse,
        6,
    ));
    fuse_heavy.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::CrystallizeCommit,
        7,
    ));

    let input_phases = [32i8; MAX_ATTENTION_HEADS];
    let input_words = load_q7_words_64(&input_phases);
    let bias_words = load_q7_words_64(&[30i8; MAX_ATTENTION_HEADS]);

    let act_peel = peel_heavy.evaluate_circuit_activation(
        &bias_words,
        &input_words,
        StigmergicScaleGranularity::CellTrail,
    );
    let act_fuse = fuse_heavy.evaluate_circuit_activation(
        &bias_words,
        &input_words,
        StigmergicScaleGranularity::MacroCoalition,
    );
    assert!(act_peel.peel_gate_q7 > act_fuse.peel_gate_q7);
    assert!(act_fuse.fuse_gate_q7 > act_peel.fuse_gate_q7);
    assert_eq!(act_peel.trajectory_depth, 3);
    assert_eq!(act_fuse.trajectory_depth, 3);

    // Verify CognitiveTrajectoryLedger acts as a composable SubjectivePersona Gateway origin.
    let fact = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 2.5, 0x0010_0000, 101);
    let composite_frame = (&fact, &peel_heavy).synthesize_gateway_frame();
    assert_ne!(
        composite_frame.lineage_flags & CAUSAL_AXIS_OBJECTIVE_FACT,
        0
    );
    assert_ne!(
        composite_frame.lineage_flags & CAUSAL_AXIS_SUBJECTIVE_PERSONA,
        0
    );

    // 2. Verify tuple & array StigmergicScaleLayer scaling (N = 1, 2, 3).
    let l0 = ScaleLayerNode::new(fact, StigmergicScaleGranularity::MacroCoalition);
    let l1 = ScaleLayerNode::new(
        SubjectivePersona::new([16i8; MAX_ATTENTION_HEADS], 1.5, 0x0020_0000, 202),
        StigmergicScaleGranularity::CellTrail,
    );
    let l2 = ScaleLayerNode::new(peel_heavy, StigmergicScaleGranularity::PointerBit);
    let tuple_stack = (l0.signal, l1.signal);
    let array_stack = [
        ScaleLayerNode::new(
            fact.synthesize_gateway_frame(),
            StigmergicScaleGranularity::MacroCoalition,
        ),
        ScaleLayerNode::new(
            l1.signal.synthesize_gateway_frame(),
            StigmergicScaleGranularity::CellTrail,
        ),
        ScaleLayerNode::new(
            l2.signal.synthesize_gateway_frame(),
            StigmergicScaleGranularity::PointerBit,
        ),
    ];
    assert_eq!(l0.layer_count(), 1);
    assert_eq!(tuple_stack.layer_count(), 2);
    assert_eq!(array_stack.layer_count(), 3);

    let proj_1 = l0.project_scale_layer(&input_words, &peel_heavy, 0, 1);
    let proj_3 = array_stack.project_scale_layer(&input_words, &peel_heavy, 0, 3);
    assert_eq!(proj_1.total_layers, 1);
    assert_eq!(proj_3.total_layers, 3);
    // Additional layers peel more signal energy, changing the unpeeled residual field.
    assert_ne!(proj_1.residual_words, input_words);
    assert_ne!(proj_3.residual_words, proj_1.residual_words);

    // 3. Verify ScaleLayerOrchestrator executes zero-alloc elastic depth-N cascades and persists trajectory.
    let mut substrate = crate::stigmergy::substrate::StigmergicSubstrate::new();
    let mut ledger = LedgerSlot::default();
    let mut trail = PointerTrail {
        accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
        stigmergic_density: 1.0,
        ..Default::default()
    };
    let layer_array = [
        ScaleLayerNode::new(
            GatewayFrame::default(),
            StigmergicScaleGranularity::MacroCoalition,
        ),
        ScaleLayerNode::new(
            l0.signal.synthesize_gateway_frame(),
            StigmergicScaleGranularity::CellTrail,
        ),
        ScaleLayerNode::new(
            peel_heavy.synthesize_gateway_frame(),
            StigmergicScaleGranularity::PointerBit,
        ),
    ];
    let (orch_proj, updated_ledger, swarm_rep) = substrate
        .orchestrate_adaptive_depth_n(
            &layer_array,
            3,
            ScaleCascadeInput {
                input_ptr: TaggedPtr::default(),
                input_words: &input_words,
                bloom_filter: None,
            },
            &mut ledger,
            &mut trail,
        )
        .unwrap();
    assert_eq!(orch_proj.total_layers, 3);
    assert!(updated_ledger.latest_step().is_some());
    assert_eq!(ledger.telemetry_mini_ledger, updated_ledger.steps);
    assert_eq!(swarm_rep.evolved_ptr.raw(), trail.current_ptr);
}

#[test]
fn test_synaptic_transitions_cross_trajectory_and_adaptive_cascade() {
    // 1. Temporal synaptic momentum differentiates bottom-up integration from peel streaks.
    let mut bu_cycle = CognitiveTrajectoryLedger::new();
    bu_cycle.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::PointerBit,
        CognitiveOpKind::DecomposePeel,
        6,
    ));
    bu_cycle.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::RouteSubHead,
        6,
    ));
    bu_cycle.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::SynthesizeFuse,
        7,
    ));
    let (bu_mom, td_mom, rep_streak, net_vel) = bu_cycle.synaptic_transition_stats();
    assert!(bu_mom > td_mom);
    assert_eq!(rep_streak, 0);
    assert!(net_vel > 0);

    let mut peel_streak = CognitiveTrajectoryLedger::new();
    peel_streak.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::DecomposePeel,
        6,
    ));
    peel_streak.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::DecomposePeel,
        7,
    ));
    let (p_bu, p_td, _, p_vel) = peel_streak.synaptic_transition_stats();
    assert!(p_td > p_bu);
    assert!(p_vel < 0);

    // 2. Cross-trajectory activation catches omitted decomposition in a shallow draft.
    let mut shallow_draft = CognitiveTrajectoryLedger::new();
    shallow_draft.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::Observe,
        3,
    ));
    let q_words = load_q7_words_64(&[24i8; MAX_ATTENTION_HEADS]);
    let d_words = load_q7_words_64(&[20i8; MAX_ATTENTION_HEADS]);
    let cross =
        peel_streak.evaluate_cross_trajectory_activation(&shallow_draft, &q_words, &d_words);
    assert_eq!(cross.op_kind(), CognitiveOpKind::DecomposePeel);
    assert!(cross.peel_gate_q7 >= 80);

    // 3. Adaptive early-exit skips redundant deeper layers when residual energy is low and
    //    fuse_gate_q7 is already saturated.
    let mut substrate = crate::stigmergy::substrate::StigmergicSubstrate::new();
    let mut early_slot = LedgerSlot::default();
    let mut early_ledger = CognitiveTrajectoryLedger::new();
    early_ledger.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::PointerBit,
        CognitiveOpKind::DecomposePeel,
        5,
    ));
    early_ledger.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::SynthesizeFuse,
        7,
    ));
    early_ledger.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::CrystallizeCommit,
        7,
    ));
    early_ledger.write_to_ledger_slot(&mut early_slot);

    let aligned_words = load_q7_words_64(&[8i8; MAX_ATTENTION_HEADS]);
    let aligned_frame =
        ObjectiveFact::new([8i8; MAX_ATTENTION_HEADS], 3.0, 0, 1).synthesize_gateway_frame();
    let early_layers = [
        ScaleLayerNode::new(aligned_frame, StigmergicScaleGranularity::MacroCoalition),
        ScaleLayerNode::new(aligned_frame, StigmergicScaleGranularity::CellTrail),
        ScaleLayerNode::new(aligned_frame, StigmergicScaleGranularity::PointerBit),
    ];
    let mut early_trail = PointerTrail::default();
    let (_, out_early, _) = substrate
        .orchestrate_adaptive_depth_n(
            &early_layers,
            3,
            ScaleCascadeInput {
                input_ptr: TaggedPtr::default(),
                input_words: &aligned_words,
                bloom_filter: None,
            },
            &mut early_slot,
            &mut early_trail,
        )
        .unwrap();
    // Early-exit stopped after layer 0, so only 1 step was added (3 -> 4 total steps).
    assert_eq!(out_early.op_counts().1, 4);
}

#[test]
fn test_cognitive_circuit_plasticity_composition_and_unified_container_roundtrip() {
    let mut circuit = CognitiveTrajectoryLedger::new();
    let act_fuse = CognitiveCircuitActivation {
        peel_gate_q7: 30,
        fuse_gate_q7: 110,
        repulse_sensitivity_q7: 20,
        selected_op: CognitiveOpKind::SynthesizeFuse as u8,
        preferred_next_scale: StigmergicScaleGranularity::MacroCoalition as u8,
        intensity_q7: 96,
        circuit_resonance_q7: 48,
        trajectory_depth: 1,
    };
    circuit.reinforce_from_activation(&act_fuse, true);
    let first_depth = circuit.op_counts().1;
    assert_eq!(first_depth, 1);
    let int_before = circuit.latest_step().unwrap().intensity();
    // Repeating the identical converged activation strengthens in-place (homeostatic saturation)
    // instead of flooding all 8 slots.
    circuit.reinforce_from_activation(&act_fuse, true);
    assert_eq!(circuit.op_counts().1, 1);
    assert!(circuit.latest_step().unwrap().intensity() > int_before);

    // Divergent outcome with high repulsion sensitivity shifts via RepulseAudit.
    let act_repulse = CognitiveCircuitActivation {
        repulse_sensitivity_q7: 96,
        preferred_next_scale: StigmergicScaleGranularity::TypeGateway as u8,
        intensity_q7: 100,
        ..Default::default()
    };
    circuit.reinforce_from_activation(&act_repulse, false);
    assert_eq!(circuit.op_counts().1, 2);
    assert_eq!(
        circuit.latest_step().unwrap().op(),
        CognitiveOpKind::RepulseAudit
    );

    // Compositional blending between two circuits preserves both bounds and mid-blend.
    let mut overlay = CognitiveTrajectoryLedger::new();
    overlay.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::DecomposePeel,
        6,
    ));
    assert_eq!(circuit.compose_persona_circuit(&overlay, 0), circuit);
    assert_eq!(circuit.compose_persona_circuit(&overlay, 128), overlay);
    let blended = circuit.compose_persona_circuit(&overlay, 64);
    assert_ne!(blended, circuit);
    assert!(blended.op_counts().1 >= 2);

    // Zero-copy unified .lmnk roundtrip via to_persona_ledger_slot & cognitive_trajectory_ledger.
    let slot = blended.to_persona_ledger_slot();
    let anchors = [PhaseAnchorRecord::new_empty(1, ATTRACTOR_SLOT_PERSONA, 0)];
    let trails = [PointerTrail::default()];
    let req_len =
        LmnkContainerView::required_standard_byte_len(anchors.len(), trails.len(), false, true, 0)
            .unwrap();
    let mut buf = [0u8; 1024];
    let off = buf.as_ptr().align_offset(128);
    let written = LmnkContainerView::write_standard_sections_to_bytes(
        StandardSectionsInput {
            container_flags: LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            anchors: &anchors,
            trails: &trails,
            bloom: None,
            ledger: Some(&slot),
            provenance_reports: &[],
            timestamp_epoch_sec: 1_700_000_000,
        },
        &mut buf[off..off + req_len],
    )
    .unwrap();
    let view = LmnkContainerView::parse(&buf[off..off + written]).unwrap();
    assert_eq!(view.cognitive_trajectory_ledger(), Some(blended));
}

#[test]
fn test_hybrid_scale_circuit_and_adaptive_threshold_policy() {
    let mut micro = CognitiveTrajectoryLedger::new();
    micro.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::DecomposePeel,
        6,
    ));
    let mut macro_c = CognitiveTrajectoryLedger::new();
    macro_c.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::SynthesizeFuse,
        7,
    ));

    let hybrid = HybridScaleCircuit::new(micro, macro_c);
    assert_eq!(hybrid.layer_count(), 2);
    assert_eq!(
        *hybrid.circuit_for_scale(StigmergicScaleGranularity::PointerBit),
        micro
    );
    assert_eq!(
        *hybrid.circuit_for_scale(StigmergicScaleGranularity::MacroCoalition),
        macro_c
    );

    let input_words = load_q7_words_64(&[20i8; MAX_ATTENTION_HEADS]);
    let p_coarse = hybrid.project_scale_layer(&input_words, &macro_c, 0, 2);
    let p_fine = hybrid.project_scale_layer(&input_words, &micro, 1, 2);
    assert_eq!(
        p_coarse.granularity,
        StigmergicScaleGranularity::MacroCoalition
    );
    assert_eq!(p_fine.granularity, StigmergicScaleGranularity::CellTrail);
    assert_ne!(p_coarse.peeled_words, p_fine.peeled_words);

    let strict_policy = AdaptiveThresholdPolicy::new(StigmergicThresholds {
        reinforcement_gain: 1.20,
        evaporation_retain: 0.70,
        ..StigmergicThresholds::default()
    });
    let relaxed_policy = AdaptiveThresholdPolicy::new(StigmergicThresholds {
        reinforcement_gain: 0.25,
        evaporation_retain: 0.96,
        ..StigmergicThresholds::default()
    });
    assert!(strict_policy.early_exit_fuse_floor_q7() > relaxed_policy.early_exit_fuse_floor_q7());
    assert!(strict_policy.peel_escalation_floor_q7() < relaxed_policy.peel_escalation_floor_q7());

    let mut substrate = crate::stigmergy::substrate::StigmergicSubstrate::new();
    let mut slot = LedgerSlot::default();
    let mut trail = PointerTrail::default();
    let layers = [hybrid];
    let (proj, out_ledger, _) = substrate
        .orchestrate_adaptive_depth_n_with_policy(
            &layers,
            1,
            ScaleCascadeInput {
                input_ptr: TaggedPtr::default(),
                input_words: &input_words,
                bloom_filter: None,
            },
            &strict_policy,
            &mut slot,
            &mut trail,
        )
        .unwrap();
    assert_eq!(proj.total_layers, 1);
    assert!(out_ledger.latest_step().is_some());
}

#[test]
fn test_scalable_ram_cognitive_circuit_bank_and_folding() {
    let mut low: LowSpecCognitiveBank = CognitiveCircuitBank::new();
    let mut high: HighCapCognitiveBank = CognitiveCircuitBank::new();
    assert_eq!(size_of::<LowSpecCognitiveBank>(), 64);
    assert_eq!(size_of::<HighCapCognitiveBank>(), 1024);

    // Pushing 20 steps into an 8-unit bank spills evicted steps hierarchically into units[1..3].
    for idx in 0..20u8 {
        let step = CognitiveStepOp::pack(
            StigmergicScaleGranularity::from_u8(idx & 3),
            if (idx & 1) == 0 {
                CognitiveOpKind::DecomposePeel
            } else {
                CognitiveOpKind::SynthesizeFuse
            },
            (idx % 6) + 1,
        );
        low.push_step_hierarchical(step);
        high.push_step_hierarchical(step);
    }
    assert_eq!(low.units[0].op_counts().1, 8);
    assert_eq!(low.units[1].op_counts().1, 8);
    assert_eq!(low.units[2].op_counts().1, 4);
    assert_eq!(low.units[3].op_counts().1, 0);

    let input_words = load_q7_words_64(&[24i8; MAX_ATTENTION_HEADS]);
    let bias_words = load_q7_words_64(&[22i8; MAX_ATTENTION_HEADS]);
    let bank_act = high.evaluate_bank_activation(
        &bias_words,
        &input_words,
        StigmergicScaleGranularity::CellTrail,
    );
    assert_eq!(bank_act.trajectory_depth, 20);
    assert!(bank_act.intensity_q7 > 0);

    // Bank acts directly as a ManifoldGatewaySignal via consolidated 8-byte folding.
    let frame = high.synthesize_gateway_frame();
    assert_ne!(frame.lineage_flags & CAUSAL_AXIS_SUBJECTIVE_PERSONA, 0);

    // Slice implementation allows runtime-sized RAM banks without fixed const N.
    let mut dynamic_units = [CognitiveTrajectoryLedger::new(); 4];
    dynamic_units.as_mut_slice().reinforce_bank(&bank_act, true);
    assert!(dynamic_units[0].latest_step().is_some());
}

#[test]
fn test_ram_phase_kv_cache_exact_hit_resonance_lru_and_gateway_synthesis() {
    let mut cache: LowSpecPhaseKvCache = RamPhaseKvCache::new();
    assert_eq!(size_of::<LowSpecPhaseKvCache>(), 16 * 128);

    let mut circuit_a = CognitiveTrajectoryLedger::new();
    circuit_a.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::DecomposePeel,
        6,
    ));
    let phases_a = [24i8; MAX_ATTENTION_HEADS];
    let act_a = CognitiveCircuitActivation {
        peel_gate_q7: 96,
        intensity_q7: 96,
        ..Default::default()
    };

    cache.store_context_keyframe(KvKeyframeStoreInput {
        context_hash: 0xAAAA_0001,
        token_index: 10,
        anchor_id: 101,
        phases: &phases_a,
        circuit: circuit_a,
        activation: act_a,
        epoch: 1,
    });

    // 1. Exact context hash hit returns slot 0 with exact_hit = true and bumps reuse_count.
    let q_words_a = load_q7_words_64(&phases_a);
    let (idx, exact, res) = cache
        .lookup_context_or_resonate(0xAAAA_0001, &q_words_a, 16, 2)
        .expect("exact hit");
    assert_eq!(idx, 0);
    assert!(exact);
    assert_eq!(res, 127);
    assert_eq!(cache.slots[0].reuse_count, 1);
    assert!(cache.slots[0].verify_integrity().is_ok());

    // 2. Unknown hash still resonates associatively when query phases align with stored keyframe.
    let q_words_near = load_q7_words_64(&[22i8; MAX_ATTENTION_HEADS]);
    let (res_idx, exact_2, res_score) = cache
        .lookup_context_or_resonate(0xBBBB_0002, &q_words_near, 16, 3)
        .expect("associative resonance hit");
    assert_eq!(res_idx, 0);
    assert!(!exact_2);
    assert!(res_score >= 16);

    // 3. Cached entry directly synthesizes a composite GatewayFrame (Fact + Persona circuit).
    let frame = cache.slots[0].synthesize_gateway_frame();
    assert_ne!(frame.lineage_flags & CAUSAL_AXIS_OBJECTIVE_FACT, 0);
    assert_ne!(frame.lineage_flags & CAUSAL_AXIS_SUBJECTIVE_PERSONA, 0);

    // 4. LRU eviction replaces oldest epoch when capacity is full.
    let mut tiny = [RamPhaseKvEntry::default(); 2];
    tiny.as_mut_slice()
        .store_context_keyframe(KvKeyframeStoreInput {
            context_hash: 1,
            token_index: 0,
            anchor_id: 1,
            phases: &phases_a,
            circuit: circuit_a,
            activation: act_a,
            epoch: 10,
        });
    tiny.as_mut_slice()
        .store_context_keyframe(KvKeyframeStoreInput {
            context_hash: 2,
            token_index: 0,
            anchor_id: 2,
            phases: &phases_a,
            circuit: circuit_a,
            activation: act_a,
            epoch: 20,
        });
    tiny.as_mut_slice()
        .store_context_keyframe(KvKeyframeStoreInput {
            context_hash: 3,
            token_index: 0,
            anchor_id: 3,
            phases: &phases_a,
            circuit: circuit_a,
            activation: act_a,
            epoch: 30,
        });
    assert_eq!(tiny[0].context_hash, 3);
    assert_eq!(tiny[1].context_hash, 2);
}

#[test]
fn test_ram_phase_kv_gop_cache_pframe_ring_and_zero_forward_seek() {
    let mut gop_cache: LowSpecPhaseKvGopCache = RamPhaseKvGopCache::new();
    let mut circuit = CognitiveTrajectoryLedger::new();
    circuit.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::SynthesizeFuse,
        5,
    ));
    let act = CognitiveCircuitActivation::default();
    let p0 = [10i8; MAX_ATTENTION_HEADS];
    let p1 = [18i8; MAX_ATTENTION_HEADS];
    let p2 = [28i8; MAX_ATTENTION_HEADS];
    let p3 = [36i8; MAX_ATTENTION_HEADS];

    let slot = gop_cache
        .store_keyframe_and_reset_gop(KvKeyframeStoreInput {
            context_hash: 0xCAFE_0001,
            token_index: 100,
            anchor_id: 7,
            phases: &p0,
            circuit,
            activation: act,
            epoch: 1,
        })
        .expect("slot allocated");
    assert_eq!(gop_cache.slot_pframe_count(slot), 0);

    gop_cache.append_context_pframe(slot, &p1, 1, 2).unwrap();
    gop_cache.append_context_pframe(slot, &p2, 1, 3).unwrap();
    gop_cache.append_context_pframe(slot, &p3, 1, 4).unwrap();
    assert_eq!(gop_cache.slot_pframe_count(slot), 3);
    assert_eq!(gop_cache.total_stored_pframes(), 3);

    let (restored, log_p, r_bar, repulsed) = gop_cache.seek_slot_history(slot, 1, 0.35, 5).unwrap();
    assert_eq!(restored, p3);
    assert!(log_p.is_finite());
    assert!(r_bar > 0.9);
    assert_eq!(repulsed, 0);

    let frame = gop_cache
        .synthesize_slot_history_gateway_frame(slot, 1, 0.35, 6)
        .unwrap();
    assert_ne!(frame.lineage_flags & CAUSAL_AXIS_OBJECTIVE_FACT, 0);
    assert_ne!(frame.lineage_flags & CAUSAL_AXIS_SUBJECTIVE_PERSONA, 0);
    assert_eq!(gop_cache.entries()[slot].reuse_count, 2);

    // Small-amplitude orthogonal, opposite, and zero vectors must not falsely collide.
    let mut orth_cache: LowSpecPhaseKvGopCache = RamPhaseKvGopCache::new();
    let mut topic_a = [0i8; MAX_ATTENTION_HEADS];
    topic_a[0] = -1;
    let mut topic_b = [0i8; MAX_ATTENTION_HEADS];
    topic_b[4] = -1;
    let mut topic_neg = [0i8; MAX_ATTENTION_HEADS];
    topic_neg[0] = 1;
    orth_cache
        .store_keyframe_and_reset_gop(KvKeyframeStoreInput {
            context_hash: 0x1111,
            token_index: 1,
            anchor_id: 1,
            phases: &topic_a,
            circuit,
            activation: act,
            epoch: 1,
        })
        .unwrap();
    assert!(
        orth_cache
            .lookup_context_or_resonate(0x2222, &load_q7_words_64(&topic_b), 16, 2)
            .is_none()
    );
    assert!(
        orth_cache
            .lookup_context_or_resonate(0x3333, &load_q7_words_64(&topic_neg), 16, 3)
            .is_none()
    );
    assert!(
        orth_cache
            .lookup_context_or_resonate(0x4444, &[0u64; NUM_STIGMERGIC_GRANULES], 16, 4)
            .is_none()
    );

    // Autonomous S^(H-1) concentration-of-measure resonance floor scales inversely with sqrt(H)
    // and steps RamPhaseKvGopCache through SkippedLiminal -> StoredKeyframe -> ResonatedSeeked.
    let policy = AdaptiveThresholdPolicy::default();
    let floor_h8 = policy.kv_resonance_floor_for_heads_q7(8, 0, 16);
    let floor_h24 = policy.kv_resonance_floor_for_heads_q7(24, 0, 16);
    let floor_h64 = policy.kv_resonance_floor_for_heads_q7(64, 0, 16);
    assert!(floor_h8 > floor_h24 && floor_h24 > floor_h64);
    assert!((46..=58).contains(&floor_h24));

    let mut auto_cache: LowSpecPhaseKvGopCache = RamPhaseKvGopCache::new();
    let mut auto_circuit = circuit;
    let empty_frame = GatewayFrame::default();
    assert_eq!(
        auto_cache
            .step_with_gateway_and_override(
                KvGatewayStepInput {
                    context_hash: 0,
                    token_index: 1,
                    frame: &empty_frame,
                    activation: act,
                    policy: &policy,
                    override_min_resonance_q7: None,
                    epoch: 1,
                },
                &mut auto_circuit,
            )
            .unwrap(),
        RamPhaseKvStepOutcome::SkippedLiminal
    );
    let mut fact_arr = [0i8; MAX_ATTENTION_HEADS];
    fact_arr[0] = 24;
    fact_arr[1] = -12;
    let mut fact_frame = ObjectiveFact::new(fact_arr, 2.5, 0, 77).synthesize_gateway_frame();
    fact_frame.active_head_count = 24;
    assert!(matches!(
        auto_cache
            .step_with_gateway_and_override(
                KvGatewayStepInput {
                    context_hash: 0x5555,
                    token_index: 10,
                    frame: &fact_frame,
                    activation: act,
                    policy: &policy,
                    override_min_resonance_q7: None,
                    epoch: 2,
                },
                &mut auto_circuit,
            )
            .unwrap(),
        RamPhaseKvStepOutcome::StoredKeyframe { slot_idx: 0, .. }
    ));
    assert!(matches!(
        auto_cache
            .step_with_gateway_and_override(
                KvGatewayStepInput {
                    context_hash: 0x6666,
                    token_index: 12,
                    frame: &fact_frame,
                    activation: act,
                    policy: &policy,
                    override_min_resonance_q7: None,
                    epoch: 3,
                },
                &mut auto_circuit,
            )
            .unwrap(),
        RamPhaseKvStepOutcome::ResonatedSeeked {
            slot_idx: 0,
            exact_hit: false,
            pframe_count: 1,
            ..
        }
    ));

    // When P-Frame count reaches GOP length P=4 (or seek repulsion fires), the next hit
    // triggers FrameDirective::ReAnchorKeyframe via SystemSpineState and resets pframe_count to 0.
    let mut delta_arr = fact_arr;
    for step_idx in 0..3u32 {
        delta_arr[2] = (step_idx as i8 + 1) * 4;
        let mut step_frame = ObjectiveFact::new(delta_arr, 2.5, 0, 77).synthesize_gateway_frame();
        step_frame.active_head_count = 24;
        let out = auto_cache
            .step_with_gateway_and_override(
                KvGatewayStepInput {
                    context_hash: 0x5555,
                    token_index: 14 + step_idx,
                    frame: &step_frame,
                    activation: act,
                    policy: &policy,
                    override_min_resonance_q7: None,
                    epoch: 4 + step_idx,
                },
                &mut auto_circuit,
            )
            .unwrap();
        assert!(matches!(
            out,
            RamPhaseKvStepOutcome::ResonatedSeeked {
                slot_idx: 0,
                exact_hit: true,
                ..
            }
        ));
    }
    assert_eq!(auto_cache.slot_pframe_count(0), 4);

    delta_arr[2] = 20;
    let mut reanchor_frame = ObjectiveFact::new(delta_arr, 2.5, 0, 77).synthesize_gateway_frame();
    reanchor_frame.active_head_count = 24;
    let reanchored = auto_cache
        .step_with_gateway_and_override(
            KvGatewayStepInput {
                context_hash: 0x5555,
                token_index: 20,
                frame: &reanchor_frame,
                activation: act,
                policy: &policy,
                override_min_resonance_q7: None,
                epoch: 8,
            },
            &mut auto_circuit,
        )
        .unwrap();
    assert!(matches!(
        reanchored,
        RamPhaseKvStepOutcome::ResonatedSeeked {
            slot_idx: 0,
            exact_hit: true,
            pframe_count: 0,
            ..
        }
    ));
    assert_eq!(auto_cache.base.entries()[0].anchor_phases[2], 20);

    // Guard/repulsion seek triggers immediate re-anchoring before P=4 expires.
    let mut opp_arr = [-120i8; MAX_ATTENTION_HEADS];
    opp_arr[0] = -96;
    let mut opp_frame = ObjectiveFact::new(opp_arr, 2.5, 0, 78).synthesize_gateway_frame();
    opp_frame.active_head_count = 24;
    let _ = auto_cache
        .step_with_gateway_and_override(
            KvGatewayStepInput {
                context_hash: 0x5555,
                token_index: 21,
                frame: &opp_frame,
                activation: act,
                policy: &policy,
                override_min_resonance_q7: None,
                epoch: 9,
            },
            &mut auto_circuit,
        )
        .unwrap();
    assert_eq!(auto_cache.slot_pframe_count(0), 1);
    auto_cache.pframes[0][0] =
        crate::foundation::KvMotionDeltaPacket::new([-120; MAX_ATTENTION_HEADS]);
    let repulsed_out = auto_cache
        .step_with_gateway_and_override(
            KvGatewayStepInput {
                context_hash: 0x5555,
                token_index: 22,
                frame: &reanchor_frame,
                activation: act,
                policy: &policy,
                override_min_resonance_q7: None,
                epoch: 10,
            },
            &mut auto_circuit,
        )
        .unwrap();
    assert!(matches!(
        repulsed_out,
        RamPhaseKvStepOutcome::ResonatedSeeked {
            slot_idx: 0,
            exact_hit: true,
            pframe_count: 0,
            seek_repulsed,
            ..
        } if seek_repulsed > 0
    ));

    // 4-tuple (SystemConstraint, ObjectiveFact, SubjectivePersona, GuardPhaseAnchor) composes guard severity,
    // and CognitiveTrajectoryLedger::modulate_gateway_frame_at_scale routes `SubheadRouter`/`EntropyController`
    // cleanly.
    let guard = GuardPhaseAnchor::new([24i8; MAX_ATTENTION_HEADS], 0.85, 0.5);
    let guard_composite = (
        SystemConstraint::new((), 0.0, 0.0, 0xFF),
        ObjectiveFact::new([24i8; MAX_ATTENTION_HEADS], 2.0, 0, 11),
        SubjectivePersona::new([20i8; MAX_ATTENTION_HEADS], 1.5, 0, 22),
        guard,
    );
    let mut g_frame = guard_composite.synthesize_gateway_frame();
    assert!(g_frame.repulsion_severity >= 0.55);
    assert_ne!(g_frame.lineage_flags & CAUSAL_AXIS_CONTRADICTION_GUARD, 0);

    let mut peel_ledger = CognitiveTrajectoryLedger::new();
    peel_ledger.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::CellTrail,
        CognitiveOpKind::DecomposePeel,
        7,
    ));
    peel_ledger.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::PointerBit,
        CognitiveOpKind::DecomposePeel,
        7,
    ));
    let mut peel_frame = GatewayFrame::default();
    let mut arr_a = [16i8; MAX_ATTENTION_HEADS];
    for h in &mut arr_a[8..16] {
        *h = -80;
    }
    let w_a = load_q7_words_64(&arr_a);
    let w_b = load_q7_words_64(&[16i8; MAX_ATTENTION_HEADS]);
    let peel_act = peel_ledger.modulate_gateway_frame_at_scale(
        &mut peel_frame,
        &w_a,
        &w_b,
        StigmergicScaleGranularity::CellTrail,
    );
    assert_eq!(peel_act.op_kind(), CognitiveOpKind::DecomposePeel);
    assert!(peel_frame.subhead_fact_heads > GatewayFrame::default().subhead_fact_heads);
    assert_eq!(
        peel_frame.subhead_fact_heads
            + peel_frame.subhead_logic_heads
            + peel_frame.subhead_explore_heads,
        MAX_ATTENTION_HEADS as u8
    );

    let mut fuse_ledger = CognitiveTrajectoryLedger::new();
    fuse_ledger.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::TypeGateway,
        CognitiveOpKind::SynthesizeFuse,
        7,
    ));
    fuse_ledger.push_step(CognitiveStepOp::pack(
        StigmergicScaleGranularity::MacroCoalition,
        CognitiveOpKind::CrystallizeCommit,
        7,
    ));
    let mut fuse_slot = LedgerSlot::default();
    fuse_ledger.write_to_ledger_slot(&mut fuse_slot);
    let mut substrate =
        crate::stigmergy::substrate::StigmergicSubstrate::with_dimensions_and_thresholds(
            64,
            64,
            16,
            crate::stigmergy::substrate::StigmergicThresholds::default(),
        )
        .unwrap();
    let mut trail = PointerTrail::default();
    let policy = AdaptiveThresholdPolicy::default();
    let sys = SystemConstraint::new((), 0.0, 0.0, 0xFF);
    let probe = ManifoldProbeBaseline::new();
    let geom = RopeGeometryDescriptor {
        layout: RopePairLayout::HalfSplit,
        head_dim: 64,
        rotary_dim: 64,
        num_heads: 64,
    };
    let sig = (
        ObjectiveFact::new([18i8; MAX_ATTENTION_HEADS], 2.0, 0, 1),
        SubjectivePersona::new([18i8; MAX_ATTENTION_HEADS], 1.5, 0, 2),
    );
    let (mod_frame, _, _) = substrate
        .orchestrate_gateway_layer_with_core_layers_safe(
            GatewayLayerInput {
                signal: &sig,
                geometry: &geom,
                core_ctx: None,
                scale: StigmergicScaleGranularity::MacroCoalition,
                entropy_h_tilde: 0.35,
                unresolved_damping_u7: 128,
            },
            GatewayPolicyContext {
                policy: &policy,
                system: &sys,
                probe: &probe,
            },
            &mut fuse_slot,
            &mut trail,
            GatewayLayerTargets {
                query_tensor: None,
                rank1_hidden: None,
                logits_opt: None,
            },
        )
        .unwrap();
    assert!(mod_frame.effective_inv_temp > 1.0);
    assert!(mod_frame.subhead_logic_heads >= GatewayFrame::default().subhead_logic_heads);

    // Identical runtime inputs yield distinct StigmergicThresholds and distinct
    // StigmergicActionDecision regimes purely from Analytical vs Integrative circuit habits.
    let peel_thresh = peel_ledger.bias_stigmergic_thresholds(
        StigmergicThresholds::default(),
        StigmergicScaleGranularity::CellTrail,
    );
    let fuse_thresh = fuse_ledger.bias_stigmergic_thresholds(
        StigmergicThresholds::default(),
        StigmergicScaleGranularity::MacroCoalition,
    );
    assert!(peel_thresh.subhead_route_density_thresh < fuse_thresh.subhead_route_density_thresh);
    assert!(fuse_thresh.crystallize_density_thresh < peel_thresh.crystallize_density_thresh);

    let fixed_trail = PointerTrail {
        stigmergic_density: 1.95,
        step_count: 8,
        ..Default::default()
    };
    let mut peel_hub_slot = LedgerSlot::default();
    peel_ledger.write_to_ledger_slot(&mut peel_hub_slot);
    let mut peel_trail = fixed_trail;
    let (_, _, dec_peel) = substrate
        .orchestrate_gateway_layer_with_core_layers_safe(
            GatewayLayerInput {
                signal: &sig,
                geometry: &geom,
                core_ctx: None,
                scale: StigmergicScaleGranularity::CellTrail,
                entropy_h_tilde: 0.45,
                unresolved_damping_u7: 128,
            },
            GatewayPolicyContext {
                policy: &policy,
                system: &sys,
                probe: &probe,
            },
            &mut peel_hub_slot,
            &mut peel_trail,
            GatewayLayerTargets {
                query_tensor: None,
                rank1_hidden: None,
                logits_opt: None,
            },
        )
        .unwrap();

    let mut fuse_hub_slot = LedgerSlot::default();
    fuse_ledger.write_to_ledger_slot(&mut fuse_hub_slot);
    let mut fuse_trail = fixed_trail;
    let (_, _, dec_fuse) = substrate
        .orchestrate_gateway_layer_with_core_layers_safe(
            GatewayLayerInput {
                signal: &sig,
                geometry: &geom,
                core_ctx: None,
                scale: StigmergicScaleGranularity::MacroCoalition,
                entropy_h_tilde: 0.45,
                unresolved_damping_u7: 128,
            },
            GatewayPolicyContext {
                policy: &policy,
                system: &sys,
                probe: &probe,
            },
            &mut fuse_hub_slot,
            &mut fuse_trail,
            GatewayLayerTargets {
                query_tensor: None,
                rank1_hidden: None,
                logits_opt: None,
            },
        )
        .unwrap();
    assert_ne!(dec_peel, dec_fuse);
    let _ = &mut g_frame;
}
