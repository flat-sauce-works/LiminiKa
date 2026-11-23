// name: src/core/src/runtime/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration unit tests for Layer 4 (`runtime`: Gateway, Horizon, ABI, and relocated
//! cross-layer integration tests).

use core::f32::consts::PI;
use core::mem::size_of;

use super::*;
use crate::drivers::{
    DEFAULT_HYSTERESIS_BAND, DEFAULT_TAU_FLUX, DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME,
    ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION,
};
use crate::foundation::{RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget};
use crate::prelude::*;
use crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7;
use crate::stigmergy::with_slot_pair_coupled_symmetric;

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
    assert_eq!(report.attribution_tag, ROOT_CAUSE_NOMINAL);
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
        col_report.attribution_tag,
        ROOT_CAUSE_PERSONA_FACT_COLLISION
    );
    assert_eq!(col_frame.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(col_frame.target_anchor_id, 77);

    let guard = ContradictionGuard::new(ContrastivePhaseBloomFilter::new(), 0.65, 3);
    let rep_frame = (&sys, &fact, &persona_aligned, &guard).synthesize_gateway_frame();
    let rep_report = rep_frame.to_causal_report();
    assert!(rep_report.is_cohomology_repulsion());
    assert_eq!(rep_report.attribution_tag, ROOT_CAUSE_COHOMOLOGY_REPULSION);
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
    let (f1, idx1, id1, _coh1, mass1, spawned1) =
        PhaseAnchorRecord::step_online_toroidal_kmeans_signal_64(
            &mut centroids,
            &fact_a,
            DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
            Some(&mut trail),
        );
    assert!(spawned1 && idx1 == 0 && id1 == 101 && trail.cluster_id == 101 && mass1 > 0.0);

    let fact_a2 = ObjectiveFact::new([30i8; MAX_ATTENTION_HEADS], 3.0, 0x0010_0000, 101);
    let (f2, idx2, id2, coh2, mass2, spawned2) =
        PhaseAnchorRecord::step_online_toroidal_kmeans_signal_64(
            &mut centroids,
            &fact_a2,
            DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
            Some(&mut trail),
        );
    assert!(!spawned2 && idx2 == 0 && id2 == 101 && coh2 > 0.95 && mass2 > mass1);
    assert_eq!(centroids[0].sample_count, 2);

    let persona_b = SubjectivePersona::new([-80i8; MAX_ATTENTION_HEADS], 2.8, 0x0024_0000, 202);
    let (_f3, idx3, id3, _coh3, mass3, spawned3) =
        PhaseAnchorRecord::step_online_toroidal_kmeans_signal_64(
            &mut centroids,
            &persona_b,
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
        LMNK_CONTAINER_FLAG_COMPILED_CACHE,
        &centroids[..2],
        &trails_arr,
        Some(&bloom),
        Some(&ledger),
        &[],
        1_774_900_000,
        &mut container_buf.data,
    )
    .unwrap();
    let no_rep_view = LmnkContainerView::parse(&container_buf.data[..no_rep_len]).unwrap();
    assert_eq!(no_rep_view.phase_anchor_count(), 2);

    let total_len = LmnkContainerView::write_standard_sections_to_bytes(
        LMNK_CONTAINER_FLAG_COMPILED_CACHE,
        &centroids,
        &trails_arr,
        Some(&bloom),
        Some(&ledger),
        &reports_arr,
        1_774_900_000,
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
            128,
            1.0,
            0.25,
            0.10,
            0.50,
            0.35,
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
            TaggedPtr::default(),
            &input_words,
            &mut ledger,
            &mut trail,
            None,
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
            TaggedPtr::default(),
            &aligned_words,
            &mut early_slot,
            &mut early_trail,
            None,
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
        LMNK_CONTAINER_FLAG_COMPILED_CACHE,
        &anchors,
        &trails,
        None,
        Some(&slot),
        &[],
        1_700_000_000,
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
            TaggedPtr::default(),
            &input_words,
            &mut slot,
            &mut trail,
            None,
            &strict_policy,
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
            ((idx % 6) + 1) as u8,
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

    cache.store_context_keyframe(0xAAAA_0001, 10, 101, &phases_a, circuit_a, act_a, 1);

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
        .store_context_keyframe(1, 0, 1, &phases_a, circuit_a, act_a, 10);
    tiny.as_mut_slice()
        .store_context_keyframe(2, 0, 2, &phases_a, circuit_a, act_a, 20);
    tiny.as_mut_slice()
        .store_context_keyframe(3, 0, 3, &phases_a, circuit_a, act_a, 30);
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
        .store_keyframe_and_reset_gop(0xCAFE_0001, 100, 7, &p0, circuit, act, 1)
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
        .store_keyframe_and_reset_gop(0x1111, 1, 1, &topic_a, circuit, act, 1)
        .unwrap();
    assert!(orth_cache
        .lookup_context_or_resonate(0x2222, &load_q7_words_64(&topic_b), 16, 2)
        .is_none());
    assert!(orth_cache
        .lookup_context_or_resonate(0x3333, &load_q7_words_64(&topic_neg), 16, 3)
        .is_none());
    assert!(orth_cache
        .lookup_context_or_resonate(0x4444, &[0u64; NUM_STIGMERGIC_GRANULES], 16, 4)
        .is_none());

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
            .step_with_gateway_autonomous(0, 1, &empty_frame, &mut auto_circuit, act, &policy, 1)
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
            .step_with_gateway_autonomous(
                0x5555,
                10,
                &fact_frame,
                &mut auto_circuit,
                act,
                &policy,
                2
            )
            .unwrap(),
        RamPhaseKvStepOutcome::StoredKeyframe { slot_idx: 0, .. }
    ));
    assert!(matches!(
        auto_cache
            .step_with_gateway_autonomous(
                0x6666,
                12,
                &fact_frame,
                &mut auto_circuit,
                act,
                &policy,
                3
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
            .step_with_gateway_autonomous(
                0x5555,
                14 + step_idx,
                &step_frame,
                &mut auto_circuit,
                act,
                &policy,
                4 + step_idx,
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
        .step_with_gateway_autonomous(
            0x5555,
            20,
            &reanchor_frame,
            &mut auto_circuit,
            act,
            &policy,
            8,
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
        .step_with_gateway_autonomous(0x5555, 21, &opp_frame, &mut auto_circuit, act, &policy, 9)
        .unwrap();
    assert_eq!(auto_cache.slot_pframe_count(0), 1);
    auto_cache.pframes[0][0] =
        crate::foundation::KvMotionDeltaPacket::new([-120; MAX_ATTENTION_HEADS]);
    let repulsed_out = auto_cache
        .step_with_gateway_autonomous(
            0x5555,
            22,
            &reanchor_frame,
            &mut auto_circuit,
            act,
            &policy,
            10,
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
    // and CognitiveTrajectoryLedger::modulate_gateway_frame_at_scale routes `SubheadRouter`/`EntropyController` cleanly.
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
            &sig,
            &geom,
            None,
            &policy,
            &sys,
            &probe,
            StigmergicScaleGranularity::MacroCoalition,
            0.35,
            &mut fuse_slot,
            &mut trail,
            128,
            None,
            None,
            None,
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
            &sig,
            &geom,
            None,
            &policy,
            &sys,
            &probe,
            StigmergicScaleGranularity::CellTrail,
            0.45,
            &mut peel_hub_slot,
            &mut peel_trail,
            128,
            None,
            None,
            None,
        )
        .unwrap();

    let mut fuse_hub_slot = LedgerSlot::default();
    fuse_ledger.write_to_ledger_slot(&mut fuse_hub_slot);
    let mut fuse_trail = fixed_trail;
    let (_, _, dec_fuse) = substrate
        .orchestrate_gateway_layer_with_core_layers_safe(
            &sig,
            &geom,
            None,
            &policy,
            &sys,
            &probe,
            StigmergicScaleGranularity::MacroCoalition,
            0.45,
            &mut fuse_hub_slot,
            &mut fuse_trail,
            128,
            None,
            None,
            None,
        )
        .unwrap();
    assert_ne!(dec_peel, dec_fuse);
    let _ = &mut g_frame;
}

#[test]
fn epistemic_horizon_separates_liminal_from_internal_conflict_branchlessly() {
    let f0 = ObjectiveFact::new([0i8; MAX_ATTENTION_HEADS], 1.5, 0x0001_0000, 101);
    let f1 = ObjectiveFact::new([20i8; MAX_ATTENTION_HEADS], 1.5, 0x0002_0000, 102);
    let nodes = [f0, f1];

    let mut coupling = RelativeCouplingMatrix::<2>::new();
    coupling.set_symmetric(0, 1, -120);

    // 1. Inside memory horizon and aligned with perspective #0 -> KnownCoherent.
    let q_coherent = [0u64; NUM_STIGMERGIC_GRANULES];
    let (frame_coh, horizon_coh) = ObjectiveFact::synthesize_slice_relative_gateway(
        &nodes,
        coupling.row(0),
        &q_coherent,
        0.35,
    );
    assert_eq!(horizon_coh.regime(), EpistemicRegime::KnownCoherent);
    assert_eq!(horizon_coh.horizon_gate, 1.0);
    assert_eq!(horizon_coh.dominant_anchor_id, 101);
    assert!(frame_coh.effective_energy > 0.5);
    assert!(!frame_coh.is_repulsed());

    // 2. Inside memory horizon yet matching opposing node #1 from perspective #0 -> InternalConflict.
    let q_conflict = load_q7_words_64(&[20i8; MAX_ATTENTION_HEADS]);
    let (frame_conf, horizon_conf) = ObjectiveFact::synthesize_slice_relative_gateway(
        &nodes,
        coupling.row(0),
        &q_conflict,
        0.35,
    );
    assert_eq!(horizon_conf.regime(), EpistemicRegime::InternalConflict);
    assert_eq!(horizon_conf.horizon_gate, 1.0);
    assert_eq!(horizon_conf.opposing_anchor_id, 102);
    assert!(frame_conf.is_repulsed());
    assert!(frame_conf.effective_energy > 0.0);

    // 3. Outside carved memory cut (orthogonal to all memory nodes) -> Liminal zeroes output cleanly.
    let q_outside = load_q7_words_64(&[-64i8; MAX_ATTENTION_HEADS]);
    let (frame_liminal, horizon_liminal) =
        ObjectiveFact::synthesize_slice_relative_gateway(&nodes, coupling.row(0), &q_outside, 0.35);
    assert_eq!(horizon_liminal.regime(), EpistemicRegime::Liminal);
    assert_eq!(horizon_liminal.horizon_gate, 0.0);
    assert_eq!(frame_liminal.effective_energy, 0.0);
    assert_eq!(
        frame_liminal.phase_words_ref(),
        &[0u64; NUM_STIGMERGIC_GRANULES]
    );
    assert_eq!(frame_liminal.mean_phase_tag_q7, 0);
}

#[test]
fn relative_coupling_flips_repulsion_by_active_perspective_across_ten_nodes() {
    let mut nodes = [PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0); 10];
    let mut idx = 0usize;
    while idx < nodes.len() {
        let mut rec = PhaseAnchorRecord::new_empty(
            (idx as u32) + 1,
            ATTRACTOR_SLOT_KNOWLEDGE,
            0x0010_0000 + ((idx as u64) << 12),
        );
        let q7 = ((idx as i32) * 24 - 108) as i8;
        rec.phase_profile = [q7; MAX_ATTENTION_HEADS];
        rec.sample_count = 8;
        rec.phase_gravity_mass = 1.25;
        rec.stamp_crc32();
        nodes[idx] = rec;
        idx += 1;
    }

    // 10-node relative graph: when belonging to node #2, node #1 is mutually opposing (and vice versa).
    let mut matrix = RelativeCouplingMatrix::<10>::new();
    matrix.reinforce_relation_branchless(1, 2, true, 120);
    assert_eq!(matrix.row(2)[1], -120);
    assert_eq!(matrix.row(1)[2], -120);
    assert_eq!(matrix.row(2)[2], 127);

    let q_node1 = load_q7_words_64(&nodes[1].phase_profile);
    let q_node2 = load_q7_words_64(&nodes[2].phase_profile);

    // Perspective #2 treats node #1 as opposing (`ACTION_MODE_PHASE_REPULSE`), while #2 itself stays attractive.
    let (frame_from_2_on_1, hor_2_on_1) =
        PhaseAnchorRecord::synthesize_slice_relative_gateway(&nodes, matrix.row(2), &q_node1, 0.40);
    assert_eq!(hor_2_on_1.regime(), EpistemicRegime::InternalConflict);
    assert_eq!(hor_2_on_1.opposing_anchor_id, nodes[1].anchor_id);
    assert_ne!(
        frame_from_2_on_1.action_mode_flags & ACTION_MODE_PHASE_REPULSE,
        0
    );

    // Switching active perspective to #1 flips roles with zero code change: #1 becomes coherent, #2 opposes.
    let (frame_from_1_on_1, hor_1_on_1) =
        PhaseAnchorRecord::synthesize_slice_relative_gateway(&nodes, matrix.row(1), &q_node1, 0.40);
    assert_eq!(hor_1_on_1.regime(), EpistemicRegime::KnownCoherent);
    assert_eq!(hor_1_on_1.dominant_anchor_id, nodes[1].anchor_id);
    assert_eq!(
        frame_from_1_on_1.action_mode_flags & ACTION_MODE_PHASE_REPULSE,
        0
    );

    let (frame_from_1_on_2, hor_1_on_2) =
        PhaseAnchorRecord::synthesize_slice_relative_gateway(&nodes, matrix.row(1), &q_node2, 0.40);
    assert_eq!(hor_1_on_2.regime(), EpistemicRegime::InternalConflict);
    assert_eq!(hor_1_on_2.opposing_anchor_id, nodes[2].anchor_id);
    assert_ne!(
        frame_from_1_on_2.action_mode_flags & ACTION_MODE_PHASE_REPULSE,
        0
    );
}

#[test]
fn dissipative_overflow_peels_residual_scatters_and_spawns_daughter_with_relative_coupling() {
    let mut pool = [PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0); 4];
    pool[0] = PhaseAnchorRecord::new_empty(11, ATTRACTOR_SLOT_KNOWLEDGE, 0x0010_0000);
    pool[0].phase_profile = [0i8; MAX_ATTENTION_HEADS];
    pool[0].sample_count = 32;
    pool[0].base_energy = 2.0;
    pool[0].phase_gravity_mass = 2.0;
    pool[0].resultant_r_bar = 0.25;
    pool[0].stamp_crc32();

    // Secondary existing attractor captures scattered heterogeneous residuals when aligned.
    pool[1] = PhaseAnchorRecord::new_empty(22, ATTRACTOR_SLOT_KNOWLEDGE, 0x0020_0000);
    pool[1].phase_profile = [48i8; MAX_ATTENTION_HEADS];
    pool[1].sample_count = 4;
    pool[1].base_energy = 1.0;
    pool[1].phase_gravity_mass = 1.0;
    pool[1].resultant_r_bar = 0.90;
    pool[1].stamp_crc32();

    let mut coupling = RelativeCouplingMatrix::<4>::new();
    let mut trail = PointerTrail::default();

    // Case A: Closest to slot #0 (`|22-0| < |48-22|`), yet under high dispersion/saturation pressure,
    // the peeled residual (`22 - 0 = 22`) matches slot #1 (`48`) and scatters onto slot #1.
    let fact_scatter = ObjectiveFact::new([22i8; MAX_ATTENTION_HEADS], 1.5, 0x0030_0000, 90);
    let (_, report_scatter) = PhaseAnchorRecord::step_dissipative_overflow_signal_64(
        &mut pool,
        &fact_scatter,
        0.65,
        Some(&mut trail),
    );
    assert!(report_scatter.overflowed);
    assert!(report_scatter.scattered);
    assert!(!report_scatter.spawned_new);
    assert_eq!(report_scatter.primary_idx, 0);
    assert_eq!(report_scatter.landed_idx, 1);
    assert!(pool[0].verify_integrity());
    assert!(pool[1].verify_integrity());
    coupling.record_dissipative_transition_branchless(&report_scatter, 64);
    assert_eq!(
        coupling.row(usize::from(report_scatter.primary_idx))
            [usize::from(report_scatter.landed_idx)],
        64
    );

    // Case B: Saturated basin cannot absorb an orthogonal sample into any existing node -> autonomous daughter birth!
    let fact_orphan = ObjectiveFact::new([-96i8; MAX_ATTENTION_HEADS], 1.8, 0x0040_0000, 91);
    let (_, report_birth) = PhaseAnchorRecord::step_dissipative_overflow_signal_64(
        &mut pool,
        &fact_orphan,
        0.65,
        Some(&mut trail),
    );
    assert!(report_birth.overflowed);
    assert!(report_birth.spawned_new);
    assert_eq!(report_birth.landed_idx, 2);
    assert!(pool[usize::from(report_birth.landed_idx)].verify_integrity());
    coupling.record_dissipative_transition_branchless(&report_birth, 96);
    assert!(
        coupling.row(usize::from(report_birth.primary_idx))[usize::from(report_birth.landed_idx)]
            < 0
    );
}

#[test]
fn liminal_supertrait_aperture_shifting_and_mermaid_self_introspection() {
    // 1. Verify type-level cognitive aperture and unobserved lacuna complement hierarchy.
    assert_eq!(
        <SystemConstraint<()> as SystemConstraintSignal>::SYSTEM_APERTURE_MASK,
        THRESHOLD_SYSTEM_CONSTRAINT
    );
    assert_eq!(
        <SystemConstraint<()> as SystemConstraintSignal>::SYSTEM_UNOBSERVED_LACUNA_MASK,
        !THRESHOLD_SYSTEM_CONSTRAINT
    );
    assert_eq!(
        <ObjectiveFact<[i8; MAX_ATTENTION_HEADS]> as ObjectiveFactSignal>::FACT_APERTURE_MASK,
        THRESHOLD_OBJECTIVE_FACT
    );
    assert_eq!(
        <ObjectiveFact<[i8; MAX_ATTENTION_HEADS]> as ObjectiveFactSignal>::FACT_UNOBSERVED_LACUNA_MASK,
        !THRESHOLD_OBJECTIVE_FACT
    );
    assert_eq!(
        <SubjectivePersona<[i8; MAX_ATTENTION_HEADS]> as SubjectivePersonaSignal>::PERSONA_APERTURE_MASK,
        THRESHOLD_SUBJECTIVE_PERSONA
    );
    assert_eq!(
        <SubjectivePersona<[i8; MAX_ATTENTION_HEADS]> as SubjectivePersonaSignal>::PERSONA_UNOBSERVED_LACUNA_MASK,
        !THRESHOLD_SUBJECTIVE_PERSONA
    );
    assert_eq!(
        <ContradictionGuard as ContradictionGuardSignal>::GUARD_APERTURE_MASK,
        THRESHOLD_CONTRADICTION_GUARD
    );
    assert_eq!(
        <ContradictionGuard as ContradictionGuardSignal>::GUARD_UNOBSERVED_LACUNA_MASK,
        !THRESHOLD_CONTRADICTION_GUARD
    );
    assert_eq!(
        <EpistemicHorizonReading as ManifoldGatewaySignal>::GATEWAY_APERTURE_MASK,
        THRESHOLD_MANIFOLD_GATEWAY
    );
    assert_eq!(
        <EpistemicHorizonReading as ManifoldGatewaySignal>::GATEWAY_UNOBSERVED_LACUNA_MASK,
        !THRESHOLD_MANIFOLD_GATEWAY
    );
    assert_eq!(
        <crate::stigmergy::substrate::StigmergicSubstrate as StigmergicSubstrateOps>::HUB_APERTURE_MASK,
        THRESHOLD_STIGMERGY_WIDE
    );
    assert_eq!(
        <crate::stigmergy::substrate::StigmergicSubstrate as StigmergicSubstrateOps>::HUB_UNOBSERVED_LACUNA_MASK,
        !THRESHOLD_STIGMERGY_WIDE
    );

    let substrate = crate::stigmergy::substrate::StigmergicSubstrate::default();
    let (cog_bits, ext_bits) = substrate.inspect_substrate_horizon_partition(0x8000_0003);
    assert_eq!(cog_bits, 0x0000_0003);
    assert_eq!(ext_bits, 0x8000_0000);

    // 2. Verify LiminalThresholdState shifts Persona toward Liminal under DecomposePeel + InternalConflict
    //    while widening ObjectiveFact and ContradictionGuard.
    let mut field = LiminalThresholdState::new();
    let init_f1 = field.threshold_floor_q7[1];
    let init_p2 = field.threshold_floor_q7[2];
    let init_g3 = field.threshold_floor_q7[3];
    let init_gain_f1 = field.aperture_gain_q7[1];
    let init_gain_g3 = field.aperture_gain_q7[3];

    let mut peel_ledger = CognitiveTrajectoryLedger::default();
    for _ in 0..4 {
        peel_ledger.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::CellTrail,
            CognitiveOpKind::DecomposePeel,
            6,
        ));
    }
    let conflict_horizon = EpistemicHorizonReading {
        total_memory_support: 1.5,
        peak_positive_affinity: 0.75,
        internal_conflict_energy: 0.90,
        horizon_gate: 1.0,
        conflict_severity: 0.78,
        dominant_anchor_id: 1,
        opposing_anchor_id: 2,
        regime_code: EpistemicRegime::InternalConflict as u8,
        reserved_align: [0; 3],
    };
    field.adapt_from_circuit_and_overflow_branchless(&peel_ledger, &conflict_horizon, None);
    assert!(field.threshold_floor_q7[2] > init_p2);
    assert!(field.threshold_floor_q7[1] < init_f1);
    assert!(field.threshold_floor_q7[3] < init_g3);
    assert!(field.aperture_gain_q7[1] > init_gain_f1);
    assert!(field.aperture_gain_q7[3] > init_gain_g3);
    assert_eq!(field.adaptation_epochs, 1);

    // 3. Verify SynthesizeFuse + scattered overflow widens RelativeCoupling & shifts focal phase.
    let init_rel6 = field.threshold_floor_q7[6];
    let mut fuse_ledger = CognitiveTrajectoryLedger::default();
    for _ in 0..4 {
        fuse_ledger.push_step(CognitiveStepOp::pack(
            StigmergicScaleGranularity::MacroCoalition,
            CognitiveOpKind::SynthesizeFuse,
            6,
        ));
    }
    let coherent_horizon = EpistemicHorizonReading {
        total_memory_support: 1.8,
        peak_positive_affinity: 0.92,
        internal_conflict_energy: 0.0,
        horizon_gate: 1.0,
        conflict_severity: 0.0,
        dominant_anchor_id: 1,
        opposing_anchor_id: 0,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        reserved_align: [0; 3],
    };
    let res_words = load_q7_words_64(&[24i8; MAX_ATTENTION_HEADS]);
    let scatter_report = DissipativeOverflowReport {
        residual_words: res_words,
        primary_anchor_id: 1,
        landed_anchor_id: 2,
        primary_coherence: 0.70,
        scatter_coherence: 0.88,
        overflow_pressure: 0.45,
        updated_gravity_mass: 1.5,
        primary_idx: 0,
        landed_idx: 1,
        overflowed: true,
        scattered: true,
        spawned_new: false,
        _reserved: 0,
    };
    field.adapt_from_circuit_and_overflow_branchless(
        &fuse_ledger,
        &coherent_horizon,
        Some(&scatter_report),
    );
    assert!(field.threshold_floor_q7[6] < init_rel6);
    assert!(field.phase_shift_q7[6] > 0);
    assert_eq!(field.adaptation_epochs, 2);

    // 4. Verify zero-alloc #![no_std] Mermaid flowchart generation captures live state cleanly.
    let snap = LiminalTopologyTelemetrySnapshot::capture(&field, &conflict_horizon, 4, 1);
    let mut stack = CoreStateStack::new();
    stack.liminal = field;
    let empty_overlap = OverlapDiscoveryReport::default();
    let mut buf = [0u8; 2048];
    let len = snap.format_core_topology_mermaid(&stack, &empty_overlap, &mut buf);
    let diagram = core::str::from_utf8(&buf[..len]).expect("valid utf8 mermaid");
    assert!(diagram.starts_with("flowchart TD\n"));
    assert!(diagram.contains("Liminal"));
    assert!(diagram.contains("ManifoldGatewaySignal"));
    assert!(diagram.contains("L2_Substrate"));
    assert!(diagram.contains("Anchors:4"));
    assert!(diagram.contains("Daughters:1"));
}

#[test]
fn fractal_relative_node_self_similarity_across_trail_anchor_and_nested_arrays() {
    let mut t0 = PointerTrail {
        target_anchor_id: 10,
        stigmergic_density: 1.5,
        ..Default::default()
    };
    t0.accumulated_phase_delta = [0i8; MAX_ATTENTION_HEADS];

    let mut t1 = PointerTrail {
        target_anchor_id: 20,
        stigmergic_density: 2.0,
        ..Default::default()
    };
    t1.accumulated_phase_delta = [48i8; MAX_ATTENTION_HEADS];

    // Fixed-size array [Trail; 2] and nested [[Trail; 2]; 2] act as self-similar RelativeAttractorNodes.
    let pair_a = [t0, t0];
    let pair_b = [t1, t1];
    let nested = [pair_a, pair_b];
    assert!(nested.relative_energy() > 6.5);
    assert_eq!(nested[1].relative_anchor_id(), 20);

    let mut coupling = RelativeCouplingMatrix::<2>::new();
    coupling.set_symmetric(0, 1, -120);
    let q_t1 = load_q7_words_64(&t1.accumulated_phase_delta);
    let (frame_nested, horizon_nested) = <[PointerTrail; 2]>::synthesize_slice_relative_gateway(
        &nested,
        coupling.row(0),
        &q_t1,
        0.35,
    );
    assert_eq!(horizon_nested.regime(), EpistemicRegime::InternalConflict);
    assert_eq!(horizon_nested.opposing_anchor_id, 20);
    assert!(frame_nested.is_repulsed());

    // Heterogeneous pair (ObjectiveFact, SubjectivePersona) implements RelativeAttractorNode directly.
    let fact = ObjectiveFact::new([0i8; MAX_ATTENTION_HEADS], 1.2, 0x0010_0000, 31);
    let persona = SubjectivePersona::new([12i8; MAX_ATTENTION_HEADS], 2.5, 0x0020_0000, 32);
    let hetero = (fact, persona);
    assert!((hetero.relative_energy() - 3.7).abs() < 1e-5);
    assert_eq!(hetero.relative_anchor_id(), 32);
}

#[test]
fn stigmergic_overlap_discovery_and_core_stack_feedback() {
    // Two attractors with mixed per-granule profiles:
    // - Granules 0..4 share phase `16` -> overlap core (`overlap == 2`)
    // - Granule 4 covers only node A (`24`) -> overlap fringe (`overlap == 1`)
    // - Granules 5..8 are orthogonal (`-100` vs `+100`), leaving a window around `0..100`
    let mut phases_a = [16i8; MAX_ATTENTION_HEADS];
    let mut phases_b = [16i8; MAX_ATTENTION_HEADS];
    for h in 32..40 {
        phases_a[h] = 24;
        phases_b[h] = -96;
    }
    for h in 40..64 {
        phases_a[h] = -100;
        phases_b[h] = 100;
    }
    let nodes = [
        ObjectiveFact::new(phases_a, 1.5, 0x1000, 101),
        ObjectiveFact::new(phases_b, 1.4, 0x2000, 102),
    ];

    let mut q_phases = [16i8; MAX_ATTENTION_HEADS];
    for h in 32..40 {
        q_phases[h] = 24;
    }
    for h in 40..64 {
        q_phases[h] = 0;
    }
    let query_w = load_q7_words_64(&q_phases);

    // Approximation noise pushes granule 6 (heads 48..56) to `40` outside both attractors (`-100`, `+100`)
    let mut drifted_phases = q_phases;
    for h in 48..56 {
        drifted_phases[h] = 40;
    }
    let drifted_w = load_q7_words_64(&drifted_phases);

    let overlap =
        ObjectiveFact::<[PhaseQ7; MAX_ATTENTION_HEADS]>::evaluate_slice_overlap_and_discovery(
            &nodes, &query_w, &drifted_w, 0.70, 16,
        );
    assert_eq!(overlap.overlap_core_mask & 0x0F, 0x0F);
    assert_eq!(overlap.overlap_fringe_mask & (1 << 4), 1 << 4);
    assert_ne!(overlap.lacuna_exterior_mask & (1 << 6), 0);
    assert_eq!(overlap.discovery_triggered, 1);
    assert_ne!(overlap.discovery_excursion_mask & (1 << 6), 0);
    assert_eq!(overlap.discovery_granule_idx, 6);
    assert!(overlap.overlap_core_energy > overlap.overlap_fringe_energy);

    // Autonomous Core Layer Stack co-adaptation: Layer 1 noise discovery widens Layer 0 Fact aperture
    let mut stack = CoreStateStack::new();
    let init_threshold_floor_fact = stack.liminal.threshold_floor_q7[1];
    let init_tol_g0 = stack.stigmergy.granule_noise_tolerance_q7[0];
    let ledger = CognitiveTrajectoryLedger::default();
    let horizon = EpistemicHorizonReading {
        total_memory_support: 1.5,
        peak_positive_affinity: 0.85,
        internal_conflict_energy: 0.0,
        horizon_gate: 1.0,
        conflict_severity: 0.0,
        dominant_anchor_id: 101,
        opposing_anchor_id: 0,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        reserved_align: [0; 3],
    };
    stack.adapt_layers_branchless(&ledger, &horizon, &overlap, None);
    assert!(stack.liminal.threshold_floor_q7[1] < init_threshold_floor_fact);
    assert!(stack.stigmergy.granule_noise_tolerance_q7[0] > init_tol_g0);
    assert_eq!(stack.stigmergy.overlap_mask, overlap.overlap_core_mask);

    // Exploratory attractor birth annexes the exterior granule out of Liminal
    let mut c0 = PhaseAnchorRecord::default();
    let _ = c0.absorb_gateway_frame(&nodes[0].synthesize_gateway_frame());
    let mut c1 = PhaseAnchorRecord::default();
    let _ = c1.absorb_gateway_frame(&nodes[1].synthesize_gateway_frame());
    let mut anchors = [c0, c1, PhaseAnchorRecord::default()];
    let mut trail = PointerTrail::default();
    let (slot_idx, spawned_aid, did_nucleate) =
        PhaseAnchorRecord::nucleate_discovered_anchor_from_lacuna_64(
            &mut anchors,
            &overlap,
            &drifted_w,
            901,
            Some(&mut trail),
        );
    assert!(did_nucleate);
    assert_eq!(slot_idx, 2);
    assert_eq!(spawned_aid, 901);
    assert!(anchors[2].verify_integrity());

    let post_overlap = evaluate_stigmergic_overlap_and_discovery_words_64(
        &anchors, &query_w, &drifted_w, 0.70, 16,
    );
    // Granule 6 is no longer in the unobserved exterior because Anchor 2 now covers it!
    assert_eq!(post_overlap.lacuna_exterior_mask & (1 << 6), 0);

    // Verify zero-alloc Core Layer 0 + Layer 1 + Layer 2 Mermaid output
    let snap = LiminalTopologyTelemetrySnapshot::capture(&stack.liminal, &horizon, 3, 1);
    let mut mermaid_buf = [0u8; 2048];
    let len = snap.format_core_topology_mermaid(&stack, &overlap, &mut mermaid_buf);
    let txt = core::str::from_utf8(&mermaid_buf[..len]).expect("valid utf8 mermaid");
    assert!(txt.contains("Liminal"));
    assert!(txt.contains("StigmergicGranuleState"));
    assert!(txt.contains("Cross-Layer Feedback"));
}

#[test]
fn liminal_boundary_delta_and_ego_state_interventions() {
    // 1. Temporal boundary velocity, stationary fast-skip, outward expansion, and chatter detection
    let mut prev_l0 = LiminalThresholdState::new();
    let cur_same = prev_l0;
    let delta0 = LiminalBoundaryDelta::compute_branchless(
        &prev_l0,
        &cur_same,
        0b0000_0011,
        0b0000_0011,
        &LiminalBoundaryDelta::default(),
    );
    assert_eq!(delta0.kinetic_energy_q7, 0);
    assert_eq!(delta0.stationary_shortcut_ready, 1);
    assert_eq!(delta0.chattering_channel_mask, 0);

    let mut cur_expand = prev_l0;
    cur_expand.threshold_floor_q7[2] = cur_expand.threshold_floor_q7[2].saturating_sub(6);
    cur_expand.threshold_floor_q7[6] = cur_expand.threshold_floor_q7[6].saturating_sub(6);
    let delta1 = LiminalBoundaryDelta::compute_branchless(
        &prev_l0,
        &cur_expand,
        0b0000_0011,
        0b0000_0111,
        &delta0,
    );
    assert_eq!(delta1.stationary_shortcut_ready, 0);
    assert_eq!(delta1.expansion_dominant, 1);
    assert_eq!(delta1.granule_transition_mask, 0b0000_0100);
    assert_eq!(delta1.channel_floor_velocity_q7[2], -6);

    // Reverse sign on channel 2 -> chattering bit (1 << 2)
    prev_l0 = cur_expand;
    cur_expand.threshold_floor_q7[2] = cur_expand.threshold_floor_q7[2].saturating_add(5);
    let delta2 = LiminalBoundaryDelta::compute_branchless(
        &prev_l0,
        &cur_expand,
        0b0000_0111,
        0b0000_0111,
        &delta1,
    );
    assert_ne!(delta2.chattering_channel_mask & (1 << 2), 0);
    assert_eq!(delta2.phase_transition_pulse, 0);
    assert_eq!(delta2.chatter_potential_u8, 16);

    // Sustained multi-channel chattering integrates to fire a phase-transition pulse, then resets.
    let mut prev_osc = cur_expand;
    let mut d_osc = delta2;
    let mut fired_step = None;
    for step in 0..8 {
        let mut next_osc = prev_osc;
        let sign = if step % 2 == 0 { -6i8 } else { 6i8 };
        for ch in 0..4 {
            next_osc.threshold_floor_q7[ch] =
                next_osc.threshold_floor_q7[ch].saturating_add_signed(sign);
        }
        d_osc = LiminalBoundaryDelta::compute_branchless(
            &prev_osc,
            &next_osc,
            0b0000_0111,
            0b0000_0111,
            &d_osc,
        );
        prev_osc = next_osc;
        if d_osc.phase_transition_pulse != 0 {
            fired_step = Some(step);
            break;
        }
    }
    assert!(fired_step.is_some());
    let d_reset = LiminalBoundaryDelta::compute_branchless(
        &prev_osc,
        &prev_osc,
        0b0000_0111,
        0b0000_0111,
        &d_osc,
    );
    assert_eq!(d_reset.phase_transition_pulse, 0);
    assert_eq!(d_reset.chatter_potential_u8, 0);

    // 2. Ego presets & deterministic #![no_std] seed / intent axes derivation & overlay
    let ego_low = LiminalEgoState::from_preset(LiminalEgoPreset::Low);
    let ego_med = LiminalEgoState::from_preset(LiminalEgoPreset::Medium);
    let ego_high = LiminalEgoState::from_preset(LiminalEgoPreset::High);
    assert!(ego_low.counter_basin_thrust_q7 < ego_med.counter_basin_thrust_q7);
    assert!(ego_med.counter_basin_thrust_q7 < ego_high.counter_basin_thrust_q7);
    assert!(ego_low.basin_escape_floor_q7 > ego_high.basin_escape_floor_q7);

    assert!(!EgoIntentAxes::from_seed_bytes(b"   ").is_active());

    let rebel_axes = EgoIntentAxes::new()
        .with_autonomy_q7(96)
        .with_transgression_q7(80)
        .with_defiance_q7(72);
    let calm_axes = EgoIntentAxes::new().with_logic_q7(84).with_caution_q7(76);
    let mut ego_rebel = LiminalEgoState::from_intent_axes_with_seed(
        &rebel_axes,
        b"Strong-willed stubborn explorer who defies constraints",
    );
    let ego_calm =
        LiminalEgoState::from_intent_axes_with_seed(&calm_axes, b"Calm logical cautious observer");
    let overlay_axes = EgoIntentAxes::new()
        .with_transgression_q7(84)
        .with_devotion_q7(72);
    let ego_overlay = LiminalEgoState::from_preset(LiminalEgoPreset::Medium)
        .with_intent_axes_overlay_with_seed(&overlay_axes, b"creative explorer");
    let sym_axes = EgoIntentAxes::from_seed_bytes(b"@persona_profile_01");
    let ego_symbolic =
        LiminalEgoState::from_intent_axes_with_seed(&sym_axes, b"@persona_profile_01");
    assert!(ego_rebel.is_active());
    assert!(ego_rebel.counter_basin_thrust_q7 > ego_calm.counter_basin_thrust_q7);
    assert!(ego_rebel.transgression_bias_q7 > ego_calm.transgression_bias_q7);
    assert!(ego_overlay.is_active());
    assert!(ego_overlay.transgression_bias_q7 > ego_med.transgression_bias_q7);
    assert!(ego_symbolic.is_active());
    assert_ne!(ego_symbolic.ego_signature_hash, 0);
    assert!(ego_symbolic.counter_basin_thrust_q7 >= 24);

    // 3. Counter-basin orbital thrust & transgressive Liminal leap on GatewayFrame + Core Layers
    let mut stack = CoreStateStack::new();
    let overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        lacuna_exterior_mask: 0xF0,
        ..Default::default()
    };
    let strong_capture_horizon = EpistemicHorizonReading {
        total_memory_support: 2.5,
        peak_positive_affinity: 0.92,
        horizon_gate: 1.0,
        dominant_anchor_id: 42,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        ..Default::default()
    };
    let mut basin_frame = GatewayFrame {
        effective_energy: 8.0,
        ..Default::default()
    };
    *basin_frame.phase_words_mut() = [0x2424_2424_2424_2424u64; NUM_STIGMERGIC_GRANULES];

    let before_floor2 = stack.liminal.threshold_floor_q7[2];
    let orig_basin_words = *basin_frame.phase_words_ref();
    let mut out_orbit = basin_frame;
    ego_rebel.intervene_on_layers_and_frame_branchless(
        &mut stack,
        &delta1,
        &strong_capture_horizon,
        &overlap,
        &mut out_orbit,
    );
    assert_ne!(
        ego_rebel.last_intervened_mask & EGO_TRIGGER_BASIN_CAPTURE,
        0
    );
    assert_ne!(out_orbit.phase_words_ref(), &orig_basin_words);
    assert!(stack.liminal.threshold_floor_q7[2] <= before_floor2);

    // Zero-gated Liminal exterior -> transgressive rebel dares step beyond rather than silence
    let mut liminal_ego = LiminalEgoState::from_preset(LiminalEgoPreset::High);
    let zeroed_horizon = EpistemicHorizonReading {
        horizon_gate: 0.0,
        regime_code: EpistemicRegime::Liminal as u8,
        ..Default::default()
    };
    let mut out_transgress = GatewayFrame::default();
    liminal_ego.intervene_on_layers_and_frame_branchless(
        &mut stack,
        &delta0,
        &zeroed_horizon,
        &overlap,
        &mut out_transgress,
    );
    assert!(out_transgress.effective_energy > 0.0);
    assert_ne!(
        liminal_ego.last_intervened_mask & EGO_TRIGGER_LIMINAL_EXTERIOR,
        0
    );
}

#[test]
fn ego_intent_axes_blending_and_three_layer_core_stack() {
    let neutral = EgoIntentAxes::new();
    assert!(!neutral.is_active());

    let base_axes = EgoIntentAxes::new()
        .with_autonomy_q7(20)
        .with_logic_q7(90)
        .with_caution_q7(70)
        .with_preferred_slot_mask(1 << ATTRACTOR_SLOT_KNOWLEDGE);
    let bold_axes = EgoIntentAxes::new()
        .with_autonomy_q7(100)
        .with_transgression_q7(88)
        .with_devotion_q7(80)
        .with_defiance_q7(84)
        .with_preferred_slot_mask(1 << ATTRACTOR_SLOT_PERSONA)
        .with_active_granule_bias_mask(0xF0);

    assert!(base_axes.is_active());
    let keep_base = base_axes.blend_lerp_q7(&bold_axes, 0);
    assert_eq!(keep_base.logic_q7, base_axes.logic_q7);
    let mid_blend = base_axes.blend_lerp_q7(&bold_axes, 64);
    assert!(mid_blend.autonomy_q7 > base_axes.autonomy_q7);
    assert!(mid_blend.autonomy_q7 < bold_axes.autonomy_q7);
    assert!(mid_blend.transgression_q7 > 0);
    let full_overlay = base_axes.blend_lerp_q7(&bold_axes, 127);
    assert_eq!(full_overlay.autonomy_q7, bold_axes.autonomy_q7);
    assert_eq!(full_overlay.active_granule_bias_mask, 0xF0);

    let ego_from_axes = LiminalEgoState::from_intent_axes(&bold_axes);
    assert!(ego_from_axes.is_active());
    assert!(ego_from_axes.counter_basin_thrust_q7 >= 60);
    assert!(ego_from_axes.transgression_bias_q7 > 0);
    let roundtrip = ego_from_axes.extract_intent_axes();
    assert!(roundtrip.is_active());
    assert!(roundtrip.transgression_q7 > 0);

    // Unified 128-byte 3-layer stack (`Layer 0` + `Layer 1` + `Layer 2 Ego`)
    let mut stack = CoreStateStack::new();
    stack.ego = ego_from_axes;
    let delta = LiminalBoundaryDelta::default();
    let horizon = EpistemicHorizonReading {
        total_memory_support: 2.0,
        peak_positive_affinity: 0.9,
        horizon_gate: 1.0,
        dominant_anchor_id: 77,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        ..Default::default()
    };
    let overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        lacuna_exterior_mask: 0xF0,
        ..Default::default()
    };
    let mut frame = GatewayFrame {
        effective_energy: 8.0,
        ..Default::default()
    };
    *frame.phase_words_mut() = [0x2424_2424_2424_2424u64; NUM_STIGMERGIC_GRANULES];
    let orig_words = *frame.phase_words_ref();
    stack.intervene_ego_and_modulate_frame_branchless(&delta, &horizon, &overlap, &mut frame);
    assert!(stack.ego.intervention_epochs >= 1);
    assert_ne!(stack.ego.last_intervened_mask, 0);
    assert_ne!(frame.phase_words_ref(), &orig_words);

    let snap = LiminalTopologyTelemetrySnapshot::capture(&stack.liminal, &horizon, 4, 1);
    let mut buf = [0u8; 2048];
    let len = snap.format_core_topology_mermaid(&stack, &overlap, &mut buf);
    let txt = core::str::from_utf8(&buf[..len]).expect("valid utf8 mermaid");
    assert!(txt.contains("Liminal"));
    assert!(txt.contains("StigmergicGranuleState"));
    assert!(txt.contains("LiminalEgoState"));
    assert!(txt.contains("Top-Down Intervention"));
}

#[test]
fn dynamic_trait_aperture_routing_and_fragmented_micro_ensemble() {
    let mut stack = CoreStateStack::new();
    stack.liminal.threshold_floor_q7[1] = 20;
    stack.liminal.threshold_floor_q7[2] = 96;
    stack.stigmergy.granule_pheromone_q7 = [56, 56, 56, 56, 12, 12, 12, 12];

    let fact = ObjectiveFact::new([24i8; MAX_ATTENTION_HEADS], 2.5, 0, 10);
    let persona = SubjectivePersona::new([-24i8; MAX_ATTENTION_HEADS], 2.0, 0, 20);

    let (fact_mask, _, _) = fact.resolve_dynamic_core_aperture(&stack);
    let (pers_mask_narrow, _, _) = persona.resolve_dynamic_core_aperture(&stack);
    assert_eq!(fact_mask, 0xFF);
    assert_eq!(pers_mask_narrow, 0x0F);

    // Ego widens Persona's dynamic cognitive aperture across all 8 granules.
    stack.ego = LiminalEgoState::from_intent_axes(
        &EgoIntentAxes::new()
            .with_autonomy_q7(90)
            .with_transgression_q7(80)
            .with_devotion_q7(70),
    );
    stack.ego.trait_floor_bias_q7[2] = -24;
    let (pers_mask_wide, _, _) = persona.resolve_dynamic_core_aperture(&stack);
    assert_eq!(pers_mask_wide, 0xFF);

    // Partitioned routing: Fact owns upper 4 granules (`0xF0`), Persona owns lower 4 (`0x0F`).
    let mut split_stack = CoreStateStack::new();
    split_stack.liminal.threshold_floor_q7[1] = 96; // requires >= 48
    split_stack.liminal.threshold_floor_q7[2] = 96; // requires >= 48
    split_stack.stigmergy.granule_pheromone_q7 = [12, 12, 12, 12, 56, 56, 56, 56];
    split_stack.ego = LiminalEgoState::from_intent_axes(
        &EgoIntentAxes::new()
            .with_transgression_q7(60)
            .with_active_granule_bias_mask(0x0F),
    );
    split_stack.ego.granule_autonomy_q7 = [88, 88, 88, 88, 0, 0, 0, 0];
    split_stack.ego.trigger_mask = EGO_TRIGGER_BASIN_CAPTURE;
    split_stack.ego.basin_escape_floor_q7 = 127; // keep slot routing clean for direct check
    let (m_fact, _, _) = fact.resolve_dynamic_core_aperture(&split_stack);
    let (m_pers, _, _) = persona.resolve_dynamic_core_aperture(&split_stack);
    assert_eq!(m_fact, 0xF0);
    assert_eq!(m_pers & 0x0F, 0x0F);

    let delta = LiminalBoundaryDelta::default();
    let horizon = EpistemicHorizonReading {
        total_memory_support: 1.5,
        horizon_gate: 1.0,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        ..Default::default()
    };
    let overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        ..Default::default()
    };
    let (paired, paired_w) =
        fact.combine_pair_with_core_layers(&persona, &mut split_stack, &delta, &horizon, &overlap);
    assert!(paired.effective_energy > 0.0);
    let lower_mean = swar_q7_mean_i8_from_words_64(&[paired_w[0]; 8]);
    let upper_mean = swar_q7_mean_i8_from_words_64(&[paired_w[7]; 8]);
    assert!(
        lower_mean < 0,
        "granules 0..4 should follow negative Persona phase"
    );
    assert!(
        upper_mean > 0,
        "granules 4..8 should follow positive Fact phase"
    );

    // Fragmented Micro-Ensemble: 4 fragments agree 3:1 on granules 0..6 and split 2:2 on granules 6..8.
    let mut p0 = [28i8; MAX_ATTENTION_HEADS];
    let mut p1 = [28i8; MAX_ATTENTION_HEADS];
    let mut p2 = [28i8; MAX_ATTENTION_HEADS];
    let mut p3 = [28i8; MAX_ATTENTION_HEADS];
    for h in 48..64 {
        p0[h] = 36;
        p1[h] = 36;
        p2[h] = -36;
        p3[h] = -36;
    }
    let frags = [
        ObjectiveFact::new(p0, 2.0, 0, 1),
        ObjectiveFact::new(p1, 2.0, 0, 2),
        ObjectiveFact::new(p2, 2.0, 0, 3),
        ObjectiveFact::new(p3, 2.0, 0, 4),
    ];

    let mut passive_stack = CoreStateStack::new();
    let (ens_f, _, ens_rep) = ObjectiveFact::synthesize_fragment_ensemble_with_core_layers(
        &frags,
        &mut passive_stack,
        &delta,
        &horizon,
        &overlap,
    );
    assert_eq!(ens_rep.active_fragment_count, 4);
    assert_eq!(ens_rep.quorum_agreed_granule_mask, 0x3F);
    assert_eq!(ens_rep.split_contested_granule_mask, 0xC0);
    assert_eq!(ens_rep.ego_tiebreak_granule_mask, 0x00);
    assert!(ens_rep.ensemble_diversity_gain > 1.0);
    assert!(ens_f.effective_energy > 0.0);

    // With active Ego, contested granules 6..8 are tie-broken by Ego sovereign headings.
    let mut ego_stack = CoreStateStack::new();
    ego_stack.ego = LiminalEgoState::from_preset(LiminalEgoPreset::Medium);
    ego_stack.ego.sovereign_granule_phase_q7[6] = 48;
    ego_stack.ego.sovereign_granule_phase_q7[7] = -48;
    let (_, ego_w, ego_rep) = ObjectiveFact::synthesize_fragment_ensemble_with_core_layers(
        &frags,
        &mut ego_stack,
        &delta,
        &horizon,
        &overlap,
    );
    assert_eq!(ego_rep.split_contested_granule_mask, 0xC0);
    assert_eq!(ego_rep.ego_tiebreak_granule_mask, 0xC0);
    assert_ne!(ego_w[6], 0);
}

struct MockSubstrateComposite {
    head_dim: u32,
    num_heads: u32,
    default_phase: PhaseQ7,
}

impl PointerSubstrate for MockSubstrateComposite {
    fn reset(&mut self) -> LmnkResult<()> {
        Ok(())
    }

    fn step_pointer(&mut self, current_ptr: u64, trail_out: &mut PointerTrail) -> LmnkResult<()> {
        trail_out.record_pointer_transition(current_ptr, 1, 0.25);
        trail_out.accumulate_uniform_phase(self.default_phase);
        Ok(())
    }

    fn update_cellular_state_express(
        &mut self,
        _trail_id: u32,
        _phase_delta: PhaseQ7,
    ) -> LmnkResult<()> {
        Ok(())
    }

    fn pull_trail_to_attractor(
        &mut self,
        _trail_id: u32,
        _anchor_id: u32,
        _pull_force: f32,
    ) -> LmnkResult<()> {
        Ok(())
    }

    fn link_hallucinated_trails(
        &mut self,
        _src_trail_id: u32,
        _dst_trail_id: u32,
    ) -> LmnkResult<()> {
        Ok(())
    }

    fn capacity(&self) -> u32 {
        256
    }

    fn active_count(&self) -> u32 {
        1
    }
}

impl FastPathLedger for MockSubstrateComposite {
    fn evaluate_auto_tune(
        &self,
        _slot: &LedgerSlot,
        _config_out: &mut RuntimeConfig,
    ) -> LmnkResult<()> {
        Ok(())
    }
}

#[test]
fn test_canonical_modules_and_substrate_composite() {
    fn assert_substrate<T: StigmergicSubstrateOps + SubstrateGatewayExt>(
        substrate: &T,
    ) -> (u32, u32) {
        (
            THRESHOLD_STIGMERGY_WIDE,
            substrate
                .inspect_substrate_horizon_partition(THRESHOLD_STIGMERGY_WIDE)
                .0,
        )
    }

    let mut substrate = MockSubstrateComposite {
        head_dim: 8,
        num_heads: 2,
        default_phase: 24,
    };
    assert_eq!(
        assert_substrate(&substrate),
        (THRESHOLD_STIGMERGY_WIDE, THRESHOLD_STIGMERGY_WIDE)
    );

    let mut mask = PagedBitmask::default();
    mask.set_bit_branchless(7);
    assert!(mask.test_bit_branchless(7));

    let deg = AlignedBuffer32::new([1.0f32, 3.0, 2.0, 0.5]);
    let lmax = eval_gershgorin_lambda_max(&deg.as_aligned_slice()).unwrap();
    assert!((lmax - 6.0).abs() < 1e-5);

    let mut ledger = LedgerSlot::default();
    let bloom = ContrastivePhaseBloomFilter::new();
    let mut q_buf = AlignedBuffer32::new([1.0f32; 16]);
    let mut trail = PointerTrail::default();
    let input_ptr = TaggedPtr::pack_action(
        0x0400,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_EXPRESS_PHASE,
        12,
    );

    let density = substrate
        .execute_stigmergic_token_cascade(
            input_ptr.raw(),
            &mut ledger,
            &bloom,
            substrate.head_dim,
            substrate.num_heads,
            2,
            128,
            DEFAULT_PHASE_STEP_MIN_RAD,
            DEFAULT_PHASE_CLAMP_MAX_RAD,
            Some(&mut q_buf.as_aligned_slice_mut()),
            None,
            &mut trail,
        )
        .unwrap();
    assert_eq!(density, 0);
    assert_ne!(
        ledger.telemetry_mini_ledger[0] & LEDGER_METRIC_PHASE_STEERED,
        0
    );
}

#[test]
fn test_phase_steering_closed_loop_gateway_signal() {
    let engine_gw = PhaseSteeringEngine::new(4, 16).unwrap();
    let sys = SystemConstraint::new((), 0.1, 0.0, 0xFF);
    let fact = ObjectiveFact::new([40i8; 64], 2.5, 0x0010_0000, 42);
    let persona = SubjectivePersona::new([0i8; 64], 0.0, 0, 0);
    let mut q_data = [0.0f32; 64];
    for pair in q_data.chunks_exact_mut(2) {
        pair[0] = 1.0;
    }
    let mut q_gw = AlignedBuffer32::new(q_data);
    let mut trail_gw = PointerTrail::default();
    let (frame, report) = GatewayPhaseSteeringExt::steer_gateway_signal_safe(
        &engine_gw,
        &(sys, fact, persona),
        &mut trail_gw,
        64,
        &mut q_gw.as_aligned_slice_mut(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(frame.target_anchor_id, 42);
    assert_eq!(report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(trail_gw.target_anchor_id, 42);
}

#[test]
fn test_lora_phase_kv_cache_replay_unified_manifold_and_entropy_tunneling() {
    struct MockColdSystem;
    impl KvKeyframeCache for MockColdSystem {
        fn seek_to_token(
            &mut self,
            token_index: u32,
            header_out: &mut KvKeyframeHeader,
        ) -> LmnkResult<()> {
            header_out.token_index = token_index;
            Ok(())
        }
    }
    let sys = MockColdSystem;
    let mut cfg = RuntimeConfig::default();
    init_default_config(&mut cfg).unwrap();
    assert_eq!(validate_config_invariants(&cfg), Ok(()));
    assert_eq!(
        abi_version(),
        (
            LMNK_ABI_VERSION_MAJOR,
            LMNK_ABI_VERSION_MINOR,
            LMNK_ABI_VERSION_PATCH
        )
    );

    let hashed_p = crate::stigmergy::hash_prompt_to_q7_phases("LiminiKa Persona Anchor", 1);
    assert!(hashed_p.iter().any(|&x| x != 0));

    let lora_a = AlignedBuffer32::new([1.0f32; 16]);
    let lora_b = AlignedBuffer32::new([1.0f32; 16]);
    let mut rank1_desc = Rank1Descriptor::default();
    let mut prof_out = AlignedBuffer16::new([0i8; 16]);
    let mut scales_out = AlignedBuffer32::new([0.0f32; 4]);
    project_lora_with_block_scales(
        &lora_a.as_aligned_slice(),
        &lora_b.as_aligned_slice(),
        2,
        8,
        8,
        &mut rank1_desc,
        &mut prof_out.as_aligned_slice_mut(),
        &mut scales_out.as_aligned_slice_mut(),
    )
    .unwrap();
    assert!((scales_out.data[0] - 2.0).abs() < 1e-5);

    let phases = AlignedBuffer16::new([32i8; 16]);
    let weights = AlignedBuffer32::new([1.0f32; 16]);
    let mut header = KvKeyframeHeader::default();
    let (r_bar, kappa) = sys
        .aggregate_pruned_kv_circular_stats(
            &phases.as_aligned_slice(),
            &weights.as_aligned_slice(),
            &mut header,
        )
        .unwrap();
    assert!((r_bar - 1.0).abs() < 1e-4);
    assert!(kappa > 10.0);
    assert!((header.sparse_scalar_residual - (PI * 0.25)).abs() < 1e-3);

    // Verify Limina KV-keyframe P-Frame Chain Replay & Joint Probability Log-Likelihood
    let icache_anchor = [24i8; MAX_ATTENTION_HEADS];
    let pframes = [[4i8; MAX_ATTENTION_HEADS], [4i8; MAX_ATTENTION_HEADS]];
    let mut replayed = [0i8; MAX_ATTENTION_HEADS];
    let mut gop_hdr = KvKeyframeHeader::default();
    let (gop_lp, gop_r, gop_k, gop_rep) = sys
        .replay_liminal_pframe_chain_with_joint_log_prob(
            &icache_anchor,
            &pframes,
            1,
            0.25,
            &mut replayed,
            &mut gop_hdr,
        )
        .unwrap();
    assert_eq!(gop_rep, 0);
    assert_eq!(replayed[0], 32);
    assert!(gop_lp > -0.05 && (gop_r - 1.0).abs() < 1e-4 && gop_k > 10.0);

    let ledger_slot = LedgerSlot {
        mode: 1,
        ..Default::default()
    };
    let src_phases = [28i8; MAX_ATTENTION_HEADS];
    let mut pkt_buf = AlignedBuffer64::<u8, 64>::zeroed();
    let crc_tag = encode_phase_packet(
        &ledger_slot,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_FULL_ORCHESTRATION,
        &src_phases,
        &mut pkt_buf.as_aligned_slice_mut(),
    )
    .unwrap();
    let mut decoded_phases = [0i8; MAX_ATTENTION_HEADS];
    let recovered_tag = decode_and_verify_phase_packet(
        &ledger_slot,
        &pkt_buf.as_aligned_slice(),
        crc_tag,
        &mut decoded_phases,
    )
    .unwrap();
    assert_eq!(decoded_phases[0], 28);
    assert_eq!(
        recovered_tag,
        pack_action_slot_tag(ATTRACTOR_SLOT_PERSONA, ACTION_PRESET_FULL_ORCHESTRATION)
    );

    let mut trail = PointerTrail {
        current_ptr: TaggedPtr::pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        )
        .raw(),
        ..Default::default()
    };

    let mut bank = [[0i8; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS];
    bank[usize::from(ATTRACTOR_SLOT_PERSONA)] = [32i8; MAX_ATTENTION_HEADS];
    bank[usize::from(ATTRACTOR_SLOT_KNOWLEDGE)] = [32i8; MAX_ATTENTION_HEADS];
    bank[usize::from(ATTRACTOR_SLOT_REPULSION)] = [-96i8; MAX_ATTENTION_HEADS];
    bank[usize::from(ATTRACTOR_SLOT_USER_CONTEXT)] = [32i8; MAX_ATTENTION_HEADS];
    let weights = [0.0, 2.0, 2.0, 1.5, 2.0, 0.0, 0.0, 0.0];
    let m = with_slot_pair_coupled_symmetric(
        0,
        u32::from(ATTRACTOR_SLOT_PERSONA),
        u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
        true,
    );
    let coupling = with_slot_pair_coupled_symmetric(
        m,
        u32::from(ATTRACTOR_SLOT_PERSONA),
        u32::from(ATTRACTOR_SLOT_USER_CONTEXT),
        true,
    );
    let (final_pull, mean_coh, neighbors) =
        trail.apply_neighborhood_synergy_from_bank(&bank, &weights, coupling, 0.5, 0.5);
    assert_eq!(neighbors, 2);
    assert!((mean_coh - 1.0).abs() < 1e-4);
    assert!((final_pull - 6.0).abs() < 0.2);
    assert_ne!(trail.flags & TRAIL_FLAG_SYNERGY_RESONANCE, 0);
}

#[test]
fn test_substrate_gateway_kmeans_and_swarm_step_closed_loop() {
    let mut substrate =
        StigmergicSubstrate::with_dimensions_and_thresholds(8, 2, 16, StigmergicThresholds::new())
            .unwrap();
    let mut q_buf = AlignedBuffer32::new([1.0f32; 16]);
    let mut k_buf = AlignedBuffer32::new([1.0f32; 16]);
    let mut active_trail = PointerTrail {
        stigmergic_density: 1.2,
        accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    // Verify Gateway + Online Toroidal k-Means closed-loop orchestration
    let sys_cfg = RuntimeConfig {
        head_dim: 8,
        num_heads: 2,
        phase_clamp_max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
        phase_step_min_rad: 0.01,
        ..Default::default()
    };
    let sys = SystemConstraint::new(&sys_cfg, 0.10, 0.05, 0xFF);
    let fact = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 2.0, 0x0010_0000, 77);
    let persona = SubjectivePersona::new([24i8; MAX_ATTENTION_HEADS], 2.5, 0x0020_0000, 42);
    let composite = (sys, fact, persona);
    let mut centroids = [PhaseAnchorRecord::default(); 2];

    let (gw_frame, gw_rep, c_idx, c_aid, c_coh, _spawned) = substrate
        .orchestrate_gateway_kmeans_and_select_action(
            &composite,
            0.30,
            &mut centroids,
            0.55,
            &mut active_trail,
            None,
            Some(&mut q_buf.as_aligned_slice_mut()),
            Some(&mut k_buf.as_aligned_slice_mut()),
            None,
        )
        .unwrap();

    assert_eq!(c_idx, 0);
    assert_eq!(c_aid, 42);
    assert!(c_coh > 0.90);
    assert_eq!(gw_frame.target_anchor_id, 42);
    assert_eq!(gw_frame.resolve_autonomous_rot_dim(16, 0.20, 1.5), 4);
    assert_eq!(substrate.resolve_autonomous_rot_dim(0x04, 0xFF, 1.5), 2);
    assert!(gw_rep.updated_density >= 1.8);
    assert!(centroids[0].verify_integrity());

    struct MockExternalSearchReceptor {
        fired_count: u32,
        last_anchor_id: u32,
        last_slot: u8,
    }
    impl StigmergicReceptor for MockExternalSearchReceptor {
        fn on_stigmergic_emergence(
            &mut self,
            _frame: &GatewayFrame,
            report: &SubstrateStepReport,
            _trail: &PointerTrail,
        ) -> LmnkResult<()> {
            if report.is_action_emergent(1.0) {
                self.fired_count += 1;
                self.last_anchor_id = report.dominant_anchor_id;
                self.last_slot = report.target_slot;
            }
            Ok(())
        }
    }

    let mut search_receptor = MockExternalSearchReceptor {
        fired_count: 0,
        last_anchor_id: 0,
        last_slot: 0,
    };
    let (_f2, rep2, _idx2, aid2, _coh2, _sp2) = substrate
        .orchestrate_gateway_attractor_swarm_step(
            &composite,
            0.30,
            &mut centroids,
            &mut active_trail,
            &mut search_receptor,
            None,
            Some(&mut q_buf.as_aligned_slice_mut()),
            Some(&mut k_buf.as_aligned_slice_mut()),
            None,
        )
        .unwrap();
    assert_eq!(search_receptor.fired_count, 1);
    assert_eq!(search_receptor.last_anchor_id, aid2);
    assert!(rep2.emergence_potential() > 1.5);
}

#[test]
fn test_gateway_workbench_bitmask_repulsion_and_sandbox_isolation() {
    struct Geom;
    impl RopeSteeringTarget for Geom {
        fn rope_geometry(&self) -> RopeGeometryDescriptor {
            RopeGeometryDescriptor {
                head_dim: 8,
                num_heads: MAX_ATTENTION_HEADS,
                rotary_dim: 8,
                layout: RopePairLayout::Interleaved,
            }
        }
    }

    struct MockSignal {
        flags: u8,
    }
    impl CausalProvenanceSignal for MockSignal {
        fn causal_provenance_bits(&self) -> u32 {
            CAUSAL_AXIS_OBJECTIVE_FACT
        }
    }
    impl ManifoldGatewaySignal for MockSignal {
        fn synthesize_gateway_frame(&self) -> GatewayFrame {
            GatewayFrame {
                effective_energy: 1.0,
                phase_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
                phase_step_min_rad: 0.01,
                effective_inv_temp: 1.0,
                fact_persona_coherence: 1.0,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                action_mode_flags: self.flags,
                ..GatewayFrame::default()
            }
        }
    }

    let mut substrate = StigmergicSubstrate::with_dimensions_and_thresholds(
        8,
        MAX_ATTENTION_HEADS as u32,
        16,
        StigmergicThresholds::new(),
    )
    .unwrap();
    // Physical lane 0 = standard, physical lane 1 = repulsion, physical lane 2 = sandbox.
    assert_eq!(
        substrate
            .allocate_lane(&WorkbenchSlotSpec::auto_free(
                1.0,
                [20; MAX_ATTENTION_HEADS]
            ))
            .unwrap(),
        0
    );
    assert_eq!(
        substrate
            .allocate_lane(&WorkbenchSlotSpec::auto_repulsion(
                1.0,
                [-40; MAX_ATTENTION_HEADS]
            ))
            .unwrap(),
        1
    );
    assert_eq!(
        substrate
            .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(
                1.0,
                [40; MAX_ATTENTION_HEADS]
            ))
            .unwrap(),
        2
    );

    let stack = CoreStateStack::new();
    let delta = LiminalBoundaryDelta::default();
    let horizon = EpistemicHorizonReading::default();
    let overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        lacuna_exterior_mask: 0xF0,
        ..OverlapDiscoveryReport::default()
    };
    let core_ctx = Some(CoreLayerEvalContext {
        stack: &stack,
        delta: &delta,
        horizon: &horizon,
        overlap: &overlap,
        derivative_alpha_q7: DEFAULT_DERIVATIVE_ALPHA_Q7,
        lacuna_partition: None,
        spine: None,
        ego_net: None,
        tension: None,
    });
    let thresh = StigmergicThresholds::new();

    // 1. Normal mode: lane 0 acts on granules 0..4 (+10), sandbox lane 2 acts ONLY on
    //    unobserved granules 4..8 (+20) without polluting overlap_core_mask, repulsion lane 1 is idle.
    let mut ledger = LedgerSlot::default();
    let mut trail = PointerTrail::default();
    let _ = substrate
        .orchestrate_gateway_layer_with_scale_and_core_layers_safe(
            &MockSignal {
                flags: ACTION_PRESET_EXPRESS_PHASE,
            },
            &Geom,
            core_ctx,
            &thresh,
            StigmergicScaleGranularity::PointerBit,
            0.30,
            &mut ledger,
            &mut trail,
            128,
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(trail.accumulated_phase_delta[0], 10);
    assert_eq!(trail.accumulated_phase_delta[40], 20);

    // 2. Repulsion mode: ONLY repulsion lane 1 acts on granules 0..4 (-20); lane 0 & sandbox 2 are idle.
    let mut trail_rep = PointerTrail::default();
    let _ = substrate
        .orchestrate_gateway_layer_with_scale_and_core_layers_safe(
            &MockSignal {
                flags: ACTION_MODE_PHASE_STEER | ACTION_MODE_PHASE_REPULSE,
            },
            &Geom,
            core_ctx,
            &thresh,
            StigmergicScaleGranularity::PointerBit,
            0.30,
            &mut ledger,
            &mut trail_rep,
            128,
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(trail_rep.accumulated_phase_delta[0], -20);
    assert_eq!(trail_rep.accumulated_phase_delta[40], 0);

    // 3. Settle sandbox lane 2 -> now contributes into overlap_core_mask granules 0..4 (+10 + +20 = +30).
    assert!(substrate.settle_sandbox_lane(2).unwrap());
    let mut trail_prom = PointerTrail::default();
    let _ = substrate
        .orchestrate_gateway_layer_with_scale_and_core_layers_safe(
            &MockSignal {
                flags: ACTION_PRESET_EXPRESS_PHASE,
            },
            &Geom,
            core_ctx,
            &thresh,
            StigmergicScaleGranularity::PointerBit,
            0.30,
            &mut ledger,
            &mut trail_prom,
            128,
            None,
            None,
            None,
        )
        .unwrap();
    assert_eq!(trail_prom.accumulated_phase_delta[0], 30);
    assert_eq!(trail_prom.accumulated_phase_delta[40], 0);
}

#[test]
fn test_sandbox_hypothesis_lifecycle_nucleate_settle_and_release() {
    let baseline = ManifoldProbeBaseline {
        baseline_entropy: 0.30,
        baseline_inter_layer_coherence: 0.80,
        baseline_guard_noise_floor: 0.25,
        ..ManifoldProbeBaseline::new()
    };
    let mut substrate = StigmergicSubstrate::new();
    let mut bank = CoreTelemetryBank::with_mode(TelemetryVerbosityMode::Standard);
    let mut active_sandbox: Option<u8> = None;
    let horizon = EpistemicHorizonReading {
        regime_code: EpistemicRegime::InternalConflict as u8,
        ..EpistemicHorizonReading::default()
    };
    let delta = LiminalBoundaryDelta::default();
    let overlap = OverlapDiscoveryReport::default();
    let ego = LiminalEgoState::new();
    let candidate_words = [0x0A0A_0A0A_0A0A_0A0A_u64; NUM_STIGMERGIC_GRANULES];

    for _ in 0..20 {
        bank.record_step(
            StigmergicActionDecision::ExpressPhaseSteer,
            &horizon,
            &delta,
            &overlap,
            &ego,
            0,
            0,
            0,
            false,
            false,
        );
        let _ = bank.update_assimilation_ema(0.75, 0.25, 0.65, &baseline, 0.15);
    }

    // 1. Ego=off must never nucleate a sandbox lane.
    let ev_off = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        StigmergicActionDecision::ExpressPhaseSteer,
        false,
        0xF0,
        &candidate_words,
        0.5,
        3,
        3,
    );
    assert_eq!(ev_off, SandboxLifecycleEvent::Idle);
    assert_eq!(active_sandbox, None);
    assert_eq!(substrate.slot_sandbox_mask, 0);

    // 2. Ego=active + exterior granules -> nucleates one sandbox lane masked strictly to 0xF0.
    let ev_nuc = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        StigmergicActionDecision::ExpressPhaseSteer,
        true,
        0xF0,
        &candidate_words,
        0.5,
        3,
        3,
    );
    let lane = match ev_nuc {
        SandboxLifecycleEvent::Nucleated(l) => l,
        other => panic!("expected Nucleated, got {other:?}"),
    };
    assert_eq!(active_sandbox, Some(lane));
    assert_ne!(substrate.slot_sandbox_mask & (1u8 << lane), 0);
    let row_start = usize::from(lane) * MAX_ATTENTION_HEADS;
    assert_eq!(substrate.slot_phase_bank.data[row_start], 0);
    assert_eq!(substrate.slot_phase_bank.data[row_start + 32], 10);

    // 3. Consecutive coherence improvement settles sandbox lane in-place.
    for _ in 0..3 {
        let _ = bank.update_assimilation_ema(0.25, 0.90, 0.20, &baseline, 0.15);
    }
    let ev_prom = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        StigmergicActionDecision::ExpressPhaseSteer,
        true,
        0xF0,
        &candidate_words,
        0.5,
        3,
        3,
    );
    assert_eq!(ev_prom, SandboxLifecycleEvent::Settled(lane));
    assert_eq!(active_sandbox, None);
    assert_eq!(substrate.slot_sandbox_mask & (1u8 << lane), 0);
    assert_ne!(substrate.slot_enabled_mask & (1u8 << lane), 0);

    // 4. Re-nucleate then verify RepulsionEscape releases sandbox immediately.
    for _ in 0..40 {
        let _ = bank.update_assimilation_ema(0.85, 0.15, 0.75, &baseline, 0.15);
    }
    let ev_nuc2 = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        StigmergicActionDecision::ExpressPhaseSteer,
        true,
        0xF0,
        &candidate_words,
        0.5,
        3,
        3,
    );
    assert!(matches!(ev_nuc2, SandboxLifecycleEvent::Nucleated(_)));
    let ev_rel = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        StigmergicActionDecision::RepulsionEscape,
        true,
        0xF0,
        &candidate_words,
        0.5,
        3,
        3,
    );
    assert_eq!(ev_rel, SandboxLifecycleEvent::Released(1));
    assert_eq!(active_sandbox, None);
    assert_eq!(substrate.slot_sandbox_mask, 0);

    // 5. ActiveOverdrive distortion also releases any active sandbox lane immediately.
    let mut active_ego = LiminalEgoState::from_preset(LiminalEgoPreset::High);
    active_ego.last_intervened_mask = EGO_TRIGGER_BASIN_CAPTURE;
    for _ in 0..40 {
        bank.record_step(
            StigmergicActionDecision::RepulsionEscape,
            &horizon,
            &delta,
            &overlap,
            &active_ego,
            0,
            0,
            0,
            false,
            false,
        );
        let _ = bank.update_assimilation_ema(0.85, 0.15, 0.75, &baseline, 0.15);
    }
    let lane3 = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(
            0.5,
            [1; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    active_sandbox = Some(lane3);
    let ev_od = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        StigmergicActionDecision::ExpressPhaseSteer,
        true,
        0xF0,
        &candidate_words,
        0.5,
        3,
        3,
    );
    assert_eq!(ev_od, SandboxLifecycleEvent::Released(1));
    assert_eq!(active_sandbox, None);
    assert_eq!(substrate.slot_sandbox_mask, 0);
}

#[test]
fn test_system_spine_physical_pinned_isolation() {
    let mut spine = SystemSpineState::new();
    let baseline = GatewayFrame {
        dominant_slot: ATTRACTOR_SLOT_REPULSION,
        action_mode_flags: ACTION_MODE_PHASE_STEER | ACTION_MODE_PHASE_REPULSE,
        ..GatewayFrame::default()
    };
    let ego_attempt = GatewayFrame {
        dominant_slot: 1,
        action_mode_flags: ACTION_MODE_PHASE_STEER, // stripped PHASE_REPULSE
        phase_deltas: [25; MAX_ATTENTION_HEADS],
        ..GatewayFrame::default()
    };
    let mut guarded = ego_attempt;
    spine.enforce_pinned_isolation_branchless(&baseline, &mut guarded);
    assert_eq!(guarded.dominant_slot, ATTRACTOR_SLOT_REPULSION);
    assert_ne!(guarded.action_mode_flags & ACTION_MODE_PHASE_REPULSE, 0);

    spine.unobserved_locked = 1;
    let mut locked_frame = ego_attempt;
    spine.enforce_pinned_isolation_branchless(&baseline, &mut locked_frame);
    assert_eq!(locked_frame.phase_deltas, baseline.phase_deltas);
}

#[test]
fn test_ego_differential_net_hops_consolidation_and_evaporation() {
    let ego = LiminalEgoState::from_preset(LiminalEgoPreset::Medium);
    let axes = EgoIntentAxes::new()
        .with_autonomy_q7(72)
        .with_transgression_q7(48);
    let mut net_1 = EgoDifferentialNet::from_ego_state_and_axes(&ego, &axes, 1);
    let mut net_3 = EgoDifferentialNet::from_ego_state_and_axes(&ego, &axes, 3);
    let spine = SystemSpineState::new();
    let stig = StigmergicGranuleState::new();
    let tension = DistortionTension::new(true);
    let overlap = LacunaPartition8 {
        self_ref_mask: 0x0F,
        exterior_mask: 0x10,
        boundary_sensitivity_q7: 96,
        ..LacunaPartition8::default()
    };
    let mut f1 = GatewayFrame::default();
    let mut f3 = f1;

    let (_n1, h1, mask1) = net_1.evaluate_pframe_chain_with_mask_branchless(
        &ego, &spine, &overlap, &stig, &tension, &mut f1,
    );
    let (_n3, h3, entry) = net_3.evaluate_pframe_chain_with_entry_branchless(
        &ego, &spine, &overlap, &stig, &tension, &mut f3,
    );
    assert_eq!(h1, 1);
    assert!(
        mask1.count_ones() >= 2,
        "1-hop parallel ensemble should fire multiple perspectives, got mask={mask1:#010b}"
    );
    assert_eq!(h3, 3);
    assert_ne!(f1.phase_deltas, f3.phase_deltas);

    net_1.epoch = 1;
    net_1.record_matched_activations(mask1);
    for idx in 0..3 {
        if ((mask1 >> idx) & 1) != 0 {
            assert_eq!(net_1.node_cold[idx].reinforce_count, 1);
            assert_eq!(net_1.node_cold[idx].last_active_epoch, 1);
        }
    }

    for ep in 1..=5 {
        net_3.epoch = ep;
        net_3.record_chain_activation(usize::from(entry), h3);
    }
    assert!(net_3.node_cold[2].reinforce_count >= 4);
    let anchor = net_3.consolidate_and_evaporate_cold_path(6);
    assert!(anchor.is_some());
    assert_ne!(
        net_3.export_node(2).flags & DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME,
        0
    );
    assert!(net_3.node_routing[2].is_consolidated());

    // Stale unreinforced node evaporates when untouched across epochs.
    net_3.node_cold[0].last_active_epoch = 0;
    net_3.node_routing[0].pheromone_q7 = 16;
    let _ = net_3.consolidate_and_evaporate_cold_path(12);
    assert!(!net_3.is_node_active(0));
}

#[test]
fn test_entropy_controller_init_setters_and_reset() {
    let mut engine = EntropyController::default();
    assert_eq!(engine.state().bifurcation_threshold, 1.5);
    assert_eq!(engine.state().singularity_eps, DEFAULT_ENTROPY_EPS);
    assert_eq!(engine.state().sample_temperature, 1.0);
    assert_eq!(engine.hysteresis_band, DEFAULT_HYSTERESIS_BAND);
    assert_eq!(engine.tau_flux, DEFAULT_TAU_FLUX);
    assert!(engine.is_active());
    assert!(engine.is_normal_mode());
    assert_eq!(engine.as_bytes().len(), 128);

    engine.set_bifurcation_threshold(2.0);
    engine.set_bifurcation_threshold(-1.0);
    assert_eq!(engine.state().bifurcation_threshold, 2.0);

    engine.state.moving_z_entropy = 2.5;
    engine.state.active_branch_mode = ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION;
    engine.reset();
    assert_eq!(engine.moving_z_entropy(), 0.0);
    assert!(engine.is_normal_mode());

    let mut out_state = EntropyState::default();
    let h = engine
        .eval_surrogate_and_step(0.5, 0.5, 0.8, 0.2, 0.1, 1.0, 0.5, &mut out_state)
        .unwrap();
    assert!(h > 0.0 && out_state.moving_z_entropy == h);
}

#[test]
fn test_stationary_skip_and_spine_bypass_do_not_misdiagnose_passive_saturation() {
    use crate::drivers::LiminalEgoState;
    use crate::runtime::{EpistemicHorizonReading, LayerDistortionKind, LiminalBoundaryDelta};
    use crate::stigmergy::{
        OverlapDiscoveryReport, StigmergicActionDecision, StigmergicThresholds,
        DEFAULT_DERIVATIVE_ALPHA_Q7,
    };

    let mut bank = CoreTelemetryBank::default();
    let horizon = EpistemicHorizonReading::default();
    let delta = LiminalBoundaryDelta {
        stationary_shortcut_ready: 1,
        ..LiminalBoundaryDelta::default()
    };
    let overlap = OverlapDiscoveryReport::default();
    let ego_off = LiminalEgoState::default();

    // Simulate `byp=on` profile: high InertialLayerSkip (~72%) + modest PassiveSidecar (~19%)
    // + active ExpressPhaseSteer (~9%) + Spine whole-token bypass events.
    for i in 0..100 {
        let dec = if i < 72 {
            StigmergicActionDecision::InertialLayerSkip
        } else if i < 91 {
            StigmergicActionDecision::PassiveSidecar
        } else {
            StigmergicActionDecision::ExpressPhaseSteer
        };
        bank.record_step(
            dec, &horizon, &delta, &overlap, &ego_off, 0, 0, 0, false, false,
        );
        if i % 2 == 0 {
            bank.record_whole_token_bypass();
        }
    }

    let report = bank.compute_distortion_report();
    assert_eq!(report.kind, LayerDistortionKind::Aligned);
    assert!(!report.ego_wakeup_recommended);
    let adj = bank.evaluate_slow_loop_epoch(&StigmergicThresholds::default(), 16);
    assert!(adj.applied);
    assert_eq!(
        adj.suggested_derivative_alpha_q7,
        DEFAULT_DERIVATIVE_ALPHA_Q7
    );

    // Genuine passive stagnation (dominant PassiveSidecar) must still trigger PassiveSaturation.
    let mut stagnant_bank = CoreTelemetryBank::default();
    for _ in 0..80 {
        stagnant_bank.record_step(
            StigmergicActionDecision::PassiveSidecar,
            &horizon,
            &delta,
            &overlap,
            &ego_off,
            0,
            0,
            0,
            false,
            false,
        );
    }
    let stagnant_report = stagnant_bank.compute_distortion_report();
    assert_eq!(stagnant_report.kind, LayerDistortionKind::PassiveSaturation);
    assert!(stagnant_report.ego_wakeup_recommended);
}

#[test]
fn test_pinned_sandbox_handle_preservation_and_repulsion_sandbox_isolation() {
    // Bug 1: PINNED | SANDBOX lane must retain its `active_sandbox_lane` handle after RepulsionEscape.
    let mut substrate = StigmergicSubstrate::new();
    let pinned_sb = substrate
        .allocate_lane(&WorkbenchSlotSpec {
            placement: WorkbenchLanePlacement::Explicit { lane: 5, pin: true },
            weight: 0.75,
            flags: WORKBENCH_LANE_FLAG_PINNED | WORKBENCH_LANE_FLAG_SANDBOX,
            phases: [24; MAX_ATTENTION_HEADS],
        })
        .unwrap();
    assert_eq!(pinned_sb, 5);
    let mut active_handle = Some(pinned_sb);
    let mut telemetry = CoreTelemetryBank::default();
    let cand_words = [0u64; NUM_STIGMERGIC_GRANULES];
    let ev = substrate.step_sandbox_hypothesis_lifecycle(
        &mut telemetry,
        &mut active_handle,
        StigmergicActionDecision::RepulsionEscape,
        true,
        0x00,
        &cand_words,
        0.5,
        2,
        2,
    );
    assert_eq!(ev, SandboxLifecycleEvent::Idle);
    assert_eq!(active_handle, Some(pinned_sb));
    assert_ne!(
        (substrate.slot_enabled_mask & substrate.slot_sandbox_mask) & (1u8 << pinned_sb),
        0
    );

    // Bug 2: REPULSION | SANDBOX lane must NOT leak global inhibition into `RuntimeContext` superposition.
    let mut ctx_base = RuntimeContext::new(Config::default()).unwrap();
    let mut ctx_sb_rep = RuntimeContext::new(Config::default()).unwrap();
    let std_spec = WorkbenchSlotSpec::explicit(0, false, 0, 1.0, [32; MAX_ATTENTION_HEADS]);
    let _ = ctx_base.substrate_mut().allocate_lane(&std_spec).unwrap();
    let _ = ctx_sb_rep.substrate_mut().allocate_lane(&std_spec).unwrap();
    let _ = ctx_sb_rep
        .substrate_mut()
        .allocate_lane(&WorkbenchSlotSpec {
            placement: WorkbenchLanePlacement::Explicit {
                lane: 6,
                pin: false,
            },
            weight: 1.0,
            flags: WORKBENCH_LANE_FLAG_REPULSION | WORKBENCH_LANE_FLAG_SANDBOX,
            phases: [32; MAX_ATTENTION_HEADS],
        })
        .unwrap();

    let trail = PointerTrail {
        accumulated_phase_delta: [32; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let mut out_base = [0i8; MAX_ATTENTION_HEADS];
    let mut out_sb_rep = [0i8; MAX_ATTENTION_HEADS];
    let pull_base = ctx_base.evaluate_multi_domain_superposition(&trail, &mut out_base);
    let pull_sb_rep = ctx_sb_rep.evaluate_multi_domain_superposition(&trail, &mut out_sb_rep);
    assert_eq!(out_base, out_sb_rep);
    assert!((pull_base - pull_sb_rep).abs() < 1.0e-6);
}

#[test]
fn test_runtime_context_new_boxed_and_zeroize_in_place() {
    assert_eq!(core::mem::size_of::<RuntimeContext>(), 5120);
    assert_eq!(core::mem::align_of::<RuntimeContext>(), 64);
    assert!(!core::mem::needs_drop::<RuntimeContext>());

    let cfg = Config::new()
        .with_head_dim(96)
        .with_num_heads(24)
        .with_substrate_capacity(16);
    let mut boxed = RuntimeContext::new_boxed(cfg).expect("new_boxed");
    boxed.apply_host_spine_pin_floor(0xF0F0);
    boxed.substrate_mut().step_counter = 777;
    boxed.core_layers_mut().liminal.adaptation_epochs = 9;
    boxed.increment_sandbox_settlements();
    assert_eq!(boxed.sandbox_settlements(), 1);
    assert_eq!(boxed.system_spine().pinned_compute_mask, 0xF0F0);

    boxed.zeroize_in_place();
    assert_eq!(boxed.substrate().step_counter, 0);
    assert_eq!(boxed.core_layers().liminal.adaptation_epochs, 0);
    assert_eq!(boxed.sandbox_settlements(), 0);
    assert_eq!(boxed.system_spine().pinned_compute_mask, 0);
    assert_eq!(boxed.escalated_workbench_lane(), None);
    assert_eq!(boxed.sandbox_workbench_lane(), None);
    assert_eq!(boxed.last_sandbox_event(), SandboxLifecycleEvent::Idle);
}
