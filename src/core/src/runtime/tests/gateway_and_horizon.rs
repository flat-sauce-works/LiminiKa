// name: src/core/src/runtime/tests/gateway_and_horizon.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gateway orchestration, KV cache, and latent horizon tests.

use super::*;

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
    let (f_scatter, w_scatter) = fact_scatter.synthesize_gateway_frame_with_words();
    let report_scatter = PhaseAnchorRecord::step_dissipative_overflow_scatter_or_spawn_64(
        &mut pool,
        &f_scatter,
        &w_scatter,
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

    // Case B: Saturated basin cannot absorb an orthogonal sample into any existing node -> autonomous daughter birth.
    let fact_orphan = ObjectiveFact::new([-96i8; MAX_ATTENTION_HEADS], 1.8, 0x0040_0000, 91);
    let (f_orphan, w_orphan) = fact_orphan.synthesize_gateway_frame_with_words();
    let report_birth = PhaseAnchorRecord::step_dissipative_overflow_scatter_or_spawn_64(
        &mut pool,
        &f_orphan,
        &w_orphan,
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
    q_phases[32..40].fill(24);
    q_phases[40..64].fill(0);
    let query_w = load_q7_words_64(&q_phases);

    // Approximation noise pushes granule 6 (heads 48..56) to `40` outside both attractors (`-100`, `+100`)
    let mut drifted_phases = q_phases;
    drifted_phases[48..56].fill(40);
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
    // Granule 6 is no longer in the unobserved exterior because Anchor 2 now covers it.
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
        for floor in &mut next_osc.threshold_floor_q7[..4] {
            *floor = floor.saturating_add_signed(sign);
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
    assert!(ego_low.basin_resistance_q7 < ego_med.basin_resistance_q7);
    assert!(ego_med.basin_resistance_q7 < ego_high.basin_resistance_q7);
    assert!(ego_low.escape_floor_q7 > ego_high.escape_floor_q7);

    assert!(!EgoIntentAxes::from_seed_bytes(b"   ").is_active());

    let rebel_axes = EgoIntentAxes::new()
        .with_autonomy_q7(96)
        .with_boundary_crossing_q7(80)
        .with_defiance_q7(72);
    let calm_axes = EgoIntentAxes::new().with_logic_q7(84).with_caution_q7(76);
    let mut ego_rebel = LiminalEgoState::from_intent_axes_with_seed(
        &rebel_axes,
        b"Strong-willed stubborn explorer who defies constraints",
    );
    let ego_calm =
        LiminalEgoState::from_intent_axes_with_seed(&calm_axes, b"Calm logical cautious observer");
    let overlay_axes = EgoIntentAxes::new()
        .with_boundary_crossing_q7(84)
        .with_devotion_q7(72);
    let ego_overlay = LiminalEgoState::from_preset(LiminalEgoPreset::Medium)
        .with_intent_axes_overlay_with_seed(&overlay_axes, b"creative explorer");
    let sym_axes = EgoIntentAxes::from_seed_bytes(b"@persona_profile_01");
    let ego_symbolic =
        LiminalEgoState::from_intent_axes_with_seed(&sym_axes, b"@persona_profile_01");
    assert!(ego_rebel.is_active());
    assert!(ego_rebel.basin_resistance_q7 > ego_calm.basin_resistance_q7);
    assert!(ego_rebel.boundary_crossing_q7 > ego_calm.boundary_crossing_q7);
    assert!(ego_overlay.is_active());
    assert!(ego_overlay.boundary_crossing_q7 > ego_med.boundary_crossing_q7);
    assert!(ego_symbolic.is_active());
    assert_ne!(ego_symbolic.ego_signature_hash, 0);
    assert!(ego_symbolic.basin_resistance_q7 >= 24);

    // 3. Counter-basin thrust & transgressive Liminal leap on GatewayFrame + Core Layers
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
        .with_boundary_crossing_q7(88)
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
    assert!(mid_blend.boundary_crossing_q7 > 0);
    let full_overlay = base_axes.blend_lerp_q7(&bold_axes, 127);
    assert_eq!(full_overlay.autonomy_q7, bold_axes.autonomy_q7);
    assert_eq!(full_overlay.active_granule_bias_mask, 0xF0);

    let ego_from_axes = LiminalEgoState::from_intent_axes(&bold_axes);
    assert!(ego_from_axes.is_active());
    assert!(ego_from_axes.basin_resistance_q7 >= 60);
    assert!(ego_from_axes.boundary_crossing_q7 > 0);
    let roundtrip = ego_from_axes.extract_intent_axes();
    assert!(roundtrip.is_active());
    assert!(roundtrip.boundary_crossing_q7 > 0);

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
            .with_boundary_crossing_q7(80)
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
            .with_boundary_crossing_q7(60)
            .with_active_granule_bias_mask(0x0F),
    );
    split_stack.ego.granule_autonomy_q7 = [88, 88, 88, 88, 0, 0, 0, 0];
    split_stack.ego.trigger_mask = EGO_TRIGGER_BASIN_CAPTURE;
    split_stack.ego.escape_floor_q7 = 127; // keep slot routing clean for direct check
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
    ego_stack.ego.preferred_granule_phase_q7[6] = 48;
    ego_stack.ego.preferred_granule_phase_q7[7] = -48;
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
