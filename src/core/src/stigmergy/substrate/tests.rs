// name: src/core/src/stigmergy/substrate/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::foundation::{AlignedBuffer32, pack_action_slot_tag};
use crate::liminal::LMNK_ERR_SUBSTRATE_FULL;
use crate::stigmergy::trail::PointerSubstrate;
use crate::stigmergy::{
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION,
    MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE,
};

#[test]
fn test_stigmergic_thresholds_ladder_all_regimes() {
    let thresh = StigmergicThresholds::new();
    assert!(thresh.validate().is_ok());

    // 0. PassiveSidecar: low density (0.10 < 0.35), normal entropy (0.30 > 0.15)
    let (dec0, flags0, slot0, _) =
        thresh.select_action_decision(0.10, 0.30, 0.80, false, ATTRACTOR_SLOT_PERSONA);
    assert_eq!(dec0, StigmergicActionDecision::PassiveSidecar);
    assert_eq!(flags0, ACTION_PRESET_PASSIVE_SIDECAR);
    assert_eq!(slot0, ATTRACTOR_SLOT_PERSONA);

    // 1. InertialLayerSkip: low entropy (0.08 <= 0.15)
    let (dec1, flags1, _, extra1) =
        thresh.select_action_decision(0.50, 0.08, 0.90, false, ATTRACTOR_SLOT_PERSONA);
    assert_eq!(dec1, StigmergicActionDecision::InertialLayerSkip);
    assert_eq!(flags1, ACTION_PRESET_INERTIAL_SKIP);
    assert_ne!(extra1 & TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);

    // 2. ExpressPhaseSteer: 0.35 <= density < 0.85, normal entropy
    let (dec2, flags2, _, _) =
        thresh.select_action_decision(0.55, 0.30, 0.85, false, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(dec2, StigmergicActionDecision::ExpressPhaseSteer);
    assert_eq!(flags2, ACTION_PRESET_EXPRESS_PHASE);

    // 2b. High-entropy fallback requires at least express_steer_density_thresh; sub-express stays PassiveSidecar.
    let (dec2_low, _, _, _) =
        thresh.select_action_decision(0.10, 0.60, 0.85, false, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(dec2_low, StigmergicActionDecision::PassiveSidecar);
    let (dec2_fb, _, _, _) =
        thresh.select_action_decision(0.55, 0.60, 0.85, false, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(dec2_fb, StigmergicActionDecision::SubHeadRoute);

    // 3. SubHeadRoute: 0.85 <= density < 1.50
    let (dec3, flags3, _, _) =
        thresh.select_action_decision(1.10, 0.30, 0.85, false, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(dec3, StigmergicActionDecision::SubHeadRoute);
    assert_eq!(flags3, ACTION_MODE_PHASE_STEER | ACTION_MODE_SUBHEAD_ROUTE);

    // 4. FullOrchestration: 1.50 <= density < 2.20
    let (dec4, flags4, _, _) =
        thresh.select_action_decision(1.80, 0.30, 0.85, false, ATTRACTOR_SLOT_PERSONA);
    assert_eq!(dec4, StigmergicActionDecision::FullOrchestration);
    assert_eq!(flags4, ACTION_PRESET_FULL_ORCHESTRATION);

    // 5. CrystallizeMacroBasin: density >= 2.20 and coherence >= 0.85
    let (dec5, flags5, _, extra5) =
        thresh.select_action_decision(2.50, 0.10, 0.92, false, ATTRACTOR_SLOT_PERSONA);
    assert_eq!(dec5, StigmergicActionDecision::CrystallizeMacroBasin);
    assert_eq!(flags5, ACTION_PRESET_FULL_ORCHESTRATION);
    assert_ne!(extra5 & TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, 0);

    // 6. RepulsionEscape: coherence < -0.25 or repulsion_triggered
    let (dec6, flags6, slot6, extra6) =
        thresh.select_action_decision(2.50, 0.30, -0.60, false, ATTRACTOR_SLOT_PERSONA);
    assert_eq!(dec6, StigmergicActionDecision::RepulsionEscape);
    assert_eq!(flags6, ACTION_PRESET_REPULSION_ESCAPE);
    assert_eq!(slot6, ATTRACTOR_SLOT_REPULSION);
    assert_ne!(extra6 & TRAIL_FLAG_REPULSION_ACTIVE, 0);
}

#[test]
fn test_substrate_engine_stigmergic_orchestration_and_gateway_kmeans() {
    let mut substrate =
        StigmergicSubstrate::with_dimensions_and_thresholds(8, 2, 16, StigmergicThresholds::new())
            .unwrap();

    // Populate an inline candidate trail
    let mut t0 = PointerTrail {
        stigmergic_density: 1.2,
        target_anchor_id: 42,
        accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let ptr0 = TaggedPtr::pack_action(
        0x1000,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_EXPRESS_PHASE,
        24,
    );
    substrate.step_pointer(ptr0.raw(), &mut t0).unwrap();
    assert_eq!(substrate.active_count(), 1);

    let mut q_buf = AlignedBuffer32::new([1.0f32; 16]);
    let mut k_buf = AlignedBuffer32::new([1.0f32; 16]);
    let mut logits = AlignedBuffer32::new([2.0f32; 8]);
    let mut scores = AlignedBuffer32::new([0.0f32; 8]);
    let mut active_trail = PointerTrail {
        stigmergic_density: 1.2,
        accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };

    let report = substrate
        .orchestrate_next_action_step_core(
            ptr0,
            0.35,
            ATTRACTOR_SLOT_PERSONA,
            &mut active_trail,
            Some(&mut scores.as_aligned_slice_mut()),
            Some(&mut q_buf.as_aligned_slice_mut()),
            Some(&mut k_buf.as_aligned_slice_mut()),
            Some(&mut logits.as_aligned_slice_mut()),
            true,
            None,
            crate::foundation::DEFAULT_PHASE_STEP_MIN_RAD,
            crate::foundation::DEFAULT_PHASE_CLAMP_MAX_RAD,
        )
        .unwrap();

    assert_eq!(
        report.action_decision(),
        StigmergicActionDecision::FullOrchestration
    );
    assert!(report.phase_coherence > 0.95);
    assert_eq!(active_trail.target_anchor_id, 42);

    // Multi-hop chain test:
    // Verify multi-hop pointer-chain traversal (trail[0] -> trail[1] -> trail[2])
    substrate.trails[0].linked_trail_id = 1;
    substrate.trails[0].stigmergic_density = 0.8;
    substrate.trails[0].accumulated_phase_delta = [8i8; MAX_ATTENTION_HEADS];
    substrate.trails[1].linked_trail_id = 2;
    substrate.trails[1].stigmergic_density = 1.1;
    substrate.trails[1].accumulated_phase_delta = [10i8; MAX_ATTENTION_HEADS];
    substrate.trails[2].linked_trail_id = 2; // terminal self-link stops traversal cleanly
    substrate.trails[2].stigmergic_density = 1.6;
    substrate.trails[2].target_anchor_id = 777;
    substrate.trails[2].current_ptr = TaggedPtr::pack(
        0x2000,
        pack_action_slot_tag(ATTRACTOR_SLOT_PERSONA, ACTION_PRESET_FULL_ORCHESTRATION),
        12,
    )
    .raw();
    substrate.trails[2].accumulated_phase_delta = [12i8; MAX_ATTENTION_HEADS];
    substrate.active_count = 3;

    let mut chain_trail = PointerTrail::default();
    let (chain_rep, hops, term_idx, term_aid, chain_coh) = substrate
        .traverse_pointer_chain_and_orchestrate(
            ChainTraverseParams {
                start_trail_idx: 0,
                max_hops: 8,
                entropy_h_tilde: 0.25,
            },
            &mut chain_trail,
            None,
            Some(&mut q_buf.as_aligned_slice_mut()),
            Some(&mut k_buf.as_aligned_slice_mut()),
            None,
        )
        .unwrap();
    assert_eq!(hops, 3);
    assert_eq!(term_idx, 2);
    assert_eq!(term_aid, 777);
    assert!(chain_coh > 0.95);
    assert_eq!(chain_rep.dominant_anchor_id, 777);
    assert!(substrate.trails[1].stigmergic_density > 1.1);
}

#[test]
fn test_workbench_free_by_default_allocator_pin_evict_decay_and_sandbox() {
    let mut substrate = StigmergicSubstrate::new();
    assert_eq!(substrate.active_lane_mask(), 0x00);
    assert_eq!(substrate.pinned_lane_mask(), 0x00);
    assert_eq!(substrate.repulsion_lane_mask(), 0x00);
    assert_eq!(substrate.sandbox_lane_mask(), 0x00);
    assert_eq!(substrate.free_lane_mask(), 0xFF);
    assert_eq!(substrate.lane_utilization_count(), 0);
    assert_eq!(substrate.slot_weights, [0.0; MAX_ATTRACTOR_DOMAINS]);

    // Deterministic AutoFree allocation fills physical lanes 0..=7 in ascending order.
    let mut allocated = [0u8; MAX_ATTRACTOR_DOMAINS];
    for (i, slot) in allocated.iter_mut().enumerate() {
        let weight = 0.25 + (i as f32) * 0.25;
        *slot = substrate
            .allocate_lane(&WorkbenchSlotSpec::auto_free(
                weight,
                [(i as i8) + 1; MAX_ATTENTION_HEADS],
            ))
            .unwrap();
        assert_eq!(*slot, i as u8);
    }
    assert_eq!(substrate.active_lane_mask(), 0xFF);
    assert_eq!(substrate.lane_utilization_count(), 8);
    assert_eq!(substrate.free_lane_mask(), 0x00);

    // Pin lowest-weight lane 0: overflow AutoFree skips pinned lane 0 and evicts lane 1 (0.50).
    substrate.pin_lane(0, true).unwrap();
    assert_eq!(substrate.pinned_lane_mask(), 0x01);
    let evicted_lane = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_repulsion(
            1.25,
            [-12; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    assert_eq!(evicted_lane, 1);
    assert_eq!(substrate.repulsion_lane_mask(), 1 << 1);
    assert_eq!(substrate.slot_phase_bank.data[0], 1);
    assert_eq!(substrate.slot_phase_bank.data[MAX_ATTENTION_HEADS], -12);

    // Pin all 8 lanes -> AutoFree returns LMNK_ERR_SUBSTRATE_FULL without mutating.
    for lane in 0..MAX_ATTRACTOR_DOMAINS as u8 {
        substrate.pin_lane(lane, true).unwrap();
    }
    assert_eq!(
        substrate.allocate_lane(&WorkbenchSlotSpec::auto_free(1.0, [4; MAX_ATTENTION_HEADS])),
        Err(LMNK_ERR_SUBSTRATE_FULL)
    );

    // Unpin lanes 1..=7, keep lane 0 pinned, then release_unpinned_lanes keeps only lane 0.
    for lane in 1..MAX_ATTRACTOR_DOMAINS as u8 {
        substrate.pin_lane(lane, false).unwrap();
    }
    let released = substrate.release_unpinned_lanes();
    assert_eq!(released, 7);
    assert_eq!(substrate.active_lane_mask(), 0x01);
    assert_eq!(substrate.pinned_lane_mask(), 0x01);
    assert_eq!(substrate.repulsion_lane_mask(), 0x00);
    assert_eq!(substrate.slot_phase_bank.data[MAX_ATTENTION_HEADS], 0);

    // Allocate sandbox & standard unpinned lanes, verify decay_and_reclaim preserves pinned lane 0.
    let std_lane = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_free(
            0.80,
            [16; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    let sb_lane = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(
            0.20,
            [32; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    assert_eq!(std_lane, 1);
    assert_eq!(sb_lane, 2);
    assert_eq!(substrate.sandbox_lane_mask(), 1 << 2);

    let reclaimed = substrate.decay_and_reclaim_lanes(0.50, 0.25);
    assert_eq!(reclaimed, 1);
    assert_eq!(substrate.active_lane_mask(), (1 << 0) | (1 << 1));
    assert_eq!(substrate.sandbox_lane_mask(), 0);
    assert!((substrate.slot_weights[0] - 0.25).abs() < 1e-6);
    assert!((substrate.slot_weights[1] - 0.40).abs() < 1e-6);

    // Allocate sandbox lane again -> settle_sandbox_lane clears sandbox bit in-place;
    // release_sandbox_lanes prunes unpinned sandbox lanes.
    let sb2 = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(
            0.60,
            [24; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    assert_eq!(sb2, 2);
    assert!(substrate.settle_sandbox_lane(sb2).unwrap());
    assert_eq!(substrate.sandbox_lane_mask(), 0);
    assert_ne!(substrate.active_lane_mask() & (1 << sb2), 0);

    let sb3 = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(
            0.60,
            [28; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    assert_eq!(sb3, 3);
    assert_eq!(substrate.release_sandbox_lanes(), 1);
    assert_eq!(substrate.active_lane_mask() & (1 << sb3), 0);
}

#[test]
fn test_stigmergic_calculus_convex_combination_and_boundary_gate() {
    use crate::stigmergy::granule::StigmergicGranuleState;

    // 1. 4p(1-p) boundary sensitivity gate vanishes at ignorance (0) and saturation (>= 128),
    //    returns 3 at 127, and peaks at 64 (127).
    assert_eq!(StigmergicGranuleState::boundary_sensitivity_gate_q7(0), 0);
    assert_eq!(
        StigmergicGranuleState::boundary_sensitivity_gate_q7(64),
        127
    );
    assert_eq!(StigmergicGranuleState::boundary_sensitivity_gate_q7(127), 3);
    assert_eq!(StigmergicGranuleState::boundary_sensitivity_gate_q7(128), 0);
    assert_eq!(StigmergicGranuleState::boundary_sensitivity_gate_q7(255), 0);

    // 2. alpha = 0 reduces identically to pure integral density.
    let (eff_zero, _) =
        StigmergicThresholds::compute_effective_density_calculus(0.10, 0.30, 0.80, 0, 4, true);
    assert!((eff_zero - 0.30).abs() < 1e-6);

    // 3. Single-step small noise just above deadband (0.015625) without prev_rising is suppressed.
    let (eff_noise, rising_flag) =
        StigmergicThresholds::compute_effective_density_calculus(0.10, 0.12, 0.80, 64, 4, false);
    assert!(rising_flag);
    assert!(eff_noise < 0.12);

    // 4. Two consecutive rising steps (or strong rising edge) boost effective density into Express range.
    let (eff_boost, _) =
        StigmergicThresholds::compute_effective_density_calculus(0.10, 0.30, 0.80, 64, 4, true);
    assert!(eff_boost > 0.45);
    assert!(eff_boost <= MAX_STIGMERGIC_DENSITY);

    // 5. Saturated steady state (delta == 0) is pulled down by (1 - alpha) to relieve PassiveSaturation.
    let (eff_sat, rising_sat) =
        StigmergicThresholds::compute_effective_density_calculus(2.00, 2.00, 0.90, 64, 4, true);
    assert!(!rising_sat);
    assert!(eff_sat < 1.10);
}

#[test]
fn test_substrate_bitcut8_quadrants_layout_and_eviction() {
    assert_eq!(size_of::<StigmergicSubstrate>(), 2816);
    assert_eq!(align_of::<StigmergicSubstrate>(), 64);

    let mut substrate = StigmergicSubstrate::new();
    // Lane 0: standard attractive (unpinned)
    let l0 = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_free(
            1.0,
            [12; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    // Lane 1: pinned attractive
    let l1 = substrate
        .allocate_lane(&WorkbenchSlotSpec::explicit(
            1,
            true,
            0,
            1.5,
            [24; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    // Lane 2: global repulsion
    let l2 = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_repulsion(
            0.8,
            [-16; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    // Lane 3: pure sandbox hypothesis
    let l3 = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(
            0.6,
            [32; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    // Lane 4: REPULSION | SANDBOX | PINNED
    let l4 = substrate
        .allocate_lane(&WorkbenchSlotSpec {
            placement: WorkbenchLanePlacement::Explicit { lane: 4, pin: true },
            weight: 0.9,
            flags: WORKBENCH_LANE_FLAG_PINNED
                | WORKBENCH_LANE_FLAG_REPULSION
                | WORKBENCH_LANE_FLAG_SANDBOX,
            phases: [-32; MAX_ATTENTION_HEADS],
        })
        .unwrap();

    let (active_cut, free_cut) = substrate.lane_lifecycle_quadrants();
    assert_eq!(active_cut.inside, (1 << l1) | (1 << l4));
    assert_eq!(active_cut.outside, (1 << l0) | (1 << l2) | (1 << l3));
    assert_eq!(free_cut.inside, 0);
    assert_eq!(
        active_cut.inside | active_cut.outside | free_cut.inside | free_cut.outside,
        0xFF
    );

    let (rep_cut, attract_cut) = substrate.lane_role_quadrants();
    assert_eq!(rep_cut.outside, 1 << l2);
    assert_eq!(rep_cut.inside, 1 << l4);
    assert_eq!(attract_cut.outside, (1 << l0) | (1 << l1));
    assert_eq!(attract_cut.inside, 1 << l3);
    assert_eq!(
        rep_cut.outside | rep_cut.inside | attract_cut.outside | attract_cut.inside,
        substrate.active_lane_mask()
    );
}
