// name: src/core/src/traits/telemetry/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::traits::primitives::ACTION_PRESET_EXPRESS_PHASE;
use crate::traits::primitives::ATTRACTOR_SLOT_PERSONA;

struct DummyDaes;
impl FastPathLedger for DummyDaes {
    fn evaluate_auto_tune(
        &self,
        _slot: &gcso_ledger_slot_t,
        _config_out: &mut gcso_config_t,
    ) -> GcsoResult<()> {
        Ok(())
    }
}

#[test]
fn test_paged_bitmask_bloom_trail_swarm_and_daes_constant_time() {
    let mut m1 = gcso_paged_bitmask_t::default();
    let mut m2 = gcso_paged_bitmask_t::default();
    m1.set_bit_branchless(0);
    m1.set_bit_branchless(65);
    m1.set_bit_branchless(255);
    m2.set_bit_branchless(65);
    m2.set_bit_branchless(128);
    m2.set_bit_branchless(255);

    assert_eq!(m1.popcount(), 3);
    assert_eq!(m1.stigmergic_adjacency_overlap(&m2), 2);
    assert!((m1.stigmergic_cosine_similarity(&m2) - (2.0 / 3.0)).abs() < 1e-4);
    assert!((m1.stigmergic_jaccard_similarity(&m2) - 0.5).abs() < 1e-4);

    let mut filter = ContrastivePhaseBloomFilter::new();
    let mut pattern = [12i8; MAX_ATTENTION_HEADS];
    pattern[63] = -40;
    filter.insert_repulsive_pattern(&pattern);
    filter.insert_repulsive_pattern(&pattern);
    assert_eq!(filter.query_repulsive_density(&pattern), 2);

    let init_ptr = gcso_tagged_ptr_t::pack_action(
        0x4000,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_EXPRESS_PHASE,
        24,
    );
    let (evolved_ptr, density, triggered) =
        filter.apply_repulsion_and_evolve_ptr(init_ptr, &mut pattern, 2, 128);
    assert!(triggered && density == 2 && pattern[0] == -12 && pattern[63] == 40);
    assert_eq!(evolved_ptr.target_attractor_slot(), ATTRACTOR_SLOT_REPULSION);

    for step in 0..100u8 {
        let p = [(step as i8).wrapping_mul(3); MAX_ATTENTION_HEADS];
        filter.insert_repulsive_pattern_weighted(&p, 8);
        assert!(filter.total_mass() <= DEFAULT_BLOOM_MAX_MASS_THRESHOLD + 32);
    }

    let mut trail = gcso_pointer_trail_t {
        stigmergic_density: 1.0,
        target_anchor_id: 42,
        ..Default::default()
    };
    trail.accumulate_uniform_phase(16);
    let (r_bar, mean_q7, kappa, grav) = trail.evaluate_trail_circular_stats_and_gravity();
    assert!(r_bar > 0.99 && mean_q7 == 16 && kappa > 10.0 && grav > 1.0);

    let mut target_phases = [24i8; MAX_ATTENTION_HEADS];
    target_phases[48..64].fill(112);

    let report = trail.try_resolve_action_in_swarm(
        init_ptr,
        &target_phases,
        64,
        1,
        DEFAULT_STIGMERGIC_TOLERANCE_Q7,
        DEFAULT_STIGMERGIC_QUORUM_THRESHOLD,
        0.10,
        0.15,
        None,
        2,
        128,
    );
    assert_eq!(report.hot_path_resolved, 1);
    assert_eq!(report.resolved_cell_count, 6);
    assert_eq!(report.quorum_mask, 0b0011_1111);
    assert_eq!(trail.cluster_id, 42);
    assert_eq!(report.evolved_ptr.target_attractor_slot(), 42 & 7);

    let mut daes_ctrl = DummyDaes;
    let mut ledger_slot = gcso_ledger_slot_t::default();
    daes_ctrl.register_fast_path_shortcut(&mut ledger_slot, init_ptr.raw(), 0x9999);
    let (hit, resolved_ptr) = daes_ctrl.resolve_or_pass_tagged_ptr(&mut ledger_slot, init_ptr);
    assert!(hit && resolved_ptr.raw() == 0x9999);
    daes_ctrl
        .telemetry_push(
            &mut ledger_slot,
            LEDGER_METRIC_FAST_PATH_HIT | LEDGER_METRIC_PHASE_STEERED,
        )
        .unwrap();
    let (mask, count) = daes_ctrl.telemetry_snapshot_metrics(&ledger_slot);
    assert_eq!(
        (mask, count),
        (LEDGER_METRIC_FAST_PATH_HIT | LEDGER_METRIC_PHASE_STEERED, 1)
    );
}

#[test]
fn test_liminal_bridge_order_invariant_unified_field_and_bayesian_pointer_rewire() {
    let mut mem_a = gcso_pointer_trail_t {
        current_ptr: 0x1000,
        step_count: 5,
        target_anchor_id: 42,
        attractor_pull_force: 2.0,
        accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };

    let mut mem_b = gcso_pointer_trail_t {
        current_ptr: 0x2000,
        stigmergic_density: 1.0,
        attractor_pull_force: 1.0,
        accumulated_phase_delta: [48i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };

    let (coh, log_prob, repulsed) =
        mem_a.link_with_next_hallucinated_trail(9, &mut mem_b, 64, 0.25);
    assert!(!repulsed && coh > 0.90 && log_prob > -0.10);
    assert_eq!(mem_a.linked_trail_id, 9);
    assert_eq!(mem_b.prev_ptr, 0x1000);
    assert_eq!(mem_b.step_count, 6);
    assert_eq!(mem_b.target_anchor_id, 42);
    assert_eq!(mem_b.cluster_id, 42);
    assert_eq!(mem_b.accumulated_phase_delta[0], 40);
    assert_ne!(mem_b.flags & TRAIL_FLAG_SYNERGY_RESONANCE, 0);

    let anchor_phases = [40i8; MAX_ATTENTION_HEADS];
    let (res_pull, res_coh) = mem_b.apply_unified_field_anchor_resonance(&anchor_phases, 1.5);
    assert!((res_coh - 1.0).abs() < 1e-4 && (res_pull - 3.0).abs() < 1e-4);

    let anchors_fwd = [
        [16i8; MAX_ATTENTION_HEADS],
        [40i8; MAX_ATTENTION_HEADS],
        [64i8; MAX_ATTENTION_HEADS],
    ];
    let energies_fwd = [1.0f32, 2.5, 1.0];
    let anchors_rev = [
        [64i8; MAX_ATTENTION_HEADS],
        [40i8; MAX_ATTENTION_HEADS],
        [16i8; MAX_ATTENTION_HEADS],
    ];
    let energies_rev = [1.0f32, 2.5, 1.0];

    let mut trail_fwd = mem_b;
    let mut trail_rev = mem_b;
    let (pull_f, coh_f, cnt_f) =
        trail_fwd.apply_unified_field_anchor_slice_resonance(&anchors_fwd, &energies_fwd, None);
    let (pull_r, coh_r, cnt_r) =
        trail_rev.apply_unified_field_anchor_slice_resonance(&anchors_rev, &energies_rev, None);
    assert_eq!((cnt_f, cnt_r), (3, 3));
    assert!((pull_f - pull_r).abs() < 1e-5 && (coh_f - coh_r).abs() < 1e-5);
    assert_eq!((trail_fwd.target_anchor_id, trail_rev.target_anchor_id), (1, 1));

    let cand_bad = gcso_pointer_trail_t {
        stigmergic_density: 5.0,
        accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };

    let cand_good = gcso_pointer_trail_t {
        stigmergic_density: 3.0,
        accumulated_phase_delta: [42i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };

    let candidates = [cand_bad, cand_good];
    let mut scores = [0.5f32, 0.5f32];
    let self_w = crate::traits::primitives::load_q7_words_64(&mem_b.accumulated_phase_delta);
    let (selected_id, best_score, mean_coh, coh_cnt, found, _bridged_w, _win_coh) =
        select_rewire_bridge_and_modulate_with_preloaded_words_64(
            &mut mem_b,
            self_w,
            &candidates,
            Some(&mut scores),
            128,
            2.0,
            0.5,
            0.25,
            0.25,
            64,
            8.0,
            0,
            None,
            0.0,
        );
    assert!(found && selected_id == 1 && coh_cnt == 1);
    assert_eq!(mem_b.linked_trail_id, 1);
    assert!(best_score > 0.5 && mean_coh > -0.1 && scores[1] > scores[0]);

    let mut mem_contradict = gcso_pointer_trail_t {
        current_ptr: 0x3000,
        stigmergic_density: 2.0,
        accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let (coh_bad, _lp_bad, repulsed_bad) =
        mem_contradict.bridge_liminal_hallucinated_trail(&mem_a, 64, 0.25);
    assert!(repulsed_bad && coh_bad < -0.90);
    assert_eq!(mem_contradict.accumulated_phase_delta[0], 96);
    assert_ne!(mem_contradict.flags & TRAIL_FLAG_REPULSION_ACTIVE, 0);
}
