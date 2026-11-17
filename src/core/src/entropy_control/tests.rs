// name: src/core/src/entropy_control/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
#[allow(unused_imports)]
use super::{closed_loop::*, controller::*, kernels::*};
use crate::traits::aligned::AlignedBuffer32;
use crate::traits::primitives::{
    ACTION_PRESET_EXPRESS_PHASE, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
};
use crate::traits::telemetry::PagedBitmaskOps;

#[test]
fn test_surrogate_and_moving_z_entropy() {
    let entropy = compute_sparse_surrogate_entropy(0.5, 0.5, DEFAULT_ENTROPY_EPS);
    assert!((entropy - LN_2).abs() < 1e-6);
    assert_eq!(
        compute_sparse_surrogate_entropy(1.0, 0.0, DEFAULT_ENTROPY_EPS),
        0.0
    );
    assert_eq!(
        compute_sparse_surrogate_entropy(-0.5, 0.5, DEFAULT_ENTROPY_EPS),
        0.0
    );

    let asym_h = compute_sparse_surrogate_entropy(0.8, 0.2, DEFAULT_ENTROPY_EPS);
    assert!(asym_h > 0.45 && asym_h < LN_2);

    let unnorm_h = compute_sparse_surrogate_entropy(1.6, 0.4, DEFAULT_ENTROPY_EPS);
    assert!((asym_h - unnorm_h).abs() < 1e-6);

    let tail_h = compute_sparse_surrogate_entropy(0.99, 0.01, DEFAULT_ENTROPY_EPS);
    assert!(tail_h > 0.03 && tail_h < 0.07);

    let h_tilde = compute_moving_z_score_entropy(0.5, 0.8, 0.2, 0.1, 1e-6, 1.0, 0.5);
    assert!(h_tilde > 0.5);
    assert_eq!(
        compute_moving_z_score_entropy(0.5, 0.1, 0.2, 0.1, 1e-6, 1.0, 0.5),
        0.5
    );
    assert_eq!(
        compute_moving_z_score_entropy(0.0, 0.8, 0.2, 0.1, 1e-6, 1.0, 0.5),
        0.0
    );
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
    engine.state.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
    engine.reset();
    assert_eq!(engine.moving_z_entropy(), 0.0);
    assert!(engine.is_normal_mode());
}

#[test]
fn test_eval_stateful_modes_spike_ceiling_and_dwell_anti_chattering() {
    let mut engine = EntropyController::default().with_min_mode_dwell_steps(3);
    engine.set_bifurcation_threshold(1.0);
    engine.set_hysteresis_band(0.05);
    engine.set_tau_flux(0.5);
    let mut out_state = gcso_edbc_state_t::default();

    engine.state.moving_z_entropy = 2.0;
    engine.eval_stateful(10.0, &mut out_state).unwrap();
    assert_eq!(
        out_state.active_branch_mode,
        EDBC_MODE_PHASE_CONJUGATE_REPULSION
    );
    assert_eq!(engine.mode_dwell_steps, 1);

    let (emit_iframe, gop_len, can_skip, gate_scale) =
        engine.evaluate_adaptive_compute_and_gop_budget(6, 4, 32, 0.20);
    assert!(emit_iframe);
    assert_eq!(gop_len, 6);
    assert!(!can_skip);
    assert!((gate_scale - 1.0).abs() < 1e-5);

    engine.state.moving_z_entropy = 0.2;
    engine.eval_stateful(0.0, &mut out_state).unwrap();
    assert_eq!(
        out_state.active_branch_mode,
        EDBC_MODE_PHASE_CONJUGATE_REPULSION
    );
    assert_eq!(engine.mode_dwell_steps, 2);

    engine.eval_stateful(0.0, &mut out_state).unwrap();
    assert_eq!(out_state.active_branch_mode, EDBC_MODE_NORMAL_ENTROPY);

    let (emit_after, _gop_after, can_skip_after, _gate_after) =
        engine.evaluate_adaptive_compute_and_gop_budget(2, 4, 32, 0.25);
    assert!(!emit_after);
    assert!(can_skip_after);

    engine.reset();
    engine.eval_stateful(1.0e5, &mut out_state).unwrap();
    assert!(out_state.moving_z_entropy <= 1.0 + 1e-5);
    for _ in 0..15 {
        engine.eval_stateful(0.0, &mut out_state).unwrap();
    }
    assert!(engine.is_normal_mode());

    let (_flux, accel) = engine.eval_entropy_flux_acceleration();
    assert!(accel.is_finite());

    assert_eq!(
        engine.eval_stateful(f32::NAN, &mut out_state),
        Err(GCSO_ERROR_EDBC_SINGULARITY)
    );
}

#[test]
fn test_liminal_void_barriers_pointer_coupled_cohomology_limina_and_unified_field() {
    let mut engine = EntropyController::default();
    let mut arr = [2.0f32; 35];
    arr[32..].copy_from_slice(&[3.0f32; 3]);
    let buf = AlignedBuffer32::new(arr);
    let slice = buf.as_aligned_slice();

    let mut score = 0.0f32;
    engine
        .eval_liminal_void_dyadic128(&slice, &mut score)
        .unwrap();
    let expected = (32.0 * 4.0 + 3.0 * 9.0) / 35.0;
    assert!((score - expected).abs() < 1e-4);

    let (mean_sq, coh) = engine
        .eval_liminal_void_with_coherence_ratio(&slice)
        .unwrap();
    assert!((mean_sq - expected).abs() < 1e-4);
    assert!(coh > 0.95 && coh <= 1.0);

    let buf128 = AlignedBuffer32::new([2.0f32; 128]);
    let mut score128 = 0.0f32;
    engine
        .eval_liminal_void_dyadic128(&buf128.as_aligned_slice(), &mut score128)
        .unwrap();
    assert!((score128 - 4.0).abs() < 1e-6);

    assert!((engine.eval_barrier(2.0, 1.0).unwrap() - 0.5).abs() < 1e-5);
    assert!(engine.eval_barrier(0.0, 1.0).unwrap() <= MAX_BARRIER_CEILING);
    assert_eq!(
        engine.eval_barrier(2.0, -1.0),
        Err(GCSO_ERROR_EDBC_SINGULARITY)
    );
    assert!(
        (eval_barrier_pitchfork(2.0, 1.0, 1.0, 1.0, 128).unwrap() - (1.0 / 512.0)).abs() < 1e-6
    );
    assert_eq!(
        eval_barrier_pitchfork(0.5, 1.0, 1.0, 1.0, 128).unwrap(),
        0.0
    );

    let ptr = gcso_tagged_ptr_t::pack_action(
        0x4000,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_EXPRESS_PHASE,
        16,
    );
    engine.state.moving_z_entropy = 2.0;
    let (st, evolved_ptr) = engine
        .eval_stateful_with_liminal_void_and_pointer(
            10.0,
            0.2,
            1.0,
            1.0,
            0.5,
            ptr,
            1.0,
            ATTRACTOR_SLOT_KNOWLEDGE,
            1.5,
        )
        .unwrap();
    assert_eq!(st.active_branch_mode, EDBC_MODE_PHASE_CONJUGATE_REPULSION);
    assert!(st.sample_temperature > 1.0);
    assert_eq!(
        evolved_ptr.target_attractor_slot(),
        ATTRACTOR_SLOT_REPULSION
    );
    assert_eq!(
        evolved_ptr.action_mode_flags() & ACTION_PRESET_REPULSION_ESCAPE,
        ACTION_PRESET_REPULSION_ESCAPE
    );

    let mut phases = [0i8; MAX_ATTENTION_HEADS];
    let (st_bif, _ptr_bif, x_star) = engine
        .step_and_inject_tunneling_perturbation(
            10.0,
            3.0,
            4.0,
            1.0,
            0.5,
            128,
            48,
            ptr,
            1.0,
            ATTRACTOR_SLOT_KNOWLEDGE,
            1.5,
            &mut phases,
        )
        .unwrap();
    assert_eq!(st_bif.active_branch_mode, EDBC_MODE_PITCHFORK_BIFURCATION);
    assert!(x_star > 0.0);
    assert_eq!(phases[0], 0);
    assert_ne!(phases[48], 0);

    engine.reset();
    let r_cochain = AlignedBuffer32::new([2.0f32; 8]);
    let p_exact = AlignedBuffer32::new([0.5f32; 8]);
    let gamma_cycle = AlignedBuffer32::new([1.0f32; 8]);
    let mut active_p = [24i8; MAX_ATTENTION_HEADS];
    let mut bloom = ContrastivePhaseBloomFilter::new();

    let (st_coh, ptr_coh, mean_obs, dual_pair, rep_trig) = engine
        .step_with_cohomology_and_bloom_guard(
            0.1,
            0.9,
            1.0,
            1.0,
            0.5,
            &r_cochain.as_aligned_slice(),
            &p_exact.as_aligned_slice(),
            &gamma_cycle.as_aligned_slice(),
            1.0,
            ptr,
            1.0,
            ATTRACTOR_SLOT_PERSONA,
            1.5,
            &mut active_p,
            &mut bloom,
        )
        .unwrap();
    assert!(rep_trig);
    assert!((mean_obs - 2.25).abs() < 1e-5);
    assert!((dual_pair - 1.0).abs() < 1e-5);
    assert_eq!(
        st_coh.active_branch_mode,
        EDBC_MODE_PHASE_CONJUGATE_REPULSION
    );
    assert!(st_coh.sample_temperature > 1.0);
    assert_eq!(ptr_coh.target_attractor_slot(), ATTRACTOR_SLOT_REPULSION);
    assert_eq!(active_p[0], -24);
    assert_eq!(
        bloom.query_repulsive_density(&[24i8; MAX_ATTENTION_HEADS]),
        1
    );

    engine.reset();
    let unified_anchors = [[-96i8; MAX_ATTENTION_HEADS], [32i8; MAX_ATTENTION_HEADS]];
    let unified_energies = [0.2f32, 3.0f32];
    let mut uf_trail = gcso_pointer_trail_t {
        current_ptr: ptr.raw(),
        stigmergic_density: 1.0,
        accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let (_st_uf, _ptr_uf, tot_pull, mean_c, dom_id) = engine
        .step_with_unified_field_and_pointer(
            0.1,
            0.8,
            &unified_anchors,
            &unified_energies,
            1.5,
            &mut uf_trail,
        )
        .unwrap();
    assert_eq!(dom_id, 1);
    assert_eq!(uf_trail.cluster_id, 1);
    assert!(tot_pull > 5.0 && mean_c > 0.85);

    let t_next = gcso_pointer_trail_t {
        current_ptr: ptr.raw(),
        stigmergic_density: 1.2,
        accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let mut chain = [uf_trail, t_next];
    let (_st_lim, _ptr_lim, joint_lp, hol_obs, lim_rep) = engine
        .step_with_liminal_trail_chain_and_bloom_guard(
            &mut chain,
            64,
            0.25,
            0.20,
            0.1,
            0.8,
            1.5,
            ATTRACTOR_SLOT_PERSONA,
            &mut bloom,
        )
        .unwrap();
    assert!(!lim_rep && hol_obs < 0.05 && joint_lp > -0.05);
    assert_eq!(chain[1].cluster_id, 1);

    let t_contradict = gcso_pointer_trail_t {
        current_ptr: ptr.raw(),
        stigmergic_density: 1.2,
        accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let mut bad_chain = [uf_trail, t_contradict];
    let (st_bad, ptr_bad, _bad_lp, _bad_obs, bad_rep) = engine
        .step_with_liminal_trail_chain_and_bloom_guard(
            &mut bad_chain,
            64,
            0.25,
            0.20,
            0.1,
            0.8,
            1.5,
            ATTRACTOR_SLOT_PERSONA,
            &mut bloom,
        )
        .unwrap();
    assert!(bad_rep);
    assert_eq!(
        st_bad.active_branch_mode,
        EDBC_MODE_PHASE_CONJUGATE_REPULSION
    );
    assert_eq!(ptr_bad.target_attractor_slot(), ATTRACTOR_SLOT_REPULSION);

    let (_st_surr, _ptr_surr, void_s, coh_s) = engine
        .eval_surrogate_with_liminal_void_and_pointer(
            0.5,
            0.5,
            0.8,
            0.2,
            0.1,
            1.0,
            0.5,
            &slice,
            1.0,
            1.0,
            0.5,
            ptr,
            1.0,
            ATTRACTOR_SLOT_PERSONA,
            1.5,
        )
        .unwrap();
    assert!((void_s - expected).abs() < 1e-4);
    assert!(coh_s > 0.95);
}

#[test]
fn test_blanket_bayesian_pointer_selection_temperature_and_dyadic_tree() {
    let mut engine = EntropyController::default();
    engine.state.moving_z_entropy = 1.6;
    let ptr = gcso_tagged_ptr_t::pack_action(
        0x5000,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_EXPRESS_PHASE,
        16,
    );

    let mut src_trail = gcso_pointer_trail_t {
        current_ptr: ptr.raw(),
        stigmergic_density: 1.8,
        attractor_pull_force: 2.0,
        target_anchor_id: 19,
        cluster_id: 19,
        accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };

    let cand_bad = gcso_pointer_trail_t {
        current_ptr: 0x5100,
        stigmergic_density: 4.5,
        accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let cand_good = gcso_pointer_trail_t {
        current_ptr: 0x5200,
        stigmergic_density: 3.0,
        target_anchor_id: 27,
        accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let mut candidates = [cand_bad, cand_good];
    let mut bloom = ContrastivePhaseBloomFilter::new();
    let mut bayes_scores = AlignedBuffer32::new([0.0f32; 8]);

    let (st_bayes, _ptr_bayes, sel_id, bayes_score, trans_lp, tun_x, found) = engine
        .step_with_bayesian_pointer_selection_and_tunneling(
            &mut src_trail,
            &mut candidates,
            2.0,
            2.5,
            2.0,
            1.0,
            0.5,
            128,
            48,
            2.0,
            0.5,
            0.25,
            1.0,
            64,
            0.25,
            1.5,
            &mut bloom,
            Some(&mut bayes_scores.as_aligned_slice_mut()),
        )
        .unwrap();
    assert!(found);
    assert_eq!(st_bayes.active_branch_mode, EDBC_MODE_PITCHFORK_BIFURCATION);
    assert!(tun_x > 0.0);
    assert_eq!(sel_id, 1);
    assert_eq!(src_trail.linked_trail_id, 1);
    assert_eq!(src_trail.target_anchor_id, 27);
    assert_eq!(src_trail.cluster_id, 27);
    assert!(bayes_score > 0.5 && trans_lp > -0.25);
    assert!(bayes_scores.data[1] > 0.8 && bayes_scores.data[0] < -0.5);

    let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
    let (mean_coh, eff_inv_temp, best_id, coh_cnt) = engine
        .modulate_candidate_trails_with_edbc_temperature(
            &mut attn_scores.as_aligned_slice_mut(),
            &src_trail,
            &candidates,
            128,
            2.0,
            0.5,
            0.25,
            0.25,
            1.0,
            1.0,
            st_bayes.moving_z_entropy,
            0.9,
            1.5,
            1.5,
            0.5,
            0.5,
        )
        .unwrap();
    assert!(mean_coh.is_finite());
    assert!(eff_inv_temp > 0.5);
    assert_eq!(best_id, 1);
    assert_eq!(coh_cnt, 1);
    assert!(attn_scores.data[1] > attn_scores.data[0]);

    let mut m_cur = gcso_paged_bitmask_t::default();
    let mut m_anc = gcso_paged_bitmask_t::default();
    m_cur.set_bit_branchless(3);
    m_anc.set_bit_branchless(3);
    let (_st_tree, qdpt_lvl, d_v, tree_ptr) = engine
        .resolve_dyadic_tree_and_step_edbc(
            &m_cur,
            &m_anc,
            0.05,
            0,
            0.10,
            &[0x100, 0x200, 0x300, 0x400],
            &[4, 8, 12, 16],
            0.1,
            0.8,
            1.5,
            &mut src_trail,
        )
        .unwrap();
    assert_eq!(qdpt_lvl, 0);
    assert_eq!(d_v, 0.0);
    assert_eq!(tree_ptr.unpack_addr(), 0x5100);
    assert_eq!(src_trail.cluster_id, 27);
}

#[test]
fn test_entropy_origin1_system_constraint_gateway_kmeans_and_gcso_crystallization() {
    let mut engine = EntropyController::default();
    assert_eq!(engine.active_swarm_cell_mask(), 0x0F);
    assert!(engine.requests_inertial_layer_skip());
    let sys_only_frame = engine.synthesize_gateway_frame();
    assert_eq!(sys_only_frame.active_swarm_cell_mask, 0x0F);
    assert_ne!(sys_only_frame.action_mode_flags & ACTION_MODE_LAYER_SKIP, 0);

    let mut st_init = gcso_edbc_state_t::default();
    engine.eval_stateful(2.5, &mut st_init).unwrap();
    assert_eq!(engine.active_swarm_cell_mask(), 0xFF);

    let fact = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 2.5, 0x0012_3400, 33);
    let persona = SubjectivePersona::new([30i8; MAX_ATTENTION_HEADS], 2.0, 0x0020_0000, 19);
    let composite_gateway = (engine, fact, persona);

    let mut trail = gcso_pointer_trail_t::default();
    let mut logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
    let mut crystal =
        CrystallizedAttractorRecord::new_empty(33, ATTRACTOR_SLOT_KNOWLEDGE, 0x0012_3400);

    let mut prev_mass = 0.0_f32;
    for _ in 0..4 {
        let (_st, evolved_ptr, report, eff_inv_temp, mass) = engine
            .step_gateway_signal_with_defaults(
                0.35,
                0.85,
                &composite_gateway,
                &mut trail,
                Some(&mut logits.as_aligned_slice_mut()),
                Some(&mut crystal),
            )
            .unwrap();
        assert_eq!(evolved_ptr.unpack_addr(), 0x0012_3400);
        assert_eq!(trail.target_anchor_id, 33);
        assert_eq!(trail.cluster_id, 33);
        assert!(eff_inv_temp > 1.0);
        assert!(mass > prev_mass);
        assert!(report.fact_persona_coherence > 0.90);
        prev_mass = mass;
    }
    assert_eq!(crystal.sample_count, 4);
    assert!(crystal.verify_integrity());
    assert_ne!(
        crystal.provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
        0
    );

    let records = [crystal];
    let mut eval_trail = gcso_pointer_trail_t {
        attractor_pull_force: 1.5,
        accumulated_phase_delta: [26i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let (_st_ens, _ptr_ens, tot_pull, mean_coh, dom_id) = engine
        .step_crystallized_ensemble_with_defaults(0.25, 0.75, &records, &mut eval_trail)
        .unwrap();
    assert_eq!(dom_id, 33);
    assert_eq!(eval_trail.target_anchor_id, 33);
    assert_eq!(eval_trail.cluster_id, 33);
    assert!(tot_pull > 4.0 && mean_coh > 0.90);

    let mut macro_centroids = [
        CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0),
        CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
    ];
    let (_st_k1, _ptr_k1, _rep_k1, _inv_k1, c_idx1, c_id1, c_mass1, spawned1) = engine
        .step_gateway_kmeans_with_defaults(
            0.35,
            0.85,
            &composite_gateway,
            &mut macro_centroids,
            &mut eval_trail,
            None,
        )
        .unwrap();
    assert!(spawned1);
    assert_eq!(c_idx1, 0);
    assert_eq!(c_id1, 33);
    assert!(c_mass1 > 0.0);

    let opposed_persona =
        SubjectivePersona::new([-84i8; MAX_ATTENTION_HEADS], 2.8, 0x0028_0000, 99);
    let (_st_k2, _ptr_k2, _rep_k2, _inv_k2, c_idx2, c_id2, _c_mass2, spawned2) = engine
        .step_gateway_kmeans_with_defaults(
            0.40,
            0.85,
            &opposed_persona,
            &mut macro_centroids,
            &mut eval_trail,
            None,
        )
        .unwrap();
    assert!(spawned2);
    assert_eq!(c_idx2, 1);
    assert_eq!(c_id2, 99);

    let mut c2f_trail = gcso_pointer_trail_t {
        accumulated_phase_delta: [-80i8; MAX_ATTENTION_HEADS],
        stigmergic_density: 1.5,
        attractor_pull_force: 1.5,
        ..Default::default()
    };
    let mut c2f_candidates = [
        gcso_pointer_trail_t {
            cluster_id: 33,
            target_anchor_id: 33,
            stigmergic_density: 2.0,
            accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        },
        gcso_pointer_trail_t {
            cluster_id: 99,
            target_anchor_id: 990,
            stigmergic_density: 2.5,
            accumulated_phase_delta: [-82i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        },
    ];
    let mut bloom = ContrastivePhaseBloomFilter::new();
    let (_st_c2f, _ptr_c2f, m_idx, m_id, sel_id, _bs, _lp, _tx, found_c2f) = engine
        .route_coarse_to_fine_and_step_bayesian_tunneling(
            &macro_centroids,
            64,
            &mut c2f_trail,
            &mut c2f_candidates,
            0.45,
            0.90,
            1.2,
            0.8,
            0.5,
            128,
            48,
            2.0,
            0.5,
            0.35,
            1.0,
            64,
            0.25,
            1.5,
            &mut bloom,
            None,
        )
        .unwrap();
    assert_eq!(m_idx, 1);
    assert_eq!(m_id, 99);
    assert!(found_c2f);
    assert_eq!(sel_id, 1);
    assert_eq!(c2f_trail.cluster_id, 99);
    assert_eq!(c2f_trail.target_anchor_id, 990);
    assert_eq!(c2f_candidates[1].cluster_id, 99);
}
