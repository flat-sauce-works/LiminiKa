// name: src/core/src/phase_steering/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
#[allow(unused_imports)]
use super::{closed_loop::*, engine::*};
use crate::traits::aligned::{AlignedBuffer128, AlignedBuffer32, AlignedBuffer64};
use crate::traits::gateway::{ObjectiveFact, SubjectivePersona, SystemConstraint};
use crate::traits::primitives::branchless_sincos_q7;
use crate::traits::primitives::{
    ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION, ATTRACTOR_SLOT_KNOWLEDGE,
    ATTRACTOR_SLOT_PERSONA, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
};
use core::f32::consts::PI;

#[test]
fn test_phase_steering_engine_creation_aligned_dispatch_and_pod_layout() {
    assert!(PhaseSteeringEngine::new(128, 32).is_ok());
    assert!(PhaseSteeringEngine::new(127, 32).is_err());
    assert!(PhaseSteeringEngine::new(0, 32).is_err());
    assert!(PhaseSteeringEngine::new(128, 0).is_err());

    let def = PhaseSteeringEngine::default()
        .with_ripa_clamp_max_rad(0.18)
        .with_qdps_min_step_rad(0.01);
    assert_eq!(def.head_dim(), 128);
    assert_eq!(def.num_heads(), 32);
    assert!((def.ripa_clamp_max_rad() - 0.18).abs() < 1e-6);
    assert!((def.qdps_min_step_rad() - 0.01).abs() < 1e-6);
    assert_eq!(def.as_bytes().len(), 16);

    let eng_small = PhaseSteeringEngine::new(8, 2).unwrap();
    let mut q128 = AlignedBuffer128::new([
        1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
    ]);
    let p16 = AlignedBuffer16::new([64i8; 16]);
    eng_small
        .apply_phase_steering_aligned::<128>(
            &mut q128.as_aligned_slice_mut(),
            &p16.as_aligned_slice(),
        )
        .unwrap();
    assert!((q128.data[0] - 0.0).abs() < 1e-4);
    assert!((q128.data[1] - 1.0).abs() < 1e-4);
}

#[test]
fn test_qdps_filter_and_swar_scaling() {
    let engine = PhaseSteeringEngine::new(128, 16).unwrap();
    let mut raw_buf =
        AlignedBuffer16::new([1i8, 5, -2, 10, -128, 4, -1, 8, 6, -6, 12, -12, 0, 2, -3, 16]);
    let mut aligned = raw_buf.as_aligned_slice_mut();

    assert!(engine.qdps_filter_step(&mut aligned, 0.1).is_ok());
    let res = aligned.as_slice();
    assert_eq!(res[0], 0);
    assert_eq!(res[1], 5);
    assert_eq!(res[2], 0);
    assert_eq!(res[3], 10);
    assert_eq!(res[4], -128);
    assert_eq!(res[5], 0);
    assert_eq!(res[6], 0);
    assert_eq!(res[7], 8);
    assert_eq!(res[15], 16);

    assert!(engine.scale_phases_swar_u7(&mut aligned, 128).is_ok());
    assert_eq!(aligned.as_slice()[3], 10);
    assert!(engine.scale_phases_swar_u7(&mut aligned, 64).is_ok());
    assert_eq!(aligned.as_slice()[3], 5);
    assert_eq!(aligned.as_slice()[7], 4);
    assert_eq!(aligned.as_slice()[15], 8);
}

#[test]
fn test_lazy_unwrap_override_and_3way_qkv_steering() {
    let engine = PhaseSteeringEngine::new(4, 1).unwrap();
    let mut q_buf = AlignedBuffer32::new([1.0f32, 0.0, 0.0, 1.0]);
    let c_buf = AlignedBuffer32::new([PI * 0.5]);

    engine
        .lazy_unwrap_override(&mut q_buf.as_aligned_slice_mut(), &c_buf.as_aligned_slice())
        .unwrap();
    assert!((q_buf.data[0] - 0.0).abs() < 1e-4);
    assert!((q_buf.data[1] - 1.0).abs() < 1e-4);
    assert!((q_buf.data[2] - (-1.0)).abs() < 1e-4);
    assert!((q_buf.data[3] - 0.0).abs() < 1e-4);

    let mut q3 = AlignedBuffer32::new([1.0f32, 0.0, 2.0, 0.0]);
    let mut k3 = AlignedBuffer32::new([0.0f32, 1.0, 0.0, 2.0]);
    let mut v3 = AlignedBuffer32::new([1.0f32, 1.0, 1.0, 1.0]);
    let phases = AlignedBuffer16::new([64i8; 16]);

    engine
        .apply_phase_steering_fused_qkv(
            &mut q3.as_aligned_slice_mut(),
            &mut k3.as_aligned_slice_mut(),
            &mut v3.as_aligned_slice_mut(),
            &phases.as_aligned_slice(),
        )
        .unwrap();
    assert!((q3.data[0] - 0.0).abs() < 1e-4);
    assert!((q3.data[1] - 1.0).abs() < 1e-4);
    assert!((k3.data[0] - (-1.0)).abs() < 1e-4);
    assert!((k3.data[1] - 0.0).abs() < 1e-4);
    assert!((v3.data[0] - (-1.0)).abs() < 1e-4);
    assert!((v3.data[1] - 1.0).abs() < 1e-4);
}

#[test]
fn test_slerp_norm_guard_logit_shift_and_procrustes_feedback() {
    let engine = PhaseSteeringEngine::new(4, 2).unwrap();
    let mut t_buf = AlignedBuffer32::new([2.0f32; 8]);
    engine
        .slerp_norm_guard_stable(&mut t_buf.as_aligned_slice_mut(), 1.0, 1.0)
        .unwrap();
    let norm_sq: f32 = t_buf.data.iter().map(|v| v * v).sum();
    assert!((crate::traits::primitives::branchless_sqrt_f32(norm_sq) - 1.0).abs() < 1e-5);

    let mut logits = AlignedBuffer32::new([1.0f32; 8]);
    let mut phases_arr = [0i8; 16];
    phases_arr[0] = 32;
    phases_arr[1] = 64;
    let phases = AlignedBuffer16::new(phases_arr);
    engine
        .fused_logit_shift(
            &mut logits.as_aligned_slice_mut(),
            &phases.as_aligned_slice(),
        )
        .unwrap();
    let expected_shift = (48.0 * PI) / 128.0;
    for &val in &logits.data {
        assert!((val - (1.0 + expected_shift)).abs() < 1e-4);
    }

    let engine_1h = PhaseSteeringEngine::new(4, 1).unwrap();
    let mut q_fb = AlignedBuffer32::new([1.0f32, 0.0, 0.0, 1.0]);
    let tgt = AlignedBuffer32::new([0.0f32, 1.0, -1.0, 0.0]);
    let mut trail = gcso_pointer_trail_t::default();
    let fb_deltas = engine_1h
        .steer_trail_with_procrustes_feedback(
            &mut trail,
            &mut q_fb.as_aligned_slice_mut(),
            &tgt.as_aligned_slice(),
            128,
            0.01,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        )
        .unwrap();
    assert_eq!(fb_deltas.data[0], 64);
    assert_eq!(trail.accumulated_phase_delta[0], 64);
    assert!((q_fb.data[2] - 1.0).abs() > 1e-3);

    let engine_8h = PhaseSteeringEngine::new(16, 8).unwrap();
    let mut src_8h = AlignedBuffer32::new([0.0f32; 128]);
    for pair in src_8h.data.chunks_exact_mut(2) {
        pair[0] = 1.0;
        pair[1] = 0.0;
    }
    let mut tgt_8h = src_8h;
    let p_64 = AlignedBuffer16::new([64i8; 16]);
    engine_8h
        .apply_phase_steering(&mut tgt_8h.as_aligned_slice_mut(), &p_64.as_aligned_slice())
        .unwrap();
    let mut out_8h = AlignedBuffer16::new([0i8; 16]);
    engine_8h
        .compute_procrustes_phase_delta(
            &src_8h.as_aligned_slice(),
            &tgt_8h.as_aligned_slice(),
            &mut out_8h.as_aligned_slice_mut(),
        )
        .unwrap();
    for h in 0..8 {
        assert_eq!(out_8h.data[h], 64);
    }

    let (s_pi, c_pi) = branchless_sincos_q7(-128);
    assert!(s_pi.abs() < 1e-6);
    assert!((c_pi - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_seam_free_circular_unwrap_swarm_quorum_unified_field_limina_and_pspm() {
    let engine = PhaseSteeringEngine::new(4, 16).unwrap();
    let mut ctx_accum = AlignedBuffer16::new([120i8; 16]);
    let step_deltas = AlignedBuffer16::new([24i8; 16]);

    engine
        .accumulate_context_phases_circular_q7(
            &mut ctx_accum.as_aligned_slice_mut(),
            &step_deltas.as_aligned_slice(),
            0.01,
        )
        .unwrap();
    assert_eq!(ctx_accum.data[0], -112);

    let q_phases = AlignedBuffer16::new([-100i8; 16]);
    let mut q_tensor = AlignedBuffer32::new([1.0f32; 64]);
    let eff = engine
        .apply_seam_free_lazy_unwrap_q7(
            &mut q_tensor.as_aligned_slice_mut(),
            None,
            &q_phases.as_aligned_slice(),
            &ctx_accum.as_aligned_slice(),
            0.01,
            None,
        )
        .unwrap();
    assert_eq!(eff.data[0], 12);

    let ctx_antipodal = AlignedBuffer16::new([27i8; 16]);
    let eff_geo = engine
        .apply_seam_free_lazy_unwrap_q7(
            &mut q_tensor.as_aligned_slice_mut(),
            None,
            &q_phases.as_aligned_slice(),
            &ctx_antipodal.as_aligned_slice(),
            0.0,
            Some(DEFAULT_RIPA_CLAMP_MAX_RAD),
        )
        .unwrap();
    assert_eq!(eff_geo.data[0], -1);

    let ptr = gcso_tagged_ptr_t::pack_action(
        0x1000,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_EXPRESS_PHASE,
        0,
    );
    let mut q_swarm = AlignedBuffer32::new([1.0f32; 64]);
    let deltas_64 = AlignedBuffer16::new([64i8; 16]);
    engine
        .apply_swarm_quorum_gated_steering_safe(
            ptr,
            0b0000_0001,
            0,
            &deltas_64.as_aligned_slice(),
            &mut q_swarm.as_aligned_slice_mut(),
            None,
            0.20,
        )
        .unwrap();
    assert!((q_swarm.data[2] - 1.0).abs() > 1e-3);
    assert_eq!(&q_swarm.data[32..36], &[1.0f32; 4]);

    let mut trail = gcso_pointer_trail_t {
        current_ptr: gcso_tagged_ptr_t::pack_action(
            0x2000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_FULL_ORCHESTRATION,
            0,
        )
        .raw(),
        accumulated_phase_delta: [30i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let tgt_64 = AlignedBuffer16::new([32i8; MAX_ATTENTION_HEADS]);
    let mut q_closed = AlignedBuffer32::new([1.0f32; 64]);
    let report = engine
        .resolve_and_apply_swarm_phase_steering_safe(
            &mut trail,
            &tgt_64.as_aligned_slice(),
            128,
            4,
            6,
            0,
            0.10,
            0.15,
            None,
            2,
            128,
            0.01,
            0.20,
            &mut q_closed.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert_eq!(report.hot_path_resolved, 1);
    assert_eq!(report.resolved_cell_count, 8);
    assert_eq!(report.quorum_mask, 0xFF);

    let unified_anchors = [[-96i8; MAX_ATTENTION_HEADS], [32i8; MAX_ATTENTION_HEADS]];
    let unified_energies = [0.2f32, 3.0f32];
    let mut uf_trail = gcso_pointer_trail_t {
        current_ptr: ptr.raw(),
        accumulated_phase_delta: [30i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let uf_report = engine
        .resolve_unified_field_swarm_and_steer_safe(
            &mut uf_trail,
            &unified_anchors,
            &unified_energies,
            None,
            128,
            4,
            6,
            64,
            0.10,
            0.15,
            None,
            2,
            128,
            0.01,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            &mut q_closed.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert_eq!(uf_report.hot_path_resolved, 1);
    assert_eq!(uf_trail.target_anchor_id, 1);
    assert_eq!(uf_trail.cluster_id, 1);

    let mut next_mem_trail = gcso_pointer_trail_t {
        current_ptr: ptr.raw(),
        stigmergic_density: 1.0,
        attractor_pull_force: 1.0,
        accumulated_phase_delta: [36i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
    let (lim_coh, lim_lp, lim_rep) = engine
        .bridge_liminal_trail_and_steer_qk_safe(
            &uf_trail,
            &mut next_mem_trail,
            64,
            0.25,
            0.01,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            2.0,
            Some(&unified_anchors),
            Some(&mut attn_scores.as_aligned_slice_mut()),
            &mut q_closed.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert!(!lim_rep && lim_coh > 0.95 && lim_lp > -0.05);
    assert_eq!(next_mem_trail.target_anchor_id, 1);
    assert!(attn_scores.data[1] > 0.5 && attn_scores.data[0] < -0.5);

    let mut slot_bank =
        AlignedBuffer64::<gcso_q7_t, { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS }>::splat(32);
    {
        let mut bank_mut = slot_bank.as_aligned_slice_mut();
        let slot_p = bank_mut
            .slot_exact_mut::<MAX_ATTENTION_HEADS>(ATTRACTOR_SLOT_PERSONA as usize)
            .unwrap();
        slot_p[48..64].fill(112);
        let slot_k = bank_mut
            .slot_exact_mut::<MAX_ATTENTION_HEADS>(ATTRACTOR_SLOT_KNOWLEDGE as usize)
            .unwrap();
        slot_k.fill(32);
    }
    trail.accumulated_phase_delta = [30i8; MAX_ATTENTION_HEADS];
    trail.current_ptr = gcso_tagged_ptr_t::pack_action(
        0x3000,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_FULL_ORCHESTRATION,
        0,
    )
    .raw();
    let slot_weights = [1.0f32; MAX_ATTRACTOR_DOMAINS];
    let bank_report = engine
        .resolve_slot_bank_swarm_and_steer_safe(
            &mut trail,
            &slot_bank.relax_slice::<16>(),
            &slot_weights,
            DEFAULT_SLOT_COUPLING_MATRIX_8X8,
            None,
            64,
            4,
            7,
            64,
            0.10,
            0.15,
            None,
            2,
            128,
            0.01,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            &mut q_closed.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert!(bank_report.resolved_cell_count >= 7);

    let pa = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
    let slerped = engine
        .slerp_and_apply_phase_steering_safe(
            &pa.as_aligned_slice(),
            &tgt_64.as_aligned_slice(),
            64,
            0.01,
            0.20,
            &mut q_closed.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert_eq!(slerped.data[0], 16);

    let slerp_t0 = engine
        .slerp_and_apply_phase_steering_safe(
            &pa.as_aligned_slice(),
            &tgt_64.as_aligned_slice(),
            0,
            0.01,
            0.20,
            &mut q_closed.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert_eq!(slerp_t0.data[0], 0);
    let slerp_t128 = engine
        .slerp_and_apply_phase_steering_safe(
            &pa.as_aligned_slice(),
            &tgt_64.as_aligned_slice(),
            128,
            0.01,
            0.20,
            &mut q_closed.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert_eq!(slerp_t128.data[0], 32);

    let subhead_cfg = gcso_subhead_config_t {
        num_fact_heads: 4,
        num_logic_heads: 8,
        num_explore_heads: 4,
        fact_phase_gain: 1.0,
        logic_phase_gain: 1.0,
        explore_phase_gain: 1.1,
        ..Default::default()
    };
    let mut q_pspm = AlignedBuffer32::new([1.0f32; 64]);
    engine
        .dispatch_subhead_slot_ensemble_geodesic(
            &mut q_pspm.as_aligned_slice_mut(),
            &subhead_cfg,
            &slot_bank.relax_slice::<16>(),
            0,
            1,
            2,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            8,
            4,
        )
        .unwrap();
    assert!((q_pspm.data[2] - 1.0).abs() > 1e-3);

    let (coh, r_bar, mean_q7, kappa) = engine
        .evaluate_phase_coherence_and_resultant_64(
            &tgt_64.as_aligned_slice(),
            &tgt_64.as_aligned_slice(),
        )
        .unwrap();
    assert!((coh - 1.0).abs() < 1e-4);
    assert!((r_bar - 1.0).abs() < 1e-4);
    assert_eq!(mean_q7, 32);
    assert!(kappa > 10.0);
}

#[test]
fn test_blanket_bayesian_pointer_trail_selection_unified_pspm_and_gateway_crystallization() {
    let engine = PhaseSteeringEngine::new(4, 16).unwrap();
    let ptr = gcso_tagged_ptr_t::pack_action(
        0x4000,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_FULL_ORCHESTRATION,
        0,
    );

    let mut src_trail = gcso_pointer_trail_t {
        current_ptr: ptr.raw(),
        stigmergic_density: 1.5,
        attractor_pull_force: 2.0,
        target_anchor_id: 9,
        cluster_id: 9,
        accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };

    let cand_contradict = gcso_pointer_trail_t {
        current_ptr: 0x4100,
        stigmergic_density: 4.0,
        accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let cand_coherent = gcso_pointer_trail_t {
        current_ptr: 0x4200,
        stigmergic_density: 3.0,
        target_anchor_id: 19,
        accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let mut candidates = [cand_contradict, cand_coherent];
    let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
    let mut q_buf = AlignedBuffer32::new([1.0f32; 64]);
    let mut k_buf = AlignedBuffer32::new([0.5f32; 64]);

    let (sel_id, bayes_score, trans_coh, trans_lp, found) = engine
        .select_rewire_and_steer_bayesian_with_defaults(
            &mut src_trail,
            &mut candidates,
            2.0,
            0.5,
            0.25,
            1.0,
            64,
            0.25,
            Some(&mut attn_scores.as_aligned_slice_mut()),
            &mut q_buf.as_aligned_slice_mut(),
            Some(&mut k_buf.as_aligned_slice_mut()),
        )
        .unwrap();
    assert!(found);
    assert_eq!(sel_id, 1);
    assert_eq!(src_trail.linked_trail_id, 1);
    assert_eq!(src_trail.target_anchor_id, 19);
    assert_eq!(src_trail.cluster_id, 19);
    assert!(bayes_score > 0.5 && trans_coh > 0.95 && trans_lp > -0.05);
    assert!(attn_scores.data[1] > 1.0 && attn_scores.data[0] < -0.5);

    let unified_anchors = [[-96i8; MAX_ATTENTION_HEADS], [30i8; MAX_ATTENTION_HEADS]];
    let unified_energies = [0.1f32, 3.5f32];
    let subhead_cfg = gcso_subhead_config_t {
        num_fact_heads: 4,
        num_logic_heads: 8,
        num_explore_heads: 4,
        fact_phase_gain: 1.0,
        logic_phase_gain: 1.0,
        explore_phase_gain: 1.2,
        ..Default::default()
    };
    let mut branch_logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0, 0.0]);
    let (final_pull, mean_coh, inv_temp, dom_id) = engine
        .steer_unified_field_pspm_and_inv_temperature_safe(
            &mut src_trail,
            &unified_anchors,
            &unified_energies,
            None,
            Some((&subhead_cfg, 8i8)),
            Some((&mut branch_logits.as_aligned_slice_mut(), 1.0)),
            0.01,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            &mut q_buf.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert_eq!(dom_id, 1);
    assert!(final_pull > 7.0 && mean_coh > 0.40);
    assert!(inv_temp > 1.9);
    assert!(branch_logits.data[0] > 1.9);

    let sys_sig = SystemConstraint::new((), 0.20, 0.05, 0xFF);
    let fact_sig = ObjectiveFact::new(
        "RAG fact: zero-allocation Rust kernel",
        3.0,
        0x0024_0000,
        88,
    );
    let pers_sig = SubjectivePersona::new("Analytical co-architect persona", 2.0, 0x0000_9000, 12);
    let mut crystal =
        CrystallizedAttractorRecord::new_empty(88, ATTRACTOR_SLOT_KNOWLEDGE, 0x0024_0000);
    let mut gw_logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);

    let (gw_frame, gw_report, grav_mass) = engine
        .steer_and_crystallize_gateway_signal_safe(
            &(&sys_sig, &fact_sig, &pers_sig),
            &mut src_trail,
            &mut crystal,
            64,
            &mut q_buf.as_aligned_slice_mut(),
            Some(&mut k_buf.as_aligned_slice_mut()),
            Some(&mut gw_logits.as_aligned_slice_mut()),
        )
        .unwrap();
    assert_eq!(gw_frame.mapped_section_offset48, 0x0024_0000);
    assert_eq!(src_trail.target_anchor_id, 88);
    assert_eq!(gw_report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
    assert!(grav_mass > 0.0 && crystal.sample_count == 1 && crystal.verify_integrity());

    let scaled_cfg16 = gw_frame.to_subhead_config_for_heads(16);
    assert_eq!(
        scaled_cfg16.num_fact_heads + scaled_cfg16.num_logic_heads + scaled_cfg16.num_explore_heads,
        16
    );

    let mut cand_swarm = [cand_contradict, cand_coherent];
    let mut swarm_scores = AlignedBuffer32::new([0.0f32; 8]);
    let (_gw_f2, _gw_r2, sw_rep, sel_swarm_id, found_swarm) = engine
        .steer_gateway_swarm_bayesian_and_crystallize_safe(
            &(&sys_sig, &fact_sig, &pers_sig),
            &mut src_trail,
            Some(&mut cand_swarm),
            Some(&mut swarm_scores.as_aligned_slice_mut()),
            96,
            16,
            6,
            64,
            0.15,
            0.10,
            None,
            3,
            64,
            2.0,
            0.5,
            0.25,
            1.0,
            64,
            -0.25,
            4,
            &mut q_buf.as_aligned_slice_mut(),
            Some(&mut k_buf.as_aligned_slice_mut()),
            Some(&mut gw_logits.as_aligned_slice_mut()),
            Some(&mut crystal),
        )
        .unwrap();
    assert!(sw_rep.quorum_mask != 0);
    assert!(found_swarm);
    assert_eq!(sel_swarm_id, 1);
    assert_eq!(crystal.sample_count, 2);
    assert!(crystal.verify_integrity());

    let mut macro_centroids = [
        CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0),
        CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
    ];
    let fact_cluster_sig = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 3.0, 0x0010_0000, 101);
    let (_f_km1, _r_km1, c_idx1, c_id1, _c_coh1, c_mass1, spawned1) = engine
        .steer_and_kmeans_crystallize_with_defaults(
            &fact_cluster_sig,
            &mut src_trail,
            &mut macro_centroids,
            &mut q_buf.as_aligned_slice_mut(),
            None,
            None,
        )
        .unwrap();
    assert!(spawned1);
    assert_eq!(c_idx1, 0);
    assert_eq!(c_id1, 101);
    assert!(c_mass1 > 0.0);

    let persona_cluster_sig =
        SubjectivePersona::new([-84i8; MAX_ATTENTION_HEADS], 2.8, 0x0028_0000, 202);
    let (_f_km2, _r_km2, c_idx2, c_id2, _c_coh2, _c_mass2, spawned2) = engine
        .steer_and_kmeans_crystallize_with_defaults(
            &persona_cluster_sig,
            &mut src_trail,
            &mut macro_centroids,
            &mut q_buf.as_aligned_slice_mut(),
            None,
            None,
        )
        .unwrap();
    assert!(spawned2);
    assert_eq!(c_idx2, 1);
    assert_eq!(c_id2, 202);

    let mut c2f_trail = gcso_pointer_trail_t {
        accumulated_phase_delta: [-80i8; MAX_ATTENTION_HEADS],
        stigmergic_density: 1.5,
        attractor_pull_force: 1.5,
        ..Default::default()
    };
    let mut c2f_candidates = [
        gcso_pointer_trail_t {
            cluster_id: 101,
            target_anchor_id: 101,
            stigmergic_density: 2.0,
            accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        },
        gcso_pointer_trail_t {
            cluster_id: 202,
            target_anchor_id: 2020,
            stigmergic_density: 2.5,
            accumulated_phase_delta: [-82i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        },
    ];
    let mut c2f_scores = AlignedBuffer32::new([0.0f32; 8]);
    let (m_idx, m_id, m_coh, fine_sel_id, _fine_bs, _fine_lp, fine_found) = engine
        .route_coarse_to_fine_and_steer_bayesian_trails_safe(
            &macro_centroids,
            64,
            &mut c2f_trail,
            &mut c2f_candidates,
            2.0,
            0.5,
            0.35,
            1.0,
            64,
            0.25,
            0.01,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            Some(&mut c2f_scores.as_aligned_slice_mut()),
            &mut q_buf.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert_eq!(m_idx, 1);
    assert_eq!(m_id, 202);
    assert!(m_coh > 0.95);
    assert!(fine_found);
    assert_eq!(fine_sel_id, 1);
    assert_eq!(c2f_trail.cluster_id, 202);
    assert_eq!(c2f_trail.target_anchor_id, 2020);
    assert_eq!(c2f_candidates[1].cluster_id, 202);

    let mut c2f_pruned_trail = gcso_pointer_trail_t {
        accumulated_phase_delta: [-80i8; MAX_ATTENTION_HEADS],
        stigmergic_density: 1.5,
        attractor_pull_force: 1.5,
        ..Default::default()
    };
    let (mp_idx, mp_id, mp_coh, mp_sel_id, _mp_bs, _mp_lp, mp_found) = engine
        .route_coarse_to_fine_and_steer_bayesian_trails_safe(
            &macro_centroids,
            64,
            &mut c2f_pruned_trail,
            &mut c2f_candidates,
            2.0,
            0.5,
            0.35,
            1.0,
            64,
            0.25,
            0.01,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            None,
            &mut q_buf.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert_eq!(mp_idx, 1);
    assert_eq!(mp_id, 202);
    assert!(mp_coh > 0.95);
    assert!(mp_found);
    assert_eq!(mp_sel_id, 1);
    assert_eq!(c2f_pruned_trail.cluster_id, 202);

    let (ens_pull, ens_dom_id, _ens_clamp) = engine
        .steer_crystallized_ensemble_zero_copy_safe(
            &macro_centroids,
            &mut c2f_trail,
            0.01,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            &mut q_buf.as_aligned_slice_mut(),
            None,
        )
        .unwrap();
    assert_eq!(ens_dom_id, 202);
    assert!(ens_pull > 1.5);
}
