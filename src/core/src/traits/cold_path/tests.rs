// name: src/core/src/traits/cold_path/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::traits::primitives::pack_action_slot_tag;
use crate::abi::gcso_rank1_residual_descriptor_t;
use crate::traits::primitives::{
    with_slot_pair_coupled_symmetric, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_FULL_ORCHESTRATION, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_USER_CONTEXT,
};
use crate::traits::{AlignedBuffer16, AlignedBuffer32, AlignedBuffer64};
use core::f32::consts::PI;

#[test]
fn test_sheaf_cohomology_symmetric_degrees_dual_pairing_and_qdpt() {
    let degrees = AlignedBuffer32::new([2.0f32, 5.5, 3.0, 1.0, 4.0, 2.5, 1.5, 0.5]);
    let lambda_max = eval_gershgorin_lambda_max(&degrees.as_aligned_slice()).unwrap();
    assert!((lambda_max - 11.0).abs() < 1e-5);

    let mut m0 = gcso_paged_bitmask_t::default();
    let mut m1 = gcso_paged_bitmask_t::default();
    let mut m2 = gcso_paged_bitmask_t::default();
    m0.set_bit_branchless(1);
    m0.set_bit_branchless(2);
    m1.set_bit_branchless(1);
    m1.set_bit_branchless(2);
    m2.set_bit_branchless(10);

    let masks = [m0, m1, m2];
    let mut deg_out = AlignedBuffer32::new([0.0f32; 4]);
    let lmax_blocks =
        eval_stigmergic_block_degrees(&masks, &mut deg_out.as_aligned_slice_mut()).unwrap();
    assert!((deg_out.data[0] - 1.0).abs() < 1e-4);
    assert!((deg_out.data[1] - 1.0).abs() < 1e-4);
    assert_eq!(deg_out.data[2], 0.0);
    assert!((lmax_blocks - 2.0).abs() < 1e-4);

    let coeffs = compute_tikhonov_chebyshev_coeffs_deg3(lambda_max, 0.1).unwrap();
    let eig_in = AlignedBuffer32::new([0.0f32, 5.5, 11.0, 2.75, 1.0, 2.0, 3.0, 4.0, 8.0, 9.0]);
    let mut filt_out = AlignedBuffer32::new([0.0f32; 10]);
    eval_chebyshev_spectral_filter_deg3(
        &eig_in.as_aligned_slice(),
        &mut filt_out.as_aligned_slice_mut(),
        11.0,
        &coeffs,
    )
    .unwrap();
    assert!(filt_out.data[0] > filt_out.data[2]);
    assert!(filt_out.data[4] > filt_out.data[9]);

    let r = AlignedBuffer32::new([2.0f32, 2.0, 2.0, 2.0]);
    let p = AlignedBuffer32::new([0.5f32, 0.5, 0.5, 0.5]);
    let gamma = AlignedBuffer32::new([1.0f32, 1.0, 1.0, 1.0]);
    let (mean_obs, dual_pair, obstructed) = evaluate_hodge_obstruction_and_dual_pairing(
        &r.as_aligned_slice(),
        &p.as_aligned_slice(),
        &gamma.as_aligned_slice(),
        1.0,
    )
    .unwrap();
    assert!(obstructed);
    assert!((mean_obs - 2.25).abs() < 1e-5);
    assert!((dual_pair - 1.0).abs() < 1e-5);

    let base_ptr = gcso_tagged_ptr_t::pack_action(
        0x1000,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_EXPRESS_PHASE,
        10,
    );
    let (qdpt_lvl, _dv, cascaded_ptr) = resolve_dyadic_qdpt_and_cascade_ptr(
        base_ptr,
        &m0,
        &m1,
        0.05,
        0,
        0.1,
        &[0x100, 0x200, 0x300, 0x400],
        &[4, 8, 12, 16],
    )
    .unwrap();
    assert_eq!(qdpt_lvl, 0);
    assert_eq!(cascaded_ptr.unpack_addr(), 0x1100);
    assert_eq!(cascaded_ptr.unpack_phase(), 14);

    let t0 = gcso_pointer_trail_t {
        accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let t1 = gcso_pointer_trail_t {
        accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let t2 = gcso_pointer_trail_t {
        accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let mut chain = [t0, t1, t2];
    let mut bloom = ContrastivePhaseBloomFilter::new();

    let (bridged_lp, rep_cnt) =
        bridge_and_filter_liminal_trail_chain(&mut chain, 64, 0.25, &mut bloom).unwrap();
    assert_eq!(rep_cnt, 0);
    assert!(bridged_lp > -0.10);

    let (joint_lp, hol_obs, is_obs) =
        evaluate_trail_chain_cohomology_and_joint_log_prob(&chain, 0.20).unwrap();
    assert!(!is_obs && hol_obs < 0.05 && joint_lp > -0.05);
}

#[test]
fn test_l2p_svd_pprc_limina_replay_unified_field_and_entropy_tunneling() {
    struct MockColdSystem;
    impl KvKeyframeCache for MockColdSystem {
        fn seek_to_token(
            &mut self,
            token_index: u32,
            header_out: &mut gcso_kv_keyframe_header_t,
        ) -> GcsoResult<()> {
            header_out.token_index = token_index;
            Ok(())
        }
    }
    impl PersonaPatcher for MockColdSystem {
        fn apply_patch(&mut self, _patch_data: &[u8]) -> GcsoResult<()> {
            Ok(())
        }
    }
    impl EntropyEvaluator for MockColdSystem {
        fn eval_stateful(
            &mut self,
            token_z_score: f32,
            state_out: &mut gcso_edbc_state_t,
        ) -> GcsoResult<()> {
            state_out.moving_z_entropy = token_z_score;
            Ok(())
        }
    }
    impl EdbcBranchController for MockColdSystem {
        fn active_branch_mode(&self) -> u32 {
            1
        }
        fn set_bifurcation_threshold(&mut self, _threshold: f32) {}
        fn set_hysteresis_band(&mut self, _band: f32) {}
        fn set_tau_flux(&mut self, _tau_flux: f32) {}
    }
    impl AttractorField for MockColdSystem {
        fn anchor_count(&self) -> u32 {
            1
        }
        fn apply_evaporative_decay(&mut self) {}
        fn total_potential_energy(&self) -> f32 {
            2.0
        }
        fn pressure_ratio(&self) -> f32 {
            0.1
        }
        fn evaporate_weakest_anchor(&mut self) -> Option<u32> {
            None
        }
        fn add_system_prompt_anchor(&mut self, _prompt: &str, _weight: f32) -> GcsoResult<u32> {
            Ok(1)
        }
        fn add_embedding_anchor(
            &mut self,
            _embedding: &AlignedSlice32<'_, f32>,
            _anchor_type: gcso_anchor_type_t,
            _weight: f32,
        ) -> GcsoResult<u32> {
            Ok(1)
        }
        fn add_anchor(
            &mut self,
            _anchor_type: gcso_anchor_type_t,
            _vec: &AlignedSlice32<'_, f32>,
            _dim: usize,
        ) -> GcsoResult<u32> {
            Ok(1)
        }
        fn inject_phase_repulsion(
            &mut self,
            _repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
            _gain: f32,
        ) -> GcsoResult<()> {
            Ok(())
        }
        fn aggregate_bottom_up(&mut self) -> GcsoResult<u32> {
            Ok(1)
        }
        fn clear_anchors(&mut self) -> GcsoResult<()> {
            Ok(())
        }
    }
    impl MultiDomainAttractorSpace for MockColdSystem {
        fn advance_step(&mut self) {}
        fn set_domain_weight(&mut self, domain_id: u32, _weight: f32) -> GcsoResult<()> {
            if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            Ok(())
        }
        fn set_domain_enabled(&mut self, domain_id: u32, _enabled: bool) -> GcsoResult<()> {
            if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            Ok(())
        }
        fn find_dominant_domain(&self) -> (u32, f32) {
            (u32::from(ATTRACTOR_SLOT_PERSONA), 2.0)
        }
        fn slot_coupling_matrix(&self) -> u64 {
            let m = with_slot_pair_coupled_symmetric(
                0,
                u32::from(ATTRACTOR_SLOT_PERSONA),
                u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
                true,
            );
            with_slot_pair_coupled_symmetric(
                m,
                u32::from(ATTRACTOR_SLOT_PERSONA),
                u32::from(ATTRACTOR_SLOT_USER_CONTEXT),
                true,
            )
        }
        fn evaluate_slot_force(
            &self,
            domain_id: u32,
            _trail: &gcso_pointer_trail_t,
            slot_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        ) -> GcsoResult<f32> {
            match domain_id {
                1 | 2 | 4 => {
                    *slot_phase_out = [32i8; MAX_ATTENTION_HEADS];
                    Ok(2.0)
                }
                3 => {
                    *slot_phase_out = [-96i8; MAX_ATTENTION_HEADS];
                    Ok(1.5)
                }
                0..=7 => {
                    *slot_phase_out = [0i8; MAX_ATTENTION_HEADS];
                    Ok(0.0)
                }
                _ => Err(GCSO_ERROR_INVALID_ARGUMENT),
            }
        }
        fn evaluate_multi_domain_superposition(
            &self,
            trail: &gcso_pointer_trail_t,
            composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        ) -> f32 {
            self.evaluate_unified_superposition_from_working_bank(
                trail,
                0.35,
                0.50,
                composite_phase_out,
            )
            .unwrap_or(0.0)
        }
        fn clear_all_domains(&mut self) -> GcsoResult<()> {
            Ok(())
        }
    }

    let mut sys = MockColdSystem;
    let mut cfg = gcso_config_t::default();
    init_default_config(&mut cfg).unwrap();
    assert_eq!(validate_config_invariants(&cfg), Ok(()));
    assert_eq!(
        abi_version(),
        (
            crate::abi::GCSO_ABI_VERSION_MAJOR,
            crate::abi::GCSO_ABI_VERSION_MINOR,
            crate::abi::GCSO_ABI_VERSION_PATCH
        )
    );

    let hashed_p = sys.hash_prompt_to_phase_profile("LiminiKa Persona Anchor", 1);
    assert!(hashed_p.iter().any(|&x| x != 0));

    let lora_a = AlignedBuffer32::new([1.0f32; 16]);
    let lora_b = AlignedBuffer32::new([1.0f32; 16]);
    let mut rank1_desc = gcso_rank1_residual_descriptor_t::default();
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
    let mut header = gcso_kv_keyframe_header_t::default();
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
    let mut gop_hdr = gcso_kv_keyframe_header_t::default();
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

    let daes_net = gcso_ledger_slot_t {
        mode: 1,
        ..Default::default()
    };
    let src_phases = [28i8; MAX_ATTENTION_HEADS];
    let mut pkt_buf = AlignedBuffer64::<u8, 64>::zeroed();
    let crc_tag = encode_dnp_phase_packet(
        &daes_net,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_FULL_ORCHESTRATION,
        &src_phases,
        &mut pkt_buf.as_aligned_slice_mut(),
    )
    .unwrap();
    let mut decoded_phases = [0i8; MAX_ATTENTION_HEADS];
    let recovered_tag = decode_and_verify_dnp_packet(
        &daes_net,
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

    let mut trail = gcso_pointer_trail_t {
        current_ptr: gcso_tagged_ptr_t::pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        )
        .raw(),
        ..Default::default()
    };

    let rebalanced = sys
        .rebalance_working_lane_weights_from_trail(&trail)
        .unwrap();
    assert!(rebalanced[usize::from(ATTRACTOR_SLOT_PERSONA)] >= 1.0);

    let (final_pull, mean_coh, neighbors) = sys
        .steer_trail_by_coupled_neighborhood(0.5, 0.5, &mut trail)
        .unwrap();
    assert_eq!(neighbors, 2);
    assert!((mean_coh - 1.0).abs() < 1e-4);
    assert!((final_pull - 6.0).abs() < 0.2);
    assert_ne!(trail.flags & TRAIL_FLAG_SYNERGY_RESONANCE, 0);
}
