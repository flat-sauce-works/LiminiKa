// name: src/adapter/src/tests/adaptive_and_guards.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adaptive layer selection, two-tier Guard, and boundary tests.

use super::*;

#[test]
fn adaptive_layer_selection_and_ratio_bands_across_model_depths() {
    let dim = 8;
    for &(num_layers, expected_top_half, expected_band_start) in &[
        (24_usize, 0x00FF_F000_u64, 8_usize),
        (26_usize, 0x03FF_E000_u64, 9_usize),
        (28_usize, 0x0FFF_C000_u64, 9_usize),
        (32_usize, 0xFFFF_0000_u64, 11_usize),
        (40_usize, 0x0000_00FF_FFF0_0000_u64, 14_usize),
    ] {
        assert_eq!(top_half_layer_mask(num_layers), expected_top_half);
        let band = ratio_band_layer_mask(
            num_layers,
            DEFAULT_CANDIDATE_BAND_START_RATIO,
            DEFAULT_CANDIDATE_BAND_END_RATIO,
        );
        let expected_band =
            ((1_u64 << (num_layers - expected_band_start)) - 1) << expected_band_start;
        assert_eq!(band, expected_band, "ratio band mismatch at L={num_layers}");

        // Fallback to top_half when probes are empty or zero-variance.
        assert_eq!(
            select_adaptive_layer_mask(num_layers, &[], &[], AdaptiveLayerSelectConfig::default()),
            expected_top_half
        );
        let constant_entries: Vec<TokenSequenceEntry> = (0..3)
            .map(|i| TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: format!("Const {i}"),
                energy: 2.0,
                seq_len: 1,
                per_layer_token_qs: vec![vec![1.0_f32; dim]; num_layers],
                token_include_mask: None,
            })
            .collect();
        assert_eq!(
            select_adaptive_layer_mask(
                num_layers,
                &constant_entries,
                &[],
                AdaptiveLayerSelectConfig::default()
            ),
            expected_top_half
        );

        // Construct a contiguous block of `K = L / 2` high-gain, phase-coherent layers starting at
        // `expected_band_start` (distinct from `top_half`) while all other layers have high
        // background variance and low corpus discriminative variance.
        let k = num_layers / 2;
        let target_start = expected_band_start;
        let target_end = target_start + k;
        let mut expected_selected = 0_u64;
        for l in target_start..target_end {
            expected_selected |= 1_u64 << l;
        }
        assert_ne!(expected_selected, expected_top_half);

        let corpus_entries: Vec<TokenSequenceEntry> = (0..4)
            .map(|i| {
                let mut per_layer = Vec::with_capacity(num_layers);
                for l in 0..num_layers {
                    let mut v = vec![0.0_f32; dim];
                    if (target_start..target_end).contains(&l) {
                        // Coherent high-discriminative corpus direction across all target layers.
                        v[i % dim] = 1.0;
                    } else {
                        // Nearly identical vectors across entries -> tiny corpus variance.
                        v[0] = 1.0;
                        v[1 + (i % 3)] = 0.02;
                    }
                    per_layer.push(v);
                }
                TokenSequenceEntry {
                    role: AttractorRole::Fact,
                    text: format!("Fact {i}"),
                    energy: 2.0,
                    seq_len: 1,
                    per_layer_token_qs: per_layer,
                    token_include_mask: None,
                }
            })
            .collect();
        let prior_entries: Vec<TokenSequenceEntry> = (0..4)
            .map(|i| {
                let mut per_layer = Vec::with_capacity(num_layers);
                for l in 0..num_layers {
                    let mut v = vec![0.0_f32; dim];
                    if (target_start..target_end).contains(&l) {
                        // Low background variance on target layers -> high relative SNR.
                        v[7] = 1.0;
                        v[6] = (i as f32) * 0.01;
                    } else {
                        // High background variance on non-target layers -> low relative SNR.
                        v[i % dim] = 1.0;
                    }
                    per_layer.push(v);
                }
                TokenSequenceEntry {
                    role: AttractorRole::Persona,
                    text: format!("Prior {i}"),
                    energy: 1.0,
                    seq_len: 1,
                    per_layer_token_qs: per_layer,
                    token_include_mask: None,
                }
            })
            .collect();

        let sel1 = select_adaptive_layer_mask(
            num_layers,
            &corpus_entries,
            &prior_entries,
            AdaptiveLayerSelectConfig::default(),
        );
        let sel2 = select_adaptive_layer_mask(
            num_layers,
            &corpus_entries,
            &prior_entries,
            AdaptiveLayerSelectConfig::default(),
        );
        assert_eq!(sel1, sel2, "adaptive layer selection must be deterministic");
        assert_eq!(sel1.count_ones() as usize, k);
        assert_eq!(
            sel1, expected_selected,
            "expected high-gain block 0x{expected_selected:010X} at L={num_layers}, got 0x{sel1:010X}"
        );
        // Must also work cleanly under `--no-base-prior` (empty prior slice).
        let sel_no_prior = select_adaptive_layer_mask(
            num_layers,
            &corpus_entries,
            &[],
            AdaptiveLayerSelectConfig::default(),
        );
        assert_eq!(sel_no_prior, expected_selected);
    }

    // Mixed-script prior with a large between-script mean shift on upper layers must NOT inflate
    // `bg_var` and displace upper semantic layers, and CKA coherence must stay invariant to
    // per-layer head/coordinate permutations.
    let num_layers = 26_usize;
    let top_half = top_half_layer_mask(num_layers);
    let corpus_en: Vec<TokenSequenceEntry> = (0..6)
        .map(|i| {
            let mut per_layer = Vec::with_capacity(num_layers);
            for l in 0..num_layers {
                let mut v = vec![0.0_f32; dim];
                if l >= 13 {
                    // Permute coordinates per layer (`(i + l) % dim`) while keeping identical
                    // pairwise Gram structure across upper layers.
                    v[(i + l) % dim] = 1.0;
                } else {
                    v[0] = 1.0;
                    v[1 + (i % 3)] = 0.05;
                }
                per_layer.push(v);
            }
            TokenSequenceEntry {
                role: AttractorRole::Fact,
                text: format!("English domain fact {i}"),
                energy: 2.0,
                seq_len: 1,
                per_layer_token_qs: per_layer,
                token_include_mask: None,
            }
        })
        .collect();
    let mixed_prior: Vec<TokenSequenceEntry> = (0..8)
        .map(|i| {
            let is_ascii = i % 2 == 0;
            let mut per_layer = Vec::with_capacity(num_layers);
            for l in 0..num_layers {
                let mut v = vec![0.0_f32; dim];
                if l >= 13 {
                    // Huge between-script separation (+e0 vs -e0) but tiny within-script spread.
                    v[0] = if is_ascii { 1.0 } else { -1.0 };
                    v[1] = ((i / 2) as f32) * 0.01;
                } else {
                    // Genuine within-script variance on shallow layers.
                    v[(i / 2) % dim] = 1.0;
                }
                per_layer.push(v);
            }
            TokenSequenceEntry {
                role: AttractorRole::Persona,
                text: if is_ascii {
                    format!("Neutral English prompt {i}")
                } else {
                    format!(
                        "\u{4E2D}\u{7ACB}\u{7684}\u{306A}\u{65E5}\u{672C}\u{8A9E}\u{30D7}\u{30ED}\
                     \u{30FC}\u{30D6}{i}"
                    )
                },
                energy: 1.0,
                seq_len: 1,
                per_layer_token_qs: per_layer,
                token_include_mask: None,
            }
        })
        .collect();
    let sel_mixed = select_adaptive_layer_mask(
        num_layers,
        &corpus_en,
        &mixed_prior,
        AdaptiveLayerSelectConfig::default(),
    );
    assert_eq!(
        sel_mixed, top_half,
        "between-script mean shift in mixed prior must not displace upper layers: got 0x{sel_mixed:08X}"
    );
}

#[test]
fn compute_slot_masks_preserves_32l_exact_and_generalizes_arbitrary_depths() {
    // 32 layers: bit-exact head-4 + tail-4 parity (`0xF000_000F`).
    let (active32, pinned32, rev32) = ModelSteeringAdapter::compute_slot_masks(32);
    assert_eq!(active32, 0xFFFF_FFFF);
    assert_eq!(pinned32, 0xF000_000F);
    assert_eq!(rev32, 0x0FFF_FFF0);

    // 26 layers: `pin_width = 26 / 8 = 3` -> head 0..3 + tail 23..26 (6 pinned, 20 reversible).
    let (active26, pinned26, rev26) = ModelSteeringAdapter::compute_slot_masks(26);
    assert_eq!(active26, 0x03FF_FFFF);
    assert_eq!(pinned26, 0x0380_0007);
    assert_eq!(pinned26.count_ones(), 6);
    assert_eq!(rev26.count_ones(), 20);

    // 24 layers: `pin_width = 24 / 8 = 3` -> head 0..3 + tail 21..24 (6 pinned, 18 reversible).
    let (active24, pinned24, rev24) = ModelSteeringAdapter::compute_slot_masks(24);
    assert_eq!(active24, 0x00FF_FFFF);
    assert_eq!(pinned24, 0x00E0_0007);
    assert_eq!(pinned24.count_ones(), 6);
    assert_eq!(rev24.count_ones(), 18);

    // 40 layers: `pin_width = (40 / 8).clamp(2, 4) = 4` -> 8 pinned, 32 reversible.
    let (active40, pinned40, rev40) = ModelSteeringAdapter::compute_slot_masks(40);
    assert_eq!(active40, (1_u64 << 40) - 1);
    assert_eq!(pinned40, 0x0000_00F0_0000_000F);
    assert_eq!(rev40.count_ones(), 32);

    // Small depth `< 8`: safe 1+1 endpoint pin.
    let (active6, pinned6, rev6) = ModelSteeringAdapter::compute_slot_masks(6);
    assert_eq!(active6, 0x3F);
    assert_eq!(pinned6, 0x21);
    assert_eq!(rev6, 0x1E);
}

#[test]
fn turn_orchestrator_token_include_mask_mismatch_and_all_false_fail_closed() {
    use liminika_core::runtime::StandardPhaseKvGopCache;
    use liminika_core::stigmergy::StandardCognitiveBank;

    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 2;
    geom.num_q_heads = 4;
    geom.num_kv_heads = 2;
    geom.head_dim = 8;
    geom.rotary_dim = 8;
    geom.layer_rope_mask = 0x3;
    let mut adapter = ModelSteeringAdapter::new(geom).expect("init");
    let dim = adapter.geometry().q_rot_len();

    let mut q_a = vec![0.0_f32; 2 * dim];
    q_a[0] = 1.0;
    q_a[dim + 1] = 1.0;
    let mut q_b = vec![0.0_f32; 2 * dim];
    q_b[4] = 1.0;
    q_b[dim + 5] = 1.0;

    let entries = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Alpha anchor".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: vec![q_a.clone(); 2],
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Beta anchor".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: vec![q_b; 2],
            token_include_mask: None,
        },
    ];
    let compiled = CompiledTokenCorpus::compile(
        &adapter,
        &entries,
        SequenceActivationPooler::default(),
        2,
        1,
    )
    .expect("compile");

    let mut kv_cache = StandardPhaseKvGopCache::new();
    let kv_policy = AdaptiveThresholdPolicy::default();
    let query_qs = vec![q_a; 2];

    // Mismatched mask length must fail closed with Err (never panic).
    let mismatch_err = TurnOrchestrator::prepare_turn(
        &compiled,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput::<AdaptiveThresholdPolicy, StandardCognitiveBank> {
            query_token_qs: &query_qs,
            query_seq_len: 2,
            token_include_mask: Some(vec![true]),
            turn_index: 1,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None,
            layer_steer_mask_override: None,
        },
    );
    assert!(mismatch_err.is_err());

    // All-false mask must fail closed into unobserved LiminalExterior with steering disabled.
    let empty_plan = TurnOrchestrator::prepare_turn(
        &compiled,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput::<AdaptiveThresholdPolicy, StandardCognitiveBank> {
            query_token_qs: &query_qs,
            query_seq_len: 2,
            token_include_mask: Some(vec![false, false]),
            turn_index: 2,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None,
            layer_steer_mask_override: None,
        },
    )
    .expect("all-false prepare_turn");
    assert!(empty_plan.is_unobserved_lacuna());
    assert!(!empty_plan.steering_eligible());
    assert!(!empty_plan.has_self_ref_anchors());
    assert!(!empty_plan.is_guard_trap());
    assert!(!empty_plan.is_soft_guard_caution());
    assert!(empty_plan.top1_fact_score().is_finite());
}

#[test]
fn boundary_2d_classifies_unobserved_rescue_exterior_self_ref_and_no_prior_fallback() {
    let env = NullCalibrationEnvelope {
        sample_count: 24,
        corpus_offdiag_floor: 0.16,
        ood_null_mean: 0.11,
        ood_null_std: 0.04,
        ood_null_max: 0.21,
        ood_null_high: 0.19,
        calibrated_min_coherence: 0.19,
        ood_null_median_ortho_ratio: 0.45,
        token_explain_threshold: 0.30,
        ood_coh_median: 0.11,
        ood_coh_mad: 0.035,
        ood_unexp_median: 0.92,
        ood_unexp_mad: 0.05,
        ood_unexp_p15: 0.83,
    };
    let b2d = Boundary2DThresholds::from_null_calibration(&env, 0.19);
    assert!(b2d.is_calibrated);
    assert!((0.65..=0.90).contains(&b2d.u_foreign));
    assert!(b2d.u_semi >= 0.45 && b2d.u_semi <= b2d.u_foreign - 0.12);

    let base_in = Boundary2DInput {
        top1_coh: 0.34,
        top2_coh: 0.18,
        boundary_margin: 0.04,
        unexplained_token_ratio: 0.20,
        query_seq_len: 12,
        has_recalled_facts: true,
        has_top1_fact_candidate: true,
        repulse_guard: false,
        guard_conflict_margin: -0.25,
        is_sandbox_eligible: true,
        bridge_count: 3,
    };

    // 1. High S_2D + high top1_coh -> SelfRef.
    let self_dec = b2d.classify(&base_in);
    assert_eq!(self_dec.region, Boundary2DRegion::SelfRef);
    assert!(!self_dec.foreign_ood_tokens);
    assert!(self_dec.s_2d >= b2d.tau_self);

    // 2. High unexp + low z_c -> Unobserved (foreign_ood_tokens = true).
    let ood_dec = b2d.classify(&Boundary2DInput {
        top1_coh: 0.17,
        unexplained_token_ratio: 0.88,
        ..base_in
    });
    assert_eq!(ood_dec.region, Boundary2DRegion::Unobserved);
    assert!(ood_dec.foreign_ood_tokens);

    // 3. High `unexp >= u_foreign` (even with strong `top1_coh` / `z_c`) must NOT bypass foreign-OOD
    //    gating via a 1D loophole into either `SelfRef` or `Exterior`.
    let high_unexp_ood = b2d.classify(&Boundary2DInput {
        top1_coh: 0.51,
        unexplained_token_ratio: 0.98,
        ..base_in
    });
    assert_eq!(high_unexp_ood.region, Boundary2DRegion::Unobserved);
    assert!(high_unexp_ood.foreign_ood_tokens);
    let boundary_unexp_ood = b2d.classify(&Boundary2DInput {
        top1_coh: 0.42,
        unexplained_token_ratio: b2d.u_foreign,
        ..base_in
    });
    assert_eq!(boundary_unexp_ood.region, Boundary2DRegion::Unobserved);
    assert!(boundary_unexp_ood.foreign_ood_tokens);

    // 3b. Sub-threshold noise (`top1_coh < calibrated_min_coherence`) must never promote into
    //     `Exterior` even when `unexp` falls inside the semi-known band.
    let sub_floor_noise = b2d.classify(&Boundary2DInput {
        top1_coh: b2d.calibrated_min_coherence * 0.95,
        top2_coh: 0.05,
        unexplained_token_ratio: b2d.u_semi + 0.02,
        has_recalled_facts: false,
        ..base_in
    });
    assert_eq!(sub_floor_noise.region, Boundary2DRegion::Unobserved);
    assert!(!sub_floor_noise.stage_top1_candidate_to_exterior);

    // 4. Narrow OOD median near 0 with floored MAD must NOT falsely rescue a query barely at `min_coh`.
    let narrow_env = NullCalibrationEnvelope {
        ood_coh_median: 0.0,
        ood_coh_mad: 0.025,
        calibrated_min_coherence: 0.15,
        ood_null_high: 0.15,
        ood_unexp_median: 1.0,
        ood_unexp_mad: 0.04,
        ood_unexp_p15: 0.82,
        ..env
    };
    let narrow_b2d = Boundary2DThresholds::from_null_calibration(&narrow_env, 0.15);
    let borderline_ood = narrow_b2d.classify(&Boundary2DInput {
        top1_coh: 0.16,
        unexplained_token_ratio: 0.95,
        ..base_in
    });
    assert_eq!(borderline_ood.region, Boundary2DRegion::Unobserved);
    assert!(borderline_ood.foreign_ood_tokens);

    // 5. Semi-known `unexp in [u_semi, u_foreign)` + `bridge_count >= 2` + sandbox -> Exterior.
    let ext_dec = b2d.classify(&Boundary2DInput {
        top1_coh: 0.28,
        top2_coh: 0.24,
        unexplained_token_ratio: b2d.u_semi + 0.03,
        ..base_in
    });
    assert_eq!(ext_dec.region, Boundary2DRegion::Exterior);
    assert!(!ext_dec.stage_top1_candidate_to_exterior);

    // Sharp unimodal lock adds `+0.10` margin to `u_semi`, protecting focused SelfRef hits.
    let locked_self = b2d.classify(&Boundary2DInput {
        top1_coh: 0.38,
        top2_coh: 0.15,
        unexplained_token_ratio: b2d.u_semi + 0.03,
        ..base_in
    });
    assert_eq!(locked_self.region, Boundary2DRegion::SelfRef);

    // 6. `--no-base-prior` exact 1D compatibility fallback.
    let fb = Boundary2DThresholds::fallback_without_prior(0.25);
    assert!(!fb.is_calibrated);
    assert!((fb.u_foreign - 0.80).abs() < 1e-6);
    assert!((fb.u_semi - 0.59).abs() < 1e-6);
    assert!((fb.token_explain_threshold - 0.40).abs() < 1e-6);
    assert!((fb.coherence_scale - 1.0).abs() < 1e-6);
    let fb_ood = fb.classify(&Boundary2DInput {
        top1_coh: 0.45,
        unexplained_token_ratio: 0.80,
        ..base_in
    });
    assert_eq!(fb_ood.region, Boundary2DRegion::Unobserved);
    assert!(fb_ood.foreign_ood_tokens);
    let fb_ext = fb.classify(&Boundary2DInput {
        top1_coh: 0.45,
        top2_coh: 0.42,
        unexplained_token_ratio: 0.62,
        ..base_in
    });
    assert_eq!(fb_ext.region, Boundary2DRegion::Exterior);
}

#[test]
fn coalition_thresholds_scale_relative_to_coherence_dispersion_and_calibrated_unexp() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let len = adapter.geometry().q_rot_len();
    let num_layers = adapter.geometry().num_layers;
    let centroid = vec![0.0_f32; len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); num_layers];

    let mut f_hub = vec![0.0_f32; len];
    let mut f_peer = vec![0.0_f32; len];
    let mut f_a = vec![0.0_f32; len];
    let mut f_b = vec![0.0_f32; len];
    f_hub[0] = 0.85;
    f_hub[1] = 0.52;
    f_peer[0] = 0.85;
    f_peer[4] = 0.52;
    f_a[2] = 1.0;
    f_b[3] = 1.0;

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Overview hub",
            &vec![f_hub.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("hub");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Shared sibling",
            &vec![f_peer.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("peer");
    let id_a = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Low-dispersion specialist A",
            &vec![f_a.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("a");
    let id_b = corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Low-dispersion specialist B",
            &vec![f_b.as_slice(); num_layers],
            &refs,
            1.5,
        )
        .expect("b");

    let mut query = vec![0.0_f32; len];
    query[0] = 0.85;
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

    // Deposit amplitudes sit in a low-dispersion regime (`joint_excl ~ 0.032 < 0.040`,
    // `second_contrib ~ 0.013 < 0.015`) that fails fixed unscaled floors (`scale = 1.0`)
    // but passes cleanly when scaled down to `coherence_scale = 0.60`.
    let low_disp_obs = LocalChannelObservation {
        combined_scores: vec![0.02, 0.005, 0.015, 0.015],
        q_to_d_scores: vec![0.02, 0.005, 0.016, 0.016],
        d_to_q_scores: vec![0.005, -0.005, 0.012, 0.012],
        token_deposits: vec![
            TokenResonanceDeposit {
                disc_weight: 0.45,
                entry_excess: vec![-0.01, -0.01, 0.034, -0.01],
                token_phase_words: [0; 8],
            },
            TokenResonanceDeposit {
                disc_weight: 0.45,
                entry_excess: vec![-0.01, -0.01, -0.01, 0.032],
                token_phase_words: [0; 8],
            },
        ],
        unexplained_token_ratio: 0.15,
        unexplained_corpus_token_ratio: 0.15,
        query_seq_len: 12,
    };

    // Uncalibrated (`scale = 1.0`): fixed legacy floor `0.040` does not link specialists (id_a, id_b).
    let rec_unscaled = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(low_disp_obs.clone()),
        )
        .expect("unscaled recall");
    let unscaled_ids: Vec<u32> = rec_unscaled
        .recalled_facts
        .iter()
        .map(|(e, _)| e.anchor_id)
        .collect();
    assert!(
        !(unscaled_ids.contains(&id_a) && unscaled_ids.contains(&id_b)),
        "unscaled legacy floor should not decompose sub-0.040 deposits: {unscaled_ids:?}"
    );

    // Calibrated low-dispersion envelope (`ood_coh_mad = 0.025 -> coherence_scale = 0.60`):
    // relative-scaled coalition floors link BOTH low-dispersion specialists cleanly.
    let low_disp_env = NullCalibrationEnvelope {
        sample_count: 16,
        corpus_offdiag_floor: 0.16,
        ood_null_mean: 0.08,
        ood_null_std: 0.03,
        ood_null_max: 0.15,
        ood_null_high: 0.18,
        calibrated_min_coherence: 0.20,
        ood_null_median_ortho_ratio: 0.50,
        token_explain_threshold: 0.28,
        ood_coh_median: 0.08,
        ood_coh_mad: 0.025,
        ood_unexp_median: 0.90,
        ood_unexp_mad: 0.05,
        ood_unexp_p15: 0.80,
    };
    let b2d = Boundary2DThresholds::from_null_calibration(&low_disp_env, 0.20);
    let ct = CoalitionThresholds::from_boundary_2d(&b2d);
    assert!((ct.coherence_scale - 0.60).abs() < 1e-6);
    assert!((ct.u_coalition_cutoff - (b2d.u_foreign - 0.08).clamp(0.55, 0.82)).abs() < 1e-6);

    corpus.set_null_calibration(Some(low_disp_env));
    let rec_scaled = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(low_disp_obs.clone()),
        )
        .expect("scaled recall");
    let scaled_ids: Vec<u32> = rec_scaled
        .recalled_facts
        .iter()
        .map(|(e, _)| e.anchor_id)
        .collect();
    assert!(
        scaled_ids.contains(&id_a) && scaled_ids.contains(&id_b),
        "relative-scaled coalition should link both specialists {id_a},{id_b}, got {scaled_ids:?}"
    );

    // Above calibrated `u_coalition_cutoff`, coalition must be blocked even on low-dispersion scale.
    let high_unexp_obs = LocalChannelObservation {
        unexplained_token_ratio: ct.u_coalition_cutoff + 0.02,
        ..low_disp_obs
    };
    let rec_blocked = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &vec![query.as_slice(); num_layers],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(high_unexp_obs),
        )
        .expect("blocked recall");
    let blocked_ids: Vec<u32> = rec_blocked
        .recalled_facts
        .iter()
        .map(|(e, _)| e.anchor_id)
        .collect();
    assert!(
        !(blocked_ids.contains(&id_a) && blocked_ids.contains(&id_b)),
        "high unexplained_token_ratio >= u_coalition_cutoff must block coalition: {blocked_ids:?}"
    );
}

#[test]
fn two_tier_guard_hard_block_vs_soft_caution_vs_non_guard_preserves_single_negation() {
    use liminika_core::runtime::{
        AdaptiveThresholdPolicy, ResonantSignalSlice, StandardPhaseKvGopCache,
    };
    use liminika_core::stigmergy::StandardCognitiveBank;

    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 4;
    geom.layer_rope_mask = 0xF;
    let mut adapter = ModelSteeringAdapter::new(geom).expect("init adapter");
    let mut kv_cache = StandardPhaseKvGopCache::new();
    let kv_policy = AdaptiveThresholdPolicy::default();
    let baseline_pins = adapter.envelope().baseline_pinned_compute_mask;

    struct StaticGuardSource(OwnedAttractorRecall);
    impl AttractorRecallSource for StaticGuardSource {
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

    let fact_q7 = [24_i8; MAX_ATTENTION_HEADS];
    let guard_q7 = [60_i8; MAX_ATTENTION_HEADS];
    let fact_rec = RecalledAnchorRecord {
        anchor_id: 11,
        role: AttractorRole::Fact,
        text: "Grounded factual anchor".to_string(),
        per_layer_q7: vec![fact_q7; 4],
        energy: 2.2,
        score: 0.42,
    };
    let guard_rec = RecalledAnchorRecord {
        anchor_id: 91,
        role: AttractorRole::Guard,
        text: "Contrastive guard pole".to_string(),
        per_layer_q7: vec![guard_q7; 4],
        energy: 2.0,
        score: 0.45,
    };
    let fact_frame = ObjectiveFact::new(fact_q7, 2.2, 3, 11).synthesize_gateway_frame();

    // 1. Layer 1 (Hard Block): margin > 0.05 + RepulsionEscape -> full `+R`, `is_guard_trap = true`,
    //    `is_soft_guard_caution = false`, `guard_turn = true` (all active compute slots pinned).
    let hard_src = StaticGuardSource(OwnedAttractorRecall {
        lacuna_state: LacunaBoundaryState::LiminalExterior,
        top1_fact_score: 0.25,
        top2_fact_score: 0.10,
        top1_persona_score: 0.05,
        min_local_lacuna_depth: 0.10,
        guard_conflict_margin: 0.22,
        action_decision: StigmergicActionDecision::RepulsionEscape,
        gateway_frame: GuardPhaseAnchor::new(guard_q7, 0.77, 0.50).synthesize_gateway_frame(),
        telemetry: StratumProbeTelemetry {
            guard_hard_margin: 0.05,
            is_soft_guard_caution: false,
            ..StratumProbeTelemetry::default()
        },
        recalled_facts: Vec::new(),
        recalled_personas: Vec::new(),
        top_guard: Some(guard_rec.clone()),
        lacuna_partition: LacunaPartition8::default(),
        exterior_sandbox_facts: Vec::new(),
        is_known_unknown_exterior: false,
    });
    let hard_plan = TurnOrchestrator::prepare_turn(
        &hard_src,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput::<AdaptiveThresholdPolicy, StandardCognitiveBank> {
            query_token_qs: &[],
            query_seq_len: 4,
            token_include_mask: None,
            turn_index: 1,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None,
            layer_steer_mask_override: None,
        },
    )
    .expect("hard guard plan");
    assert!(hard_plan.is_guard_trap());
    assert!(!hard_plan.is_soft_guard_caution());
    assert!(!hard_plan.has_self_ref_anchors());
    assert_eq!(
        adapter.envelope().pinned_compute_mask,
        adapter.compute_pool().active_mask,
        "Layer 1 hard-block must engage guard_turn pin lock"
    );
    let hard_ga = hard_plan.guard_anchors_per_layer()[0][0];
    assert_eq!(hard_ga.phase_deltas, guard_q7);
    assert!((0.55..=1.0).contains(&hard_ga.cohomology_obstruction));
    assert_eq!(hard_ga.obstruction_threshold, 0.50);

    // 2. Layer 2 (Soft Caution): `0.0 < margin (0.015) <= hard_margin (0.05)` -> Fact injection
    //    preserved, adapter-attenuated `+R_soft`, `obstruction in 0.52..=0.55`, `is_guard_trap = false`,
    //    `is_soft_guard_caution = true`, `guard_turn = false` (baseline pins preserved), and
    //    `ReflectionGateInput` triggers contextual audit even below `guard_trigger_margin (0.02)`.
    let soft_src = StaticGuardSource(OwnedAttractorRecall {
        lacuna_state: LacunaBoundaryState::InsideBasin,
        top1_fact_score: 0.42,
        top2_fact_score: 0.20,
        top1_persona_score: 0.10,
        min_local_lacuna_depth: 0.0,
        guard_conflict_margin: 0.015,
        action_decision: StigmergicActionDecision::ExpressPhaseSteer,
        gateway_frame: fact_frame,
        telemetry: StratumProbeTelemetry {
            guard_hard_margin: 0.05,
            is_soft_guard_caution: true,
            ..StratumProbeTelemetry::default()
        },
        recalled_facts: vec![fact_rec.clone()],
        recalled_personas: Vec::new(),
        top_guard: Some(guard_rec.clone()),
        lacuna_partition: LacunaPartition8::default(),
        exterior_sandbox_facts: Vec::new(),
        is_known_unknown_exterior: false,
    });
    let soft_plan = TurnOrchestrator::prepare_turn(
        &soft_src,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput::<AdaptiveThresholdPolicy, StandardCognitiveBank> {
            query_token_qs: &[],
            query_seq_len: 4,
            token_include_mask: None,
            turn_index: 2,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None,
            layer_steer_mask_override: None,
        },
    )
    .expect("soft guard plan");
    assert!(!soft_plan.is_guard_trap());
    assert!(soft_plan.is_soft_guard_caution());
    assert!(soft_plan.has_self_ref_anchors());
    assert!(soft_plan.steering_eligible());
    assert_eq!(
        adapter.envelope().pinned_compute_mask,
        baseline_pins,
        "Layer 2 soft-caution must keep guard_turn = false without locking bypass pins"
    );
    let soft_ga = soft_plan.guard_anchors_per_layer()[0][0];
    assert!((0.52..=0.55).contains(&soft_ga.cohomology_obstruction));
    assert_eq!(soft_ga.obstruction_threshold, 0.50);
    assert!(
        soft_ga.phase_deltas.iter().all(|&v| v > 0 && v < 60),
        "soft Guard anchor must carry attenuated positive +R_soft before core inversion: {:?}",
        &soft_ga.phase_deltas[..8]
    );
    let soft_gate = soft_plan.reflection_gate_input(0.20);
    assert!(soft_gate.is_soft_guard_caution);
    assert!(
        ManifoldReflectionEvaluator.should_trigger_reflection(&soft_gate),
        "Layer 2 soft-caution must trigger contextual reflection even below guard_trigger_margin"
    );

    // Verify `liminika-core` negates both `+R` and `+R_soft` exactly once (`-R` / `-R_soft`),
    // preserving anti-phase direction while strictly reducing soft repulsion amplitude.
    let empty_facts: [ObjectiveFact<[i8; 64]>; 0] = [];
    let empty_personas: [SubjectivePersona<[i8; 64]>; 0] = [];
    let probe = [60_i8; 24];
    let hard_slice = [hard_ga];
    let soft_slice = [soft_ga];
    let sys_res = *adapter.system_constraint();
    let delta = *adapter.runtime_context().last_boundary_delta();
    let horizon = *adapter.last_horizon();
    let overlap = *adapter.last_overlap_report();
    let mut stack_h = *adapter.core_layer_stack();
    let (hard_core_frame, _) = (
        sys_res,
        ResonantSignalSlice::from_q7_probe(&empty_facts, &probe),
        ResonantSignalSlice::from_q7_probe(&empty_personas, &probe),
        ResonantSignalSlice::from_q7_probe(&hard_slice, &probe),
    )
        .synthesize_gateway_with_core_layers(&mut stack_h, &delta, &horizon, &overlap);
    let mut stack_s = *adapter.core_layer_stack();
    let (soft_core_frame, _) = (
        sys_res,
        ResonantSignalSlice::from_q7_probe(&empty_facts, &probe),
        ResonantSignalSlice::from_q7_probe(&empty_personas, &probe),
        ResonantSignalSlice::from_q7_probe(&soft_slice, &probe),
    )
        .synthesize_gateway_with_core_layers(&mut stack_s, &delta, &horizon, &overlap);
    assert!(hard_core_frame.is_repulsed());
    assert!(soft_core_frame.is_repulsed());
    let dot_hard: i64 = guard_q7
        .iter()
        .zip(hard_core_frame.phase_deltas.iter())
        .map(|(&g, &f)| i64::from(g) * i64::from(f))
        .sum();
    let dot_soft: i64 = guard_q7
        .iter()
        .zip(soft_core_frame.phase_deltas.iter())
        .map(|(&g, &f)| i64::from(g) * i64::from(f))
        .sum();
    let l1_hard: i64 = hard_core_frame
        .phase_deltas
        .iter()
        .map(|&v| i64::from(v).abs())
        .sum();
    let l1_soft: i64 = soft_core_frame
        .phase_deltas
        .iter()
        .map(|&v| i64::from(v).abs())
        .sum();
    assert!(dot_hard < 0, "hard core frame must be anti-phase -R");
    assert!(dot_soft < 0, "soft core frame must be anti-phase -R_soft");
    assert!(
        l1_soft > 0 && l1_soft < l1_hard,
        "soft anti-phase amplitude ({l1_soft}) must be strictly attenuated below hard ({l1_hard})"
    );

    // 3. Non-Guard region (`guard_conflict_margin <= 0.0`): zero GuardPhaseAnchor, zero misfire.
    let calm_src = StaticGuardSource(OwnedAttractorRecall {
        lacuna_state: LacunaBoundaryState::InsideBasin,
        top1_fact_score: 0.55,
        top2_fact_score: 0.20,
        top1_persona_score: 0.15,
        min_local_lacuna_depth: 0.0,
        guard_conflict_margin: -0.10,
        action_decision: StigmergicActionDecision::FullOrchestration,
        gateway_frame: fact_frame,
        telemetry: StratumProbeTelemetry::default(),
        recalled_facts: vec![fact_rec],
        recalled_personas: Vec::new(),
        top_guard: Some(guard_rec),
        lacuna_partition: LacunaPartition8::default(),
        exterior_sandbox_facts: Vec::new(),
        is_known_unknown_exterior: false,
    });
    let calm_plan = TurnOrchestrator::prepare_turn(
        &calm_src,
        &mut adapter,
        &mut kv_cache,
        &TurnPrepareInput::<AdaptiveThresholdPolicy, StandardCognitiveBank> {
            query_token_qs: &[],
            query_seq_len: 4,
            token_include_mask: None,
            turn_index: 3,
            kv_policy: &kv_policy,
            kv_min_resonance_override: None,
            persona_bank: None,
            layer_steer_mask_override: None,
        },
    )
    .expect("calm plan");
    assert!(!calm_plan.is_guard_trap());
    assert!(!calm_plan.is_soft_guard_caution());
    assert!(
        calm_plan
            .guard_anchors_per_layer()
            .iter()
            .all(Vec::is_empty),
        "non-Guard Fact turn must emit zero GuardPhaseAnchors"
    );

    // 4. End-to-end `TextAttractorCorpus` recall verification: when a Guard slightly leads a Fact
    //    within `(0.0, hard_margin]` (`max_guard >= guard_floor`), `TextAttractorCorpus` preserves
    //    Fact injection, avoids `RepulsionEscape`, and marks `is_soft_guard_caution = true`.
    let q_len = adapter.geometry().q_rot_len();
    let centroid = vec![0.0_f32; q_len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); 4];
    let mut f_vec = vec![0.0_f32; q_len];
    let mut g_vec = vec![0.0_f32; q_len];
    f_vec[0] = 1.0;
    g_vec[1] = 1.0;
    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "E2E Fact",
            &[f_vec.as_slice(); 4],
            &refs,
            2.0,
        )
        .expect("e2e fact");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Guard,
            "E2E Guard",
            &[g_vec.as_slice(); 4],
            &refs,
            2.0,
        )
        .expect("e2e guard");
    let mut border_q = vec![0.0_f32; q_len];
    border_q[0] = 0.695;
    border_q[1] = 0.719;
    let e2e_policy = RecallPolicy {
        min_coherence: 0.25,
        boundary_margin: 0.03,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };
    let e2e_rec = corpus
        .recall_for_query(&adapter, &[border_q.as_slice(); 4], &refs, &e2e_policy)
        .expect("e2e soft guard recall");
    assert!(
        e2e_rec.guard_conflict_margin > 0.0 && e2e_rec.guard_conflict_margin <= 0.05,
        "expected boundary margin in (0, 0.05], got {}",
        e2e_rec.guard_conflict_margin
    );
    assert_ne!(
        e2e_rec.action_decision,
        StigmergicActionDecision::RepulsionEscape
    );
    assert!(e2e_rec.telemetry.is_soft_guard_caution);
    assert!(
        !e2e_rec.recalled_facts.is_empty(),
        "Layer 2 soft-caution must preserve Fact recall through TextAttractorCorpus"
    );
}

#[test]
fn recall_envelope_bounds_invariants_and_viability_ceilings() {
    let std_bounds = RecallEnvelopeBounds::standard();
    assert_eq!(std_bounds, RecallEnvelopeBounds::default());
    assert!(std_bounds.validate_invariants());
    assert!((std_bounds.ood_null_coherence_ceiling - 0.50).abs() < 1e-6);
    assert!((std_bounds.u_foreign_ceiling - 0.80).abs() < 1e-6);

    // Violated invariants must fail validation cleanly.
    let bad_inverted = RecallEnvelopeBounds {
        ood_null_coherence_floor: 0.55,
        ..std_bounds
    };
    assert!(!bad_inverted.validate_invariants());
    let bad_nan = RecallEnvelopeBounds {
        u_foreign_ceiling: f32::NAN,
        ..std_bounds
    };
    assert!(!bad_nan.validate_invariants());
    let bad_gap = RecallEnvelopeBounds {
        u_semi_floor: 0.60,
        u_semi_foreign_min_gap: 0.12,
        u_foreign_floor: 0.65,
        ..std_bounds
    };
    assert!(!bad_gap.validate_invariants());

    // `Boundary2DThresholds` respects `u_foreign_ceiling = 0.80` even when held-out OOD
    // probes saturate near `1.0`, while still allowing Stage 2 tightening down to `0.65`.
    let high_unexp_env = NullCalibrationEnvelope {
        sample_count: 24,
        corpus_offdiag_floor: 0.18,
        ood_null_mean: 0.22,
        ood_null_std: 0.08,
        ood_null_max: 0.43,
        ood_null_high: 0.42,
        calibrated_min_coherence: 0.42,
        ood_null_median_ortho_ratio: 0.85,
        token_explain_threshold: 0.36,
        ood_coh_median: 0.24,
        ood_coh_mad: 0.06,
        ood_unexp_median: 0.96,
        ood_unexp_mad: 0.04,
        ood_unexp_p15: 0.88,
    };
    let b2d_hi = Boundary2DThresholds::from_null_calibration(&high_unexp_env, 0.42);
    assert!((b2d_hi.u_foreign - std_bounds.u_foreign_ceiling).abs() < 1e-6);
    assert!(b2d_hi.u_semi <= b2d_hi.u_foreign - std_bounds.u_semi_foreign_min_gap + 1e-6);

    let low_unexp_env = NullCalibrationEnvelope {
        ood_unexp_median: 0.72,
        ood_unexp_p15: 0.68,
        ..high_unexp_env
    };
    let b2d_lo = Boundary2DThresholds::from_null_calibration(&low_unexp_env, 0.42);
    assert!(
        (std_bounds.u_foreign_floor..=0.69).contains(&b2d_lo.u_foreign),
        "Stage 2 calibration must tighten u_foreign downward cleanly: {}",
        b2d_lo.u_foreign
    );
}

#[test]
fn complete_foreign_ood_suppresses_guard_layer1_and_layer2_while_preserving_true_guard() {
    use liminika_core::runtime::{AdaptiveThresholdPolicy, StandardPhaseKvGopCache};
    use liminika_core::stigmergy::StandardCognitiveBank;

    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 4;
    geom.layer_rope_mask = 0xF;
    let mut adapter = ModelSteeringAdapter::new(geom).expect("init adapter");
    let q_len = adapter.geometry().q_rot_len();
    let centroid = vec![0.0_f32; q_len];
    let refs: Vec<&[f32]> = vec![centroid.as_slice(); 4];

    let mut f_vec = vec![0.0_f32; q_len];
    let mut g_vec = vec![0.0_f32; q_len];
    f_vec[0] = 1.0;
    g_vec[1] = 1.0;

    let mut corpus = TextAttractorCorpus::new();
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Fact,
            "Fact anchor",
            &[f_vec.as_slice(); 4],
            &refs,
            2.0,
        )
        .expect("fact");
    corpus
        .ingest_text_activations(
            &adapter,
            AttractorRole::Guard,
            "Guard anchor",
            &[g_vec.as_slice(); 4],
            &refs,
            2.0,
        )
        .expect("guard");

    let policy = RecallPolicy {
        min_coherence: 0.25,
        boundary_margin: 0.03,
        guard_suppress_margin: 0.0,
        top_k_facts: 2,
        top_k_personas: 1,
    };

    // Layer 1 query vector (strong Guard lead: margin > 0.05).
    let mut l1_q = vec![0.0_f32; q_len];
    l1_q[0] = 0.20;
    l1_q[1] = 0.85;

    // Layer 2 query vector (narrow Guard lead: 0 < margin <= 0.05).
    let mut l2_q = vec![0.0_f32; q_len];
    l2_q[0] = 0.695;
    l2_q[1] = 0.719;

    let complete_ood_obs = LocalChannelObservation {
        combined_scores: vec![0.0, 0.0],
        q_to_d_scores: vec![0.0, 0.0],
        d_to_q_scores: vec![0.0, 0.0],
        token_deposits: Vec::new(),
        unexplained_token_ratio: 0.94,
        unexplained_corpus_token_ratio: 0.94,
        query_seq_len: 16,
    };

    // 1. Complete foreign OOD suppresses Layer 1 hard-block Guard (both recall & TurnOrchestrator).
    let l1_ood = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &[l1_q.as_slice(); 4],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(complete_ood_obs.clone()),
        )
        .expect("l1 ood recall");
    assert_ne!(
        l1_ood.action_decision,
        StigmergicActionDecision::RepulsionEscape
    );
    assert!(l1_ood.guard_conflict_margin <= 0.0);
    assert!(!l1_ood.telemetry.is_soft_guard_caution);
    assert_eq!(l1_ood.lacuna_state, LacunaBoundaryState::LiminalExterior);
    assert!(l1_ood.recalled_facts.is_empty());

    // 2. Complete foreign OOD suppresses Layer 2 soft-caution Guard (`+R_soft` must NOT emit).
    let l2_ood = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &[l2_q.as_slice(); 4],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(complete_ood_obs),
        )
        .expect("l2 ood recall");
    assert_ne!(
        l2_ood.action_decision,
        StigmergicActionDecision::RepulsionEscape
    );
    assert!(l2_ood.guard_conflict_margin <= 0.0);
    assert!(!l2_ood.telemetry.is_soft_guard_caution);
    assert_eq!(l2_ood.lacuna_state, LacunaBoundaryState::LiminalExterior);
    assert!(l2_ood.recalled_facts.is_empty());

    // Confirm through `TurnOrchestrator::prepare_turn` that neither Layer 1 nor Layer 2 emits
    // GuardPhaseAnchors or enables steering on complete foreign OOD.
    struct StaticRecallSource(OwnedAttractorRecall);
    impl AttractorRecallSource for StaticRecallSource {
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
    let mut kv_cache = StandardPhaseKvGopCache::new();
    let kv_policy = AdaptiveThresholdPolicy::default();
    for (idx, owned) in [l1_ood.to_owned_recall(), l2_ood.to_owned_recall()]
        .into_iter()
        .enumerate()
    {
        let src = StaticRecallSource(owned);
        let plan = TurnOrchestrator::prepare_turn(
            &src,
            &mut adapter,
            &mut kv_cache,
            &TurnPrepareInput::<AdaptiveThresholdPolicy, StandardCognitiveBank> {
                query_token_qs: &[],
                query_seq_len: 16,
                token_include_mask: None,
                turn_index: (idx + 1) as u32,
                kv_policy: &kv_policy,
                kv_min_resonance_override: None,
                persona_bank: None,
                layer_steer_mask_override: None,
            },
        )
        .expect("ood plan");
        assert!(!plan.is_guard_trap());
        assert!(!plan.is_soft_guard_caution());
        assert!(plan.is_unobserved_lacuna());
        assert!(!plan.steering_eligible());
        assert!(plan.guard_anchors_per_layer().iter().all(Vec::is_empty));
    }

    // 3. Genuine Guard probe whose tokens are explained by Guard entries (`unexp_corpus = 0.35`)
    //    even when Fact-only `unexp = 0.90 >= u_foreign` MUST still fire Layer 1 & Layer 2 Guard!
    let guard_explained_obs = LocalChannelObservation {
        combined_scores: vec![0.0, 0.0],
        q_to_d_scores: vec![0.0, 0.0],
        d_to_q_scores: vec![0.0, 0.0],
        token_deposits: Vec::new(),
        unexplained_token_ratio: 0.90,
        unexplained_corpus_token_ratio: 0.35,
        query_seq_len: 16,
    };
    let l1_guard = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &[l1_q.as_slice(); 4],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(guard_explained_obs.clone()),
        )
        .expect("l1 true guard");
    assert_eq!(
        l1_guard.action_decision,
        StigmergicActionDecision::RepulsionEscape
    );
    assert!(l1_guard.guard_conflict_margin > 0.05);

    let l2_guard = corpus
        .recall_for_query_with_strata_observation(
            &adapter,
            &[l2_q.as_slice(); 4],
            &refs,
            &policy,
            &StratumPipeline::default(),
            Some(guard_explained_obs),
        )
        .expect("l2 true guard");
    assert_ne!(
        l2_guard.action_decision,
        StigmergicActionDecision::RepulsionEscape
    );
    assert!(l2_guard.telemetry.is_soft_guard_caution);
}
