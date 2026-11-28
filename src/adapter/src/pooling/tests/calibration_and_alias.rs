// name: src/adapter/src/pooling/tests/calibration_and_alias.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! BasePrior, null calibration, alias probing, and startup probe parity tests.

use super::*;

#[test]
fn token_include_mask_completely_excludes_outlier_patches_and_fails_closed() {
    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 4;
    geom.num_q_heads = 4;
    geom.num_kv_heads = 2;
    geom.head_dim = 8;
    geom.rotary_dim = 8;
    geom.layer_rope_mask = 0xF;
    let adapter = ModelSteeringAdapter::new(geom).expect("init compact");
    let num_layers = adapter.geometry().num_layers;
    let dim = adapter.geometry().q_rot_len();

    let make_clean_and_padded =
        |active_dims: &[usize]| -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<bool>) {
            let mut clean_layers = Vec::with_capacity(num_layers);
            let mut padded_layers = Vec::with_capacity(num_layers);
            for l in 0..num_layers {
                let mut clean = vec![0.0_f32; active_dims.len() * dim];
                let mut padded = vec![0.0_f32; (active_dims.len() + 3) * dim];
                // Prefix 2 huge outlier patch tokens (including huge values) at slots 0..2.
                for patch_t in 0..2 {
                    for d in 0..dim {
                        padded[patch_t * dim + d] = 1_000_000.0 * ((d + l + 1) as f32);
                    }
                }
                for (idx, &slot) in active_dims.iter().enumerate() {
                    let pos = (slot + l) % dim;
                    clean[idx * dim + pos] = 1.0;
                    clean[idx * dim + ((pos + 1) % dim)] = 0.25;
                    let dst_t = idx + 2;
                    padded[dst_t * dim + pos] = 1.0;
                    padded[dst_t * dim + ((pos + 1) % dim)] = 0.25;
                }
                // Trailing excluded delimiter token with extreme negative spike.
                let end_t = active_dims.len() + 2;
                for d in 0..dim {
                    padded[end_t * dim + d] = -500_000.0;
                }
                clean_layers.push(clean);
                padded_layers.push(padded);
            }
            let mut mask = vec![false, false];
            mask.extend(std::iter::repeat_n(true, active_dims.len()));
            mask.push(false);
            (clean_layers, padded_layers, mask)
        };

    let (clean_a, pad_a, mask_a) = make_clean_and_padded(&[2, 5]);
    let (clean_b, pad_b, mask_b) = make_clean_and_padded(&[10, 13]);
    let (clean_c, pad_c, mask_c) = make_clean_and_padded(&[18, 21]);

    // 1. Direct pooler exact equivalence & None == all-true parity.
    let pooler = SequenceActivationPooler::default();
    let ref_vec = vec![0.1_f32; dim];
    let raw_clean = SequenceActivationPooler::uniform_raw_mean(&clean_a[0], 2, dim).expect("raw");
    let raw_masked =
        SequenceActivationPooler::uniform_raw_mean_masked(&pad_a[0], 5, dim, Some(&mask_a))
            .expect("raw masked");
    assert_eq!(raw_clean, raw_masked);

    let sif_clean = pooler
        .pool_layer(&clean_a[0], 2, dim, Some(&ref_vec))
        .expect("sif");
    let sif_all_true = pooler
        .pool_layer_masked(&clean_a[0], 2, dim, Some(&ref_vec), Some(&[true, true]))
        .expect("sif all true");
    let sif_masked = pooler
        .pool_layer_masked(&pad_a[0], 5, dim, Some(&ref_vec), Some(&mask_a))
        .expect("sif masked");
    assert_eq!(sif_clean, sif_all_true);
    assert_eq!(sif_clean, sif_masked);

    // 2. BasePrior + Corpus compilation + LocalObservation + Recall exact invariance.
    let clean_entries = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Clean Fact A".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: clean_a.clone(),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Clean Fact B".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: clean_b.clone(),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Clean Fact C".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: clean_c.clone(),
            token_include_mask: None,
        },
    ];
    let padded_entries = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Clean Fact A".to_string(),
            energy: 2.0,
            seq_len: 5,
            per_layer_token_qs: pad_a.clone(),
            token_include_mask: Some(mask_a.clone()),
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Clean Fact B".to_string(),
            energy: 2.0,
            seq_len: 5,
            per_layer_token_qs: pad_b,
            token_include_mask: Some(mask_b),
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Clean Fact C".to_string(),
            energy: 2.0,
            seq_len: 5,
            per_layer_token_qs: pad_c,
            token_include_mask: Some(mask_c),
        },
    ];

    let prior_clean =
        ModelBaseAttractorProfile::extract(&adapter, &clean_entries, pooler, 0.55).expect("p1");
    let prior_masked =
        ModelBaseAttractorProfile::extract(&adapter, &padded_entries, pooler, 0.55).expect("p2");
    assert_eq!(
        prior_clean.novelty_reference_per_layer,
        prior_masked.novelty_reference_per_layer
    );
    assert_eq!(
        prior_clean.token_common_mode_basis,
        prior_masked.token_common_mode_basis
    );

    let comp_clean = CompiledTokenCorpus::compile(&adapter, &clean_entries, pooler, 2, 1)
        .expect("compile clean");
    let comp_masked = CompiledTokenCorpus::compile(&adapter, &padded_entries, pooler, 2, 1)
        .expect("compile masked");
    assert_eq!(
        comp_clean.corpus_centroid_per_layer,
        comp_masked.corpus_centroid_per_layer
    );
    assert_eq!(comp_clean.entry_token_sigs, comp_masked.entry_token_sigs);

    let obs_clean = comp_clean
        .build_local_observation(&adapter, &clean_a, 2)
        .expect("obs clean");
    let obs_masked = comp_clean
        .build_local_observation_masked(&adapter, &pad_a, 5, Some(&mask_a))
        .expect("obs masked");
    assert_eq!(obs_clean.query_seq_len, obs_masked.query_seq_len);
    assert_eq!(
        obs_clean.unexplained_token_ratio.to_bits(),
        obs_masked.unexplained_token_ratio.to_bits()
    );
    assert_eq!(obs_clean.combined_scores, obs_masked.combined_scores);

    let rec_clean = comp_clean
        .recall_token_sequence(&adapter, &clean_a, 2)
        .expect("rec clean");
    let rec_masked = comp_clean
        .recall_token_sequence_with_mask(&adapter, &pad_a, 5, Some(&mask_a))
        .expect("rec masked");
    assert_eq!(rec_clean.lacuna_state, rec_masked.lacuna_state);
    assert_eq!(
        rec_clean.top1_fact_score.to_bits(),
        rec_masked.top1_fact_score.to_bits()
    );

    // 3. All-false mask fails closed without NaN/div0 into LiminalExterior / Unobserved.
    let all_false = vec![false; 5];
    let zero_pool = pooler
        .pool_layer_masked(&pad_a[0], 5, dim, Some(&ref_vec), Some(&all_false))
        .expect("all-false pool");
    assert!(zero_pool.iter().all(|&v| v == 0.0 && v.is_finite()));
    let all_false_rec = comp_clean
        .recall_token_sequence_with_mask(&adapter, &pad_a, 5, Some(&all_false))
        .expect("all-false recall");
    assert_eq!(
        all_false_rec.lacuna_state,
        LacunaBoundaryState::LiminalExterior
    );
    assert!(all_false_rec.recalled_facts.is_empty());
    assert!(!all_false_rec.is_known_unknown_exterior);
    assert!(all_false_rec.top1_fact_score.is_finite());
    assert_eq!(all_false_rec.telemetry.unexplained_token_ratio, 1.0);

    // 4. Length mismatch fails closed with Err (never panics).
    let wrong_len = vec![true; 3];
    assert!(
        pooler
            .pool_layer_masked(&pad_a[0], 5, dim, Some(&ref_vec), Some(&wrong_len))
            .is_err()
    );
    assert!(
        comp_clean
            .recall_token_sequence_with_mask(&adapter, &pad_a, 5, Some(&wrong_len))
            .is_err()
    );
}

#[test]
fn multilingual_startup_prior_probes_and_bimodal_null_envelope_defense() {
    use super::base_prior::fit_cross_entry_token_common_mode_with_scripts;
    use super::null_calibration::{ScriptGroup, aggregate_null_envelope, classify_script_group};

    // 1. 24 startup probes: 12 AsciiDominant + 12 NonAsciiDominant, 6+6 in even/odd folds.
    assert_eq!(DEFAULT_STARTUP_PRIOR_PROBES.len(), 24);
    let mut fold0_ascii = 0_usize;
    let mut fold0_non_ascii = 0_usize;
    let mut fold1_ascii = 0_usize;
    let mut fold1_non_ascii = 0_usize;
    for (idx, prompt) in DEFAULT_STARTUP_PRIOR_PROBES.iter().enumerate() {
        match (idx % 2, classify_script_group(prompt)) {
            (0, ScriptGroup::AsciiDominant) => fold0_ascii += 1,
            (0, ScriptGroup::NonAsciiDominant) => fold0_non_ascii += 1,
            (1, ScriptGroup::AsciiDominant) => fold1_ascii += 1,
            (1, ScriptGroup::NonAsciiDominant) => fold1_non_ascii += 1,
            _ => unreachable!(),
        }
    }
    assert_eq!((fold0_ascii, fold0_non_ascii), (6, 6));
    assert_eq!((fold1_ascii, fold1_non_ascii), (6, 6));

    // Original 12 English prompts must all be preserved verbatim.
    let original_en = [
        "Explain the basic design trade-offs of a modular systems architecture.",
        "Summarize how memory alignment affects hardware cache performance.",
        "What is 23 multiplied by 4?",
        "Calculate the remainder when 50 is divided by 7.",
        "Which ocean lies between Europe and North America?",
        "How many seconds are in a single hour?",
        "Write a short function signature in C that returns the square of an integer.",
        "What does a hash table use to map keys to array slots?",
        "Suggest two quick ways to keep fresh herbs crisp in the refrigerator.",
        "How can someone politely follow up on an unanswered email?",
        "Briefly explain how evaporation cools a liquid surface.",
        "Define the term algorithm in one clear sentence.",
    ];
    for en in original_en {
        assert!(
            DEFAULT_STARTUP_PRIOR_PROBES.contains(&en),
            "missing preserved English probe: {en}"
        );
    }

    // 2. Bimodal English+Japanese distribution must NOT suffer inter-script variance blow-up
    //    nor p85 quantile dilution relative to monolingual English calibration.
    let en_peaks: Vec<(ScriptGroup, f32)> = (0..12)
        .map(|i| (ScriptGroup::AsciiDominant, 0.23 + (i as f32) * 0.004))
        .collect();
    let mut mixed_peaks = en_peaks.clone();
    for i in 0..12 {
        mixed_peaks.push((ScriptGroup::NonAsciiDominant, 0.03 + (i as f32) * 0.003));
    }
    let en_toks: Vec<(ScriptGroup, f32)> = (0..12)
        .map(|i| (ScriptGroup::AsciiDominant, 0.36 + (i as f32) * 0.006))
        .collect();
    let mut mixed_toks = en_toks.clone();
    for i in 0..12 {
        mixed_toks.push((ScriptGroup::NonAsciiDominant, 0.10 + (i as f32) * 0.004));
    }
    let en_orthos: Vec<(ScriptGroup, f32)> = (0..12)
        .map(|i| (ScriptGroup::AsciiDominant, 0.45 + (i as f32) * 0.01))
        .collect();
    let mut mixed_orthos = en_orthos.clone();
    for i in 0..12 {
        mixed_orthos.push((ScriptGroup::NonAsciiDominant, 0.25 + (i as f32) * 0.01));
    }

    let en_unexps: Vec<(ScriptGroup, f32)> = (0..12)
        .map(|i| (ScriptGroup::AsciiDominant, 0.80 + (i as f32) * 0.015))
        .collect();
    let mut mixed_unexps = en_unexps.clone();
    for i in 0..12 {
        mixed_unexps.push((ScriptGroup::NonAsciiDominant, 0.92 + (i as f32) * 0.005));
    }

    let (env_en, tok_en) =
        aggregate_null_envelope(0.18, &en_peaks, &en_orthos, &en_toks, &en_unexps).expect("en env");
    let (env_mixed, tok_mixed) = aggregate_null_envelope(
        0.18,
        &mixed_peaks,
        &mixed_orthos,
        &mixed_toks,
        &mixed_unexps,
    )
    .expect("mixed env");
    let (env_mixed_2, tok_mixed_2) = aggregate_null_envelope(
        0.18,
        &mixed_peaks,
        &mixed_orthos,
        &mixed_toks,
        &mixed_unexps,
    )
    .expect("mixed env det");
    assert_eq!(env_mixed, env_mixed_2);
    assert_eq!(tok_mixed.to_bits(), tok_mixed_2.to_bits());
    assert!(
        (env_mixed.ood_null_high - env_en.ood_null_high).abs() < 1e-6,
        "bimodal mixed probes distorted ood_null_high: en={} mixed={}",
        env_en.ood_null_high,
        env_mixed.ood_null_high
    );
    assert!(
        (tok_mixed - tok_en).abs() < 1e-6,
        "bimodal mixed probes diluted token_explain_threshold: en={tok_en} mixed={tok_mixed}"
    );

    // 3. Cross-script token common-mode support filter keeps shared directions and rejects
    //    single-language-only directions in a 12+12 balanced pool.
    let dim = 16;
    let mut per_entry_toks = Vec::with_capacity(24);
    let mut scripts = Vec::with_capacity(24);
    for i in 0..24 {
        let is_ascii = i % 2 == 0;
        let mut t_shared = vec![0.0_f32; dim];
        t_shared[0] = 1.0; // Shared across all 24 EN+JA probes.
        let mut t_lang_only = vec![0.0_f32; dim];
        if is_ascii {
            t_lang_only[1] = 1.0; // Strictly English-only (12/24).
        } else {
            t_lang_only[2] = 1.0; // Strictly Japanese-only (12/24).
        }
        per_entry_toks.push(vec![t_shared, t_lang_only]);
        scripts.push(if is_ascii {
            ScriptGroup::AsciiDominant
        } else {
            ScriptGroup::NonAsciiDominant
        });
    }
    let basis = fit_cross_entry_token_common_mode_with_scripts(&per_entry_toks, Some(&scripts), 4);
    assert_eq!(
        basis.len(),
        1,
        "only the cross-lingual shared direction should survive, got {} dirs",
        basis.len()
    );
    assert!(
        basis[0][0].abs() > 0.95,
        "extracted basis must align with shared dim 0: {:?}",
        basis[0]
    );

    // 4. Antipodal cross-script offset (`+e3` in ASCII vs `-e3` in non-ASCII) must be rejected
    //    by token common-mode filtering and never absorbed into `pooled_common_mode_basis`.
    let mut antipodal_toks = Vec::with_capacity(24);
    for i in 0..24 {
        let is_ascii = i % 2 == 0;
        let mut t_shared = vec![0.0_f32; dim];
        t_shared[0] = 1.0;
        let mut t_antipodal = vec![0.0_f32; dim];
        t_antipodal[3] = if is_ascii { 1.0 } else { -1.0 };
        antipodal_toks.push(vec![t_shared, t_antipodal]);
    }
    let anti_basis =
        fit_cross_entry_token_common_mode_with_scripts(&antipodal_toks, Some(&scripts), 4);
    assert_eq!(
        anti_basis.len(),
        1,
        "antipodal between-script axis must be rejected, got {} dirs",
        anti_basis.len()
    );
    assert!(
        anti_basis[0][0].abs() > 0.95 && anti_basis[0][3].abs() < 0.05,
        "antipodal axis leaked into token_common_mode_basis: {:?}",
        anti_basis[0]
    );

    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 2;
    geom.num_q_heads = 4;
    geom.num_kv_heads = 2;
    geom.head_dim = 8;
    geom.rotary_dim = 8;
    geom.layer_rope_mask = 0x3;
    let adapter = ModelSteeringAdapter::new(geom).expect("compact adapter");
    let q_dim = adapter.geometry().q_rot_len();
    let mut mixed_samples = Vec::with_capacity(12);
    for i in 0..12 {
        let is_ascii = i % 2 == 0;
        let intra = (i / 2) as f32;
        let mut flat = vec![0.0_f32; 2 * q_dim];
        for t in 0..2 {
            // Pure between-script mean shift on dim 0 (`+1.0` EN vs `-1.0` JA).
            flat[t * q_dim] = if is_ascii { 1.0 } else { -1.0 };
            // Genuine shared within-script variation on dim 1.
            flat[t * q_dim + 1] = (intra - 2.5) * 0.25 + (t as f32) * 0.15;
            // Per-sample semantic diversity on dim 2 + (i % 6).
            flat[t * q_dim + 2 + (i % 6)] = 0.20;
        }
        mixed_samples.push(TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: if is_ascii {
                format!("Neutral ASCII sample {i}")
            } else {
                format!(
                    "\u{4E2D}\u{7ACB}\u{7684}\u{306A}\u{65E5}\u{672C}\u{8A9E}\u{6A19}\u{672C}{i}"
                )
            },
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: vec![flat; 2],
            token_include_mask: None,
        });
    }
    let mixed_prior = ModelBaseAttractorProfile::extract(
        &adapter,
        &mixed_samples,
        SequenceActivationPooler::default(),
        0.55,
    )
    .expect("extract mixed prior");
    assert!(
        !mixed_prior.pooled_common_mode_basis.is_empty(),
        "within-script centered PCA must still recover shared within-script variance"
    );
    for dir in &mixed_prior.pooled_common_mode_basis {
        // Layer 0 dim 0 and layer 1 dim 0 carry the pure between-script shift; within-script
        // residualization must prevent that shift axis from dominating any pooled basis vector.
        let shift_energy = dir[0] * dir[0] + dir[q_dim] * dir[q_dim];
        assert!(
            shift_energy < 0.05,
            "between-script mean shift leaked into pooled_common_mode_basis (energy={shift_energy:.4})"
        );
    }
}

#[test]
fn extract_leading_alias_word_is_whitespace_free_and_utf8_safe() {
    // English non-regression.
    assert_eq!(
        extract_leading_alias_word("Alpha, beta gamma"),
        Some("Alpha")
    );
    assert_eq!(
        extract_leading_alias_word("Phi-4-mini runs locally."),
        Some("Phi-4-mini")
    );
    assert_eq!(extract_leading_alias_word("A: short"), None);
    assert_eq!(extract_leading_alias_word("   "), None);

    // Non-spaced Japanese sentence must never collapse into whole-line alias.
    let ja_sentence = "\u{7FE0}\u{7D0B}\u{89E3}\u{6790}\u{5668}\u{306F}\u{84BC}\u{7A79}\u{89B3}\u{6E2C}\
     \u{5854}\u{306E}\u{5730}\u{4E0B}\u{7B2C}4\u{5C64}\u{306B}\u{8A2D}\u{7F6E}\u{3055}\u{308C}\u{305F}\
     \u{7121}\u{96FB}\u{529B}\u{306E}\u{5149}\u{5B66}\u{4F4D}\u{76F8}\u{8A08}\u{3067}\u{3042}\u{308B}\
     \u{3002}";
    let ja_span = extract_leading_alias_word(ja_sentence).expect("ja span");
    assert_eq!(ja_span.chars().count(), 8);
    assert_eq!(
        ja_span,
        "\u{7FE0}\u{7D0B}\u{89E3}\u{6790}\u{5668}\u{306F}\u{84BC}\u{7A79}"
    );
    assert!(ja_span.chars().count() < ja_sentence.chars().count() / 2);

    // Leading CJK brackets/quotes and punctuation delimiters.
    assert_eq!(
        extract_leading_alias_word(
            "\u{300C}\u{6771}\u{4EAC}\u{300D}\u{306F}\u{65E5}\u{672C}\u{306E}\u{9996}\u{90FD}\u{3067}\u{3059}\u{3002}"
        ),
        Some("\u{6771}\u{4EAC}")
    );
    assert_eq!(
        extract_leading_alias_word(
            "\u{FF08}\u{8A2D}\u{8A08}\u{FF09}\u{65B9}\u{91DD}\u{3092}\u{78BA}\u{8A8D}\u{3059}\u{308B}"
        ),
        Some("\u{8A2D}\u{8A08}")
    );
    assert_eq!(
        extract_leading_alias_word(
            "\u{7D05}\u{7389}\u{3001}\u{7FE0}\u{7D0B}\u{3001}\u{767D}\u{78C1}"
        ),
        Some("\u{7D05}\u{7389}")
    );
    assert_eq!(extract_leading_alias_word("\u{300C}\u{300D}"), None);

    // Mixed ASCII + CJK without spaces stays bounded on UTF-8 char boundaries.
    let mixed = "HTTP\u{901A}\u{4FE1}\u{30D7}\u{30ED}\u{30C8}\u{30B3}\u{30EB}\u{306E}\u{4ED5}\u{69D8}\u{6982}\u{8981}";
    let mixed_span = extract_leading_alias_word(mixed).expect("mixed span");
    assert_eq!(mixed_span.chars().count(), 8);
    assert_eq!(mixed_span, "HTTP\u{901A}\u{4FE1}\u{30D7}\u{30ED}");
}

#[test]
fn null_calibration_2pass_2d_stats_cross_script_envelope_relaxed_clamp_and_determinism() {
    use super::null_calibration::{
        ScriptGroup, aggregate_null_envelope, resolve_token_explain_threshold,
    };

    // 1. Relaxed `token_explain_threshold` clamp (`0.22..=0.62`, fallback `0.40`).
    let low_cos: Vec<(ScriptGroup, f32)> = (0..8)
        .map(|i| (ScriptGroup::AsciiDominant, 0.16 + (i as f32) * 0.005))
        .collect();
    let low_thresh = resolve_token_explain_threshold(&low_cos);
    assert!(
        (0.22..0.35).contains(&low_thresh),
        "low-cosine model should relax below legacy 0.35 floor, got {low_thresh}"
    );
    let very_low: Vec<(ScriptGroup, f32)> = vec![(ScriptGroup::AsciiDominant, 0.05); 8];
    assert!((resolve_token_explain_threshold(&very_low) - 0.22).abs() < 1e-6);

    let high_cos: Vec<(ScriptGroup, f32)> = vec![(ScriptGroup::AsciiDominant, 0.56); 8];
    let high_thresh = resolve_token_explain_threshold(&high_cos);
    assert!(
        (high_thresh - 0.60).abs() < 1e-6,
        "high-cosine model should expand above legacy 0.56 ceiling, got {high_thresh}"
    );
    assert!((resolve_token_explain_threshold(&[]) - DEFAULT_TOKEN_EXPLAIN_THRESHOLD).abs() < 1e-6);

    // 2. Conservative cross-script 2D envelopes:
    //    - `ood_coh_median` takes higher script (upper bound preventing OOD leakage)
    //    - `ood_unexp_median` / `ood_unexp_p15` take lower script (lower bound preventing OOD false pass)
    let peaks: Vec<(ScriptGroup, f32)> = (0..6)
        .map(|i| (ScriptGroup::AsciiDominant, 0.20 + (i as f32) * 0.01))
        .chain((0..6).map(|i| (ScriptGroup::NonAsciiDominant, 0.04 + (i as f32) * 0.005)))
        .collect();
    let orthos: Vec<(ScriptGroup, f32)> = vec![(ScriptGroup::AsciiDominant, 0.50); 6]
        .into_iter()
        .chain(vec![(ScriptGroup::NonAsciiDominant, 0.30); 6])
        .collect();
    let toks: Vec<(ScriptGroup, f32)> = vec![(ScriptGroup::AsciiDominant, 0.28); 6]
        .into_iter()
        .chain(vec![(ScriptGroup::NonAsciiDominant, 0.14); 6])
        .collect();
    let unexps: Vec<(ScriptGroup, f32)> = (0..6)
        .map(|i| (ScriptGroup::AsciiDominant, 0.78 + (i as f32) * 0.02))
        .chain((0..6).map(|i| (ScriptGroup::NonAsciiDominant, 0.94 + (i as f32) * 0.01)))
        .collect();

    let (env1, t1) = aggregate_null_envelope(0.16, &peaks, &orthos, &toks, &unexps).expect("env1");
    let (env2, t2) = aggregate_null_envelope(0.16, &peaks, &orthos, &toks, &unexps).expect("env2");
    assert_eq!(env1, env2);
    assert_eq!(t1.to_bits(), t2.to_bits());
    assert!(
        env1.ood_coh_median >= 0.22,
        "ood_coh_median should adopt conservative upper script (ASCII ~0.225), got {}",
        env1.ood_coh_median
    );
    assert!(env1.ood_coh_mad >= OOD_COH_MAD_FLOOR);
    assert!(
        env1.ood_unexp_median <= 0.85,
        "ood_unexp_median should adopt conservative lower script (ASCII ~0.83), got {}",
        env1.ood_unexp_median
    );
    assert!(
        env1.ood_unexp_p15 <= 0.80,
        "ood_unexp_p15 should adopt conservative lower script (ASCII 0.78), got {}",
        env1.ood_unexp_p15
    );
    assert!(env1.ood_unexp_mad >= OOD_UNEXP_MAD_FLOOR);

    // 3. Empty `unexp_samples` falls back to safe OOD defaults `(1.0, 0.04, 0.80)`.
    let (env_empty_u, _) =
        aggregate_null_envelope(0.16, &peaks, &orthos, &toks, &[]).expect("empty unexp");
    assert!((env_empty_u.ood_unexp_median - 1.0).abs() < 1e-6);
    assert!((env_empty_u.ood_unexp_mad - OOD_UNEXP_MAD_FLOOR).abs() < 1e-6);
    assert!((env_empty_u.ood_unexp_p15 - 0.80).abs() < 1e-6);

    // 4. End-to-end 2-pass cross-fitting populates corpus.null_calibration() deterministically
    //    while `--no-base-prior` keeps `null_calibration == None` and `tok_exp == 0.40`.
    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 4;
    geom.num_q_heads = 4;
    geom.num_kv_heads = 2;
    geom.head_dim = 8;
    geom.rotary_dim = 8;
    geom.layer_rope_mask = 0xF;
    let adapter = ModelSteeringAdapter::new(geom).expect("compact adapter");
    let dim = adapter.geometry().q_rot_len();
    let make_seq = |slots: &[usize]| -> Vec<Vec<f32>> {
        let mut flat = vec![0.0_f32; slots.len() * dim];
        for (t, &s) in slots.iter().enumerate() {
            flat[t * dim + (s % dim)] = 1.0;
            flat[t * dim + ((s + 1) % dim)] = 0.25;
        }
        vec![flat; 4]
    };
    let facts = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact one".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[0, 2]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact two".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[4, 6]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact three".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[8, 10]),
            token_include_mask: None,
        },
    ];
    let prior_samples: Vec<TokenSequenceEntry> = (0..8)
        .map(|i| TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: if i % 2 == 0 {
                format!("Neutral ASCII probe {i}")
            } else {
                format!("\u{4E2D}\u{7ACB}\u{30D7}\u{30ED}\u{30FC}\u{30D6}\u{6A19}\u{672C}{i}")
            },
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[14 + i, 20 + i]),
            token_include_mask: None,
        })
        .collect();
    let prior = ModelBaseAttractorProfile::extract(
        &adapter,
        &prior_samples,
        SequenceActivationPooler::default(),
        0.55,
    )
    .expect("prior");

    let no_prior =
        CompiledTokenCorpus::compile(&adapter, &facts, SequenceActivationPooler::default(), 2, 1)
            .expect("no prior");
    assert!(no_prior.null_calibration.is_none());
    assert!(no_prior.corpus.null_calibration().is_none());
    assert!((no_prior.token_explain_threshold - DEFAULT_TOKEN_EXPLAIN_THRESHOLD).abs() < 1e-6);

    let run_a = CompiledTokenCorpus::compile_with_base_prior(
        &adapter,
        &facts,
        SequenceActivationPooler::default(),
        2,
        1,
        Some(&prior),
    )
    .expect("with prior A");
    let run_b = CompiledTokenCorpus::compile_with_base_prior(
        &adapter,
        &facts,
        SequenceActivationPooler::default(),
        2,
        1,
        Some(&prior),
    )
    .expect("with prior B");
    let env_a = run_a.null_calibration.expect("env_a");
    let env_b = run_b.null_calibration.expect("env_b");
    assert_eq!(env_a, env_b);
    assert_eq!(run_a.corpus.null_calibration(), Some(env_a));
    assert_eq!(env_a.sample_count, 8);
    assert!(env_a.ood_coh_mad >= OOD_COH_MAD_FLOOR);
    assert!(env_a.ood_unexp_mad >= OOD_UNEXP_MAD_FLOOR);
    assert!((0.0..=1.0).contains(&env_a.ood_unexp_median));
    assert!((0.0..=1.0).contains(&env_a.ood_unexp_p15));
    assert!((0.22..=0.62).contains(&env_a.token_explain_threshold));

    // 5. `ood_null_high` removes the legacy `0.36` cap for high-cosine models (`~0.42..0.44`)
    //    while honoring `RecallEnvelopeBounds::standard().ood_null_coherence_ceiling` (`0.50`)
    //    so extreme anisotropic distributions (`0.60+`) never collapse 2D `SELF_REF` recall.
    let high_cos_peaks: Vec<(ScriptGroup, f32)> = (0..12)
        .map(|i| (ScriptGroup::AsciiDominant, 0.34 + (i as f32) * 0.008))
        .collect();
    let (env_high, _) = aggregate_null_envelope(0.18, &high_cos_peaks, &orthos, &toks, &unexps)
        .expect("high cos env");
    assert!(
        env_high.ood_null_high > 0.40 && env_high.ood_null_high <= 0.50,
        "high-cosine OOD tail must exceed legacy 0.36 cap without exceeding 0.50 ceiling, got {}",
        env_high.ood_null_high
    );
    let extreme_aniso_peaks: Vec<(ScriptGroup, f32)> = (0..12)
        .map(|i| (ScriptGroup::NonAsciiDominant, 0.50 + (i as f32) * 0.012))
        .collect();
    let (env_aniso, _) =
        aggregate_null_envelope(0.20, &extreme_aniso_peaks, &orthos, &toks, &unexps)
            .expect("aniso env");
    assert!(
        (env_aniso.ood_null_high
            - crate::RecallEnvelopeBounds::standard().ood_null_coherence_ceiling)
            .abs()
            < 1e-6,
        "extreme anisotropic OOD tail must clamp at 0.50 viability ceiling, got {}",
        env_aniso.ood_null_high
    );
}

#[test]
fn base_prior_discriminative_orthogonalization_protects_corpus_signals_across_all_three_paths() {
    use super::base_prior::{
        build_corpus_discriminative_subspace, orthogonalize_prior_basis_against_discriminative,
    };

    // 1. Direct linear-algebra invariants on discriminative subspace & noise-amplification guard.
    let dim = 16;
    // All 4 corpus entries share BOS/sink energy on dim 0 (`0.80`), while differing on dims 1..=4.
    let mut centroids = Vec::new();
    for i in 0..4_usize {
        let mut c = vec![0.0_f32; dim];
        c[0] = 0.80;
        c[1 + i] = 0.60;
        centroids.push(c);
    }
    let disc = build_corpus_discriminative_subspace(&centroids, &[]);
    assert_eq!(disc.len(), 3, "4 entries span rank-3 inter-entry subspace");
    // Shared BOS/sink dim 0 must cancel identically in centered inter-entry differences.
    for u in &disc {
        assert!(
            u[0].abs() < 1e-5,
            "shared BOS/sink dim 0 leaked into discriminative subspace: {}",
            u[0]
        );
    }

    // (a) Pure BOS/sink prior along dim 0 passes through untouched (`cos > 0.99`).
    let mut bos_prior = vec![0.0_f32; dim];
    bos_prior[0] = 1.0;
    // (b) Topical prior aligned with corpus discrimination (`(e1 - e2)/sqrt(2)`) has 0% energy
    //     outside `disc` -> MUST be discarded (never re-normalized from noise).
    let mut overlap_prior = vec![0.0_f32; dim];
    overlap_prior[1] = std::f32::consts::FRAC_1_SQRT_2;
    overlap_prior[2] = -std::f32::consts::FRAC_1_SQRT_2;
    // (c) Mixed prior: 75% orthogonal framing (dim 10) + 25% corpus discrimination (`e1 - e2`):
    //     discriminative component must be stripped while retaining orthogonal framing on dim 10.
    let mut mixed_prior = vec![0.0_f32; dim];
    mixed_prior[10] = 0.75_f32.sqrt();
    mixed_prior[1] = 0.5 * std::f32::consts::FRAC_1_SQRT_2;
    mixed_prior[2] = -0.5 * std::f32::consts::FRAC_1_SQRT_2;

    let filtered = orthogonalize_prior_basis_against_discriminative(
        &[bos_prior.clone(), overlap_prior, mixed_prior],
        &disc,
    );
    assert_eq!(
        filtered.len(),
        2,
        "fully overlapping topical prior must be discarded; BOS and mixed-orthogonal must survive"
    );
    assert!(filtered[0][0].abs() > 0.99);
    assert!(filtered[1][10].abs() > 0.99);
    for u in &disc {
        let dot: f32 = filtered[1].iter().zip(u.iter()).map(|(&a, &b)| a * b).sum();
        assert!(
            dot.abs() < 1e-5,
            "filtered prior must be strictly orthogonal to discriminative subspace: dot={dot}"
        );
    }

    // Single-entry and empty corpora must safely skip without NaN.
    assert!(build_corpus_discriminative_subspace(&centroids[..1], &[]).is_empty());
    assert!(build_corpus_discriminative_subspace(&[], &[]).is_empty());
    let single_pass =
        orthogonalize_prior_basis_against_discriminative(std::slice::from_ref(&bos_prior), &[]);
    assert_eq!(single_pass.len(), 1);
    assert!(single_pass[0].iter().all(|v| v.is_finite()));

    // 2. End-to-end 3-path check (batch + 2-fold calibration + streaming) with a base_prior whose
    //    token common-mode overlaps corpus Fact discrimination alongside a shared BOS/sink axis.
    let mut geom = sample_gqa_partial_rope();
    geom.num_layers = 4;
    geom.num_q_heads = 4;
    geom.num_kv_heads = 2;
    geom.head_dim = 8;
    geom.rotary_dim = 8;
    geom.layer_rope_mask = 0xF;
    let adapter = ModelSteeringAdapter::new(geom).expect("compact adapter");
    let q_dim = adapter.geometry().q_rot_len();
    let make_seq = |toks: &[(usize, f32)]| -> Vec<Vec<f32>> {
        let mut flat = vec![0.0_f32; toks.len() * q_dim];
        for (t, &(slot, amp)) in toks.iter().enumerate() {
            flat[t * q_dim] += 0.65; // Shared BOS/sink framing on dim 0.
            flat[t * q_dim + (slot % q_dim)] += amp;
        }
        vec![flat; 4]
    };

    let entries = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact Alpha memory alignment".to_string(),
            energy: 2.0,
            seq_len: 3,
            per_layer_token_qs: make_seq(&[(2, 1.0), (3, 0.9), (4, 0.8)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact Beta cache budget".to_string(),
            energy: 2.0,
            seq_len: 3,
            per_layer_token_qs: make_seq(&[(6, 1.0), (7, 0.9), (8, 0.8)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact Gamma phase clamp".to_string(),
            energy: 2.0,
            seq_len: 3,
            per_layer_token_qs: make_seq(&[(10, 1.0), (11, 0.9), (12, 0.8)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact Delta storage layout".to_string(),
            energy: 2.0,
            seq_len: 3,
            per_layer_token_qs: make_seq(&[(14, 1.0), (15, 0.9), (16, 0.8)]),
            token_include_mask: None,
        },
    ];

    // Startup prior samples share BOS dim 0 plus a technical direction on dim 2 (overlapping Fact
    // Alpha) alongside distinct neutral topic dimensions so prior PCA extracts both dim 0 and dim 2.
    let prior_samples: Vec<TokenSequenceEntry> = (0..8)
        .map(|i| TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: if i % 2 == 0 {
                format!("Neutral technical probe {i}")
            } else {
                format!("\u{4E2D}\u{7ACB}\u{7684}\u{306A}\u{6280}\u{8853}\u{30D7}\u{30ED}\u{30FC}\u{30D6}{i}")
            },
            energy: 1.0,
            seq_len: 3,
            per_layer_token_qs: make_seq(&[(2, 0.55), (20 + (i % 6), 0.90), (26 + (i % 5), 0.85)]),
            token_include_mask: None,
        })
        .collect();
    let prior = ModelBaseAttractorProfile::extract(
        &adapter,
        &prior_samples,
        SequenceActivationPooler::default(),
        0.55,
    )
    .expect("extract prior");

    let batch = CompiledTokenCorpus::compile_with_base_prior(
        &adapter,
        &entries,
        SequenceActivationPooler::default(),
        2,
        1,
        Some(&prior),
    )
    .expect("batch compile");
    let mut distiller = StreamingCorpusDistiller::new(&adapter, StreamingDistillConfig::default());
    for e in &entries {
        let _ = distiller
            .push_captured(
                &adapter,
                e.role,
                e.text.as_str(),
                e.energy,
                CapturedTokenSequence::new(e.seq_len, e.per_layer_token_qs.clone()),
            )
            .expect("stream push");
    }
    let (streamed, _) = distiller
        .finalize(&adapter, Some(&prior))
        .expect("stream finalize");

    assert!(batch.null_calibration.is_some());
    assert!(streamed.null_calibration.is_some());

    // Query targeting Fact Alpha (dims 2, 3, 4) must recall anchor #1 cleanly in BOTH paths.
    let q_alpha = make_seq(&[(2, 1.0), (3, 0.9), (4, 0.8)]);
    let rec_batch = batch
        .recall_token_sequence(&adapter, &q_alpha, 3)
        .expect("batch recall");
    let rec_stream = streamed
        .recall_token_sequence(&adapter, &q_alpha, 3)
        .expect("stream recall");
    assert_eq!(
        rec_batch.recalled_facts.first().map(|(e, _)| e.anchor_id),
        Some(1),
        "batch path must protect Fact Alpha discrimination under overlapping base_prior"
    );
    assert_eq!(
        rec_stream.recalled_facts.first().map(|(e, _)| e.anchor_id),
        Some(1),
        "streaming path must protect Fact Alpha discrimination under overlapping base_prior"
    );

    // Pure BOS/sink OOD query (only dim 0 + unrelated dims 30, 31) must be rejected in BOTH paths.
    let q_bos_ood = make_seq(&[(30, 0.15), (31, 0.15)]);
    let ood_batch = batch
        .recall_token_sequence(&adapter, &q_bos_ood, 2)
        .expect("batch ood");
    let ood_stream = streamed
        .recall_token_sequence(&adapter, &q_bos_ood, 2)
        .expect("stream ood");
    assert!(ood_batch.recalled_facts.is_empty());
    assert_eq!(ood_batch.lacuna_state, LacunaBoundaryState::LiminalExterior);
    assert!(ood_stream.recalled_facts.is_empty());
    assert_eq!(
        ood_stream.lacuna_state,
        LacunaBoundaryState::LiminalExterior
    );
}

#[test]
fn default_startup_prior_probes_external_txt_is_bit_identical_and_parity_balanced() {
    use liminika_core::foundation::compute_crc32_ieee;

    assert_eq!(DEFAULT_STARTUP_PRIOR_PROBES.len(), 24);
    let mut even_ascii = 0usize;
    let mut even_non_ascii = 0usize;
    let mut odd_ascii = 0usize;
    let mut odd_non_ascii = 0usize;

    for (idx, probe) in DEFAULT_STARTUP_PRIOR_PROBES.iter().enumerate() {
        assert!(!probe.is_empty(), "probe {idx} must be non-empty");
        assert!(
            !probe.contains('\r') && !probe.contains('\n'),
            "probe {idx} must not contain CR or LF"
        );
        let is_ascii = probe.is_ascii();
        match (idx % 2 == 0, is_ascii) {
            (true, true) => even_ascii += 1,
            (true, false) => even_non_ascii += 1,
            (false, true) => odd_ascii += 1,
            (false, false) => odd_non_ascii += 1,
        }
    }

    assert_eq!(even_ascii, 6);
    assert_eq!(even_non_ascii, 6);
    assert_eq!(odd_ascii, 6);
    assert_eq!(odd_non_ascii, 6);

    let joined = DEFAULT_STARTUP_PRIOR_PROBES.join("\n");
    assert_eq!(joined.len(), 1924);
    assert_eq!(compute_crc32_ieee(joined.as_bytes()), 0xe3a6_3b06);
}
