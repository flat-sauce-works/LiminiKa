// name: src/adapter/src/pooling/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{LacunaBoundaryState, RopeGeometry, RopeLayout, TextAttractorEntry};

const fn sample_gqa_partial_rope() -> RopeGeometry {
    RopeGeometry::new(RopeLayout::HalfSplit, 128, 96, 24, 8, 32, 0xFFFF_FFFF)
}

#[test]
fn novelty_sif_pooling_resists_attention_sink_norm_spike_and_boilerplate() {
    let dim = 8;
    // Reference represents boilerplate direction [1, 0, 0, ...].
    let ref_dir = vec![1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // Sequence:
    //   token 0: huge 25x BOS/sink spike along boilerplate axis [25, 0, ...]
    //   token 1..3: normal boilerplate tokens [1, 0, ...]
    //   token 4: rare entity signal along orthogonal axis [0, 0, 1, 0, ...]
    let mut seq = Vec::new();
    seq.extend_from_slice(&[25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    for _ in 0..3 {
        seq.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    }
    seq.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let pooler = SequenceActivationPooler::default();
    let pooled_sif = pooler
        .pool_layer(&seq, 5, dim, Some(&ref_dir))
        .expect("sif pool");
    let pooled_mean = SequenceActivationPooler::uniform_unit_mean(&seq, 5, dim).expect("mean pool");

    // Novelty weighting must amplify the distinctive entity coordinate (`idx = 2`) well above
    // uniform mean, while the 25x magnitude sink at token 0 cannot blow up `idx = 0`.
    assert!(
        pooled_sif[2] > pooled_mean[2] * 2.5,
        "expected SIF novelty boost on idx=2: sif={} mean={}",
        pooled_sif[2],
        pooled_mean[2]
    );
    assert!(pooled_sif[2] > pooled_sif[0]);
}

#[test]
fn novelty_sif_pooling_is_token_order_invariant() {
    let dim = 4;
    let ref_dir = vec![1.0_f32, 0.0, 0.0, 0.0];
    let a = [1.0_f32, 0.1, 0.0, 0.0];
    let b = [0.2_f32, 0.9, 0.3, 0.0];
    let c = [0.1_f32, 0.0, 0.0, 1.0];

    let mut fwd = Vec::new();
    fwd.extend_from_slice(&a);
    fwd.extend_from_slice(&b);
    fwd.extend_from_slice(&c);

    let mut rev = Vec::new();
    rev.extend_from_slice(&c);
    rev.extend_from_slice(&a);
    rev.extend_from_slice(&b);

    let pooler = SequenceActivationPooler::default();
    let p1 = pooler.pool_layer(&fwd, 3, dim, Some(&ref_dir)).expect("p1");
    let p2 = pooler.pool_layer(&rev, 3, dim, Some(&ref_dir)).expect("p2");
    for (x, y) in p1.iter().zip(p2.iter()) {
        assert!((x - y).abs() < 1e-6, "order variance detected: {x} vs {y}");
    }
}

#[test]
fn compiled_token_corpus_recovers_diluted_entity_while_rejecting_ood() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let num_layers = adapter.geometry().num_layers;
    let dim = adapter.geometry().q_rot_len();

    let make_seq = |boilerplate_tokens: usize, entity_dim: usize| -> Vec<Vec<f32>> {
        let mut layer_buf = vec![0.0_f32; (boilerplate_tokens + 1) * dim];
        for t in 0..boilerplate_tokens {
            layer_buf[t * dim] = 1.0;
        }
        let e_off = boilerplate_tokens * dim;
        layer_buf[e_off + entity_dim] = 1.0;
        vec![layer_buf; num_layers]
    };

    let entries = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Entity A".to_string(),
            energy: 2.0,
            seq_len: 7,
            per_layer_token_qs: make_seq(6, 1),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Entity B".to_string(),
            energy: 2.0,
            seq_len: 7,
            per_layer_token_qs: make_seq(6, 2),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Entity C".to_string(),
            energy: 2.0,
            seq_len: 7,
            per_layer_token_qs: make_seq(6, 3),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Entity D".to_string(),
            energy: 2.0,
            seq_len: 7,
            per_layer_token_qs: make_seq(6, 4),
            token_include_mask: None,
        },
    ];

    let compiled = CompiledTokenCorpus::compile(
        &adapter,
        &entries,
        SequenceActivationPooler::default(),
        1,
        1,
    )
    .expect("compile");

    // In-domain diluted query targeting Entity C (entity_dim = 3) recovers #3.
    let query_c = make_seq(6, 3);
    let recall_c = compiled
        .recall_token_sequence(&adapter, &query_c, 7)
        .expect("recall C");
    assert_eq!(
        recall_c.recalled_facts.first().map(|(e, _)| e.anchor_id),
        Some(3)
    );

    // Completely out-of-domain entity (entity_dim = 500) must fall in Liminal.
    let query_ood = make_seq(6, 500);
    let recall_ood = compiled
        .recall_token_sequence(&adapter, &query_ood, 7)
        .expect("recall OOD");
    assert_eq!(
        recall_ood.lacuna_state,
        LacunaBoundaryState::LiminalExterior
    );
    assert!(recall_ood.recalled_facts.is_empty());
}

#[test]
fn stratum_signal_maps_to_core_gateway_blanket_frame() {
    use crate::{
        RecallBoundaryMetrics, RecallGateDecision, StratumProbeTelemetry, StratumSignal,
        TextAttractorEntry,
    };
    use liminika_core::stigmergy::StigmergicActionDecision;

    let entries = vec![
        TextAttractorEntry {
            anchor_id: 7,
            role: AttractorRole::Fact,
            text: "Fact 7".to_string(),
            per_layer_q7: vec![[12_i8; 64]],
            energy: 2.0,
            residual_sig: vec![],
        },
        TextAttractorEntry {
            anchor_id: 101,
            role: AttractorRole::Persona,
            text: "Persona 101".to_string(),
            per_layer_q7: vec![[5_i8; 64]],
            energy: 1.5,
            residual_sig: vec![],
        },
    ];
    let signal = StratumSignal {
        candidate_scores: vec![0.45, 0.30],
        local_token_scores: vec![0.60, 0.25],
        q_to_d_scores: vec![],
        d_to_q_scores: vec![],
        token_deposits: vec![],
        linked_fact_indices: vec![],
        decomposition_triggered: false,
        local_lacuna_depths: vec![0.0, 0.0],
        stigmergic_weights: vec![1.2, 1.0],
        boundary_metrics: RecallBoundaryMetrics {
            top1_fact_score: 0.45,
            top2_fact_score: 0.10,
            top1_persona_score: 0.30,
            max_guard_score: -0.2,
            guard_conflict_margin: -0.65,
        },
        gate_decision: RecallGateDecision {
            lacuna_state: LacunaBoundaryState::InsideBasin,
            suppress_fact_injection: false,
        },
        action_decision: StigmergicActionDecision::CrystallizeMacroBasin,
        gateway_frame: Default::default(),
        swarm_report: Default::default(),
        local_channel_preferred: false,
        telemetry: StratumProbeTelemetry::default(),
    };

    let mapped = signal.map_to_gateway(&entries, 0);
    assert_eq!(mapped.fact.map(|f| f.anchor_id), Some(7));
    assert_eq!(mapped.persona.map(|p| p.anchor_id), Some(101));
    assert_eq!(mapped.frame.target_anchor_id, 7);
}

#[test]
fn compiled_token_corpus_assigns_lower_specificity_to_shared_tokens() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let geom = adapter.geometry();
    let dim = geom.q_rot_len();
    let num_layers = geom.num_layers;

    let make_seq_with_tokens = |token_dims: &[usize]| -> Vec<Vec<f32>> {
        let s = token_dims.len();
        let mut per_layer = vec![vec![0.0_f32; s * dim]; num_layers];
        for layer_buf in &mut per_layer {
            for (t, &d_idx) in token_dims.iter().enumerate() {
                let pos = d_idx % dim;
                layer_buf[t * dim + pos] = 1.0;
                layer_buf[t * dim + ((pos + 1) % dim)] = 0.25;
            }
        }
        per_layer
    };

    // Dimension `10` is shared across 3 entries; dimension `25` appears only in entry 0.
    let e0 = TokenSequenceEntry {
        role: AttractorRole::Fact,
        text: "Shared 10 and unique 25".to_string(),
        energy: 2.0,
        seq_len: 2,
        per_layer_token_qs: make_seq_with_tokens(&[10, 25]),
        token_include_mask: None,
    };
    let e1 = TokenSequenceEntry {
        role: AttractorRole::Fact,
        text: "Shared 10 and unique 40".to_string(),
        energy: 2.0,
        seq_len: 2,
        per_layer_token_qs: make_seq_with_tokens(&[10, 40]),
        token_include_mask: None,
    };
    let e2 = TokenSequenceEntry {
        role: AttractorRole::Fact,
        text: "Shared 10 and unique 55".to_string(),
        energy: 2.0,
        seq_len: 2,
        per_layer_token_qs: make_seq_with_tokens(&[10, 55]),
        token_include_mask: None,
    };
    let compiled = CompiledTokenCorpus::compile(
        &adapter,
        &[e0, e1, e2],
        SequenceActivationPooler::default(),
        2,
        1,
    )
    .expect("compile");
    assert_eq!(compiled.entry_token_weights.len(), 3);
    let min_w0 = compiled.entry_token_weights[0]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min);
    let max_w0 = compiled.entry_token_weights[0]
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    assert!(
        max_w0 > min_w0 * 1.25,
        "unique token must have strictly higher specificity weight than shared token: \
         min={min_w0} max={max_w0}"
    );
}

#[test]
fn compiled_token_corpus_decomposes_bimodal_compound_query_end_to_end() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let geom = adapter.geometry();
    let dim = geom.q_rot_len();
    let num_layers = geom.num_layers;

    let make_seq_with_tokens = |token_dims: &[usize]| -> Vec<Vec<f32>> {
        let s = token_dims.len();
        let mut per_layer = vec![vec![0.0_f32; s * dim]; num_layers];
        for layer_buf in &mut per_layer {
            for (t, &d_idx) in token_dims.iter().enumerate() {
                let pos = d_idx % dim;
                layer_buf[t * dim + pos] = 1.0;
                layer_buf[t * dim + ((pos + 2) % dim)] = 0.25;
            }
        }
        per_layer
    };

    let entries = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Overview Hub (10, 12, 14)".to_string(),
            energy: 2.0,
            seq_len: 3,
            per_layer_token_qs: make_seq_with_tokens(&[10, 12, 14]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Shared Sibling (10, 20)".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[10, 20]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Specialist Alpha (35, 39)".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[35, 39]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Specialist Beta (55, 59)".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[55, 59]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Guard,
            text: "Forbidden Guard (75, 79)".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[75, 79]),
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

    // Query asks about shared domain framing (10, 10) plus both Alpha (35) and Beta (55).
    let q_seq = make_seq_with_tokens(&[10, 10, 35, 55]);
    let recall = compiled
        .recall_token_sequence(&adapter, &q_seq, 4)
        .expect("bimodal recall");
    let recalled_ids: Vec<u32> = recall
        .recalled_facts
        .iter()
        .map(|(e, _)| e.anchor_id)
        .collect();
    assert_eq!(recalled_ids.len(), 2, "got {recalled_ids:?}");
    assert!(
        recalled_ids.contains(&3) && recalled_ids.contains(&4),
        "expected Alpha(#3) and Beta(#4), got {recalled_ids:?}"
    );
}

#[test]
fn model_base_attractor_profile_partitions_sub_basins_and_orthogonalizes_corpus_prior() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let geom = adapter.geometry();
    let dim = geom.q_rot_len();
    let num_layers = geom.num_layers;

    let make_seq = |tok_specs: &[(usize, f32)]| -> Vec<Vec<f32>> {
        let s = tok_specs.len();
        let mut per_layer = vec![vec![0.0_f32; s * dim]; num_layers];
        for layer_buf in &mut per_layer {
            for (t, &(d_idx, framing_strength)) in tok_specs.iter().enumerate() {
                layer_buf[t * dim] = framing_strength;
                let pos = d_idx % dim;
                layer_buf[t * dim + pos] = 1.0;
            }
        }
        per_layer
    };

    // Six baseline unsteered samples spanning two distinct model regimes (`20` and `60`)
    // plus shared model framing along dimension `0`.
    let base_samples = vec![
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime A sample 1".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(20, 0.85), (21, 0.70)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime A sample 2".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(20, 0.80), (21, 0.65)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime A sample 3".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(20, 0.90), (21, 0.75)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime B sample 1".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(60, 0.85), (61, 0.70)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime B sample 2".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(60, 0.80), (61, 0.65)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime B sample 3".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(60, 0.90), (61, 0.75)]),
            token_include_mask: None,
        },
    ];

    let prior = ModelBaseAttractorProfile::extract(
        &adapter,
        &base_samples,
        SequenceActivationPooler::default(),
        0.50,
    )
    .expect("extract prior");
    assert_eq!(prior.sample_count, 6);
    assert_eq!(
        prior.sub_attractors.len(),
        2,
        "expected two distinct base sub-attractors, got {}",
        prior.sub_attractors.len()
    );
    assert!(
        !prior.pooled_common_mode_basis.is_empty(),
        "expected at least one global base-model framing component"
    );

    // Compile external facts with the base-model prior and verify recall selects target #2.
    let facts = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact Alpha".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(30, 0.85), (31, 0.65)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact Beta".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(80, 0.85), (81, 0.65)]),
            token_include_mask: None,
        },
    ];
    let compiled = CompiledTokenCorpus::compile_with_base_prior(
        &adapter,
        &facts,
        SequenceActivationPooler::default(),
        2,
        1,
        Some(&prior),
    )
    .expect("compile with base prior");
    let query = make_seq(&[(80, 0.90), (81, 0.75)]);
    let recall = compiled
        .recall_token_sequence(&adapter, &query, 2)
        .expect("recall with base prior");
    assert_eq!(
        recall.recalled_facts.first().map(|(e, _)| e.anchor_id),
        Some(2)
    );
}

#[test]
fn append_entry_token_observations_preserves_pooled_sigs_matches_sequential_and_is_atomic() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let num_layers = adapter.geometry().num_layers;
    let dim = adapter.geometry().q_rot_len();
    let make_seq = |toks: &[(usize, f32)]| -> Vec<Vec<f32>> {
        let mut per_layer = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut flat = vec![0.0_f32; toks.len() * dim];
            for (t, &(slot, amp)) in toks.iter().enumerate() {
                flat[t * dim + slot] = amp;
                flat[t * dim + (slot + 1) % dim] = 0.20;
            }
            per_layer.push(flat);
        }
        per_layer
    };

    let entries = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Overview Fact".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(10, 0.9), (12, 0.7)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Initial-split Fact".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(40, 0.9), (12, 0.7)]),
            token_include_mask: None,
        },
    ];

    let mut compiled_batch = CompiledTokenCorpus::compile(
        &adapter,
        &entries,
        SequenceActivationPooler::default(),
        2,
        1,
    )
    .expect("compile");
    let mut compiled_seq = compiled_batch.clone();
    let before_sig = compiled_batch.corpus.entries()[1].residual_sig.clone();

    let obs_entry0 = EntryTokenObservation {
        entry_idx: 0,
        seq_len: 2,
        per_layer_token_qs: make_seq(&[(90, 0.92), (92, 0.85)]),
    };
    let obs_entry1 = EntryTokenObservation {
        entry_idx: 1,
        seq_len: 2,
        per_layer_token_qs: make_seq(&[(70, 0.95), (72, 0.90)]),
    };

    // Batch append with an invalid trailing entry must fail atomically without mutating state.
    let bad_batch = vec![
        obs_entry1.clone(),
        EntryTokenObservation {
            entry_idx: 9,
            seq_len: 2,
            per_layer_token_qs: obs_entry1.per_layer_token_qs.clone(),
        },
    ];
    assert!(
        compiled_batch
            .append_entry_token_observations(&adapter, &bad_batch)
            .is_err()
    );
    assert_eq!(compiled_batch.entry_token_sigs[1].len(), 2);

    // Valid batch append matches sequential single-item application bit-for-bit.
    compiled_batch
        .append_entry_token_observations(&adapter, &[obs_entry0.clone(), obs_entry1.clone()])
        .expect("batch append");
    compiled_seq
        .append_entry_token_observations(&adapter, &[obs_entry0])
        .expect("seq append 0");
    compiled_seq
        .append_entry_token_observations(&adapter, std::slice::from_ref(&obs_entry1))
        .expect("seq append 1");

    assert_eq!(compiled_batch.corpus.entries()[1].residual_sig, before_sig);
    assert_eq!(
        compiled_batch.entry_token_sigs,
        compiled_seq.entry_token_sigs
    );
    assert_eq!(
        compiled_batch.entry_token_weights,
        compiled_seq.entry_token_weights
    );
    assert_eq!(
        compiled_batch.entry_token_hub_bias,
        compiled_seq.entry_token_hub_bias
    );
    assert_eq!(
        compiled_batch.global_token_hub_mean.to_bits(),
        compiled_seq.global_token_hub_mean.to_bits()
    );
    assert_eq!(compiled_batch.entry_token_sigs[0].len(), 4);
    assert_eq!(compiled_batch.entry_token_sigs[1].len(), 4);

    let obs = compiled_batch
        .build_local_observation(&adapter, &obs_entry1.per_layer_token_qs, 2)
        .expect("obs");
    assert!(
        obs.q_to_d_scores[1] > obs.q_to_d_scores[0],
        "expected entry 1 observation match ({}) > entry 0 ({})",
        obs.q_to_d_scores[1],
        obs.q_to_d_scores[0]
    );
}

#[test]
fn same_role_token_specificity_ignores_contrastive_guard_echoes() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let num_layers = adapter.geometry().num_layers;
    let dim = adapter.geometry().q_rot_len();
    let make_seq = |toks: &[(usize, f32)]| -> Vec<Vec<f32>> {
        let mut per_layer = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut flat = vec![0.0_f32; toks.len() * dim];
            for (t, &(slot, amp)) in toks.iter().enumerate() {
                flat[t * dim + slot] = amp;
                flat[t * dim + (slot + 1) % dim] = 0.15;
            }
            per_layer.push(flat);
        }
        per_layer
    };

    // Entry 0 (Fact) and Entry 2 (Guard) share entity token slot 30, while Entry 1 (Fact)
    // shares token slot 50 with nothing else; Guard echo must not dilute Entry 0's specificity.
    let entries = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Entity Fact".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(30, 0.95), (70, 0.85)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Other Fact".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(50, 0.95), (90, 0.85)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Guard,
            text: "Contrastive Guard echoing entity".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(30, 0.95), (110, 0.85)]),
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
    assert!(
        compiled.entry_token_weights[0].iter().all(|&w| w > 0.85),
        "Guard echo should not penalize Fact token specificity: {:?}",
        compiled.entry_token_weights[0]
    );
}

#[test]
fn fused_exact_token_calibrator_matches_naive_two_pass_bit_for_bit() {
    fn naive_two_pass(
        entry_token_sigs: &[Vec<Vec<f32>>],
        entries: &[TextAttractorEntry],
    ) -> TokenCalibrationStats {
        let n_entries = entry_token_sigs.len();
        let mut entry_token_weights = Vec::with_capacity(n_entries);
        for i in 0..n_entries {
            let role_i = entries.get(i).map(|e| e.role);
            let mut weights_i = Vec::with_capacity(entry_token_sigs[i].len());
            for d_tok in &entry_token_sigs[i] {
                let mut other_sim_sq = 0.0_f32;
                for (j, other_toks) in entry_token_sigs.iter().enumerate() {
                    let same_role = role_i
                        .zip(entries.get(j).map(|e| e.role))
                        .is_none_or(|(ri, rj)| ri == rj);
                    if i == j || other_toks.is_empty() || !same_role {
                        continue;
                    }
                    let best_j = other_toks
                        .iter()
                        .map(|od| TextAttractorCorpus::cosine_f32(d_tok, od))
                        .fold(0.0_f32, f32::max);
                    let excess = (best_j - 0.30).max(0.0);
                    other_sim_sq += excess * excess;
                }
                weights_i.push((1.0 / (1.0 + 3.0 * other_sim_sq)).clamp(0.10, 1.0));
            }
            entry_token_weights.push(weights_i);
        }
        let mut entry_token_hub_bias = vec![0.0_f32; n_entries];
        let mut global_token_hub_mean = 0.0_f32;
        if n_entries >= 2 {
            let denom = (n_entries - 1) as f32;
            for j in 0..n_entries {
                let mut sum = 0.0_f32;
                for i in 0..n_entries {
                    if i == j {
                        continue;
                    }
                    sum += AdaptiveTokenSpecificityCalibrator::compute_unweighted_maxsim(
                        &entry_token_sigs[i],
                        &entry_token_sigs[j],
                    );
                }
                entry_token_hub_bias[j] = sum / denom;
            }
            global_token_hub_mean = entry_token_hub_bias.iter().sum::<f32>() / (n_entries as f32);
        }
        TokenCalibrationStats {
            entry_token_weights,
            entry_token_hub_bias,
            global_token_hub_mean,
        }
    }

    let dim = 16;
    let n_entries = 12;
    let mut entries = Vec::with_capacity(n_entries);
    let mut sigs = Vec::with_capacity(n_entries);
    for i in 0..n_entries {
        let role = match i % 3 {
            0 => AttractorRole::Fact,
            1 => AttractorRole::Persona,
            _ => AttractorRole::Guard,
        };
        entries.push(TextAttractorEntry {
            anchor_id: (i + 1) as u32,
            role,
            text: format!("Entry {i}"),
            per_layer_q7: Vec::new(),
            energy: 1.5,
            residual_sig: Vec::new(),
        });
        let tok_count = if i == 5 { 0 } else { 3 + (i % 3) };
        let mut entry_toks = Vec::with_capacity(tok_count);
        for t in 0..tok_count {
            let mut v = vec![0.0_f32; dim];
            v[0] = 0.45;
            v[(i + t) % dim] += 0.85;
            v[(3 * i + 2 * t + 1) % dim] -= 0.35;
            entry_toks.push(v);
        }
        sigs.push(entry_toks);
    }

    let fused =
        AdaptiveTokenSpecificityCalibrator::default().calibrate_token_stats(&sigs, &entries);
    let naive = naive_two_pass(&sigs, &entries);

    assert_eq!(
        fused.entry_token_weights.len(),
        naive.entry_token_weights.len()
    );
    for (row_f, row_n) in fused
        .entry_token_weights
        .iter()
        .zip(naive.entry_token_weights.iter())
    {
        assert_eq!(row_f.len(), row_n.len());
        for (&wf, &wn) in row_f.iter().zip(row_n.iter()) {
            assert_eq!(wf.to_bits(), wn.to_bits());
        }
    }
    for (&hf, &hn) in fused
        .entry_token_hub_bias
        .iter()
        .zip(naive.entry_token_hub_bias.iter())
    {
        assert_eq!(hf.to_bits(), hn.to_bits());
    }
    assert_eq!(
        fused.global_token_hub_mean.to_bits(),
        naive.global_token_hub_mean.to_bits()
    );
}

#[test]
fn adaptive_token_calibrator_scales_deterministically_above_exact_limit() {
    let n_entries = 80;
    assert!(n_entries > EXACT_TOKEN_STATS_LIMIT);
    let dim = 96;
    let mut entries = Vec::with_capacity(n_entries);
    let mut sigs = Vec::with_capacity(n_entries);

    for i in 0..n_entries {
        let role = if i < 68 {
            AttractorRole::Fact
        } else if i < 74 {
            AttractorRole::Persona
        } else {
            AttractorRole::Guard
        };
        entries.push(TextAttractorEntry {
            anchor_id: (i + 1) as u32,
            role,
            text: format!("Synthetic entry {i}"),
            per_layer_q7: Vec::new(),
            energy: 1.5,
            residual_sig: Vec::new(),
        });
        // Token 0 is shared across every entry (axis 0); Token 1 is entry-specific (axis 1 + i).
        let mut shared_tok = vec![0.0_f32; dim];
        shared_tok[0] = 1.0;
        let mut unique_tok = vec![0.0_f32; dim];
        unique_tok[1 + i] = 1.0;
        sigs.push(vec![shared_tok, unique_tok]);
    }

    let calibrator = AdaptiveTokenSpecificityCalibrator::default();
    let run1 = calibrator.calibrate_token_stats(&sigs, &entries);
    let run2 = calibrator.calibrate_token_stats(&sigs, &entries);

    // Determinism: two independent runs must be bit-identical without any RNG.
    assert_eq!(run1, run2);
    assert!(run1.global_token_hub_mean.is_finite());
    assert!(!run1.global_token_hub_mean.is_nan());
    assert_eq!(run1.entry_token_weights.len(), n_entries);
    assert_eq!(run1.entry_token_hub_bias.len(), n_entries);
    for (i, ws) in run1.entry_token_weights.iter().enumerate() {
        assert_eq!(ws.len(), 2);
        for &w in ws {
            assert!(w.is_finite() && (0.10..=1.0).contains(&w));
        }
        assert!(
            ws[1] > ws[0],
            "entry {i}: unique token weight ({}) should exceed shared token weight ({})",
            ws[1],
            ws[0]
        );
    }
    for &hb in &run1.entry_token_hub_bias {
        assert!(hb.is_finite() && !hb.is_nan());
    }
}

#[test]
fn non_leaking_base_prior_null_calibration_raises_floor_and_attenuates_common_mode() {
    let adapter = ModelSteeringAdapter::new(sample_gqa_partial_rope()).expect("init");
    let geom = adapter.geometry();
    let dim = geom.q_rot_len();
    let num_layers = geom.num_layers;

    // Build 4 orthogonal domain Fact entries on dimensions 10..42, plus a shared common-mode
    // carrier on dimension 0 that represents a generic base-model attractor.
    let make_seq = |dims: &[(usize, f32)]| -> Vec<Vec<f32>> {
        let mut per_layer = vec![vec![0.0_f32; 2 * dim]; num_layers];
        for layer_buf in &mut per_layer {
            for t in 0..2 {
                for &(d, val) in dims {
                    layer_buf[t * dim + ((d + t) % dim)] += val;
                }
            }
        }
        per_layer
    };

    let entries = vec![
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Domain fact alpha".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(0, 0.30), (10, 1.0)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Domain fact beta".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(0, 0.30), (20, 1.0)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Domain fact gamma".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(0, 0.30), (30, 1.0)]),
            token_include_mask: None,
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Domain fact delta".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(0, 0.30), (40, 1.0)]),
            token_include_mask: None,
        },
    ];

    // 8 neutral base-prior calibration samples aligned strongly with common-mode dim 0 plus
    // distinct neutral directions (60 + 4*i), completely disjoint from domain facts 10..40.
    let neutral_entries: Vec<TokenSequenceEntry> = (0..8)
        .map(|i| TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: format!("Neutral prior probe {i}"),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(0, 0.90), (60 + 4 * i, 0.45)]),
            token_include_mask: None,
        })
        .collect();
    let base_prior = ModelBaseAttractorProfile::extract(
        &adapter,
        &neutral_entries,
        SequenceActivationPooler::default(),
        0.92,
    )
    .expect("fit prior");
    assert_eq!(base_prior.calibration_samples.len(), 8);

    let compiled_no_prior = CompiledTokenCorpus::compile(
        &adapter,
        &entries,
        SequenceActivationPooler::default(),
        2,
        1,
    )
    .expect("compile no prior");
    let compiled_with_prior = CompiledTokenCorpus::compile_with_base_prior(
        &adapter,
        &entries,
        SequenceActivationPooler::default(),
        2,
        1,
        Some(&base_prior),
    )
    .expect("compile with prior");

    let env = compiled_with_prior
        .null_calibration
        .expect("null calibration envelope must be populated");
    assert_eq!(
        env.sample_count, 8,
        "2-fold cross-fitting must evaluate all 8 held-out base-prior samples"
    );
    assert!(
        compiled_with_prior.recall_policy.min_coherence
            >= compiled_no_prior.recall_policy.min_coherence - 1e-6,
        "null calibration must not lower min_coherence below corpus floor"
    );

    // Pure common-mode OOD query (almost all energy on dim 0) must yield low `query_ortho_ratio`
    // and be suppressed into `LiminalExterior`.
    let common_mode_query = make_seq(&[(0, 1.0), (10, 0.05)]);
    let recall = compiled_with_prior
        .recall_token_sequence(&adapter, &common_mode_query, 2)
        .expect("common mode recall");
    assert!(
        recall.telemetry.query_ortho_ratio < 0.35,
        "common-mode OOD query must have low residual ortho ratio, got {}",
        recall.telemetry.query_ortho_ratio
    );
    assert_eq!(recall.lacuna_state, LacunaBoundaryState::LiminalExterior);
    assert!(
        recall.recalled_facts.is_empty(),
        "common-mode OOD query must not recall domain facts"
    );
}

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
                format!("中立的な日本語標本{i}")
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
    let ja_sentence = "翠紋解析器は蒼穹観測塔の地下第4層に設置された無電力の光学位相計である。";
    let ja_span = extract_leading_alias_word(ja_sentence).expect("ja span");
    assert_eq!(ja_span.chars().count(), 8);
    assert_eq!(ja_span, "翠紋解析器は蒼穹");
    assert!(ja_span.chars().count() < ja_sentence.chars().count() / 2);

    // Leading CJK brackets/quotes and punctuation delimiters.
    assert_eq!(
        extract_leading_alias_word("「東京」は日本の首都です。"),
        Some("東京")
    );
    assert_eq!(
        extract_leading_alias_word("（設計）方針を確認する"),
        Some("設計")
    );
    assert_eq!(extract_leading_alias_word("紅玉、翠紋、白磁"), Some("紅玉"));
    assert_eq!(extract_leading_alias_word("「」"), None);

    // Mixed ASCII + CJK without spaces stays bounded on UTF-8 char boundaries.
    let mixed = "HTTP通信プロトコルの仕様概要";
    let mixed_span = extract_leading_alias_word(mixed).expect("mixed span");
    assert_eq!(mixed_span.chars().count(), 8);
    assert_eq!(mixed_span, "HTTP通信プロ");
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
                format!("中立プローブ標本{i}")
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
                format!("中立的な技術プローブ{i}")
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
