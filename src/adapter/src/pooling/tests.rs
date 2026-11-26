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
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Entity B".to_string(),
            energy: 2.0,
            seq_len: 7,
            per_layer_token_qs: make_seq(6, 2),
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Entity C".to_string(),
            energy: 2.0,
            seq_len: 7,
            per_layer_token_qs: make_seq(6, 3),
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Entity D".to_string(),
            energy: 2.0,
            seq_len: 7,
            per_layer_token_qs: make_seq(6, 4),
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
    };
    let e1 = TokenSequenceEntry {
        role: AttractorRole::Fact,
        text: "Shared 10 and unique 40".to_string(),
        energy: 2.0,
        seq_len: 2,
        per_layer_token_qs: make_seq_with_tokens(&[10, 40]),
    };
    let e2 = TokenSequenceEntry {
        role: AttractorRole::Fact,
        text: "Shared 10 and unique 55".to_string(),
        energy: 2.0,
        seq_len: 2,
        per_layer_token_qs: make_seq_with_tokens(&[10, 55]),
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
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Shared Sibling (10, 20)".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[10, 20]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Specialist Alpha (35, 39)".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[35, 39]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Specialist Beta (55, 59)".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[55, 59]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Guard,
            text: "Forbidden Guard (75, 79)".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq_with_tokens(&[75, 79]),
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
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime A sample 2".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(20, 0.80), (21, 0.65)]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime A sample 3".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(20, 0.90), (21, 0.75)]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime B sample 1".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(60, 0.85), (61, 0.70)]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime B sample 2".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(60, 0.80), (61, 0.65)]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Persona,
            text: "Base regime B sample 3".to_string(),
            energy: 1.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(60, 0.90), (61, 0.75)]),
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
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Fact Beta".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(80, 0.85), (81, 0.65)]),
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
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Initial-split Fact".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(40, 0.9), (12, 0.7)]),
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
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Other Fact".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(50, 0.95), (90, 0.85)]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Guard,
            text: "Contrastive Guard echoing entity".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(30, 0.95), (110, 0.85)]),
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
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Domain fact beta".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(0, 0.30), (20, 1.0)]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Domain fact gamma".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(0, 0.30), (30, 1.0)]),
        },
        TokenSequenceEntry {
            role: AttractorRole::Fact,
            text: "Domain fact delta".to_string(),
            energy: 2.0,
            seq_len: 2,
            per_layer_token_qs: make_seq(&[(0, 0.30), (40, 1.0)]),
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
