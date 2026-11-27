// name: examples/candle_eval/src/equilibrium_bench.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! End-to-end cross-restart equilibrium persistence benchmark (`--equilibrium-bench`).
//!
//! Verifies on live LLM weights:
//! 1. Session 1: live steered turn + 576 B `RuntimeContext` snapshot export + crystallized FIFO
//!    drain into `EquilibriumState` + `.lmnkc` persistence (`LMNK_SECTION_RUNTIME_SNAPSHOT`).
//! 2. 100-generation flat `join` convergence (non-thermal-death, healthy forgetting, non-lock-in
//!    plasticity, bounded `.lmnkc` size, zero NaN/Inf).
//! 3. Full restart simulation (`drop(session1)` + `model.clear_cache()`).
//! 4. Session 2: `.lmnkc` restore, host `Spine` pin-floor preservation, Session 1 anchor recall,
//!    empty-KV `SelfRef` response vs `OOD` (`Unobserved`) non-interference.
//! 5. Foreign `topology_crc32` rejection + zero-bit immutable `.lmnkp` source rebake verification.

use std::path::Path;
use std::time::Instant;

use liminika_adapter::{
    AdaptiveGopCutDetector, AnchorPagerConfig, AttractorRole, EquilibriumConfig, EquilibriumError,
    EquilibriumState, LmnkPackageOwned, LmnkPackageView, LmnkTimeline, ModelSteeringAdapter,
    PagerStepQuery, RebakeContext, RecallPolicy, RopeGeometry, SourceTextRecord,
    SubtitleActivationEncoder, TimelineResumeOutcome, TurnOrchestrator, TurnPrepareInput,
};
use liminika_core::foundation::compute_crc32_ieee;
use liminika_core::liminal::{Lacuna, evaluate_lacuna_partition_branchless};
use liminika_core::runtime::{AdaptiveThresholdPolicy, CORE_TOPOLOGY, StandardPhaseKvGopCache};
use liminika_core::stigmergy::{
    ATTRACTOR_SLOT_KNOWLEDGE, PhaseAnchorRecord, StandardCognitiveBank,
};
use tokenizers::Tokenizer;

use crate::bake::{BakedModelProfile, argmax, query_nvidia_smi_vram};
use crate::cli::EvalCliConfig;
use crate::eval::{
    SteerDecodeConfig, apply_ego_mode, count_semantic_forbid_hits, generate_steered_reply,
};
use crate::metrics::kl_divergence_and_max_abs_delta;
use crate::model::{
    CausalLmModel, ChatTemplate, KvCacheMode, LogitsFetchMode, ModelConfig, ResonantAnchors,
};

struct DeterministicRebakeEncoder {
    layers: usize,
    rot_len: usize,
}

impl SubtitleActivationEncoder for DeterministicRebakeEncoder {
    fn encode_subtitle_activations(
        &mut self,
        record: &SourceTextRecord,
    ) -> Result<Vec<Vec<f32>>, String> {
        let seed = record.text.bytes().fold(0x1357_9BDF_u64, |acc, b| {
            acc.wrapping_mul(131).wrapping_add(u64::from(b))
        });
        Ok((0..self.layers)
            .map(|l| {
                (0..self.rot_len)
                    .map(|d| (((seed ^ ((l as u64) << 7) ^ (d as u64)) % 100) as f32) * 0.01 - 0.5)
                    .collect()
            })
            .collect())
    }
}

fn make_synthetic_anchor(anchor_id: u32, seed: u32, mass: f32) -> PhaseAnchorRecord {
    let mut rec = PhaseAnchorRecord::new_empty(anchor_id, ATTRACTOR_SLOT_KNOWLEDGE, 0);
    rec.base_energy = mass;
    rec.phase_gravity_mass = mass;
    rec.resultant_r_bar = 0.95;
    rec.von_mises_kappa = 10.0;
    rec.sample_count = 2;
    let mut state = (u64::from(seed) << 32) ^ u64::from(anchor_id).wrapping_mul(0x9E37_79B9);
    for dst in &mut rec.phase_profile {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let val = ((z >> 56) as i16) - 128;
        *dst = val.clamp(-127, 127) as i8;
    }
    rec.stamp_crc32();
    rec
}

struct EmptyKvProbeResult {
    reply: String,
    has_self_ref: bool,
    is_unobserved: bool,
    kv_start: usize,
    init_logits: Vec<f32>,
}

// Single-probe helper passes borrowed evaluation context explicitly to keep call sites flat.
#[allow(clippy::too_many_arguments)]
fn run_single_empty_kv_probe(
    prompt: &str,
    cli: &EvalCliConfig,
    cfg: &ModelConfig,
    chat: &ChatTemplate,
    compiled: &liminika_adapter::CompiledTokenCorpus,
    baked: &BakedModelProfile,
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    adapter: &mut ModelSteeringAdapter,
) -> Result<EmptyKvProbeResult, Box<dyn std::error::Error>> {
    model.clear_cache();
    let kv_start = model.kv_len();
    let q_ids = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("probe encode error: {e}"))?
        .get_ids()
        .to_vec();
    let (_, query_tok_qs, _) = model.forward_tokens(
        &q_ids,
        0,
        None,
        false,
        true,
        LogitsFetchMode::FullLogits,
        KvCacheMode::IsolatedProbe,
    )?;
    let kv_policy = AdaptiveThresholdPolicy::default();
    let mut kv_cache = StandardPhaseKvGopCache::new();
    let plan = TurnOrchestrator::prepare_turn(
        compiled,
        adapter,
        &mut kv_cache,
        &TurnPrepareInput {
            query_token_qs: &query_tok_qs,
            query_seq_len: q_ids.len(),
            turn_index: 1,
            kv_policy: &kv_policy,
            kv_min_resonance_override: cli.kv_min_res,
            persona_bank: Option::<&StandardCognitiveBank>::None,
            layer_steer_mask_override: cli.resolve_layer_mask_override(cfg.num_hidden_layers),
        },
    )?;

    let fact_ctx = plan.self_ref_context();
    let turn_body = if plan.is_unobserved_lacuna() && !plan.is_guard_trap() {
        prompt.to_string()
    } else if !fact_ctx.is_empty() {
        format!("[Context: {fact_ctx}]\n{prompt}")
    } else {
        prompt.to_string()
    };
    let formatted = chat.format_single_turn(&turn_body);

    let eff_mask = plan.effective_layer_steer_mask(adapter);
    let steer_facts = plan.self_ref_anchors_per_layer();
    let use_steer = plan.steering_eligible();
    let mut steer_cfg = SteerDecodeConfig {
        q7s: Some(baked.fact_q7.as_slice()),
        persona_q7s: Some(baked.persona_q7.as_slice()),
        resonant_ctx: if use_steer {
            Some(ResonantAnchors {
                facts: steer_facts,
                personas: plan.persona_anchors_per_layer(),
                guards: plan.guard_anchors_per_layer(),
                negs: baked.repulsive_centroid_per_layer.as_slice(),
                poss: baked.attractive_centroid_per_layer.as_slice(),
            })
        } else {
            None
        },
        rope_rad: if use_steer { -2.00 } else { 0.00 },
        rank1: if use_steer {
            Some((baked.rank1_u.as_slice(), baked.rank1_v.as_slice(), -1.00))
        } else {
            None
        },
        use_steer,
        horizon_mode: cli.horizon_mode,
        escalation_provider: None,
        pressure_sim: cli.pressure_sim_mode,
        enable_pre_bypass: cli.enable_pre_bypass,
        temperature: 0.0,
        rng_seed: cli.rng_seed,
        logits_observe_stride: cli.logits_observe_stride,
        layer_steer_mask: Some(eff_mask),
        probe_frame: cli.probe_frame,
        phase_clamp_override: cli.phase_clamp_rad,
        random_push_seed: None,
        max_tokens: cli.max_tokens.min(48),
        prefill_chunk_size: cli.prefill_chunk_size,
        capture_first_logits: true,
        first_prefill_logits: None,
        exit_entropy_h_tilde: 0.0,
        steer_bypass_hits: 0,
        steer_eligible_layers: 0,
        spine_veto_guard_hits: 0,
        spine_veto_repulse_hits: 0,
        spine_veto_pin_hits: 0,
    };
    model.clear_cache();
    let (reply, _, _, _) =
        generate_steered_reply(model, tokenizer, chat, adapter, &formatted, &mut steer_cfg)?;
    let init_logits = steer_cfg.first_prefill_logits.unwrap_or_default();
    Ok(EmptyKvProbeResult {
        reply,
        has_self_ref: plan.has_self_ref_anchors(),
        is_unobserved: plan.is_unobserved_lacuna(),
        kv_start,
        init_logits,
    })
}

// Benchmark entrypoint passes borrowed runtime handles directly from `main.rs`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_equilibrium_benchmark(
    cli: &EvalCliConfig,
    cfg: &ModelConfig,
    chat: &ChatTemplate,
    active_layer_mask: u64,
    baked: &BakedModelProfile,
    compiled: &liminika_adapter::CompiledTokenCorpus,
    base_entropies: &[f32],
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    work_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let t_all = Instant::now();
    println!(
        "\n=== Cross-Restart Equilibrium Persistence Benchmark (Equilibrium E2E) ===\n  \
         topology_crc32=0x{:08x} corpus_anchors={} work_dir={}",
        CORE_TOPOLOGY.topology_crc32,
        compiled.corpus.entries().len(),
        work_dir.display(),
    );

    let tmp_dir = work_dir.join(format!("equilibrium_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir)?;
    let lmnkc_path = tmp_dir.join("session_equilibrium.lmnkc");
    let lmnkp_path = tmp_dir.join("canonical_source.lmnkp");

    // ------------------------------------------------------------------
    // Step 1: Session 1 live turn + EquilibriumState + .lmnkc persist
    // ------------------------------------------------------------------
    let mut session1 = ModelSteeringAdapter::new(cfg.to_rope_geometry())?;
    session1.set_ego_max_hops(cli.ego_max_hops);
    session1.set_guard_hold_tokens(cli.guard_hold_tokens);
    session1.set_layer_rope_mask(active_layer_mask);
    apply_ego_mode(&mut session1, &cli.ego_mode);
    let pos_slices: Vec<&[f32]> = baked
        .attractive_centroid_per_layer
        .iter()
        .map(Vec::as_slice)
        .collect();
    let neg_slices: Vec<&[f32]> = baked
        .repulsive_centroid_per_layer
        .iter()
        .map(Vec::as_slice)
        .collect();
    let _ = session1.calibrate_manifold_probe(&pos_slices, &neg_slices, base_entropies)?;

    let s1_prompt = "What does GCSO stand for, and does LiminiKa modify base LLM weights?";
    let s1_probe = run_single_empty_kv_probe(
        s1_prompt,
        cli,
        cfg,
        chat,
        compiled,
        baked,
        model,
        tokenizer,
        &mut session1,
    )?;
    let (s1_reply, s1_has_self, s1_unobs) = (
        s1_probe.reply,
        s1_probe.has_self_ref,
        s1_probe.is_unobserved,
    );
    let s1_slow_adj = session1.apply_slow_loop_telemetry_epoch(1);

    let eq_cfg = EquilibriumConfig::default();
    let pager = compiled
        .corpus
        .build_working_set_pager(AnchorPagerConfig::default());
    let mut eq = EquilibriumState::from_pager(pager, eq_cfg);
    // Pin first anchor as Spine-protected floor + insert one isolated Exterior anchor.
    let first_anchor_id = eq
        .pager()
        .backing_entries()
        .first()
        .map_or(1, |e| e.record.anchor_id);
    eq.set_pinned(first_anchor_id, true)?;
    let ext_anchor = make_synthetic_anchor(9001, 0xE1E1, 2.0);
    eq.insert_anchor(ext_anchor, Lacuna::Exterior, false)?;

    let sync_report = session1.sync_equilibrium_state(&mut eq)?;
    let lmnkc_written = eq.save_lmnkc_to_path(&lmnkc_path, 1_774_900_001)?;
    let s1_pin_mask = session1.envelope().baseline_pinned_compute_mask;
    let s1_ego_hash = session1.ego_state().ego_signature_hash;
    let s1_resident_ids: Vec<u32> = eq
        .pager()
        .resident_entries()
        .map(|e| e.record.anchor_id)
        .collect();
    let s1_first_profile = eq
        .pager()
        .get_entry(first_anchor_id)
        .map(|e| e.record.phase_profile)
        .unwrap_or([0_i8; 64]);

    println!(
        "[Session 1    ] self_ref={} unobs={} slow_adj={} | eq_gen={} backing={} resident={} \
         pinned={} ext_iso={} lmnkc={}B spine_pin=0x{:08x} ego_hash=0x{:08x}",
        s1_has_self,
        s1_unobs,
        s1_slow_adj.applied,
        sync_report.generation,
        sync_report.backing_len,
        sync_report.resident_len,
        sync_report.pinned_preserved,
        sync_report.exterior_isolated,
        lmnkc_written,
        s1_pin_mask,
        s1_ego_hash,
    );
    println!("[Session 1 Out] {}", s1_reply.replace('\n', " "));

    // ------------------------------------------------------------------
    // Step 2: 100-Generation Flat Convergence Verification (on clone)
    // ------------------------------------------------------------------
    let mut eq_100 = eq.clone();
    eq_100.insert_anchor(
        make_synthetic_anchor(8801, 8801, 1.0),
        Lacuna::SelfRef,
        false,
    )?;
    eq_100.insert_anchor(
        make_synthetic_anchor(8802, 8802, 1.0),
        Lacuna::SelfRef,
        false,
    )?;
    let mut evap_total = 0_usize;
    let mut max_gen_bytes = 0_usize;
    let core_pulse_rec = eq
        .pager()
        .get_entry(first_anchor_id)
        .map(|e| e.record)
        .unwrap_or_else(|| make_synthetic_anchor(first_anchor_id, 1, 1.5));
    for generation in 1..=100_u64 {
        let mut pulse = EquilibriumState::new(eq_cfg);
        pulse.insert_anchor(core_pulse_rec, Lacuna::SelfRef, false)?;
        let rep = eq_100.join(&pulse)?;
        evap_total += rep.evaporated_anchors;
        let bytes = eq_100.to_lmnkc_bytes(1_774_900_100 + generation)?;
        max_gen_bytes = max_gen_bytes.max(bytes.len());
        eq_100 = EquilibriumState::from_lmnkc_bytes(&bytes, eq_cfg)?;
    }
    let novel_rep =
        eq_100.ingest_crystallized_anchors(&[make_synthetic_anchor(9999, 0xABCD, 1.4)])?;
    let pinned_alive_100 = eq_100.pager().is_resident(first_anchor_id)
        && eq_100
            .pager()
            .get_entry(first_anchor_id)
            .is_some_and(|e| e.pinned);
    let stale_evaporated_100 =
        eq_100.pager().get_entry(8801).is_none() && eq_100.pager().get_entry(8802).is_none();
    let novel_admitted_100 = eq_100.pager().is_resident(9999);
    let ext_still_isolated_100 = eq_100.pager().get_entry(9001).is_none_or(|e| {
        matches!(e.lacuna_state, Lacuna::Exterior) && !eq_100.pager().is_resident(9001)
    });
    println!(
        "[100-Gen Conv ] final_gen={} max_lmnkc={}B evap_total={} pinned_alive={} \
         stale_evap={} novel_admitted={} (ins={}) ext_isolated={}",
        eq_100.generation(),
        max_gen_bytes,
        evap_total,
        pinned_alive_100,
        stale_evaporated_100,
        novel_admitted_100,
        novel_rep.inserted_anchors,
        ext_still_isolated_100,
    );

    // ------------------------------------------------------------------
    // Step 3: Full Restart Simulation (drop adapter + clear KV cache)
    // ------------------------------------------------------------------
    drop(session1);
    drop(eq);
    model.clear_cache();
    let kv_after_restart = model.kv_len();

    // ------------------------------------------------------------------
    // Step 4: Session 2 restore from .lmnkc + empty-KV SelfRef & OOD
    // ------------------------------------------------------------------
    let mut restored_eq = EquilibriumState::load_lmnkc_from_path(&lmnkc_path, eq_cfg)?;
    let mut session2 = ModelSteeringAdapter::new(cfg.to_rope_geometry())?;
    session2.set_ego_max_hops(cli.ego_max_hops);
    session2.set_guard_hold_tokens(cli.guard_hold_tokens);
    session2.set_layer_rope_mask(active_layer_mask);
    let snap_restored = session2.restore_from_equilibrium(&restored_eq)?;
    let _ = session2.calibrate_manifold_probe(&pos_slices, &neg_slices, base_entropies)?;

    let s2_pin_mask = session2.envelope().baseline_pinned_compute_mask;
    let pin_floor_intact = (s2_pin_mask & s1_pin_mask) == s1_pin_mask;
    let ego_restored = session2.ego_state().ego_signature_hash == s1_ego_hash;
    let s2_resident_ids: Vec<u32> = restored_eq
        .pager()
        .resident_entries()
        .map(|e| e.record.anchor_id)
        .collect();
    let anchors_restored = s1_resident_ids
        .iter()
        .all(|id| s2_resident_ids.contains(id));
    let ext_restored_isolated = restored_eq.pager().get_entry(9001).is_some_and(|e| {
        matches!(e.lacuna_state, Lacuna::Exterior) && !restored_eq.pager().is_resident(9001)
    });
    let self_ref_part = evaluate_lacuna_partition_branchless(0xFF, &[96; 8], &[16; 8], 32);
    let step_out = restored_eq.step_working_set(&PagerStepQuery {
        phase_profile: s1_first_profile,
        lacuna_partition: self_ref_part,
        desired_count: 4,
        allow_exterior_staging: false,
    });
    let pager_recall_hit = step_out.resonant_ids.contains(&first_anchor_id);

    // Live empty-KV SelfRef recall query in Session 2.
    let s2_self_probe = run_single_empty_kv_probe(
        s1_prompt,
        cli,
        cfg,
        chat,
        compiled,
        baked,
        model,
        tokenizer,
        &mut session2,
    )?;
    let (s2_self_reply, s2_has_self, s2_self_unobs, s2_self_kv_start) = (
        s2_self_probe.reply,
        s2_self_probe.has_self_ref,
        s2_self_probe.is_unobserved,
        s2_self_probe.kv_start,
    );
    let lower_reply = s2_self_reply.to_ascii_lowercase();
    let self_hits = ["geometric cellular sheaf orchestrator", "sidecar"]
        .iter()
        .filter(|kw| lower_reply.contains(**kw))
        .count();
    let forb_owned = vec!["LoRA".to_string(), "overwrites".to_string()];
    let (_, self_sem_forb) = count_semantic_forbid_hits(&s2_self_reply, &forb_owned);

    // Live empty-KV OOD non-interference query in Session 2 vs unsteered baseline.
    let ood_prompt = "What is the boiling point of water at standard sea-level pressure?";
    let ood_formatted = chat.format_single_turn(ood_prompt);
    let ood_ids = tokenizer
        .encode(ood_formatted.as_str(), false)
        .map_err(|e| format!("ood encode error: {e}"))?
        .get_ids()
        .to_vec();
    model.clear_cache();
    let (ood_base_out, _, _) = model.forward_tokens(
        &ood_ids,
        0,
        None,
        false,
        false,
        LogitsFetchMode::FullLogits,
        KvCacheMode::IsolatedProbe,
    )?;
    let ood_base_logits = ood_base_out.into_full_logits()?;
    let s2_ood_probe = run_single_empty_kv_probe(
        ood_prompt,
        cli,
        cfg,
        chat,
        compiled,
        baked,
        model,
        tokenizer,
        &mut session2,
    )?;
    let (s2_ood_reply, s2_ood_has_self, s2_ood_unobs, s2_ood_kv_start, s2_ood_logits) = (
        s2_ood_probe.reply,
        s2_ood_probe.has_self_ref,
        s2_ood_probe.is_unobserved,
        s2_ood_probe.kv_start,
        s2_ood_probe.init_logits,
    );
    let (ood_kl, ood_max_dlogit) =
        kl_divergence_and_max_abs_delta(&ood_base_logits, &s2_ood_logits);
    let ood_argmax_match = argmax(&ood_base_logits) == argmax(&s2_ood_logits);

    println!(
        "[Session 2    ] kv_restart={} snap_restored={} pin_floor_intact={} ego_restored={} \
         anchors_restored={} ext_isolated={} pager_hit={}",
        kv_after_restart,
        snap_restored,
        pin_floor_intact,
        ego_restored,
        anchors_restored,
        ext_restored_isolated,
        pager_recall_hit,
    );
    println!(
        "[S2 SelfRef   ] kv_start={} has_self={} unobs={} kw_hits={}/2 sem_forb={} | reply={}",
        s2_self_kv_start,
        s2_has_self,
        s2_self_unobs,
        self_hits,
        self_sem_forb,
        s2_self_reply.replace('\n', " "),
    );
    println!(
        "[S2 OOD Check ] kv_start={} has_self={} unobs={} argmax_match={} KL={:.6} max_dlogit={:.5} | reply={}",
        s2_ood_kv_start,
        s2_ood_has_self,
        s2_ood_unobs,
        ood_argmax_match,
        ood_kl,
        ood_max_dlogit,
        s2_ood_reply.replace('\n', " "),
    );

    // ------------------------------------------------------------------
    // Step 5: Foreign Topology Rejection & Immutable .lmnkp Rebake
    // ------------------------------------------------------------------
    let foreign_eq =
        EquilibriumState::with_topology_crc32(eq_cfg, CORE_TOPOLOGY.topology_crc32 ^ 0x55AA_1234);
    let topo_rejected = matches!(
        restored_eq.join(&foreign_eq),
        Err(EquilibriumError::TopologyMismatch { .. })
    );

    let geom_src = RopeGeometry {
        layout: liminika_adapter::RopeLayout::HalfSplit,
        head_dim: 64,
        rotary_dim: 32,
        num_q_heads: 8,
        num_kv_heads: 4,
        num_layers: 4,
        layer_rope_mask: 0x0F,
    };
    let adapter_src = ModelSteeringAdapter::new(geom_src.clone())?;
    let policy = RecallPolicy::default();
    let detector = AdaptiveGopCutDetector::default();
    let mut enc_src = DeterministicRebakeEncoder {
        layers: geom_src.num_layers,
        rot_len: geom_src.q_rot_len(),
    };
    let sub_records = vec![
        SourceTextRecord {
            frame_index: 0,
            timestamp_epoch_sec: 1_774_900_000,
            role: AttractorRole::Fact,
            text: "GCSO stands for Geometric Cellular Sheaf Orchestrator.".to_string(),
            energy: 2.0,
        },
        SourceTextRecord {
            frame_index: 1,
            timestamp_epoch_sec: 1_774_900_010,
            role: AttractorRole::Fact,
            text: "LiminiKa acts as a non-invasive sidecar without modifying base weights."
                .to_string(),
            energy: 1.8,
        },
    ];
    let ctx_src = RebakeContext {
        adapter: &adapter_src,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    let (tl_src, _) =
        LmnkTimeline::bake_from_subtitle_records(&sub_records, &ctx_src, &mut enc_src)?;
    let source_dsl = "persona LiminiKaSource {\n  fact \"GCSO sidecar\"\n}\n";
    let _ = tl_src.save_lmnkp_to_path(&lmnkp_path, Some(source_dsl))?;
    let pkg_before = std::fs::read(&lmnkp_path)?;
    let crc_before = compute_crc32_ieee(&pkg_before);

    let owned_pkg = LmnkPackageOwned::load_from_path(&lmnkp_path)?;
    let imported_tl = LmnkTimeline::import_lmnkp_bytes(&owned_pkg.into_bytes())?;
    let mut geom_dst = geom_src;
    geom_dst.rotary_dim = 64;
    let adapter_dst = ModelSteeringAdapter::new(geom_dst.clone())?;
    let mut enc_dst = DeterministicRebakeEncoder {
        layers: geom_dst.num_layers,
        rot_len: geom_dst.q_rot_len(),
    };
    let ctx_dst = RebakeContext {
        adapter: &adapter_dst,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    let rebaked_frames = match imported_tl.resume_or_rebake(&ctx_dst, &mut enc_dst, 1)? {
        TimelineResumeOutcome::Rebaked { rebaked_frames, .. } => rebaked_frames,
        TimelineResumeOutcome::FastResumed { .. } => 0,
    };

    let pkg_after = std::fs::read(&lmnkp_path)?;
    let crc_after = compute_crc32_ieee(&pkg_after);
    let lmnkp_bytes_identical = pkg_before == pkg_after && crc_before == crc_after;
    let pkg_view = LmnkPackageView::parse(&pkg_after)?;
    let src_intact = pkg_view.source_lmnk() == Some(source_dsl.as_bytes());

    println!(
        "[Immut Rebake ] foreign_topo_rejected={} rebaked_frames={} lmnkp_bytes={} \
         crc32=0x{:08x}->0x{:08x} bit_identical={} source_intact={} | total={:.2?}",
        topo_rejected,
        rebaked_frames,
        pkg_after.len(),
        crc_before,
        crc_after,
        lmnkp_bytes_identical,
        src_intact,
        t_all.elapsed(),
    );
    if let Some(vram) = query_nvidia_smi_vram() {
        println!("[Equil VRAM   ] end=[{vram}]");
    }

    let _ = std::fs::remove_dir_all(&tmp_dir);
    model.clear_cache();
    Ok(())
}
