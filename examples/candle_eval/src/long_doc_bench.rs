// name: examples/candle_eval/src/long_doc_bench.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Long-document 1-pass streaming distillation, `AnchorWorkingSetPager` paging, and multi-chunk
//! prefill verification harness (`--long-doc-bench=<path>`).

use std::path::Path;
use std::time::Instant;

use liminika_adapter::{
    AnchorPagerConfig, AttractorRole, ModelSteeringAdapter, PagerStepQuery,
    StreamingCorpusDistiller, StreamingDistillConfig, split_raw_document_chunks,
};
use liminika_core::liminal::evaluate_lacuna_partition_branchless;
use tokenizers::Tokenizer;

use crate::bake::{LmTokenEncoder, argmax, query_nvidia_smi_vram};
use crate::metrics::kl_divergence_and_max_abs_delta;
use crate::model::{CausalLmModel, ChatTemplate, KvCacheMode, LogitsFetchMode};

#[cfg(target_os = "windows")]
fn current_process_rss_mib() -> Option<f64> {
    let pid = std::process::id();
    let out = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!("(Get-Process -Id {pid}).WorkingSet64"),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let bytes: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(bytes as f64 / (1024.0 * 1024.0))
}

#[cfg(not(target_os = "windows"))]
fn current_process_rss_mib() -> Option<f64> {
    None
}

pub(crate) fn run_long_doc_benchmark(
    doc_path: &Path,
    chunk_size: usize,
    layer_mask: u64,
    model: &mut CausalLmModel,
    tokenizer: &Tokenizer,
    chat: &ChatTemplate,
    adapter: &mut ModelSteeringAdapter,
) -> Result<(), Box<dyn std::error::Error>> {
    let raw_doc = std::fs::read_to_string(doc_path)?;
    let raw_bytes = raw_doc.len();
    let chunks = split_raw_document_chunks(&raw_doc, 720, 48);
    let rss_before = current_process_rss_mib();
    let vram_before = query_nvidia_smi_vram();

    println!(
        "\n=== Long-Document 1-Pass Streaming Distillation & Chunked Prefill Benchmark ===\n  \
         doc={} ({} bytes) | raw_chunks={} | prefill_chunk_size={} | layer_mask=0x{:08X}",
        doc_path.display(),
        raw_bytes,
        chunks.len(),
        chunk_size,
        layer_mask,
    );

    adapter.set_layer_rope_mask(layer_mask);
    let t_distill = Instant::now();
    let mut distiller = StreamingCorpusDistiller::new(adapter, StreamingDistillConfig::default());
    {
        let mut encoder = LmTokenEncoder::new(model, tokenizer, chat);
        for chunk in &chunks {
            let _ = distiller.push_text(&mut encoder, adapter, AttractorRole::Fact, chunk, 1.5)?;
        }
    }
    let encode_elapsed = t_distill.elapsed();
    let t_fin = Instant::now();
    let (compiled, report) = distiller.finalize(adapter, None)?;
    let fin_elapsed = t_fin.elapsed();
    let total_distill_elapsed = t_distill.elapsed();
    let rss_after_distill = current_process_rss_mib();
    let vram_after_distill = query_nvidia_smi_vram();

    let raw_discarded_mib = report.raw_activation_bytes_discarded as f64 / (1024.0 * 1024.0);
    let rss_delta_note = match (rss_before, rss_after_distill) {
        (Some(b), Some(a)) => format!("{b:.1}MiB -> {a:.1}MiB (delta={:+.1}MiB)", a - b),
        _ => "n/a".to_string(),
    };
    println!(
        "[StreamDistill] chunks={} tokens={} anchors={} retained_salient={} discarded_raw={:.1}MiB | \
         encode={:.2?} finalize={:.2?} total={:.2?} | host_rss={rss_delta_note}",
        report.chunks_ingested,
        report.total_tokens_processed,
        report.anchors_generated,
        report.retained_salient_tokens,
        raw_discarded_mib,
        encode_elapsed,
        fin_elapsed,
        total_distill_elapsed,
    );
    if let Some(vram) = vram_after_distill.as_deref() {
        println!("[StreamDistill VRAM] after_distill=[{vram}]");
    }

    // Exercise AnchorWorkingSetPager (>64 anchors) + .lmnkc container roundtrip.
    let pager = compiled
        .corpus
        .build_working_set_pager(AnchorPagerConfig::default());
    let lmnkc_bytes = pager.to_container_bytes(1_700_000_000)?;
    let mut restored_pager = liminika_adapter::AnchorWorkingSetPager::from_container_bytes(
        &lmnkc_bytes,
        AnchorPagerConfig::default(),
    )?;
    // Pin first 4 anchors as Spine-protected floor.
    for pin_id in 1..=4_u32 {
        if restored_pager.get_entry(pin_id).is_some() {
            restored_pager.set_pinned(pin_id, true)?;
        }
    }
    let self_ref_part = evaluate_lacuna_partition_branchless(0xFF, &[96; 8], &[16; 8], 32);
    let unobs_part = evaluate_lacuna_partition_branchless(0x00, &[0; 8], &[16; 8], 32);

    let total_anchors = restored_pager.backing_len();
    for step in 0..48_usize {
        let target_id = ((step * 7 + 65) % total_anchors.max(1) + 1) as u32;
        if let Some(entry) = restored_pager.get_entry(target_id) {
            let _ = restored_pager.step_working_set(&PagerStepQuery {
                phase_profile: entry.record.phase_profile,
                lacuna_partition: self_ref_part,
                desired_count: 4,
                allow_exterior_staging: false,
            });
        }
    }
    // Also exercise Unobserved zero-churn skip.
    for _ in 0..8 {
        let _ = restored_pager.step_working_set(&PagerStepQuery {
            phase_profile: [0_i8; 64],
            lacuna_partition: unobs_part,
            desired_count: 4,
            allow_exterior_staging: false,
        });
    }
    let pstats = restored_pager.stats();
    let pinned_intact = (1..=4_u32).all(|id| restored_pager.is_resident(id));
    println!(
        "[AnchorPager  ] backing={} resident={}/{} lmnkc_bytes={} | page_ins={} page_outs={} \
         pinned_prot={} unobs_skips={} pinned_intact={}",
        restored_pager.backing_len(),
        restored_pager.resident_len(),
        restored_pager.config().working_set_capacity,
        lmnkc_bytes.len(),
        pstats.page_ins,
        pstats.page_outs,
        pstats.pinned_protections,
        pstats.unobserved_skips,
        pinned_intact,
    );

    // Multi-chunk prefill (> 1,024 tokens) parity & speed vs monolithic prefill + dynamic RoPE.
    let full_ids = tokenizer
        .encode(raw_doc.as_str(), false)
        .map_err(|e| format!("long doc encode error: {e}"))?
        .get_ids()
        .to_vec();
    let parity_len = full_ids.len().min(1280);
    let parity_ids = &full_ids[..parity_len];

    model.clear_cache();
    let t_mono = Instant::now();
    let (mono_out, _, _) = model.forward_tokens(
        parity_ids,
        0,
        None,
        false,
        false,
        LogitsFetchMode::FullLogits,
        KvCacheMode::Append,
    )?;
    let mono_ms = t_mono.elapsed().as_secs_f64() * 1000.0;
    let mono_logits = mono_out.into_full_logits()?;
    let mono_vram = query_nvidia_smi_vram();

    model.clear_cache();
    let t_chunk = Instant::now();
    let (chunk_out, pf_report) = model.prefill_tokens_chunked(
        parity_ids,
        0,
        chunk_size,
        None,
        None,
        false,
        LogitsFetchMode::FullLogits,
    )?;
    let chunk_ms = t_chunk.elapsed().as_secs_f64() * 1000.0;
    let chunk_logits = chunk_out.into_full_logits()?;
    let chunk_vram = query_nvidia_smi_vram();
    let (kl, max_abs_diff) = kl_divergence_and_max_abs_delta(&mono_logits, &chunk_logits);
    let argmax_match = argmax(&mono_logits) == argmax(&chunk_logits);
    let chunk_tps = parity_len as f64 / (chunk_ms / 1000.0).max(1e-6);

    println!(
        "[ChunkPrefill ] parity_tokens={} chunks={} chunk_size={} | mono={:.1}ms chunked={:.1}ms ({:.1} tok/s) | \
         argmax_match={} KL={:.6} max_abs_dlogit={:.5} kv_len={}",
        parity_len,
        pf_report.chunks_executed,
        pf_report.chunk_size,
        mono_ms,
        chunk_ms,
        chunk_tps,
        argmax_match,
        kl,
        max_abs_diff,
        model.kv_len(),
    );

    // Exercise dynamic RotaryCache growth beyond the initial 2048 window (e.g. 2560 tokens).
    let ext_len = full_ids.len().min(2560);
    if ext_len > 2048 {
        let rope_before = model.rope_cached_len();
        model.clear_cache();
        let t_ext = Instant::now();
        let (_, ext_report) = model.prefill_tokens_chunked(
            &full_ids[..ext_len],
            0,
            chunk_size,
            None,
            None,
            false,
            LogitsFetchMode::DeviceArgmaxOnly,
        )?;
        let ext_ms = t_ext.elapsed().as_secs_f64() * 1000.0;
        let ext_tps = ext_len as f64 / (ext_ms / 1000.0).max(1e-6);
        println!(
            "[DynamicRoPE  ] tokens={} chunks={} rope_cache={}->{} elapsed={:.1}ms ({:.1} tok/s) kv_len={}",
            ext_len,
            ext_report.chunks_executed,
            rope_before,
            model.rope_cached_len(),
            ext_ms,
            ext_tps,
            model.kv_len(),
        );
    }

    if let (Some(b), Some(m), Some(c)) = (vram_before, mono_vram, chunk_vram) {
        println!("[LongDoc VRAM ] before=[{b}] mono_prefill=[{m}] chunked_prefill=[{c}]");
    }
    model.clear_cache();
    Ok(())
}
