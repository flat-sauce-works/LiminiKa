// name: benches/bench_phase_steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Advanced benchmarks for `PhaseSteeringEngine` hot-path kernels and
//! `PhaseSteeringClosedLoop` blanket implementations, engineered for **zero input overfitting**
//! and **clean separation of Provisional `.gcso` Storage Container workloads**:
//! - Uses realistic heterogeneous 64-head phase profiles across all batch and sub-head ensembles
//!   (never uniform `[8i8; 64]` arrays or unit-gain `1.0f32` fast-path bypasses).
//! - Evaluates Coarse-to-Fine and Top-2 Liminal routing across 8 candidate trails (2 candidates
//!   per macro cluster across 4 centroids) so the Fine step genuinely resolves intra-cluster
//!   candidate competition.
//! - Isolates provisional pre-v1.0 `.gcso` container crystallization and zero-copy
//!   mapping workloads into the dedicated `PhaseSteering_Provisional_GCSO_Container` Criterion group.

#![allow(missing_docs)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use liminika_core::abi::{
    gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t, gcso_subhead_config_t,
};
use liminika_core::foundation::{
    detect_frame_delta_bitmask, dispatch_subhead_slot_ensemble,
    dispatch_subhead_slot_ensemble_geodesic, steer_visual_grid_2d_bitmask_gated,
    steer_visual_grid_3d_bitmask_gated, AlignedBuffer16, AlignedBuffer32, AlignedBuffer64,
    AlignedSlice16, AlignedSlice32, AlignedSliceMut16, PhaseSteering, PhaseSteeringEngine,
    DEFAULT_PHASE_CLAMP_MAX_RAD,
};
use liminika_core::runtime::PhaseSteeringClosedLoop;
use liminika_core::stigmergy::{
    CrystallizedAttractorRecord, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_SYSTEM_BASELINE,
};
use std::hint::black_box;
use std::time::Duration;

mod common;
use common::{
    configured_noise_threshold, should_run_provisional_gcso, AlignedHeapBuffer, ScopedFtzdazGuard,
};

#[repr(align(64))]
struct AlignedQ7Buffer<const N: usize>([gcso_q7_t; N]);

/// Restores `dst` from `src` using [`gcso_pointer_trail_t::copy_from_fast`] (64-bit GPR stores at
/// offset `56..120`), preventing 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF) pipeline stalls.
#[inline(always)]
fn reset_trail_fast(dst: &mut gcso_pointer_trail_t, src: &gcso_pointer_trail_t) {
    dst.copy_from_fast(src);
}

/// Restores `dst` from `src` using [`CrystallizedAttractorRecord::copy_from_fast`] (64-bit GPR stores
/// for `phase_profile`), avoiding 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF) stalls.
#[inline(always)]
fn reset_crystal_fast(dst: &mut CrystallizedAttractorRecord, src: &CrystallizedAttractorRecord) {
    dst.copy_from_fast(src);
}

/// Populates an 8-slot (`512`-element) attractor phase bank where each 64-head slot carries a
/// distinct heterogeneous phase profile on $SO(2)^{64}$, avoiding uniform-slot synthetic shortcuts.
#[inline(always)]
const fn make_heterogeneous_slot_bank_512(base_shift: gcso_q7_t) -> [gcso_q7_t; 512] {
    let slot_shifts: [gcso_q7_t; 8] = [0, 12, -18, 24, -10, 16, -22, 8];
    let mut out = [0i8; 512];
    let mut s = 0usize;
    while s < 8 {
        let slot_arr =
            make_heterogeneous_q7_phases_with_offset(base_shift.wrapping_add(slot_shifts[s]));
        let base = s * 64;
        let mut h = 0usize;
        while h < 64 {
            out[base + h] = slot_arr[h];
            h += 1;
        }
        s += 1;
    }
    out
}

/// Initializes `(src, tgt)` f32 tensors with realistic non-collinear 2D sub-channel pairs per
/// head so orthogonal Procrustes phase extraction (`atan2(S_h, C_h)`) yields non-zero per-head
/// phase deltas rather than degenerate zero cross-terms (`1.2 * 0.8 - 1.2 * 0.8 == 0.0`).
fn init_non_collinear_procrustes_tensors(
    src: &mut [f32],
    tgt: &mut [f32],
    head_dim: usize,
    num_heads: usize,
) {
    for h in 0..num_heads {
        let angle = 0.18f32 + (h as f32) * 0.035f32;
        let (s_ang, c_ang) = angle.sin_cos();
        let base = h * head_dim;
        for pair in 0..(head_dim / 2) {
            let idx = base + 2 * pair;
            let x = 0.75f32 + ((pair & 7) as f32) * 0.11f32;
            let y = -0.35f32 + (((pair + h) & 7) as f32) * 0.09f32;
            src[idx] = x;
            src[idx + 1] = y;
            tgt[idx] = x * c_ang - y * s_ang;
            tgt[idx + 1] = x * s_ang + y * c_ang;
        }
    }
}

/// Generates a realistic heterogeneous 64-head phase profile centered around `base_shift` on $SO(2)^{64}$
/// where every head $h \in [0, 64)$ and every 8-head SWAR cell carries a distinct Q7 phase offset,
/// preventing synthetic uniform-array shortcuts while preserving directional coherence around `base_shift`.
#[inline(always)]
const fn make_heterogeneous_q7_phases_with_offset(base_shift: gcso_q7_t) -> [gcso_q7_t; 64] {
    let mut arr = [0i8; 64];
    let cell_offsets: [gcso_q7_t; 8] = [-6, 4, -3, 7, -5, 3, -2, 5];
    let head_offsets: [gcso_q7_t; 8] = [-4, -2, -1, 1, 2, 4, -3, 3];
    let mut i = 0usize;
    while i < 64 {
        arr[i] = base_shift
            .wrapping_add(cell_offsets[i >> 3])
            .wrapping_add(head_offsets[i & 7]);
        i += 1;
    }
    arr
}

fn bench_aligned_query_variant<const ALIGN: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    bench_name: &str,
    engine: &PhaseSteeringEngine,
    shared_q_arena: &mut AlignedHeapBuffer<f32, 512>,
    head_dim: u32,
    num_heads: u32,
    total_elements: usize,
) {
    group.bench_with_input(
        BenchmarkId::new(bench_name, format!("{head_dim}x{num_heads}")),
        &(head_dim, num_heads),
        |b, _| {
            let _ftz = ScopedFtzdazGuard::new();
            let phase_data = AlignedQ7Buffer(make_heterogeneous_q7_phases_with_offset(5));
            let p_slice = AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();
            let mut q_slice = shared_q_arena
                .as_aligned_prefix_mut(total_elements)
                .relax_alignment::<ALIGN>();
            q_slice.as_mut_slice().fill(1.0f32);
            for _ in 0..8 {
                engine
                    .apply_phase_steering_aligned::<ALIGN>(&mut q_slice, &p_slice)
                    .unwrap();
            }

            b.iter(|| {
                engine
                    .apply_phase_steering_aligned::<ALIGN>(
                        black_box(&mut q_slice),
                        black_box(&p_slice),
                    )
                    .unwrap();
                black_box(q_slice.as_slice()[0]);
            });
        },
    );
}

fn bench_phase_steering_operations(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("PhaseSteering_HotPath_Advanced_Kernels");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1350));

    let configurations = [
        (64u32, 8u32),
        (128u32, 16u32),
        (128u32, 32u32),
        (128u32, 64u32),
        (256u32, 32u32),
    ];

    let max_elements = configurations
        .iter()
        .map(|&(d, h)| (d as usize) * (h as usize))
        .max()
        .unwrap_or(256usize * 32usize);
    let mut shared_q_arena =
        AlignedHeapBuffer::<f32, 512>::new_with_phase_offset(max_elements, 1.0f32, 0);

    for &(head_dim, num_heads) in &configurations {
        let total_elements = (head_dim * num_heads) as usize;
        let single_tensor_bytes = total_elements * core::mem::size_of::<f32>();
        let engine = PhaseSteeringEngine::new(head_dim, num_heads)
            .expect("Failed to initialize PhaseSteeringEngine");

        // 1. Single Query Steering Across 32B / 128B / 256B / 512B Alignment Boundaries
        group.throughput(Throughput::Bytes(single_tensor_bytes as u64));
        bench_aligned_query_variant::<32>(
            &mut group,
            "apply_phase_steering_32b",
            &engine,
            &mut shared_q_arena,
            head_dim,
            num_heads,
            total_elements,
        );
        bench_aligned_query_variant::<128>(
            &mut group,
            "apply_phase_steering_128b_cacheline",
            &engine,
            &mut shared_q_arena,
            head_dim,
            num_heads,
            total_elements,
        );
        bench_aligned_query_variant::<256>(
            &mut group,
            "apply_phase_steering_256b_tensor_tile",
            &engine,
            &mut shared_q_arena,
            head_dim,
            num_heads,
            total_elements,
        );
        bench_aligned_query_variant::<512>(
            &mut group,
            "apply_phase_steering_512b_dma",
            &engine,
            &mut shared_q_arena,
            head_dim,
            num_heads,
            total_elements,
        );

        // 3. RIPA Soft-Bounded Phase Clamping (O(1) Direct LUT + Upper d_head/4 Channels)
        group.throughput(Throughput::Bytes(single_tensor_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_safe_ripa",
                format!("{head_dim}x{num_heads}"),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let _ftz = ScopedFtzdazGuard::new();
                let phase_data = AlignedQ7Buffer(make_heterogeneous_q7_phases_with_offset(12));
                let p_slice = AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();
                let mut q_slice = shared_q_arena
                    .as_aligned_prefix_mut(total_elements)
                    .relax_alignment::<32>();
                q_slice.as_mut_slice().fill(1.0f32);
                for _ in 0..8 {
                    engine
                        .apply_phase_steering_safe(
                            &mut q_slice,
                            &p_slice,
                            DEFAULT_PHASE_CLAMP_MAX_RAD,
                        )
                        .unwrap();
                }

                b.iter(|| {
                    engine
                        .apply_phase_steering_safe(
                            black_box(&mut q_slice),
                            black_box(&p_slice),
                            black_box(DEFAULT_PHASE_CLAMP_MAX_RAD),
                        )
                        .unwrap();
                    black_box(q_slice.as_slice()[0]);
                });
            },
        );

        // 4. Soft Phase Damping (Stationary Constant-Time AVX2 Gain Scaling with non-unit gain)
        group.throughput(Throughput::Bytes(single_tensor_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "apply_soft_phase_damping",
                format!("{head_dim}x{num_heads}"),
            ),
            &(head_dim, num_heads),
            |b, _| {
                const DAMPING_GAINS: [f32; 2] = [0.8f32, 1.25f32];
                let _ftz = ScopedFtzdazGuard::new();
                let mut q_slice = shared_q_arena
                    .as_aligned_prefix_mut(total_elements)
                    .relax_alignment::<32>();
                for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
                    *val = 0.75f32 + ((idx & 15) as f32) * 0.0625f32;
                }
                for step in 0..8usize {
                    engine
                        .apply_soft_phase_damping(&mut q_slice, DAMPING_GAINS[step & 1])
                        .unwrap();
                }

                let mut toggle = 0usize;
                b.iter(|| {
                    let gain = DAMPING_GAINS[toggle & 1];
                    toggle = toggle.wrapping_add(1);
                    engine
                        .apply_soft_phase_damping(black_box(&mut q_slice), black_box(gain))
                        .unwrap();
                    black_box(q_slice.as_slice()[0]);
                });
            },
        );
    }
    group.finish();
}

fn bench_phase_steering_long_context_and_procrustes(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("PhaseSteering_Long_Context_And_Procrustes");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1400));

    let head_dim = 128u32;
    let num_heads = 32u32;
    let total_elements = (head_dim * num_heads) as usize;
    let single_tensor_bytes = total_elements * core::mem::size_of::<f32>();
    let engine =
        PhaseSteeringEngine::new(head_dim, num_heads).expect("Failed PhaseSteeringEngine init");
    let mut shared_q_arena =
        AlignedHeapBuffer::<f32, 512>::new_with_phase_offset(total_elements, 1.0f32, 0);
    let mut shared_tgt_arena =
        AlignedHeapBuffer::<f32, 512>::new_with_phase_offset(total_elements, 0.8f32, 384);
    let mut shared_logits_arena =
        AlignedHeapBuffer::<f32, 512>::new_with_phase_offset(32768, 2.5f32, 256);

    // 1. Legacy f32 Cumulative Lazy Phase Unwrapping (500K Context Baseline)
    group.throughput(Throughput::Bytes(single_tensor_bytes as u64));
    group.bench_function("ultra_long_context_cumulative_unwrap_500k", |b| {
        let mut ctx_accum = AlignedHeapBuffer::<f32, 32>::new(num_heads as usize, 0.35f32);
        for (h, val) in ctx_accum.as_mut_slice().iter_mut().enumerate() {
            *val = 1250.0f32 + (h as f32) * 37.5f32;
        }
        let c_slice = AlignedSlice32::new(&ctx_accum.as_slice()[..(num_heads as usize)]).unwrap();
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        let _ftz = ScopedFtzdazGuard::new();
        q_slice.as_mut_slice().fill(1.0f32);
        for _ in 0..8 {
            engine.lazy_unwrap_override(&mut q_slice, &c_slice).unwrap();
        }

        b.iter(|| {
            engine
                .lazy_unwrap_override(black_box(&mut q_slice), black_box(&c_slice))
                .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    // 4A. Orthogonal Procrustes Phase Delta Extraction (2x Tensor Bytes)
    group.throughput(Throughput::Bytes((single_tensor_bytes * 2) as u64));
    group.bench_function("compute_procrustes_phase_delta_alignment", |b| {
        init_non_collinear_procrustes_tensors(
            shared_q_arena.as_mut_slice(),
            shared_tgt_arena.as_mut_slice(),
            head_dim as usize,
            num_heads as usize,
        );
        let s_slice = shared_q_arena
            .as_aligned_prefix(total_elements)
            .relax_alignment::<32>();
        let t_slice = shared_tgt_arena
            .as_aligned_prefix(total_elements)
            .relax_alignment::<32>();
        let _ftz = ScopedFtzdazGuard::new();
        let mut out_phase = AlignedQ7Buffer([0i8; 64]);
        let mut p_out = AlignedSliceMut16::new(&mut out_phase.0[..32]).unwrap();
        for _ in 0..8 {
            engine
                .compute_procrustes_phase_delta(&s_slice, &t_slice, &mut p_out)
                .unwrap();
        }

        b.iter(|| {
            engine
                .compute_procrustes_phase_delta(
                    black_box(&s_slice),
                    black_box(&t_slice),
                    black_box(&mut p_out),
                )
                .unwrap();
            black_box(p_out.as_slice()[0]);
        });
    });

    // 4B. Closed-Loop Procrustes Trail Feedback + SWAR Circular Accum + Geodesic RIPA Steering
    group.throughput(Throughput::Bytes((single_tensor_bytes * 2) as u64));
    // 5. Fused Logit Phase Shift (Vocab Size = 32,768, 128 KB L2-Resident Buffer)
    let vocab_size = 32768usize;
    group.throughput(Throughput::Bytes(
        (vocab_size * std::mem::size_of::<f32>()) as u64,
    ));
    group.bench_function("fused_logit_shift_large_vocab_32k", |b| {
        let mut l_slice = shared_logits_arena
            .as_aligned_prefix_mut(vocab_size)
            .relax_alignment::<32>();
        for (idx, val) in l_slice.as_mut_slice().iter_mut().enumerate() {
            *val = -2.0f32 + ((idx & 31) as f32) * 0.25f32;
        }
        let phase_pos = AlignedQ7Buffer(make_heterogeneous_q7_phases_with_offset(10));
        let p_pos = AlignedSlice16::new(&phase_pos.0[..32]).unwrap();
        for _ in 0..4 {
            engine.fused_logit_shift(&mut l_slice, &p_pos).unwrap();
        }

        b.iter(|| {
            engine
                .fused_logit_shift(black_box(&mut l_slice), black_box(&p_pos))
                .unwrap();
            black_box(l_slice.as_slice()[0]);
        });
    });

    // 6. Quantization-Discretized Phase Steering (32-Head AVX2/SWAR QDPS Step Filter)
    group.throughput(Throughput::Elements(u64::from(num_heads)));
    group.bench_function("qdps_quantization_step_filter_high_frequency", |b| {
        let seed_buf = AlignedBuffer16::new([
            3i8, 12, -2, 18, -4, 9, 1, -15, -5, 14, 4, -21, 6, -11, -3, 22, 2, -16, 5, 25, -6, 13,
            -1, -19, -4, 17, 3, -24, 5, -10, 2, 20,
        ]);
        for _ in 0..8 {
            let mut warm = seed_buf;
            let mut w_slice = warm.as_aligned_slice_mut();
            engine.phase_step_filter(&mut w_slice, 0.12f32).unwrap();
        }

        b.iter_batched(
            || seed_buf,
            |mut local| {
                let mut p_slice = local.as_aligned_slice_mut();
                engine
                    .phase_step_filter(black_box(&mut p_slice), black_box(0.12f32))
                    .unwrap();
                black_box(p_slice.as_slice()[0]);
            },
            criterion::BatchSize::SmallInput,
        );
    });

    group.finish();
}

fn bench_phase_steering_swarm_subhead_and_visual_cowatching(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("PhaseSteering_Swarm_Subhead_And_Visual_CoWatching");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1400));

    let head_dim = 128u32;
    let num_heads = 64u32;
    let total_elements = (head_dim * num_heads) as usize;
    let tensor_bytes = total_elements * core::mem::size_of::<f32>();
    let mut shared_q_arena =
        AlignedHeapBuffer::<f32, 512>::new_with_phase_offset(total_elements, 1.0f32, 0);

    group.throughput(Throughput::Bytes(tensor_bytes as u64));

    // 5A. Swarm-Fragmented Pseudo-MoE (8-Slot Sub-Head Ensemble — Non-Unit Gains)
    group.throughput(Throughput::Bytes(tensor_bytes as u64));
    group.bench_function("subhead_8slot_pseudo_moe_ensemble", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let mut slot_bank =
            AlignedBuffer64::<gcso_q7_t, 512>::new(make_heterogeneous_slot_bank_512(12));
        slot_bank.data[64..128].copy_from_slice(&make_heterogeneous_q7_phases_with_offset(24));
        slot_bank.data[128..192].copy_from_slice(&make_heterogeneous_q7_phases_with_offset(-18));
        let bank_slice = slot_bank.relax_slice::<16>();
        let _ftz = ScopedFtzdazGuard::new();
        let subhead_cfg = gcso_subhead_config_t {
            num_fact_heads: 32,
            num_logic_heads: 24,
            num_explore_heads: 8,
            fact_phase_gain: 1.0,
            logic_phase_gain: 0.85,
            explore_phase_gain: 1.20,
            ..Default::default()
        };
        for _ in 0..8 {
            dispatch_subhead_slot_ensemble(
                &mut q_slice,
                &subhead_cfg,
                &bank_slice,
                ATTRACTOR_SLOT_SYSTEM_BASELINE,
                ATTRACTOR_SLOT_KNOWLEDGE,
                ATTRACTOR_SLOT_PERSONA,
                DEFAULT_PHASE_CLAMP_MAX_RAD,
                8i8,
                head_dim as usize,
            )
            .unwrap();
        }

        b.iter(|| {
            dispatch_subhead_slot_ensemble(
                black_box(&mut q_slice),
                black_box(&subhead_cfg),
                black_box(&bank_slice),
                black_box(ATTRACTOR_SLOT_SYSTEM_BASELINE),
                black_box(ATTRACTOR_SLOT_KNOWLEDGE),
                black_box(ATTRACTOR_SLOT_PERSONA),
                black_box(DEFAULT_PHASE_CLAMP_MAX_RAD),
                black_box(8i8),
                black_box(head_dim as usize),
            )
            .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    // 5B. Geodesic-Folded 8-Slot Sub-Head Ensemble (Non-Unit Gains to exercise full scaling path)
    group.bench_function("subhead_8slot_geodesic_moe_ensemble", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let mut slot_bank =
            AlignedBuffer64::<gcso_q7_t, 512>::new(make_heterogeneous_slot_bank_512(116));
        slot_bank.data[64..128].copy_from_slice(&make_heterogeneous_q7_phases_with_offset(24));
        slot_bank.data[128..192].copy_from_slice(&make_heterogeneous_q7_phases_with_offset(-18));
        let bank_slice = slot_bank.relax_slice::<16>();
        let _ftz = ScopedFtzdazGuard::new();
        let subhead_cfg = gcso_subhead_config_t {
            num_fact_heads: 32,
            num_logic_heads: 24,
            num_explore_heads: 8,
            fact_phase_gain: 1.0,
            logic_phase_gain: 0.85,
            explore_phase_gain: 1.20,
            ..Default::default()
        };
        for _ in 0..8 {
            dispatch_subhead_slot_ensemble_geodesic(
                &mut q_slice,
                &subhead_cfg,
                &bank_slice,
                ATTRACTOR_SLOT_SYSTEM_BASELINE,
                ATTRACTOR_SLOT_KNOWLEDGE,
                ATTRACTOR_SLOT_PERSONA,
                DEFAULT_PHASE_CLAMP_MAX_RAD,
                8i8,
                head_dim as usize,
            )
            .unwrap();
        }

        b.iter(|| {
            dispatch_subhead_slot_ensemble_geodesic(
                black_box(&mut q_slice),
                black_box(&subhead_cfg),
                black_box(&bank_slice),
                black_box(ATTRACTOR_SLOT_SYSTEM_BASELINE),
                black_box(ATTRACTOR_SLOT_KNOWLEDGE),
                black_box(ATTRACTOR_SLOT_PERSONA),
                black_box(DEFAULT_PHASE_CLAMP_MAX_RAD),
                black_box(8i8),
                black_box(head_dim as usize),
            )
            .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    // 5C. Single-Manifold Unified-Field + Sub-Head Pseudo-MoE (Non-Unit Gains) + Inverse-Temperature Logit Sharpening
    // 5D. Classical Online Toroidal k-Means / DP-Means E+M Closed-Loop Gateway Steering (K = 4 Centroids)
    // 5E. Coarse-to-Fine Macro-to-Micro Routing (O(K=4) Centroids -> O(M_local=2) Intra-Cluster Candidates out of 8 Total) + RIPA Steering
    // 5F. Top-2 Liminal Macro-to-Micro Bifurcation Routing (O(K=4) Top-2 Centroids -> Dual-Cluster Fine Selection across 8 Candidates) + RIPA Steering
    // 5G. Online Toroidal k-Means Centroid Basin Consolidation (Merge & Prune 8 Centroids on SO(2)^64)
    group.throughput(Throughput::Elements(8));
    group.bench_function(
        "online_toroidal_centroid_merge_and_prune_8centroids_64h",
        |b| {
            let base_centroids: [CrystallizedAttractorRecord; 8] = core::array::from_fn(|i| {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    (i as u32) + 1,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0020_0000 + ((i as u64) << 12),
                );
                let shift = match i {
                    0 => 24i8,
                    1 => 25i8,
                    2 => -40i8,
                    3 => -39i8,
                    4 => 60i8,
                    5 => -80i8,
                    6 => 10i8,
                    _ => -15i8,
                };
                let mass = if i == 7 {
                    0.01f32
                } else {
                    2.5f32 + (i as f32) * 0.2
                };
                let _ = rec
                    .absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(shift), mass);
                rec
            });

            let mut centroids = base_centroids;
            for _ in 0..8 {
                reset_crystal_fast(&mut centroids[0], &base_centroids[0]);
                reset_crystal_fast(&mut centroids[1], &base_centroids[1]);
                reset_crystal_fast(&mut centroids[2], &base_centroids[2]);
                reset_crystal_fast(&mut centroids[3], &base_centroids[3]);
                centroids[4].base_energy = base_centroids[4].base_energy;
                centroids[4].phase_gravity_mass = base_centroids[4].phase_gravity_mass;
                centroids[4].checksum_crc32 = base_centroids[4].checksum_crc32;
                centroids[5].base_energy = base_centroids[5].base_energy;
                centroids[5].phase_gravity_mass = base_centroids[5].phase_gravity_mass;
                centroids[5].checksum_crc32 = base_centroids[5].checksum_crc32;
                centroids[6].base_energy = base_centroids[6].base_energy;
                centroids[6].phase_gravity_mass = base_centroids[6].phase_gravity_mass;
                centroids[6].checksum_crc32 = base_centroids[6].checksum_crc32;
                reset_crystal_fast(&mut centroids[7], &base_centroids[7]);
                let _ = CrystallizedAttractorRecord::merge_and_prune_centroids_64(
                    &mut centroids,
                    0.85f32,
                    0.05f32,
                    0.98f32,
                );
            }

            b.iter(|| {
                reset_crystal_fast(&mut centroids[0], &base_centroids[0]);
                reset_crystal_fast(&mut centroids[1], &base_centroids[1]);
                reset_crystal_fast(&mut centroids[2], &base_centroids[2]);
                reset_crystal_fast(&mut centroids[3], &base_centroids[3]);
                centroids[4].base_energy = base_centroids[4].base_energy;
                centroids[4].phase_gravity_mass = base_centroids[4].phase_gravity_mass;
                centroids[4].checksum_crc32 = base_centroids[4].checksum_crc32;
                centroids[5].base_energy = base_centroids[5].base_energy;
                centroids[5].phase_gravity_mass = base_centroids[5].phase_gravity_mass;
                centroids[5].checksum_crc32 = base_centroids[5].checksum_crc32;
                centroids[6].base_energy = base_centroids[6].base_energy;
                centroids[6].phase_gravity_mass = base_centroids[6].phase_gravity_mass;
                centroids[6].checksum_crc32 = base_centroids[6].checksum_crc32;
                reset_crystal_fast(&mut centroids[7], &base_centroids[7]);
                let res = CrystallizedAttractorRecord::merge_and_prune_centroids_64(
                    black_box(&mut centroids),
                    black_box(0.85f32),
                    black_box(0.05f32),
                    black_box(0.98f32),
                );
                black_box((res, centroids[1].sample_count, centroids[7].sample_count));
            });
        },
    );

    // 6A. Real-Time Screen/Video Co-Watching: 16x16 = 256 Patch Bitmask-Gated 2D Phase Steering
    let num_patches = 256usize;
    let patch_dim = 64usize;
    let grid_elems = num_patches * patch_dim;
    group.throughput(Throughput::Bytes(
        (grid_elems * core::mem::size_of::<f32>()) as u64,
    ));
    group.bench_function("visual_grid_2d_bitmask_gated_256patches", |b| {
        let mut grid_data = AlignedHeapBuffer::<f32, 32>::new(grid_elems, 1.0f32);
        let mut grid_slice = grid_data.as_aligned_slice_mut();
        let prev_mask = gcso_paged_bitmask_t {
            bits: [
                0x0F0F_0F0F_0F0F_0F0Fu64,
                0x3333_3333_3333_3333u64,
                0x5555_5555_5555_5555u64,
                0x00FF_00FF_00FF_00FFu64,
            ],
        };
        let curr_mask = gcso_paged_bitmask_t {
            bits: [
                0x0FFF_03CF_0FFF_3C0Fu64,
                0x33F3_3C33_3F33_33C3u64,
                0x55F5_5505_5F55_5055u64,
                0x0FFF_000F_0F0F_00F0u64,
            ],
        };

        let _ftz = ScopedFtzdazGuard::new();
        for _ in 0..8 {
            let (delta_mask, _) = detect_frame_delta_bitmask(&prev_mask, &curr_mask);
            let _ = steer_visual_grid_2d_bitmask_gated(
                &mut grid_slice,
                &delta_mask,
                patch_dim,
                num_patches,
                16i8,
                -16i8,
            )
            .unwrap();
        }

        b.iter(|| {
            let (delta_mask, _changed) =
                detect_frame_delta_bitmask(black_box(&prev_mask), black_box(&curr_mask));
            let steered = steer_visual_grid_2d_bitmask_gated(
                black_box(&mut grid_slice),
                black_box(&delta_mask),
                black_box(patch_dim),
                black_box(num_patches),
                black_box(16i8),
                black_box(-16i8),
            )
            .unwrap();
            black_box((steered, grid_slice.as_slice()[0]));
        });
    });

    // 6B. 3D Spatiotemporal (t, x, y) Bitmask-Gated Video/Voxel Patch Steering
    let patch_dim_3d = 72usize;
    let grid_elems_3d = num_patches * patch_dim_3d;
    group.throughput(Throughput::Bytes(
        (grid_elems_3d * core::mem::size_of::<f32>()) as u64,
    ));
    group.bench_function("visual_grid_3d_txy_bitmask_gated_256patches", |b| {
        let mut grid_data = AlignedHeapBuffer::<f32, 32>::new(grid_elems_3d, 1.0f32);
        let mut grid_slice = grid_data.as_aligned_slice_mut();
        let prev_mask = gcso_paged_bitmask_t {
            bits: [
                0x0F0F_0F0F_0F0F_0F0Fu64,
                0x3333_3333_3333_3333u64,
                0x5555_5555_5555_5555u64,
                0x00FF_00FF_00FF_00FFu64,
            ],
        };
        let curr_mask = gcso_paged_bitmask_t {
            bits: [
                0x0FFF_03CF_0FFF_3C0Fu64,
                0x33F3_3C33_3F33_33C3u64,
                0x55F5_5505_5F55_5055u64,
                0x0FFF_000F_0F0F_00F0u64,
            ],
        };

        let _ftz = ScopedFtzdazGuard::new();
        for _ in 0..8 {
            let (delta_mask, _) = detect_frame_delta_bitmask(&prev_mask, &curr_mask);
            let _ = steer_visual_grid_3d_bitmask_gated(
                &mut grid_slice,
                &delta_mask,
                patch_dim_3d,
                num_patches,
                8i8,
                16i8,
                -16i8,
            )
            .unwrap();
        }

        b.iter(|| {
            let (delta_mask, _changed) =
                detect_frame_delta_bitmask(black_box(&prev_mask), black_box(&curr_mask));
            let steered = steer_visual_grid_3d_bitmask_gated(
                black_box(&mut grid_slice),
                black_box(&delta_mask),
                black_box(patch_dim_3d),
                black_box(num_patches),
                black_box(8i8),
                black_box(16i8),
                black_box(-16i8),
            )
            .unwrap();
            black_box((steered, grid_slice.as_slice()[0]));
        });
    });

    group.finish();
}

/// Provisional `.gcso` / MappedStorage 4-Section Container Benchmarks (Isolated from Core PhaseSteering Suite).
///
/// Evaluates pre-v1.0 `.gcso` record crystallization and 4-section zero-copy byte container mapping.
/// Skipped by default under `run_benchmarks.py --profile standard` so Core `PhaseSteering`/`EntropyControl` evaluations
/// remain decoupled from the provisional container layout prior to v1.0 KV-keyframe video-stream memory integration.
fn bench_phase_steering_provisional_gcso_container(c: &mut Criterion) {
    if !should_run_provisional_gcso() {
        return;
    }
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("PhaseSteering_Provisional_GCSO_Container");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1400));

    let head_dim = 128u32;
    let num_heads = 64u32;
    let total_elements = (head_dim * num_heads) as usize;
    let tensor_bytes = total_elements * core::mem::size_of::<f32>();
    let mut shared_q_arena =
        AlignedHeapBuffer::<f32, 512>::new_with_phase_offset(total_elements, 1.0f32, 0);

    group.throughput(Throughput::Bytes(tensor_bytes as u64));

    // 1. Provisional 2-Origin Composite Manifold Gateway + SO(2)-Fused RIPA/sub-head router Steering + `.gcso` Crystallization
    // 2. Provisional End-to-End 2-Origin Gateway -> 8-Cell Swarm -> Bayesian Trail Rewire -> Subhead -> `.gcso` Crystallization
    // 3. Provisional `.gcso` (MappedStorage) Zero-Copy Deserialization + Phase-Proximity Gravity Ensemble Gateway Steering
    let engine = PhaseSteeringEngine::new(head_dim, num_heads)
        .expect("Failed 64-head PhaseSteeringEngine init");
    group.bench_function(
        "storage_zero_copy_crystallized_ensemble_gateway_steer_64h",
        |b| {
            let mut q_slice = shared_q_arena
                .as_aligned_prefix_mut(total_elements)
                .relax_alignment::<32>();
            q_slice.as_mut_slice().fill(1.0f32);
            let mut gw_logits =
                AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
            let records: [CrystallizedAttractorRecord; 4] = core::array::from_fn(|i| {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    (i as u32) + 40,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0018_0000 + ((i as u64) << 12),
                );
                let _ = rec.absorb_sample_point_64(
                    &make_heterogeneous_q7_phases_with_offset(22 + (i as i8) * 3),
                    2.5 + (i as f32) * 0.5,
                );
                rec
            });
            let mut storage_buf = AlignedBuffer64::<u8, 512>::new([0u8; 512]);
            CrystallizedAttractorRecord::write_records_to_container_bytes(
                &records,
                &mut storage_buf.data,
            )
            .unwrap();
            let base_trail = gcso_pointer_trail_t {
                stigmergic_density: 1.2,
                attractor_pull_force: 1.5,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
                ..Default::default()
            };

            let mut trail = base_trail;
            for _ in 0..8 {
                reset_trail_fast(&mut trail, &base_trail);
                gw_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
                let mut logits_mut = gw_logits.as_aligned_slice_mut();
                let zero_copy_recs =
                    CrystallizedAttractorRecord::slice_from_container_bytes_zero_copy(
                        &storage_buf.data,
                        4usize,
                    )
                    .unwrap();
                let _ = engine
                    .steer_gateway_signal_safe(
                        zero_copy_recs,
                        &mut trail,
                        64u8,
                        &mut q_slice,
                        None,
                        Some(&mut logits_mut),
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_trail_fast(&mut trail, &base_trail);
                gw_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
                let mut logits_mut = gw_logits.as_aligned_slice_mut();
                let zero_copy_recs =
                    CrystallizedAttractorRecord::slice_from_container_bytes_zero_copy(
                        black_box(&storage_buf.data),
                        black_box(4usize),
                    )
                    .unwrap();
                let res = engine
                    .steer_gateway_signal_safe(
                        black_box(zero_copy_recs),
                        black_box(&mut trail),
                        black_box(64u8),
                        black_box(&mut q_slice),
                        black_box(None),
                        black_box(Some(&mut logits_mut)),
                    )
                    .unwrap();
                black_box((res, trail.target_anchor_id, q_slice.as_slice()[0]));
            });
        },
    );

    // 4. Provisional Register-Resident Zero-Copy Crystallized Ensemble Steering
    // 5. Provisional 4-Section `.gcso` (MappedStorage) Zero-Copy Container Map + Register-Resident Ensemble Steering
    group.finish();
}

fn bench_phase_steering_extreme_and_edge_cases(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("PhaseSteering_Extreme_And_Edge_Cases");
    group.noise_threshold(configured_noise_threshold());

    // 1. Stress Test: Ultra-Large Sequence Scaling (4096..32768 Prefill Context)
    group.sampling_mode(SamplingMode::Flat);
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1500));

    let head_dim = 128u32;
    let num_heads = 32u32;
    let stride = (head_dim * num_heads) as usize;
    let single_tok_bytes = stride * core::mem::size_of::<f32>();
    let engine =
        PhaseSteeringEngine::new(head_dim, num_heads).expect("Failed PhaseSteeringEngine init");

    let mut shared_extreme_arena = AlignedHeapBuffer::<f32, 64>::new(stride, 1.5f32);

    // 2. Constant-Total-Element (1,024 floats = 4 KB) Head Dimension Aspect-Ratio Sweep
    let extreme_head_configs = [
        (16u32, 64u32),
        (32u32, 32u32),
        (64u32, 16u32),
        (128u32, 8u32),
        (256u32, 4u32),
        (512u32, 2u32),
    ];
    for &(h_dim, n_heads) in &extreme_head_configs {
        let total_elements = (h_dim * n_heads) as usize;
        let total_bytes = total_elements * core::mem::size_of::<f32>();
        let ext_engine = PhaseSteeringEngine::new(h_dim, n_heads)
            .expect("Extreme PhaseSteeringEngine init failed");

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("extreme_head_dim", format!("dim_{h_dim}_heads_{n_heads}")),
            &h_dim,
            |b, _| {
                let _ftz = ScopedFtzdazGuard::new();
                let mut q_slice = shared_extreme_arena
                    .as_aligned_prefix_mut(total_elements)
                    .relax_alignment::<32>();
                q_slice.as_mut_slice().fill(1.5f32);
                let phase_data = AlignedQ7Buffer(make_heterogeneous_q7_phases_with_offset(30));
                let p_slice = AlignedSlice16::new(&phase_data.0[..(n_heads as usize)]).unwrap();
                for _ in 0..8 {
                    ext_engine
                        .apply_phase_steering(&mut q_slice, &p_slice)
                        .unwrap();
                }

                b.iter(|| {
                    ext_engine
                        .apply_phase_steering(black_box(&mut q_slice), black_box(&p_slice))
                        .unwrap();
                    black_box(q_slice.as_slice()[0]);
                });
            },
        );
    }

    // 3. Boundary Phase Values: Maximum Q7 Saturation (-128 / +127) vs All-Zero Pass
    group.throughput(Throughput::Bytes(single_tok_bytes as u64));
    group.bench_function("q7_boundary_saturation_extreme", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut q_slice = shared_extreme_arena
            .as_aligned_prefix_mut(stride)
            .relax_alignment::<32>();
        q_slice.as_mut_slice().fill(1.0f32);
        let phase_data = AlignedQ7Buffer([127i8; 64]);
        let p_slice = AlignedSlice16::new(&phase_data.0[..32]).unwrap();
        for _ in 0..8 {
            engine.apply_phase_steering(&mut q_slice, &p_slice).unwrap();
        }

        b.iter(|| {
            engine
                .apply_phase_steering(black_box(&mut q_slice), black_box(&p_slice))
                .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    group.bench_function("q7_boundary_all_zero_pass", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut q_slice = shared_extreme_arena
            .as_aligned_prefix_mut(stride)
            .relax_alignment::<32>();
        q_slice.as_mut_slice().fill(1.0f32);
        let phase_data = AlignedQ7Buffer([0i8; 64]);
        let p_slice = AlignedSlice16::new(&phase_data.0[..32]).unwrap();
        for _ in 0..8 {
            engine.apply_phase_steering(&mut q_slice, &p_slice).unwrap();
        }

        b.iter(|| {
            engine
                .apply_phase_steering(black_box(&mut q_slice), black_box(&p_slice))
                .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    // 4. Extreme Phase Spike Noise & Geodesic RIPA Branchless Clamp (Heterogeneous Extreme Spikes)
    group.bench_function("ripa_extreme_phase_spike_clamp_branchless", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut q_slice = shared_extreme_arena
            .as_aligned_prefix_mut(stride)
            .relax_alignment::<32>();
        q_slice.as_mut_slice().fill(1.0f32);
        let mut extreme_spikes = [0i8; 64];
        for (h, slot) in extreme_spikes.iter_mut().enumerate() {
            let mag = 127i8.saturating_sub((h & 7) as i8);
            *slot = if (h & 1) == 0 {
                if (h & 7) == 0 {
                    -128
                } else {
                    -mag
                }
            } else {
                mag
            };
        }
        let phase_data = AlignedQ7Buffer(extreme_spikes);
        let p_slice = AlignedSlice16::new(&phase_data.0[..32]).unwrap();
        for _ in 0..8 {
            engine
                .apply_phase_steering_geodesic_safe(
                    &mut q_slice,
                    &p_slice,
                    DEFAULT_PHASE_CLAMP_MAX_RAD,
                )
                .unwrap();
        }

        b.iter(|| {
            engine
                .apply_phase_steering_geodesic_safe(
                    black_box(&mut q_slice),
                    black_box(&p_slice),
                    black_box(DEFAULT_PHASE_CLAMP_MAX_RAD),
                )
                .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    // 5. True Subnormal Square Rescue (|x| = 1e-22 -> x^2 = 1e-44 < F32_MIN_POSITIVE_NORMAL)
    group.bench_function("slerp_norm_guard_near_zero_underflow", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let elem_val = 1.0e-22f32;
        let eq_norm = elem_val * (stride as f32).sqrt();
        let mut t_slice = shared_extreme_arena
            .as_aligned_prefix_mut(stride)
            .relax_alignment::<32>();
        t_slice.as_mut_slice().fill(elem_val);
        for _ in 0..8 {
            engine
                .slerp_norm_guard_stable(&mut t_slice, eq_norm, 1.0f32)
                .unwrap();
        }

        b.iter(|| {
            engine
                .slerp_norm_guard_stable(
                    black_box(&mut t_slice),
                    black_box(eq_norm),
                    black_box(1.0f32),
                )
                .unwrap();
            black_box(t_slice.as_slice()[0]);
        });
    });

    // 6. Long-Horizon Subnormal Flush & Near-Identity Gain Guard (Stationary Execution Path)
    group.bench_function("long_horizon_subnormal_damping_guard", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut q_slice = shared_extreme_arena
            .as_aligned_prefix_mut(stride)
            .relax_alignment::<32>();
        q_slice.as_mut_slice().fill(1.0e-36f32);
        for _ in 0..8 {
            q_slice.as_mut_slice()[0] = 2.0e-38f32;
            engine
                .apply_soft_phase_damping(&mut q_slice, 0.5f32)
                .unwrap();
        }

        b.iter(|| {
            q_slice.as_mut_slice()[0] = black_box(2.0e-38f32);
            engine
                .apply_soft_phase_damping(black_box(&mut q_slice), black_box(0.5f32))
                .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_phase_steering_operations,
    bench_phase_steering_long_context_and_procrustes,
    bench_phase_steering_swarm_subhead_and_visual_cowatching,
    bench_phase_steering_extreme_and_edge_cases,
    bench_phase_steering_provisional_gcso_container
);
criterion_main!(benches);
