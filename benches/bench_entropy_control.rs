// name: benches/bench_entropy_control.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Benchmarks for `EntropyController` and cohomology/QDPT kernels, engineered for **zero
//! input overfitting**:
//! - Covers stateful entropy mode transitions, sparse surrogate Taylor-Horner paths,
//!   dyadic vector evaluation, continuous and long-dialogue stream simulations,
//!   contrastive Bloom filter self-normalization, Chebyshev spectral filtering,
//!   Hodge obstruction pairing, and non-finite / odd-remainder boundary cases.

use criterion::{
    criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use liminika_core::abi::{
    lmnk_entropy_state_t, lmnk_paged_bitmask_t, lmnk_q7_t, lmnk_tagged_ptr_t,
};
use liminika_core::drivers::spine::entropy::{
    compute_moving_z_score_entropy, compute_sparse_surrogate_entropy, eval_barrier_pitchfork,
    EntropyController,
};
use liminika_core::foundation::{
    compute_tikhonov_chebyshev_coeffs_deg3, eval_chebyshev_spectral_filter_deg3,
    evaluate_hodge_obstruction_and_dual_pairing, AlignedBuffer32, AlignedSlice32,
    ACTION_PRESET_EXPRESS_PHASE,
};
use liminika_core::liminal::{DEFAULT_ENTROPY_EPS, MAX_ATTENTION_HEADS};
use liminika_core::stigmergy::{
    eval_stigmergic_block_degrees, resolve_dyadic_qdpt_and_cascade_ptr,
    ContrastivePhaseBloomFilter, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
};
use std::hint::black_box;
use std::time::Duration;

mod common;
use common::{configured_noise_threshold, AlignedHeapBuffer, ScopedFtzdazGuard};

/// Generates a deterministic 64-supernode f32 field with non-zero spatial coboundary gradients
/// across adjacent supernodes so Sheaf Cohomology and Hodge Obstruction kernels evaluate genuine
/// non-zero coboundary residuals rather than flat constant vectors.
#[inline(always)]
const fn make_heterogeneous_supernode_profile_64(base_val: f32, spread: f32) -> [f32; 64] {
    let offsets: [f32; 8] = [-0.45, 0.30, -0.15, 0.50, -0.35, 0.25, -0.20, 0.40];
    let mut out = [0.0f32; 64];
    let mut i = 0usize;
    while i < 64 {
        let cell_bias = ((i >> 3) as f32 - 3.5f32) * 0.06f32;
        out[i] = base_val + spread * (offsets[i & 7] + cell_bias);
        i += 1;
    }
    out
}

/// Restores `dst` from `src` using [`EntropyController::restore_from`] (width-matched scalar stores),
/// preventing the 256-bit `vmovaps` YMM store-to-scalar-load STLF pipeline stall inside `b.iter`.
#[inline(always)]
fn reset_engine_fast(dst: &mut EntropyController, src: &EntropyController) {
    dst.restore_from(src);
}

fn bench_entropy_control_stateful_eval(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EntropyControl_Stateful_Entropy_Evaluation");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1350));
    group.throughput(Throughput::Elements(1));

    // 1. Single Token Stateful Evaluation Benchmarks (Pre-warmed Stationary Regimes)
    group.bench_function("eval_stateful_normal_mode", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut engine = EntropyController::default();
        let mut state_out = lmnk_entropy_state_t::default();
        for _ in 0..8 {
            engine.eval_stateful(0.42f32, &mut state_out).unwrap();
        }

        b.iter(|| {
            engine
                .eval_stateful(black_box(0.42f32), black_box(&mut state_out))
                .unwrap();
            black_box(state_out.active_branch_mode);
        });
    });

    group.bench_function("eval_stateful_bifurcation_mode_transition", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut base_engine = EntropyController::default();
        base_engine.set_bifurcation_threshold(1.0);
        let mut state_out = lmnk_entropy_state_t::default();
        for _ in 0..8 {
            base_engine.eval_stateful(0.95f32, &mut state_out).unwrap();
        }

        let mut engine = base_engine;

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            engine
                .eval_stateful(black_box(2.40f32), black_box(&mut state_out))
                .unwrap();
            black_box(state_out.active_branch_mode);
        });
    });

    group.bench_function("eval_stateful_repulsion_mode_transition", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut base_engine = EntropyController::default();
        base_engine.set_bifurcation_threshold(1.0);
        base_engine.set_tau_flux(0.2);
        let mut state_out = lmnk_entropy_state_t::default();
        for _ in 0..8 {
            base_engine.eval_stateful(1.30f32, &mut state_out).unwrap();
        }

        let mut engine = base_engine;

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            engine
                .eval_stateful(black_box(6.20f32), black_box(&mut state_out))
                .unwrap();
            black_box(state_out.active_branch_mode);
        });
    });

    // 2A. Direct Sparse Surrogate Entropy (0-ln Taylor-Horner Path) & Moving Z-Score Evaluation
    group.bench_function("compute_sparse_surrogate_and_moving_z_entropy", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        for _ in 0..8 {
            let h_s = compute_sparse_surrogate_entropy(0.72f32, 0.21f32, DEFAULT_ENTROPY_EPS);
            black_box(compute_moving_z_score_entropy(
                h_s, 0.35f32, 0.12f32, 0.08f32, 1e-6f32, 1.0f32, 0.5f32,
            ));
        }

        b.iter(|| {
            let h_surr = compute_sparse_surrogate_entropy(
                black_box(0.72f32),
                black_box(0.21f32),
                black_box(DEFAULT_ENTROPY_EPS),
            );
            compute_moving_z_score_entropy(
                black_box(h_surr),
                black_box(0.35f32),
                black_box(0.12f32),
                black_box(0.08f32),
                black_box(1e-6f32),
                black_box(1.0f32),
                black_box(0.5f32),
            )
        });
    });

    // 2B. Standalone 0-ln 9-Term Even Taylor-Horner Polynomial Surrogate Entropy (Non-Unit Top-2 Sum = 0.87)
    group.bench_function("compute_sparse_surrogate_taylor_horner_0ln", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        for _ in 0..8 {
            black_box(compute_sparse_surrogate_entropy(
                0.68f32,
                0.19f32,
                DEFAULT_ENTROPY_EPS,
            ));
        }

        b.iter(|| {
            compute_sparse_surrogate_entropy(
                black_box(0.68f32),
                black_box(0.19f32),
                black_box(DEFAULT_ENTROPY_EPS),
            )
        });
    });

    // 2C. Zero-Division Deterministic / Symmetric Fast-Path Surrogate Entropy
    group.bench_function("compute_sparse_surrogate_deterministic_fastpath", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        for _ in 0..8 {
            let hg = compute_sparse_surrogate_entropy(0.99f32, 0.0f32, DEFAULT_ENTROPY_EPS);
            let ht = compute_sparse_surrogate_entropy(0.50f32, 0.50f32, DEFAULT_ENTROPY_EPS);
            black_box(hg + ht);
        }

        b.iter(|| {
            let h_greedy = compute_sparse_surrogate_entropy(
                black_box(0.99f32),
                black_box(0.0f32),
                black_box(DEFAULT_ENTROPY_EPS),
            );
            let h_tie = compute_sparse_surrogate_entropy(
                black_box(0.50f32),
                black_box(0.50f32),
                black_box(DEFAULT_ENTROPY_EPS),
            );
            black_box(h_greedy + h_tie)
        });
    });

    // 3. Fused Top-2 Surrogate + Moving Z-Score + Stateful EntropyControl Step
    group.bench_function("eval_surrogate_and_step_fused", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut engine = EntropyController::default();
        let mut state_out = lmnk_entropy_state_t::default();
        for _ in 0..8 {
            engine
                .eval_surrogate_and_step(0.68, 0.24, 0.35, 0.15, 0.09, 1.0, 0.5, &mut state_out)
                .unwrap();
        }

        b.iter(|| {
            let (p_max, p_2nd, cos_d) = black_box((0.68f32, 0.24f32, 0.35f32));
            let (alpha, beta, d0, d_tau) = black_box((0.15f32, 0.09f32, 1.0f32, 0.5f32));
            let h = engine
                .eval_surrogate_and_step(
                    p_max,
                    p_2nd,
                    cos_d,
                    alpha,
                    beta,
                    d0,
                    d_tau,
                    black_box(&mut state_out),
                )
                .unwrap();
            black_box((h, state_out.active_branch_mode));
        });
    });

    // 4. O(1) Branchless 2nd-Order Entropy Flux Acceleration & Adaptive KV-keyframe GOP Budget
    group.bench_function("evaluate_adaptive_compute_and_gop_budget", |b| {
        let mut engine = EntropyController::default();
        let mut state_out = lmnk_entropy_state_t::default();
        for z in [0.4f32, 0.7, 1.1, 1.8] {
            engine.eval_stateful(z, &mut state_out).unwrap();
        }

        b.iter(|| {
            let (flux, accel) = engine.eval_entropy_flux_acceleration();
            let budget = engine.evaluate_adaptive_compute_and_gop_budget(
                black_box(8u32),
                black_box(4u32),
                black_box(32u32),
                black_box(0.25f32),
            );
            black_box((flux, accel, budget));
        });
    });

    group.finish();
}

fn bench_entropy_control_lacuna_dyadic(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EntropyControl_Lacuna_Dyadic_Vector_Alignment");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1350));

    let vector_dims = [128usize, 512, 2048, 8192, 32768, 65536, 131072, 262144];
    let shared_lacuna_arena = AlignedHeapBuffer::<f32, 64>::new_heterogeneous_f32(262144, 0.65f32);

    for dim in vector_dims {
        let total_bytes = dim * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("eval_lacuna_energy_dyadic128", format!("dim_{dim}")),
            &dim,
            |b, &d| {
                let _ftz = ScopedFtzdazGuard::new();
                let slice_32 = AlignedSlice32::new(shared_lacuna_arena.as_prefix(d)).unwrap();
                let engine = EntropyController::default();
                let mut lacuna_score = 0.0f32;
                for _ in 0..8 {
                    engine
                        .eval_lacuna_energy_dyadic128(&slice_32, &mut lacuna_score)
                        .unwrap();
                }

                b.iter(|| {
                    engine
                        .eval_lacuna_energy_dyadic128(
                            black_box(&slice_32),
                            black_box(&mut lacuna_score),
                        )
                        .unwrap();
                    black_box(lacuna_score);
                });
            },
        );
    }

    group.finish();
}

fn bench_entropy_control_stream_simulation(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EntropyControl_Continuous_Stream_Simulation");
    group.noise_threshold(configured_noise_threshold());
    group.warm_up_time(Duration::from_millis(500));

    let sequence_lengths = [100usize, 1000, 10000, 100000];

    for seq_len in sequence_lengths {
        if seq_len >= 100_000 {
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(45);
            group.warm_up_time(Duration::from_millis(500));
            group.measurement_time(Duration::from_millis(1400));
        } else if seq_len >= 10_000 {
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(60);
            group.warm_up_time(Duration::from_millis(400));
            group.measurement_time(Duration::from_millis(1100));
        } else {
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(100);
            group.warm_up_time(Duration::from_millis(500));
            group.measurement_time(Duration::from_millis(1400));
        }

        group.throughput(Throughput::Elements(seq_len as u64));

        group.bench_with_input(
            BenchmarkId::new("simulate_token_stream", format!("seq_{seq_len}")),
            &seq_len,
            |b, &len| {
                let mut z_buf = AlignedHeapBuffer::<f32, 64>::new_heterogeneous_f32(len, 0.5f32);
                for (i, slot) in z_buf.as_mut_slice().iter_mut().enumerate() {
                    let base = 0.5 + 1.5 * ((i as f32 * 0.017).sin().abs());
                    *slot = if i % 150 == 0 { base + 4.5 } else { base };
                }
                let z_scores = z_buf.as_prefix(len);

                let mut engine = EntropyController::default();
                let mut state_out = lmnk_entropy_state_t::default();
                for &z in z_scores {
                    engine.eval_stateful(z, &mut state_out).unwrap();
                }

                b.iter(|| {
                    engine.reset();
                    for &z in z_scores {
                        engine.eval_stateful(black_box(z), &mut state_out).unwrap();
                    }
                    black_box((state_out.moving_z_entropy, state_out.active_branch_mode));
                });
            },
        );
    }

    group.finish();
}

fn bench_entropy_control_long_dialogue_reliability(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EntropyControl_Long_Dialogue_Reliability");
    group.noise_threshold(configured_noise_threshold());

    // 1. Very Long Token Stream Simulation (500,000 Tokens)
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(350));
    group.measurement_time(Duration::from_millis(1100));

    let long_stream_len = 500000usize;
    let mut shared_500k_buf = AlignedHeapBuffer::<f32, 64>::new(long_stream_len, 0.4f32);
    for (i, slot) in shared_500k_buf.as_mut_slice().iter_mut().enumerate() {
        *slot = 0.4 + 1.8 * ((i as f32 * 0.003).sin().abs());
    }
    group.throughput(Throughput::Elements(long_stream_len as u64));
    group.bench_function("simulate_ultra_long_stream_500k", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let z_scores = shared_500k_buf.as_slice();

        let mut engine = EntropyController::default();
        let mut state_out = lmnk_entropy_state_t::default();
        for _ in 0..2 {
            engine.reset();
            for &z in z_scores {
                engine.eval_stateful(z, &mut state_out).unwrap();
            }
        }

        b.iter(|| {
            engine.reset();
            for &z in z_scores {
                engine.eval_stateful(black_box(z), &mut state_out).unwrap();
            }
            black_box((state_out.moving_z_entropy, state_out.active_branch_mode));
        });
    });

    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(60);
    group.warm_up_time(Duration::from_millis(400));
    group.measurement_time(Duration::from_millis(1200));

    // 2. High-Frequency Chattering Oscillation Baseline (50,000 Rapid Transitions, Dwell = 0)
    let chattering_len = 50000usize;
    let tau = 1.5f32;
    let delta = 0.15f32;
    let mut shared_osc_50k = AlignedHeapBuffer::<f32, 64>::new(chattering_len, tau);
    for (i, slot) in shared_osc_50k.as_mut_slice().iter_mut().enumerate() {
        *slot = if (i & 1) == 0 {
            tau + delta + 0.08
        } else {
            tau - delta - 0.08
        };
    }
    group.throughput(Throughput::Elements(chattering_len as u64));
    group.bench_function("chattering_rapid_oscillation_50k_tokens", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let oscillating_z = shared_osc_50k.as_slice();

        let mut engine = EntropyController::default();
        engine.set_bifurcation_threshold(tau);
        engine.set_hysteresis_band(delta);
        let mut state_out = lmnk_entropy_state_t::default();
        for _ in 0..2 {
            engine.reset();
            for &z in oscillating_z {
                engine.eval_stateful(z, &mut state_out).unwrap();
            }
        }

        b.iter(|| {
            engine.reset();
            for &z in oscillating_z {
                engine.eval_stateful(black_box(z), &mut state_out).unwrap();
            }
            black_box((state_out.moving_z_entropy, state_out.active_branch_mode));
        });
    });

    // 3. Dwell-Step Inertia Anti-Chattering Stabilization (50,000 Rapid Transitions, Dwell = 4)
    group.throughput(Throughput::Elements(chattering_len as u64));
    group.bench_function("chattering_dwell_inertia_stabilized_50k_tokens", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let oscillating_z = shared_osc_50k.as_slice();

        let mut engine = EntropyController::default().with_min_mode_dwell_steps(4);
        engine.set_bifurcation_threshold(tau);
        engine.set_hysteresis_band(delta);
        let mut state_out = lmnk_entropy_state_t::default();
        for _ in 0..2 {
            engine.reset();
            for &z in oscillating_z {
                engine.eval_stateful(z, &mut state_out).unwrap();
            }
        }

        b.iter(|| {
            engine.reset();
            for &z in oscillating_z {
                engine.eval_stateful(black_box(z), &mut state_out).unwrap();
            }
            black_box((state_out.moving_z_entropy, state_out.active_branch_mode));
        });
    });

    // 4. Long-Horizon 100K Repulsive Bloom Filter Self-Normalization Stress Test
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(600));
    group.measurement_time(Duration::from_millis(2200));

    let bloom_ops_len = 100000usize;
    group.throughput(Throughput::Elements(bloom_ops_len as u64));
    group.bench_function("long_horizon_bloom_self_normalization_100k", |b| {
        let patterns: [[lmnk_q7_t; MAX_ATTENTION_HEADS]; 8] = core::array::from_fn(|k| {
            let mut arr = [0i8; MAX_ATTENTION_HEADS];
            let mut h = 0usize;
            while h < MAX_ATTENTION_HEADS {
                arr[h] = (((h + k * 13) * 19) & 0x7F) as i8 - 64;
                h += 1;
            }
            arr
        });

        b.iter(|| {
            let mut bloom = ContrastivePhaseBloomFilter::new();
            let mut repulsed = 0u32;
            for i in 0..bloom_ops_len {
                let pat = &patterns[i & 7];
                bloom.insert_repulsive_pattern(black_box(pat));
                if (i & 15) == 0 && bloom.query_repulsive_density(black_box(pat)) >= 4 {
                    repulsed = repulsed.wrapping_add(1);
                }
            }
            black_box((bloom.total_mass(), repulsed));
        });
    });

    group.finish();
}

fn bench_entropy_control_pointer_tunneling_and_cohomology(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EntropyControl_Pointer_Tunneling_And_Cohomology");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1350));

    // 1. Pointer-Coupled EntropyControl Step + Eyring-Kramers Temperature + Stigmergic Mode Evolution
    group.throughput(Throughput::Elements(1));
    group.bench_function("eval_stateful_with_lacuna_and_pointer_evolution", |b| {
        let mut base_engine = EntropyController::default().with_min_mode_dwell_steps(3);
        let mut st_warm = lmnk_entropy_state_t::default();
        for _ in 0..8 {
            base_engine.eval_stateful(1.1f32, &mut st_warm).unwrap();
        }
        let ptr = lmnk_tagged_ptr_t::pack_action(
            0x4000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        );

        let mut engine = base_engine;

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            let res = engine
                .eval_stateful_with_lacuna_and_pointer(
                    black_box(2.2f32),
                    black_box(0.35f32),
                    black_box(1.2f32),
                    black_box(0.8f32),
                    black_box(0.5f32),
                    black_box(ptr),
                    black_box(1.1f32),
                    black_box(ATTRACTOR_SLOT_KNOWLEDGE),
                    black_box(1.5f32),
                )
                .unwrap();
            black_box(res);
        });
    });

    // 2. Matrix-Free Stigmergic Block Degrees & 3rd-Order Chebyshev Filter
    group.throughput(Throughput::Elements(16));
    group.bench_function(
        "matrix_free_stigmergic_block_degrees_and_chebyshev_deg3",
        |b| {
            let masks: [lmnk_paged_bitmask_t; 16] =
                core::array::from_fn(|i| lmnk_paged_bitmask_t {
                    bits: [
                        0xFFFF_0000_FFFF_0000u64.rotate_left(i as u32),
                        0x00FF_00FF_00FF_00FFu64.rotate_right(i as u32),
                        0xF0F0_F0F0_F0F0_F0F0u64,
                        0xAAAA_5555_AAAA_5555u64,
                    ],
                });
            let mut deg_buf = AlignedBuffer32::new([0.0f32; 16]);
            let mut filt_buf = AlignedBuffer32::new([0.0f32; 16]);

            b.iter(|| {
                let lambda_max = eval_stigmergic_block_degrees(
                    black_box(&masks),
                    black_box(&mut deg_buf.as_aligned_slice_mut()),
                )
                .unwrap();
                let coeffs = compute_tikhonov_chebyshev_coeffs_deg3(
                    black_box(lambda_max.max(1.0)),
                    black_box(0.05f32),
                )
                .unwrap();
                let deg_slice = deg_buf.as_aligned_slice();
                let mut filt_slice = filt_buf.as_aligned_slice_mut();
                eval_chebyshev_spectral_filter_deg3(
                    black_box(&deg_slice),
                    black_box(&mut filt_slice),
                    black_box(lambda_max.max(1.0)),
                    black_box(&coeffs),
                )
                .unwrap();
                black_box(filt_slice.as_slice()[0]);
            });
        },
    );

    // 3. Hodge Obstruction Residual & Dual Cellular Sheaf Homology Pairing (64 Supernodes)
    group.throughput(Throughput::Bytes(
        (64 * 3 * core::mem::size_of::<f32>()) as u64,
    ));
    group.bench_function("hodge_obstruction_and_dual_homology_pairing_64", |b| {
        let r_buf = AlignedBuffer32::new(make_heterogeneous_supernode_profile_64(1.25f32, 0.35f32));
        let p_buf = AlignedBuffer32::new(make_heterogeneous_supernode_profile_64(0.45f32, 0.20f32));
        let g_buf = AlignedBuffer32::new(make_heterogeneous_supernode_profile_64(0.80f32, 0.28f32));
        let r_s = r_buf.as_aligned_slice();
        let p_s = p_buf.as_aligned_slice();
        let g_s = g_buf.as_aligned_slice();

        b.iter(|| {
            let res = evaluate_hodge_obstruction_and_dual_pairing(
                black_box(&r_s),
                black_box(&p_s),
                black_box(&g_s),
                black_box(0.25f32),
            )
            .unwrap();
            black_box(res);
        });
    });

    // 4. 256-Bit Dyadic Pseudo-Ultrametric QDPT Resolution & Pointer Cascading
    group.throughput(Throughput::Elements(1));
    group.bench_function("resolve_dyadic_qdpt_and_cascade_tagged_ptr", |b| {
        let parent_ptr = lmnk_tagged_ptr_t::pack_action(
            0x8000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            10,
        );
        let curr_mask = lmnk_paged_bitmask_t {
            bits: [0xFFFF_FFFF_0000_0000, 0x1234_5678_9ABC_DEF0, 0, 0],
        };
        let anchor_mask = lmnk_paged_bitmask_t {
            bits: [0xFFFF_FFFF_0000_0000, 0x1234_5600_0000_0000, 0, 0],
        };
        let offsets = [0x100u64, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800];
        let deltas = [4i8, 8, 12, 16, 20, 24, 28, 32];

        b.iter(|| {
            let res = resolve_dyadic_qdpt_and_cascade_ptr(
                black_box(parent_ptr),
                black_box(&curr_mask),
                black_box(&anchor_mask),
                black_box(0.15f32),
                black_box(3u32),
                black_box(0.10f32),
                black_box(&offsets),
                black_box(&deltas),
            )
            .unwrap();
            black_box(res);
        });
    });

    group.finish();
}

fn bench_entropy_control_extreme_and_edge_cases(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EntropyControl_Extreme_And_Edge_Cases");
    group.noise_threshold(configured_noise_threshold());

    // 1. Long-Run Token Stream Simulation (1,000,000 Tokens with Periodic Outliers)
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(600));
    group.measurement_time(Duration::from_millis(2200));

    let ultra_long_stream_len = 1000000usize;
    let mut shared_1m_buf = AlignedHeapBuffer::<f32, 64>::new(ultra_long_stream_len, 0.5f32);
    for (i, slot) in shared_1m_buf.as_mut_slice().iter_mut().enumerate() {
        *slot = if i % 10000 == 5000 {
            1.0e4f32
        } else {
            0.5 + 2.0 * ((i as f32 * 0.005).sin().abs())
        };
    }
    group.throughput(Throughput::Elements(ultra_long_stream_len as u64));
    group.bench_function("simulate_ultra_long_stream_1m", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let z_scores = shared_1m_buf.as_slice();

        let mut engine = EntropyController::default();
        let mut state_out = lmnk_entropy_state_t::default();
        for _ in 0..2 {
            engine.reset();
            for &z in z_scores {
                engine.eval_stateful(z, &mut state_out).unwrap();
            }
        }

        b.iter(|| {
            engine.reset();
            for &z in z_scores {
                engine.eval_stateful(black_box(z), &mut state_out).unwrap();
            }
            black_box((state_out.moving_z_entropy, state_out.active_branch_mode));
        });
    });

    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1350));

    // 2. Chattering Hysteresis Stress Test (1,000 Tokens)
    let boundary_osc_len = 1000usize;
    group.throughput(Throughput::Elements(boundary_osc_len as u64));
    group.bench_function("chattering_hysteresis_boundary_oscillation", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let tau = 1.5f32;
        let delta = 0.15f32;
        let mut osc_buf = AlignedHeapBuffer::<f32, 64>::new(boundary_osc_len, tau);
        for (i, slot) in osc_buf.as_mut_slice().iter_mut().enumerate() {
            *slot = if (i & 1) == 0 {
                tau + delta + 0.05
            } else {
                tau - delta - 0.05
            };
        }
        let oscillating_z = osc_buf.as_slice();

        let mut engine = EntropyController::default();
        engine.set_bifurcation_threshold(tau);
        engine.set_hysteresis_band(delta);
        let mut state_out = lmnk_entropy_state_t::default();
        for &z in oscillating_z {
            engine.eval_stateful(z, &mut state_out).unwrap();
        }

        b.iter(|| {
            engine.reset();
            for &z in oscillating_z {
                engine.eval_stateful(black_box(z), &mut state_out).unwrap();
            }
            black_box((state_out.moving_z_entropy, state_out.active_branch_mode));
        });
    });

    // 3. Outlier & Non-Finite (NaN / +Inf / 1e9) Spike Recovery Stream
    let spike_stream_len = 1024usize;
    group.throughput(Throughput::Elements(spike_stream_len as u64));
    group.bench_function("outlier_and_nonfinite_spike_recovery_stream", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut spike_buf = AlignedHeapBuffer::<f32, 64>::new(spike_stream_len, 0.45f32);
        for (i, slot) in spike_buf.as_mut_slice().iter_mut().enumerate() {
            *slot = match i & 63 {
                16 => 1.0e9f32,
                32 => f32::INFINITY,
                48 => f32::NAN,
                _ => 0.45f32,
            };
        }
        let z_scores = spike_buf.as_slice();

        let mut engine = EntropyController::default();
        let mut state_out = lmnk_entropy_state_t::default();
        for &z in z_scores {
            let _ = engine.eval_stateful(z, &mut state_out);
        }

        b.iter(|| {
            engine.reset();
            for &z in z_scores {
                let _ = engine.eval_stateful(black_box(z), &mut state_out);
            }
            black_box(state_out.active_branch_mode);
        });
    });

    // 4. Unaligned Dyadic Vector Remainder Handling
    let shared_odd_arena = AlignedHeapBuffer::<f32, 64>::new_heterogeneous_f32(131073, 0.55f32);
    let odd_dims = [65537usize, 131073usize];
    for d in odd_dims {
        group.throughput(Throughput::Bytes((d * std::mem::size_of::<f32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("lacuna_energy_dyadic_odd_remainder", format!("dim_{d}")),
            &d,
            |b, &dim_size| {
                let slice_32 = AlignedSlice32::new(shared_odd_arena.as_prefix(dim_size)).unwrap();
                let engine = EntropyController::default();
                let mut lacuna_score = 0.0f32;
                engine
                    .eval_lacuna_energy_dyadic128(&slice_32, &mut lacuna_score)
                    .unwrap();

                b.iter(|| {
                    engine
                        .eval_lacuna_energy_dyadic128(
                            black_box(&slice_32),
                            black_box(&mut lacuna_score),
                        )
                        .unwrap();
                    black_box(lacuna_score);
                });
            },
        );
    }

    // 5. Pitchfork Bifurcation Barrier Scale Invariance Evaluation (Lemma 3.1)
    group.throughput(Throughput::Elements(1));
    group.bench_function("eval_barrier_pitchfork_lemma3_1", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        b.iter(|| {
            eval_barrier_pitchfork(
                black_box(2.5f32),
                black_box(1.0f32),
                black_box(1.2f32),
                black_box(0.8f32),
                black_box(128u32),
            )
            .unwrap()
        });
    });

    // 6. Barrier Evaluation Near Epsilon Singularity (Liminal -> 0.0)
    group.throughput(Throughput::Elements(1));
    group.bench_function("eval_barrier_epsilon_singularity_boundary", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let engine = EntropyController::default();
        b.iter(|| {
            engine
                .eval_barrier(black_box(0.000_000_000_001f32), black_box(1.0f32))
                .unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_entropy_control_stateful_eval,
    bench_entropy_control_lacuna_dyadic,
    bench_entropy_control_stream_simulation,
    bench_entropy_control_long_dialogue_reliability,
    bench_entropy_control_pointer_tunneling_and_cohomology,
    bench_entropy_control_extreme_and_edge_cases
);
criterion_main!(benches);
