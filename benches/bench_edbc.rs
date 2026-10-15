// name: benches/bench_edbc.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Advanced benchmarks for Entropy-Driven Decoding Branch Controller (EDBC),
//! stateful mode transition lifecycle simulations, CVoid coherence calculations,
//! and Pitchfork/Eyring-Kramers barrier evaluations with randomized stochastic stress tests.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use liminika_core::abi::gcso_edbc_state_t;
use liminika_core::edbc::{eval_barrier_pitchfork, EdbcEngine};
use liminika_core::traits::{AlignedSlice32, EntropyEvaluator};
use std::alloc::{alloc, dealloc, Layout};
use std::hint::black_box;

/// Heap buffer with strict memory alignment guarantee to satisfy SIMD and cacheline constraints.
struct AlignedHeapBuffer {
    ptr: *mut f32,
    layout: Layout,
    len: usize,
}

impl AlignedHeapBuffer {
    /// Allocate a new aligned heap buffer for f32 elements.
    fn new(len: usize, align: usize, initial_val: f32) -> Self {
        let layout = Layout::from_size_align(len * std::mem::size_of::<f32>(), align)
            .expect("Invalid layout for AlignedHeapBuffer");
        // SAFETY: `layout` is guaranteed to have non-zero size and valid power-of-two alignment.
        let ptr = unsafe { alloc(layout) as *mut f32 };
        assert!(
            !ptr.is_null(),
            "Heap allocation failed for AlignedHeapBuffer"
        );

        // SAFETY: `ptr` is verified non-null above and points to a newly allocated memory region valid for `len` f32 elements.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, len);
            slice.fill(initial_val);
        }

        Self { ptr, layout, len }
    }

    /// Access the buffer as a mutable slice.
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: `self.ptr` points to a valid heap allocation of `self.len` f32 elements and is guaranteed non-null.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedHeapBuffer {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `self.layout` using `alloc` and is deallocated exactly once on drop.
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

fn bench_edbc_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("EDBC_Entropy_And_Barrier_Advanced_Kernels");

    // 1. Stateful Z-Score Entropy & Hysteresis Mode Switching (Continuous trajectory)
    group.bench_function("edbc_eval_stateful_step_continuous", |b| {
        let mut engine = EdbcEngine::default();
        let mut out_state = gcso_edbc_state_t::default();
        let mut z_score = 1.2f32;

        b.iter(|| {
            z_score += 0.01f32;
            if z_score > 3.0f32 {
                z_score = 0.5f32;
            }
            engine
                .eval_stateful(black_box(z_score), black_box(&mut out_state))
                .unwrap();
        });
    });

    // 1.1 Chattering Stress Test (Rapid boundary oscillation simulating adversarial decoding states)
    group.bench_function("edbc_eval_stateful_chattering_stress", |b| {
        let mut engine = EdbcEngine::default();
        let mut out_state = gcso_edbc_state_t::default();
        let boundary_oscillation = [
            1.48f32, 1.52, 1.49, 1.51, 1.48, 1.53, 1.50, 1.47, 1.52, 1.50,
        ];

        b.iter(|| {
            for &z in &boundary_oscillation {
                engine
                    .eval_stateful(black_box(z), black_box(&mut out_state))
                    .unwrap();
            }
        });
    });

    // 2. Full Lifecycle Sequence Simulation (Normal ⇄ Bifurcation ⇄ Repulsion mode loop)
    group.bench_function("edbc_eval_stateful_full_lifecycle_sequence", |b| {
        let mut engine = EdbcEngine::default();
        let mut out_state = gcso_edbc_state_t::default();
        let stimulus_sequence = [
            0.4f32, 0.6, 1.2, 1.8, 2.5, 4.0, 5.2, 3.1, 1.5, 0.8, 0.3, 0.5, 2.1, 3.8, 4.5, 1.1, 0.2,
            0.9, 2.8, 4.9, 3.5, 1.9, 0.7, 0.4,
        ];

        b.iter(|| {
            for &z in &stimulus_sequence {
                engine
                    .eval_stateful(black_box(z), black_box(&mut out_state))
                    .unwrap();
            }
        });
    });

    // 2.1 Stochastically Perturbed Stress Sequence (Simulating unpredictable real-world token uncertainty)
    group.bench_function("edbc_eval_stateful_stochastic_stress", |b| {
        let mut engine = EdbcEngine::default();
        let mut out_state = gcso_edbc_state_t::default();
        let mut lcg_state = 42u32;
        let mut next_z = || {
            lcg_state = lcg_state.wrapping_mul(1103515245).wrapping_add(12345);
            ((lcg_state >> 16) as f32 / 65535.0) * 5.0
        };

        b.iter(|| {
            let z = next_z();
            engine
                .eval_stateful(black_box(z), black_box(&mut out_state))
                .unwrap();
        });
    });

    // 3. Parameterized CVoid Dyadic Coherence Score Evaluation (Extended up to 2048 dim for deep layer stress testing)
    for &dim in &[64usize, 128usize, 256usize, 512usize, 1024usize, 2048usize] {
        let bytes = dim * core::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("edbc_eval_cvoid_dyadic_dim", dim),
            &dim,
            |b, &d| {
                let engine = EdbcEngine::default();
                let mut vec_data = AlignedHeapBuffer::new(d, 32, 1.5f32);
                let mut void_score = 0.0f32;

                b.iter(|| {
                    let slice_mut = vec_data.as_mut_slice();
                    let key_slice = AlignedSlice32::new(slice_mut).unwrap();
                    engine
                        .eval_cvoid_dyadic128(black_box(&key_slice), black_box(&mut void_score))
                        .unwrap();
                });
            },
        );
    }

    // 4. Eyring-Kramers Potential Barrier Height
    group.bench_function("edbc_eval_barrier_eyring_kramers", |b| {
        let engine = EdbcEngine::default();
        let void_score = 2.5f32;
        let tau_eff = 1.0f32;

        b.iter(|| {
            black_box(
                engine
                    .eval_barrier(black_box(void_score), black_box(tau_eff))
                    .unwrap(),
            );
        });
    });

    // 5. Lemma 3.1 Pitchfork Bifurcation Barrier Height across Head Dimensions
    for &head_dim in &[32u32, 64u32, 128u32, 256u32] {
        let cvoid = 2.0f32;
        let tau_eff = 1.0f32;
        let alpha = 1.0f32;
        let beta = 1.0f32;

        group.bench_with_input(
            BenchmarkId::new("edbc_eval_barrier_pitchfork_head_dim", head_dim),
            &head_dim,
            |b, &h_dim| {
                b.iter(|| {
                    black_box(
                        eval_barrier_pitchfork(
                            black_box(cvoid),
                            black_box(tau_eff),
                            black_box(alpha),
                            black_box(beta),
                            black_box(h_dim),
                        )
                        .unwrap(),
                    );
                });
            },
        );
    }

    // 6. Combined CVoid Coherence + Barrier Pipeline
    group.bench_function("edbc_cvoid_and_barrier_pipeline", |b| {
        let engine = EdbcEngine::default();
        let mut vec_data = AlignedHeapBuffer::new(128, 32, 1.8f32);
        let mut void_score = 0.0f32;
        let tau_eff = 1.0f32;

        b.iter(|| {
            let slice_mut = vec_data.as_mut_slice();
            let key_slice = AlignedSlice32::new(slice_mut).unwrap();
            engine
                .eval_cvoid_dyadic128(black_box(&key_slice), black_box(&mut void_score))
                .unwrap();
            black_box(
                engine
                    .eval_barrier(black_box(void_score), black_box(tau_eff))
                    .unwrap(),
            );
        });
    });

    group.finish();
}

criterion_group!(benches, bench_edbc_operations);
criterion_main!(benches);
