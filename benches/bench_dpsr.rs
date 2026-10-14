// name: benches/bench_dpsr.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Advanced benchmarks for Dynamic Phase-Shifted RoPE (DPSR) hot-path kernels,
//! featuring parameterized batch sizes, throughput measurements, pipeline sequences,
//! lazy phase unwrapping, norm-guarded stabilization, and multi-alignment cache evaluations.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use liminika_core::abi::gcso_q7_t;
use liminika_core::dpsr::DpsrEngine;
use liminika_core::traits::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut128, AlignedSliceMut16, AlignedSliceMut32,
    AlignedSliceMut64, PhaseSteering,
};
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
        let ptr = unsafe { alloc(layout) as *mut f32 };
        assert!(
            !ptr.is_null(),
            "Heap allocation failed for AlignedHeapBuffer"
        );

        // Initialize buffer with valid non-zero values to prevent singularity errors
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr, len);
            slice.fill(initial_val);
        }

        Self { ptr, layout, len }
    }

    /// Access the buffer as a mutable slice.
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedHeapBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

#[repr(align(16))]
struct AlignedQ7Buffer<const N: usize>([gcso_q7_t; N]);

fn bench_dpsr_steering_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("DPSR_HotPath_Advanced_Kernels");

    let configurations = vec![
        (64u32, 8u32),
        (128u32, 16u32),
        (128u32, 32u32),
        (128u32, 64u32),
        (256u32, 32u32), // Large-scale head dimension stress evaluation
    ];

    for &(head_dim, num_heads) in &configurations {
        let total_elements = (head_dim * num_heads) as usize;
        let total_bytes = total_elements * core::mem::size_of::<f32>();
        let engine = DpsrEngine::new(head_dim, num_heads).expect("Failed to initialize DpsrEngine");

        group.throughput(Throughput::Bytes(total_bytes as u64));

        // 1. Alignment Comparison: Unaligned vs 32B vs 64B vs 128B Cacheline Boundaries
        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_unaligned_baseline",
                format!("{}x{}", head_dim, num_heads),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let mut q_vec = vec![1.0f32; total_elements.max(16384)];
                let phase_data = AlignedQ7Buffer([5i8; 64]);

                b.iter(|| {
                    let q_slice_32 = AlignedSliceMut32::new(&mut q_vec[..total_elements]);
                    let p_slice =
                        AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();

                    if let Ok(mut slice) = q_slice_32 {
                        engine
                            .apply_phase_steering(black_box(&mut slice), black_box(&p_slice))
                            .unwrap();
                    } else {
                        black_box(&mut q_vec[..total_elements]);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_32b",
                format!("{}x{}", head_dim, num_heads),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let mut q_data = AlignedHeapBuffer::new(16384, 32, 1.0f32);
                let phase_data = AlignedQ7Buffer([5i8; 64]);

                b.iter(|| {
                    let slice_mut = q_data.as_mut_slice();
                    let mut q_slice =
                        AlignedSliceMut32::new(&mut slice_mut[..total_elements]).unwrap();
                    let p_slice =
                        AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();

                    engine
                        .apply_phase_steering(black_box(&mut q_slice), black_box(&p_slice))
                        .unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_64b",
                format!("{}x{}", head_dim, num_heads),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let mut q_data = AlignedHeapBuffer::new(16384, 64, 1.0f32);
                let phase_data = AlignedQ7Buffer([5i8; 64]);

                b.iter(|| {
                    let slice_mut = q_data.as_mut_slice();
                    let mut q_slice =
                        AlignedSliceMut64::new(&mut slice_mut[..total_elements]).unwrap();
                    let p_slice =
                        AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();
                    let mut q_slice_32 = AlignedSliceMut32::new(q_slice.as_mut_slice()).unwrap();

                    engine
                        .apply_phase_steering(black_box(&mut q_slice_32), black_box(&p_slice))
                        .unwrap();
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_128b",
                format!("{}x{}", head_dim, num_heads),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let mut q_data = AlignedHeapBuffer::new(16384, 128, 1.0f32);
                let phase_data = AlignedQ7Buffer([5i8; 64]);

                b.iter(|| {
                    let slice_mut = q_data.as_mut_slice();
                    let mut q_slice =
                        AlignedSliceMut128::new(&mut slice_mut[..total_elements]).unwrap();
                    let p_slice =
                        AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();
                    let mut q_slice_32 = AlignedSliceMut32::new(q_slice.as_mut_slice()).unwrap();

                    engine
                        .apply_phase_steering(black_box(&mut q_slice_32), black_box(&p_slice))
                        .unwrap();
                });
            },
        );

        // 2. Fused Query & Key Phase Steering
        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_fused",
                format!("{}x{}", head_dim, num_heads),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let mut q_data = AlignedHeapBuffer::new(16384, 32, 1.0f32);
                let mut k_data = AlignedHeapBuffer::new(16384, 32, 1.0f32);
                let phase_data = AlignedQ7Buffer([8i8; 64]);

                b.iter(|| {
                    let q_slice_mut = q_data.as_mut_slice();
                    let k_slice_mut = k_data.as_mut_slice();
                    let mut q_slice =
                        AlignedSliceMut32::new(&mut q_slice_mut[..total_elements]).unwrap();
                    let mut k_slice =
                        AlignedSliceMut32::new(&mut k_slice_mut[..total_elements]).unwrap();
                    let p_slice =
                        AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();

                    engine
                        .apply_phase_steering_fused(
                            black_box(&mut q_slice),
                            black_box(&mut k_slice),
                            black_box(&p_slice),
                        )
                        .unwrap();
                });
            },
        );

        // 3. RIPA Low-Frequency Bounded Steering
        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_safe_ripa",
                format!("{}x{}", head_dim, num_heads),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let mut q_data = AlignedHeapBuffer::new(16384, 32, 1.0f32);
                let phase_data = AlignedQ7Buffer([12i8; 64]);
                let max_rad = 0.08726646f32;

                b.iter(|| {
                    let slice_mut = q_data.as_mut_slice();
                    let mut q_slice =
                        AlignedSliceMut32::new(&mut slice_mut[..total_elements]).unwrap();
                    let p_slice =
                        AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();

                    engine
                        .apply_phase_steering_safe(
                            black_box(&mut q_slice),
                            black_box(&p_slice),
                            black_box(max_rad),
                        )
                        .unwrap();
                });
            },
        );

        // 4. Lazy Phase Unwrapping
        group.bench_with_input(
            BenchmarkId::new(
                "lazy_unwrap_override",
                format!("{}x{}", head_dim, num_heads),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let mut q_data = AlignedHeapBuffer::new(16384, 32, 1.0f32);
                let mut context_accum = AlignedHeapBuffer::new(64, 32, 0.1f32);

                b.iter(|| {
                    let q_slice_mut = q_data.as_mut_slice();
                    let c_slice_mut = context_accum.as_mut_slice();
                    let mut q_slice =
                        AlignedSliceMut32::new(&mut q_slice_mut[..total_elements]).unwrap();
                    let c_slice =
                        AlignedSlice32::new(&c_slice_mut[..(num_heads as usize)]).unwrap();

                    engine
                        .lazy_unwrap_override(black_box(&mut q_slice), black_box(&c_slice))
                        .unwrap();
                });
            },
        );
    }

    // 5. QDPS Filter Step
    for &num_heads in &[16usize, 32usize, 64usize] {
        let engine = DpsrEngine::new(128, num_heads as u32).unwrap();
        let min_step_rad = 0.05f32;

        group.throughput(Throughput::Elements(num_heads as u64));
        group.bench_with_input(
            BenchmarkId::new("qdps_filter_step", num_heads),
            &num_heads,
            |b, &n| {
                b.iter_batched(
                    || {
                        AlignedQ7Buffer([
                            1, 5, -2, 10, -128, 4, 0, 15, 2, 3, 4, 5, 6, 7, 8, 9, 1, 5, -2, 10,
                            -128, 4, 0, 15, 2, 3, 4, 5, 6, 7, 8, 9, 2, 4, 6, 8, -10, -20, 30, 40,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4, 5,
                            6, 7, 8,
                        ])
                    },
                    |mut phase_buf| {
                        let mut p_slice = AlignedSliceMut16::new(&mut phase_buf.0[..n]).unwrap();
                        engine
                            .qdps_filter_step(black_box(&mut p_slice), black_box(min_step_rad))
                            .unwrap();
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    // 6. Procrustes Phase Delta
    let default_engine = DpsrEngine::new(128, 32).unwrap();
    group.bench_function("dpsr_compute_procrustes_phase_delta_128x32", |b| {
        let mut src_data = AlignedHeapBuffer::new(4096, 32, 1.0f32);
        let mut tgt_data = AlignedHeapBuffer::new(4096, 32, 0.5f32);
        let mut out_phase = AlignedQ7Buffer([0i8; 32]);

        b.iter(|| {
            let src_slice = AlignedSlice32::new(src_data.as_mut_slice()).unwrap();
            let tgt_slice = AlignedSlice32::new(tgt_data.as_mut_slice()).unwrap();
            let mut out_slice = AlignedSliceMut16::new(&mut out_phase.0).unwrap();

            default_engine
                .compute_procrustes_phase_delta(
                    black_box(&src_slice),
                    black_box(&tgt_slice),
                    black_box(&mut out_slice),
                )
                .unwrap();
        });
    });

    // 7. Fused Logit Shift (Vocabulary scale: 32k, 64k & 128k)
    let mut logits_buf = AlignedHeapBuffer::new(131072, 32, 1.0f32);
    for &vocab_size in &[32000usize, 64000usize, 128000usize] {
        group.throughput(Throughput::Bytes(
            (vocab_size * core::mem::size_of::<f32>()) as u64,
        ));
        group.bench_with_input(
            BenchmarkId::new("dpsr_fused_logit_shift_vocab", vocab_size),
            &vocab_size,
            |b, &v_size| {
                let phase_data = AlignedQ7Buffer([10i8; 32]);

                b.iter(|| {
                    let logits_slice_mut = logits_buf.as_mut_slice();
                    let mut logits_slice =
                        AlignedSliceMut32::new(&mut logits_slice_mut[..v_size]).unwrap();
                    let p_slice = AlignedSlice16::new(&phase_data.0).unwrap();

                    default_engine
                        .fused_logit_shift(black_box(&mut logits_slice), black_box(&p_slice))
                        .unwrap();
                });
            },
        );
    }

    // 8. Slerp Norm Guard Stabilization
    group.bench_function("dpsr_slerp_norm_guard_stable_128x32", |b| {
        let mut t_data = AlignedHeapBuffer::new(4096, 32, 1.0f32);
        let norm_lower = 0.9f32;
        let norm_upper = 1.1f32;

        b.iter(|| {
            let mut t_slice = AlignedSliceMut32::new(t_data.as_mut_slice()).unwrap();
            default_engine
                .slerp_norm_guard_stable(
                    black_box(&mut t_slice),
                    black_box(norm_lower),
                    black_box(norm_upper),
                )
                .unwrap();
        });
    });

    // 9. Hot-Path Comprehensive Pipeline
    group.bench_function("dpsr_hotpath_comprehensive_pipeline", |b| {
        let mut q_data = AlignedHeapBuffer::new(4096, 32, 1.0f32);
        let phase_data = AlignedQ7Buffer([6i8; 32]);
        let damping_factor = 0.98f32;
        let norm_lower = 0.5f32;
        let norm_upper = 2.0f32;

        b.iter(|| {
            let mut q_slice = AlignedSliceMut32::new(q_data.as_mut_slice()).unwrap();
            let p_slice = AlignedSlice16::new(&phase_data.0).unwrap();

            default_engine
                .apply_phase_steering(black_box(&mut q_slice), black_box(&p_slice))
                .unwrap();
            default_engine
                .apply_soft_phase_damping(black_box(&mut q_slice), black_box(damping_factor))
                .unwrap();
            default_engine
                .slerp_norm_guard_stable(
                    black_box(&mut q_slice),
                    black_box(norm_lower),
                    black_box(norm_upper),
                )
                .unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, bench_dpsr_steering_operations);
criterion_main!(benches);
