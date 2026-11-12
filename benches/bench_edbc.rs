// name: benches/bench_edbc.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Comprehensive benchmarks for Entropy-Driven Decoding Branch Controller (EDBC) kernels and
//! `EdbcClosedLoopCoordinator` blanket implementations, engineered for **zero input overfitting**
//! and **clean separation of Provisional `.gcso` / ZIMMS Container workloads**:
//! - Evaluates Coarse-to-Fine and Top-2 Liminal routing across 8 candidate trails (2 candidates
//!   per macro cluster across 4 centroids) so the Fine step genuinely resolves intra-cluster
//!   candidate competition.
//! - Isolates provisional pre-v1.0 `.gcso` / ZIMMS container crystallization and zero-copy
//!   deserialization workloads into the dedicated `EDBC_Provisional_GCSO_Container` Criterion group.

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
    gcso_daes_slot_t, gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t,
    gcso_tagged_ptr_t,
};
use liminika_core::edbc::{
    compute_moving_z_score_entropy, compute_sparse_surrogate_entropy, eval_barrier_pitchfork,
    EdbcClosedLoopCoordinator, EdbcEngine,
};
use liminika_core::traits::{
    AlignedBuffer128, AlignedBuffer32, AlignedBuffer64, AlignedSlice32, ContradictionGuard,
    ContrastivePhaseBloomFilter, CrystallizedAttractorRecord, EntropyEvaluator,
    GcsoFourSectionView, HierarchicalTreeTable, ManifoldGatewaySignal, ObjectiveFact,
    SheafCohomologyEvaluator, StigmergicActionPtrOps, SubjectivePersona,
    ACTION_PRESET_EXPRESS_PHASE, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    MAX_ATTENTION_HEADS,
};
use std::alloc::{alloc, dealloc, Layout};
use std::hint::black_box;
use std::time::Duration;

/// Scoped x86_64 MXCSR Flush-To-Zero (`FTZ`, bit 15 = `0x8000`) and Denormals-Are-Zero
/// (`DAZ`, bit 6 = `0x0040`) guard preventing hardware subnormal microcode traps across
/// multi-hundred-thousand-iteration Criterion linear sample batches.
struct ScopedFtzdazGuard {
    #[cfg(target_arch = "x86_64")]
    prev_csr: u32,
}

impl ScopedFtzdazGuard {
    #[inline(always)]
    fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: Reading and setting FTZ (`0x8000`) and DAZ (`0x0040`) bits in MXCSR via
        // `stmxcsr` / `ldmxcsr` is safe per-thread floating-point state control.
        unsafe {
            let mut prev_csr = 0u32;
            core::arch::asm!(
                "stmxcsr [{p}]",
                p = in(reg) &mut prev_csr,
                options(nostack, preserves_flags),
            );
            let new_csr = prev_csr | 0x8040;
            core::arch::asm!(
                "ldmxcsr [{p}]",
                p = in(reg) &new_csr,
                options(readonly, nostack, preserves_flags),
            );
            Self { prev_csr }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {}
        }
    }
}

impl Drop for ScopedFtzdazGuard {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: Restores the previous MXCSR register state captured at construction via `ldmxcsr`.
        unsafe {
            core::arch::asm!(
                "ldmxcsr [{p}]",
                p = in(reg) &self.prev_csr,
                options(readonly, nostack, preserves_flags),
            );
        }
    }
}

/// Generic heap buffer with strict 32-byte (or 64-byte cacheline) alignment for SIMD CVoid evaluation.
struct AlignedVectorBuffer<T, const ALIGN: usize = 32> {
    ptr: *mut T,
    layout: Layout,
    len: usize,
}

impl<T: Copy, const ALIGN: usize> AlignedVectorBuffer<T, ALIGN> {
    fn new(len: usize, initial_val: T) -> Self {
        assert!(ALIGN > 0 && ALIGN.is_power_of_two());
        let elem_size = std::mem::size_of::<T>();
        assert!(len > 0 && elem_size > 0);
        let byte_len = len.checked_mul(elem_size).expect("Byte length overflow");
        let alloc_align = if ALIGN < 4096 { 4096 } else { ALIGN };
        let layout = Layout::from_size_align(byte_len, alloc_align).expect("Invalid layout");
        // SAFETY: `layout` has non-zero size and power-of-two alignment.
        let ptr = unsafe { alloc(layout).cast::<T>() };
        assert!(!ptr.is_null(), "Heap allocation failed");
        // SAFETY: `ptr` is non-null and valid for `len` elements.
        unsafe {
            std::slice::from_raw_parts_mut(ptr, len).fill(initial_val);
        }
        Self { ptr, layout, len }
    }

    fn new_heterogeneous_f32(len: usize, base_val: f32) -> AlignedVectorBuffer<f32, ALIGN> {
        let mut buf = AlignedVectorBuffer::<f32, ALIGN>::new(len, base_val);
        for (i, elem) in buf.as_mut_slice().iter_mut().enumerate() {
            let wave = (((i.wrapping_mul(17).wrapping_add((i >> 7).wrapping_mul(31))) & 255)
                as f32)
                * (1.0f32 / 255.0f32);
            *elem = base_val + 0.90f32 * wave;
        }
        buf
    }

    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        // SAFETY: `self.ptr` points to a valid allocation of `self.len` elements.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `self.ptr` points to a valid and uniquely borrowed allocation of `self.len` elements.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    #[inline(always)]
    fn as_prefix(&self, prefix_len: usize) -> &[T] {
        assert!(prefix_len <= self.len);
        // SAFETY: `self.ptr` points to a valid allocation of `prefix_len <= self.len` elements.
        unsafe { std::slice::from_raw_parts(self.ptr, prefix_len) }
    }
}

impl<T, const ALIGN: usize> Drop for AlignedVectorBuffer<T, ALIGN> {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` was allocated with `self.layout`.
        unsafe {
            dealloc(self.ptr.cast::<u8>(), self.layout);
        }
    }
}

/// Generates a deterministic 64-supernode f32 field with non-zero spatial coboundary gradients
/// across adjacent supernodes so Sheaf Cohomology and Hodge Obstruction kernels evaluate genuine
/// non-zero coboundary residuals rather than flat constant vectors.
#[inline(always)]
const fn make_heterogeneous_supernode_field_64(base_val: f32, spread: f32) -> [f32; 64] {
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

/// Restores `dst` from `src` using 64-bit GPR stores (`store_q7_words_64`) at exact 8-byte offsets
/// (`offset 56..120`), preventing the 256-bit YMM-store-to-64-bit-GPR-load Store-to-Load Forwarding
/// (STLF) pipeline stall (`~15–20 cycles`) caused by `*dst = *src` 128-byte struct assignment.
#[inline(always)]
fn reset_trail_fast(dst: &mut gcso_pointer_trail_t, src: &gcso_pointer_trail_t) {
    dst.current_ptr = src.current_ptr;
    dst.prev_ptr = src.prev_ptr;
    dst.user_data = src.user_data;
    dst.transition_cost = src.transition_cost;
    dst.step_count = src.step_count;
    dst.stigmergic_density = src.stigmergic_density;
    dst.target_anchor_id = src.target_anchor_id;
    dst.cluster_id = src.cluster_id;
    dst.linked_trail_id = src.linked_trail_id;
    dst.attractor_pull_force = src.attractor_pull_force;
    dst.flags = src.flags;
    *liminika_core::traits::branchless::trail_phase_words_mut(dst) =
        *liminika_core::traits::branchless::trail_phase_words_ref(src);
}

/// Restores `dst` from `src` using [`EdbcEngine::restore_from`] (width-matched scalar stores),
/// preventing the 256-bit `vmovaps` YMM store-to-scalar-load STLF pipeline stall inside `b.iter`.
#[inline(always)]
fn reset_engine_fast(dst: &mut EdbcEngine, src: &EdbcEngine) {
    dst.restore_from(src);
}

#[inline(always)]
fn reset_crystal_fast(dst: &mut CrystallizedAttractorRecord, src: &CrystallizedAttractorRecord) {
    dst.magic = src.magic;
    dst.version = src.version;
    dst.anchor_id = src.anchor_id;
    dst.sample_count = src.sample_count;
    dst.zimms_section_offset48 = src.zimms_section_offset48;
    dst.base_energy = src.base_energy;
    dst.resultant_r_bar = src.resultant_r_bar;
    dst.von_mises_kappa = src.von_mises_kappa;
    dst.phase_gravity_mass = src.phase_gravity_mass;
    dst.ripa_clamp_rad = src.ripa_clamp_rad;
    dst.effective_inv_temp = src.effective_inv_temp;
    dst.provenance_bits = src.provenance_bits;
    dst.checksum_crc32 = src.checksum_crc32;
    dst.dominant_slot = src.dominant_slot;
    dst.action_mode_flags = src.action_mode_flags;
    dst.active_swarm_mask = src.active_swarm_mask;
    dst.dominant_cause_code = src.dominant_cause_code;
    dst.reserved_align64 = src.reserved_align64;
    *dst.phase_words_mut() = *src.phase_words_ref();
}

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

/// Reads `LIMINIKA_NOISE_THRESHOLD` (defaulting to `0.025` = 2.5%) so every Criterion group
/// suppresses sub-30ns RDTSC timer-quantum and background OS scheduler false alarms whether
/// invoked via `python scripts/run_benchmarks.py` or bare `cargo bench`.
#[inline]
fn configured_noise_threshold() -> f64 {
    std::env::var("LIMINIKA_NOISE_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|&v| v.is_finite() && v > 0.0 && v < 1.0)
        .unwrap_or(0.025)
}

/// Determines whether the isolated pre-v1.0 `Provisional_GCSO_Container` benchmark group
/// should be registered and executed.
///
/// Enabled only when `LIMINIKA_RUN_PROVISIONAL_GCSO` is set to `"1"`/`"true"`, or when the
/// user explicitly passes a positional Criterion CLI filter targeting `Provisional_GCSO_Container`.
#[inline]
fn should_run_provisional_gcso() -> bool {
    if let Ok(val) = std::env::var("LIMINIKA_RUN_PROVISIONAL_GCSO") {
        let trimmed = val.trim();
        if trimmed == "1" || trimmed.eq_ignore_ascii_case("true") {
            return true;
        }
        if trimmed == "0" || trimmed.eq_ignore_ascii_case("false") {
            return false;
        }
    }
    std::env::args()
        .skip(1)
        .any(|arg| !arg.starts_with('-') && arg.to_ascii_lowercase().contains("provisional"))
}

fn bench_edbc_stateful_eval(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EDBC_Stateful_Entropy_Evaluation");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1350));
    group.throughput(Throughput::Elements(1));

    // 1. Single Token Stateful Evaluation Benchmarks (Pre-warmed Stationary Regimes)
    group.bench_function("eval_stateful_normal_mode", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut engine = EdbcEngine::default();
        let mut state_out = gcso_edbc_state_t::default();
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
        let mut base_engine = EdbcEngine::default();
        base_engine.set_bifurcation_threshold(1.0);
        let mut state_out = gcso_edbc_state_t::default();
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
        let mut base_engine = EdbcEngine::default();
        base_engine.set_bifurcation_threshold(1.0);
        base_engine.set_tau_flux(0.2);
        let mut state_out = gcso_edbc_state_t::default();
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
            let h_s = compute_sparse_surrogate_entropy(0.72f32, 0.21f32, 1e-12f32);
            black_box(compute_moving_z_score_entropy(
                h_s, 0.35f32, 0.12f32, 0.08f32, 1e-6f32, 1.0f32, 0.5f32,
            ));
        }

        b.iter(|| {
            let h_surr = compute_sparse_surrogate_entropy(
                black_box(0.72f32),
                black_box(0.21f32),
                black_box(1e-12f32),
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

    // 2B. Standalone 0-ln 9-Term Even Taylor-Horner Polynomial Surrogate Entropy
    group.bench_function("compute_sparse_surrogate_taylor_horner_0ln", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        for _ in 0..8 {
            black_box(compute_sparse_surrogate_entropy(0.72f32, 0.28f32, 1e-12f32));
        }

        b.iter(|| {
            compute_sparse_surrogate_entropy(
                black_box(0.72f32),
                black_box(0.28f32),
                black_box(1e-12f32),
            )
        });
    });

    // 2C. Zero-Division Deterministic / Symmetric Fast-Path Surrogate Entropy
    group.bench_function("compute_sparse_surrogate_deterministic_fastpath", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        for _ in 0..8 {
            let hg = compute_sparse_surrogate_entropy(0.99f32, 0.0f32, 1e-12f32);
            let ht = compute_sparse_surrogate_entropy(0.50f32, 0.50f32, 1e-12f32);
            black_box(hg + ht);
        }

        b.iter(|| {
            let h_greedy = compute_sparse_surrogate_entropy(
                black_box(0.99f32),
                black_box(0.0f32),
                black_box(1e-12f32),
            );
            let h_tie = compute_sparse_surrogate_entropy(
                black_box(0.50f32),
                black_box(0.50f32),
                black_box(1e-12f32),
            );
            black_box(h_greedy + h_tie)
        });
    });

    // 3. Fused Top-2 Surrogate + Moving Z-Score + Stateful EDBC Step
    group.bench_function("eval_surrogate_and_step_fused", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut engine = EdbcEngine::default();
        let mut state_out = gcso_edbc_state_t::default();
        for _ in 0..8 {
            engine
                .eval_surrogate_and_step(0.68, 0.24, 0.35, 0.15, 0.09, 1.0, 0.5, &mut state_out)
                .unwrap();
        }

        b.iter(|| {
            let (p_max, p_2nd, cos_d) = black_box((0.68f32, 0.24f32, 0.35f32));
            let h = engine
                .eval_surrogate_and_step(
                    p_max,
                    p_2nd,
                    cos_d,
                    0.15f32,
                    0.09f32,
                    1.0f32,
                    0.5f32,
                    &mut state_out,
                )
                .unwrap();
            black_box((h, state_out.active_branch_mode));
        });
    });

    // 4. O(1) Branchless 2nd-Order Entropy Flux Acceleration & Adaptive PPRC GOP Budget
    group.bench_function("evaluate_adaptive_compute_and_pprc_gop_budget", |b| {
        let mut engine = EdbcEngine::default();
        let mut state_out = gcso_edbc_state_t::default();
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

fn bench_edbc_cvoid_dyadic(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EDBC_CVoid_Dyadic_Vector_Alignment");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1350));

    let vector_dims = [128usize, 512, 2048, 8192, 32768, 65536, 131072, 262144];
    let shared_cvoid_arena = AlignedVectorBuffer::<f32, 64>::new_heterogeneous_f32(262144, 0.65f32);

    for dim in vector_dims {
        let total_bytes = dim * std::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(total_bytes as u64));

        group.bench_with_input(
            BenchmarkId::new("eval_cvoid_dyadic128", format!("dim_{dim}")),
            &dim,
            |b, &d| {
                let _ftz = ScopedFtzdazGuard::new();
                let slice_32 = AlignedSlice32::new(shared_cvoid_arena.as_prefix(d)).unwrap();
                let engine = EdbcEngine::default();
                let mut void_score = 0.0f32;
                for _ in 0..8 {
                    engine
                        .eval_cvoid_dyadic128(&slice_32, &mut void_score)
                        .unwrap();
                }

                b.iter(|| {
                    engine
                        .eval_cvoid_dyadic128(black_box(&slice_32), black_box(&mut void_score))
                        .unwrap();
                    black_box(void_score);
                });
            },
        );
    }

    let coh_dim = 4096usize;
    group.throughput(Throughput::Bytes(
        (coh_dim * std::mem::size_of::<f32>()) as u64,
    ));
    group.bench_function("eval_cvoid_with_l1_l2_coherence_ratio_dim_4096", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let slice_32 = AlignedSlice32::new(shared_cvoid_arena.as_prefix(coh_dim)).unwrap();
        let engine = EdbcEngine::default();
        for _ in 0..8 {
            let _ = engine.eval_cvoid_with_coherence_ratio(&slice_32).unwrap();
        }

        b.iter(|| {
            let res = engine
                .eval_cvoid_with_coherence_ratio(black_box(&slice_32))
                .unwrap();
            black_box(res);
        });
    });

    group.bench_function(
        "eval_surrogate_with_cvoid_and_pointer_fused_dim_4096",
        |b| {
            let _ftz = ScopedFtzdazGuard::new();
            let slice_32 = AlignedSlice32::new(shared_cvoid_arena.as_prefix(coh_dim)).unwrap();
            let mut base_engine = EdbcEngine::default().with_min_mode_dwell_steps(3);
            let mut st_warm = gcso_edbc_state_t::default();
            for _ in 0..8 {
                base_engine.eval_stateful(0.55f32, &mut st_warm).unwrap();
            }
            let ptr = gcso_tagged_ptr_t::pack_action(
                0x4000,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_EXPRESS_PHASE,
                16,
            );

            let mut engine = base_engine;
            for _ in 0..8 {
                reset_engine_fast(&mut engine, &base_engine);
                let _ = engine
                    .eval_surrogate_with_cvoid_and_pointer(
                        0.62f32,
                        0.31f32,
                        0.38f32,
                        0.15f32,
                        0.08f32,
                        1.0f32,
                        0.5f32,
                        &slice_32,
                        1.2f32,
                        0.8f32,
                        0.5f32,
                        ptr,
                        1.1f32,
                        ATTRACTOR_SLOT_KNOWLEDGE,
                        1.5f32,
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_engine_fast(&mut engine, &base_engine);
                let res = engine
                    .eval_surrogate_with_cvoid_and_pointer(
                        black_box(0.62f32),
                        black_box(0.31f32),
                        black_box(0.38f32),
                        black_box(0.15f32),
                        black_box(0.08f32),
                        black_box(1.0f32),
                        black_box(0.5f32),
                        black_box(&slice_32),
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
        },
    );

    group.finish();
}

fn bench_edbc_stream_simulation(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EDBC_Continuous_Stream_Simulation");
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
                let mut z_buf = AlignedVectorBuffer::<f32, 64>::new_heterogeneous_f32(len, 0.5f32);
                for (i, slot) in z_buf.as_mut_slice().iter_mut().enumerate() {
                    let base = 0.5 + 1.5 * ((i as f32 * 0.017).sin().abs());
                    *slot = if i % 150 == 0 { base + 4.5 } else { base };
                }
                let z_scores = z_buf.as_prefix(len);

                let mut engine = EdbcEngine::default();
                let mut state_out = gcso_edbc_state_t::default();
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

fn bench_edbc_long_dialogue_reliability(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EDBC_Long_Dialogue_Reliability");
    group.noise_threshold(configured_noise_threshold());

    // 1. Ultra-Long Token Stream Simulation (500,000 Tokens)
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(350));
    group.measurement_time(Duration::from_millis(1100));

    let long_stream_len = 500000usize;
    let mut shared_500k_buf = AlignedVectorBuffer::<f32, 64>::new(long_stream_len, 0.4f32);
    for (i, slot) in shared_500k_buf.as_mut_slice().iter_mut().enumerate() {
        *slot = 0.4 + 1.8 * ((i as f32 * 0.003).sin().abs());
    }
    group.throughput(Throughput::Elements(long_stream_len as u64));
    group.bench_function("simulate_ultra_long_stream_500k", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let z_scores = shared_500k_buf.as_slice();

        let mut engine = EdbcEngine::default();
        let mut state_out = gcso_edbc_state_t::default();
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
    let mut shared_osc_50k = AlignedVectorBuffer::<f32, 64>::new(chattering_len, tau);
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

        let mut engine = EdbcEngine::default();
        engine.set_bifurcation_threshold(tau);
        engine.set_hysteresis_band(delta);
        let mut state_out = gcso_edbc_state_t::default();
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

        let mut engine = EdbcEngine::default().with_min_mode_dwell_steps(4);
        engine.set_bifurcation_threshold(tau);
        engine.set_hysteresis_band(delta);
        let mut state_out = gcso_edbc_state_t::default();
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
        let patterns: [[gcso_q7_t; MAX_ATTENTION_HEADS]; 8] = core::array::from_fn(|k| {
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

fn bench_edbc_pointer_tunneling_and_cohomology(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EDBC_Pointer_Tunneling_And_Cohomology");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1350));

    // 1. Pointer-Coupled EDBC Step + Eyring-Kramers Temperature + Stigmergic Mode Evolution
    group.throughput(Throughput::Elements(1));
    group.bench_function("eval_stateful_with_cvoid_and_pointer_evolution", |b| {
        let mut base_engine = EdbcEngine::default().with_min_mode_dwell_steps(3);
        let mut st_warm = gcso_edbc_state_t::default();
        for _ in 0..8 {
            base_engine.eval_stateful(1.1f32, &mut st_warm).unwrap();
        }
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x4000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        );

        let mut engine = base_engine;

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            let res = engine
                .eval_stateful_with_cvoid_and_pointer(
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

    // 2. Closed-Loop Pitchfork Bifurcation + SplitMix64 Metastable Tunneling Perturbation
    group.throughput(Throughput::Elements(1));
    group.bench_function("step_and_inject_metastable_tunneling_perturbation", |b| {
        let mut base_engine = EdbcEngine::default();
        base_engine.state.moving_z_entropy = 2.0;
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x4000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        );
        let base_phases = make_heterogeneous_q7_phases_with_offset(22);
        let base_phases_w = liminika_core::traits::branchless::load_q7_words_64(&base_phases);
        let _ftz = ScopedFtzdazGuard::new();
        let mut phases = base_phases;
        let mut engine = base_engine;
        for _ in 0..8 {
            reset_engine_fast(&mut engine, &base_engine);
            liminika_core::traits::branchless::store_q7_words_64(&mut phases, &base_phases_w);
            let _ = engine
                .step_and_inject_tunneling_perturbation(
                    3.2f32,
                    2.8f32,
                    2.0f32,
                    1.0f32,
                    0.5f32,
                    128u32,
                    48usize,
                    ptr,
                    1.0f32,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    1.5f32,
                    &mut phases,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            liminika_core::traits::branchless::store_q7_words_64(&mut phases, &base_phases_w);
            let res = engine
                .step_and_inject_tunneling_perturbation(
                    black_box(3.2f32),
                    black_box(2.8f32),
                    black_box(2.0f32),
                    black_box(1.0f32),
                    black_box(0.5f32),
                    black_box(128u32),
                    black_box(48usize),
                    black_box(ptr),
                    black_box(1.0f32),
                    black_box(ATTRACTOR_SLOT_KNOWLEDGE),
                    black_box(1.5f32),
                    black_box(&mut phases),
                )
                .unwrap();
            black_box(res);
        });
    });

    // 3A. Blanket Coordinator: Single Unified Attractor Manifold Folding + EDBC Pointer Evolution
    group.throughput(Throughput::Elements(1));
    group.bench_function("step_with_unified_field_and_pointer_64h", |b| {
        let base_engine = EdbcEngine::default();
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x4400,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        );
        let unified_anchors = [
            make_heterogeneous_q7_phases_with_offset(-96),
            make_heterogeneous_q7_phases_with_offset(28),
            make_heterogeneous_q7_phases_with_offset(32),
            make_heterogeneous_q7_phases_with_offset(16),
        ];
        let unified_energies = [0.2f32, 2.5f32, 3.2f32, 1.0f32];
        let base_trail = gcso_pointer_trail_t {
            current_ptr: ptr.raw(),
            stigmergic_density: 1.2,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let _ftz = ScopedFtzdazGuard::new();
        let mut trail = base_trail;
        let mut engine = base_engine;
        for _ in 0..8 {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            let _ = engine
                .step_with_unified_field_and_pointer(
                    0.85f32,
                    0.85f32,
                    &unified_anchors,
                    &unified_energies,
                    1.5f32,
                    &mut trail,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            let res = engine
                .step_with_unified_field_and_pointer(
                    black_box(0.85f32),
                    black_box(0.85f32),
                    black_box(&unified_anchors),
                    black_box(&unified_energies),
                    black_box(1.5f32),
                    black_box(&mut trail),
                )
                .unwrap();
            black_box((res, trail.cluster_id));
        });
    });

    // 3B. Blanket Coordinator: 4-Hop Limina Trail-Chain Cohomology & Joint Probability Chain Rule Guard
    group.throughput(Throughput::Elements(4));
    group.bench_function("step_with_liminal_trail_chain_and_bloom_guard_4hop", |b| {
        let base_engine = EdbcEngine::default();
        let base_bloom = ContrastivePhaseBloomFilter::new();
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x4800,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            12,
        );
        let base_chain = [
            gcso_pointer_trail_t {
                current_ptr: ptr.raw(),
                stigmergic_density: 1.0,
                target_anchor_id: 21,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(20),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                current_ptr: ptr.raw(),
                stigmergic_density: 1.2,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(24),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                current_ptr: ptr.raw(),
                stigmergic_density: 1.4,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(28),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                current_ptr: ptr.raw(),
                stigmergic_density: 1.6,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(32),
                ..Default::default()
            },
        ];

        let mut engine = base_engine;
        let mut bloom = base_bloom;
        let mut chain = base_chain;
        for _ in 0..8 {
            reset_engine_fast(&mut engine, &base_engine);
            if !bloom.is_empty() {
                bloom = base_bloom;
            }
            reset_trail_fast(&mut chain[1], &base_chain[1]);
            reset_trail_fast(&mut chain[2], &base_chain[2]);
            reset_trail_fast(&mut chain[3], &base_chain[3]);
            let _ = engine
                .step_with_liminal_trail_chain_and_bloom_guard(
                    &mut chain,
                    64u8,
                    0.25f32,
                    0.20f32,
                    0.65f32,
                    0.80f32,
                    1.5f32,
                    ATTRACTOR_SLOT_PERSONA,
                    &mut bloom,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            if !bloom.is_empty() {
                bloom = base_bloom;
            }
            reset_trail_fast(&mut chain[1], &base_chain[1]);
            reset_trail_fast(&mut chain[2], &base_chain[2]);
            reset_trail_fast(&mut chain[3], &base_chain[3]);
            let res = engine
                .step_with_liminal_trail_chain_and_bloom_guard(
                    black_box(&mut chain),
                    black_box(64u8),
                    black_box(0.25f32),
                    black_box(0.20f32),
                    black_box(0.65f32),
                    black_box(0.80f32),
                    black_box(1.5f32),
                    black_box(ATTRACTOR_SLOT_PERSONA),
                    black_box(&mut bloom),
                )
                .unwrap();
            black_box((
                res.0.active_branch_mode,
                res.1.raw(),
                res.2,
                res.3,
                chain[3].cluster_id,
            ));
        });
    });

    // 3C. Blanket Coordinator: Pitchfork Metastable Tunneling + Bayesian Pointer-Trail Selection & Limina Rewiring
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        "step_with_bayesian_pointer_selection_and_tunneling_64h",
        |b| {
            let mut base_engine = EdbcEngine::default();
            base_engine.state.moving_z_entropy = 1.6;
            let base_bloom = ContrastivePhaseBloomFilter::new();
            let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
            let ptr = gcso_tagged_ptr_t::pack_action(
                0x4C00,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_EXPRESS_PHASE,
                16,
            );
            let base_src = gcso_pointer_trail_t {
                current_ptr: ptr.raw(),
                stigmergic_density: 1.5,
                attractor_pull_force: 2.0,
                linked_trail_id: 1,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(28),
                ..Default::default()
            };
            let base_candidates = [
                gcso_pointer_trail_t {
                    stigmergic_density: 4.0,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-96),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 3.0,
                    target_anchor_id: 27,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(30),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 2.2,
                    target_anchor_id: 14,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(24),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 1.5,
                    target_anchor_id: 9,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(18),
                    ..Default::default()
                },
            ];
            let _ftz = ScopedFtzdazGuard::new();
            let mut candidates = base_candidates;
            let mut src_trail = base_src;
            let mut engine = base_engine;
            let mut bloom = base_bloom;
            for _ in 0..8 {
                reset_engine_fast(&mut engine, &base_engine);
                if !bloom.is_empty() {
                    bloom = base_bloom;
                }
                reset_trail_fast(&mut src_trail, &base_src);
                reset_trail_fast(&mut candidates[1], &base_candidates[1]);
                attn_scores.data = [0.0f32; 8];
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let _ = engine
                    .step_with_bayesian_pointer_selection_and_tunneling(
                        &mut src_trail,
                        &mut candidates,
                        2.4f32,
                        2.5f32,
                        2.0f32,
                        1.0f32,
                        0.5f32,
                        128u32,
                        48usize,
                        2.0f32,
                        0.5f32,
                        0.25f32,
                        1.0f32,
                        64u8,
                        0.25f32,
                        1.5f32,
                        &mut bloom,
                        Some(&mut scores_mut),
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_engine_fast(&mut engine, &base_engine);
                if !bloom.is_empty() {
                    bloom = base_bloom;
                }
                reset_trail_fast(&mut src_trail, &base_src);
                reset_trail_fast(&mut candidates[1], &base_candidates[1]);
                attn_scores.data = [0.0f32; 8];
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let res = engine
                    .step_with_bayesian_pointer_selection_and_tunneling(
                        black_box(&mut src_trail),
                        black_box(&mut candidates),
                        black_box(2.4f32),
                        black_box(2.5f32),
                        black_box(2.0f32),
                        black_box(1.0f32),
                        black_box(0.5f32),
                        black_box(128u32),
                        black_box(48usize),
                        black_box(2.0f32),
                        black_box(0.5f32),
                        black_box(0.25f32),
                        black_box(1.0f32),
                        black_box(64u8),
                        black_box(0.25f32),
                        black_box(1.5f32),
                        black_box(&mut bloom),
                        black_box(Some(&mut scores_mut)),
                    )
                    .unwrap();
                black_box((
                    res.0.active_branch_mode,
                    res.1.raw(),
                    res.2,
                    res.3,
                    scores_mut.as_slice()[1],
                    src_trail.cluster_id,
                ));
            });
        },
    );

    // 3D. Blanket Coordinator: Zero-Copy Candidate-Trail Bayes-Attention Score Modulation with EDBC Temperature
    group.throughput(Throughput::Elements(8));
    group.bench_function(
        "modulate_candidate_trails_with_edbc_temperature_8cand",
        |b| {
            let engine = EdbcEngine::default();
            let mut attn_scores = AlignedBuffer32::new([0.5f32; 8]);
            let src_trail = gcso_pointer_trail_t {
                stigmergic_density: 1.8,
                linked_trail_id: 2,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(28),
                ..Default::default()
            };
            let candidates: [gcso_pointer_trail_t; 8] =
                core::array::from_fn(|i| gcso_pointer_trail_t {
                    stigmergic_density: 1.0 + (i as f32) * 0.4,
                    target_anchor_id: (i as u32) + 1,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(if i == 0 {
                        -96i8
                    } else {
                        (20 + i * 2) as i8
                    }),
                    ..Default::default()
                });

            let _ftz = ScopedFtzdazGuard::new();
            for _ in 0..8 {
                attn_scores.data = [0.5f32; 8];
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let _ = engine
                    .modulate_candidate_trails_with_edbc_temperature(
                        &mut scores_mut,
                        &src_trail,
                        &candidates,
                        128u32,
                        2.0f32,
                        0.5f32,
                        0.25f32,
                        0.25f32,
                        1.0f32,
                        1.0f32,
                        0.85f32,
                        0.90f32,
                        1.5f32,
                        1.2f32,
                        0.8f32,
                        0.5f32,
                    )
                    .unwrap();
            }

            b.iter(|| {
                attn_scores.data = [0.5f32; 8];
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let res = engine
                    .modulate_candidate_trails_with_edbc_temperature(
                        black_box(&mut scores_mut),
                        black_box(&src_trail),
                        black_box(&candidates),
                        black_box(128u32),
                        black_box(2.0f32),
                        black_box(0.5f32),
                        black_box(0.25f32),
                        black_box(0.25f32),
                        black_box(1.0f32),
                        black_box(1.0f32),
                        black_box(0.85f32),
                        black_box(0.90f32),
                        black_box(1.5f32),
                        black_box(1.2f32),
                        black_box(0.8f32),
                        black_box(0.5f32),
                    )
                    .unwrap();
                black_box((res, scores_mut.as_slice()[2]));
            });
        },
    );

    // 3E. Online Toroidal k-Means / DP-Means E+M Step + EDBC Gateway Coordinator (4 Centroids)
    group.throughput(Throughput::Elements(4));
    group.bench_function(
        "toroidal_kmeans_dp_means_and_edbc_gateway_step_4centroids",
        |b| {
            let base_engine = EdbcEngine::default();
            let fact = ObjectiveFact::new(
                make_heterogeneous_q7_phases_with_offset(28),
                2.8,
                0x0012_3400,
                108,
            );
            let persona = SubjectivePersona::new(
                make_heterogeneous_q7_phases_with_offset(30),
                2.2,
                0x0020_0000,
                19,
            );
            let pre_frame = (&fact, &persona).synthesize_gateway_frame();
            let base_centroids: [CrystallizedAttractorRecord; 4] = core::array::from_fn(|i| {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    (i as u32) + 100,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0020_0000 + ((i as u64) << 12),
                );
                let _ = rec.absorb_sample_point_64(
                    &make_heterogeneous_q7_phases_with_offset(16 + (i as i8) * 6),
                    2.0 + (i as f32) * 0.5,
                );
                rec
            });
            let base_trail = gcso_pointer_trail_t {
                stigmergic_density: 1.2,
                attractor_pull_force: 1.5,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
                ..Default::default()
            };
            let mut centroids = base_centroids;
            let mut trail = base_trail;
            let mut engine = base_engine;
            for _ in 0..8 {
                reset_engine_fast(&mut engine, &base_engine);
                reset_trail_fast(&mut trail, &base_trail);
                reset_crystal_fast(&mut centroids[2], &base_centroids[2]);
                let _ = engine
                    .step_gateway_kmeans_with_defaults(
                        0.38f32,
                        0.82f32,
                        &pre_frame,
                        &mut centroids,
                        &mut trail,
                        None,
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_engine_fast(&mut engine, &base_engine);
                reset_trail_fast(&mut trail, &base_trail);
                reset_crystal_fast(&mut centroids[2], &base_centroids[2]);
                let res = engine
                    .step_gateway_kmeans_with_defaults(
                        black_box(0.38f32),
                        black_box(0.82f32),
                        black_box(&pre_frame),
                        black_box(&mut centroids),
                        black_box(&mut trail),
                        None,
                    )
                    .unwrap();
                black_box((
                    res.0.active_branch_mode,
                    res.1.raw(),
                    res.2.provenance_bits,
                    res.3,
                    res.4,
                    trail.cluster_id,
                ));
            });
        },
    );

    // 3F. Coarse-to-Fine Macro Centroid Route (O(K=4) -> O(M_local=2) out of 8 Candidates) + Pitchfork Metastable Tunneling + Bayesian Trail Rewire
    group.throughput(Throughput::Elements(8));
    group.bench_function(
        "coarse_to_fine_route_and_edbc_bayesian_tunneling_64h",
        |b| {
            let mut base_engine = EdbcEngine::default();
            base_engine.state.moving_z_entropy = 1.6;
            let base_bloom = ContrastivePhaseBloomFilter::new();
            let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
            let centroids: [CrystallizedAttractorRecord; 4] = core::array::from_fn(|i| {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    (i as u32) + 200,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0030_0000 + ((i as u64) << 12),
                );
                let _ = rec.absorb_sample_point_64(
                    &make_heterogeneous_q7_phases_with_offset(12 + (i as i8) * 8),
                    2.5 + (i as f32) * 0.5,
                );
                rec
            });
            let base_src = gcso_pointer_trail_t {
                current_ptr: gcso_tagged_ptr_t::pack_action(
                    0x4C00,
                    ATTRACTOR_SLOT_PERSONA,
                    ACTION_PRESET_EXPRESS_PHASE,
                    16,
                )
                .raw(),
                stigmergic_density: 1.5,
                attractor_pull_force: 2.0,
                linked_trail_id: 1,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(28),
                ..Default::default()
            };
            // 8 Candidates (2 per cluster across clusters 200..203) so the Fine step genuinely evaluates intra-cluster candidates
            let base_candidates: [gcso_pointer_trail_t; 8] = [
                gcso_pointer_trail_t {
                    stigmergic_density: 1.5,
                    target_anchor_id: 200,
                    cluster_id: 200,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(12),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 1.2,
                    target_anchor_id: 200,
                    cluster_id: 200,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(10),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 2.2,
                    target_anchor_id: 201,
                    cluster_id: 201,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(20),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 1.9,
                    target_anchor_id: 201,
                    cluster_id: 201,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(18),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 2.6,
                    target_anchor_id: 202,
                    cluster_id: 202,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(24),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 3.2,
                    target_anchor_id: 202,
                    cluster_id: 202,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(28),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 2.4,
                    target_anchor_id: 203,
                    cluster_id: 203,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(36),
                    ..Default::default()
                },
                gcso_pointer_trail_t {
                    stigmergic_density: 2.0,
                    target_anchor_id: 203,
                    cluster_id: 203,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(40),
                    ..Default::default()
                },
            ];
            let mut candidates = base_candidates;

            let mut src_trail = base_src;
            let mut engine = base_engine;
            let mut bloom = base_bloom;
            for _ in 0..8 {
                reset_engine_fast(&mut engine, &base_engine);
                if !bloom.is_empty() {
                    bloom = base_bloom;
                }
                reset_trail_fast(&mut src_trail, &base_src);
                reset_trail_fast(&mut candidates[5], &base_candidates[5]);
                attn_scores.data = [0.0f32; 8];
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let _ = engine
                    .route_coarse_to_fine_and_step_bayesian_tunneling(
                        &centroids,
                        64u8,
                        &mut src_trail,
                        &mut candidates,
                        2.1f32,
                        2.5f32,
                        2.0f32,
                        1.0f32,
                        0.5f32,
                        128u32,
                        48usize,
                        2.0f32,
                        0.5f32,
                        0.25f32,
                        1.0f32,
                        64u8,
                        0.25f32,
                        1.5f32,
                        &mut bloom,
                        Some(&mut scores_mut),
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_engine_fast(&mut engine, &base_engine);
                if !bloom.is_empty() {
                    bloom = base_bloom;
                }
                reset_trail_fast(&mut src_trail, &base_src);
                reset_trail_fast(&mut candidates[5], &base_candidates[5]);
                attn_scores.data = [0.0f32; 8];
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let res = engine
                    .route_coarse_to_fine_and_step_bayesian_tunneling(
                        black_box(&centroids),
                        black_box(64u8),
                        black_box(&mut src_trail),
                        black_box(&mut candidates),
                        black_box(2.1f32),
                        black_box(2.5f32),
                        black_box(2.0f32),
                        black_box(1.0f32),
                        black_box(0.5f32),
                        black_box(128u32),
                        black_box(48usize),
                        black_box(2.0f32),
                        black_box(0.5f32),
                        black_box(0.25f32),
                        black_box(1.0f32),
                        black_box(64u8),
                        black_box(0.25f32),
                        black_box(1.5f32),
                        black_box(&mut bloom),
                        black_box(Some(&mut scores_mut)),
                    )
                    .unwrap();
                black_box((res, scores_mut.as_slice()[5]));
            });
        },
    );

    // 3G. Top-2 Liminal Macro Centroid Bifurcation Route (8 Candidates across 4 Centroids) + Pitchfork Metastable Tunneling + Bayesian Trail Rewire
    group.throughput(Throughput::Elements(8));
    group.bench_function("top2_liminal_route_and_edbc_bayesian_tunneling_64h", |b| {
        let mut base_engine = EdbcEngine::default();
        base_engine.state.moving_z_entropy = 1.6;
        let base_bloom = ContrastivePhaseBloomFilter::new();
        let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
        let centroids: [CrystallizedAttractorRecord; 4] = [
            {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    200,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0030_0000,
                );
                let _ =
                    rec.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(-60), 2.0);
                rec
            },
            {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    201,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0031_0000,
                );
                let _ =
                    rec.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(26), 3.0);
                rec
            },
            {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    202,
                    ATTRACTOR_SLOT_PERSONA,
                    0x0032_0000,
                );
                let _ =
                    rec.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(30), 2.9);
                rec
            },
            {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    203,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0033_0000,
                );
                let _ =
                    rec.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(-35), 1.5);
                rec
            },
        ];
        let base_src = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_t::pack_action(
                0x4C00,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_EXPRESS_PHASE,
                16,
            )
            .raw(),
            stigmergic_density: 1.5,
            attractor_pull_force: 2.0,
            linked_trail_id: 1,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(28),
            ..Default::default()
        };
        let base_candidates: [gcso_pointer_trail_t; 8] = [
            gcso_pointer_trail_t {
                stigmergic_density: 1.5,
                target_anchor_id: 200,
                cluster_id: 200,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-60),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 1.2,
                target_anchor_id: 200,
                cluster_id: 200,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-56),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.5,
                target_anchor_id: 201,
                cluster_id: 201,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(23),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.8,
                target_anchor_id: 201,
                cluster_id: 201,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.7,
                target_anchor_id: 202,
                cluster_id: 202,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(25),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 3.2,
                target_anchor_id: 202,
                cluster_id: 202,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(28),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 1.4,
                target_anchor_id: 203,
                cluster_id: 203,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-35),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 1.1,
                target_anchor_id: 203,
                cluster_id: 203,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-30),
                ..Default::default()
            },
        ];
        let mut candidates = base_candidates;

        let mut src_trail = base_src;
        let mut engine = base_engine;
        let mut bloom = base_bloom;
        for _ in 0..8 {
            reset_engine_fast(&mut engine, &base_engine);
            if !bloom.is_empty() {
                bloom = base_bloom;
            }
            reset_trail_fast(&mut src_trail, &base_src);
            reset_trail_fast(&mut candidates[5], &base_candidates[5]);
            attn_scores.data = [0.0f32; 8];
            let mut scores_mut = attn_scores.as_aligned_slice_mut();
            let _ = engine
                .route_top2_liminal_and_step_bayesian_tunneling(
                    &centroids,
                    64u8,
                    0.20f32,
                    &mut src_trail,
                    &mut candidates,
                    2.1f32,
                    2.5f32,
                    2.0f32,
                    1.0f32,
                    0.5f32,
                    128u32,
                    48usize,
                    2.0f32,
                    0.5f32,
                    0.25f32,
                    1.0f32,
                    64u8,
                    0.25f32,
                    1.5f32,
                    &mut bloom,
                    Some(&mut scores_mut),
                )
                .unwrap();
        }

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            if !bloom.is_empty() {
                bloom = base_bloom;
            }
            reset_trail_fast(&mut src_trail, &base_src);
            reset_trail_fast(&mut candidates[5], &base_candidates[5]);
            attn_scores.data = [0.0f32; 8];
            let mut scores_mut = attn_scores.as_aligned_slice_mut();
            let res = engine
                .route_top2_liminal_and_step_bayesian_tunneling(
                    black_box(&centroids),
                    black_box(64u8),
                    black_box(0.20f32),
                    black_box(&mut src_trail),
                    black_box(&mut candidates),
                    black_box(2.1f32),
                    black_box(2.5f32),
                    black_box(2.0f32),
                    black_box(1.0f32),
                    black_box(0.5f32),
                    black_box(128u32),
                    black_box(48usize),
                    black_box(2.0f32),
                    black_box(0.5f32),
                    black_box(0.25f32),
                    black_box(1.0f32),
                    black_box(64u8),
                    black_box(0.25f32),
                    black_box(1.5f32),
                    black_box(&mut bloom),
                    black_box(Some(&mut scores_mut)),
                )
                .unwrap();
            black_box((res, scores_mut.as_slice()[5]));
        });
    });

    // 4. Closed-Loop Sheaf Cohomology + Dual Homology Pairing + Repulsive Bloom Filter Guard
    group.throughput(Throughput::Bytes(
        (64 * 3 * core::mem::size_of::<f32>()) as u64,
    ));
    group.bench_function("step_with_cohomology_and_bloom_guard_64", |b| {
        let base_engine = EdbcEngine::default();
        let base_bloom = ContrastivePhaseBloomFilter::new();
        let r_buf = AlignedBuffer32::new(make_heterogeneous_supernode_field_64(0.45f32, 0.22f32));
        let p_buf = AlignedBuffer32::new(make_heterogeneous_supernode_field_64(0.30f32, 0.18f32));
        let g_buf = AlignedBuffer32::new(make_heterogeneous_supernode_field_64(0.55f32, 0.25f32));
        let r_s = r_buf.as_aligned_slice();
        let p_s = p_buf.as_aligned_slice();
        let g_s = g_buf.as_aligned_slice();
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x5000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            12,
        );
        let base_phases = make_heterogeneous_q7_phases_with_offset(24);
        let base_phases_w = liminika_core::traits::branchless::load_q7_words_64(&base_phases);
        let _ftz = ScopedFtzdazGuard::new();
        let mut phases = base_phases;
        let mut bloom = base_bloom;
        let mut engine = base_engine;
        for _ in 0..8 {
            reset_engine_fast(&mut engine, &base_engine);
            if !bloom.is_empty() {
                bloom = base_bloom;
                liminika_core::traits::branchless::store_q7_words_64(&mut phases, &base_phases_w);
            }
            let _ = engine
                .step_with_cohomology_and_bloom_guard(
                    2.6f32,
                    0.42f32,
                    1.4f32,
                    1.0f32,
                    0.5f32,
                    &r_s,
                    &p_s,
                    &g_s,
                    0.25f32,
                    ptr,
                    1.1f32,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    1.5f32,
                    &mut phases,
                    &mut bloom,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            if !bloom.is_empty() {
                bloom = base_bloom;
                liminika_core::traits::branchless::store_q7_words_64(&mut phases, &base_phases_w);
            }
            let res = engine
                .step_with_cohomology_and_bloom_guard(
                    black_box(2.6f32),
                    black_box(0.42f32),
                    black_box(1.4f32),
                    black_box(1.0f32),
                    black_box(0.5f32),
                    black_box(&r_s),
                    black_box(&p_s),
                    black_box(&g_s),
                    black_box(0.25f32),
                    black_box(ptr),
                    black_box(1.1f32),
                    black_box(ATTRACTOR_SLOT_KNOWLEDGE),
                    black_box(1.5f32),
                    black_box(&mut phases),
                    black_box(&mut bloom),
                )
                .unwrap();
            black_box((res, phases[0]));
        });
    });

    // 5. Matrix-Free Stigmergic Block Degrees & 3rd-Order Chebyshev Filter
    group.throughput(Throughput::Elements(16));
    group.bench_function(
        "matrix_free_stigmergic_block_degrees_and_chebyshev_deg3",
        |b| {
            let engine = EdbcEngine::default();
            let masks: [gcso_paged_bitmask_t; 16] =
                core::array::from_fn(|i| gcso_paged_bitmask_t {
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
                let lambda_max = engine
                    .eval_stigmergic_block_degrees(
                        black_box(&masks),
                        black_box(&mut deg_buf.as_aligned_slice_mut()),
                    )
                    .unwrap();
                let coeffs = engine
                    .compute_tikhonov_chebyshev_coeffs_deg3(
                        black_box(lambda_max.max(1.0)),
                        black_box(0.05f32),
                    )
                    .unwrap();
                let deg_slice = deg_buf.as_aligned_slice();
                let mut filt_slice = filt_buf.as_aligned_slice_mut();
                engine
                    .eval_chebyshev_spectral_filter_deg3(
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

    // 6. Hodge Obstruction Residual & Dual Cellular Sheaf Homology Pairing (64 Supernodes)
    group.throughput(Throughput::Bytes(
        (64 * 3 * core::mem::size_of::<f32>()) as u64,
    ));
    group.bench_function("hodge_obstruction_and_dual_homology_pairing_64", |b| {
        let engine = EdbcEngine::default();
        let r_buf = AlignedBuffer32::new(make_heterogeneous_supernode_field_64(1.25f32, 0.35f32));
        let p_buf = AlignedBuffer32::new(make_heterogeneous_supernode_field_64(0.45f32, 0.20f32));
        let g_buf = AlignedBuffer32::new(make_heterogeneous_supernode_field_64(0.80f32, 0.28f32));
        let r_s = r_buf.as_aligned_slice();
        let p_s = p_buf.as_aligned_slice();
        let g_s = g_buf.as_aligned_slice();

        b.iter(|| {
            let res = engine
                .evaluate_hodge_obstruction_and_dual_pairing(
                    black_box(&r_s),
                    black_box(&p_s),
                    black_box(&g_s),
                    black_box(0.25f32),
                )
                .unwrap();
            black_box(res);
        });
    });

    // 7A. 256-Bit Dyadic Pseudo-Ultrametric QDPT Resolution & Pointer Cascading
    group.throughput(Throughput::Elements(1));
    group.bench_function("resolve_dyadic_qdpt_and_cascade_tagged_ptr", |b| {
        let engine = EdbcEngine::default();
        let parent_ptr = gcso_tagged_ptr_t::pack_action(
            0x8000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            10,
        );
        let curr_mask = gcso_paged_bitmask_t {
            bits: [0xFFFF_FFFF_0000_0000, 0x1234_5678_9ABC_DEF0, 0, 0],
        };
        let anchor_mask = gcso_paged_bitmask_t {
            bits: [0xFFFF_FFFF_0000_0000, 0x1234_5600_0000_0000, 0, 0],
        };
        let offsets = [0x100u64, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800];
        let deltas = [4i8, 8, 12, 16, 20, 24, 28, 32];

        b.iter(|| {
            let res = engine
                .resolve_dyadic_qdpt_and_cascade_ptr(
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

    // 7B. Blanket Coordinator: 256-Bit Dyadic Tree QDPT Resolution + Coupled EDBC Step
    group.throughput(Throughput::Elements(1));
    group.bench_function("resolve_dyadic_tree_and_step_edbc_closed_loop", |b| {
        let base_engine = EdbcEngine::default();
        let parent_ptr = gcso_tagged_ptr_t::pack_action(
            0x8000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            10,
        );
        let curr_mask = gcso_paged_bitmask_t {
            bits: [0xFFFF_FFFF_0000_0000, 0x1234_5678_9ABC_DEF0, 0, 0],
        };
        let anchor_mask = gcso_paged_bitmask_t {
            bits: [0xFFFF_FFFF_0000_0000, 0x1234_5600_0000_0000, 0, 0],
        };
        let offsets = [0x100u64, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 0x800];
        let deltas = [4i8, 8, 12, 16, 20, 24, 28, 32];
        let base_trail = gcso_pointer_trail_t {
            current_ptr: parent_ptr.raw(),
            stigmergic_density: 1.2,
            target_anchor_id: 27,
            cluster_id: 27,
            ..Default::default()
        };

        let _ftz = ScopedFtzdazGuard::new();
        let mut trail = base_trail;
        let mut engine = base_engine;
        for _ in 0..8 {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            let _ = engine
                .resolve_dyadic_tree_and_step_edbc(
                    &curr_mask,
                    &anchor_mask,
                    0.15f32,
                    3u32,
                    0.10f32,
                    &offsets,
                    &deltas,
                    0.35f32,
                    0.80f32,
                    1.5f32,
                    &mut trail,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            let res = engine
                .resolve_dyadic_tree_and_step_edbc(
                    black_box(&curr_mask),
                    black_box(&anchor_mask),
                    black_box(0.15f32),
                    black_box(3u32),
                    black_box(0.10f32),
                    black_box(&offsets),
                    black_box(&deltas),
                    black_box(0.35f32),
                    black_box(0.80f32),
                    black_box(1.5f32),
                    black_box(&mut trail),
                )
                .unwrap();
            black_box((res, trail.cluster_id));
        });
    });

    group.finish();
}

/// Provisional `.gcso` / ZIMMS 4-Section Container Benchmarks (Isolated from Core EDBC Suite).
///
/// Evaluates pre-v1.0 `.gcso` record crystallization, 4-axis provenance records, and 4-section
/// zero-copy byte container verification. Skipped by default under `run_benchmarks.py --profile standard`
/// so Core `DPSR`/`EDBC` evaluations remain decoupled from the provisional container layout.
fn bench_edbc_provisional_gcso_container(c: &mut Criterion) {
    if !should_run_provisional_gcso() {
        return;
    }
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EDBC_Provisional_GCSO_Container");
    group.noise_threshold(configured_noise_threshold());
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1350));

    // 1. Provisional 2-Origin Composite Manifold Gateway + `.gcso` Crystallization Step
    group.throughput(Throughput::Elements(1));
    group.bench_function("step_gateway_signal_and_crystallize_attractor_64h", |b| {
        let base_engine = EdbcEngine::default();
        let fact = ObjectiveFact::new(
            make_heterogeneous_q7_phases_with_offset(28),
            2.5,
            0x0012_3400,
            33,
        );
        let persona = SubjectivePersona::new(
            make_heterogeneous_q7_phases_with_offset(30),
            2.0,
            0x0020_0000,
            19,
        );
        let gateway_tuple = (fact, persona);
        let mut logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
        let mut base_crystal =
            CrystallizedAttractorRecord::new_empty(33, ATTRACTOR_SLOT_KNOWLEDGE, 0x0012_3400);
        let _ =
            base_crystal.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(26), 2.0);
        let base_trail = gcso_pointer_trail_t {
            stigmergic_density: 1.0,
            attractor_pull_force: 1.5,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let mut trail = base_trail;
        let mut crystal = base_crystal;
        let mut engine = base_engine;
        for _ in 0..8 {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            reset_crystal_fast(&mut crystal, &base_crystal);
            logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
            let mut logits_mut = logits.as_aligned_slice_mut();
            let _ = engine
                .step_gateway_signal_with_defaults(
                    0.42f32,
                    0.85f32,
                    &gateway_tuple,
                    &mut trail,
                    Some(&mut logits_mut),
                    Some(&mut crystal),
                )
                .unwrap();
        }

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            reset_crystal_fast(&mut crystal, &base_crystal);
            logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
            let mut logits_mut = logits.as_aligned_slice_mut();
            let res = engine
                .step_gateway_signal_with_defaults(
                    black_box(0.42f32),
                    black_box(0.85f32),
                    black_box(&gateway_tuple),
                    black_box(&mut trail),
                    black_box(Some(&mut logits_mut)),
                    black_box(Some(&mut crystal)),
                )
                .unwrap();
            black_box((
                res.0.active_branch_mode,
                res.1.raw(),
                res.2.provenance_bits,
                res.3,
                res.4,
                logits_mut.as_slice()[0],
                crystal.phase_gravity_mass,
            ));
        });
    });

    // 2. Provisional Full 4-Axis (EdbcEngine Origin-1 x Fact x Persona x ContradictionGuard) Causal Provenance Step
    group.throughput(Throughput::Elements(1));
    group.bench_function(
        "step_gateway_4axis_contradiction_guard_and_provenance_64h",
        |b| {
            let base_engine = EdbcEngine::default();
            let sys_snapshot = base_engine;
            let fact = ObjectiveFact::new(
                make_heterogeneous_q7_phases_with_offset(28),
                2.8,
                0x0012_3400,
                33,
            );
            let persona = SubjectivePersona::new(
                make_heterogeneous_q7_phases_with_offset(30),
                2.2,
                0x0020_0000,
                19,
            );
            let mut bloom = ContrastivePhaseBloomFilter::new();
            bloom.insert_repulsive_pattern(&make_heterogeneous_q7_phases_with_offset(-96));
            let guard = ContradictionGuard::new(bloom, 0.15, 4);
            let gateway_4tuple = (&sys_snapshot, &fact, &persona, &guard);
            let mut logits =
                AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
            let mut base_crystal =
                CrystallizedAttractorRecord::new_empty(33, ATTRACTOR_SLOT_KNOWLEDGE, 0x0012_3400);
            let _ = base_crystal
                .absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(26), 2.0);
            let base_trail = gcso_pointer_trail_t {
                stigmergic_density: 1.1,
                attractor_pull_force: 1.5,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
                ..Default::default()
            };

            let mut trail = base_trail;
            let mut crystal = base_crystal;
            let mut engine = base_engine;
            for _ in 0..8 {
                reset_engine_fast(&mut engine, &base_engine);
                reset_trail_fast(&mut trail, &base_trail);
                reset_crystal_fast(&mut crystal, &base_crystal);
                logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
                let mut logits_mut = logits.as_aligned_slice_mut();
                let _ = engine
                    .step_gateway_signal_with_defaults(
                        0.38f32,
                        0.82f32,
                        &gateway_4tuple,
                        &mut trail,
                        Some(&mut logits_mut),
                        Some(&mut crystal),
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_engine_fast(&mut engine, &base_engine);
                reset_trail_fast(&mut trail, &base_trail);
                reset_crystal_fast(&mut crystal, &base_crystal);
                logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
                let mut logits_mut = logits.as_aligned_slice_mut();
                let res = engine
                    .step_gateway_signal_with_defaults(
                        black_box(0.38f32),
                        black_box(0.82f32),
                        black_box(&gateway_4tuple),
                        black_box(&mut trail),
                        black_box(Some(&mut logits_mut)),
                        black_box(Some(&mut crystal)),
                    )
                    .unwrap();
                black_box((
                    res.0.active_branch_mode,
                    res.1.raw(),
                    res.2.provenance_bits,
                    res.3,
                    res.4,
                    logits_mut.as_slice()[0],
                    crystal.phase_gravity_mass,
                ));
            });
        },
    );

    // 3. Provisional `.gcso` CrystallizedAttractorRecord Ensemble Gravity Folding + EDBC Step
    group.throughput(Throughput::Elements(4));
    group.bench_function("step_crystallized_ensemble_gravity_fold_4records", |b| {
        let base_engine = EdbcEngine::default();
        let records: [CrystallizedAttractorRecord; 4] = core::array::from_fn(|i| {
            let mut rec = CrystallizedAttractorRecord::new_empty(
                (i as u32) + 10,
                ATTRACTOR_SLOT_KNOWLEDGE,
                0x0010_0000 + ((i as u64) << 12),
            );
            let _ = rec.absorb_sample_point_64(
                &make_heterogeneous_q7_phases_with_offset(20 + (i as i8) * 4),
                2.0 + (i as f32) * 0.5,
            );
            rec
        });
        let base_trail = gcso_pointer_trail_t {
            stigmergic_density: 1.2,
            attractor_pull_force: 1.5,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let mut trail = base_trail;
        let mut engine = base_engine;
        for _ in 0..8 {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            let _ = engine
                .step_crystallized_ensemble_with_defaults(0.35f32, 0.80f32, &records, &mut trail)
                .unwrap();
        }

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            let res = engine
                .step_crystallized_ensemble_with_defaults(
                    black_box(0.35f32),
                    black_box(0.80f32),
                    black_box(&records),
                    black_box(&mut trail),
                )
                .unwrap();
            black_box((
                res.0.active_branch_mode,
                res.1.raw(),
                res.2,
                res.3,
                trail.target_anchor_id,
            ));
        });
    });

    // 4. Provisional `.gcso` (ZIMMS) Zero-Copy Deserialization + Crystallized Ensemble Gravity Folding EDBC Step
    group.throughput(Throughput::Elements(4));
    group.bench_function("zimms_zero_copy_crystallized_ensemble_step_4records", |b| {
        let base_engine = EdbcEngine::default();
        let records: [CrystallizedAttractorRecord; 4] = core::array::from_fn(|i| {
            let mut rec = CrystallizedAttractorRecord::new_empty(
                (i as u32) + 10,
                ATTRACTOR_SLOT_KNOWLEDGE,
                0x0010_0000 + ((i as u64) << 12),
            );
            let _ = rec.absorb_sample_point_64(
                &make_heterogeneous_q7_phases_with_offset(20 + (i as i8) * 4),
                2.0 + (i as f32) * 0.5,
            );
            rec
        });
        let mut zimms_buf = AlignedBuffer64::<u8, 512>::new([0u8; 512]);
        CrystallizedAttractorRecord::write_records_to_zimms_bytes(&records, &mut zimms_buf.data)
            .unwrap();
        let base_trail = gcso_pointer_trail_t {
            stigmergic_density: 1.2,
            attractor_pull_force: 1.5,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let mut trail = base_trail;
        let mut engine = base_engine;
        for _ in 0..8 {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            let zero_copy_recs = CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy(
                &zimms_buf.data,
                4usize,
            )
            .unwrap();
            let _ = engine
                .step_crystallized_ensemble_with_defaults(
                    0.35f32,
                    0.80f32,
                    zero_copy_recs,
                    &mut trail,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_engine_fast(&mut engine, &base_engine);
            reset_trail_fast(&mut trail, &base_trail);
            let zero_copy_recs = CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy(
                black_box(&zimms_buf.data),
                black_box(4usize),
            )
            .unwrap();
            let res = engine
                .step_crystallized_ensemble_with_defaults(
                    black_box(0.35f32),
                    black_box(0.80f32),
                    black_box(zero_copy_recs),
                    black_box(&mut trail),
                )
                .unwrap();
            black_box((
                res.0.active_branch_mode,
                res.1.raw(),
                res.2,
                res.3,
                trail.target_anchor_id,
            ));
        });
    });

    // 5. Provisional 4-Section `.gcso` Container Zero-Copy Write, CRC32 Verify & Crystallized Ensemble Step
    group.throughput(Throughput::Elements(4));
    group.bench_function(
        "four_section_gcso_zero_copy_write_verify_and_step_4records",
        |b| {
            let base_engine = EdbcEngine::default();
            let records: [CrystallizedAttractorRecord; 4] = core::array::from_fn(|i| {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    (i as u32) + 10,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0010_0000 + ((i as u64) << 12),
                );
                let _ = rec.absorb_sample_point_64(
                    &make_heterogeneous_q7_phases_with_offset(20 + (i as i8) * 4),
                    2.0 + (i as f32) * 0.5,
                );
                rec
            });
            let base_trail = gcso_pointer_trail_t {
                stigmergic_density: 1.2,
                attractor_pull_force: 1.5,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
                ..Default::default()
            };
            let base_trails = [base_trail];
            let mut trail = base_trail;
            let bloom = ContrastivePhaseBloomFilter::new();
            let daes = gcso_daes_slot_t::default();
            let mut container_buf = AlignedBuffer128::<u8, 1024>::new([0u8; 1024]);
            let mut engine = base_engine;
            for _ in 0..8 {
                reset_engine_fast(&mut engine, &base_engine);
                let written = GcsoFourSectionView::write_four_section_container_to_bytes(
                    &records,
                    &base_trails,
                    Some(&bloom),
                    Some(&daes),
                    1_774_900_000,
                    &mut container_buf.data,
                )
                .unwrap();
                let view =
                    GcsoFourSectionView::from_zimms_bytes_zero_copy(&container_buf.data[..written])
                        .unwrap();
                reset_trail_fast(&mut trail, &view.trails[0]);
                let _ = engine
                    .step_crystallized_ensemble_with_defaults(
                        0.35f32,
                        0.80f32,
                        view.crystals,
                        &mut trail,
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_engine_fast(&mut engine, &base_engine);
                let written = GcsoFourSectionView::write_four_section_container_to_bytes(
                    black_box(&records),
                    black_box(&base_trails),
                    black_box(Some(&bloom)),
                    black_box(Some(&daes)),
                    black_box(1_774_900_000),
                    black_box(&mut container_buf.data),
                )
                .unwrap();
                let view = GcsoFourSectionView::from_zimms_bytes_zero_copy(black_box(
                    &container_buf.data[..written],
                ))
                .unwrap();
                reset_trail_fast(&mut trail, &view.trails[0]);
                let res = engine
                    .step_crystallized_ensemble_with_defaults(
                        black_box(0.35f32),
                        black_box(0.80f32),
                        black_box(view.crystals),
                        black_box(&mut trail),
                    )
                    .unwrap();
                black_box((
                    res.0.active_branch_mode,
                    res.1.raw(),
                    res.2,
                    res.3,
                    trail.target_anchor_id,
                ));
            });
        },
    );

    group.finish();
}

fn bench_edbc_extreme_and_edge_cases(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("EDBC_Extreme_And_Edge_Cases");
    group.noise_threshold(configured_noise_threshold());

    // 1. Long-Run Token Stream Simulation (1,000,000 Tokens with Periodic Outliers)
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(600));
    group.measurement_time(Duration::from_millis(2200));

    let ultra_long_stream_len = 1000000usize;
    let mut shared_1m_buf = AlignedVectorBuffer::<f32, 64>::new(ultra_long_stream_len, 0.5f32);
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

        let mut engine = EdbcEngine::default();
        let mut state_out = gcso_edbc_state_t::default();
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
        let mut osc_buf = AlignedVectorBuffer::<f32, 64>::new(boundary_osc_len, tau);
        for (i, slot) in osc_buf.as_mut_slice().iter_mut().enumerate() {
            *slot = if (i & 1) == 0 {
                tau + delta + 0.05
            } else {
                tau - delta - 0.05
            };
        }
        let oscillating_z = osc_buf.as_slice();

        let mut engine = EdbcEngine::default();
        engine.set_bifurcation_threshold(tau);
        engine.set_hysteresis_band(delta);
        let mut state_out = gcso_edbc_state_t::default();
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
        let mut spike_buf = AlignedVectorBuffer::<f32, 64>::new(spike_stream_len, 0.45f32);
        for (i, slot) in spike_buf.as_mut_slice().iter_mut().enumerate() {
            *slot = match i & 63 {
                16 => 1.0e9f32,
                32 => f32::INFINITY,
                48 => f32::NAN,
                _ => 0.45f32,
            };
        }
        let z_scores = spike_buf.as_slice();

        let mut engine = EdbcEngine::default();
        let mut state_out = gcso_edbc_state_t::default();
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
    let shared_odd_arena = AlignedVectorBuffer::<f32, 64>::new_heterogeneous_f32(131073, 0.55f32);
    let odd_dims = [65537usize, 131073usize];
    for d in odd_dims {
        group.throughput(Throughput::Bytes((d * std::mem::size_of::<f32>()) as u64));
        group.bench_with_input(
            BenchmarkId::new("cvoid_dyadic_odd_remainder", format!("dim_{d}")),
            &d,
            |b, &dim_size| {
                let slice_32 = AlignedSlice32::new(shared_odd_arena.as_prefix(dim_size)).unwrap();
                let engine = EdbcEngine::default();
                let mut void_score = 0.0f32;
                engine
                    .eval_cvoid_dyadic128(&slice_32, &mut void_score)
                    .unwrap();

                b.iter(|| {
                    engine
                        .eval_cvoid_dyadic128(black_box(&slice_32), black_box(&mut void_score))
                        .unwrap();
                    black_box(void_score);
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

    // 6. Barrier Evaluation Near Epsilon Singularity (CVoid -> 0.0)
    group.throughput(Throughput::Elements(1));
    group.bench_function("eval_barrier_epsilon_singularity_boundary", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let engine = EdbcEngine::default();
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
    bench_edbc_stateful_eval,
    bench_edbc_cvoid_dyadic,
    bench_edbc_stream_simulation,
    bench_edbc_long_dialogue_reliability,
    bench_edbc_pointer_tunneling_and_cohomology,
    bench_edbc_extreme_and_edge_cases,
    bench_edbc_provisional_gcso_container
);
criterion_main!(benches);
