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
    gcso_daes_slot_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pspm_config_t, gcso_q7_t,
    gcso_tagged_ptr_t,
};
use liminika_core::phase_steering::{PhaseSteeringClosedLoop, PhaseSteeringEngine};
use liminika_core::traits::{
    AlignedBuffer128, AlignedBuffer16, AlignedBuffer32, AlignedBuffer64, AlignedSlice,
    AlignedSlice16, AlignedSlice32, AlignedSliceMut, AlignedSliceMut16,
    ContrastivePhaseBloomFilter, CrystallizedAttractorRecord, GcsoFourSectionView, ObjectiveFact,
    PhaseSteering, PspmRouter, StigmergicActionPtrOps, SubjectivePersona, SystemConstraint,
    VisualAttentionSteering, ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION,
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_SYSTEM_BASELINE,
    DEFAULT_RIPA_CLAMP_MAX_RAD, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
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

/// Generic heap buffer with strict power-of-two alignment and optional cache-set phase offset
/// (`phase_offset_elems`) to prevent L1D 4K Load-Store Queue aliasing across Q, K, and V tensors.
struct AlignedHeapBuffer<T, const ALIGN: usize> {
    base_ptr: *mut T,
    data_ptr: *mut T,
    layout: Layout,
    len: usize,
}

impl<T: Copy, const ALIGN: usize> AlignedHeapBuffer<T, ALIGN> {
    fn new(len: usize, initial_val: T) -> Self {
        Self::new_with_phase_offset(len, initial_val, 0)
    }

    fn new_with_phase_offset(len: usize, initial_val: T, phase_offset_elems: usize) -> Self {
        assert!(ALIGN > 0 && ALIGN.is_power_of_two());
        let elem_size = std::mem::size_of::<T>();
        assert!(len > 0 && elem_size > 0);
        assert!((phase_offset_elems * elem_size) % ALIGN == 0);
        let alloc_elems = len
            .checked_add(phase_offset_elems)
            .expect("Element count overflow");
        let byte_len = alloc_elems
            .checked_mul(elem_size)
            .expect("Byte length overflow");
        let alloc_align = if ALIGN < 4096 { 4096 } else { ALIGN };
        let layout = Layout::from_size_align(byte_len, alloc_align).expect("Invalid layout");
        // SAFETY: `layout` has non-zero size and power-of-two alignment.
        let base_ptr = unsafe { alloc(layout).cast::<T>() };
        assert!(!base_ptr.is_null(), "Heap allocation failed");
        // SAFETY: `base_ptr` is valid for `alloc_elems` elements.
        unsafe {
            std::slice::from_raw_parts_mut(base_ptr, alloc_elems).fill(initial_val);
        }
        // SAFETY: `phase_offset_elems <= alloc_elems` and `(phase_offset_elems * elem_size) % ALIGN == 0`.
        let data_ptr = unsafe { base_ptr.add(phase_offset_elems) };
        Self {
            base_ptr,
            data_ptr,
            layout,
            len,
        }
    }

    #[inline(always)]
    fn as_slice(&self) -> &[T] {
        // SAFETY: `self.data_ptr` is valid for `self.len` elements.
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len) }
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `self.data_ptr` is valid and uniquely borrowed for `self.len` elements.
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr, self.len) }
    }

    #[inline(always)]
    fn as_aligned_slice_mut(&mut self) -> AlignedSliceMut<'_, T, ALIGN> {
        // SAFETY: `self.data_ptr` satisfies `ALIGN`.
        unsafe { AlignedSliceMut::new_unchecked(self.as_mut_slice()) }
    }

    #[inline(always)]
    fn as_aligned_prefix(&self, prefix_len: usize) -> AlignedSlice<'_, T, ALIGN> {
        assert!(prefix_len <= self.len);
        // SAFETY: `self.data_ptr` satisfies `ALIGN` and is valid for `prefix_len <= self.len` elements.
        unsafe {
            AlignedSlice::new_unchecked(std::slice::from_raw_parts(self.data_ptr, prefix_len))
        }
    }

    #[inline(always)]
    fn as_aligned_prefix_mut(&mut self, prefix_len: usize) -> AlignedSliceMut<'_, T, ALIGN> {
        assert!(prefix_len <= self.len);
        // SAFETY: `self.data_ptr` satisfies `ALIGN` and is valid for `prefix_len <= self.len` elements.
        unsafe {
            AlignedSliceMut::new_unchecked(std::slice::from_raw_parts_mut(
                self.data_ptr,
                prefix_len,
            ))
        }
    }
}

impl<T, const ALIGN: usize> Drop for AlignedHeapBuffer<T, ALIGN> {
    fn drop(&mut self) {
        // SAFETY: `self.base_ptr` was allocated with `self.layout`.
        unsafe {
            dealloc(self.base_ptr.cast::<u8>(), self.layout);
        }
    }
}

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

/// Realistic heterogeneous 64-head phase profile on the torus $SO(2)^{64}$ where every head
/// $h \in [0, 64)$ carries a distinct Q7 phase rotation (`arr[h] != arr[h + 1]`), preventing
/// synthetic uniform-word or uniform-head run-coalescing shortcuts during benchmarking.
#[inline(always)]
const fn make_heterogeneous_q7_phases() -> [gcso_q7_t; 64] {
    let mut arr = [0i8; 64];
    let domain_bases: [gcso_q7_t; 8] = [14, -22, 19, 35, 18, -28, 24, -16];
    let head_offsets: [gcso_q7_t; 8] = [-4, -2, -1, 1, 2, 4, -3, 3];
    let mut i = 0usize;
    while i < 64 {
        arr[i] = domain_bases[i >> 3].saturating_add(head_offsets[i & 7]);
        i += 1;
    }
    arr
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
    let mut shared_k_arena =
        AlignedHeapBuffer::<f32, 32>::new_with_phase_offset(max_elements, 1.0f32, 384);
    let mut shared_v_arena =
        AlignedHeapBuffer::<f32, 32>::new_with_phase_offset(max_elements, 1.0f32, 768);

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

        // 2. Fused QK (2x Bytes), Geodesic Fused QK, and Fused QKV (3x Bytes)
        group.throughput(Throughput::Bytes((single_tensor_bytes * 2) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_fused_qk",
                format!("{head_dim}x{num_heads}"),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let _ftz = ScopedFtzdazGuard::new();
                let phase_data = AlignedQ7Buffer(make_heterogeneous_q7_phases_with_offset(7));
                let p_slice = AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();
                let mut q_slice = shared_q_arena
                    .as_aligned_prefix_mut(total_elements)
                    .relax_alignment::<32>();
                let mut k_slice = shared_k_arena.as_aligned_prefix_mut(total_elements);
                q_slice.as_mut_slice().fill(1.0f32);
                k_slice.as_mut_slice().fill(1.0f32);
                for _ in 0..8 {
                    engine
                        .apply_phase_steering_fused(&mut q_slice, &mut k_slice, &p_slice)
                        .unwrap();
                }

                b.iter(|| {
                    engine
                        .apply_phase_steering_fused(
                            black_box(&mut q_slice),
                            black_box(&mut k_slice),
                            black_box(&p_slice),
                        )
                        .unwrap();
                    black_box((q_slice.as_slice()[0], k_slice.as_slice()[0]));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_geodesic_fused_qk",
                format!("{head_dim}x{num_heads}"),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let _ftz = ScopedFtzdazGuard::new();
                let phase_data = AlignedQ7Buffer(make_heterogeneous_q7_phases_with_offset(112));
                let p_slice = AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();
                let mut q_slice = shared_q_arena
                    .as_aligned_prefix_mut(total_elements)
                    .relax_alignment::<32>();
                let mut k_slice = shared_k_arena.as_aligned_prefix_mut(total_elements);
                q_slice.as_mut_slice().fill(1.0f32);
                k_slice.as_mut_slice().fill(1.0f32);
                for _ in 0..8 {
                    engine
                        .apply_phase_steering_geodesic_fused_safe(
                            &mut q_slice,
                            &mut k_slice,
                            &p_slice,
                            DEFAULT_RIPA_CLAMP_MAX_RAD,
                        )
                        .unwrap();
                }

                b.iter(|| {
                    engine
                        .apply_phase_steering_geodesic_fused_safe(
                            black_box(&mut q_slice),
                            black_box(&mut k_slice),
                            black_box(&p_slice),
                            black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                        )
                        .unwrap();
                    black_box((q_slice.as_slice()[0], k_slice.as_slice()[0]));
                });
            },
        );

        group.throughput(Throughput::Bytes((single_tensor_bytes * 3) as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "apply_phase_steering_fused_qkv",
                format!("{head_dim}x{num_heads}"),
            ),
            &(head_dim, num_heads),
            |b, _| {
                let _ftz = ScopedFtzdazGuard::new();
                let phase_data = AlignedQ7Buffer(make_heterogeneous_q7_phases_with_offset(7));
                let p_slice = AlignedSlice16::new(&phase_data.0[..(num_heads as usize)]).unwrap();
                let mut q_slice = shared_q_arena
                    .as_aligned_prefix_mut(total_elements)
                    .relax_alignment::<32>();
                let mut k_slice = shared_k_arena.as_aligned_prefix_mut(total_elements);
                let mut v_slice = shared_v_arena.as_aligned_prefix_mut(total_elements);
                q_slice.as_mut_slice().fill(1.0f32);
                k_slice.as_mut_slice().fill(1.0f32);
                v_slice.as_mut_slice().fill(1.0f32);
                for _ in 0..8 {
                    engine
                        .apply_phase_steering_fused_qkv(
                            &mut q_slice,
                            &mut k_slice,
                            &mut v_slice,
                            &p_slice,
                        )
                        .unwrap();
                }

                b.iter(|| {
                    engine
                        .apply_phase_steering_fused_qkv(
                            black_box(&mut q_slice),
                            black_box(&mut k_slice),
                            black_box(&mut v_slice),
                            black_box(&p_slice),
                        )
                        .unwrap();
                    black_box((
                        q_slice.as_slice()[0],
                        k_slice.as_slice()[0],
                        v_slice.as_slice()[0],
                    ));
                });
            },
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
                            DEFAULT_RIPA_CLAMP_MAX_RAD,
                        )
                        .unwrap();
                }

                b.iter(|| {
                    engine
                        .apply_phase_steering_safe(
                            black_box(&mut q_slice),
                            black_box(&p_slice),
                            black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
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

fn bench_phase_steering_batch_sequence_scaling(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("PhaseSteering_Batch_Sequence_Scaling");
    group.noise_threshold(configured_noise_threshold());
    group.warm_up_time(Duration::from_millis(500));

    let head_dim = 128u32;
    let num_heads = 32u32;
    let stride = (head_dim * num_heads) as usize;
    let engine = PhaseSteeringEngine::new(head_dim, num_heads)
        .expect("Failed to initialize PhaseSteeringEngine");

    let batch_sizes = [1usize, 4, 16, 64, 256, 1024];
    let coherent_hetero_phases = make_heterogeneous_q7_phases_with_offset(8);
    let multi_domain_hetero_phases = make_heterogeneous_q7_phases();
    let mut l1_q_batch = AlignedHeapBuffer::<f32, 64>::new(stride * 4, 1.0f32);
    let mut small_q_batch = AlignedHeapBuffer::<f32, 64>::new(stride * 64, 1.0f32);
    let mut shared_q_batch: Option<AlignedHeapBuffer<f32, 64>> = None;

    for &num_tokens in &batch_sizes {
        if num_tokens >= 256 && shared_q_batch.is_none() {
            let mut arena = AlignedHeapBuffer::<f32, 64>::new(stride * 1024, 1.0f32);
            let warm_phase = AlignedQ7Buffer(coherent_hetero_phases);
            let warm_p = AlignedSlice16::new(&warm_phase.0[..32]).unwrap();
            let mut full_q = arena
                .as_aligned_prefix_mut(stride * 1024)
                .relax_alignment::<32>();
            for _ in 0..4 {
                engine
                    .apply_phase_steering_batch(&mut full_q, &warm_p, 1024)
                    .unwrap();
            }
            shared_q_batch = Some(arena);
        }
        if num_tokens >= 256 {
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(60);
            group.warm_up_time(Duration::from_millis(450));
            group.measurement_time(Duration::from_millis(1150));
        } else {
            group.sampling_mode(SamplingMode::Flat);
            group.sample_size(100);
            group.warm_up_time(Duration::from_millis(450));
            group.measurement_time(Duration::from_millis(1250));
        }
        let total_elements = stride * num_tokens;
        let total_bytes = total_elements * core::mem::size_of::<f32>();
        group.throughput(Throughput::Bytes(total_bytes as u64));

        // Uses realistic per-head heterogeneous phases (`coherent_hetero_phases`) rather than uniform `[8i8; 64]`
        group.bench_with_input(
            BenchmarkId::new("batch_steering_query_only", format!("tokens_{num_tokens}")),
            &num_tokens,
            |b, &tokens| {
                let _ftz = ScopedFtzdazGuard::new();
                let arena = if tokens <= 4 {
                    &mut l1_q_batch
                } else if tokens <= 64 {
                    &mut small_q_batch
                } else {
                    shared_q_batch.as_mut().unwrap()
                };
                let mut q_slice = arena
                    .as_aligned_prefix_mut(total_elements)
                    .relax_alignment::<32>();
                if tokens <= 64 {
                    q_slice.as_mut_slice().fill(1.0f32);
                }
                let phase_data = AlignedQ7Buffer(coherent_hetero_phases);
                let p_slice = AlignedSlice16::new(&phase_data.0[..32]).unwrap();
                for _ in 0..8 {
                    engine
                        .apply_phase_steering_batch(&mut q_slice, &p_slice, tokens)
                        .unwrap();
                }

                b.iter(|| {
                    engine
                        .apply_phase_steering_batch(
                            black_box(&mut q_slice),
                            black_box(&p_slice),
                            black_box(tokens),
                        )
                        .unwrap();
                    black_box(q_slice.as_slice()[0]);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "batch_steering_heterogeneous_heads",
                format!("tokens_{num_tokens}"),
            ),
            &num_tokens,
            |b, &tokens| {
                let _ftz = ScopedFtzdazGuard::new();
                let arena = if tokens <= 4 {
                    &mut l1_q_batch
                } else if tokens <= 64 {
                    &mut small_q_batch
                } else {
                    shared_q_batch.as_mut().unwrap()
                };
                let mut q_slice = arena
                    .as_aligned_prefix_mut(total_elements)
                    .relax_alignment::<32>();
                if tokens <= 64 {
                    q_slice.as_mut_slice().fill(1.0f32);
                }
                let phase_data = AlignedQ7Buffer(multi_domain_hetero_phases);
                let p_slice = AlignedSlice16::new(&phase_data.0[..32]).unwrap();
                for _ in 0..8 {
                    engine
                        .apply_phase_steering_batch(&mut q_slice, &p_slice, tokens)
                        .unwrap();
                }

                b.iter(|| {
                    engine
                        .apply_phase_steering_batch(
                            black_box(&mut q_slice),
                            black_box(&p_slice),
                            black_box(tokens),
                        )
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

    // 2. Seam-Free S^1 Geodesic-Folded Lazy Phase Unwrapping in Q7 + QDPS Step Filter
    group.throughput(Throughput::Bytes(single_tensor_bytes as u64));
    group.bench_function("seam_free_circular_lazy_unwrap_q7_500k", |b| {
        let q_phases = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(-100));
        let ctx_phases = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(-112));
        let qp_slice = q_phases.as_aligned_slice();
        let cp_slice = ctx_phases.as_aligned_slice();
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        let _ftz = ScopedFtzdazGuard::new();
        q_slice.as_mut_slice().fill(1.0f32);
        for _ in 0..8 {
            let _ = engine
                .apply_seam_free_lazy_unwrap_q7(
                    &mut q_slice,
                    None,
                    &qp_slice,
                    &cp_slice,
                    0.01f32,
                    Some(DEFAULT_RIPA_CLAMP_MAX_RAD),
                )
                .unwrap();
        }

        b.iter(|| {
            let eff = engine
                .apply_seam_free_lazy_unwrap_q7(
                    black_box(&mut q_slice),
                    None,
                    black_box(&qp_slice),
                    black_box(&cp_slice),
                    black_box(0.01f32),
                    black_box(Some(DEFAULT_RIPA_CLAMP_MAX_RAD)),
                )
                .unwrap();
            black_box((eff, q_slice.as_slice()[0]));
        });
    });

    // 3. 16-Head Unrolled SWAR Circular Context Phase Accumulation + Pre-QDPS Filtering
    group.throughput(Throughput::Elements(u64::from(num_heads)));
    group.bench_function("circular_context_accum_swar_q7", |b| {
        let mut ctx_accum = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(120));
        let step_deltas = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(24));
        let del_slice = step_deltas.as_aligned_slice();
        let mut ctx_mut = ctx_accum.as_aligned_slice_mut();
        for _ in 0..8 {
            engine
                .accumulate_context_phases_circular_q7(&mut ctx_mut, &del_slice, 0.01f32)
                .unwrap();
        }

        b.iter(|| {
            engine
                .accumulate_context_phases_circular_q7(
                    black_box(&mut ctx_mut),
                    black_box(&del_slice),
                    black_box(0.01f32),
                )
                .unwrap();
            black_box(ctx_mut.as_slice()[0]);
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
    group.bench_function("steer_trail_with_procrustes_feedback_closed_loop", |b| {
        init_non_collinear_procrustes_tensors(
            shared_q_arena.as_mut_slice(),
            shared_tgt_arena.as_mut_slice(),
            head_dim as usize,
            num_heads as usize,
        );
        let t_slice = shared_tgt_arena
            .as_aligned_prefix(total_elements)
            .relax_alignment::<32>();
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        let base_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(20),
            ..Default::default()
        };
        let _ftz = ScopedFtzdazGuard::new();
        let mut trail = base_trail;
        for _ in 0..8 {
            reset_trail_fast(&mut trail, &base_trail);
            let _ = engine
                .steer_trail_with_procrustes_feedback(
                    &mut trail,
                    &mut q_slice,
                    &t_slice,
                    96u8,
                    0.01f32,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut trail, &base_trail);
            let deltas = engine
                .steer_trail_with_procrustes_feedback(
                    black_box(&mut trail),
                    black_box(&mut q_slice),
                    black_box(&t_slice),
                    black_box(96u8),
                    black_box(0.01f32),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                )
                .unwrap();
            black_box((deltas.data[0], q_slice.as_slice()[0]));
        });
    });

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
            engine.qdps_filter_step(&mut w_slice, 0.12f32).unwrap();
        }

        b.iter_batched(
            || seed_buf,
            |mut local| {
                let mut p_slice = local.as_aligned_slice_mut();
                engine
                    .qdps_filter_step(black_box(&mut p_slice), black_box(0.12f32))
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
    let engine = PhaseSteeringEngine::new(head_dim, num_heads)
        .expect("Failed 64-head PhaseSteeringEngine init");
    let mut shared_q_arena =
        AlignedHeapBuffer::<f32, 512>::new_with_phase_offset(total_elements, 1.0f32, 0);

    group.throughput(Throughput::Bytes(tensor_bytes as u64));

    // 1A. 8-Cell SWAR Micro-Swarm Partial-Quorum RIPA Phase Steering (7/8 Cells Resolved)
    group.bench_function("swarm_quorum_gated_steering_safe_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let deltas_64 = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(24));
        let d_slice = deltas_64.as_aligned_slice();
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            12,
        );
        for _ in 0..8 {
            engine
                .apply_pointer_swarm_quorum_gated_steering_safe(
                    ptr,
                    0b1111_0111u8,
                    32u8,
                    &d_slice,
                    &mut q_slice,
                    None,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                )
                .unwrap();
        }

        b.iter(|| {
            engine
                .apply_pointer_swarm_quorum_gated_steering_safe(
                    black_box(ptr),
                    black_box(0b1111_0111u8),
                    black_box(32u8),
                    black_box(&d_slice),
                    black_box(&mut q_slice),
                    None,
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                )
                .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    // 1B. 8-Cell SWAR Micro-Swarm Full-Quorum (0xFF) Zero-Copy Fast-Path Geodesic RIPA Steering
    group.bench_function("swarm_quorum_full_consensus_fastpath_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let deltas_64 = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(24));
        let d_slice = deltas_64.as_aligned_slice();
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            12,
        );
        for _ in 0..8 {
            engine
                .apply_pointer_swarm_quorum_gated_steering_safe(
                    ptr,
                    0xFFu8,
                    32u8,
                    &d_slice,
                    &mut q_slice,
                    None,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                )
                .unwrap();
        }

        b.iter(|| {
            engine
                .apply_pointer_swarm_quorum_gated_steering_safe(
                    black_box(ptr),
                    black_box(0xFFu8),
                    black_box(32u8),
                    black_box(&d_slice),
                    black_box(&mut q_slice),
                    None,
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                )
                .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    // 2A. Closed-Loop 8-Cell SWAR Micro-Swarm Consensus Resolution + Register QDPS/Geodesic Steering
    group.bench_function("closed_loop_swarm_resolve_and_steer_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let tgt_64 = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(32));
        let tgt_slice = tgt_64.as_aligned_slice();
        let bloom = ContrastivePhaseBloomFilter::new();
        let initial_trail = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_t::pack_action(
                0x2000,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_FULL_ORCHESTRATION,
                0,
            )
            .raw(),
            stigmergic_density: 0.5,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let _ftz = ScopedFtzdazGuard::new();
        let mut trail = initial_trail;
        for _ in 0..8 {
            reset_trail_fast(&mut trail, &initial_trail);
            let _ = engine
                .resolve_and_apply_swarm_phase_steering_safe(
                    &mut trail,
                    &tgt_slice,
                    96u8,
                    6u8,
                    6u8,
                    32u8,
                    0.18f32,
                    0.12f32,
                    Some(&bloom),
                    3u8,
                    64u8,
                    0.01f32,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    &mut q_slice,
                    None,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut trail, &initial_trail);
            let report = engine
                .resolve_and_apply_swarm_phase_steering_safe(
                    black_box(&mut trail),
                    black_box(&tgt_slice),
                    black_box(96u8),
                    black_box(6u8),
                    black_box(6u8),
                    black_box(32u8),
                    black_box(0.18f32),
                    black_box(0.12f32),
                    black_box(Some(&bloom)),
                    black_box(3u8),
                    black_box(64u8),
                    black_box(0.01f32),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    black_box(&mut q_slice),
                    None,
                )
                .unwrap();
            black_box((report, q_slice.as_slice()[0]));
        });
    });

    // 2B-1. Single Unified Attractor Manifold 1-Anchor Fast-Path Swarm Resolution + RIPA Steering
    group.bench_function("unified_field_swarm_single_anchor_fastpath_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let single_anchor = [make_heterogeneous_q7_phases_with_offset(30)];
        let single_energy = [3.0f32];
        let bloom = ContrastivePhaseBloomFilter::new();
        let initial_trail = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_t::pack_action(
                0x2600,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_FULL_ORCHESTRATION,
                0,
            )
            .raw(),
            stigmergic_density: 0.75,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let _ftz = ScopedFtzdazGuard::new();
        let mut trail = initial_trail;
        for _ in 0..8 {
            reset_trail_fast(&mut trail, &initial_trail);
            let _ = engine
                .resolve_unified_field_swarm_and_steer_safe(
                    &mut trail,
                    &single_anchor,
                    &single_energy,
                    None::<&gcso_paged_bitmask_t>,
                    96u8,
                    6u8,
                    6u8,
                    32u8,
                    0.18f32,
                    0.12f32,
                    Some(&bloom),
                    3u8,
                    64u8,
                    0.01f32,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    &mut q_slice,
                    None,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut trail, &initial_trail);
            let report = engine
                .resolve_unified_field_swarm_and_steer_safe(
                    black_box(&mut trail),
                    black_box(&single_anchor),
                    black_box(&single_energy),
                    black_box(None::<&gcso_paged_bitmask_t>),
                    black_box(96u8),
                    black_box(6u8),
                    black_box(6u8),
                    black_box(32u8),
                    black_box(0.18f32),
                    black_box(0.12f32),
                    black_box(Some(&bloom)),
                    black_box(3u8),
                    black_box(64u8),
                    black_box(0.01f32),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    black_box(&mut q_slice),
                    None,
                )
                .unwrap();
            black_box((report, trail.target_anchor_id, q_slice.as_slice()[0]));
        });
    });

    // 2B-2. Single Unified Attractor Manifold 4-Anchor Dynamic Folding + Swarm Resolution + RIPA Steering
    group.bench_function("unified_field_swarm_resolve_and_steer_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let unified_anchors = [
            make_heterogeneous_q7_phases_with_offset(-64),
            make_heterogeneous_q7_phases_with_offset(28),
            make_heterogeneous_q7_phases_with_offset(32),
            make_heterogeneous_q7_phases_with_offset(16),
        ];
        let unified_energies = [0.25f32, 2.5f32, 3.2f32, 1.0f32];
        let bloom = ContrastivePhaseBloomFilter::new();
        let initial_trail = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_t::pack_action(
                0x2800,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_FULL_ORCHESTRATION,
                0,
            )
            .raw(),
            stigmergic_density: 0.75,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let _ftz = ScopedFtzdazGuard::new();
        let mut trail = initial_trail;
        for _ in 0..8 {
            reset_trail_fast(&mut trail, &initial_trail);
            let _ = engine
                .resolve_unified_field_swarm_and_steer_safe(
                    &mut trail,
                    &unified_anchors,
                    &unified_energies,
                    None::<&gcso_paged_bitmask_t>,
                    96u8,
                    6u8,
                    6u8,
                    32u8,
                    0.18f32,
                    0.12f32,
                    Some(&bloom),
                    3u8,
                    64u8,
                    0.01f32,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    &mut q_slice,
                    None,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut trail, &initial_trail);
            let report = engine
                .resolve_unified_field_swarm_and_steer_safe(
                    black_box(&mut trail),
                    black_box(&unified_anchors),
                    black_box(&unified_energies),
                    black_box(None::<&gcso_paged_bitmask_t>),
                    black_box(96u8),
                    black_box(6u8),
                    black_box(6u8),
                    black_box(32u8),
                    black_box(0.18f32),
                    black_box(0.12f32),
                    black_box(Some(&bloom)),
                    black_box(3u8),
                    black_box(64u8),
                    black_box(0.01f32),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    black_box(&mut q_slice),
                    None,
                )
                .unwrap();
            black_box((report, trail.target_anchor_id, q_slice.as_slice()[0]));
        });
    });

    // 2C. 8-Slot Attractor Bank Swarm Resolution + Coupled Neighbor Recruitment + RIPA Steering
    group.bench_function("slot_bank_swarm_lateral_recruit_and_steer_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let mut slot_bank =
            AlignedBuffer64::<gcso_q7_t, 512>::new(make_heterogeneous_slot_bank_512(24));
        let persona_base = usize::from(ATTRACTOR_SLOT_PERSONA) * 64;
        let persona_aligned = make_heterogeneous_q7_phases_with_offset(28);
        let persona_divergent = make_heterogeneous_q7_phases_with_offset(-96);
        slot_bank.data[persona_base..persona_base + 56].copy_from_slice(&persona_aligned[..56]);
        slot_bank.data[persona_base + 56..persona_base + 64]
            .copy_from_slice(&persona_divergent[56..64]);
        let knowledge_base = usize::from(ATTRACTOR_SLOT_KNOWLEDGE) * 64;
        slot_bank.data[knowledge_base..knowledge_base + 64]
            .copy_from_slice(&make_heterogeneous_q7_phases_with_offset(30));
        let bank_slice = slot_bank.relax_slice::<16>();
        let slot_weights = [1.0f32; MAX_ATTRACTOR_DOMAINS];
        let bloom = ContrastivePhaseBloomFilter::new();

        let initial_trail = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_t::pack_action(
                0x3000,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_FULL_ORCHESTRATION,
                0,
            )
            .raw(),
            stigmergic_density: 0.5,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let _ftz = ScopedFtzdazGuard::new();
        let mut trail = initial_trail;
        for _ in 0..8 {
            reset_trail_fast(&mut trail, &initial_trail);
            let _ = engine
                .resolve_slot_bank_swarm_and_steer_safe(
                    &mut trail,
                    &bank_slice,
                    &slot_weights,
                    DEFAULT_SLOT_COUPLING_MATRIX_8X8,
                    None::<&gcso_paged_bitmask_t>,
                    96u8,
                    8u8,
                    7u8,
                    32u8,
                    0.18f32,
                    0.12f32,
                    Some(&bloom),
                    3u8,
                    64u8,
                    0.01f32,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    &mut q_slice,
                    None,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut trail, &initial_trail);
            let report = engine
                .resolve_slot_bank_swarm_and_steer_safe(
                    black_box(&mut trail),
                    black_box(&bank_slice),
                    black_box(&slot_weights),
                    black_box(DEFAULT_SLOT_COUPLING_MATRIX_8X8),
                    black_box(None::<&gcso_paged_bitmask_t>),
                    black_box(96u8),
                    black_box(8u8),
                    black_box(7u8),
                    black_box(32u8),
                    black_box(0.18f32),
                    black_box(0.12f32),
                    black_box(Some(&bloom)),
                    black_box(3u8),
                    black_box(64u8),
                    black_box(0.01f32),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    black_box(&mut q_slice),
                    None,
                )
                .unwrap();
            black_box((report, q_slice.as_slice()[0]));
        });
    });

    // 2D. Bayes-Attention Pointer-Trail Selection, Rewiring, Score Modulation & RIPA Steering
    group.bench_function("bayesian_pointer_trail_select_rewire_and_steer_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
        let base_src = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_t::pack_action(
                0x3400,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_EXPRESS_PHASE,
                16,
            )
            .raw(),
            stigmergic_density: 1.5,
            attractor_pull_force: 2.0,
            linked_trail_id: 2,
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
                stigmergic_density: 2.0,
                target_anchor_id: 11,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(24),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 3.5,
                target_anchor_id: 29,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(30),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 1.2,
                target_anchor_id: 7,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(12),
                ..Default::default()
            },
        ];
        let _ftz = ScopedFtzdazGuard::new();
        let mut candidates = base_candidates;
        let mut src_trail = base_src;
        for _ in 0..8 {
            reset_trail_fast(&mut src_trail, &base_src);
            reset_trail_fast(&mut candidates[2], &base_candidates[2]);
            attn_scores.data = [0.0f32; 8];
            let mut scores_mut = attn_scores.as_aligned_slice_mut();
            let _ = engine
                .select_rewire_and_steer_bayesian_with_defaults(
                    &mut src_trail,
                    &mut candidates,
                    2.0f32,
                    0.5f32,
                    0.25f32,
                    1.0f32,
                    64u8,
                    0.25f32,
                    Some(&mut scores_mut),
                    &mut q_slice,
                    None,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut src_trail, &base_src);
            reset_trail_fast(&mut candidates[2], &base_candidates[2]);
            attn_scores.data = [0.0f32; 8];
            let mut scores_mut = attn_scores.as_aligned_slice_mut();
            let res = engine
                .select_rewire_and_steer_bayesian_with_defaults(
                    black_box(&mut src_trail),
                    black_box(&mut candidates),
                    black_box(2.0f32),
                    black_box(0.5f32),
                    black_box(0.25f32),
                    black_box(1.0f32),
                    black_box(64u8),
                    black_box(0.25f32),
                    black_box(Some(&mut scores_mut)),
                    black_box(&mut q_slice),
                    None,
                )
                .unwrap();
            black_box((res, scores_mut.as_slice()[2], q_slice.as_slice()[0]));
        });
    });

    // 2E. Limina Inter-Memory Bridging + Attention x PhaseSteering Score Modulation
    group.bench_function(
        "liminal_trail_bridge_and_attention_score_modulate_64h",
        |b| {
            let mut q_slice = shared_q_arena
                .as_aligned_prefix_mut(total_elements)
                .relax_alignment::<32>();
            q_slice.as_mut_slice().fill(1.0f32);
            let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
            let prev_trail = gcso_pointer_trail_t {
                current_ptr: 0x1000,
                stigmergic_density: 1.8,
                attractor_pull_force: 2.5,
                target_anchor_id: 17,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(24),
                ..Default::default()
            };
            let base_curr = gcso_pointer_trail_t {
                current_ptr: 0x2000,
                stigmergic_density: 1.2,
                attractor_pull_force: 1.5,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(32),
                ..Default::default()
            };
            let memory_anchors = [
                make_heterogeneous_q7_phases_with_offset(-96),
                make_heterogeneous_q7_phases_with_offset(28),
                make_heterogeneous_q7_phases_with_offset(30),
                make_heterogeneous_q7_phases_with_offset(0),
            ];

            let _ftz = ScopedFtzdazGuard::new();
            let mut curr_trail = base_curr;
            for _ in 0..8 {
                reset_trail_fast(&mut curr_trail, &base_curr);
                attn_scores.data = [0.0f32; 8];
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let _ = engine
                    .bridge_liminal_trail_and_steer_qk_safe(
                        &prev_trail,
                        &mut curr_trail,
                        64u8,
                        0.25f32,
                        0.01f32,
                        DEFAULT_RIPA_CLAMP_MAX_RAD,
                        2.0f32,
                        Some(&memory_anchors),
                        Some(&mut scores_mut),
                        &mut q_slice,
                        None,
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_trail_fast(&mut curr_trail, &base_curr);
                attn_scores.data = [0.0f32; 8];
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let res = engine
                    .bridge_liminal_trail_and_steer_qk_safe(
                        black_box(&prev_trail),
                        black_box(&mut curr_trail),
                        black_box(64u8),
                        black_box(0.25f32),
                        black_box(0.01f32),
                        black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                        black_box(2.0f32),
                        black_box(Some(&memory_anchors)),
                        black_box(Some(&mut scores_mut)),
                        black_box(&mut q_slice),
                        None,
                    )
                    .unwrap();
                black_box((res, scores_mut.as_slice()[1], q_slice.as_slice()[0]));
            });
        },
    );

    // 2F. Zero-Copy Candidate-Trail Bayes-Attention Score Modulation (8 Candidates)
    group.throughput(Throughput::Elements(8));
    group.bench_function(
        "modulate_candidate_trail_attention_scores_zero_copy_8cand",
        |b| {
            let mut attn_scores = AlignedBuffer32::new([0.5f32; 8]);
            let src_trail = gcso_pointer_trail_t {
                stigmergic_density: 1.8,
                linked_trail_id: 3,
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
                    .modulate_candidate_trail_attention_scores_zero_copy(
                        &mut scores_mut,
                        &src_trail,
                        &candidates,
                        2.0f32,
                        0.5f32,
                        0.25f32,
                        0.25f32,
                        1.0f32,
                    )
                    .unwrap();
            }

            b.iter(|| {
                attn_scores.data = [0.5f32; 8];
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let res = engine
                    .modulate_candidate_trail_attention_scores_zero_copy(
                        black_box(&mut scores_mut),
                        black_box(&src_trail),
                        black_box(&candidates),
                        black_box(2.0f32),
                        black_box(0.5f32),
                        black_box(0.25f32),
                        black_box(0.25f32),
                        black_box(1.0f32),
                    )
                    .unwrap();
                black_box((res, scores_mut.as_slice()[3]));
            });
        },
    );

    // 3. Geodesic S^1 Slerp Phase Interpolation + QDPS Filter + RIPA Steering (64 Heads)
    group.throughput(Throughput::Bytes(tensor_bytes as u64));
    group.bench_function("slerp_geodesic_s1_and_steer_safe_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let phases_a = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(-20));
        let phases_b = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(44));
        let _ftz = ScopedFtzdazGuard::new();
        let pa = phases_a.as_aligned_slice();
        let pb = phases_b.as_aligned_slice();
        for _ in 0..8 {
            let _ = engine
                .slerp_and_apply_phase_steering_safe(
                    &pa,
                    &pb,
                    64u8,
                    0.01f32,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    &mut q_slice,
                    None,
                )
                .unwrap();
        }

        b.iter(|| {
            let out = engine
                .slerp_and_apply_phase_steering_safe(
                    black_box(&pa),
                    black_box(&pb),
                    black_box(64u8),
                    black_box(0.01f32),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    black_box(&mut q_slice),
                    None,
                )
                .unwrap();
            black_box((out, q_slice.as_slice()[0]));
        });
    });

    // 4. 64-Head Circular Phase Coherence & von Mises Concentration Evaluation
    group.throughput(Throughput::Elements(MAX_ATTENTION_HEADS as u64));
    group.bench_function("phase_coherence_and_circular_resultant_64h", |b| {
        let phases_a = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(28));
        let phases_b = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(32));
        let pa = phases_a.as_aligned_slice();
        let pb = phases_b.as_aligned_slice();

        b.iter(|| {
            let stats = engine
                .evaluate_phase_coherence_and_resultant_64(black_box(&pa), black_box(&pb))
                .unwrap();
            black_box(stats);
        });
    });

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
        let pspm_cfg = gcso_pspm_config_t {
            num_fact_heads: 32,
            num_logic_heads: 24,
            num_explore_heads: 8,
            fact_phase_gain: 1.0,
            logic_phase_gain: 0.85,
            explore_phase_gain: 1.20,
            ..Default::default()
        };
        for _ in 0..8 {
            engine
                .dispatch_subhead_slot_ensemble(
                    &mut q_slice,
                    &pspm_cfg,
                    &bank_slice,
                    ATTRACTOR_SLOT_SYSTEM_BASELINE,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    ATTRACTOR_SLOT_PERSONA,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    8i8,
                    head_dim as usize,
                )
                .unwrap();
        }

        b.iter(|| {
            engine
                .dispatch_subhead_slot_ensemble(
                    black_box(&mut q_slice),
                    black_box(&pspm_cfg),
                    black_box(&bank_slice),
                    black_box(ATTRACTOR_SLOT_SYSTEM_BASELINE),
                    black_box(ATTRACTOR_SLOT_KNOWLEDGE),
                    black_box(ATTRACTOR_SLOT_PERSONA),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
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
        let pspm_cfg = gcso_pspm_config_t {
            num_fact_heads: 32,
            num_logic_heads: 24,
            num_explore_heads: 8,
            fact_phase_gain: 1.0,
            logic_phase_gain: 0.85,
            explore_phase_gain: 1.20,
            ..Default::default()
        };
        for _ in 0..8 {
            engine
                .dispatch_subhead_slot_ensemble_geodesic(
                    &mut q_slice,
                    &pspm_cfg,
                    &bank_slice,
                    ATTRACTOR_SLOT_SYSTEM_BASELINE,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    ATTRACTOR_SLOT_PERSONA,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    8i8,
                    head_dim as usize,
                )
                .unwrap();
        }

        b.iter(|| {
            engine
                .dispatch_subhead_slot_ensemble_geodesic(
                    black_box(&mut q_slice),
                    black_box(&pspm_cfg),
                    black_box(&bank_slice),
                    black_box(ATTRACTOR_SLOT_SYSTEM_BASELINE),
                    black_box(ATTRACTOR_SLOT_KNOWLEDGE),
                    black_box(ATTRACTOR_SLOT_PERSONA),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    black_box(8i8),
                    black_box(head_dim as usize),
                )
                .unwrap();
            black_box(q_slice.as_slice()[0]);
        });
    });

    // 5C. Single-Manifold Unified-Field + Sub-Head Pseudo-MoE (Non-Unit Gains) + Inverse-Temperature Logit Sharpening
    group.bench_function(
        "unified_field_subhead_and_inv_temp_logit_sharpen_64h",
        |b| {
            let mut q_slice = shared_q_arena
                .as_aligned_prefix_mut(total_elements)
                .relax_alignment::<32>();
            for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
                *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
            }
            let mut temp_logits =
                AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
            let unified_anchors = [
                make_heterogeneous_q7_phases_with_offset(-96),
                make_heterogeneous_q7_phases_with_offset(32),
            ];
            let unified_energies = [0.2f32, 3.0f32];
            let pspm_cfg = gcso_pspm_config_t {
                num_fact_heads: 32,
                num_logic_heads: 24,
                num_explore_heads: 8,
                fact_phase_gain: 1.0,
                logic_phase_gain: 0.90,
                explore_phase_gain: 1.15,
                ..Default::default()
            };
            let base_trail = gcso_pointer_trail_t {
                current_ptr: 0x3800,
                stigmergic_density: 1.4,
                attractor_pull_force: 1.5,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(28),
                ..Default::default()
            };

            let _ftz = ScopedFtzdazGuard::new();
            let mut trail = base_trail;
            for _ in 0..8 {
                reset_trail_fast(&mut trail, &base_trail);
                temp_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
                let mut logits_mut = temp_logits.as_aligned_slice_mut();
                let _ = engine
                    .steer_unified_field_pspm_and_inv_temperature_safe(
                        &mut trail,
                        &unified_anchors,
                        &unified_energies,
                        None,
                        Some((&pspm_cfg, 8i8)),
                        Some((&mut logits_mut, 0.75f32)),
                        0.01f32,
                        DEFAULT_RIPA_CLAMP_MAX_RAD,
                        &mut q_slice,
                        None,
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_trail_fast(&mut trail, &base_trail);
                temp_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
                let mut logits_mut = temp_logits.as_aligned_slice_mut();
                let res = engine
                    .steer_unified_field_pspm_and_inv_temperature_safe(
                        black_box(&mut trail),
                        black_box(&unified_anchors),
                        black_box(&unified_energies),
                        black_box(None),
                        black_box(Some((&pspm_cfg, 8i8))),
                        black_box(Some((&mut logits_mut, 0.75f32))),
                        black_box(0.01f32),
                        black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                        black_box(&mut q_slice),
                        None,
                    )
                    .unwrap();
                black_box((res, logits_mut.as_slice()[0], q_slice.as_slice()[0]));
            });
        },
    );

    // 5D. Classical Online Toroidal k-Means / DP-Means E+M Closed-Loop Gateway Steering (K = 4 Centroids)
    group.bench_function("online_toroidal_kmeans_gateway_steer_4centroids_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let mut gw_logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
        let sys_sig = SystemConstraint::new((), 0.15, 0.05, 0xFF);
        let fact_sig = ObjectiveFact::new(
            make_heterogeneous_q7_phases_with_offset(28),
            3.0,
            0x0024_0000,
            88,
        );
        let pers_sig = SubjectivePersona::new(
            make_heterogeneous_q7_phases_with_offset(30),
            2.0,
            0x0000_9000,
            12,
        );
        let composite_signal = (&sys_sig, &fact_sig, &pers_sig);
        let base_centroids: [CrystallizedAttractorRecord; 4] = core::array::from_fn(|i| {
            let mut rec = CrystallizedAttractorRecord::new_empty(
                (i as u32) + 10,
                ATTRACTOR_SLOT_KNOWLEDGE,
                0x0010_0000 + ((i as u64) << 12),
            );
            let _ = rec.absorb_sample_point_64(
                &make_heterogeneous_q7_phases_with_offset(16 + (i as i8) * 6),
                2.0 + (i as f32) * 0.5,
            );
            rec
        });
        let base_trail = gcso_pointer_trail_t {
            stigmergic_density: 1.4,
            attractor_pull_force: 1.5,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };
        let mut centroids = base_centroids;
        let mut trail = base_trail;
        for _ in 0..8 {
            reset_trail_fast(&mut trail, &base_trail);
            reset_crystal_fast(&mut centroids[2], &base_centroids[2]);
            gw_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
            let mut logits_mut = gw_logits.as_aligned_slice_mut();
            let _ = engine
                .steer_and_kmeans_crystallize_gateway_signal_safe(
                    &composite_signal,
                    &mut trail,
                    &mut centroids,
                    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                    64u8,
                    &mut q_slice,
                    None,
                    Some(&mut logits_mut),
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut trail, &base_trail);
            reset_crystal_fast(&mut centroids[2], &base_centroids[2]);
            gw_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
            let mut logits_mut = gw_logits.as_aligned_slice_mut();
            let res = engine
                .steer_and_kmeans_crystallize_gateway_signal_safe(
                    black_box(&composite_signal),
                    black_box(&mut trail),
                    black_box(&mut centroids),
                    black_box(DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE),
                    black_box(64u8),
                    black_box(&mut q_slice),
                    None,
                    black_box(Some(&mut logits_mut)),
                )
                .unwrap();
            black_box((res, trail.cluster_id, q_slice.as_slice()[0]));
        });
    });

    // 5E. Coarse-to-Fine Macro-to-Micro Routing (O(K=4) Centroids -> O(M_local=2) Intra-Cluster Candidates out of 8 Total) + RIPA Steering
    group.bench_function("coarse_to_fine_macro_to_micro_routing_and_steer_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let centroids: [CrystallizedAttractorRecord; 4] = core::array::from_fn(|i| {
            let mut rec = CrystallizedAttractorRecord::new_empty(
                ((i as u32) + 1) * 100,
                ATTRACTOR_SLOT_KNOWLEDGE,
                0x0010_0000 + ((i as u64) << 12),
            );
            let _ = rec.absorb_sample_point_64(
                &make_heterogeneous_q7_phases_with_offset(if i == 2 {
                    28i8
                } else {
                    -60 + (i as i8) * 20
                }),
                3.0,
            );
            rec
        });
        let base_src = gcso_pointer_trail_t {
            stigmergic_density: 1.5,
            attractor_pull_force: 2.0,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };
        // 8 Candidates (2 per cluster across clusters 100, 200, 300, 400) so the Fine step genuinely
        // evaluates multiple candidates inside the winning macro cluster (cluster 300: indices 4 and 5).
        let base_candidates: [gcso_pointer_trail_t; 8] = [
            gcso_pointer_trail_t {
                stigmergic_density: 3.0,
                cluster_id: 100,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-60),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.2,
                cluster_id: 100,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-55),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.5,
                cluster_id: 200,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-40),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.1,
                cluster_id: 200,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-36),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.6,
                target_anchor_id: 301,
                cluster_id: 300,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(23),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 3.5,
                target_anchor_id: 302,
                cluster_id: 300,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(29),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 1.8,
                cluster_id: 400,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(0),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 1.4,
                cluster_id: 400,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(6),
                ..Default::default()
            },
        ];
        let mut candidates = base_candidates;

        let mut src_trail = base_src;
        for _ in 0..8 {
            reset_trail_fast(&mut src_trail, &base_src);
            reset_trail_fast(&mut candidates[5], &base_candidates[5]);
            let _ = engine
                .route_coarse_to_fine_and_steer_bayesian_trails_safe(
                    &centroids,
                    64u8,
                    &mut src_trail,
                    &mut candidates,
                    2.0f32,
                    0.5f32,
                    0.25f32,
                    0.50f32,
                    64u8,
                    0.25f32,
                    0.01f32,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    None,
                    &mut q_slice,
                    None,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut src_trail, &base_src);
            reset_trail_fast(&mut candidates[5], &base_candidates[5]);
            let res = engine
                .route_coarse_to_fine_and_steer_bayesian_trails_safe(
                    black_box(&centroids),
                    black_box(64u8),
                    black_box(&mut src_trail),
                    black_box(&mut candidates),
                    black_box(2.0f32),
                    black_box(0.5f32),
                    black_box(0.25f32),
                    black_box(0.50f32),
                    black_box(64u8),
                    black_box(0.25f32),
                    black_box(0.01f32),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    None,
                    black_box(&mut q_slice),
                    None,
                )
                .unwrap();
            black_box((res, src_trail.linked_trail_id, q_slice.as_slice()[0]));
        });
    });

    // 5F. Top-2 Liminal Macro-to-Micro Bifurcation Routing (O(K=4) Top-2 Centroids -> Dual-Cluster Fine Selection across 8 Candidates) + RIPA Steering
    group.bench_function("top2_liminal_bifurcation_route_and_steer_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let centroids: [CrystallizedAttractorRecord; 4] = [
            {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    100,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0010_0000,
                );
                let _ =
                    rec.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(-70), 2.0);
                rec
            },
            {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    200,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0011_0000,
                );
                let _ =
                    rec.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(24), 3.0);
                rec
            },
            {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    300,
                    ATTRACTOR_SLOT_PERSONA,
                    0x0012_0000,
                );
                let _ =
                    rec.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(28), 2.9);
                rec
            },
            {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    400,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0013_0000,
                );
                let _ =
                    rec.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(-45), 1.5);
                rec
            },
        ];
        let base_src = gcso_pointer_trail_t {
            stigmergic_density: 1.5,
            attractor_pull_force: 2.0,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };
        // 8 Candidates (2 per cluster across 4 centroids) so Top-2 Liminal routing evaluates 4 active candidates across the 2 winning clusters (clusters 200 & 300).
        let base_candidates: [gcso_pointer_trail_t; 8] = [
            gcso_pointer_trail_t {
                stigmergic_density: 2.5,
                cluster_id: 100,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-70),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.1,
                cluster_id: 100,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-66),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.7,
                target_anchor_id: 201,
                cluster_id: 200,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(21),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 3.1,
                target_anchor_id: 202,
                cluster_id: 200,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(24),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 2.9,
                target_anchor_id: 301,
                cluster_id: 300,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(25),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 3.6,
                target_anchor_id: 302,
                cluster_id: 300,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(28),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 1.8,
                cluster_id: 400,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-45),
                ..Default::default()
            },
            gcso_pointer_trail_t {
                stigmergic_density: 1.4,
                cluster_id: 400,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(-40),
                ..Default::default()
            },
        ];
        let mut candidates = base_candidates;

        let mut src_trail = base_src;
        for _ in 0..8 {
            reset_trail_fast(&mut src_trail, &base_src);
            reset_trail_fast(&mut candidates[5], &base_candidates[5]);
            let _ = engine
                .route_top2_liminal_and_steer_bayesian_trails_safe(
                    &centroids,
                    64u8,
                    0.20f32,
                    &mut src_trail,
                    &mut candidates,
                    2.0f32,
                    0.5f32,
                    0.25f32,
                    0.50f32,
                    64u8,
                    0.25f32,
                    0.01f32,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    None,
                    &mut q_slice,
                    None,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut src_trail, &base_src);
            reset_trail_fast(&mut candidates[5], &base_candidates[5]);
            let res = engine
                .route_top2_liminal_and_steer_bayesian_trails_safe(
                    black_box(&centroids),
                    black_box(64u8),
                    black_box(0.20f32),
                    black_box(&mut src_trail),
                    black_box(&mut candidates),
                    black_box(2.0f32),
                    black_box(0.5f32),
                    black_box(0.25f32),
                    black_box(0.50f32),
                    black_box(64u8),
                    black_box(0.25f32),
                    black_box(0.01f32),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    None,
                    black_box(&mut q_slice),
                    None,
                )
                .unwrap();
            black_box((res, src_trail.linked_trail_id, q_slice.as_slice()[0]));
        });
    });

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
            let (delta_mask, _) = engine.detect_frame_delta_bitmask(&prev_mask, &curr_mask);
            let _ = engine
                .steer_visual_grid_2d_bitmask_gated(
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
                engine.detect_frame_delta_bitmask(black_box(&prev_mask), black_box(&curr_mask));
            let steered = engine
                .steer_visual_grid_2d_bitmask_gated(
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
            let (delta_mask, _) = engine.detect_frame_delta_bitmask(&prev_mask, &curr_mask);
            let _ = engine
                .steer_visual_grid_3d_bitmask_gated(
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
                engine.detect_frame_delta_bitmask(black_box(&prev_mask), black_box(&curr_mask));
            let steered = engine
                .steer_visual_grid_3d_bitmask_gated(
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

    // 7. Zero-Hyperparameter Swarm & Blanket Autonomous Frequency Aperture Steering (64 Heads)
    group.throughput(Throughput::Bytes(tensor_bytes as u64));
    group.bench_function("apply_autonomous_swarm_aperture_steering_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let deltas_64 = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(24));
        let d_slice = deltas_64.as_aligned_slice();
        let _ftz = ScopedFtzdazGuard::new();
        for _ in 0..8 {
            let _ = engine
                .apply_autonomous_swarm_phase_steering(
                    &mut q_slice,
                    None,
                    &d_slice,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    0x06,
                    0xFF,
                    0.22f32,
                    2.4f32,
                )
                .unwrap();
        }

        b.iter(|| {
            let rot_dim = engine
                .apply_autonomous_swarm_phase_steering(
                    black_box(&mut q_slice),
                    None,
                    black_box(&d_slice),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    black_box(0x06),
                    black_box(0xFF),
                    black_box(0.22f32),
                    black_box(2.4f32),
                )
                .unwrap();
            black_box((rot_dim, q_slice.as_slice()[0]));
        });
    });

    group.finish();
}

/// Provisional `.gcso` / MappedStorage 4-Section Container Benchmarks (Isolated from Core PhaseSteering Suite).
///
/// Evaluates pre-v1.0 `.gcso` record crystallization and 4-section zero-copy byte container mapping.
/// Skipped by default under `run_benchmarks.py --profile standard` so Core `PhaseSteering`/`EntropyControl` evaluations
/// remain decoupled from the provisional container layout prior to v1.0 PPRC video-stream memory integration.
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
    let engine = PhaseSteeringEngine::new(head_dim, num_heads)
        .expect("Failed 64-head PhaseSteeringEngine init");
    let mut shared_q_arena =
        AlignedHeapBuffer::<f32, 512>::new_with_phase_offset(total_elements, 1.0f32, 0);

    group.throughput(Throughput::Bytes(tensor_bytes as u64));

    // 1. Provisional 2-Origin Composite Manifold Gateway + SO(2)-Fused RIPA/PSPM Steering + `.gcso` Crystallization
    group.bench_function("gateway_signal_steer_and_crystallize_attractor_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
        let mut gw_logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
        let sys_sig = SystemConstraint::new((), 0.20, 0.05, 0xFF);
        let fact_sig = ObjectiveFact::new(
            make_heterogeneous_q7_phases_with_offset(28),
            3.0,
            0x0024_0000,
            88,
        );
        let pers_sig = SubjectivePersona::new(
            make_heterogeneous_q7_phases_with_offset(30),
            2.0,
            0x0000_9000,
            12,
        );
        let composite_signal = (&sys_sig, &fact_sig, &pers_sig);
        let mut base_crystal =
            CrystallizedAttractorRecord::new_empty(88, ATTRACTOR_SLOT_KNOWLEDGE, 0x0024_0000);
        let _ =
            base_crystal.absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(26), 2.0);
        let base_trail = gcso_pointer_trail_t {
            stigmergic_density: 1.2,
            attractor_pull_force: 1.5,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let mut trail = base_trail;
        let mut crystal = base_crystal;
        for _ in 0..8 {
            reset_trail_fast(&mut trail, &base_trail);
            reset_crystal_fast(&mut crystal, &base_crystal);
            gw_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
            let mut logits_mut = gw_logits.as_aligned_slice_mut();
            let _ = engine
                .steer_and_crystallize_gateway_signal_safe(
                    &composite_signal,
                    &mut trail,
                    &mut crystal,
                    64u8,
                    &mut q_slice,
                    None,
                    Some(&mut logits_mut),
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut trail, &base_trail);
            reset_crystal_fast(&mut crystal, &base_crystal);
            gw_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
            let mut logits_mut = gw_logits.as_aligned_slice_mut();
            let res = engine
                .steer_and_crystallize_gateway_signal_safe(
                    black_box(&composite_signal),
                    black_box(&mut trail),
                    black_box(&mut crystal),
                    black_box(64u8),
                    black_box(&mut q_slice),
                    None,
                    black_box(Some(&mut logits_mut)),
                )
                .unwrap();
            black_box((res, logits_mut.as_slice()[0], q_slice.as_slice()[0]));
        });
    });

    // 2. Provisional End-to-End 2-Origin Gateway -> 8-Cell Swarm -> Bayesian Trail Rewire -> Subhead -> `.gcso` Crystallization
    group.bench_function(
        "gateway_swarm_bayesian_subhead_and_crystallize_full_64h",
        |b| {
            let mut q_slice = shared_q_arena
                .as_aligned_prefix_mut(total_elements)
                .relax_alignment::<32>();
            q_slice.as_mut_slice().fill(1.0f32);
            let mut gw_logits =
                AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
            let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
            let sys_sig = SystemConstraint::new((), 0.20, 0.05, 0xFF);
            let fact_sig = ObjectiveFact::new(
                make_heterogeneous_q7_phases_with_offset(28),
                3.0,
                0x0024_0000,
                88,
            );
            let pers_sig = SubjectivePersona::new(
                make_heterogeneous_q7_phases_with_offset(30),
                2.0,
                0x0000_9000,
                12,
            );
            let composite_signal = (&sys_sig, &fact_sig, &pers_sig);
            let bloom = ContrastivePhaseBloomFilter::new();
            let mut base_crystal =
                CrystallizedAttractorRecord::new_empty(88, ATTRACTOR_SLOT_KNOWLEDGE, 0x0024_0000);
            let _ = base_crystal
                .absorb_sample_point_64(&make_heterogeneous_q7_phases_with_offset(26), 2.0);
            let base_trail = gcso_pointer_trail_t {
                stigmergic_density: 1.5,
                attractor_pull_force: 2.0,
                linked_trail_id: 1,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
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
                    target_anchor_id: 88,
                    accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(30),
                    ..Default::default()
                },
            ];
            let mut candidates = base_candidates;

            let mut trail = base_trail;
            let mut crystal = base_crystal;
            for _ in 0..8 {
                reset_trail_fast(&mut trail, &base_trail);
                reset_crystal_fast(&mut crystal, &base_crystal);
                reset_trail_fast(&mut candidates[1], &base_candidates[1]);
                gw_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
                attn_scores.data = [0.0f32; 8];
                let mut logits_mut = gw_logits.as_aligned_slice_mut();
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let _ = engine
                    .steer_gateway_swarm_bayesian_and_crystallize_safe(
                        &composite_signal,
                        &mut trail,
                        Some(&mut candidates),
                        Some(&mut scores_mut),
                        96u8,
                        16u8,
                        6u8,
                        64u8,
                        0.15f32,
                        0.10f32,
                        Some(&bloom),
                        3u8,
                        64u8,
                        2.0f32,
                        0.5f32,
                        0.25f32,
                        1.0f32,
                        64u8,
                        0.25f32,
                        4i8,
                        &mut q_slice,
                        None,
                        Some(&mut logits_mut),
                        Some(&mut crystal),
                    )
                    .unwrap();
            }

            b.iter(|| {
                reset_trail_fast(&mut trail, &base_trail);
                reset_crystal_fast(&mut crystal, &base_crystal);
                reset_trail_fast(&mut candidates[1], &base_candidates[1]);
                gw_logits.data = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25];
                attn_scores.data = [0.0f32; 8];
                let mut logits_mut = gw_logits.as_aligned_slice_mut();
                let mut scores_mut = attn_scores.as_aligned_slice_mut();
                let res = engine
                    .steer_gateway_swarm_bayesian_and_crystallize_safe(
                        black_box(&composite_signal),
                        black_box(&mut trail),
                        black_box(Some(&mut candidates)),
                        black_box(Some(&mut scores_mut)),
                        black_box(96u8),
                        black_box(16u8),
                        black_box(6u8),
                        black_box(64u8),
                        black_box(0.15f32),
                        black_box(0.10f32),
                        black_box(Some(&bloom)),
                        black_box(3u8),
                        black_box(64u8),
                        black_box(2.0f32),
                        black_box(0.5f32),
                        black_box(0.25f32),
                        black_box(1.0f32),
                        black_box(64u8),
                        black_box(0.25f32),
                        black_box(4i8),
                        black_box(&mut q_slice),
                        None,
                        black_box(Some(&mut logits_mut)),
                        black_box(Some(&mut crystal)),
                    )
                    .unwrap();
                black_box((res, scores_mut.as_slice()[1], q_slice.as_slice()[0]));
            });
        },
    );

    // 3. Provisional `.gcso` (MappedStorage) Zero-Copy Deserialization + Phase-Proximity Gravity Ensemble Gateway Steering
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
            CrystallizedAttractorRecord::write_records_to_zimms_bytes(
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
                let zero_copy_recs = CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy(
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
                let zero_copy_recs = CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy(
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
    group.bench_function("zero_copy_register_crystallized_ensemble_steer_64h", |b| {
        let mut q_slice = shared_q_arena
            .as_aligned_prefix_mut(total_elements)
            .relax_alignment::<32>();
        for (idx, val) in q_slice.as_mut_slice().iter_mut().enumerate() {
            *val = 0.75f32 + ((idx & 7) as f32) * 0.09f32;
        }
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
        let base_trail = gcso_pointer_trail_t {
            stigmergic_density: 1.2,
            attractor_pull_force: 1.5,
            accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
            ..Default::default()
        };

        let mut trail = base_trail;
        for _ in 0..8 {
            reset_trail_fast(&mut trail, &base_trail);
            let _ = engine
                .steer_crystallized_ensemble_zero_copy_safe(
                    &records,
                    &mut trail,
                    0.01f32,
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                    &mut q_slice,
                    None,
                )
                .unwrap();
        }

        b.iter(|| {
            reset_trail_fast(&mut trail, &base_trail);
            let res = engine
                .steer_crystallized_ensemble_zero_copy_safe(
                    black_box(&records),
                    black_box(&mut trail),
                    black_box(0.01f32),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                    black_box(&mut q_slice),
                    None,
                )
                .unwrap();
            black_box((res, trail.target_anchor_id, q_slice.as_slice()[0]));
        });
    });

    // 5. Provisional 4-Section `.gcso` (MappedStorage) Zero-Copy Container Map + Register-Resident Ensemble Steering
    group.bench_function(
        "four_section_gcso_zero_copy_map_and_register_steer_64h",
        |b| {
            let mut q_slice = shared_q_arena
                .as_aligned_prefix_mut(total_elements)
                .relax_alignment::<32>();
            q_slice.as_mut_slice().fill(1.0f32);
            let crystals: [CrystallizedAttractorRecord; 2] = core::array::from_fn(|i| {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    (i as u32) + 50,
                    ATTRACTOR_SLOT_KNOWLEDGE,
                    0x0018_0000 + ((i as u64) << 12),
                );
                let _ = rec.absorb_sample_point_64(
                    &make_heterogeneous_q7_phases_with_offset(24 + (i as i8) * 4),
                    3.0,
                );
                rec
            });
            let trails = [gcso_pointer_trail_t {
                stigmergic_density: 1.5,
                attractor_pull_force: 2.0,
                target_anchor_id: 50,
                cluster_id: 50,
                accumulated_phase_delta: make_heterogeneous_q7_phases_with_offset(26),
                ..Default::default()
            }];
            let bloom = ContrastivePhaseBloomFilter::new();
            let daes = gcso_daes_slot_t::default();
            let mut container = AlignedBuffer128::<u8, 768>::new([0u8; 768]);
            let written = GcsoFourSectionView::write_four_section_container_to_bytes(
                &crystals,
                &trails,
                Some(&bloom),
                Some(&daes),
                1_774_900_000,
                &mut container.data,
            )
            .unwrap();

            let mut trail = trails[0];
            for _ in 0..8 {
                let view =
                    GcsoFourSectionView::from_zimms_bytes_zero_copy(&container.data[..written])
                        .unwrap();
                reset_trail_fast(&mut trail, &view.trails[0]);
                let _ = engine
                    .steer_crystallized_ensemble_zero_copy_safe(
                        view.crystals,
                        &mut trail,
                        0.01f32,
                        DEFAULT_RIPA_CLAMP_MAX_RAD,
                        &mut q_slice,
                        None,
                    )
                    .unwrap();
            }

            b.iter(|| {
                let view = GcsoFourSectionView::from_zimms_bytes_zero_copy(black_box(
                    &container.data[..written],
                ))
                .unwrap();
                reset_trail_fast(&mut trail, &view.trails[0]);
                let res = engine
                    .steer_crystallized_ensemble_zero_copy_safe(
                        black_box(view.crystals),
                        black_box(&mut trail),
                        black_box(0.01f32),
                        black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
                        black_box(&mut q_slice),
                        None,
                    )
                    .unwrap();
                black_box((res, trail.target_anchor_id, q_slice.as_slice()[0]));
            });
        },
    );

    group.finish();
}

fn bench_phase_steering_extreme_and_edge_cases(c: &mut Criterion) {
    let _ftz = ScopedFtzdazGuard::new();
    let mut group = c.benchmark_group("PhaseSteering_Extreme_And_Edge_Cases");
    group.noise_threshold(configured_noise_threshold());

    // 1. Stress Test: Ultra-Large Sequence Scaling (4096..32768 Prefill Context)
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(600));
    group.measurement_time(Duration::from_millis(1800));

    let large_contexts = [4096usize, 8192, 16384, 32768];
    let head_dim = 128u32;
    let num_heads = 32u32;
    let stride = (head_dim * num_heads) as usize;
    let single_tok_bytes = stride * core::mem::size_of::<f32>();
    let engine =
        PhaseSteeringEngine::new(head_dim, num_heads).expect("Failed PhaseSteeringEngine init");
    let mut shared_prefill_batch = AlignedHeapBuffer::<f32, 64>::new(stride * 32768, 1.0f32);

    for &tokens in &large_contexts {
        let total_elements = stride * tokens;
        let total_bytes = total_elements * core::mem::size_of::<f32>();

        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("ultra_large_context_prefill", format!("tokens_{tokens}")),
            &tokens,
            |b, &t_count| {
                let mut q_slice = shared_prefill_batch
                    .as_aligned_prefix_mut(total_elements)
                    .relax_alignment::<32>();
                let phase_data = AlignedQ7Buffer(make_heterogeneous_q7_phases_with_offset(15));
                let p_slice = AlignedSlice16::new(&phase_data.0[..32]).unwrap();
                engine
                    .apply_phase_steering_batch(&mut q_slice, &p_slice, t_count)
                    .unwrap();

                b.iter(|| {
                    engine
                        .apply_phase_steering_batch(
                            black_box(&mut q_slice),
                            black_box(&p_slice),
                            black_box(t_count),
                        )
                        .unwrap();
                    black_box(q_slice.as_slice()[0]);
                });
            },
        );
    }

    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(100);
    group.warm_up_time(Duration::from_millis(450));
    group.measurement_time(Duration::from_millis(1500));

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
                    DEFAULT_RIPA_CLAMP_MAX_RAD,
                )
                .unwrap();
        }

        b.iter(|| {
            engine
                .apply_phase_steering_geodesic_safe(
                    black_box(&mut q_slice),
                    black_box(&p_slice),
                    black_box(DEFAULT_RIPA_CLAMP_MAX_RAD),
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

    // 7. Long-Horizon 1M-Token S^1 Geodesic Seam Crossing Cascade (1,024 Wrap Crossings)
    let cascade_steps = 1024usize;
    group.throughput(Throughput::Elements(cascade_steps as u64));
    group.bench_function("long_horizon_1m_seam_crossing_cascade", |b| {
        let _ftz = ScopedFtzdazGuard::new();
        let mut q_slice = shared_extreme_arena
            .as_aligned_prefix_mut(stride)
            .relax_alignment::<32>();
        q_slice.as_mut_slice().fill(1.0f32);
        let init_ctx = make_heterogeneous_q7_phases_with_offset(110);
        let init_ctx_w = liminika_core::traits::primitives::load_q7_words_64(&init_ctx);
        let mut ctx_accum = AlignedBuffer16::new(init_ctx);
        let step_deltas = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(29));
        let q_phases = AlignedBuffer16::new(make_heterogeneous_q7_phases_with_offset(-96));
        let del_slice = step_deltas.as_aligned_slice();
        let qp_slice = q_phases.as_aligned_slice();
        for _ in 0..4 {
            liminika_core::traits::primitives::store_q7_words_64(&mut ctx_accum.data, &init_ctx_w);
            let mut ctx_mut = ctx_accum.as_aligned_slice_mut();
            for _ in 0..cascade_steps {
                engine
                    .accumulate_context_phases_circular_q7(&mut ctx_mut, &del_slice, 0.01f32)
                    .unwrap();
            }
        }

        b.iter(|| {
            liminika_core::traits::primitives::store_q7_words_64(&mut ctx_accum.data, &init_ctx_w);
            let step_rad = black_box(0.01f32);
            {
                let mut ctx_mut = ctx_accum.as_aligned_slice_mut();
                for _ in 0..cascade_steps {
                    engine
                        .accumulate_context_phases_circular_q7(&mut ctx_mut, &del_slice, step_rad)
                        .unwrap();
                }
            }
            let cp_slice = ctx_accum.as_aligned_slice();
            let eff = engine
                .apply_seam_free_lazy_unwrap_q7(
                    black_box(&mut q_slice),
                    None,
                    black_box(&qp_slice),
                    black_box(&cp_slice),
                    black_box(0.01f32),
                    black_box(Some(DEFAULT_RIPA_CLAMP_MAX_RAD)),
                )
                .unwrap();
            black_box((eff[0], q_slice.as_slice()[0]));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_phase_steering_operations,
    bench_phase_steering_batch_sequence_scaling,
    bench_phase_steering_long_context_and_procrustes,
    bench_phase_steering_swarm_subhead_and_visual_cowatching,
    bench_phase_steering_extreme_and_edge_cases,
    bench_phase_steering_provisional_gcso_container
);
criterion_main!(benches);
