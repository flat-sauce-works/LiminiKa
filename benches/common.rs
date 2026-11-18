// name: benches/common.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared benchmark utilities (FTZ/DAZ MXCSR guard, cache-line-aligned heap
//! buffers, and Criterion configuration helpers) for LiminiKa benchmarks.

#![allow(dead_code)]
#![allow(missing_docs)]

use liminika_core::traits::{AlignedSlice, AlignedSliceMut};
use std::alloc::{alloc, dealloc, Layout};

/// Scoped x86_64 MXCSR Flush-To-Zero (`FTZ`, bit 15 = `0x8000`) and Denormals-Are-Zero
/// (`DAZ`, bit 6 = `0x0040`) guard preventing hardware subnormal microcode traps across
/// multi-hundred-thousand-iteration Criterion linear sample batches.
pub(crate) struct ScopedFtzdazGuard {
    #[cfg(target_arch = "x86_64")]
    prev_csr: u32,
}

impl ScopedFtzdazGuard {
    #[inline(always)]
    pub(crate) fn new() -> Self {
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
/// (`phase_offset_elems`) to prevent L1D 4K Load-Store Queue aliasing across tensors.
pub(crate) struct AlignedHeapBuffer<T, const ALIGN: usize = 32> {
    base_ptr: *mut T,
    data_ptr: *mut T,
    layout: Layout,
    len: usize,
}

impl<T: Copy, const ALIGN: usize> AlignedHeapBuffer<T, ALIGN> {
    pub(crate) fn new(len: usize, initial_val: T) -> Self {
        Self::new_with_phase_offset(len, initial_val, 0)
    }

    pub(crate) fn new_with_phase_offset(
        len: usize,
        initial_val: T,
        phase_offset_elems: usize,
    ) -> Self {
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

    pub(crate) fn new_heterogeneous_f32(
        len: usize,
        base_val: f32,
    ) -> AlignedHeapBuffer<f32, ALIGN> {
        let mut buf = AlignedHeapBuffer::<f32, ALIGN>::new(len, base_val);
        for (i, elem) in buf.as_mut_slice().iter_mut().enumerate() {
            let wave = (((i.wrapping_mul(17).wrapping_add((i >> 7).wrapping_mul(31))) & 255)
                as f32)
                * (1.0f32 / 255.0f32);
            *elem = base_val + 0.90f32 * wave;
        }
        buf
    }

    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: `self.data_ptr` is valid for `self.len` elements.
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len) }
    }

    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `self.data_ptr` is valid and uniquely borrowed for `self.len` elements.
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr, self.len) }
    }

    #[inline(always)]
    pub(crate) fn as_prefix(&self, prefix_len: usize) -> &[T] {
        assert!(prefix_len <= self.len);
        // SAFETY: `self.data_ptr` is valid for `prefix_len <= self.len` elements.
        unsafe { std::slice::from_raw_parts(self.data_ptr, prefix_len) }
    }

    #[inline(always)]
    pub(crate) fn as_aligned_slice_mut(&mut self) -> AlignedSliceMut<'_, T, ALIGN> {
        // SAFETY: `self.data_ptr` satisfies `ALIGN`.
        unsafe { AlignedSliceMut::new_unchecked(self.as_mut_slice()) }
    }

    #[inline(always)]
    pub(crate) fn as_aligned_prefix(&self, prefix_len: usize) -> AlignedSlice<'_, T, ALIGN> {
        assert!(prefix_len <= self.len);
        // SAFETY: `self.data_ptr` satisfies `ALIGN` and is valid for `prefix_len <= self.len` elements.
        unsafe {
            AlignedSlice::new_unchecked(std::slice::from_raw_parts(self.data_ptr, prefix_len))
        }
    }

    #[inline(always)]
    pub(crate) fn as_aligned_prefix_mut(
        &mut self,
        prefix_len: usize,
    ) -> AlignedSliceMut<'_, T, ALIGN> {
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

/// Reads `LIMINIKA_NOISE_THRESHOLD` (defaulting to `0.025` = 2.5%) so every Criterion group
/// suppresses sub-30ns RDTSC timer-quantum and background OS scheduler false alarms whether
/// invoked via `python scripts/run_benchmarks.py` or bare `cargo bench`.
#[inline]
pub(crate) fn configured_noise_threshold() -> f64 {
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
pub(crate) fn should_run_provisional_gcso() -> bool {
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
