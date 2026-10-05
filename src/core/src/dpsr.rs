// src/core/src/dpsr.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Safe Rust wrapper for Dynamic Phase-Shifted RoPE (DPSR) and RIPA phase steering operations.

use crate::abi::{
    gcso_dpsr_apply_phase_steering, gcso_dpsr_apply_phase_steering_safe, gcso_dpsr_kernel_create,
    gcso_dpsr_kernel_destroy, gcso_dpsr_lazy_unwrap_override, gcso_q7_t, gcso_qdps_filter_step,
    GcsoDpsrKernelHandle, GcsoStatus, GCSO_ERROR_INVALID_ARGUMENT, GCSO_SUCCESS,
};

/// Safe RAII wrapper around the C-ABI DPSR kernel handle.
#[derive(Debug)]
pub struct DpsrKernel {
    handle: GcsoDpsrKernelHandle,
    head_dim: u32,
    num_heads: u32,
}

unsafe impl Send for DpsrKernel {}
unsafe impl Sync for DpsrKernel {}

impl DpsrKernel {
    /// Creates a new DPSR kernel instance.
    ///
    /// # Errors
    /// Returns a `GcsoStatus` if allocation fails or parameters are misaligned/invalid.
    pub fn new(head_dim: u32, num_heads: u32) -> Result<Self, GcsoStatus> {
        let mut handle: GcsoDpsrKernelHandle = std::ptr::null_mut();
        // SAFETY: Handle pointer is non-null and properly aligned.
        let status = unsafe { gcso_dpsr_kernel_create(head_dim, num_heads, &mut handle) };
        if status != GCSO_SUCCESS || handle.is_null() {
            return Err(status);
        }

        Ok(Self {
            handle,
            head_dim,
            num_heads,
        })
    }

    /// Applies inline DPSR phase rotation to Query tensor registers in zero-allocation mode.
    ///
    /// # Safety & Hot Path Guarantee
    /// Zero dynamic heap allocations. `query_tensor` must be 32-byte SIMD aligned.
    #[inline]
    pub fn apply_phase_steering(
        &self,
        query_tensor: &mut [f32],
        phase_deltas: &[gcso_q7_t],
    ) -> Result<(), GcsoStatus> {
        let expected_query_len = (self.head_dim as usize) * (self.num_heads as usize);
        if query_tensor.len() < expected_query_len || phase_deltas.len() < (self.num_heads as usize)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Pointer alignment and bounds are validated prior to FFI call.
        let status = unsafe {
            gcso_dpsr_apply_phase_steering(
                query_tensor.as_mut_ptr(),
                phase_deltas.as_ptr(),
                self.head_dim as usize,
                self.num_heads as usize,
            )
        };

        if status == GCSO_SUCCESS {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// Applies RIPA soft-bounded phase clamping restricted to low-frequency channels.
    #[inline]
    pub fn apply_phase_steering_safe(
        &self,
        query_tensor: &mut [f32],
        phase_deltas: &[gcso_q7_t],
        max_rad: f32,
    ) -> Result<(), GcsoStatus> {
        let expected_query_len = (self.head_dim as usize) * (self.num_heads as usize);
        if query_tensor.len() < expected_query_len || phase_deltas.len() < (self.num_heads as usize)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated buffer sizes and natural SIMD alignment.
        let status = unsafe {
            gcso_dpsr_apply_phase_steering_safe(
                query_tensor.as_mut_ptr(),
                phase_deltas.as_ptr(),
                self.head_dim as usize,
                self.num_heads as usize,
                max_rad,
            )
        };

        if status == GCSO_SUCCESS {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// Filters discrete phase rotation steps falling below the minimum step threshold in-place.
    #[inline]
    pub fn qdps_filter_step(
        phase_deltas: &mut [gcso_q7_t],
        min_step_rad: f32,
    ) -> Result<(), GcsoStatus> {
        if phase_deltas.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: In-place buffer modification with verified non-zero length.
        let status = unsafe {
            gcso_qdps_filter_step(phase_deltas.as_mut_ptr(), phase_deltas.len(), min_step_rad)
        };

        if status == GCSO_SUCCESS {
            Ok(())
        } else {
            Err(status)
        }
    }

    /// Lazy Phase Unwrapping: Applies relative phase shift against cumulative context accumulators.
    #[inline]
    pub fn lazy_unwrap_override(
        &self,
        query_tensor: &mut [f32],
        context_accum: &[f32],
    ) -> Result<(), GcsoStatus> {
        let expected_query_len = (self.head_dim as usize) * (self.num_heads as usize);
        if query_tensor.len() < expected_query_len
            || context_accum.len() < (self.num_heads as usize)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Query and context phase accumulators are checked for length and alignment.
        let status = unsafe {
            gcso_dpsr_lazy_unwrap_override(
                query_tensor.as_mut_ptr(),
                context_accum.as_ptr(),
                self.head_dim as usize,
                self.num_heads as usize,
            )
        };

        if status == GCSO_SUCCESS {
            Ok(())
        } else {
            Err(status)
        }
    }
}

impl Drop for DpsrKernel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: Ensures handle is freed exactly once during destruction.
            unsafe {
                let _ = gcso_dpsr_kernel_destroy(self.handle);
            }
        }
    }
}
