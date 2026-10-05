// name: src/core/src/traits.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core trait definitions for GCSO engine components.
//!
//! These traits form the formal architectural boundaries between Hot Path execution
//! ($\mathcal{O}(1)$ zero-allocation token loops), Mezzo cellular block coordination,
//! and Cold Path operations (asynchronous attractor field steering, Sheaf entropy evaluations,
//! and ZIMMS container persistence).

use std::ops::{Deref, DerefMut, Index, IndexMut, Range};

use crate::abi::{
    gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_daes_slot_t,
    gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t,
    gcso_pspm_config_t, gcso_q7_t, gcso_srl_descriptor_t, gcso_zimms_descriptor_t, GcsoStatus,
    GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER,
};

/// Type alias for GCSO result responses across core components.
pub type GcsoResult<T> = Result<T, GcsoStatus>;

// ===================================================================
// Const-Generic Zero-Copy Aligned Slice Wrappers
// ===================================================================

/// Zero-copy SIMD/Cacheline aligned slice wrapper to eliminate runtime alignment checks in the hot path.
#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AlignedSlice<'a, T, const ALIGN: usize = 32> {
    slice: &'a [T],
}

impl<'a, T, const ALIGN: usize> AlignedSlice<'a, T, ALIGN> {
    /// Creates a new aligned slice wrapper after validating memory alignment.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `ALIGN` is not a non-zero power of two.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if address is not `ALIGN`-byte aligned and not empty.
    #[inline]
    pub fn new(slice: &'a [T]) -> GcsoResult<Self> {
        if ALIGN == 0 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if slice.is_empty() || ((slice.as_ptr() as usize) & (ALIGN - 1)) == 0 {
            Ok(Self { slice })
        } else {
            Err(GCSO_ERROR_MISALIGNED_POINTER)
        }
    }

    /// Returns the underlying slice reference guaranteed to be aligned.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &'a [T] {
        self.slice
    }

    /// Returns raw pointer to the underlying aligned buffer.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.slice.as_ptr()
    }

    /// Returns the alignment byte boundary constraint for this wrapper.
    #[inline]
    #[must_use]
    pub const fn align(&self) -> usize {
        ALIGN
    }

    /// Returns the number of elements in the slice.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Returns `true` if the slice has a length of 0.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Checks if the provided slice satisfies the specified alignment requirement.
    #[inline]
    #[must_use]
    pub fn is_aligned(slice: &[T]) -> bool {
        ALIGN != 0
            && ALIGN.is_power_of_two()
            && (slice.is_empty() || ((slice.as_ptr() as usize) & (ALIGN - 1)) == 0)
    }

    /// Creates a subslice if the byte offset preserves `ALIGN` alignment boundaries.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if index range is out of bounds.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if subslice start address is misaligned.
    #[inline]
    pub fn subslice(&self, range: Range<usize>) -> GcsoResult<Self> {
        let sub = self.slice.get(range).ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        Self::new(sub)
    }

    /// Splits the aligned slice at the given index if the boundary satisfies alignment constraints.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `mid` exceeds slice length.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if the right subslice start address is misaligned.
    #[inline]
    pub fn split_at(&self, mid: usize) -> GcsoResult<(Self, Self)> {
        if mid > self.slice.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (left, right) = self.slice.split_at(mid);
        Ok((Self::new(left)?, Self::new(right)?))
    }
}

impl<'a, T, const ALIGN: usize> Deref for AlignedSlice<'a, T, ALIGN> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> AsRef<[T]> for AlignedSlice<'a, T, ALIGN> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> Index<usize> for AlignedSlice<'a, T, ALIGN> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.slice[index]
    }
}

impl<'a, T, const ALIGN: usize> Index<Range<usize>> for AlignedSlice<'a, T, ALIGN> {
    type Output = [T];

    #[inline]
    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.slice[range]
    }
}

impl<'a, T, const ALIGN: usize> IntoIterator for AlignedSlice<'a, T, ALIGN> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.slice.iter()
    }
}

impl<'a, 'b, T, const ALIGN: usize> IntoIterator for &'b AlignedSlice<'a, T, ALIGN> {
    type Item = &'b T;
    type IntoIter = std::slice::Iter<'b, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.slice.iter()
    }
}

impl<'a, T, const ALIGN: usize> TryFrom<&'a [T]> for AlignedSlice<'a, T, ALIGN> {
    type Error = GcsoStatus;

    #[inline]
    fn try_from(slice: &'a [T]) -> Result<Self, Self::Error> {
        Self::new(slice)
    }
}

/// Zero-copy SIMD/Cacheline aligned mutable slice wrapper.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq)]
pub struct AlignedSliceMut<'a, T, const ALIGN: usize = 32> {
    slice: &'a mut [T],
}

impl<'a, T, const ALIGN: usize> AlignedSliceMut<'a, T, ALIGN> {
    /// Creates a new mutable aligned slice wrapper after validating memory alignment.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `ALIGN` is not a non-zero power of two.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if address is not `ALIGN`-byte aligned and not empty.
    #[inline]
    pub fn new(slice: &'a mut [T]) -> GcsoResult<Self> {
        if ALIGN == 0 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if slice.is_empty() || ((slice.as_ptr() as usize) & (ALIGN - 1)) == 0 {
            Ok(Self { slice })
        } else {
            Err(GCSO_ERROR_MISALIGNED_POINTER)
        }
    }

    /// Returns the underlying mutable slice reference guaranteed to be aligned.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.slice
    }

    /// Reborrows the mutable slice as an immutable `AlignedSlice`.
    #[inline]
    #[must_use]
    pub fn as_aligned_slice(&self) -> AlignedSlice<'_, T, ALIGN> {
        AlignedSlice { slice: self.slice }
    }

    /// Returns raw mutable pointer to the underlying aligned buffer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.slice.as_mut_ptr()
    }

    /// Returns raw immutable pointer to the underlying aligned buffer.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.slice.as_ptr()
    }

    /// Returns the alignment byte boundary constraint for this wrapper.
    #[inline]
    #[must_use]
    pub const fn align(&self) -> usize {
        ALIGN
    }

    /// Returns the number of elements in the mutable slice.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    /// Returns `true` if the mutable slice has a length of 0.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    /// Checks if the provided slice satisfies the specified alignment requirement.
    #[inline]
    #[must_use]
    pub fn is_aligned(slice: &[T]) -> bool {
        ALIGN != 0
            && ALIGN.is_power_of_two()
            && (slice.is_empty() || ((slice.as_ptr() as usize) & (ALIGN - 1)) == 0)
    }

    /// Creates a mutable subslice if the byte offset preserves `ALIGN` alignment boundaries.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if index range is out of bounds.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if subslice start address is misaligned.
    #[inline]
    pub fn subslice_mut(&mut self, range: Range<usize>) -> GcsoResult<Self> {
        let sub = self.slice.get_mut(range).ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        Self::new(sub)
    }

    /// Splits the mutable slice at the given index if the boundary satisfies alignment constraints.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `mid` exceeds slice length.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if the right subslice start address is misaligned.
    #[inline]
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> GcsoResult<(AlignedSliceMut<'_, T, ALIGN>, AlignedSliceMut<'_, T, ALIGN>)> {
        if mid > self.slice.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (left, right) = self.slice.split_at_mut(mid);
        Ok((AlignedSliceMut::new(left)?, AlignedSliceMut::new(right)?))
    }

    /// Copies elements from an aligned source slice into `self` in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if slice lengths do not match.
    #[inline]
    pub fn copy_from_aligned_slice(&mut self, src: &AlignedSlice<'_, T, ALIGN>) -> GcsoResult<()>
    where
        T: Copy,
    {
        if self.slice.len() != src.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.slice.copy_from_slice(src.as_slice());
        Ok(())
    }
}

impl<'a, T, const ALIGN: usize> Deref for AlignedSliceMut<'a, T, ALIGN> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> DerefMut for AlignedSliceMut<'a, T, ALIGN> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> AsRef<[T]> for AlignedSliceMut<'a, T, ALIGN> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> AsMut<[T]> for AlignedSliceMut<'a, T, ALIGN> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.slice
    }
}

impl<'a, T, const ALIGN: usize> Index<usize> for AlignedSliceMut<'a, T, ALIGN> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.slice[index]
    }
}

impl<'a, T, const ALIGN: usize> IndexMut<usize> for AlignedSliceMut<'a, T, ALIGN> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.slice[index]
    }
}

impl<'a, T, const ALIGN: usize> Index<Range<usize>> for AlignedSliceMut<'a, T, ALIGN> {
    type Output = [T];

    #[inline]
    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.slice[range]
    }
}

impl<'a, T, const ALIGN: usize> IndexMut<Range<usize>> for AlignedSliceMut<'a, T, ALIGN> {
    #[inline]
    fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
        &mut self.slice[range]
    }
}

impl<'a, T, const ALIGN: usize> IntoIterator for AlignedSliceMut<'a, T, ALIGN> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.slice.iter_mut()
    }
}

impl<'a, 'b, T, const ALIGN: usize> IntoIterator for &'b mut AlignedSliceMut<'a, T, ALIGN> {
    type Item = &'b mut T;
    type IntoIter = std::slice::IterMut<'b, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.slice.iter_mut()
    }
}

impl<'a, T, const ALIGN: usize> TryFrom<&'a mut [T]> for AlignedSliceMut<'a, T, ALIGN> {
    type Error = GcsoStatus;

    #[inline]
    fn try_from(slice: &'a mut [T]) -> Result<Self, Self::Error> {
        Self::new(slice)
    }
}

/// Type aliases for common alignment requirements (16-byte DPSR, 32-byte SIMD, 64-byte Cacheline, 128-byte Dual Cacheline).
pub type AlignedSlice16<'a, T> = AlignedSlice<'a, T, 16>;
pub type AlignedSliceMut16<'a, T> = AlignedSliceMut<'a, T, 16>;
pub type AlignedSlice32<'a, T> = AlignedSlice<'a, T, 32>;
pub type AlignedSliceMut32<'a, T> = AlignedSliceMut<'a, T, 32>;
pub type AlignedSlice64<'a, T> = AlignedSlice<'a, T, 64>;
pub type AlignedSliceMut64<'a, T> = AlignedSliceMut<'a, T, 64>;
pub type AlignedSlice128<'a, T> = AlignedSlice<'a, T, 128>;
pub type AlignedSliceMut128<'a, T> = AlignedSliceMut<'a, T, 128>;

/// Helper function to check pointer alignment for a generic custom byte boundary.
#[inline]
#[must_use]
pub fn is_aligned_to<T>(slice: &[T], align: usize) -> bool {
    align != 0
        && align.is_power_of_two()
        && (slice.is_empty() || ((slice.as_ptr() as usize) & (align - 1)) == 0)
}

// ===================================================================
// System Information & Capability Query Trait
// ===================================================================

/// Trait for querying system ABI versions and runtime compute capabilities.
pub trait SystemCapabilityQuery: Send + Sync {
    /// Returns the semantic ABI version tuple `(major, minor, patch)`.
    fn abi_version(&self) -> (u32, u32, u32);

    /// Returns the static ABI version string.
    fn abi_version_string(&self) -> &'static str;

    /// Queries active compute capability bitmask flags.
    fn query_capability(&self) -> gcso_capability_flags_t;

    /// Populates default runtime configuration parameters.
    fn init_default_config(&self, config: &mut gcso_config_t) -> GcsoResult<()>;
}

// ===================================================================
// Unified High-Level Runtime Context Facade Trait
// ===================================================================

/// Unified Hot-Path Runtime Context trait for token-by-token inference loops.
///
/// Implementations MUST guarantee zero dynamic allocations (`malloc`, `Box`, `Vec`)
/// during per-token inference calls (`step_token`).
pub trait GcsoRuntimeContext: Send + Sync {
    /// Resets transient phase accumulators and step counters without freeing tables.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if runtime context state is corrupted.
    fn reset(&mut self) -> GcsoResult<()>;

    /// Registers a natural language system prompt text anchor in phase space.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ACTION_HUB_FULL` if max prompt anchors capacity is reached.
    fn set_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<()>;

    /// Executes a per-token Hot Path inference step in zero-allocation mode.
    ///
    /// Applies QDPS step filtering, RIPA soft-bounded phase steering on Query registers,
    /// and populates the 128-byte pointer trail status in $\mathcal{O}(1)$ time.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if context is uninitialized.
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if tensor buffers violate 32-byte alignment.
    fn step_token(
        &mut self,
        token_id: u32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> GcsoResult<()>;

    /// Serializes active runtime context state into a binary snapshot.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_BUFFER_TOO_SMALL` if output buffer size is insufficient.
    fn serialize(&self, buffer: &mut [u8], required_size: &mut usize) -> GcsoResult<()>;

    /// Deserializes binary snapshot buffer to restore context state in-place.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_CONTAINER_CORRUPTED` if checksum or magic header validation fails.
    fn deserialize(&mut self, buffer: &[u8]) -> GcsoResult<()>;

    /// Seeks to a specific token position in the PPRC keyframe KV cache.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_PPRC_SEEK_FAILED` if `token_index` is out of bounds.
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> GcsoResult<()>;
}

// ===================================================================
// Hot-Path Traits (Zero Dynamic Allocation, Static Dispatch Target)
// ===================================================================

/// Hot-path trait for dynamic phase-steering algorithms on Query tensors.
///
/// # Performance Invariants
/// Implementations MUST NOT perform dynamic heap allocations (`malloc`, `Box`, `Vec`).
/// All methods are designed for static dispatch and MUST maintain $\mathcal{O}(d_{\mathrm{head}})$
/// time complexity per head.
pub trait PhaseSteering: Send + Sync {
    /// Returns the target head dimension ($d_{\mathrm{head}}$).
    fn head_dim(&self) -> u32;

    /// Returns the total number of attention heads.
    fn num_heads(&self) -> u32;

    /// Applies inline dynamic phase steering to a Query tensor in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if buffer lengths or head dimensions are invalid.
    fn apply_phase_steering(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &[gcso_q7_t],
    ) -> GcsoResult<()>;

    /// Applies RIPA soft-bounded phase steering restricted to low-frequency channels.
    ///
    /// Clamps rotation angles strictly on low-frequency head dimensions.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `max_rad` is non-positive or buffers mismatch.
    fn apply_phase_steering_safe(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &[gcso_q7_t],
        max_rad: f32,
    ) -> GcsoResult<()>;

    /// QDPS discrete filter: cuts off phase rotation steps falling below `min_step_rad`.
    ///
    /// Prevents grid jitter and numerical oscillation under ultra-low quantization.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `min_step_rad` is negative or buffer is empty.
    fn qdps_filter_step(
        &self,
        phase_deltas: &mut [gcso_q7_t],
        min_step_rad: f32,
    ) -> GcsoResult<()>;

    /// Lazy Phase Unwrapping: Applies relative phase difference against context accumulator.
    ///
    /// Modulates Query registers without modifying Key-Value caches in VRAM.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if tensor buffers do not match head config.
    fn lazy_unwrap_override(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        context_accum: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<()>;

    /// Executes norm-guarded Slerp phase stabilization on state vectors in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if bounds are invalid.
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if floating-point vector norm is NaN or Infinite.
    fn slerp_norm_guard_stable(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        norm_lower: f32,
        norm_upper: f32,
    ) -> GcsoResult<()>;

    /// Applies fused inline logit phase shift prior to LM Head Softmax.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `logits` or `phase_deltas` are empty.
    fn fused_logit_shift(
        &self,
        logits: &mut [f32],
        phase_deltas: &[gcso_q7_t],
    ) -> GcsoResult<()>;

    /// Computes Procrustes phase delta alignment between source and target state representations.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if dimension mismatch or invalid scale occurs.
    fn compute_procrustes_phase_delta(
        &self,
        source: &AlignedSlice32<'_, f32>,
        target: &AlignedSlice32<'_, f32>,
        phase_out: &mut [gcso_q7_t],
    ) -> GcsoResult<()>;
}

/// Hot-path trait for $\mathcal{O}(1)$ tagged pointer transitions and action hub state updates.
///
/// Implements stigmergic memory traversal over sidecar pointer tables (SPT).
pub trait PointerActionHub: Send + Sync {
    /// Advances a tagged pointer transition in zero-allocation mode.
    ///
    /// Updates `trail_out` in-place without dynamic heap allocation.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if context is uninitialized.
    fn step_pointer(
        &self,
        current_ptr: u64,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<()>;

    /// Computes direct 256-slot hash index for 64-bit pointer trail caching using SplitMix64.
    #[must_use]
    fn hash_slot256_index(&self, ptr: u64) -> u32;

    /// Pulls active pointer trails toward a macro target anchor via pull force $F_{\mathrm{pull}}$.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `pull_force` is NaN or Infinite.
    fn pull_trail_to_attractor(
        &self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> GcsoResult<()>;

    /// Links adjacent cellular hallucinated trails into contiguous trace graphs.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if action hub state is invalid.
    fn link_hallucinated_trails(
        &self,
        src_trail_id: u32,
        dst_trail_id: u32,
    ) -> GcsoResult<()>;

    /// Performs bit-tree reduction across paged bitmasks in $\mathcal{O}(1)$ time.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `bitmasks` is empty.
    fn reduce_bit_tree(
        &self,
        bitmasks: &[gcso_paged_bitmask_t],
        reduced_out: &mut gcso_paged_bitmask_t,
    ) -> GcsoResult<()>;

    /// Evaluates SIMD/Warp bitmask reduction over PagedBlock KV caches.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `kv_bits` is empty.
    fn paged_block_warp_bitmask(
        &self,
        kv_bits: &[u64],
        mask_out: &mut gcso_paged_bitmask_t,
    ) -> GcsoResult<()>;

    /// Returns the maximum slot capacity of the Action Hub table.
    fn capacity(&self) -> u32;

    /// Returns the active pointer trail count currently registered.
    fn active_count(&self) -> u32;
}

/// Hot-path trait for Dynamic Adaptive Extension Scratchpad (DAES) acceleration.
///
/// Operates $\mathcal{O}(1)$ fast-path shortcuts and in-register telemetry push.
pub trait DaesScratchpad: Send + Sync {
    /// Executes fast-path shortcut lookup in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if slot operating mode is not 0.
    fn fast_path_lookup(
        &self,
        slot: &gcso_daes_slot_t,
        input_key: u64,
    ) -> GcsoResult<u64>;

    /// Pushes profiling metric byte code into the DAES telemetry ring ledger in-place.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_NULL_POINTER` if `slot` is uninitialized.
    fn telemetry_push(
        &self,
        slot: &mut gcso_daes_slot_t,
        metric_code: u8,
    ) -> GcsoResult<()>;

    /// Sets the operating mode of the DAES slot (0 = Scratchpad, 1 = Plugin, 2 = Shared IPC Buffer).
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if mode exceeds 2.
    fn set_mode(
        &self,
        slot: &mut gcso_daes_slot_t,
        mode: u32,
    ) -> GcsoResult<()>;

    /// Evaluates telemetry ledger to auto-tune PSPM ratios, RIPA clamps, and EDBC thresholds.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if evaluation fails.
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_daes_slot_t,
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()>;
}

/// Mezzo-path trait for cellular swarm cell chunk step processing across PagedBlocks.
pub trait SwarmCellChunk: Send + Sync {
    /// Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `chunk_len` is 0.
    fn step_chunk(
        &self,
        mask: &gcso_paged_bitmask_t,
        chunk_len: u32,
    ) -> GcsoResult<()>;
}

/// Hot-path trait for Sub-Head Phase Group Allocation (PSPM Router).
///
/// Routes attention heads into Fact, Logic, and Explore sub-groups in a single pass.
pub trait PspmRouter: Send + Sync {
    /// Dispatches PSPM head-group phase profiles in a single forward pass.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `head_dim` is odd or zero.
    fn dispatch_single_pass(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        head_dim: usize,
    ) -> GcsoResult<()>;
}

/// Hot-path trait for Sparse Residual Adapter Layer (SRL) Dynamic Rank-1 evaluations.
///
/// Evaluates $\mathbf{y} = W_{\mathrm{base}}\mathbf{x} + \mathbf{s} \odot (\mathbf{u}(\mathbf{v}^T \mathbf{x}))$.
pub trait SrlAdapter: Send + Sync {
    /// Evaluates SRL Dynamic Rank-1 outer product in zero-allocation mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if dimensions are zero or scalar scale is invalid.
    fn eval_rank1(
        &self,
        y_out: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        descriptor: &gcso_srl_descriptor_t,
    ) -> GcsoResult<()>;
}

// ===================================================================
// Cold-Path Traits (Asynchronous Steering, Memory & Attractor Control)
// ===================================================================

/// Cold-path trait for entropy-driven branch evaluation and bifurcation tracking.
///
/// Evaluates Moving Z-Score Normalized Attention Entropy ($\tilde{H}$) and controls
/// Pitchfork Bifurcation mode transitions.
pub trait EntropyEvaluator: Send + Sync {
    /// Evaluates token Z-score activation entropy and updates dynamic branch mode.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if `token_z_score` contains `NaN` or `Inf`.
    fn eval_stateful(
        &mut self,
        token_z_score: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> GcsoResult<()>;

    /// Computes CVoid Coherent Vector Alignment Metric for Out-of-Distribution Latent Space.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `key_vector` is empty.
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if numerical sum-of-squares is NaN/Inf.
    fn eval_cvoid_dyadic128(
        &self,
        key_vector: &AlignedSlice32<'_, f32>,
        void_score_out: &mut f32,
    ) -> GcsoResult<()>;

    /// Calculates Eyring-Kramers potential barrier height ($\Delta V = \tau_{\mathrm{eff}} / \mathcal{C}_{\mathrm{void}}$).
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_EDBC_SINGULARITY` if numerical instability is detected or $\tau_{\mathrm{eff}} \le 0$.
    fn eval_barrier(&self, void_score: f32, tau_eff: f32) -> GcsoResult<f32>;
}

/// Cold-path trait for macro attractor field steering and repulsion control.
pub trait AttractorField: Send + Sync {
    /// Registers a system prompt anchor in phase space (Primary Baseline Endpoint).
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ACTION_HUB_FULL` if anchor capacity is reached.
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<u32>;

    /// Registers a dense feature embedding vector as a continuous attractor anchor.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ACTION_HUB_FULL` if anchor capacity is reached.
    fn add_embedding_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32>;

    /// Registers a raw topological anchor point in the attractor field.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ACTION_HUB_FULL` if anchor capacity is reached.
    fn add_anchor(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> GcsoResult<u32>;

    /// Injects a phase-conjugate repulsion vector ($-\boldsymbol{\Delta\theta}$) to suppress hallucination.
    ///
    /// Converts false local minima energy valleys into repulsive potential peaks.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `gain` is `NaN` or non-positive.
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &[gcso_q7_t],
        gain: f32,
    ) -> GcsoResult<()>;

    /// Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_STATE` if aggregation state is invalid.
    fn aggregate_bottom_up(&mut self) -> GcsoResult<u32>;
}

/// Cold-path trait for dynamic persona phase modulation patches without altering base weights.
pub trait PersonaPatcher: Send + Sync {
    /// Applies binary persona phase modulation patch to runtime context.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if patch data is empty.
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()>;
}

/// Cold-path trait for Training-Free LoRA-to-Phase SVD Projection (L2P-SVD).
pub trait L2pSvdProjector: Send + Sync {
    /// Projects fine-tuned LoRA matrices via first-order SVD into phase profiles and Rank-1 SRL vectors.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if dimensions, rank, or outputs are invalid.
    fn project_lora(
        &self,
        lora_a: &[f32],
        lora_b: &[f32],
        rank: usize,
        dim_in: usize,
        dim_out: usize,
        srl_out: &mut gcso_srl_descriptor_t,
        phase_profile_out: &mut [gcso_q7_t],
    ) -> GcsoResult<()>;
}

/// Mezzo/Cold-path trait for Predictive Phase-Motion & Residual Compensation (PPRC).
pub trait PprcCache: Send + Sync {
    /// Seeks to a specific token index in the keyframe KV cache without forward passes.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_PPRC_SEEK_FAILED` if `token_index` is out of bounds.
    fn seek_to_token(
        &self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> GcsoResult<()>;
}

/// Cold-path trait for Zero-Overhead In-Memory Mapped Storage (ZIMMS).
pub trait ZimmsStorage: Send + Sync {
    /// Serializes active runtime state into `.gcso` binary snapshot buffer.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_BUFFER_TOO_SMALL` if output buffer size is insufficient.
    fn serialize_snapshot(
        &self,
        buffer: &mut [u8],
        required_size: &mut usize,
    ) -> GcsoResult<()>;

    /// Deserializes binary snapshot buffer to restore context state.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_CONTAINER_CORRUPTED` if checksum or magic numbers fail.
    /// Returns `GCSO_ERROR_VERSION_MISMATCH` if ABI version is incompatible.
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()>;

    /// Opens memory-mapped zero-copy handle for `.gcso` container payload.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_ZIMMS_MAPPING_FAILED` if file path is invalid or mapping fails.
    fn open_mmap(
        &mut self,
        file_path: &str,
        descriptor_out: &mut gcso_zimms_descriptor_t,
    ) -> GcsoResult<()>;

    /// Unmaps zero-copy ZIMMS memory handle and releases DMA resources.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_MISALIGNED_POINTER` if descriptor is invalid.
    fn close_mmap(
        &mut self,
        descriptor: &mut gcso_zimms_descriptor_t,
    ) -> GcsoResult<()>;
}

// ===================================================================
// Unit Tests for Aligned Slice Wrappers
// ===================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_slice_valid() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 64]);
        let buf = AlignedBuffer([0.0; 64]);
        let slice = AlignedSlice32::new(&buf.0).unwrap();
        assert_eq!(slice.len(), 64);
        assert!(!slice.is_empty());
        assert_eq!(slice.align(), 32);
        assert_eq!(slice[0], 0.0);
        assert_eq!(&slice[0..16], &[0.0; 16]);
    }

    #[test]
    fn test_aligned_slice_empty() {
        let empty: &[f32] = &[];
        let slice = AlignedSlice32::new(empty).unwrap();
        assert!(slice.is_empty());
        assert_eq!(slice.len(), 0);
    }

    #[test]
    fn test_aligned_slice_misaligned() {
        #[repr(align(64))]
        struct AlignedBuffer([u8; 64]);
        let buf = AlignedBuffer([0; 64]);
        let unaligned = &buf.0[1..33];
        let res = AlignedSlice32::new(unaligned);
        assert_eq!(res, Err(GCSO_ERROR_MISALIGNED_POINTER));
    }

    #[test]
    fn test_aligned_slice_subslice_and_split() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 64]);
        let mut buf = AlignedBuffer([1.0; 64]);
        let mut slice_mut = AlignedSliceMut64::new(&mut buf.0).unwrap();
        assert_eq!(slice_mut.len(), 64);

        let (left, right) = slice_mut.split_at_mut(32).unwrap();
        assert_eq!(left.len(), 32);
        assert_eq!(right.len(), 32);
    }

    #[test]
    fn test_aligned_slice_copy_from_aligned() {
        #[repr(align(64))]
        struct AlignedBuffer([f32; 32]);
        let src_buf = AlignedBuffer([2.5; 32]);
        let mut dst_buf = AlignedBuffer([0.0; 32]);

        let src_slice = AlignedSlice32::new(&src_buf.0).unwrap();
        let mut dst_slice = AlignedSliceMut32::new(&mut dst_buf.0).unwrap();

        dst_slice.copy_from_aligned_slice(&src_slice).unwrap();
        assert_eq!(dst_slice[0], 2.5);
        assert_eq!(dst_slice[31], 2.5);
    }
}