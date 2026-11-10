// name: src/core/src/traits/primitives.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.0 Foundation 1: Base C-ABI v0.1.1 (`0x00000101`) constants, Plain Old Data (`Pod`)
//! byte-level safety contracts, `.gcso` descriptor/snapshot validators, branchless IEEE 802.3
//! CRC32 integrity primitives, TaggedPtr 5-bit ActionMode & 3-bit AttractorSlot codecs,
//! $8 \times 8$ Attractor Slot coupling matrix helpers, and lifetime-bound FFI handle guards
//! for the Geometric Cellular Sheaf Orchestrator (GCSO).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::cast_possible_truncation)]

use core::f32::consts::{LN_2, PI};
use core::fmt;
use core::marker::PhantomData;
use core::mem::{align_of, size_of, size_of_val};
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};
use core::slice;

use crate::abi::{
    gcso_config_t, gcso_daes_slot_t, gcso_descriptor_header_t, gcso_edbc_state_t,
    gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t, gcso_pspm_config_t,
    gcso_q7_t, gcso_snapshot_header_t, gcso_srl_descriptor_t, gcso_status_t, gcso_tagged_ptr_t,
    gcso_zimms_descriptor_t, GcsoStatus, GCSO_ABI_VERSION_HEX, GCSO_ERROR_BUFFER_TOO_SMALL,
    GCSO_ERROR_CONTAINER_CORRUPTED, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER,
    GCSO_ERROR_NULL_POINTER, GCSO_ERROR_VERSION_MISMATCH,
};

// Re-export canonical strongly-typed opaque handles from `crate::abi`.
pub use crate::abi::{
    GcsoActionHubHandle, GcsoActionHubOpaque, GcsoAttractorFieldHandle, GcsoAttractorFieldOpaque,
    GcsoContainerHandle, GcsoContainerOpaque, GcsoContextHandle, GcsoContextOpaque,
    GcsoDaesSlotHandle, GcsoDaesSlotOpaque, GcsoDpsrKernelHandle, GcsoDpsrKernelOpaque,
    GcsoEdbcControllerHandle, GcsoEdbcControllerOpaque, GcsoPspmRouterHandle, GcsoPspmRouterOpaque,
    GcsoSrlAdapterHandle, GcsoSrlAdapterOpaque,
};

/// Type alias for GCSO core result responses mapping to [`GcsoStatus`].
pub type GcsoResult<T> = Result<T, GcsoStatus>;

// ===================================================================
// 1. Layer 0 Mathematical & Capacity Constants
// ===================================================================

/// Scale factor for Q7 quantized phase representation ($\beta_{\mathrm{Q7}} = \pi / 128$).
pub const Q7_PHASE_SCALE_RAD: f32 = PI / 128.0;

/// Reciprocal scale factor for converting radians to Q7 fixed-point ($128 / \pi$).
pub const Q7_PHASE_INV_SCALE_RAD: f32 = 128.0 / PI;

/// Unit normalized Q7 scale factor ($\beta_{\mathrm{Q7,unit}} = 1 / 128$).
pub const Q7_UNIT_SCALE: f32 = 1.0 / 128.0;

/// Minimum representable Q7 fixed-point value ($-128 \leftrightarrow -\pi\text{ rad}$).
pub const Q7_MIN: gcso_q7_t = -128;

/// Maximum representable Q7 fixed-point value ($+127 \leftrightarrow +\frac{127}{128}\pi\text{ rad}$).
pub const Q7_MAX: gcso_q7_t = 127;

/// Neutral zero phase in Q7 fixed-point format.
pub const Q7_ZERO: gcso_q7_t = 0;

/// Default logarithmic singularity prevention epsilon ($\epsilon_{\mathrm{log}} = 10^{-12}$).
pub const DEFAULT_ENTROPY_EPS: f32 = 1.0e-12;

/// Upper bound of normalized 2-state surrogate attention entropy ($\ln 2 \approx 0.6931472$).
pub const MAX_SURROGATE_ENTROPY_LN2: f32 = LN_2;

/// Default RIPA soft-bounded phase clamp limit in radians ($5^\circ \approx 0.08726646\text{ rad}$).
pub const DEFAULT_RIPA_CLAMP_MAX_RAD: f32 = 0.087_266_46;

/// Default QDPS minimum phase step cutoff in radians ($0.01\text{ rad}$).
pub const DEFAULT_QDPS_MIN_STEP_RAD: f32 = 0.01;

/// Maximum number of attention heads supported in fixed-capacity trail and phase arrays (`64`).
pub const MAX_ATTENTION_HEADS: usize = 64;

/// Default number of tokens per PagedAttention KV cache block (`32`).
pub const DEFAULT_PAGED_BLOCK_SIZE: u32 = 32;

/// Default slot capacity of the Action Hub (Sidecar Pointer Table) (`256`).
pub const DEFAULT_ACTION_HUB_CAPACITY: u32 = 256;

/// Fixed capacity of topological anchors per Macro Attractor Field accumulator (`64`).
pub const MAX_ANCHORS_PER_FIELD: usize = 64;

/// Maximum number of parallel domain accumulator slots in a `MultiDomainAttractorSpace` (`8`).
pub const MAX_ATTRACTOR_DOMAINS: usize = 8;

/// Polynomial degree of the Cold-Path Chebyshev spectral filter (`3`).
pub const CHEBYSHEV_FILTER_DEGREE: usize = 3;

/// Number of expansion coefficients $[c_0, c_1, c_2, c_3]$ in a 3rd-order Chebyshev filter (`4`).
pub const CHEBYSHEV_COEFF_LEN: usize = 4;

/// Number of 64-bit words in a 256-bit `gcso_paged_bitmask_t` (`4`).
pub const PAGED_BITMASK_WORDS: usize = 4;

/// Total number of discrete bits in a `gcso_paged_bitmask_t` (`256`).
pub const PAGED_BITMASK_BITS: u32 = 256;

/// Number of fast-path shortcut slots in a `gcso_daes_slot_t` (`4`).
pub const DAES_SHORTCUT_SLOTS: usize = 4;

/// Byte length of the circular mini telemetry ledger in a `gcso_daes_slot_t` (`8`).
pub const DAES_MINI_LEDGER_LEN: usize = 8;

/// Binary magic constant for `.gcso` snapshot headers (`0x4F534347` = ASCII `"GCSO"`).
pub const GCSO_SNAPSHOT_MAGIC_U32: u32 = 0x4F53_4347;

// ===================================================================
// 2. Attractor Slot IDs & 8x8 Cross-Slot Coupling Matrix Primitives
// ===================================================================

/// Attractor Domain Slot 0: Primary System Baseline & Core Identity Anchor.
pub const ATTRACTOR_SLOT_SYSTEM_BASELINE: u8 = 0;

/// Attractor Domain Slot 1: Character Persona & Emotional Style Attractor.
pub const ATTRACTOR_SLOT_PERSONA: u8 = 1;

/// Attractor Domain Slot 2: Factual Knowledge & World Lore Anchor.
pub const ATTRACTOR_SLOT_KNOWLEDGE: u8 = 2;

/// Attractor Domain Slot 3: Phase-Conjugate Anti-Hallucination Repulsion Field ($-\boldsymbol{\Delta\theta}$).
pub const ATTRACTOR_SLOT_REPULSION: u8 = 3;

/// Attractor Domain Slot 4: Dynamic User Context & Episodic Dialogue Memory.
pub const ATTRACTOR_SLOT_USER_CONTEXT: u8 = 4;

/// Attractor Domain Slot 5: User-Defined Custom Attractor Slot 5 (e.g., Live Visual Co-Watching).
pub const ATTRACTOR_SLOT_CUSTOM_5: u8 = 5;

/// Attractor Domain Slot 6: User-Defined Custom Attractor Slot 6 (e.g., Mood / Circadian Drift).
pub const ATTRACTOR_SLOT_CUSTOM_6: u8 = 6;

/// Attractor Domain Slot 7: User-Defined Custom Attractor Slot 7 (e.g., Workshop Extension Trigger).
pub const ATTRACTOR_SLOT_CUSTOM_7: u8 = 7;

/// Bitmask isolating a valid 3-bit Attractor Domain Slot ID (`0..=7`).
pub const ATTRACTOR_SLOT_ID_MASK: u8 = 0x07;

/// Default $8 \times 8$ Attractor Slot coupling bitmatrix (`u64`) enabling symmetric cross-slot
/// synergy across all 8 parallel attractor domain slots.
pub const DEFAULT_SLOT_COUPLING_MATRIX_8X8: u64 = 0xFFFF_FFFF_FFFF_FFFF;

/// Computes the bit position (`0..=63`) in an $8 \times 8$ `u64` coupling bitmatrix for `(slot_a, slot_b)`.
#[inline(always)]
#[must_use]
pub const fn slot_coupling_bit_index(slot_a: u32, slot_b: u32) -> u32 {
    ((slot_a & 7) << 3) | (slot_b & 7)
}

/// Branchlessly tests whether `slot_a` and `slot_b` are coupled in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn is_slot_pair_coupled(coupling_matrix_8x8: u64, slot_a: u32, slot_b: u32) -> bool {
    ((coupling_matrix_8x8 >> slot_coupling_bit_index(slot_a, slot_b)) & 1) != 0
}

/// Extracts the 8-bit row mask (`bits 0..=7`) representing all target slots coupled with `slot_a`
/// in $\mathcal{O}(1)$ via a single register shift.
#[inline(always)]
#[must_use]
pub const fn coupled_slot_row_mask(coupling_matrix_8x8: u64, slot_a: u32) -> u8 {
    ((coupling_matrix_8x8 >> ((slot_a & 7) << 3)) & 0xFF) as u8
}

/// Returns the number of target slots (`0..=8`) coupled with `slot_a` in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn coupled_slot_count(coupling_matrix_8x8: u64, slot_a: u32) -> u32 {
    coupled_slot_row_mask(coupling_matrix_8x8, slot_a).count_ones()
}

/// Branchlessly sets or clears the directed coupling bit `(slot_a, slot_b)` in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn with_slot_pair_coupled(
    coupling_matrix_8x8: u64,
    slot_a: u32,
    slot_b: u32,
    enabled: bool,
) -> u64 {
    let bit_mask = 1u64 << slot_coupling_bit_index(slot_a, slot_b);
    let enable_mask = (enabled as u64).wrapping_neg();
    (coupling_matrix_8x8 & !bit_mask) | (bit_mask & enable_mask)
}

/// Branchlessly sets or clears symmetric bidirectional coupling between `slot_a` and `slot_b`
/// (`(slot_a, slot_b)` and `(slot_b, slot_a)`) in `coupling_matrix_8x8`.
#[inline(always)]
#[must_use]
pub const fn with_slot_pair_coupled_symmetric(
    coupling_matrix_8x8: u64,
    slot_a: u32,
    slot_b: u32,
    enabled: bool,
) -> u64 {
    let m1 = with_slot_pair_coupled(coupling_matrix_8x8, slot_a, slot_b, enabled);
    with_slot_pair_coupled(m1, slot_b, slot_a, enabled)
}

// ===================================================================
// 3. Pointer-Driven Action Mode & Tag Bitfield Codecs
// ===================================================================

/// Action Mode Bit 0 (`0x01`): Enable inline DPSR phase rotation on Query/Key registers.
pub const ACTION_MODE_DPSR_STEER: u8 = 1 << 0;

/// Action Mode Bit 1 (`0x02`): Enable intermediate layer skipping with SWAR phase compensation.
pub const ACTION_MODE_LAYER_SKIP: u8 = 1 << 1;

/// Action Mode Bit 2 (`0x04`): Enable Contrastive Phase Bloom Filter anti-phase repulsion ($-\boldsymbol{\Delta\theta}$).
pub const ACTION_MODE_PHASE_REPULSE: u8 = 1 << 2;

/// Action Mode Bit 3 (`0x08`): Enable single-pass PSPM sub-head group routing (Fact / Logic / Explore).
pub const ACTION_MODE_PSPM_ROUTE: u8 = 1 << 3;

/// Action Mode Bit 4 (`0x10`): Enable SRL Dynamic Rank-1 outer product residual injection ($\mathbf{u}\mathbf{v}^T$).
pub const ACTION_MODE_SRL_INJECT: u8 = 1 << 4;

/// Composite Action Mode preset for passive sidecar observation (`0x00`).
pub const ACTION_PRESET_PASSIVE_SIDECAR: u8 = 0x00;

/// Composite Action Mode preset for lightweight express phase-steered sidecar execution (`0x01`).
pub const ACTION_PRESET_EXPRESS_PHASE: u8 = ACTION_MODE_DPSR_STEER;

/// Composite Action Mode preset for low-entropy inertial layer-skipping fast-forward (`0x03`).
pub const ACTION_PRESET_INERTIAL_SKIP: u8 = ACTION_MODE_DPSR_STEER | ACTION_MODE_LAYER_SKIP;

/// Composite Action Mode preset for anti-hallucination repulsion and PSPM exploration routing (`0x0D`).
pub const ACTION_PRESET_REPULSION_ESCAPE: u8 =
    ACTION_MODE_DPSR_STEER | ACTION_MODE_PHASE_REPULSE | ACTION_MODE_PSPM_ROUTE;

/// Composite Action Mode preset for full-stack deep persona & knowledge orchestration (`0x19`).
pub const ACTION_PRESET_FULL_ORCHESTRATION: u8 =
    ACTION_MODE_DPSR_STEER | ACTION_MODE_PSPM_ROUTE | ACTION_MODE_SRL_INJECT;

/// Bitmask extracting the 5-bit Action Mode flags from an 8-bit `gcso_tagged_ptr_t` tag (`bits 0..=4`).
pub const TAG_ACTION_MODE_MASK: u8 = 0x1F;

/// Bit shift offset for encoding/extracting the 3-bit Target Attractor Slot ID (`0..=7`) in an 8-bit tag (`bits 5..=7`).
pub const TAG_ATTRACTOR_SLOT_SHIFT: u32 = 5;

/// Packs a 3-bit Target Attractor Slot ID (`0..=7`) and a 5-bit Action Mode mask (`0x00..=0x1F`)
/// into a single 8-bit tag byte for [`gcso_tagged_ptr_t`].
#[inline(always)]
#[must_use]
pub const fn pack_action_slot_tag(attractor_slot_id: u8, action_mode_flags: u8) -> u8 {
    ((attractor_slot_id & ATTRACTOR_SLOT_ID_MASK) << TAG_ATTRACTOR_SLOT_SHIFT)
        | (action_mode_flags & TAG_ACTION_MODE_MASK)
}

/// Extracts the 5-bit Action Mode bitmask (`ACTION_MODE_*`) from an 8-bit [`gcso_tagged_ptr_t`] tag byte.
#[inline(always)]
#[must_use]
pub const fn unpack_action_mode_from_tag(tag: u8) -> u8 {
    tag & TAG_ACTION_MODE_MASK
}

/// Extracts the 3-bit Target Attractor Slot ID (`0..=7`) from an 8-bit [`gcso_tagged_ptr_t`] tag byte.
#[inline(always)]
#[must_use]
pub const fn unpack_attractor_slot_from_tag(tag: u8) -> u8 {
    (tag >> TAG_ATTRACTOR_SLOT_SHIFT) & ATTRACTOR_SLOT_ID_MASK
}

// ===================================================================
// 4. Plain Old Data (Pod) Safety Contract & Zero-Copy Byte/Slice Casting
// ===================================================================

/// Compile-time helper verifying that a requested alignment `ALIGN` is a non-zero power of two
/// and at least `align_of::<T>()`.
#[allow(dead_code)]
struct PtrAlignCheck<T, const ALIGN: usize>(PhantomData<T>);
impl<T, const ALIGN: usize> PtrAlignCheck<T, ALIGN> {
    const VALID: () = assert!(
        ALIGN > 0 && ALIGN.is_power_of_two() && ALIGN >= align_of::<T>(),
        "Requested ALIGN must be a non-zero power of two and >= align_of::<T>()"
    );
}

/// Branchlessly checks whether two half-open byte intervals
/// `[a_start, a_start + a_byte_len)` and `[b_start, b_start + b_byte_len)` overlap in virtual memory.
#[inline(always)]
#[must_use]
pub const fn raw_byte_ranges_overlap(
    a_start: usize,
    a_byte_len: usize,
    b_start: usize,
    b_byte_len: usize,
) -> bool {
    let non_empty = (a_byte_len > 0) & (b_byte_len > 0);
    let a_end = a_start.saturating_add(a_byte_len);
    let b_end = b_start.saturating_add(b_byte_len);
    non_empty & (a_start < b_end) & (b_start < a_end)
}

#[inline(always)]
#[must_use]
const fn raw_regions_overlap_or_same_zst(
    a_start: usize,
    a_size: usize,
    b_start: usize,
    b_size: usize,
) -> bool {
    let zst_same_addr = ((a_size == 0) | (b_size == 0)) & (a_start == b_start);
    raw_byte_ranges_overlap(a_start, a_size, b_start, b_size) | zst_same_addr
}

/// Marker trait for Plain Old Data (`Pod`) types that contain no uninitialized padding traps,
/// can be safely zero-initialized, viewed as raw byte slices, and cast across aligned SIMD
/// or ZIMMS memory-mapped boundaries.
///
/// # Safety
/// Implementing types must be `#[repr(C)]`, `#[repr(transparent)]`, or primitive numeric types,
/// have no invalid bit patterns (all-zero bytes and arbitrary byte patterns must represent valid
/// instances), and contain no interior mutability, pointers with provenance invariants, or drop glue.
pub unsafe trait Pod: Copy + Sized + Send + Sync + 'static {
    /// Size of `Self` in bytes (`core::mem::size_of::<Self>()`).
    const BYTE_SIZE: usize = size_of::<Self>();

    /// Required alignment of `Self` in bytes (`core::mem::align_of::<Self>()`).
    const ALIGNMENT: usize = align_of::<Self>();

    /// Compile-time flag indicating whether `Self` is a Zero-Sized Type (`BYTE_SIZE == 0`).
    const IS_ZST: bool = size_of::<Self>() == 0;

    /// Returns a zero-initialized instance of `Self`.
    #[inline(always)]
    #[must_use]
    fn zeroed() -> Self {
        // SAFETY: All-zero bit patterns are guaranteed valid by the `Pod` safety contract.
        unsafe { core::mem::zeroed() }
    }

    /// Views `self` as an immutable byte slice of length `size_of::<Self>()`.
    #[inline(always)]
    #[must_use]
    fn as_bytes(&self) -> &[u8] {
        // SAFETY: `Self` is `Pod` with no uninitialized padding traps, and `self` is a valid reference.
        unsafe { slice::from_raw_parts((self as *const Self).cast::<u8>(), size_of::<Self>()) }
    }

    /// Views `self` as a mutable byte slice of length `size_of::<Self>()`.
    #[inline(always)]
    #[must_use]
    fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: `Self` is `Pod` and uniquely borrowed via `&mut self`.
        unsafe { slice::from_raw_parts_mut((self as *mut Self).cast::<u8>(), size_of::<Self>()) }
    }

    /// Views a contiguous slice of `Self` as an immutable byte slice in zero-allocation mode.
    #[inline(always)]
    #[must_use]
    fn slice_as_bytes(slice: &[Self]) -> &[u8] {
        if Self::IS_ZST || slice.is_empty() {
            return &[];
        }
        // SAFETY: `Self` is `Pod` with no uninitialized padding bytes across contiguous array layouts.
        unsafe { slice::from_raw_parts(slice.as_ptr().cast::<u8>(), size_of_val(slice)) }
    }

    /// Views a contiguous mutable slice of `Self` as a mutable byte slice in zero-allocation mode.
    #[inline(always)]
    #[must_use]
    fn slice_as_bytes_mut(slice: &mut [Self]) -> &mut [u8] {
        if Self::IS_ZST || slice.is_empty() {
            return &mut [];
        }
        let byte_len = size_of_val(slice);
        // SAFETY: `Self` is `Pod` and `slice` is uniquely borrowed.
        unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<u8>(), byte_len) }
    }

    /// Zero-copy casts an aligned byte slice at `offset` into an immutable reference `&Self`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes.len() < offset + size_of::<Self>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the byte address at `offset` violates `align_of::<Self>()`.
    #[inline(always)]
    fn from_bytes_at(bytes: &[u8], offset: usize) -> GcsoResult<&Self> {
        let end = offset
            .checked_add(size_of::<Self>())
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        // SAFETY: `offset <= bytes.len()` verified above.
        let ptr = unsafe { bytes.as_ptr().add(offset) };
        if ((ptr as usize) & (align_of::<Self>() - 1)) != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Bounds and alignment for `Self: Pod` are verified above.
        Ok(unsafe { &*ptr.cast::<Self>() })
    }

    /// Zero-copy casts an aligned mutable byte slice at `offset` into a mutable reference `&mut Self`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes.len() < offset + size_of::<Self>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the byte address at `offset` violates `align_of::<Self>()`.
    #[inline(always)]
    fn from_bytes_mut_at(bytes: &mut [u8], offset: usize) -> GcsoResult<&mut Self> {
        let end = offset
            .checked_add(size_of::<Self>())
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        // SAFETY: `offset <= bytes.len()` verified above.
        let ptr = unsafe { bytes.as_mut_ptr().add(offset) };
        if ((ptr as usize) & (align_of::<Self>() - 1)) != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Bounds, unique borrow, and alignment for `Self: Pod` are verified above.
        Ok(unsafe { &mut *ptr.cast::<Self>() })
    }

    /// Zero-copy casts an aligned byte region starting at `offset` into a slice `&[Self]` of `count` elements.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `Self` is zero-sized.
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes` cannot hold `count * size_of::<Self>()` bytes from `offset`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the starting address violates `align_of::<Self>()`.
    #[inline(always)]
    fn slice_from_bytes_at(bytes: &[u8], offset: usize, count: usize) -> GcsoResult<&[Self]> {
        if Self::IS_ZST {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let byte_len = count
            .checked_mul(size_of::<Self>())
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        let end = offset
            .checked_add(byte_len)
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if count == 0 {
            return Ok(&[]);
        }
        // SAFETY: `offset + byte_len <= bytes.len()` verified above.
        let ptr = unsafe { bytes.as_ptr().add(offset) };
        if ((ptr as usize) & (align_of::<Self>() - 1)) != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Bounds and alignment for `count` elements of `Self: Pod` are verified above.
        Ok(unsafe { slice::from_raw_parts(ptr.cast::<Self>(), count) })
    }

    /// Zero-copy casts an aligned mutable byte region starting at `offset` into `&mut [Self]` of `count` elements.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `Self` is zero-sized.
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes` cannot hold `count * size_of::<Self>()` bytes from `offset`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the starting address violates `align_of::<Self>()`.
    #[inline(always)]
    fn slice_from_bytes_mut_at(
        bytes: &mut [u8],
        offset: usize,
        count: usize,
    ) -> GcsoResult<&mut [Self]> {
        if Self::IS_ZST {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let byte_len = count
            .checked_mul(size_of::<Self>())
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        let end = offset
            .checked_add(byte_len)
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if count == 0 {
            return Ok(&mut []);
        }
        // SAFETY: `offset + byte_len <= bytes.len()` verified above.
        let ptr = unsafe { bytes.as_mut_ptr().add(offset) };
        if ((ptr as usize) & (align_of::<Self>() - 1)) != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Bounds, unique borrow, and alignment for `count` elements of `Self: Pod` verified above.
        Ok(unsafe { slice::from_raw_parts_mut(ptr.cast::<Self>(), count) })
    }

    /// Safely reads an instance of `Self` from a potentially unaligned byte slice at `offset`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes.len() < offset + size_of::<Self>()`.
    #[inline(always)]
    fn read_unaligned_from(bytes: &[u8], offset: usize) -> GcsoResult<Self> {
        let end = offset
            .checked_add(size_of::<Self>())
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        // SAFETY: Bounds checked above; `Self` is `Pod` so any bit pattern is valid via `read_unaligned`.
        Ok(unsafe { ptr::read_unaligned(bytes.as_ptr().add(offset).cast::<Self>()) })
    }

    /// Safely writes `self` into a potentially unaligned byte slice at `offset`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes.len() < offset + size_of::<Self>()`.
    #[inline(always)]
    fn write_unaligned_to(&self, bytes: &mut [u8], offset: usize) -> GcsoResult<()> {
        let end = offset
            .checked_add(size_of::<Self>())
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        // SAFETY: Bounds checked above; `Self` is `Pod` with no padding traps.
        unsafe {
            ptr::write_unaligned(bytes.as_mut_ptr().add(offset).cast::<Self>(), *self);
        }
        Ok(())
    }

    /// Bulk-copies `dst.len()` elements of `Self` from a potentially unaligned byte buffer at `offset`
    /// using a single `ptr::copy_nonoverlapping` (`memcpy`) operation.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes` has fewer than `offset + size_of_val(dst)` bytes.
    #[inline(always)]
    fn read_unaligned_slice_from(bytes: &[u8], offset: usize, dst: &mut [Self]) -> GcsoResult<()> {
        let byte_len = size_of_val(dst);
        let end = offset
            .checked_add(byte_len)
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if byte_len > 0 {
            // SAFETY: `bytes[offset..end]` and `dst` are valid, non-overlapping regions of `byte_len` bytes,
            // and `u8` has alignment 1 so unaligned source reads into `dst` are sound for `Self: Pod`.
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr().add(offset),
                    dst.as_mut_ptr().cast::<u8>(),
                    byte_len,
                );
            }
        }
        Ok(())
    }

    /// Bulk-writes `src` into a potentially unaligned byte buffer at `offset` using a single
    /// `ptr::copy_nonoverlapping` (`memcpy`) operation.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes` has fewer than `offset + size_of_val(src)` bytes.
    #[inline(always)]
    fn write_unaligned_slice_to(src: &[Self], bytes: &mut [u8], offset: usize) -> GcsoResult<()> {
        let byte_len = size_of_val(src);
        let end = offset
            .checked_add(byte_len)
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if byte_len > 0 {
            // SAFETY: `src` and `bytes[offset..end]` are valid, non-overlapping regions of `byte_len` bytes,
            // and `Self: Pod` contains no uninitialized padding bytes.
            unsafe {
                ptr::copy_nonoverlapping(
                    src.as_ptr().cast::<u8>(),
                    bytes.as_mut_ptr().add(offset),
                    byte_len,
                );
            }
        }
        Ok(())
    }

    /// Safely reinterprets a slice of `Self` as a slice of another `Pod` type `U`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slice base pointer violates `align_of::<U>()`.
    #[inline(always)]
    fn try_cast_slice<U: Pod>(slice: &[Self]) -> GcsoResult<&[U]> {
        let total_bytes = size_of_val(slice);
        let target_size = size_of::<U>();
        if (target_size == 0) | ((total_bytes % target_size) != 0) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let is_aligned = slice.is_empty() | (((slice.as_ptr() as usize) & (align_of::<U>() - 1)) == 0);
        if !is_aligned {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Both `Self` and `U` are `Pod`, and alignment/byte-size invariants are verified above.
        Ok(unsafe { slice::from_raw_parts(slice.as_ptr().cast::<U>(), total_bytes / target_size) })
    }

    /// Safely reinterprets a mutable slice of `Self` as a mutable slice of another `Pod` type `U`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `U` is zero-sized or total byte length is not a multiple of `size_of::<U>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the slice base pointer violates `align_of::<U>()`.
    #[inline(always)]
    fn try_cast_slice_mut<U: Pod>(slice: &mut [Self]) -> GcsoResult<&mut [U]> {
        let total_bytes = size_of_val(slice);
        let target_size = size_of::<U>();
        if (target_size == 0) | ((total_bytes % target_size) != 0) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let is_aligned = slice.is_empty() | (((slice.as_ptr() as usize) & (align_of::<U>() - 1)) == 0);
        if !is_aligned {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: Both `Self` and `U` are `Pod`, and alignment/byte-size invariants are verified above.
        Ok(unsafe {
            slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<U>(), total_bytes / target_size)
        })
    }
}

macro_rules! impl_pod_for_primitives {
    ($($ty:ty),* $(,)?) => {$(
        // SAFETY: Primitive numeric and C-ABI POD structures have well-defined bit layouts
        // where all-zero bit patterns are valid and no drop glue or hidden padding exists.
        unsafe impl Pod for $ty {}
    )*};
}

impl_pod_for_primitives!(
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    u128,
    i128,
    usize,
    isize,
    f32,
    f64,
    gcso_status_t,
    gcso_tagged_ptr_t,
    gcso_descriptor_header_t,
    gcso_paged_bitmask_t,
    gcso_pointer_trail_t,
    gcso_daes_slot_t,
    gcso_config_t,
    gcso_edbc_state_t,
    gcso_zimms_descriptor_t,
    gcso_pspm_config_t,
    gcso_srl_descriptor_t,
    gcso_snapshot_header_t,
    gcso_pprc_keyframe_header_t,
);

// SAFETY: Fixed-size arrays of `Pod` elements have contiguous layout with no extra padding traps.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

// ===================================================================
// 5. Descriptor & .gcso Snapshot Validators + Branchless IEEE 802.3 CRC32
// ===================================================================

/// Constructs a canonical `gcso_descriptor_header_t` for type `T` stamped with ABI version `0.1.1` (`0x00000101`).
#[inline(always)]
#[must_use]
pub const fn init_descriptor_header<T>() -> gcso_descriptor_header_t {
    gcso_descriptor_header_t {
        struct_size: size_of::<T>() as u32,
        abi_version: GCSO_ABI_VERSION_HEX,
    }
}

/// Validates a `gcso_descriptor_header_t` against expected structure byte size and unified ABI version `0.1.1` (`0x00000101`).
///
/// # Errors
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `struct_size` or `abi_version` does not match.
#[inline(always)]
pub const fn validate_descriptor_header<T>(header: &gcso_descriptor_header_t) -> GcsoResult<()> {
    let valid = (header.struct_size as usize == size_of::<T>())
        & (header.abi_version == GCSO_ABI_VERSION_HEX);
    if valid {
        Ok(())
    } else {
        Err(GCSO_ERROR_VERSION_MISMATCH)
    }
}

/// Incrementally updates an IEEE 802.3 CRC32 checksum (`0xEDB88320` reflected polynomial) over `bytes`
/// starting from a previous finalized CRC value `crc_seed` (`0` for initial chunk) using the
/// 64B-aligned 256-entry lookup table (`CRC32_IEEE_LUT_256`) with 8-byte unrolled loop execution.
#[inline(always)]
#[must_use]
pub const fn update_crc32_ieee(crc_seed: u32, bytes: &[u8]) -> u32 {
    super::branchless::branchless_update_crc32_ieee(crc_seed, bytes)
}

/// Computes a standard IEEE 802.3 CRC32 checksum (`0xEDB88320` reflected polynomial) over `bytes`
/// using the 64B-aligned 256-entry lookup table (`CRC32_IEEE_LUT_256`) in `#![no_std]` `const fn` mode.
#[inline(always)]
#[must_use]
pub const fn compute_crc32_ieee(bytes: &[u8]) -> u32 {
    super::branchless::branchless_update_crc32_ieee(0, bytes)
}

/// Constructs a canonical `gcso_snapshot_header_t` initialized with magic `"GCSO"` (`0x4F534347`),
/// ABI version `0.1.1` (`0x00000101`), and `total_size`.
#[inline(always)]
#[must_use]
pub const fn init_snapshot_header(
    total_size: u64,
    timestamp_epoch_sec: u64,
) -> gcso_snapshot_header_t {
    let hdr_size = size_of::<gcso_snapshot_header_t>() as u64;
    gcso_snapshot_header_t {
        magic: GCSO_SNAPSHOT_MAGIC_U32,
        version: GCSO_ABI_VERSION_HEX,
        total_size,
        action_hub_offset: hdr_size,
        attractor_field_offset: hdr_size,
        dpsr_state_offset: hdr_size,
        srl_state_offset: hdr_size,
        edbc_state_offset: hdr_size,
        checksum_crc32: 0,
        daes_slot_offset: hdr_size as u32,
        timestamp_epoch_sec,
        reserved_padding: [0; 56],
    }
}

/// Validates a `gcso_snapshot_header_t` against the `.gcso` container magic (`0x4F534347`),
/// unified ABI version (`0x00000101`), and payload buffer bounds.
///
/// # Errors
/// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `magic` is invalid, `total_size` is smaller than
/// `size_of::<gcso_snapshot_header_t>()`, `available_bytes < total_size`, or any track offset exceeds `total_size`.
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `version != GCSO_ABI_VERSION_HEX`.
#[inline]
pub const fn validate_snapshot_header(
    header: &gcso_snapshot_header_t,
    available_bytes: usize,
) -> GcsoResult<()> {
    if header.magic != GCSO_SNAPSHOT_MAGIC_U32 {
        return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
    }
    if header.version != GCSO_ABI_VERSION_HEX {
        return Err(GCSO_ERROR_VERSION_MISMATCH);
    }
    let min_hdr = size_of::<gcso_snapshot_header_t>() as u64;
    let total = header.total_size;
    let daes_off = header.daes_slot_offset as u64;
    let offsets_in_bounds = (total >= min_hdr)
        & ((available_bytes as u64) >= total)
        & (header.action_hub_offset >= min_hdr)
        & (header.action_hub_offset <= total)
        & (header.attractor_field_offset >= min_hdr)
        & (header.attractor_field_offset <= total)
        & (header.dpsr_state_offset >= min_hdr)
        & (header.dpsr_state_offset <= total)
        & (header.srl_state_offset >= min_hdr)
        & (header.srl_state_offset <= total)
        & (header.edbc_state_offset >= min_hdr)
        & (header.edbc_state_offset <= total)
        & (daes_off >= min_hdr)
        & (daes_off <= total);
    if offsets_in_bounds {
        Ok(())
    } else {
        Err(GCSO_ERROR_CONTAINER_CORRUPTED)
    }
}

/// Validates a `.gcso` snapshot header, computes the IEEE 802.3 CRC32 checksum over the payload
/// bytes following the 128-byte header (`container_bytes[128..total_size]`), and stamps
/// `header.checksum_crc32` in-place.
///
/// # Errors
/// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if header offsets or payload bounds are invalid.
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `header.version != GCSO_ABI_VERSION_HEX`.
#[inline]
pub fn finalize_snapshot_header_crc32(
    header: &mut gcso_snapshot_header_t,
    container_bytes: &[u8],
) -> GcsoResult<u32> {
    validate_snapshot_header(header, container_bytes.len())?;
    let hdr_size = size_of::<gcso_snapshot_header_t>();
    let crc = compute_crc32_ieee(&container_bytes[hdr_size..(header.total_size as usize)]);
    header.checksum_crc32 = crc;
    Ok(crc)
}

/// Validates a `.gcso` snapshot header and verifies the IEEE 802.3 CRC32 checksum over the payload
/// bytes following the 128-byte header (`container_bytes[128..total_size]`).
///
/// # Errors
/// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if header bounds or CRC32 verification fail.
/// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `header.version != GCSO_ABI_VERSION_HEX`.
#[inline]
pub fn verify_snapshot_payload_crc32(
    header: &gcso_snapshot_header_t,
    container_bytes: &[u8],
) -> GcsoResult<()> {
    validate_snapshot_header(header, container_bytes.len())?;
    if header.checksum_crc32 != 0 {
        let hdr_size = size_of::<gcso_snapshot_header_t>();
        let computed = compute_crc32_ieee(&container_bytes[hdr_size..(header.total_size as usize)]);
        if computed != header.checksum_crc32 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
    }
    Ok(())
}

// ===================================================================
// 6. Type-Safe Lifetime-Bound FFI Handle Guards
// ===================================================================

#[inline(always)]
fn validate_non_null_aligned_ptr<T>(raw_ptr: *mut T, align_mask: usize) -> GcsoResult<NonNull<T>> {
    if raw_ptr.is_null() {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    if ((raw_ptr as usize) & align_mask) != 0 {
        return Err(GCSO_ERROR_MISALIGNED_POINTER);
    }
    // SAFETY: Nullness checked above.
    Ok(unsafe { NonNull::new_unchecked(raw_ptr) })
}

/// Lifetime-bound immutable FFI opaque handle wrapper guaranteeing non-null and aligned pointer invariants.
#[repr(transparent)]
pub struct FfiHandleRef<'a, T> {
    ptr: NonNull<T>,
    _marker: PhantomData<&'a T>,
}

// SAFETY: `FfiHandleRef<'a, T>` is semantically equivalent to `&'a T`.
unsafe impl<'a, T: Sync> Send for FfiHandleRef<'a, T> {}
// SAFETY: Shared immutable access across threads is sound when `T: Sync`.
unsafe impl<'a, T: Sync> Sync for FfiHandleRef<'a, T> {}

impl<'a, T> Copy for FfiHandleRef<'a, T> {}

impl<'a, T> Clone for FfiHandleRef<'a, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> fmt::Debug for FfiHandleRef<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("FfiHandleRef").field(&self.ptr).finish()
    }
}

impl<'a, T> PartialEq for FfiHandleRef<'a, T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<'a, T> Eq for FfiHandleRef<'a, T> {}

impl<'a, T> FfiHandleRef<'a, T> {
    /// Validates a raw pointer from C-ABI and wraps it in a non-null, naturally aligned `FfiHandleRef`.
    ///
    /// # Safety
    /// Caller must ensure `raw_ptr` points to a valid, initialized instance of `T` that is not
    /// mutated for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw(raw_ptr: *const T) -> GcsoResult<Self> {
        let ptr = validate_non_null_aligned_ptr(raw_ptr.cast_mut(), align_of::<T>() - 1)?;
        Ok(Self {
            ptr,
            _marker: PhantomData,
        })
    }

    /// Validates a raw pointer from C-ABI against an explicit power-of-two hardware alignment `ALIGN`
    /// (`ALIGN >= align_of::<T>()`).
    ///
    /// # Safety
    /// Caller must ensure `raw_ptr` points to a valid, initialized instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub unsafe fn from_raw_aligned<const ALIGN: usize>(raw_ptr: *const T) -> GcsoResult<Self> {
        let () = PtrAlignCheck::<T, ALIGN>::VALID;
        let ptr = validate_non_null_aligned_ptr(raw_ptr.cast_mut(), ALIGN - 1)?;
        Ok(Self {
            ptr,
            _marker: PhantomData,
        })
    }

    /// Validates an optional raw pointer from C-ABI: returns `Ok(None)` if `raw_ptr` is null,
    /// or `Ok(Some(FfiHandleRef))` if non-null and properly aligned.
    ///
    /// # Safety
    /// If non-null, caller must ensure `raw_ptr` points to a valid instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` is non-null and violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw_opt(raw_ptr: *const T) -> GcsoResult<Option<Self>> {
        if raw_ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: Delegated to `from_raw` after non-null check.
            unsafe { Self::from_raw(raw_ptr).map(Some) }
        }
    }

    /// Enforces `GCSO_RESTRICT` non-aliasing invariant against a mutable `FfiHandleMut`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two handles overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint<U>(&self, other: &FfiHandleMut<'_, U>) -> GcsoResult<()> {
        other.ensure_disjoint_ref(self)
    }

    /// Safely reinterprets `self` as a reference handle to another `Pod` type `U` of equal or smaller size.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `size_of::<U>() == 0` or `size_of::<U>() > size_of::<T>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the pointer violates `align_of::<U>()`.
    #[inline(always)]
    pub fn try_cast<U: Pod>(&self) -> GcsoResult<FfiHandleRef<'a, U>>
    where
        T: Pod,
    {
        if U::IS_ZST || size_of::<U>() > size_of::<T>() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Both `T` and `U` are `Pod`, `size_of::<U>() <= size_of::<T>()`, and `from_raw` checks `align_of::<U>()`.
        unsafe { FfiHandleRef::<U>::from_raw(self.ptr.as_ptr().cast::<U>()) }
    }

    /// Returns the underlying `NonNull<T>` pointer.
    #[inline(always)]
    #[must_use]
    pub const fn as_non_null(&self) -> NonNull<T> {
        self.ptr
    }

    /// Returns the underlying raw immutable pointer `*const T`.
    #[inline(always)]
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Dereferences the validated handle into an immutable reference `&'a T` with lifetime `'a`.
    #[inline(always)]
    #[must_use]
    pub const fn as_ref(&self) -> &'a T {
        // SAFETY: Validity and alignment guaranteed at construction.
        unsafe { &*self.ptr.as_ptr() }
    }

    /// Copies the underlying value out of the validated handle when `T: Copy`.
    #[inline(always)]
    #[must_use]
    pub const fn read_copy(&self) -> T
    where
        T: Copy,
    {
        // SAFETY: Validity and alignment guaranteed at construction.
        unsafe { *self.ptr.as_ptr() }
    }
}

impl<'a, T> Deref for FfiHandleRef<'a, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, T> AsRef<T> for FfiHandleRef<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        Self::as_ref(self)
    }
}

/// Lifetime-bound mutable FFI opaque handle wrapper guaranteeing non-null, aligned, and exclusive pointer invariants.
#[repr(transparent)]
pub struct FfiHandleMut<'a, T> {
    ptr: NonNull<T>,
    _marker: PhantomData<&'a mut T>,
}

// SAFETY: `FfiHandleMut<'a, T>` is semantically equivalent to `&'a mut T`.
unsafe impl<'a, T: Send> Send for FfiHandleMut<'a, T> {}
// SAFETY: Shared references to `FfiHandleMut` only allow immutable access when `T: Sync`.
unsafe impl<'a, T: Sync> Sync for FfiHandleMut<'a, T> {}

impl<'a, T> fmt::Debug for FfiHandleMut<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("FfiHandleMut").field(&self.ptr).finish()
    }
}

impl<'a, T> PartialEq for FfiHandleMut<'a, T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<'a, T> Eq for FfiHandleMut<'a, T> {}

impl<'a, T> FfiHandleMut<'a, T> {
    /// Validates a raw mutable pointer from C-ABI and wraps it in a non-null, naturally aligned `FfiHandleMut`.
    ///
    /// # Safety
    /// Caller must ensure `raw_ptr` points to a valid, uniquely borrowed instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw(raw_ptr: *mut T) -> GcsoResult<Self> {
        let ptr = validate_non_null_aligned_ptr(raw_ptr, align_of::<T>() - 1)?;
        Ok(Self {
            ptr,
            _marker: PhantomData,
        })
    }

    /// Validates a raw mutable pointer from C-ABI against an explicit power-of-two hardware alignment `ALIGN`
    /// (`ALIGN >= align_of::<T>()`).
    ///
    /// # Safety
    /// Caller must ensure `raw_ptr` points to a valid, uniquely borrowed instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `raw_ptr` is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` is not `ALIGN`-byte aligned.
    #[inline(always)]
    pub unsafe fn from_raw_aligned<const ALIGN: usize>(raw_ptr: *mut T) -> GcsoResult<Self> {
        let () = PtrAlignCheck::<T, ALIGN>::VALID;
        let ptr = validate_non_null_aligned_ptr(raw_ptr, ALIGN - 1)?;
        Ok(Self {
            ptr,
            _marker: PhantomData,
        })
    }

    /// Validates an optional raw mutable pointer from C-ABI: returns `Ok(None)` if `raw_ptr` is null,
    /// or `Ok(Some(FfiHandleMut))` if non-null and properly aligned.
    ///
    /// # Safety
    /// If non-null, caller must ensure `raw_ptr` points to a valid, uniquely borrowed instance of `T` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `raw_ptr` is non-null and violates `align_of::<T>()`.
    #[inline(always)]
    pub unsafe fn from_raw_opt(raw_ptr: *mut T) -> GcsoResult<Option<Self>> {
        if raw_ptr.is_null() {
            Ok(None)
        } else {
            // SAFETY: Delegated to `from_raw` after non-null check.
            unsafe { Self::from_raw(raw_ptr).map(Some) }
        }
    }

    /// Validates two mutable FFI pointers and enforces `GCSO_RESTRICT` non-overlapping memory ranges.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a` and `ptr_b` point to valid initialized instances of `T` and `U` for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if either pointer is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if either pointer is misaligned.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if their memory footprints overlap.
    #[inline(always)]
    pub unsafe fn from_raw_disjoint_pair<U>(
        ptr_a: *mut T,
        ptr_b: *mut U,
    ) -> GcsoResult<(Self, FfiHandleMut<'a, U>)> {
        // SAFETY: Validated for non-nullness, alignment, and pairwise disjointness before returning.
        let handle_a = unsafe { Self::from_raw(ptr_a)? };
        // SAFETY: Validated for non-nullness and alignment, then checked against `handle_a` below.
        let handle_b = unsafe { FfiHandleMut::<U>::from_raw(ptr_b)? };
        handle_a.ensure_disjoint(&handle_b)?;
        Ok((handle_a, handle_b))
    }

    /// Validates three mutable FFI pointers and enforces pairwise `GCSO_RESTRICT` non-overlapping memory ranges.
    ///
    /// # Safety
    /// Caller must ensure `ptr_a`, `ptr_b`, and `ptr_c` point to valid initialized instances for lifetime `'a`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if any pointer is null.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if any pointer is misaligned.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any two memory footprints overlap.
    #[inline(always)]
    pub unsafe fn from_raw_disjoint_triple<U, V>(
        ptr_a: *mut T,
        ptr_b: *mut U,
        ptr_c: *mut V,
    ) -> GcsoResult<(Self, FfiHandleMut<'a, U>, FfiHandleMut<'a, V>)> {
        // SAFETY: Validated for non-nullness, alignment, and pairwise disjointness before returning.
        let handle_a = unsafe { Self::from_raw(ptr_a)? };
        // SAFETY: Checked for disjointness below.
        let handle_b = unsafe { FfiHandleMut::<U>::from_raw(ptr_b)? };
        // SAFETY: Checked for disjointness below.
        let handle_c = unsafe { FfiHandleMut::<V>::from_raw(ptr_c)? };
        handle_a.ensure_disjoint(&handle_b)?;
        handle_a.ensure_disjoint(&handle_c)?;
        handle_b.ensure_disjoint(&handle_c)?;
        Ok((handle_a, handle_b, handle_c))
    }

    /// Branchlessly checks whether the byte range of `self` overlaps with another `FfiHandleMut`.
    #[inline(always)]
    #[must_use]
    pub fn overlaps<U>(&self, other: &FfiHandleMut<'_, U>) -> bool {
        raw_regions_overlap_or_same_zst(
            self.ptr.as_ptr() as usize,
            size_of::<T>(),
            other.ptr.as_ptr() as usize,
            size_of::<U>(),
        )
    }

    /// Enforces `GCSO_RESTRICT` non-aliasing invariant against another mutable FFI handle.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two handles overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint<U>(&self, other: &FfiHandleMut<'_, U>) -> GcsoResult<()> {
        if self.overlaps(other) {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        } else {
            Ok(())
        }
    }

    /// Enforces `GCSO_RESTRICT` non-aliasing invariant against an immutable `FfiHandleRef`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if the two handles overlap in memory.
    #[inline(always)]
    pub fn ensure_disjoint_ref<U>(&self, other: &FfiHandleRef<'_, U>) -> GcsoResult<()> {
        if raw_regions_overlap_or_same_zst(
            self.ptr.as_ptr() as usize,
            size_of::<T>(),
            other.as_ptr() as usize,
            size_of::<U>(),
        ) {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        } else {
            Ok(())
        }
    }

    /// Safely reinterprets `self` as a mutable handle to another `Pod` type `U` of equal or smaller size.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `size_of::<U>() == 0` or `size_of::<U>() > size_of::<T>()`.
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if the pointer violates `align_of::<U>()`.
    #[inline(always)]
    pub fn try_cast_mut<U: Pod>(&mut self) -> GcsoResult<FfiHandleMut<'_, U>>
    where
        T: Pod,
    {
        if U::IS_ZST || size_of::<U>() > size_of::<T>() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Both `T` and `U` are `Pod`, `size_of::<U>() <= size_of::<T>()`, and `from_raw` checks `align_of::<U>()`.
        unsafe { FfiHandleMut::<U>::from_raw(self.ptr.as_ptr().cast::<U>()) }
    }

    /// Returns the underlying `NonNull<T>` pointer.
    #[inline(always)]
    #[must_use]
    pub const fn as_non_null(&self) -> NonNull<T> {
        self.ptr
    }

    /// Returns the underlying raw mutable pointer `*mut T`.
    #[inline(always)]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Borrows the validated handle as an immutable `FfiHandleRef`.
    #[inline(always)]
    #[must_use]
    pub const fn as_handle_ref(&self) -> FfiHandleRef<'_, T> {
        FfiHandleRef {
            ptr: self.ptr,
            _marker: PhantomData,
        }
    }

    /// Borrows the validated handle as an immutable reference `&T`.
    #[inline(always)]
    #[must_use]
    pub const fn as_ref(&self) -> &T {
        // SAFETY: Validity and alignment guaranteed at construction.
        unsafe { &*self.ptr.as_ptr() }
    }

    /// Borrows the validated handle as a mutable reference `&mut T`.
    #[inline(always)]
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn as_mut(&mut self) -> &mut T {
        // SAFETY: Validity, alignment, and unique borrow guaranteed at construction.
        unsafe { &mut *self.ptr.as_ptr() }
    }

    /// Writes `value` directly into the validated target memory location.
    #[inline(always)]
    pub fn write(&mut self, value: T) {
        // SAFETY: Validity, alignment, and exclusive borrow guaranteed at construction.
        unsafe {
            *self.ptr.as_ptr() = value;
        }
    }

    /// Consumes the handle and converts it into a mutable reference `&'a mut T` with lifetime `'a`.
    #[inline(always)]
    #[must_use]
    pub fn into_mut(self) -> &'a mut T {
        // SAFETY: Validity, alignment, and unique borrow guaranteed at construction.
        unsafe { &mut *self.ptr.as_ptr() }
    }
}

impl<'a, T> Deref for FfiHandleMut<'a, T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<'a, T> DerefMut for FfiHandleMut<'a, T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}

impl<'a, T> AsRef<T> for FfiHandleMut<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &T {
        Self::as_ref(self)
    }
}

impl<'a, T> AsMut<T> for FfiHandleMut<'a, T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut T {
        Self::as_mut(self)
    }
}

// Compile-time structural and bitfield invariants for Layer 0 primitives.
const _: () = {
    assert!(GCSO_ABI_VERSION_HEX == 0x0000_0101);
    assert!(MAX_ATTRACTOR_DOMAINS == 8);
    assert!(MAX_ANCHORS_PER_FIELD == 64);
    assert!(MAX_ATTENTION_HEADS == 64);
    assert!((ATTRACTOR_SLOT_ID_MASK as usize) == (MAX_ATTRACTOR_DOMAINS - 1));
    assert!(pack_action_slot_tag(7, 0x1F) == 0xFF);
    assert!(unpack_attractor_slot_from_tag(pack_action_slot_tag(5, 0x13)) == 5);
    assert!(unpack_action_mode_from_tag(pack_action_slot_tag(5, 0x13)) == 0x13);
    assert!(slot_coupling_bit_index(7, 7) == 63);
    assert!(coupled_slot_row_mask(DEFAULT_SLOT_COUPLING_MATRIX_8X8, 3) == 0xFF);
    assert!(coupled_slot_count(DEFAULT_SLOT_COUPLING_MATRIX_8X8, 3) == 8);
    assert!(compute_crc32_ieee(b"123456789") == 0xCBF4_3926);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opaque_handles_type_safety() {
        assert_eq!(size_of::<GcsoContextHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoContainerHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoActionHubHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoDaesSlotHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoAttractorFieldHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoEdbcControllerHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoDpsrKernelHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoPspmRouterHandle>(), size_of::<*mut ()>());
        assert_eq!(size_of::<GcsoSrlAdapterHandle>(), size_of::<*mut ()>());
    }

    #[test]
    fn test_action_slot_tag_and_coupling_matrix_primitives() {
        let tag = pack_action_slot_tag(ATTRACTOR_SLOT_PERSONA, ACTION_PRESET_FULL_ORCHESTRATION);
        assert_eq!(unpack_attractor_slot_from_tag(tag), ATTRACTOR_SLOT_PERSONA);
        assert_eq!(
            unpack_action_mode_from_tag(tag),
            ACTION_PRESET_FULL_ORCHESTRATION
        );

        let mut matrix = 0u64;
        assert!(!is_slot_pair_coupled(
            matrix,
            u32::from(ATTRACTOR_SLOT_PERSONA),
            u32::from(ATTRACTOR_SLOT_KNOWLEDGE)
        ));

        matrix = with_slot_pair_coupled_symmetric(
            matrix,
            u32::from(ATTRACTOR_SLOT_PERSONA),
            u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
            true,
        );
        assert!(is_slot_pair_coupled(
            matrix,
            u32::from(ATTRACTOR_SLOT_PERSONA),
            u32::from(ATTRACTOR_SLOT_KNOWLEDGE)
        ));
        assert_eq!(
            coupled_slot_row_mask(matrix, u32::from(ATTRACTOR_SLOT_PERSONA)),
            1 << ATTRACTOR_SLOT_KNOWLEDGE
        );
        assert_eq!(
            coupled_slot_count(matrix, u32::from(ATTRACTOR_SLOT_PERSONA)),
            1
        );

        matrix = with_slot_pair_coupled_symmetric(
            matrix,
            u32::from(ATTRACTOR_SLOT_PERSONA),
            u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
            false,
        );
        assert_eq!(matrix, 0);
    }

    #[test]
    fn test_ffi_handle_ref_mut_cast_and_disjoint_validation() {
        let mut config = gcso_config_t::default();
        let raw_ptr: *mut gcso_config_t = &mut config;

        // SAFETY: `raw_ptr` points to a valid stack-allocated `gcso_config_t`.
        let handle_ref = unsafe { FfiHandleRef::from_raw(raw_ptr) }.unwrap();
        let cloned_ref = handle_ref;
        assert_eq!(cloned_ref.read_copy().head_dim, 0);

        let cast_u32 = handle_ref.try_cast::<u32>().unwrap();
        assert_eq!(*cast_u32, 0);

        // SAFETY: `raw_ptr` is uniquely borrowed here.
        let mut handle_mut = unsafe { FfiHandleMut::from_raw(raw_ptr) }.unwrap();
        handle_mut.head_dim = 128;
        assert_eq!(handle_mut.into_mut().head_dim, 128);
        assert_eq!(config.head_dim, 128);

        let mut a = 1u32;
        let mut b = 2u32;
        let mut c = 3u32;
        // SAFETY: `a`, `b`, and `c` are distinct stack variables.
        unsafe {
            let (mut ha, mut hb, mut hc) =
                FfiHandleMut::from_raw_disjoint_triple(&mut a, &mut b, &mut c).unwrap();
            ha.write(10);
            hb.write(20);
            hc.write(30);
            assert!(FfiHandleMut::from_raw_disjoint_pair(&mut a, &mut a).is_err());
        }
        assert_eq!((a, b, c), (10, 20, 30));

        let null_ptr: *const gcso_config_t = core::ptr::null();
        // SAFETY: Testing null pointer rejection and optional null handling.
        unsafe {
            assert_eq!(
                FfiHandleRef::from_raw(null_ptr),
                Err(GCSO_ERROR_NULL_POINTER)
            );
            assert!(FfiHandleRef::from_raw_opt(null_ptr).unwrap().is_none());
        }
    }

    #[test]
    fn test_pod_unaligned_bulk_io_and_incremental_crc32() {
        let hdr = init_descriptor_header::<gcso_config_t>();
        assert_eq!(validate_descriptor_header::<gcso_config_t>(&hdr), Ok(()));

        let mut buf = [0u8; 32];
        let src_arr = [hdr, hdr];
        assert_eq!(
            gcso_descriptor_header_t::write_unaligned_slice_to(&src_arr, &mut buf, 3),
            Ok(())
        );
        let mut dst_arr = [gcso_descriptor_header_t::default(); 2];
        assert_eq!(
            gcso_descriptor_header_t::read_unaligned_slice_from(&buf, 3, &mut dst_arr),
            Ok(())
        );
        assert_eq!(dst_arr, src_arr);

        let full_crc = compute_crc32_ieee(b"123456789");
        assert_eq!(full_crc, 0xCBF4_3926);
        let part1 = update_crc32_ieee(0, b"1234");
        let part2 = update_crc32_ieee(part1, b"56789");
        assert_eq!(part2, full_crc);

        let mut container = [0u8; 256];
        container[128..137].copy_from_slice(b"123456789");
        let mut snap = init_snapshot_header(256, 1_774_900_000);
        let stamped_crc = finalize_snapshot_header_crc32(&mut snap, &container).unwrap();
        assert_eq!(stamped_crc, snap.checksum_crc32);
        assert_eq!(verify_snapshot_payload_crc32(&snap, &container), Ok(()));

        container[128] ^= 0xFF;
        assert_eq!(
            verify_snapshot_payload_crc32(&snap, &container),
            Err(GCSO_ERROR_CONTAINER_CORRUPTED)
        );
    }
}