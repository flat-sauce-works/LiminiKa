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

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]
#![allow(clippy::manual_unwrap_or)]
#![allow(clippy::too_many_arguments)]

use core::f32::consts::{FRAC_PI_2, LN_2, PI};
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
        let is_aligned =
            slice.is_empty() | (((slice.as_ptr() as usize) & (align_of::<U>() - 1)) == 0);
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
        let is_aligned =
            slice.is_empty() | (((slice.as_ptr() as usize) & (align_of::<U>() - 1)) == 0);
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
    PprcKeyframePacket,
    PprcMotionDeltaPacket,
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
    branchless_update_crc32_ieee(crc_seed, bytes)
}

/// Computes a standard IEEE 802.3 CRC32 checksum (`0xEDB88320` reflected polynomial) over `bytes`
/// using the 64B-aligned 256-entry lookup table (`CRC32_IEEE_LUT_256`) in `#![no_std]` `const fn` mode.
#[inline(always)]
#[must_use]
pub const fn compute_crc32_ieee(bytes: &[u8]) -> u32 {
    branchless_update_crc32_ieee(0, bytes)
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

/// 128-byte cacheline-aligned Plain Old Data (`Pod`) **PPRC Keyframe (I-Frame) Packet** (Section 4.3).
///
/// Packs the 64-byte [`gcso_pprc_keyframe_header_t`] metadata header (`offset 0..64`) and the
/// uncompressed 64-head Q7 phase anchor snapshot (`offset 64..128`) for lightweight video-codec-style
/// `.gcso` memory & persona streams (Steam Workshop distribution ready).
///
/// The 24-byte `header.reserved` field encodes:
/// - `reserved[0..4]`: Little-endian IEEE 802.3 CRC32 checksum of `anchor_phases`.
/// - `reserved[4..8]`: Little-endian 32-bit `provenance_bits` causal mask.
/// - `reserved[8..12]`: Little-endian 32-bit macro Toroidal $k$-Means `cluster_id`.
/// - `reserved[12]`: 3-bit working-lane `slot_id` (`0..=7`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PprcKeyframePacket {
    /// 64-byte C-ABI v0.1.1 PPRC keyframe index header (`offset 0..64`, `frame_type == 0`).
    pub header: gcso_pprc_keyframe_header_t,
    /// 64-head Q7 phase anchor snapshot on $SO(2)^{64}$ (`offset 64..128`).
    pub anchor_phases: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

impl Default for PprcKeyframePacket {
    #[inline(always)]
    fn default() -> Self {
        Self {
            header: gcso_pprc_keyframe_header_t {
                frame_type: 0,
                token_index: 0,
                gop_length: 1,
                composite_vector_length: 0.0,
                von_mises_kappa: 0.0,
                sparse_scalar_residual: 0.0,
                icache_payload_offset: 0,
                pcache_payload_offset: 0,
                reserved: [0; 24],
            },
            anchor_phases: [0; MAX_ATTENTION_HEADS],
        }
    }
}

impl PprcKeyframePacket {
    /// Computes the IEEE 802.3 CRC32 checksum over the 64-head `anchor_phases` payload in `const fn` mode.
    #[inline(always)]
    #[must_use]
    pub const fn compute_anchor_crc32(anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> u32 {
        let mut raw_bytes = [0u8; MAX_ATTENTION_HEADS];
        let mut idx = 0usize;
        while idx < MAX_ATTENTION_HEADS {
            raw_bytes[idx] = anchor_phases[idx] as u8;
            idx += 1;
        }
        compute_crc32_ieee(&raw_bytes)
    }

    /// Constructs a 128-byte [`PprcKeyframePacket`] (I-Frame Key Anchor) stamped with IEEE 802.3
    /// CRC32 integrity protection and causal provenance metadata.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        token_index: u32,
        gop_length: u32,
        composite_vector_length: f32,
        von_mises_kappa: f32,
        sparse_scalar_residual: f32,
        provenance_bits: u32,
        cluster_id: u32,
        slot_id: u8,
        anchor_phases: [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> Self {
        let crc = Self::compute_anchor_crc32(&anchor_phases);
        let crc_le = crc.to_le_bytes();
        let prov_le = provenance_bits.to_le_bytes();
        let cid_le = cluster_id.to_le_bytes();
        let mut reserved = [0u8; 24];
        reserved[0] = crc_le[0];
        reserved[1] = crc_le[1];
        reserved[2] = crc_le[2];
        reserved[3] = crc_le[3];
        reserved[4] = prov_le[0];
        reserved[5] = prov_le[1];
        reserved[6] = prov_le[2];
        reserved[7] = prov_le[3];
        reserved[8] = cid_le[0];
        reserved[9] = cid_le[1];
        reserved[10] = cid_le[2];
        reserved[11] = cid_le[3];
        reserved[12] = slot_id & ATTRACTOR_SLOT_ID_MASK;

        Self {
            header: gcso_pprc_keyframe_header_t {
                frame_type: 0,
                token_index,
                gop_length: if gop_length > 0 { gop_length } else { 1 },
                composite_vector_length,
                von_mises_kappa,
                sparse_scalar_residual,
                icache_payload_offset: 64,
                pcache_payload_offset: 128,
                reserved,
            },
            anchor_phases,
        }
    }

    /// Extracts the stamped IEEE 802.3 CRC32 checksum of `anchor_phases` from `header.reserved[0..4]`.
    #[inline(always)]
    #[must_use]
    pub const fn payload_crc32(&self) -> u32 {
        u32::from_le_bytes([
            self.header.reserved[0],
            self.header.reserved[1],
            self.header.reserved[2],
            self.header.reserved[3],
        ])
    }

    /// Extracts the 32-bit `provenance_bits` causal mask from `header.reserved[4..8]`.
    #[inline(always)]
    #[must_use]
    pub const fn provenance_bits(&self) -> u32 {
        u32::from_le_bytes([
            self.header.reserved[4],
            self.header.reserved[5],
            self.header.reserved[6],
            self.header.reserved[7],
        ])
    }

    /// Extracts the 32-bit Toroidal $k$-Means `cluster_id` from `header.reserved[8..12]`.
    #[inline(always)]
    #[must_use]
    pub const fn cluster_id(&self) -> u32 {
        u32::from_le_bytes([
            self.header.reserved[8],
            self.header.reserved[9],
            self.header.reserved[10],
            self.header.reserved[11],
        ])
    }

    /// Extracts the 3-bit working-lane `slot_id` (`0..=7`) from `header.reserved[12]`.
    #[inline(always)]
    #[must_use]
    pub const fn slot_id(&self) -> u8 {
        self.header.reserved[12] & ATTRACTOR_SLOT_ID_MASK
    }

    /// Verifies that `header.frame_type == 0` (I-Frame) and that the IEEE 802.3 CRC32 checksum of
    /// `anchor_phases` matches `self.payload_crc32()`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `frame_type != 0` or CRC32 verification fails.
    #[inline(always)]
    pub const fn verify_integrity(&self) -> GcsoResult<()> {
        if self.header.frame_type != 0 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        let expected_crc = self.payload_crc32();
        if expected_crc != 0 && Self::compute_anchor_crc32(&self.anchor_phases) != expected_crc {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        Ok(())
    }
}

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) **PPRC Predictive Motion Delta (P-Frame) Packet** (Section 4.3).
///
/// Stores the 64-head inter-step Q7 phase motion vector $\boldsymbol{\Delta\theta}_t \in \mathrm{Q7}^{64}$
/// (`offset 0..64`) between consecutive steps inside a Variable Group of Pictures (GOP) memory/persona stream.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PprcMotionDeltaPacket {
    /// 64-head Q7 predictive phase motion delta vector $\boldsymbol{\Delta\theta}_t$ (`offset 0..64`).
    pub motion_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

impl Default for PprcMotionDeltaPacket {
    #[inline(always)]
    fn default() -> Self {
        Self::new([0; MAX_ATTENTION_HEADS])
    }
}

impl PprcMotionDeltaPacket {
    /// Constructs a 64-byte aligned [`PprcMotionDeltaPacket`] from `motion_deltas`.
    #[inline(always)]
    #[must_use]
    pub const fn new(motion_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS]) -> Self {
        Self { motion_deltas }
    }
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

// ===================================================================
// 7. Static 64B-Aligned Lookup Tables, Branchless Math & 8-Lane SWAR Primitives
// ===================================================================

/// 64-bit SWAR sign-bit mask (`0x80` per byte lane across 8 lanes).
pub const SWAR_SIGN_MASK_64: u64 = 0x8080_8080_8080_8080;
/// 64-bit SWAR lower-7-bit payload mask (`0x7F` per byte lane across 8 lanes).
pub const SWAR_LOW7_MASK_64: u64 = 0x7F7F_7F7F_7F7F_7F7F;
/// 64-bit SWAR even-byte isolation mask (`0x00FF` per 16-bit lane across 4 lanes).
pub const SWAR_EVEN_BYTES_64: u64 = 0x00FF_00FF_00FF_00FF;
/// 64-bit SWAR odd-byte isolation mask (`0xFF00` per 16-bit lane across 4 lanes).
pub const SWAR_ODD_BYTES_64: u64 = 0xFF00_FF00_FF00_FF00;
/// 64-bit SWAR byte-splat multiplier (`0x01` per byte lane across 8 lanes).
pub const SWAR_SPLAT_BYTE_64: u64 = 0x0101_0101_0101_0101;
/// 64-bit SWAR 16-bit lane horizontal reduction multiplier (`0x0001` per 16-bit lane).
pub const SWAR_SUM_U16_64: u64 = 0x0001_0001_0001_0001;
/// 32-bit SWAR sign-bit mask (`0x80` per byte lane across 4 lanes).
pub const SWAR_SIGN_MASK_32: u32 = 0x8080_8080;
/// 32-bit SWAR lower-7-bit payload mask (`0x7F` per byte lane across 4 lanes).
pub const SWAR_LOW7_MASK_32: u32 = 0x7F7F_7F7F;
const INV_TWO_PI_F64: f64 = 1.0 / (2.0 * core::f64::consts::PI);
const TWO_PI_F64: f64 = 2.0 * core::f64::consts::PI;
const INV_LN2_F32: f32 = core::f32::consts::LOG2_E;
const LN2_HI_F32: f32 = 0.693_145_75;
const LN2_LO_F32: f32 = 1.428_606_8e-6;
pub(crate) const INPLACE_GAIN_OVERFLOW_CEIL: f32 = 1.0e30;
pub(crate) const MAX_BAYESIAN_STIGMERGIC_DENSITY: f32 = 16.0;
pub(crate) const INV_64_Q14_F32: f32 = f32::from_bits((127 - 20) << 23);
pub(crate) const INV_8_Q14_F32: f32 = f32::from_bits((127 - 17) << 23);
pub(crate) const SUBNORMAL_SQ_RISK_FLOOR: f32 = 1.0e-18;
pub(crate) const Q14_BIAS_I32: i32 = 16_384;
pub(crate) const Q14_BIAS_X8_I32: i32 = 8 * Q14_BIAS_I32;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub(crate) const Q14_BIAS_X64_I32: i32 = 64 * Q14_BIAS_I32;

/// Minimum positive normal IEEE-754 `f32` value ($2^{-126} \approx 1.17549435 \times 10^{-38}$).
pub const F32_MIN_POSITIVE_NORMAL: f32 = f32::from_bits(0x0080_0000);
/// Default upper ceiling for token Z-score inputs ($10.0\sigma$).
pub const DEFAULT_MAX_VALID_Z_SCORE: f32 = 10.0;
/// Precomputed normalization scale `Q7_PHASE_SCALE_RAD / DEFAULT_RIPA_CLAMP_MAX_RAD`.
pub const DEFAULT_RIPA_Q7_TO_NORM_SCALE: f32 = Q7_PHASE_SCALE_RAD / DEFAULT_RIPA_CLAMP_MAX_RAD;
/// Minimum clamped inverse-temperature scale for destructive phase interference ($0.25$).
pub const MIN_PHASE_INTERFERENCE_INV_TEMP: f32 = 0.25;
/// Maximum clamped inverse-temperature scale for constructive phase interference ($4.0$).
pub const MAX_PHASE_INTERFERENCE_INV_TEMP: f32 = 4.0;
/// Default softening epsilon $\epsilon_{\mathrm{grav}}$ for Phase-Proximity Gravity on $SO(2)^{64}$ ($0.25$).
pub const DEFAULT_PHASE_GRAVITY_EPS: f32 = 0.25;

pub(crate) const TANH_LUT_MAX_X: f32 = 4.97;
pub(crate) const TANH_LUT_STEP: f32 = TANH_LUT_MAX_X / 256.0;
pub(crate) const TANH_LUT_INV_STEP: f32 = 256.0 / TANH_LUT_MAX_X;

#[repr(C, align(64))]
pub(crate) struct AlignedLut256<T>(pub [T; 256]);

#[repr(C, align(64))]
pub(crate) struct AlignedLut257<T>(pub [T; 257]);

#[repr(C, align(64))]
pub(crate) struct AlignedLut2x257<T>(pub [[T; 257]; 2]);

const VON_MISES_KAPPA_SPLIT_R: f32 = 0.875;
const VON_MISES_KAPPA_MAX_R: f32 = 0.995;
const VON_MISES_KAPPA_LO_STEP: f32 = VON_MISES_KAPPA_SPLIT_R / 256.0;
const VON_MISES_KAPPA_HI_STEP: f32 = (VON_MISES_KAPPA_MAX_R - VON_MISES_KAPPA_SPLIT_R) / 256.0;
const VON_MISES_KAPPA_LO_SCALE: f32 = 256.0 / VON_MISES_KAPPA_SPLIT_R;
const VON_MISES_KAPPA_HI_SCALE: f32 = 256.0 / (VON_MISES_KAPPA_MAX_R - VON_MISES_KAPPA_SPLIT_R);

#[repr(C, align(64))]
pub(crate) struct AlignedLut4x256<T>(pub [[T; 256]; 4]);

#[repr(C, align(64))]
pub(crate) struct AlignedLut16x256<T>(pub [[T; 256]; 16]);

/// 64B-aligned 16KB Slice-by-16 IEEE 802.3 CRC32 lookup table (`0xEDB8_8320` reflected polynomial)
/// enabling 16-byte (`2 x u64`) parallel XOR-table folding per step (`16x` fewer serial dependency
/// steps than byte-serial lookup and `2x` fewer than Slice-by-8 on `.gcso` ZIMMS containers and
/// 128-byte `CrystallizedAttractorRecord` blocks).
pub(crate) const CRC32_IEEE_SLICE16_LUT: AlignedLut16x256<u32> = {
    let mut lut = [[0u32; 256]; 16];
    let mut i = 0usize;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0usize;
        while bit < 8 {
            let mask = (c & 1).wrapping_neg();
            c = (c >> 1) ^ (0xEDB8_8320 & mask);
            bit += 1;
        }
        lut[0][i] = c;
        i += 1;
    }
    let mut s = 1usize;
    while s < 16 {
        let mut idx = 0usize;
        while idx < 256 {
            let prev = lut[s - 1][idx];
            lut[s][idx] = (prev >> 8) ^ lut[0][(prev & 0xFF) as usize];
            idx += 1;
        }
        s += 1;
    }
    AlignedLut16x256(lut)
};

const Q7_QUADRANT_SIN_LUT: [f32; 65] = {
    let mut lut = [0.0_f32; 65];
    let mut k = 1usize;
    while k < 64 {
        let x = (k as f32) * Q7_PHASE_SCALE_RAD;
        let x2 = x * x;
        lut[k] = x
            * (1.0
                + x2 * (-0.166_666_67
                    + x2 * (0.008_333_334
                        + x2 * (-0.000_198_412_7
                            + x2 * (0.000_002_755_732 + x2 * -0.000_000_025_052_108)))));
        k += 1;
    }
    lut[64] = 1.0;
    lut
};

pub(crate) const Q7_FULL_SIN_LUT: AlignedLut256<f32> = {
    let mut lut = [0.0_f32; 256];
    let mut b = 0usize;
    while b < 256 {
        let phase_q7 = (b as u8) as gcso_q7_t;
        let u = branchless_q7_abs_u8(phase_q7);
        let gt_64 = u > 64;
        let k = branchless_select_u8(gt_64, 128u8.wrapping_sub(u), u) as usize;
        let sin_mag = Q7_QUADRANT_SIN_LUT[k];
        let sin_neg = (phase_q7 < 0) & (k != 0);
        lut[b] = f32::from_bits(sin_mag.to_bits() | ((sin_neg as u32) << 31));
        b += 1;
    }
    AlignedLut256(lut)
};

pub(crate) const Q7_FULL_COS_LUT: AlignedLut256<f32> = {
    let mut lut = [1.0_f32; 256];
    let mut b = 0usize;
    while b < 256 {
        let phase_q7 = (b as u8) as gcso_q7_t;
        let u = branchless_q7_abs_u8(phase_q7);
        let gt_64 = u > 64;
        let k = branchless_select_u8(gt_64, 128u8.wrapping_sub(u), u) as usize;
        let cos_mag = Q7_QUADRANT_SIN_LUT[64 - k];
        let cos_neg = gt_64 & (k != 64);
        lut[b] = f32::from_bits(cos_mag.to_bits() | ((cos_neg as u32) << 31));
        b += 1;
    }
    AlignedLut256(lut)
};

pub(crate) const Q7_FULL_SINCOS_LUT: AlignedLut256<(f32, f32)> = {
    let mut lut = [(0.0_f32, 1.0_f32); 256];
    let mut b = 0usize;
    while b < 256 {
        lut[b] = (Q7_FULL_SIN_LUT.0[b], Q7_FULL_COS_LUT.0[b]);
        b += 1;
    }
    AlignedLut256(lut)
};

/// 64B-aligned 2KB interleaved `(sin, cos)` LUT indexed directly by `phase_q7 as u8`.
pub(crate) const Q7_SINCOS_LUT_256: AlignedLut256<(f32, f32)> = Q7_FULL_SINCOS_LUT;

/// 64B-aligned 512B Q14 fixed-point (`1.0 == 16384`) cosine LUT for 1-cycle integer ALU reductions.
pub(crate) const Q7_FULL_COS_I16_LUT: AlignedLut256<i16> = {
    let mut lut = [16384_i16; 256];
    let mut b = 0usize;
    while b < 256 {
        let c = Q7_FULL_COS_LUT.0[b] * 16384.0;
        lut[b] = (c + if c >= 0.0 { 0.5 } else { -0.5 }) as i16;
        b += 1;
    }
    AlignedLut256(lut)
};

/// 64B-aligned 1KB Q14 fixed-point `[sin_q14, cos_q14]` LUT for 1-cycle integer ALU reductions.
pub(crate) const Q7_FULL_SINCOS_I16_LUT: AlignedLut256<[i16; 2]> = {
    let mut lut = [[0_i16, 16384_i16]; 256];
    let mut b = 0usize;
    while b < 256 {
        let s = Q7_FULL_SIN_LUT.0[b] * 16384.0;
        let c = Q7_FULL_COS_LUT.0[b] * 16384.0;
        lut[b] = [
            (s + if s >= 0.0 { 0.5 } else { -0.5 }) as i16,
            (c + if c >= 0.0 { 0.5 } else { -0.5 }) as i16,
        ];
        b += 1;
    }
    AlignedLut256(lut)
};

/// 64B-aligned 2KB SWAR-packed biased Q14 `(sin, cos)` LUT where each `u64` entry stores:
/// - Bits `0..31`: `(cos_q14 + 16384) as u32` $\in [0, 32768]$
/// - Bits `32..63`: `((sin_q14 + 16384) as u32 as u64) << 32`
///
/// Summing 64 entries in a single `u64` register (`<= 64 * 32768 = 2^21 << 2^32`) accumulates
/// both `sum_sin_q14` and `sum_cos_q14` simultaneously with zero cross-lane carry.
pub(crate) const Q7_FULL_SINCOS_BIASED_U64_LUT: AlignedLut256<u64> = {
    let mut lut = [0u64; 256];
    let mut b = 0usize;
    while b < 256 {
        let [s_i16, c_i16] = Q7_FULL_SINCOS_I16_LUT.0[b];
        let u_sin = ((s_i16 as i32) + Q14_BIAS_I32) as u32 as u64;
        let u_cos = ((c_i16 as i32) + Q14_BIAS_I32) as u32 as u64;
        lut[b] = (u_sin << 32) | u_cos;
        b += 1;
    }
    AlignedLut256(lut)
};

const TANH_LUT_257: AlignedLut257<f32> = {
    let mut lut = [0.0_f32; 257];
    let mut k = 1usize;
    while k <= 256 {
        let x = (k as f32) * TANH_LUT_STEP;
        let x2 = x * x;
        let x4 = x2 * x2;
        lut[k] = (x * (945.0 + 105.0 * x2 + x4)) / (945.0 + 420.0 * x2 + 15.0 * x4);
        k += 1;
    }
    AlignedLut257(lut)
};

/// Evaluates $\tanh(u)$ for a non-negative finite float $u \ge 0$ using the 64-byte aligned
/// 1KB `TANH_LUT_257_REF` table (`step = 1/64`) and a 2nd-order analytic ODE Taylor expansion
/// ($\tanh' = 1 - y^2$, $\tanh'' = -2y(1 - y^2)$) in `0` divisions and `0` reciprocals.
#[inline(always)]
#[must_use]
pub const fn tanh_lut_eval_nonneg(u_nonneg: f32) -> f32 {
    let clamped = branchless_select_f32(u_nonneg < TANH_LUT_MAX_X, u_nonneg, TANH_LUT_MAX_X);
    let idx_raw = (clamped * TANH_LUT_INV_STEP) as usize;
    let idx = branchless_select_usize(idx_raw < 256, idx_raw, 255);
    let d = clamped - (idx as f32) * TANH_LUT_STEP;
    let y0 = TANH_LUT_257_REF[idx];
    y0 + d * (1.0 - y0 * y0) * (1.0 - y0 * d)
}

#[inline(always)]
pub(crate) const fn sincos_ripa_poly(safe_theta: f32) -> (f32, f32) {
    let x2 = safe_theta * safe_theta;
    let sin_t =
        safe_theta * (1.0 + x2 * (-0.166_666_67 + x2 * (0.008_333_334 - x2 * 0.000_198_412_7)));
    let cos_t = 1.0
        + x2 * (-0.5 + x2 * (0.041_666_668 + x2 * (-0.001_388_888_9 + x2 * 0.000_024_801_587)));
    (sin_t, cos_t)
}

pub(crate) const Q7_DEFAULT_RIPA_SINCOS_LUT_256: AlignedLut256<(f32, f32)> = {
    let mut lut = [(0.0_f32, 1.0_f32); 256];
    let mut b = 0usize;
    while b < 256 {
        let phase_q7 = (b as u8) as gcso_q7_t;
        let u = branchless_q7_abs_u8(phase_q7);
        let safe_theta = DEFAULT_RIPA_CLAMP_MAX_RAD
            * tanh_lut_eval_nonneg((u as f32) * DEFAULT_RIPA_Q7_TO_NORM_SCALE);
        let (sin_mag, cos_t) = sincos_ripa_poly(safe_theta);
        let sin_neg = (phase_q7 < 0) & (u != 0);
        lut[b] = (
            f32::from_bits(sin_mag.to_bits() | ((sin_neg as u32) << 31)),
            cos_t,
        );
        b += 1;
    }
    AlignedLut256(lut)
};

pub(crate) const Q7_GEODESIC_DEFAULT_RIPA_SINCOS_LUT_256: AlignedLut256<(f32, f32)> = {
    let mut table = [(0.0_f32, 1.0_f32); 256];
    let mut idx = 0usize;
    while idx < 256 {
        let folded = branchless_q7_fold_s1_geodesic(idx as i8);
        table[idx] = Q7_DEFAULT_RIPA_SINCOS_LUT_256.0[(folded as u8) as usize];
        idx += 1;
    }
    AlignedLut256(table)
};

pub(crate) const POPCOUNT_RSQRT_LUT_257: AlignedLut257<f32> = {
    let mut lut = [0.0_f32; 257];
    let mut k = 1usize;
    while k <= 256 {
        lut[k] = branchless_rsqrt_f32(k as f32);
        k += 1;
    }
    AlignedLut257(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 256-entry `(sin, cos)` Q7 table.
pub(crate) const Q7_SINCOS_LUT_256_REF: &[(f32, f32); 256] = &Q7_SINCOS_LUT_256.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry `sin` Q7 table.
pub(crate) const Q7_FULL_SIN_REF: &[f32; 256] = &Q7_FULL_SIN_LUT.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry `cos` Q7 table.
pub(crate) const Q7_FULL_COS_REF: &[f32; 256] = &Q7_FULL_COS_LUT.0;

/// 64B-aligned 2KB precomputed `(align_gain, inv_denom)` table for [`branchless_phase_proximity_gravity_f32`]
/// under `DEFAULT_PHASE_GRAVITY_EPS` (`0.25`) across all 256 Q7 circular difference bytes `0..=255`,
/// enabling 0-ULP bit-exact gravity evaluation in 1 LUT load + 2 `vmulss` (`0` Newton-Raphson iterations).
pub(crate) const Q7_DEFAULT_GRAVITY_PAIR_LUT_256: AlignedLut256<(f32, f32)> = {
    let mut lut = [(1.0_f32, 2.0_f32 / DEFAULT_PHASE_GRAVITY_EPS); 256];
    let mut b = 1usize;
    while b < 256 {
        let coh = Q7_FULL_COS_LUT.0[b];
        let chord_dist_sq = 2.0 * (1.0 - coh);
        let denom = chord_dist_sq + DEFAULT_PHASE_GRAVITY_EPS;
        let align_gain = branchless_max_f32(0.05, 1.0 + coh);
        let inv_denom = fast_inv_pos_normal_f32(denom);
        lut[b] = (align_gain, inv_denom);
        b += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 256-entry default gravity `(align_gain, inv_denom)` table.
pub(crate) const Q7_DEFAULT_GRAVITY_PAIR_REF: &[(f32, f32); 256] =
    &Q7_DEFAULT_GRAVITY_PAIR_LUT_256.0;

/// Evaluates Phase-Proximity Gravity $F_{\mathrm{grav}}(M_{\mathrm{grav}}, \cos(\Delta\theta_{\mathrm{Q7}}), \epsilon_{\mathrm{default}})$
/// directly from a uniform Q7 circular difference byte `diff_u8` in 1 LUT load + 2 `vmulss` (`0` Newton-Raphson reciprocals, `0` ULP drift).
#[inline(always)]
#[must_use]
pub(crate) const fn branchless_q7_default_gravity_from_diff_u8(
    gravity_mass: f32,
    diff_u8: u8,
) -> f32 {
    let clean_mass = branchless_flush_subnormal_f32(branchless_relu_f32(gravity_mass));
    let (align_gain, inv_denom) = Q7_DEFAULT_GRAVITY_PAIR_REF[diff_u8 as usize];
    branchless_flush_subnormal_f32((clean_mass * align_gain) * inv_denom)
}
/// Promoted `.rodata` reference to the 64B-aligned 256-entry Q14 `i16` cosine table.
pub(crate) const Q7_FULL_COS_I16_REF: &[i16; 256] = &Q7_FULL_COS_I16_LUT.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry SWAR-packed biased Q14 `(sin, cos)` table.
pub(crate) const Q7_FULL_SINCOS_BIASED_U64_REF: &[u64; 256] = &Q7_FULL_SINCOS_BIASED_U64_LUT.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry default RIPA `(sin, cos)` table.
pub(crate) const Q7_DEFAULT_RIPA_SINCOS_REF: &[(f32, f32); 256] = &Q7_DEFAULT_RIPA_SINCOS_LUT_256.0;
/// Promoted `.rodata` reference to the 64B-aligned 256-entry geodesic default RIPA `(sin, cos)` table.
pub(crate) const Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_GEODESIC_DEFAULT_RIPA_SINCOS_LUT_256.0;

#[inline(always)]
const fn build_q7_fused_self_ripa_sincos_lut(fold_geodesic: bool) -> AlignedLut256<(f32, f32)> {
    let mut table = [(0.0_f32, 1.0_f32); 256];
    let mut i = 0usize;
    while i < 256 {
        let (sin_base, cos_base) = Q7_DEFAULT_RIPA_SINCOS_LUT_256.0[i];
        let (sin_pspm, cos_pspm) = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_RIPA_SINCOS_LUT_256.0[i]
        } else {
            Q7_DEFAULT_RIPA_SINCOS_LUT_256.0[i]
        };
        let sin_comb = sin_base * cos_pspm + cos_base * sin_pspm;
        let cos_comb = cos_base * cos_pspm - sin_base * sin_pspm;
        table[i] = (sin_comb, cos_comb);
        i += 1;
    }
    AlignedLut256(table)
}

/// Precomputed 64-byte-aligned 2KB LUT of fused $SO(2)$ double-rotation `(sin(2*theta), cos(2*theta))`
/// when `base_ripa_q7 == pspm_slot_q7` under `DEFAULT_RIPA_CLAMP_MAX_RAD` (non-geodesic PSPM).
pub(crate) const Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_LUT_256: AlignedLut256<(f32, f32)> =
    build_q7_fused_self_ripa_sincos_lut(false);

/// Precomputed 64-byte-aligned 2KB LUT of fused $SO(2)$ double-rotation `(sin(theta_b + theta_p), cos(theta_b + theta_p))`
/// when `base_ripa_q7 == pspm_slot_q7` under `DEFAULT_RIPA_CLAMP_MAX_RAD` (geodesic PSPM).
pub(crate) const Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_LUT_256: AlignedLut256<(f32, f32)> =
    build_q7_fused_self_ripa_sincos_lut(true);

pub(crate) const Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_LUT_256.0;

pub(crate) const Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF: &[(f32, f32); 256] =
    &Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_LUT_256.0;
/// Promoted `.rodata` reference to the 64B-aligned 257-entry `tanh` table.
pub(crate) const TANH_LUT_257_REF: &[f32; 257] = &TANH_LUT_257.0;
/// Promoted `.rodata` reference to the 64B-aligned 257-entry popcount `rsqrt` table.
pub(crate) const POPCOUNT_RSQRT_LUT_257_REF: &[f32; 257] = &POPCOUNT_RSQRT_LUT_257.0;

#[inline(always)]
pub(crate) const fn branchless_popcount_rsqrt_u32(popcount: u32) -> f32 {
    let idx = branchless_select_usize((popcount as usize) <= 256, popcount as usize, 256);
    POPCOUNT_RSQRT_LUT_257_REF[idx]
}

/// Promoted `.rodata` reference to the 64B-aligned 16KB Slice-by-16 IEEE 802.3 CRC32 table,
/// ensuring debug builds (`opt-level = 0`, `cargo test`) index directly into `.rodata` via an
/// 8-byte pointer rather than materializing 16KB stack temporaries per table lookup.
pub(crate) const CRC32_IEEE_SLICE16_REF: &[[u32; 256]; 16] = &CRC32_IEEE_SLICE16_LUT.0;

#[inline(always)]
const fn crc32_slice4_step(crc: u32, b0: u8, b1: u8, b2: u8, b3: u8) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    let word = u32::from_le_bytes([b0, b1, b2, b3]) ^ crc;
    lut[3][(word & 0xFF) as usize]
        ^ lut[2][((word >> 8) & 0xFF) as usize]
        ^ lut[1][((word >> 16) & 0xFF) as usize]
        ^ lut[0][(word >> 24) as usize]
}

#[inline(always)]
pub(crate) const fn crc32_slice8_step_u64(crc: u32, word64_le: u64) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    let lo = (word64_le as u32) ^ crc;
    let hi = (word64_le >> 32) as u32;
    (lut[7][(lo & 0xFF) as usize]
        ^ lut[6][((lo >> 8) & 0xFF) as usize]
        ^ lut[5][((lo >> 16) & 0xFF) as usize]
        ^ lut[4][(lo >> 24) as usize])
        ^ (lut[3][(hi & 0xFF) as usize]
            ^ lut[2][((hi >> 8) & 0xFF) as usize]
            ^ lut[1][((hi >> 16) & 0xFF) as usize]
            ^ lut[0][(hi >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_slice16_dep_lo0(lo0: u32) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    (lut[15][(lo0 & 0xFF) as usize]
        ^ lut[14][((lo0 >> 8) & 0xFF) as usize])
        ^ (lut[13][((lo0 >> 16) & 0xFF) as usize]
            ^ lut[12][(lo0 >> 24) as usize])
}

/// 64B-aligned 4KB GF(2) linear operator table for $L^2(x)$ (`32`-byte zero-shift modulo IEEE 802.3 CRC32),
/// enabling $2 \times 32\text{B}$ parallel tree reduction per 64B cacheline (`2x` shorter serial dependency chain).
pub(crate) const CRC32_IEEE_L2_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let shift = byte_pos * 8;
        let mut b = 0usize;
        while b < 256 {
            let x = (b as u32) << shift;
            lut[byte_pos][b] = crc32_slice16_dep_lo0(crc32_slice16_dep_lo0(x));
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) linear operator table for $L^3(x) = L^1(L^2(x))$ (`48`-byte zero-shift
/// modulo IEEE 802.3 CRC32), enabling 4-way parallel 16B chunk evaluation per 64B cacheline
/// (`2` serial L1D load stages instead of `3..4`).
pub(crate) const CRC32_IEEE_L3_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let l2 = CRC32_IEEE_L2_LUT.0[byte_pos][b];
            lut[byte_pos][b] = crc32_slice16_dep_lo0(l2);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) linear operator table for $L^4(x)$ (`64`-byte zero-shift modulo IEEE 802.3 CRC32),
/// enabling single-step 64B block folding (`128B` `CrystallizedAttractorRecord` in `2` parallel 64B halves).
pub(crate) const CRC32_IEEE_L4_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let l2 = CRC32_IEEE_L2_LUT.0[byte_pos][b];
            lut[byte_pos][b] = crc32_slice16_dep_lo0(crc32_slice16_dep_lo0(l2));
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) linear operator table for $L^8(x) = L^4(L^4(x))$ (`128`-byte zero-shift
/// modulo IEEE 802.3 CRC32), enabling single-step 128B block folding across stamped
/// `CrystallizedAttractorRecord` arrays and 256B ($4 \times 64\text{B}$) container blocks.
pub(crate) const CRC32_IEEE_L8_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let l4 = CRC32_IEEE_L4_LUT.0[byte_pos][b];
            let l6 = crc32_slice16_dep_lo0(crc32_slice16_dep_lo0(l4));
            lut[byte_pos][b] = crc32_slice16_dep_lo0(crc32_slice16_dep_lo0(l6));
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB $L^2(x)$ 32-byte shift table.
pub(crate) const CRC32_IEEE_L2_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L2_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^3(x)$ 48-byte shift table.
pub(crate) const CRC32_IEEE_L3_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L3_LUT.0;
/// 64B-aligned 4KB GF(2) operator table for $L^5(x) = L^1(L^4(x))$ (advancing a 32-bit CRC state by 80 zero bytes).
pub(crate) const CRC32_IEEE_L5_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            lut[byte_pos][b] = crc32_slice16_dep_lo0(CRC32_IEEE_L4_LUT.0[byte_pos][b]);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) operator table for $L^6(x) = L^2(L^4(x))$ (advancing a 32-bit CRC state by 96 zero bytes).
pub(crate) const CRC32_IEEE_L6_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            lut[byte_pos][b] = crc32_slice16_dep_lo0(CRC32_IEEE_L5_LUT.0[byte_pos][b]);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB $L^4(x)$ 64-byte shift table.
pub(crate) const CRC32_IEEE_L4_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L4_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^5(x)$ 80-byte shift table.
pub(crate) const CRC32_IEEE_L5_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L5_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^6(x)$ 96-byte shift table.
pub(crate) const CRC32_IEEE_L6_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L6_LUT.0;
/// 64B-aligned 4KB GF(2) operator table for $L^{16}(x) = L^8(L^8(x))$ (`256`-byte / 2-record zero-shift).
pub(crate) const CRC32_IEEE_L16_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            lut[byte_pos][b] = crc32_step_l8(CRC32_IEEE_L8_LUT.0[byte_pos][b]);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// 64B-aligned 4KB GF(2) operator table for $L^{32}(x) = L^{16}(L^{16}(x))$ (`512`-byte / 4-record zero-shift).
pub(crate) const CRC32_IEEE_L32_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let l16 = CRC32_IEEE_L16_LUT.0[byte_pos][b];
            lut[byte_pos][b] = crc32_step_l8(crc32_step_l8(l16));
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB $L^8(x)$ 128-byte shift table.
pub(crate) const CRC32_IEEE_L8_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L8_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^{16}(x)$ 256-byte shift table.
pub(crate) const CRC32_IEEE_L16_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L16_LUT.0;
/// Promoted `.rodata` reference to the 4KB $L^{32}(x)$ 512-byte shift table.
pub(crate) const CRC32_IEEE_L32_REF: &[[u32; 256]; 4] = &CRC32_IEEE_L32_LUT.0;

#[inline(always)]
pub(crate) const fn crc32_step_l2(x: u32) -> u32 {
    let lut = CRC32_IEEE_L2_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l3(x: u32) -> u32 {
    let lut = CRC32_IEEE_L3_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l4(x: u32) -> u32 {
    let lut = CRC32_IEEE_L4_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l5(x: u32) -> u32 {
    let lut = CRC32_IEEE_L5_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l6(x: u32) -> u32 {
    let lut = CRC32_IEEE_L6_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l8(x: u32) -> u32 {
    let lut = CRC32_IEEE_L8_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l16(x: u32) -> u32 {
    let lut = CRC32_IEEE_L16_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_step_l32(x: u32) -> u32 {
    let lut = CRC32_IEEE_L32_REF;
    (lut[0][(x & 0xFF) as usize] ^ lut[1][((x >> 8) & 0xFF) as usize])
        ^ (lut[2][((x >> 16) & 0xFF) as usize] ^ lut[3][(x >> 24) as usize])
}

/// 64B-aligned 4KB GF(2) affine operator table fusing $L^4(L_{12}(\text{record\_crc32})) \oplus (\neg\text{record\_crc32}) \oplus L^8(\text{0xFFFF\_FFFF})$
/// into a single 4-lookup stage (`0` branches, `1` lookup stage instead of `2` serial stages).
pub(crate) const CRC32_IEEE_CRYSTAL_STAMP_LUT: AlignedLut4x256<u32> = {
    const L8_ONES: u32 = crc32_step_l8(0xFFFF_FFFFu32);
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let shift = byte_pos * 8;
        let mut b = 0usize;
        while b < 256 {
            let d = CRC32_IEEE_SLICE16_LUT.0[11 - byte_pos][b];
            let mut v = crc32_step_l4(d) ^ ((b as u32) << shift);
            if byte_pos == 0 {
                v ^= 0xFFFF_FFFFu32 ^ L8_ONES;
            }
            lut[byte_pos][b] = v;
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB 1-stage crystal stamp CRC32 folding table.
pub(crate) const CRC32_IEEE_CRYSTAL_STAMP_REF: &[[u32; 256]; 4] = &CRC32_IEEE_CRYSTAL_STAMP_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for low-byte `sample_count` increments
/// (`offset 12..16` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 112 bytes),
/// updating a stationary centroid's `checksum_crc32` in 1 table lookup (`1` cycle).
pub(crate) const CRC32_CRYSTAL_SC_BYTE0_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        let l0 = CRC32_IEEE_SLICE16_LUT.0[3][d];
        lut[d] = crc32_step_l4(crc32_step_l2(crc32_slice16_dep_lo0(l0)));
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `sample_count` byte-0 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_SC_BYTE0_REF: &[u32; 256] = &CRC32_CRYSTAL_SC_BYTE0_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for low-byte `anchor_id` values
/// (`offset 8..12` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 116 bytes via $L^7$),
/// evaluating `anchor_id < 256` contributions in 1 table lookup (`1` cycle, `0` serial $L^3/L^4$ stages).
pub(crate) const CRC32_CRYSTAL_AID_BYTE0_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        let l0 = CRC32_IEEE_SLICE16_LUT.0[7][d];
        lut[d] = crc32_step_l4(crc32_step_l3(l0));
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `anchor_id` byte-0 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_AID_BYTE0_REF: &[u32; 256] = &CRC32_CRYSTAL_AID_BYTE0_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-1 `anchor_id` values
/// (`offset 9` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 116 bytes via $L^7$),
/// evaluating `anchor_id < 65536` contributions in 2 parallel table lookups (`0` serial $L^3/L^4$ stages).
pub(crate) const CRC32_CRYSTAL_AID_BYTE1_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        let l0 = CRC32_IEEE_SLICE16_LUT.0[6][d];
        lut[d] = crc32_step_l4(crc32_step_l3(l0));
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `anchor_id` byte-1 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_AID_BYTE1_REF: &[u32; 256] = &CRC32_CRYSTAL_AID_BYTE1_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-1 (`(zoff >> 8) & 0xFF`, `offset 17`)
/// of `zimms_section_offset48` inside a 128-byte `CrystallizedAttractorRecord`, pre-shifted by
/// 96 bytes via $L^6$ so 4KB-aligned ZIMMS section offsets fold in 1 lookup stage (`0` serial $L^6$ stages).
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE1_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l6(CRC32_IEEE_SLICE16_LUT.0[14][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `zimms_section_offset48` byte-1 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE1_REF: &[u32; 256] = &CRC32_CRYSTAL_ZOFF_BYTE1_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-2 (`(zoff >> 16) & 0xFF`, `offset 18`)
/// of `zimms_section_offset48` inside a 128-byte `CrystallizedAttractorRecord`, pre-shifted by
/// 96 bytes via $L^6$ so 64KB-aligned ZIMMS section offsets fold in 1 lookup stage (`0` serial $L^6$ stages).
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE2_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l6(CRC32_IEEE_SLICE16_LUT.0[13][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `zimms_section_offset48` byte-2 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE2_REF: &[u32; 256] = &CRC32_CRYSTAL_ZOFF_BYTE2_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-3 (`(zoff >> 24) & 0xFF`, `offset 19`)
/// of `zimms_section_offset48` inside a 128-byte `CrystallizedAttractorRecord`, pre-shifted by
/// 96 bytes via $L^6$ so 64KB-aligned ZIMMS section offsets fold in 1 lookup stage (`0` serial $L^6$ stages).
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE3_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l6(CRC32_IEEE_SLICE16_LUT.0[12][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `zimms_section_offset48` byte-3 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE3_REF: &[u32; 256] = &CRC32_CRYSTAL_ZOFF_BYTE3_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-0 `provenance_bits` values
/// (`offset 48` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 64 bytes via $L^5$),
/// evaluating low-byte provenance bitmasks in 1 table lookup (`0` serial $L^4$ stages).
pub(crate) const CRC32_CRYSTAL_PROV_BYTE0_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l4(CRC32_IEEE_SLICE16_LUT.0[15][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `provenance_bits` byte-0 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_PROV_BYTE0_REF: &[u32; 256] = &CRC32_CRYSTAL_PROV_BYTE0_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-1 `provenance_bits` values
/// (`offset 49` inside a 128-byte `CrystallizedAttractorRecord`, shifted by 64 bytes via $L^5$),
/// evaluating 16-bit provenance bitmasks (`provenance_bits <= 0xFFFF`) in 2 parallel table lookups.
pub(crate) const CRC32_CRYSTAL_PROV_BYTE1_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l4(CRC32_IEEE_SLICE16_LUT.0[14][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `provenance_bits` byte-1 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_PROV_BYTE1_REF: &[u32; 256] = &CRC32_CRYSTAL_PROV_BYTE1_LUT.0;

/// Pre-shifted 8-entry GF(2) differential CRC32 table for `dominant_slot & 7` (`offset 56`),
/// shifted by 64 bytes via $L^4$.
pub(crate) const CRC32_CRYSTAL_SLOT_BYTE56_LUT_8: [u32; 8] = {
    let mut lut = [0u32; 8];
    let mut s = 0usize;
    while s < 8 {
        lut[s] = crc32_step_l4(CRC32_IEEE_SLICE16_LUT.0[7][s]);
        s += 1;
    }
    lut
};

/// 64B-aligned 1KB GF(2) differential CRC32 table for `action_mode_flags` (`offset 57`, shifted by 64 bytes via $L^4$).
pub(crate) const CRC32_CRYSTAL_ACTION_BYTE57_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l4(CRC32_IEEE_SLICE16_LUT.0[6][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `action_mode_flags` (`offset 57`) differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ACTION_BYTE57_REF: &[u32; 256] = &CRC32_CRYSTAL_ACTION_BYTE57_LUT.0;

/// 64B-aligned 4KB GF(2) differential CRC32 table for `base_energy` (`offset 24..28`),
/// pre-shifted to the end of the 128-byte `CrystallizedAttractorRecord` via $L^6(\text{indep\_lo1}(x))$.
pub(crate) const CRC32_CRYSTAL_BE24_FINAL_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let c1 = CRC32_IEEE_SLICE16_LUT.0[7 - byte_pos][b];
            lut[byte_pos][b] = crc32_step_l6(c1);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB `base_energy` (`offset 24..28`) final 128B CRC32 table.
pub(crate) const CRC32_CRYSTAL_BE24_FINAL_REF: &[[u32; 256]; 4] = &CRC32_CRYSTAL_BE24_FINAL_LUT.0;

/// 64B-aligned 4KB GF(2) differential CRC32 table for `phase_gravity_mass` (`offset 36..40`),
/// pre-shifted to the end of the 128-byte `CrystallizedAttractorRecord` via $L^5(\text{indep\_hi0}(x))$.
pub(crate) const CRC32_CRYSTAL_GM36_FINAL_LUT: AlignedLut4x256<u32> = {
    let mut lut = [[0u32; 256]; 4];
    let mut byte_pos = 0usize;
    while byte_pos < 4 {
        let mut b = 0usize;
        while b < 256 {
            let c2 = CRC32_IEEE_SLICE16_LUT.0[11 - byte_pos][b];
            lut[byte_pos][b] = crc32_step_l5(c2);
            b += 1;
        }
        byte_pos += 1;
    }
    AlignedLut4x256(lut)
};

/// Promoted `.rodata` reference to the 4KB `phase_gravity_mass` (`offset 36..40`) final 128B CRC32 table.
pub(crate) const CRC32_CRYSTAL_GM36_FINAL_REF: &[[u32; 256]; 4] = &CRC32_CRYSTAL_GM36_FINAL_LUT.0;

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 24..28`
/// (`base_energy`) in a single 4-lookup parallel stage (`0` serial $L^6$ dependency).
#[inline(always)]
pub(crate) const fn crc32_crystal_be24_final_u32(be_bits: u32) -> u32 {
    let lut = CRC32_CRYSTAL_BE24_FINAL_REF;
    (lut[0][(be_bits & 0xFF) as usize] ^ lut[1][((be_bits >> 8) & 0xFF) as usize])
        ^ (lut[2][((be_bits >> 16) & 0xFF) as usize] ^ lut[3][(be_bits >> 24) as usize])
}

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 36..40`
/// (`phase_gravity_mass`) in a single 4-lookup parallel stage (`0` serial $L^5$ dependency).
#[inline(always)]
pub(crate) const fn crc32_crystal_gm36_final_u32(gm_bits: u32) -> u32 {
    let lut = CRC32_CRYSTAL_GM36_FINAL_REF;
    (lut[0][(gm_bits & 0xFF) as usize] ^ lut[1][((gm_bits >> 8) & 0xFF) as usize])
        ^ (lut[2][((gm_bits >> 16) & 0xFF) as usize] ^ lut[3][(gm_bits >> 24) as usize])
}

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 16..20`
/// (`zimms_section_offset48 as u32`) using the L1D-resident `CRC32_IEEE_SLICE16_REF` and `CRC32_IEEE_L6_REF` tables.
#[inline(always)]
pub(crate) const fn crc32_crystal_zoff16_final_u32(zoff_lo: u32) -> u32 {
    if zoff_lo == 0 {
        0
    } else {
        crc32_step_l6(crc32_slice16_dep_lo0(zoff_lo))
    }
}

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 40..44`
/// (`ripa_clamp_rad`) using the L1D-resident `CRC32_IEEE_SLICE16_REF` and `CRC32_IEEE_L5_REF` tables.
#[inline(always)]
pub(crate) const fn crc32_crystal_cl40_final_u32(cl_bits: u32) -> u32 {
    if cl_bits == 0 {
        0
    } else {
        crc32_step_l5(crc32_slice16_indep_lo1_u32(cl_bits))
    }
}

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 44..48`
/// (`effective_inv_temp`) using the L1D-resident `CRC32_IEEE_SLICE16_REF` and `CRC32_IEEE_L5_REF` tables.
#[inline(always)]
pub(crate) const fn crc32_crystal_tm44_final_u32(tm_bits: u32) -> u32 {
    if tm_bits == 0 {
        0
    } else {
        crc32_step_l5(crc32_slice16_indep_hi1_u32(tm_bits))
    }
}

#[inline(always)]
pub(crate) const fn crc32_slice16_indep_lo1_u32(lo1: u32) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    (lut[7][(lo1 & 0xFF) as usize] ^ lut[6][((lo1 >> 8) & 0xFF) as usize])
        ^ (lut[5][((lo1 >> 16) & 0xFF) as usize] ^ lut[4][(lo1 >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_slice16_indep_hi0_u32(hi0: u32) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    (lut[11][(hi0 & 0xFF) as usize] ^ lut[10][((hi0 >> 8) & 0xFF) as usize])
        ^ (lut[9][((hi0 >> 16) & 0xFF) as usize] ^ lut[8][(hi0 >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_slice16_indep_hi1_u32(hi1: u32) -> u32 {
    let lut = CRC32_IEEE_SLICE16_REF;
    (lut[3][(hi1 & 0xFF) as usize] ^ lut[2][((hi1 >> 8) & 0xFF) as usize])
        ^ (lut[1][((hi1 >> 16) & 0xFF) as usize] ^ lut[0][(hi1 >> 24) as usize])
}

#[inline(always)]
pub(crate) const fn crc32_slice16_indep_u64x2(w0_le: u64, w1_le: u64) -> u32 {
    let hi0 = (w0_le >> 32) as u32;
    let c_hi0 = if hi0 == 0 {
        0
    } else {
        crc32_slice16_indep_hi0_u32(hi0)
    };
    if w1_le == 0 {
        return c_hi0;
    }
    let lo1 = w1_le as u32;
    let hi1 = (w1_le >> 32) as u32;
    c_hi0
        ^ (if lo1 == 0 {
            0
        } else {
            crc32_slice16_indep_lo1_u32(lo1)
        })
        ^ (if hi1 == 0 {
            0
        } else {
            crc32_slice16_indep_hi1_u32(hi1)
        })
}

/// Evaluates the accumulator-independent 128-byte stamped crystal contribution
/// $L^4(L_{12}(\text{record\_crc32})) \oplus (\neg\text{record\_crc32}) \oplus L^8(\text{0xFFFF\_FFFF})$
/// in 4 parallel table lookups (`0` branches).
#[inline(always)]
pub(crate) const fn crc32_crystal_stamp_contrib_128b(record_crc32: u32) -> u32 {
    let lut = CRC32_IEEE_CRYSTAL_STAMP_REF;
    (lut[0][(record_crc32 & 0xFF) as usize] ^ lut[1][((record_crc32 >> 8) & 0xFF) as usize])
        ^ (lut[2][((record_crc32 >> 16) & 0xFF) as usize]
            ^ lut[3][(record_crc32 >> 24) as usize])
}

/// Advances an uninverted IEEE 802.3 CRC32 accumulator (`crc`) across an entire 128-byte stamped
/// `CrystallizedAttractorRecord` using GF(2) linearity ($L^8(\text{crc}) \oplus \text{stamp}(\text{record\_crc32})$)
/// in a single 4-lookup stage (`0` branches).
#[inline(always)]
pub(crate) const fn crc32_fold_stamped_crystal_128b(crc: u32, record_crc32: u32) -> u32 {
    crc32_step_l8(crc) ^ crc32_crystal_stamp_contrib_128b(record_crc32)
}

/// Updates a stamped `CrystallizedAttractorRecord`'s IEEE 802.3 `checksum_crc32` in $\mathcal{O}(1)$
/// via GF(2) linearity when `phase_profile` (`offset 64..128`), `anchor_id`, `zimms_section_offset48`,
/// `resultant_r_bar`, and `von_mises_kappa` remained stationary while `sample_count` (`offset 12..16`),
/// `base_energy` (`offset 24..28`), `phase_gravity_mass` (`offset 36..40`), `ripa_clamp_rad`
/// (`offset 40..44`), `effective_inv_temp` (`offset 44..48`), `provenance_bits` (`offset 48..52`),
/// or `w7_lo` (`dominant_slot` / `action_mode_flags` at `offset 56..58`) changed.
#[inline(always)]
pub(crate) const fn crc32_update_crystal_stationary_fields_delta(
    old_crc32: u32,
    delta_sample_count: u32,
    delta_base_energy_bits: u32,
    delta_gravity_mass_bits: u32,
    delta_clamp_bits: u32,
    delta_inv_temp_bits: u32,
    delta_provenance_bits: u32,
    delta_w7_lo: u32,
) -> u32 {
    let sc_fast = (delta_sample_count >> 8) == 0;
    if sc_fast
        & (((delta_base_energy_bits | delta_gravity_mass_bits)
            | (delta_clamp_bits | delta_inv_temp_bits)
            | (delta_provenance_bits | delta_w7_lo))
            == 0)
    {
        return old_crc32 ^ CRC32_CRYSTAL_SC_BYTE0_REF[delta_sample_count as usize];
    }
    let prov_fast = (delta_provenance_bits >> 16) == 0;
    let mut direct_xor = if sc_fast {
        CRC32_CRYSTAL_SC_BYTE0_REF[delta_sample_count as usize]
    } else {
        0
    };
    if delta_base_energy_bits != 0 {
        direct_xor ^= crc32_crystal_be24_final_u32(delta_base_energy_bits);
    }
    if delta_gravity_mass_bits != 0 {
        direct_xor ^= crc32_crystal_gm36_final_u32(delta_gravity_mass_bits);
    }
    if delta_clamp_bits != 0 {
        direct_xor ^= crc32_crystal_cl40_final_u32(delta_clamp_bits);
    }
    if delta_inv_temp_bits != 0 {
        direct_xor ^= crc32_crystal_tm44_final_u32(delta_inv_temp_bits);
    }
    if delta_w7_lo != 0 {
        let slot_contrib = if (delta_w7_lo & 0xF8) == 0 {
            CRC32_CRYSTAL_SLOT_BYTE56_LUT_8[(delta_w7_lo & 7) as usize]
        } else {
            crc32_step_l4(crc32_slice16_indep_lo1_u32(delta_w7_lo & 0xFF))
        };
        direct_xor ^= slot_contrib
            ^ CRC32_CRYSTAL_ACTION_BYTE57_REF[((delta_w7_lo >> 8) & 0xFF) as usize];
    }
    if prov_fast {
        direct_xor ^= CRC32_CRYSTAL_PROV_BYTE0_REF[(delta_provenance_bits & 0xFF) as usize]
            ^ CRC32_CRYSTAL_PROV_BYTE1_REF[((delta_provenance_bits >> 8) & 0xFF) as usize];
        if sc_fast {
            return old_crc32 ^ direct_xor;
        }
    }
    let sc_l2 = if sc_fast {
        0
    } else {
        crc32_step_l2(crc32_slice16_indep_hi1_u32(delta_sample_count))
    };
    let prov_rem = if prov_fast { 0 } else { delta_provenance_bits };
    let l5_in = sc_l2 ^ prov_rem;
    let l5_out = if l5_in == 0 { 0 } else { crc32_step_l5(l5_in) };
    old_crc32 ^ direct_xor ^ l5_out
}

/// Updates a stamped `CrystallizedAttractorRecord`'s IEEE 802.3 `checksum_crc32` in $\mathcal{O}(1)$
/// via GF(2) linearity when only `base_energy` (`offset 24..28`) and `phase_gravity_mass`
/// (`offset 36..40`) changed (such as during multiplicative evaporative decay in
/// `merge_and_prune_centroids_64`), using 8 parallel table lookups with `0` branches.
#[inline(always)]
pub(crate) const fn crc32_update_crystal_energy_and_mass_delta(
    old_crc32: u32,
    delta_base_energy_bits: u32,
    delta_gravity_mass_bits: u32,
) -> u32 {
    old_crc32
        ^ crc32_crystal_be24_final_u32(delta_base_energy_bits)
        ^ crc32_crystal_gm36_final_u32(delta_gravity_mass_bits)
}

/// Precomputed Slice-by-16 accumulator contribution for the invariant 8-byte header
/// `w0 = (GCSO_CRYSTAL_RECORD_MAGIC as u64) | ((GCSO_ABI_VERSION_HEX as u64) << 32)`
/// XORed with the initial IEEE 802.3 seed `0xFFFF_FFFF`, eliminating 8 table lookups on every
/// `CrystallizedAttractorRecord` checksum evaluation.
pub(crate) const CRYSTAL_CRC32_W0_SEED_CONTRIB: u32 =
    crc32_slice16_dep_lo0(0x4C54_5343u32 ^ 0xFFFF_FFFFu32)
        ^ crc32_slice16_indep_hi0_u32(crate::abi::GCSO_ABI_VERSION_HEX);

/// Precomputed Slice-by-16 accumulator contribution for the default `w5` (`ripa_clamp_rad = 0.24`,
/// `effective_inv_temp = 1.0`) field pair inside a `CrystallizedAttractorRecord`, eliminating 8 table lookups.
pub(crate) const CRYSTAL_CRC32_DEFAULT_W5_CONTRIB: u32 =
    crc32_slice16_indep_lo1_u32(DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
        ^ crc32_slice16_indep_hi1_u32(0x3F80_0000u32);

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^7(\text{CRYSTAL\_CRC32\_W0\_SEED\_CONTRIB})$
/// for the invariant `(GCSO_CRYSTAL_RECORD_MAGIC, GCSO_CRYSTAL_RECORD_VERSION)` header and initial seed.
pub(crate) const CRYSTAL_CRC32_W0_FINAL_CONTRIB: u32 =
    crc32_step_l4(crc32_step_l3(CRYSTAL_CRC32_W0_SEED_CONTRIB));

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^5(\text{CRYSTAL\_CRC32\_DEFAULT\_W5\_CONTRIB})$
/// for the default `w5` (`ripa_clamp_rad = 0.24`, `effective_inv_temp = 1.0`) field pair.
pub(crate) const CRYSTAL_CRC32_DEFAULT_W5_FINAL_CONTRIB: u32 =
    crc32_step_l5(CRYSTAL_CRC32_DEFAULT_W5_CONTRIB);

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^6(\text{indep\_hi1}(\text{0x3F80\_0000}))$
/// for `resultant_r_bar == 1.0` (`offset 28..32`), eliminating 4 table lookups on coherent centroids.
pub(crate) const CRYSTAL_CRC32_RBAR_ONE_FINAL_CONTRIB: u32 =
    crc32_step_l6(crc32_slice16_indep_hi1_u32(0x3F80_0000u32));

/// Exact IEEE-754 `f32::to_bits()` representation of `VON_MISES_KAPPA_AT_MAX_R` ($\bar{R}_c \ge 0.995$).
pub(crate) const VON_MISES_KAPPA_AT_MAX_R_BITS: u32 = VON_MISES_KAPPA_AT_MAX_R.to_bits();

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^5(\text{dep\_lo0}(\text{VON\_MISES\_KAPPA\_AT\_MAX\_R\_BITS}))$
/// for `von_mises_kappa == VON_MISES_KAPPA_AT_MAX_R` (`offset 32..36`), eliminating 4 table lookups
/// on every byte-uniform or maximally coherent centroid ($\bar{R}_c \ge 0.995$).
pub(crate) const CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB: u32 =
    crc32_step_l5(crc32_slice16_dep_lo0(VON_MISES_KAPPA_AT_MAX_R_BITS));

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^5(\text{dep\_lo0}(\text{0x4248\_0000}))$ for
/// `von_mises_kappa == 50.0` (`offset 32..36`), eliminating 4 table lookups.
pub(crate) const CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB: u32 =
    crc32_step_l5(crc32_slice16_dep_lo0(0x4248_0000u32));

/// Precomputed 64-byte GF(2) shift $L^4(\text{0xFFFF\_FFFF})$ of the initial IEEE 802.3 CRC32 seed.
pub(crate) const CRC32_L4_INITIAL_SEED: u32 = crc32_step_l4(0xFFFF_FFFFu32);

/// Precomputed 64-byte IEEE 802.3 CRC32 accumulator state over the invariant fields of a default
/// 4-section `.gcso` `gcso_zimms_descriptor_t` (`mapped_address = 0`, `file_size_bytes = 0`,
/// `dma_buffer_handle = 0`, `flags = 4`, `fd_handle = -1`, `reserved = [0; 32]`).
pub(crate) const CRC32_ZIMMS_DESC_BASE_SEED: u32 =
    CRC32_L4_INITIAL_SEED ^ crc32_step_l2(crc32_slice16_indep_u64x2(0, 0xFFFF_FFFF_0000_0004u64));

/// Evaluates the 64-byte header (`offset 0..64`) GF(2) differential IEEE 802.3 CRC32 state
/// `(final_128b_xor, c3_at_offset_64)` using strictly the L1D-resident `CRC32_IEEE_SLICE16_REF`
/// table and scalar constants (`0` separate 4KB `FINAL_LUT` table accesses), allowing `c3` to fold
/// directly into `phase_profile` word 0 (`dw[0]`) with zero `crc32_step_l4` overhead.
#[inline(always)]
pub(crate) const fn crc32_crystal_header_delta_c3(
    delta_anchor_id: u32,
    delta_sample_count: u32,
    delta_zimms_offset48: u64,
    delta_base_energy_bits: u32,
    delta_r_bar_bits: u32,
    delta_kappa_bits: u32,
    delta_gravity_mass_bits: u32,
    delta_clamp_bits: u32,
    delta_inv_temp_bits: u32,
    delta_provenance_bits: u32,
    delta_w7_lo: u32,
) -> (u32, u32) {
    let mut final_xor = 0u32;

    if delta_base_energy_bits != 0 {
        final_xor ^= crc32_crystal_be24_final_u32(delta_base_energy_bits);
    }
    if delta_gravity_mass_bits != 0 {
        final_xor ^= crc32_crystal_gm36_final_u32(delta_gravity_mass_bits);
    }

    let c0 = if ((delta_anchor_id >> 16) == 0) & ((delta_sample_count >> 8) == 0) {
        final_xor ^= CRC32_CRYSTAL_AID_BYTE0_REF[(delta_anchor_id & 0xFF) as usize]
            ^ CRC32_CRYSTAL_AID_BYTE1_REF[((delta_anchor_id >> 8) & 0xFF) as usize]
            ^ CRC32_CRYSTAL_SC_BYTE0_REF[delta_sample_count as usize];
        0
    } else {
        let aid_c = if (delta_anchor_id >> 16) == 0 {
            final_xor ^= CRC32_CRYSTAL_AID_BYTE0_REF[(delta_anchor_id & 0xFF) as usize]
                ^ CRC32_CRYSTAL_AID_BYTE1_REF[((delta_anchor_id >> 8) & 0xFF) as usize];
            0
        } else {
            crc32_slice16_indep_lo1_u32(delta_anchor_id)
        };
        let sc_c = if (delta_sample_count >> 8) == 0 {
            final_xor ^= CRC32_CRYSTAL_SC_BYTE0_REF[delta_sample_count as usize];
            0
        } else {
            crc32_slice16_indep_hi1_u32(delta_sample_count)
        };
        aid_c ^ sc_c
    };

    let z_lo = (delta_zimms_offset48 as u32) ^ c0;
    let z_hi = (delta_zimms_offset48 >> 32) as u32;
    let z_c = if (z_lo | z_hi) == 0 {
        0
    } else if (c0 == 0) & (z_hi == 0) & ((z_lo & 0xFF) == 0) {
        final_xor ^= CRC32_CRYSTAL_ZOFF_BYTE1_REF[((z_lo >> 8) & 0xFF) as usize]
            ^ CRC32_CRYSTAL_ZOFF_BYTE2_REF[((z_lo >> 16) & 0xFF) as usize]
            ^ CRC32_CRYSTAL_ZOFF_BYTE3_REF[(z_lo >> 24) as usize];
        0
    } else if z_hi == 0 {
        crc32_slice16_dep_lo0(z_lo)
    } else {
        crc32_slice16_dep_lo0(z_lo) ^ crc32_slice16_indep_hi0_u32(z_hi)
    };
    let rb_c = if delta_r_bar_bits == 0 {
        0
    } else if delta_r_bar_bits == 0x3F80_0000u32 {
        final_xor ^= CRYSTAL_CRC32_RBAR_ONE_FINAL_CONTRIB;
        0
    } else {
        crc32_slice16_indep_hi1_u32(delta_r_bar_bits)
    };
    let c1 = z_c ^ rb_c;
    if c1 != 0 {
        final_xor ^= crc32_step_l6(c1);
    }

    let kp_c = if delta_kappa_bits == 0 {
        0
    } else if delta_kappa_bits == VON_MISES_KAPPA_AT_MAX_R_BITS {
        final_xor ^= CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB;
        0
    } else if delta_kappa_bits == 0x4248_0000u32 {
        final_xor ^= CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB;
        0
    } else {
        crc32_slice16_dep_lo0(delta_kappa_bits)
    };
    let cl_c = if delta_clamp_bits == 0 {
        0
    } else {
        crc32_slice16_indep_lo1_u32(delta_clamp_bits)
    };
    let tm_c = if delta_inv_temp_bits == 0 {
        0
    } else {
        crc32_slice16_indep_hi1_u32(delta_inv_temp_bits)
    };
    let c2 = kp_c ^ cl_c ^ tm_c;
    if c2 != 0 {
        final_xor ^= crc32_step_l5(c2);
    }

    let prov_c = if (delta_provenance_bits >> 16) == 0 {
        final_xor ^= CRC32_CRYSTAL_PROV_BYTE0_REF[(delta_provenance_bits & 0xFF) as usize]
            ^ CRC32_CRYSTAL_PROV_BYTE1_REF[((delta_provenance_bits >> 8) & 0xFF) as usize];
        0
    } else {
        crc32_slice16_dep_lo0(delta_provenance_bits)
    };
    let w7_c = if ((delta_w7_lo >> 16) == 0) & ((delta_w7_lo & 0xF8) == 0) {
        final_xor ^= CRC32_CRYSTAL_SLOT_BYTE56_LUT_8[(delta_w7_lo & 7) as usize]
            ^ CRC32_CRYSTAL_ACTION_BYTE57_REF[((delta_w7_lo >> 8) & 0xFF) as usize];
        0
    } else {
        crc32_slice16_indep_lo1_u32(delta_w7_lo)
    };
    (final_xor, prov_c ^ w7_c)
}

/// Updates a stamped `CrystallizedAttractorRecord`'s IEEE 802.3 `checksum_crc32` via GF(2)
/// linearity across any absorption step, folding `c3` directly into `dw[0]` via
/// `crc32_step_block64_words(c3, &dw)` when `phase_profile` changed (`0` `crc32_step_l4` lookups).
#[inline(always)]
pub(crate) const fn crc32_update_crystal_absorption_delta(
    old_crc32: u32,
    delta_anchor_id: u32,
    delta_sample_count: u32,
    delta_zimms_offset48: u64,
    delta_base_energy_bits: u32,
    delta_r_bar_bits: u32,
    delta_kappa_bits: u32,
    delta_gravity_mass_bits: u32,
    delta_clamp_bits: u32,
    delta_inv_temp_bits: u32,
    delta_provenance_bits: u32,
    delta_w7_lo: u32,
    cur_w: &[u64; 8],
    next_w: &[u64; 8],
) -> u32 {
    let (final_xor, c3) = crc32_crystal_header_delta_c3(
        delta_anchor_id,
        delta_sample_count,
        delta_zimms_offset48,
        delta_base_energy_bits,
        delta_r_bar_bits,
        delta_kappa_bits,
        delta_gravity_mass_bits,
        delta_clamp_bits,
        delta_inv_temp_bits,
        delta_provenance_bits,
        delta_w7_lo,
    );

    if cur_w[0] == next_w[0]
        && ((((cur_w[1] ^ next_w[1]) | (cur_w[2] ^ next_w[2]))
            | ((cur_w[3] ^ next_w[3]) | (cur_w[4] ^ next_w[4])))
            | (((cur_w[5] ^ next_w[5]) | (cur_w[6] ^ next_w[6])) | (cur_w[7] ^ next_w[7])))
            == 0
    {
        return old_crc32 ^ final_xor ^ (if c3 == 0 { 0 } else { crc32_step_l4(c3) });
    }
    let dw = [
        cur_w[0] ^ next_w[0],
        cur_w[1] ^ next_w[1],
        cur_w[2] ^ next_w[2],
        cur_w[3] ^ next_w[3],
        cur_w[4] ^ next_w[4],
        cur_w[5] ^ next_w[5],
        cur_w[6] ^ next_w[6],
        cur_w[7] ^ next_w[7],
    ];
    old_crc32 ^ final_xor ^ crc32_step_block64_words(c3, &dw)
}

#[inline(always)]
pub(crate) const fn crc32_slice16_step_u64x2(crc: u32, w0_le: u64, w1_le: u64) -> u32 {
    let lo0 = (w0_le as u32) ^ crc;
    let base = if lo0 == 0 {
        0
    } else {
        crc32_slice16_dep_lo0(lo0)
    };
    if ((w0_le >> 32) | w1_le) == 0 {
        return base;
    }
    base ^ crc32_slice16_indep_u64x2(w0_le, w1_le)
}

#[inline(always)]
pub(crate) const fn crc32_block16_zero_seed_u64x2(w0_le: u64, w1_le: u64) -> u32 {
    if (w0_le | w1_le) == 0 {
        return 0;
    }
    let lo0 = w0_le as u32;
    let base = if lo0 == 0 {
        0
    } else {
        crc32_slice16_dep_lo0(lo0)
    };
    if ((w0_le >> 32) | w1_le) == 0 {
        return base;
    }
    base ^ crc32_slice16_indep_u64x2(w0_le, w1_le)
}

#[inline(always)]
const fn crc32_uniform_word_zero_seed_raw(w0: u64) -> u32 {
    let p0 = crc32_block16_zero_seed_u64x2(w0, w0);
    let h1 = crc32_slice16_dep_lo0(p0) ^ p0;
    crc32_step_l2(h1) ^ h1
}

/// 64B-aligned 1KB GF(2)-linear lookup table mapping any uniform byte `b in 0..=255` to the exact
/// unseeded 64-byte IEEE 802.3 CRC32 operator $\mathcal{C}_0([b; 64])$ (`1` table lookup instead of `36`).
pub(crate) const CRC32_UNIFORM64_ZERO_SEED_LUT_256: AlignedLut256<u32> = {
    let mut basis = [0u32; 8];
    let mut k = 0usize;
    while k < 8 {
        let w_bit = (1u64 << k).wrapping_mul(SWAR_SPLAT_BYTE_64);
        basis[k] = crc32_uniform_word_zero_seed_raw(w_bit);
        k += 1;
    }
    let mut lut = [0u32; 256];
    let mut b = 1usize;
    while b < 256 {
        let lsb = b.trailing_zeros() as usize;
        lut[b] = lut[b & (b - 1)] ^ basis[lsb];
        b += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 256-entry uniform 64-byte zero-seed CRC32 LUT.
pub(crate) const CRC32_UNIFORM64_ZERO_SEED_REF: &[u32; 256] = &CRC32_UNIFORM64_ZERO_SEED_LUT_256.0;

#[inline(always)]
const fn crc32_uniform56_pad8_zero_seed_raw(w0: u64) -> u32 {
    let u64_crc = crc32_uniform_word_zero_seed_raw(w0);
    let lo7 = w0 as u32;
    let hi7 = (w0 >> 32) as u32;
    u64_crc
        ^ (if lo7 == 0 { 0 } else { crc32_slice16_indep_lo1_u32(lo7) })
        ^ (if hi7 == 0 { 0 } else { crc32_slice16_indep_hi1_u32(hi7) })
}

/// 64B-aligned 1KB GF(2)-linear lookup table mapping any uniform byte `b in 0..=255` to the exact
/// unseeded 64-byte IEEE 802.3 CRC32 operator over 56 uniform bytes followed by 8 zero bytes
/// (`[b; 56] ++ [0; 8]`, matching `gcso_pointer_trail_t` Block 1 `offset 64..128` in `1` lookup instead of `9`).
pub(crate) const CRC32_UNIFORM56_PAD8_ZERO_SEED_LUT_256: AlignedLut256<u32> = {
    let mut basis = [0u32; 8];
    let mut k = 0usize;
    while k < 8 {
        let w_bit = (1u64 << k).wrapping_mul(SWAR_SPLAT_BYTE_64);
        basis[k] = crc32_uniform56_pad8_zero_seed_raw(w_bit);
        k += 1;
    }
    let mut lut = [0u32; 256];
    let mut b = 1usize;
    while b < 256 {
        let lsb = b.trailing_zeros() as usize;
        lut[b] = lut[b & (b - 1)] ^ basis[lsb];
        b += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 256-entry 56-byte-uniform + 8-zero-byte CRC32 LUT.
pub(crate) const CRC32_UNIFORM56_PAD8_ZERO_SEED_REF: &[u32; 256] =
    &CRC32_UNIFORM56_PAD8_ZERO_SEED_LUT_256.0;

/// Evaluates the raw IEEE 802.3 CRC32 operator over a 64-byte (`[u64; 8]`) block starting from
/// `seed == 0` via 4-way independent 16B Slice-by-16 evaluation (`p0, p1, p2, p3` in parallel)
/// followed by GF(2) operator folding (`crc32_step_l3(p0) ^ crc32_step_l2(p1) ^ crc32_slice16_dep_lo0(p2) ^ p3`),
/// reducing serial L1D load dependency depth to 2 stages while fast-pathing byte/word-uniform blocks
/// and `gcso_pointer_trail_t` Block 1 (`56B` uniform phase delta + `8B` zero reserved padding).
/// Evaluates the zero-seed IEEE 802.3 CRC32 of a 64-byte block `[w0, w0, w0, w0, w0, w0, w0, w7]`
/// whose first 7 64-bit words are already known to equal `w0` (`0` duplicate uniformity checks).
#[inline(always)]
#[must_use]
const fn crc32_block64_uniform_prefix56_zero_seed(w0: u64, w7: u64) -> u32 {
    if w0 == 0 {
        if w7 == 0 {
            return 0;
        }
        let lo7 = w7 as u32;
        let hi7 = (w7 >> 32) as u32;
        return (if lo7 == 0 { 0 } else { crc32_slice16_indep_lo1_u32(lo7) })
            ^ (if hi7 == 0 { 0 } else { crc32_slice16_indep_hi1_u32(hi7) });
    }
    let b0 = (w0 & 0xFF) as usize;
    let is_byte_uni = w0 == w0.rotate_right(8);
    if w7 == w0 {
        return if is_byte_uni {
            CRC32_UNIFORM64_ZERO_SEED_REF[b0]
        } else {
            crc32_uniform_word_zero_seed_raw(w0)
        };
    }
    if w7 == 0 && is_byte_uni {
        return CRC32_UNIFORM56_PAD8_ZERO_SEED_REF[b0];
    }
    let u64_crc = if is_byte_uni {
        CRC32_UNIFORM64_ZERO_SEED_REF[b0]
    } else {
        crc32_uniform_word_zero_seed_raw(w0)
    };
    let dw7 = w0 ^ w7;
    let lo7 = dw7 as u32;
    let hi7 = (dw7 >> 32) as u32;
    u64_crc
        ^ (if lo7 == 0 { 0 } else { crc32_slice16_indep_lo1_u32(lo7) })
        ^ (if hi7 == 0 { 0 } else { crc32_slice16_indep_hi1_u32(hi7) })
}

/// Evaluates the raw IEEE 802.3 CRC32 operator over 8 native `u64` words (64 bytes)
/// starting from initial seed `0`, returning the unfinalized accumulator state.
#[inline(always)]
#[must_use]
pub const fn crc32_block64_zero_seed_words(w: &[u64; 8]) -> u32 {
    let w0 = w[0];
    if w0 == w[1]
        && (((w0 ^ w[2]) | (w0 ^ w[3]))
            | ((w0 ^ w[4]) | (w0 ^ w[5]))
            | (w0 ^ w[6]))
            == 0
    {
        return crc32_block64_uniform_prefix56_zero_seed(w0, w[7]);
    }
    let w1 = w[1];
    let p0 = crc32_block16_zero_seed_u64x2(w0, w1);
    if (w0 == w[2])
        & (w1 == w[3])
        & ((((w0 ^ w[4]) | (w1 ^ w[5])) | ((w0 ^ w[6]) | (w1 ^ w[7]))) == 0)
    {
        let p01 = crc32_slice16_dep_lo0(p0) ^ p0;
        return crc32_step_l2(p01) ^ p01;
    }
    let p1 = crc32_block16_zero_seed_u64x2(w[2], w[3]);
    let p2 = crc32_block16_zero_seed_u64x2(w[4], w[5]);
    let p3 = crc32_block16_zero_seed_u64x2(w[6], w[7]);
    let s0 = if p0 == 0 { 0 } else { crc32_step_l3(p0) };
    let s1 = if p1 == 0 { 0 } else { crc32_step_l2(p1) };
    let s2 = if p2 == 0 { 0 } else { crc32_slice16_dep_lo0(p2) };
    (s0 ^ s1) ^ (s2 ^ p3)
}

#[inline(always)]
pub(crate) const fn crc32_step_block64_words(crc: u32, w: &[u64; 8]) -> u32 {
    if crc == 0 {
        return crc32_block64_zero_seed_words(w);
    }
    let w0 = w[0];
    if w0 == w[1]
        && (((w0 ^ w[2]) | (w0 ^ w[3]))
            | ((w0 ^ w[4]) | (w0 ^ w[5]))
            | (w0 ^ w[6]))
            == 0
    {
        let shifted_seed = if crc == 0xFFFF_FFFFu32 {
            CRC32_L4_INITIAL_SEED
        } else {
            crc32_step_l4(crc)
        };
        return shifted_seed ^ crc32_block64_uniform_prefix56_zero_seed(w0, w[7]);
    }
    let p0 = crc32_slice16_step_u64x2(crc, w0, w[1]);
    let p1 = crc32_block16_zero_seed_u64x2(w[2], w[3]);
    let p2 = crc32_block16_zero_seed_u64x2(w[4], w[5]);
    let p3 = crc32_block16_zero_seed_u64x2(w[6], w[7]);
    let s0 = if p0 == 0 { 0 } else { crc32_step_l3(p0) };
    let s1 = if p1 == 0 { 0 } else { crc32_step_l2(p1) };
    let s2 = if p2 == 0 { 0 } else { crc32_slice16_dep_lo0(p2) };
    (s0 ^ s1) ^ (s2 ^ p3)
}

/// Computes the IEEE 802.3 CRC32 checksum over 16 little-endian 64-bit words (128 bytes) using
/// GF(2) parallel $2 \times 64\text{B}$ tree folding (`L^4(block0) ^ block1`), cutting the serial
/// dependency depth from 8 stages down to 2 and leveraging `CRC32_UNIFORM64_ZERO_SEED_REF`.
#[inline]
#[must_use]
pub const fn branchless_crc32_128b_words(w: &[u64; 16]) -> u32 {
    let w0 = [w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]];
    let w1 = [w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15]];
    let block0_crc = crc32_step_block64_words(0xFFFF_FFFFu32, &w0);
    let block1_crc = crc32_block64_zero_seed_words(&w1);
    !(crc32_step_l4(block0_crc) ^ block1_crc)
}

#[inline(always)]
const fn load_u64_le_bytes_at(bytes: &[u8], i: usize) -> u64 {
    u64::from_le_bytes([
        bytes[i],
        bytes[i + 1],
        bytes[i + 2],
        bytes[i + 3],
        bytes[i + 4],
        bytes[i + 5],
        bytes[i + 6],
        bytes[i + 7],
    ])
}

/// Incrementally updates an IEEE 802.3 CRC32 checksum (`crc_seed`) over `bytes` using the
/// 64B-aligned 16KB Slice-by-16 `CRC32_IEEE_SLICE16_LUT` table and GF(2) $2 \times 64\text{B}$ /
/// $2 \times 32\text{B}$ parallel tree reduction (`crc32_step_block64_words` + `crc32_step_l4`).
#[inline]
pub const fn branchless_update_crc32_ieee(crc_seed: u32, bytes: &[u8]) -> u32 {
    let mut crc = !crc_seed;
    let len = bytes.len();
    let mut i = 0usize;
    while i + 128 <= len {
        let w0 = [
            load_u64_le_bytes_at(bytes, i),
            load_u64_le_bytes_at(bytes, i + 8),
            load_u64_le_bytes_at(bytes, i + 16),
            load_u64_le_bytes_at(bytes, i + 24),
            load_u64_le_bytes_at(bytes, i + 32),
            load_u64_le_bytes_at(bytes, i + 40),
            load_u64_le_bytes_at(bytes, i + 48),
            load_u64_le_bytes_at(bytes, i + 56),
        ];
        let w1 = [
            load_u64_le_bytes_at(bytes, i + 64),
            load_u64_le_bytes_at(bytes, i + 72),
            load_u64_le_bytes_at(bytes, i + 80),
            load_u64_le_bytes_at(bytes, i + 88),
            load_u64_le_bytes_at(bytes, i + 96),
            load_u64_le_bytes_at(bytes, i + 104),
            load_u64_le_bytes_at(bytes, i + 112),
            load_u64_le_bytes_at(bytes, i + 120),
        ];
        let c0 = crc32_step_block64_words(crc, &w0);
        let c1 = crc32_block64_zero_seed_words(&w1);
        crc = crc32_step_l4(c0) ^ c1;
        i += 128;
    }
    while i + 64 <= len {
        let w = [
            load_u64_le_bytes_at(bytes, i),
            load_u64_le_bytes_at(bytes, i + 8),
            load_u64_le_bytes_at(bytes, i + 16),
            load_u64_le_bytes_at(bytes, i + 24),
            load_u64_le_bytes_at(bytes, i + 32),
            load_u64_le_bytes_at(bytes, i + 40),
            load_u64_le_bytes_at(bytes, i + 48),
            load_u64_le_bytes_at(bytes, i + 56),
        ];
        crc = crc32_step_block64_words(crc, &w);
        i += 64;
    }
    while i + 16 <= len {
        let w0 = load_u64_le_bytes_at(bytes, i);
        let w1 = load_u64_le_bytes_at(bytes, i + 8);
        crc = crc32_slice16_step_u64x2(crc, w0, w1);
        i += 16;
    }
    if i + 8 <= len {
        let w = load_u64_le_bytes_at(bytes, i);
        crc = crc32_slice8_step_u64(crc, w);
        i += 8;
    }
    if i + 4 <= len {
        crc = crc32_slice4_step(crc, bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]);
        i += 4;
    }
    while i < len {
        crc = (crc >> 8) ^ CRC32_IEEE_SLICE16_REF[0][((crc as u8) ^ bytes[i]) as usize];
        i += 1;
    }
    !crc
}

/// Computes the standard IEEE 802.3 CRC32 checksum over `bytes` using 8KB Slice-by-8 table lookups.
#[inline(always)]
pub const fn branchless_crc32_ieee(bytes: &[u8]) -> u32 {
    branchless_update_crc32_ieee(0, bytes)
}

/// Fast positive normal reciprocal $1.0 / x$ for $x \ge \text{F32\_MIN\_POSITIVE\_NORMAL}$
/// using IEEE-754 magic bit seed and 3 Newton-Raphson iterations (`0` subnormal/validity branches).
#[inline(always)]
pub(crate) const fn fast_inv_pos_normal_f32(x: f32) -> f32 {
    let y0 = f32::from_bits(0x7EF1_27EA_u32.wrapping_sub(x.to_bits()));
    let y1 = y0 * (2.0 - x * y0);
    let e1 = 1.0 - x * y1;
    y1 + y1 * (e1 + e1 * e1)
}

const LN_1P_LUT_STEP: f32 = 0.0625;
const LN_1P_LUT_INV_STEP: f32 = 16.0;

#[inline(always)]
const fn const_ln_1p_exact_f32(rho_nonneg: f32) -> f32 {
    if rho_nonneg <= 0.0 {
        return 0.0;
    }
    let val = 1.0 + rho_nonneg;
    let bits = val.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    let u = (mantissa - 1.0) / (mantissa + 1.0);
    let u2 = u * u;
    let poly = u
        * (2.0
            + u2 * (0.666_666_7
                + u2 * (0.4 + u2 * (0.285_714_3 + u2 * (0.222_222_22 + u2 * 0.181_818_19)))));
    (exponent as f32) * LN_2 + poly
}

/// 64-byte aligned 257-entry `(ln(1 + x0), 1 / (1 + x0))` lookup table over $[0.0, 16.0]$
/// with exact grid spacing $h = 1/16 = 0.0625$.
pub(crate) const LN_1P_LUT_257: AlignedLut257<(f32, f32)> = {
    let mut lut = [(0.0_f32, 1.0_f32); 257];
    let mut k = 1usize;
    while k <= 256 {
        let x0 = (k as f32) * LN_1P_LUT_STEP;
        lut[k] = (const_ln_1p_exact_f32(x0), 1.0 / (1.0 + x0));
        k += 1;
    }
    AlignedLut257(lut)
};

/// Promoted `.rodata` reference to the 64B-aligned 257-entry `(ln(1 + x0), 1 / (1 + x0))` table.
pub(crate) const LN_1P_LUT_257_REF: &[(f32, f32); 257] = &LN_1P_LUT_257.0;

/// Fast branchless natural logarithm $\ln(1 + \rho)$ for bounded non-negative density $\rho \in [0, 16]$
/// using `LN_1P_LUT_257` and 2nd-order Taylor interpolation (`0` branches, `0` NR reciprocals).
#[inline(always)]
pub(crate) const fn fast_ln_1p_bounded_f32(rho_clamped: f32) -> f32 {
    let clamped = branchless_clamp_f32(rho_clamped, 0.0, 16.0);
    let idx_raw = (clamped * LN_1P_LUT_INV_STEP) as usize;
    let idx = branchless_select_usize(idx_raw <= 256, idx_raw, 256);
    let d = clamped - (idx as f32) * LN_1P_LUT_STEP;
    let (y0, inv_1px0) = LN_1P_LUT_257_REF[idx];
    let u = d * inv_1px0;
    y0 + u * (1.0 - 0.5 * u)
}

/// Precomputed maturity gravity scale $1.0 + 0.10 \ln(1 + \min(N, 16))$ for integer sample counts $N \in [0, 16]$.
pub(crate) const MATURITY_SCALE_LUT_17: [f32; 17] = {
    let mut lut = [1.0_f32; 17];
    let mut n = 1usize;
    while n <= 16 {
        lut[n] = 1.0 + 0.10 * LN_1P_LUT_257.0[n << 4].0;
        n += 1;
    }
    lut
};

/// Evaluates the centroid maturity gravity multiplier $1.0 + 0.10 \ln(1 + \min(N, 16))$ in 1 cycle.
#[inline(always)]
pub(crate) const fn branchless_maturity_scale_u32(sample_count: u32) -> f32 {
    let idx = branchless_select_usize((sample_count as usize) <= 16, sample_count as usize, 16);
    MATURITY_SCALE_LUT_17[idx]
}

#[inline(always)]
pub(crate) const fn fast_inv_len_f32(n: usize) -> f32 {
    if n > 0 && (n & (n - 1)) == 0 && n <= (1usize << 62) {
        f32::from_bits((127u32.wrapping_sub(n.trailing_zeros())) << 23)
    } else if n > 0 && n <= 256 {
        let lut = POPCOUNT_RSQRT_LUT_257_REF;
        let r = lut[n];
        r * r
    } else if n > 0 {
        fast_inv_pos_normal_f32(n as f32)
    } else {
        0.0
    }
}

/// Branchless bitwise selection for 8-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_u8(cond: bool, if_true: u8, if_false: u8) -> u8 {
    let m = (cond as u8).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 8-bit signed integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_i8(cond: bool, if_true: i8, if_false: i8) -> i8 {
    let m = (cond as i8).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 16-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_u16(cond: bool, if_true: u16, if_false: u16) -> u16 {
    let m = (cond as u16).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 32-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_u32(cond: bool, if_true: u32, if_false: u32) -> u32 {
    let m = (cond as u32).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 64-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_u64(cond: bool, if_true: u64, if_false: u64) -> u64 {
    let m = (cond as u64).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for pointer-sized unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_usize(cond: bool, if_true: usize, if_false: usize) -> usize {
    let m = (cond as usize).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 32-bit signed integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_i32(cond: bool, if_true: i32, if_false: i32) -> i32 {
    let m = (cond as i32).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 64-bit signed integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_i64(cond: bool, if_true: i64, if_false: i64) -> i64 {
    let m = (cond as i64).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for Q7 fixed-point integers (`cond ? if_true : if_false`).
#[inline(always)]
pub const fn branchless_select_q7(
    cond: bool,
    if_true: gcso_q7_t,
    if_false: gcso_q7_t,
) -> gcso_q7_t {
    branchless_select_i8(cond, if_true, if_false)
}

/// Branchless selection for IEEE-754 single-precision floats (`cond ? if_true : if_false`),
/// remaining in the floating-point XMM register file (`vminss` / `vmaxss` / `vblendvps`)
/// without XMM-to-GPR (`vmovd`) cross-domain bypass stalls.
#[inline(always)]
pub const fn branchless_select_f32(cond: bool, if_true: f32, if_false: f32) -> f32 {
    if cond {
        if_true
    } else {
        if_false
    }
}

/// Branchless IEEE-754 finiteness check (`!is_nan() && !is_infinite()`) evaluated directly
/// in XMM registers (`vucomiss`) without XMM-to-GPR `vmovd` transfer stalls.
#[inline(always)]
pub const fn branchless_is_finite_f32(x: f32) -> bool {
    (x > -f32::INFINITY) & (x < f32::INFINITY)
}

/// Branchless IEEE-754 `NaN` check evaluated directly in XMM registers (`vucomiss`).
#[inline(always)]
pub const fn branchless_is_nan_f32(x: f32) -> bool {
    x.is_nan()
}

/// Branchlessly flushes IEEE-754 subnormal (denormal) floats, non-finite values (`±Inf`, `NaN`),
/// and `-0.0` to `+0.0`.
#[inline(always)]
pub const fn branchless_flush_subnormal_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    let exp_bits = bits & 0x7F80_0000;
    let is_normal_finite = (exp_bits != 0) & (exp_bits != 0x7F80_0000);
    f32::from_bits(bits & (is_normal_finite as u32).wrapping_neg())
}

/// Branchless IEEE-754 absolute value (`|x|`).
#[inline(always)]
pub const fn branchless_abs_f32(x: f32) -> f32 {
    f32::from_bits(x.to_bits() & 0x7FFF_FFFF)
}

/// Branchless IEEE-754 `copysign(mag, sgn)`.
#[inline(always)]
pub const fn branchless_copysign_f32(mag: f32, sgn: f32) -> f32 {
    f32::from_bits((mag.to_bits() & 0x7FFF_FFFF) | (sgn.to_bits() & 0x8000_0000))
}

/// Branchless signum returning `-1.0`, `0.0`, or `+1.0`.
#[inline(always)]
pub const fn branchless_signum_f32(x: f32) -> f32 {
    let is_nonzero_finite = ((x.to_bits() & 0x7FFF_FFFF) != 0) & (!branchless_is_nan_f32(x));
    let signed_one = branchless_copysign_f32(1.0, x);
    branchless_select_f32(is_nonzero_finite, signed_one, 0.0)
}

/// Branchless Rectified Linear Unit ($\max(0.0, x)$), flushing `NaN`, negatives, and `-0.0` to `+0.0`
/// in a single XMM `vmaxss` comparison (`0` XMM-to-GPR round-trips).
#[inline(always)]
pub const fn branchless_relu_f32(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Branchless `NaN`-resilient IEEE-754 maximum ($\max(a, b)$).
#[inline(always)]
pub const fn branchless_max_f32(a: f32, b: f32) -> f32 {
    if (a > b) | branchless_is_nan_f32(b) {
        a
    } else {
        b
    }
}

/// Branchless `NaN`-resilient IEEE-754 minimum ($\min(a, b)$).
#[inline(always)]
pub const fn branchless_min_f32(a: f32, b: f32) -> f32 {
    if (a < b) | branchless_is_nan_f32(b) {
        a
    } else {
        b
    }
}

/// Branchless interval clamp ($\mathrm{clamp}(x, \mathrm{lo}, \mathrm{hi})$), lowering to a pair
/// of hardware XMM `vmaxss` + `vminss` instructions when `lo` and `hi` are finite bounds.
#[inline(always)]
pub const fn branchless_clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
    let mx = if (x > lo) | branchless_is_nan_f32(lo) {
        x
    } else {
        lo
    };
    if (mx < hi) | branchless_is_nan_f32(hi) {
        mx
    } else {
        hi
    }
}

/// Sanitizes a single-token Z-score input into $[0.0, \text{max\_ceiling}]$.
#[inline(always)]
pub const fn branchless_sanitize_z_score_f32(z_score: f32, max_ceiling: f32) -> f32 {
    let safe_ceil = branchless_select_f32(
        (max_ceiling > 0.0) & (max_ceiling < f32::INFINITY),
        max_ceiling,
        DEFAULT_MAX_VALID_Z_SCORE,
    );
    let clean = branchless_select_f32((z_score > 0.0) & (z_score < f32::INFINITY), z_score, 0.0);
    branchless_min_f32(clean, safe_ceil)
}

/// Branchless linear interpolation: $a + t(b - a)$ for $t \in [0, 1]$.
#[inline(always)]
pub const fn branchless_lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    let tc = branchless_clamp_f32(t, 0.0, 1.0);
    a + tc * (b - a)
}

/// Branchless cubic Hermite smoothstep interpolation ($3u^2 - 2u^3$) for $x \in [\text{edge0}, \text{edge1}]$.
#[inline(always)]
pub const fn branchless_smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let span = edge1 - edge0;
    let valid_span = (span > 1.0e-12) & branchless_is_finite_f32(span);
    let inv_span = branchless_inv_pos_f32(branchless_select_f32(valid_span, span, 1.0));
    let u = branchless_select_f32(
        valid_span,
        branchless_clamp_f32((x - edge0) * inv_span, 0.0, 1.0),
        branchless_select_f32(x >= edge1, 1.0, 0.0),
    );
    u * u * (3.0 - 2.0 * u)
}

/// Branchless $SO(2)$ 2D rotation operator applying $(q_0 \cos\theta - q_1 \sin\theta, q_0 \sin\theta + q_1 \cos\theta)$.
#[inline(always)]
pub const fn branchless_rotate_so2_pair(q0: f32, q1: f32, sin_t: f32, cos_t: f32) -> (f32, f32) {
    (q0 * cos_t - q1 * sin_t, q0 * sin_t + q1 * cos_t)
}

/// Branchless composite DPSR + Isotropic Block-Diagonal Scaling operator (Theorem 1.1).
#[inline(always)]
pub const fn branchless_rotate_so2_pair_scaled(
    q0: f32,
    q1: f32,
    sin_t: f32,
    cos_t: f32,
    scale: f32,
) -> (f32, f32) {
    (
        (q0 * cos_t - q1 * sin_t) * scale,
        (q0 * sin_t + q1 * cos_t) * scale,
    )
}

#[inline(always)]
const fn unpack_pos_f32_exp_and_mant(raw_bits: u32) -> (i32, u32) {
    if raw_bits < 0x0080_0000 {
        let lz = (raw_bits | 1).leading_zeros();
        let shift = lz.wrapping_sub(8);
        (-118i32 - (lz as i32), (raw_bits << shift) & 0x007F_FFFF)
    } else {
        ((((raw_bits >> 23) & 0xFF) as i32) - 127, raw_bits & 0x007F_FFFF)
    }
}

/// Division-free (`0 fdiv`) branchless `#![no_std]` IEEE-754 reciprocal ($1 / x$)
/// for positive finite floats (`val > 0.0`), executing a direct 3-step Newton-Raphson magic-seed
/// reciprocal on normal floats (`1.0e-36 <= val <= 1.0e36`, `0` exponent/mantissa unpacking) and
/// falling back to two-scale exponent reconstruction for subnormal or extreme floats.
#[inline(always)]
pub const fn branchless_inv_pos_f32(val: f32) -> f32 {
    let raw_bits = val.to_bits();
    if (raw_bits >= 0x0416_872B) & (raw_bits <= 0x7B60_435E) {
        return fast_inv_pos_normal_f32(val);
    }
    if (raw_bits == 0) | (raw_bits >= 0x7F80_0000) {
        return 0.0;
    }
    let (e_unbiased, mant23) = unpack_pos_f32_exp_and_mant(raw_bits);
    let m_bits = mant23 | 0x3F80_0000;
    let m = f32::from_bits(m_bits);

    let mut r = f32::from_bits(0x7EF3_11C2_u32.wrapping_sub(m_bits));
    r = r * (2.0 - m * r);
    r = r * (2.0 - m * r);
    r = r * (2.0 - m * r);

    let e_half1 = e_unbiased >> 1;
    let e_half2 = e_unbiased - e_half1;
    let scale1 = f32::from_bits(((127 - e_half1) as u32) << 23);
    let scale2 = f32::from_bits(((127 - e_half2) as u32) << 23);
    branchless_flush_subnormal_f32((r * scale1) * scale2)
}

/// Branchless `#![no_std]` hyperbolic tangent ($\tanh(x)$) using Padé $(5, 4)$ rational approximation.
#[inline(always)]
pub const fn branchless_tanh_f32(x: f32) -> f32 {
    let safe_x = branchless_select_f32(branchless_is_nan_f32(x), 0.0, x);
    let clamped = branchless_clamp_f32(safe_x, -TANH_LUT_MAX_X, TANH_LUT_MAX_X);
    let x2 = clamped * clamped;
    let x4 = x2 * x2;
    (clamped * (945.0 + 105.0 * x2 + x4))
        * fast_inv_pos_normal_f32(945.0 + 420.0 * x2 + 15.0 * x4)
}

/// Branchless `#![no_std]` logistic sigmoid ($\sigma(x) = 1 / (1 + e^{-x})$).
#[inline(always)]
pub const fn branchless_sigmoid_f32(x: f32) -> f32 {
    0.5 + 0.5 * branchless_tanh_f32(0.5 * x)
}

/// Branchless RIPA soft-bounded phase angle clamp: $\Delta\theta_{\mathrm{safe}} = \theta_{\max} \cdot \tanh(\Delta\theta / \theta_{\max})$.
#[inline(always)]
pub const fn branchless_ripa_clamp_rad(delta_rad: f32, max_rad: f32) -> f32 {
    let valid_max = (max_rad > 1.0e-9) & branchless_is_finite_f32(max_rad);
    let safe_max = branchless_select_f32(valid_max, max_rad, DEFAULT_RIPA_CLAMP_MAX_RAD);
    let inv_max = branchless_inv_pos_f32(safe_max);
    branchless_select_f32(
        valid_max,
        safe_max * branchless_tanh_f32(delta_rad * inv_max),
        0.0,
    )
}

/// Branchless `#![no_std]` natural logarithm ($\ln(x)$).
#[inline(always)]
pub const fn branchless_ln_f32(val: f32) -> f32 {
    let bits_raw = val.to_bits();
    if bits_raw == 0x3F80_0000 {
        return 0.0;
    }
    if (bits_raw >= 0x0080_0000) & (bits_raw < 0x7F80_0000) {
        let exponent = ((bits_raw >> 23) & 0xFF) as i32 - 127;
        let mantissa = f32::from_bits((bits_raw & 0x007F_FFFF) | 0x3F80_0000);
        let u = (mantissa - 1.0) * fast_inv_pos_normal_f32(mantissa + 1.0);
        let u2 = u * u;
        let poly =
            u * (2.0 + u2 * (0.666_666_7 + u2 * (0.4 + u2 * (0.285_714_3 + u2 * 0.222_222_2))));
        return (exponent as f32) * LN_2 + poly;
    }
    if (bits_raw == 0) | (bits_raw >= 0x7F80_0000) {
        return 0.0;
    }
    let (exponent, mant23) = unpack_pos_f32_exp_and_mant(bits_raw);
    let mantissa = f32::from_bits(mant23 | 0x3F80_0000);
    let u = (mantissa - 1.0) * fast_inv_pos_normal_f32(mantissa + 1.0);
    let u2 = u * u;
    let poly = u * (2.0 + u2 * (0.666_666_7 + u2 * (0.4 + u2 * (0.285_714_3 + u2 * 0.222_222_2))));
    (exponent as f32) * LN_2 + poly
}

/// Branchless `#![no_std]` fast exponential ($e^x$) using Cody-Waite base-2 range reduction and Padé $(3, 3)$.
#[inline(always)]
pub const fn branchless_exp_f32(x: f32) -> f32 {
    let invalid_or_under = branchless_is_nan_f32(x) | (x < -87.3);
    let clamped =
        branchless_clamp_f32(branchless_select_f32(invalid_or_under, 0.0, x), -87.0, 88.0);
    let k_float = clamped * INV_LN2_F32;
    let k = (k_float + branchless_select_f32(k_float >= 0.0, 0.5, -0.5)) as i32;
    let kf = k as f32;
    let r = (clamped - kf * LN2_HI_F32) - kf * LN2_LO_F32;
    let r2 = r * r;
    let even_part = 120.0 + 12.0 * r2;
    let odd_part = r * (60.0 + r2);
    let exp_r = (even_part + odd_part) * fast_inv_pos_normal_f32(even_part - odd_part);
    let scale_2k = f32::from_bits(((k + 127) as u32) << 23);
    branchless_select_f32(invalid_or_under, 0.0, exp_r * scale_2k)
}

/// Fast division-free (`0 fdiv`) square root $\sqrt{x} \in [0.0, 1.0]$ for bounded unit-interval
/// squared resultant lengths $x = \bar{S}^2 + \bar{C}^2 \in [0.0, 1.01]$ (`0` exponent unpacking).
#[inline(always)]
pub(crate) const fn fast_sqrt_unit_f32(r_sq: f32) -> f32 {
    if r_sq <= 1.0e-12 {
        return 0.0;
    }
    let half_x = 0.5 * r_sq;
    let r0 = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(r_sq.to_bits() >> 1));
    let r = r0 * (1.5 - half_x * r0 * r0);
    let s0 = r_sq * r;
    let y = s0 + (0.5 * r) * (r_sq - s0 * s0);
    branchless_select_f32(y < 1.0, y, 1.0)
}

/// Division-free (`0 fdiv`) branchless `#![no_std]` IEEE-754 square root ($\sqrt{x}$),
/// executing a direct magic-seed 2-step Newton-Raphson + Goldschmidt refinement on normal floats
/// (`F32_MIN_POSITIVE_NORMAL <= val <= 1.0e34`, `0` exponent/mantissa unpacking) and falling back
/// to exponent-normalized reconstruction for subnormal or extreme floats.
#[inline(always)]
pub const fn branchless_sqrt_f32(val: f32) -> f32 {
    let raw_bits = val.to_bits();
    if (raw_bits >= 0x0080_0000) & (raw_bits <= 0x781A_530C) {
        let half_x = 0.5 * val;
        let mut r = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(raw_bits >> 1));
        r = r * (1.5 - half_x * r * r);
        r = r * (1.5 - half_x * r * r);
        let s0 = val * r;
        return s0 + (0.5 * r) * (val - s0 * s0);
    }
    if (raw_bits == 0) | (raw_bits >= 0x7F80_0000) {
        return 0.0;
    }
    let (e_unbiased, mant23) = unpack_pos_f32_exp_and_mant(raw_bits);
    let odd_exp = (e_unbiased & 1) as u32;
    let m_bits = mant23 | ((127 + odd_exp) << 23);
    let m = f32::from_bits(m_bits);
    let half_m = 0.5 * m;

    let mut r = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(m_bits >> 1));
    r = r * (1.5 - half_m * r * r);
    r = r * (1.5 - half_m * r * r);
    r = r * (1.5 - half_m * r * r);

    let s0 = m * r;
    let y = s0 + (0.5 * r) * (m - s0 * s0);
    let half_exp_shift = ((e_unbiased >> 1) as u32) << 23;
    f32::from_bits(y.to_bits().wrapping_add(half_exp_shift))
}

/// Division-free (`0 fdiv`) branchless `#![no_std]` IEEE-754 reciprocal square root ($1 / \sqrt{x}$),
/// executing a direct magic-seed 3-step Newton-Raphson refinement on normal floats
/// (`F32_MIN_POSITIVE_NORMAL <= val <= 1.0e34`, `0` exponent/mantissa unpacking) and falling back
/// to exponent-normalized reconstruction for subnormal or extreme floats.
#[inline(always)]
pub const fn branchless_rsqrt_f32(val: f32) -> f32 {
    let raw_bits = val.to_bits();
    if (raw_bits >= 0x0080_0000) & (raw_bits <= 0x781A_530C) {
        let half_x = 0.5 * val;
        let mut y = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(raw_bits >> 1));
        y = y * (1.5 - half_x * y * y);
        y = y * (1.5 - half_x * y * y);
        return y * (1.5 - half_x * y * y);
    }
    if (raw_bits == 0) | (raw_bits >= 0x7F80_0000) {
        return 0.0;
    }
    let (e_unbiased, mant23) = unpack_pos_f32_exp_and_mant(raw_bits);
    let odd_exp = (e_unbiased & 1) as u32;
    let m_bits = mant23 | ((127 + odd_exp) << 23);
    let m = f32::from_bits(m_bits);
    let half_m = 0.5 * m;

    let mut y = f32::from_bits(0x5F37_5A86_u32.wrapping_sub(m_bits >> 1));
    y = y * (1.5 - half_m * y * y);
    y = y * (1.5 - half_m * y * y);
    y = y * (1.5 - half_m * y * y);

    let neg_half_exp = ((-(e_unbiased >> 1)) as u32) << 23;
    f32::from_bits(y.to_bits().wrapping_add(neg_half_exp))
}

#[inline(always)]
const fn sincos_poly_half_pi(x_folded: f32, sign_cos: f32) -> (f32, f32) {
    let x2 = x_folded * x_folded;
    let sin_poly = x_folded
        * (1.0
            + x2 * (-0.166_666_67
                + x2 * (0.008_333_334
                    + x2 * (-0.000_198_412_7
                        + x2 * (0.000_002_755_732 + x2 * -0.000_000_025_052_108)))));
    let cos_poly = 1.0
        + x2 * (-0.5
            + x2 * (0.041_666_668
                + x2 * (-0.001_388_888_9 + x2 * (0.000_024_801_587 + x2 * -0.000_000_275_573_2))));
    (sin_poly, sign_cos * cos_poly)
}

#[inline(always)]
pub(crate) const fn sincos_reduced_pi(x: f32) -> (f32, f32) {
    let gt_half_pi = x > FRAC_PI_2;
    let lt_neg_half_pi = x < -FRAC_PI_2;
    let x_folded = branchless_select_f32(
        gt_half_pi,
        PI - x,
        branchless_select_f32(lt_neg_half_pi, -PI - x, x),
    );
    let sign_cos = branchless_select_f32(gt_half_pi | lt_neg_half_pi, -1.0, 1.0);
    sincos_poly_half_pi(x_folded, sign_cos)
}

/// Branchless `#![no_std]` fused $(\sin\theta, \cos\theta)$ evaluator using 53-bit `f64` $2\pi$ range reduction.
#[inline(always)]
pub const fn branchless_sincos_f32(val: f32) -> (f32, f32) {
    let valid = branchless_is_finite_f32(val);
    let safe_val = branchless_select_f32(valid, val, 0.0) as f64;
    let ratio = safe_val * INV_TWO_PI_F64;
    let k = ((ratio + if ratio >= 0.0 { 0.5 } else { -0.5 }) as i64) as f64;
    let x = (safe_val - k * TWO_PI_F64) as f32;
    let (sin_v, cos_v) = sincos_reduced_pi(x);
    (
        branchless_select_f32(valid, sin_v, 0.0),
        branchless_select_f32(valid, cos_v, 1.0),
    )
}

/// Branchless direct $(\sin\theta, \cos\theta)$ evaluator for angles bounded within $[-\pi/2, \pi/2]$.
#[inline(always)]
pub const fn branchless_sincos_half_pi_f32(theta_rad: f32) -> (f32, f32) {
    let valid = branchless_is_finite_f32(theta_rad);
    let safe_theta = branchless_clamp_f32(
        branchless_select_f32(valid, theta_rad, 0.0),
        -FRAC_PI_2,
        FRAC_PI_2,
    );
    let (sin_v, cos_v) = sincos_poly_half_pi(safe_theta, 1.0);
    (
        branchless_select_f32(valid, sin_v, 0.0),
        branchless_select_f32(valid, cos_v, 1.0),
    )
}

/// Direct $\mathcal{O}(1)$ 64B-aligned 256-entry LUT lookup for RIPA $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$
/// under [`DEFAULT_RIPA_CLAMP_MAX_RAD`].
#[inline(always)]
pub const fn branchless_sincos_q7_ripa_default_lut(phase_q7: gcso_q7_t) -> (f32, f32) {
    let lut = Q7_DEFAULT_RIPA_SINCOS_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Direct $\mathcal{O}(1)$ 64B-aligned 256-entry LUT lookup combining $S^1$ geodesic folding
/// with RIPA soft-clamping under [`DEFAULT_RIPA_CLAMP_MAX_RAD`].
#[inline(always)]
pub const fn branchless_sincos_q7_geodesic_ripa_default_lut(phase_q7: gcso_q7_t) -> (f32, f32) {
    let lut = Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Evaluates the signed RIPA-clamped radian angle $\Delta\theta_{\mathrm{safe}} = \mathrm{sgn}(q_7)\cdot |\theta_{\max}| \cdot \tanh(|q_7|\cdot s)$
/// using the 257-entry `TANH_LUT_257` table.
#[inline(always)]
pub(crate) const fn eval_ripa_clamped_theta_rad_preinv(
    phase_q7: gcso_q7_t,
    max_rad: f32,
    q7_to_norm_scale: f32,
) -> f32 {
    let u_abs = branchless_q7_abs_u8(phase_q7);
    let norm_u = (u_abs as f32) * branchless_abs_f32(q7_to_norm_scale);
    let safe_theta_mag = branchless_abs_f32(max_rad) * tanh_lut_eval_nonneg(norm_u);
    let is_neg = ((phase_q7 < 0) ^ (max_rad < 0.0)) & (u_abs != 0);
    f32::from_bits(safe_theta_mag.to_bits() | ((is_neg as u32) << 31))
}

/// Division-free (`0 fdiv`) branchless RIPA soft-clamped $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$
/// using precomputed `q7_to_norm_scale = Q7_PHASE_SCALE_RAD / max_rad`.
#[inline(always)]
pub const fn branchless_sincos_q7_ripa_preinv(
    phase_q7: gcso_q7_t,
    max_rad: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if (max_rad == DEFAULT_RIPA_CLAMP_MAX_RAD)
        & (q7_to_norm_scale == DEFAULT_RIPA_Q7_TO_NORM_SCALE)
    {
        return branchless_sincos_q7_ripa_default_lut(phase_q7);
    }
    let u_abs = branchless_q7_abs_u8(phase_q7);
    let norm_u = (u_abs as f32) * branchless_abs_f32(q7_to_norm_scale);
    let safe_theta_mag = branchless_abs_f32(max_rad) * tanh_lut_eval_nonneg(norm_u);
    let (sin_mag, cos_val) = sincos_ripa_poly(safe_theta_mag);
    let sin_neg = ((phase_q7 < 0) ^ (max_rad < 0.0)) & (u_abs != 0);
    (
        f32::from_bits(sin_mag.to_bits() | ((sin_neg as u32) << 31)),
        cos_val,
    )
}

/// Standalone division-free branchless RIPA soft-clamped $(\sin\theta_{\mathrm{safe}}, \cos\theta_{\mathrm{safe}})$ from Q7.
#[inline(always)]
pub const fn branchless_sincos_q7_ripa(phase_q7: gcso_q7_t, max_rad: f32) -> (f32, f32) {
    if max_rad == DEFAULT_RIPA_CLAMP_MAX_RAD {
        return branchless_sincos_q7_ripa_default_lut(phase_q7);
    }
    let valid_max = (max_rad > 1.0e-9) & branchless_is_finite_f32(max_rad);
    let eff_max = branchless_min_f32(
        branchless_select_f32(valid_max, max_rad, DEFAULT_RIPA_CLAMP_MAX_RAD),
        FRAC_PI_2,
    );
    let q7_to_norm_scale = Q7_PHASE_SCALE_RAD * branchless_inv_pos_f32(eff_max);
    let (sin_t, cos_t) = branchless_sincos_q7_ripa_preinv(phase_q7, eff_max, q7_to_norm_scale);
    (
        branchless_select_f32(valid_max, sin_t, 0.0),
        branchless_select_f32(valid_max, cos_t, 1.0),
    )
}

/// Direct 64B-aligned 256-entry interleaved LUT $(\sin\theta, \cos\theta)$ from Q7.
#[inline(always)]
pub const fn branchless_sincos_q7(phase_q7: gcso_q7_t) -> (f32, f32) {
    let lut = Q7_SINCOS_LUT_256_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Direct 256-entry LUT $\sin\theta$ from a Q7 phase angle (`q7 * pi / 128`).
#[inline(always)]
pub const fn branchless_sin_q7(phase_q7: gcso_q7_t) -> f32 {
    let lut = Q7_FULL_SIN_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Direct 256-entry LUT $\cos\theta$ from a Q7 phase angle (`q7 * pi / 128`).
#[inline(always)]
pub const fn branchless_cos_q7(phase_q7: gcso_q7_t) -> f32 {
    let lut = Q7_FULL_COS_REF;
    lut[(phase_q7 as u8) as usize]
}

/// Branchless `#![no_std]` four-quadrant inverse tangent ($\mathrm{atan2}(y, x) \in [-\pi, \pi]$)
/// using `fast_inv_pos_normal_f32` (since $\max(|x|, |y|) \ge 10^{-30} \gg \text{F32\_MIN\_POSITIVE\_NORMAL}$).
#[inline(always)]
pub const fn branchless_atan2_f32(y: f32, x: f32) -> f32 {
    let valid = branchless_is_finite_f32(x) & branchless_is_finite_f32(y);
    let safe_x = branchless_select_f32(valid, x, 1.0);
    let safe_y = branchless_select_f32(valid, y, 0.0);
    let ax = branchless_abs_f32(safe_x);
    let ay = branchless_abs_f32(safe_y);
    let both_zero = (ax < 1.0e-30) & (ay < 1.0e-30);
    let swap = ax < ay;
    let min_v = branchless_select_f32(swap, ax, ay);
    let max_v = branchless_max_f32(branchless_select_f32(swap, ay, ax), 1.0e-30);
    let inv_max = if max_v <= 1.0e30 {
        fast_inv_pos_normal_f32(max_v)
    } else {
        branchless_inv_pos_f32(max_v)
    };
    let r = branchless_clamp_f32(min_v * inv_max, 0.0, 1.0);
    let r2 = r * r;
    let poly = r
        * (0.999_977_26
            + r2 * (-0.332_623_47
                + r2 * (0.193_543_46
                    + r2 * (-0.116_432_87 + r2 * (0.052_653_32 - 0.011_721_2 * r2)))));
    let mut angle = branchless_select_f32(swap, FRAC_PI_2 - poly, poly);
    angle = branchless_select_f32(safe_x < 0.0, PI - angle, angle);
    angle = branchless_copysign_f32(angle, safe_y);
    branchless_select_f32(valid & !both_zero, angle, 0.0)
}

/// Fused `#![no_std]` four-quadrant $\mathrm{atan2}(y, x)$ directly quantized into `gcso_q7_t`
/// ($[-128, 127]$), folding the $\frac{128}{\pi}$ scale factor directly into the minimax polynomial
/// coefficients (`0` Q7 error across $[-180^\circ, 180^\circ]$).
#[inline(always)]
pub const fn branchless_atan2_to_q7_fast(y: f32, x: f32) -> gcso_q7_t {
    let valid = branchless_is_finite_f32(x) & branchless_is_finite_f32(y);
    let safe_x = branchless_select_f32(valid, x, 1.0);
    let safe_y = branchless_select_f32(valid, y, 0.0);
    let ax = branchless_abs_f32(safe_x);
    let ay = branchless_abs_f32(safe_y);
    let both_zero = (ax < 1.0e-30) & (ay < 1.0e-30);
    if (!valid) | both_zero {
        return 0;
    }
    let swap = ax < ay;
    let min_v = branchless_select_f32(swap, ax, ay);
    let max_v = branchless_max_f32(branchless_select_f32(swap, ay, ax), 1.0e-30);
    let inv_max = if max_v <= 1.0e30 {
        fast_inv_pos_normal_f32(max_v)
    } else {
        branchless_inv_pos_f32(max_v)
    };
    let r = branchless_clamp_f32(min_v * inv_max, 0.0, 1.0);
    let r2 = r * r;
    let poly_q7 = r
        * (40.742_737
            + r2 * (-13.553_533
                + r2 * (7.886_416
                    + r2 * (-4.744_295 + r2 * (2.145_485 - 0.477_609 * r2)))));
    let mut ang_q7 = branchless_select_f32(swap, 64.0 - poly_q7, poly_q7);
    ang_q7 = branchless_select_f32(safe_x < 0.0, 128.0 - ang_q7, ang_q7);
    ang_q7 = branchless_copysign_f32(ang_q7, safe_y);
    let rounded = (ang_q7 + branchless_select_f32(ang_q7 >= 0.0, 0.5, -0.5)) as i32;
    branchless_select_i32(
        rounded < -128,
        -128,
        branchless_select_i32(rounded > 127, 127, rounded),
    ) as gcso_q7_t
}

#[inline(always)]
const fn eval_von_mises_kappa_clamped(r: f32) -> f32 {
    let r2 = r * r;
    let r3 = r2 * r;
    let k1 = 2.0 * r + r3 + 0.833_333_3 * (r3 * r2);
    let k2 = fast_inv_pos_normal_f32(1.28 * branchless_max_f32(1.0 - r2, 1.0e-4));
    let k3 = fast_inv_pos_normal_f32(
        branchless_max_f32(r, 0.5) * branchless_max_f32(1.0 - r, 1.0e-4) * (3.0 - r),
    );
    let w12 = branchless_sigmoid_f32((r - 0.53) * 40.0);
    let w23 = branchless_sigmoid_f32((r - 0.85) * 40.0);
    let blend12 = k1 + w12 * (k2 - k1);
    blend12 + w23 * (k3 - blend12)
}

pub(crate) const VON_MISES_KAPPA_AT_MAX_R: f32 =
    eval_von_mises_kappa_clamped(VON_MISES_KAPPA_MAX_R);

/// 64B-aligned 2-zone (`[0.0, 0.875]` and `[0.875, 0.995]`, `2 x 257` `f32` entries = `2056` bytes)
/// precomputed von Mises concentration $\kappa_c(\bar{R})$ table eliminating 4 Newton-Raphson
/// inversions and 2 Padé `tanh` evaluations per call on the hot/mezzo path.
pub(crate) static VON_MISES_KAPPA_2ZONE_LUT: AlignedLut2x257<f32> = {
    let mut lut = [[0.0_f32; 257]; 2];
    let mut i = 0usize;
    while i < 257 {
        lut[0][i] = eval_von_mises_kappa_clamped((i as f32) * VON_MISES_KAPPA_LO_STEP);
        lut[1][i] = eval_von_mises_kappa_clamped(
            VON_MISES_KAPPA_SPLIT_R + (i as f32) * VON_MISES_KAPPA_HI_STEP,
        );
        i += 1;
    }
    AlignedLut2x257(lut)
};

const VON_MISES_KAPPA_2ZONE_REF: &[[f32; 257]; 2] = &VON_MISES_KAPPA_2ZONE_LUT.0;

/// Branchless sigmoidal-blended von Mises concentration parameter $\kappa_c$ from $\bar{R}_c \in [0, 1)$,
/// evaluated in strictly constant time (`0` Newton-Raphson divisions, `0` Padé `tanh` evaluations)
/// via `VON_MISES_KAPPA_2ZONE_LUT` with a 1-cycle ceiling return for $\bar{R}_c \ge 0.995$.
#[inline(always)]
pub const fn branchless_von_mises_kappa(r_bar: f32) -> f32 {
    if r_bar >= VON_MISES_KAPPA_MAX_R {
        return VON_MISES_KAPPA_AT_MAX_R;
    }
    let clean_r = branchless_relu_f32(r_bar);
    if clean_r <= 0.0 {
        return 0.0;
    }
    let is_hi = clean_r > VON_MISES_KAPPA_SPLIT_R;
    let zone = is_hi as usize;
    let pos = branchless_select_f32(
        is_hi,
        (clean_r - VON_MISES_KAPPA_SPLIT_R) * VON_MISES_KAPPA_HI_SCALE,
        clean_r * VON_MISES_KAPPA_LO_SCALE,
    );
    let idx_raw = pos as usize;
    let idx = branchless_select_usize(idx_raw < 256, idx_raw, 255);
    let frac = pos - (idx as f32);
    let lut = &VON_MISES_KAPPA_2ZONE_REF[zone];
    let v0 = lut[idx];
    let v1 = lut[idx + 1];
    v0 + frac * (v1 - v0)
}

/// Evaluates Phase-Proximity Gravity $F_{\mathrm{grav}} = \frac{M_{\mathrm{grav}} \cdot \max(0, 1 + \mathcal{C}_{\mathrm{phase}})}{d_{\mathrm{chord}}^2 + \epsilon_{\mathrm{grav}}}$
/// on $SO(2)^{64}$ where $d_{\mathrm{chord}}^2 = 2(1 - \mathcal{C}_{\mathrm{phase}})$ (`0` branches, `0 fdiv`).
#[inline(always)]
pub const fn branchless_phase_proximity_gravity_f32(
    gravity_mass: f32,
    phase_coherence: f32,
    softening_eps: f32,
) -> f32 {
    let clean_mass = branchless_flush_subnormal_f32(branchless_relu_f32(gravity_mass));
    let is_default_eps = softening_eps == DEFAULT_PHASE_GRAVITY_EPS;
    if (phase_coherence == 1.0) & is_default_eps {
        return branchless_flush_subnormal_f32(clean_mass * (2.0 / DEFAULT_PHASE_GRAVITY_EPS));
    }
    let coh = branchless_clamp_f32(
        branchless_select_f32(branchless_is_finite_f32(phase_coherence), phase_coherence, 0.0),
        -1.0,
        1.0,
    );
    let eps = if is_default_eps {
        DEFAULT_PHASE_GRAVITY_EPS
    } else {
        branchless_max_f32(
            branchless_select_f32(
                branchless_is_finite_f32(softening_eps) & (softening_eps > 1.0e-4),
                softening_eps,
                DEFAULT_PHASE_GRAVITY_EPS,
            ),
            1.0e-4,
        )
    };
    let chord_dist_sq = 2.0 * (1.0 - coh);
    let denom = chord_dist_sq + eps;
    let align_gain = branchless_max_f32(0.05, 1.0 + coh);
    branchless_flush_subnormal_f32((clean_mass * align_gain) * fast_inv_pos_normal_f32(denom))
}

/// Branchless scale-invariant Eyring-Kramers Langevin pitchfork tunneling escape probability (Lemma 3.1).
#[inline(always)]
pub const fn branchless_eyring_kramers_escape(
    cvoid: f32,
    tau_eff: f32,
    alpha: f32,
    beta: f32,
    d0: f32,
) -> f32 {
    let valid = branchless_is_finite_f32(cvoid)
        & branchless_is_finite_f32(tau_eff)
        & branchless_is_finite_f32(alpha)
        & branchless_is_finite_f32(beta)
        & branchless_is_finite_f32(d0)
        & (beta > 1.0e-12)
        & (d0 > 1.0e-12);
    let safe_denom = branchless_select_f32(valid, 4.0 * beta * d0, 1.0);
    let excess = branchless_relu_f32(cvoid - tau_eff);
    let safe_alpha = branchless_select_f32(valid, alpha, 0.0);
    let ratio = (safe_alpha * safe_alpha * excess * excess) * fast_inv_pos_normal_f32(safe_denom);
    branchless_select_f32(valid, branchless_exp_f32(-ratio), 0.0)
}

/// Converts a Q7 fixed-point phase value to radians ($\theta = \mathrm{q7} \cdot \pi / 128$).
#[inline(always)]
pub const fn branchless_q7_to_rad(phase_q7: gcso_q7_t) -> f32 {
    (phase_q7 as f32) * Q7_PHASE_SCALE_RAD
}

/// Branchlessly quantizes a radian angle into Q7 fixed-point format ($[-128, 127]$).
#[inline(always)]
pub const fn branchless_rad_to_q7(rad: f32) -> gcso_q7_t {
    let valid = branchless_is_finite_f32(rad);
    let scaled = branchless_select_f32(valid, rad, 0.0) * Q7_PHASE_INV_SCALE_RAD;
    let rounded = (scaled + branchless_select_f32(scaled >= 0.0, 0.5, -0.5)) as i32;
    let clamped = branchless_select_i32(
        rounded < -128,
        -128,
        branchless_select_i32(rounded > 127, 127, rounded),
    );
    branchless_select_i32(valid, clamped, 0) as gcso_q7_t
}

/// Converts a non-negative radian step threshold into a ceiling-rounded Q7 integer cutoff (`0..=128`).
#[inline(always)]
pub const fn branchless_rad_to_qdps_step_u8(min_step_rad: f32) -> u8 {
    let bits = min_step_rad.to_bits();
    if (bits & 0x7FFF_FFFF) == 0 || (bits >> 31) != 0 || (bits & 0x7F80_0000) == 0x7F80_0000 {
        return 0;
    }
    if bits <= Q7_PHASE_SCALE_RAD.to_bits() {
        return 1;
    }
    let raw_q7 = min_step_rad * Q7_PHASE_INV_SCALE_RAD;
    let int_part = raw_q7 as i32;
    let step = int_part + (((int_part as f32) < raw_q7) as i32);
    branchless_select_i32(step > 128, 128, branchless_select_i32(step < 0, 0, step)) as u8
}

/// Branchless Q7 fixed-point saturating addition ($[-128, 127]$).
#[inline(always)]
pub const fn branchless_q7_add_sat(a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
    let sum = (a as i32) + (b as i32);
    let clamped_hi = branchless_select_i32(sum > (Q7_MAX as i32), Q7_MAX as i32, sum);
    branchless_select_i32(clamped_hi < (Q7_MIN as i32), Q7_MIN as i32, clamped_hi) as gcso_q7_t
}

/// Branchless Q7 fixed-point saturating subtraction ($[-128, 127]$).
#[inline(always)]
pub const fn branchless_q7_sub_sat(a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
    let diff = (a as i32) - (b as i32);
    let clamped_hi = branchless_select_i32(diff > (Q7_MAX as i32), Q7_MAX as i32, diff);
    branchless_select_i32(clamped_hi < (Q7_MIN as i32), Q7_MIN as i32, clamped_hi) as gcso_q7_t
}

/// Branchless Q7 circular phase difference $(a - b) \pmod{256}$ on $S^1 \cong SO(2)$.
#[inline(always)]
pub const fn branchless_q7_circular_diff(a: gcso_q7_t, b: gcso_q7_t) -> gcso_q7_t {
    a.wrapping_sub(b)
}

/// Branchlessly folds a circular Q7 phase angle on $S^1$ onto the principal geodesic wave `[-64, +64]`.
#[inline(always)]
pub const fn branchless_q7_fold_s1_geodesic(phase_q7: gcso_q7_t) -> gcso_q7_t {
    let b = phase_q7 as u8;
    let fold_mask = ((b ^ (b << 1)) >> 7).wrapping_neg();
    let mirrored = 0x80u8.wrapping_sub(b);
    ((b & !fold_mask) | (mirrored & fold_mask)) as gcso_q7_t
}

/// Branchless Q7 shortest-arc geodesic interpolation on $S^1 \cong SO(2)$ (`t_u7 in 0..=128`).
#[inline(always)]
pub const fn branchless_q7_circular_lerp(a: gcso_q7_t, b: gcso_q7_t, t_u7: u8) -> gcso_q7_t {
    let clamped_t = branchless_select_u8(t_u7 > 128, 128, t_u7);
    a.wrapping_add(branchless_q7_scale_u7(
        branchless_q7_circular_diff(b, a),
        clamped_t,
    ))
}

/// Branchless Q7 phase-conjugate negation ($-\Delta\theta$) saturating $-128 \to +127$.
#[inline(always)]
pub const fn branchless_q7_neg_sat(x: gcso_q7_t) -> gcso_q7_t {
    let neg = -(x as i32);
    branchless_select_i32(neg > 127, 127, neg) as gcso_q7_t
}

/// Branchless Q7 absolute value returning `[0, 128]` as `u8`.
#[inline(always)]
pub const fn branchless_q7_abs_u8(x: gcso_q7_t) -> u8 {
    let val = x as i32;
    let mask = val >> 31;
    ((val ^ mask) - mask) as u8
}

/// Branchless Q7 fixed-point integer scaling: $\lfloor (\mathrm{val} \cdot \mathrm{scale\_u7}) / 128 \rfloor$.
#[inline(always)]
pub const fn branchless_q7_scale_u7(val: gcso_q7_t, scale_u7: u8) -> gcso_q7_t {
    let scaled = ((val as i32) * (scale_u7 as i32)) >> 7;
    let clamped_hi = branchless_select_i32(scaled > 127, 127, scaled);
    branchless_select_i32(clamped_hi < -128, -128, clamped_hi) as gcso_q7_t
}

/// Packs 8 `gcso_q7_t` lanes into a single 64-bit little-endian SWAR word.
#[inline(always)]
pub const fn swar_q7_pack_x8(lanes: &[gcso_q7_t; 8]) -> u64 {
    u64::from_le_bytes([
        lanes[0] as u8,
        lanes[1] as u8,
        lanes[2] as u8,
        lanes[3] as u8,
        lanes[4] as u8,
        lanes[5] as u8,
        lanes[6] as u8,
        lanes[7] as u8,
    ])
}

/// Unpacks a 64-bit little-endian SWAR word into 8 `gcso_q7_t` lanes.
#[inline(always)]
pub const fn swar_q7_unpack_x8(packed: u64) -> [gcso_q7_t; 8] {
    let b = packed.to_le_bytes();
    [
        b[0] as gcso_q7_t,
        b[1] as gcso_q7_t,
        b[2] as gcso_q7_t,
        b[3] as gcso_q7_t,
        b[4] as gcso_q7_t,
        b[5] as gcso_q7_t,
        b[6] as gcso_q7_t,
        b[7] as gcso_q7_t,
    ]
}

/// True bit-parallel SWAR 8-lane unsigned byte halving decay (`b_i >> 1`).
#[inline(always)]
pub const fn swar_u8_halve_decay_x8(packed_u8: u64) -> u64 {
    (packed_u8 >> 1) & SWAR_LOW7_MASK_64
}

/// True bit-parallel SWAR 8-lane unsigned byte saturating addition (`min(a_i + b_i, 255)`).
#[inline(always)]
pub const fn swar_u8_add_sat_x8(a: u64, b: u64) -> u64 {
    let low7_sum = (a & SWAR_LOW7_MASK_64).wrapping_add(b & SWAR_LOW7_MASK_64);
    let high_bits = (a ^ b) & SWAR_SIGN_MASK_64;
    let carry_from_low7 = low7_sum & SWAR_SIGN_MASK_64;
    let overflow_msb = ((a & b) | ((a | b) & carry_from_low7)) & SWAR_SIGN_MASK_64;
    let one_hi = high_bits ^ carry_from_low7;
    let sat_mask = overflow_msb | overflow_msb.wrapping_sub(overflow_msb >> 7);
    (low7_sum ^ one_hi) | sat_mask
}

/// True bit-parallel SWAR 8-lane unsigned byte saturating subtraction (`max(b_i - sub_step, 0)`).
#[inline(always)]
pub const fn swar_u8_sub_sat_x8(packed_u8: u64, sub_step: u8) -> u64 {
    let b = (sub_step as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let diff_low7 = (packed_u8 | SWAR_SIGN_MASK_64).wrapping_sub(b & SWAR_LOW7_MASK_64);
    let ge_bit7 = ((packed_u8 & !b) | (!(packed_u8 ^ b) & diff_low7)) & SWAR_SIGN_MASK_64;
    let keep_mask = (ge_bit7 >> 7).wrapping_mul(0xFF);
    let diff_unwrapped = diff_low7 ^ ((packed_u8 ^ !b) & SWAR_SIGN_MASK_64);
    diff_unwrapped & keep_mask
}

/// True bit-parallel SWAR 8-lane unsigned byte elementwise maximum (`max(a_i, b_i)`).
#[inline(always)]
pub const fn swar_u8_max_x8(a: u64, b: u64) -> u64 {
    let diff_low7 =
        ((a & SWAR_LOW7_MASK_64) | SWAR_SIGN_MASK_64).wrapping_sub(b & SWAR_LOW7_MASK_64);
    let a_ge_b_msb = ((a & !b) | ((!(a ^ b)) & diff_low7)) & SWAR_SIGN_MASK_64;
    let mask = a_ge_b_msb | a_ge_b_msb.wrapping_sub(a_ge_b_msb >> 7);
    (a & mask) | (b & !mask)
}

#[inline(always)]
pub(crate) const fn swar_u8_nonzero_msb_mask(w: u64) -> u64 {
    let has_low7 = (w & SWAR_LOW7_MASK_64).wrapping_add(SWAR_LOW7_MASK_64);
    (w | has_low7) & SWAR_SIGN_MASK_64
}

/// Computes the exact horizontal sum (`0..=2040`) of 8 packed unsigned bytes in `packed_u8`.
#[inline(always)]
pub const fn swar_u8_sum_x8(packed_u8: u64) -> u32 {
    let sum16 =
        (packed_u8 & SWAR_EVEN_BYTES_64).wrapping_add((packed_u8 >> 8) & SWAR_EVEN_BYTES_64);
    (sum16.wrapping_mul(SWAR_SUM_U16_64) >> 48) as u32
}

/// Computes the exact rounded mean $\lfloor (\sum_{i=0}^7 b_i + 4) / 8 \rfloor$ of 8 packed `u8` lanes.
#[inline(always)]
pub const fn swar_u8_mean_round_x8(packed_u8: u64) -> u8 {
    ((swar_u8_sum_x8(packed_u8) + 4) >> 3) as u8
}

/// Counts the number of non-zero byte lanes (`0..=8`) inside a 64-bit word `packed_u8`.
#[inline(always)]
pub const fn swar_u8_nonzero_count_x8(packed_u8: u64) -> u32 {
    swar_u8_nonzero_msb_mask(packed_u8).count_ones()
}

/// True bit-parallel SWAR 4-lane Q7 saturating addition packed in a 32-bit word.
#[inline(always)]
pub const fn swar_q7_add_sat_x4(a_packed: u32, b_packed: u32) -> u32 {
    let sum_low7 = (a_packed & SWAR_LOW7_MASK_32).wrapping_add(b_packed & SWAR_LOW7_MASK_32);
    let sum_unwrapped = sum_low7 ^ ((a_packed ^ b_packed) & SWAR_SIGN_MASK_32);
    let ovf_sign = !(a_packed ^ b_packed) & (a_packed ^ sum_unwrapped) & SWAR_SIGN_MASK_32;
    let ovf_mask = (ovf_sign >> 7).wrapping_mul(0xFF);
    let sat_val = SWAR_LOW7_MASK_32.wrapping_add((a_packed & SWAR_SIGN_MASK_32) >> 7);
    (sum_unwrapped & !ovf_mask) | (sat_val & ovf_mask)
}

/// True bit-parallel SWAR 4-lane Q7 saturating subtraction packed in a 32-bit word.
#[inline(always)]
pub const fn swar_q7_sub_sat_x4(a_packed: u32, b_packed: u32) -> u32 {
    let diff_low7 = (a_packed | SWAR_SIGN_MASK_32).wrapping_sub(b_packed & SWAR_LOW7_MASK_32);
    let diff_unwrapped = diff_low7 ^ ((a_packed ^ !b_packed) & SWAR_SIGN_MASK_32);
    let ovf_sign = (a_packed ^ b_packed) & (a_packed ^ diff_unwrapped) & SWAR_SIGN_MASK_32;
    let ovf_mask = (ovf_sign >> 7).wrapping_mul(0xFF);
    let sat_val = SWAR_LOW7_MASK_32.wrapping_add((a_packed & SWAR_SIGN_MASK_32) >> 7);
    (diff_unwrapped & !ovf_mask) | (sat_val & ovf_mask)
}

/// True bit-parallel SWAR 4-lane Q7 phase-conjugate negation ($-\boldsymbol{\Delta\theta}$) in a 32-bit word.
#[inline(always)]
pub const fn swar_q7_neg_sat_x4(packed: u32) -> u32 {
    swar_q7_sub_sat_x4(0, packed)
}

/// True bit-parallel SWAR 8-lane Q7 saturating addition packed in a 64-bit word.
#[inline(always)]
pub const fn swar_q7_add_sat_x8(a_packed: u64, b_packed: u64) -> u64 {
    let sum_low7 = (a_packed & SWAR_LOW7_MASK_64).wrapping_add(b_packed & SWAR_LOW7_MASK_64);
    let sum_unwrapped = sum_low7 ^ ((a_packed ^ b_packed) & SWAR_SIGN_MASK_64);
    let ovf_sign = !(a_packed ^ b_packed) & (a_packed ^ sum_unwrapped) & SWAR_SIGN_MASK_64;
    let ovf_mask = (ovf_sign >> 7).wrapping_mul(0xFF);
    let sat_val = SWAR_LOW7_MASK_64.wrapping_add((a_packed & SWAR_SIGN_MASK_64) >> 7);
    (sum_unwrapped & !ovf_mask) | (sat_val & ovf_mask)
}

/// True bit-parallel SWAR 8-lane Q7 saturating subtraction packed in a 64-bit word.
#[inline(always)]
pub const fn swar_q7_sub_sat_x8(a_packed: u64, b_packed: u64) -> u64 {
    let diff_low7 = (a_packed | SWAR_SIGN_MASK_64).wrapping_sub(b_packed & SWAR_LOW7_MASK_64);
    let diff_unwrapped = diff_low7 ^ ((a_packed ^ !b_packed) & SWAR_SIGN_MASK_64);
    let ovf_sign = (a_packed ^ b_packed) & (a_packed ^ diff_unwrapped) & SWAR_SIGN_MASK_64;
    let ovf_mask = (ovf_sign >> 7).wrapping_mul(0xFF);
    let sat_val = SWAR_LOW7_MASK_64.wrapping_add((a_packed & SWAR_SIGN_MASK_64) >> 7);
    (diff_unwrapped & !ovf_mask) | (sat_val & ovf_mask)
}

/// True bit-parallel SWAR 8-lane Q7 phase-conjugate negation ($-\boldsymbol{\Delta\theta}$) in a 64-bit word.
#[inline(always)]
pub const fn swar_q7_neg_sat_x8(packed: u64) -> u64 {
    swar_q7_sub_sat_x8(0, packed)
}

/// True bit-parallel SWAR 8-lane Q7 circular phase difference $(a_i - b_i) \pmod{256}$ on $S^1$.
#[inline(always)]
pub const fn swar_q7_circular_diff_x8(a_packed: u64, b_packed: u64) -> u64 {
    let diff_low7 = (a_packed | SWAR_SIGN_MASK_64).wrapping_sub(b_packed & SWAR_LOW7_MASK_64);
    diff_low7 ^ ((a_packed ^ !b_packed) & SWAR_SIGN_MASK_64)
}

/// True bit-parallel SWAR 8-lane circular phase addition $(a_i + b_i) \pmod{256}$ on $S^1$.
#[inline(always)]
pub const fn swar_q7_circular_add_x8(a_packed: u64, b_packed: u64) -> u64 {
    let sum_low7 = (a_packed & SWAR_LOW7_MASK_64).wrapping_add(b_packed & SWAR_LOW7_MASK_64);
    sum_low7 ^ ((a_packed ^ b_packed) & SWAR_SIGN_MASK_64)
}

/// True bit-parallel SWAR 8-lane Q7 absolute value (`|x_i| in [0, 128]`) packed in a 64-bit word.
#[inline(always)]
pub const fn swar_q7_abs_u8_x8(packed: u64) -> u64 {
    let sign_bit = (packed & SWAR_SIGN_MASK_64) >> 7;
    let sign_mask = sign_bit.wrapping_mul(0xFF);
    (packed ^ sign_mask).wrapping_add(sign_bit)
}

/// True bit-parallel SWAR 8-lane $S^1$ geodesic folding (`[-128, 127] -> [-64, 64]`).
#[inline(always)]
pub const fn swar_q7_fold_s1_geodesic_x8(packed: u64) -> u64 {
    let fold_bit7 = (packed ^ (packed << 1)) & SWAR_SIGN_MASK_64;
    let fold_mask = (fold_bit7 >> 7).wrapping_mul(0xFF);
    let mirrored =
        SWAR_SIGN_MASK_64.wrapping_sub(packed & SWAR_LOW7_MASK_64) ^ (packed & SWAR_SIGN_MASK_64);
    (packed & !fold_mask) | (mirrored & fold_mask)
}

/// True bit-parallel SWAR 8-lane Q7 rounded halving average $\lfloor(a_i + b_i + 1) / 2\rfloor$.
#[inline(always)]
pub const fn swar_q7_avg_round_x8(a_packed: u64, b_packed: u64) -> u64 {
    let ua = a_packed ^ SWAR_SIGN_MASK_64;
    let ub = b_packed ^ SWAR_SIGN_MASK_64;
    let uavg = (ua | ub).wrapping_sub(((ua ^ ub) >> 1) & SWAR_LOW7_MASK_64);
    uavg ^ SWAR_SIGN_MASK_64
}

/// Precomputed-bias SWAR 8-lane Q7 scale for `s <= 128` where `bias_restore = (128 - s) * 0x0101_0101_0101_0101`.
#[inline(always)]
pub const fn swar_q7_scale_le128_with_bias_x8(packed: u64, s: u64, bias_restore: u64) -> u64 {
    let u = packed ^ SWAR_SIGN_MASK_64;
    let u_even = u & SWAR_EVEN_BYTES_64;
    let u_odd = (u >> 8) & SWAR_EVEN_BYTES_64;
    let scaled_even = (u_even.wrapping_mul(s) >> 7) & SWAR_EVEN_BYTES_64;
    let scaled_odd = (u_odd.wrapping_mul(s) << 1) & SWAR_ODD_BYTES_64;
    (scaled_even | scaled_odd).wrapping_add(bias_restore) ^ SWAR_SIGN_MASK_64
}

#[inline(always)]
pub(crate) const fn swar_q7_scale_le128_x8(packed: u64, s: u64) -> u64 {
    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    swar_q7_scale_le128_with_bias_x8(packed, s, bias_restore)
}

/// True bit-parallel SWAR 8-lane Q7 fixed-point gain scaling in strictly constant time (`0` branches).
#[inline(always)]
pub const fn swar_q7_scale_u7_x8(packed: u64, scale_u7: u8) -> u64 {
    let gt_128 = scale_u7 > 128;
    let s_sub = branchless_select_u8(gt_128, scale_u7.wrapping_sub(128), scale_u7) as u64;
    let base_scaled = swar_q7_scale_le128_x8(packed, s_sub);
    let extra = branchless_select_u64(gt_128, packed, 0);
    swar_q7_add_sat_x8(base_scaled, extra)
}

#[inline(always)]
pub(crate) const fn swar_q7_qdps_filter_with_thresh_x8(
    packed: u64,
    thresh_x8: u64,
    valid_step_mask: u64,
) -> u64 {
    let abs_lanes = swar_q7_abs_u8_x8(packed);
    let ge_bit7 = (((abs_lanes | SWAR_SIGN_MASK_64).wrapping_sub(thresh_x8))
        | (abs_lanes & SWAR_SIGN_MASK_64))
        & SWAR_SIGN_MASK_64;
    let keep_mask = (ge_bit7 >> 7).wrapping_mul(0xFF);
    packed & keep_mask & valid_step_mask
}

/// True bit-parallel SWAR 8-lane QDPS discrete phase step cutoff filter in strictly constant time (`0` branches).
#[inline(always)]
pub const fn swar_q7_qdps_filter_x8(packed: u64, min_step_q7: u8) -> u64 {
    let thresh_x8 = ((min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let valid_step_mask = ((min_step_q7 <= 127) as u64).wrapping_neg();
    swar_q7_qdps_filter_with_thresh_x8(packed, thresh_x8, valid_step_mask)
}

/// Applies the 8-lane SWAR QDPS discrete phase step cutoff filter across 8 register-resident
/// `u64` SWAR words using a pre-evaluated word-uniformity flag (`w_is_uniform`), returning
/// `(filtered_words, filtered_is_uniform)`.
#[inline(always)]
#[must_use]
pub const fn swar_q7_qdps_filter_words_with_uniform_flag_64(
    w: &[u64; 8],
    w_is_uniform: bool,
    min_step_q7: u8,
) -> ([u64; 8], bool) {
    if min_step_q7 <= 1 {
        return (*w, w_is_uniform);
    }
    if min_step_q7 > 127 {
        return ([0u64; 8], true);
    }
    let thresh_x8 = ((min_step_q7 & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    if w_is_uniform {
        let w0 = w[0];
        if w0 == 0 {
            return ([0u64; 8], true);
        }
        let b0 = (w0 & 0xFF) as u8;
        if w0 == w0.rotate_right(8) {
            let abs_b0 = (b0 as i8).unsigned_abs();
            return if abs_b0 >= min_step_q7 {
                (*w, true)
            } else {
                ([0u64; 8], true)
            };
        }
        return (
            [swar_q7_qdps_filter_with_thresh_x8(w0, thresh_x8, u64::MAX); 8],
            true,
        );
    }
    let out = [
        swar_q7_qdps_filter_with_thresh_x8(w[0], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[1], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[2], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[3], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[4], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[5], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[6], thresh_x8, u64::MAX),
        swar_q7_qdps_filter_with_thresh_x8(w[7], thresh_x8, u64::MAX),
    ];
    (out, is_word_array_uniform_64(&out))
}

/// Applies the 8-lane SWAR QDPS discrete phase step cutoff filter across all 8 register-resident
/// `u64` SWAR words (`64` attention heads) with hoisted `thresh_x8` and `valid_step_mask` in
/// strictly constant time (`0` data-dependent branches).
#[inline(always)]
#[must_use]
pub const fn swar_q7_qdps_filter_words_64(w: &[u64; 8], min_step_q7: u8) -> [u64; 8] {
    if min_step_q7 <= 1 {
        return *w;
    }
    if min_step_q7 > 127 {
        return [0u64; 8];
    }
    swar_q7_qdps_filter_words_with_uniform_flag_64(w, is_word_array_uniform_64(w), min_step_q7).0
}

#[inline(always)]
pub(crate) const fn load_q7_word_at(arr: &[gcso_q7_t; MAX_ATTENTION_HEADS], base: usize) -> u64 {
    u64::from_le_bytes([
        arr[base] as u8,
        arr[base + 1] as u8,
        arr[base + 2] as u8,
        arr[base + 3] as u8,
        arr[base + 4] as u8,
        arr[base + 5] as u8,
        arr[base + 6] as u8,
        arr[base + 7] as u8,
    ])
}

#[inline(always)]
pub(crate) const fn store_q7_word_at(
    arr: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    base: usize,
    word: u64,
) {
    let b = word.to_le_bytes();
    arr[base] = b[0] as gcso_q7_t;
    arr[base + 1] = b[1] as gcso_q7_t;
    arr[base + 2] = b[2] as gcso_q7_t;
    arr[base + 3] = b[3] as gcso_q7_t;
    arr[base + 4] = b[4] as gcso_q7_t;
    arr[base + 5] = b[5] as gcso_q7_t;
    arr[base + 6] = b[6] as gcso_q7_t;
    arr[base + 7] = b[7] as gcso_q7_t;
}

/// Loads all 64 Q7 head phases into 8 `u64` SWAR registers using 8 direct 64-bit GPR word loads
/// (`mov r64, qword ptr [arr + 8*c]`), avoiding 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF)
/// stalls and 64-byte cache-line split penalties when reading `trail.accumulated_phase_delta` (offset 56).
#[inline(always)]
pub const fn load_q7_words_64(arr: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> [u64; 8] {
    [
        load_q7_word_at(arr, 0),
        load_q7_word_at(arr, 8),
        load_q7_word_at(arr, 16),
        load_q7_word_at(arr, 24),
        load_q7_word_at(arr, 32),
        load_q7_word_at(arr, 40),
        load_q7_word_at(arr, 48),
        load_q7_word_at(arr, 56),
    ]
}

/// Borrows `trail.accumulated_phase_delta` (`offset 56` of `#[repr(C, align(128))]` [`gcso_pointer_trail_t`],
/// guaranteed 8-byte aligned) as a zero-copy `&[u64; 8]` SWAR reference on little-endian targets,
/// eliminating 64-byte stack copies and 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF) stalls.
#[inline(always)]
#[must_use]
pub const fn trail_phase_words_ref(trail: &gcso_pointer_trail_t) -> &[u64; 8] {
    const _: () = assert!(cfg!(target_endian = "little"));
    // SAFETY: `gcso_pointer_trail_t` is `#[repr(C, align(128))]` and `accumulated_phase_delta` is at
    // byte offset 56 (`56 % 8 == 0`, 8-byte aligned), with identical 64-byte layout to `[u64; 8]`.
    unsafe { &*(trail.accumulated_phase_delta.as_ptr().cast::<[u64; 8]>()) }
}

/// Mutably borrows `trail.accumulated_phase_delta` (`offset 56` of `#[repr(C, align(128))]`
/// [`gcso_pointer_trail_t`], guaranteed 8-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
#[inline(always)]
#[must_use]
pub const fn trail_phase_words_mut(trail: &mut gcso_pointer_trail_t) -> &mut [u64; 8] {
    const _: () = assert!(cfg!(target_endian = "little"));
    // SAFETY: `gcso_pointer_trail_t` is `#[repr(C, align(128))]` and `accumulated_phase_delta` is at
    // byte offset 56 (`56 % 8 == 0`, 8-byte aligned), with identical 64-byte layout to `[u64; 8]`.
    unsafe { &mut *(trail.accumulated_phase_delta.as_mut_ptr().cast::<[u64; 8]>()) }
}

/// Stores 8 `u64` SWAR registers into a 64-head Q7 phase array using 8 direct 64-bit GPR word stores
/// (`mov qword ptr [arr + 8*c], r64`), eliminating narrow-GPR-store-to-wide-YMM-load Store-to-Load
/// Forwarding (STLF) stalls (`~15–20 cycles`) and 64-byte cache-line split stores on `trail.accumulated_phase_delta`.
#[inline(always)]
pub const fn store_q7_words_64(arr: &mut [gcso_q7_t; MAX_ATTENTION_HEADS], w: &[u64; 8]) {
    store_q7_word_at(arr, 0, w[0]);
    store_q7_word_at(arr, 8, w[1]);
    store_q7_word_at(arr, 16, w[2]);
    store_q7_word_at(arr, 24, w[3]);
    store_q7_word_at(arr, 32, w[4]);
    store_q7_word_at(arr, 40, w[5]);
    store_q7_word_at(arr, 48, w[6]);
    store_q7_word_at(arr, 56, w[7]);
}

/// Returns `true` if all 8 `u64` SWAR words in `w` are identical (`w[0] == w[1] == ... == w[7]`)
/// using a branchless 8-word XOR-OR reduction tree (`0` branches).
#[inline(always)]
pub const fn is_word_array_uniform_64(w: &[u64; 8]) -> bool {
    let w0 = w[0];
    if w0 != w[1] {
        return false;
    }
    (((w0 ^ w[2]) | (w0 ^ w[3])) | ((w0 ^ w[4]) | (w0 ^ w[5])) | ((w0 ^ w[6]) | (w0 ^ w[7])))
        == 0
}

#[inline(always)]
const fn swar_u8_pair_sum16(packed_u8: u64) -> u64 {
    (packed_u8 & SWAR_EVEN_BYTES_64).wrapping_add((packed_u8 >> 8) & SWAR_EVEN_BYTES_64)
}

/// Computes the rounded arithmetic mean Q7 phase across 64 heads directly from two 256-bit
/// AVX2 vectors `(v0, v1)` using unsigned byte SAD (`vpsadbw`) against zero after sign-bit
/// flipping (`^ 0x80`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_q7_mean_i8_from_vec64(
    v0: core::arch::x86_64::__m256i,
    v1: core::arch::x86_64::__m256i,
) -> gcso_q7_t {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe {
        use core::arch::x86_64::{
            _mm256_add_epi64, _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_sad_epu8,
            _mm256_set1_epi8, _mm256_setzero_si256, _mm256_xor_si256, _mm_add_epi64,
            _mm_cvtsi128_si64, _mm_unpackhi_epi64,
        };
        let sign = _mm256_set1_epi8(-128i8);
        let zero = _mm256_setzero_si256();
        let sad0 = _mm256_sad_epu8(_mm256_xor_si256(v0, sign), zero);
        let sad1 = _mm256_sad_epu8(_mm256_xor_si256(v1, sign), zero);
        let sum256 = _mm256_add_epi64(sad0, sad1);
        let sum128 = _mm_add_epi64(
            _mm256_castsi256_si128(sum256),
            _mm256_extracti128_si256::<1>(sum256),
        );
        let sum64 = _mm_add_epi64(sum128, _mm_unpackhi_epi64(sum128, sum128));
        let total_u8 = _mm_cvtsi128_si64(sum64) as i32;
        let signed_sum = total_u8 - (64 * 128);
        let rounded = (signed_sum + branchless_select_i32(signed_sum >= 0, 32, -32)) / 64;
        branchless_select_i32(
            rounded < -128,
            -128,
            branchless_select_i32(rounded > 127, 127, rounded),
        ) as gcso_q7_t
    }
}

/// Computes the rounded arithmetic mean Q7 phase across 8 packed `u64` SWAR words (64 heads)
/// using a pre-evaluated word-uniformity flag (`words_is_uniform`).
#[inline(always)]
pub const fn swar_q7_mean_i8_from_words_with_uniform_flag_64(
    words: &[u64; 8],
    words_is_uniform: bool,
) -> gcso_q7_t {
    if words_is_uniform {
        let w0 = words[0];
        if w0 == w0.rotate_right(8) {
            return (w0 & 0xFF) as u8 as gcso_q7_t;
        }
        let signed_sum8 = (swar_u8_sum_x8(w0 ^ SWAR_SIGN_MASK_64) as i32) - (8 * 128);
        let rounded8 = (signed_sum8 + branchless_select_i32(signed_sum8 >= 0, 4, -4)) / 8;
        return branchless_select_i32(
            rounded8 < -128,
            -128,
            branchless_select_i32(rounded8 > 127, 127, rounded8),
        ) as gcso_q7_t;
    }
    let p01 = swar_u8_pair_sum16(words[0] ^ SWAR_SIGN_MASK_64)
        .wrapping_add(swar_u8_pair_sum16(words[1] ^ SWAR_SIGN_MASK_64));
    let p23 = swar_u8_pair_sum16(words[2] ^ SWAR_SIGN_MASK_64)
        .wrapping_add(swar_u8_pair_sum16(words[3] ^ SWAR_SIGN_MASK_64));
    let p45 = swar_u8_pair_sum16(words[4] ^ SWAR_SIGN_MASK_64)
        .wrapping_add(swar_u8_pair_sum16(words[5] ^ SWAR_SIGN_MASK_64));
    let p67 = swar_u8_pair_sum16(words[6] ^ SWAR_SIGN_MASK_64)
        .wrapping_add(swar_u8_pair_sum16(words[7] ^ SWAR_SIGN_MASK_64));
    let sum16_all = p01.wrapping_add(p23).wrapping_add(p45.wrapping_add(p67));
    let total_u8 = (sum16_all.wrapping_mul(SWAR_SUM_U16_64) >> 48) as i32;
    let signed_sum = total_u8 - (64 * 128);
    let rounded = (signed_sum + branchless_select_i32(signed_sum >= 0, 32, -32)) / 64;
    branchless_select_i32(
        rounded < -128,
        -128,
        branchless_select_i32(rounded > 127, 127, rounded),
    ) as gcso_q7_t
}

/// Computes the rounded arithmetic mean Q7 phase across all 8 packed `u64` SWAR words (64 heads)
/// in strictly constant time (`0` data-dependent branches).
#[inline(always)]
pub const fn swar_q7_mean_i8_from_words_64(words: &[u64; 8]) -> gcso_q7_t {
    swar_q7_mean_i8_from_words_with_uniform_flag_64(words, is_word_array_uniform_64(words))
}

/// Returns `true` if any of the 64 Q7 head phases in `phases` is non-zero (`0` branches).
#[inline(always)]
pub const fn is_phase_profile_nonzero_64(phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> bool {
    let w = load_q7_words_64(phases);
    (((w[0] | w[1]) | (w[2] | w[3])) | ((w[4] | w[5]) | (w[6] | w[7]))) != 0
}

/// Accumulates `deltas` into `acc` across all 64 attention heads using 8-word SWAR saturating addition.
#[inline(always)]
pub const fn branchless_q7_accumulate_64(
    acc: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    deltas: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) {
    let a = load_q7_words_64(acc);
    let d = load_q7_words_64(deltas);
    store_q7_words_64(
        acc,
        &[
            swar_q7_add_sat_x8(a[0], d[0]),
            swar_q7_add_sat_x8(a[1], d[1]),
            swar_q7_add_sat_x8(a[2], d[2]),
            swar_q7_add_sat_x8(a[3], d[3]),
            swar_q7_add_sat_x8(a[4], d[4]),
            swar_q7_add_sat_x8(a[5], d[5]),
            swar_q7_add_sat_x8(a[6], d[6]),
            swar_q7_add_sat_x8(a[7], d[7]),
        ],
    );
}

/// Subtracts `b` from `a` across all 64 attention heads using 8-word SWAR saturating subtraction.
#[inline(always)]
pub const fn branchless_q7_sub_64(
    a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let wa = load_q7_words_64(a);
    let wb = load_q7_words_64(b);
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    store_q7_words_64(
        &mut out,
        &[
            swar_q7_sub_sat_x8(wa[0], wb[0]),
            swar_q7_sub_sat_x8(wa[1], wb[1]),
            swar_q7_sub_sat_x8(wa[2], wb[2]),
            swar_q7_sub_sat_x8(wa[3], wb[3]),
            swar_q7_sub_sat_x8(wa[4], wb[4]),
            swar_q7_sub_sat_x8(wa[5], wb[5]),
            swar_q7_sub_sat_x8(wa[6], wb[6]),
            swar_q7_sub_sat_x8(wa[7], wb[7]),
        ],
    );
    out
}

/// Conjugates all 64 Q7 head phases ($-\boldsymbol{\Delta\theta}$) using 8-word SWAR saturating negation.
#[inline(always)]
pub const fn branchless_q7_conjugate_64(
    phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let w = load_q7_words_64(phases);
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    store_q7_words_64(
        &mut out,
        &[
            swar_q7_neg_sat_x8(w[0]),
            swar_q7_neg_sat_x8(w[1]),
            swar_q7_neg_sat_x8(w[2]),
            swar_q7_neg_sat_x8(w[3]),
            swar_q7_neg_sat_x8(w[4]),
            swar_q7_neg_sat_x8(w[5]),
            swar_q7_neg_sat_x8(w[6]),
            swar_q7_neg_sat_x8(w[7]),
        ],
    );
    out
}

/// Scales all 64 Q7 head phases in-place by `scale_u7 / 128` using 8-word SWAR arithmetic.
#[inline(always)]
pub const fn branchless_q7_scale_64(phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS], scale_u7: u8) {
    let gt_128 = scale_u7 > 128;
    let s = branchless_select_u8(gt_128, scale_u7.wrapping_sub(128), scale_u7) as u64;
    let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let w = load_q7_words_64(phases);
    let s0 = swar_q7_scale_le128_with_bias_x8(w[0], s, bias_restore);
    let s1 = swar_q7_scale_le128_with_bias_x8(w[1], s, bias_restore);
    let s2 = swar_q7_scale_le128_with_bias_x8(w[2], s, bias_restore);
    let s3 = swar_q7_scale_le128_with_bias_x8(w[3], s, bias_restore);
    let s4 = swar_q7_scale_le128_with_bias_x8(w[4], s, bias_restore);
    let s5 = swar_q7_scale_le128_with_bias_x8(w[5], s, bias_restore);
    let s6 = swar_q7_scale_le128_with_bias_x8(w[6], s, bias_restore);
    let s7 = swar_q7_scale_le128_with_bias_x8(w[7], s, bias_restore);

    store_q7_words_64(
        phases,
        &[
            swar_q7_add_sat_x8(s0, branchless_select_u64(gt_128, w[0], 0)),
            swar_q7_add_sat_x8(s1, branchless_select_u64(gt_128, w[1], 0)),
            swar_q7_add_sat_x8(s2, branchless_select_u64(gt_128, w[2], 0)),
            swar_q7_add_sat_x8(s3, branchless_select_u64(gt_128, w[3], 0)),
            swar_q7_add_sat_x8(s4, branchless_select_u64(gt_128, w[4], 0)),
            swar_q7_add_sat_x8(s5, branchless_select_u64(gt_128, w[5], 0)),
            swar_q7_add_sat_x8(s6, branchless_select_u64(gt_128, w[6], 0)),
            swar_q7_add_sat_x8(s7, branchless_select_u64(gt_128, w[7], 0)),
        ],
    );
}

/// Applies the 8-lane SWAR QDPS discrete phase step cutoff filter across all 64 attention heads
/// in strictly constant time (`0` data-dependent branches) via [`swar_q7_qdps_filter_words_64`].
#[inline(always)]
pub const fn branchless_q7_qdps_filter_64(
    phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    min_step_q7: u8,
) {
    if min_step_q7 <= 1 {
        return;
    }
    if min_step_q7 > 127 {
        *phases = [0; MAX_ATTENTION_HEADS];
        return;
    }
    let w = load_q7_words_64(phases);
    let out_w = swar_q7_qdps_filter_words_64(&w, min_step_q7);
    store_q7_words_64(phases, &out_w);
}

/// Branchless QDPS discrete phase step filter in the Q7 integer domain (Section 2.3).
#[inline(always)]
pub const fn branchless_qdps_filter_q7(delta: gcso_q7_t, min_step_q7: i32) -> gcso_q7_t {
    let val = delta as i32;
    let sign_mask = val >> 31;
    let abs_val = (val ^ sign_mask) - sign_mask;
    let keep_mask = (((abs_val >= min_step_q7) & (min_step_q7 <= 127)) as i32).wrapping_neg();
    (val & keep_mask) as gcso_q7_t
}

/// Branchless $\mathcal{O}(1)$ SplitMix64 bijective hash mixer for pointer trails and bitmasks.
#[inline(always)]
pub const fn branchless_splitmix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Branchless hysteresis quantizer converting continuous depth into discrete QDPT level (Section 1.3).
#[inline(always)]
pub const fn branchless_hysteresis_quantize_depth(
    continuous_depth: f32,
    current_discrete_level: u32,
    hysteresis_band: f32,
    max_level: u32,
) -> u32 {
    let cur_f = current_discrete_level as f32;
    let delta = branchless_relu_f32(hysteresis_band);
    let step_up = continuous_depth > (cur_f + 0.5 + delta);
    let step_down = (continuous_depth < (cur_f - 0.5 - delta)) & (current_discrete_level > 0);
    let candidate_up = branchless_select_u32(
        current_discrete_level < max_level,
        current_discrete_level.wrapping_add(1),
        max_level,
    );
    let candidate_down = current_discrete_level.saturating_sub(1);
    branchless_select_u32(
        step_up,
        candidate_up,
        branchless_select_u32(step_down, candidate_down, current_discrete_level),
    )
}

#[inline(always)]
pub(crate) fn load_q7_x8_slice(slice: &[gcso_q7_t], offset: usize) -> u64 {
    debug_assert!(offset + 8 <= slice.len());
    // SAFETY: Caller guarantees `offset + 8 <= slice.len()`.
    let raw = unsafe { core::ptr::read_unaligned(slice.as_ptr().add(offset).cast::<u64>()) };
    u64::from_le(raw)
}

#[inline(always)]
pub(crate) fn store_q7_x8_slice(slice: &mut [gcso_q7_t], offset: usize, word: u64) {
    debug_assert!(offset + 8 <= slice.len());
    // SAFETY: Caller guarantees `offset + 8 <= slice.len()`.
    unsafe {
        core::ptr::write_unaligned(slice.as_mut_ptr().add(offset).cast::<u64>(), word.to_le());
    }
}

#[inline(always)]
pub(crate) fn swar_add_sat_q7_slice_in_place(
    acc: &mut [gcso_q7_t],
    delta: &[gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` are checked before each 32-byte unaligned load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_adds_epi8, _mm256_loadu_si256, _mm256_storeu_si256,
            };
            let acc_p = acc.as_mut_ptr().cast::<__m256i>();
            let del_p = delta.as_ptr().cast::<__m256i>();
            while i + 32 <= len {
                let off = i >> 5;
                let va = _mm256_loadu_si256(acc_p.add(off));
                let vd = _mm256_loadu_si256(del_p.add(off));
                _mm256_storeu_si256(acc_p.add(off), _mm256_adds_epi8(va, vd));
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let w0 = swar_q7_add_sat_x8(load_q7_x8_slice(acc, i), load_q7_x8_slice(delta, i));
        let w1 = swar_q7_add_sat_x8(load_q7_x8_slice(acc, i + 8), load_q7_x8_slice(delta, i + 8));
        store_q7_x8_slice(acc, i, w0);
        store_q7_x8_slice(acc, i + 8, w1);
        i += 16;
    }
    while i + 8 <= len {
        let w = swar_q7_add_sat_x8(load_q7_x8_slice(acc, i), load_q7_x8_slice(delta, i));
        store_q7_x8_slice(acc, i, w);
        i += 8;
    }
    while i < len {
        acc[i] = branchless_q7_add_sat(acc[i], delta[i]);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_sub_sat_q7_slices(
    a: &[gcso_q7_t],
    b: &[gcso_q7_t],
    out: &mut [gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` are checked before each 32-byte unaligned load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_subs_epi8,
            };
            let ap = a.as_ptr().cast::<__m256i>();
            let bp = b.as_ptr().cast::<__m256i>();
            let op = out.as_mut_ptr().cast::<__m256i>();
            while i + 32 <= len {
                let off = i >> 5;
                let va = _mm256_loadu_si256(ap.add(off));
                let vb = _mm256_loadu_si256(bp.add(off));
                _mm256_storeu_si256(op.add(off), _mm256_subs_epi8(va, vb));
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let w0 = swar_q7_sub_sat_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        let w1 = swar_q7_sub_sat_x8(load_q7_x8_slice(a, i + 8), load_q7_x8_slice(b, i + 8));
        store_q7_x8_slice(out, i, w0);
        store_q7_x8_slice(out, i + 8, w1);
        i += 16;
    }
    while i + 8 <= len {
        let w = swar_q7_sub_sat_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        store_q7_x8_slice(out, i, w);
        i += 8;
    }
    while i < len {
        out[i] = branchless_q7_sub_sat(a[i], b[i]);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_circular_diff_q7_slices(
    a: &[gcso_q7_t],
    b: &[gcso_q7_t],
    out: &mut [gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` (`32` or `64` fast paths) are checked before each 32B load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_loadu_si256, _mm256_storeu_si256, _mm256_sub_epi8,
            };
            let ap = a.as_ptr().cast::<__m256i>();
            let bp = b.as_ptr().cast::<__m256i>();
            let op = out.as_mut_ptr().cast::<__m256i>();
            if len == MAX_ATTENTION_HEADS {
                let va0 = _mm256_loadu_si256(ap);
                let vb0 = _mm256_loadu_si256(bp);
                let va1 = _mm256_loadu_si256(ap.add(1));
                let vb1 = _mm256_loadu_si256(bp.add(1));
                _mm256_storeu_si256(op, _mm256_sub_epi8(va0, vb0));
                _mm256_storeu_si256(op.add(1), _mm256_sub_epi8(va1, vb1));
                return;
            }
            if len == 32 {
                let va0 = _mm256_loadu_si256(ap);
                let vb0 = _mm256_loadu_si256(bp);
                _mm256_storeu_si256(op, _mm256_sub_epi8(va0, vb0));
                return;
            }
            while i + 32 <= len {
                let off = i >> 5;
                let va = _mm256_loadu_si256(ap.add(off));
                let vb = _mm256_loadu_si256(bp.add(off));
                _mm256_storeu_si256(op.add(off), _mm256_sub_epi8(va, vb));
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let w0 = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        let w1 = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i + 8), load_q7_x8_slice(b, i + 8));
        store_q7_x8_slice(out, i, w0);
        store_q7_x8_slice(out, i + 8, w1);
        i += 16;
    }
    while i + 8 <= len {
        let w = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        store_q7_x8_slice(out, i, w);
        i += 8;
    }
    while i < len {
        out[i] = branchless_q7_circular_diff(a[i], b[i]);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_circular_diff_and_fold_geodesic_q7_slices(
    a: &[gcso_q7_t],
    b: &[gcso_q7_t],
    out: &mut [gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` (`32` or `64` fast paths) are checked before each 32B load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_add_epi8, _mm256_blendv_epi8, _mm256_loadu_si256,
                _mm256_set1_epi8, _mm256_storeu_si256, _mm256_sub_epi8, _mm256_xor_si256,
            };
            let ap = a.as_ptr().cast::<__m256i>();
            let bp = b.as_ptr().cast::<__m256i>();
            let op = out.as_mut_ptr().cast::<__m256i>();
            let v_128 = _mm256_set1_epi8(-128);
            if len == MAX_ATTENTION_HEADS {
                let vd0 = _mm256_sub_epi8(_mm256_loadu_si256(ap), _mm256_loadu_si256(bp));
                let vd1 =
                    _mm256_sub_epi8(_mm256_loadu_si256(ap.add(1)), _mm256_loadu_si256(bp.add(1)));
                let m0 = _mm256_xor_si256(vd0, _mm256_add_epi8(vd0, vd0));
                let m1 = _mm256_xor_si256(vd1, _mm256_add_epi8(vd1, vd1));
                _mm256_storeu_si256(op, _mm256_blendv_epi8(vd0, _mm256_sub_epi8(v_128, vd0), m0));
                _mm256_storeu_si256(
                    op.add(1),
                    _mm256_blendv_epi8(vd1, _mm256_sub_epi8(v_128, vd1), m1),
                );
                return;
            }
            if len == 32 {
                let vd0 = _mm256_sub_epi8(_mm256_loadu_si256(ap), _mm256_loadu_si256(bp));
                let m0 = _mm256_xor_si256(vd0, _mm256_add_epi8(vd0, vd0));
                _mm256_storeu_si256(op, _mm256_blendv_epi8(vd0, _mm256_sub_epi8(v_128, vd0), m0));
                return;
            }
            while i + 32 <= len {
                let off = i >> 5;
                let vd = _mm256_sub_epi8(
                    _mm256_loadu_si256(ap.add(off)),
                    _mm256_loadu_si256(bp.add(off)),
                );
                let mask = _mm256_xor_si256(vd, _mm256_add_epi8(vd, vd));
                let folded = _mm256_blendv_epi8(vd, _mm256_sub_epi8(v_128, vd), mask);
                _mm256_storeu_si256(op.add(off), folded);
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let diff_w0 = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        let diff_w1 =
            swar_q7_circular_diff_x8(load_q7_x8_slice(a, i + 8), load_q7_x8_slice(b, i + 8));
        store_q7_x8_slice(out, i, swar_q7_fold_s1_geodesic_x8(diff_w0));
        store_q7_x8_slice(out, i + 8, swar_q7_fold_s1_geodesic_x8(diff_w1));
        i += 16;
    }
    while i + 8 <= len {
        let diff_w = swar_q7_circular_diff_x8(load_q7_x8_slice(a, i), load_q7_x8_slice(b, i));
        store_q7_x8_slice(out, i, swar_q7_fold_s1_geodesic_x8(diff_w));
        i += 8;
    }
    while i < len {
        out[i] = branchless_q7_fold_s1_geodesic(branchless_q7_circular_diff(a[i], b[i]));
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_circular_add_and_fold_geodesic_q7_slice_in_place(
    bus: &mut [gcso_q7_t],
    loc: &[gcso_q7_t],
    len: usize,
) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= len` (`32` or `64` fast paths) are checked before each 32B load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_add_epi8, _mm256_blendv_epi8, _mm256_loadu_si256,
                _mm256_set1_epi8, _mm256_storeu_si256, _mm256_sub_epi8, _mm256_xor_si256,
            };
            let bp = bus.as_mut_ptr().cast::<__m256i>();
            let lp = loc.as_ptr().cast::<__m256i>();
            let v_128 = _mm256_set1_epi8(-128);
            if len == MAX_ATTENTION_HEADS {
                let vs0 = _mm256_add_epi8(_mm256_loadu_si256(bp), _mm256_loadu_si256(lp));
                let vs1 =
                    _mm256_add_epi8(_mm256_loadu_si256(bp.add(1)), _mm256_loadu_si256(lp.add(1)));
                let m0 = _mm256_xor_si256(vs0, _mm256_add_epi8(vs0, vs0));
                let m1 = _mm256_xor_si256(vs1, _mm256_add_epi8(vs1, vs1));
                _mm256_storeu_si256(bp, _mm256_blendv_epi8(vs0, _mm256_sub_epi8(v_128, vs0), m0));
                _mm256_storeu_si256(
                    bp.add(1),
                    _mm256_blendv_epi8(vs1, _mm256_sub_epi8(v_128, vs1), m1),
                );
                return;
            }
            if len == 32 {
                let vs0 = _mm256_add_epi8(_mm256_loadu_si256(bp), _mm256_loadu_si256(lp));
                let m0 = _mm256_xor_si256(vs0, _mm256_add_epi8(vs0, vs0));
                _mm256_storeu_si256(bp, _mm256_blendv_epi8(vs0, _mm256_sub_epi8(v_128, vs0), m0));
                return;
            }
            while i + 32 <= len {
                let off = i >> 5;
                let vs = _mm256_add_epi8(
                    _mm256_loadu_si256(bp.add(off)),
                    _mm256_loadu_si256(lp.add(off)),
                );
                let mask = _mm256_xor_si256(vs, _mm256_add_epi8(vs, vs));
                let folded = _mm256_blendv_epi8(vs, _mm256_sub_epi8(v_128, vs), mask);
                _mm256_storeu_si256(bp.add(off), folded);
                i += 32;
            }
        }
    }
    while i + 16 <= len {
        let b0 = load_q7_x8_slice(bus, i);
        let l0 = load_q7_x8_slice(loc, i);
        let b1 = load_q7_x8_slice(bus, i + 8);
        let l1 = load_q7_x8_slice(loc, i + 8);
        store_q7_x8_slice(
            bus,
            i,
            swar_q7_fold_s1_geodesic_x8(swar_q7_circular_add_x8(b0, l0)),
        );
        store_q7_x8_slice(
            bus,
            i + 8,
            swar_q7_fold_s1_geodesic_x8(swar_q7_circular_add_x8(b1, l1)),
        );
        i += 16;
    }
    while i + 8 <= len {
        let bw = load_q7_x8_slice(bus, i);
        let lw = load_q7_x8_slice(loc, i);
        store_q7_x8_slice(
            bus,
            i,
            swar_q7_fold_s1_geodesic_x8(swar_q7_circular_add_x8(bw, lw)),
        );
        i += 8;
    }
    while i < len {
        bus[i] = branchless_q7_fold_s1_geodesic(bus[i].wrapping_add(loc[i]));
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_fold_s1_geodesic_q7_slice_in_place(phases: &mut [gcso_q7_t], limit: usize) {
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Bounds `i + 32 <= limit` (`limit <= phases.len()`) are checked before each 32B load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_add_epi8, _mm256_blendv_epi8, _mm256_loadu_si256,
                _mm256_set1_epi8, _mm256_storeu_si256, _mm256_sub_epi8, _mm256_xor_si256,
            };
            let pp = phases.as_mut_ptr().cast::<__m256i>();
            let v_128 = _mm256_set1_epi8(-128);
            if limit == MAX_ATTENTION_HEADS {
                let v0 = _mm256_loadu_si256(pp);
                let v1 = _mm256_loadu_si256(pp.add(1));
                let m0 = _mm256_xor_si256(v0, _mm256_add_epi8(v0, v0));
                let m1 = _mm256_xor_si256(v1, _mm256_add_epi8(v1, v1));
                _mm256_storeu_si256(pp, _mm256_blendv_epi8(v0, _mm256_sub_epi8(v_128, v0), m0));
                _mm256_storeu_si256(
                    pp.add(1),
                    _mm256_blendv_epi8(v1, _mm256_sub_epi8(v_128, v1), m1),
                );
                return;
            }
            if limit == 32 {
                let v0 = _mm256_loadu_si256(pp);
                let m0 = _mm256_xor_si256(v0, _mm256_add_epi8(v0, v0));
                _mm256_storeu_si256(pp, _mm256_blendv_epi8(v0, _mm256_sub_epi8(v_128, v0), m0));
                return;
            }
            while i + 32 <= limit {
                let off = i >> 5;
                let v = _mm256_loadu_si256(pp.add(off));
                let mask = _mm256_xor_si256(v, _mm256_add_epi8(v, v));
                let folded = _mm256_blendv_epi8(v, _mm256_sub_epi8(v_128, v), mask);
                _mm256_storeu_si256(pp.add(off), folded);
                i += 32;
            }
        }
    }
    while i + 16 <= limit {
        let w0 = load_q7_x8_slice(phases, i);
        let w1 = load_q7_x8_slice(phases, i + 8);
        store_q7_x8_slice(phases, i, swar_q7_fold_s1_geodesic_x8(w0));
        store_q7_x8_slice(phases, i + 8, swar_q7_fold_s1_geodesic_x8(w1));
        i += 16;
    }
    while i + 8 <= limit {
        let w = load_q7_x8_slice(phases, i);
        store_q7_x8_slice(phases, i, swar_q7_fold_s1_geodesic_x8(w));
        i += 8;
    }
    while i < limit {
        phases[i] = branchless_q7_fold_s1_geodesic(phases[i]);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn swar_qdps_filter_q7_slice_in_place(
    phases: &mut [gcso_q7_t],
    limit: usize,
    min_step_q7: i32,
) {
    let clamped_step = branchless_select_i32(
        min_step_q7 > 128,
        128,
        branchless_select_i32(min_step_q7 < 0, 0, min_step_q7),
    ) as u8;
    // Since `|x| >= 1` for every non-zero `gcso_q7_t` lane (`|0| = 0`), `clamped_step <= 1` is an exact identity.
    if clamped_step <= 1 {
        return;
    }
    let end = branchless_select_usize(phases.len() < limit, phases.len(), limit);
    if clamped_step > 127 {
        phases[..end].fill(0);
        return;
    }
    let mut i = 0usize;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `i + 32 <= end <= phases.len()` is checked before each 32-byte unaligned load/store.
        unsafe {
            use core::arch::x86_64::{
                __m256i, _mm256_abs_epi8, _mm256_and_si256, _mm256_cmpeq_epi8,
                _mm256_loadu_si256, _mm256_max_epu8, _mm256_set1_epi8, _mm256_storeu_si256,
            };
            let pp = phases.as_mut_ptr().cast::<__m256i>();
            let v_thresh = _mm256_set1_epi8(clamped_step as i8);
            if end == MAX_ATTENTION_HEADS {
                let v0 = _mm256_loadu_si256(pp);
                let v1 = _mm256_loadu_si256(pp.add(1));
                let a0 = _mm256_abs_epi8(v0);
                let a1 = _mm256_abs_epi8(v1);
                let m0 = _mm256_cmpeq_epi8(_mm256_max_epu8(a0, v_thresh), a0);
                let m1 = _mm256_cmpeq_epi8(_mm256_max_epu8(a1, v_thresh), a1);
                _mm256_storeu_si256(pp, _mm256_and_si256(v0, m0));
                _mm256_storeu_si256(pp.add(1), _mm256_and_si256(v1, m1));
                return;
            }
            if end == 32 {
                let v0 = _mm256_loadu_si256(pp);
                let a0 = _mm256_abs_epi8(v0);
                let m0 = _mm256_cmpeq_epi8(_mm256_max_epu8(a0, v_thresh), a0);
                _mm256_storeu_si256(pp, _mm256_and_si256(v0, m0));
                return;
            }
            while i + 32 <= end {
                let off = i >> 5;
                let v0 = _mm256_loadu_si256(pp.add(off));
                let a0 = _mm256_abs_epi8(v0);
                let m0 = _mm256_cmpeq_epi8(_mm256_max_epu8(a0, v_thresh), a0);
                _mm256_storeu_si256(pp.add(off), _mm256_and_si256(v0, m0));
                i += 32;
            }
        }
    }
    let thresh_x8 = ((clamped_step & 0x7F) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    if i == 0 && end == MAX_ATTENTION_HEADS {
        // SAFETY: `end == MAX_ATTENTION_HEADS <= phases.len()` and `align_of::<gcso_q7_t>() == 1`.
        let arr = unsafe { &mut *phases.as_mut_ptr().cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        let w = load_q7_words_64(arr);
        store_q7_words_64(
            arr,
            &[
                swar_q7_qdps_filter_with_thresh_x8(w[0], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[1], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[2], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[3], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[4], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[5], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[6], thresh_x8, u64::MAX),
                swar_q7_qdps_filter_with_thresh_x8(w[7], thresh_x8, u64::MAX),
            ],
        );
        return;
    }
    while i + 16 <= end {
        let w0 = load_q7_x8_slice(phases, i);
        let w1 = load_q7_x8_slice(phases, i + 8);
        store_q7_x8_slice(
            phases,
            i,
            swar_q7_qdps_filter_with_thresh_x8(w0, thresh_x8, u64::MAX),
        );
        store_q7_x8_slice(
            phases,
            i + 8,
            swar_q7_qdps_filter_with_thresh_x8(w1, thresh_x8, u64::MAX),
        );
        i += 16;
    }
    while i + 8 <= end {
        let w = load_q7_x8_slice(phases, i);
        store_q7_x8_slice(
            phases,
            i,
            swar_q7_qdps_filter_with_thresh_x8(w, thresh_x8, u64::MAX),
        );
        i += 8;
    }
    while i < end {
        phases[i] = branchless_qdps_filter_q7(phases[i], i32::from(clamped_step));
        i += 1;
    }
}

#[inline(always)]
pub(crate) const fn sanitize_inplace_gain_f32(sample_elem: f32, raw_gain: f32) -> f32 {
    let valid_gain = branchless_is_finite_f32(raw_gain);
    let clean_gain = branchless_select_f32(valid_gain, raw_gain, 1.0);
    let snapped_gain = branchless_select_f32(
        branchless_abs_f32(clean_gain - 1.0) > 1.0e-7,
        clean_gain,
        1.0,
    );
    let abs_sample = branchless_abs_f32(sample_elem);
    let abs_gain = branchless_abs_f32(snapped_gain);
    let subnormal_sample_trap =
        (abs_sample > 0.0) & (abs_sample < F32_MIN_POSITIVE_NORMAL) & (abs_gain < 1.0);
    let overflow_trap = (abs_sample > INPLACE_GAIN_OVERFLOW_CEIL) & (abs_gain > 1.0);
    branchless_select_f32(subnormal_sample_trap | overflow_trap, 1.0, snapped_gain)
}

#[inline(always)]
pub(crate) const fn eval_ripa_sincos_q7_fast(
    phase_q7: gcso_q7_t,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if is_default {
        branchless_sincos_q7_ripa_default_lut(phase_q7)
    } else {
        branchless_sincos_q7_ripa_preinv(phase_q7, eff_max, q7_to_norm_scale)
    }
}

#[inline(always)]
pub(crate) const fn eval_geodesic_ripa_sincos_q7_fast(
    phase_q7: gcso_q7_t,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if is_default {
        branchless_sincos_q7_geodesic_ripa_default_lut(phase_q7)
    } else {
        let folded = branchless_q7_fold_s1_geodesic(phase_q7);
        branchless_sincos_q7_ripa_preinv(folded, eff_max, q7_to_norm_scale)
    }
}

/// Evaluates the $SO(2)$-composed $(\sin(\theta_{\mathrm{base}} + \theta_{\mathrm{pspm}}), \cos(\theta_{\mathrm{base}} + \theta_{\mathrm{pspm}}))$
/// for a single attention head, using two 1-cycle LUT loads when both RIPA clamps equal
/// [`DEFAULT_RIPA_CLAMP_MAX_RAD`], or summing the two `tanh`-clamped radian angles and evaluating
/// `sincos_ripa_poly` **once** instead of twice when custom RIPA clamps are active.
#[inline(always)]
pub(crate) const fn eval_fused_ripa_pspm_sincos_q7_fast(
    base_q7: gcso_q7_t,
    geodesic_base: bool,
    is_default_base: bool,
    eff_base_max: f32,
    base_norm_scale: f32,
    pspm_q7: gcso_q7_t,
    geodesic_pspm: bool,
    is_default_pspm: bool,
    eff_pspm_max: f32,
    pspm_norm_scale: f32,
) -> (f32, f32) {
    let b_q7 = if geodesic_base {
        branchless_q7_fold_s1_geodesic(base_q7)
    } else {
        base_q7
    };
    let p_q7 = if geodesic_pspm {
        branchless_q7_fold_s1_geodesic(pspm_q7)
    } else {
        pspm_q7
    };
    if is_default_base & is_default_pspm {
        if b_q7 == p_q7 {
            return Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF[(b_q7 as u8) as usize];
        }
        if (base_q7 == pspm_q7) & (!geodesic_base) & geodesic_pspm {
            return Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF[(base_q7 as u8) as usize];
        }
        let lut = Q7_DEFAULT_RIPA_SINCOS_REF;
        let (s_b, c_b) = lut[(b_q7 as u8) as usize];
        let (s_p, c_p) = lut[(p_q7 as u8) as usize];
        return (s_b * c_p + c_b * s_p, c_b * c_p - s_b * s_p);
    }
    if (b_q7 == p_q7) & (eff_base_max.to_bits() == eff_pspm_max.to_bits()) {
        if b_q7 == 0 {
            return (0.0, 1.0);
        }
        let theta_comb =
            eval_ripa_clamped_theta_rad_preinv(b_q7, eff_base_max + eff_base_max, base_norm_scale);
        let abs_comb = branchless_abs_f32(theta_comb);
        return if abs_comb <= FRAC_PI_2 {
            sincos_ripa_poly(theta_comb)
        } else {
            sincos_reduced_pi(theta_comb)
        };
    }
    let theta_b = if b_q7 == 0 {
        0.0
    } else {
        eval_ripa_clamped_theta_rad_preinv(b_q7, eff_base_max, base_norm_scale)
    };
    let theta_p = if p_q7 == 0 {
        0.0
    } else {
        eval_ripa_clamped_theta_rad_preinv(p_q7, eff_pspm_max, pspm_norm_scale)
    };
    let theta_comb = theta_b + theta_p;
    let abs_comb = branchless_abs_f32(theta_comb);
    if abs_comb <= FRAC_PI_2 {
        sincos_ripa_poly(theta_comb)
    } else {
        sincos_reduced_pi(theta_comb)
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
    assert!(size_of::<PprcKeyframePacket>() == 128);
    assert!(align_of::<PprcKeyframePacket>() == 64);
    assert!(size_of::<PprcMotionDeltaPacket>() == 64);
    assert!(align_of::<PprcMotionDeltaPacket>() == 64);
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