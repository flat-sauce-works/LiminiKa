// name: src/core/src/traits/primitives/constants_pod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Layer 0 mathematical/capacity constants, cognitive aperture masks, slot codecs, and Pod safety contracts.

use super::*;
#[allow(unused_imports)]
use super::{lut_crc32::*, packets_ffi::*, scalar_math::*, swar_ops::*};

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

/// Number of fast-path shortcut slots in a `gcso_ledger_slot_t` (`4`).
pub const FAST_PATH_SHORTCUT_SLOTS: usize = 4;

/// Byte length of the circular mini telemetry ledger in a `gcso_ledger_slot_t` (`8`).
pub const LEDGER_HISTORY_LEN: usize = 8;

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

// ===================================================================
// 2b. Liminal Cognitive Aperture Masks (Type-Level Horizon Boundaries)
// ===================================================================

/// Cognitive aperture bit for Origin 1: hardware & QoS resource constraints (`SystemConstraintSignal`).
pub const APERTURE_SYSTEM_CONSTRAINT: u32 = 1 << 0;

/// Cognitive aperture bit for Origin 2A: factual knowledge grounding (`ObjectiveFactSignal`).
pub const APERTURE_OBJECTIVE_FACT: u32 = 1 << 1;

/// Cognitive aperture bit for Origin 2B: subjective persona grounding (`SubjectivePersonaSignal`).
pub const APERTURE_SUBJECTIVE_PERSONA: u32 = 1 << 2;

/// Cognitive aperture bit for Origin 2C: contradiction guard grounding (`ContradictionGuardSignal`).
pub const APERTURE_CONTRADICTION_GUARD: u32 = 1 << 3;

/// Combined causal provenance cognitive aperture across system and grounding origins (`0x0000_000F`).
pub const APERTURE_CAUSAL_PROVENANCE: u32 = 0x0000_000F;

/// Cognitive aperture bit for recurrent cognitive circuit plasticity (`CognitiveTrajectoryLedger`).
pub const APERTURE_COGNITIVE_CIRCUIT: u32 = 1 << 4;

/// Cognitive aperture bit for phase KV context memory (`RamPhaseKvCacheOps`).
pub const APERTURE_KV_PHASE_CACHE: u32 = 1 << 5;

/// Cognitive aperture bit for perspective-relative attractor coupling (`RelativeCouplingMatrix`).
pub const APERTURE_RELATIVE_COUPLING: u32 = 1 << 6;

/// Cognitive aperture bit for dissipative overflow crystallization (`CrystallizedAttractorRecord`).
pub const APERTURE_DISSIPATIVE_CRYSTAL: u32 = 1 << 7;

/// Wide composite cognitive aperture spanned by [`crate::traits::gateway::ManifoldGatewaySignal`] (`0x0000_00FF`).
pub const APERTURE_MANIFOLD_GATEWAY: u32 = 0x0000_00FF;

/// Cognitive aperture bit for $SO(2)$ phase steering (`PhaseSteering`).
pub const APERTURE_PHASE_STEERING: u32 = 1 << 8;

/// Cognitive aperture bit for Fact/Logic/Explore sub-head routing (`SubHeadRouter`).
pub const APERTURE_SUBHEAD_ROUTING: u32 = 1 << 9;

/// Cognitive aperture bit for Rank-1 hidden residual injection (`Rank1ResidualAdapter`).
pub const APERTURE_RANK1_RESIDUAL: u32 = 1 << 10;

/// Cognitive aperture bit for tagged pointer transitions & trails (`PointerActionHub`).
pub const APERTURE_POINTER_TRAIL: u32 = 1 << 11;

/// Cognitive aperture bit for fast-path L1 ledger shortcuts (`FastPathLedger`).
pub const APERTURE_FAST_PATH_LEDGER: u32 = 1 << 12;

/// Cognitive aperture bit for 8-granule stigmergic consensus (`StigmergicReconstitutionOps`).
pub const APERTURE_STIGMERGIC_CONSENSUS: u32 = 1 << 13;

/// Cognitive aperture bit for sheaf cohomology & EDBC entropy control.
pub const APERTURE_COHOMOLOGY_EDBC: u32 = 1 << 14;

/// Cognitive aperture bit for zero-copy `.gcso` mapped storage & ZiMMS I/O.
pub const APERTURE_MAPPED_STORAGE: u32 = 1 << 15;

/// Broadest operational cognitive aperture spanned by [`crate::traits::action_hub::ActionHub`] (`0x0000_FFFF`).
pub const APERTURE_ACTION_HUB_WIDE: u32 = 0x0000_FFFF;

/// Number of dynamic channel slots tracked by `LiminalApertureField`.
pub const NUM_LIMINAL_APERTURE_CHANNELS: usize = 8;

/// Number of orthogonal self-tuning autonomous core layers (`Layer 0: Liminal`, `Layer 1: Stigmergic`).
pub const NUM_CORE_AUTONOMOUS_LAYERS: usize = 2;

/// Autonomous Core Layer 0 identifier: Epistemic Horizon & Trait Aperture (`LiminalApertureField`).
pub const CORE_LAYER_LIMINAL: usize = 0;

/// Autonomous Core Layer 1 identifier: Stigmergic Granule Approximation, Venn Overlap & Serendipity (`StigmergicGranuleField`).
pub const CORE_LAYER_STIGMERGIC: usize = 1;

/// Ego intervention trigger bit (`0x01`): Counter-vortex orbital thrust when attractor pull threatens capture.
pub const EGO_TRIGGER_VORTEX_SWALLOW: u8 = 1 << 0;
/// Ego intervention trigger bit (`0x02`): Defiant intervention under internal contradiction (`H^1 != 0`).
pub const EGO_TRIGGER_INTERNAL_CONFLICT: u8 = 1 << 1;
/// Ego intervention trigger bit (`0x04`): Transgressive step when standing at a `Liminal` horizon boundary.
pub const EGO_TRIGGER_LIMINAL_EXTERIOR: u8 = 1 << 2;
/// Ego intervention trigger bit (`0x08`): Boundary damping when temporal boundary velocity oscillates.
pub const EGO_TRIGGER_BOUNDARY_CHATTER: u8 = 1 << 3;
/// Ego intervention trigger bit (`0x10`): Self-initiated perturbation when boundaries stagnate completely.
pub const EGO_TRIGGER_BOUNDARY_STAGNATION: u8 = 1 << 4;
/// Default active bitmask enabling all 5 deterministic `LiminalEgoField` triggers (`0x1F`).
pub const EGO_TRIGGER_ALL_DEFAULT: u8 = 0x1F;

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

/// Action Mode Bit 3 (`0x08`): Enable single-pass sub-head router sub-head group routing (Fact / Logic / Explore).
pub const ACTION_MODE_SUBHEAD_ROUTE: u8 = 1 << 3;

/// Action Mode Bit 4 (`0x10`): Enable Rank-1 residual Dynamic Rank-1 outer product residual injection ($\mathbf{u}\mathbf{v}^T$).
pub const ACTION_MODE_RANK1_INJECT: u8 = 1 << 4;

/// Composite Action Mode preset for passive sidecar observation (`0x00`).
pub const ACTION_PRESET_PASSIVE_SIDECAR: u8 = 0x00;

/// Composite Action Mode preset for lightweight express phase-steered sidecar execution (`0x01`).
pub const ACTION_PRESET_EXPRESS_PHASE: u8 = ACTION_MODE_DPSR_STEER;

/// Composite Action Mode preset for low-entropy inertial layer-skipping fast-forward (`0x03`).
pub const ACTION_PRESET_INERTIAL_SKIP: u8 = ACTION_MODE_DPSR_STEER | ACTION_MODE_LAYER_SKIP;

/// Composite Action Mode preset for anti-hallucination repulsion and sub-head router exploration routing (`0x0D`).
pub const ACTION_PRESET_REPULSION_ESCAPE: u8 =
    ACTION_MODE_DPSR_STEER | ACTION_MODE_PHASE_REPULSE | ACTION_MODE_SUBHEAD_ROUTE;

/// Composite Action Mode preset for full-stack deep persona & knowledge orchestration (`0x19`).
pub const ACTION_PRESET_FULL_ORCHESTRATION: u8 =
    ACTION_MODE_DPSR_STEER | ACTION_MODE_SUBHEAD_ROUTE | ACTION_MODE_RANK1_INJECT;

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
pub(super) struct PtrAlignCheck<T, const ALIGN: usize>(PhantomData<T>);
impl<T, const ALIGN: usize> PtrAlignCheck<T, ALIGN> {
    pub(super) const VALID: () = assert!(
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
pub(super) const fn raw_regions_overlap_or_same_zst(
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
/// or mapped-storage memory-mapped boundaries.
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
    gcso_ledger_slot_t,
    gcso_config_t,
    gcso_edbc_state_t,
    gcso_mapped_storage_descriptor_t,
    gcso_subhead_config_t,
    gcso_rank1_residual_descriptor_t,
    gcso_snapshot_header_t,
    gcso_kv_keyframe_header_t,
    KvKeyframePacket,
    KvMotionDeltaPacket,
);

// SAFETY: Fixed-size arrays of `Pod` elements have contiguous layout with no extra padding traps.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

