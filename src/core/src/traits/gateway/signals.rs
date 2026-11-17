// name: src/core/src/traits/gateway/signals.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Gateway provenance signals, 2-origin domain traits (`SystemConstraint`, `ObjectiveFact`,
//! `SubjectivePersona`, `ContradictionGuard`), [`GatewayFrame`] synthesis, and policy traits.

use super::*;
#[allow(unused_imports)]
use super::frame::*;

// 1. Causal Provenance Flags, Root-Cause Codes & Gateway Constants
// ===================================================================

/// Provenance axis bit (`bit 0`): Origin 1 (`SystemConstraintSignal`) contributed to this frame.
pub const CAUSAL_AXIS_SYSTEM_CONSTRAINT: u32 = 1 << 0;
/// Provenance axis bit (`bit 1`): Origin 2-A (`ObjectiveFactSignal`) contributed to this frame.
pub const CAUSAL_AXIS_OBJECTIVE_FACT: u32 = 1 << 1;
/// Provenance axis bit (`bit 2`): Origin 2-B (`SubjectivePersonaSignal`) contributed to this frame.
pub const CAUSAL_AXIS_SUBJECTIVE_PERSONA: u32 = 1 << 2;
/// Provenance axis bit (`bit 3`): Origin 2-C (`ContradictionGuardSignal`) contributed to this frame.
pub const CAUSAL_AXIS_CONTRADICTION_GUARD: u32 = 1 << 3;

/// Provenance cause bit (`bit 8`): System QoS or low-entropy inertial throttle active.
pub const CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE: u32 = 1 << 8;
/// Provenance cause bit (`bit 9`): Objective fact retrieval energy was weak or absent.
pub const CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK: u32 = 1 << 9;
/// Provenance cause bit (`bit 10`): Subjective persona collided with objective fact ($\mathcal{C}_{\mathrm{FP}} < -\tau$).
pub const CAUSAL_CAUSE_PERSONA_FACT_COLLISION: u32 = 1 << 10;
/// Provenance cause bit (`bit 11`): Cohomological obstruction ($H^1 \neq 0$) or Bloom repulsion triggered.
pub const CAUSAL_CAUSE_COHOMOLOGY_REPULSION: u32 = 1 << 11;
/// Provenance cause bit (`bit 12`): Attractor sample count & coherence reached `.gcso` crystallization maturity.
pub const CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED: u32 = 1 << 12;

/// Root-cause diagnosis code `0`: Nominal synergistic orchestration across active origins.
pub const ROOT_CAUSE_NOMINAL: u8 = 0;
/// Root-cause diagnosis code `1`: System QoS / VRAM pressure / inertial layer-skip throttle.
pub const ROOT_CAUSE_SYSTEM_QOS_THROTTLE: u8 = 1;
/// Root-cause diagnosis code `2`: Objective fact retrieval miss or weak grounding energy.
pub const ROOT_CAUSE_FACT_RETRIEVAL_MISS: u8 = 2;
/// Root-cause diagnosis code `3`: Destructive phase collision between Persona and Objective Fact.
pub const ROOT_CAUSE_PERSONA_FACT_COLLISION: u8 = 3;
/// Root-cause diagnosis code `4`: Cohomological obstruction ($H^1 \neq 0$) or Bloom repetition escape.
pub const ROOT_CAUSE_COHOMOLOGY_REPULSION: u8 = 4;

/// Narrow RIPA phase clamp authority (`0.14 rad`) for Objective Fact grounding (`Fact` sub-heads).
pub const DEFAULT_FACT_RIPA_CLAMP_RAD: f32 = 0.14;
/// Wide RIPA phase clamp authority (`0.24 rad`) for Subjective Persona modulation (`Explore/Logic` sub-heads).
pub const DEFAULT_PERSONA_RIPA_CLAMP_RAD: f32 = 0.24;
/// Default destructive phase coherence threshold (`0.25`) detecting Persona-vs-Fact collision ($\mathcal{C}_{\mathrm{FP}} < -0.25$).
pub const DEFAULT_PERSONA_FACT_COLLISION_THRESH: f32 = 0.25;
/// Default minimum cosine phase coherence threshold (`0.35`) for Online Toroidal DP-Means cluster splitting.
pub const DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE: f32 = 0.35;
/// Default cosine phase coherence threshold (`0.85`) for autonomous online consolidation of redundant macro centroids.
pub const DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE: f32 = 0.85;
/// Default secondary-to-primary gravity ratio threshold (`0.55`) triggering Top-2 Liminal Bifurcation Routing.
pub const DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO: f32 = 0.55;

/// Magic identifier (`0x4C545343` = ASCII `"CSTL"`) for `.gcso` [`CrystallizedAttractorRecord`] blocks.
pub const GCSO_CRYSTAL_RECORD_MAGIC: u32 = 0x4C54_5343;
/// Version constant for [`CrystallizedAttractorRecord`] binary serialization (`0x0000_0101` = v0.1.1).
pub const GCSO_CRYSTAL_RECORD_VERSION: u32 = GCSO_ABI_VERSION_HEX;
/// Fixed byte length (`192` bytes = `128B` `gcso_snapshot_header_t` + `64B` `gcso_mapped_storage_descriptor_t`) of Section 0.
pub const GCSO_FOUR_SECTION_HEADER_BYTES: usize = 192;
/// Byte offset (`128`) within the 192-byte Section 0 preamble where [`crate::abi::gcso_mapped_storage_descriptor_t`] is stored.
pub const GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET: usize = 128;

pub(super) const INV_FACT_RIPA_CLAMP_RAD: f32 = 1.0 / DEFAULT_FACT_RIPA_CLAMP_RAD;
pub(crate) const MAX_REGISTER_CACHED_CENTROIDS: usize = 16;

// ===================================================================
// 2. Branchless Phase Projection Helpers (`RawPhaseProjectable`)
// ===================================================================

#[inline(always)]
#[must_use]
pub(super) const fn swar_clamp_q7_min_to_neg127_x8(w: u64) -> u64 {
    let low7 = w & SWAR_LOW7_MASK_64;
    let has_low7 = low7.wrapping_add(SWAR_LOW7_MASK_64) & SWAR_SIGN_MASK_64;
    let is_80 = (w & SWAR_SIGN_MASK_64) & !has_low7;
    w | (is_80 >> 7)
}

/// Projects an arbitrary byte slice (`bytes`) directly into 8 register-resident `u64` SWAR words
/// (`[u64; 8]` representing 64 `gcso_q7_t` heads) using 8-byte SplitMix64 mixing and `-128 -> -127` clamping.
#[inline]
#[must_use]
pub const fn project_bytes_swar_to_words_64(bytes: &[u8]) -> [u64; NUM_STIGMERGIC_GRANULES] {
    let len = bytes.len();
    if len == 0 {
        return [0u64; NUM_STIGMERGIC_GRANULES];
    }
    let mut seed = (len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut i = 0usize;
    while i + 8 <= len {
        let w = u64::from_le_bytes([
            bytes[i],
            bytes[i + 1],
            bytes[i + 2],
            bytes[i + 3],
            bytes[i + 4],
            bytes[i + 5],
            bytes[i + 6],
            bytes[i + 7],
        ]);
        seed = branchless_splitmix64(seed ^ w);
        i += 8;
    }
    if i < len {
        let mut tail = 0u64;
        let mut shift = 0u32;
        while i < len {
            tail |= (bytes[i] as u64) << shift;
            shift += 8;
            i += 1;
        }
        seed = branchless_splitmix64(seed ^ tail);
    }
    let mut words = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut cell = 0usize;
    while cell < NUM_STIGMERGIC_GRANULES {
        let mixed = branchless_splitmix64(
            seed.wrapping_add((cell as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
        );
        words[cell] = swar_clamp_q7_min_to_neg127_x8(mixed);
        cell += 1;
    }
    words
}

/// Projects an arbitrary byte slice (`bytes`) into a 64-head Q7 phase profile (`[gcso_q7_t; 64]`).
#[inline]
#[must_use]
pub const fn project_bytes_swar_to_q7_64(bytes: &[u8]) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let words = project_bytes_swar_to_words_64(bytes);
    words_to_q7_array_64(&words)
}

/// Projects an `f32` embedding slice directly into 8 register-resident `u64` SWAR words (`[u64; 8]`)
/// via pairwise $SO(2)$ angles evaluated with 8-lane SIMD (`atan2_to_q7_word_x8_simd`), avoiding any
/// 64-byte stack array roundtrip when synthesizing `ObjectiveFact<&[f32]>` or `SubjectivePersona<&[f32]>`.
#[inline]
#[must_use]
pub fn project_f32_slice_to_words_64(values: &[f32]) -> [u64; NUM_STIGMERGIC_GRANULES] {
    let n = values.len();
    if n == 0 {
        return [0u64; NUM_STIGMERGIC_GRANULES];
    }
    let mut words = [0u64; NUM_STIGMERGIC_GRANULES];
    if n >= 128 {
        let mut cell = 0usize;
        while cell < NUM_STIGMERGIC_GRANULES {
            let base_i = cell << 4;
            words[cell] = atan2_interleaved_pairs_to_q7_word_x8_simd(&values[base_i..base_i + 16]);
            cell += 1;
        }
        return words;
    }
    if (n >= 2) & ((n & (n - 1)) == 0) {
        if n == 2 {
            let b0 = branchless_atan2_to_q7_fast(values[1], values[0]) as u8;
            return [(b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64); NUM_STIGMERGIC_GRANULES];
        }
        if n == 4 {
            let b0 = branchless_atan2_to_q7_fast(values[1], values[0]) as u8 as u64;
            let b1 = branchless_atan2_to_q7_fast(values[3], values[2]) as u8 as u64;
            let pair16 = b0 | (b1 << 8);
            return [pair16.wrapping_mul(0x0001_0001_0001_0001); NUM_STIGMERGIC_GRANULES];
        }
        if n == 8 {
            let b0 = branchless_atan2_to_q7_fast(values[1], values[0]) as u8 as u64;
            let b1 = branchless_atan2_to_q7_fast(values[3], values[2]) as u8 as u64;
            let b2 = branchless_atan2_to_q7_fast(values[5], values[4]) as u8 as u64;
            let b3 = branchless_atan2_to_q7_fast(values[7], values[6]) as u8 as u64;
            let quad32 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            return [quad32 | (quad32 << 32); NUM_STIGMERGIC_GRANULES];
        }
        if n == 16 {
            let w0 = atan2_interleaved_pairs_to_q7_word_x8_simd(&values[..16]);
            return [w0; NUM_STIGMERGIC_GRANULES];
        }
        let unique_cells = branchless_select_usize(
            (n >> 4) < NUM_STIGMERGIC_GRANULES,
            n >> 4,
            NUM_STIGMERGIC_GRANULES,
        );
        let mut cell = 0usize;
        while cell < unique_cells {
            let base_i = cell << 4;
            words[cell] = atan2_interleaved_pairs_to_q7_word_x8_simd(&values[base_i..base_i + 16]);
            cell += 1;
        }
        while cell < NUM_STIGMERGIC_GRANULES {
            words[cell] = words[cell & (unique_cells - 1)];
            cell += 1;
        }
        return words;
    }
    let mut cell = 0usize;
    while cell < NUM_STIGMERGIC_GRANULES {
        let base_h = cell << 3;
        let sin_arr = [
            values[((base_h) << 1 | 1) % n],
            values[((base_h + 1) << 1 | 1) % n],
            values[((base_h + 2) << 1 | 1) % n],
            values[((base_h + 3) << 1 | 1) % n],
            values[((base_h + 4) << 1 | 1) % n],
            values[((base_h + 5) << 1 | 1) % n],
            values[((base_h + 6) << 1 | 1) % n],
            values[((base_h + 7) << 1 | 1) % n],
        ];
        let cos_arr = [
            values[((base_h) << 1) % n],
            values[((base_h + 1) << 1) % n],
            values[((base_h + 2) << 1) % n],
            values[((base_h + 3) << 1) % n],
            values[((base_h + 4) << 1) % n],
            values[((base_h + 5) << 1) % n],
            values[((base_h + 6) << 1) % n],
            values[((base_h + 7) << 1) % n],
        ];
        words[cell] = atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr);
        cell += 1;
    }
    words
}

/// Projects an `f32` embedding slice into a 64-head Q7 phase profile (`[gcso_q7_t; 64]`) via
/// pairwise $SO(2)$ angles evaluated with 8-lane SIMD (`atan2_to_q7_word_x8_simd`).
#[inline]
#[must_use]
pub fn project_f32_slice_to_q7_64(values: &[f32]) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let words = project_f32_slice_to_words_64(values);
    words_to_q7_array_64(&words)
}

/// Computes the circular mean Q7 phase tag across 64 attention heads in branchless mode.
#[inline(always)]
#[must_use]
pub fn mean_q7_phase_tag_64(phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> gcso_q7_t {
    branchless_q7_circular_resultant_64(phases).1
}

#[inline(always)]
#[must_use]
pub(crate) fn update_aligned64_blocks_crc32_raw(mut crc: u32, bytes: &[u8]) -> u32 {
    let len = bytes.len();
    let num_blocks = len >> 6;
    // SAFETY: Caller guarantees `bytes` is 8-byte aligned and `len` is a multiple of 64.
    let blocks =
        unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<[u64; 8]>(), num_blocks) };
    let mut b = 0usize;
    while b + 4 <= num_blocks {
        let z0 = crc32_block64_zero_seed_words(&blocks[b]);
        let z1 = crc32_block64_zero_seed_words(&blocks[b + 1]);
        let z01 = (if z0 == 0 { 0 } else { crc32_step_l4(z0) }) ^ z1;
        let z2 = crc32_block64_zero_seed_words(&blocks[b + 2]);
        let z3 = crc32_block64_zero_seed_words(&blocks[b + 3]);
        let c23 = (if z2 == 0 { 0 } else { crc32_step_l4(z2) }) ^ z3;
        crc = crc32_step_l16(crc) ^ (if z01 == 0 { 0 } else { crc32_step_l8(z01) }) ^ c23;
        b += 4;
    }
    while b + 2 <= num_blocks {
        let z0 = crc32_block64_zero_seed_words(&blocks[b]);
        let z1 = crc32_block64_zero_seed_words(&blocks[b + 1]);
        let z01 = (if z0 == 0 { 0 } else { crc32_step_l4(z0) }) ^ z1;
        crc = crc32_step_l8(crc) ^ z01;
        b += 2;
    }
    if b < num_blocks {
        crc = crc32_step_block64_words(crc, &blocks[b]);
    }
    crc
}

#[inline(always)]
#[must_use]
pub(crate) fn compute_aligned64_payload_crc32_ieee(bytes: &[u8]) -> u32 {
    #[cfg(target_endian = "little")]
    {
        if (bytes.len() & 63) == 0 && is_aligned_to(bytes, align_of::<u64>()) {
            return !update_aligned64_blocks_crc32_raw(0xFFFF_FFFFu32, bytes);
        }
    }
    branchless_crc32_ieee(bytes)
}

#[inline(always)]
#[must_use]
pub(crate) const fn words_to_q7_array_64(
    words: &[u64; NUM_STIGMERGIC_GRANULES],
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    // SAFETY: `[u64; 8]` and `[i8; 64]` have identical size (64 bytes) and Pod bit validity.
    unsafe {
        core::mem::transmute::<[u64; NUM_STIGMERGIC_GRANULES], [gcso_q7_t; MAX_ATTENTION_HEADS]>(
            *words,
        )
    }
}

#[inline(always)]
#[must_use]
pub(crate) const fn fast_mean_q7_tag_from_words_64(words: &[u64; NUM_STIGMERGIC_GRANULES]) -> gcso_q7_t {
    let raw = swar_q7_mean_i8_from_words_64(words);
    branchless_select_i8(raw == -128, -127, raw)
}

/// Trait for raw payloads that can be projected onto the 64-head $SO(2)^{64}$ Q7 phase manifold.
pub trait RawPhaseProjectable: BranchlessKernel {
    /// Projects `self` onto a 64-head Q7 phase array (`[gcso_q7_t; 64]`).
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS];

    /// Projects `self` directly into 8 register-resident `u64` SWAR words (`[u64; 8]`).
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        load_q7_words_64(&self.project_to_q7_64())
    }
}

impl RawPhaseProjectable for () {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        [0i8; MAX_ATTENTION_HEADS]
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        [0u64; NUM_STIGMERGIC_GRANULES]
    }
}

impl RawPhaseProjectable for [gcso_q7_t; MAX_ATTENTION_HEADS] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        *self
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        load_q7_words_64(self)
    }
}

impl RawPhaseProjectable for [u8] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self)
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_bytes_swar_to_words_64(self)
    }
}

impl<const N: usize> RawPhaseProjectable for [u8; N] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self.as_slice())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_bytes_swar_to_words_64(self.as_slice())
    }
}

impl RawPhaseProjectable for str {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self.as_bytes())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_bytes_swar_to_words_64(self.as_bytes())
    }
}

impl RawPhaseProjectable for &str {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self.as_bytes())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_bytes_swar_to_words_64(self.as_bytes())
    }
}

impl RawPhaseProjectable for &[u8] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self)
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_bytes_swar_to_words_64(self)
    }
}

impl RawPhaseProjectable for [f32] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_64(self)
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_f32_slice_to_words_64(self)
    }
}

impl RawPhaseProjectable for &[f32] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_64(self)
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_f32_slice_to_words_64(self)
    }
}

impl<const N: usize> RawPhaseProjectable for [f32; N] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_64(self.as_slice())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_f32_slice_to_words_64(self.as_slice())
    }
}

impl<'a, const ALIGN: usize> RawPhaseProjectable for AlignedSlice<'a, f32, ALIGN> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_64(self.as_slice())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_f32_slice_to_words_64(self.as_slice())
    }
}

impl<'a, const ALIGN: usize> RawPhaseProjectable for AlignedSlice<'a, u8, ALIGN> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self.as_slice())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        project_bytes_swar_to_words_64(self.as_slice())
    }
}

impl<'a, const ALIGN: usize> RawPhaseProjectable for AlignedSlice<'a, gcso_q7_t, ALIGN> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        let s = self.as_slice();
        if s.len() >= MAX_ATTENTION_HEADS {
            let mut out = [0i8; MAX_ATTENTION_HEADS];
            out.copy_from_slice(&s[..MAX_ATTENTION_HEADS]);
            out
        } else {
            project_bytes_swar_to_q7_64(gcso_q7_t::slice_as_bytes(s))
        }
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        let s = self.as_slice();
        if s.len() >= MAX_ATTENTION_HEADS {
            // SAFETY: `s.len() >= 64` and `gcso_q7_t` has alignment 1.
            let arr_ref = unsafe { &*(s.as_ptr().cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>()) };
            load_q7_words_64(arr_ref)
        } else {
            project_bytes_swar_to_words_64(gcso_q7_t::slice_as_bytes(s))
        }
    }
}

impl RawPhaseProjectable for gcso_pointer_trail_t {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.accumulated_phase_delta
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *trail_phase_words_ref(self)
    }
}

impl RawPhaseProjectable for GatewayFrame {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *self.phase_words_ref()
    }
}

impl RawPhaseProjectable for CrystallizedAttractorRecord {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_profile
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *self.phase_words_ref()
    }
}

// ===================================================================
// 3. Root Supertrait (`CausalProvenanceSignal`) & 16-Byte Report
// ===================================================================

/// 16-byte (`#[repr(C, align(16))]`) [`Pod`] diagnostic report enabling $\mathcal{O}(1)$
/// root-cause ("Why") isolation across System Constraints, Objective Facts, Subjective Persona,
/// and Contradiction Guards.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CausalProvenanceReport {
    /// Bitwise OR of active origin axes (`CAUSAL_AXIS_*`) and diagnostic causes (`CAUSAL_CAUSE_*`).
    pub provenance_bits: u32,
    /// Cosine phase coherence $\mathcal{C}_{\mathrm{FP}} \in [-1, 1]$ between Objective Fact and Subjective Persona.
    pub fact_persona_coherence: f32,
    /// Peak anomaly or repulsion severity in $[0, 1]$ across all contributing signals.
    pub anomaly_severity: f32,
    /// Primary root-cause classification code (`ROOT_CAUSE_*`).
    pub dominant_cause_code: u8,
    /// Dominant 3-bit working lane ID (`ATTRACTOR_SLOT_*`, `0..=7`).
    pub dominant_slot: u8,
    /// Resolved 5-bit `ACTION_MODE_*` execution flags.
    pub action_mode_flags: u8,
    /// Active 8-cell Micro-Swarm mask (`0x00..=0xFF`).
    pub active_swarm_mask: u8,
}

// SAFETY: `CausalProvenanceReport` is `#[repr(C, align(16))]` with 16 bytes of primitive fields and zero padding.
unsafe impl Pod for CausalProvenanceReport {}

impl CausalProvenanceReport {
    /// Returns `true` if all bits in `flag_mask` are set in `self.provenance_bits`.
    #[inline(always)]
    #[must_use]
    pub const fn has_flag(&self, flag_mask: u32) -> bool {
        (self.provenance_bits & flag_mask) == flag_mask
    }

    /// Returns `true` when a destructive phase collision between Objective Fact and Subjective Persona was detected.
    #[inline(always)]
    #[must_use]
    pub const fn is_persona_fact_collision(&self) -> bool {
        (self.provenance_bits & CAUSAL_CAUSE_PERSONA_FACT_COLLISION) != 0
    }

    /// Returns `true` when cohomological obstruction ($H^1 \neq 0$) or Bloom repetition repulsion was triggered.
    #[inline(always)]
    #[must_use]
    pub const fn is_cohomology_repulsion(&self) -> bool {
        (self.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0
    }
}

/// Root causal signal supertrait refining [`Liminal`] into a finite 4-axis causal window
/// for all signals entering the GCSO Manifold Gateway.
pub trait CausalProvenanceSignal: Liminal + BranchlessKernel {
    /// Static causal provenance cognitive aperture across system and grounding origins (`0x0000_000F`).
    const CAUSAL_APERTURE_MASK: u32 = APERTURE_CAUSAL_PROVENANCE;

    /// Type-level complement identifying axes outside this causal window as unknowable (`!APERTURE`).
    const CAUSAL_UNKNOWABLE_VOID_MASK: u32 = !Self::CAUSAL_APERTURE_MASK;

    /// Returns the bitwise provenance flags (`CAUSAL_AXIS_* | CAUSAL_CAUSE_*`) contributed by `self`.
    fn causal_provenance_bits(&self) -> u32;

    /// Returns the active unknowable complement bitmask derived strictly as `!self.causal_provenance_bits()`.
    #[inline(always)]
    fn active_unknowable_void_bits(&self) -> u32 {
        !self.causal_provenance_bits()
    }

    /// Returns the normalized anomaly or constraint severity in $[0.0, 1.0]$.
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        0.0
    }

    /// Branchlessly resolves this signal's live dynamic cognitive aperture
    /// `(active_granule_mask, effective_gain_q7, phase_shift_q7)` across all three sheets of
    /// [`AutonomousCoreLayerStack`] (`Layer 0: Liminal`, `Layer 1: Stigmergic`, `Layer 2: Ego`).
    #[inline]
    fn resolve_dynamic_core_aperture(&self, stack: &AutonomousCoreLayerStack) -> (u8, u8, i8) {
        let bits = self.causal_provenance_bits();
        let ego_active = stack.ego_layer.is_active();
        let ego_trans_pos = ego_active & (stack.ego_layer.transgression_bias_q7 > 0);
        let mut active_granule_mask = 0u8;
        let mut weighted_gain_sum = 0u32;
        let mut weighted_shift_sum = 0i32;
        let mut active_axis_count = 0u32;

        let mut ch = 0usize;
        while ch < 4 {
            let axis_on = ((bits >> ch) & 1) != 0;
            let base_floor = i32::from(stack.liminal_layer.void_floor_q7[ch]);
            let base_gain = i32::from(stack.liminal_layer.aperture_gain_q7[ch]);
            let ego_floor_bias = branchless_select_i32(
                ego_active,
                i32::from(stack.ego_layer.trait_floor_bias_q7[ch]),
                0,
            );
            let ego_gain_bias = branchless_select_i32(
                ego_active,
                i32::from(stack.ego_layer.trait_gain_bias_q7[ch]),
                0,
            );
            let eff_floor = (base_floor + ego_floor_bias).clamp(4, 127) as u8;
            let eff_gain = (base_gain + ego_gain_bias).clamp(16, 127) as u8;
            let ap_open = ((stack.liminal_layer.active_aperture_mask >> ch) & 1) != 0;

            let mut ch_mask = 0u8;
            let mut g = 0usize;
            while g < NUM_STIGMERGIC_GRANULES {
                let bit = 1u8 << g;
                let pher = stack.stigmergic_layer.granule_pheromone_q7[g];
                let in_venn_core = (stack.stigmergic_layer.venn_overlap_mask & bit) != 0;
                let in_serendip = (stack.stigmergic_layer.serendipity_excursion_mask & bit) != 0;
                let ego_auto = stack.ego_layer.granule_autonomy_q7[g];

                let venn_bonus = branchless_select_u8(in_venn_core, 24, 0);
                let ch_accepts_ego_transgress =
                    (ch == 2) | (stack.ego_layer.trait_floor_bias_q7[ch] < 0);
                let ego_bonus = branchless_select_u8(
                    ego_trans_pos & ch_accepts_ego_transgress & (ego_auto >= 24),
                    ego_auto >> 1,
                    0,
                );
                let eff_density = pher.saturating_add(venn_bonus).saturating_add(ego_bonus);

                let req_floor = branchless_select_u8(
                    ch == 2,
                    (eff_floor >> 1).saturating_sub(branchless_select_u8(in_serendip, 8, 0)),
                    eff_floor >> 1,
                );
                let g_open = ap_open & (eff_density >= req_floor);
                ch_mask |= branchless_select_u8(g_open, bit, 0);
                g += 1;
            }

            active_granule_mask |= branchless_select_u8(axis_on, ch_mask, 0);
            weighted_gain_sum += branchless_select_u32(axis_on, u32::from(eff_gain), 0);
            weighted_shift_sum +=
                branchless_select_i32(axis_on, i32::from(stack.liminal_layer.phase_shift_q7[ch]), 0);
            active_axis_count += u32::from(axis_on);
            ch += 1;
        }

        let unattributed = active_axis_count == 0;
        let has_layer1_mask =
            (stack.stigmergic_layer.venn_overlap_mask | stack.stigmergic_layer.serendipity_excursion_mask)
                != 0;
        let fallback_mask = stack.stigmergic_layer.venn_overlap_mask
            | stack.stigmergic_layer.serendipity_excursion_mask
            | branchless_select_u8(ego_trans_pos | !has_layer1_mask, 0xFF, 0);
        let out_mask = branchless_select_u8(unattributed, fallback_mask, active_granule_mask);
        let out_gain = match active_axis_count {
            0 => stack.liminal_layer.aperture_gain_q7[7],
            1 => weighted_gain_sum.min(127) as u8,
            2 => (weighted_gain_sum >> 1).min(127) as u8,
            3 => ((weighted_gain_sum * 43) >> 7).min(127) as u8,
            _ => (weighted_gain_sum >> 2).min(127) as u8,
        };
        let out_shift = match active_axis_count {
            0 => stack.liminal_layer.phase_shift_q7[7],
            1 => weighted_shift_sum.clamp(-127, 127) as i8,
            2 => (weighted_shift_sum >> 1).clamp(-127, 127) as i8,
            3 => ((weighted_shift_sum * 43) >> 7).clamp(-127, 127) as i8,
            _ => (weighted_shift_sum >> 2).clamp(-127, 127) as i8,
        };
        (out_mask, out_gain.max(16), out_shift)
    }
}

impl<T: CausalProvenanceSignal + ?Sized> CausalProvenanceSignal for &T {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        (**self).causal_provenance_bits()
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        (**self).causal_anomaly_severity()
    }
    #[inline(always)]
    fn resolve_dynamic_core_aperture(&self, stack: &AutonomousCoreLayerStack) -> (u8, u8, i8) {
        (**self).resolve_dynamic_core_aperture(stack)
    }
}

// ===================================================================
// 4. Origin 1: System & Physical Constraint Blanket (`SystemConstraintSignal`)
// ===================================================================

/// **Origin 1 Supertrait (`SystemConstraintSignal: CausalProvenanceSignal`)**:
/// Hardware resource, game-coexistence QoS, and EDBC entropy constraints governing compute budgeting.
pub trait SystemConstraintSignal: CausalProvenanceSignal {
    /// Static cognitive aperture bit isolating Origin 1 system & QoS telemetry (`1 << 0`).
    const SYSTEM_APERTURE_MASK: u32 = APERTURE_SYSTEM_CONSTRAINT;

    /// Complement bitmask outside Origin 1 (`!APERTURE_SYSTEM_CONSTRAINT`).
    const SYSTEM_UNKNOWABLE_VOID_MASK: u32 = !Self::SYSTEM_APERTURE_MASK;

    /// Normalized VRAM / system memory pressure in $[0.0, 1.0]$.
    fn system_vram_pressure(&self) -> f32;

    /// Normalized GPU/CPU frame-time overrun ratio in $[0.0, 1.0]$ for AAA game coexistence.
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        0.0
    }

    /// 8-bit bitmask of active 8-head SWAR Micro-Swarm cells (`0xFF` = all 64 heads active).
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        let pressure = branchless_max_f32(
            branchless_clamp_f32(self.system_vram_pressure(), 0.0, 1.0),
            branchless_clamp_f32(self.system_gpu_frame_overrun(), 0.0, 1.0),
        );
        branchless_select_u8(
            pressure > 0.85,
            0x0F,
            branchless_select_u8(pressure > 0.65, 0x3F, 0xFF),
        )
    }

    /// Returns `true` when inertial layer skipping (`ACTION_MODE_LAYER_SKIP`) is recommended.
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        let overrun = branchless_clamp_f32(self.system_gpu_frame_overrun(), 0.0, 1.0);
        let vram = branchless_clamp_f32(self.system_vram_pressure(), 0.0, 1.0);
        (overrun > 0.80) | (vram > 0.90)
    }

    /// Recommended QDPS minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 {
        let pressure = branchless_max_f32(
            branchless_clamp_f32(self.system_vram_pressure(), 0.0, 1.0),
            branchless_clamp_f32(self.system_gpu_frame_overrun(), 0.0, 1.0),
        );
        DEFAULT_QDPS_MIN_STEP_RAD * (1.0 + 2.0 * pressure)
    }
}

impl<T: SystemConstraintSignal + ?Sized> SystemConstraintSignal for &T {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        (**self).system_vram_pressure()
    }
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        (**self).system_gpu_frame_overrun()
    }
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        (**self).active_swarm_cells_mask()
    }
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        (**self).recommend_layer_skip()
    }
    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 {
        (**self).qdps_min_step_rad_override()
    }
}

/// Zero-cost wrapper classifying any system/resource state `T` into **Origin 1 (`SystemConstraintSignal`)**.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SystemConstraint<T = ()> {
    /// Wrapped inner state or telemetry payload.
    pub inner: T,
    /// Normalized VRAM / memory pressure in $[0.0, 1.0]$.
    pub vram_pressure: f32,
    /// Normalized GPU/CPU frame overrun in $[0.0, 1.0]$.
    pub gpu_frame_overrun: f32,
    /// Explicit QDPS minimum step cutoff in radians.
    pub qdps_min_step_rad: f32,
    /// Active 8-cell Micro-Swarm bitmask (`0x00..=0xFF`).
    pub active_swarm_mask: u8,
    /// Explicit inertial layer-skip recommendation flag.
    pub force_layer_skip: bool,
}

impl<T> SystemConstraint<T> {
    /// Constructs a new [`SystemConstraint`] wrapper with explicit QoS parameters.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        inner: T,
        vram_pressure: f32,
        gpu_frame_overrun: f32,
        active_swarm_mask: u8,
    ) -> Self {
        let p_vram = branchless_clamp_f32(branchless_relu_f32(vram_pressure), 0.0, 1.0);
        let p_gpu = branchless_clamp_f32(branchless_relu_f32(gpu_frame_overrun), 0.0, 1.0);
        let max_p = branchless_max_f32(p_vram, p_gpu);
        Self {
            inner,
            vram_pressure: p_vram,
            gpu_frame_overrun: p_gpu,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD * (1.0 + 2.0 * max_p),
            active_swarm_mask: branchless_select_u8(
                active_swarm_mask != 0,
                active_swarm_mask,
                0xFF,
            ),
            force_layer_skip: (p_gpu > 0.80) | (p_vram > 0.90),
        }
    }

    /// Builder method overriding `force_layer_skip`.
    #[inline(always)]
    #[must_use]
    pub const fn with_layer_skip(mut self, skip: bool) -> Self {
        self.force_layer_skip = skip;
        self
    }

    /// Builder method overriding `qdps_min_step_rad`.
    #[inline(always)]
    #[must_use]
    pub const fn with_qdps_min_step_rad(mut self, step_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(step_rad) & (step_rad >= 0.0);
        self.qdps_min_step_rad = branchless_select_f32(valid, step_rad, self.qdps_min_step_rad);
        self
    }
}

impl<T: BranchlessKernel> CausalProvenanceSignal for SystemConstraint<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let throttle = self.force_layer_skip | (self.active_swarm_mask != 0xFF)
            | (self.vram_pressure > 0.70) | (self.gpu_frame_overrun > 0.70);
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | branchless_select_u32(throttle, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(self.vram_pressure, self.gpu_frame_overrun)
    }
}

impl<T: BranchlessKernel> SystemConstraintSignal for SystemConstraint<T> {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        self.vram_pressure
    }
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        self.gpu_frame_overrun
    }
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        self.active_swarm_mask
    }
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        self.force_layer_skip
    }
    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 {
        self.qdps_min_step_rad
    }
}

impl CausalProvenanceSignal for gcso_edbc_state_t {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_rep = self.active_branch_mode == 2;
        let is_low_e = (self.active_branch_mode == 0) & (self.moving_z_entropy < 0.12);
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | branchless_select_u32(is_rep, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
            | branchless_select_u32(is_low_e, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        let h_sev =
            branchless_clamp_f32(branchless_relu_f32(self.moving_z_entropy) * 0.25, 0.0, 1.0);
        branchless_select_f32(self.active_branch_mode == 2, 0.85, h_sev)
    }
}

impl SystemConstraintSignal for gcso_edbc_state_t {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        branchless_clamp_f32(branchless_relu_f32(self.moving_z_entropy) * 0.25, 0.0, 1.0)
    }
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        branchless_clamp_f32(branchless_relu_f32(self.sliding_entropy_rate), 0.0, 1.0)
    }
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        let low_e = (self.active_branch_mode == 0) & (self.moving_z_entropy < 0.10)
            & (self.sliding_entropy_rate <= 0.5);
        branchless_select_u8(low_e, 0x0F, 0xFF)
    }
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        (self.active_branch_mode == 0)
            & (self.moving_z_entropy <= 0.12)
            & (self.sliding_entropy_rate <= 0.5)
    }
}

impl CausalProvenanceSignal for gcso_config_t {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
    }
}

impl SystemConstraintSignal for gcso_config_t {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        0.0
    }
    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 {
        let valid =
            branchless_is_finite_f32(self.qdps_min_step_rad) & (self.qdps_min_step_rad >= 0.0);
        branchless_select_f32(valid, self.qdps_min_step_rad, DEFAULT_QDPS_MIN_STEP_RAD)
    }
}

/// Cold-Path and startup/periodic environment-probe snapshot characterizing the active LLM's
/// RoPE subspace geometry, neutral entropy baseline, and inter-layer phase coherence.
///
/// Populated before startup and refreshed on periodic Cold-Path probes or model swaps so
/// [`StigmergicThresholdPolicy`] implementations can self-calibrate without hardcoded per-model constants.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManifoldProbeBaseline {
    /// Active attention heads observed in the target model (`1..=MAX_ATTENTION_HEADS`).
    pub active_heads: u16,
    /// Number of transformer layers selected by `layer_rope_mask` (`>= 1`).
    pub steered_layers: u16,
    /// Baseline normalized EDBC entropy observed on neutral calibration prompts ($\tilde{H}_0 \in [0, 1]$).
    pub baseline_entropy: f32,
    /// Mean positive inter-layer / intra-turn phase trajectory coherence measured during calibration ($\bar{c} \in [0, 1]$).
    pub baseline_inter_layer_coherence: f32,
    /// Neutral-token baseline cosine similarity against contrastive guard anchors ($\in [0, 1]$).
    pub baseline_guard_noise_floor: f32,
    /// Monotonically incremented Cold-Path calibration / re-allocation epoch counter.
    pub probe_epoch: u32,
}

impl Default for ManifoldProbeBaseline {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl ManifoldProbeBaseline {
    /// Creates a neutral default [`ManifoldProbeBaseline`] suitable before runtime probing runs.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active_heads: MAX_ATTENTION_HEADS as u16,
            steered_layers: 16,
            baseline_entropy: 0.35,
            baseline_inter_layer_coherence: 0.75,
            baseline_guard_noise_floor: 0.45,
            probe_epoch: 0,
        }
    }
}

/// Trait abstracting dynamic derivation of [`StigmergicThresholds`] from hardware/telemetry
/// constraints (`SystemConstraintSignal`), Cold-Path model probe measurements (`ManifoldProbeBaseline`),
/// active pointer-trail state, and instantaneous entropy (`H_tilde`).
pub trait StigmergicThresholdPolicy {
    /// Resolves effective [`StigmergicThresholds`] for the current inference step without heap allocation.
    fn resolve_thresholds<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &gcso_pointer_trail_t,
        entropy_h_tilde: f32,
    ) -> StigmergicThresholds;

    /// Resolves [`StigmergicThresholds`] and specializes `ActionHub` decision gates using an
    /// optional [`CognitiveTrajectoryLedger`] habit state at `scale`.
    ///
    /// Blanket default delegates to [`Self::resolve_thresholds`] followed by
    /// [`CognitiveTrajectoryLedger::bias_stigmergic_thresholds`] so identical inputs trigger
    /// distinct regime ladders according to active cognitive circuit habits.
    #[inline(always)]
    fn resolve_thresholds_with_circuit<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &gcso_pointer_trail_t,
        entropy_h_tilde: f32,
        circuit: Option<&CognitiveTrajectoryLedger>,
        scale: StigmergicScaleGranularity,
    ) -> StigmergicThresholds {
        let base = self.resolve_thresholds(system, probe, trail, entropy_h_tilde);
        match circuit {
            Some(ledger) => ledger.bias_stigmergic_thresholds(base, scale),
            None => base,
        }
    }
}

impl<T: StigmergicThresholdPolicy + ?Sized> StigmergicThresholdPolicy for &T {
    #[inline(always)]
    fn resolve_thresholds<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &gcso_pointer_trail_t,
        entropy_h_tilde: f32,
    ) -> StigmergicThresholds {
        (**self).resolve_thresholds(system, probe, trail, entropy_h_tilde)
    }

    #[inline(always)]
    fn resolve_thresholds_with_circuit<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &gcso_pointer_trail_t,
        entropy_h_tilde: f32,
        circuit: Option<&CognitiveTrajectoryLedger>,
        scale: StigmergicScaleGranularity,
    ) -> StigmergicThresholds {
        (**self).resolve_thresholds_with_circuit(
            system,
            probe,
            trail,
            entropy_h_tilde,
            circuit,
            scale,
        )
    }
}

impl StigmergicThresholdPolicy for StigmergicThresholds {
    #[inline(always)]
    fn resolve_thresholds<C: SystemConstraintSignal + ?Sized>(
        &self,
        _system: &C,
        _probe: &ManifoldProbeBaseline,
        _trail: &gcso_pointer_trail_t,
        _entropy_h_tilde: f32,
    ) -> StigmergicThresholds {
        *self
    }
}

/// Closed-form, branchless adaptive threshold policy calibrating [`StigmergicThresholds`]
/// from live [`SystemConstraintSignal`] pressure, [`ManifoldProbeBaseline`] model geometry,
/// and pointer-trail stability.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveThresholdPolicy {
    /// Nominal baseline thresholds before runtime/probe calibration.
    pub base: StigmergicThresholds,
    /// Minimum/maximum clamp multiplier applied to dynamic gain & ladder adaptation.
    pub min_scale: f32,
    /// Maximum clamp multiplier applied to dynamic gain adaptation.
    pub max_scale: f32,
}

impl Default for AdaptiveThresholdPolicy {
    #[inline(always)]
    fn default() -> Self {
        Self::new(StigmergicThresholds::default())
    }
}

impl AdaptiveThresholdPolicy {
    /// Creates an [`AdaptiveThresholdPolicy`] anchored around `base`.
    #[inline(always)]
    #[must_use]
    pub const fn new(base: StigmergicThresholds) -> Self {
        Self {
            base,
            min_scale: 0.65,
            max_scale: 1.85,
        }
    }
}

/// Calibratable Q0.7 integer gating thresholds for [`CognitiveTrajectoryLedger`] cascades
/// and closed-loop self-reflection, avoiding hardcoded dataset-specific constants.
pub trait CognitiveCircuitThresholdPolicy {
    /// Minimum `fuse_gate_q7` required for adaptive early-exit (`0..=128`).
    fn early_exit_fuse_floor_q7(&self) -> u8 {
        96
    }

    /// Maximum `repulse_sensitivity_q7` allowed during early-exit (`0..=128`).
    fn early_exit_repulse_ceiling_q7(&self) -> u8 {
        72
    }

    /// Maximum mean residual phase deviation (`0..=128`) tolerated for early-exit.
    fn early_exit_residual_max_q7(&self) -> u8 {
        16
    }

    /// Minimum firing intensity (`0..=128`) triggering bounded 1-step recurrent refinement.
    fn recurrent_refine_intensity_floor_q7(&self) -> u8 {
        80
    }

    /// Minimum `peel_gate_q7` (`0..=128`) for escalating omitted compound decomposition.
    fn peel_escalation_floor_q7(&self) -> u8 {
        84
    }

    /// Derives the scale-invariant $S^{H-1}$ phase-resonance acceptance floor (`[24, 96]` in Q7)
    /// from the $2 / \sqrt{H}$ concentration-of-measure noise floor and cache occupancy pressure.
    #[inline]
    fn kv_resonance_floor_for_heads_q7(
        &self,
        active_heads: u16,
        occupied_slots: usize,
        capacity: usize,
    ) -> gcso_q7_t {
        let h = f32::from(active_heads.clamp(1, MAX_ATTENTION_HEADS as u16));
        let half = 0.5_f32 * h;
        let y0 = f32::from_bits(0x5f37_5a86_u32.wrapping_sub(h.to_bits() >> 1));
        let y1 = y0 * (1.5_f32 - half * y0 * y0);
        let inv_sqrt_h = y1 * (1.5_f32 - half * y1 * y1);
        let base_floor = 2.0_f32 * inv_sqrt_h * 127.0_f32;
        let cap = capacity.max(1) as f32;
        let occ_ratio = branchless_clamp_f32((occupied_slots as f32) * fast_inv_pos_normal_f32(cap), 0.0, 1.0);
        let pressure_relax = 1.0_f32 - 0.15_f32 * occ_ratio;
        let raw = (base_floor * pressure_relax + 0.5_f32) as i32;
        raw.clamp(24, 96) as gcso_q7_t
    }

    /// Returns the circular coherence threshold below which P-Frame seek steps trigger repulsion.
    #[inline(always)]
    fn kv_contradiction_threshold(&self) -> f32 {
        0.35
    }

    /// Derives the circuit composition blend weight (`16..=96` in Q7) from `frame`'s
    /// Fact-Persona coherence so aligned frames reinforce faster without hardcoded mixing ratios.
    #[inline(always)]
    fn kv_circuit_blend_q7(&self, frame: &GatewayFrame) -> u8 {
        let coh = if branchless_is_finite_f32(frame.fact_persona_coherence) {
            branchless_clamp_f32(frame.fact_persona_coherence, -1.0, 1.0)
        } else {
            0.0
        };
        let raw = ((0.5_f32 + 0.5_f32 * coh) * 96.0_f32 + 0.5_f32) as i32;
        raw.clamp(16, 96) as u8
    }
}

impl CognitiveCircuitThresholdPolicy for AdaptiveThresholdPolicy {
    #[inline(always)]
    fn early_exit_fuse_floor_q7(&self) -> u8 {
        let def_gain = crate::action_hub::DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN;
        let coh_shift = ((self.base.reinforcement_gain.clamp(0.10, 1.50) - def_gain) * 32.0) as i32;
        (96 + coh_shift).clamp(72, 120) as u8
    }

    #[inline(always)]
    fn early_exit_residual_max_q7(&self) -> u8 {
        let scaled = ((16.0 * 1.85) / self.max_scale.clamp(0.8, 3.0)) as i32;
        scaled.clamp(8, 32) as u8
    }

    #[inline(always)]
    fn peel_escalation_floor_q7(&self) -> u8 {
        let def_evap = crate::action_hub::DEFAULT_STIGMERGIC_EVAPORATION_RETAIN;
        let margin_adj = ((self.base.evaporation_retain.clamp(0.50, 0.99) - def_evap) * 80.0) as i32;
        (84 + margin_adj).clamp(64, 112) as u8
    }

    #[inline]
    fn kv_resonance_floor_for_heads_q7(
        &self,
        active_heads: u16,
        occupied_slots: usize,
        capacity: usize,
    ) -> gcso_q7_t {
        let h = f32::from(active_heads.clamp(1, MAX_ATTENTION_HEADS as u16));
        let half = 0.5_f32 * h;
        let y0 = f32::from_bits(0x5f37_5a86_u32.wrapping_sub(h.to_bits() >> 1));
        let y1 = y0 * (1.5_f32 - half * y0 * y0);
        let inv_sqrt_h = y1 * (1.5_f32 - half * y1 * y1);
        let def_gain = crate::action_hub::DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN;
        let gain_shift = ((self.base.reinforcement_gain.clamp(0.10, 1.50) - def_gain) * 16.0) as i32;
        let cap = capacity.max(1) as f32;
        let occ_ratio = branchless_clamp_f32((occupied_slots as f32) * fast_inv_pos_normal_f32(cap), 0.0, 1.0);
        let base_floor = 2.0_f32 * inv_sqrt_h * 127.0_f32 * (1.0_f32 - 0.15_f32 * occ_ratio);
        ((base_floor + 0.5_f32) as i32 + gain_shift).clamp(24, 96) as gcso_q7_t
    }

    #[inline(always)]
    fn kv_contradiction_threshold(&self) -> f32 {
        branchless_clamp_f32(self.base.contradiction_coherence_thresh, 0.15, 0.75)
    }
}

impl StigmergicThresholdPolicy for AdaptiveThresholdPolicy {
    #[inline]
    fn resolve_thresholds<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &gcso_pointer_trail_t,
        entropy_h_tilde: f32,
    ) -> StigmergicThresholds {
        let p_vram = branchless_clamp_f32(branchless_relu_f32(system.system_vram_pressure()), 0.0, 1.0);
        let p_overrun =
            branchless_clamp_f32(branchless_relu_f32(system.system_gpu_frame_overrun()), 0.0, 1.0);
        let pressure = branchless_max_f32(p_vram, p_overrun);

        let base_coh = branchless_clamp_f32(
            branchless_select_f32(
                branchless_is_finite_f32(probe.baseline_inter_layer_coherence),
                probe.baseline_inter_layer_coherence,
                0.75,
            ),
            0.0,
            1.0,
        );
        let retain_0 =
            branchless_clamp_f32(branchless_relu_f32(self.base.evaporation_retain), 0.0, 0.995);
        let is_subspace = probe.active_heads > 0 && probe.active_heads < MAX_ATTENTION_HEADS as u16;
        let eff_retain = if is_subspace {
            retain_0 * (0.85_f32 + 0.15_f32 * base_coh)
        } else {
            retain_0
        };
        let eff_traj = if is_subspace {
            0.38_f32 + 0.52_f32 * base_coh
        } else {
            0.50_f32 + 0.50_f32 * base_coh
        };
        // Equilibrate deposit gain against model-specific inter-layer decay & trajectory damping.
        // Reference equilibrium factor at default baseline (`c=0.75`, `retain=0.95`, `subspace`): 0.1444.
        let loss_over_traj2 = (1.0_f32 - eff_retain) * fast_inv_pos_normal_f32(eff_traj * eff_traj);
        let raw_gain_scale = loss_over_traj2 * (1.0_f32 / 0.1444_f32);
        let gain_scale = branchless_clamp_f32(raw_gain_scale, self.min_scale, self.max_scale);
        let pressure_damp = 1.0_f32 - 0.35_f32 * pressure;

        let dyn_gain =
            branchless_clamp_f32(self.base.reinforcement_gain * gain_scale * pressure_damp, 0.05, 1.25);

        // Sustained high-pull trails gradually ease crystallization, while resource pressure raises skip/Rank-1 residual gates.
        let sustained_pull = if trail.step_count >= 16 {
            branchless_clamp_f32((trail.attractor_pull_force - 0.65_f32) * 2.0_f32, 0.0, 1.0)
        } else {
            0.0_f32
        };
        let skip_thresh = branchless_clamp_f32(
            self.base.inertial_skip_entropy_thresh * (1.0_f32 + 0.75_f32 * pressure),
            0.05,
            0.45,
        );
        let express_thresh = self.base.express_steer_density_thresh * (1.0_f32 + 0.35_f32 * pressure);
        let pspm_thresh = self.base.subhead_route_density_thresh * (1.0_f32 + 0.30_f32 * pressure);
        let srl_thresh = self.base.rank1_inject_density_thresh * (1.0_f32 + 0.50_f32 * pressure);
        let crystal_thresh =
            self.base.crystallize_density_thresh * (1.0_f32 - 0.15_f32 * sustained_pull + 0.25_f32 * pressure);

        // Gently relax contradiction sensitivity when probe observes noisy guard floor or high entropy.
        let h_valid = branchless_select_f32(
            branchless_is_finite_f32(entropy_h_tilde),
            branchless_clamp_f32(entropy_h_tilde, 0.0, 1.0),
            0.35,
        );
        let noise_adj = branchless_clamp_f32(
            branchless_relu_f32(probe.baseline_guard_noise_floor - 0.50_f32) * 0.40_f32,
            0.0,
            0.25,
        );
        let contradict_thresh = branchless_clamp_f32(
            self.base.contradiction_coherence_thresh + noise_adj + 0.05_f32 * h_valid,
            0.15,
            0.75,
        );

        StigmergicThresholds {
            evaporation_retain: retain_0,
            reinforcement_gain: dyn_gain,
            inertial_skip_entropy_thresh: skip_thresh,
            express_steer_density_thresh: express_thresh,
            subhead_route_density_thresh: pspm_thresh,
            rank1_inject_density_thresh: srl_thresh,
            crystallize_density_thresh: crystal_thresh,
            contradiction_coherence_thresh: contradict_thresh,
        }
    }
}

// ===================================================================
// 5. Origin 2: Cognitive Grounding Hierarchy (`Fact` x `Persona` x `ContradictionGuard`)
// ===================================================================

/// **Origin 2 Parent Supertrait (`CognitiveGroundingSignal: CausalProvenanceSignal`)**:
/// Common phase-projection and `.gcso` (mapped-storage) 48-bit section-offset addressing foundation.
pub trait CognitiveGroundingSignal: CausalProvenanceSignal {
    /// Static cognitive aperture spanning Origin 2A (Fact), 2B (Persona), and 2C (Guard) (`0x0E`).
    const GROUNDING_APERTURE_MASK: u32 =
        APERTURE_OBJECTIVE_FACT | APERTURE_SUBJECTIVE_PERSONA | APERTURE_CONTRADICTION_GUARD;

    /// Complement bitmask outside Origin 2 semantic grounding (`!GROUNDING_APERTURE_MASK`).
    const GROUNDING_UNKNOWABLE_VOID_MASK: u32 = !Self::GROUNDING_APERTURE_MASK;

    /// Projects this grounding signal onto the 64-head $SO(2)^{64}$ Q7 phase manifold.
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS];

    /// Fast zero-copy borrow of the 64-head Q7 phase profile when already materialized in memory.
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        None
    }

    /// Projects this grounding signal directly into 8 register-resident `u64` SWAR words (`[u64; 8]`).
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        match self.grounding_phases_ref_64() {
            Some(arr_ref) => load_q7_words_64(arr_ref),
            None => load_q7_words_64(&self.project_grounding_phases_64()),
        }
    }

    /// Base gravitational pull energy $E_{\mathrm{base}} \ge 0$ of this grounding signal.
    fn grounding_base_energy(&self) -> f32;

    /// Canonical 48-bit `.gcso` (mapped-storage) section or RAG record offset (`bits 0..=47` of `gcso_tagged_ptr_t`).
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        0
    }

    /// Full 32-bit macro attractor anchor identifier on the Single Unified Attractor Manifold.
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        0
    }

    /// Active attention-head subspace dimension $H \in [1, 64]$ over which phase coherence is evaluated.
    #[inline(always)]
    fn active_head_count(&self) -> u16 {
        MAX_ATTENTION_HEADS as u16
    }

    /// Single-pass grounding envelope `(phase_words, base_energy, zimms_offset48, anchor_id, active_heads)`
    /// avoiding redundant slice folds on composite or resonant signals.
    #[inline(always)]
    fn synthesize_grounding_envelope(&self) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u64, u32, u16) {
        (
            self.project_grounding_words_64(),
            self.grounding_base_energy(),
            self.mapped_section_offset48(),
            self.grounding_anchor_id(),
            self.active_head_count(),
        )
    }
}

impl<T: CognitiveGroundingSignal + ?Sized> CognitiveGroundingSignal for &T {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        (**self).project_grounding_phases_64()
    }
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        (**self).grounding_phases_ref_64()
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        (**self).project_grounding_words_64()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        (**self).grounding_base_energy()
    }
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        (**self).mapped_section_offset48()
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        (**self).grounding_anchor_id()
    }
    #[inline(always)]
    fn active_head_count(&self) -> u16 {
        (**self).active_head_count()
    }
    #[inline(always)]
    fn synthesize_grounding_envelope(&self) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u64, u32, u16) {
        (**self).synthesize_grounding_envelope()
    }
}

/// **Origin 2-A Subtrait (`ObjectiveFactSignal: CognitiveGroundingSignal`)**:
/// External RAG chunks, database records, and objective conversation logs (`Slot 2`, narrow RIPA clamp `0.14 rad`).
pub trait ObjectiveFactSignal: CognitiveGroundingSignal {
    /// Narrow single-axis cognitive aperture for factual grounding (`1 << 1`).
    const FACT_APERTURE_MASK: u32 = APERTURE_OBJECTIVE_FACT;

    /// Complement bitmask treating non-factual axes as unknowable (`!APERTURE_OBJECTIVE_FACT`).
    const FACT_UNKNOWABLE_VOID_MASK: u32 = !Self::FACT_APERTURE_MASK;

    /// RIPA phase clamp limit in radians for objective factual grounding (defaults to `0.14 rad`).
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 {
        DEFAULT_FACT_RIPA_CLAMP_RAD
    }

    /// sub-head router `Fact` sub-head phase gain multiplier (defaults to `1.15`).
    #[inline(always)]
    fn fact_pspm_gain(&self) -> f32 {
        1.15
    }
}

impl<T: ObjectiveFactSignal + ?Sized> ObjectiveFactSignal for &T {
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 {
        (**self).fact_ripa_clamp_rad()
    }
    #[inline(always)]
    fn fact_pspm_gain(&self) -> f32 {
        (**self).fact_pspm_gain()
    }
}

/// **Origin 2-B Subtrait (`SubjectivePersonaSignal: CognitiveGroundingSignal`)**:
/// System prompts, character beliefs, stylistic tone, and subjective episodic memory (`Slot 1`, wide RIPA clamp `0.24 rad`).
pub trait SubjectivePersonaSignal: CognitiveGroundingSignal {
    /// Narrow single-axis cognitive aperture for subjective persona tone (`1 << 2`).
    const PERSONA_APERTURE_MASK: u32 = APERTURE_SUBJECTIVE_PERSONA;

    /// Complement bitmask treating non-persona axes as unknowable (`!APERTURE_SUBJECTIVE_PERSONA`).
    const PERSONA_UNKNOWABLE_VOID_MASK: u32 = !Self::PERSONA_APERTURE_MASK;

    /// RIPA phase clamp limit in radians for subjective persona modulation (defaults to `0.24 rad`).
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 {
        DEFAULT_PERSONA_RIPA_CLAMP_RAD
    }

    /// sub-head router `Explore` / `Logic` sub-head phase gain multiplier (defaults to `1.20`).
    #[inline(always)]
    fn persona_pspm_gain(&self) -> f32 {
        1.20
    }
}

impl<T: SubjectivePersonaSignal + ?Sized> SubjectivePersonaSignal for &T {
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 {
        (**self).persona_ripa_clamp_rad()
    }
    #[inline(always)]
    fn persona_pspm_gain(&self) -> f32 {
        (**self).persona_pspm_gain()
    }
}

/// **Origin 2-C Subtrait (`ContradictionGuardSignal: CognitiveGroundingSignal`)**:
/// Sheaf cohomology obstruction ($H^1(K; \mathcal{F}) \neq 0$) and [`ContrastivePhaseBloomFilter`]
/// repetition / self-contradiction guard triggering anti-phase $-\boldsymbol{\Delta\theta}$ escape.
pub trait ContradictionGuardSignal: CognitiveGroundingSignal {
    /// Single-axis cognitive aperture for contradiction guard repulsion (`1 << 3`).
    const GUARD_APERTURE_MASK: u32 = APERTURE_CONTRADICTION_GUARD;

    /// Complement bitmask treating non-guard axes as unknowable (`!APERTURE_CONTRADICTION_GUARD`).
    const GUARD_UNKNOWABLE_VOID_MASK: u32 = !Self::GUARD_APERTURE_MASK;

    /// Evaluates repulsive severity in $[0.0, 1.0]$ against `candidate_phases`.
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32;

    /// Evaluates repulsive severity directly from 8 register-resident `u64` SWAR words (`candidate_words`).
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let _ = candidate_words;
        self.evaluate_repulsion_severity(candidate_phases)
    }

    /// Returns `true` if [`Self::evaluate_repulsion_severity_from_words`] inspects `candidate_phases`
    /// rather than operating purely on `candidate_words` in registers.
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        true
    }
}

impl<T: ContradictionGuardSignal + ?Sized> ContradictionGuardSignal for &T {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        (**self).evaluate_repulsion_severity(candidate_phases)
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        (**self).evaluate_repulsion_severity_from_words(candidate_words, candidate_phases)
    }

    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        (**self).requires_phase_slice_for_repulsion()
    }
}

/// Concrete wrapper classifying any [`RawPhaseProjectable`] payload `T` as an **Objective Fact** (`Origin 2-A`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectiveFact<T> {
    /// Underlying raw payload (`&str`, `&[u8]`, `&[f32]`, or `[gcso_q7_t; 64]`).
    pub payload: T,
    /// Base gravitational pull energy $E_{\mathrm{fact}} \ge 0$.
    pub base_energy: f32,
    /// Direct 48-bit `.gcso` (mapped-storage) section offset (`bits 0..=47`).
    pub zimms_offset48: u64,
    /// Full 32-bit macro anchor ID on the unified manifold.
    pub anchor_id: u32,
    /// RIPA phase clamp limit in radians (defaults to [`DEFAULT_FACT_RIPA_CLAMP_RAD`]).
    pub ripa_clamp_rad: f32,
}

impl<T> ObjectiveFact<T> {
    /// Constructs a new [`ObjectiveFact`] with explicit base energy, 48-bit mapped-storage offset, and 32-bit anchor ID.
    #[inline(always)]
    #[must_use]
    pub const fn new(payload: T, base_energy: f32, zimms_offset48: u64, anchor_id: u32) -> Self {
        Self {
            payload, base_energy: branchless_flush_subnormal_f32(branchless_relu_f32(base_energy)),
            zimms_offset48: zimms_offset48 & gcso_tagged_ptr_t::ADDR_MASK, anchor_id,
            ripa_clamp_rad: DEFAULT_FACT_RIPA_CLAMP_RAD,
        }
    }

    /// Builder method overriding the RIPA clamp limit in radians.
    #[inline(always)]
    #[must_use]
    pub const fn with_ripa_clamp_rad(mut self, max_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(max_rad) & (max_rad > 0.0);
        self.ripa_clamp_rad = branchless_select_f32(valid, max_rad, self.ripa_clamp_rad);
        self
    }
}

impl<T: RawPhaseProjectable> CausalProvenanceSignal for ObjectiveFact<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let weak = self.base_energy < 0.25;
        CAUSAL_AXIS_OBJECTIVE_FACT
            | branchless_select_u32(weak, CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_select_f32(self.base_energy < 0.25, 0.35, 0.0)
    }
}

impl<T: RawPhaseProjectable> CognitiveGroundingSignal for ObjectiveFact<T> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.payload.project_to_q7_64()
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.payload.project_to_words_64()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.base_energy
    }
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        self.zimms_offset48
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl<T: RawPhaseProjectable> ObjectiveFactSignal for ObjectiveFact<T> {
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 {
        self.ripa_clamp_rad
    }
}

/// Concrete wrapper classifying any [`RawPhaseProjectable`] payload `T` as a **Subjective Persona** (`Origin 2-B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubjectivePersona<T> {
    /// Underlying raw payload (`&str`, `&[u8]`, `&[f32]`, or `[gcso_q7_t; 64]`).
    pub payload: T,
    /// Base gravitational pull energy $E_{\mathrm{persona}} \ge 0$.
    pub base_energy: f32,
    /// Direct 48-bit `.gcso` (mapped-storage) persona section offset (`bits 0..=47`).
    pub zimms_offset48: u64,
    /// Full 32-bit macro anchor ID on the unified manifold.
    pub anchor_id: u32,
    /// RIPA phase clamp limit in radians (defaults to [`DEFAULT_PERSONA_RIPA_CLAMP_RAD`]).
    pub ripa_clamp_rad: f32,
}

impl<T> SubjectivePersona<T> {
    /// Constructs a new [`SubjectivePersona`] with explicit base energy, 48-bit mapped-storage offset, and 32-bit anchor ID.
    #[inline(always)]
    #[must_use]
    pub const fn new(payload: T, base_energy: f32, zimms_offset48: u64, anchor_id: u32) -> Self {
        Self {
            payload, base_energy: branchless_flush_subnormal_f32(branchless_relu_f32(base_energy)),
            zimms_offset48: zimms_offset48 & gcso_tagged_ptr_t::ADDR_MASK, anchor_id,
            ripa_clamp_rad: DEFAULT_PERSONA_RIPA_CLAMP_RAD,
        }
    }

    /// Builder method overriding the RIPA clamp limit in radians.
    #[inline(always)]
    #[must_use]
    pub const fn with_ripa_clamp_rad(mut self, max_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(max_rad) & (max_rad > 0.0);
        self.ripa_clamp_rad = branchless_select_f32(valid, max_rad, self.ripa_clamp_rad);
        self
    }
}

impl<T: RawPhaseProjectable> CausalProvenanceSignal for SubjectivePersona<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        CAUSAL_AXIS_SUBJECTIVE_PERSONA
    }
}

impl<T: RawPhaseProjectable> CognitiveGroundingSignal for SubjectivePersona<T> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.payload.project_to_q7_64()
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.payload.project_to_words_64()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.base_energy
    }
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        self.zimms_offset48
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl<T: RawPhaseProjectable> SubjectivePersonaSignal for SubjectivePersona<T> {
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 {
        self.ripa_clamp_rad
    }
}

/// Concrete wrapper for **Origin 2-C (`ContradictionGuardSignal`)** combining an optional
/// [`ContrastivePhaseBloomFilter`] (or `()`) with a sheaf cohomology obstruction residual.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContradictionGuard<B = ContrastivePhaseBloomFilter> {
    /// Underlying Bloom filter or unit `()`.
    pub bloom: B,
    /// Observed cohomological obstruction residual norm $\| \mathbf{r}_{\mathrm{obs}} \| \ge 0$.
    pub cohomology_obstruction: f32,
    /// Threshold above which `cohomology_obstruction` triggers repulsive escape (defaults to `0.25`).
    pub obstruction_threshold: f32,
    /// Repulsive Bloom bucket density threshold (`1..=15`, defaults to `4`).
    pub bloom_repulsion_threshold: u8,
}

impl<B> ContradictionGuard<B> {
    /// Constructs a new [`ContradictionGuard`] with explicit cohomological obstruction and Bloom threshold.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        bloom: B,
        cohomology_obstruction: f32,
        bloom_repulsion_threshold: u8,
    ) -> Self {
        Self {
            bloom,
            cohomology_obstruction: branchless_flush_subnormal_f32(branchless_relu_f32(
                cohomology_obstruction,
            )),
            obstruction_threshold: 0.25,
            bloom_repulsion_threshold: branchless_select_u8(
                bloom_repulsion_threshold > 0,
                bloom_repulsion_threshold,
                4,
            ),
        }
    }

    /// Builder method setting the cohomological obstruction threshold.
    #[inline(always)]
    #[must_use]
    pub const fn with_obstruction_threshold(mut self, thresh: f32) -> Self {
        let valid = branchless_is_finite_f32(thresh) & (thresh > 0.0);
        self.obstruction_threshold =
            branchless_select_f32(valid, thresh, self.obstruction_threshold);
        self
    }
}

impl<B: BranchlessKernel> CausalProvenanceSignal for ContradictionGuard<B> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let obstructed = self.cohomology_obstruction > self.obstruction_threshold;
        CAUSAL_AXIS_CONTRADICTION_GUARD
            | branchless_select_u32(obstructed, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        if self.cohomology_obstruction <= 0.0 {
            return 0.0;
        }
        let inv_thresh = if self.obstruction_threshold == 0.25 {
            4.0
        } else if self.obstruction_threshold >= 1.0e-30 {
            fast_inv_pos_normal_f32(self.obstruction_threshold)
        } else {
            branchless_inv_pos_f32(self.obstruction_threshold)
        };
        let ratio = self.cohomology_obstruction * inv_thresh;
        branchless_clamp_f32(ratio * 0.5, 0.0, 1.0)
    }
}

impl<B: BranchlessKernel> CognitiveGroundingSignal for ContradictionGuard<B> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        [0i8; MAX_ATTENTION_HEADS]
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        [0u64; NUM_STIGMERGIC_GRANULES]
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.cohomology_obstruction
    }
}

impl ContradictionGuardSignal for ContradictionGuard<ContrastivePhaseBloomFilter> {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let words = load_q7_words_64(candidate_phases);
        self.evaluate_repulsion_severity_from_words(&words, candidate_phases)
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        let obs_sev = branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        );
        if obs_sev >= 1.0 {
            return 1.0;
        }
        let density = self.bloom.query_repulsive_density_from_words(candidate_words);
        let bloom_hit = density >= self.bloom_repulsion_threshold;
        let bloom_sev = branchless_select_f32(
            bloom_hit,
            branchless_clamp_f32(0.60 + (density as f32) * 0.05, 0.60, 1.0),
            0.0,
        );
        branchless_max_f32(obs_sev, bloom_sev)
    }

    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }
}

impl ContradictionGuardSignal for ContradictionGuard<()> {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        )
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        _candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        )
    }

    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }
}

/// 64-byte Q7 phase anchor for contrastive/negative guard attractors, enabling
/// query-resonance repulsion checks inside [`ResonantSignalSlice`] and 3-origin tuples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardPhaseAnchor {
    /// Per-head Q7 phase deltas (`[i8; 64]`).
    pub phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Observed cohomological obstruction residual norm $\| \mathbf{r}_{\mathrm{obs}} \| \ge 0$.
    pub cohomology_obstruction: f32,
    /// Threshold above which `cohomology_obstruction` triggers repulsive escape.
    pub obstruction_threshold: f32,
}

impl GuardPhaseAnchor {
    /// Constructs a new [`GuardPhaseAnchor`] from a 64-head Q7 phase array.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
        cohomology_obstruction: f32,
        obstruction_threshold: f32,
    ) -> Self {
        Self {
            phase_deltas,
            cohomology_obstruction: branchless_flush_subnormal_f32(branchless_relu_f32(
                cohomology_obstruction,
            )),
            obstruction_threshold,
        }
    }
}

impl CausalProvenanceSignal for GuardPhaseAnchor {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let obstructed = self.cohomology_obstruction > self.obstruction_threshold;
        CAUSAL_AXIS_CONTRADICTION_GUARD
            | branchless_select_u32(obstructed, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        let obstructed = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obstructed,
            branchless_clamp_f32(self.cohomology_obstruction, 0.0, 1.0),
            0.0,
        )
    }
}

impl CognitiveGroundingSignal for GuardPhaseAnchor {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.phase_deltas.project_to_words_64()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.cohomology_obstruction
    }
}

impl ContradictionGuardSignal for GuardPhaseAnchor {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        )
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        _candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        )
    }

    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }
}

impl CausalProvenanceSignal for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        CAUSAL_AXIS_CONTRADICTION_GUARD
    }
}

impl CognitiveGroundingSignal for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        [0i8; MAX_ATTENTION_HEADS]
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        [0u64; NUM_STIGMERGIC_GRANULES]
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        0.0
    }
}

impl ContradictionGuardSignal for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let words = load_q7_words_64(candidate_phases);
        self.evaluate_repulsion_severity_from_words(&words, candidate_phases)
    }
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let density = self.query_repulsive_density_from_words(candidate_words);
        branchless_select_f32(
            density >= 3,
            branchless_clamp_f32(0.60 + (density as f32) * 0.05, 0.60, 1.0),
            0.0,
        )
    }
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }
}

