// name: src/core/src/traits/gateway.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.3 2-Origin Supertrait Hierarchy, Causal Provenance, Single Composite Manifold Gateway,
//! and autonomous Gateway-derived physical parameter synthesis.
//!
//! Macro-basin clustering ([`CrystallizedAttractorRecord`]) and 4-section `.gcso` zero-copy
//! container persistence ([`GcsoFourSectionView`]) live in [`crate::crystal`] and
//! [`crate::storage`] respectively and are re-exported here for backward compatibility.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
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
#![allow(clippy::field_reassign_with_default)]
use core::mem::{align_of, offset_of, size_of};
use super::aligned::{AlignedSlice, AlignedSlice32, AlignedSliceMut32, is_aligned_to};
use super::branchless::{
    BranchlessKernel, atan2_interleaved_pairs_to_q7_word_x8_simd, atan2_to_q7_word_x8_simd,
    branchless_phase_coherence_inv_temp_scale, branchless_q7_circular_resultant_64,
    coherence_from_uniform_words_x8, coherence_from_word_arrays_with_both_uniform_flags_64,
    scale_f32_slice_unrolled8, swar_slerp_words_with_uniform_flag_64, swar_words_equal_64,
    swar_words_nonzero_64,
};
use super::primitives::{
    SWAR_LOW7_MASK_64, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, branchless_abs_f32,
    branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_crc32_ieee,
    branchless_flush_subnormal_f32, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_lerp_f32, branchless_max_f32, branchless_min_f32, branchless_rad_to_qdps_step_u8,
    branchless_relu_f32, branchless_select_f32, branchless_select_i8, branchless_select_u32,
    branchless_select_u64, branchless_select_u8, branchless_select_usize,
    branchless_sincos_q7_ripa, branchless_splitmix64, crc32_block64_zero_seed_words,
    crc32_step_block64_words, crc32_step_l16, crc32_step_l4, crc32_step_l8,
    fast_inv_pos_normal_f32, is_word_array_uniform_64, load_q7_words_64, swar_q7_circular_add_x8,
    swar_q7_circular_diff_x8, swar_q7_mean_i8_from_words_64,
    swar_q7_mean_i8_from_words_with_uniform_flag_64, swar_q7_neg_sat_x8,
    swar_q7_qdps_filter_words_64, swar_q7_qdps_filter_words_with_uniform_flag_64,
    swar_q7_scale_le128_with_bias_x8, trail_phase_words_mut, trail_phase_words_ref,
};
use super::primitives::{
    ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE,
    ACTION_MODE_PSPM_ROUTE, ACTION_MODE_SRL_INJECT, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE,
    ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE, DEFAULT_QDPS_MIN_STEP_RAD,
    DEFAULT_RIPA_CLAMP_MAX_RAD, GcsoResult, MAX_ATTENTION_HEADS, Pod, pack_action_slot_tag,
};
use super::rope::{
    PhaseSteering, PspmRouter, dispatch_fused_ripa_and_pspm_words_core,
    resolve_autonomous_swarm_rot_dim, rotate_fused_self_ripa_and_pspm_default_words,
    rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words, steer_from_swar_words_core_safe,
};
use super::telemetry::{
    ContrastivePhaseBloomFilter, DAES_METRIC_FAST_PATH_HIT, DAES_METRIC_LAYER_SKIP_COMPENSATED,
    DAES_METRIC_PHASE_STEERED, DAES_METRIC_PSPM_ROUTED, DAES_METRIC_REPULSION_APPLIED,
    DAES_METRIC_SLOT_SYNERGY_ACTIVE, DAES_METRIC_SRL_INJECTED, MAX_ATTRACTOR_PULL_FORCE,
    NUM_SWARM_MICRO_CELLS, SwarmResolutionReport, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
use super::{ActionHub, validate_optional_qk_tensors};
use crate::abi::{
    GCSO_ABI_VERSION_HEX, GCSO_ERROR_INVALID_ARGUMENT, gcso_config_t, gcso_daes_slot_t,
    gcso_edbc_state_t, gcso_pointer_trail_t, gcso_pspm_config_t, gcso_q7_t, gcso_srl_descriptor_t,
    gcso_tagged_ptr_t,
};

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
/// Fixed byte length (`192` bytes = `128B` `gcso_snapshot_header_t` + `64B` `gcso_zimms_descriptor_t`) of Section 0.
pub const GCSO_FOUR_SECTION_HEADER_BYTES: usize = 192;
/// Byte offset (`128`) within the 192-byte Section 0 preamble where [`crate::abi::gcso_zimms_descriptor_t`] is stored.
pub const GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET: usize = 128;

const INV_FACT_RIPA_CLAMP_RAD: f32 = 1.0 / DEFAULT_FACT_RIPA_CLAMP_RAD;
pub(crate) const MAX_REGISTER_CACHED_CENTROIDS: usize = 16;

// ===================================================================
// 2. Branchless Phase Projection Helpers (`RawPhaseProjectable`)
// ===================================================================

#[inline(always)]
#[must_use]
const fn swar_clamp_q7_min_to_neg127_x8(w: u64) -> u64 {
    let low7 = w & SWAR_LOW7_MASK_64;
    let has_low7 = low7.wrapping_add(SWAR_LOW7_MASK_64) & SWAR_SIGN_MASK_64;
    let is_80 = (w & SWAR_SIGN_MASK_64) & !has_low7;
    w | (is_80 >> 7)
}

/// Projects an arbitrary byte slice (`bytes`) directly into 8 register-resident `u64` SWAR words
/// (`[u64; 8]` representing 64 `gcso_q7_t` heads) using 8-byte SplitMix64 mixing and `-128 -> -127` clamping.
#[inline]
#[must_use]
pub const fn project_bytes_swar_to_words_64(bytes: &[u8]) -> [u64; NUM_SWARM_MICRO_CELLS] {
    let len = bytes.len();
    if len == 0 {
        return [0u64; NUM_SWARM_MICRO_CELLS];
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
    let mut words = [0u64; NUM_SWARM_MICRO_CELLS];
    let mut cell = 0usize;
    while cell < NUM_SWARM_MICRO_CELLS {
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
pub fn project_f32_slice_to_words_64(values: &[f32]) -> [u64; NUM_SWARM_MICRO_CELLS] {
    let n = values.len();
    if n == 0 {
        return [0u64; NUM_SWARM_MICRO_CELLS];
    }
    let mut words = [0u64; NUM_SWARM_MICRO_CELLS];
    if n >= 128 {
        let mut cell = 0usize;
        while cell < NUM_SWARM_MICRO_CELLS {
            let base_i = cell << 4;
            words[cell] = atan2_interleaved_pairs_to_q7_word_x8_simd(&values[base_i..base_i + 16]);
            cell += 1;
        }
        return words;
    }
    if (n >= 2) & ((n & (n - 1)) == 0) {
        if n == 2 {
            let b0 = branchless_atan2_to_q7_fast(values[1], values[0]) as u8;
            return [(b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64); NUM_SWARM_MICRO_CELLS];
        }
        if n == 4 {
            let b0 = branchless_atan2_to_q7_fast(values[1], values[0]) as u8 as u64;
            let b1 = branchless_atan2_to_q7_fast(values[3], values[2]) as u8 as u64;
            let pair16 = b0 | (b1 << 8);
            return [pair16.wrapping_mul(0x0001_0001_0001_0001); NUM_SWARM_MICRO_CELLS];
        }
        if n == 8 {
            let b0 = branchless_atan2_to_q7_fast(values[1], values[0]) as u8 as u64;
            let b1 = branchless_atan2_to_q7_fast(values[3], values[2]) as u8 as u64;
            let b2 = branchless_atan2_to_q7_fast(values[5], values[4]) as u8 as u64;
            let b3 = branchless_atan2_to_q7_fast(values[7], values[6]) as u8 as u64;
            let quad32 = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
            return [quad32 | (quad32 << 32); NUM_SWARM_MICRO_CELLS];
        }
        if n == 16 {
            let w0 = atan2_interleaved_pairs_to_q7_word_x8_simd(&values[..16]);
            return [w0; NUM_SWARM_MICRO_CELLS];
        }
        let unique_cells = branchless_select_usize(
            (n >> 4) < NUM_SWARM_MICRO_CELLS,
            n >> 4,
            NUM_SWARM_MICRO_CELLS,
        );
        let mut cell = 0usize;
        while cell < unique_cells {
            let base_i = cell << 4;
            words[cell] = atan2_interleaved_pairs_to_q7_word_x8_simd(&values[base_i..base_i + 16]);
            cell += 1;
        }
        while cell < NUM_SWARM_MICRO_CELLS {
            words[cell] = words[cell & (unique_cells - 1)];
            cell += 1;
        }
        return words;
    }
    let mut cell = 0usize;
    while cell < NUM_SWARM_MICRO_CELLS {
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
    words: &[u64; NUM_SWARM_MICRO_CELLS],
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    // SAFETY: `[u64; 8]` and `[i8; 64]` have identical size (64 bytes) and Pod bit validity.
    unsafe {
        core::mem::transmute::<[u64; NUM_SWARM_MICRO_CELLS], [gcso_q7_t; MAX_ATTENTION_HEADS]>(
            *words,
        )
    }
}

#[inline(always)]
#[must_use]
pub(crate) const fn fast_mean_q7_tag_from_words_64(words: &[u64; NUM_SWARM_MICRO_CELLS]) -> gcso_q7_t {
    let raw = swar_q7_mean_i8_from_words_64(words);
    branchless_select_i8(raw == -128, -127, raw)
}

/// Trait for raw payloads that can be projected onto the 64-head $SO(2)^{64}$ Q7 phase manifold.
pub trait RawPhaseProjectable: BranchlessKernel {
    /// Projects `self` onto a 64-head Q7 phase array (`[gcso_q7_t; 64]`).
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS];

    /// Projects `self` directly into 8 register-resident `u64` SWAR words (`[u64; 8]`).
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        load_q7_words_64(&self.project_to_q7_64())
    }
}

impl RawPhaseProjectable for () {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        [0i8; MAX_ATTENTION_HEADS]
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        [0u64; NUM_SWARM_MICRO_CELLS]
    }
}

impl RawPhaseProjectable for [gcso_q7_t; MAX_ATTENTION_HEADS] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        *self
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        load_q7_words_64(self)
    }
}

impl RawPhaseProjectable for [u8] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self)
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        project_bytes_swar_to_words_64(self)
    }
}

impl<const N: usize> RawPhaseProjectable for [u8; N] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self.as_slice())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        project_bytes_swar_to_words_64(self.as_slice())
    }
}

impl RawPhaseProjectable for str {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self.as_bytes())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        project_bytes_swar_to_words_64(self.as_bytes())
    }
}

impl RawPhaseProjectable for &str {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self.as_bytes())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        project_bytes_swar_to_words_64(self.as_bytes())
    }
}

impl RawPhaseProjectable for &[u8] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self)
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        project_bytes_swar_to_words_64(self)
    }
}

impl RawPhaseProjectable for [f32] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_64(self)
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        project_f32_slice_to_words_64(self)
    }
}

impl RawPhaseProjectable for &[f32] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_64(self)
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        project_f32_slice_to_words_64(self)
    }
}

impl<const N: usize> RawPhaseProjectable for [f32; N] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_64(self.as_slice())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        project_f32_slice_to_words_64(self.as_slice())
    }
}

impl<'a, const ALIGN: usize> RawPhaseProjectable for AlignedSlice<'a, f32, ALIGN> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_64(self.as_slice())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        project_f32_slice_to_words_64(self.as_slice())
    }
}

impl<'a, const ALIGN: usize> RawPhaseProjectable for AlignedSlice<'a, u8, ALIGN> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_bytes_swar_to_q7_64(self.as_slice())
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
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
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
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
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        *trail_phase_words_ref(self)
    }
}

impl RawPhaseProjectable for GatewayFrame {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        *self.phase_words_ref()
    }
}

impl RawPhaseProjectable for CrystallizedAttractorRecord {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_profile
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
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

/// Root supertrait for all signals entering the GCSO Manifold Gateway.
pub trait CausalProvenanceSignal: BranchlessKernel {
    /// Returns the bitwise provenance flags (`CAUSAL_AXIS_* | CAUSAL_CAUSE_*`) contributed by `self`.
    fn causal_provenance_bits(&self) -> u32;

    /// Returns the normalized anomaly or constraint severity in $[0.0, 1.0]$.
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        0.0
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
}

// ===================================================================
// 4. Origin 1: System & Physical Constraint Blanket (`SystemConstraintSignal`)
// ===================================================================

/// **Origin 1 Supertrait (`SystemConstraintSignal: CausalProvenanceSignal`)**:
/// Hardware resource, game-coexistence QoS, and EDBC entropy constraints governing compute budgeting.
pub trait SystemConstraintSignal: CausalProvenanceSignal {
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

// ===================================================================
// 5. Origin 2: Cognitive Grounding Hierarchy (`Fact` x `Persona` x `ContradictionGuard`)
// ===================================================================

/// **Origin 2 Parent Supertrait (`CognitiveGroundingSignal: CausalProvenanceSignal`)**:
/// Common phase-projection and `.gcso` (ZIMMS) 48-bit section-offset addressing foundation.
pub trait CognitiveGroundingSignal: CausalProvenanceSignal {
    /// Projects this grounding signal onto the 64-head $SO(2)^{64}$ Q7 phase manifold.
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS];

    /// Fast zero-copy borrow of the 64-head Q7 phase profile when already materialized in memory.
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        None
    }

    /// Projects this grounding signal directly into 8 register-resident `u64` SWAR words (`[u64; 8]`).
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        match self.grounding_phases_ref_64() {
            Some(arr_ref) => load_q7_words_64(arr_ref),
            None => load_q7_words_64(&self.project_grounding_phases_64()),
        }
    }

    /// Base gravitational pull energy $E_{\mathrm{base}} \ge 0$ of this grounding signal.
    fn grounding_base_energy(&self) -> f32;

    /// Canonical 48-bit `.gcso` (ZIMMS) section or RAG record offset (`bits 0..=47` of `gcso_tagged_ptr_t`).
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 {
        0
    }

    /// Full 32-bit macro attractor anchor identifier on the Single Unified Attractor Manifold.
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        0
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
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        (**self).project_grounding_words_64()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        (**self).grounding_base_energy()
    }
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 {
        (**self).zimms_section_offset48()
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        (**self).grounding_anchor_id()
    }
}

/// **Origin 2-A Subtrait (`ObjectiveFactSignal: CognitiveGroundingSignal`)**:
/// External RAG chunks, database records, and objective conversation logs (`Slot 2`, narrow RIPA clamp `0.14 rad`).
pub trait ObjectiveFactSignal: CognitiveGroundingSignal {
    /// RIPA phase clamp limit in radians for objective factual grounding (defaults to `0.14 rad`).
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 {
        DEFAULT_FACT_RIPA_CLAMP_RAD
    }

    /// PSPM `Fact` sub-head phase gain multiplier (defaults to `1.15`).
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
    /// RIPA phase clamp limit in radians for subjective persona modulation (defaults to `0.24 rad`).
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 {
        DEFAULT_PERSONA_RIPA_CLAMP_RAD
    }

    /// PSPM `Explore` / `Logic` sub-head phase gain multiplier (defaults to `1.20`).
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
    /// Evaluates repulsive severity in $[0.0, 1.0]$ against `candidate_phases`.
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32;

    /// Evaluates repulsive severity directly from 8 register-resident `u64` SWAR words (`candidate_words`).
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
    /// Direct 48-bit `.gcso` (ZIMMS) section offset (`bits 0..=47`).
    pub zimms_offset48: u64,
    /// Full 32-bit macro anchor ID on the unified manifold.
    pub anchor_id: u32,
    /// RIPA phase clamp limit in radians (defaults to [`DEFAULT_FACT_RIPA_CLAMP_RAD`]).
    pub ripa_clamp_rad: f32,
}

impl<T> ObjectiveFact<T> {
    /// Constructs a new [`ObjectiveFact`] with explicit base energy, 48-bit ZIMMS offset, and 32-bit anchor ID.
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
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        self.payload.project_to_words_64()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.base_energy
    }
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 {
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
    /// Direct 48-bit `.gcso` (ZIMMS) persona section offset (`bits 0..=47`).
    pub zimms_offset48: u64,
    /// Full 32-bit macro anchor ID on the unified manifold.
    pub anchor_id: u32,
    /// RIPA phase clamp limit in radians (defaults to [`DEFAULT_PERSONA_RIPA_CLAMP_RAD`]).
    pub ripa_clamp_rad: f32,
}

impl<T> SubjectivePersona<T> {
    /// Constructs a new [`SubjectivePersona`] with explicit base energy, 48-bit ZIMMS offset, and 32-bit anchor ID.
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
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        self.payload.project_to_words_64()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.base_energy
    }
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 {
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
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        [0u64; NUM_SWARM_MICRO_CELLS]
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
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
        _candidate_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        [0u64; NUM_SWARM_MICRO_CELLS]
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
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS],
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

// ===================================================================
// 6. Single Composite Gateway Frame (`GatewayFrame` 128B Pod) & Trait
// ===================================================================

/// 128-byte (2-cacheline, `#[repr(C, align(64))]`) [`Pod`] composite descriptor produced by
/// [`ManifoldGatewaySignal::synthesize_gateway_frame`].
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatewayFrame {
    /// Synthesized 64-bit tagged pointer (`[Phase(8b) | Slot(3b) | ActionMode(5b) | ZimmsAddr(48b)]`, `offset 0..8`).
    pub tagged_ptr: u64,
    /// Direct 48-bit `.gcso` (ZIMMS) section offset (`bits 0..=47`, `offset 8..16`).
    pub zimms_section_offset48: u64,
    /// Full 32-bit macro attractor anchor ID on the Single Unified Attractor Manifold (`offset 16..20`).
    pub target_anchor_id: u32,
    /// Combined gravitational pull energy $E_{\mathrm{eff}} \ge 0$ (`offset 20..24`).
    pub effective_energy: f32,
    /// Normalized RIPA soft-clamping limit in radians ($\theta_{\max}$, `offset 24..28`).
    pub ripa_clamp_rad: f32,
    /// QDPS minimum step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$, `offset 28..32`).
    pub qdps_min_step_rad: f32,
    /// Multiplicative inverse-temperature logit scale $\beta_{\mathrm{eff}} \in [0.25, 4.0]$ (`offset 32..36`).
    pub effective_inv_temp: f32,
    /// Cosine phase coherence $\mathcal{C}_{\mathrm{FP}} \in [-1, 1]$ between Objective Fact and Subjective Persona (`offset 36..40`).
    pub fact_persona_coherence: f32,
    /// Repulsive anomaly severity in $[0.0, 1.0]$ (`> 0.5` indicates anti-phase repulsion, `offset 40..44`).
    pub repulsion_severity: f32,
    /// Bitwise OR of active origin axes (`CAUSAL_AXIS_*`) and diagnostic causes (`CAUSAL_CAUSE_*`, `offset 44..48`).
    pub provenance_bits: u32,
    /// Primary root-cause classification code (`ROOT_CAUSE_*`, `offset 48`).
    pub dominant_cause_code: u8,
    /// Dominant 3-bit working lane ID (`ATTRACTOR_SLOT_*`, `0..=7`, `offset 49`).
    pub dominant_slot: u8,
    /// Resolved 5-bit `ACTION_MODE_*` execution flags (`offset 50`).
    pub action_mode_flags: u8,
    /// Active 8-cell Micro-Swarm mask (`0x00..=0xFF`, `offset 51`).
    pub active_swarm_cell_mask: u8,
    /// Ceiling-rounded Q7 integer step filter cutoff (`0..=128`, `offset 52`).
    pub qdps_min_step_q7: u8,
    /// Recommended PSPM `Fact` sub-head count (`offset 53`).
    pub pspm_fact_heads: u8,
    /// Recommended PSPM `Logic` sub-head count (`offset 54`).
    pub pspm_logic_heads: u8,
    /// Recommended PSPM `Explore` sub-head count (`offset 55`).
    pub pspm_explore_heads: u8,
    /// Quantized PSPM `Fact` sub-head phase gain in Q7 (`128` = `1.0x`, `offset 56`).
    pub pspm_fact_gain_q7: u8,
    /// Quantized PSPM `Logic` sub-head phase gain in Q7 (`128` = `1.0x`, `offset 57`).
    pub pspm_logic_gain_q7: u8,
    /// Quantized PSPM `Explore` sub-head phase gain in Q7 (`128` = `1.0x`, `offset 58`).
    pub pspm_explore_gain_q7: u8,
    /// Fast 64-head SWAR arithmetic mean Q7 phase tag $\bar{\theta}_{Q7} \in [-127, 127]$ (`offset 59`).
    pub mean_phase_tag_q7: gcso_q7_t,
    /// Active attention head count for this frame (`1..=64`, defaults to `64`, `offset 60..62`).
    pub active_head_count: u16,
    /// Frame status and routing telemetry flags (`offset 62..64`).
    pub frame_flags: u16,
    /// Synthesized 64-head Q7 phase delta profile on $SO(2)^{64}$ (`64` bytes at `offset 64..128`).
    pub phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

// SAFETY: `GatewayFrame` is `#[repr(C, align(64))]` of exact size 128 bytes with zero uninitialized padding.
unsafe impl Pod for GatewayFrame {}

impl Default for GatewayFrame {
    #[inline(always)]
    fn default() -> Self {
        Self {
            tagged_ptr: gcso_tagged_ptr_t::pack(
                0,
                pack_action_slot_tag(ATTRACTOR_SLOT_SYSTEM_BASELINE, ACTION_PRESET_EXPRESS_PHASE),
                0,
            )
            .raw(),
            zimms_section_offset48: 0,
            target_anchor_id: 0,
            effective_energy: 0.0,
            ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
            effective_inv_temp: 1.0,
            fact_persona_coherence: 1.0,
            repulsion_severity: 0.0,
            provenance_bits: 0,
            dominant_cause_code: ROOT_CAUSE_NOMINAL,
            dominant_slot: ATTRACTOR_SLOT_SYSTEM_BASELINE,
            action_mode_flags: ACTION_PRESET_EXPRESS_PHASE,
            active_swarm_cell_mask: 0xFF,
            qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
            pspm_fact_heads: 24,
            pspm_logic_heads: 24,
            pspm_explore_heads: 16,
            pspm_fact_gain_q7: 128,
            pspm_logic_gain_q7: 128,
            pspm_explore_gain_q7: 128,
            mean_phase_tag_q7: 0,
            active_head_count: MAX_ATTENTION_HEADS as u16,
            frame_flags: 0,
            phase_deltas: [0i8; MAX_ATTENTION_HEADS],
        }
    }
}

impl GatewayFrame {
    /// Zero-copy `const` reference reinterpretation of `self.phase_deltas` (`offset 64..128`,
    /// guaranteed 64-byte aligned by `#[repr(C, align(64))]`) as 8 native `u64` SWAR words (`&[u64; 8]`).
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_ref(&self) -> &[u64; NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `GatewayFrame` is `#[repr(C, align(64))]` and `phase_deltas` is at byte offset 64
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe { &*(self.phase_deltas.as_ptr() as *const [u64; NUM_SWARM_MICRO_CELLS]) }
    }

    /// Borrows `self.phase_deltas` (`offset 64` of `#[repr(C, align(64))]` [`GatewayFrame`],
    /// guaranteed 64-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR word array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_mut(&mut self) -> &mut [u64; NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `GatewayFrame` is `#[repr(C, align(64))]` and `phase_deltas` is at byte offset 64
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe { &mut *(self.phase_deltas.as_mut_ptr() as *mut [u64; NUM_SWARM_MICRO_CELLS]) }
    }

    /// Returns `true` when this [`GatewayFrame`] carries an active cohomology or Bloom repulsion guard
    /// that must be rejected before mutating a [`CrystallizedAttractorRecord`] centroid.
    #[inline(always)]
    #[must_use]
    pub const fn is_repulsed(&self) -> bool {
        (self.repulsion_severity > 0.50)
            | ((self.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0)
            | (self.dominant_cause_code == ROOT_CAUSE_COHOMOLOGY_REPULSION)
            | ((self.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0)
    }

    /// Resolves the **Swarm- & Blanket-Driven Autonomous Frequency Aperture** (`rot_dim`, even
    /// integer in `2..=head_dim`) directly from this [`GatewayFrame`]'s `provenance_bits` and
    /// `active_swarm_cell_mask` combined with `moving_z_entropy` and `von_mises_kappa`.
    #[inline(always)]
    #[must_use]
    pub const fn resolve_autonomous_rot_dim(
        &self,
        head_dim: usize,
        moving_z_entropy: f32,
        von_mises_kappa: f32,
    ) -> usize {
        resolve_autonomous_swarm_rot_dim(
            head_dim,
            self.provenance_bits,
            self.active_swarm_cell_mask,
            moving_z_entropy,
            von_mises_kappa,
        )
    }

    /// Extracts the 16-byte [`CausalProvenanceReport`] from this [`GatewayFrame`] in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn to_causal_report(&self) -> CausalProvenanceReport {
        CausalProvenanceReport {
            provenance_bits: self.provenance_bits,
            fact_persona_coherence: self.fact_persona_coherence,
            anomaly_severity: self.repulsion_severity,
            dominant_cause_code: self.dominant_cause_code,
            dominant_slot: self.dominant_slot,
            action_mode_flags: self.action_mode_flags,
            active_swarm_mask: self.active_swarm_cell_mask,
        }
    }

    /// Converts the Gateway's recommended Fact/Logic/Explore head split into a [`gcso_pspm_config_t`]
    /// scaled proportionally to `total_heads`.
    #[inline(always)]
    #[must_use]
    pub const fn to_pspm_config_for_heads(&self, total_heads: usize) -> gcso_pspm_config_t {
        let h = if total_heads > MAX_ATTENTION_HEADS {
            MAX_ATTENTION_HEADS
        } else {
            total_heads
        };
        if h == 0 {
            return gcso_pspm_config_t {
                num_fact_heads: 0,
                num_logic_heads: 0,
                num_explore_heads: 0,
                flags: 0,
                fact_phase_gain: 1.0,
                logic_phase_gain: 1.0,
                explore_phase_gain: 1.0,
                reserved: [0; 12],
            };
        }
        let f_raw = ((self.pspm_fact_heads as usize) * h + 32) >> 6;
        let l_raw = ((self.pspm_logic_heads as usize) * h + 32) >> 6;
        let f_cnt = if f_raw > h { h } else { f_raw };
        let rem = h - f_cnt;
        let l_cnt = if l_raw > rem { rem } else { l_raw };
        let e_cnt = h - f_cnt - l_cnt;
        gcso_pspm_config_t {
            num_fact_heads: f_cnt as u16,
            num_logic_heads: l_cnt as u16,
            num_explore_heads: e_cnt as u16,
            flags: 1,
            fact_phase_gain: (self.pspm_fact_gain_q7 as f32) * (1.0 / 128.0),
            logic_phase_gain: (self.pspm_logic_gain_q7 as f32) * (1.0 / 128.0),
            explore_phase_gain: (self.pspm_explore_gain_q7 as f32) * (1.0 / 128.0),
            reserved: [0; 12],
        }
    }

    /// Returns `true` when all three PSPM sub-head group gains (`pspm_fact_gain_q7`,
    /// `pspm_logic_gain_q7`, `pspm_explore_gain_q7`) equal unit gain (`128` in Q0.7).
    #[inline(always)]
    #[must_use]
    pub const fn has_unit_pspm_gains(&self) -> bool {
        (self.pspm_fact_gain_q7 == 128)
            & (self.pspm_logic_gain_q7 == 128)
            & (self.pspm_explore_gain_q7 == 128)
    }

    /// Autonomously derives the 8-cell Micro-Swarm step gain (`32..=128` in Q0.7) from this
    /// [`GatewayFrame`]'s `effective_energy` and `fact_persona_coherence` without manual tuning.
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_swarm_step_gain_u7(&self) -> u8 {
        let e = branchless_clamp_f32(branchless_relu_f32(self.effective_energy), 0.0, 4.0);
        let coh = branchless_clamp_f32(self.fact_persona_coherence, 0.0, 1.0);
        let scaled = 48.0 + e * 16.0 + coh * 16.0;
        let clamped = branchless_clamp_f32(scaled, 32.0, 128.0);
        (clamped + 0.5) as u8
    }

    /// Autonomously derives the Micro-Swarm phase convergence tolerance in Q0.7 (`8..=20`) from
    /// this [`GatewayFrame`]'s active causal origins (`provenance_bits`).
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_swarm_tolerance_q7(&self) -> u8 {
        let has_fact = (self.provenance_bits & CAUSAL_AXIS_OBJECTIVE_FACT) != 0;
        let has_persona = (self.provenance_bits & CAUSAL_AXIS_SUBJECTIVE_PERSONA) != 0;
        let base = branchless_select_u8(has_persona, 16, 12);
        branchless_select_u8(has_fact, 10, base)
    }

    /// Autonomously derives the Micro-Swarm quorum cell count threshold (`4..=7` out of 8 cells)
    /// from this [`GatewayFrame`]'s active causal origins and repulsion severity.
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_swarm_quorum_threshold(&self) -> u8 {
        let strict = ((self.provenance_bits
            & (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_CONTRADICTION_GUARD))
            != 0)
            | (self.repulsion_severity > 0.25);
        branchless_select_u8(strict, 6, 5)
    }

    /// Autonomously derives the unresolved-cell damping factor (`32..=96` in Q0.7) from this
    /// [`GatewayFrame`]'s `repulsion_severity`.
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_unresolved_damping_u7(&self) -> u8 {
        branchless_select_u8(self.is_repulsed(), 32, 64)
    }

    /// Autonomously derives the Bayesian Limina geodesic interpolation weight (`32..=96` in Q0.7)
    /// from this [`GatewayFrame`]'s `fact_persona_coherence`.
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_liminal_interpolation_u7(&self) -> u8 {
        let coh = branchless_clamp_f32(self.fact_persona_coherence, -1.0, 1.0);
        let scaled = 64.0 + coh * 32.0;
        (branchless_clamp_f32(scaled, 32.0, 96.0) + 0.5) as u8
    }

    /// Rotates a 32-float `(Q, K, V)` block triple in-place by the RIPA-clamped $SO(2)$ phase angle
    /// of `head_idx` (`0..64`).
    #[inline(always)]
    pub fn rotate_qkv_block32_in_place(
        &self,
        q_block32: &mut [f32; 32],
        k_block32: &mut [f32; 32],
        v_block32: &mut [f32; 32],
        head_idx: usize,
    ) {
        let phase_q7 = self.phase_deltas[head_idx & (MAX_ATTENTION_HEADS - 1)];
        let (sin_t, cos_t) = branchless_sincos_q7_ripa(phase_q7, self.ripa_clamp_rad);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY: `target_feature = "avx2"` is enabled and `q_block32`, `k_block32`, `v_block32` are disjoint 32-float arrays.
            unsafe {
                use core::arch::x86_64::_mm256_set1_ps;
                super::branchless::avx2_rotate_so2_triple_block32(
                    q_block32.as_mut_ptr(),
                    k_block32.as_mut_ptr(),
                    v_block32.as_mut_ptr(),
                    _mm256_set1_ps(cos_t),
                    _mm256_set1_ps(sin_t),
                );
            }
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut idx = 0usize;
            while idx < 32 {
                let (q0, q1) =
                    super::primitives::branchless_rotate_so2_pair(
                        q_block32[idx],
                        q_block32[idx + 1],
                        sin_t,
                        cos_t,
                    );
                q_block32[idx] = q0;
                q_block32[idx + 1] = q1;
                let (k0, k1) =
                    super::primitives::branchless_rotate_so2_pair(
                        k_block32[idx],
                        k_block32[idx + 1],
                        sin_t,
                        cos_t,
                    );
                k_block32[idx] = k0;
                k_block32[idx + 1] = k1;
                let (v0, v1) =
                    super::primitives::branchless_rotate_so2_pair(
                        v_block32[idx],
                        v_block32[idx + 1],
                        sin_t,
                        cos_t,
                    );
                v_block32[idx] = v0;
                v_block32[idx + 1] = v1;
                idx += 2;
            }
        }
    }
}

/// **The Single Composite Manifold Gateway Trait (`ManifoldGatewaySignal: CausalProvenanceSignal`)**.
pub trait ManifoldGatewaySignal: CausalProvenanceSignal {
    /// Synthesizes all constituent blanket signals into a 128-byte [`GatewayFrame`] in
    /// constant-time branchless arithmetic ($C_{\mathrm{alloc}} = 0$).
    fn synthesize_gateway_frame(&self) -> GatewayFrame;

    /// Synthesizes `self` into both a 128-byte [`GatewayFrame`] and its register-resident 8 `u64`
    /// SWAR phase words (`[u64; 8]`), eliminating stack store-to-load forwarding stalls across
    /// downstream ensemble folding, `ActionHub` steering, and Online Toroidal $k$-Means steps.
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let frame = self.synthesize_gateway_frame();
        let words = *frame.phase_words_ref();
        (frame, words)
    }

    /// Lightweight internal/ensemble fold hook returning `(GatewayFrame, [u64; 8])` where
    /// `GatewayFrame.phase_deltas`, `mean_phase_tag_q7`, and `tagged_ptr` may be deferred until
    /// the outer `[T]` ensemble accumulator completes its final reduction.
    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        self.synthesize_gateway_frame_with_words()
    }

    /// Fast predicate returning `true` if `self` carries an active non-passive signal during `[T]` ensemble folding.
    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        true
    }

    /// Synthesizes a contiguous slice of `Self` into a 128-byte [`GatewayFrame`] and its 8
    /// register-resident `u64` SWAR phase words (`[u64; 8]`), accumulating scalar metadata in
    /// registers and materializing `GatewayFrame` only once upon loop completion.
    #[inline]
    fn synthesize_gateway_slice_with_words(
        slice: &[Self],
    ) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])
    where
        Self: Sized,
    {
        let len = slice.len();
        if len == 0 {
            return (GatewayFrame::default(), [0u64; NUM_SWARM_MICRO_CELLS]);
        }
        if len == 1 {
            return slice[0].synthesize_gateway_frame_with_words();
        }

        let mut acc_w = [0u64; NUM_SWARM_MICRO_CELLS];
        let mut acc_is_uniform = true;
        let mut tot_e = 0.0_f32;
        let mut best_e = 0.0_f32;
        let mut has_active = false;

        let mut zimms_addr = 0u64;
        let mut target_anchor_id = 0u32;
        let mut ripa_clamp_rad = DEFAULT_RIPA_CLAMP_MAX_RAD;
        let mut qdps_min_step_rad = DEFAULT_QDPS_MIN_STEP_RAD;
        let mut eff_inv_temp = 1.0_f32;
        let mut fp_coh = 1.0_f32;
        let mut rep_sev = 0.0_f32;
        let mut prov_bits = 0u32;
        let mut dom_cause = ROOT_CAUSE_NOMINAL;
        let mut dom_slot = ATTRACTOR_SLOT_SYSTEM_BASELINE;
        let mut action_flags = ACTION_PRESET_EXPRESS_PHASE;
        let mut swarm_mask = 0xFFu8;
        let mut qdps_step_q7 = branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD);
        let mut pspm_fact = 24u8;
        let mut pspm_logic = 24u8;
        let mut pspm_explore = 16u8;
        let mut fact_gain_q7 = 128u8;
        let mut logic_gain_q7 = 128u8;
        let mut explore_gain_q7 = 128u8;
        let mut active_heads = MAX_ATTENTION_HEADS as u16;

        let mut i = 0usize;
        while i < len {
            if !slice[i].has_active_gateway_ensemble_signal() {
                i += 1;
                continue;
            }
            let (f, f_w) = slice[i].synthesize_gateway_ensemble_item();
            let f_e = branchless_flush_subnormal_f32(branchless_relu_f32(f.effective_energy));
            let f_has_signal = (f_e > 1.0e-12) | (f.repulsion_severity > 1.0e-12)
                | (f.provenance_bits != 0) | ((f.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0);

            if f_has_signal {
                if !has_active {
                    acc_w = f_w;
                    acc_is_uniform = is_word_array_uniform_64(&f_w);
                    tot_e = f_e;
                    best_e = f_e;
                    zimms_addr = f.zimms_section_offset48;
                    target_anchor_id = f.target_anchor_id;
                    ripa_clamp_rad = f.ripa_clamp_rad;
                    qdps_min_step_rad = f.qdps_min_step_rad;
                    eff_inv_temp = f.effective_inv_temp;
                    fp_coh = f.fact_persona_coherence;
                    rep_sev = f.repulsion_severity;
                    prov_bits = f.provenance_bits;
                    dom_cause = f.dominant_cause_code;
                    dom_slot = f.dominant_slot;
                    action_flags = f.action_mode_flags;
                    swarm_mask = f.active_swarm_cell_mask;
                    qdps_step_q7 = f.qdps_min_step_q7;
                    pspm_fact = f.pspm_fact_heads;
                    pspm_logic = f.pspm_logic_heads;
                    pspm_explore = f.pspm_explore_heads;
                    fact_gain_q7 = f.pspm_fact_gain_q7;
                    logic_gain_q7 = f.pspm_logic_gain_q7;
                    explore_gain_q7 = f.pspm_explore_gain_q7;
                    active_heads = f.active_head_count;
                    has_active = true;
                } else {
                    if f_e > 1.0e-12 {
                        let new_tot = tot_e + f_e;
                        let safe_tot = branchless_select_f32(new_tot > 1.0e-12, new_tot, 1.0);
                        let w_ratio =
                            branchless_clamp_f32(f_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
                        let s_u64 = ((w_ratio * 128.0 + 0.5) as u8).min(128) as u64;
                        if s_u64 >= 128 {
                            acc_w = f_w;
                            acc_is_uniform = is_word_array_uniform_64(&f_w);
                        } else if s_u64 > 0 && !swar_words_equal_64(&acc_w, &f_w) {
                            let f_is_uniform = acc_is_uniform && is_word_array_uniform_64(&f_w);
                            let bias_restore =
                                (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                            let (next_acc, next_uni) = swar_slerp_words_with_uniform_flag_64(
                                &acc_w,
                                acc_is_uniform,
                                &f_w,
                                f_is_uniform,
                                s_u64,
                                bias_restore,
                            );
                            acc_w = next_acc;
                            acc_is_uniform = next_uni;
                        }
                        eff_inv_temp =
                            branchless_lerp_f32(eff_inv_temp, f.effective_inv_temp, w_ratio);
                        fp_coh = branchless_lerp_f32(fp_coh, f.fact_persona_coherence, w_ratio);

                        let take_dom = f_e > best_e;
                        best_e = branchless_select_f32(take_dom, f_e, best_e);
                        target_anchor_id =
                            branchless_select_u32(take_dom, f.target_anchor_id, target_anchor_id);
                        zimms_addr = branchless_select_u64(
                            take_dom & (f.zimms_section_offset48 != 0),
                            f.zimms_section_offset48,
                            zimms_addr,
                        );
                        dom_slot = branchless_select_u8(take_dom, f.dominant_slot, dom_slot);
                        pspm_fact = branchless_select_u8(take_dom, f.pspm_fact_heads, pspm_fact);
                        pspm_logic = branchless_select_u8(take_dom, f.pspm_logic_heads, pspm_logic);
                        pspm_explore =
                            branchless_select_u8(take_dom, f.pspm_explore_heads, pspm_explore);
                        fact_gain_q7 =
                            branchless_select_u8(take_dom, f.pspm_fact_gain_q7, fact_gain_q7);
                        logic_gain_q7 =
                            branchless_select_u8(take_dom, f.pspm_logic_gain_q7, logic_gain_q7);
                        explore_gain_q7 =
                            branchless_select_u8(take_dom, f.pspm_explore_gain_q7, explore_gain_q7);
                        tot_e = new_tot;
                    }
                    action_flags |= f.action_mode_flags;
                    swarm_mask &= f.active_swarm_cell_mask;
                    prov_bits |= f.provenance_bits;
                    rep_sev = branchless_max_f32(rep_sev, f.repulsion_severity);
                    if f.dominant_cause_code > dom_cause {
                        dom_cause = f.dominant_cause_code;
                    }
                }
            }
            i += 1;
        }

        if !has_active {
            return slice[0].synthesize_gateway_frame_with_words();
        }

        let phase_deltas = words_to_q7_array_64(&acc_w);
        let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&acc_w, acc_is_uniform);
        let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
        let tagged_ptr = gcso_tagged_ptr_t::pack(zimms_addr, packed_tag, tag_q7).raw();
        (
            GatewayFrame {
                tagged_ptr,
                zimms_section_offset48: zimms_addr,
                target_anchor_id,
                effective_energy: branchless_min_f32(tot_e, MAX_ATTRACTOR_PULL_FORCE),
                ripa_clamp_rad,
                qdps_min_step_rad,
                effective_inv_temp: eff_inv_temp,
                fact_persona_coherence: fp_coh,
                repulsion_severity: rep_sev,
                provenance_bits: prov_bits,
                dominant_cause_code: dom_cause,
                dominant_slot: dom_slot,
                action_mode_flags: action_flags,
                active_swarm_cell_mask: swarm_mask,
                qdps_min_step_q7: qdps_step_q7,
                pspm_fact_heads: pspm_fact,
                pspm_logic_heads: pspm_logic,
                pspm_explore_heads: pspm_explore,
                pspm_fact_gain_q7: fact_gain_q7,
                pspm_logic_gain_q7: logic_gain_q7,
                pspm_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7,
                active_head_count: active_heads,
                frame_flags: packed_tag as u16,
                phase_deltas,
            },
            acc_w,
        )
    }
}

impl<T: ManifoldGatewaySignal + ?Sized> ManifoldGatewaySignal for &T {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        (**self).synthesize_gateway_frame()
    }
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        (**self).synthesize_gateway_frame_with_words()
    }
    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        (**self).synthesize_gateway_ensemble_item()
    }
    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        (**self).has_active_gateway_ensemble_signal()
    }
}

impl CausalProvenanceSignal for GatewayFrame {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.provenance_bits
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.repulsion_severity
    }
}

impl ManifoldGatewaySignal for GatewayFrame {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        *self
    }
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        (*self, *self.phase_words_ref())
    }
}

/// Synthesizes both a 128-byte [`GatewayFrame`] and its register-resident 8 `u64` SWAR phase words
/// (`[u64; 8]`) from optional **Origin 1 (`SystemConstraintSignal`)** and **Origin 2
/// (`ObjectiveFactSignal`, `SubjectivePersonaSignal`, `ContradictionGuardSignal`)** components.
#[inline]
#[must_use]
pub fn synthesize_multi_origin_gateway_frame_with_words<S, F, P, G>(
    sys_opt: Option<&S>,
    fact_opt: Option<&F>,
    persona_opt: Option<&P>,
    guard_opt: Option<&G>,
) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])
where
    S: SystemConstraintSignal + ?Sized,
    F: ObjectiveFactSignal + ?Sized,
    P: SubjectivePersonaSignal + ?Sized,
    G: ContradictionGuardSignal + ?Sized,
{
    let mut prov_bits = 0u32;
    let mut active_swarm_mask = 0xFFu8;
    let mut force_skip = false;
    let mut qdps_rad = DEFAULT_QDPS_MIN_STEP_RAD;
    let mut sys_sev = 0.0_f32;

    if let Some(sys) = sys_opt {
        prov_bits |= sys.causal_provenance_bits() | CAUSAL_AXIS_SYSTEM_CONSTRAINT;
        active_swarm_mask = sys.active_swarm_cells_mask();
        force_skip = sys.recommend_layer_skip();
        qdps_rad = branchless_max_f32(0.0, sys.qdps_min_step_rad_override());
        sys_sev = branchless_clamp_f32(sys.causal_anomaly_severity(), 0.0, 1.0);
        if force_skip | (active_swarm_mask != 0xFF) {
            prov_bits |= CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE;
        }
    }
    let qdps_step_u8 = branchless_rad_to_qdps_step_u8(qdps_rad);

    let mut has_fact = false;
    let mut fact_e = 0.0_f32;
    let mut fact_addr = 0u64;
    let mut fact_id = 0u32;
    let mut fact_clamp = DEFAULT_FACT_RIPA_CLAMP_RAD;
    let mut fact_gain_q7 = 128u8;
    let mut fact_w = [0u64; NUM_SWARM_MICRO_CELLS];

    if let Some(fact) = fact_opt {
        has_fact = true;
        prov_bits |= fact.causal_provenance_bits() | CAUSAL_AXIS_OBJECTIVE_FACT;
        fact_e = branchless_flush_subnormal_f32(branchless_relu_f32(fact.grounding_base_energy()));
        fact_addr = fact.zimms_section_offset48() & gcso_tagged_ptr_t::ADDR_MASK;
        fact_id = fact.grounding_anchor_id();
        let fc = fact.fact_ripa_clamp_rad();
        fact_clamp = branchless_select_f32(
            branchless_is_finite_f32(fc) & (fc > 0.0),
            fc,
            DEFAULT_FACT_RIPA_CLAMP_RAD,
        );
        let fg = branchless_clamp_f32(fact.fact_pspm_gain(), 0.25, 1.99);
        fact_gain_q7 = (fg * 128.0 + 0.5) as u8;
        fact_w = fact.project_grounding_words_64();
        if fact_e < 0.25 {
            prov_bits |= CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK;
        }
    }

    let mut has_persona = false;
    let mut pers_e = 0.0_f32;
    let mut pers_addr = 0u64;
    let mut pers_id = 0u32;
    let mut pers_clamp = DEFAULT_PERSONA_RIPA_CLAMP_RAD;
    let mut pers_gain_q7 = 128u8;
    let mut pers_w = [0u64; NUM_SWARM_MICRO_CELLS];

    if let Some(pers) = persona_opt {
        has_persona = true;
        prov_bits |= pers.causal_provenance_bits() | CAUSAL_AXIS_SUBJECTIVE_PERSONA;
        pers_e = branchless_flush_subnormal_f32(branchless_relu_f32(pers.grounding_base_energy()));
        pers_addr = pers.zimms_section_offset48() & gcso_tagged_ptr_t::ADDR_MASK;
        pers_id = pers.grounding_anchor_id();
        let pc = pers.persona_ripa_clamp_rad();
        pers_clamp = branchless_select_f32(
            branchless_is_finite_f32(pc) & (pc > 0.0),
            pc,
            DEFAULT_PERSONA_RIPA_CLAMP_RAD,
        );
        let pg = branchless_clamp_f32(pers.persona_pspm_gain(), 0.25, 1.99);
        pers_gain_q7 = (pg * 128.0 + 0.5) as u8;
        pers_w = pers.project_grounding_words_64();
    }

    let mut fp_coh = 1.0_f32;
    let mut collision = false;
    let mut blended_w = [0u64; NUM_SWARM_MICRO_CELLS];
    let mut blended_uni = true;
    #[allow(unused_mut)]
    let mut avx_precomputed_tag: Option<gcso_q7_t> = None;
    let mut eff_energy = 0.0_f32;
    let mut raw_clamp = DEFAULT_PERSONA_RIPA_CLAMP_RAD;
    let mut dom_slot = ATTRACTOR_SLOT_SYSTEM_BASELINE;
    let mut dom_anchor_id = 0u32;
    let mut zimms_addr = 0u64;
    let mut pspm_fact = 24u8;
    let mut pspm_logic = 24u8;
    let mut pspm_explore = 16u8;

    if has_fact & has_persona {
        let fact_uni = is_word_array_uniform_64(&fact_w);
        let pers_uni = is_word_array_uniform_64(&pers_w);
        let both_uni = fact_uni & pers_uni;
        let same_fp = if both_uni {
            fact_w[0] == pers_w[0]
        } else {
            swar_words_equal_64(&fact_w, &pers_w)
        };
        let diff_w0 = if both_uni {
            swar_q7_circular_diff_x8(pers_w[0], fact_w[0])
        } else {
            0
        };
        fp_coh = if same_fp {
            1.0
        } else if both_uni {
            coherence_from_uniform_words_x8(pers_w[0], fact_w[0])
        } else {
            coherence_from_word_arrays_with_both_uniform_flags_64(
                &pers_w, pers_uni, &fact_w, fact_uni,
            )
        };
        collision = fp_coh < -DEFAULT_PERSONA_FACT_COLLISION_THRESH;
        if collision {
            prov_bits |= CAUSAL_CAUSE_PERSONA_FACT_COLLISION;
        }
        let eff_pers_e = branchless_select_f32(
            collision,
            pers_e * 0.20,
            pers_e * branchless_clamp_f32(1.0 + 0.5 * fp_coh, 0.5, 1.5),
        );
        let tot_e = fact_e + eff_pers_e;
        let safe_tot = branchless_select_f32(tot_e > 1.0e-12, tot_e, 1.0);
        let pers_ratio =
            branchless_clamp_f32(eff_pers_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
        let s_u64 = ((pers_ratio * 128.0 + 0.5) as u8).min(128) as u64;
        if same_fp | (s_u64 == 0) {
            blended_w = fact_w;
            blended_uni = fact_uni;
        } else if s_u64 >= 128 {
            blended_w = pers_w;
            blended_uni = pers_uni;
        } else if both_uni {
            let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let step0 = swar_q7_scale_le128_with_bias_x8(diff_w0, s_u64, bias_restore);
            blended_w = [swar_q7_circular_add_x8(fact_w[0], step0); NUM_SWARM_MICRO_CELLS];
            blended_uni = true;
        } else {
            let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let (bw, bu) = swar_slerp_words_with_uniform_flag_64(
                &fact_w,
                fact_uni,
                &pers_w,
                pers_uni,
                s_u64,
                bias_restore,
            );
            blended_w = bw;
            blended_uni = bu;
        }
        let synergy_boost =
            branchless_select_f32(collision, 1.0, 1.0 + 0.25 * branchless_relu_f32(fp_coh));
        eff_energy = branchless_min_f32(tot_e * synergy_boost, MAX_ATTRACTOR_PULL_FORCE);
        raw_clamp = branchless_lerp_f32(fact_clamp, pers_clamp, pers_ratio);

        let fact_dom = (fact_e >= pers_e) | collision;
        dom_slot = branchless_select_u8(fact_dom, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA);
        dom_anchor_id = branchless_select_u32(
            fact_dom & (fact_id != 0),
            fact_id,
            branchless_select_u32(pers_id != 0, pers_id, fact_id),
        );
        zimms_addr = branchless_select_u64(
            fact_addr != 0,
            fact_addr,
            branchless_select_u64(pers_addr != 0, pers_addr, 0),
        );
        pspm_fact = branchless_select_u8(fact_dom, 32, 20);
        pspm_logic = 24;
        pspm_explore = branchless_select_u8(fact_dom, 8, 20);
    } else if has_fact {
        blended_w = fact_w;
        blended_uni = is_word_array_uniform_64(&fact_w);
        eff_energy = branchless_min_f32(fact_e, MAX_ATTRACTOR_PULL_FORCE);
        raw_clamp = fact_clamp;
        dom_slot = ATTRACTOR_SLOT_KNOWLEDGE;
        dom_anchor_id = fact_id;
        zimms_addr = fact_addr;
        pspm_fact = 36;
        pspm_logic = 20;
        pspm_explore = 8;
    } else if has_persona {
        blended_w = pers_w;
        blended_uni = is_word_array_uniform_64(&pers_w);
        eff_energy = branchless_min_f32(pers_e, MAX_ATTRACTOR_PULL_FORCE);
        raw_clamp = pers_clamp;
        dom_slot = ATTRACTOR_SLOT_PERSONA;
        dom_anchor_id = pers_id;
        zimms_addr = pers_addr;
        pspm_fact = 16;
        pspm_logic = 24;
        pspm_explore = 24;
    }

    let norm_ripa_clamp = {
        if raw_clamp < DEFAULT_FACT_RIPA_CLAMP_RAD {
            let auth_ratio = branchless_clamp_f32(raw_clamp * INV_FACT_RIPA_CLAMP_RAD, 0.10, 1.0);
            let auth_u64 = ((auth_ratio * 128.0 + 0.5) as u8).min(128) as u64;
            if auth_u64 < 128 {
                let bias_auth = (128u64.wrapping_sub(auth_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                if blended_uni {
                    blended_w = [swar_q7_scale_le128_with_bias_x8(
                        blended_w[0],
                        auth_u64,
                        bias_auth,
                    ); NUM_SWARM_MICRO_CELLS];
                } else {
                    blended_w = [
                        swar_q7_scale_le128_with_bias_x8(blended_w[0], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[1], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[2], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[3], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[4], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[5], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[6], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[7], auth_u64, bias_auth),
                    ];
                }
            }
        }
        DEFAULT_RIPA_CLAMP_MAX_RAD
    };

    if qdps_step_u8 > 1 {
        let (fw, fu) = swar_q7_qdps_filter_words_with_uniform_flag_64(
            &blended_w,
            blended_uni,
            qdps_step_u8,
        );
        blended_w = fw;
        blended_uni = fu;
    }

    let mut rep_sev = 0.0_f32;
    if let Some(guard) = guard_opt {
        const ZERO_PHASES_64: [gcso_q7_t; MAX_ATTENTION_HEADS] = [0i8; MAX_ATTENTION_HEADS];
        prov_bits |= guard.causal_provenance_bits() | CAUSAL_AXIS_CONTRADICTION_GUARD;
        let g_sev = if guard.requires_phase_slice_for_repulsion() {
            let guard_phases = words_to_q7_array_64(&blended_w);
            guard.evaluate_repulsion_severity_from_words(&blended_w, &guard_phases)
        } else {
            guard.evaluate_repulsion_severity_from_words(&blended_w, &ZERO_PHASES_64)
        };
        rep_sev = branchless_clamp_f32(
            branchless_max_f32(g_sev, guard.causal_anomaly_severity()),
            0.0,
            1.0,
        );
    }
    let repulsion_triggered =
        (rep_sev > 0.50) | ((prov_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
    if repulsion_triggered {
        prov_bits |= CAUSAL_CAUSE_COHOMOLOGY_REPULSION;
        avx_precomputed_tag = None;
        if blended_uni {
            blended_w = [swar_q7_neg_sat_x8(blended_w[0]); NUM_SWARM_MICRO_CELLS];
        } else {
            blended_w = [
                swar_q7_neg_sat_x8(blended_w[0]),
                swar_q7_neg_sat_x8(blended_w[1]),
                swar_q7_neg_sat_x8(blended_w[2]),
                swar_q7_neg_sat_x8(blended_w[3]),
                swar_q7_neg_sat_x8(blended_w[4]),
                swar_q7_neg_sat_x8(blended_w[5]),
                swar_q7_neg_sat_x8(blended_w[6]),
                swar_q7_neg_sat_x8(blended_w[7]),
            ];
        }
        dom_slot = ATTRACTOR_SLOT_REPULSION;
        rep_sev = branchless_max_f32(rep_sev, 0.85);
    }
    let phase_deltas = words_to_q7_array_64(&blended_w);

    let mut action_flags = if repulsion_triggered {
        ACTION_PRESET_REPULSION_ESCAPE
    } else if force_skip {
        ACTION_PRESET_INERTIAL_SKIP
    } else if has_fact | has_persona {
        ACTION_MODE_DPSR_STEER | ACTION_MODE_PSPM_ROUTE
            | branchless_select_u8(eff_energy >= 2.0, ACTION_MODE_SRL_INJECT, 0)
    } else {
        ACTION_PRESET_EXPRESS_PHASE
    };
    if force_skip & (!repulsion_triggered) {
        action_flags |= ACTION_MODE_LAYER_SKIP;
    }

    let dominant_cause = if repulsion_triggered {
        ROOT_CAUSE_COHOMOLOGY_REPULSION
    } else if collision {
        ROOT_CAUSE_PERSONA_FACT_COLLISION
    } else if (prov_bits & CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK) != 0 {
        ROOT_CAUSE_FACT_RETRIEVAL_MISS
    } else if (prov_bits & CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE) != 0 {
        ROOT_CAUSE_SYSTEM_QOS_THROTTLE
    } else {
        ROOT_CAUSE_NOMINAL
    };

    let inv_temp = if repulsion_triggered {
        0.75
    } else if has_fact | has_persona {
        branchless_phase_coherence_inv_temp_scale(fp_coh, 0.65)
    } else {
        1.0
    };

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if !blended_uni && avx_precomputed_tag.is_none() {
        // SAFETY: `target_feature = "avx2"` is enabled at compile time.
        let (v0, v1) = unsafe { super::branchless::avx2_words64_to_vec64(&blended_w) };
        avx_precomputed_tag =
            Some(unsafe { super::primitives::avx2_q7_mean_i8_from_vec64(v0, v1) });
    }
    let tag_q7 = match avx_precomputed_tag {
        Some(t) => t,
        None => swar_q7_mean_i8_from_words_with_uniform_flag_64(&blended_w, blended_uni),
    };
    let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
    let tagged_ptr = gcso_tagged_ptr_t::pack(zimms_addr, packed_tag, tag_q7).raw();
    let logic_gain_q7 = ((fact_gain_q7 as u16 + pers_gain_q7 as u16) >> 1) as u8;

    (
        GatewayFrame {
            tagged_ptr,
            zimms_section_offset48: zimms_addr,
            target_anchor_id: dom_anchor_id,
            effective_energy: eff_energy,
            ripa_clamp_rad: norm_ripa_clamp,
            qdps_min_step_rad: qdps_rad,
            effective_inv_temp: inv_temp,
            fact_persona_coherence: fp_coh,
            repulsion_severity: branchless_max_f32(rep_sev, sys_sev),
            provenance_bits: prov_bits,
            dominant_cause_code: dominant_cause,
            dominant_slot: dom_slot,
            action_mode_flags: action_flags,
            active_swarm_cell_mask: active_swarm_mask,
            qdps_min_step_q7: qdps_step_u8,
            pspm_fact_heads: pspm_fact,
            pspm_logic_heads: pspm_logic,
            pspm_explore_heads: pspm_explore,
            pspm_fact_gain_q7: fact_gain_q7,
            pspm_logic_gain_q7: logic_gain_q7,
            pspm_explore_gain_q7: pers_gain_q7,
            mean_phase_tag_q7: tag_q7,
            active_head_count: MAX_ATTENTION_HEADS as u16,
            frame_flags: packed_tag as u16,
            phase_deltas,
        },
        blended_w,
    )
}

/// Synthesizes a 128-byte [`GatewayFrame`] from optional **Origin 1 (`SystemConstraintSignal`)**
/// and **Origin 2 (`ObjectiveFactSignal`, `SubjectivePersonaSignal`, `ContradictionGuardSignal`)** components.
#[inline(always)]
#[must_use]
pub fn synthesize_multi_origin_gateway_frame<S, F, P, G>(
    sys_opt: Option<&S>,
    fact_opt: Option<&F>,
    persona_opt: Option<&P>,
    guard_opt: Option<&G>,
) -> GatewayFrame
where
    S: SystemConstraintSignal + ?Sized,
    F: ObjectiveFactSignal + ?Sized,
    P: SubjectivePersonaSignal + ?Sized,
    G: ContradictionGuardSignal + ?Sized,
{
    synthesize_multi_origin_gateway_frame_with_words(sys_opt, fact_opt, persona_opt, guard_opt).0
}

// ===================================================================
// 7. Blanket `ManifoldGatewaySignal` Implementations for Wrappers, Tuples & Ensembles
// ===================================================================

impl<T: BranchlessKernel> ManifoldGatewaySignal for SystemConstraint<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            Self,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(Some(self), None, None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl ManifoldGatewaySignal for gcso_edbc_state_t {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            Self,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(Some(self), None, None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl ManifoldGatewaySignal for gcso_config_t {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            Self,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(Some(self), None, None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<T: RawPhaseProjectable> ManifoldGatewaySignal for ObjectiveFact<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            SystemConstraint<()>,
            Self,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(None, Some(self), None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<T: RawPhaseProjectable> ManifoldGatewaySignal for SubjectivePersona<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            SystemConstraint<()>,
            ObjectiveFact<[u8; 0]>,
            Self,
            ContradictionGuard<()>,
        >(None, None, Some(self), None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<B: BranchlessKernel> ManifoldGatewaySignal for ContradictionGuard<B>
where
    Self: ContradictionGuardSignal,
{
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            SystemConstraint<()>,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            Self,
        >(None, None, None, Some(self))
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<F: ObjectiveFactSignal, P: SubjectivePersonaSignal> CausalProvenanceSignal for (F, P) {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.0.causal_provenance_bits() | self.1.causal_provenance_bits()
            | CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(self.0.causal_anomaly_severity(), self.1.causal_anomaly_severity())
    }
}

impl<F: ObjectiveFactSignal, P: SubjectivePersonaSignal> ManifoldGatewaySignal for (F, P) {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            SystemConstraint<()>,
            F,
            P,
            ContradictionGuard<()>,
        >(None, Some(&self.0), Some(&self.1), None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<S: SystemConstraintSignal, F: ObjectiveFactSignal, P: SubjectivePersonaSignal>
    CausalProvenanceSignal for (S, F, P)
{
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.0.causal_provenance_bits() | self.1.causal_provenance_bits()
            | self.2.causal_provenance_bits() | CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(
            self.0.causal_anomaly_severity(),
            branchless_max_f32(
                self.1.causal_anomaly_severity(),
                self.2.causal_anomaly_severity(),
            ),
        )
    }
}

impl<S: SystemConstraintSignal, F: ObjectiveFactSignal, P: SubjectivePersonaSignal>
    ManifoldGatewaySignal for (S, F, P)
{
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<S, F, P, ContradictionGuard<()>>(
            Some(&self.0),
            Some(&self.1),
            Some(&self.2),
            None,
        )
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<
        S: SystemConstraintSignal,
        F: ObjectiveFactSignal,
        P: SubjectivePersonaSignal,
        G: ContradictionGuardSignal,
    > CausalProvenanceSignal for (S, F, P, G)
{
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.0.causal_provenance_bits() | self.1.causal_provenance_bits()
            | self.2.causal_provenance_bits() | self.3.causal_provenance_bits()
            | CAUSAL_AXIS_SYSTEM_CONSTRAINT | CAUSAL_AXIS_OBJECTIVE_FACT
            | CAUSAL_AXIS_SUBJECTIVE_PERSONA | CAUSAL_AXIS_CONTRADICTION_GUARD
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(
            branchless_max_f32(
                self.0.causal_anomaly_severity(),
                self.1.causal_anomaly_severity(),
            ),
            branchless_max_f32(
                self.2.causal_anomaly_severity(),
                self.3.causal_anomaly_severity(),
            ),
        )
    }
}

impl<
        S: SystemConstraintSignal,
        F: ObjectiveFactSignal,
        P: SubjectivePersonaSignal,
        G: ContradictionGuardSignal,
    > ManifoldGatewaySignal for (S, F, P, G)
{
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<S, F, P, G>(
            Some(&self.0),
            Some(&self.1),
            Some(&self.2),
            Some(&self.3),
        )
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<T: ManifoldGatewaySignal> CausalProvenanceSignal for Option<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        match self {
            Some(inner) => inner.causal_provenance_bits(),
            None => 0,
        }
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        match self {
            Some(inner) => inner.causal_anomaly_severity(),
            None => 0.0,
        }
    }
}

impl<T: ManifoldGatewaySignal> ManifoldGatewaySignal for Option<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        match self {
            Some(inner) => inner.synthesize_gateway_frame_with_words(),
            None => (GatewayFrame::default(), [0u64; NUM_SWARM_MICRO_CELLS]),
        }
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        match self {
            Some(inner) => inner.has_active_gateway_ensemble_signal(),
            None => false,
        }
    }
}

impl<T: ManifoldGatewaySignal> CausalProvenanceSignal for [T] {
    #[inline]
    fn causal_provenance_bits(&self) -> u32 {
        let mut bits = 0u32;
        let mut i = 0usize;
        while i < self.len() {
            bits |= self[i].causal_provenance_bits();
            i += 1;
        }
        bits
    }
    #[inline]
    fn causal_anomaly_severity(&self) -> f32 {
        let mut max_sev = 0.0_f32;
        let mut i = 0usize;
        while i < self.len() {
            max_sev = branchless_max_f32(max_sev, self[i].causal_anomaly_severity());
            i += 1;
        }
        max_sev
    }
}

impl<T: ManifoldGatewaySignal> ManifoldGatewaySignal for [T] {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        T::synthesize_gateway_slice_with_words(self)
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        T::synthesize_gateway_slice_with_words(self).0
    }
}

impl<T: ManifoldGatewaySignal, const N: usize> CausalProvenanceSignal for [T; N] {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.as_slice().causal_provenance_bits()
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.as_slice().causal_anomaly_severity()
    }
}

impl<T: ManifoldGatewaySignal, const N: usize> ManifoldGatewaySignal for [T; N] {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        self.as_slice().synthesize_gateway_frame_with_words()
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.as_slice().synthesize_gateway_frame_with_words().0
    }
}

// ===================================================================
// 8. Re-Exports of `.gcso` Crystallized Attractor Record & 4-Section ZIMMS View
// ===================================================================

pub use crate::crystal::CrystallizedAttractorRecord;
pub use crate::storage::GcsoFourSectionView;

// ===================================================================
// 9. Blanket Extension Traits on `PhaseSteering + PspmRouter` & `ActionHub`
// ===================================================================

#[inline]
pub(crate) fn accumulate_trail_and_steer_gateway_words_core(
    frame: &GatewayFrame,
    frm_w: &[u64; NUM_SWARM_MICRO_CELLS],
    trail_inout: &mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    head_dim: usize,
    num_heads: usize,
    query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(bool, bool, bool)> {
    let frm_nonzero = swar_words_nonzero_64(frm_w);
    let next_w = if frm_nonzero {
        *trail_phase_words_mut(trail_inout) = *frm_w;
        *frm_w
    } else {
        *trail_phase_words_ref(trail_inout)
    };
    trail_inout.attractor_pull_force = branchless_min_f32(
        branchless_max_f32(trail_inout.attractor_pull_force, frame.effective_energy),
        MAX_ATTRACTOR_PULL_FORCE,
    );

    let layer_skipped = (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0;
    let mut steered = false;
    let mut pspm_routed = false;

    if layer_skipped {
        trail_inout.flags |= TRAIL_FLAG_INERTIAL_SKIP_ACTIVE;
    } else if query_tensor.is_some() | key_tensor.is_some() {
        steered = true;
        let swar_words = if frame.qdps_min_step_q7 <= 1 {
            next_w
        } else if frame.qdps_min_step_q7 > 127 {
            [0u64; NUM_SWARM_MICRO_CELLS]
        } else {
            swar_q7_qdps_filter_words_64(&next_w, frame.qdps_min_step_q7)
        };
        let clean_damping =
            branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);

        let has_nonzero_words = if frm_nonzero & (frame.qdps_min_step_q7 <= 1) {
            true
        } else {
            swar_words_nonzero_64(&swar_words)
        };
        if (frame.action_mode_flags & ACTION_MODE_PSPM_ROUTE) != 0
            && frame.active_swarm_cell_mask == 0xFF
        {
            let is_def_clamp = (frame.ripa_clamp_rad.to_bits()
                == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
                & frame.has_unit_pspm_gains()
                & (num_heads > 0)
                & (num_heads <= MAX_ATTENTION_HEADS);
            match (query_tensor, key_tensor) {
                (Some(q), Some(k)) => {
                    if has_nonzero_words {
                        if is_def_clamp {
                            let rot_dim = frame.resolve_autonomous_rot_dim(head_dim, 0.0, 0.0);
                            rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
                                q.as_mut_slice(),
                                k.as_mut_slice(),
                                &swar_words,
                                num_heads,
                                head_dim,
                                rot_dim,
                            );
                        } else {
                            let pspm_cfg = frame.to_pspm_config_for_heads(num_heads);
                            dispatch_fused_ripa_and_pspm_words_core(
                                q,
                                &swar_words,
                                frame.ripa_clamp_rad,
                                &pspm_cfg,
                                &swar_words,
                                &swar_words,
                                &swar_words,
                                frame.ripa_clamp_rad,
                                0,
                                head_dim,
                                true,
                            )?;
                            steer_from_swar_words_core_safe(
                                &swar_words,
                                0xFF,
                                clean_damping,
                                head_dim,
                                num_heads,
                                false,
                                frame.ripa_clamp_rad,
                                k,
                                None,
                            );
                        }
                    }
                    pspm_routed = true;
                }
                (Some(q), None) => {
                    if has_nonzero_words {
                        if is_def_clamp {
                            let rot_dim = frame.resolve_autonomous_rot_dim(head_dim, 0.0, 0.0);
                            rotate_fused_self_ripa_and_pspm_default_words(
                                q.as_mut_slice(),
                                &swar_words,
                                num_heads,
                                head_dim,
                                rot_dim,
                                true,
                            );
                        } else {
                            let pspm_cfg = frame.to_pspm_config_for_heads(num_heads);
                            dispatch_fused_ripa_and_pspm_words_core(
                                q,
                                &swar_words,
                                frame.ripa_clamp_rad,
                                &pspm_cfg,
                                &swar_words,
                                &swar_words,
                                &swar_words,
                                frame.ripa_clamp_rad,
                                0,
                                head_dim,
                                true,
                            )?;
                        }
                    }
                    pspm_routed = true;
                }
                (None, Some(k)) => {
                    if has_nonzero_words {
                        steer_from_swar_words_core_safe(
                            &swar_words,
                            0xFF,
                            clean_damping,
                            head_dim,
                            num_heads,
                            false,
                            frame.ripa_clamp_rad,
                            k,
                            None,
                        );
                    }
                }
                (None, None) => {}
            }
        } else if has_nonzero_words {
            match (query_tensor, key_tensor) {
                (Some(q), k_opt) => {
                    steer_from_swar_words_core_safe(
                        &swar_words,
                        frame.active_swarm_cell_mask,
                        clean_damping,
                        head_dim,
                        num_heads,
                        false,
                        frame.ripa_clamp_rad,
                        q,
                        k_opt,
                    );
                }
                (None, Some(k)) => {
                    steer_from_swar_words_core_safe(
                        &swar_words,
                        frame.active_swarm_cell_mask,
                        clean_damping,
                        head_dim,
                        num_heads,
                        false,
                        frame.ripa_clamp_rad,
                        k,
                        None,
                    );
                }
                (None, None) => {}
            }
        }
    }

    if let Some(logits) = logits_opt {
        if !logits.is_empty() {
            scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
        }
    }

    Ok((layer_skipped, steered, pspm_routed))
}

/// Blanket-implemented closed-loop Gateway & Crystallized Ensemble steering extension trait for
/// any type implementing [`PhaseSteering`] + [`PspmRouter`].
pub trait GatewayPhaseSteeringExt: PhaseSteering + PspmRouter {
    /// Synthesizes any [`ManifoldGatewaySignal`] into a 128-byte [`GatewayFrame`] and register-resident
    /// phase words, updates `trail_inout`, and executes $SO(2)$-fused RIPA + PSPM sub-head steering.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if head dimensions, radian clamps, or tensor slices
    /// violate layout or `GCSO_RESTRICT` disjointness invariants.
    #[inline(always)]
    fn steer_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail_inout: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        let report = frame.to_causal_report();
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS
            || frame.qdps_min_step_rad < 0.0
            || !branchless_is_finite_f32(frame.qdps_min_step_rad)
            || frame.ripa_clamp_rad <= 0.0
            || !branchless_is_finite_f32(frame.ripa_clamp_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        validate_optional_qk_tensors(Some(&*query_tensor), key_tensor.as_deref(), expected_len)?;

        trail_inout.prev_ptr = trail_inout.current_ptr;
        trail_inout.current_ptr = frame.tagged_ptr;
        trail_inout.step_count = trail_inout.step_count.wrapping_add(1);
        if frame.target_anchor_id != 0 {
            trail_inout.target_anchor_id = frame.target_anchor_id;
            trail_inout.cluster_id = frame.target_anchor_id;
        } else {
            trail_inout.cluster_id = u32::from(frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK);
        }

        let _ = accumulate_trail_and_steer_gateway_words_core(
            &frame,
            &frm_w,
            trail_inout,
            unresolved_damping_u7,
            head_dim,
            num_heads,
            Some(query_tensor),
            key_tensor,
            logits_opt,
        )?;
        Ok((frame, report))
    }

    /// Folds a zero-copy slice of `.gcso` [`CrystallizedAttractorRecord`]s (`records`) onto
    /// `trail_inout` weighted by **Phase-Proximity Gravity** directly in 8 `u64` SWAR registers.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if head dimensions, `min_step_rad`, or tensor slices
    /// violate layout or `GCSO_RESTRICT` disjointness invariants.
    #[inline(always)]
    fn steer_crystallized_ensemble_zero_copy_safe(
        &self,
        records: &[CrystallizedAttractorRecord],
        trail_inout: &mut gcso_pointer_trail_t,
        min_step_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, u32, f32)> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        validate_optional_qk_tensors(Some(&*query_tensor), key_tensor.as_deref(), expected_len)?;

        let init_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(trail_inout.attractor_pull_force));
        let (folded_w, tot_pull, dom_id, dom_clamp) =
            CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64(
                records,
                &mut trail_inout.accumulated_phase_delta,
                init_pull,
            );

        if dom_id != 0 {
            trail_inout.target_anchor_id = dom_id;
            trail_inout.cluster_id = dom_id;
        }
        trail_inout.attractor_pull_force = branchless_min_f32(tot_pull, MAX_ATTRACTOR_PULL_FORCE);
        trail_inout.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | TRAIL_FLAG_SYNERGY_RESONANCE;

        let eff_ripa = branchless_select_f32(
            branchless_is_finite_f32(dom_clamp) & (dom_clamp > 0.0),
            dom_clamp,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        let min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let qdps_words = if min_step_q7 <= 1 {
            folded_w
        } else if min_step_q7 > 127 {
            [0u64; NUM_SWARM_MICRO_CELLS]
        } else {
            swar_q7_qdps_filter_words_64(&folded_w, min_step_q7)
        };
        let is_default_ripa = branchless_abs_f32(eff_ripa - DEFAULT_RIPA_CLAMP_MAX_RAD) <= 1.0e-5;
        let swar_words = if is_default_ripa {
            qdps_words
        } else {
            let ratio_u8 =
                (branchless_clamp_f32(eff_ripa * (1.0 / DEFAULT_RIPA_CLAMP_MAX_RAD), 0.10, 1.0)
                    * 128.0
                    + 0.5) as u8;
            let s_u64 = ratio_u8.min(128) as u64;
            let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            [
                swar_q7_scale_le128_with_bias_x8(qdps_words[0], s_u64, bias_restore),
                swar_q7_scale_le128_with_bias_x8(qdps_words[1], s_u64, bias_restore),
                swar_q7_scale_le128_with_bias_x8(qdps_words[2], s_u64, bias_restore),
                swar_q7_scale_le128_with_bias_x8(qdps_words[3], s_u64, bias_restore),
                swar_q7_scale_le128_with_bias_x8(qdps_words[4], s_u64, bias_restore),
                swar_q7_scale_le128_with_bias_x8(qdps_words[5], s_u64, bias_restore),
                swar_q7_scale_le128_with_bias_x8(qdps_words[6], s_u64, bias_restore),
                swar_q7_scale_le128_with_bias_x8(qdps_words[7], s_u64, bias_restore),
            ]
        };

        if swar_words_nonzero_64(&swar_words) {
            steer_from_swar_words_core_safe(
                &swar_words,
                0xFF,
                128,
                head_dim,
                num_heads,
                false,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                query_tensor,
                key_tensor,
            );
        }

        Ok((tot_pull, dom_id, eff_ripa))
    }
}

impl<T: PhaseSteering + PspmRouter + ?Sized> GatewayPhaseSteeringExt for T {}

/// Blanket-implemented extension trait equipping any [`ActionHub`] with single-call
/// [`ManifoldGatewaySignal`] ingestion, Online Toroidal $k$-Means / DP-Means macro clustering,
/// online basin consolidation, and `.gcso` crystallization.
pub trait ActionHubGatewayExt: ActionHub {
    /// Internal register-resident helper that ingests `signal` via `synthesize_gateway_frame_with_words`,
    /// steps `trail_out` and `daes_slot`, and steers `query_tensor` / `key_tensor` directly from registers.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`ActionHub`].
    #[inline(always)]
    fn ingest_gateway_and_steer_with_words_core<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        mut query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        srl_activation_and_desc: Option<(&AlignedSlice32<'_, f32>, &gcso_srl_descriptor_t)>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        [u64; NUM_SWARM_MICRO_CELLS],
    )> {
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        let report = frame.to_causal_report();
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            frame.qdps_min_step_rad,
            frame.ripa_clamp_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;

        let (bypass_hit, effective_ptr) =
            self.resolve_or_pass_tagged_ptr(daes_slot, gcso_tagged_ptr_t(frame.tagged_ptr));
        let frm_nonzero = swar_words_nonzero_64(&frm_w);
        if frm_nonzero {
            *trail_phase_words_mut(trail_out) = frm_w;
        }
        if frame.target_anchor_id != 0 {
            trail_out.target_anchor_id = frame.target_anchor_id;
            trail_out.cluster_id = frame.target_anchor_id;
        } else {
            trail_out.cluster_id = frame.dominant_slot as u32;
        }
        trail_out.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_out.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        // When `frm_w` is non-zero, pass `effective_ptr.with_phase(0)` to `step_pointer` so
        // `step_pointer` does not double-accumulate `mean_phase_tag_q7` onto `frm_w` and so
        // `ActionHubEngine::trails[slot_idx]` receives the exact `GatewayFrame` phase profile.
        let step_ptr = if frm_nonzero {
            effective_ptr.with_phase(0)
        } else {
            effective_ptr
        };
        self.step_pointer(step_ptr.raw(), trail_out)?;
        trail_out.current_ptr = effective_ptr.raw();

        let is_repulsed = report.is_cohomology_repulsion()
            | ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0);
        let mut metrics = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(is_repulsed, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(
                (frame.provenance_bits
                    & (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA))
                    == (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA),
                DAES_METRIC_SLOT_SYNERGY_ACTIVE,
                0,
            );

        let (layer_skipped, steered, pspm_routed) = accumulate_trail_and_steer_gateway_words_core(
            &frame,
            &frm_w,
            trail_out,
            128,
            head_dim,
            num_heads,
            query_tensor.as_deref_mut(),
            key_tensor,
            logits_opt,
        )?;

        if layer_skipped {
            metrics |= DAES_METRIC_LAYER_SKIP_COMPENSATED;
        } else {
            metrics |= branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
                | branchless_select_u8(pspm_routed, DAES_METRIC_PSPM_ROUTED, 0);
            if let (Some(q), Some((x_in, desc))) = (query_tensor, srl_activation_and_desc) {
                if (frame.action_mode_flags & ACTION_MODE_SRL_INJECT) != 0 {
                    self.eval_rank1_fused(q, x_in, desc)?;
                    metrics |= DAES_METRIC_SRL_INJECTED;
                }
            }
        }

        let _ = self.telemetry_push(daes_slot, metrics);
        Ok((frame, report, frm_w))
    }

    /// Ingests any [`ManifoldGatewaySignal`] into `self`, steps `trail_out` and `daes_slot`,
    /// and executes $SO(2)$-fused RIPA + PSPM sub-head steering.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`ActionHub`].
    #[inline(always)]
    fn ingest_gateway_and_steer_safe<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        srl_activation_and_desc: Option<(&AlignedSlice32<'_, f32>, &gcso_srl_descriptor_t)>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
        let (frame, report, _frm_w) = self.ingest_gateway_and_steer_with_words_core(
            signal,
            daes_slot,
            trail_out,
            query_tensor,
            key_tensor,
            logits_opt,
            srl_activation_and_desc,
        )?;
        Ok((frame, report))
    }

    /// Ingests `signal`, resolves the 8-Cell SWAR Micro-Swarm on `trail_out`, and absorbs the
    /// resulting [`GatewayFrame`] into `crystal_record`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`Self::ingest_gateway_and_steer_safe`].
    #[inline(always)]
    fn ingest_gateway_swarm_and_crystallize_safe<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        crystal_record: &mut CrystallizedAttractorRecord,
        step_gain_u7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        SwarmResolutionReport,
        f32,
    )> {
        let (frame, report, init_frm_w) = self.ingest_gateway_and_steer_with_words_core(
            signal,
            daes_slot,
            trail_out,
            query_tensor,
            key_tensor,
            logits_opt,
            None,
        )?;
        let cur_trail_w = if swar_words_nonzero_64(&init_frm_w) {
            init_frm_w
        } else {
            *trail_phase_words_ref(trail_out)
        };
        let (swarm_report, frm_w) = trail_out.try_resolve_action_in_swarm_with_cur_words(
            gcso_tagged_ptr_t(frame.tagged_ptr),
            &cur_trail_w,
            &init_frm_w,
            false,
            frame.active_swarm_cell_mask,
            step_gain_u7,
            frame.qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );
        let grav_mass = crystal_record.absorb_gateway_frame_with_words(&frame, &frm_w);
        Ok((frame, report, swarm_report, grav_mass))
    }

    /// Executes the complete closed-loop **2-Origin Gateway $\to$ Classical Online Toroidal
    /// $k$-Means / DP-Means Macro Clustering (`centroids`) $\to$ `ActionHub` / `RoPE` / `PSPM`
    /// Steering** pipeline in a single zero-allocation call.
    ///
    /// # Errors
    /// Propagates any error from [`Self::ingest_gateway_and_steer_safe`].
    #[inline(always)]
    fn ingest_gateway_kmeans_and_steer_safe<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport, usize, u32, f32, bool)> {
        let (frame, report, frm_w) = self.ingest_gateway_and_steer_with_words_core(
            signal,
            daes_slot,
            trail_out,
            query_tensor,
            key_tensor,
            logits_opt,
            None,
        )?;
        let (cluster_idx, anchor_id, _coh, mass, spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                &frame,
                &frm_w,
                split_coherence_thresh,
                Some(trail_out),
            );
        Ok((frame, report, cluster_idx, anchor_id, mass, spawned))
    }

    /// Executes the complete closed-loop **Gateway $\to$ `ActionHub` Steering $\to$ Online Toroidal
    /// $k$-Means / DP-Means E+M Step $\to$ Autonomous Basin Consolidation & Evaporative Pruning**
    /// (`merge_and_prune_centroids_64`) in a single zero-allocation call.
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, assigned_cluster_idx, assigned_anchor_id, updated_gravity_mass, spawned_new_cluster, merged_clusters, pruned_clusters)`.
    ///
    /// # Errors
    /// Propagates any error from [`Self::ingest_gateway_kmeans_and_steer_safe`].
    #[inline]
    fn ingest_gateway_kmeans_consolidate_and_steer_safe<S: ManifoldGatewaySignal + ?Sized>(
        &mut self,
        signal: &S,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        merge_coherence_thresh: f32,
        min_gravity_mass: f32,
        evaporative_decay_factor: f32,
        daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        usize,
        u32,
        f32,
        bool,
        u32,
        u32,
    )> {
        let (frame, report, cluster_idx, anchor_id, mass, spawned) = self
            .ingest_gateway_kmeans_and_steer_safe(
                signal,
                centroids,
                split_coherence_thresh,
                daes_slot,
                trail_out,
                query_tensor,
                key_tensor,
                logits_opt,
            )?;
        let (merged, pruned) = CrystallizedAttractorRecord::merge_and_prune_centroids_64(
            centroids,
            merge_coherence_thresh,
            min_gravity_mass,
            evaporative_decay_factor,
        );
        Ok((
            frame,
            report,
            cluster_idx,
            anchor_id,
            mass,
            spawned,
            merged,
            pruned,
        ))
    }
}

impl<T: ActionHub + ?Sized> ActionHubGatewayExt for T {}

impl CausalProvenanceSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0)
    }
}

impl CognitiveGroundingSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_profile
    }
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        Some(&self.phase_profile)
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        *self.phase_words_ref()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.phase_gravity_mass
    }
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 {
        self.zimms_section_offset48
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl ObjectiveFactSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 {
        self.ripa_clamp_rad
    }
}

impl SubjectivePersonaSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 {
        self.ripa_clamp_rad
    }
}

impl ManifoldGatewaySignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > 1.0e-12);
        if !is_active {
            let packed_tag =
                pack_action_slot_tag(self.dominant_slot, ACTION_PRESET_PASSIVE_SIDECAR);
            return (
                GatewayFrame {
                    tagged_ptr: gcso_tagged_ptr_t::pack(self.zimms_section_offset48, packed_tag, 0)
                        .raw(),
                    zimms_section_offset48: self.zimms_section_offset48,
                    target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot,
                    action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask,
                    dominant_cause_code: self.dominant_cause_code,
                    frame_flags: packed_tag as u16,
                    ..GatewayFrame::default()
                },
                [0u64; NUM_SWARM_MICRO_CELLS],
            );
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let tag_q7 = fast_mean_q7_tag_from_words_64(&w);
        let eff_action = self.action_mode_flags;
        let packed_tag = pack_action_slot_tag(self.dominant_slot, eff_action);
        let tagged_ptr =
            gcso_tagged_ptr_t::pack(self.zimms_section_offset48, packed_tag, tag_q7).raw();
        let prov_bits = self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let pspm_fact =
            branchless_select_u8(is_fact_slot, 36, branchless_select_u8(is_persona_slot, 16, 24));
        let pspm_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let pspm_explore =
            branchless_select_u8(is_fact_slot, 8, branchless_select_u8(is_persona_slot, 24, 16));
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (
            GatewayFrame {
                tagged_ptr,
                zimms_section_offset48: self.zimms_section_offset48,
                target_anchor_id: self.anchor_id,
                effective_energy: branchless_min_f32(
                    self.phase_gravity_mass,
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
                effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar,
                repulsion_severity: 0.0,
                provenance_bits: prov_bits,
                dominant_cause_code: self.dominant_cause_code,
                dominant_slot: self.dominant_slot,
                action_mode_flags: eff_action,
                active_swarm_cell_mask: self.active_swarm_mask,
                qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
                pspm_fact_heads: pspm_fact,
                pspm_logic_heads: pspm_logic,
                pspm_explore_heads: pspm_explore,
                pspm_fact_gain_q7: fact_gain_q7,
                pspm_logic_gain_q7: logic_gain_q7,
                pspm_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas: self.phase_profile,
            },
            w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > 1.0e-12);
        if !is_active {
            return (
                GatewayFrame {
                    zimms_section_offset48: self.zimms_section_offset48,
                    target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot,
                    action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask,
                    dominant_cause_code: self.dominant_cause_code,
                    ..GatewayFrame::default()
                },
                [0u64; NUM_SWARM_MICRO_CELLS],
            );
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let prov_bits = self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);
        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let pspm_fact =
            branchless_select_u8(is_fact_slot, 36, branchless_select_u8(is_persona_slot, 16, 24));
        let pspm_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let pspm_explore =
            branchless_select_u8(is_fact_slot, 8, branchless_select_u8(is_persona_slot, 24, 16));
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (
            GatewayFrame {
                tagged_ptr: 0,
                zimms_section_offset48: self.zimms_section_offset48,
                target_anchor_id: self.anchor_id,
                effective_energy: branchless_min_f32(
                    self.phase_gravity_mass,
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
                effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar,
                repulsion_severity: 0.0,
                provenance_bits: prov_bits,
                dominant_cause_code: self.dominant_cause_code,
                dominant_slot: self.dominant_slot,
                action_mode_flags: self.action_mode_flags,
                active_swarm_cell_mask: self.active_swarm_mask,
                qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
                pspm_fact_heads: pspm_fact,
                pspm_logic_heads: pspm_logic,
                pspm_explore_heads: pspm_explore,
                pspm_fact_gain_q7: fact_gain_q7,
                pspm_logic_gain_q7: logic_gain_q7,
                pspm_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: 0,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: 0,
                phase_deltas: [0i8; MAX_ATTENTION_HEADS],
            },
            w,
        )
    }

    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        (self.sample_count > 0) & (self.phase_gravity_mass > 1.0e-12)
    }

    #[inline]
    fn synthesize_gateway_slice_with_words(
        records: &[Self],
    ) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let len = records.len();
        if len == 0 {
            return (GatewayFrame::default(), [0u64; NUM_SWARM_MICRO_CELLS]);
        }
        if len == 1 {
            return records[0].synthesize_gateway_frame_with_words();
        }

        let mut acc_w = [0u64; NUM_SWARM_MICRO_CELLS];
        let mut acc_is_uniform = true;
        let mut tot_e = 0.0_f32;
        let mut best_e = 0.0_f32;
        let mut weighted_inv_temp = 0.0_f32;
        let mut weighted_coh = 0.0_f32;
        let mut dom_idx = 0usize;
        let mut dom_zimms_addr = 0u64;
        let mut action_flags = ACTION_PRESET_PASSIVE_SIDECAR;
        let mut swarm_mask = 0xFFu8;
        let mut prov_bits = 0u32;
        let mut dom_cause = ROOT_CAUSE_NOMINAL;
        let mut has_active = false;

        let mut i = 0usize;
        while i < len {
            let rec = &records[i];
            if (rec.sample_count > 0) & (rec.phase_gravity_mass > 1.0e-12) {
                let rec_w = rec.phase_words_ref();
                let rec_is_uniform = if !has_active || acc_is_uniform {
                    is_word_array_uniform_64(rec_w)
                } else {
                    false
                };
                let f_e = branchless_min_f32(
                    branchless_flush_subnormal_f32(branchless_relu_f32(rec.phase_gravity_mass)),
                    MAX_ATTRACTOR_PULL_FORCE,
                );
                let is_mature = (rec.sample_count >= 2) & (rec.resultant_r_bar >= 0.45);
                let rec_prov = rec.provenance_bits
                    | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

                if !has_active {
                    acc_w = *rec_w;
                    acc_is_uniform = rec_is_uniform;
                    tot_e = f_e;
                    best_e = f_e;
                    weighted_inv_temp = f_e * rec.effective_inv_temp;
                    weighted_coh = f_e * rec.resultant_r_bar;
                    dom_idx = i;
                    dom_zimms_addr = rec.zimms_section_offset48;
                    action_flags = rec.action_mode_flags;
                    swarm_mask = rec.active_swarm_mask;
                    prov_bits = rec_prov;
                    dom_cause = rec.dominant_cause_code;
                    has_active = true;
                } else if f_e > 1.0e-12 {
                    let new_tot = tot_e + f_e;
                    if !swar_words_equal_64(&acc_w, rec_w) {
                        let safe_tot = branchless_select_f32(new_tot > 1.0e-12, new_tot, 1.0);
                        let w_ratio = branchless_clamp_f32(
                            f_e * fast_inv_pos_normal_f32(safe_tot),
                            0.0,
                            1.0,
                        );
                        let s_u8 = ((w_ratio * 128.0 + 0.5) as u8).min(128);
                        let s_u64 = s_u8 as u64;
                        let bias_restore =
                            (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let (next_acc, next_uni) = swar_slerp_words_with_uniform_flag_64(
                            &acc_w,
                            acc_is_uniform,
                            rec_w,
                            rec_is_uniform,
                            s_u64,
                            bias_restore,
                        );
                        acc_w = next_acc;
                        acc_is_uniform = next_uni;
                    }
                    weighted_inv_temp += f_e * rec.effective_inv_temp;
                    weighted_coh += f_e * rec.resultant_r_bar;
                    let take_dom = f_e > best_e;
                    best_e = branchless_select_f32(take_dom, f_e, best_e);
                    dom_idx = branchless_select_usize(take_dom, i, dom_idx);
                    dom_zimms_addr = branchless_select_u64(
                        take_dom & (rec.zimms_section_offset48 != 0),
                        rec.zimms_section_offset48,
                        branchless_select_u64(
                            dom_zimms_addr != 0,
                            dom_zimms_addr,
                            rec.zimms_section_offset48,
                        ),
                    );
                    action_flags |= rec.action_mode_flags;
                    swarm_mask &= rec.active_swarm_mask;
                    prov_bits |= rec_prov;
                    dom_cause =
                        branchless_select_u8(
                            rec.dominant_cause_code > dom_cause,
                            rec.dominant_cause_code,
                            dom_cause,
                        );
                    tot_e = new_tot;
                }
            }
            i += 1;
        }

        let safe_tot_e = branchless_select_f32(tot_e > 1.0e-12, tot_e, 1.0);
        let inv_tot_e = fast_inv_pos_normal_f32(safe_tot_e);
        let eff_inv_temp = branchless_select_f32(
            tot_e > 1.0e-12,
            branchless_clamp_f32(weighted_inv_temp * inv_tot_e, 0.25, 4.0),
            records[dom_idx].effective_inv_temp,
        );
        let fp_coh = branchless_select_f32(
            tot_e > 1.0e-12,
            branchless_clamp_f32(weighted_coh * inv_tot_e, 0.0, 1.0),
            records[dom_idx].resultant_r_bar,
        );
        let dom = &records[dom_idx];
        let dom_slot = dom.dominant_slot;
        let is_fact_slot = dom_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = dom_slot == ATTRACTOR_SLOT_PERSONA;
        let pspm_fact =
            branchless_select_u8(is_fact_slot, 36, branchless_select_u8(is_persona_slot, 16, 24));
        let pspm_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let pspm_explore =
            branchless_select_u8(is_fact_slot, 8, branchless_select_u8(is_persona_slot, 24, 16));
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        let phase_deltas = words_to_q7_array_64(&acc_w);
        let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&acc_w, acc_is_uniform);
        let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
        let tagged_ptr = gcso_tagged_ptr_t::pack(dom_zimms_addr, packed_tag, tag_q7).raw();

        (
            GatewayFrame {
                tagged_ptr,
                zimms_section_offset48: dom_zimms_addr,
                target_anchor_id: dom.anchor_id,
                effective_energy: branchless_min_f32(tot_e, MAX_ATTRACTOR_PULL_FORCE),
                ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
                effective_inv_temp: eff_inv_temp,
                fact_persona_coherence: fp_coh,
                repulsion_severity: 0.0,
                provenance_bits: prov_bits,
                dominant_cause_code: dom_cause,
                dominant_slot: dom_slot,
                action_mode_flags: action_flags,
                active_swarm_cell_mask: swarm_mask,
                qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
                pspm_fact_heads: pspm_fact,
                pspm_logic_heads: pspm_logic,
                pspm_explore_heads: pspm_explore,
                pspm_fact_gain_q7: fact_gain_q7,
                pspm_logic_gain_q7: logic_gain_q7,
                pspm_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas,
            },
            acc_w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

// ===================================================================
// 10. Compile-Time Layout Verification & Unit Tests
// ===================================================================

const _: () = {
    assert!(size_of::<CausalProvenanceReport>() == 16);
    assert!(align_of::<CausalProvenanceReport>() == 16);
    assert!(offset_of!(CausalProvenanceReport, provenance_bits) == 0);
    assert!(offset_of!(CausalProvenanceReport, fact_persona_coherence) == 4);
    assert!(offset_of!(CausalProvenanceReport, anomaly_severity) == 8);
    assert!(offset_of!(CausalProvenanceReport, dominant_cause_code) == 12);

    assert!(size_of::<GatewayFrame>() == 128);
    assert!(align_of::<GatewayFrame>() == 64);
    assert!(offset_of!(GatewayFrame, tagged_ptr) == 0);
    assert!(offset_of!(GatewayFrame, zimms_section_offset48) == 8);
    assert!(offset_of!(GatewayFrame, target_anchor_id) == 16);
    assert!(offset_of!(GatewayFrame, effective_energy) == 20);
    assert!(offset_of!(GatewayFrame, pspm_fact_gain_q7) == 56);
    assert!(offset_of!(GatewayFrame, mean_phase_tag_q7) == 59);
    assert!(offset_of!(GatewayFrame, active_head_count) == 60);
    assert!(offset_of!(GatewayFrame, frame_flags) == 62);
    assert!(offset_of!(GatewayFrame, phase_deltas) == 64);

    assert!(size_of::<CrystallizedAttractorRecord>() == 128);
    assert!(align_of::<CrystallizedAttractorRecord>() == 64);
    assert!(offset_of!(CrystallizedAttractorRecord, magic) == 0);
    assert!(offset_of!(CrystallizedAttractorRecord, version) == 4);
    assert!(offset_of!(CrystallizedAttractorRecord, anchor_id) == 8);
    assert!(offset_of!(CrystallizedAttractorRecord, sample_count) == 12);
    assert!(offset_of!(CrystallizedAttractorRecord, zimms_section_offset48) == 16);
    assert!(offset_of!(CrystallizedAttractorRecord, checksum_crc32) == 52);
    assert!(offset_of!(CrystallizedAttractorRecord, dominant_slot) == 56);
    assert!(offset_of!(CrystallizedAttractorRecord, reserved_align64) == 60);
    assert!(offset_of!(CrystallizedAttractorRecord, phase_profile) == 64);
    assert!(GCSO_FOUR_SECTION_HEADER_BYTES == 192);
    assert!(GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET == 128);
    assert!(
        CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0).verify_integrity()
    );
    assert!(
        CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0).verify_integrity()
    );
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::gcso_daes_slot_t;
    use crate::traits::aligned::{AlignedBuffer128, AlignedBuffer64};
    use crate::traits::telemetry::TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;

    #[test]
    fn test_two_origin_supertrait_synthesis_and_causal_provenance() {
        let sys = SystemConstraint::new((), 0.20, 0.10, 0xFF);
        let fact = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 3.0, 0x0012_3400, 77);
        let persona_aligned =
            SubjectivePersona::new([30i8; MAX_ATTENTION_HEADS], 2.0, 0x0020_0000, 12);

        let (frame, words) = (&sys, &fact, &persona_aligned).synthesize_gateway_frame_with_words();
        assert_eq!(words, load_q7_words_64(&frame.phase_deltas));
        let report = frame.to_causal_report();
        assert_eq!(frame.zimms_section_offset48, 0x0012_3400);
        assert_eq!(frame.target_anchor_id, 77);
        assert_eq!(frame.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
        assert_eq!(report.dominant_cause_code, ROOT_CAUSE_NOMINAL);
        assert!(report.fact_persona_coherence > 0.95);
        assert!((frame.ripa_clamp_rad - DEFAULT_RIPA_CLAMP_MAX_RAD).abs() < 1e-6);

        let f32_emb = [0.5_f32, 0.866_025_4, -0.5, 0.866_025_4];
        let fact_f32 = ObjectiveFact::new(&f32_emb[..], 2.5, 0x0015_0000, 88);
        let (f_f32, w_f32) = fact_f32.synthesize_gateway_frame_with_words();
        assert_eq!(w_f32, load_q7_words_64(&f_f32.phase_deltas));

        let persona_opposed =
            SubjectivePersona::new([-96i8; MAX_ATTENTION_HEADS], 2.5, 0x0020_0000, 12);
        let col_frame = (&sys, &fact, &persona_opposed).synthesize_gateway_frame();
        let col_report = col_frame.to_causal_report();
        assert!(col_report.is_persona_fact_collision());
        assert_eq!(col_report.dominant_cause_code, ROOT_CAUSE_PERSONA_FACT_COLLISION);
        assert_eq!(col_frame.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
        assert_eq!(col_frame.target_anchor_id, 77);

        let guard = ContradictionGuard::new(ContrastivePhaseBloomFilter::new(), 0.65, 3);
        let rep_frame = (&sys, &fact, &persona_aligned, &guard).synthesize_gateway_frame();
        let rep_report = rep_frame.to_causal_report();
        assert!(rep_report.is_cohomology_repulsion());
        assert_eq!(rep_report.dominant_cause_code, ROOT_CAUSE_COHOMOLOGY_REPULSION);
        assert_eq!(rep_frame.dominant_slot, ATTRACTOR_SLOT_REPULSION);
    }

    #[test]
    fn test_online_toroidal_kmeans_dp_means_consolidation_and_four_section_view() {
        let mut centroids = [
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
        ];
        assert_eq!(centroids[0].causal_provenance_bits(), 0);
        assert_eq!(
            centroids[0].synthesize_gateway_frame().action_mode_flags,
            ACTION_PRESET_PASSIVE_SIDECAR
        );

        let mut trail = gcso_pointer_trail_t::default();

        let fact_a = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 2.5, 0x0010_0000, 101);
        let (f1, idx1, id1, _coh1, mass1, spawned1) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_signal_64(
                &mut centroids,
                &fact_a,
                DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                Some(&mut trail),
            );
        assert!(spawned1 && idx1 == 0 && id1 == 101 && trail.cluster_id == 101 && mass1 > 0.0);

        let fact_a2 = ObjectiveFact::new([30i8; MAX_ATTENTION_HEADS], 3.0, 0x0010_0000, 101);
        let (f2, idx2, id2, coh2, mass2, spawned2) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_signal_64(
                &mut centroids,
                &fact_a2,
                DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                Some(&mut trail),
            );
        assert!(!spawned2 && idx2 == 0 && id2 == 101 && coh2 > 0.95 && mass2 > mass1);
        assert_eq!(centroids[0].sample_count, 2);

        let persona_b = SubjectivePersona::new([-80i8; MAX_ATTENTION_HEADS], 2.8, 0x0024_0000, 202);
        let (_f3, idx3, id3, _coh3, mass3, spawned3) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_signal_64(
                &mut centroids,
                &persona_b,
                DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                Some(&mut trail),
            );
        assert!(spawned3 && idx3 == 1 && id3 == 202 && trail.cluster_id == 202 && mass3 > 0.0);

        let (act_cnt, tot_smp, mean_r, mean_k, _max_coh) =
            CrystallizedAttractorRecord::diagnose_macro_distribution_64(&centroids);
        assert_eq!(act_cnt, 2);
        assert_eq!(tot_smp, 3);
        assert!(mean_r > 0.85 && mean_k > 2.0);

        let closed_loop_frame = centroids.synthesize_gateway_frame();
        assert!(closed_loop_frame.effective_energy > 2.0);
        assert_ne!(
            closed_loop_frame.provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
            0
        );

        let batch_frames = [f1, f2];
        let (_b_idx, b_id, b_coh, b_spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_batch_frames_64(
                &mut centroids,
                &batch_frames,
                DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                Some(&mut trail),
            );
        assert_eq!(b_id, 101);
        assert_eq!(b_spawned, 0);
        assert!(b_coh > 0.95 && centroids[0].verify_integrity());

        // Populate slot 2 with a near-duplicate of slot 0 and verify online consolidation
        centroids[2] = CrystallizedAttractorRecord::new_empty(303, ATTRACTOR_SLOT_KNOWLEDGE, 0);
        let _ = centroids[2].absorb_sample_point_64(&[29i8; MAX_ATTENTION_HEADS], 1.2);
        let (merged_cnt, pruned_cnt) = CrystallizedAttractorRecord::merge_and_prune_centroids_64(
            &mut centroids,
            DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
            0.10,
            0.98,
        );
        assert_eq!(merged_cnt, 1);
        assert_eq!(pruned_cnt, 0);
        assert_eq!(centroids[2].sample_count, 0);
        assert!(centroids[0].verify_integrity() && centroids[2].verify_integrity());

        let mut query_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [-76i8; MAX_ATTENTION_HEADS],
            stigmergic_density: 1.0,
            attractor_pull_force: 1.0,
            ..Default::default()
        };
        let (routed_idx, routed_id, routed_coh, routed_grav) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_64(
                &centroids,
                &mut query_trail,
                64,
            );
        assert_eq!(routed_idx, 1);
        assert_eq!(routed_id, 202);
        assert_eq!(query_trail.cluster_id, 202);
        assert_eq!(
            gcso_tagged_ptr_t(query_trail.current_ptr).unpack_addr(),
            0x0024_0000
        );
        assert!(routed_coh > 0.95 && routed_grav > 1.0);

        let mut zimms_buf = AlignedBuffer64::<u8, 384>::new([0u8; 384]);
        let written = CrystallizedAttractorRecord::write_records_to_zimms_bytes(
            &centroids,
            &mut zimms_buf.data,
        )
        .unwrap();
        assert_eq!(written, 384);
        let restored =
            CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy(&zimms_buf.data, 3)
                .unwrap();
        assert_eq!(restored[0].anchor_id, 101);
        assert_eq!(restored[1].anchor_id, 202);

        let bloom = ContrastivePhaseBloomFilter::new();
        let daes = gcso_daes_slot_t::default();
        let reports_arr = [f1.to_causal_report(), f2.to_causal_report()];
        let trails_arr = [query_trail];
        let mut container_buf = AlignedBuffer128::<u8, 1024>::new([0u8; 1024]);
        let no_rep_len = GcsoFourSectionView::write_four_section_container_to_bytes(
            &centroids[..2],
            &trails_arr,
            Some(&bloom),
            Some(&daes),
            1_774_900_000,
            &mut container_buf.data,
        )
        .unwrap();
        let no_rep_view =
            GcsoFourSectionView::from_zimms_bytes_zero_copy(&container_buf.data[..no_rep_len])
                .unwrap();
        assert_eq!(no_rep_view.crystals.len(), 2);

        let total_len = GcsoFourSectionView::write_four_section_with_reports_to_bytes(
            &centroids,
            &trails_arr,
            Some(&bloom),
            Some(&daes),
            &reports_arr,
            1_774_900_000,
            &mut container_buf.data,
        )
        .unwrap();

        let (_ev_frame, ev_idx, ev_id, ev_mass, _ev_spawned, ev_merged) =
            GcsoFourSectionView::evolve_four_section_kmeans_in_place(
                &mut container_buf.data[..total_len],
                &fact_a2,
                DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
                DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE,
                0,
            )
            .unwrap();
        assert_eq!(ev_idx, 0);
        assert_eq!(ev_id, 101);
        assert_eq!(ev_merged, 0);
        assert!(ev_mass > 1.0);

        let (r_idx, r_id, _r_coh, r_grav) =
            GcsoFourSectionView::route_four_section_coarse_to_fine_in_place(
                &mut container_buf.data[..total_len],
                0,
                32,
            )
            .unwrap();
        assert_eq!(r_idx, 1);
        assert_eq!(r_id, 202);
        assert!(r_grav > 1.0);

        let view =
            GcsoFourSectionView::from_zimms_bytes_zero_copy(&container_buf.data[..total_len])
                .unwrap();
        assert_eq!(view.header.attractor_field_offset, 192);
        assert_eq!(view.zimms_descriptor.file_size_bytes, total_len as u64);
        assert_eq!(view.zimms_descriptor.flags, 4);
        assert_eq!(view.crystals.len(), 3);
        assert_eq!(view.trails.len(), 1);
        assert_eq!(view.trails[0].cluster_id, 202);
        assert_eq!(view.provenance_reports.len(), 2);
        assert!(view.bloom.is_some() && view.daes.is_some());
        assert_eq!(view.daes.unwrap().telemetry_ring_head, 1);
        assert_ne!(view.daes.unwrap().fast_path_bypass_mask, 0);

        // Verify Top-2 Liminal Bifurcation Routing between two balanced attractors
        let mut lim_centroids = [
            CrystallizedAttractorRecord::new_empty(101, ATTRACTOR_SLOT_KNOWLEDGE, 0x0010_0000),
            CrystallizedAttractorRecord::new_empty(202, ATTRACTOR_SLOT_PERSONA, 0x0024_0000),
        ];
        let _ = lim_centroids[0].absorb_sample_point_64(&[16i8; MAX_ATTENTION_HEADS], 2.5);
        let _ = lim_centroids[1].absorb_sample_point_64(&[48i8; MAX_ATTENTION_HEADS], 2.5);
        let mut lim_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            stigmergic_density: 1.0,
            attractor_pull_force: 1.0,
            ..Default::default()
        };
        let (_k1, _k2, aid1, aid2, lim_coh, lim_grav, is_liminal) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_64(
                &lim_centroids,
                &mut lim_trail,
                64,
                DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO,
            );
        assert!(is_liminal && aid1 != 0 && aid2 != 0 && aid1 != aid2);
        assert!(lim_coh > 0.70 && lim_grav > 1.0);
        assert_ne!(lim_trail.flags & TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        // Verify Continuous Geodesic Affinity Prior on unstamped candidate trails (cluster_id == 0)
        let cand_aligned = gcso_pointer_trail_t {
            accumulated_phase_delta: [18i8; MAX_ATTENTION_HEADS],
            stigmergic_density: 1.0,
            cluster_id: 0,
            ..Default::default()
        };
        let cand_orthogonal = gcso_pointer_trail_t {
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            stigmergic_density: 1.0,
            cluster_id: 0,
            ..Default::default()
        };
        let mut src_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [16i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let (win_c, sel_tid, best_sc, best_c, found_ok) =
            CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64(
                &lim_centroids,
                &mut src_trail,
                &[cand_orthogonal, cand_aligned],
                128,
                1.0,
                0.25,
                0.10,
                0.50,
                0.35,
            );
        assert!(found_ok && win_c == 0 && sel_tid == 1 && best_sc > 0.5 && best_c > 0.95);
    }
}
