// name: src/core/src/traits/gateway.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.3 2-Origin Supertrait Hierarchy, Causal Provenance, Single Composite Manifold Gateway,
//! Classical Online Toroidal $k$-Means / DP-Means Macro Clustering, Online Basin Consolidation,
//! and `.gcso` 4-Section Zero-Copy Persistence.

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

use core::mem::{align_of, offset_of, size_of};

use crate::abi::{
    gcso_config_t, gcso_daes_slot_t, gcso_edbc_state_t, gcso_pointer_trail_t, gcso_pspm_config_t,
    gcso_q7_t, gcso_snapshot_header_t, gcso_srl_descriptor_t, gcso_tagged_ptr_t,
    gcso_zimms_descriptor_t, GCSO_ABI_VERSION_HEX, GCSO_ERROR_BUFFER_TOO_SMALL,
    GCSO_ERROR_CONTAINER_CORRUPTED, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER,
};

use super::aligned::{is_aligned_to, AlignedSlice, AlignedSlice32, AlignedSliceMut32};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::branchless::{
    avx2_coherence_from_diff_vec64, avx2_coherence_from_diff_vec64_with_floor,
    avx2_qdps_filter_vec64, avx2_scale_vec64, avx2_slerp_from_diff_vec64, avx2_slerp_vec64,
    avx2_vec64_to_words64, avx2_words64_to_vec64,
};
use super::branchless::coherence_above_floor_from_word_arrays_64;
use super::branchless::{
    branchless_abs_f32, branchless_atan2_to_q7_fast, branchless_clamp_f32, branchless_crc32_ieee,
    branchless_flush_subnormal_f32, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_lerp_f32, branchless_maturity_scale_u32, branchless_max_f32, branchless_min_f32,
    branchless_phase_coherence_inv_temp_scale, branchless_phase_proximity_gravity_f32,
    branchless_popcount_rsqrt_u32, branchless_q7_circular_resultant_64, branchless_q7_coherence_64,
    branchless_q7_default_gravity_from_diff_u8,
    branchless_rad_to_qdps_step_u8, branchless_relu_f32, branchless_rsqrt_f32,
    branchless_select_f32, branchless_select_i8, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, branchless_select_usize, branchless_splitmix64,
    branchless_toroidal_centroid_step_words_with_coh_and_uni_64,
    branchless_toroidal_centroid_step_words_with_coh_uni_and_changed_64,
    coherence_from_uniform_words_x8,
    coherence_from_word_arrays_64, coherence_from_word_arrays_with_both_uniform_flags_64,
    crc32_block64_zero_seed_words, crc32_crystal_zoff16_final_u32,
    crc32_crystal_stamp_contrib_128b, crc32_fold_stamped_crystal_128b,
    crc32_slice16_dep_lo0, crc32_slice16_indep_hi0_u32, crc32_slice16_indep_hi1_u32,
    crc32_slice16_indep_lo1_u32, crc32_slice16_step_u64x2,
    crc32_step_block64_words, crc32_step_l16, crc32_step_l3, crc32_step_l32,
    crc32_step_l4, crc32_step_l5, crc32_step_l6, crc32_step_l8,
    branchless_q7_circular_lerp, crc32_update_crystal_absorption_delta,
    crc32_update_crystal_energy_and_mass_delta, crc32_update_crystal_stationary_fields_delta,
    crc32_crystal_be24_final_u32, crc32_crystal_gm36_final_u32,
    CRC32_CRYSTAL_ACTION_BYTE57_REF, CRC32_CRYSTAL_AID_BYTE0_REF, CRC32_CRYSTAL_AID_BYTE1_REF,
    CRC32_CRYSTAL_PROV_BYTE0_REF, CRC32_CRYSTAL_PROV_BYTE1_REF, CRC32_CRYSTAL_SC_BYTE0_REF,
    CRC32_CRYSTAL_SLOT_BYTE56_LUT_8, CRC32_CRYSTAL_ZOFF_BYTE1_REF, CRC32_CRYSTAL_ZOFF_BYTE2_REF,
    CRC32_CRYSTAL_ZOFF_BYTE3_REF, CRC32_IEEE_SLICE16_REF,
    CRC32_ZIMMS_DESC_BASE_SEED, CRYSTAL_CRC32_DEFAULT_W5_FINAL_CONTRIB,
    CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB, CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB,
    CRYSTAL_CRC32_RBAR_ONE_FINAL_CONTRIB, CRYSTAL_CRC32_W0_FINAL_CONTRIB,
    VON_MISES_KAPPA_AT_MAX_R_BITS,
    Q7_FULL_COS_REF,
    fast_inv_len_f32, fast_inv_pos_normal_f32, fast_ln_1p_bounded_f32, is_word_array_uniform_64,
    load_q7_words_64, trail_phase_words_mut, trail_phase_words_ref,
    scale_f32_slice_unrolled8, store_q7_words_64,
    swar_q7_circular_add_x8, swar_q7_circular_diff_x8,
    swar_q7_mean_i8_from_words_64, swar_q7_mean_i8_from_words_with_uniform_flag_64,
    swar_q7_neg_sat_x8, swar_q7_pack_x8, swar_q7_qdps_filter_words_with_uniform_flag_64,
    swar_q7_scale_le128_with_bias_x8,
    swar_slerp_words_with_uniform_flag_64, BranchlessKernel,
    DEFAULT_PHASE_GRAVITY_EPS,
};
use super::primitives::{
    init_snapshot_header, pack_action_slot_tag, validate_snapshot_header, GcsoResult, Pod,
    ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_REPULSE,
    ACTION_MODE_PSPM_ROUTE, ACTION_MODE_SRL_INJECT, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR, ACTION_PRESET_REPULSION_ESCAPE,
    ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE, DAES_MINI_LEDGER_LEN,
    DAES_SHORTCUT_SLOTS, DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD,
    GCSO_SNAPSHOT_MAGIC_U32, MAX_ATTENTION_HEADS,
};
use super::rope::{
    dispatch_fused_ripa_and_pspm_words_core, rotate_fused_self_ripa_and_pspm_default_words,
    rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words, steer_from_swar_words_core_safe,
    PhaseSteering, PspmRouter,
};
use super::telemetry::{
    ContrastivePhaseBloomFilter, SwarmResolutionReport,
    DAES_METRIC_FAST_PATH_HIT, DAES_METRIC_LAYER_SKIP_COMPENSATED, DAES_METRIC_PHASE_STEERED,
    DAES_METRIC_PSPM_ROUTED, DAES_METRIC_REPULSION_APPLIED, DAES_METRIC_SLOT_SYNERGY_ACTIVE,
    DAES_METRIC_SRL_INJECTED, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    NUM_SWARM_MICRO_CELLS, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
use super::{
    filter_qdps_words_in_registers_64, rad_to_qdps_min_step_q7_u8, swar_words_equal_64,
    swar_words_nonzero_64, validate_optional_qk_tensors, ActionHub,
};

// ===================================================================
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
/// Byte offset (`128`) within the 192-byte Section 0 preamble where [`gcso_zimms_descriptor_t`] is stored.
pub const GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET: usize = 128;

const SWAR_SPLAT_BYTE_64: u64 = 0x0101_0101_0101_0101;
const SWAR_SIGN_MASK_64: u64 = 0x8080_8080_8080_8080;
const SWAR_LOW7_MASK_64: u64 = 0x7F7F_7F7F_7F7F_7F7F;
const INV_FACT_RIPA_CLAMP_RAD: f32 = 1.0 / DEFAULT_FACT_RIPA_CLAMP_RAD;
const MAX_REGISTER_CACHED_CENTROIDS: usize = 16;

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
    if len == 0 { return [0u64; NUM_SWARM_MICRO_CELLS]; }
    let mut seed = (len as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut i = 0usize;
    while i + 8 <= len {
        let w = u64::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2],
            bytes[i + 3], bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]);
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
        let mixed = branchless_splitmix64(seed.wrapping_add((cell as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)));
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
/// via pairwise $SO(2)$ angles evaluated with [`branchless_atan2_to_q7_fast`], avoiding any 64-byte
/// stack array roundtrip when synthesizing `ObjectiveFact<&[f32]>` or `SubjectivePersona<&[f32]>`.
#[inline]
#[must_use]
pub const fn project_f32_slice_to_words_64(values: &[f32]) -> [u64; NUM_SWARM_MICRO_CELLS] {
    let n = values.len();
    if n == 0 { return [0u64; NUM_SWARM_MICRO_CELLS]; }
    let mut words = [0u64; NUM_SWARM_MICRO_CELLS];
    if n >= 128 {
        let mut cell = 0usize;
        while cell < NUM_SWARM_MICRO_CELLS {
            let base_i = cell << 4;
            let q0 = branchless_atan2_to_q7_fast(values[base_i + 1], values[base_i]);
            let q1 = branchless_atan2_to_q7_fast(values[base_i + 3], values[base_i + 2]);
            let q2 = branchless_atan2_to_q7_fast(values[base_i + 5], values[base_i + 4]);
            let q3 = branchless_atan2_to_q7_fast(values[base_i + 7], values[base_i + 6]);
            let q4 = branchless_atan2_to_q7_fast(values[base_i + 9], values[base_i + 8]);
            let q5 = branchless_atan2_to_q7_fast(values[base_i + 11], values[base_i + 10]);
            let q6 = branchless_atan2_to_q7_fast(values[base_i + 13], values[base_i + 12]);
            let q7 = branchless_atan2_to_q7_fast(values[base_i + 15], values[base_i + 14]);
            words[cell] = swar_q7_pack_x8(&[q0, q1, q2, q3, q4, q5, q6, q7]);
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
            let q0 = branchless_atan2_to_q7_fast(values[1], values[0]);
            let q1 = branchless_atan2_to_q7_fast(values[3], values[2]);
            let q2 = branchless_atan2_to_q7_fast(values[5], values[4]);
            let q3 = branchless_atan2_to_q7_fast(values[7], values[6]);
            let q4 = branchless_atan2_to_q7_fast(values[9], values[8]);
            let q5 = branchless_atan2_to_q7_fast(values[11], values[10]);
            let q6 = branchless_atan2_to_q7_fast(values[13], values[12]);
            let q7 = branchless_atan2_to_q7_fast(values[15], values[14]);
            return [swar_q7_pack_x8(&[q0, q1, q2, q3, q4, q5, q6, q7]); NUM_SWARM_MICRO_CELLS];
        }
        let mask = n - 1;
        let unique_cells = branchless_select_usize(
            (n >> 4) < NUM_SWARM_MICRO_CELLS,
            n >> 4,
            NUM_SWARM_MICRO_CELLS,
        );
        let mut cell = 0usize;
        while cell < unique_cells {
            let base_i = cell << 4;
            let q0 = branchless_atan2_to_q7_fast(values[(base_i + 1) & mask], values[base_i & mask]);
            let q1 = branchless_atan2_to_q7_fast(values[(base_i + 3) & mask], values[(base_i + 2) & mask]);
            let q2 = branchless_atan2_to_q7_fast(values[(base_i + 5) & mask], values[(base_i + 4) & mask]);
            let q3 = branchless_atan2_to_q7_fast(values[(base_i + 7) & mask], values[(base_i + 6) & mask]);
            let q4 = branchless_atan2_to_q7_fast(values[(base_i + 9) & mask], values[(base_i + 8) & mask]);
            let q5 = branchless_atan2_to_q7_fast(values[(base_i + 11) & mask], values[(base_i + 10) & mask]);
            let q6 = branchless_atan2_to_q7_fast(values[(base_i + 13) & mask], values[(base_i + 12) & mask]);
            let q7 = branchless_atan2_to_q7_fast(values[(base_i + 15) & mask], values[(base_i + 14) & mask]);
            words[cell] = swar_q7_pack_x8(&[q0, q1, q2, q3, q4, q5, q6, q7]);
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
        let q0 = branchless_atan2_to_q7_fast(values[((base_h) << 1 | 1) % n], values[((base_h) << 1) % n]);
        let q1 = branchless_atan2_to_q7_fast(values[((base_h + 1) << 1 | 1) % n], values[((base_h + 1) << 1) % n]);
        let q2 = branchless_atan2_to_q7_fast(values[((base_h + 2) << 1 | 1) % n], values[((base_h + 2) << 1) % n]);
        let q3 = branchless_atan2_to_q7_fast(values[((base_h + 3) << 1 | 1) % n], values[((base_h + 3) << 1) % n]);
        let q4 = branchless_atan2_to_q7_fast(values[((base_h + 4) << 1 | 1) % n], values[((base_h + 4) << 1) % n]);
        let q5 = branchless_atan2_to_q7_fast(values[((base_h + 5) << 1 | 1) % n], values[((base_h + 5) << 1) % n]);
        let q6 = branchless_atan2_to_q7_fast(values[((base_h + 6) << 1 | 1) % n], values[((base_h + 6) << 1) % n]);
        let q7 = branchless_atan2_to_q7_fast(values[((base_h + 7) << 1 | 1) % n], values[((base_h + 7) << 1) % n]);
        words[cell] = swar_q7_pack_x8(&[q0, q1, q2, q3, q4, q5, q6, q7]);
        cell += 1;
    }
    words
}

/// Projects an `f32` embedding slice into a 64-head Q7 phase profile (`[gcso_q7_t; 64]`) via
/// pairwise $SO(2)$ angles evaluated with [`branchless_atan2_to_q7_fast`].
#[inline]
#[must_use]
pub const fn project_f32_slice_to_q7_64(values: &[f32]) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    let words = project_f32_slice_to_words_64(values);
    words_to_q7_array_64(&words)
}

/// Computes the circular mean Q7 phase tag across 64 attention heads in branchless mode.
#[inline(always)] #[must_use]
pub fn mean_q7_phase_tag_64(phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> gcso_q7_t {
    branchless_q7_circular_resultant_64(phases).1
}

#[inline(always)] #[must_use]
fn update_aligned64_blocks_crc32_raw(mut crc: u32, bytes: &[u8]) -> u32 {
    let len = bytes.len();
    let num_blocks = len >> 6;
    // SAFETY: Caller guarantees `bytes` is 8-byte aligned and `len` is a multiple of 64.
    let blocks = unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<[u64; 8]>(), num_blocks) };
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

#[inline(always)] #[must_use]
fn compute_aligned64_payload_crc32_ieee(bytes: &[u8]) -> u32 {
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
const fn words_to_q7_array_64(
    words: &[u64; NUM_SWARM_MICRO_CELLS],
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    // SAFETY: `[u64; 8]` and `[i8; 64]` have identical size (64 bytes) and Pod bit validity.
    unsafe {
        core::mem::transmute::<[u64; NUM_SWARM_MICRO_CELLS], [gcso_q7_t; MAX_ATTENTION_HEADS]>(
            *words,
        )
    }
}

#[inline(always)] #[must_use]
const fn fast_mean_q7_tag_from_words_64(words: &[u64; NUM_SWARM_MICRO_CELLS]) -> gcso_q7_t {
    let raw = swar_q7_mean_i8_from_words_64(words);
    branchless_select_i8(raw == -128, -127, raw)
}

/// Trait for raw payloads that can be projected onto the 64-head $SO(2)^{64}$ Q7 phase manifold.
pub trait RawPhaseProjectable: BranchlessKernel {
    /// Projects `self` onto a 64-head Q7 phase array (`[gcso_q7_t; 64]`).
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS];

    /// Projects `self` directly into 8 register-resident `u64` SWAR words (`[u64; 8]`).
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { load_q7_words_64(&self.project_to_q7_64()) }
}

impl RawPhaseProjectable for () {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { [0i8; MAX_ATTENTION_HEADS] }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { [0u64; NUM_SWARM_MICRO_CELLS] }
}

impl RawPhaseProjectable for [gcso_q7_t; MAX_ATTENTION_HEADS] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { *self }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { load_q7_words_64(self) }
}

impl RawPhaseProjectable for [u8] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_bytes_swar_to_q7_64(self) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_bytes_swar_to_words_64(self) }
}

impl<const N: usize> RawPhaseProjectable for [u8; N] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_bytes_swar_to_q7_64(self.as_slice()) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_bytes_swar_to_words_64(self.as_slice()) }
}

impl RawPhaseProjectable for str {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_bytes_swar_to_q7_64(self.as_bytes()) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_bytes_swar_to_words_64(self.as_bytes()) }
}

impl RawPhaseProjectable for &str {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_bytes_swar_to_q7_64(self.as_bytes()) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_bytes_swar_to_words_64(self.as_bytes()) }
}

impl RawPhaseProjectable for &[u8] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_bytes_swar_to_q7_64(self) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_bytes_swar_to_words_64(self) }
}

impl RawPhaseProjectable for [f32] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_f32_slice_to_q7_64(self) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_f32_slice_to_words_64(self) }
}

impl RawPhaseProjectable for &[f32] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_f32_slice_to_q7_64(self) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_f32_slice_to_words_64(self) }
}

impl<const N: usize> RawPhaseProjectable for [f32; N] {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_f32_slice_to_q7_64(self.as_slice()) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_f32_slice_to_words_64(self.as_slice()) }
}

impl<'a, const ALIGN: usize> RawPhaseProjectable for AlignedSlice<'a, f32, ALIGN> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_f32_slice_to_q7_64(self.as_slice()) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_f32_slice_to_words_64(self.as_slice()) }
}

impl<'a, const ALIGN: usize> RawPhaseProjectable for AlignedSlice<'a, u8, ALIGN> {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { project_bytes_swar_to_q7_64(self.as_slice()) }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { project_bytes_swar_to_words_64(self.as_slice()) }
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
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { self.accumulated_phase_delta }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { *trail_phase_words_ref(self) }
}

impl RawPhaseProjectable for GatewayFrame {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { self.phase_deltas }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { *self.phase_words_ref() }
}

impl RawPhaseProjectable for CrystallizedAttractorRecord {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { self.phase_profile }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { *self.phase_words_ref() }
}

// ===================================================================
// 3. Root Supertrait (`CausalProvenanceSignal`) & 16-Byte Report
// ===================================================================

/// 16-byte (`#[repr(C, align(16))]`) [`Pod`] diagnostic report enabling $\mathcal{O}(1)$
/// root-cause ("Why") isolation across System Constraints, Objective Facts, Subjective Persona,
/// and Contradiction Guards.
#[repr(C, align(16))] #[derive(Debug, Clone, Copy, PartialEq, Default)] pub struct CausalProvenanceReport {
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
    #[inline(always)] #[must_use]
    pub const fn has_flag(&self, flag_mask: u32) -> bool { (self.provenance_bits & flag_mask) == flag_mask }

    /// Returns `true` when a destructive phase collision between Objective Fact and Subjective Persona was detected.
    #[inline(always)] #[must_use]
    pub const fn is_persona_fact_collision(&self) -> bool { (self.provenance_bits & CAUSAL_CAUSE_PERSONA_FACT_COLLISION) != 0 }

    /// Returns `true` when cohomological obstruction ($H^1 \neq 0$) or Bloom repetition repulsion was triggered.
    #[inline(always)] #[must_use]
    pub const fn is_cohomology_repulsion(&self) -> bool { (self.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0 }
}

/// Root supertrait for all signals entering the GCSO Manifold Gateway.
pub trait CausalProvenanceSignal: BranchlessKernel {
    /// Returns the bitwise provenance flags (`CAUSAL_AXIS_* | CAUSAL_CAUSE_*`) contributed by `self`.
    fn causal_provenance_bits(&self) -> u32;

    /// Returns the normalized anomaly or constraint severity in $[0.0, 1.0]$.
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 { 0.0 }
}

impl<T: CausalProvenanceSignal + ?Sized> CausalProvenanceSignal for &T {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 { (**self).causal_provenance_bits() }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 { (**self).causal_anomaly_severity() }
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
    fn system_gpu_frame_overrun(&self) -> f32 { 0.0 }

    /// 8-bit bitmask of active 8-head SWAR Micro-Swarm cells (`0xFF` = all 64 heads active).
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        let pressure = branchless_max_f32(branchless_clamp_f32(self.system_vram_pressure(), 0.0, 1.0),
            branchless_clamp_f32(self.system_gpu_frame_overrun(), 0.0, 1.0));
        branchless_select_u8(pressure > 0.85, 0x0F, branchless_select_u8(pressure > 0.65, 0x3F, 0xFF))
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
        let pressure = branchless_max_f32(branchless_clamp_f32(self.system_vram_pressure(), 0.0, 1.0),
            branchless_clamp_f32(self.system_gpu_frame_overrun(), 0.0, 1.0));
        DEFAULT_QDPS_MIN_STEP_RAD * (1.0 + 2.0 * pressure)
    }
}

impl<T: SystemConstraintSignal + ?Sized> SystemConstraintSignal for &T {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 { (**self).system_vram_pressure() }
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 { (**self).system_gpu_frame_overrun() }
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 { (**self).active_swarm_cells_mask() }
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool { (**self).recommend_layer_skip() }
    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 { (**self).qdps_min_step_rad_override() }
}

/// Zero-cost wrapper classifying any system/resource state `T` into **Origin 1 (`SystemConstraintSignal`)**.
#[repr(C)] #[derive(Debug, Clone, Copy, PartialEq)] pub struct SystemConstraint<T = ()> {
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
    pub const fn new(inner: T, vram_pressure: f32, gpu_frame_overrun: f32,
        active_swarm_mask: u8) -> Self {
        let p_vram = branchless_clamp_f32(branchless_relu_f32(vram_pressure), 0.0, 1.0);
        let p_gpu = branchless_clamp_f32(branchless_relu_f32(gpu_frame_overrun), 0.0, 1.0);
        let max_p = branchless_max_f32(p_vram, p_gpu);
        Self {
            inner, vram_pressure: p_vram,
            gpu_frame_overrun: p_gpu, qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD * (1.0 + 2.0 * max_p),
            active_swarm_mask: branchless_select_u8(active_swarm_mask != 0, active_swarm_mask, 0xFF,
            ), force_layer_skip: (p_gpu > 0.80) | (p_vram > 0.90),
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
        CAUSAL_AXIS_SYSTEM_CONSTRAINT | branchless_select_u32(throttle, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 { branchless_max_f32(self.vram_pressure, self.gpu_frame_overrun) }
}

impl<T: BranchlessKernel> SystemConstraintSignal for SystemConstraint<T> {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 { self.vram_pressure }
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 { self.gpu_frame_overrun }
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 { self.active_swarm_mask }
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool { self.force_layer_skip }
    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 { self.qdps_min_step_rad }
}

impl CausalProvenanceSignal for gcso_edbc_state_t {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_rep = self.active_branch_mode == 2;
        let is_low_e = (self.active_branch_mode == 0) & (self.moving_z_entropy < 0.12);
        CAUSAL_AXIS_SYSTEM_CONSTRAINT | branchless_select_u32(is_rep, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
            | branchless_select_u32(is_low_e, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        let h_sev = branchless_clamp_f32(branchless_relu_f32(self.moving_z_entropy) * 0.25, 0.0, 1.0);
        branchless_select_f32(self.active_branch_mode == 2, 0.85, h_sev)
    }
}

impl SystemConstraintSignal for gcso_edbc_state_t {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 { branchless_clamp_f32(branchless_relu_f32(self.moving_z_entropy) * 0.25, 0.0, 1.0) }
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 { branchless_clamp_f32(branchless_relu_f32(self.sliding_entropy_rate), 0.0, 1.0) }
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        let low_e = (self.active_branch_mode == 0) & (self.moving_z_entropy < 0.10)
            & (self.sliding_entropy_rate <= 0.5);
        branchless_select_u8(low_e, 0x0F, 0xFF)
    }
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        (self.active_branch_mode == 0) & (self.moving_z_entropy <= 0.12) & (self.sliding_entropy_rate <= 0.5)
    }
}

impl CausalProvenanceSignal for gcso_config_t {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 { CAUSAL_AXIS_SYSTEM_CONSTRAINT }
}

impl SystemConstraintSignal for gcso_config_t {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 { 0.0 }
    #[inline(always)]
    fn qdps_min_step_rad_override(&self) -> f32 {
        let valid = branchless_is_finite_f32(self.qdps_min_step_rad) & (self.qdps_min_step_rad >= 0.0);
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
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> { None }

    /// Projects this grounding signal directly into 8 register-resident `u64` SWAR words (`[u64; 8]`).
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        match self.grounding_phases_ref_64() {
            Some(arr_ref) => load_q7_words_64(arr_ref), None => load_q7_words_64(&self.project_grounding_phases_64()),
        }
    }

    /// Base gravitational pull energy $E_{\mathrm{base}} \ge 0$ of this grounding signal.
    fn grounding_base_energy(&self) -> f32;

    /// Canonical 48-bit `.gcso` (ZIMMS) section or RAG record offset (`bits 0..=47` of `gcso_tagged_ptr_t`).
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 { 0 }

    /// Full 32-bit macro attractor anchor identifier on the Single Unified Attractor Manifold.
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 { 0 }
}

impl<T: CognitiveGroundingSignal + ?Sized> CognitiveGroundingSignal for &T {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { (**self).project_grounding_phases_64() }
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> { (**self).grounding_phases_ref_64() }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { (**self).project_grounding_words_64() }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 { (**self).grounding_base_energy() }
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 { (**self).zimms_section_offset48() }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 { (**self).grounding_anchor_id() }
}

/// **Origin 2-A Subtrait (`ObjectiveFactSignal: CognitiveGroundingSignal`)**:
/// External RAG chunks, database records, and objective conversation logs (`Slot 2`, narrow RIPA clamp `0.14 rad`).
pub trait ObjectiveFactSignal: CognitiveGroundingSignal {
    /// RIPA phase clamp limit in radians for objective factual grounding (defaults to `0.14 rad`).
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 { DEFAULT_FACT_RIPA_CLAMP_RAD }

    /// PSPM `Fact` sub-head phase gain multiplier (defaults to `1.15`).
    #[inline(always)]
    fn fact_pspm_gain(&self) -> f32 { 1.15 }
}

impl<T: ObjectiveFactSignal + ?Sized> ObjectiveFactSignal for &T {
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 { (**self).fact_ripa_clamp_rad() }
    #[inline(always)]
    fn fact_pspm_gain(&self) -> f32 { (**self).fact_pspm_gain() }
}

/// **Origin 2-B Subtrait (`SubjectivePersonaSignal: CognitiveGroundingSignal`)**:
/// System prompts, character beliefs, stylistic tone, and subjective episodic memory (`Slot 1`, wide RIPA clamp `0.24 rad`).
pub trait SubjectivePersonaSignal: CognitiveGroundingSignal {
    /// RIPA phase clamp limit in radians for subjective persona modulation (defaults to `0.24 rad`).
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 { DEFAULT_PERSONA_RIPA_CLAMP_RAD }

    /// PSPM `Explore` / `Logic` sub-head phase gain multiplier (defaults to `1.20`).
    #[inline(always)]
    fn persona_pspm_gain(&self) -> f32 { 1.20 }
}

impl<T: SubjectivePersonaSignal + ?Sized> SubjectivePersonaSignal for &T {
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 { (**self).persona_ripa_clamp_rad() }
    #[inline(always)]
    fn persona_pspm_gain(&self) -> f32 { (**self).persona_pspm_gain() }
}

/// **Origin 2-C Subtrait (`ContradictionGuardSignal: CognitiveGroundingSignal`)**:
/// Sheaf cohomology obstruction ($H^1(K; \mathcal{F}) \neq 0$) and [`ContrastivePhaseBloomFilter`]
/// repetition / self-contradiction guard triggering anti-phase $-\boldsymbol{\Delta\theta}$ escape.
pub trait ContradictionGuardSignal: CognitiveGroundingSignal {
    /// Evaluates repulsive severity in $[0.0, 1.0]$ against `candidate_phases`.
    fn evaluate_repulsion_severity(&self, candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32;

    /// Evaluates repulsive severity directly from 8 register-resident `u64` SWAR words (`candidate_words`).
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(&self,
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS], candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        let _ = candidate_words;
        self.evaluate_repulsion_severity(candidate_phases)
    }

    /// Returns `true` if [`Self::evaluate_repulsion_severity_from_words`] inspects `candidate_phases`
    /// rather than operating purely on `candidate_words` in registers.
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool { true }
}

impl<T: ContradictionGuardSignal + ?Sized> ContradictionGuardSignal for &T {
    #[inline(always)]
    fn evaluate_repulsion_severity(&self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        (**self).evaluate_repulsion_severity(candidate_phases)
    }
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(&self,
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS], candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 { (**self).evaluate_repulsion_severity_from_words(candidate_words, candidate_phases) }
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool { (**self).requires_phase_slice_for_repulsion() }
}

/// Concrete wrapper classifying any [`RawPhaseProjectable`] payload `T` as an **Objective Fact** (`Origin 2-A`).
#[repr(C)] #[derive(Debug, Clone, Copy, PartialEq)] pub struct ObjectiveFact<T> {
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
    #[inline(always)] #[must_use]
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
        CAUSAL_AXIS_OBJECTIVE_FACT | branchless_select_u32(weak, CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 { branchless_select_f32(self.base_energy < 0.25, 0.35, 0.0) }
}

impl<T: RawPhaseProjectable> CognitiveGroundingSignal for ObjectiveFact<T> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { self.payload.project_to_q7_64() }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { self.payload.project_to_words_64() }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 { self.base_energy }
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 { self.zimms_offset48 }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 { self.anchor_id }
}

impl<T: RawPhaseProjectable> ObjectiveFactSignal for ObjectiveFact<T> {
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 { self.ripa_clamp_rad }
}

/// Concrete wrapper classifying any [`RawPhaseProjectable`] payload `T` as a **Subjective Persona** (`Origin 2-B`).
#[repr(C)] #[derive(Debug, Clone, Copy, PartialEq)] pub struct SubjectivePersona<T> {
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
    #[inline(always)] #[must_use]
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
    fn causal_provenance_bits(&self) -> u32 { CAUSAL_AXIS_SUBJECTIVE_PERSONA }
}

impl<T: RawPhaseProjectable> CognitiveGroundingSignal for SubjectivePersona<T> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { self.payload.project_to_q7_64() }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { self.payload.project_to_words_64() }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 { self.base_energy }
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 { self.zimms_offset48 }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 { self.anchor_id }
}

impl<T: RawPhaseProjectable> SubjectivePersonaSignal for SubjectivePersona<T> {
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 { self.ripa_clamp_rad }
}

/// Concrete wrapper for **Origin 2-C (`ContradictionGuardSignal`)** combining an optional
/// [`ContrastivePhaseBloomFilter`] (or `()`) with a sheaf cohomology obstruction residual.
#[repr(C)] #[derive(Debug, Clone, Copy, PartialEq)] pub struct ContradictionGuard<B = ContrastivePhaseBloomFilter> {
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
    pub const fn new(bloom: B,
        cohomology_obstruction: f32, bloom_repulsion_threshold: u8) -> Self {
        Self {
            bloom, cohomology_obstruction: branchless_flush_subnormal_f32(branchless_relu_f32(
                cohomology_obstruction)),
            obstruction_threshold: 0.25, bloom_repulsion_threshold: branchless_select_u8(
                bloom_repulsion_threshold > 0, bloom_repulsion_threshold, 4),
        }
    }

    /// Builder method setting the cohomological obstruction threshold.
    #[inline(always)]
    #[must_use]
    pub const fn with_obstruction_threshold(mut self, thresh: f32) -> Self {
        let valid = branchless_is_finite_f32(thresh) & (thresh > 0.0);
        self.obstruction_threshold = branchless_select_f32(valid, thresh, self.obstruction_threshold);
        self
    }
}

impl CausalProvenanceSignal for ContradictionGuard<ContrastivePhaseBloomFilter> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let obstructed = self.cohomology_obstruction > self.obstruction_threshold;
        CAUSAL_AXIS_CONTRADICTION_GUARD | branchless_select_u32(obstructed, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
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

impl CognitiveGroundingSignal for ContradictionGuard<ContrastivePhaseBloomFilter> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { [0i8; MAX_ATTENTION_HEADS] }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { [0u64; NUM_SWARM_MICRO_CELLS] }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 { self.cohomology_obstruction }
}

impl ContradictionGuardSignal for ContradictionGuard<ContrastivePhaseBloomFilter> {
    #[inline(always)]
    fn evaluate_repulsion_severity(&self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        let words = load_q7_words_64(candidate_phases);
        self.evaluate_repulsion_severity_from_words(&words, candidate_phases)
    }
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(&self,
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS], _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        let obs_sev = branchless_select_f32(obs_exceeded,
            branchless_clamp_f32(0.55 + (self.cohomology_obstruction - self.obstruction_threshold), 0.55, 1.0), 0.0);
        if obs_sev >= 1.0 {
            return 1.0;
        }
        let density = self.bloom.query_repulsive_density_from_words(candidate_words);
        let bloom_hit = density >= self.bloom_repulsion_threshold;
        let bloom_sev = branchless_select_f32(bloom_hit,
            branchless_clamp_f32(0.60 + (density as f32) * 0.05, 0.60, 1.0), 0.0);
        branchless_max_f32(obs_sev, bloom_sev)
    }
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool { false }
}

impl CausalProvenanceSignal for ContradictionGuard<()> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let obstructed = self.cohomology_obstruction > self.obstruction_threshold;
        CAUSAL_AXIS_CONTRADICTION_GUARD | branchless_select_u32(obstructed, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
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

impl CognitiveGroundingSignal for ContradictionGuard<()> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { [0i8; MAX_ATTENTION_HEADS] }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { [0u64; NUM_SWARM_MICRO_CELLS] }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 { self.cohomology_obstruction }
}

impl ContradictionGuardSignal for ContradictionGuard<()> {
    #[inline(always)]
    fn evaluate_repulsion_severity(&self,
        _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(obs_exceeded,
            branchless_clamp_f32(0.55 + (self.cohomology_obstruction - self.obstruction_threshold), 0.55, 1.0), 0.0)
    }
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(&self,
        _candidate_words: &[u64; NUM_SWARM_MICRO_CELLS], _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(obs_exceeded,
            branchless_clamp_f32(0.55 + (self.cohomology_obstruction - self.obstruction_threshold), 0.55, 1.0), 0.0)
    }
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool { false }
}

impl CausalProvenanceSignal for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 { CAUSAL_AXIS_CONTRADICTION_GUARD }
}

impl CognitiveGroundingSignal for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { [0i8; MAX_ATTENTION_HEADS] }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { [0u64; NUM_SWARM_MICRO_CELLS] }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 { 0.0 }
}

impl ContradictionGuardSignal for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn evaluate_repulsion_severity(&self,
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        let words = load_q7_words_64(candidate_phases);
        self.evaluate_repulsion_severity_from_words(&words, candidate_phases)
    }
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(&self,
        candidate_words: &[u64; NUM_SWARM_MICRO_CELLS], _candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> f32 {
        let density = self.query_repulsive_density_from_words(candidate_words);
        branchless_select_f32(density >= 3, branchless_clamp_f32(0.60 + (density as f32) * 0.05, 0.60, 1.0), 0.0)
    }
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool { false }
}

// ===================================================================
// 6. Single Composite Gateway Frame (`GatewayFrame` 128B Pod) & Trait
// ===================================================================

/// 128-byte (2-cacheline, `#[repr(C, align(64))]`) [`Pod`] composite descriptor produced by
/// [`ManifoldGatewaySignal::synthesize_gateway_frame`].
#[repr(C, align(64))] #[derive(Debug, Clone, Copy, PartialEq)] pub struct GatewayFrame {
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
            tagged_ptr: gcso_tagged_ptr_t::pack(0,
                pack_action_slot_tag(ATTRACTOR_SLOT_SYSTEM_BASELINE, ACTION_PRESET_EXPRESS_PHASE), 0).raw(),
            zimms_section_offset48: 0, target_anchor_id: 0,
            effective_energy: 0.0, ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD, effective_inv_temp: 1.0,
            fact_persona_coherence: 1.0, repulsion_severity: 0.0,
            provenance_bits: 0, dominant_cause_code: ROOT_CAUSE_NOMINAL,
            dominant_slot: ATTRACTOR_SLOT_SYSTEM_BASELINE, action_mode_flags: ACTION_PRESET_EXPRESS_PHASE,
            active_swarm_cell_mask: 0xFF, qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
            pspm_fact_heads: 24, pspm_logic_heads: 24, pspm_explore_heads: 16, pspm_fact_gain_q7: 128,
            pspm_logic_gain_q7: 128, pspm_explore_gain_q7: 128,
            mean_phase_tag_q7: 0, active_head_count: MAX_ATTENTION_HEADS as u16,
            frame_flags: 0, phase_deltas: [0i8; MAX_ATTENTION_HEADS],
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

    /// Extracts the 16-byte [`CausalProvenanceReport`] from this [`GatewayFrame`] in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn to_causal_report(&self) -> CausalProvenanceReport {
        CausalProvenanceReport {
            provenance_bits: self.provenance_bits, fact_persona_coherence: self.fact_persona_coherence,
            anomaly_severity: self.repulsion_severity, dominant_cause_code: self.dominant_cause_code,
            dominant_slot: self.dominant_slot, action_mode_flags: self.action_mode_flags,
            active_swarm_mask: self.active_swarm_cell_mask,
        }
    }

    /// Converts the Gateway's recommended Fact/Logic/Explore head split into a [`gcso_pspm_config_t`]
    /// scaled proportionally to `total_heads`.
    #[inline(always)] #[must_use]
    pub const fn to_pspm_config_for_heads(&self, total_heads: usize) -> gcso_pspm_config_t {
        let h = if total_heads > MAX_ATTENTION_HEADS {
            MAX_ATTENTION_HEADS
        } else {
            total_heads
        };
        if h == 0 {
            return gcso_pspm_config_t {
                num_fact_heads: 0, num_logic_heads: 0, num_explore_heads: 0, flags: 0,
                fact_phase_gain: 1.0, logic_phase_gain: 1.0, explore_phase_gain: 1.0, reserved: [0; 12],
            };
        }
        let f_raw = ((self.pspm_fact_heads as usize) * h + 32) >> 6;
        let l_raw = ((self.pspm_logic_heads as usize) * h + 32) >> 6;
        let f_cnt = if f_raw > h { h } else { f_raw };
        let rem = h - f_cnt;
        let l_cnt = if l_raw > rem { rem } else { l_raw };
        let e_cnt = h - f_cnt - l_cnt;
        let lut_normalized = branchless_abs_f32(self.ripa_clamp_rad - DEFAULT_RIPA_CLAMP_MAX_RAD) <= 1.0e-6;
        gcso_pspm_config_t {
            num_fact_heads: f_cnt as u16, num_logic_heads: l_cnt as u16, num_explore_heads: e_cnt as u16, flags: 1,
            fact_phase_gain: if lut_normalized { 1.0 } else { (self.pspm_fact_gain_q7 as f32) * (1.0 / 128.0) },
            logic_phase_gain: if lut_normalized { 1.0 } else { (self.pspm_logic_gain_q7 as f32) * (1.0 / 128.0) },
            explore_phase_gain: if lut_normalized { 1.0 } else { (self.pspm_explore_gain_q7 as f32) * (1.0 / 128.0) },
            reserved: [0; 12],
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
        let words = load_q7_words_64(&frame.phase_deltas);
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
    fn has_active_gateway_ensemble_signal(&self) -> bool { true }

    /// Synthesizes a contiguous slice of `Self` into a 128-byte [`GatewayFrame`] and its 8
    /// register-resident `u64` SWAR phase words (`[u64; 8]`), accumulating scalar metadata in
    /// registers and materializing `GatewayFrame` only once upon loop completion.
    #[inline(always)]
    fn synthesize_gateway_slice_with_words(
        slice: &[Self],
    ) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])
    where
        Self: Sized,
    {
        let len = slice.len();
        if len == 0 { return (GatewayFrame::default(), [0u64; NUM_SWARM_MICRO_CELLS]); }
        if len == 1 { return slice[0].synthesize_gateway_frame_with_words(); }

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
                        let w_ratio = branchless_clamp_f32(f_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
                        let s_u64 = ((w_ratio * 128.0 + 0.5) as u8).min(128) as u64;
                        if s_u64 >= 128 {
                            acc_w = f_w;
                            acc_is_uniform = is_word_array_uniform_64(&f_w);
                        } else if s_u64 > 0 && !swar_words_equal_64(&acc_w, &f_w) {
                            let f_is_uniform = acc_is_uniform && is_word_array_uniform_64(&f_w);
                            let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
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
                        eff_inv_temp = branchless_lerp_f32(eff_inv_temp, f.effective_inv_temp, w_ratio);
                        fp_coh = branchless_lerp_f32(fp_coh, f.fact_persona_coherence, w_ratio);

                        let take_dom = f_e > best_e;
                        best_e = branchless_select_f32(take_dom, f_e, best_e);
                        target_anchor_id = branchless_select_u32(take_dom, f.target_anchor_id, target_anchor_id);
                        zimms_addr = branchless_select_u64(
                            take_dom & (f.zimms_section_offset48 != 0),
                            f.zimms_section_offset48,
                            zimms_addr,
                        );
                        dom_slot = branchless_select_u8(take_dom, f.dominant_slot, dom_slot);
                        pspm_fact = branchless_select_u8(take_dom, f.pspm_fact_heads, pspm_fact);
                        pspm_logic = branchless_select_u8(take_dom, f.pspm_logic_heads, pspm_logic);
                        pspm_explore = branchless_select_u8(take_dom, f.pspm_explore_heads, pspm_explore);
                        fact_gain_q7 = branchless_select_u8(take_dom, f.pspm_fact_gain_q7, fact_gain_q7);
                        logic_gain_q7 = branchless_select_u8(take_dom, f.pspm_logic_gain_q7, logic_gain_q7);
                        explore_gain_q7 = branchless_select_u8(take_dom, f.pspm_explore_gain_q7, explore_gain_q7);
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

        if !has_active { return slice[0].synthesize_gateway_frame_with_words(); }

        let phase_deltas = words_to_q7_array_64(&acc_w);
        let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&acc_w, acc_is_uniform);
        let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
        let tagged_ptr = gcso_tagged_ptr_t::pack(zimms_addr, packed_tag, tag_q7).raw();
        (GatewayFrame {
                tagged_ptr, zimms_section_offset48: zimms_addr,
                target_anchor_id, effective_energy: branchless_min_f32(tot_e, MAX_ATTRACTOR_PULL_FORCE),
                ripa_clamp_rad, qdps_min_step_rad,
                effective_inv_temp: eff_inv_temp, fact_persona_coherence: fp_coh,
                repulsion_severity: rep_sev, provenance_bits: prov_bits,
                dominant_cause_code: dom_cause, dominant_slot: dom_slot,
                action_mode_flags: action_flags, active_swarm_cell_mask: swarm_mask,
                qdps_min_step_q7: qdps_step_q7, pspm_fact_heads: pspm_fact,
                pspm_logic_heads: pspm_logic, pspm_explore_heads: pspm_explore,
                pspm_fact_gain_q7: fact_gain_q7, pspm_logic_gain_q7: logic_gain_q7,
                pspm_explore_gain_q7: explore_gain_q7, mean_phase_tag_q7: tag_q7,
                active_head_count: active_heads, frame_flags: packed_tag as u16, phase_deltas,
            },
            acc_w)
    }
}

impl<T: ManifoldGatewaySignal + ?Sized> ManifoldGatewaySignal for &T {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { (**self).synthesize_gateway_frame() }
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) { (**self).synthesize_gateway_frame_with_words() }
    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) { (**self).synthesize_gateway_ensemble_item() }
    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool { (**self).has_active_gateway_ensemble_signal() }
}

impl CausalProvenanceSignal for GatewayFrame {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 { self.provenance_bits }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 { self.repulsion_severity }
}

impl ManifoldGatewaySignal for GatewayFrame {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { *self }
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        (*self, *self.phase_words_ref())
    }
}

/// Synthesizes both a 128-byte [`GatewayFrame`] and its register-resident 8 `u64` SWAR phase words
/// (`[u64; 8]`) from optional **Origin 1 (`SystemConstraintSignal`)** and **Origin 2
/// (`ObjectiveFactSignal`, `SubjectivePersonaSignal`, `ContradictionGuardSignal`)** components.
#[inline(always)]
#[must_use]
pub fn synthesize_multi_origin_gateway_frame_with_words<S, F, P, G>(sys_opt: Option<&S>,
    fact_opt: Option<&F>, persona_opt: Option<&P>,
    guard_opt: Option<&G>) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS])
where
    S: SystemConstraintSignal + ?Sized, F: ObjectiveFactSignal + ?Sized,
    P: SubjectivePersonaSignal + ?Sized, G: ContradictionGuardSignal + ?Sized, {
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
        if force_skip | (active_swarm_mask != 0xFF) { prov_bits |= CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE; }
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
        fact_clamp = branchless_select_f32(branchless_is_finite_f32(fc) & (fc > 0.0), fc, DEFAULT_FACT_RIPA_CLAMP_RAD,
        );
        let fg = branchless_clamp_f32(fact.fact_pspm_gain(), 0.25, 1.99);
        fact_gain_q7 = (fg * 128.0 + 0.5) as u8;
        fact_w = fact.project_grounding_words_64();
        if fact_e < 0.25 { prov_bits |= CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK; }
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
        pers_clamp = branchless_select_f32(branchless_is_finite_f32(pc) & (pc > 0.0),
            pc, DEFAULT_PERSONA_RIPA_CLAMP_RAD);
        let pg = branchless_clamp_f32(pers.persona_pspm_gain(), 0.25, 1.99);
        pers_gain_q7 = (pg * 128.0 + 0.5) as u8;
        pers_w = pers.project_grounding_words_64();
    }

    let mut fp_coh = 1.0_f32;
    let mut collision = false;
    let mut blended_w = [0u64; NUM_SWARM_MICRO_CELLS];
    let mut blended_uni = true;
    #[allow(unused_mut)]
    let mut avx_post_applied = false;
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
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut avx_fp_vecs = None;
        fp_coh = if same_fp {
            1.0
        } else if both_uni {
            coherence_from_uniform_words_x8(pers_w[0], fact_w[0])
        } else {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                unsafe {
                    use core::arch::x86_64::_mm256_sub_epi8;
                    let (vf0, vf1) =
                        super::branchless::avx2_words64_to_vec64_with_uniform(&fact_w, fact_uni);
                    let (vp0, vp1) =
                        super::branchless::avx2_words64_to_vec64_with_uniform(&pers_w, pers_uni);
                    let vd0 = _mm256_sub_epi8(vp0, vf0);
                    let vd1 = _mm256_sub_epi8(vp1, vf1);
                    avx_fp_vecs = Some((vf0, vf1, vd0, vd1));
                    avx2_coherence_from_diff_vec64(vd0, vd1)
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
                coherence_from_word_arrays_with_both_uniform_flags_64(
                    &pers_w,
                    pers_uni,
                    &fact_w,
                    fact_uni,
                )
            }
        };
        collision = fp_coh < -DEFAULT_PERSONA_FACT_COLLISION_THRESH;
        if collision { prov_bits |= CAUSAL_CAUSE_PERSONA_FACT_COLLISION; }
        let eff_pers_e = branchless_select_f32(collision,
            pers_e * 0.20, pers_e * branchless_clamp_f32(1.0 + 0.5 * fp_coh, 0.5, 1.5));
        let tot_e = fact_e + eff_pers_e;
        let safe_tot = branchless_select_f32(tot_e > 1.0e-12, tot_e, 1.0);
        let pers_ratio = branchless_clamp_f32(eff_pers_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
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
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                if let Some((vf0, vf1, vd0, vd1)) = avx_fp_vecs {
                    unsafe {
                        let (mut vb0, mut vb1) =
                            avx2_slerp_from_diff_vec64(vf0, vf1, vd0, vd1, s_u64);
                        let raw_c = branchless_lerp_f32(fact_clamp, pers_clamp, pers_ratio);
                        if raw_c < DEFAULT_FACT_RIPA_CLAMP_RAD {
                            let auth_ratio =
                                branchless_clamp_f32(raw_c * INV_FACT_RIPA_CLAMP_RAD, 0.10, 1.0);
                            let auth_u64 = ((auth_ratio * 128.0 + 0.5) as u8).min(128) as u64;
                            if auth_u64 < 128 {
                                (vb0, vb1) = avx2_scale_vec64(vb0, vb1, auth_u64);
                            }
                        }
                        if qdps_step_u8 > 1 {
                            (vb0, vb1) = avx2_qdps_filter_vec64(vb0, vb1, qdps_step_u8);
                        }
                        avx_precomputed_tag =
                            Some(super::branchless::avx2_q7_mean_i8_from_vec64(vb0, vb1));
                        blended_w = avx2_vec64_to_words64(vb0, vb1);
                        blended_uni = false;
                        avx_post_applied = true;
                    }
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
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
        }

        let synergy_boost = branchless_select_f32(collision, 1.0, 1.0 + 0.25 * branchless_relu_f32(fp_coh));
        eff_energy = branchless_min_f32(tot_e * synergy_boost, MAX_ATTRACTOR_PULL_FORCE);
        raw_clamp = branchless_lerp_f32(fact_clamp, pers_clamp, pers_ratio);

        let fact_dom = (fact_e >= pers_e) | collision;
        dom_slot = branchless_select_u8(fact_dom, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA);
        dom_anchor_id = branchless_select_u32(fact_dom & (fact_id != 0),
            fact_id, branchless_select_u32(pers_id != 0, pers_id, fact_id));
        zimms_addr = branchless_select_u64(fact_addr != 0,
            fact_addr, branchless_select_u64(pers_addr != 0, pers_addr, 0));
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
        if !avx_post_applied && raw_clamp < DEFAULT_FACT_RIPA_CLAMP_RAD {
            let auth_ratio = branchless_clamp_f32(raw_clamp * INV_FACT_RIPA_CLAMP_RAD, 0.10, 1.0);
            let auth_u64 = ((auth_ratio * 128.0 + 0.5) as u8).min(128) as u64;
            if auth_u64 < 128 {
                let bias_auth = (128u64.wrapping_sub(auth_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                if blended_uni {
                    blended_w = [swar_q7_scale_le128_with_bias_x8(blended_w[0], auth_u64, bias_auth); NUM_SWARM_MICRO_CELLS];
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

    if !avx_post_applied && qdps_step_u8 > 1 {
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
        rep_sev = branchless_clamp_f32(branchless_max_f32(g_sev, guard.causal_anomaly_severity()), 0.0, 1.0);
    }
    let repulsion_triggered = (rep_sev > 0.50) | ((prov_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
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
    if force_skip & (!repulsion_triggered) { action_flags |= ACTION_MODE_LAYER_SKIP; }

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

    let tag_q7 = match avx_precomputed_tag {
        Some(t) => t,
        None => swar_q7_mean_i8_from_words_with_uniform_flag_64(&blended_w, blended_uni),
    };
    let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
    let tagged_ptr = gcso_tagged_ptr_t::pack(zimms_addr, packed_tag, tag_q7).raw();
    let logic_gain_q7 = ((fact_gain_q7 as u16 + pers_gain_q7 as u16) >> 1) as u8;

    (GatewayFrame {
            tagged_ptr, zimms_section_offset48: zimms_addr,
            target_anchor_id: dom_anchor_id, effective_energy: eff_energy,
            ripa_clamp_rad: norm_ripa_clamp, qdps_min_step_rad: qdps_rad,
            effective_inv_temp: inv_temp, fact_persona_coherence: fp_coh,
            repulsion_severity: branchless_max_f32(rep_sev, sys_sev), provenance_bits: prov_bits,
            dominant_cause_code: dominant_cause, dominant_slot: dom_slot,
            action_mode_flags: action_flags, active_swarm_cell_mask: active_swarm_mask,
            qdps_min_step_q7: qdps_step_u8, pspm_fact_heads: pspm_fact,
            pspm_logic_heads: pspm_logic, pspm_explore_heads: pspm_explore,
            pspm_fact_gain_q7: fact_gain_q7, pspm_logic_gain_q7: logic_gain_q7,
            pspm_explore_gain_q7: pers_gain_q7, mean_phase_tag_q7: tag_q7,
            active_head_count: MAX_ATTENTION_HEADS as u16, frame_flags: packed_tag as u16, phase_deltas,
        },
        blended_w)
}

/// Synthesizes a 128-byte [`GatewayFrame`] from optional **Origin 1 (`SystemConstraintSignal`)**
/// and **Origin 2 (`ObjectiveFactSignal`, `SubjectivePersonaSignal`, `ContradictionGuardSignal`)** components.
#[inline(always)]
#[must_use]
pub fn synthesize_multi_origin_gateway_frame<S, F, P, G>(sys_opt: Option<&S>,
    fact_opt: Option<&F>, persona_opt: Option<&P>, guard_opt: Option<&G>) -> GatewayFrame
where
    S: SystemConstraintSignal + ?Sized, F: ObjectiveFactSignal + ?Sized,
    P: SubjectivePersonaSignal + ?Sized, G: ContradictionGuardSignal + ?Sized,
{ synthesize_multi_origin_gateway_frame_with_words(sys_opt, fact_opt, persona_opt, guard_opt).0 }

// ===================================================================
// 7. Blanket `ManifoldGatewaySignal` Implementations for Wrappers, Tuples & Ensembles
// ===================================================================

impl<T: BranchlessKernel> ManifoldGatewaySignal for SystemConstraint<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::< Self, ObjectiveFact<[u8; 0]>, SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>>(Some(self), None, None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
}

impl ManifoldGatewaySignal for gcso_edbc_state_t {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::< Self, ObjectiveFact<[u8; 0]>, SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>>(Some(self), None, None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
}

impl ManifoldGatewaySignal for gcso_config_t {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::< Self, ObjectiveFact<[u8; 0]>, SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>>(Some(self), None, None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
}

impl<T: RawPhaseProjectable> ManifoldGatewaySignal for ObjectiveFact<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::< SystemConstraint<()>, Self, SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>>(None, Some(self), None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
}

impl<T: RawPhaseProjectable> ManifoldGatewaySignal for SubjectivePersona<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::< SystemConstraint<()>, ObjectiveFact<[u8; 0]>, Self,
            ContradictionGuard<()>>(None, None, Some(self), None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
}

impl ManifoldGatewaySignal for ContradictionGuard<ContrastivePhaseBloomFilter> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::< SystemConstraint<()>,
            ObjectiveFact<[u8; 0]>, SubjectivePersona<[u8; 0]>, Self>(None, None, None, Some(self))
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
}

impl ManifoldGatewaySignal for ContradictionGuard<()> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::< SystemConstraint<()>,
            ObjectiveFact<[u8; 0]>, SubjectivePersona<[u8; 0]>, Self>(None, None, None, Some(self))
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
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
        synthesize_multi_origin_gateway_frame_with_words::< SystemConstraint<()>, F, P,
            ContradictionGuard<()>>(None, Some(&self.0), Some(&self.1), None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
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
        branchless_max_f32(self.0.causal_anomaly_severity(), branchless_max_f32(self.1.causal_anomaly_severity(),
                self.2.causal_anomaly_severity()))
    }
}

impl<S: SystemConstraintSignal, F: ObjectiveFactSignal, P: SubjectivePersonaSignal>
    ManifoldGatewaySignal for (S, F, P)
{
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<S, F, P, ContradictionGuard<()>>(Some(&self.0),
            Some(&self.1), Some(&self.2), None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
}

impl< S: SystemConstraintSignal, F: ObjectiveFactSignal, P: SubjectivePersonaSignal,
        G: ContradictionGuardSignal, > CausalProvenanceSignal for (S, F, P, G)
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
        branchless_max_f32(branchless_max_f32(self.0.causal_anomaly_severity(), self.1.causal_anomaly_severity(),
            ), branchless_max_f32(self.2.causal_anomaly_severity(), self.3.causal_anomaly_severity()))
    }
}

impl< S: SystemConstraintSignal, F: ObjectiveFactSignal, P: SubjectivePersonaSignal,
        G: ContradictionGuardSignal, > ManifoldGatewaySignal for (S, F, P, G)
{
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        synthesize_multi_origin_gateway_frame_with_words::<S, F, P, G>(Some(&self.0), Some(&self.1), Some(&self.2),
            Some(&self.3))
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
}

impl<T: ManifoldGatewaySignal> CausalProvenanceSignal for Option<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        match self { Some(inner) => inner.causal_provenance_bits(), None => 0, }
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        match self { Some(inner) => inner.causal_anomaly_severity(), None => 0.0, }
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
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
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
    fn synthesize_gateway_frame(&self) -> GatewayFrame { T::synthesize_gateway_slice_with_words(self).0 }
}

impl<T: ManifoldGatewaySignal, const N: usize> CausalProvenanceSignal for [T; N] {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 { self.as_slice().causal_provenance_bits() }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 { self.as_slice().causal_anomaly_severity() }
}

impl<T: ManifoldGatewaySignal, const N: usize> ManifoldGatewaySignal for [T; N] {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) { self.as_slice().synthesize_gateway_frame_with_words() }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.as_slice().synthesize_gateway_frame_with_words().0 }
}

// ===================================================================
// 8. `.gcso` Crystallized Attractor Record & Online Toroidal $k$-Means / DP-Means
// ===================================================================

/// 128-byte (`#[repr(C, align(64))]`) [`Pod`] record representing a bottom-up crystallized
/// macro attractor centroid on $SO(2)^{64}$ persisted inside `.gcso` (ZIMMS) containers.
#[repr(C, align(64))] #[derive(Debug, Clone, Copy, PartialEq)] pub struct CrystallizedAttractorRecord {
    /// Format magic identifier (`GCSO_CRYSTAL_RECORD_MAGIC` = `0x4C545343` `"CSTL"`, `offset 0..4`).
    pub magic: u32,
    /// Record ABI version (`GCSO_CRYSTAL_RECORD_VERSION` = `0x00000101`, `offset 4..8`).
    pub version: u32,
    /// Full 32-bit macro attractor anchor ID on the Single Unified Attractor Manifold (`offset 8..12`).
    pub anchor_id: u32,
    /// Number of coherent sample points absorbed into this centroid (`offset 12..16`).
    pub sample_count: u32,
    /// Canonical 48-bit `.gcso` (ZIMMS) section offset (`bits 0..=47`, `offset 16..24`).
    pub zimms_section_offset48: u64,
    /// Accumulated base gravitational energy $E_{\mathrm{base}} \ge 0$ (`offset 24..28`).
    pub base_energy: f32,
    /// Circular mean resultant length $\bar{R}_c \in [0, 1]$ across the 64 heads (`offset 28..32`).
    pub resultant_r_bar: f32,
    /// von Mises concentration parameter $\kappa_c \ge 0$ of the crystallized centroid (`offset 32..36`).
    pub von_mises_kappa: f32,
    /// Phase-Proximity Gravity mass $M_{\mathrm{grav}} = E_{\mathrm{base}} \cdot \bar{R}_c (1 + 0.1 \ln(1 + N))$ (`offset 36..40`).
    pub phase_gravity_mass: f32,
    /// Effective RIPA clamp limit in radians ($\theta_{\max}$, `offset 40..44`).
    pub ripa_clamp_rad: f32,
    /// Effective multiplicative inverse-temperature scale $\beta_{\mathrm{eff}}$ (`offset 44..48`).
    pub effective_inv_temp: f32,
    /// Accumulated causal provenance bitmask (`CAUSAL_AXIS_* | CAUSAL_CAUSE_*`, `offset 48..52`).
    pub provenance_bits: u32,
    /// IEEE 802.3 CRC32 checksum computed over bytes `[0..52]` and `[56..128]` (`offset 52..56`).
    pub checksum_crc32: u32,
    /// Dominant 3-bit working lane ID (`ATTRACTOR_SLOT_*`, `0..=7`, `offset 56`).
    pub dominant_slot: u8,
    /// Resolved 5-bit `ACTION_MODE_*` execution flags (`offset 57`).
    pub action_mode_flags: u8,
    /// Active 8-cell Micro-Swarm mask (`0x00..=0xFF`, `offset 58`).
    pub active_swarm_mask: u8,
    /// Primary root-cause classification code (`ROOT_CAUSE_*`, `offset 59`).
    pub dominant_cause_code: u8,
    /// Reserved 32-bit zero field aligning `phase_profile` to exact byte offset `64` (`offset 60..64`).
    pub reserved_align64: u32,
    /// Crystallized 64-head Q7 centroid phase profile on $SO(2)^{64}$ (`64` bytes at `offset 64..128`).
    pub phase_profile: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

// SAFETY: `CrystallizedAttractorRecord` is `#[repr(C, align(64))]` of exact size 128 bytes with zero padding.
unsafe impl Pod for CrystallizedAttractorRecord {}

impl Default for CrystallizedAttractorRecord {
    #[inline(always)]
    fn default() -> Self { Self::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0) }
}

impl CrystallizedAttractorRecord {
    /// Borrows `self.phase_profile` (`offset 64` of `#[repr(C, align(64))]` [`Self`], guaranteed
    /// 64-byte aligned) as a zero-copy `&[u64; 8]` SWAR reference on little-endian targets,
    /// eliminating 64-byte stack copies and 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF) stalls.
    /// Borrows `self.phase_profile` (`offset 64` of `#[repr(C, align(64))]`
    /// [`CrystallizedAttractorRecord`], guaranteed 64-byte aligned) as a zero-copy `&[u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_ref(&self) -> &[u64; NUM_SWARM_MICRO_CELLS] {
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: `CrystallizedAttractorRecord` is `#[repr(C, align(64))]` and `phase_profile`
        // is at exact byte offset 64 (`64 % 8 == 0`), with identical 64-byte layout to `[u64; 8]`.
        unsafe { &*(self.phase_profile.as_ptr().cast::<[u64; NUM_SWARM_MICRO_CELLS]>()) }
    }

    /// Mutably borrows `self.phase_profile` (`offset 64` of `#[repr(C, align(64))]`
    /// [`CrystallizedAttractorRecord`], guaranteed 64-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_mut(&mut self) -> &mut [u64; NUM_SWARM_MICRO_CELLS] {
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: `CrystallizedAttractorRecord` is `#[repr(C, align(64))]` and `phase_profile`
        // is at exact byte offset 64 (`64 % 8 == 0`), with identical 64-byte layout to `[u64; 8]`.
        unsafe { &mut *(self.phase_profile.as_mut_ptr().cast::<[u64; NUM_SWARM_MICRO_CELLS]>()) }
    }

    /// Restores `self` from `base` using scalar field stores for the first 64-byte cache line
    /// (`offset 0..64`) and 64-bit/SIMD word stores for `phase_profile` (`offset 64..128`),
    /// avoiding 256-bit YMM store-to-scalar-load forwarding (STLF) pipeline stalls when
    /// `self` is immediately inspected by scalar header loads in hot loops.
    #[inline(always)]
    pub fn restore_from(&mut self, base: &Self) {
        self.magic = base.magic;
        self.version = base.version;
        self.anchor_id = base.anchor_id;
        self.sample_count = base.sample_count;
        self.zimms_section_offset48 = base.zimms_section_offset48;
        self.base_energy = base.base_energy;
        self.resultant_r_bar = base.resultant_r_bar;
        self.von_mises_kappa = base.von_mises_kappa;
        self.phase_gravity_mass = base.phase_gravity_mass;
        self.ripa_clamp_rad = base.ripa_clamp_rad;
        self.effective_inv_temp = base.effective_inv_temp;
        self.provenance_bits = base.provenance_bits;
        self.checksum_crc32 = base.checksum_crc32;
        self.dominant_slot = base.dominant_slot;
        self.action_mode_flags = base.action_mode_flags;
        self.active_swarm_mask = base.active_swarm_mask;
        self.dominant_cause_code = base.dominant_cause_code;
        *self.phase_words_mut() = *base.phase_words_ref();
    }

    #[inline(always)]
    fn clear_empty_in_place(&mut self, dominant_slot: u8) {
        let slot = dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        self.magic = GCSO_CRYSTAL_RECORD_MAGIC;
        self.version = GCSO_CRYSTAL_RECORD_VERSION;
        self.anchor_id = 0;
        self.sample_count = 0;
        self.zimms_section_offset48 = 0;
        self.base_energy = 0.0;
        self.resultant_r_bar = 0.0;
        self.von_mises_kappa = 0.0;
        self.phase_gravity_mass = 0.0;
        self.ripa_clamp_rad = DEFAULT_RIPA_CLAMP_MAX_RAD;
        self.effective_inv_temp = 1.0;
        self.provenance_bits = 0;
        self.dominant_slot = slot;
        self.action_mode_flags = ACTION_PRESET_PASSIVE_SIDECAR;
        self.active_swarm_mask = 0xFF;
        self.dominant_cause_code = ROOT_CAUSE_NOMINAL;
        self.reserved_align64 = 0;
        store_q7_words_64(&mut self.phase_profile, &[0u64; NUM_SWARM_MICRO_CELLS]);
        self.checksum_crc32 = Self::EMPTY_CRYSTAL_CRC32_BY_SLOT[slot as usize];
    }

    #[inline(always)] #[must_use]
    const fn new_unstamped(anchor_id: u32, dominant_slot: u8, zimms_section_offset48: u64) -> Self {
        Self {
            magic: GCSO_CRYSTAL_RECORD_MAGIC, version: GCSO_CRYSTAL_RECORD_VERSION, anchor_id, sample_count: 0,
            zimms_section_offset48: zimms_section_offset48 & gcso_tagged_ptr_t::ADDR_MASK, base_energy: 0.0,
            resultant_r_bar: 0.0, von_mises_kappa: 0.0,
            phase_gravity_mass: 0.0, ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            effective_inv_temp: 1.0, provenance_bits: 0,
            checksum_crc32: 0, dominant_slot: dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR, active_swarm_mask: 0xFF,
            dominant_cause_code: ROOT_CAUSE_NOMINAL, reserved_align64: 0, phase_profile: [0i8; MAX_ATTENTION_HEADS],
        }
    }

    #[inline(always)]
    #[must_use]
    const fn pack_w7_lo(&self) -> u32 {
        u32::from_le_bytes([
            self.dominant_slot,
            self.action_mode_flags,
            self.active_swarm_mask,
            self.dominant_cause_code,
        ])
    }

    #[inline(always)]
    #[must_use]
    const fn compute_payload_crc32_with_words_raw(
        &self,
        phase_words: &[u64; NUM_SWARM_MICRO_CELLS],
    ) -> u32 {
        let w2 = self.zimms_section_offset48;
        let be_bits = self.base_energy.to_bits();
        let rb_bits = self.resultant_r_bar.to_bits();
        let kp_bits = self.von_mises_kappa.to_bits();
        let gm_bits = self.phase_gravity_mass.to_bits();
        let cl_bits = self.ripa_clamp_rad.to_bits();
        let tm_bits = self.effective_inv_temp.to_bits();
        let lut = CRC32_IEEE_SLICE16_REF;
        let mut final_acc = 0u32;

        let c0 = if (self.magic == GCSO_CRYSTAL_RECORD_MAGIC)
            & (self.version == GCSO_CRYSTAL_RECORD_VERSION)
        {
            final_acc ^= CRYSTAL_CRC32_W0_FINAL_CONTRIB;
            if ((self.anchor_id >> 16) == 0) & ((self.sample_count >> 8) == 0) {
                final_acc ^= CRC32_CRYSTAL_AID_BYTE0_REF[(self.anchor_id & 0xFF) as usize]
                    ^ CRC32_CRYSTAL_AID_BYTE1_REF[((self.anchor_id >> 8) & 0xFF) as usize]
                    ^ CRC32_CRYSTAL_SC_BYTE0_REF[self.sample_count as usize];
                0
            } else {
                let aid_c = if (self.anchor_id >> 8) == 0 {
                    lut[7][self.anchor_id as usize]
                } else {
                    crc32_slice16_indep_lo1_u32(self.anchor_id)
                };
                let sc_c = if (self.sample_count >> 8) == 0 {
                    lut[3][self.sample_count as usize]
                } else {
                    crc32_slice16_indep_hi1_u32(self.sample_count)
                };
                aid_c ^ sc_c
            }
        } else {
            let w0 = (self.magic as u64) | ((self.version as u64) << 32);
            let w1 = (self.anchor_id as u64) | ((self.sample_count as u64) << 32);
            crc32_slice16_step_u64x2(0xFFFF_FFFFu32, w0, w1)
        };

        if be_bits != 0 {
            final_acc ^= crc32_crystal_be24_final_u32(be_bits);
        }
        if gm_bits != 0 {
            final_acc ^= crc32_crystal_gm36_final_u32(gm_bits);
        }

        let w2_lo = (w2 as u32) ^ c0;
        let w2_hi = (w2 >> 32) as u32;
        let w2_c = if (w2_lo | w2_hi) == 0 {
            0
        } else if (c0 == 0) & (w2_hi == 0) & ((w2_lo & 0xFF) == 0) {
            final_acc ^= CRC32_CRYSTAL_ZOFF_BYTE1_REF[((w2_lo >> 8) & 0xFF) as usize]
                ^ CRC32_CRYSTAL_ZOFF_BYTE2_REF[((w2_lo >> 16) & 0xFF) as usize]
                ^ CRC32_CRYSTAL_ZOFF_BYTE3_REF[(w2_lo >> 24) as usize];
            0
        } else if w2_hi == 0 {
            crc32_slice16_dep_lo0(w2_lo)
        } else {
            crc32_slice16_dep_lo0(w2_lo) ^ crc32_slice16_indep_hi0_u32(w2_hi)
        };
        let rb_c = if rb_bits == 0x3F80_0000u32 {
            final_acc ^= CRYSTAL_CRC32_RBAR_ONE_FINAL_CONTRIB;
            0
        } else if rb_bits == 0 {
            0
        } else {
            crc32_slice16_indep_hi1_u32(rb_bits)
        };
        let c1 = w2_c ^ rb_c;
        if c1 != 0 {
            final_acc ^= crc32_step_l6(c1);
        }

        let kp_c = if kp_bits == VON_MISES_KAPPA_AT_MAX_R_BITS {
            final_acc ^= CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB;
            0
        } else if kp_bits == 0x4248_0000u32 {
            final_acc ^= CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB;
            0
        } else if kp_bits == 0 {
            0
        } else {
            crc32_slice16_dep_lo0(kp_bits)
        };
        const PERSONA_W5_FINAL_CONTRIB: u32 = CRYSTAL_CRC32_DEFAULT_W5_FINAL_CONTRIB
            ^ crc32_step_l5(crc32_slice16_indep_lo1_u32(
                DEFAULT_PERSONA_RIPA_CLAMP_RAD.to_bits() ^ DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits(),
            ));
        const FACT_W5_FINAL_CONTRIB: u32 = CRYSTAL_CRC32_DEFAULT_W5_FINAL_CONTRIB
            ^ crc32_step_l5(crc32_slice16_indep_lo1_u32(
                DEFAULT_FACT_RIPA_CLAMP_RAD.to_bits() ^ DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits(),
            ));
        let w5_c = if tm_bits == 0x3F80_0000u32 {
            if cl_bits == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits() {
                final_acc ^= CRYSTAL_CRC32_DEFAULT_W5_FINAL_CONTRIB;
                0
            } else if cl_bits == DEFAULT_PERSONA_RIPA_CLAMP_RAD.to_bits() {
                final_acc ^= PERSONA_W5_FINAL_CONTRIB;
                0
            } else if cl_bits == DEFAULT_FACT_RIPA_CLAMP_RAD.to_bits() {
                final_acc ^= FACT_W5_FINAL_CONTRIB;
                0
            } else {
                crc32_slice16_indep_lo1_u32(cl_bits) ^ crc32_slice16_indep_hi1_u32(tm_bits)
            }
        } else {
            crc32_slice16_indep_lo1_u32(cl_bits) ^ crc32_slice16_indep_hi1_u32(tm_bits)
        };
        let c2 = kp_c ^ w5_c;
        if c2 != 0 {
            final_acc ^= crc32_step_l5(c2);
        }

        let prov_c = if (self.provenance_bits >> 16) == 0 {
            final_acc ^= CRC32_CRYSTAL_PROV_BYTE0_REF[(self.provenance_bits & 0xFF) as usize]
                ^ CRC32_CRYSTAL_PROV_BYTE1_REF[((self.provenance_bits >> 8) & 0xFF) as usize];
            0
        } else {
            crc32_slice16_dep_lo0(self.provenance_bits)
        };
        const DEFAULT_W7_MASK_CAUSE_CRC_C3: u32 =
            CRC32_IEEE_SLICE16_REF[5][0xFF] ^ CRC32_IEEE_SLICE16_REF[4][ROOT_CAUSE_NOMINAL as usize];
        const DEFAULT_W7_MASK_CAUSE_FINAL_CONTRIB: u32 = crc32_step_l4(DEFAULT_W7_MASK_CAUSE_CRC_C3);
        let slot_fast = (self.dominant_slot & 0xF8) == 0;
        let mask_cause_default =
            (self.active_swarm_mask == 0xFF) & (self.dominant_cause_code == ROOT_CAUSE_NOMINAL);
        let w7_c = if slot_fast & mask_cause_default {
            final_acc ^= CRC32_CRYSTAL_SLOT_BYTE56_LUT_8[(self.dominant_slot & 7) as usize]
                ^ CRC32_CRYSTAL_ACTION_BYTE57_REF[self.action_mode_flags as usize]
                ^ DEFAULT_W7_MASK_CAUSE_FINAL_CONTRIB;
            0
        } else {
            (lut[7][self.dominant_slot as usize] ^ lut[6][self.action_mode_flags as usize])
                ^ if mask_cause_default {
                    DEFAULT_W7_MASK_CAUSE_CRC_C3
                } else {
                    lut[5][self.active_swarm_mask as usize]
                        ^ lut[4][self.dominant_cause_code as usize]
                }
        };
        let res_c = if self.reserved_align64 == 0 {
            0
        } else {
            crc32_slice16_indep_hi1_u32(self.reserved_align64)
        };
        let c3 = prov_c ^ w7_c ^ res_c;

        let block1_crc = crc32_step_block64_words(c3, phase_words);
        !(final_acc ^ block1_crc)
    }

    const EMPTY_CRYSTAL_CRC32_BY_SLOT: [u32; 8] = [
        Self::new_unstamped(0, 0, 0).compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 1, 0).compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 2, 0).compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 3, 0).compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 4, 0).compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 5, 0).compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 6, 0).compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 7, 0).compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
    ];

    /// Creates an empty, CRC32-stamped [`CrystallizedAttractorRecord`] ready to absorb sample points.
    #[inline] #[must_use]
    pub const fn new_empty(anchor_id: u32, dominant_slot: u8, zimms_section_offset48: u64) -> Self {
        let mut rec = Self::new_unstamped(anchor_id, dominant_slot, zimms_section_offset48);
        let zoff = rec.zimms_section_offset48;
        let base_crc =
            Self::EMPTY_CRYSTAL_CRC32_BY_SLOT[(dominant_slot & ATTRACTOR_SLOT_ID_MASK) as usize];
        if zoff == 0 {
            if anchor_id == 0 {
                rec.checksum_crc32 = base_crc;
            } else if anchor_id < 256 {
                rec.checksum_crc32 = base_crc ^ CRC32_CRYSTAL_AID_BYTE0_REF[anchor_id as usize];
            } else if anchor_id < 65536 {
                rec.checksum_crc32 = base_crc
                    ^ CRC32_CRYSTAL_AID_BYTE0_REF[(anchor_id & 0xFF) as usize]
                    ^ CRC32_CRYSTAL_AID_BYTE1_REF[((anchor_id >> 8) & 0xFF) as usize];
            } else {
                let c0 = crc32_slice16_indep_lo1_u32(anchor_id);
                rec.checksum_crc32 = base_crc ^ crc32_step_l4(crc32_step_l3(c0));
            }
        } else if ((zoff >> 32) == 0) & ((anchor_id >> 16) == 0) {
            let z_lo = zoff as u32;
            let z_crc = if (z_lo & 0xFFFF) == 0 {
                CRC32_CRYSTAL_ZOFF_BYTE2_REF[((z_lo >> 16) & 0xFF) as usize]
                    ^ CRC32_CRYSTAL_ZOFF_BYTE3_REF[(z_lo >> 24) as usize]
            } else {
                crc32_crystal_zoff16_final_u32(z_lo)
            };
            rec.checksum_crc32 = base_crc
                ^ CRC32_CRYSTAL_AID_BYTE0_REF[(anchor_id & 0xFF) as usize]
                ^ CRC32_CRYSTAL_AID_BYTE1_REF[((anchor_id >> 8) & 0xFF) as usize]
                ^ z_crc;
        } else {
            let c0 = if anchor_id == 0 {
                0
            } else if anchor_id < 256 {
                CRC32_IEEE_SLICE16_REF[7][anchor_id as usize]
            } else {
                crc32_slice16_indep_lo1_u32(anchor_id)
            };
            let w2_lo = (zoff as u32) ^ c0;
            let w2_hi = (zoff >> 32) as u32;
            let c1 = if (w2_hi == 0) & ((w2_lo & 0xFFFF) == 0) {
                CRC32_IEEE_SLICE16_REF[13][((w2_lo >> 16) & 0xFF) as usize]
                    ^ CRC32_IEEE_SLICE16_REF[12][(w2_lo >> 24) as usize]
            } else if w2_hi == 0 {
                crc32_slice16_dep_lo0(w2_lo)
            } else {
                crc32_slice16_dep_lo0(w2_lo) ^ crc32_slice16_indep_hi0_u32(w2_hi)
            };
            rec.checksum_crc32 = base_crc ^ crc32_step_l6(c1);
        }
        rec
    }

    /// Computes the IEEE 802.3 CRC32 checksum over this 128-byte record using L1D-resident
    /// `CRC32_IEEE_SLICE16_REF` block folding (`c0 -> c1 -> c2 -> c3`) fused directly into
    /// `crc32_step_block64_words(c3, phase_words)` (`0` extra 4KB `FINAL_LUT` table accesses and
    /// `0` `crc32_step_l4(c3)` lookups on non-uniform records).
    #[inline(always)] #[must_use]
    pub const fn compute_payload_crc32_with_words(&self, phase_words: &[u64; NUM_SWARM_MICRO_CELLS]) -> u32 {
        self.compute_payload_crc32_with_words_raw(phase_words)
    }

    /// Computes the IEEE 802.3 CRC32 checksum over this 128-byte record (with `checksum_crc32` at
    /// `offset 52..56` treated as zero) using 16 unrolled Slice-by-8 `u64` word steps.
    #[inline]
    #[must_use]
    pub const fn compute_payload_crc32(&self) -> u32 {
        self.compute_payload_crc32_with_words(self.phase_words_ref())
    }

    /// Recomputes and stamps `self.checksum_crc32` in-place using pre-loaded `phase_words`.
    #[inline(always)]
    pub fn stamp_crc32_with_words(&mut self, phase_words: &[u64; NUM_SWARM_MICRO_CELLS]) {
        self.checksum_crc32 = self.compute_payload_crc32_with_words(phase_words);
    }

    /// Recomputes and stamps `self.checksum_crc32` in-place.
    #[inline(always)]
    pub fn stamp_crc32(&mut self) { self.checksum_crc32 = self.compute_payload_crc32(); }

    /// Verifies `magic`, `version`, and the IEEE 802.3 `checksum_crc32` integrity of this record.
    #[inline(always)]
    #[must_use]
    pub const fn verify_integrity(&self) -> bool {
        if (self.magic != GCSO_CRYSTAL_RECORD_MAGIC) | (self.version != GCSO_CRYSTAL_RECORD_VERSION) {
            return false;
        }
        let pw = self.phase_words_ref();
        if self.sample_count == 0
            && ((self.anchor_id
                | self.base_energy.to_bits()
                | self.resultant_r_bar.to_bits()
                | self.von_mises_kappa.to_bits()
                | self.phase_gravity_mass.to_bits()
                | self.provenance_bits
                | self.reserved_align64)
                == 0)
            && (self.zimms_section_offset48 == 0)
            && (self.ripa_clamp_rad.to_bits() == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
            && (self.effective_inv_temp.to_bits() == 0x3F80_0000u32)
            && ((self.dominant_slot & !ATTRACTOR_SLOT_ID_MASK) == 0)
            && (self.action_mode_flags == ACTION_PRESET_PASSIVE_SIDECAR)
            && (self.active_swarm_mask == 0xFF)
            && (self.dominant_cause_code == ROOT_CAUSE_NOMINAL)
            && ((((pw[0] | pw[1]) | (pw[2] | pw[3])) | ((pw[4] | pw[5]) | (pw[6] | pw[7]))) == 0)
        {
            return self.checksum_crc32 == Self::EMPTY_CRYSTAL_CRC32_BY_SLOT[self.dominant_slot as usize];
        }
        self.checksum_crc32 == self.compute_payload_crc32_with_words(pw)
    }

    /// Internal register-resident M-step helper accepting the already-loaded `cur_w` in registers
    /// and an optional `precomputed_coh_opt` from the E-step to eliminate duplicate 64-head coherence evaluation.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn absorb_sample_words_with_cur_coh_and_uni_unstamped(&mut self,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS], sample_w: &[u64; NUM_SWARM_MICRO_CELLS],
        precomputed_coh_opt: Option<f32>, uni_flags_opt: Option<(bool, bool)>,
        sample_energy: f32, add_count: u32,
    ) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, bool) {
        let clean_e = branchless_flush_subnormal_f32(branchless_relu_f32(sample_energy));
        if clean_e <= 1.0e-12 || add_count == 0 { return (*cur_w, self.phase_gravity_mass, false); }

        let prev_count = self.sample_count;
        let prev_be_bits = self.base_energy.to_bits();
        let prev_rb_bits = self.resultant_r_bar.to_bits();
        let (c_uni, sample_uni) = match uni_flags_opt {
            Some(pair) => pair,
            None => (
                is_word_array_uniform_64(cur_w),
                is_word_array_uniform_64(sample_w),
            ),
        };
        let (next_w, raw_next_e, r_bar, kappa, words_changed) =
            branchless_toroidal_centroid_step_words_with_coh_uni_and_changed_64(
                cur_w,
                prev_count,
                self.base_energy,
                self.resultant_r_bar,
                self.von_mises_kappa,
                sample_w,
                precomputed_coh_opt,
                clean_e,
                c_uni,
                sample_uni,
            );
        if words_changed {
            *self.phase_words_mut() = next_w;
        }

        let next_e = branchless_min_f32(raw_next_e, MAX_ATTRACTOR_PULL_FORCE * 4.0);
        let next_count = prev_count.saturating_add(add_count);
        self.sample_count = next_count;
        self.base_energy = next_e;
        self.resultant_r_bar = r_bar;
        self.von_mises_kappa = kappa;

        let grav_mass = if (!words_changed)
            & (next_e.to_bits() == prev_be_bits)
            & (prev_count >= 8)
            & (r_bar.to_bits() == prev_rb_bits)
        {
            self.phase_gravity_mass
        } else {
            let maturity_scale = branchless_maturity_scale_u32(next_count);
            let gm = branchless_flush_subnormal_f32(
                next_e * branchless_clamp_f32(r_bar, 0.10, 1.0) * maturity_scale,
            );
            self.phase_gravity_mass = gm;
            gm
        };
        if self.action_mode_flags == ACTION_PRESET_PASSIVE_SIDECAR {
            self.action_mode_flags = ACTION_PRESET_EXPRESS_PHASE;
        }

        if (next_count >= 2) & (r_bar >= 0.45) { self.provenance_bits |= CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED; }
        (next_w, grav_mass, words_changed)
    }

    #[inline(always)]
    fn absorb_sample_words_with_cur_unstamped(&mut self,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS], sample_w: &[u64; NUM_SWARM_MICRO_CELLS],
        sample_energy: f32, add_count: u32) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, bool) {
        self.absorb_sample_words_with_cur_coh_and_uni_unstamped(cur_w, sample_w, None, None, sample_energy, add_count)
    }

    #[inline(always)]
    fn absorb_gateway_frame_with_cur_coh_and_uni_unstamped(&mut self,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS], frame: &GatewayFrame,
        frame_words: &[u64; NUM_SWARM_MICRO_CELLS], precomputed_coh_opt: Option<f32>,
        uni_flags_opt: Option<(bool, bool)>,
    ) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, bool, bool) {
        if frame.is_repulsed() { return (*cur_w, self.phase_gravity_mass, false, false); }

        if self.sample_count == 0 {
            if frame.target_anchor_id != 0 { self.anchor_id = frame.target_anchor_id; }
            if frame.zimms_section_offset48 != 0 { self.zimms_section_offset48 = frame.zimms_section_offset48; }
            self.dominant_slot = frame.dominant_slot;
            self.ripa_clamp_rad = frame.ripa_clamp_rad;
            self.effective_inv_temp = frame.effective_inv_temp;
        } else {
            if self.ripa_clamp_rad.to_bits() != frame.ripa_clamp_rad.to_bits() {
                self.ripa_clamp_rad = branchless_lerp_f32(self.ripa_clamp_rad, frame.ripa_clamp_rad, 0.25);
            }
            if self.effective_inv_temp.to_bits() != frame.effective_inv_temp.to_bits() {
                self.effective_inv_temp = branchless_lerp_f32(self.effective_inv_temp, frame.effective_inv_temp, 0.25);
            }
        }
        self.action_mode_flags = frame.action_mode_flags;
        self.active_swarm_mask = frame.active_swarm_cell_mask;
        self.provenance_bits |= frame.provenance_bits;

        let (next_w, mass, words_changed) = self.absorb_sample_words_with_cur_coh_and_uni_unstamped(
            cur_w,
            frame_words,
            precomputed_coh_opt,
            uni_flags_opt,
            frame.effective_energy,
            1,
        );
        (next_w, mass, true, words_changed)
    }

    #[inline(always)]
    fn absorb_gateway_frame_with_cur_unstamped(&mut self,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS], frame: &GatewayFrame,
        frame_words: &[u64; NUM_SWARM_MICRO_CELLS]) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, bool, bool) {
        self.absorb_gateway_frame_with_cur_coh_and_uni_unstamped(cur_w, frame, frame_words, None, None)
    }

    #[inline(always)]
    fn stamp_or_update_absorption_crc32(
        &mut self,
        prev_crc: u32,
        prev_aid: u32,
        prev_count: u32,
        prev_zoff: u64,
        prev_be_bits: u32,
        prev_rb_bits: u32,
        prev_kp_bits: u32,
        prev_gm_bits: u32,
        prev_clamp_bits: u32,
        prev_temp_bits: u32,
        prev_prov: u32,
        prev_w7_lo: u32,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS],
        next_w: &[u64; NUM_SWARM_MICRO_CELLS],
        words_changed: bool,
    ) {
        if prev_crc != 0 {
            let delta_aid = prev_aid ^ self.anchor_id;
            let delta_zoff = prev_zoff ^ self.zimms_section_offset48;
            let delta_rb = prev_rb_bits ^ self.resultant_r_bar.to_bits();
            let delta_kp = prev_kp_bits ^ self.von_mises_kappa.to_bits();
            let delta_cl = prev_clamp_bits ^ self.ripa_clamp_rad.to_bits();
            let delta_tm = prev_temp_bits ^ self.effective_inv_temp.to_bits();
            let delta_w7 = prev_w7_lo ^ self.pack_w7_lo();
            if (!words_changed)
                & (delta_aid == 0)
                & (delta_zoff == 0)
                & ((delta_rb | delta_kp) == 0)
                & ((delta_w7 >> 16) == 0)
                & ((delta_w7 & 0xF8) == 0)
            {
                self.checksum_crc32 = crc32_update_crystal_stationary_fields_delta(
                    prev_crc,
                    prev_count ^ self.sample_count,
                    prev_be_bits ^ self.base_energy.to_bits(),
                    prev_gm_bits ^ self.phase_gravity_mass.to_bits(),
                    delta_cl,
                    delta_tm,
                    prev_prov ^ self.provenance_bits,
                    delta_w7,
                );
            } else {
                self.checksum_crc32 = crc32_update_crystal_absorption_delta(
                    prev_crc,
                    delta_aid,
                    prev_count ^ self.sample_count,
                    delta_zoff,
                    prev_be_bits ^ self.base_energy.to_bits(),
                    delta_rb,
                    delta_kp,
                    prev_gm_bits ^ self.phase_gravity_mass.to_bits(),
                    delta_cl,
                    delta_tm,
                    prev_prov ^ self.provenance_bits,
                    delta_w7,
                    cur_w,
                    next_w,
                );
            }
        } else {
            self.stamp_crc32_with_words(next_w);
        }
    }

    /// Absorbs a 64-head sample point (`sample_phases`, `sample_energy`) into this centroid on
    /// $SO(2)^{64}$, updates $(\bar{R}_c, \kappa_c, M_{\mathrm{grav}})$, and stamps `checksum_crc32`.
    #[inline]
    pub fn absorb_sample_point_64(&mut self,
        sample_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS], sample_energy: f32) -> f32 {
        let prev_crc = self.checksum_crc32;
        let prev_count = self.sample_count;
        let prev_be_bits = self.base_energy.to_bits();
        let prev_rb_bits = self.resultant_r_bar.to_bits();
        let prev_kp_bits = self.von_mises_kappa.to_bits();
        let prev_gm_bits = self.phase_gravity_mass.to_bits();
        let prev_clamp_bits = self.ripa_clamp_rad.to_bits();
        let prev_temp_bits = self.effective_inv_temp.to_bits();
        let prev_prov = self.provenance_bits;
        let prev_w7_lo = self.pack_w7_lo();
        let cur_w = *self.phase_words_ref();
        let sample_w = load_q7_words_64(sample_phases);
        let (next_w, mass, words_changed) =
            self.absorb_sample_words_with_cur_unstamped(&cur_w, &sample_w, sample_energy, 1);
        self.stamp_or_update_absorption_crc32(
            prev_crc,
            self.anchor_id,
            prev_count,
            self.zimms_section_offset48,
            prev_be_bits,
            prev_rb_bits,
            prev_kp_bits,
            prev_gm_bits,
            prev_clamp_bits,
            prev_temp_bits,
            prev_prov,
            prev_w7_lo,
            &cur_w,
            &next_w,
            words_changed,
        );
        mass
    }

    /// Absorbs a [`GatewayFrame`] and its pre-loaded 8 `u64` SWAR words (`frame_words`) into this
    /// centroid without reloading `frame.phase_deltas`, stamping `checksum_crc32` once at the end.
    #[inline]
    pub fn absorb_gateway_frame_with_words(&mut self,
        frame: &GatewayFrame, frame_words: &[u64; NUM_SWARM_MICRO_CELLS]) -> f32 {
        if frame.is_repulsed() {
            return self.phase_gravity_mass;
        }
        let prev_crc = self.checksum_crc32;
        let prev_aid = self.anchor_id;
        let prev_count = self.sample_count;
        let prev_zoff = self.zimms_section_offset48;
        let prev_be_bits = self.base_energy.to_bits();
        let prev_rb_bits = self.resultant_r_bar.to_bits();
        let prev_kp_bits = self.von_mises_kappa.to_bits();
        let prev_gm_bits = self.phase_gravity_mass.to_bits();
        let prev_clamp_bits = self.ripa_clamp_rad.to_bits();
        let prev_temp_bits = self.effective_inv_temp.to_bits();
        let prev_prov = self.provenance_bits;
        let prev_w7_lo = self.pack_w7_lo();
        let cur_w = *self.phase_words_ref();
        let (next_w, mass, modified, words_changed) =
            self.absorb_gateway_frame_with_cur_unstamped(&cur_w, frame, frame_words);
        if modified {
            self.stamp_or_update_absorption_crc32(
                prev_crc,
                prev_aid,
                prev_count,
                prev_zoff,
                prev_be_bits,
                prev_rb_bits,
                prev_kp_bits,
                prev_gm_bits,
                prev_clamp_bits,
                prev_temp_bits,
                prev_prov,
                prev_w7_lo,
                &cur_w,
                &next_w,
                words_changed,
            );
        }
        mass
    }

    /// Absorbs a [`GatewayFrame`] and its pre-loaded 8 `u64` SWAR words (`frame_words`) with
    /// explicit evolved `(dominant_slot_override, action_mode_override, extra_provenance_bits)`
    /// without copying a 128-byte [`GatewayFrame`] on the stack.
    #[inline]
    pub fn absorb_gateway_frame_with_words_and_overrides(
        &mut self,
        frame: &GatewayFrame,
        frame_words: &[u64; NUM_SWARM_MICRO_CELLS],
        dominant_slot_override: u8,
        action_mode_override: u8,
        extra_provenance_bits: u32,
    ) -> f32 {
        if frame.is_repulsed() {
            return self.phase_gravity_mass;
        }
        let prev_crc = self.checksum_crc32;
        let prev_aid = self.anchor_id;
        let prev_count = self.sample_count;
        let prev_zoff = self.zimms_section_offset48;
        let prev_be_bits = self.base_energy.to_bits();
        let prev_rb_bits = self.resultant_r_bar.to_bits();
        let prev_kp_bits = self.von_mises_kappa.to_bits();
        let prev_gm_bits = self.phase_gravity_mass.to_bits();
        let prev_clamp_bits = self.ripa_clamp_rad.to_bits();
        let prev_temp_bits = self.effective_inv_temp.to_bits();
        let prev_prov = self.provenance_bits;
        let prev_w7_lo = self.pack_w7_lo();
        let cur_w = *self.phase_words_ref();
        let (next_w, mass, modified, words_changed) =
            self.absorb_gateway_frame_with_cur_unstamped(&cur_w, frame, frame_words);
        if modified {
            self.dominant_slot = dominant_slot_override;
            self.action_mode_flags = action_mode_override;
            self.provenance_bits |= extra_provenance_bits;
            self.stamp_or_update_absorption_crc32(
                prev_crc,
                prev_aid,
                prev_count,
                prev_zoff,
                prev_be_bits,
                prev_rb_bits,
                prev_kp_bits,
                prev_gm_bits,
                prev_clamp_bits,
                prev_temp_bits,
                prev_prov,
                prev_w7_lo,
                &cur_w,
                &next_w,
                words_changed,
            );
        }
        mass
    }

    /// Absorbs a [`GatewayFrame`] into this [`CrystallizedAttractorRecord`] and returns `phase_gravity_mass`.
    #[inline(always)]
    pub fn absorb_gateway_frame(&mut self, frame: &GatewayFrame) -> f32 {
        self.absorb_gateway_frame_with_words(frame, frame.phase_words_ref())
    }

    /// Evaluates **Phase-Proximity Gravity** and cosine phase coherence between `query_phases` and `self`.
    #[inline]
    #[must_use]
    pub fn evaluate_phase_proximity_gravity_64(&self,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> (f32, f32) {
        if self.sample_count == 0 || self.phase_gravity_mass <= 1.0e-12 { return (0.0, 0.0); }
        let coh = branchless_q7_coherence_64(query_phases, &self.phase_profile);
        let grav = branchless_phase_proximity_gravity_f32(self.phase_gravity_mass, coh, DEFAULT_PHASE_GRAVITY_EPS);
        (grav, coh)
    }

    #[inline(always)]
    #[must_use]
    fn find_nearest_toroidal_centroid_with_cache_64(
        centroids: &[Self],
        sample_words: &[u64; NUM_SWARM_MICRO_CELLS],
        word_cache_opt: Option<(
            &[[u64; NUM_SWARM_MICRO_CELLS]; MAX_REGISTER_CACHED_CENTROIDS],
            u16,
            u16,
        )>,
    ) -> (
        usize,
        f32,
        f32,
        Option<usize>,
        usize,
        [u64; NUM_SWARM_MICRO_CELLS],
        bool,
        bool,
        Option<u8>,
    ) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (
                0,
                -1.0,
                0.0,
                None,
                0,
                [0u64; NUM_SWARM_MICRO_CELLS],
                true,
                true,
                None,
            );
        }

        let sample_is_uniform = is_word_array_uniform_64(sample_words);
        let sample_w0 = sample_words[0];
        let sample_b0 = (sample_w0 & 0xFF) as u8;
        let sample_is_byte_uniform =
            sample_is_uniform & (sample_w0 == (sample_b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64));
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (va0, va1) = unsafe {
            super::branchless::avx2_words64_to_vec64_with_uniform(sample_words, sample_is_uniform)
        };
        let mut best_idx = 0usize;
        let mut best_coh = -2.0_f32;
        let mut best_mass = 0.0_f32;
        let mut best_w = [0u64; NUM_SWARM_MICRO_CELLS];
        let mut best_is_uniform = false;
        let mut best_byte_diff: Option<u8> = None;
        let mut first_empty: Option<usize> = None;
        let mut weakest_idx = 0usize;
        let mut weakest_mass = 1.0e30_f32;
        let mut any_active = false;

        let mut i = 0usize;
        while i < k_len {
            let rec = &centroids[i];
            let mass = rec.phase_gravity_mass;
            let is_active = (rec.sample_count > 0) & (mass > 1.0e-12);
            if !is_active {
                if first_empty.is_none() {
                    first_empty = Some(i);
                }
                if 0.0 < weakest_mass {
                    weakest_mass = 0.0;
                    weakest_idx = i;
                }
            } else {
                match word_cache_opt {
                    Some((cache, uni_mask, byte_uni_mask)) => {
                        let cw = &cache[i];
                        let c_uni = ((uni_mask >> i) & 1) != 0;
                        let c_byte_uni = sample_is_byte_uniform & (((byte_uni_mask >> i) & 1) != 0);
                        let (coh, diff_opt) = if c_byte_uni {
                            let cw0 = cw[0];
                            if cw0 == sample_w0 {
                                (1.0, Some(0u8))
                            } else {
                                let d_u8 = sample_b0.wrapping_sub(cw0 as u8);
                                (Q7_FULL_COS_REF[d_u8 as usize], Some(d_u8))
                            }
                        } else if sample_is_uniform & c_uni {
                            let cw0 = cw[0];
                            if cw0 == sample_w0 {
                                (1.0, Some(0u8))
                            } else {
                                (coherence_from_uniform_words_x8(sample_w0, cw0), None)
                            }
                        } else {
                            let floor = if any_active { best_coh } else { -2.0 };
                            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                            let c_val = {
                                let _ = uni_mask;
                                // SAFETY: `cw` is a 64-byte `[u64; 8]` array and AVX2 is enabled.
                                unsafe {
                                    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
                                    let ptr = cw.as_ptr().cast::<__m256i>();
                                    let vb0 = _mm256_loadu_si256(ptr);
                                    let vb1 = _mm256_loadu_si256(ptr.add(1));
                                    super::branchless::avx2_coherence_from_diff_vec64_with_floor(
                                        _mm256_sub_epi8(va0, vb0),
                                        _mm256_sub_epi8(va1, vb1),
                                        floor,
                                    )
                                }
                            };
                            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                            let c_val = {
                                let b_uni = sample_is_uniform & c_uni;
                                coherence_above_floor_from_word_arrays_64(
                                    sample_words,
                                    sample_is_uniform,
                                    cw,
                                    b_uni,
                                    floor,
                                )
                            };
                            (c_val, None)
                        };
                        let take = (!any_active)
                            | (coh > best_coh)
                            | ((coh == best_coh) & (mass > best_mass));
                        if take {
                            best_idx = i;
                            best_coh = coh;
                            best_mass = mass;
                            best_is_uniform = c_uni;
                            best_byte_diff = diff_opt;
                        }
                    }
                    None => {
                        let rec_w = rec.phase_words_ref();
                        let rec_uni = if sample_is_uniform {
                            is_word_array_uniform_64(rec_w)
                        } else {
                            false
                        };
                        let (coh, diff_opt) = if rec_uni {
                            let rw0 = rec_w[0];
                            let rb0 = (rw0 & 0xFF) as u8;
                            if rw0 == sample_w0 {
                                (1.0, Some(0u8))
                            } else if sample_is_byte_uniform
                                && (rw0 == (rb0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64))
                            {
                                let d_u8 = sample_b0.wrapping_sub(rb0);
                                (Q7_FULL_COS_REF[d_u8 as usize], Some(d_u8))
                            } else {
                                (coherence_from_uniform_words_x8(sample_w0, rw0), None)
                            }
                        } else {
                            let floor = if any_active { best_coh } else { -2.0 };
                            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                            let c_val = {
                                // SAFETY: `rec.phase_profile` is 64-byte aligned and AVX2 is enabled.
                                unsafe {
                                    use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
                                    let ptr = rec.phase_profile.as_ptr().cast::<__m256i>();
                                    let vb0 = _mm256_loadu_si256(ptr);
                                    let vb1 = _mm256_loadu_si256(ptr.add(1));
                                    super::branchless::avx2_coherence_from_diff_vec64_with_floor(
                                        _mm256_sub_epi8(va0, vb0),
                                        _mm256_sub_epi8(va1, vb1),
                                        floor,
                                    )
                                }
                            };
                            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                            let c_val = coherence_above_floor_from_word_arrays_64(
                                sample_words,
                                sample_is_uniform,
                                rec_w,
                                false,
                                floor,
                            );
                            (c_val, None)
                        };
                        let take = (!any_active)
                            | (coh > best_coh)
                            | ((coh == best_coh) & (mass > best_mass));
                        if take {
                            best_idx = i;
                            best_coh = coh;
                            best_mass = mass;
                            best_is_uniform = rec_uni;
                            best_byte_diff = diff_opt;
                        }
                    }
                }
                any_active = true;

                let is_weaker = mass < weakest_mass;
                weakest_idx = branchless_select_usize(is_weaker, i, weakest_idx);
                weakest_mass = branchless_select_f32(is_weaker, mass, weakest_mass);
            }
            i += 1;
        }

        if any_active {
            best_w = match word_cache_opt {
                Some((cache, _, _)) => cache[best_idx],
                None => *centroids[best_idx].phase_words_ref(),
            };
        } else {
            best_coh = 0.0;
        }
        (
            best_idx,
            best_coh,
            best_mass,
            first_empty,
            weakest_idx,
            best_w,
            sample_is_uniform,
            best_is_uniform,
            best_byte_diff,
        )
    }

    #[inline(always)]
    #[must_use]
    fn find_nearest_toroidal_centroid_index_with_cache_64(
        centroids: &[Self],
        sample_words: &[u64; NUM_SWARM_MICRO_CELLS],
        word_cache_opt: Option<(
            &[[u64; NUM_SWARM_MICRO_CELLS]; MAX_REGISTER_CACHED_CENTROIDS],
            u16,
            u16,
        )>,
    ) -> (usize, f32, f32, Option<usize>, usize, bool) {
        let (
            best_idx,
            best_coh,
            best_mass,
            first_empty,
            weakest_idx,
            _best_w,
            _sample_uni,
            _best_uni,
            best_byte_diff,
        ) = Self::find_nearest_toroidal_centroid_with_cache_64(
            centroids,
            sample_words,
            word_cache_opt,
        );
        let any_active = !centroids.is_empty()
            && (centroids[best_idx].sample_count > 0)
            && (best_mass > 1.0e-12);
        let best_grav = if any_active {
            match best_byte_diff {
                Some(diff_u8) => branchless_q7_default_gravity_from_diff_u8(best_mass, diff_u8),
                None => branchless_phase_proximity_gravity_f32(
                    best_mass,
                    best_coh,
                    DEFAULT_PHASE_GRAVITY_EPS,
                ),
            }
        } else {
            0.0
        };
        (
            best_idx,
            best_coh,
            best_grav,
            first_empty,
            weakest_idx,
            any_active,
        )
    }

    #[inline(always)]
    #[must_use]
    fn find_nearest_toroidal_centroid_words_and_profile_64(
        centroids: &[Self],
        sample_words: &[u64; NUM_SWARM_MICRO_CELLS],
    ) -> (
        usize,
        f32,
        f32,
        Option<usize>,
        usize,
        [u64; NUM_SWARM_MICRO_CELLS],
    ) {
        let (best_idx, best_coh, best_mass, first_empty, weakest_idx, best_w, _, _, _) =
            Self::find_nearest_toroidal_centroid_with_cache_64(centroids, sample_words, None);
        (best_idx, best_coh, best_mass, first_empty, weakest_idx, best_w)
    }

    /// **Classical Online Toroidal $k$-Means E-Step ($\mathcal{O}(K)$ 64-Bit SWAR Nearest-Centroid Search)**:
    /// Returns `(nearest_idx, best_coherence, best_gravity_pull, first_empty_slot_opt, weakest_slot_idx)`.
    #[inline(always)]
    #[must_use]
    pub fn find_nearest_toroidal_centroid_words_64(
        centroids: &[Self],
        sample_words: &[u64; NUM_SWARM_MICRO_CELLS],
    ) -> (usize, f32, f32, Option<usize>, usize) {
        let (best_idx, best_coh, best_grav, first_empty, weakest_idx, _any_active) =
            Self::find_nearest_toroidal_centroid_index_with_cache_64(centroids, sample_words, None);
        (best_idx, best_coh, best_grav, first_empty, weakest_idx)
    }

    #[inline(always)]
    fn init_spawned_slot_header_in_place(&mut self, assigned_id: u32, dominant_slot: u8, zoff48: u64) {
        let slot = dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let was_active = (self.sample_count != 0) | (self.magic != GCSO_CRYSTAL_RECORD_MAGIC);
        self.magic = GCSO_CRYSTAL_RECORD_MAGIC;
        self.version = GCSO_CRYSTAL_RECORD_VERSION;
        self.anchor_id = assigned_id;
        self.sample_count = 0;
        self.zimms_section_offset48 = zoff48 & gcso_tagged_ptr_t::ADDR_MASK;
        self.base_energy = 0.0;
        self.resultant_r_bar = 0.0;
        self.von_mises_kappa = 0.0;
        self.phase_gravity_mass = 0.0;
        self.ripa_clamp_rad = DEFAULT_RIPA_CLAMP_MAX_RAD;
        self.effective_inv_temp = 1.0;
        self.provenance_bits = 0;
        self.checksum_crc32 = Self::EMPTY_CRYSTAL_CRC32_BY_SLOT[slot as usize];
        self.dominant_slot = slot;
        self.action_mode_flags = ACTION_PRESET_PASSIVE_SIDECAR;
        self.active_swarm_mask = 0xFF;
        self.dominant_cause_code = ROOT_CAUSE_NOMINAL;
        self.reserved_align64 = 0;
        if was_active {
            *self.phase_words_mut() = [0u64; NUM_SWARM_MICRO_CELLS];
        }
    }

    #[inline(always)]
    fn select_or_spawn_centroid_slot_64(centroids: &mut [Self], frame: &GatewayFrame,
        nearest_idx: usize, best_coh: f32, empty_opt: Option<usize>, weakest_idx: usize,
        best_w: [u64; NUM_SWARM_MICRO_CELLS], safe_split: f32,
    ) -> (usize, bool, [u64; NUM_SWARM_MICRO_CELLS]) {
        let needs_split = (centroids[nearest_idx].sample_count == 0) | (best_coh < safe_split);
        if needs_split {
            if let Some(empty_i) = empty_opt {
                let assigned_id = if frame.target_anchor_id != 0 {
                    frame.target_anchor_id
                } else {
                    (empty_i as u32) + 1
                };
                centroids[empty_i].init_spawned_slot_header_in_place(
                    assigned_id,
                    frame.dominant_slot,
                    frame.zimms_section_offset48,
                );
                (empty_i, true, [0u64; NUM_SWARM_MICRO_CELLS])
            } else if frame.effective_energy > centroids[weakest_idx].phase_gravity_mass {
                let assigned_id = if frame.target_anchor_id != 0 {
                    frame.target_anchor_id
                } else {
                    (weakest_idx as u32) + 1
                };
                centroids[weakest_idx].init_spawned_slot_header_in_place(
                    assigned_id,
                    frame.dominant_slot,
                    frame.zimms_section_offset48,
                );
                (weakest_idx, true, [0u64; NUM_SWARM_MICRO_CELLS])
            } else {
                (nearest_idx, false, best_w)
            }
        } else {
            (nearest_idx, false, best_w)
        }
    }

    #[inline] pub(crate) fn step_online_toroidal_kmeans_frame_with_words_64(centroids: &mut [Self],
        frame: &GatewayFrame, sample_w: &[u64; NUM_SWARM_MICRO_CELLS],
        split_coherence_thresh: f32, trail_opt: Option<&mut gcso_pointer_trail_t>) -> (usize, u32, f32, f32, bool) {
        let k_len = centroids.len();
        if k_len == 0 { return (0, frame.target_anchor_id, 0.0, 0.0, false); }

        let is_repulsed = (frame.repulsion_severity > 0.50)
            | ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0)
            | (frame.dominant_cause_code == ROOT_CAUSE_COHOMOLOGY_REPULSION)
            | ((frame.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
        let (
            nearest_idx,
            best_coh,
            _best_grav,
            empty_opt,
            weakest_idx,
            best_w,
            sample_is_uniform,
            best_is_uniform,
            _,
        ) = Self::find_nearest_toroidal_centroid_with_cache_64(centroids, sample_w, None);

        if is_repulsed {
            let dom_id = centroids[nearest_idx].anchor_id;
            return (nearest_idx, dom_id, best_coh, centroids[nearest_idx].phase_gravity_mass, false);
        }

        let safe_split = branchless_clamp_f32(split_coherence_thresh, -0.95, 0.95);
        let (target_idx, spawned_new, cur_target_w) = Self::select_or_spawn_centroid_slot_64(
            centroids, frame, nearest_idx, best_coh, empty_opt, weakest_idx, best_w, safe_split);

        let pre_coh = if spawned_new { Some(1.0) } else { Some(best_coh) };
        let uni_flags = Some((if spawned_new { true } else { best_is_uniform }, sample_is_uniform));
        let fallback_aid = (target_idx as u32) + 1;
        let target = &mut centroids[target_idx];
        let prev_crc = target.checksum_crc32;
        let prev_count = target.sample_count;
        let prev_aid = if spawned_new { 0 } else { target.anchor_id };
        let prev_zoff = if spawned_new { 0 } else { target.zimms_section_offset48 };
        let prev_be_bits = target.base_energy.to_bits();
        let prev_gm_bits = target.phase_gravity_mass.to_bits();
        let prev_prov = target.provenance_bits;
        let prev_rb_bits = target.resultant_r_bar.to_bits();
        let prev_kp_bits = target.von_mises_kappa.to_bits();
        let prev_clamp_bits = target.ripa_clamp_rad.to_bits();
        let prev_temp_bits = target.effective_inv_temp.to_bits();
        let prev_w7_lo = target.pack_w7_lo();
        let (next_w, updated_mass, _modified, words_changed) = target
            .absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
                &cur_target_w,
                frame,
                sample_w,
                pre_coh,
                uni_flags,
            );
        let winning_anchor_id = if target.anchor_id != 0 {
            target.anchor_id
        } else {
            fallback_aid
        };
        target.anchor_id = winning_anchor_id;
        let eff_prev_count = if !spawned_new { prev_count } else { 0 };
        target.stamp_or_update_absorption_crc32(
            prev_crc,
            prev_aid,
            eff_prev_count,
            prev_zoff,
            prev_be_bits,
            prev_rb_bits,
            prev_kp_bits,
            prev_gm_bits,
            prev_clamp_bits,
            prev_temp_bits,
            prev_prov,
            prev_w7_lo,
            &cur_target_w,
            &next_w,
            words_changed,
        );
        let final_coh = if spawned_new { 1.0 } else { best_coh };

        if let Some(trail) = trail_opt {
            trail.target_anchor_id = winning_anchor_id;
            trail.cluster_id = winning_anchor_id;
            trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
        }

        (target_idx, winning_anchor_id, final_coh, updated_mass, spawned_new)
    }

    /// **Classical Online Toroidal $k$-Means / DP-Means Closed-Loop E+M Step ($\mathcal{O}(K)$)**:
    /// Returns `(assigned_cluster_idx, assigned_anchor_id, centroid_coherence, updated_gravity_mass, spawned_new_cluster)`.
    #[inline(always)]
    pub fn step_online_toroidal_kmeans_frame_64(centroids: &mut [Self],
        frame: &GatewayFrame, split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>) -> (usize, u32, f32, f32, bool) {
        Self::step_online_toroidal_kmeans_frame_with_words_64(centroids, frame, frame.phase_words_ref(),
            split_coherence_thresh, trail_opt)
    }

    /// **Batched Online Toroidal $k$-Means / DP-Means E+M Step ($\mathcal{O}(B \cdot K)$)**:
    /// Processes `frames` in registers (caching up to 16 centroid SWAR profiles across the entire batch
    /// to eliminate repeated 64B memory reloads) and stamps `checksum_crc32` only once per modified centroid.
    /// Returns `(last_assigned_idx, last_anchor_id, mean_batch_coherence, total_spawned_clusters)`.
    #[inline]
    pub fn step_online_toroidal_kmeans_batch_frames_64(centroids: &mut [Self],
        frames: &[GatewayFrame], split_coherence_thresh: f32,
        mut trail_opt: Option<&mut gcso_pointer_trail_t>) -> (usize, u32, f32, u32) {
        let k_len = centroids.len();
        if k_len == 0 || frames.is_empty() { return (0, 0, 0.0, 0); }

        let safe_split = branchless_clamp_f32(split_coherence_thresh, -0.95, 0.95);
        let mut dirty_mask = 0u64;
        let mut last_idx = 0usize;
        let mut last_anchor_id = 0u32;
        let mut coh_sum = 0.0_f32;
        let mut valid_count = 0usize;
        let mut spawned_total = 0u32;

        // Pre-cache up to 16 centroid 8-word SWAR profiles and uniformity masks on the stack to eliminate B * K reloads.
        let use_reg_cache = k_len <= MAX_REGISTER_CACHED_CENTROIDS;
        let mut word_cache = [[0u64; NUM_SWARM_MICRO_CELLS]; MAX_REGISTER_CACHED_CENTROIDS];
        let mut uni_mask = 0u16;
        let mut byte_uni_mask = 0u16;
        if use_reg_cache {
            let mut c = 0usize;
            while c < k_len {
                if centroids[c].sample_count > 0 {
                    let cw = load_q7_words_64(&centroids[c].phase_profile);
                    word_cache[c] = cw;
                    if is_word_array_uniform_64(&cw) {
                        let bit = 1u16 << c;
                        uni_mask |= bit;
                        let w0 = cw[0];
                        if w0 == (w0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                            byte_uni_mask |= bit;
                        }
                    }
                }
                c += 1;
            }
        }

        for frame in frames {
            let is_repulsed = (frame.repulsion_severity > 0.50)
                | ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0)
                | (frame.dominant_cause_code == ROOT_CAUSE_COHOMOLOGY_REPULSION)
                | ((frame.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
            if is_repulsed { continue; }

            let sample_w = frame.phase_words_ref();
            let cache_opt = if use_reg_cache {
                Some((&word_cache, uni_mask, byte_uni_mask))
            } else {
                None
            };
            let (
                nearest_idx,
                best_coh,
                _best_grav,
                empty_opt,
                weakest_idx,
                best_w,
                sample_is_uniform,
                best_is_uniform,
                _,
            ) = Self::find_nearest_toroidal_centroid_with_cache_64(centroids, sample_w, cache_opt);
            let (target_idx, spawned_new, cur_target_w) = Self::select_or_spawn_centroid_slot_64(
                centroids, frame, nearest_idx, best_coh, empty_opt, weakest_idx, best_w, safe_split);

            let pre_coh = if spawned_new { Some(1.0) } else { Some(best_coh) };
            let uni_flags = Some((if spawned_new { true } else { best_is_uniform }, sample_is_uniform));
            let (next_w, _mass, _modified, words_changed) = centroids[target_idx]
                .absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
                    &cur_target_w,
                    frame,
                    sample_w,
                    pre_coh,
                    uni_flags,
                );
            if use_reg_cache && (spawned_new || words_changed) {
                word_cache[target_idx] = next_w;
                let bit = 1u16 << target_idx;
                if is_word_array_uniform_64(&next_w) {
                    uni_mask |= bit;
                    let w0 = next_w[0];
                    if w0 == (w0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                        byte_uni_mask |= bit;
                    } else {
                        byte_uni_mask &= !bit;
                    }
                } else {
                    uni_mask &= !bit;
                    byte_uni_mask &= !bit;
                }
            }
            let winning_anchor_id = if centroids[target_idx].anchor_id != 0 {
                centroids[target_idx].anchor_id
            } else {
                (target_idx as u32) + 1
            };
            centroids[target_idx].anchor_id = winning_anchor_id;

            if target_idx < 64 { dirty_mask |= 1u64 << target_idx; }
            last_idx = target_idx;
            last_anchor_id = winning_anchor_id;
            coh_sum += if spawned_new { 1.0 } else { best_coh };
            valid_count += 1;
            spawned_total += u32::from(spawned_new);
        }

        if k_len <= 64 {
            let mut m = dirty_mask;
            while m != 0 {
                let c = m.trailing_zeros() as usize;
                m &= m - 1;
                if use_reg_cache {
                    centroids[c].stamp_crc32_with_words(&word_cache[c]);
                } else {
                    centroids[c].stamp_crc32();
                }
            }
        } else {
            for rec in centroids.iter_mut() { if rec.sample_count > 0 { rec.stamp_crc32(); } }
        }

        if let Some(ref mut trail) = trail_opt {
            if last_anchor_id != 0 {
                trail.target_anchor_id = last_anchor_id;
                trail.cluster_id = last_anchor_id;
                trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
            }
        }

        let mean_coh = if valid_count > 0 {
            branchless_clamp_f32(coh_sum * fast_inv_pos_normal_f32(valid_count as f32), -1.0, 1.0)
        } else {
            0.0
        };
        (last_idx, last_anchor_id, mean_coh, spawned_total)
    }

    /// Synthesizes any [`ManifoldGatewaySignal`] via [`ManifoldGatewaySignal::synthesize_gateway_frame_with_words`]
    /// and executes `Self::step_online_toroidal_kmeans_frame_with_words_64` on `centroids` without reloading phase words.
    #[inline(always)]
    pub fn step_online_toroidal_kmeans_signal_64<S: ManifoldGatewaySignal + ?Sized>(
        centroids: &mut [Self], signal: &S, split_coherence_thresh: f32, trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (GatewayFrame, usize, u32, f32, f32, bool) {
        let (frame, sample_w) = signal.synthesize_gateway_frame_with_words();
        let (idx, anchor_id, coh, mass, spawned) = Self::step_online_toroidal_kmeans_frame_with_words_64(
                centroids, &frame, &sample_w, split_coherence_thresh, trail_opt);
        (frame, idx, anchor_id, coh, mass, spawned)
    }

    #[inline(always)]
    fn merge_other_centroid_words_in_place(
        &mut self,
        w_a: &[u64; NUM_SWARM_MICRO_CELLS],
        a_is_uniform: bool,
        w_b: &[u64; NUM_SWARM_MICRO_CELLS],
        b_is_uniform: bool,
        coh: f32,
        b_anchor_id: u32,
        b_sample_count: u32,
        b_zimms_offset48: u64,
        b_base_energy: f32,
        b_resultant_r_bar: f32,
        b_von_mises_kappa: f32,
        b_gravity_mass: f32,
        b_ripa_clamp_rad: f32,
        b_effective_inv_temp: f32,
        b_provenance_bits: u32,
        b_dominant_slot: u8,
        b_action_mode_flags: u8,
        b_active_swarm_mask: u8,
        b_dominant_cause_code: u8,
    ) -> [u64; NUM_SWARM_MICRO_CELLS] {
        let b_dom = b_gravity_mass > self.phase_gravity_mass;
        let (
            cur_dom_w,
            dom_uni,
            src_w,
            src_uni,
            dom_sc,
            dom_e,
            dom_r,
            dom_k,
            src_e,
            src_sc,
            src_r,
            src_k,
        ) = if b_dom {
            self.anchor_id = b_anchor_id;
            self.dominant_slot = b_dominant_slot;
            self.ripa_clamp_rad = b_ripa_clamp_rad;
            self.effective_inv_temp = b_effective_inv_temp;
            self.dominant_cause_code = b_dominant_cause_code;
            if b_zimms_offset48 != 0 {
                self.zimms_section_offset48 = b_zimms_offset48;
            }
            (
                w_b,
                b_is_uniform,
                w_a,
                a_is_uniform,
                b_sample_count,
                b_base_energy,
                b_resultant_r_bar,
                b_von_mises_kappa,
                self.base_energy,
                self.sample_count,
                self.resultant_r_bar,
                self.von_mises_kappa,
            )
        } else {
            if self.zimms_section_offset48 == 0 {
                self.zimms_section_offset48 = b_zimms_offset48;
            }
            (
                w_a,
                a_is_uniform,
                w_b,
                b_is_uniform,
                self.sample_count,
                self.base_energy,
                self.resultant_r_bar,
                self.von_mises_kappa,
                b_base_energy,
                b_sample_count,
                b_resultant_r_bar,
                b_von_mises_kappa,
            )
        };

        let _ = (src_r, src_k);
        let (next_w, raw_next_e, r_bar, kappa) =
            branchless_toroidal_centroid_step_words_with_coh_and_uni_64(
                cur_dom_w,
                dom_sc.max(1),
                dom_e,
                dom_r,
                dom_k,
                src_w,
                Some(coh),
                src_e,
                dom_uni,
                src_uni,
            );
        let next_e = branchless_min_f32(raw_next_e, MAX_ATTRACTOR_PULL_FORCE * 4.0);
        let next_count = dom_sc.saturating_add(src_sc);
        self.sample_count = next_count;
        self.base_energy = next_e;
        self.resultant_r_bar = r_bar;
        self.von_mises_kappa = kappa;
        let maturity_scale = branchless_maturity_scale_u32(next_count);
        self.phase_gravity_mass = branchless_flush_subnormal_f32(
            next_e * branchless_clamp_f32(r_bar, 0.10, 1.0) * maturity_scale,
        );
        self.provenance_bits |= b_provenance_bits | CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED;
        self.action_mode_flags |= b_action_mode_flags;
        if self.action_mode_flags == ACTION_PRESET_PASSIVE_SIDECAR {
            self.action_mode_flags = ACTION_PRESET_EXPRESS_PHASE;
        }
        self.active_swarm_mask &= b_active_swarm_mask;
        next_w
    }

    /// **Autonomous Online Toroidal Centroid Consolidation & Evaporative Pruning ($\mathcal{O}(K^2)$ SWAR)**:
    /// 1. Applies multiplicative `evaporative_decay_factor` $\in (0, 1]$ to active centroids and prunes
    ///    stale noise centroids with `phase_gravity_mass < min_gravity_mass` back to [`Self::new_empty`].
    /// 2. Consolidates any pair of active centroids $(i, j)$ whose phase coherence exceeds
    ///    `merge_coherence_thresh` (e.g. [`DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE`]) into the
    ///    dominant-mass centroid via weighted geodesic SLERP (`branchless_toroidal_centroid_step_words_64`),
    ///    freeing the redundant slot back to [`ACTION_PRESET_PASSIVE_SIDECAR`] and stamping CRC32 once per dirty slot.
    ///
    /// Returns `(merged_cluster_count, pruned_cluster_count)`.
    #[inline]
    pub fn merge_and_prune_centroids_64(centroids: &mut [Self],
        merge_coherence_thresh: f32, min_gravity_mass: f32, evaporative_decay_factor: f32) -> (u32, u32) {
        let k_len = centroids.len();
        if k_len == 0 { return (0, 0); }

        let decay = branchless_clamp_f32(branchless_select_f32(
                branchless_is_finite_f32(evaporative_decay_factor) & (evaporative_decay_factor > 0.0),
                evaporative_decay_factor, 1.0), 0.01, 1.0);
        let apply_decay = decay < 0.999_999;
        let min_mass = branchless_flush_subnormal_f32(branchless_relu_f32(min_gravity_mass));
        let safe_merge = branchless_clamp_f32(merge_coherence_thresh, 0.50, 0.999);
        let mut dirty_mask = 0u64;
        let mut pruned_count = 0u32;
        let mut merged_count = 0u32;

        let use_reg_cache = k_len <= MAX_REGISTER_CACHED_CENTROIDS;
        let mut active_mask = 0u16;
        let mut uniform_mask = 0u16;
        let mut byte_uniform_mask = 0u16;
        let mut active_indices = [0u8; MAX_REGISTER_CACHED_CENTROIDS];
        let mut active_cnt = 0usize;
        // Cache only word 0 (`8` bytes per centroid, `128B` total instead of a `1024B` `[[u64; 8]; 16]`
        // stack frame), eliminating 1KB stack zeroing and 64B inner-loop stack copies while reading
        // full 64B profiles directly from L1D-resident `centroids` only when needed.
        let mut w0_cache = [0u64; MAX_REGISTER_CACHED_CENTROIDS];
        let mut decay_delta_be = [0u32; MAX_REGISTER_CACHED_CENTROIDS];
        let mut decay_delta_gm = [0u32; MAX_REGISTER_CACHED_CENTROIDS];

        let mut i = 0usize;
        while i < k_len {
            if centroids[i].sample_count > 0 {
                let prev_be_bits = if apply_decay { centroids[i].base_energy.to_bits() } else { 0 };
                let prev_gm_bits = if apply_decay { centroids[i].phase_gravity_mass.to_bits() } else { 0 };
                let (new_be, new_gm) = if apply_decay {
                    (
                        branchless_flush_subnormal_f32(centroids[i].base_energy * decay),
                        branchless_flush_subnormal_f32(centroids[i].phase_gravity_mass * decay),
                    )
                } else {
                    (centroids[i].base_energy, centroids[i].phase_gravity_mass)
                };
                if new_gm < min_mass {
                    let slot = centroids[i].dominant_slot;
                    centroids[i].clear_empty_in_place(slot);
                    pruned_count += 1;
                } else {
                    if apply_decay {
                        centroids[i].base_energy = new_be;
                        centroids[i].phase_gravity_mass = new_gm;
                    }
                    if new_gm > 1.0e-12 {
                        if use_reg_cache {
                            let i_bit = 1u16 << i;
                            active_mask |= i_bit;
                            active_indices[active_cnt] = i as u8;
                            let cw = centroids[i].phase_words_ref();
                            let w0 = cw[0];
                            w0_cache[i] = w0;
                            if is_word_array_uniform_64(cw) {
                                uniform_mask |= i_bit;
                                if w0 == (w0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                                    byte_uniform_mask |= i_bit;
                                }
                            }
                            if apply_decay {
                                decay_delta_be[i] = prev_be_bits ^ new_be.to_bits();
                                decay_delta_gm[i] = prev_gm_bits ^ new_gm.to_bits();
                            }
                        } else if apply_decay {
                            if centroids[i].checksum_crc32 != 0 {
                                centroids[i].checksum_crc32 = crc32_update_crystal_energy_and_mass_delta(
                                    centroids[i].checksum_crc32,
                                    prev_be_bits ^ new_be.to_bits(),
                                    prev_gm_bits ^ new_gm.to_bits(),
                                );
                            } else if i < 64 {
                                dirty_mask |= 1u64 << i;
                            }
                        }
                        active_cnt += 1;
                    } else if apply_decay {
                        if centroids[i].checksum_crc32 != 0 {
                            centroids[i].checksum_crc32 = crc32_update_crystal_energy_and_mass_delta(
                                centroids[i].checksum_crc32,
                                prev_be_bits ^ new_be.to_bits(),
                                prev_gm_bits ^ new_gm.to_bits(),
                            );
                        } else {
                            centroids[i].stamp_crc32();
                        }
                    }
                }
            }
            i += 1;
        }

        if active_cnt < 2 {
            if use_reg_cache {
                if apply_decay && active_cnt == 1 {
                    let idx = usize::from(active_indices[0]);
                    if centroids[idx].checksum_crc32 != 0 {
                        centroids[idx].checksum_crc32 = crc32_update_crystal_energy_and_mass_delta(
                            centroids[idx].checksum_crc32,
                            decay_delta_be[idx],
                            decay_delta_gm[idx],
                        );
                    } else {
                        centroids[idx].stamp_crc32();
                    }
                }
            } else if dirty_mask != 0 {
                if k_len <= 64 {
                    let mut m = dirty_mask;
                    while m != 0 {
                        let idx = m.trailing_zeros() as usize;
                        m &= m - 1;
                        centroids[idx].stamp_crc32();
                    }
                } else {
                    for rec in centroids.iter_mut() { if rec.sample_count > 0 { rec.stamp_crc32(); } }
                }
            }
            return (0, pruned_count);
        }

        if use_reg_cache {
            let mut ia = 0usize;
            while ia < active_cnt {
                let a = usize::from(active_indices[ia]);
                let a_bit = 1u16 << a;
                if (active_mask & a_bit) != 0 {
                    let mut a_merged = false;
                    let mut orig_w_a = [0u64; NUM_SWARM_MICRO_CELLS];
                    let mut w_a = [0u64; NUM_SWARM_MICRO_CELLS];
                    let mut prev_crc = 0u32;
                    let mut prev_count = 0u32;
                    let mut prev_aid = 0u32;
                    let mut prev_zoff = 0u64;
                    let mut prev_be_bits = 0u32;
                    let mut prev_gm_bits = 0u32;
                    let mut prev_rb_bits = 0u32;
                    let mut prev_kp_bits = 0u32;
                    let mut prev_clamp_bits = 0u32;
                    let mut prev_temp_bits = 0u32;
                    let mut prev_prov = 0u32;
                    let mut prev_w7_lo = 0u32;
                    if ia + 1 < active_cnt {
                        let mut a_is_uniform = (uniform_mask & a_bit) != 0;
                        let mut a_is_byte_uniform = (byte_uniform_mask & a_bit) != 0;
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        let mut va_loaded = false;
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        let (mut va0, mut va1) = unsafe {
                            use core::arch::x86_64::_mm256_setzero_si256;
                            (_mm256_setzero_si256(), _mm256_setzero_si256())
                        };
                        let mut ib = ia + 1;
                        while ib < active_cnt {
                            let b = usize::from(active_indices[ib]);
                            let b_bit = 1u16 << b;
                            if (active_mask & b_bit) != 0 {
                                let b_is_uniform = (uniform_mask & b_bit) != 0;
                                let b_is_byte_uniform = (byte_uniform_mask & b_bit) != 0;
                                let wb0 = w0_cache[b];
                                let coh = if a_is_byte_uniform & b_is_byte_uniform {
                                    let wa0 = w0_cache[a];
                                    if wa0 == wb0 {
                                        1.0
                                    } else {
                                        Q7_FULL_COS_REF
                                            [(wa0 as u8).wrapping_sub(wb0 as u8) as usize]
                                    }
                                } else if a_is_uniform & b_is_uniform {
                                    coherence_from_uniform_words_x8(w0_cache[a], wb0)
                                } else {
                                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                                    {
                                        unsafe {
                                            use core::arch::x86_64::{
                                                __m256i, _mm256_loadu_si256, _mm256_sub_epi8,
                                            };
                                            if !va_loaded {
                                                if a_merged {
                                                    (va0, va1) = avx2_words64_to_vec64(&w_a);
                                                } else {
                                                    let pa = centroids[a]
                                                        .phase_profile
                                                        .as_ptr()
                                                        .cast::<__m256i>();
                                                    va0 = _mm256_loadu_si256(pa);
                                                    va1 = _mm256_loadu_si256(pa.add(1));
                                                }
                                                va_loaded = true;
                                            }
                                            let pb = centroids[b]
                                                .phase_profile
                                                .as_ptr()
                                                .cast::<__m256i>();
                                            let vb0 = _mm256_loadu_si256(pb);
                                            let vb1 = _mm256_loadu_si256(pb.add(1));
                                            avx2_coherence_from_diff_vec64_with_floor(
                                                _mm256_sub_epi8(va0, vb0),
                                                _mm256_sub_epi8(va1, vb1),
                                                safe_merge,
                                            )
                                        }
                                    }
                                    #[cfg(not(all(
                                        target_arch = "x86_64",
                                        target_feature = "avx2"
                                    )))]
                                    {
                                        let cur_a_ref = if a_merged {
                                            &w_a
                                        } else {
                                            centroids[a].phase_words_ref()
                                        };
                                        coherence_above_floor_from_word_arrays_64(
                                            cur_a_ref,
                                            a_is_uniform,
                                            centroids[b].phase_words_ref(),
                                            b_is_uniform,
                                            safe_merge,
                                        )
                                    }
                                };
                                if coh >= safe_merge {
                                    let w_b = if b_is_uniform {
                                        [wb0; NUM_SWARM_MICRO_CELLS]
                                    } else {
                                        *centroids[b].phase_words_ref()
                                    };
                                    if !a_merged {
                                        orig_w_a = if a_is_uniform {
                                            [w0_cache[a]; NUM_SWARM_MICRO_CELLS]
                                        } else {
                                            *centroids[a].phase_words_ref()
                                        };
                                        w_a = orig_w_a;
                                        prev_crc = centroids[a].checksum_crc32;
                                        prev_count = centroids[a].sample_count;
                                        prev_aid = centroids[a].anchor_id;
                                        prev_zoff = centroids[a].zimms_section_offset48;
                                        prev_be_bits =
                                            centroids[a].base_energy.to_bits() ^ decay_delta_be[a];
                                        prev_gm_bits = centroids[a].phase_gravity_mass.to_bits()
                                            ^ decay_delta_gm[a];
                                        prev_rb_bits = centroids[a].resultant_r_bar.to_bits();
                                        prev_kp_bits = centroids[a].von_mises_kappa.to_bits();
                                        prev_clamp_bits = centroids[a].ripa_clamp_rad.to_bits();
                                        prev_temp_bits = centroids[a].effective_inv_temp.to_bits();
                                        prev_prov = centroids[a].provenance_bits;
                                        prev_w7_lo = centroids[a].pack_w7_lo();
                                    }
                                    let b_crc = centroids[b].checksum_crc32;
                                    let b_aid = centroids[b].anchor_id;
                                    let b_sc = centroids[b].sample_count;
                                    let b_zoff = centroids[b].zimms_section_offset48;
                                    let b_be = centroids[b].base_energy;
                                    let b_rbar = centroids[b].resultant_r_bar;
                                    let b_kap = centroids[b].von_mises_kappa;
                                    let b_mass = centroids[b].phase_gravity_mass;
                                    let b_clamp = centroids[b].ripa_clamp_rad;
                                    let b_temp = centroids[b].effective_inv_temp;
                                    let b_prov = centroids[b].provenance_bits;
                                    let b_slot = centroids[b].dominant_slot;
                                    let b_act = centroids[b].action_mode_flags;
                                    let b_swm = centroids[b].active_swarm_mask;
                                    let b_cause = centroids[b].dominant_cause_code;
                                    let b_w7_lo = centroids[b].pack_w7_lo();
                                    let b_be_pre_bits = b_be.to_bits() ^ decay_delta_be[b];
                                    let b_gm_pre_bits = b_mass.to_bits() ^ decay_delta_gm[b];
                                    w_a = centroids[a].merge_other_centroid_words_in_place(
                                        &w_a, a_is_uniform, &w_b, b_is_uniform, coh, b_aid, b_sc,
                                        b_zoff, b_be, b_rbar, b_kap, b_mass, b_clamp, b_temp,
                                        b_prov, b_slot, b_act, b_swm, b_cause,
                                    );
                                    if b_crc != 0
                                        && !swar_words_equal_64(&w_a, &orig_w_a)
                                        && swar_words_equal_64(&w_a, &w_b)
                                    {
                                        orig_w_a = w_b;
                                        prev_crc = b_crc;
                                        prev_count = b_sc;
                                        prev_aid = b_aid;
                                        prev_zoff = b_zoff;
                                        prev_be_bits = b_be_pre_bits;
                                        prev_gm_bits = b_gm_pre_bits;
                                        prev_rb_bits = b_rbar.to_bits();
                                        prev_kp_bits = b_kap.to_bits();
                                        prev_clamp_bits = b_clamp.to_bits();
                                        prev_temp_bits = b_temp.to_bits();
                                        prev_prov = b_prov;
                                        prev_w7_lo = b_w7_lo;
                                    }
                                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                                    {
                                        va_loaded = false;
                                    }
                                    a_is_uniform = if a_is_uniform & b_is_uniform {
                                        true
                                    } else {
                                        is_word_array_uniform_64(&w_a)
                                    };
                                    a_is_byte_uniform = a_is_uniform
                                        & (w_a[0]
                                            == (w_a[0] & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64));
                                    w0_cache[a] = w_a[0];
                                    a_merged = true;
                                    centroids[b].clear_empty_in_place(b_slot);
                                    active_mask &= !b_bit;
                                    merged_count += 1;
                                }
                            }
                            ib += 1;
                        }
                    }
                    if a_merged {
                        let words_changed = !swar_words_equal_64(&w_a, &orig_w_a);
                        if words_changed {
                            store_q7_words_64(&mut centroids[a].phase_profile, &w_a);
                        }
                        centroids[a].stamp_or_update_absorption_crc32(
                            prev_crc,
                            prev_aid,
                            prev_count,
                            prev_zoff,
                            prev_be_bits,
                            prev_rb_bits,
                            prev_kp_bits,
                            prev_gm_bits,
                            prev_clamp_bits,
                            prev_temp_bits,
                            prev_prov,
                            prev_w7_lo,
                            &orig_w_a,
                            &w_a,
                            words_changed,
                        );
                    } else if apply_decay {
                        if centroids[a].checksum_crc32 != 0 {
                            centroids[a].checksum_crc32 = crc32_update_crystal_energy_and_mass_delta(
                                centroids[a].checksum_crc32,
                                decay_delta_be[a],
                                decay_delta_gm[a],
                            );
                        } else {
                            centroids[a].stamp_crc32();
                        }
                    }
                }
                ia += 1;
            }
        } else {
            let mut a = 0usize;
            while a < k_len {
                if centroids[a].sample_count > 0 && centroids[a].phase_gravity_mass > 1.0e-12 {
                    let orig_w_a = load_q7_words_64(&centroids[a].phase_profile);
                    let mut w_a = orig_w_a;
                    let mut a_merged = false;
                    let mut b = a + 1;
                    while b < k_len {
                        if centroids[b].sample_count > 0 && centroids[b].phase_gravity_mass > 1.0e-12 {
                            let w_b = load_q7_words_64(&centroids[b].phase_profile);
                            let coh = coherence_from_word_arrays_64(&w_a, &w_b);
                            if coh >= safe_merge {
                                let b_aid = centroids[b].anchor_id;
                                let b_sc = centroids[b].sample_count;
                                let b_zoff = centroids[b].zimms_section_offset48;
                                let b_be = centroids[b].base_energy;
                                let b_rbar = centroids[b].resultant_r_bar;
                                let b_kap = centroids[b].von_mises_kappa;
                                let b_mass = centroids[b].phase_gravity_mass;
                                let b_clamp = centroids[b].ripa_clamp_rad;
                                let b_temp = centroids[b].effective_inv_temp;
                                let b_prov = centroids[b].provenance_bits;
                                let b_slot = centroids[b].dominant_slot;
                                let b_act = centroids[b].action_mode_flags;
                                let b_swm = centroids[b].active_swarm_mask;
                                let b_cause = centroids[b].dominant_cause_code;
                                let a_uni = is_word_array_uniform_64(&w_a);
                                let b_uni = is_word_array_uniform_64(&w_b);
                                w_a = centroids[a].merge_other_centroid_words_in_place(
                                    &w_a, a_uni, &w_b, b_uni, coh, b_aid, b_sc, b_zoff, b_be,
                                    b_rbar, b_kap, b_mass, b_clamp, b_temp, b_prov, b_slot, b_act,
                                    b_swm, b_cause,
                                );
                                a_merged = true;
                                centroids[b] = Self::new_empty(0, b_slot, 0);
                                if b < 64 { dirty_mask &= !(1u64 << b); }
                                merged_count += 1;
                            }
                        }
                        b += 1;
                    }
                    if a_merged {
                        if !swar_words_equal_64(&w_a, &orig_w_a) {
                            store_q7_words_64(&mut centroids[a].phase_profile, &w_a);
                        }
                        centroids[a].stamp_crc32_with_words(&w_a);
                        if a < 64 { dirty_mask &= !(1u64 << a); }
                    } else if a < 64 && (dirty_mask & (1u64 << a)) != 0 {
                        centroids[a].stamp_crc32_with_words(&w_a);
                        dirty_mask &= !(1u64 << a);
                    }
                }
                a += 1;
            }
            if k_len > 64 && dirty_mask != 0 {
                for rec in centroids.iter_mut() { if rec.sample_count > 0 { rec.stamp_crc32(); } }
            }
        }

        (merged_count, pruned_count)
    }

    /// Evaluates macro distribution diagnostics across `centroids` in $\mathcal{O}(K^2)$ SWAR:
    /// returns `(active_clusters, total_samples, mean_resultant_r_bar, mean_von_mises_kappa, max_pairwise_coherence)`.
    #[inline]
    #[must_use]
    pub fn diagnose_macro_distribution_64(centroids: &[Self]) -> (u32, u32, f32, f32, f32) {
        let k_len = centroids.len();
        if k_len == 0 { return (0, 0, 0.0, 0.0, 0.0); }

        let mut active_cnt = 0u32;
        let mut tot_samples = 0u32;
        let mut sum_r = 0.0_f32;
        let mut sum_kappa = 0.0_f32;
        let mut max_pair_coh = -1.0_f32;
        let mut has_pair = false;

        if k_len <= MAX_REGISTER_CACHED_CENTROIDS {
            let mut active_indices = [0u8; MAX_REGISTER_CACHED_CENTROIDS];
            let mut uniform_mask = 0u16;
            let mut byte_uniform_mask = 0u16;
            let mut n_act = 0usize;
            let mut i = 0usize;
            while i < k_len {
                let rec = &centroids[i];
                if rec.sample_count > 0 && rec.phase_gravity_mass > 1.0e-12 {
                    let w = rec.phase_words_ref();
                    active_indices[n_act] = i as u8;
                    if is_word_array_uniform_64(w) {
                        let bit = 1u16 << n_act;
                        uniform_mask |= bit;
                        let w0 = w[0];
                        if w0 == (w0 & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                            byte_uniform_mask |= bit;
                        }
                    }
                    n_act += 1;
                    tot_samples = tot_samples.saturating_add(rec.sample_count);
                    sum_r += rec.resultant_r_bar;
                    sum_kappa += rec.von_mises_kappa;
                }
                i += 1;
            }
            active_cnt = n_act as u32;
            let mut ia = 0usize;
            while ia + 1 < n_act {
                let idx_i = usize::from(active_indices[ia]);
                let w_i = centroids[idx_i].phase_words_ref();
                let i_uni = ((uniform_mask >> ia) & 1) != 0;
                let i_byte_uni = ((byte_uniform_mask >> ia) & 1) != 0;
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                let mut vi_loaded = false;
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                let (mut vi0, mut vi1) = unsafe {
                    use core::arch::x86_64::_mm256_setzero_si256;
                    (_mm256_setzero_si256(), _mm256_setzero_si256())
                };
                let mut ib = ia + 1;
                while ib < n_act {
                    let idx_j = usize::from(active_indices[ib]);
                    let w_j = centroids[idx_j].phase_words_ref();
                    let j_uni = ((uniform_mask >> ib) & 1) != 0;
                    let j_byte_uni = ((byte_uniform_mask >> ib) & 1) != 0;
                    let coh = if i_byte_uni & j_byte_uni {
                        let wi0 = w_i[0];
                        let wj0 = w_j[0];
                        if wi0 == wj0 {
                            1.0
                        } else {
                            Q7_FULL_COS_REF[(wi0 as u8).wrapping_sub(wj0 as u8) as usize]
                        }
                    } else if i_uni & j_uni {
                        coherence_from_uniform_words_x8(w_i[0], w_j[0])
                    } else {
                        let floor = if has_pair { max_pair_coh } else { -2.0 };
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        {
                            unsafe {
                                use core::arch::x86_64::{
                                    __m256i, _mm256_loadu_si256, _mm256_sub_epi8,
                                };
                                if !vi_loaded {
                                    let pi =
                                        centroids[idx_i].phase_profile.as_ptr().cast::<__m256i>();
                                    vi0 = _mm256_loadu_si256(pi);
                                    vi1 = _mm256_loadu_si256(pi.add(1));
                                    vi_loaded = true;
                                }
                                let pj = centroids[idx_j].phase_profile.as_ptr().cast::<__m256i>();
                                let vj0 = _mm256_loadu_si256(pj);
                                let vj1 = _mm256_loadu_si256(pj.add(1));
                                avx2_coherence_from_diff_vec64_with_floor(
                                    _mm256_sub_epi8(vi0, vj0),
                                    _mm256_sub_epi8(vi1, vj1),
                                    floor,
                                )
                            }
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                        {
                            coherence_above_floor_from_word_arrays_64(
                                w_i,
                                i_uni,
                                w_j,
                                j_uni,
                                floor,
                            )
                        }
                    };
                    max_pair_coh = branchless_select_f32((!has_pair) | (coh > max_pair_coh), coh, max_pair_coh);
                    has_pair = true;
                    ib += 1;
                }
                ia += 1;
            }
        } else {
            let mut i = 0usize;
            while i < k_len {
                let rec_i = &centroids[i];
                if rec_i.sample_count > 0 && rec_i.phase_gravity_mass > 1.0e-12 {
                    active_cnt += 1;
                    tot_samples = tot_samples.saturating_add(rec_i.sample_count);
                    sum_r += rec_i.resultant_r_bar;
                    sum_kappa += rec_i.von_mises_kappa;
                    let w_i = load_q7_words_64(&rec_i.phase_profile);
                    let mut j = i + 1;
                    while j < k_len {
                        let rec_j = &centroids[j];
                        if rec_j.sample_count > 0 && rec_j.phase_gravity_mass > 1.0e-12 {
                            let w_j = load_q7_words_64(&rec_j.phase_profile);
                            let coh = coherence_from_word_arrays_64(&w_i, &w_j);
                            max_pair_coh = branchless_select_f32((!has_pair) | (coh > max_pair_coh), coh, max_pair_coh);
                            has_pair = true;
                        }
                        j += 1;
                    }
                }
                i += 1;
            }
        }

        if active_cnt == 0 { return (0, 0, 0.0, 0.0, 0.0); }
        let inv_act = fast_inv_pos_normal_f32(active_cnt as f32);
        let mean_r = branchless_clamp_f32(sum_r * inv_act, 0.0, 1.0);
        let mean_k = branchless_flush_subnormal_f32(sum_kappa * inv_act);
        let pair_coh = branchless_select_f32(has_pair, max_pair_coh, 0.0);
        (active_cnt, tot_samples, mean_r, mean_k, pair_coh)
    }

    #[inline(always)]
    fn apply_macro_centroid_pull_to_trail_64(trail: &mut gcso_pointer_trail_t,
        trail_w: &[u64; NUM_SWARM_MICRO_CELLS], trail_is_uniform: bool,
        target_w: &[u64; NUM_SWARM_MICRO_CELLS], target_is_uniform: bool,
        winner: &Self, anchor_id: u32, eff_coh: f32, tot_grav: f32, pull_step_u7: u8,
        extra_flags: u32) -> [u64; NUM_SWARM_MICRO_CELLS] {
        let both_uni = trail_is_uniform & target_is_uniform;
        let tw0 = trail_w[0];
        let gw0 = target_w[0];
        let tb0 = (tw0 & 0xFF) as u8;
        let gb0 = (gw0 & 0xFF) as u8;
        let both_byte_uni = both_uni
            & (tw0 == (tb0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64))
            & (gw0 == (gb0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64));
        let equal_wt = if both_uni {
            tw0 == gw0
        } else {
            swar_words_equal_64(trail_w, target_w)
        };
        let mut scalar_tag_opt: Option<gcso_q7_t> = None;
        let (bridged_w, bridged_uni) = if pull_step_u7 == 0 || equal_wt {
            if both_byte_uni {
                scalar_tag_opt = Some(tb0 as gcso_q7_t);
            }
            (*trail_w, trail_is_uniform)
        } else if both_byte_uni {
            let b_lerp = if pull_step_u7 >= 128 {
                gb0 as gcso_q7_t
            } else {
                branchless_q7_circular_lerp(tb0 as gcso_q7_t, gb0 as gcso_q7_t, pull_step_u7)
            };
            scalar_tag_opt = Some(b_lerp);
            let w_lerp = ((b_lerp as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let bw = [w_lerp; NUM_SWARM_MICRO_CELLS];
            *trail_phase_words_mut(trail) = bw;
            (bw, true)
        } else {
            let s_u64 = branchless_select_u8(pull_step_u7 > 128, 128, pull_step_u7) as u64;
            if s_u64 >= 128 {
                *trail_phase_words_mut(trail) = *target_w;
                (*target_w, target_is_uniform)
            } else if both_uni {
                let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                let (bw, b_uni) = swar_slerp_words_with_uniform_flag_64(
                    trail_w,
                    true,
                    target_w,
                    true,
                    s_u64,
                    bias_restore,
                );
                *trail_phase_words_mut(trail) = bw;
                (bw, b_uni)
            } else {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    unsafe {
                        use core::arch::x86_64::{__m256i, _mm256_loadu_si256};
                        let (vt0, vt1) = super::branchless::avx2_words64_to_vec64_with_uniform(
                            trail_w,
                            trail_is_uniform,
                        );
                        let (vg0, vg1) = if core::ptr::eq(target_w, winner.phase_words_ref()) {
                            let wp = winner.phase_profile.as_ptr().cast::<__m256i>();
                            (_mm256_loadu_si256(wp), _mm256_loadu_si256(wp.add(1)))
                        } else {
                            super::branchless::avx2_words64_to_vec64_with_uniform(
                                target_w,
                                target_is_uniform,
                            )
                        };
                        let (vb0, vb1) = avx2_slerp_vec64(vt0, vt1, vg0, vg1, s_u64);
                        if winner.zimms_section_offset48 != 0 {
                            scalar_tag_opt =
                                Some(super::branchless::avx2_q7_mean_i8_from_vec64(vb0, vb1));
                        }
                        let bw = avx2_vec64_to_words64(vb0, vb1);
                        *trail_phase_words_mut(trail) = bw;
                        (bw, false)
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let (bw, b_uni) = swar_slerp_words_with_uniform_flag_64(
                        trail_w,
                        trail_is_uniform,
                        target_w,
                        target_is_uniform,
                        s_u64,
                        bias_restore,
                    );
                    *trail_phase_words_mut(trail) = bw;
                    (bw, b_uni)
                }
            }
        };

        trail.target_anchor_id = anchor_id;
        trail.cluster_id = anchor_id;
        trail.attractor_pull_force = branchless_min_f32(trail.attractor_pull_force + 0.5 * tot_grav,
            MAX_ATTRACTOR_PULL_FORCE);
        trail.stigmergic_density = branchless_min_f32(trail.stigmergic_density + 0.20 * branchless_relu_f32(eff_coh),
            MAX_STIGMERGIC_DENSITY);
        if winner.zimms_section_offset48 != 0 {
            let tag_q7 = match scalar_tag_opt {
                Some(t) => t,
                None => swar_q7_mean_i8_from_words_with_uniform_flag_64(&bridged_w, bridged_uni),
            };
            let packed_tag = pack_action_slot_tag(winner.dominant_slot, winner.action_mode_flags);
            trail.prev_ptr = trail.current_ptr;
            trail.current_ptr = gcso_tagged_ptr_t::pack(winner.zimms_section_offset48, packed_tag, tag_q7).raw();
        }
        trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | extra_flags;
        bridged_w
    }

    #[inline(always)]
    pub(crate) fn route_coarse_to_fine_trail_from_preloaded_words_64(
        centroids: &[Self],
        trail: &mut gcso_pointer_trail_t,
        trail_w: [u64; NUM_SWARM_MICRO_CELLS],
        pull_step_u7: u8,
    ) -> (usize, u32, f32, f32, [u64; NUM_SWARM_MICRO_CELLS], [u64; NUM_SWARM_MICRO_CELLS]) {
        let (
            best_idx,
            best_coh,
            best_mass,
            _empty_opt,
            _weakest_idx,
            win_w,
            trail_uni,
            win_uni,
            best_byte_diff,
        ) = Self::find_nearest_toroidal_centroid_with_cache_64(centroids, &trail_w, None);
        if centroids.is_empty() || centroids[best_idx].sample_count == 0 || best_mass <= 1.0e-12 {
            return (0, trail.target_anchor_id, 0.0, 0.0, trail_w, [0u64; NUM_SWARM_MICRO_CELLS]);
        }

        let best_grav = match best_byte_diff {
            Some(diff_u8) => branchless_q7_default_gravity_from_diff_u8(best_mass, diff_u8),
            None => branchless_phase_proximity_gravity_f32(
                best_mass,
                best_coh,
                DEFAULT_PHASE_GRAVITY_EPS,
            ),
        };
        let winner = &centroids[best_idx];
        let anchor_id = if winner.anchor_id != 0 {
            winner.anchor_id
        } else {
            (best_idx as u32) + 1
        };
        let bridged_w = Self::apply_macro_centroid_pull_to_trail_64(
            trail,
            &trail_w,
            trail_uni,
            winner.phase_words_ref(),
            win_uni,
            winner,
            anchor_id,
            best_coh,
            best_grav,
            pull_step_u7,
            0,
        );
        (best_idx, anchor_id, best_coh, best_grav, bridged_w, win_w)
    }

    #[inline(always)] pub(crate) fn route_coarse_to_fine_trail_with_words_64(centroids: &[Self],
        trail: &mut gcso_pointer_trail_t, pull_step_u7: u8,
    ) -> (usize, u32, f32, f32, [u64; NUM_SWARM_MICRO_CELLS], [u64; NUM_SWARM_MICRO_CELLS]) {
        let trail_w = *trail_phase_words_ref(trail);
        Self::route_coarse_to_fine_trail_from_preloaded_words_64(
            centroids,
            trail,
            trail_w,
            pull_step_u7,
        )
    }

    /// **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K)$)**:
    /// Returns `(selected_cluster_idx, selected_anchor_id, centroid_coherence, gravity_pull)`.
    #[inline]
    pub fn route_coarse_to_fine_trail_64(centroids: &[Self],
        trail: &mut gcso_pointer_trail_t, pull_step_u7: u8) -> (usize, u32, f32, f32) {
        let (best_idx, anchor_id, best_coh, best_grav, _bridged_w, _win_w) =
            Self::route_coarse_to_fine_trail_with_words_64(centroids, trail, pull_step_u7);
        (best_idx, anchor_id, best_coh, best_grav)
    }

    #[inline(always)]
    pub(crate) fn route_top2_liminal_centroids_from_preloaded_words_64(
        centroids: &[Self],
        trail: &mut gcso_pointer_trail_t,
        trail_w: [u64; NUM_SWARM_MICRO_CELLS],
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
    ) -> (
        usize,
        usize,
        u32,
        u32,
        f32,
        f32,
        bool,
        [u64; NUM_SWARM_MICRO_CELLS],
        [u64; NUM_SWARM_MICRO_CELLS],
        [u64; NUM_SWARM_MICRO_CELLS],
    ) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (0, 0, trail.target_anchor_id, 0, 0.0, 0.0, false, trail_w, trail_w, trail_w);
        }

        let mut idx1 = 0usize;
        let mut coh1 = -2.0_f32;
        let mut grav1 = 0.0_f32;
        let mut uni1 = false;

        let trail_is_uniform = is_word_array_uniform_64(&trail_w);
        let trail_w0 = trail_w[0];
        let trail_b0 = (trail_w0 & 0xFF) as u8;
        let trail_is_byte_uniform =
            trail_is_uniform & (trail_w0 == (trail_b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64));
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (va0, va1) = unsafe {
            super::branchless::avx2_words64_to_vec64_with_uniform(&trail_w, trail_is_uniform)
        };
        let mut byte_uni1 = false;
        let mut idx2 = 0usize;
        let mut coh2 = -2.0_f32;
        let mut grav2 = 0.0_f32;
        let mut uni2 = false;
        let mut byte_uni2 = false;
        let mut active_found = 0u32;

        let mut i = 0usize;
        while i < k_len {
            let rec = &centroids[i];
            let mass = rec.phase_gravity_mass;
            if (rec.sample_count > 0) & (mass > 1.0e-12) {
                let rec_w = rec.phase_words_ref();
                let rw0 = rec_w[0];
                let rb0 = (rw0 & 0xFF) as u8;
                let rec_uni = if trail_is_uniform {
                    is_word_array_uniform_64(rec_w)
                } else {
                    false
                };
                let rec_byte_uni = rec_uni
                    & trail_is_byte_uniform
                    & ((rw0 == trail_w0) | (rw0 == (rb0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64)));
                let (coh, grav) = if rec_uni {
                    if rw0 == trail_w0 {
                        (1.0, branchless_q7_default_gravity_from_diff_u8(mass, 0))
                    } else if rec_byte_uni {
                        let d_u8 = trail_b0.wrapping_sub(rb0);
                        (
                            Q7_FULL_COS_REF[d_u8 as usize],
                            branchless_q7_default_gravity_from_diff_u8(mass, d_u8),
                        )
                    } else {
                        let c_val = coherence_above_floor_from_word_arrays_64(
                            &trail_w,
                            true,
                            rec_w,
                            true,
                            if (active_found >= 2) & (grav2 > mass) { -0.50 } else { -2.0 },
                        );
                        let g_val = if c_val > -1.5 {
                            branchless_phase_proximity_gravity_f32(
                                mass,
                                c_val,
                                DEFAULT_PHASE_GRAVITY_EPS,
                            )
                        } else {
                            0.0
                        };
                        (c_val, g_val)
                    }
                } else {
                    let floor = if (active_found >= 2) & (grav2 > mass) {
                        -0.50
                    } else {
                        -2.0
                    };
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                    let c_val = {
                        unsafe {
                            use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
                            let ptr = rec.phase_profile.as_ptr().cast::<__m256i>();
                            let vb0 = _mm256_loadu_si256(ptr);
                            let vb1 = _mm256_loadu_si256(ptr.add(1));
                            super::branchless::avx2_coherence_from_diff_vec64_with_floor(
                                _mm256_sub_epi8(va0, vb0),
                                _mm256_sub_epi8(va1, vb1),
                                floor,
                            )
                        }
                    };
                    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                    let c_val = coherence_above_floor_from_word_arrays_64(
                        &trail_w,
                        trail_is_uniform,
                        rec_w,
                        false,
                        floor,
                    );
                    let g_val = if c_val > -1.5 {
                        branchless_phase_proximity_gravity_f32(
                            mass,
                            c_val,
                            DEFAULT_PHASE_GRAVITY_EPS,
                        )
                    } else {
                        0.0
                    };
                    (c_val, g_val)
                };
                if coh > -1.5 {
                    let better1 =
                        (active_found == 0) | (grav > grav1) | ((grav == grav1) & (coh > coh1));
                    if better1 {
                        idx2 = idx1;
                        coh2 = coh1;
                        grav2 = grav1;
                        uni2 = uni1;
                        byte_uni2 = byte_uni1;
                        idx1 = i;
                        coh1 = coh;
                        grav1 = grav;
                        uni1 = rec_uni;
                        byte_uni1 = rec_byte_uni;
                    } else if (active_found == 1) | (grav > grav2) | ((grav == grav2) & (coh > coh2)) {
                        idx2 = i;
                        coh2 = coh;
                        grav2 = grav;
                        uni2 = rec_uni;
                        byte_uni2 = rec_byte_uni;
                    }
                }
                active_found += 1;
            }
            i += 1;
        }

        if active_found == 0 {
            return (0, 0, trail.target_anchor_id, 0, 0.0, 0.0, false, trail_w, trail_w, trail_w);
        }

        let w1 = *centroids[idx1].phase_words_ref();
        let safe_ratio = branchless_clamp_f32(liminal_gravity_ratio_thresh, 0.10, 0.99);
        let mass_ratio_ok =
            centroids[idx2].phase_gravity_mass >= centroids[idx1].phase_gravity_mass * safe_ratio;
        let is_liminal = (active_found >= 2)
            & (grav2 > 1.0e-12)
            & ((grav2 >= grav1 * safe_ratio) | (mass_ratio_ok & (coh2 >= safe_ratio)))
            & (coh2 > -0.25);

        let w1_uni = if trail_is_uniform {
            uni1
        } else {
            is_word_array_uniform_64(&w1)
        };
        let (target_w, target_uni, w2_opt) = if is_liminal {
            let w2 = *centroids[idx2].phase_words_ref();
            let w2_uni = if trail_is_uniform {
                uni2
            } else if w1_uni {
                is_word_array_uniform_64(&w2)
            } else {
                false
            };
            let tot_g = grav1 + grav2;
            let safe_tot = branchless_select_f32(tot_g > 1.0e-12, tot_g, 1.0);
            let w2_ratio =
                branchless_clamp_f32(grav2 * fast_inv_pos_normal_f32(safe_tot), 0.10, 0.50);
            let s_lim = ((w2_ratio * 128.0 + 0.5) as u8).min(64) as u64;
            if byte_uni1 & byte_uni2 {
                let b1 = (w1[0] & 0xFF) as gcso_q7_t;
                let b2 = (w2[0] & 0xFF) as gcso_q7_t;
                let b_lim = branchless_q7_circular_lerp(b1, b2, s_lim as u8);
                let w_lim = ((b_lim as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
                ([w_lim; NUM_SWARM_MICRO_CELLS], true, w2)
            } else if w1_uni & w2_uni {
                let bias_lim = (128u64.wrapping_sub(s_lim)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                let (tw, t_uni) =
                    swar_slerp_words_with_uniform_flag_64(&w1, true, &w2, true, s_lim, bias_lim);
                (tw, t_uni, w2)
            } else {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    unsafe {
                        use core::arch::x86_64::{__m256i, _mm256_loadu_si256};
                        let p1 = centroids[idx1].phase_profile.as_ptr().cast::<__m256i>();
                        let p2 = centroids[idx2].phase_profile.as_ptr().cast::<__m256i>();
                        let (v1_0, v1_1) = (_mm256_loadu_si256(p1), _mm256_loadu_si256(p1.add(1)));
                        let (v2_0, v2_1) = (_mm256_loadu_si256(p2), _mm256_loadu_si256(p2.add(1)));
                        let (vt0, vt1) = avx2_slerp_vec64(v1_0, v1_1, v2_0, v2_1, s_lim);
                        (avx2_vec64_to_words64(vt0, vt1), false, w2)
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    let bias_lim = (128u64.wrapping_sub(s_lim)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                    let (tw, t_uni) = swar_slerp_words_with_uniform_flag_64(
                        &w1, w1_uni, &w2, w2_uni, s_lim, bias_lim,
                    );
                    (tw, t_uni, w2)
                }
            }
        } else {
            (w1, w1_uni, [0u64; NUM_SWARM_MICRO_CELLS])
        };

        let win1 = &centroids[idx1];
        let aid1 = if win1.anchor_id != 0 { win1.anchor_id } else { (idx1 as u32) + 1 };
        let aid2 = if is_liminal {
            let win2 = &centroids[idx2];
            if win2.anchor_id != 0 { win2.anchor_id } else { (idx2 as u32) + 1 }
        } else {
            0
        };

        let tot_grav = branchless_select_f32(is_liminal, grav1 + 0.5 * grav2, grav1);
        let eff_coh = branchless_select_f32(is_liminal, 0.5 * (coh1 + coh2), coh1);
        let extra_flags = branchless_select_u32(is_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);
        let target_ref = if is_liminal {
            &target_w
        } else {
            win1.phase_words_ref()
        };
        let bridged_w = Self::apply_macro_centroid_pull_to_trail_64(
            trail,
            &trail_w,
            trail_is_uniform,
            target_ref,
            target_uni,
            win1,
            aid1,
            eff_coh,
            tot_grav,
            pull_step_u7,
            extra_flags,
        );
        let out_w2 = if is_liminal { w2_opt } else { bridged_w };

        (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, bridged_w, w1, out_w2)
    }

    #[inline(always)]
    pub(crate) fn route_top2_liminal_centroids_with_words_64(
        centroids: &[Self],
        trail: &mut gcso_pointer_trail_t,
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
    ) -> (
        usize,
        usize,
        u32,
        u32,
        f32,
        f32,
        bool,
        [u64; NUM_SWARM_MICRO_CELLS],
        [u64; NUM_SWARM_MICRO_CELLS],
        [u64; NUM_SWARM_MICRO_CELLS],
    ) {
        let trail_w = *trail_phase_words_ref(trail);
        Self::route_top2_liminal_centroids_from_preloaded_words_64(
            centroids,
            trail,
            trail_w,
            pull_step_u7,
            liminal_gravity_ratio_thresh,
        )
    }

    /// **Top-2 Liminal Bifurcation Macro Routing ($\mathcal{O}(K)$ 64-Bit SWAR)**:
    /// Evaluates the Top-2 nearest macro centroids $(k_1^*, k_2^*)$ on $SO(2)^{64}$ in a single pass.
    /// When the secondary centroid's gravitational pull satisfies $G_2 \ge \tau_{\mathrm{limina}} G_1$
    /// (e.g. [`DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO`]), interpolates along the shortest $S^1$
    /// geodesic between both centroids (bridging the *Limina* inter-memory boundary) and stamps
    /// `TRAIL_FLAG_LIMINAL_BEAT_ACTIVE | TRAIL_FLAG_SYNERGY_RESONANCE`.
    ///
    /// Returns `(primary_idx, secondary_idx, primary_anchor_id, secondary_anchor_id, blended_coherence, total_gravity, is_liminal_bifurcation)`.
    #[inline]
    pub fn route_top2_liminal_centroids_64(centroids: &[Self],
        trail: &mut gcso_pointer_trail_t, pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32) -> (usize, usize, u32, u32, f32, f32, bool) {
        let (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, _bw, _w1, _w2) =
            Self::route_top2_liminal_centroids_with_words_64(centroids, trail, pull_step_u7,
                liminal_gravity_ratio_thresh);
        (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal)
    }

    /// **Coarse-to-Fine Candidate-Trail Routing ($\mathcal{O}(K + M)$)**:
    /// Returns `(winning_centroid_idx, selected_trail_id, best_bayesian_score, best_coherence, found_coherent)`.
    #[inline]
    pub fn route_coarse_to_fine_candidates_64(centroids: &[Self],
        src_trail: &mut gcso_pointer_trail_t, candidate_trails: &[gcso_pointer_trail_t],
        head_dim: u32, phase_interference_gain: f32, stigmergic_prior_gain: f32, pointer_link_bonus: f32,
        cluster_bonus_gain: f32, contradiction_threshold: f32) -> (usize, u32, f32, f32, bool) {
        let self_w = *trail_phase_words_ref(src_trail);
        let (win_k, _win_coh, _win_grav, _empty_opt, _weak_idx, win_w) =
            Self::find_nearest_toroidal_centroid_words_and_profile_64(centroids, &self_w);
        let has_win_centroid = !centroids.is_empty() && centroids[win_k].sample_count > 0;

        let macro_anchor_id = if has_win_centroid {
            let aid = centroids[win_k].anchor_id;
            if aid != 0 {
                aid
            } else {
                (win_k as u32) + 1
            }
        } else {
            src_trail.cluster_id
        };
        if macro_anchor_id != 0 {
            src_trail.cluster_id = macro_anchor_id;
            src_trail.target_anchor_id = macro_anchor_id;
        }

        if candidate_trails.is_empty() { return (win_k, src_trail.linked_trail_id, 0.0, 0.0, false); }

        let safe_d = branchless_select_u32(head_dim > 0, head_dim, 128);
        let inv_sqrt_d = if safe_d <= 256 {
            branchless_popcount_rsqrt_u32(safe_d)
        } else {
            branchless_rsqrt_f32(safe_d as f32)
        };
        let phase_scale = branchless_flush_subnormal_f32(branchless_relu_f32(phase_interference_gain)) * inv_sqrt_d;
        let clean_eta = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_prior_gain));
        let clean_link = branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));
        let clean_cluster_bonus = branchless_flush_subnormal_f32(branchless_relu_f32(cluster_bonus_gain));
        let safe_thresh = branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let cur_linked_id = src_trail.linked_trail_id;
        let self_is_uniform = is_word_array_uniform_64(&self_w);
        let self_w0 = self_w[0];
        let self_b0 = (self_w0 & 0xFF) as u8;
        let self_is_byte_uniform =
            self_is_uniform & (self_w0 == (self_b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64));
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (self_v0, self_v1) = unsafe {
            super::branchless::avx2_words64_to_vec64_with_uniform(&self_w, self_is_uniform)
        };
        let mut win_meta_evaluated = false;
        let mut win_eq_self = false;
        let mut win_is_uniform = false;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (mut win_v0, mut win_v1) = (self_v0, self_v1);

        let mut best_id = cur_linked_id;
        let mut best_score = -1.0e30_f32;
        let mut best_coh = 0.0_f32;
        let mut found_valid = false;

        let mut idx = 0usize;
        while idx < candidate_trails.len() {
            let cand = &candidate_trails[idx];
            if (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0 {
                idx += 1;
                continue;
            }
            let cand_w = trail_phase_words_ref(cand);
            let cand_is_uniform = if self_is_uniform {
                is_word_array_uniform_64(cand_w)
            } else {
                false
            };
            let coh = if cand_is_uniform {
                let cw0 = cand_w[0];
                if cw0 == self_w0 {
                    1.0
                } else if self_is_byte_uniform
                    && (cw0 == ((cw0 & 0xFF) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64))
                {
                    Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
                } else {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &self_w, true, cand_w, true,
                    )
                }
            } else {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    unsafe {
                        use core::arch::x86_64::{__m256i, _mm256_loadu_si256, _mm256_sub_epi8};
                        let cp = cand.accumulated_phase_delta.as_ptr().cast::<__m256i>();
                        let c0 = _mm256_loadu_si256(cp);
                        let c1 = _mm256_loadu_si256(cp.add(1));
                        avx2_coherence_from_diff_vec64(
                            _mm256_sub_epi8(self_v0, c0),
                            _mm256_sub_epi8(self_v1, c1),
                        )
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &self_w,
                        self_is_uniform,
                        cand_w,
                        false,
                    )
                }
            };
            if coh >= -safe_thresh {
                let cand_id = idx as u32;
                let clean_rho = branchless_clamp_f32(branchless_relu_f32(cand.stigmergic_density),
                    0.0, MAX_STIGMERGIC_DENSITY);
                let prior_term = clean_eta * fast_ln_1p_bounded_f32(clean_rho);
                let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
                let in_macro_cluster = (macro_anchor_id != 0) & ((cand.cluster_id == macro_anchor_id)
                        | (cand.target_anchor_id == macro_anchor_id));
                let geo_macro_affinity = if in_macro_cluster {
                    1.0
                } else if has_win_centroid & (clean_cluster_bonus > 0.0) {
                    if !win_meta_evaluated {
                        win_eq_self = swar_words_equal_64(&self_w, &win_w);
                        win_is_uniform = if win_eq_self {
                            self_is_uniform
                        } else {
                            is_word_array_uniform_64(&win_w)
                        };
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        if !win_eq_self {
                            (win_v0, win_v1) = unsafe {
                                super::branchless::avx2_words64_to_vec64_with_uniform(
                                    &win_w,
                                    win_is_uniform,
                                )
                            };
                        }
                        win_meta_evaluated = true;
                    }
                    let c_win_coh = if win_eq_self {
                        coh
                    } else {
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        {
                            let cand_uni_for_win = if self_is_uniform {
                                cand_is_uniform
                            } else if win_is_uniform {
                                is_word_array_uniform_64(cand_w)
                            } else {
                                false
                            };
                            if win_is_uniform & cand_uni_for_win {
                                coherence_from_word_arrays_with_both_uniform_flags_64(
                                    &win_w, true, cand_w, true,
                                )
                            } else {
                                unsafe {
                                    use core::arch::x86_64::{
                                        __m256i, _mm256_loadu_si256, _mm256_sub_epi8,
                                    };
                                    let cp =
                                        cand.accumulated_phase_delta.as_ptr().cast::<__m256i>();
                                    let c0 = _mm256_loadu_si256(cp);
                                    let c1 = _mm256_loadu_si256(cp.add(1));
                                    avx2_coherence_from_diff_vec64(
                                        _mm256_sub_epi8(win_v0, c0),
                                        _mm256_sub_epi8(win_v1, c1),
                                    )
                                }
                            }
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                        {
                            let cand_uni_for_win = if self_is_uniform {
                                cand_is_uniform
                            } else if win_is_uniform {
                                is_word_array_uniform_64(cand_w)
                            } else {
                                false
                            };
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &win_w,
                                win_is_uniform,
                                cand_w,
                                cand_uni_for_win,
                            )
                        }
                    };
                    branchless_relu_f32(c_win_coh)
                } else {
                    0.0
                };
                let score = branchless_flush_subnormal_f32((phase_scale * coh)
                        + prior_term
                        + branchless_select_f32(is_linked, clean_link, 0.0)
                        + (clean_cluster_bonus * geo_macro_affinity));
                let take = (!found_valid) | (score > best_score);
                best_id = branchless_select_u32(take, cand_id, best_id);
                best_score = branchless_select_f32(take, score, best_score);
                best_coh = branchless_select_f32(take, coh, best_coh);
                found_valid = true;
            }
            idx += 1;
        }

        if found_valid {
            src_trail.linked_trail_id = best_id;
            src_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_CRYSTALLIZED_CANDIDATE;
        }
        let out_score = branchless_select_f32(found_valid, best_score, 0.0);
        (win_k, best_id, out_score, best_coh, found_valid)
    }

    /// Folds a slice of `.gcso` [`CrystallizedAttractorRecord`]s onto `trail_phases_inout`
    /// weighted by **Phase-Proximity Gravity** in 8 register-resident `u64` SWAR words,
    /// returning `(folded_words, total_pull, dominant_anchor_id, dominant_ripa_clamp_rad, mean_coherence)`
    /// in a single pass (`0` duplicate coherence scans).
    #[inline]
    pub fn fold_crystallized_ensemble_words_and_coherence_64(
        records: &[Self],
        trail_phases_inout: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        init_pull_weight: f32,
    ) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, u32, f32, f32) {
        let init_w = load_q7_words_64(trail_phases_inout);
        let init_is_uniform = is_word_array_uniform_64(&init_w);
        let init_w0 = init_w[0];
        let init_b0 = (init_w0 & 0xFF) as u8;
        let init_is_byte_uniform =
            init_is_uniform & (init_w0 == (init_b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64));
        let init_nonzero = if init_is_uniform {
            init_w0 != 0
        } else {
            (((init_w0 | init_w[1]) | (init_w[2] | init_w[3]))
                | ((init_w[4] | init_w[5]) | (init_w[6] | init_w[7])))
                != 0
        };
        let mut tot_pull = branchless_flush_subnormal_f32(branchless_relu_f32(init_pull_weight));
        if records.len() == 1 && !init_nonzero && tot_pull <= 1.0e-12 {
            let rec = &records[0];
            if rec.sample_count > 0 && rec.phase_gravity_mass > 1.0e-12 {
                let rec_w = *rec.phase_words_ref();
                let grav = branchless_phase_proximity_gravity_f32(
                    rec.phase_gravity_mass,
                    1.0,
                    DEFAULT_PHASE_GRAVITY_EPS,
                );
                store_q7_words_64(trail_phases_inout, &rec_w);
                return (rec_w, grav, rec.anchor_id, rec.ripa_clamp_rad, 1.0);
            }
            return (init_w, 0.0, 0, DEFAULT_RIPA_CLAMP_MAX_RAD, 0.0);
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let mut acc_w = init_w;
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let mut acc_is_uniform = init_is_uniform;
        let mut best_grav = 0.0_f32;
        let mut dom_id = 0u32;
        let mut dom_clamp = DEFAULT_RIPA_CLAMP_MAX_RAD;
        let mut coh_sum = 0.0_f32;
        let mut active_cnt = 0usize;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let acc_w = {
            // SAFETY: Guarded by `target_feature = "avx2"`. Fast-paths word-uniform records directly
            // in a single 64-bit GPR (`acc_w0`) with zero YMM loads or `vpextrq` extractions, and
            // seamlessly transitions to YMM-resident `acc_v0, acc_v1` if any record is non-uniform.
            unsafe {
                use core::arch::x86_64::{
                    __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x,
                    _mm256_sub_epi8, _mm256_testz_si256, _mm256_xor_si256,
                };
                let mut init_vec_ready = !init_is_uniform;
                let (mut init_v0, mut init_v1) = if init_vec_ready {
                    crate::traits::branchless::avx2_words64_to_vec64_with_uniform(&init_w, false)
                } else {
                    let z = _mm256_set1_epi64x(init_w[0] as i64);
                    (z, z)
                };
                let mut acc_uni = init_is_uniform;
                let mut acc_w0 = init_w[0];
                let mut acc_v0 = init_v0;
                let mut acc_v1 = init_v1;
                let mut i = 0usize;
                while i < records.len() {
                    let rec = &records[i];
                    if rec.sample_count > 0 && rec.phase_gravity_mass > 1.0e-12 {
                        let check_uni = acc_uni & (init_is_uniform | !init_nonzero);
                        let rec_w = rec.phase_words_ref();
                        let rec_uni = if check_uni {
                            is_word_array_uniform_64(rec_w)
                        } else {
                            false
                        };
                        if rec_uni & acc_uni & (init_is_uniform | !init_nonzero) {
                            let rw0 = rec_w[0];
                            let rb0 = (rw0 & 0xFF) as u8;
                            let (coh, grav) = if (!init_nonzero) | (rw0 == init_w0) {
                                (
                                    1.0,
                                    branchless_q7_default_gravity_from_diff_u8(
                                        rec.phase_gravity_mass,
                                        0,
                                    ),
                                )
                            } else if init_is_byte_uniform
                                && (rw0 == (rb0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64))
                            {
                                let d_u8 = init_b0.wrapping_sub(rb0);
                                (
                                    Q7_FULL_COS_REF[d_u8 as usize],
                                    branchless_q7_default_gravity_from_diff_u8(
                                        rec.phase_gravity_mass,
                                        d_u8,
                                    ),
                                )
                            } else {
                                let c_val = coherence_from_word_arrays_with_both_uniform_flags_64(
                                    &init_w, true, rec_w, true,
                                );
                                (
                                    c_val,
                                    branchless_phase_proximity_gravity_f32(
                                        rec.phase_gravity_mass,
                                        c_val,
                                        DEFAULT_PHASE_GRAVITY_EPS,
                                    ),
                                )
                            };
                            coh_sum += coh;
                            active_cnt += 1;
                            let new_tot = tot_pull + grav;
                            if grav > 1.0e-12 {
                                if tot_pull <= 1.0e-12 {
                                    acc_w0 = rw0;
                                } else if acc_w0 != rw0 {
                                    let safe_tot =
                                        branchless_select_f32(new_tot > 1.0e-12, new_tot, 1.0);
                                    let s_u8 = ((grav * fast_inv_pos_normal_f32(safe_tot)) * 128.0
                                        + 0.5) as u8;
                                    let s_u64 = s_u8.min(128) as u64;
                                    if s_u64 >= 128 {
                                        acc_w0 = rw0;
                                    } else if s_u64 > 0 {
                                        let bias_restore = (128u64.wrapping_sub(s_u64))
                                            .wrapping_mul(SWAR_SPLAT_BYTE_64);
                                        acc_w0 = super::branchless::swar_slerp_word_unchecked_x8(
                                            acc_w0,
                                            rw0,
                                            s_u64,
                                            bias_restore,
                                        );
                                    }
                                }
                            }
                            let is_dom = grav > best_grav;
                            best_grav = branchless_select_f32(is_dom, grav, best_grav);
                            dom_id = branchless_select_u32(is_dom, rec.anchor_id, dom_id);
                            dom_clamp =
                                branchless_select_f32(is_dom, rec.ripa_clamp_rad, dom_clamp);
                            tot_pull = new_tot;
                        } else {
                            if !init_vec_ready {
                                let z = _mm256_set1_epi64x(init_w[0] as i64);
                                init_v0 = z;
                                init_v1 = z;
                                init_vec_ready = true;
                            }
                            if acc_uni {
                                let az = _mm256_set1_epi64x(acc_w0 as i64);
                                acc_v0 = az;
                                acc_v1 = az;
                                acc_uni = false;
                            }
                            let rec_ptr = rec.phase_profile.as_ptr().cast::<__m256i>();
                            let rv0 = _mm256_loadu_si256(rec_ptr);
                            let rv1 = _mm256_loadu_si256(rec_ptr.add(1));
                            let coh = if !init_nonzero {
                                1.0
                            } else {
                                avx2_coherence_from_diff_vec64(
                                    _mm256_sub_epi8(init_v0, rv0),
                                    _mm256_sub_epi8(init_v1, rv1),
                                )
                            };
                            coh_sum += coh;
                            active_cnt += 1;
                            let grav = branchless_phase_proximity_gravity_f32(
                                rec.phase_gravity_mass,
                                coh,
                                DEFAULT_PHASE_GRAVITY_EPS,
                            );
                            let new_tot = tot_pull + grav;
                            if grav > 1.0e-12 {
                                if tot_pull <= 1.0e-12 {
                                    if rec_uni {
                                        acc_w0 = rec_w[0];
                                        acc_uni = true;
                                    } else {
                                        acc_v0 = rv0;
                                        acc_v1 = rv1;
                                    }
                                } else {
                                    let diff_or = _mm256_or_si256(
                                        _mm256_xor_si256(acc_v0, rv0),
                                        _mm256_xor_si256(acc_v1, rv1),
                                    );
                                    if _mm256_testz_si256(diff_or, diff_or) == 0 {
                                        let safe_tot =
                                            branchless_select_f32(new_tot > 1.0e-12, new_tot, 1.0);
                                        let s_u8 = ((grav * fast_inv_pos_normal_f32(safe_tot))
                                            * 128.0
                                            + 0.5)
                                            as u8;
                                        let s_u64 = s_u8.min(128) as u64;
                                        if s_u64 >= 128 {
                                            acc_v0 = rv0;
                                            acc_v1 = rv1;
                                        } else if s_u64 > 0 {
                                            (acc_v0, acc_v1) =
                                                avx2_slerp_vec64(acc_v0, acc_v1, rv0, rv1, s_u64);
                                        }
                                    }
                                }
                            }

                            let is_dom = grav > best_grav;
                            best_grav = branchless_select_f32(is_dom, grav, best_grav);
                            dom_id = branchless_select_u32(is_dom, rec.anchor_id, dom_id);
                            dom_clamp =
                                branchless_select_f32(is_dom, rec.ripa_clamp_rad, dom_clamp);
                            tot_pull = new_tot;
                        }
                    }
                    i += 1;
                }
                if acc_uni {
                    [acc_w0; NUM_SWARM_MICRO_CELLS]
                } else {
                    crate::traits::branchless::avx2_vec64_to_words64(acc_v0, acc_v1)
                }
            }
        };

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut i = 0usize;
            while i < records.len() {
                let rec = &records[i];
                if rec.sample_count > 0 && rec.phase_gravity_mass > 1.0e-12 {
                    let rec_w = rec.phase_words_ref();
                    let rec_is_uniform = if (init_nonzero & init_is_uniform) | acc_is_uniform {
                        is_word_array_uniform_64(rec_w)
                    } else {
                        false
                    };
                    let rw0 = rec_w[0];
                    let rb0 = (rw0 & 0xFF) as u8;
                    let (coh, grav) = if (!init_nonzero)
                        | (rec_is_uniform & init_is_uniform & (rw0 == init_w0))
                    {
                        (
                            1.0,
                            branchless_q7_default_gravity_from_diff_u8(rec.phase_gravity_mass, 0),
                        )
                    } else if rec_is_uniform
                        & init_is_byte_uniform
                        & (rw0 == (rb0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64))
                    {
                        let d_u8 = init_b0.wrapping_sub(rb0);
                        (
                            Q7_FULL_COS_REF[d_u8 as usize],
                            branchless_q7_default_gravity_from_diff_u8(
                                rec.phase_gravity_mass,
                                d_u8,
                            ),
                        )
                    } else {
                        let c_val = coherence_from_word_arrays_with_both_uniform_flags_64(
                            &init_w,
                            init_is_uniform,
                            rec_w,
                            rec_is_uniform,
                        );
                        (
                            c_val,
                            branchless_phase_proximity_gravity_f32(
                                rec.phase_gravity_mass,
                                c_val,
                                DEFAULT_PHASE_GRAVITY_EPS,
                            ),
                        )
                    };
                    coh_sum += coh;
                    active_cnt += 1;
                    let new_tot = tot_pull + grav;
                    if grav > 1.0e-12 {
                        if tot_pull <= 1.0e-12 {
                            acc_w = *rec_w;
                            acc_is_uniform = rec_is_uniform;
                        } else if !swar_words_equal_64(&acc_w, rec_w) {
                            let safe_tot = branchless_select_f32(new_tot > 1.0e-12, new_tot, 1.0);
                            let s_u8 = ((grav * fast_inv_pos_normal_f32(safe_tot)) * 128.0 + 0.5) as u8;
                            let s_u64 = s_u8.min(128) as u64;
                            if s_u64 >= 128 {
                                acc_w = *rec_w;
                                acc_is_uniform = rec_is_uniform;
                            } else if s_u64 > 0 {
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
                        }
                    }

                    let is_dom = grav > best_grav;
                    best_grav = branchless_select_f32(is_dom, grav, best_grav);
                    dom_id = branchless_select_u32(is_dom, rec.anchor_id, dom_id);
                    dom_clamp = branchless_select_f32(is_dom, rec.ripa_clamp_rad, dom_clamp);
                    tot_pull = new_tot;
                }
                i += 1;
            }
        }

        store_q7_words_64(trail_phases_inout, &acc_w);
        let mean_coh = if active_cnt == 0 {
            0.0
        } else {
            branchless_clamp_f32(coh_sum * fast_inv_len_f32(active_cnt), -1.0, 1.0)
        };
        (acc_w, tot_pull, dom_id, dom_clamp, mean_coh)
    }

    /// Folds a slice of `.gcso` [`CrystallizedAttractorRecord`]s onto `trail_phases_inout`
    /// weighted by **Phase-Proximity Gravity** in 8 register-resident `u64` SWAR words,
    /// evaluating proximity coherence against the initial query profile (`init_w`) and
    /// fast-pathing uninitialized/zero query trails and the first active attractor (`tot_pull <= 1.0e-12`).
    #[inline(always)]
    pub fn fold_crystallized_ensemble_words_with_init_pull_64(
        records: &[Self],
        trail_phases_inout: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        init_pull_weight: f32,
    ) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, u32, f32) {
        let (acc_w, tot_pull, dom_id, dom_clamp, _mean_coh) =
            Self::fold_crystallized_ensemble_words_and_coherence_64(
                records,
                trail_phases_inout,
                init_pull_weight,
            );
        (acc_w, tot_pull, dom_id, dom_clamp)
    }

    /// Folds a slice of `.gcso` [`CrystallizedAttractorRecord`]s onto `trail_phases_inout`
    /// weighted by Phase-Proximity Gravity (`init_pull_weight = 0.0`).
    #[inline(always)]
    pub fn fold_crystallized_ensemble_64(records: &[Self],
        trail_phases_inout: &mut [gcso_q7_t; MAX_ATTENTION_HEADS]) -> (f32, u32, f32) {
        let (_w, tot_pull, dom_id, dom_clamp) = Self::fold_crystallized_ensemble_words_with_init_pull_64(
                records, trail_phases_inout, 0.0);
        (tot_pull, dom_id, dom_clamp)
    }

    /// Zero-copy deserialization of `count` [`CrystallizedAttractorRecord`]s from a 64-byte aligned
    /// `.gcso` (ZIMMS) memory-mapped byte slice with header magic/version and CRC32 validation.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`], [`GCSO_ERROR_BUFFER_TOO_SMALL`], or
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`] on validation failure.
    #[inline]
    pub fn slice_from_zimms_bytes_zero_copy(bytes: &[u8], count: usize) -> GcsoResult<&[Self]> {
        Self::slice_from_zimms_bytes_with_crc_mode(bytes, count, true)
    }

    #[inline(always)]
    fn slice_from_zimms_bytes_with_crc_mode(bytes: &[u8], count: usize, verify_per_record_crc: bool,
    ) -> GcsoResult<&[Self]> {
        if count == 0 { return Ok(&[]); }
        let req_bytes = count.checked_mul(size_of::<Self>()).ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if bytes.len() < req_bytes { return Err(GCSO_ERROR_BUFFER_TOO_SMALL); }
        if !is_aligned_to(bytes, align_of::<Self>()) { return Err(GCSO_ERROR_MISALIGNED_POINTER); }
        // SAFETY: `bytes` is verified 64-byte aligned and at least `count * 128` bytes long, and `Self: Pod`.
        let recs = unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<Self>(), count) };
        let mut i = 0usize;
        if verify_per_record_crc {
            while i < count {
                if !recs[i].verify_integrity() { return Err(GCSO_ERROR_CONTAINER_CORRUPTED); }
                i += 1;
            }
        } else {
            while i < count {
                let r = &recs[i];
                if (r.magic != GCSO_CRYSTAL_RECORD_MAGIC)
                    | (r.version != GCSO_CRYSTAL_RECORD_VERSION)
                    | (r.checksum_crc32 == 0)
                { return Err(GCSO_ERROR_CONTAINER_CORRUPTED); }
                i += 1;
            }
        }
        Ok(recs)
    }

    /// Serializes `records` into a `.gcso` (ZIMMS) byte slice (`dst_bytes`), stamping each
    /// record's IEEE 802.3 CRC32 checksum when not already stamped.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < records.len() * 128`.
    #[inline]
    pub fn write_records_to_zimms_bytes(records: &[Self], dst_bytes: &mut [u8]) -> GcsoResult<usize> {
        let total_bytes = records.len().checked_mul(size_of::<Self>()).ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if dst_bytes.len() < total_bytes { return Err(GCSO_ERROR_BUFFER_TOO_SMALL); }
        if total_bytes == 0 { return Ok(0); }
        dst_bytes[..total_bytes].copy_from_slice(Self::slice_as_bytes(records));
        let mut idx = 0usize;
        while idx < records.len() {
            let r = &records[idx];
            let valid_hdr =
                (r.magic == GCSO_CRYSTAL_RECORD_MAGIC) & (r.version == GCSO_CRYSTAL_RECORD_VERSION);
            if (!valid_hdr) | (r.checksum_crc32 == 0) {
                let base_off = idx * size_of::<Self>();
                let crc = if valid_hdr {
                    r.compute_payload_crc32()
                } else {
                    let mut rec = *r;
                    rec.magic = GCSO_CRYSTAL_RECORD_MAGIC;
                    rec.version = GCSO_CRYSTAL_RECORD_VERSION;
                    dst_bytes[base_off..base_off + 4]
                        .copy_from_slice(&GCSO_CRYSTAL_RECORD_MAGIC.to_le_bytes());
                    dst_bytes[base_off + 4..base_off + 8]
                        .copy_from_slice(&GCSO_CRYSTAL_RECORD_VERSION.to_le_bytes());
                    rec.compute_payload_crc32()
                };
                dst_bytes[base_off + 52..base_off + 56].copy_from_slice(&crc.to_le_bytes());
            }
            idx += 1;
        }
        Ok(total_bytes)
    }
}

impl CausalProvenanceSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        self.provenance_bits | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0)
    }
}

impl CognitiveGroundingSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] { self.phase_profile }
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> { Some(&self.phase_profile) }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] { *self.phase_words_ref() }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 { self.phase_gravity_mass }
    #[inline(always)]
    fn zimms_section_offset48(&self) -> u64 { self.zimms_section_offset48 }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 { self.anchor_id }
}

impl ObjectiveFactSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn fact_ripa_clamp_rad(&self) -> f32 { self.ripa_clamp_rad }
}

impl SubjectivePersonaSignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn persona_ripa_clamp_rad(&self) -> f32 { self.ripa_clamp_rad }
}

impl ManifoldGatewaySignal for CrystallizedAttractorRecord {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > 1.0e-12);
        if !is_active {
            let packed_tag = pack_action_slot_tag(self.dominant_slot, ACTION_PRESET_PASSIVE_SIDECAR);
            return (GatewayFrame {
                    tagged_ptr: gcso_tagged_ptr_t::pack(self.zimms_section_offset48, packed_tag, 0).raw(),
                    zimms_section_offset48: self.zimms_section_offset48, target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot, action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask, dominant_cause_code: self.dominant_cause_code,
                    frame_flags: packed_tag as u16,..GatewayFrame::default()
                },
                [0u64; NUM_SWARM_MICRO_CELLS]);
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let tag_q7 = fast_mean_q7_tag_from_words_64(&w);
        let eff_action = self.action_mode_flags;
        let packed_tag = pack_action_slot_tag(self.dominant_slot, eff_action);
        let tagged_ptr = gcso_tagged_ptr_t::pack(self.zimms_section_offset48, packed_tag, tag_q7).raw();
        let prov_bits = self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let pspm_fact = branchless_select_u8(is_fact_slot, 36, branchless_select_u8(is_persona_slot, 16, 24));
        let pspm_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let pspm_explore = branchless_select_u8(is_fact_slot, 8, branchless_select_u8(is_persona_slot, 24, 16));
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (GatewayFrame {
                tagged_ptr, zimms_section_offset48: self.zimms_section_offset48,
                target_anchor_id: self.anchor_id, effective_energy: branchless_min_f32(
                    self.phase_gravity_mass, MAX_ATTRACTOR_PULL_FORCE), ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD, effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar, repulsion_severity: 0.0,
                provenance_bits: prov_bits, dominant_cause_code: self.dominant_cause_code,
                dominant_slot: self.dominant_slot, action_mode_flags: eff_action,
                active_swarm_cell_mask: self.active_swarm_mask,
                qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
                pspm_fact_heads: pspm_fact, pspm_logic_heads: pspm_logic,
                pspm_explore_heads: pspm_explore, pspm_fact_gain_q7: fact_gain_q7,
                pspm_logic_gain_q7: logic_gain_q7, pspm_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7, active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16, phase_deltas: self.phase_profile,
            },
            w)
    }

    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let is_active = (self.sample_count > 0) & (self.phase_gravity_mass > 1.0e-12);
        if !is_active {
            return (GatewayFrame {
                    zimms_section_offset48: self.zimms_section_offset48, target_anchor_id: self.anchor_id,
                    dominant_slot: self.dominant_slot, action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
                    active_swarm_cell_mask: self.active_swarm_mask, dominant_cause_code: self.dominant_cause_code,
                    ..GatewayFrame::default()
                },
                [0u64; NUM_SWARM_MICRO_CELLS]);
        }
        let is_mature = (self.sample_count >= 2) & (self.resultant_r_bar >= 0.45);
        let w = *self.phase_words_ref();
        let prov_bits = self.provenance_bits
            | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);
        let is_fact_slot = self.dominant_slot == ATTRACTOR_SLOT_KNOWLEDGE;
        let is_persona_slot = self.dominant_slot == ATTRACTOR_SLOT_PERSONA;
        let pspm_fact = branchless_select_u8(is_fact_slot, 36, branchless_select_u8(is_persona_slot, 16, 24));
        let pspm_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let pspm_explore = branchless_select_u8(is_fact_slot, 8, branchless_select_u8(is_persona_slot, 24, 16));
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        (GatewayFrame {
                tagged_ptr: 0, zimms_section_offset48: self.zimms_section_offset48,
                target_anchor_id: self.anchor_id, effective_energy: branchless_min_f32(
                    self.phase_gravity_mass, MAX_ATTRACTOR_PULL_FORCE), ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD, effective_inv_temp: self.effective_inv_temp,
                fact_persona_coherence: self.resultant_r_bar, repulsion_severity: 0.0,
                provenance_bits: prov_bits, dominant_cause_code: self.dominant_cause_code,
                dominant_slot: self.dominant_slot, action_mode_flags: self.action_mode_flags,
                active_swarm_cell_mask: self.active_swarm_mask,
                qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
                pspm_fact_heads: pspm_fact, pspm_logic_heads: pspm_logic,
                pspm_explore_heads: pspm_explore, pspm_fact_gain_q7: fact_gain_q7,
                pspm_logic_gain_q7: logic_gain_q7, pspm_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: 0, active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: 0, phase_deltas: [0i8; MAX_ATTENTION_HEADS],
            },
            w)
    }

    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        (self.sample_count > 0) & (self.phase_gravity_mass > 1.0e-12)
    }

    #[inline(always)]
    fn synthesize_gateway_slice_with_words(
        records: &[Self],
    ) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let len = records.len();
        if len == 0 { return (GatewayFrame::default(), [0u64; NUM_SWARM_MICRO_CELLS]); }
        if len == 1 { return records[0].synthesize_gateway_frame_with_words(); }

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
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

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (acc_w, avx2_tag_q7) = {
            // SAFETY: Guarded by `target_feature = "avx2"`. Fast-paths word-uniform records directly
            // in a single 64-bit GPR (`acc_w0`) with zero YMM loads or `vpextrq` extractions, and
            // seamlessly transitions to YMM-resident `acc_v0, acc_v1` if any record is non-uniform.
            unsafe {
                use core::arch::x86_64::{
                    __m256i, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi64x,
                    _mm256_setzero_si256, _mm256_sub_epi8, _mm256_testz_si256,
                };
                let mut acc_w0 = 0u64;
                let mut acc_v0 = _mm256_setzero_si256();
                let mut acc_v1 = _mm256_setzero_si256();
                let mut i = 0usize;
                while i < len {
                    let rec = &records[i];
                    if (rec.sample_count > 0) & (rec.phase_gravity_mass > 1.0e-12) {
                        let rec_w = rec.phase_words_ref();
                        let f_e = branchless_min_f32(
                            branchless_flush_subnormal_f32(branchless_relu_f32(
                                rec.phase_gravity_mass,
                            )),
                            MAX_ATTRACTOR_PULL_FORCE,
                        );
                        let is_mature = (rec.sample_count >= 2) & (rec.resultant_r_bar >= 0.45);
                        let rec_prov = rec.provenance_bits
                            | branchless_select_u32(
                                is_mature,
                                CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
                                0,
                            );

                        if !has_active {
                            acc_is_uniform = is_word_array_uniform_64(rec_w);
                            if acc_is_uniform {
                                acc_w0 = rec_w[0];
                            } else {
                                let rec_ptr = rec.phase_profile.as_ptr().cast::<__m256i>();
                                acc_v0 = _mm256_loadu_si256(rec_ptr);
                                acc_v1 = _mm256_loadu_si256(rec_ptr.add(1));
                            }
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
                        } else {
                            if f_e > 1.0e-12 {
                                let new_tot = tot_e + f_e;
                                let rw0 = rec_w[0];
                                if acc_is_uniform && rw0 == acc_w0 && is_word_array_uniform_64(rec_w) {
                                    // Identical uniform phase profile: skip reciprocal & SLERP completely.
                                } else {
                                    let safe_tot =
                                        branchless_select_f32(new_tot > 1.0e-12, new_tot, 1.0);
                                    let w_ratio = branchless_clamp_f32(
                                        f_e * fast_inv_pos_normal_f32(safe_tot),
                                        0.0,
                                        1.0,
                                    );
                                    let s_u8 = ((w_ratio * 128.0 + 0.5) as u8).min(128);
                                    let s_u64 = s_u8 as u64;
                                    let rec_uni = if acc_is_uniform || s_u64 >= 128 {
                                        is_word_array_uniform_64(rec_w)
                                    } else {
                                        false
                                    };
                                    if acc_is_uniform && rec_uni {
                                        if s_u64 >= 128 {
                                            acc_w0 = rw0;
                                        } else if s_u64 > 0 && acc_w0 != rw0 {
                                            let ab0 = (acc_w0 & 0xFF) as u8;
                                            let rb0 = (rw0 & 0xFF) as u8;
                                            if acc_w0 == acc_w0.rotate_right(8)
                                                && rw0 == rw0.rotate_right(8)
                                            {
                                            let b_lerp = branchless_q7_circular_lerp(
                                                ab0 as gcso_q7_t,
                                                rb0 as gcso_q7_t,
                                                s_u8,
                                            );
                                            acc_w0 = ((b_lerp as u8) as u64)
                                                .wrapping_mul(SWAR_SPLAT_BYTE_64);
                                        } else {
                                            let bias_restore = (128u64.wrapping_sub(s_u64))
                                                .wrapping_mul(SWAR_SPLAT_BYTE_64);
                                                acc_w0 = super::branchless::swar_slerp_word_unchecked_x8(
                                                    acc_w0,
                                                    rw0,
                                                    s_u64,
                                                    bias_restore,
                                                );
                                            }
                                        }
                                    } else {
                                        if acc_is_uniform {
                                            let az = _mm256_set1_epi64x(acc_w0 as i64);
                                            acc_v0 = az;
                                            acc_v1 = az;
                                            acc_is_uniform = false;
                                        }
                                        let rec_ptr = rec.phase_profile.as_ptr().cast::<__m256i>();
                                        let fv0 = _mm256_loadu_si256(rec_ptr);
                                        let fv1 = _mm256_loadu_si256(rec_ptr.add(1));
                                        if s_u64 >= 128 {
                                            if rec_uni {
                                                acc_w0 = rec_w[0];
                                                acc_is_uniform = true;
                                            } else {
                                                acc_v0 = fv0;
                                                acc_v1 = fv1;
                                            }
                                        } else if s_u64 > 0 {
                                            let diff0 = _mm256_sub_epi8(fv0, acc_v0);
                                            let diff1 = _mm256_sub_epi8(fv1, acc_v1);
                                            let diff_or = _mm256_or_si256(diff0, diff1);
                                            if _mm256_testz_si256(diff_or, diff_or) == 0 {
                                                (acc_v0, acc_v1) =
                                                    super::branchless::avx2_slerp_from_diff_vec64(
                                                        acc_v0, acc_v1, diff0, diff1, s_u64,
                                                    );
                                            }
                                        }
                                    }
                                }
                                weighted_inv_temp += f_e * rec.effective_inv_temp;
                                weighted_coh += f_e * rec.resultant_r_bar;

                                let take_dom = f_e > best_e;
                                best_e = branchless_select_f32(take_dom, f_e, best_e);
                                dom_idx = branchless_select_usize(take_dom, i, dom_idx);
                                dom_zimms_addr = branchless_select_u64(
                                    take_dom & (rec.zimms_section_offset48 != 0),
                                    rec.zimms_section_offset48,
                                    dom_zimms_addr,
                                );
                                tot_e = new_tot;
                            }
                            action_flags |= rec.action_mode_flags;
                            swarm_mask &= rec.active_swarm_mask;
                            prov_bits |= rec_prov;
                            if rec.dominant_cause_code > dom_cause {
                                dom_cause = rec.dominant_cause_code;
                            }
                        }
                    }
                    i += 1;
                }
                if acc_is_uniform {
                    let w_arr = [acc_w0; NUM_SWARM_MICRO_CELLS];
                    (
                        w_arr,
                        swar_q7_mean_i8_from_words_with_uniform_flag_64(&w_arr, true),
                    )
                } else {
                    let tag = super::branchless::avx2_q7_mean_i8_from_vec64(acc_v0, acc_v1);
                    (
                        crate::traits::branchless::avx2_vec64_to_words64(acc_v0, acc_v1),
                        tag,
                    )
                }
            }
        };

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut i = 0usize;
            while i < len {
                let rec = &records[i];
                if (rec.sample_count > 0) & (rec.phase_gravity_mass > 1.0e-12) {
                    let f_w = *rec.phase_words_ref();
                    let f_e = branchless_min_f32(
                        branchless_flush_subnormal_f32(branchless_relu_f32(rec.phase_gravity_mass)),
                        MAX_ATTRACTOR_PULL_FORCE,
                    );
                    let is_mature = (rec.sample_count >= 2) & (rec.resultant_r_bar >= 0.45);
                    let rec_prov = rec.provenance_bits
                        | branchless_select_u32(is_mature, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

                    if !has_active {
                        acc_w = f_w;
                        acc_is_uniform = is_word_array_uniform_64(&f_w);
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
                    } else {
                        if f_e > 1.0e-12 {
                            let new_tot = tot_e + f_e;
                            let safe_tot = branchless_select_f32(new_tot > 1.0e-12, new_tot, 1.0);
                            let w_ratio = branchless_clamp_f32(f_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
                            let s_u64 = ((w_ratio * 128.0 + 0.5) as u8).min(128) as u64;
                            if s_u64 >= 128 {
                                acc_w = f_w;
                                acc_is_uniform = is_word_array_uniform_64(&f_w);
                            } else if s_u64 > 0 && !swar_words_equal_64(&acc_w, &f_w) {
                                let f_is_uniform = acc_is_uniform && is_word_array_uniform_64(&f_w);
                                let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
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
                            weighted_inv_temp += f_e * rec.effective_inv_temp;
                            weighted_coh += f_e * rec.resultant_r_bar;

                            let take_dom = f_e > best_e;
                            best_e = branchless_select_f32(take_dom, f_e, best_e);
                            dom_idx = branchless_select_usize(take_dom, i, dom_idx);
                            dom_zimms_addr = branchless_select_u64(
                                take_dom & (rec.zimms_section_offset48 != 0),
                                rec.zimms_section_offset48,
                                dom_zimms_addr,
                            );
                            tot_e = new_tot;
                        }
                        action_flags |= rec.action_mode_flags;
                        swarm_mask &= rec.active_swarm_mask;
                        prov_bits |= rec_prov;
                        if rec.dominant_cause_code > dom_cause {
                            dom_cause = rec.dominant_cause_code;
                        }
                    }
                }
                i += 1;
            }
        }

        if !has_active { return records[0].synthesize_gateway_frame_with_words(); }

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
        let pspm_fact = branchless_select_u8(is_fact_slot, 36, branchless_select_u8(is_persona_slot, 16, 24));
        let pspm_logic = branchless_select_u8(is_fact_slot, 20, 24);
        let pspm_explore = branchless_select_u8(is_fact_slot, 8, branchless_select_u8(is_persona_slot, 24, 16));
        let fact_gain_q7 = branchless_select_u8(is_fact_slot, 147, 128);
        let explore_gain_q7 = branchless_select_u8(is_persona_slot, 154, 128);
        let logic_gain_q7 = ((fact_gain_q7 as u16 + explore_gain_q7 as u16) >> 1) as u8;

        let phase_deltas = words_to_q7_array_64(&acc_w);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let tag_q7 = avx2_tag_q7;
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&acc_w, acc_is_uniform);
        let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
        let tagged_ptr = gcso_tagged_ptr_t::pack(dom_zimms_addr, packed_tag, tag_q7).raw();

        (GatewayFrame {
                tagged_ptr, zimms_section_offset48: dom_zimms_addr,
                target_anchor_id: dom.anchor_id,
                effective_energy: branchless_min_f32(tot_e, MAX_ATTRACTOR_PULL_FORCE),
                ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
                qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
                effective_inv_temp: eff_inv_temp, fact_persona_coherence: fp_coh,
                repulsion_severity: 0.0, provenance_bits: prov_bits,
                dominant_cause_code: dom_cause, dominant_slot: dom_slot,
                action_mode_flags: action_flags, active_swarm_cell_mask: swarm_mask,
                qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
                pspm_fact_heads: pspm_fact, pspm_logic_heads: pspm_logic,
                pspm_explore_heads: pspm_explore, pspm_fact_gain_q7: fact_gain_q7,
                pspm_logic_gain_q7: logic_gain_q7, pspm_explore_gain_q7: explore_gain_q7,
                mean_phase_tag_q7: tag_q7, active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16, phase_deltas,
            },
            acc_w)
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame { self.synthesize_gateway_frame_with_words().0 }
}

// ===================================================================
// 8.5. 4-Section `.gcso` (ZIMMS) Zero-Copy Container View (`GcsoFourSectionView`)
// ===================================================================

/// Zero-copy view over a 128B-aligned 4-section `.gcso` (ZIMMS) memory-mapped container.
#[derive(Debug, Clone, Copy, PartialEq)] pub struct GcsoFourSectionView<'a> {
    /// Section 0a: Validated 128-byte `.gcso` snapshot header (`offset 0..128`).
    pub header: &'a gcso_snapshot_header_t,
    /// Section 0b: Validated 64-byte ZIMMS memory-map & track descriptor (`offset 128..192`).
    pub zimms_descriptor: &'a gcso_zimms_descriptor_t,
    /// Section 1: Zero-copy slice of crystallized macro attractor centroids (`128B` each).
    pub crystals: &'a [CrystallizedAttractorRecord],
    /// Section 2: Zero-copy slice of stigmergic pointer trails (`128B` each).
    pub trails: &'a [gcso_pointer_trail_t],
    /// Section 3a: Optional zero-copy reference to the 64-byte Contrastive Phase Bloom Filter.
    pub bloom: Option<&'a ContrastivePhaseBloomFilter>,
    /// Section 3b: Optional zero-copy reference to the 64-byte DAES telemetry & shortcut slot.
    pub daes: Option<&'a gcso_daes_slot_t>,
    /// Section 3c: Zero-copy slice of 16-byte [`CausalProvenanceReport`] diagnostic history entries.
    pub provenance_reports: &'a [CausalProvenanceReport],
}

impl<'a> GcsoFourSectionView<'a> {
    /// Computes the exact 128-byte-aligned container byte length required for the 4 sections (`num_reports = 0`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] on arithmetic overflow.
    #[inline]
    pub const fn required_byte_len(num_crystals: usize, num_trails: usize, has_bloom: bool,
        has_daes: bool) -> GcsoResult<usize> {
        Self::required_byte_len_with_reports(num_crystals, num_trails, has_bloom, has_daes, 0)
    }

    /// Computes the exact 128-byte-aligned container byte length required for the 4 sections including `num_reports`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] on arithmetic overflow.
    #[inline]
    pub const fn required_byte_len_with_reports(num_crystals: usize, num_trails: usize, has_bloom: bool,
        has_daes: bool, num_reports: usize) -> GcsoResult<usize> {
        let off_crystals = GCSO_FOUR_SECTION_HEADER_BYTES;
        let Some(crystals_bytes) = num_crystals.checked_mul(size_of::<CrystallizedAttractorRecord>()) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let Some(after_crystals) = off_crystals.checked_add(crystals_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let off_trails = (after_crystals + 127) & !127usize;
        let Some(trails_bytes) = num_trails.checked_mul(size_of::<gcso_pointer_trail_t>()) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let Some(off_bloom) = off_trails.checked_add(trails_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let bloom_bytes = if has_bloom {
            size_of::<ContrastivePhaseBloomFilter>()
        } else {
            0
        };
        let Some(off_daes) = off_bloom.checked_add(bloom_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let daes_bytes = if has_daes {
            size_of::<gcso_daes_slot_t>()
        } else {
            0
        };
        let Some(off_reports) = off_daes.checked_add(daes_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let Some(reports_bytes) = num_reports.checked_mul(size_of::<CausalProvenanceReport>()) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let Some(raw_end) = off_reports.checked_add(reports_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let min_end = if raw_end < GCSO_FOUR_SECTION_HEADER_BYTES {
            GCSO_FOUR_SECTION_HEADER_BYTES
        } else {
            raw_end
        };
        Ok((min_end + 127) & !127usize)
    }

    /// Serializes `crystals`, `trails`, `bloom_opt`, and `daes_opt` into `dst_bytes` and stamps IEEE 802.3 CRC32s.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < required_byte_len(...)`.
    #[inline]
    pub fn write_four_section_container_to_bytes(crystals: &[CrystallizedAttractorRecord],
        trails: &[gcso_pointer_trail_t], bloom_opt: Option<&ContrastivePhaseBloomFilter>,
        daes_opt: Option<&gcso_daes_slot_t>, timestamp_epoch_sec: u64, dst_bytes: &mut [u8]) -> GcsoResult<usize> {
        Self::write_four_section_with_reports_to_bytes(crystals, trails, bloom_opt,
            daes_opt, &[], timestamp_epoch_sec, dst_bytes)
    }

    /// Serializes `crystals`, `trails`, `bloom_opt`, `daes_opt`, and `provenance_reports` into `dst_bytes`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < required_byte_len_with_reports(...)`.
    #[inline]
    pub fn write_four_section_with_reports_to_bytes(crystals: &[CrystallizedAttractorRecord],
        trails: &[gcso_pointer_trail_t], bloom_opt: Option<&ContrastivePhaseBloomFilter>,
        daes_opt: Option<&gcso_daes_slot_t>, provenance_reports: &[CausalProvenanceReport],
        timestamp_epoch_sec: u64, dst_bytes: &mut [u8]) -> GcsoResult<usize> {
        let total_len = Self::required_byte_len_with_reports(crystals.len(), trails.len(), bloom_opt.is_some(),
            daes_opt.is_some(), provenance_reports.len())?;
        if dst_bytes.len() < total_len { return Err(GCSO_ERROR_BUFFER_TOO_SMALL); }

        let buf = &mut dst_bytes[..total_len];
        let off_crystals = GCSO_FOUR_SECTION_HEADER_BYTES;
        let crystals_bytes = CrystallizedAttractorRecord::write_records_to_zimms_bytes(crystals,
            &mut buf[off_crystals..])?;
        let after_crystals = off_crystals + crystals_bytes;
        let off_trails = (after_crystals + 127) & !127usize;
        if off_trails > after_crystals { buf[after_crystals..off_trails].fill(0); }

        gcso_pointer_trail_t::write_unaligned_slice_to(trails, buf, off_trails)?;
        let off_bloom = off_trails + core::mem::size_of_val(trails);
        if let Some(bloom) = bloom_opt { bloom.write_unaligned_to(buf, off_bloom)?; }
        let off_daes = off_bloom
            + if bloom_opt.is_some() {
                size_of::<ContrastivePhaseBloomFilter>()
            } else {
                0
            };
        if let Some(daes) = daes_opt { daes.write_unaligned_to(buf, off_daes)?; }
        let off_reports = off_daes
            + if daes_opt.is_some() {
                size_of::<gcso_daes_slot_t>()
            } else {
                0
            };
        CausalProvenanceReport::write_unaligned_slice_to(provenance_reports, buf, off_reports)?;
        let end_reports = off_reports + core::mem::size_of_val(provenance_reports);
        if total_len > end_reports { buf[end_reports..total_len].fill(0); }

        if let Ok(desc_out) =
            gcso_zimms_descriptor_t::from_bytes_mut_at(buf, GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET)
        {
            desc_out.mapped_address = 0;
            desc_out.file_size_bytes = total_len as u64;
            desc_out.dma_buffer_handle = 0;
            desc_out.flags = 4;
            desc_out.fd_handle = -1;
            desc_out.reserved = [0; 32];
        } else {
            let zimms_desc = gcso_zimms_descriptor_t {
                mapped_address: 0,
                file_size_bytes: total_len as u64,
                dma_buffer_handle: 0,
                flags: 4,
                fd_handle: -1,
                reserved: [0; 32],
            };
            zimms_desc.write_unaligned_to(buf, GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET)?;
        }

        let hdr_size = size_of::<gcso_snapshot_header_t>();
        let checksum_crc32 = {
            #[cfg(target_endian = "little")]
            {
                if provenance_reports.is_empty() && is_aligned_to(buf, align_of::<u64>()) {
                    let fs_lo = total_len as u32;
                    let fs_hi = ((total_len as u64) >> 32) as u32;
                    let fs_c = (if fs_lo == 0 {
                        0
                    } else {
                        crc32_slice16_indep_lo1_u32(fs_lo)
                    }) ^ (if fs_hi == 0 {
                        0
                    } else {
                        crc32_slice16_indep_hi1_u32(fs_hi)
                    });
                    let mut crc = CRC32_ZIMMS_DESC_BASE_SEED
                        ^ (if fs_c == 0 { 0 } else { crc32_step_l3(fs_c) });
                    #[inline(always)]
                    fn stamped_crystal_crc(
                        crystals: &[CrystallizedAttractorRecord],
                        buf: &[u8],
                        off_crystals: usize,
                        idx: usize,
                    ) -> u32 {
                        let r = &crystals[idx];
                        let valid_hdr = (r.magic == GCSO_CRYSTAL_RECORD_MAGIC)
                            & (r.version == GCSO_CRYSTAL_RECORD_VERSION);
                        if valid_hdr && r.checksum_crc32 != 0 {
                            r.checksum_crc32
                        } else {
                            let c_off =
                                off_crystals + idx * size_of::<CrystallizedAttractorRecord>() + 52;
                            u32::from_le_bytes([
                                buf[c_off],
                                buf[c_off + 1],
                                buf[c_off + 2],
                                buf[c_off + 3],
                            ])
                        }
                    }
                    let num_crystals = crystals.len();
                    let mut idx = 0usize;
                    while idx + 4 <= num_crystals {
                        let s0 = crc32_crystal_stamp_contrib_128b(stamped_crystal_crc(
                            crystals,
                            buf,
                            off_crystals,
                            idx,
                        ));
                        let s1 = crc32_crystal_stamp_contrib_128b(stamped_crystal_crc(
                            crystals,
                            buf,
                            off_crystals,
                            idx + 1,
                        ));
                        let s2 = crc32_crystal_stamp_contrib_128b(stamped_crystal_crc(
                            crystals,
                            buf,
                            off_crystals,
                            idx + 2,
                        ));
                        let s3 = crc32_crystal_stamp_contrib_128b(stamped_crystal_crc(
                            crystals,
                            buf,
                            off_crystals,
                            idx + 3,
                        ));
                        let p01 = crc32_step_l8(s0) ^ s1;
                        let p23 = crc32_step_l8(s2) ^ s3;
                        crc = crc32_step_l32(crc) ^ crc32_step_l16(p01) ^ p23;
                        idx += 4;
                    }
                    while idx + 2 <= num_crystals {
                        let s0 = crc32_crystal_stamp_contrib_128b(stamped_crystal_crc(
                            crystals,
                            buf,
                            off_crystals,
                            idx,
                        ));
                        let s1 = crc32_crystal_stamp_contrib_128b(stamped_crystal_crc(
                            crystals,
                            buf,
                            off_crystals,
                            idx + 1,
                        ));
                        crc = crc32_step_l16(crc) ^ crc32_step_l8(s0) ^ s1;
                        idx += 2;
                    }
                    if idx < num_crystals {
                        let rec_crc = stamped_crystal_crc(crystals, buf, off_crystals, idx);
                        crc = crc32_fold_stamped_crystal_128b(crc, rec_crc);
                    }
                    if off_trails > after_crystals {
                        crc = crc32_step_l4(crc);
                    }
                    if total_len > off_trails {
                        crc = update_aligned64_blocks_crc32_raw(crc, &buf[off_trails..total_len]);
                    }
                    !crc
                } else {
                    Self::compute_container_crc32_with_stamped_crystals(
                        buf,
                        hdr_size,
                        off_crystals,
                        crystals.len(),
                        after_crystals,
                        total_len,
                    )
                }
            }
            #[cfg(not(target_endian = "little"))]
            {
                Self::compute_container_crc32_with_stamped_crystals(
                    buf,
                    hdr_size,
                    off_crystals,
                    crystals.len(),
                    after_crystals,
                    total_len,
                )
            }
        };

        if let Ok(hdr_out) = gcso_snapshot_header_t::from_bytes_mut_at(buf, 0) {
            hdr_out.magic = GCSO_SNAPSHOT_MAGIC_U32;
            hdr_out.version = GCSO_ABI_VERSION_HEX;
            hdr_out.total_size = total_len as u64;
            hdr_out.action_hub_offset = off_trails as u64;
            hdr_out.attractor_field_offset = off_crystals as u64;
            hdr_out.dpsr_state_offset = after_crystals as u64;
            hdr_out.srl_state_offset = off_reports as u64;
            hdr_out.edbc_state_offset = off_bloom as u64;
            hdr_out.checksum_crc32 = checksum_crc32;
            hdr_out.daes_slot_offset = off_daes as u32;
            hdr_out.timestamp_epoch_sec = timestamp_epoch_sec;
            hdr_out.reserved_padding = [0u8; 56];
            hdr_out.reserved_padding[0] = u8::from(bloom_opt.is_some());
            hdr_out.reserved_padding[1] = u8::from(daes_opt.is_some());
            let rep_cnt_le = (provenance_reports.len() as u32).to_le_bytes();
            hdr_out.reserved_padding[4] = rep_cnt_le[0];
            hdr_out.reserved_padding[5] = rep_cnt_le[1];
            hdr_out.reserved_padding[6] = rep_cnt_le[2];
            hdr_out.reserved_padding[7] = rep_cnt_le[3];
        } else {
            let mut header = init_snapshot_header(total_len as u64, timestamp_epoch_sec);
            header.attractor_field_offset = off_crystals as u64;
            header.dpsr_state_offset = after_crystals as u64;
            header.action_hub_offset = off_trails as u64;
            header.srl_state_offset = off_reports as u64;
            header.edbc_state_offset = off_bloom as u64;
            header.daes_slot_offset = off_daes as u32;
            header.checksum_crc32 = checksum_crc32;
            header.reserved_padding[0] = u8::from(bloom_opt.is_some());
            header.reserved_padding[1] = u8::from(daes_opt.is_some());
            let rep_cnt_le = (provenance_reports.len() as u32).to_le_bytes();
            header.reserved_padding[4] = rep_cnt_le[0];
            header.reserved_padding[5] = rep_cnt_le[1];
            header.reserved_padding[6] = rep_cnt_le[2];
            header.reserved_padding[7] = rep_cnt_le[3];
            header.write_unaligned_to(buf, 0)?;
        }
        Ok(total_len)
    }

    /// Maps a 128-byte aligned `.gcso` (ZIMMS) byte slice into a zero-copy [`GcsoFourSectionView`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`], or [`GCSO_ERROR_CONTAINER_CORRUPTED`] on validation failure.
    #[inline]
    pub fn from_zimms_bytes_zero_copy(bytes: &'a [u8]) -> GcsoResult<Self> {
        if bytes.len() < GCSO_FOUR_SECTION_HEADER_BYTES { return Err(GCSO_ERROR_BUFFER_TOO_SMALL); }
        if !is_aligned_to(bytes, 128) { return Err(GCSO_ERROR_MISALIGNED_POINTER); }

        let header = gcso_snapshot_header_t::from_bytes_at(bytes, 0)?;
        validate_snapshot_header(header, bytes.len())?;
        let zimms_descriptor = gcso_zimms_descriptor_t::from_bytes_at(bytes, GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET)?;
        if zimms_descriptor.file_size_bytes != 0 && zimms_descriptor.file_size_bytes != header.total_size
        { return Err(GCSO_ERROR_CONTAINER_CORRUPTED); }

        let total_len = header.total_size as usize;
        let hdr_size = size_of::<gcso_snapshot_header_t>();
        let off_crystals = header.attractor_field_offset as usize;
        let end_crystals = header.dpsr_state_offset as usize;
        let off_trails = header.action_hub_offset as usize;
        let off_bloom = header.edbc_state_offset as usize;
        let off_daes = header.daes_slot_offset as usize;

        if off_crystals < GCSO_FOUR_SECTION_HEADER_BYTES || off_crystals > end_crystals
            || end_crystals > off_trails || off_trails > off_bloom || off_bloom > off_daes || off_daes > total_len
        { return Err(GCSO_ERROR_CONTAINER_CORRUPTED); }

        let crystals_byte_len = end_crystals - off_crystals;
        if (crystals_byte_len % size_of::<CrystallizedAttractorRecord>()) != 0 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        let num_crystals = crystals_byte_len / size_of::<CrystallizedAttractorRecord>();
        let crystals = CrystallizedAttractorRecord::slice_from_zimms_bytes_with_crc_mode(
            &bytes[off_crystals..end_crystals], num_crystals, header.checksum_crc32 == 0)?;
        if header.checksum_crc32 != 0 {
            let computed = Self::compute_container_crc32_with_stamped_crystals(
                bytes,
                hdr_size,
                off_crystals,
                num_crystals,
                end_crystals,
                total_len,
            );
            if computed != header.checksum_crc32 { return Err(GCSO_ERROR_CONTAINER_CORRUPTED); }
        }

        let trails_byte_len = off_bloom - off_trails;
        if (trails_byte_len % size_of::<gcso_pointer_trail_t>()) != 0 { return Err(GCSO_ERROR_CONTAINER_CORRUPTED); }
        let num_trails = trails_byte_len / size_of::<gcso_pointer_trail_t>();
        let trails = gcso_pointer_trail_t::slice_from_bytes_at(bytes, off_trails, num_trails)?;

        let has_bloom = (header.reserved_padding[0] != 0)
            & ((off_daes - off_bloom) >= size_of::<ContrastivePhaseBloomFilter>());
        let bloom = if has_bloom {
            Some(ContrastivePhaseBloomFilter::from_bytes_at(bytes, off_bloom)?)
        } else {
            None
        };

        let has_daes = (header.reserved_padding[1] != 0) & ((total_len - off_daes) >= size_of::<gcso_daes_slot_t>());
        let daes = if has_daes {
            Some(gcso_daes_slot_t::from_bytes_at(bytes, off_daes)?)
        } else {
            None
        };

        let num_reports = u32::from_le_bytes([header.reserved_padding[4],
            header.reserved_padding[5], header.reserved_padding[6], header.reserved_padding[7]]) as usize;
        let off_reports = off_daes
            + if has_daes {
                size_of::<gcso_daes_slot_t>()
            } else {
                0
            };
        let provenance_reports = if num_reports > 0 {
            CausalProvenanceReport::slice_from_bytes_at(bytes, off_reports, num_reports)?
        } else { &[]
        };

        Ok(Self {
            header, zimms_descriptor, crystals, trails, bloom, daes, provenance_reports,
        })
    }

    #[inline(always)] pub(crate) fn validate_four_section_mut_offsets(zimms_bytes: &[u8]) -> GcsoResult<(
        gcso_snapshot_header_t, usize, usize, usize, usize, usize, usize, usize)> {
        if zimms_bytes.len() < GCSO_FOUR_SECTION_HEADER_BYTES { return Err(GCSO_ERROR_BUFFER_TOO_SMALL); }
        if !is_aligned_to(zimms_bytes, 128) { return Err(GCSO_ERROR_MISALIGNED_POINTER); }

        let header = *gcso_snapshot_header_t::from_bytes_at(zimms_bytes, 0)?;
        validate_snapshot_header(&header, zimms_bytes.len())?;
        let total_len = header.total_size as usize;
        let off_crystals = header.attractor_field_offset as usize;
        let end_crystals = header.dpsr_state_offset as usize;
        let off_trails = header.action_hub_offset as usize;
        let off_bloom = header.edbc_state_offset as usize;
        let off_daes = header.daes_slot_offset as usize;

        if off_crystals < GCSO_FOUR_SECTION_HEADER_BYTES || off_crystals > end_crystals
            || end_crystals > off_trails || off_trails > off_bloom || off_bloom > off_daes || off_daes > total_len
            || (off_crystals % align_of::<CrystallizedAttractorRecord>()) != 0
            || (off_trails % align_of::<gcso_pointer_trail_t>()) != 0
        { return Err(GCSO_ERROR_CONTAINER_CORRUPTED); }

        let crystals_bytes = end_crystals - off_crystals;
        let trails_bytes = off_bloom - off_trails;
        if (crystals_bytes % size_of::<CrystallizedAttractorRecord>()) != 0
            || (trails_bytes % size_of::<gcso_pointer_trail_t>()) != 0
        { return Err(GCSO_ERROR_CONTAINER_CORRUPTED); }
        let num_crystals = crystals_bytes / size_of::<CrystallizedAttractorRecord>();
        let num_trails = trails_bytes / size_of::<gcso_pointer_trail_t>();
        Ok((header, total_len, off_crystals, num_crystals, off_trails, num_trails, off_bloom, off_daes))
    }

    #[inline(always)] pub(crate) fn update_four_section_daes_in_place(zimms_bytes: &mut [u8],
        header: &gcso_snapshot_header_t, total_len: usize, off_daes: usize,
        metric_code: u8, shortcut_ptr: u64) -> GcsoResult<bool> {
        let has_daes = (header.reserved_padding[1] != 0) & ((total_len - off_daes) >= size_of::<gcso_daes_slot_t>());
        if has_daes {
            let daes_mut = gcso_daes_slot_t::from_bytes_mut_at(zimms_bytes, off_daes)?;
            let ring_head = (daes_mut.telemetry_ring_head as usize) & (DAES_MINI_LEDGER_LEN - 1);
            daes_mut.telemetry_mini_ledger[ring_head] = metric_code;
            daes_mut.telemetry_ring_head = daes_mut.telemetry_ring_head.wrapping_add(1);
            if shortcut_ptr != 0 {
                let sc_idx = ((shortcut_ptr.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
                daes_mut.fast_path_shortcuts[sc_idx & (DAES_SHORTCUT_SLOTS - 1)] = shortcut_ptr;
                daes_mut.fast_path_bypass_mask |= shortcut_ptr;
            }
        }
        Ok(has_daes)
    }

    #[inline(always)] #[must_use]
    fn compute_container_crc32_with_stamped_crystals(
        buf: &[u8],
        hdr_size: usize,
        off_crystals: usize,
        num_crystals: usize,
        after_crystals: usize,
        total_len: usize,
    ) -> u32 {
        #[cfg(target_endian = "little")]
        {
            if is_aligned_to(buf, align_of::<u64>())
                && ((off_crystals - hdr_size) & 63) == 0
                && ((total_len - after_crystals) & 63) == 0
            {
                let mut crc = if (off_crystals - hdr_size) == 64 {
                    // SAFETY: `buf` is 8-byte aligned and `hdr_size..off_crystals` is 64 bytes.
                    let w = unsafe { &*buf.as_ptr().add(hdr_size).cast::<[u64; 8]>() };
                    if ((w[0] | w[2]) == 0)
                        & (w[3] == 0xFFFF_FFFF_0000_0004u64)
                        & (((w[4] | w[5]) | (w[6] | w[7])) == 0)
                    {
                        let fs_lo = w[1] as u32;
                        let fs_hi = (w[1] >> 32) as u32;
                        let fs_c = (if fs_lo == 0 {
                            0
                        } else {
                            crc32_slice16_indep_lo1_u32(fs_lo)
                        }) ^ (if fs_hi == 0 {
                            0
                        } else {
                            crc32_slice16_indep_hi1_u32(fs_hi)
                        });
                        CRC32_ZIMMS_DESC_BASE_SEED
                            ^ (if fs_c == 0 { 0 } else { crc32_step_l3(fs_c) })
                    } else {
                        crc32_step_block64_words(0xFFFF_FFFFu32, w)
                    }
                } else {
                    update_aligned64_blocks_crc32_raw(0xFFFF_FFFFu32, &buf[hdr_size..off_crystals])
                };
                let mut idx = 0usize;
                while idx + 4 <= num_crystals {
                    let c_off0 = off_crystals + idx * size_of::<CrystallizedAttractorRecord>() + 52;
                    let c_off1 = c_off0 + size_of::<CrystallizedAttractorRecord>();
                    let c_off2 = c_off1 + size_of::<CrystallizedAttractorRecord>();
                    let c_off3 = c_off2 + size_of::<CrystallizedAttractorRecord>();
                    let rec_crc0 = u32::from_le_bytes([
                        buf[c_off0],
                        buf[c_off0 + 1],
                        buf[c_off0 + 2],
                        buf[c_off0 + 3],
                    ]);
                    let rec_crc1 = u32::from_le_bytes([
                        buf[c_off1],
                        buf[c_off1 + 1],
                        buf[c_off1 + 2],
                        buf[c_off1 + 3],
                    ]);
                    let rec_crc2 = u32::from_le_bytes([
                        buf[c_off2],
                        buf[c_off2 + 1],
                        buf[c_off2 + 2],
                        buf[c_off2 + 3],
                    ]);
                    let rec_crc3 = u32::from_le_bytes([
                        buf[c_off3],
                        buf[c_off3 + 1],
                        buf[c_off3 + 2],
                        buf[c_off3 + 3],
                    ]);
                    let s0 = crc32_crystal_stamp_contrib_128b(rec_crc0);
                    let s1 = crc32_crystal_stamp_contrib_128b(rec_crc1);
                    let s2 = crc32_crystal_stamp_contrib_128b(rec_crc2);
                    let s3 = crc32_crystal_stamp_contrib_128b(rec_crc3);
                    let p01 = crc32_step_l8(s0) ^ s1;
                    let p23 = crc32_step_l8(s2) ^ s3;
                    crc = crc32_step_l32(crc) ^ crc32_step_l16(p01) ^ p23;
                    idx += 4;
                }
                while idx + 2 <= num_crystals {
                    let c_off0 = off_crystals + idx * size_of::<CrystallizedAttractorRecord>() + 52;
                    let c_off1 = c_off0 + size_of::<CrystallizedAttractorRecord>();
                    let rec_crc0 = u32::from_le_bytes([
                        buf[c_off0],
                        buf[c_off0 + 1],
                        buf[c_off0 + 2],
                        buf[c_off0 + 3],
                    ]);
                    let rec_crc1 = u32::from_le_bytes([
                        buf[c_off1],
                        buf[c_off1 + 1],
                        buf[c_off1 + 2],
                        buf[c_off1 + 3],
                    ]);
                    let s0 = crc32_crystal_stamp_contrib_128b(rec_crc0);
                    let s1 = crc32_crystal_stamp_contrib_128b(rec_crc1);
                    crc = crc32_step_l16(crc) ^ crc32_step_l8(s0) ^ s1;
                    idx += 2;
                }
                if idx < num_crystals {
                    let c_off = off_crystals + idx * size_of::<CrystallizedAttractorRecord>() + 52;
                    let rec_crc = u32::from_le_bytes([
                        buf[c_off],
                        buf[c_off + 1],
                        buf[c_off + 2],
                        buf[c_off + 3],
                    ]);
                    crc = crc32_fold_stamped_crystal_128b(crc, rec_crc);
                }
                let mut rem_off = after_crystals;
                if rem_off + 64 <= total_len {
                    // SAFETY: `buf` is 8-byte aligned and `rem_off + 64 <= total_len`.
                    let w_pad = unsafe { &*buf.as_ptr().add(rem_off).cast::<[u64; 8]>() };
                    if (((w_pad[0] | w_pad[1]) | (w_pad[2] | w_pad[3]))
                        | ((w_pad[4] | w_pad[5]) | (w_pad[6] | w_pad[7])))
                        == 0
                    {
                        crc = crc32_step_l4(crc);
                        rem_off += 64;
                    }
                }
                crc = update_aligned64_blocks_crc32_raw(crc, &buf[rem_off..total_len]);
                return !crc;
            }
        }
        compute_aligned64_payload_crc32_ieee(&buf[hdr_size..total_len])
    }

    #[inline(always)] pub(crate) fn finalize_four_section_header_in_place(zimms_bytes: &mut [u8],
        header: &mut gcso_snapshot_header_t, total_len: usize) -> GcsoResult<()> {
        let hdr_size = size_of::<gcso_snapshot_header_t>();
        let off_crystals = header.attractor_field_offset as usize;
        let end_crystals = header.dpsr_state_offset as usize;
        let num_crystals = (end_crystals - off_crystals) / size_of::<CrystallizedAttractorRecord>();
        let crc = Self::compute_container_crc32_with_stamped_crystals(
            zimms_bytes,
            hdr_size,
            off_crystals,
            num_crystals,
            end_crystals,
            total_len,
        );
        header.checksum_crc32 = crc;
        let crc_off = offset_of!(gcso_snapshot_header_t, checksum_crc32);
        zimms_bytes[crc_off..crc_off + 4].copy_from_slice(&crc.to_le_bytes());
        Ok(())
    }

    /// **Zero-Copy In-Place `.gcso` 4-Section Container Evolution**:
    /// Mutates a 128-byte aligned `.gcso` (ZIMMS) container buffer (`zimms_bytes`) directly in-place:
    /// 1. Validates Section 0 (`gcso_snapshot_header_t`) and section offsets.
    /// 2. Synthesizes `signal` via [`ManifoldGatewaySignal::synthesize_gateway_frame_with_words`]
    ///    and executes Online Toroidal $k$-Means / DP-Means (`step_online_toroidal_kmeans_frame_with_words_64`)
    ///    directly on Section 1 (`&mut [CrystallizedAttractorRecord]`) and Section 2 (`trails.get_mut(trail_idx)`).
    /// 3. Optionally consolidates redundant centroids when `merge_coherence_thresh < 1.0` via
    ///    [`CrystallizedAttractorRecord::merge_and_prune_centroids_64`].
    /// 4. If Section 3a (`bloom`) is present and `frame` triggered repulsion, records the anti-phase
    ///    pattern in-place; if Section 3c (`provenance_reports`) has capacity, writes `frame.to_causal_report()`
    ///    into the circular diagnostic history slot in-place.
    /// 5. Recomputes and stamps `header.checksum_crc32` in-place.
    ///
    /// Returns `(GatewayFrame, assigned_cluster_idx, assigned_anchor_id, updated_gravity_mass, spawned_new_cluster, merged_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_MISALIGNED_POINTER`], or
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `zimms_bytes` fails layout or alignment checks.
    #[inline]
    pub fn evolve_four_section_kmeans_in_place<S: ManifoldGatewaySignal + ?Sized>(zimms_bytes: &mut [u8],
        signal: &S, split_coherence_thresh: f32, merge_coherence_thresh: f32, trail_idx: usize,
    ) -> GcsoResult<(GatewayFrame, usize, u32, f32, bool, u32)> {
        let (mut header, total_len, off_crystals, num_crystals, off_trails, num_trails, off_bloom, off_daes) =
            Self::validate_four_section_mut_offsets(zimms_bytes)?;

        let (frame, sample_w) = signal.synthesize_gateway_frame_with_words();
        let (crystals_mut, trails_mut) = {
            let (head_bytes, tail_bytes) = zimms_bytes.split_at_mut(off_trails);
            let crystals = CrystallizedAttractorRecord::slice_from_bytes_mut_at(
                head_bytes,
                off_crystals,
                num_crystals,
            )?;
            let trails = gcso_pointer_trail_t::slice_from_bytes_mut_at(tail_bytes, 0, num_trails)?;
            (crystals, trails)
        };

        let (c_idx, c_id, _coh, mass, spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(crystals_mut,
                &frame, &sample_w, split_coherence_thresh, trails_mut.get_mut(trail_idx));
        let merged = if merge_coherence_thresh < 1.0 && crystals_mut.len() >= 2 {
            CrystallizedAttractorRecord::merge_and_prune_centroids_64(crystals_mut, merge_coherence_thresh, 0.0,
                1.0).0
        } else {
            0
        };

        let rep = frame.to_causal_report();
        let has_bloom = (header.reserved_padding[0] != 0)
            & ((off_daes - off_bloom) >= size_of::<ContrastivePhaseBloomFilter>());
        if has_bloom && rep.is_cohomology_repulsion() {
            let bloom_mut = ContrastivePhaseBloomFilter::from_bytes_mut_at(zimms_bytes, off_bloom)?;
            bloom_mut.insert_repulsive_pattern_words_weighted(&sample_w, 2);
        }

        let metric_code = branchless_select_u8(rep.is_cohomology_repulsion(),
            DAES_METRIC_REPULSION_APPLIED, DAES_METRIC_PHASE_STEERED,
        ) | branchless_select_u8((frame.action_mode_flags & ACTION_MODE_PSPM_ROUTE) != 0,
            DAES_METRIC_PSPM_ROUTED, 0,
        ) | branchless_select_u8((frame.provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED) != 0,
            DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        let shortcut_ptr = branchless_select_u64(rep.is_cohomology_repulsion(), 0, frame.tagged_ptr);
        let has_daes = Self::update_four_section_daes_in_place(zimms_bytes, &header,
            total_len, off_daes, metric_code, shortcut_ptr)?;

        let num_reports = u32::from_le_bytes([header.reserved_padding[4],
            header.reserved_padding[5], header.reserved_padding[6], header.reserved_padding[7]]) as usize;
        if num_reports > 0 {
            let off_reports = off_daes
                + if has_daes {
                    size_of::<gcso_daes_slot_t>()
                } else {
                    0
                };
            let rep_end = off_reports.checked_add(num_reports * size_of::<CausalProvenanceReport>())
                .ok_or(GCSO_ERROR_CONTAINER_CORRUPTED)?;
            if rep_end <= total_len {
                let write_cursor = u32::from_le_bytes([header.reserved_padding[8],
                    header.reserved_padding[9], header.reserved_padding[10], header.reserved_padding[11]]) as usize;
                let slot_idx = write_cursor % num_reports;
                let slot_off = off_reports + slot_idx * size_of::<CausalProvenanceReport>();
                rep.write_unaligned_to(zimms_bytes, slot_off)?;
                let next_cursor = ((write_cursor.wrapping_add(1)) as u32).to_le_bytes();
                header.reserved_padding[8] = next_cursor[0];
                header.reserved_padding[9] = next_cursor[1];
                header.reserved_padding[10] = next_cursor[2];
                header.reserved_padding[11] = next_cursor[3];
                let cursor_off = offset_of!(gcso_snapshot_header_t, reserved_padding) + 8;
                zimms_bytes[cursor_off..cursor_off + 4].copy_from_slice(&next_cursor);
            }
        }

        Self::finalize_four_section_header_in_place(zimms_bytes, &mut header, total_len)?;
        Ok((frame, c_idx, c_id, mass, spawned, merged))
    }

    #[inline(always)] pub(crate) fn route_four_section_coarse_to_fine_with_words_in_place(zimms_bytes: &mut [u8],
        trail_idx: usize, pull_step_u7: u8) -> GcsoResult<(usize, u32, f32, f32, [u64; NUM_SWARM_MICRO_CELLS])> {
        let (mut header, total_len, off_crystals, num_crystals, off_trails, num_trails, _off_bloom, _off_daes) =
            Self::validate_four_section_mut_offsets(zimms_bytes)?;
        if trail_idx >= num_trails { return Err(GCSO_ERROR_INVALID_ARGUMENT); }

        let (crystals_ref, trails_mut) = {
            let (head_bytes, tail_bytes) = zimms_bytes.split_at_mut(off_trails);
            let crystals = CrystallizedAttractorRecord::slice_from_bytes_at(
                head_bytes,
                off_crystals,
                num_crystals,
            )?;
            let trails = gcso_pointer_trail_t::slice_from_bytes_mut_at(tail_bytes, 0, num_trails)?;
            (crystals, trails)
        };

        let (best_idx, anchor_id, best_coh, best_grav, bridged_w, _win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(crystals_ref,
                &mut trails_mut[trail_idx], pull_step_u7);
        Self::finalize_four_section_header_in_place(zimms_bytes, &mut header, total_len)?;
        Ok((best_idx, anchor_id, best_coh, best_grav, bridged_w))
    }

    /// **Zero-Copy In-Place Coarse-to-Fine Trail Routing on a 4-Section `.gcso` Container**:
    /// Routes `trails[trail_idx]` inside Section 2 against the crystallized macro centroids in
    /// Section 1 (`crystals`) via [`CrystallizedAttractorRecord::route_coarse_to_fine_trail_64`]
    /// and updates `header.checksum_crc32` in-place.
    ///
    /// Returns `(selected_cluster_idx, selected_anchor_id, centroid_coherence, gravity_pull)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`GCSO_ERROR_INVALID_ARGUMENT`], or [`GCSO_ERROR_CONTAINER_CORRUPTED`] on validation failure.
    #[inline]
    pub fn route_four_section_coarse_to_fine_in_place(zimms_bytes: &mut [u8],
        trail_idx: usize, pull_step_u7: u8) -> GcsoResult<(usize, u32, f32, f32)> {
        let (best_idx, anchor_id, best_coh, best_grav, _bridged_w) =
            Self::route_four_section_coarse_to_fine_with_words_in_place(zimms_bytes, trail_idx, pull_step_u7)?;
        Ok((best_idx, anchor_id, best_coh, best_grav))
    }

    /// **Zero-Copy In-Place Top-2 Liminal Bifurcation Routing on a 4-Section `.gcso` Container**:
    /// Routes `trails[trail_idx]` inside Section 2 against the Top-2 nearest macro centroids in
    /// Section 1 (`crystals`) via [`CrystallizedAttractorRecord::route_top2_liminal_centroids_64`],
    /// updates Section 3b (`daes`) telemetry if present, and stamps `header.checksum_crc32` in-place.
    ///
    /// Returns `(primary_idx, secondary_idx, primary_anchor_id, secondary_anchor_id, blended_coherence, total_gravity, is_liminal_bifurcation)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`GCSO_ERROR_INVALID_ARGUMENT`], or [`GCSO_ERROR_CONTAINER_CORRUPTED`] on validation failure.
    #[inline]
    pub fn route_four_section_top2_liminal_in_place(zimms_bytes: &mut [u8],
        trail_idx: usize, pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32) -> GcsoResult<(usize, usize, u32, u32, f32, f32, bool)> {
        let (mut header, total_len, off_crystals, num_crystals, off_trails, num_trails, _off_bloom, off_daes) =
            Self::validate_four_section_mut_offsets(zimms_bytes)?;
        if trail_idx >= num_trails { return Err(GCSO_ERROR_INVALID_ARGUMENT); }

        let (crystals_ref, trails_mut) = {
            let (head_bytes, tail_bytes) = zimms_bytes.split_at_mut(off_trails);
            let crystals = CrystallizedAttractorRecord::slice_from_bytes_at(
                head_bytes,
                off_crystals,
                num_crystals,
            )?;
            let trails = gcso_pointer_trail_t::slice_from_bytes_mut_at(tail_bytes, 0, num_trails)?;
            (crystals, trails)
        };

        let (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_64(crystals_ref,
                &mut trails_mut[trail_idx], pull_step_u7, liminal_gravity_ratio_thresh);
        let updated_ptr = trails_mut[trail_idx].current_ptr;
        let metric_code = branchless_select_u8((aid1 != 0) | is_liminal, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        let _ = Self::update_four_section_daes_in_place(zimms_bytes, &header,
            total_len, off_daes, metric_code, updated_ptr)?;
        Self::finalize_four_section_header_in_place(zimms_bytes, &mut header, total_len)?;
        Ok((idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal))
    }
}

// ===================================================================
// 9. Blanket Extension Traits on `PhaseSteering + PspmRouter` & `ActionHub`
// ===================================================================

#[inline(always)]
fn accumulate_trail_and_steer_gateway_words_core(
    frame: &GatewayFrame, frm_w: &[u64; NUM_SWARM_MICRO_CELLS],
    trail_inout: &mut gcso_pointer_trail_t, unresolved_damping_u7: u8,
    head_dim: usize, num_heads: usize,
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
        branchless_max_f32(trail_inout.attractor_pull_force, frame.effective_energy), MAX_ATTRACTOR_PULL_FORCE);

    let layer_skipped = (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0;
    let mut steered = false;
    let mut pspm_routed = false;

    if layer_skipped {
        trail_inout.flags |= TRAIL_FLAG_INERTIAL_SKIP_ACTIVE;
    } else if query_tensor.is_some() | key_tensor.is_some() {
        steered = true;
        let swar_words = if frame.qdps_min_step_q7 <= 1 || frm_nonzero {
            next_w
        } else if frame.qdps_min_step_q7 > 127 {
            [0u64; NUM_SWARM_MICRO_CELLS]
        } else {
            filter_qdps_words_in_registers_64(&next_w, frame.qdps_min_step_q7)
        };
        let clean_damping = branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);

        let has_nonzero_words = if frm_nonzero {
            true
        } else {
            swar_words_nonzero_64(&swar_words)
        };
        if (frame.action_mode_flags & ACTION_MODE_PSPM_ROUTE) != 0 && frame.active_swarm_cell_mask == 0xFF
        {
            let is_def_clamp = (frame.ripa_clamp_rad.to_bits() == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
                & (num_heads > 0)
                & (num_heads <= MAX_ATTENTION_HEADS);
            match (query_tensor, key_tensor) {
                (Some(q), Some(k)) => {
                    if has_nonzero_words {
                        if is_def_clamp {
                            let rot_dim = ((head_dim >> 2) & !1).max(2).min(head_dim);
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
                            let rot_dim = ((head_dim >> 2) & !1).max(2).min(head_dim);
                            rotate_fused_self_ripa_and_pspm_default_words(
                                q,
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
                    steer_from_swar_words_core_safe(&swar_words, frame.active_swarm_cell_mask, clean_damping,
                        head_dim, num_heads, false, frame.ripa_clamp_rad, q, k_opt);
                }
                (None, Some(k)) => {
                    steer_from_swar_words_core_safe(&swar_words, frame.active_swarm_cell_mask, clean_damping,
                        head_dim, num_heads, false, frame.ripa_clamp_rad, k, None);
                }
                (None, None) => {}
            }
        }
    }

    if let Some(logits) = logits_opt {
        if !logits.is_empty() { scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp); }
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
    fn steer_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(&self,
        signal: &S, trail_inout: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8, query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>, logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        let report = frame.to_causal_report();
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS || frame.qdps_min_step_rad < 0.0
            || !branchless_is_finite_f32(frame.qdps_min_step_rad) || frame.ripa_clamp_rad <= 0.0
            || !branchless_is_finite_f32(frame.ripa_clamp_rad)
        { return Err(GCSO_ERROR_INVALID_ARGUMENT); }
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

        let _ = accumulate_trail_and_steer_gateway_words_core(&frame, &frm_w, trail_inout,
            unresolved_damping_u7, head_dim, num_heads, Some(query_tensor), key_tensor, logits_opt)?;
        Ok((frame, report))
    }

    /// Folds a zero-copy slice of `.gcso` [`CrystallizedAttractorRecord`]s (`records`) onto
    /// `trail_inout` weighted by **Phase-Proximity Gravity** directly in 8 `u64` SWAR registers.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if head dimensions, `min_step_rad`, or tensor slices
    /// violate layout or `GCSO_RESTRICT` disjointness invariants.
    #[inline(always)]
    fn steer_crystallized_ensemble_zero_copy_safe(&self,
        records: &[CrystallizedAttractorRecord], trail_inout: &mut gcso_pointer_trail_t,
        min_step_rad: f32, query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>) -> GcsoResult<(f32, u32, f32)> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS || min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad)
        { return Err(GCSO_ERROR_INVALID_ARGUMENT); }
        validate_optional_qk_tensors(Some(&*query_tensor), key_tensor.as_deref(), expected_len)?;

        let init_pull = branchless_flush_subnormal_f32(branchless_relu_f32(trail_inout.attractor_pull_force));
        let (folded_w, tot_pull, dom_id, dom_clamp) =
            CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64(records,
                &mut trail_inout.accumulated_phase_delta, init_pull);

        if dom_id != 0 {
            trail_inout.target_anchor_id = dom_id;
            trail_inout.cluster_id = dom_id;
        }
        trail_inout.attractor_pull_force = branchless_min_f32(tot_pull, MAX_ATTRACTOR_PULL_FORCE);
        trail_inout.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | TRAIL_FLAG_SYNERGY_RESONANCE;

        let eff_ripa = branchless_select_f32(branchless_is_finite_f32(dom_clamp) & (dom_clamp > 0.0),
            dom_clamp, DEFAULT_RIPA_CLAMP_MAX_RAD);
        let min_step_q7 = rad_to_qdps_min_step_q7_u8(min_step_rad);
        let qdps_words = if min_step_q7 <= 1 {
            folded_w
        } else if min_step_q7 > 127 {
            [0u64; NUM_SWARM_MICRO_CELLS]
        } else {
            filter_qdps_words_in_registers_64(&folded_w, min_step_q7)
        };
        let is_default_ripa = branchless_abs_f32(eff_ripa - DEFAULT_RIPA_CLAMP_MAX_RAD) <= 1.0e-5;
        let swar_words = if is_default_ripa {
            qdps_words
        } else {
            let ratio_u8 = (branchless_clamp_f32(eff_ripa * (1.0 / DEFAULT_RIPA_CLAMP_MAX_RAD), 0.10, 1.0)
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
            steer_from_swar_words_core_safe(&swar_words, 0xFF, 128, head_dim, num_heads, false,
                DEFAULT_RIPA_CLAMP_MAX_RAD, query_tensor, key_tensor);
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
    fn ingest_gateway_and_steer_with_words_core<S: ManifoldGatewaySignal + ?Sized>(&mut self,
        signal: &S, daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t, mut query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>, logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        srl_activation_and_desc: Option<(&AlignedSlice32<'_, f32>, &gcso_srl_descriptor_t)>) -> GcsoResult<(
        GatewayFrame, CausalProvenanceReport, [u64; NUM_SWARM_MICRO_CELLS])> {
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        let report = frame.to_causal_report();
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(frame.qdps_min_step_rad,
            frame.ripa_clamp_rad, query_tensor.as_deref(), key_tensor.as_deref())?;

        let (bypass_hit, effective_ptr) =
            self.resolve_or_pass_tagged_ptr(daes_slot, gcso_tagged_ptr_t(frame.tagged_ptr));
        self.step_pointer(effective_ptr.raw(), trail_out)?;

        if frame.target_anchor_id != 0 {
            trail_out.target_anchor_id = frame.target_anchor_id;
            trail_out.cluster_id = frame.target_anchor_id;
        } else {
            trail_out.cluster_id = frame.dominant_slot as u32;
        }

        let is_repulsed = report.is_cohomology_repulsion()
            | ((frame.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0);
        let mut metrics = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(is_repulsed, DAES_METRIC_REPULSION_APPLIED, 0) | branchless_select_u8(
                (frame.provenance_bits & (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA))
                    == (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA), DAES_METRIC_SLOT_SYNERGY_ACTIVE,
                0);

        let (layer_skipped, steered, pspm_routed) = accumulate_trail_and_steer_gateway_words_core(
            &frame, &frm_w, trail_out, 128, head_dim, num_heads,
            query_tensor.as_deref_mut(), key_tensor, logits_opt)?;

        if layer_skipped {
            metrics |= DAES_METRIC_LAYER_SKIP_COMPENSATED;
        } else {
            metrics |= branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
                | branchless_select_u8(pspm_routed, DAES_METRIC_PSPM_ROUTED, 0);
            if let (Some(q), Some((x_in, desc))) = (query_tensor, srl_activation_and_desc)
            {
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
    fn ingest_gateway_and_steer_safe<S: ManifoldGatewaySignal + ?Sized>(&mut self,
        signal: &S, daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t, query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>, logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        srl_activation_and_desc: Option<(&AlignedSlice32<'_, f32>, &gcso_srl_descriptor_t)>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
        let (frame, report, _frm_w) = self.ingest_gateway_and_steer_with_words_core(signal, daes_slot, trail_out,
            query_tensor, key_tensor, logits_opt, srl_activation_and_desc)?;
        Ok((frame, report))
    }

    /// Ingests `signal`, resolves the 8-Cell SWAR Micro-Swarm on `trail_out`, and absorbs the
    /// resulting [`GatewayFrame`] into `crystal_record`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`Self::ingest_gateway_and_steer_safe`].
    #[inline(always)]
    fn ingest_gateway_swarm_and_crystallize_safe<S: ManifoldGatewaySignal + ?Sized>(&mut self,
        signal: &S, daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t, crystal_record: &mut CrystallizedAttractorRecord,
        step_gain_u7: u8, tolerance_q7: u8, quorum_threshold: u8, entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32, bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8, repulsion_scale_u7: u8,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>, key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>) -> GcsoResult<(GatewayFrame, CausalProvenanceReport,
        SwarmResolutionReport, f32)> {
        let (frame, report, init_frm_w) = self.ingest_gateway_and_steer_with_words_core(signal, daes_slot, trail_out,
            query_tensor, key_tensor, logits_opt, None)?;
        let cur_trail_w = if swar_words_nonzero_64(&init_frm_w) {
            init_frm_w
        } else {
            *trail_phase_words_ref(trail_out)
        };
        let (swarm_report, frm_w) = trail_out.try_resolve_action_in_swarm_with_cur_words(
            gcso_tagged_ptr_t(frame.tagged_ptr), &cur_trail_w, &init_frm_w, false,
            frame.active_swarm_cell_mask, step_gain_u7, frame.qdps_min_step_q7, tolerance_q7,
            quorum_threshold, entropy_h_tilde, low_entropy_skip_thresh, bloom_filter,
            repulsion_threshold, repulsion_scale_u7);
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
    fn ingest_gateway_kmeans_and_steer_safe<S: ManifoldGatewaySignal + ?Sized>(&mut self,
        signal: &S, centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32, daes_slot: &mut gcso_daes_slot_t,
        trail_out: &mut gcso_pointer_trail_t, query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>, logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport, usize, u32, f32, bool)> {
        let (frame, report, frm_w) = self.ingest_gateway_and_steer_with_words_core(signal, daes_slot, trail_out,
            query_tensor, key_tensor, logits_opt, None)?;
        let (cluster_idx, anchor_id, _coh, mass, spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(centroids, &frame, &frm_w,
                split_coherence_thresh, Some(trail_out));
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
    fn ingest_gateway_kmeans_consolidate_and_steer_safe<S: ManifoldGatewaySignal + ?Sized>(&mut self,
        signal: &S, centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32, merge_coherence_thresh: f32,
        min_gravity_mass: f32, evaporative_decay_factor: f32,
        daes_slot: &mut gcso_daes_slot_t, trail_out: &mut gcso_pointer_trail_t,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>, key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>) -> GcsoResult<(GatewayFrame, CausalProvenanceReport,
        usize, u32, f32, bool, u32, u32)> {
        let (frame, report, cluster_idx, anchor_id, mass, spawned) = self.ingest_gateway_kmeans_and_steer_safe(
                signal, centroids, split_coherence_thresh, daes_slot, trail_out, query_tensor, key_tensor, logits_opt,
            )?;
        let (merged, pruned) = CrystallizedAttractorRecord::merge_and_prune_centroids_64(centroids,
            merge_coherence_thresh, min_gravity_mass, evaporative_decay_factor);
        Ok((frame, report, cluster_idx, anchor_id, mass, spawned, merged, pruned))
    }
}

impl<T: ActionHub + ?Sized> ActionHubGatewayExt for T {}

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
    assert!(CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0).verify_integrity());
    assert!(CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0).verify_integrity());
};

#[cfg(test)] mod tests {
    use super::*;
    use crate::traits::aligned::{AlignedBuffer128, AlignedBuffer64};

    #[test]
    fn test_two_origin_supertrait_synthesis_and_causal_provenance() {
        let sys = SystemConstraint::new((), 0.20, 0.10, 0xFF);
        let fact = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 3.0, 0x0012_3400, 77);
        let persona_aligned = SubjectivePersona::new([30i8; MAX_ATTENTION_HEADS], 2.0, 0x0020_0000, 12);

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

        let persona_opposed = SubjectivePersona::new([-96i8; MAX_ATTENTION_HEADS], 2.5, 0x0020_0000, 12);
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
        let mut centroids = [CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0)];
        assert_eq!(centroids[0].causal_provenance_bits(), 0);
        assert_eq!(centroids[0].synthesize_gateway_frame().action_mode_flags, ACTION_PRESET_PASSIVE_SIDECAR);

        let mut trail = gcso_pointer_trail_t::default();

        let fact_a = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 2.5, 0x0010_0000, 101);
        let (f1, idx1, id1, _coh1, mass1, spawned1) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_signal_64(&mut centroids,
                &fact_a, DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, Some(&mut trail));
        assert!(spawned1 && idx1 == 0 && id1 == 101 && trail.cluster_id == 101 && mass1 > 0.0);

        let fact_a2 = ObjectiveFact::new([30i8; MAX_ATTENTION_HEADS], 3.0, 0x0010_0000, 101);
        let (f2, idx2, id2, coh2, mass2, spawned2) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_signal_64(&mut centroids,
                &fact_a2, DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, Some(&mut trail));
        assert!(!spawned2 && idx2 == 0 && id2 == 101 && coh2 > 0.95 && mass2 > mass1);
        assert_eq!(centroids[0].sample_count, 2);

        let persona_b = SubjectivePersona::new([-80i8; MAX_ATTENTION_HEADS], 2.8, 0x0024_0000, 202);
        let (_f3, idx3, id3, _coh3, mass3, spawned3) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_signal_64(&mut centroids,
                &persona_b, DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, Some(&mut trail));
        assert!(spawned3 && idx3 == 1 && id3 == 202 && trail.cluster_id == 202 && mass3 > 0.0);

        let (act_cnt, tot_smp, mean_r, mean_k, _max_coh) =
            CrystallizedAttractorRecord::diagnose_macro_distribution_64(&centroids);
        assert_eq!(act_cnt, 2);
        assert_eq!(tot_smp, 3);
        assert!(mean_r > 0.85 && mean_k > 2.0);

        let closed_loop_frame = centroids.synthesize_gateway_frame();
        assert!(closed_loop_frame.effective_energy > 2.0);
        assert_ne!(closed_loop_frame.provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, 0);

        let batch_frames = [f1, f2];
        let (_b_idx, b_id, b_coh, b_spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_batch_frames_64(&mut centroids,
                &batch_frames, DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, Some(&mut trail));
        assert_eq!(b_id, 101);
        assert_eq!(b_spawned, 0);
        assert!(b_coh > 0.95 && centroids[0].verify_integrity());

        // Populate slot 2 with a near-duplicate of slot 0 and verify online consolidation
        centroids[2] = CrystallizedAttractorRecord::new_empty(303, ATTRACTOR_SLOT_KNOWLEDGE, 0);
        let _ = centroids[2].absorb_sample_point_64(&[29i8; MAX_ATTENTION_HEADS], 1.2);
        let (merged_cnt, pruned_cnt) = CrystallizedAttractorRecord::merge_and_prune_centroids_64(&mut centroids,
            DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE, 0.10, 0.98);
        assert_eq!(merged_cnt, 1);
        assert_eq!(pruned_cnt, 0);
        assert_eq!(centroids[2].sample_count, 0);
        assert!(centroids[0].verify_integrity() && centroids[2].verify_integrity());

        let mut query_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [-76i8; MAX_ATTENTION_HEADS], stigmergic_density: 1.0,
            attractor_pull_force: 1.0,..Default::default()
        };
        let (routed_idx, routed_id, routed_coh, routed_grav) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_64(&centroids, &mut query_trail, 64);
        assert_eq!(routed_idx, 1);
        assert_eq!(routed_id, 202);
        assert_eq!(query_trail.cluster_id, 202);
        assert_eq!(gcso_tagged_ptr_t(query_trail.current_ptr).unpack_addr(), 0x0024_0000);
        assert!(routed_coh > 0.95 && routed_grav > 1.0);

        let mut zimms_buf = AlignedBuffer64::<u8, 384>::new([0u8; 384]);
        let written = CrystallizedAttractorRecord::write_records_to_zimms_bytes(&centroids, &mut zimms_buf.data)
        .unwrap();
        assert_eq!(written, 384);
        let restored = CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy(&zimms_buf.data, 3).unwrap();
        assert_eq!(restored[0].anchor_id, 101);
        assert_eq!(restored[1].anchor_id, 202);

        let bloom = ContrastivePhaseBloomFilter::new();
        let daes = gcso_daes_slot_t::default();
        let reports_arr = [f1.to_causal_report(), f2.to_causal_report()];
        let trails_arr = [query_trail];
        let mut container_buf = AlignedBuffer128::<u8, 1024>::new([0u8; 1024]);
        let no_rep_len = GcsoFourSectionView::write_four_section_container_to_bytes(&centroids[..2],
            &trails_arr, Some(&bloom), Some(&daes), 1_774_900_000, &mut container_buf.data).unwrap();
        let no_rep_view = GcsoFourSectionView::from_zimms_bytes_zero_copy(&container_buf.data[..no_rep_len]).unwrap();
        assert_eq!(no_rep_view.crystals.len(), 2);

        let total_len = GcsoFourSectionView::write_four_section_with_reports_to_bytes(&centroids,
            &trails_arr, Some(&bloom), Some(&daes), &reports_arr, 1_774_900_000, &mut container_buf.data).unwrap();

        let (_ev_frame, ev_idx, ev_id, ev_mass, _ev_spawned, ev_merged) =
            GcsoFourSectionView::evolve_four_section_kmeans_in_place(&mut container_buf.data[..total_len],
                &fact_a2, DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE, DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE, 0,
            ).unwrap();
        assert_eq!(ev_idx, 0);
        assert_eq!(ev_id, 101);
        assert_eq!(ev_merged, 0);
        assert!(ev_mass > 1.0);

        let (r_idx, r_id, _r_coh, r_grav) = GcsoFourSectionView::route_four_section_coarse_to_fine_in_place(
                &mut container_buf.data[..total_len], 0, 32).unwrap();
        assert_eq!(r_idx, 1);
        assert_eq!(r_id, 202);
        assert!(r_grav > 1.0);

        let view = GcsoFourSectionView::from_zimms_bytes_zero_copy(&container_buf.data[..total_len]).unwrap();
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
        let mut lim_centroids = [CrystallizedAttractorRecord::new_empty(101, ATTRACTOR_SLOT_KNOWLEDGE, 0x0010_0000),
            CrystallizedAttractorRecord::new_empty(202, ATTRACTOR_SLOT_PERSONA, 0x0024_0000)];
        let _ = lim_centroids[0].absorb_sample_point_64(&[16i8; MAX_ATTENTION_HEADS], 2.5);
        let _ = lim_centroids[1].absorb_sample_point_64(&[48i8; MAX_ATTENTION_HEADS], 2.5);
        let mut lim_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS], stigmergic_density: 1.0,
            attractor_pull_force: 1.0,..Default::default()
        };
        let (_k1, _k2, aid1, aid2, lim_coh, lim_grav, is_liminal) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_64(&lim_centroids, &mut lim_trail, 64,
                DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO);
        assert!(is_liminal && aid1 != 0 && aid2 != 0 && aid1 != aid2);
        assert!(lim_coh > 0.70 && lim_grav > 1.0);
        assert_ne!(lim_trail.flags & TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        // Verify Continuous Geodesic Affinity Prior on unstamped candidate trails (cluster_id == 0)
        let cand_aligned = gcso_pointer_trail_t {
            accumulated_phase_delta: [18i8; MAX_ATTENTION_HEADS], stigmergic_density: 1.0,
            cluster_id: 0,..Default::default()
        };
        let cand_orthogonal = gcso_pointer_trail_t {
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS], stigmergic_density: 1.0,
            cluster_id: 0,..Default::default()
        };
        let mut src_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [16i8; MAX_ATTENTION_HEADS],..Default::default()
        };
        let (win_c, sel_tid, best_sc, best_c, found_ok) =
            CrystallizedAttractorRecord::route_coarse_to_fine_candidates_64(&lim_centroids,
                &mut src_trail, &[cand_orthogonal, cand_aligned], 128, 1.0, 0.25, 0.10, 0.50, 0.35);
        assert!(found_ok && win_c == 0 && sel_tid == 1 && best_sc > 0.5 && best_c > 0.95);
    }
}