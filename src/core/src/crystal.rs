// name: src/core/src/crystal.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `.gcso` Crystallized Attractor Record (`CrystallizedAttractorRecord`, 128B `Pod` `"CSTL"`),
//! Online Toroidal $k$-Means / DP-Means Macro Clustering, Coarse-to-Fine & Top-2 Liminal
//! Bifurcation Routing, and `[CrystallizedAttractorRecord]` `ManifoldGatewaySignal` synthesis.

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

use crate::abi::{
    gcso_pointer_trail_t, gcso_q7_t, gcso_tagged_ptr_t, GCSO_ERROR_BUFFER_TOO_SMALL,
    GCSO_ERROR_CONTAINER_CORRUPTED, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER,
};
use crate::traits::aligned::is_aligned_to;
use crate::traits::branchless::{
    branchless_q7_coherence_64, branchless_toroidal_centroid_step_words_with_coh_and_uni_64,
    branchless_toroidal_centroid_step_words_with_coh_uni_and_changed_64,
    coherence_above_floor_from_word_arrays_64, coherence_from_uniform_words_x8,
    coherence_from_word_arrays_with_both_uniform_flags_64, swar_slerp_words_with_uniform_flag_64,
    swar_words_equal_64,
};
use crate::traits::gateway::{
    GatewayFrame, ManifoldGatewaySignal, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, DEFAULT_FACT_RIPA_CLAMP_RAD, DEFAULT_PERSONA_RIPA_CLAMP_RAD,
    GCSO_CRYSTAL_RECORD_MAGIC, GCSO_CRYSTAL_RECORD_VERSION, MAX_REGISTER_CACHED_CENTROIDS,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_NOMINAL,
};
use crate::traits::primitives::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_lerp_f32,
    branchless_maturity_scale_u32, branchless_max_f32, branchless_min_f32,
    branchless_phase_proximity_gravity_f32, branchless_popcount_rsqrt_u32,
    branchless_q7_default_gravity_from_diff_u8, branchless_relu_f32, branchless_rsqrt_f32,
    branchless_select_f32, branchless_select_u32, branchless_select_usize,
    crc32_crystal_be24_final_u32, crc32_crystal_gm36_final_u32, crc32_crystal_zoff16_final_u32,
    crc32_slice16_dep_lo0, crc32_slice16_indep_hi0_u32, crc32_slice16_indep_hi1_u32,
    crc32_slice16_indep_lo1_u32, crc32_slice16_step_u64x2, crc32_step_block64_words, crc32_step_l3,
    crc32_step_l4, crc32_step_l5, crc32_step_l6, crc32_update_crystal_absorption_delta,
    crc32_update_crystal_energy_and_mass_delta, crc32_update_crystal_stationary_fields_delta,
    fast_inv_len_f32, fast_inv_pos_normal_f32, fast_ln_1p_bounded_f32, is_word_array_uniform_64,
    load_q7_words_64, pack_action_slot_tag, store_q7_words_64,
    swar_q7_mean_i8_from_words_with_uniform_flag_64, trail_phase_words_mut, trail_phase_words_ref,
    GcsoResult, Pod, ACTION_MODE_PHASE_REPULSE, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_PASSIVE_SIDECAR, ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_PERSONA,
    CRC32_CRYSTAL_ACTION_BYTE57_REF, CRC32_CRYSTAL_AID_BYTE0_REF, CRC32_CRYSTAL_AID_BYTE1_REF,
    CRC32_CRYSTAL_PROV_BYTE0_REF, CRC32_CRYSTAL_PROV_BYTE1_REF, CRC32_CRYSTAL_SC_BYTE0_REF,
    CRC32_CRYSTAL_SLOT_BYTE56_LUT_8, CRC32_CRYSTAL_ZOFF_BYTE1_REF, CRC32_CRYSTAL_ZOFF_BYTE2_REF,
    CRC32_CRYSTAL_ZOFF_BYTE3_REF, CRC32_IEEE_SLICE16_REF, CRYSTAL_CRC32_DEFAULT_W5_FINAL_CONTRIB,
    CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB, CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB,
    CRYSTAL_CRC32_RBAR_ONE_FINAL_CONTRIB, CRYSTAL_CRC32_W0_FINAL_CONTRIB,
    DEFAULT_PHASE_GRAVITY_EPS, DEFAULT_RIPA_CLAMP_MAX_RAD, MAX_ATTENTION_HEADS, Q7_FULL_COS_REF,
    SWAR_SPLAT_BYTE_64, VON_MISES_KAPPA_AT_MAX_R_BITS,
};
use crate::traits::telemetry::{
    MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY, NUM_SWARM_MICRO_CELLS,
    TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
    TRAIL_FLAG_SYNERGY_RESONANCE,
};

// ===================================================================
// 8. `.gcso` Crystallized Attractor Record & Online Toroidal $k$-Means / DP-Means
// ===================================================================

/// 128-byte (`#[repr(C, align(64))]`) [`Pod`] record representing a bottom-up crystallized
/// macro attractor centroid on $SO(2)^{64}$ persisted inside `.gcso` (ZIMMS) containers.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrystallizedAttractorRecord {
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
    fn default() -> Self {
        Self::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0)
    }
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
        unsafe {
            &*(self
                .phase_profile
                .as_ptr()
                .cast::<[u64; NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Mutably borrows `self.phase_profile` (`offset 64` of `#[repr(C, align(64))]`
    /// [`CrystallizedAttractorRecord`], guaranteed 64-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_mut(&mut self) -> &mut [u64; NUM_SWARM_MICRO_CELLS] {
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: `CrystallizedAttractorRecord` is `#[repr(C, align(64))]` and `phase_profile`
        // is at exact byte offset 64 (`64 % 8 == 0`), with identical 64-byte layout to `[u64; 8]`.
        unsafe {
            &mut *(self
                .phase_profile
                .as_mut_ptr()
                .cast::<[u64; NUM_SWARM_MICRO_CELLS]>())
        }
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
        self.init_spawned_slot_header_in_place(0, dominant_slot, 0);
        *self.phase_words_mut() = [0u64; NUM_SWARM_MICRO_CELLS];
    }

    #[inline(always)]
    #[must_use]
    const fn new_unstamped(anchor_id: u32, dominant_slot: u8, zimms_section_offset48: u64) -> Self {
        Self {
            magic: GCSO_CRYSTAL_RECORD_MAGIC,
            version: GCSO_CRYSTAL_RECORD_VERSION,
            anchor_id,
            sample_count: 0,
            zimms_section_offset48: zimms_section_offset48 & gcso_tagged_ptr_t::ADDR_MASK,
            base_energy: 0.0,
            resultant_r_bar: 0.0,
            von_mises_kappa: 0.0,
            phase_gravity_mass: 0.0,
            ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            effective_inv_temp: 1.0,
            provenance_bits: 0,
            checksum_crc32: 0,
            dominant_slot: dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
            active_swarm_mask: 0xFF,
            dominant_cause_code: ROOT_CAUSE_NOMINAL,
            reserved_align64: 0,
            phase_profile: [0i8; MAX_ATTENTION_HEADS],
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

    #[inline]
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
        const DEFAULT_W7_MASK_CAUSE_CRC_C3: u32 = CRC32_IEEE_SLICE16_REF[5][0xFF]
            ^ CRC32_IEEE_SLICE16_REF[4][ROOT_CAUSE_NOMINAL as usize];
        const DEFAULT_W7_MASK_CAUSE_FINAL_CONTRIB: u32 =
            crc32_step_l4(DEFAULT_W7_MASK_CAUSE_CRC_C3);
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
        Self::new_unstamped(0, 0, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 1, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 2, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 3, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 4, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 5, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 6, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
        Self::new_unstamped(0, 7, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_SWARM_MICRO_CELLS]),
    ];

    /// Creates an empty, CRC32-stamped [`CrystallizedAttractorRecord`] ready to absorb sample points.
    #[inline]
    #[must_use]
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
            } else if (z_lo & 0xFF) == 0 {
                CRC32_CRYSTAL_ZOFF_BYTE1_REF[((z_lo >> 8) & 0xFF) as usize]
                    ^ CRC32_CRYSTAL_ZOFF_BYTE2_REF[((z_lo >> 16) & 0xFF) as usize]
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
    #[inline(always)]
    #[must_use]
    pub const fn compute_payload_crc32_with_words(
        &self,
        phase_words: &[u64; NUM_SWARM_MICRO_CELLS],
    ) -> u32 {
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
    pub fn stamp_crc32(&mut self) {
        self.checksum_crc32 = self.compute_payload_crc32();
    }

    /// Restores `self` from `src` using width-matched scalar writes and 64-bit GPR stores
    /// (`*self.phase_words_mut() = *src.phase_words_ref()`), avoiding 256-bit YMM-to-GPR
    /// Store-to-Load Forwarding (STLF) pipeline stalls inside tight loops.
    #[inline(always)]
    pub fn copy_from_fast(&mut self, src: &Self) {
        self.magic = src.magic;
        self.version = src.version;
        self.anchor_id = src.anchor_id;
        self.sample_count = src.sample_count;
        self.zimms_section_offset48 = src.zimms_section_offset48;
        self.base_energy = src.base_energy;
        self.resultant_r_bar = src.resultant_r_bar;
        self.von_mises_kappa = src.von_mises_kappa;
        self.phase_gravity_mass = src.phase_gravity_mass;
        self.ripa_clamp_rad = src.ripa_clamp_rad;
        self.effective_inv_temp = src.effective_inv_temp;
        self.provenance_bits = src.provenance_bits;
        self.checksum_crc32 = src.checksum_crc32;
        self.dominant_slot = src.dominant_slot;
        self.action_mode_flags = src.action_mode_flags;
        self.active_swarm_mask = src.active_swarm_mask;
        self.dominant_cause_code = src.dominant_cause_code;
        self.reserved_align64 = src.reserved_align64;
        *self.phase_words_mut() = *src.phase_words_ref();
    }

    /// Verifies `magic`, `version`, and the IEEE 802.3 `checksum_crc32` integrity of this record.
    #[inline(always)]
    #[must_use]
    pub const fn verify_integrity(&self) -> bool {
        if (self.magic != GCSO_CRYSTAL_RECORD_MAGIC) | (self.version != GCSO_CRYSTAL_RECORD_VERSION)
        {
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
            return self.checksum_crc32
                == Self::EMPTY_CRYSTAL_CRC32_BY_SLOT[self.dominant_slot as usize];
        }
        self.checksum_crc32 == self.compute_payload_crc32_with_words(pw)
    }

    /// Internal register-resident M-step helper accepting the already-loaded `cur_w` in registers
    /// and an optional `precomputed_coh_opt` from the E-step to eliminate duplicate 64-head coherence evaluation.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn absorb_sample_words_with_cur_coh_and_uni_unstamped(
        &mut self,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS],
        sample_w: &[u64; NUM_SWARM_MICRO_CELLS],
        precomputed_coh_opt: Option<f32>,
        uni_flags_opt: Option<(bool, bool)>,
        sample_energy: f32,
        add_count: u32,
    ) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, bool) {
        let clean_e = branchless_flush_subnormal_f32(branchless_relu_f32(sample_energy));
        if clean_e <= 1.0e-12 || add_count == 0 {
            return (*cur_w, self.phase_gravity_mass, false);
        }

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

        if (next_count >= 2) & (r_bar >= 0.45) {
            self.provenance_bits |= CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED;
        }
        (next_w, grav_mass, words_changed)
    }

    #[inline(always)]
    fn absorb_sample_words_with_cur_unstamped(
        &mut self,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS],
        sample_w: &[u64; NUM_SWARM_MICRO_CELLS],
        sample_energy: f32,
        add_count: u32,
    ) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, bool) {
        self.absorb_sample_words_with_cur_coh_and_uni_unstamped(
            cur_w,
            sample_w,
            None,
            None,
            sample_energy,
            add_count,
        )
    }

    #[inline(always)]
    fn absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
        &mut self,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS],
        frame: &GatewayFrame,
        frame_words: &[u64; NUM_SWARM_MICRO_CELLS],
        precomputed_coh_opt: Option<f32>,
        uni_flags_opt: Option<(bool, bool)>,
    ) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, bool, bool) {
        if frame.is_repulsed() {
            return (*cur_w, self.phase_gravity_mass, false, false);
        }

        if self.sample_count == 0 {
            if frame.target_anchor_id != 0 {
                self.anchor_id = frame.target_anchor_id;
            }
            if frame.zimms_section_offset48 != 0 {
                self.zimms_section_offset48 = frame.zimms_section_offset48;
            }
            self.dominant_slot = frame.dominant_slot;
            self.ripa_clamp_rad = frame.ripa_clamp_rad;
            self.effective_inv_temp = frame.effective_inv_temp;
        } else {
            if self.ripa_clamp_rad.to_bits() != frame.ripa_clamp_rad.to_bits() {
                self.ripa_clamp_rad =
                    branchless_lerp_f32(self.ripa_clamp_rad, frame.ripa_clamp_rad, 0.25);
            }
            if self.effective_inv_temp.to_bits() != frame.effective_inv_temp.to_bits() {
                self.effective_inv_temp =
                    branchless_lerp_f32(self.effective_inv_temp, frame.effective_inv_temp, 0.25);
            }
        }
        self.action_mode_flags = frame.action_mode_flags;
        self.active_swarm_mask = frame.active_swarm_cell_mask;
        self.provenance_bits |= frame.provenance_bits;

        let (next_w, mass, words_changed) = self
            .absorb_sample_words_with_cur_coh_and_uni_unstamped(
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
    fn absorb_gateway_frame_with_cur_unstamped(
        &mut self,
        cur_w: &[u64; NUM_SWARM_MICRO_CELLS],
        frame: &GatewayFrame,
        frame_words: &[u64; NUM_SWARM_MICRO_CELLS],
    ) -> ([u64; NUM_SWARM_MICRO_CELLS], f32, bool, bool) {
        self.absorb_gateway_frame_with_cur_coh_and_uni_unstamped(
            cur_w,
            frame,
            frame_words,
            None,
            None,
        )
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
    pub fn absorb_sample_point_64(
        &mut self,
        sample_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        sample_energy: f32,
    ) -> f32 {
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
    pub fn absorb_gateway_frame_with_words(
        &mut self,
        frame: &GatewayFrame,
        frame_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
    pub fn evaluate_phase_proximity_gravity_64(
        &self,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> (f32, f32) {
        if self.sample_count == 0 || self.phase_gravity_mass <= 1.0e-12 {
            return (0.0, 0.0);
        }
        let coh = branchless_q7_coherence_64(query_phases, &self.phase_profile);
        let grav = branchless_phase_proximity_gravity_f32(
            self.phase_gravity_mass,
            coh,
            DEFAULT_PHASE_GRAVITY_EPS,
        );
        (grav, coh)
    }

    #[inline(always)]
    #[must_use]
    pub(crate) fn find_nearest_toroidal_centroid_with_cache_64(
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
        let sample_is_byte_uniform = sample_is_uniform & (sample_w0 == sample_w0.rotate_right(8));
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
                let (cw, c_uni, c_byte_uni) = match word_cache_opt {
                    Some((cache, uni_mask, byte_uni_mask)) => (
                        &cache[i],
                        ((uni_mask >> i) & 1) != 0,
                        sample_is_byte_uniform & (((byte_uni_mask >> i) & 1) != 0),
                    ),
                    None => {
                        let rw = rec.phase_words_ref();
                        let ru = if sample_is_uniform {
                            is_word_array_uniform_64(rw)
                        } else {
                            false
                        };
                        let rbu = ru & sample_is_byte_uniform & (rw[0] == rw[0].rotate_right(8));
                        (rw, ru, rbu)
                    }
                };
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
                    (
                        coherence_above_floor_from_word_arrays_64(
                            sample_words,
                            sample_is_uniform,
                            cw,
                            c_uni,
                            floor,
                        ),
                        None,
                    )
                };
                let take =
                    (!any_active) | (coh > best_coh) | ((coh == best_coh) & (mass > best_mass));
                if take {
                    best_idx = i;
                    best_coh = coh;
                    best_mass = mass;
                    best_is_uniform = c_uni;
                    best_byte_diff = diff_opt;
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
    fn init_spawned_slot_header_in_place(
        &mut self,
        assigned_id: u32,
        dominant_slot: u8,
        zoff48: u64,
    ) {
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
    fn select_or_spawn_centroid_slot_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        nearest_idx: usize,
        best_coh: f32,
        empty_opt: Option<usize>,
        weakest_idx: usize,
        best_w: [u64; NUM_SWARM_MICRO_CELLS],
        safe_split: f32,
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

    #[inline]
    pub(crate) fn step_online_toroidal_kmeans_frame_with_words_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        sample_w: &[u64; NUM_SWARM_MICRO_CELLS],
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, f32, f32, bool) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (0, frame.target_anchor_id, 0.0, 0.0, false);
        }

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
            return (
                nearest_idx,
                dom_id,
                best_coh,
                centroids[nearest_idx].phase_gravity_mass,
                false,
            );
        }

        let safe_split = branchless_clamp_f32(split_coherence_thresh, -0.95, 0.95);
        let (target_idx, spawned_new, cur_target_w) = Self::select_or_spawn_centroid_slot_64(
            centroids,
            frame,
            nearest_idx,
            best_coh,
            empty_opt,
            weakest_idx,
            best_w,
            safe_split,
        );

        let pre_coh = if spawned_new {
            Some(1.0)
        } else {
            Some(best_coh)
        };
        let uni_flags = Some((
            if spawned_new { true } else { best_is_uniform },
            sample_is_uniform,
        ));
        let fallback_aid = (target_idx as u32) + 1;
        let target = &mut centroids[target_idx];
        let prev_crc = target.checksum_crc32;
        let prev_count = target.sample_count;
        let prev_aid = if spawned_new { 0 } else { target.anchor_id };
        let prev_zoff = if spawned_new {
            0
        } else {
            target.zimms_section_offset48
        };
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

        (
            target_idx,
            winning_anchor_id,
            final_coh,
            updated_mass,
            spawned_new,
        )
    }

    /// **Classical Online Toroidal $k$-Means / DP-Means Closed-Loop E+M Step ($\mathcal{O}(K)$)**:
    /// Returns `(assigned_cluster_idx, assigned_anchor_id, centroid_coherence, updated_gravity_mass, spawned_new_cluster)`.
    #[inline(always)]
    pub fn step_online_toroidal_kmeans_frame_64(
        centroids: &mut [Self],
        frame: &GatewayFrame,
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, f32, f32, bool) {
        Self::step_online_toroidal_kmeans_frame_with_words_64(
            centroids,
            frame,
            frame.phase_words_ref(),
            split_coherence_thresh,
            trail_opt,
        )
    }

    /// Executes Online Toroidal $k$-Means / DP-Means across a batch of [`GatewayFrame`]s,
    /// returning `(last_assigned_idx, last_assigned_anchor_id, mean_coherence, spawned_clusters_count)`.
    #[inline]
    pub fn step_online_toroidal_kmeans_batch_frames_64(
        centroids: &mut [Self],
        frames: &[GatewayFrame],
        split_coherence_thresh: f32,
        mut trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (usize, u32, f32, u32) {
        if centroids.is_empty() || frames.is_empty() {
            return (0, 0, 0.0, 0);
        }
        let mut valid_count = 0u32;
        let mut spawned_count = 0u32;
        let mut coh_sum = 0.0_f32;
        let mut last_idx = 0usize;
        let mut last_aid = 0u32;
        for frame in frames {
            if frame.is_repulsed() {
                continue;
            }
            let (idx, aid, coh, _mass, spawned) =
                Self::step_online_toroidal_kmeans_frame_with_words_64(
                    centroids,
                    frame,
                    frame.phase_words_ref(),
                    split_coherence_thresh,
                    trail_opt.as_deref_mut(),
                );
            valid_count += 1;
            if spawned {
                spawned_count += 1;
            }
            coh_sum += coh;
            last_idx = idx;
            last_aid = aid;
        }
        let mean_coh = if valid_count > 0 {
            branchless_clamp_f32(
                coh_sum * fast_inv_pos_normal_f32(valid_count as f32),
                -1.0,
                1.0,
            )
        } else {
            0.0
        };
        (last_idx, last_aid, mean_coh, spawned_count)
    }

    /// **Online Toroidal $k$-Means / DP-Means on Any [`ManifoldGatewaySignal`]**:
    /// Synthesizes `signal` into a [`GatewayFrame`] and executes E-step + M-step over `centroids`.
    #[inline]
    pub fn step_online_toroidal_kmeans_signal_64<S: ManifoldGatewaySignal + ?Sized>(
        centroids: &mut [Self],
        signal: &S,
        split_coherence_thresh: f32,
        trail_opt: Option<&mut gcso_pointer_trail_t>,
    ) -> (GatewayFrame, usize, u32, f32, f32, bool) {
        let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
        let (cluster_idx, anchor_id, coh, mass, spawned) =
            Self::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                &frame,
                &frm_w,
                split_coherence_thresh,
                trail_opt,
            );
        (frame, cluster_idx, anchor_id, coh, mass, spawned)
    }
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
    /// **Online Basin Consolidation (Centroid Merging & Evaporation Pruning on $SO(2)^{64}$)**:
    /// Applies `retention_decay`, prunes centroids below `min_gravity_mass`, and merges centroids
    /// whose pairwise phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \ge \text{merge\_coherence\_thresh}$.
    #[inline]
    pub fn merge_and_prune_centroids_64(
        centroids: &mut [Self],
        merge_coherence_thresh: f32,
        min_gravity_mass: f32,
        retention_decay: f32,
    ) -> (u32, u32) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (0, 0);
        }
        let safe_merge = branchless_clamp_f32(merge_coherence_thresh, 0.25, 0.999);
        let safe_retain = branchless_clamp_f32(retention_decay, 0.0, 1.0);
        let apply_decay = safe_retain < 0.999_999;
        let min_mass = branchless_max_f32(0.0, min_gravity_mass);

        let mut pruned_count = 0u32;
        let mut merged_count = 0u32;
        let mut active_cnt = 0usize;

        let mut i = 0usize;
        while i < k_len {
            if centroids[i].sample_count > 0 {
                if apply_decay {
                    let prev_be_bits = centroids[i].base_energy.to_bits();
                    let prev_gm_bits = centroids[i].phase_gravity_mass.to_bits();
                    let next_be =
                        branchless_flush_subnormal_f32(centroids[i].base_energy * safe_retain);
                    let next_gm = branchless_flush_subnormal_f32(
                        centroids[i].phase_gravity_mass * safe_retain,
                    );
                    centroids[i].base_energy = next_be;
                    centroids[i].phase_gravity_mass = next_gm;
                    if (centroids[i].magic == GCSO_CRYSTAL_RECORD_MAGIC)
                        & (centroids[i].version == GCSO_CRYSTAL_RECORD_VERSION)
                        & (centroids[i].checksum_crc32 != 0)
                    {
                        centroids[i].checksum_crc32 = crc32_update_crystal_energy_and_mass_delta(
                            centroids[i].checksum_crc32,
                            prev_be_bits ^ next_be.to_bits(),
                            prev_gm_bits ^ next_gm.to_bits(),
                        );
                    } else {
                        centroids[i].stamp_crc32();
                    }
                }
                if centroids[i].phase_gravity_mass < min_mass {
                    let slot = centroids[i].dominant_slot;
                    centroids[i].clear_empty_in_place(slot);
                    pruned_count += 1;
                } else {
                    active_cnt += 1;
                }
            }
            i += 1;
        }

        if active_cnt < 2 {
            return (0, pruned_count);
        }

        let mut a = 0usize;
        while a < k_len {
            if centroids[a].sample_count > 0 && centroids[a].phase_gravity_mass > 1.0e-12 {
                let mut w_a = *centroids[a].phase_words_ref();
                let mut a_uni = is_word_array_uniform_64(&w_a);
                let mut a_merged = false;
                let mut b = a + 1;
                while b < k_len {
                    if centroids[b].sample_count > 0 && centroids[b].phase_gravity_mass > 1.0e-12 {
                        let w_b = *centroids[b].phase_words_ref();
                        let b_uni = if a_uni {
                            is_word_array_uniform_64(&w_b)
                        } else {
                            false
                        };
                        let coh = coherence_above_floor_from_word_arrays_64(
                            &w_a, a_uni, &w_b, b_uni, safe_merge,
                        );
                        if coh >= safe_merge {
                            let (
                                b_aid,
                                b_sc,
                                b_zoff,
                                b_be,
                                b_rbar,
                                b_kap,
                                b_mass,
                                b_clamp,
                                b_temp,
                                b_prov,
                                b_slot,
                                b_act,
                                b_swm,
                                b_cause,
                            ) = (
                                centroids[b].anchor_id,
                                centroids[b].sample_count,
                                centroids[b].zimms_section_offset48,
                                centroids[b].base_energy,
                                centroids[b].resultant_r_bar,
                                centroids[b].von_mises_kappa,
                                centroids[b].phase_gravity_mass,
                                centroids[b].ripa_clamp_rad,
                                centroids[b].effective_inv_temp,
                                centroids[b].provenance_bits,
                                centroids[b].dominant_slot,
                                centroids[b].action_mode_flags,
                                centroids[b].active_swarm_mask,
                                centroids[b].dominant_cause_code,
                            );
                            w_a = centroids[a].merge_other_centroid_words_in_place(
                                &w_a, a_uni, &w_b, b_uni, coh, b_aid, b_sc, b_zoff, b_be, b_rbar,
                                b_kap, b_mass, b_clamp, b_temp, b_prov, b_slot, b_act, b_swm,
                                b_cause,
                            );
                            a_uni = if a_uni & b_uni {
                                true
                            } else {
                                is_word_array_uniform_64(&w_a)
                            };
                            a_merged = true;
                            centroids[b].clear_empty_in_place(b_slot);
                            merged_count += 1;
                        }
                    }
                    b += 1;
                }
                if a_merged {
                    *centroids[a].phase_words_mut() = w_a;
                    centroids[a].stamp_crc32_with_words(&w_a);
                }
            }
            a += 1;
        }

        (merged_count, pruned_count)
    }
}

impl CrystallizedAttractorRecord {
    /// Evaluates macro distribution diagnostics across `centroids` in $\mathcal{O}(K^2)$ SWAR:
    /// returns `(active_clusters, total_samples, mean_resultant_r_bar, mean_von_mises_kappa, max_pairwise_coherence)`.
    #[inline]
    #[must_use]
    pub fn diagnose_macro_distribution_64(centroids: &[Self]) -> (u32, u32, f32, f32, f32) {
        let k_len = centroids.len();
        if k_len == 0 {
            return (0, 0, 0.0, 0.0, 0.0);
        }
        let mut active_cnt = 0u32;
        let mut tot_samples = 0u32;
        let mut sum_r = 0.0_f32;
        let mut sum_kappa = 0.0_f32;
        let mut max_pair_coh = -1.0_f32;
        let mut has_pair = false;

        let mut i = 0usize;
        while i < k_len {
            let rec_i = &centroids[i];
            if rec_i.sample_count > 0 && rec_i.phase_gravity_mass > 1.0e-12 {
                active_cnt += 1;
                tot_samples = tot_samples.saturating_add(rec_i.sample_count);
                sum_r += rec_i.resultant_r_bar;
                sum_kappa += rec_i.von_mises_kappa;
                let w_i = rec_i.phase_words_ref();
                let i_uni = is_word_array_uniform_64(w_i);
                let mut j = i + 1;
                while j < k_len {
                    let rec_j = &centroids[j];
                    if rec_j.sample_count > 0 && rec_j.phase_gravity_mass > 1.0e-12 {
                        let w_j = rec_j.phase_words_ref();
                        let j_uni = if i_uni {
                            is_word_array_uniform_64(w_j)
                        } else {
                            false
                        };
                        let floor = if has_pair { max_pair_coh } else { -2.0 };
                        let coh = coherence_above_floor_from_word_arrays_64(
                            w_i, i_uni, w_j, j_uni, floor,
                        );
                        max_pair_coh = branchless_select_f32(
                            (!has_pair) | (coh > max_pair_coh),
                            coh,
                            max_pair_coh,
                        );
                        has_pair = true;
                    }
                    j += 1;
                }
            }
            i += 1;
        }
        if active_cnt == 0 {
            return (0, 0, 0.0, 0.0, 0.0);
        }
        let inv_act = fast_inv_pos_normal_f32(active_cnt as f32);
        let mean_r = branchless_clamp_f32(sum_r * inv_act, 0.0, 1.0);
        let mean_k = branchless_flush_subnormal_f32(sum_kappa * inv_act);
        let pair_coh = branchless_select_f32(has_pair, max_pair_coh, 0.0);
        (active_cnt, tot_samples, mean_r, mean_k, pair_coh)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn apply_macro_centroid_pull_to_trail_64(
        trail: &mut gcso_pointer_trail_t,
        trail_w: &[u64; NUM_SWARM_MICRO_CELLS],
        trail_is_uniform: bool,
        target_w: &[u64; NUM_SWARM_MICRO_CELLS],
        target_is_uniform: bool,
        winner: &Self,
        anchor_id: u32,
        eff_coh: f32,
        tot_grav: f32,
        pull_step_u7: u8,
        extra_flags: u32,
    ) -> [u64; NUM_SWARM_MICRO_CELLS] {
        let s_u64 = pull_step_u7.min(128) as u64;
        let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let (bridged_w, bridged_uni) = swar_slerp_words_with_uniform_flag_64(
            trail_w,
            trail_is_uniform,
            target_w,
            target_is_uniform,
            s_u64,
            bias_restore,
        );
        if pull_step_u7 > 0 && !swar_words_equal_64(trail_w, &bridged_w) {
            *trail_phase_words_mut(trail) = bridged_w;
        }
        trail.target_anchor_id = anchor_id;
        trail.cluster_id = anchor_id;
        trail.attractor_pull_force = branchless_min_f32(
            trail.attractor_pull_force + 0.5 * tot_grav,
            MAX_ATTRACTOR_PULL_FORCE,
        );
        trail.stigmergic_density = branchless_min_f32(
            trail.stigmergic_density + 0.20 * branchless_relu_f32(eff_coh),
            MAX_STIGMERGIC_DENSITY,
        );
        if winner.zimms_section_offset48 != 0 {
            let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&bridged_w, bridged_uni);
            let packed_tag = pack_action_slot_tag(winner.dominant_slot, winner.action_mode_flags);
            trail.prev_ptr = trail.current_ptr;
            trail.current_ptr =
                gcso_tagged_ptr_t::pack(winner.zimms_section_offset48, packed_tag, tag_q7).raw();
        }
        trail.flags |=
            TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | extra_flags;
        bridged_w
    }
    pub(crate) fn route_coarse_to_fine_trail_from_preloaded_words_64(
        centroids: &[Self],
        trail: &mut gcso_pointer_trail_t,
        trail_w: [u64; NUM_SWARM_MICRO_CELLS],
        pull_step_u7: u8,
    ) -> (
        usize,
        u32,
        f32,
        f32,
        [u64; NUM_SWARM_MICRO_CELLS],
        [u64; NUM_SWARM_MICRO_CELLS],
    ) {
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
            return (
                0,
                trail.target_anchor_id,
                0.0,
                0.0,
                trail_w,
                [0u64; NUM_SWARM_MICRO_CELLS],
            );
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

    #[inline(always)]
    pub(crate) fn route_coarse_to_fine_trail_with_words_64(
        centroids: &[Self],
        trail: &mut gcso_pointer_trail_t,
        pull_step_u7: u8,
    ) -> (
        usize,
        u32,
        f32,
        f32,
        [u64; NUM_SWARM_MICRO_CELLS],
        [u64; NUM_SWARM_MICRO_CELLS],
    ) {
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
    pub fn route_coarse_to_fine_trail_64(
        centroids: &[Self],
        trail: &mut gcso_pointer_trail_t,
        pull_step_u7: u8,
    ) -> (usize, u32, f32, f32) {
        let (best_idx, anchor_id, best_coh, best_grav, _bridged_w, _win_w) =
            Self::route_coarse_to_fine_trail_with_words_64(centroids, trail, pull_step_u7);
        (best_idx, anchor_id, best_coh, best_grav)
    }

    #[inline]
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
            return (
                0,
                0,
                trail.target_anchor_id,
                0,
                0.0,
                0.0,
                false,
                trail_w,
                trail_w,
                trail_w,
            );
        }

        let trail_is_uniform = is_word_array_uniform_64(&trail_w);
        let trail_w0 = trail_w[0];
        let trail_b0 = (trail_w0 & 0xFF) as u8;
        let trail_is_byte_uniform = trail_is_uniform & (trail_w0 == trail_w0.rotate_right(8));
        let mut idx1 = 0usize;
        let mut coh1 = -2.0_f32;
        let mut grav1 = 0.0_f32;
        let mut uni1 = false;
        let mut idx2 = 0usize;
        let mut coh2 = -2.0_f32;
        let mut grav2 = 0.0_f32;
        let mut uni2 = false;
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
                    & ((rw0 == trail_w0) | (rw0 == rw0.rotate_right(8)));
                let (coh, grav) = if rec_uni & (rw0 == trail_w0) {
                    (1.0, branchless_q7_default_gravity_from_diff_u8(mass, 0))
                } else if rec_byte_uni {
                    let d_u8 = trail_b0.wrapping_sub(rb0);
                    (
                        Q7_FULL_COS_REF[d_u8 as usize],
                        branchless_q7_default_gravity_from_diff_u8(mass, d_u8),
                    )
                } else {
                    let floor = if (active_found >= 2) & (grav2 > mass) {
                        -0.50
                    } else {
                        -2.0
                    };
                    let c_val = coherence_above_floor_from_word_arrays_64(
                        &trail_w,
                        trail_is_uniform,
                        rec_w,
                        rec_uni,
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
                        idx1 = i;
                        coh1 = coh;
                        grav1 = grav;
                        uni1 = rec_uni;
                    } else if (active_found == 1)
                        | (grav > grav2)
                        | ((grav == grav2) & (coh > coh2))
                    {
                        idx2 = i;
                        coh2 = coh;
                        grav2 = grav;
                        uni2 = rec_uni;
                    }
                }
                active_found += 1;
            }
            i += 1;
        }

        if active_found == 0 {
            return (
                0,
                0,
                trail.target_anchor_id,
                0,
                0.0,
                0.0,
                false,
                trail_w,
                trail_w,
                trail_w,
            );
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
            let bias_lim = (128u64.wrapping_sub(s_lim)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let (tw, t_uni) =
                swar_slerp_words_with_uniform_flag_64(&w1, w1_uni, &w2, w2_uni, s_lim, bias_lim);
            (tw, t_uni, w2)
        } else {
            (w1, w1_uni, [0u64; NUM_SWARM_MICRO_CELLS])
        };

        let win1 = &centroids[idx1];
        let aid1 = if win1.anchor_id != 0 {
            win1.anchor_id
        } else {
            (idx1 as u32) + 1
        };
        let aid2 = if is_liminal {
            let win2 = &centroids[idx2];
            if win2.anchor_id != 0 {
                win2.anchor_id
            } else {
                (idx2 as u32) + 1
            }
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

        (
            idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, bridged_w, w1, out_w2,
        )
    }
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
    /// (e.g. [`crate::traits::DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO`]), interpolates along the shortest $S^1$
    /// geodesic between both centroids (bridging the *Limina* inter-memory boundary) and stamps
    /// `TRAIL_FLAG_LIMINAL_BEAT_ACTIVE | TRAIL_FLAG_SYNERGY_RESONANCE`.
    ///
    /// Returns `(primary_idx, secondary_idx, primary_anchor_id, secondary_anchor_id, blended_coherence, total_gravity, is_liminal_bifurcation)`.
    #[inline]
    pub fn route_top2_liminal_centroids_64(
        centroids: &[Self],
        trail: &mut gcso_pointer_trail_t,
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
    ) -> (usize, usize, u32, u32, f32, f32, bool) {
        let (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, _bw, _w1, _w2) =
            Self::route_top2_liminal_centroids_with_words_64(
                centroids,
                trail,
                pull_step_u7,
                liminal_gravity_ratio_thresh,
            );
        (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal)
    }

    /// **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K) + \mathcal{O}(M_{\mathrm{local}})$)**:
    /// 1. Executes the $\mathcal{O}(K)$ Online Toroidal $k$-Means E-step to identify the winning
    ///    macro centroid `win_k` and stamps `src_trail.cluster_id = macro_anchor_id`.
    /// 2. Evaluates `candidate_trails` with a macro-cluster affinity bonus (`cluster_bonus_gain`)
    ///    for trails belonging to `macro_anchor_id` (or geometrically aligned with `centroids[win_k]`),
    ///    links the winning candidate into `src_trail.linked_trail_id`, and returns
    ///    `(winning_centroid_idx, winning_trail_id, best_score, best_coherence, found_valid)`.
    #[inline]
    pub fn route_coarse_to_fine_candidates_64(
        centroids: &[Self],
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &[gcso_pointer_trail_t],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        cluster_bonus_gain: f32,
        contradiction_threshold: f32,
    ) -> (usize, u32, f32, f32, bool) {
        let self_w = *trail_phase_words_ref(src_trail);
        let (
            win_k,
            _win_coh,
            _win_grav,
            _empty_opt,
            _weak_idx,
            win_w,
            self_is_uniform,
            win_uni_from_estep,
            _,
        ) = Self::find_nearest_toroidal_centroid_with_cache_64(centroids, &self_w, None);
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

        if candidate_trails.is_empty() {
            return (win_k, src_trail.linked_trail_id, 0.0, 0.0, false);
        }

        let safe_d = branchless_select_u32(head_dim > 0, head_dim, 128);
        let inv_sqrt_d = if safe_d <= 256 {
            branchless_popcount_rsqrt_u32(safe_d)
        } else {
            branchless_rsqrt_f32(safe_d as f32)
        };
        let phase_scale =
            branchless_flush_subnormal_f32(branchless_relu_f32(phase_interference_gain))
                * inv_sqrt_d;
        let clean_eta = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_prior_gain));
        let clean_link = branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));
        let clean_cluster_bonus =
            branchless_flush_subnormal_f32(branchless_relu_f32(cluster_bonus_gain));
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let cur_linked_id = src_trail.linked_trail_id;
        let self_w0 = self_w[0];
        let self_b0 = (self_w0 & 0xFF) as u8;
        let self_is_byte_uniform = self_is_uniform & (self_w0 == self_w0.rotate_right(8));
        let mut win_meta_evaluated = false;
        let mut win_eq_self = false;
        let mut win_is_uniform = false;

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
            let cand_id = idx as u32;
            let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
            let link_term = branchless_select_f32(is_linked, clean_link, 0.0);
            let in_macro_cluster = (macro_anchor_id != 0)
                & ((cand.cluster_id == macro_anchor_id)
                    | (cand.target_anchor_id == macro_anchor_id));
            let is_unstamped_for_geo = (!in_macro_cluster)
                & has_win_centroid
                & (clean_cluster_bonus > 0.0)
                & (cand.cluster_id == 0)
                & (cand.target_anchor_id == 0);
            let max_macro_bonus = branchless_select_f32(
                in_macro_cluster | is_unstamped_for_geo,
                clean_cluster_bonus,
                0.0,
            );
            let clean_rho = branchless_clamp_f32(
                branchless_relu_f32(cand.stigmergic_density),
                0.0,
                MAX_STIGMERGIC_DENSITY,
            );
            let prior_term = clean_eta * fast_ln_1p_bounded_f32(clean_rho);
            let non_phase_upper = prior_term + link_term + max_macro_bonus;
            if found_valid && (phase_scale + non_phase_upper <= best_score) {
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
                } else if self_is_byte_uniform && (cw0 == cw0.rotate_right(8)) {
                    Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
                } else {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &self_w, true, cand_w, true,
                    )
                }
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(
                    &self_w,
                    self_is_uniform,
                    cand_w,
                    false,
                )
            };
            if coh >= -safe_thresh {
                let base_score_no_macro = (phase_scale * coh) + prior_term + link_term;
                if !found_valid || (base_score_no_macro + max_macro_bonus > best_score) {
                    let geo_macro_affinity = if in_macro_cluster {
                        1.0
                    } else if is_unstamped_for_geo {
                        if !win_meta_evaluated {
                            win_eq_self = swar_words_equal_64(&self_w, &win_w);
                            win_is_uniform = if win_eq_self {
                                self_is_uniform
                            } else if self_is_uniform {
                                win_uni_from_estep
                            } else {
                                is_word_array_uniform_64(&win_w)
                            };
                            win_meta_evaluated = true;
                        }
                        let c_win_coh = if win_eq_self {
                            coh
                        } else {
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
                        };
                        branchless_relu_f32(c_win_coh)
                    } else {
                        0.0
                    };
                    let score = branchless_flush_subnormal_f32(
                        base_score_no_macro + (clean_cluster_bonus * geo_macro_affinity),
                    );
                    let take = (!found_valid) | (score > best_score);
                    best_id = branchless_select_u32(take, cand_id, best_id);
                    best_score = branchless_select_f32(take, score, best_score);
                    best_coh = branchless_select_f32(take, coh, best_coh);
                    found_valid = true;
                }
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
        let init_is_byte_uniform = init_is_uniform & (init_w0 == init_w0.rotate_right(8));
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
        let mut acc_w = init_w;
        let mut acc_is_uniform = init_is_uniform;
        let mut best_grav = 0.0_f32;
        let mut dom_id = 0u32;
        let mut dom_clamp = DEFAULT_RIPA_CLAMP_MAX_RAD;
        let mut coh_sum = 0.0_f32;
        let mut active_cnt = 0usize;

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
                } else if rec_is_uniform & init_is_byte_uniform & (rw0 == rw0.rotate_right(8)) {
                    let d_u8 = init_b0.wrapping_sub(rb0);
                    (
                        Q7_FULL_COS_REF[d_u8 as usize],
                        branchless_q7_default_gravity_from_diff_u8(rec.phase_gravity_mass, d_u8),
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
    pub fn fold_crystallized_ensemble_64(
        records: &[Self],
        trail_phases_inout: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> (f32, u32, f32) {
        let (_w, tot_pull, dom_id, dom_clamp) =
            Self::fold_crystallized_ensemble_words_with_init_pull_64(
                records,
                trail_phases_inout,
                0.0,
            );
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
        if count == 0 {
            return Ok(&[]);
        }
        let req_bytes = count
            .checked_mul(size_of::<Self>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if bytes.len() < req_bytes {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if !is_aligned_to(bytes, align_of::<Self>()) {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: `bytes` is verified 64-byte aligned and at least `count * 128` bytes long, and `Self: Pod`.
        let recs = unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<Self>(), count) };
        let mut i = 0usize;
        while i < count {
            if !recs[i].verify_integrity() {
                return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
            }
            i += 1;
        }
        Ok(recs)
    }

    /// Serializes `records` into a `.gcso` (ZIMMS) byte slice (`dst_bytes`), stamping each
    /// record's IEEE 802.3 CRC32 checksum when not already stamped.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < records.len() * 128`.
    #[inline]
    pub fn write_records_to_zimms_bytes(
        records: &[Self],
        dst_bytes: &mut [u8],
    ) -> GcsoResult<usize> {
        let total_bytes = records
            .len()
            .checked_mul(size_of::<Self>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if dst_bytes.len() < total_bytes {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if total_bytes == 0 {
            return Ok(0);
        }
        #[cfg(target_endian = "little")]
        if is_aligned_to(dst_bytes, align_of::<u64>()) {
            // Write 64-bit `u64` words directly so subsequent 64-bit CRC32 & header validation loads
            // in `from_zimms_bytes_zero_copy` forward cleanly from the x86 Store Buffer with zero
            // 256-bit-to-64-bit Store-to-Load Forwarding (STLF) stalls.
            let expected_w0 =
                (GCSO_CRYSTAL_RECORD_MAGIC as u64) | ((GCSO_CRYSTAL_RECORD_VERSION as u64) << 32);
            let dst_ptr = dst_bytes.as_mut_ptr().cast::<u64>();
            let mut idx = 0usize;
            while idx < records.len() {
                let r = &records[idx];
                let src_w = unsafe { &*core::ptr::from_ref(r).cast::<[u64; 16]>() };
                let mut w0 = src_w[0];
                let mut w6 = src_w[6];
                let valid_hdr = w0 == expected_w0;
                if (!valid_hdr) | ((w6 >> 32) == 0) {
                    let crc = if valid_hdr {
                        r.compute_payload_crc32()
                    } else {
                        let mut rec = *r;
                        rec.magic = GCSO_CRYSTAL_RECORD_MAGIC;
                        rec.version = GCSO_CRYSTAL_RECORD_VERSION;
                        w0 = expected_w0;
                        rec.compute_payload_crc32()
                    };
                    w6 = (w6 & 0xFFFF_FFFFu64) | ((crc as u64) << 32);
                }
                let out_w = unsafe { &mut *dst_ptr.add(idx * 16).cast::<[u64; 16]>() };
                out_w[0] = w0;
                out_w[1] = src_w[1];
                out_w[2] = src_w[2];
                out_w[3] = src_w[3];
                out_w[4] = src_w[4];
                out_w[5] = src_w[5];
                out_w[6] = w6;
                out_w[7] = src_w[7];
                out_w[8] = src_w[8];
                out_w[9] = src_w[9];
                out_w[10] = src_w[10];
                out_w[11] = src_w[11];
                out_w[12] = src_w[12];
                out_w[13] = src_w[13];
                out_w[14] = src_w[14];
                out_w[15] = src_w[15];
                idx += 1;
            }
            return Ok(total_bytes);
        }
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

const _: () = {
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
};
