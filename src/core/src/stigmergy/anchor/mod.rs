// name: src/core/src/stigmergy/anchor/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `.lmnk` Crystallized Attractor Record (`PhaseAnchorRecord`, 128B `Pod` `"LNKR"`),
//! Online Toroidal $k$-Means / DP-Means Macro Clustering, Coarse-to-Fine & Top-2 Liminal
//! Bifurcation Routing, and trail crystallization coordination.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::mem::{align_of, offset_of, size_of};

use self::crc32_delta::{
    ANCHOR_CRC32_DEFAULT_W5_FINAL_CONTRIB, ANCHOR_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB,
    ANCHOR_CRC32_KAPPA_MAX_FINAL_CONTRIB, ANCHOR_CRC32_RBAR_ONE_FINAL_CONTRIB,
    ANCHOR_CRC32_W0_FINAL_CONTRIB, AnchorHeaderDelta, CRC32_ANCHOR_ACTION_BYTE57_REF,
    CRC32_ANCHOR_AID_BYTE0_REF, CRC32_ANCHOR_AID_BYTE1_REF, CRC32_ANCHOR_PROV_BYTE0_REF,
    CRC32_ANCHOR_PROV_BYTE1_REF, CRC32_ANCHOR_SC_BYTE0_REF, CRC32_ANCHOR_SLOT_BYTE56_LUT_8,
    CRC32_ANCHOR_ZOFF_BYTE1_REF, CRC32_ANCHOR_ZOFF_BYTE2_REF, CRC32_ANCHOR_ZOFF_BYTE3_REF,
    crc32_anchor_be24_final_u32, crc32_anchor_gm36_final_u32, crc32_anchor_zoff16_final_u32,
    crc32_update_anchor_absorption_delta, crc32_update_anchor_stationary_fields_delta,
};
use crate::foundation::{
    ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_PASSIVE_SIDECAR, ATTRACTOR_SLOT_ID_MASK,
    DEFAULT_PHASE_CLAMP_MAX_RAD,
};
use crate::foundation::{
    CRC32_IEEE_SLICE16_REF, Pod, Q7_FULL_COS_REF, RawPhaseProjectable, TaggedPtr,
    VON_MISES_KAPPA_AT_MAX_R_BITS, atan2_to_q7_word_x8_simd, branchless_atan2_to_q7_fast,
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_maturity_scale_u32, branchless_min_f32, branchless_relu_f32, branchless_sincos_q7,
    branchless_von_mises_kappa, circular_r_bar_from_words_64,
    coherence_above_floor_from_word_arrays_64, coherence_from_uniform_words_x8,
    crc32_slice16_dep_lo0, crc32_slice16_indep_hi0_u32, crc32_slice16_indep_hi1_u32,
    crc32_slice16_indep_lo1_u32, crc32_slice16_step_u64x2, crc32_step_block64_words, crc32_step_l3,
    crc32_step_l4, crc32_step_l5, crc32_step_l6, is_aligned_to, is_word_array_uniform_64,
    load_q7_words_64, step_toroidal_centroid_words_cached_64, store_q7_words_64,
};
use crate::liminal::{
    AbstractDomain, DEFAULT_ENTROPY_EPS, Grounded, Indivisible, LMNK_ABI_VERSION_HEX,
    LMNK_ERR_ALIGNMENT, LMNK_ERR_BUFFER_TOO_SMALL, LMNK_ERR_CONTAINER_CORRUPTED,
    LMNK_ERR_INVALID_ARGUMENT, LmnkResult, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES, PhaseQ7,
    Scoped, branchless_select_f32, branchless_select_q7, branchless_select_usize,
    is_compatible_abi_version,
};

/// Zero-byte domain-tagged 64-head Q7 phase profile for grounded crystallization (`Grounded` phase).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseProfile64(pub [PhaseQ7; MAX_ATTENTION_HEADS]);

impl AbstractDomain for PhaseProfile64 {
    const DOMAIN_ID: &'static str = "stigmergy.phase_profile_64";
}

/// Zero-byte domain-tagged borrowed 64-head Q7 phase profile for grounded sample absorption (`Grounded` phase).
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct PhaseProfileRef64<'a>(pub &'a [PhaseQ7; MAX_ATTENTION_HEADS]);

impl AbstractDomain for PhaseProfileRef64<'_> {
    const DOMAIN_ID: &'static str = "stigmergy.phase_profile_ref_64";
}

/// Zero-byte domain-tagged borrowed 8-granule SWAR phase-word array for register-resident grounded absorption (`Grounded` phase).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseWordsRef64<'a>(pub &'a [u64; NUM_STIGMERGIC_GRANULES]);

impl AbstractDomain for PhaseWordsRef64<'_> {
    const DOMAIN_ID: &'static str = "stigmergy.phase_words_ref_64";
}

const _: () = {
    assert!(size_of::<Indivisible<Grounded, PhaseProfile64>>() == MAX_ATTENTION_HEADS);
    assert!(
        align_of::<Indivisible<Grounded, PhaseProfile64>>()
            == align_of::<[PhaseQ7; MAX_ATTENTION_HEADS]>()
    );
    assert!(
        size_of::<Indivisible<Grounded, PhaseProfileRef64<'static>>>()
            == size_of::<&[PhaseQ7; MAX_ATTENTION_HEADS]>()
    );
    assert!(
        align_of::<Indivisible<Grounded, PhaseProfileRef64<'static>>>()
            == align_of::<&[PhaseQ7; MAX_ATTENTION_HEADS]>()
    );
    assert!(
        size_of::<Indivisible<Grounded, PhaseWordsRef64<'static>>>()
            == size_of::<&[u64; NUM_STIGMERGIC_GRANULES]>()
    );
    assert!(
        align_of::<Indivisible<Grounded, PhaseWordsRef64<'static>>>()
            == align_of::<&[u64; NUM_STIGMERGIC_GRANULES]>()
    );
};
use crate::stigmergy::trail::{PointerTrail, trail_phase_words_ref};
use crate::stigmergy::{ATTRACTOR_SLOT_PERSONA, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY};

/// Nominal cause code (`0`) stored in [`PhaseAnchorRecord::root_cause_code`].
pub const ANCHOR_ROOT_CAUSE_NOMINAL: u8 = 0;

/// Provenance maturity flag (`bit 12`) stamped on [`PhaseAnchorRecord::lineage_flags`]
/// once sample count and circular coherence cross the crystallization gate.
pub const ANCHOR_LINEAGE_MATURE_BIT: u32 = 1 << 12;

/// Narrow phase clamp authority (`0.14 rad`) for Objective Fact grounding (`Fact` sub-heads).
pub const DEFAULT_FACT_PHASE_CLAMP_RAD: f32 = 0.14;

/// Wide phase clamp authority (`0.24 rad`) for Subjective Persona modulation (`Explore/Logic` sub-heads).
pub const DEFAULT_PERSONA_PHASE_CLAMP_RAD: f32 = 0.24;

/// Default minimum cosine phase coherence threshold (`0.35`) for Online Toroidal DP-Means cluster splitting.
pub const DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE: f32 = 0.35;

/// Default cosine phase coherence threshold (`0.85`) for autonomous online consolidation of redundant macro centroids.
pub const DEFAULT_TOROIDAL_CENTROID_MERGE_COHERENCE: f32 = 0.85;

/// Default secondary-to-primary gravity ratio threshold (`0.55`) triggering Top-2 Liminal Bifurcation Routing.
pub const DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO: f32 = 0.55;

/// Magic identifier (`0x524B4E4C` = ASCII `"LNKR"`) for `.lmnk` `PhaseAnchorRecord` blocks.
pub const LMNK_PHASE_ANCHOR_MAGIC: u32 = 0x524B_4E4C;

/// Version constant for `PhaseAnchorRecord` binary serialization ([`LMNK_ABI_VERSION_HEX`]).
pub const LMNK_PHASE_ANCHOR_VERSION: u32 = LMNK_ABI_VERSION_HEX;

/// Fixed capacity of phase anchors per accumulator (`64`).
pub const MAX_PHASE_ANCHORS: usize = 64;

/// Maximum dimension supported for embedding vectors in inline zero-alloc storage (`128`).
pub const MAX_EMBEDDING_DIM: usize = 128;

pub(crate) const MAX_REGISTER_CACHED_CENTROIDS: usize = 16;

pub(crate) mod crc32_delta;
/// Online toroidal $k$-means / DP-Means merging and pruning.
pub mod kmeans;
/// Coarse-to-fine and Top-2 bifurcation routing over phase anchors.
pub mod routing;
/// Episodic phase sample projection and steering traits.
pub mod sample_steering;

pub use self::routing::CoarseToFineParams;
pub use self::sample_steering::*;

/// Outcome emitted by branchless dissipative overflow, residual scatter, and autonomous
/// daughter-attractor birth (`PhaseAnchorRecord::step_dissipative_overflow_scatter_or_spawn_64`).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DissipativeOverflowReport {
    /// Peeled 64-head SWAR residual phase words (`sample_w - primary_w`) propagated on overflow.
    pub residual_words: [u64; NUM_STIGMERGIC_GRANULES],
    /// Anchor ID of `primary_idx`.
    pub primary_anchor_id: u32,
    /// Anchor ID of `landed_idx`.
    pub landed_anchor_id: u32,
    /// Raw phase coherence against `primary_idx`.
    pub primary_coherence: f32,
    /// Residual phase coherence against the best secondary scatter candidate.
    pub scatter_coherence: f32,
    /// Saturation/dispersion pressure subtracted from `primary_coherence` (`(1 - r_bar) * 0.35 + ln(1 + count) * 0.06`).
    pub overflow_pressure: f32,
    /// Updated phase-gravity mass of `landed_idx`.
    pub updated_gravity_mass: f32,
    /// Nearest primary attractor slot index evaluated before overflow pressure.
    pub primary_idx: u16,
    /// Slot index where the sample or its peeled residual settled (`primary_idx`, secondary scatter target, or spawned daughter).
    pub landed_idx: u16,
    /// `true` when `primary_idx` overflowed its capacity/coherence bound.
    pub overflowed: bool,
    /// `true` when the overflow residual settled onto an existing secondary attractor (`landed_idx != primary_idx`).
    pub scattered: bool,
    /// `true` when unsettled overflow autonomously spawned a new daughter attractor.
    pub spawned_new: bool,
    /// Explicit trailing byte padding to preserve a packed 96-byte layout.
    pub _reserved: u8,
}

const _: () = {
    assert!(core::mem::size_of::<DissipativeOverflowReport>() == 96);
    assert!(core::mem::align_of::<DissipativeOverflowReport>() == 32);
    assert!(core::mem::offset_of!(DissipativeOverflowReport, residual_words) == 0);
};

impl Default for DissipativeOverflowReport {
    #[inline]
    fn default() -> Self {
        Self {
            residual_words: [0u64; NUM_STIGMERGIC_GRANULES],
            primary_anchor_id: 0,
            landed_anchor_id: 0,
            primary_coherence: 0.0,
            scatter_coherence: 0.0,
            overflow_pressure: 0.0,
            updated_gravity_mass: 0.0,
            primary_idx: 0,
            landed_idx: 0,
            overflowed: false,
            scattered: false,
            spawned_new: false,
            _reserved: 0,
        }
    }
}

/// 128-byte (`#[repr(C, align(64))]`) [`Pod`] record representing a bottom-up crystallized
/// macro attractor centroid on $SO(2)^{64}$ persisted inside `.lmnk` (mapped-storage) containers.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseAnchorRecord {
    /// Format magic identifier (`LMNK_PHASE_ANCHOR_MAGIC` = `0x524B4E4C` `"LNKR"`, `offset 0..4`).
    pub magic: u32,
    /// Record ABI version (`LMNK_PHASE_ANCHOR_VERSION` = [`crate::liminal::LMNK_ABI_VERSION_HEX`], `offset 4..8`).
    pub version: u32,
    /// Full 32-bit macro attractor anchor ID on the Single Unified Attractor Manifold (`offset 8..12`).
    pub anchor_id: u32,
    /// Number of coherent sample points absorbed into this centroid (`offset 12..16`).
    pub sample_count: u32,
    /// Canonical 48-bit `.lmnk` (mapped-storage) section offset (`bits 0..=47`, `offset 16..24`).
    pub mapped_section_offset48: u64,
    /// Accumulated base gravitational energy $E_{\mathrm{base}} \ge 0$ (`offset 24..28`).
    pub base_energy: f32,
    /// Circular mean resultant length $\bar{R}_c \in [0, 1]$ across the 64 heads (`offset 28..32`).
    pub resultant_r_bar: f32,
    /// von Mises concentration parameter $\kappa_c \ge 0$ of the crystallized centroid (`offset 32..36`).
    pub von_mises_kappa: f32,
    /// Phase-Proximity Gravity mass $M_{\mathrm{grav}} = E_{\mathrm{base}} \cdot \bar{R}_c (1 + 0.1 \ln(1 + N))$ (`offset 36..40`).
    pub phase_gravity_mass: f32,
    /// Effective `PhaseClamp` limit in radians ($\theta_{\max}$, `offset 40..44`).
    pub phase_clamp_rad: f32,
    /// Effective multiplicative inverse-temperature scale $\beta_{\mathrm{eff}}$ (`offset 44..48`).
    pub effective_inv_temp: f32,
    /// Accumulated causal provenance bitmask (`CAUSAL_AXIS_* | CAUSAL_CAUSE_*`, `offset 48..52`).
    pub lineage_flags: u32,
    /// IEEE 802.3 CRC32 checksum computed over bytes `[0..52]` and `[56..128]` (`offset 52..56`).
    pub checksum_crc32: u32,
    /// Dominant 3-bit working lane ID (`ATTRACTOR_SLOT_*`, `0..=7`, `offset 56`).
    pub dominant_slot: u8,
    /// Resolved 5-bit `ACTION_MODE_*` execution flags (`offset 57`).
    pub action_mode_flags: u8,
    /// Active 8-cell Micro-Swarm mask (`0x00..=0xFF`, `offset 58`).
    pub active_swarm_mask: u8,
    /// Primary root-cause classification code (`ROOT_CAUSE_*`, `offset 59`).
    pub root_cause_code: u8,
    /// Reserved 32-bit zero field aligning `phase_profile` to exact byte offset `64` (`offset 60..64`).
    pub reserved_align64: u32,
    /// Crystallized 64-head Q7 centroid phase profile on $SO(2)^{64}$ (`64` bytes at `offset 64..128`).
    pub phase_profile: [PhaseQ7; MAX_ATTENTION_HEADS],
}

// SAFETY: `PhaseAnchorRecord` is `#[repr(C, align(64))]` of exact size 128 bytes with zero padding.
unsafe impl Pod for PhaseAnchorRecord {}

impl Default for PhaseAnchorRecord {
    #[inline(always)]
    fn default() -> Self {
        Self::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0)
    }
}

impl PhaseAnchorRecord {
    /// Borrows `self.phase_profile` (`offset 64` of `#[repr(C, align(64))]` [`Self`], guaranteed
    /// 64-byte aligned) as a zero-copy `&[u64; 8]` SWAR reference on little-endian targets,
    /// eliminating 64-byte stack copies and 256-bit YMM-to-GPR Store-to-Load Forwarding (STLF) stalls.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_ref(&self) -> &[u64; NUM_STIGMERGIC_GRANULES] {
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: `PhaseAnchorRecord` is `#[repr(C, align(64))]` and `phase_profile`
        // is at exact byte offset 64 (`64 % 8 == 0`), with identical 64-byte layout to `[u64; 8]`.
        unsafe {
            &*(self
                .phase_profile
                .as_ptr()
                .cast::<[u64; NUM_STIGMERGIC_GRANULES]>())
        }
    }

    /// Mutably borrows `self.phase_profile` (`offset 64` of `#[repr(C, align(64))]`
    /// [`PhaseAnchorRecord`], guaranteed 64-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_mut(&mut self) -> &mut [u64; NUM_STIGMERGIC_GRANULES] {
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: `PhaseAnchorRecord` is `#[repr(C, align(64))]` and `phase_profile`
        // is at exact byte offset 64 (`64 % 8 == 0`), with identical 64-byte layout to `[u64; 8]`.
        unsafe {
            &mut *(self
                .phase_profile
                .as_mut_ptr()
                .cast::<[u64; NUM_STIGMERGIC_GRANULES]>())
        }
    }

    #[inline(always)]
    pub(crate) fn clear_empty_in_place(&mut self, dominant_slot: u8) {
        self.init_spawned_slot_header_in_place(0, dominant_slot, 0);
        *self.phase_words_mut() = [0u64; NUM_STIGMERGIC_GRANULES];
    }

    #[inline(always)]
    #[must_use]
    const fn new_unstamped(
        anchor_id: u32,
        dominant_slot: u8,
        mapped_section_offset48: u64,
    ) -> Self {
        Self {
            magic: LMNK_PHASE_ANCHOR_MAGIC,
            version: LMNK_PHASE_ANCHOR_VERSION,
            anchor_id,
            sample_count: 0,
            mapped_section_offset48: mapped_section_offset48 & TaggedPtr::ADDR_MASK,
            base_energy: 0.0,
            resultant_r_bar: 0.0,
            von_mises_kappa: 0.0,
            phase_gravity_mass: 0.0,
            phase_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            effective_inv_temp: 1.0,
            lineage_flags: 0,
            checksum_crc32: 0,
            dominant_slot: dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            action_mode_flags: ACTION_PRESET_PASSIVE_SIDECAR,
            active_swarm_mask: 0xFF,
            root_cause_code: ANCHOR_ROOT_CAUSE_NOMINAL,
            reserved_align64: 0,
            phase_profile: [0i8; MAX_ATTENTION_HEADS],
        }
    }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn pack_w7_lo(&self) -> u32 {
        u32::from_le_bytes([
            self.dominant_slot,
            self.action_mode_flags,
            self.active_swarm_mask,
            self.root_cause_code,
        ])
    }

    #[inline]
    #[must_use]
    const fn compute_payload_crc32_with_words_raw(
        &self,
        phase_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ) -> u32 {
        let w2 = self.mapped_section_offset48;
        let be_bits = self.base_energy.to_bits();
        let rb_bits = self.resultant_r_bar.to_bits();
        let kp_bits = self.von_mises_kappa.to_bits();
        let gm_bits = self.phase_gravity_mass.to_bits();
        let cl_bits = self.phase_clamp_rad.to_bits();
        let tm_bits = self.effective_inv_temp.to_bits();
        let lut = CRC32_IEEE_SLICE16_REF;
        let mut final_acc = 0u32;

        let c0 = if (self.magic == LMNK_PHASE_ANCHOR_MAGIC)
            & (self.version == LMNK_PHASE_ANCHOR_VERSION)
        {
            final_acc ^= ANCHOR_CRC32_W0_FINAL_CONTRIB;
            if ((self.anchor_id >> 16) == 0) & ((self.sample_count >> 8) == 0) {
                final_acc ^= CRC32_ANCHOR_AID_BYTE0_REF[(self.anchor_id & 0xFF) as usize]
                    ^ CRC32_ANCHOR_AID_BYTE1_REF[((self.anchor_id >> 8) & 0xFF) as usize]
                    ^ CRC32_ANCHOR_SC_BYTE0_REF[self.sample_count as usize];
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
            final_acc ^= crc32_anchor_be24_final_u32(be_bits);
        }
        if gm_bits != 0 {
            final_acc ^= crc32_anchor_gm36_final_u32(gm_bits);
        }

        let w2_lo = (w2 as u32) ^ c0;
        let w2_hi = (w2 >> 32) as u32;
        let w2_c = if (w2_lo | w2_hi) == 0 {
            0
        } else if (c0 == 0) & (w2_hi == 0) & ((w2_lo & 0xFF) == 0) {
            final_acc ^= CRC32_ANCHOR_ZOFF_BYTE1_REF[((w2_lo >> 8) & 0xFF) as usize]
                ^ CRC32_ANCHOR_ZOFF_BYTE2_REF[((w2_lo >> 16) & 0xFF) as usize]
                ^ CRC32_ANCHOR_ZOFF_BYTE3_REF[(w2_lo >> 24) as usize];
            0
        } else if w2_hi == 0 {
            crc32_slice16_dep_lo0(w2_lo)
        } else {
            crc32_slice16_dep_lo0(w2_lo) ^ crc32_slice16_indep_hi0_u32(w2_hi)
        };
        let rb_c = if rb_bits == 0x3F80_0000u32 {
            final_acc ^= ANCHOR_CRC32_RBAR_ONE_FINAL_CONTRIB;
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
            final_acc ^= ANCHOR_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB;
            0
        } else if kp_bits == 0x4248_0000u32 {
            final_acc ^= ANCHOR_CRC32_KAPPA_MAX_FINAL_CONTRIB;
            0
        } else if kp_bits == 0 {
            0
        } else {
            crc32_slice16_dep_lo0(kp_bits)
        };
        const PERSONA_W5_FINAL_CONTRIB: u32 = ANCHOR_CRC32_DEFAULT_W5_FINAL_CONTRIB
            ^ crc32_step_l5(crc32_slice16_indep_lo1_u32(
                DEFAULT_PERSONA_PHASE_CLAMP_RAD.to_bits() ^ DEFAULT_PHASE_CLAMP_MAX_RAD.to_bits(),
            ));
        const FACT_W5_FINAL_CONTRIB: u32 = ANCHOR_CRC32_DEFAULT_W5_FINAL_CONTRIB
            ^ crc32_step_l5(crc32_slice16_indep_lo1_u32(
                DEFAULT_FACT_PHASE_CLAMP_RAD.to_bits() ^ DEFAULT_PHASE_CLAMP_MAX_RAD.to_bits(),
            ));
        let w5_c = if tm_bits == 0x3F80_0000u32 {
            if cl_bits == DEFAULT_PHASE_CLAMP_MAX_RAD.to_bits() {
                final_acc ^= ANCHOR_CRC32_DEFAULT_W5_FINAL_CONTRIB;
                0
            } else if cl_bits == DEFAULT_PERSONA_PHASE_CLAMP_RAD.to_bits() {
                final_acc ^= PERSONA_W5_FINAL_CONTRIB;
                0
            } else if cl_bits == DEFAULT_FACT_PHASE_CLAMP_RAD.to_bits() {
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

        let prov_c = if (self.lineage_flags >> 16) == 0 {
            final_acc ^= CRC32_ANCHOR_PROV_BYTE0_REF[(self.lineage_flags & 0xFF) as usize]
                ^ CRC32_ANCHOR_PROV_BYTE1_REF[((self.lineage_flags >> 8) & 0xFF) as usize];
            0
        } else {
            crc32_slice16_dep_lo0(self.lineage_flags)
        };
        const DEFAULT_W7_MASK_CAUSE_CRC_C3: u32 = CRC32_IEEE_SLICE16_REF[5][0xFF]
            ^ CRC32_IEEE_SLICE16_REF[4][ANCHOR_ROOT_CAUSE_NOMINAL as usize];
        const DEFAULT_W7_MASK_CAUSE_FINAL_CONTRIB: u32 =
            crc32_step_l4(DEFAULT_W7_MASK_CAUSE_CRC_C3);
        let slot_fast = (self.dominant_slot & 0xF8) == 0;
        let mask_cause_default =
            (self.active_swarm_mask == 0xFF) & (self.root_cause_code == ANCHOR_ROOT_CAUSE_NOMINAL);
        let w7_c = if slot_fast & mask_cause_default {
            final_acc ^= CRC32_ANCHOR_SLOT_BYTE56_LUT_8[(self.dominant_slot & 7) as usize]
                ^ CRC32_ANCHOR_ACTION_BYTE57_REF[self.action_mode_flags as usize]
                ^ DEFAULT_W7_MASK_CAUSE_FINAL_CONTRIB;
            0
        } else {
            (lut[7][self.dominant_slot as usize] ^ lut[6][self.action_mode_flags as usize])
                ^ if mask_cause_default {
                    DEFAULT_W7_MASK_CAUSE_CRC_C3
                } else {
                    lut[5][self.active_swarm_mask as usize] ^ lut[4][self.root_cause_code as usize]
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

    pub(crate) const EMPTY_ANCHOR_CRC32_BY_SLOT: [u32; 8] = [
        Self::new_unstamped(0, 0, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_STIGMERGIC_GRANULES]),
        Self::new_unstamped(0, 1, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_STIGMERGIC_GRANULES]),
        Self::new_unstamped(0, 2, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_STIGMERGIC_GRANULES]),
        Self::new_unstamped(0, 3, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_STIGMERGIC_GRANULES]),
        Self::new_unstamped(0, 4, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_STIGMERGIC_GRANULES]),
        Self::new_unstamped(0, 5, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_STIGMERGIC_GRANULES]),
        Self::new_unstamped(0, 6, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_STIGMERGIC_GRANULES]),
        Self::new_unstamped(0, 7, 0)
            .compute_payload_crc32_with_words_raw(&[0u64; NUM_STIGMERGIC_GRANULES]),
    ];

    /// Creates an empty, CRC32-stamped [`PhaseAnchorRecord`] ready to absorb sample points.
    #[inline]
    #[must_use]
    pub const fn new_empty(
        anchor_id: u32,
        dominant_slot: u8,
        mapped_section_offset48: u64,
    ) -> Self {
        let mut rec = Self::new_unstamped(anchor_id, dominant_slot, mapped_section_offset48);
        let zoff = rec.mapped_section_offset48;
        let base_crc =
            Self::EMPTY_ANCHOR_CRC32_BY_SLOT[(dominant_slot & ATTRACTOR_SLOT_ID_MASK) as usize];
        if zoff == 0 {
            if anchor_id == 0 {
                rec.checksum_crc32 = base_crc;
            } else if anchor_id < 256 {
                rec.checksum_crc32 = base_crc ^ CRC32_ANCHOR_AID_BYTE0_REF[anchor_id as usize];
            } else if anchor_id < 65536 {
                rec.checksum_crc32 = base_crc
                    ^ CRC32_ANCHOR_AID_BYTE0_REF[(anchor_id & 0xFF) as usize]
                    ^ CRC32_ANCHOR_AID_BYTE1_REF[((anchor_id >> 8) & 0xFF) as usize];
            } else {
                let c0 = crc32_slice16_indep_lo1_u32(anchor_id);
                rec.checksum_crc32 = base_crc ^ crc32_step_l4(crc32_step_l3(c0));
            }
        } else if ((zoff >> 32) == 0) & ((anchor_id >> 16) == 0) {
            let z_lo = zoff as u32;
            let z_crc = if (z_lo & 0xFFFF) == 0 {
                CRC32_ANCHOR_ZOFF_BYTE2_REF[((z_lo >> 16) & 0xFF) as usize]
                    ^ CRC32_ANCHOR_ZOFF_BYTE3_REF[(z_lo >> 24) as usize]
            } else if (z_lo & 0xFF) == 0 {
                CRC32_ANCHOR_ZOFF_BYTE1_REF[((z_lo >> 8) & 0xFF) as usize]
                    ^ CRC32_ANCHOR_ZOFF_BYTE2_REF[((z_lo >> 16) & 0xFF) as usize]
                    ^ CRC32_ANCHOR_ZOFF_BYTE3_REF[(z_lo >> 24) as usize]
            } else {
                crc32_anchor_zoff16_final_u32(z_lo)
            };
            rec.checksum_crc32 = base_crc
                ^ CRC32_ANCHOR_AID_BYTE0_REF[(anchor_id & 0xFF) as usize]
                ^ CRC32_ANCHOR_AID_BYTE1_REF[((anchor_id >> 8) & 0xFF) as usize]
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
        phase_words: &[u64; NUM_STIGMERGIC_GRANULES],
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
    pub fn stamp_crc32_with_words(&mut self, phase_words: &[u64; NUM_STIGMERGIC_GRANULES]) {
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
        self.mapped_section_offset48 = src.mapped_section_offset48;
        self.base_energy = src.base_energy;
        self.resultant_r_bar = src.resultant_r_bar;
        self.von_mises_kappa = src.von_mises_kappa;
        self.phase_gravity_mass = src.phase_gravity_mass;
        self.phase_clamp_rad = src.phase_clamp_rad;
        self.effective_inv_temp = src.effective_inv_temp;
        self.lineage_flags = src.lineage_flags;
        self.checksum_crc32 = src.checksum_crc32;
        self.dominant_slot = src.dominant_slot;
        self.action_mode_flags = src.action_mode_flags;
        self.active_swarm_mask = src.active_swarm_mask;
        self.root_cause_code = src.root_cause_code;
        self.reserved_align64 = src.reserved_align64;
        *self.phase_words_mut() = *src.phase_words_ref();
    }

    /// Verifies `magic`, ABI-compatible `version` (`MAJOR.MINOR` strict match, `file_patch <= runtime_patch`),
    /// and the IEEE 802.3 `checksum_crc32` integrity of this record.
    #[inline(always)]
    #[must_use]
    pub const fn verify_integrity(&self) -> bool {
        if (self.magic != LMNK_PHASE_ANCHOR_MAGIC)
            | !is_compatible_abi_version(self.version, LMNK_PHASE_ANCHOR_VERSION)
        {
            return false;
        }
        let pw = self.phase_words_ref();
        if self.version == LMNK_PHASE_ANCHOR_VERSION
            && self.sample_count == 0
            && ((self.anchor_id
                | self.base_energy.to_bits()
                | self.resultant_r_bar.to_bits()
                | self.von_mises_kappa.to_bits()
                | self.phase_gravity_mass.to_bits()
                | self.lineage_flags
                | self.reserved_align64)
                == 0)
            && (self.mapped_section_offset48 == 0)
            && (self.phase_clamp_rad.to_bits() == DEFAULT_PHASE_CLAMP_MAX_RAD.to_bits())
            && (self.effective_inv_temp.to_bits() == 0x3F80_0000u32)
            && ((self.dominant_slot & !ATTRACTOR_SLOT_ID_MASK) == 0)
            && (self.action_mode_flags == ACTION_PRESET_PASSIVE_SIDECAR)
            && (self.active_swarm_mask == 0xFF)
            && (self.root_cause_code == ANCHOR_ROOT_CAUSE_NOMINAL)
            && ((((pw[0] | pw[1]) | (pw[2] | pw[3])) | ((pw[4] | pw[5]) | (pw[6] | pw[7]))) == 0)
        {
            return self.checksum_crc32
                == Self::EMPTY_ANCHOR_CRC32_BY_SLOT[self.dominant_slot as usize];
        }
        self.checksum_crc32 == self.compute_payload_crc32_with_words(pw)
    }

    /// Internal register-resident M-step helper accepting the already-loaded `cur_w` in registers
    /// and an optional `precomputed_coh_opt` from the E-step to eliminate duplicate 64-head coherence evaluation.
    #[inline(always)]
    pub(crate) fn absorb_sample_words_with_cur_coh_and_uni_unstamped(
        &mut self,
        cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
        sample_w: Indivisible<Grounded, PhaseWordsRef64<'_>>,
        precomputed_coh_opt: Option<f32>,
        uni_flags_opt: Option<(bool, bool)>,
        sample_energy: f32,
        add_count: u32,
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, bool) {
        let sample_w = sample_w.into_inner().0;
        let clean_e = branchless_flush_subnormal_f32(branchless_relu_f32(sample_energy));
        if clean_e <= DEFAULT_ENTROPY_EPS || add_count == 0 {
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
            step_toroidal_centroid_words_cached_64(
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
            self.lineage_flags |= ANCHOR_LINEAGE_MATURE_BIT;
        }
        (next_w, grav_mass, words_changed)
    }
}

/// Return tuple from [`PhaseAnchorRecord::find_nearest_toroidal_centroid_with_cache_64`].
pub(crate) type NearestCentroidMatch = (
    usize,
    f32,
    f32,
    Option<usize>,
    usize,
    [u64; NUM_STIGMERGIC_GRANULES],
    bool,
    bool,
    Option<u8>,
);

/// Snapshot of [`PhaseAnchorRecord`] header fields captured before an absorption step for
/// incremental GF(2) CRC32 updates.
#[derive(Clone, Copy)]
pub(crate) struct AnchorHeaderSnapshot {
    pub crc: u32,
    pub anchor_id: u32,
    pub sample_count: u32,
    pub mapped_offset48: u64,
    pub base_energy_bits: u32,
    pub r_bar_bits: u32,
    pub kappa_bits: u32,
    pub gravity_mass_bits: u32,
    pub clamp_bits: u32,
    pub inv_temp_bits: u32,
    pub provenance_bits: u32,
    pub w7_lo: u32,
}

impl PhaseAnchorRecord {
    #[inline(always)]
    pub(crate) fn capture_header_snapshot(&self) -> AnchorHeaderSnapshot {
        AnchorHeaderSnapshot {
            crc: self.checksum_crc32,
            anchor_id: self.anchor_id,
            sample_count: self.sample_count,
            mapped_offset48: self.mapped_section_offset48,
            base_energy_bits: self.base_energy.to_bits(),
            r_bar_bits: self.resultant_r_bar.to_bits(),
            kappa_bits: self.von_mises_kappa.to_bits(),
            gravity_mass_bits: self.phase_gravity_mass.to_bits(),
            clamp_bits: self.phase_clamp_rad.to_bits(),
            inv_temp_bits: self.effective_inv_temp.to_bits(),
            provenance_bits: self.lineage_flags,
            w7_lo: self.pack_w7_lo(),
        }
    }

    #[inline(always)]
    pub(crate) fn stamp_or_update_absorption_crc32(
        &mut self,
        prev: AnchorHeaderSnapshot,
        cur_w: &[u64; NUM_STIGMERGIC_GRANULES],
        next_w: &[u64; NUM_STIGMERGIC_GRANULES],
        words_changed: bool,
    ) {
        if prev.crc != 0 {
            let delta = AnchorHeaderDelta {
                anchor_id: prev.anchor_id ^ self.anchor_id,
                sample_count: prev.sample_count ^ self.sample_count,
                mapped_offset48: prev.mapped_offset48 ^ self.mapped_section_offset48,
                base_energy_bits: prev.base_energy_bits ^ self.base_energy.to_bits(),
                r_bar_bits: prev.r_bar_bits ^ self.resultant_r_bar.to_bits(),
                kappa_bits: prev.kappa_bits ^ self.von_mises_kappa.to_bits(),
                gravity_mass_bits: prev.gravity_mass_bits ^ self.phase_gravity_mass.to_bits(),
                clamp_bits: prev.clamp_bits ^ self.phase_clamp_rad.to_bits(),
                inv_temp_bits: prev.inv_temp_bits ^ self.effective_inv_temp.to_bits(),
                provenance_bits: prev.provenance_bits ^ self.lineage_flags,
                w7_lo: prev.w7_lo ^ self.pack_w7_lo(),
            };
            if (!words_changed)
                & (delta.anchor_id == 0)
                & (delta.mapped_offset48 == 0)
                & ((delta.r_bar_bits | delta.kappa_bits) == 0)
                & ((delta.w7_lo >> 16) == 0)
            {
                self.checksum_crc32 = crc32_update_anchor_stationary_fields_delta(
                    prev.crc,
                    delta.sample_count,
                    delta.base_energy_bits,
                    delta.gravity_mass_bits,
                    delta.clamp_bits,
                    delta.inv_temp_bits,
                    delta.provenance_bits,
                    delta.w7_lo,
                );
            } else {
                self.checksum_crc32 =
                    crc32_update_anchor_absorption_delta(prev.crc, delta, cur_w, next_w);
            }
        } else {
            self.stamp_crc32_with_words(next_w);
        }
    }
}

impl PhaseAnchorRecord {
    /// Absorbs a [`Grounded`] 64-head sample point (`sample_phases`, `sample_energy`) into this
    /// centroid on $SO(2)^{64}$, updates $(\bar{R}_c, \kappa_c, M_{\mathrm{grav}})$, and stamps
    /// `checksum_crc32`.
    ///
    /// Statically excludes [`crate::liminal::Unmarked`], [`crate::liminal::Bottom`],
    /// [`crate::liminal::BoundaryViolation`], and unbridged
    /// [`crate::liminal::SpeculativeDiscovery`] from crystallizing into persistent
    /// attractor memory without monomorphizing the absorption body.
    #[inline]
    pub fn absorb_sample_point_64(
        &mut self,
        sample_phases: Indivisible<Grounded, PhaseProfileRef64<'_>>,
        sample_energy: f32,
    ) -> f32 {
        let prev = self.capture_header_snapshot();
        let cur_w = *self.phase_words_ref();
        let sample_w = load_q7_words_64(sample_phases.as_ref().0);
        let grounded_w = Scoped::interior(PhaseWordsRef64(&sample_w)).into_grounded();
        let (next_w, mass, words_changed) = self
            .absorb_sample_words_with_cur_coh_and_uni_unstamped(
                &cur_w,
                grounded_w,
                None,
                None,
                sample_energy,
                1,
            );
        self.stamp_or_update_absorption_crc32(prev, &cur_w, &next_w, words_changed);
        mass
    }

    /// Non-generic [`Indivisible<Grounded, _>`] closure constructor crystallizing a verified phase
    /// profile into a CRC32-stamped I-Frame [`PhaseAnchorRecord`].
    #[inline(always)]
    #[must_use]
    pub fn from_grounded_delta_64(
        anchor_id: u32,
        dominant_slot: u8,
        sample_count: u32,
        base_energy: f32,
        phase_profile: Indivisible<Grounded, PhaseProfile64>,
    ) -> Self {
        let mut rec = Self::new_empty(anchor_id, dominant_slot, 0);
        rec.phase_profile = phase_profile.into_inner().0;
        rec.base_energy = base_energy;
        rec.sample_count = sample_count;
        rec.stamp_crc32();
        rec
    }

    #[inline]
    #[must_use]
    pub(crate) fn find_nearest_toroidal_centroid_with_cache_64(
        centroids: &[Self],
        sample_words: &[u64; NUM_STIGMERGIC_GRANULES],
        word_cache_opt: Option<(
            &[[u64; NUM_STIGMERGIC_GRANULES]; MAX_REGISTER_CACHED_CENTROIDS],
            u16,
            u16,
        )>,
    ) -> NearestCentroidMatch {
        let k_len = centroids.len();
        if k_len == 0 {
            return (
                0,
                -1.0,
                0.0,
                None,
                0,
                [0u64; NUM_STIGMERGIC_GRANULES],
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
        let mut best_w = [0u64; NUM_STIGMERGIC_GRANULES];
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
            let is_active = (rec.sample_count > 0) & (mass > DEFAULT_ENTROPY_EPS);
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
    pub(crate) fn init_spawned_slot_header_in_place(
        &mut self,
        assigned_id: u32,
        dominant_slot: u8,
        zoff48: u64,
    ) {
        let slot = dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let was_active = (self.sample_count != 0) | (self.magic != LMNK_PHASE_ANCHOR_MAGIC);
        self.magic = LMNK_PHASE_ANCHOR_MAGIC;
        self.version = LMNK_PHASE_ANCHOR_VERSION;
        self.anchor_id = assigned_id;
        self.sample_count = 0;
        self.mapped_section_offset48 = zoff48 & TaggedPtr::ADDR_MASK;
        self.base_energy = 0.0;
        self.resultant_r_bar = 0.0;
        self.von_mises_kappa = 0.0;
        self.phase_gravity_mass = 0.0;
        self.phase_clamp_rad = DEFAULT_PHASE_CLAMP_MAX_RAD;
        self.effective_inv_temp = 1.0;
        self.lineage_flags = 0;
        self.checksum_crc32 = Self::EMPTY_ANCHOR_CRC32_BY_SLOT[slot as usize];
        self.dominant_slot = slot;
        self.action_mode_flags = ACTION_PRESET_PASSIVE_SIDECAR;
        self.active_swarm_mask = 0xFF;
        self.root_cause_code = ANCHOR_ROOT_CAUSE_NOMINAL;
        self.reserved_align64 = 0;
        if was_active {
            *self.phase_words_mut() = [0u64; NUM_STIGMERGIC_GRANULES];
        }
    }
}

impl PhaseAnchorRecord {
    /// Zero-copy deserialization of `count` [`PhaseAnchorRecord`]s from a 64-byte aligned
    /// `.lmnk` (mapped-storage) memory-mapped byte slice with header magic/version and CRC32 validation.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_ALIGNMENT`], [`LMNK_ERR_BUFFER_TOO_SMALL`], or
    /// [`LMNK_ERR_CONTAINER_CORRUPTED`] on validation failure.
    #[inline]
    pub fn slice_from_container_bytes_zero_copy(bytes: &[u8], count: usize) -> LmnkResult<&[Self]> {
        if count == 0 {
            return Ok(&[]);
        }
        let req_bytes = count
            .checked_mul(size_of::<Self>())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        if bytes.len() < req_bytes {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if !is_aligned_to(bytes, align_of::<Self>()) {
            return Err(LMNK_ERR_ALIGNMENT);
        }
        // SAFETY: `bytes` is verified 64-byte aligned and at least `count * 128` bytes long, and `Self: Pod`.
        let recs = unsafe { core::slice::from_raw_parts(bytes.as_ptr().cast::<Self>(), count) };
        let mut i = 0usize;
        while i < count {
            if !recs[i].verify_integrity() {
                return Err(LMNK_ERR_CONTAINER_CORRUPTED);
            }
            i += 1;
        }
        Ok(recs)
    }

    /// Serializes `records` into a `.lmnk` (mapped-storage) byte slice (`dst_bytes`), stamping each
    /// record's IEEE 802.3 CRC32 checksum when not already stamped.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < records.len() * 128`.
    #[inline]
    pub fn write_records_to_container_bytes(
        records: &[Self],
        dst_bytes: &mut [u8],
    ) -> LmnkResult<usize> {
        let total_bytes = records
            .len()
            .checked_mul(size_of::<Self>())
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        if dst_bytes.len() < total_bytes {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if total_bytes == 0 {
            return Ok(0);
        }
        #[cfg(target_endian = "little")]
        if is_aligned_to(dst_bytes, align_of::<u64>()) {
            // Write 64-bit `u64` words directly so subsequent 64-bit CRC32 & header validation loads
            // in `from_container_bytes_zero_copy` forward cleanly from the x86 Store Buffer with zero
            // 256-bit-to-64-bit Store-to-Load Forwarding (STLF) stalls.
            let expected_w0 =
                (LMNK_PHASE_ANCHOR_MAGIC as u64) | ((LMNK_PHASE_ANCHOR_VERSION as u64) << 32);
            let dst_base = dst_bytes.as_mut_ptr();
            let mut idx = 0usize;
            while idx < records.len() {
                let r = &records[idx];
                // SAFETY: `PhaseAnchorRecord` is a 128-byte `Pod` with `align >= 8`, so reading as `[u64; 16]` is valid.
                let src_w = unsafe { &*core::ptr::from_ref(r).cast::<[u64; 16]>() };
                let mut w0 = src_w[0];
                let mut w6 = src_w[6];
                let valid_hdr = w0 == expected_w0;
                if (!valid_hdr) | ((w6 >> 32) == 0) {
                    let crc = if valid_hdr {
                        r.compute_payload_crc32()
                    } else {
                        let mut rec = *r;
                        rec.magic = LMNK_PHASE_ANCHOR_MAGIC;
                        rec.version = LMNK_PHASE_ANCHOR_VERSION;
                        w0 = expected_w0;
                        rec.compute_payload_crc32()
                    };
                    w6 = (w6 & 0xFFFF_FFFFu64) | ((crc as u64) << 32);
                }
                let out_words = [
                    w0, src_w[1], src_w[2], src_w[3], src_w[4], src_w[5], w6, src_w[7], src_w[8],
                    src_w[9], src_w[10], src_w[11], src_w[12], src_w[13], src_w[14], src_w[15],
                ];
                // SAFETY: `dst_bytes.len() >= records.len() * 128` (`idx < records.len()`). Using
                // `write_unaligned` avoids constructing a `&mut [u64; 16]` reference when `dst_bytes`
                // comes from a 1-byte-aligned `[u8; N]` or `Vec<u8>` whose address happens to be 8-byte aligned.
                unsafe {
                    dst_base
                        .add(idx * size_of::<Self>())
                        .cast::<[u64; 16]>()
                        .write_unaligned(out_words);
                }
                idx += 1;
            }
            return Ok(total_bytes);
        }
        dst_bytes[..total_bytes].copy_from_slice(Self::slice_as_bytes(records));
        let mut idx = 0usize;
        while idx < records.len() {
            let r = &records[idx];
            let valid_hdr =
                (r.magic == LMNK_PHASE_ANCHOR_MAGIC) & (r.version == LMNK_PHASE_ANCHOR_VERSION);
            if (!valid_hdr) | (r.checksum_crc32 == 0) {
                let base_off = idx * size_of::<Self>();
                let crc = if valid_hdr {
                    r.compute_payload_crc32()
                } else {
                    let mut rec = *r;
                    rec.magic = LMNK_PHASE_ANCHOR_MAGIC;
                    rec.version = LMNK_PHASE_ANCHOR_VERSION;
                    dst_bytes[base_off..base_off + 4]
                        .copy_from_slice(&LMNK_PHASE_ANCHOR_MAGIC.to_le_bytes());
                    dst_bytes[base_off + 4..base_off + 8]
                        .copy_from_slice(&LMNK_PHASE_ANCHOR_VERSION.to_le_bytes());
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
    assert!(MAX_PHASE_ANCHORS == 64);
    assert!(size_of::<PhaseAnchorRecord>() == 128);
    assert!(align_of::<PhaseAnchorRecord>() == 64);
    assert!(offset_of!(PhaseAnchorRecord, magic) == 0);
    assert!(offset_of!(PhaseAnchorRecord, version) == 4);
    assert!(offset_of!(PhaseAnchorRecord, anchor_id) == 8);
    assert!(offset_of!(PhaseAnchorRecord, sample_count) == 12);
    assert!(offset_of!(PhaseAnchorRecord, mapped_section_offset48) == 16);
    assert!(offset_of!(PhaseAnchorRecord, base_energy) == 24);
    assert!(offset_of!(PhaseAnchorRecord, resultant_r_bar) == 28);
    assert!(offset_of!(PhaseAnchorRecord, von_mises_kappa) == 32);
    assert!(offset_of!(PhaseAnchorRecord, phase_gravity_mass) == 36);
    assert!(offset_of!(PhaseAnchorRecord, phase_clamp_rad) == 40);
    assert!(offset_of!(PhaseAnchorRecord, effective_inv_temp) == 44);
    assert!(offset_of!(PhaseAnchorRecord, lineage_flags) == 48);
    assert!(offset_of!(PhaseAnchorRecord, checksum_crc32) == 52);
    assert!(offset_of!(PhaseAnchorRecord, dominant_slot) == 56);
    assert!(offset_of!(PhaseAnchorRecord, root_cause_code) == 59);
    assert!(offset_of!(PhaseAnchorRecord, reserved_align64) == 60);
    assert!(offset_of!(PhaseAnchorRecord, phase_profile) == 64);
};

impl RawPhaseProjectable for PhaseAnchorRecord {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.phase_profile
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *self.phase_words_ref()
    }
}

/// Executes **Persistent Topological Mode Quantization** in row-major order,
/// returning `Ok((0.0, 0.0))` with zeroed `profile_out` if total trail weight is zero.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `coherent_trails` is empty or `min_coherence_r_bar` is non-finite.
#[inline]
pub fn quantize_persistent_topological_mode(
    coherent_trails: &[PointerTrail],
    min_coherence_r_bar: f32,
    profile_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
) -> LmnkResult<(f32, f32)> {
    if coherent_trails.is_empty() || !branchless_is_finite_f32(min_coherence_r_bar) {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }

    let mut candidate = [0i8; MAX_ATTENTION_HEADS];
    let candidate_words: [u64; 8];
    if coherent_trails.len() == 1 {
        let trail = &coherent_trails[0];
        let d_bounded = branchless_clamp_f32(trail.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
        let f_bounded =
            branchless_clamp_f32(trail.attractor_pull_force, 1.0, MAX_ATTRACTOR_PULL_FORCE);
        let w = branchless_flush_subnormal_f32(d_bounded * f_bounded);
        if w <= 1.0e-10 {
            *profile_out = [0i8; MAX_ATTENTION_HEADS];
            return Ok((0.0, 0.0));
        }
        candidate = trail.accumulated_phase_delta;
        candidate_words = *trail_phase_words_ref(trail);
    } else {
        let mut sum_s = [[0.0_f32; 8]; 8];
        let mut sum_c = [[0.0_f32; 8]; 8];
        let mut word_s = [0.0_f32; 8];
        let mut word_c = [0.0_f32; 8];
        let mut total_w = 0.0_f32;
        let mut all_byte_uniform = true;
        let mut all_word_uniform = true;
        let mut uniform_s = 0.0_f32;
        let mut uniform_c = 0.0_f32;

        for trail in coherent_trails {
            let d_bounded =
                branchless_clamp_f32(trail.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
            let f_bounded =
                branchless_clamp_f32(trail.attractor_pull_force, 1.0, MAX_ATTRACTOR_PULL_FORCE);
            let w = branchless_flush_subnormal_f32(d_bounded * f_bounded);
            if w > 0.0 {
                total_w += w;
                let tw = *trail_phase_words_ref(trail);
                let w0 = tw[0];
                if is_word_array_uniform_64(&tw) {
                    if w0 == w0.rotate_right(8) {
                        let (s, c) = branchless_sincos_q7(w0 as PhaseQ7);
                        uniform_s += w * s;
                        uniform_c += w * c;
                    } else {
                        all_byte_uniform = false;
                        let b = w0.to_le_bytes();
                        let mut lane = 0usize;
                        while lane < 8 {
                            let (s, c) = branchless_sincos_q7(b[lane] as PhaseQ7);
                            word_s[lane] += w * s;
                            word_c[lane] += w * c;
                            lane += 1;
                        }
                    }
                } else {
                    all_byte_uniform = false;
                    all_word_uniform = false;
                    let mut cell = 0usize;
                    while cell < 8 {
                        let b = tw[cell].to_le_bytes();
                        let mut lane = 0usize;
                        while lane < 8 {
                            let (s, c) = branchless_sincos_q7(b[lane] as PhaseQ7);
                            sum_s[cell][lane] += w * s;
                            sum_c[cell][lane] += w * c;
                            lane += 1;
                        }
                        cell += 1;
                    }
                }
            }
        }
        if total_w <= 1.0e-10 {
            *profile_out = [0i8; MAX_ATTENTION_HEADS];
            return Ok((0.0, 0.0));
        }
        if all_byte_uniform {
            let mag_sq = uniform_s * uniform_s + uniform_c * uniform_c;
            let has_mag = mag_sq > 1.0e-20;
            let q7 = branchless_select_q7(
                has_mag,
                branchless_atan2_to_q7_fast(uniform_s, uniform_c),
                0,
            );
            let r_bar = branchless_select_f32(has_mag, 1.0, 0.0);
            let kappa_c = branchless_von_mises_kappa(r_bar);
            let keep = r_bar >= branchless_clamp_f32(min_coherence_r_bar, 0.0, 1.0);
            *profile_out = [branchless_select_q7(keep, q7, 0); MAX_ATTENTION_HEADS];
            return Ok((r_bar, kappa_c));
        } else if all_word_uniform {
            let mut s_cell = [0.0_f32; 8];
            let mut c_cell = [0.0_f32; 8];
            let mut lane = 0usize;
            while lane < 8 {
                s_cell[lane] = word_s[lane] + uniform_s;
                c_cell[lane] = word_c[lane] + uniform_c;
                lane += 1;
            }
            let word0 = atan2_to_q7_word_x8_simd(&s_cell, &c_cell);
            candidate_words = [word0; 8];
            store_q7_words_64(&mut candidate, &candidate_words);
        } else {
            let has_word_or_uni = (uniform_s != 0.0)
                | (uniform_c != 0.0)
                | (word_s != [0.0_f32; 8])
                | (word_c != [0.0_f32; 8]);
            if has_word_or_uni {
                let mut cell = 0usize;
                while cell < 8 {
                    let mut lane = 0usize;
                    while lane < 8 {
                        sum_s[cell][lane] += word_s[lane] + uniform_s;
                        sum_c[cell][lane] += word_c[lane] + uniform_c;
                        lane += 1;
                    }
                    cell += 1;
                }
            }
            let mut cw = [0u64; 8];
            let mut cell = 0usize;
            while cell < 8 {
                cw[cell] = atan2_to_q7_word_x8_simd(&sum_s[cell], &sum_c[cell]);
                cell += 1;
            }
            candidate_words = cw;
            store_q7_words_64(&mut candidate, &candidate_words);
        }
    }

    let r_bar = circular_r_bar_from_words_64(&candidate_words);
    let kappa_c = branchless_von_mises_kappa(r_bar);
    let keep = r_bar >= branchless_clamp_f32(min_coherence_r_bar, 0.0, 1.0);
    if keep {
        *profile_out = candidate;
    } else {
        *profile_out = [0i8; MAX_ATTENTION_HEADS];
    }
    Ok((r_bar, kappa_c))
}

#[cfg(test)]
mod grounded_gate_tests {
    use super::*;
    use crate::liminal::{
        BridgedDiscovery, Scoped, ScopedCut, SpeculativeDiscovery, VerifiedAligned,
    };

    #[test]
    fn test_grounded_closure_gate_allows_verified_and_bridged_blocks_unbridged() {
        let phases = [18_i8; MAX_ATTENTION_HEADS];

        let mut direct = PhaseAnchorRecord::new_empty(7, ATTRACTOR_SLOT_PERSONA, 0);
        let direct_grounded = Scoped::interior(PhaseProfileRef64(&phases)).into_grounded();
        assert_eq!(
            direct_grounded.domain_id(),
            "stigmergy.phase_profile_ref_64"
        );
        let direct_mass = direct.absorb_sample_point_64(direct_grounded, 1.5);

        let mut aligned_rec = PhaseAnchorRecord::new_empty(7, ATTRACTOR_SLOT_PERSONA, 0);
        let verified: VerifiedAligned<PhaseProfileRef64<'_>> =
            match Scoped::from_boundary(PhaseProfileRef64(&phases)).cut_by_lacuna_q7(true, 1, 4) {
                ScopedCut::SelfRef(s) => s,
                _ => panic!("expected verified aligned cut"),
            };
        let aligned_mass = aligned_rec.absorb_sample_point_64(verified.into_grounded(), 1.5);
        assert_eq!(aligned_mass.to_bits(), direct_mass.to_bits());
        assert_eq!(aligned_rec, direct);

        let spec: SpeculativeDiscovery<PhaseProfileRef64<'_>> =
            match Scoped::from_delta(PhaseProfileRef64(&phases)).cut_by_lacuna_q7(true, 24, 4) {
                ScopedCut::Exterior(e) => e,
                _ => panic!("expected speculative discovery cut"),
            };
        let blocked = spec.bridge_to_shared(0).unwrap_err();
        let bridged: BridgedDiscovery<PhaseProfileRef64<'_>> = blocked.bridge_to_shared(1).unwrap();

        let mut bridged_rec = PhaseAnchorRecord::new_empty(7, ATTRACTOR_SLOT_PERSONA, 0);
        let bridged_mass = bridged_rec.absorb_sample_point_64(bridged.into_grounded(), 1.5);
        assert_eq!(bridged_mass.to_bits(), direct_mass.to_bits());
        assert_eq!(bridged_rec, direct);
    }

    #[test]
    fn test_quantize_persistent_topological_mode_single_and_multi_trail() {
        let mut out = [0i8; MAX_ATTENTION_HEADS];
        assert_eq!(
            quantize_persistent_topological_mode(&[], 0.5, &mut out),
            Err(LMNK_ERR_INVALID_ARGUMENT)
        );
        assert_eq!(
            quantize_persistent_topological_mode(&[PointerTrail::default()], f32::NAN, &mut out),
            Err(LMNK_ERR_INVALID_ARGUMENT)
        );

        let t0 = PointerTrail {
            stigmergic_density: 1.5,
            attractor_pull_force: 1.0,
            accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let (r0, k0) = quantize_persistent_topological_mode(&[t0], 0.8, &mut out).unwrap();
        assert!(r0 > 0.99 && k0 > 10.0);
        assert_eq!(out, [28i8; MAX_ATTENTION_HEADS]);

        let t1 = PointerTrail {
            stigmergic_density: 2.0,
            attractor_pull_force: 1.0,
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut t2 = PointerTrail {
            stigmergic_density: 2.5,
            attractor_pull_force: 1.0,
            accumulated_phase_delta: [30i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        t2.accumulated_phase_delta[0] = 34;
        let (r_m, _k_m) =
            quantize_persistent_topological_mode(&[t0, t1, t2], 0.8, &mut out).unwrap();
        assert!(r_m > 0.95);
        assert!((28..=34).contains(&out[0]));
        assert!((28..=32).contains(&out[1]));
    }
}
