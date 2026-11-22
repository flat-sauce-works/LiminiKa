// name: src/core/src/stigmergy/anchor/crc32_delta.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 128-byte `"LNKR"` [`super::PhaseAnchorRecord`] GF(2) differential IEEE 802.3 CRC32
//! lookup tables and O(1) incremental checksum update helpers.

use crate::foundation::lut_crc32::{
    crc32_slice16_dep_lo0, crc32_slice16_indep_hi0_u32, crc32_slice16_indep_hi1_u32,
    crc32_slice16_indep_lo1_u32, crc32_step_block64_words, crc32_step_l2, crc32_step_l3,
    crc32_step_l4, crc32_step_l5, crc32_step_l6, crc32_step_l8, AlignedLut256, AlignedLut4x256,
    CRC32_IEEE_SLICE16_LUT, VON_MISES_KAPPA_AT_MAX_R_BITS,
};
use crate::foundation::DEFAULT_PHASE_CLAMP_MAX_RAD;

use super::{LMNK_PHASE_ANCHOR_MAGIC, LMNK_PHASE_ANCHOR_VERSION};

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
/// (`offset 12..16` inside a 128-byte `PhaseAnchorRecord`, shifted by 112 bytes),
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
/// (`offset 8..12` inside a 128-byte `PhaseAnchorRecord`, shifted by 116 bytes via $L^7$),
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
/// (`offset 9` inside a 128-byte `PhaseAnchorRecord`, shifted by 116 bytes via $L^7$),
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
/// of `mapped_section_offset48` inside a 128-byte `PhaseAnchorRecord`, pre-shifted by
/// 96 bytes via $L^6$ so 4KB-aligned mapped-storage section offsets fold in 1 lookup stage (`0` serial $L^6$ stages).
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE1_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l6(CRC32_IEEE_SLICE16_LUT.0[14][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `mapped_section_offset48` byte-1 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE1_REF: &[u32; 256] = &CRC32_CRYSTAL_ZOFF_BYTE1_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-2 (`(zoff >> 16) & 0xFF`, `offset 18`)
/// of `mapped_section_offset48` inside a 128-byte `PhaseAnchorRecord`, pre-shifted by
/// 96 bytes via $L^6$ so 64KB-aligned mapped-storage section offsets fold in 1 lookup stage (`0` serial $L^6$ stages).
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE2_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l6(CRC32_IEEE_SLICE16_LUT.0[13][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `mapped_section_offset48` byte-2 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE2_REF: &[u32; 256] = &CRC32_CRYSTAL_ZOFF_BYTE2_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-3 (`(zoff >> 24) & 0xFF`, `offset 19`)
/// of `mapped_section_offset48` inside a 128-byte `PhaseAnchorRecord`, pre-shifted by
/// 96 bytes via $L^6$ so 64KB-aligned mapped-storage section offsets fold in 1 lookup stage (`0` serial $L^6$ stages).
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE3_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l6(CRC32_IEEE_SLICE16_LUT.0[12][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `mapped_section_offset48` byte-3 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_ZOFF_BYTE3_REF: &[u32; 256] = &CRC32_CRYSTAL_ZOFF_BYTE3_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-0 `lineage_flags` values
/// (`offset 48` inside a 128-byte `PhaseAnchorRecord`, shifted by 64 bytes via $L^5$),
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

/// Promoted `.rodata` reference to the 1KB `lineage_flags` byte-0 differential CRC32 table.
pub(crate) const CRC32_CRYSTAL_PROV_BYTE0_REF: &[u32; 256] = &CRC32_CRYSTAL_PROV_BYTE0_LUT.0;

/// 64B-aligned 1KB GF(2) differential CRC32 table for byte-1 `lineage_flags` values
/// (`offset 49` inside a 128-byte `PhaseAnchorRecord`, shifted by 64 bytes via $L^5$),
/// evaluating 16-bit provenance bitmasks (`lineage_flags <= 0xFFFF`) in 2 parallel table lookups.
pub(crate) const CRC32_CRYSTAL_PROV_BYTE1_LUT: AlignedLut256<u32> = {
    let mut lut = [0u32; 256];
    let mut d = 0usize;
    while d < 256 {
        lut[d] = crc32_step_l4(CRC32_IEEE_SLICE16_LUT.0[14][d]);
        d += 1;
    }
    AlignedLut256(lut)
};

/// Promoted `.rodata` reference to the 1KB `lineage_flags` byte-1 differential CRC32 table.
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
/// pre-shifted to the end of the 128-byte `PhaseAnchorRecord` via $L^6(\text{indep\_lo1}(x))$.
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
/// pre-shifted to the end of the 128-byte `PhaseAnchorRecord` via $L^5(\text{indep\_hi0}(x))$.
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
/// (`mapped_section_offset48 as u32`) using the L1D-resident `CRC32_IEEE_SLICE16_REF` and `CRC32_IEEE_L6_REF` tables.
#[inline(always)]
pub(crate) const fn crc32_crystal_zoff16_final_u32(zoff_lo: u32) -> u32 {
    if zoff_lo == 0 {
        0
    } else {
        crc32_step_l6(crc32_slice16_dep_lo0(zoff_lo))
    }
}

/// Evaluates the exact 128B final GF(2) CRC32 contribution of a 32-bit word at `offset 40..44`
/// (`phase_clamp_rad`) using the L1D-resident `CRC32_IEEE_SLICE16_REF` and `CRC32_IEEE_L5_REF` tables.
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

/// Evaluates the accumulator-independent 128-byte stamped crystal contribution
/// $L^4(L_{12}(\text{record\_crc32})) \oplus (\neg\text{record\_crc32}) \oplus L^8(\text{0xFFFF\_FFFF})$
/// in 4 parallel table lookups (`0` branches).
#[inline(always)]
pub(crate) const fn crc32_crystal_stamp_contrib_128b(record_crc32: u32) -> u32 {
    let lut = CRC32_IEEE_CRYSTAL_STAMP_REF;
    (lut[0][(record_crc32 & 0xFF) as usize] ^ lut[1][((record_crc32 >> 8) & 0xFF) as usize])
        ^ (lut[2][((record_crc32 >> 16) & 0xFF) as usize] ^ lut[3][(record_crc32 >> 24) as usize])
}

/// Advances an uninverted IEEE 802.3 CRC32 accumulator (`crc`) across an entire 128-byte stamped
/// `PhaseAnchorRecord` using GF(2) linearity ($L^8(\text{crc}) \oplus \text{stamp}(\text{record\_crc32})$)
/// in a single 4-lookup stage (`0` branches).
#[inline(always)]
pub(crate) const fn crc32_fold_stamped_crystal_128b(crc: u32, record_crc32: u32) -> u32 {
    crc32_step_l8(crc) ^ crc32_crystal_stamp_contrib_128b(record_crc32)
}

/// Updates a stamped `PhaseAnchorRecord`'s IEEE 802.3 `checksum_crc32` in $\mathcal{O}(1)$
/// via GF(2) linearity when `phase_profile` (`offset 64..128`), `anchor_id`, `mapped_section_offset48`,
/// `resultant_r_bar`, and `von_mises_kappa` remained stationary while `sample_count` (`offset 12..16`),
/// `base_energy` (`offset 24..28`), `phase_gravity_mass` (`offset 36..40`), `phase_clamp_rad`
/// (`offset 40..44`), `effective_inv_temp` (`offset 44..48`), `lineage_flags` (`offset 48..52`),
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
        direct_xor ^=
            slot_contrib ^ CRC32_CRYSTAL_ACTION_BYTE57_REF[((delta_w7_lo >> 8) & 0xFF) as usize];
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

/// Updates a stamped `PhaseAnchorRecord`'s IEEE 802.3 `checksum_crc32` in $\mathcal{O}(1)$
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
/// `w0 = (LMNK_PHASE_ANCHOR_MAGIC as u64) | ((LMNK_PHASE_ANCHOR_VERSION as u64) << 32)`
/// XORed with the initial IEEE 802.3 seed `0xFFFF_FFFF`, eliminating 8 table lookups on every
/// `PhaseAnchorRecord` checksum evaluation.
pub(crate) const CRYSTAL_CRC32_W0_SEED_CONTRIB: u32 =
    crc32_slice16_dep_lo0(LMNK_PHASE_ANCHOR_MAGIC ^ 0xFFFF_FFFFu32)
        ^ crc32_slice16_indep_hi0_u32(LMNK_PHASE_ANCHOR_VERSION);

/// Precomputed Slice-by-16 accumulator contribution for the default `w5` (`phase_clamp_rad = 0.24`,
/// `effective_inv_temp = 1.0`) field pair inside a `PhaseAnchorRecord`, eliminating 8 table lookups.
pub(crate) const CRYSTAL_CRC32_DEFAULT_W5_CONTRIB: u32 =
    crc32_slice16_indep_lo1_u32(DEFAULT_PHASE_CLAMP_MAX_RAD.to_bits())
        ^ crc32_slice16_indep_hi1_u32(0x3F80_0000u32);

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^7(\text{CRYSTAL\_CRC32\_W0\_SEED\_CONTRIB})$
/// for the invariant `(LMNK_PHASE_ANCHOR_MAGIC, LMNK_PHASE_ANCHOR_VERSION)` header and initial seed.
pub(crate) const CRYSTAL_CRC32_W0_FINAL_CONTRIB: u32 =
    crc32_step_l4(crc32_step_l3(CRYSTAL_CRC32_W0_SEED_CONTRIB));

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^5(\text{CRYSTAL\_CRC32\_DEFAULT\_W5\_CONTRIB})$
/// for the default `w5` (`phase_clamp_rad = 0.24`, `effective_inv_temp = 1.0`) field pair.
pub(crate) const CRYSTAL_CRC32_DEFAULT_W5_FINAL_CONTRIB: u32 =
    crc32_step_l5(CRYSTAL_CRC32_DEFAULT_W5_CONTRIB);

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^6(\text{indep\_hi1}(\text{0x3F80\_0000}))$
/// for `resultant_r_bar == 1.0` (`offset 28..32`), eliminating 4 table lookups on coherent centroids.
pub(crate) const CRYSTAL_CRC32_RBAR_ONE_FINAL_CONTRIB: u32 =
    crc32_step_l6(crc32_slice16_indep_hi1_u32(0x3F80_0000u32));

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^5(\text{dep\_lo0}(\text{VON\_MISES\_KAPPA\_AT\_MAX\_R\_BITS}))$
/// for `von_mises_kappa == VON_MISES_KAPPA_AT_MAX_R` (`offset 32..36`), eliminating 4 table lookups
/// on every byte-uniform or maximally coherent centroid ($\bar{R}_c \ge 0.995$).
pub(crate) const CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB: u32 =
    crc32_step_l5(crc32_slice16_dep_lo0(VON_MISES_KAPPA_AT_MAX_R_BITS));

/// Pre-shifted final 128B GF(2) CRC32 contribution $L^5(\text{dep\_lo0}(\text{0x4248\_0000}))$ for
/// `von_mises_kappa == 50.0` (`offset 32..36`), eliminating 4 table lookups.
pub(crate) const CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB: u32 =
    crc32_step_l5(crc32_slice16_dep_lo0(0x4248_0000u32));

/// Evaluates the 64-byte header (`offset 0..64`) GF(2) differential IEEE 802.3 CRC32 state
/// `(final_128b_xor, c3_at_offset_64)` using strictly the L1D-resident `CRC32_IEEE_SLICE16_REF`
/// table and scalar constants (`0` separate 4KB `FINAL_LUT` table accesses), allowing `c3` to fold
/// directly into `phase_profile` word 0 (`dw[0]`) with zero `crc32_step_l4` overhead.
#[inline]
pub(crate) const fn crc32_crystal_header_delta_c3(
    delta_anchor_id: u32,
    delta_sample_count: u32,
    delta_mapped_offset48: u64,
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

    let z_lo = (delta_mapped_offset48 as u32) ^ c0;
    let z_hi = (delta_mapped_offset48 >> 32) as u32;
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

/// Updates a stamped `PhaseAnchorRecord`'s IEEE 802.3 `checksum_crc32` via GF(2)
/// linearity across any absorption step, folding `c3` directly into `dw[0]` via
/// `crc32_step_block64_words(c3, &dw)` when `phase_profile` changed (`0` `crc32_step_l4` lookups).
#[inline(always)]
pub(crate) const fn crc32_update_crystal_absorption_delta(
    old_crc32: u32,
    delta_anchor_id: u32,
    delta_sample_count: u32,
    delta_mapped_offset48: u64,
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
        delta_mapped_offset48,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crystal_kappa_crc32_constants_match_gf2_shift() {
        assert_eq!(
            CRYSTAL_CRC32_KAPPA_AT_MAX_R_FINAL_CONTRIB,
            crc32_step_l5(crc32_slice16_dep_lo0(VON_MISES_KAPPA_AT_MAX_R_BITS))
        );
        assert_eq!(
            CRYSTAL_CRC32_KAPPA_MAX_FINAL_CONTRIB,
            crc32_step_l5(crc32_slice16_dep_lo0(0x4248_0000u32))
        );
    }
}
