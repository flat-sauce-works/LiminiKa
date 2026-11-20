// name: src/core/src/foundation/projection.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Branchless raw byte and `f32` embedding phase projections ([`RawPhaseProjectable`]).

use crate::foundation::aligned::AlignedSlice;
use crate::foundation::lut_crc32::{SWAR_LOW7_MASK_64, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64};
use crate::foundation::pod::Pod;
use crate::foundation::scalar_math::branchless_atan2_to_q7_fast;
use crate::foundation::simd_stats::{
    atan2_interleaved_pairs_to_q7_word_x8_simd, atan2_to_q7_word_x8_simd,
};
use crate::foundation::swar_simd::{branchless_splitmix64, load_q7_words_64, words_to_q7_array_64};
use crate::liminal::{
    branchless_select_usize, gcso_q7_t, Liminal, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
};

#[inline(always)]
#[must_use]
pub(crate) const fn swar_clamp_q7_min_to_neg127_x8(w: u64) -> u64 {
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

/// Trait for raw payloads that can be projected onto the 64-head $SO(2)^{64}$ Q7 phase manifold.
pub trait RawPhaseProjectable: Liminal {
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
