// name: src/core/src/stigmergy/ledger.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical Layer 2 [`LedgerSlot`] POD descriptor, 256-bit `PagedBitmask`
//! algebra, 64-byte [`ContrastivePhaseBloomFilter`], and [`FastPathLedger`] capability trait.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use crate::foundation::{
    PAGED_BITMASK_BITS, PagedBitmask, Pod, SWAR_EVEN_BYTES_64, SWAR_LOW7_MASK_64,
    SWAR_SPLAT_BYTE_64, SWAR_SUM_U16_64, TaggedPtr, branchless_clamp_f32,
    branchless_popcount_rsqrt_u32, branchless_relu_f32, branchless_splitmix64, branchless_tanh_f32,
    is_word_array_uniform_64, load_q7_words_64, store_q7_words_64, swar_q7_add_sat_x8,
    swar_q7_neg_sat_x8, swar_q7_scale_le128_with_bias_x8, swar_u8_add_sat_x8,
    swar_u8_halve_decay_x8, swar_u8_max_x8, swar_u8_nonzero_count_x8, swar_u8_nonzero_msb_mask,
    swar_u8_sub_sat_x8, swar_words_nonzero_64,
};
use crate::liminal::{
    LMNK_ERR_INVALID_ARGUMENT, LMNK_ERR_INVALID_STATE, LmnkResult, MAX_ATTENTION_HEADS,
    NUM_STIGMERGIC_GRANULES, PhaseQ7, RuntimeConfig, branchless_select_u8, branchless_select_u32,
    branchless_select_u64,
};
use crate::stigmergy::{
    DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS, DEFAULT_BLOOM_MAX_MASS_THRESHOLD, FAST_PATH_SHORTCUT_SLOTS,
    LEDGER_HISTORY_LEN,
};

/// Dynamic Adaptive Extension Scratchpad (FastPathLedger) slot layout (64 bytes, 1 cache line, Section 5.4.F).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LedgerSlot {
    /// Active extension execution mode (`0` = Scratchpad, `1` = Dynamic Plugin, `2` = Shared IPC Buffer).
    pub mode: u32,
    /// Ring buffer head write index.
    pub telemetry_ring_head: u16,
    /// Ring buffer tail read index.
    pub telemetry_ring_tail: u16,
    /// Total fast-path lookup cache hits recorded.
    pub cache_hit_count: u32,
    /// Bitmask flags configuring autonomous auto-tuning behavior.
    pub auto_tune_flags: u32,
    /// 64-bit bitmask specifying fast-path bypass criteria.
    pub fast_path_bypass_mask: u64,
    /// Four cached 64-bit tagged jump pointers for $\mathcal{O}(1)$ accelerated execution.
    pub fast_path_shortcuts: [u64; 4],
    /// Circular 8-byte ledger array storing recent Nano/Micro telemetry codes.
    pub telemetry_mini_ledger: [u8; 8],
}

// SAFETY: `LedgerSlot` is `#[repr(C, align(64))]` with primitive fields only and zero padding.
unsafe impl Pod for LedgerSlot {}

impl LedgerSlot {
    /// Executes strictly branchless $\mathcal{O}(1)$ fast-path shortcut lookup directly in-place
    /// on `self` without copying the 64-byte `LedgerSlot` onto the stack.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn lookup_fast_path_branchless(&self, input_key: u64) -> (bool, u64) {
        let mask_hit = ((self.fast_path_bypass_mask & input_key) != 0) & (self.mode == 0);
        let idx = ((input_key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
        let candidate = self.fast_path_shortcuts[idx & (FAST_PATH_SHORTCUT_SLOTS - 1)];
        let fallback = self.fast_path_shortcuts[0];
        let resolved = branchless_select_u64(candidate != 0, candidate, fallback);
        let valid_hit = mask_hit & (resolved != 0);
        (valid_hit, branchless_select_u64(valid_hit, resolved, 0))
    }

    /// Branchlessly resolves `input_ptr` against `self.fast_path_shortcuts` in-place (`0` stack
    /// copy, `0` 256-bit-to-64-bit Store-to-Load Forwarding pipeline stall).
    #[inline(always)]
    pub(crate) fn resolve_or_pass_in_place(&mut self, input_ptr: TaggedPtr) -> (bool, TaggedPtr) {
        let (hit, shortcut_raw) = self.lookup_fast_path_branchless(input_ptr.raw());
        self.cache_hit_count = self.cache_hit_count.wrapping_add(u32::from(hit));
        let eff_raw = branchless_select_u64(hit, shortcut_raw, input_ptr.raw());
        (hit, TaggedPtr(eff_raw))
    }

    /// Pushes a profiling metric byte code (`LEDGER_METRIC_*`) into `self.telemetry_mini_ledger` in-place.
    #[inline(always)]
    pub(crate) fn push_telemetry_metric(&mut self, metric_code: u8) {
        let head = (self.telemetry_ring_head as usize) & (LEDGER_HISTORY_LEN - 1);
        self.telemetry_mini_ledger[head] = metric_code;
        self.telemetry_ring_head = self.telemetry_ring_head.wrapping_add(1);
    }
}

// ===================================================================
// 2. 256-Bit PagedBlock Bitmask Algebra (`PagedBitmask`)
// ===================================================================

impl PagedBitmask {
    /// Returns `true` if all 256 bits across the four 64-bit words are zero.
    #[inline(always)]
    pub fn is_all_zero(&self) -> bool {
        ((self.bits[0] | self.bits[1]) | (self.bits[2] | self.bits[3])) == 0
    }

    /// Returns the total Hamming weight (`popcount`) across all 256 bits in $\mathcal{O}(1)$.
    #[inline(always)]
    pub fn popcount(&self) -> u32 {
        (self.bits[0].count_ones() + self.bits[1].count_ones())
            + (self.bits[2].count_ones() + self.bits[3].count_ones())
    }

    /// Sets bit `bit_idx` ($0 \le \text{bit\_idx} < 256$) branchlessly. Out-of-range indices safely no-op.
    #[inline(always)]
    pub fn set_bit_branchless(&mut self, bit_idx: u32) {
        let in_bounds_mask = ((bit_idx < PAGED_BITMASK_BITS) as u64).wrapping_neg();
        let bit_val = (1u64 << (bit_idx & 63)) & in_bounds_mask;
        self.bits[((bit_idx >> 6) as usize) & 3] |= bit_val;
    }

    /// Clears bit `bit_idx` ($0 \le \text{bit\_idx} < 256$) branchlessly. Out-of-range indices safely no-op.
    #[inline(always)]
    pub fn clear_bit_branchless(&mut self, bit_idx: u32) {
        let in_bounds_mask = ((bit_idx < PAGED_BITMASK_BITS) as u64).wrapping_neg();
        let clear_mask = !((1u64 << (bit_idx & 63)) & in_bounds_mask);
        self.bits[((bit_idx >> 6) as usize) & 3] &= clear_mask;
    }

    /// Tests whether bit `bit_idx` ($0 \le \text{bit\_idx} < 256$) is set branchlessly.
    #[inline(always)]
    pub fn test_bit_branchless(&self, bit_idx: u32) -> bool {
        let in_bounds = bit_idx < PAGED_BITMASK_BITS;
        let word = self.bits[((bit_idx >> 6) as usize) & 3];
        in_bounds & (((word >> (bit_idx & 63)) & 1) != 0)
    }

    /// Returns the lowest set bit index (`0..=255`), or `256` (`PAGED_BITMASK_BITS`) if all bits are zero.
    #[inline(always)]
    pub fn first_set_bit(&self) -> u32 {
        let z0 = self.bits[0] == 0;
        let z01 = z0 & (self.bits[1] == 0);
        let z012 = z01 & (self.bits[2] == 0);
        let all_zero = z012 & (self.bits[3] == 0);
        let active_w = branchless_select_u64(
            z0,
            branchless_select_u64(
                z01,
                branchless_select_u64(z012, self.bits[3], self.bits[2]),
                self.bits[1],
            ),
            self.bits[0],
        );
        let base_tz = ((z0 as u32) + (z01 as u32) + (z012 as u32)) << 6;
        branchless_select_u32(
            all_zero,
            PAGED_BITMASK_BITS,
            base_tz + (active_w | (1u64 << 63)).trailing_zeros(),
        )
    }

    /// Returns the highest set bit index (`0..=255`), or `256` (`PAGED_BITMASK_BITS`) if all bits are zero.
    #[inline(always)]
    pub fn last_set_bit(&self) -> u32 {
        let e3 = self.bits[3] == 0;
        let e32 = e3 & (self.bits[2] == 0);
        let e321 = e32 & (self.bits[1] == 0);
        let all_zero = e321 & (self.bits[0] == 0);
        let active_w = branchless_select_u64(
            e3,
            branchless_select_u64(
                e32,
                branchless_select_u64(e321, self.bits[0], self.bits[1]),
                self.bits[2],
            ),
            self.bits[3],
        );
        let base_lz = ((e3 as u32) + (e32 as u32) + (e321 as u32)) << 6;
        branchless_select_u32(
            all_zero,
            PAGED_BITMASK_BITS,
            255u32.wrapping_sub(base_lz + (active_w | 1).leading_zeros()),
        )
    }

    /// Compresses the 256-bit PagedBlock bitmask into an **8-bit Micro-Swarm Activity Mask**.
    #[inline(always)]
    pub fn subswarm_activity_mask_8(&self) -> u8 {
        (((self.bits[0] & 0xFFFF_FFFF) != 0) as u8)
            | ((((self.bits[0] >> 32) != 0) as u8) << 1)
            | ((((self.bits[1] & 0xFFFF_FFFF) != 0) as u8) << 2)
            | ((((self.bits[1] >> 32) != 0) as u8) << 3)
            | ((((self.bits[2] & 0xFFFF_FFFF) != 0) as u8) << 4)
            | ((((self.bits[2] >> 32) != 0) as u8) << 5)
            | ((((self.bits[3] & 0xFFFF_FFFF) != 0) as u8) << 6)
            | ((((self.bits[3] >> 32) != 0) as u8) << 7)
    }

    /// Returns the active token count (`0..=32`) for each of the 8 Micro-Swarm 32-bit sub-regions.
    #[inline(always)]
    pub fn subswarm_popcounts_8(&self) -> [u8; NUM_STIGMERGIC_GRANULES] {
        [
            (self.bits[0] as u32).count_ones() as u8,
            ((self.bits[0] >> 32) as u32).count_ones() as u8,
            (self.bits[1] as u32).count_ones() as u8,
            ((self.bits[1] >> 32) as u32).count_ones() as u8,
            (self.bits[2] as u32).count_ones() as u8,
            ((self.bits[2] >> 32) as u32).count_ones() as u8,
            (self.bits[3] as u32).count_ones() as u8,
            ((self.bits[3] >> 32) as u32).count_ones() as u8,
        ]
    }

    /// Computes bitwise NOT ($\neg \mathbf{M}_i$) in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub fn bitnot_mask(&self) -> Self {
        Self {
            bits: [!self.bits[0], !self.bits[1], !self.bits[2], !self.bits[3]],
        }
    }

    /// Computes bitwise AND ($\mathbf{M}_i \land \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub fn bitand_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] & other.bits[0],
                self.bits[1] & other.bits[1],
                self.bits[2] & other.bits[2],
                self.bits[3] & other.bits[3],
            ],
        }
    }

    /// Computes bitwise OR ($\mathbf{M}_i \lor \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub fn bitor_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] | other.bits[0],
                self.bits[1] | other.bits[1],
                self.bits[2] | other.bits[2],
                self.bits[3] | other.bits[3],
            ],
        }
    }

    /// Computes bitwise XOR ($\mathbf{M}_i \oplus \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub fn bitxor_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] ^ other.bits[0],
                self.bits[1] ^ other.bits[1],
                self.bits[2] ^ other.bits[2],
                self.bits[3] ^ other.bits[3],
            ],
        }
    }

    /// Computes bitwise relative complement ($\mathbf{M}_i \land \neg \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub fn bitandnot_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] & !other.bits[0],
                self.bits[1] & !other.bits[1],
                self.bits[2] & !other.bits[2],
                self.bits[3] & !other.bits[3],
            ],
        }
    }

    /// Accumulates `other` into `self` via bitwise OR in-place ($\mathbf{M}_i \gets \mathbf{M}_i \lor \mathbf{M}_j$).
    #[inline(always)]
    pub fn merge_or_assign(&mut self, other: &Self) {
        self.bits[0] |= other.bits[0];
        self.bits[1] |= other.bits[1];
        self.bits[2] |= other.bits[2];
        self.bits[3] |= other.bits[3];
    }

    /// Computes Hamming distance $\mathrm{popcount}(\mathbf{M}_i \oplus \mathbf{M}_j)$ in $\mathcal{O}(1)$.
    #[inline(always)]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        ((self.bits[0] ^ other.bits[0]).count_ones() + (self.bits[1] ^ other.bits[1]).count_ones())
            + ((self.bits[2] ^ other.bits[2]).count_ones()
                + (self.bits[3] ^ other.bits[3]).count_ones())
    }

    /// Evaluates discrete Stigmergic Graph Laplacian adjacency weight $A_{ij} = \mathbf{M}_i^T \mathbf{M}_j$.
    #[inline(always)]
    pub fn stigmergic_adjacency_overlap(&self, other: &Self) -> u32 {
        ((self.bits[0] & other.bits[0]).count_ones() + (self.bits[1] & other.bits[1]).count_ones())
            + ((self.bits[2] & other.bits[2]).count_ones()
                + (self.bits[3] & other.bits[3]).count_ones())
    }

    /// Evaluates normalized Jaccard similarity $\frac{\vert{}\mathbf{M}_i \cap
    /// \mathbf{M}_j\vert{}}{\vert{}\mathbf{M}_i \cup \mathbf{M}_j\vert{}} \in [0, 1]$
    /// in division-free (`0 fdiv`) constant time via `POPCOUNT_RSQRT_LUT_257`.
    #[inline(always)]
    pub fn stigmergic_jaccard_similarity(&self, other: &Self) -> f32 {
        let inter = self.stigmergic_adjacency_overlap(other);
        let union = ((self.bits[0] | other.bits[0]).count_ones()
            + (self.bits[1] | other.bits[1]).count_ones())
            + ((self.bits[2] | other.bits[2]).count_ones()
                + (self.bits[3] | other.bits[3]).count_ones());
        let rsqrt_u = branchless_popcount_rsqrt_u32(union);
        branchless_clamp_f32((inter as f32) * (rsqrt_u * rsqrt_u), 0.0, 1.0)
    }

    /// Evaluates non-negative binary cosine similarity $\frac{\mathbf{M}_i^T
    /// \mathbf{M}_j}{\sqrt{\Vert{}\mathbf{M}_i\Vert{}_1 \Vert{}\mathbf{M}_j\Vert{}_1}} \in [0, 1]$.
    #[inline(always)]
    pub fn stigmergic_cosine_similarity(&self, other: &Self) -> f32 {
        let inter = self.stigmergic_adjacency_overlap(other);
        let rsqrt_a = branchless_popcount_rsqrt_u32(self.popcount());
        let rsqrt_b = branchless_popcount_rsqrt_u32(other.popcount());
        (inter as f32) * (rsqrt_a * rsqrt_b)
    }

    /// Evaluates binary cosine similarity using a precomputed reciprocal square root `inv_sqrt_norm_self`
    /// and the 64B-aligned `POPCOUNT_RSQRT_LUT_257` table in strictly constant time.
    #[inline(always)]
    pub fn stigmergic_cosine_similarity_with_inv_norm(
        &self,
        inv_sqrt_norm_self: f32,
        other: &Self,
    ) -> f32 {
        let inter = self.stigmergic_adjacency_overlap(other);
        let inv_sqrt_other = branchless_popcount_rsqrt_u32(other.popcount());
        let clean_self = branchless_relu_f32(inv_sqrt_norm_self);
        (inter as f32) * clean_self * inv_sqrt_other
    }

    /// Evaluates the Lowest Common Ancestor (LCA) prefix depth $l_{\mathrm{lca}}(x, y) \in [0, 256]$
    /// between two 256-bit dyadic keys in $\mathcal{O}(1)$ branchless operations (Lemma 1.1).
    #[inline(always)]
    pub fn dyadic_lca_depth(&self, other: &Self) -> u32 {
        dyadic_lca_depth_raw(self, other)
    }
}

#[inline(always)]
const fn dyadic_lca_depth_raw(mask_x: &PagedBitmask, mask_y: &PagedBitmask) -> u32 {
    let d0 = mask_x.bits[0] ^ mask_y.bits[0];
    let d1 = mask_x.bits[1] ^ mask_y.bits[1];
    let d2 = mask_x.bits[2] ^ mask_y.bits[2];
    let d3 = mask_x.bits[3] ^ mask_y.bits[3];
    let eq0 = d0 == 0;
    let eq01 = eq0 & (d1 == 0);
    let eq012 = eq01 & (d2 == 0);
    let all_zero = eq012 & (d3 == 0);
    let active_w = branchless_select_u64(
        eq0,
        branchless_select_u64(eq01, branchless_select_u64(eq012, d3, d2), d1),
        d0,
    );
    let base_lz = ((eq0 as u32) + (eq01 as u32) + (eq012 as u32)) << 6;
    branchless_select_u32(
        all_zero,
        PAGED_BITMASK_BITS,
        base_lz + (active_w | 1).leading_zeros(),
    )
}

/// Evaluates the normalized pseudo-ultrametric tree distance $d_{\mathcal{V}}(x, y) \in [0, 1)$
/// directly from two 256-bit dyadic bitmasks in $\mathcal{O}(1)$ (Lemma 1.1).
#[inline(always)]
#[must_use]
pub const fn eval_bitmask_dyadic_ultrametric(
    mask_x: &PagedBitmask,
    mask_y: &PagedBitmask,
    w_key: f32,
) -> f32 {
    let l_lca = dyadic_lca_depth_raw(mask_x, mask_y);
    let diff = PAGED_BITMASK_BITS.saturating_sub(l_lca) as f32;
    branchless_tanh_f32(branchless_relu_f32(w_key) * diff)
}

#[inline(always)]
#[must_use]
pub(crate) const fn swar_u8_pair_sum_u16(w: u64) -> u64 {
    (w & SWAR_EVEN_BYTES_64).wrapping_add((w >> 8) & SWAR_EVEN_BYTES_64)
}

#[inline(always)]
#[must_use]
pub(crate) const fn swar_total_mass_from_words(w: &[u64; 8]) -> u32 {
    let s01 = swar_u8_pair_sum_u16(w[0]).wrapping_add(swar_u8_pair_sum_u16(w[1]));
    let s23 = swar_u8_pair_sum_u16(w[2]).wrapping_add(swar_u8_pair_sum_u16(w[3]));
    let s45 = swar_u8_pair_sum_u16(w[4]).wrapping_add(swar_u8_pair_sum_u16(w[5]));
    let s67 = swar_u8_pair_sum_u16(w[6]).wrapping_add(swar_u8_pair_sum_u16(w[7]));
    let s_all = s01.wrapping_add(s23).wrapping_add(s45.wrapping_add(s67));
    (s_all.wrapping_mul(SWAR_SUM_U16_64) >> 48) as u32
}

#[inline(always)]
#[must_use]
pub(crate) const fn swar_active_buckets_from_words(w: &[u64; 8]) -> u32 {
    let nz01 = swar_u8_nonzero_msb_mask(w[0]) | (swar_u8_nonzero_msb_mask(w[1]) >> 1);
    let nz23 = (swar_u8_nonzero_msb_mask(w[2]) >> 2) | (swar_u8_nonzero_msb_mask(w[3]) >> 3);
    let nz45 = (swar_u8_nonzero_msb_mask(w[4]) >> 4) | (swar_u8_nonzero_msb_mask(w[5]) >> 5);
    let nz67 = (swar_u8_nonzero_msb_mask(w[6]) >> 6) | (swar_u8_nonzero_msb_mask(w[7]) >> 7);
    ((nz01 | nz23) | (nz45 | nz67)).count_ones()
}

// ===================================================================
// 4. Constant-Time Contrastive Phase Memory Counting Bloom Filter
// ===================================================================

/// Cacheline-aligned (64-byte) Counting Bloom Filter for Contrastive Phase Memory (Section 3.3).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContrastivePhaseBloomFilter {
    /// 64 saturating 8-bit frequency counters occupying exactly 1 CPU/GPU cacheline.
    pub counters: [u8; 64],
}

impl Default for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl ContrastivePhaseBloomFilter {
    /// Constructs a zeroed `ContrastivePhaseBloomFilter`.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counters: [0u8; 64],
        }
    }

    /// Borrows `self.counters` (`#[repr(C, align(64))]`) as a zero-copy 64-byte aligned `&[u64; 8]`
    /// SWAR word array on little-endian targets.
    #[inline(always)]
    #[must_use]
    pub const fn counter_words_ref(&self) -> &[u64; 8] {
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: `ContrastivePhaseBloomFilter` is `#[repr(C, align(64))]` (`64 % 8 == 0`),
        // and `counters: [u8; 64]` has identical size and bit validity as `[u64; 8]`.
        unsafe { &*(self.counters.as_ptr().cast::<[u64; 8]>()) }
    }

    /// Mutably borrows `self.counters` (`#[repr(C, align(64))]`) as a zero-copy 64-byte aligned
    /// `&mut [u64; 8]` SWAR word array on little-endian targets.
    #[inline(always)]
    #[must_use]
    pub const fn counter_words_mut(&mut self) -> &mut [u64; 8] {
        const _: () = assert!(cfg!(target_endian = "little"));
        // SAFETY: `ContrastivePhaseBloomFilter` is `#[repr(C, align(64))]` (`64 % 8 == 0`),
        // and `counters: [u8; 64]` has identical size and bit validity as `[u64; 8]`.
        unsafe { &mut *(self.counters.as_mut_ptr().cast::<[u64; 8]>()) }
    }

    /// Resets all 64 counters to zero in-place.
    #[inline(always)]
    pub const fn clear(&mut self) {
        *self.counter_words_mut() = [0u64; 8];
    }

    /// Returns `true` in $\mathcal{O}(1)$ if all 64 counter buckets are zero.
    #[inline(always)]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        !swar_words_nonzero_64(self.counter_words_ref())
    }

    /// Halves all 64 8-bit counters in-place (`c_i >>= 1`) using unrolled 8-lane SWAR (`swar_u8_halve_decay_x8`).
    #[inline(always)]
    pub const fn halve_counters_swar(&mut self) {
        let w = *self.counter_words_ref();
        *self.counter_words_mut() = [
            swar_u8_halve_decay_x8(w[0]),
            swar_u8_halve_decay_x8(w[1]),
            swar_u8_halve_decay_x8(w[2]),
            swar_u8_halve_decay_x8(w[3]),
            swar_u8_halve_decay_x8(w[4]),
            swar_u8_halve_decay_x8(w[5]),
            swar_u8_halve_decay_x8(w[6]),
            swar_u8_halve_decay_x8(w[7]),
        ];
    }

    /// Evaluates whether the Counting Bloom Filter has crossed `max_mass_threshold` or
    /// [`DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS`] (`48` buckets) and halves all 8 SWAR words in-place
    /// when saturated, skipping redundant 64-byte stores when below threshold (`0` narrow-to-wide STLF stalls).
    #[inline(always)]
    pub const fn auto_halve_if_saturated(&mut self, max_mass_threshold: u32) -> bool {
        let w = *self.counter_words_ref();
        let mass = swar_total_mass_from_words(&w);
        let active_buckets = swar_active_buckets_from_words(&w);

        let saturated =
            (mass > max_mass_threshold) | (active_buckets > DEFAULT_BLOOM_MAX_ACTIVE_BUCKETS);
        if saturated {
            *self.counter_words_mut() = [
                (w[0] >> 1) & SWAR_LOW7_MASK_64,
                (w[1] >> 1) & SWAR_LOW7_MASK_64,
                (w[2] >> 1) & SWAR_LOW7_MASK_64,
                (w[3] >> 1) & SWAR_LOW7_MASK_64,
                (w[4] >> 1) & SWAR_LOW7_MASK_64,
                (w[5] >> 1) & SWAR_LOW7_MASK_64,
                (w[6] >> 1) & SWAR_LOW7_MASK_64,
                (w[7] >> 1) & SWAR_LOW7_MASK_64,
            ];
        }
        saturated
    }

    /// Computes 4 independent 6-bit slot indices ($0..64$) by folding all 64 Q7 head phases
    /// across 4 parallel superscalar multiply lanes in $\mathcal{O}(1)$ directly from 8 SWAR words.
    #[inline(always)]
    #[must_use]
    pub const fn compute_hash_slots_from_words(w: &[u64; 8]) -> [usize; 4] {
        const K0: u64 = 0x9E37_79B9_7F4A_7C15;
        const K1: u64 = 0xBF58_476D_1CE4_E5B9;
        const K2: u64 = 0x94D0_49BB_1331_11EB;
        const K3: u64 = 0xD6E8_FEB8_6659_FD93;

        let m0 = (w[0] ^ K0).wrapping_mul(K1) ^ w[4].rotate_left(17);
        let m1 = (w[1] ^ K1).wrapping_mul(K2) ^ w[5].rotate_left(29);
        let m2 = (w[2] ^ K2).wrapping_mul(K3) ^ w[6].rotate_left(41);
        let m3 = (w[3] ^ K3).wrapping_mul(K0) ^ w[7].rotate_left(53);

        let f01 = (m0 ^ m1.rotate_left(13)).wrapping_mul(K1);
        let f23 = (m2 ^ m3.rotate_left(37)).wrapping_mul(K2);
        let mix = branchless_splitmix64(f01.wrapping_add(f23));
        [
            (mix & 0x3F) as usize,
            ((mix >> 16) & 0x3F) as usize,
            ((mix >> 32) & 0x3F) as usize,
            ((mix >> 48) & 0x3F) as usize,
        ]
    }

    /// Computes 4 independent 6-bit slot indices ($0..64$) by folding all 64 Q7 head phases
    /// across 4 parallel superscalar multiply lanes in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn compute_hash_slots(phase_pattern: &[PhaseQ7; MAX_ATTENTION_HEADS]) -> [usize; 4] {
        Self::compute_hash_slots_from_words(&load_q7_words_64(phase_pattern))
    }

    /// Queries the minimum collision count (repulsive potential intensity) directly from 8 SWAR words,
    /// short-circuiting when the 64-byte Bloom filter is empty and evaluating the 4-way minimum
    /// branchlessly across active hash buckets.
    #[inline(always)]
    #[must_use]
    pub const fn query_repulsive_density_from_words(&self, w: &[u64; 8]) -> u8 {
        if self.is_empty() {
            return 0;
        }
        let slots = Self::compute_hash_slots_from_words(w);
        let c0 = self.counters[slots[0]];
        let c1 = self.counters[slots[1]];
        let c2 = self.counters[slots[2]];
        let c3 = self.counters[slots[3]];
        let m01 = branchless_select_u8(c0 < c1, c0, c1);
        let m23 = branchless_select_u8(c2 < c3, c2, c3);
        branchless_select_u8(m01 < m23, m01, m23)
    }

    /// Applies Phase-Conjugate Attractor Repulsion ($-\boldsymbol{\Delta\theta}$) in-place directly
    /// on 8 register-resident SWAR words `words: &mut [u64; 8]` in strictly constant time (`0` branches),
    /// fast-pathing word-uniform 64-head profiles (`1` SWAR evaluation instead of `8`) and
    /// returning `(density, repulsion_triggered)`.
    #[inline(always)]
    pub const fn apply_phase_conjugate_repulsion_words(
        &self,
        words: &mut [u64; 8],
        density_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (u8, bool) {
        let density = self.query_repulsive_density_from_words(words);
        let active = (density >= density_threshold) & (density > 0);
        if !active {
            return (density, false);
        }
        let gt_128 = repulsion_scale_u7 > 128;
        let s_sub = branchless_select_u8(
            gt_128,
            repulsion_scale_u7.wrapping_sub(128),
            repulsion_scale_u7,
        ) as u64;
        let bias_restore = (128u64.wrapping_sub(s_sub)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let extra_mask = (gt_128 as u64).wrapping_neg();

        if is_word_array_uniform_64(words) {
            let neg0 = swar_q7_neg_sat_x8(words[0]);
            let rep0 = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(neg0, s_sub, bias_restore),
                neg0 & extra_mask,
            );
            *words = [rep0; 8];
            return (density, true);
        }

        let mut c = 0usize;
        while c < 8 {
            let cur = words[c];
            let neg = swar_q7_neg_sat_x8(cur);
            words[c] = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(neg, s_sub, bias_restore),
                neg & extra_mask,
            );
            c += 1;
        }
        (density, true)
    }

    /// Records a spurious local minimum phase pattern directly from 8 register-resident `u64` SWAR
    /// words with a custom saturating weight step (`0` stack reload, `0` branches).
    #[inline(always)]
    pub const fn insert_repulsive_pattern_words_weighted(&mut self, words: &[u64; 8], weight: u8) {
        let _ = self.auto_halve_if_saturated(DEFAULT_BLOOM_MAX_MASS_THRESHOLD);
        let slots = Self::compute_hash_slots_from_words(words);
        self.counters[slots[0]] = self.counters[slots[0]].saturating_add(weight);
        self.counters[slots[1]] = self.counters[slots[1]].saturating_add(weight);
        self.counters[slots[2]] = self.counters[slots[2]].saturating_add(weight);
        self.counters[slots[3]] = self.counters[slots[3]].saturating_add(weight);
    }

    /// Records a spurious local minimum phase pattern into the Counting Bloom Filter in strictly
    /// constant time (`0` branches).
    #[inline(always)]
    pub const fn insert_repulsive_pattern(
        &mut self,
        phase_pattern: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) {
        self.insert_repulsive_pattern_weighted(phase_pattern, 1);
    }

    /// Records a spurious local minimum phase pattern with a custom saturating weight step,
    /// performing branchless constant-time SWAR self-normalization (`auto_halve_if_saturated`).
    #[inline(always)]
    pub const fn insert_repulsive_pattern_weighted(
        &mut self,
        phase_pattern: &[PhaseQ7; MAX_ATTENTION_HEADS],
        weight: u8,
    ) {
        self.insert_repulsive_pattern_words_weighted(&load_q7_words_64(phase_pattern), weight);
    }

    /// Queries the minimum collision count (repulsive potential intensity) in strictly constant time (`0` branches).
    #[inline(always)]
    #[must_use]
    pub const fn query_repulsive_density(
        &self,
        phase_pattern: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> u8 {
        self.query_repulsive_density_from_words(&load_q7_words_64(phase_pattern))
    }

    /// Evaluates normalized repulsive potential energy in floating-point domain ($\in [0, \mathrm{gain}]$).
    #[inline(always)]
    #[must_use]
    pub const fn query_repulsion_potential(
        &self,
        phase_pattern: &[PhaseQ7; MAX_ATTENTION_HEADS],
        gain: f32,
    ) -> f32 {
        let density = self.query_repulsive_density(phase_pattern) as f32;
        branchless_relu_f32(gain) * (density * (1.0 / 255.0))
    }

    /// Applies Phase-Conjugate Attractor Repulsion ($-\boldsymbol{\Delta\theta}$) in-place across
    /// `candidate_phases` and returns `(density, repulsion_triggered, post_repulsion_words)` directly
    /// in 8 `u64` SWAR registers (`0` stack reload for downstream `PhaseStepFilter`/`PhaseClamp` or
    /// unified-field kernels).
    #[inline(always)]
    pub const fn apply_phase_conjugate_repulsion_to_words(
        &self,
        candidate_phases: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
        density_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (u8, bool, [u64; 8]) {
        let mut words = load_q7_words_64(candidate_phases);
        let (density, active) = self.apply_phase_conjugate_repulsion_words(
            &mut words,
            density_threshold,
            repulsion_scale_u7,
        );
        if active {
            store_q7_words_64(candidate_phases, &words);
        }
        (density, active, words)
    }

    /// Applies evaporative decay across all 64 counters in-place in strictly constant time (`0` branches).
    #[inline(always)]
    pub const fn decay_counters(&mut self, decay_step: u8) {
        let w = *self.counter_words_ref();
        *self.counter_words_mut() = [
            swar_u8_sub_sat_x8(w[0], decay_step),
            swar_u8_sub_sat_x8(w[1], decay_step),
            swar_u8_sub_sat_x8(w[2], decay_step),
            swar_u8_sub_sat_x8(w[3], decay_step),
            swar_u8_sub_sat_x8(w[4], decay_step),
            swar_u8_sub_sat_x8(w[5], decay_step),
            swar_u8_sub_sat_x8(w[6], decay_step),
            swar_u8_sub_sat_x8(w[7], decay_step),
        ];
    }

    /// Merges another `ContrastivePhaseBloomFilter` into `self` via 8-lane SWAR saturating addition
    /// in strictly constant time (`0` branches).
    #[inline(always)]
    pub const fn merge_counters_sat(&mut self, other: &Self) {
        let a = *self.counter_words_ref();
        let b = *other.counter_words_ref();
        *self.counter_words_mut() = [
            swar_u8_add_sat_x8(a[0], b[0]),
            swar_u8_add_sat_x8(a[1], b[1]),
            swar_u8_add_sat_x8(a[2], b[2]),
            swar_u8_add_sat_x8(a[3], b[3]),
            swar_u8_add_sat_x8(a[4], b[4]),
            swar_u8_add_sat_x8(a[5], b[5]),
            swar_u8_add_sat_x8(a[6], b[6]),
            swar_u8_add_sat_x8(a[7], b[7]),
        ];
        let _ = self.auto_halve_if_saturated(DEFAULT_BLOOM_MAX_MASS_THRESHOLD);
    }

    /// Merges another `ContrastivePhaseBloomFilter` into `self` via 8-lane SWAR elementwise maximum
    /// in strictly constant time (`0` branches).
    #[inline(always)]
    pub const fn merge_counters_max(&mut self, other: &Self) {
        let a = *self.counter_words_ref();
        let b = *other.counter_words_ref();
        *self.counter_words_mut() = [
            swar_u8_max_x8(a[0], b[0]),
            swar_u8_max_x8(a[1], b[1]),
            swar_u8_max_x8(a[2], b[2]),
            swar_u8_max_x8(a[3], b[3]),
            swar_u8_max_x8(a[4], b[4]),
            swar_u8_max_x8(a[5], b[5]),
            swar_u8_max_x8(a[6], b[6]),
            swar_u8_max_x8(a[7], b[7]),
        ];
    }

    /// Returns the total accumulated counter mass across all 64 slots in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn total_mass(&self) -> u32 {
        swar_total_mass_from_words(self.counter_words_ref())
    }

    /// Returns the number of non-zero counter buckets (`0..=64`) currently active in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn active_bucket_count(&self) -> u32 {
        swar_active_buckets_from_words(self.counter_words_ref())
    }
}

// SAFETY: `ContrastivePhaseBloomFilter` is a 64-byte aligned array of `u8` with no padding traps.
unsafe impl Pod for ContrastivePhaseBloomFilter {}

// ===================================================================
// 7. Dynamic Adaptive Extension Scratchpad (`FastPathLedger`)
// ===================================================================

/// Lv.1 Capability Trait for the 64-byte Dynamic Adaptive Extension Scratchpad (FastPathLedger):
/// $\mathcal{O}(1)$ fast-path tagged pointer shortcuts, 8-byte circular telemetry mini-ledger,
/// and dynamic parameter auto-tuning.
pub trait FastPathLedger {
    /// Resets active telemetry mini ledger and cache counters in-place.
    ///
    /// # Errors
    /// Returns an error status only if a custom implementor rejects resetting a locked slot.
    #[inline(always)]
    fn reset_telemetry(&mut self, slot: &mut LedgerSlot) -> LmnkResult<()> {
        slot.telemetry_ring_head = 0;
        slot.telemetry_ring_tail = 0;
        slot.cache_hit_count = 0;
        slot.telemetry_mini_ledger = [0u8; LEDGER_HISTORY_LEN];
        Ok(())
    }

    /// Executes fast-path shortcut lookup in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_STATE`] if `slot.mode != 0` or no matching non-zero shortcut is registered.
    #[inline(always)]
    fn fast_path_lookup(&self, slot: &LedgerSlot, input_key: u64) -> LmnkResult<u64> {
        let (hit, val) = self.fast_path_lookup_branchless(slot, input_key);
        if hit {
            Ok(val)
        } else {
            Err(LMNK_ERR_INVALID_STATE)
        }
    }

    /// Executes strictly branchless $\mathcal{O}(1)$ fast-path lookup across all 4 shortcut registers.
    #[inline(always)]
    #[must_use]
    fn fast_path_lookup_branchless(&self, slot: &LedgerSlot, input_key: u64) -> (bool, u64) {
        let mask_hit = ((slot.fast_path_bypass_mask & input_key) != 0) & (slot.mode == 0);
        let idx = ((input_key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
        let candidate = slot.fast_path_shortcuts[idx & (FAST_PATH_SHORTCUT_SLOTS - 1)];
        let fallback = slot.fast_path_shortcuts[0];
        let resolved = branchless_select_u64(candidate != 0, candidate, fallback);
        let valid_hit = mask_hit & (resolved != 0);
        (valid_hit, branchless_select_u64(valid_hit, resolved, 0))
    }

    /// Branchlessly resolves `input_ptr` against the FastPathLedger fast-path shortcut table in $\mathcal{O}(1)$.
    #[inline(always)]
    fn resolve_or_pass_tagged_ptr(
        &self,
        slot: &mut LedgerSlot,
        input_ptr: TaggedPtr,
    ) -> (bool, TaggedPtr) {
        let (hit, shortcut_raw) = self.fast_path_lookup_branchless(slot, input_ptr.raw());
        slot.cache_hit_count = slot.cache_hit_count.wrapping_add(u32::from(hit));
        let eff_raw = branchless_select_u64(hit, shortcut_raw, input_ptr.raw());
        (hit, TaggedPtr(eff_raw))
    }

    /// Registers a 64-bit tagged pointer shortcut into the FastPathLedger fast-path table in $\mathcal{O}(1)$.
    #[inline(always)]
    fn register_fast_path_shortcut(
        &mut self,
        slot: &mut LedgerSlot,
        input_key: u64,
        target_shortcut: u64,
    ) {
        let idx = ((input_key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
        slot.fast_path_shortcuts[idx & (FAST_PATH_SHORTCUT_SLOTS - 1)] = target_shortcut;
        slot.fast_path_bypass_mask |= branchless_select_u64(target_shortcut != 0, input_key, 0);
    }

    /// Pushes a profiling metric byte code (`LEDGER_METRIC_*`) into the FastPathLedger telemetry ring ledger in-place.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_LEDGER_FULL`] only if non-overwriting ring semantics are enforced.
    #[inline(always)]
    fn telemetry_push(&mut self, slot: &mut LedgerSlot, metric_code: u8) -> LmnkResult<()> {
        let head = (slot.telemetry_ring_head as usize) & (LEDGER_HISTORY_LEN - 1);
        slot.telemetry_mini_ledger[head] = metric_code;
        slot.telemetry_ring_head = slot.telemetry_ring_head.wrapping_add(1);
        Ok(())
    }

    /// Summarizes the 8-byte `telemetry_mini_ledger` in $\mathcal{O}(1)$ using 8-lane SWAR,
    /// returning `(combined_metric_or_mask, active_entry_count)`.
    #[inline(always)]
    #[must_use]
    fn telemetry_snapshot_metrics(&self, slot: &LedgerSlot) -> (u8, u32) {
        let word = u64::from_le_bytes(slot.telemetry_mini_ledger);
        let fold32 = (word as u32) | ((word >> 32) as u32);
        let fold16 = (fold32 & 0xFFFF) | (fold32 >> 16);
        let combined_mask = ((fold16 & 0xFF) | (fold16 >> 8)) as u8;
        (combined_mask, swar_u8_nonzero_count_x8(word))
    }

    /// Sets the operating mode of the ledger slot (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC Buffer).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `mode > 2`.
    #[inline(always)]
    fn set_mode(&mut self, slot: &mut LedgerSlot, mode: u32) -> LmnkResult<()> {
        if mode > 2 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        slot.mode = mode;
        Ok(())
    }

    /// Evaluates telemetry ledger to auto-tune sub-head router ratios, `PhaseClamp` limits, and
    /// `EntropyController` thresholds.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_STATE`] if `slot` is in an incompatible mode.
    fn evaluate_auto_tune(
        &self,
        slot: &LedgerSlot,
        config_out: &mut RuntimeConfig,
    ) -> LmnkResult<()>;

    /// Returns the cumulative cache hit count recorded in the slot.
    #[inline(always)]
    fn cache_hit_count(&self, slot: &LedgerSlot) -> u32 {
        slot.cache_hit_count
    }
}
