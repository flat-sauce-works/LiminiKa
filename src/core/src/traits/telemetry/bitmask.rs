// name: src/core/src/traits/telemetry/bitmask.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 256-bit PagedBlock bitmask algebra (`PagedBitmaskOps`) and dyadic ultrametric helpers.

use super::*;

// ===================================================================
// 2. 256-Bit PagedBlock Bitmask Algebra (`PagedBitmaskOps`)
// ===================================================================

/// Branchless $\mathcal{O}(1)$ bitwise, stigmergic graph adjacency, and 8-cell sub-swarm partitioning
/// operations on 256-bit [`gcso_paged_bitmask_t`].
pub trait PagedBitmaskOps: Sized {
    /// Returns `true` if all 256 bits across the four 64-bit words are zero.
    fn is_all_zero(&self) -> bool;
    /// Returns the total Hamming weight (`popcount`) across all 256 bits in $\mathcal{O}(1)$.
    fn popcount(&self) -> u32;
    /// Sets bit `bit_idx` ($0 \le \text{bit\_idx} < 256$) branchlessly. Out-of-range indices safely no-op.
    fn set_bit_branchless(&mut self, bit_idx: u32);
    /// Clears bit `bit_idx` ($0 \le \text{bit\_idx} < 256$) branchlessly. Out-of-range indices safely no-op.
    fn clear_bit_branchless(&mut self, bit_idx: u32);
    /// Tests whether bit `bit_idx` ($0 \le \text{bit\_idx} < 256$) is set branchlessly.
    fn test_bit_branchless(&self, bit_idx: u32) -> bool;
    /// Returns the lowest set bit index (`0..=255`), or `256` (`PAGED_BITMASK_BITS`) if all bits are zero.
    fn first_set_bit(&self) -> u32;
    /// Returns the highest set bit index (`0..=255`), or `256` (`PAGED_BITMASK_BITS`) if all bits are zero.
    fn last_set_bit(&self) -> u32;
    /// Compresses the 256-bit PagedBlock bitmask into an **8-bit Micro-Swarm Activity Mask**.
    fn subswarm_activity_mask_8(&self) -> u8;
    /// Returns the active token count (`0..=32`) for each of the 8 Micro-Swarm 32-bit sub-regions.
    fn subswarm_popcounts_8(&self) -> [u8; NUM_SWARM_MICRO_CELLS];
    /// Computes bitwise NOT ($\neg \mathbf{M}_i$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitnot_mask(&self) -> Self;
    /// Computes bitwise AND ($\mathbf{M}_i \land \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitand_mask(&self, other: &Self) -> Self;
    /// Computes bitwise OR ($\mathbf{M}_i \lor \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitor_mask(&self, other: &Self) -> Self;
    /// Computes bitwise XOR ($\mathbf{M}_i \oplus \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitxor_mask(&self, other: &Self) -> Self;
    /// Computes bitwise relative complement ($\mathbf{M}_i \land \neg \mathbf{M}_j$) in $\mathcal{O}(1)$.
    #[must_use]
    fn bitandnot_mask(&self, other: &Self) -> Self;
    /// Accumulates `other` into `self` via bitwise OR in-place ($\mathbf{M}_i \gets \mathbf{M}_i \lor \mathbf{M}_j$).
    fn merge_or_assign(&mut self, other: &Self);
    /// Computes Hamming distance $\mathrm{popcount}(\mathbf{M}_i \oplus \mathbf{M}_j)$ in $\mathcal{O}(1)$.
    fn hamming_distance(&self, other: &Self) -> u32;
    /// Evaluates discrete Stigmergic Graph Laplacian adjacency weight $A_{ij} = \mathbf{M}_i^T \mathbf{M}_j$.
    fn stigmergic_adjacency_overlap(&self, other: &Self) -> u32;
    /// Evaluates normalized Jaccard similarity $\frac{\vert{}\mathbf{M}_i \cap \mathbf{M}_j\vert{}}{\vert{}\mathbf{M}_i \cup \mathbf{M}_j\vert{}} \in [0, 1]$
    /// in division-free (`0 fdiv`) constant time via `POPCOUNT_RSQRT_LUT_257`.
    fn stigmergic_jaccard_similarity(&self, other: &Self) -> f32;
    /// Evaluates non-negative binary cosine similarity $\frac{\mathbf{M}_i^T \mathbf{M}_j}{\sqrt{\Vert{}\mathbf{M}_i\Vert{}_1 \Vert{}\mathbf{M}_j\Vert{}_1}} \in [0, 1]$.
    fn stigmergic_cosine_similarity(&self, other: &Self) -> f32;

    /// Evaluates binary cosine similarity using a precomputed reciprocal square root `inv_sqrt_norm_self`
    /// and the 64B-aligned `POPCOUNT_RSQRT_LUT_257` table in strictly constant time.
    #[inline(always)]
    fn stigmergic_cosine_similarity_with_inv_norm(
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
    fn dyadic_lca_depth(&self, other: &Self) -> u32;
}

impl PagedBitmaskOps for gcso_paged_bitmask_t {
    #[inline(always)]
    fn is_all_zero(&self) -> bool {
        ((self.bits[0] | self.bits[1]) | (self.bits[2] | self.bits[3])) == 0
    }

    #[inline(always)]
    fn popcount(&self) -> u32 {
        (self.bits[0].count_ones() + self.bits[1].count_ones())
            + (self.bits[2].count_ones() + self.bits[3].count_ones())
    }

    #[inline(always)]
    fn set_bit_branchless(&mut self, bit_idx: u32) {
        let in_bounds_mask = ((bit_idx < PAGED_BITMASK_BITS) as u64).wrapping_neg();
        let bit_val = (1u64 << (bit_idx & 63)) & in_bounds_mask;
        self.bits[((bit_idx >> 6) as usize) & 3] |= bit_val;
    }

    #[inline(always)]
    fn clear_bit_branchless(&mut self, bit_idx: u32) {
        let in_bounds_mask = ((bit_idx < PAGED_BITMASK_BITS) as u64).wrapping_neg();
        let clear_mask = !((1u64 << (bit_idx & 63)) & in_bounds_mask);
        self.bits[((bit_idx >> 6) as usize) & 3] &= clear_mask;
    }

    #[inline(always)]
    fn test_bit_branchless(&self, bit_idx: u32) -> bool {
        let in_bounds = bit_idx < PAGED_BITMASK_BITS;
        let word = self.bits[((bit_idx >> 6) as usize) & 3];
        in_bounds & (((word >> (bit_idx & 63)) & 1) != 0)
    }

    #[inline(always)]
    fn first_set_bit(&self) -> u32 {
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

    #[inline(always)]
    fn last_set_bit(&self) -> u32 {
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

    #[inline(always)]
    fn subswarm_activity_mask_8(&self) -> u8 {
        (((self.bits[0] & 0xFFFF_FFFF) != 0) as u8)
            | ((((self.bits[0] >> 32) != 0) as u8) << 1)
            | ((((self.bits[1] & 0xFFFF_FFFF) != 0) as u8) << 2)
            | ((((self.bits[1] >> 32) != 0) as u8) << 3)
            | ((((self.bits[2] & 0xFFFF_FFFF) != 0) as u8) << 4)
            | ((((self.bits[2] >> 32) != 0) as u8) << 5)
            | ((((self.bits[3] & 0xFFFF_FFFF) != 0) as u8) << 6)
            | ((((self.bits[3] >> 32) != 0) as u8) << 7)
    }

    #[inline(always)]
    fn subswarm_popcounts_8(&self) -> [u8; NUM_SWARM_MICRO_CELLS] {
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

    #[inline(always)]
    fn bitnot_mask(&self) -> Self {
        Self {
            bits: [!self.bits[0], !self.bits[1], !self.bits[2], !self.bits[3]],
        }
    }

    #[inline(always)]
    fn bitand_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] & other.bits[0],
                self.bits[1] & other.bits[1],
                self.bits[2] & other.bits[2],
                self.bits[3] & other.bits[3],
            ],
        }
    }

    #[inline(always)]
    fn bitor_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] | other.bits[0],
                self.bits[1] | other.bits[1],
                self.bits[2] | other.bits[2],
                self.bits[3] | other.bits[3],
            ],
        }
    }

    #[inline(always)]
    fn bitxor_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] ^ other.bits[0],
                self.bits[1] ^ other.bits[1],
                self.bits[2] ^ other.bits[2],
                self.bits[3] ^ other.bits[3],
            ],
        }
    }

    #[inline(always)]
    fn bitandnot_mask(&self, other: &Self) -> Self {
        Self {
            bits: [
                self.bits[0] & !other.bits[0],
                self.bits[1] & !other.bits[1],
                self.bits[2] & !other.bits[2],
                self.bits[3] & !other.bits[3],
            ],
        }
    }

    #[inline(always)]
    fn merge_or_assign(&mut self, other: &Self) {
        self.bits[0] |= other.bits[0];
        self.bits[1] |= other.bits[1];
        self.bits[2] |= other.bits[2];
        self.bits[3] |= other.bits[3];
    }

    #[inline(always)]
    fn hamming_distance(&self, other: &Self) -> u32 {
        ((self.bits[0] ^ other.bits[0]).count_ones() + (self.bits[1] ^ other.bits[1]).count_ones())
            + ((self.bits[2] ^ other.bits[2]).count_ones()
                + (self.bits[3] ^ other.bits[3]).count_ones())
    }

    #[inline(always)]
    fn stigmergic_adjacency_overlap(&self, other: &Self) -> u32 {
        ((self.bits[0] & other.bits[0]).count_ones() + (self.bits[1] & other.bits[1]).count_ones())
            + ((self.bits[2] & other.bits[2]).count_ones()
                + (self.bits[3] & other.bits[3]).count_ones())
    }

    #[inline(always)]
    fn stigmergic_jaccard_similarity(&self, other: &Self) -> f32 {
        let inter = self.stigmergic_adjacency_overlap(other);
        let union = ((self.bits[0] | other.bits[0]).count_ones()
            + (self.bits[1] | other.bits[1]).count_ones())
            + ((self.bits[2] | other.bits[2]).count_ones()
                + (self.bits[3] | other.bits[3]).count_ones());
        let rsqrt_u = branchless_popcount_rsqrt_u32(union);
        branchless_clamp_f32((inter as f32) * (rsqrt_u * rsqrt_u), 0.0, 1.0)
    }

    #[inline(always)]
    fn stigmergic_cosine_similarity(&self, other: &Self) -> f32 {
        let inter = self.stigmergic_adjacency_overlap(other);
        let rsqrt_a = branchless_popcount_rsqrt_u32(self.popcount());
        let rsqrt_b = branchless_popcount_rsqrt_u32(other.popcount());
        (inter as f32) * (rsqrt_a * rsqrt_b)
    }

    #[inline(always)]
    fn dyadic_lca_depth(&self, other: &Self) -> u32 {
        let d0 = self.bits[0] ^ other.bits[0];
        let d1 = self.bits[1] ^ other.bits[1];
        let d2 = self.bits[2] ^ other.bits[2];
        let d3 = self.bits[3] ^ other.bits[3];
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
}

/// Evaluates the normalized pseudo-ultrametric tree distance $d_{\mathcal{V}}(x, y) \in [0, 1)$
/// directly from two 256-bit dyadic bitmasks in $\mathcal{O}(1)$ (Lemma 1.1).
#[inline(always)]
#[must_use]
pub const fn eval_bitmask_dyadic_ultrametric(
    mask_x: &gcso_paged_bitmask_t,
    mask_y: &gcso_paged_bitmask_t,
    w_key: f32,
) -> f32 {
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
    let l_lca = branchless_select_u32(
        all_zero,
        PAGED_BITMASK_BITS,
        base_lz + (active_w | 1).leading_zeros(),
    );
    let diff = PAGED_BITMASK_BITS.saturating_sub(l_lca) as f32;
    branchless_tanh_f32(branchless_relu_f32(w_key) * diff)
}

