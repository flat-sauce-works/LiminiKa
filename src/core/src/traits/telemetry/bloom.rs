// name: src/core/src/traits/telemetry/bloom.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 64-byte constant-time Contrastive Phase Memory Counting Bloom Filter.

use super::swarm::*;
use super::*;

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
    pub const fn compute_hash_slots(
        phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> [usize; 4] {
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
        phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) {
        self.insert_repulsive_pattern_weighted(phase_pattern, 1);
    }

    /// Records a spurious local minimum phase pattern with a custom saturating weight step,
    /// performing branchless constant-time SWAR self-normalization (`auto_halve_if_saturated`).
    #[inline(always)]
    pub const fn insert_repulsive_pattern_weighted(
        &mut self,
        phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        weight: u8,
    ) {
        self.insert_repulsive_pattern_words_weighted(&load_q7_words_64(phase_pattern), weight);
    }

    /// Queries the minimum collision count (repulsive potential intensity) in strictly constant time (`0` branches).
    #[inline(always)]
    #[must_use]
    pub const fn query_repulsive_density(
        &self,
        phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> u8 {
        self.query_repulsive_density_from_words(&load_q7_words_64(phase_pattern))
    }

    /// Evaluates normalized repulsive potential energy in floating-point domain ($\in [0, \mathrm{gain}]$).
    #[inline(always)]
    #[must_use]
    pub const fn query_repulsion_potential(
        &self,
        phase_pattern: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        gain: f32,
    ) -> f32 {
        let density = self.query_repulsive_density(phase_pattern) as f32;
        branchless_relu_f32(gain) * (density * (1.0 / 255.0))
    }

    /// Applies Phase-Conjugate Attractor Repulsion ($-\boldsymbol{\Delta\theta}$) in-place across
    /// `candidate_phases` and returns `(density, repulsion_triggered, post_repulsion_words)` directly
    /// in 8 `u64` SWAR registers (`0` stack reload for downstream QDPS/RIPA or unified-field kernels).
    #[inline(always)]
    pub const fn apply_phase_conjugate_repulsion_to_words(
        &self,
        candidate_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
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

    /// Applies Phase-Conjugate Attractor Repulsion ($-\boldsymbol{\Delta\theta}$) in-place across
    /// all 64 attention heads in strictly constant time (`0` branches) using `branchless_select_u64`
    /// and hoisted `bias_restore`.
    #[inline(always)]
    pub const fn apply_phase_conjugate_repulsion(
        &self,
        candidate_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        density_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> u8 {
        self.apply_phase_conjugate_repulsion_to_words(
            candidate_phases,
            density_threshold,
            repulsion_scale_u7,
        )
        .0
    }

    /// Evaluates `apply_phase_conjugate_repulsion` on `candidate_phases` and simultaneously evolves
    /// `tagged_ptr` into [`ACTION_PRESET_REPULSION_ESCAPE`] targeting [`ATTRACTOR_SLOT_REPULSION`] (`lane 3`).
    #[inline(always)]
    pub fn apply_repulsion_and_evolve_ptr(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        candidate_phases: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        density_threshold: u8,
        repulsion_scale_u7: u8,
    ) -> (gcso_tagged_ptr_t, u8, bool) {
        let density = self.apply_phase_conjugate_repulsion(
            candidate_phases,
            density_threshold,
            repulsion_scale_u7,
        );
        let triggered = (density >= density_threshold) & (density > 0);
        let rep_mode = tagged_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
        let rep_ptr = tagged_ptr
            .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
            .conjugate_phase();
        let out_ptr = gcso_tagged_ptr_t(branchless_select_u64(
            triggered,
            rep_ptr.raw(),
            tagged_ptr.raw(),
        ));
        (out_ptr, density, triggered)
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

