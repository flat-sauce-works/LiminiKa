// name: src/adapter/src/host/pager.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded working-set anchor pager ([`AnchorWorkingSetPager`]) bridging large `.lmnkc` anchor
//! banks (`> 64` [`PhaseAnchorRecord`] entries) onto a fixed-capacity resident working set
//! (`<= 64` by default) using `liminal` axioms (`BitCut64`, `Lacuna`, `LacunaPartition8`) and
//! `DistortionTension` homeostatic pressure tracking.

use std::collections::HashMap;

use liminika_core::drivers::DistortionTension;
use liminika_core::liminal::{
    BitCut64, LMNK_ERR_TOPOLOGY_MISMATCH, Lacuna, LacunaPartition8, MAX_ATTENTION_HEADS,
};
use liminika_core::runtime::{
    CORE_TOPOLOGY, LMNK_CONTAINER_FLAG_COMPILED_CACHE, LmnkContainerView,
};
use liminika_core::stigmergy::{MAX_PHASE_ANCHORS, PhaseAnchorRecord};

/// Default resident working-set ceiling aligned with [`MAX_PHASE_ANCHORS`] (`64`).
pub const DEFAULT_WORKING_SET_CAPACITY: usize = MAX_PHASE_ANCHORS;

/// Default SWAR coarse prefilter Top-K candidate budget (`32`).
pub const DEFAULT_PREFILTER_TOP_K: usize = 32;

/// Default per-step dissipative excitation decay factor (`0.92`).
pub const DEFAULT_GRAVITY_DECAY: f32 = 0.92;

/// Default minimum composite resonance score required to page in a non-resident anchor (`0.15`).
pub const DEFAULT_PROMOTION_FLOOR: f32 = 0.15;

/// Extracts a branchless 64-bit [`BitCut64`] structural polarity signature from a 64-head Q7
/// phase profile (`inside` = strictly positive heads, `outside` = strictly negative heads,
/// zero heads omitted from both halves).
#[inline]
#[must_use]
pub fn phase_profile_bitcut64(phase_profile: &[i8; MAX_ATTENTION_HEADS]) -> BitCut64 {
    let mut nonzero_mask = 0_u64;
    let mut positive_mask = 0_u64;
    for (idx, &val) in phase_profile.iter().enumerate() {
        let bit = 1_u64 << idx;
        if val != 0 {
            nonzero_mask |= bit;
            if val > 0 {
                positive_mask |= bit;
            }
        }
    }
    BitCut64::split(nonzero_mask, positive_mask)
}

/// Computes normalized coarse polarity similarity in `[-1.0, 1.0]` between two [`BitCut64`]
/// signatures using constant-time bitwise population counts (`0` allocations).
#[inline]
#[must_use]
pub(crate) fn bitcut64_polarity_similarity(a: BitCut64, b: BitCut64) -> f32 {
    let a_active = a.inside | a.outside;
    let b_active = b.inside | b.outside;
    let union_count = (a_active | b_active).count_ones();
    if union_count == 0 {
        return 0.0;
    }
    let agreed = a.intersect(b).count_ones() + (a.outside & b.outside).count_ones();
    let opposed = a.excess_over(b).count_ones() + b.excess_over(a).count_ones();
    (agreed as f32 - opposed as f32) / (union_count as f32)
}

/// Computes exact 64-head Q7 cosine similarity in `[-1.0, 1.0]` for Top-K candidate reranking.
#[inline]
#[must_use]
pub(crate) fn phase_profile_cosine_q7(
    a: &[i8; MAX_ATTENTION_HEADS],
    b: &[i8; MAX_ATTENTION_HEADS],
) -> f32 {
    let mut dot = 0_i64;
    let mut na = 0_i64;
    let mut nb = 0_i64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xi = i64::from(x);
        let yi = i64::from(y);
        dot += xi * yi;
        na += xi * xi;
        nb += yi * yi;
    }
    let denom = ((na as f64) * (nb as f64)).sqrt();
    if denom > 1e-12 {
        (dot as f64 / denom).clamp(-1.0, 1.0) as f32
    } else {
        0.0
    }
}

/// Configuration for [`AnchorWorkingSetPager`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorPagerConfig {
    /// Maximum number of resident anchors kept in the active working set (default `64`).
    pub working_set_capacity: usize,
    /// Number of top coarse [`BitCut64`] candidates forwarded to exact Q7 scoring (default `32`).
    pub prefilter_top_k: usize,
    /// Multiplicative dissipative decay applied to resident excitation each active step (`0.50..=0.999`).
    pub gravity_decay: f32,
    /// Minimum fine resonance score required before promoting an anchor into the working set.
    pub promotion_floor: f32,
}

impl Default for AnchorPagerConfig {
    #[inline]
    fn default() -> Self {
        Self {
            working_set_capacity: DEFAULT_WORKING_SET_CAPACITY,
            prefilter_top_k: DEFAULT_PREFILTER_TOP_K,
            gravity_decay: DEFAULT_GRAVITY_DECAY,
            promotion_floor: DEFAULT_PROMOTION_FLOOR,
        }
    }
}

impl AnchorPagerConfig {
    /// Returns a sanitized copy with all fields clamped to safe operational ranges.
    #[inline]
    #[must_use]
    pub fn sanitized(self) -> Self {
        let decay = if self.gravity_decay.is_finite() {
            self.gravity_decay.clamp(0.50, 0.999)
        } else {
            DEFAULT_GRAVITY_DECAY
        };
        let floor = if self.promotion_floor.is_finite() {
            self.promotion_floor.clamp(-1.0, 1.0)
        } else {
            DEFAULT_PROMOTION_FLOOR
        };
        Self {
            working_set_capacity: self.working_set_capacity.max(1),
            prefilter_top_k: self.prefilter_top_k.max(1),
            gravity_decay: decay,
            promotion_floor: floor,
        }
    }
}

/// Backing-bank entry managed by [`AnchorWorkingSetPager`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnchorSlotEntry {
    /// Underlying 128-byte core [`PhaseAnchorRecord`] (`Pod`).
    pub record: PhaseAnchorRecord,
    /// Branchless 64-bit polarity signature derived from `record.phase_profile`.
    pub signature: BitCut64,
    /// Dissipative working-set excitation updated on hits and decayed each active step.
    pub excitation: f32,
    /// Spine pin floor: when `true`, this entry is guaranteed resident and never evicted.
    pub pinned: bool,
    /// Canonical 4-state [`Lacuna`] classification gating promotion and isolation.
    pub lacuna_state: Lacuna,
    /// Monotonic step counter of the most recent promotion or hit (for deterministic tie-breaking).
    pub last_step: u64,
}

impl AnchorSlotEntry {
    /// Constructs a new [`AnchorSlotEntry`] from a [`PhaseAnchorRecord`].
    #[inline]
    #[must_use]
    pub fn new(record: PhaseAnchorRecord, lacuna_state: Lacuna, pinned: bool) -> Self {
        let signature = phase_profile_bitcut64(&record.phase_profile);
        let init_excitation = record.phase_gravity_mass.max(record.base_energy).max(0.0);
        Self {
            record,
            signature,
            excitation: if init_excitation.is_finite() {
                init_excitation
            } else {
                0.0
            },
            pinned,
            lacuna_state,
            last_step: 0,
        }
    }

    #[inline]
    pub(crate) fn retention_priority(&self) -> f32 {
        let mass = if self.record.phase_gravity_mass.is_finite() {
            self.record.phase_gravity_mass.max(0.0)
        } else {
            0.0
        };
        let exc = if self.excitation.is_finite() {
            self.excitation.max(0.0)
        } else {
            0.0
        };
        exc + 0.25 * mass
    }
}

/// Observable working-set paging statistics maintained by [`AnchorWorkingSetPager`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AnchorPagerStats {
    /// Number of anchors promoted from the backing bank into the resident working set.
    pub page_ins: u64,
    /// Number of unpinned anchors evicted from the resident working set under capacity pressure.
    pub page_outs: u64,
    /// Number of eviction evaluations where a `pinned` anchor was shielded by the Spine pin floor.
    pub pinned_protections: u64,
    /// Number of steps short-circuited with zero swap churn on `Lacuna::Unobserved` / `Undefined`.
    pub unobserved_skips: u64,
    /// Number of `Lacuna::Exterior` candidates withheld from unverified `SelfRef` promotion.
    pub exterior_holds: u64,
    /// Total monotonic [`AnchorWorkingSetPager::step_working_set`] calls executed.
    pub total_steps: u64,
}

/// Abstract domain-agnostic query input for [`AnchorWorkingSetPager::step_working_set`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagerStepQuery {
    /// Target 64-head Q7 phase profile to match against the backing anchor bank.
    pub phase_profile: [i8; MAX_ATTENTION_HEADS],
    /// 4-region [`LacunaPartition8`] governing unobserved zero-churn cuts and boundary tension.
    pub lacuna_partition: LacunaPartition8,
    /// Maximum number of top resonant candidates to ensure resident in this step.
    pub desired_count: usize,
    /// When `true`, `Lacuna::Exterior` anchors may occupy a staged slot while remaining tagged
    /// `Lacuna::Exterior`; when `false`, they are never paged in without explicit settlement.
    pub allow_exterior_staging: bool,
}

impl Default for PagerStepQuery {
    fn default() -> Self {
        Self {
            phase_profile: [0_i8; MAX_ATTENTION_HEADS],
            lacuna_partition: LacunaPartition8::default(),
            desired_count: 4,
            allow_exterior_staging: false,
        }
    }
}

/// Result emitted by [`AnchorWorkingSetPager::step_working_set`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagerStepOutcome {
    /// Anchor IDs paged into the resident working set during this step.
    pub promoted_ids: Vec<u32>,
    /// Anchor IDs evicted from the resident working set during this step.
    pub evicted_ids: Vec<u32>,
    /// Ordered top resonant anchor IDs selected by fine Q7 scoring (up to `desired_count`).
    pub resonant_ids: Vec<u32>,
    /// `true` when `Lacuna::Unobserved` or `Lacuna::Undefined` triggered a zero-churn skip.
    pub skipped_unobserved: bool,
    /// Snapshot of homeostatic [`DistortionTension`] after this paging step.
    pub tension_state: DistortionTension,
}

/// Domain-agnostic bounded working-set pager over arbitrarily large [`PhaseAnchorRecord`] banks.
///
/// Separates coarse SWAR [`BitCut64`] Top-K prefiltering from exact 64-head Q7 scoring, enforces
/// `Lacuna::Unobserved` zero-churn skipping and `Lacuna::Exterior` non-promotion isolation, and
/// protects `pinned` anchors under a deterministic dissipative eviction policy with dense slot
/// compaction.
#[derive(Debug, Clone)]
pub struct AnchorWorkingSetPager {
    config: AnchorPagerConfig,
    topology_crc32: u32,
    backing_bank: Vec<AnchorSlotEntry>,
    id_to_bank_idx: HashMap<u32, usize>,
    resident_indices: Vec<usize>,
    resident_flags: Vec<bool>,
    step_counter: u64,
    tension: DistortionTension,
    stats: AnchorPagerStats,
}

impl AnchorWorkingSetPager {
    /// Creates an empty [`AnchorWorkingSetPager`] bound to the active [`CORE_TOPOLOGY`] CRC32.
    #[must_use]
    pub fn new(config: AnchorPagerConfig) -> Self {
        let cfg = config.sanitized();
        Self {
            config: cfg,
            topology_crc32: CORE_TOPOLOGY.topology_crc32,
            backing_bank: Vec::new(),
            id_to_bank_idx: HashMap::new(),
            resident_indices: Vec::with_capacity(cfg.working_set_capacity),
            resident_flags: Vec::new(),
            step_counter: 0,
            tension: DistortionTension::new(true),
            stats: AnchorPagerStats::default(),
        }
    }

    /// Creates an empty [`AnchorWorkingSetPager`] bound to an explicit `topology_crc32`.
    #[must_use]
    pub(crate) fn with_topology_crc32(config: AnchorPagerConfig, topology_crc32: u32) -> Self {
        let mut pager = Self::new(config);
        pager.topology_crc32 = topology_crc32;
        pager
    }

    /// Returns a read-only slice over all entries currently held in the backing bank.
    #[inline]
    #[must_use]
    pub fn backing_entries(&self) -> &[AnchorSlotEntry] {
        &self.backing_bank
    }

    /// Replaces the backing bank atomically and recomputes dense resident working-set slots.
    ///
    /// All `pinned` entries are placed into the resident working set first (failing closed if
    /// pinned count exceeds `config.working_set_capacity`), followed by the highest-retention
    /// `Lacuna::SelfRef` entries up to capacity. `Lacuna::Exterior` and `Unobserved` entries are
    /// never promoted into `SelfRef` residency here.
    pub(crate) fn replace_backing_entries(
        &mut self,
        mut entries: Vec<AnchorSlotEntry>,
    ) -> Result<(), String> {
        let pinned_count = entries.iter().filter(|e| e.pinned).count();
        if pinned_count > self.config.working_set_capacity {
            return Err(format!(
                "pinned anchor count {pinned_count} exceeds working set capacity {}",
                self.config.working_set_capacity
            ));
        }
        for entry in &mut entries {
            entry.signature = phase_profile_bitcut64(&entry.record.phase_profile);
            if !entry.excitation.is_finite() || entry.excitation < 0.0 {
                entry.excitation = 0.0;
            }
            if entry.record.checksum_crc32 == 0 {
                entry.record.stamp_crc32();
            }
        }
        let mut id_map = HashMap::with_capacity(entries.len());
        let mut deduped = Vec::with_capacity(entries.len());
        for entry in entries {
            let aid = entry.record.anchor_id;
            if let Some(&pos) = id_map.get(&aid) {
                deduped[pos] = entry;
            } else {
                let pos = deduped.len();
                id_map.insert(aid, pos);
                deduped.push(entry);
            }
        }
        let mut resident_flags = vec![false; deduped.len()];
        let mut resident_indices = Vec::with_capacity(self.config.working_set_capacity);

        // 1. Place Spine-pinned anchors first in deterministic anchor_id order.
        let mut pinned_order: Vec<(usize, u32)> = deduped
            .iter()
            .enumerate()
            .filter(|(_, e)| e.pinned)
            .map(|(idx, e)| (idx, e.record.anchor_id))
            .collect();
        pinned_order.sort_by_key(|&(_, id)| id);
        for (idx, _) in pinned_order {
            resident_flags[idx] = true;
            resident_indices.push(idx);
        }

        // 2. Fill remaining working-set capacity with highest-retention SelfRef entries.
        let remaining = self
            .config
            .working_set_capacity
            .saturating_sub(resident_indices.len());
        if remaining > 0 {
            let mut ranked: Vec<(usize, f32, u64, u32)> = deduped
                .iter()
                .enumerate()
                .filter(|(idx, e)| {
                    !resident_flags[*idx] && matches!(e.lacuna_state, Lacuna::SelfRef)
                })
                .map(|(idx, e)| (idx, e.retention_priority(), e.last_step, e.record.anchor_id))
                .collect();
            ranked.sort_by(|a, b| {
                b.1.total_cmp(&a.1)
                    .then_with(|| b.2.cmp(&a.2))
                    .then_with(|| a.3.cmp(&b.3))
            });
            for (idx, _, _, _) in ranked.into_iter().take(remaining) {
                resident_flags[idx] = true;
                resident_indices.push(idx);
            }
        }

        self.backing_bank = deduped;
        self.id_to_bank_idx = id_map;
        self.resident_flags = resident_flags;
        self.resident_indices = resident_indices;
        Ok(())
    }

    /// Feeds homeostatic churn pressure back into the pager's [`DistortionTension`] controller.
    pub(crate) fn record_tension_feedback(&mut self, churn_q7: u8, boundary_sens_q7: u8) {
        self.tension.step_hot_path_branchless(
            churn_q7.min(127),
            boundary_sens_q7.min(127),
            0,
            0,
            false,
        );
    }

    /// Returns the active sanitized [`AnchorPagerConfig`].
    #[inline]
    #[must_use]
    pub const fn config(&self) -> AnchorPagerConfig {
        self.config
    }

    /// Returns the container topology CRC32 bound to this pager.
    #[inline]
    #[must_use]
    pub const fn topology_crc32(&self) -> u32 {
        self.topology_crc32
    }

    /// Returns the total number of anchors held in the backing bank.
    #[inline]
    #[must_use]
    pub fn backing_len(&self) -> usize {
        self.backing_bank.len()
    }

    /// Returns the current number of resident anchors (`<= config.working_set_capacity`).
    #[inline]
    #[must_use]
    pub fn resident_len(&self) -> usize {
        self.resident_indices.len()
    }

    /// Returns the number of currently `pinned` resident anchors.
    #[must_use]
    pub fn pinned_len(&self) -> usize {
        self.resident_indices
            .iter()
            .filter(|&&idx| self.backing_bank[idx].pinned)
            .count()
    }

    /// Returns cumulative paging telemetry counters.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> AnchorPagerStats {
        self.stats
    }

    /// Returns the current homeostatic [`DistortionTension`] controller state.
    #[inline]
    #[must_use]
    pub const fn tension(&self) -> DistortionTension {
        self.tension
    }

    /// Returns `true` if `anchor_id` is currently resident in the bounded working set.
    #[must_use]
    pub fn is_resident(&self, anchor_id: u32) -> bool {
        self.id_to_bank_idx
            .get(&anchor_id)
            .copied()
            .is_some_and(|idx| self.resident_flags[idx])
    }

    /// Looks up a backing-bank entry by `anchor_id`.
    #[must_use]
    pub fn get_entry(&self, anchor_id: u32) -> Option<&AnchorSlotEntry> {
        let idx = *self.id_to_bank_idx.get(&anchor_id)?;
        self.backing_bank.get(idx)
    }

    /// Iterates over all resident [`AnchorSlotEntry`] items in dense slot order.
    pub fn resident_entries(&self) -> impl Iterator<Item = &AnchorSlotEntry> {
        self.resident_indices
            .iter()
            .map(|&idx| &self.backing_bank[idx])
    }

    /// Materializes a contiguous snapshot of resident [`PhaseAnchorRecord`] entries (`<= capacity`).
    #[must_use]
    pub fn resident_records(&self) -> Vec<PhaseAnchorRecord> {
        self.resident_indices
            .iter()
            .map(|&idx| self.backing_bank[idx].record)
            .collect()
    }

    /// Loads an entire [`LmnkContainerView`] anchor bank (supporting arbitrarily many `> 64`
    /// [`PhaseAnchorRecord`] entries) into a fresh [`AnchorWorkingSetPager`], pre-warming the
    /// bounded working set with the highest-mass anchors up to `config.working_set_capacity`.
    ///
    /// # Errors
    /// Returns an error if container parsing fails or `topology_crc32` mismatches [`CORE_TOPOLOGY`].
    pub fn from_container_bytes(bytes: &[u8], config: AnchorPagerConfig) -> Result<Self, String> {
        let view = LmnkContainerView::parse(bytes).map_err(|status| {
            if status == LMNK_ERR_TOPOLOGY_MISMATCH {
                format!(
                    "container topology mismatch against active CORE_TOPOLOGY (0x{:08x})",
                    CORE_TOPOLOGY.topology_crc32
                )
            } else {
                format!(
                    "container validation failed: {} ({})",
                    status.as_str(),
                    status.code()
                )
            }
        })?;
        let count = view.phase_anchor_count();
        let mut pager = Self::new(config);
        pager.topology_crc32 = view.header().topology_crc32;
        pager.backing_bank.reserve(count);
        pager.resident_flags.reserve(count);

        for i in 0..count {
            if let Some(rec) = view.read_phase_anchor(i) {
                pager.insert_anchor_backing_only(rec, Lacuna::SelfRef, false);
            }
        }
        pager.prewarm_by_gravity();
        Ok(pager)
    }

    /// Constructs a pager directly from a slice of [`PhaseAnchorRecord`] entries (e.g., distilled
    /// in memory), pre-warming the working set by gravitational mass up to capacity.
    #[must_use]
    pub fn from_records(records: &[PhaseAnchorRecord], config: AnchorPagerConfig) -> Self {
        let mut pager = Self::new(config);
        pager.backing_bank.reserve(records.len());
        pager.resident_flags.reserve(records.len());
        for &rec in records {
            pager.insert_anchor_backing_only(rec, Lacuna::SelfRef, false);
        }
        pager.prewarm_by_gravity();
        pager
    }

    /// Serializes the full backing anchor bank into a 64B-aligned `.lmnkc` binary container
    /// validated by [`LmnkContainerView`].
    ///
    /// # Errors
    /// Returns an error if the backing bank is empty or container sizing/serialization fails.
    pub fn to_container_bytes(&self, timestamp_epoch_sec: u64) -> Result<Vec<u8>, String> {
        if self.backing_bank.is_empty() {
            return Err("cannot serialize empty AnchorWorkingSetPager backing bank".to_string());
        }
        let mut records: Vec<PhaseAnchorRecord> = Vec::with_capacity(self.backing_bank.len());
        for slot in &self.backing_bank {
            let mut rec = slot.record;
            rec.stamp_crc32();
            records.push(rec);
        }
        let req_len =
            LmnkContainerView::required_standard_byte_len(records.len(), 0, false, false, 0)
                .map_err(|e| format!("required_standard_byte_len failed: {e}"))?;
        let mut out = vec![0_u8; req_len];
        let written = LmnkContainerView::write_standard_sections_to_bytes(
            liminika_core::StandardSectionsInput {
                container_flags: LMNK_CONTAINER_FLAG_COMPILED_CACHE,
                anchors: &records,
                trails: &[],
                bloom: None,
                ledger: None,
                provenance_reports: &[],
                timestamp_epoch_sec,
            },
            &mut out,
        )
        .map_err(|e| format!("write_standard_sections_to_bytes failed: {e}"))?;
        out.truncate(written);
        Ok(out)
    }

    fn insert_anchor_backing_only(
        &mut self,
        mut record: PhaseAnchorRecord,
        lacuna_state: Lacuna,
        pinned: bool,
    ) -> usize {
        if record.checksum_crc32 == 0 {
            record.stamp_crc32();
        }
        let entry = AnchorSlotEntry::new(record, lacuna_state, pinned);
        if let Some(&existing_idx) = self.id_to_bank_idx.get(&record.anchor_id) {
            self.backing_bank[existing_idx] = entry;
            existing_idx
        } else {
            let idx = self.backing_bank.len();
            self.backing_bank.push(entry);
            self.resident_flags.push(false);
            self.id_to_bank_idx.insert(record.anchor_id, idx);
            idx
        }
    }

    fn prewarm_by_gravity(&mut self) {
        let mut ranked: Vec<(usize, f32, u32)> = self
            .backing_bank
            .iter()
            .enumerate()
            .filter(|(_, e)| matches!(e.lacuna_state, Lacuna::SelfRef))
            .map(|(idx, e)| (idx, e.retention_priority(), e.record.anchor_id))
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.2.cmp(&b.2)));
        for (idx, _, _) in ranked.into_iter().take(self.config.working_set_capacity) {
            if !self.resident_flags[idx] {
                self.resident_flags[idx] = true;
                self.resident_indices.push(idx);
            }
        }
    }

    /// Inserts or updates an anchor in the backing bank.
    ///
    /// When `pinned` is `true`, the anchor is guaranteed residency in the working set (evicting
    /// the lowest-priority unpinned slot if necessary). Fails closed if all working-set slots are
    /// already pinned so a pinned anchor is never evicted.
    ///
    /// # Errors
    /// Returns an error if `pinned` is `true` and the working set is completely saturated with
    /// other pinned anchors.
    pub fn insert_anchor(
        &mut self,
        record: PhaseAnchorRecord,
        lacuna_state: Lacuna,
        pinned: bool,
    ) -> Result<(), String> {
        let idx = self.insert_anchor_backing_only(record, lacuna_state, pinned);
        if pinned {
            if !self.resident_flags[idx] {
                if self.resident_indices.len() < self.config.working_set_capacity {
                    self.resident_flags[idx] = true;
                    self.resident_indices.push(idx);
                    self.stats.page_ins = self.stats.page_ins.saturating_add(1);
                } else if let Some(evicted_id) = self.evict_one_unpinned_and_insert(idx) {
                    let _ = evicted_id;
                } else {
                    self.backing_bank[idx].pinned = false;
                    return Err(format!(
                        "working set capacity {} is fully saturated with pinned anchors",
                        self.config.working_set_capacity
                    ));
                }
            }
        } else if matches!(lacuna_state, Lacuna::SelfRef)
            && !self.resident_flags[idx]
            && self.resident_indices.len() < self.config.working_set_capacity
        {
            self.resident_flags[idx] = true;
            self.resident_indices.push(idx);
            self.stats.page_ins = self.stats.page_ins.saturating_add(1);
        }
        Ok(())
    }

    /// Updates the Spine pin-floor flag on an existing anchor in the backing bank.
    ///
    /// Pinning an anchor ensures it becomes resident immediately (evicting an unpinned anchor if
    /// needed) and cannot be evicted while pinned.
    ///
    /// # Errors
    /// Returns an error if `anchor_id` does not exist or if pinning exceeds working-set capacity
    /// because all slots are already pinned.
    pub fn set_pinned(&mut self, anchor_id: u32, pinned: bool) -> Result<(), String> {
        let Some(&idx) = self.id_to_bank_idx.get(&anchor_id) else {
            return Err(format!("anchor_id {anchor_id} not found in backing bank"));
        };
        if !pinned {
            self.backing_bank[idx].pinned = false;
            return Ok(());
        }
        if self.resident_flags[idx] {
            self.backing_bank[idx].pinned = true;
            return Ok(());
        }
        if self.resident_indices.len() < self.config.working_set_capacity {
            self.backing_bank[idx].pinned = true;
            self.resident_flags[idx] = true;
            self.resident_indices.push(idx);
            self.stats.page_ins = self.stats.page_ins.saturating_add(1);
            return Ok(());
        }
        if self.evict_one_unpinned_and_insert(idx).is_some() {
            self.backing_bank[idx].pinned = true;
            Ok(())
        } else {
            Err(format!(
                "cannot pin anchor_id {anchor_id}: all {} working-set slots are pinned",
                self.config.working_set_capacity
            ))
        }
    }

    /// Explicitly settles a `Lacuna::Exterior` anchor into `Lacuna::SelfRef` eligibility.
    ///
    /// This is the sole state transition allowing an isolated exterior anchor to enter normal
    /// `SelfRef` working-set promotion.
    ///
    /// # Errors
    /// Returns an error if `anchor_id` is absent or marked `Lacuna::Undefined`.
    pub fn settle_exterior_anchor(&mut self, anchor_id: u32) -> Result<(), String> {
        let Some(&idx) = self.id_to_bank_idx.get(&anchor_id) else {
            return Err(format!("anchor_id {anchor_id} not found in backing bank"));
        };
        if matches!(self.backing_bank[idx].lacuna_state, Lacuna::Undefined) {
            return Err(format!(
                "cannot settle terminal Lacuna::Undefined anchor_id {anchor_id}"
            ));
        }
        self.backing_bank[idx].lacuna_state = Lacuna::SelfRef;
        Ok(())
    }

    fn select_eviction_victim(&mut self) -> Option<usize> {
        let mut best_pos: Option<usize> = None;
        let mut best_key: (f32, u64, u32) = (f32::INFINITY, u64::MAX, u32::MAX);
        let mut saw_pinned = false;

        for (slot_pos, &bank_idx) in self.resident_indices.iter().enumerate() {
            let entry = &self.backing_bank[bank_idx];
            if entry.pinned {
                saw_pinned = true;
                continue;
            }
            let key = (
                entry.retention_priority(),
                entry.last_step,
                entry.record.anchor_id,
            );
            let is_lower = match key.0.total_cmp(&best_key.0) {
                std::cmp::Ordering::Less => true,
                std::cmp::Ordering::Equal => (key.1, key.2) < (best_key.1, best_key.2),
                std::cmp::Ordering::Greater => false,
            };
            if best_pos.is_none() || is_lower {
                best_pos = Some(slot_pos);
                best_key = key;
            }
        }
        if saw_pinned {
            self.stats.pinned_protections = self.stats.pinned_protections.saturating_add(1);
        }
        best_pos
    }

    fn evict_one_unpinned_and_insert(&mut self, incoming_bank_idx: usize) -> Option<u32> {
        let victim_slot_pos = self.select_eviction_victim()?;
        // Compact resident slots densely by removing victim and appending incoming.
        let victim_bank_idx = self.resident_indices.remove(victim_slot_pos);
        self.resident_flags[victim_bank_idx] = false;
        let evicted_id = self.backing_bank[victim_bank_idx].record.anchor_id;

        self.resident_flags[incoming_bank_idx] = true;
        self.resident_indices.push(incoming_bank_idx);
        self.stats.page_outs = self.stats.page_outs.saturating_add(1);
        self.stats.page_ins = self.stats.page_ins.saturating_add(1);
        Some(evicted_id)
    }

    /// Executes one bounded working-set paging step against `query`.
    ///
    /// 1. Short-circuits with zero swap churn on `Lacuna::Unobserved` or `Lacuna::Undefined`.
    /// 2. Applies dissipative gravity decay to resident unpinned/pinned excitations.
    /// 3. Runs coarse $O(N)$ SWAR [`BitCut64`] Top-K prefiltering over the backing bank.
    /// 4. Reranks only Top-K candidates via exact 64-head Q7 cosine similarity.
    /// 5. Enforces `Lacuna::Exterior` non-promotion and `Spine` pin-floor eviction protection.
    #[must_use]
    pub fn step_working_set(&mut self, query: &PagerStepQuery) -> PagerStepOutcome {
        self.step_counter = self.step_counter.saturating_add(1);
        self.stats.total_steps = self.stats.total_steps.saturating_add(1);

        if query.lacuna_partition.is_unobserved_aperture_zero()
            || query.lacuna_partition.has_undefined_bottom()
            || self.backing_bank.is_empty()
            || query.desired_count == 0
        {
            self.stats.unobserved_skips = self.stats.unobserved_skips.saturating_add(1);
            return PagerStepOutcome {
                promoted_ids: Vec::new(),
                evicted_ids: Vec::new(),
                resonant_ids: Vec::new(),
                skipped_unobserved: true,
                tension_state: self.tension,
            };
        }

        // Step 1: dissipative gravity decay across resident slots.
        let decay = self.config.gravity_decay;
        for &bank_idx in &self.resident_indices {
            self.backing_bank[bank_idx].excitation *= decay;
        }

        // Step 2: coarse SWAR BitCut64 prefilter across the full backing bank (O(N * log K) / O(N)).
        let query_sig = phase_profile_bitcut64(&query.phase_profile);
        let top_k = self
            .config
            .prefilter_top_k
            .max(query.desired_count)
            .min(self.backing_bank.len());
        let mut coarse_candidates: Vec<(usize, f32, u32)> = Vec::with_capacity(top_k + 1);
        let mut min_coarse_pos = 0_usize;
        let mut min_coarse_score = f32::INFINITY;

        for (bank_idx, entry) in self.backing_bank.iter().enumerate() {
            if matches!(entry.lacuna_state, Lacuna::Unobserved | Lacuna::Undefined) {
                continue;
            }
            let sig_sim = bitcut64_polarity_similarity(query_sig, entry.signature);
            let grav_bonus = (entry.record.phase_gravity_mass.max(0.0) * 0.05).min(0.25);
            let coarse = sig_sim + grav_bonus;
            let anchor_id = entry.record.anchor_id;

            if coarse_candidates.len() < top_k {
                coarse_candidates.push((bank_idx, coarse, anchor_id));
                if coarse_candidates.len() == top_k {
                    min_coarse_pos = 0;
                    min_coarse_score = coarse_candidates[0].1;
                    for (p, &(_, sc, _)) in coarse_candidates.iter().enumerate().skip(1) {
                        if sc < min_coarse_score {
                            min_coarse_score = sc;
                            min_coarse_pos = p;
                        }
                    }
                }
            } else if coarse > min_coarse_score {
                coarse_candidates[min_coarse_pos] = (bank_idx, coarse, anchor_id);
                min_coarse_pos = 0;
                min_coarse_score = coarse_candidates[0].1;
                for (p, &(_, sc, _)) in coarse_candidates.iter().enumerate().skip(1) {
                    if sc < min_coarse_score {
                        min_coarse_score = sc;
                        min_coarse_pos = p;
                    }
                }
            }
        }

        // Step 3: exact 64-head Q7 scoring strictly over Top-K coarse candidates.
        let mut fine_ranked: Vec<(usize, f32, u32)> = Vec::with_capacity(coarse_candidates.len());
        for (bank_idx, _, anchor_id) in coarse_candidates {
            let entry = &self.backing_bank[bank_idx];
            let cos_q7 = phase_profile_cosine_q7(&query.phase_profile, &entry.record.phase_profile);
            let grav = (entry.record.phase_gravity_mass.max(0.0) * 0.04).min(0.20);
            let fine = cos_q7 + grav;
            fine_ranked.push((bank_idx, fine, anchor_id));
        }
        fine_ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.2.cmp(&b.2)));

        // Step 4: working-set promotion with Lacuna::Exterior isolation and Spine pin protection.
        let mut promoted_ids = Vec::new();
        let mut evicted_ids = Vec::new();
        let mut resonant_ids = Vec::new();

        for (bank_idx, fine_score, anchor_id) in fine_ranked {
            if resonant_ids.len() >= query.desired_count {
                break;
            }
            if fine_score < self.config.promotion_floor {
                break;
            }
            let lacuna = self.backing_bank[bank_idx].lacuna_state;
            if matches!(lacuna, Lacuna::Exterior) && !query.allow_exterior_staging {
                self.stats.exterior_holds = self.stats.exterior_holds.saturating_add(1);
                continue;
            }

            resonant_ids.push(anchor_id);
            let step_now = self.step_counter;
            let boost = fine_score.max(0.0);

            if self.resident_flags[bank_idx] {
                let entry = &mut self.backing_bank[bank_idx];
                entry.excitation = (entry.excitation + boost).min(16.0);
                entry.last_step = step_now;
                continue;
            }

            if self.resident_indices.len() < self.config.working_set_capacity {
                self.resident_flags[bank_idx] = true;
                self.resident_indices.push(bank_idx);
                self.stats.page_ins = self.stats.page_ins.saturating_add(1);
                let entry = &mut self.backing_bank[bank_idx];
                entry.excitation = (entry.excitation + boost).min(16.0);
                entry.last_step = step_now;
                promoted_ids.push(anchor_id);
            } else if let Some(evicted_id) = self.evict_one_unpinned_and_insert(bank_idx) {
                let entry = &mut self.backing_bank[bank_idx];
                entry.excitation = (entry.excitation + boost).min(16.0);
                entry.last_step = step_now;
                promoted_ids.push(anchor_id);
                evicted_ids.push(evicted_id);
            }
        }

        // Step 5: reflect working-set churn in homeostatic DistortionTension.
        let churn_q7 =
            ((promoted_ids.len() * 127) / self.config.working_set_capacity.max(1)).min(127) as u8;
        self.tension.step_hot_path_branchless(
            churn_q7,
            query.lacuna_partition.boundary_sensitivity_q7,
            0,
            0,
            false,
        );

        PagerStepOutcome {
            promoted_ids,
            evicted_ids,
            resonant_ids,
            skipped_unobserved: false,
            tension_state: self.tension,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use liminika_core::liminal::evaluate_lacuna_partition_branchless;
    use liminika_core::stigmergy::ATTRACTOR_SLOT_KNOWLEDGE;

    fn make_record(anchor_id: u32, seed: u32, mass: f32) -> PhaseAnchorRecord {
        let mut rec = PhaseAnchorRecord::new_empty(anchor_id, ATTRACTOR_SLOT_KNOWLEDGE, 0);
        rec.base_energy = mass;
        rec.phase_gravity_mass = mass;
        rec.sample_count = 4;
        let mut state = (u64::from(seed) << 32) ^ u64::from(anchor_id).wrapping_mul(0x9E37_79B9);
        for dst in &mut rec.phase_profile {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            let val = ((z >> 56) as i16) - 128;
            *dst = val.clamp(-127, 127) as i8;
        }
        rec.stamp_crc32();
        rec
    }

    fn self_ref_partition() -> LacunaPartition8 {
        evaluate_lacuna_partition_branchless(0xFF, &[96; 8], &[16; 8], 32)
    }

    fn unobserved_partition() -> LacunaPartition8 {
        evaluate_lacuna_partition_branchless(0x00, &[0; 8], &[16; 8], 32)
    }

    #[test]
    fn pages_beyond_64_respects_capacity_and_determinism() {
        let records: Vec<PhaseAnchorRecord> = (1..=512)
            .map(|id| make_record(id, id * 7 + 13, 1.0 + (id % 5) as f32 * 0.1))
            .collect();

        let mut pager_a =
            AnchorWorkingSetPager::from_records(&records, AnchorPagerConfig::default());
        let mut pager_b =
            AnchorWorkingSetPager::from_records(&records, AnchorPagerConfig::default());
        assert_eq!(pager_a.backing_len(), 512);
        assert_eq!(pager_a.resident_len(), 64);

        for target_id in [100_u32, 250, 400, 512, 77, 333] {
            let q_profile = records[(target_id - 1) as usize].phase_profile;
            let q = PagerStepQuery {
                phase_profile: q_profile,
                lacuna_partition: self_ref_partition(),
                desired_count: 4,
                allow_exterior_staging: false,
            };
            let out_a = pager_a.step_working_set(&q);
            let out_b = pager_b.step_working_set(&q);
            assert_eq!(out_a, out_b);
            assert!(!out_a.skipped_unobserved);
            assert!(pager_a.resident_len() <= 64);
            assert!(
                pager_a.is_resident(target_id),
                "target anchor {target_id} should be paged into working set"
            );
            assert_eq!(out_a.resonant_ids[0], target_id);
        }
        assert!(pager_a.stats().page_ins > 0);
        assert!(pager_a.stats().page_outs > 0);
    }

    #[test]
    fn pinned_spine_floor_never_evicts_under_pressure() {
        let cfg = AnchorPagerConfig {
            working_set_capacity: 8,
            prefilter_top_k: 32,
            gravity_decay: 0.70,
            promotion_floor: 0.10,
        };
        let mut pager = AnchorWorkingSetPager::new(cfg);
        let pinned_ids = [1_u32, 2, 3];
        for id in 1..=256_u32 {
            let rec = make_record(id, id * 17 + 5, if id <= 3 { 0.05 } else { 2.0 });
            pager
                .insert_anchor(rec, Lacuna::SelfRef, pinned_ids.contains(&id))
                .expect("insert");
        }
        assert_eq!(pager.resident_len(), 8);
        assert_eq!(pager.pinned_len(), 3);

        for target_id in 20..=120_u32 {
            let q_profile = pager.get_entry(target_id).unwrap().record.phase_profile;
            let _ = pager.step_working_set(&PagerStepQuery {
                phase_profile: q_profile,
                lacuna_partition: self_ref_partition(),
                desired_count: 4,
                allow_exterior_staging: false,
            });
            assert!(pager.resident_len() <= 8);
            for &pid in &pinned_ids {
                assert!(
                    pager.is_resident(pid),
                    "pinned anchor {pid} was evicted at step targeting {target_id}"
                );
            }
        }
        assert!(pager.stats().pinned_protections > 0);
    }

    #[test]
    fn unobserved_zero_churn_and_exterior_isolation() {
        let mut pager = AnchorWorkingSetPager::new(AnchorPagerConfig {
            working_set_capacity: 4,
            ..AnchorPagerConfig::default()
        });
        for id in 1..=8_u32 {
            pager
                .insert_anchor(make_record(id, id * 31, 1.0), Lacuna::SelfRef, false)
                .expect("insert self_ref");
        }
        let ext_rec = make_record(999, 999 * 31, 3.0);
        pager
            .insert_anchor(ext_rec, Lacuna::Exterior, false)
            .expect("insert exterior");
        assert!(!pager.is_resident(999));

        // 1. Unobserved query performs zero swaps.
        let stats_before = pager.stats();
        let unobs_out = pager.step_working_set(&PagerStepQuery {
            phase_profile: ext_rec.phase_profile,
            lacuna_partition: unobserved_partition(),
            desired_count: 2,
            allow_exterior_staging: false,
        });
        assert!(unobs_out.skipped_unobserved);
        assert!(unobs_out.promoted_ids.is_empty());
        assert!(unobs_out.evicted_ids.is_empty());
        assert_eq!(pager.stats().page_ins, stats_before.page_ins);
        assert_eq!(pager.stats().page_outs, stats_before.page_outs);
        assert_eq!(pager.stats().unobserved_skips, 1);

        // 2. Exterior anchor is never auto-promoted to SelfRef without explicit settlement.
        let ext_out = pager.step_working_set(&PagerStepQuery {
            phase_profile: ext_rec.phase_profile,
            lacuna_partition: self_ref_partition(),
            desired_count: 2,
            allow_exterior_staging: false,
        });
        assert!(!ext_out.promoted_ids.contains(&999));
        assert!(!pager.is_resident(999));
        assert!(pager.stats().exterior_holds >= 1);
        assert_eq!(pager.get_entry(999).unwrap().lacuna_state, Lacuna::Exterior);

        // 3. After explicit settlement, anchor 999 becomes eligible for SelfRef promotion.
        pager.settle_exterior_anchor(999).expect("settle exterior");
        assert_eq!(pager.get_entry(999).unwrap().lacuna_state, Lacuna::SelfRef);
        let settled_out = pager.step_working_set(&PagerStepQuery {
            phase_profile: ext_rec.phase_profile,
            lacuna_partition: self_ref_partition(),
            desired_count: 2,
            allow_exterior_staging: false,
        });
        assert!(settled_out.promoted_ids.contains(&999));
        assert!(pager.is_resident(999));
    }

    #[test]
    fn container_roundtrip_beyond_64_anchors() {
        let records: Vec<PhaseAnchorRecord> = (1..=300)
            .map(|id| make_record(id, id * 29 + 3, 1.25))
            .collect();
        let pager = AnchorWorkingSetPager::from_records(&records, AnchorPagerConfig::default());
        let bytes = pager.to_container_bytes(1_700_000_000).expect("serialize");

        let mut restored =
            AnchorWorkingSetPager::from_container_bytes(&bytes, AnchorPagerConfig::default())
                .expect("restore");
        assert_eq!(restored.backing_len(), 300);
        assert_eq!(restored.resident_len(), 64);
        assert_eq!(restored.topology_crc32(), CORE_TOPOLOGY.topology_crc32);

        let target = records[280];
        let out = restored.step_working_set(&PagerStepQuery {
            phase_profile: target.phase_profile,
            lacuna_partition: self_ref_partition(),
            desired_count: 2,
            allow_exterior_staging: false,
        });
        assert_eq!(out.resonant_ids[0], target.anchor_id);
        assert!(restored.is_resident(target.anchor_id));
    }
}
