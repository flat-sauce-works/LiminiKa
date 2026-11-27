// name: src/adapter/src/host/equilibrium.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Domain-agnostic persistent equilibrium state ([`EquilibriumState`]) implementing flat monadic
//! renormalization (`join: M<M<A>> -> M<A>`) within a matching `CORE_TOPOLOGY` hash, bounded
//! working-set plasticity, `LMNK_SECTION_RUNTIME_SNAPSHOT` persistence, and monotonic `Spine`
//! pin-floor protection.

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use liminika_core::drivers::{
    EnvelopeBounds, EnvelopeState, HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
    SPINE_BYPASS_VETO_KNOWN_MASK, SPINE_CONTROL_KNOWN_FLAGS, SPINE_ENTROPY_Q15_ONE,
    SystemSpineState,
};
use liminika_core::foundation::Pod;
use liminika_core::liminal::{
    LMNK_ERR_TOPOLOGY_MISMATCH, Lacuna, LacunaPartition8, MAX_ATTENTION_HEADS,
};
use liminika_core::runtime::{
    CORE_TOPOLOGY, LMNK_CONTAINER_FLAG_COMPILED_CACHE, LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT,
    LMNK_MAX_CONTAINER_BYTES, LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES,
    LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES, LMNK_SECTION_PHASE_ANCHORS, LMNK_SECTION_RUNTIME_SNAPSHOT,
    LmnkContainerHeader, LmnkContainerView, LmnkSectionInput, required_container_byte_len,
    write_container_sections_to_bytes,
};
use liminika_core::stigmergy::PhaseAnchorRecord;

use crate::host::integrity::{
    IntegrityPolicy, SigningKeyPair, sign_lmnkc_container, verify_lmnkc_integrity,
};
use crate::host::pager::{
    AnchorPagerConfig, AnchorSlotEntry, AnchorWorkingSetPager, PagerStepOutcome, PagerStepQuery,
    bitcut64_polarity_similarity, phase_profile_bitcut64, phase_profile_cosine_q7,
};
use crate::host::path_guard::validate_local_output_path;

/// Custom `.lmnkc` section kind storing versioned host equilibrium metadata (`0x0101`).
///
/// Strictly separated from core standard section IDs (`1..=8`) and reserved security section IDs
/// (`0xFF01`/`0xFF02`). Containers lacking this section remain 100% backward compatible.
pub const HOST_EQUILIBRIUM_META_SECTION_KIND: u16 = 0x0101;

const EQUILIBRIUM_META_MAGIC: [u8; 4] = *b"LEQM";
const EQUILIBRIUM_META_VERSION: u32 = 1;
const EQUILIBRIUM_META_HEADER_BYTES: usize = 24;
const EQUILIBRIUM_META_RECORD_BYTES: usize = 20;
const SNAPSHOT_SPINE_PAYLOAD_OFFSET: usize = 320;
const SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET: usize = 352;
const SNAPSHOT_ENVELOPE_BOUNDS_PAYLOAD_OFFSET: usize = 416;
const SNAPSHOT_SLOT_PINNED_MASK_OFFSET: usize = 21;
const SNAPSHOT_STEP_COUNTER_OFFSET: usize = 0;

/// Default maximum total anchors retained in the bounded backing bank (`256`).
pub const DEFAULT_MAX_BACKING_ANCHORS: usize = 256;

/// Default per-generation dissipative gravity decay factor (`0.92`) for unpinned anchors.
pub const DEFAULT_EQUILIBRIUM_DECAY: f32 = 0.92;

/// Default minimum gravity mass (`0.05`) below which unreinforced unpinned anchors evaporate.
pub const DEFAULT_EVAPORATION_FLOOR: f32 = 0.05;

/// Default Q7 cosine threshold (`0.85`) for consolidating near-duplicate anchors on $SO(2)^{64}$.
pub const DEFAULT_MERGE_COHERENCE_THRESHOLD: f32 = 0.85;

/// Default soft upper bound (`4.0`) on anchor `phase_gravity_mass` and `base_energy`.
pub const DEFAULT_MAX_GRAVITY_MASS: f32 = 4.0;

/// Default bounded reinforcement gain (`0.50`) applied when matching anchors coalesce across `join`.
pub const DEFAULT_REINFORCEMENT_GAIN: f32 = 0.50;

/// Default steady-state working-set excitation ceiling (`2.0`) for older reinforced unpinned slots.
pub const DEFAULT_STEADY_EXCITATION_CAP: f32 = 2.0;

/// Default arrival excitation (`3.5`) granted to newly settled anchors so saturated equilibria
/// remain plastic without displacing `Spine`-pinned slots.
pub const DEFAULT_NOVEL_ARRIVAL_EXCITATION: f32 = 3.5;

const MIN_BITCUT_MERGE_PREFILTER: f32 = 0.50;
const MIN_PINNED_GRAVITY_FLOOR: f32 = 0.25;
const MIN_UNPINNED_PLASTICITY_WEIGHT: f64 = 0.15;

/// Structured errors returned by [`EquilibriumState`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EquilibriumError {
    /// Incoming state or `.lmnkc` container belongs to a different `CORE_TOPOLOGY` hash.
    ///
    /// Callers must re-bake from immutable source archives (`.lmnkp`) instead of merging across
    /// incompatible topologies.
    TopologyMismatch {
        /// Expected topology CRC32 of the receiver or active `CORE_TOPOLOGY`.
        expected: u32,
        /// Observed topology CRC32 of the incoming state or container.
        actual: u32,
    },
    /// Container structure, CRC32, ABI version, or equilibrium metadata payload failed validation.
    ContainerInvalid(String),
    /// Embedded `LMNK_SECTION_RUNTIME_SNAPSHOT` bytes failed structural or spine validation.
    SnapshotInvalid(String),
    /// Pinning requirements exceed the bounded resident working-set capacity.
    PinnedCapacityExceeded {
        /// Configured resident working-set capacity.
        capacity: usize,
    },
    /// Filesystem I/O failure during `.lmnkc` persistence or loading.
    Io(String),
}

impl fmt::Display for EquilibriumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TopologyMismatch { expected, actual } => write!(
                f,
                "topology mismatch (rebake from immutable .lmnkp source required): \
                 expected 0x{expected:08x}, actual 0x{actual:08x}"
            ),
            Self::ContainerInvalid(msg) => write!(f, "invalid .lmnkc equilibrium container: {msg}"),
            Self::SnapshotInvalid(msg) => write!(f, "invalid runtime snapshot payload: {msg}"),
            Self::PinnedCapacityExceeded { capacity } => write!(
                f,
                "pinned anchors exceed resident working-set capacity ({capacity})"
            ),
            Self::Io(msg) => write!(f, "equilibrium I/O error: {msg}"),
        }
    }
}

impl std::error::Error for EquilibriumError {}

/// Operational invariants controlling [`EquilibriumState`] flat renormalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquilibriumConfig {
    /// Underlying [`AnchorWorkingSetPager`] configuration.
    pub pager: AnchorPagerConfig,
    /// Hard upper bound on total backing-bank anchors persisted in `.lmnkc` (default `256`).
    pub max_backing_anchors: usize,
    /// Per-generation multiplicative decay applied to unpinned anchors (default `0.92`).
    pub gravity_decay: f32,
    /// Minimum gravity mass below which unpinned anchors evaporate (default `0.05`).
    pub evaporation_floor: f32,
    /// Minimum Q7 cosine coherence required to coalesce near-duplicate anchors (default `0.85`).
    pub merge_coherence_threshold: f32,
    /// Soft upper ceiling on anchor `base_energy` and `phase_gravity_mass` (default `4.0`).
    pub max_gravity_mass: f32,
    /// Bounded reinforcement weight applied during anchor coalescence (default `0.50`).
    pub reinforcement_gain: f32,
    /// Steady-state excitation cap for long-lived unpinned anchors (default `2.0`).
    pub steady_excitation_cap: f32,
    /// Initial arrival excitation for novel unpinned anchors (default `3.5`).
    pub novel_arrival_excitation: f32,
}

impl Default for EquilibriumConfig {
    #[inline]
    fn default() -> Self {
        Self {
            pager: AnchorPagerConfig::default(),
            max_backing_anchors: DEFAULT_MAX_BACKING_ANCHORS,
            gravity_decay: DEFAULT_EQUILIBRIUM_DECAY,
            evaporation_floor: DEFAULT_EVAPORATION_FLOOR,
            merge_coherence_threshold: DEFAULT_MERGE_COHERENCE_THRESHOLD,
            max_gravity_mass: DEFAULT_MAX_GRAVITY_MASS,
            reinforcement_gain: DEFAULT_REINFORCEMENT_GAIN,
            steady_excitation_cap: DEFAULT_STEADY_EXCITATION_CAP,
            novel_arrival_excitation: DEFAULT_NOVEL_ARRIVAL_EXCITATION,
        }
    }
}

impl EquilibriumConfig {
    /// Returns a sanitized copy with all fields clamped to numerically finite, safe ranges.
    #[inline]
    #[must_use]
    pub fn sanitized(self) -> Self {
        let pager = self.pager.sanitized();
        let max_backing_anchors = self.max_backing_anchors.max(pager.working_set_capacity);
        let gravity_decay = if self.gravity_decay.is_finite() {
            self.gravity_decay.clamp(0.50, 0.995)
        } else {
            DEFAULT_EQUILIBRIUM_DECAY
        };
        let evaporation_floor = if self.evaporation_floor.is_finite() {
            self.evaporation_floor.clamp(0.001, 0.50)
        } else {
            DEFAULT_EVAPORATION_FLOOR
        };
        let merge_coherence_threshold = if self.merge_coherence_threshold.is_finite() {
            self.merge_coherence_threshold.clamp(0.50, 0.999)
        } else {
            DEFAULT_MERGE_COHERENCE_THRESHOLD
        };
        let max_gravity_mass = if self.max_gravity_mass.is_finite() {
            self.max_gravity_mass.clamp(1.0, 16.0)
        } else {
            DEFAULT_MAX_GRAVITY_MASS
        };
        let reinforcement_gain = if self.reinforcement_gain.is_finite() {
            self.reinforcement_gain.clamp(0.05, 1.0)
        } else {
            DEFAULT_REINFORCEMENT_GAIN
        };
        let steady_excitation_cap = if self.steady_excitation_cap.is_finite() {
            self.steady_excitation_cap.clamp(0.5, 8.0)
        } else {
            DEFAULT_STEADY_EXCITATION_CAP
        };
        let novel_arrival_excitation = if self.novel_arrival_excitation.is_finite() {
            self.novel_arrival_excitation
                .clamp(steady_excitation_cap + 0.25, 12.0)
        } else {
            (steady_excitation_cap + 1.5).min(12.0)
        };
        Self {
            pager,
            max_backing_anchors,
            gravity_decay,
            evaporation_floor,
            merge_coherence_threshold,
            max_gravity_mass,
            reinforcement_gain,
            steady_excitation_cap,
            novel_arrival_excitation,
        }
    }
}

/// Convergence summary emitted by [`EquilibriumState::join`] and
/// [`EquilibriumState::ingest_crystallized_anchors`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EquilibriumJoinReport {
    /// Generation counter after reconciliation.
    pub generation: u64,
    /// Number of incoming anchors coalesced into existing anchors.
    pub merged_anchors: usize,
    /// Number of novel anchors newly admitted into the backing bank.
    pub inserted_anchors: usize,
    /// Number of unreinforced unpinned anchors evaporated or capacity-pruned.
    pub evaporated_anchors: usize,
    /// Number of `Spine`-pinned anchors preserved intact.
    pub pinned_preserved: usize,
    /// Number of `Lacuna::Exterior` anchors kept isolated without `SelfRef` promotion.
    pub exterior_isolated: usize,
    /// `true` when the embedded 448-byte `RuntimeSnapshot` payload was updated or merged.
    pub snapshot_updated: bool,
    /// Total backing-bank anchors after reconciliation (`<= config.max_backing_anchors`).
    pub backing_len: usize,
    /// Resident working-set anchors after reconciliation (`<= config.pager.working_set_capacity`).
    pub resident_len: usize,
}

/// Domain-agnostic persistent equilibrium state combining a bounded [`AnchorWorkingSetPager`],
/// an optional validated `LMNK_SECTION_RUNTIME_SNAPSHOT` payload, and flat monadic `join`
/// reconciliation within a single `CORE_TOPOLOGY` hash.
#[derive(Debug, Clone)]
pub struct EquilibriumState {
    config: EquilibriumConfig,
    topology_crc32: u32,
    pager: AnchorWorkingSetPager,
    runtime_snapshot: Option<[u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES]>,
    generation: u64,
}

impl EquilibriumState {
    /// Creates an empty [`EquilibriumState`] bound to the active [`CORE_TOPOLOGY`] CRC32.
    #[must_use]
    pub fn new(config: EquilibriumConfig) -> Self {
        let cfg = config.sanitized();
        Self {
            config: cfg,
            topology_crc32: CORE_TOPOLOGY.topology_crc32,
            pager: AnchorWorkingSetPager::new(cfg.pager),
            runtime_snapshot: None,
            generation: 0,
        }
    }

    /// Creates an empty [`EquilibriumState`] bound to an explicit `topology_crc32` (used for
    /// topology-mismatch validation and cross-topology isolation).
    #[must_use]
    pub fn with_topology_crc32(config: EquilibriumConfig, topology_crc32: u32) -> Self {
        let cfg = config.sanitized();
        Self {
            config: cfg,
            topology_crc32,
            pager: AnchorWorkingSetPager::with_topology_crc32(cfg.pager, topology_crc32),
            runtime_snapshot: None,
            generation: 0,
        }
    }

    /// Wraps an existing [`AnchorWorkingSetPager`] into an [`EquilibriumState`].
    #[must_use]
    pub fn from_pager(pager: AnchorWorkingSetPager, config: EquilibriumConfig) -> Self {
        let mut cfg = config.sanitized();
        cfg.pager = pager.config();
        cfg.max_backing_anchors = cfg.max_backing_anchors.max(cfg.pager.working_set_capacity);
        let topology_crc32 = pager.topology_crc32();
        Self {
            config: cfg,
            topology_crc32,
            pager,
            runtime_snapshot: None,
            generation: 0,
        }
    }

    /// Returns the sanitized [`EquilibriumConfig`].
    #[inline]
    #[must_use]
    pub const fn config(&self) -> EquilibriumConfig {
        self.config
    }

    /// Returns the `CORE_TOPOLOGY` CRC32 bound to this equilibrium state.
    #[inline]
    #[must_use]
    pub const fn topology_crc32(&self) -> u32 {
        self.topology_crc32
    }

    /// Returns the monotonic flat renormalization generation counter.
    #[inline]
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns a shared reference to the underlying bounded [`AnchorWorkingSetPager`].
    #[inline]
    #[must_use]
    pub const fn pager(&self) -> &AnchorWorkingSetPager {
        &self.pager
    }

    /// Returns the raw 448-byte `LMNK_SECTION_RUNTIME_SNAPSHOT` payload if present.
    #[inline]
    #[must_use]
    pub const fn runtime_snapshot_payload(
        &self,
    ) -> Option<&[u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES]> {
        self.runtime_snapshot.as_ref()
    }

    /// Wraps the stored 448-byte snapshot payload into a validated 576-byte standalone `.lmnk`
    /// snapshot container compatible with `RuntimeContext::deserialize_from_bytes`.
    #[must_use]
    pub fn runtime_snapshot_container_bytes(
        &self,
    ) -> Option<[u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES]> {
        let payload = self.runtime_snapshot.as_ref()?;
        let sections = [LmnkSectionInput {
            kind: LMNK_SECTION_RUNTIME_SNAPSHOT,
            flags: 0,
            payload,
        }];
        let mut buf = [0_u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
        let written = write_container_sections_to_bytes(
            LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT,
            1_774_900_000,
            &sections,
            &mut buf,
        )
        .ok()?;
        if written == LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES {
            Some(buf)
        } else {
            None
        }
    }

    /// Extracts the persisted [`SystemSpineState`] from the stored snapshot payload, if present.
    #[must_use]
    pub fn snapshot_spine_state(&self) -> Option<SystemSpineState> {
        let payload = self.runtime_snapshot.as_ref()?;
        SystemSpineState::read_unaligned_from(payload, SNAPSHOT_SPINE_PAYLOAD_OFFSET).ok()
    }

    /// Extracts the persisted [`EnvelopeState`] from the stored snapshot payload, if present.
    #[must_use]
    pub fn snapshot_envelope_state(&self) -> Option<EnvelopeState> {
        let payload = self.runtime_snapshot.as_ref()?;
        EnvelopeState::read_unaligned_from(payload, SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET).ok()
    }

    /// Extracts the persisted [`EnvelopeBounds`] from the stored snapshot payload, if present.
    #[must_use]
    pub fn snapshot_envelope_bounds(&self) -> Option<EnvelopeBounds> {
        let payload = self.runtime_snapshot.as_ref()?;
        EnvelopeBounds::read_unaligned_from(payload, SNAPSHOT_ENVELOPE_BOUNDS_PAYLOAD_OFFSET).ok()
    }

    /// Executes one bounded working-set step through the embedded [`AnchorWorkingSetPager`].
    #[must_use]
    pub fn step_working_set(&mut self, query: &PagerStepQuery) -> PagerStepOutcome {
        self.pager.step_working_set(query)
    }

    /// Inserts or updates a single anchor in the underlying pager.
    ///
    /// # Errors
    /// Returns [`EquilibriumError::PinnedCapacityExceeded`] if pinning saturates capacity or
    /// [`EquilibriumError::ContainerInvalid`] for terminal `Lacuna::Undefined` records.
    pub fn insert_anchor(
        &mut self,
        record: PhaseAnchorRecord,
        lacuna_state: Lacuna,
        pinned: bool,
    ) -> Result<(), EquilibriumError> {
        if matches!(lacuna_state, Lacuna::Undefined) {
            return Err(EquilibriumError::ContainerInvalid(
                "cannot insert terminal Lacuna::Undefined anchor".to_string(),
            ));
        }
        self.pager
            .insert_anchor(record, lacuna_state, pinned)
            .map_err(|_| EquilibriumError::PinnedCapacityExceeded {
                capacity: self.config.pager.working_set_capacity,
            })
    }

    /// Updates the Spine pin-floor flag on an existing anchor.
    ///
    /// # Errors
    /// Returns [`EquilibriumError`] if the anchor is absent or working-set slots are saturated.
    pub fn set_pinned(&mut self, anchor_id: u32, pinned: bool) -> Result<(), EquilibriumError> {
        self.pager.set_pinned(anchor_id, pinned).map_err(|msg| {
            if msg.contains("pinned") {
                EquilibriumError::PinnedCapacityExceeded {
                    capacity: self.config.pager.working_set_capacity,
                }
            } else {
                EquilibriumError::ContainerInvalid(msg)
            }
        })
    }

    /// Explicitly settles an isolated `Lacuna::Exterior` anchor into `Lacuna::SelfRef` eligibility.
    ///
    /// # Errors
    /// Returns [`EquilibriumError::ContainerInvalid`] if the anchor is missing or `Undefined`.
    pub fn settle_exterior_anchor(&mut self, anchor_id: u32) -> Result<(), EquilibriumError> {
        self.pager
            .settle_exterior_anchor(anchor_id)
            .map_err(EquilibriumError::ContainerInvalid)
    }

    /// Validates and installs a 576-byte standalone `RuntimeContext` snapshot container (or any
    /// `.lmnkc` container carrying `LMNK_SECTION_RUNTIME_SNAPSHOT`), monotonically preserving any
    /// previously latched [`EnvelopeState`] pin floors.
    ///
    /// # Errors
    /// Returns [`EquilibriumError::TopologyMismatch`] or [`EquilibriumError::SnapshotInvalid`].
    pub fn update_runtime_snapshot_from_container(
        &mut self,
        container_bytes: &[u8],
    ) -> Result<(), EquilibriumError> {
        let view = parse_container_checked(container_bytes, self.topology_crc32)?;
        let raw_payload = view
            .section_bytes(LMNK_SECTION_RUNTIME_SNAPSHOT)
            .ok_or_else(|| {
                EquilibriumError::SnapshotInvalid(
                    "container missing LMNK_SECTION_RUNTIME_SNAPSHOT".to_string(),
                )
            })?;
        let incoming = validate_snapshot_payload(raw_payload)?;
        self.runtime_snapshot = Some(match self.runtime_snapshot {
            Some(prev) => reconcile_snapshot_payloads(&prev, &incoming, true)?,
            None => enforce_immutable_spine_floor(incoming)?,
        });
        Ok(())
    }

    /// Monotonically raises the [`EnvelopeState`] compute pin floor stored inside the embedded
    /// runtime snapshot (if present) so subsequent restores can never weaken host pin guarantees.
    pub fn enforce_snapshot_spine_pin_floor(&mut self, required_pinned_compute_mask: u64) {
        if let Some(mut payload) = self.runtime_snapshot
            && let Ok(mut envelope) =
                EnvelopeState::read_unaligned_from(&payload, SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET)
        {
            envelope.apply_host_pin_floor(
                required_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
            );
            if envelope
                .write_unaligned_to(&mut payload, SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET)
                .is_ok()
            {
                self.runtime_snapshot = Some(payload);
            }
        }
    }

    /// Absorbs a slice of domain-agnostic crystallized [`PhaseAnchorRecord`] entries (e.g.,
    /// drained from a downstream Slow-Loop consolidation queue) via flat renormalization.
    ///
    /// # Errors
    /// Returns [`EquilibriumError`] if Spine-pinned capacity constraints are violated.
    #[must_use = "EquilibriumJoinReport summarizes convergence and bounded working-set updates"]
    pub fn ingest_crystallized_anchors(
        &mut self,
        anchors: &[PhaseAnchorRecord],
    ) -> Result<EquilibriumJoinReport, EquilibriumError> {
        let mut incoming = Self::with_topology_crc32(self.config, self.topology_crc32);
        incoming.generation = self.generation.saturating_add(1);
        for &rec in anchors {
            if !rec.verify_integrity() {
                continue;
            }
            let _ = incoming.insert_anchor(rec, Lacuna::SelfRef, false);
        }
        self.join(&incoming)
    }

    /// Flat monadic renormalization (`join: M<M<A>> -> M<A>`) within a matching `topology_crc32`.
    ///
    /// Guarantees:
    /// 1. **Strict topology boundary**: rejects mismatched `topology_crc32` without mutating `self`.
    /// 2. **Abstract gravity decay & healthy forgetting**: decays unpinned anchors by
    ///    `config.gravity_decay` and evaporates unreinforced entries below `evaporation_floor`.
    /// 3. **Spine pin-floor monotonicity**: `pinned` anchors never decay away or get evicted, and
    ///    merged `RuntimeSnapshot` payloads monotonically OR all `SystemSpineState` pin masks.
    /// 4. **Stable $SO(2)^{64}$ coalescence**: coalesces identical or near-duplicate compatible
    ///    anchors via circular vector synthesis (zero polarity inversion, zero collapse), while
    ///    keeping unsettled `Lacuna::Exterior` anchors isolated from `SelfRef`.
    /// 5. **Bounded non-lock-in plasticity**: caps steady-state mass/excitation and grants novel
    ///    arrivals working-set admission without exceeding `max_backing_anchors` or capacity.
    ///
    /// # Errors
    /// Returns [`EquilibriumError::TopologyMismatch`], [`EquilibriumError::SnapshotInvalid`], or
    /// [`EquilibriumError::PinnedCapacityExceeded`].
    #[must_use = "EquilibriumJoinReport summarizes convergence and bounded working-set updates"]
    pub fn join(&mut self, incoming: &Self) -> Result<EquilibriumJoinReport, EquilibriumError> {
        if self.topology_crc32 != incoming.topology_crc32 {
            return Err(EquilibriumError::TopologyMismatch {
                expected: self.topology_crc32,
                actual: incoming.topology_crc32,
            });
        }

        let next_generation = self.generation.max(incoming.generation).saturating_add(1);
        let cfg = self.config;
        let mut working: Vec<AnchorSlotEntry> = Vec::with_capacity(
            self.pager
                .backing_len()
                .saturating_add(incoming.pager.backing_len()),
        );
        let mut evaporated_anchors = 0_usize;

        // Step 1: Apply dissipative gravity decay to existing backing entries (Spine pins exempt).
        for &entry in self.pager.backing_entries() {
            if matches!(entry.lacuna_state, Lacuna::Undefined) {
                evaporated_anchors += 1;
                continue;
            }
            let mut next = sanitize_slot_entry(entry, cfg);
            if next.pinned {
                next.record.phase_gravity_mass = next
                    .record
                    .phase_gravity_mass
                    .clamp(MIN_PINNED_GRAVITY_FLOOR, cfg.max_gravity_mass);
                next.record.base_energy = next
                    .record
                    .base_energy
                    .clamp(MIN_PINNED_GRAVITY_FLOOR, cfg.max_gravity_mass);
                next.excitation = next
                    .excitation
                    .clamp(MIN_PINNED_GRAVITY_FLOOR, cfg.steady_excitation_cap);
                next.record.stamp_crc32();
                working.push(next);
            } else {
                next.record.base_energy *= cfg.gravity_decay;
                next.record.phase_gravity_mass *= cfg.gravity_decay;
                next.excitation =
                    (next.excitation * cfg.gravity_decay).min(cfg.steady_excitation_cap);
                if next.record.phase_gravity_mass < cfg.evaporation_floor {
                    evaporated_anchors += 1;
                } else {
                    next.record.stamp_crc32();
                    working.push(next);
                }
            }
        }

        // Step 2: Coalesce incoming anchors (exact ID or near-duplicate compatible phase).
        let mut merged_anchors = 0_usize;
        let mut inserted_anchors = 0_usize;

        for &raw_in in incoming.pager.backing_entries() {
            if matches!(raw_in.lacuna_state, Lacuna::Undefined) {
                continue;
            }
            let inc = sanitize_slot_entry(raw_in, cfg);
            let inc_sig = phase_profile_bitcut64(&inc.record.phase_profile);

            // Check exact anchor_id match first (only if Lacuna & slot compatible).
            let exact_pos = working.iter().position(|e| {
                e.record.anchor_id == inc.record.anchor_id
                    && e.record.dominant_slot == inc.record.dominant_slot
                    && e.lacuna_state == inc.lacuna_state
            });

            let merge_target = if let Some(pos) = exact_pos {
                let cos = phase_profile_cosine_q7(
                    &working[pos].record.phase_profile,
                    &inc.record.phase_profile,
                );
                if cos >= 0.0 { Some(pos) } else { None }
            } else {
                // Coarse BitCut64 prefilter + exact Q7 cosine for near-duplicate consolidation.
                let mut best: Option<(usize, f32)> = None;
                for (pos, existing) in working.iter().enumerate() {
                    if existing.record.dominant_slot != inc.record.dominant_slot
                        || existing.lacuna_state != inc.lacuna_state
                    {
                        continue;
                    }
                    let coarse = bitcut64_polarity_similarity(existing.signature, inc_sig);
                    if coarse < MIN_BITCUT_MERGE_PREFILTER {
                        continue;
                    }
                    let cos = phase_profile_cosine_q7(
                        &existing.record.phase_profile,
                        &inc.record.phase_profile,
                    );
                    if cos >= cfg.merge_coherence_threshold
                        && best.is_none_or(|(_, best_cos)| cos > best_cos)
                    {
                        best = Some((pos, cos));
                    }
                }
                best.map(|(pos, _)| pos)
            };

            if let Some(pos) = merge_target {
                coalesce_slot_entry(&mut working[pos], &inc, next_generation, cfg);
                merged_anchors += 1;
            } else {
                let mut novel = inc;
                // Avoid anchor_id collision with an incompatible existing slot (e.g. Exterior vs SelfRef).
                if working
                    .iter()
                    .any(|e| e.record.anchor_id == novel.record.anchor_id)
                {
                    novel.record.anchor_id = allocate_next_anchor_id(&working);
                }
                novel.record.base_energy = novel
                    .record
                    .base_energy
                    .clamp(cfg.evaporation_floor, cfg.max_gravity_mass);
                novel.record.phase_gravity_mass = novel
                    .record
                    .phase_gravity_mass
                    .clamp(cfg.evaporation_floor, cfg.max_gravity_mass);
                novel.excitation = if matches!(novel.lacuna_state, Lacuna::SelfRef) {
                    cfg.novel_arrival_excitation
                } else {
                    novel
                        .record
                        .phase_gravity_mass
                        .min(cfg.steady_excitation_cap)
                };
                novel.last_step = next_generation;
                novel.signature = inc_sig;
                novel.record.stamp_crc32();
                working.push(novel);
                inserted_anchors += 1;
            }
        }

        // Step 3: Enforce hard bounded backing-bank ceiling without ever pruning pinned anchors.
        let pinned_count = working.iter().filter(|e| e.pinned).count();
        if pinned_count > cfg.pager.working_set_capacity {
            return Err(EquilibriumError::PinnedCapacityExceeded {
                capacity: cfg.pager.working_set_capacity,
            });
        }
        if working.len() > cfg.max_backing_anchors {
            let excess = working.len() - cfg.max_backing_anchors;
            let mut unpinned: Vec<(usize, f32, u64, u32)> = working
                .iter()
                .enumerate()
                .filter(|(_, e)| !e.pinned)
                .map(|(idx, e)| (idx, e.retention_priority(), e.last_step, e.record.anchor_id))
                .collect();
            unpinned.sort_by(|a, b| {
                a.1.total_cmp(&b.1)
                    .then_with(|| a.2.cmp(&b.2))
                    .then_with(|| a.3.cmp(&b.3))
            });
            let mut drop_flags = vec![false; working.len()];
            for (idx, _, _, _) in unpinned.into_iter().take(excess) {
                drop_flags[idx] = true;
                evaporated_anchors += 1;
            }
            let mut kept = Vec::with_capacity(cfg.max_backing_anchors);
            for (idx, entry) in working.into_iter().enumerate() {
                if !drop_flags[idx] {
                    kept.push(entry);
                }
            }
            working = kept;
        }

        // Phase 4: Reconcile optional 448-byte RuntimeSnapshot payloads transactionally.
        let (next_snapshot, snapshot_updated) =
            match (self.runtime_snapshot, incoming.runtime_snapshot) {
                (None, None) => (None, false),
                (Some(cur), None) => (Some(enforce_immutable_spine_floor(cur)?), false),
                (None, Some(inc)) => (Some(enforce_immutable_spine_floor(inc)?), true),
                (Some(cur), Some(inc)) => {
                    let prefer_incoming = incoming.generation >= self.generation;
                    let merged = reconcile_snapshot_payloads(&cur, &inc, prefer_incoming)?;
                    (Some(merged), true)
                }
            };

        // Phase 5: Commit atomically into the embedded pager and feed homeostatic tension.
        let pinned_preserved = working.iter().filter(|e| e.pinned).count();
        let exterior_isolated = working
            .iter()
            .filter(|e| matches!(e.lacuna_state, Lacuna::Exterior))
            .count();
        self.pager.replace_backing_entries(working).map_err(|_| {
            EquilibriumError::PinnedCapacityExceeded {
                capacity: cfg.pager.working_set_capacity,
            }
        })?;
        self.runtime_snapshot = next_snapshot;
        self.generation = next_generation;

        let churn_count = inserted_anchors.saturating_add(evaporated_anchors);
        let churn_q7 = ((churn_count * 127) / cfg.pager.working_set_capacity.max(1)).min(127) as u8;
        let boundary_part = LacunaPartition8::default();
        self.pager
            .record_tension_feedback(churn_q7, boundary_part.boundary_sensitivity_q7);

        Ok(EquilibriumJoinReport {
            generation: self.generation,
            merged_anchors,
            inserted_anchors,
            evaporated_anchors,
            pinned_preserved,
            exterior_isolated,
            snapshot_updated,
            backing_len: self.pager.backing_len(),
            resident_len: self.pager.resident_len(),
        })
    }

    /// Serializes this equilibrium state into a 64B-aligned `.lmnkc` binary container validated
    /// by [`LmnkContainerView`], embedding:
    /// - `LMNK_SECTION_PHASE_ANCHORS` (when non-empty),
    /// - `LMNK_SECTION_RUNTIME_SNAPSHOT` (448-byte payload when present), and
    /// - `HOST_EQUILIBRIUM_META_SECTION_KIND` (`0x0101`: `generation`, `pinned`, `lacuna_state`,
    ///   `excitation`, `last_step`).
    ///
    /// # Errors
    /// Returns [`EquilibriumError::ContainerInvalid`] if both anchors and snapshot are empty or
    /// container serialization fails.
    pub fn to_lmnkc_bytes(&self, timestamp_epoch_sec: u64) -> Result<Vec<u8>, EquilibriumError> {
        let entries = self.pager.backing_entries();
        if entries.is_empty() && self.runtime_snapshot.is_none() {
            return Err(EquilibriumError::ContainerInvalid(
                "cannot serialize empty EquilibriumState without anchors or snapshot".to_string(),
            ));
        }

        let mut anchor_payload: Vec<u8> = Vec::new();
        if !entries.is_empty() {
            let mut records = Vec::with_capacity(entries.len());
            for slot in entries {
                let mut rec = slot.record;
                rec.stamp_crc32();
                records.push(rec);
            }
            anchor_payload.resize(records.len() * core::mem::size_of::<PhaseAnchorRecord>(), 0);
            PhaseAnchorRecord::write_records_to_container_bytes(&records, &mut anchor_payload)
                .map_err(|e| {
                    EquilibriumError::ContainerInvalid(format!(
                        "PhaseAnchorRecord serialization failed: {e}"
                    ))
                })?;
        }

        let meta_payload = encode_equilibrium_meta_payload(self.generation, entries)?;

        let mut sections: Vec<LmnkSectionInput<'_>> = Vec::with_capacity(3);
        let mut flags = LMNK_CONTAINER_FLAG_COMPILED_CACHE;
        if let Some(ref snap) = self.runtime_snapshot {
            flags |= LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT;
            sections.push(LmnkSectionInput {
                kind: LMNK_SECTION_RUNTIME_SNAPSHOT,
                flags: 0,
                payload: snap,
            });
        }
        if !anchor_payload.is_empty() {
            sections.push(LmnkSectionInput {
                kind: LMNK_SECTION_PHASE_ANCHORS,
                flags: 0,
                payload: &anchor_payload,
            });
        }
        sections.push(LmnkSectionInput {
            kind: HOST_EQUILIBRIUM_META_SECTION_KIND,
            flags: 0,
            payload: &meta_payload,
        });

        let req_len = required_container_byte_len(&sections).map_err(|e| {
            EquilibriumError::ContainerInvalid(format!("container sizing failed: {e}"))
        })?;
        let mut out = vec![0_u8; req_len];
        let written =
            write_container_sections_to_bytes(flags, timestamp_epoch_sec, &sections, &mut out)
                .map_err(|e| {
                    EquilibriumError::ContainerInvalid(format!("container write failed: {e}"))
                })?;
        out.truncate(written);
        Ok(out)
    }

    /// Serializes this equilibrium state into a signed 64B-aligned `.lmnkc` container carrying
    /// `ContentHash (0xFF01)` (SHA-256) and `Signature (0xFF02)` (Ed25519).
    ///
    /// # Errors
    /// Returns [`EquilibriumError::ContainerInvalid`] if serialization or signing fails.
    pub fn to_signed_lmnkc_bytes(
        &self,
        timestamp_epoch_sec: u64,
        keypair: &SigningKeyPair,
    ) -> Result<Vec<u8>, EquilibriumError> {
        let raw = self.to_lmnkc_bytes(timestamp_epoch_sec)?;
        sign_lmnkc_container(&raw, keypair).map_err(|e| {
            EquilibriumError::ContainerInvalid(format!("failed to sign .lmnkc container: {e}"))
        })
    }

    /// Parses and restores an [`EquilibriumState`] from `.lmnkc` bytes with full CRC32, ABI,
    /// `CORE_TOPOLOGY`, and fail-closed cryptographic integrity verification under
    /// [`IntegrityPolicy::AllowUnsigned`].
    ///
    /// Fully backward compatible with legacy unsigned `.lmnkc` containers lacking
    /// `LMNK_SECTION_RUNTIME_SNAPSHOT`, `HOST_EQUILIBRIUM_META_SECTION_KIND`, or `0xFF01`/`0xFF02`.
    ///
    /// # Errors
    /// Returns [`EquilibriumError::TopologyMismatch`], [`EquilibriumError::ContainerInvalid`],
    /// [`EquilibriumError::SnapshotInvalid`], or [`EquilibriumError::PinnedCapacityExceeded`].
    pub fn from_lmnkc_bytes(
        bytes: &[u8],
        config: EquilibriumConfig,
    ) -> Result<Self, EquilibriumError> {
        Self::from_lmnkc_bytes_with_policy(bytes, config, &IntegrityPolicy::AllowUnsigned)
    }

    /// Parses and restores an [`EquilibriumState`] from `.lmnkc` bytes while enforcing an explicit
    /// cryptographic [`IntegrityPolicy`].
    ///
    /// # Errors
    /// Returns [`EquilibriumError::TopologyMismatch`], [`EquilibriumError::ContainerInvalid`],
    /// [`EquilibriumError::SnapshotInvalid`], or [`EquilibriumError::PinnedCapacityExceeded`].
    pub fn from_lmnkc_bytes_with_policy(
        bytes: &[u8],
        config: EquilibriumConfig,
        policy: &IntegrityPolicy,
    ) -> Result<Self, EquilibriumError> {
        let cfg = config.sanitized();
        let view = parse_container_checked(bytes, CORE_TOPOLOGY.topology_crc32)?;
        verify_lmnkc_integrity(bytes, policy)
            .map_err(|e| EquilibriumError::ContainerInvalid(e.to_string()))?;
        let topology_crc32 = view.header().topology_crc32;

        let runtime_snapshot = match view.section_bytes(LMNK_SECTION_RUNTIME_SNAPSHOT) {
            Some(raw) => Some(enforce_immutable_spine_floor(validate_snapshot_payload(
                raw,
            )?)?),
            None => None,
        };

        let (generation, meta_by_id) = match view.section_bytes(HOST_EQUILIBRIUM_META_SECTION_KIND)
        {
            Some(raw) => decode_equilibrium_meta_payload(raw)?,
            None => (0_u64, HashMap::new()),
        };

        let count = view.phase_anchor_count();
        let mut entries = Vec::with_capacity(count);
        for idx in 0..count {
            let Some(rec) = view.read_phase_anchor(idx) else {
                continue;
            };
            let entry = if let Some(&meta) = meta_by_id.get(&rec.anchor_id) {
                let mut slot = AnchorSlotEntry::new(rec, meta.lacuna_state, meta.pinned);
                slot.excitation = meta.excitation;
                slot.last_step = meta.last_step;
                slot
            } else {
                AnchorSlotEntry::new(rec, Lacuna::SelfRef, false)
            };
            entries.push(sanitize_slot_entry(entry, cfg));
        }

        let mut pager = AnchorWorkingSetPager::with_topology_crc32(cfg.pager, topology_crc32);
        pager.replace_backing_entries(entries).map_err(|_| {
            EquilibriumError::PinnedCapacityExceeded {
                capacity: cfg.pager.working_set_capacity,
            }
        })?;

        Ok(Self {
            config: cfg,
            topology_crc32,
            pager,
            runtime_snapshot,
            generation,
        })
    }

    /// Writes the serialized `.lmnkc` equilibrium container to `path`, returning bytes written.
    ///
    /// # Errors
    /// Returns [`EquilibriumError`] on unsafe output path, serialization, or filesystem failure.
    pub fn save_lmnkc_to_path(
        &self,
        path: impl AsRef<Path>,
        timestamp_epoch_sec: u64,
    ) -> Result<usize, EquilibriumError> {
        let p = path.as_ref();
        validate_local_output_path(p).map_err(|e| {
            EquilibriumError::Io(format!("unsafe output path {}: {e}", p.display()))
        })?;
        if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                EquilibriumError::Io(format!(
                    "failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let bytes = self.to_lmnkc_bytes(timestamp_epoch_sec)?;
        std::fs::write(p, &bytes).map_err(|e| {
            EquilibriumError::Io(format!("failed to write .lmnkc {}: {e}", p.display()))
        })?;
        Ok(bytes.len())
    }

    /// Reads and validates an `.lmnkc` equilibrium container from `path` via bounded single-FD read.
    ///
    /// # Errors
    /// Returns [`EquilibriumError`] on I/O, size cap, topology mismatch, or validation failure.
    pub fn load_lmnkc_from_path(
        path: impl AsRef<Path>,
        config: EquilibriumConfig,
    ) -> Result<Self, EquilibriumError> {
        Self::load_lmnkc_from_path_with_policy(path, config, &IntegrityPolicy::AllowUnsigned)
    }

    /// Reads and validates an `.lmnkc` equilibrium container from `path` under an explicit
    /// cryptographic [`IntegrityPolicy`].
    ///
    /// # Errors
    /// Returns [`EquilibriumError`] on I/O, size cap, topology mismatch, or integrity/policy failure.
    pub fn load_lmnkc_from_path_with_policy(
        path: impl AsRef<Path>,
        config: EquilibriumConfig,
        policy: &IntegrityPolicy,
    ) -> Result<Self, EquilibriumError> {
        let p = path.as_ref();
        let file = File::open(p).map_err(|e| {
            EquilibriumError::Io(format!("failed to open .lmnkc {}: {e}", p.display()))
        })?;
        let meta_len = file
            .metadata()
            .map_err(|e| {
                EquilibriumError::Io(format!("failed to stat .lmnkc {}: {e}", p.display()))
            })?
            .len();
        if meta_len > LMNK_MAX_CONTAINER_BYTES as u64 {
            return Err(EquilibriumError::ContainerInvalid(format!(
                "container {} ({meta_len} B) exceeds cap {LMNK_MAX_CONTAINER_BYTES}",
                p.display()
            )));
        }
        let mut bytes = Vec::with_capacity((meta_len as usize).min(LMNK_MAX_CONTAINER_BYTES));
        file.take((LMNK_MAX_CONTAINER_BYTES as u64).saturating_add(1))
            .read_to_end(&mut bytes)
            .map_err(|e| {
                EquilibriumError::Io(format!("failed to read .lmnkc {}: {e}", p.display()))
            })?;
        if bytes.len() > LMNK_MAX_CONTAINER_BYTES {
            return Err(EquilibriumError::ContainerInvalid(format!(
                "container {} exceeds read cap {LMNK_MAX_CONTAINER_BYTES}",
                p.display()
            )));
        }
        Self::from_lmnkc_bytes_with_policy(&bytes, config, policy)
    }
}

#[derive(Debug, Clone, Copy)]
struct PersistedAnchorMeta {
    pinned: bool,
    lacuna_state: Lacuna,
    excitation: f32,
    last_step: u64,
}

fn parse_container_checked(
    bytes: &[u8],
    expected_topology: u32,
) -> Result<LmnkContainerView<'_>, EquilibriumError> {
    let view = LmnkContainerView::parse(bytes).map_err(|status| {
        if status == LMNK_ERR_TOPOLOGY_MISMATCH {
            let actual = LmnkContainerHeader::from_le_bytes(bytes)
                .map(|h| h.topology_crc32)
                .unwrap_or(0);
            EquilibriumError::TopologyMismatch {
                expected: expected_topology,
                actual,
            }
        } else {
            EquilibriumError::ContainerInvalid(format!(
                "LmnkContainerView rejected container: {} ({})",
                status.as_str(),
                status.code()
            ))
        }
    })?;
    if view.header().topology_crc32 != expected_topology {
        return Err(EquilibriumError::TopologyMismatch {
            expected: expected_topology,
            actual: view.header().topology_crc32,
        });
    }
    Ok(view)
}

fn validate_snapshot_payload(
    raw: &[u8],
) -> Result<[u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES], EquilibriumError> {
    if raw.len() != LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES {
        return Err(EquilibriumError::SnapshotInvalid(format!(
            "expected {} B RuntimeSnapshot payload, got {} B",
            LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES,
            raw.len()
        )));
    }
    let mut payload = [0_u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES];
    payload.copy_from_slice(raw);
    let spine = SystemSpineState::read_unaligned_from(&payload, SNAPSHOT_SPINE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("failed to decode SystemSpineState".to_string())
        })?;
    if (spine.control_flags & !SPINE_CONTROL_KNOWN_FLAGS) != 0
        || spine.last_observed_entropy_q15 > SPINE_ENTROPY_Q15_ONE
        || (spine.last_veto_mask & !SPINE_BYPASS_VETO_KNOWN_MASK) != 0
    {
        return Err(EquilibriumError::SnapshotInvalid(
            "SystemSpineState failed control/veto invariant checks".to_string(),
        ));
    }
    let envelope = EnvelopeState::read_unaligned_from(&payload, SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET)
        .map_err(|_| {
        EquilibriumError::SnapshotInvalid("failed to decode EnvelopeState".to_string())
    })?;
    if !envelope.validate_invariants() {
        return Err(EquilibriumError::SnapshotInvalid(
            "EnvelopeState failed range/flag invariant checks".to_string(),
        ));
    }
    let bounds =
        EnvelopeBounds::read_unaligned_from(&payload, SNAPSHOT_ENVELOPE_BOUNDS_PAYLOAD_OFFSET)
            .map_err(|_| {
                EquilibriumError::SnapshotInvalid("failed to decode EnvelopeBounds".to_string())
            })?;
    if !bounds.validate_invariants() {
        return Err(EquilibriumError::SnapshotInvalid(
            "EnvelopeBounds failed range/ordering invariant checks".to_string(),
        ));
    }
    Ok(payload)
}

fn enforce_immutable_spine_floor(
    mut payload: [u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES],
) -> Result<[u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES], EquilibriumError> {
    let mut spine = SystemSpineState::read_unaligned_from(&payload, SNAPSHOT_SPINE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("failed to read SystemSpineState".to_string())
        })?;
    if spine.max_observe_stride == 0 {
        spine.max_observe_stride = 1;
    }
    spine
        .write_unaligned_to(&mut payload, SNAPSHOT_SPINE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("failed to encode SystemSpineState".to_string())
        })?;

    let mut envelope =
        EnvelopeState::read_unaligned_from(&payload, SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET).map_err(
            |_| EquilibriumError::SnapshotInvalid("failed to read EnvelopeState".to_string()),
        )?;
    let default_envelope = EnvelopeState::new();
    envelope.apply_host_pin_floor(HOST_IMMUTABLE_PINNED_COMPUTE_MASK);
    envelope.pinned_aperture_mask |= default_envelope.pinned_aperture_mask;
    envelope.pinned_slot_mask |= default_envelope.pinned_slot_mask;
    envelope.pinned_action_mode_mask |= default_envelope.pinned_action_mode_mask;
    envelope
        .write_unaligned_to(&mut payload, SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("failed to encode EnvelopeState".to_string())
        })?;
    Ok(payload)
}

fn read_u64_le_at(buf: &[u8], offset: usize) -> u64 {
    let Some(slice) = buf.get(offset..offset + 8) else {
        return 0;
    };
    let Ok(arr) = <[u8; 8]>::try_from(slice) else {
        return 0;
    };
    u64::from_le_bytes(arr)
}

fn reconcile_snapshot_payloads(
    cur: &[u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES],
    inc: &[u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES],
    prefer_incoming: bool,
) -> Result<[u8; LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES], EquilibriumError> {
    let cur_spine = SystemSpineState::read_unaligned_from(cur, SNAPSHOT_SPINE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("corrupt current SystemSpineState".to_string())
        })?;
    let inc_spine = SystemSpineState::read_unaligned_from(inc, SNAPSHOT_SPINE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("corrupt incoming SystemSpineState".to_string())
        })?;
    let cur_env = EnvelopeState::read_unaligned_from(cur, SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("corrupt current EnvelopeState".to_string())
        })?;
    let inc_env = EnvelopeState::read_unaligned_from(inc, SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("corrupt incoming EnvelopeState".to_string())
        })?;

    let cur_bounds =
        EnvelopeBounds::read_unaligned_from(cur, SNAPSHOT_ENVELOPE_BOUNDS_PAYLOAD_OFFSET).map_err(
            |_| EquilibriumError::SnapshotInvalid("corrupt current EnvelopeBounds".to_string()),
        )?;
    let inc_bounds =
        EnvelopeBounds::read_unaligned_from(inc, SNAPSHOT_ENVELOPE_BOUNDS_PAYLOAD_OFFSET).map_err(
            |_| EquilibriumError::SnapshotInvalid("corrupt incoming EnvelopeBounds".to_string()),
        )?;
    if !cur_bounds.validate_invariants() || !inc_bounds.validate_invariants() {
        return Err(EquilibriumError::SnapshotInvalid(
            "EnvelopeBounds failed invariant checks during reconciliation".to_string(),
        ));
    }

    let cur_steps = read_u64_le_at(cur, SNAPSHOT_STEP_COUNTER_OFFSET);
    let inc_steps = read_u64_le_at(inc, SNAPSHOT_STEP_COUNTER_OFFSET);
    let take_incoming = if prefer_incoming {
        true
    } else {
        inc_steps > cur_steps
    };

    let mut out = if take_incoming { *inc } else { *cur };
    let mut merged_spine = if take_incoming { inc_spine } else { cur_spine };
    let winner_bounds = if take_incoming {
        inc_bounds
    } else {
        cur_bounds
    };
    if merged_spine.max_observe_stride == 0 {
        merged_spine.max_observe_stride = 1;
    }

    // Winner inherits temporary locks (`unobserved_locked`, `irreversible_step_lock`) without OR;
    // all 5 persistent pin-floor masks are monotonically OR-merged across both snapshots.
    let mut merged_env = if take_incoming { inc_env } else { cur_env };
    let default_env = EnvelopeState::new();
    merged_env.baseline_pinned_compute_mask |= cur_env.baseline_pinned_compute_mask
        | inc_env.baseline_pinned_compute_mask
        | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
    merged_env.pinned_compute_mask |= merged_env.baseline_pinned_compute_mask;
    merged_env.pinned_aperture_mask |= cur_env.pinned_aperture_mask
        | inc_env.pinned_aperture_mask
        | default_env.pinned_aperture_mask;
    merged_env.pinned_slot_mask |=
        cur_env.pinned_slot_mask | inc_env.pinned_slot_mask | default_env.pinned_slot_mask;
    merged_env.pinned_action_mode_mask |= cur_env.pinned_action_mode_mask
        | inc_env.pinned_action_mode_mask
        | default_env.pinned_action_mode_mask;

    out[SNAPSHOT_SLOT_PINNED_MASK_OFFSET] |=
        cur[SNAPSHOT_SLOT_PINNED_MASK_OFFSET] | inc[SNAPSHOT_SLOT_PINNED_MASK_OFFSET];
    merged_spine
        .write_unaligned_to(&mut out, SNAPSHOT_SPINE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("failed to write merged SystemSpineState".to_string())
        })?;
    merged_env
        .write_unaligned_to(&mut out, SNAPSHOT_ENVELOPE_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("failed to write merged EnvelopeState".to_string())
        })?;
    winner_bounds
        .write_unaligned_to(&mut out, SNAPSHOT_ENVELOPE_BOUNDS_PAYLOAD_OFFSET)
        .map_err(|_| {
            EquilibriumError::SnapshotInvalid("failed to write winner EnvelopeBounds".to_string())
        })?;
    Ok(out)
}

fn sanitize_slot_entry(mut entry: AnchorSlotEntry, cfg: EquilibriumConfig) -> AnchorSlotEntry {
    if !entry.record.base_energy.is_finite() || entry.record.base_energy < 0.0 {
        entry.record.base_energy = 0.0;
    }
    if !entry.record.phase_gravity_mass.is_finite() || entry.record.phase_gravity_mass < 0.0 {
        entry.record.phase_gravity_mass = entry.record.base_energy;
    }
    entry.record.base_energy = entry.record.base_energy.min(cfg.max_gravity_mass);
    entry.record.phase_gravity_mass = entry.record.phase_gravity_mass.min(cfg.max_gravity_mass);
    if !entry.record.resultant_r_bar.is_finite() {
        entry.record.resultant_r_bar = 1.0;
    }
    entry.record.resultant_r_bar = entry.record.resultant_r_bar.clamp(0.0, 1.0);
    if !entry.record.von_mises_kappa.is_finite() || entry.record.von_mises_kappa < 0.0 {
        entry.record.von_mises_kappa = 0.0;
    }
    if !entry.excitation.is_finite() || entry.excitation < 0.0 {
        entry.excitation = entry.record.phase_gravity_mass;
    }
    entry.signature = phase_profile_bitcut64(&entry.record.phase_profile);
    entry
}

fn coalesce_slot_entry(
    dst: &mut AnchorSlotEntry,
    inc: &AnchorSlotEntry,
    generation: u64,
    cfg: EquilibriumConfig,
) {
    let dst_mass = dst.record.phase_gravity_mass.max(cfg.evaporation_floor);
    let inc_mass = inc.record.phase_gravity_mass.max(cfg.evaporation_floor);

    // Keep Spine-pinned anchor profiles highly stable against unpinned incoming drift while
    // preserving smooth circular plasticity for unpinned equilibria.
    let (w_dst, w_inc) = if dst.pinned && !inc.pinned {
        (0.90_f64, 0.10_f64)
    } else if !dst.pinned && inc.pinned {
        (0.10_f64, 0.90_f64)
    } else {
        let total = f64::from(dst_mass) + f64::from(inc_mass);
        let raw_inc_w = if total > 1e-9 {
            f64::from(inc_mass) / total
        } else {
            0.5
        };
        let bounded_inc_w = raw_inc_w.clamp(
            MIN_UNPINNED_PLASTICITY_WEIGHT,
            1.0 - MIN_UNPINNED_PLASTICITY_WEIGHT,
        );
        (1.0 - bounded_inc_w, bounded_inc_w)
    };

    let (merged_profile, mean_r_bar) = circular_merge_phase_profiles_q7(
        &dst.record.phase_profile,
        w_dst,
        &inc.record.phase_profile,
        w_inc,
    );

    dst.record.phase_profile = merged_profile;
    dst.signature = phase_profile_bitcut64(&dst.record.phase_profile);
    dst.pinned = dst.pinned || inc.pinned;
    dst.record.sample_count = dst
        .record
        .sample_count
        .saturating_add(inc.record.sample_count.max(1));
    dst.record.base_energy = (dst.record.base_energy
        + cfg.reinforcement_gain * inc.record.base_energy)
        .min(cfg.max_gravity_mass);
    dst.record.phase_gravity_mass = (dst.record.phase_gravity_mass
        + cfg.reinforcement_gain * inc.record.phase_gravity_mass)
        .min(cfg.max_gravity_mass);
    if dst.pinned {
        dst.record.base_energy = dst.record.base_energy.max(MIN_PINNED_GRAVITY_FLOOR);
        dst.record.phase_gravity_mass = dst.record.phase_gravity_mass.max(MIN_PINNED_GRAVITY_FLOOR);
    }
    dst.record.resultant_r_bar = mean_r_bar.clamp(0.0, 1.0);
    dst.record.von_mises_kappa = if mean_r_bar < 0.98 {
        (mean_r_bar * (2.0 - mean_r_bar * mean_r_bar) / (1.0 - mean_r_bar * mean_r_bar))
            .clamp(0.0, 50.0)
    } else {
        50.0
    };
    dst.record.lineage_flags |= inc.record.lineage_flags;
    dst.record.action_mode_flags |= inc.record.action_mode_flags;
    dst.excitation = (dst.excitation + cfg.reinforcement_gain * inc.record.phase_gravity_mass)
        .min(cfg.steady_excitation_cap);
    dst.last_step = generation;
    dst.record.stamp_crc32();
}

/// Circular vector synthesis on $S^1$ per Q7 head with zero-collapse protection (retains the
/// stronger head phase on near-antipodal cancellation instead of collapsing to zero).
fn circular_merge_phase_profiles_q7(
    a: &[i8; MAX_ATTENTION_HEADS],
    w_a: f64,
    b: &[i8; MAX_ATTENTION_HEADS],
    w_b: f64,
) -> ([i8; MAX_ATTENTION_HEADS], f32) {
    let mut out = [0_i8; MAX_ATTENTION_HEADS];
    let q7_to_rad = PI / 128.0;
    let rad_to_q7 = 128.0 / PI;
    let w_sum = (w_a + w_b).max(1e-9);
    let mut r_sum = 0.0_f64;
    let mut active_heads = 0_usize;

    for idx in 0..MAX_ATTENTION_HEADS {
        let qa = a[idx];
        let qb = b[idx];
        if qa == 0 && qb == 0 {
            out[idx] = 0;
            continue;
        }
        active_heads += 1;
        let ta = f64::from(qa) * q7_to_rad;
        let tb = f64::from(qb) * q7_to_rad;
        let cx = w_a * ta.cos() + w_b * tb.cos();
        let sy = w_a * ta.sin() + w_b * tb.sin();
        let norm = (cx * cx + sy * sy).sqrt();
        if norm < 1e-4 * w_sum {
            out[idx] = if w_b > w_a { qb } else { qa };
            r_sum += 0.5;
        } else {
            let angle = sy.atan2(cx);
            let q = (angle * rad_to_q7).round() as i32;
            out[idx] = q.clamp(-127, 127) as i8;
            r_sum += (norm / w_sum).clamp(0.0, 1.0);
        }
    }

    let mean_r = if active_heads == 0 {
        1.0_f32
    } else {
        (r_sum / active_heads as f64).clamp(0.0, 1.0) as f32
    };
    (out, mean_r)
}

fn allocate_next_anchor_id(entries: &[AnchorSlotEntry]) -> u32 {
    let max_id = entries
        .iter()
        .map(|e| e.record.anchor_id)
        .max()
        .unwrap_or(0);
    let mut cand = max_id.wrapping_add(1).max(1);
    while entries.iter().any(|e| e.record.anchor_id == cand) {
        cand = cand.wrapping_add(1).max(1);
    }
    cand
}

fn encode_equilibrium_meta_payload(
    generation: u64,
    entries: &[AnchorSlotEntry],
) -> Result<Vec<u8>, EquilibriumError> {
    let count_u32 = u32::try_from(entries.len()).map_err(|_| {
        EquilibriumError::ContainerInvalid("anchor metadata count exceeds u32".to_string())
    })?;
    let total_bytes = EQUILIBRIUM_META_HEADER_BYTES
        .checked_add(entries.len().saturating_mul(EQUILIBRIUM_META_RECORD_BYTES))
        .ok_or_else(|| {
            EquilibriumError::ContainerInvalid("equilibrium metadata size overflow".to_string())
        })?;
    let mut out = Vec::with_capacity(total_bytes);
    out.extend_from_slice(&EQUILIBRIUM_META_MAGIC);
    out.extend_from_slice(&EQUILIBRIUM_META_VERSION.to_le_bytes());
    out.extend_from_slice(&generation.to_le_bytes());
    out.extend_from_slice(&count_u32.to_le_bytes());
    out.extend_from_slice(&0_u32.to_le_bytes()); // reserved

    for entry in entries {
        let mut flags = 0_u8;
        if entry.pinned {
            flags |= 1_u8 << 0;
        }
        flags |= (entry.lacuna_state.code() & 0x03) << 1;
        out.extend_from_slice(&entry.record.anchor_id.to_le_bytes());
        out.push(flags);
        out.extend_from_slice(&[0_u8; 3]);
        out.extend_from_slice(&entry.excitation.to_bits().to_le_bytes());
        out.extend_from_slice(&entry.last_step.to_le_bytes());
    }
    Ok(out)
}

fn decode_equilibrium_meta_payload(
    bytes: &[u8],
) -> Result<(u64, HashMap<u32, PersistedAnchorMeta>), EquilibriumError> {
    if bytes.len() < EQUILIBRIUM_META_HEADER_BYTES {
        return Err(EquilibriumError::ContainerInvalid(
            "equilibrium metadata section shorter than 24-byte header".to_string(),
        ));
    }
    if bytes[0..4] != EQUILIBRIUM_META_MAGIC {
        return Err(EquilibriumError::ContainerInvalid(
            "invalid equilibrium metadata magic".to_string(),
        ));
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != EQUILIBRIUM_META_VERSION {
        return Err(EquilibriumError::ContainerInvalid(format!(
            "unsupported equilibrium metadata version: {version}"
        )));
    }
    let generation = read_u64_le_at(bytes, 8);
    let count = u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
    let reserved = u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]);
    if reserved != 0 {
        return Err(EquilibriumError::ContainerInvalid(
            "non-zero reserved field in equilibrium metadata header".to_string(),
        ));
    }
    let expected_len = EQUILIBRIUM_META_HEADER_BYTES
        .checked_add(
            count
                .checked_mul(EQUILIBRIUM_META_RECORD_BYTES)
                .ok_or_else(|| {
                    EquilibriumError::ContainerInvalid(
                        "metadata record length overflow".to_string(),
                    )
                })?,
        )
        .ok_or_else(|| {
            EquilibriumError::ContainerInvalid("metadata total length overflow".to_string())
        })?;
    if bytes.len() != expected_len {
        return Err(EquilibriumError::ContainerInvalid(format!(
            "equilibrium metadata length mismatch: expected {expected_len}, got {}",
            bytes.len()
        )));
    }

    let mut map = HashMap::with_capacity(count);
    let mut off = EQUILIBRIUM_META_HEADER_BYTES;
    for _ in 0..count {
        let anchor_id =
            u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        let flags = bytes[off + 4];
        if (flags & !0x07) != 0 || bytes[off + 5] != 0 || bytes[off + 6] != 0 || bytes[off + 7] != 0
        {
            return Err(EquilibriumError::ContainerInvalid(
                "reserved flags set in equilibrium anchor metadata".to_string(),
            ));
        }
        let pinned = (flags & 0x01) != 0;
        let lacuna_state = Lacuna::from_code((flags >> 1) & 0x03);
        if matches!(lacuna_state, Lacuna::Undefined) {
            return Err(EquilibriumError::ContainerInvalid(format!(
                "terminal Lacuna::Undefined persisted for anchor_id {anchor_id}"
            )));
        }
        let excitation = f32::from_bits(u32::from_le_bytes([
            bytes[off + 8],
            bytes[off + 9],
            bytes[off + 10],
            bytes[off + 11],
        ]));
        if !excitation.is_finite() || excitation < 0.0 {
            return Err(EquilibriumError::ContainerInvalid(format!(
                "non-finite or negative excitation persisted for anchor_id {anchor_id}"
            )));
        }
        let last_step = read_u64_le_at(bytes, off + 12);
        map.insert(
            anchor_id,
            PersistedAnchorMeta {
                pinned,
                lacuna_state,
                excitation,
                last_step,
            },
        );
        off += EQUILIBRIUM_META_RECORD_BYTES;
    }
    Ok((generation, map))
}

#[cfg(test)]
mod tests {
    use super::*;
    use liminika_core::runtime::{Config, RuntimeContext};
    use liminika_core::stigmergy::ATTRACTOR_SLOT_KNOWLEDGE;

    fn make_record(anchor_id: u32, seed: u32, mass: f32) -> PhaseAnchorRecord {
        let mut rec = PhaseAnchorRecord::new_empty(anchor_id, ATTRACTOR_SLOT_KNOWLEDGE, 0);
        rec.base_energy = mass;
        rec.phase_gravity_mass = mass;
        rec.resultant_r_bar = 0.95;
        rec.von_mises_kappa = 10.0;
        rec.sample_count = 2;
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

    fn make_snapshot_bytes(
        pin_mask: u64,
        step_counter: u64,
    ) -> [u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES] {
        let cfg = Config::new().with_head_dim(64).with_num_heads(8);
        let mut ctx = RuntimeContext::new(cfg).expect("runtime ctx");
        ctx.apply_host_spine_pin_floor(pin_mask);
        ctx.substrate_mut().step_counter = step_counter;
        let mut buf = [0_u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
        let mut req = 0_usize;
        ctx.serialize_to_bytes(&mut buf, &mut req)
            .expect("serialize snapshot");
        assert_eq!(req, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES);
        buf
    }

    #[test]
    fn hundred_generations_prevent_thermal_death_and_evaporate_stale_memories() {
        let cfg = EquilibriumConfig::default();
        let mut state = EquilibriumState::new(cfg);

        // 2 Spine-pinned anchors (never reinforced again) + 2 reinforced core anchors + 4 stale anchors.
        let pinned_a = make_record(1, 101, 1.0);
        let pinned_b = make_record(2, 202, 1.0);
        let core_a = make_record(10, 1010, 1.2);
        let core_b = make_record(11, 1111, 1.2);
        state
            .insert_anchor(pinned_a, Lacuna::SelfRef, true)
            .expect("pin a");
        state
            .insert_anchor(pinned_b, Lacuna::SelfRef, true)
            .expect("pin b");
        state
            .insert_anchor(core_a, Lacuna::SelfRef, false)
            .expect("core a");
        state
            .insert_anchor(core_b, Lacuna::SelfRef, false)
            .expect("core b");
        for stale_id in 50..54_u32 {
            state
                .insert_anchor(
                    make_record(stale_id, stale_id * 19, 1.0),
                    Lacuna::SelfRef,
                    false,
                )
                .expect("stale");
        }

        let mut total_evaporated = 0_usize;
        for generation in 1..=100_u64 {
            let mut pulse = EquilibriumState::new(cfg);
            pulse
                .insert_anchor(core_a, Lacuna::SelfRef, false)
                .expect("pulse a");
            pulse
                .insert_anchor(core_b, Lacuna::SelfRef, false)
                .expect("pulse b");
            let report = state.join(&pulse).expect("join");
            total_evaporated += report.evaporated_anchors;
            assert_eq!(report.generation, generation);

            // Round-trip through .lmnkc every generation to prove flat persistence parity.
            let bytes = state
                .to_lmnkc_bytes(1_774_900_000 + generation)
                .expect("serialize");
            state = EquilibriumState::from_lmnkc_bytes(&bytes, cfg).expect("restore");
        }

        // Non-thermal-death: pinned & reinforced anchors survive 100 generations with finite mass.
        for id in [1_u32, 2, 10, 11] {
            let entry = state
                .pager()
                .get_entry(id)
                .expect("core/pinned must survive");
            assert!(
                state.pager().is_resident(id),
                "anchor {id} must be resident"
            );
            assert!(entry.record.phase_gravity_mass.is_finite());
            assert!(entry.record.phase_gravity_mass >= MIN_PINNED_GRAVITY_FLOOR);
            assert!(entry.record.phase_gravity_mass <= cfg.max_gravity_mass + 1e-6);
        }
        assert!(state.pager().get_entry(1).unwrap().pinned);
        assert!(state.pager().get_entry(2).unwrap().pinned);

        // Healthy forgetting: unreinforced stale anchors evaporated cleanly.
        assert!(total_evaporated >= 4);
        for stale_id in 50..54_u32 {
            assert!(
                state.pager().get_entry(stale_id).is_none(),
                "stale unpinned anchor {stale_id} should have evaporated"
            );
        }
    }

    #[test]
    fn hundred_generations_prevent_lock_in_and_keep_size_bounded() {
        let cfg = EquilibriumConfig {
            pager: AnchorPagerConfig {
                working_set_capacity: 16,
                ..AnchorPagerConfig::default()
            },
            max_backing_anchors: 32,
            ..EquilibriumConfig::default()
        };
        let mut state = EquilibriumState::new(cfg);

        // 2 pinned + 30 saturated unpinned anchors.
        state
            .insert_anchor(make_record(1, 11, 2.0), Lacuna::SelfRef, true)
            .expect("pin 1");
        state
            .insert_anchor(make_record(2, 22, 2.0), Lacuna::SelfRef, true)
            .expect("pin 2");
        for id in 10..40_u32 {
            state
                .insert_anchor(make_record(id, id * 37 + 3, 2.0), Lacuna::SelfRef, false)
                .expect("seed");
        }

        let mut max_lmnkc_bytes = 0_usize;
        for generation in 1..=100_u64 {
            let mut wave = EquilibriumState::new(cfg);
            // Reinforce existing 16 unpinned anchors to saturate the equilibrium.
            for id in 10..26_u32 {
                wave.insert_anchor(make_record(id, id * 37 + 3, 1.8), Lacuna::SelfRef, false)
                    .expect("wave");
            }
            // Inject rotating transient anchors to stress backing-bank bounds.
            let transient_id = 1000 + (generation as u32);
            wave.insert_anchor(
                make_record(transient_id, transient_id * 13 + 7, 1.0),
                Lacuna::SelfRef,
                false,
            )
            .expect("transient");

            let report = state.join(&wave).expect("join");
            assert!(report.backing_len <= cfg.max_backing_anchors);
            assert!(report.resident_len <= cfg.pager.working_set_capacity);
            assert!(state.pager().is_resident(1));
            assert!(state.pager().is_resident(2));

            let bytes = state
                .to_lmnkc_bytes(1_774_900_000 + generation)
                .expect("to_lmnkc");
            max_lmnkc_bytes = max_lmnkc_bytes.max(bytes.len());
            for entry in state.pager().backing_entries() {
                assert!(entry.record.base_energy.is_finite());
                assert!(entry.record.phase_gravity_mass.is_finite());
                assert!(entry.record.resultant_r_bar.is_finite());
                assert!(entry.record.von_mises_kappa.is_finite());
                assert!(entry.excitation.is_finite());
            }
        }

        // Verify bounded container size (32 anchors * 128 B + metadata + headers < 8 KiB).
        assert!(
            max_lmnkc_bytes < 8192,
            "unexpected lmnkc bloat: {max_lmnkc_bytes} B"
        );

        // Plasticity / non-lock-in check: inject a brand-new crystallized anchor at generation 101.
        let novel = make_record(9999, 0xCAFE_1234, 1.25);
        let report = state
            .ingest_crystallized_anchors(&[novel])
            .expect("ingest novel");
        assert_eq!(report.inserted_anchors, 1);
        assert!(
            state.pager().is_resident(9999),
            "novel crystallized anchor must enter working set without lock-in after 100 gens"
        );
        assert!(state.pager().is_resident(1));
        assert!(state.pager().is_resident(2));
    }

    #[test]
    fn topology_mismatch_rejected_and_exterior_never_promoted() {
        let cfg = EquilibriumConfig::default();
        let mut a = EquilibriumState::with_topology_crc32(cfg, CORE_TOPOLOGY.topology_crc32);
        let b =
            EquilibriumState::with_topology_crc32(cfg, CORE_TOPOLOGY.topology_crc32 ^ 0xDEAD_BEEF);
        let err = a.join(&b).expect_err("topology mismatch must be rejected");
        assert_eq!(
            err,
            EquilibriumError::TopologyMismatch {
                expected: CORE_TOPOLOGY.topology_crc32,
                actual: CORE_TOPOLOGY.topology_crc32 ^ 0xDEAD_BEEF,
            }
        );

        // Unsettled Lacuna::Exterior stays isolated across join and .lmnkc roundtrip.
        let mut inc = EquilibriumState::new(cfg);
        let ext = make_record(777, 7777, 3.0);
        inc.insert_anchor(ext, Lacuna::Exterior, false)
            .expect("insert ext");
        let report = a.join(&inc).expect("join same topology");
        assert_eq!(report.exterior_isolated, 1);
        assert!(!a.pager().is_resident(777));
        assert_eq!(
            a.pager().get_entry(777).unwrap().lacuna_state,
            Lacuna::Exterior
        );

        let bytes = a.to_lmnkc_bytes(1_774_900_100).expect("serialize");
        let restored = EquilibriumState::from_lmnkc_bytes(&bytes, cfg).expect("restore");
        assert!(!restored.pager().is_resident(777));
        assert_eq!(
            restored.pager().get_entry(777).unwrap().lacuna_state,
            Lacuna::Exterior
        );
    }

    #[test]
    fn runtime_snapshot_roundtrip_monotonic_pins_and_legacy_compat() {
        let cfg = EquilibriumConfig::default();
        let snap_a = make_snapshot_bytes(0b0011, 10);
        let snap_b = make_snapshot_bytes(0b1100, 25);

        let mut eq_a = EquilibriumState::new(cfg);
        eq_a.insert_anchor(make_record(1, 10, 1.5), Lacuna::SelfRef, true)
            .expect("pin 1");
        eq_a.update_runtime_snapshot_from_container(&snap_a)
            .expect("snap a");

        let mut eq_b = EquilibriumState::new(cfg);
        eq_b.insert_anchor(make_record(2, 20, 1.5), Lacuna::SelfRef, false)
            .expect("anchor 2");
        eq_b.update_runtime_snapshot_from_container(&snap_b)
            .expect("snap b");

        let report = eq_a.join(&eq_b).expect("join snapshots");
        assert!(report.snapshot_updated);

        let merged_env = eq_a.snapshot_envelope_state().expect("merged envelope");
        assert_eq!(merged_env.baseline_pinned_compute_mask & 0b1111, 0b1111);

        // .lmnkc roundtrip preserves both RuntimeSnapshot section and host metadata.
        let lmnkc = eq_a.to_lmnkc_bytes(1_774_900_500).expect("to_lmnkc");
        let view = LmnkContainerView::parse(&lmnkc).expect("parse view");
        assert!(view.section_bytes(LMNK_SECTION_RUNTIME_SNAPSHOT).is_some());
        assert!(
            view.section_bytes(HOST_EQUILIBRIUM_META_SECTION_KIND)
                .is_some()
        );

        let restored = EquilibriumState::from_lmnkc_bytes(&lmnkc, cfg).expect("from_lmnkc");
        assert_eq!(restored.generation(), eq_a.generation());
        assert!(restored.pager().get_entry(1).unwrap().pinned);
        let restored_env = restored
            .snapshot_envelope_state()
            .expect("restored envelope");
        assert_eq!(restored_env.baseline_pinned_compute_mask & 0b1111, 0b1111);

        // Restore into a live RuntimeContext with a stronger host pin floor (bit 5).
        let live_cfg = Config::new().with_head_dim(64).with_num_heads(8);
        let mut live_ctx = RuntimeContext::new(live_cfg).expect("live ctx");
        live_ctx.apply_host_spine_pin_floor(1_u64 << 5);
        let snap_576 = restored
            .runtime_snapshot_container_bytes()
            .expect("576B snapshot");
        live_ctx
            .deserialize_from_bytes(&snap_576)
            .expect("deserialize into live ctx");
        assert_eq!(
            live_ctx.envelope().baseline_pinned_compute_mask & ((1 << 5) | 0b1111),
            (1 << 5) | 0b1111
        );

        // Legacy .lmnkc without RuntimeSnapshot or EquilibriumMeta loads cleanly.
        let legacy_pager = AnchorWorkingSetPager::from_records(
            &[make_record(42, 420, 1.5)],
            AnchorPagerConfig::default(),
        );
        let legacy_bytes = legacy_pager
            .to_container_bytes(1_774_900_000)
            .expect("legacy");
        let legacy_eq =
            EquilibriumState::from_lmnkc_bytes(&legacy_bytes, cfg).expect("load legacy");
        assert_eq!(legacy_eq.generation(), 0);
        assert!(legacy_eq.runtime_snapshot_payload().is_none());
        assert!(legacy_eq.pager().is_resident(42));
    }

    #[test]
    fn circular_merge_preserves_wraparound_and_avoids_zero_collapse() {
        let a = [126_i8; MAX_ATTENTION_HEADS];
        let b = [-126_i8; MAX_ATTENTION_HEADS];
        let (merged, r_bar) = circular_merge_phase_profiles_q7(&a, 0.5, &b, 0.5);
        // On S^1, +126 and -126 are near +/-pi (+/-127), never 0!
        assert!(
            merged[0].unsigned_abs() >= 125,
            "wraparound collapsed to {}",
            merged[0]
        );
        assert!(r_bar > 0.95);

        // Exact antipodal cancellation retains stronger weight's phase rather than collapsing to 0.
        let pos = [64_i8; MAX_ATTENTION_HEADS];
        let neg = [-64_i8; MAX_ATTENTION_HEADS];
        let (anti, _) = circular_merge_phase_profiles_q7(&pos, 0.6, &neg, 0.4);
        assert_eq!(anti[0], 64);
    }

    #[test]
    fn signed_equilibrium_lmnkc_enforces_policy_and_rejects_downgrade_or_tampering() {
        let cfg = EquilibriumConfig::default();
        let mut eq = EquilibriumState::new(cfg);
        eq.insert_anchor(make_record(7, 77, 1.5), Lacuna::SelfRef, true)
            .expect("insert");
        let kp = SigningKeyPair::from_seed_bytes([0x5A_u8; 32]);
        let wrong_kp = SigningKeyPair::from_seed_bytes([0xA5_u8; 32]);
        let pk = kp.verifying_key();

        let unsigned = eq.to_lmnkc_bytes(1_774_901_000).expect("unsigned");
        assert!(EquilibriumState::from_lmnkc_bytes(&unsigned, cfg).is_ok());
        assert!(
            EquilibriumState::from_lmnkc_bytes_with_policy(
                &unsigned,
                cfg,
                &IntegrityPolicy::RequireSignature(pk),
            )
            .is_err()
        );

        let signed = eq
            .to_signed_lmnkc_bytes(1_774_901_000, &kp)
            .expect("signed");
        let restored = EquilibriumState::from_lmnkc_bytes_with_policy(
            &signed,
            cfg,
            &IntegrityPolicy::RequireSignature(pk),
        )
        .expect("verified restore");
        assert!(restored.pager().get_entry(7).unwrap().pinned);

        assert!(
            EquilibriumState::from_lmnkc_bytes_with_policy(
                &signed,
                cfg,
                &IntegrityPolicy::RequireSignature(wrong_kp.verifying_key()),
            )
            .is_err()
        );

        assert!(
            eq.save_lmnkc_to_path(Path::new("out/CON.lmnkc"), 1_774_901_000)
                .is_err()
        );
    }
}
