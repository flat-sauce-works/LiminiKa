// name: src/core/src/attractor/snapshot.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Crystallized anchor records, 8.25 KB Pod growth snapshots, and domain accumulators.

use super::*;

// ===================================================================
// 3. Pillar 3: Compact `.gcso` Pod Snapshot Structures (`8.25 KB`)
// ===================================================================

/// Compact 128-byte (2-cacheline) [`Pod`] record storing a single crystallized or active sample
/// point on the $SO(2)^{64}$ manifold inside a `.gcso` snapshot (`AttractorGrowthSnapshot`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrystallizedAnchorRecord {
    /// Target phase coordinates across 64 attention heads (Q7 fixed-point, 64 bytes, offset 0..64).
    pub phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Unique anchor identifier (`1`-based, `0` = empty slot, offset 64..68).
    pub id: u32,
    /// Classification type of topological anchor (`gcso_anchor_type_t`, offset 68..72).
    pub anchor_type: gcso_anchor_type_t,
    /// Originating working lane identifier (`0..=7`, offset 72..76).
    pub domain_id: u32,
    /// Half-life in token steps for temporal decay (offset 76..80).
    pub half_life_steps: u32,
    /// Intrinsic mass / baseline salience (offset 80..84).
    pub weight: f32,
    /// Dynamic gravitational pull force $F_{\mathrm{pull}}$ (offset 84..88).
    pub pull_force: f32,
    /// Accumulated stigmergic trail density $\rho_{\mathrm{stigmergy}}$ (offset 88..92).
    pub stigmergic_density: f32,
    /// Circular resultant vector length $\bar{R}_c \in [0, 1]$ of this sample's phase profile (offset 92..96).
    pub resultant_r_bar: f32,
    /// Von Mises concentration parameter $\kappa_c$ of this sample's phase profile (offset 96..100).
    pub von_mises_kappa: f32,
    /// Active flag (`1` = active, `0` = inactive, offset 100).
    pub is_active: u8,
    /// Crystallized flag (`1` = bottom-up crystallized, `0` = transient, offset 101).
    pub is_crystallized: u8,
    /// Explicit 2-byte alignment padding before 8-byte `last_active_step` (offset 102..104).
    pub reserved_flags: u16,
    /// Token step index when this anchor was last activated (offset 104..112).
    pub last_active_step: u64,
    /// Explicit trailing padding to 128-byte (2-cacheline) boundary (offset 112..128).
    pub reserved_padding: [u8; 16],
}

// SAFETY: `CrystallizedAnchorRecord` is `#[repr(C, align(64))]` of size 128 with zero implicit padding traps.
unsafe impl Pod for CrystallizedAnchorRecord {}

impl Default for CrystallizedAnchorRecord {
    #[inline(always)]
    fn default() -> Self {
        Self::zeroed()
    }
}

impl EpisodicPhaseSample for CrystallizedAnchorRecord {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(
            self.is_active != 0,
            branchless_relu_f32(self.weight) * branchless_relu_f32(self.pull_force),
            0.0,
        )
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        branchless_clamp_f32(self.resultant_r_bar, 0.25, 1.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        self.domain_id & ((MAX_ATTRACTOR_DOMAINS as u32) - 1)
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        self.anchor_type
    }
}

impl CrystallizedAnchorRecord {
    /// Zero-copy `const` reference reinterpretation of `self.phase_deltas` (`offset 0..64`,
    /// guaranteed 64-byte aligned by `#[repr(C, align(64))]`) as 8 native `u64` SWAR words (`&[u64; 8]`).
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `CrystallizedAnchorRecord` is `#[repr(C, align(64))]` and `phase_deltas` is at offset 0
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe {
            &*(self
                .phase_deltas
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Mutably borrows `self.phase_deltas` (`offset 0` of `#[repr(C, align(64))]`
    /// [`CrystallizedAnchorRecord`], guaranteed 64-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_mut(&mut self) -> &mut [u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `CrystallizedAnchorRecord` is `#[repr(C, align(64))]` and `phase_deltas` is at offset 0
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe {
            &mut *(self
                .phase_deltas
                .as_mut_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// All-zero 128-byte empty record constant enabling 512-byte GF(2) $L^{32}$ zero-block skipping
    /// during [`AttractorGrowthSnapshot::compute_payload_crc32`].
    pub const EMPTY: Self = Self {
        phase_deltas: [0; MAX_ATTENTION_HEADS],
        id: 0,
        anchor_type: GCSO_ANCHOR_TYPE_SYSTEM_PROMPT,
        domain_id: DOMAIN_SYSTEM_BASELINE,
        half_life_steps: 0,
        weight: 0.0,
        pull_force: 0.0,
        stigmergic_density: 0.0,
        resultant_r_bar: 0.0,
        von_mises_kappa: 0.0,
        is_active: 0,
        is_crystallized: 0,
        reserved_flags: 0,
        last_active_step: 0,
        reserved_padding: [0; 16],
    };

    /// Returns `true` when all 128 bytes (`16 x u64` words) of this record are zero, enabling
    /// sound GF(2) zero-block skipping in [`AttractorGrowthSnapshot::finalize_crc32_for_active_anchors`].
    #[inline(always)]
    #[must_use]
    pub const fn is_zero_blocks_fast(&self) -> bool {
        // SAFETY: `CrystallizedAnchorRecord` is 128 bytes and 64-byte aligned (`>= 8-byte` `u64` alignment).
        let w = unsafe { &*core::ptr::from_ref(self).cast::<[u64; 16]>() };
        (((w[0] | w[1]) | (w[2] | w[3]))
            | ((w[4] | w[5]) | (w[6] | w[7]))
            | ((w[8] | w[9]) | (w[10] | w[11]))
            | ((w[12] | w[13]) | (w[14] | w[15])))
            == 0
    }

    /// Packs a full [`AttractorAnchor`] into a compact 128-byte [`CrystallizedAnchorRecord`],
    /// computing its 64-head circular statistics $(\bar{R}_c, \kappa_c)$ in constant time.
    #[inline]
    #[must_use]
    pub fn from_anchor(anchor: &AttractorAnchor) -> Self {
        let r_bar = circular_r_bar_from_words_64(anchor.phase_words_ref());
        let kappa = branchless_von_mises_kappa(r_bar);
        Self {
            phase_deltas: anchor.phase_deltas,
            id: anchor.id,
            anchor_type: anchor.anchor_type,
            domain_id: anchor.domain_id,
            half_life_steps: anchor.half_life_steps,
            weight: anchor.weight,
            pull_force: anchor.pull_force,
            stigmergic_density: anchor.stigmergic_density,
            resultant_r_bar: r_bar,
            von_mises_kappa: kappa,
            is_active: anchor.is_active,
            is_crystallized: anchor.is_crystallized,
            reserved_flags: 0,
            last_active_step: anchor.last_active_step,
            reserved_padding: [0; 16],
        }
    }

    /// Expands this 128-byte record back into a full 640-byte runtime [`AttractorAnchor`].
    #[inline]
    #[must_use]
    pub const fn to_anchor(&self) -> AttractorAnchor {
        let mut vector_data = [0.0_f32; MAX_EMBEDDING_DIM];
        let mut i = 0usize;
        while i < 16 {
            vector_data[i] = (self.phase_deltas[i] as f32) * (1.0 / 128.0);
            i += 1;
        }
        AttractorAnchor {
            id: self.id,
            anchor_type: self.anchor_type,
            weight: self.weight,
            pull_force: self.pull_force,
            stigmergic_density: self.stigmergic_density,
            half_life_steps: branchless_select_u32(
                self.half_life_steps == 0,
                1000,
                self.half_life_steps,
            ),
            last_active_step: self.last_active_step,
            domain_id: self.domain_id & ((MAX_ATTRACTOR_DOMAINS as u32) - 1),
            vector_dim: 16,
            is_active: self.is_active,
            is_crystallized: self.is_crystallized,
            reserved_padding: [0; 24],
            phase_deltas: self.phase_deltas,
            vector_data,
        }
    }

    /// Writes this 128-byte record directly into `dst` in-place, touching only Cache Line 0
    /// (`offset 0..64`), Cache Line 1 (`phase_deltas`, `offset 64..128`), and the first 16
    /// floats (`64` bytes) of `vector_data` without constructing a 640-byte stack temporary or
    /// zeroing the remaining 448 bytes of unused `vector_data[16..128]`.
    #[inline(always)]
    pub fn write_to_anchor_in_place(
        &self,
        dst: &mut AttractorAnchor,
        domain_override: Option<u32>,
    ) {
        dst.id = self.id;
        dst.anchor_type = self.anchor_type;
        dst.weight = self.weight;
        dst.pull_force = self.pull_force;
        dst.stigmergic_density = self.stigmergic_density;
        dst.half_life_steps =
            branchless_select_u32(self.half_life_steps == 0, 1000, self.half_life_steps);
        dst.last_active_step = self.last_active_step;
        dst.domain_id =
            domain_override.unwrap_or(self.domain_id) & ((MAX_ATTRACTOR_DOMAINS as u32) - 1);
        dst.vector_dim = 16;
        dst.is_active = self.is_active;
        dst.is_crystallized = self.is_crystallized;
        dst.phase_deltas = self.phase_deltas;
        let mut i = 0usize;
        while i < 16 {
            dst.vector_data[i] = (self.phase_deltas[i] as f32) * (1.0 / 128.0);
            i += 1;
        }
    }

    #[allow(clippy::wrong_self_convention)]
    #[inline(always)]
    #[must_use]
    const fn to_crystallized_attractor_record_unstamped(&self) -> CrystallizedAttractorRecord {
        let mut rec = CrystallizedAttractorRecord::new_empty(
            0,
            (self.domain_id as u8) & ((MAX_ATTRACTOR_DOMAINS as u8) - 1),
            0,
        );
        rec.anchor_id = self.id;
        rec.phase_profile = self.phase_deltas;
        rec.sample_count = branchless_select_u32(self.is_active != 0, 1, 0);
        rec.base_energy = branchless_relu_f32(self.weight);
        rec.resultant_r_bar = branchless_clamp_f32(self.resultant_r_bar, 0.0, 1.0);
        rec.von_mises_kappa = branchless_relu_f32(self.von_mises_kappa);
        rec.phase_gravity_mass = branchless_clamp_f32(
            branchless_relu_f32(self.weight) * branchless_relu_f32(self.pull_force),
            0.0,
            MAX_ATTRACTOR_PULL_FORCE,
        );
        rec.provenance_bits = CAUSAL_AXIS_SUBJECTIVE_PERSONA
            | branchless_select_u32(
                self.is_crystallized != 0,
                CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
                0,
            );
        rec.dominant_cause_code = ROOT_CAUSE_NOMINAL;
        rec
    }

    /// Converts this snapshot [`CrystallizedAnchorRecord`] into a 2-Origin Gateway / Online Toroidal
    /// $k$-Means [`CrystallizedAttractorRecord`] (`128B` `Pod`), computing `checksum_crc32` once.
    #[inline]
    #[must_use]
    pub const fn to_crystallized_attractor_record(&self) -> CrystallizedAttractorRecord {
        let mut rec = self.to_crystallized_attractor_record_unstamped();
        rec.checksum_crc32 = rec.compute_payload_crc32_with_words(self.phase_words_ref());
        rec
    }

    /// Constructs a [`CrystallizedAnchorRecord`] from a 2-Origin Gateway / Online Toroidal $k$-Means
    /// [`CrystallizedAttractorRecord`].
    #[inline]
    #[must_use]
    pub const fn from_crystallized_attractor_record(rec: &CrystallizedAttractorRecord) -> Self {
        let active = branchless_select_u8(rec.sample_count > 0, 1, 0);
        Self {
            phase_deltas: rec.phase_profile,
            id: rec.anchor_id,
            anchor_type: GCSO_ANCHOR_TYPE_CRYSTALLIZED,
            domain_id: (rec.dominant_slot as u32) & ((MAX_ATTRACTOR_DOMAINS as u32) - 1),
            half_life_steps: 1000,
            weight: branchless_max_f32(rec.base_energy, 1.0),
            pull_force: branchless_max_f32(rec.phase_gravity_mass, 0.25),
            stigmergic_density: branchless_clamp_f32(
                rec.phase_gravity_mass,
                0.25,
                MAX_STIGMERGIC_DENSITY,
            ),
            resultant_r_bar: rec.resultant_r_bar,
            von_mises_kappa: rec.von_mises_kappa,
            is_active: active,
            is_crystallized: active,
            reserved_flags: 0,
            last_active_step: 0,
            reserved_padding: [0; 16],
        }
    }
}

impl CausalProvenanceSignal for CrystallizedAnchorRecord {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        branchless_select_u32(
            self.is_active != 0,
            CAUSAL_AXIS_SUBJECTIVE_PERSONA
                | branchless_select_u32(
                    self.is_crystallized != 0,
                    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
                    0,
                ),
            0,
        )
    }
}

impl CognitiveGroundingSignal for CrystallizedAnchorRecord {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        Some(&self.phase_deltas)
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        *self.phase_words_ref()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.sample_mass()
    }

    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.id
    }
}

impl SubjectivePersonaSignal for CrystallizedAnchorRecord {}

impl ManifoldGatewaySignal for CrystallizedAnchorRecord {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; crate::traits::NUM_SWARM_MICRO_CELLS]) {
        self.to_crystallized_attractor_record_unstamped()
            .synthesize_gateway_frame_with_words()
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.to_crystallized_attractor_record_unstamped()
            .synthesize_gateway_frame()
    }
}

/// Self-contained 8,448-byte (`8.25 KB`, 132 cachelines) 64-byte-aligned [`Pod`] snapshot structure
/// for `.gcso` ZIMMS persistence and Steam Workshop distribution of a grown personality.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttractorGrowthSnapshot {
    /// Binary magic constant (`GCSO_SNAPSHOT_MAGIC_U32 = 0x4F534347`, offset 0..4).
    pub magic: u32,
    /// Unified ABI version (`GCSO_ABI_VERSION_HEX = 0x00000101`, offset 4..8).
    pub abi_version: u32,
    /// Number of active anchor records stored in `anchors` (`0..=64`, offset 8..12).
    pub active_anchor_count: u32,
    /// Number of bottom-up crystallized anchors in `anchors` (offset 12..16).
    pub crystallized_anchor_count: u32,
    /// Global token step counter at snapshot creation (offset 16..24).
    pub global_step_counter: u64,
    /// Active $8 \times 8$ cross-slot coupling bitmatrix (offset 24..32).
    pub coupling_matrix_8x8: u64,
    /// Synthesized persona resultant vector length $\bar{R}_c \in [0, 1]$ (offset 32..36).
    pub composite_r_bar: f32,
    /// Synthesized persona von Mises concentration $\kappa_c$ (offset 36..40).
    pub composite_kappa: f32,
    /// Total potential energy across the attractor space (offset 40..44).
    pub total_potential_energy: f32,
    /// IEEE 802.3 CRC32 checksum over payload bytes `[64..8448]` (offset 44..48).
    pub payload_crc32: u32,
    /// Active working lane bitmask (`0..=0xFF`, offset 48..52).
    pub active_domains_mask: u32,
    /// Global anti-hallucination repulsion gain factor (offset 52..56).
    pub repulsion_gain: f32,
    /// Explicit header padding to 64-byte cacheline boundary (offset 56..64).
    pub reserved_header: [u8; 8],
    /// Synthesized persistent 64-head Q7 persona phase profile (offset 64..128).
    pub persona_phase_profile: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Global anti-phase repulsion profile vector (offset 128..192).
    pub repulsion_phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Relative working lane weights (`8 * 4 = 32` bytes, offset 192..224).
    pub domain_weights: [f32; MAX_ATTRACTOR_DOMAINS],
    /// Explicit metadata padding to 64-byte cacheline boundary (offset 224..256).
    pub reserved_meta: [u8; 32],
    /// Compact crystallized/active anchor records (`64 * 128 = 8192` bytes, offset 256..8448).
    pub anchors: [CrystallizedAnchorRecord; MAX_ANCHORS_PER_FIELD],
}

// SAFETY: `AttractorGrowthSnapshot` is `#[repr(C, align(64))]` of size 8448 where all fields are `Pod`.
unsafe impl Pod for AttractorGrowthSnapshot {}

impl Default for AttractorGrowthSnapshot {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl AttractorGrowthSnapshot {
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn init_header_and_persona_from_words(
        &mut self,
        global_step_counter: u64,
        coupling_matrix_8x8: u64,
        active_domains_mask: u32,
        domain_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        repulsion_gain: f32,
        repulsion_phase_deltas: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        total_potential_energy: f32,
        run_w: &[u64; 8],
    ) -> usize {
        let prev_valid = (self.magic == GCSO_SNAPSHOT_MAGIC_U32)
            & (self.abi_version == GCSO_ABI_VERSION_HEX)
            & ((self.active_anchor_count as usize) <= MAX_ANCHORS_PER_FIELD);
        let prev_clear_end = if prev_valid {
            self.active_anchor_count as usize
        } else {
            MAX_ANCHORS_PER_FIELD
        };
        self.magic = GCSO_SNAPSHOT_MAGIC_U32;
        self.abi_version = GCSO_ABI_VERSION_HEX;
        self.global_step_counter = global_step_counter;
        self.coupling_matrix_8x8 = coupling_matrix_8x8;
        self.active_domains_mask = active_domains_mask;
        self.domain_weights = *domain_weights;
        self.repulsion_gain = repulsion_gain;
        self.reserved_header = [0; 8];
        self.repulsion_phase_deltas = *repulsion_phase_deltas;
        self.reserved_meta = [0; 32];
        self.total_potential_energy = total_potential_energy;
        store_q7_words_64(&mut self.persona_phase_profile, run_w);
        let r_bar = circular_r_bar_from_words_64(run_w);
        self.composite_r_bar = r_bar;
        self.composite_kappa = branchless_von_mises_kappa(r_bar);
        prev_clear_end
    }
    /// Borrows `self.persona_phase_profile` (`offset 64` of `#[repr(C, align(64))]`
    /// [`AttractorGrowthSnapshot`], guaranteed 64-byte aligned) as a zero-copy `&[u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn persona_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorGrowthSnapshot` is `#[repr(C, align(64))]` and `persona_phase_profile`
        // is at byte offset 64 (64-byte aligned, >= 8-byte alignment of `u64`).
        unsafe {
            &*(self
                .persona_phase_profile
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Borrows `self.repulsion_phase_deltas` (`offset 128` of `#[repr(C, align(64))]`
    /// [`AttractorGrowthSnapshot`], guaranteed 64-byte aligned) as a zero-copy `&[u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn repulsion_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorGrowthSnapshot` is `#[repr(C, align(64))]` and `repulsion_phase_deltas`
        // is at byte offset 128 (64-byte aligned, >= 8-byte alignment of `u64`).
        unsafe {
            &*(self
                .repulsion_phase_deltas
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Constructs an empty, header-initialized `AttractorGrowthSnapshot` stamped with magic `"GCSO"`
    /// (`0x4F534347`) and ABI version `0.1.1` (`0x00000101`).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            magic: GCSO_SNAPSHOT_MAGIC_U32,
            abi_version: GCSO_ABI_VERSION_HEX,
            active_anchor_count: 0,
            crystallized_anchor_count: 0,
            global_step_counter: 0,
            coupling_matrix_8x8: DEFAULT_SLOT_COUPLING_MATRIX_8X8,
            composite_r_bar: 0.0,
            composite_kappa: 0.0,
            total_potential_energy: 0.0,
            payload_crc32: 0,
            active_domains_mask: 0xFF,
            repulsion_gain: 0.0,
            reserved_header: [0; 8],
            persona_phase_profile: [0; MAX_ATTENTION_HEADS],
            repulsion_phase_deltas: [0; MAX_ATTENTION_HEADS],
            domain_weights: [1.0; MAX_ATTRACTOR_DOMAINS],
            reserved_meta: [0; 32],
            anchors: [CrystallizedAnchorRecord::EMPTY; MAX_ANCHORS_PER_FIELD],
        }
    }

    #[inline(always)]
    const fn is_zero_anchor_blocks(b0: &[u64; 8], b1: &[u64; 8]) -> bool {
        if (b1[0] | b1[1] | b1[3]) != 0 {
            return false;
        }
        (((b0[0] | b0[1]) | (b0[2] | b0[3]))
            | ((b0[4] | b0[5]) | (b0[6] | b0[7]))
            | ((b1[2] | b1[4]) | (b1[5] | b1[6]) | b1[7]))
            == 0
    }

    /// Computes the IEEE 802.3 CRC32 checksum over the 64-byte-aligned payload bytes `[64..8448]`
    /// (`131` 64-byte blocks) by aligning the 3 fixed metadata blocks (`blocks[0..3]`) first and
    /// stepping across 4-anchor (`512B`) and 2-anchor (`256B`) zero blocks via GF(2) $L^{32}$ / $L^{16}$ operators.
    #[inline]
    #[must_use]
    pub const fn compute_payload_crc32(&self) -> u32 {
        const PAYLOAD_BLOCKS_64: usize =
            (size_of::<AttractorGrowthSnapshot>() - SNAPSHOT_HEADER_BYTES) / 64;
        // SAFETY: `self` is `#[repr(C, align(64))]` of size 8448 bytes; offset 64 is 64-byte aligned
        // and followed by exactly `131` initialized 64-byte (`[u64; 8]`) blocks.
        let blocks: &[[u64; 8]; PAYLOAD_BLOCKS_64] = unsafe {
            &*(self as *const Self)
                .cast::<u8>()
                .add(SNAPSHOT_HEADER_BYTES)
                .cast::<[[u64; 8]; PAYLOAD_BLOCKS_64]>()
        };
        let c0 = crc32_step_block64_words(0xFFFF_FFFF_u32, &blocks[0]);
        let c1 = crc32_step_block64_words(c0, &blocks[1]);
        let mut crc = crc32_step_block64_words(c1, &blocks[2]);

        let mut i = 3usize;
        while i + 8 <= PAYLOAD_BLOCKS_64 {
            let z_a0 = Self::is_zero_anchor_blocks(&blocks[i], &blocks[i + 1]);
            let z_a1 = Self::is_zero_anchor_blocks(&blocks[i + 2], &blocks[i + 3]);
            if z_a0 && z_a1 {
                if Self::is_zero_anchor_blocks(&blocks[i + 4], &blocks[i + 5])
                    && Self::is_zero_anchor_blocks(&blocks[i + 6], &blocks[i + 7])
                {
                    crc = crc32_step_l32(crc);
                    i += 8;
                    continue;
                }
                crc = crc32_step_l16(crc);
                i += 4;
                continue;
            }
            let p01 = if z_a0 {
                0
            } else {
                let z0 = crc32_block64_zero_seed_words(&blocks[i]);
                crc32_step_block64_words(z0, &blocks[i + 1])
            };
            let p23 = if z_a1 {
                0
            } else {
                let a2 = crc32_block64_zero_seed_words(&blocks[i + 2]);
                crc32_step_block64_words(a2, &blocks[i + 3])
            };
            crc = crc32_step_l16(crc) ^ (if p01 == 0 { 0 } else { crc32_step_l8(p01) }) ^ p23;
            i += 4;
        }
        while i + 4 <= PAYLOAD_BLOCKS_64 {
            let z_a0 = Self::is_zero_anchor_blocks(&blocks[i], &blocks[i + 1]);
            let z_a1 = Self::is_zero_anchor_blocks(&blocks[i + 2], &blocks[i + 3]);
            if z_a0 && z_a1 {
                crc = crc32_step_l16(crc);
            } else {
                let p01 = if z_a0 {
                    0
                } else {
                    let z0 = crc32_block64_zero_seed_words(&blocks[i]);
                    crc32_step_block64_words(z0, &blocks[i + 1])
                };
                let p23 = if z_a1 {
                    0
                } else {
                    let a2 = crc32_block64_zero_seed_words(&blocks[i + 2]);
                    crc32_step_block64_words(a2, &blocks[i + 3])
                };
                crc = crc32_step_l16(crc) ^ (if p01 == 0 { 0 } else { crc32_step_l8(p01) }) ^ p23;
            }
            i += 4;
        }
        !crc
    }

    /// Computes the IEEE 802.3 CRC32 checksum over payload bytes `[64..8448]` when
    /// `self.anchors[active_anchors..64]` is guaranteed to be 100% zero bytes
    /// ([`CrystallizedAnchorRecord::EMPTY`]), folding the trailing `(64 - active_anchors)`
    /// zero anchor slots via $\mathcal{O}(1)$ GF(2) $L^{32}$ / $L^{16}$ / $L^8$ operators
    /// without reading `7.8 KB` of zero-initialized tail memory.
    #[inline]
    #[must_use]
    pub const fn compute_payload_crc32_with_known_zero_tail(&self, active_anchors: usize) -> u32 {
        const PAYLOAD_BLOCKS_64: usize =
            (size_of::<AttractorGrowthSnapshot>() - SNAPSHOT_HEADER_BYTES) / 64;
        let active = if active_anchors < MAX_ANCHORS_PER_FIELD {
            active_anchors
        } else {
            MAX_ANCHORS_PER_FIELD
        };
        // SAFETY: `self` is `#[repr(C, align(64))]` of size 8448 bytes; offset 64 is 64-byte aligned
        // and followed by `131` initialized 64-byte (`[u64; 8]`) blocks.
        let blocks: &[[u64; 8]; PAYLOAD_BLOCKS_64] = unsafe {
            &*(self as *const Self)
                .cast::<u8>()
                .add(SNAPSHOT_HEADER_BYTES)
                .cast::<[[u64; 8]; PAYLOAD_BLOCKS_64]>()
        };
        let c0 = crc32_step_block64_words(0xFFFF_FFFF_u32, &blocks[0]);
        let c1 = crc32_step_block64_words(c0, &blocks[1]);
        let mut crc = crc32_step_block64_words(c1, &blocks[2]);

        let active_end_block = 3 + active * 2;
        let mut i = 3usize;
        while i + 4 <= active_end_block {
            let z_a0 = Self::is_zero_anchor_blocks(&blocks[i], &blocks[i + 1]);
            let z_a1 = Self::is_zero_anchor_blocks(&blocks[i + 2], &blocks[i + 3]);
            if z_a0 && z_a1 {
                crc = crc32_step_l16(crc);
            } else {
                let p01 = if z_a0 {
                    0
                } else {
                    let z0 = crc32_block64_zero_seed_words(&blocks[i]);
                    crc32_step_block64_words(z0, &blocks[i + 1])
                };
                let p23 = if z_a1 {
                    0
                } else {
                    let a2 = crc32_block64_zero_seed_words(&blocks[i + 2]);
                    crc32_step_block64_words(a2, &blocks[i + 3])
                };
                crc = crc32_step_l16(crc) ^ (if p01 == 0 { 0 } else { crc32_step_l8(p01) }) ^ p23;
            }
            i += 4;
        }
        if i + 2 <= active_end_block {
            let z_a0 = Self::is_zero_anchor_blocks(&blocks[i], &blocks[i + 1]);
            if z_a0 {
                crc = crc32_step_l8(crc);
            } else {
                let z0 = crc32_step_block64_words(crc, &blocks[i]);
                crc = crc32_step_block64_words(z0, &blocks[i + 1]);
            }
        }
        let mut rem_zero_anchors = MAX_ANCHORS_PER_FIELD - active;
        while rem_zero_anchors >= 4 {
            crc = crc32_step_l32(crc);
            rem_zero_anchors -= 4;
        }
        if rem_zero_anchors >= 2 {
            crc = crc32_step_l16(crc);
            rem_zero_anchors -= 2;
        }
        if rem_zero_anchors >= 1 {
            crc = crc32_step_l8(crc);
        }
        !crc
    }

    /// Computes and stamps the IEEE 802.3 CRC32 checksum over payload bytes `[64..8448]`.
    #[inline]
    pub fn finalize_crc32(&mut self) -> u32 {
        let crc = self.compute_payload_crc32();
        self.payload_crc32 = crc;
        crc
    }

    /// Computes and stamps the IEEE 802.3 CRC32 checksum over payload bytes `[64..8448]`
    /// when `self.anchors[active_anchors..64]` is guaranteed to be zeroed.
    #[inline]
    pub fn finalize_crc32_for_active_anchors(&mut self, active_anchors: usize) -> u32 {
        let crc = self.compute_payload_crc32_with_known_zero_tail(active_anchors);
        self.payload_crc32 = crc;
        crc
    }

    /// Verifies the snapshot magic (`0x4F534347`), ABI version (`0x00000101`), anchor count bounds,
    /// and IEEE 802.3 CRC32 payload checksum.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if magic, bounds, or CRC32 verification fails.
    /// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `abi_version != GCSO_ABI_VERSION_HEX`.
    #[inline]
    pub fn verify_integrity(&self) -> GcsoResult<()> {
        if self.magic != GCSO_SNAPSHOT_MAGIC_U32 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        if self.abi_version != GCSO_ABI_VERSION_HEX {
            return Err(GCSO_ERROR_VERSION_MISMATCH);
        }
        if (self.active_anchor_count as usize) > MAX_ANCHORS_PER_FIELD
            || self.crystallized_anchor_count > self.active_anchor_count
        {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        let computed_crc = self.compute_payload_crc32();
        if self.payload_crc32 != computed_crc {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        Ok(())
    }

    /// Zero-copy casts and verifies an `AttractorGrowthSnapshot` directly from a 64-byte-aligned
    /// `.gcso` ZIMMS memory-mapped slice at `offset`.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes` cannot hold `size_of::<Self>()`,
    /// [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`] if `offset` is not 64-byte aligned, or
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`] / [`GCSO_ERROR_VERSION_MISMATCH`] if integrity checks fail.
    #[inline]
    pub fn from_bytes_verified_at(bytes: &[u8], offset: usize) -> GcsoResult<&Self> {
        let snap = Self::from_bytes_at(bytes, offset)?;
        snap.verify_integrity()?;
        Ok(snap)
    }
}

// ===================================================================
// 4. Domain Accumulator Summary & Single Phase-Field Anchor
// ===================================================================

/// Summary descriptor capturing accumulator pressure metrics for a single attractor lane.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DomainAccumulatorSummary {
    /// Lane / domain identifier (`0..=7`).
    pub domain_id: u32,
    /// Number of active anchors registered in this lane.
    pub active_count: u32,
    /// Maximum anchor capacity of this lane (`<= 64`).
    pub capacity: u32,
    /// Number of bottom-up crystallized (invariant) anchors.
    pub crystallized_count: u32,
    /// Accumulator pressure ratio (`active_count / capacity`).
    pub pressure_ratio: f32,
    /// Total potential energy accumulated across active anchors.
    pub total_potential_energy: f32,
    /// Accumulated energy dissipated during evaporative displacement.
    pub dissipated_energy_total: f32,
    /// Autonomous or user-configured relative superposition weight scalar.
    pub domain_weight: f32,
}

// SAFETY: `DomainAccumulatorSummary` is a 32-byte `#[repr(C, align(16))]` struct of primitive `u32`/`f32` fields.
unsafe impl Pod for DomainAccumulatorSummary {}

/// Single topological anchor point living on the unified $SO(2)^{64}$ phase field (`640` bytes = `10` cache lines).
///
/// # 3-Tier Cache-Line Layout (`#[repr(C, align(64))]`)
/// - **Cache Line 0 (`offset 0..64`)**: Hot scalar metadata (`id`, `anchor_type`, `weight`,
///   `pull_force`, `stigmergic_density`, `half_life_steps`, `last_active_step`, `domain_id`,
///   `vector_dim`, `is_active`, `is_crystallized`, `reserved_padding`) so active-slot screening
///   and `compute_energy_score` touch strictly a single 64-byte L1D cache line.
/// - **Cache Line 1 (`offset 64..128`)**: `phase_deltas: [gcso_q7_t; 64]`, strictly 64-byte
///   cacheline-aligned so 8-word SWAR (`phase_words_ref`) and AVX2 loads never cross a cache line.
/// - **Cache Lines 2..9 (`offset 128..640`)**: Cold `vector_data: [f32; 128]` (`512` bytes),
///   untouched during hot/mezzo phase folding.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttractorAnchor {
    /// Unique anchor identifier (1-based index, `0` indicates unallocated slot).
    pub id: u32,
    /// Classification type of topological anchor (`gcso_anchor_type_t`).
    pub anchor_type: gcso_anchor_type_t,
    /// Intrinsic mass / baseline salience of the anchor.
    pub weight: f32,
    /// Dynamic gravitational pull force exerted by the anchor ($F_{\mathrm{pull}}$).
    pub pull_force: f32,
    /// Accumulated stigmergic trail density ($\rho_{\mathrm{stigmergy}}$).
    pub stigmergic_density: f32,
    /// Half-life in token steps for temporal energy decay under accumulator pressure.
    pub half_life_steps: u32,
    /// Token step index when this anchor was last activated or traversed.
    pub last_active_step: u64,
    /// Originating working lane identifier (`0..=7`) associated with this anchor.
    pub domain_id: u32,
    /// Active dimension of stored vector data (`0..=128`).
    pub vector_dim: u16,
    /// Flag indicating whether anchor is active (`1` = active, `0` = inactive).
    pub is_active: u8,
    /// Flag indicating whether anchor was bottom-up crystallized from high-density trails (`1` or `0`).
    pub is_crystallized: u8,
    /// Explicit padding completing Cache Line 0 (`offset 40..64`).
    pub reserved_padding: [u8; 24],
    /// Target phase coordinates across 64 attention heads (Q7 fixed-point, Cache Line 1: `offset 64..128`).
    pub phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Continuous feature embedding vector buffer (up to 128 dimensions, Cache Lines 2..9: `offset 128..640`).
    pub vector_data: [f32; MAX_EMBEDDING_DIM],
}

// SAFETY: `AttractorAnchor` is `#[repr(C, align(64))]` with explicit padding and primitive fields where all bit patterns are valid.
unsafe impl Pod for AttractorAnchor {}

impl AttractorAnchor {
    /// Deactivates this anchor slot in-place by clearing only its active/crystallized flags,
    /// ID, and scalar weights (`20` bytes instead of writing a full `640`-byte `AttractorAnchor::default()`),
    /// eliminating `620` bytes of dead `vector_data` / `phase_deltas` memory traffic per deactivated slot.
    #[inline(always)]
    pub const fn deactivate_in_place(&mut self) {
        self.id = 0;
        self.weight = 0.0;
        self.pull_force = 0.0;
        self.stigmergic_density = 0.0;
        self.is_active = 0;
        self.is_crystallized = 0;
    }
}

impl Default for AttractorAnchor {
    #[inline(always)]
    fn default() -> Self {
        Self {
            id: 0,
            anchor_type: GCSO_ANCHOR_TYPE_SYSTEM_PROMPT,
            weight: 1.0,
            pull_force: 1.0,
            stigmergic_density: 1.0,
            half_life_steps: 1000,
            last_active_step: 0,
            domain_id: DOMAIN_SYSTEM_BASELINE,
            vector_dim: 0,
            is_active: 0,
            is_crystallized: 0,
            reserved_padding: [0; 24],
            phase_deltas: [0; MAX_ATTENTION_HEADS],
            vector_data: [0.0; MAX_EMBEDDING_DIM],
        }
    }
}

impl EpisodicPhaseSample for AttractorAnchor {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(
            self.is_active != 0,
            branchless_relu_f32(self.weight) * branchless_relu_f32(self.pull_force),
            0.0,
        )
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        let r_bar = circular_r_bar_from_words_64(self.phase_words_ref());
        branchless_clamp_f32(r_bar, 0.25, 1.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        self.domain_id & ((MAX_ATTRACTOR_DOMAINS as u32) - 1)
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        self.anchor_type
    }
}

impl CausalProvenanceSignal for AttractorAnchor {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        branchless_select_u32(
            self.is_active != 0,
            CAUSAL_AXIS_SUBJECTIVE_PERSONA
                | branchless_select_u32(
                    self.is_crystallized != 0,
                    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
                    0,
                ),
            0,
        )
    }
}

impl CognitiveGroundingSignal for AttractorAnchor {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        Some(&self.phase_deltas)
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        *self.phase_words_ref()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.sample_mass()
    }

    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.id
    }
}

impl SubjectivePersonaSignal for AttractorAnchor {}

impl ManifoldGatewaySignal for AttractorAnchor {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; crate::traits::NUM_SWARM_MICRO_CELLS]) {
        if self.is_active == 0 {
            return (
                GatewayFrame::zeroed(),
                [0u64; crate::traits::NUM_SWARM_MICRO_CELLS],
            );
        }
        CrystallizedAnchorRecord::from_anchor(self).synthesize_gateway_frame_with_words()
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        if self.is_active == 0 {
            return GatewayFrame::zeroed();
        }
        CrystallizedAnchorRecord::from_anchor(self).synthesize_gateway_frame()
    }
}

impl AttractorAnchor {
    /// Zero-copy `const` reference reinterpretation of `self.phase_deltas` (`offset 64..128`,
    /// guaranteed 64-byte cacheline-aligned by `#[repr(C, align(64))]`) as 8 native `u64` SWAR words (`&[u64; 8]`).
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorAnchor` is `#[repr(C, align(64))]` and `phase_deltas` is at byte offset 64
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe {
            &*(self
                .phase_deltas
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Mutably borrows `self.phase_deltas` (`offset 64` of `#[repr(C, align(64))]`
    /// [`AttractorAnchor`], guaranteed 64-byte cacheline-aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_mut(&mut self) -> &mut [u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorAnchor` is `#[repr(C, align(64))]` and `phase_deltas` is at byte offset 64
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe {
            &mut *(self
                .phase_deltas
                .as_mut_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Computes composite potential energy score with division-free (`0 fdiv`) branchless
    /// temporal decay under Accumulator Pressure.
    #[inline(always)]
    #[must_use]
    pub const fn compute_energy_score(&self, current_step: u64) -> f32 {
        let active = self.is_active != 0;
        let dt = current_step.saturating_sub(self.last_active_step) as f32;
        let half_life = branchless_select_f32(
            self.half_life_steps == 0,
            1000.0,
            self.half_life_steps as f32,
        );
        let inv_half_life = fast_inv_pos_normal_f32(half_life);
        let temporal_decay = fast_inv_pos_normal_f32(1.0 + dt * inv_half_life);
        let clean_w = branchless_relu_f32(self.weight);
        let clean_p = branchless_relu_f32(self.pull_force);
        let clean_rho = branchless_clamp_f32(self.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
        let base_energy = clean_w * clean_p + clean_rho;
        let energy = branchless_flush_subnormal_f32(base_energy * temporal_decay);
        branchless_select_f32(active, energy, 0.0)
    }

    /// Evaluates **Zero-Hyperparameter Phase-Proximity Gravity** on the unified phase field
    /// in strictly constant time (`0` branches on phase non-zero checks, `0 fdiv`) by delegating
    /// to register-resident [`branchless_q7_resonant_pull_64`].
    ///
    /// Returns `(effective_pull_force, phase_coherence)`.
    #[inline(always)]
    #[must_use]
    pub fn compute_resonant_pull(
        &self,
        current_step: u64,
        trail: &gcso_pointer_trail_t,
    ) -> (f32, f32) {
        let active = self.is_active != 0;
        let energy = self.compute_energy_score(current_step);
        let base_pull = branchless_select_f32(
            active,
            branchless_relu_f32(self.weight) * branchless_relu_f32(self.pull_force) * energy,
            0.0,
        );
        let (res_pull, coh) = branchless_q7_resonant_pull_64(
            &trail.accumulated_phase_delta,
            &self.phase_deltas,
            base_pull,
        );
        (
            branchless_select_f32(
                active,
                branchless_min_f32(res_pull, MAX_ATTRACTOR_PULL_FORCE),
                0.0,
            ),
            branchless_select_f32(active, coh, 0.0),
        )
    }

    /// Recharges anchor activation timestamp and boosts stigmergic density bounded by [`MAX_STIGMERGIC_DENSITY`].
    #[inline(always)]
    pub fn touch(&mut self, current_step: u64, density_delta: f32) {
        self.last_active_step = current_step;
        let boost = branchless_flush_subnormal_f32(branchless_relu_f32(density_delta));
        self.stigmergic_density = branchless_min_f32(
            branchless_relu_f32(self.stigmergic_density) + boost,
            MAX_STIGMERGIC_DENSITY,
        );
    }
}
