// name: src/core/src/drivers/ego/differential.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-parametric directed P-Frame residual network ([`EgoDifferentialNet`]) and
//! directed P-Frame [`DistortionDelta`] packets (128 B) evaluating word-level P-Frame
//! ensembles and chains without importing `SystemSpineState` or `GatewayFrame`.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::mem::{align_of, offset_of, size_of};

use crate::drivers::ego::intent::{EgoIntentAxes, LiminalEgoState};
use crate::foundation::{
    branchless_q7_weighted_ensemble_contiguous_words_64, load_q7_words_64, swar_q7_circular_add_x8,
    swar_q7_scale_le128_with_bias_x8, Pod, SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{
    branchless_select_u32, branchless_select_u8, AttributionTag, BitCut8, LacunaPartition8,
    PhaseQ7, Scoped, ScopedCut, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
    SCOPE_REGION_EXTERIOR, SCOPE_REGION_INTERIOR, SCOPE_REGION_SELF_REF,
};
use crate::stigmergy::anchor::{PhaseAnchorRecord, PhaseProfile64};
use crate::stigmergy::granule::StigmergicGranuleState;
use crate::stigmergy::ATTRACTOR_SLOT_SYSTEM_BASELINE;

/// Maximum number of inline P-Frame [`DistortionDelta`] nodes hosted by [`EgoDifferentialNet`].
pub const MAX_EGO_PFRAME_NODES: usize = 8;

/// Sentinel next-hop index marking termination of a [`DistortionDelta`] pointer chain.
pub const DISTORTION_DELTA_NO_NEXT_HOP: u16 = 0xFFFF;

/// Default maximum hop depth allowed when evaluating branched [`DistortionDelta`] chains (`4`).
pub const MAX_DISTORTION_CHAIN_HOPS: u8 = 4;

/// Flag bit `0`: [`DistortionDelta`] node is active and eligible for lacuna trigger matching.
pub const DISTORTION_DELTA_FLAG_ACTIVE: u32 = 1 << 0;

/// Flag bit `1`: [`DistortionDelta`] node was consolidated into an I-Frame phase anchor.
pub const DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME: u32 = 1 << 1;

/// Flag bit `2`: [`DistortionDelta`] node carries valid model-baked 64-head Q7 phase deltas.
pub const DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID: u32 = 1 << 2;

/// First-class 128-byte (`#[repr(C, align(64))]`) directed P-Frame differential node representing
/// a gauge-relative deviation $\vec{\Delta}_t$ from `SystemSpineState` trajectory $f_{\mathrm{sys}}$.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistortionDelta {
    /// Logical P-Frame node identifier (`offset 0..4`).
    pub node_id: u32,
    /// Reference logical I-Frame anchor identifier (`offset 4..8`).
    pub anchor_ref_id: u32,
    /// Next chained P-Frame node index (`0..8`, or `0xFFFF` if terminal, `offset 8..10`).
    pub next_hop_idx: u16,
    /// Hop depth along the chain (`1..=4`, `offset 10`).
    pub hop_depth: u8,
    /// 8-granule lacuna support bitmask required to trigger this delta (`offset 11`).
    pub lacuna_trigger_mask: u8,
    /// Bitmask of allowed scope regions (`bit 0=S&O`, `bit 1=S\O`, `bit 2=O\S`, `offset 12`).
    pub scope_region_selector: u8,
    /// Stigmergic trace density in Q0.7 (`0..=127`, `offset 13`).
    pub pheromone_q7: u8,
    /// Directed delta scaling gain in Q0.7 (`0..=127`, `offset 14`).
    pub gain_q7: u8,
    /// 2-bit [`AttributionTag`] (`Stigmergy`/`Ego`, `offset 15`).
    pub attribution_tag: u8,
    /// Model-agnostic logical `EgoIntentAxes` transition (`16 B`, `offset 16..32`).
    pub intent_delta: EgoIntentAxes,
    /// Accumulated positive reinforcement counter for Cold-Path I-Frame consolidation (`offset 32..36`).
    pub reinforce_count: u32,
    /// Last active step epoch for stigmergic evaporation pruning (`offset 36..40`).
    pub last_active_epoch: u32,
    /// Optional integrity checksum over logical delta payload (`offset 40..44`).
    pub checksum_crc32: u32,
    /// Lifecycle and validity bitflags (`DISTORTION_DELTA_FLAG_*`, `offset 44..48`).
    pub flags: u32,
    /// Reserved aligned header bytes (`offset 48..64`).
    pub reserved_header: [u8; 16],
    /// Model-baked 64-head gauge-relative phase delta $\vec{\Delta}_t$ (`offset 64..128`).
    pub delta_phase_q7: [PhaseQ7; MAX_ATTENTION_HEADS],
}

impl Default for DistortionDelta {
    #[inline(always)]
    fn default() -> Self {
        Self::new_empty(0, 0)
    }
}

impl DistortionDelta {
    /// Constructs a zeroed inactive 128-byte [`DistortionDelta`] packet.
    #[inline(always)]
    #[must_use]
    pub const fn new_empty(node_id: u32, anchor_ref_id: u32) -> Self {
        Self {
            node_id,
            anchor_ref_id,
            next_hop_idx: DISTORTION_DELTA_NO_NEXT_HOP,
            hop_depth: 1,
            lacuna_trigger_mask: 0xFF,
            scope_region_selector: SCOPE_REGION_SELF_REF | SCOPE_REGION_INTERIOR,
            pheromone_q7: 64,
            gain_q7: 64,
            attribution_tag: AttributionTag::STIGMERGY_EGO.raw(),
            intent_delta: EgoIntentAxes::new(),
            reinforce_count: 0,
            last_active_epoch: 0,
            checksum_crc32: 0,
            flags: 0,
            reserved_header: [0; 16],
            delta_phase_q7: [0; MAX_ATTENTION_HEADS],
        }
    }

    /// Returns `true` when this node is marked active (`DISTORTION_DELTA_FLAG_ACTIVE`).
    #[inline(always)]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        (self.flags & DISTORTION_DELTA_FLAG_ACTIVE) != 0
    }

    /// Returns `true` when this node is active (`DISTORTION_DELTA_FLAG_ACTIVE`) and carries
    /// valid model-baked 64-head deltas (`DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID`).
    #[inline(always)]
    #[must_use]
    pub const fn is_baked_active(&self) -> bool {
        const MASK: u32 = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        (self.flags & MASK) == MASK
    }
}

// SAFETY: `DistortionDelta` is `#[repr(C, align(64))]`, exact size 128 bytes with zero padding.
unsafe impl Pod for DistortionDelta {}

/// Compact 8-byte (`#[repr(C, align(8))]`) hot-path routing and gating descriptor for one P-Frame
/// node inside [`EgoDifferentialNet`].
///
/// Eight [`PFrameRoutingMeta`] descriptors fit into a single 64-byte cache line (`CL0`), allowing
/// 1-hop parallel perspective screening without touching phase arrays or cold provenance fields.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PFrameRoutingMeta {
    /// Next chained P-Frame node index (`0..8`, or `0xFFFF` if terminal, `offset 0..2`).
    pub next_hop_idx: u16,
    /// Hop depth along the chain (`1..=4`, `offset 2`).
    pub hop_depth: u8,
    /// 8-granule lacuna support bitmask required to trigger this delta (`offset 3`).
    pub lacuna_trigger_mask: u8,
    /// Bitmask of allowed scope regions (`bit 0=S&O`, `bit 1=S\O`, `bit 2=O\S`, `offset 4`).
    pub scope_region_selector: u8,
    /// Stigmergic trace density in Q0.7 (`0..=127`, `offset 5`).
    pub pheromone_q7: u8,
    /// Directed delta scaling gain in Q0.7 (`0..=127`, `offset 6`).
    pub gain_q7: u8,
    /// Packed 2-bit [`AttributionTag`] (`bits 0..=1`) and lifecycle flags (`bits 2..=4`, `offset 7`).
    pub flags_and_tag: u8,
}

impl PFrameRoutingMeta {
    /// Bit `2`: node is active and eligible for lacuna trigger matching.
    pub const FLAG_ACTIVE_BIT: u8 = 1 << 2;
    /// Bit `3`: node was consolidated into an I-Frame phase anchor.
    pub const FLAG_CONSOLIDATED_BIT: u8 = 1 << 3;
    /// Bit `4`: node carries valid model-baked 64-head Q7 phase deltas.
    pub const FLAG_BAKED_VALID_BIT: u8 = 1 << 4;
    /// Low 2-bit mask extracting [`AttributionTag::raw()`].
    pub const TAG_MASK: u8 = 0x03;

    /// Constructs an inactive default 8-byte [`PFrameRoutingMeta`].
    #[inline(always)]
    #[must_use]
    pub const fn new_empty() -> Self {
        Self {
            next_hop_idx: DISTORTION_DELTA_NO_NEXT_HOP,
            hop_depth: 1,
            lacuna_trigger_mask: 0xFF,
            scope_region_selector: SCOPE_REGION_SELF_REF | SCOPE_REGION_INTERIOR,
            pheromone_q7: 64,
            gain_q7: 64,
            flags_and_tag: AttributionTag::STIGMERGY_EGO.raw() & Self::TAG_MASK,
        }
    }

    /// Packs `DISTORTION_DELTA_FLAG_*` bits and a 2-bit [`AttributionTag`] into `flags_and_tag`.
    #[inline(always)]
    #[must_use]
    pub const fn pack_flags_and_tag(distortion_flags: u32, tag: u8) -> u8 {
        let a = if (distortion_flags & DISTORTION_DELTA_FLAG_ACTIVE) != 0 {
            Self::FLAG_ACTIVE_BIT
        } else {
            0
        };
        let c = if (distortion_flags & DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME) != 0 {
            Self::FLAG_CONSOLIDATED_BIT
        } else {
            0
        };
        let b = if (distortion_flags & DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID) != 0 {
            Self::FLAG_BAKED_VALID_BIT
        } else {
            0
        };
        (tag & Self::TAG_MASK) | a | c | b
    }

    /// Unpacks lifecycle flags back into `DISTORTION_DELTA_FLAG_*` bits.
    #[inline(always)]
    #[must_use]
    pub const fn raw_distortion_flags(self) -> u32 {
        let a = if (self.flags_and_tag & Self::FLAG_ACTIVE_BIT) != 0 {
            DISTORTION_DELTA_FLAG_ACTIVE
        } else {
            0
        };
        let c = if (self.flags_and_tag & Self::FLAG_CONSOLIDATED_BIT) != 0 {
            DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME
        } else {
            0
        };
        let b = if (self.flags_and_tag & Self::FLAG_BAKED_VALID_BIT) != 0 {
            DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID
        } else {
            0
        };
        a | c | b
    }

    /// Extracts the 2-bit [`AttributionTag`] code (`0..=3`).
    #[inline(always)]
    #[must_use]
    pub const fn attribution_tag(self) -> u8 {
        self.flags_and_tag & Self::TAG_MASK
    }

    /// Returns `true` when this node is marked active.
    #[inline(always)]
    #[must_use]
    pub const fn is_active(self) -> bool {
        (self.flags_and_tag & Self::FLAG_ACTIVE_BIT) != 0
    }

    /// Returns `true` when this node is active and carries valid model-baked phase deltas.
    #[inline(always)]
    #[must_use]
    pub const fn is_baked_active(self) -> bool {
        const MASK: u8 =
            PFrameRoutingMeta::FLAG_ACTIVE_BIT | PFrameRoutingMeta::FLAG_BAKED_VALID_BIT;
        (self.flags_and_tag & MASK) == MASK
    }

    /// Returns `true` when this node has been consolidated into an I-Frame anchor.
    #[inline(always)]
    #[must_use]
    pub const fn is_consolidated(self) -> bool {
        (self.flags_and_tag & Self::FLAG_CONSOLIDATED_BIT) != 0
    }

    /// Sets or clears the active bit in `flags_and_tag`.
    #[inline(always)]
    pub const fn set_active(&mut self, active: bool) {
        if active {
            self.flags_and_tag |= Self::FLAG_ACTIVE_BIT;
        } else {
            self.flags_and_tag &= !Self::FLAG_ACTIVE_BIT;
        }
    }

    /// Sets or clears the consolidated I-Frame bit in `flags_and_tag`.
    #[inline(always)]
    pub const fn set_consolidated(&mut self, consolidated: bool) {
        if consolidated {
            self.flags_and_tag |= Self::FLAG_CONSOLIDATED_BIT;
        } else {
            self.flags_and_tag &= !Self::FLAG_CONSOLIDATED_BIT;
        }
    }
}

impl Default for PFrameRoutingMeta {
    #[inline(always)]
    fn default() -> Self {
        Self::new_empty()
    }
}

// SAFETY: `PFrameRoutingMeta` is `#[repr(C, align(8))]`, exact size 8 bytes with primitive integer fields only.
unsafe impl Pod for PFrameRoutingMeta {}

/// Cold-path provenance and reinforcement metadata (`32 B`, `#[repr(C, align(16))]`) for one
/// P-Frame node inside [`EgoDifferentialNet`], segregated from CL0 hot screening.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PFrameColdMeta {
    /// Model-agnostic logical `EgoIntentAxes` transition (`16 B`, `offset 0..16`).
    pub intent_delta: EgoIntentAxes,
    /// Logical P-Frame node identifier (`offset 16..20`).
    pub node_id: u32,
    /// Reference logical I-Frame anchor identifier (`offset 20..24`).
    pub anchor_ref_id: u32,
    /// Accumulated positive reinforcement counter for Cold-Path I-Frame consolidation (`offset 24..28`).
    pub reinforce_count: u32,
    /// Last active step epoch for stigmergic evaporation pruning (`offset 28..32`).
    pub last_active_epoch: u32,
}

impl PFrameColdMeta {
    /// Constructs an inactive 32-byte [`PFrameColdMeta`] packet.
    #[inline(always)]
    #[must_use]
    pub const fn new_empty(node_id: u32, anchor_ref_id: u32) -> Self {
        Self {
            intent_delta: EgoIntentAxes::new(),
            node_id,
            anchor_ref_id,
            reinforce_count: 0,
            last_active_epoch: 0,
        }
    }
}

impl Default for PFrameColdMeta {
    #[inline(always)]
    fn default() -> Self {
        Self::new_empty(0, 0)
    }
}

// SAFETY: `PFrameColdMeta` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for PFrameColdMeta {}

/// Inline fixed-capacity (`8` nodes, `0` heap allocations) non-parametric P-Frame residual
/// neural network evaluating directed [`DistortionDelta`] pointer chains (up to `max_hops`
/// in `1..=4`) triggered by scope region support and stigmergic density.
///
/// Stored in Structure-of-Arrays (SoA) cacheline layout (`896 B`, `#[repr(C, align(64))]`):
/// - CL0 (`0..64`): `node_routing` (`[PFrameRoutingMeta; 8]`) for 1-CL Hot-Path screening
/// - CL1..8 (`64..576`): `node_phases` (`[[PhaseQ7; 64]; 8]`) for zero-copy ensemble evaluation
/// - CL9..12 (`576..832`): `node_cold` (`[PFrameColdMeta; 8]`) for Cold-Path rebake/consolidation
/// - CL13 (`832..896`): network control scalars
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EgoDifferentialNet {
    /// Hot-path routing and gating metadata (`64 B` = CL0 `0..64`).
    pub node_routing: [PFrameRoutingMeta; MAX_EGO_PFRAME_NODES],
    /// Contiguous 64-head Q7 delta profiles (`512 B` = CL1..8 `64..576`).
    pub node_phases: [[PhaseQ7; MAX_ATTENTION_HEADS]; MAX_EGO_PFRAME_NODES],
    /// Cold-path intent, anchor, and epoch metadata (`256 B` = CL9..12 `576..832`).
    pub node_cold: [PFrameColdMeta; MAX_EGO_PFRAME_NODES],
    /// Monotonic Cold-Path maintenance epoch counter (`offset 832..836`).
    pub epoch: u32,
    /// 8-bit bitmask marking populated nodes (`0..=7`, `offset 836`).
    pub active_node_mask: u8,
    /// Maximum hop traversal bound (`1..=4`, allows clean 1-hop vs multi-hop comparison, `offset 837`).
    pub max_hops: u8,
    /// Minimum hop depth before Cold-Path GOP consolidation crystallizes an I-Frame anchor (`offset 838`).
    pub consolidation_hop_limit: u8,
    /// Stigmergic pheromone floor (`0..=127`) below which unused P-Frame nodes evaporate (`offset 839`).
    pub evaporation_floor_q7: u8,
    /// Hop count traversed on the most recent evaluation (`offset 840`).
    pub last_hops_traversed: u8,
    /// Mean differential magnitude (`0..=127`) observed on the most recent evaluation (`offset 841`).
    pub last_delta_norm_q7: u8,
    /// Explicit zero-padding completing the 896-byte cacheline block (`offset 842..896`).
    pub reserved_tail: [u8; 54],
}

const _: () = {
    assert!(size_of::<PFrameRoutingMeta>() == 8);
    assert!(align_of::<PFrameRoutingMeta>() == 8);
    assert!(size_of::<PFrameColdMeta>() == 32);
    assert!(align_of::<PFrameColdMeta>() == 16);
    assert!(size_of::<EgoDifferentialNet>() == 896);
    assert!(align_of::<EgoDifferentialNet>() == 64);
    assert!(offset_of!(EgoDifferentialNet, node_routing) == 0);
    assert!(offset_of!(EgoDifferentialNet, node_phases) == 64);
    assert!(offset_of!(EgoDifferentialNet, node_cold) == 576);
    assert!(offset_of!(EgoDifferentialNet, epoch) == 832);
};

impl Default for EgoDifferentialNet {
    #[inline(always)]
    fn default() -> Self {
        Self::new_empty(1)
    }
}

impl EgoDifferentialNet {
    /// Creates an empty inert [`EgoDifferentialNet`] with bounded `max_hops` (`1..=4`).
    #[inline(always)]
    #[must_use]
    pub const fn new_empty(max_hops: u8) -> Self {
        let clamped_hops = if max_hops == 0 {
            1
        } else if max_hops > MAX_DISTORTION_CHAIN_HOPS {
            MAX_DISTORTION_CHAIN_HOPS
        } else {
            max_hops
        };
        Self {
            node_routing: [PFrameRoutingMeta::new_empty(); MAX_EGO_PFRAME_NODES],
            node_phases: [[0; MAX_ATTENTION_HEADS]; MAX_EGO_PFRAME_NODES],
            node_cold: [PFrameColdMeta::new_empty(0, 0); MAX_EGO_PFRAME_NODES],
            epoch: 0,
            active_node_mask: 0,
            max_hops: clamped_hops,
            consolidation_hop_limit: 3,
            evaporation_floor_q7: 16,
            last_hops_traversed: 0,
            last_delta_norm_q7: 0,
            reserved_tail: [0; 54],
        }
    }

    /// Initializes a zeroed `dst` allocation of [`EgoDifferentialNet`] in-place without
    /// materializing the full 896-byte struct on the stack.
    ///
    /// # Safety
    /// Caller must ensure `dst` points to writable, zero-initialized, properly aligned memory
    /// for one `EgoDifferentialNet`.
    #[inline]
    pub unsafe fn init_in_place_on_zeroed(dst: *mut Self, max_hops: u8) {
        let clamped_hops = max_hops.clamp(1, MAX_DISTORTION_CHAIN_HOPS);
        // SAFETY: Caller guarantees `dst` is valid, writable, aligned, and zero-initialized.
        unsafe {
            let routing_ptr = core::ptr::addr_of_mut!((*dst).node_routing);
            let mut idx = 0usize;
            while idx < MAX_EGO_PFRAME_NODES {
                core::ptr::write(
                    (*routing_ptr).as_mut_ptr().add(idx),
                    PFrameRoutingMeta::new_empty(),
                );
                idx += 1;
            }
            core::ptr::write(core::ptr::addr_of_mut!((*dst).max_hops), clamped_hops);
            core::ptr::write(core::ptr::addr_of_mut!((*dst).consolidation_hop_limit), 3);
            core::ptr::write(core::ptr::addr_of_mut!((*dst).evaporation_floor_q7), 16);
        }
    }

    /// Returns `true` when slot `idx` is active in both `active_node_mask` and routing metadata.
    #[inline(always)]
    #[must_use]
    pub const fn is_node_active(&self, idx: usize) -> bool {
        if idx >= MAX_EGO_PFRAME_NODES {
            return false;
        }
        ((self.active_node_mask >> idx) & 1) != 0 && self.node_routing[idx].is_active()
    }

    /// Returns `true` when slot `idx` is active and carries valid model-baked phase deltas.
    #[inline(always)]
    #[must_use]
    pub const fn is_node_baked_active(&self, idx: usize) -> bool {
        if idx >= MAX_EGO_PFRAME_NODES {
            return false;
        }
        ((self.active_node_mask >> idx) & 1) != 0 && self.node_routing[idx].is_baked_active()
    }

    /// Reconstructs the 128-byte [`DistortionDelta`] packet view for node `idx`.
    #[inline]
    #[must_use]
    pub const fn export_node(&self, idx: usize) -> DistortionDelta {
        if idx >= MAX_EGO_PFRAME_NODES {
            return DistortionDelta::new_empty(0, 0);
        }
        let r = self.node_routing[idx];
        let c = self.node_cold[idx];
        DistortionDelta {
            node_id: c.node_id,
            anchor_ref_id: c.anchor_ref_id,
            next_hop_idx: r.next_hop_idx,
            hop_depth: r.hop_depth,
            lacuna_trigger_mask: r.lacuna_trigger_mask,
            scope_region_selector: r.scope_region_selector,
            pheromone_q7: r.pheromone_q7,
            gain_q7: r.gain_q7,
            attribution_tag: r.attribution_tag(),
            intent_delta: c.intent_delta,
            reinforce_count: c.reinforce_count,
            last_active_epoch: c.last_active_epoch,
            checksum_crc32: 0,
            flags: r.raw_distortion_flags(),
            reserved_header: [0; 16],
            delta_phase_q7: self.node_phases[idx],
        }
    }

    /// Writes a 128-byte [`DistortionDelta`] packet into SoA slot `idx`, syncing `active_node_mask`.
    #[inline]
    pub fn set_node(&mut self, idx: usize, delta: DistortionDelta) {
        if idx >= MAX_EGO_PFRAME_NODES {
            return;
        }
        self.node_routing[idx] = PFrameRoutingMeta {
            next_hop_idx: delta.next_hop_idx,
            hop_depth: delta.hop_depth,
            lacuna_trigger_mask: delta.lacuna_trigger_mask,
            scope_region_selector: delta.scope_region_selector,
            pheromone_q7: delta.pheromone_q7,
            gain_q7: delta.gain_q7,
            flags_and_tag: PFrameRoutingMeta::pack_flags_and_tag(
                delta.flags,
                delta.attribution_tag,
            ),
        };
        self.node_phases[idx] = delta.delta_phase_q7;
        self.node_cold[idx] = PFrameColdMeta {
            intent_delta: delta.intent_delta,
            node_id: delta.node_id,
            anchor_ref_id: delta.anchor_ref_id,
            reinforce_count: delta.reinforce_count,
            last_active_epoch: delta.last_active_epoch,
        };
        let bit = 1u8 << idx;
        if delta.is_active() {
            self.active_node_mask |= bit;
        } else {
            self.active_node_mask &= !bit;
        }
    }

    /// Deterministically initializes the 3-node P-Frame net from [`LiminalEgoState`] and
    /// [`EgoIntentAxes`], evaluated as a 1-hop parallel perspective ensemble when `max_hops == 1`
    /// or as a sequential multi-hop chain when `max_hops >= 2`:
    /// - Node 0 (`S & O`): persona heading alignment delta
    /// - Node 1 (`S \ O`): sovereign counter-basin stylistic delta
    /// - Node 2 (`O \ S`): transgressive exterior bridge delta
    #[inline]
    #[must_use]
    pub fn from_ego_state_and_axes(
        ego: &LiminalEgoState,
        axes: &EgoIntentAxes,
        max_hops: u8,
    ) -> Self {
        let mut net = Self::new_empty(max_hops);
        if !ego.is_active() {
            return net;
        }

        let make_phase = |base: &[PhaseQ7; NUM_STIGMERGIC_GRANULES],
                          shift: i8,
                          flip_odd: bool|
         -> [PhaseQ7; MAX_ATTENTION_HEADS] {
            let mut out = [0 as PhaseQ7; MAX_ATTENTION_HEADS];
            let mut g = 0usize;
            while g < NUM_STIGMERGIC_GRANULES {
                let gval = i32::from(base[g]);
                let dir = if flip_odd && ((g & 1) == 1) {
                    -gval
                } else {
                    gval
                };
                let s = (dir + i32::from(shift)).clamp(-127, 127) as i8;
                let base_h = g << 3;
                let mut k = 0usize;
                while k < 8 {
                    let mod_off = ((k as i32) - 3) * (i32::from(shift) >> 4);
                    out[base_h + k] = (i32::from(s) + mod_off).clamp(-127, 127) as PhaseQ7;
                    k += 1;
                }
                g += 1;
            }
            out
        };

        let mut n0 = DistortionDelta::new_empty(1, u32::from(ATTRACTOR_SLOT_SYSTEM_BASELINE) + 1);
        n0.next_hop_idx = 1;
        n0.hop_depth = 1;
        n0.lacuna_trigger_mask = if axes.active_granule_bias_mask == 0 {
            0xFF
        } else {
            axes.active_granule_bias_mask
        };
        n0.scope_region_selector = SCOPE_REGION_SELF_REF;
        n0.pheromone_q7 = 88;
        n0.gain_q7 = ego.counter_basin_thrust_q7.clamp(32, 127);
        n0.intent_delta = *axes;
        n0.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        n0.delta_phase_q7 = make_phase(&ego.sovereign_granule_phase_q7, 4, false);

        let mut n1 = DistortionDelta::new_empty(2, u32::from(ATTRACTOR_SLOT_SYSTEM_BASELINE) + 2);
        n1.next_hop_idx = 2;
        n1.hop_depth = 2;
        n1.lacuna_trigger_mask = 0xAA;
        n1.scope_region_selector = SCOPE_REGION_SELF_REF | SCOPE_REGION_INTERIOR;
        n1.pheromone_q7 = 76;
        n1.gain_q7 = ego.irrational_defiance_q7.clamp(24, 127);
        n1.intent_delta = *axes;
        n1.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        n1.delta_phase_q7 = make_phase(
            &ego.sovereign_granule_phase_q7,
            ego.transgression_bias_q7 >> 2,
            true,
        );

        let mut n2 = DistortionDelta::new_empty(3, u32::from(ATTRACTOR_SLOT_SYSTEM_BASELINE) + 3);
        n2.next_hop_idx = DISTORTION_DELTA_NO_NEXT_HOP;
        n2.hop_depth = 3;
        n2.lacuna_trigger_mask = 0x55;
        n2.scope_region_selector =
            SCOPE_REGION_EXTERIOR | SCOPE_REGION_INTERIOR | SCOPE_REGION_SELF_REF;
        n2.pheromone_q7 = 64;
        n2.gain_q7 = ego.transgression_bias_q7.unsigned_abs().clamp(24, 127);
        n2.intent_delta = *axes;
        n2.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        n2.delta_phase_q7 = make_phase(
            &ego.sovereign_granule_phase_q7,
            -(ego.transgression_bias_q7 >> 2),
            false,
        );

        net.set_node(0, n0);
        net.set_node(1, n1);
        net.set_node(2, n2);
        net
    }

    /// Records positive traversal along up to `hops_traversed` chain hops starting at `entry_idx`,
    /// advancing `reinforce_count` and stamping `last_active_epoch = self.epoch` for Cold-Path GOP
    /// consolidation and evaporation pruning.
    #[inline]
    pub fn record_chain_activation(&mut self, entry_idx: usize, hops_traversed: u8) {
        let mut cur = entry_idx;
        let mut h = 0u8;
        let mut visited_mask: u8 = 0;
        let cap = hops_traversed.min(MAX_DISTORTION_CHAIN_HOPS);
        while h < cap && cur < MAX_EGO_PFRAME_NODES {
            let bit = 1u8 << (cur & 7);
            if (visited_mask & bit) != 0 || (self.active_node_mask & bit) == 0 {
                break;
            }
            visited_mask |= bit;
            let cold = &mut self.node_cold[cur];
            cold.reinforce_count = cold.reinforce_count.saturating_add(1);
            cold.last_active_epoch = self.epoch;
            cur = usize::from(self.node_routing[cur].next_hop_idx);
            h = h.saturating_add(1);
        }
        self.last_hops_traversed = h;
    }

    /// Records positive activation for every node bit set in `matched_mask`, advancing
    /// `reinforce_count`, nudging `pheromone_q7` (capped at `127`), and stamping
    /// `last_active_epoch = self.epoch` for Cold-Path GOP consolidation and evaporation pruning.
    #[inline]
    pub fn record_matched_activations(&mut self, matched_mask: u8) {
        let active_matched = matched_mask & self.active_node_mask;
        self.last_hops_traversed = if self.max_hops <= 1 {
            u8::from(active_matched != 0)
        } else {
            (active_matched.count_ones() as u8).min(self.max_hops)
        };
        for (idx, (routing, cold)) in self
            .node_routing
            .iter_mut()
            .zip(self.node_cold.iter_mut())
            .enumerate()
            .take(MAX_EGO_PFRAME_NODES)
        {
            if ((active_matched >> idx) & 1) != 0 {
                cold.reinforce_count = cold.reinforce_count.saturating_add(1);
                routing.pheromone_q7 = routing.pheromone_q7.saturating_add(1).min(127);
                cold.last_active_epoch = self.epoch;
            }
        }
    }

    /// Cold-Path GOP maintenance: consolidates reinforced multi-hop P-Frame chains into an I-Frame
    /// [`PhaseAnchorRecord`] and evaporates stale unreinforced P-Frame nodes below `evaporation_floor_q7`.
    #[inline]
    #[must_use]
    pub fn consolidate_and_evaporate_cold_path(
        &mut self,
        current_epoch: u32,
    ) -> Option<PhaseAnchorRecord> {
        self.epoch = current_epoch;
        let mut crystallized: Option<PhaseAnchorRecord> = None;

        for (idx, ((routing, phases), cold)) in self
            .node_routing
            .iter_mut()
            .zip(self.node_phases.iter())
            .zip(self.node_cold.iter())
            .enumerate()
            .take(MAX_EGO_PFRAME_NODES)
        {
            let bit = 1u8 << idx;
            if (self.active_node_mask & bit) == 0 {
                continue;
            }
            if !routing.is_active() {
                self.active_node_mask &= !bit;
                continue;
            }
            if routing.hop_depth >= self.consolidation_hop_limit
                && cold.reinforce_count >= 4
                && !routing.is_consolidated()
            {
                let bridge_count = cold.reinforce_count.min(u32::from(u8::MAX)) as u8;
                let grounded_profile = match Scoped::from_delta(PhaseProfile64(*phases))
                    .cut_by_lacuna_q7(true, routing.gain_q7.max(1), 0)
                {
                    ScopedCut::SelfRef(aligned) => Some(aligned.into_grounded()),
                    ScopedCut::Exterior(spec) => spec
                        .bridge_to_shared(bridge_count)
                        .ok()
                        .map(Scoped::into_grounded),
                    ScopedCut::Unmarked(_) | ScopedCut::Bottom(_) => None,
                };
                if let Some(grounded) = grounded_profile {
                    let rec = PhaseAnchorRecord::from_grounded_delta_64(
                        cold.anchor_ref_id,
                        ATTRACTOR_SLOT_SYSTEM_BASELINE,
                        cold.reinforce_count,
                        f32::from(routing.gain_q7) * (1.0 / 127.0),
                        grounded,
                    );
                    routing.set_consolidated(true);
                    crystallized = Some(rec);
                }
            }

            if current_epoch.saturating_sub(cold.last_active_epoch) >= 4 {
                let decay = (routing.pheromone_q7 >> 3).max(1);
                routing.pheromone_q7 = routing.pheromone_q7.saturating_sub(decay);
                if routing.pheromone_q7 < self.evaporation_floor_q7 {
                    routing.set_active(false);
                    self.active_node_mask &= !bit;
                }
            }
        }
        crystallized
    }

    /// Single-source-of-truth word-level P-Frame evaluator independent of `SystemSpineState`,
    /// `DistortionTension`, and `GatewayFrame`.
    ///
    /// - When `self.max_hops <= 1`, executes the **1-Hop Parallel Perspective Ensemble**: evaluates
    ///   all active baked nodes against `partition` in parallel, accumulates weighted per-head
    ///   Q7 deltas in `i32`, and normalizes by an integer reciprocal LUT (`0 fdiv`, `0` heap alloc)
    ///   before circular Q7 addition onto `base_words`.
    /// - When `self.max_hops >= 2`, executes the **Multi-Hop Sequential Chain**: selects the
    ///   strongest matching entry node and follows `next_hop_idx` up to
    ///   `self.max_hops.min(MAX_DISTORTION_CHAIN_HOPS)`.
    ///
    /// Returns `(out_words, delta_norm_q7, hops_traversed, entry_idx, matched_mask)`.
    #[inline]
    #[must_use]
    pub fn evaluate_pframe_words_branchless(
        &self,
        ego: &LiminalEgoState,
        partition: &LacunaPartition8,
        stigmergy: &StigmergicGranuleState,
        effective_ego_scale_q7: u8,
        base_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], u8, u8, u8, u8) {
        if !ego.is_active() || effective_ego_scale_q7 == 0 || self.active_node_mask == 0 {
            return (*base_words, 0, 0, 0, 0);
        }

        let region_bits =
            branchless_select_u8(partition.self_ref_mask != 0, SCOPE_REGION_SELF_REF, 0)
                | branchless_select_u8(partition.interior_mask != 0, SCOPE_REGION_INTERIOR, 0)
                | branchless_select_u8(partition.exterior_mask != 0, SCOPE_REGION_EXTERIOR, 0);
        let region_fallback = branchless_select_u8(
            region_bits != 0,
            region_bits,
            SCOPE_REGION_SELF_REF | SCOPE_REGION_INTERIOR,
        );
        let combined_support =
            partition.self_ref_mask | partition.interior_mask | partition.exterior_mask;
        let support_mask = branchless_select_u8(combined_support != 0, combined_support, 0xFF);

        let sens_boost_q7 = u32::from(partition.boundary_sensitivity_q7).max(48);
        let eff_scale = u32::from(effective_ego_scale_q7);
        let mut ph_sum = 0u32;
        for &ph in stigmergy
            .granule_pheromone_q7
            .iter()
            .take(NUM_STIGMERGIC_GRANULES)
        {
            ph_sum += u32::from(ph);
        }
        let mean_stig_ph = (ph_sum >> 3).clamp(24, 127);

        if self.max_hops <= 1 {
            let mut matched_mask = 0u8;
            let mut weights_q7 = [0u8; MAX_EGO_PFRAME_NODES];

            for (i, routing) in self
                .node_routing
                .iter()
                .enumerate()
                .take(MAX_EGO_PFRAME_NODES)
            {
                let active = ((self.active_node_mask >> i) & 1) != 0 && routing.is_baked_active();
                let reg_ok =
                    BitCut8::split(routing.scope_region_selector, region_fallback).inside != 0;
                let trig_ok = BitCut8::split(routing.lacuna_trigger_mask, support_mask).inside != 0;
                if active && reg_ok && trig_ok {
                    matched_mask |= 1u8 << i;
                    let hop_weight_q7 =
                        (((u32::from(routing.gain_q7) * eff_scale) >> 7) * sens_boost_q7) >> 7;
                    let ph_blend_q7 = (u32::from(routing.pheromone_q7) + mean_stig_ph) >> 1;
                    let dens_weight_q7 = ((hop_weight_q7 * ph_blend_q7) >> 7).clamp(8, 127) as u8;
                    weights_q7[i] = dens_weight_q7;
                }
            }

            if matched_mask == 0 {
                return (*base_words, 0, 0, 0, 0);
            }

            let (out_words, avg_delta, entry_idx) =
                branchless_q7_weighted_ensemble_contiguous_words_64(
                    &self.node_phases,
                    &weights_q7,
                    matched_mask,
                    base_words,
                );
            return (out_words, avg_delta, 1, entry_idx, matched_mask);
        }

        let mut best_entry_idx = 0usize;
        let mut best_entry_score = 0u32;
        for (i, routing) in self
            .node_routing
            .iter()
            .enumerate()
            .take(MAX_EGO_PFRAME_NODES)
        {
            let active = ((self.active_node_mask >> i) & 1) != 0 && routing.is_baked_active();
            let reg_ok = BitCut8::split(routing.scope_region_selector, region_fallback).inside != 0;
            let trig_overlap = BitCut8::split(routing.lacuna_trigger_mask, support_mask)
                .inside
                .count_ones();
            let score = if active && reg_ok && trig_overlap > 0 {
                u32::from(routing.pheromone_q7).saturating_mul(trig_overlap)
            } else {
                0
            };
            let take = score > best_entry_score;
            best_entry_score = branchless_select_u32(take, score, best_entry_score);
            best_entry_idx = if take { i } else { best_entry_idx };
        }
        if best_entry_score == 0 {
            return (*base_words, 0, 0, 0, 0);
        }

        let hop_cap = self.max_hops.min(MAX_DISTORTION_CHAIN_HOPS);
        let mut acc_words = *base_words;
        let mut accum_norm_sum = 0u32;
        let mut hops_traversed = 0u8;
        let mut visited_mask = 0u8;
        let mut cur_idx = best_entry_idx as u16;
        let mut hop = 0u8;

        while hop < MAX_DISTORTION_CHAIN_HOPS {
            let slot = usize::from(cur_idx);
            let slot_bit = 1u8 << (slot & 7);
            let in_bounds = (hop < hop_cap)
                && (slot < MAX_EGO_PFRAME_NODES)
                && ((visited_mask & slot_bit) == 0)
                && ((self.active_node_mask & slot_bit) != 0);
            if !in_bounds {
                break;
            }
            let routing = &self.node_routing[slot];
            if !routing.is_baked_active() {
                break;
            }
            visited_mask |= slot_bit;
            let hop_weight_q7 =
                (((u32::from(routing.gain_q7) * eff_scale) >> 7) * sens_boost_q7) >> 7;
            let dens_weight_q7 = ((hop_weight_q7 * mean_stig_ph) >> 7).clamp(8, 127) as u64;
            let bias_restore =
                (128u64.wrapping_sub(dens_weight_q7)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let node_words = load_q7_words_64(&self.node_phases[slot]);
            for (acc_w, &nw) in acc_words
                .iter_mut()
                .zip(node_words.iter())
                .take(NUM_STIGMERGIC_GRANULES)
            {
                let d_w = swar_q7_scale_le128_with_bias_x8(nw, dens_weight_q7, bias_restore);
                *acc_w = swar_q7_circular_add_x8(*acc_w, d_w);
            }
            accum_norm_sum = accum_norm_sum.saturating_add(dens_weight_q7 as u32);
            hops_traversed = hops_traversed.saturating_add(1);
            cur_idx = routing.next_hop_idx;
            hop = hop.saturating_add(1);
        }

        const INV_HOPS_Q7: [u32; 5] = [0, 128, 64, 43, 32];
        let avg_delta = ((accum_norm_sum * INV_HOPS_Q7[usize::from(hops_traversed.min(4))]) >> 7)
            .min(127) as u8;
        (
            acc_words,
            avg_delta,
            hops_traversed,
            best_entry_idx as u8,
            visited_mask,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drivers::ego::intent::LiminalEgoPreset;

    #[test]
    fn test_one_hop_parallel_perspective_ensemble_and_reinforcement() {
        let ego = LiminalEgoState::from_preset(LiminalEgoPreset::High);
        let axes = EgoIntentAxes::new()
            .with_autonomy_q7(72)
            .with_transgression_q7(48);
        let mut net_1 = EgoDifferentialNet::from_ego_state_and_axes(&ego, &axes, 1);
        let net_3 = EgoDifferentialNet::from_ego_state_and_axes(&ego, &axes, 3);
        let partition = LacunaPartition8 {
            self_ref_mask: 0x0F,
            interior_mask: 0x30,
            exterior_mask: 0x40,
            unobserved_mask: 0x80,
            boundary_sensitivity_q7: 96,
            ..LacunaPartition8::default()
        };
        let stig = StigmergicGranuleState::new();
        let base_words = [0u64; NUM_STIGMERGIC_GRANULES];

        let (w1, norm1, hops1, entry1, mask1) =
            net_1.evaluate_pframe_words_branchless(&ego, &partition, &stig, 96, &base_words);
        let (w3, _norm3, hops3, _entry3, mask3) =
            net_3.evaluate_pframe_words_branchless(&ego, &partition, &stig, 96, &base_words);

        assert_eq!(hops1, 1);
        assert!(norm1 > 0);
        assert!(
            mask1.count_ones() >= 2,
            "1-hop ensemble should fire multiple perspectives, got mask={mask1:#010b}"
        );
        assert_eq!(entry1, mask1.trailing_zeros() as u8);
        assert_eq!(hops3, 3);
        assert_eq!(mask3, 0b0000_0111);
        assert_ne!(
            w1, w3,
            "1-hop normalized ensemble should differ from 3-hop chain"
        );

        let ph_before: [u8; 3] = [
            net_1.node_routing[0].pheromone_q7,
            net_1.node_routing[1].pheromone_q7,
            net_1.node_routing[2].pheromone_q7,
        ];
        net_1.epoch = 5;
        net_1.record_matched_activations(mask1);
        for (idx, prev_ph) in ph_before.iter().enumerate() {
            if ((mask1 >> idx) & 1) != 0 {
                assert_eq!(net_1.node_cold[idx].reinforce_count, 1);
                assert_eq!(net_1.node_cold[idx].last_active_epoch, 5);
                assert_eq!(
                    net_1.node_routing[idx].pheromone_q7,
                    prev_ph.saturating_add(1).min(127)
                );
            }
        }
        // Verify export_node / set_node roundtrip preserves all DistortionDelta metadata & phase bytes.
        let exported = net_1.export_node(0);
        assert_eq!(exported.delta_phase_q7, net_1.node_phases[0]);
        assert!(exported.is_baked_active());
        let mut net_rt = EgoDifferentialNet::new_empty(1);
        net_rt.set_node(0, exported);
        assert_eq!(net_rt.export_node(0), exported);
        assert!(net_rt.is_node_baked_active(0));
    }
}
