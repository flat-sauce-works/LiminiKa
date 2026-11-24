// name: src/core/src/runtime/topology.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Static 64-byte Core Topology Matrix ([`LiminalTopologyMatrix`], [`CORE_TOPOLOGY`]) with
//! compile-time IEEE 802.3 CRC32 verification and branchless `BitCut8` channel admission control.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::mem::{align_of, offset_of, size_of};

use crate::foundation::{compute_crc32_ieee, Pod};
use crate::liminal::{BitCut8, LMNK_ABI_VERSION_HEX};

/// Binary magic constant for [`LiminalTopologyMatrix`] (`0x504F_544C` = ASCII `"LTOP"`).
pub const LMNK_TOPOLOGY_MAGIC: u32 = 0x504F_544C;

/// Canonical number of unidirectional architectural layers in `liminika-core` (`5`).
pub const LMNK_TOPOLOGY_LAYER_COUNT: u8 = 5;

/// Canonical number of physical horizon / threshold channels (`8`).
pub const LMNK_TOPOLOGY_CHANNEL_COUNT: u8 = 8;

/// DAG invariant bit `0`: `liminal` (Lv.0) and `foundation` (Lv.1) form parallel root layers.
pub const LMNK_DAG_FLAG_Y_SHAPED_PARALLEL_ROOTS: u16 = 1 << 0;

/// DAG invariant bit `1`: `drivers/spine` and `drivers/ego` are mutually isolated.
pub const LMNK_DAG_FLAG_SPINE_EGO_ISOLATED: u16 = 1 << 1;

/// DAG invariant bit `2`: `liminal/axiom/cut.rs` and `liminal/axiom/lacuna.rs` are mutually orthogonal.
pub const LMNK_DAG_FLAG_AXIOM_CUT_LACUNA_ORTHOGONAL: u16 = 1 << 2;

/// DAG invariant bit `3`: `liminika-core` enforces `#![no_std]` zero-allocation boundaries.
pub const LMNK_DAG_FLAG_NO_STD_CORE: u16 = 1 << 3;

/// Canonical composite DAG invariant flags for `LiminiKa` `v0.3.0`.
pub const LMNK_TOPOLOGY_DAG_FLAGS_CANONICAL: u16 = LMNK_DAG_FLAG_Y_SHAPED_PARALLEL_ROOTS
    | LMNK_DAG_FLAG_SPINE_EGO_ISOLATED
    | LMNK_DAG_FLAG_AXIOM_CUT_LACUNA_ORTHOGONAL
    | LMNK_DAG_FLAG_NO_STD_CORE;

/// Canonical 5-layer unidirectional DAG upstream bitmasks (`dag_edge_matrix[layer]`).
///
/// - `0` (`liminal`): `0b0000_0000` (depends only on `core`)
/// - `1` (`foundation`): `0b0000_0001` (`liminal` constants/select only)
/// - `2` (`stigmergy`): `0b0000_0011` (`liminal`, `foundation`)
/// - `3` (`drivers`): `0b0000_0111` (`liminal`, `foundation`, `stigmergy`)
/// - `4` (`runtime`): `0b0000_1111` (`liminal`, `foundation`, `stigmergy`, `drivers`)
pub const LMNK_TOPOLOGY_DAG_EDGE_MATRIX: [u8; 8] = [
    0b0000_0000,
    0b0000_0001,
    0b0000_0011,
    0b0000_0111,
    0b0000_1111,
    0,
    0,
    0,
];

/// Canonical host-pinned channel bitmask (`SystemConstraint` channel `0` | `ContradictionGuard` channel `3`).
pub const LMNK_TOPOLOGY_PINNED_CHANNEL_MASK: u8 = (1 << 0) | (1 << 3);

/// Canonical isolated exterior-only sandbox channel bitmask (`channel 7`).
pub const LMNK_TOPOLOGY_SANDBOX_CHANNEL_MASK: u8 = 1 << 7;

/// Static 64-byte (`#[repr(C, align(64))]`) architectural topology and admission descriptor
/// locking the 5-layer DAG structure, channel admission masks, and baseline Q0.7 apertures.
///
/// Canonical Little-Endian layout with zero implicit padding (`topology_crc32` covers bytes `0..60`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiminalTopologyMatrix {
    /// Structure identifier magic constant ([`LMNK_TOPOLOGY_MAGIC`] = `"LTOP"`, `offset 0..4`).
    pub magic: u32,
    /// ABI version matching [`LMNK_ABI_VERSION_HEX`] (`offset 4..8`).
    pub version: u32,
    /// Architectural layer count (`5`, `offset 8`).
    pub layer_count: u8,
    /// Horizon / threshold channel count (`8`, `offset 9`).
    pub channel_count: u8,
    /// Structural DAG invariant bitmask (`LMNK_DAG_FLAG_*`, `offset 10..12`).
    pub dag_flags: u16,
    /// Allowed upstream layer bitmasks per layer `0..5` (`offset 12..20`).
    pub dag_edge_matrix: [u8; 8],
    /// Per-channel `BitCut8` admission mask governing slot/role activation (`offset 20..28`).
    pub channel_admission_mask: [u8; 8],
    /// Immutable `SystemSpineState` pinned channel mask (`offset 28`).
    pub pinned_channel_mask: u8,
    /// Isolated exterior-only sandbox channel mask (`offset 29`).
    pub sandbox_channel_mask: u8,
    /// Explicit zero alignment padding (`offset 30..32`).
    pub reserved_mid: [u8; 2],
    /// Baseline Q0.7 aperture weights for channels `0..8` (`offset 32..40`).
    pub channel_aperture_q7: [u8; 8],
    /// Explicit zero tail padding (`offset 40..60`).
    pub reserved_tail: [u8; 20],
    /// Compile-time IEEE 802.3 CRC32 over canonical Little-Endian bytes `0..60` (`offset 60..64`).
    pub topology_crc32: u32,
}

// SAFETY: `LiminalTopologyMatrix` is `#[repr(C, align(64))]` of exact size 64 bytes with zero implicit padding.
unsafe impl Pod for LiminalTopologyMatrix {}

impl Default for LiminalTopologyMatrix {
    #[inline(always)]
    fn default() -> Self {
        Self::canonical_v0_3_0()
    }
}

impl LiminalTopologyMatrix {
    /// Serializes the protected prefix (`bytes 0..60`, excluding `topology_crc32`) into canonical
    /// Little-Endian byte order.
    #[inline(always)]
    #[must_use]
    pub const fn to_prefix_le_bytes(&self) -> [u8; 60] {
        let m = self.magic.to_le_bytes();
        let v = self.version.to_le_bytes();
        let f = self.dag_flags.to_le_bytes();
        [
            m[0],
            m[1],
            m[2],
            m[3],
            v[0],
            v[1],
            v[2],
            v[3],
            self.layer_count,
            self.channel_count,
            f[0],
            f[1],
            self.dag_edge_matrix[0],
            self.dag_edge_matrix[1],
            self.dag_edge_matrix[2],
            self.dag_edge_matrix[3],
            self.dag_edge_matrix[4],
            self.dag_edge_matrix[5],
            self.dag_edge_matrix[6],
            self.dag_edge_matrix[7],
            self.channel_admission_mask[0],
            self.channel_admission_mask[1],
            self.channel_admission_mask[2],
            self.channel_admission_mask[3],
            self.channel_admission_mask[4],
            self.channel_admission_mask[5],
            self.channel_admission_mask[6],
            self.channel_admission_mask[7],
            self.pinned_channel_mask,
            self.sandbox_channel_mask,
            self.reserved_mid[0],
            self.reserved_mid[1],
            self.channel_aperture_q7[0],
            self.channel_aperture_q7[1],
            self.channel_aperture_q7[2],
            self.channel_aperture_q7[3],
            self.channel_aperture_q7[4],
            self.channel_aperture_q7[5],
            self.channel_aperture_q7[6],
            self.channel_aperture_q7[7],
            self.reserved_tail[0],
            self.reserved_tail[1],
            self.reserved_tail[2],
            self.reserved_tail[3],
            self.reserved_tail[4],
            self.reserved_tail[5],
            self.reserved_tail[6],
            self.reserved_tail[7],
            self.reserved_tail[8],
            self.reserved_tail[9],
            self.reserved_tail[10],
            self.reserved_tail[11],
            self.reserved_tail[12],
            self.reserved_tail[13],
            self.reserved_tail[14],
            self.reserved_tail[15],
            self.reserved_tail[16],
            self.reserved_tail[17],
            self.reserved_tail[18],
            self.reserved_tail[19],
        ]
    }

    /// Computes the IEEE 802.3 CRC32 checksum over canonical Little-Endian bytes `0..60`.
    #[inline(always)]
    #[must_use]
    pub const fn compute_crc32(&self) -> u32 {
        compute_crc32_ieee(&self.to_prefix_le_bytes())
    }

    /// Returns `true` when `magic`, `version`, and `topology_crc32` match this matrix.
    #[inline(always)]
    #[must_use]
    pub const fn verify_crc32(&self) -> bool {
        (self.magic == LMNK_TOPOLOGY_MAGIC)
            & (self.version == LMNK_ABI_VERSION_HEX)
            & (self.topology_crc32 == self.compute_crc32())
    }

    /// Constructs the canonical `LiminiKa` `v0.3.0` topology matrix with stamped compile-time CRC32.
    #[inline(always)]
    #[must_use]
    pub const fn canonical_v0_3_0() -> Self {
        let mut matrix = Self {
            magic: LMNK_TOPOLOGY_MAGIC,
            version: LMNK_ABI_VERSION_HEX,
            layer_count: LMNK_TOPOLOGY_LAYER_COUNT,
            channel_count: LMNK_TOPOLOGY_CHANNEL_COUNT,
            dag_flags: LMNK_TOPOLOGY_DAG_FLAGS_CANONICAL,
            dag_edge_matrix: LMNK_TOPOLOGY_DAG_EDGE_MATRIX,
            // Channels 0 and 3 are pinned to their own explicit bits; channel 7 excludes pinned bits.
            channel_admission_mask: [0x01, 0xFF, 0xFF, 0x08, 0xFF, 0xFF, 0xFF, 0xF6],
            pinned_channel_mask: LMNK_TOPOLOGY_PINNED_CHANNEL_MASK,
            sandbox_channel_mask: LMNK_TOPOLOGY_SANDBOX_CHANNEL_MASK,
            reserved_mid: [0; 2],
            channel_aperture_q7: [96; 8],
            reserved_tail: [0; 20],
            topology_crc32: 0,
        };
        matrix.topology_crc32 = matrix.compute_crc32();
        matrix
    }

    /// Evaluates branchless $\mathcal{O}(1)$ `BitCut8` channel admission against this matrix.
    ///
    /// Enforces:
    /// 1. Bounds validity (`channel_idx < channel_count <= 8`).
    /// 2. Non-empty cut overlap with `channel_admission_mask[channel_idx]`.
    /// 3. Pinned `Spine` invariance (channels in `pinned_channel_mask` require their own bit).
    /// 4. Sandbox isolation (channels in `sandbox_channel_mask` require non-pinned bits).
    #[inline(always)]
    #[must_use]
    pub const fn admits_channel_branchless(
        &self,
        channel_idx: usize,
        requested_cut: BitCut8,
    ) -> bool {
        let max_ch = if (self.channel_count as usize) <= 8 {
            self.channel_count as usize
        } else {
            8
        };
        let in_bounds = channel_idx < max_ch;
        let safe_idx = if in_bounds { channel_idx } else { 0 };
        let ch_bit = 1u8 << (safe_idx as u32);
        let raw = requested_cut.raw();

        let has_admitted = (raw & self.channel_admission_mask[safe_idx]) != 0;
        let is_pinned = (self.pinned_channel_mask & ch_bit) != 0;
        let pinned_ok = !is_pinned | ((raw & ch_bit) != 0);
        let is_sandbox = (self.sandbox_channel_mask & ch_bit) != 0;
        let sandbox_ok = !is_sandbox | ((raw & !self.pinned_channel_mask) != 0);

        in_bounds & has_admitted & pinned_ok & sandbox_ok
    }
}

/// Canonical compile-time `LiminiKa` `v0.3.0` topology matrix.
pub const CORE_TOPOLOGY: LiminalTopologyMatrix = LiminalTopologyMatrix::canonical_v0_3_0();

const _: () = {
    assert!(size_of::<LiminalTopologyMatrix>() == 64);
    assert!(align_of::<LiminalTopologyMatrix>() == 64);
    assert!(offset_of!(LiminalTopologyMatrix, magic) == 0);
    assert!(offset_of!(LiminalTopologyMatrix, version) == 4);
    assert!(offset_of!(LiminalTopologyMatrix, layer_count) == 8);
    assert!(offset_of!(LiminalTopologyMatrix, channel_count) == 9);
    assert!(offset_of!(LiminalTopologyMatrix, dag_flags) == 10);
    assert!(offset_of!(LiminalTopologyMatrix, dag_edge_matrix) == 12);
    assert!(offset_of!(LiminalTopologyMatrix, channel_admission_mask) == 20);
    assert!(offset_of!(LiminalTopologyMatrix, pinned_channel_mask) == 28);
    assert!(offset_of!(LiminalTopologyMatrix, sandbox_channel_mask) == 29);
    assert!(offset_of!(LiminalTopologyMatrix, reserved_mid) == 30);
    assert!(offset_of!(LiminalTopologyMatrix, channel_aperture_q7) == 32);
    assert!(offset_of!(LiminalTopologyMatrix, reserved_tail) == 40);
    assert!(offset_of!(LiminalTopologyMatrix, topology_crc32) == 60);
    assert!(CORE_TOPOLOGY.verify_crc32());
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_topology_matrix_crc_and_admission() {
        assert!(CORE_TOPOLOGY.verify_crc32());
        assert_eq!(CORE_TOPOLOGY.magic, LMNK_TOPOLOGY_MAGIC);
        assert_eq!(CORE_TOPOLOGY.layer_count, 5);
        assert_eq!(CORE_TOPOLOGY.channel_count, 8);
        assert_eq!(
            CORE_TOPOLOGY.topology_crc32,
            compute_crc32_ieee(&CORE_TOPOLOGY.to_prefix_le_bytes())
        );

        let mut tampered = CORE_TOPOLOGY;
        tampered.dag_edge_matrix[1] = 0b0001_1111;
        assert!(!tampered.verify_crc32());

        // Authorized slots
        assert!(CORE_TOPOLOGY.admits_channel_branchless(0, BitCut8::split(0x01, 0xFF)));
        assert!(CORE_TOPOLOGY.admits_channel_branchless(1, BitCut8::split(0x02, 0xFF)));
        assert!(CORE_TOPOLOGY.admits_channel_branchless(3, BitCut8::split(0x08, 0xFF)));
        assert!(CORE_TOPOLOGY.admits_channel_branchless(7, BitCut8::split(0x80, 0xFF)));

        // Unauthorized slots: out of bounds, empty cut, wrong bit for pinned, pinned-only sandbox
        assert!(!CORE_TOPOLOGY.admits_channel_branchless(8, BitCut8::split(0xFF, 0xFF)));
        assert!(!CORE_TOPOLOGY.admits_channel_branchless(1, BitCut8::split(0x00, 0xFF)));
        assert!(!CORE_TOPOLOGY.admits_channel_branchless(0, BitCut8::split(0x02, 0xFF)));
        assert!(!CORE_TOPOLOGY.admits_channel_branchless(3, BitCut8::split(0x04, 0xFF)));
        assert!(!CORE_TOPOLOGY.admits_channel_branchless(7, BitCut8::split(0x09, 0xFF)));
    }
}
