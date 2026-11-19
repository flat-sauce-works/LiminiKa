// name: src/core/src/liminal/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 0 (`Liminal`): Foundational boundary axioms, zero-sized scope markers, canonical
//! 3-state `Lacuna` classification, threshold sensitivity, and C-ABI invariant constants.
//!
//! This layer depends strictly on Rust `core` and defines the pure topological vocabulary
//! shared by all upper layers (`foundation`, `stigmergy`, `drivers`, `runtime`).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod constants;
pub mod lacuna;
pub mod scope;
pub mod select;
pub mod threshold;

pub(crate) use self::constants::MAX_REGISTER_CACHED_CENTROIDS;
pub use self::constants::*;
pub use self::lacuna::*;
pub use self::scope::*;
pub use self::select::*;
pub use self::threshold::*;

const _: () = {
    assert!(
        GCSO_ABI_VERSION_HEX
            == ((GCSO_ABI_VERSION_MAJOR << 16)
                | (GCSO_ABI_VERSION_MINOR << 8)
                | GCSO_ABI_VERSION_PATCH)
    );
    assert!(GCSO_ABI_VERSION_HEX >= 0x0000_0103);
    assert!(MAX_ATTRACTOR_DOMAINS == 8);
    assert!(MAX_ANCHORS_PER_FIELD == 64);
    assert!(MAX_ATTENTION_HEADS == 64);
    assert!(NUM_STIGMERGIC_GRANULES * HEADS_PER_STIGMERGIC_GRANULE == MAX_ATTENTION_HEADS);
    assert!((ATTRACTOR_SLOT_ID_MASK as usize) == (MAX_ATTRACTOR_DOMAINS - 1));
    assert!(pack_action_slot_tag(7, 0x1F) == 0xFF);
    assert!(unpack_attractor_slot_from_tag(pack_action_slot_tag(5, 0x13)) == 5);
    assert!(unpack_action_mode_from_tag(pack_action_slot_tag(5, 0x13)) == 0x13);
    assert!(slot_coupling_bit_index(7, 7) == 63);
    assert!(coupled_slot_row_mask(DEFAULT_SLOT_COUPLING_MATRIX_8X8, 3) == 0xFF);
    assert!(coupled_slot_count(DEFAULT_SLOT_COUPLING_MATRIX_8X8, 3) == 8);
};
