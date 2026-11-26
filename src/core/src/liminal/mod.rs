// name: src/core/src/liminal/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 0 (`Liminal`): Foundational boundary axioms, zero-sized scope markers, orthogonal
//! 4-state `Lacuna` classification, threshold sensitivity, and C-ABI invariant constants.
//!
//! This layer depends strictly on Rust `core` and defines the pure topological vocabulary
//! shared by all upper layers (`foundation`, `stigmergy`, `drivers`, `runtime`).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod axiom;
pub mod bitcut;
pub mod constants;
pub mod partition;
pub mod select;
pub mod threshold;

pub use self::axiom::*;
pub use self::bitcut::{BitCut, BitCut8, BitCut32, BitCut64};
pub use self::constants::*;
pub use self::partition::*;
pub use self::select::*;
pub use self::threshold::*;

const _: () = {
    assert!(
        LMNK_ABI_VERSION_HEX
            == ((LMNK_ABI_VERSION_MAJOR << 16)
                | (LMNK_ABI_VERSION_MINOR << 8)
                | LMNK_ABI_VERSION_PATCH)
    );
    assert!(LMNK_ABI_VERSION_HEX >= 0x0000_0300);
    assert!(MAX_ATTRACTOR_DOMAINS == 8);
    assert!(MAX_ATTENTION_HEADS == 64);
    assert!(NUM_STIGMERGIC_GRANULES * HEADS_PER_STIGMERGIC_GRANULE == MAX_ATTENTION_HEADS);
};
