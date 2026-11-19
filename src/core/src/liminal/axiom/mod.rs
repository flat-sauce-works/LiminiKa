// name: src/core/src/liminal/axiom/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Isolated Level 0 boundary axiom kernel (`cut` and `lacuna`).
//!
//! Keeping `Scoped::from_raw` scoped to `pub(super)` inside `liminal::axiom` prevents even
//! sibling helper modules in `liminal` from forging boundary scopes directly.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod cut;
pub mod lacuna;

pub use self::cut::*;
pub use self::lacuna::*;
