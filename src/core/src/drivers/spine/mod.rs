// name: src/core/src/drivers/spine/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Spine driver axis (`System`): deterministic I-Frame mechanical spine,
//! physical `pinned` isolation, anonymous compute pool dispatch, and entropy decoding gate.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

pub mod compute_pool;
pub mod entropy;
pub mod state;

pub use self::compute_pool::*;
pub use self::entropy::*;
pub use self::state::*;
