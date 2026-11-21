// name: src/core/src/runtime/horizon/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 4 Latent Horizon (`epistemic`, `kv_cache`, and dynamic slide-in `slots`).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

pub mod epistemic;
pub mod kv_cache;
pub mod slots;

pub use self::{epistemic::*, kv_cache::*, slots::*};
