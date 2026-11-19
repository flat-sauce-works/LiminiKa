// name: src/core/src/drivers/ego/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ego driver axis (`Ego`): anisotropic P-Frame differential residual network and intent fields.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

pub mod differential;
pub mod intent;

pub use self::differential::*;
pub use self::intent::*;
