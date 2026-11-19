// name: src/core/src/drivers/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 3 (`Drivers`): Dual execution axes (`spine` vs. `ego`) isolated behind a single shared
//! interface (`distortion`). `spine` and `ego` never import one another.
//!
//! Depends strictly on `core`, [`crate::liminal`], [`crate::foundation`], and [`crate::stigmergy`].

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

pub mod distortion;
pub mod ego;
pub mod spine;

pub use self::distortion::*;
pub use self::ego::*;
pub use self::spine::*;
