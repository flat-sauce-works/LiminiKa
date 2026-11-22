// name: src/core/src/runtime/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 4 (`Runtime`): Stateful orchestration context ([`RuntimeContext`]), configuration
//! ([`Config`]), telemetry ([`CoreTelemetryBank`]), Composite Manifold Gateway (`gateway`),
//! Latent Horizon (`horizon`), `.lmnk` zero-copy `storage`, and C11/C++20 `v0.2.0` FFI boundary (`abi`).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

pub mod abi;
pub mod boundary_delta;
pub mod context;
pub mod gateway;
pub mod horizon;
pub mod storage;
pub mod telemetry;
pub mod topology;

pub use self::boundary_delta::{LiminalBoundaryDelta, CHATTER_FIRE_THRESHOLD_U8};
pub use self::context::*;
pub use self::gateway::*;
pub use self::horizon::*;
pub use self::storage::*;
pub use self::telemetry::*;
pub use self::topology::*;

#[cfg(test)]
mod tests;
