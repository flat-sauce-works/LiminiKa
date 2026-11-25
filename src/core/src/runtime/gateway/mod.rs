// name: src/core/src/runtime/gateway/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 4 Gateway (`signals`, `frame`, `orchestrator`, and distributed `bridges`
//! implementing Gateway/Signal integration over Lv.1-Lv.3 types).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

/// Default destructive phase coherence threshold (`0.25`) detecting Persona-vs-Fact collision ($\mathcal{C}_{\mathrm{FP}} < -0.25$).
pub const DEFAULT_PERSONA_FACT_COLLISION_THRESH: f32 = 0.25;

pub mod bridges;
pub mod frame;
pub mod orchestrator;
pub mod signals;

pub(crate) use self::frame::tangent_cosine_active_heads_64;
pub use self::{bridges::*, frame::*, orchestrator::*, signals::*};
