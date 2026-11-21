// name: src/core/src/runtime/gateway/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 4 Gateway (`signals`, `frame`, `orchestrator`, and distributed `bridges`
//! implementing Gateway/Signal integration over Lv.1-Lv.3 types).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

pub use crate::stigmergy::{
    CAUSAL_AXIS_CONTRADICTION_GUARD, CAUSAL_AXIS_OBJECTIVE_FACT, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_AXIS_SYSTEM_CONSTRAINT, CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK,
    CAUSAL_CAUSE_PERSONA_FACT_COLLISION, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_FACT_RETRIEVAL_MISS, ROOT_CAUSE_NOMINAL,
    ROOT_CAUSE_PERSONA_FACT_COLLISION, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};

/// Default destructive phase coherence threshold (`0.25`) detecting Persona-vs-Fact collision ($\mathcal{C}_{\mathrm{FP}} < -0.25$).
pub const DEFAULT_PERSONA_FACT_COLLISION_THRESH: f32 = 0.25;

pub mod bridges;
pub mod frame;
pub mod orchestrator;
pub mod signals;

pub(crate) use self::frame::tangent_cosine_active_heads_64;
pub use self::{bridges::*, frame::*, orchestrator::*, signals::*};
