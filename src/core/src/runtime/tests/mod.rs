// name: src/core/src/runtime/tests/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Runtime unit test suite split by subsystem.

use core::f32::consts::PI;
use core::mem::size_of;

use super::*;
use crate::drivers::{
    DEFAULT_HYSTERESIS_BAND, DEFAULT_TAU_FLUX, DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME,
    ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION,
};
use crate::foundation::{RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget};
use crate::prelude::*;
use crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7;
use crate::stigmergy::with_slot_pair_coupled_symmetric;

mod drivers_and_envelope;
mod gateway_and_horizon;
mod lifecycle_and_abi;
