// name: src/adapter/src/tests/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adapter unit test suite split by subsystem.

use super::*;

const fn sample_gqa_partial_rope() -> RopeGeometry {
    RopeGeometry::new(RopeLayout::HalfSplit, 128, 96, 24, 8, 32, 0xFFFF_FFFF)
}

mod adaptive_and_guards;
mod recall_and_strata;
mod steering;
mod timeline_and_codec;
mod turn_and_orchestration;
