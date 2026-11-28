// name: src/adapter/src/pooling/tests/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pooling and calibration unit test suite.

use super::*;
use crate::{LacunaBoundaryState, RopeGeometry, RopeLayout, TextAttractorEntry};

const fn sample_gqa_partial_rope() -> RopeGeometry {
    RopeGeometry::new(RopeLayout::HalfSplit, 128, 96, 24, 8, 32, 0xFFFF_FFFF)
}

mod calibration_and_alias;
mod corpus_and_streaming;
