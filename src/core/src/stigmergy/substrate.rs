// name: src/core/src/stigmergy/substrate.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified 64-byte Cognitive Substrate (`LiminalThresholdField` + `StigmergicGranuleField`).

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]

use crate::foundation::Pod;
use crate::liminal::LiminalThresholdField;
use crate::stigmergy::granule::StigmergicGranuleField;

/// Unified single-cacheline (`64 B`, `#[repr(C, align(64))]`) cognitive substrate pairing the two
/// foundational geometric primitives: [`LiminalThresholdField`] (`0..32`) and
/// [`StigmergicGranuleField`] (`32..64`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CognitiveSubstrate {
    /// Boundary apertures, cognitive window floors, and domain gates (`offset 0..32`).
    pub liminal: LiminalThresholdField,
    /// 8-granule stigmergic step gains, noise tolerances, and volatile pheromones (`offset 32..64`).
    pub stigmergy: StigmergicGranuleField,
}

impl CognitiveSubstrate {
    /// Constructs a neutral 64-byte [`CognitiveSubstrate`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            liminal: LiminalThresholdField::new(),
            stigmergy: StigmergicGranuleField::new(),
        }
    }
}

// SAFETY: `CognitiveSubstrate` is `#[repr(C, align(64))]`, exact size 64 bytes with zero padding.
unsafe impl Pod for CognitiveSubstrate {}
