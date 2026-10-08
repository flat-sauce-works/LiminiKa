// name: src/core/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core execution engine library for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! This crate serves as the primary low-level engine for LiminiKa, providing:
//! - **Ground Truth Core Abstractions**: Central trait contracts (`traits.rs`).
//! - **Hot Path Execution Kernels**: Dynamic Phase-Shifted RoPE (`dpsr.rs`).
//! - **Stigmergic Memory Systems**: Sidecar Pointer Table (SPT) and Swarm Cells (`swarm.rs`).
//! - **Dynamic Entropy Controllers**: Entropy-Driven Decoding Branch Controller (`edbc.rs`).
//! - **Adapter & Mathematical Operations**: Sparse Residual Adapter Layer (`srl.rs`) and Math utilities (`math.rs`).
//! - **Persistence & Containers**: `.gcso` Unified Container Storage (`storage.rs`).
//! - **C-ABI / FFI Interoperability**: FFI boundary safety barriers (`abi.rs`).

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

// Core architecture module declarations
pub mod abi;
pub mod dpsr;
pub mod edbc;
pub mod math;
pub mod srl;
pub mod storage;
pub mod swarm;
pub mod traits;

// ===================================================================
// Primary Re-exports for Ergonomics & Ground Truth Alignment
// ===================================================================

// Re-export Ground Truth contracts, memory alignment wrappers, and core traits
pub use traits::{
    AlignedSlice, AlignedSlice128, AlignedSlice16, AlignedSlice32, AlignedSlice64, AlignedSliceMut,
    AlignedSliceMut128, AlignedSliceMut16, AlignedSliceMut32, AlignedSliceMut64, AttractorField,
    DaesScratchpad, EntropyEvaluator, GcsoResult, GcsoRuntimeContext, HotPathExecutionEngine,
    L2pSvdProjector, PersonaPatcher, PhaseSteering, PointerActionHub, PprcCache, PspmRouter,
    SrlAdapter, SwarmCellChunk, SystemCapabilityQuery, ZimmsStorage,
};

// Re-export C-ABI status codes, primitive aliases, and layout structs
pub use abi::{
    gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_daes_slot_t,
    gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t,
    gcso_pspm_config_t, gcso_q7_t, gcso_snapshot_header_t, gcso_srl_descriptor_t,
    gcso_zimms_descriptor_t, GcsoStatus, GCSO_SUCCESS,
};

// Re-export primary engine structs for direct module usage
pub use dpsr::DpsrEngine;

/// Static package version string constant obtained from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the static package version string defined in `Cargo.toml`.
#[must_use]
#[inline]
pub fn version() -> &'static str {
    VERSION
}