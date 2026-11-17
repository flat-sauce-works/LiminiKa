// name: src/core/src/traits/primitives.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.0 Foundation 1: Base C-ABI v0.1.1 (`0x00000101`) constants, Plain Old Data (`Pod`)
//! byte-level safety contracts, `.gcso` descriptor/snapshot validators, branchless IEEE 802.3
//! CRC32 integrity primitives, TaggedPtr 5-bit ActionMode & 3-bit AttractorSlot codecs,
//! $8 \times 8$ Attractor Slot coupling matrix helpers, and lifetime-bound FFI handle guards
//! for the Geometric Cellular Sheaf Orchestrator (GCSO).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::cast_possible_truncation)]

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]
#![allow(clippy::manual_unwrap_or)]
#![allow(clippy::too_many_arguments)]

use core::f32::consts::{FRAC_PI_2, LN_2, PI};
use core::fmt;
use core::marker::PhantomData;
use core::mem::{align_of, size_of, size_of_val};
use core::ops::{Deref, DerefMut};
use core::ptr::{self, NonNull};
use core::slice;

use crate::abi::{
    gcso_config_t, gcso_ledger_slot_t, gcso_descriptor_header_t, gcso_edbc_state_t,
    gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_kv_keyframe_header_t, gcso_subhead_config_t,
    gcso_q7_t, gcso_snapshot_header_t, gcso_rank1_residual_descriptor_t, gcso_status_t, gcso_tagged_ptr_t,
    gcso_mapped_storage_descriptor_t, GcsoStatus, GCSO_ABI_VERSION_HEX, GCSO_ERROR_BUFFER_TOO_SMALL,
    GCSO_ERROR_CONTAINER_CORRUPTED, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER,
    GCSO_ERROR_NULL_POINTER, GCSO_ERROR_VERSION_MISMATCH,
};

// Re-export canonical strongly-typed opaque handles from `crate::abi`.
pub use crate::abi::{
    GcsoActionHubHandle, GcsoActionHubOpaque, GcsoAttractorFieldHandle, GcsoAttractorFieldOpaque,
    GcsoContainerHandle, GcsoContainerOpaque, GcsoContextHandle, GcsoContextOpaque,
    GcsoLedgerSlotHandle, GcsoLedgerSlotOpaque, GcsoDpsrKernelHandle, GcsoDpsrKernelOpaque,
    GcsoEdbcControllerHandle, GcsoEdbcControllerOpaque, GcsoSubHeadRouterHandle, GcsoSubHeadRouterOpaque,
    GcsoRank1ResidualHandle, GcsoRank1ResidualOpaque,
};

/// Type alias for GCSO core result responses mapping to [`GcsoStatus`].
pub type GcsoResult<T> = Result<T, GcsoStatus>;


pub mod constants_pod;
pub mod packets_ffi;
pub mod lut_crc32;
pub mod scalar_math;
pub mod swar_ops;

pub use self::{
    constants_pod::*, lut_crc32::*, packets_ffi::*, scalar_math::*, swar_ops::*,
};

// Compile-time structural and bitfield invariants for Layer 0 primitives.
const _: () = {
    assert!(GCSO_ABI_VERSION_HEX == 0x0000_0101);
    assert!(MAX_ATTRACTOR_DOMAINS == 8);
    assert!(MAX_ANCHORS_PER_FIELD == 64);
    assert!(MAX_ATTENTION_HEADS == 64);
    assert!((ATTRACTOR_SLOT_ID_MASK as usize) == (MAX_ATTRACTOR_DOMAINS - 1));
    assert!(pack_action_slot_tag(7, 0x1F) == 0xFF);
    assert!(unpack_attractor_slot_from_tag(pack_action_slot_tag(5, 0x13)) == 5);
    assert!(unpack_action_mode_from_tag(pack_action_slot_tag(5, 0x13)) == 0x13);
    assert!(slot_coupling_bit_index(7, 7) == 63);
    assert!(coupled_slot_row_mask(DEFAULT_SLOT_COUPLING_MATRIX_8X8, 3) == 0xFF);
    assert!(coupled_slot_count(DEFAULT_SLOT_COUPLING_MATRIX_8X8, 3) == 8);
    assert!(compute_crc32_ieee(b"123456789") == 0xCBF4_3926);
    assert!(size_of::<KvKeyframePacket>() == 128);
    assert!(align_of::<KvKeyframePacket>() == 64);
    assert!(size_of::<KvMotionDeltaPacket>() == 64);
    assert!(align_of::<KvMotionDeltaPacket>() == 64);
};

#[cfg(test)]
mod tests;
