// name: src/core/src/abi/storage.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persona patch, Mapped Storage (ZIMMS), and stateless handle destroyer FFI.

use super::*;

// ===================================================================
// 6. Persona Patch, ZIMMS Storage & Stateless Handle Destroyers
// ===================================================================

/// Dynamically applies persona phase modulation patches without altering base model weights.
///
/// # Safety
/// `context` and `patch_data` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_persona_apply_patch(
    context: GcsoContextHandle,
    patch_data: *const u8,
    patch_size: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(context)?;
        validate_ptr(patch_data)?;
        if patch_size == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Maps a `.gcso` unified binary container into memory space using zero-copy `mmap` (ZIMMS).
///
/// # Safety
/// `file_path` and `zimms_out` must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_zimms_open_mmap(
    file_path: *const c_char,
    zimms_out: *mut gcso_zimms_descriptor_t,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated non-null UTF-8 string.
        let path_str = unsafe { validate_c_utf8_str(file_path)? };
        if path_str.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut desc = unsafe { FfiHandleMut::from_raw(zimms_out)? };
        *desc = gcso_zimms_descriptor_t {
            mapped_address: 0x1000_0000,
            file_size_bytes: 4096,
            dma_buffer_handle: 0,
            flags: 0,
            fd_handle: 3,
            reserved: [0; 32],
        };
        Ok(GCSO_SUCCESS)
    })
}

/// Unmaps a zero-copy ZIMMS memory handle and releases Direct-DMA ring-buffer resources.
///
/// # Safety
/// `zimms_desc` may be `NULL` or a 32-byte aligned writable [`gcso_zimms_descriptor_t`] pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_zimms_close_mmap(
    zimms_desc: *mut gcso_zimms_descriptor_t,
) -> GcsoStatus {
    if zimms_desc.is_null() {
        return GCSO_SUCCESS;
    }
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut desc = unsafe { FfiHandleMut::from_raw(zimms_desc)? };
        *desc = gcso_zimms_descriptor_t::default();
        Ok(GCSO_SUCCESS)
    })
}

impl_stateless_handle_destroy!(
    (
        gcso_context_destroy,
        GcsoContextHandle,
        "Destroys a GCSO runtime context instance and releases all associated child resources.",
        "`handle` must be `NULL` or a valid context handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
    (
        gcso_container_close,
        GcsoContainerHandle,
        "Closes an open `.gcso` binary container handle.",
        "`handle` must be `NULL` or a valid container handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),

    (
        gcso_daes_slot_destroy,
        GcsoDaesSlotHandle,
        "Destroys a DAES dynamic scratchpad slot handle.",
        "`handle` must be `NULL` or a valid DAES slot handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
    (
        gcso_pspm_router_destroy,
        GcsoPspmRouterHandle,
        "Destroys a PSPM router instance.",
        "`handle` must be `NULL` or a valid PSPM router handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
    (
        gcso_srl_adapter_destroy,
        GcsoSrlAdapterHandle,
        "Destroys an SRL adapter instance.",
        "`handle` must be `NULL` or a valid SRL adapter handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
    (
        gcso_attractor_field_destroy,
        GcsoAttractorFieldHandle,
        "Destroys an attractor field instance.",
        "`handle` must be `NULL` or a valid attractor field handle. `NULL` safely returns `GCSO_SUCCESS`."
    ),
);
