// name: src/kernels/common/system.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "gcso_internal.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace liminika {

// Static string literal defining the current GCSO C-ABI release version (v2.0.0 conforming strictly to c_abi_spec.md).
constexpr const char* GCSO_KERNEL_VERSION_STRING = "2.0.0";

} // namespace liminika

GCSO_EXTERN_C_BEGIN

/**
 * @brief Retrieves the numeric version identifiers for the GCSO C-ABI.
 * Conforms strictly to ABI version 2.0.0 specified in c_abi_spec.md and gcso_snapshot_header_t (0x00020000).
 *
 * Verifies output pointer alignment and populates semantic versioning fields safely.
 *
 * @param major Pointer to store the major version number (2).
 * @param minor Pointer to store the minor version number (0).
 * @param patch Pointer to store the patch version number (0).
 */
GCSO_API void GCSO_CALL gcso_abi_get_version(
    uint32_t* GCSO_RESTRICT major,
    uint32_t* GCSO_RESTRICT minor,
    uint32_t* GCSO_RESTRICT patch
) GCSO_NOEXCEPT {
    if (major != nullptr) {
        if (GCSO_LIKELY(reinterpret_cast<uintptr_t>(major) % alignof(uint32_t) == 0)) {
            *major = 2;
        }
    }
    if (minor != nullptr) {
        if (GCSO_LIKELY(reinterpret_cast<uintptr_t>(minor) % alignof(uint32_t) == 0)) {
            *minor = 0;
        }
    }
    if (patch != nullptr) {
        if (GCSO_LIKELY(reinterpret_cast<uintptr_t>(patch) % alignof(uint32_t) == 0)) {
            *patch = 0;
        }
    }
}

/**
 * @brief Returns the static semantic version string literal for the GCSO kernel engine ("2.0.0").
 *
 * @return Const pointer to null-terminated static version string.
 */
GCSO_NODISCARD GCSO_API const char* GCSO_CALL gcso_abi_get_version_string(void) GCSO_NOEXCEPT {
    return liminika::GCSO_KERNEL_VERSION_STRING;
}

/**
 * @brief Maps gcso_status_t status and error codes to descriptive static string representations.
 *
 * Provides 1:1 cross-language error diagnosis across C-ABI, C++, and Rust FFI boundaries.
 *
 * @param status Status code enum value.
 * @return Const pointer to descriptive static string representation.
 */
GCSO_NODISCARD GCSO_API const char* GCSO_CALL gcso_status_to_string(gcso_status_t status) GCSO_NOEXCEPT {
    switch (status) {
        case GCSO_SUCCESS:
            return "GCSO_SUCCESS";
        case GCSO_ERROR_INVALID_ARGUMENT:
            return "GCSO_ERROR_INVALID_ARGUMENT";
        case GCSO_ERROR_OUT_OF_MEMORY:
            return "GCSO_ERROR_OUT_OF_MEMORY";
        case GCSO_ERROR_BUFFER_TOO_SMALL:
            return "GCSO_ERROR_BUFFER_TOO_SMALL";
        case GCSO_ERROR_PANIC_CAUGHT:
            return "GCSO_ERROR_PANIC_CAUGHT";
        case GCSO_ERROR_NULL_POINTER:
            return "GCSO_ERROR_NULL_POINTER";
        case GCSO_ERROR_INVALID_STATE:
            return "GCSO_ERROR_INVALID_STATE";
        case GCSO_ERROR_VERSION_MISMATCH:
            return "GCSO_ERROR_VERSION_MISMATCH";
        case GCSO_ERROR_MISALIGNED_POINTER:
            return "GCSO_ERROR_MISALIGNED_POINTER";
        case GCSO_ERROR_IO_FAILURE:
            return "GCSO_ERROR_IO_FAILURE";
        case GCSO_ERROR_ACTION_HUB_FULL:
            return "GCSO_ERROR_ACTION_HUB_FULL";
        case GCSO_ERROR_NOT_IMPLEMENTED:
            return "GCSO_ERROR_NOT_IMPLEMENTED";
        case GCSO_ERROR_ATTRACTOR_NOT_FOUND:
            return "GCSO_ERROR_ATTRACTOR_NOT_FOUND";
        case GCSO_ERROR_DPSR_PHASE_OVERFLOW:
            return "GCSO_ERROR_DPSR_PHASE_OVERFLOW";
        case GCSO_ERROR_QDPS_UNDERFLOW:
            return "GCSO_ERROR_QDPS_UNDERFLOW";
        case GCSO_ERROR_EDBC_SINGULARITY:
            return "GCSO_ERROR_EDBC_SINGULARITY";
        case GCSO_ERROR_CONTAINER_CORRUPTED:
            return "GCSO_ERROR_CONTAINER_CORRUPTED";
        case GCSO_ERROR_ZIMMS_MAPPING_FAILED:
            return "GCSO_ERROR_ZIMMS_MAPPING_FAILED";
        case GCSO_ERROR_PSPM_ROUTING_FAILED:
            return "GCSO_ERROR_PSPM_ROUTING_FAILED";
        case GCSO_ERROR_PPRC_SEEK_FAILED:
            return "GCSO_ERROR_PPRC_SEEK_FAILED";
        case GCSO_ERROR_OBSTRUCTION_UNRESOLVED:
            return "GCSO_ERROR_OBSTRUCTION_UNRESOLVED";
        case GCSO_ERROR_EXTENSION_NOT_LOADED:
            return "GCSO_ERROR_EXTENSION_NOT_LOADED";
        case GCSO_ERROR_DAES_SCRATCHPAD_FULL:
            return "GCSO_ERROR_DAES_SCRATCHPAD_FULL";
        case GCSO_ERROR_UNKNOWN:
        default:
            return "GCSO_ERROR_UNKNOWN";
    }
}

/**
 * @brief Queries current compute capability flags (CPU SIMD, CUDA, Vulkan, Metal).
 *
 * @param flags Pointer to store 64-bit bitmask capability flags.
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER / GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_abi_query_capability(
    gcso_capability_flags_t* GCSO_RESTRICT flags
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(flags == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(flags) % alignof(gcso_capability_flags_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        gcso_capability_flags_t caps = 0x0000000000000001ULL; // Base CPU Capability Flag

#if defined(LIMINIKA_ENABLE_CUDA)
        caps |= (1ULL << 1);
#endif

#if defined(LIMINIKA_ENABLE_VULKAN)
        caps |= (1ULL << 2);
#endif

#if defined(LIMINIKA_ENABLE_METAL)
        caps |= (1ULL << 3);
#endif

        *flags = caps;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Populates default settings for a gcso_config_t structure strictly conforming to c_abi_spec.md Section 5.4.D.
 *
 * Configures optimal parameters for 2-4GB VRAM execution targets, setting default head dimensions (128),
 * head counts (32), paged block sizes (32 tokens), RIPA soft-clamp bounds (5 degrees = 0.087266 rad), and
 * QDPS minimum step thresholds (0.01 rad) to avoid grid jitter under ultra-low quantization.
 *
 * @param config Pointer to user-allocated configuration descriptor to populate.
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER / GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_config_init_default(
    gcso_config_t* GCSO_RESTRICT config
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(config == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(config) % alignof(gcso_config_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        std::memset(config, 0, sizeof(gcso_config_t));
        config->head_dim = 128;
        config->num_heads = 32;
        config->paged_block_size = 32;
        config->q7_phase_scale = 3.14159265358979323846f / 128.0f;
        config->ripa_clamp_max_rad = 0.0872664626f; // 5 degrees soft clamp limit
        config->qdps_min_step_rad = 0.01f;          // QDPS cutoff threshold
        config->entropy_singularity_eps = 1e-12f;   // Singularity guard eps_log
        config->max_prompt_anchors = 64;
        config->action_hub_capacity = 256;
        config->enable_cuda_warp_shuffle = 1;
        config->enable_zero_alloc_strict = 1;
        config->daes_mode = 0; // Default: Fast-Path & Telemetry Scratchpad Mode
        config->reserved_flags = 0;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Safely frees a heap-allocated string returned across the FFI boundary.
 *
 * Safe No-Op: Accepts NULL pointers safely returning without action.
 *
 * @param str Const string pointer to free (safe no-op if NULL).
 */
GCSO_API void GCSO_CALL gcso_free_string(
    const char* str
) GCSO_NOEXCEPT {
    if (str != nullptr) {
        std::free(const_cast<char*>(str));
    }
}

GCSO_EXTERN_C_END