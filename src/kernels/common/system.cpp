// name: src/kernels/common/system.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "gcso_internal.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>

GCSO_EXTERN_C_BEGIN

/**
 * @brief Packs a 48-bit canonical address, 8-bit tag, and Q7 phase offset into a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_pack(
    uint64_t raw_addr,
    uint8_t tag,
    gcso_q7_t phase
) GCSO_NOEXCEPT {
    return gcso_tagged_ptr_pack_inline(raw_addr, tag, phase);
}

/**
 * @brief Unpacks the 48-bit canonical address from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_unpack_addr(
    uint64_t ptr
) GCSO_NOEXCEPT {
    return gcso_tagged_ptr_unpack_addr_inline(ptr);
}

/**
 * @brief Unpacks the 8-bit domain/action tag from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint8_t GCSO_CALL gcso_tagged_ptr_unpack_tag(
    uint64_t ptr
) GCSO_NOEXCEPT {
    return gcso_tagged_ptr_unpack_tag_inline(ptr);
}

/**
 * @brief Unpacks the 8-bit Q7 phase state from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API gcso_q7_t GCSO_CALL gcso_tagged_ptr_unpack_phase(
    uint64_t ptr
) GCSO_NOEXCEPT {
    return gcso_tagged_ptr_unpack_phase_inline(ptr);
}

/**
 * @brief Packs a 48-bit address, 3-bit Attractor Slot ID, 5-bit Action Mode mask, and Q7 phase into a tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_pack_action(
    uint64_t raw_addr,
    uint8_t attractor_slot,
    uint8_t action_mode_flags,
    gcso_q7_t phase
) GCSO_NOEXCEPT {
    const auto tag = static_cast<uint8_t>(
        ((attractor_slot & GCSO_ATTRACTOR_SLOT_ID_MASK) << 5U) |
        (action_mode_flags & GCSO_ACTION_MODE_ALL_MASK)
    );
    return gcso_tagged_ptr_pack_inline(raw_addr, tag, phase);
}

/**
 * @brief Extracts the 5-bit Action Mode mask from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint8_t GCSO_CALL gcso_tagged_ptr_unpack_action_mode(
    uint64_t ptr
) GCSO_NOEXCEPT {
    return static_cast<uint8_t>(gcso_tagged_ptr_unpack_tag_inline(ptr) & GCSO_ACTION_MODE_ALL_MASK);
}

/**
 * @brief Extracts the 3-bit Target Attractor Working Lane ID (0..7) from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint8_t GCSO_CALL gcso_tagged_ptr_unpack_attractor_slot(
    uint64_t ptr
) GCSO_NOEXCEPT {
    return static_cast<uint8_t>((gcso_tagged_ptr_unpack_tag_inline(ptr) >> 5U) & GCSO_ATTRACTOR_SLOT_ID_MASK);
}

/**
 * @brief Evolves a 64-bit tagged pointer's Action Mode flags and Target Attractor Slot in O(1) time.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_evolve_mode_stigmergically(
    uint64_t ptr,
    float stigmergic_density,
    float entropy_h_tilde,
    uint8_t repulsion_triggered,
    uint8_t dominant_attractor_slot,
    float low_entropy_skip_thresh,
    float high_density_srl_thresh
) GCSO_NOEXCEPT {
    const uint64_t addr = gcso_tagged_ptr_unpack_addr_inline(ptr);
    gcso_q7_t phase = gcso_tagged_ptr_unpack_phase_inline(ptr);

    uint8_t slot = dominant_attractor_slot & GCSO_ATTRACTOR_SLOT_ID_MASK;
    uint8_t mode = GCSO_ACTION_MODE_PHASE_STEER;

    if (repulsion_triggered != 0U) {
        slot = GCSO_ATTRACTOR_SLOT_REPULSION;
        mode = static_cast<uint8_t>(
            GCSO_ACTION_MODE_PHASE_STEER |
            GCSO_ACTION_MODE_PSPM_ROUTE |
            GCSO_ACTION_MODE_REPULSION_GUARD
        );
        phase = (phase == -128) ? static_cast<gcso_q7_t>(127) : static_cast<gcso_q7_t>(-phase);
    } else if (entropy_h_tilde < low_entropy_skip_thresh && stigmergic_density >= 0.80f) {
        mode = GCSO_ACTION_MODE_LAYER_SKIP;
    } else if (stigmergic_density >= high_density_srl_thresh) {
        mode = static_cast<uint8_t>(
            GCSO_ACTION_MODE_PHASE_STEER |
            GCSO_ACTION_MODE_PSPM_ROUTE |
            GCSO_ACTION_MODE_SRL_INJECT
        );
    } else if (stigmergic_density >= 0.50f || entropy_h_tilde >= 0.35f) {
        mode = static_cast<uint8_t>(GCSO_ACTION_MODE_PHASE_STEER | GCSO_ACTION_MODE_PSPM_ROUTE);
    }

    return gcso_tagged_ptr_pack_action(addr, slot, mode, phase);
}

/**
 * @brief Retrieves the numeric version identifiers for the GCSO C-ABI (v0.1.1).
 *
 * @param major Pointer to store the major version number.
 * @param minor Pointer to store the minor version number.
 * @param patch Pointer to store the patch version number.
 * @return GCSO_SUCCESS, GCSO_ERROR_NULL_POINTER, or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_abi_get_version(
    uint32_t* GCSO_RESTRICT major,
    uint32_t* GCSO_RESTRICT minor,
    uint32_t* GCSO_RESTRICT patch
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(major == nullptr || minor == nullptr || patch == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(major) % alignof(uint32_t) != 0 ||
                      reinterpret_cast<uintptr_t>(minor) % alignof(uint32_t) != 0 ||
                      reinterpret_cast<uintptr_t>(patch) % alignof(uint32_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }

    try {
        *major = GCSO_ABI_VERSION_MAJOR;
        *minor = GCSO_ABI_VERSION_MINOR;
        *patch = GCSO_ABI_VERSION_PATCH;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Returns the static semantic version string literal for the GCSO kernel engine ("0.1.1").
 *
 * @return Const pointer to null-terminated static version string.
 */
GCSO_NODISCARD GCSO_API const char* GCSO_CALL gcso_abi_get_version_string(void) GCSO_NOEXCEPT {
    return GCSO_ABI_VERSION_STRING;
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
        gcso_capability_flags_t caps = GCSO_CAP_CPU_BASELINE;

#if defined(LIMINIKA_ENABLE_CUDA)
        caps |= GCSO_CAP_CUDA_BACKEND;
#endif

#if defined(LIMINIKA_ENABLE_VULKAN)
        caps |= GCSO_CAP_VULKAN_BACKEND;
#endif

#if defined(LIMINIKA_ENABLE_METAL) || defined(__APPLE__)
        caps |= GCSO_CAP_METAL_BACKEND;
#endif

        *flags = caps;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Populates default settings for a gcso_config_t structure strictly conforming to GCSO standards.
 *
 * Configures optimal parameters for 2-4GB VRAM execution targets, setting default head dimensions (128),
 * head counts (32), paged block sizes (32 tokens), RIPA soft-clamp bounds (5 degrees = 0.087266 rad), and
 * qdps minimum step thresholds (0.01 rad) to avoid grid jitter under ultra-low quantization.
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
 * @brief Validates a gcso_config_t descriptor against GCSO v0.1.1 architectural invariants.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_config_validate(
    const gcso_config_t* GCSO_RESTRICT config
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(config == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(config) % alignof(gcso_config_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        const bool valid_dims =
            (config->head_dim != 0U) &&
            ((config->head_dim & 1U) == 0U) &&
            (config->num_heads != 0U) &&
            (config->num_heads <= GCSO_Q7_PHASE_ARRAY_LEN) &&
            (config->paged_block_size == 16U || config->paged_block_size == 32U);
        const bool valid_scales =
            std::isfinite(config->q7_phase_scale) && (config->q7_phase_scale > 0.0f) &&
            std::isfinite(config->ripa_clamp_max_rad) && (config->ripa_clamp_max_rad > 0.0f) &&
            std::isfinite(config->qdps_min_step_rad) && (config->qdps_min_step_rad >= 0.0f) &&
            std::isfinite(config->entropy_singularity_eps) && (config->entropy_singularity_eps > 0.0f);
        const bool valid_caps =
            (config->max_prompt_anchors != 0U) &&
            (config->action_hub_capacity != 0U) &&
            (config->daes_mode <= 2U);

        return (valid_dims && valid_scales && valid_caps)
                   ? GCSO_SUCCESS
                   : GCSO_ERROR_INVALID_ARGUMENT;
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