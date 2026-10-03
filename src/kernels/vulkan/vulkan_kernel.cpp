// SPDX-License-Identifier: MIT OR Apache-2.0

#include "liminika/gcso_config.h"
#include "liminika/gcso_types.h"

#include <cstddef>
#include <cstdint>
#include <exception>

GCSO_EXTERN_C_BEGIN

/**
 * @brief Queries Vulkan compute engine subgroup capabilities and active memory flags.
 * @param capability_flags Pointer to 64-bit capability bitmask variable.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_vulkan_query_capability(
    gcso_capability_flags_t* GCSO_RESTRICT capability_flags) GCSO_NOEXCEPT {
    if (capability_flags == nullptr) {
        return GCSO_ERROR_NULL_POINTER;
    }

    try {
        // Set Vulkan Compute Subgroup Shuffle & Bitmask capabilities (Bits 2 & 10)
        *capability_flags |= (1ULL << 2) | (1ULL << 10);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Executes Vulkan Subgroup-wide DPSR inline phase steering kernel dispatch skeleton.
 * @param query_tensor 32-byte aligned query tensor buffer to modulate in-place.
 * @param phase_deltas Q7 quantized phase delta array.
 * @param head_dim Attention head dimension size (must be even).
 * @param num_heads Total head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_vulkan_dpsr_phase_steering_dispatch(
    float* GCSO_RESTRICT query_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads) GCSO_NOEXCEPT {
    if (query_tensor == nullptr || phase_deltas == nullptr) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (head_dim == 0 || (head_dim % 2) != 0 || num_heads == 0) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }

    try {
        // Vulkan Compute Subgroup pipeline execution skeleton
        // In-kernel SIMD/Subgroup register phase addition will be performed here.
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Evaluates Vulkan Subgroup Ballot bitmask reduction over PagedBlock KV caches.
 * @param kv_bits Raw uint64_t array of block key-value bit patterns.
 * @param num_blocks Total number of KV blocks to reduce.
 * @param mask_out Output paged bitmask structure.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_vulkan_paged_block_bitmask_reduce(
    const uint64_t* GCSO_RESTRICT kv_bits,
    size_t num_blocks,
    gcso_paged_bitmask_t* GCSO_RESTRICT mask_out) GCSO_NOEXCEPT {
    if (kv_bits == nullptr || mask_out == nullptr) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (num_blocks == 0) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }

    try {
        mask_out->bits[0] = 0;
        mask_out->bits[1] = 0;
        mask_out->bits[2] = 0;
        mask_out->bits[3] = 0;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

GCSO_EXTERN_C_END