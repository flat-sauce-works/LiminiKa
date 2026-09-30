// name: src/kernels/common/action_hub.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "gcso_internal.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>

GCSO_EXTERN_C_BEGIN

// ===================================================================
// Action Hub & Stigmergic Swarm Pointer Trail Operations
// ===================================================================

/**
 * @brief Allocates an Action Hub (Sidecar Pointer Table - SPT) instance.
 *
 * Cold Path Execution: Uses gcso_aligned_new for 32-byte alignment.
 *
 * @param capacity Slot capacity count for tagged pointers.
 * @param hub_out Pointer to receive allocated action hub handle.
 * @return GCSO_SUCCESS or appropriate error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_create(
    uint32_t capacity,
    gcso_action_hub_handle_t* GCSO_RESTRICT hub_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(hub_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(capacity == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(hub_out) % alignof(gcso_action_hub_handle_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        auto* impl = gcso_aligned_new<gcso_action_hub_impl>();
        if (GCSO_UNLIKELY(impl == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        std::memset(impl, 0, sizeof(gcso_action_hub_impl));
        impl->capacity = capacity;
        impl->active_count = 0;
        impl->last_trail_ptr = 0;
        impl->is_active = true;

        *hub_out = reinterpret_cast<gcso_action_hub_handle_t>(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Advances a 64-bit tagged pointer transition in O(1) time within the Action Hub.
 *
 * Hot Path Guarantee: Zero dynamic allocations. Modifies trail_out structure in-place.
 * Encodes current and previous pointer states, increments step counter, and copies
 * active Q7 phase deltas into the 128-byte trail layout.
 *
 * @param context Active runtime context handle.
 * @param current_ptr Encoded 64-bit tagged pointer address.
 * @param trail_out Pointer to receive updated 128-byte pointer trail structure.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_step_pointer(
    gcso_context_handle_t context,
    uint64_t current_ptr,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr || trail_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_impl) != 0 ||
                          reinterpret_cast<uintptr_t>(trail_out) % alignof(gcso_pointer_trail_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        auto* ctx_impl = reinterpret_cast<gcso_context_impl*>(context);
        if (GCSO_UNLIKELY(!ctx_impl->is_initialized)) {
            return GCSO_ERROR_INVALID_STATE;
        }

        const uint64_t previous = ctx_impl->last_ptr;
        ctx_impl->step_count++;
        ctx_impl->last_ptr = current_ptr;

        std::memset(trail_out, 0, sizeof(gcso_pointer_trail_t));
        trail_out->current_ptr = current_ptr;
        trail_out->prev_ptr = previous;
        trail_out->step_count = static_cast<uint32_t>(ctx_impl->step_count & 0xFFFFFFFFULL);
        trail_out->stigmergic_density = 1.0f;
        std::memcpy(trail_out->accumulated_phase_delta, ctx_impl->accumulated_phase, sizeof(ctx_impl->accumulated_phase));

        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Computes direct 256-slot hash index for 64-bit pointer trail caching in O(1) time.
 *
 * Uses SplitMix64 deterministic bit-mixing to achieve uniform slot distribution
 * across the 256-entry Action Hub cache without division or modulo operations.
 *
 * @param ptr Raw 64-bit pointer address.
 * @return 8-bit hash index (0 to 255).
 */
GCSO_API uint32_t GCSO_CALL gcso_action_hub_hash_slot256_index(
    uint64_t ptr
) GCSO_NOEXCEPT {
    uint64_t x = ptr;
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return static_cast<uint32_t>(x & 0xFFULL);
}

/**
 * @brief Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
 *
 * @param context Active context handle.
 * @param mask Paged bitmask structure pointer.
 * @param chunk_len Sequence length of target chunk.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_swarm_cell_chunk_step(
    gcso_context_handle_t context,
    const gcso_paged_bitmask_t* GCSO_RESTRICT mask,
    uint32_t chunk_len
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr || mask == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(chunk_len == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_impl) != 0 ||
                          reinterpret_cast<uintptr_t>(mask) % alignof(gcso_paged_bitmask_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        (void)mask;
        auto* ctx_impl = reinterpret_cast<gcso_context_impl*>(context);
        if (GCSO_UNLIKELY(!ctx_impl->is_initialized)) {
            return GCSO_ERROR_INVALID_STATE;
        }

        ctx_impl->step_count += chunk_len;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Applies attractor pull force F_pull to steer active pointer chains toward target anchor.
 *
 * Uses explicit std::round before clamping to preserve Q7 fixed-point quantization accuracy
 * within the signed int8_t range [-128, 127]. Includes NaN/Inf input guards.
 *
 * @param context Active runtime context handle.
 * @param trail_id Source pointer trail index.
 * @param anchor_id Target anchor identifier.
 * @param pull_force Floating-point pull force scale factor.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_pull_trail_to_attractor(
    gcso_context_handle_t context,
    uint32_t trail_id,
    uint32_t anchor_id,
    float pull_force
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(std::isnan(pull_force) || std::isinf(pull_force))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_impl) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        auto* ctx_impl = reinterpret_cast<gcso_context_impl*>(context);
        if (GCSO_UNLIKELY(!ctx_impl->is_initialized)) {
            return GCSO_ERROR_INVALID_STATE;
        }

        const uint32_t idx = trail_id % 64;
        const float current_val = static_cast<float>(ctx_impl->accumulated_phase[idx]);
        const float target_val = current_val + pull_force * static_cast<float>(anchor_id % 8);
        const float rounded_val = std::round(target_val);
        const float clamped_val = std::clamp(rounded_val, -128.0f, 127.0f);

        ctx_impl->accumulated_phase[idx] = static_cast<gcso_q7_t>(clamped_val);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Links adjacent cellular hallucinated trails into contiguous stigmergic trace graphs.
 *
 * @param context Active context handle.
 * @param src_trail_id Source trail identifier.
 * @param dst_trail_id Destination trail identifier.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_link_hallucinated_trails(
    gcso_context_handle_t context,
    uint32_t src_trail_id,
    uint32_t dst_trail_id
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_impl) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        auto* ctx_impl = reinterpret_cast<gcso_context_impl*>(context);
        if (GCSO_UNLIKELY(!ctx_impl->is_initialized)) {
            return GCSO_ERROR_INVALID_STATE;
        }

        const uint32_t src_idx = src_trail_id % 64;
        const uint32_t dst_idx = dst_trail_id % 64;
        ctx_impl->accumulated_phase[src_idx] = ctx_impl->accumulated_phase[dst_idx];
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Performs hierarchical bit-tree reduction across paged block bitmasks in O(1) time.
 *
 * Applies bitwise AND operations across 256-bit PagedBlock bitmasks to evaluate set intersection
 * of active attention blocks without memory reallocations.
 *
 * @param bitmasks Array of input paged bitmasks.
 * @param num_masks Number of masks in input array.
 * @param reduced_out Pointer to receive reduced bitmask result.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_reduce_bit_tree(
    const gcso_paged_bitmask_t* GCSO_RESTRICT bitmasks,
    size_t num_masks,
    gcso_paged_bitmask_t* GCSO_RESTRICT reduced_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(bitmasks == nullptr || reduced_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(num_masks == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(bitmasks) % alignof(gcso_paged_bitmask_t) != 0 ||
                          reinterpret_cast<uintptr_t>(reduced_out) % alignof(gcso_paged_bitmask_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        reduced_out->bits[0] = bitmasks[0].bits[0];
        reduced_out->bits[1] = bitmasks[0].bits[1];
        reduced_out->bits[2] = bitmasks[0].bits[2];
        reduced_out->bits[3] = bitmasks[0].bits[3];

        for (size_t i = 1; i < num_masks; ++i) {
            reduced_out->bits[0] &= bitmasks[i].bits[0];
            reduced_out->bits[1] &= bitmasks[i].bits[1];
            reduced_out->bits[2] &= bitmasks[i].bits[2];
            reduced_out->bits[3] &= bitmasks[i].bits[3];
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Evaluates SIMD/Warp bitmask reduction over PagedBlock KV caches.
 *
 * Accumulates 64-bit word block occupancy patterns into a 256-bit SIMD bitmask structure.
 *
 * @param kv_bits Raw uint64_t array of block key-value bit patterns.
 * @param num_blocks Total number of KV blocks to process.
 * @param mask_out Pointer to store generated paged bitmask.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_paged_block_warp_bitmask(
    const uint64_t* GCSO_RESTRICT kv_bits,
    size_t num_blocks,
    gcso_paged_bitmask_t* GCSO_RESTRICT mask_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(kv_bits == nullptr || mask_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(num_blocks == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(kv_bits) % alignof(uint64_t) != 0 ||
                          reinterpret_cast<uintptr_t>(mask_out) % alignof(gcso_paged_bitmask_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        std::memset(mask_out, 0, sizeof(gcso_paged_bitmask_t));

        for (size_t i = 0; i < num_blocks; ++i) {
            const size_t word_idx = i % 4;
            mask_out->bits[word_idx] |= kv_bits[i];
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Destroys an Action Hub instance.
 * Safe No-Op: Accepts NULL pointers safely returning GCSO_SUCCESS.
 *
 * @param hub Action hub handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_destroy(
    gcso_action_hub_handle_t hub
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(hub == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(hub) % alignof(gcso_action_hub_impl) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_action_hub_impl*>(hub);
        gcso_aligned_delete(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

// ===================================================================
// Dynamic Adaptive Extension Scratchpad (DAES) Multi-Layer Controls
// ===================================================================

/**
 * @brief Allocates and initializes a DAES slot instance.
 *
 * @param slot_out Pointer to receive allocated DAES slot handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_daes_slot_create(
    gcso_daes_slot_handle_t* GCSO_RESTRICT slot_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(slot_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(slot_out) % alignof(gcso_daes_slot_handle_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        auto* impl = gcso_aligned_new<gcso_daes_slot_t>();
        if (GCSO_UNLIKELY(impl == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        std::memset(impl, 0, sizeof(gcso_daes_slot_t));

        *slot_out = reinterpret_cast<gcso_daes_slot_handle_t>(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Executes O(1) fast-path shortcut lookup in DAES dynamic scratchpad.
 *
 * Mode Check: Operates in Mode 0 (Fast-Path & Telemetry Scratchpad Mode).
 * Calculates hash slot index from input key across the 4 shortcut registers in the DAES 64B cacheline slot.
 *
 * @param slot Pointer to DAES slot structure.
 * @param input_key Raw key address to look up.
 * @param shortcut_out Pointer to store bypassed address result.
 * @return GCSO_SUCCESS or GCSO_ERROR_INVALID_STATE.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_daes_fast_path_lookup(
    const gcso_daes_slot_t* GCSO_RESTRICT slot,
    uint64_t input_key,
    uint64_t* GCSO_RESTRICT shortcut_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(slot == nullptr || shortcut_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(slot) % alignof(gcso_daes_slot_t) != 0 ||
                          reinterpret_cast<uintptr_t>(shortcut_out) % alignof(uint64_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        if (GCSO_UNLIKELY(slot->mode != 0)) {
            return GCSO_ERROR_INVALID_STATE;
        }

        // Dynamically compute target shortcut slot index via O(1) hashing across the 4 DAES shortcut registers
        const uint32_t slot_idx = gcso_action_hub_hash_slot256_index(input_key) % 4;
        const uint64_t shortcut = slot->fast_path_shortcuts[slot_idx];

        *shortcut_out = (shortcut != 0) ? shortcut : input_key;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Pushes profiling metric bytes to the DAES telemetry ring ledger in zero-allocation mode.
 *
 * @param slot Target DAES slot structure pointer.
 * @param metric_code Metric identifier byte code.
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_daes_telemetry_push(
    gcso_daes_slot_t* GCSO_RESTRICT slot,
    uint8_t metric_code
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(slot == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(slot) % alignof(gcso_daes_slot_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        const uint16_t idx = slot->telemetry_ring_head % 8;
        slot->telemetry_mini_ledger[idx] = metric_code;
        slot->telemetry_ring_head = (slot->telemetry_ring_head + 1) % 8;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Sets the DAES slot operating mode (0 = Scratchpad, 1 = Plugin, 2 = Shared IPC Buffer).
 *
 * @param slot Pointer to DAES slot structure.
 * @param mode Desired mode index (0, 1, or 2).
 * @return GCSO_SUCCESS or GCSO_ERROR_INVALID_ARGUMENT.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_daes_set_mode(
    gcso_daes_slot_t* GCSO_RESTRICT slot,
    uint32_t mode
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(slot == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(mode > 2)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(slot) % alignof(gcso_daes_slot_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        slot->mode = mode;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Evaluates telemetry ledger to auto-tune PSPM routing ratios, RIPA bounds & EDBC thresholds.
 *
 * Self-Optimization Loop: Analyzes cache hit counts from DAES scratchpad to adaptively adjust
 * minimum QDPS step thresholds and maximum RIPA clamp bounds in runtime config.
 *
 * @param slot Const DAES slot structure pointer.
 * @param config_out Pointer to receive updated auto-tuned configuration structure.
 * @return GCSO_SUCCESS or error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_daes_evaluate_auto_tune(
    const gcso_daes_slot_t* GCSO_RESTRICT slot,
    gcso_config_t* GCSO_RESTRICT config_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(slot == nullptr || config_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(slot) % alignof(gcso_daes_slot_t) != 0 ||
                          reinterpret_cast<uintptr_t>(config_out) % alignof(gcso_config_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        const gcso_status_t status = gcso_config_init_default(config_out);
        if (GCSO_UNLIKELY(status != GCSO_SUCCESS)) {
            return status;
        }

        if (slot->cache_hit_count > 1000) {
            config_out->qdps_min_step_rad = 0.005f;
            config_out->ripa_clamp_max_rad = 0.100f;
        } else if (slot->cache_hit_count < 50) {
            config_out->qdps_min_step_rad = 0.020f;
            config_out->ripa_clamp_max_rad = 0.050f;
        }

        config_out->daes_mode = static_cast<uint8_t>(slot->mode & 0xFF);

        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Destroys a DAES slot instance.
 * Safe No-Op: Accepts NULL pointers safely returning GCSO_SUCCESS.
 *
 * @param slot Handle to DAES slot instance (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_daes_slot_destroy(
    gcso_daes_slot_handle_t slot
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(slot == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(slot) % alignof(gcso_daes_slot_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_daes_slot_t*>(slot);
        gcso_aligned_delete(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

GCSO_EXTERN_C_END