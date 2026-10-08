// name: src/kernels/common/runtime_context.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "gcso_internal.h"
#include "liminika/gcso_config.h"

#include <cstring>
#include <exception>

namespace {

/**
 * @brief Computes a CRC32 checksum over payload buffers for .gcso container verification.
 * Optimized with explicit ternary evaluation to prevent sign conversion warnings under MSVC /W4.
 *
 * @param data Byte pointer to payload buffer.
 * @param len Length of byte payload in bytes.
 * @return Calculated uint32_t CRC32 value.
 */
uint32_t compute_crc32(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ ((crc & 1) ? 0x82F63B78U : 0U);
        }
    }
    return ~crc;
}

} // namespace

GCSO_EXTERN_C_BEGIN

/**
 * @brief Allocates and initializes an opaque GCSO runtime context instance.
 *
 * Executed in Cold Path. Uses gcso_aligned_new for strict 64-byte alignment to fit CPU/GPU cache lines.
 * Validates configuration boundaries (even head_dim, head count <= 64) before constructing state.
 *
 * @param config Pointer to initial configuration settings.
 * @param context_out Pointer to receive created runtime context handle.
 * @return GCSO_SUCCESS or appropriate error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_create(
    const gcso_config_t* GCSO_RESTRICT config,
    gcso_context_handle_t* GCSO_RESTRICT context_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(config == nullptr || context_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(config) % alignof(gcso_config_t) != 0 ||
                      reinterpret_cast<uintptr_t>(context_out) % alignof(gcso_context_handle_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if (GCSO_UNLIKELY(config->head_dim == 0 || (config->head_dim % 2 != 0) ||
                      config->num_heads == 0 || config->num_heads > 64)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        auto* impl = gcso_aligned_new<gcso_context_impl>();
        if (GCSO_UNLIKELY(impl == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }

        std::memcpy(&impl->config, config, sizeof(gcso_config_t));
        impl->active_token_id = 0;
        impl->prompt_anchor_count = 0;
        impl->head_dim = config->head_dim;
        impl->num_heads = config->num_heads;
        impl->step_count = 0;
        impl->last_ptr = 0;
        std::memset(impl->accumulated_phase, 0, sizeof(impl->accumulated_phase));
        impl->is_initialized = true;
        impl->is_active = true;

        *context_out = reinterpret_cast<gcso_context_handle_t>(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Resets transient phase accumulators and step counters without freeing allocated tables.
 * Safe No-Op: Accepts NULL pointers safely returning GCSO_SUCCESS.
 *
 * @param context Handle to active runtime context (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_reset(
    gcso_context_handle_t context
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_impl) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_context_impl*>(context);
        impl->active_token_id = 0;
        impl->prompt_anchor_count = 0;
        impl->step_count = 0;
        impl->last_ptr = 0;
        std::memset(impl->accumulated_phase, 0, sizeof(impl->accumulated_phase));
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Primary Baseline Endpoint: Projects natural language system prompt text as an Anchor Attractor.
 *
 * Registers natural language prompts as primary baseline attractor targets in phase space without
 * rewriting base weight tensors in VRAM. Validates capacity bounds against max_prompt_anchors.
 *
 * @param context Active context handle.
 * @param prompt_text Null-terminated UTF-8 system prompt string.
 * @param weight Attractor pull force scale factor.
 * @return GCSO_SUCCESS or GCSO_ERROR_ACTION_HUB_FULL when capacity is reached.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_set_system_prompt_anchor(
    gcso_context_handle_t context,
    const char* GCSO_RESTRICT prompt_text,
    float weight
) GCSO_NOEXCEPT {
    (void)weight;

    if (GCSO_UNLIKELY(context == nullptr || prompt_text == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_impl) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_context_impl*>(context);
        if (GCSO_UNLIKELY(!impl->is_initialized)) {
            return GCSO_ERROR_INVALID_STATE;
        }

        if (GCSO_UNLIKELY(impl->prompt_anchor_count >= impl->config.max_prompt_anchors)) {
            return GCSO_ERROR_ACTION_HUB_FULL;
        }

        impl->prompt_anchor_count++;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Per-token Hot Path inference step.
 *
 * Hot Path Guarantee: Zero Dynamic Allocations in the Hot Path. Executes QDPS step filtering,
 * RIPA soft-bounded phase steering on Query registers, and updates pointer trail structures in O(1) time.
 *
 * @param context Active runtime context handle.
 * @param token_id Input token sequence ID.
 * @param query_tensor 32-byte aligned query tensor buffer to modulate in-place.
 * @param key_tensor 32-byte aligned key tensor buffer.
 * @param trail_out Pointer to receive updated 128-byte pointer trail status.
 * @return GCSO_SUCCESS or appropriate error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_step_token(
    gcso_context_handle_t context,
    uint32_t token_id,
    float* GCSO_RESTRICT query_tensor,
    float* GCSO_RESTRICT key_tensor,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_impl) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_context_impl*>(context);
        if (GCSO_UNLIKELY(!impl->is_initialized)) {
            return GCSO_ERROR_INVALID_STATE;
        }

        impl->active_token_id = token_id;
        impl->step_count++;

        if (key_tensor != nullptr) {
            if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(key_tensor) % 32 != 0)) {
                return GCSO_ERROR_MISALIGNED_POINTER;
            }
        }

        if (query_tensor != nullptr) {
            if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0)) {
                return GCSO_ERROR_MISALIGNED_POINTER;
            }

            const gcso_status_t qdps_status = gcso_qdps_filter_step(
                impl->accumulated_phase,
                impl->num_heads,
                impl->config.qdps_min_step_rad
            );
            if (GCSO_UNLIKELY(qdps_status != GCSO_SUCCESS)) {
                return qdps_status;
            }

            const gcso_status_t status = gcso_dpsr_apply_phase_steering_safe(
                query_tensor,
                impl->accumulated_phase,
                impl->head_dim,
                impl->num_heads,
                impl->config.ripa_clamp_max_rad
            );
            if (GCSO_UNLIKELY(status != GCSO_SUCCESS)) {
                return status;
            }
        }

        if (trail_out != nullptr) {
            if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(trail_out) % alignof(gcso_pointer_trail_t) != 0)) {
                return GCSO_ERROR_MISALIGNED_POINTER;
            }
            std::memset(trail_out, 0, sizeof(gcso_pointer_trail_t));
            trail_out->current_ptr = static_cast<uint64_t>(token_id);
            trail_out->prev_ptr = impl->last_ptr;
            trail_out->step_count = static_cast<uint32_t>(impl->step_count & 0xFFFFFFFFULL);
            trail_out->stigmergic_density = 1.0f;
            std::memcpy(trail_out->accumulated_phase_delta, impl->accumulated_phase, sizeof(impl->accumulated_phase));
        }

        impl->last_ptr = static_cast<uint64_t>(token_id);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Serializes runtime state into binary .gcso snapshot format.
 *
 * Synchronized with GCSO_ABI_VERSION_HEX (0x00000101) for ABI v0.1.1.
 *
 * @param context Active runtime context handle.
 * @param buffer Target byte buffer (pass NULL to query required size).
 * @param buffer_size Pointer to buffer capacity variable (in/out).
 * @return GCSO_SUCCESS or GCSO_ERROR_BUFFER_TOO_SMALL.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_serialize(
    gcso_context_handle_t context,
    uint8_t* GCSO_RESTRICT buffer,
    size_t* GCSO_RESTRICT buffer_size
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr || buffer_size == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_impl) != 0 ||
                      reinterpret_cast<uintptr_t>(buffer_size) % alignof(size_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        const size_t required_bytes = sizeof(gcso_snapshot_header_t) + sizeof(gcso_context_impl);
        if (buffer == nullptr) {
            *buffer_size = required_bytes;
            return GCSO_SUCCESS;
        }
        if (*buffer_size < required_bytes) {
            *buffer_size = required_bytes;
            return GCSO_ERROR_BUFFER_TOO_SMALL;
        }
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(buffer) % alignof(gcso_snapshot_header_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        auto* impl = reinterpret_cast<gcso_context_impl*>(context);
        std::memset(buffer, 0, required_bytes);

        auto* header = reinterpret_cast<gcso_snapshot_header_t*>(buffer);
        header->magic = 0x4F534347;             // ASCII "GCSO"
        header->version = GCSO_ABI_VERSION_HEX; // Unified ABI version 0.1.1 (0x00000101)
        header->total_size = static_cast<uint64_t>(required_bytes);

        const uint64_t payload_base = sizeof(gcso_snapshot_header_t);
        header->action_hub_offset = payload_base + offsetof(gcso_context_impl, last_ptr);
        header->attractor_field_offset = payload_base + offsetof(gcso_context_impl, prompt_anchor_count);
        header->dpsr_state_offset = payload_base + offsetof(gcso_context_impl, accumulated_phase);
        header->srl_state_offset = payload_base + offsetof(gcso_context_impl, config);
        header->edbc_state_offset = payload_base + offsetof(gcso_context_impl, step_count);
        header->daes_slot_offset = static_cast<uint32_t>(payload_base + offsetof(gcso_context_impl, active_token_id));
        header->timestamp_epoch_sec = 1774900000ULL;

        std::memcpy(buffer + sizeof(gcso_snapshot_header_t), impl, sizeof(gcso_context_impl));
        header->checksum_crc32 = compute_crc32(buffer + sizeof(gcso_snapshot_header_t), sizeof(gcso_context_impl));

        *buffer_size = required_bytes;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Restores runtime context state from deserialized binary .gcso snapshot.
 *
 * Validates container magic header (0x4F534347 "GCSO"), ABI version compatibility (0x00000101),
 * size boundaries, and CRC32 payload checksum before allocating state.
 *
 * @param buffer Memory-mapped or allocated input snapshot buffer.
 * @param buffer_size Byte size of input snapshot buffer.
 * @param context_out Pointer to receive restored context handle.
 * @return GCSO_SUCCESS or GCSO_ERROR_CONTAINER_CORRUPTED / GCSO_ERROR_VERSION_MISMATCH.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_deserialize(
    const uint8_t* GCSO_RESTRICT buffer,
    size_t buffer_size,
    gcso_context_handle_t* GCSO_RESTRICT context_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(buffer == nullptr || context_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context_out) % alignof(gcso_context_handle_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(buffer) % alignof(gcso_snapshot_header_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        if (buffer_size < sizeof(gcso_snapshot_header_t)) {
            return GCSO_ERROR_CONTAINER_CORRUPTED;
        }

        const auto* header = reinterpret_cast<const gcso_snapshot_header_t*>(buffer);
        if (header->magic != 0x4F534347) {
            return GCSO_ERROR_CONTAINER_CORRUPTED;
        }
        if (header->version != GCSO_ABI_VERSION_HEX) {
            return GCSO_ERROR_VERSION_MISMATCH;
        }

        const size_t expected_payload_size = sizeof(gcso_context_impl);
        if (buffer_size < sizeof(gcso_snapshot_header_t) + expected_payload_size ||
            header->total_size > buffer_size ||
            header->total_size < sizeof(gcso_snapshot_header_t) + expected_payload_size) {
            return GCSO_ERROR_CONTAINER_CORRUPTED;
        }

        const uint32_t payload_crc = compute_crc32(buffer + sizeof(gcso_snapshot_header_t), expected_payload_size);
        if (header->checksum_crc32 != 0 && header->checksum_crc32 != payload_crc) {
            return GCSO_ERROR_CONTAINER_CORRUPTED;
        }

        auto* impl = gcso_aligned_new<gcso_context_impl>();
        if (GCSO_UNLIKELY(impl == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }

        std::memcpy(impl, buffer + sizeof(gcso_snapshot_header_t), sizeof(gcso_context_impl));
        if (GCSO_UNLIKELY(impl->head_dim == 0 || (impl->head_dim % 2 != 0) ||
                          impl->num_heads == 0 || impl->num_heads > 64)) {
            gcso_aligned_delete(impl);
            return GCSO_ERROR_CONTAINER_CORRUPTED;
        }
        impl->is_initialized = true;
        impl->is_active = true;

        *context_out = reinterpret_cast<gcso_context_handle_t>(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Destroys a GCSO runtime context instance. Safe No-Op if NULL.
 *
 * @param context Runtime context handle to free (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_destroy(
    gcso_context_handle_t context
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_impl) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_context_impl*>(context);
        gcso_aligned_delete(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Destroys a container handle and releases mapped storage resources. Safe No-Op if NULL.
 *
 * @param container Container handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_container_destroy(
    gcso_container_handle_t container
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(container == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(container) % alignof(gcso_container_impl) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_container_impl*>(container);
        gcso_aligned_delete(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

GCSO_EXTERN_C_END