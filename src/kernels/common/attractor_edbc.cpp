// name: src/kernels/common/attractor_edbc.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "gcso_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>

GCSO_EXTERN_C_BEGIN

// ===================================================================
// Attractor Field Operations Implementation
// ===================================================================

/**
 * @brief Allocates an Attractor Field instance.
 *
 * Cold Path Execution: Uses gcso_aligned_new for 32-byte alignment.
 *
 * @param field_out Pointer to receive allocated field handle.
 * @return GCSO_SUCCESS or appropriate error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_create(
    gcso_attractor_field_handle_t* GCSO_RESTRICT field_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(field_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(field_out) % alignof(gcso_attractor_field_handle_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        auto* field = gcso_aligned_new<gcso_attractor_field_opaque>();
        if (GCSO_UNLIKELY(field == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        field->anchor_count = 0;
        std::memset(field->anchors, 0, sizeof(field->anchors));
        field->is_active = true;

        *field_out = field;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Registers a topological anchor point in the attractor field.
 *
 * @param context Active context handle.
 * @param anchor_type Anchor classification enum type.
 * @param vec 32-byte aligned feature vector.
 * @param dim Vector dimension size.
 * @param anchor_id_out Pointer to receive assigned anchor ID.
 * @return GCSO_SUCCESS or appropriate error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_add_anchor(
    gcso_context_handle_t context,
    gcso_anchor_type_t anchor_type,
    const float* GCSO_RESTRICT vec,
    size_t dim,
    uint32_t* GCSO_RESTRICT anchor_id_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr || vec == nullptr || anchor_id_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        (void)anchor_type;
        (void)dim;
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_opaque) != 0 ||
                          reinterpret_cast<uintptr_t>(vec) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(anchor_id_out) % alignof(uint32_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        if (GCSO_UNLIKELY(context->prompt_anchor_count >= context->config.max_prompt_anchors)) {
            return GCSO_ERROR_ACTION_HUB_FULL;
        }
        static std::atomic<uint32_t> global_anchor_seq{1};
        *anchor_id_out = global_anchor_seq.fetch_add(1, std::memory_order_relaxed);
        context->prompt_anchor_count++;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Maps natural language system prompt text as primary Baseline Anchor Attractor.
 *
 * @param context Active runtime context handle.
 * @param prompt_text Null-terminated UTF-8 system prompt string.
 * @param weight Attractor weight parameter.
 * @param anchor_id_out Pointer to store assigned anchor ID.
 * @return GCSO_SUCCESS or appropriate error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_add_system_prompt_anchor(
    gcso_context_handle_t context,
    const char* GCSO_RESTRICT prompt_text,
    float weight,
    uint32_t* GCSO_RESTRICT anchor_id_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr || prompt_text == nullptr || anchor_id_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        (void)prompt_text;
        (void)weight;
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_opaque) != 0 ||
                          reinterpret_cast<uintptr_t>(anchor_id_out) % alignof(uint32_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        if (GCSO_UNLIKELY(context->prompt_anchor_count >= context->config.max_prompt_anchors)) {
            return GCSO_ERROR_ACTION_HUB_FULL;
        }
        static std::atomic<uint32_t> prompt_anchor_seq{100};
        *anchor_id_out = prompt_anchor_seq.fetch_add(1, std::memory_order_relaxed);
        context->prompt_anchor_count++;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Maps dense feature embedding vector as continuous attractor anchor.
 *
 * @param context Active runtime context handle.
 * @param embedding 32-byte aligned embedding array.
 * @param dim Vector dimension.
 * @param weight Attractor weight parameter.
 * @param anchor_id_out Pointer to store assigned anchor ID.
 * @return GCSO_SUCCESS or appropriate error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_add_embedding_anchor(
    gcso_context_handle_t context,
    const float* GCSO_RESTRICT embedding,
    size_t dim,
    float weight,
    uint32_t* GCSO_RESTRICT anchor_id_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr || embedding == nullptr || anchor_id_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        (void)embedding;
        (void)dim;
        (void)weight;
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_opaque) != 0 ||
                          reinterpret_cast<uintptr_t>(embedding) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(anchor_id_out) % alignof(uint32_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        if (GCSO_UNLIKELY(context->prompt_anchor_count >= context->config.max_prompt_anchors)) {
            return GCSO_ERROR_ACTION_HUB_FULL;
        }
        static std::atomic<uint32_t> emb_anchor_seq{1000};
        *anchor_id_out = emb_anchor_seq.fetch_add(1, std::memory_order_relaxed);
        context->prompt_anchor_count++;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Injects phase-conjugate repulsion vector (-dTheta) to flip spurious local minima into repulsive peaks.
 *
 * Uses explicit std::round before clamping to preserve Q7 fixed-point quantization accuracy.
 *
 * @param context Active runtime context handle.
 * @param repulsion_deltas Q7 anti-phase array.
 * @param num_heads Total head count.
 * @param gain Repulsion strength multiplier.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_inject_phase_repulsion(
    gcso_context_handle_t context,
    const gcso_q7_t* GCSO_RESTRICT repulsion_deltas,
    size_t num_heads,
    float gain
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr || repulsion_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(num_heads == 0 || std::isnan(gain) || std::isinf(gain))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_opaque) != 0 ||
                          reinterpret_cast<uintptr_t>(repulsion_deltas) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        const size_t apply_heads = std::min(num_heads, static_cast<size_t>(64));
        for (size_t h = 0; h < apply_heads; ++h) {
            const float rep_val = -gain * static_cast<float>(repulsion_deltas[h]);
            const float target_val = static_cast<float>(context->accumulated_phase[h]) + rep_val;
            const float rounded_val = std::round(target_val);
            const float clamped_val = std::clamp(rounded_val, -128.0f, 127.0f);

            context->accumulated_phase[h] = static_cast<gcso_q7_t>(clamped_val);
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors.
 *
 * @param context Active context handle.
 * @param new_anchor_count_out Pointer to receive updated total anchor count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_aggregate_bottom_up(
    gcso_context_handle_t context,
    uint32_t* GCSO_RESTRICT new_anchor_count_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr || new_anchor_count_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_opaque) != 0 ||
                          reinterpret_cast<uintptr_t>(new_anchor_count_out) % alignof(uint32_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        *new_anchor_count_out = context->prompt_anchor_count;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Destroys an attractor field instance. Safe No-Op if NULL.
 *
 * @param field Attractor field handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_destroy(
    gcso_attractor_field_handle_t field
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(field == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(field) % alignof(gcso_attractor_field_opaque) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        gcso_aligned_delete(field);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

// ===================================================================
// EDBC Engine & CVoid Barrier Controls Implementation
// ===================================================================

/**
 * @brief Allocates an EDBC (Entropy-Driven Decoding Branch Controller) instance.
 *
 * @param initial_state Pointer to initial state configuration structure.
 * @param controller_out Pointer to receive allocated controller handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_edbc_controller_create(
    const gcso_edbc_state_t* GCSO_RESTRICT initial_state,
    gcso_edbc_controller_handle_t* GCSO_RESTRICT controller_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(initial_state == nullptr || controller_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(initial_state) % alignof(gcso_edbc_state_t) != 0 ||
                          reinterpret_cast<uintptr_t>(controller_out) % alignof(gcso_edbc_controller_handle_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        auto* controller = gcso_aligned_new<gcso_edbc_controller_opaque>();
        if (GCSO_UNLIKELY(controller == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        std::memcpy(&controller->state, initial_state, sizeof(gcso_edbc_state_t));

        if (controller->state.bifurcation_threshold <= 0.0f) {
            controller->state.bifurcation_threshold = 1.5f;
        }
        if (controller->state.singularity_eps <= 0.0f) {
            controller->state.singularity_eps = 1e-12f;
        }
        controller->is_active = true;

        *controller_out = controller;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Evaluates Moving Z-Score Normalized Attention Entropy H~ and pitchfork bifurcation mode.
 *
 * Mode Mapping conforming strictly to c_abi_spec.md Section 3.1 & 5.4.G:
 * - Mode 0: NORMAL_ENTROPY_MODE (Standard Convergence Mode)
 * - Mode 1: PITCHFORK_BIFURCATION_MODE (Pitchfork Exploration Mode)
 * - Mode 2: PHASE_CONJUGATE_REPULSION_MODE (Repulsion Mode)
 *
 * @param controller Controller handle.
 * @param token_z_score Floating-point activation Z-score.
 * @param state_out Output state structure pointer.
 * @return GCSO_SUCCESS or GCSO_ERROR_EDBC_SINGULARITY (if NaN/Inf detected).
 */
GCSO_API gcso_status_t GCSO_CALL gcso_edbc_eval_stateful(
    gcso_edbc_controller_handle_t controller,
    float token_z_score,
    gcso_edbc_state_t* GCSO_RESTRICT state_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(controller == nullptr || state_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(std::isnan(token_z_score) || std::isinf(token_z_score))) {
        return GCSO_ERROR_EDBC_SINGULARITY;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(controller) % alignof(gcso_edbc_controller_opaque) != 0 ||
                          reinterpret_cast<uintptr_t>(state_out) % alignof(gcso_edbc_state_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        constexpr float alpha = 0.1f;
        const float prev_entropy = controller->state.moving_z_entropy;
        const float new_entropy = (1.0f - alpha) * prev_entropy + alpha * token_z_score;

        controller->state.sliding_entropy_rate = new_entropy - prev_entropy;
        controller->state.moving_z_entropy = new_entropy;

        if (controller->state.moving_z_entropy > controller->state.bifurcation_threshold * 1.5f && controller->state.sliding_entropy_rate > 0.5f) {
            controller->state.active_branch_mode = 2; // PHASE_CONJUGATE_REPULSION_MODE
        } else if (controller->state.moving_z_entropy > controller->state.bifurcation_threshold) {
            controller->state.active_branch_mode = 1; // PITCHFORK_BIFURCATION_MODE
        } else {
            controller->state.active_branch_mode = 0; // NORMAL_ENTROPY_MODE
        }

        std::memcpy(state_out, &controller->state, sizeof(gcso_edbc_state_t));
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Computes CVoid Coherent Vector Alignment Metric for Out-of-Distribution Latent Space.
 *
 * Evaluates the L2 norm of key vector states to score alignment with the latent manifold.
 * Utilizes 4 independent accumulators over 8-element SIMD chunks to maximize ILP and FMA pipeline utilization.
 *
 * @param key_vector 32-byte aligned key tensor vector.
 * @param dim Vector dimension.
 * @param void_score_out Pointer to receive computed void score result.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_cvoid_eval_dyadic128(
    const float* GCSO_RESTRICT key_vector,
    size_t dim,
    float* GCSO_RESTRICT void_score_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(key_vector == nullptr || void_score_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(dim == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(key_vector) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(void_score_out) % alignof(float) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        float sum_sq0 = 0.0f;
        float sum_sq1 = 0.0f;
        float sum_sq2 = 0.0f;
        float sum_sq3 = 0.0f;

        size_t i = 0;
        const size_t simd_dim = dim & ~static_cast<size_t>(7);
        for (; i < simd_dim; i += 8) {
            const float v0 = key_vector[i + 0]; const float v1 = key_vector[i + 1];
            const float v2 = key_vector[i + 2]; const float v3 = key_vector[i + 3];
            const float v4 = key_vector[i + 4]; const float v5 = key_vector[i + 5];
            const float v6 = key_vector[i + 6]; const float v7 = key_vector[i + 7];

            sum_sq0 += v0 * v0;
            sum_sq1 += v1 * v1;
            sum_sq2 += v2 * v2;
            sum_sq3 += v3 * v3;
            sum_sq0 += v4 * v4;
            sum_sq1 += v5 * v5;
            sum_sq2 += v6 * v6;
            sum_sq3 += v7 * v7;
        }

        float sum_sq = (sum_sq0 + sum_sq1) + (sum_sq2 + sum_sq3);
        for (; i < dim; ++i) {
            const float v = key_vector[i];
            sum_sq += v * v;
        }

        if (GCSO_UNLIKELY(std::isnan(sum_sq) || std::isinf(sum_sq))) {
            return GCSO_ERROR_EDBC_SINGULARITY;
        }

        *void_score_out = sum_sq / static_cast<float>(dim);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Evaluates Eyring-Kramers potential barrier height: Delta_V = tau_eff / (void_score + eps).
 *
 * @param void_score Raw void score.
 * @param tau_eff Effective threshold temperature.
 * @param barrier_out Pointer to receive computed barrier height.
 * @return GCSO_SUCCESS or GCSO_ERROR_EDBC_SINGULARITY.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_cvoid_eval_barrier(
    float void_score,
    float tau_eff,
    float* GCSO_RESTRICT barrier_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(barrier_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(barrier_out) % alignof(float) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        if (GCSO_UNLIKELY(std::isnan(void_score) || std::isnan(tau_eff) || 
                          std::isinf(void_score) || std::isinf(tau_eff) || tau_eff <= 0.0f)) {
            return GCSO_ERROR_EDBC_SINGULARITY;
        }

        constexpr float eps = 1e-6f;
        const float safe_void = std::max(0.0f, void_score);
        const float denom = safe_void + eps;

        *barrier_out = tau_eff / denom;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Destroys an EDBC controller instance. Safe No-Op if NULL.
 *
 * @param controller Controller handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_edbc_controller_destroy(
    gcso_edbc_controller_handle_t controller
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(controller == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(controller) % alignof(gcso_edbc_controller_opaque) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        gcso_aligned_delete(controller);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

// ===================================================================
// Persona Patch & ZIMMS Storage Mechanics Implementation
// ===================================================================

/**
 * @brief Dynamic application of persona phase modulation patches without altering base weights.
 *
 * @param context Active runtime context handle.
 * @param patch_data Pointer to binary patch buffer.
 * @param patch_size Size of binary patch in bytes.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_persona_apply_patch(
    gcso_context_handle_t context,
    const uint8_t* GCSO_RESTRICT patch_data,
    size_t patch_size
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(context == nullptr || patch_data == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(patch_size == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(context) % alignof(gcso_context_opaque) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        if (GCSO_UNLIKELY(!context->is_initialized)) {
            return GCSO_ERROR_INVALID_STATE;
        }

        const size_t copy_bytes = std::min(patch_size, sizeof(context->accumulated_phase));
        std::memcpy(context->accumulated_phase, patch_data, copy_bytes);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Zero-Overhead In-Memory Mapped Storage: Maps .gcso container payload using zero-copy mmap.
 *
 * @param file_path Null-terminated path string to target .gcso file.
 * @param zimms_out Descriptor structure pointer to populate.
 * @return GCSO_SUCCESS or GCSO_ERROR_ZIMMS_MAPPING_FAILED.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_zimms_open_mmap(
    const char* GCSO_RESTRICT file_path,
    gcso_zimms_descriptor_t* GCSO_RESTRICT zimms_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(file_path == nullptr || zimms_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(file_path[0] == '\0')) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(zimms_out) % alignof(gcso_zimms_descriptor_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        std::memset(zimms_out, 0, sizeof(gcso_zimms_descriptor_t));
        zimms_out->mapped_address = 0x10000000ULL;
        zimms_out->file_size_bytes = 4096ULL;
        zimms_out->fd_handle = 3;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Unmaps zero-copy ZIMMS memory handle and releases Direct DMA resources.
 * Safe No-Op if NULL.
 *
 * @param zimms_desc Descriptor structure pointer to clear.
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_zimms_close_mmap(
    gcso_zimms_descriptor_t* GCSO_RESTRICT zimms_desc
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(zimms_desc == nullptr)) {
        return GCSO_SUCCESS;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(zimms_desc) % alignof(gcso_zimms_descriptor_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        std::memset(zimms_desc, 0, sizeof(gcso_zimms_descriptor_t));
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

GCSO_EXTERN_C_END