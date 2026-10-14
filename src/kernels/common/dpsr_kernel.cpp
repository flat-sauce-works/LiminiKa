// name: src/kernels/common/dpsr_kernel.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "gcso_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>

GCSO_EXTERN_C_BEGIN

/**
 * @brief Creates a DPSR phase steering kernel instance using strongly-typed opaque handles.
 *
 * Cold Path Execution: Allocates memory aligned to 16 bytes. Validates that head_dim is even
 * for strict compliance with 2D SO(2) planar decomposition.
 *
 * @param head_dim Attention head dimension (must be even for 2D SO(2) decomposition).
 * @param num_heads Total number of attention heads.
 * @param kernel_out Pointer to receive created kernel handle.
 * @return GCSO_SUCCESS or appropriate error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_kernel_create(
    uint32_t head_dim,
    uint32_t num_heads,
    gcso_dpsr_kernel_handle_t* GCSO_RESTRICT kernel_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(kernel_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(kernel_out) % alignof(gcso_dpsr_kernel_handle_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        auto* kernel = gcso_aligned_new<gcso_dpsr_kernel_opaque>();
        if (GCSO_UNLIKELY(kernel == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        kernel->head_dim = head_dim;
        kernel->num_heads = num_heads;
        kernel->is_active = true;

        *kernel_out = kernel;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Applies inline DPSR phase rotation to Query tensor registers.
 *
 * Hot Path Guarantee: Zero dynamic memory allocations. Unrolls loops over 4-pair (8-float) chunks
 * to maximize Instruction-Level Parallelism (ILP) and FMA pipeline utilization.
 *
 * Lie Group Invariant: Abides strictly by SO(2)^(d_head/2) commutativity with standard RoPE.
 *
 * @param query_tensor 32-byte aligned query tensor buffer to modulate in-place.
 * @param phase_deltas Q7 quantized phase delta array per head (scale beta_Q7 = pi / 128).
 * @param head_dim Attention head dimension size (must be even).
 * @param num_heads Total head count.
 * @return GCSO_SUCCESS or appropriate error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering(
    float* GCSO_RESTRICT query_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(phase_deltas) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        constexpr float q7_scale = 3.14159265358979323846f / 128.0f;
        const size_t pairs_per_head = head_dim / 2;

        for (size_t h = 0; h < num_heads; ++h) {
            const float theta = static_cast<float>(phase_deltas[h]) * q7_scale;
            if (theta == 0.0f) {
                continue;
            }
            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);
            float* GCSO_RESTRICT head_q = query_tensor + (h * head_dim);

            // Unroll 4 pairs (8 floats) per iteration for maximum vector pipeline utilization
            size_t k = 0;
            const size_t simd_pairs = pairs_per_head & ~static_cast<size_t>(3);
            for (; k < simd_pairs; k += 4) {
                const float q0 = head_q[2 * k + 0]; const float q1 = head_q[2 * k + 1];
                const float q2 = head_q[2 * k + 2]; const float q3 = head_q[2 * k + 3];
                const float q4 = head_q[2 * k + 4]; const float q5 = head_q[2 * k + 5];
                const float q6 = head_q[2 * k + 6]; const float q7 = head_q[2 * k + 7];

                head_q[2 * k + 0] = q0 * cos_t - q1 * sin_t;
                head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
                head_q[2 * k + 2] = q2 * cos_t - q3 * sin_t;
                head_q[2 * k + 3] = q2 * sin_t + q3 * cos_t;
                head_q[2 * k + 4] = q4 * cos_t - q5 * sin_t;
                head_q[2 * k + 5] = q4 * sin_t + q5 * cos_t;
                head_q[2 * k + 6] = q6 * cos_t - q7 * sin_t;
                head_q[2 * k + 7] = q6 * sin_t + q7 * cos_t;
            }

            for (; k < pairs_per_head; ++k) {
                const float q0 = head_q[2 * k + 0];
                const float q1 = head_q[2 * k + 1];
                head_q[2 * k + 0] = q0 * cos_t - q1 * sin_t;
                head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
            }
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Applies RIPA soft-bounded tanh clamping restricted strictly to low-frequency channels.
 *
 * Mathematical & Physical Rationale:
 * In standard RoPE encoding, rotation frequencies follow theta_k = 10000^(-2k/d). Higher pair indices
 * k correspond to lower frequencies maintaining core context structure.
 * RIPA restricts phase modulation strictly to the upper d_head / 4 dimensions (the top d_head / 8 pairs,
 * starting from start_k = pairs_per_head - low_freq_pairs up to pairs_per_head).
 *
 * Soft Clamping Formula: theta_safe = max_rad * tanh(theta / max_rad)
 *
 * @param query_tensor 32-byte aligned query tensor buffer.
 * @param phase_deltas Q7 quantized phase delta array.
 * @param head_dim Head dimension size (must be even).
 * @param num_heads Total head count.
 * @param max_rad Maximum allowed phase limit in radians (e.g. 0.087 rad approx 5 degrees).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_safe(
    float* GCSO_RESTRICT query_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads,
    float max_rad
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0 || std::isnan(max_rad) || std::isinf(max_rad))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(phase_deltas) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        constexpr float q7_scale = 3.14159265358979323846f / 128.0f;
        const size_t pairs_per_head = head_dim / 2;

        const size_t low_freq_pairs = std::max(static_cast<size_t>(1), head_dim / 8);
        const size_t start_k = (pairs_per_head > low_freq_pairs) ? (pairs_per_head - low_freq_pairs) : 0;

        for (size_t h = 0; h < num_heads; ++h) {
            float theta = static_cast<float>(phase_deltas[h]) * q7_scale;
            if (max_rad > 0.0f) {
                theta = max_rad * std::tanh(theta / max_rad);
            }

            if (theta == 0.0f) {
                continue;
            }

            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);
            float* GCSO_RESTRICT head_q = query_tensor + (h * head_dim);

            size_t k = start_k;
            const size_t remaining_pairs = pairs_per_head - start_k;
            const size_t simd_pairs = start_k + (remaining_pairs & ~static_cast<size_t>(3));

            for (; k < simd_pairs; k += 4) {
                const float q0 = head_q[2 * k + 0]; const float q1 = head_q[2 * k + 1];
                const float q2 = head_q[2 * k + 2]; const float q3 = head_q[2 * k + 3];
                const float q4 = head_q[2 * k + 4]; const float q5 = head_q[2 * k + 5];
                const float q6 = head_q[2 * k + 6]; const float q7 = head_q[2 * k + 7];

                head_q[2 * k + 0] = q0 * cos_t - q1 * sin_t;
                head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
                head_q[2 * k + 2] = q2 * cos_t - q3 * sin_t;
                head_q[2 * k + 3] = q2 * sin_t + q3 * cos_t;
                head_q[2 * k + 4] = q4 * cos_t - q5 * sin_t;
                head_q[2 * k + 5] = q4 * sin_t + q5 * cos_t;
                head_q[2 * k + 6] = q6 * cos_t - q7 * sin_t;
                head_q[2 * k + 7] = q6 * sin_t + q7 * cos_t;
            }

            for (; k < pairs_per_head; ++k) {
                const float q0 = head_q[2 * k + 0];
                const float q1 = head_q[2 * k + 1];
                head_q[2 * k + 0] = q0 * cos_t - q1 * sin_t;
                head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
            }
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief QDPS discrete filter: cuts off phase rotation steps falling below min_step_rad.
 *
 * Pre-evaluates integer Q7 thresholds to execute zero-overhead integer domain comparisons in the hot loop.
 *
 * @param phase_deltas Q7 phase array to filter in-place.
 * @param len Array element count.
 * @param min_step_rad Minimum step threshold angle in radians.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_qdps_filter_step(
    gcso_q7_t* GCSO_RESTRICT phase_deltas,
    size_t len,
    float min_step_rad
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(len == 0 || std::isnan(min_step_rad) || std::isinf(min_step_rad) || min_step_rad < 0.0f)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(phase_deltas) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        constexpr float q7_scale = 3.14159265358979323846f / 128.0f;
        const int32_t min_step_q7 = static_cast<int32_t>(std::ceil(min_step_rad / q7_scale));

        if (min_step_q7 <= 0) {
            return GCSO_SUCCESS;
        }
        if (min_step_q7 > 127) {
            std::memset(phase_deltas, 0, len * sizeof(gcso_q7_t));
            return GCSO_SUCCESS;
        }

        for (size_t i = 0; i < len; ++i) {
            if (std::abs(static_cast<int32_t>(phase_deltas[i])) < min_step_q7) {
                phase_deltas[i] = 0;
            }
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Lazy Phase Unwrapping: Applies relative SO(2) phase rotations against context accumulator on Query side.
 *
 * Corrected Geometrical Behavior: Executes planar 2D SO(2) rotations based on context phase accumulation
 * instead of linear vector additions, maintaining SO(2)^(d_head/2) Lie Group invariants per c_abi_spec.md.
 *
 * @param query_tensor 32-byte aligned target query tensor.
 * @param context_accum Cumulative context phase angle vector per head in radians.
 * @param head_dim Head dimension size (must be even).
 * @param num_heads Total head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_lazy_unwrap_override(
    float* GCSO_RESTRICT query_tensor,
    const float* GCSO_RESTRICT context_accum,
    size_t head_dim,
    size_t num_heads
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || context_accum == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(context_accum) % 32 != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        const size_t pairs_per_head = head_dim / 2;
        for (size_t h = 0; h < num_heads; ++h) {
            const float theta = context_accum[h];
            if (theta == 0.0f) {
                continue;
            }
            const float cos_t = std::cos(theta);
            const float sin_t = std::sin(theta);
            float* GCSO_RESTRICT head_q = query_tensor + (h * head_dim);

            size_t k = 0;
            const size_t simd_pairs = pairs_per_head & ~static_cast<size_t>(3);
            for (; k < simd_pairs; k += 4) {
                const float q0 = head_q[2 * k + 0]; const float q1 = head_q[2 * k + 1];
                const float q2 = head_q[2 * k + 2]; const float q3 = head_q[2 * k + 3];
                const float q4 = head_q[2 * k + 4]; const float q5 = head_q[2 * k + 5];
                const float q6 = head_q[2 * k + 6]; const float q7 = head_q[2 * k + 7];

                head_q[2 * k + 0] = q0 * cos_t - q1 * sin_t;
                head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
                head_q[2 * k + 2] = q2 * cos_t - q3 * sin_t;
                head_q[2 * k + 3] = q2 * sin_t + q3 * cos_t;
                head_q[2 * k + 4] = q4 * cos_t - q5 * sin_t;
                head_q[2 * k + 5] = q4 * sin_t + q5 * cos_t;
                head_q[2 * k + 6] = q6 * cos_t - q7 * sin_t;
                head_q[2 * k + 7] = q6 * sin_t + q7 * cos_t;
            }

            for (; k < pairs_per_head; ++k) {
                const float q0 = head_q[2 * k + 0];
                const float q1 = head_q[2 * k + 1];
                head_q[2 * k + 0] = q0 * cos_t - q1 * sin_t;
                head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
            }
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Executes norm-guarded Slerp phase stabilization on state vectors.
 *
 * Utilizes 4 independent accumulators to unroll L2 norm computation, maximizing instruction-level parallelism.
 *
 * @param tensor Vector tensor buffer to normalize in-place.
 * @param dim Total vector length.
 * @param norm_lower Minimum allowed L2 norm bound.
 * @param norm_upper Maximum allowed L2 norm bound.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_slerp_norm_guard_stable(
    float* GCSO_RESTRICT tensor,
    size_t dim,
    float norm_lower,
    float norm_upper
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(tensor == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(dim == 0 || norm_lower < 0.0f || norm_upper < norm_lower)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(tensor) % 32 != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        float sum_sq0 = 0.0f;
        float sum_sq1 = 0.0f;
        float sum_sq2 = 0.0f;
        float sum_sq3 = 0.0f;

        size_t i = 0;
        const size_t simd_len = dim & ~static_cast<size_t>(7);
        for (; i < simd_len; i += 8) {
            sum_sq0 += tensor[i + 0] * tensor[i + 0];
            sum_sq1 += tensor[i + 1] * tensor[i + 1];
            sum_sq2 += tensor[i + 2] * tensor[i + 2];
            sum_sq3 += tensor[i + 3] * tensor[i + 3];
            sum_sq0 += tensor[i + 4] * tensor[i + 4];
            sum_sq1 += tensor[i + 5] * tensor[i + 5];
            sum_sq2 += tensor[i + 6] * tensor[i + 6];
            sum_sq3 += tensor[i + 7] * tensor[i + 7];
        }

        float norm_sq = (sum_sq0 + sum_sq1) + (sum_sq2 + sum_sq3);
        for (; i < dim; ++i) {
            norm_sq += tensor[i] * tensor[i];
        }

        const float norm = std::sqrt(norm_sq);
        if (GCSO_UNLIKELY(std::isnan(norm) || std::isinf(norm))) {
            return GCSO_ERROR_EDBC_SINGULARITY;
        }

        if (norm < 1e-12f) {
            return GCSO_SUCCESS;
        }

        float scale = 1.0f;
        if (norm < norm_lower) {
            scale = norm_lower / norm;
        } else if (norm > norm_upper) {
            scale = norm_upper / norm;
        }

        if (std::abs(scale - 1.0f) > 1e-6f) {
            size_t j = 0;
            for (; j < simd_len; j += 8) {
                tensor[j + 0] *= scale;
                tensor[j + 1] *= scale;
                tensor[j + 2] *= scale;
                tensor[j + 3] *= scale;
                tensor[j + 4] *= scale;
                tensor[j + 5] *= scale;
                tensor[j + 6] *= scale;
                tensor[j + 7] *= scale;
            }
            for (; j < dim; ++j) {
                tensor[j] *= scale;
            }
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Fused inline logit phase shift prior to LM Head Softmax.
 *
 * @param logits Logit score vector buffer.
 * @param vocab_size Vocabulary dimension.
 * @param phase_deltas Q7 phase deltas.
 * @param num_heads Head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_fused_logit_shift(
    float* GCSO_RESTRICT logits,
    size_t vocab_size,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas,
    size_t num_heads
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(logits == nullptr || phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(num_heads == 0 || vocab_size == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(logits) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(phase_deltas) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        constexpr float q7_scale = 3.14159265358979323846f / 128.0f;
        float aggregate_phase_bias = 0.0f;
        for (size_t h = 0; h < num_heads; ++h) {
            aggregate_phase_bias += static_cast<float>(phase_deltas[h]) * q7_scale;
        }
        const float mean_bias = aggregate_phase_bias / static_cast<float>(num_heads);

        size_t v = 0;
        const size_t simd_vocab = vocab_size & ~static_cast<size_t>(7);
        for (; v < simd_vocab; v += 8) {
            logits[v + 0] += mean_bias;
            logits[v + 1] += mean_bias;
            logits[v + 2] += mean_bias;
            logits[v + 3] += mean_bias;
            logits[v + 4] += mean_bias;
            logits[v + 5] += mean_bias;
            logits[v + 6] += mean_bias;
            logits[v + 7] += mean_bias;
        }
        for (; v < vocab_size; ++v) {
            logits[v] += mean_bias;
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Destroys a DPSR kernel instance. Safe No-Op if NULL.
 *
 * @param kernel Kernel handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_kernel_destroy(
    gcso_dpsr_kernel_handle_t kernel
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(kernel == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(kernel) % alignof(gcso_dpsr_kernel_opaque) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        gcso_aligned_delete(kernel);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Allocates a PSPM router instance.
 *
 * @param config PSPM router configuration pointer.
 * @param router_out Pointer to receive allocated router handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_pspm_router_create(
    const gcso_pspm_config_t* GCSO_RESTRICT config,
    gcso_pspm_router_handle_t* GCSO_RESTRICT router_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(config == nullptr || router_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(config) % alignof(gcso_pspm_config_t) != 0 ||
                          reinterpret_cast<uintptr_t>(router_out) % alignof(gcso_pspm_router_handle_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        auto* router = gcso_aligned_new<gcso_pspm_router_opaque>();
        if (GCSO_UNLIKELY(router == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        std::memcpy(&router->config, config, sizeof(gcso_pspm_config_t));
        router->is_active = true;

        *router_out = router;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Dispatches PSPM head-group phase profiles in a single pass.
 *
 * Single-Pass Head Group Ensembling: Splits attention heads into Fact, Logic, and Explore sub-groups,
 * applying unique phase profile gains in a single forward pass without parameter duplication.
 *
 * @param query_tensor 32-byte aligned query tensor buffer.
 * @param pspm_cfg PSPM routing configuration.
 * @param head_dim Head dimension.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_pspm_dispatch_single_pass(
    float* GCSO_RESTRICT query_tensor,
    const gcso_pspm_config_t* GCSO_RESTRICT pspm_cfg,
    size_t head_dim
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || pspm_cfg == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(pspm_cfg) % alignof(gcso_pspm_config_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        const size_t fact_heads = static_cast<size_t>(pspm_cfg->num_fact_heads);
        const size_t logic_heads = static_cast<size_t>(pspm_cfg->num_logic_heads);
        const size_t explore_heads = static_cast<size_t>(pspm_cfg->num_explore_heads);

        const size_t fact_elems = fact_heads * head_dim;
        const size_t logic_elems = logic_heads * head_dim;
        const size_t explore_elems = explore_heads * head_dim;

        size_t i = 0;
        const size_t fact_simd = fact_elems & ~static_cast<size_t>(7);
        for (; i < fact_simd; i += 8) {
            query_tensor[i + 0] *= pspm_cfg->fact_phase_gain;
            query_tensor[i + 1] *= pspm_cfg->fact_phase_gain;
            query_tensor[i + 2] *= pspm_cfg->fact_phase_gain;
            query_tensor[i + 3] *= pspm_cfg->fact_phase_gain;
            query_tensor[i + 4] *= pspm_cfg->fact_phase_gain;
            query_tensor[i + 5] *= pspm_cfg->fact_phase_gain;
            query_tensor[i + 6] *= pspm_cfg->fact_phase_gain;
            query_tensor[i + 7] *= pspm_cfg->fact_phase_gain;
        }
        for (; i < fact_elems; ++i) {
            query_tensor[i] *= pspm_cfg->fact_phase_gain;
        }

        float* GCSO_RESTRICT logic_q = query_tensor + fact_elems;
        size_t j = 0;
        const size_t logic_simd = logic_elems & ~static_cast<size_t>(7);
        for (; j < logic_simd; j += 8) {
            logic_q[j + 0] *= pspm_cfg->logic_phase_gain;
            logic_q[j + 1] *= pspm_cfg->logic_phase_gain;
            logic_q[j + 2] *= pspm_cfg->logic_phase_gain;
            logic_q[j + 3] *= pspm_cfg->logic_phase_gain;
            logic_q[j + 4] *= pspm_cfg->logic_phase_gain;
            logic_q[j + 5] *= pspm_cfg->logic_phase_gain;
            logic_q[j + 6] *= pspm_cfg->logic_phase_gain;
            logic_q[j + 7] *= pspm_cfg->logic_phase_gain;
        }
        for (; j < logic_elems; ++j) {
            logic_q[j] *= pspm_cfg->logic_phase_gain;
        }

        float* GCSO_RESTRICT explore_q = query_tensor + fact_elems + logic_elems;
        size_t k = 0;
        const size_t explore_simd = explore_elems & ~static_cast<size_t>(7);
        for (; k < explore_simd; k += 8) {
            explore_q[k + 0] *= pspm_cfg->explore_phase_gain;
            explore_q[k + 1] *= pspm_cfg->explore_phase_gain;
            explore_q[k + 2] *= pspm_cfg->explore_phase_gain;
            explore_q[k + 3] *= pspm_cfg->explore_phase_gain;
            explore_q[k + 4] *= pspm_cfg->explore_phase_gain;
            explore_q[k + 5] *= pspm_cfg->explore_phase_gain;
            explore_q[k + 6] *= pspm_cfg->explore_phase_gain;
            explore_q[k + 7] *= pspm_cfg->explore_phase_gain;
        }
        for (; k < explore_elems; ++k) {
            explore_q[k] *= pspm_cfg->explore_phase_gain;
        }

        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Destroys a PSPM router instance. Safe No-Op if NULL.
 *
 * @param router Router handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_pspm_router_destroy(
    gcso_pspm_router_handle_t router
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(router == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(router) % alignof(gcso_pspm_router_opaque) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        gcso_aligned_delete(router);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Creates a Sparse Residual Adapter Layer (SRL) instance.
 *
 * @param descriptor Pointer to SRL descriptor.
 * @param adapter_out Pointer to receive allocated adapter handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_srl_adapter_create(
    const gcso_srl_descriptor_t* GCSO_RESTRICT descriptor,
    gcso_srl_adapter_handle_t* GCSO_RESTRICT adapter_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(descriptor == nullptr || adapter_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(descriptor) % alignof(gcso_srl_descriptor_t) != 0 ||
                          reinterpret_cast<uintptr_t>(adapter_out) % alignof(gcso_srl_adapter_handle_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        auto* adapter = gcso_aligned_new<gcso_srl_adapter_opaque>();
        if (GCSO_UNLIKELY(adapter == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        std::memcpy(&adapter->descriptor, descriptor, sizeof(gcso_srl_descriptor_t));
        adapter->is_active = true;

        *adapter_out = adapter;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Evaluates SRL Dynamic Rank-1 outer product. Executed in zero-allocation mode.
 * Formula: y = W_base * x + s (*) (u * (v^T * x)).
 *
 * @param y_out Output vector buffer to accumulate into in-place.
 * @param x_in Input vector buffer.
 * @param srl_desc SRL descriptor structure pointer.
 * @param dim_in Input vector dimension.
 * @param dim_out Output vector dimension.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_srl_eval_rank1(
    float* GCSO_RESTRICT y_out,
    const float* GCSO_RESTRICT x_in,
    const gcso_srl_descriptor_t* GCSO_RESTRICT srl_desc,
    size_t dim_in,
    size_t dim_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(y_out == nullptr || x_in == nullptr || srl_desc == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(dim_in == 0 || dim_out == 0 || std::isnan(srl_desc->scale_factor) || std::isinf(srl_desc->scale_factor))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(y_out) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(x_in) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(srl_desc) % alignof(gcso_srl_descriptor_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        const auto* GCSO_RESTRICT vector_u = reinterpret_cast<const float*>(static_cast<uintptr_t>(srl_desc->u_vector_ptr));
        const auto* GCSO_RESTRICT vector_v = reinterpret_cast<const float*>(static_cast<uintptr_t>(srl_desc->v_vector_ptr));
        const auto* GCSO_RESTRICT gain_s   = reinterpret_cast<const float*>(static_cast<uintptr_t>(srl_desc->gain_scalar_ptr));

        if (GCSO_UNLIKELY(vector_u == nullptr || vector_v == nullptr)) {
            return GCSO_ERROR_NULL_POINTER;
        }
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(vector_u) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(vector_v) % 32 != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        if (gain_s != nullptr) {
            if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(gain_s) % 32 != 0)) {
                return GCSO_ERROR_MISALIGNED_POINTER;
            }
        }

        float v_dot_x0 = 0.0f;
        float v_dot_x1 = 0.0f;
        float v_dot_x2 = 0.0f;
        float v_dot_x3 = 0.0f;

        size_t i = 0;
        const size_t simd_in = dim_in & ~static_cast<size_t>(7);
        for (; i < simd_in; i += 8) {
            v_dot_x0 += vector_v[i + 0] * x_in[i + 0];
            v_dot_x1 += vector_v[i + 1] * x_in[i + 1];
            v_dot_x2 += vector_v[i + 2] * x_in[i + 2];
            v_dot_x3 += vector_v[i + 3] * x_in[i + 3];
            v_dot_x0 += vector_v[i + 4] * x_in[i + 4];
            v_dot_x1 += vector_v[i + 5] * x_in[i + 5];
            v_dot_x2 += vector_v[i + 6] * x_in[i + 6];
            v_dot_x3 += vector_v[i + 7] * x_in[i + 7];
        }

        float v_dot_x = (v_dot_x0 + v_dot_x1) + (v_dot_x2 + v_dot_x3);
        for (; i < dim_in; ++i) {
            v_dot_x += vector_v[i] * x_in[i];
        }

        const float scaled_dot = srl_desc->scale_factor * v_dot_x;
        size_t j = 0;
        const size_t simd_out = dim_out & ~static_cast<size_t>(7);

        if (gain_s != nullptr) {
            for (; j < simd_out; j += 8) {
                y_out[j + 0] += gain_s[j + 0] * scaled_dot * vector_u[j + 0];
                y_out[j + 1] += gain_s[j + 1] * scaled_dot * vector_u[j + 1];
                y_out[j + 2] += gain_s[j + 2] * scaled_dot * vector_u[j + 2];
                y_out[j + 3] += gain_s[j + 3] * scaled_dot * vector_u[j + 3];
                y_out[j + 4] += gain_s[j + 4] * scaled_dot * vector_u[j + 4];
                y_out[j + 5] += gain_s[j + 5] * scaled_dot * vector_u[j + 5];
                y_out[j + 6] += gain_s[j + 6] * scaled_dot * vector_u[j + 6];
                y_out[j + 7] += gain_s[j + 7] * scaled_dot * vector_u[j + 7];
            }
            for (; j < dim_out; ++j) {
                y_out[j] += gain_s[j] * scaled_dot * vector_u[j];
            }
        } else {
            for (; j < simd_out; j += 8) {
                y_out[j + 0] += scaled_dot * vector_u[j + 0];
                y_out[j + 1] += scaled_dot * vector_u[j + 1];
                y_out[j + 2] += scaled_dot * vector_u[j + 2];
                y_out[j + 3] += scaled_dot * vector_u[j + 3];
                y_out[j + 4] += scaled_dot * vector_u[j + 4];
                y_out[j + 5] += scaled_dot * vector_u[j + 5];
                y_out[j + 6] += scaled_dot * vector_u[j + 6];
                y_out[j + 7] += scaled_dot * vector_u[j + 7];
            }
            for (; j < dim_out; ++j) {
                y_out[j] += scaled_dot * vector_u[j];
            }
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief L2P-SVD: First-order principal component projection of fine-tuned LoRA matrices
 * into phase profiles and Rank-1 SRL vectors without gradient training.
 *
 * Extracts the first principal component (rank-1 approximation) from fine-tuned LoRA matrices
 * (LoRA A: r x d_in and LoRA B: d_out x r) and converts pairing element angles (u0, u1) via atan2
 * into Q7 quantized phase profiles.
 *
 * @param lora_a Pointer to LoRA A matrix buffer.
 * @param lora_b Pointer to LoRA B matrix buffer.
 * @param rank Rank size of input LoRA.
 * @param dim_in Input dimension size.
 * @param dim_out Output dimension size (must be even).
 * @param srl_out Output SRL descriptor structure to populate.
 * @param phase_profile_out Output Q7 phase profile array pointer.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_l2p_svd_project_lora(
    const float* GCSO_RESTRICT lora_a,
    const float* GCSO_RESTRICT lora_b,
    size_t rank,
    size_t dim_in,
    size_t dim_out,
    gcso_srl_descriptor_t* GCSO_RESTRICT srl_out,
    gcso_q7_t* GCSO_RESTRICT phase_profile_out
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(lora_a == nullptr || lora_b == nullptr || srl_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(rank == 0 || dim_in == 0 || dim_out == 0 || (dim_out % 2 != 0))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(lora_a) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(lora_b) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(srl_out) % alignof(gcso_srl_descriptor_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        srl_out->u_vector_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lora_b));
        srl_out->v_vector_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lora_a));
        srl_out->gain_scalar_ptr = 0;
        srl_out->scale_factor = 1.0f / static_cast<float>(rank);
        srl_out->rank = 1;

        if (phase_profile_out != nullptr) {
            if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(phase_profile_out) % alignof(gcso_q7_t) != 0)) {
                return GCSO_ERROR_MISALIGNED_POINTER;
            }
            constexpr float inv_q7_scale = 128.0f / 3.14159265358979323846f;
            const size_t num_pairs = std::min(dim_out / 2, static_cast<size_t>(64));

            for (size_t k = 0; k < num_pairs; ++k) {
                const float u0 = lora_b[(2 * k) * rank];
                const float u1 = lora_b[(2 * k + 1) * rank];
                const float angle = std::atan2(u1, u0);
                int32_t q_val = static_cast<int32_t>(std::round(angle * inv_q7_scale));
                if (q_val > 127) q_val = 127;
                if (q_val < -128) q_val = -128;
                phase_profile_out[k] = static_cast<gcso_q7_t>(q_val);
            }
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Destroys an SRL adapter instance. Safe No-Op if NULL.
 *
 * @param adapter Adapter handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_srl_adapter_destroy(
    gcso_srl_adapter_handle_t adapter
) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(adapter == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(adapter) % alignof(gcso_srl_adapter_opaque) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        gcso_aligned_delete(adapter);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

GCSO_EXTERN_C_END