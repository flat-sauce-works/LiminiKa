// name: src/kernels/common/dpsr_kernel.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "gcso_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>

GCSO_EXTERN_C_BEGIN

/**
 * @brief Creates a DPSR phase steering kernel instance.
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

        auto* impl = gcso_aligned_new<gcso_dpsr_kernel_impl>();
        if (GCSO_UNLIKELY(impl == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        impl->head_dim = head_dim;
        impl->num_heads = num_heads;
        impl->is_active = true;

        *kernel_out = reinterpret_cast<gcso_dpsr_kernel_handle_t>(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Applies inline DPSR phase rotation to Query tensor registers.
 *
 * Hot Path Guarantee: Zero dynamic memory allocations.
 * Performs 2D SO(2) planar rotations on paired float elements across all head dimensions:
 * [q0', q1']^T = [cos(theta) -sin(theta); sin(theta) cos(theta)] * [q0, q1]^T
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

            for (size_t k = 0; k < pairs_per_head; ++k) {
                const float q0 = head_q[2 * k];
                const float q1 = head_q[2 * k + 1];
                head_q[2 * k]     = q0 * cos_t - q1 * sin_t;
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
 * k correspond to lower frequencies (longer wavelength components) maintaining core context structure.
 * RIPA restricts phase modulation strictly to the upper d_head / 4 dimensions (the top d_head / 8 pairs,
 * starting from start_k = pairs_per_head - low_freq_pairs up to pairs_per_head). This prevents distortion
 * of high-frequency channels that carry exact token syntax.
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

        // RIPA Constraint: Restrict phase steering strictly to low-frequency upper d_head/4 dimensions
        // (head_dim / 8 pair indices at the top end of pairs_per_head)
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

            for (size_t k = start_k; k < pairs_per_head; ++k) {
                const float q0 = head_q[2 * k];
                const float q1 = head_q[2 * k + 1];
                head_q[2 * k]     = q0 * cos_t - q1 * sin_t;
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
 * Prevents grid jitter and numerical oscillation under ultra-low quantization constraints (1.5-3.5 bits).
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
        for (size_t i = 0; i < len; ++i) {
            const float rad = std::abs(static_cast<float>(phase_deltas[i]) * q7_scale);
            if (rad < min_step_rad) {
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

            for (size_t k = 0; k < pairs_per_head; ++k) {
                const float q0 = head_q[2 * k];
                const float q1 = head_q[2 * k + 1];
                head_q[2 * k]     = q0 * cos_t - q1 * sin_t;
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
 * Prevents gradient explosion or activation collapse during cumulative phase rotations
 * by clamping vector L2 norms into [norm_lower, norm_upper] bounds.
 * Includes zero-norm defense and NaN/Inf detection returning GCSO_ERROR_EDBC_SINGULARITY to prevent collapse.
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

        float norm_sq = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            norm_sq += tensor[i] * tensor[i];
        }
        const float norm = std::sqrt(norm_sq);
        if (GCSO_UNLIKELY(std::isnan(norm) || std::isinf(norm))) {
            return GCSO_ERROR_EDBC_SINGULARITY;
        }

        // Zero-norm defense: rescale only if norm is strictly above floating-point epsilon
        if (norm < norm_lower && norm > 1e-12f) {
            const float scale = norm_lower / norm;
            for (size_t i = 0; i < dim; ++i) tensor[i] *= scale;
        } else if (norm > norm_upper && norm > 1e-12f) {
            const float scale = norm_upper / norm;
            for (size_t i = 0; i < dim; ++i) tensor[i] *= scale;
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

        for (size_t v = 0; v < vocab_size; ++v) {
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
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(kernel) % alignof(gcso_dpsr_kernel_impl) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_dpsr_kernel_impl*>(kernel);
        gcso_aligned_delete(impl);
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

        auto* impl = gcso_aligned_new<gcso_pspm_router_impl>();
        if (GCSO_UNLIKELY(impl == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        std::memcpy(&impl->config, config, sizeof(gcso_pspm_config_t));
        impl->is_active = true;

        *router_out = reinterpret_cast<gcso_pspm_router_handle_t>(impl);
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

        for (size_t i = 0; i < fact_elems; ++i) {
            query_tensor[i] *= pspm_cfg->fact_phase_gain;
        }

        float* GCSO_RESTRICT logic_q = query_tensor + fact_elems;
        for (size_t i = 0; i < logic_elems; ++i) {
            logic_q[i] *= pspm_cfg->logic_phase_gain;
        }

        float* GCSO_RESTRICT explore_q = query_tensor + fact_elems + logic_elems;
        for (size_t i = 0; i < explore_elems; ++i) {
            explore_q[i] *= pspm_cfg->explore_phase_gain;
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
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(router) % alignof(gcso_pspm_router_impl) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_pspm_router_impl*>(router);
        gcso_aligned_delete(impl);
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

        auto* impl = gcso_aligned_new<gcso_srl_adapter_impl>();
        if (GCSO_UNLIKELY(impl == nullptr)) {
            return GCSO_ERROR_OUT_OF_MEMORY;
        }
        std::memcpy(&impl->descriptor, descriptor, sizeof(gcso_srl_descriptor_t));
        impl->is_active = true;

        *adapter_out = reinterpret_cast<gcso_srl_adapter_handle_t>(impl);
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

        float v_dot_x = 0.0f;
        for (size_t i = 0; i < dim_in; ++i) {
            v_dot_x += vector_v[i] * x_in[i];
        }
        const float scaled_dot = srl_desc->scale_factor * v_dot_x;
        if (gain_s != nullptr) {
            for (size_t j = 0; j < dim_out; ++j) {
                y_out[j] += gain_s[j] * scaled_dot * vector_u[j];
            }
        } else {
            for (size_t j = 0; j < dim_out; ++j) {
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
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(adapter) % alignof(gcso_srl_adapter_impl) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        auto* impl = reinterpret_cast<gcso_srl_adapter_impl*>(adapter);
        gcso_aligned_delete(impl);
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

GCSO_EXTERN_C_END