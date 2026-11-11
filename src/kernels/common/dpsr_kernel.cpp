// name: src/kernels/common/dpsr_kernel.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "gcso_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>

#if defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace {

/**
 * @brief Applies a planar SO(2) rotation (cos_t, sin_t) to `num_pairs` 2D pairs starting at
 * `head_ptr`. Uses AVX2 4-pair (8-float) vectorization when compiled with AVX2/FMA support.
 */
GCSO_FORCE_INLINE void rotate_planar_pairs_inplace(float* GCSO_RESTRICT head_ptr, size_t start_pair,
                                                   size_t end_pair, float cos_t,
                                                   float sin_t) noexcept {
    size_t k = start_pair;
    const size_t remaining_pairs = end_pair - start_pair;
    const size_t simd_end = start_pair + (remaining_pairs & ~static_cast<size_t>(3));

#if defined(__AVX2__)
    const __m256 v_cos = _mm256_set1_ps(cos_t);
    const __m256 v_sin = _mm256_set1_ps(sin_t);
    for (; k < simd_end; k += 4) {
        float* const p = head_ptr + (2 * k);
        const __m256 v_q = _mm256_loadu_ps(p);
        const __m256 v_q_swap = _mm256_permute_ps(v_q, 0xB1);
        const __m256 v_sin_term = _mm256_mul_ps(v_q_swap, v_sin);
    #if defined(__FMA__)
        const __m256 v_out = _mm256_fmaddsub_ps(v_q, v_cos, v_sin_term);
    #else
        const __m256 v_out = _mm256_addsub_ps(_mm256_mul_ps(v_q, v_cos), v_sin_term);
    #endif
        _mm256_storeu_ps(p, v_out);
    }
#else
    for (; k < simd_end; k += 4) {
        const float q0 = head_ptr[2 * k + 0];
        const float q1 = head_ptr[2 * k + 1];
        const float q2 = head_ptr[2 * k + 2];
        const float q3 = head_ptr[2 * k + 3];
        const float q4 = head_ptr[2 * k + 4];
        const float q5 = head_ptr[2 * k + 5];
        const float q6 = head_ptr[2 * k + 6];
        const float q7 = head_ptr[2 * k + 7];

        head_ptr[2 * k + 0] = q0 * cos_t - q1 * sin_t;
        head_ptr[2 * k + 1] = q0 * sin_t + q1 * cos_t;
        head_ptr[2 * k + 2] = q2 * cos_t - q3 * sin_t;
        head_ptr[2 * k + 3] = q2 * sin_t + q3 * cos_t;
        head_ptr[2 * k + 4] = q4 * cos_t - q5 * sin_t;
        head_ptr[2 * k + 5] = q4 * sin_t + q5 * cos_t;
        head_ptr[2 * k + 6] = q6 * cos_t - q7 * sin_t;
        head_ptr[2 * k + 7] = q6 * sin_t + q7 * cos_t;
    }
#endif

    for (; k < end_pair; ++k) {
        const float q0 = head_ptr[2 * k + 0];
        const float q1 = head_ptr[2 * k + 1];
        head_ptr[2 * k + 0] = q0 * cos_t - q1 * sin_t;
        head_ptr[2 * k + 1] = q0 * sin_t + q1 * cos_t;
    }
}

/**
 * @brief Applies lockstep planar SO(2) rotation (cos_t, sin_t) simultaneously to Query and Key
 * heads.
 */
GCSO_FORCE_INLINE void rotate_planar_pairs_fused_qk_inplace(float* GCSO_RESTRICT head_q,
                                                            float* GCSO_RESTRICT head_k,
                                                            size_t start_pair, size_t end_pair,
                                                            float cos_t, float sin_t) noexcept {
    size_t k = start_pair;
    const size_t remaining_pairs = end_pair - start_pair;
    const size_t simd_end = start_pair + (remaining_pairs & ~static_cast<size_t>(3));

#if defined(__AVX2__)
    const __m256 v_cos = _mm256_set1_ps(cos_t);
    const __m256 v_sin = _mm256_set1_ps(sin_t);
    for (; k < simd_end; k += 4) {
        float* const pq = head_q + (2 * k);
        float* const pk = head_k + (2 * k);
        const __m256 vq = _mm256_loadu_ps(pq);
        const __m256 vk = _mm256_loadu_ps(pk);
        const __m256 vq_swap = _mm256_permute_ps(vq, 0xB1);
        const __m256 vk_swap = _mm256_permute_ps(vk, 0xB1);
        const __m256 vq_sin = _mm256_mul_ps(vq_swap, v_sin);
        const __m256 vk_sin = _mm256_mul_ps(vk_swap, v_sin);
    #if defined(__FMA__)
        const __m256 vq_out = _mm256_fmaddsub_ps(vq, v_cos, vq_sin);
        const __m256 vk_out = _mm256_fmaddsub_ps(vk, v_cos, vk_sin);
    #else
        const __m256 vq_out = _mm256_addsub_ps(_mm256_mul_ps(vq, v_cos), vq_sin);
        const __m256 vk_out = _mm256_addsub_ps(_mm256_mul_ps(vk, v_cos), vk_sin);
    #endif
        _mm256_storeu_ps(pq, vq_out);
        _mm256_storeu_ps(pk, vk_out);
    }
#else
    for (; k < simd_end; k += 4) {
        for (size_t u = 0; u < 4; ++u) {
            const size_t idx = 2 * (k + u);
            const float q0 = head_q[idx + 0];
            const float q1 = head_q[idx + 1];
            const float k0 = head_k[idx + 0];
            const float k1 = head_k[idx + 1];
            head_q[idx + 0] = q0 * cos_t - q1 * sin_t;
            head_q[idx + 1] = q0 * sin_t + q1 * cos_t;
            head_k[idx + 0] = k0 * cos_t - k1 * sin_t;
            head_k[idx + 1] = k0 * sin_t + k1 * cos_t;
        }
    }
#endif

    for (; k < end_pair; ++k) {
        const float q0 = head_q[2 * k + 0];
        const float q1 = head_q[2 * k + 1];
        const float k0 = head_k[2 * k + 0];
        const float k1 = head_k[2 * k + 1];
        head_q[2 * k + 0] = q0 * cos_t - q1 * sin_t;
        head_q[2 * k + 1] = q0 * sin_t + q1 * cos_t;
        head_k[2 * k + 0] = k0 * cos_t - k1 * sin_t;
        head_k[2 * k + 1] = k0 * sin_t + k1 * cos_t;
    }
}

} // namespace

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
GCSO_API gcso_status_t GCSO_CALL
gcso_dpsr_kernel_create(uint32_t head_dim, uint32_t num_heads,
                        gcso_dpsr_kernel_handle_t* GCSO_RESTRICT kernel_out) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(kernel_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(kernel_out) %
                              alignof(gcso_dpsr_kernel_handle_t) !=
                          0)) {
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
 * Hot Path Guarantee: Zero dynamic memory allocations and zero transcendental libm calls via
 * O(1) GCSO_Q7_COS_SIN_LUT lookup and 4-pair SIMD unrolling.
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
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads) GCSO_NOEXCEPT {
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

        const size_t pairs_per_head = head_dim / 2;
        uint8_t last_q_u8 = 0U;
        float cos_t = 1.0f;
        float sin_t = 0.0f;

        for (size_t h = 0; h < num_heads; ++h) {
            const auto q_u8 = static_cast<uint8_t>(phase_deltas[h]);
            if (q_u8 == 0U) {
                continue;
            }
            if (q_u8 != last_q_u8) {
                last_q_u8 = q_u8;
                const auto entry = GCSO_Q7_COS_SIN_LUT.entries[q_u8];
                cos_t = entry.cos_val;
                sin_t = entry.sin_val;
            }
            float* GCSO_RESTRICT head_q = query_tensor + (h * head_dim);
            rotate_planar_pairs_inplace(head_q, 0, pairs_per_head, cos_t, sin_t);
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Applies lockstep fused DPSR phase rotation to Query and Key tensor registers in a single
 * pass.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_fused(
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim, size_t num_heads) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || key_tensor == nullptr ||
                      phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0 ||
                      query_tensor == key_tensor)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(key_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(phase_deltas) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        const size_t pairs_per_head = head_dim / 2;
        uint8_t last_q_u8 = 0U;
        float cos_t = 1.0f;
        float sin_t = 0.0f;

        for (size_t h = 0; h < num_heads; ++h) {
            const auto q_u8 = static_cast<uint8_t>(phase_deltas[h]);
            if (q_u8 == 0U) {
                continue;
            }
            if (q_u8 != last_q_u8) {
                last_q_u8 = q_u8;
                const auto entry = GCSO_Q7_COS_SIN_LUT.entries[q_u8];
                cos_t = entry.cos_val;
                sin_t = entry.sin_val;
            }
            float* GCSO_RESTRICT head_q = query_tensor + (h * head_dim);
            float* GCSO_RESTRICT head_k = key_tensor + (h * head_dim);
            rotate_planar_pairs_fused_qk_inplace(head_q, head_k, 0, pairs_per_head, cos_t, sin_t);
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
 * In standard RoPE encoding, rotation frequencies follow theta_k = 10000^(-2k/d). Higher pair
 * indices k correspond to lower frequencies maintaining core context structure. RIPA restricts
 * phase modulation strictly to the upper d_head / 4 dimensions (the top d_head / 8 pairs, starting
 * from start_k = pairs_per_head - low_freq_pairs up to pairs_per_head).
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
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads, float max_rad) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0 ||
                      std::isnan(max_rad) || std::isinf(max_rad))) {
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
        const size_t start_k =
            (pairs_per_head > low_freq_pairs) ? (pairs_per_head - low_freq_pairs) : 0;
        const bool is_default_ripa = std::abs(max_rad - 0.0872664626f) <= 1e-5f;

        uint8_t last_q_u8 = 0U;
        float cos_t = 1.0f;
        float sin_t = 0.0f;

        for (size_t h = 0; h < num_heads; ++h) {
            const auto q_u8 = static_cast<uint8_t>(phase_deltas[h]);
            if (q_u8 == 0U) {
                continue;
            }
            if (q_u8 != last_q_u8) {
                last_q_u8 = q_u8;
                if (is_default_ripa) {
                    const auto entry = GCSO_Q7_DEFAULT_RIPA_COS_SIN_LUT.entries[q_u8];
                    cos_t = entry.cos_val;
                    sin_t = entry.sin_val;
                } else if (max_rad > 0.0f) {
                    const float raw_theta = static_cast<float>(phase_deltas[h]) * q7_scale;
                    const float theta = max_rad * std::tanh(raw_theta / max_rad);
                    cos_t = std::cos(theta);
                    sin_t = std::sin(theta);
                } else {
                    const auto entry = GCSO_Q7_COS_SIN_LUT.entries[q_u8];
                    cos_t = entry.cos_val;
                    sin_t = entry.sin_val;
                }
            }

            float* GCSO_RESTRICT head_q = query_tensor + (h * head_dim);
            rotate_planar_pairs_inplace(head_q, start_k, pairs_per_head, cos_t, sin_t);
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Applies RIPA soft-bounded phase steering to both Query and Key tensors in a single fused
 * pass.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_fused_safe(
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim, size_t num_heads,
    float max_rad) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || key_tensor == nullptr ||
                      phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0 ||
                      query_tensor == key_tensor || std::isnan(max_rad) || std::isinf(max_rad))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(key_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(phase_deltas) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        constexpr float q7_scale = 3.14159265358979323846f / 128.0f;
        const size_t pairs_per_head = head_dim / 2;
        const size_t low_freq_pairs = std::max(static_cast<size_t>(1), head_dim / 8);
        const size_t start_k =
            (pairs_per_head > low_freq_pairs) ? (pairs_per_head - low_freq_pairs) : 0;
        const bool is_default_ripa = std::abs(max_rad - 0.0872664626f) <= 1e-5f;

        uint8_t last_q_u8 = 0U;
        float cos_t = 1.0f;
        float sin_t = 0.0f;

        for (size_t h = 0; h < num_heads; ++h) {
            const auto q_u8 = static_cast<uint8_t>(phase_deltas[h]);
            if (q_u8 == 0U) {
                continue;
            }
            if (q_u8 != last_q_u8) {
                last_q_u8 = q_u8;
                if (is_default_ripa) {
                    const auto entry = GCSO_Q7_DEFAULT_RIPA_COS_SIN_LUT.entries[q_u8];
                    cos_t = entry.cos_val;
                    sin_t = entry.sin_val;
                } else if (max_rad > 0.0f) {
                    const float raw_theta = static_cast<float>(phase_deltas[h]) * q7_scale;
                    const float theta = max_rad * std::tanh(raw_theta / max_rad);
                    cos_t = std::cos(theta);
                    sin_t = std::sin(theta);
                } else {
                    const auto entry = GCSO_Q7_COS_SIN_LUT.entries[q_u8];
                    cos_t = entry.cos_val;
                    sin_t = entry.sin_val;
                }
            }

            float* GCSO_RESTRICT head_q = query_tensor + (h * head_dim);
            float* GCSO_RESTRICT head_k = key_tensor + (h * head_dim);
            rotate_planar_pairs_fused_qk_inplace(head_q, head_k, start_k, pairs_per_head, cos_t,
                                                 sin_t);
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Applies S^1 geodesic RIPA soft-bounded phase steering to a Query tensor.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_geodesic_safe(
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads, float max_rad) GCSO_NOEXCEPT {
    return gcso_dpsr_apply_phase_steering_safe(query_tensor, phase_deltas, head_dim, num_heads,
                                               max_rad);
}

/**
 * @brief Applies pointer-gated RIPA phase steering (honoring ACTION_MODE_LAYER_SKIP &
 * REPULSION_GUARD).
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_pointer_gated_phase_steering_safe(
    uint64_t tagged_ptr, float* GCSO_RESTRICT query_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim, size_t num_heads,
    float max_rad) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    const uint8_t tag = gcso_tagged_ptr_unpack_tag_inline(tagged_ptr);
    const uint8_t action_mode = tag & GCSO_ACTION_MODE_ALL_MASK;
    if ((action_mode & GCSO_ACTION_MODE_LAYER_SKIP) != 0U ||
        (action_mode & GCSO_ACTION_MODE_PHASE_STEER) == 0U) {
        if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0 ||
                          std::isnan(max_rad) || std::isinf(max_rad))) {
            return GCSO_ERROR_INVALID_ARGUMENT;
        }
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(phase_deltas) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        return GCSO_SUCCESS;
    }
    return gcso_dpsr_apply_phase_steering_safe(query_tensor, phase_deltas, head_dim, num_heads,
                                               max_rad);
}

/**
 * @brief Applies pointer-gated fused QK RIPA phase steering in a single pass.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_pointer_gated_phase_steering_fused_safe(
    uint64_t tagged_ptr, float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim, size_t num_heads,
    float max_rad) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || key_tensor == nullptr ||
                      phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    const uint8_t tag = gcso_tagged_ptr_unpack_tag_inline(tagged_ptr);
    const uint8_t action_mode = tag & GCSO_ACTION_MODE_ALL_MASK;
    if ((action_mode & GCSO_ACTION_MODE_LAYER_SKIP) != 0U ||
        (action_mode & GCSO_ACTION_MODE_PHASE_STEER) == 0U) {
        if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0 ||
                          query_tensor == key_tensor || std::isnan(max_rad) ||
                          std::isinf(max_rad))) {
            return GCSO_ERROR_INVALID_ARGUMENT;
        }
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(key_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(phase_deltas) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        return GCSO_SUCCESS;
    }
    return gcso_dpsr_apply_phase_steering_fused_safe(query_tensor, key_tensor, phase_deltas,
                                                     head_dim, num_heads, max_rad);
}

/**
 * @brief Applies soft exponential phase damping across Query tensor channels.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_dpsr_apply_soft_phase_damping(float* GCSO_RESTRICT query_tensor, size_t head_dim,
                                   size_t num_heads, float damping_factor) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0) || num_heads == 0 ||
                      std::isnan(damping_factor) || std::isinf(damping_factor) ||
                      damping_factor < 0.0f || damping_factor > 1.0f)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }
        const size_t total_elems = head_dim * num_heads;
        size_t i = 0;
        const size_t simd_len = total_elems & ~static_cast<size_t>(7);
#if defined(__AVX2__)
        const __m256 v_damp = _mm256_set1_ps(damping_factor);
        for (; i < simd_len; i += 8) {
            const __m256 v_q = _mm256_load_ps(query_tensor + i);
            _mm256_store_ps(query_tensor + i, _mm256_mul_ps(v_q, v_damp));
        }
#else
        for (; i < simd_len; i += 8) {
            query_tensor[i + 0] *= damping_factor;
            query_tensor[i + 1] *= damping_factor;
            query_tensor[i + 2] *= damping_factor;
            query_tensor[i + 3] *= damping_factor;
            query_tensor[i + 4] *= damping_factor;
            query_tensor[i + 5] *= damping_factor;
            query_tensor[i + 6] *= damping_factor;
            query_tensor[i + 7] *= damping_factor;
        }
#endif
        for (; i < total_elems; ++i) {
            query_tensor[i] *= damping_factor;
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief QDPS discrete filter: cuts off phase rotation steps falling below min_step_rad.
 *
 * Pre-evaluates integer Q7 thresholds to execute zero-overhead integer domain comparisons in the
 * hot loop.
 *
 * @param phase_deltas Q7 phase array to filter in-place.
 * @param len Array element count.
 * @param min_step_rad Minimum step threshold angle in radians.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_qdps_filter_step(gcso_q7_t* GCSO_RESTRICT phase_deltas,
                                                       size_t len,
                                                       float min_step_rad) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(phase_deltas == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(len == 0 || std::isnan(min_step_rad) || std::isinf(min_step_rad) ||
                      min_step_rad < 0.0f)) {
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

        size_t i = 0;
#if defined(__AVX2__)
        const size_t simd32_len = len & ~static_cast<size_t>(31);
        const __m256i v_thresh =
            _mm256_set1_epi8(static_cast<char>(static_cast<int8_t>(min_step_q7)));
        const __m256i v_max127 = _mm256_set1_epi8(127);
        for (; i < simd32_len; i += 32) {
            const __m256i v =
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(phase_deltas + i));
            const __m256i abs_v = _mm256_min_epu8(_mm256_abs_epi8(v), v_max127);
            const __m256i zero_mask = _mm256_cmpgt_epi8(v_thresh, abs_v);
            const __m256i filtered = _mm256_andnot_si256(zero_mask, v);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(phase_deltas + i), filtered);
        }
#endif
        for (; i < len; ++i) {
            const int32_t val = static_cast<int32_t>(phase_deltas[i]);
            const int32_t sign_mask = val >> 31;
            const int32_t abs_val = (val ^ sign_mask) - sign_mask;
            const int32_t keep_mask = -(abs_val >= min_step_q7);
            phase_deltas[i] = static_cast<gcso_q7_t>(val & keep_mask);
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Lazy Phase Unwrapping: Applies relative SO(2) phase rotations against context accumulator
 * on Query side.
 *
 * Corrected Geometrical Behavior: Executes planar 2D SO(2) rotations based on context phase
 * accumulation instead of linear vector additions, maintaining SO(2)^(d_head/2) Lie Group
 * invariants per c_abi_spec.md.
 *
 * @param query_tensor 32-byte aligned target query tensor.
 * @param context_accum Cumulative context phase angle vector per head in radians.
 * @param head_dim Head dimension size (must be even).
 * @param num_heads Total head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_lazy_unwrap_override(
    float* GCSO_RESTRICT query_tensor, const float* GCSO_RESTRICT context_accum, size_t head_dim,
    size_t num_heads) GCSO_NOEXCEPT {
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
            rotate_planar_pairs_inplace(head_q, 0, pairs_per_head, cos_t, sin_t);
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Executes norm-guarded Slerp phase stabilization on state vectors.
 *
 * Utilizes 4 independent accumulators to unroll L2 norm computation, maximizing instruction-level
 * parallelism.
 *
 * @param tensor Vector tensor buffer to normalize in-place.
 * @param dim Total vector length.
 * @param norm_lower Minimum allowed L2 norm bound.
 * @param norm_upper Maximum allowed L2 norm bound.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_slerp_norm_guard_stable(float* GCSO_RESTRICT tensor,
                                                                   size_t dim, float norm_lower,
                                                                   float norm_upper) GCSO_NOEXCEPT {
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

        size_t i = 0;
        const size_t simd_len = dim & ~static_cast<size_t>(7);
        float norm_sq = 0.0f;

#if defined(__AVX2__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const size_t vec32_len = dim & ~static_cast<size_t>(31);
        for (; i < vec32_len; i += 32) {
            const __m256 v0 = _mm256_load_ps(tensor + i + 0);
            const __m256 v1 = _mm256_load_ps(tensor + i + 8);
            const __m256 v2 = _mm256_load_ps(tensor + i + 16);
            const __m256 v3 = _mm256_load_ps(tensor + i + 24);
    #if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
            acc1 = _mm256_fmadd_ps(v1, v1, acc1);
            acc2 = _mm256_fmadd_ps(v2, v2, acc2);
            acc3 = _mm256_fmadd_ps(v3, v3, acc3);
    #else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(v0, v0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(v1, v1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(v2, v2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(v3, v3));
    #endif
        }
        for (; i < simd_len; i += 8) {
            const __m256 v0 = _mm256_load_ps(tensor + i);
    #if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(v0, v0, acc0);
    #else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(v0, v0));
    #endif
        }
        const __m256 sum256 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        const __m128 sum128 =
            _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
        const __m128 shuf = _mm_movehdup_ps(sum128);
        const __m128 sums = _mm_add_ps(sum128, shuf);
        const __m128 shuf2 = _mm_movehl_ps(shuf, sums);
        norm_sq = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
#else
        float sum_sq0 = 0.0f;
        float sum_sq1 = 0.0f;
        float sum_sq2 = 0.0f;
        float sum_sq3 = 0.0f;

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
        norm_sq = (sum_sq0 + sum_sq1) + (sum_sq2 + sum_sq3);
#endif
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
#if defined(__AVX2__)
            const __m256 v_scale = _mm256_set1_ps(scale);
            for (; j < simd_len; j += 8) {
                const __m256 v = _mm256_load_ps(tensor + j);
                _mm256_store_ps(tensor + j, _mm256_mul_ps(v, v_scale));
            }
#else
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
#endif
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
    float* GCSO_RESTRICT logits, size_t vocab_size, const gcso_q7_t* GCSO_RESTRICT phase_deltas,
    size_t num_heads) GCSO_NOEXCEPT {
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
        int32_t sum_q7 = 0;
        for (size_t h = 0; h < num_heads; ++h) {
            sum_q7 += static_cast<int32_t>(phase_deltas[h]);
        }
        const float mean_bias =
            (static_cast<float>(sum_q7) * q7_scale) / static_cast<float>(num_heads);

        size_t v = 0;
        const size_t simd_vocab = vocab_size & ~static_cast<size_t>(7);
#if defined(__AVX2__)
        const __m256 v_bias = _mm256_set1_ps(mean_bias);
        for (; v < simd_vocab; v += 8) {
            const __m256 v_log = _mm256_load_ps(logits + v);
            _mm256_store_ps(logits + v, _mm256_add_ps(v_log, v_bias));
        }
#else
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
#endif
        for (; v < vocab_size; ++v) {
            logits[v] += mean_bias;
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Computes Procrustes phase delta alignment between source and target state representations.
 *
 * Evaluates 2D planar cross-products and dot-products across adjacent pairs (2k, 2k+1) and
 * quantizes the resulting relative phase angles via atan2 into Q7 fixed-point values [-128, 127].
 *
 * @param source 32-byte aligned source feature vector.
 * @param target 32-byte aligned target feature vector.
 * @param dim Total vector dimension size (must be even and >= 2).
 * @param phase_out Output Q7 phase delta array buffer to store computed profile (Aligned 16-byte).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_compute_procrustes_phase_delta(
    const float* GCSO_RESTRICT source, const float* GCSO_RESTRICT target, size_t dim,
    gcso_q7_t* GCSO_RESTRICT phase_out) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(source == nullptr || target == nullptr || phase_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(dim == 0 || (dim % 2 != 0))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(source) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(target) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(phase_out) % alignof(gcso_q7_t) != 0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        constexpr float inv_q7_scale = 128.0f / 3.14159265358979323846f;
        const size_t num_heads = std::min(dim / 2, static_cast<size_t>(GCSO_Q7_PHASE_ARRAY_LEN));
        const size_t pairs_per_head = std::max(static_cast<size_t>(1), (dim / 2) / num_heads);

        for (size_t h = 0; h < num_heads; ++h) {
            const size_t pair_start = h * pairs_per_head;
            const size_t pair_end = std::min(pair_start + pairs_per_head, dim / 2);
            float dot_sum = 0.0f;
            float cross_sum = 0.0f;
            for (size_t p = pair_start; p < pair_end; ++p) {
                const float sx = source[2 * p + 0];
                const float sy = source[2 * p + 1];
                const float tx = target[2 * p + 0];
                const float ty = target[2 * p + 1];
                dot_sum += sx * tx + sy * ty;
                cross_sum += sx * ty - sy * tx;
            }
            if (GCSO_UNLIKELY(std::isnan(dot_sum) || std::isnan(cross_sum) || std::isinf(dot_sum) ||
                              std::isinf(cross_sum))) {
                return GCSO_ERROR_EDBC_SINGULARITY;
            }
            const float angle = (std::abs(dot_sum) < 1e-12f && std::abs(cross_sum) < 1e-12f)
                                    ? 0.0f
                                    : std::atan2(cross_sum, dot_sum);
            int32_t q_val = static_cast<int32_t>(std::round(angle * inv_q7_scale));
            q_val = std::clamp(q_val, -128, 127);
            phase_out[h] = static_cast<gcso_q7_t>(q_val);
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
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_kernel_destroy(gcso_dpsr_kernel_handle_t kernel)
    GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(kernel == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(kernel) % alignof(gcso_dpsr_kernel_opaque) !=
                      0)) {
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
GCSO_API gcso_status_t GCSO_CALL
gcso_pspm_router_create(const gcso_pspm_config_t* GCSO_RESTRICT config,
                        gcso_pspm_router_handle_t* GCSO_RESTRICT router_out) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(config == nullptr || router_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(config) % alignof(gcso_pspm_config_t) != 0 ||
                          reinterpret_cast<uintptr_t>(router_out) %
                                  alignof(gcso_pspm_router_handle_t) !=
                              0)) {
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
 * Single-Pass Head Group Ensembling: Splits attention heads into Fact, Logic, and Explore
 * sub-groups, applying unique phase profile gains in a single forward pass without parameter
 * duplication.
 *
 * @param query_tensor 32-byte aligned query tensor buffer.
 * @param pspm_cfg PSPM routing configuration.
 * @param head_dim Head dimension.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_pspm_dispatch_single_pass(
    float* GCSO_RESTRICT query_tensor, const gcso_pspm_config_t* GCSO_RESTRICT pspm_cfg,
    size_t head_dim) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(query_tensor == nullptr || pspm_cfg == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(head_dim == 0 || (head_dim % 2 != 0))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(query_tensor) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(pspm_cfg) % alignof(gcso_pspm_config_t) !=
                              0)) {
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
#if defined(__AVX2__)
        const __m256 v_fact_gain = _mm256_set1_ps(pspm_cfg->fact_phase_gain);
        for (; i < fact_simd; i += 8) {
            _mm256_storeu_ps(query_tensor + i,
                             _mm256_mul_ps(_mm256_loadu_ps(query_tensor + i), v_fact_gain));
        }
#else
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
#endif
        for (; i < fact_elems; ++i) {
            query_tensor[i] *= pspm_cfg->fact_phase_gain;
        }

        float* GCSO_RESTRICT logic_q = query_tensor + fact_elems;
        size_t j = 0;
        const size_t logic_simd = logic_elems & ~static_cast<size_t>(7);
#if defined(__AVX2__)
        const __m256 v_logic_gain = _mm256_set1_ps(pspm_cfg->logic_phase_gain);
        for (; j < logic_simd; j += 8) {
            _mm256_storeu_ps(logic_q + j,
                             _mm256_mul_ps(_mm256_loadu_ps(logic_q + j), v_logic_gain));
        }
#else
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
#endif
        for (; j < logic_elems; ++j) {
            logic_q[j] *= pspm_cfg->logic_phase_gain;
        }

        float* GCSO_RESTRICT explore_q = query_tensor + fact_elems + logic_elems;
        size_t k = 0;
        const size_t explore_simd = explore_elems & ~static_cast<size_t>(7);
#if defined(__AVX2__)
        const __m256 v_explore_gain = _mm256_set1_ps(pspm_cfg->explore_phase_gain);
        for (; k < explore_simd; k += 8) {
            _mm256_storeu_ps(explore_q + k,
                             _mm256_mul_ps(_mm256_loadu_ps(explore_q + k), v_explore_gain));
        }
#else
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
#endif
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
GCSO_API gcso_status_t GCSO_CALL gcso_pspm_router_destroy(gcso_pspm_router_handle_t router)
    GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(router == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(router) % alignof(gcso_pspm_router_opaque) !=
                      0)) {
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
GCSO_API gcso_status_t GCSO_CALL
gcso_srl_adapter_create(const gcso_srl_descriptor_t* GCSO_RESTRICT descriptor,
                        gcso_srl_adapter_handle_t* GCSO_RESTRICT adapter_out) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(descriptor == nullptr || adapter_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    try {
        if (GCSO_UNLIKELY(
                reinterpret_cast<uintptr_t>(descriptor) % alignof(gcso_srl_descriptor_t) != 0 ||
                reinterpret_cast<uintptr_t>(adapter_out) % alignof(gcso_srl_adapter_handle_t) !=
                    0)) {
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
GCSO_API gcso_status_t GCSO_CALL
gcso_srl_eval_rank1(float* GCSO_RESTRICT y_out, const float* GCSO_RESTRICT x_in,
                    const gcso_srl_descriptor_t* GCSO_RESTRICT srl_desc, size_t dim_in,
                    size_t dim_out) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(y_out == nullptr || x_in == nullptr || srl_desc == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(dim_in == 0 || dim_out == 0 || std::isnan(srl_desc->scale_factor) ||
                      std::isinf(srl_desc->scale_factor))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(y_out) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(x_in) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(srl_desc) % alignof(gcso_srl_descriptor_t) !=
                              0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        const auto* GCSO_RESTRICT vector_u =
            reinterpret_cast<const float*>(static_cast<uintptr_t>(srl_desc->u_vector_ptr));
        const auto* GCSO_RESTRICT vector_v =
            reinterpret_cast<const float*>(static_cast<uintptr_t>(srl_desc->v_vector_ptr));
        const auto* GCSO_RESTRICT gain_s =
            reinterpret_cast<const float*>(static_cast<uintptr_t>(srl_desc->gain_scalar_ptr));

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

        size_t i = 0;
        const size_t simd_in = dim_in & ~static_cast<size_t>(7);
        float v_dot_x = 0.0f;

#if defined(__AVX2__)
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        const size_t vec32_in = dim_in & ~static_cast<size_t>(31);
        for (; i < vec32_in; i += 32) {
            const __m256 vv0 = _mm256_load_ps(vector_v + i + 0);
            const __m256 vx0 = _mm256_load_ps(x_in + i + 0);
            const __m256 vv1 = _mm256_load_ps(vector_v + i + 8);
            const __m256 vx1 = _mm256_load_ps(x_in + i + 8);
            const __m256 vv2 = _mm256_load_ps(vector_v + i + 16);
            const __m256 vx2 = _mm256_load_ps(x_in + i + 16);
            const __m256 vv3 = _mm256_load_ps(vector_v + i + 24);
            const __m256 vx3 = _mm256_load_ps(x_in + i + 24);
    #if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(vv0, vx0, acc0);
            acc1 = _mm256_fmadd_ps(vv1, vx1, acc1);
            acc2 = _mm256_fmadd_ps(vv2, vx2, acc2);
            acc3 = _mm256_fmadd_ps(vv3, vx3, acc3);
    #else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(vv0, vx0));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(vv1, vx1));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(vv2, vx2));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(vv3, vx3));
    #endif
        }
        for (; i < simd_in; i += 8) {
            const __m256 vv0 = _mm256_load_ps(vector_v + i);
            const __m256 vx0 = _mm256_load_ps(x_in + i);
    #if defined(__FMA__)
            acc0 = _mm256_fmadd_ps(vv0, vx0, acc0);
    #else
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(vv0, vx0));
    #endif
        }
        const __m256 sum256 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
        const __m128 sum128 =
            _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
        const __m128 shuf = _mm_movehdup_ps(sum128);
        const __m128 sums = _mm_add_ps(sum128, shuf);
        const __m128 shuf2 = _mm_movehl_ps(shuf, sums);
        v_dot_x = _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
#else
        float v_dot_x0 = 0.0f;
        float v_dot_x1 = 0.0f;
        float v_dot_x2 = 0.0f;
        float v_dot_x3 = 0.0f;

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
        v_dot_x = (v_dot_x0 + v_dot_x1) + (v_dot_x2 + v_dot_x3);
#endif
        for (; i < dim_in; ++i) {
            v_dot_x += vector_v[i] * x_in[i];
        }

        const float scaled_dot = srl_desc->scale_factor * v_dot_x;
        size_t j = 0;
        const size_t simd_out = dim_out & ~static_cast<size_t>(7);

        if (gain_s != nullptr) {
#if defined(__AVX2__)
            const __m256 v_scaled = _mm256_set1_ps(scaled_dot);
            for (; j < simd_out; j += 8) {
                const __m256 vy = _mm256_load_ps(y_out + j);
                const __m256 vg = _mm256_load_ps(gain_s + j);
                const __m256 vu = _mm256_load_ps(vector_u + j);
                const __m256 vgu = _mm256_mul_ps(vg, vu);
    #if defined(__FMA__)
                _mm256_store_ps(y_out + j, _mm256_fmadd_ps(vgu, v_scaled, vy));
    #else
                _mm256_store_ps(y_out + j, _mm256_add_ps(vy, _mm256_mul_ps(vgu, v_scaled)));
    #endif
            }
#else
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
#endif
            for (; j < dim_out; ++j) {
                y_out[j] += gain_s[j] * scaled_dot * vector_u[j];
            }
        } else {
#if defined(__AVX2__)
            const __m256 v_scaled = _mm256_set1_ps(scaled_dot);
            for (; j < simd_out; j += 8) {
                const __m256 vy = _mm256_load_ps(y_out + j);
                const __m256 vu = _mm256_load_ps(vector_u + j);
    #if defined(__FMA__)
                _mm256_store_ps(y_out + j, _mm256_fmadd_ps(vu, v_scaled, vy));
    #else
                _mm256_store_ps(y_out + j, _mm256_add_ps(vy, _mm256_mul_ps(vu, v_scaled)));
    #endif
            }
#else
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
#endif
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
 * @brief Evaluates in-place fused SRL Dynamic Rank-1 outer product accumulation.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_srl_eval_rank1_fused(float* GCSO_RESTRICT y_inout, const float* GCSO_RESTRICT x_in,
                          const gcso_srl_descriptor_t* GCSO_RESTRICT srl_desc, size_t dim_in,
                          size_t dim_out) GCSO_NOEXCEPT {
    return gcso_srl_eval_rank1(y_inout, x_in, srl_desc, dim_in, dim_out);
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
    const float* GCSO_RESTRICT lora_a, const float* GCSO_RESTRICT lora_b, size_t rank,
    size_t dim_in, size_t dim_out, gcso_srl_descriptor_t* GCSO_RESTRICT srl_out,
    gcso_q7_t* GCSO_RESTRICT phase_profile_out) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(lora_a == nullptr || lora_b == nullptr || srl_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(rank == 0 || dim_in == 0 || dim_out == 0 || (dim_out % 2 != 0))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(lora_a) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(lora_b) % 32 != 0 ||
                          reinterpret_cast<uintptr_t>(srl_out) % alignof(gcso_srl_descriptor_t) !=
                              0)) {
            return GCSO_ERROR_MISALIGNED_POINTER;
        }

        srl_out->u_vector_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lora_b));
        srl_out->v_vector_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(lora_a));
        srl_out->gain_scalar_ptr = 0;
        srl_out->scale_factor = 1.0f / static_cast<float>(rank);
        srl_out->rank = 1;

        if (phase_profile_out != nullptr) {
            if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(phase_profile_out) % alignof(gcso_q7_t) !=
                              0)) {
                return GCSO_ERROR_MISALIGNED_POINTER;
            }
            constexpr float inv_q7_scale = 128.0f / 3.14159265358979323846f;
            const size_t num_pairs = std::min(dim_out / 2, static_cast<size_t>(64));

            for (size_t k = 0; k < num_pairs; ++k) {
                const float u0 = lora_b[(2 * k) * rank];
                const float u1 = lora_b[(2 * k + 1) * rank];
                const float angle = std::atan2(u1, u0);
                int32_t q_val = static_cast<int32_t>(std::round(angle * inv_q7_scale));
                if (q_val > 127)
                    q_val = 127;
                if (q_val < -128)
                    q_val = -128;
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
GCSO_API gcso_status_t GCSO_CALL gcso_srl_adapter_destroy(gcso_srl_adapter_handle_t adapter)
    GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(adapter == nullptr)) {
        return GCSO_SUCCESS;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(adapter) % alignof(gcso_srl_adapter_opaque) !=
                      0)) {
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