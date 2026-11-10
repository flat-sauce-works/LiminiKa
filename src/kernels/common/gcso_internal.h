// name: src/kernels/common/gcso_internal.h
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_INTERNAL_H
#define LIMINIKA_GCSO_INTERNAL_H

// Mandatory C-ABI public headers (Required order: config -> types -> abi)
#include "liminika/gcso_config.h"
#include "liminika/gcso_types.h"
#include "liminika/gcso_abi.h"

// Standard C++20 headers
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

// ===================================================================
// Compile-Time Constexpr Math & Lookup Table Generators (C++20)
// ===================================================================

namespace gcso_detail {

constexpr double GCSO_PI_F64 = 3.14159265358979323846264338327950288;
constexpr double GCSO_Q7_SCALE_F64 = GCSO_PI_F64 / 128.0;
constexpr double GCSO_DEFAULT_RIPA_MAX_RAD_F64 = 0.08726646259971647884618453842443;

/**
 * @brief Compile-time 14-term Taylor series cosine evaluation on [-pi, pi] in IEEE-754 double.
 */
constexpr double constexpr_cos(double x) noexcept {
    const double x2 = x * x;
    double term = 1.0;
    double sum = 1.0;
    for (int k = 1; k <= 14; ++k) {
        term *= -x2 / static_cast<double>((2 * k - 1) * (2 * k));
        sum += term;
    }
    return sum;
}

/**
 * @brief Compile-time 14-term Taylor series sine evaluation on [-pi, pi] in IEEE-754 double.
 */
constexpr double constexpr_sin(double x) noexcept {
    const double x2 = x * x;
    double term = x;
    double sum = x;
    for (int k = 1; k <= 14; ++k) {
        term *= -x2 / static_cast<double>((2 * k) * (2 * k + 1));
        sum += term;
    }
    return sum;
}

/**
 * @brief Compile-time exponential exp(x) evaluation for RIPA soft-clamping tanh(x).
 */
constexpr double constexpr_exp(double x) noexcept {
    if (x > 20.0) {
        return 485165195.4097903;
    }
    if (x < -20.0) {
        return 1.0 / 485165195.4097903;
    }
    int halvings = 0;
    double scaled = x;
    while (scaled > 0.25 || scaled < -0.25) {
        scaled *= 0.5;
        ++halvings;
    }
    double term = 1.0;
    double sum = 1.0;
    for (int k = 1; k <= 18; ++k) {
        term *= scaled / static_cast<double>(k);
        sum += term;
    }
    for (int i = 0; i < halvings; ++i) {
        sum *= sum;
    }
    return sum;
}

/**
 * @brief Compile-time hyperbolic tangent tanh(x) in IEEE-754 double.
 */
constexpr double constexpr_tanh(double x) noexcept {
    if (x > 20.0) {
        return 1.0;
    }
    if (x < -20.0) {
        return -1.0;
    }
    const double e2x = constexpr_exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
}

struct GCSO_ALIGNAS(8) Q7CosSinPair {
    float cos_val;
    float sin_val;
};

struct GCSO_ALIGNAS(64) Q7CosSinTable256 {
    std::array<Q7CosSinPair, 256> entries;
};

constexpr Q7CosSinTable256 build_q7_full_cos_sin_lut() noexcept {
    Q7CosSinTable256 table{};
    for (int u = 0; u < 256; ++u) {
        const auto q = static_cast<int8_t>(static_cast<uint8_t>(u));
        const double theta = static_cast<double>(q) * GCSO_Q7_SCALE_F64;
        table.entries[static_cast<std::size_t>(u)] = Q7CosSinPair{
            static_cast<float>(constexpr_cos(theta)),
            static_cast<float>(constexpr_sin(theta))
        };
    }
    return table;
}

constexpr Q7CosSinTable256 build_q7_default_ripa_cos_sin_lut() noexcept {
    Q7CosSinTable256 table{};
    for (int u = 0; u < 256; ++u) {
        const auto q = static_cast<int8_t>(static_cast<uint8_t>(u));
        const double raw_theta = static_cast<double>(q) * GCSO_Q7_SCALE_F64;
        const double safe_theta = GCSO_DEFAULT_RIPA_MAX_RAD_F64 *
                                  constexpr_tanh(raw_theta / GCSO_DEFAULT_RIPA_MAX_RAD_F64);
        table.entries[static_cast<std::size_t>(u)] = Q7CosSinPair{
            static_cast<float>(constexpr_cos(safe_theta)),
            static_cast<float>(constexpr_sin(safe_theta))
        };
    }
    return table;
}

constexpr std::array<uint32_t, 256> build_crc32c_lut() noexcept {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1U) ^ ((crc & 1U) != 0U ? 0x82F63B78U : 0U);
        }
        table[i] = crc;
    }
    return table;
}

} // namespace gcso_detail

/** @brief 256-entry cacheline-aligned full-range Q7 (cos, sin) lookup table in .rodata. */
inline constexpr gcso_detail::Q7CosSinTable256 GCSO_Q7_COS_SIN_LUT =
    gcso_detail::build_q7_full_cos_sin_lut();

/** @brief 256-entry cacheline-aligned default RIPA (0.087266 rad) Q7 (cos, sin) lookup table. */
inline constexpr gcso_detail::Q7CosSinTable256 GCSO_Q7_DEFAULT_RIPA_COS_SIN_LUT =
    gcso_detail::build_q7_default_ripa_cos_sin_lut();

/** @brief 256-entry O(1)-per-byte CRC32C lookup table for .gcso container verification. */
inline constexpr std::array<uint32_t, 256> GCSO_CRC32C_LUT =
    gcso_detail::build_crc32c_lut();

// ===================================================================
// Over-aligned Safe Memory Allocation Helpers (C++20)
// ===================================================================

/**
 * @brief Allocates and constructs an over-aligned heap object safely in zero-allocation failure mode.
 *
 * Enforces C++20 std::align_val_t over-alignment constraints and returns nullptr on failure without throwing exceptions.
 *
 * @tparam T Object type to instantiate.
 * @tparam Args Constructor argument pack types.
 * @param args Constructor arguments forwarded to object initializer.
 * @return Pointer to allocated and initialized object, or nullptr if allocation fails.
 */
template <typename T, typename... Args>
GCSO_NODISCARD GCSO_INLINE T* gcso_aligned_new(Args&&... args) GCSO_NOEXCEPT {
    try {
        constexpr std::size_t req_align = alignof(T) > sizeof(void*) ? alignof(T) : alignof(std::max_align_t);
        void* ptr = ::operator new(sizeof(T), std::align_val_t(req_align), std::nothrow);
        if (GCSO_UNLIKELY(ptr == nullptr)) {
            return nullptr;
        }
        return ::new (ptr) T(std::forward<Args>(args)...);
    } catch (...) {
        return nullptr;
    }
}

/**
 * @brief Safely destroys and deallocates an over-aligned heap object created via gcso_aligned_new.
 *
 * Suppresses destructor exceptions to strictly maintain the GCSO_NOEXCEPT contract across FFI boundaries.
 *
 * @tparam T Object type to destroy.
 * @param ptr Pointer to instance to deallocate (safe no-op if nullptr).
 */
template <typename T>
GCSO_INLINE void gcso_aligned_delete(T* ptr) GCSO_NOEXCEPT {
    if (GCSO_LIKELY(ptr != nullptr)) {
        constexpr std::size_t req_align = alignof(T) > sizeof(void*) ? alignof(T) : alignof(std::max_align_t);
        try {
            ptr->~T();
        } catch (...) {
            // Suppress destructor exceptions across FFI boundary
        }
        ::operator delete(ptr, std::align_val_t(req_align), std::nothrow);
    }
}

/**
 * @brief Checks if a given raw memory pointer satisfies a specified byte alignment requirement.
 *
 * @tparam T Pointer underlying element type.
 * @param ptr Memory address pointer to validate.
 * @param align_bytes Required alignment boundary in bytes (must be non-zero power of two).
 * @return True if non-null and correctly aligned; false otherwise.
 */
template <typename T>
GCSO_NODISCARD GCSO_INLINE bool gcso_is_aligned(const T* ptr, std::size_t align_bytes) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(ptr == nullptr || align_bytes == 0 || (align_bytes & (align_bytes - 1)) != 0)) {
        return false;
    }
    return (reinterpret_cast<std::uintptr_t>(ptr) % align_bytes) == 0;
}

/**
 * @brief Performs lock-free atomic pointer comparison and swap for zero-copy state transitions.
 *
 * @tparam T Target pointer underlying type.
 * @param target Atomic reference pointer to exchange.
 * @param expected Pointer value expected to be held in target.
 * @param desired Pointer value to store in target upon match.
 * @return True if CAS operation succeeded; false otherwise.
 */
template <typename T>
GCSO_INLINE bool gcso_atomic_ptr_swap(std::atomic<T*>& target, T* expected, T* desired) GCSO_NOEXCEPT {
    return target.compare_exchange_strong(expected, desired, std::memory_order_acq_rel, std::memory_order_acquire);
}

// ===================================================================
// Strongly-Typed Internal Opaque Implementation Structures
// Direct concrete definitions matching C-ABI handle pointer prototypes
// ===================================================================

/**
 * @brief Internal runtime execution context layout (matching gcso_context_opaque).
 * Aligned to 64 bytes to fit CPU/GPU cache line boundaries without false sharing.
 */
struct GCSO_ALIGNAS(64) gcso_context_opaque {
    gcso_config_t config;              ///< Global execution configuration snapshot.
    uint32_t active_token_id;          ///< Active token sequence ID in decoding step.
    uint32_t prompt_anchor_count;      ///< Number of registered system prompt anchors.
    uint32_t head_dim;                 ///< Attention head dimension (must be even).
    uint32_t num_heads;                ///< Total number of attention heads (max 64).
    uint64_t step_count;               ///< Monotonically increasing inference step counter.
    uint64_t last_ptr;                 ///< Tagged pointer address from previous step.
    gcso_q7_t accumulated_phase[64];   ///< Q7 fixed-point cumulative phase vector per head.
    bool is_initialized;               ///< State flag indicating valid initialization.
    bool is_active;                    ///< Lifecycle flag tracking context activity.
};

/**
 * @brief Internal Action Hub layout (matching gcso_action_hub_opaque).
 * Aligned to 32 bytes for optimized vector lookup and cacheline efficiency.
 */
struct GCSO_ALIGNAS(32) gcso_action_hub_opaque {
    uint32_t capacity;         ///< Maximum tagged pointer slot capacity.
    uint32_t active_count;     ///< Current active pointer trail count.
    uint64_t last_trail_ptr;   ///< Address of most recently transitioned pointer trail.
    bool is_active;            ///< Lifecycle state flag.
};

/**
 * @brief Internal Attractor Field layout (matching gcso_attractor_field_opaque).
 * Aligned to 32 bytes for topological anchor management.
 */
struct GCSO_ALIGNAS(32) gcso_attractor_field_opaque {
    uint32_t anchor_count;    ///< Total number of active topological anchors.
    uint8_t anchors[256];     ///< Static feature space allocation for anchor vectors.
    bool is_active;           ///< Lifecycle state flag.
};

/**
 * @brief Internal Dynamic Phase-Shifted RoPE kernel state (matching gcso_dpsr_kernel_opaque).
 * Aligned to 16 bytes.
 */
struct GCSO_ALIGNAS(16) gcso_dpsr_kernel_opaque {
    uint32_t head_dim;   ///< Target attention head dimension.
    uint32_t num_heads;  ///< Total number of attention heads.
    bool is_active;      ///< Lifecycle state flag.
};

/**
 * @brief Internal Phase-Steered Parallel Multi-head router (matching gcso_pspm_router_opaque).
 * Aligned to 16 bytes.
 */
struct GCSO_ALIGNAS(16) gcso_pspm_router_opaque {
    gcso_pspm_config_t config;   ///< Sub-head group allocation configuration.
    bool is_active;              ///< Lifecycle state flag.
};

/**
 * @brief Internal Sparse Residual Adapter Layer layout (matching gcso_srl_adapter_opaque).
 * Aligned to 32 bytes.
 */
struct GCSO_ALIGNAS(32) gcso_srl_adapter_opaque {
    gcso_srl_descriptor_t descriptor;   ///< Rank-1 outer product descriptor layout.
    bool is_active;                     ///< Lifecycle state flag.
};

/**
 * @brief Internal Entropy-Driven Decoding Branch Controller layout (matching gcso_edbc_controller_opaque).
 * Aligned to 32 bytes.
 */
struct GCSO_ALIGNAS(32) gcso_edbc_controller_opaque {
    gcso_edbc_state_t state;   ///< Active dynamic entropy controller state tracking.
    bool is_active;            ///< Lifecycle state flag.
};

/**
 * @brief Internal binary container instance layout (matching gcso_container_opaque).
 * Aligned to 16 bytes.
 */
struct GCSO_ALIGNAS(16) gcso_container_opaque {
    uint32_t container_id;   ///< Unique identifier for binary snapshot handle.
    bool is_active;          ///< Lifecycle state flag.
};

#endif // LIMINIKA_GCSO_INTERNAL_H