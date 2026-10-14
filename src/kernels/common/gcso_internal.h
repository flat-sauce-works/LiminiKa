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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

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

// Internal implementation aliases for strict backward layout compatibility
using gcso_context_impl = gcso_context_opaque;
using gcso_action_hub_impl = gcso_action_hub_opaque;
using gcso_attractor_field_impl = gcso_attractor_field_opaque;
using gcso_dpsr_kernel_impl = gcso_dpsr_kernel_opaque;
using gcso_pspm_router_impl = gcso_pspm_router_opaque;
using gcso_srl_adapter_impl = gcso_srl_adapter_opaque;
using gcso_edbc_controller_impl = gcso_edbc_controller_opaque;
using gcso_container_impl = gcso_container_opaque;

#endif // LIMINIKA_GCSO_INTERNAL_H