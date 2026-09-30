// name: src/kernels/common/gcso_internal.h
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_INTERNAL_H
#define LIMINIKA_GCSO_INTERNAL_H

#include "liminika/gcso_abi.h"
#include "liminika/gcso_config.h"
#include "liminika/gcso_types.h"

#include <cstddef>
#include <new>
#include <utility>

// ===================================================================
// Over-aligned Safe Memory Allocation Helpers (C++17/C++20)
// ===================================================================

/**
 * @brief Safely allocates and constructs an over-aligned structure using std::align_val_t.
 *
 * Guarantees strict compliance with alignas(16), alignas(32), alignas(64), and alignas(128)
 * memory boundary requirements across heterogeneous architectures without throwing exceptions
 * across FFI boundaries. Explicitly deallocates memory if object construction throws an exception
 * to prevent resource leaks.
 *
 * @tparam T Structure type to allocate.
 * @tparam Args Argument types forwarded to constructor.
 * @param args Arguments forwarded to constructor.
 * @return Pointer to allocated and initialized object, or nullptr on allocation failure.
 */
template <typename T, typename... Args>
inline T* gcso_aligned_new(Args&&... args) GCSO_NOEXCEPT {
    void* ptr = ::operator new(sizeof(T), std::align_val_t{alignof(T)}, std::nothrow);
    if (GCSO_UNLIKELY(ptr == nullptr)) {
        return nullptr;
    }
    try {
        return new (ptr) T(std::forward<Args>(args)...);
    } catch (...) {
        ::operator delete(ptr, std::align_val_t{alignof(T)}, std::nothrow);
        return nullptr;
    }
}

/**
 * @brief Safely destroys and deallocates an over-aligned structure using std::align_val_t.
 *
 * Suppresses destructor exceptions to strictly maintain the GCSO_NOEXCEPT contract across
 * C-ABI and Rust FFI boundaries. Guarantees that memory is unconditionally freed even if
 * the object's destructor throws an exception.
 *
 * @tparam T Structure type to destroy.
 * @param ptr Pointer to instance to deallocate. Safe no-op if nullptr.
 */
template <typename T>
inline void gcso_aligned_delete(T* ptr) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(ptr == nullptr)) {
        return;
    }
    try {
        ptr->~T();
    } catch (...) {
        // Suppress any destruction exceptions to preserve GCSO_NOEXCEPT contract across FFI
    }
    // Deallocate memory unconditionally using std::nothrow to ensure no resource leaks
    ::operator delete(ptr, std::align_val_t{alignof(T)}, std::nothrow);
}

// ===================================================================
// Internal Opaque Implementation Structures for GCSO Kernel Engine
// ===================================================================

/**
 * @brief Internal runtime execution context layout.
 * Aligned to 64 bytes to fit CPU/GPU cache line boundaries without false sharing.
 */
struct GCSO_ALIGNAS(64) gcso_context_impl {
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
 * @brief Internal Action Hub (Sidecar Pointer Table - SPT) implementation structure.
 * Aligned to 32 bytes for optimized vector lookup and cacheline efficiency.
 */
struct GCSO_ALIGNAS(32) gcso_action_hub_impl {
    uint32_t capacity;         ///< Maximum tagged pointer slot capacity.
    uint32_t active_count;     ///< Current active pointer trail count.
    uint64_t last_trail_ptr;   ///< Address of most recently transitioned pointer trail.
    bool is_active;            ///< Lifecycle state flag.
};

/**
 * @brief Internal Attractor Field implementation structure.
 * Aligned to 32 bytes for topological anchor management.
 */
struct GCSO_ALIGNAS(32) gcso_attractor_field_impl {
    uint32_t anchor_count;    ///< Total number of active topological anchors.
    uint8_t anchors[256];     ///< Static feature space allocation for anchor vectors.
    bool is_active;           ///< Lifecycle state flag.
};

/**
 * @brief Internal Dynamic Phase-Shifted RoPE (DPSR) kernel state structure.
 * Aligned to 16 bytes.
 */
struct GCSO_ALIGNAS(16) gcso_dpsr_kernel_impl {
    uint32_t head_dim;   ///< Target attention head dimension.
    uint32_t num_heads;  ///< Total number of attention heads.
    bool is_active;      ///< Lifecycle state flag.
};

/**
 * @brief Internal Phase-Steered Parallel Multi-head (PSPM) router structure.
 * Aligned to 16 bytes.
 */
struct GCSO_ALIGNAS(16) gcso_pspm_router_impl {
    gcso_pspm_config_t config;   ///< Sub-head group allocation configuration.
    bool is_active;              ///< Lifecycle state flag.
};

/**
 * @brief Internal Sparse Residual Adapter Layer (SRL) instance layout.
 * Aligned to 32 bytes.
 */
struct GCSO_ALIGNAS(32) gcso_srl_adapter_impl {
    gcso_srl_descriptor_t descriptor;   ///< Rank-1 outer product descriptor layout.
    bool is_active;                     ///< Lifecycle state flag.
};

/**
 * @brief Internal Entropy-Driven Decoding Branch Controller (EDBC) layout.
 * Aligned to 32 bytes.
 */
struct GCSO_ALIGNAS(32) gcso_edbc_controller_impl {
    gcso_edbc_state_t state;   ///< Active dynamic entropy controller state tracking.
    bool is_active;            ///< Lifecycle state flag.
};

/**
 * @brief Internal binary container instance layout.
 * Aligned to 16 bytes.
 */
struct GCSO_ALIGNAS(16) gcso_container_impl {
    uint32_t container_id;   ///< Unique identifier for binary snapshot handle.
    bool is_active;          ///< Lifecycle state flag.
};

#endif // LIMINIKA_GCSO_INTERNAL_H