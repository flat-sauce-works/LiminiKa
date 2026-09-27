# GCSO C-ABI Specification (`c_abi_spec.md`)

This document defines the C-ABI (Application Binary Interface) specifications for the **GCSO (Geometric Cellular Sheaf Orchestrator)** core runtime within the LiminiKa project.

The C-ABI serves as the explicit, non-intrusive boundary layer connecting high-level orchestrators (Rust Core / LiminiKa DSL Compiler) and low-level compute kernels (C++20 / CUDA / Metal / Vulkan / CPU), as well as enabling direct sidecar integration with third-party inference engines (e.g., llama.cpp, vLLM, TensorRT-LLM).

---

## 1. General Principles

1. **Permissive & Clean IP Compliance & Isolation Boundary**
   * All code interfaces, exported headers, C/C++ implementations, build scripts, and test snippets strictly adhere to permissive dual-licensing (**MIT OR Apache-2.0** for source code and headers, **CC BY 4.0** for specification documentation).
   * Clean-room implementation is strictly enforced: no source code, proprietary algorithms, internal header structures, or routines from copyleft projects (GPL, LGPL, AGPL, SSPL) or third-party inference engines (llama.cpp, vLLM, Hugging Face transformers, etc.) are copied, ported, referenced, or adapted.
   * To prevent license contamination and build dependency coupling, GCSO headers never `#include` or reference third-party inference engine header files. All sidecar integrations operate strictly via raw pointer tables (`GCSOSidecarTable`).
   * Every header file, source file, export macro, and code snippet must explicitly declare `// SPDX-License-Identifier: MIT OR Apache-2.0` in its header comment block.
   * Runtime engine functions remain completely model-agnostic, operating strictly on raw tensor memory buffers without embedded dependencies on proprietary weight binaries or model-specific licensing restrictions.

2. **Action Hub & Pointer-Chain Stigmergy (Intelligence as a Chain of Pointers & Attractor Steering)**
   * The GCSO runtime operates under the core philosophical premise that **intelligence emerges from structured chains of pointers**.
   * Directly executing macro cellular sheaf operations or full Hodge decompositions on every token step introduces prohibitive computational overhead for target environments (2–4 GB VRAM). To bypass this while retaining the ability to link hallucination boundaries across phase spaces, GCSO treats the execution trajectory of pointers within the Sidecar Pointer Table (`GCSOSidecarTable`: SPT) as **Bottom-Up Stigmergy**.
   * As pointer operations transition during token decoding, their trajectories leave lightweight environmental traces in the Stigmergic Medium (`stigmergic_medium_ptr`). Higher-level macro rules observe these trajectory traces and steer the pointers into target Attractor Basins via lock-free atomic pointer swapping (`next_spt`, `phase_offsets_q7`), or apply Phase-Conjugate Repulsion to escape spurious local minima without locking the Hot Path, achieving complex topological steering at $\mathcal{O}(1)$ execution cost.
   * Binding to a third-party host inference engine (via raw tensor pointers like `query_ptr`, `key_ptr`, `logits_ptr`) represents merely *one specific execution action slot* within this broader action hub. Individual tensor pointer slots in `GCSOSidecarTable` are optional; if set to `NULL`, the runtime gracefully skips that specific execution action (no-op) and returns `GCSO_SUCCESS` without error.

3. **C++20 Standard Baseline with Cross-Platform & Restricted Environment Resilience**
   * Modern C++20 features (such as `std::atomic_ref`, `std::bit_cast`, compile-time concepts, `constexpr`, and `[[likely]]` / `[[unlikely]]` attribute specifiers) form the primary implementation baseline for low-level compute kernels and FFI bridges.
   * To guarantee seamless integration across legacy environments, C11 FFI layers, restricted embedded toolchains, and varied compilers (GCC, Clang, MSVC, NVCC, Metal Shader Compiler), all platform-dependent extensions are encapsulated using universal macro abstractions (`GCSO_ALIGNAS`, `GCSO_NOEXCEPT`, `GCSO_CONSTEXPR`, `GCSO_NODISCARD`, `GCSO_INLINE`, `GCSO_RESTRICT`, `GCSO_LIKELY`, `GCSO_UNLIKELY`).
   * When performing lock-free pointer swaps across thread or FFI boundaries, memory ordering strictly observes Acquire-Release semantics via C++20 `std::atomic_ref` (where `__cpp_lib_atomic_ref` is available) or C11 `<stdatomic.h>` / atomic intrinsics.

4. **Fractal Multi-Granularity Cascading (Nano / Micro / Mezzo / Macro Layers)**
   * The `GCSOSidecarTable` functions as a fractal node bridging execution granularities and bottom-up trajectory steering:
     * **Nano Level (`GCSO_GRANULARITY_NANO = 0`)**: Evaluates in-register RIPA clamping and updates local bitmasks within 32-thread warps.
     * **Micro Level (`GCSO_GRANULARITY_MICRO = 1`)**: Performs $\mathcal{O}(1)$ SPT index lookups (`gcso_spt_hash_slot_index`), fused DPSR phase additions, and warp bitmask reductions recorded in `local_state_mask`.
     * **Mezzo Level (`GCSO_GRANULARITY_MEZZO = 2`)**: Aggregates PagedBlock states across 16–32 token chunks via hierarchical bit-tree operations propagating through `parent_spt` / `child_spt_array`.
     * **Macro Level (`GCSO_GRANULARITY_MACRO = 3`)**: Asynchronously evaluates Moving Z-Score entropy ( $\tilde{H}$ ), tracking pointer trajectories as bottom-up stigmergic environment updates (`GCSO_ACTION_ID_STIGMERGIC_TRACE`) and triggering atomic pointer swaps on lower-level SPT nodes (`GCSO_ACTION_ID_ATTRACTOR_STEER`, `GCSO_ACTION_ID_PHASE_CONJUGATE_REPEL`) to steer global trajectories toward target attractors without locking the Hot Path.
   * To prevent stack overflow and deadlock caused by circular pointer links in dynamic environments, cascade evaluation functions strictly enforce a maximum recursion/traversal depth (`GCSO_MAX_CASCADE_DEPTH = 16`).

5. **Zero-Allocation in the Hot Path, 64-bit Assumptions & Cache-Line Alignment**
   * Functions invoked during the per-token decoding loop (Hot Path) must never execute dynamic memory allocations (`malloc`, `free`, `new`, `delete`), OS blocking system calls, or thread instantiation.
   * All working memory buffers, Sidecar Pointer Table (SPT) lookups, and phase rotation arrays must be completely pre-allocated and bound during the Cold Path initialization phase.
   * `GCSOSidecarTable` is strictly structured to an exact 128-byte layout with 64-byte alignment (matching 2 CPU/GPU cache lines) on 64-bit architectures (x86_64, AArch64), completely preventing cache-line splits and false sharing during high-speed thread-warp traversals. Layout integrity is validated at compile-time via C11 `_Static_assert` / C++20 `static_assert`.
   * On successful execution paths, Hot Path functions must complete without writing to Thread-Local Storage (TLS) or modifying global error state to prevent unnecessary memory bus traffic and L1/L2 cache-line invalidation.

6. **Exception Boundary Safety, Explicit Ownership & Symbol Visibility**
   * All C API exports are wrapped in `extern "C"` blocks, declared with explicit calling conventions via `GCSO_CALL`, and exported using the `GCSO_API` visibility macro.
   * C++20 ABI implementations must be declared `noexcept` via `GCSO_NOEXCEPT`. Internal C++ exceptions must be caught using `try-catch` blocks, setting thread-local error messages via `gcso_set_last_error_message()`, and mapping to appropriate `GCSOStatus` error codes before crossing FFI boundaries.
   * Mandatory input arguments receiving `NULL` when non-null is required must return `GCSO_ERROR_NULL_POINTER` safely without causing undefined behavior. Unaligned pointer inputs must return `GCSO_ERROR_UNALIGNED_POINTER`.
   * Rust FFI entry points invoking C-ABI routines must wrap executions with `std::panic::catch_unwind` to catch unwinding panics across language boundaries and map them to `GCSO_ERROR_PANIC_CAUGHT`.
   * Every opaque handle allocated by C/C++ must have a dedicated, paired free function (e.g., `gcso_context_free`, `gcso_edbc_free`, `gcso_container_close`, `gcso_swarm_cell_free`, `gcso_spt_unbind`). All free functions safely accept `NULL` pointers as no-ops, returning `GCSO_SUCCESS`.

7. **Header File Structure**
   * The specifications in this document correspond directly to public headers under `include/liminika/`:
     * `include/liminika/gcso_types.h`: Primitive types, cross-platform/C++20 macros, status codes, error setters/getters/clearers, SPT integrated action hub definitions, initialization macros, and opaque handle declarations.
     * `include/liminika/gcso_config.h`: Configuration parameters and layout specifications (`GCSOConfig`).
     * `include/liminika/gcso_abi.h`: Main C-ABI function prototypes and exported symbol declarations.

---

## 2. Macro Definitions, Types & Status Codes (`gcso_types.h`)

```c
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_TYPES_H
#define LIMINIKA_GCSO_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(GCSO_BUILD_DLL)
    #define GCSO_API __declspec(dllexport)
  #else
    #define GCSO_API __declspec(dllimport)
  #endif
  #define GCSO_CALL __stdcall
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define GCSO_API __attribute__((visibility("default")))
  #else
    #define GCSO_API
  #endif
  #define GCSO_CALL
#endif

// Check for C++20 baseline
#if defined(__cplusplus)
  #if __cplusplus >= 202002L || (defined(_MSVC_LANG) && _MSVC_LANG >= 202002L)
    #define GCSO_CPP20_OR_LATER 1
  #endif
#endif

// Cross-language alignment, constexpr, branch prediction, and exception compatibility macros for C11 and C++20
#ifdef __cplusplus
  #define GCSO_NOEXCEPT noexcept
  #define GCSO_ALIGNAS(x) alignas(x)
  #define GCSO_CONSTEXPR constexpr
  #if defined(GCSO_CPP20_OR_LATER)
    #define GCSO_LIKELY(x)   (x) [[likely]]
    #define GCSO_UNLIKELY(x) (x) [[unlikely]]
    #define GCSO_NODISCARD   [[nodiscard]]
  #else
    #if defined(__GNUC__) || defined(__clang__)
      #define GCSO_LIKELY(x)   (__builtin_expect(!!(x), 1))
      #define GCSO_UNLIKELY(x) (__builtin_expect(!!(x), 0))
    #else
      #define GCSO_LIKELY(x)   (x)
      #define GCSO_UNLIKELY(x) (x)
    #endif
    #if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
      #define GCSO_NODISCARD [[nodiscard]]
    #else
      #define GCSO_NODISCARD
    #endif
  #endif
  #if defined(_MSC_VER)
    #define GCSO_INLINE __forceinline
    #define GCSO_RESTRICT __restrict
  #else
    #define GCSO_INLINE inline __attribute__((always_inline))
    #define GCSO_RESTRICT __restrict__
  #endif
#else
  #define GCSO_NOEXCEPT
  #define GCSO_CONSTEXPR
  #define GCSO_NODISCARD
  #if defined(__GNUC__) || defined(__clang__)
    #define GCSO_LIKELY(x)   (__builtin_expect(!!(x), 1))
    #define GCSO_UNLIKELY(x) (__builtin_expect(!!(x), 0))
    #define GCSO_INLINE inline __attribute__((always_inline))
    #define GCSO_RESTRICT __restrict__
  #elif defined(_MSC_VER)
    #define GCSO_LIKELY(x)   (x)
    #define GCSO_UNLIKELY(x) (x)
    #define GCSO_INLINE __forceinline
    #define GCSO_RESTRICT __restrict
  #else
    #define GCSO_LIKELY(x)   (x)
    #define GCSO_UNLIKELY(x) (x)
    #define GCSO_INLINE inline
    #define GCSO_RESTRICT
  #endif
  #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #include <stdalign.h>
    #define GCSO_ALIGNAS(x) alignas(x)
  #elif defined(_MSC_VER)
    #define GCSO_ALIGNAS(x) __declspec(align(x))
  #elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_ALIGNAS(x) __attribute__((aligned(x)))
  #else
    #define GCSO_ALIGNAS(x)
  #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GCSO_ABI_VERSION_MAJOR 0
#define GCSO_ABI_VERSION_MINOR 2
#define GCSO_ABI_VERSION_PATCH 6

#define GCSO_MAKE_VERSION(major, minor, patch) \
    (((uint32_t)(major) << 16) | ((uint32_t)(minor) << 8) | ((uint32_t)(patch)))

#define GCSO_ABI_VERSION_COMBINED \
    GCSO_MAKE_VERSION(GCSO_ABI_VERSION_MAJOR, GCSO_ABI_VERSION_MINOR, GCSO_ABI_VERSION_PATCH)

#define GCSO_SPT_STRUCT_VERSION 1U
#define GCSO_MAX_CASCADE_DEPTH  16U

#define GCSO_Q7_SCALE 0.0078125f // 1.0f / 128.0f (Q7 Fixed-point scale factor beta_Q7)
#define GCSO_MAX_HEAD_DIM 256
#define GCSO_PAGED_BLOCK_SIZE 32
#define GCSO_SPT_MAX_SLOTS 256

// Granularity Level Definitions for Fractal Multi-Layer Structure
#define GCSO_GRANULARITY_NANO  0U // Warp / SIMD Level
#define GCSO_GRANULARITY_MICRO 1U // Token / PagedBlock Level
#define GCSO_GRANULARITY_MEZZO 2U // Chunk / Cellular Hub Level
#define GCSO_GRANULARITY_MACRO 3U // Session / Attractor Level

// Action ID Bitmask Flags for Action Hub Dispatch (Can be bitwise OR-ed together)
#define GCSO_ACTION_ID_NOP                   0x0000U
#define GCSO_ACTION_ID_DPSR_STEERING        0x0001U
#define GCSO_ACTION_ID_LOGIT_SHIFT          0x0002U
#define GCSO_ACTION_ID_PSPM_ROUTING         0x0004U
#define GCSO_ACTION_ID_SRL_UPDATE           0x0008U
#define GCSO_ACTION_ID_STIGMERGIC_TRACE      0x0010U // Bottom-up pointer trajectory trace
#define GCSO_ACTION_ID_ATTRACTOR_STEER       0x0020U // Dynamic attractor basin steering
#define GCSO_ACTION_ID_PHASE_CONJUGATE_REPEL 0x0040U // Anti-phase repulsion for hallucinatory local minima
#define GCSO_ACTION_ID_DNP_PACKET_SYNC      0x0100U
#define GCSO_ACTION_ID_USER_CUSTOM_BASE      0x1000U // Reserved base for custom DSL user actions

// Tensor Data Type Enumeration for Multi-Precision Host Engines
typedef uint16_t gcso_tensor_type_t;
#define GCSO_TENSOR_TYPE_FP32  0U
#define GCSO_TENSOR_TYPE_FP16  1U
#define GCSO_TENSOR_TYPE_BF16  2U
#define GCSO_TENSOR_TYPE_FP8   3U

typedef int8_t   gcso_q7_t;      // Q7 fixed-point phase offset representation
typedef uint32_t gcso_slot_id_t; // Sidecar Pointer Table (SPT) slot index

// Execution Outcome Status Codes (Underlying type uint32_t for ABI stability)
typedef uint32_t GCSOStatus;
typedef uint32_t LiminiKaStatus; // Unified standard type alias for backwards compatibility

#define GCSO_SUCCESS                         0U
#define GCSO_ERROR_INVALID_ARGUMENT          1U
#define GCSO_ERROR_OUT_OF_MEMORY             2U
#define GCSO_ERROR_BUFFER_TOO_SMALL          3U
#define GCSO_ERROR_NOT_INITIALIZED           4U
#define GCSO_ERROR_PANIC_CAUGHT              5U
#define GCSO_ERROR_INTERNAL_EXCEPTION        6U
#define GCSO_ERROR_IO_FAILURE                7U
#define GCSO_ERROR_UNSUPPORTED_HARDWARE      8U
#define GCSO_ERROR_ABI_MISMATCH              9U
#define GCSO_ERROR_NULL_POINTER              10U
#define GCSO_ERROR_INVALID_STATE             11U
#define GCSO_ERROR_SIDECAR_NOT_BOUND         12U
#define GCSO_ERROR_ALIGNMENT_MISMATCH        13U
#define GCSO_ERROR_CONTAINER_CORRUPT         14U
#define GCSO_ERROR_UNALIGNED_POINTER         15U
#define GCSO_ERROR_ACTION_DISPATCH_FAILED    16U
#define GCSO_ERROR_RECURSION_LIMIT_EXCEEDED  17U // Exceeded GCSO_MAX_CASCADE_DEPTH limit
#define GCSO_ERROR_ATOMIC_SWAP_FAILED        18U // Atomic lock-free pointer swap operation failed
#define GCSO_ERROR_OUT_OF_RANGE              19U // Index or memory offset out of range
#define GCSO_ERROR_KEY_NOT_FOUND             20U // Target key or track not found in container

// Forward declaration for chained SPT nodes
typedef struct GCSOSidecarTable GCSOSidecarTable;

// Action Dispatch Function Pointer Hook Prototype
typedef GCSOStatus (GCSO_CALL *GCSOActionDispatchFn)(
    GCSOSidecarTable* spt,
    uint32_t action_id,
    void* action_payload,
    void* user_data
);

// Local Micro/Mezzo Evaluation Hook Prototype
typedef GCSOStatus (GCSO_CALL *GCSOEvalDispatchFn)(
    const GCSOSidecarTable* spt,
    uint64_t state_flags,
    float* out_score
);

// Sidecar Pointer Table (SPT) - Integrated Action Hub & Fractal Node Layout
// Exactly 128 bytes with 64-byte alignment (2 Cache Lines on 64-bit architectures)
typedef struct GCSO_ALIGNAS(64) GCSOSidecarTable {
    // --- [Block 1: Host Inference Action Tensor Slots] (32 bytes, Offset 0..31) ---
    union {
        struct {
            void*           query_ptr;             // Host engine Query tensor buffer pointer (8 bytes, Offset 0)
            void*           key_ptr;               // Host engine Key tensor buffer pointer (8 bytes, Offset 8)
            void*           value_ptr;             // Host engine Value tensor buffer pointer (8 bytes, Offset 16)
            void*           logits_ptr;            // Host engine Logit output buffer pointer (8 bytes, Offset 24)
        };
        void*               action_slots[4];       // Generic action slot array for versatile hub access (32 bytes, Offset 0..31)
    };

    // --- [Block 2: Phase Steering & Stigmergic Hub Slot] (32 bytes, Offset 32..63) ---
    const int8_t*           phase_offsets_q7;      // Q7 phase offset lookup buffer (atomic pointer swap target) (8 bytes, Offset 32)
    void*                   stigmergic_medium_ptr; // Pointer to Stigmergic Medium / .gcso Container Hub (8 bytes, Offset 40)
    uint32_t                stride_layer;          // Byte stride between layer activations (4 bytes, Offset 48)
    uint32_t                num_phase_slots;       // Valid slots count in phase_offsets_q7 buffer (4 bytes, Offset 52)
    uint16_t                struct_version;        // SPT layout version identifier (2 bytes, Offset 56)
    uint16_t                tensor_data_type;      // Tensor data precision (gcso_tensor_type_t) (2 bytes, Offset 58)
    uint16_t                stride_head;           // Byte stride between query head activations (2 bytes, Offset 60)
    uint16_t                stride_kv_head;        // Byte stride between KV head activations for GQA (2 bytes, Offset 62)

    // --- [Block 3: Dispatch Hooks & Atomic State] (32 bytes, Offset 64..95) ---
    uint32_t                granularity_level;     // Granularity Tier: 0=Nano, 1=Micro, 2=Mezzo, 3=Macro (4 bytes, Offset 64)
    uint32_t                num_children;          // Number of active child SPT nodes in child_spt_array (4 bytes, Offset 68)
    GCSOActionDispatchFn    action_dispatch_fn;    // Event dispatch hook function pointer (8 bytes, Offset 72)
    GCSOEvalDispatchFn      eval_dispatch_fn;      // Local macro evaluation function pointer (8 bytes, Offset 80)
    uint64_t                local_state_mask;      // Atomic bitmask recording local evaluation state (8 bytes, Offset 88)

    // --- [Block 4: Fractal Cascade & Pointer Chain Hub] (32 bytes, Offset 96..127) ---
    GCSOSidecarTable*       next_spt;              // Next SPT node in horizontal chain (atomic pointer swap target) (8 bytes, Offset 96)
    GCSOSidecarTable*       parent_spt;            // Pointer to parent macro SPT node (8 bytes, Offset 104)
    GCSOSidecarTable**      child_spt_array;       // Pointer to array of child micro SPT node pointers (8 bytes, Offset 112)
    void*                   user_data;             // User context passed to dispatch hooks (8 bytes, Offset 120)
} GCSOSidecarTable;

// Static compile-time verification of struct layout (128 bytes size, 64 bytes alignment)
#ifdef __cplusplus
  static_assert(sizeof(GCSOSidecarTable) == 128, "GCSOSidecarTable size must be exactly 128 bytes");
  static_assert(alignof(GCSOSidecarTable) == 64, "GCSOSidecarTable alignment must be exactly 64 bytes");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  _Static_assert(sizeof(GCSOSidecarTable) == 128, "GCSOSidecarTable size must be exactly 128 bytes");
  _Static_assert(_Alignof(GCSOSidecarTable) == 64, "GCSOSidecarTable alignment must be exactly 64 bytes");
#endif

// Default Static Initialization Macro for GCSOSidecarTable
#define GCSO_SIDECAR_TABLE_INIT { \
    { { NULL, NULL, NULL, NULL } }, \
    NULL, NULL, 0U, 0U,             \
    GCSO_SPT_STRUCT_VERSION,        \
    GCSO_TENSOR_TYPE_FP16,          \
    0U, 0U,                         \
    GCSO_GRANULARITY_MICRO,         \
    0U, NULL, NULL, 0ULL,           \
    NULL, NULL, NULL, NULL          \
}

// Opaque handle declarations
typedef struct GCSOContext GCSOContext;
typedef struct GCSOEdbcState GCSOEdbcState;
typedef struct GCSOContainer GCSOContainer;
typedef struct GCSOSwarmCell GCSOSwarmCell;

// Thread-local error string accessors
GCSO_API const char* GCSO_CALL gcso_get_last_error_message(void) GCSO_NOEXCEPT;
GCSO_API void        GCSO_CALL gcso_set_last_error_message(const char* msg) GCSO_NOEXCEPT;
GCSO_API void        GCSO_CALL gcso_clear_last_error_message(void) GCSO_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#endif // LIMINIKA_GCSO_TYPES_H

```

---

## 3. Configuration Struct & Memory Layout (`gcso_config.h`)

To guarantee binary compatibility across C11, C++20, Rust FFI, and sidecar host applications, `GCSOConfig` includes an explicit `struct_version` at Offset 0 (set to `GCSO_CONFIG_STRUCT_VERSION`), accompanied by explicit padding (`_reserved[9]` from Offset 39 to 47) to guarantee an exact 48-byte layout with 8-byte alignment, supporting GQA (`num_kv_heads`) and SRL FFN dimensions (`ffn_dim`).

```c
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_CONFIG_H
#define LIMINIKA_GCSO_CONFIG_H

#include "gcso_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GCSO_CONFIG_STRUCT_VERSION GCSO_ABI_VERSION_COMBINED

typedef struct GCSO_ALIGNAS(8) GCSOConfig {
    uint32_t struct_version;      // Offset 0, Size 4 (GCSO_CONFIG_STRUCT_VERSION check)
    uint32_t num_layers;          // Offset 4, Size 4
    uint32_t num_heads;           // Offset 8, Size 4 (Query heads count)
    uint32_t num_kv_heads;        // Offset 12, Size 4 (Key/Value heads count for GQA)
    uint32_t head_dim;            // Offset 16, Size 4 (d_head, must be an even number)
    uint32_t ffn_dim;             // Offset 20, Size 4 (FFN intermediate dim for SRL)
    uint32_t max_context_length;  // Offset 24, Size 4
    uint32_t vocab_size;          // Offset 28, Size 4 (Model vocabulary size)
    float    ripa_clamp_rad;      // Offset 32, Size 4 (RIPA limit, default ~0.087 rad)
    uint8_t  enable_srl;          // Offset 36, Size 1 (1 = true, 0 = false)
    uint8_t  enable_edbc;         // Offset 37, Size 1 (1 = true, 0 = false)
    uint8_t  enable_sidecar_mode; // Offset 38, Size 1 (1 = true, 0 = false)
    uint8_t  _reserved[9];        // Offset 39, Size 9 (Explicit padding for 48-byte struct alignment)
} GCSOConfig;

// Static compile-time verification of GCSOConfig layout (48 bytes size, 8 bytes alignment)
#ifdef __cplusplus
  static_assert(sizeof(GCSOConfig) == 48, "GCSOConfig size must be exactly 48 bytes");
  static_assert(alignof(GCSOConfig) == 8, "GCSOConfig alignment must be exactly 8 bytes");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
  _Static_assert(sizeof(GCSOConfig) == 48, "GCSOConfig size must be exactly 48 bytes");
  _Static_assert(_Alignof(GCSOConfig) == 8, "GCSOConfig alignment must be exactly 8 bytes");
#endif

// Default Static Initialization Macro for GCSOConfig
#define GCSO_CONFIG_INIT { \
    GCSO_CONFIG_STRUCT_VERSION, \
    0U, 0U, 0U, 0U, 0U, 0U, 0U, \
    0.0872664626f,              \
    0U, 0U, 0U,                 \
    {0}                         \
}

#ifdef __cplusplus
}
#endif

#endif // LIMINIKA_GCSO_CONFIG_H

```

---

## 4. C-ABI Function References (`gcso_abi.h`)

### 4.1 System Utilities & Versioning [Cold Path]

#### `gcso_get_version`

Retrieves the runtime major, minor, and patch version numbers.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API void GCSO_CALL gcso_get_version(
    uint32_t* major,              // [out] Major version
    uint32_t* minor,              // [out] Minor version
    uint32_t* patch               // [out] Patch version
) GCSO_NOEXCEPT;

```

#### `gcso_check_abi_version`

Verifies compatibility between the caller and runtime ABI versions.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_check_abi_version(
    uint32_t major,               // [in] Expected major version
    uint32_t minor                // [in] Expected minor version
) GCSO_NOEXCEPT;

```

---

### 4.2 Lifecycle & Configuration [Cold Path]

#### `gcso_init`

Initializes the GCSO runtime environment and pre-allocates execution buffers for standalone or sidecar mode.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_init(
    const GCSOConfig* config,   // [in]  Pointer to runtime configuration
    GCSOContext**     out_ctx   // [out] Pointer to allocated GCSOContext handle
) GCSO_NOEXCEPT;

```

#### `gcso_context_free`

Destroys the GCSO runtime environment and releases all allocated memory. Accepts `NULL` gracefully as a no-op returning `GCSO_SUCCESS`.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_context_free(
    GCSOContext* ctx            // [in, nullable] Handle to destroy
) GCSO_NOEXCEPT;

```

---

### 4.3 Dynamic Phase-Shifted RoPE (DPSR) Kernels & Tensor Operations [Nano / Micro Hot Path Execution]

> **Strict Rule**: All functions in Section 4.3 execute inside the hot path decoding loop and must operate with **zero dynamic allocations (`malloc`/`free`/`new`/`delete`)** and **zero blocking system calls**. Successful execution paths must not write to TLS. If `query_ptr` or required tensor pointers are `NULL`, functions return `GCSO_SUCCESS` immediately as a safe no-op.

#### `gcso_dpsr_init` [Cold Path]

Initializes internal DPSR execution pipelines and binds pre-allocated memory buffers.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_dpsr_init(
    GCSOContext*      ctx,      // [in, out] Active GCSO context
    const GCSOConfig* config    // [in]      Configuration reference
) GCSO_NOEXCEPT;

```

#### `gcso_dpsr_apply_phase_steering` [Nano / Micro Hot Path]

Applies relative phase shift (Lazy Phase Unwrapping) inline to Query tensor registers of standalone or sidecar host engines.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_dpsr_apply_phase_steering(
    GCSOContext*     ctx,             // [in, out] Active context
    void*            query_ptr,       // [in, out, nullable] Query tensor buffer
    const gcso_q7_t* phase_delta_q7,  // [in]      Q7 phase differential array
    uint32_t         seq_len,         // [in]      Sequence length
    uint32_t         layer_idx,       // [in]      Target layer index
    uint32_t         head_idx         // [in]      Target head index
) GCSO_NOEXCEPT;

```

#### `gcso_dpsr_apply_phase_steering_safe` [Nano / Micro Hot Path]

Applies Restricted Inline Phase Alignment (RIPA) with $\tanh$ soft-clamping restricted to low-frequency channels ($d_{\mathrm{head}}/4$).

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_dpsr_apply_phase_steering_safe(
    GCSOContext*     ctx,             // [in, out] Active context
    void*            query_ptr,       // [in, out, nullable] Query tensor buffer
    const gcso_q7_t* phase_delta_q7,  // [in]      Q7 phase differential array
    float            clamp_threshold, // [in]      RIPA clamp rad threshold
    uint32_t         layer_idx,       // [in]      Target layer index
    uint32_t         head_idx         // [in]      Target head index
) GCSO_NOEXCEPT;

```

#### `gcso_dpsr_lazy_unwrap_override` [Micro Hot Path]

Overrides cumulative context phase delta on Query tensors without re-rotating KV cache vectors.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_dpsr_lazy_unwrap_override(
    GCSOContext*     ctx,                     // [in, out] Active context
    void*            query_ptr,               // [in, out, nullable] Query tensor buffer
    const gcso_q7_t* context_accum_phase_q7, // [in]      Accumulated context phase
    uint32_t         layer_idx,               // [in]      Target layer index
    uint32_t         head_idx                 // [in]      Target head index
) GCSO_NOEXCEPT;

```

#### `gcso_dpsr_apply_soft_phase_damping` [Micro Hot Path]

Applies soft phase damping to mitigate output divergence during phase transitions.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_dpsr_apply_soft_phase_damping(
    GCSOContext* ctx,           // [in, out] Active context
    void*        query_ptr,     // [in, out, nullable] Query tensor buffer
    float        damping_factor,// [in]      Damping scale [0.0, 1.0]
    uint32_t     layer_idx,     // [in]      Target layer index
    uint32_t     head_idx       // [in]      Target head index
) GCSO_NOEXCEPT;

```

#### `gcso_dpsr_slerp_norm_guard_stable` [Micro Hot Path]

Executes norm-guarded Slerp (Spherical Linear Interpolation) stabilization on phase vectors to prevent numeric underflow/overflow.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_dpsr_slerp_norm_guard_stable(
    const float* phase_a,       // [in]  Input phase vector A
    const float* phase_b,       // [in]  Input phase vector B
    float        t,             // [in]  Interpolation factor [0.0, 1.0]
    float*       out_phase,     // [out] Interpolated phase vector
    uint32_t     dim            // [in]  Dimension
) GCSO_NOEXCEPT;

```

#### `gcso_dpsr_fused_logit_shift` [Micro Hot Path]

Applies fused phase shift directly to logit output registers of the host engine immediately preceding token sampling.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_dpsr_fused_logit_shift(
    void*            logits,          // [in, out, nullable] Logit buffer
    const gcso_q7_t* phase_delta_q7,  // [in]      Q7 phase shift array
    uint32_t         vocab_size       // [in]      Vocabulary size
) GCSO_NOEXCEPT;

```

#### `gcso_dpsr_compute_procrustes_phase_delta` [Cold Path]

Computes the optimal phase rotation matrix alignment delta ( $\Delta\boldsymbol{\theta}$ ) via Procrustes analysis on latent manifolds.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_dpsr_compute_procrustes_phase_delta(
    const float* src_matrix,    // [in]  Source tensor matrix
    const float* tgt_matrix,    // [in]  Target tensor matrix
    uint32_t     rows,          // [in]  Matrix rows
    uint32_t     cols,          // [in]  Matrix columns
    gcso_q7_t*   out_delta_q7   // [out] Computed Q7 phase delta
) GCSO_NOEXCEPT;

```

---

### 4.4 Sidecar Pointer Table (SPT) Action Hub & Swarm Stigmergic Operations [Micro / Mezzo / Macro Path]

#### `gcso_spt_validate` [Cold Path / Sidecar Validation]

Validates that all action slots, host engine tensor pointers, and phase offset arrays in `GCSOSidecarTable` satisfy 64-byte alignment and 128-byte size boundaries before binding. Optional tensor pointers set to `NULL` are validated as valid no-op slots.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_validate(
    const GCSOSidecarTable* spt // [in] Pointer table configuration to validate
) GCSO_NOEXCEPT;

```

#### `gcso_spt_bind` [Cold Path / Sidecar Setup]

Binds an external sidecar host engine's buffer pointers and action hooks to the Sidecar Pointer Table (SPT) Action Hub for zero-copy phase steering and pointer chaining.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_bind(
    GCSOContext*            ctx,      // [in, out] Active context
    const GCSOSidecarTable* spt_in,   // [in]      Host engine binding configuration
    GCSOSidecarTable**      out_spt   // [out]     Allocated Sidecar Pointer Table handle
) GCSO_NOEXCEPT;

```

#### `gcso_spt_dispatch_action` [Micro / Macro Action Hub Dispatch]

Dispatches an action through the Integrated Action Hub (SPT) by traversing the pointer chain or invoking registered `action_dispatch_fn` hooks in $\mathcal{O}(1)$ time.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_dispatch_action(
    GCSOSidecarTable* spt,            // [in] Active Integrated Action Hub node
    uint32_t          action_id,      // [in] Bitmask flag for target action (e.g., GCSO_ACTION_ID_DPSR_STEERING)
    void*             action_payload  // [in, nullable] Payload associated with action
) GCSO_NOEXCEPT;

```

#### `gcso_spt_atomic_swap_next` [Macro Attractor Steering / Lock-Free Hot-Path Update]

Executes an atomic lock-free pointer swap on `next_spt` observing Acquire-Release semantics (via C++20 `std::atomic_ref`), redirecting the execution trajectory without locking the Hot Path thread.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_atomic_swap_next(
    GCSOSidecarTable* spt,            // [in, out] Active SPT node
    GCSOSidecarTable* new_next_spt    // [in, nullable] Target next SPT node pointer
) GCSO_NOEXCEPT;

```

#### `gcso_spt_atomic_swap_phase` [Macro Attractor Steering / Lock-Free Hot-Path Update]

Executes an atomic lock-free pointer swap on `phase_offsets_q7` observing Acquire-Release semantics (via C++20 `std::atomic_ref`), instantly modulating active phase vectors across tokens.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_atomic_swap_phase(
    GCSOSidecarTable* spt,            // [in, out] Active SPT node
    const int8_t*     new_phase_q7    // [in, nullable] Target Q7 phase offset array pointer
) GCSO_NOEXCEPT;

```

#### `gcso_spt_trace_stigmergic_trajectory` [Micro / Macro Bottom-Up Stigmergy Trace]

Aggregates pointer traversal steps through `GCSOSidecarTable` nodes during execution into a 64-bit trajectory hash and writes it into the environmental memory (Stigmergic Medium) in $\mathcal{O}(1)$ without allocating memory.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_trace_stigmergic_trajectory(
    GCSOSidecarTable* spt,            // [in] Active SPT node
    uint64_t*         out_traj_hash   // [out] Computed 64-bit trajectory hash ID
) GCSO_NOEXCEPT;

```

#### `gcso_spt_deposit_stigmergic_trace` [Micro / Macro Bottom-Up Pheromone Deposit]

Directly deposits a stigmergic trajectory trace and pheromone intensity weight into `stigmergic_medium_ptr` in $\mathcal{O}(1)$, reflecting micro execution steps as environmental field updates.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_deposit_stigmergic_trace(
    GCSOSidecarTable* spt,            // [in, out] Active SPT node
    uint64_t          trajectory_hash,// [in]      Computed 64-bit trajectory hash ID
    float             trace_weight,   // [in]      Pheromone deposit intensity weight [0.0, 1.0]
    uint32_t          decay_factor    // [in]      Evaporation decay rate parameter
) GCSO_NOEXCEPT;

```

#### `gcso_spt_steer_to_attractor` [Macro Attractor Basin Steering]

Performs atomic lock-free pointer swaps on `next_spt` or `phase_offsets_q7` based on target attractor potential values, pulling micro pointer execution trajectories into macro attraction basins.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_steer_to_attractor(
    GCSOSidecarTable* spt,             // [in, out] Active SPT node
    uint64_t          attractor_hash,  // [in] Target attractor trajectory ID
    float             attract_gain     // [in] Attraction gain scale [0.0, 1.0]
) GCSO_NOEXCEPT;

```

#### `gcso_spt_repel_from_hallucination` [Macro Phase-Conjugate Repulsion]

Applies Phase-Conjugate Attractor Repulsion ( $-\boldsymbol{\Delta\theta}_{\mathrm{hallucination}}$ ) when a spurious local minimum (hallucination) is detected, flipping the potential valley into a repulsive peak to guide pointer trajectories away without altering weights.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_repel_from_hallucination(
    GCSOSidecarTable* spt,             // [in, out] Active SPT node
    uint64_t          hallucination_id,// [in] Target spurious local minimum ID
    float             repel_gain       // [in] Repulsion strength scale [0.0, 1.0]
) GCSO_NOEXCEPT;

```

#### `gcso_spt_cascade_eval` [Mezzo / Macro Bottom-Up Evaluation]

Cascades local micro state masks through the fractal SPT pointer tree (`parent_spt`), executing `eval_dispatch_fn` to perform lock-free state reductions up to the macro layer. Traversal recursion depth is bounded by `GCSO_MAX_CASCADE_DEPTH` to prevent stack overflows.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_cascade_eval(
    GCSOSidecarTable* spt,            // [in] Active SPT node
    uint64_t          local_flags,    // [in] Micro evaluation flags
    float*            out_macro_score // [out] Aggregated macro score
) GCSO_NOEXCEPT;

```

#### `gcso_spt_unbind` [Cold Path / Sidecar Teardown]

Unbinds and releases the Sidecar Pointer Table (SPT) handle without invalidating referenced host engine memory buffers. Accepts `NULL` gracefully as a no-op returning `GCSO_SUCCESS`.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_spt_unbind(
    GCSOSidecarTable* spt             // [in, nullable] Handle to unbind
) GCSO_NOEXCEPT;

```

#### `gcso_swarm_cell_create` [Cold Path]

Allocates and initializes a new Swarm Cell instance for cellular topology management.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_swarm_cell_create(
    uint32_t        cell_id,    // [in]  Identifier for the swarm cell
    GCSOSwarmCell** out_cell    // [out] Allocated GCSOSwarmCell handle
) GCSO_NOEXCEPT;

```

#### `gcso_swarm_cell_chunk_step` [Micro / Mezzo Path]

Processes micro activation chunk steps for localized cellular topology coordinator execution.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_swarm_cell_chunk_step(
    GCSOContext*  ctx,            // [in, out] Active context
    uint32_t      chunk_id,       // [in]      Target activation chunk ID
    const float*  in_activations, // [in]      Activation buffer
    size_t        activation_len  // [in]      Length of activation array
) GCSO_NOEXCEPT;

```

#### `gcso_swarm_hub_reduce_bit_tree` [Mezzo Path]

Performs hierarchical bit-tree reduction across cellular hub blocks, cascading state up to parent hubs via `parent_spt`.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_swarm_hub_reduce_bit_tree(
    GCSOContext* ctx,           // [in, out] Active context
    uint64_t*    bit_tree_ptr,  // [in, out] Bit-tree structure buffer
    uint32_t     tree_depth     // [in]      Depth of the bit-tree
) GCSO_NOEXCEPT;

```

#### `gcso_swarm_cell_state_update_express` [Micro Path]

Executes expedited non-blocking state updates for cellular swarm agents.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_swarm_cell_state_update_express(
    GCSOSwarmCell* cell,        // [in, out] Swarm cell instance
    uint32_t       state_flags  // [in]      Express state flags
) GCSO_NOEXCEPT;

```

#### `gcso_swarm_paged_block_warp_bitmask` [Nano / Micro Path]

Executes warp-cooperative PagedBlock bitmask reduction and updates local swarm cell state in $\mathcal{O}(1)$.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_swarm_paged_block_warp_bitmask(
    GCSOContext* ctx,                  // [in, out] Active context
    uint32_t     block_id,             // [in]      PagedBlock ID
    uint64_t     in_bitmask,           // [in]      Input warp bitmask
    uint64_t*    out_reduced_bitmask   // [out]     Reduced bitmask output
) GCSO_NOEXCEPT;

```

#### `gcso_spt_hash_slot_index` [Nano / Micro Path]

Computes $\mathcal{O}(1)$ index lookup into Sidecar Pointer Table (SPT) for phase offsets and action chain targets.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API gcso_slot_id_t GCSO_CALL gcso_spt_hash_slot_index(
    uint32_t layer_idx,         // [in] Layer index
    uint32_t head_idx,          // [in] Head index
    uint32_t token_pos          // [in] Token sequence position
) GCSO_NOEXCEPT;

```

#### `gcso_swarm_cell_free` [Cold Path]

Frees the allocated Swarm Cell instance handle. Accepts `NULL` gracefully as a no-op returning `GCSO_SUCCESS`.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_swarm_cell_free(
    GCSOSwarmCell* swarm        // [in, nullable] Handle to destroy
) GCSO_NOEXCEPT;

```

---

### 4.5 EDBC & Dynamic Entropy Control [Macro Evaluation Path]

#### `gcso_edbc_init` [Cold Path]

Initializes EDBC state and Sliding-Window Entropy Rate Integrator.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_edbc_init(
    const GCSOConfig* config,   // [in]  Configuration reference
    GCSOEdbcState**   out_edbc   // [out] Allocated GCSOEdbcState handle
) GCSO_NOEXCEPT;

```

#### `gcso_edbc_eval_stateful` [Macro Path]

Evaluates Moving Z-Score Normalized Attention Entropy ( $\tilde{H}$ ) and determines branch routing (Exploration vs Convergence).

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_edbc_eval_stateful(
    GCSOEdbcState* edbc,                  // [in, out] Active EDBC state
    const void*    logits_ptr,            // [in, nullable] Logits buffer
    uint32_t       vocab_size,            // [in]      Vocabulary size
    float*         out_normalized_entropy,// [out]     Computed entropy (\tilde{H})
    bool*          out_trigger_tunneling  // [out]     Metastable transition flag
) GCSO_NOEXCEPT;

```

#### `gcso_edbc_cvoid_eval_dyadic128` [Macro Path]

Evaluates Coherent Vector Alignment Metric ( $\mathcal{C}_{\mathrm{void}}$ ) across 128-bit Dyadic key structures.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_edbc_cvoid_eval_dyadic128(
    GCSOEdbcState* edbc,            // [in, out] Active EDBC state
    const uint64_t dyadic_key[2],   // [in]      128-bit Dyadic key
    float*         out_cvoid_score  // [out]     Resulting alignment score
) GCSO_NOEXCEPT;

```

#### `gcso_edbc_cvoid_eval_barrier` [Macro Path]

Evaluates the Eyring-Kramers potential barrier deformation model for metastable state transitions.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_edbc_cvoid_eval_barrier(
    GCSOEdbcState* edbc,            // [in, out] Active EDBC state
    float          cvoid_score,     // [in]      Current C_void score
    float*         out_barrier_h    // [out]     Computed potential barrier height
) GCSO_NOEXCEPT;

```

#### `gcso_edbc_free` [Cold Path]

Frees the allocated EDBC state controller context. Accepts `NULL` gracefully as a no-op returning `GCSO_SUCCESS`.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_edbc_free(
    GCSOEdbcState* edbc          // [in, nullable] Handle to destroy
) GCSO_NOEXCEPT;

```

---

### 4.6 Sparse Residual Adapter Layer (SRL) Operations [Micro / Hot Path Execution]

#### `gcso_srl_dynamic_mlp_gate_eval` [Micro Hot Path]

Evaluates Dynamic Rank-1 Sparse Residual Adapter Layer (SRL) vector transformations.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_srl_dynamic_mlp_gate_eval(
    GCSOContext* ctx,             // [in, out] Active context
    const float* in_activation,   // [in]      Input activation array
    const float* vector_u,        // [in]      FP8 Vector U
    const float* vector_v,        // [in]      FP8 Vector V
    const float* scale_s,         // [in]      Gain scale vector S
    float*       out_activation,  // [out]     Output activation array
    uint32_t     dim_in,          // [in]      Input dimension
    uint32_t     dim_out          // [in]      Output dimension
) GCSO_NOEXCEPT;

```

---

### 4.7 `.gcso` Container, Memory, Persona & Sidecar Attachment Operations [Cold Path & Async I/O]

#### `gcso_container_open_mmap` [Cold Path]

Opens `.gcso` binary file or sidecar package via memory-mapping (Zero-Overhead In-Memory Mapped Storage: ZIMMS).

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_container_open_mmap(
    const char*     file_path,      // [in]  Path to .gcso container or .gcsopack file
    GCSOContainer** out_container   // [out] Opened container handle
) GCSO_NOEXCEPT;

```

#### `gcso_container_get_track` [Cold Path]

Retrieves a reference to a specific track (`CORE`, `I-CACHE`, `P-CACHE`, `SNAPSHOT`) from an opened container.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_container_get_track(
    GCSOContainer*  container,      // [in]  Opened container
    const char*     track_name,     // [in]  Track name string
    const uint8_t** out_data_ptr,   // [out] Pointer to track binary data
    size_t*         out_data_len    // [out] Track data byte length
) GCSO_NOEXCEPT;

```

#### `gcso_pprc_seek_to_token` [Cold Path]

Performs Zero-Forward Latency Seek (Instantaneous Replay) by restoring phase motion vectors from P-Cache.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_pprc_seek_to_token(
    GCSOContext*   ctx,             // [in, out] Active context
    GCSOContainer* container,       // [in]      Opened container
    uint32_t       target_token_pos // [in]      Target seek index
) GCSO_NOEXCEPT;

```

#### `gcso_mem_pushout_align` [Cold Path]

Aligns memory layout offsets for Direct-DMA Ring-Buffer offloading.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_mem_pushout_align(
    GCSOContext* ctx,               // [in, out] Active context
    size_t       alignment_bytes    // [in]      Alignment boundary (e.g., 64, 128)
) GCSO_NOEXCEPT;

```

#### `gcso_mem_stigmergic_offload` [Cold Path]

Offloads inactive KV phase traces directly into environmental memory containers.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_mem_stigmergic_offload(
    GCSOContext*   ctx,             // [in, out] Active context
    GCSOContainer* container        // [in, out] Target memory container
) GCSO_NOEXCEPT;

```

#### `gcso_persona_apply_patch` [Cold Path]

Applies dynamic persona profile patches onto the active runtime context or sidecar host engine.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_persona_apply_patch(
    GCSOContext*   ctx,             // [in, out] Active context
    const uint8_t* patch_data,      // [in]      Binary patch buffer
    size_t         patch_len        // [in]      Patch size in bytes
) GCSO_NOEXCEPT;

```

#### `gcso_dnp_dispatch_packet` [Cold Path]

Dispatches a Dynamic Node Protocol (GCSO-DNP) packet for swarm topology synchronization.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_dnp_dispatch_packet(
    GCSOContext*   ctx,             // [in, out] Active context
    const uint8_t* packet_buf,      // [in]      Serialized DNP packet
    size_t         packet_len       // [in]      Packet byte size
) GCSO_NOEXCEPT;

```

#### `gcso_container_close` [Cold Path]

Unmaps and closes an opened `.gcso` container or sidecar file. Accepts `NULL` gracefully as a no-op returning `GCSO_SUCCESS`.

```c
// SPDX-License-Identifier: MIT OR Apache-2.0
GCSO_API GCSOStatus GCSO_CALL gcso_container_close(
    GCSOContainer* container        // [in, nullable] Container handle to destroy
) GCSO_NOEXCEPT;

```

---

### 4.8 Sidecar Integration Lifecycle & Integrated Action Hooks

When integrated as a sidecar middleware alongside external inference engines (e.g., llama.cpp), execution follows a strict 5-stage non-intrusive lifecycle centered around the Action Hub:

1. **Initialization Phase [Cold Path]**:
Call `gcso_init()` with `enable_sidecar_mode = 1`. Validate host engine buffers and action slots via `gcso_spt_validate()`, then bind Query/Key/Value/Logit pointers, event dispatch hooks, and evaluation hooks to the SPT via `gcso_spt_bind()`.
2. **Query RoPE Execution Hook [Nano / Micro Hot Path]**:
Immediately prior to computing dot-product attention in each Transformer layer, invoke `gcso_dpsr_apply_phase_steering_safe()`. If `query_ptr` is bound, this performs in-place Q7 phase additions ( $\boldsymbol{\Delta\theta}$ ) on Query registers in $\mathcal{O}(1)$ without modifying KV cache vectors or host engine weight memory. If `query_ptr` is `NULL`, the hook safely skips steering without error.
3. **LM Head Logit Shift Hook [Micro Hot Path]**:
Prior to token sampling at the final layer, invoke `gcso_dpsr_fused_logit_shift()`. If `logits_ptr` is bound, this steers logit distributions according to persona profiles. If `logits_ptr` is `NULL`, it safely returns `GCSO_SUCCESS`.
4. **Post-Step Entropy Evaluation, Stigmergic Trace & Attractor Steering [Micro / Macro Path]**:
After token output, pass logits to `gcso_edbc_eval_stateful()` to update Moving Z-Score entropy $\tilde{H}$. Record pointer execution steps into the Stigmergic Medium via `gcso_spt_trace_stigmergic_trajectory()` or deposit trace weights via `gcso_spt_deposit_stigmergic_trace()`. If an entropy surge or hallucinatory local minimum is detected, trigger `gcso_spt_steer_to_attractor()`, `gcso_spt_repel_from_hallucination()`, or perform atomic lock-free pointer swaps (`gcso_spt_atomic_swap_next` / `gcso_spt_atomic_swap_phase`) to redirect pointer chain trajectories into target attractor basins or apply phase-conjugate repulsion (`GCSO_ACTION_ID_PHASE_CONJUGATE_REPEL`).
5. **Teardown Phase [Cold Path]**:
Upon session termination, invoke `gcso_spt_unbind()` to gracefully unbind handles without freeing host engine memory, followed by `gcso_context_free()`.
