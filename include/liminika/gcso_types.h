// name: include/liminika/gcso_types.h
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_TYPES_H
    #define LIMINIKA_GCSO_TYPES_H

    #include "liminika/gcso_config.h"

GCSO_EXTERN_C_BEGIN

// Quantized 7-bit signed fixed-point integer (scale beta_Q7 = 1/128)
typedef int8_t gcso_q7_t;

// 64-bit feature bitmask capability flags
typedef uint64_t gcso_capability_flags_t;

/**
 * @brief Standardized 32-bit status return codes for 1:1 cross-language FFI mapping.
 */
typedef enum gcso_status {
    GCSO_SUCCESS = 0,
    GCSO_ERROR_INVALID_ARGUMENT = -1,
    GCSO_ERROR_OUT_OF_MEMORY = -2,
    GCSO_ERROR_BUFFER_TOO_SMALL = -3,
    GCSO_ERROR_PANIC_CAUGHT = -4,
    GCSO_ERROR_NULL_POINTER = -5,
    GCSO_ERROR_INVALID_STATE = -6,
    GCSO_ERROR_VERSION_MISMATCH = -7,
    GCSO_ERROR_MISALIGNED_POINTER = -8,
    GCSO_ERROR_IO_FAILURE = -9,
    GCSO_ERROR_ACTION_HUB_FULL = -10,
    GCSO_ERROR_NOT_IMPLEMENTED = -11,
    GCSO_ERROR_ATTRACTOR_NOT_FOUND = -20,
    GCSO_ERROR_DPSR_PHASE_OVERFLOW = -30,
    GCSO_ERROR_QDPS_UNDERFLOW = -31,
    GCSO_ERROR_EDBC_SINGULARITY = -40,
    GCSO_ERROR_CONTAINER_CORRUPTED = -50,
    GCSO_ERROR_ZIMMS_MAPPING_FAILED = -51,
    GCSO_ERROR_PSPM_ROUTING_FAILED = -60,
    GCSO_ERROR_PPRC_SEEK_FAILED = -70,
    GCSO_ERROR_OBSTRUCTION_UNRESOLVED = -80,
    GCSO_ERROR_EXTENSION_NOT_LOADED = -90,
    GCSO_ERROR_DAES_SCRATCHPAD_FULL = -91,
    GCSO_ERROR_UNKNOWN = -0x7FFFFFFF
} gcso_status_t;

/**
 * @brief Classification types for topological attractor field anchors.
 */
typedef enum gcso_anchor_type {
    GCSO_ANCHOR_TYPE_SYSTEM_PROMPT = 0,
    GCSO_ANCHOR_TYPE_EMBEDDING = 1,
    GCSO_ANCHOR_TYPE_PHASE_REPULSE = 2,
    GCSO_ANCHOR_TYPE_TOPOLOGICAL = 3,
    GCSO_ANCHOR_TYPE_CRYSTALLIZED = 4
} gcso_anchor_type_t;

/**
 * @brief Descriptor header for size and ABI version validation.
 */
typedef struct GCSO_ALIGNAS(4) gcso_descriptor_header {
    uint32_t struct_size;
    uint32_t abi_version;
} gcso_descriptor_header_t;

/**
 * @brief 256-bit bitmask layout aligned to 32 bytes for Warp/SIMD reductions.
 */
typedef struct GCSO_ALIGNAS(32) gcso_paged_bitmask {
    uint64_t bits[4];
} gcso_paged_bitmask_t;

/**
 * @brief Stigmergic pointer trail structure aligned to 128 bytes (2 cache lines).
 */
typedef struct GCSO_ALIGNAS(128) gcso_pointer_trail {
    uint64_t current_ptr;
    uint64_t prev_ptr;
    uint64_t user_data;
    int32_t transition_cost;
    uint32_t step_count;
    float stigmergic_density;
    uint32_t target_anchor_id;
    uint32_t cluster_id;
    uint32_t linked_trail_id;
    float attractor_pull_force;
    uint32_t flags;
    gcso_q7_t accumulated_phase_delta[64];
    uint8_t reserved_padding[8];
} gcso_pointer_trail_t;

/**
 * @brief Dynamic Adaptive Extension Scratchpad (DAES) slot layout (64 bytes).
 */
typedef struct GCSO_ALIGNAS(64) gcso_daes_slot {
    uint32_t mode;
    uint16_t telemetry_ring_head;
    uint16_t telemetry_ring_tail;
    uint32_t cache_hit_count;
    uint32_t auto_tune_flags;
    uint64_t fast_path_bypass_mask;
    uint64_t fast_path_shortcuts[4];
    uint8_t telemetry_mini_ledger[8];
} gcso_daes_slot_t;

/**
 * @brief Global configuration descriptor structure aligned to 16 bytes (64 bytes total).
 */
typedef struct GCSO_ALIGNAS(16) gcso_config {
    uint32_t head_dim;
    uint32_t num_heads;
    uint32_t paged_block_size;
    float q7_phase_scale;
    float ripa_clamp_max_rad;
    float qdps_min_step_rad;
    float entropy_singularity_eps;
    uint32_t max_prompt_anchors;
    uint32_t action_hub_capacity;
    uint8_t enable_cuda_warp_shuffle;
    uint8_t enable_zero_alloc_strict;
    uint8_t daes_mode;
    uint8_t reserved_flags;
    uint8_t reserved[24];
} gcso_config_t;

/**
 * @brief Dynamic entropy controller state tracking structure (64 bytes).
 */
typedef struct GCSO_ALIGNAS(32) gcso_edbc_state {
    float moving_z_entropy;
    float bifurcation_threshold;
    float singularity_eps;
    float sliding_entropy_rate;
    float repulsion_gain;
    float sample_temperature;
    uint32_t active_branch_mode;
    uint8_t reserved[36];
} gcso_edbc_state_t;

/**
 * @brief Zero-copy memory mapped storage descriptor (64 bytes).
 */
typedef struct GCSO_ALIGNAS(32) gcso_zimms_descriptor {
    uint64_t mapped_address;
    uint64_t file_size_bytes;
    uint64_t dma_buffer_handle;
    uint32_t flags;
    int32_t fd_handle;
    uint8_t reserved[32];
} gcso_zimms_descriptor_t;

/**
 * @brief Sub-head group router configuration descriptor (32 bytes).
 */
typedef struct GCSO_ALIGNAS(16) gcso_pspm_config {
    uint16_t num_fact_heads;
    uint16_t num_logic_heads;
    uint16_t num_explore_heads;
    uint16_t flags;
    float fact_phase_gain;
    float logic_phase_gain;
    float explore_phase_gain;
    uint8_t reserved[12];
} gcso_pspm_config_t;

/**
 * @brief Sparse Residual Adapter Layer (SRL) Rank-1 descriptor (64 bytes).
 */
typedef struct GCSO_ALIGNAS(32) gcso_srl_descriptor {
    uint32_t layer_idx;
    uint32_t rank;
    uint64_t u_vector_ptr;
    uint64_t v_vector_ptr;
    uint64_t gain_scalar_ptr;
    float scale_factor;
    uint32_t flags;
    uint8_t reserved[24];
} gcso_srl_descriptor_t;

/**
 * @brief Unified binary snapshot container header structure (128 bytes).
 */
typedef struct GCSO_ALIGNAS(64) gcso_snapshot_header {
    uint32_t magic;
    uint32_t version;
    uint64_t total_size;
    uint64_t action_hub_offset;
    uint64_t attractor_field_offset;
    uint64_t dpsr_state_offset;
    uint64_t srl_state_offset;
    uint64_t edbc_state_offset;
    uint32_t checksum_crc32;
    uint32_t daes_slot_offset;
    uint64_t timestamp_epoch_sec;
    uint8_t reserved_padding[56];
} gcso_snapshot_header_t;

/**
 * @brief PPRC Keyframe KV cache index header structure (64 bytes).
 */
typedef struct GCSO_ALIGNAS(32) gcso_pprc_keyframe_header {
    uint32_t frame_type;
    uint32_t token_index;
    uint32_t gop_length;
    float composite_vector_length;
    float von_mises_kappa;
    float sparse_scalar_residual;
    uint64_t icache_payload_offset;
    uint64_t pcache_payload_offset;
    uint8_t reserved[24];
} gcso_pprc_keyframe_header_t;

// Opaque Facade Handles
typedef struct gcso_context_opaque* gcso_context_handle_t;
typedef struct gcso_container_opaque* gcso_container_handle_t;
typedef struct gcso_action_hub_opaque* gcso_action_hub_handle_t;
typedef struct gcso_daes_slot_opaque* gcso_daes_slot_handle_t;
typedef struct gcso_attractor_field_opaque* gcso_attractor_field_handle_t;
typedef struct gcso_edbc_controller_opaque* gcso_edbc_controller_handle_t;
typedef struct gcso_dpsr_kernel_opaque* gcso_dpsr_kernel_handle_t;
typedef struct gcso_pspm_router_opaque* gcso_pspm_router_handle_t;
typedef struct gcso_srl_adapter_opaque* gcso_srl_adapter_handle_t;

<<<<<<< HEAD
// Static Assertions for Layout Invariants
GCSO_STATIC_ASSERT(sizeof(gcso_q7_t) == 1, "gcso_q7_t must be 1 byte");
GCSO_STATIC_ASSERT(sizeof(gcso_descriptor_header_t) == 8,
                   "gcso_descriptor_header_t must be 8 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_paged_bitmask_t) == 32, "gcso_paged_bitmask_t must be 32 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_pointer_trail_t) == 128, "gcso_pointer_trail_t must be 128 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_daes_slot_t) == 64, "gcso_daes_slot_t must be 64 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_config_t) == 64, "gcso_config_t must be 64 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_edbc_state_t) == 64, "gcso_edbc_state_t must be 64 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_zimms_descriptor_t) == 64,
                   "gcso_zimms_descriptor_t must be 64 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_pspm_config_t) == 32, "gcso_pspm_config_t must be 32 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_srl_descriptor_t) == 64, "gcso_srl_descriptor_t must be 64 bytes");
<<<<<<< HEAD
GCSO_STATIC_ASSERT(sizeof(gcso_snapshot_header_t) == 128,
                   "gcso_snapshot_header_t must be 128 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_pprc_keyframe_header_t) == 64,
                   "gcso_pprc_keyframe_header_t must be 64 bytes");
=======
// EDBC state layout (64 Bytes)
typedef struct GCSO_ALIGNAS(32) gcso_edbc_state {
    float moving_z_entropy;
    float bifurcation_threshold;
    float singularity_eps;
    float sliding_entropy_rate;
    float repulsion_gain;
    float sample_temperature;
    uint32_t active_branch_mode;
    uint8_t reserved[36];
} gcso_edbc_state_t;

// SRL descriptor layout (64 Bytes)
typedef struct GCSO_ALIGNAS(32) gcso_srl_descriptor {
    uint32_t layer_idx;
    uint32_t rank;
    uint64_t u_vector_ptr;
    uint64_t v_vector_ptr;
    uint64_t gain_scalar_ptr;
    float scale_factor;
    uint32_t flags;
    uint8_t reserved[24];
} gcso_srl_descriptor_t;

// ZIMMS descriptor layout (64 Bytes)
typedef struct GCSO_ALIGNAS(32) gcso_zimms_descriptor {
    uint64_t mapped_address;
    uint64_t file_size_bytes;
    uint64_t dma_buffer_handle;
    uint32_t flags;
    int32_t fd_handle;
    uint8_t reserved[32];
} gcso_zimms_descriptor_t;

            // Compile-time structure size and alignment assertions
            #ifdef __cplusplus
static_assert(sizeof(gcso_descriptor_header_t) == 8, "Size mismatch: gcso_descriptor_header_t");
static_assert(sizeof(gcso_paged_bitmask_t) == 32, "Size mismatch: gcso_paged_bitmask_t");
static_assert(alignof(gcso_paged_bitmask_t) == 32, "Align mismatch: gcso_paged_bitmask_t");
static_assert(sizeof(gcso_pointer_trail_t) == 128, "Size mismatch: gcso_pointer_trail_t");
static_assert(alignof(gcso_pointer_trail_t) == 128, "Align mismatch: gcso_pointer_trail_t");
static_assert(sizeof(gcso_config_t) == 64, "Size mismatch: gcso_config_t");
static_assert(sizeof(gcso_snapshot_header_t) == 128, "Size mismatch: gcso_snapshot_header_t");
static_assert(sizeof(gcso_daes_slot_t) == 64, "Size mismatch: gcso_daes_slot_t");
static_assert(sizeof(gcso_pprc_keyframe_header_t) == 64,
              "Size mismatch: gcso_pprc_keyframe_header_t");
static_assert(sizeof(gcso_pspm_config_t) == 32, "Size mismatch: gcso_pspm_config_t");
static_assert(sizeof(gcso_edbc_state_t) == 64, "Size mismatch: gcso_edbc_state_t");
static_assert(sizeof(gcso_srl_descriptor_t) == 64, "Size mismatch: gcso_srl_descriptor_t");
static_assert(sizeof(gcso_zimms_descriptor_t) == 64, "Size mismatch: gcso_zimms_descriptor_t");
            #endif
>>>>>>> 631dd2990e7e914074e1e2e891e7b8af8ac59c1c
=======
GCSO_STATIC_ASSERT(sizeof(gcso_snapshot_header_t) == 128,
                   "gcso_snapshot_header_t must be 128 bytes");
GCSO_STATIC_ASSERT(sizeof(gcso_pprc_keyframe_header_t) == 64,
                   "gcso_pprc_keyframe_header_t must be 64 bytes");
>>>>>>> 073c96b9ba43a51e79e76a2ea8e74feffeea69ee

GCSO_EXTERN_C_END

    #endif // LIMINIKA_GCSO_TYPES_H