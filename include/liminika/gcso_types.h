// name: include/liminika/gcso_types.h
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_TYPES_H
#define LIMINIKA_GCSO_TYPES_H

#include "liminika/gcso_config.h"

GCSO_EXTERN_C_BEGIN

/* Quantized 7-bit signed fixed-point integer (scale beta_Q7 = pi / 128) */
typedef int8_t gcso_q7_t;

/* 64-bit feature bitmask capability flags */
typedef uint64_t gcso_capability_flags_t;

/* Domain Constant Identifiers */
#define GCSO_SNAPSHOT_MAGIC           0x4F534347U /* "GCSO" in ASCII */
#define GCSO_Q7_PHASE_ARRAY_LEN       64U
#define GCSO_DAES_SHORTCUTS_LEN       4U
#define GCSO_DAES_MINI_LEDGER_LEN     8U
#define GCSO_PAGED_BITMASK_WORDS      4U

/**
 * @brief Standardized 32-bit status return codes for 1:1 cross-language FFI mapping.
 */
typedef enum gcso_status GCSO_ENUM_UNDERLYING(int32_t) {
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
typedef enum gcso_anchor_type GCSO_ENUM_UNDERLYING(uint32_t) {
    GCSO_ANCHOR_TYPE_SYSTEM_PROMPT = 0,
    GCSO_ANCHOR_TYPE_EMBEDDING = 1,
    GCSO_ANCHOR_TYPE_PHASE_REPULSE = 2,
    GCSO_ANCHOR_TYPE_TOPOLOGICAL = 3,
    GCSO_ANCHOR_TYPE_CRYSTALLIZED = 4
} gcso_anchor_type_t;

/**
 * @brief Descriptor header for size and ABI version validation.
 */
struct GCSO_ALIGN_PREFIX(4) gcso_descriptor_header {
    uint32_t struct_size;
    uint32_t abi_version;
} GCSO_ALIGN_SUFFIX(4);
typedef struct gcso_descriptor_header gcso_descriptor_header_t;

/**
 * @brief Strongly-typed aligned raw memory buffer descriptor.
 * Total size is 32 bytes (16-byte aligned, explicit 8-byte padding).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_raw_buffer {
    uint8_t* GCSO_RESTRICT data_ptr;
    size_t size_bytes;
    size_t alignment_bytes;
    uint8_t reserved_padding[8];
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_raw_buffer gcso_raw_buffer_t;

/**
 * @brief 256-bit bitmask layout aligned to 32 bytes for Warp/SIMD reductions.
 */
struct GCSO_ALIGN_PREFIX(32) gcso_paged_bitmask {
    uint64_t bits[GCSO_PAGED_BITMASK_WORDS];
} GCSO_ALIGN_SUFFIX(32);
typedef struct gcso_paged_bitmask gcso_paged_bitmask_t;

/**
 * @brief Stigmergic pointer trail structure aligned to 128 bytes (2 cache lines).
 */
struct GCSO_ALIGN_PREFIX(128) gcso_pointer_trail {
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
    gcso_q7_t accumulated_phase_delta[GCSO_Q7_PHASE_ARRAY_LEN];
    uint8_t reserved_padding[8];
} GCSO_ALIGN_SUFFIX(128);
typedef struct gcso_pointer_trail gcso_pointer_trail_t;

/**
 * @brief Dynamic Adaptive Extension Scratchpad (DAES) slot layout (64 bytes).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_daes_slot {
    uint32_t mode;
    uint16_t telemetry_ring_head;
    uint16_t telemetry_ring_tail;
    uint32_t cache_hit_count;
    uint32_t auto_tune_flags;
    uint64_t fast_path_bypass_mask;
    uint64_t fast_path_shortcuts[GCSO_DAES_SHORTCUTS_LEN];
    uint8_t telemetry_mini_ledger[GCSO_DAES_MINI_LEDGER_LEN];
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_daes_slot gcso_daes_slot_t;

/**
 * @brief Global configuration descriptor structure aligned to 16 bytes (64 bytes total).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_config {
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
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_config gcso_config_t;

/**
 * @brief Dynamic entropy controller state tracking structure (64 bytes).
 */
struct GCSO_ALIGN_PREFIX(32) gcso_edbc_state {
    float moving_z_entropy;
    float bifurcation_threshold;
    float singularity_eps;
    float sliding_entropy_rate;
    float repulsion_gain;
    float sample_temperature;
    uint32_t active_branch_mode;
    uint8_t reserved[36];
} GCSO_ALIGN_SUFFIX(32);
typedef struct gcso_edbc_state gcso_edbc_state_t;

/**
 * @brief Zero-copy memory mapped storage descriptor (64 bytes).
 */
struct GCSO_ALIGN_PREFIX(32) gcso_zimms_descriptor {
    uint64_t mapped_address;
    uint64_t file_size_bytes;
    uint64_t dma_buffer_handle;
    uint32_t flags;
    int32_t fd_handle;
    uint8_t reserved[32];
} GCSO_ALIGN_SUFFIX(32);
typedef struct gcso_zimms_descriptor gcso_zimms_descriptor_t;

/**
 * @brief Sub-head group router configuration descriptor (32 bytes).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_pspm_config {
    uint16_t num_fact_heads;
    uint16_t num_logic_heads;
    uint16_t num_explore_heads;
    uint16_t flags;
    float fact_phase_gain;
    float logic_phase_gain;
    float explore_phase_gain;
    uint8_t reserved[12];
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_pspm_config gcso_pspm_config_t;

/**
 * @brief Sparse Residual Adapter Layer (SRL) Rank-1 descriptor (64 bytes).
 */
struct GCSO_ALIGN_PREFIX(32) gcso_srl_descriptor {
    uint32_t layer_idx;
    uint32_t rank;
    uint64_t u_vector_ptr;
    uint64_t v_vector_ptr;
    uint64_t gain_scalar_ptr;
    float scale_factor;
    uint32_t flags;
    uint8_t reserved[24];
} GCSO_ALIGN_SUFFIX(32);
typedef struct gcso_srl_descriptor gcso_srl_descriptor_t;

/**
 * @brief Unified binary snapshot container header structure (128 bytes).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_snapshot_header {
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
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_snapshot_header gcso_snapshot_header_t;

/**
 * @brief PPRC Keyframe KV cache index header structure (64 bytes).
 */
struct GCSO_ALIGN_PREFIX(32) gcso_pprc_keyframe_header {
    uint32_t frame_type;
    uint32_t token_index;
    uint32_t gop_length;
    float composite_vector_length;
    float von_mises_kappa;
    float sparse_scalar_residual;
    uint64_t icache_payload_offset;
    uint64_t pcache_payload_offset;
    uint8_t reserved[24];
} GCSO_ALIGN_SUFFIX(32);
typedef struct gcso_pprc_keyframe_header gcso_pprc_keyframe_header_t;

/*
 * Strongly-Typed Opaque Handles eliminating generic raw `c_void` / `void*` pointers.
 * Each domain component has a distinct opaque type to prevent type confusion at compile time.
 */

struct gcso_context_opaque;
typedef struct gcso_context_opaque* gcso_context_handle_t;

struct gcso_container_opaque;
typedef struct gcso_container_opaque* gcso_container_handle_t;

struct gcso_action_hub_opaque;
typedef struct gcso_action_hub_opaque* gcso_action_hub_handle_t;

struct gcso_daes_slot_opaque;
typedef struct gcso_daes_slot_opaque* gcso_daes_slot_handle_t;

struct gcso_attractor_field_opaque;
typedef struct gcso_attractor_field_opaque* gcso_attractor_field_handle_t;

struct gcso_edbc_controller_opaque;
typedef struct gcso_edbc_controller_opaque* gcso_edbc_controller_handle_t;

struct gcso_dpsr_kernel_opaque;
typedef struct gcso_dpsr_kernel_opaque* gcso_dpsr_kernel_handle_t;

struct gcso_pspm_router_opaque;
typedef struct gcso_pspm_router_opaque* gcso_pspm_router_handle_t;

struct gcso_srl_adapter_opaque;
typedef struct gcso_srl_adapter_opaque* gcso_srl_adapter_handle_t;

/* Static Assertions for Size Invariants Matching traits.rs / abi.rs */
GCSO_STATIC_ASSERT(sizeof(gcso_status_t) == 4, "gcso_status_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_anchor_type_t) == 4, "gcso_anchor_type_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_q7_t) == 1, "gcso_q7_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_descriptor_header_t) == 8, "gcso_descriptor_header_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_raw_buffer_t) == 32, "gcso_raw_buffer_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_paged_bitmask_t) == 32, "gcso_paged_bitmask_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_pointer_trail_t) == 128, "gcso_pointer_trail_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_daes_slot_t) == 64, "gcso_daes_slot_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_config_t) == 64, "gcso_config_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_edbc_state_t) == 64, "gcso_edbc_state_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_zimms_descriptor_t) == 64, "gcso_zimms_descriptor_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_pspm_config_t) == 32, "gcso_pspm_config_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_srl_descriptor_t) == 64, "gcso_srl_descriptor_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_snapshot_header_t) == 128, "gcso_snapshot_header_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_pprc_keyframe_header_t) == 64, "gcso_pprc_keyframe_header_t size mismatch");

/* Static Assertions for Alignment Invariants Matching traits.rs / abi.rs */
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_raw_buffer_t) == 16, "gcso_raw_buffer_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_paged_bitmask_t) == 32, "gcso_paged_bitmask_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_pointer_trail_t) == 128, "gcso_pointer_trail_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_daes_slot_t) == 64, "gcso_daes_slot_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_config_t) == 16, "gcso_config_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_edbc_state_t) == 32, "gcso_edbc_state_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_zimms_descriptor_t) == 32, "gcso_zimms_descriptor_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_pspm_config_t) == 16, "gcso_pspm_config_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_srl_descriptor_t) == 32, "gcso_srl_descriptor_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_snapshot_header_t) == 64, "gcso_snapshot_header_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_pprc_keyframe_header_t) == 32, "gcso_pprc_keyframe_header_t alignment mismatch");

/* Static Assertions for Exact Field Offsets Matching Rust abi.rs & traits.rs */
GCSO_STATIC_ASSERT(offsetof(gcso_descriptor_header_t, struct_size) == 0,
                   "gcso_descriptor_header_t::struct_size offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_descriptor_header_t, abi_version) == 4,
                   "gcso_descriptor_header_t::abi_version offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_raw_buffer_t, data_ptr) == 0,
                   "gcso_raw_buffer_t::data_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_raw_buffer_t, size_bytes) == 8,
                   "gcso_raw_buffer_t::size_bytes offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_raw_buffer_t, alignment_bytes) == 16,
                   "gcso_raw_buffer_t::alignment_bytes offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, current_ptr) == 0,
                   "gcso_pointer_trail_t::current_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, prev_ptr) == 8,
                   "gcso_pointer_trail_t::prev_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, user_data) == 16,
                   "gcso_pointer_trail_t::user_data offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, transition_cost) == 24,
                   "gcso_pointer_trail_t::transition_cost offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, step_count) == 28,
                   "gcso_pointer_trail_t::step_count offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, stigmergic_density) == 32,
                   "gcso_pointer_trail_t::stigmergic_density offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, target_anchor_id) == 36,
                   "gcso_pointer_trail_t::target_anchor_id offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, cluster_id) == 40,
                   "gcso_pointer_trail_t::cluster_id offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, linked_trail_id) == 44,
                   "gcso_pointer_trail_t::linked_trail_id offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, attractor_pull_force) == 48,
                   "gcso_pointer_trail_t::attractor_pull_force offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, flags) == 52,
                   "gcso_pointer_trail_t::flags offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, accumulated_phase_delta) == 56,
                   "gcso_pointer_trail_t::accumulated_phase_delta offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_daes_slot_t, mode) == 0,
                   "gcso_daes_slot_t::mode offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_daes_slot_t, telemetry_ring_head) == 4,
                   "gcso_daes_slot_t::telemetry_ring_head offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_daes_slot_t, telemetry_ring_tail) == 6,
                   "gcso_daes_slot_t::telemetry_ring_tail offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_daes_slot_t, cache_hit_count) == 8,
                   "gcso_daes_slot_t::cache_hit_count offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_daes_slot_t, auto_tune_flags) == 12,
                   "gcso_daes_slot_t::auto_tune_flags offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_daes_slot_t, fast_path_bypass_mask) == 16,
                   "gcso_daes_slot_t::fast_path_bypass_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_daes_slot_t, fast_path_shortcuts) == 24,
                   "gcso_daes_slot_t::fast_path_shortcuts offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_daes_slot_t, telemetry_mini_ledger) == 56,
                   "gcso_daes_slot_t::telemetry_mini_ledger offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_config_t, head_dim) == 0,
                   "gcso_config_t::head_dim offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, num_heads) == 4,
                   "gcso_config_t::num_heads offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, paged_block_size) == 8,
                   "gcso_config_t::paged_block_size offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, q7_phase_scale) == 12,
                   "gcso_config_t::q7_phase_scale offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, ripa_clamp_max_rad) == 16,
                   "gcso_config_t::ripa_clamp_max_rad offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, qdps_min_step_rad) == 20,
                   "gcso_config_t::qdps_min_step_rad offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, entropy_singularity_eps) == 24,
                   "gcso_config_t::entropy_singularity_eps offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, max_prompt_anchors) == 28,
                   "gcso_config_t::max_prompt_anchors offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, action_hub_capacity) == 32,
                   "gcso_config_t::action_hub_capacity offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, enable_cuda_warp_shuffle) == 36,
                   "gcso_config_t::enable_cuda_warp_shuffle offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, enable_zero_alloc_strict) == 37,
                   "gcso_config_t::enable_zero_alloc_strict offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, daes_mode) == 38,
                   "gcso_config_t::daes_mode offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, reserved_flags) == 39,
                   "gcso_config_t::reserved_flags offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_edbc_state_t, moving_z_entropy) == 0,
                   "gcso_edbc_state_t::moving_z_entropy offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_edbc_state_t, bifurcation_threshold) == 4,
                   "gcso_edbc_state_t::bifurcation_threshold offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_edbc_state_t, singularity_eps) == 8,
                   "gcso_edbc_state_t::singularity_eps offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_edbc_state_t, sliding_entropy_rate) == 12,
                   "gcso_edbc_state_t::sliding_entropy_rate offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_edbc_state_t, repulsion_gain) == 16,
                   "gcso_edbc_state_t::repulsion_gain offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_edbc_state_t, sample_temperature) == 20,
                   "gcso_edbc_state_t::sample_temperature offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_edbc_state_t, active_branch_mode) == 24,
                   "gcso_edbc_state_t::active_branch_mode offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_zimms_descriptor_t, mapped_address) == 0,
                   "gcso_zimms_descriptor_t::mapped_address offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_zimms_descriptor_t, file_size_bytes) == 8,
                   "gcso_zimms_descriptor_t::file_size_bytes offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_zimms_descriptor_t, dma_buffer_handle) == 16,
                   "gcso_zimms_descriptor_t::dma_buffer_handle offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_zimms_descriptor_t, flags) == 24,
                   "gcso_zimms_descriptor_t::flags offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_zimms_descriptor_t, fd_handle) == 28,
                   "gcso_zimms_descriptor_t::fd_handle offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_pspm_config_t, num_fact_heads) == 0,
                   "gcso_pspm_config_t::num_fact_heads offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pspm_config_t, num_logic_heads) == 2,
                   "gcso_pspm_config_t::num_logic_heads offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pspm_config_t, num_explore_heads) == 4,
                   "gcso_pspm_config_t::num_explore_heads offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pspm_config_t, flags) == 6,
                   "gcso_pspm_config_t::flags offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pspm_config_t, fact_phase_gain) == 8,
                   "gcso_pspm_config_t::fact_phase_gain offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pspm_config_t, logic_phase_gain) == 12,
                   "gcso_pspm_config_t::logic_phase_gain offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pspm_config_t, explore_phase_gain) == 16,
                   "gcso_pspm_config_t::explore_phase_gain offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_srl_descriptor_t, layer_idx) == 0,
                   "gcso_srl_descriptor_t::layer_idx offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_srl_descriptor_t, rank) == 4,
                   "gcso_srl_descriptor_t::rank offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_srl_descriptor_t, u_vector_ptr) == 8,
                   "gcso_srl_descriptor_t::u_vector_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_srl_descriptor_t, v_vector_ptr) == 16,
                   "gcso_srl_descriptor_t::v_vector_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_srl_descriptor_t, gain_scalar_ptr) == 24,
                   "gcso_srl_descriptor_t::gain_scalar_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_srl_descriptor_t, scale_factor) == 32,
                   "gcso_srl_descriptor_t::scale_factor offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_srl_descriptor_t, flags) == 36,
                   "gcso_srl_descriptor_t::flags offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, magic) == 0,
                   "gcso_snapshot_header_t::magic offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, version) == 4,
                   "gcso_snapshot_header_t::version offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, total_size) == 8,
                   "gcso_snapshot_header_t::total_size offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, action_hub_offset) == 16,
                   "gcso_snapshot_header_t::action_hub_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, attractor_field_offset) == 24,
                   "gcso_snapshot_header_t::attractor_field_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, dpsr_state_offset) == 32,
                   "gcso_snapshot_header_t::dpsr_state_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, srl_state_offset) == 40,
                   "gcso_snapshot_header_t::srl_state_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, edbc_state_offset) == 48,
                   "gcso_snapshot_header_t::edbc_state_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, checksum_crc32) == 56,
                   "gcso_snapshot_header_t::checksum_crc32 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, daes_slot_offset) == 60,
                   "gcso_snapshot_header_t::daes_slot_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, timestamp_epoch_sec) == 64,
                   "gcso_snapshot_header_t::timestamp_epoch_sec offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_pprc_keyframe_header_t, frame_type) == 0,
                   "gcso_pprc_keyframe_header_t::frame_type offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pprc_keyframe_header_t, token_index) == 4,
                   "gcso_pprc_keyframe_header_t::token_index offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pprc_keyframe_header_t, gop_length) == 8,
                   "gcso_pprc_keyframe_header_t::gop_length offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pprc_keyframe_header_t, composite_vector_length) == 12,
                   "gcso_pprc_keyframe_header_t::composite_vector_length offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pprc_keyframe_header_t, von_mises_kappa) == 16,
                   "gcso_pprc_keyframe_header_t::von_mises_kappa offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pprc_keyframe_header_t, sparse_scalar_residual) == 20,
                   "gcso_pprc_keyframe_header_t::sparse_scalar_residual offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pprc_keyframe_header_t, icache_payload_offset) == 24,
                   "gcso_pprc_keyframe_header_t::icache_payload_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_pprc_keyframe_header_t, pcache_payload_offset) == 32,
                   "gcso_pprc_keyframe_header_t::pcache_payload_offset offset mismatch");

GCSO_EXTERN_C_END

#endif // LIMINIKA_GCSO_TYPES_H