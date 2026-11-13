// name: include/liminika/gcso_types.h
// SPDX-License-Identifier: MIT OR Apache-2.0

/**
 * @file gcso_types.h
 * @brief C-ABI v0.1.1 POD types, bitmasks, status codes, and compile-time layout assertions.
 *
 * All struct sizes, alignments, and field offsets in this header are verified at compile time
 * via `GCSO_STATIC_ASSERT` to maintain 1:1 ABI parity with `src/core/src/abi/` and
 * `src/core/src/traits/`.
 *
 * Legacy C-ABI v0.1.1 symbol prefixes (`dpsr`, `edbc`, `srl`, `zimms`) are preserved for binary
 * and source compatibility and map to the Rust core modules:
 * - `dpsr`  -> `phase_steering` (`PhaseSteeringEngine`)
 * - `edbc`  -> `entropy_control` (`EntropyController`)
 * - `srl`   -> `residual_adapter` (`ResidualAdapterEngine`)
 * - `zimms` -> `storage` (`MappedStorageEngine`)
 */

#ifndef LIMINIKA_GCSO_TYPES_H
#define LIMINIKA_GCSO_TYPES_H

#include "liminika/gcso_config.h"

GCSO_EXTERN_C_BEGIN

/* Quantized 7-bit signed fixed-point integer (scale beta_Q7 = pi / 128) */
typedef int8_t gcso_q7_t;

/* 64-bit feature bitmask capability flags */
typedef uint64_t gcso_capability_flags_t;

/* 64-bit packed tagged memory pointer ([Phase(8b) | Tag(8b) | Addr(48b)]) */
typedef uint64_t gcso_tagged_ptr_t;

/* 8-bit discriminant for 7-regime stigmergic next-action selection */
typedef uint8_t gcso_stigmergic_action_decision_t;

/* ===================================================================
 * Domain Constant & Bitmask Identifiers (ABI v0.1.1, 1:1 with src/core)
 * =================================================================== */
#define GCSO_SNAPSHOT_MAGIC 0x4F534347U         /* "GCSO" in ASCII */
#define GCSO_CRYSTAL_RECORD_MAGIC 0x4C545343U   /* "CSTL" in ASCII */
#define GCSO_CRYSTAL_RECORD_VERSION 0x00000101U /* Unified v0.1.1 */
#define GCSO_FOUR_SECTION_HEADER_BYTES 192U
#define GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET 128U
#define GCSO_Q7_PHASE_ARRAY_LEN 64U
#define GCSO_BLOOM_FILTER_COUNTERS_LEN 64U
#define GCSO_DAES_SHORTCUTS_LEN 4U
#define GCSO_DAES_MINI_LEDGER_LEN 8U
#define GCSO_PAGED_BITMASK_WORDS 4U
#define GCSO_NUM_SWARM_MICRO_CELLS 8U
#define GCSO_MAX_ATTRACTOR_DOMAINS 8U

/* Hardware Capability Bitmask Constants */
#define GCSO_CAP_CPU_BASELINE (1ULL << 0)
#define GCSO_CAP_CUDA_BACKEND (1ULL << 1)
#define GCSO_CAP_VULKAN_BACKEND (1ULL << 2)
#define GCSO_CAP_METAL_BACKEND (1ULL << 3)

/* 64-Bit Tagged Pointer Masks & Bit Shifts */
#define GCSO_TAGGED_PTR_ADDR_MASK 0x0000FFFFFFFFFFFFULL
#define GCSO_TAGGED_PTR_TAG_MASK 0x00FF000000000000ULL
#define GCSO_TAGGED_PTR_PHASE_MASK 0xFF00000000000000ULL
#define GCSO_TAGGED_PTR_TAG_SHIFT 48U
#define GCSO_TAGGED_PTR_PHASE_SHIFT 56U
#define GCSO_TAG_ACTION_MODE_MASK 0x1FU
#define GCSO_TAG_ATTRACTOR_SLOT_SHIFT 5U

/* 3-Bit Attractor Working Lane Slot Identifiers (0..7, matching primitives.rs) */
#define GCSO_ATTRACTOR_SLOT_SYSTEM_BASELINE 0U
#define GCSO_ATTRACTOR_SLOT_PERSONA 1U
#define GCSO_ATTRACTOR_SLOT_KNOWLEDGE 2U
#define GCSO_ATTRACTOR_SLOT_REPULSION 3U
#define GCSO_ATTRACTOR_SLOT_USER_CONTEXT 4U
#define GCSO_ATTRACTOR_SLOT_CUSTOM_5 5U
#define GCSO_ATTRACTOR_SLOT_CUSTOM_6 6U
#define GCSO_ATTRACTOR_SLOT_CUSTOM_7 7U
/* Semantic aliases for custom working lanes 4..7 */
#define GCSO_ATTRACTOR_SLOT_SPATIAL GCSO_ATTRACTOR_SLOT_USER_CONTEXT
#define GCSO_ATTRACTOR_SLOT_TEMPORAL GCSO_ATTRACTOR_SLOT_CUSTOM_5
#define GCSO_ATTRACTOR_SLOT_USER_INTENT GCSO_ATTRACTOR_SLOT_CUSTOM_6
#define GCSO_ATTRACTOR_SLOT_CRYSTALLIZED GCSO_ATTRACTOR_SLOT_CUSTOM_7
#define GCSO_ATTRACTOR_SLOT_ID_MASK 0x07U
#define GCSO_DEFAULT_SLOT_COUPLING_MATRIX_8X8 0xFFFFFFFFFFFFFFFFULL

/* 5-Bit Action Mode Execution Flags (1:1 bit parity with primitives.rs) */
#define GCSO_ACTION_MODE_DPSR_STEER (1U << 0)    /* 0x01U */
#define GCSO_ACTION_MODE_LAYER_SKIP (1U << 1)    /* 0x02U */
#define GCSO_ACTION_MODE_PHASE_REPULSE (1U << 2) /* 0x04U */
#define GCSO_ACTION_MODE_PSPM_ROUTE (1U << 3)    /* 0x08U */
#define GCSO_ACTION_MODE_SRL_INJECT (1U << 4)    /* 0x10U */
#define GCSO_ACTION_MODE_ALL_MASK 0x1FU

/* Canonical Action Mode Presets (1:1 parity with primitives.rs) */
#define GCSO_ACTION_PRESET_PASSIVE_SIDECAR 0x00U
#define GCSO_ACTION_PRESET_EXPRESS_PHASE (GCSO_ACTION_MODE_DPSR_STEER)
#define GCSO_ACTION_PRESET_INERTIAL_SKIP (GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_LAYER_SKIP)
#define GCSO_ACTION_PRESET_REPULSION_ESCAPE \
    (GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_PHASE_REPULSE | GCSO_ACTION_MODE_PSPM_ROUTE)
#define GCSO_ACTION_PRESET_FULL_ORCHESTRATION \
    (GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_PSPM_ROUTE | GCSO_ACTION_MODE_SRL_INJECT)

/* Pointer Trail Status Flags (1:1 bit parity with telemetry.rs) */
#define GCSO_TRAIL_FLAG_DPSR_ACTIVE (1U << 0)            /* 0x00000001U */
#define GCSO_TRAIL_FLAG_REPULSION_ACTIVE (1U << 1)       /* 0x00000002U */
#define GCSO_TRAIL_FLAG_CRYSTALLIZED_CANDIDATE (1U << 2) /* 0x00000004U */
#define GCSO_TRAIL_FLAG_SYNERGY_RESONANCE (1U << 3)      /* 0x00000008U */
#define GCSO_TRAIL_FLAG_LIMINAL_BEAT_ACTIVE (1U << 4)    /* 0x00000010U */
#define GCSO_TRAIL_FLAG_INERTIAL_SKIP_ACTIVE (1U << 5)   /* 0x00000020U */
#define GCSO_TRAIL_FLAG_SWARM_RESOLVED (1U << 6)         /* 0x00000040U */
#define GCSO_TRAIL_FLAG_SWARM_ESCALATED (1U << 7)        /* 0x00000080U */
#define GCSO_TRAIL_SWARM_QUORUM_SHIFT 8U
#define GCSO_TRAIL_SWARM_QUORUM_MASK 0x0000FF00U

/* DAES Telemetry Ring Ledger Metric Bitflags (1:1 bit parity with telemetry.rs) */
#define GCSO_DAES_METRIC_FAST_PATH_HIT (1U << 0)
#define GCSO_DAES_METRIC_PHASE_STEERED (1U << 1)
#define GCSO_DAES_METRIC_REPULSION_APPLIED (1U << 2)
#define GCSO_DAES_METRIC_LAYER_SKIP_COMPENSATED (1U << 3)
#define GCSO_DAES_METRIC_PSPM_ROUTED (1U << 4)
#define GCSO_DAES_METRIC_SRL_INJECTED (1U << 5)
#define GCSO_DAES_METRIC_LAZY_UNWRAP_APPLIED (1U << 6)
#define GCSO_DAES_METRIC_SLOT_SYNERGY_ACTIVE (1U << 7)

/* 4-Axis Causal Provenance Bitflags & Root Cause Codes (1:1 parity with gateway.rs) */
#define GCSO_CAUSAL_AXIS_SYSTEM_CONSTRAINT (1U << 0)
#define GCSO_CAUSAL_AXIS_OBJECTIVE_FACT (1U << 1)
#define GCSO_CAUSAL_AXIS_SUBJECTIVE_PERSONA (1U << 2)
#define GCSO_CAUSAL_AXIS_CONTRADICTION_GUARD (1U << 3)

#define GCSO_CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE (1U << 8)
#define GCSO_CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK (1U << 9)
#define GCSO_CAUSAL_CAUSE_PERSONA_FACT_COLLISION (1U << 10)
#define GCSO_CAUSAL_CAUSE_COHOMOLOGY_REPULSION (1U << 11)
#define GCSO_CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED (1U << 12)

#define GCSO_ROOT_CAUSE_NOMINAL 0U
#define GCSO_ROOT_CAUSE_SYSTEM_QOS_THROTTLE 1U
#define GCSO_ROOT_CAUSE_FACT_RETRIEVAL_MISS 2U
#define GCSO_ROOT_CAUSE_PERSONA_FACT_COLLISION 3U
#define GCSO_ROOT_CAUSE_COHOMOLOGY_REPULSION 4U

/**
 * @brief Standardized 32-bit status return codes for 1:1 cross-language FFI mapping.
 */
typedef enum gcso_status GCSO_ENUM_UNDERLYING(int32_t){
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
    GCSO_ERROR_UNKNOWN = -0x7FFFFFFF} gcso_status_t;

/**
 * @brief Classification types for topological attractor field anchors.
 */
typedef enum gcso_anchor_type GCSO_ENUM_UNDERLYING(uint32_t){
    GCSO_ANCHOR_TYPE_SYSTEM_PROMPT = 0, GCSO_ANCHOR_TYPE_EMBEDDING = 1,
    GCSO_ANCHOR_TYPE_PHASE_REPULSE = 2, GCSO_ANCHOR_TYPE_TOPOLOGICAL = 3,
    GCSO_ANCHOR_TYPE_CRYSTALLIZED = 4} gcso_anchor_type_t;

/**
 * @brief 7-regime stigmergic action decisions matching Rust StigmergicActionDecision (0..=6).
 */
typedef enum gcso_stigmergic_action_decision GCSO_ENUM_UNDERLYING(uint8_t){
    GCSO_DECISION_PASSIVE_SIDECAR = 0,        GCSO_DECISION_INERTIAL_LAYER_SKIP = 1,
    GCSO_DECISION_EXPRESS_PHASE_STEER = 2,    GCSO_DECISION_PSPM_SUB_HEAD_ROUTE = 3,
    GCSO_DECISION_FULL_ORCHESTRATION_SRL = 4, GCSO_DECISION_CRYSTALLIZE_MACRO_BASIN = 5,
    GCSO_DECISION_REPULSION_ESCAPE = 6} gcso_stigmergic_action_decision_enum_t;

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

/**
 * @brief Stigmergic threshold configuration governing pheromone dynamics & 7-regime selection (32
 * bytes).
 */
struct GCSO_ALIGN_PREFIX(32) gcso_stigmergic_thresholds {
    float evaporation_retain;
    float reinforcement_gain;
    float inertial_skip_entropy_thresh;
    float express_steer_density_thresh;
    float pspm_route_density_thresh;
    float srl_inject_density_thresh;
    float crystallize_density_thresh;
    float contradiction_coherence_thresh;
} GCSO_ALIGN_SUFFIX(32);
typedef struct gcso_stigmergic_thresholds gcso_stigmergic_thresholds_t;

/**
 * @brief Stigmergic action orchestration step report (32 bytes).
 */
struct GCSO_ALIGN_PREFIX(32) gcso_action_orchestration_report {
    uint64_t evolved_ptr;
    float updated_density;
    float phase_coherence;
    float transition_log_prob;
    uint32_t selected_trail_id;
    uint32_t dominant_anchor_id;
    uint8_t decision;
    uint8_t action_mode_flags;
    uint8_t target_slot;
    uint8_t bloom_repulsion_density;
} GCSO_ALIGN_SUFFIX(32);
typedef struct gcso_action_orchestration_report gcso_action_orchestration_report_t;

/**
 * @brief 8-cell Micro-Swarm quorum resolution summary report (16 bytes).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_swarm_resolution_report {
    gcso_tagged_ptr_t evolved_ptr;
    float stigmergic_delta_density;
    uint8_t quorum_mask;
    uint8_t resolved_cell_count;
    uint8_t mean_residual_q7;
    uint8_t hot_path_resolved;
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_swarm_resolution_report gcso_swarm_resolution_report_t;

/**
 * @brief 2-Origin Composite Manifold Gateway frame (128 bytes, 64-byte cacheline aligned).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_gateway_frame {
    uint64_t tagged_ptr;
    uint64_t zimms_section_offset48;
    uint32_t target_anchor_id;
    float effective_energy;
    float ripa_clamp_rad;
    float qdps_min_step_rad;
    float effective_inv_temp;
    float fact_persona_coherence;
    float repulsion_severity;
    uint32_t provenance_bits;
    uint8_t dominant_cause_code;
    uint8_t dominant_slot;
    uint8_t action_mode_flags;
    uint8_t active_swarm_cell_mask;
    uint8_t qdps_min_step_q7;
    uint8_t pspm_fact_heads;
    uint8_t pspm_logic_heads;
    uint8_t pspm_explore_heads;
    uint8_t pspm_fact_gain_q7;
    uint8_t pspm_logic_gain_q7;
    uint8_t pspm_explore_gain_q7;
    gcso_q7_t mean_phase_tag_q7;
    uint16_t active_head_count;
    uint16_t frame_flags;
    gcso_q7_t phase_deltas[GCSO_Q7_PHASE_ARRAY_LEN];
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_gateway_frame gcso_gateway_frame_t;

/**
 * @brief 4-Axis Causal Provenance diagnostic report (16 bytes, 16-byte aligned).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_causal_provenance_report {
    uint32_t provenance_bits;
    float fact_persona_coherence;
    float anomaly_severity;
    uint8_t dominant_cause_code;
    uint8_t dominant_slot;
    uint8_t action_mode_flags;
    uint8_t active_swarm_mask;
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_causal_provenance_report gcso_causal_provenance_report_t;

/**
 * @brief Crystallized macro attractor centroid record on SO(2)^64 persisted in .gcso (128 bytes).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_crystallized_attractor_record {
    uint32_t magic;
    uint32_t version;
    uint32_t anchor_id;
    uint32_t sample_count;
    uint64_t zimms_section_offset48;
    float base_energy;
    float resultant_r_bar;
    float von_mises_kappa;
    float phase_gravity_mass;
    float ripa_clamp_rad;
    float effective_inv_temp;
    uint32_t provenance_bits;
    uint32_t checksum_crc32;
    uint8_t dominant_slot;
    uint8_t action_mode_flags;
    uint8_t active_swarm_mask;
    uint8_t dominant_cause_code;
    uint32_t reserved_align64;
    gcso_q7_t phase_profile[GCSO_Q7_PHASE_ARRAY_LEN];
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_crystallized_attractor_record gcso_crystallized_attractor_record_t;

/**
 * @brief Cacheline-aligned Counting Bloom Filter for Contrastive Phase Memory (64 bytes).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_contrastive_bloom_filter {
    uint8_t counters[GCSO_BLOOM_FILTER_COUNTERS_LEN];
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_contrastive_bloom_filter gcso_contrastive_bloom_filter_t;

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

/* Static Assertions for Size Invariants Matching src/core/src/abi/ & traits/ */
GCSO_STATIC_ASSERT(sizeof(gcso_status_t) == 4, "gcso_status_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_anchor_type_t) == 4, "gcso_anchor_type_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_q7_t) == 1, "gcso_q7_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_tagged_ptr_t) == 8, "gcso_tagged_ptr_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_stigmergic_action_decision_t) == 1,
                   "gcso_stigmergic_action_decision_t size mismatch");
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
GCSO_STATIC_ASSERT(sizeof(gcso_pprc_keyframe_header_t) == 64,
                   "gcso_pprc_keyframe_header_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_stigmergic_thresholds_t) == 32,
                   "gcso_stigmergic_thresholds_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_action_orchestration_report_t) == 32,
                   "gcso_action_orchestration_report_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_swarm_resolution_report_t) == 16,
                   "gcso_swarm_resolution_report_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_gateway_frame_t) == 128, "gcso_gateway_frame_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_causal_provenance_report_t) == 16,
                   "gcso_causal_provenance_report_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_crystallized_attractor_record_t) == 128,
                   "gcso_crystallized_attractor_record_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_contrastive_bloom_filter_t) == 64,
                   "gcso_contrastive_bloom_filter_t size mismatch");

/* Static Assertions for Alignment Invariants Matching src/core/src/abi/ & traits/ */
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_raw_buffer_t) == 16, "gcso_raw_buffer_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_paged_bitmask_t) == 32,
                   "gcso_paged_bitmask_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_pointer_trail_t) == 128,
                   "gcso_pointer_trail_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_daes_slot_t) == 64, "gcso_daes_slot_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_config_t) == 16, "gcso_config_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_edbc_state_t) == 32, "gcso_edbc_state_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_zimms_descriptor_t) == 32,
                   "gcso_zimms_descriptor_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_pspm_config_t) == 16, "gcso_pspm_config_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_srl_descriptor_t) == 32,
                   "gcso_srl_descriptor_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_snapshot_header_t) == 64,
                   "gcso_snapshot_header_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_pprc_keyframe_header_t) == 32,
                   "gcso_pprc_keyframe_header_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_stigmergic_thresholds_t) == 32,
                   "gcso_stigmergic_thresholds_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_action_orchestration_report_t) == 32,
                   "gcso_action_orchestration_report_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_swarm_resolution_report_t) == 16,
                   "gcso_swarm_resolution_report_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_gateway_frame_t) == 64,
                   "gcso_gateway_frame_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_causal_provenance_report_t) == 16,
                   "gcso_causal_provenance_report_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_crystallized_attractor_record_t) == 64,
                   "gcso_crystallized_attractor_record_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_contrastive_bloom_filter_t) == 64,
                   "gcso_contrastive_bloom_filter_t alignment mismatch");

/* Static Assertions for Exact Field Offsets Matching Rust abi/ & traits/ */
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
GCSO_STATIC_ASSERT(offsetof(gcso_pointer_trail_t, reserved_padding) == 120,
                   "gcso_pointer_trail_t::reserved_padding offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_daes_slot_t, mode) == 0, "gcso_daes_slot_t::mode offset mismatch");
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

/* Static Assertions for v0.1.1 Gateway, Toroidal k-Means, Swarm & Orchestration Structs */
GCSO_STATIC_ASSERT(offsetof(gcso_stigmergic_thresholds_t, evaporation_retain) == 0,
                   "gcso_stigmergic_thresholds_t::evaporation_retain offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_stigmergic_thresholds_t, contradiction_coherence_thresh) == 28,
                   "gcso_stigmergic_thresholds_t::contradiction_coherence_thresh offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_action_orchestration_report_t, evolved_ptr) == 0,
                   "gcso_action_orchestration_report_t::evolved_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_action_orchestration_report_t, selected_trail_id) == 20,
                   "gcso_action_orchestration_report_t::selected_trail_id offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_action_orchestration_report_t, dominant_anchor_id) == 24,
                   "gcso_action_orchestration_report_t::dominant_anchor_id offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_action_orchestration_report_t, decision) == 28,
                   "gcso_action_orchestration_report_t::decision offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_swarm_resolution_report_t, evolved_ptr) == 0,
                   "gcso_swarm_resolution_report_t::evolved_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_swarm_resolution_report_t, stigmergic_delta_density) == 8,
                   "gcso_swarm_resolution_report_t::stigmergic_delta_density offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_swarm_resolution_report_t, quorum_mask) == 12,
                   "gcso_swarm_resolution_report_t::quorum_mask offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_gateway_frame_t, tagged_ptr) == 0,
                   "gcso_gateway_frame_t::tagged_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_gateway_frame_t, zimms_section_offset48) == 8,
                   "gcso_gateway_frame_t::zimms_section_offset48 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_gateway_frame_t, target_anchor_id) == 16,
                   "gcso_gateway_frame_t::target_anchor_id offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_gateway_frame_t, provenance_bits) == 44,
                   "gcso_gateway_frame_t::provenance_bits offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_gateway_frame_t, dominant_cause_code) == 48,
                   "gcso_gateway_frame_t::dominant_cause_code offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_gateway_frame_t, active_head_count) == 60,
                   "gcso_gateway_frame_t::active_head_count offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_gateway_frame_t, phase_deltas) == 64,
                   "gcso_gateway_frame_t::phase_deltas offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_causal_provenance_report_t, provenance_bits) == 0,
                   "gcso_causal_provenance_report_t::provenance_bits offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_causal_provenance_report_t, dominant_cause_code) == 12,
                   "gcso_causal_provenance_report_t::dominant_cause_code offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_crystallized_attractor_record_t, magic) == 0,
                   "gcso_crystallized_attractor_record_t::magic offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_crystallized_attractor_record_t, zimms_section_offset48) == 16,
                   "gcso_crystallized_attractor_record_t::zimms_section_offset48 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_crystallized_attractor_record_t, provenance_bits) == 48,
                   "gcso_crystallized_attractor_record_t::provenance_bits offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_crystallized_attractor_record_t, checksum_crc32) == 52,
                   "gcso_crystallized_attractor_record_t::checksum_crc32 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_crystallized_attractor_record_t, dominant_slot) == 56,
                   "gcso_crystallized_attractor_record_t::dominant_slot offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_crystallized_attractor_record_t, phase_profile) == 64,
                   "gcso_crystallized_attractor_record_t::phase_profile offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_contrastive_bloom_filter_t, counters) == 0,
                   "gcso_contrastive_bloom_filter_t::counters offset mismatch");

GCSO_EXTERN_C_END

#endif // LIMINIKA_GCSO_TYPES_H