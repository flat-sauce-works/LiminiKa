// name: include/liminika/gcso_types.h
// SPDX-License-Identifier: MIT OR Apache-2.0

/**
 * @file gcso_types.h
 * @brief C-ABI POD types, bitmasks, status codes, and compile-time layout assertions.
 *
 * All struct sizes, alignments, and field offsets in this header are verified at compile time
 * via `GCSO_STATIC_ASSERT` to maintain 1:1 ABI parity with `src/core/src/abi/` and
 * `src/core/src/traits/`.
 *
 * C-ABI symbol prefixes (`dpsr`, `edbc`, `srl`, `zimms`) map to the Rust core modules:
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
 * Domain Constant & Bitmask Identifiers (1:1 with src/core)
 * =================================================================== */
#define GCSO_SNAPSHOT_MAGIC 0x4F534347U         /* "GCSO" in ASCII */
#define GCSO_CRYSTAL_RECORD_MAGIC 0x4C545343U   /* "CSTL" in ASCII */
#define GCSO_CRYSTAL_RECORD_VERSION 0x00000103U /* Unified v0.1.3 */
#define GCSO_FOUR_SECTION_HEADER_BYTES 192U
#define GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET 128U
#define GCSO_Q7_PHASE_ARRAY_LEN 64U
#define GCSO_BLOOM_FILTER_COUNTERS_LEN 64U
#define GCSO_FAST_PATH_SHORTCUT_SLOTS 4U
#define GCSO_LEDGER_HISTORY_LEN 8U
#define GCSO_PAGED_BITMASK_WORDS 4U
#define GCSO_NUM_STIGMERGIC_GRANULES 8U
#define GCSO_NUM_SWARM_MICRO_CELLS GCSO_NUM_STIGMERGIC_GRANULES
#define GCSO_MAX_ATTRACTOR_DOMAINS 8U
#define GCSO_NUM_LIMINAL_APERTURE_CHANNELS 8U
#define GCSO_NUM_CORE_AUTONOMOUS_LAYERS 3U
#define GCSO_CORE_LAYER_LIMINAL 0U
#define GCSO_CORE_LAYER_STIGMERGIC 1U
#define GCSO_CORE_LAYER_EGO 2U

/* 2 Primitives x 2 Drivers Structural Attribution Identifiers (1:1 with constants_pod.rs) */
#define GCSO_SUBSTRATE_LIMINAL 0U
#define GCSO_SUBSTRATE_STIGMERGY 1U
#define GCSO_DRIVER_SYSTEM 0U
#define GCSO_DRIVER_EGO 1U
#define GCSO_ATTRIBUTION_TAG(substrate, driver) \
    ((uint8_t)(((uint8_t)(substrate) & 1U) | (((uint8_t)(driver) & 1U) << 1U)))
#define GCSO_MAX_EGO_PFRAME_NODES 8U
#define GCSO_DISTORTION_DELTA_NO_NEXT_HOP 0xFFFFU
#define GCSO_MAX_DISTORTION_CHAIN_HOPS 4U
#define GCSO_DISTORTION_DELTA_FLAG_ACTIVE (1U << 0)
#define GCSO_DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME (1U << 1)
#define GCSO_DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID (1U << 2)
#define GCSO_VENN_REGION_SHARED_CORE (1U << 0)
#define GCSO_VENN_REGION_SELF_EXCLUSIVE (1U << 1)
#define GCSO_VENN_REGION_KNOWN_UNKNOWN (1U << 2)

/* Liminal Cognitive Aperture Bitmasks (1:1 with primitives/constants_pod.rs) */
#define GCSO_APERTURE_SYSTEM_CONSTRAINT (1U << 0)
#define GCSO_APERTURE_OBJECTIVE_FACT (1U << 1)
#define GCSO_APERTURE_SUBJECTIVE_PERSONA (1U << 2)
#define GCSO_APERTURE_CONTRADICTION_GUARD (1U << 3)
#define GCSO_APERTURE_CAUSAL_PROVENANCE 0x0000000FU
#define GCSO_APERTURE_COGNITIVE_CIRCUIT (1U << 4)
#define GCSO_APERTURE_KV_PHASE_CACHE (1U << 5)
#define GCSO_APERTURE_RELATIVE_COUPLING (1U << 6)
#define GCSO_APERTURE_DISSIPATIVE_CRYSTAL (1U << 7)
#define GCSO_APERTURE_MANIFOLD_GATEWAY 0x000000FFU
#define GCSO_APERTURE_PHASE_STEERING (1U << 8)
#define GCSO_APERTURE_SUBHEAD_ROUTING (1U << 9)
#define GCSO_APERTURE_RANK1_RESIDUAL (1U << 10)
#define GCSO_APERTURE_POINTER_TRAIL (1U << 11)
#define GCSO_APERTURE_FAST_PATH_LEDGER (1U << 12)
#define GCSO_APERTURE_STIGMERGIC_CONSENSUS (1U << 13)
#define GCSO_APERTURE_COHOMOLOGY_EDBC (1U << 14)
#define GCSO_APERTURE_MAPPED_STORAGE (1U << 15)
#define GCSO_APERTURE_ACTION_HUB_WIDE 0x0000FFFFU

/* LiminalEgoField Trigger Bitmasks (1:1 with primitives/constants_pod.rs) */
#define GCSO_EGO_TRIGGER_BASIN_CAPTURE (1U << 0)
#define GCSO_EGO_TRIGGER_INTERNAL_CONFLICT (1U << 1)
#define GCSO_EGO_TRIGGER_LIMINAL_EXTERIOR (1U << 2)
#define GCSO_EGO_TRIGGER_BOUNDARY_CHATTER (1U << 3)
#define GCSO_EGO_TRIGGER_BOUNDARY_STAGNATION (1U << 4)
#define GCSO_EGO_TRIGGER_LAYER_DISTORTION (1U << 5)
#define GCSO_EGO_TRIGGER_ALL_DEFAULT 0x1FU

/* Cognitive Trajectory Scale Granularity & Op Identifiers */
#define GCSO_SCALE_POINTER_BIT 0U
#define GCSO_SCALE_CELL_TRAIL 1U
#define GCSO_SCALE_TYPE_GATEWAY 2U
#define GCSO_SCALE_MACRO_COALITION 3U
#define GCSO_COG_OP_OBSERVE 0U
#define GCSO_COG_OP_DECOMPOSE_PEEL 1U
#define GCSO_COG_OP_SYNTHESIZE_FUSE 2U
#define GCSO_COG_OP_REPULSE_AUDIT 3U
#define GCSO_COG_OP_BRIDGE_SHIFT 4U
#define GCSO_COG_OP_STEER_ALIGN 5U
#define GCSO_COG_OP_ROUTE_SUBHEAD 6U
#define GCSO_COG_OP_CRYSTALLIZE_COMMIT 7U

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

/* 3-Bit Semantic Attractor Provenance / Affinity Identifiers (0..7, matching primitives.rs;
 * decoupled from physical workbench lanes 0..7) */
#define GCSO_ATTRACTOR_SLOT_SYSTEM_BASELINE 0U
#define GCSO_ATTRACTOR_SLOT_PERSONA 1U
#define GCSO_ATTRACTOR_SLOT_KNOWLEDGE 2U
#define GCSO_ATTRACTOR_SLOT_REPULSION 3U
#define GCSO_ATTRACTOR_SLOT_USER_CONTEXT 4U
#define GCSO_ATTRACTOR_SLOT_TEMPORAL 5U
#define GCSO_ATTRACTOR_SLOT_USER_INTENT 6U
#define GCSO_ATTRACTOR_SLOT_CRYSTALLIZED 7U
#define GCSO_ATTRACTOR_SLOT_ID_MASK 0x07U
#define GCSO_DEFAULT_SLOT_COUPLING_MATRIX_8X8 0xFFFFFFFFFFFFFFFFULL

/* Anonymous 8-Lane Workbench Pool Constants (matching primitives.rs) */
#define GCSO_WORKBENCH_LANE_COUNT 8U
#define GCSO_WORKBENCH_LANE_AUTO 0xFFU
#define GCSO_WORKBENCH_LANE_FLAG_PINNED (1U << 0)
#define GCSO_WORKBENCH_LANE_FLAG_REPULSION (1U << 1)
#define GCSO_WORKBENCH_LANE_FLAG_SANDBOX (1U << 2)

/* 5-Bit Action Mode Execution Flags (1:1 bit parity with primitives.rs) */
#define GCSO_ACTION_MODE_DPSR_STEER (1U << 0)    /* 0x01U */
#define GCSO_ACTION_MODE_LAYER_SKIP (1U << 1)    /* 0x02U */
#define GCSO_ACTION_MODE_PHASE_REPULSE (1U << 2) /* 0x04U */
#define GCSO_ACTION_MODE_SUBHEAD_ROUTE (1U << 3) /* 0x08U */
#define GCSO_ACTION_MODE_RANK1_INJECT (1U << 4)  /* 0x10U */
#define GCSO_ACTION_MODE_ALL_MASK 0x1FU

/* Canonical Action Mode Presets (1:1 parity with primitives.rs) */
#define GCSO_ACTION_PRESET_PASSIVE_SIDECAR 0x00U
#define GCSO_ACTION_PRESET_EXPRESS_PHASE (GCSO_ACTION_MODE_DPSR_STEER)
#define GCSO_ACTION_PRESET_INERTIAL_SKIP (GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_LAYER_SKIP)
#define GCSO_ACTION_PRESET_REPULSION_ESCAPE \
    (GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_PHASE_REPULSE | GCSO_ACTION_MODE_SUBHEAD_ROUTE)
#define GCSO_ACTION_PRESET_FULL_ORCHESTRATION \
    (GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_SUBHEAD_ROUTE | GCSO_ACTION_MODE_RANK1_INJECT)

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

/* FastPathLedger Telemetry Ring Ledger Metric Bitflags (1:1 bit parity with telemetry.rs) */
#define GCSO_LEDGER_METRIC_FAST_PATH_HIT (1U << 0)
#define GCSO_LEDGER_METRIC_PHASE_STEERED (1U << 1)
#define GCSO_LEDGER_METRIC_REPULSION_APPLIED (1U << 2)
#define GCSO_LEDGER_METRIC_LAYER_SKIP_COMPENSATED (1U << 3)
#define GCSO_LEDGER_METRIC_SUBHEAD_ROUTED (1U << 4)
#define GCSO_LEDGER_METRIC_RANK1_INJECTED (1U << 5)
#define GCSO_LEDGER_METRIC_LAZY_UNWRAP_APPLIED (1U << 6)
#define GCSO_LEDGER_METRIC_SLOT_SYNERGY_ACTIVE (1U << 7)
#define GCSO_LEDGER_FLAG_COGNITIVE_TRAJECTORY (1U << 0)

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
    GCSO_ERROR_MAPPED_STORAGE_FAILED = -51,
    GCSO_ERROR_SUBHEAD_ROUTING_FAILED = -60,
    GCSO_ERROR_KV_KEYFRAME_SEEK_FAILED = -70,
    GCSO_ERROR_OBSTRUCTION_UNRESOLVED = -80,
    GCSO_ERROR_EXTENSION_NOT_LOADED = -90,
    GCSO_ERROR_LEDGER_FULL = -91,
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
    GCSO_DECISION_PASSIVE_SIDECAR = 0,     GCSO_DECISION_INERTIAL_LAYER_SKIP = 1,
    GCSO_DECISION_EXPRESS_PHASE_STEER = 2, GCSO_DECISION_SUBHEAD_ROUTE = 3,
    GCSO_DECISION_FULL_ORCHESTRATION = 4,  GCSO_DECISION_CRYSTALLIZE_MACRO_BASIN = 5,
    GCSO_DECISION_REPULSION_ESCAPE = 6} gcso_stigmergic_action_decision_enum_t;

/**
 * @brief 3-regime epistemic horizon classification matching Rust EpistemicRegime (0..=2).
 */
typedef enum gcso_epistemic_regime GCSO_ENUM_UNDERLYING(uint8_t){
    GCSO_EPISTEMIC_KNOWN_COHERENT = 0, GCSO_EPISTEMIC_INTERNAL_CONFLICT = 1,
    GCSO_EPISTEMIC_LIMINAL = 2} gcso_epistemic_regime_enum_t;

/**
 * @brief Subjective Ego preset intensity discriminant matching Rust LiminalEgoPreset (0..=3).
 */
typedef enum gcso_liminal_ego_preset GCSO_ENUM_UNDERLYING(uint8_t){
    GCSO_EGO_PRESET_CUSTOM = 0, GCSO_EGO_PRESET_LOW = 1, GCSO_EGO_PRESET_MEDIUM = 2,
    GCSO_EGO_PRESET_HIGH = 3} gcso_liminal_ego_preset_enum_t;

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
 * @brief Dynamic Adaptive Extension Scratchpad (FastPathLedger) slot layout (64 bytes).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_ledger_slot {
    uint32_t mode;
    uint16_t telemetry_ring_head;
    uint16_t telemetry_ring_tail;
    uint32_t cache_hit_count;
    uint32_t auto_tune_flags;
    uint64_t fast_path_bypass_mask;
    uint64_t fast_path_shortcuts[GCSO_FAST_PATH_SHORTCUT_SLOTS];
    uint8_t telemetry_mini_ledger[GCSO_LEDGER_HISTORY_LEN];
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_ledger_slot gcso_ledger_slot_t;

/**
 * @brief Global configuration descriptor structure aligned to 16 bytes (64 bytes total).
 */
/**
 * @brief Memory pairing convention for Rotary Position Embedding (RoPE) channels.
 */
typedef enum gcso_rope_layout {
    GCSO_ROPE_LAYOUT_INTERLEAVED = 0,
    GCSO_ROPE_LAYOUT_HALF_SPLIT = 1
} gcso_rope_layout_t;

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
    uint8_t ledger_mode;
    uint8_t reserved_flags;
    uint32_t rotary_dim;
    uint8_t rope_layout;
    uint8_t reserved[19];
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
struct GCSO_ALIGN_PREFIX(32) gcso_mapped_storage_descriptor {
    uint64_t mapped_address;
    uint64_t file_size_bytes;
    uint64_t dma_buffer_handle;
    uint32_t flags;
    int32_t fd_handle;
    uint8_t reserved[32];
} GCSO_ALIGN_SUFFIX(32);
typedef struct gcso_mapped_storage_descriptor gcso_mapped_storage_descriptor_t;

/**
 * @brief Sub-head group router configuration descriptor (32 bytes).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_subhead_config {
    uint16_t num_fact_heads;
    uint16_t num_logic_heads;
    uint16_t num_explore_heads;
    uint16_t flags;
    float fact_phase_gain;
    float logic_phase_gain;
    float explore_phase_gain;
    uint8_t reserved[12];
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_subhead_config gcso_subhead_config_t;

/**
 * @brief Sparse Residual Adapter Layer (Rank-1 residual) Rank-1 descriptor (64 bytes).
 */
struct GCSO_ALIGN_PREFIX(32) gcso_rank1_residual_descriptor {
    uint32_t layer_idx;
    uint32_t rank;
    uint64_t u_vector_ptr;
    uint64_t v_vector_ptr;
    uint64_t gain_scalar_ptr;
    float scale_factor;
    uint32_t flags;
    uint8_t reserved[24];
} GCSO_ALIGN_SUFFIX(32);
typedef struct gcso_rank1_residual_descriptor gcso_rank1_residual_descriptor_t;

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
    uint64_t rank1_state_offset;
    uint64_t edbc_state_offset;
    uint32_t checksum_crc32;
    uint32_t ledger_slot_offset;
    uint64_t timestamp_epoch_sec;
    uint8_t reserved_padding[56];
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_snapshot_header gcso_snapshot_header_t;

/**
 * @brief KV-keyframe Keyframe KV cache index header structure (64 bytes).
 */
struct GCSO_ALIGN_PREFIX(32) gcso_kv_keyframe_header {
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
typedef struct gcso_kv_keyframe_header gcso_kv_keyframe_header_t;

/**
 * @brief Stigmergic threshold configuration governing pheromone dynamics & 7-regime selection (32
 * bytes).
 */
struct GCSO_ALIGN_PREFIX(32) gcso_stigmergic_thresholds {
    float evaporation_retain;
    float reinforcement_gain;
    float inertial_skip_entropy_thresh;
    float express_steer_density_thresh;
    float subhead_route_density_thresh;
    float rank1_inject_density_thresh;
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
 * @brief Primary S-initial alias for the 16-byte Stigmergic Reconstitution report.
 */
typedef gcso_swarm_resolution_report_t gcso_reconstitution_report_t;

/**
 * @brief 2-Origin Composite Manifold Gateway frame (128 bytes, 64-byte cacheline aligned).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_gateway_frame {
    uint64_t tagged_ptr;
    uint64_t mapped_section_offset48;
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
    uint8_t subhead_fact_heads;
    uint8_t subhead_logic_heads;
    uint8_t subhead_explore_heads;
    uint8_t subhead_fact_gain_q7;
    uint8_t subhead_logic_gain_q7;
    uint8_t subhead_explore_gain_q7;
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
    uint64_t mapped_section_offset48;
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

/**
 * @brief 1-byte compact cognitive step opcode ([Scale(2b) | Op(3b) | Intensity(3b)]).
 */
typedef uint8_t gcso_cognitive_step_op_t;

/**
 * @brief 8-byte register-resident activation output of a cognitive trajectory circuit node.
 */
struct GCSO_ALIGN_PREFIX(8) gcso_cognitive_circuit_activation {
    uint8_t peel_gate_q7;
    uint8_t fuse_gate_q7;
    uint8_t repulse_sensitivity_q7;
    uint8_t selected_op;
    uint8_t preferred_next_scale;
    uint8_t intensity_q7;
    gcso_q7_t circuit_resonance_q7;
    uint8_t trajectory_depth;
} GCSO_ALIGN_SUFFIX(8);
typedef struct gcso_cognitive_circuit_activation gcso_cognitive_circuit_activation_t;

/**
 * @brief 8-byte chronological cognitive trajectory ledger mirroring
 * gcso_ledger_slot_t::telemetry_mini_ledger.
 */
struct GCSO_ALIGN_PREFIX(8) gcso_cognitive_trajectory_ledger {
    gcso_cognitive_step_op_t steps[GCSO_LEDGER_HISTORY_LEN];
} GCSO_ALIGN_SUFFIX(8);
typedef struct gcso_cognitive_trajectory_ledger gcso_cognitive_trajectory_ledger_t;

/**
 * @brief Branchless epistemic horizon measurement over a finite local memory manifold (32 bytes).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_epistemic_horizon_reading {
    float total_memory_support;
    float peak_positive_affinity;
    float internal_conflict_energy;
    float horizon_gate;
    float conflict_severity;
    uint32_t dominant_anchor_id;
    uint32_t opposing_anchor_id;
    uint8_t regime_code;
    uint8_t reserved_align[3];
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_epistemic_horizon_reading gcso_epistemic_horizon_reading_t;

/**
 * @brief Core Layer 0 autonomous LiminalVoid aperture controller (32 bytes, 16-byte aligned).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_liminal_aperture_field {
    uint8_t void_floor_q7[GCSO_NUM_LIMINAL_APERTURE_CHANNELS];
    uint8_t aperture_gain_q7[GCSO_NUM_LIMINAL_APERTURE_CHANNELS];
    gcso_q7_t phase_shift_q7[GCSO_NUM_LIMINAL_APERTURE_CHANNELS];
    uint32_t active_aperture_mask;
    uint32_t adaptation_epochs;
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_liminal_aperture_field gcso_liminal_aperture_field_t;

/**
 * @brief Core Layer 1 autonomous Stigmergic granule approximation controller (32 bytes, 16-byte
 * aligned).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_stigmergic_granule_field {
    uint8_t granule_step_gain_q7[GCSO_NUM_STIGMERGIC_GRANULES];
    uint8_t granule_noise_tolerance_q7[GCSO_NUM_STIGMERGIC_GRANULES];
    uint8_t granule_pheromone_q7[GCSO_NUM_STIGMERGIC_GRANULES];
    uint8_t venn_overlap_mask;
    uint8_t serendipity_excursion_mask;
    uint8_t quorum_threshold;
    uint8_t serendipity_floor_q7;
    uint32_t adaptation_epochs;
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_stigmergic_granule_field gcso_stigmergic_granule_field_t;

/**
 * @brief Venn-diagram multi-attractor overlap and LiminalVoid serendipity report (32 bytes).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_stigmergic_venn_serendipity_report {
    float venn_intersection_energy;
    float penumbra_attenuation_energy;
    float void_excursion_energy;
    uint32_t dominant_overlap_anchor_id;
    uint64_t serendipitous_seed_word;
    uint8_t venn_core_mask;
    uint8_t venn_penumbra_mask;
    uint8_t void_exterior_mask;
    uint8_t serendipity_excursion_mask;
    uint8_t max_overlap_depth;
    uint8_t serendipitous_granule_idx;
    uint8_t serendipity_triggered;
    uint8_t reserved_align;
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_stigmergic_venn_serendipity_report gcso_stigmergic_venn_serendipity_report_t;

/**
 * @brief Normalized intermediate semantic Ego intent axes (16 bytes, 16-byte aligned).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_ego_intent_axes {
    int8_t autonomy_q7;
    int8_t transgression_q7;
    int8_t devotion_q7;
    int8_t logic_q7;
    int8_t caution_q7;
    int8_t defiance_q7;
    int8_t synergy_q7;
    int8_t vitality_q7;
    uint8_t preferred_slot_mask;
    uint8_t active_granule_bias_mask;
    uint8_t confidence_q7;
    uint8_t flags;
    uint32_t signature_hash;
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_ego_intent_axes gcso_ego_intent_axes_t;

/**
 * @brief Wide-aperture Subjective Ego boundary redefinition controller (64 bytes, 64-byte aligned).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_liminal_ego_field {
    gcso_q7_t sovereign_granule_phase_q7[GCSO_NUM_STIGMERGIC_GRANULES];
    int8_t trait_floor_bias_q7[GCSO_NUM_LIMINAL_APERTURE_CHANNELS];
    int8_t trait_gain_bias_q7[GCSO_NUM_LIMINAL_APERTURE_CHANNELS];
    int8_t slot_affinity_bias_q7[GCSO_MAX_ATTRACTOR_DOMAINS];
    uint8_t granule_autonomy_q7[GCSO_NUM_STIGMERGIC_GRANULES];
    uint8_t counter_basin_thrust_q7;
    uint8_t basin_escape_floor_q7;
    int8_t transgression_bias_q7;
    uint8_t irrational_defiance_q7;
    uint8_t trigger_mask;
    uint8_t forced_action_mode_or_mask;
    uint8_t suppressed_action_mode_mask;
    uint8_t preset_code;
    uint32_t ego_signature_hash;
    uint32_t intervention_epochs;
    uint8_t last_intervened_mask;
    uint8_t reserved[7];
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_liminal_ego_field gcso_liminal_ego_field_t;

/**
 * @brief Fragmented micro-ensemble consensus report across 8 SWAR granules (16 bytes, 16-byte
 * aligned).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_stigmergic_fragment_ensemble_report {
    float ensemble_consensus_energy;
    float ensemble_diversity_gain;
    uint8_t quorum_agreed_granule_mask;
    uint8_t split_contested_granule_mask;
    uint8_t ego_tiebreak_granule_mask;
    uint8_t active_fragment_count;
    uint8_t dominant_fragment_slot;
    uint8_t reserved[3];
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_stigmergic_fragment_ensemble_report gcso_stigmergic_fragment_ensemble_report_t;

/**
 * @brief Unified 3-layer autonomous core stack (128 bytes = 2 cache lines, 64-byte aligned).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_autonomous_core_layer_stack {
    gcso_liminal_aperture_field_t liminal_layer;
    gcso_stigmergic_granule_field_t stigmergic_layer;
    gcso_liminal_ego_field_t ego_layer;
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_autonomous_core_layer_stack gcso_autonomous_core_layer_stack_t;

/**
 * @brief Self-introspecting Liminal topology telemetry snapshot (32 bytes, 16-byte aligned).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_liminal_topology_snapshot {
    uint8_t void_floor_q7[GCSO_NUM_LIMINAL_APERTURE_CHANNELS];
    uint8_t aperture_gain_q7[GCSO_NUM_LIMINAL_APERTURE_CHANNELS];
    gcso_q7_t phase_shift_q7[GCSO_NUM_LIMINAL_APERTURE_CHANNELS];
    uint32_t active_aperture_mask;
    uint8_t regime_code;
    uint8_t conflict_severity_q7;
    uint8_t active_crystals;
    uint8_t spawned_daughters;
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_liminal_topology_snapshot gcso_liminal_topology_snapshot_t;

/**
 * @brief Temporal boundary velocity & chattering analyzer (16 bytes, 16-byte aligned).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_liminal_boundary_delta {
    int8_t channel_floor_velocity_q7[GCSO_NUM_LIMINAL_APERTURE_CHANNELS];
    uint8_t granule_transition_mask;
    uint8_t chattering_channel_mask;
    uint8_t kinetic_energy_q7;
    uint8_t stationary_shortcut_ready;
    uint8_t expansion_dominant;
    uint8_t chatter_potential_u8;
    uint8_t phase_transition_pulse;
    uint8_t reserved;
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_liminal_boundary_delta gcso_liminal_boundary_delta_t;

/**
 * @brief Lightweight 2-bit attribution tag byte ([Driver(1b) | Substrate(1b)]).
 */
typedef uint8_t gcso_attribution_tag_t;

/**
 * @brief Unified 64-byte single-cacheline CognitiveSubstrate (Liminal + Stigmergy).
 */
struct GCSO_ALIGN_PREFIX(64) gcso_cognitive_substrate {
    gcso_liminal_aperture_field_t liminal;
    gcso_stigmergic_granule_field_t stigmergy;
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_cognitive_substrate gcso_cognitive_substrate_t;

/**
 * @brief Branchless 16-byte 4-region Venn partition across 8 SWAR granules.
 */
struct GCSO_ALIGN_PREFIX(16) gcso_liminal_venn_partition {
    uint8_t self_mask;
    uint8_t other_mask;
    uint8_t shared_core_mask;
    uint8_t self_exclusive_mask;
    uint8_t known_unknown_mask;
    uint8_t true_void_mask;
    uint8_t bridge_count;
    uint8_t other_concentration_q7;
    uint8_t boundary_sensitivity_q7;
    uint8_t known_unknown_density_q7;
    uint8_t sandbox_eligible;
    uint8_t true_void_aperture_zero;
    uint8_t attribution_tag;
    uint8_t reserved_align[3];
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_liminal_venn_partition gcso_liminal_venn_partition_t;

/**
 * @brief Mechanical SystemSpineState owning physical pinned isolation masks (32 bytes, 16-byte
 * aligned).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_system_spine_state {
    uint32_t pinned_aperture_mask;
    uint8_t pinned_slot_mask;
    uint8_t pinned_action_mode_mask;
    uint8_t ood_void_locked;
    uint8_t irreversible_step_lock;
    uint64_t pinned_compute_mask;
    uint32_t gauge_anchor_epoch;
    uint32_t slow_loop_epoch;
    uint8_t whole_token_bypass_active;
    uint8_t bypass_dwell_steps;
    uint8_t bypass_cooldown_steps;
    uint8_t attribution_tag;
    uint8_t reserved_align[4];
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_system_spine_state gcso_system_spine_state_t;

/**
 * @brief Directed 128-byte P-Frame differential node relative to SystemSpine I-Frame trajectory.
 */
struct GCSO_ALIGN_PREFIX(64) gcso_distortion_delta {
    uint32_t node_id;
    uint32_t anchor_ref_id;
    uint16_t next_hop_idx;
    uint8_t hop_depth;
    uint8_t venn_trigger_mask;
    uint8_t venn_region_selector;
    uint8_t pheromone_q7;
    uint8_t gain_q7;
    uint8_t attribution_tag;
    gcso_ego_intent_axes_t intent_delta;
    uint32_t reinforce_count;
    uint32_t last_active_epoch;
    uint32_t checksum_crc32;
    uint32_t flags;
    uint8_t reserved_header[16];
    gcso_q7_t delta_phase_q7[GCSO_Q7_PHASE_ARRAY_LEN];
} GCSO_ALIGN_SUFFIX(64);
typedef struct gcso_distortion_delta gcso_distortion_delta_t;

/**
 * @brief Asymmetric 32-byte homeostatic DistortionTension state (16-byte aligned).
 */
struct GCSO_ALIGN_PREFIX(16) gcso_distortion_tension {
    uint8_t tension_q7;
    uint8_t deadband_low_q7;
    uint8_t deadband_high_q7;
    uint8_t effective_ego_scale_q7;
    uint8_t excitation_potential_u8;
    uint8_t suppression_cooldown_steps;
    uint8_t boundary_sensitivity_q7;
    uint8_t state_code;
    uint32_t immediate_damping_events;
    uint32_t slow_excitation_events;
    uint8_t last_ego_delta_norm_q7;
    uint8_t attribution_tag;
    uint8_t guard_proximity_q7;
    uint8_t exit_entropy_q7;
    uint32_t cumulative_delta_l1;
    uint32_t step_epochs;
    uint8_t reserved_align[4];
} GCSO_ALIGN_SUFFIX(16);
typedef struct gcso_distortion_tension gcso_distortion_tension_t;

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

struct gcso_ledger_slot_opaque;
typedef struct gcso_ledger_slot_opaque* gcso_ledger_slot_handle_t;

struct gcso_attractor_field_opaque;
typedef struct gcso_attractor_field_opaque* gcso_attractor_field_handle_t;

struct gcso_edbc_controller_opaque;
typedef struct gcso_edbc_controller_opaque* gcso_edbc_controller_handle_t;

struct gcso_dpsr_kernel_opaque;
typedef struct gcso_dpsr_kernel_opaque* gcso_dpsr_kernel_handle_t;

struct gcso_subhead_router_opaque;
typedef struct gcso_subhead_router_opaque* gcso_subhead_router_handle_t;

struct gcso_rank1_residual_opaque;
typedef struct gcso_rank1_residual_opaque* gcso_rank1_residual_handle_t;

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
GCSO_STATIC_ASSERT(sizeof(gcso_ledger_slot_t) == 64, "gcso_ledger_slot_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_config_t) == 64, "gcso_config_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_edbc_state_t) == 64, "gcso_edbc_state_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_mapped_storage_descriptor_t) == 64,
                   "gcso_mapped_storage_descriptor_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_subhead_config_t) == 32, "gcso_subhead_config_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_rank1_residual_descriptor_t) == 64,
                   "gcso_rank1_residual_descriptor_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_snapshot_header_t) == 128, "gcso_snapshot_header_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_kv_keyframe_header_t) == 64,
                   "gcso_kv_keyframe_header_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_stigmergic_thresholds_t) == 32,
                   "gcso_stigmergic_thresholds_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_action_orchestration_report_t) == 32,
                   "gcso_action_orchestration_report_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_swarm_resolution_report_t) == 16,
                   "gcso_swarm_resolution_report_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_reconstitution_report_t) == 16,
                   "gcso_reconstitution_report_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_gateway_frame_t) == 128, "gcso_gateway_frame_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_causal_provenance_report_t) == 16,
                   "gcso_causal_provenance_report_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_crystallized_attractor_record_t) == 128,
                   "gcso_crystallized_attractor_record_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_contrastive_bloom_filter_t) == 64,
                   "gcso_contrastive_bloom_filter_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_cognitive_step_op_t) == 1, "gcso_cognitive_step_op_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_cognitive_circuit_activation_t) == 8,
                   "gcso_cognitive_circuit_activation_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_cognitive_trajectory_ledger_t) == 8,
                   "gcso_cognitive_trajectory_ledger_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_epistemic_horizon_reading_t) == 32,
                   "gcso_epistemic_horizon_reading_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_liminal_aperture_field_t) == 32,
                   "gcso_liminal_aperture_field_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_stigmergic_granule_field_t) == 32,
                   "gcso_stigmergic_granule_field_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_stigmergic_venn_serendipity_report_t) == 32,
                   "gcso_stigmergic_venn_serendipity_report_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_autonomous_core_layer_stack_t) == 128,
                   "gcso_autonomous_core_layer_stack_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_liminal_topology_snapshot_t) == 32,
                   "gcso_liminal_topology_snapshot_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_liminal_boundary_delta_t) == 16,
                   "gcso_liminal_boundary_delta_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_ego_intent_axes_t) == 16, "gcso_ego_intent_axes_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_stigmergic_fragment_ensemble_report_t) == 16,
                   "gcso_stigmergic_fragment_ensemble_report_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_liminal_ego_field_t) == 64,
                   "gcso_liminal_ego_field_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_attribution_tag_t) == 1, "gcso_attribution_tag_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_cognitive_substrate_t) == 64,
                   "gcso_cognitive_substrate_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_liminal_venn_partition_t) == 16,
                   "gcso_liminal_venn_partition_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_system_spine_state_t) == 32,
                   "gcso_system_spine_state_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_distortion_delta_t) == 128, "gcso_distortion_delta_t size mismatch");
GCSO_STATIC_ASSERT(sizeof(gcso_distortion_tension_t) == 32,
                   "gcso_distortion_tension_t size mismatch");

/* Static Assertions for Alignment Invariants Matching src/core/src/abi/ & traits/ */
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_raw_buffer_t) == 16, "gcso_raw_buffer_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_paged_bitmask_t) == 32,
                   "gcso_paged_bitmask_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_pointer_trail_t) == 128,
                   "gcso_pointer_trail_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_ledger_slot_t) == 64, "gcso_ledger_slot_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_config_t) == 16, "gcso_config_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_edbc_state_t) == 32, "gcso_edbc_state_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_mapped_storage_descriptor_t) == 32,
                   "gcso_mapped_storage_descriptor_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_subhead_config_t) == 16,
                   "gcso_subhead_config_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_rank1_residual_descriptor_t) == 32,
                   "gcso_rank1_residual_descriptor_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_snapshot_header_t) == 64,
                   "gcso_snapshot_header_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_kv_keyframe_header_t) == 32,
                   "gcso_kv_keyframe_header_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_stigmergic_thresholds_t) == 32,
                   "gcso_stigmergic_thresholds_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_action_orchestration_report_t) == 32,
                   "gcso_action_orchestration_report_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_swarm_resolution_report_t) == 16,
                   "gcso_swarm_resolution_report_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_reconstitution_report_t) == 16,
                   "gcso_reconstitution_report_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_gateway_frame_t) == 64,
                   "gcso_gateway_frame_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_causal_provenance_report_t) == 16,
                   "gcso_causal_provenance_report_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_crystallized_attractor_record_t) == 64,
                   "gcso_crystallized_attractor_record_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_contrastive_bloom_filter_t) == 64,
                   "gcso_contrastive_bloom_filter_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_cognitive_circuit_activation_t) == 8,
                   "gcso_cognitive_circuit_activation_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_cognitive_trajectory_ledger_t) == 8,
                   "gcso_cognitive_trajectory_ledger_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_epistemic_horizon_reading_t) == 16,
                   "gcso_epistemic_horizon_reading_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_liminal_aperture_field_t) == 16,
                   "gcso_liminal_aperture_field_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_stigmergic_granule_field_t) == 16,
                   "gcso_stigmergic_granule_field_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_stigmergic_venn_serendipity_report_t) == 16,
                   "gcso_stigmergic_venn_serendipity_report_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_autonomous_core_layer_stack_t) == 64,
                   "gcso_autonomous_core_layer_stack_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_liminal_topology_snapshot_t) == 16,
                   "gcso_liminal_topology_snapshot_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_liminal_boundary_delta_t) == 16,
                   "gcso_liminal_boundary_delta_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_ego_intent_axes_t) == 16,
                   "gcso_ego_intent_axes_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_stigmergic_fragment_ensemble_report_t) == 16,
                   "gcso_stigmergic_fragment_ensemble_report_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_liminal_ego_field_t) == 64,
                   "gcso_liminal_ego_field_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_cognitive_substrate_t) == 64,
                   "gcso_cognitive_substrate_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_liminal_venn_partition_t) == 16,
                   "gcso_liminal_venn_partition_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_system_spine_state_t) == 16,
                   "gcso_system_spine_state_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_distortion_delta_t) == 64,
                   "gcso_distortion_delta_t alignment mismatch");
GCSO_STATIC_ASSERT(GCSO_ALIGNOF(gcso_distortion_tension_t) == 16,
                   "gcso_distortion_tension_t alignment mismatch");

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

GCSO_STATIC_ASSERT(offsetof(gcso_ledger_slot_t, mode) == 0,
                   "gcso_ledger_slot_t::mode offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ledger_slot_t, telemetry_ring_head) == 4,
                   "gcso_ledger_slot_t::telemetry_ring_head offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ledger_slot_t, telemetry_ring_tail) == 6,
                   "gcso_ledger_slot_t::telemetry_ring_tail offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ledger_slot_t, cache_hit_count) == 8,
                   "gcso_ledger_slot_t::cache_hit_count offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ledger_slot_t, auto_tune_flags) == 12,
                   "gcso_ledger_slot_t::auto_tune_flags offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ledger_slot_t, fast_path_bypass_mask) == 16,
                   "gcso_ledger_slot_t::fast_path_bypass_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ledger_slot_t, fast_path_shortcuts) == 24,
                   "gcso_ledger_slot_t::fast_path_shortcuts offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ledger_slot_t, telemetry_mini_ledger) == 56,
                   "gcso_ledger_slot_t::telemetry_mini_ledger offset mismatch");

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
GCSO_STATIC_ASSERT(offsetof(gcso_config_t, ledger_mode) == 38,
                   "gcso_config_t::ledger_mode offset mismatch");
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

GCSO_STATIC_ASSERT(offsetof(gcso_mapped_storage_descriptor_t, mapped_address) == 0,
                   "gcso_mapped_storage_descriptor_t::mapped_address offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_mapped_storage_descriptor_t, file_size_bytes) == 8,
                   "gcso_mapped_storage_descriptor_t::file_size_bytes offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_mapped_storage_descriptor_t, dma_buffer_handle) == 16,
                   "gcso_mapped_storage_descriptor_t::dma_buffer_handle offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_mapped_storage_descriptor_t, flags) == 24,
                   "gcso_mapped_storage_descriptor_t::flags offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_mapped_storage_descriptor_t, fd_handle) == 28,
                   "gcso_mapped_storage_descriptor_t::fd_handle offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_subhead_config_t, num_fact_heads) == 0,
                   "gcso_subhead_config_t::num_fact_heads offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_subhead_config_t, num_logic_heads) == 2,
                   "gcso_subhead_config_t::num_logic_heads offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_subhead_config_t, num_explore_heads) == 4,
                   "gcso_subhead_config_t::num_explore_heads offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_subhead_config_t, flags) == 6,
                   "gcso_subhead_config_t::flags offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_subhead_config_t, fact_phase_gain) == 8,
                   "gcso_subhead_config_t::fact_phase_gain offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_subhead_config_t, logic_phase_gain) == 12,
                   "gcso_subhead_config_t::logic_phase_gain offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_subhead_config_t, explore_phase_gain) == 16,
                   "gcso_subhead_config_t::explore_phase_gain offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_rank1_residual_descriptor_t, layer_idx) == 0,
                   "gcso_rank1_residual_descriptor_t::layer_idx offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_rank1_residual_descriptor_t, rank) == 4,
                   "gcso_rank1_residual_descriptor_t::rank offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_rank1_residual_descriptor_t, u_vector_ptr) == 8,
                   "gcso_rank1_residual_descriptor_t::u_vector_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_rank1_residual_descriptor_t, v_vector_ptr) == 16,
                   "gcso_rank1_residual_descriptor_t::v_vector_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_rank1_residual_descriptor_t, gain_scalar_ptr) == 24,
                   "gcso_rank1_residual_descriptor_t::gain_scalar_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_rank1_residual_descriptor_t, scale_factor) == 32,
                   "gcso_rank1_residual_descriptor_t::scale_factor offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_rank1_residual_descriptor_t, flags) == 36,
                   "gcso_rank1_residual_descriptor_t::flags offset mismatch");

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
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, rank1_state_offset) == 40,
                   "gcso_snapshot_header_t::rank1_state_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, edbc_state_offset) == 48,
                   "gcso_snapshot_header_t::edbc_state_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, checksum_crc32) == 56,
                   "gcso_snapshot_header_t::checksum_crc32 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, ledger_slot_offset) == 60,
                   "gcso_snapshot_header_t::ledger_slot_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_snapshot_header_t, timestamp_epoch_sec) == 64,
                   "gcso_snapshot_header_t::timestamp_epoch_sec offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_kv_keyframe_header_t, frame_type) == 0,
                   "gcso_kv_keyframe_header_t::frame_type offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_kv_keyframe_header_t, token_index) == 4,
                   "gcso_kv_keyframe_header_t::token_index offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_kv_keyframe_header_t, gop_length) == 8,
                   "gcso_kv_keyframe_header_t::gop_length offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_kv_keyframe_header_t, composite_vector_length) == 12,
                   "gcso_kv_keyframe_header_t::composite_vector_length offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_kv_keyframe_header_t, von_mises_kappa) == 16,
                   "gcso_kv_keyframe_header_t::von_mises_kappa offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_kv_keyframe_header_t, sparse_scalar_residual) == 20,
                   "gcso_kv_keyframe_header_t::sparse_scalar_residual offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_kv_keyframe_header_t, icache_payload_offset) == 24,
                   "gcso_kv_keyframe_header_t::icache_payload_offset offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_kv_keyframe_header_t, pcache_payload_offset) == 32,
                   "gcso_kv_keyframe_header_t::pcache_payload_offset offset mismatch");

/* Static Assertions for Gateway, Toroidal k-Means, Swarm & Orchestration Structs */
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
GCSO_STATIC_ASSERT(offsetof(gcso_gateway_frame_t, mapped_section_offset48) == 8,
                   "gcso_gateway_frame_t::mapped_section_offset48 offset mismatch");
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
GCSO_STATIC_ASSERT(offsetof(gcso_crystallized_attractor_record_t, mapped_section_offset48) == 16,
                   "gcso_crystallized_attractor_record_t::mapped_section_offset48 offset mismatch");
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

GCSO_STATIC_ASSERT(offsetof(gcso_reconstitution_report_t, evolved_ptr) == 0,
                   "gcso_reconstitution_report_t::evolved_ptr offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_reconstitution_report_t, quorum_mask) == 12,
                   "gcso_reconstitution_report_t::quorum_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_reconstitution_report_t, hot_path_resolved) == 15,
                   "gcso_reconstitution_report_t::hot_path_resolved offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_cognitive_circuit_activation_t, peel_gate_q7) == 0,
                   "gcso_cognitive_circuit_activation_t::peel_gate_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_cognitive_circuit_activation_t, trajectory_depth) == 7,
                   "gcso_cognitive_circuit_activation_t::trajectory_depth offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_cognitive_trajectory_ledger_t, steps) == 0,
                   "gcso_cognitive_trajectory_ledger_t::steps offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_epistemic_horizon_reading_t, total_memory_support) == 0,
                   "gcso_epistemic_horizon_reading_t::total_memory_support offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_epistemic_horizon_reading_t, dominant_anchor_id) == 20,
                   "gcso_epistemic_horizon_reading_t::dominant_anchor_id offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_epistemic_horizon_reading_t, regime_code) == 28,
                   "gcso_epistemic_horizon_reading_t::regime_code offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_liminal_aperture_field_t, void_floor_q7) == 0,
                   "gcso_liminal_aperture_field_t::void_floor_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_aperture_field_t, active_aperture_mask) == 24,
                   "gcso_liminal_aperture_field_t::active_aperture_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_aperture_field_t, adaptation_epochs) == 28,
                   "gcso_liminal_aperture_field_t::adaptation_epochs offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_stigmergic_granule_field_t, granule_step_gain_q7) == 0,
                   "gcso_stigmergic_granule_field_t::granule_step_gain_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_stigmergic_granule_field_t, venn_overlap_mask) == 24,
                   "gcso_stigmergic_granule_field_t::venn_overlap_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_stigmergic_granule_field_t, serendipity_floor_q7) == 27,
                   "gcso_stigmergic_granule_field_t::serendipity_floor_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_stigmergic_granule_field_t, adaptation_epochs) == 28,
                   "gcso_stigmergic_granule_field_t::adaptation_epochs offset mismatch");

GCSO_STATIC_ASSERT(
    offsetof(gcso_stigmergic_venn_serendipity_report_t, venn_intersection_energy) == 0,
    "gcso_stigmergic_venn_serendipity_report_t::venn_intersection_energy offset mismatch");
GCSO_STATIC_ASSERT(
    offsetof(gcso_stigmergic_venn_serendipity_report_t, dominant_overlap_anchor_id) == 12,
    "gcso_stigmergic_venn_serendipity_report_t::dominant_overlap_anchor_id offset mismatch");
GCSO_STATIC_ASSERT(
    offsetof(gcso_stigmergic_venn_serendipity_report_t, serendipitous_seed_word) == 16,
    "gcso_stigmergic_venn_serendipity_report_t::serendipitous_seed_word offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_stigmergic_venn_serendipity_report_t, venn_core_mask) == 24,
                   "gcso_stigmergic_venn_serendipity_report_t::venn_core_mask offset mismatch");
GCSO_STATIC_ASSERT(
    offsetof(gcso_stigmergic_venn_serendipity_report_t, serendipity_triggered) == 30,
    "gcso_stigmergic_venn_serendipity_report_t::serendipity_triggered offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_autonomous_core_layer_stack_t, liminal_layer) == 0,
                   "gcso_autonomous_core_layer_stack_t::liminal_layer offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_autonomous_core_layer_stack_t, stigmergic_layer) == 32,
                   "gcso_autonomous_core_layer_stack_t::stigmergic_layer offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_autonomous_core_layer_stack_t, ego_layer) == 64,
                   "gcso_autonomous_core_layer_stack_t::ego_layer offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_ego_intent_axes_t, preferred_slot_mask) == 8,
                   "gcso_ego_intent_axes_t::preferred_slot_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ego_intent_axes_t, active_granule_bias_mask) == 9,
                   "gcso_ego_intent_axes_t::active_granule_bias_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ego_intent_axes_t, flags) == 11,
                   "gcso_ego_intent_axes_t::flags offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_ego_intent_axes_t, signature_hash) == 12,
                   "gcso_ego_intent_axes_t::signature_hash offset mismatch");

GCSO_STATIC_ASSERT(
    offsetof(gcso_stigmergic_fragment_ensemble_report_t, quorum_agreed_granule_mask) == 8,
    "gcso_stigmergic_fragment_ensemble_report_t::quorum_agreed_granule_mask offset mismatch");
GCSO_STATIC_ASSERT(
    offsetof(gcso_stigmergic_fragment_ensemble_report_t, ego_tiebreak_granule_mask) == 10,
    "gcso_stigmergic_fragment_ensemble_report_t::ego_tiebreak_granule_mask offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_liminal_topology_snapshot_t, void_floor_q7) == 0,
                   "gcso_liminal_topology_snapshot_t::void_floor_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_topology_snapshot_t, active_aperture_mask) == 24,
                   "gcso_liminal_topology_snapshot_t::active_aperture_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_topology_snapshot_t, regime_code) == 28,
                   "gcso_liminal_topology_snapshot_t::regime_code offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_topology_snapshot_t, spawned_daughters) == 31,
                   "gcso_liminal_topology_snapshot_t::spawned_daughters offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_liminal_boundary_delta_t, channel_floor_velocity_q7) == 0,
                   "gcso_liminal_boundary_delta_t::channel_floor_velocity_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_boundary_delta_t, kinetic_energy_q7) == 10,
                   "gcso_liminal_boundary_delta_t::kinetic_energy_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_boundary_delta_t, stationary_shortcut_ready) == 11,
                   "gcso_liminal_boundary_delta_t::stationary_shortcut_ready offset mismatch");

GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, sovereign_granule_phase_q7) == 0,
                   "gcso_liminal_ego_field_t::sovereign_granule_phase_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, trait_floor_bias_q7) == 8,
                   "gcso_liminal_ego_field_t::trait_floor_bias_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, trait_gain_bias_q7) == 16,
                   "gcso_liminal_ego_field_t::trait_gain_bias_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, slot_affinity_bias_q7) == 24,
                   "gcso_liminal_ego_field_t::slot_affinity_bias_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, granule_autonomy_q7) == 32,
                   "gcso_liminal_ego_field_t::granule_autonomy_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, counter_basin_thrust_q7) == 40,
                   "gcso_liminal_ego_field_t::counter_basin_thrust_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, basin_escape_floor_q7) == 41,
                   "gcso_liminal_ego_field_t::basin_escape_floor_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, trigger_mask) == 44,
                   "gcso_liminal_ego_field_t::trigger_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, preset_code) == 47,
                   "gcso_liminal_ego_field_t::preset_code offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, ego_signature_hash) == 48,
                   "gcso_liminal_ego_field_t::ego_signature_hash offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, intervention_epochs) == 52,
                   "gcso_liminal_ego_field_t::intervention_epochs offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_ego_field_t, last_intervened_mask) == 56,
                   "gcso_liminal_ego_field_t::last_intervened_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_cognitive_substrate_t, liminal) == 0,
                   "gcso_cognitive_substrate_t::liminal offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_cognitive_substrate_t, stigmergy) == 32,
                   "gcso_cognitive_substrate_t::stigmergy offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_venn_partition_t, shared_core_mask) == 2,
                   "gcso_liminal_venn_partition_t::shared_core_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_venn_partition_t, known_unknown_mask) == 4,
                   "gcso_liminal_venn_partition_t::known_unknown_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_liminal_venn_partition_t, boundary_sensitivity_q7) == 8,
                   "gcso_liminal_venn_partition_t::boundary_sensitivity_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_system_spine_state_t, pinned_compute_mask) == 8,
                   "gcso_system_spine_state_t::pinned_compute_mask offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_system_spine_state_t, whole_token_bypass_active) == 24,
                   "gcso_system_spine_state_t::whole_token_bypass_active offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_distortion_delta_t, intent_delta) == 16,
                   "gcso_distortion_delta_t::intent_delta offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_distortion_delta_t, delta_phase_q7) == 64,
                   "gcso_distortion_delta_t::delta_phase_q7 offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_distortion_tension_t, immediate_damping_events) == 8,
                   "gcso_distortion_tension_t::immediate_damping_events offset mismatch");
GCSO_STATIC_ASSERT(offsetof(gcso_distortion_tension_t, slow_excitation_events) == 12,
                   "gcso_distortion_tension_t::slow_excitation_events offset mismatch");

GCSO_EXTERN_C_END

#endif // LIMINIKA_GCSO_TYPES_H