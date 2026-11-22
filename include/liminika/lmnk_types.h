// name: include/liminika/lmnk_types.h
// SPDX-License-Identifier: MIT OR Apache-2.0

/**
 * @file lmnk_types.h
 * @brief LiminiKa v0.2.0 (`0x00000200`) canonical ABI types, POD layouts, and status codes.
 *
 * Portable Device-Ready POD Contract:
 * - Canonical byte order across the FFI and container surface is Little-Endian.
 * - Device-ready POD structures strictly use fixed-width primitives (`int8_t`, `uint8_t`,
 *   `uint16_t`, `uint32_t`, `uint64_t`, `float`), never `size_t`/`uintptr_t` or host pointers,
 *   have zero implicit compiler padding (all padding is explicit `reserved*` fields), use
 *   power-of-two alignments/sizes (`16 B..=128 B` where specified), and lay out 64-head phase
 *   arrays (`lmnk_q7_t[64]`) at 64-byte boundaries directly readable as `u32x16` / `u64x8`
 *   (compatible with WGSL, Vulkan `std430`, Metal, CUDA, x86_64 64B cachelines, and ARM64 128B CL).
 * - Host FFI Descriptor classification: `lmnk_rank1_descriptor_t` carries raw 64-bit host vector
 *   pointers (`u_vector_ptr`, `v_vector_ptr`, `gain_scalar_ptr`) and is scoped exclusively to
 * CPU/FFI boundaries rather than cross-device shader buffers.
 *
 * Canonical 4-State Lacuna Classification:
 * - `0` (`Unobserved`): well-defined zero-aperture non-observation
 * - `1` (`SelfRef`): stationary self-reference inside deadband
 * - `2` (`Exterior`): observed exterior delta beyond deadband
 * - `3` (`Undefined`): non-finite or out-of-range fault state requiring immediate isolation
 */

#ifndef LIMINIKA_LMNK_TYPES_H
#define LIMINIKA_LMNK_TYPES_H

#include "lmnk_config.h"

#include <stddef.h>
#include <stdint.h>

LMNK_EXTERN_C_BEGIN

/* -------------------------------------------------------------------------- */
/* Version & capability constants                                             */
/* -------------------------------------------------------------------------- */
#define LMNK_ABI_VERSION_MAJOR 0U
#define LMNK_ABI_VERSION_MINOR 2U
#define LMNK_ABI_VERSION_PATCH 0U
#define LMNK_ABI_VERSION_HEX 0x00000200U
#define LMNK_ABI_VERSION_STRING "0.2.0"

#define LMNK_CAP_SWAR_BASELINE (1ULL << 0)
#define LMNK_CAP_SIMD_AVX2 (1ULL << 1)
#define LMNK_CAP_SIMD_NEON (1ULL << 2)
#define LMNK_CAP_SIMD_WASM128 (1ULL << 3)
#define LMNK_CAP_ZERO_ALLOC_CORE (1ULL << 4)
#define LMNK_CAP_INPLACE_CONTEXT_INIT (1ULL << 5)

/* -------------------------------------------------------------------------- */
/* Primitive scalar aliases & container format constants                      */
/* -------------------------------------------------------------------------- */
typedef int8_t lmnk_q7_t;
typedef uint64_t lmnk_capability_flags_t;
typedef uint64_t lmnk_tagged_ptr_t;
typedef uint8_t lmnk_action_decision_t;
typedef uint8_t lmnk_attribution_tag_t;
typedef int32_t lmnk_status_t;

#define LMNK_CONTAINER_MAGIC 0x4B4E4D4CU    /* "LMNK" in Little-Endian ASCII */
#define LMNK_TOPOLOGY_MAGIC 0x504F544CU     /* "LTOP" in Little-Endian ASCII */
#define LMNK_PHASE_ANCHOR_MAGIC 0x524B4E4CU /* "LNKR" in Little-Endian ASCII */
#define LMNK_PHASE_ANCHOR_VERSION 0x00000200U
#define LMNK_CONTAINER_HEADER_BYTES 64U
#define LMNK_SECTION_ENTRY_BYTES 16U
#define LMNK_MAX_CONTAINER_SECTIONS 32U
#define LMNK_MAX_CONTAINER_BYTES (64U * 1024U * 1024U)

#define LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT (1U << 0)
#define LMNK_CONTAINER_FLAG_COMPILED_CACHE (1U << 1)
#define LMNK_CONTAINER_FLAG_PORTABLE_PACKAGE (1U << 2)

#define LMNK_SECTION_RUNTIME_SNAPSHOT 1U
#define LMNK_SECTION_PHASE_ANCHORS 2U
#define LMNK_SECTION_POINTER_TRAILS 3U
#define LMNK_SECTION_BLOOM_FILTER 4U
#define LMNK_SECTION_LEDGER_SLOT 5U
#define LMNK_SECTION_CAUSAL_PROVENANCE 6U
#define LMNK_SECTION_LOGICAL_PFRAMES 7U
#define LMNK_SECTION_RECALL_METADATA 8U
#define LMNK_SECTION_CONTENT_HASH 0xFF01U
#define LMNK_SECTION_SIGNATURE 0xFF02U

#define LMNK_TOPOLOGY_LAYER_COUNT 5U
#define LMNK_TOPOLOGY_CHANNEL_COUNT 8U
#define LMNK_DAG_FLAG_Y_SHAPED_PARALLEL_ROOTS (1U << 0)
#define LMNK_DAG_FLAG_SPINE_EGO_ISOLATED (1U << 1)
#define LMNK_DAG_FLAG_AXIOM_CUT_LACUNA_ORTHOGONAL (1U << 2)
#define LMNK_DAG_FLAG_NO_STD_CORE (1U << 3)
#define LMNK_TOPOLOGY_PINNED_CHANNEL_MASK ((uint8_t)((1U << 0) | (1U << 3)))
#define LMNK_TOPOLOGY_SANDBOX_CHANNEL_MASK ((uint8_t)(1U << 7))

#define LMNK_Q7_PHASE_ARRAY_LEN 64U
#define LMNK_MAX_ATTENTION_HEADS 64U
#define LMNK_NUM_STIGMERGIC_GRANULES 8U
#define LMNK_HEADS_PER_STIGMERGIC_GRANULE 8U
#define LMNK_MAX_ATTRACTOR_DOMAINS 8U
#define LMNK_NUM_LIMINAL_THRESHOLD_CHANNELS 8U
#define LMNK_BLOOM_FILTER_COUNTERS_LEN 64U
#define LMNK_FAST_PATH_SHORTCUT_SLOTS 4U
#define LMNK_LEDGER_HISTORY_LEN 8U
#define LMNK_PAGED_BITMASK_WORDS 4U
#define LMNK_MAX_PHASE_ANCHORS 64U
#define LMNK_MAX_EGO_PFRAME_NODES 8U
#define LMNK_DISTORTION_DELTA_NO_NEXT_HOP 0xFFFFU
#define LMNK_MAX_DISTORTION_CHAIN_HOPS 4U

/* -------------------------------------------------------------------------- */
/* Status codes                                                               */
/* -------------------------------------------------------------------------- */
#define LMNK_OK ((lmnk_status_t)0)
#define LMNK_ERR_INVALID_ARGUMENT ((lmnk_status_t) - 1)
#define LMNK_ERR_OUT_OF_MEMORY ((lmnk_status_t) - 2)
#define LMNK_ERR_BUFFER_TOO_SMALL ((lmnk_status_t) - 3)
#define LMNK_ERR_PANIC_CAUGHT ((lmnk_status_t) - 4)
#define LMNK_ERR_NULL_POINTER ((lmnk_status_t) - 5)
#define LMNK_ERR_INVALID_STATE ((lmnk_status_t) - 6)
#define LMNK_ERR_VERSION_MISMATCH ((lmnk_status_t) - 7)
#define LMNK_ERR_ALIGNMENT ((lmnk_status_t) - 8)
#define LMNK_ERR_IO_FAILURE ((lmnk_status_t) - 9)
#define LMNK_ERR_SUBSTRATE_FULL ((lmnk_status_t) - 10)
#define LMNK_ERR_NOT_IMPLEMENTED ((lmnk_status_t) - 11)
#define LMNK_ERR_ANCHOR_NOT_FOUND ((lmnk_status_t) - 20)
#define LMNK_ERR_PHASE_OVERFLOW ((lmnk_status_t) - 30)
#define LMNK_ERR_PHASE_STEP_UNDERFLOW ((lmnk_status_t) - 31)
#define LMNK_ERR_ENTROPY_SINGULARITY ((lmnk_status_t) - 40)
#define LMNK_ERR_CONTAINER_CORRUPTED ((lmnk_status_t) - 50)
#define LMNK_ERR_STORAGE_MAP_FAILED ((lmnk_status_t) - 51)
#define LMNK_ERR_TOPOLOGY_MISMATCH ((lmnk_status_t) - 52)
#define LMNK_ERR_SUBHEAD_ROUTING_FAILED ((lmnk_status_t) - 60)
#define LMNK_ERR_KV_KEYFRAME_SEEK_FAILED ((lmnk_status_t) - 70)
#define LMNK_ERR_OBSTRUCTION_UNRESOLVED ((lmnk_status_t) - 80)
#define LMNK_ERR_EXTENSION_NOT_LOADED ((lmnk_status_t) - 90)
#define LMNK_ERR_LEDGER_FULL ((lmnk_status_t) - 91)
#define LMNK_ERR_UNKNOWN ((lmnk_status_t) - 0x7FFFFFFF)

#define LMNK_SUCCEEDED(status) ((status) == LMNK_OK)
#define LMNK_FAILED(status) ((status) != LMNK_OK)

/* -------------------------------------------------------------------------- */
/* Primitive/Driver attribution, Lacuna states, Scope & Causal constants      */
/* -------------------------------------------------------------------------- */
#define LMNK_PRIMITIVE_LIMINAL 0U
#define LMNK_PRIMITIVE_STIGMERGY 1U
#define LMNK_DRIVER_SYSTEM 0U
#define LMNK_DRIVER_EGO 1U
#define LMNK_ATTRIBUTION_TAG(primitive, driver) \
    ((uint8_t)((((driver) & 0x01U) << 1) | ((primitive) & 0x01U)))

#define LMNK_LACUNA_UNOBSERVED 0U
#define LMNK_LACUNA_SELF_REF 1U
#define LMNK_LACUNA_EXTERIOR 2U
#define LMNK_LACUNA_UNDEFINED 3U

#define LMNK_SCOPE_REGION_SELF_REF (1U << 0)
#define LMNK_SCOPE_REGION_INTERIOR (1U << 1)
#define LMNK_SCOPE_REGION_EXTERIOR (1U << 2)

#define LMNK_LACUNA_PARTITION_FLAG_SANDBOX_ELIGIBLE (1U << 0)
#define LMNK_LACUNA_PARTITION_FLAG_UNOBSERVED_APERTURE_ZERO (1U << 1)
#define LMNK_LACUNA_PARTITION_FLAG_HAS_UNDEFINED_BOTTOM (1U << 2)
#define LMNK_LACUNA_PARTITION_FLAG_SELF_ALIGNED (1U << 3)

#define LMNK_TAGGED_PTR_ADDR_MASK 0x0000FFFFFFFFFFFFULL
#define LMNK_TAGGED_PTR_TAG_MASK 0x00FF000000000000ULL
#define LMNK_TAGGED_PTR_PHASE_MASK 0xFF00000000000000ULL
#define LMNK_TAGGED_PTR_TAG_SHIFT 48U
#define LMNK_TAGGED_PTR_PHASE_SHIFT 56U

#define LMNK_ATTRACTOR_SLOT_ID_MASK 0x07U
#define LMNK_ATTRACTOR_SLOT_SYSTEM_BASELINE 0U
#define LMNK_ATTRACTOR_SLOT_PERSONA 1U
#define LMNK_ATTRACTOR_SLOT_KNOWLEDGE 2U
#define LMNK_ATTRACTOR_SLOT_REPULSION 3U
#define LMNK_ATTRACTOR_SLOT_USER_CONTEXT 4U
#define LMNK_ATTRACTOR_SLOT_TEMPORAL 5U
#define LMNK_ATTRACTOR_SLOT_USER_INTENT 6U
#define LMNK_ATTRACTOR_SLOT_CRYSTALLIZED 7U

#define LMNK_ACTION_MODE_PHASE_STEER (1U << 0)
#define LMNK_ACTION_MODE_LAYER_SKIP (1U << 1)
#define LMNK_ACTION_MODE_PHASE_REPULSE (1U << 2)
#define LMNK_ACTION_MODE_SUBHEAD_ROUTE (1U << 3)
#define LMNK_ACTION_MODE_RANK1_INJECT (1U << 4)

#define LMNK_ACTION_PRESET_PASSIVE_SIDECAR 0x00U
#define LMNK_ACTION_PRESET_EXPRESS_PHASE LMNK_ACTION_MODE_PHASE_STEER
#define LMNK_ACTION_PRESET_INERTIAL_SKIP \
    (LMNK_ACTION_MODE_PHASE_STEER | LMNK_ACTION_MODE_LAYER_SKIP)
#define LMNK_ACTION_PRESET_REPULSION_ESCAPE \
    (LMNK_ACTION_MODE_PHASE_STEER | LMNK_ACTION_MODE_PHASE_REPULSE | LMNK_ACTION_MODE_SUBHEAD_ROUTE)
#define LMNK_ACTION_PRESET_FULL_ORCHESTRATION \
    (LMNK_ACTION_MODE_PHASE_STEER | LMNK_ACTION_MODE_SUBHEAD_ROUTE | LMNK_ACTION_MODE_RANK1_INJECT)

#define LMNK_ROPE_LAYOUT_INTERLEAVED 0U
#define LMNK_ROPE_LAYOUT_HALF_SPLIT 1U

#define LMNK_STIGMERGIC_DECISION_PASSIVE_SIDECAR 0U
#define LMNK_STIGMERGIC_DECISION_INERTIAL_LAYER_SKIP 1U
#define LMNK_STIGMERGIC_DECISION_EXPRESS_PHASE_STEER 2U
#define LMNK_STIGMERGIC_DECISION_SUBHEAD_ROUTE 3U
#define LMNK_STIGMERGIC_DECISION_FULL_ORCHESTRATION 4U
#define LMNK_STIGMERGIC_DECISION_CRYSTALLIZE_MACRO_BASIN 5U
#define LMNK_STIGMERGIC_DECISION_REPULSION_ESCAPE 6U

#define LMNK_TRAIL_FLAG_PHASE_STEER_ACTIVE (1U << 0)
#define LMNK_TRAIL_FLAG_REPULSION_ACTIVE (1U << 1)
#define LMNK_TRAIL_FLAG_CRYSTALLIZED_CANDIDATE (1U << 2)
#define LMNK_TRAIL_FLAG_SYNERGY_RESONANCE (1U << 3)
#define LMNK_TRAIL_FLAG_LIMINAL_BEAT_ACTIVE (1U << 4)
#define LMNK_TRAIL_FLAG_INERTIAL_SKIP_ACTIVE (1U << 5)
#define LMNK_TRAIL_FLAG_SWARM_RESOLVED (1U << 6)
#define LMNK_TRAIL_FLAG_SWARM_ESCALATED (1U << 7)
#define LMNK_TRAIL_FLAG_PHEROMONE_RISING (1U << 16)

#define LMNK_ANCHOR_ATTRIBUTION_NOMINAL 0U
#define LMNK_ANCHOR_LINEAGE_MATURE_BIT (1U << 12)

#define LMNK_ENTROPY_GATE_MODE_NORMAL_ENTROPY 0U
#define LMNK_ENTROPY_GATE_MODE_PITCHFORK_BIFURCATION 1U
#define LMNK_ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION 2U

#define LMNK_CAUSAL_AXIS_SYSTEM_CONSTRAINT (1U << 0)
#define LMNK_CAUSAL_AXIS_OBJECTIVE_FACT (1U << 1)
#define LMNK_CAUSAL_AXIS_SUBJECTIVE_PERSONA (1U << 2)
#define LMNK_CAUSAL_AXIS_CONTRADICTION_GUARD (1U << 3)
#define LMNK_CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE (1U << 8)
#define LMNK_CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK (1U << 9)
#define LMNK_CAUSAL_CAUSE_PERSONA_FACT_COLLISION (1U << 10)
#define LMNK_CAUSAL_CAUSE_COHOMOLOGY_REPULSION (1U << 11)
#define LMNK_CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED (1U << 12)

#define LMNK_EGO_PRESET_CUSTOM 0U
#define LMNK_EGO_PRESET_LOW 1U
#define LMNK_EGO_PRESET_MEDIUM 2U
#define LMNK_EGO_PRESET_HIGH 3U

#define LMNK_EGO_TRIGGER_BASIN_CAPTURE (1U << 0)
#define LMNK_EGO_TRIGGER_INTERNAL_CONFLICT (1U << 1)
#define LMNK_EGO_TRIGGER_LIMINAL_EXTERIOR (1U << 2)
#define LMNK_EGO_TRIGGER_BOUNDARY_CHATTER (1U << 3)
#define LMNK_EGO_TRIGGER_BOUNDARY_STAGNATION (1U << 4)
#define LMNK_EGO_TRIGGER_LAYER_DISTORTION (1U << 5)
#define LMNK_EGO_TRIGGER_ALL_DEFAULT 0x1FU

#define LMNK_SPINE_BYPASS_VETO_NONE 0x00U
#define LMNK_SPINE_BYPASS_VETO_GUARD_ACTIVE (1U << 0)
#define LMNK_SPINE_BYPASS_VETO_REPULSION_ACTIVE (1U << 1)
#define LMNK_SPINE_BYPASS_VETO_PINNED_LOCK (1U << 2)

#define LMNK_DISTORTION_DELTA_FLAG_ACTIVE (1U << 0)
#define LMNK_DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME (1U << 1)
#define LMNK_DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID (1U << 2)

#define LMNK_DISTORTION_TENSION_BALANCED_DEADBAND 0U
#define LMNK_DISTORTION_TENSION_PASSIVE_EXCITATION 1U
#define LMNK_DISTORTION_TENSION_ACTIVE_IMMEDIATE_DAMPING 2U

/* -------------------------------------------------------------------------- */
/* Canonical C-ABI POD structures (exact 1:1 Rust repr(C, align) parity)      */
/* -------------------------------------------------------------------------- */

struct LMNK_ALIGN_PREFIX(4) lmnk_descriptor_header {
    uint32_t struct_size;
    uint32_t abi_version;
} LMNK_ALIGN_SUFFIX(4);
typedef struct lmnk_descriptor_header lmnk_descriptor_header_t;

struct LMNK_ALIGN_PREFIX(32) lmnk_paged_bitmask {
    uint64_t bits[4];
} LMNK_ALIGN_SUFFIX(32);
typedef struct lmnk_paged_bitmask lmnk_paged_bitmask_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_pointer_trail {
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
    uint8_t reserved_header[8];
    lmnk_q7_t accumulated_phase_delta[64];
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_pointer_trail lmnk_pointer_trail_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_ledger_slot {
    uint32_t mode;
    uint16_t telemetry_ring_head;
    uint16_t telemetry_ring_tail;
    uint32_t cache_hit_count;
    uint32_t auto_tune_flags;
    uint64_t fast_path_bypass_mask;
    uint64_t fast_path_shortcuts[4];
    uint8_t telemetry_mini_ledger[8];
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_ledger_slot lmnk_ledger_slot_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_config {
    uint32_t head_dim;
    uint32_t num_heads;
    uint32_t paged_block_size;
    float q7_phase_scale;
    float phase_clamp_max_rad;
    float phase_step_min_rad;
    float entropy_singularity_eps;
    uint32_t max_prompt_anchors;
    uint32_t substrate_capacity;
    uint32_t rotary_dim;
    uint8_t execution_flags;
    uint8_t enable_zero_alloc_strict;
    uint8_t ledger_mode;
    uint8_t rope_layout;
    uint8_t reserved[20];
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_config lmnk_config_t;

struct LMNK_ALIGN_PREFIX(32) lmnk_entropy_state {
    float moving_z_entropy;
    float bifurcation_threshold;
    float singularity_eps;
    float sliding_entropy_rate;
    float repulsion_gain;
    float sample_temperature;
    uint32_t active_branch_mode;
    uint32_t flags;
} LMNK_ALIGN_SUFFIX(32);
typedef struct lmnk_entropy_state lmnk_entropy_state_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_subhead_config {
    uint16_t num_fact_heads;
    uint16_t num_logic_heads;
    uint16_t num_explore_heads;
    uint16_t flags;
    float fact_phase_gain;
    float logic_phase_gain;
    float explore_phase_gain;
    uint8_t reserved[12];
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_subhead_config lmnk_subhead_config_t;

/**
 * @brief Host FFI Descriptor for Rank-1 outer-product residual injection.
 * @note Host-CPU/FFI scoped (holds 64-bit host vector pointers rather than device POD scalars).
 */
struct LMNK_ALIGN_PREFIX(32) lmnk_rank1_descriptor {
    uint32_t layer_idx;
    uint32_t rank;
    uint64_t u_vector_ptr;
    uint64_t v_vector_ptr;
    uint64_t gain_scalar_ptr;
    float scale_factor;
    uint32_t flags;
    uint8_t reserved[24];
} LMNK_ALIGN_SUFFIX(32);
typedef struct lmnk_rank1_descriptor lmnk_rank1_descriptor_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_container_header {
    uint32_t magic;
    uint32_t version;
    uint32_t topology_crc32;
    uint32_t container_flags;
    uint64_t total_size;
    uint64_t timestamp_epoch_sec;
    uint32_t section_table_offset;
    uint16_t section_count;
    uint16_t header_size;
    uint8_t reserved_future[20];
    uint32_t header_crc32;
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_container_header lmnk_container_header_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_section_entry {
    uint16_t kind;
    uint16_t flags;
    uint32_t offset;
    uint32_t length;
    uint32_t payload_crc32;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_section_entry lmnk_section_entry_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_topology_matrix {
    uint32_t magic;
    uint32_t version;
    uint8_t layer_count;
    uint8_t channel_count;
    uint16_t dag_flags;
    uint8_t dag_edge_matrix[8];
    uint8_t channel_admission_mask[8];
    uint8_t pinned_channel_mask;
    uint8_t sandbox_channel_mask;
    uint8_t reserved_mid[2];
    uint8_t channel_aperture_q7[8];
    uint8_t reserved_tail[20];
    uint32_t topology_crc32;
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_topology_matrix lmnk_topology_matrix_t;

struct LMNK_ALIGN_PREFIX(32) lmnk_kv_keyframe_header {
    uint32_t frame_type;
    uint32_t token_index;
    uint32_t gop_length;
    float composite_vector_length;
    float von_mises_kappa;
    float sparse_scalar_residual;
    uint64_t icache_payload_offset;
    uint64_t pcache_payload_offset;
    uint8_t reserved[24];
} LMNK_ALIGN_SUFFIX(32);
typedef struct lmnk_kv_keyframe_header lmnk_kv_keyframe_header_t;

struct LMNK_ALIGN_PREFIX(32) lmnk_stigmergic_thresholds {
    float evaporation_retain;
    float reinforcement_gain;
    float inertial_skip_entropy_thresh;
    float express_steer_density_thresh;
    float subhead_route_density_thresh;
    float rank1_inject_density_thresh;
    float crystallize_density_thresh;
    float contradiction_coherence_thresh;
} LMNK_ALIGN_SUFFIX(32);
typedef struct lmnk_stigmergic_thresholds lmnk_stigmergic_thresholds_t;

struct LMNK_ALIGN_PREFIX(32) lmnk_substrate_step_report {
    uint64_t evolved_ptr;
    float updated_density;
    float phase_coherence;
    float transition_log_prob;
    uint32_t selected_trail_id;
    uint32_t dominant_anchor_id;
    lmnk_action_decision_t decision;
    uint8_t action_mode_flags;
    uint8_t target_slot;
    uint8_t bloom_repulsion_density;
} LMNK_ALIGN_SUFFIX(32);
typedef struct lmnk_substrate_step_report lmnk_substrate_step_report_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_gateway_frame {
    uint64_t tagged_ptr;
    uint64_t mapped_section_offset48;
    uint32_t target_anchor_id;
    float effective_energy;
    float phase_clamp_rad;
    float phase_step_min_rad;
    float effective_inv_temp;
    float fact_persona_coherence;
    float repulsion_severity;
    uint32_t lineage_flags;
    uint8_t attribution_tag;
    uint8_t dominant_slot;
    uint8_t action_mode_flags;
    uint8_t active_swarm_cell_mask;
    uint8_t phase_step_min_q7;
    uint8_t subhead_fact_heads;
    uint8_t subhead_logic_heads;
    uint8_t subhead_explore_heads;
    uint8_t subhead_fact_gain_q7;
    uint8_t subhead_logic_gain_q7;
    uint8_t subhead_explore_gain_q7;
    lmnk_q7_t mean_phase_tag_q7;
    uint16_t active_head_count;
    uint16_t frame_flags;
    lmnk_q7_t phase_deltas[64];
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_gateway_frame lmnk_gateway_frame_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_causal_report {
    uint32_t lineage_flags;
    float fact_persona_coherence;
    float anomaly_severity;
    uint8_t attribution_tag;
    uint8_t dominant_slot;
    uint8_t action_mode_flags;
    uint8_t active_swarm_mask;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_causal_report lmnk_causal_report_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_phase_anchor_record {
    uint32_t magic;
    uint32_t version;
    uint32_t anchor_id;
    uint32_t sample_count;
    uint64_t mapped_section_offset48;
    float base_energy;
    float resultant_r_bar;
    float von_mises_kappa;
    float phase_gravity_mass;
    float phase_clamp_rad;
    float effective_inv_temp;
    uint32_t lineage_flags;
    uint32_t checksum_crc32;
    uint8_t dominant_slot;
    uint8_t action_mode_flags;
    uint8_t active_swarm_mask;
    uint8_t attribution_tag;
    uint32_t reserved_align64;
    lmnk_q7_t phase_profile[64];
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_phase_anchor_record lmnk_phase_anchor_record_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_bloom_filter {
    uint8_t counters[64];
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_bloom_filter lmnk_bloom_filter_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_reconstitution_report {
    lmnk_tagged_ptr_t evolved_ptr;
    float stigmergic_delta_density;
    uint8_t quorum_mask;
    uint8_t resolved_cell_count;
    uint8_t mean_residual_q7;
    uint8_t hot_path_resolved;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_reconstitution_report lmnk_reconstitution_report_t;

struct LMNK_ALIGN_PREFIX(8) lmnk_cognitive_ledger {
    uint8_t steps[8];
} LMNK_ALIGN_SUFFIX(8);
typedef struct lmnk_cognitive_ledger lmnk_cognitive_ledger_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_epistemic_horizon {
    float total_memory_support;
    float peak_positive_affinity;
    float internal_conflict_energy;
    float horizon_gate;
    float conflict_severity;
    uint32_t dominant_anchor_id;
    uint32_t opposing_anchor_id;
    uint8_t regime_code;
    uint8_t reserved_align[3];
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_epistemic_horizon lmnk_epistemic_horizon_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_liminal_threshold_state {
    uint8_t threshold_floor_q7[8];
    uint8_t aperture_gain_q7[8];
    int8_t phase_shift_q7[8];
    uint32_t active_aperture_mask;
    uint32_t adaptation_epochs;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_liminal_threshold_state lmnk_liminal_threshold_state_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_stigmergic_granule_state {
    uint8_t granule_step_gain_q7[8];
    uint8_t granule_noise_tolerance_q7[8];
    uint8_t granule_pheromone_q7[8];
    uint8_t overlap_mask;
    uint8_t discovery_excursion_mask;
    uint8_t quorum_threshold;
    uint8_t discovery_floor_q7;
    uint32_t adaptation_epochs;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_stigmergic_granule_state lmnk_stigmergic_granule_state_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_overlap_discovery_report {
    float overlap_core_energy;
    float overlap_fringe_energy;
    float lacuna_excursion_energy;
    uint32_t dominant_overlap_anchor_id;
    uint64_t discovery_seed_word;
    uint8_t overlap_core_mask;
    uint8_t overlap_fringe_mask;
    uint8_t lacuna_exterior_mask;
    uint8_t discovery_excursion_mask;
    uint8_t max_overlap_depth;
    uint8_t discovery_granule_idx;
    uint8_t discovery_triggered;
    uint8_t reserved_align;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_overlap_discovery_report lmnk_overlap_discovery_report_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_ego_intent_axes {
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
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_ego_intent_axes lmnk_ego_intent_axes_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_liminal_ego_state {
    lmnk_q7_t sovereign_granule_phase_q7[8];
    int8_t trait_floor_bias_q7[8];
    int8_t trait_gain_bias_q7[8];
    int8_t slot_affinity_bias_q7[8];
    uint8_t granule_autonomy_q7[8];
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
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_liminal_ego_state lmnk_liminal_ego_state_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_fragment_ensemble_report {
    float ensemble_consensus_energy;
    float ensemble_diversity_gain;
    uint8_t quorum_agreed_granule_mask;
    uint8_t split_contested_granule_mask;
    uint8_t ego_tiebreak_granule_mask;
    uint8_t active_fragment_count;
    uint8_t dominant_fragment_slot;
    uint8_t reserved[3];
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_fragment_ensemble_report lmnk_fragment_ensemble_report_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_core_state_stack {
    lmnk_liminal_threshold_state_t liminal;
    lmnk_stigmergic_granule_state_t stigmergy;
    lmnk_liminal_ego_state_t ego;
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_core_state_stack lmnk_core_state_stack_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_topology_snapshot {
    uint8_t threshold_floor_q7[8];
    uint8_t aperture_gain_q7[8];
    int8_t phase_shift_q7[8];
    uint32_t active_aperture_mask;
    uint8_t regime_code;
    uint8_t conflict_severity_q7;
    uint8_t active_crystals;
    uint8_t spawned_daughters;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_topology_snapshot lmnk_topology_snapshot_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_boundary_delta {
    int8_t channel_floor_velocity_q7[8];
    uint8_t granule_transition_mask;
    uint8_t chattering_channel_mask;
    uint8_t kinetic_energy_q7;
    uint8_t stationary_shortcut_ready;
    uint8_t expansion_dominant;
    uint8_t chatter_potential_u8;
    uint8_t phase_transition_pulse;
    uint8_t reserved;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_boundary_delta lmnk_boundary_delta_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_core_primitive_state {
    lmnk_liminal_threshold_state_t liminal;
    lmnk_stigmergic_granule_state_t stigmergy;
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_core_primitive_state lmnk_core_primitive_state_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_lacuna_partition {
    uint8_t self_mask;
    uint8_t observed_mask;
    uint8_t self_ref_mask;
    uint8_t interior_mask;
    uint8_t exterior_mask;
    uint8_t unobserved_mask;
    uint8_t undefined_mask;
    uint8_t bridge_count;
    uint8_t boundary_sensitivity_q7;
    uint8_t exterior_concentration_q7;
    uint8_t exterior_density_q7;
    uint8_t partition_flags;
    uint16_t packed_lacuna_u16;
    uint8_t dominant_lacuna;
    uint8_t attribution_tag;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_lacuna_partition lmnk_lacuna_partition_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_spine_state {
    uint32_t pinned_aperture_mask;
    uint8_t pinned_slot_mask;
    uint8_t pinned_action_mode_mask;
    uint8_t unobserved_locked;
    uint8_t irreversible_step_lock;
    uint64_t pinned_compute_mask;
    uint32_t gauge_anchor_epoch;
    uint32_t slow_loop_epoch;
    uint8_t whole_token_bypass_active;
    uint8_t bypass_dwell_steps;
    uint8_t bypass_cooldown_steps;
    uint8_t attribution_tag;
    uint8_t reserved_align[4];
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_spine_state lmnk_spine_state_t;

struct LMNK_ALIGN_PREFIX(64) lmnk_distortion_delta {
    uint32_t node_id;
    uint32_t anchor_ref_id;
    uint16_t next_hop_idx;
    uint8_t hop_depth;
    uint8_t lacuna_trigger_mask;
    uint8_t scope_region_selector;
    uint8_t pheromone_q7;
    uint8_t gain_q7;
    uint8_t attribution_tag;
    lmnk_ego_intent_axes_t intent_delta;
    uint32_t reinforce_count;
    uint32_t last_active_epoch;
    uint32_t checksum_crc32;
    uint32_t flags;
    uint8_t reserved_header[16];
    lmnk_q7_t delta_phase_q7[64];
} LMNK_ALIGN_SUFFIX(64);
typedef struct lmnk_distortion_delta lmnk_distortion_delta_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_distortion_tension {
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
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_distortion_tension lmnk_distortion_tension_t;

/* -------------------------------------------------------------------------- */
/* Opaque stateful object handles                                             */
/* -------------------------------------------------------------------------- */
typedef struct LmnkContextOpaque LmnkContextOpaque;
typedef LmnkContextOpaque* lmnk_context_handle_t;
typedef LmnkContextOpaque* LmnkContextHandle;

typedef struct LmnkSubstrateOpaque LmnkSubstrateOpaque;
typedef LmnkSubstrateOpaque* lmnk_substrate_handle_t;
typedef LmnkSubstrateOpaque* LmnkSubstrateHandle;

typedef struct LmnkEntropyControllerOpaque LmnkEntropyControllerOpaque;
typedef LmnkEntropyControllerOpaque* lmnk_entropy_controller_handle_t;
typedef LmnkEntropyControllerOpaque* LmnkEntropyControllerHandle;

typedef struct LmnkPhaseSteeringOpaque LmnkPhaseSteeringOpaque;
typedef LmnkPhaseSteeringOpaque* lmnk_phase_steering_handle_t;
typedef LmnkPhaseSteeringOpaque* LmnkPhaseSteeringHandle;

/* -------------------------------------------------------------------------- */
/* Compile-time size, alignment, and field-offset invariants                  */
/* -------------------------------------------------------------------------- */
LMNK_STATIC_ASSERT(sizeof(lmnk_descriptor_header_t) == 8, "lmnk_descriptor_header_t size mismatch");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_descriptor_header_t) == 4,
                   "lmnk_descriptor_header_t align mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_paged_bitmask_t) == 32, "lmnk_paged_bitmask_t size mismatch");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_paged_bitmask_t) == 32, "lmnk_paged_bitmask_t align mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_pointer_trail_t) == 128, "lmnk_pointer_trail_t must be 128 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_pointer_trail_t) == 64,
                   "lmnk_pointer_trail_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_pointer_trail_t, reserved_header) == 56,
                   "lmnk_pointer_trail_t reserved_header offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_pointer_trail_t, accumulated_phase_delta) == 64,
                   "lmnk_pointer_trail_t phase array offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_ledger_slot_t) == 64, "lmnk_ledger_slot_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_ledger_slot_t) == 64,
                   "lmnk_ledger_slot_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_ledger_slot_t, fast_path_shortcuts) == 24,
                   "lmnk_ledger_slot_t shortcuts offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_config_t) == 64, "lmnk_config_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_config_t) == 64, "lmnk_config_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_config_t, execution_flags) == 40,
                   "lmnk_config_t execution_flags offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_config_t, reserved) == 44,
                   "lmnk_config_t reserved offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_entropy_state_t) == 32, "lmnk_entropy_state_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_entropy_state_t) == 32,
                   "lmnk_entropy_state_t must be 32-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_entropy_state_t, active_branch_mode) == 24,
                   "lmnk_entropy_state_t branch mode offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_entropy_state_t, flags) == 28,
                   "lmnk_entropy_state_t flags offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_subhead_config_t) == 32, "lmnk_subhead_config_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_subhead_config_t) == 16,
                   "lmnk_subhead_config_t must be 16-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_rank1_descriptor_t) == 64,
                   "lmnk_rank1_descriptor_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_rank1_descriptor_t) == 32,
                   "lmnk_rank1_descriptor_t must be 32-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_rank1_descriptor_t, u_vector_ptr) == 8,
                   "lmnk_rank1_descriptor_t u_vector_ptr offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_container_header_t) == 64,
                   "lmnk_container_header_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_container_header_t) == 64,
                   "lmnk_container_header_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_container_header_t, topology_crc32) == 8,
                   "lmnk_container_header_t topology_crc32 offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_container_header_t, header_crc32) == 60,
                   "lmnk_container_header_t header_crc32 offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_section_entry_t) == 16, "lmnk_section_entry_t must be 16 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_section_entry_t) == 16,
                   "lmnk_section_entry_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_section_entry_t, payload_crc32) == 12,
                   "lmnk_section_entry_t payload_crc32 offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_topology_matrix_t) == 64, "lmnk_topology_matrix_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_topology_matrix_t) == 64,
                   "lmnk_topology_matrix_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_topology_matrix_t, dag_edge_matrix) == 12,
                   "lmnk_topology_matrix_t dag_edge_matrix offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_topology_matrix_t, pinned_channel_mask) == 28,
                   "lmnk_topology_matrix_t pinned_channel_mask offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_topology_matrix_t, topology_crc32) == 60,
                   "lmnk_topology_matrix_t topology_crc32 offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_kv_keyframe_header_t) == 64,
                   "lmnk_kv_keyframe_header_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_kv_keyframe_header_t) == 32,
                   "lmnk_kv_keyframe_header_t must be 32-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_stigmergic_thresholds_t) == 32,
                   "lmnk_stigmergic_thresholds_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_stigmergic_thresholds_t) == 32,
                   "lmnk_stigmergic_thresholds_t must be 32-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_substrate_step_report_t) == 32,
                   "lmnk_substrate_step_report_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_substrate_step_report_t) == 32,
                   "lmnk_substrate_step_report_t must be 32-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_substrate_step_report_t, dominant_anchor_id) == 24,
                   "lmnk_substrate_step_report_t anchor offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_substrate_step_report_t, decision) == 28,
                   "lmnk_substrate_step_report_t decision offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_gateway_frame_t) == 128, "lmnk_gateway_frame_t must be 128 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_gateway_frame_t) == 64,
                   "lmnk_gateway_frame_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_gateway_frame_t, phase_deltas) == 64,
                   "lmnk_gateway_frame_t phase_deltas offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_causal_report_t) == 16, "lmnk_causal_report_t must be 16 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_causal_report_t) == 16,
                   "lmnk_causal_report_t must be 16-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_phase_anchor_record_t) == 128,
                   "lmnk_phase_anchor_record_t must be 128 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_phase_anchor_record_t) == 64,
                   "lmnk_phase_anchor_record_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_phase_anchor_record_t, lineage_flags) == 48,
                   "lmnk_phase_anchor_record_t lineage_flags offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_phase_anchor_record_t, checksum_crc32) == 52,
                   "lmnk_phase_anchor_record_t checksum_crc32 offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_phase_anchor_record_t, dominant_slot) == 56,
                   "lmnk_phase_anchor_record_t dominant_slot offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_phase_anchor_record_t, phase_profile) == 64,
                   "lmnk_phase_anchor_record_t phase_profile offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_bloom_filter_t) == 64, "lmnk_bloom_filter_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_bloom_filter_t) == 64,
                   "lmnk_bloom_filter_t must be 64-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_reconstitution_report_t) == 16,
                   "lmnk_reconstitution_report_t must be 16 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_reconstitution_report_t) == 16,
                   "lmnk_reconstitution_report_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_reconstitution_report_t, evolved_ptr) == 0,
                   "lmnk_reconstitution_report_t evolved_ptr offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_cognitive_ledger_t) == 8, "lmnk_cognitive_ledger_t must be 8 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_cognitive_ledger_t) == 8,
                   "lmnk_cognitive_ledger_t must be 8-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_epistemic_horizon_t) == 32,
                   "lmnk_epistemic_horizon_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_epistemic_horizon_t) == 16,
                   "lmnk_epistemic_horizon_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_epistemic_horizon_t, dominant_anchor_id) == 20,
                   "lmnk_epistemic_horizon_t anchor offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_epistemic_horizon_t, regime_code) == 28,
                   "lmnk_epistemic_horizon_t regime offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_liminal_threshold_state_t) == 32,
                   "lmnk_liminal_threshold_state_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_liminal_threshold_state_t) == 16,
                   "lmnk_liminal_threshold_state_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_liminal_threshold_state_t, active_aperture_mask) == 24,
                   "lmnk_liminal_threshold_state_t aperture offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_stigmergic_granule_state_t) == 32,
                   "lmnk_stigmergic_granule_state_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_stigmergic_granule_state_t) == 16,
                   "lmnk_stigmergic_granule_state_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_stigmergic_granule_state_t, overlap_mask) == 24,
                   "lmnk_stigmergic_granule_state_t overlap_mask offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_stigmergic_granule_state_t, adaptation_epochs) == 28,
                   "lmnk_stigmergic_granule_state_t epochs offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_overlap_discovery_report_t) == 32,
                   "lmnk_overlap_discovery_report_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_overlap_discovery_report_t) == 16,
                   "lmnk_overlap_discovery_report_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_overlap_discovery_report_t, overlap_core_mask) == 24,
                   "lmnk_overlap_discovery_report_t core_mask offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_overlap_discovery_report_t, lacuna_exterior_mask) == 26,
                   "lmnk_overlap_discovery_report_t exterior_mask offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_ego_intent_axes_t) == 16, "lmnk_ego_intent_axes_t must be 16 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_ego_intent_axes_t) == 16,
                   "lmnk_ego_intent_axes_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_ego_intent_axes_t, signature_hash) == 12,
                   "lmnk_ego_intent_axes_t signature_hash offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_liminal_ego_state_t) == 64,
                   "lmnk_liminal_ego_state_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_liminal_ego_state_t) == 64,
                   "lmnk_liminal_ego_state_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_liminal_ego_state_t, reserved) == 57,
                   "lmnk_liminal_ego_state_t reserved offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_fragment_ensemble_report_t) == 16,
                   "lmnk_fragment_ensemble_report_t must be 16 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_fragment_ensemble_report_t) == 16,
                   "lmnk_fragment_ensemble_report_t must be 16-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_core_state_stack_t) == 128,
                   "lmnk_core_state_stack_t must be 128 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_core_state_stack_t) == 64,
                   "lmnk_core_state_stack_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_core_state_stack_t, liminal) == 0,
                   "lmnk_core_state_stack_t liminal offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_core_state_stack_t, stigmergy) == 32,
                   "lmnk_core_state_stack_t stigmergy offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_core_state_stack_t, ego) == 64,
                   "lmnk_core_state_stack_t ego offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_topology_snapshot_t) == 32,
                   "lmnk_topology_snapshot_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_topology_snapshot_t) == 16,
                   "lmnk_topology_snapshot_t must be 16-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_boundary_delta_t) == 16, "lmnk_boundary_delta_t must be 16 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_boundary_delta_t) == 16,
                   "lmnk_boundary_delta_t must be 16-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_core_primitive_state_t) == 64,
                   "lmnk_core_primitive_state_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_core_primitive_state_t) == 64,
                   "lmnk_core_primitive_state_t must be 64-byte aligned");

LMNK_STATIC_ASSERT(sizeof(lmnk_lacuna_partition_t) == 16,
                   "lmnk_lacuna_partition_t must be 16 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_lacuna_partition_t) == 16,
                   "lmnk_lacuna_partition_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_lacuna_partition_t, packed_lacuna_u16) == 12,
                   "lmnk_lacuna_partition_t packed offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_spine_state_t) == 32, "lmnk_spine_state_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_spine_state_t) == 16,
                   "lmnk_spine_state_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_spine_state_t, pinned_compute_mask) == 8,
                   "lmnk_spine_state_t pinned_compute_mask offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_distortion_delta_t) == 128,
                   "lmnk_distortion_delta_t must be 128 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_distortion_delta_t) == 64,
                   "lmnk_distortion_delta_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_distortion_delta_t, intent_delta) == 16,
                   "lmnk_distortion_delta_t intent_delta offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_distortion_delta_t, delta_phase_q7) == 64,
                   "lmnk_distortion_delta_t delta_phase_q7 offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_distortion_tension_t) == 32,
                   "lmnk_distortion_tension_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_distortion_tension_t) == 16,
                   "lmnk_distortion_tension_t must be 16-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_distortion_tension_t, immediate_damping_events) == 8,
                   "lmnk_distortion_tension_t damping offset mismatch");

LMNK_EXTERN_C_END

#endif /* LIMINIKA_LMNK_TYPES_H */
