// name: include/liminika/lmnk_types.h
// SPDX-License-Identifier: MIT OR Apache-2.0

/**
 * @file lmnk_types.h
 * @brief LiminiKa v0.4.0 (`0x00000400`) canonical ABI types, versioned envelopes, POD layouts,
 *        status codes, and `static inline` tagged-pointer helpers.
 *
 * Portable Wire & Device Contract:
 * - Canonical byte order across the FFI and `.lmnk` container surface is Little-Endian.
 * - Portable Device-Ready PODs (`lmnk_arena_span_t`, `lmnk_rank1_descriptor_t`,
 *   `lmnk_pointer_trail_t`, `lmnk_gateway_frame_t`, `lmnk_phase_anchor_record_t`,
 *   `lmnk_lacuna_partition_t`, `lmnk_container_header_t`, etc.) use fixed-width primitives only,
 *   carry zero raw pointers, have zero implicit compiler padding, and do NOT carry `struct_size`.
 * - Versioned Call Envelopes (`lmnk_config_t`, `lmnk_abi_info_t`, `lmnk_runtime_policy_t`,
 *   `lmnk_turn_begin_params_t`, `lmnk_decode_step_params_t`, `lmnk_decode_step_report_t`,
 *   `lmnk_runtime_state_view_t`, `lmnk_lacuna_eval_params_t`, `lmnk_container_verify_report_t`,
 *   and transient pointer-carrying `*_params_t` / `*_view_t` / `lmnk_section_input_t`) carry a
 *   leading `uint32_t struct_size` field validated fail-closed across the C-ABI boundary.
 * - Transient pointer-carrying host envelopes are strictly in-memory call descriptors and are
 *   NEVER persisted or hashed. All tensor buffer pointers are strongly typed `float*` (`0 void*`).
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
#define LMNK_ABI_VERSION_MINOR 4U
#define LMNK_ABI_VERSION_PATCH 0U
#define LMNK_ABI_VERSION_HEX 0x00000400U
#define LMNK_ABI_COMPAT_MASK 0xFFFFFF00U
#define LMNK_ABI_COMPAT_VERSION_HEX (LMNK_ABI_VERSION_HEX & LMNK_ABI_COMPAT_MASK)
#define LMNK_ABI_VERSION_STRING "0.4.0"

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
#define LMNK_PHASE_ANCHOR_VERSION 0x00000400U
#define LMNK_CONTAINER_HEADER_BYTES 64U
#define LMNK_SECTION_ENTRY_BYTES 16U
#define LMNK_MAX_CONTAINER_SECTIONS 32U
#define LMNK_MAX_CONTAINER_BYTES (64U * 1024U * 1024U)
#define LMNK_RANK1_GAIN_UNUSED 0xFFFFFFFFU

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
#define LMNK_DAG_FLAG_THREE_DRIVERS_ISOLATED (1U << 4)
#define LMNK_TOPOLOGY_DAG_FLAGS_CANONICAL                                        \
    (LMNK_DAG_FLAG_Y_SHAPED_PARALLEL_ROOTS | LMNK_DAG_FLAG_SPINE_EGO_ISOLATED | \
     LMNK_DAG_FLAG_AXIOM_CUT_LACUNA_ORTHOGONAL | LMNK_DAG_FLAG_NO_STD_CORE |     \
     LMNK_DAG_FLAG_THREE_DRIVERS_ISOLATED)
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
/* Envelope option flags (v0.4.0)                                             */
/* -------------------------------------------------------------------------- */
#define LMNK_POLICY_FLAG_SPINE (1U << 0)
#define LMNK_POLICY_FLAG_COMPUTE_POOL (1U << 1)
#define LMNK_POLICY_KNOWN_FLAGS (LMNK_POLICY_FLAG_SPINE | LMNK_POLICY_FLAG_COMPUTE_POOL)

#define LMNK_TURN_FLAG_GUARD_ACTIVE (1U << 0)
#define LMNK_TURN_FLAG_UNOBSERVED_LOCKED (1U << 1)
#define LMNK_TURN_KNOWN_FLAGS (LMNK_TURN_FLAG_GUARD_ACTIVE | LMNK_TURN_FLAG_UNOBSERVED_LOCKED)

#define LMNK_ROPE_STEER_FLAG_UNCLAMPED (1U << 0)
#define LMNK_ROPE_STEER_KNOWN_FLAGS LMNK_ROPE_STEER_FLAG_UNCLAMPED

#define LMNK_RANK1_FLAG_FUSED (1U << 0)
#define LMNK_RANK1_KNOWN_FLAGS LMNK_RANK1_FLAG_FUSED

/* -------------------------------------------------------------------------- */
/* Primitive/Driver attribution, Lacuna states, Scope & Causal constants      */
/* -------------------------------------------------------------------------- */
#define LMNK_PRIMITIVE_LIMINAL 0U
#define LMNK_PRIMITIVE_STIGMERGY 1U
#define LMNK_DRIVER_SYSTEM 0U
#define LMNK_DRIVER_EGO 1U
#define LMNK_DRIVER_ENVELOPE 2U
#define LMNK_ATTRIBUTION_TAG(primitive, driver) \
    ((uint8_t)((((driver) & 0x03U) << 1) | ((primitive) & 0x01U)))

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

#define LMNK_TRAIL_FLAG_PHASE_STEER_ACTIVE (1U << 0)
#define LMNK_TRAIL_FLAG_REPULSION_ACTIVE (1U << 1)
#define LMNK_TRAIL_FLAG_CRYSTALLIZED_CANDIDATE (1U << 2)
#define LMNK_TRAIL_FLAG_SYNERGY_RESONANCE (1U << 3)
#define LMNK_TRAIL_FLAG_LIMINAL_BEAT_ACTIVE (1U << 4)
#define LMNK_TRAIL_FLAG_INERTIAL_SKIP_ACTIVE (1U << 5)
#define LMNK_TRAIL_FLAG_SWARM_RESOLVED (1U << 6)
#define LMNK_TRAIL_FLAG_SWARM_ESCALATED (1U << 7)
#define LMNK_TRAIL_FLAG_PHEROMONE_RISING (1U << 16)

#define LMNK_ANCHOR_ROOT_CAUSE_NOMINAL 0U
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

#define LMNK_HOST_IMMUTABLE_PINNED_COMPUTE_MASK (1ULL << 0)

#define LMNK_SPINE_BYPASS_VETO_NONE 0x00U
#define LMNK_SPINE_BYPASS_VETO_GUARD_ACTIVE (1U << 0)
#define LMNK_SPINE_BYPASS_VETO_REPULSION_ACTIVE (1U << 1)
#define LMNK_SPINE_BYPASS_VETO_PINNED_LOCK (1U << 2)
#define LMNK_SPINE_BYPASS_VETO_KNOWN_MASK 0x07U

#define LMNK_SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE (1U << 0)
#define LMNK_SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED (1U << 1)
#define LMNK_SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID (1U << 2)
#define LMNK_SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT (1U << 3)
#define LMNK_SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED (1U << 4)
#define LMNK_SPINE_CONTROL_KNOWN_FLAGS 0x1FU

#define LMNK_SPINE_STEP_EVENT_NONE 0x00U
#define LMNK_SPINE_STEP_EVENT_BOUNDARY_CHURN (1U << 0)
#define LMNK_SPINE_STEP_EVENT_LOGITS_OBSERVE_SKIPPED (1U << 1)
#define LMNK_SPINE_STEP_EVENT_TRANSIENT_GUARD_RELAXED (1U << 2)
#define LMNK_SPINE_STEP_EVENT_GUARD_REESCALATED (1U << 3)
#define LMNK_SPINE_STEP_EVENT_INSTABILITY_RECOVERY (1U << 4)

#define LMNK_SPINE_DECODE_INPUT_FLAG_REPULSION_ACTIVE (1U << 0)
#define LMNK_SPINE_DECODE_INPUT_FLAG_GREEDY_SAMPLING (1U << 1)
#define LMNK_SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED (1U << 2)
#define LMNK_SPINE_DECODE_INPUT_FLAG_REWARD_OVERRIDE (1U << 3)
#define LMNK_SPINE_DECODE_FLAG_REWARD_OVERRIDE LMNK_SPINE_DECODE_INPUT_FLAG_REWARD_OVERRIDE

#define LMNK_SPINE_LOGITS_OBSERVE_FULL 0U
#define LMNK_SPINE_LOGITS_SKIP_TO_DEVICE_ARGMAX 1U
#define LMNK_SPINE_SENTINEL_AUTO_U8 0xFFU
#define LMNK_SPINE_MIN_DYNAMIC_OBSERVE_STRIDE 2U
#define LMNK_SPINE_ENTROPY_Q15_ONE 32768U
#define LMNK_SPINE_ENTROPY_CALM_THRESH_Q15 25559U
#define LMNK_SPINE_ENTROPY_GUARD_RELAX_MAX_Q15 23593U
#define LMNK_SPINE_ENTROPY_SURGE_THRESH_Q15 27853U
#define LMNK_SPINE_ENTROPY_REWARD_LOW_THRESH_Q15 18022U
#define LMNK_SPINE_ENTROPY_TENSION_FLOOR_Q15 4915U

#define LMNK_DISTORTION_DELTA_FLAG_ACTIVE (1U << 0)
#define LMNK_DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME (1U << 1)
#define LMNK_DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID (1U << 2)

#define LMNK_DISTORTION_TENSION_BALANCED_DEADBAND 0U
#define LMNK_DISTORTION_TENSION_PASSIVE_EXCITATION 1U
#define LMNK_DISTORTION_TENSION_ACTIVE_IMMEDIATE_DAMPING 2U

/* -------------------------------------------------------------------------- */
/* Header-only tagged pointer bit-packers (zero FFI call overhead)            */
/* -------------------------------------------------------------------------- */
#if defined(__GNUC__) || defined(__clang__)
    #define LMNK_STATIC_INLINE static inline __attribute__((unused))
#else
    #define LMNK_STATIC_INLINE static inline
#endif

LMNK_STATIC_INLINE uint64_t lmnk_tagged_ptr_pack(uint64_t raw_addr, uint8_t tag, lmnk_q7_t phase) {
    const uint64_t addr_bits = raw_addr & LMNK_TAGGED_PTR_ADDR_MASK;
    const uint64_t tag_bits = ((uint64_t)tag) << LMNK_TAGGED_PTR_TAG_SHIFT;
    const uint64_t phase_bits = ((uint64_t)((uint8_t)phase)) << LMNK_TAGGED_PTR_PHASE_SHIFT;
    return addr_bits | tag_bits | phase_bits;
}

LMNK_STATIC_INLINE uint64_t lmnk_tagged_ptr_unpack_addr(uint64_t ptr) {
    return ptr & LMNK_TAGGED_PTR_ADDR_MASK;
}

LMNK_STATIC_INLINE uint8_t lmnk_tagged_ptr_unpack_tag(uint64_t ptr) {
    return (uint8_t)((ptr >> LMNK_TAGGED_PTR_TAG_SHIFT) & 0xFFULL);
}

LMNK_STATIC_INLINE lmnk_q7_t lmnk_tagged_ptr_unpack_phase(uint64_t ptr) {
    return (lmnk_q7_t)((uint8_t)((ptr >> LMNK_TAGGED_PTR_PHASE_SHIFT) & 0xFFULL));
}

LMNK_STATIC_INLINE uint64_t lmnk_tagged_ptr_pack_action(uint64_t raw_addr, uint8_t attractor_slot,
                                                        uint8_t action_mode_flags,
                                                        lmnk_q7_t phase) {
    const uint8_t tag =
        (uint8_t)((((uint32_t)(attractor_slot & LMNK_ATTRACTOR_SLOT_ID_MASK)) << 5U) |
                  ((uint32_t)(action_mode_flags & 0x1FU)));
    return lmnk_tagged_ptr_pack(raw_addr, tag, phase);
}

LMNK_STATIC_INLINE uint8_t lmnk_tagged_ptr_unpack_action_mode(uint64_t ptr) {
    return (uint8_t)(lmnk_tagged_ptr_unpack_tag(ptr) & 0x1FU);
}

LMNK_STATIC_INLINE uint8_t lmnk_tagged_ptr_unpack_attractor_slot(uint64_t ptr) {
    return (uint8_t)((lmnk_tagged_ptr_unpack_tag(ptr) >> 5U) & LMNK_ATTRACTOR_SLOT_ID_MASK);
}

/* -------------------------------------------------------------------------- */
/* Portable Device-Ready POD structures (1:1 Rust repr(C, align) parity)      */
/* -------------------------------------------------------------------------- */

struct LMNK_ALIGN_PREFIX(4) lmnk_arena_span {
    uint32_t offset;
    uint32_t len;
} LMNK_ALIGN_SUFFIX(4);
typedef struct lmnk_arena_span lmnk_arena_span_t;

struct LMNK_ALIGN_PREFIX(32) lmnk_rank1_descriptor {
    uint32_t layer_idx;
    uint32_t rank;
    lmnk_arena_span_t u_span;
    lmnk_arena_span_t v_span;
    uint32_t gain_offset;
    float scale_factor;
} LMNK_ALIGN_SUFFIX(32);
typedef struct lmnk_rank1_descriptor lmnk_rank1_descriptor_t;

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
    uint32_t struct_size;
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
    uint8_t guard_hold_tokens;
    uint8_t max_observe_stride;
    uint8_t reserved[14];
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
    uint8_t root_cause_code;
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
    uint8_t root_cause_code;
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
    uint8_t root_cause_code;
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
    int8_t boundary_crossing_q7;
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
    lmnk_q7_t preferred_granule_phase_q7[8];
    int8_t trait_floor_bias_q7[8];
    int8_t trait_gain_bias_q7[8];
    int8_t slot_affinity_bias_q7[8];
    uint8_t granule_autonomy_q7[8];
    uint8_t basin_resistance_q7;
    uint8_t escape_floor_q7;
    int8_t boundary_crossing_q7;
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
    uint8_t active_anchors;
    uint8_t spawned_anchors;
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

struct LMNK_ALIGN_PREFIX(2) lmnk_system_pressure_reading {
    uint16_t frame_time_pressure_q15;
    uint16_t bandwidth_pressure_q15;
    uint16_t l3_load_pressure_q15;
    uint8_t irreversible_step;
    uint8_t reserved_pad;
} LMNK_ALIGN_SUFFIX(2);
typedef struct lmnk_system_pressure_reading lmnk_system_pressure_reading_t;

struct LMNK_ALIGN_PREFIX(8) lmnk_compute_dispatch_mask {
    uint64_t execute_mask;
    uint64_t bypass_mask;
    uint64_t throttle_mask;
    uint64_t stationary_evidence_mask;
    uint8_t primary_lane;
    uint8_t fallback_lane;
    uint8_t regime_u8;
    uint8_t slow_loop_migration_ready;
    uint16_t relative_pressure_q15;
    uint8_t active_pheromone_floor_q7;
    uint8_t whole_token_steering_bypass;
} LMNK_ALIGN_SUFFIX(8);
typedef struct lmnk_compute_dispatch_mask lmnk_compute_dispatch_mask_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_spine_decode_input {
    lmnk_system_pressure_reading_t pressure;
    uint16_t observed_exit_entropy_q15;
    uint16_t decode_step_index;
    uint8_t exit_entropy_lacuna_code;
    uint8_t boundary_sensitivity_q7;
    uint8_t min_calm_dwell;
    uint8_t flags;
    uint8_t exit_reward_override_q7;
    uint8_t reserved[15];
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_spine_decode_input lmnk_spine_decode_input_t;

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
/* Versioned v0.4.0 Fixed-Width Call Envelopes & Views (leading struct_size)  */
/* -------------------------------------------------------------------------- */

typedef struct lmnk_abi_info {
    uint32_t struct_size;
    uint32_t abi_version_hex;
    uint32_t context_size;
    uint32_t context_align;
    uint32_t snapshot_max_bytes;
    uint32_t topology_crc32;
    uint64_t capability_flags;
    uint8_t reserved[32];
} lmnk_abi_info_t;

typedef struct lmnk_runtime_policy {
    uint32_t struct_size;
    uint32_t flags;
    uint64_t baseline_pinned_compute_mask;
    uint64_t active_compute_mask;
    uint64_t repulsion_compute_mask;
    uint64_t reversible_compute_mask;
    uint16_t baseline_pressure_q15;
    uint8_t guard_hold_tokens;
    uint8_t max_observe_stride;
    uint8_t pre_bypass_enabled;
    uint8_t reserved[3];
} lmnk_runtime_policy_t;

typedef struct lmnk_turn_begin_params {
    uint32_t struct_size;
    uint32_t flags;
    uint8_t reserved[8];
} lmnk_turn_begin_params_t;

struct LMNK_ALIGN_PREFIX(16) lmnk_decode_step_params {
    uint32_t struct_size;
    uint8_t external_guard_active;
    uint8_t reserved[11];
    lmnk_spine_decode_input_t decode_input;
} LMNK_ALIGN_SUFFIX(16);
typedef struct lmnk_decode_step_params lmnk_decode_step_params_t;

typedef struct lmnk_decode_step_report {
    uint32_t struct_size;
    uint8_t next_observe_mode;
    uint8_t veto_mask;
    uint8_t event_flags;
    uint8_t effective_observe_stride;
    lmnk_compute_dispatch_mask_t dispatch;
    uint8_t reserved[32];
} lmnk_decode_step_report_t;

typedef struct lmnk_runtime_state_view {
    uint32_t struct_size;
    uint8_t control_flags;
    uint8_t last_veto_mask;
    uint8_t guard_hold_tokens;
    uint8_t max_observe_stride;
    uint64_t baseline_pinned_compute_mask;
    uint64_t pinned_compute_mask;
    uint64_t last_executed_mask;
    uint64_t active_compute_mask;
    uint16_t last_observed_entropy_q15;
    uint16_t baseline_pressure_q15;
    uint8_t whole_token_bypass_active;
    uint8_t unobserved_locked;
    uint8_t reserved[18];
} lmnk_runtime_state_view_t;

typedef struct lmnk_lacuna_eval_params {
    uint32_t struct_size;
    uint8_t self_ref_mask;
    uint8_t deadband_q7;
    uint8_t reserved[2];
    uint8_t other_density_q7[8];
    uint8_t noise_tolerance_q7[8];
} lmnk_lacuna_eval_params_t;

typedef struct lmnk_container_verify_report {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t topology_crc32;
    uint32_t container_flags;
    uint64_t total_size;
    uint64_t timestamp_epoch_sec;
    uint16_t section_count;
    uint8_t reserved[6];
} lmnk_container_verify_report_t;

/* -------------------------------------------------------------------------- */
/* Transient Host Call Envelopes (Pointer-Carrying, Never Persisted)          */
/* -------------------------------------------------------------------------- */

typedef struct lmnk_rope_steer_params {
    uint32_t struct_size;
    uint32_t flags;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t num_heads;
    uint8_t rope_layout;
    uint8_t reserved0[3];
    float max_rad;
    uint32_t num_tokens;
    uint32_t token_stride_elems;
    uint32_t reserved1;
    float* query_tensor;
    const lmnk_q7_t* phase_deltas;
} lmnk_rope_steer_params_t;

typedef struct lmnk_rope_procrustes_params {
    uint32_t struct_size;
    uint32_t flags;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t num_heads;
    uint8_t rope_layout;
    uint8_t reserved0[3];
    uint32_t num_tokens;
    uint32_t token_stride_elems;
    const float* q_before;
    const float* q_after;
    lmnk_q7_t* q7_phase_out;
} lmnk_rope_procrustes_params_t;

typedef struct lmnk_rank1_eval_params {
    uint32_t struct_size;
    uint32_t flags;
    uint32_t dim_in;
    uint32_t dim_out;
    uint32_t arena_len;
    uint32_t num_tokens;
    uint32_t token_stride_elems;
    uint32_t reserved0;
    const lmnk_rank1_descriptor_t* desc;
    const float* arena;
    const float* x_in;
    float* y_inout;
} lmnk_rank1_eval_params_t;

typedef struct lmnk_gateway_steer_params {
    uint32_t struct_size;
    uint32_t flags;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t num_heads;
    uint8_t rope_layout;
    uint8_t unresolved_damping_u7;
    uint8_t reserved0[2];
    float fact_energy;
    float persona_energy;
    uint32_t num_tokens;
    uint32_t token_stride_elems;
    float* query_tensor;
    const lmnk_q7_t* fact_phases;
    const lmnk_q7_t* persona_phases;
    lmnk_pointer_trail_t* trail_inout;
    lmnk_gateway_frame_t* frame_out;
    lmnk_causal_report_t* report_out;
} lmnk_gateway_steer_params_t;

typedef struct lmnk_section_input {
    uint32_t struct_size;
    uint16_t kind;
    uint16_t flags;
    uint32_t payload_len;
    uint32_t reserved0;
    const uint8_t* payload;
} lmnk_section_input_t;

typedef struct lmnk_container_write_params {
    uint32_t struct_size;
    uint32_t container_flags;
    uint64_t timestamp_epoch_sec;
    uint32_t section_count;
    uint32_t reserved0;
    const lmnk_section_input_t* sections;
    uint8_t* out_bytes;
    size_t out_capacity;
    size_t* written_bytes_out;
} lmnk_container_write_params_t;

typedef struct lmnk_container_section_view {
    uint32_t struct_size;
    uint16_t kind;
    uint16_t flags;
    uint32_t offset;
    uint32_t length;
    uint32_t payload_crc32;
    uint32_t reserved0;
    const uint8_t* payload_ptr;
} lmnk_container_section_view_t;

/* -------------------------------------------------------------------------- */
/* Opaque stateful runtime context handle                                     */
/* -------------------------------------------------------------------------- */
typedef struct LmnkContextOpaque LmnkContextOpaque;
typedef LmnkContextOpaque* lmnk_context_handle_t;
typedef LmnkContextOpaque* LmnkContextHandle;

/* -------------------------------------------------------------------------- */
/* Compile-time size, alignment, and field-offset invariants                  */
/* -------------------------------------------------------------------------- */
LMNK_STATIC_ASSERT(sizeof(lmnk_arena_span_t) == 8, "lmnk_arena_span_t must be 8 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_arena_span_t) == 4, "lmnk_arena_span_t align mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_rank1_descriptor_t) == 32,
                   "lmnk_rank1_descriptor_t must be 32 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_rank1_descriptor_t) == 32,
                   "lmnk_rank1_descriptor_t must be 32-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_rank1_descriptor_t, u_span) == 8,
                   "lmnk_rank1_descriptor_t u_span offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_rank1_descriptor_t, v_span) == 16,
                   "lmnk_rank1_descriptor_t v_span offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_rank1_descriptor_t, gain_offset) == 24,
                   "lmnk_rank1_descriptor_t gain_offset offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_rank1_descriptor_t, scale_factor) == 28,
                   "lmnk_rank1_descriptor_t scale_factor offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_config_t) == 64, "lmnk_config_t must be 64 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_config_t) == 64, "lmnk_config_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(offsetof(lmnk_config_t, struct_size) == 0,
                   "lmnk_config_t struct_size offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_config_t, head_dim) == 4,
                   "lmnk_config_t head_dim offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_config_t, execution_flags) == 44,
                   "lmnk_config_t execution_flags offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_config_t, guard_hold_tokens) == 48,
                   "lmnk_config_t guard_hold_tokens offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_config_t, max_observe_stride) == 49,
                   "lmnk_config_t max_observe_stride offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_config_t, reserved) == 50,
                   "lmnk_config_t reserved offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_abi_info_t) == 64, "lmnk_abi_info_t must be 64 bytes");
LMNK_STATIC_ASSERT(offsetof(lmnk_abi_info_t, capability_flags) == 24,
                   "lmnk_abi_info_t capability_flags offset mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_runtime_policy_t) == 48, "lmnk_runtime_policy_t must be 48 bytes");
LMNK_STATIC_ASSERT(offsetof(lmnk_runtime_policy_t, baseline_pressure_q15) == 40,
                   "lmnk_runtime_policy_t baseline_pressure_q15 offset mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_turn_begin_params_t) == 16,
                   "lmnk_turn_begin_params_t must be 16 bytes");
LMNK_STATIC_ASSERT(sizeof(lmnk_decode_step_params_t) == 48,
                   "lmnk_decode_step_params_t must be 48 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_decode_step_params_t) == 16,
                   "lmnk_decode_step_params_t align mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_decode_step_params_t, decode_input) == 16,
                   "lmnk_decode_step_params_t decode_input offset mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_decode_step_report_t) == 80,
                   "lmnk_decode_step_report_t must be 80 bytes");
LMNK_STATIC_ASSERT(offsetof(lmnk_decode_step_report_t, dispatch) == 8,
                   "lmnk_decode_step_report_t dispatch offset mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_runtime_state_view_t) == 64,
                   "lmnk_runtime_state_view_t must be 64 bytes");
LMNK_STATIC_ASSERT(offsetof(lmnk_runtime_state_view_t, active_compute_mask) == 32,
                   "lmnk_runtime_state_view_t active_compute_mask offset mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_lacuna_eval_params_t) == 24,
                   "lmnk_lacuna_eval_params_t must be 24 bytes");
LMNK_STATIC_ASSERT(sizeof(lmnk_container_verify_report_t) == 40,
                   "lmnk_container_verify_report_t must be 40 bytes");

LMNK_STATIC_ASSERT(offsetof(lmnk_rope_steer_params_t, query_tensor) == 40,
                   "lmnk_rope_steer_params_t query_tensor offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_rope_procrustes_params_t, q_before) == 32,
                   "lmnk_rope_procrustes_params_t q_before offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_rank1_eval_params_t, desc) == 32,
                   "lmnk_rank1_eval_params_t desc offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_gateway_steer_params_t, query_tensor) == 40,
                   "lmnk_gateway_steer_params_t query_tensor offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_section_input_t, payload) == 16,
                   "lmnk_section_input_t payload offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_container_write_params_t, sections) == 24,
                   "lmnk_container_write_params_t sections offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_container_section_view_t, payload_ptr) == 24,
                   "lmnk_container_section_view_t payload_ptr offset mismatch");

LMNK_STATIC_ASSERT(sizeof(lmnk_pointer_trail_t) == 128, "lmnk_pointer_trail_t must be 128 bytes");
LMNK_STATIC_ASSERT(LMNK_ALIGNOF(lmnk_pointer_trail_t) == 64,
                   "lmnk_pointer_trail_t must be 64-byte aligned");
LMNK_STATIC_ASSERT(sizeof(lmnk_gateway_frame_t) == 128, "lmnk_gateway_frame_t must be 128 bytes");
LMNK_STATIC_ASSERT(offsetof(lmnk_gateway_frame_t, root_cause_code) == 48,
                   "lmnk_gateway_frame_t root_cause_code offset mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_causal_report_t) == 16, "lmnk_causal_report_t must be 16 bytes");
LMNK_STATIC_ASSERT(offsetof(lmnk_causal_report_t, root_cause_code) == 12,
                   "lmnk_causal_report_t root_cause_code offset mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_phase_anchor_record_t) == 128,
                   "lmnk_phase_anchor_record_t must be 128 bytes");
LMNK_STATIC_ASSERT(offsetof(lmnk_phase_anchor_record_t, root_cause_code) == 59,
                   "lmnk_phase_anchor_record_t root_cause_code offset mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_liminal_ego_state_t) == 64,
                   "lmnk_liminal_ego_state_t must be 64 bytes");
LMNK_STATIC_ASSERT(offsetof(lmnk_liminal_ego_state_t, basin_resistance_q7) == 40,
                   "lmnk_liminal_ego_state_t basin_resistance_q7 offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_liminal_ego_state_t, escape_floor_q7) == 41,
                   "lmnk_liminal_ego_state_t escape_floor_q7 offset mismatch");
LMNK_STATIC_ASSERT(offsetof(lmnk_topology_snapshot_t, spawned_anchors) == 31,
                   "lmnk_topology_snapshot_t spawned_anchors offset mismatch");
LMNK_STATIC_ASSERT(sizeof(lmnk_lacuna_partition_t) == 16,
                   "lmnk_lacuna_partition_t must be 16 bytes");
LMNK_STATIC_ASSERT(sizeof(lmnk_container_header_t) == 64,
                   "lmnk_container_header_t must be 64 bytes");
LMNK_STATIC_ASSERT(sizeof(lmnk_section_entry_t) == 16, "lmnk_section_entry_t must be 16 bytes");
LMNK_STATIC_ASSERT(sizeof(lmnk_topology_matrix_t) == 64, "lmnk_topology_matrix_t must be 64 bytes");

LMNK_EXTERN_C_END

#endif /* LIMINIKA_LMNK_TYPES_H */
