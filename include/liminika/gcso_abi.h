// name: include/liminika/gcso_abi.h
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_ABI_H
#define LIMINIKA_GCSO_ABI_H

#include "liminika/gcso_config.h"
#include "liminika/gcso_types.h"

// ===================================================================
// 0. Version Identifiers (Unified C-ABI v0.1.1 Specification)
// ===================================================================

#define GCSO_ABI_VERSION_MAJOR 0U
#define GCSO_ABI_VERSION_MINOR 1U
#define GCSO_ABI_VERSION_PATCH 1U
#define GCSO_ABI_VERSION_HEX 0x00000101U
#define GCSO_ABI_VERSION_STRING "0.1.1"

GCSO_EXTERN_C_BEGIN

// ===================================================================
// 1. System & Capability Query Interface
// ===================================================================

/**
 * @brief Retrieves the numeric version identifiers for the GCSO C-ABI (v0.1.1).
 * Corresponds to SystemCapabilityQuery trait in traits.rs.
 * @param major Pointer to store the major version number (aligned to uint32_t).
 * @param minor Pointer to store the minor version number (aligned to uint32_t).
 * @param patch Pointer to store the patch version number (aligned to uint32_t).
 * @return GCSO_SUCCESS, GCSO_ERROR_NULL_POINTER, or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_abi_get_version(uint32_t* GCSO_RESTRICT major, uint32_t* GCSO_RESTRICT minor,
                     uint32_t* GCSO_RESTRICT patch) GCSO_NOEXCEPT;

/**
 * @brief Returns the static semantic version string literal for the GCSO kernel engine ("0.1.1").
 * Corresponds to SystemCapabilityQuery trait in traits.rs.
 * @return Const pointer to null-terminated static version string.
 */
GCSO_NODISCARD GCSO_API const char* GCSO_CALL gcso_abi_get_version_string(void) GCSO_NOEXCEPT;

/**
 * @brief Converts a gcso_status_t error code into its corresponding literal string representation.
 * @param status Status code enum value.
 * @return Const pointer to descriptive static string representation.
 */
GCSO_NODISCARD GCSO_API const char* GCSO_CALL gcso_status_to_string(gcso_status_t status)
    GCSO_NOEXCEPT;

/**
 * @brief Queries current execution hardware capabilities and available compute backends.
 * Corresponds to SystemCapabilityQuery trait in traits.rs.
 * @param flags Pointer to store 64-bit bitmask capability flags.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_abi_query_capability(gcso_capability_flags_t* GCSO_RESTRICT flags) GCSO_NOEXCEPT;

/**
 * @brief Initializes a gcso_config_t structure with standard hardware and model defaults.
 * Corresponds to SystemCapabilityQuery trait in traits.rs.
 * @param config Pointer to the user-allocated configuration descriptor to populate (16-byte
 * aligned).
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER / GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_config_init_default(gcso_config_t* GCSO_RESTRICT config) GCSO_NOEXCEPT;

/**
 * @brief Validates a gcso_config_t descriptor against GCSO v0.1.1 architectural invariants.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_config_validate(const gcso_config_t* GCSO_RESTRICT config) GCSO_NOEXCEPT;

/**
 * @brief Safely frees a dynamically allocated string returned by GCSO APIs.
 * @param str_ptr Const string pointer to free (safe no-op if NULL).
 */
GCSO_API void GCSO_CALL gcso_free_string(const char* str_ptr) GCSO_NOEXCEPT;

// ===================================================================
// 2. High-Level Runtime Context Facade Interface
// ===================================================================

/**
 * @brief Allocates and initializes a new GCSO execution context facade instance.
 * Corresponds to GcsoRuntimeContext trait in traits.rs.
 * @param config Pointer to initialized runtime configuration structure (16-byte aligned).
 * @param context_out Pointer to receive the strongly-typed runtime context handle.
 * @return GCSO_SUCCESS or error code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_context_create(const gcso_config_t* GCSO_RESTRICT config,
                    gcso_context_handle_t* GCSO_RESTRICT context_out) GCSO_NOEXCEPT;

/**
 * @brief Resets transient state variables and phase accumulators without freeing allocated tables.
 * Corresponds to GcsoRuntimeContext::reset in traits.rs.
 * @param context Handle to the active runtime context (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_context_reset(gcso_context_handle_t context)
    GCSO_NOEXCEPT;

/**
 * @brief Primary Baseline Endpoint: Projects natural language system prompt text as an Anchor
 * Attractor. Corresponds to GcsoRuntimeContext::set_system_prompt_anchor in traits.rs.
 * @param context Active context handle.
 * @param prompt_text Null-terminated UTF-8 system prompt string.
 * @param weight Attractor pull force scale.
 * @return GCSO_SUCCESS or error code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_context_set_system_prompt_anchor(
    gcso_context_handle_t context, const char* GCSO_RESTRICT prompt_text,
    float weight) GCSO_NOEXCEPT;

/**
 * @brief Executes per-token Hot Path inference step in zero-allocation mode.
 * Applies DPSR phase steering, QDPS step filtering, and updates pointer trails in O(1) time.
 * Corresponds to GcsoRuntimeContext::step_token in traits.rs.
 * @param context Active runtime context handle.
 * @param token_id Input token sequence ID.
 * @param query_tensor 32-byte aligned query tensor buffer to modulate in-place (size: head_dim *
 * num_heads).
 * @param key_tensor 32-byte aligned key tensor buffer (size: head_dim * num_heads).
 * @param trail_out Pointer to receive updated 128-byte pointer trail status (128-byte aligned).
 * @return GCSO_SUCCESS or error code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_context_step_token(
    gcso_context_handle_t context, uint32_t token_id, float* GCSO_RESTRICT query_tensor,
    float* GCSO_RESTRICT key_tensor, gcso_pointer_trail_t* GCSO_RESTRICT trail_out) GCSO_NOEXCEPT;

/**
 * @brief Serializes runtime state into a binary .gcso snapshot format.
 * Corresponds to GcsoRuntimeContext::serialize in traits.rs.
 * @param context Active runtime context handle.
 * @param buffer Target byte buffer (pass NULL to query required size).
 * @param buffer_size Pointer to buffer size variable (in/out).
 * @return GCSO_SUCCESS or GCSO_ERROR_BUFFER_TOO_SMALL.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_context_serialize(gcso_context_handle_t context, uint8_t* GCSO_RESTRICT buffer,
                       size_t* GCSO_RESTRICT buffer_size) GCSO_NOEXCEPT;

/**
 * @brief Deserializes a binary .gcso snapshot to restore context runtime state.
 * Corresponds to GcsoRuntimeContext::deserialize in traits.rs.
 * @param buffer Memory-mapped or allocated input snapshot buffer.
 * @param buffer_size Byte size of input snapshot buffer.
 * @param context_out Pointer to receive restored context handle.
 * @return GCSO_SUCCESS or GCSO_ERROR_CONTAINER_CORRUPTED.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_context_deserialize(const uint8_t* GCSO_RESTRICT buffer, size_t buffer_size,
                         gcso_context_handle_t* GCSO_RESTRICT context_out) GCSO_NOEXCEPT;

/**
 * @brief Seeks to a specific token index in the PPRC keyframe KV cache.
 * Corresponds to PprcCache / GcsoRuntimeContext::seek_to_token in traits.rs.
 * @param context Active context handle.
 * @param token_index Target token index to seek to.
 * @param keyframe_header_out Pointer to receive resulting keyframe header structure (32-byte
 * aligned).
 * @return GCSO_SUCCESS or error code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_pprc_seek_to_token(
    gcso_context_handle_t context, uint32_t token_index,
    gcso_pprc_keyframe_header_t* GCSO_RESTRICT keyframe_header_out) GCSO_NOEXCEPT;

/**
 * @brief Destroys a GCSO runtime context instance and releases associated memory.
 * @param context Runtime context handle to free (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_destroy(gcso_context_handle_t context) GCSO_NOEXCEPT;

/**
 * @brief Destroys a container handle and releases mapped storage resources (C++ kernel symbol).
 * @param container Container handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_container_destroy(gcso_container_handle_t container)
    GCSO_NOEXCEPT;

/**
 * @brief Closes an open .gcso binary container handle (Rust core symbol).
 * @param handle Container handle to close (safe no-op if NULL).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_container_close(gcso_container_handle_t handle) GCSO_NOEXCEPT;

// ===================================================================
// 3. Action Hub, Stigmergic Pointer Trail & DAES Acceleration Interface
// ===================================================================

/**
 * @brief Allocates an Action Hub (Sidecar Pointer Table) instance.
 * Corresponds to PointerActionHub trait in traits.rs.
 * @param capacity Slot capacity count for tagged pointers.
 * @param hub_out Pointer to receive allocated action hub handle.
 * @return GCSO_SUCCESS or error code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_create(
    uint32_t capacity, gcso_action_hub_handle_t* GCSO_RESTRICT hub_out) GCSO_NOEXCEPT;

/**
 * @brief Initializes a 32-byte gcso_stigmergic_thresholds_t structure with canonical v0.1.1
 * defaults.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_init_default_thresholds(
    gcso_stigmergic_thresholds_t* GCSO_RESTRICT thresholds_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates the 7-regime stigmergic threshold ladder on a trail in branchless constant time.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_evaluate_stigmergic_thresholds(
    const gcso_stigmergic_thresholds_t* GCSO_RESTRICT thresholds_opt,
    const gcso_pointer_trail_t* GCSO_RESTRICT trail, float entropy_h_tilde, float phase_coherence,
    float transition_log_prob, uint32_t selected_trail_id, uint8_t bloom_repulsion_density,
    uint8_t dominant_slot,
    gcso_action_orchestration_report_t* GCSO_RESTRICT report_out) GCSO_NOEXCEPT;

/**
 * @brief Executes a full Stigmergic Threshold Action-Selecting Orchestration Step in zero-alloc
 * mode.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_orchestrate_stigmergic_step_safe(
    gcso_action_hub_handle_t hub_opt,
    const gcso_stigmergic_thresholds_t* GCSO_RESTRICT thresholds_opt, uint64_t input_ptr,
    float entropy_h_tilde, uint8_t dominant_slot, gcso_pointer_trail_t* GCSO_RESTRICT trail_inout,
    float* GCSO_RESTRICT attention_scores_inout, size_t num_scores,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor, size_t head_dim,
    size_t num_heads, float* GCSO_RESTRICT logits_inout, size_t vocab_size,
    gcso_action_orchestration_report_t* GCSO_RESTRICT report_out) GCSO_NOEXCEPT;

/**
 * @brief Advances a 64-bit tagged pointer transition in O(1) time within the Action Hub.
 * Corresponds to PointerActionHub::step_pointer in traits.rs.
 * @param context Active runtime context handle.
 * @param current_ptr Encoded 64-bit tagged pointer address.
 * @param trail_out Pointer to receive updated pointer trail structure (128-byte aligned).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_action_hub_step_pointer(gcso_context_handle_t context, uint64_t current_ptr,
                             gcso_pointer_trail_t* GCSO_RESTRICT trail_out) GCSO_NOEXCEPT;

/**
 * @brief Evolves a pointer trail's embedded current_ptr Action Mode flags and Target Attractor Slot
 * in O(1) time.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_evolve_trail_mode(
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, float entropy_h_tilde,
    uint8_t repulsion_triggered, uint8_t dominant_attractor_slot, float low_entropy_skip_thresh,
    float high_density_srl_thresh, uint64_t* GCSO_RESTRICT evolved_ptr_out) GCSO_NOEXCEPT;

/**
 * @brief Bridges the inter-memory Limina gap between prev_trail and curr_trail_inout in O(1) time.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_bridge_liminal_trails(
    const gcso_pointer_trail_t* GCSO_RESTRICT prev_trail,
    gcso_pointer_trail_t* GCSO_RESTRICT curr_trail_inout, uint8_t interpolation_u7,
    float contradiction_threshold, float* GCSO_RESTRICT coherence_out,
    float* GCSO_RESTRICT log_prob_out, uint8_t* GCSO_RESTRICT contradiction_repulsed_out)
    GCSO_NOEXCEPT;

/**
 * @brief Scans candidate memory trails, selects the optimal coherent trail maximizing the
 * Bayes-Attention score, rewires linked_trail_id, and bridges the Limina gap.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_select_and_bridge_best_next_trail(
    gcso_pointer_trail_t* GCSO_RESTRICT src_trail_inout,
    gcso_pointer_trail_t* GCSO_RESTRICT candidate_trails_inout, size_t num_candidates,
    uint32_t head_dim, float phase_interference_gain, float stigmergic_prior_gain,
    float pointer_link_bonus, uint8_t interpolation_u7, float contradiction_threshold,
    uint32_t* GCSO_RESTRICT selected_trail_id_out, float* GCSO_RESTRICT bayesian_score_out,
    float* GCSO_RESTRICT log_prob_out, uint8_t* GCSO_RESTRICT found_out) GCSO_NOEXCEPT;

/**
 * @brief Directly modulates pre-Softmax Attention scores in-place from src_trail and
 * candidate_trails.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_action_hub_modulate_attention_scores_by_candidate_trails(
    float* GCSO_RESTRICT attention_scores_inout,
    const gcso_pointer_trail_t* GCSO_RESTRICT src_trail,
    const gcso_pointer_trail_t* GCSO_RESTRICT candidate_trails, size_t num_candidates,
    uint32_t head_dim, float phase_interference_gain, float stigmergic_prior_gain,
    float pointer_link_bonus, float contradiction_threshold, float repulsion_penalty,
    float* GCSO_RESTRICT mean_coherence_out, uint32_t* GCSO_RESTRICT best_candidate_id_out,
    uint32_t* GCSO_RESTRICT coherent_count_out) GCSO_NOEXCEPT;

/**
 * @brief Executes single-pass fused Bayesian pointer-trail selection, rewiring, Limina Slerp
 * bridging, and optional pre-Softmax Attention score modulation.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_action_hub_select_bridge_and_modulate_bayesian_trails(
    gcso_pointer_trail_t* GCSO_RESTRICT src_trail_inout,
    const gcso_pointer_trail_t* GCSO_RESTRICT candidate_trails, size_t num_candidates,
    float* GCSO_RESTRICT attention_scores_inout, uint32_t head_dim, float phase_interference_gain,
    float stigmergic_prior_gain, float pointer_link_bonus, float contradiction_threshold,
    uint8_t interpolation_u7, float repulsion_penalty,
    uint32_t* GCSO_RESTRICT selected_trail_id_out, float* GCSO_RESTRICT bayesian_score_out,
    float* GCSO_RESTRICT mean_coherence_out, uint32_t* GCSO_RESTRICT coherent_count_out,
    uint8_t* GCSO_RESTRICT found_out) GCSO_NOEXCEPT;

/**
 * @brief Executes Coarse-to-Fine macro-centroid trail routing (O(K)) without Q/K tensor steering.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_route_coarse_to_fine_centroid_only(
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout,
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids, size_t num_centroids,
    uint8_t pull_step_u7, size_t* GCSO_RESTRICT winning_centroid_idx_out,
    uint32_t* GCSO_RESTRICT selected_anchor_id_out, float* GCSO_RESTRICT centroid_coherence_out,
    float* GCSO_RESTRICT gravity_pull_out) GCSO_NOEXCEPT;

/**
 * @brief Executes Coarse-to-Fine macro-to-micro pointer-trail routing (O(K + M)) across centroids
 * and candidate trails.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_route_coarse_to_fine_trail(
    gcso_pointer_trail_t* GCSO_RESTRICT src_trail_inout,
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids, size_t num_centroids,
    const gcso_pointer_trail_t* GCSO_RESTRICT candidate_trails, size_t num_candidates,
    uint32_t head_dim, float phase_interference_gain, float stigmergic_prior_gain,
    float pointer_link_bonus, float cluster_bonus_gain, float contradiction_threshold,
    size_t* GCSO_RESTRICT winning_centroid_idx_out, uint32_t* GCSO_RESTRICT selected_trail_id_out,
    float* GCSO_RESTRICT best_score_out, float* GCSO_RESTRICT best_coherence_out,
    uint8_t* GCSO_RESTRICT found_out) GCSO_NOEXCEPT;

/**
 * @brief Executes single-pass fused Coarse-to-Fine macro-to-micro candidate-trail routing (O(K+M)),
 * Limina Slerp bridging, and in-register QDPS/RIPA phase steering.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_action_hub_route_coarse_to_fine_trail_and_steer_qk_safe(
    gcso_pointer_trail_t* GCSO_RESTRICT src_trail_inout,
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids, size_t num_centroids,
    const gcso_pointer_trail_t* GCSO_RESTRICT candidate_trails, size_t num_candidates,
    size_t head_dim, size_t num_heads, float phase_interference_gain, float stigmergic_prior_gain,
    float pointer_link_bonus, float cluster_bonus_gain, float contradiction_threshold,
    uint8_t interpolation_u7, float min_step_rad, float* GCSO_RESTRICT query_tensor,
    float* GCSO_RESTRICT key_tensor, size_t* GCSO_RESTRICT winning_centroid_idx_out,
    uint32_t* GCSO_RESTRICT selected_trail_id_out, float* GCSO_RESTRICT best_score_out,
    float* GCSO_RESTRICT best_coherence_out, uint8_t* GCSO_RESTRICT found_out) GCSO_NOEXCEPT;

/**
 * @brief Executes full Coarse-to-Fine Bayesian routing, Limina Slerp bridging, optional pre-Softmax
 * Attention score modulation, SO(2)-fused RIPA/PSPM steering, and optional DAES telemetry logging.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_action_hub_route_coarse_to_fine_bayesian_and_steer_qk_safe(
    gcso_pointer_trail_t* GCSO_RESTRICT src_trail_inout,
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids, size_t num_centroids,
    uint8_t coarse_pull_u7, const gcso_pointer_trail_t* GCSO_RESTRICT candidate_trails,
    size_t num_candidates, float* GCSO_RESTRICT attention_scores_inout, size_t head_dim,
    size_t num_heads, float phase_interference_gain, float stigmergic_prior_gain,
    float pointer_link_bonus, float repulsion_penalty, uint8_t fine_interpolation_u7,
    float contradiction_threshold, const gcso_pspm_config_t* GCSO_RESTRICT pspm_cfg_opt,
    float min_step_rad, gcso_daes_slot_t* GCSO_RESTRICT daes_slot_inout,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor,
    size_t* GCSO_RESTRICT winning_centroid_idx_out, uint32_t* GCSO_RESTRICT macro_anchor_id_out,
    uint32_t* GCSO_RESTRICT selected_trail_id_out, float* GCSO_RESTRICT bayesian_score_out,
    float* GCSO_RESTRICT mean_coherence_out, uint32_t* GCSO_RESTRICT coherent_count_out,
    uint8_t* GCSO_RESTRICT found_out) GCSO_NOEXCEPT;

/**
 * @brief Resolves an Action Hub phase command across the 8 parallel 8-head SWAR Micro-Swarm cells.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_resolve_trail_in_swarm(
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, uint64_t action_ptr,
    const gcso_q7_t* GCSO_RESTRICT target_phases, uint8_t active_cell_mask, uint8_t step_gain_u7,
    uint8_t qdps_min_step_q7, uint8_t tolerance_q7, uint8_t quorum_threshold, float entropy_h_tilde,
    float low_entropy_skip_thresh,
    gcso_swarm_resolution_report_t* GCSO_RESTRICT report_out) GCSO_NOEXCEPT;

/**
 * @brief Computes direct 256-slot hash index for 64-bit pointer trail caching using SplitMix64.
 * Corresponds to PointerActionHub::hash_slot256_index in traits.rs.
 * @param ptr Raw 64-bit pointer address.
 * @return 8-bit hash index (0 to 255).
 */
GCSO_NODISCARD GCSO_API uint32_t GCSO_CALL gcso_action_hub_hash_slot256_index(uint64_t ptr)
    GCSO_NOEXCEPT;

/**
 * @brief Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
 * Corresponds to SwarmCellChunk trait in traits.rs.
 * @param context Active context handle.
 * @param mask Paged bitmask structure pointer (32-byte aligned).
 * @param chunk_len Sequence length of target chunk.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_swarm_cell_chunk_step(
    gcso_context_handle_t context, const gcso_paged_bitmask_t* GCSO_RESTRICT mask,
    uint32_t chunk_len) GCSO_NOEXCEPT;

/**
 * @brief Applies attractor pull force F_pull to steer active pointer chains toward target anchor.
 * Corresponds to PointerActionHub::pull_trail_to_attractor in traits.rs.
 * @param context Active runtime context handle.
 * @param trail_id Source pointer trail index.
 * @param anchor_id Target anchor identifier.
 * @param pull_force Floating-point pull force scale.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_action_hub_pull_trail_to_attractor(gcso_context_handle_t context, uint32_t trail_id,
                                        uint32_t anchor_id, float pull_force) GCSO_NOEXCEPT;

/**
 * @brief Links adjacent cellular hallucinated trails into contiguous stigmergic trace graphs.
 * Corresponds to PointerActionHub::link_hallucinated_trails in traits.rs.
 * @param context Active context handle.
 * @param src_trail_id Source trail identifier.
 * @param dst_trail_id Destination trail identifier.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_link_hallucinated_trails(
    gcso_context_handle_t context, uint32_t src_trail_id, uint32_t dst_trail_id) GCSO_NOEXCEPT;

/**
 * @brief Performs hierarchical bit-tree reduction across paged block bitmasks.
 * Corresponds to PointerActionHub::reduce_bit_tree in traits.rs.
 * @param bitmasks Array of input paged bitmasks (32-byte aligned).
 * @param num_masks Number of masks in input array.
 * @param reduced_out Pointer to receive reduced bitmask result (32-byte aligned).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_reduce_bit_tree(
    const gcso_paged_bitmask_t* GCSO_RESTRICT bitmasks, size_t num_masks,
    gcso_paged_bitmask_t* GCSO_RESTRICT reduced_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates SIMD/Warp bitmask reduction over PagedBlock KV caches.
 * Corresponds to PointerActionHub::paged_block_warp_bitmask in traits.rs.
 * @param kv_bits Raw uint64_t array of block key-value bit patterns.
 * @param num_blocks Total number of KV blocks to process.
 * @param mask_out Pointer to store generated paged bitmask (32-byte aligned).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_paged_block_warp_bitmask(
    const uint64_t* GCSO_RESTRICT kv_bits, size_t num_blocks,
    gcso_paged_bitmask_t* GCSO_RESTRICT mask_out) GCSO_NOEXCEPT;

/**
 * @brief Destroys an Action Hub instance and frees internal slot tables.
 * @param hub Action hub handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_destroy(gcso_action_hub_handle_t hub)
    GCSO_NOEXCEPT;

/**
 * @brief Allocates and initializes a DAES slot.
 * Corresponds to DaesScratchpad trait in traits.rs.
 * @param slot_out Pointer to receive allocated DAES slot handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_daes_slot_create(gcso_daes_slot_handle_t* GCSO_RESTRICT slot_out) GCSO_NOEXCEPT;

/**
 * @brief Resets active DAES telemetry mini ledger and cache hit counters in-place.
 * Corresponds to DaesScratchpad::reset_telemetry in traits.rs.
 * @param slot Target DAES slot structure pointer to reset (64-byte aligned).
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_daes_reset_telemetry(gcso_daes_slot_t* GCSO_RESTRICT slot) GCSO_NOEXCEPT;

/**
 * @brief Executes O(1) fast-path shortcut lookup in DAES dynamic scratchpad.
 * Corresponds to DaesScratchpad::fast_path_lookup in traits.rs.
 * @param slot Pointer to DAES slot structure (64-byte aligned).
 * @param input_key Raw key address to look up.
 * @param shortcut_out Pointer to store bypassed address result.
 * @return GCSO_SUCCESS or GCSO_ERROR_INVALID_STATE.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_daes_fast_path_lookup(const gcso_daes_slot_t* GCSO_RESTRICT slot, uint64_t input_key,
                           uint64_t* GCSO_RESTRICT shortcut_out) GCSO_NOEXCEPT;

/**
 * @brief Pushes profiling metric bytes to the DAES telemetry ring ledger in zero-allocation mode.
 * Corresponds to DaesScratchpad::telemetry_push in traits.rs.
 * @param slot Target DAES slot structure pointer (64-byte aligned).
 * @param metric_code Metric identifier byte code.
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_daes_telemetry_push(gcso_daes_slot_t* GCSO_RESTRICT slot, uint8_t metric_code) GCSO_NOEXCEPT;

/**
 * @brief Sets the DAES slot operating mode (0 = Scratchpad, 1 = Plugin, 2 = Shared IPC Buffer).
 * Corresponds to DaesScratchpad::set_mode in traits.rs.
 * @param slot Pointer to DAES slot structure (64-byte aligned).
 * @param mode Desired mode index (0, 1, or 2).
 * @return GCSO_SUCCESS or GCSO_ERROR_INVALID_ARGUMENT.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_daes_set_mode(gcso_daes_slot_t* GCSO_RESTRICT slot, uint32_t mode) GCSO_NOEXCEPT;

/**
 * @brief Evaluates telemetry ledger to auto-tune PSPM routing ratios, RIPA bounds & EDBC
 * thresholds. Corresponds to DaesScratchpad::evaluate_auto_tune in traits.rs.
 * @param slot Const DAES slot structure pointer (64-byte aligned).
 * @param config_out Pointer to receive updated auto-tuned configuration structure (16-byte
 * aligned).
 * @return GCSO_SUCCESS or error code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_daes_evaluate_auto_tune(const gcso_daes_slot_t* GCSO_RESTRICT slot,
                             gcso_config_t* GCSO_RESTRICT config_out) GCSO_NOEXCEPT;

/**
 * @brief Destroys a DAES slot instance and frees allocated scratchpad memory.
 * @param slot Handle to DAES slot instance (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_daes_slot_destroy(gcso_daes_slot_handle_t slot) GCSO_NOEXCEPT;

// ===================================================================
// 4. DPSR, QDPS, PSPM & SRL Interface
// ===================================================================

/**
 * @brief Creates a DPSR phase steering kernel instance.
 * Corresponds to PhaseSteering trait in traits.rs.
 * @param head_dim Attention head dimension (must be even).
 * @param num_heads Total number of attention heads.
 * @param kernel_out Pointer to receive created kernel handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_dpsr_kernel_create(uint32_t head_dim, uint32_t num_heads,
                        gcso_dpsr_kernel_handle_t* GCSO_RESTRICT kernel_out) GCSO_NOEXCEPT;

/**
 * @brief Applies inline DPSR phase rotation to Query tensor registers.
 * Lie Group Invariant: Commutative with standard RoPE in SO(2)^(d_head / 2).
 * Corresponds to PhaseSteering::apply_phase_steering in traits.rs.
 * @param query_tensor 32-byte aligned query tensor buffer.
 * @param phase_deltas Q7 quantized phase delta array per head (Aligned 16-byte).
 * @param head_dim Head dimension size (must be even).
 * @param num_heads Total head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering(
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads) GCSO_NOEXCEPT;

/**
 * @brief Applies RIPA soft-bounded tanh clamping on low-frequency channels (upper d_head / 4
 * dimensions). Corresponds to PhaseSteering::apply_phase_steering_safe in traits.rs.
 * @param query_tensor 32-byte aligned query tensor buffer.
 * @param phase_deltas Q7 quantized phase delta array (Aligned 16-byte).
 * @param head_dim Head dimension size (must be even).
 * @param num_heads Total head count.
 * @param max_rad Maximum allowed phase limit in radians.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_safe(
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads, float max_rad) GCSO_NOEXCEPT;

/**
 * @brief Applies Order-Invariant Single-Manifold Unified-Field Phase Steering across num_anchors.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_unified_field_phase_steering_safe(
    const gcso_q7_t* GCSO_RESTRICT trail_phases, float trail_init_weight,
    const gcso_q7_t* GCSO_RESTRICT anchor_phases, const float* GCSO_RESTRICT anchor_base_energies,
    size_t num_anchors, float min_step_rad, float max_rad, float* GCSO_RESTRICT query_tensor,
    float* GCSO_RESTRICT key_tensor, size_t head_dim, size_t num_heads,
    float* GCSO_RESTRICT total_pull_out, float* GCSO_RESTRICT mean_coherence_out,
    uint32_t* GCSO_RESTRICT dominant_anchor_id_out) GCSO_NOEXCEPT;

/**
 * @brief Directly modulates pre-Softmax candidate memory/key Attention scores in-place by
 * Attention x DPSR Phase Interference across key_anchor_phases.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_modulate_attention_scores_by_phase_bank(
    float* GCSO_RESTRICT attention_scores_inout, const gcso_q7_t* GCSO_RESTRICT query_phases,
    const gcso_q7_t* GCSO_RESTRICT key_anchor_phases, size_t num_candidates, uint32_t head_dim,
    float interference_gain, float* GCSO_RESTRICT mean_coherence_out) GCSO_NOEXCEPT;

/**
 * @brief QDPS discrete filter: cuts off phase rotation steps falling below min_step_rad.
 * Corresponds to PhaseSteering::qdps_filter_step in traits.rs.
 * @param phase_deltas Q7 phase array to filter in-place (Aligned 16-byte).
 * @param len Array element count.
 * @param min_step_rad Minimum step threshold angle in radians.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_qdps_filter_step(
    gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t len, float min_step_rad) GCSO_NOEXCEPT;

/**
 * @brief Lazy Phase Unwrapping: Applies relative phase shift against context accumulator on Query
 * side. Corresponds to PhaseSteering::lazy_unwrap_override in traits.rs.
 * @param query_tensor 32-byte aligned target query tensor.
 * @param context_accum Cumulative context phase vector (32-byte aligned).
 * @param head_dim Head dimension (must be even).
 * @param num_heads Head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_lazy_unwrap_override(
    float* GCSO_RESTRICT query_tensor, const float* GCSO_RESTRICT context_accum, size_t head_dim,
    size_t num_heads) GCSO_NOEXCEPT;

/**
 * @brief Executes norm-guarded Slerp phase stabilization on state vectors.
 * Corresponds to PhaseSteering::slerp_norm_guard_stable in traits.rs.
 * @param tensor Vector tensor buffer to normalize in-place (32-byte aligned).
 * @param dim Total vector length.
 * @param norm_lower Minimum allowed L2 norm bound.
 * @param norm_upper Maximum allowed L2 norm bound.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_slerp_norm_guard_stable(
    float* GCSO_RESTRICT tensor, size_t dim, float norm_lower, float norm_upper) GCSO_NOEXCEPT;

/**
 * @brief Fused inline logit phase shift prior to LM Head Softmax.
 * Corresponds to PhaseSteering::fused_logit_shift in traits.rs.
 * @param logits Logit score vector buffer (32-byte aligned).
 * @param vocab_size Vocabulary dimension.
 * @param phase_deltas Q7 phase deltas (Aligned 16-byte).
 * @param num_heads Head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_fused_logit_shift(
    float* GCSO_RESTRICT logits, size_t vocab_size, const gcso_q7_t* GCSO_RESTRICT phase_deltas,
    size_t num_heads) GCSO_NOEXCEPT;

/**
 * @brief Computes Procrustes phase delta alignment between source and target state representations.
 * Corresponds to PhaseSteering::compute_procrustes_phase_delta in traits.rs.
 * @param source 32-byte aligned source feature vector.
 * @param target 32-byte aligned target feature vector.
 * @param dim Total vector dimension size.
 * @param phase_out Output Q7 phase delta array buffer to store computed profile (Aligned 16-byte).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_compute_procrustes_phase_delta(
    const float* GCSO_RESTRICT source, const float* GCSO_RESTRICT target, size_t dim,
    gcso_q7_t* GCSO_RESTRICT phase_out) GCSO_NOEXCEPT;

/**
 * @brief Destroys a DPSR kernel instance.
 * @param kernel Kernel handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_kernel_destroy(gcso_dpsr_kernel_handle_t kernel)
    GCSO_NOEXCEPT;

/**
 * @brief Allocates a PSPM router instance.
 * Corresponds to PspmRouter trait in traits.rs.
 * @param config PSPM router configuration pointer (16-byte aligned).
 * @param router_out Pointer to receive allocated router handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_pspm_router_create(const gcso_pspm_config_t* GCSO_RESTRICT config,
                        gcso_pspm_router_handle_t* GCSO_RESTRICT router_out) GCSO_NOEXCEPT;

/**
 * @brief Dispatches PSPM head-group phase profiles (Fact, Logic, Explore) in a single pass.
 * Corresponds to PspmRouter::dispatch_single_pass in traits.rs.
 * @param query_tensor 32-byte aligned query tensor buffer.
 * @param pspm_cfg PSPM routing configuration (16-byte aligned).
 * @param head_dim Head dimension (must be even).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_pspm_dispatch_single_pass(
    float* GCSO_RESTRICT query_tensor, const gcso_pspm_config_t* GCSO_RESTRICT pspm_cfg,
    size_t head_dim) GCSO_NOEXCEPT;

/**
 * @brief Destroys a PSPM router instance.
 * @param router Router handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_pspm_router_destroy(gcso_pspm_router_handle_t router)
    GCSO_NOEXCEPT;

/**
 * @brief Creates a Sparse Residual Adapter Layer (SRL) instance.
 * Corresponds to SrlAdapter trait in traits.rs.
 * @param descriptor Pointer to SRL descriptor (32-byte aligned).
 * @param adapter_out Pointer to receive allocated adapter handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_srl_adapter_create(const gcso_srl_descriptor_t* GCSO_RESTRICT descriptor,
                        gcso_srl_adapter_handle_t* GCSO_RESTRICT adapter_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates SRL Dynamic Rank-1 outer product: y = alpha * s (*) (u * (v^T * x)).
 * Corresponds to SrlAdapter::eval_rank1 in traits.rs.
 * @param y_out Output vector buffer to write into (32-byte aligned).
 * @param x_in Input vector buffer (32-byte aligned).
 * @param srl_desc SRL descriptor structure pointer (32-byte aligned).
 * @param dim_in Input dimension.
 * @param dim_out Output dimension.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_srl_eval_rank1(float* GCSO_RESTRICT y_out, const float* GCSO_RESTRICT x_in,
                    const gcso_srl_descriptor_t* GCSO_RESTRICT srl_desc, size_t dim_in,
                    size_t dim_out) GCSO_NOEXCEPT;

/**
 * @brief L2P-SVD: Projects fine-tuned LoRA matrices via SVD into phase profiles and SRL vectors.
 * Corresponds to L2pSvdProjector trait in traits.rs.
 * @param lora_a Pointer to LoRA A matrix buffer (32-byte aligned).
 * @param lora_b Pointer to LoRA B matrix buffer (32-byte aligned).
 * @param rank Rank size of input LoRA.
 * @param dim_in Input dimension.
 * @param dim_out Output dimension (must be even).
 * @param srl_out Output SRL descriptor structure to populate (32-byte aligned).
 * @param phase_profile_out Output Q7 phase profile array pointer (Aligned 16-byte).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_l2p_svd_project_lora(
    const float* GCSO_RESTRICT lora_a, const float* GCSO_RESTRICT lora_b, size_t rank,
    size_t dim_in, size_t dim_out, gcso_srl_descriptor_t* GCSO_RESTRICT srl_out,
    gcso_q7_t* GCSO_RESTRICT phase_profile_out) GCSO_NOEXCEPT;

/**
 * @brief Destroys an SRL adapter instance.
 * @param adapter Adapter handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_srl_adapter_destroy(gcso_srl_adapter_handle_t adapter)
    GCSO_NOEXCEPT;

// ===================================================================
// 5. Attractor Field, EDBC Engine & cvoid Alignment Interface
// ===================================================================

/**
 * @brief Allocates an Attractor Field instance.
 * Corresponds to AttractorField trait in traits.rs.
 * @param field_out Pointer to receive allocated field handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_attractor_field_create(gcso_attractor_field_handle_t* GCSO_RESTRICT field_out) GCSO_NOEXCEPT;

/**
 * @brief Registers a topological anchor point in the attractor field.
 * Corresponds to AttractorField::add_anchor in traits.rs.
 * @param context Active context handle.
 * @param anchor_type Anchor classification enum type.
 * @param vec 32-byte aligned feature vector.
 * @param dim Vector dimension size.
 * @param anchor_id_out Pointer to receive assigned anchor ID.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_add_anchor(
    gcso_context_handle_t context, gcso_anchor_type_t anchor_type, const float* GCSO_RESTRICT vec,
    size_t dim, uint32_t* GCSO_RESTRICT anchor_id_out) GCSO_NOEXCEPT;

/**
 * @brief Maps natural language system prompt text as primary Anchor Attractor.
 * Corresponds to AttractorField::add_system_prompt_anchor in traits.rs.
 * @param context Active runtime context handle.
 * @param prompt_text Null-terminated UTF-8 system prompt string.
 * @param weight Attractor weight parameter.
 * @param anchor_id_out Pointer to store assigned anchor ID.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_add_system_prompt_anchor(
    gcso_context_handle_t context, const char* GCSO_RESTRICT prompt_text, float weight,
    uint32_t* GCSO_RESTRICT anchor_id_out) GCSO_NOEXCEPT;

/**
 * @brief Maps dense feature embedding vector as continuous attractor anchor.
 * Corresponds to AttractorField::add_embedding_anchor in traits.rs.
 * @param context Active runtime context handle.
 * @param embedding 32-byte aligned embedding array.
 * @param dim Vector dimension.
 * @param weight Attractor weight parameter.
 * @param anchor_id_out Pointer to store assigned anchor ID.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_add_embedding_anchor(
    gcso_context_handle_t context, const float* GCSO_RESTRICT embedding, size_t dim, float weight,
    uint32_t* GCSO_RESTRICT anchor_id_out) GCSO_NOEXCEPT;

/**
 * @brief Injects phase-conjugate repulsion vector (-dTheta) to flip spurious local minima into
 * repulsive peaks. Corresponds to AttractorField::inject_phase_repulsion in traits.rs.
 * @param context Active runtime context handle.
 * @param repulsion_deltas Q7 anti-phase array (Aligned 16-byte).
 * @param num_heads Total head count.
 * @param gain Repulsion strength multiplier.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_inject_phase_repulsion(
    gcso_context_handle_t context, const gcso_q7_t* GCSO_RESTRICT repulsion_deltas,
    size_t num_heads, float gain) GCSO_NOEXCEPT;

/**
 * @brief Evaluates Zero-Hyperparameter Phase-Proximity Gravity on the Single Unified Attractor
 * Phase Manifold between trail_phases and anchor_phases.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_eval_resonant_pull(
    const gcso_q7_t* GCSO_RESTRICT trail_phases, const gcso_q7_t* GCSO_RESTRICT anchor_phases,
    float base_energy, float* GCSO_RESTRICT resonant_pull_out,
    float* GCSO_RESTRICT coherence_out) GCSO_NOEXCEPT;

/**
 * @brief Order-invariant folding of num_anchors on the Single Unified Attractor Phase Manifold.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_fold_unified_field(
    const gcso_q7_t* GCSO_RESTRICT trail_phases, float trail_init_weight,
    const gcso_q7_t* GCSO_RESTRICT anchor_phases, const float* GCSO_RESTRICT anchor_base_energies,
    size_t num_anchors, gcso_q7_t* GCSO_RESTRICT folded_phases_out,
    float* GCSO_RESTRICT total_pull_out, float* GCSO_RESTRICT mean_coherence_out,
    uint32_t* GCSO_RESTRICT dominant_anchor_id_out,
    uint32_t* GCSO_RESTRICT active_anchor_count_out) GCSO_NOEXCEPT;

/**
 * @brief Steers trail_inout in-place directly on the Single Unified Attractor Phase Manifold.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_steer_trail_unified_field(
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, const gcso_q7_t* GCSO_RESTRICT anchor_phases,
    const float* GCSO_RESTRICT anchor_base_energies, size_t num_anchors,
    const gcso_q7_t* GCSO_RESTRICT repulsion_phases, float inhibition_strength,
    float* GCSO_RESTRICT final_pull_out, float* GCSO_RESTRICT mean_coherence_out,
    uint32_t* GCSO_RESTRICT active_anchor_count_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates non-linear Cross-Slot Attractor Synergy and Liminal Beat Perturbation between
 * two 64-head Q7 attractor slot profiles gated by an 8x8 coupling bitmatrix.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_eval_cross_slot_synergy(
    const gcso_q7_t* GCSO_RESTRICT phases_a, float weight_a,
    const gcso_q7_t* GCSO_RESTRICT phases_b, float weight_b, uint32_t slot_a, uint32_t slot_b,
    uint64_t coupling_matrix_8x8, float synergy_gain, gcso_q7_t* GCSO_RESTRICT fused_phases_out,
    float* GCSO_RESTRICT effective_pull_out, float* GCSO_RESTRICT coherence_out) GCSO_NOEXCEPT;

/**
 * @brief Applies branchless Mutual Slot Inhibition in-place to a 64-head Q7 phase profile.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_apply_mutual_slot_inhibition(
    gcso_q7_t* GCSO_RESTRICT target_phases_inout, float target_weight,
    const gcso_q7_t* GCSO_RESTRICT inhibitor_phases, float inhibition_strength,
    float* GCSO_RESTRICT damped_weight_out) GCSO_NOEXCEPT;

/**
 * @brief Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors.
 * Corresponds to AttractorField::aggregate_bottom_up in traits.rs.
 * @param context Active context handle.
 * @param new_anchor_count_out Pointer to receive updated total anchor count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_aggregate_bottom_up(
    gcso_context_handle_t context, uint32_t* GCSO_RESTRICT new_anchor_count_out) GCSO_NOEXCEPT;

/**
 * @brief Clears all registered topological anchors from the attractor field.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_attractor_field_clear_anchors(gcso_attractor_field_handle_t field) GCSO_NOEXCEPT;

/**
 * @brief Destroys an attractor field instance.
 * @param field Attractor field handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_destroy(gcso_attractor_field_handle_t field)
    GCSO_NOEXCEPT;

/**
 * @brief Allocates an EDBC (Entropy-Driven Decoding Branch Controller) instance.
 * Corresponds to EntropyEvaluator trait in traits.rs.
 * @param initial_state Pointer to initial state configuration structure (32-byte aligned).
 * @param controller_out Pointer to receive allocated controller handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_edbc_controller_create(
    const gcso_edbc_state_t* GCSO_RESTRICT initial_state,
    gcso_edbc_controller_handle_t* GCSO_RESTRICT controller_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates Moving Z-Score Normalized Attention Entropy H~ and pitchfork bifurcation mode.
 * Corresponds to EntropyEvaluator::eval_stateful in traits.rs.
 * @param controller Controller handle.
 * @param token_z_score Floating-point activation Z-score.
 * @param state_out Output state structure pointer (32-byte aligned).
 * @return GCSO_SUCCESS or GCSO_ERROR_EDBC_SINGULARITY (if NaN/Inf detected).
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_edbc_eval_stateful(gcso_edbc_controller_handle_t controller, float token_z_score,
                        gcso_edbc_state_t* GCSO_RESTRICT state_out) GCSO_NOEXCEPT;

/**
 * @brief Computes cvoid Coherent Vector Alignment Metric for Out-of-Distribution Latent Space.
 * Corresponds to EntropyEvaluator::eval_cvoid_dyadic128 in traits.rs.
 * @param key_vector 32-byte aligned key tensor vector.
 * @param dim Vector dimension.
 * @param void_score_out Pointer to receive computed void score result.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_cvoid_eval_dyadic128(const float* GCSO_RESTRICT key_vector, size_t dim,
                          float* GCSO_RESTRICT void_score_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates Eyring-Kramers potential barrier height value with singularity check.
 * Corresponds to EntropyEvaluator::eval_barrier in traits.rs.
 * @param void_score Raw void score.
 * @param tau_eff Effective threshold temperature.
 * @param barrier_out Pointer to receive computed barrier height.
 * @return GCSO_SUCCESS or GCSO_ERROR_EDBC_SINGULARITY.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_cvoid_eval_barrier(
    float void_score, float tau_eff, float* GCSO_RESTRICT barrier_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates scale-invariant Pitchfork Bifurcation potential barrier height (Whitepaper Lemma
 * 3.1).
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_cvoid_eval_barrier_pitchfork(float cvoid, float tau_eff, float alpha, float beta,
                                  uint32_t head_dim, float* GCSO_RESTRICT barrier_out)
    GCSO_NOEXCEPT;

/**
 * @brief Destroys an EDBC controller instance.
 * @param controller Controller handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_edbc_controller_destroy(gcso_edbc_controller_handle_t controller) GCSO_NOEXCEPT;

// ===================================================================
// 6. Persona Patch & ZIMMS Storage Mechanics Interface
// ===================================================================

/**
 * @brief Dynamic application of persona phase modulation patches without altering base weights.
 * Corresponds to PersonaPatcher trait in traits.rs.
 * @param context Active runtime context handle.
 * @param patch_data Pointer to binary patch buffer.
 * @param patch_size Size of binary patch in bytes.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_persona_apply_patch(gcso_context_handle_t context, const uint8_t* GCSO_RESTRICT patch_data,
                         size_t patch_size) GCSO_NOEXCEPT;

/**
 * @brief Zero-Overhead In-Memory Mapped Storage: Maps .gcso container payload using zero-copy mmap.
 * Corresponds to ZimmsStorage::open_mmap in traits.rs and attractor_edbc.cpp.
 * @param file_path Null-terminated path string to target .gcso file.
 * @param zimms_out Descriptor structure pointer to populate (32-byte aligned).
 * @return GCSO_SUCCESS or GCSO_ERROR_ZIMMS_MAPPING_FAILED.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_zimms_open_mmap(const char* GCSO_RESTRICT file_path,
                     gcso_zimms_descriptor_t* GCSO_RESTRICT zimms_out) GCSO_NOEXCEPT;

/**
 * @brief Unmaps zero-copy ZIMMS memory handle and releases Direct DMA resources.
 * Corresponds to ZimmsStorage::close_mmap in traits.rs.
 * @param zimms_desc Descriptor structure pointer to clear (32-byte aligned).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_zimms_close_mmap(gcso_zimms_descriptor_t* GCSO_RESTRICT zimms_desc) GCSO_NOEXCEPT;

// ===================================================================
// 7. Tagged Pointer, 2-Origin Composite Manifold Gateway, Online
//    Toroidal k-Means, Coarse-to-Fine Routing & 4-Section .gcso
//    Zero-Copy Persistence Interface (v0.1.1 Extensions)
// ===================================================================

/**
 * @brief Inline branchless helper packing a 48-bit canonical address, 8-bit domain/action tag,
 * and Q7 phase offset into a 64-bit tagged pointer word.
 */
static GCSO_FORCE_INLINE gcso_tagged_ptr_t
gcso_tagged_ptr_pack_inline(uint64_t raw_addr, uint8_t tag, gcso_q7_t phase) GCSO_NOEXCEPT {
    const uint64_t addr = raw_addr & GCSO_TAGGED_PTR_ADDR_MASK;
    const uint64_t tag_bits = ((uint64_t)tag) << GCSO_TAGGED_PTR_TAG_SHIFT;
    const uint64_t phase_bits = ((uint64_t)((uint8_t)phase)) << GCSO_TAGGED_PTR_PHASE_SHIFT;
    return addr | tag_bits | phase_bits;
}

/**
 * @brief Inline branchless helper unpacking the 48-bit canonical address from a tagged pointer.
 */
static GCSO_FORCE_INLINE uint64_t gcso_tagged_ptr_unpack_addr_inline(gcso_tagged_ptr_t ptr)
    GCSO_NOEXCEPT {
    return ptr & GCSO_TAGGED_PTR_ADDR_MASK;
}

/**
 * @brief Inline branchless helper unpacking the 8-bit tag from a tagged pointer.
 */
static GCSO_FORCE_INLINE uint8_t gcso_tagged_ptr_unpack_tag_inline(gcso_tagged_ptr_t ptr)
    GCSO_NOEXCEPT {
    return (uint8_t)((ptr & GCSO_TAGGED_PTR_TAG_MASK) >> GCSO_TAGGED_PTR_TAG_SHIFT);
}

/**
 * @brief Inline branchless helper unpacking the Q7 phase offset from a tagged pointer.
 */
static GCSO_FORCE_INLINE gcso_q7_t gcso_tagged_ptr_unpack_phase_inline(gcso_tagged_ptr_t ptr)
    GCSO_NOEXCEPT {
    return (gcso_q7_t)((uint8_t)(ptr >> GCSO_TAGGED_PTR_PHASE_SHIFT));
}

/**
 * @brief Packs a 48-bit canonical address, 8-bit tag, and Q7 phase offset into a 64-bit tagged
 * pointer.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_pack(uint64_t raw_addr, uint8_t tag,
                                                                gcso_q7_t phase) GCSO_NOEXCEPT;

/**
 * @brief Unpacks the 48-bit canonical address from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_unpack_addr(uint64_t ptr) GCSO_NOEXCEPT;

/**
 * @brief Unpacks the 8-bit domain/action tag from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint8_t GCSO_CALL gcso_tagged_ptr_unpack_tag(uint64_t ptr) GCSO_NOEXCEPT;

/**
 * @brief Unpacks the 8-bit Q7 phase state from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API gcso_q7_t GCSO_CALL gcso_tagged_ptr_unpack_phase(uint64_t ptr)
    GCSO_NOEXCEPT;

/**
 * @brief Packs a 48-bit address, 3-bit Attractor Slot ID, 5-bit Action Mode mask, and Q7 phase into
 * a tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL
gcso_tagged_ptr_pack_action(uint64_t raw_addr, uint8_t attractor_slot, uint8_t action_mode_flags,
                            gcso_q7_t phase) GCSO_NOEXCEPT;

/**
 * @brief Extracts the 5-bit Action Mode mask from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint8_t GCSO_CALL gcso_tagged_ptr_unpack_action_mode(uint64_t ptr)
    GCSO_NOEXCEPT;

/**
 * @brief Extracts the 3-bit Target Attractor Working Lane ID (0..7) from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint8_t GCSO_CALL gcso_tagged_ptr_unpack_attractor_slot(uint64_t ptr)
    GCSO_NOEXCEPT;

/**
 * @brief Evolves a 64-bit tagged pointer's Action Mode flags and Target Attractor Slot in O(1)
 * time.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_evolve_mode_stigmergically(
    uint64_t ptr, float stigmergic_density, float entropy_h_tilde, uint8_t repulsion_triggered,
    uint8_t dominant_attractor_slot, float low_entropy_skip_thresh,
    float high_density_srl_thresh) GCSO_NOEXCEPT;

/**
 * @brief Applies lockstep fused DPSR phase rotation to Query and Key tensor registers in a single
 * pass.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_fused(
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim, size_t num_heads) GCSO_NOEXCEPT;

/**
 * @brief Applies RIPA soft-bounded phase steering to both Query and Key tensors in a single fused
 * pass.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_fused_safe(
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim, size_t num_heads,
    float max_rad) GCSO_NOEXCEPT;

/**
 * @brief Applies S^1 geodesic RIPA soft-bounded phase steering to a Query tensor.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_geodesic_safe(
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads, float max_rad) GCSO_NOEXCEPT;

/**
 * @brief Applies pointer-gated RIPA phase steering (honoring ACTION_MODE_DPSR_STEER).
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_pointer_gated_phase_steering_safe(
    uint64_t tagged_ptr, float* GCSO_RESTRICT query_tensor,
    const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim, size_t num_heads,
    float max_rad) GCSO_NOEXCEPT;

/**
 * @brief Applies pointer-gated fused QK RIPA phase steering in a single pass.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_dpsr_apply_pointer_gated_phase_steering_fused_safe(uint64_t tagged_ptr,
                                                        float* GCSO_RESTRICT query_tensor,
                                                        float* GCSO_RESTRICT key_tensor,
                                                        const gcso_q7_t* GCSO_RESTRICT phase_deltas,
                                                        size_t head_dim, size_t num_heads,
                                                        float max_rad) GCSO_NOEXCEPT;

/**
 * @brief Applies soft exponential phase damping across Query tensor channels.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_dpsr_apply_soft_phase_damping(float* GCSO_RESTRICT query_tensor, size_t head_dim,
                                   size_t num_heads, float damping_factor) GCSO_NOEXCEPT;

/**
 * @brief Evaluates in-place fused SRL Dynamic Rank-1 outer product accumulation.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_srl_eval_rank1_fused(float* GCSO_RESTRICT y_inout, const float* GCSO_RESTRICT x_in,
                          const gcso_srl_descriptor_t* GCSO_RESTRICT srl_desc, size_t dim_in,
                          size_t dim_out) GCSO_NOEXCEPT;

/**
 * @brief Projects an arbitrary byte slice into a 64-head Q7 phase profile via 64-bit SWAR mixing.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_gateway_project_bytes_to_q7_64(const uint8_t* GCSO_RESTRICT bytes, size_t byte_len,
                                    gcso_q7_t* GCSO_RESTRICT phases_out) GCSO_NOEXCEPT;

/**
 * @brief Projects a dense f32 embedding slice into a 64-head Q7 phase profile via planar atan2.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_gateway_project_f32_to_q7_64(const float* GCSO_RESTRICT values, size_t num_values, float gain,
                                  gcso_q7_t* GCSO_RESTRICT phases_out) GCSO_NOEXCEPT;

/**
 * @brief Synthesizes a 128-byte 2-Origin Composite Manifold Gateway frame and 4-Axis Causal
 * Provenance report.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_gateway_synthesize_multi_origin_frame(
    float vram_pressure, float gpu_frame_overrun, uint8_t active_swarm_cell_mask,
    uint8_t recommend_layer_skip, const gcso_q7_t* GCSO_RESTRICT fact_phases, float fact_confidence,
    uint64_t fact_zimms_offset48, uint32_t fact_anchor_id,
    const gcso_q7_t* GCSO_RESTRICT persona_phases, float persona_strength,
    uint64_t persona_zimms_offset48, uint32_t persona_anchor_id,
    const gcso_contrastive_bloom_filter_t* GCSO_RESTRICT bloom_filter_opt,
    float contradiction_threshold, uint8_t bloom_repulsion_threshold,
    gcso_gateway_frame_t* GCSO_RESTRICT frame_out,
    gcso_causal_provenance_report_t* GCSO_RESTRICT report_out) GCSO_NOEXCEPT;

/**
 * @brief Executes closed-loop 2-Origin Composite Manifold Gateway Steering in register-resident
 * mode.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_steer_gateway_frame_safe(
    const gcso_gateway_frame_t* GCSO_RESTRICT frame,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, uint8_t unresolved_damping_u7,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor, size_t head_dim,
    size_t num_heads, float* GCSO_RESTRICT logits_inout, size_t vocab_size,
    gcso_causal_provenance_report_t* GCSO_RESTRICT report_out) GCSO_NOEXCEPT;

/**
 * @brief Executes single-pass fused 2-Origin Gateway Steering and Online Toroidal k-Means /
 * DP-Means macro crystallization.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_dpsr_steer_and_kmeans_crystallize_gateway_frame_safe(
    const gcso_gateway_frame_t* GCSO_RESTRICT frame,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, uint8_t unresolved_damping_u7,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor, size_t head_dim,
    size_t num_heads, float* GCSO_RESTRICT logits_inout, size_t vocab_size,
    gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids_inout, size_t num_centroids,
    float split_coherence_threshold, gcso_causal_provenance_report_t* GCSO_RESTRICT report_out,
    size_t* GCSO_RESTRICT assigned_centroid_idx_out, float* GCSO_RESTRICT assigned_coherence_out,
    float* GCSO_RESTRICT updated_gravity_mass_out) GCSO_NOEXCEPT;

/**
 * @brief One-shot 2-Origin Multi-Blanket Gateway synthesis, SO(2)-fused RIPA/PSPM steering, and
 * optional Online Toroidal k-Means / DP-Means macro crystallization from Q7 phase pointers.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_gateway_synthesize_kmeans_and_steer_qk_safe(
    float vram_pressure, float gpu_frame_overrun, uint8_t active_swarm_cell_mask,
    uint8_t recommend_layer_skip, const gcso_q7_t* GCSO_RESTRICT fact_phases, float fact_confidence,
    uint64_t fact_zimms_offset48, uint32_t fact_anchor_id,
    const gcso_q7_t* GCSO_RESTRICT persona_phases, float persona_strength,
    uint64_t persona_zimms_offset48, uint32_t persona_anchor_id,
    const gcso_contrastive_bloom_filter_t* GCSO_RESTRICT bloom_filter_opt,
    float contradiction_threshold, uint8_t bloom_repulsion_threshold,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, uint8_t unresolved_damping_u7,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor, size_t head_dim,
    size_t num_heads, float* GCSO_RESTRICT logits_inout, size_t vocab_size,
    gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids_inout, size_t num_centroids,
    float split_coherence_threshold, gcso_gateway_frame_t* GCSO_RESTRICT frame_out,
    gcso_causal_provenance_report_t* GCSO_RESTRICT report_out,
    size_t* GCSO_RESTRICT assigned_centroid_idx_out, float* GCSO_RESTRICT assigned_coherence_out,
    float* GCSO_RESTRICT updated_gravity_mass_out) GCSO_NOEXCEPT;

/**
 * @brief One-shot Raw-Byte 2-Origin Gateway synthesis, SO(2)-fused RIPA/PSPM steering, and optional
 * Online Toroidal k-Means / DP-Means macro crystallization.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_gateway_project_bytes_synthesize_kmeans_and_steer_qk_safe(
    float vram_pressure, float gpu_frame_overrun, uint8_t active_swarm_cell_mask,
    uint8_t recommend_layer_skip, const uint8_t* GCSO_RESTRICT fact_bytes, size_t fact_byte_len,
    float fact_confidence, uint64_t fact_zimms_offset48, uint32_t fact_anchor_id,
    const uint8_t* GCSO_RESTRICT persona_bytes, size_t persona_byte_len, float persona_strength,
    uint64_t persona_zimms_offset48, uint32_t persona_anchor_id,
    const gcso_contrastive_bloom_filter_t* GCSO_RESTRICT bloom_filter_opt,
    float contradiction_threshold, uint8_t bloom_repulsion_threshold,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, uint8_t unresolved_damping_u7,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor, size_t head_dim,
    size_t num_heads, float* GCSO_RESTRICT logits_inout, size_t vocab_size,
    gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids_inout, size_t num_centroids,
    float split_coherence_threshold, gcso_gateway_frame_t* GCSO_RESTRICT frame_out,
    gcso_causal_provenance_report_t* GCSO_RESTRICT report_out,
    size_t* GCSO_RESTRICT assigned_centroid_idx_out, float* GCSO_RESTRICT assigned_coherence_out,
    float* GCSO_RESTRICT updated_gravity_mass_out) GCSO_NOEXCEPT;

/**
 * @brief One-shot Raw-f32 Embedding 2-Origin Gateway synthesis, SO(2)-fused RIPA/PSPM steering, and
 * optional Online Toroidal k-Means / DP-Means macro crystallization.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_gateway_project_f32_synthesize_kmeans_and_steer_qk_safe(
    float vram_pressure, float gpu_frame_overrun, uint8_t active_swarm_cell_mask,
    uint8_t recommend_layer_skip, const float* GCSO_RESTRICT fact_values, size_t fact_len,
    float fact_projection_gain, float fact_confidence, uint64_t fact_zimms_offset48,
    uint32_t fact_anchor_id, const float* GCSO_RESTRICT persona_values, size_t persona_len,
    float persona_projection_gain, float persona_strength, uint64_t persona_zimms_offset48,
    uint32_t persona_anchor_id,
    const gcso_contrastive_bloom_filter_t* GCSO_RESTRICT bloom_filter_opt,
    float contradiction_threshold, uint8_t bloom_repulsion_threshold,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, uint8_t unresolved_damping_u7,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor, size_t head_dim,
    size_t num_heads, float* GCSO_RESTRICT logits_inout, size_t vocab_size,
    gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids_inout, size_t num_centroids,
    float split_coherence_threshold, gcso_gateway_frame_t* GCSO_RESTRICT frame_out,
    gcso_causal_provenance_report_t* GCSO_RESTRICT report_out,
    size_t* GCSO_RESTRICT assigned_centroid_idx_out, float* GCSO_RESTRICT assigned_coherence_out,
    float* GCSO_RESTRICT updated_gravity_mass_out) GCSO_NOEXCEPT;

/**
 * @brief Synthesizes a closed-loop GatewayFrame and optional CausalProvenanceReport directly from
 * crystallized macro centroids.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_gateway_synthesize_from_crystallized_records(
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT records, size_t num_records,
    gcso_gateway_frame_t* GCSO_RESTRICT frame_out,
    gcso_causal_provenance_report_t* GCSO_RESTRICT report_out) GCSO_NOEXCEPT;

/**
 * @brief Executes one-shot Closed-Loop Macro-Distribution Gateway Re-Injection and SO(2)-fused
 * RIPA/PSPM steering from crystallized records.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_dpsr_steer_closed_loop_crystallized_gateway_safe(
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT records, size_t num_records,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, uint8_t unresolved_damping_u7,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor, size_t head_dim,
    size_t num_heads, float* GCSO_RESTRICT logits_inout, size_t vocab_size,
    gcso_gateway_frame_t* GCSO_RESTRICT frame_out,
    gcso_causal_provenance_report_t* GCSO_RESTRICT report_out) GCSO_NOEXCEPT;

/**
 * @brief Folds num_records crystallized attractor centroids onto trail_inout via Phase-Proximity
 * Gravity and steers Query/Key tensors in zero-copy register-resident mode.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_dpsr_steer_crystallized_ensemble_zero_copy_safe(
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT records, size_t num_records,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, float min_step_rad,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor, size_t head_dim,
    size_t num_heads, float* GCSO_RESTRICT total_pull_out,
    uint32_t* GCSO_RESTRICT dominant_anchor_id_out,
    float* GCSO_RESTRICT effective_ripa_rad_out) GCSO_NOEXCEPT;

/**
 * @brief Executes Online Toroidal k-Means / DP-Means E+M step over K crystallized attractor
 * centroids.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_step_online_toroidal_kmeans(
    gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids_inout, size_t num_centroids,
    const gcso_gateway_frame_t* GCSO_RESTRICT frame, float split_coherence_threshold,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout,
    size_t* GCSO_RESTRICT assigned_centroid_idx_out, float* GCSO_RESTRICT assigned_coherence_out,
    float* GCSO_RESTRICT updated_gravity_mass_out) GCSO_NOEXCEPT;

/**
 * @brief Executes Batched Online Toroidal k-Means / DP-Means macro clustering (O(B * K)) with
 * deferred IEEE 802.3 CRC32 stamping.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_attractor_step_online_toroidal_kmeans_batch(
    gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids_inout, size_t num_centroids,
    const gcso_gateway_frame_t* GCSO_RESTRICT frames, size_t num_frames,
    float split_coherence_threshold, gcso_pointer_trail_t* GCSO_RESTRICT trail_inout,
    size_t* GCSO_RESTRICT last_assigned_idx_out, uint32_t* GCSO_RESTRICT last_anchor_id_out,
    float* GCSO_RESTRICT mean_batch_coherence_out,
    uint32_t* GCSO_RESTRICT spawned_clusters_out) GCSO_NOEXCEPT;

/**
 * @brief Executes Coarse-to-Fine macro-to-micro centroid routing and fused QK RIPA phase steering.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_action_hub_route_coarse_to_fine_and_steer_qk_safe(
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout,
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT centroids, size_t num_centroids,
    uint8_t pull_step_u7, float min_step_rad, float* GCSO_RESTRICT query_tensor,
    float* GCSO_RESTRICT key_tensor, size_t head_dim, size_t num_heads,
    size_t* GCSO_RESTRICT winning_centroid_idx_out, uint32_t* GCSO_RESTRICT selected_anchor_id_out,
    float* GCSO_RESTRICT centroid_coherence_out,
    float* GCSO_RESTRICT gravity_pull_out) GCSO_NOEXCEPT;

/**
 * @brief Serializes num_records Section-1 CrystallizedAttractorRecords into a .gcso (ZIMMS) byte
 * buffer with IEEE 802.3 CRC32 stamping.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_container_write_crystallized_records_zero_copy(
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT records, size_t num_records,
    uint8_t* GCSO_RESTRICT out_bytes, size_t* GCSO_RESTRICT buffer_size_inout) GCSO_NOEXCEPT;

/**
 * @brief Validates and zero-copy maps num_records Section-1 CrystallizedAttractorRecords from a
 * 64-byte aligned .gcso (ZIMMS) byte buffer.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_container_verify_crystallized_records_zero_copy(
    const uint8_t* GCSO_RESTRICT zimms_bytes, size_t byte_len, size_t num_records,
    const gcso_crystallized_attractor_record_t** GCSO_RESTRICT records_ptr_out) GCSO_NOEXCEPT;

/**
 * @brief Serializes a 4-section .gcso unified binary container including Section 3c
 * CausalProvenanceReport diagnostic history entries.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_container_write_four_section_with_reports_zero_copy(
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT crystals, size_t num_crystals,
    const gcso_pointer_trail_t* GCSO_RESTRICT trails, size_t num_trails,
    const gcso_contrastive_bloom_filter_t* GCSO_RESTRICT bloom_opt,
    const gcso_daes_slot_t* GCSO_RESTRICT daes_opt,
    const gcso_causal_provenance_report_t* GCSO_RESTRICT reports, size_t num_reports,
    uint64_t timestamp_epoch_sec, uint8_t* GCSO_RESTRICT out_bytes,
    size_t* GCSO_RESTRICT buffer_size_inout) GCSO_NOEXCEPT;

/**
 * @brief Serializes a 4-section .gcso unified binary container in zero-copy aligned layout.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_container_write_four_section_zero_copy(
    const gcso_crystallized_attractor_record_t* GCSO_RESTRICT crystals, size_t num_crystals,
    const gcso_pointer_trail_t* GCSO_RESTRICT trails, size_t num_trails,
    const gcso_contrastive_bloom_filter_t* GCSO_RESTRICT bloom_opt,
    const gcso_daes_slot_t* GCSO_RESTRICT daes_opt, uint64_t timestamp_epoch_sec,
    uint8_t* GCSO_RESTRICT out_bytes, size_t* GCSO_RESTRICT buffer_size_inout) GCSO_NOEXCEPT;

/**
 * @brief Validates and zero-copy maps a 128B-aligned 4-section .gcso container including Section 0b
 * gcso_zimms_descriptor_t and Section 3c CausalProvenanceReports.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_container_verify_four_section_with_reports_zero_copy(
    const uint8_t* GCSO_RESTRICT zimms_bytes, size_t byte_len,
    gcso_zimms_descriptor_t* GCSO_RESTRICT zimms_desc_out,
    size_t* GCSO_RESTRICT num_crystals_out, size_t* GCSO_RESTRICT num_trails_out,
    size_t* GCSO_RESTRICT num_reports_out,
    const gcso_crystallized_attractor_record_t** GCSO_RESTRICT crystals_ptr_out,
    const gcso_pointer_trail_t** GCSO_RESTRICT trails_ptr_out,
    const gcso_contrastive_bloom_filter_t** GCSO_RESTRICT bloom_ptr_out,
    const gcso_daes_slot_t** GCSO_RESTRICT daes_ptr_out,
    const gcso_causal_provenance_report_t** GCSO_RESTRICT reports_ptr_out) GCSO_NOEXCEPT;

/**
 * @brief Verifies and zero-copy maps a 4-section .gcso unified binary container.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_container_verify_four_section_zero_copy(
    const uint8_t* GCSO_RESTRICT zimms_bytes, size_t byte_len,
    size_t* GCSO_RESTRICT num_crystals_out, size_t* GCSO_RESTRICT num_trails_out,
    const gcso_crystallized_attractor_record_t** GCSO_RESTRICT crystals_ptr_out,
    const gcso_pointer_trail_t** GCSO_RESTRICT trails_ptr_out) GCSO_NOEXCEPT;

/**
 * @brief Maps a 128B-aligned 4-section .gcso container in-place, folds Section 1 centroids onto
 * trail_inout, applies Section 3a Bloom repulsion if active, and steers Query/Key tensors.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_container_map_and_steer_four_section_zero_copy_safe(
    const uint8_t* GCSO_RESTRICT zimms_bytes, size_t byte_len,
    gcso_pointer_trail_t* GCSO_RESTRICT trail_inout, float min_step_rad,
    float* GCSO_RESTRICT query_tensor, float* GCSO_RESTRICT key_tensor, size_t head_dim,
    size_t num_heads, size_t* GCSO_RESTRICT num_crystals_out, float* GCSO_RESTRICT total_pull_out,
    uint32_t* GCSO_RESTRICT dominant_anchor_id_out,
    float* GCSO_RESTRICT effective_ripa_rad_out) GCSO_NOEXCEPT;

/**
 * @brief Maps a 128B-aligned 4-section .gcso container in-place, executes Coarse-to-Fine routing
 * (O(K+M)) over Section 1 centroids and Section 2 trails, applies Section 3a Bloom repulsion, and
 * steers Query/Key tensors.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_container_map_route_coarse_to_fine_and_steer_four_section_zero_copy_safe(
    const uint8_t* GCSO_RESTRICT zimms_bytes, size_t byte_len,
    gcso_pointer_trail_t* GCSO_RESTRICT src_trail_inout, size_t head_dim, size_t num_heads,
    float phase_interference_gain, float stigmergic_prior_gain, float pointer_link_bonus,
    float cluster_bonus_gain, float contradiction_threshold, uint8_t interpolation_u7,
    uint8_t bloom_repulsion_threshold, float min_step_rad, float* GCSO_RESTRICT query_tensor,
    float* GCSO_RESTRICT key_tensor, size_t* GCSO_RESTRICT winning_centroid_idx_out,
    uint32_t* GCSO_RESTRICT selected_trail_id_out, float* GCSO_RESTRICT best_score_out,
    float* GCSO_RESTRICT best_coherence_out, uint8_t* GCSO_RESTRICT bloom_repulsed_out,
    uint8_t* GCSO_RESTRICT found_out) GCSO_NOEXCEPT;

GCSO_EXTERN_C_END

#endif // LIMINIKA_GCSO_ABI_H