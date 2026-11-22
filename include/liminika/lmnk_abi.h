// name: include/liminika/lmnk_abi.h
// SPDX-License-Identifier: MIT OR Apache-2.0

/**
 * @file lmnk_abi.h
 * @brief LiminiKa Core Engine C-ABI v0.2.0 (`0x00000200`) Public Functional Interface.
 *
 * Exposes exact 1:1 C11/C++20 bindings for `liminika-core` (`runtime/abi`):
 * - Version, capability, and configuration validation
 * - Zero-heap in-place and boxed `RuntimeContext` lifecycle
 * - Zero-copy 4-section `.lmnk` container write/verify/steer operations
 * - RoPE phase steering, sub-head routing, Rank-1 / LoRA projections, and gateway synthesis
 * - Stigmergic substrate orchestration, Bloom filter, FastPath ledger, and paged bit-tree reduction
 * - Online toroidal k-Means / DP-Means (`"LNKR"` `lmnk_phase_anchor_record_t`), entropy control,
 *   epistemic horizon, 8-granule lacuna partition, Ego intervention, and zero-alloc Mermaid output.
 */

#ifndef LIMINIKA_LMNK_ABI_H
#define LIMINIKA_LMNK_ABI_H

#include "lmnk_types.h"

LMNK_EXTERN_C_BEGIN

/* -------------------------------------------------------------------------- */
/* 1. Version, Capability & Configuration Management                          */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API uint32_t LMNK_CALL lmnk_abi_version_major(void) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API uint32_t LMNK_CALL lmnk_abi_version_minor(void) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API uint32_t LMNK_CALL lmnk_abi_version_patch(void) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API uint32_t LMNK_CALL lmnk_abi_version_hex(void) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API const char* LMNK_CALL lmnk_abi_version_string(void) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API const char* LMNK_CALL lmnk_status_to_string(
    lmnk_status_t status) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_abi_query_capability(
    uint64_t* LMNK_RESTRICT flags_out) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_config_init_default(
    lmnk_config_t* LMNK_RESTRICT config) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_config_validate(
    const lmnk_config_t* LMNK_RESTRICT config) LMNK_NOEXCEPT;

/* -------------------------------------------------------------------------- */
/* 2. Zero-Heap In-Place & Boxed RuntimeContext Lifecycle                     */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API size_t LMNK_CALL lmnk_context_size(void) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API size_t LMNK_CALL lmnk_context_align(void) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_init_in_place(
    const lmnk_config_t* LMNK_RESTRICT config,
    uint8_t* LMNK_RESTRICT mem_out,
    size_t mem_bytes) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_deinit_in_place(
    lmnk_context_handle_t context) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_create(
    const lmnk_config_t* LMNK_RESTRICT config,
    lmnk_context_handle_t* LMNK_RESTRICT context_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_destroy(
    lmnk_context_handle_t context) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_reset(
    lmnk_context_handle_t context) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_step_token(
    lmnk_context_handle_t context,
    uint32_t token_id,
    float* LMNK_RESTRICT query_tensor,
    float* LMNK_RESTRICT key_tensor,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_serialize(
    lmnk_context_handle_t context,
    uint8_t* LMNK_RESTRICT buffer,
    size_t* LMNK_RESTRICT buffer_size) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_deserialize(
    const uint8_t* LMNK_RESTRICT buffer,
    size_t buffer_size,
    lmnk_context_handle_t* LMNK_RESTRICT context_out) LMNK_NOEXCEPT;

/* -------------------------------------------------------------------------- */
/* 3. Tagged Pointer Packing & Mode Evolution                                 */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API uint64_t LMNK_CALL lmnk_tagged_ptr_pack(
    uint64_t raw_addr,
    uint8_t tag,
    lmnk_q7_t phase) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API uint64_t LMNK_CALL lmnk_tagged_ptr_unpack_addr(uint64_t ptr) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API uint8_t LMNK_CALL lmnk_tagged_ptr_unpack_tag(uint64_t ptr) LMNK_NOEXCEPT;
LMNK_NODISCARD LMNK_API lmnk_q7_t LMNK_CALL lmnk_tagged_ptr_unpack_phase(uint64_t ptr) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API uint64_t LMNK_CALL lmnk_tagged_ptr_pack_action(
    uint64_t raw_addr,
    uint8_t attractor_slot,
    uint8_t action_mode_flags,
    lmnk_q7_t phase) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API uint8_t LMNK_CALL lmnk_tagged_ptr_unpack_action_mode(
    uint64_t ptr) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API uint8_t LMNK_CALL lmnk_tagged_ptr_unpack_attractor_slot(
    uint64_t ptr) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API uint64_t LMNK_CALL lmnk_tagged_ptr_evolve_mode(
    uint64_t ptr,
    float stigmergic_density,
    float entropy_h_tilde,
    uint8_t repulsion_triggered,
    uint8_t dominant_attractor_slot,
    float low_entropy_skip_thresh,
    float high_density_rank1_thresh) LMNK_NOEXCEPT;

/* -------------------------------------------------------------------------- */
/* 4. Zero-Copy 4-Section Container Serialization, Verification & Steering    */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_container_write_four_section_with_reports_zero_copy(
    const lmnk_phase_anchor_record_t* LMNK_RESTRICT anchors,
    size_t num_anchors,
    const lmnk_pointer_trail_t* LMNK_RESTRICT trails,
    size_t num_trails,
    const lmnk_bloom_filter_t* LMNK_RESTRICT bloom_opt,
    const lmnk_ledger_slot_t* LMNK_RESTRICT ledger_opt,
    const lmnk_causal_report_t* LMNK_RESTRICT reports,
    size_t num_reports,
    uint64_t timestamp_epoch_sec,
    uint8_t* LMNK_RESTRICT out_bytes,
    size_t* LMNK_RESTRICT buffer_size_inout) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_container_write_four_section_zero_copy(
    const lmnk_phase_anchor_record_t* LMNK_RESTRICT anchors,
    size_t num_anchors,
    const lmnk_pointer_trail_t* LMNK_RESTRICT trails,
    size_t num_trails,
    const lmnk_bloom_filter_t* LMNK_RESTRICT bloom_opt,
    const lmnk_ledger_slot_t* LMNK_RESTRICT ledger_opt,
    uint64_t timestamp_epoch_sec,
    uint8_t* LMNK_RESTRICT out_bytes,
    size_t* LMNK_RESTRICT buffer_size_inout) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_container_verify_four_section_with_reports_zero_copy(
    const uint8_t* LMNK_RESTRICT container_bytes,
    size_t byte_len,
    size_t* LMNK_RESTRICT num_anchors_out,
    size_t* LMNK_RESTRICT num_trails_out,
    size_t* LMNK_RESTRICT num_reports_out,
    const lmnk_phase_anchor_record_t** LMNK_RESTRICT anchors_ptr_out,
    const lmnk_pointer_trail_t** LMNK_RESTRICT trails_ptr_out,
    const lmnk_bloom_filter_t** LMNK_RESTRICT bloom_ptr_out,
    const lmnk_ledger_slot_t** LMNK_RESTRICT ledger_ptr_out,
    const lmnk_causal_report_t** LMNK_RESTRICT reports_ptr_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_container_verify_four_section_zero_copy(
    const uint8_t* LMNK_RESTRICT container_bytes,
    size_t byte_len,
    size_t* LMNK_RESTRICT num_anchors_out,
    size_t* LMNK_RESTRICT num_trails_out,
    const lmnk_phase_anchor_record_t** LMNK_RESTRICT anchors_ptr_out,
    const lmnk_pointer_trail_t** LMNK_RESTRICT trails_ptr_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_container_map_and_steer_four_section_zero_copy_safe(
    const uint8_t* LMNK_RESTRICT container_bytes,
    size_t byte_len,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_inout,
    float min_step_rad,
    float* LMNK_RESTRICT query_tensor,
    float* LMNK_RESTRICT key_tensor,
    size_t head_dim,
    size_t num_heads,
    size_t* LMNK_RESTRICT num_anchors_out,
    float* LMNK_RESTRICT total_pull_out,
    uint32_t* LMNK_RESTRICT dominant_anchor_id_out,
    float* LMNK_RESTRICT effective_phase_clamp_rad_out) LMNK_NOEXCEPT;

/* -------------------------------------------------------------------------- */
/* 5. RoPE Phase Steering, Gateway Synthesis, Subhead & Rank-1 Projections    */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_create(
    uint32_t head_dim,
    uint32_t num_heads,
    lmnk_phase_steering_handle_t* LMNK_RESTRICT kernel_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_apply(
    float* LMNK_RESTRICT query_tensor,
    const lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_apply_fused(
    float* LMNK_RESTRICT query_tensor,
    float* LMNK_RESTRICT key_tensor,
    const lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_apply_safe(
    float* LMNK_RESTRICT query_tensor,
    const lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads,
    float max_rad) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_apply_with_config_safe(
    float* LMNK_RESTRICT query_tensor,
    const lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    const lmnk_config_t* LMNK_RESTRICT config) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_phase_steering_steer_with_gateway_and_config_safe(
    float* LMNK_RESTRICT query_tensor,
    const lmnk_q7_t* LMNK_RESTRICT fact_phases,
    float fact_energy,
    const lmnk_q7_t* LMNK_RESTRICT persona_phases,
    float persona_energy,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_inout,
    const lmnk_config_t* LMNK_RESTRICT config) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_phase_steering_compute_procrustes_half_split(
    const float* LMNK_RESTRICT q_before,
    const float* LMNK_RESTRICT q_after,
    size_t head_dim,
    size_t rotary_dim,
    size_t num_heads,
    lmnk_q7_t* LMNK_RESTRICT q7_phase_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_apply_fused_safe(
    float* LMNK_RESTRICT query_tensor,
    float* LMNK_RESTRICT key_tensor,
    const lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads,
    float max_rad) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_apply_geodesic_safe(
    float* LMNK_RESTRICT query_tensor,
    const lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads,
    float max_rad) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_apply_pointer_gated_safe(
    uint64_t tagged_ptr,
    float* LMNK_RESTRICT query_tensor,
    const lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads,
    float max_rad) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_phase_steering_apply_pointer_gated_fused_safe(
    uint64_t tagged_ptr,
    float* LMNK_RESTRICT query_tensor,
    float* LMNK_RESTRICT key_tensor,
    const lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    size_t head_dim,
    size_t num_heads,
    float max_rad) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_gateway_project_bytes_to_q7_64(
    const uint8_t* LMNK_RESTRICT bytes,
    size_t byte_len,
    lmnk_q7_t* LMNK_RESTRICT phases_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_gateway_project_f32_to_q7_64(
    const float* LMNK_RESTRICT values,
    size_t num_values,
    float gain,
    lmnk_q7_t* LMNK_RESTRICT phases_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_gateway_synthesize_multi_origin_frame(
    float vram_pressure,
    float gpu_frame_overrun,
    uint8_t active_swarm_cell_mask,
    uint8_t recommend_layer_skip,
    const lmnk_q7_t* LMNK_RESTRICT fact_phases,
    float fact_confidence,
    uint64_t fact_mapped_offset48,
    uint32_t fact_anchor_id,
    const lmnk_q7_t* LMNK_RESTRICT persona_phases,
    float persona_strength,
    uint64_t persona_mapped_offset48,
    uint32_t persona_anchor_id,
    const lmnk_bloom_filter_t* LMNK_RESTRICT bloom_filter_opt,
    float contradiction_threshold,
    uint8_t bloom_repulsion_threshold,
    lmnk_gateway_frame_t* LMNK_RESTRICT frame_out,
    lmnk_causal_report_t* LMNK_RESTRICT report_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_steer_gateway_frame_safe(
    const lmnk_gateway_frame_t* LMNK_RESTRICT frame,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_inout,
    uint8_t unresolved_damping_u7,
    float* LMNK_RESTRICT query_tensor,
    float* LMNK_RESTRICT key_tensor,
    size_t head_dim,
    size_t num_heads,
    float* LMNK_RESTRICT logits_inout,
    size_t vocab_size,
    lmnk_causal_report_t* LMNK_RESTRICT report_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_gateway_project_bytes_synthesize_kmeans_and_steer_qk_safe(
    float vram_pressure,
    float gpu_frame_overrun,
    uint8_t active_swarm_cell_mask,
    uint8_t recommend_layer_skip,
    const uint8_t* LMNK_RESTRICT fact_bytes,
    size_t fact_byte_len,
    float fact_confidence,
    uint64_t fact_mapped_offset48,
    uint32_t fact_anchor_id,
    const uint8_t* LMNK_RESTRICT persona_bytes,
    size_t persona_byte_len,
    float persona_strength,
    uint64_t persona_mapped_offset48,
    uint32_t persona_anchor_id,
    const lmnk_bloom_filter_t* LMNK_RESTRICT bloom_filter_opt,
    float contradiction_threshold,
    uint8_t bloom_repulsion_threshold,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_inout,
    uint8_t unresolved_damping_u7,
    float* LMNK_RESTRICT query_tensor,
    float* LMNK_RESTRICT key_tensor,
    size_t head_dim,
    size_t num_heads,
    float* LMNK_RESTRICT logits_inout,
    size_t vocab_size,
    lmnk_phase_anchor_record_t* LMNK_RESTRICT centroids_inout,
    size_t num_centroids,
    float split_coherence_threshold,
    lmnk_gateway_frame_t* LMNK_RESTRICT frame_out,
    lmnk_causal_report_t* LMNK_RESTRICT report_out,
    size_t* LMNK_RESTRICT assigned_centroid_idx_out,
    float* LMNK_RESTRICT assigned_coherence_out,
    float* LMNK_RESTRICT updated_gravity_mass_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_gateway_project_f32_synthesize_kmeans_and_steer_qk_safe(
    float vram_pressure,
    float gpu_frame_overrun,
    uint8_t active_swarm_cell_mask,
    uint8_t recommend_layer_skip,
    const float* LMNK_RESTRICT fact_values,
    size_t fact_len,
    float fact_projection_gain,
    float fact_confidence,
    uint64_t fact_mapped_offset48,
    uint32_t fact_anchor_id,
    const float* LMNK_RESTRICT persona_values,
    size_t persona_len,
    float persona_projection_gain,
    float persona_strength,
    uint64_t persona_mapped_offset48,
    uint32_t persona_anchor_id,
    const lmnk_bloom_filter_t* LMNK_RESTRICT bloom_filter_opt,
    float contradiction_threshold,
    uint8_t bloom_repulsion_threshold,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_inout,
    uint8_t unresolved_damping_u7,
    float* LMNK_RESTRICT query_tensor,
    float* LMNK_RESTRICT key_tensor,
    size_t head_dim,
    size_t num_heads,
    float* LMNK_RESTRICT logits_inout,
    size_t vocab_size,
    lmnk_phase_anchor_record_t* LMNK_RESTRICT centroids_inout,
    size_t num_centroids,
    float split_coherence_threshold,
    lmnk_gateway_frame_t* LMNK_RESTRICT frame_out,
    lmnk_causal_report_t* LMNK_RESTRICT report_out,
    size_t* LMNK_RESTRICT assigned_centroid_idx_out,
    float* LMNK_RESTRICT assigned_coherence_out,
    float* LMNK_RESTRICT updated_gravity_mass_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_apply_soft_damping(
    float* LMNK_RESTRICT query_tensor,
    size_t head_dim,
    size_t num_heads,
    float damping_factor) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_step_filter(
    lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    size_t len,
    float min_step_rad) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_lazy_unwrap_override(
    float* LMNK_RESTRICT query_tensor,
    const float* LMNK_RESTRICT context_accum,
    size_t head_dim,
    size_t num_heads) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_slerp_norm_guard_stable(
    float* LMNK_RESTRICT tensor,
    size_t dim,
    float norm_lower,
    float norm_upper) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_fused_logit_shift(
    float* LMNK_RESTRICT logits,
    size_t vocab_size,
    const lmnk_q7_t* LMNK_RESTRICT phase_deltas,
    size_t num_heads) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_compute_procrustes_delta(
    const float* LMNK_RESTRICT source,
    const float* LMNK_RESTRICT target,
    size_t dim,
    lmnk_q7_t* LMNK_RESTRICT phase_out) LMNK_NOEXCEPT;

LMNK_API lmnk_status_t LMNK_CALL lmnk_phase_steering_destroy(
    lmnk_phase_steering_handle_t kernel) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_subhead_dispatch_single_pass(
    float* LMNK_RESTRICT query_tensor,
    const lmnk_subhead_config_t* LMNK_RESTRICT subhead_cfg,
    size_t head_dim) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_rank1_eval(
    float* LMNK_RESTRICT y_out,
    const float* LMNK_RESTRICT x_in,
    const lmnk_rank1_descriptor_t* LMNK_RESTRICT rank1_desc,
    size_t dim_in,
    size_t dim_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_rank1_eval_fused(
    float* LMNK_RESTRICT y_inout,
    const float* LMNK_RESTRICT x_in,
    const lmnk_rank1_descriptor_t* LMNK_RESTRICT rank1_desc,
    size_t dim_in,
    size_t dim_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_lora_project(
    const float* LMNK_RESTRICT lora_a,
    const float* LMNK_RESTRICT lora_b,
    size_t rank,
    size_t dim_in,
    size_t dim_out,
    lmnk_rank1_descriptor_t* LMNK_RESTRICT rank1_out,
    lmnk_q7_t* LMNK_RESTRICT phase_profile_out) LMNK_NOEXCEPT;

/* -------------------------------------------------------------------------- */
/* 6. Stigmergic Substrate, Swarm Trail Resolution & FastPath Ledger          */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_create(
    uint32_t capacity,
    lmnk_substrate_handle_t* LMNK_RESTRICT hub_out) LMNK_NOEXCEPT;

LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_destroy(
    lmnk_substrate_handle_t handle) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_init_default_thresholds(
    lmnk_stigmergic_thresholds_t* LMNK_RESTRICT thresholds_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_evaluate_thresholds(
    const lmnk_stigmergic_thresholds_t* LMNK_RESTRICT thresholds_opt,
    const lmnk_pointer_trail_t* LMNK_RESTRICT trail,
    float entropy_h_tilde,
    float phase_coherence,
    float transition_log_prob,
    uint32_t selected_trail_id,
    uint8_t bloom_repulsion_density,
    uint8_t dominant_slot,
    lmnk_substrate_step_report_t* LMNK_RESTRICT report_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_step_safe(
    lmnk_substrate_handle_t hub_opt,
    const lmnk_stigmergic_thresholds_t* LMNK_RESTRICT thresholds_opt,
    uint64_t input_ptr,
    float entropy_h_tilde,
    uint8_t dominant_slot,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_inout,
    float* LMNK_RESTRICT attention_scores_inout,
    size_t num_scores,
    float* LMNK_RESTRICT query_tensor,
    float* LMNK_RESTRICT key_tensor,
    size_t head_dim,
    size_t num_heads,
    float* LMNK_RESTRICT logits_inout,
    size_t vocab_size,
    lmnk_substrate_step_report_t* LMNK_RESTRICT report_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_step_pointer(
    lmnk_context_handle_t context,
    uint64_t current_ptr,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_resolve_trail_in_swarm(
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_inout,
    uint64_t action_ptr,
    const lmnk_q7_t* LMNK_RESTRICT target_phases,
    uint8_t active_cell_mask,
    uint8_t step_gain_u7,
    uint8_t min_step_q7,
    uint8_t tolerance_q7,
    uint8_t quorum_threshold,
    float entropy_h_tilde,
    float low_entropy_skip_thresh,
    lmnk_reconstitution_report_t* LMNK_RESTRICT report_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API uint32_t LMNK_CALL lmnk_substrate_hash_slot256_index(
    uint64_t ptr) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_pull_trail_to_anchor(
    lmnk_context_handle_t context,
    uint32_t trail_id,
    uint32_t anchor_id,
    float pull_force) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_link_hallucinated_trails(
    lmnk_context_handle_t context,
    uint32_t src_trail_id,
    uint32_t dst_trail_id) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_reduce_bit_tree(
    const lmnk_paged_bitmask_t* LMNK_RESTRICT bitmasks,
    size_t num_masks,
    lmnk_paged_bitmask_t* LMNK_RESTRICT reduced_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_substrate_paged_block_warp_bitmask(
    const uint64_t* LMNK_RESTRICT kv_bits,
    size_t num_blocks,
    lmnk_paged_bitmask_t* LMNK_RESTRICT mask_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_ledger_reset_telemetry(
    lmnk_ledger_slot_t* LMNK_RESTRICT slot) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_ledger_fast_path_lookup(
    const lmnk_ledger_slot_t* LMNK_RESTRICT slot,
    uint64_t input_key,
    uint64_t* LMNK_RESTRICT shortcut_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_ledger_telemetry_push(
    lmnk_ledger_slot_t* LMNK_RESTRICT slot,
    uint8_t metric_code) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_ledger_set_mode(
    lmnk_ledger_slot_t* LMNK_RESTRICT slot,
    uint32_t mode) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_ledger_evaluate_auto_tune(
    const lmnk_ledger_slot_t* LMNK_RESTRICT slot,
    lmnk_config_t* LMNK_RESTRICT config_out) LMNK_NOEXCEPT;

/* -------------------------------------------------------------------------- */
/* 7. Toroidal Anchor Clustering, Entropy, Horizon, Stack, Ego & Topology     */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_anchor_step_online_toroidal_kmeans(
    lmnk_phase_anchor_record_t* LMNK_RESTRICT centroids_inout,
    size_t num_centroids,
    const lmnk_gateway_frame_t* LMNK_RESTRICT frame,
    float split_coherence_threshold,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_inout,
    size_t* LMNK_RESTRICT assigned_centroid_idx_out,
    float* LMNK_RESTRICT assigned_coherence_out,
    float* LMNK_RESTRICT updated_gravity_mass_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_anchor_step_online_toroidal_kmeans_batch(
    lmnk_phase_anchor_record_t* LMNK_RESTRICT centroids_inout,
    size_t num_centroids,
    const lmnk_gateway_frame_t* LMNK_RESTRICT frames,
    size_t num_frames,
    float split_coherence_threshold,
    lmnk_pointer_trail_t* LMNK_RESTRICT trail_inout,
    size_t* LMNK_RESTRICT last_assigned_idx_out,
    uint32_t* LMNK_RESTRICT last_anchor_id_out,
    float* LMNK_RESTRICT mean_batch_coherence_out,
    uint32_t* LMNK_RESTRICT spawned_clusters_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_entropy_controller_create(
    const lmnk_entropy_state_t* LMNK_RESTRICT initial_state,
    lmnk_entropy_controller_handle_t* LMNK_RESTRICT controller_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_entropy_eval_stateful(
    lmnk_entropy_controller_handle_t controller,
    float token_z_score,
    lmnk_entropy_state_t* LMNK_RESTRICT state_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_liminal_eval_dyadic128(
    const float* LMNK_RESTRICT key_vector,
    size_t dim,
    float* LMNK_RESTRICT lacuna_score_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_liminal_eval_barrier(
    float lacuna_score,
    float tau_eff,
    float* LMNK_RESTRICT barrier_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_liminal_eval_barrier_pitchfork(
    float liminal_score,
    float tau_eff,
    float alpha,
    float beta,
    uint32_t head_dim,
    float* LMNK_RESTRICT barrier_out) LMNK_NOEXCEPT;

LMNK_API lmnk_status_t LMNK_CALL lmnk_entropy_controller_destroy(
    lmnk_entropy_controller_handle_t controller) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_epistemic_evaluate_horizon(
    const lmnk_phase_anchor_record_t* LMNK_RESTRICT records,
    size_t num_records,
    const lmnk_q7_t* LMNK_RESTRICT query_phases,
    float horizon_floor,
    lmnk_epistemic_horizon_t* LMNK_RESTRICT reading_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_stigmergic_evaluate_overlap_and_discovery(
    lmnk_phase_anchor_record_t* LMNK_RESTRICT centroids_inout,
    size_t num_centroids,
    const lmnk_q7_t* LMNK_RESTRICT query_phases,
    const lmnk_q7_t* LMNK_RESTRICT drifted_phases,
    float aperture_floor,
    uint8_t discovery_floor_q7,
    uint8_t nucleate_on_discovery,
    uint32_t new_anchor_id,
    lmnk_overlap_discovery_report_t* LMNK_RESTRICT report_out,
    uint8_t* LMNK_RESTRICT did_nucleate_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_core_stack_init_default(
    lmnk_core_state_stack_t* LMNK_RESTRICT stack_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_core_stack_adapt(
    lmnk_core_state_stack_t* LMNK_RESTRICT stack_inout,
    const lmnk_cognitive_ledger_t* LMNK_RESTRICT ledger,
    const lmnk_epistemic_horizon_t* LMNK_RESTRICT horizon,
    const lmnk_overlap_discovery_report_t* LMNK_RESTRICT overlap_report) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_topology_format_mermaid(
    const lmnk_core_state_stack_t* LMNK_RESTRICT stack,
    const lmnk_epistemic_horizon_t* LMNK_RESTRICT horizon,
    const lmnk_overlap_discovery_report_t* LMNK_RESTRICT overlap_report,
    uint8_t active_anchors,
    uint8_t spawned_daughters,
    uint8_t* LMNK_RESTRICT out_buf,
    size_t buf_cap,
    size_t* LMNK_RESTRICT bytes_written_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_ego_state_init_preset(
    uint8_t preset_code,
    lmnk_liminal_ego_state_t* LMNK_RESTRICT ego_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_ego_state_derive_from_text(
    const uint8_t* LMNK_RESTRICT utf8_ptr,
    size_t utf8_len,
    lmnk_liminal_ego_state_t* LMNK_RESTRICT ego_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_ego_state_intervene_frame(
    lmnk_liminal_ego_state_t* LMNK_RESTRICT ego_inout,
    lmnk_core_state_stack_t* LMNK_RESTRICT stack_inout,
    const lmnk_boundary_delta_t* LMNK_RESTRICT delta,
    const lmnk_epistemic_horizon_t* LMNK_RESTRICT horizon,
    const lmnk_overlap_discovery_report_t* LMNK_RESTRICT overlap_report,
    lmnk_gateway_frame_t* LMNK_RESTRICT frame_inout) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_liminal_evaluate_lacuna_partition(
    uint8_t self_mask,
    const uint8_t* LMNK_RESTRICT other_granule_density_q7,
    const uint8_t* LMNK_RESTRICT granule_noise_tolerance_q7,
    uint8_t concentration_floor_q7,
    lmnk_lacuna_partition_t* LMNK_RESTRICT partition_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_distortion_tension_step_hot_path(
    lmnk_distortion_tension_t* LMNK_RESTRICT tension_inout,
    uint8_t ego_delta_norm_q7,
    uint8_t boundary_sensitivity_q7,
    uint8_t exit_entropy_q7,
    uint8_t guard_proximity_q7,
    uint8_t active_overdrive) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_distortion_delta_apply_chain(
    const lmnk_distortion_delta_t* LMNK_RESTRICT nodes,
    size_t num_nodes,
    const lmnk_liminal_ego_state_t* LMNK_RESTRICT ego,
    const lmnk_spine_state_t* LMNK_RESTRICT spine,
    const lmnk_lacuna_partition_t* LMNK_RESTRICT partition,
    const lmnk_stigmergic_granule_state_t* LMNK_RESTRICT stigmergy,
    const lmnk_distortion_tension_t* LMNK_RESTRICT tension,
    uint8_t max_hops,
    lmnk_gateway_frame_t* LMNK_RESTRICT frame_inout,
    uint8_t* LMNK_RESTRICT delta_norm_out,
    uint8_t* LMNK_RESTRICT hops_traversed_out) LMNK_NOEXCEPT;

LMNK_EXTERN_C_END

#endif /* LIMINIKA_LMNK_ABI_H */
