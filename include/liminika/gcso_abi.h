// name: include/liminika/gcso_abi.h
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_ABI_H
#define LIMINIKA_GCSO_ABI_H

#include "liminika/gcso_types.h"

GCSO_EXTERN_C_BEGIN

<<<<<<< HEAD
// ===================================================================
// 1. System & Capability Query Interface
// ===================================================================

/**
 * @brief Retrieves the numeric version of the GCSO C-ABI.
 * @param major Pointer to store the major version number.
 * @param minor Pointer to store the minor version number.
 * @param patch Pointer to store the patch version number.
 */
<<<<<<< HEAD
GCSO_API void GCSO_CALL gcso_abi_get_version(
    uint32_t* GCSO_RESTRICT major,
    uint32_t* GCSO_RESTRICT minor,
    uint32_t* GCSO_RESTRICT patch
) GCSO_NOEXCEPT;
=======
// System & Capability Query Interface
GCSO_API void GCSO_CALL gcso_abi_get_version(uint32_t* GCSO_RESTRICT major,
                                             uint32_t* GCSO_RESTRICT minor,
                                             uint32_t* GCSO_RESTRICT patch) GCSO_NOEXCEPT;
>>>>>>> 631dd2990e7e914074e1e2e891e7b8af8ac59c1c
=======
GCSO_API void GCSO_CALL gcso_abi_get_version(uint32_t* GCSO_RESTRICT major,
                                             uint32_t* GCSO_RESTRICT minor,
                                             uint32_t* GCSO_RESTRICT patch) GCSO_NOEXCEPT;
>>>>>>> 073c96b9ba43a51e79e76a2ea8e74feffeea69ee

/**
 * @brief Returns the static semantic version string literal for the GCSO kernel engine.
 * @return Const pointer to null-terminated version string.
 */
GCSO_NODISCARD GCSO_API const char* GCSO_CALL gcso_abi_get_version_string(void) GCSO_NOEXCEPT;

<<<<<<< HEAD
/**
 * @brief Converts a gcso_status_t error code into its corresponding literal string representation.
 * @param status Status code enum value.
 * @return Const pointer to descriptive static string representation.
 */
GCSO_NODISCARD GCSO_API const char* GCSO_CALL gcso_status_to_string(gcso_status_t status)
    GCSO_NOEXCEPT;

/**
 * @brief Queries current execution hardware capabilities and available compute backends.
 * @param flags Pointer to store 64-bit bitmask capability flags.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_abi_query_capability(gcso_capability_flags_t* GCSO_RESTRICT flags) GCSO_NOEXCEPT;

/**
 * @brief Initializes a gcso_config_t structure with standard hardware and model defaults.
 * @param config Pointer to the user-allocated configuration descriptor to populate.
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_config_init_default(gcso_config_t* GCSO_RESTRICT config)
    GCSO_NOEXCEPT;

/**
 * @brief Safely frees a dynamically allocated string returned by GCSO APIs.
 * @param str Const string pointer to free (safe no-op if NULL).
 */
GCSO_API void GCSO_CALL gcso_free_string(const char* str) GCSO_NOEXCEPT;

// ===================================================================
// 2. High-Level Runtime Context Facade Interface
// ===================================================================

/**
 * @brief Allocates and initializes a new GCSO execution context facade instance.
 * @param config Pointer to initialized runtime configuration structure.
 * @param context_out Pointer to receive the opaque runtime context handle.
 * @return GCSO_SUCCESS or error code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_context_create(const gcso_config_t* GCSO_RESTRICT config,
                    gcso_context_handle_t* GCSO_RESTRICT context_out) GCSO_NOEXCEPT;

/**
 * @brief Resets transient state variables and phase accumulators without freeing allocated tables.
 * @param context Handle to the active runtime context.
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
<<<<<<< HEAD
GCSO_API gcso_status_t GCSO_CALL gcso_context_reset(
    gcso_context_handle_t context
) GCSO_NOEXCEPT;
=======
GCSO_API GCSO_NODISCARD const char* GCSO_CALL gcso_status_to_string(gcso_status_t status)
    GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL
gcso_abi_query_capability(gcso_capability_flags_t* GCSO_RESTRICT flags) GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL gcso_config_init_default(gcso_config_t* GCSO_RESTRICT config)
    GCSO_NOEXCEPT;

// High-Level Runtime Context Facade Interface
GCSO_API gcso_status_t GCSO_CALL
gcso_context_create(const gcso_config_t* GCSO_RESTRICT config,
                    gcso_context_handle_t* GCSO_RESTRICT context_out) GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL gcso_context_reset(gcso_context_handle_t context) GCSO_NOEXCEPT;
>>>>>>> 631dd2990e7e914074e1e2e891e7b8af8ac59c1c
=======
GCSO_API gcso_status_t GCSO_CALL gcso_context_reset(gcso_context_handle_t context) GCSO_NOEXCEPT;
>>>>>>> 073c96b9ba43a51e79e76a2ea8e74feffeea69ee

/**
 * @brief Primary Baseline Endpoint: Projects natural language system prompt text as an Anchor
 * Attractor.
 * @param context Active context handle.
 * @param prompt_text Null-terminated UTF-8 system prompt string.
 * @param weight Attractor pull force scale.
 * @return GCSO_SUCCESS or error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_set_system_prompt_anchor(
    gcso_context_handle_t context, const char* GCSO_RESTRICT prompt_text,
    float weight) GCSO_NOEXCEPT;

/**
 * @brief Executes per-token Hot Path inference step in zero-allocation mode.
 * Applies DPSR phase steering, QDPS step filtering, and updates pointer trails.
 * @param context Active runtime context handle.
 * @param token_id Input token sequence ID.
 * @param query_tensor 32-byte aligned query tensor buffer to modulate in-place.
 * @param key_tensor 32-byte aligned key tensor buffer.
 * @param trail_out Pointer to receive updated 128-byte pointer trail status.
 * @return GCSO_SUCCESS or error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_context_step_token(
    gcso_context_handle_t context, uint32_t token_id, float* GCSO_RESTRICT query_tensor,
    float* GCSO_RESTRICT key_tensor, gcso_pointer_trail_t* GCSO_RESTRICT trail_out) GCSO_NOEXCEPT;

<<<<<<< HEAD
/**
 * @brief Serializes runtime state into a binary .gcso snapshot format.
 * @param context Active runtime context handle.
 * @param buffer Target byte buffer (pass NULL to query required size).
 * @param buffer_size Pointer to buffer size variable (in/out).
 * @return GCSO_SUCCESS or GCSO_ERROR_BUFFER_TOO_SMALL.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_context_serialize(gcso_context_handle_t context, uint8_t* GCSO_RESTRICT buffer,
                       size_t* GCSO_RESTRICT buffer_size) GCSO_NOEXCEPT;

/**
 * @brief Deserializes a binary .gcso snapshot to restore context runtime state.
 * @param buffer Memory-mapped or allocated input snapshot buffer.
 * @param buffer_size Byte size of input snapshot buffer.
 * @param context_out Pointer to receive restored context handle.
 * @return GCSO_SUCCESS or GCSO_ERROR_CONTAINER_CORRUPTED.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_context_deserialize(const uint8_t* GCSO_RESTRICT buffer, size_t buffer_size,
                         gcso_context_handle_t* GCSO_RESTRICT context_out) GCSO_NOEXCEPT;

/**
 * @brief Destroys a GCSO runtime context instance and releases associated memory.
 * @param context Runtime context handle to free (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
<<<<<<< HEAD
GCSO_API gcso_status_t GCSO_CALL gcso_context_destroy(
    gcso_context_handle_t context
) GCSO_NOEXCEPT;
=======
GCSO_API gcso_status_t GCSO_CALL
gcso_context_serialize(gcso_context_handle_t context, uint8_t* GCSO_RESTRICT buffer,
                       size_t* GCSO_RESTRICT buffer_size) GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL
gcso_context_deserialize(const uint8_t* GCSO_RESTRICT buffer, size_t buffer_size,
                         gcso_context_handle_t* GCSO_RESTRICT context_out) GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL gcso_context_destroy(gcso_context_handle_t context) GCSO_NOEXCEPT;
>>>>>>> 631dd2990e7e914074e1e2e891e7b8af8ac59c1c
=======
GCSO_API gcso_status_t GCSO_CALL gcso_context_destroy(gcso_context_handle_t context) GCSO_NOEXCEPT;
>>>>>>> 073c96b9ba43a51e79e76a2ea8e74feffeea69ee

/**
 * @brief Destroys a container handle and releases mapped storage resources.
 * @param container Container handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_container_destroy(gcso_container_handle_t container)
    GCSO_NOEXCEPT;

// ===================================================================
// 3. Action Hub, Stigmergic Pointer Trail & DAES Acceleration Interface
// ===================================================================

/**
 * @brief Allocates an Action Hub (Sidecar Pointer Table) instance.
 * @param capacity Slot capacity count for tagged pointers.
 * @param hub_out Pointer to receive allocated action hub handle.
 * @return GCSO_SUCCESS or error code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_create(
    uint32_t capacity, gcso_action_hub_handle_t* GCSO_RESTRICT hub_out) GCSO_NOEXCEPT;

/**
 * @brief Advances a 64-bit tagged pointer transition in O(1) time within the Action Hub.
 * @param context Active runtime context handle.
 * @param current_ptr Encoded 64-bit tagged pointer address.
 * @param trail_out Pointer to receive updated pointer trail structure.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_action_hub_step_pointer(gcso_context_handle_t context, uint64_t current_ptr,
                             gcso_pointer_trail_t* GCSO_RESTRICT trail_out) GCSO_NOEXCEPT;

/**
 * @brief Computes direct 256-slot hash index for 64-bit pointer trail caching.
 * @param ptr Raw 64-bit pointer address.
 * @return 8-bit hash index (0 to 255).
 */
GCSO_API uint32_t GCSO_CALL gcso_action_hub_hash_slot256_index(uint64_t ptr) GCSO_NOEXCEPT;

/**
 * @brief Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
 * @param context Active context handle.
 * @param mask Paged bitmask structure pointer.
 * @param chunk_len Sequence length of target chunk.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_swarm_cell_chunk_step(
    gcso_context_handle_t context, const gcso_paged_bitmask_t* GCSO_RESTRICT mask,
    uint32_t chunk_len) GCSO_NOEXCEPT;

/**
 * @brief Applies attractor pull force F_pull to steer active pointer chains toward target anchor.
 * @param context Active runtime context handle.
 * @param trail_id Source pointer trail index.
 * @param anchor_id Target anchor identifier.
 * @param pull_force Floating-point pull force scale.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_action_hub_pull_trail_to_attractor(gcso_context_handle_t context, uint32_t trail_id,
                                        uint32_t anchor_id, float pull_force) GCSO_NOEXCEPT;

/**
 * @brief Links adjacent cellular hallucinated trails into contiguous stigmergic trace graphs.
 * @param context Active context handle.
 * @param src_trail_id Source trail identifier.
 * @param dst_trail_id Destination trail identifier.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_link_hallucinated_trails(
    gcso_context_handle_t context, uint32_t src_trail_id, uint32_t dst_trail_id) GCSO_NOEXCEPT;

/**
 * @brief Performs hierarchical bit-tree reduction across paged block bitmasks.
 * @param bitmasks Array of input paged bitmasks.
 * @param num_masks Number of masks in input array.
 * @param reduced_out Pointer to receive reduced bitmask result.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_reduce_bit_tree(
    const gcso_paged_bitmask_t* GCSO_RESTRICT bitmasks, size_t num_masks,
    gcso_paged_bitmask_t* GCSO_RESTRICT reduced_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates SIMD/Warp bitmask reduction over PagedBlock KV caches.
 * @param kv_bits Raw uint64_t array of block key-value bit patterns.
 * @param num_blocks Total number of KV blocks to process.
 * @param mask_out Pointer to store generated paged bitmask.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_action_hub_paged_block_warp_bitmask(
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
 * @brief Allocates and initializes a DAES (Dynamic Adaptive Extension Scratchpad) slot.
 * @param slot_out Pointer to receive allocated DAES slot handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_daes_slot_create(gcso_daes_slot_handle_t* GCSO_RESTRICT slot_out) GCSO_NOEXCEPT;

/**
 * @brief Executes O(1) fast-path shortcut lookup in DAES dynamic scratchpad.
 * @param slot Pointer to DAES slot structure.
 * @param input_key Raw key address to look up.
 * @param shortcut_out Pointer to store bypassed address result.
 * @return GCSO_SUCCESS or GCSO_ERROR_INVALID_STATE.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_daes_fast_path_lookup(const gcso_daes_slot_t* GCSO_RESTRICT slot, uint64_t input_key,
                           uint64_t* GCSO_RESTRICT shortcut_out) GCSO_NOEXCEPT;

/**
 * @brief Pushes profiling metric bytes to the DAES telemetry ring ledger in zero-allocation mode.
 * @param slot Target DAES slot structure pointer.
 * @param metric_code Metric identifier byte code.
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_daes_telemetry_push(gcso_daes_slot_t* GCSO_RESTRICT slot,
                                                          uint8_t metric_code) GCSO_NOEXCEPT;

/**
 * @brief Sets the DAES slot operating mode (0 = Scratchpad, 1 = Plugin, 2 = Shared IPC Buffer).
 * @param slot Pointer to DAES slot structure.
 * @param mode Desired mode index (0, 1, or 2).
 * @return GCSO_SUCCESS or GCSO_ERROR_INVALID_ARGUMENT.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_daes_set_mode(gcso_daes_slot_t* GCSO_RESTRICT slot,
                                                    uint32_t mode) GCSO_NOEXCEPT;

/**
 * @brief Evaluates telemetry ledger to auto-tune PSPM routing ratios, RIPA bounds & EDBC
 * thresholds.
 * @param slot Const DAES slot structure pointer.
 * @param config_out Pointer to receive updated auto-tuned configuration structure.
 * @return GCSO_SUCCESS or error code.
 */
GCSO_API gcso_status_t GCSO_CALL
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
 * @param head_dim Attention head dimension (must be even).
 * @param num_heads Total number of attention heads.
 * @param kernel_out Pointer to receive created kernel handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_dpsr_kernel_create(uint32_t head_dim, uint32_t num_heads,
                        gcso_dpsr_kernel_handle_t* GCSO_RESTRICT kernel_out) GCSO_NOEXCEPT;

/**
 * @brief Applies inline DPSR phase rotation to Query tensor registers.
 * @param query_tensor 32-byte aligned query tensor buffer.
 * @param phase_deltas Q7 quantized phase delta array.
 * @param head_dim Head dimension size.
 * @param num_heads Total head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering(
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads) GCSO_NOEXCEPT;

/**
 * @brief Applies RIPA soft-bounded tanh clamping on low-frequency channels (upper d_head / 4
 * dimensions).
 * @param query_tensor 32-byte aligned query tensor buffer.
 * @param phase_deltas Q7 quantized phase delta array.
 * @param head_dim Head dimension size.
 * @param num_heads Total head count.
 * @param max_rad Maximum allowed phase limit in radians.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_safe(
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads, float max_rad) GCSO_NOEXCEPT;

/**
 * @brief QDPS discrete filter: cuts off phase rotation steps falling below min_step_rad.
 * @param phase_deltas Q7 phase array to filter in-place.
 * @param len Array element count.
 * @param min_step_rad Minimum step threshold angle in radians.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_qdps_filter_step(gcso_q7_t* GCSO_RESTRICT phase_deltas,
                                                       size_t len,
                                                       float min_step_rad) GCSO_NOEXCEPT;

/**
 * @brief Lazy Phase Unwrapping: Applies relative phase shift against context accumulator on Query
 * side.
 * @param query_tensor 32-byte aligned target query tensor.
 * @param context_accum Cumulative context phase vector.
 * @param head_dim Head dimension.
 * @param num_heads Head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_lazy_unwrap_override(
    float* GCSO_RESTRICT query_tensor, const float* GCSO_RESTRICT context_accum, size_t head_dim,
    size_t num_heads) GCSO_NOEXCEPT;

/**
 * @brief Executes norm-guarded Slerp phase stabilization on state vectors.
 * @param tensor Vector tensor buffer to normalize in-place.
 * @param dim Total vector length.
 * @param norm_lower Minimum allowed L2 norm bound.
 * @param norm_upper Maximum allowed L2 norm bound.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_slerp_norm_guard_stable(float* GCSO_RESTRICT tensor,
                                                                   size_t dim, float norm_lower,
                                                                   float norm_upper) GCSO_NOEXCEPT;

/**
 * @brief Fused inline logit phase shift prior to LM Head Softmax.
 * @param logits Logit score vector buffer.
 * @param vocab_size Vocabulary dimension.
 * @param phase_deltas Q7 phase deltas.
 * @param num_heads Head count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_fused_logit_shift(
    float* GCSO_RESTRICT logits, size_t vocab_size, const gcso_q7_t* GCSO_RESTRICT phase_deltas,
    size_t num_heads) GCSO_NOEXCEPT;

/**
 * @brief Destroys a DPSR kernel instance.
 * @param kernel Kernel handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_kernel_destroy(gcso_dpsr_kernel_handle_t kernel)
    GCSO_NOEXCEPT;

/**
 * @brief Allocates a PSPM (Phase-Steered Parallel Multi-head) router instance.
 * @param config PSPM router configuration pointer.
 * @param router_out Pointer to receive allocated router handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_pspm_router_create(const gcso_pspm_config_t* GCSO_RESTRICT config,
                        gcso_pspm_router_handle_t* GCSO_RESTRICT router_out) GCSO_NOEXCEPT;

/**
 * @brief Dispatches PSPM head-group phase profiles (Fact, Logic, Explore) in a single pass.
 * @param query_tensor 32-byte aligned query tensor buffer.
 * @param pspm_cfg PSPM routing configuration.
 * @param head_dim Head dimension.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_pspm_dispatch_single_pass(
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
 * @param descriptor Pointer to SRL descriptor.
 * @param adapter_out Pointer to receive allocated adapter handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_srl_adapter_create(const gcso_srl_descriptor_t* GCSO_RESTRICT descriptor,
                        gcso_srl_adapter_handle_t* GCSO_RESTRICT adapter_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates SRL Dynamic Rank-1 outer product: y = W_base*x + s (*) (u * (v^T * x)).
 * @param y_out Output vector buffer to accumulate into in-place.
 * @param x_in Input vector buffer.
 * @param srl_desc SRL descriptor structure pointer.
 * @param dim_in Input dimension.
 * @param dim_out Output dimension.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_srl_eval_rank1(float* GCSO_RESTRICT y_out, const float* GCSO_RESTRICT x_in,
                    const gcso_srl_descriptor_t* GCSO_RESTRICT srl_desc, size_t dim_in,
                    size_t dim_out) GCSO_NOEXCEPT;

/**
 * @brief L2P-SVD: Projects fine-tuned LoRA matrices via SVD into phase profiles and SRL vectors.
 * @param lora_a Pointer to LoRA A matrix buffer.
 * @param lora_b Pointer to LoRA B matrix buffer.
 * @param rank Rank size of input LoRA.
 * @param dim_in Input dimension.
 * @param dim_out Output dimension.
 * @param srl_out Output SRL descriptor structure to populate.
 * @param phase_profile_out Output Q7 phase profile array pointer.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_l2p_svd_project_lora(
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
// 5. Attractor Field, EDBC Engine & CVoid Barrier Interface
// ===================================================================

/**
 * @brief Allocates an Attractor Field instance.
 * @param field_out Pointer to receive allocated field handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_attractor_field_create(gcso_attractor_field_handle_t* GCSO_RESTRICT field_out) GCSO_NOEXCEPT;

/**
 * @brief Registers a topological anchor point in the attractor field.
 * @param context Active context handle.
 * @param anchor_type Anchor classification enum type.
 * @param vec 32-byte aligned feature vector.
 * @param dim Vector dimension size.
 * @param anchor_id_out Pointer to receive assigned anchor ID.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_add_anchor(
    gcso_context_handle_t context, gcso_anchor_type_t anchor_type, const float* GCSO_RESTRICT vec,
    size_t dim, uint32_t* GCSO_RESTRICT anchor_id_out) GCSO_NOEXCEPT;

/**
 * @brief Maps natural language system prompt text as primary Anchor Attractor.
 * @param context Active runtime context handle.
 * @param prompt_text Null-terminated UTF-8 system prompt string.
 * @param weight Attractor weight parameter.
 * @param anchor_id_out Pointer to store assigned anchor ID.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_add_system_prompt_anchor(
    gcso_context_handle_t context, const char* GCSO_RESTRICT prompt_text, float weight,
    uint32_t* GCSO_RESTRICT anchor_id_out) GCSO_NOEXCEPT;

/**
 * @brief Maps dense feature embedding vector as continuous attractor anchor.
 * @param context Active runtime context handle.
 * @param embedding 32-byte aligned embedding array.
 * @param dim Vector dimension.
 * @param weight Attractor weight parameter.
 * @param anchor_id_out Pointer to store assigned anchor ID.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_add_embedding_anchor(
    gcso_context_handle_t context, const float* GCSO_RESTRICT embedding, size_t dim, float weight,
    uint32_t* GCSO_RESTRICT anchor_id_out) GCSO_NOEXCEPT;

/**
 * @brief Injects phase-conjugate repulsion vector (-dTheta) to flip spurious local minima into
 * repulsive peaks.
 * @param context Active runtime context handle.
 * @param repulsion_deltas Q7 anti-phase array.
 * @param num_heads Total head count.
 * @param gain Repulsion strength multiplier.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_inject_phase_repulsion(
    gcso_context_handle_t context, const gcso_q7_t* GCSO_RESTRICT repulsion_deltas,
    size_t num_heads, float gain) GCSO_NOEXCEPT;

/**
 * @brief Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors.
 * @param context Active context handle.
 * @param new_anchor_count_out Pointer to receive updated total anchor count.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_aggregate_bottom_up(
    gcso_context_handle_t context, uint32_t* GCSO_RESTRICT new_anchor_count_out) GCSO_NOEXCEPT;

/**
 * @brief Destroys an attractor field instance.
 * @param field Attractor field handle to destroy (safe no-op if NULL).
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_attractor_field_destroy(gcso_attractor_field_handle_t field)
    GCSO_NOEXCEPT;

/**
 * @brief Allocates an EDBC (Entropy-Driven Decoding Branch Controller) instance.
 * @param initial_state Pointer to initial state configuration structure.
 * @param controller_out Pointer to receive allocated controller handle.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_edbc_controller_create(
    const gcso_edbc_state_t* GCSO_RESTRICT initial_state,
    gcso_edbc_controller_handle_t* GCSO_RESTRICT controller_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates Moving Z-Score Normalized Attention Entropy H~ and pitchfork bifurcation mode.
 * @param controller Controller handle.
 * @param token_z_score Floating-point activation Z-score.
 * @param state_out Output state structure pointer.
 * @return GCSO_SUCCESS or GCSO_ERROR_EDBC_SINGULARITY (if NaN/Inf detected).
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_edbc_eval_stateful(gcso_edbc_controller_handle_t controller, float token_z_score,
                        gcso_edbc_state_t* GCSO_RESTRICT state_out) GCSO_NOEXCEPT;

/**
 * @brief Computes CVoid Coherent Vector Alignment Metric for Out-of-Distribution Latent Space.
 * @param key_vector 32-byte aligned key tensor vector.
 * @param dim Vector dimension.
 * @param void_score_out Pointer to receive computed void score result.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_cvoid_eval_dyadic128(const float* GCSO_RESTRICT key_vector, size_t dim,
                          float* GCSO_RESTRICT void_score_out) GCSO_NOEXCEPT;

/**
 * @brief Evaluates Eyring-Kramers potential barrier height value with singularity check.
 * @param void_score Raw void score.
 * @param tau_eff Effective threshold temperature.
 * @param barrier_out Pointer to receive computed barrier height.
 * @return GCSO_SUCCESS or GCSO_ERROR_EDBC_SINGULARITY.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_cvoid_eval_barrier(
    float void_score, float tau_eff, float* GCSO_RESTRICT barrier_out) GCSO_NOEXCEPT;

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
 * @param context Active runtime context handle.
 * @param patch_data Pointer to binary patch buffer.
 * @param patch_size Size of binary patch in bytes.
 * @return GCSO_SUCCESS or error status code.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_persona_apply_patch(gcso_context_handle_t context,
                                                          const uint8_t* GCSO_RESTRICT patch_data,
                                                          size_t patch_size) GCSO_NOEXCEPT;

/**
 * @brief Zero-Overhead In-Memory Mapped Storage: Maps .gcso container payload using zero-copy mmap.
 * @param file_path Null-terminated path string to target .gcso file.
 * @param zimms_out Descriptor structure pointer to populate.
 * @return GCSO_SUCCESS or GCSO_ERROR_ZIMMS_MAPPING_FAILED.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_zimms_open_mmap(const char* GCSO_RESTRICT file_path,
                     gcso_zimms_descriptor_t* GCSO_RESTRICT zimms_out) GCSO_NOEXCEPT;

/**
 * @brief Unmaps zero-copy ZIMMS memory handle and releases Direct DMA resources.
 * @param zimms_desc Descriptor structure pointer to clear.
 * @return GCSO_SUCCESS or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_zimms_close_mmap(gcso_zimms_descriptor_t* GCSO_RESTRICT zimms_desc) GCSO_NOEXCEPT;

GCSO_EXTERN_C_END

#endif // LIMINIKA_GCSO_ABI_H