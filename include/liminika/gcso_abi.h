// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_ABI_H
#define LIMINIKA_GCSO_ABI_H

#include "liminika/gcso_config.h"
#include "liminika/gcso_types.h"

GCSO_EXTERN_C_BEGIN

// System & Capability Query Interface
GCSO_API void GCSO_CALL gcso_abi_get_version(uint32_t* GCSO_RESTRICT major,
                                             uint32_t* GCSO_RESTRICT minor,
                                             uint32_t* GCSO_RESTRICT patch) GCSO_NOEXCEPT;

GCSO_API GCSO_NODISCARD const char* GCSO_CALL gcso_abi_get_version_string(void) GCSO_NOEXCEPT;

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

GCSO_API gcso_status_t GCSO_CALL gcso_context_set_system_prompt_anchor(
    gcso_context_handle_t context, const char* GCSO_RESTRICT prompt_text,
    float weight) GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL gcso_context_step_token(
    gcso_context_handle_t context, uint32_t token_id, float* GCSO_RESTRICT query_tensor,
    float* GCSO_RESTRICT key_tensor, gcso_pointer_trail_t* GCSO_RESTRICT trail_out) GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL
gcso_context_serialize(gcso_context_handle_t context, uint8_t* GCSO_RESTRICT buffer,
                       size_t* GCSO_RESTRICT buffer_size) GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL
gcso_context_deserialize(const uint8_t* GCSO_RESTRICT buffer, size_t buffer_size,
                         gcso_context_handle_t* GCSO_RESTRICT context_out) GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL gcso_context_destroy(gcso_context_handle_t context) GCSO_NOEXCEPT;

// DPSR Kernel & Steering Interface
GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering(
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads) GCSO_NOEXCEPT;

GCSO_API gcso_status_t GCSO_CALL gcso_dpsr_apply_phase_steering_safe(
    float* GCSO_RESTRICT query_tensor, const gcso_q7_t* GCSO_RESTRICT phase_deltas, size_t head_dim,
    size_t num_heads, float max_rad) GCSO_NOEXCEPT;

GCSO_EXTERN_C_END

#endif // LIMINIKA_GCSO_ABI_H