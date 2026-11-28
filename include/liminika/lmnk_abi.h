// name: include/liminika/lmnk_abi.h
// SPDX-License-Identifier: MIT OR Apache-2.0

/**
 * @file lmnk_abi.h
 * @brief LiminiKa Core Engine C-ABI v0.4.0 (`0x00000400`) Public Functional Interface (27 exports).
 *
 * Exposes an exact 1:1, versioned parameter struct (`struct_size`), `float*`-only (`0 void*`)
 * C11/C++20 surface for `liminika-core`:
 * - Two-stage ABI handshake & configuration (`lmnk_abi_*`, `lmnk_config_*`)
 * - Zero-heap in-place and boxed `RuntimeContext` lifecycle, policy, decode step, state view,
 *   boundary signal emission, and `.lmnk` snapshot persistence (`lmnk_context_*`)
 * - Hot-path RoPE steering, Procrustes Q7 recovery, arena-indexed Rank-1 evaluation,
 *   2-origin Gateway steering, 8-granule Lacuna partition evaluation, and Tier-1 standalone
 *   safety-envelope stepping (`lmnk_envelope_step`)
 * - Generic zero-copy `.lmnk` container write, verify, and section lookup (`lmnk_container_*`)
 */

#ifndef LIMINIKA_LMNK_ABI_H
#define LIMINIKA_LMNK_ABI_H

#include "lmnk_types.h"

LMNK_EXTERN_C_BEGIN

/* -------------------------------------------------------------------------- */
/* 1. Two-Stage Version Handshake & Configuration (6 functions)               */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API uint32_t LMNK_CALL lmnk_abi_version_hex(void) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API const char* LMNK_CALL lmnk_abi_version_string(void) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API const char* LMNK_CALL lmnk_status_to_string(lmnk_status_t status)
    LMNK_NOEXCEPT;

/**
 * Populates a versioned handshake struct with ABI version, context size/alignment, snapshot budget,
 * topology CRC32, and runtime capability flags.
 * Caller must set `info_out->struct_size = sizeof(lmnk_abi_info_t)` before calling.
 */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_abi_info(lmnk_abi_info_t* LMNK_RESTRICT info_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_config_init_default(lmnk_config_t* LMNK_RESTRICT config) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_config_validate(const lmnk_config_t* LMNK_RESTRICT config) LMNK_NOEXCEPT;

/* -------------------------------------------------------------------------- */
/* 2. RuntimeContext Lifecycle, Policy, Decode, State & Snapshot (12 functions)*/
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_context_init_in_place(const lmnk_config_t* LMNK_RESTRICT config,
                           uint8_t* LMNK_RESTRICT mem_out, size_t mem_bytes) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_context_deinit_in_place(lmnk_context_handle_t context) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_context_create(const lmnk_config_t* LMNK_RESTRICT config,
                    lmnk_context_handle_t* LMNK_RESTRICT context_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_destroy(lmnk_context_handle_t context)
    LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_reset(lmnk_context_handle_t context)
    LMNK_NOEXCEPT;

/**
 * Configures spine and/or anonymous compute-pool policies atomically (`0 alloc`).
 * Caller must set `policy->struct_size = sizeof(lmnk_runtime_policy_t)`.
 */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_configure_policy(
    lmnk_context_handle_t context, const lmnk_runtime_policy_t* LMNK_RESTRICT policy) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_context_begin_turn(lmnk_context_handle_t context,
                        const lmnk_turn_begin_params_t* LMNK_RESTRICT params) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_step_decode(
    lmnk_context_handle_t context, const lmnk_decode_step_params_t* LMNK_RESTRICT params,
    lmnk_decode_step_report_t* LMNK_RESTRICT report_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_context_get_state(lmnk_context_handle_t context,
                       lmnk_runtime_state_view_t* LMNK_RESTRICT state_out) LMNK_NOEXCEPT;

/**
 * Emits a raw 32-byte POD `lmnk_boundary_signal_t` from `context` (`0 alloc`, `0 fdiv`).
 * Caller must set `params->struct_size = sizeof(lmnk_boundary_emit_params_t)`.
 */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL lmnk_context_emit_boundary_signal(
    lmnk_context_handle_t context, const lmnk_boundary_emit_params_t* LMNK_RESTRICT params,
    lmnk_boundary_signal_t* LMNK_RESTRICT signal_out) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_context_serialize(lmnk_context_handle_t context, uint8_t* LMNK_RESTRICT buffer,
                       size_t* LMNK_RESTRICT buffer_size) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_context_deserialize(const uint8_t* LMNK_RESTRICT buffer, size_t buffer_size,
                         lmnk_context_handle_t* LMNK_RESTRICT context_out) LMNK_NOEXCEPT;

/* -------------------------------------------------------------------------- */
/* 3. Hot-Path Kernels, Lacuna & Tier-1 Standalone Envelope (6 functions)     */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_rope_steer_q(const lmnk_rope_steer_params_t* LMNK_RESTRICT params) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_rope_procrustes_q7(const lmnk_rope_procrustes_params_t* LMNK_RESTRICT params) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_rank1_eval(const lmnk_rank1_eval_params_t* LMNK_RESTRICT params) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_gateway_steer(const lmnk_gateway_steer_params_t* LMNK_RESTRICT params) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_lacuna_partition_eval(const lmnk_lacuna_eval_params_t* LMNK_RESTRICT params,
                           lmnk_lacuna_partition_t* LMNK_RESTRICT out) LMNK_NOEXCEPT;

/**
 * Evaluates one Tier-1 standalone safety-envelope step directly on `state_inout` (`0 alloc`).
 * Caller must set `params->struct_size = sizeof(lmnk_envelope_step_params_t)` (`96`).
 * When `LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS` is set, `params->bounds` is ignored and canonical
 * defaults are applied.
 */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_envelope_step(lmnk_envelope_state_t* LMNK_RESTRICT state_inout,
                   const lmnk_envelope_step_params_t* LMNK_RESTRICT params,
                   lmnk_envelope_feedback_t* LMNK_RESTRICT feedback_out) LMNK_NOEXCEPT;

/* -------------------------------------------------------------------------- */
/* 4. Generic Unified `.lmnk` Container I/O (3 functions)                     */
/* -------------------------------------------------------------------------- */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_container_write(const lmnk_container_write_params_t* LMNK_RESTRICT params) LMNK_NOEXCEPT;

LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_container_verify(const uint8_t* LMNK_RESTRICT bytes, size_t byte_len,
                      lmnk_container_verify_report_t* LMNK_RESTRICT report_out) LMNK_NOEXCEPT;

/**
 * Verifies a `.lmnk` container buffer and returns a zero-copy view of section `kind`.
 * Returns `LMNK_ERR_ANCHOR_NOT_FOUND` when the container is valid but `kind` is absent.
 */
LMNK_NODISCARD LMNK_API lmnk_status_t LMNK_CALL
lmnk_container_find_section(const uint8_t* LMNK_RESTRICT bytes, size_t byte_len, uint16_t kind,
                            lmnk_container_section_view_t* LMNK_RESTRICT view_out) LMNK_NOEXCEPT;

LMNK_EXTERN_C_END

#endif /* LIMINIKA_LMNK_ABI_H */
