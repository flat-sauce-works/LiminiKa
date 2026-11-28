# LiminiKa C-ABI Specification (`v0.4.0` / `0x0000_0400`)

## 1. Overview & Binary Compatibility Contract

`liminika-core` exposes a compact, `#![no_std]`, zero-allocation-on-hot-path C11/C++20 ABI defined across three self-contained headers under [`include/liminika/`](../../include/liminika/):

- [`lmnk_config.h`](../../include/liminika/lmnk_config.h): Portability, alignment (`LMNK_ALIGN_PREFIX`/`LMNK_ALIGN_SUFFIX`/`LMNK_ALIGNAS`/`LMNK_ALIGNOF`), visibility (`LMNK_API`/`LMNK_CALL`), `LMNK_NODISCARD`, `LMNK_RESTRICT`, `LMNK_NOEXCEPT`, and `LMNK_STATIC_ASSERT` macros.
- [`lmnk_types.h`](../../include/liminika/lmnk_types.h): Canonical version/capability constants, status codes, flags, 7 `static inline` tagged-pointer helpers, device-ready POD structures, versioned parameter/report/view structs, and compile-time `sizeof`/`alignof`/`offsetof` assertions.
- [`lmnk_abi.h`](../../include/liminika/lmnk_abi.h): Exactly **27** exported `lmnk_*` C-ABI functions.

### 1.1 ABI Invariants

- **Version & Pre-1.0 Compatibility Policy**:
  - `LMNK_ABI_VERSION_MAJOR = 0`, `LMNK_ABI_VERSION_MINOR = 4`, `LMNK_ABI_VERSION_PATCH = 0`.
  - `LMNK_ABI_VERSION_HEX = 0x0000_0400`, `LMNK_ABI_COMPAT_MASK = 0xFFFF_FF00`, `LMNK_ABI_COMPAT_VERSION_HEX = 0x0000_0400`, `LMNK_ABI_VERSION_STRING = "0.4.0"`.
  - Writers stamp exact `LMNK_ABI_VERSION_HEX`. Readers enforce `is_compatible_abi_version`: `file_version != 0`, identical `MAJOR.MINOR` prefix (`& 0xFFFF_FF00`), and `file_patch <= runtime_patch`.
- **Zero `void*` Surface**: All tensor buffers use strongly typed `float*` / `const float*`, byte buffers use `uint8_t*` / `const uint8_t*`, and opaque contexts use `lmnk_context_handle_t` (`struct LmnkContextOpaque*`).
- **Mandatory Result Checking**: Every exported function carries `LMNK_NODISCARD`.
- **Bidirectional `struct_size` Fail-Closed Envelopes**: All versioned parameter, report, and view structs carry a leading `uint32_t struct_size` initialized by the caller to `sizeof(T)` and validated fail-closed on both input and output paths.
- **Strict Provenance**: `lmnk_rank1_descriptor_t` (`32 B`, `align(32)`) stores index-based `lmnk_arena_span_t { uint32_t offset; uint32_t len; }` (`8 B`) pairs resolved against a caller-provided `const float* arena` pointer with zero integer-to-pointer casts.
- **Endianness & Zero Implicit Padding**: Canonical wire byte order is Little-Endian. Portable POD types use fixed-width primitives only with explicit reserved padding bytes.

### 1.2 Capability Flags (`lmnk_abi_info_t.capability_flags`)

| Constant | Bit | Meaning |
| :--- | :---: | :--- |
| `LMNK_CAP_SWAR_BASELINE` | `1 << 0` | Portable Little-Endian SWAR baseline support |
| `LMNK_CAP_SIMD_AVX2` | `1 << 1` | x86_64 AVX2 vector acceleration active |
| `LMNK_CAP_SIMD_NEON` | `1 << 2` | AArch64 NEON vector acceleration active |
| `LMNK_CAP_SIMD_WASM128` | `1 << 3` | WebAssembly 128-bit SIMD active |
| `LMNK_CAP_ZERO_ALLOC_CORE` | `1 << 4` | Strict `#![no_std]` zero-allocation hot path |
| `LMNK_CAP_INPLACE_CONTEXT_INIT` | `1 << 5` | Caller-buffered zero-heap `in_place` context lifecycle |

---

## 2. Exported Function Surface (27 Functions)

Rust exports (`src/core/src/runtime/abi/{runtime,kernels}.rs`) and C declarations (`include/liminika/lmnk_abi.h`) are verified 1:1 by `cargo xtask --check`.

### 2.1 Handshake & Configuration (6 Functions)

| Function | Signature Summary | Purpose |
| :--- | :--- | :--- |
| `lmnk_abi_version_hex` | `(void) -> uint32_t` | Returns `0x00000400U`. |
| `lmnk_abi_version_string` | `(void) -> const char*` | Returns static NUL-terminated `"0.4.0"`. |
| `lmnk_status_to_string` | `(lmnk_status_t) -> const char*` | Maps status code to static ASCII symbol string. |
| `lmnk_abi_info` | `(lmnk_abi_info_t* info_out) -> lmnk_status_t` | Populates ABI version, `RuntimeContext` size (`5,184 B`) / align (`64`), snapshot budget (`576 B`), `topology_crc32`, and capabilities. |
| `lmnk_config_init_default` | `(lmnk_config_t* config) -> lmnk_status_t` | Initializes canonical default `lmnk_config_t` (`64 B`). |
| `lmnk_config_validate` | `(const lmnk_config_t* config) -> lmnk_status_t` | Branchlessly validates `struct_size`, dimensions, scales, and capacities. |

### 2.2 `RuntimeContext` Lifecycle, Policy, Decode, State & Snapshot (12 Functions)

| Function | Signature Summary | Purpose |
| :--- | :--- | :--- |
| `lmnk_context_init_in_place` | `(const lmnk_config_t*, uint8_t* mem_out, size_t mem_bytes)` | Initializes a `RuntimeContext` (`5,184 B`, 64 B aligned) in caller-owned memory (`0 alloc`). |
| `lmnk_context_deinit_in_place` | `(lmnk_context_handle_t context)` | Volatile-zeroizes and tears down an `in_place` context. |
| `lmnk_context_create` | `(const lmnk_config_t*, lmnk_context_handle_t* context_out)` | Heap-allocates and initializes a boxed `RuntimeContext` (`alloc` feature). |
| `lmnk_context_destroy` | `(lmnk_context_handle_t context)` | Volatile-zeroizes and frees a boxed `RuntimeContext`. |
| `lmnk_context_reset` | `(lmnk_context_handle_t context)` | Resets dynamic turn/substrate state while preserving immutable host pin floors. |
| `lmnk_context_configure_policy` | `(lmnk_context_handle_t, const lmnk_runtime_policy_t*)` | Atomically updates spine, compute pool, and/or envelope policy (`LMNK_POLICY_FLAG_*`). |
| `lmnk_context_begin_turn` | `(lmnk_context_handle_t, const lmnk_turn_begin_params_t*)` | Initializes per-turn guard / unobserved lock flags (`LMNK_TURN_FLAG_*`). |
| `lmnk_context_step_decode` | `(lmnk_context_handle_t, const lmnk_decode_step_params_t*, lmnk_decode_step_report_t*)` | Executes one decode control step (`Spine` + `Envelope` + `DistortionTension`), returning dispatch mask, veto mask, and logits observe directive. |
| `lmnk_context_get_state` | `(lmnk_context_handle_t, lmnk_runtime_state_view_t*)` | Copies an abstracted read-only `64 B` runtime state snapshot (including spine pins, bypass status, and envelope contraction/breach/baseline fields). |
| `lmnk_context_emit_boundary_signal` | `(lmnk_context_handle_t, const lmnk_boundary_emit_params_t*, lmnk_boundary_signal_t*)` | Emits a `32 B` POD `lmnk_boundary_signal_t` (`0 alloc`, `0 fdiv`). |
| `lmnk_context_serialize` | `(lmnk_context_handle_t, uint8_t* buffer, size_t* buffer_size)` | Two-pass sizing or serialization into a `576 B` unified `.lmnk` snapshot container. |
| `lmnk_context_deserialize` | `(const uint8_t* buffer, size_t buffer_size, lmnk_context_handle_t* context_out)` | Verifies header, ABI version, `topology_crc32`, and CRC32, restoring context while enforcing host pin floors. |

### 2.3 Hot-Path Kernels, Lacuna & Standalone Envelope (6 Functions)

| Function | Signature Summary | Purpose |
| :--- | :--- | :--- |
| `lmnk_rope_steer_q` | `(const lmnk_rope_steer_params_t*)` | In-place query-only ($Q$) $SO(2)$ RoPE phase rotation (`float*`, Q0.7 deltas, clamped or unclamped). |
| `lmnk_rope_procrustes_q7` | `(const lmnk_rope_procrustes_params_t*)` | Recovers per-head Q0.7 phase deltas between `q_before` and `q_after` (`float*`). |
| `lmnk_rank1_eval` | `(const lmnk_rank1_eval_params_t*)` | Evaluates arena-indexed Rank-1 residual injection (`lmnk_rank1_descriptor_t` + `const float* arena`). |
| `lmnk_gateway_steer` | `(const lmnk_gateway_steer_params_t*)` | Synthesizes 2-origin (Fact/Persona) gateway steering, updates `lmnk_pointer_trail_t`, and applies query rotation. |
| `lmnk_lacuna_partition_eval` | `(const lmnk_lacuna_eval_params_t*, lmnk_lacuna_partition_t*)` | Branchlessly evaluates 8-granule 4-state `Lacuna` partition masks. |
| `lmnk_envelope_step` | `(lmnk_envelope_state_t*, const lmnk_envelope_step_params_t*, lmnk_envelope_feedback_t*)` | Evaluates one Tier-1 standalone safety-envelope step (`0 alloc`, `0 fdiv`) directly on `lmnk_envelope_state_t`. |

### 2.4 Generic Unified `.lmnk` Container I/O (3 Functions)

| Function | Signature Summary | Purpose |
| :--- | :--- | :--- |
| `lmnk_container_write` | `(const lmnk_container_write_params_t*)` | Writes a 64B-aligned `.lmnk` container stamped with active ABI version, `CORE_TOPOLOGY.topology_crc32`, and IEEE 802.3 CRC32s. |
| `lmnk_container_verify` | `(const uint8_t* bytes, size_t byte_len, lmnk_container_verify_report_t*)` | Validates container bounds, magic, ABI version, `topology_crc32`, monotonic aligned spans, duplicate kinds, and CRC32s. |
| `lmnk_container_find_section` | `(const uint8_t* bytes, size_t byte_len, uint16_t kind, lmnk_container_section_view_t*)` | Verifies container and returns a zero-copy pointer/length view of section `kind` (or `LMNK_ERR_ANCHOR_NOT_FOUND`). |

### 2.5 Header-Only Tagged Pointer Helpers (7 `static inline` Functions)

Defined in `lmnk_types.h` (`0` FFI call overhead): `lmnk_tagged_ptr_pack`, `lmnk_tagged_ptr_unpack_addr`, `lmnk_tagged_ptr_unpack_tag`, `lmnk_tagged_ptr_unpack_phase`, `lmnk_tagged_ptr_pack_action`, `lmnk_tagged_ptr_unpack_action_mode`, `lmnk_tagged_ptr_unpack_attractor_slot`.

---

## 3. Verified Memory Layouts (`v0.4.0`)

### 3.1 `RuntimeContext` Layout

`RuntimeContext` is `#[repr(C, align(64))]`, `!needs_drop`, sized at **`5,184 B`** (`81` cache lines), ordered Hot -> Warm -> Cold:

- `raw_config` (`RuntimeConfig`, `64 B`) @ offset `0`
- `envelope` (`EnvelopeState`, `64 B`) @ offset `320`
- `system_spine` (`SystemSpineState`, `32 B`) @ offset `384`
- `envelope_bounds` (`EnvelopeBounds`, `32 B`) @ offset `416`
- `derived_config` (`24 B`) @ offset `592`
- `allocation_mode` (`u8`) @ offset `654`
- `telemetry` (`CoreTelemetryBank`, `576 B`) @ offset `4608`

### 3.2 Canonical Struct Sizes & Alignments

| C Type (`lmnk_types.h`) | Rust Type (`liminika-core`) | Size (B) | Align (B) | Leading `struct_size` |
| :--- | :--- | ---: | ---: | :---: |
| `lmnk_arena_span_t` | `ArenaSpan` | `8` | `4` | No (POD) |
| `lmnk_rank1_descriptor_t` | `Rank1Descriptor` | `32` | `32` | No (POD) |
| `lmnk_lacuna_partition_t` | `LacunaPartition8` | `16` | `16` | No (POD) |
| `lmnk_boundary_signal_t` | `BoundarySignal` | `32` | `32` | No (POD) |
| `lmnk_envelope_bounds_t` | `EnvelopeBounds` | `32` | `32` | No (POD) |
| `lmnk_envelope_feedback_t` | `EnvelopeFeedback` | `16` | `16` | No (POD) |
| `lmnk_envelope_state_t` | `EnvelopeState` | `64` | `64` | No (POD) |
| `lmnk_system_pressure_reading_t` | `SystemPressureReading` | `8` | `2` | No (POD) |
| `lmnk_compute_dispatch_mask_t` | `ComputeDispatchMask` | `40` | `8` | No (POD) |
| `lmnk_spine_decode_input_t` | `SpineDecodeInput` | `32` | `16` | No (POD) |
| `lmnk_distortion_tension_t` | `DistortionTension` | `32` | `16` | No (POD) |
| `lmnk_distortion_delta_t` | `DistortionDelta` | `128` | `64` | No (POD) |
| `lmnk_pointer_trail_t` | `PointerTrail` | `128` | `64` | No (POD) |
| `lmnk_gateway_frame_t` | `GatewayFrame` | `128` | `64` | No (POD) |
| `lmnk_phase_anchor_record_t` | `PhaseAnchorRecord` | `128` | `64` | No (POD) |
| `lmnk_core_state_stack_t` | `CoreStateStack` | `128` | `64` | No (POD) |
| `lmnk_container_header_t` | `LmnkContainerHeader` | `64` | `64` | No (POD) |
| `lmnk_section_entry_t` | `LmnkSectionEntry` | `16` | `16` | No (POD) |
| `lmnk_topology_matrix_t` | `LiminalTopologyMatrix` | `64` | `64` | No (POD) |
| `lmnk_config_t` | `RuntimeConfig` | `64` | `64` | Yes (`64`) |
| `lmnk_abi_info_t` | `lmnk_abi_info_t` | `64` | `8` | Yes (`64`) |
| `lmnk_runtime_policy_t` | `lmnk_runtime_policy_t` | `64` | `8` | Yes (`64`) |
| `lmnk_turn_begin_params_t` | `lmnk_turn_begin_params_t` | `16` | `4` | Yes (`16`) |
| `lmnk_decode_step_params_t` | `lmnk_decode_step_params_t` | `48` | `16` | Yes (`48`) |
| `lmnk_decode_step_report_t` | `lmnk_decode_step_report_t` | `64` | `8` | Yes (`64`) |
| `lmnk_runtime_state_view_t` | `lmnk_runtime_state_view_t` | `64` | `8` | Yes (`64`) |
| `lmnk_envelope_step_params_t` | `lmnk_envelope_step_params_t` | `96` | `32` | Yes (`96`) |
| `lmnk_boundary_emit_params_t` | `lmnk_boundary_emit_params_t` | `16` | `4` | Yes (`16`) |
| `lmnk_lacuna_eval_params_t` | `lmnk_lacuna_eval_params_t` | `24` | `4` | Yes (`24`) |
| `lmnk_container_verify_report_t` | `lmnk_container_verify_report_t` | `40` | `8` | Yes (`40`) |

---

## 4. 4-State `Lacuna`, Attribution Tags & Status Codes

### 4.1 Canonical 4-State `Lacuna` (`LMNK_LACUNA_*`)

| Constant | Value | Epistemic Meaning |
| :--- | :---: | :--- |
| `LMNK_LACUNA_UNOBSERVED` | `0` | Zero-aperture non-observation (`APERTURE_ZERO`, no OOD bleed) |
| `LMNK_LACUNA_SELF_REF` | `1` | Stationary self-reference inside deadband (shortcut eligible) |
| `LMNK_LACUNA_EXTERIOR` | `2` | Observed exterior delta staged in isolated sandbox lanes |
| `LMNK_LACUNA_UNDEFINED` | `3` | Non-finite or out-of-range fault (`fail-closed` isolation) |

### 4.2 3-Bit Attribution Tag (`LMNK_ATTRIBUTION_TAG(primitive, driver)`)

- Bit `0`: `LMNK_PRIMITIVE_LIMINAL = 0`, `LMNK_PRIMITIVE_STIGMERGY = 1`.
- Bits `1..=2`: `LMNK_DRIVER_SYSTEM = 0`, `LMNK_DRIVER_EGO = 1`, `LMNK_DRIVER_ENVELOPE = 2`.

### 4.3 Status Codes (`lmnk_status_t`)

| Code | Name | Meaning |
| ---: | :--- | :--- |
| `0` | `LMNK_OK` | Success |
| `-1` | `LMNK_ERR_INVALID_ARGUMENT` | Invalid argument, unknown flag, or `struct_size` mismatch |
| `-2` | `LMNK_ERR_OUT_OF_MEMORY` | Cold-path allocation failure |
| `-3` | `LMNK_ERR_BUFFER_TOO_SMALL` | Output buffer smaller than required bytes |
| `-4` | `LMNK_ERR_PANIC_CAUGHT` | Internal panic trapped at FFI boundary |
| `-5` | `LMNK_ERR_NULL_POINTER` | Unexpected `NULL` pointer |
| `-6` | `LMNK_ERR_INVALID_STATE` | Invalid lifecycle state (e.g., `deinit` on boxed handle) |
| `-7` | `LMNK_ERR_VERSION_MISMATCH` | Incompatible ABI or container version |
| `-8` | `LMNK_ERR_ALIGNMENT` | Required alignment violated |
| `-9` | `LMNK_ERR_IO_FAILURE` | Storage or I/O error |
| `-10` | `LMNK_ERR_SUBSTRATE_FULL` | Substrate lane capacity saturated |
| `-11` | `LMNK_ERR_NOT_IMPLEMENTED` | Feature unavailable in current build |
| `-20` | `LMNK_ERR_ANCHOR_NOT_FOUND` | Anchor ID or container section kind absent |
| `-30` | `LMNK_ERR_PHASE_OVERFLOW` | Phase accumulation exceeded bound |
| `-31` | `LMNK_ERR_PHASE_STEP_UNDERFLOW` | Phase step below minimum filter threshold |
| `-40` | `LMNK_ERR_ENTROPY_SINGULARITY` | Numerical singularity in entropy evaluation |
| `-50` | `LMNK_ERR_CONTAINER_CORRUPTED` | Container magic, span, or CRC32 check failed |
| `-51` | `LMNK_ERR_STORAGE_MAP_FAILED` | Storage buffer mapping failed |
| `-52` | `LMNK_ERR_TOPOLOGY_MISMATCH` | Container `topology_crc32` mismatches active `CORE_TOPOLOGY` |
| `-60` | `LMNK_ERR_SUBHEAD_ROUTING_FAILED` | Sub-head group routing failed |
| `-70` | `LMNK_ERR_KV_KEYFRAME_SEEK_FAILED` | Keyframe index lookup or seek failed |
| `-80` | `LMNK_ERR_OBSTRUCTION_UNRESOLVED` | Unresolved cohomological obstruction |
| `-90` | `LMNK_ERR_EXTENSION_NOT_LOADED` | Extension slot inactive |
| `-91` | `LMNK_ERR_LEDGER_FULL` | Fast-path ledger ring saturated |
| `-0x7FFFFFFF` | `LMNK_ERR_UNKNOWN` | Unknown internal error |

---

## 5. Minimal C11 Zero-Heap Usage Pattern

```c
#include "liminika/lmnk_abi.h"
#include <stdalign.h>
#include <string.h>

int main(void) {
    lmnk_abi_info_t info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    if (LMNK_FAILED(lmnk_abi_info(&info))) {
        return 1;
    }

    lmnk_config_t cfg;
    if (LMNK_FAILED(lmnk_config_init_default(&cfg))) {
        return 2;
    }

    alignas(64) uint8_t ctx_buf[5184];
    if (info.context_size > sizeof(ctx_buf)) {
        return 3;
    }
    if (LMNK_FAILED(lmnk_context_init_in_place(&cfg, ctx_buf, sizeof(ctx_buf)))) {
        return 4;
    }

    lmnk_context_handle_t ctx = (lmnk_context_handle_t)ctx_buf;
    lmnk_runtime_state_view_t view;
    memset(&view, 0, sizeof(view));
    view.struct_size = sizeof(view);
    if (LMNK_FAILED(lmnk_context_get_state(ctx, &view))) {
        (void)lmnk_context_deinit_in_place(ctx);
        return 5;
    }

    return LMNK_SUCCEEDED(lmnk_context_deinit_in_place(ctx)) ? 0 : 6;
}
```
