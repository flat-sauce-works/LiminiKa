// name: tests/ffi/test_c_abi_smoke.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "liminika/lmnk_config.h"
#include "liminika/lmnk_types.h"
#include "liminika/lmnk_abi.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {

#define LMNK_TEST_CHECK(cond)                                                                \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "LMNK C-ABI smoke check failed at line %d: %s\n", __LINE__, \
                         #cond);                                                             \
            std::abort();                                                                    \
        }                                                                                    \
    } while (0)

void test_static_invariants() {
    static_assert(LMNK_ABI_VERSION_MAJOR == 0U);
    static_assert(LMNK_ABI_VERSION_MINOR == 4U);
    static_assert(LMNK_ABI_VERSION_PATCH == 0U);
    static_assert(LMNK_ABI_VERSION_HEX == 0x00000400U);
    static_assert(LMNK_ABI_COMPAT_MASK == 0xFFFFFF00U);
    static_assert(LMNK_ABI_COMPAT_VERSION_HEX == 0x00000400U);
    static_assert(LMNK_PHASE_ANCHOR_MAGIC == 0x524B4E4CU);
    static_assert(LMNK_PHASE_ANCHOR_VERSION == LMNK_ABI_VERSION_HEX);

    static_assert(sizeof(lmnk_arena_span_t) == 8U);
    static_assert(sizeof(lmnk_rank1_descriptor_t) == 32U);
    static_assert(alignof(lmnk_rank1_descriptor_t) == 32U);
    static_assert(offsetof(lmnk_rank1_descriptor_t, u_span) == 8U);
    static_assert(offsetof(lmnk_rank1_descriptor_t, v_span) == 16U);
    static_assert(offsetof(lmnk_rank1_descriptor_t, gain_offset) == 24U);
    static_assert(offsetof(lmnk_rank1_descriptor_t, scale_factor) == 28U);

    static_assert(sizeof(lmnk_config_t) == 64U);
    static_assert(alignof(lmnk_config_t) == 64U);
    static_assert(offsetof(lmnk_config_t, struct_size) == 0U);
    static_assert(offsetof(lmnk_config_t, head_dim) == 4U);
    static_assert(offsetof(lmnk_config_t, guard_hold_tokens) == 48U);
    static_assert(offsetof(lmnk_config_t, max_observe_stride) == 49U);
    static_assert(offsetof(lmnk_config_t, reserved) == 50U);

    static_assert(sizeof(lmnk_abi_info_t) == 64U);
    static_assert(sizeof(lmnk_runtime_policy_t) == 48U);
    static_assert(sizeof(lmnk_turn_begin_params_t) == 16U);
    static_assert(sizeof(lmnk_decode_step_params_t) == 48U);
    static_assert(alignof(lmnk_decode_step_params_t) == 16U);
    static_assert(sizeof(lmnk_decode_step_report_t) == 80U);
    static_assert(sizeof(lmnk_runtime_state_view_t) == 64U);
    static_assert(sizeof(lmnk_lacuna_eval_params_t) == 24U);
    static_assert(sizeof(lmnk_container_verify_report_t) == 40U);

    static_assert(sizeof(lmnk_pointer_trail_t) == 128U);
    static_assert(alignof(lmnk_pointer_trail_t) == 64U);
    static_assert(sizeof(lmnk_gateway_frame_t) == 128U);
    static_assert(offsetof(lmnk_gateway_frame_t, root_cause_code) == 48U);
    static_assert(sizeof(lmnk_causal_report_t) == 16U);
    static_assert(offsetof(lmnk_causal_report_t, root_cause_code) == 12U);
    static_assert(sizeof(lmnk_phase_anchor_record_t) == 128U);
    static_assert(offsetof(lmnk_phase_anchor_record_t, root_cause_code) == 59U);
    static_assert(sizeof(lmnk_liminal_ego_state_t) == 64U);
    static_assert(offsetof(lmnk_liminal_ego_state_t, basin_resistance_q7) == 40U);
    static_assert(offsetof(lmnk_liminal_ego_state_t, escape_floor_q7) == 41U);
    static_assert(offsetof(lmnk_topology_snapshot_t, spawned_anchors) == 31U);
    static_assert(sizeof(lmnk_lacuna_partition_t) == 16U);
    static_assert(LMNK_ERR_TOPOLOGY_MISMATCH == -52);
}

void test_handshake_config_and_inline_tagged_ptr() {
    LMNK_TEST_CHECK(lmnk_abi_version_hex() == LMNK_ABI_VERSION_HEX);

    const char* version_str = lmnk_abi_version_string();
    LMNK_TEST_CHECK(version_str != nullptr);
    LMNK_TEST_CHECK(std::strcmp(version_str, LMNK_ABI_VERSION_STRING) == 0);
    LMNK_TEST_CHECK(std::strcmp(version_str, "0.4.0") == 0);

    const char* ok_str = lmnk_status_to_string(LMNK_OK);
    LMNK_TEST_CHECK(ok_str != nullptr);
    LMNK_TEST_CHECK(std::strcmp(ok_str, "LMNK_OK") == 0);

    // Guard 1: fail-closed struct_size validation on lmnk_abi_info
    lmnk_abi_info_t bad_info{};
    LMNK_TEST_CHECK(lmnk_abi_info(&bad_info) == LMNK_ERR_INVALID_ARGUMENT);
    bad_info.struct_size = UINT32_MAX;
    LMNK_TEST_CHECK(lmnk_abi_info(&bad_info) == LMNK_ERR_INVALID_ARGUMENT);

    lmnk_abi_info_t info{};
    info.struct_size = static_cast<uint32_t>(sizeof(info));
    LMNK_TEST_CHECK(lmnk_abi_info(&info) == LMNK_OK);
    LMNK_TEST_CHECK(info.abi_version_hex == LMNK_ABI_VERSION_HEX);
    LMNK_TEST_CHECK(info.context_size == 5120U);
    LMNK_TEST_CHECK(info.context_align == 64U);
    LMNK_TEST_CHECK(info.snapshot_max_bytes == 512U);
    LMNK_TEST_CHECK(info.topology_crc32 != 0U);
    LMNK_TEST_CHECK((info.capability_flags & LMNK_CAP_SWAR_BASELINE) != 0ULL);
    LMNK_TEST_CHECK((info.capability_flags & LMNK_CAP_ZERO_ALLOC_CORE) != 0ULL);
    LMNK_TEST_CHECK((info.capability_flags & LMNK_CAP_INPLACE_CONTEXT_INIT) != 0ULL);

    lmnk_config_t cfg{};
    LMNK_TEST_CHECK(lmnk_config_init_default(&cfg) == LMNK_OK);
    LMNK_TEST_CHECK(cfg.struct_size == sizeof(lmnk_config_t));
    LMNK_TEST_CHECK(lmnk_config_validate(&cfg) == LMNK_OK);

    lmnk_config_t bad_cfg = cfg;
    bad_cfg.struct_size = 0U;
    LMNK_TEST_CHECK(lmnk_config_validate(&bad_cfg) == LMNK_ERR_INVALID_ARGUMENT);
    bad_cfg = cfg;
    bad_cfg.head_dim = 0U;
    LMNK_TEST_CHECK(lmnk_config_validate(&bad_cfg) == LMNK_ERR_INVALID_ARGUMENT);

    // Header-only static inline tagged-pointer bit-packers
    const uint64_t packed = lmnk_tagged_ptr_pack(0x0000123456789AB0ULL, 0xA5U, -17);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_addr(packed) == 0x0000123456789AB0ULL);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_tag(packed) == 0xA5U);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_phase(packed) == -17);

    const uint64_t act = lmnk_tagged_ptr_pack_action(
        0x0000123456789AB0ULL, LMNK_ATTRACTOR_SLOT_KNOWLEDGE,
        static_cast<uint8_t>(LMNK_ACTION_MODE_PHASE_STEER | LMNK_ACTION_MODE_SUBHEAD_ROUTE), 24);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_addr(act) == 0x0000123456789AB0ULL);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_attractor_slot(act) == LMNK_ATTRACTOR_SLOT_KNOWLEDGE);
    LMNK_TEST_CHECK(
        lmnk_tagged_ptr_unpack_action_mode(act) ==
        static_cast<uint8_t>(LMNK_ACTION_MODE_PHASE_STEER | LMNK_ACTION_MODE_SUBHEAD_ROUTE));
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_phase(act) == 24);
}

uint32_t test_crc32_ieee(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFU;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint32_t>(data[i]);
        for (int b = 0; b < 8; ++b) {
            const uint32_t mask = static_cast<uint32_t>(-static_cast<int32_t>(crc & 1U));
            crc = (crc >> 1) ^ (0xEDB88320U & mask);
        }
    }
    return ~crc;
}

void test_context_lifecycle_policy_decode_and_snapshot() {
    lmnk_abi_info_t info{};
    info.struct_size = static_cast<uint32_t>(sizeof(info));
    LMNK_TEST_CHECK(lmnk_abi_info(&info) == LMNK_OK);

    lmnk_config_t cfg{};
    LMNK_TEST_CHECK(lmnk_config_init_default(&cfg) == LMNK_OK);

    // 1. Boxed lifecycle + policy + turn + decode + abstracted state view + serialize/deserialize
    lmnk_context_handle_t ctx = nullptr;
    LMNK_TEST_CHECK(lmnk_context_create(&cfg, &ctx) == LMNK_OK);
    LMNK_TEST_CHECK(ctx != nullptr);
    LMNK_TEST_CHECK(lmnk_context_deinit_in_place(ctx) == LMNK_ERR_INVALID_STATE);

    lmnk_runtime_policy_t pol{};
    pol.struct_size = static_cast<uint32_t>(sizeof(pol));
    pol.flags = LMNK_POLICY_FLAG_SPINE | LMNK_POLICY_FLAG_COMPUTE_POOL;
    pol.baseline_pinned_compute_mask = 0x0081ULL;
    pol.active_compute_mask = 0x0000FFFFULL;
    pol.repulsion_compute_mask = 0x00000004ULL;
    pol.reversible_compute_mask = 0x0000FFFCULL;
    pol.baseline_pressure_q15 = 8000U;
    pol.guard_hold_tokens = 3U;
    pol.max_observe_stride = 4U;
    pol.pre_bypass_enabled = 1U;
    LMNK_TEST_CHECK(lmnk_context_configure_policy(ctx, &pol) == LMNK_OK);

    lmnk_runtime_state_view_t state{};
    state.struct_size = static_cast<uint32_t>(sizeof(state));
    LMNK_TEST_CHECK(lmnk_context_get_state(ctx, &state) == LMNK_OK);
    LMNK_TEST_CHECK((state.baseline_pinned_compute_mask & 0x0081ULL) == 0x0081ULL);
    LMNK_TEST_CHECK(
        (state.baseline_pinned_compute_mask & LMNK_HOST_IMMUTABLE_PINNED_COMPUTE_MASK) != 0ULL);
    LMNK_TEST_CHECK(state.active_compute_mask == 0x0000FFFFULL);
    LMNK_TEST_CHECK(state.guard_hold_tokens == 3U);
    LMNK_TEST_CHECK(state.max_observe_stride == 4U);

    lmnk_turn_begin_params_t turn{};
    turn.struct_size = static_cast<uint32_t>(sizeof(turn));
    LMNK_TEST_CHECK(lmnk_context_begin_turn(ctx, &turn) == LMNK_OK);

    alignas(16) lmnk_decode_step_params_t step_in{};
    step_in.struct_size = static_cast<uint32_t>(sizeof(step_in));
    step_in.decode_input.pressure.frame_time_pressure_q15 = 8192U;
    step_in.decode_input.observed_exit_entropy_q15 = 9830U;
    step_in.decode_input.exit_entropy_lacuna_code = LMNK_LACUNA_SELF_REF;
    step_in.decode_input.boundary_sensitivity_q7 = 8U;
    step_in.decode_input.min_calm_dwell = 2U;
    step_in.decode_input.flags = LMNK_SPINE_DECODE_INPUT_FLAG_GREEDY_SAMPLING |
                                 LMNK_SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED |
                                 LMNK_SPINE_DECODE_FLAG_REWARD_OVERRIDE;
    step_in.decode_input.exit_reward_override_q7 = 200U;

    lmnk_decode_step_report_t step_rep{};
    step_rep.struct_size = static_cast<uint32_t>(sizeof(step_rep));
    LMNK_TEST_CHECK(lmnk_context_step_decode(ctx, &step_in, &step_rep) == LMNK_OK);
    LMNK_TEST_CHECK((step_rep.dispatch.execute_mask & 0x0081ULL) == 0x0081ULL);
    LMNK_TEST_CHECK(step_rep.next_observe_mode == LMNK_SPINE_LOGITS_OBSERVE_FULL);

    // Serialize -> Deserialize + topology mismatch check
    size_t required_bytes = 0U;
    LMNK_TEST_CHECK(lmnk_context_serialize(ctx, nullptr, &required_bytes) == LMNK_OK);
    LMNK_TEST_CHECK(required_bytes == info.snapshot_max_bytes);
    auto snap = std::make_unique<uint8_t[]>(required_bytes);
    size_t cap = required_bytes;
    LMNK_TEST_CHECK(lmnk_context_serialize(ctx, snap.get(), &cap) == LMNK_OK);

    lmnk_context_handle_t restored = nullptr;
    LMNK_TEST_CHECK(lmnk_context_deserialize(snap.get(), cap, &restored) == LMNK_OK);
    LMNK_TEST_CHECK(restored != nullptr);
    LMNK_TEST_CHECK(lmnk_context_reset(restored) == LMNK_OK);
    LMNK_TEST_CHECK(lmnk_context_destroy(restored) == LMNK_OK);

    snap[8] ^= 0x01U;
    const uint32_t crc = test_crc32_ieee(snap.get(), 60U);
    snap[60] = static_cast<uint8_t>(crc & 0xFFU);
    snap[61] = static_cast<uint8_t>((crc >> 8U) & 0xFFU);
    snap[62] = static_cast<uint8_t>((crc >> 16U) & 0xFFU);
    snap[63] = static_cast<uint8_t>((crc >> 24U) & 0xFFU);
    lmnk_context_handle_t bad_ctx = nullptr;
    LMNK_TEST_CHECK(lmnk_context_deserialize(snap.get(), cap, &bad_ctx) ==
                    LMNK_ERR_TOPOLOGY_MISMATCH);
    LMNK_TEST_CHECK(bad_ctx == nullptr);
    LMNK_TEST_CHECK(lmnk_context_destroy(ctx) == LMNK_OK);

    // 2. Zero-heap in-place lifecycle + zero-scrub
    const size_t ctx_size = static_cast<size_t>(info.context_size);
    const size_t ctx_align = static_cast<size_t>(info.context_align);
    const size_t backing_cap = ctx_size + ctx_align;
    auto backing = std::make_unique<uint8_t[]>(backing_cap);
    void* aligned_ptr = backing.get();
    size_t space = backing_cap;
    LMNK_TEST_CHECK(std::align(ctx_align, ctx_size, aligned_ptr, space) != nullptr);
    auto* ctx_bytes = static_cast<uint8_t*>(aligned_ptr);

    LMNK_TEST_CHECK(lmnk_context_init_in_place(&cfg, ctx_bytes, ctx_size - 1U) ==
                    LMNK_ERR_BUFFER_TOO_SMALL);
    LMNK_TEST_CHECK(lmnk_context_init_in_place(&cfg, ctx_bytes + 1U, ctx_size) ==
                    LMNK_ERR_ALIGNMENT);
    LMNK_TEST_CHECK(lmnk_context_init_in_place(&cfg, ctx_bytes, ctx_size) == LMNK_OK);

    auto in_place = reinterpret_cast<lmnk_context_handle_t>(ctx_bytes);
    LMNK_TEST_CHECK(lmnk_context_destroy(in_place) == LMNK_ERR_INVALID_STATE);
    LMNK_TEST_CHECK(lmnk_context_deinit_in_place(in_place) == LMNK_OK);
    for (size_t i = 0; i < ctx_size; ++i) {
        LMNK_TEST_CHECK(ctx_bytes[i] == 0U);
    }
    LMNK_TEST_CHECK(lmnk_context_deinit_in_place(in_place) == LMNK_ERR_INVALID_STATE);
}

void test_kernels_and_generic_container() {
    // 1. RoPE steer + Procrustes + num_tokens != 1 fail-closed
    alignas(32) float q_orig[8] = {1.0F, 0.25F, 0.5F, -0.5F, 1.0F, 0.25F, 0.5F, -0.5F};
    alignas(32) float q_work[8];
    std::memcpy(q_work, q_orig, sizeof(q_orig));
    alignas(16) const lmnk_q7_t deltas[2] = {16, -16};
    alignas(16) lmnk_q7_t recovered[2] = {0, 0};

    lmnk_rope_steer_params_t steer{};
    steer.struct_size = static_cast<uint32_t>(sizeof(steer));
    steer.head_dim = 4U;
    steer.rotary_dim = 4U;
    steer.num_heads = 2U;
    steer.rope_layout = LMNK_ROPE_LAYOUT_HALF_SPLIT;
    steer.max_rad = 0.12F;
    steer.num_tokens = 2U;
    steer.query_tensor = q_work;
    steer.phase_deltas = deltas;
    LMNK_TEST_CHECK(lmnk_rope_steer_q(&steer) == LMNK_ERR_NOT_IMPLEMENTED);

    steer.num_tokens = 1U;
    LMNK_TEST_CHECK(lmnk_rope_steer_q(&steer) == LMNK_OK);

    lmnk_rope_procrustes_params_t proc{};
    proc.struct_size = static_cast<uint32_t>(sizeof(proc));
    proc.head_dim = 4U;
    proc.rotary_dim = 4U;
    proc.num_heads = 2U;
    proc.rope_layout = LMNK_ROPE_LAYOUT_HALF_SPLIT;
    proc.num_tokens = 1U;
    proc.q_before = q_orig;
    proc.q_after = q_work;
    proc.q7_phase_out = recovered;
    LMNK_TEST_CHECK(lmnk_rope_procrustes_q7(&proc) == LMNK_OK);
    LMNK_TEST_CHECK(recovered[0] > 0 && recovered[1] < 0);

    // 2. Arena-indexed Rank-1 evaluation + overflow fail-closed
    const float arena[8] = {2.0F, 2.0F, 2.0F, 2.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    alignas(32) lmnk_rank1_descriptor_t desc{};
    desc.layer_idx = 0U;
    desc.rank = 1U;
    desc.u_span.offset = 0U;
    desc.u_span.len = 4U;
    desc.v_span.offset = 4U;
    desc.v_span.len = 4U;
    desc.gain_offset = LMNK_RANK1_GAIN_UNUSED;
    desc.scale_factor = 1.0F;

    alignas(32) const float x_in[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    alignas(32) float y_out[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    lmnk_rank1_eval_params_t r1{};
    r1.struct_size = static_cast<uint32_t>(sizeof(r1));
    r1.dim_in = 4U;
    r1.dim_out = 4U;
    r1.arena_len = 8U;
    r1.num_tokens = 1U;
    r1.desc = &desc;
    r1.arena = arena;
    r1.x_in = x_in;
    r1.y_inout = y_out;
    LMNK_TEST_CHECK(lmnk_rank1_eval(&r1) == LMNK_OK);
    for (float v : y_out) {
        LMNK_TEST_CHECK(std::fabs(v - 20.0F) < 1.0e-4F);
    }

    // Overflowing ArenaSpan rejected fail-closed
    alignas(32) lmnk_rank1_descriptor_t bad_desc = desc;
    bad_desc.u_span.offset = UINT32_MAX - 1U;
    bad_desc.u_span.len = 4U;
    r1.desc = &bad_desc;
    LMNK_TEST_CHECK(lmnk_rank1_eval(&r1) == LMNK_ERR_INVALID_ARGUMENT);

    // 3. Gateway steer + Lacuna eval
    alignas(64) lmnk_pointer_trail_t trail{};
    alignas(64) lmnk_gateway_frame_t frame{};
    alignas(16) lmnk_causal_report_t report{};
    lmnk_gateway_steer_params_t gw{};
    gw.struct_size = static_cast<uint32_t>(sizeof(gw));
    gw.head_dim = 4U;
    gw.rotary_dim = 4U;
    gw.num_heads = 2U;
    gw.rope_layout = LMNK_ROPE_LAYOUT_INTERLEAVED;
    gw.unresolved_damping_u7 = 128U;
    gw.fact_energy = 2.0F;
    gw.persona_energy = 1.5F;
    gw.num_tokens = 1U;
    gw.query_tensor = q_work;
    gw.fact_phases = deltas;
    gw.persona_phases = deltas;
    gw.trail_inout = &trail;
    gw.frame_out = &frame;
    gw.report_out = &report;
    LMNK_TEST_CHECK(lmnk_gateway_steer(&gw) == LMNK_OK);
    LMNK_TEST_CHECK(trail.step_count == 1U);

    lmnk_lacuna_eval_params_t lac{};
    lac.struct_size = static_cast<uint32_t>(sizeof(lac));
    lac.self_ref_mask = 0x03U;
    lac.deadband_q7 = 40U;
    const uint8_t densities[8] = {80U, 72U, 0U, 0U, 92U, 88U, 0U, 0U};
    const uint8_t tols[8] = {10U, 10U, 10U, 10U, 10U, 10U, 10U, 10U};
    std::memcpy(lac.other_density_q7, densities, sizeof(densities));
    std::memcpy(lac.noise_tolerance_q7, tols, sizeof(tols));
    alignas(16) lmnk_lacuna_partition_t part{};
    LMNK_TEST_CHECK(lmnk_lacuna_partition_eval(&lac, &part) == LMNK_OK);
    LMNK_TEST_CHECK(part.self_ref_mask == 0x03U);
    LMNK_TEST_CHECK(part.exterior_mask == 0x30U);

    // 4. Generic container write / verify / find_section (including unaligned container read)
    const uint8_t custom_payload[16] = {1U, 2U,  3U,  4U,  5U,  6U,  7U,  8U,
                                        9U, 10U, 11U, 12U, 13U, 14U, 15U, 16U};
    lmnk_section_input_t sec{};
    sec.struct_size = static_cast<uint32_t>(sizeof(sec));
    sec.kind = LMNK_SECTION_RECALL_METADATA;
    sec.flags = 0U;
    sec.payload_len = static_cast<uint32_t>(sizeof(custom_payload));
    sec.payload = custom_payload;

    size_t req_len = 0U;
    lmnk_container_write_params_t wparams{};
    wparams.struct_size = static_cast<uint32_t>(sizeof(wparams));
    wparams.container_flags = LMNK_CONTAINER_FLAG_COMPILED_CACHE;
    wparams.timestamp_epoch_sec = 1774900000ULL;
    wparams.section_count = 1U;
    wparams.sections = &sec;
    wparams.written_bytes_out = &req_len;
    LMNK_TEST_CHECK(lmnk_container_write(&wparams) == LMNK_OK);
    LMNK_TEST_CHECK(req_len >= LMNK_CONTAINER_HEADER_BYTES + sizeof(custom_payload));

    auto container = std::make_unique<uint8_t[]>(req_len);
    size_t written = 0U;
    wparams.out_bytes = container.get();
    wparams.out_capacity = req_len;
    wparams.written_bytes_out = &written;
    LMNK_TEST_CHECK(lmnk_container_write(&wparams) == LMNK_OK);
    LMNK_TEST_CHECK(written == req_len);

    lmnk_container_verify_report_t vrep{};
    vrep.struct_size = static_cast<uint32_t>(sizeof(vrep));
    LMNK_TEST_CHECK(lmnk_container_verify(container.get(), written, &vrep) == LMNK_OK);
    LMNK_TEST_CHECK(vrep.abi_version == LMNK_ABI_VERSION_HEX);
    LMNK_TEST_CHECK(vrep.section_count == 1U);

    lmnk_container_section_view_t sview{};
    sview.struct_size = static_cast<uint32_t>(sizeof(sview));
    LMNK_TEST_CHECK(lmnk_container_find_section(container.get(), written,
                                                LMNK_SECTION_RECALL_METADATA, &sview) == LMNK_OK);
    LMNK_TEST_CHECK(sview.length == sizeof(custom_payload));
    LMNK_TEST_CHECK(sview.payload_ptr != nullptr);
    LMNK_TEST_CHECK(std::memcmp(sview.payload_ptr, custom_payload, sizeof(custom_payload)) == 0);

    lmnk_container_section_view_t missing{};
    missing.struct_size = static_cast<uint32_t>(sizeof(missing));
    LMNK_TEST_CHECK(lmnk_container_find_section(container.get(), written, LMNK_SECTION_BLOOM_FILTER,
                                                &missing) == LMNK_ERR_ANCHOR_NOT_FOUND);
}

} // namespace

int main() {
    test_static_invariants();
    test_handshake_config_and_inline_tagged_ptr();
    test_context_lifecycle_policy_decode_and_snapshot();
    test_kernels_and_generic_container();
    return 0;
}
