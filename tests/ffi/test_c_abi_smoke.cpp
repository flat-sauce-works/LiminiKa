// name: tests/ffi/test_c_abi_smoke.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "liminika/lmnk_abi.h"
#include "liminika/lmnk_config.h"
#include "liminika/lmnk_types.h"

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
    static_assert(LMNK_ABI_VERSION_MINOR == 2U);
    static_assert(LMNK_ABI_VERSION_PATCH == 0U);
    static_assert(LMNK_ABI_VERSION_HEX == 0x00000200U);
    static_assert(LMNK_PHASE_ANCHOR_MAGIC == 0x524B4E4CU);
    static_assert(LMNK_PHASE_ANCHOR_VERSION == LMNK_ABI_VERSION_HEX);
    static_assert(sizeof(lmnk_config_t) == 64U);
    static_assert(alignof(lmnk_config_t) == 64U);
    static_assert(sizeof(lmnk_pointer_trail_t) == 128U);
    static_assert(alignof(lmnk_pointer_trail_t) == 64U);
    static_assert(sizeof(lmnk_entropy_state_t) == 32U);
    static_assert(alignof(lmnk_entropy_state_t) == 32U);
    static_assert(sizeof(lmnk_liminal_ego_state_t) == 64U);
    static_assert(alignof(lmnk_liminal_ego_state_t) == 64U);
    static_assert(sizeof(lmnk_core_primitive_state_t) == 64U);
    static_assert(alignof(lmnk_core_primitive_state_t) == 64U);
    static_assert(sizeof(lmnk_core_state_stack_t) == 128U);
    static_assert(alignof(lmnk_core_state_stack_t) == 64U);
    static_assert(sizeof(lmnk_lacuna_partition_t) == 16U);
    static_assert(alignof(lmnk_lacuna_partition_t) == 16U);
    static_assert(sizeof(lmnk_spine_state_t) == 32U);
    static_assert(sizeof(lmnk_distortion_delta_t) == 128U);
    static_assert(alignof(lmnk_distortion_delta_t) == 64U);
    static_assert(sizeof(lmnk_distortion_tension_t) == 32U);
    static_assert(sizeof(lmnk_phase_anchor_record_t) == 128U);
    static_assert(alignof(lmnk_phase_anchor_record_t) == 64U);
    static_assert(LMNK_ATTRIBUTION_TAG(LMNK_PRIMITIVE_STIGMERGY, LMNK_DRIVER_EGO) == 3U);
    static_assert(LMNK_ATTRIBUTION_TAG(LMNK_PRIMITIVE_STIGMERGY, LMNK_DRIVER_SYSTEM) == 1U);
    static_assert(LMNK_ATTRIBUTION_TAG(LMNK_PRIMITIVE_LIMINAL, LMNK_DRIVER_EGO) == 2U);
    static_assert(sizeof(lmnk_container_header_t) == 64U);
    static_assert(alignof(lmnk_container_header_t) == 64U);
    static_assert(sizeof(lmnk_section_entry_t) == 16U);
    static_assert(alignof(lmnk_section_entry_t) == 16U);
    static_assert(sizeof(lmnk_topology_matrix_t) == 64U);
    static_assert(alignof(lmnk_topology_matrix_t) == 64U);
    static_assert(LMNK_ERR_TOPOLOGY_MISMATCH == -52);
    static_assert(offsetof(lmnk_liminal_ego_state_t, counter_basin_thrust_q7) == 40U);
    static_assert(offsetof(lmnk_liminal_ego_state_t, basin_escape_floor_q7) == 41U);
    static_assert(LMNK_ATTRACTOR_SLOT_TEMPORAL == 5U);
    static_assert(LMNK_ATTRACTOR_SLOT_USER_INTENT == 6U);
    static_assert(LMNK_ATTRACTOR_SLOT_CRYSTALLIZED == 7U);
}

void test_version_capabilities_and_tagged_ptr() {
    LMNK_TEST_CHECK(lmnk_abi_version_major() == LMNK_ABI_VERSION_MAJOR);
    LMNK_TEST_CHECK(lmnk_abi_version_minor() == LMNK_ABI_VERSION_MINOR);
    LMNK_TEST_CHECK(lmnk_abi_version_patch() == LMNK_ABI_VERSION_PATCH);
    LMNK_TEST_CHECK(lmnk_abi_version_hex() == LMNK_ABI_VERSION_HEX);

    const char* version_str = lmnk_abi_version_string();
    LMNK_TEST_CHECK(version_str != nullptr);
    LMNK_TEST_CHECK(std::strcmp(version_str, LMNK_ABI_VERSION_STRING) == 0);
    LMNK_TEST_CHECK(std::strcmp(version_str, "0.2.0") == 0);

    const char* ok_str = lmnk_status_to_string(LMNK_OK);
    LMNK_TEST_CHECK(ok_str != nullptr);
    LMNK_TEST_CHECK(std::strcmp(ok_str, "LMNK_OK") == 0);

    lmnk_capability_flags_t caps = 0ULL;
    LMNK_TEST_CHECK(lmnk_abi_query_capability(&caps) == LMNK_OK);
    const lmnk_capability_flags_t known_caps =
        LMNK_CAP_SWAR_BASELINE | LMNK_CAP_SIMD_AVX2 | LMNK_CAP_SIMD_NEON | LMNK_CAP_SIMD_WASM128 |
        LMNK_CAP_ZERO_ALLOC_CORE | LMNK_CAP_INPLACE_CONTEXT_INIT;
    LMNK_TEST_CHECK((caps & LMNK_CAP_SWAR_BASELINE) != 0ULL);
    LMNK_TEST_CHECK((caps & LMNK_CAP_ZERO_ALLOC_CORE) != 0ULL);
    LMNK_TEST_CHECK((caps & LMNK_CAP_INPLACE_CONTEXT_INIT) != 0ULL);
    LMNK_TEST_CHECK((caps & ~known_caps) == 0ULL);

    lmnk_config_t cfg{};
    LMNK_TEST_CHECK(lmnk_config_init_default(&cfg) == LMNK_OK);
    LMNK_TEST_CHECK(lmnk_config_validate(&cfg) == LMNK_OK);

    lmnk_config_t bad_cfg = cfg;
    bad_cfg.head_dim = 0U;
    LMNK_TEST_CHECK(lmnk_config_validate(&bad_cfg) == LMNK_ERR_INVALID_ARGUMENT);

    const uint64_t packed = lmnk_tagged_ptr_pack_action(
        0x0000123456789AB0ULL, LMNK_ATTRACTOR_SLOT_KNOWLEDGE,
        static_cast<uint8_t>(LMNK_ACTION_MODE_PHASE_STEER | LMNK_ACTION_MODE_SUBHEAD_ROUTE), 24);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_addr(packed) == 0x0000123456789AB0ULL);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_attractor_slot(packed) == LMNK_ATTRACTOR_SLOT_KNOWLEDGE);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_phase(packed) == 24);

    const uint64_t repulsed = lmnk_tagged_ptr_evolve_mode(
        packed, 0.90F, 0.45F, 1U, LMNK_ATTRACTOR_SLOT_KNOWLEDGE, 0.15F, 0.85F);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_attractor_slot(repulsed) ==
                    LMNK_ATTRACTOR_SLOT_REPULSION);
    LMNK_TEST_CHECK(lmnk_tagged_ptr_unpack_phase(repulsed) == -24);
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

void test_context_boxed_and_in_place_lifecycle() {
    lmnk_config_t cfg{};
    LMNK_TEST_CHECK(lmnk_config_init_default(&cfg) == LMNK_OK);

    const size_t ctx_size = lmnk_context_size();
    const size_t ctx_align = lmnk_context_align();
    LMNK_TEST_CHECK(ctx_size > 0U && (ctx_size % 64U) == 0U);
    LMNK_TEST_CHECK(ctx_align == 64U);

    alignas(64) float query[4096]{};
    alignas(64) float key[4096]{};
    for (size_t i = 0; i < 4096U; ++i) {
        query[i] = 0.25F * static_cast<float>((i % 7U) + 1U);
        key[i] = 0.50F * static_cast<float>((i % 5U) + 1U);
    }

    // 1. Boxed context create / serialize / deserialize / cross-guard check
    lmnk_context_handle_t ctx = nullptr;
    LMNK_TEST_CHECK(lmnk_context_create(&cfg, &ctx) == LMNK_OK);
    LMNK_TEST_CHECK(ctx != nullptr);
    LMNK_TEST_CHECK(lmnk_context_deinit_in_place(ctx) == LMNK_ERR_INVALID_STATE);

    alignas(64) lmnk_pointer_trail_t trail{};
    LMNK_TEST_CHECK(lmnk_context_step_token(ctx, 42U, query, key, &trail) == LMNK_OK);
    LMNK_TEST_CHECK(trail.current_ptr == 42U);
    LMNK_TEST_CHECK(trail.step_count == 1U);

    size_t required_bytes = 0;
    LMNK_TEST_CHECK(lmnk_context_serialize(ctx, nullptr, &required_bytes) == LMNK_OK);
    LMNK_TEST_CHECK(required_bytes > sizeof(lmnk_container_header_t));

    const size_t raw_cap = required_bytes + alignof(lmnk_container_header_t);
    auto raw_storage = std::make_unique<uint8_t[]>(raw_cap);
    void* aligned_ptr = raw_storage.get();
    size_t space = raw_cap;
    LMNK_TEST_CHECK(std::align(alignof(lmnk_container_header_t), required_bytes, aligned_ptr,
                               space) != nullptr);
    auto* raw_ptr = static_cast<uint8_t*>(aligned_ptr);
    size_t cap = required_bytes;
    LMNK_TEST_CHECK(lmnk_context_serialize(ctx, raw_ptr, &cap) == LMNK_OK);

    lmnk_context_handle_t restored = nullptr;
    LMNK_TEST_CHECK(lmnk_context_deserialize(raw_ptr, cap, &restored) == LMNK_OK);
    LMNK_TEST_CHECK(restored != nullptr);

    alignas(64) lmnk_pointer_trail_t restored_trail{};
    LMNK_TEST_CHECK(lmnk_context_step_token(restored, 43U, query, key, &restored_trail) == LMNK_OK);
    LMNK_TEST_CHECK(restored_trail.step_count == 2U);
    LMNK_TEST_CHECK(restored_trail.prev_ptr == 42U);

    // Mutate topology_crc32 and re-stamp header_crc32 to trigger LMNK_ERR_TOPOLOGY_MISMATCH
    raw_ptr[8] ^= 0x01U;
    const uint32_t restamped_crc = test_crc32_ieee(raw_ptr, 60U);
    raw_ptr[60] = static_cast<uint8_t>(restamped_crc & 0xFFU);
    raw_ptr[61] = static_cast<uint8_t>((restamped_crc >> 8U) & 0xFFU);
    raw_ptr[62] = static_cast<uint8_t>((restamped_crc >> 16U) & 0xFFU);
    raw_ptr[63] = static_cast<uint8_t>((restamped_crc >> 24U) & 0xFFU);
    lmnk_context_handle_t mismatch_ctx = nullptr;
    LMNK_TEST_CHECK(lmnk_context_deserialize(raw_ptr, cap, &mismatch_ctx) ==
                    LMNK_ERR_TOPOLOGY_MISMATCH);
    LMNK_TEST_CHECK(mismatch_ctx == nullptr);

    LMNK_TEST_CHECK(lmnk_context_destroy(restored) == LMNK_OK);
    LMNK_TEST_CHECK(lmnk_context_destroy(ctx) == LMNK_OK);

    // 2. Zero-heap caller-owned in-place lifecycle + zero-scrub + cross-guard check
    const size_t inplace_cap = ctx_size + ctx_align;
    auto backing = std::make_unique<uint8_t[]>(inplace_cap);
    void* ctx_aligned = backing.get();
    size_t ctx_space = inplace_cap;
    LMNK_TEST_CHECK(std::align(ctx_align, ctx_size, ctx_aligned, ctx_space) != nullptr);
    auto* ctx_bytes = static_cast<uint8_t*>(ctx_aligned);

    LMNK_TEST_CHECK(lmnk_context_init_in_place(&cfg, ctx_bytes, ctx_size - 1U) ==
                    LMNK_ERR_BUFFER_TOO_SMALL);
    LMNK_TEST_CHECK(lmnk_context_init_in_place(&cfg, ctx_bytes + 1U, ctx_size) ==
                    LMNK_ERR_ALIGNMENT);
    LMNK_TEST_CHECK(lmnk_context_init_in_place(&cfg, ctx_bytes, ctx_size) == LMNK_OK);

    auto in_place_ctx = reinterpret_cast<lmnk_context_handle_t>(ctx_bytes);
    LMNK_TEST_CHECK(lmnk_context_destroy(in_place_ctx) == LMNK_ERR_INVALID_STATE);

    alignas(64) lmnk_pointer_trail_t inplace_trail{};
    LMNK_TEST_CHECK(lmnk_context_step_token(in_place_ctx, 99U, query, key, &inplace_trail) ==
                    LMNK_OK);
    LMNK_TEST_CHECK(inplace_trail.current_ptr == 99U);
    LMNK_TEST_CHECK(inplace_trail.step_count == 1U);
    LMNK_TEST_CHECK(lmnk_context_reset(in_place_ctx) == LMNK_OK);
    LMNK_TEST_CHECK(lmnk_context_deinit_in_place(in_place_ctx) == LMNK_OK);
    for (size_t i = 0; i < ctx_size; ++i) {
        LMNK_TEST_CHECK(ctx_bytes[i] == 0U);
    }
    LMNK_TEST_CHECK(lmnk_context_deinit_in_place(in_place_ctx) == LMNK_ERR_INVALID_STATE);
}

void test_entropy_lacuna_anchor_and_mermaid() {
    alignas(32) lmnk_entropy_state_t init_state{};
    init_state.moving_z_entropy = 0.0F;
    init_state.bifurcation_threshold = 0.65F;
    init_state.singularity_eps = 1.0e-12F;
    init_state.sliding_entropy_rate = 0.0F;
    init_state.repulsion_gain = 1.0F;
    init_state.sample_temperature = 1.0F;

    lmnk_entropy_controller_handle_t entropy = nullptr;
    LMNK_TEST_CHECK(lmnk_entropy_controller_create(&init_state, &entropy) == LMNK_OK);
    LMNK_TEST_CHECK(entropy != nullptr);

    alignas(32) lmnk_entropy_state_t out_state{};
    LMNK_TEST_CHECK(lmnk_entropy_eval_stateful(entropy, 1.5F, &out_state) == LMNK_OK);
    LMNK_TEST_CHECK(out_state.sample_temperature > 0.0F);
    LMNK_TEST_CHECK(lmnk_entropy_controller_destroy(entropy) == LMNK_OK);

    alignas(64) lmnk_liminal_ego_state_t ego{};
    LMNK_TEST_CHECK(lmnk_ego_state_init_preset(LMNK_EGO_PRESET_MEDIUM, &ego) == LMNK_OK);
    ego.trigger_mask = LMNK_EGO_TRIGGER_BASIN_CAPTURE;
    LMNK_TEST_CHECK((ego.trigger_mask & LMNK_EGO_TRIGGER_BASIN_CAPTURE) != 0U);

    const uint8_t other_density[8] = {80U, 72U, 0U, 0U, 92U, 88U, 0U, 0U};
    const uint8_t noise_tol[8] = {10U, 10U, 10U, 10U, 10U, 10U, 10U, 10U};
    alignas(16) lmnk_lacuna_partition_t part{};
    LMNK_TEST_CHECK(lmnk_liminal_evaluate_lacuna_partition(0x03U, other_density, noise_tol, 40U,
                                                           &part) == LMNK_OK);
    LMNK_TEST_CHECK(part.self_ref_mask == 0x03U);
    LMNK_TEST_CHECK(part.exterior_mask == 0x30U);
    LMNK_TEST_CHECK(part.boundary_sensitivity_q7 > 0U);

    alignas(16) lmnk_distortion_tension_t tension{};
    tension.deadband_low_q7 = 16U;
    tension.deadband_high_q7 = 96U;
    tension.effective_ego_scale_q7 = 64U;
    LMNK_TEST_CHECK(lmnk_distortion_tension_step_hot_path(&tension, 110U, 32U, 96U, 96U, 1U) ==
                    LMNK_OK);
    LMNK_TEST_CHECK(tension.effective_ego_scale_q7 == 0U);
    LMNK_TEST_CHECK(tension.immediate_damping_events == 1U);

    // Online toroidal k-Means with "LNKR" PhaseAnchorRecord
    alignas(64) lmnk_phase_anchor_record_t centroid{};
    centroid.magic = LMNK_PHASE_ANCHOR_MAGIC;
    centroid.version = LMNK_PHASE_ANCHOR_VERSION;
    centroid.anchor_id = 7U;
    centroid.dominant_slot = LMNK_ATTRACTOR_SLOT_KNOWLEDGE;

    alignas(64) lmnk_gateway_frame_t frame{};
    frame.target_anchor_id = 7U;
    frame.effective_energy = 1.5F;
    frame.dominant_slot = LMNK_ATTRACTOR_SLOT_KNOWLEDGE;
    for (size_t i = 0; i < 64U; ++i) {
        frame.phase_deltas[i] = 16;
    }

    alignas(64) lmnk_pointer_trail_t kmeans_trail{};
    size_t assigned_idx = 99U;
    float assigned_coh = 0.0F;
    float updated_mass = 0.0F;
    LMNK_TEST_CHECK(lmnk_anchor_step_online_toroidal_kmeans(
                        &centroid, 1U, &frame, 0.35F, &kmeans_trail, &assigned_idx, &assigned_coh,
                        &updated_mass) == LMNK_OK);
    LMNK_TEST_CHECK(assigned_idx == 0U);
    LMNK_TEST_CHECK(centroid.sample_count >= 1U);

    // Two-pass Mermaid diagram sizing and zero-alloc formatting
    alignas(64) lmnk_core_state_stack_t stack{};
    LMNK_TEST_CHECK(lmnk_core_stack_init_default(&stack) == LMNK_OK);
    alignas(16) lmnk_epistemic_horizon_t horizon{};
    alignas(16) lmnk_overlap_discovery_report_t overlap{};

    size_t mermaid_len = 0U;
    LMNK_TEST_CHECK(lmnk_topology_format_mermaid(&stack, &horizon, &overlap, 3U, 1U, nullptr, 0U,
                                                 &mermaid_len) == LMNK_OK);
    LMNK_TEST_CHECK(mermaid_len > 64U);

    auto mermaid_buf = std::make_unique<uint8_t[]>(mermaid_len + 1U);
    size_t actual_written = 0U;
    LMNK_TEST_CHECK(lmnk_topology_format_mermaid(&stack, &horizon, &overlap, 3U, 1U,
                                                 mermaid_buf.get(), mermaid_len,
                                                 &actual_written) == LMNK_OK);
    LMNK_TEST_CHECK(actual_written == mermaid_len);
    mermaid_buf[actual_written] = 0U;
    const char* mermaid_str = reinterpret_cast<const char*>(mermaid_buf.get());
    LMNK_TEST_CHECK(std::strncmp(mermaid_str, "flowchart TD\n", 13) == 0);
    LMNK_TEST_CHECK(std::strstr(mermaid_str, "Anchors:3") != nullptr);

    // Topology matrix & unified zero-copy container write/verify/alignment rejection
    alignas(64) lmnk_topology_matrix_t topo{};
    LMNK_TEST_CHECK(lmnk_topology_get_matrix(&topo) == LMNK_OK);
    LMNK_TEST_CHECK(topo.magic == LMNK_TOPOLOGY_MAGIC);
    LMNK_TEST_CHECK(topo.layer_count == LMNK_TOPOLOGY_LAYER_COUNT);
    LMNK_TEST_CHECK(topo.channel_count == LMNK_TOPOLOGY_CHANNEL_COUNT);
    LMNK_TEST_CHECK(topo.pinned_channel_mask == LMNK_TOPOLOGY_PINNED_CHANNEL_MASK);
    LMNK_TEST_CHECK(topo.topology_crc32 != 0U);

    alignas(64) uint8_t container_buf[512]{};
    size_t cont_len = sizeof(container_buf);
    LMNK_TEST_CHECK(lmnk_container_write_zero_copy(&centroid, 1U, nullptr, 0U, nullptr, nullptr,
                                                  1774900000ULL, container_buf,
                                                  &cont_len) == LMNK_OK);
    LMNK_TEST_CHECK(cont_len >= LMNK_CONTAINER_HEADER_BYTES + 128U);

    size_t num_anchors = 0U;
    size_t num_trails = 0U;
    const lmnk_phase_anchor_record_t* anchors_ptr = nullptr;
    const lmnk_pointer_trail_t* trails_ptr = nullptr;
    LMNK_TEST_CHECK(lmnk_container_verify_zero_copy(container_buf, cont_len, &num_anchors,
                                                   &num_trails, &anchors_ptr,
                                                   &trails_ptr) == LMNK_OK);
    LMNK_TEST_CHECK(num_anchors == 1U && anchors_ptr != nullptr);
    LMNK_TEST_CHECK(num_trails == 0U && trails_ptr == nullptr);

    alignas(64) uint8_t misalign_backing[512 + 8]{};
    std::memcpy(misalign_backing + 1, container_buf, cont_len);
    LMNK_TEST_CHECK(lmnk_container_verify_zero_copy(misalign_backing + 1, cont_len, &num_anchors,
                                                   &num_trails, &anchors_ptr,
                                                   &trails_ptr) == LMNK_ERR_ALIGNMENT);
}

} // namespace

int main() {
    test_static_invariants();
    test_version_capabilities_and_tagged_ptr();
    test_context_boxed_and_in_place_lifecycle();
    test_entropy_lacuna_anchor_and_mermaid();
    return 0;
}
