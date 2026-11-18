// name: tests/ffi/test_c_abi_smoke.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "liminika/gcso_abi.h"
#include "liminika/gcso_config.h"
#include "liminika/gcso_types.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define GCSO_TEST_CHECK(cond)                                                                      \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "CTest assertion failed at %s:%d: %s\n", __FILE__, __LINE__,      \
                         #cond);                                                                   \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

namespace {

constexpr void check_static_abi_invariants() noexcept {
    static_assert(GCSO_ABI_VERSION_MAJOR == 0U);
    static_assert(GCSO_ABI_VERSION_MINOR == 1U);
    static_assert(GCSO_ABI_VERSION_PATCH == 2U);
    static_assert(GCSO_ABI_VERSION_HEX == 0x00000102U);
    static_assert(GCSO_CRYSTAL_RECORD_VERSION == GCSO_ABI_VERSION_HEX);
    static_assert(sizeof(gcso_liminal_ego_field_t) == 64U);
    static_assert(alignof(gcso_liminal_ego_field_t) == 64U);
    static_assert(offsetof(gcso_liminal_ego_field_t, counter_basin_thrust_q7) == 40U);
    static_assert(offsetof(gcso_liminal_ego_field_t, basin_escape_floor_q7) == 41U);
}

void test_version_and_capabilities() {
    uint32_t major = 99U;
    uint32_t minor = 99U;
    uint32_t patch = 99U;
    GCSO_TEST_CHECK(gcso_abi_get_version(&major, &minor, &patch) == GCSO_SUCCESS);
    GCSO_TEST_CHECK(major == GCSO_ABI_VERSION_MAJOR);
    GCSO_TEST_CHECK(minor == GCSO_ABI_VERSION_MINOR);
    GCSO_TEST_CHECK(patch == GCSO_ABI_VERSION_PATCH);

    const char* version_str = gcso_abi_get_version_string();
    GCSO_TEST_CHECK(version_str != nullptr);
    GCSO_TEST_CHECK(std::strcmp(version_str, GCSO_ABI_VERSION_STRING) == 0);
    GCSO_TEST_CHECK(std::strcmp(version_str, "0.1.2") == 0);

    const char* ok_str = gcso_status_to_string(GCSO_SUCCESS);
    GCSO_TEST_CHECK(ok_str != nullptr);
    GCSO_TEST_CHECK(std::strcmp(ok_str, "GCSO_SUCCESS") == 0);

    gcso_capability_flags_t caps = 0ULL;
    GCSO_TEST_CHECK(gcso_abi_query_capability(&caps) == GCSO_SUCCESS);
    GCSO_TEST_CHECK((caps & GCSO_CAP_CPU_BASELINE) != 0ULL);
}

void test_config_and_tagged_pointer() {
    gcso_config_t cfg {};
    GCSO_TEST_CHECK(gcso_config_init_default(&cfg) == GCSO_SUCCESS);
    GCSO_TEST_CHECK(gcso_config_validate(&cfg) == GCSO_SUCCESS);

    gcso_config_t bad_cfg = cfg;
    bad_cfg.head_dim = 3U; // Must be non-zero even
    GCSO_TEST_CHECK(gcso_config_validate(&bad_cfg) == GCSO_ERROR_INVALID_ARGUMENT);

    const uint64_t packed = gcso_tagged_ptr_pack_action(
        0x0000123456789AB0ULL, GCSO_ATTRACTOR_SLOT_KNOWLEDGE,
        static_cast<uint8_t>(GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_SUBHEAD_ROUTE), 24);
    GCSO_TEST_CHECK(gcso_tagged_ptr_unpack_addr(packed) == 0x0000123456789AB0ULL);
    GCSO_TEST_CHECK(gcso_tagged_ptr_unpack_attractor_slot(packed) == GCSO_ATTRACTOR_SLOT_KNOWLEDGE);
    GCSO_TEST_CHECK(gcso_tagged_ptr_unpack_phase(packed) == 24);

    const uint64_t repulsed = gcso_tagged_ptr_evolve_mode_stigmergically(
        packed, 0.90F, 0.45F, 1U, GCSO_ATTRACTOR_SLOT_KNOWLEDGE, 0.15F, 0.85F);
    GCSO_TEST_CHECK(gcso_tagged_ptr_unpack_attractor_slot(repulsed) == GCSO_ATTRACTOR_SLOT_REPULSION);
    GCSO_TEST_CHECK(gcso_tagged_ptr_unpack_phase(repulsed) == -24);
}

void test_context_step_and_snapshot_roundtrip() {
    gcso_config_t cfg {};
    GCSO_TEST_CHECK(gcso_config_init_default(&cfg) == GCSO_SUCCESS);
    cfg.head_dim = 8U;
    cfg.num_heads = 4U;

    gcso_context_handle_t ctx = nullptr;
    GCSO_TEST_CHECK(gcso_context_create(&cfg, &ctx) == GCSO_SUCCESS);
    GCSO_TEST_CHECK(ctx != nullptr);

    GCSO_TEST_CHECK(gcso_context_set_system_prompt_anchor(ctx, "Maintain factual invariance.",
                                                          1.0F) == GCSO_SUCCESS);

    alignas(32) float query[32] {};
    alignas(32) float key[32] {};
    for (size_t i = 0; i < 32; ++i) {
        query[i] = 0.5F + 0.01F * static_cast<float>(i);
        key[i] = 0.25F - 0.005F * static_cast<float>(i);
    }

    alignas(128) gcso_pointer_trail_t trail {};
    GCSO_TEST_CHECK(gcso_context_step_token(ctx, 42U, query, key, &trail) == GCSO_SUCCESS);
    GCSO_TEST_CHECK(trail.current_ptr == 42U);
    GCSO_TEST_CHECK(trail.step_count == 1U);

    size_t required_bytes = 0U;
    GCSO_TEST_CHECK(gcso_context_serialize(ctx, nullptr, &required_bytes) == GCSO_SUCCESS);
    GCSO_TEST_CHECK(required_bytes > sizeof(gcso_snapshot_header_t));

    // Reserve aligned buffer matching alignof(gcso_snapshot_header_t)
    struct alignas(64) AlignedBlock {
        uint8_t bytes[64];
    };
    const size_t block_count = (required_bytes + 63U) / 64U;
    std::vector<AlignedBlock> snap_buf(block_count);
    auto* raw_ptr = reinterpret_cast<uint8_t*>(snap_buf.data());
    size_t cap = block_count * 64U;
    GCSO_TEST_CHECK(gcso_context_serialize(ctx, raw_ptr, &cap) == GCSO_SUCCESS);

    gcso_context_handle_t restored = nullptr;
    GCSO_TEST_CHECK(gcso_context_deserialize(raw_ptr, cap, &restored) == GCSO_SUCCESS);
    GCSO_TEST_CHECK(restored != nullptr);

    alignas(128) gcso_pointer_trail_t restored_trail {};
    GCSO_TEST_CHECK(gcso_context_step_token(restored, 43U, query, key, &restored_trail) ==
                    GCSO_SUCCESS);
    GCSO_TEST_CHECK(restored_trail.step_count == 2U);
    GCSO_TEST_CHECK(restored_trail.prev_ptr == 42U);

    GCSO_TEST_CHECK(gcso_context_destroy(restored) == GCSO_SUCCESS);
    GCSO_TEST_CHECK(gcso_context_destroy(ctx) == GCSO_SUCCESS);
}

void test_edbc_controller_and_ego_pod() {
    alignas(32) gcso_edbc_state_t init_state {};
    init_state.bifurcation_threshold = 0.50F;
    init_state.singularity_eps = 1e-12F;
    init_state.repulsion_gain = 0.25F;
    init_state.sample_temperature = 1.0F;

    gcso_edbc_controller_handle_t edbc = nullptr;
    GCSO_TEST_CHECK(gcso_edbc_controller_create(&init_state, &edbc) == GCSO_SUCCESS);
    GCSO_TEST_CHECK(edbc != nullptr);

    alignas(32) gcso_edbc_state_t out_state {};
    GCSO_TEST_CHECK(gcso_edbc_eval_stateful(edbc, 1.5F, &out_state) == GCSO_SUCCESS);
    GCSO_TEST_CHECK(out_state.sample_temperature > 0.0F);
    GCSO_TEST_CHECK(gcso_edbc_controller_destroy(edbc) == GCSO_SUCCESS);

    alignas(64) gcso_liminal_ego_field_t ego {};
    ego.counter_basin_thrust_q7 = 48;
    ego.basin_escape_floor_q7 = 16;
    ego.trigger_mask = GCSO_EGO_TRIGGER_BASIN_CAPTURE;
    GCSO_TEST_CHECK((ego.trigger_mask & GCSO_EGO_TRIGGER_BASIN_CAPTURE) != 0U);
}

} // namespace

int main() {
    check_static_abi_invariants();
    test_version_and_capabilities();
    test_config_and_tagged_pointer();
    test_context_step_and_snapshot_roundtrip();
    test_edbc_controller_and_ego_pod();
    return 0;
}
