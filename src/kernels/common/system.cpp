// name: src/kernels/common/system.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "gcso_internal.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace {

GCSO_FORCE_INLINE uint8_t popcount_u8_branchless(uint8_t x) noexcept {
    uint8_t v = x;
    v = static_cast<uint8_t>(v - ((v >> 1U) & 0x55U));
    v = static_cast<uint8_t>((v & 0x33U) + ((v >> 2U) & 0x33U));
    return static_cast<uint8_t>((v + (v >> 4U)) & 0x0FU);
}

GCSO_FORCE_INLINE uint8_t select_u8(bool cond, uint8_t t, uint8_t f) noexcept {
    const auto mask = static_cast<uint8_t>(0U - static_cast<uint8_t>(cond ? 1U : 0U));
    return static_cast<uint8_t>((t & mask) | (f & static_cast<uint8_t>(~mask)));
}

GCSO_FORCE_INLINE uint32_t select_u32(bool cond, uint32_t t, uint32_t f) noexcept {
    const uint32_t mask = 0U - static_cast<uint32_t>(cond ? 1U : 0U);
    return (t & mask) | (f & ~mask);
}

GCSO_FORCE_INLINE bool byte_ranges_overlap(uintptr_t a, size_t a_len, uintptr_t b,
                                           size_t b_len) noexcept {
    if (a_len == 0U || b_len == 0U) {
        return false;
    }
    const uintptr_t a_end = a + a_len;
    const uintptr_t b_end = b + b_len;
    return (a < b_end) && (b < a_end);
}

GCSO_FORCE_INLINE uint64_t load_q7_word8(const gcso_q7_t* ptr) noexcept {
    uint64_t w = 0ULL;
    std::memcpy(&w, ptr, sizeof(uint64_t));
    return w;
}

GCSO_FORCE_INLINE void store_q7_word8(gcso_q7_t* ptr, uint64_t w) noexcept {
    std::memcpy(ptr, &w, sizeof(uint64_t));
}

GCSO_FORCE_INLINE uint64_t swar_scale_le128_with_bias_x8(uint64_t packed, uint64_t s,
                                                         uint64_t bias_restore) noexcept {
    constexpr uint64_t SIGN_MASK = 0x8080808080808080ULL;
    constexpr uint64_t EVEN_MASK = 0x00FF00FF00FF00FFULL;
    constexpr uint64_t ODD_MASK = 0xFF00FF00FF00FF00ULL;
    const uint64_t u = packed ^ SIGN_MASK;
    const uint64_t u_even = u & EVEN_MASK;
    const uint64_t u_odd = (u >> 8U) & EVEN_MASK;
    const uint64_t scaled_even = ((u_even * s) >> 7U) & EVEN_MASK;
    const uint64_t scaled_odd = ((u_odd * s) << 1U) & ODD_MASK;
    return ((scaled_even | scaled_odd) + bias_restore) ^ SIGN_MASK;
}

GCSO_FORCE_INLINE uint64_t swar_circular_add_x8(uint64_t a_packed, uint64_t b_packed) noexcept {
    constexpr uint64_t LOW7_MASK = 0x7F7F7F7F7F7F7F7FULL;
    constexpr uint64_t SIGN_MASK = 0x8080808080808080ULL;
    const uint64_t sum_low7 = (a_packed & LOW7_MASK) + (b_packed & LOW7_MASK);
    return sum_low7 ^ ((a_packed ^ b_packed) & SIGN_MASK);
}

GCSO_FORCE_INLINE gcso_q7_t mean_q7_from_64(const gcso_q7_t* arr) noexcept {
    int32_t total = 0;
    for (size_t i = 0U; i < GCSO_Q7_PHASE_ARRAY_LEN; ++i) {
        total += static_cast<int32_t>(arr[i]);
    }
    const int32_t bias = (total >= 0) ? 32 : -32;
    int32_t rounded = (total + bias) / 64;
    if (rounded < -128) {
        rounded = -128;
    } else if (rounded > 127) {
        rounded = 127;
    }
    return static_cast<gcso_q7_t>(rounded);
}

} // namespace

GCSO_EXTERN_C_BEGIN

/**
 * @brief Packs a 48-bit canonical address, 8-bit tag, and Q7 phase offset into a 64-bit tagged
 * pointer.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_pack(uint64_t raw_addr, uint8_t tag,
                                                                gcso_q7_t phase) GCSO_NOEXCEPT {
    return gcso_tagged_ptr_pack_inline(raw_addr, tag, phase);
}

/**
 * @brief Unpacks the 48-bit canonical address from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_unpack_addr(uint64_t ptr) GCSO_NOEXCEPT {
    return gcso_tagged_ptr_unpack_addr_inline(ptr);
}

/**
 * @brief Unpacks the 8-bit domain/action tag from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint8_t GCSO_CALL gcso_tagged_ptr_unpack_tag(uint64_t ptr) GCSO_NOEXCEPT {
    return gcso_tagged_ptr_unpack_tag_inline(ptr);
}

/**
 * @brief Unpacks the 8-bit Q7 phase state from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API gcso_q7_t GCSO_CALL gcso_tagged_ptr_unpack_phase(uint64_t ptr)
    GCSO_NOEXCEPT {
    return gcso_tagged_ptr_unpack_phase_inline(ptr);
}

/**
 * @brief Packs a 48-bit address, 3-bit Attractor Slot ID, 5-bit Action Mode mask, and Q7 phase into
 * a tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL
gcso_tagged_ptr_pack_action(uint64_t raw_addr, uint8_t attractor_slot, uint8_t action_mode_flags,
                            gcso_q7_t phase) GCSO_NOEXCEPT {
    const auto tag = static_cast<uint8_t>(((attractor_slot & GCSO_ATTRACTOR_SLOT_ID_MASK) << 5U) |
                                          (action_mode_flags & GCSO_ACTION_MODE_ALL_MASK));
    return gcso_tagged_ptr_pack_inline(raw_addr, tag, phase);
}

/**
 * @brief Extracts the 5-bit Action Mode mask from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint8_t GCSO_CALL gcso_tagged_ptr_unpack_action_mode(uint64_t ptr)
    GCSO_NOEXCEPT {
    return static_cast<uint8_t>(gcso_tagged_ptr_unpack_tag_inline(ptr) & GCSO_ACTION_MODE_ALL_MASK);
}

/**
 * @brief Extracts the 3-bit Target Attractor Working Lane ID (0..7) from a 64-bit tagged pointer.
 */
GCSO_NODISCARD GCSO_API uint8_t GCSO_CALL gcso_tagged_ptr_unpack_attractor_slot(uint64_t ptr)
    GCSO_NOEXCEPT {
    return static_cast<uint8_t>((gcso_tagged_ptr_unpack_tag_inline(ptr) >> 5U) &
                                GCSO_ATTRACTOR_SLOT_ID_MASK);
}

/**
 * @brief Evolves a 64-bit tagged pointer's Action Mode flags and Target Attractor Slot in O(1)
 * time.
 */
GCSO_NODISCARD GCSO_API uint64_t GCSO_CALL gcso_tagged_ptr_evolve_mode_stigmergically(
    uint64_t ptr, float stigmergic_density, float entropy_h_tilde, uint8_t repulsion_triggered,
    uint8_t dominant_attractor_slot, float low_entropy_skip_thresh,
    float high_density_rank1_thresh) GCSO_NOEXCEPT {
    const uint64_t addr = gcso_tagged_ptr_unpack_addr_inline(ptr);
    gcso_q7_t phase = gcso_tagged_ptr_unpack_phase_inline(ptr);

    uint8_t slot = dominant_attractor_slot & GCSO_ATTRACTOR_SLOT_ID_MASK;
    uint8_t mode = GCSO_ACTION_MODE_DPSR_STEER;

    if (repulsion_triggered != 0U) {
        slot = GCSO_ATTRACTOR_SLOT_REPULSION;
        mode = static_cast<uint8_t>(GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_SUBHEAD_ROUTE |
                                    GCSO_ACTION_MODE_PHASE_REPULSE);
        phase = (phase == -128) ? static_cast<gcso_q7_t>(127) : static_cast<gcso_q7_t>(-phase);
    } else if (entropy_h_tilde < low_entropy_skip_thresh && stigmergic_density >= 0.80f) {
        mode = GCSO_ACTION_MODE_LAYER_SKIP;
    } else if (stigmergic_density >= high_density_rank1_thresh) {
        mode = static_cast<uint8_t>(GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_SUBHEAD_ROUTE |
                                    GCSO_ACTION_MODE_RANK1_INJECT);
    } else if (stigmergic_density >= 0.50f || entropy_h_tilde >= 0.35f) {
        mode = static_cast<uint8_t>(GCSO_ACTION_MODE_DPSR_STEER | GCSO_ACTION_MODE_SUBHEAD_ROUTE);
    }

    return gcso_tagged_ptr_pack_action(addr, slot, mode, phase);
}

/**
 * @brief Retrieves the numeric version identifiers for the active GCSO C-ABI.
 *
 * @param major Pointer to store the major version number.
 * @param minor Pointer to store the minor version number.
 * @param patch Pointer to store the patch version number.
 * @return GCSO_SUCCESS, GCSO_ERROR_NULL_POINTER, or GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL
gcso_abi_get_version(uint32_t* GCSO_RESTRICT major, uint32_t* GCSO_RESTRICT minor,
                     uint32_t* GCSO_RESTRICT patch) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(major == nullptr || minor == nullptr || patch == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(major) % alignof(uint32_t) != 0 ||
                      reinterpret_cast<uintptr_t>(minor) % alignof(uint32_t) != 0 ||
                      reinterpret_cast<uintptr_t>(patch) % alignof(uint32_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }

    try {
        *major = GCSO_ABI_VERSION_MAJOR;
        *minor = GCSO_ABI_VERSION_MINOR;
        *patch = GCSO_ABI_VERSION_PATCH;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Returns the static semantic version string literal for the GCSO kernel engine
 * (`GCSO_ABI_VERSION_STRING`).
 *
 * @return Const pointer to null-terminated static version string.
 */
GCSO_NODISCARD GCSO_API const char* GCSO_CALL gcso_abi_get_version_string(void) GCSO_NOEXCEPT {
    return GCSO_ABI_VERSION_STRING;
}

/**
 * @brief Maps gcso_status_t status and error codes to descriptive static string representations.
 *
 * Provides 1:1 cross-language error diagnosis across C-ABI, C++, and Rust FFI boundaries.
 *
 * @param status Status code enum value.
 * @return Const pointer to descriptive static string representation.
 */
GCSO_NODISCARD GCSO_API const char* GCSO_CALL gcso_status_to_string(gcso_status_t status)
    GCSO_NOEXCEPT {
    switch (status) {
        case GCSO_SUCCESS:
            return "GCSO_SUCCESS";
        case GCSO_ERROR_INVALID_ARGUMENT:
            return "GCSO_ERROR_INVALID_ARGUMENT";
        case GCSO_ERROR_OUT_OF_MEMORY:
            return "GCSO_ERROR_OUT_OF_MEMORY";
        case GCSO_ERROR_BUFFER_TOO_SMALL:
            return "GCSO_ERROR_BUFFER_TOO_SMALL";
        case GCSO_ERROR_PANIC_CAUGHT:
            return "GCSO_ERROR_PANIC_CAUGHT";
        case GCSO_ERROR_NULL_POINTER:
            return "GCSO_ERROR_NULL_POINTER";
        case GCSO_ERROR_INVALID_STATE:
            return "GCSO_ERROR_INVALID_STATE";
        case GCSO_ERROR_VERSION_MISMATCH:
            return "GCSO_ERROR_VERSION_MISMATCH";
        case GCSO_ERROR_MISALIGNED_POINTER:
            return "GCSO_ERROR_MISALIGNED_POINTER";
        case GCSO_ERROR_IO_FAILURE:
            return "GCSO_ERROR_IO_FAILURE";
        case GCSO_ERROR_ACTION_HUB_FULL:
            return "GCSO_ERROR_ACTION_HUB_FULL";
        case GCSO_ERROR_NOT_IMPLEMENTED:
            return "GCSO_ERROR_NOT_IMPLEMENTED";
        case GCSO_ERROR_ATTRACTOR_NOT_FOUND:
            return "GCSO_ERROR_ATTRACTOR_NOT_FOUND";
        case GCSO_ERROR_DPSR_PHASE_OVERFLOW:
            return "GCSO_ERROR_DPSR_PHASE_OVERFLOW";
        case GCSO_ERROR_QDPS_UNDERFLOW:
            return "GCSO_ERROR_QDPS_UNDERFLOW";
        case GCSO_ERROR_EDBC_SINGULARITY:
            return "GCSO_ERROR_EDBC_SINGULARITY";
        case GCSO_ERROR_CONTAINER_CORRUPTED:
            return "GCSO_ERROR_CONTAINER_CORRUPTED";
        case GCSO_ERROR_MAPPED_STORAGE_FAILED:
            return "GCSO_ERROR_MAPPED_STORAGE_FAILED";
        case GCSO_ERROR_SUBHEAD_ROUTING_FAILED:
            return "GCSO_ERROR_SUBHEAD_ROUTING_FAILED";
        case GCSO_ERROR_KV_KEYFRAME_SEEK_FAILED:
            return "GCSO_ERROR_KV_KEYFRAME_SEEK_FAILED";
        case GCSO_ERROR_OBSTRUCTION_UNRESOLVED:
            return "GCSO_ERROR_OBSTRUCTION_UNRESOLVED";
        case GCSO_ERROR_EXTENSION_NOT_LOADED:
            return "GCSO_ERROR_EXTENSION_NOT_LOADED";
        case GCSO_ERROR_LEDGER_FULL:
            return "GCSO_ERROR_LEDGER_FULL";
        case GCSO_ERROR_UNKNOWN:
        default:
            return "GCSO_ERROR_UNKNOWN";
    }
}

/**
 * @brief Queries current compute capability flags (CPU SIMD, CUDA, Vulkan, Metal).
 *
 * @param flags Pointer to store 64-bit bitmask capability flags.
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER / GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL
gcso_abi_query_capability(gcso_capability_flags_t* GCSO_RESTRICT flags) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(flags == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(flags) % alignof(gcso_capability_flags_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        gcso_capability_flags_t caps = GCSO_CAP_CPU_BASELINE;

#if defined(LIMINIKA_ENABLE_CUDA)
        caps |= GCSO_CAP_CUDA_BACKEND;
#endif

#if defined(LIMINIKA_ENABLE_VULKAN)
        caps |= GCSO_CAP_VULKAN_BACKEND;
#endif

#if defined(LIMINIKA_ENABLE_METAL) || defined(__APPLE__)
        caps |= GCSO_CAP_METAL_BACKEND;
#endif

        *flags = caps;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Populates default settings for a gcso_config_t structure strictly conforming to GCSO
 * standards.
 *
 * Configures optimal parameters for 2-4GB VRAM execution targets, setting default head dimensions
 * (128), head counts (32), paged block sizes (32 tokens), RIPA soft-clamp bounds (5 degrees =
 * 0.087266 rad), and qdps minimum step thresholds (0.01 rad) to avoid grid jitter under ultra-low
 * quantization.
 *
 * @param config Pointer to user-allocated configuration descriptor to populate.
 * @return GCSO_SUCCESS or GCSO_ERROR_NULL_POINTER / GCSO_ERROR_MISALIGNED_POINTER.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_config_init_default(gcso_config_t* GCSO_RESTRICT config)
    GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(config == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(config) % alignof(gcso_config_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        std::memset(config, 0, sizeof(gcso_config_t));
        config->head_dim = 128;
        config->num_heads = 32;
        config->paged_block_size = 32;
        config->q7_phase_scale = 3.14159265358979323846f / 128.0f;
        config->ripa_clamp_max_rad = 0.0872664626f; // 5 degrees soft clamp limit
        config->qdps_min_step_rad = 0.01f;          // QDPS cutoff threshold
        config->entropy_singularity_eps = 1e-12f;   // Singularity guard eps_log
        config->max_prompt_anchors = 64;
        config->action_hub_capacity = 256;
        config->enable_cuda_warp_shuffle = 1;
        config->enable_zero_alloc_strict = 1;
        config->ledger_mode = 0; // Default: Fast-Path & Telemetry Scratchpad Mode
        config->reserved_flags = 0;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Validates a gcso_config_t descriptor against GCSO architectural invariants.
 */
GCSO_API gcso_status_t GCSO_CALL gcso_config_validate(const gcso_config_t* GCSO_RESTRICT config)
    GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(config == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(config) % alignof(gcso_config_t) != 0)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        const bool valid_dims =
            (config->head_dim != 0U) && ((config->head_dim & 1U) == 0U) &&
            (config->num_heads != 0U) && (config->num_heads <= GCSO_Q7_PHASE_ARRAY_LEN) &&
            (config->paged_block_size == 16U || config->paged_block_size == 32U);
        const bool valid_scales =
            std::isfinite(config->q7_phase_scale) && (config->q7_phase_scale > 0.0f) &&
            std::isfinite(config->ripa_clamp_max_rad) && (config->ripa_clamp_max_rad > 0.0f) &&
            std::isfinite(config->qdps_min_step_rad) && (config->qdps_min_step_rad >= 0.0f) &&
            std::isfinite(config->entropy_singularity_eps) &&
            (config->entropy_singularity_eps > 0.0f);
        const bool valid_caps = (config->max_prompt_anchors != 0U) &&
                                (config->action_hub_capacity != 0U) && (config->ledger_mode <= 2U);

        return (valid_dims && valid_scales && valid_caps) ? GCSO_SUCCESS
                                                          : GCSO_ERROR_INVALID_ARGUMENT;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Safely frees a heap-allocated string returned across the FFI boundary.
 *
 * Safe No-Op: Accepts NULL pointers safely returning without action.
 *
 * @param str Const string pointer to free (safe no-op if NULL).
 */
GCSO_API void GCSO_CALL gcso_free_string(const char* str) GCSO_NOEXCEPT {
    if (str != nullptr) {
        std::free(const_cast<char*>(str));
    }
}

/**
 * @brief Evaluates the branchless 4-region Venn partition (S&O, S\O, validated O\S, True Void)
 * across 8 SWAR granules.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_liminal_evaluate_venn_partition(
    uint8_t self_mask, const uint8_t* GCSO_RESTRICT other_granule_density_q7,
    const uint8_t* GCSO_RESTRICT granule_noise_tolerance_q7, uint8_t concentration_floor_q7,
    gcso_liminal_venn_partition_t* GCSO_RESTRICT partition_out) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(other_granule_density_q7 == nullptr ||
                      granule_noise_tolerance_q7 == nullptr || partition_out == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(partition_out) %
                          alignof(gcso_liminal_venn_partition_t) !=
                      0U)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    const auto p_addr = reinterpret_cast<uintptr_t>(partition_out);
    const auto o_addr = reinterpret_cast<uintptr_t>(other_granule_density_q7);
    const auto t_addr = reinterpret_cast<uintptr_t>(granule_noise_tolerance_q7);
    if (GCSO_UNLIKELY(
            byte_ranges_overlap(p_addr, sizeof(gcso_liminal_venn_partition_t), o_addr, 8U) ||
            byte_ranges_overlap(p_addr, sizeof(gcso_liminal_venn_partition_t), t_addr, 8U))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        uint8_t other_mask = 0U;
        for (size_t g = 0U; g < GCSO_NUM_STIGMERGIC_GRANULES; ++g) {
            const auto bit = static_cast<uint8_t>(1U << g);
            other_mask = static_cast<uint8_t>(
                other_mask |
                select_u8(other_granule_density_q7[g] > granule_noise_tolerance_q7[g], bit, 0U));
        }

        const auto shared_core_mask = static_cast<uint8_t>(self_mask & other_mask);
        const auto self_exclusive_mask =
            static_cast<uint8_t>(self_mask & static_cast<uint8_t>(~other_mask));
        const auto raw_ext_mask =
            static_cast<uint8_t>(static_cast<uint8_t>(~self_mask) & other_mask);
        const uint8_t bridge_count = popcount_u8_branchless(shared_core_mask);
        const uint8_t ext_cnt = popcount_u8_branchless(raw_ext_mask);

        uint8_t peak_excess = 0U;
        uint16_t sum_excess = 0U;
        for (size_t g = 0U; g < GCSO_NUM_STIGMERGIC_GRANULES; ++g) {
            const bool in_ext = ((raw_ext_mask >> g) & 1U) != 0U;
            const auto ex = static_cast<uint8_t>(
                (other_granule_density_q7[g] > granule_noise_tolerance_q7[g])
                    ? (other_granule_density_q7[g] - granule_noise_tolerance_q7[g])
                    : 0U);
            const uint8_t eff_ex = select_u8(in_ext, ex, 0U);
            peak_excess = select_u8(eff_ex > peak_excess, eff_ex, peak_excess);
            sum_excess = static_cast<uint16_t>(sum_excess + static_cast<uint16_t>(eff_ex));
        }

        static constexpr uint16_t INV_CNT_Q7[9] = {0U, 128U, 64U, 43U, 32U, 26U, 21U, 18U, 16U};
        const size_t cnt_idx = (ext_cnt > 8U) ? 8U : static_cast<size_t>(ext_cnt);
        const uint32_t scaled = (static_cast<uint32_t>(sum_excess) * INV_CNT_Q7[cnt_idx]) >> 7U;
        const auto mean_ext_excess = static_cast<uint8_t>((scaled > 127U) ? 127U : scaled);

        const auto clamped_peak = static_cast<uint8_t>((peak_excess > 127U) ? 127U : peak_excess);
        const uint16_t half_sum = static_cast<uint16_t>(sum_excess >> 1U);
        const auto clamped_half_sum = static_cast<uint8_t>((half_sum > 127U) ? 127U : half_sum);
        const uint8_t unbridged_conc =
            (clamped_half_sum > clamped_peak) ? clamped_half_sum : clamped_peak;
        const uint8_t bridged_conc =
            (clamped_peak > mean_ext_excess) ? clamped_peak : mean_ext_excess;
        const uint8_t other_concentration_q7 = select_u8(
            bridge_count >= 1U, bridged_conc, select_u8(ext_cnt >= 2U, unbridged_conc, 0U));

        const bool ext_valid =
            (raw_ext_mask != 0U) &&
            ((bridge_count >= 1U) || (other_concentration_q7 >= concentration_floor_q7));
        const uint8_t known_unknown_mask = select_u8(ext_valid, raw_ext_mask, 0U);
        const uint8_t known_unknown_density_q7 = select_u8(ext_valid, mean_ext_excess, 0U);
        const auto true_void_mask = static_cast<uint8_t>(
            ~(static_cast<uint8_t>(shared_core_mask | self_exclusive_mask | known_unknown_mask)));

        const uint8_t active_other_cnt =
            popcount_u8_branchless(static_cast<uint8_t>(shared_core_mask | known_unknown_mask));
        uint32_t p_q7 = static_cast<uint32_t>(active_other_cnt) << 4U;
        if (p_q7 > 128U) {
            p_q7 = 128U;
        }
        const uint32_t sens_raw = (p_q7 * (128U - p_q7)) >> 5U;
        const auto boundary_sensitivity_q7 =
            static_cast<uint8_t>((sens_raw > 127U) ? 127U : sens_raw);

        partition_out->self_mask = self_mask;
        partition_out->other_mask = other_mask;
        partition_out->shared_core_mask = shared_core_mask;
        partition_out->self_exclusive_mask = self_exclusive_mask;
        partition_out->known_unknown_mask = known_unknown_mask;
        partition_out->true_void_mask = true_void_mask;
        partition_out->bridge_count = bridge_count;
        partition_out->other_concentration_q7 = other_concentration_q7;
        partition_out->boundary_sensitivity_q7 = boundary_sensitivity_q7;
        partition_out->known_unknown_density_q7 = known_unknown_density_q7;
        partition_out->sandbox_eligible = (known_unknown_mask != 0U) ? 1U : 0U;
        partition_out->true_void_aperture_zero =
            ((shared_core_mask == 0U) && (known_unknown_mask == 0U)) ? 1U : 0U;
        partition_out->attribution_tag =
            GCSO_ATTRIBUTION_TAG(GCSO_SUBSTRATE_LIMINAL, GCSO_DRIVER_SYSTEM);
        partition_out->reserved_align[0] = 0U;
        partition_out->reserved_align[1] = 0U;
        partition_out->reserved_align[2] = 0U;
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Executes 1 branchless Hot-Path asymmetric homeostatic tension step.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_distortion_tension_step_hot_path(
    gcso_distortion_tension_t* GCSO_RESTRICT tension_inout, uint8_t ego_delta_norm_q7,
    uint8_t boundary_sensitivity_q7, uint8_t exit_entropy_q7, uint8_t guard_proximity_q7,
    uint8_t active_overdrive) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(tension_inout == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(reinterpret_cast<uintptr_t>(tension_inout) %
                          alignof(gcso_distortion_tension_t) !=
                      0U)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    try {
        tension_inout->last_ego_delta_norm_q7 =
            (ego_delta_norm_q7 > 127U) ? 127U : ego_delta_norm_q7;
        tension_inout->boundary_sensitivity_q7 =
            (boundary_sensitivity_q7 > 127U) ? 127U : boundary_sensitivity_q7;
        tension_inout->exit_entropy_q7 = (exit_entropy_q7 > 127U) ? 127U : exit_entropy_q7;
        tension_inout->guard_proximity_q7 = (guard_proximity_q7 > 127U) ? 127U : guard_proximity_q7;
        const uint32_t next_l1 = tension_inout->cumulative_delta_l1 +
                                 static_cast<uint32_t>(tension_inout->last_ego_delta_norm_q7);
        tension_inout->cumulative_delta_l1 =
            (next_l1 < tension_inout->cumulative_delta_l1) ? 0xFFFFFFFFU : next_l1;
        if (tension_inout->step_epochs < 0xFFFFFFFFU) {
            tension_inout->step_epochs += 1U;
        }

        const uint16_t sens_relief = static_cast<uint16_t>(boundary_sensitivity_q7 >> 3U);
        const uint16_t accum =
            static_cast<uint16_t>(static_cast<uint16_t>(tension_inout->last_ego_delta_norm_q7) +
                                  static_cast<uint16_t>(tension_inout->guard_proximity_q7 >> 1U) +
                                  static_cast<uint16_t>(tension_inout->exit_entropy_q7 >> 2U));
        const uint16_t diff =
            (accum > sens_relief) ? static_cast<uint16_t>(accum - sens_relief) : 0U;
        const auto raw_tension = static_cast<uint8_t>((diff > 127U) ? 127U : diff);
        tension_inout->tension_q7 = raw_tension;

        const bool spike = (active_overdrive != 0U) || (tension_inout->exit_entropy_q7 >= 84U) ||
                           (tension_inout->guard_proximity_q7 >= 72U) ||
                           (raw_tension > tension_inout->deadband_high_q7);
        const bool severe_guard = tension_inout->guard_proximity_q7 >= 96U;

        const uint8_t quarter = static_cast<uint8_t>(tension_inout->effective_ego_scale_q7 >> 2U);
        const uint8_t quarter_floored = (quarter > 8U) ? quarter : 8U;
        const uint8_t mild_floor =
            select_u8(tension_inout->effective_ego_scale_q7 > 0U, quarter_floored, 0U);
        const uint8_t damped_scale = select_u8(severe_guard, 0U, mild_floor);
        const uint8_t cooled =
            (tension_inout->suppression_cooldown_steps > 0U)
                ? static_cast<uint8_t>(tension_inout->suppression_cooldown_steps - 1U)
                : 0U;
        const uint16_t recov_sum =
            static_cast<uint16_t>(tension_inout->effective_ego_scale_q7) + 8U;
        const auto recov_clamped = static_cast<uint8_t>((recov_sum > 96U) ? 96U : recov_sum);
        const uint8_t healthy_recover =
            select_u8((cooled == 0U) && (tension_inout->effective_ego_scale_q7 > 0U), recov_clamped,
                      tension_inout->effective_ego_scale_q7);

        tension_inout->effective_ego_scale_q7 = select_u8(spike, damped_scale, healthy_recover);
        tension_inout->excitation_potential_u8 =
            select_u8(spike, 0U, tension_inout->excitation_potential_u8);
        tension_inout->suppression_cooldown_steps = select_u8(spike, 3U, cooled);
        tension_inout->state_code = select_u8(spike, 2U, 0U);
        if (spike && tension_inout->immediate_damping_events < 0xFFFFFFFFU) {
            tension_inout->immediate_damping_events += 1U;
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

/**
 * @brief Evaluates up to max_hops (1..=4) of a P-Frame DistortionDelta pointer chain while
 * enforcing SystemSpineState physical pinned isolation.
 */
GCSO_NODISCARD GCSO_API gcso_status_t GCSO_CALL gcso_distortion_delta_apply_chain(
    const gcso_distortion_delta_t* GCSO_RESTRICT nodes, size_t num_nodes,
    const gcso_liminal_ego_field_t* GCSO_RESTRICT ego,
    const gcso_system_spine_state_t* GCSO_RESTRICT spine,
    const gcso_liminal_venn_partition_t* GCSO_RESTRICT venn,
    const gcso_stigmergic_granule_field_t* GCSO_RESTRICT stigmergy,
    const gcso_distortion_tension_t* GCSO_RESTRICT tension, uint8_t max_hops,
    gcso_gateway_frame_t* GCSO_RESTRICT frame_inout, uint8_t* GCSO_RESTRICT delta_norm_out,
    uint8_t* GCSO_RESTRICT hops_traversed_out) GCSO_NOEXCEPT {
    if (GCSO_UNLIKELY(nodes == nullptr || ego == nullptr || spine == nullptr || venn == nullptr ||
                      stigmergy == nullptr || tension == nullptr || frame_inout == nullptr)) {
        return GCSO_ERROR_NULL_POINTER;
    }
    if (GCSO_UNLIKELY(num_nodes > GCSO_MAX_EGO_PFRAME_NODES)) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    if (GCSO_UNLIKELY(
            reinterpret_cast<uintptr_t>(nodes) % alignof(gcso_distortion_delta_t) != 0U ||
            reinterpret_cast<uintptr_t>(ego) % alignof(gcso_liminal_ego_field_t) != 0U ||
            reinterpret_cast<uintptr_t>(spine) % alignof(gcso_system_spine_state_t) != 0U ||
            reinterpret_cast<uintptr_t>(venn) % alignof(gcso_liminal_venn_partition_t) != 0U ||
            reinterpret_cast<uintptr_t>(stigmergy) % alignof(gcso_stigmergic_granule_field_t) !=
                0U ||
            reinterpret_cast<uintptr_t>(tension) % alignof(gcso_distortion_tension_t) != 0U ||
            reinterpret_cast<uintptr_t>(frame_inout) % alignof(gcso_gateway_frame_t) != 0U)) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    if (GCSO_UNLIKELY(byte_ranges_overlap(
            reinterpret_cast<uintptr_t>(frame_inout), sizeof(gcso_gateway_frame_t),
            reinterpret_cast<uintptr_t>(nodes), num_nodes * sizeof(gcso_distortion_delta_t)))) {
        return GCSO_ERROR_INVALID_ARGUMENT;
    }
    try {
        uint8_t out_norm = 0U;
        uint8_t out_hops = 0U;
        const bool bypass = (ego->trigger_mask == 0U) || (tension->effective_ego_scale_q7 == 0U) ||
                            (spine->ood_void_locked != 0U) ||
                            (spine->irreversible_step_lock != 0U) || (num_nodes == 0U);
        if (!bypass) {
            const uint8_t region_bits = static_cast<uint8_t>(
                select_u8(venn->shared_core_mask != 0U, GCSO_VENN_REGION_SHARED_CORE, 0U) |
                select_u8(venn->self_exclusive_mask != 0U, GCSO_VENN_REGION_SELF_EXCLUSIVE, 0U) |
                select_u8(venn->known_unknown_mask != 0U, GCSO_VENN_REGION_KNOWN_UNKNOWN, 0U));
            const uint8_t region_fallback =
                select_u8(region_bits != 0U, region_bits,
                          static_cast<uint8_t>(GCSO_VENN_REGION_SHARED_CORE |
                                               GCSO_VENN_REGION_SELF_EXCLUSIVE));
            const auto combined_support = static_cast<uint8_t>(
                venn->shared_core_mask | venn->self_exclusive_mask | venn->known_unknown_mask);
            const uint8_t support_mask = select_u8(combined_support != 0U, combined_support, 0xFFU);

            static constexpr uint32_t BAKED_ACTIVE_MASK =
                GCSO_DISTORTION_DELTA_FLAG_ACTIVE | GCSO_DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
            size_t best_entry_idx = 0U;
            uint32_t best_entry_score = 0U;
            for (size_t i = 0U; i < num_nodes; ++i) {
                const gcso_distortion_delta_t& n = nodes[i];
                const bool active = (n.flags & BAKED_ACTIVE_MASK) == BAKED_ACTIVE_MASK;
                const bool reg_ok = (n.venn_region_selector & region_fallback) != 0U;
                const uint8_t trig_overlap = popcount_u8_branchless(
                    static_cast<uint8_t>(n.venn_trigger_mask & support_mask));
                const uint32_t score = (active && reg_ok && trig_overlap > 0U)
                                           ? (static_cast<uint32_t>(n.pheromone_q7) * trig_overlap)
                                           : 0U;
                const bool take = score > best_entry_score;
                best_entry_score = select_u32(take, score, best_entry_score);
                if (take) {
                    best_entry_idx = i;
                }
            }

            if (best_entry_score > 0U) {
                const uint8_t clamped_hops = (max_hops == 0U) ? 1U
                                             : (max_hops > GCSO_MAX_DISTORTION_CHAIN_HOPS)
                                                 ? GCSO_MAX_DISTORTION_CHAIN_HOPS
                                                 : max_hops;
                const uint32_t sens_boost_q7 =
                    (venn->boundary_sensitivity_q7 > 48U) ? venn->boundary_sensitivity_q7 : 48U;
                const uint32_t eff_scale = tension->effective_ego_scale_q7;
                uint32_t stig_sum = 0U;
                for (size_t g = 0U; g < GCSO_NUM_STIGMERGIC_GRANULES; ++g) {
                    stig_sum += stigmergy->granule_pheromone_q7[g];
                }
                uint32_t mean_stig_ph = stig_sum >> 3U;
                if (mean_stig_ph < 24U) {
                    mean_stig_ph = 24U;
                } else if (mean_stig_ph > 127U) {
                    mean_stig_ph = 127U;
                }

                const gcso_gateway_frame_t base_frame = *frame_inout;
                uint64_t acc_words[GCSO_NUM_STIGMERGIC_GRANULES]{};
                for (size_t c = 0U; c < GCSO_NUM_STIGMERGIC_GRANULES; ++c) {
                    acc_words[c] = load_q7_word8(&base_frame.phase_deltas[c * 8U]);
                }

                uint32_t accum_norm_sum = 0U;
                auto cur_idx = static_cast<uint16_t>(best_entry_idx);
                for (uint8_t hop = 0U; hop < clamped_hops; ++hop) {
                    if (cur_idx >= num_nodes) {
                        break;
                    }
                    const gcso_distortion_delta_t& node = nodes[cur_idx];
                    if ((node.flags & BAKED_ACTIVE_MASK) != BAKED_ACTIVE_MASK) {
                        break;
                    }
                    const uint32_t hop_w =
                        (((static_cast<uint32_t>(node.gain_q7) * eff_scale) >> 7U) *
                         sens_boost_q7) >>
                        7U;
                    uint64_t dens_w = (hop_w * mean_stig_ph) >> 7U;
                    if (dens_w < 8ULL) {
                        dens_w = 8ULL;
                    } else if (dens_w > 127ULL) {
                        dens_w = 127ULL;
                    }
                    const uint64_t bias_restore = (128ULL - dens_w) * 0x0101010101010101ULL;
                    for (size_t c = 0U; c < GCSO_NUM_STIGMERGIC_GRANULES; ++c) {
                        const uint64_t nw = load_q7_word8(&node.delta_phase_q7[c * 8U]);
                        const uint64_t dw = swar_scale_le128_with_bias_x8(nw, dens_w, bias_restore);
                        acc_words[c] = swar_circular_add_x8(acc_words[c], dw);
                    }
                    accum_norm_sum += static_cast<uint32_t>(dens_w);
                    out_hops = static_cast<uint8_t>(out_hops + 1U);
                    cur_idx = node.next_hop_idx;
                }

                gcso_gateway_frame_t ego_frame = base_frame;
                for (size_t c = 0U; c < GCSO_NUM_STIGMERGIC_GRANULES; ++c) {
                    store_q7_word8(&ego_frame.phase_deltas[c * 8U], acc_words[c]);
                }
                ego_frame.mean_phase_tag_q7 = mean_q7_from_64(ego_frame.phase_deltas);

                const auto base_slot =
                    static_cast<uint8_t>(base_frame.dominant_slot & GCSO_ATTRACTOR_SLOT_ID_MASK);
                const bool slot_pinned = (spine->pinned_slot_mask & (1U << base_slot)) != 0U;
                ego_frame.dominant_slot =
                    select_u8(slot_pinned, base_frame.dominant_slot, ego_frame.dominant_slot);
                ego_frame.action_mode_flags = static_cast<uint8_t>(
                    ego_frame.action_mode_flags |
                    (base_frame.action_mode_flags & spine->pinned_action_mode_mask));

                if ((spine->ood_void_locked != 0U) || (spine->irreversible_step_lock != 0U)) {
                    *frame_inout = base_frame;
                } else {
                    const uint8_t ptr_tag = static_cast<uint8_t>(
                        ((ego_frame.dominant_slot & GCSO_ATTRACTOR_SLOT_ID_MASK)
                         << GCSO_TAG_ATTRACTOR_SLOT_SHIFT) |
                        (ego_frame.action_mode_flags & GCSO_ACTION_MODE_ALL_MASK));
                    ego_frame.frame_flags = static_cast<uint16_t>(ptr_tag);
                    ego_frame.tagged_ptr = gcso_tagged_ptr_pack_inline(
                        ego_frame.mapped_section_offset48, ptr_tag, ego_frame.mean_phase_tag_q7);
                    *frame_inout = ego_frame;
                }
                static constexpr uint32_t INV_HOPS_Q7[5] = {0U, 128U, 64U, 43U, 32U};
                const size_t h_idx = (out_hops > 4U) ? 4U : static_cast<size_t>(out_hops);
                const uint32_t avg_d = (accum_norm_sum * INV_HOPS_Q7[h_idx]) >> 7U;
                out_norm = static_cast<uint8_t>((avg_d > 127U) ? 127U : avg_d);
            }
        }
        if (delta_norm_out != nullptr) {
            *delta_norm_out = out_norm;
        }
        if (hops_traversed_out != nullptr) {
            *hops_traversed_out = out_hops;
        }
        return GCSO_SUCCESS;
    } catch (...) {
        return GCSO_ERROR_PANIC_CAUGHT;
    }
}

GCSO_EXTERN_C_END