// name: examples/c_api/phi4_gguf_hook_example.c
// SPDX-License-Identifier: MIT OR Apache-2.0

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "liminika/gcso_abi.h"

#if defined(_WIN32)
#include <malloc.h>
static void* gcso_alloc_aligned(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
static void gcso_free_aligned(void* ptr) {
    _aligned_free(ptr);
}
#else
static void* gcso_alloc_aligned(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}
static void gcso_free_aligned(void* ptr) {
    free(ptr);
}
#endif

/**
 * @brief Minimal C11 host harness demonstrating a GGUF / llama.cpp post-RoPE hook
 * using Phi-4-mini HalfSplit partial-RoPE geometry via LiminiKa C-ABI v0.1.1.
 */
int main(void) {
    gcso_config_t cfg;
    gcso_status_t status = gcso_config_init_default(&cfg);
    if (status != GCSO_SUCCESS) {
        fprintf(stderr, "gcso_config_init_default failed: %d\n", (int)status);
        return 1;
    }

    /* Configure Phi-4-mini geometry (HalfSplit, d_head=128, rotary_dim=96, H_Q=24). */
    cfg.head_dim = 128u;
    cfg.rotary_dim = 96u;
    cfg.num_heads = 24u;
    cfg.rope_layout = (uint8_t)GCSO_ROPE_LAYOUT_HALF_SPLIT;
    cfg.ripa_clamp_max_rad = 0.08f;

    status = gcso_config_validate(&cfg);
    if (status != GCSO_SUCCESS) {
        fprintf(stderr, "gcso_config_validate failed: %d\n", (int)status);
        return 1;
    }

    const size_t head_dim = (size_t)cfg.head_dim;
    const size_t rotary_dim = (size_t)cfg.rotary_dim;
    const size_t num_heads = (size_t)cfg.num_heads;
    const size_t total_q = head_dim * num_heads;

    float* q_before = (float*)gcso_alloc_aligned(32u, total_q * sizeof(float));
    float* q_after = (float*)gcso_alloc_aligned(32u, total_q * sizeof(float));
    gcso_q7_t* phase_deltas = (gcso_q7_t*)gcso_alloc_aligned(16u, num_heads * sizeof(gcso_q7_t));
    gcso_q7_t* recovered_q7 = (gcso_q7_t*)gcso_alloc_aligned(16u, num_heads * sizeof(gcso_q7_t));

    if (q_before == NULL || q_after == NULL || phase_deltas == NULL || recovered_q7 == NULL) {
        fprintf(stderr, "Aligned allocation failed\n");
        gcso_free_aligned(q_before);
        gcso_free_aligned(q_after);
        gcso_free_aligned(phase_deltas);
        gcso_free_aligned(recovered_q7);
        return 1;
    }

    for (size_t h = 0; h < num_heads; ++h) {
        phase_deltas[h] = (gcso_q7_t)((h % 2u == 0u) ? 16 : -16);
        for (size_t d = 0; d < head_dim; ++d) {
            const float val = (d % 2u == 0u) ? 1.0f : 0.25f;
            q_before[h * head_dim + d] = val;
            q_after[h * head_dim + d] = val;
        }
    }

    /* Example Top-16 layer mask (upper 16 layers of 32-layer Phi-4-mini). */
    const uint32_t top16_layer_mask = 0xFFFF0000u;
    const uint32_t example_layer_idx = 20u;
    if ((top16_layer_mask & (1u << example_layer_idx)) != 0u) {
        status = gcso_dpsr_apply_phase_steering_with_config_safe(q_after, phase_deltas, &cfg);
        if (status != GCSO_SUCCESS) {
            fprintf(stderr, "gcso_dpsr_apply_phase_steering_with_config_safe failed: %d\n",
                    (int)status);
            gcso_free_aligned(q_before);
            gcso_free_aligned(q_after);
            gcso_free_aligned(phase_deltas);
            gcso_free_aligned(recovered_q7);
            return 1;
        }
    }

    /* Verify non-rotary tail [rotary_dim .. head_dim) is untouched for every head. */
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t d = rotary_dim; d < head_dim; ++d) {
            const size_t idx = h * head_dim + d;
            if (fabsf(q_after[idx] - q_before[idx]) > 1.0e-7f) {
                fprintf(stderr, "Partial RoPE tail corrupted at head=%zu dim=%zu\n", h, d);
                gcso_free_aligned(q_before);
                gcso_free_aligned(q_after);
                gcso_free_aligned(phase_deltas);
                gcso_free_aligned(recovered_q7);
                return 1;
            }
        }
    }

    status = gcso_dpsr_compute_procrustes_phase_delta_half_split(
        q_before, q_after, head_dim, rotary_dim, num_heads, recovered_q7);
    if (status != GCSO_SUCCESS) {
        fprintf(stderr, "gcso_dpsr_compute_procrustes_phase_delta_half_split failed: %d\n",
                (int)status);
        gcso_free_aligned(q_before);
        gcso_free_aligned(q_after);
        gcso_free_aligned(phase_deltas);
        gcso_free_aligned(recovered_q7);
        return 1;
    }

    printf("Phi-4-mini C-ABI HalfSplit hook OK: layer=%u head0_q7_in=%d head0_q7_recovered=%d\n",
           example_layer_idx, (int)phase_deltas[0], (int)recovered_q7[0]);

    /* Also verify 2-Origin Gateway composite steering (structural hyperparameters). */
    gcso_pointer_trail_t* trail =
        (gcso_pointer_trail_t*)gcso_alloc_aligned(128u, sizeof(gcso_pointer_trail_t));
    gcso_q7_t* persona_deltas = (gcso_q7_t*)gcso_alloc_aligned(16u, num_heads * sizeof(gcso_q7_t));
    if (trail == NULL || persona_deltas == NULL) {
        fprintf(stderr, "Aligned Gateway buffer allocation failed\n");
        gcso_free_aligned(trail);
        gcso_free_aligned(persona_deltas);
        gcso_free_aligned(q_before);
        gcso_free_aligned(q_after);
        gcso_free_aligned(phase_deltas);
        gcso_free_aligned(recovered_q7);
        return 1;
    }
    memset(trail, 0, sizeof(gcso_pointer_trail_t));
    for (size_t h = 0; h < num_heads; ++h) {
        persona_deltas[h] = (gcso_q7_t)(-8);
    }

    status = gcso_dpsr_steer_with_gateway_and_config_safe(
        q_after, phase_deltas, 2.5f, persona_deltas, 1.8f, trail, &cfg);
    if (status != GCSO_SUCCESS) {
        fprintf(stderr, "gcso_dpsr_steer_with_gateway_and_config_safe failed: %d\n", (int)status);
        gcso_free_aligned(trail);
        gcso_free_aligned(persona_deltas);
        gcso_free_aligned(q_before);
        gcso_free_aligned(q_after);
        gcso_free_aligned(phase_deltas);
        gcso_free_aligned(recovered_q7);
        return 1;
    }

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t d = rotary_dim; d < head_dim; ++d) {
            const size_t idx = h * head_dim + d;
            if (fabsf(q_after[idx] - q_before[idx]) > 1.0e-7f) {
                fprintf(stderr, "Gateway tail corrupted at head=%zu dim=%zu\n", h, d);
                gcso_free_aligned(trail);
                gcso_free_aligned(persona_deltas);
                gcso_free_aligned(q_before);
                gcso_free_aligned(q_after);
                gcso_free_aligned(phase_deltas);
                gcso_free_aligned(recovered_q7);
                return 1;
            }
        }
    }
    printf("Phi-4-mini Gateway composite C-ABI OK: trail_step=%llu\n",
           (unsigned long long)trail->step_count);

    gcso_free_aligned(trail);
    gcso_free_aligned(persona_deltas);
    gcso_free_aligned(q_before);
    gcso_free_aligned(q_after);
    gcso_free_aligned(phase_deltas);
    gcso_free_aligned(recovered_q7);
    return 0;
}
