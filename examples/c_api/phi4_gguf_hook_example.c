// name: examples/c_api/phi4_gguf_hook_example.c
// SPDX-License-Identifier: MIT OR Apache-2.0

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "liminika/lmnk_abi.h"

#if defined(_WIN32)
    #include <malloc.h>
static void* lmnk_alloc_aligned(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}
static void lmnk_free_aligned(void* ptr) {
    _aligned_free(ptr);
}
#else
static void* lmnk_alloc_aligned(size_t alignment, size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}
static void lmnk_free_aligned(void* ptr) {
    free(ptr);
}
#endif

/**
 * @brief Minimal C11 host harness demonstrating a GGUF / llama.cpp post-RoPE hook
 * using HalfSplit partial-RoPE geometry via the LiminiKa v0.4.0 C-ABI.
 *
 * @note The LiminiKa C-ABI is model-agnostic; this example uses a 24-head,
 * d_head=128, rotary_dim=96 HalfSplit layout as a concrete case.
 */
int main(void) {
    lmnk_config_t cfg;
    lmnk_status_t status = lmnk_config_init_default(&cfg);
    if (status != LMNK_OK) {
        fprintf(stderr, "lmnk_config_init_default failed: %d\n", (int)status);
        return 1;
    }

    /* Configure HalfSplit partial-RoPE geometry (d_head=128, rotary_dim=96, H_Q=24). */
    cfg.head_dim = 128u;
    cfg.rotary_dim = 96u;
    cfg.num_heads = 24u;
    cfg.rope_layout = (uint8_t)LMNK_ROPE_LAYOUT_HALF_SPLIT;
    cfg.phase_clamp_max_rad = 0.08f;

    status = lmnk_config_validate(&cfg);
    if (status != LMNK_OK) {
        fprintf(stderr, "lmnk_config_validate failed: %d\n", (int)status);
        return 1;
    }

    const size_t head_dim = (size_t)cfg.head_dim;
    const size_t rotary_dim = (size_t)cfg.rotary_dim;
    const size_t num_heads = (size_t)cfg.num_heads;
    const size_t total_q = head_dim * num_heads;

    float* q_before = (float*)lmnk_alloc_aligned(32u, total_q * sizeof(float));
    float* q_after = (float*)lmnk_alloc_aligned(32u, total_q * sizeof(float));
    lmnk_q7_t* phase_deltas = (lmnk_q7_t*)lmnk_alloc_aligned(16u, num_heads * sizeof(lmnk_q7_t));
    lmnk_q7_t* recovered_q7 = (lmnk_q7_t*)lmnk_alloc_aligned(16u, num_heads * sizeof(lmnk_q7_t));

    if (q_before == NULL || q_after == NULL || phase_deltas == NULL || recovered_q7 == NULL) {
        fprintf(stderr, "Aligned allocation failed\n");
        lmnk_free_aligned(q_before);
        lmnk_free_aligned(q_after);
        lmnk_free_aligned(phase_deltas);
        lmnk_free_aligned(recovered_q7);
        return 1;
    }

    for (size_t h = 0; h < num_heads; ++h) {
        phase_deltas[h] = (lmnk_q7_t)((h % 2u == 0u) ? 16 : -16);
        for (size_t d = 0; d < head_dim; ++d) {
            const float val = (d % 2u == 0u) ? 1.0f : 0.25f;
            q_before[h * head_dim + d] = val;
            q_after[h * head_dim + d] = val;
        }
    }

    /* Example Top-16 layer mask (upper 16 layers of a 32-layer model). */
    const uint32_t top16_layer_mask = 0xFFFF0000u;
    const uint32_t example_layer_idx = 20u;
    if ((top16_layer_mask & (1u << example_layer_idx)) != 0u) {
        lmnk_rope_steer_params_t steer_params;
        memset(&steer_params, 0, sizeof(steer_params));
        steer_params.struct_size = (uint32_t)sizeof(steer_params);
        steer_params.head_dim = cfg.head_dim;
        steer_params.rotary_dim = cfg.rotary_dim;
        steer_params.num_heads = cfg.num_heads;
        steer_params.rope_layout = cfg.rope_layout;
        steer_params.max_rad = cfg.phase_clamp_max_rad;
        steer_params.num_tokens = 1u;
        steer_params.query_tensor = q_after;
        steer_params.phase_deltas = phase_deltas;

        status = lmnk_rope_steer_q(&steer_params);
        if (status != LMNK_OK) {
            fprintf(stderr, "lmnk_rope_steer_q failed: %d\n", (int)status);
            lmnk_free_aligned(q_before);
            lmnk_free_aligned(q_after);
            lmnk_free_aligned(phase_deltas);
            lmnk_free_aligned(recovered_q7);
            return 1;
        }
    }

    /* Verify non-rotary tail [rotary_dim .. head_dim) is untouched for every head. */
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t d = rotary_dim; d < head_dim; ++d) {
            const size_t idx = h * head_dim + d;
            if (fabsf(q_after[idx] - q_before[idx]) > 1.0e-7f) {
                fprintf(stderr, "Partial RoPE tail corrupted at head=%zu dim=%zu\n", h, d);
                lmnk_free_aligned(q_before);
                lmnk_free_aligned(q_after);
                lmnk_free_aligned(phase_deltas);
                lmnk_free_aligned(recovered_q7);
                return 1;
            }
        }
    }

    lmnk_rope_procrustes_params_t proc_params;
    memset(&proc_params, 0, sizeof(proc_params));
    proc_params.struct_size = (uint32_t)sizeof(proc_params);
    proc_params.head_dim = cfg.head_dim;
    proc_params.rotary_dim = cfg.rotary_dim;
    proc_params.num_heads = cfg.num_heads;
    proc_params.rope_layout = cfg.rope_layout;
    proc_params.num_tokens = 1u;
    proc_params.q_before = q_before;
    proc_params.q_after = q_after;
    proc_params.q7_phase_out = recovered_q7;

    status = lmnk_rope_procrustes_q7(&proc_params);
    if (status != LMNK_OK) {
        fprintf(stderr, "lmnk_rope_procrustes_q7 failed: %d\n", (int)status);
        lmnk_free_aligned(q_before);
        lmnk_free_aligned(q_after);
        lmnk_free_aligned(phase_deltas);
        lmnk_free_aligned(recovered_q7);
        return 1;
    }

    printf("C-ABI HalfSplit hook OK: layer=%u head0_q7_in=%d head0_q7_recovered=%d\n",
           example_layer_idx, (int)phase_deltas[0], (int)recovered_q7[0]);

    /* Also verify 2-Origin Gateway composite steering (structural hyperparameters). */
    lmnk_pointer_trail_t* trail =
        (lmnk_pointer_trail_t*)lmnk_alloc_aligned(64u, sizeof(lmnk_pointer_trail_t));
    lmnk_q7_t* persona_deltas = (lmnk_q7_t*)lmnk_alloc_aligned(16u, num_heads * sizeof(lmnk_q7_t));
    if (trail == NULL || persona_deltas == NULL) {
        fprintf(stderr, "Aligned Gateway buffer allocation failed\n");
        lmnk_free_aligned(trail);
        lmnk_free_aligned(persona_deltas);
        lmnk_free_aligned(q_before);
        lmnk_free_aligned(q_after);
        lmnk_free_aligned(phase_deltas);
        lmnk_free_aligned(recovered_q7);
        return 1;
    }
    memset(trail, 0, sizeof(lmnk_pointer_trail_t));
    for (size_t h = 0; h < num_heads; ++h) {
        persona_deltas[h] = (lmnk_q7_t)(-8);
    }

    lmnk_gateway_steer_params_t gw_params;
    memset(&gw_params, 0, sizeof(gw_params));
    gw_params.struct_size = (uint32_t)sizeof(gw_params);
    gw_params.head_dim = cfg.head_dim;
    gw_params.rotary_dim = cfg.rotary_dim;
    gw_params.num_heads = cfg.num_heads;
    gw_params.rope_layout = cfg.rope_layout;
    gw_params.unresolved_damping_u7 = 128u;
    gw_params.fact_energy = 2.5f;
    gw_params.persona_energy = 1.8f;
    gw_params.num_tokens = 1u;
    gw_params.query_tensor = q_after;
    gw_params.fact_phases = phase_deltas;
    gw_params.persona_phases = persona_deltas;
    gw_params.trail_inout = trail;

    status = lmnk_gateway_steer(&gw_params);
    if (status != LMNK_OK) {
        fprintf(stderr, "lmnk_gateway_steer failed: %d\n", (int)status);
        lmnk_free_aligned(trail);
        lmnk_free_aligned(persona_deltas);
        lmnk_free_aligned(q_before);
        lmnk_free_aligned(q_after);
        lmnk_free_aligned(phase_deltas);
        lmnk_free_aligned(recovered_q7);
        return 1;
    }

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t d = rotary_dim; d < head_dim; ++d) {
            const size_t idx = h * head_dim + d;
            if (fabsf(q_after[idx] - q_before[idx]) > 1.0e-7f) {
                fprintf(stderr, "Gateway tail corrupted at head=%zu dim=%zu\n", h, d);
                lmnk_free_aligned(trail);
                lmnk_free_aligned(persona_deltas);
                lmnk_free_aligned(q_before);
                lmnk_free_aligned(q_after);
                lmnk_free_aligned(phase_deltas);
                lmnk_free_aligned(recovered_q7);
                return 1;
            }
        }
    }
    printf("Gateway composite C-ABI OK: trail_step=%llu\n", (unsigned long long)trail->step_count);

    lmnk_free_aligned(trail);
    lmnk_free_aligned(persona_deltas);
    lmnk_free_aligned(q_before);
    lmnk_free_aligned(q_after);
    lmnk_free_aligned(phase_deltas);
    lmnk_free_aligned(recovered_q7);
    return 0;
}
