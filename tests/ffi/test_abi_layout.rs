// name: tests/ffi/test_abi_layout.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test verifying C-ABI struct size, alignment, and field offset invariants.

#[test]
fn test_lmnk_types_abi_layout_invariants() {
    use std::mem::{align_of, needs_drop, offset_of, size_of};

    assert_eq!(liminika_core::abi::LMNK_ABI_VERSION_HEX, 0x0000_0200);
    assert_eq!(liminika_core::abi::LMNK_ABI_VERSION_STRING, "0.2.0");

    // Verify 128-byte, 64-byte-aligned lmnk_pointer_trail_t
    assert_eq!(size_of::<liminika_core::abi::lmnk_pointer_trail_t>(), 128);
    assert_eq!(align_of::<liminika_core::abi::lmnk_pointer_trail_t>(), 64);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_pointer_trail_t, reserved_header),
        56
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_pointer_trail_t,
            accumulated_phase_delta
        ),
        64
    );

    // Verify 64-byte, 64-byte-aligned lmnk_config_t
    assert_eq!(size_of::<liminika_core::abi::lmnk_config_t>(), 64);
    assert_eq!(align_of::<liminika_core::abi::lmnk_config_t>(), 64);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_config_t, execution_flags),
        40
    );
    assert_eq!(offset_of!(liminika_core::abi::lmnk_config_t, reserved), 44);

    // Verify 64-byte, 64-byte-aligned lmnk_ledger_slot_t
    assert_eq!(size_of::<liminika_core::abi::lmnk_ledger_slot_t>(), 64);
    assert_eq!(align_of::<liminika_core::abi::lmnk_ledger_slot_t>(), 64);

    // Verify EntropyState (32 B, align 32) and RuntimeContext / StigmergicSubstrate layouts
    assert_eq!(size_of::<liminika_core::abi::lmnk_entropy_state_t>(), 32);
    assert_eq!(align_of::<liminika_core::abi::lmnk_entropy_state_t>(), 32);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_entropy_state_t, flags),
        28
    );
    assert_eq!(
        size_of::<liminika_core::stigmergy::StigmergicSubstrate>(),
        2816
    );
    assert_eq!(
        align_of::<liminika_core::stigmergy::StigmergicSubstrate>(),
        64
    );
    assert_eq!(size_of::<liminika_core::runtime::RuntimeContext>() % 64, 0);
    assert_eq!(align_of::<liminika_core::runtime::RuntimeContext>(), 64);
    assert!(!needs_drop::<liminika_core::runtime::RuntimeContext>());
    assert_eq!(
        liminika_core::abi::lmnk_context_size(),
        size_of::<liminika_core::runtime::RuntimeContext>()
    );
    assert_eq!(
        liminika_core::abi::lmnk_context_align(),
        align_of::<liminika_core::runtime::RuntimeContext>()
    );

    // Verify Cognitive Trajectory ABI types
    assert_eq!(size_of::<liminika_core::stigmergy::CognitiveStepOp>(), 1);
    assert_eq!(
        size_of::<liminika_core::stigmergy::CognitiveCircuitActivation>(),
        8
    );
    assert_eq!(
        align_of::<liminika_core::stigmergy::CognitiveCircuitActivation>(),
        8
    );
    assert_eq!(
        offset_of!(
            liminika_core::stigmergy::CognitiveCircuitActivation,
            trajectory_depth
        ),
        7
    );
    assert_eq!(size_of::<liminika_core::abi::lmnk_cognitive_ledger_t>(), 8);
    assert_eq!(align_of::<liminika_core::abi::lmnk_cognitive_ledger_t>(), 8);

    // Verify GatewayFrame (128 B, 64 B align) and CrystallizedAttractorRecord (128 B, 64 B align)
    assert_eq!(size_of::<liminika_core::abi::lmnk_gateway_frame_t>(), 128);
    assert_eq!(align_of::<liminika_core::abi::lmnk_gateway_frame_t>(), 64);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_gateway_frame_t, phase_deltas),
        64
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_phase_anchor_record_t>(),
        128
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_phase_anchor_record_t>(),
        64
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_phase_anchor_record_t,
            phase_clamp_rad
        ),
        40
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_phase_anchor_record_t,
            lineage_flags
        ),
        48
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_phase_anchor_record_t,
            attribution_tag
        ),
        59
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_phase_anchor_record_t,
            phase_profile
        ),
        64
    );

    // Verify Stigmergic Reconstitution report (16 B, 16 B align)
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_reconstitution_report_t>(),
        16
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_reconstitution_report_t>(),
        16
    );

    // Verify Liminal & 2-Layer Autonomous Core Stack structures
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_epistemic_horizon_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_epistemic_horizon_t>(),
        16
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_epistemic_horizon_t, regime_code),
        28
    );

    assert_eq!(
        size_of::<liminika_core::abi::lmnk_liminal_threshold_state_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_liminal_threshold_state_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_liminal_threshold_state_t,
            adaptation_epochs
        ),
        28
    );

    assert_eq!(
        size_of::<liminika_core::abi::lmnk_stigmergic_granule_state_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_stigmergic_granule_state_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_stigmergic_granule_state_t,
            adaptation_epochs
        ),
        28
    );

    assert_eq!(
        size_of::<liminika_core::abi::lmnk_overlap_discovery_report_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_overlap_discovery_report_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_overlap_discovery_report_t,
            discovery_seed_word
        ),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_overlap_discovery_report_t,
            discovery_triggered
        ),
        30
    );

    assert_eq!(
        size_of::<liminika_core::abi::lmnk_core_state_stack_t>(),
        128
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_core_state_stack_t>(),
        64
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_core_state_stack_t, liminal),
        0
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_core_state_stack_t, stigmergy),
        32
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_core_state_stack_t, ego),
        64
    );

    assert_eq!(
        size_of::<liminika_core::abi::lmnk_topology_snapshot_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_topology_snapshot_t>(),
        16
    );

    assert_eq!(size_of::<liminika_core::abi::lmnk_boundary_delta_t>(), 16);
    assert_eq!(align_of::<liminika_core::abi::lmnk_boundary_delta_t>(), 16);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_boundary_delta_t, kinetic_energy_q7),
        10
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_boundary_delta_t,
            stationary_shortcut_ready
        ),
        11
    );

    assert_eq!(size_of::<liminika_core::abi::lmnk_ego_intent_axes_t>(), 16);
    assert_eq!(align_of::<liminika_core::abi::lmnk_ego_intent_axes_t>(), 16);
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_ego_intent_axes_t,
            preferred_slot_mask
        ),
        8
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_ego_intent_axes_t, signature_hash),
        12
    );

    assert_eq!(
        size_of::<liminika_core::abi::lmnk_fragment_ensemble_report_t>(),
        16
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_fragment_ensemble_report_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_fragment_ensemble_report_t,
            quorum_agreed_granule_mask
        ),
        8
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_fragment_ensemble_report_t,
            ego_tiebreak_granule_mask
        ),
        10
    );

    assert_eq!(
        size_of::<liminika_core::abi::lmnk_liminal_ego_state_t>(),
        64
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_liminal_ego_state_t>(),
        64
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_liminal_ego_state_t,
            counter_basin_thrust_q7
        ),
        40
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_liminal_ego_state_t,
            basin_escape_floor_q7
        ),
        41
    );
    assert_eq!(
        liminika_core::abi::LMNK_ABI_VERSION_STRING,
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(
        liminika_core::stigmergy::LMNK_PHASE_ANCHOR_VERSION,
        liminika_core::abi::LMNK_ABI_VERSION_HEX
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_liminal_ego_state_t,
            ego_signature_hash
        ),
        48
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_liminal_ego_state_t,
            last_intervened_mask
        ),
        56
    );

    // Verify StigmergicSubstrate 2,816 B header-first Pod layout and anonymous lane bitmasks
    assert_eq!(size_of::<liminika_core::StigmergicSubstrate>(), 2816);
    assert_eq!(align_of::<liminika_core::StigmergicSubstrate>(), 64);
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, coupling_matrix_8x8),
        64
    );
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, step_counter),
        72
    );
    assert_eq!(offset_of!(liminika_core::StigmergicSubstrate, capacity), 80);
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, head_dim),
        104
    );
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, num_heads),
        106
    );
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, slot_enabled_mask),
        110
    );
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, last_decision),
        111
    );
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, slot_pinned_mask),
        112
    );
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, slot_repulsion_mask),
        113
    );
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, slot_sandbox_mask),
        114
    );
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, reserved_header),
        115
    );
    assert_eq!(
        offset_of!(liminika_core::StigmergicSubstrate, ledger_slot),
        128
    );
    assert_eq!(offset_of!(liminika_core::StigmergicSubstrate, trails), 768);
    assert_eq!(liminika_core::stigmergy::ATTRACTOR_SLOT_TEMPORAL, 5);
    assert_eq!(liminika_core::stigmergy::ATTRACTOR_SLOT_USER_INTENT, 6);
    assert_eq!(liminika_core::stigmergy::ATTRACTOR_SLOT_CRYSTALLIZED, 7);
    assert_eq!(liminika_core::stigmergy::WORKBENCH_LANE_COUNT, 8);
    assert_eq!(liminika_core::stigmergy::WORKBENCH_LANE_AUTO, 0xFF);
    assert_eq!(liminika_core::stigmergy::WORKBENCH_LANE_FLAG_PINNED, 1 << 0);
    assert_eq!(
        liminika_core::stigmergy::WORKBENCH_LANE_FLAG_REPULSION,
        1 << 1
    );
    assert_eq!(
        liminika_core::stigmergy::WORKBENCH_LANE_FLAG_SANDBOX,
        1 << 2
    );

    // Unified v0.1.3 2 Primitives x 2 Drivers & Distortion ABI types
    assert_eq!(size_of::<liminika_core::abi::lmnk_attribution_tag_t>(), 1);
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_core_primitive_state_t>(),
        64
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_core_primitive_state_t>(),
        64
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_core_primitive_state_t, stigmergy),
        32
    );
    assert_eq!(size_of::<liminika_core::abi::lmnk_lacuna_partition_t>(), 16);
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_lacuna_partition_t>(),
        16
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_lacuna_partition_t, exterior_mask),
        4
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_lacuna_partition_t,
            boundary_sensitivity_q7
        ),
        8
    );
    assert_eq!(size_of::<liminika_core::abi::lmnk_spine_state_t>(), 32);
    assert_eq!(align_of::<liminika_core::abi::lmnk_spine_state_t>(), 16);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_spine_state_t, pinned_compute_mask),
        8
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_spine_state_t,
            whole_token_bypass_active
        ),
        24
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_distortion_delta_t>(),
        128
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_distortion_delta_t>(),
        64
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_distortion_delta_t, intent_delta),
        16
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_distortion_delta_t, delta_phase_q7),
        64
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_distortion_tension_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_distortion_tension_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_distortion_tension_t,
            immediate_damping_events
        ),
        8
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_distortion_tension_t,
            slow_excitation_events
        ),
        12
    );
}
