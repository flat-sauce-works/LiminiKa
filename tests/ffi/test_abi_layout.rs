// name: tests/ffi/test_abi_layout.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test verifying C-ABI struct size, alignment, and field offset invariants.

#[test]
fn test_gcso_types_abi_layout_invariants() {
    use std::mem::{align_of, offset_of, size_of};

    // Verify 128-byte, 128-byte-aligned gcso_pointer_trail_t
    assert_eq!(size_of::<liminika_core::abi::gcso_pointer_trail_t>(), 128);
    assert_eq!(align_of::<liminika_core::abi::gcso_pointer_trail_t>(), 128);

    // Verify 64-byte, 16-byte-aligned gcso_config_t
    assert_eq!(size_of::<liminika_core::abi::gcso_config_t>(), 64);
    assert_eq!(align_of::<liminika_core::abi::gcso_config_t>(), 16);

    // Verify 64-byte, 64-byte-aligned gcso_ledger_slot_t
    assert_eq!(size_of::<liminika_core::abi::gcso_ledger_slot_t>(), 64);
    assert_eq!(align_of::<liminika_core::abi::gcso_ledger_slot_t>(), 64);

    // Verify Cognitive Trajectory ABI types
    assert_eq!(size_of::<liminika_core::traits::CognitiveStepOp>(), 1);
    assert_eq!(
        size_of::<liminika_core::traits::CognitiveCircuitActivation>(),
        8
    );
    assert_eq!(
        align_of::<liminika_core::traits::CognitiveCircuitActivation>(),
        8
    );
    assert_eq!(
        offset_of!(
            liminika_core::traits::CognitiveCircuitActivation,
            trajectory_depth
        ),
        7
    );
    assert_eq!(
        size_of::<liminika_core::abi::gcso_cognitive_trajectory_ledger_t>(),
        8
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_cognitive_trajectory_ledger_t>(),
        8
    );

    // Verify GatewayFrame (128 B, 64 B align) and CrystallizedAttractorRecord (128 B, 64 B align)
    assert_eq!(size_of::<liminika_core::abi::gcso_gateway_frame_t>(), 128);
    assert_eq!(align_of::<liminika_core::abi::gcso_gateway_frame_t>(), 64);
    assert_eq!(
        offset_of!(liminika_core::abi::gcso_gateway_frame_t, phase_deltas),
        64
    );
    assert_eq!(
        size_of::<liminika_core::abi::gcso_crystallized_attractor_record_t>(),
        128
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_crystallized_attractor_record_t>(),
        64
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_crystallized_attractor_record_t,
            phase_profile
        ),
        64
    );

    // Verify Stigmergic Reconstitution report (16 B, 16 B align)
    assert_eq!(
        size_of::<liminika_core::abi::gcso_reconstitution_report_t>(),
        16
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_reconstitution_report_t>(),
        16
    );

    // Verify Liminal & 2-Layer Autonomous Core Stack structures
    assert_eq!(
        size_of::<liminika_core::abi::gcso_epistemic_horizon_reading_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_epistemic_horizon_reading_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_epistemic_horizon_reading_t,
            regime_code
        ),
        28
    );

    assert_eq!(
        size_of::<liminika_core::abi::gcso_liminal_aperture_field_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_liminal_aperture_field_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_liminal_aperture_field_t,
            adaptation_epochs
        ),
        28
    );

    assert_eq!(
        size_of::<liminika_core::abi::gcso_stigmergic_granule_field_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_stigmergic_granule_field_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_stigmergic_granule_field_t,
            adaptation_epochs
        ),
        28
    );

    assert_eq!(
        size_of::<liminika_core::abi::gcso_stigmergic_venn_serendipity_report_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_stigmergic_venn_serendipity_report_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_stigmergic_venn_serendipity_report_t,
            serendipitous_seed_word
        ),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_stigmergic_venn_serendipity_report_t,
            serendipity_triggered
        ),
        30
    );

    assert_eq!(
        size_of::<liminika_core::abi::gcso_autonomous_core_layer_stack_t>(),
        128
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_autonomous_core_layer_stack_t>(),
        64
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_autonomous_core_layer_stack_t,
            liminal_layer
        ),
        0
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_autonomous_core_layer_stack_t,
            stigmergic_layer
        ),
        32
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_autonomous_core_layer_stack_t,
            ego_layer
        ),
        64
    );

    assert_eq!(
        size_of::<liminika_core::abi::gcso_liminal_topology_snapshot_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_liminal_topology_snapshot_t>(),
        16
    );

    assert_eq!(
        size_of::<liminika_core::abi::gcso_liminal_boundary_delta_t>(),
        16
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_liminal_boundary_delta_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_liminal_boundary_delta_t,
            kinetic_energy_q7
        ),
        10
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_liminal_boundary_delta_t,
            stationary_shortcut_ready
        ),
        11
    );

    assert_eq!(size_of::<liminika_core::abi::gcso_ego_intent_axes_t>(), 16);
    assert_eq!(align_of::<liminika_core::abi::gcso_ego_intent_axes_t>(), 16);
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_ego_intent_axes_t,
            preferred_slot_mask
        ),
        8
    );
    assert_eq!(
        offset_of!(liminika_core::abi::gcso_ego_intent_axes_t, signature_hash),
        12
    );

    assert_eq!(
        size_of::<liminika_core::abi::gcso_stigmergic_fragment_ensemble_report_t>(),
        16
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_stigmergic_fragment_ensemble_report_t>(),
        16
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_stigmergic_fragment_ensemble_report_t,
            quorum_agreed_granule_mask
        ),
        8
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_stigmergic_fragment_ensemble_report_t,
            ego_tiebreak_granule_mask
        ),
        10
    );

    assert_eq!(
        size_of::<liminika_core::abi::gcso_liminal_ego_field_t>(),
        64
    );
    assert_eq!(
        align_of::<liminika_core::abi::gcso_liminal_ego_field_t>(),
        64
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_liminal_ego_field_t,
            counter_basin_thrust_q7
        ),
        40
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_liminal_ego_field_t,
            basin_escape_floor_q7
        ),
        41
    );
    assert_eq!(
        liminika_core::abi::GCSO_ABI_VERSION_STRING,
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(
        liminika_core::traits::GCSO_CRYSTAL_RECORD_VERSION,
        liminika_core::abi::GCSO_ABI_VERSION_HEX
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_liminal_ego_field_t,
            ego_signature_hash
        ),
        48
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::gcso_liminal_ego_field_t,
            last_intervened_mask
        ),
        56
    );
}
