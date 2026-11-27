// name: tests/ffi/test_abi_layout.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test verifying C-ABI v0.4.0 struct size, alignment, and field offset invariants.

#[test]
fn test_lmnk_types_abi_layout_invariants() {
    use std::mem::{align_of, needs_drop, offset_of, size_of};

    assert_eq!(liminika_core::abi::LMNK_ABI_VERSION_HEX, 0x0000_0400);
    assert_eq!(liminika_core::abi::LMNK_ABI_VERSION_STRING, "0.4.0");
    assert_eq!(
        liminika_core::abi::LMNK_ABI_VERSION_STRING,
        env!("CARGO_PKG_VERSION")
    );
    assert_eq!(
        liminika_core::stigmergy::LMNK_PHASE_ANCHOR_VERSION,
        liminika_core::abi::LMNK_ABI_VERSION_HEX
    );

    // Verify ArenaSpan (8 B, align 4) and Rank1Descriptor (32 B, align 32, zero raw pointers)
    assert_eq!(size_of::<liminika_core::abi::lmnk_arena_span_t>(), 8);
    assert_eq!(align_of::<liminika_core::abi::lmnk_arena_span_t>(), 4);
    assert_eq!(offset_of!(liminika_core::abi::lmnk_arena_span_t, offset), 0);
    assert_eq!(offset_of!(liminika_core::abi::lmnk_arena_span_t, len), 4);

    assert_eq!(size_of::<liminika_core::abi::lmnk_rank1_descriptor_t>(), 32);
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_rank1_descriptor_t>(),
        32
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_rank1_descriptor_t, u_span),
        8
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_rank1_descriptor_t, v_span),
        16
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_rank1_descriptor_t, gain_offset),
        24
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_rank1_descriptor_t, scale_factor),
        28
    );

    // Verify 64-byte, 64-byte-aligned lmnk_config_t with leading struct_size
    assert_eq!(size_of::<liminika_core::abi::lmnk_config_t>(), 64);
    assert_eq!(align_of::<liminika_core::abi::lmnk_config_t>(), 64);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_config_t, struct_size),
        0
    );
    assert_eq!(offset_of!(liminika_core::abi::lmnk_config_t, head_dim), 4);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_config_t, execution_flags),
        44
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_config_t, guard_hold_tokens),
        48
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_config_t, max_observe_stride),
        49
    );
    assert_eq!(offset_of!(liminika_core::abi::lmnk_config_t, reserved), 50);

    // Verify versioned fixed-width v0.4.0 parameter structs and views
    assert_eq!(size_of::<liminika_core::abi::lmnk_abi_info_t>(), 64);
    assert_eq!(align_of::<liminika_core::abi::lmnk_abi_info_t>(), 8);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_abi_info_t, capability_flags),
        24
    );
    assert_eq!(size_of::<liminika_core::abi::lmnk_runtime_policy_t>(), 64);
    assert_eq!(align_of::<liminika_core::abi::lmnk_runtime_policy_t>(), 8);
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_runtime_policy_t,
            baseline_pressure_q15
        ),
        40
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_runtime_policy_t, reserved),
        45
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_turn_begin_params_t>(),
        16
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_decode_step_params_t>(),
        48
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_decode_step_params_t>(),
        16
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_decode_step_params_t, flags),
        4
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_decode_step_params_t, decode_input),
        16
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_decode_step_report_t>(),
        64
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_decode_step_report_t>(),
        8
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_decode_step_report_t, dispatch),
        8
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_decode_step_report_t, reserved),
        48
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_runtime_state_view_t>(),
        64
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_runtime_state_view_t>(),
        8
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_runtime_state_view_t,
            active_compute_mask
        ),
        32
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_runtime_state_view_t,
            envelope_contraction_q7
        ),
        46
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_runtime_state_view_t,
            envelope_violation_epochs
        ),
        48
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_runtime_state_view_t,
            envelope_baseline_pressure_q15
        ),
        52
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_runtime_state_view_t, reserved),
        56
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_system_pressure_reading_t>(),
        8
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_system_pressure_reading_t>(),
        2
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_compute_dispatch_mask_t>(),
        40
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_compute_dispatch_mask_t>(),
        8
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_spine_decode_input_t>(),
        32
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_spine_decode_input_t>(),
        16
    );
    assert_eq!(size_of::<liminika_core::abi::lmnk_boundary_signal_t>(), 32);
    assert_eq!(align_of::<liminika_core::abi::lmnk_boundary_signal_t>(), 32);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_boundary_signal_t, reserved),
        26
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_boundary_signal_t, epoch_id),
        28
    );
    assert_eq!(size_of::<liminika_core::abi::lmnk_envelope_bounds_t>(), 32);
    assert_eq!(align_of::<liminika_core::abi::lmnk_envelope_bounds_t>(), 32);
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_envelope_feedback_t>(),
        16
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_envelope_feedback_t>(),
        16
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_envelope_feedback_t, contraction_q7),
        11
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_envelope_feedback_t, breach_mask),
        12
    );
    assert_eq!(size_of::<liminika_core::abi::lmnk_envelope_state_t>(), 64);
    assert_eq!(align_of::<liminika_core::abi::lmnk_envelope_state_t>(), 64);
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_envelope_state_t,
            baseline_pressure_q15
        ),
        24
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_envelope_state_t, contraction_q7),
        33
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_envelope_state_t, reserved),
        37
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_envelope_step_params_t>(),
        96
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_envelope_step_params_t>(),
        32
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_envelope_step_params_t, bounds),
        32
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_envelope_step_params_t, signal),
        64
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_boundary_emit_params_t>(),
        16
    );
    assert_eq!(
        align_of::<liminika_core::abi::lmnk_boundary_emit_params_t>(),
        4
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_boundary_emit_params_t, epoch_id),
        12
    );
    assert_eq!(size_of::<liminika_core::drivers::SystemSpineState>(), 32);
    assert_eq!(align_of::<liminika_core::drivers::SystemSpineState>(), 32);
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_lacuna_eval_params_t>(),
        24
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_container_verify_report_t>(),
        40
    );

    // Verify transient pointer-carrying host call envelopes (target_pointer_width-aware)
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_rope_steer_params_t, query_tensor),
        40
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_rope_procrustes_params_t, q_before),
        32
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_rank1_eval_params_t, desc),
        32
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_gateway_steer_params_t,
            query_tensor
        ),
        40
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_section_input_t, payload),
        16
    );
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_container_write_params_t, sections),
        24
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_container_section_view_t,
            payload_ptr
        ),
        24
    );
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(
            size_of::<liminika_core::abi::lmnk_rope_steer_params_t>(),
            56
        );
        assert_eq!(
            size_of::<liminika_core::abi::lmnk_rope_procrustes_params_t>(),
            56
        );
        assert_eq!(
            size_of::<liminika_core::abi::lmnk_rank1_eval_params_t>(),
            64
        );
        assert_eq!(
            size_of::<liminika_core::abi::lmnk_gateway_steer_params_t>(),
            88
        );
        assert_eq!(size_of::<liminika_core::abi::lmnk_section_input_t>(), 24);
        assert_eq!(
            size_of::<liminika_core::abi::lmnk_container_write_params_t>(),
            56
        );
        assert_eq!(
            size_of::<liminika_core::abi::lmnk_container_section_view_t>(),
            32
        );
    }

    // Verify RuntimeContext & handshake consistency
    assert_eq!(size_of::<liminika_core::runtime::RuntimeContext>(), 5120);
    assert_eq!(align_of::<liminika_core::runtime::RuntimeContext>(), 64);
    assert!(!needs_drop::<liminika_core::runtime::RuntimeContext>());
    let mut info = liminika_core::abi::lmnk_abi_info_t::default();
    // SAFETY: Valid stack struct initialized with `struct_size`.
    let status = unsafe { liminika_core::abi::lmnk_abi_info(&mut info) };
    assert_eq!(status, liminika_core::abi::LMNK_OK);
    assert_eq!(
        info.context_size as usize,
        size_of::<liminika_core::runtime::RuntimeContext>()
    );
    assert_eq!(
        info.context_align as usize,
        align_of::<liminika_core::runtime::RuntimeContext>()
    );
    assert_eq!(info.snapshot_max_bytes, 576);

    // Verify core Portable PODs and renamed v0.4.0 fields
    assert_eq!(size_of::<liminika_core::abi::lmnk_pointer_trail_t>(), 128);
    assert_eq!(align_of::<liminika_core::abi::lmnk_pointer_trail_t>(), 64);
    assert_eq!(size_of::<liminika_core::abi::lmnk_gateway_frame_t>(), 128);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_gateway_frame_t, root_cause_code),
        48
    );
    assert_eq!(size_of::<liminika_core::abi::lmnk_causal_report_t>(), 16);
    assert_eq!(
        offset_of!(liminika_core::abi::lmnk_causal_report_t, root_cause_code),
        12
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_phase_anchor_record_t>(),
        128
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_phase_anchor_record_t,
            root_cause_code
        ),
        59
    );
    assert_eq!(
        size_of::<liminika_core::abi::lmnk_liminal_ego_state_t>(),
        64
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_liminal_ego_state_t,
            preferred_granule_phase_q7
        ),
        0
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_liminal_ego_state_t,
            basin_resistance_q7
        ),
        40
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_liminal_ego_state_t,
            escape_floor_q7
        ),
        41
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_liminal_ego_state_t,
            boundary_crossing_q7
        ),
        42
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_ego_intent_axes_t,
            boundary_crossing_q7
        ),
        1
    );
    assert_eq!(
        offset_of!(
            liminika_core::abi::lmnk_topology_snapshot_t,
            spawned_anchors
        ),
        31
    );
    assert_eq!(
        size_of::<liminika_core::stigmergy::StigmergicSubstrate>(),
        2816
    );
    assert_eq!(
        align_of::<liminika_core::stigmergy::StigmergicSubstrate>(),
        64
    );
    assert_eq!(size_of::<liminika_core::abi::lmnk_lacuna_partition_t>(), 16);
    assert_eq!(size_of::<liminika_core::abi::lmnk_container_header_t>(), 64);
    assert_eq!(size_of::<liminika_core::abi::lmnk_section_entry_t>(), 16);
    assert_eq!(size_of::<liminika_core::abi::lmnk_topology_matrix_t>(), 64);
    assert_eq!(size_of::<liminika_core::runtime::RamPhaseKvEntry>(), 128);
}
