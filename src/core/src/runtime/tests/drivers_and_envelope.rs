// name: src/core/src/runtime/tests/drivers_and_envelope.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Driver coordination, Spine/Ego/Envelope, and telemetry tests.

use super::*;

struct MockSubstrateComposite {
    head_dim: u32,
    num_heads: u32,
    default_phase: PhaseQ7,
}

impl PointerSubstrate for MockSubstrateComposite {
    fn reset(&mut self) -> LmnkResult<()> {
        Ok(())
    }

    fn step_pointer(&mut self, current_ptr: u64, trail_out: &mut PointerTrail) -> LmnkResult<()> {
        trail_out.record_pointer_transition(current_ptr, 1, 0.25);
        trail_out.accumulate_uniform_phase(self.default_phase);
        Ok(())
    }

    fn update_cellular_state_express(
        &mut self,
        _trail_id: u32,
        _phase_delta: PhaseQ7,
    ) -> LmnkResult<()> {
        Ok(())
    }

    fn pull_trail_to_attractor(
        &mut self,
        _trail_id: u32,
        _anchor_id: u32,
        _pull_force: f32,
    ) -> LmnkResult<()> {
        Ok(())
    }

    fn link_hallucinated_trails(
        &mut self,
        _src_trail_id: u32,
        _dst_trail_id: u32,
    ) -> LmnkResult<()> {
        Ok(())
    }

    fn capacity(&self) -> u32 {
        256
    }

    fn active_count(&self) -> u32 {
        1
    }
}

impl FastPathLedger for MockSubstrateComposite {
    fn evaluate_auto_tune(
        &self,
        _slot: &LedgerSlot,
        _config_out: &mut RuntimeConfig,
    ) -> LmnkResult<()> {
        Ok(())
    }
}

#[test]
fn test_canonical_modules_and_substrate_composite() {
    fn assert_substrate<T: StigmergicSubstrateOps + SubstrateGatewayExt>(
        substrate: &T,
    ) -> (u32, u32) {
        (
            THRESHOLD_STIGMERGY_WIDE,
            substrate
                .inspect_substrate_horizon_partition(THRESHOLD_STIGMERGY_WIDE)
                .0,
        )
    }

    let mut substrate = MockSubstrateComposite {
        head_dim: 8,
        num_heads: 2,
        default_phase: 24,
    };
    assert_eq!(
        assert_substrate(&substrate),
        (THRESHOLD_STIGMERGY_WIDE, THRESHOLD_STIGMERGY_WIDE)
    );

    let mut mask = PagedBitmask::default();
    mask.set_bit_branchless(7);
    assert!(mask.test_bit_branchless(7));

    let deg = AlignedBuffer32::new([1.0f32, 3.0, 2.0, 0.5]);
    let lmax = eval_gershgorin_lambda_max(&deg.as_aligned_slice()).unwrap();
    assert!((lmax - 6.0).abs() < 1e-5);

    let mut ledger = LedgerSlot::default();
    let bloom = ContrastivePhaseBloomFilter::new();
    let mut q_buf = AlignedBuffer32::new([1.0f32; 16]);
    let mut trail = PointerTrail::default();
    let input_ptr = TaggedPtr::pack_action(
        0x0400,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_EXPRESS_PHASE,
        12,
    );

    let density = substrate
        .execute_stigmergic_token_cascade(
            crate::stigmergy::TokenCascadeParams {
                input_ptr: input_ptr.raw(),
                bloom_filter: &bloom,
                head_dim: substrate.head_dim,
                num_heads: substrate.num_heads,
                repulsion_threshold: 2,
                repulsion_scale_u7: 128,
                phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
                max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            },
            &mut ledger,
            Some(&mut q_buf.as_aligned_slice_mut()),
            None,
            &mut trail,
        )
        .unwrap();
    assert_eq!(density, 0);
    assert_ne!(
        ledger.telemetry_mini_ledger[0] & LEDGER_METRIC_PHASE_STEERED,
        0
    );
}

#[test]
fn test_phase_steering_closed_loop_gateway_signal() {
    let engine_gw = PhaseSteeringEngine::new(4, 16).unwrap();
    let sys = SystemConstraint::new((), 0.1, 0.0, 0xFF);
    let fact = ObjectiveFact::new([40i8; 64], 2.5, 0x0010_0000, 42);
    let persona = SubjectivePersona::new([0i8; 64], 0.0, 0, 0);
    let mut q_data = [0.0f32; 64];
    for pair in q_data.as_chunks_mut::<2>().0 {
        pair[0] = 1.0;
    }
    let mut q_gw = AlignedBuffer32::new(q_data);
    let mut trail_gw = PointerTrail::default();
    let (frame, report) = GatewayPhaseSteeringExt::steer_gateway_signal_safe(
        &engine_gw,
        &(sys, fact, persona),
        &mut trail_gw,
        64,
        &mut q_gw.as_aligned_slice_mut(),
        None,
        None,
    )
    .unwrap();
    assert_eq!(frame.target_anchor_id, 42);
    assert_eq!(report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(trail_gw.target_anchor_id, 42);
}

#[test]
fn test_lora_phase_kv_cache_replay_unified_manifold_and_entropy_tunneling() {
    struct MockColdSystem;
    impl KvKeyframeCache for MockColdSystem {
        fn seek_to_token(
            &mut self,
            token_index: u32,
            header_out: &mut KvKeyframeHeader,
        ) -> LmnkResult<()> {
            header_out.token_index = token_index;
            Ok(())
        }
    }
    let sys = MockColdSystem;
    let mut cfg = RuntimeConfig::default();
    init_default_config(&mut cfg).unwrap();
    assert_eq!(validate_config_invariants(&cfg), Ok(()));
    assert_eq!(
        abi_version(),
        (
            LMNK_ABI_VERSION_MAJOR,
            LMNK_ABI_VERSION_MINOR,
            LMNK_ABI_VERSION_PATCH
        )
    );

    let hashed_p = crate::stigmergy::hash_prompt_to_q7_phases("LiminiKa Persona Anchor", 1);
    assert!(hashed_p.iter().any(|&x| x != 0));

    let lora_a = AlignedBuffer32::new([1.0f32; 16]);
    let lora_b = AlignedBuffer32::new([1.0f32; 16]);
    let mut rank1_arena = [0.0f32; 24];
    let mut rank1_desc = Rank1Descriptor::default();
    let mut prof_out = AlignedBuffer16::new([0i8; 16]);
    let mut scales_out = AlignedBuffer32::new([0.0f32; 4]);
    let written_arena = project_lora_with_block_scales(
        &lora_a.as_aligned_slice(),
        &lora_b.as_aligned_slice(),
        crate::foundation::LoraDimensions {
            rank: 2,
            dim_in: 8,
            dim_out: 8,
        },
        &mut rank1_arena,
        &mut rank1_desc,
        &mut prof_out.as_aligned_slice_mut(),
        &mut scales_out.as_aligned_slice_mut(),
    )
    .unwrap();
    assert_eq!(written_arena, 24);
    assert_eq!(rank1_desc.u_span, crate::foundation::ArenaSpan::new(0, 8));
    assert_eq!(rank1_desc.v_span, crate::foundation::ArenaSpan::new(8, 8));
    assert_eq!(rank1_desc.gain_offset, 16);
    assert!((scales_out.data[0] - 2.0).abs() < 1e-5);
    let mut y_lora = AlignedBuffer32::new([0.0f32; 8]);
    let x_lora = AlignedBuffer32::new([1.0f32; 8]);
    crate::foundation::eval_rank1(
        &mut y_lora.as_aligned_slice_mut(),
        &x_lora.as_aligned_slice(),
        &rank1_arena,
        &rank1_desc,
    )
    .unwrap();
    assert!(y_lora.data[0] > 1.0);

    let phases = AlignedBuffer16::new([32i8; 16]);
    let weights = AlignedBuffer32::new([1.0f32; 16]);
    let mut header = KvKeyframeHeader::default();
    let (r_bar, kappa) = sys
        .aggregate_pruned_kv_circular_stats(
            &phases.as_aligned_slice(),
            &weights.as_aligned_slice(),
            &mut header,
        )
        .unwrap();
    assert!((r_bar - 1.0).abs() < 1e-4);
    assert!(kappa > 10.0);
    assert!((header.sparse_scalar_residual - (PI * 0.25)).abs() < 1e-3);

    // Verify Limina KV-keyframe P-Frame Chain Replay & Joint Probability Log-Likelihood
    let icache_anchor = [24i8; MAX_ATTENTION_HEADS];
    let pframes = [[4i8; MAX_ATTENTION_HEADS], [4i8; MAX_ATTENTION_HEADS]];
    let mut replayed = [0i8; MAX_ATTENTION_HEADS];
    let mut gop_hdr = KvKeyframeHeader::default();
    let (gop_lp, gop_r, gop_k, gop_rep) = sys
        .replay_liminal_pframe_chain_with_joint_log_prob(
            &icache_anchor,
            &pframes,
            1,
            0.25,
            &mut replayed,
            &mut gop_hdr,
        )
        .unwrap();
    assert_eq!(gop_rep, 0);
    assert_eq!(replayed[0], 32);
    assert!(gop_lp > -0.05 && (gop_r - 1.0).abs() < 1e-4 && gop_k > 10.0);

    let ledger_slot = LedgerSlot {
        mode: 1,
        ..Default::default()
    };
    let src_phases = [28i8; MAX_ATTENTION_HEADS];
    let mut pkt_buf = AlignedBuffer64::<u8, 64>::zeroed();
    let crc_tag = encode_phase_packet(
        &ledger_slot,
        ATTRACTOR_SLOT_PERSONA,
        ACTION_PRESET_FULL_ORCHESTRATION,
        &src_phases,
        &mut pkt_buf.as_aligned_slice_mut(),
    )
    .unwrap();
    let mut decoded_phases = [0i8; MAX_ATTENTION_HEADS];
    let recovered_tag = decode_and_verify_phase_packet(
        &ledger_slot,
        &pkt_buf.as_aligned_slice(),
        crc_tag,
        &mut decoded_phases,
    )
    .unwrap();
    assert_eq!(decoded_phases[0], 28);
    assert_eq!(
        recovered_tag,
        pack_action_slot_tag(ATTRACTOR_SLOT_PERSONA, ACTION_PRESET_FULL_ORCHESTRATION)
    );

    let mut trail = PointerTrail {
        current_ptr: TaggedPtr::pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        )
        .raw(),
        ..Default::default()
    };

    let mut bank = [[0i8; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS];
    bank[usize::from(ATTRACTOR_SLOT_PERSONA)] = [32i8; MAX_ATTENTION_HEADS];
    bank[usize::from(ATTRACTOR_SLOT_KNOWLEDGE)] = [32i8; MAX_ATTENTION_HEADS];
    bank[usize::from(ATTRACTOR_SLOT_REPULSION)] = [-96i8; MAX_ATTENTION_HEADS];
    bank[usize::from(ATTRACTOR_SLOT_USER_CONTEXT)] = [32i8; MAX_ATTENTION_HEADS];
    let weights = [0.0, 2.0, 2.0, 1.5, 2.0, 0.0, 0.0, 0.0];
    let m = with_slot_pair_coupled_symmetric(
        0,
        u32::from(ATTRACTOR_SLOT_PERSONA),
        u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
        true,
    );
    let coupling = with_slot_pair_coupled_symmetric(
        m,
        u32::from(ATTRACTOR_SLOT_PERSONA),
        u32::from(ATTRACTOR_SLOT_USER_CONTEXT),
        true,
    );
    let (final_pull, mean_coh, neighbors) =
        trail.apply_neighborhood_synergy_from_bank(&bank, &weights, coupling, 0.5, 0.5);
    assert_eq!(neighbors, 2);
    assert!((mean_coh - 1.0).abs() < 1e-4);
    assert!((final_pull - 6.0).abs() < 0.2);
    assert_ne!(trail.flags & TRAIL_FLAG_SYNERGY_RESONANCE, 0);
}

#[test]
fn test_substrate_gateway_kmeans_and_rot_dim_resolution() {
    let substrate =
        StigmergicSubstrate::with_dimensions_and_thresholds(8, 2, 16, StigmergicThresholds::new())
            .unwrap();
    let mut active_trail = PointerTrail {
        stigmergic_density: 1.2,
        accumulated_phase_delta: [24i8; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let sys_cfg = RuntimeConfig {
        head_dim: 8,
        num_heads: 2,
        phase_clamp_max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
        phase_step_min_rad: 0.01,
        ..Default::default()
    };
    let sys = SystemConstraint::new(&sys_cfg, 0.10, 0.05, 0xFF);
    let fact = ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 2.0, 0x0010_0000, 77);
    let persona = SubjectivePersona::new([24i8; MAX_ATTENTION_HEADS], 2.5, 0x0020_0000, 42);
    let composite = (sys, fact, persona);
    let (gw_frame, gw_words) = composite.synthesize_gateway_frame_with_words();
    let mut centroids = [PhaseAnchorRecord::default(); 2];
    let (c_idx, c_aid, c_coh, _mass, _spawned) =
        PhaseAnchorRecord::step_online_toroidal_kmeans_frame_with_words_64(
            &mut centroids,
            &gw_frame,
            &gw_words,
            0.55,
            Some(&mut active_trail),
        );
    assert_eq!(c_idx, 0);
    assert_eq!(c_aid, 42);
    assert!(c_coh > 0.90);
    assert_eq!(gw_frame.target_anchor_id, 42);
    assert_eq!(gw_frame.resolve_autonomous_rot_dim(16, 0.20, 1.5), 4);
    assert_eq!(substrate.resolve_autonomous_rot_dim(0x04, 0xFF, 1.5), 2);
    assert!(centroids[0].verify_integrity());
}

#[test]
fn test_gateway_workbench_bitmask_repulsion_and_sandbox_isolation() {
    struct Geom;
    impl RopeSteeringTarget for Geom {
        fn rope_geometry(&self) -> RopeGeometryDescriptor {
            RopeGeometryDescriptor {
                head_dim: 8,
                num_heads: MAX_ATTENTION_HEADS,
                rotary_dim: 8,
                layout: RopePairLayout::Interleaved,
            }
        }
    }

    struct MockSignal {
        flags: u8,
    }
    impl CausalProvenanceSignal for MockSignal {
        fn causal_provenance_bits(&self) -> u32 {
            CAUSAL_AXIS_OBJECTIVE_FACT
        }
    }
    impl ManifoldGatewaySignal for MockSignal {
        fn synthesize_gateway_frame(&self) -> GatewayFrame {
            GatewayFrame {
                effective_energy: 1.0,
                phase_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
                phase_step_min_rad: 0.01,
                effective_inv_temp: 1.0,
                fact_persona_coherence: 1.0,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                action_mode_flags: self.flags,
                ..GatewayFrame::default()
            }
        }
    }

    let mut substrate = StigmergicSubstrate::with_dimensions_and_thresholds(
        8,
        MAX_ATTENTION_HEADS as u32,
        16,
        StigmergicThresholds::new(),
    )
    .unwrap();
    // Physical lane 0 = standard, physical lane 1 = repulsion, physical lane 2 = sandbox.
    assert_eq!(
        substrate
            .allocate_lane(&WorkbenchSlotSpec::auto_free(
                1.0,
                [20; MAX_ATTENTION_HEADS]
            ))
            .unwrap(),
        0
    );
    assert_eq!(
        substrate
            .allocate_lane(&WorkbenchSlotSpec::auto_repulsion(
                1.0,
                [-40; MAX_ATTENTION_HEADS]
            ))
            .unwrap(),
        1
    );
    assert_eq!(
        substrate
            .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(
                1.0,
                [40; MAX_ATTENTION_HEADS]
            ))
            .unwrap(),
        2
    );

    let stack = CoreStateStack::new();
    let delta = LiminalBoundaryDelta::default();
    let horizon = EpistemicHorizonReading::default();
    let overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0x0F,
        lacuna_exterior_mask: 0xF0,
        ..OverlapDiscoveryReport::default()
    };
    let core_ctx = Some(CoreLayerEvalContext {
        stack: &stack,
        delta: &delta,
        horizon: &horizon,
        overlap: &overlap,
        derivative_alpha_q7: DEFAULT_DERIVATIVE_ALPHA_Q7,
        lacuna_partition: None,
        envelope: None,
        ego_net: None,
        tension: None,
    });
    let thresh = StigmergicThresholds::new();

    // 1. Normal mode: lane 0 acts on granules 0..4 (+10), sandbox lane 2 acts ONLY on
    //    unobserved granules 4..8 (+20) without polluting overlap_core_mask, repulsion lane 1 is idle.
    let mut ledger = LedgerSlot::default();
    let mut trail = PointerTrail::default();
    let _ = substrate
        .orchestrate_gateway_layer_with_scale_and_core_layers_safe(
            GatewayLayerInput {
                signal: &MockSignal {
                    flags: ACTION_PRESET_EXPRESS_PHASE,
                },
                geometry: &Geom,
                core_ctx,
                scale: StigmergicScaleGranularity::PointerBit,
                entropy_h_tilde: 0.30,
                unresolved_damping_u7: 128,
            },
            &thresh,
            &mut ledger,
            &mut trail,
            GatewayLayerTargets {
                query_tensor: None,
                rank1_hidden: None,
                logits_opt: None,
            },
        )
        .unwrap();
    assert_eq!(trail.accumulated_phase_delta[0], 10);
    assert_eq!(trail.accumulated_phase_delta[40], 20);

    // 2. Repulsion mode: ONLY repulsion lane 1 acts on granules 0..4 (-20); lane 0 & sandbox 2 are idle.
    let mut trail_rep = PointerTrail::default();
    let _ = substrate
        .orchestrate_gateway_layer_with_scale_and_core_layers_safe(
            GatewayLayerInput {
                signal: &MockSignal {
                    flags: ACTION_MODE_PHASE_STEER | ACTION_MODE_PHASE_REPULSE,
                },
                geometry: &Geom,
                core_ctx,
                scale: StigmergicScaleGranularity::PointerBit,
                entropy_h_tilde: 0.30,
                unresolved_damping_u7: 128,
            },
            &thresh,
            &mut ledger,
            &mut trail_rep,
            GatewayLayerTargets {
                query_tensor: None,
                rank1_hidden: None,
                logits_opt: None,
            },
        )
        .unwrap();
    assert_eq!(trail_rep.accumulated_phase_delta[0], -20);
    assert_eq!(trail_rep.accumulated_phase_delta[40], 0);

    // 3. Settle sandbox lane 2 -> now contributes into overlap_core_mask granules 0..4 (+10 + +20 = +30).
    assert!(substrate.settle_sandbox_lane(2).unwrap());
    let mut trail_prom = PointerTrail::default();
    let _ = substrate
        .orchestrate_gateway_layer_with_scale_and_core_layers_safe(
            GatewayLayerInput {
                signal: &MockSignal {
                    flags: ACTION_PRESET_EXPRESS_PHASE,
                },
                geometry: &Geom,
                core_ctx,
                scale: StigmergicScaleGranularity::PointerBit,
                entropy_h_tilde: 0.30,
                unresolved_damping_u7: 128,
            },
            &thresh,
            &mut ledger,
            &mut trail_prom,
            GatewayLayerTargets {
                query_tensor: None,
                rank1_hidden: None,
                logits_opt: None,
            },
        )
        .unwrap();
    assert_eq!(trail_prom.accumulated_phase_delta[0], 30);
    assert_eq!(trail_prom.accumulated_phase_delta[40], 0);
}

#[test]
fn test_sandbox_hypothesis_lifecycle_nucleate_settle_and_release() {
    let baseline = ManifoldProbeBaseline {
        baseline_entropy: 0.30,
        baseline_inter_layer_coherence: 0.80,
        baseline_guard_noise_floor: 0.25,
        ..ManifoldProbeBaseline::new()
    };
    let mut substrate = StigmergicSubstrate::new();
    let mut bank = CoreTelemetryBank::with_mode(TelemetryVerbosityMode::Standard);
    let mut active_sandbox: Option<u8> = None;
    let horizon = EpistemicHorizonReading {
        regime_code: EpistemicRegime::InternalConflict as u8,
        ..EpistemicHorizonReading::default()
    };
    let delta = LiminalBoundaryDelta::default();
    let overlap = OverlapDiscoveryReport::default();
    let ego = LiminalEgoState::new();
    let candidate_words = [0x0A0A_0A0A_0A0A_0A0A_u64; NUM_STIGMERGIC_GRANULES];

    for _ in 0..20 {
        bank.record_step(StepTelemetryInput {
            decision: StigmergicActionDecision::ExpressPhaseSteer,
            horizon: &horizon,
            delta: &delta,
            overlap: &overlap,
            ego: &ego,
            entropy_gate_mode: 0,
            slid_in_bits: 0,
            slid_out_bits: 0,
            escalated: false,
            qos_throttled: false,
        });
        let _ = bank.update_assimilation_ema(0.75, 0.25, 0.65, &baseline, 0.15);
    }

    // 1. Ego=off must never nucleate a sandbox lane.
    let ev_off = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        SandboxLifecycleInput {
            decision: StigmergicActionDecision::ExpressPhaseSteer,
            ego_active: false,
            exterior_granule_mask: 0xF0,
            candidate_words: &candidate_words,
            candidate_weight: 0.5,
            drift_dwell_threshold: 3,
            settle_dwell_threshold: 3,
        },
    );
    assert_eq!(ev_off, SandboxLifecycleEvent::Idle);
    assert_eq!(active_sandbox, None);
    assert_eq!(substrate.slot_sandbox_mask, 0);

    // 2. Ego=active + exterior granules -> nucleates one sandbox lane masked strictly to 0xF0.
    let ev_nuc = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        SandboxLifecycleInput {
            decision: StigmergicActionDecision::ExpressPhaseSteer,
            ego_active: true,
            exterior_granule_mask: 0xF0,
            candidate_words: &candidate_words,
            candidate_weight: 0.5,
            drift_dwell_threshold: 3,
            settle_dwell_threshold: 3,
        },
    );
    let lane = match ev_nuc {
        SandboxLifecycleEvent::Nucleated(l) => l,
        other => panic!("expected Nucleated, got {other:?}"),
    };
    assert_eq!(active_sandbox, Some(lane));
    assert_ne!(substrate.slot_sandbox_mask & (1u8 << lane), 0);
    let row_start = usize::from(lane) * MAX_ATTENTION_HEADS;
    assert_eq!(substrate.slot_phase_bank.data[row_start], 0);
    assert_eq!(substrate.slot_phase_bank.data[row_start + 32], 10);

    // 3. Consecutive coherence improvement settles sandbox lane in-place.
    for _ in 0..3 {
        let _ = bank.update_assimilation_ema(0.25, 0.90, 0.20, &baseline, 0.15);
    }
    let ev_prom = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        SandboxLifecycleInput {
            decision: StigmergicActionDecision::ExpressPhaseSteer,
            ego_active: true,
            exterior_granule_mask: 0xF0,
            candidate_words: &candidate_words,
            candidate_weight: 0.5,
            drift_dwell_threshold: 3,
            settle_dwell_threshold: 3,
        },
    );
    assert_eq!(ev_prom, SandboxLifecycleEvent::Settled(lane));
    assert_eq!(active_sandbox, None);
    assert_eq!(substrate.slot_sandbox_mask & (1u8 << lane), 0);
    assert_ne!(substrate.slot_enabled_mask & (1u8 << lane), 0);

    // 4. Re-nucleate then verify RepulsionEscape releases sandbox immediately.
    for _ in 0..40 {
        let _ = bank.update_assimilation_ema(0.85, 0.15, 0.75, &baseline, 0.15);
    }
    let ev_nuc2 = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        SandboxLifecycleInput {
            decision: StigmergicActionDecision::ExpressPhaseSteer,
            ego_active: true,
            exterior_granule_mask: 0xF0,
            candidate_words: &candidate_words,
            candidate_weight: 0.5,
            drift_dwell_threshold: 3,
            settle_dwell_threshold: 3,
        },
    );
    assert!(matches!(ev_nuc2, SandboxLifecycleEvent::Nucleated(_)));
    let ev_rel = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        SandboxLifecycleInput {
            decision: StigmergicActionDecision::RepulsionEscape,
            ego_active: true,
            exterior_granule_mask: 0xF0,
            candidate_words: &candidate_words,
            candidate_weight: 0.5,
            drift_dwell_threshold: 3,
            settle_dwell_threshold: 3,
        },
    );
    assert_eq!(ev_rel, SandboxLifecycleEvent::Released(1));
    assert_eq!(active_sandbox, None);
    assert_eq!(substrate.slot_sandbox_mask, 0);

    // 5. ActiveOverdrive distortion also releases any active sandbox lane immediately.
    let mut active_ego = LiminalEgoState::from_preset(LiminalEgoPreset::High);
    active_ego.last_intervened_mask = EGO_TRIGGER_BASIN_CAPTURE;
    for _ in 0..40 {
        bank.record_step(StepTelemetryInput {
            decision: StigmergicActionDecision::RepulsionEscape,
            horizon: &horizon,
            delta: &delta,
            overlap: &overlap,
            ego: &active_ego,
            entropy_gate_mode: 0,
            slid_in_bits: 0,
            slid_out_bits: 0,
            escalated: false,
            qos_throttled: false,
        });
        let _ = bank.update_assimilation_ema(0.85, 0.15, 0.75, &baseline, 0.15);
    }
    let lane3 = substrate
        .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(
            0.5,
            [1; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    active_sandbox = Some(lane3);
    let ev_od = substrate.step_sandbox_hypothesis_lifecycle(
        &mut bank,
        &mut active_sandbox,
        SandboxLifecycleInput {
            decision: StigmergicActionDecision::ExpressPhaseSteer,
            ego_active: true,
            exterior_granule_mask: 0xF0,
            candidate_words: &candidate_words,
            candidate_weight: 0.5,
            drift_dwell_threshold: 3,
            settle_dwell_threshold: 3,
        },
    );
    assert_eq!(ev_od, SandboxLifecycleEvent::Released(1));
    assert_eq!(active_sandbox, None);
    assert_eq!(substrate.slot_sandbox_mask, 0);
}

#[test]
fn test_system_spine_physical_pinned_isolation() {
    let mut envelope = EnvelopeState::new();
    let baseline = GatewayFrame {
        dominant_slot: ATTRACTOR_SLOT_REPULSION,
        action_mode_flags: ACTION_MODE_PHASE_STEER | ACTION_MODE_PHASE_REPULSE,
        ..GatewayFrame::default()
    };
    let ego_attempt = GatewayFrame {
        dominant_slot: 1,
        action_mode_flags: ACTION_MODE_PHASE_STEER, // stripped PHASE_REPULSE
        phase_deltas: [25; MAX_ATTENTION_HEADS],
        ..GatewayFrame::default()
    };
    let mut guarded = ego_attempt;
    envelope.enforce_pinned_isolation_branchless(&baseline, &mut guarded);
    assert_eq!(guarded.dominant_slot, ATTRACTOR_SLOT_REPULSION);
    assert_ne!(guarded.action_mode_flags & ACTION_MODE_PHASE_REPULSE, 0);

    envelope.unobserved_locked = 1;
    let mut locked_frame = ego_attempt;
    envelope.enforce_pinned_isolation_branchless(&baseline, &mut locked_frame);
    assert_eq!(locked_frame.phase_deltas, baseline.phase_deltas);
}

#[test]
fn test_ego_differential_net_hops_consolidation_and_evaporation() {
    let ego = LiminalEgoState::from_preset(LiminalEgoPreset::Medium);
    let axes = EgoIntentAxes::new()
        .with_autonomy_q7(72)
        .with_boundary_crossing_q7(48);
    let mut net_1 = EgoDifferentialNet::from_ego_state_and_axes(&ego, &axes, 1);
    let mut net_3 = EgoDifferentialNet::from_ego_state_and_axes(&ego, &axes, 3);
    let envelope = EnvelopeState::new();
    let stig = StigmergicGranuleState::new();
    let tension = DistortionTension::new(true);
    let overlap = LacunaPartition8 {
        self_ref_mask: 0x0F,
        exterior_mask: 0x10,
        boundary_sensitivity_q7: 96,
        ..LacunaPartition8::default()
    };
    let mut f1 = GatewayFrame::default();
    let mut f3 = f1;

    let (_n1, h1, mask1) = net_1.evaluate_pframe_chain_with_mask_branchless(
        &ego, &envelope, &overlap, &stig, &tension, &mut f1,
    );
    let (_n3, h3, mask3) = net_3.evaluate_pframe_chain_with_mask_branchless(
        &ego, &envelope, &overlap, &stig, &tension, &mut f3,
    );
    let entry = mask3.trailing_zeros() as u8;
    assert_eq!(h1, 1);
    assert!(
        mask1.count_ones() >= 2,
        "1-hop parallel ensemble should fire multiple perspectives, got mask={mask1:#010b}"
    );
    assert_eq!(h3, 3);
    assert_ne!(f1.phase_deltas, f3.phase_deltas);

    net_1.epoch = 1;
    net_1.record_matched_activations(mask1);
    for (idx, cold) in net_1.node_cold.iter().take(3).enumerate() {
        if ((mask1 >> idx) & 1) != 0 {
            assert_eq!(cold.reinforce_count, 1);
            assert_eq!(cold.last_active_epoch, 1);
        }
    }

    for ep in 1..=5 {
        net_3.epoch = ep;
        net_3.record_chain_activation(usize::from(entry), h3);
    }
    assert!(net_3.node_cold[2].reinforce_count >= 4);
    let anchor = net_3.consolidate_and_evaporate_cold_path(6);
    assert!(anchor.is_some());
    assert_ne!(
        net_3.export_node(2).flags & DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME,
        0
    );
    assert!(net_3.node_routing[2].is_consolidated());

    // Stale unreinforced node evaporates when untouched across epochs.
    net_3.node_cold[0].last_active_epoch = 0;
    net_3.node_routing[0].pheromone_q7 = 16;
    let _ = net_3.consolidate_and_evaporate_cold_path(12);
    assert!(!net_3.is_node_active(0));
}

#[test]
fn test_entropy_controller_init_setters_and_reset() {
    let mut engine = EntropyController::default();
    assert_eq!(engine.state().bifurcation_threshold, 1.5);
    assert_eq!(engine.state().singularity_eps, DEFAULT_ENTROPY_EPS);
    assert_eq!(engine.state().sample_temperature, 1.0);
    assert_eq!(engine.hysteresis_band, DEFAULT_HYSTERESIS_BAND);
    assert_eq!(engine.tau_flux, DEFAULT_TAU_FLUX);
    assert!(engine.is_active());
    assert!(engine.is_normal_mode());
    assert_eq!(engine.as_bytes().len(), 128);

    engine.set_bifurcation_threshold(2.0);
    engine.set_bifurcation_threshold(-1.0);
    assert_eq!(engine.state().bifurcation_threshold, 2.0);

    engine.state.moving_z_entropy = 2.5;
    engine.state.active_branch_mode = ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION;
    engine.reset();
    assert_eq!(engine.moving_z_entropy(), 0.0);
    assert!(engine.is_normal_mode());

    let mut out_state = EntropyState::default();
    let h = engine
        .eval_surrogate_and_step(
            crate::drivers::spine::entropy::SurrogateEntropyInput {
                p_max: 0.5,
                p_2nd: 0.5,
                cosine_dist: 0.8,
                mu_dist: 0.2,
                sigma_dist: 0.1,
                alpha_z: 1.0,
                gamma_mod: 0.5,
            },
            &mut out_state,
        )
        .unwrap();
    assert!(h > 0.0 && out_state.moving_z_entropy == h);
}

#[test]
fn test_stationary_skip_and_spine_bypass_do_not_misdiagnose_passive_saturation() {
    use crate::drivers::LiminalEgoState;
    use crate::runtime::{EpistemicHorizonReading, LayerDistortionKind, LiminalBoundaryDelta};
    use crate::stigmergy::{
        DEFAULT_DERIVATIVE_ALPHA_Q7, OverlapDiscoveryReport, StigmergicActionDecision,
        StigmergicThresholds,
    };

    let mut bank = CoreTelemetryBank::default();
    let horizon = EpistemicHorizonReading::default();
    let delta = LiminalBoundaryDelta {
        stationary_shortcut_ready: 1,
        ..LiminalBoundaryDelta::default()
    };
    let overlap = OverlapDiscoveryReport::default();
    let ego_off = LiminalEgoState::default();

    // Simulate `byp=on` profile: high InertialLayerSkip (~72%) + modest PassiveSidecar (~19%)
    // + active ExpressPhaseSteer (~9%) + Spine whole-token bypass events.
    for i in 0..100 {
        let dec = if i < 72 {
            StigmergicActionDecision::InertialLayerSkip
        } else if i < 91 {
            StigmergicActionDecision::PassiveSidecar
        } else {
            StigmergicActionDecision::ExpressPhaseSteer
        };
        bank.record_step(StepTelemetryInput {
            decision: dec,
            horizon: &horizon,
            delta: &delta,
            overlap: &overlap,
            ego: &ego_off,
            entropy_gate_mode: 0,
            slid_in_bits: 0,
            slid_out_bits: 0,
            escalated: false,
            qos_throttled: false,
        });
        if i % 2 == 0 {
            bank.record_whole_token_bypass();
        }
    }

    let report = bank.compute_distortion_report();
    assert_eq!(report.kind, LayerDistortionKind::Aligned);
    assert!(!report.ego_wakeup_recommended);
    let adj = bank.evaluate_slow_loop_epoch(&StigmergicThresholds::default(), 16);
    assert!(adj.applied);
    assert_eq!(
        adj.suggested_derivative_alpha_q7,
        DEFAULT_DERIVATIVE_ALPHA_Q7
    );

    // Genuine passive stagnation (dominant PassiveSidecar) must still trigger PassiveSaturation.
    let mut stagnant_bank = CoreTelemetryBank::default();
    for _ in 0..80 {
        stagnant_bank.record_step(StepTelemetryInput {
            decision: StigmergicActionDecision::PassiveSidecar,
            horizon: &horizon,
            delta: &delta,
            overlap: &overlap,
            ego: &ego_off,
            entropy_gate_mode: 0,
            slid_in_bits: 0,
            slid_out_bits: 0,
            escalated: false,
            qos_throttled: false,
        });
    }
    let stagnant_report = stagnant_bank.compute_distortion_report();
    assert_eq!(stagnant_report.kind, LayerDistortionKind::PassiveSaturation);
    assert!(stagnant_report.ego_wakeup_recommended);
}

#[test]
fn test_pinned_sandbox_handle_preservation_and_repulsion_sandbox_isolation() {
    // Bug 1: PINNED | SANDBOX lane must retain its `active_sandbox_lane` handle after RepulsionEscape.
    let mut substrate = StigmergicSubstrate::new();
    let pinned_sb = substrate
        .allocate_lane(&WorkbenchSlotSpec {
            placement: WorkbenchLanePlacement::Explicit { lane: 5, pin: true },
            weight: 0.75,
            flags: WORKBENCH_LANE_FLAG_PINNED | WORKBENCH_LANE_FLAG_SANDBOX,
            phases: [24; MAX_ATTENTION_HEADS],
        })
        .unwrap();
    assert_eq!(pinned_sb, 5);
    let mut active_handle = Some(pinned_sb);
    let mut telemetry = CoreTelemetryBank::default();
    let cand_words = [0u64; NUM_STIGMERGIC_GRANULES];
    let ev = substrate.step_sandbox_hypothesis_lifecycle(
        &mut telemetry,
        &mut active_handle,
        SandboxLifecycleInput {
            decision: StigmergicActionDecision::RepulsionEscape,
            ego_active: true,
            exterior_granule_mask: 0x00,
            candidate_words: &cand_words,
            candidate_weight: 0.5,
            drift_dwell_threshold: 2,
            settle_dwell_threshold: 2,
        },
    );
    assert_eq!(ev, SandboxLifecycleEvent::Idle);
    assert_eq!(active_handle, Some(pinned_sb));
    assert_ne!(
        (substrate.slot_enabled_mask & substrate.slot_sandbox_mask) & (1u8 << pinned_sb),
        0
    );

    // Bug 2: REPULSION | SANDBOX lane must NOT leak global inhibition into `RuntimeContext` superposition.
    let mut ctx_base = RuntimeContext::new(Config::default()).unwrap();
    let mut ctx_sb_rep = RuntimeContext::new(Config::default()).unwrap();
    let std_spec = WorkbenchSlotSpec::explicit(0, false, 0, 1.0, [32; MAX_ATTENTION_HEADS]);
    let _ = ctx_base.substrate_mut().allocate_lane(&std_spec).unwrap();
    let _ = ctx_sb_rep.substrate_mut().allocate_lane(&std_spec).unwrap();
    let _ = ctx_sb_rep
        .substrate_mut()
        .allocate_lane(&WorkbenchSlotSpec {
            placement: WorkbenchLanePlacement::Explicit {
                lane: 6,
                pin: false,
            },
            weight: 1.0,
            flags: WORKBENCH_LANE_FLAG_REPULSION | WORKBENCH_LANE_FLAG_SANDBOX,
            phases: [32; MAX_ATTENTION_HEADS],
        })
        .unwrap();

    let trail = PointerTrail {
        accumulated_phase_delta: [32; MAX_ATTENTION_HEADS],
        ..Default::default()
    };
    let (out_base, pull_base) = ctx_base.evaluate_multi_domain_superposition_words(&trail);
    let (out_sb_rep, pull_sb_rep) = ctx_sb_rep.evaluate_multi_domain_superposition_words(&trail);
    assert_eq!(out_base, out_sb_rep);
    assert!((pull_base - pull_sb_rep).abs() < 1.0e-6);
}

#[test]
fn test_runtime_context_new_boxed_and_zeroize_in_place() {
    assert_eq!(core::mem::size_of::<RuntimeContext>(), 5184);
    assert_eq!(core::mem::align_of::<RuntimeContext>(), 64);
    assert!(!core::mem::needs_drop::<RuntimeContext>());

    let cfg = Config::new()
        .with_head_dim(96)
        .with_num_heads(24)
        .with_substrate_capacity(16);
    let mut boxed = RuntimeContext::new_boxed(cfg).expect("new_boxed");
    boxed.apply_host_spine_pin_floor(0xF0F0);
    boxed.substrate_mut().step_counter = 777;
    boxed.core_layers_mut().liminal.adaptation_epochs = 9;
    boxed.increment_sandbox_settlements();
    assert_eq!(boxed.sandbox_settlements(), 1);
    assert_eq!(
        boxed.envelope().pinned_compute_mask,
        0xF0F0 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
    );

    boxed.zeroize_in_place();
    assert_eq!(boxed.substrate().step_counter, 0);
    assert_eq!(boxed.core_layers().liminal.adaptation_epochs, 0);
    assert_eq!(boxed.sandbox_settlements(), 0);
    assert_eq!(boxed.envelope().pinned_compute_mask, 0);
    assert_eq!(boxed.escalated_workbench_lane(), None);
    assert_eq!(boxed.sandbox_workbench_lane(), None);
    assert_eq!(boxed.last_sandbox_event(), SandboxLifecycleEvent::Idle);
}

#[test]
fn test_guard_phase_anchor_solo_repulsion_and_i8_min_saturation() {
    let mut pole = [0i8; MAX_ATTENTION_HEADS];
    for (idx, slot) in pole.iter_mut().enumerate() {
        *slot = if idx % 2 == 0 { 42 } else { -36 };
    }
    let guard = GuardPhaseAnchor::new(pole, 0.85, 0.50);
    let (frame, words) = guard.synthesize_gateway_frame_with_words();
    assert!(frame.is_repulsed());
    assert_eq!(frame.dominant_slot, ATTRACTOR_SLOT_REPULSION);
    assert_eq!(frame.action_mode_flags, ACTION_PRESET_REPULSION_ESCAPE);
    assert_eq!(frame.root_cause_code, ROOT_CAUSE_COHOMOLOGY_REPULSION);
    assert!(frame.effective_energy >= 0.85);
    assert_eq!(words, load_q7_words_64(&frame.phase_deltas));
    for (idx, &delta) in frame.phase_deltas.iter().enumerate() {
        let expected = if idx % 2 == 0 { -42 } else { 36 };
        assert_eq!(delta, expected, "solo anti-phase mismatch at head {idx}");
    }

    // Saturating negation of `i8::MIN` (`-128`) must clamp to `+127` without wraparound.
    let min_guard = GuardPhaseAnchor::new([i8::MIN; MAX_ATTENTION_HEADS], 0.85, 0.50);
    let min_frame = min_guard.synthesize_gateway_frame();
    assert!(min_frame.is_repulsed());
    assert_eq!(min_frame.phase_deltas, [127i8; MAX_ATTENTION_HEADS]);

    // Below threshold, a solo guard pole remains quiescent with zero phase delta.
    let quiet_guard = GuardPhaseAnchor::new(pole, 0.20, 0.50);
    let quiet_frame = quiet_guard.synthesize_gateway_frame();
    assert!(!quiet_frame.is_repulsed());
    assert_eq!(quiet_frame.phase_deltas, [0i8; MAX_ATTENTION_HEADS]);
}

#[test]
fn test_simultaneous_attractive_and_repulsive_poles_preserve_orthogonal_attraction() {
    // 1. Disjoint-head attractive pole `A` (heads 0..32) + repulsive pole `R` (heads 32..64,
    //    including `i8::MIN` on head 63): `A` must remain positive and `R` must emit `-R`.
    let mut attract_arr = [0i8; MAX_ATTENTION_HEADS];
    for slot in &mut attract_arr[..32] {
        *slot = 50;
    }
    let mut repel_arr = [0i8; MAX_ATTENTION_HEADS];
    for slot in &mut repel_arr[32..63] {
        *slot = 40;
    }
    repel_arr[63] = i8::MIN;

    let sys = SystemConstraint::new((), 0.0, 0.0, 0xFF);
    let attract = ObjectiveFact::new(attract_arr, 2.0, 0, 101);
    let attract_pair = SubjectivePersona::new(attract_arr, 2.0, 0, 102);
    let repel = GuardPhaseAnchor::new(repel_arr, 0.85, 0.50);

    let frame = synthesize_multi_origin_gateway_frame(
        Some(&sys),
        Some(&attract),
        None::<&SubjectivePersona<[i8; MAX_ATTENTION_HEADS]>>,
        Some(&repel),
    );
    assert!(frame.is_repulsed());
    assert_eq!(frame.dominant_slot, ATTRACTOR_SLOT_REPULSION);
    for (idx, &delta) in frame.phase_deltas.iter().enumerate() {
        let expected = if idx < 32 {
            50
        } else if idx < 63 {
            -40
        } else {
            127
        };
        assert_eq!(delta, expected, "disjoint pole mismatch at head {idx}");
    }

    // Core-layer pair path (`apply_sys_and_guard_to_pair_frame`) must uphold the same geometry.
    let stack = CoreStateStack::new();
    let delta = LiminalBoundaryDelta::default();
    let horizon = EpistemicHorizonReading {
        total_memory_support: 1.5,
        peak_positive_affinity: 1.0,
        horizon_gate: 1.0,
        regime_code: EpistemicRegime::KnownCoherent as u8,
        ..EpistemicHorizonReading::default()
    };
    let overlap = OverlapDiscoveryReport {
        overlap_core_mask: 0xFF,
        ..OverlapDiscoveryReport::default()
    };
    let (pair_frame, _) = (sys, attract, attract_pair, repel)
        .synthesize_gateway_with_core_layers_readonly(&stack, &delta, &horizon, &overlap);
    assert!(pair_frame.is_repulsed());
    assert!(pair_frame.phase_deltas[..32].iter().all(|&d| d > 0));
    assert!(pair_frame.phase_deltas[32..63].iter().all(|&d| d < 0));
    assert_eq!(pair_frame.phase_deltas[63], 127);

    // 2. Non-disjoint orthogonal poles (`<A, R> == 0`): output preserves positive projection onto
    //    `A` while adding strictly repulsive projection onto `-R`.
    let orth_a = [40i8; MAX_ATTENTION_HEADS];
    let mut orth_r = [0i8; MAX_ATTENTION_HEADS];
    for (idx, slot) in orth_r.iter_mut().enumerate() {
        *slot = if idx % 2 == 0 { 40 } else { -40 };
    }
    let orth_frame = synthesize_multi_origin_gateway_frame(
        Some(&sys),
        Some(&ObjectiveFact::new(orth_a, 2.0, 0, 201)),
        None::<&SubjectivePersona<[i8; MAX_ATTENTION_HEADS]>>,
        Some(&GuardPhaseAnchor::new(orth_r, 0.85, 0.50)),
    );
    let dot_a: i32 = orth_frame
        .phase_deltas
        .iter()
        .zip(orth_a.iter())
        .map(|(&o, &a)| i32::from(o) * i32::from(a))
        .sum();
    let dot_r: i32 = orth_frame
        .phase_deltas
        .iter()
        .zip(orth_r.iter())
        .map(|(&o, &r)| i32::from(o) * i32::from(r))
        .sum();
    assert!(dot_a > 0, "attractive pole was inverted: dot_a={dot_a}");
    assert!(
        dot_r < 0,
        "repulsive pole failed to push away: dot_r={dot_r}"
    );

    // 3. `ResonantSignalSlice<GuardPhaseAnchor>` must project raw positive `+R` (no zero fallback
    //    when obstruction is 0, and no double negation when obstruction exceeds threshold).
    let guards_zero_obs = [GuardPhaseAnchor::new(repel_arr, 0.0, 0.72)];
    let guard_slice = ResonantSignalSlice::from_q7_probe(&guards_zero_obs, &repel_arr);
    let slice_frame = (sys, attract, attract_pair, guard_slice).synthesize_gateway_frame();
    assert!(slice_frame.is_repulsed());
    assert!(slice_frame.phase_deltas[..32].iter().all(|&d| d == 50));
    assert!(slice_frame.phase_deltas[32..63].iter().all(|&d| d == -40));
    assert_eq!(slice_frame.phase_deltas[63], 127);

    let guards_active_obs = [GuardPhaseAnchor::new(repel_arr, 0.90, 0.50)];
    let guard_slice_obs = ResonantSignalSlice::from_q7_probe(&guards_active_obs, &repel_arr);
    let slice_obs_frame = (sys, attract, attract_pair, guard_slice_obs).synthesize_gateway_frame();
    assert!(slice_obs_frame.is_repulsed());
    assert!(
        slice_obs_frame.phase_deltas[32..63]
            .iter()
            .all(|&d| d == -40)
    );
    assert_eq!(slice_obs_frame.phase_deltas[63], 127);

    // 4. Phase-less candidate detectors (`ContradictionGuard<()>`) keep candidate-phase negation.
    let scalar_guard = ContradictionGuard::new((), 0.85, 3);
    let detector_frame = synthesize_multi_origin_gateway_frame(
        Some(&sys),
        Some(&attract),
        None::<&SubjectivePersona<[i8; MAX_ATTENTION_HEADS]>>,
        Some(&scalar_guard),
    );
    assert!(detector_frame.is_repulsed());
    assert!(detector_frame.phase_deltas[..32].iter().all(|&d| d == -50));
    assert!(detector_frame.phase_deltas[32..].iter().all(|&d| d == 0));
}
