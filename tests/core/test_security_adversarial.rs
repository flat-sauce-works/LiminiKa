// name: tests/core/test_security_adversarial.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Permanent adversarial security & soundness integration suites for `liminika-core`.
//!
//! Covers:
//! 1. `.lmnkc` deterministic mutation fuzzing (`>= 2,000` mutations), alignment/topology guards,
//!    strict padding/gap/trailing/flag validation, and `Lacuna::Unobserved` reserved sections.
//! 2. Cycle & out-of-bounds termination defense for `PointerTrail` and `EgoDifferentialNet`.
//! 3. Hostile C-ABI calls (NULL, misalignment, `isize::MAX` overflow lengths, NaN/Inf, aliasing,
//!    cross-mode destroy/deinit, unaligned `lmnk_lora_project` parity) + previously untested exports.
//! 4. `SystemSpineState` privilege isolation & transactional `NaN`/`Inf`/corrupt snapshot rejection.
//! 5. UAF-free `zeroize_volatile` proof for `5,120`-byte `RuntimeContext` buffers.

use std::ptr;

use liminika_core::abi::{
    lmnk_config_init_default, lmnk_config_t, lmnk_container_map_and_steer_zero_copy_safe,
    lmnk_container_verify_with_reports_zero_copy, lmnk_context_align,
    lmnk_context_begin_turn_spine_policy, lmnk_context_configure_spine, lmnk_context_create,
    lmnk_context_deinit_in_place, lmnk_context_deserialize, lmnk_context_destroy,
    lmnk_context_get_spine_state, lmnk_context_init_in_place, lmnk_context_serialize,
    lmnk_context_size, lmnk_context_step_decode_control, lmnk_context_step_token,
    lmnk_distortion_delta_apply_chain, lmnk_distortion_tension_step_hot_path,
    lmnk_entropy_controller_create, lmnk_entropy_controller_destroy, lmnk_entropy_eval_stateful,
    lmnk_entropy_state_t, lmnk_gateway_project_bytes_to_q7_64, lmnk_gateway_project_f32_to_q7_64,
    lmnk_gateway_synthesize_multi_origin_frame, lmnk_ledger_evaluate_auto_tune,
    lmnk_ledger_fast_path_lookup, lmnk_ledger_reset_telemetry, lmnk_ledger_set_mode,
    lmnk_ledger_slot_t, lmnk_ledger_telemetry_push, lmnk_liminal_eval_barrier,
    lmnk_liminal_eval_barrier_pitchfork, lmnk_liminal_eval_dyadic128,
    lmnk_liminal_evaluate_lacuna_partition, lmnk_lora_project,
    lmnk_phase_steering_apply_fused_safe, lmnk_phase_steering_apply_safe,
    lmnk_phase_steering_compute_procrustes_half_split, lmnk_phase_steering_create,
    lmnk_phase_steering_destroy, lmnk_phase_steering_fused_logit_shift,
    lmnk_phase_steering_steer_gateway_frame_safe, lmnk_pointer_trail_t, lmnk_q7_t,
    lmnk_rank1_descriptor_t, lmnk_rank1_eval, lmnk_rank1_eval_fused, lmnk_spine_decode_input_t,
    lmnk_spine_state_t, lmnk_spine_step_plan_t, lmnk_subhead_dispatch_single_pass,
    lmnk_substrate_create, lmnk_substrate_destroy, lmnk_substrate_evaluate_thresholds,
    lmnk_substrate_hash_slot256_index, lmnk_substrate_init_default_thresholds,
    lmnk_substrate_link_hallucinated_trails, lmnk_substrate_paged_block_warp_bitmask,
    lmnk_substrate_pull_trail_to_anchor, lmnk_substrate_reduce_bit_tree,
    lmnk_substrate_resolve_trail_in_swarm, lmnk_substrate_step_pointer, lmnk_substrate_step_safe,
    lmnk_tagged_ptr_evolve_mode, lmnk_tagged_ptr_pack, lmnk_tagged_ptr_pack_action,
    lmnk_tagged_ptr_unpack_action_mode, lmnk_tagged_ptr_unpack_addr,
    lmnk_tagged_ptr_unpack_attractor_slot, lmnk_tagged_ptr_unpack_phase,
    lmnk_tagged_ptr_unpack_tag, LmnkContextHandle, LmnkEntropyControllerHandle,
    LmnkEntropyControllerOpaque, LmnkPhaseSteeringHandle, LmnkPhaseSteeringOpaque,
    LmnkSubstrateHandle, SPINE_DECODE_FLAG_REWARD_OVERRIDE,
};
use liminika_core::drivers::{
    DistortionDelta, DistortionTension, EgoDifferentialNet, LiminalEgoPreset, LiminalEgoState,
    SystemSpineState, DISTORTION_DELTA_FLAG_ACTIVE, DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID,
    HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
};
use liminika_core::foundation::{
    compute_crc32_ieee, eval_rank1, eval_rank1_fused, AlignedBuffer16, AlignedBuffer32,
    AlignedBuffer64, PagedBitmask, Pod, SubheadConfig, ACTION_MODE_PHASE_STEER,
};
use liminika_core::liminal::{
    evaluate_lacuna_partition_branchless, LacunaPartition8, LMNK_ERR_ALIGNMENT,
    LMNK_ERR_BUFFER_TOO_SMALL, LMNK_ERR_CONTAINER_CORRUPTED, LMNK_ERR_INVALID_ARGUMENT,
    LMNK_ERR_INVALID_STATE, LMNK_ERR_NULL_POINTER, LMNK_ERR_TOPOLOGY_MISMATCH, LMNK_OK,
    MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
};
use liminika_core::runtime::{
    Config, GatewayFrame, LmnkContainerView, LmnkSectionInput, RuntimeContext, CORE_TOPOLOGY,
    LMNK_CONTAINER_FLAG_COMPILED_CACHE, LMNK_CONTAINER_HEADER_BYTES, LMNK_MAX_CONTAINER_SECTIONS,
    LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES, LMNK_SECTION_BLOOM_FILTER, LMNK_SECTION_CONTENT_HASH,
    LMNK_SECTION_ENTRY_BYTES, LMNK_SECTION_PHASE_ANCHORS, LMNK_SECTION_POINTER_TRAILS,
    LMNK_SECTION_SIGNATURE,
};
use liminika_core::stigmergy::{
    ContrastivePhaseBloomFilter, LedgerSlot, PhaseAnchorRecord, PointerTrail,
    StigmergicGranuleState, StigmergicReconstitutionReport, StigmergicSubstrate,
    StigmergicThresholds, SubstrateStepReport, ATTRACTOR_SLOT_KNOWLEDGE,
};

#[derive(Clone, Copy)]
struct XorShift64(u64);

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() as usize) % bound
        }
    }
}

fn restamp_header_crc32(buf: &mut [u8]) {
    let crc = compute_crc32_ieee(&buf[0..60]);
    buf[60..64].copy_from_slice(&crc.to_le_bytes());
}

fn build_valid_multi_section_lmnkc() -> (Vec<u8>, [u8; 32], [u8; 64]) {
    let mut anchor = PhaseAnchorRecord::new_empty(101, ATTRACTOR_SLOT_KNOWLEDGE, 0);
    let mut frame = GatewayFrame {
        target_anchor_id: 101,
        effective_energy: 1.5,
        dominant_slot: ATTRACTOR_SLOT_KNOWLEDGE,
        phase_deltas: [12i8; MAX_ATTENTION_HEADS],
        ..GatewayFrame::default()
    };
    frame.target_anchor_id = 101;
    let _ = anchor.absorb_gateway_frame(&frame);
    let mut anchor_bytes = [0u8; std::mem::size_of::<PhaseAnchorRecord>()];
    PhaseAnchorRecord::write_records_to_container_bytes(&[anchor], &mut anchor_bytes)
        .expect("anchor serialize");

    let trail = PointerTrail::default();
    let mut trail_bytes = [0u8; std::mem::size_of::<PointerTrail>()];
    trail
        .write_unaligned_to(&mut trail_bytes, 0)
        .expect("trail serialize");

    let bloom = ContrastivePhaseBloomFilter::default();
    let mut bloom_bytes = [0u8; std::mem::size_of::<ContrastivePhaseBloomFilter>()];
    bloom
        .write_unaligned_to(&mut bloom_bytes, 0)
        .expect("bloom serialize");

    let mut content_hash = [0u8; 32];
    for (i, b) in content_hash.iter_mut().enumerate() {
        *b = 0xA0_u8.wrapping_add(i as u8);
    }
    let mut signature = [0u8; 64];
    for (i, b) in signature.iter_mut().enumerate() {
        *b = 0x50_u8.wrapping_add(i as u8);
    }

    let sections = [
        LmnkSectionInput {
            kind: LMNK_SECTION_PHASE_ANCHORS,
            flags: 0,
            payload: &anchor_bytes,
        },
        LmnkSectionInput {
            kind: LMNK_SECTION_POINTER_TRAILS,
            flags: 0,
            payload: &trail_bytes,
        },
        LmnkSectionInput {
            kind: LMNK_SECTION_BLOOM_FILTER,
            flags: 0,
            payload: &bloom_bytes,
        },
        LmnkSectionInput {
            kind: LMNK_SECTION_CONTENT_HASH,
            flags: 0,
            payload: &content_hash,
        },
        LmnkSectionInput {
            kind: LMNK_SECTION_SIGNATURE,
            flags: 0,
            payload: &signature,
        },
    ];

    let req = liminika_core::runtime::required_container_byte_len(&sections).expect("req len");
    let mut buf = vec![0u8; req];
    let written = liminika_core::runtime::write_container_sections_to_bytes(
        LMNK_CONTAINER_FLAG_COMPILED_CACHE,
        1_774_900_000,
        &sections,
        &mut buf,
    )
    .expect("write container");
    assert_eq!(written, req);
    (buf, content_hash, signature)
}

#[test]
fn suite1_lmnkc_container_adversarial_and_2500_mutations() {
    let (valid_buf, expected_hash, expected_sig) = build_valid_multi_section_lmnkc();

    // 1. Valid multi-section parse preserves ContentHash (0xFF01) & Signature (0xFF02) as raw Unobserved bytes.
    let view = LmnkContainerView::parse(&valid_buf).expect("valid parse");
    assert_eq!(view.phase_anchor_count(), 1);
    assert_eq!(view.pointer_trail_count(), 1);
    assert_eq!(
        view.section_bytes(LMNK_SECTION_CONTENT_HASH),
        Some(expected_hash.as_slice())
    );
    assert_eq!(
        view.section_bytes(LMNK_SECTION_SIGNATURE),
        Some(expected_sig.as_slice())
    );

    // 2. 1-byte misaligned buffer: safe unaligned parse succeeds; zero-copy aligned FFI fails with LMNK_ERR_ALIGNMENT.
    let mut aligned_backing = AlignedBuffer64::<u8, 4096>::new([0u8; 4096]);
    assert!(valid_buf.len() + 64 <= aligned_backing.as_slice().len());
    aligned_backing.as_mut_slice()[1..1 + valid_buf.len()].copy_from_slice(&valid_buf);
    let misaligned = &aligned_backing.as_slice()[1..1 + valid_buf.len()];
    assert!(LmnkContainerView::parse(misaligned).is_ok());

    let mut num_anchors = 0usize;
    let mut num_trails = 0usize;
    let mut num_reports = 0usize;
    // SAFETY: Pointers refer to aligned valid local stack variables; misaligned buffer must fail alignment check cleanly.
    let status_mis = unsafe {
        lmnk_container_verify_with_reports_zero_copy(
            misaligned.as_ptr(),
            misaligned.len(),
            &mut num_anchors,
            &mut num_trails,
            &mut num_reports,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(status_mis, LMNK_ERR_ALIGNMENT);

    let mut trail_io = lmnk_pointer_trail_t::default();
    let mut q_buf = AlignedBuffer32::<f32, 128>::new([1.0; 128]);
    let mut k_buf = AlignedBuffer32::<f32, 128>::new([1.0; 128]);
    // SAFETY: Buffer alignment check triggers LMNK_ERR_ALIGNMENT safely before zero-copy access.
    let steer_mis = unsafe {
        lmnk_container_map_and_steer_zero_copy_safe(
            misaligned.as_ptr(),
            misaligned.len(),
            &mut trail_io,
            0.05,
            q_buf.as_mut_slice().as_mut_ptr(),
            k_buf.as_mut_slice().as_mut_ptr(),
            64,
            2,
            &mut num_anchors,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(steer_mis, LMNK_ERR_ALIGNMENT);

    // 3. Mutated topology_crc32 -> LMNK_ERR_TOPOLOGY_MISMATCH.
    let mut topo_bad = valid_buf.clone();
    topo_bad[8..12].copy_from_slice(&(CORE_TOPOLOGY.topology_crc32 ^ 0xDEAD_BEEF).to_le_bytes());
    restamp_header_crc32(&mut topo_bad);
    assert_eq!(
        LmnkContainerView::parse(&topo_bad),
        Err(LMNK_ERR_TOPOLOGY_MISMATCH)
    );

    // 4. Explicit structural rejections with valid header CRC32 where applicable.
    // (a) Non-zero byte in 64B section-table alignment padding (5 entries => 64 + 80 = 144..192 is zero padding).
    let mut table_pad_bad = valid_buf.clone();
    table_pad_bad[150] = 0xAA;
    assert_eq!(
        LmnkContainerView::parse(&table_pad_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (b) Non-zero byte in 64B section trailing alignment padding (entry index 3 is ContentHash of length 32).
    let entry3_off = LMNK_CONTAINER_HEADER_BYTES + 3 * LMNK_SECTION_ENTRY_BYTES;
    let c_hash_off = u32::from_le_bytes(
        valid_buf[entry3_off + 4..entry3_off + 8]
            .try_into()
            .unwrap(),
    ) as usize;
    let mut sec_pad_bad = valid_buf.clone();
    sec_pad_bad[c_hash_off + 40] = 0x55;
    assert_eq!(
        LmnkContainerView::parse(&sec_pad_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (c) Inter-section gap (shift entry 4 offset by +64 with extra container length).
    let entry4_off = LMNK_CONTAINER_HEADER_BYTES + 4 * LMNK_SECTION_ENTRY_BYTES;
    let sig_off = u32::from_le_bytes(
        valid_buf[entry4_off + 4..entry4_off + 8]
            .try_into()
            .unwrap(),
    ) as usize;
    let mut gap_bad = Vec::with_capacity(valid_buf.len() + 64);
    gap_bad.extend_from_slice(&valid_buf[..sig_off]);
    gap_bad.extend_from_slice(&[0u8; 64]);
    gap_bad.extend_from_slice(&valid_buf[sig_off..]);
    gap_bad[entry4_off + 4..entry4_off + 8].copy_from_slice(&((sig_off + 64) as u32).to_le_bytes());
    let new_total = gap_bad.len() as u64;
    gap_bad[16..24].copy_from_slice(&new_total.to_le_bytes());
    restamp_header_crc32(&mut gap_bad);
    assert_eq!(
        LmnkContainerView::parse(&gap_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (d) Trailing unhashed bytes after final aligned section end.
    let mut trailing_bad = valid_buf.clone();
    trailing_bad.extend_from_slice(&[0u8; 64]);
    let trail_total = trailing_bad.len() as u64;
    trailing_bad[16..24].copy_from_slice(&trail_total.to_le_bytes());
    restamp_header_crc32(&mut trailing_bad);
    assert_eq!(
        LmnkContainerView::parse(&trailing_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (e) Unknown container_flags bit.
    let mut flags_bad = valid_buf.clone();
    let unknown_flag = LMNK_CONTAINER_FLAG_COMPILED_CACHE | (1u32 << 30);
    flags_bad[12..16].copy_from_slice(&unknown_flag.to_le_bytes());
    restamp_header_crc32(&mut flags_bad);
    assert_eq!(
        LmnkContainerView::parse(&flags_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (f) Non-zero reserved_future byte.
    let mut res_bad = valid_buf.clone();
    res_bad[40] = 1;
    restamp_header_crc32(&mut res_bad);
    assert_eq!(
        LmnkContainerView::parse(&res_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (g) Non-zero entry.flags.
    let mut entry_flag_bad = valid_buf.clone();
    entry_flag_bad[entry3_off + 2..entry3_off + 4].copy_from_slice(&1u16.to_le_bytes());
    assert_eq!(
        LmnkContainerView::parse(&entry_flag_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (h) section_count > 32.
    let mut count_bad = valid_buf.clone();
    count_bad[36..38].copy_from_slice(&(LMNK_MAX_CONTAINER_SECTIONS + 1).to_le_bytes());
    restamp_header_crc32(&mut count_bad);
    assert_eq!(
        LmnkContainerView::parse(&count_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (i) u32::MAX length overflow.
    let mut overflow_bad = valid_buf.clone();
    overflow_bad[entry3_off + 8..entry3_off + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        LmnkContainerView::parse(&overflow_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (j) Duplicate section.kind.
    let mut dup_bad = valid_buf.clone();
    dup_bad[entry4_off..entry4_off + 2].copy_from_slice(&LMNK_SECTION_CONTENT_HASH.to_le_bytes());
    assert_eq!(
        LmnkContainerView::parse(&dup_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // (k) entry.length == 0.
    let mut zero_len_bad = valid_buf.clone();
    zero_len_bad[entry3_off + 8..entry3_off + 12].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        LmnkContainerView::parse(&zero_len_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    // 5. >= 2,000 deterministic xorshift64 mutations (2,500 trials): zero panics, all corrupted fail.
    let mut rng = XorShift64(0x5EED_4B01_CAFE_900D);
    aligned_backing.as_mut_slice().fill(0);
    let table_start = LMNK_CONTAINER_HEADER_BYTES;
    let table_end = table_start + 5 * LMNK_SECTION_ENTRY_BYTES;
    let mut kind_only_unobserved_ok = 0usize;
    for trial in 0..2500usize {
        let mut mutated = valid_buf.clone();
        let mut flipped_only_custom_kind = false;
        match trial % 5 {
            0 => {
                // Single-byte guaranteed non-zero bit flip outside unknown custom `entry.kind` bytes.
                let mut idx = rng.next_usize(mutated.len());
                if idx >= table_start
                    && idx < table_end
                    && ((idx - table_start) % LMNK_SECTION_ENTRY_BYTES) < 2
                {
                    // Advance +2 to hit entry.flags / offset / length / crc32 instead of unhashed custom kind tag.
                    idx += 2;
                }
                let bit = 1u8 << (rng.next_usize(8) as u8);
                mutated[idx] ^= bit;
            }
            1 => {
                // Multi-byte burst mutation across integrity-checked fields/payloads.
                for _ in 0..4 {
                    let mut idx = rng.next_usize(mutated.len());
                    if idx >= table_start
                        && idx < table_end
                        && ((idx - table_start) % LMNK_SECTION_ENTRY_BYTES) < 2
                    {
                        idx += 2;
                    }
                    mutated[idx] ^= (rng.next_u64() as u8) | 1;
                }
            }
            2 => {
                // Prefix truncation (always shorter than valid_buf).
                let new_len = rng.next_usize(mutated.len());
                mutated.truncate(new_len);
            }
            3 => {
                // Mutate section entry lengths/offsets directly.
                let slot = rng.next_usize(5);
                let off = LMNK_CONTAINER_HEADER_BYTES + slot * LMNK_SECTION_ENTRY_BYTES + 4;
                let word = u32::from_le_bytes(mutated[off..off + 4].try_into().unwrap())
                    ^ ((rng.next_u64() as u32) | 1);
                mutated[off..off + 4].copy_from_slice(&word.to_le_bytes());
            }
            _ => {
                // Wholly random garbage buffer up to 512 bytes, or extension-kind probe on reserved slot.
                if trial % 25 == 24 {
                    let slot = 3 + rng.next_usize(2);
                    let off = table_start + slot * LMNK_SECTION_ENTRY_BYTES;
                    mutated[off] ^= 0x10;
                    flipped_only_custom_kind = true;
                } else {
                    let rlen = rng.next_usize(512);
                    mutated = vec![0u8; rlen];
                    for b in &mut mutated {
                        *b = rng.next_u64() as u8;
                    }
                    if mutated.len() >= 4 && &mutated[0..4] == b"LMNK" {
                        mutated[0] ^= 0xFF;
                    }
                }
            }
        }

        let res = LmnkContainerView::parse(&mutated);
        if flipped_only_custom_kind {
            if res.is_ok() {
                kind_only_unobserved_ok += 1;
            }
        } else {
            assert!(
                res.is_err(),
                "mutated container trial {trial} unexpectedly parsed as Ok"
            );

            let copy_len = mutated.len().min(aligned_backing.as_slice().len());
            aligned_backing.as_mut_slice()[..copy_len].copy_from_slice(&mutated[..copy_len]);
            // SAFETY: Aligned stack buffer of length `copy_len` verified strictly non-panicking and failing cleanly.
            let zc_status = unsafe {
                lmnk_container_verify_with_reports_zero_copy(
                    aligned_backing.as_slice().as_ptr(),
                    copy_len,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            };
            assert_ne!(zc_status, LMNK_OK);
        }
    }
    assert!(kind_only_unobserved_ok > 0);
}

#[test]
fn suite2_pointer_trail_and_ego_differential_cycle_defense() {
    // 1. PointerTrail cycle + self-loop + OOB link defense on StigmergicSubstrate.
    let mut sub = StigmergicSubstrate::new();
    sub.active_count = 4;
    for (i, trail) in sub.trails.iter_mut().enumerate().take(4) {
        trail.current_ptr = ((i as u64) + 1) << 12;
        trail.target_anchor_id = (i as u32) + 10;
        trail.stigmergic_density = 0.5;
        trail.accumulated_phase_delta = [8i8; MAX_ATTENTION_HEADS];
    }
    // Self-loop on node 0.
    sub.trails[0].linked_trail_id = 0;
    let mut trail_out = PointerTrail::default();
    let (_rep, hops_self, term_idx, _, _) = sub
        .traverse_pointer_chain_and_orchestrate(0, 8, 0.3, &mut trail_out, None, None, None, None)
        .expect("self loop bounded");
    assert_eq!(hops_self, 1);
    assert_eq!(term_idx, 0);

    // 2-node cycle (0 <-> 1) with OOB terminal edge on node 2 (u16::MAX).
    sub.trails[0].linked_trail_id = 1;
    sub.trails[1].linked_trail_id = 0;
    let (_rep2, hops_cycle, _, _, _) = sub
        .traverse_pointer_chain_and_orchestrate(0, 8, 0.3, &mut trail_out, None, None, None, None)
        .expect("2-node cycle bounded");
    assert_eq!(hops_cycle, 2);

    sub.trails[2].linked_trail_id = u32::MAX;
    let (_rep_oob, hops_oob, _, _, _) = sub
        .traverse_pointer_chain_and_orchestrate(2, 8, 0.3, &mut trail_out, None, None, None, None)
        .expect("oob hop terminates");
    assert_eq!(hops_oob, 1);

    // 2. EgoDifferentialNet cycle + OOB defense across hops = 1..=4 without duplicate contribution.
    let ego = LiminalEgoState::from_preset(LiminalEgoPreset::High);
    let partition = evaluate_lacuna_partition_branchless(
        0xFF,
        &[64; NUM_STIGMERGIC_GRANULES],
        &[8; NUM_STIGMERGIC_GRANULES],
        32,
    );
    let stig = StigmergicGranuleState::default();
    let base_words = [0u64; NUM_STIGMERGIC_GRANULES];

    for hops in 1..=4u8 {
        // Acyclic 2-node chain (0 -> 1 -> NO_NEXT_HOP) vs 2-node cycle (0 <-> 1) vs self-loop (0 -> 0).
        let make_node = |id: u32, next_hop: u16| {
            let mut n = DistortionDelta::new_empty(id, id + 1);
            n.next_hop_idx = next_hop;
            n.pheromone_q7 = 96;
            n.gain_q7 = 64;
            n.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
            n.delta_phase_q7 = [6i8; MAX_ATTENTION_HEADS];
            n
        };

        let mut net_acyclic = EgoDifferentialNet::new_empty(hops);
        net_acyclic.set_node(0, make_node(1, 1));
        net_acyclic.set_node(1, make_node(2, u16::MAX));

        let mut net_cycle = EgoDifferentialNet::new_empty(hops);
        net_cycle.set_node(0, make_node(1, 1));
        net_cycle.set_node(1, make_node(2, 0));

        let (out_a, norm_a, trav_a, entry_a, _) =
            net_acyclic.evaluate_pframe_words_branchless(&ego, &partition, &stig, 64, &base_words);
        let (out_c, norm_c, trav_c, entry_c, _) =
            net_cycle.evaluate_pframe_words_branchless(&ego, &partition, &stig, 64, &base_words);
        assert_eq!(
            out_a, out_c,
            "cycle duplicated node contribution at hops={hops}"
        );
        assert_eq!(norm_a, norm_c);
        assert_eq!(trav_a, trav_c);
        assert_eq!(entry_a, entry_c);

        // record_chain_activation must reinforce each visited node at most once.
        net_acyclic.record_chain_activation(usize::from(entry_a), 4);
        net_cycle.record_chain_activation(usize::from(entry_c), 4);
        assert_eq!(net_acyclic.node_cold[0].reinforce_count, 1);
        assert_eq!(net_cycle.node_cold[0].reinforce_count, 1);
        if hops >= 2 {
            assert_eq!(net_acyclic.node_cold[1].reinforce_count, 1);
            assert_eq!(net_cycle.node_cold[1].reinforce_count, 1);
        }

        // Self-loop + OOB target ID (>= 8) must terminate after 1 hop.
        let mut net_self = EgoDifferentialNet::new_empty(hops);
        net_self.set_node(0, make_node(1, 0));
        let (_out_s, _, trav_s, _, _) =
            net_self.evaluate_pframe_words_branchless(&ego, &partition, &stig, 64, &base_words);
        assert_eq!(trav_s, 1);
        net_self.record_chain_activation(0, 4);
        assert_eq!(net_self.node_cold[0].reinforce_count, 1);

        let mut net_oob = EgoDifferentialNet::new_empty(hops);
        net_oob.set_node(0, make_node(1, 42));
        let (_out_o, _, trav_o, _, _) =
            net_oob.evaluate_pframe_words_branchless(&ego, &partition, &stig, 64, &base_words);
        assert_eq!(trav_o, 1);
    }
}

#[test]
fn suite3_hostile_c_abi_and_extended_exports() {
    // 1. Context lifecycle: cross-mode misuse, double deinit, aliasing, isize::MAX lengths, NaN.
    let mut cfg = lmnk_config_t::default();
    // SAFETY: Stack config is 64B-aligned and writable.
    assert_eq!(unsafe { lmnk_config_init_default(&mut cfg) }, LMNK_OK);

    let req_sz = lmnk_context_size();
    assert_eq!(req_sz, 5120);
    assert_eq!(lmnk_context_align(), 64);
    let mut backing = AlignedBuffer64::<u8, 5120>::new([0u8; 5120]);
    // SAFETY: Aligned 5120B caller buffer.
    unsafe {
        assert_eq!(
            lmnk_context_init_in_place(&cfg, backing.as_mut_slice().as_mut_ptr(), req_sz),
            LMNK_OK
        );
    }
    let in_place_handle = backing
        .as_mut_slice()
        .as_mut_ptr()
        .cast::<liminika_core::abi::LmnkContextOpaque>();

    // Cross-mode destroy on in-place context must return LMNK_ERR_INVALID_STATE.
    // SAFETY: Handle points to initialized in-place context; destroy validates mode cleanly.
    let destroy_status = unsafe { lmnk_context_destroy(in_place_handle) };
    assert_eq!(destroy_status, LMNK_ERR_INVALID_STATE);

    // Aliasing q_ptr == k_ptr on step_token must be rejected BEFORE creating &mut slices.
    let mut q_shared = AlignedBuffer32::<f32, 128>::new([0.5; 128]);
    let mut trail_out = lmnk_pointer_trail_t::default();
    // SAFETY: Testing aliasing detection cleanly returns error status without UB.
    unsafe {
        assert_eq!(
            lmnk_context_step_token(
                in_place_handle,
                1,
                q_shared.as_mut_slice().as_mut_ptr(),
                q_shared.as_mut_slice().as_mut_ptr(),
                &mut trail_out,
            ),
            LMNK_ERR_INVALID_ARGUMENT
        );
    }

    // serialize with undersized length returns LMNK_ERR_BUFFER_TOO_SMALL and populates required size.
    let mut small_buf = [0u8; 64];
    let mut sz = small_buf.len();
    // SAFETY: Undersized buffer returns LMNK_ERR_BUFFER_TOO_SMALL safely.
    let ser_small =
        unsafe { lmnk_context_serialize(in_place_handle, small_buf.as_mut_ptr(), &mut sz) };
    assert_eq!(ser_small, LMNK_ERR_BUFFER_TOO_SMALL);
    assert_eq!(sz, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES);

    // Undersized deserialize rejects without panicking.
    let mut out_ctx: LmnkContextHandle = ptr::null_mut();
    // SAFETY: Rejects undersized snapshot cleanly.
    let deser_small = unsafe { lmnk_context_deserialize(small_buf.as_ptr(), 16, &mut out_ctx) };
    assert_eq!(deser_small, LMNK_ERR_BUFFER_TOO_SMALL);
    assert!(out_ctx.is_null());

    // Zero / usize::MAX lengths on serialize & deserialize rejected cleanly.
    let mut full_buf = [0u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut zero_len = 0usize;
    // SAFETY: Zero length triggers LMNK_ERR_BUFFER_TOO_SMALL without writing to `full_buf`.
    let ser_zero =
        unsafe { lmnk_context_serialize(in_place_handle, full_buf.as_mut_ptr(), &mut zero_len) };
    assert_eq!(ser_zero, LMNK_ERR_BUFFER_TOO_SMALL);
    let mut huge_len = usize::MAX;
    // SAFETY: Overflowing usize::MAX byte length fails bounds check before creating slices.
    let ser_huge =
        unsafe { lmnk_context_serialize(in_place_handle, full_buf.as_mut_ptr(), &mut huge_len) };
    assert_eq!(ser_huge, LMNK_ERR_INVALID_ARGUMENT);
    // SAFETY: Overflowing usize::MAX byte length fails bounds check in deserialize.
    let deser_huge =
        unsafe { lmnk_context_deserialize(full_buf.as_ptr(), usize::MAX, &mut out_ctx) };
    assert_eq!(deser_huge, LMNK_ERR_INVALID_ARGUMENT);

    // Clean deinit + double deinit (second call must return LMNK_ERR_INVALID_STATE, never panic/UB).
    // SAFETY: Caller-owned in-place handle is valid on first call and zeroed afterward.
    unsafe {
        assert_eq!(lmnk_context_deinit_in_place(in_place_handle), LMNK_OK);
        assert_eq!(
            lmnk_context_deinit_in_place(in_place_handle),
            LMNK_ERR_INVALID_STATE
        );
    }

    // Heap context + cross-mode deinit_in_place rejected with LMNK_ERR_INVALID_STATE.
    let mut heap_handle: LmnkContextHandle = ptr::null_mut();
    // SAFETY: Creates and cleans up heap context via matching API.
    unsafe {
        assert_eq!(lmnk_context_create(&cfg, &mut heap_handle), LMNK_OK);
        assert_eq!(
            lmnk_context_deinit_in_place(heap_handle),
            LMNK_ERR_INVALID_STATE
        );
        assert_eq!(lmnk_context_destroy(heap_handle), LMNK_OK);
    }

    // 2. Hostile lengths/NULL/misalignment/NaN/aliasing across phase steering & gateway exports.
    let mut mis_raw = [0u8; 256];
    let mis_f32_ptr = mis_raw[1..].as_mut_ptr().cast::<f32>();
    let mut good_q = AlignedBuffer32::<f32, 128>::new([1.0; 128]);
    let mut good_k = AlignedBuffer32::<f32, 128>::new([0.5; 128]);
    let phases_16 = AlignedBuffer16::<i8, 64>::new([12; 64]);
    let mut out_q7 = AlignedBuffer16::<lmnk_q7_t, 64>::new([0; 64]);

    // SAFETY: Intentionally hostile NULL, misaligned, usize::MAX, NaN, and aliasing checks.
    unsafe {
        assert_eq!(
            lmnk_phase_steering_apply_safe(
                ptr::null_mut(),
                phases_16.as_slice().as_ptr(),
                64,
                2,
                0.2
            ),
            LMNK_ERR_NULL_POINTER
        );
        assert_eq!(
            lmnk_phase_steering_apply_safe(mis_f32_ptr, phases_16.as_slice().as_ptr(), 64, 2, 0.2),
            LMNK_ERR_ALIGNMENT
        );
        assert_ne!(
            lmnk_phase_steering_apply_safe(
                good_q.as_mut_slice().as_mut_ptr(),
                phases_16.as_slice().as_ptr(),
                64,
                usize::MAX,
                0.2
            ),
            LMNK_OK
        );
        assert_ne!(
            lmnk_phase_steering_apply_safe(
                good_q.as_mut_slice().as_mut_ptr(),
                phases_16.as_slice().as_ptr(),
                64,
                2,
                f32::NAN
            ),
            LMNK_OK
        );
        assert_ne!(
            lmnk_phase_steering_apply_safe(
                good_q.as_mut_slice().as_mut_ptr(),
                phases_16.as_slice().as_ptr(),
                64,
                2,
                f32::INFINITY
            ),
            LMNK_OK
        );
        assert_eq!(
            lmnk_phase_steering_apply_fused_safe(
                good_q.as_mut_slice().as_mut_ptr(),
                good_q.as_mut_slice().as_mut_ptr(),
                phases_16.as_slice().as_ptr(),
                64,
                2,
                0.2
            ),
            LMNK_ERR_INVALID_ARGUMENT
        );
        // Procrustes read/write aliasing rejected; read/read aliasing allowed.
        assert_eq!(
            lmnk_phase_steering_compute_procrustes_half_split(
                good_q.as_slice().as_ptr(),
                good_q.as_slice().as_ptr(),
                64,
                64,
                2,
                out_q7.as_mut_slice().as_mut_ptr()
            ),
            LMNK_OK
        );
        assert_eq!(
            lmnk_phase_steering_compute_procrustes_half_split(
                good_q.as_slice().as_ptr(),
                good_k.as_slice().as_ptr(),
                64,
                64,
                2,
                good_k.as_mut_slice().as_mut_ptr().cast::<lmnk_q7_t>()
            ),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_ne!(
            lmnk_gateway_project_f32_to_q7_64(
                good_q.as_slice().as_ptr(),
                usize::MAX,
                1.0,
                out_q7.as_mut_slice().as_mut_ptr()
            ),
            LMNK_OK
        );
        assert_ne!(
            lmnk_gateway_project_bytes_to_q7_64(
                good_q.as_slice().as_ptr().cast::<u8>(),
                usize::MAX,
                out_q7.as_mut_slice().as_mut_ptr()
            ),
            LMNK_OK
        );
    }

    // 3. lmnk_lora_project unaligned multi-head parity (dim_out = 512 => num_heads = 4).
    const RANK: usize = 4;
    const DIM_IN: usize = 16;
    const DIM_OUT: usize = 512;
    let mut lora_a = AlignedBuffer32::<f32, { RANK * DIM_IN }>::new([0.0; RANK * DIM_IN]);
    let mut lora_b = AlignedBuffer32::<f32, { DIM_OUT * RANK }>::new([0.0; DIM_OUT * RANK]);
    for (i, v) in lora_a.as_mut_slice().iter_mut().enumerate() {
        *v = ((i % 7) as f32) * 0.25 - 0.5;
    }
    for (i, v) in lora_b.as_mut_slice().iter_mut().enumerate() {
        *v = ((i % 11) as f32) * 0.15 - 0.6;
    }
    let mut r1_aligned = lmnk_rank1_descriptor_t::default();
    let mut p_aligned = AlignedBuffer16::<lmnk_q7_t, 8>::new([0x5A; 8]);
    let mut r1_unaligned = lmnk_rank1_descriptor_t::default();
    let mut p_unaligned_raw = [0x7Fi8; 16];
    // SAFETY: Offset by 1 element stays in bounds of 16-byte stack array to force unaligned fallback.
    let p_unaligned_ptr = unsafe { p_unaligned_raw.as_mut_ptr().add(1) };

    // SAFETY: Aligned fast path and unaligned fallback path must write all 4 head outputs identically.
    unsafe {
        assert_eq!(
            lmnk_lora_project(
                lora_a.as_slice().as_ptr(),
                lora_b.as_slice().as_ptr(),
                RANK,
                DIM_IN,
                DIM_OUT,
                &mut r1_aligned,
                p_aligned.as_mut_slice().as_mut_ptr(),
            ),
            LMNK_OK
        );
        assert_eq!(
            lmnk_lora_project(
                lora_a.as_slice().as_ptr(),
                lora_b.as_slice().as_ptr(),
                RANK,
                DIM_IN,
                DIM_OUT,
                &mut r1_unaligned,
                p_unaligned_ptr,
            ),
            LMNK_OK
        );
    }
    assert_eq!(p_aligned.as_slice()[..4], p_unaligned_raw[1..5]);
    assert!(p_unaligned_raw[1..5].iter().any(|&v| v != 0x7F));
    assert!((r1_aligned.scale_factor - r1_unaligned.scale_factor).abs() < 1e-6);

    // Rank-1 overlap check both at C-ABI level and directly on unsafe Rust kernel API.
    let x_vec = AlignedBuffer32::<f32, 32>::new([1.0; 32]);
    let mut y_vec = AlignedBuffer32::<f32, 32>::new([0.0; 32]);
    let u_vec = AlignedBuffer32::<f32, 32>::new([0.25; 32]);
    let v_vec = AlignedBuffer32::<f32, 32>::new([0.5; 32]);
    let bad_desc = lmnk_rank1_descriptor_t {
        layer_idx: 0,
        rank: 1,
        u_vector_ptr: y_vec.as_slice().as_ptr() as u64, // Aliases y_vec mutably!
        v_vector_ptr: v_vec.as_slice().as_ptr() as u64,
        gain_scalar_ptr: 0,
        scale_factor: 1.0,
        flags: 0,
        reserved: [0; 24],
    };
    let good_desc = lmnk_rank1_descriptor_t {
        u_vector_ptr: u_vec.as_slice().as_ptr() as u64,
        ..bad_desc
    };
    // SAFETY: Testing aliasing rejection and valid execution for unsafe Rank-1 kernel + C-ABI.
    unsafe {
        let x_aligned = x_vec.as_aligned_slice();
        let mut y_aligned = y_vec.as_aligned_slice_mut();
        assert_eq!(
            eval_rank1(&mut y_aligned, &x_aligned, &bad_desc),
            Err(LMNK_ERR_INVALID_ARGUMENT)
        );
        assert_eq!(
            eval_rank1_fused(&mut y_aligned, &x_aligned, &bad_desc),
            Err(LMNK_ERR_INVALID_ARGUMENT)
        );
        assert_eq!(
            lmnk_rank1_eval(
                y_vec.as_mut_slice().as_mut_ptr(),
                x_vec.as_slice().as_ptr(),
                &good_desc,
                32,
                32
            ),
            LMNK_OK
        );
        assert_eq!(
            lmnk_rank1_eval_fused(
                y_vec.as_mut_slice().as_mut_ptr(),
                x_vec.as_slice().as_ptr(),
                &good_desc,
                32,
                32
            ),
            LMNK_OK
        );
    }

    // 4. Previously untested C-ABI families:
    // (a) lmnk_tagged_ptr_*
    let packed = lmnk_tagged_ptr_pack(0x0000_1234_5678, 0xA5, -17);
    assert_eq!(lmnk_tagged_ptr_unpack_addr(packed), 0x0000_1234_5678);
    assert_eq!(lmnk_tagged_ptr_unpack_tag(packed), 0xA5);
    assert_eq!(lmnk_tagged_ptr_unpack_phase(packed), -17);
    let act = lmnk_tagged_ptr_pack_action(0x0000_DEAD_BEE0, 3, ACTION_MODE_PHASE_STEER, 12);
    assert_eq!(lmnk_tagged_ptr_unpack_attractor_slot(act), 3);
    assert_ne!(
        lmnk_tagged_ptr_unpack_action_mode(act) & ACTION_MODE_PHASE_STEER,
        0
    );
    let evolved = lmnk_tagged_ptr_evolve_mode(act, 0.9, 0.6, 0, 2, 0.1, 0.8);
    assert_eq!(lmnk_tagged_ptr_unpack_attractor_slot(evolved), 2);

    // (b) lmnk_phase_steering_* (create/destroy, soft damping, fused logit shift, subhead dispatch, steer gateway)
    let mut ps_handle: LmnkPhaseSteeringHandle = ptr::null_mut();
    let subhead_cfg = SubheadConfig {
        num_fact_heads: 1,
        num_logic_heads: 1,
        num_explore_heads: 0,
        flags: 0,
        fact_phase_gain: 0.25,
        logic_phase_gain: 0.15,
        explore_phase_gain: 0.1,
        reserved: [0; 12],
    };
    let mut frame_out = GatewayFrame::default();
    let mut report_out = liminika_core::abi::lmnk_causal_report_t::default();
    let mut logits = AlignedBuffer32::<f32, 64>::new([0.1; 64]);
    // SAFETY: Exercising phase steering C-ABI entrypoints on valid aligned local buffers.
    unsafe {
        assert_eq!(lmnk_phase_steering_create(64, 2, &mut ps_handle), LMNK_OK);
        assert_eq!(lmnk_phase_steering_destroy(ps_handle), LMNK_OK);
        assert_eq!(
            lmnk_phase_steering_fused_logit_shift(
                logits.as_mut_slice().as_mut_ptr(),
                64,
                phases_16.as_slice().as_ptr(),
                2
            ),
            LMNK_OK
        );
        assert_eq!(
            lmnk_subhead_dispatch_single_pass(good_q.as_mut_slice().as_mut_ptr(), &subhead_cfg, 64),
            LMNK_OK
        );
        assert_eq!(
            lmnk_gateway_synthesize_multi_origin_frame(
                0.2,
                0.1,
                0xFF,
                0,
                phases_16.as_slice().as_ptr(),
                0.9,
                0x1000,
                1,
                ptr::null(),
                0.0,
                0,
                0,
                ptr::null(),
                0.6,
                3,
                &mut frame_out,
                &mut report_out,
            ),
            LMNK_OK
        );
        assert_eq!(
            lmnk_phase_steering_steer_gateway_frame_safe(
                &frame_out,
                &mut trail_out,
                64,
                good_q.as_mut_slice().as_mut_ptr(),
                good_k.as_mut_slice().as_mut_ptr(),
                64,
                2,
                logits.as_mut_slice().as_mut_ptr(),
                64,
                &mut report_out
            ),
            LMNK_OK
        );
    }

    // (c) lmnk_substrate_* including misaligned destroy & step_safe guard checks.
    let bad_sub_ptr = mis_raw[1..]
        .as_mut_ptr()
        .cast::<liminika_core::abi::LmnkSubstrateOpaque>();
    // SAFETY: Misaligned substrate handle must be rejected cleanly with LMNK_ERR_ALIGNMENT.
    unsafe {
        assert_eq!(lmnk_substrate_destroy(bad_sub_ptr), LMNK_ERR_ALIGNMENT);
    }
    let mut thresh = StigmergicThresholds::default();
    let mut sub_handle: LmnkSubstrateHandle = ptr::null_mut();
    let mut sub_rep = SubstrateStepReport::default();
    let mut swarm_rep = StigmergicReconstitutionReport::default();
    let mut ctx_sub: LmnkContextHandle = ptr::null_mut();
    let masks = AlignedBuffer32::<PagedBitmask, 2>::new([PagedBitmask::default(); 2]);
    let mut reduced_mask = PagedBitmask::default();
    let kv_bits = [1u64, 0u64, 4u64, 0u64];
    // SAFETY: Validating misaligned step_safe rejection and full substrate family behavior.
    unsafe {
        assert_eq!(lmnk_substrate_init_default_thresholds(&mut thresh), LMNK_OK);
        assert_eq!(
            lmnk_substrate_step_safe(
                bad_sub_ptr,
                &thresh,
                0x1000,
                0.3,
                1,
                &mut trail_out,
                ptr::null_mut(),
                0,
                good_q.as_mut_slice().as_mut_ptr(),
                good_k.as_mut_slice().as_mut_ptr(),
                64,
                2,
                ptr::null_mut(),
                0,
                &mut sub_rep,
            ),
            LMNK_ERR_ALIGNMENT
        );
        assert_eq!(lmnk_substrate_create(16, &mut sub_handle), LMNK_OK);
        assert_eq!(
            lmnk_substrate_evaluate_thresholds(
                &thresh,
                &trail_out,
                0.3,
                0.8,
                -0.1,
                0,
                0,
                1,
                &mut sub_rep
            ),
            LMNK_OK
        );
        assert_eq!(
            lmnk_substrate_resolve_trail_in_swarm(
                &mut trail_out,
                0x1000,
                phases_16.as_slice().as_ptr(),
                0xFF,
                64,
                2,
                16,
                4,
                0.4,
                0.15,
                &mut swarm_rep
            ),
            LMNK_OK
        );
        assert_eq!(
            lmnk_substrate_reduce_bit_tree(masks.as_slice().as_ptr(), 2, &mut reduced_mask),
            LMNK_OK
        );
        assert_eq!(
            lmnk_substrate_paged_block_warp_bitmask(kv_bits.as_ptr(), 4, &mut reduced_mask),
            LMNK_OK
        );
        assert!(lmnk_substrate_hash_slot256_index(0xDEAD_BEEF) < 256);
        assert_eq!(lmnk_context_create(&cfg, &mut ctx_sub), LMNK_OK);
        assert_eq!(
            lmnk_substrate_step_pointer(ctx_sub, 0x2000, &mut trail_out),
            LMNK_OK
        );
        assert_eq!(
            lmnk_substrate_pull_trail_to_anchor(ctx_sub, 0, 1, 0.5),
            LMNK_OK
        );
        assert_eq!(
            lmnk_substrate_pull_trail_to_anchor(ctx_sub, 0, 1, f32::NAN),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_substrate_link_hallucinated_trails(ctx_sub, 0, 1),
            LMNK_OK
        );
        assert_eq!(lmnk_context_destroy(ctx_sub), LMNK_OK);
        assert_eq!(lmnk_substrate_destroy(sub_handle), LMNK_OK);
    }

    // (d) lmnk_ledger_*
    let mut ledger = LedgerSlot::default();
    let mut shortcut = 0u64;
    let mut tuned_cfg = lmnk_config_t::default();
    // SAFETY: LedgerSlot and output pointers are valid and aligned.
    unsafe {
        assert_eq!(lmnk_ledger_reset_telemetry(&mut ledger), LMNK_OK);
        ledger.fast_path_bypass_mask = 0xFFFF;
        ledger.fast_path_shortcuts[0] = 0x1234_5678;
        assert_eq!(lmnk_ledger_set_mode(&mut ledger, 0), LMNK_OK);
        assert_eq!(
            lmnk_ledger_fast_path_lookup(&ledger, 42, &mut shortcut),
            LMNK_OK
        );
        assert_eq!(shortcut, 0x1234_5678);
        assert_eq!(lmnk_ledger_set_mode(&mut ledger, 1), LMNK_OK);
        assert_eq!(
            lmnk_ledger_fast_path_lookup(&ledger, 42, &mut shortcut),
            LMNK_ERR_INVALID_STATE
        );
        assert_eq!(lmnk_ledger_telemetry_push(&mut ledger, 0x07), LMNK_OK);
        assert_eq!(
            lmnk_ledger_evaluate_auto_tune(&ledger, &mut tuned_cfg),
            LMNK_OK
        );
    }

    // (e) lmnk_liminal_eval_* + lmnk_distortion_delta_apply_chain
    let mut lacuna_score = 0.0f32;
    let mut barrier = 0.0f32;
    let mut part = LacunaPartition8::default();
    let other_q7 = [32u8; NUM_STIGMERGIC_GRANULES];
    let tol_q7 = [4u8; NUM_STIGMERGIC_GRANULES];
    let key_32 = AlignedBuffer32::<f32, 32>::new([0.5; 32]);
    let mut delta_node = DistortionDelta::new_empty(1, 2);
    delta_node.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
    delta_node.delta_phase_q7 = [5i8; MAX_ATTENTION_HEADS];
    let ego_s = LiminalEgoState::from_preset(LiminalEgoPreset::Low);
    let spine_s = SystemSpineState::new();
    let stig_s = StigmergicGranuleState::default();
    let mut ten_s = DistortionTension::new(false);
    let mut d_norm_out = 0u8;
    let mut hops_out = 0u8;
    // SAFETY: Exercising liminal & distortion chain C-ABI with aligned local state.
    unsafe {
        assert_eq!(
            lmnk_liminal_eval_dyadic128(key_32.as_slice().as_ptr(), 32, &mut lacuna_score),
            LMNK_OK
        );
        assert_eq!(
            lmnk_liminal_eval_barrier(lacuna_score.max(0.1), 0.5, &mut barrier),
            LMNK_OK
        );
        assert_eq!(
            lmnk_liminal_eval_barrier_pitchfork(lacuna_score, 0.5, 1.0, 1.0, 64, &mut barrier),
            LMNK_OK
        );
        assert_eq!(
            lmnk_liminal_evaluate_lacuna_partition(
                0xFF,
                other_q7.as_ptr(),
                tol_q7.as_ptr(),
                16,
                &mut part
            ),
            LMNK_OK
        );
        assert_eq!(
            lmnk_distortion_tension_step_hot_path(&mut ten_s, 16, 32, 24, 8, 0),
            LMNK_OK
        );
        assert_eq!(
            lmnk_distortion_delta_apply_chain(
                &delta_node,
                1,
                &ego_s,
                &spine_s,
                &part,
                &stig_s,
                &ten_s,
                2,
                &mut frame_out,
                &mut d_norm_out,
                &mut hops_out,
            ),
            LMNK_OK
        );
        assert!(hops_out <= 2);
    }
}

fn restamp_snapshot_crcs(buf: &mut [u8]) {
    let sec_entry_off = LMNK_CONTAINER_HEADER_BYTES;
    let sec_off = u32::from_le_bytes(
        buf[sec_entry_off + 4..sec_entry_off + 8]
            .try_into()
            .unwrap(),
    ) as usize;
    let sec_len = u32::from_le_bytes(
        buf[sec_entry_off + 8..sec_entry_off + 12]
            .try_into()
            .unwrap(),
    ) as usize;
    let sec_crc = compute_crc32_ieee(&buf[sec_off..sec_off + sec_len]);
    buf[sec_entry_off + 12..sec_entry_off + 16].copy_from_slice(&sec_crc.to_le_bytes());
    restamp_header_crc32(buf);
}

#[test]
fn suite4_spine_privilege_isolation_and_transactional_snapshot_rejection() {
    let mut src = RuntimeContext::new_boxed(Config::new()).expect("src ctx");
    src.substrate_mut().step_counter = 42;
    src.substrate_mut().last_entropy_h_tilde = 0.25;
    src.substrate_mut().slot_weights[0] = 1.5;

    let mut snap = vec![0u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut req = 0usize;
    src.serialize_to_bytes(&mut snap, &mut req)
        .expect("serialize");
    assert_eq!(req, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES);

    let sec_off = u32::from_le_bytes(
        snap[LMNK_CONTAINER_HEADER_BYTES + 4..LMNK_CONTAINER_HEADER_BYTES + 8]
            .try_into()
            .unwrap(),
    ) as usize;

    // 1. Valid CRC32 snapshot trying to zero out pinned_compute_mask, baseline_pinned_compute_mask,
    //    and slot_pinned_mask must preserve host floor bits and HOST_IMMUTABLE_PINNED_COMPUTE_MASK.
    let mut priv_snap = snap.clone();
    // Payload offset 21 is slot_pinned_mask; 320 + 8 = 328 is pinned_compute_mask;
    // 320 + 32 = 352 is SystemSpineState.baseline_pinned_compute_mask.
    priv_snap[sec_off + 21] = 0;
    priv_snap[sec_off + 328..sec_off + 336].copy_from_slice(&0u64.to_le_bytes());
    priv_snap[sec_off + 352..sec_off + 360].copy_from_slice(&0u64.to_le_bytes());
    restamp_snapshot_crcs(&mut priv_snap);

    let mut dst = RuntimeContext::new_boxed(Config::new()).expect("dst ctx");
    dst.apply_host_spine_pin_floor(0x0E);
    dst.substrate_mut().slot_pinned_mask = 0x05;
    dst.deserialize_from_bytes(&priv_snap).expect("deserialize");
    assert_ne!(
        dst.system_spine().baseline_pinned_compute_mask & HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
        0
    );
    assert_eq!(dst.system_spine().baseline_pinned_compute_mask & 0x0E, 0x0E);
    assert_ne!(
        dst.system_spine().pinned_compute_mask & HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
        0
    );
    assert_eq!(dst.system_spine().pinned_compute_mask & 0x0E, 0x0E);
    assert_eq!(dst.slot_pinned_mask() & 0x05, 0x05);

    // 1b. Active transient turn-guard pins must not bleed into permanent host floor on restore.
    let active_mask = 0xFFFF_u64;
    dst.compute_pool_mut()
        .set_slot_masks(active_mask, 0x0F, 0, active_mask & !0x0F);
    dst.system_spine_mut().set_baseline_pin_mask(0x0F);
    dst.begin_turn_spine_policy(true, false);
    assert_eq!(dst.system_spine().pinned_compute_mask, active_mask);
    dst.deserialize_from_bytes(&snap)
        .expect("restore unguarded");
    assert_eq!(dst.system_spine().baseline_pinned_compute_mask, 0x0F);
    assert_eq!(dst.system_spine().pinned_compute_mask, 0x0F);

    // 2. Transactional rejection of valid-CRC32 corrupted snapshots: dst must remain untouched!
    dst.substrate_mut().step_counter = 999;
    dst.substrate_mut().last_entropy_h_tilde = 0.33;
    let expected_head_dim = dst.config().raw.head_dim;

    #[allow(clippy::type_complexity)]
    let cases: Vec<(&str, Box<dyn Fn(&mut [u8], usize)>)> = vec![
        (
            "NaN last_entropy_h_tilde",
            Box::new(|b, off| b[off + 16..off + 20].copy_from_slice(&f32::NAN.to_le_bytes())),
        ),
        (
            "Inf slot_weights[0]",
            Box::new(|b, off| b[off + 32..off + 36].copy_from_slice(&f32::INFINITY.to_le_bytes())),
        ),
        (
            "NaN EntropyState.moving_z_entropy",
            Box::new(|b, off| b[off + 64..off + 68].copy_from_slice(&f32::NAN.to_le_bytes())),
        ),
        (
            "Invalid EntropyState.active_branch_mode",
            Box::new(|b, off| b[off + 64 + 24] = 9),
        ),
        (
            "Zero head_dim",
            Box::new(|b, off| b[off + 100..off + 104].copy_from_slice(&0u32.to_le_bytes())),
        ),
        (
            "Odd head_dim",
            Box::new(|b, off| b[off + 100..off + 104].copy_from_slice(&63u32.to_le_bytes())),
        ),
        (
            "num_heads > MAX_ATTENTION_HEADS (65)",
            Box::new(|b, off| b[off + 104..off + 108].copy_from_slice(&65u32.to_le_bytes())),
        ),
        (
            "rotary_dim > head_dim",
            Box::new(|b, off| b[off + 108..off + 112].copy_from_slice(&256u32.to_le_bytes())),
        ),
        (
            "Invalid rope_layout (9)",
            Box::new(|b, off| b[off + 112] = 9),
        ),
        (
            "Out-of-range active_count",
            Box::new(|b, off| b[off + 8..off + 12].copy_from_slice(&999u32.to_le_bytes())),
        ),
        (
            "Invalid SystemSpineState.control_flags",
            Box::new(|b, off| b[off + 320 + 48] = 0x80),
        ),
        (
            "Out-of-range SystemSpineState.last_observed_entropy_q15",
            Box::new(|b, off| {
                b[off + 320 + 40..off + 320 + 42].copy_from_slice(&40_000u16.to_le_bytes())
            }),
        ),
        (
            "Invalid SystemSpineState.last_veto_mask",
            Box::new(|b, off| b[off + 320 + 49] = 0x80),
        ),
    ];

    for (label, mutate) in cases {
        let mut corrupt = snap.clone();
        mutate(&mut corrupt, sec_off);
        restamp_snapshot_crcs(&mut corrupt);
        assert_eq!(
            dst.deserialize_from_bytes(&corrupt),
            Err(LMNK_ERR_CONTAINER_CORRUPTED),
            "case '{label}' was not rejected"
        );
        // Verify zero mutation on failure (transactional rollback / pre-validation invariant).
        assert_eq!(
            dst.step_counter(),
            999,
            "step_counter mutated on failed case '{label}'"
        );
        assert!(
            (dst.substrate().last_entropy_h_tilde - 0.33).abs() < 1e-6,
            "last_entropy_h_tilde mutated on failed case '{label}'"
        );
        assert_eq!(
            dst.config().raw.head_dim,
            expected_head_dim,
            "config mutated on failed case '{label}'"
        );
    }
}

#[test]
fn suite5_uaf_free_zeroize_volatile_all_5120_bytes_proof() {
    let mut cfg = lmnk_config_t::default();
    // SAFETY: Stack config is 64B-aligned and writable.
    assert_eq!(unsafe { lmnk_config_init_default(&mut cfg) }, LMNK_OK);

    let req = lmnk_context_size();
    assert_eq!(req, 5120);
    let mut backing = AlignedBuffer64::<u8, 5120>::new([0xAA; 5120]);
    let mem_ptr = backing.as_mut_slice().as_mut_ptr();

    // SAFETY: Caller-owned 64B-aligned 5,120B buffer initialized and scrubbed via C-ABI.
    unsafe {
        assert_eq!(lmnk_context_init_in_place(&cfg, mem_ptr, req), LMNK_OK);
        let ctx_handle = mem_ptr.cast::<liminika_core::abi::LmnkContextOpaque>();
        let mut q = AlignedBuffer32::<f32, 128>::new([0.75; 128]);
        let mut k = AlignedBuffer32::<f32, 128>::new([-0.5; 128]);
        let mut trail = lmnk_pointer_trail_t::default();
        assert_eq!(
            lmnk_context_step_token(
                ctx_handle,
                7,
                q.as_mut_slice().as_mut_ptr(),
                k.as_mut_slice().as_mut_ptr(),
                &mut trail
            ),
            LMNK_OK
        );
        assert_eq!(lmnk_context_deinit_in_place(ctx_handle), LMNK_OK);
    }
    assert!(
        backing.as_slice().iter().all(|&b| b == 0x00),
        "all 5,120 bytes must equal 0x00 after lmnk_context_deinit_in_place"
    );

    // Direct Rust `RuntimeContext::zeroize_in_place(&mut ctx)` proof on boxed context.
    let mut ctx = RuntimeContext::new_boxed(Config::new()).expect("boxed ctx");
    ctx.substrate_mut().step_counter = 12345;
    ctx.substrate_mut().slot_weights = [9.0; 8];
    ctx.core_layers_mut().ego = LiminalEgoState::from_preset(LiminalEgoPreset::High);
    ctx.apply_host_spine_pin_floor(0xFFFF);
    ctx.zeroize_in_place();
    let raw_ptr = (&*ctx as *const RuntimeContext).cast::<u8>();
    // SAFETY: `ctx` is valid, live, and uniquely borrowed; inspecting initialized memory representation.
    let raw_bytes =
        unsafe { std::slice::from_raw_parts(raw_ptr, std::mem::size_of::<RuntimeContext>()) };
    assert!(
        raw_bytes.iter().all(|&b| b == 0x00),
        "RuntimeContext::zeroize_in_place must zero all 5,120 bytes cleanly"
    );
}

#[test]
fn suite6_spine_abi_adversarial_alignment_overlap_and_pin_floor_guards() {
    let mut cfg = lmnk_config_t::default();
    // SAFETY: Stack config is 64B aligned and writable.
    assert_eq!(unsafe { lmnk_config_init_default(&mut cfg) }, LMNK_OK);

    let mut backing = AlignedBuffer64::<u8, 5248>::new([0u8; 5248]);
    let base_ptr = backing.as_mut_slice().as_mut_ptr();
    let req_size = lmnk_context_size();
    assert_eq!(req_size, 5120);

    // SAFETY: Exercising hostile NULL, misaligned, and overlapping pointers against C-ABI guards.
    unsafe {
        assert_eq!(
            lmnk_context_init_in_place(&cfg, base_ptr, req_size),
            LMNK_OK
        );
        let ctx_handle = base_ptr.cast::<liminika_core::abi::LmnkContextOpaque>();
        let misalign_ctx = base_ptr
            .add(16)
            .cast::<liminika_core::abi::LmnkContextOpaque>();

        // 1. NULL and misaligned handle rejections on `configure_spine` & `get_spine_state`.
        let mut spine_out = lmnk_spine_state_t::default();
        assert_eq!(
            lmnk_context_configure_spine(ptr::null_mut(), 0, 0, 1, 1),
            LMNK_ERR_NULL_POINTER
        );
        assert_eq!(
            lmnk_context_configure_spine(misalign_ctx, 0, 0, 1, 1),
            LMNK_ERR_ALIGNMENT
        );
        assert_eq!(
            lmnk_context_get_spine_state(ptr::null_mut(), &mut spine_out),
            LMNK_ERR_NULL_POINTER
        );
        assert_eq!(
            lmnk_context_get_spine_state(ctx_handle, ptr::null_mut()),
            LMNK_ERR_NULL_POINTER
        );
        assert_eq!(
            lmnk_context_get_spine_state(misalign_ctx, &mut spine_out),
            LMNK_ERR_ALIGNMENT
        );

        let mut raw_spine_buf = AlignedBuffer64::<u8, 128>::new([0u8; 128]);
        let misalign_spine_out = raw_spine_buf.as_mut_slice().as_mut_ptr().add(16);
        assert_eq!(
            lmnk_context_get_spine_state(ctx_handle, misalign_spine_out.cast()),
            LMNK_ERR_ALIGNMENT
        );

        // Overlapping `state_out` inside the 5,120 B `context` buffer (e.g. `system_spine @ +384`) must fail!
        let overlap_spine_out = base_ptr.add(384).cast::<lmnk_spine_state_t>();
        assert_eq!(
            lmnk_context_get_spine_state(ctx_handle, overlap_spine_out),
            LMNK_ERR_INVALID_ARGUMENT
        );

        // 2. Monotonic non-strippable host pin floor (`0` preserves floor; bit 0 is always pinned).
        assert_eq!(
            lmnk_context_configure_spine(ctx_handle, 0x00A0, 4, 0, 0),
            LMNK_OK
        );
        assert_eq!(
            lmnk_context_get_spine_state(ctx_handle, &mut spine_out),
            LMNK_OK
        );
        assert_eq!(
            spine_out.baseline_pinned_compute_mask,
            0x00A0 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
        assert_eq!(
            spine_out.pinned_compute_mask,
            0x00A0 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
        assert_eq!(spine_out.guard_hold_tokens, 4);
        assert_eq!(spine_out.max_observe_stride, 1);

        // Passing `0` and running an unguarded turn cannot strip `0x00A1`.
        assert_eq!(
            lmnk_context_configure_spine(ctx_handle, 0, 2, 6, 1),
            LMNK_OK
        );
        assert_eq!(
            lmnk_context_begin_turn_spine_policy(ctx_handle, 0, 0),
            LMNK_OK
        );
        assert_eq!(
            lmnk_context_get_spine_state(ctx_handle, &mut spine_out),
            LMNK_OK
        );
        assert_eq!(
            spine_out.baseline_pinned_compute_mask,
            0x00A0 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
        assert_eq!(
            spine_out.pinned_compute_mask,
            0x00A0 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
        assert_eq!(spine_out.guard_hold_tokens, 2);
        assert_eq!(spine_out.max_observe_stride, 6);

        // 3. `lmnk_spine_step_plan_t` requires 64B alignment (`16B`-only pointer must return `LMNK_ERR_ALIGNMENT`).
        let decode_in = lmnk_spine_decode_input_t {
            flags: SPINE_DECODE_FLAG_REWARD_OVERRIDE,
            exit_reward_override_q7: 250,
            ..Default::default()
        };
        let mut plan_buf = AlignedBuffer64::<u8, 128>::new([0u8; 128]);
        let misalign_plan = plan_buf
            .as_mut_slice()
            .as_mut_ptr()
            .add(16)
            .cast::<lmnk_spine_step_plan_t>();
        assert_eq!(
            lmnk_context_step_decode_control(ctx_handle, &decode_in, 0, misalign_plan),
            LMNK_ERR_ALIGNMENT
        );
        let mut plan_ok = lmnk_spine_step_plan_t::default();
        assert_eq!(
            lmnk_context_step_decode_control(ctx_handle, &decode_in, 0, &mut plan_ok),
            LMNK_OK
        );

        assert_eq!(lmnk_context_deinit_in_place(ctx_handle), LMNK_OK);

        // 4. Concrete alignment & aliasing checks on destroy / ledger / entropy entry points.
        assert_eq!(
            lmnk_phase_steering_destroy(
                base_ptr.add(1).cast::<LmnkPhaseSteeringOpaque>() as LmnkPhaseSteeringHandle
            ),
            LMNK_ERR_ALIGNMENT
        );
        assert_eq!(
            lmnk_entropy_controller_destroy(base_ptr.add(16).cast::<LmnkEntropyControllerOpaque>()
                as LmnkEntropyControllerHandle),
            LMNK_ERR_ALIGNMENT
        );

        let mut shared_64 = AlignedBuffer64::<u8, 64>::new([0u8; 64]);
        let same_ptr = shared_64.as_mut_slice().as_mut_ptr();
        assert_eq!(
            lmnk_ledger_evaluate_auto_tune(
                same_ptr.cast::<lmnk_ledger_slot_t>(),
                same_ptr.cast::<lmnk_config_t>(),
            ),
            LMNK_ERR_INVALID_ARGUMENT
        );

        let init_ent = lmnk_entropy_state_t {
            bifurcation_threshold: 1.5,
            singularity_eps: 1e-12,
            sample_temperature: 1.0,
            ..Default::default()
        };
        let mut ent_handle: LmnkEntropyControllerHandle = ptr::null_mut();
        assert_eq!(
            lmnk_entropy_controller_create(&init_ent, &mut ent_handle),
            LMNK_OK
        );
        assert_eq!(
            lmnk_entropy_eval_stateful(ent_handle, 0.5, ent_handle.cast::<lmnk_entropy_state_t>()),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(lmnk_entropy_controller_destroy(ent_handle), LMNK_OK);
    }
}
