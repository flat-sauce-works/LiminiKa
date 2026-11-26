// name: tests/core/test_security_adversarial.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Permanent adversarial security & soundness integration suites for `liminika-core` (`v0.4.0`).
//!
//! Covers:
//! 1. `.lmnkc` deterministic mutation fuzzing (`>= 2,000` mutations), alignment/topology guards,
//!    strict padding/gap/trailing/flag validation, and `Lacuna::Unobserved` reserved sections.
//! 2. Cycle & out-of-bounds termination defense for `PointerTrail` and `EgoDifferentialNet`.
//! 3. Hostile C-ABI `0x0000_0400` calls across all 25 exports (`struct_size` fail-closed,
//!    `num_tokens != 1`, `Rank1Descriptor` `ArenaSpan` overflow, NULL, misalignment, aliasing,
//!    `usize::MAX`, `NaN`/`Inf`, cross-mode destroy/deinit) + direct Rust API parity.
//! 4. `SystemSpineState` privilege isolation & transactional `NaN`/`Inf`/corrupt snapshot rejection.
//! 5. UAF-free `zeroize_volatile` proof for `5,120`-byte `RuntimeContext` buffers.
//! 6. Spine policy ABI alignment/overlap/monotonic pin-floor guards + `0x0000_0400` version policy.

use std::ptr;

use liminika_core::abi::{
    LMNK_POLICY_FLAG_COMPUTE_POOL, LMNK_POLICY_FLAG_SPINE, LMNK_RANK1_FLAG_FUSED,
    LMNK_ROPE_STEER_FLAG_UNCLAMPED, LMNK_TURN_FLAG_GUARD_ACTIVE, LmnkContextHandle,
    SPINE_DECODE_FLAG_REWARD_OVERRIDE, lmnk_abi_info, lmnk_abi_info_t, lmnk_abi_version_hex,
    lmnk_abi_version_string, lmnk_config_init_default, lmnk_config_t, lmnk_config_validate,
    lmnk_container_find_section, lmnk_container_section_view_t, lmnk_container_verify,
    lmnk_container_verify_report_t, lmnk_container_write, lmnk_container_write_params_t,
    lmnk_context_begin_turn, lmnk_context_configure_policy, lmnk_context_create,
    lmnk_context_deinit_in_place, lmnk_context_deserialize, lmnk_context_destroy,
    lmnk_context_get_state, lmnk_context_init_in_place, lmnk_context_reset, lmnk_context_serialize,
    lmnk_context_step_decode, lmnk_decode_step_params_t, lmnk_decode_step_report_t,
    lmnk_gateway_steer, lmnk_gateway_steer_params_t, lmnk_lacuna_eval_params_t,
    lmnk_lacuna_partition_eval, lmnk_pointer_trail_t, lmnk_q7_t, lmnk_rank1_descriptor_t,
    lmnk_rank1_eval, lmnk_rank1_eval_params_t, lmnk_rope_procrustes_params_t,
    lmnk_rope_procrustes_q7, lmnk_rope_steer_params_t, lmnk_rope_steer_q, lmnk_runtime_policy_t,
    lmnk_runtime_state_view_t, lmnk_section_input_t, lmnk_spine_decode_input_t,
    lmnk_status_to_string, lmnk_turn_begin_params_t,
};
use liminika_core::drivers::{
    DISTORTION_DELTA_FLAG_ACTIVE, DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID, DistortionDelta,
    DistortionTension, EgoDifferentialNet, EntropyController, EntropyState,
    HOST_IMMUTABLE_PINNED_COMPUTE_MASK, LiminalEgoPreset, LiminalEgoState, eval_barrier_pitchfork,
};
use liminika_core::foundation::{
    ACTION_MODE_PHASE_STEER, AlignedBuffer16, AlignedBuffer32, AlignedBuffer64, ArenaSpan,
    LoraDimensions, PagedBitmask, Pod, RANK1_GAIN_UNUSED, SubheadConfig, TaggedPtr,
    compute_crc32_ieee, dispatch_single_pass, eval_rank1, eval_rank1_fused, project_lora,
};
use liminika_core::liminal::{
    LMNK_ERR_ALIGNMENT, LMNK_ERR_ANCHOR_NOT_FOUND, LMNK_ERR_BUFFER_TOO_SMALL,
    LMNK_ERR_CONTAINER_CORRUPTED, LMNK_ERR_INVALID_ARGUMENT, LMNK_ERR_INVALID_STATE,
    LMNK_ERR_NOT_IMPLEMENTED, LMNK_ERR_NULL_POINTER, LMNK_ERR_TOPOLOGY_MISMATCH, LMNK_OK,
    LacunaPartition8, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
    evaluate_lacuna_partition_branchless,
};
use liminika_core::runtime::{
    CORE_TOPOLOGY, Config, GatewayFrame, LMNK_CONTAINER_FLAG_COMPILED_CACHE,
    LMNK_CONTAINER_HEADER_BYTES, LMNK_MAX_CONTAINER_SECTIONS, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES,
    LMNK_SECTION_BLOOM_FILTER, LMNK_SECTION_CONTENT_HASH, LMNK_SECTION_ENTRY_BYTES,
    LMNK_SECTION_PHASE_ANCHORS, LMNK_SECTION_POINTER_TRAILS, LMNK_SECTION_SIGNATURE,
    LmnkContainerView, LmnkSectionInput, RuntimeContext,
};
use liminika_core::stigmergy::{
    ATTRACTOR_SLOT_KNOWLEDGE, ContrastivePhaseBloomFilter, LedgerSlot, PhaseAnchorRecord,
    PointerTrail, StigmergicGranuleState, StigmergicSubstrate,
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
    let frame = GatewayFrame {
        target_anchor_id: 101,
        effective_energy: 1.5,
        dominant_slot: ATTRACTOR_SLOT_KNOWLEDGE,
        phase_deltas: [12i8; MAX_ATTENTION_HEADS],
        ..GatewayFrame::default()
    };
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

    // 2. 1-byte misaligned buffer: safe unaligned parse and generic C-ABI verify succeed;
    //    64B zero-copy slice projection fails with LMNK_ERR_ALIGNMENT.
    let mut aligned_backing = AlignedBuffer64::<u8, 4096>::new([0u8; 4096]);
    assert!(valid_buf.len() + 64 <= aligned_backing.as_slice().len());
    aligned_backing.as_mut_slice()[1..1 + valid_buf.len()].copy_from_slice(&valid_buf);
    let misaligned = &aligned_backing.as_slice()[1..1 + valid_buf.len()];
    let mis_view = LmnkContainerView::parse(misaligned).expect("unaligned parse ok");
    assert_eq!(mis_view.phase_anchors_zero_copy(), Err(LMNK_ERR_ALIGNMENT));

    let mut verify_rep = lmnk_container_verify_report_t::default();
    let mut sec_view = lmnk_container_section_view_t::default();
    // SAFETY: Misaligned byte buffer is valid for `misaligned.len()` bytes; generic byte-oriented APIs succeed.
    unsafe {
        assert_eq!(
            lmnk_container_verify(misaligned.as_ptr(), misaligned.len(), &mut verify_rep),
            LMNK_OK
        );
        assert_eq!(verify_rep.section_count, 5);
        assert_eq!(
            lmnk_container_find_section(
                misaligned.as_ptr(),
                misaligned.len(),
                LMNK_SECTION_CONTENT_HASH,
                &mut sec_view,
            ),
            LMNK_OK
        );
        assert_eq!(sec_view.length, 32);
        assert_eq!(
            lmnk_container_find_section(
                misaligned.as_ptr(),
                misaligned.len(),
                0xBEEF,
                &mut sec_view,
            ),
            LMNK_ERR_ANCHOR_NOT_FOUND
        );
    }

    // 3. Mutated topology_crc32 -> LMNK_ERR_TOPOLOGY_MISMATCH.
    let mut topo_bad = valid_buf.clone();
    topo_bad[8..12].copy_from_slice(&(CORE_TOPOLOGY.topology_crc32 ^ 0xDEAD_BEEF).to_le_bytes());
    restamp_header_crc32(&mut topo_bad);
    assert_eq!(
        LmnkContainerView::parse(&topo_bad),
        Err(LMNK_ERR_TOPOLOGY_MISMATCH)
    );

    // 4. Explicit structural rejections with valid header CRC32 where applicable.
    let mut table_pad_bad = valid_buf.clone();
    table_pad_bad[150] = 0xAA;
    assert_eq!(
        LmnkContainerView::parse(&table_pad_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

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

    let mut trailing_bad = valid_buf.clone();
    trailing_bad.extend_from_slice(&[0u8; 64]);
    let trail_total = trailing_bad.len() as u64;
    trailing_bad[16..24].copy_from_slice(&trail_total.to_le_bytes());
    restamp_header_crc32(&mut trailing_bad);
    assert_eq!(
        LmnkContainerView::parse(&trailing_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    let mut flags_bad = valid_buf.clone();
    let unknown_flag = LMNK_CONTAINER_FLAG_COMPILED_CACHE | (1u32 << 30);
    flags_bad[12..16].copy_from_slice(&unknown_flag.to_le_bytes());
    restamp_header_crc32(&mut flags_bad);
    assert_eq!(
        LmnkContainerView::parse(&flags_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    let mut res_bad = valid_buf.clone();
    res_bad[40] = 1;
    restamp_header_crc32(&mut res_bad);
    assert_eq!(
        LmnkContainerView::parse(&res_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    let mut entry_flag_bad = valid_buf.clone();
    entry_flag_bad[entry3_off + 2..entry3_off + 4].copy_from_slice(&1u16.to_le_bytes());
    assert_eq!(
        LmnkContainerView::parse(&entry_flag_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    let mut count_bad = valid_buf.clone();
    count_bad[36..38].copy_from_slice(&(LMNK_MAX_CONTAINER_SECTIONS + 1).to_le_bytes());
    restamp_header_crc32(&mut count_bad);
    assert_eq!(
        LmnkContainerView::parse(&count_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    let mut overflow_bad = valid_buf.clone();
    overflow_bad[entry3_off + 8..entry3_off + 12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        LmnkContainerView::parse(&overflow_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

    let mut dup_bad = valid_buf.clone();
    dup_bad[entry4_off..entry4_off + 2].copy_from_slice(&LMNK_SECTION_CONTENT_HASH.to_le_bytes());
    assert_eq!(
        LmnkContainerView::parse(&dup_bad),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );

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
    let mutation_trials: usize = if cfg!(miri) { 64 } else { 2500 };
    for trial in 0..mutation_trials {
        let mut mutated = valid_buf.clone();
        let mut flipped_only_custom_kind = false;
        match trial % 5 {
            0 => {
                let mut idx = rng.next_usize(mutated.len());
                if idx >= table_start
                    && idx < table_end
                    && ((idx - table_start) % LMNK_SECTION_ENTRY_BYTES) < 2
                {
                    idx += 2;
                }
                let bit = 1u8 << (rng.next_usize(8) as u8);
                mutated[idx] ^= bit;
            }
            1 => {
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
                let new_len = rng.next_usize(mutated.len());
                mutated.truncate(new_len);
            }
            3 => {
                let slot = rng.next_usize(5);
                let off = LMNK_CONTAINER_HEADER_BYTES + slot * LMNK_SECTION_ENTRY_BYTES + 4;
                let word = u32::from_le_bytes(mutated[off..off + 4].try_into().unwrap())
                    ^ ((rng.next_u64() as u32) | 1);
                mutated[off..off + 4].copy_from_slice(&word.to_le_bytes());
            }
            _ => {
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
            let mut rep = lmnk_container_verify_report_t::default();
            // SAFETY: Aligned stack buffer of length `copy_len` verified strictly non-panicking.
            let zc_status = unsafe {
                lmnk_container_verify(aligned_backing.as_slice().as_ptr(), copy_len, &mut rep)
            };
            assert_ne!(zc_status, LMNK_OK);
        }
    }
    assert!(kind_only_unobserved_ok > 0);
}

#[test]
fn suite2_pointer_trail_and_ego_differential_cycle_defense() {
    let mut sub = StigmergicSubstrate::new();
    sub.active_count = 4;
    for (i, trail) in sub.trails.iter_mut().enumerate().take(4) {
        trail.current_ptr = ((i as u64) + 1) << 12;
        trail.target_anchor_id = (i as u32) + 10;
        trail.stigmergic_density = 0.5;
        trail.accumulated_phase_delta = [8i8; MAX_ATTENTION_HEADS];
    }
    sub.trails[0].linked_trail_id = 0;
    let mut trail_out = PointerTrail::default();
    let (_rep, hops_self, term_idx, _, _) = sub
        .traverse_pointer_chain_and_orchestrate(
            liminika_core::stigmergy::ChainTraverseParams {
                start_trail_idx: 0,
                max_hops: 8,
                entropy_h_tilde: 0.3,
            },
            &mut trail_out,
            None,
            None,
            None,
            None,
        )
        .expect("self loop bounded");
    assert_eq!(hops_self, 1);
    assert_eq!(term_idx, 0);

    sub.trails[0].linked_trail_id = 1;
    sub.trails[1].linked_trail_id = 0;
    let (_rep2, hops_cycle, _, _, _) = sub
        .traverse_pointer_chain_and_orchestrate(
            liminika_core::stigmergy::ChainTraverseParams {
                start_trail_idx: 0,
                max_hops: 8,
                entropy_h_tilde: 0.3,
            },
            &mut trail_out,
            None,
            None,
            None,
            None,
        )
        .expect("2-node cycle bounded");
    assert_eq!(hops_cycle, 2);

    sub.trails[2].linked_trail_id = u32::MAX;
    let (_rep_oob, hops_oob, _, _, _) = sub
        .traverse_pointer_chain_and_orchestrate(
            liminika_core::stigmergy::ChainTraverseParams {
                start_trail_idx: 2,
                max_hops: 8,
                entropy_h_tilde: 0.3,
            },
            &mut trail_out,
            None,
            None,
            None,
            None,
        )
        .expect("oob hop terminates");
    assert_eq!(hops_oob, 1);

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
        assert_eq!(out_a, out_c);
        assert_eq!(norm_a, norm_c);
        assert_eq!(trav_a, trav_c);
        assert_eq!(entry_a, entry_c);

        net_acyclic.record_chain_activation(usize::from(entry_a), 4);
        net_cycle.record_chain_activation(usize::from(entry_c), 4);
        assert_eq!(net_acyclic.node_cold[0].reinforce_count, 1);
        assert_eq!(net_cycle.node_cold[0].reinforce_count, 1);
        if hops >= 2 {
            assert_eq!(net_acyclic.node_cold[1].reinforce_count, 1);
            assert_eq!(net_cycle.node_cold[1].reinforce_count, 1);
        }

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
fn suite3_hostile_c_abi_v040_guards_and_direct_rust_parity() {
    // 1. Handshake & config struct_size fail-closed checks
    assert_eq!(lmnk_abi_version_hex(), 0x0000_0400);
    // SAFETY: Static C string literals.
    unsafe {
        assert!(!lmnk_abi_version_string().is_null());
        assert!(!lmnk_status_to_string(LMNK_ERR_INVALID_ARGUMENT).is_null());
    }

    let mut info = lmnk_abi_info_t::default();
    // SAFETY: Valid stack struct.
    unsafe {
        assert_eq!(lmnk_abi_info(ptr::null_mut()), LMNK_ERR_NULL_POINTER);
        info.struct_size = 0;
        assert_eq!(lmnk_abi_info(&mut info), LMNK_ERR_INVALID_ARGUMENT);
        info.struct_size = u32::MAX;
        assert_eq!(lmnk_abi_info(&mut info), LMNK_ERR_INVALID_ARGUMENT);
        info.struct_size = (std::mem::size_of::<lmnk_abi_info_t>() + 4) as u32;
        assert_eq!(lmnk_abi_info(&mut info), LMNK_ERR_INVALID_ARGUMENT);
        info = lmnk_abi_info_t::default();
        assert_eq!(lmnk_abi_info(&mut info), LMNK_OK);
    }
    assert_eq!(info.context_size, 5120);
    assert_eq!(info.context_align, 64);

    let mut cfg = lmnk_config_t::default();
    // SAFETY: Valid 64B-aligned stack struct.
    unsafe {
        assert_eq!(lmnk_config_init_default(&mut cfg), LMNK_OK);
        assert_eq!(lmnk_config_validate(&cfg), LMNK_OK);
        let mut bad_cfg = cfg;
        bad_cfg.struct_size = 0;
        assert_eq!(lmnk_config_validate(&bad_cfg), LMNK_ERR_INVALID_ARGUMENT);
        bad_cfg.struct_size = u32::MAX;
        assert_eq!(lmnk_config_validate(&bad_cfg), LMNK_ERR_INVALID_ARGUMENT);
        let mut out_ctx: LmnkContextHandle = ptr::null_mut();
        assert_eq!(
            lmnk_context_create(&bad_cfg, &mut out_ctx),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert!(out_ctx.is_null());
    }

    // 2. Context lifecycle: cross-mode misuse, double deinit, undersized & usize::MAX serialize/deserialize
    let mut backing = AlignedBuffer64::<u8, 5120>::new([0u8; 5120]);
    // SAFETY: Aligned 5120B caller buffer.
    unsafe {
        assert_eq!(
            lmnk_context_init_in_place(
                &cfg,
                backing.as_mut_slice().as_mut_ptr(),
                info.context_size as usize
            ),
            LMNK_OK
        );
    }
    let in_place_handle = backing
        .as_mut_slice()
        .as_mut_ptr()
        .cast::<liminika_core::abi::LmnkContextOpaque>();

    let mut small_buf = [0u8; 64];
    let mut sz = small_buf.len();
    let mut out_ctx: LmnkContextHandle = ptr::null_mut();
    let mut full_buf = [0u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut zero_len = 0usize;
    let mut huge_len = usize::MAX;
    // SAFETY: Exercising cross-mode, undersized, and overflow guards.
    unsafe {
        assert_eq!(
            lmnk_context_destroy(in_place_handle),
            LMNK_ERR_INVALID_STATE
        );
        assert_eq!(
            lmnk_context_serialize(in_place_handle, small_buf.as_mut_ptr(), &mut sz),
            LMNK_ERR_BUFFER_TOO_SMALL
        );
        assert_eq!(sz, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES);
        assert_eq!(
            lmnk_context_deserialize(small_buf.as_ptr(), 16, &mut out_ctx),
            LMNK_ERR_BUFFER_TOO_SMALL
        );
        assert!(out_ctx.is_null());
        assert_eq!(
            lmnk_context_serialize(in_place_handle, full_buf.as_mut_ptr(), &mut zero_len),
            LMNK_ERR_BUFFER_TOO_SMALL
        );
        assert_eq!(
            lmnk_context_serialize(in_place_handle, full_buf.as_mut_ptr(), &mut huge_len),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_context_deserialize(full_buf.as_ptr(), usize::MAX, &mut out_ctx),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(lmnk_context_reset(in_place_handle), LMNK_OK);
        assert_eq!(lmnk_context_deinit_in_place(in_place_handle), LMNK_OK);
        assert_eq!(
            lmnk_context_deinit_in_place(in_place_handle),
            LMNK_ERR_INVALID_STATE
        );
    }

    let mut heap_handle: LmnkContextHandle = ptr::null_mut();
    // SAFETY: Creates and destroys heap context with cross-mode guard verification.
    unsafe {
        assert_eq!(lmnk_context_create(&cfg, &mut heap_handle), LMNK_OK);
        assert_eq!(
            lmnk_context_deinit_in_place(heap_handle),
            LMNK_ERR_INVALID_STATE
        );
        assert_eq!(lmnk_context_destroy(heap_handle), LMNK_OK);
    }

    // 3. Hot-path envelope guards: struct_size, num_tokens != 1, NULL, misalignment, NaN, Inf, aliasing
    let mut mis_raw = [0u8; 256];
    let mis_f32_ptr = mis_raw[1..].as_mut_ptr().cast::<f32>();
    let mut good_q = AlignedBuffer32::<f32, 128>::new([1.0; 128]);
    let mut good_k = AlignedBuffer32::<f32, 128>::new([0.5; 128]);
    let phases_16 = AlignedBuffer16::<i8, 64>::new([12; 64]);
    let mut out_q7 = AlignedBuffer16::<lmnk_q7_t, 64>::new([0; 64]);

    let base_steer = lmnk_rope_steer_params_t {
        head_dim: 64,
        rotary_dim: 64,
        num_heads: 2,
        rope_layout: 0,
        max_rad: 0.2,
        query_tensor: good_q.as_mut_slice().as_mut_ptr(),
        phase_deltas: phases_16.as_slice().as_ptr(),
        ..Default::default()
    };
    // SAFETY: Intentionally hostile envelope arguments tested against fail-closed C-ABI guards.
    unsafe {
        assert_eq!(lmnk_rope_steer_q(ptr::null()), LMNK_ERR_NULL_POINTER);
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                struct_size: 0,
                ..base_steer
            }),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                struct_size: u32::MAX,
                ..base_steer
            }),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                num_tokens: 0,
                ..base_steer
            }),
            LMNK_ERR_NOT_IMPLEMENTED
        );
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                num_tokens: 2,
                ..base_steer
            }),
            LMNK_ERR_NOT_IMPLEMENTED
        );
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                flags: 1 << 31,
                ..base_steer
            }),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                query_tensor: ptr::null_mut(),
                ..base_steer
            }),
            LMNK_ERR_NULL_POINTER
        );
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                query_tensor: mis_f32_ptr,
                ..base_steer
            }),
            LMNK_ERR_ALIGNMENT
        );
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                max_rad: f32::NAN,
                ..base_steer
            }),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                max_rad: f32::INFINITY,
                ..base_steer
            }),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_rope_steer_q(&lmnk_rope_steer_params_t {
                flags: LMNK_ROPE_STEER_FLAG_UNCLAMPED,
                ..base_steer
            }),
            LMNK_OK
        );

        // Procrustes: read/read alias allowed, read/write alias rejected, num_tokens != 1 rejected
        let base_proc = lmnk_rope_procrustes_params_t {
            head_dim: 64,
            rotary_dim: 64,
            num_heads: 2,
            rope_layout: 1,
            q_before: good_q.as_slice().as_ptr(),
            q_after: good_q.as_slice().as_ptr(),
            q7_phase_out: out_q7.as_mut_slice().as_mut_ptr(),
            ..Default::default()
        };
        assert_eq!(lmnk_rope_procrustes_q7(&base_proc), LMNK_OK);
        assert_eq!(
            lmnk_rope_procrustes_q7(&lmnk_rope_procrustes_params_t {
                num_tokens: 4,
                ..base_proc
            }),
            LMNK_ERR_NOT_IMPLEMENTED
        );
        assert_eq!(
            lmnk_rope_procrustes_q7(&lmnk_rope_procrustes_params_t {
                q_after: good_k.as_slice().as_ptr(),
                q7_phase_out: good_k.as_mut_slice().as_mut_ptr().cast::<lmnk_q7_t>(),
                ..base_proc
            }),
            LMNK_ERR_INVALID_ARGUMENT
        );
    }

    // 4. Arena-indexed Rank1Descriptor overflow/OOB/aliasing + LoRA direct Rust parity
    let arena = [0.25f32; 64];
    let x_vec = AlignedBuffer32::<f32, 32>::new([1.0; 32]);
    let mut y_vec = AlignedBuffer32::<f32, 32>::new([0.0; 32]);
    let good_desc = lmnk_rank1_descriptor_t {
        layer_idx: 0,
        rank: 1,
        u_span: ArenaSpan::new(0, 32),
        v_span: ArenaSpan::new(32, 32),
        gain_offset: RANK1_GAIN_UNUSED,
        scale_factor: 1.0,
    };
    let overflow_desc = lmnk_rank1_descriptor_t {
        u_span: ArenaSpan::new(u32::MAX - 2, 32),
        ..good_desc
    };
    let oob_gain_desc = lmnk_rank1_descriptor_t {
        gain_offset: 60, // 60 + 32 > 64
        ..good_desc
    };

    assert_eq!(
        eval_rank1(
            &mut y_vec.as_aligned_slice_mut(),
            &x_vec.as_aligned_slice(),
            &arena,
            &overflow_desc,
        ),
        Err(LMNK_ERR_INVALID_ARGUMENT)
    );
    assert_eq!(
        eval_rank1_fused(
            &mut y_vec.as_aligned_slice_mut(),
            &x_vec.as_aligned_slice(),
            &arena,
            &oob_gain_desc,
        ),
        Err(LMNK_ERR_INVALID_ARGUMENT)
    );

    let r1_base = lmnk_rank1_eval_params_t {
        dim_in: 32,
        dim_out: 32,
        arena_len: arena.len() as u32,
        desc: &good_desc,
        arena: arena.as_ptr(),
        x_in: x_vec.as_slice().as_ptr(),
        y_inout: y_vec.as_mut_slice().as_mut_ptr(),
        ..Default::default()
    };
    // SAFETY: Testing Rank-1 C-ABI valid, fused, overflow, aliasing, and batch rejection.
    unsafe {
        assert_eq!(lmnk_rank1_eval(&r1_base), LMNK_OK);
        assert_eq!(
            lmnk_rank1_eval(&lmnk_rank1_eval_params_t {
                flags: LMNK_RANK1_FLAG_FUSED,
                ..r1_base
            }),
            LMNK_OK
        );
        assert_eq!(
            lmnk_rank1_eval(&lmnk_rank1_eval_params_t {
                desc: &overflow_desc,
                ..r1_base
            }),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_rank1_eval(&lmnk_rank1_eval_params_t {
                arena: y_vec.as_slice().as_ptr(),
                arena_len: 32,
                ..r1_base
            }),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_rank1_eval(&lmnk_rank1_eval_params_t {
                num_tokens: 3,
                ..r1_base
            }),
            LMNK_ERR_NOT_IMPLEMENTED
        );
    }

    // Direct Rust LoRA projection with arena packing
    const RANK: usize = 4;
    const DIM_IN: usize = 16;
    const DIM_OUT: usize = 64;
    let lora_a = AlignedBuffer32::<f32, { RANK * DIM_IN }>::new([0.25; RANK * DIM_IN]);
    let lora_b = AlignedBuffer32::<f32, { DIM_OUT * RANK }>::new([0.15; DIM_OUT * RANK]);
    let mut lora_arena = [0.0f32; DIM_OUT + DIM_IN];
    let mut r1_out = lmnk_rank1_descriptor_t::default();
    let mut lora_phases = AlignedBuffer16::<lmnk_q7_t, 8>::new([0; 8]);
    let packed_elems = project_lora(
        &lora_a.as_aligned_slice(),
        &lora_b.as_aligned_slice(),
        LoraDimensions {
            rank: RANK,
            dim_in: DIM_IN,
            dim_out: DIM_OUT,
        },
        &mut lora_arena,
        &mut r1_out,
        &mut lora_phases.as_aligned_slice_mut(),
    )
    .expect("project_lora succeeds");
    assert_eq!(packed_elems, DIM_OUT + DIM_IN);
    assert_eq!(r1_out.u_span, ArenaSpan::new(0, DIM_OUT as u32));
    assert_eq!(r1_out.v_span, ArenaSpan::new(DIM_OUT as u32, DIM_IN as u32));

    // 5. Gateway steer, Lacuna eval, Container write hostile checks
    let mut trail_out = lmnk_pointer_trail_t::default();
    let mut frame_out = GatewayFrame::default();
    let mut report_out = liminika_core::abi::lmnk_causal_report_t::default();
    let gw_base = lmnk_gateway_steer_params_t {
        head_dim: 64,
        rotary_dim: 64,
        num_heads: 2,
        rope_layout: 0,
        fact_energy: 1.5,
        persona_energy: 1.0,
        query_tensor: good_q.as_mut_slice().as_mut_ptr(),
        fact_phases: phases_16.as_slice().as_ptr(),
        persona_phases: phases_16.as_slice().as_ptr(),
        trail_inout: &mut trail_out,
        frame_out: &mut frame_out,
        report_out: &mut report_out,
        ..Default::default()
    };
    let lac_base = lmnk_lacuna_eval_params_t {
        self_ref_mask: 0xFF,
        deadband_q7: 16,
        other_density_q7: [32; 8],
        noise_tolerance_q7: [4; 8],
        ..Default::default()
    };
    let mut lac_part = LacunaPartition8::default();
    // SAFETY: Validating gateway, lacuna, and container write fail-closed guards.
    unsafe {
        assert_eq!(lmnk_gateway_steer(&gw_base), LMNK_OK);
        assert_eq!(
            lmnk_gateway_steer(&lmnk_gateway_steer_params_t {
                num_tokens: 2,
                ..gw_base
            }),
            LMNK_ERR_NOT_IMPLEMENTED
        );
        assert_eq!(
            lmnk_gateway_steer(&lmnk_gateway_steer_params_t {
                fact_energy: f32::NAN,
                ..gw_base
            }),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_lacuna_partition_eval(&lac_base, &mut lac_part),
            LMNK_OK
        );
        assert_eq!(
            lmnk_lacuna_partition_eval(
                &lmnk_lacuna_eval_params_t {
                    struct_size: 0,
                    ..lac_base
                },
                &mut lac_part
            ),
            LMNK_ERR_INVALID_ARGUMENT
        );

        let mut written = 0usize;
        let bad_sec = lmnk_section_input_t {
            struct_size: 0,
            kind: LMNK_SECTION_CONTENT_HASH,
            payload_len: 4,
            payload: [1u8; 4].as_ptr(),
            ..Default::default()
        };
        let bad_wp = lmnk_container_write_params_t {
            section_count: 1,
            sections: &bad_sec,
            written_bytes_out: &mut written,
            ..Default::default()
        };
        assert_eq!(lmnk_container_write(&bad_wp), LMNK_ERR_INVALID_ARGUMENT);
    }

    // 6. Direct Rust parity for former v0.3 kernel wrappers (zero coverage loss)
    let packed = TaggedPtr::pack(0x0000_1234_5678, 0xA5, -17);
    assert_eq!(packed.unpack_addr(), 0x0000_1234_5678);
    assert_eq!(packed.unpack_tag(), 0xA5);
    assert_eq!(packed.unpack_phase(), -17);
    let act = TaggedPtr::pack_action(0x0000_DEAD_BEE0, 3, ACTION_MODE_PHASE_STEER, 12);
    let evolved = act.evolve_mode_stigmergically(0.9, 0.6, false, 2, 0.1, 0.8);
    assert_eq!(evolved.target_attractor_slot(), 2);

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
    dispatch_single_pass(&mut good_q.as_aligned_slice_mut(), &subhead_cfg, 64)
        .expect("subhead dispatch");

    let mut ent = EntropyController::new(EntropyState {
        bifurcation_threshold: 1.5,
        singularity_eps: 1e-12,
        sample_temperature: 1.0,
        ..Default::default()
    });
    let mut ent_out = EntropyState::default();
    assert!(ent.eval_stateful(0.5, &mut ent_out).is_ok());

    let mut ten = DistortionTension::new(false);
    ten.step_hot_path_branchless(16, 32, 24, 8, false);
    assert!(
        eval_barrier_pitchfork(0.4, 0.5, 1.0, 1.0, 64)
            .expect("barrier pitchfork")
            .is_finite()
    );

    let mask = PagedBitmask::default();
    assert_eq!(mask.bits, [0u64; 4]);
    let _ = LedgerSlot::default();
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

    let mut priv_snap = snap.clone();
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
    assert_eq!(dst.system_spine().pinned_compute_mask & 0x0E, 0x0E);
    assert_eq!(dst.slot_pinned_mask() & 0x05, 0x05);

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
        assert_eq!(dst.step_counter(), 999);
        assert!((dst.substrate().last_entropy_h_tilde - 0.33).abs() < 1e-6);
        assert_eq!(dst.config().raw.head_dim, expected_head_dim);
    }
}

#[test]
fn suite5_uaf_free_zeroize_volatile_all_5120_bytes_proof() {
    let mut cfg = lmnk_config_t::default();
    let mut info = lmnk_abi_info_t::default();
    // SAFETY: Stack structs are valid and aligned.
    unsafe {
        assert_eq!(lmnk_abi_info(&mut info), LMNK_OK);
        assert_eq!(lmnk_config_init_default(&mut cfg), LMNK_OK);
    }
    let req = info.context_size as usize;
    assert_eq!(req, 5120);
    let mut backing = AlignedBuffer64::<u8, 5120>::new([0xAA; 5120]);
    let mem_ptr = backing.as_mut_slice().as_mut_ptr();

    // SAFETY: Caller-owned 64B-aligned 5,120B buffer initialized and scrubbed via C-ABI.
    unsafe {
        assert_eq!(lmnk_context_init_in_place(&cfg, mem_ptr, req), LMNK_OK);
        let ctx_handle = mem_ptr.cast::<liminika_core::abi::LmnkContextOpaque>();
        let turn = lmnk_turn_begin_params_t {
            flags: LMNK_TURN_FLAG_GUARD_ACTIVE,
            ..Default::default()
        };
        assert_eq!(lmnk_context_begin_turn(ctx_handle, &turn), LMNK_OK);
        let step_in = lmnk_decode_step_params_t::default();
        let mut step_out = lmnk_decode_step_report_t::default();
        assert_eq!(
            lmnk_context_step_decode(ctx_handle, &step_in, &mut step_out),
            LMNK_OK
        );
        assert_eq!(lmnk_context_deinit_in_place(ctx_handle), LMNK_OK);
    }
    assert!(
        backing.as_slice().iter().all(|&b| b == 0x00),
        "all 5,120 bytes must equal 0x00 after lmnk_context_deinit_in_place"
    );

    let mut ctx = RuntimeContext::new_boxed(Config::new()).expect("boxed ctx");
    ctx.substrate_mut().step_counter = 12345;
    ctx.substrate_mut().slot_weights = [9.0; 8];
    ctx.core_layers_mut().ego = LiminalEgoState::from_preset(LiminalEgoPreset::High);
    ctx.apply_host_spine_pin_floor(0xFFFF);
    ctx.zeroize_in_place();
    assert!(ctx.core_layers().as_bytes().iter().all(|&b| b == 0x00));
    assert!(ctx.system_spine().as_bytes().iter().all(|&b| b == 0x00));
    assert!(ctx.substrate().as_bytes().iter().all(|&b| b == 0x00));
    assert_eq!(ctx.compute_pool().active_mask, 0);
    assert_eq!(ctx.ego_differential_net().active_node_mask, 0);
    assert_eq!(ctx.substrate().step_counter, 0);
    assert_eq!(ctx.substrate().slot_weights, [0.0; 8]);
    assert_eq!(ctx.telemetry_snapshot().spine.total_steps, 0);
}

#[test]
fn suite6_spine_abi_adversarial_alignment_overlap_and_pin_floor_guards() {
    let mut cfg = lmnk_config_t::default();
    // SAFETY: Stack config is 64B aligned and writable.
    assert_eq!(unsafe { lmnk_config_init_default(&mut cfg) }, LMNK_OK);

    let mut backing = AlignedBuffer64::<u8, 5248>::new([0u8; 5248]);
    let base_ptr = backing.as_mut_slice().as_mut_ptr();
    let req_size = 5120usize;

    // SAFETY: Exercising hostile NULL, misaligned, struct_size, and overlapping pointers.
    unsafe {
        assert_eq!(
            lmnk_context_init_in_place(&cfg, base_ptr, req_size),
            LMNK_OK
        );
        let ctx_handle = base_ptr.cast::<liminika_core::abi::LmnkContextOpaque>();
        let misalign_ctx = base_ptr
            .add(16)
            .cast::<liminika_core::abi::LmnkContextOpaque>();

        let pol_spine = lmnk_runtime_policy_t {
            flags: LMNK_POLICY_FLAG_SPINE,
            baseline_pinned_compute_mask: 0x00A0,
            guard_hold_tokens: 4,
            max_observe_stride: 0,
            pre_bypass_enabled: 0,
            ..Default::default()
        };
        let mut state_out = lmnk_runtime_state_view_t::default();

        assert_eq!(
            lmnk_context_configure_policy(ptr::null_mut(), &pol_spine),
            LMNK_ERR_NULL_POINTER
        );
        assert_eq!(
            lmnk_context_configure_policy(misalign_ctx, &pol_spine),
            LMNK_ERR_ALIGNMENT
        );
        assert_eq!(
            lmnk_context_configure_policy(
                ctx_handle,
                &lmnk_runtime_policy_t {
                    struct_size: 0,
                    ..pol_spine
                }
            ),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_context_configure_policy(
                ctx_handle,
                &lmnk_runtime_policy_t {
                    flags: 1 << 30,
                    ..pol_spine
                }
            ),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            lmnk_context_get_state(ptr::null_mut(), &mut state_out),
            LMNK_ERR_NULL_POINTER
        );
        assert_eq!(
            lmnk_context_get_state(ctx_handle, ptr::null_mut()),
            LMNK_ERR_NULL_POINTER
        );
        assert_eq!(
            lmnk_context_get_state(misalign_ctx, &mut state_out),
            LMNK_ERR_ALIGNMENT
        );
        let mut raw_buf = AlignedBuffer64::<u8, 128>::new([0u8; 128]);
        let misalign_state = raw_buf.as_mut_slice().as_mut_ptr().add(1);
        assert_eq!(
            lmnk_context_get_state(ctx_handle, misalign_state.cast()),
            LMNK_ERR_ALIGNMENT
        );

        // Overlapping `state_out` inside the 5,120 B `context` buffer must fail.
        let overlap_state_out = base_ptr.add(384).cast::<lmnk_runtime_state_view_t>();
        assert_eq!(
            lmnk_context_get_state(ctx_handle, overlap_state_out),
            LMNK_ERR_INVALID_ARGUMENT
        );

        // Monotonic non-strippable host pin floor
        assert_eq!(
            lmnk_context_configure_policy(ctx_handle, &pol_spine),
            LMNK_OK
        );
        assert_eq!(lmnk_context_get_state(ctx_handle, &mut state_out), LMNK_OK);
        assert_eq!(
            state_out.baseline_pinned_compute_mask,
            0x00A0 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
        assert_eq!(
            state_out.pinned_compute_mask,
            0x00A0 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
        assert_eq!(state_out.guard_hold_tokens, 4);
        assert_eq!(state_out.max_observe_stride, 1);

        let pol_keep = lmnk_runtime_policy_t {
            flags: LMNK_POLICY_FLAG_SPINE | LMNK_POLICY_FLAG_COMPUTE_POOL,
            baseline_pinned_compute_mask: 0,
            active_compute_mask: 0x00FF,
            repulsion_compute_mask: u64::MAX,
            reversible_compute_mask: u64::MAX,
            baseline_pressure_q15: 0,
            guard_hold_tokens: 2,
            max_observe_stride: 6,
            pre_bypass_enabled: 1,
            ..Default::default()
        };
        assert_eq!(
            lmnk_context_configure_policy(ctx_handle, &pol_keep),
            LMNK_OK
        );
        let turn_plain = lmnk_turn_begin_params_t::default();
        assert_eq!(lmnk_context_begin_turn(ctx_handle, &turn_plain), LMNK_OK);
        assert_eq!(lmnk_context_get_state(ctx_handle, &mut state_out), LMNK_OK);
        assert_eq!(
            state_out.baseline_pinned_compute_mask,
            0x00A0 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
        assert_eq!(state_out.guard_hold_tokens, 2);
        assert_eq!(state_out.max_observe_stride, 6);

        let step_in = lmnk_decode_step_params_t {
            decode_input: lmnk_spine_decode_input_t {
                flags: SPINE_DECODE_FLAG_REWARD_OVERRIDE,
                exit_reward_override_q7: 250,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut step_rep = lmnk_decode_step_report_t::default();
        assert_eq!(
            lmnk_context_step_decode(ctx_handle, &step_in, &mut step_rep),
            LMNK_OK
        );
        assert_eq!(
            lmnk_context_step_decode(
                ctx_handle,
                &lmnk_decode_step_params_t {
                    struct_size: 0,
                    ..step_in
                },
                &mut step_rep
            ),
            LMNK_ERR_INVALID_ARGUMENT
        );
        assert_eq!(lmnk_context_deinit_in_place(ctx_handle), LMNK_OK);
    }

    // Commutativity check
    let mut ctx_ab = RuntimeContext::new(Config::new()).expect("ctx_ab");
    let mut ctx_ba = RuntimeContext::new(Config::new()).expect("ctx_ba");
    ctx_ab.configure_compute_pool(0x00FF, 0x00FF_FFFF, 0xFFFF_FFFF, 50_000);
    ctx_ab.configure_spine(0x00A0, 3, 4, true);
    ctx_ba.configure_spine(0x00A0, 3, 4, true);
    ctx_ba.configure_compute_pool(0x00FF, 0x00FF_FFFF, 0xFFFF_FFFF, 50_000);
    assert_eq!(*ctx_ab.system_spine(), *ctx_ba.system_spine());
    assert_eq!(*ctx_ab.compute_pool(), *ctx_ba.compute_pool());
    assert_eq!(ctx_ab.compute_pool().baseline_pressure_q15, 32_767);
    ctx_ab.configure_compute_pool(0x00FF, 0, 0x00FF, 0);
    assert_eq!(ctx_ab.compute_pool().baseline_pressure_q15, 32_767);
}

#[test]
fn suite6_abi_major_minor_and_patch_compatibility_policy() {
    use liminika_core::liminal::{
        LMNK_ABI_COMPAT_MASK, LMNK_ABI_COMPAT_VERSION_HEX, LMNK_ABI_VERSION_HEX,
        LMNK_ERR_VERSION_MISMATCH, is_compatible_abi_version, validate_abi_version,
    };
    use liminika_core::runtime::CORE_TOPOLOGY;

    assert_eq!(LMNK_ABI_COMPAT_MASK, 0xFFFF_FF00);
    assert_eq!(
        LMNK_ABI_COMPAT_VERSION_HEX,
        LMNK_ABI_VERSION_HEX & LMNK_ABI_COMPAT_MASK
    );
    assert_eq!(CORE_TOPOLOGY.version, LMNK_ABI_COMPAT_VERSION_HEX);

    // 1. Same MAJOR.MINOR and file_patch == runtime_patch -> accepted.
    assert!(is_compatible_abi_version(0x0000_0400, 0x0000_0400));
    assert_eq!(validate_abi_version(0x0000_0400), Ok(()));

    // 2. Same MAJOR.MINOR and file_patch < runtime_patch -> accepted (patch backward compat).
    assert!(is_compatible_abi_version(0x0000_0400, 0x0000_0401));
    assert!(is_compatible_abi_version(0x0000_0402, 0x0000_0405));

    // 3. Same MAJOR.MINOR and file_patch > runtime_patch -> rejected.
    assert!(!is_compatible_abi_version(0x0000_0401, 0x0000_0400));
    assert_eq!(
        validate_abi_version(0x0000_0401),
        Err(LMNK_ERR_VERSION_MISMATCH)
    );

    // 4. Different MAJOR.MINOR or zero -> rejected.
    for bad in [
        0x0000_0000u32,
        0x0000_0300,
        0x0000_0500,
        0x0001_0400,
        0xFFFF_FFFF,
    ] {
        assert!(!is_compatible_abi_version(bad, 0x0000_0400));
        assert!(!is_compatible_abi_version(bad, 0x0000_0401));
        assert_eq!(validate_abi_version(bad), Err(LMNK_ERR_VERSION_MISMATCH));
    }

    let anchor = PhaseAnchorRecord::new_empty(77, ATTRACTOR_SLOT_KNOWLEDGE, 0);
    assert!(anchor.verify_integrity());
    assert_eq!(anchor.version, LMNK_ABI_VERSION_HEX);

    let mut bad_patch_anchor = anchor;
    bad_patch_anchor.version = 0x0000_0401;
    bad_patch_anchor.checksum_crc32 = bad_patch_anchor.compute_payload_crc32();
    assert!(!bad_patch_anchor.verify_integrity());

    let mut container = vec![0u8; 4096];
    let written = LmnkContainerView::write_standard_sections_to_bytes(
        liminika_core::runtime::StandardSectionsInput {
            container_flags: LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            anchors: &[anchor],
            trails: &[],
            bloom: None,
            ledger: None,
            provenance_reports: &[],
            timestamp_epoch_sec: 1_774_900_000,
        },
        &mut container,
    )
    .expect("write container");
    container.truncate(written);
    let view = LmnkContainerView::parse(&container).expect("valid v0.4.0 container");
    assert_eq!(view.header().version, LMNK_ABI_VERSION_HEX);

    let ctx = RuntimeContext::new(Config::new()).expect("ctx");
    let mut snapshot = vec![0u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut snap_len = snapshot.len();
    ctx.serialize_to_bytes(&mut snapshot, &mut snap_len)
        .expect("serialize snapshot");
    let snap_view = LmnkContainerView::parse(&snapshot).expect("valid v0.4.0 snapshot");
    assert_eq!(snap_view.header().version, LMNK_ABI_VERSION_HEX);

    let patch_version_and_restamp = |base: &[u8], version: u32| -> Vec<u8> {
        let mut buf = base.to_vec();
        buf[4..8].copy_from_slice(&version.to_le_bytes());
        let crc = compute_crc32_ieee(&buf[0..60]);
        buf[60..64].copy_from_slice(&crc.to_le_bytes());
        buf
    };

    for rejected_ver in [0x0000_0401u32, 0x0000_0300, 0x0000_0500, 0x0000_0000] {
        let bad_container = patch_version_and_restamp(&container, rejected_ver);
        assert_eq!(
            LmnkContainerView::parse(&bad_container),
            Err(LMNK_ERR_VERSION_MISMATCH),
            "container version {rejected_ver:#010X} must fail with VERSION_MISMATCH"
        );

        let bad_snapshot = patch_version_and_restamp(&snapshot, rejected_ver);
        let mut restore_ctx = RuntimeContext::new(Config::new()).expect("restore_ctx");
        assert_eq!(
            restore_ctx.deserialize_from_bytes(&bad_snapshot),
            Err(LMNK_ERR_VERSION_MISMATCH),
            "snapshot version {rejected_ver:#010X} must fail with VERSION_MISMATCH"
        );
    }
}
