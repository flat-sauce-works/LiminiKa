// name: src/adapter/src/host/equilibrium/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit tests for host equilibrium state and persistence.

use super::*;
use liminika_core::runtime::{Config, RuntimeContext};
use liminika_core::stigmergy::ATTRACTOR_SLOT_KNOWLEDGE;

fn make_record(anchor_id: u32, seed: u32, mass: f32) -> PhaseAnchorRecord {
    let mut rec = PhaseAnchorRecord::new_empty(anchor_id, ATTRACTOR_SLOT_KNOWLEDGE, 0);
    rec.base_energy = mass;
    rec.phase_gravity_mass = mass;
    rec.resultant_r_bar = 0.95;
    rec.von_mises_kappa = 10.0;
    rec.sample_count = 2;
    let mut state = (u64::from(seed) << 32) ^ u64::from(anchor_id).wrapping_mul(0x9E37_79B9);
    for dst in &mut rec.phase_profile {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let val = ((z >> 56) as i16) - 128;
        *dst = val.clamp(-127, 127) as i8;
    }
    rec.stamp_crc32();
    rec
}

fn make_snapshot_bytes(
    pin_mask: u64,
    step_counter: u64,
) -> [u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES] {
    let cfg = Config::new().with_head_dim(64).with_num_heads(8);
    let mut ctx = RuntimeContext::new(cfg).expect("runtime ctx");
    ctx.apply_host_spine_pin_floor(pin_mask);
    ctx.substrate_mut().step_counter = step_counter;
    let mut buf = [0_u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut req = 0_usize;
    ctx.serialize_to_bytes(&mut buf, &mut req)
        .expect("serialize snapshot");
    assert_eq!(req, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES);
    buf
}

#[test]
fn hundred_generations_prevent_thermal_death_and_evaporate_stale_memories() {
    let cfg = EquilibriumConfig::default();
    let mut state = EquilibriumState::new(cfg);

    // 2 Spine-pinned anchors (never reinforced again) + 2 reinforced core anchors + 4 stale anchors.
    let pinned_a = make_record(1, 101, 1.0);
    let pinned_b = make_record(2, 202, 1.0);
    let core_a = make_record(10, 1010, 1.2);
    let core_b = make_record(11, 1111, 1.2);
    state
        .insert_anchor(pinned_a, Lacuna::SelfRef, true)
        .expect("pin a");
    state
        .insert_anchor(pinned_b, Lacuna::SelfRef, true)
        .expect("pin b");
    state
        .insert_anchor(core_a, Lacuna::SelfRef, false)
        .expect("core a");
    state
        .insert_anchor(core_b, Lacuna::SelfRef, false)
        .expect("core b");
    for stale_id in 50..54_u32 {
        state
            .insert_anchor(
                make_record(stale_id, stale_id * 19, 1.0),
                Lacuna::SelfRef,
                false,
            )
            .expect("stale");
    }

    let mut total_evaporated = 0_usize;
    for generation in 1..=100_u64 {
        let mut pulse = EquilibriumState::new(cfg);
        pulse
            .insert_anchor(core_a, Lacuna::SelfRef, false)
            .expect("pulse a");
        pulse
            .insert_anchor(core_b, Lacuna::SelfRef, false)
            .expect("pulse b");
        let report = state.join(&pulse).expect("join");
        total_evaporated += report.evaporated_anchors;
        assert_eq!(report.generation, generation);

        // Round-trip through .lmnkc every generation to prove flat persistence parity.
        let bytes = state
            .to_lmnkc_bytes(1_774_900_000 + generation)
            .expect("serialize");
        state = EquilibriumState::from_lmnkc_bytes(&bytes, cfg).expect("restore");
    }

    // Non-thermal-death: pinned & reinforced anchors survive 100 generations with finite mass.
    for id in [1_u32, 2, 10, 11] {
        let entry = state
            .pager()
            .get_entry(id)
            .expect("core/pinned must survive");
        assert!(
            state.pager().is_resident(id),
            "anchor {id} must be resident"
        );
        assert!(entry.record.phase_gravity_mass.is_finite());
        assert!(entry.record.phase_gravity_mass >= MIN_PINNED_GRAVITY_FLOOR);
        assert!(entry.record.phase_gravity_mass <= cfg.max_gravity_mass + 1e-6);
    }
    assert!(state.pager().get_entry(1).unwrap().pinned);
    assert!(state.pager().get_entry(2).unwrap().pinned);

    // Healthy forgetting: unreinforced stale anchors evaporated cleanly.
    assert!(total_evaporated >= 4);
    for stale_id in 50..54_u32 {
        assert!(
            state.pager().get_entry(stale_id).is_none(),
            "stale unpinned anchor {stale_id} should have evaporated"
        );
    }
}

#[test]
fn hundred_generations_prevent_lock_in_and_keep_size_bounded() {
    let cfg = EquilibriumConfig {
        pager: AnchorPagerConfig {
            working_set_capacity: 16,
            ..AnchorPagerConfig::default()
        },
        max_backing_anchors: 32,
        ..EquilibriumConfig::default()
    };
    let mut state = EquilibriumState::new(cfg);

    // 2 pinned + 30 saturated unpinned anchors.
    state
        .insert_anchor(make_record(1, 11, 2.0), Lacuna::SelfRef, true)
        .expect("pin 1");
    state
        .insert_anchor(make_record(2, 22, 2.0), Lacuna::SelfRef, true)
        .expect("pin 2");
    for id in 10..40_u32 {
        state
            .insert_anchor(make_record(id, id * 37 + 3, 2.0), Lacuna::SelfRef, false)
            .expect("seed");
    }

    let mut max_lmnkc_bytes = 0_usize;
    for generation in 1..=100_u64 {
        let mut wave = EquilibriumState::new(cfg);
        // Reinforce existing 16 unpinned anchors to saturate the equilibrium.
        for id in 10..26_u32 {
            wave.insert_anchor(make_record(id, id * 37 + 3, 1.8), Lacuna::SelfRef, false)
                .expect("wave");
        }
        // Inject rotating transient anchors to stress backing-bank bounds.
        let transient_id = 1000 + (generation as u32);
        wave.insert_anchor(
            make_record(transient_id, transient_id * 13 + 7, 1.0),
            Lacuna::SelfRef,
            false,
        )
        .expect("transient");

        let report = state.join(&wave).expect("join");
        assert!(report.backing_len <= cfg.max_backing_anchors);
        assert!(report.resident_len <= cfg.pager.working_set_capacity);
        assert!(state.pager().is_resident(1));
        assert!(state.pager().is_resident(2));

        let bytes = state
            .to_lmnkc_bytes(1_774_900_000 + generation)
            .expect("to_lmnkc");
        max_lmnkc_bytes = max_lmnkc_bytes.max(bytes.len());
        for entry in state.pager().backing_entries() {
            assert!(entry.record.base_energy.is_finite());
            assert!(entry.record.phase_gravity_mass.is_finite());
            assert!(entry.record.resultant_r_bar.is_finite());
            assert!(entry.record.von_mises_kappa.is_finite());
            assert!(entry.excitation.is_finite());
        }
    }

    // Verify bounded container size (32 anchors * 128 B + metadata + headers < 8 KiB).
    assert!(
        max_lmnkc_bytes < 8192,
        "unexpected lmnkc bloat: {max_lmnkc_bytes} B"
    );

    // Plasticity / non-lock-in check: inject a brand-new crystallized anchor at generation 101.
    let novel = make_record(9999, 0xCAFE_1234, 1.25);
    let report = state
        .ingest_crystallized_anchors(&[novel])
        .expect("ingest novel");
    assert_eq!(report.inserted_anchors, 1);
    assert!(
        state.pager().is_resident(9999),
        "novel crystallized anchor must enter working set without lock-in after 100 gens"
    );
    assert!(state.pager().is_resident(1));
    assert!(state.pager().is_resident(2));
}

#[test]
fn topology_mismatch_rejected_and_exterior_never_promoted() {
    let cfg = EquilibriumConfig::default();
    let mut a = EquilibriumState::with_topology_crc32(cfg, CORE_TOPOLOGY.topology_crc32);
    let b = EquilibriumState::with_topology_crc32(cfg, CORE_TOPOLOGY.topology_crc32 ^ 0xDEAD_BEEF);
    let err = a.join(&b).expect_err("topology mismatch must be rejected");
    assert_eq!(
        err,
        EquilibriumError::TopologyMismatch {
            expected: CORE_TOPOLOGY.topology_crc32,
            actual: CORE_TOPOLOGY.topology_crc32 ^ 0xDEAD_BEEF,
        }
    );

    // Unsettled Lacuna::Exterior stays isolated across join and .lmnkc roundtrip.
    let mut inc = EquilibriumState::new(cfg);
    let ext = make_record(777, 7777, 3.0);
    inc.insert_anchor(ext, Lacuna::Exterior, false)
        .expect("insert ext");
    let report = a.join(&inc).expect("join same topology");
    assert_eq!(report.exterior_isolated, 1);
    assert!(!a.pager().is_resident(777));
    assert_eq!(
        a.pager().get_entry(777).unwrap().lacuna_state,
        Lacuna::Exterior
    );

    let bytes = a.to_lmnkc_bytes(1_774_900_100).expect("serialize");
    let restored = EquilibriumState::from_lmnkc_bytes(&bytes, cfg).expect("restore");
    assert!(!restored.pager().is_resident(777));
    assert_eq!(
        restored.pager().get_entry(777).unwrap().lacuna_state,
        Lacuna::Exterior
    );
}

#[test]
fn runtime_snapshot_roundtrip_monotonic_pins_and_legacy_compat() {
    let cfg = EquilibriumConfig::default();
    let snap_a = make_snapshot_bytes(0b0011, 10);
    let snap_b = make_snapshot_bytes(0b1100, 25);

    let mut eq_a = EquilibriumState::new(cfg);
    eq_a.insert_anchor(make_record(1, 10, 1.5), Lacuna::SelfRef, true)
        .expect("pin 1");
    eq_a.update_runtime_snapshot_from_container(&snap_a)
        .expect("snap a");

    let mut eq_b = EquilibriumState::new(cfg);
    eq_b.insert_anchor(make_record(2, 20, 1.5), Lacuna::SelfRef, false)
        .expect("anchor 2");
    eq_b.update_runtime_snapshot_from_container(&snap_b)
        .expect("snap b");

    let report = eq_a.join(&eq_b).expect("join snapshots");
    assert!(report.snapshot_updated);

    let merged_env = eq_a.snapshot_envelope_state().expect("merged envelope");
    assert_eq!(merged_env.baseline_pinned_compute_mask & 0b1111, 0b1111);

    // .lmnkc roundtrip preserves both RuntimeSnapshot section and host metadata.
    let lmnkc = eq_a.to_lmnkc_bytes(1_774_900_500).expect("to_lmnkc");
    let view = LmnkContainerView::parse(&lmnkc).expect("parse view");
    assert!(view.section_bytes(LMNK_SECTION_RUNTIME_SNAPSHOT).is_some());
    assert!(
        view.section_bytes(HOST_EQUILIBRIUM_META_SECTION_KIND)
            .is_some()
    );

    let restored = EquilibriumState::from_lmnkc_bytes(&lmnkc, cfg).expect("from_lmnkc");
    assert_eq!(restored.generation(), eq_a.generation());
    assert!(restored.pager().get_entry(1).unwrap().pinned);
    let restored_env = restored
        .snapshot_envelope_state()
        .expect("restored envelope");
    assert_eq!(restored_env.baseline_pinned_compute_mask & 0b1111, 0b1111);

    // Restore into a live RuntimeContext with a stronger host pin floor (bit 5).
    let live_cfg = Config::new().with_head_dim(64).with_num_heads(8);
    let mut live_ctx = RuntimeContext::new(live_cfg).expect("live ctx");
    live_ctx.apply_host_spine_pin_floor(1_u64 << 5);
    let snap_576 = restored
        .runtime_snapshot_container_bytes()
        .expect("576B snapshot");
    live_ctx
        .deserialize_from_bytes(&snap_576)
        .expect("deserialize into live ctx");
    assert_eq!(
        live_ctx.envelope().baseline_pinned_compute_mask & ((1 << 5) | 0b1111),
        (1 << 5) | 0b1111
    );

    // Legacy .lmnkc without RuntimeSnapshot or EquilibriumMeta loads cleanly.
    let legacy_pager = AnchorWorkingSetPager::from_records(
        &[make_record(42, 420, 1.5)],
        AnchorPagerConfig::default(),
    );
    let legacy_bytes = legacy_pager
        .to_container_bytes(1_774_900_000)
        .expect("legacy");
    let legacy_eq = EquilibriumState::from_lmnkc_bytes(&legacy_bytes, cfg).expect("load legacy");
    assert_eq!(legacy_eq.generation(), 0);
    assert!(legacy_eq.runtime_snapshot_payload().is_none());
    assert!(legacy_eq.pager().is_resident(42));
}

#[test]
fn circular_merge_preserves_wraparound_and_avoids_zero_collapse() {
    let a = [126_i8; MAX_ATTENTION_HEADS];
    let b = [-126_i8; MAX_ATTENTION_HEADS];
    let (merged, r_bar) = circular_merge_phase_profiles_q7(&a, 0.5, &b, 0.5);
    // On S^1, +126 and -126 are near +/-pi (+/-127), never 0!
    assert!(
        merged[0].unsigned_abs() >= 125,
        "wraparound collapsed to {}",
        merged[0]
    );
    assert!(r_bar > 0.95);

    // Exact antipodal cancellation retains stronger weight's phase rather than collapsing to 0.
    let pos = [64_i8; MAX_ATTENTION_HEADS];
    let neg = [-64_i8; MAX_ATTENTION_HEADS];
    let (anti, _) = circular_merge_phase_profiles_q7(&pos, 0.6, &neg, 0.4);
    assert_eq!(anti[0], 64);
}

#[test]
fn signed_equilibrium_lmnkc_enforces_policy_and_rejects_downgrade_or_tampering() {
    let cfg = EquilibriumConfig::default();
    let mut eq = EquilibriumState::new(cfg);
    eq.insert_anchor(make_record(7, 77, 1.5), Lacuna::SelfRef, true)
        .expect("insert");
    let kp = SigningKeyPair::from_seed_bytes([0x5A_u8; 32]);
    let wrong_kp = SigningKeyPair::from_seed_bytes([0xA5_u8; 32]);
    let pk = kp.verifying_key();

    let unsigned = eq.to_lmnkc_bytes(1_774_901_000).expect("unsigned");
    assert!(EquilibriumState::from_lmnkc_bytes(&unsigned, cfg).is_ok());
    assert!(
        EquilibriumState::from_lmnkc_bytes_with_policy(
            &unsigned,
            cfg,
            &IntegrityPolicy::RequireSignature(pk),
        )
        .is_err()
    );

    let signed = eq
        .to_signed_lmnkc_bytes(1_774_901_000, &kp)
        .expect("signed");
    let restored = EquilibriumState::from_lmnkc_bytes_with_policy(
        &signed,
        cfg,
        &IntegrityPolicy::RequireSignature(pk),
    )
    .expect("verified restore");
    assert!(restored.pager().get_entry(7).unwrap().pinned);

    assert!(
        EquilibriumState::from_lmnkc_bytes_with_policy(
            &signed,
            cfg,
            &IntegrityPolicy::RequireSignature(wrong_kp.verifying_key()),
        )
        .is_err()
    );

    assert!(
        eq.save_lmnkc_to_path(Path::new("out/CON.lmnkc"), 1_774_901_000)
            .is_err()
    );
}
