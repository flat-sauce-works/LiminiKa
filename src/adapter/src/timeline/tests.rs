// name: src/adapter/src/timeline/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit tests for [`LmnkTimeline`] GOP cutting and `.lmnkc`/`.lmnkp` persistence.

use super::*;

struct DummyEncoder {
    layers: usize,
    rot_len: usize,
}

impl SubtitleActivationEncoder for DummyEncoder {
    fn encode_subtitle_activations(
        &mut self,
        record: &SourceTextRecord,
    ) -> Result<Vec<Vec<f32>>, String> {
        let seed = record.text.bytes().fold(0x1337_u64, |acc, b| {
            acc.wrapping_mul(131).wrapping_add(u64::from(b))
        });
        Ok((0..self.layers)
            .map(|l| {
                (0..self.rot_len)
                    .map(|d| (((seed ^ ((l as u64) << 6) ^ (d as u64)) % 100) as f32) * 0.01 - 0.5)
                    .collect()
            })
            .collect())
    }
}

#[test]
fn adaptive_gop_cut_detector_delegates_to_spine_transition_controller() {
    let detector = AdaptiveGopCutDetector::default();
    let calm = FrameTransitionMetrics {
        phase_l1_delta: 0.10,
        lacuna_state: LacunaBoundaryState::InsideBasin,
        frames_since_keyframe: 2,
        new_cluster_spawned: false,
    };
    assert!(!detector.should_emit_keyframe(&calm));

    assert!(detector.should_emit_keyframe(&FrameTransitionMetrics {
        lacuna_state: LacunaBoundaryState::LiminalExterior,
        ..calm
    }));
    assert!(detector.should_emit_keyframe(&FrameTransitionMetrics {
        new_cluster_spawned: true,
        ..calm
    }));
    assert!(detector.should_emit_keyframe(&FrameTransitionMetrics {
        phase_l1_delta: 0.35,
        ..calm
    }));
    assert!(!detector.should_emit_keyframe(&FrameTransitionMetrics {
        phase_l1_delta: 0.349,
        ..calm
    }));
    assert!(!detector.should_emit_keyframe(&FrameTransitionMetrics {
        phase_l1_delta: f32::NAN,
        ..calm
    }));

    // Large max_gop_len (> 255) must not wrap or prematurely expire below max_gop_len.
    let long_gop = AdaptiveGopCutDetector {
        max_gop_len: 300,
        phase_delta_cut_thresh: 0.35,
    };
    assert!(!long_gop.should_emit_keyframe(&FrameTransitionMetrics {
        frames_since_keyframe: 299,
        ..calm
    }));
    assert!(long_gop.should_emit_keyframe(&FrameTransitionMetrics {
        frames_since_keyframe: 300,
        ..calm
    }));

    // Zero cut threshold always triggers keyframe on non-negative deltas.
    let zero_thresh = AdaptiveGopCutDetector {
        max_gop_len: 16,
        phase_delta_cut_thresh: 0.0,
    };
    assert!(zero_thresh.should_emit_keyframe(&FrameTransitionMetrics {
        phase_l1_delta: 0.0,
        ..calm
    }));
}

#[test]
fn unified_lmnkc_and_lmnkp_roundtrip_rebake_and_topology_mismatch() {
    let geom_src = RopeGeometry {
        layout: RopeLayout::HalfSplit,
        head_dim: 64,
        rotary_dim: 32,
        num_q_heads: 8,
        num_kv_heads: 4,
        num_layers: 4,
        layer_rope_mask: 0x0F,
    };
    let mut adapter_src = ModelSteeringAdapter::new(geom_src.clone()).expect("src adapter");
    adapter_src.configure_ego_from_preset(LiminalEgoPreset::Medium);
    let policy = RecallPolicy::default();
    let detector = AdaptiveGopCutDetector::default();
    let mut enc_src = DummyEncoder {
        layers: geom_src.num_layers,
        rot_len: geom_src.q_rot_len(),
    };
    let records = vec![
        SourceTextRecord {
            frame_index: 0,
            timestamp_epoch_sec: 1_700_000_000,
            role: AttractorRole::Fact,
            text: "Fact alpha boundary anchor".to_string(),
            energy: 2.0,
        },
        SourceTextRecord {
            frame_index: 1,
            timestamp_epoch_sec: 1_700_000_010,
            role: AttractorRole::Fact,
            text: "Fact beta coherence continuation".to_string(),
            energy: 1.5,
        },
    ];
    let ctx_src = RebakeContext {
        adapter: &adapter_src,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    let (mut tl, _) =
        LmnkTimeline::bake_from_subtitle_records(&records, &ctx_src, &mut enc_src).expect("bake");
    tl.set_logical_pframe_topology(adapter_src.export_logical_pframe_topology());

    // 1. Unified .lmnkc container roundtrip & inner LmnkContainerView verification.
    let mut bytes = tl
        .to_container_bytes(LMNK_CONTAINER_FLAG_COMPILED_CACHE)
        .expect("to_container_bytes");
    let view = LmnkContainerView::parse(&bytes).expect("LmnkContainerView::parse");
    assert!(view.section_bytes(LMNK_SECTION_RECALL_METADATA).is_some());
    assert!(view.section_bytes(LMNK_SECTION_LOGICAL_PFRAMES).is_some());
    let decoded = LmnkTimeline::from_bytes(&bytes).expect("from_bytes");
    assert_eq!(decoded, tl);

    // 2. Empty optional sections cleanly omitted (topology = empty, no persona circuit).
    let mut tl_min = tl.clone();
    tl_min.set_logical_pframe_topology(Vec::new());
    let min_bytes = tl_min
        .to_container_bytes(LMNK_CONTAINER_FLAG_COMPILED_CACHE)
        .expect("min container");
    let min_view = LmnkContainerView::parse(&min_bytes).expect("min parse");
    assert!(
        min_view
            .section_bytes(LMNK_SECTION_LOGICAL_PFRAMES)
            .is_none()
    );
    assert_eq!(LmnkTimeline::from_bytes(&min_bytes).unwrap(), tl_min);

    // 3. .lmnkp 64B-aligned Store ZIP roundtrip.
    let pkg_bytes = tl
        .export_lmnkp_bytes(Some("persona Demo {}"))
        .expect("export_lmnkp_bytes");
    let pkg_view = LmnkPackageView::parse(&pkg_bytes).expect("package parse");
    let compiled_entry = pkg_view
        .require_aligned64_entry(LMNKP_ENTRY_COMPILED)
        .expect("aligned compiled.lmnkc");
    assert_eq!(compiled_entry.data_offset % 64, 0);
    let _ = LmnkContainerView::parse(compiled_entry.payload).expect("inner container parse");
    let from_pkg = LmnkTimeline::import_lmnkp_bytes(&pkg_bytes).expect("import_lmnkp_bytes");
    assert_eq!(from_pkg, tl);

    // 4. Cross-model rebake on geometry mismatch.
    let mut geom_dst = geom_src;
    geom_dst.rotary_dim = 64;
    let adapter_dst = ModelSteeringAdapter::new(geom_dst.clone()).expect("dst adapter");
    let mut enc_dst = DummyEncoder {
        layers: geom_dst.num_layers,
        rot_len: geom_dst.q_rot_len(),
    };
    let ctx_dst = RebakeContext {
        adapter: &adapter_dst,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    match from_pkg
        .resume_or_rebake(&ctx_dst, &mut enc_dst, 1)
        .expect("rebake")
    {
        TimelineResumeOutcome::Rebaked {
            timeline: rebaked,
            rebaked_frames,
            ..
        } => {
            assert_eq!(rebaked_frames, 2);
            assert_eq!(
                rebaked.model_fingerprint(),
                LmnkTimeline::geometry_fingerprint(&geom_dst)
            );
        }
        TimelineResumeOutcome::FastResumed { .. } => {
            panic!("expected Rebaked on altered rotary_dim")
        }
    }

    // 5. Mutated topology_crc32 with valid recomputed header_crc32 yields TopologyMismatch.
    let mut hdr = LmnkContainerHeader::from_le_bytes(&bytes).expect("hdr");
    hdr.topology_crc32 ^= 0x00FF_00AA;
    let _ = hdr.finalize_crc32();
    bytes[..64].copy_from_slice(&hdr.to_le_bytes());
    assert!(matches!(
        LmnkTimeline::from_bytes(&bytes),
        Err(LmnkTimelineError::TopologyMismatch { .. })
    ));

    // 6. Signed .lmnkc and .lmnkp policy enforcement + downgrade rejection.
    let kp = SigningKeyPair::from_seed_bytes([0x77_u8; 32]);
    let pk = kp.verifying_key();
    let raw_c = tl
        .to_container_bytes(LMNK_CONTAINER_FLAG_COMPILED_CACHE)
        .expect("raw_c");
    assert!(
        LmnkTimeline::from_bytes_with_policy(&raw_c, &IntegrityPolicy::RequireSignature(pk))
            .is_err()
    );
    let signed_c = sign_lmnkc_container(&raw_c, &kp).expect("sign lmnkc");
    let verified_tl =
        LmnkTimeline::from_bytes_with_policy(&signed_c, &IntegrityPolicy::RequireSignature(pk))
            .expect("verified lmnkc timeline");
    assert_eq!(verified_tl, tl);

    let signed_pkg = sign_lmnkp_package(&pkg_bytes, &kp).expect("sign lmnkp");
    let verified_pkg_tl = LmnkTimeline::import_lmnkp_bytes_with_policy(
        &signed_pkg,
        &IntegrityPolicy::RequireSignature(pk),
    )
    .expect("verified lmnkp timeline");
    assert_eq!(verified_pkg_tl, tl);
}
