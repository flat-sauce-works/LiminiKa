// name: src/cli/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

mod eval_runner;
mod eval_samples;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LiminiKa GCSO Core Engine v0.1.1");

    // 1. Obtain version string from dynamic C-ABI
    let version_str = unsafe {
        let ptr = liminika_core::abi::gcso_abi_get_version_string();
        std::ffi::CStr::from_ptr(ptr).to_str()?
    };
    println!("Loaded C-ABI Engine Version: {}", version_str);

    // 2. Initialize runtime base configuration via C-ABI
    let mut config = liminika_core::abi::gcso_config_t::default();
    let status = unsafe { liminika_core::abi::gcso_config_init_default(&mut config) };
    assert_eq!(status, liminika_core::abi::GCSO_SUCCESS);

    // 3. Allocate context instance handle
    let mut ctx_handle: liminika_core::abi::GcsoContextHandle = std::ptr::null_mut();
    let status = unsafe { liminika_core::abi::gcso_context_create(&config, &mut ctx_handle) };
    assert_eq!(status, liminika_core::abi::GCSO_SUCCESS);

    println!("Context initialized successfully. Ready for token stepping.");

    // 4. Optional subcommand: `eval-check [fixture_path]`
    let mut args = std::env::args().skip(1);
    if let Some(cmd) = args.next() {
        if cmd == "eval-check" {
            let path = args
                .next()
                .unwrap_or_else(|| "tests/fixtures/gcso_eval_samples.txt".to_string());
            let raw = std::fs::read_to_string(&path)?;
            let suite = eval_samples::EvalSampleSuite::parse(&raw)
                .map_err(|e| format!("failed to parse {}: {}", path, e))?;
            println!(
                "Loaded eval fixture '{}': {} facts, {} persona samples, {} guards, {} probes.",
                path,
                suite.facts.len(),
                suite.persona.len(),
                suite.guards.len(),
                suite.probes.len()
            );
            for (i, probe) in suite.probes.iter().enumerate() {
                println!(
                    "  [Probe {} | {:?}] {} (expect: {}, forbid: {})",
                    i + 1,
                    probe.slot,
                    probe.prompt,
                    probe.expected_keywords.join(", "),
                    probe.forbidden_keywords.join(", ")
                );
            }
        } else if cmd == "eval-dry-run" {
            let path = args
                .next()
                .unwrap_or_else(|| "tests/fixtures/gcso_eval_samples.txt".to_string());
            let raw = std::fs::read_to_string(&path)?;
            let suite = eval_samples::EvalSampleSuite::parse(&raw)
                .map_err(|e| format!("failed to parse {}: {}", path, e))?;

            let geom = eval_runner::RopeGeometry {
                layout: eval_runner::RopeLayout::HalfSplit,
                head_dim: 128,
                rotary_dim: 64,
                num_q_heads: 24,
                num_kv_heads: 8,
                num_layers: 32,
                layer_rope_mask: 0xFFFF_FFFF,
            };
            let mut backend = eval_runner::DummyBackend::new(geom);
            let profile = eval_runner::bake_in_memory_profile(&mut backend, &suite)
                .map_err(|e| format!("bake failed: {}", e))?;
            let angles = [0.0_f32, 0.05, 0.10, 0.15, 0.25];
            let rows = eval_runner::run_angle_sweep(&mut backend, &suite, &profile, &angles, 48)
                .map_err(|e| format!("sweep failed: {}", e))?;

            println!(
                "Dry-run sweep on '{}' (active RoPE layers: {}):",
                path, profile.active_rope_layers
            );
            println!(
                "{:>8} | {:>10} | {:>12} | {:>11} | {:>11}",
                "max_rad", "mean_KL", "d_logprob", "expect_hit", "forbid_hit"
            );
            for r in rows {
                println!(
                    "{:>8.2} | {:>10.4} | {:>+12.4} | {:>10.1}% | {:>10.1}%",
                    r.max_rad,
                    r.mean_kl_div,
                    r.mean_target_logprob_delta,
                    r.expected_hit_rate * 100.0,
                    r.forbidden_hit_rate * 100.0
                );
            }
        } else if cmd == "gcso-list" {
            use liminika_adapter::GcsoDualTrackTimeline;

            let dir = args.next().unwrap_or_else(|| "containers".to_string());
            let entries = collect_gcso_paths_creating_dir(std::path::Path::new(&dir))?;

            if entries.is_empty() {
                println!("No .gcso containers found in '{}'.", dir);
            } else {
                println!("Found {} .gcso container(s) in '{}':", entries.len(), dir);
                for path in entries {
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    match GcsoDualTrackTimeline::load_from_path(&path) {
                        Ok(tl) => println!(
                            "  {} | {} bytes | fp=0x{:016x} | frames={} | keyframes={:?}",
                            path.display(),
                            size,
                            tl.model_fingerprint(),
                            tl.frames().len(),
                            tl.keyframe_positions()
                        ),
                        Err(e) => println!("  {} | invalid ({})", path.display(), e),
                    }
                }
            }
        } else if cmd == "gcso-inspect" {
            use liminika_adapter::{AttractorContainerCodec, GcsoDualTrackTimeline};

            let path = args
                .next()
                .ok_or("usage: liminika gcso-inspect <file.gcso> [seek_frame]")?;
            let seek_frame: Option<u64> = args.next().map(|s| s.parse::<u64>()).transpose()?;
            let timeline = GcsoDualTrackTimeline::load_from_path(&path)?;
            println!(
                "Loaded .gcso '{}': fingerprint=0x{:016x} frames={} keyframes={:?}",
                path,
                timeline.model_fingerprint(),
                timeline.frames().len(),
                timeline.keyframe_positions()
            );
            for rec in timeline.export_subtitle_track(None) {
                println!(
                    "  [Frame {} | {:?} | E={:.2}] {}",
                    rec.frame_index, rec.role, rec.energy, rec.text
                );
            }
            if let Some(target) = seek_frame {
                let state = timeline.seek_zero_forward(target)?;
                println!(
                    "Zero-Forward Seek(frame={}): keyframe={} p_frames={} total_energy={:.2} anchors={}",
                    state.target_frame_index,
                    state.keyframe_index,
                    state.p_frames_applied,
                    state.total_energy,
                    state.anchor_ids.len()
                );
            }
        } else if cmd == "gcso-bake" {
            let fixture_path = args
                .next()
                .unwrap_or_else(|| "tests/fixtures/gcso_eval_samples.txt".to_string());
            let dst_path = args
                .next()
                .unwrap_or_else(|| "containers/fixture_baked.gcso".to_string());
            let profile = args.next().unwrap_or_else(|| "phi4-mini".to_string());
            let (timeline, written_bytes) = bake_fixture_to_gcso_file(
                std::path::Path::new(&fixture_path),
                std::path::Path::new(&dst_path),
                &profile,
            )?;
            println!(
                "Baked '{}' -> '{}' ({} bytes) | profile={} fp=0x{:016x} frames={} keyframes={:?}",
                fixture_path,
                dst_path,
                written_bytes,
                profile,
                timeline.model_fingerprint(),
                timeline.frames().len(),
                timeline.keyframe_positions()
            );
        } else if cmd == "gcso-rebake" {
            use liminika_adapter::{
                AdaptiveGopCutDetector, GcsoDualTrackTimeline, ModelSteeringAdapter, RebakeContext,
                RecallPolicy, TimelineResumeOutcome,
            };

            let src_path = args
                .next()
                .ok_or("usage: liminika gcso-rebake <src.gcso> <dst.gcso> [phi4-mini|alt-128]")?;
            let dst_path = args
                .next()
                .ok_or("usage: liminika gcso-rebake <src.gcso> <dst.gcso> [phi4-mini|alt-128]")?;
            let profile = args.next().unwrap_or_else(|| "alt-128".to_string());

            let target_geom = resolve_cli_rope_geometry(&profile)?;
            let src_tl = GcsoDualTrackTimeline::load_from_path(&src_path)?;
            let target_adapter = ModelSteeringAdapter::new(target_geom)?;
            let policy = RecallPolicy::default();
            let detector = AdaptiveGopCutDetector::default();
            let mut encoder = CliProjectionEncoder {
                num_layers: target_adapter.geometry().num_layers,
                rotary_dim: target_adapter.geometry().q_rot_len(),
            };
            let ctx = RebakeContext {
                adapter: &target_adapter,
                policy: &policy,
                detector: &detector,
                centroid_layers: &[],
                up_to_frame: None,
            };
            let target_frame = src_tl.frames().last().map_or(0, |f| f.frame_index);
            match src_tl.resume_or_rebake(&ctx, &mut encoder, target_frame)? {
                TimelineResumeOutcome::FastResumed { seek_state } => {
                    let bytes = src_tl.save_to_path(&dst_path)?;
                    println!(
                        "EnvProbe matched (fp=0x{:016x}); FastResumed frame={} and copied '{}' ({} bytes).",
                        src_tl.model_fingerprint(),
                        seek_state.target_frame_index,
                        dst_path,
                        bytes
                    );
                }
                TimelineResumeOutcome::Rebaked {
                    timeline,
                    seek_state,
                    source_fingerprint,
                    target_fingerprint,
                    rebaked_frames,
                    ..
                } => {
                    let bytes = timeline.save_to_path(&dst_path)?;
                    println!(
                        "Rebaked {} frames: 0x{:016x} -> 0x{:016x} | saved '{}' ({} bytes) | seek_frame={} anchors={}",
                        rebaked_frames,
                        source_fingerprint,
                        target_fingerprint,
                        dst_path,
                        bytes,
                        seek_state.target_frame_index,
                        seek_state.anchor_ids.len()
                    );
                }
            }
        } else {
            println!("Unknown or help command '{}'. Available subcommands:", cmd);
            println!("  eval-check [fixture_path]");
            println!("  eval-dry-run [fixture_path]");
            println!("  gcso-list [containers_dir]");
            println!("  gcso-inspect <file.gcso> [seek_frame]");
            println!("  gcso-bake [fixture_path] [dst.gcso] [phi4-mini|alt-128]");
            println!("  gcso-rebake <src.gcso> <dst.gcso> [phi4-mini|alt-128]");
        }
    }

    // 5. Clean up allocated context resources
    unsafe { liminika_core::abi::gcso_context_destroy(ctx_handle) };
    Ok(())
}

struct CliProjectionEncoder {
    num_layers: usize,
    rotary_dim: usize,
}

impl liminika_adapter::SubtitleActivationEncoder for CliProjectionEncoder {
    fn encode_subtitle_activations(
        &mut self,
        record: &liminika_adapter::SourceTextRecord,
    ) -> Result<Vec<Vec<f32>>, String> {
        let seed = record
            .text
            .bytes()
            .fold(0x9E37_79B9_7F4A_7C15_u64, |acc, b| {
                acc.wrapping_mul(131).wrapping_add(u64::from(b))
            });
        let mut layers = Vec::with_capacity(self.num_layers);
        for l in 0..self.num_layers {
            let mut v = Vec::with_capacity(self.rotary_dim);
            for d in 0..self.rotary_dim {
                let phase = ((seed ^ ((l as u64) << 8) ^ (d as u64)) % 100) as f32 * 0.01;
                v.push((phase - 0.5) * 2.0);
            }
            layers.push(v);
        }
        Ok(layers)
    }
}

fn resolve_cli_rope_geometry(profile: &str) -> Result<liminika_adapter::RopeGeometry, String> {
    match profile {
        "phi4-mini" => Ok(liminika_adapter::RopeGeometry::phi4_mini()),
        "alt-128" => {
            let mut g = liminika_adapter::RopeGeometry::phi4_mini();
            g.head_dim = 128;
            g.rotary_dim = 128;
            Ok(g)
        }
        other => Err(format!(
            "unknown geometry profile '{other}' (expected phi4-mini or alt-128)"
        )),
    }
}

/// Parses a sectioned evaluation fixture and compiles its attractors into a portable `.gcso`
/// dual-track container at `dst_path`.
fn bake_fixture_to_gcso_file(
    fixture_path: &std::path::Path,
    dst_path: &std::path::Path,
    profile: &str,
) -> Result<(liminika_adapter::GcsoDualTrackTimeline, usize), Box<dyn std::error::Error>> {
    use liminika_adapter::{
        AdaptiveGopCutDetector, AttractorRole, GcsoDualTrackTimeline, ModelSteeringAdapter,
        RebakeContext, RecallPolicy, SourceTextRecord,
    };

    let raw = std::fs::read_to_string(fixture_path)?;
    let suite = eval_samples::EvalSampleSuite::parse(&raw)
        .map_err(|e| format!("failed to parse {}: {}", fixture_path.display(), e))?;

    let mut records = Vec::new();
    let mut frame_index = 0_u64;
    for (role, texts, energy) in [
        (AttractorRole::Fact, suite.facts.as_slice(), 2.5_f32),
        (AttractorRole::Persona, suite.persona.as_slice(), 1.5_f32),
        (AttractorRole::Guard, suite.guards.as_slice(), 2.0_f32),
    ] {
        for text in texts {
            records.push(SourceTextRecord {
                frame_index,
                timestamp_epoch_sec: 1_700_000_000 + frame_index * 10,
                role,
                text: text.clone(),
                energy,
            });
            frame_index += 1;
        }
    }

    let target_geom = resolve_cli_rope_geometry(profile)?;
    let target_adapter = ModelSteeringAdapter::new(target_geom)?;
    let policy = RecallPolicy::default();
    let detector = AdaptiveGopCutDetector::default();
    let mut encoder = CliProjectionEncoder {
        num_layers: target_adapter.geometry().num_layers,
        rotary_dim: target_adapter.geometry().q_rot_len(),
    };
    let ctx = RebakeContext {
        adapter: &target_adapter,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    let (timeline, _) =
        GcsoDualTrackTimeline::bake_from_subtitle_records(&records, &ctx, &mut encoder)?;
    let written_bytes = timeline.save_to_path(dst_path)?;
    Ok((timeline, written_bytes))
}

/// Ensures `dir` exists on demand (resilient when `.gitkeep` is absent) and returns sorted `.gcso`
/// container paths inside it.
fn collect_gcso_paths_creating_dir(
    dir: &std::path::Path,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|ext| ext.to_str()) == Some("gcso"))
        .collect();
    entries.sort();
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::{bake_fixture_to_gcso_file, collect_gcso_paths_creating_dir};
    use liminika_adapter::{AttractorContainerCodec, GcsoDualTrackTimeline};

    #[test]
    fn gcso_list_creates_missing_directory_on_demand() {
        let base =
            std::env::temp_dir().join(format!("liminika_cli_missing_dir_{}", std::process::id()));
        let target = base.join("nested").join("containers");
        let _ = std::fs::remove_dir_all(&base);
        assert!(!target.exists());

        let found = collect_gcso_paths_creating_dir(&target)
            .expect("missing containers directory should be created automatically");
        assert!(target.is_dir());
        assert!(found.is_empty());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn gcso_bake_round_trip_creates_inspectable_container() {
        let base =
            std::env::temp_dir().join(format!("liminika_cli_bake_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create temp test dir");

        let fixture_path = base.join("sample_fixture.txt");
        let fixture_raw = include_str!("../../../tests/fixtures/gcso_eval_samples.txt");
        std::fs::write(&fixture_path, fixture_raw).expect("write temp fixture");

        let dst_path = base.join("out").join("baked.gcso");
        let (timeline, written) = bake_fixture_to_gcso_file(&fixture_path, &dst_path, "phi4-mini")
            .expect("gcso-bake should succeed");
        assert!(written > 0);
        assert!(dst_path.is_file());

        let reloaded = GcsoDualTrackTimeline::load_from_path(&dst_path).expect("reload baked gcso");
        assert_eq!(reloaded, timeline);
        assert!(!reloaded.frames().is_empty());
        assert_eq!(
            reloaded.export_subtitle_track(None).len(),
            reloaded.frames().len()
        );
        let last_idx = reloaded.frames().last().map_or(0, |f| f.frame_index);
        let seeked = reloaded
            .seek_zero_forward(last_idx)
            .expect("zero-forward seek on baked container should succeed");
        assert_eq!(seeked.target_frame_index, last_idx);

        let _ = std::fs::remove_dir_all(&base);
    }
}
