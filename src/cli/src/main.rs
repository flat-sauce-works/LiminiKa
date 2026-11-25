// name: src/cli/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Command-line entry point for inspecting, baking, packaging, and evaluating LiminiKa containers.

mod eval_runner;
mod eval_samples;

fn print_usage() {
    println!("LiminiKa CLI v{}", env!("CARGO_PKG_VERSION"));
    println!("Available subcommands:");
    println!("  check <file.lmnk>");
    println!("  eval-check [fixture_path]");
    println!("  eval-dry-run [fixture_path]");
    println!("  list [containers_dir]");
    println!(
        "  inspect <file.lmnkc|file.lmnkp> [seek_frame] [--pubkey <file|hex>] [--require-hash]"
    );
    println!("  verify <file.lmnkc|file.lmnkp> [--pubkey <file|hex>] [--require-hash]");
    println!(
        "  sign <src.lmnkc|src.lmnkp> <dst.lmnkc|dst.lmnkp> (--key <secret_file> | --hash-only)"
    );
    println!("  keygen <secret_key_out> [public_key_out]");
    println!("  bake [fixture_path] [dst.lmnkc|dst.lmnkp] [phi4-mini|alt-128] [--package]");
    println!("  rebake <src.lmnkc|src.lmnkp> <dst.lmnkc|dst.lmnkp> [phi4-mini|alt-128]");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let Some(cmd) = args.next() else {
        return run_default_c_abi_smoke();
    };

    if matches!(cmd.as_str(), "--help" | "-h" | "help") {
        print_usage();
        return Ok(());
    }

    if cmd == "check" {
        let path = args.next().ok_or("usage: liminika check <file.lmnk>")?;
        let raw = std::fs::read_to_string(&path)?;
        let program = liminika_dsl::parse_and_validate_lmnk(&raw)
            .map_err(|e| format!("failed to validate {path}: {e}"))?;
        println!(
            "Validated '{}': {} persona(s), {} session(s), {} target(s).",
            path,
            program.personas.len(),
            program.sessions.len(),
            program.targets.len()
        );
    } else if cmd == "eval-check" {
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
    } else if cmd == "list" {
        use liminika_adapter::{LmnkTimeline, LmnkTimelineError};

        let dir = args.next().unwrap_or_else(|| "containers".to_string());
        let entries = collect_lmnk_paths_creating_dir(std::path::Path::new(&dir))?;

        if entries.is_empty() {
            println!("No .lmnkc or .lmnkp artifacts found in '{}'.", dir);
        } else {
            println!(
                "Found {} .lmnkc/.lmnkp artifact(s) in '{}':",
                entries.len(),
                dir
            );
            for path in entries {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                match LmnkTimeline::load_from_path(&path) {
                    Ok(tl) => println!(
                        "  {} | {} bytes | fp=0x{:016x} | frames={} | keyframes={:?}",
                        path.display(),
                        size,
                        tl.model_fingerprint(),
                        tl.frames().len(),
                        tl.keyframe_positions()
                    ),
                    Err(LmnkTimelineError::TopologyMismatch {
                        expected_crc32,
                        found_crc32,
                    }) => println!(
                        "  {} | {} bytes | topology mismatch (expected=0x{:08x}, found=0x{:08x}; run `rebake`)",
                        path.display(),
                        size,
                        expected_crc32,
                        found_crc32
                    ),
                    Err(e) => println!("  {} | invalid ({})", path.display(), e),
                }
            }
        }
    } else if cmd == "inspect" {
        use liminika_adapter::{
            verify_lmnkc_integrity, AttractorContainerCodec, IntegrityPolicy, LmnkPackageOwned,
            LmnkTimeline, LmnkTimelineError,
        };
        use liminika_core::liminal::LMNK_ERR_TOPOLOGY_MISMATCH;
        use liminika_core::runtime::{LmnkContainerHeader, LmnkContainerView, CORE_TOPOLOGY};

        let mut pubkey_arg: Option<String> = None;
        let mut require_hash = false;
        let mut positional = Vec::new();
        while let Some(arg) = args.next() {
            if arg == "--pubkey" {
                let val = args
                    .next()
                    .ok_or("missing value after --pubkey (expected key file path or 64-hex)")?;
                pubkey_arg = Some(val);
            } else if arg == "--require-hash" {
                require_hash = true;
            } else {
                positional.push(arg);
            }
        }
        let mut pos_iter = positional.into_iter();
        let path = pos_iter.next().ok_or(
            "usage: liminika inspect <file.lmnkc|file.lmnkp> [seek_frame] [--pubkey <file|hex>] [--require-hash]",
        )?;
        let seek_frame: Option<u64> = pos_iter.next().map(|s| s.parse::<u64>()).transpose()?;
        let policy = build_integrity_policy(pubkey_arg.as_deref(), require_hash)?;
        let is_pkg = is_lmnkp_path(std::path::Path::new(&path));

        let owned_pkg = if is_pkg {
            let pkg = LmnkPackageOwned::load_from_path(&path)?;
            {
                let view = pkg.view()?;
                let report = view.verify_integrity(&policy)?;
                println!(
                    "Loaded .lmnkp package '{}': {} entries | integrity={} sha256={} hash={} sig={}",
                    path,
                    view.entries().len(),
                    report.trust_level,
                    report.sha256_hex(),
                    report.has_content_hash,
                    report.has_signature
                );
                for entry in view.entries() {
                    println!(
                        "  entry '{}' | offset={} (align64={}) | size={} bytes | crc32=0x{:08x}",
                        entry.name,
                        entry.data_offset,
                        entry.is_aligned64(),
                        entry.payload.len(),
                        entry.crc32
                    );
                }
                if let Some(manifest) = view.manifest_json()? {
                    println!("  manifest.json: {}", manifest.trim());
                }
            }
            Some(pkg)
        } else {
            None
        };

        let owned_raw = if owned_pkg.is_none() {
            Some(read_bounded_artifact(std::path::Path::new(&path))?)
        } else {
            None
        };
        let container_bytes: &[u8] = if let Some(ref pkg) = owned_pkg {
            pkg.view()?.compiled_lmnkc()?
        } else {
            owned_raw.as_deref().unwrap_or(&[])
        };

        match LmnkContainerView::parse(container_bytes) {
            Ok(view) => {
                let container_policy = if owned_pkg.is_some() {
                    IntegrityPolicy::AllowUnsigned
                } else {
                    policy
                };
                let report = verify_lmnkc_integrity(container_bytes, &container_policy)?;
                let hdr = view.header();
                println!(
                    concat!(
                        "Container '{}': abi=0x{:08x} flags=0x{:08x} topology_crc32=0x{:08x}",
                        " sections={} | integrity={} sha256={} hash={} sig={}"
                    ),
                    path,
                    hdr.version,
                    hdr.container_flags,
                    hdr.topology_crc32,
                    hdr.section_count,
                    report.trust_level,
                    report.sha256_hex(),
                    report.has_content_hash,
                    report.has_signature
                );
                for sec in view.sections() {
                    println!(
                        "  section kind=0x{:04x} flags=0x{:04x} offset={} length={} crc32=0x{:08x}",
                        sec.kind, sec.flags, sec.offset, sec.length, sec.payload_crc32
                    );
                }
            }
            Err(status) if status == LMNK_ERR_TOPOLOGY_MISMATCH => {
                let hdr = LmnkContainerHeader::from_le_bytes(container_bytes)
                    .map_err(|e| format!("failed to read container header: {e}"))?;
                println!(
                    "Container '{}' requires rebake: found topology_crc32=0x{:08x}, active=0x{:08x}",
                    path,
                    hdr.topology_crc32,
                    CORE_TOPOLOGY.topology_crc32
                );
                return Err(LmnkTimelineError::TopologyMismatch {
                    expected_crc32: CORE_TOPOLOGY.topology_crc32,
                    found_crc32: hdr.topology_crc32,
                }
                .into());
            }
            Err(status) => {
                return Err(format!("invalid .lmnkc container '{}': status={status}", path).into());
            }
        }

        let timeline = LmnkTimeline::from_bytes(container_bytes)?;
        println!(
            "Decoded timeline '{}': fingerprint=0x{:016x} frames={} keyframes={:?}",
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
    } else if cmd == "verify" {
        let mut pubkey_arg: Option<String> = None;
        let mut require_hash = false;
        let mut positional = Vec::new();
        while let Some(arg) = args.next() {
            if arg == "--pubkey" {
                let val = args
                    .next()
                    .ok_or("missing value after --pubkey (expected key file path or 64-hex)")?;
                pubkey_arg = Some(val);
            } else if arg == "--require-hash" {
                require_hash = true;
            } else {
                positional.push(arg);
            }
        }
        let path = positional.into_iter().next().ok_or(
            "usage: liminika verify <file.lmnkc|file.lmnkp> [--pubkey <file|hex>] [--require-hash]",
        )?;
        let policy = build_integrity_policy(pubkey_arg.as_deref(), require_hash)?;
        let report = run_verify_artifact(std::path::Path::new(&path), &policy)?;
        println!(
            "Verified '{}': status={} sha256={} hash={} sig={}",
            path,
            report.trust_level,
            report.sha256_hex(),
            report.has_content_hash,
            report.has_signature
        );
    } else if cmd == "sign" {
        let mut key_path: Option<String> = None;
        let mut hash_only = false;
        let mut positional = Vec::new();
        while let Some(arg) = args.next() {
            if arg == "--key" {
                let val = args
                    .next()
                    .ok_or("missing value after --key (expected secret key file path)")?;
                key_path = Some(val);
            } else if arg == "--hash-only" {
                hash_only = true;
            } else {
                positional.push(arg);
            }
        }
        let mut pos_iter = positional.into_iter();
        let src = pos_iter.next().ok_or(
            "usage: liminika sign <src.lmnkc|src.lmnkp> <dst.lmnkc|dst.lmnkp> (--key <secret_file> | --hash-only)",
        )?;
        let dst = pos_iter.next().ok_or(
            "usage: liminika sign <src.lmnkc|src.lmnkp> <dst.lmnkc|dst.lmnkp> (--key <secret_file> | --hash-only)",
        )?;
        if hash_only && key_path.is_some() {
            return Err("cannot combine --key and --hash-only".into());
        }
        if !hash_only && key_path.is_none() {
            return Err("specify either --key <secret_file> or --hash-only".into());
        }
        let summary = run_sign_artifact(
            std::path::Path::new(&src),
            std::path::Path::new(&dst),
            key_path.as_deref().map(std::path::Path::new),
        )?;
        println!(
            "Wrote '{}' ({} bytes) | status={} sha256={}{}",
            dst,
            summary.written_bytes,
            summary.report.trust_level,
            summary.report.sha256_hex(),
            summary
                .pubkey_hex
                .as_deref()
                .map_or(String::new(), |pk| format!(" pubkey={pk}"))
        );
    } else if cmd == "keygen" {
        let secret_out = args
            .next()
            .ok_or("usage: liminika keygen <secret_key_out> [public_key_out]")?;
        let pub_out = args.next().unwrap_or_else(|| format!("{secret_out}.pub"));
        let pubkey = run_keygen(
            std::path::Path::new(&secret_out),
            std::path::Path::new(&pub_out),
        )?;
        println!(
            "Generated Ed25519 keypair: secret='{}' public='{}' pubkey={}",
            secret_out,
            pub_out,
            pubkey.to_hex()
        );
    } else if cmd == "bake" {
        let mut as_package = false;
        let mut positional = Vec::new();
        for arg in args {
            if arg == "--package" {
                as_package = true;
            } else {
                positional.push(arg);
            }
        }
        let mut pos_iter = positional.into_iter();
        let fixture_path = pos_iter
            .next()
            .unwrap_or_else(|| "tests/fixtures/gcso_eval_samples.txt".to_string());
        let dst_path = pos_iter.next().unwrap_or_else(|| {
            if as_package {
                "containers/fixture_baked.lmnkp".to_string()
            } else {
                "containers/fixture_baked.lmnkc".to_string()
            }
        });
        let profile = pos_iter.next().unwrap_or_else(|| "phi4-mini".to_string());
        if dst_path.ends_with(".lmnkp") {
            as_package = true;
        }
        let (timeline, written_bytes) = bake_fixture_to_lmnk_file(
            std::path::Path::new(&fixture_path),
            std::path::Path::new(&dst_path),
            &profile,
            as_package,
        )?;
        println!(
            "Baked '{}' -> '{}' ({} bytes, package={}) | profile={} fp=0x{:016x} frames={} keyframes={:?}",
            fixture_path,
            dst_path,
            written_bytes,
            as_package,
            profile,
            timeline.model_fingerprint(),
            timeline.frames().len(),
            timeline.keyframe_positions()
        );
    } else if cmd == "rebake" {
        use liminika_adapter::{
            AdaptiveGopCutDetector, LmnkTimeline, ModelSteeringAdapter, RebakeContext,
            RecallPolicy, TimelineResumeOutcome,
        };

        let src_path = args.next().ok_or(
            "usage: liminika rebake <src.lmnkc|src.lmnkp> <dst.lmnkc|dst.lmnkp> [phi4-mini|alt-128]",
        )?;
        let dst_path = args.next().ok_or(
            "usage: liminika rebake <src.lmnkc|src.lmnkp> <dst.lmnkc|dst.lmnkp> [phi4-mini|alt-128]",
        )?;
        liminika_adapter::validate_local_output_path(std::path::Path::new(&dst_path))
            .map_err(|e| format!("unsafe output path '{dst_path}': {e}"))?;
        let profile = args.next().unwrap_or_else(|| "alt-128".to_string());
        let dst_is_pkg = std::path::Path::new(&dst_path)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("lmnkp"));

        let target_geom = resolve_cli_rope_geometry(&profile)?;
        let src_tl = LmnkTimeline::load_from_path(&src_path)?;
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
                let bytes = if dst_is_pkg {
                    src_tl.save_lmnkp_to_path(&dst_path, None)?
                } else {
                    src_tl.save_to_path(&dst_path)?
                };
                println!(
                    "EnvProbe matched (fp=0x{:016x}); FastResumed frame={} and wrote '{}' ({} bytes).",
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
                let bytes = if dst_is_pkg {
                    timeline.save_lmnkp_to_path(&dst_path, None)?
                } else {
                    timeline.save_to_path(&dst_path)?
                };
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
        eprintln!("Unknown command '{}'.", cmd);
        print_usage();
        return Err(format!("unknown subcommand '{cmd}'").into());
    }

    Ok(())
}

fn run_default_c_abi_smoke() -> Result<(), Box<dyn std::error::Error>> {
    println!("LiminiKa Core Engine v{}", env!("CARGO_PKG_VERSION"));

    // SAFETY: `lmnk_abi_version_string` returns a static NUL-terminated C string literal.
    let version_str = unsafe {
        let ptr = liminika_core::abi::lmnk_abi_version_string();
        std::ffi::CStr::from_ptr(ptr).to_str()?
    };
    println!("Loaded C-ABI Engine Version: {}", version_str);

    let mut config = liminika_core::abi::lmnk_config_t::default();
    // SAFETY: `&mut config` points to a valid, uniquely borrowed `lmnk_config_t`.
    let status = unsafe { liminika_core::abi::lmnk_config_init_default(&mut config) };
    assert_eq!(status, liminika_core::abi::LMNK_OK);

    let mut ctx_handle: liminika_core::abi::LmnkContextHandle = std::ptr::null_mut();
    // SAFETY: `&config` and `&mut ctx_handle` are valid, aligned stack references.
    let status = unsafe { liminika_core::abi::lmnk_context_create(&config, &mut ctx_handle) };
    assert_eq!(status, liminika_core::abi::LMNK_OK);

    println!("Context initialized successfully. Ready for token stepping.");

    // SAFETY: `ctx_handle` was initialized by `lmnk_context_create` and is destroyed once.
    unsafe { liminika_core::abi::lmnk_context_destroy(ctx_handle) };
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

fn is_lmnkp_path(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("lmnkp"))
}

fn read_bounded_artifact(path: &std::path::Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::io::Read as _;
    let cap = liminika_adapter::LMNKP_MAX_PACKAGE_BYTES;
    let file = std::fs::File::open(path)?;
    let meta_len = file.metadata()?.len();
    if meta_len > cap as u64 {
        return Err(format!(
            "artifact '{}' ({meta_len} B) exceeds hard cap {cap} B",
            path.display()
        )
        .into());
    }
    let mut bytes = Vec::with_capacity((meta_len as usize).min(cap));
    file.take((cap as u64).saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() > cap {
        return Err(format!(
            "artifact '{}' exceeds hard read cap {cap} B",
            path.display()
        )
        .into());
    }
    Ok(bytes)
}

fn build_integrity_policy(
    pubkey_arg: Option<&str>,
    require_hash: bool,
) -> Result<liminika_adapter::IntegrityPolicy, Box<dyn std::error::Error>> {
    use liminika_adapter::{IntegrityPolicy, VerifyingPublicKey};
    if let Some(arg) = pubkey_arg {
        let p = std::path::Path::new(arg);
        let pk = if p.is_file() {
            VerifyingPublicKey::read_hex_file(p)?
        } else {
            VerifyingPublicKey::from_hex(arg)?
        };
        Ok(IntegrityPolicy::RequireSignature(pk))
    } else if require_hash {
        Ok(IntegrityPolicy::RequireContentHash)
    } else {
        Ok(IntegrityPolicy::AllowUnsigned)
    }
}

struct CliSignSummary {
    written_bytes: usize,
    report: liminika_adapter::IntegrityReport,
    pubkey_hex: Option<String>,
}

fn run_keygen(
    secret_out: &std::path::Path,
    pub_out: &std::path::Path,
) -> Result<liminika_adapter::VerifyingPublicKey, Box<dyn std::error::Error>> {
    liminika_adapter::validate_local_output_path(secret_out)?;
    liminika_adapter::validate_local_output_path(pub_out)?;
    let kp = liminika_adapter::SigningKeyPair::generate()?;
    let pk = kp.verifying_key();
    kp.write_seed_hex_file(secret_out)?;
    pk.write_hex_file(pub_out)?;
    Ok(pk)
}

fn run_sign_artifact(
    src: &std::path::Path,
    dst: &std::path::Path,
    key_file: Option<&std::path::Path>,
) -> Result<CliSignSummary, Box<dyn std::error::Error>> {
    use liminika_adapter::{
        attach_lmnkc_content_hash, attach_lmnkp_content_hash, sign_lmnkc_container,
        sign_lmnkp_package, validate_local_output_path, verify_lmnkc_integrity,
        verify_lmnkp_integrity, IntegrityPolicy, SigningKeyPair,
    };

    validate_local_output_path(dst)?;
    let raw = read_bounded_artifact(src)?;
    let is_pkg = raw.starts_with(b"PK\x03\x04") || is_lmnkp_path(src);

    let (signed_bytes, pubkey_opt) = if let Some(kf) = key_file {
        let kp = SigningKeyPair::read_seed_hex_file(kf)?;
        let pk = kp.verifying_key();
        let out = if is_pkg {
            sign_lmnkp_package(&raw, &kp)?
        } else {
            sign_lmnkc_container(&raw, &kp)?
        };
        (out, Some(pk))
    } else {
        let out = if is_pkg {
            attach_lmnkp_content_hash(&raw)?
        } else {
            attach_lmnkc_content_hash(&raw)?
        };
        (out, None)
    };

    let check_policy = match pubkey_opt {
        Some(pk) => IntegrityPolicy::RequireSignature(pk),
        None => IntegrityPolicy::RequireContentHash,
    };
    let report = if is_pkg {
        verify_lmnkp_integrity(&signed_bytes, &check_policy)?
    } else {
        verify_lmnkc_integrity(&signed_bytes, &check_policy)?
    };

    if let Some(parent) = dst.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(dst, &signed_bytes)?;
    Ok(CliSignSummary {
        written_bytes: signed_bytes.len(),
        report,
        pubkey_hex: pubkey_opt.map(|pk| pk.to_hex()),
    })
}

fn run_verify_artifact(
    path: &std::path::Path,
    policy: &liminika_adapter::IntegrityPolicy,
) -> Result<liminika_adapter::IntegrityReport, Box<dyn std::error::Error>> {
    use liminika_adapter::{verify_lmnkc_integrity, IntegrityPolicy, LmnkPackageView};

    let raw = read_bounded_artifact(path)?;
    if raw.starts_with(b"PK\x03\x04") || is_lmnkp_path(path) {
        let view = LmnkPackageView::parse(&raw)?;
        let report = view.verify_integrity(policy)?;
        if let Ok(compiled) = view.compiled_lmnkc() {
            let _ = verify_lmnkc_integrity(compiled, &IntegrityPolicy::AllowUnsigned)?;
        }
        Ok(report)
    } else {
        Ok(verify_lmnkc_integrity(&raw, policy)?)
    }
}

/// Parses a sectioned evaluation fixture and compiles its attractors into a portable `.lmnkc`
/// dual-track container or 64B-aligned `.lmnkp` Store ZIP package at `dst_path`.
fn bake_fixture_to_lmnk_file(
    fixture_path: &std::path::Path,
    dst_path: &std::path::Path,
    profile: &str,
    as_package: bool,
) -> Result<(liminika_adapter::LmnkTimeline, usize), Box<dyn std::error::Error>> {
    liminika_adapter::validate_local_output_path(dst_path)?;
    use liminika_adapter::{
        AdaptiveGopCutDetector, AttractorRole, LmnkTimeline, ModelSteeringAdapter, RebakeContext,
        RecallPolicy, SourceTextRecord,
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
    let (timeline, _) = LmnkTimeline::bake_from_subtitle_records(&records, &ctx, &mut encoder)?;
    let written_bytes = if as_package
        || dst_path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("lmnkp"))
    {
        timeline.save_lmnkp_to_path(dst_path, None)?
    } else {
        timeline.save_to_path(dst_path)?
    };
    Ok((timeline, written_bytes))
}

/// Ensures `dir` exists on demand (resilient when `.gitkeep` is absent) and returns sorted `.lmnkc`
/// and `.lmnkp` artifact paths inside it.
fn collect_lmnk_paths_creating_dir(
    dir: &std::path::Path,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    if !dir.exists() {
        std::fs::create_dir_all(dir)?;
    }
    let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|ext| ext.to_str()),
                Some("lmnkc" | "lmnkp")
            )
        })
        .collect();
    entries.sort();
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::{bake_fixture_to_lmnk_file, collect_lmnk_paths_creating_dir};
    use liminika_adapter::{AttractorContainerCodec, LmnkPackageOwned, LmnkTimeline};

    #[test]
    fn list_creates_missing_directory_on_demand() {
        let base =
            std::env::temp_dir().join(format!("liminika_cli_missing_dir_{}", std::process::id()));
        let target = base.join("nested").join("containers");
        let _ = std::fs::remove_dir_all(&base);
        assert!(!target.exists());

        let found = collect_lmnk_paths_creating_dir(&target)
            .expect("missing containers directory should be created automatically");
        assert!(target.is_dir());
        assert!(found.is_empty());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn bake_round_trip_creates_inspectable_lmnkc_and_aligned_lmnkp() {
        let base =
            std::env::temp_dir().join(format!("liminika_cli_bake_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create temp test dir");

        let fixture_path = base.join("sample_fixture.txt");
        let fixture_raw = include_str!("../../../tests/fixtures/gcso_eval_samples.txt");
        std::fs::write(&fixture_path, fixture_raw).expect("write temp fixture");

        let dst_path = base.join("out").join("baked.lmnkc");
        let (timeline, written) =
            bake_fixture_to_lmnk_file(&fixture_path, &dst_path, "phi4-mini", false)
                .expect("bake .lmnkc should succeed");
        assert!(written > 0);
        assert!(dst_path.is_file());

        let reloaded = LmnkTimeline::load_from_path(&dst_path).expect("reload baked .lmnkc");
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

        let pkg_path = base.join("out").join("baked.lmnkp");
        let (_, pkg_written) =
            bake_fixture_to_lmnk_file(&fixture_path, &pkg_path, "phi4-mini", true)
                .expect("bake .lmnkp package should succeed");
        assert!(pkg_written > 0);
        assert!(pkg_path.is_file());

        let pkg = LmnkPackageOwned::load_from_path(&pkg_path).expect("load baked .lmnkp");
        let view = pkg.view().expect("parse baked .lmnkp");
        let compiled_entry = view
            .require_aligned64_entry("compiled.lmnkc")
            .expect("compiled.lmnkc must be 64B aligned");
        assert_eq!(compiled_entry.data_offset % 64, 0);
        let pkg_tl = LmnkTimeline::load_from_path(&pkg_path).expect("load timeline from .lmnkp");
        assert_eq!(pkg_tl, timeline);

        let listed = collect_lmnk_paths_creating_dir(&base.join("out")).expect("list artifacts");
        assert_eq!(listed, vec![dst_path, pkg_path]);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn cli_keygen_sign_verify_tamper_and_reserved_path_end_to_end() {
        use super::{build_integrity_policy, run_keygen, run_sign_artifact, run_verify_artifact};
        use liminika_adapter::IntegrityTrustLevel;

        let base =
            std::env::temp_dir().join(format!("liminika_cli_integrity_e2e_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).expect("create temp test dir");

        let fixture_path = base.join("fixture.txt");
        let fixture_raw = include_str!("../../../tests/fixtures/gcso_eval_samples.txt");
        std::fs::write(&fixture_path, fixture_raw).expect("write fixture");

        let secret_key = base.join("keys").join("signing.key");
        let public_key = base.join("keys").join("signing.pub");
        let pk = run_keygen(&secret_key, &public_key).expect("keygen");
        assert!(secret_key.is_file());
        assert!(public_key.is_file());

        // Bake unsigned .lmnkc and .lmnkp artifacts.
        let unsigned_c = base.join("raw.lmnkc");
        let unsigned_p = base.join("raw.lmnkp");
        bake_fixture_to_lmnk_file(&fixture_path, &unsigned_c, "phi4-mini", false)
            .expect("bake lmnkc");
        bake_fixture_to_lmnk_file(&fixture_path, &unsigned_p, "phi4-mini", true)
            .expect("bake lmnkp");

        let strict_policy =
            build_integrity_policy(Some(public_key.to_str().unwrap()), false).expect("policy");
        assert!(run_verify_artifact(&unsigned_c, &strict_policy).is_err());
        assert!(run_verify_artifact(&unsigned_p, &strict_policy).is_err());

        // Sign both artifacts.
        let signed_c = base.join("signed.lmnkc");
        let signed_p = base.join("signed.lmnkp");
        let sum_c =
            run_sign_artifact(&unsigned_c, &signed_c, Some(&secret_key)).expect("sign lmnkc");
        assert_eq!(
            sum_c.report.trust_level,
            IntegrityTrustLevel::SignatureVerified
        );
        assert_eq!(sum_c.pubkey_hex.as_deref(), Some(pk.to_hex().as_str()));

        let sum_p =
            run_sign_artifact(&unsigned_p, &signed_p, Some(&secret_key)).expect("sign lmnkp");
        assert_eq!(
            sum_p.report.trust_level,
            IntegrityTrustLevel::SignatureVerified
        );

        // Verify both artifacts with public key file & hex string.
        let rep_c = run_verify_artifact(&signed_c, &strict_policy).expect("verify signed lmnkc");
        assert_eq!(rep_c.trust_level, IntegrityTrustLevel::SignatureVerified);
        let hex_policy = build_integrity_policy(Some(&pk.to_hex()), false).expect("hex policy");
        let rep_p = run_verify_artifact(&signed_p, &hex_policy).expect("verify signed lmnkp");
        assert_eq!(rep_p.trust_level, IntegrityTrustLevel::SignatureVerified);

        // Tamper 1 byte in both signed files -> verify must fail!
        let mut bad_c = std::fs::read(&signed_c).expect("read signed lmnkc");
        let mid_c = bad_c.len() / 2;
        bad_c[mid_c] ^= 0x01;
        let bad_c_path = base.join("tampered.lmnkc");
        std::fs::write(&bad_c_path, &bad_c).expect("write tampered lmnkc");
        assert!(run_verify_artifact(&bad_c_path, &strict_policy).is_err());

        let mut bad_p = std::fs::read(&signed_p).expect("read signed lmnkp");
        let mid_p = bad_p.len() / 2;
        bad_p[mid_p] ^= 0x01;
        let bad_p_path = base.join("tampered.lmnkp");
        std::fs::write(&bad_p_path, &bad_p).expect("write tampered lmnkp");
        assert!(run_verify_artifact(&bad_p_path, &strict_policy).is_err());

        // Windows reserved device name outputs rejected across keygen, sign, and bake!
        assert!(run_keygen(&base.join("CON.key"), &base.join("ok.pub")).is_err());
        assert!(
            run_sign_artifact(&unsigned_c, &base.join("NUL.lmnkc"), Some(&secret_key)).is_err()
        );
        assert!(bake_fixture_to_lmnk_file(
            &fixture_path,
            &base.join("aux.lmnkp"),
            "phi4-mini",
            true
        )
        .is_err());

        let _ = std::fs::remove_dir_all(&base);
    }
}
