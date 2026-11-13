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
        }
    }

    // 5. Clean up allocated context resources
    unsafe { liminika_core::abi::gcso_context_destroy(ctx_handle) };
    Ok(())
}
