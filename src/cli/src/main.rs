// name: src/cli/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

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
        }
    }

    // 5. Clean up allocated context resources
    unsafe { liminika_core::abi::gcso_context_destroy(ctx_handle) };
    Ok(())
}
