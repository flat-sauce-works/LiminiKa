// name: xtask/src/workspace_gates.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Workspace inter-crate gate, host LLM isolation, downstream vocabulary, and file-hygiene checks.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::core_dag::BANNED_LEGACY_DECL_TOKENS;
use crate::scanner::{
    QUARANTINED_CLI_FILES, extract_non_test_code, find_banned_public_decls,
    find_function_line_span, is_ident_cont, is_test_path, iter_idents, line_number_at, read_ident,
    rel_posix, skip_ws, split_ident_tokens, strip_cfg_test_tail, walk_py_files, walk_rs_files,
};

const MAX_PROD_LOC_PER_FILE: usize = 2000;

const DOWNSTREAM_EXTRA_BANNED_TOKENS: &[&str] = &["society", "domino"];

const ADAPTER_MUT_BACKDOORS: &[&str] = &[
    "runtime_context_mut",
    "substrate_mut",
    "compute_pool_mut",
    "entropy_controller_mut",
    "residual_adapter_mut",
];

const CORE_MUTABLE_TYPES: &[&str] = &[
    "RuntimeContext",
    "CoreStateStack",
    "SystemSpineState",
    "AnonymousComputePool",
    "EntropyController",
    "EgoDifferentialNet",
    "DistortionTension",
    "PhaseSteeringEngine",
    "ResidualAdapterEngine",
    "StigmergicSubstrate",
];

const CORE_MUTABLE_METHODS: &[&str] = &[
    "reanchor_baseline_pressure",
    "reset_turn_bypass_state",
    "begin_turn_spine_policy",
    "step_hot_path_branchless",
    "evaluate_whole_token_steering_bypass_with_spine",
    "apply_preset_lanes",
    "allocate_lane",
    "release_all_lanes",
];

const ADAPTER_GATE_ASSOCIATED_CALLS: &[(&str, &[&str])] = &[
    ("TurnOrchestrator", &["prepare_turn", "commit_turn_circuit"]),
    ("ModelSteeringAdapter", &["new"]),
    (
        "LmnkTimeline",
        &[
            "new",
            "load_from_path",
            "from_bytes",
            "save_to_path",
            "record_step",
            "resume_or_rebake",
        ],
    ),
    ("LmnkPackageOwned", &["load_from_path"]),
    (
        "CompiledTokenCorpus",
        &["capture_entries", "compile", "compile_with_base_prior"],
    ),
    ("ContrastiveAnchorBank", &["from_entries"]),
];

const ADAPTER_GATE_METHODS: &[&str] = &[
    "step_decode_control",
    "reanchor_compute_pressure",
    "reset_turn_decode_state",
    "orchestrate_layer_rope_with_substrate",
    "orchestrate_layer_rope_with_spillover",
    "steer_layer_with_gateway",
    "steer_layer_q_rot",
    "apply_rank1_with_gateway_mut",
    "apply_rank1_hidden_fused",
    "orchestrate_layer_rank1_with_substrate",
    "calibrate_manifold_probe",
    "apply_slow_loop_telemetry_epoch",
    "commit_spillover_experience",
    "reset_horizon_and_core_state",
    "step_compute_dispatch_with_spine",
    "commit_compute_exit_reward",
    "measure_entropy_state",
    "step_entropy_logits",
    "set_layer_rope_mask",
    "set_ego_max_hops",
    "set_guard_hold_tokens",
    "set_max_observe_stride",
    "set_pre_bypass_enabled",
    "configure_ego_from_preset",
    "configure_ego_from_natural_language",
    "bake_logical_pframe_topology",
    "drain_crystallized_anchors",
    "restore_runtime_snapshot",
    "sync_equilibrium_state",
    "restore_from_equilibrium",
];

const HOST_LLM_BANNED_TOKENS: &[&str] = &[
    "candle",
    "tokenizer",
    "tokenizers",
    "prompt",
    "prompts",
    "logit",
    "logits",
    "vocab",
    "rotary",
    "rope",
    "phi4",
    "phi3",
    "gguf",
    "safetensors",
];

const HOST_LLM_BANNED_IDENT_EXACT: &[&str] = &[
    "kv_cache",
    "compiledtokencorpus",
    "modelsteeringadapter",
    "turnorchestrator",
    "textattractorcorpus",
    "tokensequenceencoder",
    "turnkvcachestepping",
    "ropegeometry",
    "attractorrole",
];

const HOST_ADAPTER_LLM_MODULES: &[&str] = &[
    "pooling",
    "stratum",
    "corpus_recall",
    "model_adapter",
    "turn_orchestrator",
    "reflection",
    "ego_intent",
    "timeline",
];

const CORE_CRATE_LAYERS: &[&str] = &[
    "liminal",
    "foundation",
    "stigmergy",
    "drivers",
    "runtime",
    "abi",
];

fn find_dot_method_calls(code: &str, methods: &[&str]) -> Vec<usize> {
    let bytes = code.as_bytes();
    let n = bytes.len();
    let mut hits = Vec::new();
    let mut i = 0;
    while i < n {
        if bytes[i] == b'.' {
            let ws = skip_ws(bytes, i + 1);
            if let Some((name, after)) = read_ident(code, ws)
                && methods.contains(&name)
            {
                let paren = skip_ws(bytes, after);
                if paren < n && bytes[paren] == b'(' {
                    hits.push(i);
                    i = paren + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    hits
}

/// Counts canonical mutable adapter gate calls in stripped Rust code.
pub fn count_adapter_mut_gate_calls(code: &str) -> usize {
    let mut count = find_dot_method_calls(code, ADAPTER_GATE_METHODS).len();
    for (offset, ident) in iter_idents(code) {
        let after = offset + ident.len();
        if let Some((_, methods)) = ADAPTER_GATE_ASSOCIATED_CALLS
            .iter()
            .find(|&&(ty, _)| ty == ident)
            && code[after..].starts_with("::")
            && let Some((m, _)) = read_ident(code, after + 2)
            && methods.contains(&m)
        {
            count += 1;
        }
    }
    count
}

/// Counts `liminika_core::<layer>` references per layer in stripped Rust code.
pub fn count_core_crate_layer_refs(code: &str) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for (offset, ident) in iter_idents(code) {
        if ident == "liminika_core" {
            let after = offset + ident.len();
            if code[after..].starts_with("::")
                && let Some((layer, _)) = read_ident(code, after + 2)
                && let Some(&matched) = CORE_CRATE_LAYERS.iter().find(|&&l| l == layer)
            {
                *counts.entry(matched).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// Finds raw `lmnk_*(` C-ABI call sites in stripped Rust code.
pub fn find_raw_c_abi_calls(code: &str) -> Vec<usize> {
    let bytes = code.as_bytes();
    let mut hits = Vec::new();
    for (offset, ident) in iter_idents(code) {
        if let Some(rest) = ident.strip_prefix("lmnk_")
            && !rest.is_empty()
            && rest
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        {
            let ws = skip_ws(bytes, offset + ident.len());
            if ws < bytes.len() && bytes[ws] == b'(' {
                hits.push(offset);
            }
        }
    }
    hits
}

fn find_core_mutable_engine_refs(code: &str) -> Vec<usize> {
    let mut hits = find_dot_method_calls(code, CORE_MUTABLE_METHODS);
    for (offset, ident) in iter_idents(code) {
        let after = offset + ident.len();
        if ident == "liminika_core" && code[after..].starts_with("::") {
            if let Some((layer, after_l)) = read_ident(code, after + 2)
                && code[after_l..].starts_with("::")
                && let Some((ty, _)) = read_ident(code, after_l + 2)
            {
                let forbidden = match layer {
                    "runtime" => matches!(ty, "RuntimeContext" | "CoreStateStack"),
                    "drivers" => matches!(
                        ty,
                        "SystemSpineState"
                            | "AnonymousComputePool"
                            | "EntropyController"
                            | "EgoDifferentialNet"
                            | "DistortionTension"
                    ),
                    "stigmergy" => matches!(ty, "PhaseSteeringEngine" | "ResidualAdapterEngine"),
                    _ => false,
                };
                if forbidden {
                    hits.push(offset);
                }
            }
            continue;
        }
        if CORE_MUTABLE_TYPES.contains(&ident)
            && code[after..].starts_with("::")
            && let Some((ctor, _)) = read_ident(code, after + 2)
            && matches!(ctor, "new" | "new_boxed" | "new_empty")
        {
            hits.push(offset);
        }
    }
    hits.sort_unstable();
    hits.dedup();
    hits
}

fn find_core_abi_fn_paths(code: &str) -> Vec<(usize, String)> {
    let mut hits = Vec::new();
    for (offset, ident) in iter_idents(code) {
        if ident == "liminika_core" {
            let after = offset + ident.len();
            if code[after..].starts_with("::abi::")
                && let Some((sym, _)) = read_ident(code, after + "::abi::".len())
                && sym.starts_with("lmnk_")
                && sym
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            {
                hits.push((offset, sym.to_string()));
            }
        }
    }
    hits
}

/// Scans stripped Rust code under `host/` for disallowed LLM identifier tokens or imports.
pub fn scan_host_llm_isolation_in_source(
    code_only: &str,
    rel_path: &str,
    raw_lines: &[&str],
) -> Vec<String> {
    let mut violations = Vec::new();
    for (offset, ident) in iter_idents(code_only) {
        let lower_ident = ident.to_ascii_lowercase();
        let tokens = split_ident_tokens(ident);
        let banned_toks: Vec<&str> = tokens
            .iter()
            .filter_map(|t| {
                HOST_LLM_BANNED_TOKENS
                    .iter()
                    .copied()
                    .find(|&b| b == t.as_str())
            })
            .collect();
        let hit_exact = HOST_LLM_BANNED_IDENT_EXACT.contains(&lower_ident.as_str())
            || lower_ident.contains("kv_cache");
        if !banned_toks.is_empty() || hit_exact {
            let lineno = line_number_at(code_only, offset);
            let snippet = raw_lines
                .get(lineno.saturating_sub(1))
                .map(|s| s.trim())
                .unwrap_or("");
            let reason = if banned_toks.is_empty() {
                ident.to_string()
            } else {
                banned_toks.join(", ")
            };
            violations.push(format!(
                "Host LLM isolation (disallowed LLM token `{reason}` in `{ident}`): \
                 {rel_path}:{lineno}: {snippet}"
            ));
        }

        if ident == "crate" {
            let after = offset + ident.len();
            if code_only[after..].starts_with("::")
                && let Some((module, _)) = read_ident(code_only, after + 2)
                && HOST_ADAPTER_LLM_MODULES.contains(&module)
            {
                let lineno = line_number_at(code_only, offset);
                let snippet = raw_lines
                    .get(lineno.saturating_sub(1))
                    .map(|s| s.trim())
                    .unwrap_or("");
                violations.push(format!(
                    "Host LLM isolation (disallowed adapter import `crate::{module}`): \
                     {rel_path}:{lineno}: {snippet}"
                ));
            }
        }
    }
    violations
}

/// Enforces mechanical isolation of `src/adapter/src/host/` from LLM vocabulary and modules.
pub fn check_host_llm_isolation(repo_root: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    let host_dir = repo_root.join("src/adapter/src/host");
    if !host_dir.is_dir() {
        return violations;
    }
    for rs_file in walk_rs_files(&host_dir) {
        if is_test_path(&rs_file) {
            continue;
        }
        let rel = rel_posix(repo_root, &rs_file);
        let raw = fs::read_to_string(&rs_file).expect("read host file");
        let code_only = extract_non_test_code(&raw);
        let raw_lines: Vec<&str> = raw.lines().collect();
        violations.extend(scan_host_llm_isolation_in_source(
            &code_only, &rel, &raw_lines,
        ));
    }
    violations
}

/// Verifies legacy declared-identifier bans across `src/adapter/src/` and `src/cli/src/`.
pub fn check_downstream_vocabulary(repo_root: &Path) -> Vec<String> {
    let mut banned: Vec<&str> = BANNED_LEGACY_DECL_TOKENS.to_vec();
    banned.extend_from_slice(DOWNSTREAM_EXTRA_BANNED_TOKENS);

    let mut violations = Vec::new();
    for krate in ["adapter", "cli"] {
        let crate_src = repo_root.join("src").join(krate).join("src");
        if !crate_src.is_dir() {
            continue;
        }
        for rs_file in walk_rs_files(&crate_src) {
            if is_test_path(&rs_file) {
                continue;
            }
            if krate == "cli"
                && rs_file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| QUARANTINED_CLI_FILES.contains(&n))
            {
                continue;
            }
            let rel = rel_posix(repo_root, &rs_file);
            let raw = fs::read_to_string(&rs_file).expect("read downstream file");
            let code_only = extract_non_test_code(&raw);
            violations.extend(find_banned_public_decls(
                &code_only,
                &rel,
                &banned,
                "Downstream vocabulary",
            ));
        }
    }
    violations
}

/// Enforces inter-crate boundaries across `src/dsl/`, `src/cli/`, and `examples/candle_eval/`.
pub fn check_workspace_crate_gates(repo_root: &Path) -> Vec<String> {
    let mut violations = Vec::new();

    let dsl_src = repo_root.join("src/dsl/src");
    if dsl_src.is_dir() {
        for rs_file in walk_rs_files(&dsl_src) {
            if is_test_path(&rs_file) {
                continue;
            }
            let rel = rel_posix(repo_root, &rs_file);
            let raw = fs::read_to_string(&rs_file).expect("read dsl file");
            let code_only = extract_non_test_code(&raw);
            for (offset, ident) in iter_idents(&code_only) {
                if ident == "liminika_core" {
                    let lineno = line_number_at(&code_only, offset);
                    violations.push(format!(
                        "Workspace gate (dsl -> liminika_core): {rel}:{lineno}: \
                         disallowed `liminika_core` reference in `liminika-dsl`"
                    ));
                }
            }
        }
    }

    let host_roots = [
        ("cli", repo_root.join("src/cli/src")),
        ("candle_eval", repo_root.join("examples/candle_eval/src")),
    ];
    for (crate_tag, src_dir) in host_roots {
        if !src_dir.is_dir() {
            continue;
        }
        for rs_file in walk_rs_files(&src_dir) {
            if is_test_path(&rs_file) {
                continue;
            }
            if crate_tag == "cli"
                && rs_file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| QUARANTINED_CLI_FILES.contains(&n))
            {
                continue;
            }
            let rel = rel_posix(repo_root, &rs_file);
            let raw = fs::read_to_string(&rs_file).expect("read host root file");
            let code_only = extract_non_test_code(&raw);
            let raw_lines: Vec<&str> = raw.lines().collect();

            let smoke_allowed = if rel == "src/cli/src/main.rs" {
                find_function_line_span(&code_only, "run_default_c_abi_smoke")
            } else {
                Default::default()
            };

            let snippet_at = |lineno: usize| -> &str {
                raw_lines
                    .get(lineno.saturating_sub(1))
                    .map(|s| s.trim())
                    .unwrap_or("")
            };

            for offset in find_dot_method_calls(&code_only, ADAPTER_MUT_BACKDOORS) {
                let lineno = line_number_at(&code_only, offset);
                violations.push(format!(
                    "Workspace gate (adapter *_mut backdoor): {rel}:{lineno}: {}",
                    snippet_at(lineno)
                ));
            }
            for offset in find_core_mutable_engine_refs(&code_only) {
                let lineno = line_number_at(&code_only, offset);
                if smoke_allowed.contains(&lineno) {
                    continue;
                }
                violations.push(format!(
                    "Workspace gate (direct core mutable state): {rel}:{lineno}: {}",
                    snippet_at(lineno)
                ));
            }
            for offset in find_raw_c_abi_calls(&code_only) {
                let lineno = line_number_at(&code_only, offset);
                if smoke_allowed.contains(&lineno) {
                    continue;
                }
                violations.push(format!(
                    "Workspace gate (raw C-ABI call outside allowlist): {rel}:{lineno}: {}",
                    snippet_at(lineno)
                ));
            }
            for (offset, sym) in find_core_abi_fn_paths(&code_only) {
                if sym.ends_with("_t") {
                    continue;
                }
                let lineno = line_number_at(&code_only, offset);
                if smoke_allowed.contains(&lineno) {
                    continue;
                }
                violations.push(format!(
                    "Workspace gate (raw C-ABI symbol outside allowlist): {rel}:{lineno}: {}",
                    snippet_at(lineno)
                ));
            }
        }
    }

    let _ = is_ident_cont;
    violations
}

/// Verifies `< 2,000` prod LOC/file, 2-line SPDX headers, and pure LF across Rust trees
/// (including `xtask/src/`) and `scripts/*.py`.
pub fn check_file_hygiene_and_loc(repo_root: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    let rust_roots = [
        repo_root.join("src"),
        repo_root.join("examples/candle_eval/src"),
        repo_root.join("xtask/src"),
    ];
    for root in rust_roots {
        if !root.is_dir() {
            continue;
        }
        for rs_file in walk_rs_files(&root) {
            let rel = rel_posix(repo_root, &rs_file);
            let raw_bytes = fs::read(&rs_file).expect("read rs bytes");
            if raw_bytes.windows(2).any(|w| w == b"\r\n") {
                violations.push(format!("Line ending (CRLF): {rel}"));
            }
            let text = String::from_utf8(raw_bytes).expect("valid utf-8 rs file");
            let lines: Vec<&str> = text.lines().collect();
            let expected_l1 = format!("// name: {rel}");
            if lines.first().map(|s| s.trim_end()) != Some(expected_l1.as_str()) {
                violations.push(format!(
                    "Header mismatch (line 1): {rel} (expected `{expected_l1}`)"
                ));
            }
            if lines
                .get(1)
                .is_none_or(|l| !l.contains("SPDX-License-Identifier: MIT OR Apache-2.0"))
            {
                violations.push(format!("SPDX header missing (line 2): {rel}"));
            }
            if !is_test_path(&rs_file) {
                let prod_loc = strip_cfg_test_tail(&lines).len();
                if prod_loc >= MAX_PROD_LOC_PER_FILE {
                    violations.push(format!(
                        "Prod LOC ceiling (>= {MAX_PROD_LOC_PER_FILE}): {rel}: {prod_loc} prod LOC"
                    ));
                }
            }
        }
    }

    let scripts_dir = repo_root.join("scripts");
    if scripts_dir.is_dir() {
        for py_file in walk_py_files(&scripts_dir) {
            let rel = rel_posix(repo_root, &py_file);
            let raw_bytes = fs::read(&py_file).expect("read py bytes");
            if raw_bytes.windows(2).any(|w| w == b"\r\n") {
                violations.push(format!("Line ending (CRLF): {rel}"));
            }
            let text = String::from_utf8(raw_bytes).expect("valid utf-8 py file");
            let lines: Vec<&str> = text.lines().collect();
            let expected_l1 = format!("# name: {rel}");
            if lines.first().map(|s| s.trim_end()) != Some(expected_l1.as_str()) {
                violations.push(format!(
                    "Header mismatch (line 1): {rel} (expected `{expected_l1}`)"
                ));
            }
            if lines
                .get(1)
                .is_none_or(|l| !l.contains("SPDX-License-Identifier: MIT OR Apache-2.0"))
            {
                violations.push(format!("SPDX header missing (line 2): {rel}"));
            }
        }
    }

    violations
}

/// Runs all downstream vocabulary, host isolation, workspace gate, and file-hygiene checks.
pub fn run_workspace_checks(repo_root: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    violations.extend(check_downstream_vocabulary(repo_root));
    violations.extend(check_host_llm_isolation(repo_root));
    violations.extend(check_workspace_crate_gates(repo_root));
    violations.extend(check_file_hygiene_and_loc(repo_root));
    violations
}
