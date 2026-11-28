// name: xtask/src/coupling_audit.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-blocking meta-audit radar (`--audit`) for core-primitive adoption and feedback loops.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::scanner::{
    QUARANTINED_CLI_FILES, extract_non_test_code, is_ident_cont, is_test_path, iter_idents,
    line_number_at, read_ident, rel_posix, skip_ws, strip_cfg_test_tail, walk_rs_files,
};

const DETERMINISTIC_INFRA_FILES: &[&str] = &[
    "src/adapter/src/package/mod.rs",
    "examples/candle_eval/src/metrics.rs",
    "examples/candle_eval/src/cli.rs",
    "examples/candle_eval/src/model/mod.rs",
    "examples/candle_eval/src/model/config.rs",
    "examples/candle_eval/src/model/rope.rs",
    "examples/candle_eval/src/model/layers.rs",
];

const CORE_PRIMITIVE_WORDS: &[&str] = &[
    "LacunaPartition8",
    "BitCut8",
    "BitCut32",
    "BitCut64",
    "Scoped",
    "Indivisible",
    "SystemSpineState",
    "EnvelopeState",
    "BoundarySignal",
    "DistortionTension",
    "EgoDifferentialNet",
    "CoreTelemetryBank",
    "StigmergicSubstrate",
    "PhaseAnchorRecord",
    "GatewayFrame",
    "GuardPhaseAnchor",
    "ObjectiveFact",
    "SubjectivePersona",
    "TurnOrchestrator",
    "TurnSteeringPlan",
];

const STATE_COUNTER_WORDS: &[&str] = &[
    "streak", "counter", "dwell", "retries", "epoch", "hits", "steps",
];

const CRITICAL_RETURN_METHODS: &[&str] = &[
    "consolidate_and_evaporate_cold_path",
    "step_decode_control",
    "prepare_turn",
    "apply_slow_loop_telemetry_epoch",
    "evaluate_slow_loop_telemetry",
    "evaluate_pframe_chain_branchless",
    "evaluate_pframe_chain_with_mask_branchless",
    "evaluate_ego_pframe_chain_mut",
    "step_sandbox_hypothesis_lifecycle",
    "apply_preset_lanes",
    "evaluate_policy",
    "update_assimilation_ema",
    "modulate_gateway_frame_at_scale",
    "evaluate_whole_token_bypass_with_spine",
    "step_compute_dispatch",
    "step_compute_dispatch_with_spine",
    "orchestrate_layer_rope_with_substrate",
    "orchestrate_layer_rope_with_spillover",
    "steer_layer_with_gateway",
    "steer_layer_with_substrate",
];

const CONTROL_FN_PREFIXES: &[&str] = &[
    "step",
    "evaluate",
    "apply",
    "consolidate",
    "settle",
    "orchestrate",
    "reanchor",
    "enforce",
    "adapt",
    "intervene",
    "begin_turn",
    "signal_",
    "configure_",
];

const DOWNWARD_SIGNAL_WORDS: &[&str] = &[
    "TurnSteeringPlan",
    "SpineStepPlan",
    "ComputeDispatchMask",
    "GatewayFrame",
    "CoreStateStack",
    "LacunaPartition8",
    "RecallPolicy",
    "CompiledTokenCorpus",
    "GuardPhaseAnchor",
    "ObjectiveFact",
    "SubjectivePersona",
    "EquilibriumState",
];

const UPWARD_FEEDBACK_WORDS: &[&str] = &[
    "step_decode_control",
    "commit_turn_circuit",
    "commit_compute_exit_reward",
    "apply_slow_loop_telemetry_epoch",
    "drain_crystallized_anchors",
    "record_step",
    "sync_core_layers_from_recall",
    "evaluate_ego_pframe_chain_mut",
    "consolidate_and_evaporate_cold_path",
    "step_sandbox_hypothesis_lifecycle",
    "reinforce_active_circuit_with_bank",
    "calibrate_manifold_probe",
    "signal_compute_instability",
    "update_environment_telemetry",
    "step_distortion_tension_hot_path",
    "step_working_set",
    "step_hot_path_branchless",
    "ingest_crystallized_anchors",
    "sync_equilibrium_state",
    "restore_from_equilibrium",
    "record_tension_feedback",
];

fn iter_prod_files(repo_root: &Path, dirs: &[&str]) -> Vec<(String, PathBuf)> {
    let mut out = Vec::new();
    for &rel_dir in dirs {
        let base = repo_root.join(rel_dir);
        if !base.is_dir() {
            continue;
        }
        for rs_file in walk_rs_files(&base) {
            if is_test_path(&rs_file) {
                continue;
            }
            if rel_dir.starts_with("src/cli")
                && rs_file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| QUARANTINED_CLI_FILES.contains(&n))
            {
                continue;
            }
            out.push((rel_posix(repo_root, &rs_file), rs_file));
        }
    }
    out
}

fn count_core_primitive_refs(code: &str) -> usize {
    let mut count = 0usize;
    for (offset, ident) in iter_idents(code) {
        let after = offset + ident.len();
        if ((ident == "liminika_core" || ident == "Lacuna") && code[after..].starts_with("::"))
            || CORE_PRIMITIVE_WORDS.contains(&ident)
        {
            count += 1;
        }
    }
    count
}

fn count_float_threshold_cmps(code: &str) -> usize {
    let bytes = code.as_bytes();
    let n = bytes.len();
    let mut count = 0usize;
    let mut i = 0;

    while i < n {
        let op_len = if i + 1 < n && matches!((bytes[i], bytes[i + 1]), (b'>', b'=') | (b'<', b'='))
        {
            2
        } else if matches!(bytes[i], b'>' | b'<') {
            1
        } else {
            i += 1;
            continue;
        };

        let mut j = skip_ws(bytes, i + op_len);
        if j < n && bytes[j] == b'-' {
            j += 1;
        }
        let d1_start = j;
        while j < n && bytes[j].is_ascii_digit() {
            j += 1;
        }
        if j > d1_start && j < n && bytes[j] == b'.' {
            j += 1;
            let d2_start = j;
            while j < n && bytes[j].is_ascii_digit() {
                j += 1;
            }
            if j > d2_start {
                if code[j..].starts_with("_f32") || code[j..].starts_with("_f64") {
                    j += 4;
                }
                count += 1;
                i = j;
                continue;
            }
        }
        i += op_len;
    }

    count
}

fn count_word_hits(code: &str, words: &[&str]) -> usize {
    iter_idents(code)
        .into_iter()
        .filter(|(_, w)| words.contains(w))
        .count()
}

fn audit_unintegrated_hotspots(repo_root: &Path) -> (Vec<String>, Vec<String>) {
    let mut warnings = Vec::new();
    let mut infra_notes = Vec::new();
    let scan_dirs = ["src/adapter/src", "src/cli/src", "examples/candle_eval/src"];

    for (rel_path, rs_file) in iter_prod_files(repo_root, &scan_dirs) {
        let raw = fs::read_to_string(&rs_file).expect("read audit file");
        let lines: Vec<&str> = raw.lines().collect();
        let prod_loc = strip_cfg_test_tail(&lines).len();
        if prod_loc < 400 {
            continue;
        }
        let code = extract_non_test_code(&raw);
        let core_refs = count_core_primitive_refs(&code);
        let float_cmps = count_float_threshold_cmps(&code);
        let counters = count_word_hits(&code, STATE_COUNTER_WORDS);

        if DETERMINISTIC_INFRA_FILES.contains(&rel_path.as_str()) {
            infra_notes.push(format!(
                "  [infra/deterministic] {rel_path} ({prod_loc} prod LOC, \
                 core_refs={core_refs}, float_cmps={float_cmps})"
            ));
            continue;
        }

        if core_refs == 0 && (float_cmps >= 10 || counters >= 10) {
            warnings.push(format!(
                "  [WARN:no-core-refs] {rel_path}: {prod_loc} prod LOC, \
                 core_refs=0, float_cmps={float_cmps}, state_tokens={counters}"
            ));
        } else if float_cmps >= 20 {
            let tag = if core_refs < 6 {
                "WARN:low-core-density"
            } else {
                "INFO:dense-f32-thresholds"
            };
            warnings.push(format!(
                "  [{tag}] {rel_path}: {prod_loc} prod LOC, \
                 core_refs={core_refs}, float_cmps={float_cmps}, state_tokens={counters}"
            ));
        }
    }

    (warnings, infra_notes)
}

fn audit_discarded_returns(repo_root: &Path) -> Vec<String> {
    let mut warnings = Vec::new();
    let scan_dirs = ["src/adapter/src", "src/cli/src", "examples/candle_eval/src"];

    for (rel_path, rs_file) in iter_prod_files(repo_root, &scan_dirs) {
        let raw = fs::read_to_string(&rs_file).expect("read file");
        let code = extract_non_test_code(&raw);
        let bytes = code.as_bytes();
        let n = bytes.len();
        let mut i = 0;

        while i < n {
            if (i == 0 || !is_ident_cont(bytes[i - 1]))
                && let Some(("let", after_let)) = read_ident(&code, i)
            {
                let ws1 = skip_ws(bytes, after_let);
                if ws1 > after_let
                    && let Some(("_", after_us)) = read_ident(&code, ws1)
                {
                    let ws2 = skip_ws(bytes, after_us);
                    if ws2 < n && bytes[ws2] == b'=' && bytes.get(ws2 + 1) != Some(&b'=') {
                        let expr_start = skip_ws(bytes, ws2 + 1);
                        if let Some(semi_rel) = code[expr_start..].find(';')
                            && (1..=220).contains(&semi_rel)
                        {
                            let expr = &code[expr_start..expr_start + semi_rel];
                            if let Some((_, hit)) = iter_idents(expr)
                                .into_iter()
                                .find(|(_, w)| CRITICAL_RETURN_METHODS.contains(w))
                            {
                                let lineno = line_number_at(&code, i);
                                warnings.push(format!(
                                    "  [WARN:discarded-return] {rel_path}:{lineno}: \
                                     `{hit}` discarded via `let _ =`"
                                ));
                            }
                            i = expr_start + semi_rel + 1;
                            continue;
                        }
                    }
                }
            }
            i += 1;
        }
    }

    warnings
}

fn find_pub_control_fns(code: &str) -> Vec<String> {
    let bytes = code.as_bytes();
    let n = bytes.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if (i == 0 || !is_ident_cont(bytes[i - 1]))
            && let Some(("pub", after_pub)) = read_ident(code, i)
        {
            let mut cur = skip_ws(bytes, after_pub);
            if cur > after_pub {
                if let Some(("const", after_const)) = read_ident(code, cur) {
                    let ws = skip_ws(bytes, after_const);
                    if ws > after_const {
                        cur = ws;
                    }
                }
                if let Some(("fn", after_fn)) = read_ident(code, cur) {
                    let ws = skip_ws(bytes, after_fn);
                    if ws > after_fn
                        && let Some((name, _)) = read_ident(code, ws)
                        && CONTROL_FN_PREFIXES.iter().any(|p| name.starts_with(p))
                        && name
                            .bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
                    {
                        out.push(name.to_string());
                    }
                }
            }
        }
        i += 1;
    }
    out
}

fn audit_unreferenced_core_control_surfaces(repo_root: &Path) -> Vec<String> {
    let core_src = repo_root.join("src/core/src");
    if !core_src.is_dir() {
        return Vec::new();
    }

    let mut downstream_counts: BTreeMap<String, usize> = BTreeMap::new();
    for (_rel, rs_file) in iter_prod_files(
        repo_root,
        &[
            "src/adapter/src",
            "src/cli/src",
            "examples/candle_eval/src",
            "benches",
        ],
    ) {
        let code = extract_non_test_code(&fs::read_to_string(&rs_file).expect("read downstream"));
        for (_, ident) in iter_idents(&code) {
            *downstream_counts.entry(ident.to_string()).or_insert(0) += 1;
        }
    }

    let mut defined: BTreeMap<String, String> = BTreeMap::new();
    let mut core_ident_counts: BTreeMap<String, usize> = BTreeMap::new();
    for rs_file in walk_rs_files(&core_src) {
        if is_test_path(&rs_file) {
            continue;
        }
        let code = extract_non_test_code(&fs::read_to_string(&rs_file).expect("read core file"));
        for (_, ident) in iter_idents(&code) {
            *core_ident_counts.entry(ident.to_string()).or_insert(0) += 1;
        }
        if rs_file
            .components()
            .any(|c| c.as_os_str().to_str() == Some("abi"))
        {
            continue;
        }
        let rel = rel_posix(&core_src, &rs_file);
        for name in find_pub_control_fns(&code) {
            if name.starts_with("lmnk_") {
                continue;
            }
            defined.entry(name).or_insert_with(|| rel.clone());
        }
    }

    let mut unreferenced = Vec::new();
    for (name, rel) in defined {
        if downstream_counts.contains_key(&name) {
            continue;
        }
        let core_hits = core_ident_counts.get(&name).copied().unwrap_or(0);
        if core_hits <= 1 {
            unreferenced.push(format!("  [INFO:core-control-uncalled] {rel}: fn {name}"));
        }
    }
    unreferenced
}

fn audit_module_flow_balance(repo_root: &Path) -> Vec<String> {
    let mut rows = Vec::new();
    let mut one_way = 0usize;
    let scan_dirs = ["src/adapter/src", "examples/candle_eval/src"];

    for (rel_path, rs_file) in iter_prod_files(repo_root, &scan_dirs) {
        if DETERMINISTIC_INFRA_FILES.contains(&rel_path.as_str()) {
            continue;
        }
        let code = extract_non_test_code(&fs::read_to_string(&rs_file).expect("read flow file"));
        let down = count_word_hits(&code, DOWNWARD_SIGNAL_WORDS);
        let up = count_word_hits(&code, UPWARD_FEEDBACK_WORDS);
        if down == 0 && up == 0 {
            continue;
        }
        let tag = if down > 0 && up == 0 {
            one_way += 1;
            "ONE-WAY"
        } else {
            "CLOSED"
        };
        rows.push(format!(
            "  {rel_path:<46} | down={down:>4} | up={up:>4} | {tag}"
        ));
    }

    if one_way > 0 {
        rows.push(format!(
            "  Summary: {one_way} one-way consumer module(s) without upward feedback."
        ));
    } else {
        rows.push("  Summary: 0 one-way consumer modules detected.".to_string());
    }
    rows
}

/// Renders the complete non-blocking Meta-Audit report (always exits 0).
pub fn render_coupling_audit(repo_root: &Path) -> String {
    let (hotspot_warns, infra_notes) = audit_unintegrated_hotspots(repo_root);
    let discard_warns = audit_discarded_returns(repo_root);
    let uncalled_core = audit_unreferenced_core_control_surfaces(repo_root);
    let flow_rows = audit_module_flow_balance(repo_root);

    let mut lines = vec![
        "=== Architecture Meta-Audit Radar (Non-Blocking Warnings, Exit 0) ===".to_string(),
        String::new(),
        "1. Unintegrated Control Hotspots (>= 400 prod LOC, low/zero core refs + dense float thresholds):"
            .to_string(),
    ];
    if hotspot_warns.is_empty() {
        lines.push("  None.".to_string());
    } else {
        lines.extend(hotspot_warns);
    }
    if !infra_notes.is_empty() {
        lines.push("  Exempt deterministic/IO/harness files:".to_string());
        lines.extend(infra_notes);
    }

    lines.push(String::new());
    lines.push("2. Discarded Control / Crystallization Returns (`let _ =`):".to_string());
    if discard_warns.is_empty() {
        lines.push("  None.".to_string());
    } else {
        lines.extend(discard_warns);
    }

    lines.push(String::new());
    lines.push(format!(
        "3. Public `core` Control Surfaces Unreferenced Downstream ({} total, showing up to 15):",
        uncalled_core.len()
    ));
    if uncalled_core.is_empty() {
        lines.push("  None.".to_string());
    } else {
        let shown = uncalled_core.len().min(15);
        lines.extend_from_slice(&uncalled_core[..shown]);
        if uncalled_core.len() > 15 {
            lines.push(format!("  ... ({} more)", uncalled_core.len() - 15));
        }
    }

    lines.push(String::new());
    lines.push("4. Module Downward-vs-Upward Feedback Balance:".to_string());
    lines.extend(flow_rows);
    lines.push(String::new());
    lines.join("\n")
}
