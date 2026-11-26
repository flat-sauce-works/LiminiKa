// name: xtask/src/mermaid.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mermaid DAG/workspace/liminal diagram renderers and core hotspot summaries.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::scanner::{
    QUARANTINED_CLI_FILES, is_test_path, iter_idents, read_ident, rel_posix, scan_crate_prod_stats,
    skip_ws, strip_cfg_test_tail, walk_rs_files,
};
use crate::workspace_gates::{
    count_adapter_mut_gate_calls, count_core_crate_layer_refs, find_raw_c_abi_calls,
};

type EdgeCounts = BTreeMap<(String, String), usize>;

/// Maps a core source file to its Y-shaped DAG node bucket.
pub fn bucket_for_path(core_src: &Path, rs_file: &Path) -> Option<&'static str> {
    let rel = rs_file.strip_prefix(core_src).ok()?;
    let mut comps = rel.components().filter_map(|c| c.as_os_str().to_str());
    let top = comps.next()?;
    if top == "drivers" {
        return match comps.next() {
            Some("spine") => Some("drivers::spine"),
            Some("ego") => Some("drivers::ego"),
            Some("envelope") => Some("drivers::envelope"),
            _ => Some("drivers"),
        };
    }
    match top {
        "liminal" => Some("liminal"),
        "foundation" => Some("foundation"),
        "stigmergy" => Some("stigmergy"),
        "runtime" => Some("runtime"),
        _ => None,
    }
}

fn find_crate_layer_targets(code: &str) -> Vec<&'static str> {
    let mut targets = Vec::new();
    for (offset, ident) in iter_idents(code) {
        if ident != "crate" {
            continue;
        }
        let after = offset + ident.len();
        if !code[after..].starts_with("::") {
            continue;
        }
        let Some((layer, after_l)) = read_ident(code, after + 2) else {
            continue;
        };
        match layer {
            "liminal" => targets.push("liminal"),
            "foundation" => targets.push("foundation"),
            "stigmergy" => targets.push("stigmergy"),
            "runtime" => targets.push("runtime"),
            "drivers" => {
                if code[after_l..].starts_with("::")
                    && let Some((sub, _)) = read_ident(code, after_l + 2)
                {
                    match sub {
                        "spine" => targets.push("drivers::spine"),
                        "ego" => targets.push("drivers::ego"),
                        "envelope" => targets.push("drivers::envelope"),
                        _ => targets.push("drivers"),
                    }
                } else {
                    targets.push("drivers");
                }
            }
            _ => {}
        }
    }
    targets
}

/// Collects per-layer file counts, LOC, and inter-layer `crate::` import edge counts.
pub fn collect_dag_metrics(
    core_src: &Path,
) -> (BTreeMap<String, usize>, BTreeMap<String, usize>, EdgeCounts) {
    let mut files_per_node: BTreeMap<String, usize> = BTreeMap::new();
    let mut loc_per_node: BTreeMap<String, usize> = BTreeMap::new();
    let mut edge_counts: EdgeCounts = BTreeMap::new();

    for rs_file in walk_rs_files(core_src) {
        let Some(src_bucket) = bucket_for_path(core_src, &rs_file) else {
            continue;
        };
        let text = fs::read_to_string(&rs_file).expect("read core file");
        let lines: Vec<&str> = text.lines().collect();
        *files_per_node.entry(src_bucket.to_string()).or_insert(0) += 1;
        *loc_per_node.entry(src_bucket.to_string()).or_insert(0) += lines.len();
        for (_lineno, line) in strip_cfg_test_tail(&lines) {
            let code = line.split("//").next().unwrap_or("");
            for target in find_crate_layer_targets(code) {
                let dst_bucket = if matches!(
                    target,
                    "drivers::spine" | "drivers::ego" | "drivers::envelope"
                ) {
                    target
                } else {
                    target.split("::").next().unwrap_or(target)
                };
                if dst_bucket != src_bucket {
                    *edge_counts
                        .entry((src_bucket.to_string(), dst_bucket.to_string()))
                        .or_insert(0) += 1;
                }
            }
        }
    }

    (files_per_node, loc_per_node, edge_counts)
}

const ORDERED_CORE_PAIRS: &[(&str, &str)] = &[
    ("liminal", "foundation"),
    ("liminal", "stigmergy"),
    ("foundation", "stigmergy"),
    ("liminal", "drivers::spine"),
    ("foundation", "drivers::spine"),
    ("stigmergy", "drivers::spine"),
    ("liminal", "drivers::ego"),
    ("foundation", "drivers::ego"),
    ("stigmergy", "drivers::ego"),
    ("liminal", "drivers::envelope"),
    ("foundation", "drivers::envelope"),
    ("stigmergy", "drivers::envelope"),
    ("drivers", "drivers::spine"),
    ("drivers", "drivers::ego"),
    ("drivers", "drivers::envelope"),
    ("liminal", "runtime"),
    ("foundation", "runtime"),
    ("stigmergy", "runtime"),
    ("drivers::spine", "runtime"),
    ("drivers::ego", "runtime"),
    ("drivers::envelope", "runtime"),
    ("drivers", "runtime"),
];

fn node_id(key: &str) -> &'static str {
    match key {
        "liminal" => "L0_liminal",
        "foundation" => "L1_foundation",
        "stigmergy" => "L2_stigmergy",
        "drivers::spine" => "L3_spine",
        "drivers::ego" => "L3_ego",
        "drivers::envelope" => "L3_envelope",
        "drivers" => "L3_drivers_hub",
        "runtime" => "L4_runtime",
        _ => "L_unknown",
    }
}

/// Renders the live Y-shaped 5-layer DAG as a Mermaid `flowchart TD` diagram.
pub fn render_mermaid(core_src: &Path) -> String {
    let (files_per_node, loc_per_node, edge_counts) = collect_dag_metrics(core_src);
    let node_label = |key: &str, title: &str| -> String {
        let f = files_per_node.get(key).copied().unwrap_or(0);
        let l = loc_per_node.get(key).copied().unwrap_or(0);
        format!("{title} - {f} files, {l} LOC")
    };

    let mut lines = vec![
        "flowchart TD".to_string(),
        format!(
            "    L0_liminal[\"{}\"]",
            node_label("liminal", "Lv.0 liminal (Boundary Axioms & AbstractDomain)")
        ),
        format!(
            "    L1_foundation[\"{}\"]",
            node_label(
                "foundation",
                "Lv.1 foundation (Pure SWAR, RoPE & Cohomology)"
            )
        ),
        format!(
            "    L2_stigmergy[\"{}\"]",
            node_label("stigmergy", "Lv.2 stigmergy (Substrate & PhaseAnchors)")
        ),
        "    subgraph L3_drivers[\"Lv.3 drivers (Orthogonal Axes & Tension)\"]".to_string(),
        format!(
            "        L3_spine[\"{}\"]",
            node_label(
                "drivers::spine",
                "drivers/spine (Reflex & EntropyController)"
            )
        ),
        format!(
            "        L3_ego[\"{}\"]",
            node_label("drivers::ego", "drivers/ego (Intent & EgoDifferentialNet)")
        ),
        format!(
            "        L3_envelope[\"{}\"]",
            node_label(
                "drivers::envelope",
                "drivers/envelope (Bounds & EnvelopeState)"
            )
        ),
        format!(
            "        L3_drivers_hub[\"{}\"]",
            node_label("drivers", "drivers/mod (DistortionTension)")
        ),
        "    end".to_string(),
        format!(
            "    L4_runtime[\"{}\"]",
            node_label("runtime", "Lv.4 runtime (Context, Gateway, Horizon, ABI)")
        ),
        String::new(),
    ];

    for &(provider, consumer) in ORDERED_CORE_PAIRS {
        let refs = edge_counts
            .get(&(consumer.to_string(), provider.to_string()))
            .copied()
            .unwrap_or(0);
        if refs > 0 {
            lines.push(format!(
                "    {} -->|{refs} refs| {}",
                node_id(provider),
                node_id(consumer)
            ));
        }
    }

    lines.push(String::new());
    lines.join("\n")
}

fn plural_files(count: usize) -> String {
    if count == 1 {
        "1 file".to_string()
    } else {
        format!("{count} files")
    }
}

/// Renders workspace crates -> adapter canonical gates -> core 5-layer DAG as Mermaid.
pub fn render_mermaid_workspace(repo_root: &Path) -> String {
    let core_src = repo_root.join("src/core/src");
    let (files_per_node, loc_per_node, core_edges) = collect_dag_metrics(&core_src);

    let (dsl_files, dsl_loc, _) = scan_crate_prod_stats(&repo_root.join("src/dsl/src"), &[]);
    let (cli_files, cli_loc, cli_code) =
        scan_crate_prod_stats(&repo_root.join("src/cli/src"), QUARANTINED_CLI_FILES);
    let (eval_files, eval_loc, eval_code) =
        scan_crate_prod_stats(&repo_root.join("examples/candle_eval/src"), &[]);
    let (ad_files, ad_loc, ad_code) =
        scan_crate_prod_stats(&repo_root.join("src/adapter/src"), &[]);
    let (host_files, host_loc, _) =
        scan_crate_prod_stats(&repo_root.join("src/adapter/src/host"), &[]);

    let cli_dsl_refs = iter_idents(&cli_code)
        .into_iter()
        .filter(|(_, w)| *w == "liminika_dsl")
        .count();
    let cli_adapter_gates = count_adapter_mut_gate_calls(&cli_code);
    let cli_core_vals: usize = count_core_crate_layer_refs(&cli_code).values().sum();
    let cli_cabi_smoke = find_raw_c_abi_calls(&cli_code).len();

    let eval_adapter_gates = count_adapter_mut_gate_calls(&eval_code);
    let eval_core_vals: usize = count_core_crate_layer_refs(&eval_code).values().sum();

    let ad_core = count_core_crate_layer_refs(&ad_code);
    let ad_runtime_gates = ad_core.get("runtime").copied().unwrap_or(0);
    let ad_liminal_vals = ad_core.get("liminal").copied().unwrap_or(0);
    let ad_foundation_vals = ad_core.get("foundation").copied().unwrap_or(0);
    let ad_stigmergy_vals = ad_core.get("stigmergy").copied().unwrap_or(0);
    let ad_drivers_vals = ad_core.get("drivers").copied().unwrap_or(0);

    let core_label = |key: &str, title: &str| -> String {
        let f = files_per_node.get(key).copied().unwrap_or(0);
        let l = loc_per_node.get(key).copied().unwrap_or(0);
        format!("{title} - {}, {l} LOC", plural_files(f))
    };

    let mut lines = vec![
        "flowchart TD".to_string(),
        "    %% Legend: solid arrow (-->) = Mutable State Gate; dashed arrow (-.->) = Value/Pod Ref"
            .to_string(),
        "    subgraph WS_HOSTS[\"Workspace Host Crates & Evaluation Harness\"]".to_string(),
        format!(
            "        W_dsl[\"liminika-dsl (Parser/AST) - {}, {dsl_loc} prod LOC\"]",
            plural_files(dsl_files)
        ),
        format!(
            "        W_cli[\"liminika-cli (CLI & C-ABI Smoke) - {}, {cli_loc} prod LOC\"]",
            plural_files(cli_files)
        ),
        format!(
            "        W_eval[\"candle_eval (Thin Eval Harness) - {}, {eval_loc} prod LOC\"]",
            plural_files(eval_files)
        ),
        "    end".to_string(),
        "    subgraph WS_ADAPTER[\"Canonical Adapter Gate (liminika-adapter)\"]".to_string(),
        format!(
            "        W_adapter[\"ModelSteeringAdapter + TurnOrchestrator + CompiledTokenCorpus - {}, {} prod LOC\"]",
            plural_files(ad_files.saturating_sub(host_files)),
            ad_loc.saturating_sub(host_loc)
        ),
        format!(
            "        W_host[\"App-Agnostic Host (host/ - LLM-isolated) - {}, {host_loc} prod LOC\"]",
            plural_files(host_files)
        ),
        "    end".to_string(),
        "    subgraph WS_CORE[\"liminika-core (#![no_std] 5-Layer Unidirectional DAG)\"]".to_string(),
        format!(
            "        L0_liminal[\"{}\"]",
            core_label("liminal", "Lv.0 liminal (Axioms & Lacuna)")
        ),
        format!(
            "        L1_foundation[\"{}\"]",
            core_label("foundation", "Lv.1 foundation (SWAR, RoPE, Pod)")
        ),
        format!(
            "        L2_stigmergy[\"{}\"]",
            core_label("stigmergy", "Lv.2 stigmergy (Substrate & Anchors)")
        ),
        "        subgraph WS_L3[\"Lv.3 drivers (Orthogonal Axes & Tension)\"]".to_string(),
        format!(
            "            L3_spine[\"{}\"]",
            core_label("drivers::spine", "Lv.3 drivers/spine (Spine SSOT)")
        ),
        format!(
            "            L3_ego[\"{}\"]",
            core_label("drivers::ego", "Lv.3 drivers/ego (Ego P-Frame)")
        ),
        format!(
            "            L3_envelope[\"{}\"]",
            core_label(
                "drivers::envelope",
                "Lv.3 drivers/envelope (Safety Envelope)"
            )
        ),
        format!(
            "            L3_drivers_hub[\"{}\"]",
            core_label("drivers", "Lv.3 drivers/mod (Tension)")
        ),
        "        end".to_string(),
        format!(
            "        L4_runtime[\"{}\"]",
            core_label("runtime", "Lv.4 runtime (Context, Gateway, ABI)")
        ),
        "    end".to_string(),
        String::new(),
    ];

    if cli_dsl_refs > 0 {
        lines.push(format!(
            "    W_cli -.->|\"Value/AST ({cli_dsl_refs} refs)\"| W_dsl"
        ));
    }
    lines.push(format!(
        "    W_cli -->|\"Mutable Gate ({cli_adapter_gates} calls)\"| W_adapter"
    ));
    if cli_core_vals > 0 {
        lines.push(format!(
            "    W_cli -.->|\"Value/Pod & Inspect ({cli_core_vals} refs)\"| L4_runtime"
        ));
    }
    if cli_cabi_smoke > 0 {
        lines.push(format!(
            "    W_cli -.->|\"C-ABI Smoke Allowlist ({cli_cabi_smoke} calls)\"| L4_runtime"
        ));
    }
    lines.push(format!(
        "    W_eval -->|\"Mutable Gate ({eval_adapter_gates} calls)\"| W_adapter"
    ));
    if eval_core_vals > 0 {
        lines.push(format!(
            "    W_eval -.->|\"Value/Pod & Anchors ({eval_core_vals} refs)\"| L4_runtime"
        ));
    }
    lines.push(format!(
        "    W_adapter -->|\"Mutable RuntimeContext Gate ({ad_runtime_gates} refs)\"| L4_runtime"
    ));
    if host_files > 0 {
        lines.push("    W_adapter -.->|\"Host Pager & Equilibrium Bridge\"| W_host".to_string());
        lines.push("    W_host -.->|\"Core Pod & Container View\"| L4_runtime".to_string());
    }
    if ad_liminal_vals > 0 {
        lines.push(format!(
            "    W_adapter -.->|\"Value/Pod ({ad_liminal_vals} refs)\"| L0_liminal"
        ));
    }
    if ad_foundation_vals > 0 {
        lines.push(format!(
            "    W_adapter -.->|\"Value/Trait ({ad_foundation_vals} refs)\"| L1_foundation"
        ));
    }
    if ad_stigmergy_vals > 0 {
        lines.push(format!(
            "    W_adapter -.->|\"Value/Trait ({ad_stigmergy_vals} refs)\"| L2_stigmergy"
        ));
    }
    if ad_drivers_vals > 0 {
        lines.push(format!(
            "    W_adapter -.->|\"Value/Pod ({ad_drivers_vals} refs)\"| L3_spine"
        ));
    }

    for &(provider, consumer) in ORDERED_CORE_PAIRS {
        let refs = core_edges
            .get(&(consumer.to_string(), provider.to_string()))
            .copied()
            .unwrap_or(0);
        if refs > 0 {
            lines.push(format!(
                "    {} -->|{refs} refs| {}",
                node_id(provider),
                node_id(consumer)
            ));
        }
    }

    lines.push("    L2_stigmergy ~~~ L3_drivers_hub".to_string());
    lines.push("    L3_spine ~~~ L4_runtime".to_string());
    lines.push("    L3_ego ~~~ L4_runtime".to_string());
    lines.push("    L3_envelope ~~~ L4_runtime".to_string());
    lines.push(String::new());
    lines.join("\n")
}

/// Formats top core source files by non-test LOC, unsafe tokens, raw bitmask ops, and axiom surface.
pub fn format_hotspots(core_src: &Path, top_n: usize) -> String {
    let mut rows: Vec<(usize, usize, usize, usize, usize, String)> = Vec::new();
    for rs_file in walk_rs_files(core_src) {
        if is_test_path(&rs_file) {
            continue;
        }
        let text = fs::read_to_string(&rs_file).expect("read core file");
        let lines: Vec<&str> = text.lines().collect();
        let total_loc = lines.len();
        let non_test = strip_cfg_test_tail(&lines);
        let prod_loc = non_test.len();
        let mut unsafe_hits = 0usize;
        let mut raw_mask_hits = 0usize;
        let mut scoped_refs = 0usize;
        for (_lineno, line) in non_test {
            let code = line.split("//").next().unwrap_or("");
            let idents = iter_idents(code);
            if idents.iter().any(|(_, w)| *w == "unsafe") {
                unsafe_hits += 1;
            }
            if code.contains("& !") {
                raw_mask_hits += 1;
            }
            if idents
                .iter()
                .any(|(_, w)| matches!(*w, "Scoped" | "Indivisible"))
            {
                scoped_refs += 1;
            }
        }
        let rel = rel_posix(core_src, &rs_file);
        rows.push((
            prod_loc,
            total_loc,
            unsafe_hits,
            raw_mask_hits,
            scoped_refs,
            rel,
        ));
    }

    rows.sort_by(|a, b| b.cmp(a));
    let shown = top_n.min(rows.len());
    let sep = "-".repeat(92);
    let mut out = vec![
        format!("Core Architecture Hotspots (top {shown} production files by non-test LOC):"),
        format!(
            "{:<46} | {:>7} | {:>8} | {:>6} | {:>5} | {:>5}",
            "File", "ProdLOC", "TotalLOC", "Unsafe", "Raw&!", "Axiom"
        ),
        sep.clone(),
    ];
    for (prod_loc, total_loc, unsafe_hits, raw_mask_hits, scoped_refs, rel) in &rows[..shown] {
        out.push(format!(
            "{rel:<46} | {prod_loc:>7} | {total_loc:>8} | {unsafe_hits:>6} | {raw_mask_hits:>5} | {scoped_refs:>5}"
        ));
    }
    let total_files = rows.len();
    let total_prod: usize = rows.iter().map(|r| r.0).sum();
    let total_all: usize = rows.iter().map(|r| r.1).sum();
    let total_unsafe: usize = rows.iter().map(|r| r.2).sum();
    out.push(sep);
    out.push(format!(
        "Summary: {total_files} prod files | {total_prod} prod LOC ({total_all} total LOC) | \
         {total_unsafe} non-test unsafe tokens"
    ));
    out.push(String::new());
    out.join("\n")
}

/// Metrics backing the static `liminal` Abstraction <-> Concretion map.
#[derive(Default)]
pub struct LiminalMapMetrics {
    /// Count of non-test files under `foundation/`.
    pub foundation_files: usize,
    /// Non-test LOC under `foundation/`.
    pub foundation_prod_loc: usize,
    /// Count of files under `liminal/axiom/`.
    pub axiom_files: usize,
    /// Reference count to axiom boundary gates and `BitCut`.
    pub axiom_gate_refs: usize,
    /// Reference count to `Grounded` I-Frame anchor carriers.
    pub grounded_iframe_refs: usize,
    /// Reference count to `Active` concretion engines.
    pub active_concretion_refs: usize,
    /// Reference count to abstract bypass paths.
    pub abstract_bypass_refs: usize,
    /// Reference count to exterior sandbox lanes.
    pub exterior_sandbox_refs: usize,
    /// Reference count to P-Frame Ego differential operators.
    pub pframe_ego_refs: usize,
    /// Reference count to runtime container/spine gates.
    pub runtime_gate_refs: usize,
}

/// Scans live `src/core/src` production files for Abstraction <-> Concretion metrics.
pub fn collect_liminal_map_metrics(core_src: &Path) -> LiminalMapMetrics {
    let mut m = LiminalMapMetrics::default();

    for rs_file in walk_rs_files(core_src) {
        if is_test_path(&rs_file) {
            continue;
        }
        let rel = rel_posix(core_src, &rs_file);
        let text = fs::read_to_string(&rs_file).expect("read core file");
        let lines: Vec<&str> = text.lines().collect();
        let non_test = strip_cfg_test_tail(&lines);
        if rel.starts_with("foundation/") {
            m.foundation_files += 1;
            m.foundation_prod_loc += non_test.len();
        }
        if rel.starts_with("liminal/axiom/") {
            m.axiom_files += 1;
        }
        for (_lineno, line) in non_test {
            let code = line.split("//").next().unwrap_or("");
            let bytes = code.as_bytes();
            for (offset, ident) in iter_idents(code) {
                let after = offset + ident.len();
                if matches!(
                    ident,
                    "Scoped"
                        | "Indivisible"
                        | "cut_by_lacuna"
                        | "bridge_to_shared"
                        | "BitCut"
                        | "BitCut8"
                        | "BitCut32"
                        | "BitCut64"
                ) || ident.starts_with("rescope_from_")
                {
                    m.axiom_gate_refs += 1;
                }
                if matches!(
                    ident,
                    "Grounded"
                        | "PhaseAnchorRecord"
                        | "PhaseProfile64"
                        | "PhaseProfileRef64"
                        | "PhaseWordsRef64"
                ) {
                    m.grounded_iframe_refs += 1;
                }
                if matches!(
                    ident,
                    "ActiveScope"
                        | "PhaseSteeringEngine"
                        | "ResidualAdapterEngine"
                        | "PhaseDeltaSlice"
                        | "ResidualScale"
                ) {
                    m.active_concretion_refs += 1;
                } else if ident == "Indivisible" {
                    let ws1 = skip_ws(bytes, after);
                    if ws1 < bytes.len() && bytes[ws1] == b'<' {
                        let ws2 = skip_ws(bytes, ws1 + 1);
                        if read_ident(code, ws2).is_some_and(|(w, _)| w == "Active") {
                            m.active_concretion_refs += 1;
                        }
                    }
                }
                if matches!(
                    ident,
                    "stationary_shortcut_ready" | "whole_token_steering_bypass"
                ) {
                    m.abstract_bypass_refs += 1;
                } else if ident == "Lacuna"
                    && code[after..].starts_with("::")
                    && let Some((variant, _)) = read_ident(code, after + 2)
                {
                    if matches!(variant, "Unobserved" | "SelfRef") {
                        m.abstract_bypass_refs += 1;
                    } else if variant == "Exterior" {
                        m.exterior_sandbox_refs += 1;
                    }
                }
                if matches!(
                    ident,
                    "slot_sandbox_mask" | "settle_sandbox_lane" | "SandboxLifecycleEvent"
                ) {
                    m.exterior_sandbox_refs += 1;
                }
                if matches!(
                    ident,
                    "DistortionDelta" | "EgoDifferentialNet" | "DistortionTension"
                ) {
                    m.pframe_ego_refs += 1;
                }
                if matches!(
                    ident,
                    "LmnkContainerView"
                        | "SystemSpineState"
                        | "pinned_compute_mask"
                        | "CORE_TOPOLOGY"
                ) {
                    m.runtime_gate_refs += 1;
                }
            }
        }
    }

    m
}

/// Renders the static `liminal` Abstraction <-> Concretion Mermaid map and table.
pub fn render_liminal_map(core_src: &Path) -> String {
    let m = collect_liminal_map_metrics(core_src);
    let lines = vec![
        "```mermaid".to_string(),
        "flowchart LR".to_string(),
        "    subgraph ABS[\"Abstract Realm (Low-Cost Ground State)\"]".to_string(),
        format!(
            "        ABS_FND[\"Lv.1 foundation/ Pure Compute ({} files, {} prod LOC)\"]",
            m.foundation_files, m.foundation_prod_loc
        ),
        format!(
            "        ABS_IFR[\"Indivisible(Grounded) I-Frame PhaseAnchorRecord ({} refs)\"]",
            m.grounded_iframe_refs
        ),
        format!(
            "        ABS_BYP[\"Lacuna::Unobserved and SelfRef Bypass ({} refs)\"]",
            m.abstract_bypass_refs
        ),
        "    end".to_string(),
        "    subgraph LIM[\"liminal Boundary Gates (Lv.0 Axiom + Lv.4 Runtime)\"]".to_string(),
        format!(
            "        LIM_AX[\"Lv.0 liminal/axiom seal / rescope / BitCut ({} files, {} refs)\"]",
            m.axiom_files, m.axiom_gate_refs
        ),
        format!(
            "        LIM_RT[\"Lv.4 LmnkContainerView + SystemSpineState ({} refs)\"]",
            m.runtime_gate_refs
        ),
        "    end".to_string(),
        "    subgraph CON[\"Concrete Realm (High-Cost Excited State)\"]".to_string(),
        format!(
            "        CON_ACT[\"Scoped + Indivisible(Active) RoPE / Rank-1 ({} refs)\"]",
            m.active_concretion_refs
        ),
        format!(
            "        CON_EGO[\"EgoDifferentialNet P-Frame + DistortionTension ({} refs)\"]",
            m.pframe_ego_refs
        ),
        format!(
            "        CON_EXT[\"Lacuna::Exterior Isolated Sandbox Lanes ({} refs)\"]",
            m.exterior_sandbox_refs
        ),
        "    end".to_string(),
        "    ABS_FND -.->|\"Pure Kernel Supply (Zero-Scoped)\"| CON_ACT".to_string(),
        "    ABS_IFR <-->|\"seal / ground I-Frame\"| LIM_AX".to_string(),
        "    ABS_BYP <-->|\"4p(1-p) Aperture & Spine Bypass\"| LIM_RT".to_string(),
        "    LIM_AX -->|\"rescope_from_delta / Activate\"| CON_ACT".to_string(),
        "    LIM_RT <-->|\"Host Pin Floor & Tension Damping\"| CON_EGO".to_string(),
        "    LIM_AX <-->|\"cut_by_lacuna / settle_sandbox_lane\"| CON_EXT".to_string(),
        "```".to_string(),
        String::new(),
        "## Static `liminal` Abstraction <-> Concretion Correspondence Table".to_string(),
        String::new(),
        concat!(
            "| Abstract Realm (Ground State) | `liminal` Boundary Gate / Op | ",
            "Concrete Realm (Excited State) | Live `src/core/src` Counts | ",
            "`CoreTelemetryBank` Counter |"
        )
        .to_string(),
        "| :--- | :--- | :--- | ---: | :--- |".to_string(),
        format!(
            concat!(
                "| Lv.1 `foundation/` non-`Scoped` pure compute | ",
                "Direct pure kernel call (no `liminal/axiom` import) | ",
                "Lv.2+ `Indivisible(Active)` RoPE / Rank-1 engines | ",
                "{} files / {} LOC -> {} active refs | ",
                "`stigmergy.decision_counts[0..7]` |"
            ),
            m.foundation_files, m.foundation_prod_loc, m.active_concretion_refs
        ),
        format!(
            concat!(
                "| `Indivisible(Grounded)` I-Frame `PhaseAnchorRecord` | ",
                "`seal` / `rescope_from_boundary` / `CORE_TOPOLOGY` | ",
                "`EgoDifferentialNet` P-Frame `DistortionDelta` | ",
                "{} grounded refs <-> {} P-Frame/tension refs | ",
                "`pframe_chain_evals`, `pframe_total_hops`, `ego.intervention_count` |"
            ),
            m.grounded_iframe_refs, m.pframe_ego_refs
        ),
        format!(
            concat!(
                "| `Lacuna::SelfRef` stationary bypass | ",
                "`SystemSpineState` + `pinned_compute_mask` | ",
                "`DistortionTension` damped/excited steering | ",
                "{} bypass refs <-> {} spine/container refs | ",
                "`whole_token_bypass_events`, `liminal.stationary_shortcut_hits`, ",
                "`immediate_tension_damps`, `slow_tension_excites` |"
            ),
            m.abstract_bypass_refs, m.runtime_gate_refs
        ),
        format!(
            concat!(
                "| `Lacuna::Unobserved` zero-aperture | ",
                "`cut_by_lacuna` / `LacunaPartition8` | ",
                "`Lacuna::Exterior` sandbox (`slot_sandbox_mask`) | ",
                "{} axiom gate refs <-> {} sandbox refs | ",
                "`unobserved_aperture_zeros`, `exterior_sandbox_stages`, ",
                "`liminal.slide_in_count`, `liminal.slide_out_count` |"
            ),
            m.axiom_gate_refs, m.exterior_sandbox_refs
        ),
        String::new(),
    ];
    lines.join("\n")
}
