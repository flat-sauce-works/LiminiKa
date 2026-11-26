// name: xtask/src/core_dag.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hard-error invariant checks for `liminika-core`'s 5-layer Y-shaped DAG, axioms, SSOT topology,
//! and zero-external-dependency `xtask` manifest discipline.

use std::fs;
use std::path::Path;

use liminika_core::runtime::{
    CORE_TOPOLOGY, LMNK_TOPOLOGY_CHANNEL_COUNT, LMNK_TOPOLOGY_DAG_EDGE_MATRIX,
    LMNK_TOPOLOGY_DAG_FLAGS_CANONICAL, LMNK_TOPOLOGY_LAYER_COUNT, LMNK_TOPOLOGY_MAGIC,
};

use crate::scanner::{
    advance_past_literal, blank_cfg_test_modules, extract_non_test_code, find_banned_public_decls,
    find_pub_items, is_ident_cont, is_ident_start, is_test_path, iter_idents, line_number_at,
    match_pub_vis, read_ident, rel_posix, scan_until_top_level_terminator, skip_ws,
    strip_comments_preserve_lines, walk_rs_files,
};

const RETIRED_LEGACY_PATHS: &[&str] = &[
    "traits",
    "traits.rs",
    "foundation/phase_field.rs",
    "stigmergy/action_hub.rs",
    "runtime/gateway/closed_loop.rs",
    "stigmergy/attractor/field_engine.rs",
    "stigmergy/attractor/multi_space.rs",
    "stigmergy/attractor/snapshot.rs",
    "stigmergy/crystal",
    "stigmergy/attractor",
    "runtime/abi/action_hub.rs",
    "runtime/abi/entropy_attractor.rs",
    "runtime/abi/storage.rs",
];

const REQUIRED_CANONICAL_PATHS: &[&str] = &[
    "foundation/phase_ops/mod.rs",
    "stigmergy/substrate/mod.rs",
    "stigmergy/anchor/mod.rs",
    "runtime/topology.rs",
];

/// Banned legacy declared-identifier tokens in `liminika-core`.
pub const BANNED_LEGACY_DECL_TOKENS: &[&str] = &[
    "field",
    "gcso",
    "void",
    "venn",
    "serendipity",
    "edbc",
    "srl",
    "pspm",
    "pprc",
    "zimms",
    "daes",
    "dpsr",
    "ripa",
    "qdps",
    "action_hub",
];

const ALLOWED_SUBSTRATE_DECLS: &[(&str, &str)] = &[
    ("StigmergicSubstrate", "stigmergy/substrate/mod.rs"),
    ("SubstrateStepReport", "stigmergy/substrate/mod.rs"),
    ("PointerSubstrate", "stigmergy/trail/traits.rs"),
    ("StigmergicSubstrateOps", "stigmergy/trail/traits.rs"),
    ("SubstrateGatewayExt", "runtime/gateway/orchestrator.rs"),
];

/// Sibling driver isolation pairs (`(source_driver, forbidden_sibling)`).
/// Designed for Phase 2 expansion to 3 siblings (`spine`, `ego`, `envelope`).
const DRIVER_ISOLATION_PAIRS: &[(&str, &str)] = &[("spine", "ego"), ("ego", "spine")];

const FOUNDATION_FORBIDDEN_WORDS: &[&str] = &[
    "BitCut",
    "BitCut8",
    "BitCut32",
    "BitCut64",
    "ScopedCut",
    "Indivisible",
    "IndivisiblePhase",
    "AbstractDomain",
    "Activatable",
    "Groundable",
    "ActiveScope",
    "GroundedScope",
    "GroundedOrigin",
    "SpeculativeDiscovery",
    "BoundaryViolation",
    "VerifiedAligned",
    "BridgedDiscovery",
];

const AXIOM_BITCUT_WORDS: &[&str] = &["bitcut", "BitCut", "BitCut8", "BitCut32", "BitCut64"];

const LACUNA_FORBIDDEN_CUT_WORDS: &[&str] = &[
    "Scoped",
    "ScopedCut",
    "Indivisible",
    "IndivisiblePhase",
    "Activatable",
    "Groundable",
    "ActiveScope",
    "GroundedScope",
    "Cuttable",
    "Definite",
    "BoundaryViolation",
    "SpeculativeDiscovery",
];

fn find_prefixed_module_refs(
    text: &str,
    crate_prefix: &str,
    allow_super_chain: bool,
    targets: &[&str],
) -> Vec<usize> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut hits = Vec::new();
    let mut i = 0;

    let prefix_bytes = crate_prefix.as_bytes();
    while i < n {
        if i > 0 && is_ident_cont(bytes[i - 1]) {
            i += 1;
            continue;
        }
        let mut after_prefix = None;
        if bytes[i..].starts_with(prefix_bytes) {
            after_prefix = Some(i + prefix_bytes.len());
        } else if allow_super_chain && bytes[i..].starts_with(b"super::") {
            let mut cur = i;
            while bytes[cur..].starts_with(b"super::") {
                cur += "super::".len();
            }
            after_prefix = Some(cur);
        }

        if let Some(pos) = after_prefix {
            if let Some((ident, _)) = read_ident(text, pos) {
                if targets.contains(&ident) {
                    hits.push(i);
                }
            } else if pos < n
                && bytes[pos] == b'{'
                && let Some(close_rel) = text[pos + 1..].find('}')
            {
                let brace_body = &text[pos + 1..pos + 1 + close_rel];
                if iter_idents(brace_body)
                    .iter()
                    .any(|(_, w)| targets.contains(w))
                {
                    hits.push(i);
                }
            }
            i = pos;
            continue;
        }
        i += 1;
    }

    hits
}

fn find_foundation_axiom_refs(text: &str) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut hits = Vec::new();

    for (offset, ident) in iter_idents(text) {
        let after = offset + ident.len();
        if ident == "liminal" && text[after..].starts_with("::") {
            if let Some((sub, _)) = read_ident(text, after + 2)
                && matches!(sub, "axiom" | "bitcut")
            {
                hits.push(offset);
            }
            continue;
        }
        if ident == "Scoped" {
            let ws = skip_ws(bytes, after);
            if text[after..].starts_with("::") || (ws < bytes.len() && bytes[ws] == b'<') {
                hits.push(offset);
            }
            continue;
        }
        if FOUNDATION_FORBIDDEN_WORDS.contains(&ident) {
            hits.push(offset);
        }
    }

    hits
}

fn find_lacuna_cut_refs(text: &str) -> Vec<usize> {
    let mut hits = Vec::new();
    for (offset, ident) in iter_idents(text) {
        let after = offset + ident.len();
        if LACUNA_FORBIDDEN_CUT_WORDS.contains(&ident) {
            hits.push(offset);
            continue;
        }
        if matches!(ident, "axiom" | "liminal")
            && text[after..].starts_with("::")
            && read_ident(text, after + 2).is_some_and(|(w, _)| w == "cut")
        {
            hits.push(offset);
            continue;
        }
        if ident == "super" && text[after..].starts_with("::") {
            let mut cur = offset;
            while text[cur..].starts_with("super::") {
                cur += "super::".len();
            }
            if read_ident(text, cur).is_some_and(|(w, _)| w == "cut") {
                hits.push(offset);
            }
        }
    }
    hits
}

fn find_case_insensitive_substring(text: &str, needle_lower: &str) -> Vec<usize> {
    let lower = text.to_ascii_lowercase();
    let mut hits = Vec::new();
    let mut start = 0;
    while let Some(rel) = lower[start..].find(needle_lower) {
        let pos = start + rel;
        hits.push(pos);
        start = pos + needle_lower.len();
    }
    hits
}

fn format_hit(label: &str, file: &Path, text: &str, offset: usize) -> String {
    let lineno = line_number_at(text, offset);
    let snippet = text
        .lines()
        .nth(lineno.saturating_sub(1))
        .map(str::trim)
        .unwrap_or("");
    format!(
        "{label}: {}:{lineno}: {snippet}",
        file.to_string_lossy().replace('\\', "/")
    )
}

fn check_dag_and_axiom_rules(core_src: &Path) -> Vec<String> {
    let mut violations = Vec::new();

    let mut run_rule = |label: &str, sub: &str, finder: &dyn Fn(&str) -> Vec<usize>| {
        let sub_path = core_src.join(sub);
        let rs_files = if sub_path.is_file() {
            vec![sub_path.clone()]
        } else if sub_path.is_dir() {
            walk_rs_files(&sub_path)
        } else {
            Vec::new()
        };
        if rs_files.is_empty() {
            violations.push(format!(
                "{label}: path {} is missing or contains no .rs files",
                sub_path.to_string_lossy().replace('\\', "/")
            ));
            return;
        }
        for rs_file in rs_files {
            let text = fs::read_to_string(&rs_file).expect("read rs file");
            for offset in finder(&text) {
                violations.push(format_hit(label, &rs_file, &text, offset));
            }
        }
    };

    run_rule("Lv.0 liminal -> Lv.1..4", "liminal", &|t| {
        find_prefixed_module_refs(
            t,
            "crate::",
            true,
            &["foundation", "stigmergy", "drivers", "runtime"],
        )
    });
    run_rule("Lv.1 foundation -> Lv.2..4", "foundation", &|t| {
        find_prefixed_module_refs(t, "crate::", true, &["stigmergy", "drivers", "runtime"])
    });
    run_rule(
        "Lv.1 foundation -> liminal::axiom / Scoped",
        "foundation",
        &find_foundation_axiom_refs,
    );
    run_rule("Lv.0 liminal/axiom -> bitcut", "liminal/axiom", &|t| {
        iter_idents(t)
            .into_iter()
            .filter(|(_, w)| AXIOM_BITCUT_WORDS.contains(w))
            .map(|(off, _)| off)
            .collect()
    });
    run_rule("Lv.2 stigmergy -> Lv.3..4", "stigmergy", &|t| {
        find_prefixed_module_refs(t, "crate::", true, &["drivers", "runtime"])
    });
    run_rule("Lv.3 drivers -> Lv.4", "drivers", &|t| {
        find_prefixed_module_refs(t, "crate::", true, &["runtime"])
    });
    for &(src_driver, dst_driver) in DRIVER_ISOLATION_PAIRS {
        let label = format!("drivers/{src_driver} -> drivers/{dst_driver}");
        let sub = format!("drivers/{src_driver}");
        run_rule(&label, &sub, &|t| {
            find_prefixed_module_refs(t, "crate::drivers::", true, &[dst_driver])
        });
    }
    run_rule(
        "Lv.0 axiom/cut -> axiom/lacuna",
        "liminal/axiom/cut.rs",
        &|t| find_case_insensitive_substring(t, "lacuna"),
    );
    run_rule(
        "Lv.0 axiom/lacuna -> axiom/cut",
        "liminal/axiom/lacuna.rs",
        &find_lacuna_cut_refs,
    );

    violations
}

/// Verifies `AbstractDomain` open trait contract and unconstrained `Indivisible<C, T>`.
pub fn check_abstract_domain_invariant(core_src: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    let cut_rs = core_src.join("liminal/axiom/cut.rs");
    let cut_display = cut_rs.to_string_lossy().replace('\\', "/");
    if !cut_rs.is_file() {
        return vec![format!("AbstractDomain check: missing {cut_display}")];
    }
    let text = fs::read_to_string(&cut_rs).expect("read cut.rs");
    if !text.contains("pub trait AbstractDomain") {
        violations.push(format!(
            "AbstractDomain missing `pub trait AbstractDomain` in {cut_display}"
        ));
    }
    if !text.contains("pub const fn domain_id(&self) -> &'static str") {
        violations.push(format!(
            "AbstractDomain missing `Indivisible::domain_id` in {cut_display}"
        ));
    }
    match text.find("pub struct Indivisible<") {
        None => violations.push(format!(
            "AbstractDomain check: could not locate `pub struct Indivisible` in {cut_display}"
        )),
        Some(pos) => {
            let after = &text[pos + "pub struct Indivisible<".len()..];
            if let Some(gt) = after.find('>')
                && after[..gt].contains("AbstractDomain")
            {
                violations.push(format!(
                    "AbstractDomain check: `Indivisible<C, T>` itself must not require \
                     `T: AbstractDomain` ({cut_display})"
                ));
            }
        }
    }
    violations
}

fn match_pub_trait_start(text: &str, idx: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut cur = match_pub_vis(text, idx)?;
    loop {
        let (kw, next) = read_ident(text, cur)?;
        if matches!(kw, "unsafe" | "auto") {
            let ws = skip_ws(bytes, next);
            if ws > next {
                cur = ws;
                continue;
            }
            return None;
        }
        if kw == "trait" {
            let ws = skip_ws(bytes, next);
            if ws > next {
                let (_, end) = read_ident(text, ws)?;
                return Some(end);
            }
        }
        return None;
    }
}

fn match_fn_start(text: &str, idx: usize) -> Option<(bool, String, usize)> {
    let bytes = text.as_bytes();
    let (has_pub, mut cur) = match match_pub_vis(text, idx) {
        Some(next) => (true, next),
        None => (false, idx),
    };
    loop {
        let (kw, next) = read_ident(text, cur)?;
        if matches!(kw, "default" | "const" | "async" | "unsafe") {
            let ws = skip_ws(bytes, next);
            if ws > next {
                cur = ws;
                continue;
            }
            return None;
        }
        if kw == "extern" {
            let mut ws = skip_ws(bytes, next);
            if ws > next {
                if let Some(after_lit) = advance_past_literal(text, ws) {
                    ws = skip_ws(bytes, after_lit);
                }
                cur = ws;
                continue;
            }
            return None;
        }
        if kw == "fn" {
            let ws = skip_ws(bytes, next);
            if ws > next {
                let (name, end) = read_ident(text, ws)?;
                return Some((has_pub, name.to_string(), end));
            }
        }
        return None;
    }
}

fn has_scoped_boundary_type(header: &str) -> bool {
    let bytes = header.as_bytes();
    for (offset, ident) in iter_idents(header) {
        let after = offset + ident.len();
        if ident == "ScopedCut" {
            return true;
        }
        if matches!(
            ident,
            "Scoped"
                | "VerifiedAligned"
                | "BoundaryViolation"
                | "SpeculativeDiscovery"
                | "BridgedDiscovery"
        ) {
            let ws = skip_ws(bytes, after);
            if ws < bytes.len() && bytes[ws] == b'<' {
                return true;
            }
        }
    }
    false
}

/// Scans non-test Rust source text for public cross-observer `Scoped<` or `*_scoped*` signatures.
pub fn scan_scoped_boundary_violations_in_source(source: &str, source_label: &str) -> Vec<String> {
    let mut violations = Vec::new();
    let truncated = blank_cfg_test_modules(source);
    let clean = strip_comments_preserve_lines(&truncated);
    let bytes = clean.as_bytes();
    let n = bytes.len();
    let mut brace_depth = 0usize;
    let mut pub_trait_depths: Vec<usize> = Vec::new();
    let mut i = 0;

    while i < n {
        if let Some(next_i) = advance_past_literal(&clean, i) {
            i = next_i;
            continue;
        }
        let ch = bytes[i];
        if ch == b'{' {
            brace_depth += 1;
            i += 1;
            continue;
        }
        if ch == b'}' {
            brace_depth = brace_depth.saturating_sub(1);
            while pub_trait_depths.last().is_some_and(|&d| d > brace_depth) {
                pub_trait_depths.pop();
            }
            i += 1;
            continue;
        }

        let prev_is_ident = i > 0 && is_ident_cont(bytes[i - 1]);
        if !prev_is_ident && is_ident_start(ch) {
            if let Some(trait_end) = match_pub_trait_start(&clean, i) {
                let (term_idx, term_ch) = scan_until_top_level_terminator(&clean, trait_end);
                if term_ch == Some(b'{') {
                    brace_depth += 1;
                    pub_trait_depths.push(brace_depth);
                }
                i = term_idx.saturating_add(1).min(n);
                continue;
            }

            if let Some((has_pub_vis, sig_name, fn_end)) = match_fn_start(&clean, i) {
                let in_pub_trait = pub_trait_depths.last().is_some_and(|&d| d == brace_depth);
                let is_public_sig = has_pub_vis || in_pub_trait;
                let (term_idx, term_ch) = scan_until_top_level_terminator(&clean, fn_end);
                if is_public_sig {
                    let sig_line = line_number_at(&clean, i);
                    let header = clean[i..term_idx]
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ");
                    if sig_name.contains("_scoped") {
                        violations.push(format!(
                            "Retired `*_scoped*` signature: {source_label}:{sig_line}: fn {sig_name}"
                        ));
                    }
                    if has_scoped_boundary_type(&header) {
                        violations.push(format!(
                            "Cross-observer `Scoped<` signature leak: {source_label}:{sig_line}: {header}"
                        ));
                    }
                }
                if term_ch == Some(b'{') {
                    brace_depth += 1;
                }
                i = term_idx.saturating_add(1).min(n);
                continue;
            }

            while i < n && is_ident_cont(bytes[i]) {
                i += 1;
            }
            continue;
        }

        i += 1;
    }

    violations
}

/// Verifies Lv.2..Lv.4 public APIs do not leak `Scoped<` or retired `*_scoped*` signatures.
pub fn check_cross_observer_scoped_boundaries(core_src: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    for layer in ["stigmergy", "drivers", "runtime"] {
        let layer_dir = core_src.join(layer);
        if !layer_dir.is_dir() {
            continue;
        }
        for rs_file in walk_rs_files(&layer_dir) {
            if is_test_path(&rs_file) {
                continue;
            }
            let source = fs::read_to_string(&rs_file).expect("read rs file");
            let label = rs_file.to_string_lossy().replace('\\', "/");
            violations.extend(scan_scoped_boundary_violations_in_source(&source, &label));
        }
    }
    violations
}

/// Verifies legacy declared-identifier bans and `*Substrate*` (`stigmergy`-only) ownership.
pub fn scan_vocabulary_ownership_in_source(source: &str, rel_path: &str) -> Vec<String> {
    let norm_raw = rel_path.replace('\\', "/");
    let norm_trimmed = norm_raw.trim_start_matches('/');
    let norm_rel = norm_trimmed
        .strip_prefix("src/core/src/")
        .unwrap_or(norm_trimmed);
    if norm_rel.ends_with("tests.rs") || format!("/{norm_rel}").contains("/tests/") {
        return Vec::new();
    }

    let code_only = extract_non_test_code(source);
    let mut violations = find_banned_public_decls(
        &code_only,
        norm_rel,
        BANNED_LEGACY_DECL_TOKENS,
        "Vocabulary ownership",
    );

    for (offset, name) in find_pub_items(&code_only, true) {
        if name.contains("Substrate") {
            let allowed = ALLOWED_SUBSTRATE_DECLS
                .iter()
                .any(|&(n, p)| n == name && p == norm_rel);
            if !allowed {
                let lineno = line_number_at(&code_only, offset);
                violations.push(format!(
                    "Vocabulary ownership (Substrate=stigmergy-only): {norm_rel}:{lineno}: \
                     disallowed public declaration `{name}`"
                ));
            }
        }
    }
    violations
}

/// Verifies vocabulary ownership across all non-test production files in `liminika-core`.
pub fn check_vocabulary_ownership(core_src: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    for rs_file in walk_rs_files(core_src) {
        if is_test_path(&rs_file) {
            continue;
        }
        let rel = rel_posix(core_src, &rs_file);
        let source = fs::read_to_string(&rs_file).expect("read rs file");
        violations.extend(scan_vocabulary_ownership_in_source(&source, &rel));
    }
    violations
}

/// Directly verifies `liminika_core::runtime::CORE_TOPOLOGY` as the compile-time SSOT.
pub fn check_topology_matrix_parity(core_src: &Path) -> Vec<String> {
    let mut violations = Vec::new();
    let topo_rs = core_src.join("runtime/topology.rs");
    if !topo_rs.is_file() {
        violations.push(format!(
            "Topology parity check: missing {}",
            topo_rs.to_string_lossy().replace('\\', "/")
        ));
    }
    if !CORE_TOPOLOGY.verify_crc32() {
        violations.push("Topology parity: CORE_TOPOLOGY.verify_crc32() failed".to_string());
    }
    if CORE_TOPOLOGY.magic != LMNK_TOPOLOGY_MAGIC {
        violations.push(format!(
            "Topology parity: magic expected {LMNK_TOPOLOGY_MAGIC:#010x}, got {:#010x}",
            CORE_TOPOLOGY.magic
        ));
    }
    if CORE_TOPOLOGY.layer_count != 5 || LMNK_TOPOLOGY_LAYER_COUNT != 5 {
        violations.push(format!(
            "Topology parity: layer_count expected 5, got {}",
            CORE_TOPOLOGY.layer_count
        ));
    }
    if CORE_TOPOLOGY.channel_count != 8 || LMNK_TOPOLOGY_CHANNEL_COUNT != 8 {
        violations.push(format!(
            "Topology parity: channel_count expected 8, got {}",
            CORE_TOPOLOGY.channel_count
        ));
    }
    let expected_edges: [u8; 8] = [0b00000, 0b00001, 0b00011, 0b00111, 0b01111, 0, 0, 0];
    if CORE_TOPOLOGY.dag_edge_matrix != expected_edges
        || LMNK_TOPOLOGY_DAG_EDGE_MATRIX != expected_edges
    {
        violations.push(format!(
            "Topology parity: dag_edge_matrix expected {:?}, got {:?}",
            expected_edges, CORE_TOPOLOGY.dag_edge_matrix
        ));
    }
    if CORE_TOPOLOGY.dag_flags != LMNK_TOPOLOGY_DAG_FLAGS_CANONICAL {
        violations.push(format!(
            "Topology parity: dag_flags expected {LMNK_TOPOLOGY_DAG_FLAGS_CANONICAL:#06x}, got {:#06x}",
            CORE_TOPOLOGY.dag_flags
        ));
    }
    violations
}

/// Enforces that `xtask/Cargo.toml` carries zero external crate dependencies.
pub fn check_xtask_zero_external_deps(repo_root: &Path) -> Vec<String> {
    let manifest = repo_root.join("xtask/Cargo.toml");
    let Ok(text) = fs::read_to_string(&manifest) else {
        return vec!["xtask manifest missing: xtask/Cargo.toml".to_string()];
    };
    let mut violations = Vec::new();
    let mut section = String::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let lineno = idx + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = line[1..line.len() - 1].trim().to_string();
            continue;
        }
        if section == "dependencies" {
            let Some((key, val)) = line.split_once('=') else {
                continue;
            };
            let dep = key.trim();
            if dep != "liminika-core" || !val.contains("path") {
                violations.push(format!(
                    "xtask zero-external-dep invariant: xtask/Cargo.toml:{lineno}: \
                     disallowed dependency `{dep}`"
                ));
            }
        } else if matches!(section.as_str(), "dev-dependencies" | "build-dependencies")
            || section.ends_with(".dependencies")
            || section.ends_with(".dev-dependencies")
            || section.ends_with(".build-dependencies")
        {
            violations.push(format!(
                "xtask zero-external-dep invariant: xtask/Cargo.toml:{lineno}: \
                 disallowed entry in `[{section}]`"
            ));
        }
    }
    violations
}

/// Runs all `liminika-core` DAG, axiom, `Scoped`, vocabulary, SSOT topology, and `xtask` dep checks.
pub fn run_core_checks(core_src: &Path) -> Vec<String> {
    let mut violations = Vec::new();

    for rel in RETIRED_LEGACY_PATHS {
        let legacy = core_src.join(rel);
        if legacy.exists() {
            violations.push(format!(
                "Deprecated legacy path exists: {}",
                legacy.to_string_lossy().replace('\\', "/")
            ));
        }
    }
    for rel in REQUIRED_CANONICAL_PATHS {
        let canon = core_src.join(rel);
        if !canon.exists() {
            violations.push(format!(
                "Missing required canonical path: {}",
                canon.to_string_lossy().replace('\\', "/")
            ));
        }
    }

    violations.extend(check_dag_and_axiom_rules(core_src));
    violations.extend(check_abstract_domain_invariant(core_src));
    violations.extend(check_cross_observer_scoped_boundaries(core_src));
    violations.extend(check_vocabulary_ownership(core_src));
    violations.extend(check_topology_matrix_parity(core_src));
    if let Some(repo_root) = core_src
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
    {
        violations.extend(check_xtask_zero_external_deps(repo_root));
    }
    violations
}
