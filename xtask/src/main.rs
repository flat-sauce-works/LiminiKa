// name: xtask/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-dependency Rust `xtask` CLI for LiminiKa architecture verification, workspace gates,
//! Mermaid maps, and non-blocking coupling meta-audits.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod core_dag;
mod coupling_audit;
mod mermaid;
mod scanner;
mod workspace_gates;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Default)]
struct CliArgs {
    check: bool,
    audit: bool,
    hotspots: bool,
    mermaid: bool,
    mermaid_workspace: bool,
    liminal_map: bool,
    output: Option<PathBuf>,
    help: bool,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask crate must reside directly under workspace root")
        .to_path_buf()
}

fn print_help() {
    println!(
        "Usage: cargo xtask [OPTIONS]\n\
         \n\
         Verify LiminiKa 5-layer Y-shaped DAG, workspace gates, Mermaid maps, and meta-audit.\n\
         \n\
         Options:\n\
           --check              Verify DAG, isolation, axiom orthogonality, AbstractDomain,\n\
                                Scoped boundary, workspace gates, legacy bans, Substrate\n\
                                ownership, CORE_TOPOLOGY SSOT, and file hygiene (default).\n\
           --audit              Emit non-blocking meta-audit warnings (always exits 0).\n\
           --hotspots           Print top core source files by LOC, unsafe, raw masks, axioms.\n\
           --mermaid            Emit Mermaid flowchart TD of the live core dependency graph.\n\
           --mermaid-workspace  Emit Mermaid flowchart TD of workspace crates -> adapter -> core.\n\
           --liminal-map        Emit static liminal Abstraction <-> Concretion map and table.\n\
           --output <PATH>      Optional output path when emitting a single diagram.\n\
           -h, --help           Show this help message."
    );
}

fn parse_args<I: IntoIterator<Item = String>>(iter: I) -> Result<CliArgs, String> {
    let mut args = CliArgs::default();
    let mut it = iter.into_iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--check" => args.check = true,
            "--audit" => args.audit = true,
            "--hotspots" => args.hotspots = true,
            "--mermaid" => args.mermaid = true,
            "--mermaid-workspace" => args.mermaid_workspace = true,
            "--liminal-map" => args.liminal_map = true,
            "--output" => {
                let Some(path) = it.next() else {
                    return Err("missing value for `--output`".to_string());
                };
                args.output = Some(PathBuf::from(path));
            }
            "-h" | "--help" => args.help = true,
            other => return Err(format!("unknown argument `{other}` (use `--help`)")),
        }
    }
    if !args.help
        && !(args.check
            || args.audit
            || args.hotspots
            || args.mermaid
            || args.mermaid_workspace
            || args.liminal_map)
    {
        args.check = true;
    }
    Ok(args)
}

fn run_checks(core_src: &Path, root: &Path) -> Vec<String> {
    let mut violations = core_dag::run_core_checks(core_src);
    violations.extend(workspace_gates::run_workspace_checks(root));
    violations
}

fn emit_diagram(
    content: &str,
    output_path: Option<&Path>,
    write_to_file: bool,
    label: &str,
    printed_any: bool,
) -> Result<bool, String> {
    if let Some(path) = output_path
        && write_to_file
    {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|e| format!("create dir failed: {e}"))?;
        }
        fs::write(path, content.as_bytes()).map_err(|e| format!("write failed: {e}"))?;
        println!(
            "\nWrote {label} to {}",
            path.to_string_lossy().replace('\\', "/")
        );
        return Ok(printed_any);
    }
    if printed_any {
        println!();
    }
    print!("{content}");
    Ok(true)
}

fn main() -> ExitCode {
    let args = match parse_args(env::args().skip(1)) {
        Ok(parsed) => parsed,
        Err(msg) => {
            eprintln!("Error: {msg}");
            return ExitCode::from(2);
        }
    };
    if args.help {
        print_help();
        return ExitCode::SUCCESS;
    }

    let root = repo_root();
    let core_src = root.join("src/core/src");
    let mut exit_code = ExitCode::SUCCESS;
    let mut printed_any = false;

    if args.check {
        let violations = run_checks(&core_src, &root);
        if !violations.is_empty() {
            println!(
                "Error: architecture, DAG, axiom orthogonality, boundary, \
                 workspace gate, or vocabulary ownership violation detected:"
            );
            for v in violations {
                println!("  {v}");
            }
            exit_code = ExitCode::FAILURE;
        } else {
            println!(
                "All 5-layer DAG, 3-driver isolation, axiom orthogonality, \
                 AbstractDomain, Scoped boundary, workspace gate, legacy declared-identifier bans, \
                 Substrate ownership, and file hygiene invariants passed (0 violations)."
            );
        }
        printed_any = true;
    }

    if args.audit {
        if printed_any {
            println!();
        }
        print!("{}", coupling_audit::render_coupling_audit(&root));
        printed_any = true;
    }

    if args.hotspots {
        if printed_any {
            println!();
        }
        print!("{}", mermaid::format_hotspots(&core_src, 12));
        printed_any = true;
    }

    let diagram_flags = usize::from(args.mermaid)
        + usize::from(args.mermaid_workspace)
        + usize::from(args.liminal_map);
    let write_to_file = diagram_flags == 1;

    if args.mermaid {
        match emit_diagram(
            &mermaid::render_mermaid(&core_src),
            args.output.as_deref(),
            write_to_file,
            "Mermaid diagram",
            printed_any,
        ) {
            Ok(p) => printed_any = p,
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    if args.mermaid_workspace {
        match emit_diagram(
            &mermaid::render_mermaid_workspace(&root),
            args.output.as_deref(),
            write_to_file,
            "workspace Mermaid diagram",
            printed_any,
        ) {
            Ok(p) => printed_any = p,
            Err(e) => {
                eprintln!("Error: {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    if args.liminal_map
        && let Err(e) = emit_diagram(
            &mermaid::render_liminal_map(&core_src),
            args.output.as_deref(),
            write_to_file,
            "liminal boundary map",
            printed_any,
        )
    {
        eprintln!("Error: {e}");
        return ExitCode::FAILURE;
    }

    exit_code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scanner_handles_c_strings_raw_strings_nested_comments_and_lifetimes() {
        let src = r####"
// line comment with UTF-8 em-dash — /* unmatched and "quote"
/* outer /* nested */ still comment */
pub fn demo<'a, 'static_like>(x: &'a str) -> (&'static std::ffi::CStr, char) {
    let _c = c"hello \"world\" // not comment";
    let _cr = cr#"raw c string /* not comment */ "#;
    let _br = br##"raw byte #""# string"##;
    let _ch = '\n';
    let _bch = b'x';
    (c"ok", 'a')
}
"####;
        let no_comments = scanner::strip_comments_preserve_lines(src);
        assert!(!no_comments.contains("nested"));
        assert!(no_comments.contains("c\"hello"));
        assert!(no_comments.contains("cr#\"raw c string"));

        let no_literals = scanner::strip_literals_preserve_lines(&no_comments);
        assert!(!no_literals.contains("not comment"));
        assert!(no_literals.contains("'a"));
        assert!(no_literals.contains("'static"));
        assert_eq!(src.lines().count(), no_literals.lines().count());
    }

    #[test]
    fn scanner_blanks_cfg_test_modules_preserving_line_numbers() {
        let src = concat!(
            "fn prod_a() {}\n",
            "#[cfg(test)]\n",
            "mod tests {\n",
            "    fn helper() { let _ = Scoped::<()>; }\n",
            "}\n",
            "fn prod_b() {}\n",
        );
        let stripped = scanner::extract_non_test_code(src);
        assert!(stripped.contains("prod_a"));
        assert!(stripped.contains("prod_b"));
        assert!(!stripped.contains("Scoped"));
        assert_eq!(stripped.lines().count(), 6);
    }

    #[test]
    fn detects_scoped_boundary_leak_and_ignores_private_fn() {
        let leaky = "pub fn bad(x: Scoped<Active, u32>) {}\nfn ok(x: Scoped<Active, u32>) {}\n";
        let hits = core_dag::scan_scoped_boundary_violations_in_source(leaky, "test.rs");
        assert_eq!(hits.len(), 1);
        assert!(hits[0].contains("test.rs:1"));
    }

    #[test]
    fn detects_host_llm_isolation_violations() {
        let bad_host = "use crate::pooling::Encoder;\npub fn step(logits: &[f32]) {}\n";
        let code = scanner::extract_non_test_code(bad_host);
        let lines: Vec<&str> = bad_host.lines().collect();
        let hits = workspace_gates::scan_host_llm_isolation_in_source(&code, "host/bad.rs", &lines);
        assert!(hits.iter().any(|h| h.contains("crate::pooling")));
        assert!(hits.iter().any(|h| h.contains("logits")));
    }

    #[test]
    fn detects_three_driver_sibling_isolation_violations() {
        let bad_spine = "use crate::drivers::envelope::EnvelopeState;\n";
        assert_eq!(
            core_dag::scan_driver_isolation_in_source(bad_spine, "envelope").len(),
            1
        );
        let bad_env_ego = "use super::ego::EgoDifferentialNet;\n";
        assert_eq!(
            core_dag::scan_driver_isolation_in_source(bad_env_ego, "ego").len(),
            1
        );
        let bad_env_spine = "use crate::drivers::spine::SystemSpineState;\n";
        assert_eq!(
            core_dag::scan_driver_isolation_in_source(bad_env_spine, "spine").len(),
            1
        );
        let clean_env = "use crate::liminal::BoundarySignal;\nuse crate::foundation::Pod;\n";
        assert!(core_dag::scan_driver_isolation_in_source(clean_env, "spine").is_empty());
        assert!(core_dag::scan_driver_isolation_in_source(clean_env, "ego").is_empty());
    }

    #[test]
    fn core_topology_ssot_and_xtask_zero_deps_pass() {
        let root = repo_root();
        let core_src = root.join("src/core/src");
        assert!(core_dag::check_topology_matrix_parity(&core_src).is_empty());
        assert!(core_dag::check_xtask_zero_external_deps(&root).is_empty());
    }
}
