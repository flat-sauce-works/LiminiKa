# name: scripts/verify_architecture.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Thin orchestrator CLI for LiminiKa architecture checks, Mermaid maps, and meta-audit radar."""

from __future__ import annotations

import argparse
import pathlib
import sys

_SCRIPTS_DIR = str(pathlib.Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from arch_checks import (  # noqa: E402
    CORE_SRC,
    REPO_ROOT,
    bucket_for_path,
    check_abstract_domain_invariant,
    check_cross_observer_scoped_boundaries,
    check_downstream_vocabulary,
    check_file_hygiene_and_loc,
    check_host_llm_isolation,
    check_topology_matrix_parity,
    check_vocabulary_ownership,
    check_workspace_crate_gates,
    collect_dag_metrics,
    collect_liminal_map_metrics,
    format_hotspots,
    render_coupling_audit,
    render_liminal_map,
    render_mermaid,
    render_mermaid_workspace,
    run_core_checks,
    run_workspace_checks,
    scan_scoped_boundary_violations_in_source,
    scan_vocabulary_ownership_in_source,
)

__all__ = [
    "CORE_SRC",
    "REPO_ROOT",
    "bucket_for_path",
    "check_abstract_domain_invariant",
    "check_cross_observer_scoped_boundaries",
    "check_downstream_vocabulary",
    "check_file_hygiene_and_loc",
    "check_host_llm_isolation",
    "check_topology_matrix_parity",
    "check_vocabulary_ownership",
    "check_workspace_crate_gates",
    "collect_dag_metrics",
    "collect_liminal_map_metrics",
    "format_hotspots",
    "main",
    "parse_args",
    "render_coupling_audit",
    "render_liminal_map",
    "render_mermaid",
    "render_mermaid_workspace",
    "run_checks",
    "scan_scoped_boundary_violations_in_source",
    "scan_vocabulary_ownership_in_source",
]


def run_checks(core_src: pathlib.Path) -> list[str]:
    """Run all hard-error architecture invariants and return violation descriptions."""
    violations = run_core_checks(core_src)
    repo_root = core_src.parent.parent.parent
    violations.extend(run_workspace_checks(repo_root))
    return violations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Verify LiminiKa 5-layer Y-shaped DAG, workspace gates, Mermaid maps, and meta-audit."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify DAG, isolation, axiom orthogonality, AbstractDomain, Scoped boundary, "
            "workspace gates, legacy bans, Substrate ownership, and file hygiene."
        ),
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Emit non-blocking meta-audit warnings (always exits 0).",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Emit a Mermaid flowchart TD representation of the live core dependency graph.",
    )
    parser.add_argument(
        "--mermaid-workspace",
        action="store_true",
        help=(
            "Emit a Mermaid flowchart TD representation of workspace crates -> adapter gates -> "
            "core 5-layer DAG distinguishing Value/Pod refs (-.->) from Mutable State Gates (-->)."
        ),
    )
    parser.add_argument(
        "--liminal-map",
        action="store_true",
        help="Emit the static liminal Abstraction <-> Concretion Mermaid map and correspondence table.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional output file path for --mermaid, --mermaid-workspace, or --liminal-map output.",
    )
    parser.add_argument(
        "--hotspots",
        action="store_true",
        help="Print top core source files by LOC, unsafe tokens, raw bitmask operations, and axiom surface.",
    )
    args = parser.parse_args(argv)
    if not (
        args.check
        or args.audit
        or args.mermaid
        or args.mermaid_workspace
        or args.hotspots
        or args.liminal_map
    ):
        args.check = True
    return args


def _emit_diagram(
    content: str,
    output_path: pathlib.Path | None,
    write_to_file: bool,
    label: str,
    printed_any: bool,
) -> bool:
    if output_path is not None and write_to_file:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content.encode("utf-8"))
        print(f"\nWrote {label} to {output_path.as_posix()}")
        return printed_any
    if printed_any:
        print()
    print(content, end="")
    return True


def main(argv: list[str] | None = None) -> int:
    """Run requested architecture verification and analysis commands."""
    args = parse_args(argv)
    exit_code = 0
    printed_any = False

    if args.check:
        violations = run_checks(CORE_SRC)
        if violations:
            print(
                "Error: architecture, DAG, axiom orthogonality, boundary, "
                "workspace gate, or vocabulary ownership violation detected:",
                *violations,
                sep="\n  ",
            )
            exit_code = 1
        else:
            print(
                "All 5-layer DAG, spine/ego isolation, axiom orthogonality, "
                "AbstractDomain, Scoped boundary, workspace gate, legacy declared-identifier bans, "
                "Substrate ownership, and file hygiene invariants passed (0 violations)."
            )
        printed_any = True

    if args.audit:
        if printed_any:
            print()
        print(render_coupling_audit(REPO_ROOT), end="")
        printed_any = True

    if args.hotspots:
        if printed_any:
            print()
        print(format_hotspots(CORE_SRC), end="")
        printed_any = True

    diagram_flags = int(args.mermaid) + int(args.mermaid_workspace) + int(args.liminal_map)

    if args.mermaid:
        printed_any = _emit_diagram(
            render_mermaid(CORE_SRC),
            args.output,
            diagram_flags == 1,
            "Mermaid diagram",
            printed_any,
        )

    if args.mermaid_workspace:
        printed_any = _emit_diagram(
            render_mermaid_workspace(REPO_ROOT),
            args.output,
            diagram_flags == 1,
            "workspace Mermaid diagram",
            printed_any,
        )

    if args.liminal_map:
        printed_any = _emit_diagram(
            render_liminal_map(CORE_SRC),
            args.output,
            diagram_flags == 1,
            "liminal boundary map",
            printed_any,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
