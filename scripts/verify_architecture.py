# name: scripts/verify_architecture.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Architecture, Y-shaped 5-layer DAG, axiom orthogonality, and hotspot verification tool."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from collections import Counter, defaultdict

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE_SRC = REPO_ROOT / "src" / "core" / "src"

DAG_AND_AXIOM_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "Lv.0 liminal -> Lv.1..4",
        "liminal",
        re.compile(
            r"\b(?:crate::|(?:super::)+)(?:(foundation|stigmergy|drivers|runtime)\b|\{[^}]*\b(foundation|stigmergy|drivers|runtime)\b)"
        ),
    ),
    (
        "Lv.1 foundation -> Lv.2..4",
        "foundation",
        re.compile(
            r"\b(?:crate::|(?:super::)+)(?:(stigmergy|drivers|runtime)\b|\{[^}]*\b(stigmergy|drivers|runtime)\b)"
        ),
    ),
    (
        "Lv.1 foundation -> liminal::axiom / Scoped",
        "foundation",
        re.compile(
            r"\b(?:liminal::axiom\b|Scoped\s*<|Scoped::|ScopedCut\b|Indivisible\s*<|Indivisible::|Indivisible\b|IndivisiblePhase\b|Activatable\b|Groundable\b|ActiveScope\b|GroundedScope\b|GroundedOrigin\b|SpeculativeDiscovery\b|BoundaryViolation\b|VerifiedAligned\b|BridgedDiscovery\b)"
        ),
    ),
    (
        "Lv.2 stigmergy -> Lv.3..4",
        "stigmergy",
        re.compile(
            r"\b(?:crate::|(?:super::)+)(?:(drivers|runtime)\b|\{[^}]*\b(drivers|runtime)\b)"
        ),
    ),
    (
        "Lv.3 drivers -> Lv.4",
        "drivers",
        re.compile(r"\b(?:crate::|(?:super::)+)(?:runtime\b|\{[^}]*\bruntime\b)"),
    ),
    (
        "drivers/spine -> drivers/ego",
        "drivers/spine",
        re.compile(r"\b(?:crate::drivers::|(?:super::)+)(?:ego\b|\{[^}]*\bego\b)"),
    ),
    (
        "drivers/ego -> drivers/spine",
        "drivers/ego",
        re.compile(r"\b(?:crate::drivers::|(?:super::)+)(?:spine\b|\{[^}]*\bspine\b)"),
    ),
    (
        "Lv.0 axiom/cut -> axiom/lacuna",
        "liminal/axiom/cut.rs",
        re.compile(r"lacuna", re.IGNORECASE),
    ),
    (
        "Lv.0 axiom/lacuna -> axiom/cut",
        "liminal/axiom/lacuna.rs",
        re.compile(
            r"\b(?:(?:super::)+|axiom::|liminal::)cut\b|\b(?:Scoped|ScopedCut|Indivisible|IndivisiblePhase|Activatable|Groundable|BitCut8|BitCut32|ActiveScope|GroundedScope|Cuttable|Definite|BoundaryViolation|SpeculativeDiscovery)\b"
        ),
    ),
]

RETIRED_LEGACY_PATHS: tuple[str, ...] = (
    "traits",
    "traits.rs",
    "stigmergy/action_hub.rs",
    "runtime/gateway/closed_loop.rs",
)

REQUIRED_CANONICAL_PATHS: tuple[str, ...] = ("stigmergy/substrate.rs",)

SCOPED_BOUNDARY_TYPE_RE = re.compile(
    r"\b(?:Scoped\s*<|ScopedCut\b|VerifiedAligned\s*<|BoundaryViolation\s*<|SpeculativeDiscovery\s*<|BridgedDiscovery\s*<)"
)
FN_SIG_START_RE = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+([A-Za-z0-9_]+)")
CRATE_DEP_RE = re.compile(r"\bcrate::(liminal|foundation|stigmergy|drivers(?:::(?:spine|ego))?|runtime)\b")


def strip_cfg_test_tail(lines: list[str]) -> list[tuple[int, str]]:
    """Return `(1-based lineno, line)` pairs excluding `#[cfg(test)]` module tails."""
    kept: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, 1):
        if line.strip() == "#[cfg(test)]":
            break
        kept.append((idx, line))
    return kept


def check_abstract_domain_invariant(core_src: pathlib.Path) -> list[str]:
    """Verify `AbstractDomain` open trait contract and unconstrained `Indivisible<C, T>`."""
    violations: list[str] = []
    cut_rs = core_src / "liminal" / "axiom" / "cut.rs"
    if not cut_rs.is_file():
        return [f"AbstractDomain check: missing {cut_rs.as_posix()}"]

    text = cut_rs.read_text(encoding="utf-8")
    if not re.search(r"\bpub trait AbstractDomain\b", text):
        violations.append(f"AbstractDomain missing `pub trait AbstractDomain` in {cut_rs.as_posix()}")
    if not re.search(r"pub const fn domain_id\(&self\)\s*->\s*&'static str", text):
        violations.append(f"AbstractDomain missing `Indivisible::domain_id` in {cut_rs.as_posix()}")

    struct_match = re.search(r"pub struct Indivisible<([^>{]+)>", text)
    if not struct_match:
        violations.append(f"AbstractDomain check: could not locate `pub struct Indivisible` in {cut_rs.as_posix()}")
    elif "AbstractDomain" in struct_match.group(1):
        violations.append(
            f"AbstractDomain check: `Indivisible<C, T>` itself must not require `T: AbstractDomain` ({cut_rs.as_posix()})"
        )
    return violations


def check_cross_observer_scoped_boundaries(core_src: pathlib.Path) -> list[str]:
    """Verify Lv.2..Lv.4 cross-observer APIs do not leak `Scoped<` or retired `*_scoped*` forwarders."""
    violations: list[str] = []
    for layer in ("stigmergy", "drivers", "runtime"):
        layer_dir = core_src / layer
        if not layer_dir.is_dir():
            continue
        for rs_file in sorted(layer_dir.rglob("*.rs")):
            if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
                continue
            lines = rs_file.read_text(encoding="utf-8").splitlines()
            non_test_lines = strip_cfg_test_tail(lines)

            sig_buf: list[str] = []
            sig_start_line = 0
            sig_name = ""
            for lineno, raw_line in non_test_lines:
                code = raw_line.split("//", 1)[0]
                if not sig_buf:
                    m = FN_SIG_START_RE.match(code)
                    if m:
                        sig_name = m.group(1)
                        sig_start_line = lineno
                        sig_buf = [code]
                else:
                    sig_buf.append(code)

                if sig_buf:
                    joined = " ".join(part.strip() for part in sig_buf)
                    # Signature ends at body `{` or trait declaration `;`.
                    if "{" in code or ";" in code:
                        header = re.split(r"[{;]", joined, maxsplit=1)[0]
                        if "_scoped" in sig_name:
                            violations.append(
                                f"Retired `*_scoped*` signature: {rs_file.as_posix()}:{sig_start_line}: fn {sig_name}"
                            )
                        if SCOPED_BOUNDARY_TYPE_RE.search(header):
                            violations.append(
                                f"Cross-observer `Scoped<` signature leak: {rs_file.as_posix()}:{sig_start_line}: {header.strip()}"
                            )
                        sig_buf = []
                        sig_name = ""
    return violations


def run_checks(core_src: pathlib.Path) -> list[str]:
    """Run all architecture invariants and return a list of violation descriptions."""
    violations: list[str] = []

    for rel in RETIRED_LEGACY_PATHS:
        legacy = core_src / rel
        if legacy.exists():
            violations.append(f"Deprecated legacy path exists: {legacy.as_posix()}")

    for rel in REQUIRED_CANONICAL_PATHS:
        canon = core_src / rel
        if not canon.exists():
            violations.append(f"Missing required canonical path: {canon.as_posix()}")

    for label, sub, pat in DAG_AND_AXIOM_RULES:
        sub_path = core_src / sub
        if sub_path.is_file():
            rs_files = [sub_path]
        elif sub_path.is_dir():
            rs_files = sorted(sub_path.rglob("*.rs"))
        else:
            rs_files = []

        if not rs_files:
            violations.append(f"{label}: path {sub_path.as_posix()} is missing or contains no .rs files")
            continue

        for rs_file in rs_files:
            text = rs_file.read_text(encoding="utf-8")
            lines = text.splitlines()
            for match in pat.finditer(text):
                lineno = text.count("\n", 0, match.start()) + 1
                snippet = lines[lineno - 1].strip() if 1 <= lineno <= len(lines) else match.group(0)
                violations.append(f"{label}: {rs_file.as_posix()}:{lineno}: {snippet}")

    violations.extend(check_abstract_domain_invariant(core_src))
    violations.extend(check_cross_observer_scoped_boundaries(core_src))
    return violations


def bucket_for_path(core_src: pathlib.Path, rs_file: pathlib.Path) -> str | None:
    """Map a core source file to its Y-shaped DAG node bucket."""
    rel = rs_file.relative_to(core_src).parts
    if not rel:
        return None
    top = rel[0]
    if top == "drivers" and len(rel) >= 2 and rel[1] in ("spine", "ego"):
        return f"drivers::{rel[1]}"
    if top in ("liminal", "foundation", "stigmergy", "drivers", "runtime"):
        return top
    return None


def collect_dag_metrics(
    core_src: pathlib.Path,
) -> tuple[Counter[str], Counter[str], dict[tuple[str, str], int]]:
    """Collect per-layer file counts, LOC, and inter-layer `crate::` import edge counts."""
    files_per_node: Counter[str] = Counter()
    loc_per_node: Counter[str] = Counter()
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)

    for rs_file in sorted(core_src.rglob("*.rs")):
        src_bucket = bucket_for_path(core_src, rs_file)
        if src_bucket is None:
            continue
        text = rs_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        files_per_node[src_bucket] += 1
        loc_per_node[src_bucket] += len(lines)
        for _lineno, line in strip_cfg_test_tail(lines):
            code = line.split("//", 1)[0]
            for target in CRATE_DEP_RE.findall(code):
                dst_bucket = target if target in ("drivers::spine", "drivers::ego") else target.split("::")[0]
                if dst_bucket == "drivers" and src_bucket in ("drivers::spine", "drivers::ego"):
                    dst_bucket = "drivers"
                if dst_bucket != src_bucket:
                    edge_counts[(src_bucket, dst_bucket)] += 1

    return files_per_node, loc_per_node, edge_counts


def render_mermaid(core_src: pathlib.Path) -> str:
    """Render the live Y-shaped 5-layer DAG as a Mermaid `flowchart TD` diagram."""
    files_per_node, loc_per_node, edge_counts = collect_dag_metrics(core_src)

    def node_label(key: str, title: str) -> str:
        return f'{title} - {files_per_node[key]} files, {loc_per_node[key]} LOC'

    node_ids = {
        "liminal": "L0_liminal",
        "foundation": "L1_foundation",
        "stigmergy": "L2_stigmergy",
        "drivers::spine": "L3_spine",
        "drivers::ego": "L3_ego",
        "drivers": "L3_drivers_hub",
        "runtime": "L4_runtime",
    }

    lines = [
        "flowchart TD",
        f'    L0_liminal["{node_label("liminal", "Lv.0 liminal (Boundary Axioms & AbstractDomain)")}"]',
        f'    L1_foundation["{node_label("foundation", "Lv.1 foundation (Pure SWAR, RoPE & Cohomology)")}"]',
        f'    L2_stigmergy["{node_label("stigmergy", "Lv.2 stigmergy (Substrate, Attractors & Crystal)")}"]',
        '    subgraph L3_drivers["Lv.3 drivers (Orthogonal Axes & Tension)"]',
        f'        L3_spine["{node_label("drivers::spine", "drivers/spine (Reflex & EntropyController)")}"]',
        f'        L3_ego["{node_label("drivers::ego", "drivers/ego (Intent & EgoDifferentialNet)")}"]',
        f'        L3_drivers_hub["{node_label("drivers", "drivers/mod (DistortionTension)")}"]',
        "    end",
        f'    L4_runtime["{node_label("runtime", "Lv.4 runtime (Context, Gateway, Horizon, ABI)")}"]',
        "",
    ]

    ordered_pairs = [
        ("liminal", "foundation"),
        ("liminal", "stigmergy"),
        ("foundation", "stigmergy"),
        ("liminal", "drivers::spine"),
        ("foundation", "drivers::spine"),
        ("stigmergy", "drivers::spine"),
        ("liminal", "drivers::ego"),
        ("foundation", "drivers::ego"),
        ("stigmergy", "drivers::ego"),
        ("drivers", "drivers::spine"),
        ("drivers", "drivers::ego"),
        ("liminal", "runtime"),
        ("foundation", "runtime"),
        ("stigmergy", "runtime"),
        ("drivers::spine", "runtime"),
        ("drivers::ego", "runtime"),
        ("drivers", "runtime"),
    ]
    for provider, consumer in ordered_pairs:
        refs = edge_counts.get((consumer, provider), 0)
        if refs > 0:
            lines.append(f"    {node_ids[provider]} -->|{refs} refs| {node_ids[consumer]}")

    return "\n".join(lines) + "\n"


def format_hotspots(core_src: pathlib.Path, top_n: int = 12) -> str:
    """Analyze complexity, `unsafe`, and boundary density across production files in `src/core/src`."""
    rows: list[tuple[int, int, int, int, int, str]] = []
    for rs_file in sorted(core_src.rglob("*.rs")):
        if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
            continue
        lines = rs_file.read_text(encoding="utf-8").splitlines()
        total_loc = len(lines)
        non_test = strip_cfg_test_tail(lines)
        prod_loc = len(non_test)
        unsafe_hits = 0
        raw_mask_hits = 0
        scoped_refs = 0
        for _lineno, line in non_test:
            code = line.split("//", 1)[0]
            if re.search(r"\bunsafe\b", code):
                unsafe_hits += 1
            if "& !" in code:
                raw_mask_hits += 1
            if re.search(r"\b(?:Scoped|Indivisible)\b", code):
                scoped_refs += 1
        rel = rs_file.relative_to(core_src).as_posix()
        rows.append((prod_loc, total_loc, unsafe_hits, raw_mask_hits, scoped_refs, rel))

    rows.sort(reverse=True)
    header = [
        f"Core Architecture Hotspots (top {min(top_n, len(rows))} production files by non-test LOC):",
        f"{'File':<46} | {'ProdLOC':>7} | {'TotalLOC':>8} | {'Unsafe':>6} | {'Raw&!':>5} | {'Axiom':>5}",
        "-" * 92,
    ]
    for prod_loc, total_loc, unsafe_hits, raw_mask_hits, scoped_refs, rel in rows[:top_n]:
        header.append(
            f"{rel:<46} | {prod_loc:>7} | {total_loc:>8} | {unsafe_hits:>6} | {raw_mask_hits:>5} | {scoped_refs:>5}"
        )
    total_files = len(rows)
    total_prod = sum(r[0] for r in rows)
    total_all = sum(r[1] for r in rows)
    total_unsafe = sum(r[2] for r in rows)
    header.append("-" * 92)
    header.append(
        f"Summary: {total_files} prod files | {total_prod} prod LOC ({total_all} total LOC) | {total_unsafe} non-test unsafe tokens"
    )
    return "\n".join(header) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Verify LiminiKa 5-layer Y-shaped DAG, axiom boundaries, Mermaid graph, and hotspots."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify all DAG, isolation, axiom orthogonality, AbstractDomain, and Scoped boundary rules.",
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Emit a Mermaid flowchart TD representation of the live core dependency graph.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Optional output file path for --mermaid diagram text.",
    )
    parser.add_argument(
        "--hotspots",
        action="store_true",
        help="Print top core source files by LOC, unsafe tokens, raw bitmask operations, and axiom surface.",
    )
    args = parser.parse_args(argv)
    if not (args.check or args.mermaid or args.hotspots):
        args.check = True
    return args


def main(argv: list[str] | None = None) -> int:
    """Run requested architecture verification and analysis commands."""
    args = parse_args(argv)
    exit_code = 0

    if args.check:
        violations = run_checks(CORE_SRC)
        if violations:
            print(
                "Error: 5-layer unidirectional DAG, driver isolation, axiom orthogonality, or boundary violation detected:",
                *violations,
                sep="\n  ",
            )
            exit_code = 1
        else:
            print(
                "All 5-layer unidirectional DAG, spine/ego isolation, axiom orthogonality, AbstractDomain, and Scoped boundary invariants passed (0 violations)."
            )

    if args.hotspots:
        if args.check:
            print()
        print(format_hotspots(CORE_SRC), end="")

    if args.mermaid:
        diagram = render_mermaid(CORE_SRC)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(diagram.encode("utf-8"))
            print(f"\nWrote Mermaid diagram to {args.output.as_posix()}")
        else:
            if args.check or args.hotspots:
                print()
            print(diagram, end="")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
