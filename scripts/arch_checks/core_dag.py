# name: scripts/arch_checks/core_dag.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Hard-error invariant checks for `liminika-core`'s 5-layer Y-shaped DAG and axioms."""

from __future__ import annotations

import pathlib
import re

from .common import (
    PUB_TYPE_OR_TRAIT_RE,
    advance_past_literal,
    blank_cfg_test_modules,
    extract_non_test_code,
    find_banned_public_decls,
    scan_until_top_level_terminator,
    strip_comments_preserve_lines,
)

DAG_AND_AXIOM_RULES: list[tuple[str, str, re.Pattern[str]]] = [
    (
        "Lv.0 liminal -> Lv.1..4",
        "liminal",
        re.compile(
            r"\b(?:crate::|(?:super::)+)"
            r"(?:(foundation|stigmergy|drivers|runtime)\b"
            r"|\{[^}]*\b(foundation|stigmergy|drivers|runtime)\b)"
        ),
    ),
    (
        "Lv.1 foundation -> Lv.2..4",
        "foundation",
        re.compile(
            r"\b(?:crate::|(?:super::)+)"
            r"(?:(stigmergy|drivers|runtime)\b"
            r"|\{[^}]*\b(stigmergy|drivers|runtime)\b)"
        ),
    ),
    (
        "Lv.1 foundation -> liminal::axiom / Scoped",
        "foundation",
        re.compile(
            r"\b(?:liminal::axiom\b|liminal::bitcut\b|BitCut(?:8|32|64)?\b|"
            r"Scoped\s*<|Scoped::|ScopedCut\b|Indivisible\s*<|Indivisible::|"
            r"Indivisible\b|IndivisiblePhase\b|\bAbstractDomain\b|Activatable\b|"
            r"Groundable\b|ActiveScope\b|GroundedScope\b|GroundedOrigin\b|"
            r"SpeculativeDiscovery\b|BoundaryViolation\b|VerifiedAligned\b|"
            r"BridgedDiscovery\b)"
        ),
    ),
    (
        "Lv.0 liminal/axiom -> bitcut",
        "liminal/axiom",
        re.compile(r"\b(?:bitcut|BitCut(?:8|32|64)?)\b"),
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
            r"\b(?:(?:super::)+|axiom::|liminal::)cut\b|\b(?:Scoped|ScopedCut|"
            r"Indivisible|IndivisiblePhase|Activatable|Groundable|ActiveScope|"
            r"GroundedScope|Cuttable|Definite|BoundaryViolation|"
            r"SpeculativeDiscovery)\b"
        ),
    ),
]

RETIRED_LEGACY_PATHS: tuple[str, ...] = (
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
)

REQUIRED_CANONICAL_PATHS: tuple[str, ...] = (
    "foundation/phase_ops/mod.rs",
    "stigmergy/substrate/mod.rs",
    "stigmergy/anchor/mod.rs",
    "runtime/topology.rs",
)

BANNED_LEGACY_DECL_TOKENS: frozenset[str] = frozenset(
    {
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
    }
)

ALLOWED_SUBSTRATE_DECLS: dict[str, tuple[str, ...]] = {
    "StigmergicSubstrate": ("stigmergy/substrate/mod.rs",),
    "SubstrateStepReport": ("stigmergy/substrate/mod.rs",),
    "PointerSubstrate": ("stigmergy/trail/traits.rs",),
    "StigmergicSubstrateOps": ("stigmergy/trail/traits.rs",),
    "SubstrateGatewayExt": ("runtime/gateway/orchestrator.rs",),
    "LmnkSubstrateOpaque": ("runtime/abi/handles.rs",),
    "LmnkSubstrateHandle": ("runtime/abi/handles.rs",),
    "lmnk_substrate_handle_t": ("runtime/abi/handles.rs",),
    "lmnk_substrate_step_report_t": ("runtime/abi/types.rs",),
}

SCOPED_BOUNDARY_TYPE_RE = re.compile(
    r"\b(?:Scoped\s*<|ScopedCut\b|VerifiedAligned\s*<|BoundaryViolation\s*<"
    r"|SpeculativeDiscovery\s*<|BridgedDiscovery\s*<)"
)
PUB_TRAIT_START_RE = re.compile(
    r"pub(?:\s*\([^)]*\))?\s+(?:(?:unsafe|auto)\s+)*trait\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)
FN_START_RE = re.compile(
    r"(?P<vis>pub(?:\s*\([^)]*\))?\s+)?"
    r"(?:(?:default|const|async|unsafe|extern(?:\s+r*#*\".*?\"#*)?)\s+)*"
    r"fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)


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
            "AbstractDomain check: `Indivisible<C, T>` itself must not require "
            f"`T: AbstractDomain` ({cut_rs.as_posix()})"
        )
    return violations


def scan_scoped_boundary_violations_in_source(source: str, source_label: str = "<memory>") -> list[str]:
    """Scan non-test Rust source text for public cross-observer `Scoped<` or `*_scoped*` signatures."""
    violations: list[str] = []
    truncated_source = blank_cfg_test_modules(source)
    clean_text = strip_comments_preserve_lines(truncated_source)

    n = len(clean_text)
    brace_depth = 0
    pub_trait_depths: list[int] = []
    i = 0

    while i < n:
        next_i = advance_past_literal(clean_text, i)
        if next_i is not None:
            i = next_i
            continue

        ch = clean_text[i]
        if ch == "{":
            brace_depth += 1
            i += 1
            continue
        if ch == "}":
            brace_depth = max(0, brace_depth - 1)
            while pub_trait_depths and pub_trait_depths[-1] > brace_depth:
                pub_trait_depths.pop()
            i += 1
            continue

        prev_is_ident = i > 0 and (clean_text[i - 1].isalnum() or clean_text[i - 1] == "_")
        if not prev_is_ident and (ch.isalpha() or ch == "_"):
            trait_match = PUB_TRAIT_START_RE.match(clean_text, i)
            if trait_match is not None:
                term_idx, term_ch = scan_until_top_level_terminator(clean_text, trait_match.end())
                if term_ch == "{":
                    brace_depth += 1
                    pub_trait_depths.append(brace_depth)
                i = term_idx + 1
                continue

            fn_match = FN_START_RE.match(clean_text, i)
            if fn_match is not None:
                sig_name = fn_match.group("name")
                has_pub_vis = fn_match.group("vis") is not None
                in_pub_trait = bool(pub_trait_depths) and brace_depth == pub_trait_depths[-1]
                is_public_sig = has_pub_vis or in_pub_trait

                term_idx, term_ch = scan_until_top_level_terminator(clean_text, fn_match.end())
                if is_public_sig:
                    sig_start_line = clean_text.count("\n", 0, fn_match.start()) + 1
                    header = " ".join(clean_text[fn_match.start():term_idx].split())
                    if "_scoped" in sig_name:
                        violations.append(
                            f"Retired `*_scoped*` signature: {source_label}:{sig_start_line}: fn {sig_name}"
                        )
                    if SCOPED_BOUNDARY_TYPE_RE.search(header):
                        violations.append(
                            f"Cross-observer `Scoped<` signature leak: {source_label}:{sig_start_line}: {header}"
                        )
                if term_ch == "{":
                    brace_depth += 1
                i = term_idx + 1
                continue

            while i < n and (clean_text[i].isalnum() or clean_text[i] == "_"):
                i += 1
            continue

        i += 1

    return violations


def check_cross_observer_scoped_boundaries(core_src: pathlib.Path) -> list[str]:
    """Verify Lv.2..Lv.4 public cross-observer APIs do not leak `Scoped<` or retired `*_scoped*` forwarders."""
    violations: list[str] = []
    for layer in ("stigmergy", "drivers", "runtime"):
        layer_dir = core_src / layer
        if not layer_dir.is_dir():
            continue
        for rs_file in sorted(layer_dir.rglob("*.rs")):
            if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
                continue
            source = rs_file.read_text(encoding="utf-8")
            violations.extend(scan_scoped_boundary_violations_in_source(source, rs_file.as_posix()))
    return violations


def scan_vocabulary_ownership_in_source(source: str, rel_path: str) -> list[str]:
    """Verify legacy declared-identifier bans and `*Substrate*` (`stigmergy`-only) ownership."""
    norm_rel = rel_path.replace("\\", "/").lstrip("/")
    if norm_rel.startswith("src/core/src/"):
        norm_rel = norm_rel[len("src/core/src/"):]
    if norm_rel.endswith("tests.rs") or "/tests/" in f"/{norm_rel}":
        return []

    code_only = extract_non_test_code(source)
    violations = find_banned_public_decls(
        code_only, norm_rel, BANNED_LEGACY_DECL_TOKENS, "Vocabulary ownership"
    )

    for match in PUB_TYPE_OR_TRAIT_RE.finditer(code_only):
        name = match.group("name")
        lineno = code_only.count("\n", 0, match.start()) + 1
        if "Substrate" in name:
            allowed_paths = ALLOWED_SUBSTRATE_DECLS.get(name, ())
            if norm_rel not in allowed_paths:
                violations.append(
                    f"Vocabulary ownership (Substrate=stigmergy-only): {norm_rel}:{lineno}: "
                    f"disallowed public declaration `{name}`"
                )
    return violations


def check_vocabulary_ownership(core_src: pathlib.Path) -> list[str]:
    """Verify legacy declared-identifier bans and `Substrate` (`stigmergy`-only) ownership."""
    violations: list[str] = []
    for rs_file in sorted(core_src.rglob("*.rs")):
        if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
            continue
        rel_path = rs_file.relative_to(core_src).as_posix()
        source = rs_file.read_text(encoding="utf-8")
        violations.extend(scan_vocabulary_ownership_in_source(source, rel_path))
    return violations


def check_topology_matrix_parity(core_src: pathlib.Path) -> list[str]:
    """Verify `CORE_TOPOLOGY` constants in `runtime/topology.rs` match the enforced 5-layer DAG."""
    topo_rs = core_src / "runtime" / "topology.rs"
    if not topo_rs.is_file():
        return [f"Topology parity check: missing {topo_rs.as_posix()}"]

    text = topo_rs.read_text(encoding="utf-8")
    violations: list[str] = []

    layer_m = re.search(
        r"pub\s+const\s+LMNK_TOPOLOGY_LAYER_COUNT:\s*u8\s*=\s*(\d+)\s*;",
        text,
    )
    if layer_m is None or int(layer_m.group(1)) != 5:
        got = layer_m.group(1) if layer_m else "missing"
        violations.append(
            f"Topology parity: LMNK_TOPOLOGY_LAYER_COUNT expected 5, got {got} ({topo_rs.as_posix()})"
        )

    chan_m = re.search(
        r"pub\s+const\s+LMNK_TOPOLOGY_CHANNEL_COUNT:\s*u8\s*=\s*(\d+)\s*;",
        text,
    )
    if chan_m is None or int(chan_m.group(1)) != 8:
        got = chan_m.group(1) if chan_m else "missing"
        violations.append(
            f"Topology parity: LMNK_TOPOLOGY_CHANNEL_COUNT expected 8, got {got} ({topo_rs.as_posix()})"
        )

    mat_m = re.search(
        r"pub\s+const\s+LMNK_TOPOLOGY_DAG_EDGE_MATRIX:\s*\[u8;\s*8\]\s*=\s*\[([^\]]+)\]\s*;",
        text,
        re.DOTALL,
    )
    expected_edges = [0b00000, 0b00001, 0b00011, 0b00111, 0b01111, 0, 0, 0]
    if mat_m is None:
        violations.append(
            f"Topology parity: missing LMNK_TOPOLOGY_DAG_EDGE_MATRIX ({topo_rs.as_posix()})"
        )
    else:
        raw_items = [tok.strip().replace("_", "") for tok in mat_m.group(1).split(",") if tok.strip()]
        try:
            parsed_edges = [int(tok, 0) for tok in raw_items]
        except ValueError:
            parsed_edges = []
        if parsed_edges != expected_edges:
            violations.append(
                f"Topology parity: LMNK_TOPOLOGY_DAG_EDGE_MATRIX expected "
                f"{expected_edges}, got {parsed_edges} ({topo_rs.as_posix()})"
            )

    return violations


def run_core_checks(core_src: pathlib.Path) -> list[str]:
    """Run all `liminika-core` DAG, axiom, Scoped, vocabulary, and topology invariants."""
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
    violations.extend(check_vocabulary_ownership(core_src))
    violations.extend(check_topology_matrix_parity(core_src))
    return violations
