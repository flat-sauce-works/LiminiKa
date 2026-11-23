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

PUB_TYPE_OR_TRAIT_RE = re.compile(
    r"\bpub(?:\s*\([^)]*\))?\s+(?:(?:unsafe|auto)\s+)*"
    r"(?:struct|enum|trait|type)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)

PUB_ITEM_DECL_RE = re.compile(
    r"\bpub(?:\s*\([^)]*\))?\s+(?:(?:unsafe|auto|const|async|extern(?:\s+r*#*\".*?\"#*)?)\s+)*"
    r"(?:struct|enum|trait|type|fn|const|static)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
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

DOWNSTREAM_BANNED_LEGACY_TOKENS: frozenset[str] = BANNED_LEGACY_DECL_TOKENS | frozenset(
    {
        "society",
        "domino",
    }
)

QUARANTINED_CLI_FILES: frozenset[str] = frozenset({"eval_samples.rs", "eval_runner.rs"})


def split_ident_tokens(ident: str) -> set[str]:
    """Split a snake_case or CamelCase identifier into lowercase semantic tokens."""
    tokens: set[str] = set()
    for seg in ident.split("_"):
        if not seg:
            continue
        sub = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", seg)
        tokens.update(s.lower() for s in sub)
    if "action_hub" in ident.lower():
        tokens.add("action_hub")
    return tokens


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
CRATE_DEP_RE = re.compile(r"\bcrate::(liminal|foundation|stigmergy|drivers(?:::(?:spine|ego))?|runtime)\b")


def strip_cfg_test_tail(lines: list[str]) -> list[tuple[int, str]]:
    """Return `(1-based lineno, line)` pairs excluding `#[cfg(test)]` module tails."""
    kept: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, 1):
        if line.split("//", 1)[0].strip() == "#[cfg(test)]":
            break
        kept.append((idx, line))
    return kept


def advance_past_literal(text: str, idx: int) -> int | None:
    """If `text[idx]` starts a string, raw string, or char literal, return the index after it."""
    n = len(text)
    if idx >= n:
        return None

    prev_is_ident = idx > 0 and (text[idx - 1].isalnum() or text[idx - 1] == "_")
    ch = text[idx]

    # 1. Raw string literal: r"...", r#"..."#, br"...", br#"..."#, cr"...", cr#"..."#
    if not prev_is_ident:
        r_idx = -1
        if ch == "r":
            r_idx = idx
        elif ch in ("b", "c") and idx + 1 < n and text[idx + 1] == "r":
            r_idx = idx + 1
        if r_idx != -1:
            j = r_idx + 1
            hashes = 0
            while j < n and text[j] == "#":
                hashes += 1
                j += 1
            if j < n and text[j] == '"':
                closing = '"' + ("#" * hashes)
                end = text.find(closing, j + 1)
                return n if end == -1 else end + len(closing)

    # 2. Standard, byte, or C string literal: "...", b"...", c"..."
    str_quote_idx = -1
    if ch == '"':
        str_quote_idx = idx
    elif not prev_is_ident and ch in ("b", "c") and idx + 1 < n and text[idx + 1] == '"':
        str_quote_idx = idx + 1
    if str_quote_idx != -1:
        j = str_quote_idx + 1
        while j < n:
            if text[j] == "\\":
                j += 2
            elif text[j] == '"':
                return j + 1
            else:
                j += 1
        return n

    # 3. Char or byte-char literal ('x', '\n', '\x7f', '\u{...}', b'x') vs. lifetime ('a, 'static, '_)
    char_quote_idx = -1
    if ch == "'" and not prev_is_ident:
        char_quote_idx = idx
    elif not prev_is_ident and ch == "b" and idx + 1 < n and text[idx + 1] == "'":
        char_quote_idx = idx + 1
    if char_quote_idx != -1:
        q = char_quote_idx
        if q + 1 < n and text[q + 1] == "\\":
            j = q + 2
            while j < n:
                if text[j] == "\\":
                    j += 2
                elif text[j] == "'":
                    return j + 1
                elif text[j] == "\n":
                    break
                else:
                    j += 1
        elif q + 2 < n and text[q + 1] not in ("'", "\\", "\n", "\r") and text[q + 2] == "'":
            return q + 3

    return None


def strip_comments_preserve_lines(text: str) -> str:
    """Replace `//` line comments and nested `/* ... */` block comments with spaces, preserving `\n`."""
    n = len(text)
    out: list[str] = []
    i = 0
    while i < n:
        if text[i:i + 2] == "//":
            j = text.find("\n", i + 2)
            if j == -1:
                out.append(" " * (n - i))
                break
            out.append(" " * (j - i))
            i = j
            continue

        if text[i:i + 2] == "/*":
            depth = 1
            out.append("  ")
            i += 2
            while i < n and depth > 0:
                if text[i:i + 2] == "/*":
                    depth += 1
                    out.append("  ")
                    i += 2
                elif text[i:i + 2] == "*/":
                    depth -= 1
                    out.append("  ")
                    i += 2
                else:
                    out.append("\n" if text[i] == "\n" else " ")
                    i += 1
            continue

        next_i = advance_past_literal(text, i)
        if next_i is not None:
            out.append(text[i:next_i])
            i = next_i
            continue

        out.append(text[i])
        i += 1

    return "".join(out)


def scan_until_top_level_terminator(text: str, start_idx: int) -> tuple[int, str]:
    """Scan from `start_idx` until `{` or `;` at `paren_depth == 0 and bracket_depth == 0 and angle_depth == 0`."""
    n = len(text)
    paren_depth = 0
    bracket_depth = 0
    angle_depth = 0
    j = start_idx

    while j < n:
        next_j = advance_past_literal(text, j)
        if next_j is not None:
            j = next_j
            continue

        two = text[j:j + 2]
        if two in ("->", "=>"):
            j += 2
            continue

        ch = text[j]
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "<":
            angle_depth += 1
        elif ch == ">":
            angle_depth = max(0, angle_depth - 1)
        elif ch in ("{", ";") and paren_depth == 0 and bracket_depth == 0 and angle_depth == 0:
            return j, ch

        j += 1

    return n, ""


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
    non_test_lines = strip_cfg_test_tail(source.splitlines())
    truncated_source = "\n".join(line for _lineno, line in non_test_lines)
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


def strip_literals_preserve_lines(clean_text: str) -> str:
    """Replace string and char literals with spaces while preserving line numbers."""
    out: list[str] = []
    n = len(clean_text)
    i = 0
    while i < n:
        next_i = advance_past_literal(clean_text, i)
        if next_i is not None:
            nl_count = clean_text.count("\n", i, next_i)
            out.append("\n" * nl_count if nl_count else " ")
            i = next_i
            continue
        out.append(clean_text[i])
        i += 1
    return "".join(out)


def _extract_non_test_code(source: str) -> str:
    """Strip `#[cfg(test)]` tails, comments, and string literals preserving line numbers."""
    non_test_lines = strip_cfg_test_tail(source.splitlines())
    truncated = "\n".join(line for _lineno, line in non_test_lines)
    clean_text = strip_comments_preserve_lines(truncated)
    return strip_literals_preserve_lines(clean_text)


def _find_banned_public_decls(
    code_only: str,
    rel_path: str,
    banned_tokens: frozenset[str],
    label: str,
) -> list[str]:
    """Return violations for any public item declaration whose identifier hits `banned_tokens`."""
    violations: list[str] = []
    for match in PUB_ITEM_DECL_RE.finditer(code_only):
        name = match.group("name")
        lineno = code_only.count("\n", 0, match.start()) + 1
        banned_hits = sorted(split_ident_tokens(name) & banned_tokens)
        if banned_hits:
            violations.append(
                f"{label} (banned legacy token {', '.join(banned_hits)}): "
                f"{rel_path}:{lineno}: disallowed public declaration `{name}`"
            )
    return violations


def scan_vocabulary_ownership_in_source(source: str, rel_path: str) -> list[str]:
    """Verify legacy declared-identifier bans and `*Substrate*` (`stigmergy`-only) ownership."""
    norm_rel = rel_path.replace("\\", "/").lstrip("/")
    if norm_rel.startswith("src/core/src/"):
        norm_rel = norm_rel[len("src/core/src/"):]
    if norm_rel.endswith("tests.rs") or "/tests/" in f"/{norm_rel}":
        return []

    code_only = _extract_non_test_code(source)
    violations = _find_banned_public_decls(
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


def check_downstream_vocabulary(repo_root: pathlib.Path) -> list[str]:
    """Verify legacy declared-identifier bans across `src/adapter/src/` and `src/cli/src/`."""
    violations: list[str] = []
    for crate in ("adapter", "cli"):
        crate_src = repo_root / "src" / crate / "src"
        if not crate_src.is_dir():
            continue
        for rs_file in sorted(crate_src.rglob("*.rs")):
            if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
                continue
            if crate == "cli" and rs_file.name in QUARANTINED_CLI_FILES:
                continue
            rel_path = rs_file.relative_to(repo_root).as_posix()
            code_only = _extract_non_test_code(rs_file.read_text(encoding="utf-8"))
            violations.extend(
                _find_banned_public_decls(
                    code_only,
                    rel_path,
                    DOWNSTREAM_BANNED_LEGACY_TOKENS,
                    "Downstream vocabulary",
                )
            )
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
    violations.extend(check_vocabulary_ownership(core_src))
    violations.extend(check_downstream_vocabulary(core_src.parent.parent.parent))
    violations.extend(check_topology_matrix_parity(core_src))
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
        f'    L2_stigmergy["{node_label("stigmergy", "Lv.2 stigmergy (Substrate & PhaseAnchors)")}"]',
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
        f"Summary: {total_files} prod files | {total_prod} prod LOC "
        f"({total_all} total LOC) | {total_unsafe} non-test unsafe tokens"
    )
    return "\n".join(header) + "\n"


def collect_liminal_map_metrics(core_src: pathlib.Path) -> dict[str, int]:
    """Scan live `src/core/src` production files for Abstraction <-> Concretion metrics."""
    metrics: dict[str, int] = {
        "foundation_files": 0,
        "foundation_prod_loc": 0,
        "axiom_files": 0,
        "axiom_gate_refs": 0,
        "grounded_iframe_refs": 0,
        "active_concretion_refs": 0,
        "abstract_bypass_refs": 0,
        "exterior_sandbox_refs": 0,
        "pframe_ego_refs": 0,
        "runtime_gate_refs": 0,
    }
    patterns = {
        "axiom_gate_refs": re.compile(
            r"\b(?:Scoped|Indivisible|rescope_from_|cut_by_lacuna|"
            r"bridge_to_shared|BitCut(?:8|32|64)?)\b"
        ),
        "grounded_iframe_refs": re.compile(
            r"\b(?:Grounded|PhaseAnchorRecord|PhaseProfile64|PhaseProfileRef64|"
            r"PhaseWordsRef64)\b"
        ),
        "active_concretion_refs": re.compile(
            r"\b(?:ActiveScope|PhaseSteeringEngine|ResidualAdapterEngine|"
            r"PhaseDeltaSlice|ResidualScale|Indivisible\s*<\s*Active)\b"
        ),
        "abstract_bypass_refs": re.compile(
            r"\b(?:Lacuna::Unobserved|Lacuna::SelfRef|"
            r"stationary_shortcut_ready|whole_token_steering_bypass)\b"
        ),
        "exterior_sandbox_refs": re.compile(
            r"\b(?:Lacuna::Exterior|slot_sandbox_mask|settle_sandbox_lane|"
            r"SandboxLifecycleEvent)\b"
        ),
        "pframe_ego_refs": re.compile(
            r"\b(?:DistortionDelta|EgoDifferentialNet|DistortionTension)\b"
        ),
        "runtime_gate_refs": re.compile(
            r"\b(?:LmnkContainerView|SystemSpineState|pinned_compute_mask|"
            r"CORE_TOPOLOGY)\b"
        ),
    }

    for rs_file in sorted(core_src.rglob("*.rs")):
        if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
            continue
        rel = rs_file.relative_to(core_src).parts
        non_test = strip_cfg_test_tail(rs_file.read_text(encoding="utf-8").splitlines())
        if rel and rel[0] == "foundation":
            metrics["foundation_files"] += 1
            metrics["foundation_prod_loc"] += len(non_test)
        if len(rel) >= 2 and rel[0] == "liminal" and rel[1] == "axiom":
            metrics["axiom_files"] += 1
        for _lineno, line in non_test:
            code = line.split("//", 1)[0]
            for key, pat in patterns.items():
                metrics[key] += len(pat.findall(code))

    return metrics


def render_liminal_map(core_src: pathlib.Path) -> str:
    """Render the static `liminal` Abstraction <-> Concretion Mermaid map and table."""
    m = collect_liminal_map_metrics(core_src)
    lines = [
        "```mermaid",
        "flowchart LR",
        '    subgraph ABS["Abstract Realm (Low-Cost Ground State)"]',
        (
            '        ABS_FND["Lv.1 foundation/ Pure Compute '
            f'({m["foundation_files"]} files, {m["foundation_prod_loc"]} prod LOC)"]'
        ),
        (
            '        ABS_IFR["Indivisible(Grounded) I-Frame PhaseAnchorRecord '
            f'({m["grounded_iframe_refs"]} refs)"]'
        ),
        (
            '        ABS_BYP["Lacuna::Unobserved and SelfRef Bypass '
            f'({m["abstract_bypass_refs"]} refs)"]'
        ),
        "    end",
        '    subgraph LIM["liminal Boundary Gates (Lv.0 Axiom + Lv.4 Runtime)"]',
        (
            '        LIM_AX["Lv.0 liminal/axiom seal / rescope / BitCut '
            f'({m["axiom_files"]} files, {m["axiom_gate_refs"]} refs)"]'
        ),
        (
            '        LIM_RT["Lv.4 LmnkContainerView + SystemSpineState '
            f'({m["runtime_gate_refs"]} refs)"]'
        ),
        "    end",
        '    subgraph CON["Concrete Realm (High-Cost Excited State)"]',
        (
            '        CON_ACT["Scoped + Indivisible(Active) RoPE / Rank-1 '
            f'({m["active_concretion_refs"]} refs)"]'
        ),
        (
            '        CON_EGO["EgoDifferentialNet P-Frame + DistortionTension '
            f'({m["pframe_ego_refs"]} refs)"]'
        ),
        (
            '        CON_EXT["Lacuna::Exterior Isolated Sandbox Lanes '
            f'({m["exterior_sandbox_refs"]} refs)"]'
        ),
        "    end",
        '    ABS_FND -.->|"Pure Kernel Supply (Zero-Scoped)"| CON_ACT',
        '    ABS_IFR <-->|"seal / ground I-Frame"| LIM_AX',
        '    ABS_BYP <-->|"4p(1-p) Aperture & Spine Bypass"| LIM_RT',
        '    LIM_AX -->|"rescope_from_delta / Activate"| CON_ACT',
        '    LIM_RT <-->|"Host Pin Floor & Tension Damping"| CON_EGO',
        '    LIM_AX <-->|"cut_by_lacuna / settle_sandbox_lane"| CON_EXT',
        "```",
        "",
        "## Static `liminal` Abstraction <-> Concretion Correspondence Table",
        "",
        (
            "| Abstract Realm (Ground State) | `liminal` Boundary Gate / Op | "
            "Concrete Realm (Excited State) | Live `src/core/src` Counts | "
            "`CoreTelemetryBank` Counter |"
        ),
        "| :--- | :--- | :--- | ---: | :--- |",
        (
            "| Lv.1 `foundation/` non-`Scoped` pure compute | "
            "Direct pure kernel call (no `liminal/axiom` import) | "
            "Lv.2+ `Indivisible(Active)` RoPE / Rank-1 engines | "
            f"{m['foundation_files']} files / {m['foundation_prod_loc']} LOC -> "
            f"{m['active_concretion_refs']} active refs | "
            "`stigmergy.decision_counts[0..7]` |"
        ),
        (
            "| `Indivisible(Grounded)` I-Frame `PhaseAnchorRecord` | "
            "`seal` / `rescope_from_boundary` / `CORE_TOPOLOGY` | "
            "`EgoDifferentialNet` P-Frame `DistortionDelta` | "
            f"{m['grounded_iframe_refs']} grounded refs <-> "
            f"{m['pframe_ego_refs']} P-Frame/tension refs | "
            "`pframe_chain_evals`, `pframe_total_hops`, `ego.intervention_count` |"
        ),
        (
            "| `Lacuna::SelfRef` stationary bypass | "
            "`SystemSpineState` + `pinned_compute_mask` | "
            "`DistortionTension` damped/excited steering | "
            f"{m['abstract_bypass_refs']} bypass refs <-> "
            f"{m['runtime_gate_refs']} spine/container refs | "
            "`whole_token_bypass_events`, `liminal.stationary_shortcut_hits`, "
            "`immediate_tension_damps`, `slow_tension_excites` |"
        ),
        (
            "| `Lacuna::Unobserved` zero-aperture | "
            "`cut_by_lacuna` / `LacunaPartition8` | "
            "`Lacuna::Exterior` sandbox (`slot_sandbox_mask`) | "
            f"{m['axiom_gate_refs']} axiom gate refs <-> "
            f"{m['exterior_sandbox_refs']} sandbox refs | "
            "`unobserved_aperture_zeros`, `exterior_sandbox_stages`, "
            "`liminal.slide_in_count`, `liminal.slide_out_count` |"
        ),
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Verify LiminiKa 5-layer Y-shaped DAG, axiom boundaries, Mermaid graph, and hotspots."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify DAG, isolation, axiom orthogonality, AbstractDomain, "
            "Scoped boundary, legacy declared-identifier bans, and Substrate ownership rules."
        ),
    )
    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Emit a Mermaid flowchart TD representation of the live core dependency graph.",
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
        help="Optional output file path for --mermaid or --liminal-map output.",
    )
    parser.add_argument(
        "--hotspots",
        action="store_true",
        help="Print top core source files by LOC, unsafe tokens, raw bitmask operations, and axiom surface.",
    )
    args = parser.parse_args(argv)
    if not (args.check or args.mermaid or args.hotspots or args.liminal_map):
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
                "Error: architecture, DAG, axiom orthogonality, boundary, "
                "or vocabulary ownership violation detected:",
                *violations,
                sep="\n  ",
            )
            exit_code = 1
        else:
            print(
                "All 5-layer DAG, spine/ego isolation, axiom orthogonality, "
                "AbstractDomain, Scoped boundary, legacy declared-identifier bans, "
                "and Substrate ownership invariants passed (0 violations)."
            )

    if args.hotspots:
        if args.check:
            print()
        print(format_hotspots(CORE_SRC), end="")

    if args.mermaid:
        diagram = render_mermaid(CORE_SRC)
        if args.output is not None and not args.liminal_map:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(diagram.encode("utf-8"))
            print(f"\nWrote Mermaid diagram to {args.output.as_posix()}")
        else:
            if args.check or args.hotspots:
                print()
            print(diagram, end="")

    if args.liminal_map:
        liminal_doc = render_liminal_map(CORE_SRC)
        if args.output is not None and not args.mermaid:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(liminal_doc.encode("utf-8"))
            print(f"\nWrote liminal boundary map to {args.output.as_posix()}")
        else:
            if args.check or args.hotspots or args.mermaid:
                print()
            print(liminal_doc, end="")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
