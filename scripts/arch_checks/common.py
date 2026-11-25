# name: scripts/arch_checks/common.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Shared paths, Rust lexical helpers, and non-test source extraction utilities."""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CORE_SRC = REPO_ROOT / "src" / "core" / "src"
QUARANTINED_CLI_FILES: frozenset[str] = frozenset({"eval_samples.rs", "eval_runner.rs"})

PUB_TYPE_OR_TRAIT_RE = re.compile(
    r"\bpub(?:\s*\([^)]*\))?\s+(?:(?:unsafe|auto)\s+)*"
    r"(?:struct|enum|trait|type)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)

PUB_ITEM_DECL_RE = re.compile(
    r"\bpub(?:\s*\([^)]*\))?\s+(?:(?:unsafe|auto|const|async|extern(?:\s+r*#*\".*?\"#*)?)\s+)*"
    r"(?:struct|enum|trait|type|fn|const|static)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b"
)

CFG_TEST_ATTR_RE = re.compile(r"^#\[\s*cfg\s*\(\s*test\s*\)\s*\]$")
MOD_DECL_RE = re.compile(r"^(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\b")
OUTER_ATTR_START_RE = re.compile(r"^#\[")


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


def find_cfg_test_module_lines(lines: list[str]) -> set[int]:
    """Return 0-based line indices belonging exclusively to `#[cfg(test)] mod ...` items."""
    if not lines or not any("cfg" in line and "test" in line for line in lines):
        return set()
    clean_text = strip_literals_preserve_lines(strip_comments_preserve_lines("\n".join(lines)))
    clean_lines = clean_text.splitlines()
    while len(clean_lines) < len(lines):
        clean_lines.append("")

    excluded: set[int] = set()
    n = len(clean_lines)
    i = 0
    while i < n:
        if not CFG_TEST_ATTR_RE.match(clean_lines[i].strip()):
            i += 1
            continue

        j = i + 1
        while j < n:
            stripped = clean_lines[j].strip()
            if not stripped:
                j += 1
                continue
            if OUTER_ATTR_START_RE.match(stripped):
                bracket_depth = stripped.count("[") - stripped.count("]")
                j += 1
                while j < n and bracket_depth > 0:
                    bracket_depth += clean_lines[j].count("[") - clean_lines[j].count("]")
                    j += 1
                continue
            break

        if j >= n or not MOD_DECL_RE.match(clean_lines[j].lstrip()):
            i += 1
            continue

        brace_depth = 0
        saw_open_brace = False
        k = j
        while k < n:
            line_clean = clean_lines[k]
            if not saw_open_brace:
                brace_pos = line_clean.find("{")
                semi_pos = line_clean.find(";")
                if semi_pos != -1 and (brace_pos == -1 or semi_pos < brace_pos):
                    excluded.update(range(i, k + 1))
                    k += 1
                    break
                if brace_pos != -1:
                    saw_open_brace = True
                    for ch in line_clean[brace_pos:]:
                        if ch == "{":
                            brace_depth += 1
                        elif ch == "}":
                            brace_depth -= 1
                    if brace_depth <= 0:
                        excluded.update(range(i, k + 1))
                        k += 1
                        break
            else:
                for ch in line_clean:
                    if ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1
                if brace_depth <= 0:
                    excluded.update(range(i, k + 1))
                    k += 1
                    break
            k += 1
        else:
            excluded.update(range(i, n))
            k = n
        i = k

    return excluded


def strip_cfg_test_tail(lines: list[str]) -> list[tuple[int, str]]:
    """Return `(1-based lineno, line)` pairs excluding `#[cfg(test)] mod ...` blocks/declarations."""
    excluded = find_cfg_test_module_lines(lines)
    return [(idx, line) for idx, line in enumerate(lines, 1) if (idx - 1) not in excluded]


def blank_cfg_test_modules(source: str) -> str:
    """Replace `#[cfg(test)] mod ...` lines with empty lines to preserve 1-based line numbers."""
    lines = source.splitlines()
    excluded = find_cfg_test_module_lines(lines)
    return "\n".join("" if idx in excluded else line for idx, line in enumerate(lines))


def extract_non_test_code(source: str) -> str:
    """Strip `#[cfg(test)]` modules, comments, and string literals preserving line numbers."""
    truncated = blank_cfg_test_modules(source)
    clean_text = strip_comments_preserve_lines(truncated)
    return strip_literals_preserve_lines(clean_text)


def scan_until_top_level_terminator(text: str, start_idx: int) -> tuple[int, str]:
    """Scan from `start_idx` until `{` or `;` at zero paren/bracket/angle depth."""
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


def find_function_line_span(code_only: str, fn_name: str) -> set[int]:
    """Return 1-based line numbers spanning `fn <fn_name>(...) { ... }` in stripped code."""
    match = re.search(rf"\bfn\s+{re.escape(fn_name)}\b", code_only)
    if match is None:
        return set()
    brace_start = code_only.find("{", match.end())
    if brace_start == -1:
        return set()
    depth = 0
    end_idx = len(code_only)
    for pos in range(brace_start, len(code_only)):
        ch = code_only[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = pos + 1
                break
    start_line = code_only.count("\n", 0, match.start()) + 1
    end_line = code_only.count("\n", 0, end_idx) + 1
    return set(range(start_line, end_line + 1))


def scan_crate_prod_stats(
    src_dir: pathlib.Path,
    skip_names: frozenset[str] = frozenset(),
) -> tuple[int, int, str]:
    """Return `(prod_files, prod_loc, combined_non_test_code)` for a Rust crate `src/` dir."""
    files = 0
    prod_loc = 0
    chunks: list[str] = []
    if not src_dir.is_dir():
        return 0, 0, ""
    for rs_file in sorted(src_dir.rglob("*.rs")):
        if rs_file.name == "tests.rs" or "tests" in rs_file.parts or rs_file.name in skip_names:
            continue
        raw = rs_file.read_text(encoding="utf-8")
        files += 1
        prod_loc += len(strip_cfg_test_tail(raw.splitlines()))
        chunks.append(extract_non_test_code(raw))
    return files, prod_loc, "\n".join(chunks)


def find_banned_public_decls(
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
