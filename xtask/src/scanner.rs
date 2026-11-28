// name: xtask/src/scanner.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-dependency Rust lexical scanner, path walkers, and non-test source extraction utilities.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Quarantined CLI evaluation files exempt from downstream vocabulary and gate checks.
pub const QUARANTINED_CLI_FILES: &[&str] = &["eval_samples.rs", "eval_runner.rs"];

/// Returns `true` when `b` can start an ASCII Rust identifier.
#[inline]
pub fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

/// Returns `true` when `b` can continue an ASCII Rust identifier.
#[inline]
pub fn is_ident_cont(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Formats `path` relative to `base` using forward slashes.
pub fn rel_posix(base: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(base).unwrap_or(path);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// Returns `true` when `path` is `tests.rs` or resides inside a `tests/` directory.
pub fn is_test_path(path: &Path) -> bool {
    if path.file_name().and_then(|n| n.to_str()) == Some("tests.rs") {
        return true;
    }
    path.components()
        .any(|c| c.as_os_str().to_str() == Some("tests"))
}

/// Recursively collects all `.rs` files under `dir`, sorted by POSIX path.
pub fn walk_rs_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_ext_recursive(dir, "rs", false, &mut out);
    out.sort_by_key(|a| rel_posix(dir, a));
    out
}

/// Recursively collects all `.py` files under `dir` (skipping `__pycache__`), sorted by POSIX path.
pub fn walk_py_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_ext_recursive(dir, "py", true, &mut out);
    out.sort_by_key(|a| rel_posix(dir, a));
    out
}

fn walk_ext_recursive(dir: &Path, ext: &str, skip_pycache: bool, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if skip_pycache && path.file_name().and_then(|n| n.to_str()) == Some("__pycache__") {
                continue;
            }
            walk_ext_recursive(&path, ext, skip_pycache, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}

/// Returns the 1-based line number of `byte_offset` inside `text`.
pub fn line_number_at(text: &str, byte_offset: usize) -> usize {
    let end = byte_offset.min(text.len());
    text.as_bytes()[..end]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
        + 1
}

/// Splits a `snake_case` or `CamelCase` identifier into lowercase semantic tokens.
pub fn split_ident_tokens(ident: &str) -> BTreeSet<String> {
    let mut tokens = BTreeSet::new();
    for seg in ident.split('_') {
        if seg.is_empty() {
            continue;
        }
        let bytes = seg.as_bytes();
        let n = bytes.len();
        let mut i = 0;
        while i < n {
            if bytes[i].is_ascii_digit() {
                let start = i;
                while i < n && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                tokens.insert(seg[start..i].to_ascii_lowercase());
            } else if bytes[i].is_ascii_uppercase() {
                let start = i;
                if i + 1 < n && bytes[i + 1].is_ascii_uppercase() {
                    while i < n && bytes[i].is_ascii_uppercase() {
                        if i + 1 < n && bytes[i + 1].is_ascii_lowercase() {
                            break;
                        }
                        i += 1;
                    }
                    tokens.insert(seg[start..i].to_ascii_lowercase());
                } else {
                    i += 1;
                    while i < n && bytes[i].is_ascii_lowercase() {
                        i += 1;
                    }
                    tokens.insert(seg[start..i].to_ascii_lowercase());
                }
            } else if bytes[i].is_ascii_lowercase() {
                let start = i;
                while i < n && bytes[i].is_ascii_lowercase() {
                    i += 1;
                }
                tokens.insert(seg[start..i].to_ascii_lowercase());
            } else {
                i += 1;
            }
        }
    }
    if ident.to_ascii_lowercase().contains("action_hub") {
        tokens.insert("action_hub".to_string());
    }
    tokens
}

/// If `text[idx..]` starts a string, raw string (`r`, `br`, `cr`), or char literal, returns the
/// byte offset immediately following the literal. Lifetimes (`'a`, `'static`, `'_`) return `None`.
pub fn advance_past_literal(text: &str, idx: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    if idx >= n {
        return None;
    }

    let prev_is_ident = idx > 0 && is_ident_cont(bytes[idx - 1]);
    let ch = bytes[idx];

    // 1. Raw string literal: r"...", r#"..."#, br"...", br#"..."#, cr"...", cr#"..."#
    if !prev_is_ident {
        let r_idx = if ch == b'r' {
            Some(idx)
        } else if matches!(ch, b'b' | b'c') && idx + 1 < n && bytes[idx + 1] == b'r' {
            Some(idx + 1)
        } else {
            None
        };
        if let Some(ri) = r_idx {
            let mut j = ri + 1;
            let mut hashes = 0usize;
            while j < n && bytes[j] == b'#' {
                hashes += 1;
                j += 1;
            }
            if j < n && bytes[j] == b'"' {
                let mut closing = String::with_capacity(1 + hashes);
                closing.push('"');
                for _ in 0..hashes {
                    closing.push('#');
                }
                return match text[j + 1..].find(&closing) {
                    Some(rel) => Some(j + 1 + rel + closing.len()),
                    None => Some(n),
                };
            }
        }
    }

    // 2. Standard, byte, or C string literal: "...", b"...", c"..."
    let str_quote_idx = if ch == b'"' {
        Some(idx)
    } else if !prev_is_ident && matches!(ch, b'b' | b'c') && idx + 1 < n && bytes[idx + 1] == b'"' {
        Some(idx + 1)
    } else {
        None
    };
    if let Some(sq) = str_quote_idx {
        let mut j = sq + 1;
        while j < n {
            if bytes[j] == b'\\' {
                j += 2;
            } else if bytes[j] == b'"' {
                return Some(j + 1);
            } else {
                j += 1;
            }
        }
        return Some(n);
    }

    // 3. Char or byte-char literal ('x', '\n', '\x7f', '\u{...}', b'x') vs. lifetime ('a, '_)
    let char_quote_idx = if ch == b'\'' && !prev_is_ident {
        Some(idx)
    } else if !prev_is_ident && ch == b'b' && idx + 1 < n && bytes[idx + 1] == b'\'' {
        Some(idx + 1)
    } else {
        None
    };
    if let Some(q) = char_quote_idx {
        if q + 1 < n && bytes[q + 1] == b'\\' {
            let mut j = q + 2;
            while j < n {
                if bytes[j] == b'\\' {
                    j += 2;
                } else if bytes[j] == b'\'' {
                    return Some(j + 1);
                } else if bytes[j] == b'\n' {
                    break;
                } else {
                    j += 1;
                }
            }
        } else if q + 1 < n
            && let Some(first_char) = text[q + 1..].chars().next()
            && !matches!(first_char, '\'' | '\\' | '\n' | '\r')
        {
            let after = q + 1 + first_char.len_utf8();
            if after < n && bytes[after] == b'\'' {
                return Some(after + 1);
            }
        }
    }

    None
}

/// Replaces `//` line comments and nested `/* ... */` block comments with spaces, preserving `\n`.
pub fn strip_comments_preserve_lines(text: &str) -> String {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut out = String::with_capacity(n);
    let mut i = 0;

    while i < n {
        if i + 1 < n && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            let nl = text[i + 2..].find('\n').map(|r| i + 2 + r);
            match nl {
                None => {
                    for _ in i..n {
                        out.push(' ');
                    }
                    break;
                }
                Some(j) => {
                    for _ in i..j {
                        out.push(' ');
                    }
                    i = j;
                }
            }
            continue;
        }

        if i + 1 < n && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            let mut depth = 1usize;
            out.push_str("  ");
            i += 2;
            while i < n && depth > 0 {
                if i + 1 < n && bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if i + 1 < n && bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                } else if bytes[i] == b'\n' {
                    out.push('\n');
                    i += 1;
                } else {
                    out.push(' ');
                    i += 1;
                }
            }
            continue;
        }

        if let Some(next_i) = advance_past_literal(text, i) {
            out.push_str(&text[i..next_i]);
            i = next_i;
            continue;
        }

        let ch = text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

/// Replaces string and char literals with spaces while preserving line numbers.
pub fn strip_literals_preserve_lines(clean_text: &str) -> String {
    let n = clean_text.len();
    let mut out = String::with_capacity(n);
    let mut i = 0;

    while i < n {
        if let Some(next_i) = advance_past_literal(clean_text, i) {
            let nl_count = clean_text.as_bytes()[i..next_i]
                .iter()
                .filter(|&&b| b == b'\n')
                .count();
            if nl_count > 0 {
                for _ in 0..nl_count {
                    out.push('\n');
                }
            } else {
                out.push(' ');
            }
            i = next_i;
            continue;
        }
        let ch = clean_text[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn is_cfg_test_attr(trimmed: &str) -> bool {
    let Some(rest) = trimmed.strip_prefix('#') else {
        return false;
    };
    let rest = rest.trim_start();
    let Some(rest) = rest.strip_prefix('[') else {
        return false;
    };
    let Some(inner) = rest.trim_end().strip_suffix(']') else {
        return false;
    };
    let inner = inner.trim();
    let Some(rest) = inner.strip_prefix("cfg") else {
        return false;
    };
    let rest = rest.trim_start();
    let Some(rest) = rest.strip_prefix('(') else {
        return false;
    };
    let Some(arg) = rest.trim_end().strip_suffix(')') else {
        return false;
    };
    arg.trim() == "test"
}

fn is_mod_decl(trimmed_start: &str) -> bool {
    let mut rest = trimmed_start;
    if let Some(after_pub) = rest.strip_prefix("pub") {
        let bytes = after_pub.as_bytes();
        if bytes.first().is_none_or(|&b| !is_ident_cont(b)) {
            let after_ws = after_pub.trim_start();
            if let Some(after_lp) = after_ws.strip_prefix('(') {
                if let Some(rp) = after_lp.find(')') {
                    rest = after_lp[rp + 1..].trim_start();
                } else {
                    return false;
                }
            } else if after_pub.len() != after_ws.len() {
                rest = after_ws;
            } else {
                return false;
            }
        }
    }
    let Some(after_mod) = rest.strip_prefix("mod") else {
        return false;
    };
    if after_mod
        .as_bytes()
        .first()
        .is_none_or(|b| !b.is_ascii_whitespace())
    {
        return false;
    }
    let name_part = after_mod.trim_start();
    name_part
        .as_bytes()
        .first()
        .is_some_and(|&b| is_ident_start(b))
}

/// Returns 0-based line indices belonging exclusively to `#[cfg(test)] mod ...` items.
pub fn find_cfg_test_module_lines(lines: &[&str]) -> BTreeSet<usize> {
    if lines.is_empty()
        || !lines
            .iter()
            .any(|line| line.contains("cfg") && line.contains("test"))
    {
        return BTreeSet::new();
    }
    let joined = lines.join("\n");
    let clean_text = strip_literals_preserve_lines(&strip_comments_preserve_lines(&joined));
    let mut clean_lines: Vec<&str> = clean_text.split('\n').collect();
    while clean_lines.len() < lines.len() {
        clean_lines.push("");
    }

    let mut excluded = BTreeSet::new();
    let n = clean_lines.len();
    let mut i = 0;

    while i < n {
        if !is_cfg_test_attr(clean_lines[i].trim()) {
            i += 1;
            continue;
        }

        let mut j = i + 1;
        while j < n {
            let stripped = clean_lines[j].trim();
            if stripped.is_empty() {
                j += 1;
                continue;
            }
            if stripped.starts_with("#[") {
                let mut bracket_depth = stripped.bytes().filter(|&b| b == b'[').count() as isize
                    - stripped.bytes().filter(|&b| b == b']').count() as isize;
                j += 1;
                while j < n && bracket_depth > 0 {
                    bracket_depth += clean_lines[j].bytes().filter(|&b| b == b'[').count() as isize
                        - clean_lines[j].bytes().filter(|&b| b == b']').count() as isize;
                    j += 1;
                }
                continue;
            }
            break;
        }

        if j >= n || !is_mod_decl(clean_lines[j].trim_start()) {
            i += 1;
            continue;
        }

        let mut brace_depth = 0isize;
        let mut saw_open_brace = false;
        let mut k = j;
        let mut closed = false;
        while k < n {
            let line_clean = clean_lines[k];
            if !saw_open_brace {
                let brace_pos = line_clean.find('{');
                let semi_pos = line_clean.find(';');
                if let Some(sp) = semi_pos
                    && brace_pos.is_none_or(|bp| sp < bp)
                {
                    for idx in i..=k {
                        excluded.insert(idx);
                    }
                    k += 1;
                    closed = true;
                    break;
                }
                if let Some(bp) = brace_pos {
                    saw_open_brace = true;
                    for b in line_clean[bp..].bytes() {
                        if b == b'{' {
                            brace_depth += 1;
                        } else if b == b'}' {
                            brace_depth -= 1;
                        }
                    }
                    if brace_depth <= 0 {
                        for idx in i..=k {
                            excluded.insert(idx);
                        }
                        k += 1;
                        closed = true;
                        break;
                    }
                }
            } else {
                for b in line_clean.bytes() {
                    if b == b'{' {
                        brace_depth += 1;
                    } else if b == b'}' {
                        brace_depth -= 1;
                    }
                }
                if brace_depth <= 0 {
                    for idx in i..=k {
                        excluded.insert(idx);
                    }
                    k += 1;
                    closed = true;
                    break;
                }
            }
            k += 1;
        }
        if !closed {
            for idx in i..n {
                excluded.insert(idx);
            }
            k = n;
        }
        i = k;
    }

    excluded
}

/// Returns `(1-based lineno, line)` pairs excluding `#[cfg(test)] mod ...` blocks.
pub fn strip_cfg_test_tail<'a>(lines: &[&'a str]) -> Vec<(usize, &'a str)> {
    let excluded = find_cfg_test_module_lines(lines);
    lines
        .iter()
        .enumerate()
        .filter(|(idx, _)| !excluded.contains(idx))
        .map(|(idx, &line)| (idx + 1, line))
        .collect()
}

/// Replaces `#[cfg(test)] mod ...` lines with empty lines to preserve 1-based line numbers.
pub fn blank_cfg_test_modules(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let excluded = find_cfg_test_module_lines(&lines);
    lines
        .iter()
        .enumerate()
        .map(|(idx, &line)| if excluded.contains(&idx) { "" } else { line })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Strips `#[cfg(test)]` modules, comments, and string literals preserving 1-based line numbers.
pub fn extract_non_test_code(source: &str) -> String {
    let truncated = blank_cfg_test_modules(source);
    let clean = strip_comments_preserve_lines(&truncated);
    strip_literals_preserve_lines(&clean)
}

/// Scans from `start_idx` until `{` or `;` at zero paren/bracket/angle depth.
pub fn scan_until_top_level_terminator(text: &str, start_idx: usize) -> (usize, Option<u8>) {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut paren_depth = 0usize;
    let mut bracket_depth = 0usize;
    let mut angle_depth = 0usize;
    let mut j = start_idx;

    while j < n {
        if let Some(next_j) = advance_past_literal(text, j) {
            j = next_j;
            continue;
        }
        if j + 1 < n
            && ((bytes[j] == b'-' && bytes[j + 1] == b'>')
                || (bytes[j] == b'=' && bytes[j + 1] == b'>'))
        {
            j += 2;
            continue;
        }
        match bytes[j] {
            b'(' => paren_depth += 1,
            b')' => paren_depth = paren_depth.saturating_sub(1),
            b'[' => bracket_depth += 1,
            b']' => bracket_depth = bracket_depth.saturating_sub(1),
            b'<' => angle_depth += 1,
            b'>' => angle_depth = angle_depth.saturating_sub(1),
            b'{' | b';' if paren_depth == 0 && bracket_depth == 0 && angle_depth == 0 => {
                return (j, Some(bytes[j]));
            }
            _ => {}
        }
        j += 1;
    }

    (n, None)
}

/// Returns 1-based line numbers spanning `fn <fn_name>(...) { ... }` in stripped code.
pub fn find_function_line_span(code_only: &str, fn_name: &str) -> BTreeSet<usize> {
    let idents = iter_idents(code_only);
    let mut match_start = None;
    let mut match_end = 0usize;
    for pair in idents.windows(2) {
        if pair[0].1 == "fn" && pair[1].1 == fn_name {
            let between = &code_only[pair[0].0 + 2..pair[1].0];
            if !between.is_empty() && between.bytes().all(|b| b.is_ascii_whitespace()) {
                match_start = Some(pair[0].0);
                match_end = pair[1].0 + pair[1].1.len();
                break;
            }
        }
    }
    let Some(start) = match_start else {
        return BTreeSet::new();
    };
    let Some(rel_brace) = code_only[match_end..].find('{') else {
        return BTreeSet::new();
    };
    let brace_start = match_end + rel_brace;
    let mut depth = 0usize;
    let mut end_idx = code_only.len();
    for (offset, b) in code_only.as_bytes()[brace_start..].iter().enumerate() {
        if *b == b'{' {
            depth += 1;
        } else if *b == b'}' {
            depth -= 1;
            if depth == 0 {
                end_idx = brace_start + offset + 1;
                break;
            }
        }
    }
    let start_line = line_number_at(code_only, start);
    let end_line = line_number_at(code_only, end_idx);
    (start_line..=end_line).collect()
}

/// Returns all `(byte_offset, ident)` occurrences in `text`.
pub fn iter_idents(text: &str) -> Vec<(usize, &str)> {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < n && is_ident_cont(bytes[i]) {
                i += 1;
            }
            out.push((start, &text[start..i]));
        } else {
            i += 1;
        }
    }
    out
}

/// Skips whitespace starting at `idx`.
pub fn skip_ws(bytes: &[u8], mut idx: usize) -> usize {
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    idx
}

/// Parses an identifier at `idx`, returning `(ident, next_idx)`.
pub fn read_ident(text: &str, idx: usize) -> Option<(&str, usize)> {
    let bytes = text.as_bytes();
    if idx >= bytes.len() || !is_ident_start(bytes[idx]) {
        return None;
    }
    let mut end = idx + 1;
    while end < bytes.len() && is_ident_cont(bytes[end]) {
        end += 1;
    }
    Some((&text[idx..end], end))
}

/// Parses optional `pub` or `pub(...)` visibility prefix at `idx`.
/// Returns `Some(next_idx)` when visibility was matched (requiring trailing whitespace).
pub fn match_pub_vis(text: &str, idx: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    if idx > 0 && is_ident_cont(bytes[idx - 1]) {
        return None;
    }
    let (word, after_pub) = read_ident(text, idx)?;
    if word != "pub" {
        return None;
    }
    let after_ws = skip_ws(bytes, after_pub);
    if after_ws < bytes.len() && bytes[after_ws] == b'(' {
        let close = text[after_ws + 1..].find(')')?;
        let after_rp = after_ws + 1 + close + 1;
        let next = skip_ws(bytes, after_rp);
        if next > after_rp { Some(next) } else { None }
    } else if after_ws > after_pub {
        Some(after_ws)
    } else {
        None
    }
}

/// Finds all public item declarations (`pub(...) [qualifiers] <kind> <name>`) in stripped code.
/// When `types_and_traits_only` is `true`, matches only `struct`, `enum`, `trait`, `type`.
pub fn find_pub_items(code_only: &str, types_and_traits_only: bool) -> Vec<(usize, String)> {
    let bytes = code_only.as_bytes();
    let n = bytes.len();
    let mut out = Vec::new();
    let mut i = 0;

    while i < n {
        if let Some(mut cur) = match_pub_vis(code_only, i) {
            while let Some((kw, next)) = read_ident(code_only, cur) {
                if types_and_traits_only {
                    if matches!(kw, "unsafe" | "auto") {
                        let ws = skip_ws(bytes, next);
                        if ws > next {
                            cur = ws;
                            continue;
                        }
                    }
                    if matches!(kw, "struct" | "enum" | "trait" | "type") {
                        let ws = skip_ws(bytes, next);
                        if ws > next
                            && let Some((name, _)) = read_ident(code_only, ws)
                        {
                            out.push((i, name.to_string()));
                        }
                    }
                    break;
                } else {
                    if matches!(kw, "unsafe" | "auto" | "const" | "async") {
                        let ws = skip_ws(bytes, next);
                        if ws > next {
                            cur = ws;
                            continue;
                        }
                    }
                    if kw == "extern" {
                        let mut ws = skip_ws(bytes, next);
                        if ws > next {
                            if let Some(after_lit) = advance_past_literal(code_only, ws) {
                                ws = skip_ws(bytes, after_lit);
                            }
                            cur = ws;
                            continue;
                        }
                    }
                    if matches!(
                        kw,
                        "struct" | "enum" | "trait" | "type" | "fn" | "const" | "static"
                    ) {
                        let ws = skip_ws(bytes, next);
                        if ws > next
                            && let Some((name, _)) = read_ident(code_only, ws)
                        {
                            out.push((i, name.to_string()));
                        }
                    }
                    break;
                }
            }
            i += 3;
            continue;
        }
        i += 1;
    }

    out
}

/// Returns violations for any public item declaration whose identifier hits `banned_tokens`.
pub fn find_banned_public_decls(
    code_only: &str,
    rel_path: &str,
    banned_tokens: &[&str],
    label: &str,
) -> Vec<String> {
    let mut violations = Vec::new();
    for (offset, name) in find_pub_items(code_only, false) {
        let lineno = line_number_at(code_only, offset);
        let toks = split_ident_tokens(&name);
        let hits: Vec<&str> = toks
            .iter()
            .filter_map(|t| banned_tokens.iter().copied().find(|&b| b == t.as_str()))
            .collect();
        if !hits.is_empty() {
            violations.push(format!(
                "{label} (banned legacy token {}): {rel_path}:{lineno}: disallowed public declaration `{name}`",
                hits.join(", ")
            ));
        }
    }
    violations
}

/// Returns `(prod_files, prod_loc, combined_non_test_code)` for a Rust crate `src/` directory.
pub fn scan_crate_prod_stats(src_dir: &Path, skip_names: &[&str]) -> (usize, usize, String) {
    if !src_dir.is_dir() {
        return (0, 0, String::new());
    }
    let mut files = 0usize;
    let mut prod_loc = 0usize;
    let mut chunks = Vec::new();
    for rs_file in walk_rs_files(src_dir) {
        if is_test_path(&rs_file) {
            continue;
        }
        if let Some(name) = rs_file.file_name().and_then(|n| n.to_str())
            && skip_names.contains(&name)
        {
            continue;
        }
        let raw = fs::read_to_string(&rs_file).expect("read rs file");
        let lines: Vec<&str> = raw.lines().collect();
        files += 1;
        prod_loc += strip_cfg_test_tail(&lines).len();
        chunks.push(extract_non_test_code(&raw));
    }
    (files, prod_loc, chunks.join("\n"))
}
