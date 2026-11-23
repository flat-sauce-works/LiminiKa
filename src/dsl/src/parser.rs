// name: src/dsl/src/parser.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-dependency recursive-descent parser for `.lmnk` declarative files.

use crate::ast::{
    AttractRule, AttractorsBlock, Declaration, PersonaDecl, RepelRule, SessionDecl, TargetDecl,
};

/// Hard upper bound on `.lmnk` source text size (`1 MiB`).
pub const LMNK_DSL_MAX_INPUT_BYTES: usize = 1024 * 1024;
/// Hard upper bound on lexed `.lmnk` tokens.
pub const LMNK_DSL_MAX_TOKENS: usize = 65_536;
/// Hard upper bound on top-level `.lmnk` declarations.
pub const LMNK_DSL_MAX_DECLARATIONS: usize = 256;
/// Hard upper bound on `attract` + `repel` rules per persona block.
pub const LMNK_DSL_MAX_RULES_PER_PERSONA: usize = 1_024;

#[derive(Debug, Clone, PartialEq)]
enum Token<'a> {
    Ident(&'a str),
    Str(&'a str),
    Number(f32),
    LBrace,
    RBrace,
    At,
    Eq,
}

fn push_token<'a>(out: &mut Vec<Token<'a>>, tok: Token<'a>) -> Result<(), String> {
    if out.len() >= LMNK_DSL_MAX_TOKENS {
        return Err(format!(
            ".lmnk token count exceeds maximum {LMNK_DSL_MAX_TOKENS}"
        ));
    }
    out.push(tok);
    Ok(())
}

fn tokenize(src: &str) -> Result<Vec<Token<'_>>, String> {
    if src.len() > LMNK_DSL_MAX_INPUT_BYTES {
        return Err(format!(
            ".lmnk input size ({} B) exceeds maximum {LMNK_DSL_MAX_INPUT_BYTES}",
            src.len()
        ));
    }
    let bytes = src.as_bytes();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        // Skip `#` and `//` line comments.
        if b == b'#' || (b == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        match b {
            b'{' => {
                push_token(&mut out, Token::LBrace)?;
                i += 1;
            }
            b'}' => {
                push_token(&mut out, Token::RBrace)?;
                i += 1;
            }
            b'@' => {
                push_token(&mut out, Token::At)?;
                i += 1;
            }
            b'=' => {
                push_token(&mut out, Token::Eq)?;
                i += 1;
            }
            b'"' => {
                i += 1;
                let start = i;
                while i < bytes.len() && bytes[i] != b'"' {
                    let ch = bytes[i];
                    if ch == b'\n' {
                        return Err("unterminated string literal in .lmnk".to_string());
                    }
                    if ch < 0x20 || ch == 0x7F {
                        return Err(format!(
                            "control character 0x{ch:02X} rejected inside .lmnk string literal"
                        ));
                    }
                    i += 1;
                }
                if i >= bytes.len() {
                    return Err("unterminated string literal at EOF".to_string());
                }
                push_token(&mut out, Token::Str(&src[start..i]))?;
                i += 1;
            }
            c if c.is_ascii_digit() || c == b'-' => {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                let raw = &src[start..i];
                let val = raw
                    .parse::<f32>()
                    .map_err(|_| format!("invalid numeric literal '{raw}'"))?;
                if !val.is_finite() {
                    return Err(format!("non-finite numeric literal '{raw}'"));
                }
                push_token(&mut out, Token::Number(val))?;
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                i += 1;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
                {
                    i += 1;
                }
                push_token(&mut out, Token::Ident(&src[start..i]))?;
            }
            other => {
                return Err(format!(
                    "unexpected character '{}' in .lmnk at byte {}",
                    other as char, i
                ));
            }
        }
    }
    Ok(out)
}

struct Cursor<'a> {
    toks: &'a [Token<'a>],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<&Token<'a>> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<&Token<'a>> {
        let t = self.toks.get(self.pos);
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect_ident(&mut self, ctx: &str) -> Result<String, String> {
        match self.next() {
            Some(Token::Ident(s)) => Ok((*s).to_string()),
            other => Err(format!("expected identifier for {ctx}, found {other:?}")),
        }
    }

    fn expect_str(&mut self, ctx: &str) -> Result<String, String> {
        match self.next() {
            Some(Token::Str(s)) => Ok((*s).to_string()),
            other => Err(format!(
                "expected string literal for {ctx}, found {other:?}"
            )),
        }
    }

    fn expect_number(&mut self, ctx: &str) -> Result<f32, String> {
        match self.next() {
            Some(Token::Number(v)) => Ok(*v),
            other => Err(format!("expected number for {ctx}, found {other:?}")),
        }
    }

    fn expect_lbrace(&mut self, ctx: &str) -> Result<(), String> {
        match self.next() {
            Some(Token::LBrace) => Ok(()),
            other => Err(format!("expected '{{' for {ctx}, found {other:?}")),
        }
    }

    fn expect_at(&mut self, ctx: &str) -> Result<(), String> {
        match self.next() {
            Some(Token::At) => Ok(()),
            other => Err(format!("expected '@' for {ctx}, found {other:?}")),
        }
    }

    fn expect_eq(&mut self, ctx: &str) -> Result<(), String> {
        match self.next() {
            Some(Token::Eq) => Ok(()),
            other => Err(format!("expected '=' for {ctx}, found {other:?}")),
        }
    }
}

fn parse_attractors_block(cur: &mut Cursor<'_>) -> Result<AttractorsBlock, String> {
    cur.expect_lbrace("attractors")?;
    let mut block = AttractorsBlock::default();
    loop {
        match cur.peek() {
            Some(Token::RBrace) => {
                cur.next();
                break;
            }
            Some(Token::Ident("attract")) => {
                if block.attract_rules.len() + block.repel_rules.len()
                    >= LMNK_DSL_MAX_RULES_PER_PERSONA
                {
                    return Err(format!(
                        "attractors rule count exceeds maximum {LMNK_DSL_MAX_RULES_PER_PERSONA}"
                    ));
                }
                cur.next();
                let source_path = cur.expect_str("attract source path")?;
                cur.expect_at("attract weight")?;
                let weight = cur.expect_number("attract weight")?;
                block.attract_rules.push(AttractRule {
                    source_path,
                    weight,
                });
            }
            Some(Token::Ident("repel")) => {
                if block.attract_rules.len() + block.repel_rules.len()
                    >= LMNK_DSL_MAX_RULES_PER_PERSONA
                {
                    return Err(format!(
                        "attractors rule count exceeds maximum {LMNK_DSL_MAX_RULES_PER_PERSONA}"
                    ));
                }
                cur.next();
                let target_pattern = cur.expect_str("repel pattern")?;
                cur.expect_at("repel weight")?;
                let weight = cur.expect_number("repel weight")?;
                block.repel_rules.push(RepelRule {
                    target_pattern,
                    weight,
                });
            }
            other => {
                return Err(format!("unexpected token in attractors block: {other:?}"));
            }
        }
    }
    Ok(block)
}

fn parse_persona_decl(cur: &mut Cursor<'_>) -> Result<PersonaDecl, String> {
    let name = cur.expect_ident("persona name")?;
    let mut parent_preset = None;
    if matches!(cur.peek(), Some(Token::Ident("extends"))) {
        cur.next();
        parent_preset = Some(cur.expect_str("persona extends")?);
    }
    cur.expect_lbrace("persona body")?;
    let mut attractors = AttractorsBlock::default();
    let mut seen_attractors = false;
    loop {
        match cur.peek() {
            Some(Token::RBrace) => {
                cur.next();
                break;
            }
            Some(Token::Ident("attractors")) => {
                if seen_attractors {
                    return Err(format!("duplicate 'attractors' block in persona '{name}'"));
                }
                seen_attractors = true;
                cur.next();
                attractors = parse_attractors_block(cur)?;
            }
            other => {
                return Err(format!("unexpected token in persona '{name}': {other:?}"));
            }
        }
    }
    Ok(PersonaDecl {
        name,
        parent_preset,
        attractors,
    })
}

fn parse_session_decl(cur: &mut Cursor<'_>) -> Result<SessionDecl, String> {
    let name = cur.expect_ident("session name")?;
    match cur.next() {
        Some(Token::Ident("using")) => {}
        other => {
            return Err(format!(
                "expected 'using' after session '{name}', found {other:?}"
            ));
        }
    }
    let persona_ref = cur.expect_ident("session persona reference")?;
    let mut target_ref = None;
    if matches!(cur.peek(), Some(Token::Ident("on"))) {
        cur.next();
        target_ref = Some(cur.expect_ident("session target reference")?);
    }
    cur.expect_lbrace("session body")?;
    let mut mode = None;
    loop {
        match cur.peek() {
            Some(Token::RBrace) => {
                cur.next();
                break;
            }
            Some(Token::Ident("mode")) => {
                if mode.is_some() {
                    return Err(format!("duplicate 'mode' field in session '{name}'"));
                }
                cur.next();
                cur.expect_eq("session mode")?;
                mode = Some(cur.expect_ident("session mode value")?);
            }
            other => {
                return Err(format!("unexpected token in session '{name}': {other:?}"));
            }
        }
    }
    Ok(SessionDecl {
        name,
        persona_ref,
        target_ref,
        mode,
    })
}

fn parse_target_decl(cur: &mut Cursor<'_>) -> Result<TargetDecl, String> {
    let name = cur.expect_ident("target name")?;
    cur.expect_lbrace("target body")?;
    let mut target = TargetDecl {
        name: name.clone(),
        ..Default::default()
    };
    let mut seen_model = false;
    let mut seen_vram = false;
    let mut seen_export = false;
    loop {
        match cur.peek() {
            Some(Token::RBrace) => {
                cur.next();
                break;
            }
            Some(Token::Ident("model")) => {
                if seen_model {
                    return Err(format!("duplicate 'model' field in target '{name}'"));
                }
                seen_model = true;
                cur.next();
                cur.expect_eq("target model")?;
                target.model_path = cur.expect_str("target model path")?;
                if target.model_path.trim().is_empty() {
                    return Err(format!("target '{name}' model path must not be empty"));
                }
            }
            Some(Token::Ident("vram_budget_mb")) => {
                if seen_vram {
                    return Err(format!(
                        "duplicate 'vram_budget_mb' field in target '{name}'"
                    ));
                }
                seen_vram = true;
                cur.next();
                cur.expect_eq("target vram_budget_mb")?;
                let v = cur.expect_number("target vram_budget_mb")?;
                if !v.is_finite() || v.fract() != 0.0 || !(1.0..=65536.0).contains(&v) {
                    return Err(format!(
                        "target '{name}' vram_budget_mb ({v}) must be an integer in 1..=65536"
                    ));
                }
                target.vram_budget_mb = v as u32;
            }
            Some(Token::Ident("export")) => {
                if seen_export {
                    return Err(format!("duplicate 'export' field in target '{name}'"));
                }
                seen_export = true;
                cur.next();
                cur.expect_eq("target export")?;
                target.export_path = cur.expect_str("target export path")?;
                if target.export_path.trim().is_empty() {
                    return Err(format!("target '{name}' export path must not be empty"));
                }
            }
            other => {
                return Err(format!("unexpected token in target '{name}': {other:?}"));
            }
        }
    }
    Ok(target)
}

/// Parses a complete `.lmnk` specification into top-level AST [`Declaration`] entries.
///
/// # Errors
/// Returns a descriptive syntax error string when tokenization, resource limits, or grammar matching fails.
pub fn parse_lmnk_program(input: &str) -> Result<Vec<Declaration>, String> {
    let tokens = tokenize(input)?;
    let mut cur = Cursor {
        toks: &tokens,
        pos: 0,
    };
    let mut decls = Vec::new();
    while let Some(tok) = cur.peek() {
        if decls.len() >= LMNK_DSL_MAX_DECLARATIONS {
            return Err(format!(
                ".lmnk declaration count exceeds maximum {LMNK_DSL_MAX_DECLARATIONS}"
            ));
        }
        match tok {
            Token::Ident("persona") => {
                cur.next();
                decls.push(Declaration::Persona(parse_persona_decl(&mut cur)?));
            }
            Token::Ident("session") => {
                cur.next();
                decls.push(Declaration::Session(parse_session_decl(&mut cur)?));
            }
            Token::Ident("target") => {
                cur.next();
                decls.push(Declaration::Target(parse_target_decl(&mut cur)?));
            }
            other => {
                return Err(format!(
                    "expected top-level declaration ('persona', 'session', 'target'), found {other:?}"
                ));
            }
        }
    }
    if decls.is_empty() {
        return Err("empty .lmnk specification (expected at least one declaration)".to_string());
    }
    Ok(decls)
}
