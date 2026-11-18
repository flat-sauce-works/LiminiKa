// name: src/dsl/src/parser.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-dependency recursive-descent parser for `.lmnk` declarative files.

use crate::ast::{
    AttractRule, AttractorsBlock, Declaration, PersonaDecl, RepelRule, SessionDecl, TargetDecl,
};

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

fn tokenize(src: &str) -> Result<Vec<Token<'_>>, String> {
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
                out.push(Token::LBrace);
                i += 1;
            }
            b'}' => {
                out.push(Token::RBrace);
                i += 1;
            }
            b'@' => {
                out.push(Token::At);
                i += 1;
            }
            b'=' => {
                out.push(Token::Eq);
                i += 1;
            }
            b'"' => {
                i += 1;
                let start = i;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\n' {
                        return Err("unterminated string literal in .lmnk".to_string());
                    }
                    i += 1;
                }
                if i >= bytes.len() {
                    return Err("unterminated string literal at EOF".to_string());
                }
                out.push(Token::Str(&src[start..i]));
                i += 1;
            }
            c if c.is_ascii_digit() || c == b'-' => {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
                    i += 1;
                }
                let raw = &src[start..i];
                let val = raw
                    .parse::<f32>()
                    .map_err(|_| format!("invalid numeric literal '{raw}'"))?;
                out.push(Token::Number(val));
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                i += 1;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-')
                {
                    i += 1;
                }
                out.push(Token::Ident(&src[start..i]));
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
    loop {
        match cur.peek() {
            Some(Token::RBrace) => {
                cur.next();
                break;
            }
            Some(Token::Ident("attractors")) => {
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
    loop {
        match cur.peek() {
            Some(Token::RBrace) => {
                cur.next();
                break;
            }
            Some(Token::Ident("model")) => {
                cur.next();
                cur.expect_eq("target model")?;
                target.model_path = cur.expect_str("target model path")?;
            }
            Some(Token::Ident("vram_budget_mb")) => {
                cur.next();
                cur.expect_eq("target vram_budget_mb")?;
                let v = cur.expect_number("target vram_budget_mb")?;
                if !v.is_finite() || v <= 0.0 {
                    return Err("target vram_budget_mb must be positive".to_string());
                }
                target.vram_budget_mb = v as u32;
            }
            Some(Token::Ident("export")) => {
                cur.next();
                cur.expect_eq("target export")?;
                target.export_path = cur.expect_str("target export path")?;
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
/// Returns a descriptive syntax error string when tokenization or grammar matching fails.
pub fn parse_lmnk_program(input: &str) -> Result<Vec<Declaration>, String> {
    let tokens = tokenize(input)?;
    let mut cur = Cursor {
        toks: &tokens,
        pos: 0,
    };
    let mut decls = Vec::new();
    while let Some(tok) = cur.peek() {
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
