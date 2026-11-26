// name: src/dsl/src/compiler.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Semantic validation and compilation summary for parsed `.lmnk` DSL programs.

use std::collections::HashSet;

use crate::ast::{Declaration, PersonaDecl, SessionDecl, TargetDecl};
use crate::parser::{LMNK_DSL_MAX_DECLARATIONS, LMNK_DSL_MAX_RULES_PER_PERSONA};

fn validate_safe_dsl_path(path: &str, ctx: &str) -> Result<(), String> {
    if path.trim().is_empty() {
        return Err(format!("{ctx} has an empty path"));
    }
    if path.bytes().any(|b| b < 0x20 || b == 0x7F) {
        return Err(format!(
            "{ctx} contains forbidden NUL or control characters"
        ));
    }
    if path.split(['/', '\\']).any(|seg| seg == "..") {
        return Err(format!(
            "{ctx} '{path}' contains forbidden path traversal segment '..'"
        ));
    }
    Ok(())
}

/// Validated summary produced by compiling a `.lmnk` specification.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledDslProgram {
    /// Declared hardware targets.
    pub targets: Vec<TargetDecl>,
    /// Declared persona topologies.
    pub personas: Vec<PersonaDecl>,
    /// Declared runtime sessions.
    pub sessions: Vec<SessionDecl>,
}

/// Validates semantic invariants across parsed `.lmnk` declarations.
///
/// Enforces:
/// - At least one `persona` or `session` declaration and bounded declaration/rule counts.
/// - Unique declaration names per namespace (`target`, `persona`, `session`).
/// - Sanitized path strings (no `..` traversal segments, NUL, or control bytes).
/// - Strictly positive, finite weights on all `attract` / `repel` rules and non-empty source paths.
/// - Every `session ... using <Persona>` resolves to a declared `persona` in the program.
/// - Optional `session ... on <Target>` resolves to a declared `target` in the program.
///
/// # Errors
/// Returns a descriptive validation error message on semantic violations.
pub fn compile_lmnk_declarations(decls: &[Declaration]) -> Result<CompiledDslProgram, String> {
    if decls.len() > LMNK_DSL_MAX_DECLARATIONS {
        return Err(format!(
            ".lmnk declaration count ({}) exceeds maximum {LMNK_DSL_MAX_DECLARATIONS}",
            decls.len()
        ));
    }
    let mut targets = Vec::new();
    let mut personas = Vec::new();
    let mut sessions = Vec::new();
    let mut seen_targets: HashSet<&str> = HashSet::new();
    let mut seen_personas: HashSet<&str> = HashSet::new();
    let mut seen_sessions: HashSet<&str> = HashSet::new();

    for d in decls {
        match d {
            Declaration::Target(t) => {
                if !seen_targets.insert(t.name.as_str()) {
                    return Err(format!("duplicate target declaration '{}'", t.name));
                }
                if !t.model_path.is_empty() {
                    validate_safe_dsl_path(&t.model_path, &format!("target '{}' model", t.name))?;
                }
                if !t.export_path.is_empty() {
                    validate_safe_dsl_path(&t.export_path, &format!("target '{}' export", t.name))?;
                }
                targets.push(t.clone());
            }
            Declaration::Persona(p) => {
                if !seen_personas.insert(p.name.as_str()) {
                    return Err(format!("duplicate persona declaration '{}'", p.name));
                }
                if let Some(ref parent) = p.parent_preset {
                    validate_safe_dsl_path(parent, &format!("persona '{}' extends", p.name))?;
                }
                if p.attractors.attract_rules.len() + p.attractors.repel_rules.len()
                    > LMNK_DSL_MAX_RULES_PER_PERSONA
                {
                    return Err(format!(
                        "persona '{}' rule count exceeds maximum {LMNK_DSL_MAX_RULES_PER_PERSONA}",
                        p.name
                    ));
                }
                for rule in &p.attractors.attract_rules {
                    validate_safe_dsl_path(
                        &rule.source_path,
                        &format!("persona '{}' attract rule", p.name),
                    )?;
                    if !rule.weight.is_finite() || rule.weight <= 0.0 {
                        return Err(format!(
                            "persona '{}' has non-positive or non-finite attract weight {}",
                            p.name, rule.weight
                        ));
                    }
                }
                for rule in &p.attractors.repel_rules {
                    if rule.target_pattern.trim().is_empty()
                        || rule.target_pattern.bytes().any(|b| b < 0x20 || b == 0x7F)
                    {
                        return Err(format!(
                            "persona '{}' has an empty or control-char repel pattern",
                            p.name
                        ));
                    }
                    if !rule.weight.is_finite() || rule.weight <= 0.0 {
                        return Err(format!(
                            "persona '{}' has non-positive or non-finite repel weight {}",
                            p.name, rule.weight
                        ));
                    }
                }
                personas.push(p.clone());
            }
            Declaration::Session(s) => {
                if !seen_sessions.insert(s.name.as_str()) {
                    return Err(format!("duplicate session declaration '{}'", s.name));
                }
                sessions.push(s.clone());
            }
        }
    }

    if personas.is_empty() && sessions.is_empty() {
        return Err("specification must declare at least one 'persona' or 'session'".to_string());
    }

    for s in &sessions {
        if !personas.iter().any(|p| p.name == s.persona_ref) {
            return Err(format!(
                "session '{}' references undeclared persona '{}'",
                s.name, s.persona_ref
            ));
        }
        if let Some(ref target_name) = s.target_ref
            && !targets.iter().any(|t| &t.name == target_name)
        {
            return Err(format!(
                "session '{}' references undeclared target '{}'",
                s.name, target_name
            ));
        }
    }

    Ok(CompiledDslProgram {
        targets,
        personas,
        sessions,
    })
}
