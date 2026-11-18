// name: src/dsl/src/compiler.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Semantic validation and compilation summary for parsed `.lmnk` DSL programs.

use crate::ast::{Declaration, PersonaDecl, SessionDecl, TargetDecl};

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
/// - At least one `persona` or `session` declaration.
/// - Strictly positive, finite weights on all `attract` / `repel` rules and non-empty source paths.
/// - Every `session ... using <Persona>` resolves to a declared `persona` in the program.
/// - Optional `session ... on <Target>` resolves to a declared `target` in the program.
///
/// # Errors
/// Returns a descriptive validation error message on semantic violations.
pub fn compile_lmnk_declarations(decls: &[Declaration]) -> Result<CompiledDslProgram, String> {
    let mut targets = Vec::new();
    let mut personas = Vec::new();
    let mut sessions = Vec::new();

    for d in decls {
        match d {
            Declaration::Target(t) => targets.push(t.clone()),
            Declaration::Persona(p) => {
                for rule in &p.attractors.attract_rules {
                    if rule.source_path.trim().is_empty() {
                        return Err(format!(
                            "persona '{}' has an attract rule with an empty source path",
                            p.name
                        ));
                    }
                    if !rule.weight.is_finite() || rule.weight <= 0.0 {
                        return Err(format!(
                            "persona '{}' has non-positive or non-finite attract weight {}",
                            p.name, rule.weight
                        ));
                    }
                }
                for rule in &p.attractors.repel_rules {
                    if rule.target_pattern.trim().is_empty() {
                        return Err(format!(
                            "persona '{}' has a repel rule with an empty pattern",
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
            Declaration::Session(s) => sessions.push(s.clone()),
        }
    }

    for s in &sessions {
        if !personas.iter().any(|p| p.name == s.persona_ref) {
            return Err(format!(
                "session '{}' references undeclared persona '{}'",
                s.name, s.persona_ref
            ));
        }
        if let Some(ref target_name) = s.target_ref {
            if !targets.iter().any(|t| &t.name == target_name) {
                return Err(format!(
                    "session '{}' references undeclared target '{}'",
                    s.name, target_name
                ));
            }
        }
    }

    Ok(CompiledDslProgram {
        targets,
        personas,
        sessions,
    })
}
