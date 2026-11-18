// name: src/dsl/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declarative DSL frontend crate (`.lmnk`) for LiminiKa.

pub mod ast;
pub mod compiler;
pub mod parser;

pub use ast::{
    AttractRule, AttractorsBlock, Declaration, PersonaDecl, RepelRule, SessionDecl, TargetDecl,
};
pub use compiler::{compile_lmnk_declarations, CompiledDslProgram};
pub use parser::parse_lmnk_program;

/// Returns the `liminika-dsl` crate version string from `Cargo.toml`.
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Parses and semantically validates a `.lmnk` specification string.
///
/// # Errors
/// Returns a syntax or semantic validation error description.
pub fn parse_and_validate_lmnk(input: &str) -> Result<CompiledDslProgram, String> {
    let decls = parse_lmnk_program(input)?;
    compile_lmnk_declarations(&decls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn parses_and_validates_poc_lmnk() {
        let poc = include_str!("../../../examples/dsl/limina_poc.lmnk");
        let program = parse_and_validate_lmnk(poc).expect("limina_poc.lmnk should validate");
        assert_eq!(program.personas.len(), 1);
        assert_eq!(program.personas[0].name, "Limina");
        assert_eq!(
            program.personas[0].parent_preset.as_deref(),
            Some("presets/persona/limina_base")
        );
        assert_eq!(program.personas[0].attractors.attract_rules.len(), 1);
        assert_eq!(program.sessions.len(), 1);
        assert_eq!(program.sessions[0].name, "DesktopAssistant");
        assert_eq!(program.sessions[0].persona_ref, "Limina");
        assert_eq!(
            program.sessions[0].mode.as_deref(),
            Some("creative_exploration")
        );
    }

    #[test]
    fn rejects_undeclared_persona_reference() {
        let bad = "session S using MissingPersona { mode = strict }";
        let err = parse_and_validate_lmnk(bad).unwrap_err();
        assert!(err.contains("MissingPersona"));
    }
}
