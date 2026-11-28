// name: src/dsl/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declarative DSL frontend crate (`.lmnk`) for LiminiKa.

pub mod ast;
pub mod compiler;
pub mod parser;

pub use ast::{
    AttractRule, AttractorsBlock, Declaration, PersonaDecl, RepelRule, SessionDecl, TargetDecl,
};
pub use compiler::{CompiledDslProgram, compile_lmnk_declarations};
pub use parser::{
    LMNK_DSL_MAX_DECLARATIONS, LMNK_DSL_MAX_INPUT_BYTES, LMNK_DSL_MAX_RULES_PER_PERSONA,
    LMNK_DSL_MAX_TOKENS, parse_lmnk_program,
};

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

    #[test]
    fn rejects_adversarial_dsl_inputs_and_resource_limits() {
        // 1-4. Large resource-cap stress inputs (1 MiB / 65k tokens / 1k rules) are slow under
        // the Miri interpreter and contain zero `unsafe`, so skip them under `cfg(miri)`.
        #[cfg(not(miri))]
        {
            // 1. Oversized input (> 1 MiB)
            let huge = " ".repeat(LMNK_DSL_MAX_INPUT_BYTES + 1);
            assert!(parse_and_validate_lmnk(&huge).is_err());

            // 2. Token limit (> 65,536)
            let too_many_toks = "{ ".repeat(LMNK_DSL_MAX_TOKENS + 1);
            assert!(parse_and_validate_lmnk(&too_many_toks).is_err());

            // 3. Declaration limit (> 256)
            let mut many_decls = String::new();
            for i in 0..=LMNK_DSL_MAX_DECLARATIONS {
                many_decls.push_str(&format!("persona P{i} {{ }}\n"));
            }
            assert!(parse_and_validate_lmnk(&many_decls).is_err());

            // 4. Rule limit per persona (> 1,024)
            let mut many_rules = String::from("persona Flood {\n  attractors {\n");
            for i in 0..=LMNK_DSL_MAX_RULES_PER_PERSONA {
                many_rules.push_str(&format!("    attract \"presets/r{i}.lmnk\" @ 0.5\n"));
            }
            many_rules.push_str("  }\n}\n");
            assert!(parse_and_validate_lmnk(&many_rules).is_err());
        }

        // 5. NUL and ASCII control chars inside string literals
        assert!(parse_and_validate_lmnk("persona P extends \"presets/a\0b\" { }").is_err());
        assert!(
            parse_and_validate_lmnk(
                "persona P { attractors { attract \"presets/\x1b[31m\" @ 0.5 } }"
            )
            .is_err()
        );

        // 6. Non-finite scientific-notation floats (1e50) and invalid vram_budget_mb
        assert!(
            parse_and_validate_lmnk("persona P { attractors { attract \"a.lmnk\" @ 1e50 } }")
                .is_err()
        );
        assert!(
            parse_and_validate_lmnk(
                "persona P {}\ntarget T { vram_budget_mb = 1.5 }\nsession S using P on T { }"
            )
            .is_err()
        );
        assert!(
            parse_and_validate_lmnk(
                "persona P {}\ntarget T { vram_budget_mb = 0 }\nsession S using P on T { }"
            )
            .is_err()
        );
        assert!(
            parse_and_validate_lmnk(
                "persona P {}\ntarget T { vram_budget_mb = 65537 }\nsession S using P on T { }"
            )
            .is_err()
        );

        // 7. Path traversal '..' across source_path, parent_preset, model_path, export_path
        assert!(
            parse_and_validate_lmnk(
                "persona P { attractors { attract \"../../etc/passwd\" @ 0.5 } }"
            )
            .is_err()
        );
        assert!(parse_and_validate_lmnk("persona P extends \"../secret\" { }").is_err());
        assert!(
            parse_and_validate_lmnk("persona P {}\ntarget T { model = \"models/../escape.gguf\" }")
                .is_err()
        );
        assert!(
            parse_and_validate_lmnk(
                "persona P {}\ntarget T { export = \"out\\\\..\\\\escape.lmnkc\" }"
            )
            .is_err()
        );

        // 8. Duplicate declaration names, duplicate fields, and target-only (empty persona/session)
        assert!(parse_and_validate_lmnk("persona P {}\npersona P {}").is_err());
        assert!(
            parse_and_validate_lmnk("persona P {}\nsession S using P {}\nsession S using P {}")
                .is_err()
        );
        assert!(parse_and_validate_lmnk("persona P { attractors {} attractors {} }").is_err());
        assert!(
            parse_and_validate_lmnk("persona P {}\nsession S using P { mode = a mode = b }")
                .is_err()
        );
        assert!(
            parse_and_validate_lmnk(
                "persona P {}\ntarget T { vram_budget_mb = 512 vram_budget_mb = 1024 }"
            )
            .is_err()
        );
        assert!(parse_and_validate_lmnk("target Solo { vram_budget_mb = 1024 }").is_err());
    }

    #[test]
    fn deterministic_xorshift64_1200_mutations_never_panic_on_poc_lmnk() {
        let base = include_str!("../../../examples/dsl/limina_poc.lmnk");
        assert!(parse_and_validate_lmnk(base).is_ok());

        let mut seed = 0xD51F_0200_9876_5432_u64;
        let mut next_u64 = || -> u64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let base_bytes = base.as_bytes();
        let trials: usize = if cfg!(miri) { 64 } else { 1200 };
        let min_rejected: usize = if cfg!(miri) { 20 } else { 500 };
        let mut rejected_count = 0usize;
        for trial in 0..trials {
            let mut mutated = base_bytes.to_vec();
            match trial % 4 {
                0 => {
                    let idx = (next_u64() as usize) % mutated.len();
                    mutated[idx] ^= 1u8 << ((next_u64() as u8) & 7);
                }
                1 => {
                    for _ in 0..3 {
                        let idx = (next_u64() as usize) % mutated.len();
                        mutated[idx] = (next_u64() as u8) & 0x7F;
                    }
                }
                2 => {
                    let trunc = (next_u64() as usize) % mutated.len();
                    mutated.truncate(trunc);
                }
                _ => {
                    let idx = (next_u64() as usize) % mutated.len();
                    mutated.insert(idx, 0);
                }
            }
            if let Ok(as_str) = std::str::from_utf8(&mutated) {
                if parse_and_validate_lmnk(as_str).is_err() {
                    rejected_count += 1;
                }
            } else {
                rejected_count += 1;
            }
        }
        assert!(rejected_count >= min_rejected);
    }
}
