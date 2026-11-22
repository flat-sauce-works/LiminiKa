// name: src/dsl/src/ast.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Abstract Syntax Tree (AST) definitions for LiminiKa DSL (`.lmnk`) specifications.

/// Top-level declaration in a `.lmnk` specification.
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    /// Hardware target and VRAM budget declaration.
    Target(TargetDecl),
    /// Persona topology and attractor rule declaration.
    Persona(PersonaDecl),
    /// Active runtime session binding a persona and steering mode.
    Session(SessionDecl),
}

/// Hardware target and deployment allocation descriptor.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TargetDecl {
    /// Target identifier name.
    pub name: String,
    /// Path or URI to the base LLM weights.
    pub model_path: String,
    /// Sidecar VRAM budget in MiB.
    pub vram_budget_mb: u32,
    /// Optional export container path.
    pub export_path: String,
}

/// Persona topology, attractor dynamics, and inheritance definition.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PersonaDecl {
    /// Persona identifier name.
    pub name: String,
    /// Optional parent preset or base persona path (`extends "..."`).
    pub parent_preset: Option<String>,
    /// Positive attraction and phase-conjugate repulsion rules.
    pub attractors: AttractorsBlock,
}

/// Attractor rule block (positive attraction and phase-conjugate repulsion).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AttractorsBlock {
    /// Positive attraction rules toward fixture/document sources.
    pub attract_rules: Vec<AttractRule>,
    /// Phase-conjugate repulsion rules suppressing unwanted traits or premises.
    pub repel_rules: Vec<RepelRule>,
}

/// Positive attraction rule toward a document or fixture anchor.
#[derive(Debug, Clone, PartialEq)]
pub struct AttractRule {
    /// Source file path or identifier.
    pub source_path: String,
    /// Normalized attraction weight (`> 0.0`).
    pub weight: f32,
}

/// Phase-conjugate repulsion rule suppressing out-of-character behaviors.
#[derive(Debug, Clone, PartialEq)]
pub struct RepelRule {
    /// Pattern string to suppress.
    pub target_pattern: String,
    /// Normalized repulsion weight (`> 0.0`).
    pub weight: f32,
}

/// Runtime session binding a persona and execution mode.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionDecl {
    /// Session identifier name.
    pub name: String,
    /// Referenced persona name (`using <Persona>`).
    pub persona_ref: String,
    /// Optional referenced target name (`on <Target>`).
    pub target_ref: Option<String>,
    /// Session steering mode (`mode = <ident>`).
    pub mode: Option<String>,
}
