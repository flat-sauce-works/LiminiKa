// src/dsl/src/ast.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Abstract Syntax Tree (AST) definitions for LiminiKa DSL declarative specifications.

/// Top-level declaration module in LiminiKa DSL (.lmnk).
#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Target(TargetDecl),
    Persona(PersonaDecl),
    Session(SessionDecl),
}

/// Hardware target and deployment allocation descriptor.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TargetDecl {
    pub name: String,
    pub model_path: String,
    pub vram_budget_mb: u32,
    pub export_path: String,
}

/// Persona topology, attractor dynamics, and memory policy definition.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PersonaDecl {
    pub name: String,
    pub parent_preset: Option<String>,
    pub coherence_level: String,
    pub attractors: AttractorsBlock,
    pub style_adapters: Vec<StyleAdapterSpec>,
    pub memory_container: Option<MemorySpec>,
}

/// External Attractor Fields (Positive attraction & Phase-Conjugate Repulsion).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AttractorsBlock {
    pub attract_rules: Vec<AttractRule>,
    pub repel_rules: Vec<RepelRule>,
}

/// Positive attraction rule towards document or prompt anchor.
#[derive(Debug, Clone, PartialEq)]
pub struct AttractRule {
    pub source_path: String,
    pub weight: f32,
    pub focus_topic: Option<String>,
}

/// Phase-conjugate repulsion rule to suppress out-of-character behaviors.
#[derive(Debug, Clone, PartialEq)]
pub struct RepelRule {
    pub target_patterns: Vec<String>,
    pub weight: f32,
}

/// PSPM Sub-head style routing specification.
#[derive(Debug, Clone, PartialEq)]
pub struct StyleAdapterSpec {
    pub adapter_path: String,
    pub fact_ratio: f32,
    pub logic_ratio: f32,
    pub explore_ratio: f32,
}

/// Stigmergic memory & PPRC key-frame retention policy.
#[derive(Debug, Clone, PartialEq)]
pub struct MemorySpec {
    pub container_path: String,
    pub decay_half_life_tokens: u32,
    pub protected_fact_keys: Vec<String>,
}

/// Runtime dynamics and EDBC event engine session.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionDecl {
    pub name: String,
    pub persona_ref: String,
    pub target_ref: String,
    pub mode: String,
    pub event_rules: Vec<EventRule>,
}

/// Reactive EDBC event trigger rule.
#[derive(Debug, Clone, PartialEq)]
pub struct EventRule {
    pub condition: EventCondition,
    pub action: EventAction,
}

/// Event condition triggering topological steering.
#[derive(Debug, Clone, PartialEq)]
pub enum EventCondition {
    TopicMatch(String),
    HallucinationDetected,
    EntropySpike,
}

/// Topological phase steering action triggered by EDBC.
#[derive(Debug, Clone, PartialEq)]
pub enum EventAction {
    BoostPhase(f32),
    RepelPhase { damp: f32 },
    TriggerTunneling { energy_level: String },
}