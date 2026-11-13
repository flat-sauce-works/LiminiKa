// name: src/cli/src/eval_samples.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-dependency loader for real-LLM RoPE steering evaluation fixtures.
//!
//! Parses sectioned text fixtures (such as `tests/fixtures/gcso_eval_samples.txt`) into
//! in-memory attractor seeds (`facts`, `persona`, `guards`) and scored evaluation probes
//! (`probes`) so external model harnesses can run 3-way comparisons (Baseline vs. Oracle vs.
//! Steered) before finalizing `.gcso` v1.0 persistence.

/// Target slot category for an evaluation probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeSlot {
    /// Objective factual recall (`ObjectiveFact`, Slot 2).
    Fact,
    /// Stylistic persona consistency (`SubjectivePersona`, Slot 1).
    Persona,
    /// Contrastive hallucination / prompt-drift resistance (`ContradictionGuard`, Slot 3).
    Guard,
}

/// Single Q&A probe used to score Baseline, Oracle, and LiminiKa-steered outputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalProbe {
    /// Target attractor slot category.
    pub slot: ProbeSlot,
    /// User prompt sent to the model.
    pub prompt: String,
    /// Keywords expected in a successful response (or used to track target token log-probs).
    pub expected_keywords: Vec<String>,
    /// Keywords that indicate hallucination or contrastive guard failure.
    pub forbidden_keywords: Vec<String>,
}

/// Parsed collection of in-memory attractor seed texts and evaluation probes.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EvalSampleSuite {
    /// Factual statements for `ObjectiveFact` (Slot 2) calibration.
    pub facts: Vec<String>,
    /// Voice/style examples for `SubjectivePersona` (Slot 1) calibration.
    pub persona: Vec<String>,
    /// Undesired statements for `ContradictionGuard` (Slot 3) repulsion.
    pub guards: Vec<String>,
    /// Structured probes for automated 3-way evaluation and angle sweeps.
    pub probes: Vec<EvalProbe>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Section {
    None,
    Facts,
    Persona,
    Guards,
    Probes,
}

impl EvalSampleSuite {
    /// Parses a sectioned evaluation fixture string into an [`EvalSampleSuite`].
    ///
    /// Lines beginning with `#` and blank lines are ignored.
    ///
    /// # Errors
    ///
    /// Returns a descriptive error string if an `[EVAL_PROBES]` line has an invalid
    /// column count or an unknown slot tag.
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut suite = Self::default();
        let mut section = Section::None;

        for (idx, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            match line {
                "[FACT_ATTRACTORS]" => section = Section::Facts,
                "[PERSONA_ATTRACTORS]" => section = Section::Persona,
                "[CONTRASTIVE_GUARDS]" => section = Section::Guards,
                "[EVAL_PROBES]" => section = Section::Probes,
                _ => match section {
                    Section::None => {
                        return Err(format!("line {}: content outside any section", idx + 1));
                    }
                    Section::Facts => suite.facts.push(line.to_string()),
                    Section::Persona => suite.persona.push(line.to_string()),
                    Section::Guards => suite.guards.push(line.to_string()),
                    Section::Probes => suite.probes.push(parse_probe_line(line, idx + 1)?),
                },
            }
        }

        Ok(suite)
    }
}

fn parse_probe_line(line: &str, line_num: usize) -> Result<EvalProbe, String> {
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    if parts.len() != 4 {
        return Err(format!(
            "line {}: expected 4 pipe-separated fields, found {}",
            line_num,
            parts.len()
        ));
    }

    let slot = match parts[0] {
        "FACT" => ProbeSlot::Fact,
        "PERSONA" => ProbeSlot::Persona,
        "GUARD" => ProbeSlot::Guard,
        other => {
            return Err(format!("line {}: unknown probe slot '{}'", line_num, other));
        }
    };

    Ok(EvalProbe {
        slot,
        prompt: parts[1].to_string(),
        expected_keywords: split_keywords(parts[2]),
        forbidden_keywords: split_keywords(parts[3]),
    })
}

fn split_keywords(field: &str) -> Vec<String> {
    field
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(ToString::to_string)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gcso_eval_samples_fixture() {
        let raw = include_str!("../../../tests/fixtures/gcso_eval_samples.txt");
        let suite = EvalSampleSuite::parse(raw).expect("fixture should parse cleanly");

        assert!(suite.facts.len() >= 6);
        assert!(suite.persona.len() >= 3);
        assert!(suite.guards.len() >= 3);
        assert!(suite.probes.len() >= 5);

        assert_eq!(suite.probes[0].slot, ProbeSlot::Fact);
        assert!(suite.probes[0]
            .expected_keywords
            .iter()
            .any(|k| k.contains("Geometric Cellular Sheaf Orchestrator")));
    }
}
