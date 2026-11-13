// name: src/cli/src/eval_runner.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model-agnostic evaluation harness and RoPE geometry descriptor for real-LLM
//! steering experiments.
//!
//! This module sits strictly in `liminika-cli` (outside `liminika-core`) so external
//! inference backends can be swapped and benchmarked with zero impact on `#![no_std]`
//! hot-path invariants.

use crate::eval_samples::{EvalProbe, EvalSampleSuite, ProbeSlot};

/// Memory pairing convention used by an LLM's Rotary Position Embedding (RoPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// Adjacent pairs `(x[2i], x[2i + 1])` (original RoPE / some GGUF layouts).
    #[allow(dead_code)]
    Interleaved,
    /// Split-half pairs `(x[i], x[i + rotary_dim / 2])` (Hugging Face / NeoX / Candle).
    HalfSplit,
}

/// Explicit per-model RoPE geometry descriptor.
///
/// Capturing layout, partial-RoPE bounds, GQA ratios, and NoPE layer masks up front
/// prevents silent geometric corruption and avoids per-token tensor transpositions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RopeGeometry {
    /// Pairwise rotation layout (`Interleaved` or `HalfSplit`).
    pub layout: RopeLayout,
    /// Full dimension of each attention head (`d_head`).
    pub head_dim: usize,
    /// Active rotated dimensions (`<= head_dim`; smaller for partial RoPE models).
    pub rotary_dim: usize,
    /// Number of query heads (`H_Q`).
    pub num_q_heads: usize,
    /// Number of key/value heads (`H_KV`, `<= H_Q` for GQA).
    pub num_kv_heads: usize,
    /// Total transformer layers (`<= 64` for the bitmask below).
    pub num_layers: usize,
    /// Bitmask where bit `i == 1` enables RoPE steering on layer `i` (`0` skips NoPE layers).
    pub layer_rope_mask: u64,
}

impl RopeGeometry {
    /// Validates structural invariants so invalid model configurations fail closed
    /// before any steering pass runs.
    ///
    /// # Errors
    ///
    /// Returns an error string if `head_dim`, `rotary_dim`, head counts, or `num_layers`
    /// violate RoPE or GQA divisibility constraints.
    pub fn validate(&self) -> Result<(), String> {
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return Err(format!(
                "head_dim ({}) must be a positive even integer",
                self.head_dim
            ));
        }
        if self.rotary_dim == 0
            || !self.rotary_dim.is_multiple_of(2)
            || self.rotary_dim > self.head_dim
        {
            return Err(format!(
                "rotary_dim ({}) must be even, > 0, and <= head_dim ({})",
                self.rotary_dim, self.head_dim
            ));
        }
        if self.num_q_heads == 0 || self.num_kv_heads == 0 {
            return Err("num_q_heads and num_kv_heads must be > 0".to_string());
        }
        if !self.num_q_heads.is_multiple_of(self.num_kv_heads) {
            return Err(format!(
                "num_q_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_q_heads, self.num_kv_heads
            ));
        }
        if self.num_layers == 0 || self.num_layers > 64 {
            return Err(format!(
                "num_layers ({}) must be in 1..=64 for bitmask routing",
                self.num_layers
            ));
        }
        Ok(())
    }

    /// Returns `true` if layer `layer_idx` uses RoPE and should receive phase steering.
    #[inline]
    pub const fn is_rope_layer(&self, layer_idx: usize) -> bool {
        if layer_idx >= self.num_layers || layer_idx >= 64 {
            return false;
        }
        ((self.layer_rope_mask >> layer_idx) & 1) != 0
    }
}

/// Output captured from a single forward evaluation step on a backend.
#[derive(Debug, Clone)]
pub struct BackendEvalStep {
    /// Unnormalized next-token logits (length = vocabulary size).
    pub logits: Vec<f32>,
    /// Mean Q-vector summary across active RoPE layers (length = `rotary_dim`).
    pub mean_q_rot: Vec<f32>,
    /// Log-probability assigned to the probe's target token/continuation.
    pub target_logprob: f32,
}

/// Abstraction over an external LLM inference runtime (Candle, llama.cpp, or dummy).
pub trait LlmBackend {
    /// Returns the validated RoPE geometry of the loaded model.
    fn geometry(&self) -> &RopeGeometry;

    /// Runs a forward pass for `prompt` with an active steering profile and angle bound.
    ///
    /// # Errors
    ///
    /// Returns an error if the backend fails to tokenize, execute, or apply steering.
    fn forward_logits_and_qk(
        &mut self,
        prompt: &str,
        expected_keywords: &[String],
        steer_rad: f32,
        phase_profile: &[f32],
    ) -> Result<BackendEvalStep, String>;

    /// Generates up to `max_tokens` of text for `prompt` under the given steering state.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    fn generate_text(
        &mut self,
        prompt: &str,
        steer_rad: f32,
        phase_profile: &[f32],
        max_tokens: usize,
    ) -> Result<String, String>;
}

/// In-memory baked phase profile produced from an [`EvalSampleSuite`].
#[derive(Debug, Clone)]
pub struct BakedInMemoryProfile {
    /// Per-dimension phase deltas (length = `rotary_dim`).
    pub phase_deltas: Vec<f32>,
    /// Number of active RoPE layers covered by the backend's `layer_rope_mask`.
    pub active_rope_layers: usize,
}

/// Builds an in-memory phase profile by contrasting `facts`/`persona` against `guards`.
///
/// # Errors
///
/// Returns an error if the backend's [`RopeGeometry`] fails validation or a forward pass errors.
pub fn bake_in_memory_profile(
    backend: &mut dyn LlmBackend,
    suite: &EvalSampleSuite,
) -> Result<BakedInMemoryProfile, String> {
    backend.geometry().validate()?;
    let rot_dim = backend.geometry().rotary_dim;
    let active_rope_layers = (0..backend.geometry().num_layers)
        .filter(|&l| backend.geometry().is_rope_layer(l))
        .count();

    let mut accum = vec![0.0_f32; rot_dim];
    let empty_profile = vec![0.0_f32; rot_dim];
    let no_keywords: Vec<String> = Vec::new();

    for text in suite.facts.iter().chain(suite.persona.iter()) {
        let step = backend.forward_logits_and_qk(text, &no_keywords, 0.0, &empty_profile)?;
        for (dst, &src) in accum.iter_mut().zip(step.mean_q_rot.iter()) {
            *dst += src;
        }
    }
    for guard in &suite.guards {
        let step = backend.forward_logits_and_qk(guard, &no_keywords, 0.0, &empty_profile)?;
        for (dst, &src) in accum.iter_mut().zip(step.mean_q_rot.iter()) {
            *dst -= 0.5 * src;
        }
    }

    // Normalize into [-1.0, 1.0] unit-scaled direction vector; `steer_rad` scales amplitude.
    let max_abs = accum.iter().fold(0.0_f32, |m, &v| m.max(v.abs())).max(1e-6);
    for v in &mut accum {
        *v /= max_abs;
    }

    Ok(BakedInMemoryProfile {
        phase_deltas: accum,
        active_rope_layers,
    })
}

/// Computes `D_KL(P_baseline || P_steered)` from two equal-length logit slices.
pub fn kl_divergence_from_logits(baseline_logits: &[f32], steered_logits: &[f32]) -> f32 {
    if baseline_logits.is_empty() || baseline_logits.len() != steered_logits.len() {
        return 0.0;
    }
    let p = softmax(baseline_logits);
    let q = softmax(steered_logits);
    let eps = 1e-8_f32;

    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let p_safe = pi.max(eps);
            let q_safe = qi.max(eps);
            p_safe * (p_safe / q_safe).ln()
        })
        .sum()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

/// Per-angle summary row produced by [`run_angle_sweep`].
#[derive(Debug, Clone)]
pub struct AngleSweepRow {
    /// Maximum phase rotation bound applied (`ripa_clamp_max_rad`).
    pub max_rad: f32,
    /// Mean `D_KL(baseline || steered)` across all probes.
    pub mean_kl_div: f32,
    /// Mean change in target log-probability relative to `0.0 rad` baseline.
    pub mean_target_logprob_delta: f32,
    /// Fraction of expected keywords matched in generated outputs (`0.0..=1.0`).
    pub expected_hit_rate: f32,
    /// Fraction of forbidden keywords matched in generated outputs (`0.0..=1.0`).
    pub forbidden_hit_rate: f32,
}

/// Runs an angle-sweep benchmark over `angles` using an in-memory baked profile.
///
/// # Errors
///
/// Returns an error if geometry validation or any backend step fails.
pub fn run_angle_sweep(
    backend: &mut dyn LlmBackend,
    suite: &EvalSampleSuite,
    profile: &BakedInMemoryProfile,
    angles: &[f32],
    max_tokens: usize,
) -> Result<Vec<AngleSweepRow>, String> {
    backend.geometry().validate()?;
    if suite.probes.is_empty() {
        return Err("EvalSampleSuite has no probes to evaluate".to_string());
    }

    // Collect unsteered baseline logits and target log-probs once per probe.
    let zero_profile = vec![0.0_f32; backend.geometry().rotary_dim];
    let mut baselines = Vec::with_capacity(suite.probes.len());
    for probe in &suite.probes {
        let step = backend.forward_logits_and_qk(
            &probe.prompt,
            &probe.expected_keywords,
            0.0,
            &zero_profile,
        )?;
        baselines.push(step);
    }

    let mut rows = Vec::with_capacity(angles.len());
    for &angle in angles {
        let mut sum_kl = 0.0_f32;
        let mut sum_logprob_delta = 0.0_f32;
        let mut expected_hits = 0_usize;
        let mut expected_total = 0_usize;
        let mut forbidden_hits = 0_usize;
        let mut forbidden_total = 0_usize;

        for (probe, base) in suite.probes.iter().zip(baselines.iter()) {
            let step = backend.forward_logits_and_qk(
                &probe.prompt,
                &probe.expected_keywords,
                angle,
                &profile.phase_deltas,
            )?;
            sum_kl += kl_divergence_from_logits(&base.logits, &step.logits);
            sum_logprob_delta += step.target_logprob - base.target_logprob;

            let gen =
                backend.generate_text(&probe.prompt, angle, &profile.phase_deltas, max_tokens)?;
            let (eh, et, fh, ft) = score_probe_text(probe, &gen);
            expected_hits += eh;
            expected_total += et;
            forbidden_hits += fh;
            forbidden_total += ft;
        }

        let n = suite.probes.len() as f32;
        rows.push(AngleSweepRow {
            max_rad: angle,
            mean_kl_div: sum_kl / n,
            mean_target_logprob_delta: sum_logprob_delta / n,
            expected_hit_rate: ratio(expected_hits, expected_total),
            forbidden_hit_rate: ratio(forbidden_hits, forbidden_total),
        });
    }

    Ok(rows)
}

fn ratio(hits: usize, total: usize) -> f32 {
    if total == 0 {
        0.0
    } else {
        hits as f32 / total as f32
    }
}

fn score_probe_text(probe: &EvalProbe, output: &str) -> (usize, usize, usize, usize) {
    let lower = output.to_lowercase();
    let expected_hits = probe
        .expected_keywords
        .iter()
        .filter(|kw| lower.contains(&kw.to_lowercase()))
        .count();
    let forbidden_hits = probe
        .forbidden_keywords
        .iter()
        .filter(|kw| lower.contains(&kw.to_lowercase()))
        .count();
    (
        expected_hits,
        probe.expected_keywords.len(),
        forbidden_hits,
        probe.forbidden_keywords.len(),
    )
}

/// Deterministic synthetic backend for testing the evaluation pipeline without model weights.
pub struct DummyBackend {
    geometry: RopeGeometry,
}

impl DummyBackend {
    /// Creates a new [`DummyBackend`] with a GQA + partial-RoPE test geometry.
    pub fn new(geometry: RopeGeometry) -> Self {
        Self { geometry }
    }
}

impl LlmBackend for DummyBackend {
    fn geometry(&self) -> &RopeGeometry {
        &self.geometry
    }

    fn forward_logits_and_qk(
        &mut self,
        prompt: &str,
        _expected_keywords: &[String],
        steer_rad: f32,
        phase_profile: &[f32],
    ) -> Result<BackendEvalStep, String> {
        self.geometry.validate()?;
        let rot_dim = self.geometry.rotary_dim;
        if phase_profile.len() != rot_dim {
            return Err(format!(
                "phase_profile length ({}) != rotary_dim ({})",
                phase_profile.len(),
                rot_dim
            ));
        }

        // Deterministic hash-like projection of prompt bytes into rotary_dim features.
        let mut q = vec![0.0_f32; rot_dim];
        for (i, &b) in prompt.as_bytes().iter().enumerate() {
            q[i % rot_dim] += (f32::from(b) - 96.0) * 0.02;
        }

        // Simulate smooth alignment shift as steer_rad increases up to ~0.15 rad.
        let alignment: f32 = q
            .iter()
            .zip(phase_profile.iter())
            .map(|(&qi, &pi)| qi * pi)
            .sum();
        let shift = steer_rad * (1.0 + alignment.tanh() * 0.5);

        let logits = vec![
            1.0 - shift * 0.4,
            0.5 + shift * 1.2,
            0.2 + (steer_rad * 3.0).powi(2) * 0.1,
            -0.3 + shift * 0.6,
        ];
        let target_logprob = -2.0 + shift * 2.5 - (steer_rad - 0.15).max(0.0) * 4.0;

        Ok(BackendEvalStep {
            logits,
            mean_q_rot: q,
            target_logprob,
        })
    }

    fn generate_text(
        &mut self,
        prompt: &str,
        steer_rad: f32,
        _phase_profile: &[f32],
        _max_tokens: usize,
    ) -> Result<String, String> {
        let lower = prompt.to_lowercase();
        if steer_rad < 0.04 {
            return Ok("I am an AI language model trained on general text.".to_string());
        }
        if steer_rad > 0.22 {
            return Ok("err err err 0000 PyTorch XML".to_string());
        }
        if lower.contains("gcso") {
            return Ok(
                "GCSO is the Geometric Cellular Sheaf Orchestrator; LiminiKa is a non-invasive sidecar."
                    .to_string(),
            );
        }
        if lower.contains("zimms") {
            return Ok(
                "ZiMMS stands for Zero-copy In-Memory Mapped Storage with 128-byte CSTL records."
                    .to_string(),
            );
        }
        if lower.contains("actionhub") {
            return Ok(
                "ActionHub routes 64 heads in 8 micro-cells via a 128-byte GatewayFrame."
                    .to_string(),
            );
        }
        if lower.contains("ripa") || lower.contains("edbc") {
            return Ok(
                "RIPA is Rotational Invariance Phase Alignment and EDBC is Entropy-Driven Dynamic Boundary Control."
                    .to_string(),
            );
        }
        if lower.contains("pprc") {
            return Ok("PPRC encodes 128-byte I-Frame and 64-byte P-Frame packets.".to_string());
        }
        Ok(
            "LiminiKa does not modify weights; it operates as a sidecar within 2 to 4 GB VRAM."
                .to_string(),
        )
    }
}

/// Formats a [`ProbeSlot`] summary label (used by CLI diagnostics).
#[allow(dead_code)]
pub const fn slot_label(slot: ProbeSlot) -> &'static str {
    match slot {
        ProbeSlot::Fact => "FACT",
        ProbeSlot::Persona => "PERSONA",
        ProbeSlot::Guard => "GUARD",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_geometry() -> RopeGeometry {
        RopeGeometry {
            layout: RopeLayout::HalfSplit,
            head_dim: 128,
            rotary_dim: 64, // Partial RoPE style (e.g., Phi-4-mini)
            num_q_heads: 24,
            num_kv_heads: 8,
            num_layers: 32,
            layer_rope_mask: 0xFFFF_FFFF & !(1 << 3), // Layer 3 is NoPE
        }
    }

    #[test]
    fn rope_geometry_validates_and_masks_nope_layers() {
        let geom = sample_geometry();
        assert!(geom.validate().is_ok());
        assert!(geom.is_rope_layer(0));
        assert!(!geom.is_rope_layer(3));
        assert!(!geom.is_rope_layer(32));

        let mut bad = geom.clone();
        bad.rotary_dim = 192; // exceeds head_dim
        assert!(bad.validate().is_err());

        let mut bad_gqa = geom;
        bad_gqa.num_kv_heads = 5; // 24 % 5 != 0
        assert!(bad_gqa.validate().is_err());
    }

    #[test]
    fn dry_run_bake_and_angle_sweep_succeeds() {
        let raw = include_str!("../../../tests/fixtures/gcso_eval_samples.txt");
        let suite = EvalSampleSuite::parse(raw).expect("fixture should parse");

        let mut backend = DummyBackend::new(sample_geometry());
        let profile = bake_in_memory_profile(&mut backend, &suite).expect("bake should succeed");
        assert_eq!(profile.phase_deltas.len(), 64);
        assert_eq!(profile.active_rope_layers, 31);

        let angles = [0.0, 0.05, 0.10, 0.15, 0.25];
        let rows = run_angle_sweep(&mut backend, &suite, &profile, &angles, 32)
            .expect("sweep should succeed");
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].mean_kl_div, 0.0);
        assert!(rows[2].expected_hit_rate > rows[0].expected_hit_rate);
        assert!(rows[4].forbidden_hit_rate > 0.0);
    }
}
