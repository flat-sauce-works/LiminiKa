// name: src/core/src/runtime/telemetry.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core-First Lazy-Pull `3 + System` telemetry bank, adaptive verbosity modes,
//! Bounded Slow-Loop self-tuning, and `#![no_std]` Mermaid topology self-introspection.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

use crate::drivers::{
    DistortionTension, LiminalEgoState, EGO_TRIGGER_BASIN_CAPTURE, EGO_TRIGGER_BOUNDARY_CHATTER,
    EGO_TRIGGER_BOUNDARY_STAGNATION, EGO_TRIGGER_INTERNAL_CONFLICT, EGO_TRIGGER_LIMINAL_EXTERIOR,
};
use crate::foundation::{branchless_clamp_f32, branchless_is_finite_f32, Pod};
use crate::liminal::{
    EpistemicRegime, LacunaPartition8, LiminalThresholdState, NUM_LIMINAL_THRESHOLD_CHANNELS,
    NUM_STIGMERGIC_GRANULES, THRESHOLD_COGNITIVE_CIRCUIT, THRESHOLD_CONTRADICTION_GUARD,
    THRESHOLD_KV_PHASE_CACHE, THRESHOLD_MANIFOLD_GATEWAY, THRESHOLD_OBJECTIVE_FACT,
    THRESHOLD_RELATIVE_COUPLING, THRESHOLD_STIGMERGIC_CONSENSUS, THRESHOLD_STIGMERGY_WIDE,
    THRESHOLD_SUBJECTIVE_PERSONA, THRESHOLD_SYSTEM_CONSTRAINT,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::context::CoreStateStack;
use crate::runtime::gateway::signals::ManifoldProbeBaseline;
use crate::runtime::horizon::EpistemicHorizonReading;
use crate::stigmergy::{
    OverlapDiscoveryReport, StigmergicActionDecision, StigmergicGranuleState, StigmergicThresholds,
};

/// Classification of cross-layer geometric/control distortion relative to Layer 2 Ego alignment.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerDistortionKind {
    /// Core layers and `StigmergicSubstrate` remain balanced and aligned (`0`).
    #[default]
    Aligned = 0,
    /// Passive saturation or drift without sufficient Ego corrective intervention (`1`).
    PassiveSaturation = 1,
    /// Excessive or unaligned Ego overdrive causing saturation or conflict mismatch (`2`).
    ActiveOverdrive = 2,
}

/// Lazy-computed diagnostic summary of cross-layer distortion (`Distortion`) used to trigger
/// corrective Ego awakening on passive stalls or Ego cooldown on active overdrive.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerDistortionReport {
    /// Largest decision share among the 7 `StigmergicSubstrate` regimes (`0.0..=1.0`).
    pub action_saturation_ratio: f32,
    /// Passive distortion score combining unguided stalls and single-mode saturation (`0.0..=1.0`).
    pub passive_distortion_ratio: f32,
    /// Active overdrive distortion score combining unaligned Ego steps and saturated Ego bias (`0.0..=1.0`).
    pub active_distortion_ratio: f32,
    /// Overall cross-layer mismatch rate (`0.0..=1.0`).
    pub mismatch_ratio: f32,
    /// Dominant distortion state.
    pub kind: LayerDistortionKind,
    /// `true` when passive distortion warrants waking/boosting Layer 2 Ego.
    pub ego_wakeup_recommended: bool,
    /// `true` when active distortion warrants cooling down Layer 2 Ego thrust/defiance.
    pub ego_cooldown_recommended: bool,
}

/// Static compile-time descriptor binding a threshold channel node to its static cognitive
/// threshold and unknowable exterior complement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiminalHorizonNodeDescriptor {
    /// Mermaid graph node identifier (`"N0"`..`"N7"`).
    pub node_id: &'static str,
    /// Canonical capability or domain label associated with this threshold channel.
    pub trait_name: &'static str,
    /// Channel index (`0..8`) inside [`LiminalThresholdState`].
    pub channel_idx: usize,
    /// Static cognitive threshold mask owned by this channel node.
    pub static_aperture_mask: u32,
    /// Type-level unobserved exterior complement (`!static_aperture_mask`).
    pub unobserved_lacuna_mask: u32,
}

/// Canonical 8-channel capability/domain topology table derived from `liminal::threshold` channels.
pub const LMNK_LIMINAL_HORIZON_TOPOLOGY: [LiminalHorizonNodeDescriptor;
    NUM_LIMINAL_THRESHOLD_CHANNELS] = [
    LiminalHorizonNodeDescriptor {
        node_id: "N0",
        trait_name: "SystemConstraintSignal",
        channel_idx: 0,
        static_aperture_mask: THRESHOLD_SYSTEM_CONSTRAINT,
        unobserved_lacuna_mask: !THRESHOLD_SYSTEM_CONSTRAINT,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N1",
        trait_name: "ObjectiveFactSignal",
        channel_idx: 1,
        static_aperture_mask: THRESHOLD_OBJECTIVE_FACT,
        unobserved_lacuna_mask: !THRESHOLD_OBJECTIVE_FACT,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N2",
        trait_name: "SubjectivePersonaSignal",
        channel_idx: 2,
        static_aperture_mask: THRESHOLD_SUBJECTIVE_PERSONA,
        unobserved_lacuna_mask: !THRESHOLD_SUBJECTIVE_PERSONA,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N3",
        trait_name: "ContradictionGuardSignal",
        channel_idx: 3,
        static_aperture_mask: THRESHOLD_CONTRADICTION_GUARD,
        unobserved_lacuna_mask: !THRESHOLD_CONTRADICTION_GUARD,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N4",
        trait_name: "CognitiveTrajectoryLedger",
        channel_idx: 4,
        static_aperture_mask: THRESHOLD_COGNITIVE_CIRCUIT,
        unobserved_lacuna_mask: !THRESHOLD_COGNITIVE_CIRCUIT,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N5",
        trait_name: "RamPhaseKvCacheOps",
        channel_idx: 5,
        static_aperture_mask: THRESHOLD_KV_PHASE_CACHE,
        unobserved_lacuna_mask: !THRESHOLD_KV_PHASE_CACHE,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N6",
        trait_name: "RelativeCouplingMatrix",
        channel_idx: 6,
        static_aperture_mask: THRESHOLD_RELATIVE_COUPLING,
        unobserved_lacuna_mask: !THRESHOLD_RELATIVE_COUPLING,
    },
    LiminalHorizonNodeDescriptor {
        node_id: "N7",
        trait_name: "StigmergicSubstrateOps",
        channel_idx: 7,
        static_aperture_mask: THRESHOLD_STIGMERGY_WIDE,
        unobserved_lacuna_mask: !THRESHOLD_STIGMERGY_WIDE,
    },
];

/// Compact `32`-byte (`#[repr(C, align(16))]`) self-introspecting telemetry snapshot capturing
/// live [`LiminalThresholdState`] boundaries, active epistemic regime, and anchor population counts.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiminalTopologyTelemetrySnapshot {
    /// Per-channel `Liminal` floor values in Q0.7 (`0..=127`).
    pub threshold_floor_q7: [u8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Per-channel cognitive aperture gain values in Q0.7 (`0..=127`).
    pub aperture_gain_q7: [u8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Per-channel signed focal phase shifts in Q7 (`-128..=127`).
    pub phase_shift_q7: [i8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Live active cognitive aperture bitmask.
    pub active_aperture_mask: u32,
    /// Encoded [`EpistemicRegime`] byte (`0`, `1`, `2`).
    pub regime_code: u8,
    /// Quantized conflict severity in Q0.7 (`0..=127`).
    pub conflict_severity_q7: u8,
    /// Number of active macro attractors currently populated.
    pub active_crystals: u8,
    /// Number of autonomously spawned daughter attractors recorded.
    pub spawned_daughters: u8,
}

struct BufCursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> BufCursor<'a> {
    #[inline]
    fn write_bytes(&mut self, bytes: &[u8]) {
        let rem = self.buf.len().saturating_sub(self.pos);
        let n = bytes.len().min(rem);
        if n > 0 {
            self.buf[self.pos..self.pos + n].copy_from_slice(&bytes[..n]);
        }
        self.pos = self.pos.saturating_add(bytes.len());
    }

    #[inline]
    fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
    }

    #[inline]
    fn write_u8_dec(&mut self, val: u8) {
        let mut tmp = [0u8; 3];
        let mut v = val;
        let mut len = 0usize;
        if v >= 100 {
            tmp[len] = b'0' + (v / 100);
            len += 1;
            v %= 100;
            tmp[len] = b'0' + (v / 10);
            len += 1;
            tmp[len] = b'0' + (v % 10);
            len += 1;
        } else if v >= 10 {
            tmp[len] = b'0' + (v / 10);
            len += 1;
            tmp[len] = b'0' + (v % 10);
            len += 1;
        } else {
            tmp[len] = b'0' + v;
            len += 1;
        }
        self.write_bytes(&tmp[..len]);
    }

    #[inline]
    fn write_i8_dec(&mut self, val: i8) {
        if val < 0 {
            self.write_bytes(b"-");
            self.write_u8_dec(val.unsigned_abs());
        } else {
            self.write_bytes(b"+");
            self.write_u8_dec(val as u8);
        }
    }

    #[inline]
    fn write_hex_u16(&mut self, val: u32) {
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        let out = [
            b'0',
            b'x',
            HEX[((val >> 12) & 0xF) as usize],
            HEX[((val >> 8) & 0xF) as usize],
            HEX[((val >> 4) & 0xF) as usize],
            HEX[(val & 0xF) as usize],
        ];
        self.write_bytes(&out);
    }

    #[inline]
    fn write_hex_u8(&mut self, val: u8) {
        const HEX: &[u8; 16] = b"0123456789ABCDEF";
        let out = [
            b'0',
            b'x',
            HEX[((val >> 4) & 0xF) as usize],
            HEX[(val & 0xF) as usize],
        ];
        self.write_bytes(&out);
    }
}

impl LiminalTopologyTelemetrySnapshot {
    /// Captures a compact self-introspecting telemetry snapshot from `field` and `horizon`.
    #[inline]
    #[must_use]
    pub fn capture(
        field: &LiminalThresholdState,
        horizon: &EpistemicHorizonReading,
        active_crystals: u8,
        spawned_daughters: u8,
    ) -> Self {
        let sev = (branchless_clamp_f32(horizon.conflict_severity, 0.0, 1.0) * 127.0 + 0.5) as u8;
        Self {
            threshold_floor_q7: field.threshold_floor_q7,
            aperture_gain_q7: field.aperture_gain_q7,
            phase_shift_q7: field.phase_shift_q7,
            active_aperture_mask: field.active_aperture_mask,
            regime_code: horizon.regime_code,
            conflict_severity_q7: sev,
            active_crystals,
            spawned_daughters,
        }
    }

    #[inline]
    fn format_mermaid_flowchart_core(
        &self,
        layer1_opt: Option<(&StigmergicGranuleState, &OverlapDiscoveryReport)>,
        layer2_opt: Option<&LiminalEgoState>,
        buf: &mut [u8],
    ) -> usize {
        let mut cur = BufCursor { buf, pos: 0 };
        let regime_str = match EpistemicRegime::from_code(self.regime_code) {
            EpistemicRegime::KnownCoherent => "KnownCoherent",
            EpistemicRegime::InternalConflict => "InternalConflict",
            EpistemicRegime::Liminal => "Liminal",
        };
        cur.write_str("flowchart TD\n");
        cur.write_str("    LIM[\"Liminal (Boundary Horizon & Exterior)\"]\n");

        match layer1_opt {
            Some((stigmergy, report)) => {
                cur.write_str("    STIG[\"StigmergicGranuleState | OverlapCore:");
                cur.write_hex_u8(report.overlap_core_mask);
                cur.write_str(" | Discovery:");
                cur.write_hex_u8(report.discovery_excursion_mask);
                cur.write_str(" | Quorum:");
                cur.write_u8_dec(stigmergy.quorum_threshold);
                cur.write_str("/8 | Epochs:");
                cur.write_u8_dec((stigmergy.adaptation_epochs & 0xFF) as u8);
                cur.write_str("\"]\n");
            }
            None => {
                cur.write_str("    STIG[\"Stigmergic Approximation | Apt:");
                cur.write_hex_u16(THRESHOLD_STIGMERGIC_CONSENSUS);
                cur.write_str("\"]\n");
            }
        }

        if let Some(ego) = layer2_opt {
            cur.write_str("    EGO[\"LiminalEgoState | Thrust:");
            cur.write_u8_dec(ego.counter_basin_thrust_q7);
            cur.write_str("/127 | Transgress:");
            cur.write_i8_dec(ego.transgression_bias_q7);
            cur.write_str(" | Fired:");
            cur.write_hex_u8(ego.last_intervened_mask);
            cur.write_str(" | Epochs:");
            cur.write_u8_dec((ego.intervention_epochs & 0xFF) as u8);
            cur.write_str("\"]\n");
        }

        let mut i = 0usize;
        while i < 7 {
            let desc = &LMNK_LIMINAL_HORIZON_TOPOLOGY[i];
            let ch = desc.channel_idx;
            let active = (self.active_aperture_mask & (1u32 << ch)) != 0;
            cur.write_str("    ");
            cur.write_str(desc.node_id);
            cur.write_str("[\"");
            cur.write_str(desc.trait_name);
            cur.write_str(" | Apt:");
            cur.write_hex_u16(desc.static_aperture_mask);
            cur.write_str(" | LacunaFloor:");
            cur.write_u8_dec(self.threshold_floor_q7[ch]);
            cur.write_str("/127 | Gain:");
            cur.write_u8_dec(self.aperture_gain_q7[ch]);
            cur.write_str("/127 | Shift:");
            cur.write_i8_dec(self.phase_shift_q7[ch]);
            if active {
                cur.write_str(" | ACTIVE\"]\n");
            } else {
                cur.write_str(" | COLLAPSED\"]\n");
            }
            i += 1;
        }

        cur.write_str("    GW[\"ManifoldGatewaySignal | Apt:");
        cur.write_hex_u16(THRESHOLD_MANIFOLD_GATEWAY);
        cur.write_str(" | Regime:");
        cur.write_str(regime_str);
        cur.write_str(" | Conflict:");
        cur.write_u8_dec(self.conflict_severity_q7);
        cur.write_str("/127\"]\n");

        cur.write_str("    HUB[\"L2_Substrate | Apt:");
        cur.write_hex_u16(THRESHOLD_STIGMERGY_WIDE);
        cur.write_str(" | LiminalFloor:");
        cur.write_u8_dec(self.threshold_floor_q7[7]);
        cur.write_str("/127 | Anchors:");
        cur.write_u8_dec(self.active_crystals);
        cur.write_str(" | Daughters:");
        cur.write_u8_dec(self.spawned_daughters);
        cur.write_str("\"]\n");

        cur.write_str("    LIM -. \"Liminal Cut\" .-> N0 & N1 & N2 & N3 & N4 & N5 & N6\n");
        cur.write_str("    LIM <-->|\"Cross-Layer Feedback\"| STIG\n");
        if layer2_opt.is_some() {
            cur.write_str("    EGO -. \"Top-Down Intervention\" .-> LIM & STIG\n");
            cur.write_str("    EGO -->|\"Counter-Basin / Defiance\"| GW\n");
        }
        cur.write_str("    N0 & N1 & N2 & N3 & N4 & N5 & N6 --> GW\n");
        cur.write_str("    GW --> HUB\n");
        cur.write_str("    STIG -->|\"Frag / Drift / Reconstitute\"| HUB\n");
        cur.pos
    }

    /// Renders all three autonomous core layers (`Liminal` Layer 0, `StigmergicGranuleState`
    /// Layer 1, and `LiminalEgoState` Layer 2) alongside overlap & discovery telemetry as a
    /// `#![no_std]` zero-allocation Mermaid `flowchart TD` diagram directly into `buf`.
    #[inline]
    pub fn format_core_topology_mermaid(
        &self,
        stack: &CoreStateStack,
        overlap: &OverlapDiscoveryReport,
        buf: &mut [u8],
    ) -> usize {
        self.format_core_topology_mermaid_with_distortion(stack, overlap, None, buf)
    }

    /// Renders all three autonomous core layers plus optional cross-layer [`LayerDistortionReport`]
    /// diagnostics (`Aligned` / `PassiveSaturation` / `ActiveOverdrive`) into `buf`.
    #[inline]
    pub fn format_core_topology_mermaid_with_distortion(
        &self,
        stack: &CoreStateStack,
        overlap: &OverlapDiscoveryReport,
        distortion_opt: Option<&LayerDistortionReport>,
        buf: &mut [u8],
    ) -> usize {
        let base_len = self.format_mermaid_flowchart_core(
            Some((&stack.stigmergy, overlap)),
            Some(&stack.ego),
            buf,
        );
        let Some(dist) = distortion_opt else {
            return base_len;
        };
        let mut cur = BufCursor { buf, pos: base_len };
        let kind_str = match dist.kind {
            LayerDistortionKind::Aligned => "Aligned",
            LayerDistortionKind::PassiveSaturation => "PassiveSaturation",
            LayerDistortionKind::ActiveOverdrive => "ActiveOverdrive",
        };
        let sat_pct = (dist.action_saturation_ratio * 100.0).clamp(0.0, 100.0) as u8;
        let pass_pct = (dist.passive_distortion_ratio * 100.0).clamp(0.0, 100.0) as u8;
        let act_pct = (dist.active_distortion_ratio * 100.0).clamp(0.0, 100.0) as u8;
        cur.write_str("    DIST[\"LayerDistortion | Kind:");
        cur.write_str(kind_str);
        cur.write_str(" | Sat:");
        cur.write_u8_dec(sat_pct);
        cur.write_str("% | Pass:");
        cur.write_u8_dec(pass_pct);
        cur.write_str("% | Act:");
        cur.write_u8_dec(act_pct);
        cur.write_str("%");
        if dist.ego_wakeup_recommended {
            cur.write_str(" | WAKEUP");
        } else if dist.ego_cooldown_recommended {
            cur.write_str(" | COOLDOWN");
        }
        cur.write_str("\"]\n");
        cur.write_str("    HUB -. \"Saturation / Mismatch\" .-> DIST\n");
        cur.write_str("    DIST <-->|\"Wakeup / Cooldown\"| EGO\n");
        cur.pos
    }
}

const EMA_Q15_ONE: u16 = 32_768;
const EMA_Q15_TO_F32: f32 = 1.0_f32 / 32_768.0_f32;

#[inline(always)]
fn f32_unit_to_q15(val: f32) -> u16 {
    if !branchless_is_finite_f32(val) {
        return 0;
    }
    let clamped = branchless_clamp_f32(val, 0.0_f32, 1.0_f32);
    (clamped * 32_768.0_f32 + 0.5_f32) as u16
}

#[inline(always)]
const fn step_ema_q15(prev: u16, sample: u16, initialized: bool) -> u16 {
    if !initialized {
        sample
    } else {
        (((prev as u32) * 31 + (sample as u32) + 16) >> 5) as u16
    }
}

/// Granularity tier controlling how much telemetry is captured on each step.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TelemetryVerbosityMode {
    /// Minimal-overhead tier for game-concurrent operation: only increments `SystemSpineTelemetry::total_steps`
    /// and reads live compute state lazily on Pull.
    Coarse = 0,
    /// Default tier capturing continuous `3 + System` summary counters (`Standard` + `Detailed` share
    /// `continuous_standard_steps` so Slow-Loop denominators remain consistent across promotions).
    #[default]
    Standard = 1,
    /// Full research/diagnostic tier sliding in [`DetailedLayerTelemetry`] per-channel, per-granule,
    /// and per-trigger counters.
    Detailed = 2,
}

/// Capture policy deciding whether verbosity stays manually pinned or slides dynamically between tiers.
#[repr(C, u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelemetryCapturePolicy {
    /// Fixes telemetry to a single mode for reproducible benchmarking and ablations.
    Pinned(TelemetryVerbosityMode),
    /// Adapts telemetry granularity dynamically while preserving dwell anti-chattering.
    Auto {
        /// Nominal baseline tier (`Coarse` or `Standard`).
        base: TelemetryVerbosityMode,
        /// Transitions temporarily to [`TelemetryVerbosityMode::Detailed`] during destabilization.
        spike_to_detailed: bool,
        /// Drops to [`TelemetryVerbosityMode::Coarse`] when `System` QoS throttling is active.
        qos_throttle_to_coarse: bool,
    },
}

impl Default for TelemetryCapturePolicy {
    #[inline(always)]
    fn default() -> Self {
        Self::Auto {
            base: TelemetryVerbosityMode::Standard,
            spike_to_detailed: true,
            qos_throttle_to_coarse: true,
        }
    }
}

/// Layer 0 (`Liminal`) telemetry domain: cognitive boundaries, chatter, and slide-in aperture activity.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiminalTelemetry {
    /// Step counts per [`EpistemicRegime`] (`[0]=KnownCoherent`, `[1]=InternalConflict`, `[2]=Liminal`).
    pub regime_counts: [u64; 3],
    /// Steps observing boundary chatter (`chattering_channel_mask != 0`).
    pub chatter_events: u64,
    /// Steps eligible for quiescent Layer 0 stationary shortcut (`stationary_shortcut_ready != 0`).
    pub stationary_shortcut_hits: u64,
    /// Dynamic slide-in transitions (`slid_in_bits != 0`).
    pub slide_in_count: u64,
    /// Dynamic slide-out retractions (`slid_out_bits != 0`).
    pub slide_out_count: u64,
    /// Steps stuck in conflict or liminal drift without Ego intervention (`last_intervened_mask == 0`).
    pub passive_stall_events: u64,
}

/// Layer 1 (`Stigmergic`) telemetry domain: 7-regime `StigmergicSubstrate` decisions, overlap, and quorums.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StigmergicTelemetry {
    /// Per-regime [`StigmergicActionDecision`] invocation counts (`0..=6`).
    pub decision_counts: [u64; 7],
    /// Steps where excursion discovery detected exterior overlap.
    pub overlap_discovery_hits: u64,
    /// Steps where micro-swarm granules maintained non-empty overlap core consensus.
    pub quorum_core_hits: u64,
}

/// Layer 2 (`Ego`) telemetry domain: top-down interventions, tie-breaks, and cold-path escalations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EgoTelemetry {
    /// Steps where Layer 2 Ego actively intervened (`last_intervened_mask != 0`).
    pub intervention_count: u64,
    /// Steps where Ego resolved boundary conflict or basin capture.
    pub tiebreak_count: u64,
    /// Steps where cold-path escalation swapped external phases into a working lane.
    pub escalation_swaps: u64,
    /// Steps where Ego intervened yet ended in repulsion or unaligned conflict without overlap quorum.
    pub unaligned_intervention_count: u64,
}

/// Vertical `+ System` execution spine telemetry: cross-layer `EntropyController`, QoS, mode switches, and Slow-Loop state.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SystemSpineTelemetry {
    /// Monotonic total steps observed (incremented across all modes, including `Coarse`).
    pub total_steps: u64,
    /// Continuous denominator of steps recorded in `Standard` or `Detailed` (safe for Slow-Loop).
    pub continuous_standard_steps: u64,
    /// Steps experiencing `System` QoS throttling.
    pub qos_throttle_events: u64,
    /// Steps in `EntropyController` bifurcation branch mode (`active_branch_mode == 1`).
    pub entropy_gate_bifurcation_events: u64,
    /// Steps in `EntropyController` repulsion branch mode (`active_branch_mode >= 2`).
    pub entropy_gate_repulsion_events: u64,
    /// Automatic or manual verbosity expansion transitions.
    pub mode_expansions: u64,
    /// Automatic or manual verbosity retraction transitions.
    pub mode_retractions: u64,
    /// Number of Slow-Loop epoch evaluations performed.
    pub slow_loop_epochs: u64,
    /// Number of auxiliary residual spillover cache hits (`RamPhaseKvCache` recall).
    pub experience_cache_hits: u64,
    /// Number of verified auxiliary residual experiences promoted into RAM cache.
    pub experience_cache_stores: u64,
    /// Number of stale or contradicted cache entries evicted below pheromone floor.
    pub experience_cache_evictions: u64,
    /// Latest bounded Slow-Loop repulse bias adjustment in Q0.7 (`-16..=16`).
    pub slow_loop_repulse_bias_q7: i8,
    /// Latest bounded Slow-Loop `min_slide_out_dwell` delta (`-1..=2`).
    pub slow_loop_dwell_delta: i8,
    /// Explicit C alignment padding.
    pub reserved: [u8; 6],
}

/// Optional/slide-in fine-grained telemetry captured exclusively while active mode is [`TelemetryVerbosityMode::Detailed`].
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DetailedLayerTelemetry {
    /// Per-channel Layer 0 boundary chatter counts (`0..8`).
    pub liminal_channel_chatter_counts: [u64; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Per-granule stigmergic granule overlap activation counts (`0..8`).
    pub stigmergic_granule_overlap_hits: [u64; NUM_STIGMERGIC_GRANULES],
    /// Per-flag Layer 2 Ego trigger breakdown:
    /// `[0]=BasinCapture, [1]=InternalConflict, [2]=LiminalExterior, [3]=BoundaryChatter, [4]=BoundaryStagnation`.
    pub ego_trigger_counts: [u64; 5],
    /// Steps captured while [`TelemetryVerbosityMode::Detailed`] was slid in.
    pub detailed_captured_steps: u64,
}

/// Result returned by Slow-Loop epoch evaluation over continuous `Standard`/`Detailed` counters.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SlowLoopTelemetryAdjustment {
    /// `true` when enough continuous samples existed and active mode was not `Coarse`.
    pub applied: bool,
    /// Continuous `Standard`/`Detailed` steps evaluated for this adjustment.
    pub sampled_steps: u64,
    /// Bounded `StigmergicThresholds` calibrated from telemetry rates (clamped within `+/-15%` of base).
    pub tuned_thresholds: StigmergicThresholds,
    /// Bounded recommendation for `min_slide_out_dwell` offset (`-1..=2`).
    pub suggested_slide_out_dwell_delta: i8,
    /// Bounded recommendation for repulsion bias in Q0.7 (`-16..=16`).
    pub suggested_repulse_bias_q7: i8,
    /// Updated Q0.7 derivative weight $\alpha \in [16, 96]$ for convex integral/derivative pheromone synthesis.
    pub suggested_derivative_alpha_q7: u8,
    /// Cross-layer distortion diagnosis evaluated over the epoch samples.
    pub distortion: LayerDistortionReport,
}

/// Lazy-Pull unified telemetry snapshot combining `3 + System` domains, derived ratios, and
/// live [`LiminalTopologyTelemetrySnapshot`] state.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CoreTelemetrySnapshot {
    /// Active telemetry verbosity at snapshot capture time.
    pub mode: TelemetryVerbosityMode,
    /// Configured verbosity policy.
    pub policy: TelemetryCapturePolicy,
    /// Layer 0 (`Liminal`) telemetry counters.
    pub liminal: LiminalTelemetry,
    /// Layer 1 (`Stigmergic`) telemetry counters.
    pub stigmergy: StigmergicTelemetry,
    /// Layer 2 (`Ego`) telemetry counters.
    pub ego: EgoTelemetry,
    /// Vertical `+ System` spine counters.
    pub spine: SystemSpineTelemetry,
    /// Detailed breakdown if `Detailed` mode captured data or is currently active.
    pub detailed: Option<DetailedLayerTelemetry>,
    /// Lazy-computed `ActionDecision` percentage share (`0.0..=100.0`) over `continuous_standard_steps`.
    pub decision_ratios_pct: [f32; 7],
    /// Lazy-computed share of steps spent in `InternalConflict` or `Liminal` regime (`0.0..=1.0`).
    pub liminal_excursion_ratio: f32,
    /// Lazy-computed share of steps eligible for Layer 0 stationary shortcut (`0.0..=1.0`).
    pub stationary_shortcut_ratio: f32,
    /// Lazy-computed share of steps with active Layer 2 Ego intervention (`0.0..=1.0`).
    pub ego_intervention_ratio: f32,
    /// Compact 32-byte topological snapshot of live Layer 0 boundaries.
    pub topology: LiminalTopologyTelemetrySnapshot,
    /// Cross-layer geometric/control distortion metrics (`Distortion`).
    pub distortion: LayerDistortionReport,
    /// Bitmask of external `LatentHorizonSlot`s currently slid in.
    pub active_horizon_slots_mask: u32,
    /// Rolling `0 fdiv` EMA ($\alpha = 1/32$) of normalized entropy (`0.0..=1.0`).
    pub entropy_ema: f32,
    /// Rolling `0 fdiv` EMA ($\alpha = 1/32$) of trajectory/phase coherence (`0.0..=1.0`).
    pub coherence_ema: f32,
    /// Rolling `0 fdiv` EMA ($\alpha = 1/32$) of guard noise/similarity (`0.0..=1.0`).
    pub guard_ema: f32,
    /// Gauge-relative assimilation drift $D_{\mathrm{assim}} \in [0.0, 1.0]$ from [`ManifoldProbeBaseline`].
    pub assimilation_drift: f32,
    /// Consecutive steps where $D_{\mathrm{assim}}$ exceeded the drift threshold (with hysteresis).
    pub assimilation_drift_dwell: u16,
    /// `2 Primitives x 2 Drivers` structural attribution counts indexed by `AttributionTag::raw()` (`0..=3`).
    pub attribution_counts: [u64; 4],
    /// Cumulative `SystemSpine` token-level whole-steering bypass hits (`0` D2H syncs).
    pub whole_token_bypass_events: u64,
    /// Cumulative [`DistortionTension`] immediate Hot-Path Ego damping events.
    pub immediate_tension_damps: u64,
    /// Cumulative [`DistortionTension`] Slow-Loop stagnation excitation events.
    pub slow_tension_excites: u64,
    /// Cumulative P-Frame chain evaluations with `>= 1` hop traversed.
    pub pframe_chain_evals: u64,
    /// Cumulative P-Frame hops traversed across [`crate::drivers::EgoDifferentialNet`].
    pub pframe_total_hops: u64,
    /// Cumulative steps where validated $O \setminus S$ staged an isolated non-copy sandbox lane.
    pub exterior_sandbox_stages: u64,
    /// Cumulative steps where Unobserved ($\overline{S \cup O}$) locked aperture to `0`.
    pub unobserved_aperture_zeros: u64,
}

/// Standalone `#![no_std]` zero-alloc `3 + System` telemetry bank stored beside compute engines.
///
/// Hot `Coarse`-mode control and counters (`policy`, `active_mode`, and `spine.total_steps` /
/// `spine.qos_throttle_events`) reside in Cache Line 0 (`0..64`), while `detailed` is isolated at
/// offset `384` (Cache Line 6) to avoid polluting Hot/Warm cache lines.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoreTelemetryBank {
    policy: TelemetryCapturePolicy,
    active_mode: TelemetryVerbosityMode,
    derivative_alpha_q7: u8,
    ema_initialized: bool,
    reserved_cl0: u8,
    /// Vertical `+ System` spine counters (hot `Coarse` counters packed within `0..64`).
    pub spine: SystemSpineTelemetry,
    attribution_counts: [u64; 4],
    whole_token_bypass_events: u64,
    immediate_tension_damps: u64,
    slow_tension_excites: u64,
    pframe_chain_evals: u64,
    pframe_total_hops: u64,
    exterior_sandbox_stages: u64,
    unobserved_aperture_zeros: u64,
    /// Layer 0 (`Liminal`) counters.
    pub liminal: LiminalTelemetry,
    /// Layer 1 (`Stigmergic`) counters.
    pub stigmergy: StigmergicTelemetry,
    /// Layer 2 (`Ego`) counters.
    pub ego: EgoTelemetry,
    /// Consecutive quiescent steps required before retracting from `Detailed` back to `base`.
    pub retract_dwell_threshold: u16,
    entropy_ema_q15: u16,
    coherence_ema_q15: u16,
    guard_ema_q15: u16,
    assimilation_drift_q15: u16,
    assimilation_drift_dwell: u16,
    sandbox_coherence_improve_dwell: u16,
    reserved_warm: [u16; 5],
    /// Slide-in detailed counters (untouched unless `active_mode == Detailed`; CL6-isolated).
    pub detailed: DetailedLayerTelemetry,
}

const _: () = {
    assert!(core::mem::size_of::<TelemetryCapturePolicy>() == 4);
    assert!(core::mem::align_of::<CoreTelemetryBank>() == 64);
    assert!(core::mem::size_of::<CoreTelemetryBank>() == 576);
    assert!(core::mem::offset_of!(CoreTelemetryBank, spine) == 8);
    assert!(core::mem::offset_of!(CoreTelemetryBank, detailed) == 384);
};

impl Default for CoreTelemetryBank {
    #[inline]
    fn default() -> Self {
        Self::new(TelemetryCapturePolicy::default())
    }
}

impl CoreTelemetryBank {
    /// Creates a new [`CoreTelemetryBank`] governed by `policy`.
    #[inline]
    #[must_use]
    pub const fn new(policy: TelemetryCapturePolicy) -> Self {
        let initial_mode = match policy {
            TelemetryCapturePolicy::Pinned(m) => m,
            TelemetryCapturePolicy::Auto { base, .. } => base,
        };
        Self {
            policy,
            active_mode: initial_mode,
            derivative_alpha_q7: crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7,
            ema_initialized: false,
            reserved_cl0: 0,
            spine: SystemSpineTelemetry {
                total_steps: 0,
                continuous_standard_steps: 0,
                qos_throttle_events: 0,
                entropy_gate_bifurcation_events: 0,
                entropy_gate_repulsion_events: 0,
                mode_expansions: 0,
                mode_retractions: 0,
                slow_loop_epochs: 0,
                experience_cache_hits: 0,
                experience_cache_stores: 0,
                experience_cache_evictions: 0,
                slow_loop_repulse_bias_q7: 0,
                slow_loop_dwell_delta: 0,
                reserved: [0; 6],
            },
            attribution_counts: [0; 4],
            whole_token_bypass_events: 0,
            immediate_tension_damps: 0,
            slow_tension_excites: 0,
            pframe_chain_evals: 0,
            pframe_total_hops: 0,
            exterior_sandbox_stages: 0,
            unobserved_aperture_zeros: 0,
            liminal: LiminalTelemetry {
                regime_counts: [0; 3],
                chatter_events: 0,
                stationary_shortcut_hits: 0,
                slide_in_count: 0,
                slide_out_count: 0,
                passive_stall_events: 0,
            },
            stigmergy: StigmergicTelemetry {
                decision_counts: [0; 7],
                overlap_discovery_hits: 0,
                quorum_core_hits: 0,
            },
            ego: EgoTelemetry {
                intervention_count: 0,
                tiebreak_count: 0,
                escalation_swaps: 0,
                unaligned_intervention_count: 0,
            },
            retract_dwell_threshold: 2,
            entropy_ema_q15: 0,
            coherence_ema_q15: 0,
            guard_ema_q15: 0,
            assimilation_drift_q15: 0,
            assimilation_drift_dwell: 0,
            sandbox_coherence_improve_dwell: 0,
            reserved_warm: [0; 5],
            detailed: DetailedLayerTelemetry {
                liminal_channel_chatter_counts: [0; NUM_LIMINAL_THRESHOLD_CHANNELS],
                stigmergic_granule_overlap_hits: [0; NUM_STIGMERGIC_GRANULES],
                ego_trigger_counts: [0; 5],
                detailed_captured_steps: 0,
            },
        }
    }

    /// Initializes non-zero fields of [`CoreTelemetryBank`] directly inside an already-zeroed target
    /// buffer without materializing a 576-byte stack temporary.
    ///
    /// # Safety
    /// Caller guarantees `ptr` points to a valid, properly aligned, all-zero `CoreTelemetryBank`.
    #[inline]
    pub(crate) unsafe fn init_in_place_on_zeroed(ptr: *mut Self, policy: TelemetryCapturePolicy) {
        let initial_mode = match policy {
            TelemetryCapturePolicy::Pinned(m) => m,
            TelemetryCapturePolicy::Auto { base, .. } => base,
        };
        // SAFETY: `ptr` is valid, aligned, and zero-initialized.
        unsafe {
            core::ptr::addr_of_mut!((*ptr).policy).write(policy);
            core::ptr::addr_of_mut!((*ptr).active_mode).write(initial_mode);
            core::ptr::addr_of_mut!((*ptr).derivative_alpha_q7)
                .write(crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7);
            core::ptr::addr_of_mut!((*ptr).retract_dwell_threshold).write(2);
        }
    }

    /// Returns the Slow-Loop Q0.7 derivative weight $\alpha \in [0, 127]$ (read-only on the Hot Path).
    #[inline(always)]
    #[must_use]
    pub const fn derivative_alpha_q7(&self) -> u8 {
        self.derivative_alpha_q7
    }

    /// Sets the Slow-Loop Q0.7 derivative weight $\alpha \in [0, 127]$ (e.g., `0` for pure integral ablation).
    #[inline(always)]
    pub fn set_derivative_alpha_q7(&mut self, alpha_q7: u8) {
        self.derivative_alpha_q7 = alpha_q7.min(127);
    }

    /// Creates a bank pinned directly to a fixed [`TelemetryVerbosityMode`].
    #[inline]
    #[must_use]
    pub const fn with_mode(mode: TelemetryVerbosityMode) -> Self {
        Self::new(TelemetryCapturePolicy::Pinned(mode))
    }

    /// Returns the currently active [`TelemetryVerbosityMode`].
    #[inline(always)]
    #[must_use]
    pub const fn mode(&self) -> TelemetryVerbosityMode {
        self.active_mode
    }

    /// Returns the configured [`TelemetryCapturePolicy`].
    #[inline(always)]
    #[must_use]
    pub const fn policy(&self) -> TelemetryCapturePolicy {
        self.policy
    }

    /// Updates the [`TelemetryCapturePolicy`] and aligns `active_mode` immediately if pinned or base changes.
    #[inline]
    pub fn set_policy(&mut self, policy: TelemetryCapturePolicy) {
        self.policy = policy;
        let next = match policy {
            TelemetryCapturePolicy::Pinned(m) => m,
            TelemetryCapturePolicy::Auto { base, .. } => base,
        };
        if (next as u8) > (self.active_mode as u8) {
            self.spine.mode_expansions = self.spine.mode_expansions.saturating_add(1);
        } else if (next as u8) < (self.active_mode as u8) {
            self.spine.mode_retractions = self.spine.mode_retractions.saturating_add(1);
        }
        self.active_mode = next;
    }

    /// Clears all accumulated counters across `3 + System` domains while retaining policy and dwell config.
    #[inline]
    pub fn reset(&mut self) {
        self.liminal = LiminalTelemetry::default();
        self.stigmergy = StigmergicTelemetry::default();
        self.ego = EgoTelemetry::default();
        self.spine = SystemSpineTelemetry::default();
        self.detailed = DetailedLayerTelemetry::default();
        self.ema_initialized = false;
        self.entropy_ema_q15 = 0;
        self.coherence_ema_q15 = 0;
        self.guard_ema_q15 = 0;
        self.assimilation_drift_q15 = 0;
        self.assimilation_drift_dwell = 0;
        self.sandbox_coherence_improve_dwell = 0;
        self.derivative_alpha_q7 = crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7;
        self.attribution_counts = [0; 4];
        self.whole_token_bypass_events = 0;
        self.immediate_tension_damps = 0;
        self.slow_tension_excites = 0;
        self.pframe_chain_evals = 0;
        self.pframe_total_hops = 0;
        self.exterior_sandbox_stages = 0;
        self.unobserved_aperture_zeros = 0;
    }

    /// Records a `SystemSpine` whole-token stationary bypass event without polluting
    /// stigmergic action saturation counters.
    #[inline(always)]
    pub fn record_whole_token_bypass(&mut self) {
        self.whole_token_bypass_events = self.whole_token_bypass_events.saturating_add(1);
    }

    /// Records lightweight `2 Primitives x 2 Drivers` structural attribution, P-Frame chain hops,
    /// `DistortionTension` events, exterior $O \setminus S$ sandbox staging, and pre-inference bypass hits.
    #[inline]
    pub fn record_two_by_two_step(
        &mut self,
        tag_raw: u8,
        whole_token_bypass: bool,
        lacuna_partition: &LacunaPartition8,
        tension: &DistortionTension,
        pframe_hops: u8,
    ) {
        let idx = usize::from(tag_raw & 0x03);
        self.attribution_counts[idx] = self.attribution_counts[idx].saturating_add(1);
        if whole_token_bypass {
            self.whole_token_bypass_events = self.whole_token_bypass_events.saturating_add(1);
        }
        if lacuna_partition.is_sandbox_eligible() {
            self.exterior_sandbox_stages = self.exterior_sandbox_stages.saturating_add(1);
        }
        if lacuna_partition.is_unobserved_aperture_zero() {
            self.unobserved_aperture_zeros = self.unobserved_aperture_zeros.saturating_add(1);
        }
        if pframe_hops > 0 {
            self.pframe_chain_evals = self.pframe_chain_evals.saturating_add(1);
            self.pframe_total_hops = self
                .pframe_total_hops
                .saturating_add(u64::from(pframe_hops));
        }
        self.immediate_tension_damps = self
            .immediate_tension_damps
            .max(u64::from(tension.immediate_damping_events));
        self.slow_tension_excites = self
            .slow_tension_excites
            .max(u64::from(tension.slow_excitation_events));
    }

    /// Updates `0 fdiv` rolling Q0.15 EMAs ($\alpha = 1/32$) and gauge-relative assimilation drift
    /// against `baseline` in `Standard` or `Detailed` mode, returning the updated drift dwell count.
    ///
    /// In [`TelemetryVerbosityMode::Coarse`], EMA updates and drift dwell accumulation pause
    /// immediately to keep Hot-Path overhead zero.
    #[inline]
    pub fn update_assimilation_ema(
        &mut self,
        entropy: f32,
        coherence: f32,
        guard: f32,
        baseline: &ManifoldProbeBaseline,
        drift_threshold: f32,
    ) -> u16 {
        if matches!(self.active_mode, TelemetryVerbosityMode::Coarse) {
            return self.assimilation_drift_dwell;
        }
        let ent_q15 = f32_unit_to_q15(entropy);
        let coh_q15 = f32_unit_to_q15(coherence);
        let grd_q15 = f32_unit_to_q15(guard);
        let init = self.ema_initialized;
        let prev_coh = self.coherence_ema_q15;

        self.entropy_ema_q15 = step_ema_q15(self.entropy_ema_q15, ent_q15, init);
        self.coherence_ema_q15 = step_ema_q15(self.coherence_ema_q15, coh_q15, init);
        self.guard_ema_q15 = step_ema_q15(self.guard_ema_q15, grd_q15, init);
        self.ema_initialized = true;

        let h0 = f32_unit_to_q15(baseline.baseline_entropy);
        let c0 = f32_unit_to_q15(baseline.baseline_inter_layer_coherence);
        let g0 = f32_unit_to_q15(baseline.baseline_guard_noise_floor);

        let coh_drop = c0.saturating_sub(self.coherence_ema_q15);
        let ent_excess = self.entropy_ema_q15.saturating_sub(h0);
        let guard_excess = self.guard_ema_q15.saturating_sub(g0);
        let drift_raw =
            u32::from(coh_drop) + (u32::from(ent_excess) >> 1) + (u32::from(guard_excess) >> 1);
        self.assimilation_drift_q15 = drift_raw.min(u32::from(EMA_Q15_ONE)) as u16;

        let tau_q15 = f32_unit_to_q15(drift_threshold).max(1);
        let hysteresis_floor = tau_q15.saturating_sub(tau_q15 >> 3);
        let enough_samples = self.spine.continuous_standard_steps >= 16;
        if enough_samples && self.assimilation_drift_q15 >= tau_q15 {
            self.assimilation_drift_dwell = self.assimilation_drift_dwell.saturating_add(1);
        } else if self.assimilation_drift_q15 < hysteresis_floor {
            self.assimilation_drift_dwell = 0;
        }

        if init && (coh_q15 >= c0 || coh_q15 > prev_coh) {
            self.sandbox_coherence_improve_dwell =
                self.sandbox_coherence_improve_dwell.saturating_add(1);
        } else {
            self.sandbox_coherence_improve_dwell = 0;
        }

        self.assimilation_drift_dwell
    }

    /// Returns the latest gauge-relative assimilation drift $D_{\mathrm{assim}} \in [0.0, 1.0]$.
    #[inline(always)]
    #[must_use]
    pub fn assimilation_drift(&self) -> f32 {
        f32::from(self.assimilation_drift_q15) * EMA_Q15_TO_F32
    }

    /// Returns the consecutive assimilation drift dwell steps.
    #[inline(always)]
    #[must_use]
    pub const fn assimilation_drift_dwell(&self) -> u16 {
        self.assimilation_drift_dwell
    }

    /// Returns the consecutive coherence improvement steps observed for sandbox promotion.
    #[inline(always)]
    #[must_use]
    pub const fn sandbox_coherence_improve_dwell(&self) -> u16 {
        self.sandbox_coherence_improve_dwell
    }

    /// Resets the assimilation drift dwell counter (e.g., after sandbox promotion or release).
    #[inline(always)]
    pub const fn reset_assimilation_dwell(&mut self) {
        self.assimilation_drift_dwell = 0;
    }

    /// Resets the sandbox coherence improvement dwell counter upon fresh sandbox nucleation.
    #[inline(always)]
    pub const fn reset_sandbox_improve_dwell(&mut self) {
        self.sandbox_coherence_improve_dwell = 0;
    }

    /// Synchronizes Tier-1 residual experience cache cumulative metrics (`hits`, `stores`,
    /// `evictions`) onto `+ System` telemetry in zero-alloc mode.
    #[inline(always)]
    pub fn sync_experience_metrics(&mut self, hits: u64, stores: u64, evictions: u64) {
        self.spine.experience_cache_hits = hits;
        self.spine.experience_cache_stores = stores;
        self.spine.experience_cache_evictions = evictions;
    }

    /// Evaluates automatic verbosity policy with dwell anti-chattering and returns the updated mode.
    #[inline]
    pub fn evaluate_policy(
        &mut self,
        destabilized: bool,
        quiescent_dwell: u16,
        qos_throttled: bool,
    ) -> TelemetryVerbosityMode {
        let next = match self.policy {
            TelemetryCapturePolicy::Pinned(m) => m,
            TelemetryCapturePolicy::Auto {
                base,
                spike_to_detailed,
                qos_throttle_to_coarse,
            } => {
                if qos_throttle_to_coarse && qos_throttled {
                    TelemetryVerbosityMode::Coarse
                } else if spike_to_detailed && destabilized {
                    TelemetryVerbosityMode::Detailed
                } else if quiescent_dwell >= self.retract_dwell_threshold {
                    base
                } else {
                    self.active_mode
                }
            }
        };
        if (next as u8) > (self.active_mode as u8) {
            self.spine.mode_expansions = self.spine.mode_expansions.saturating_add(1);
        } else if (next as u8) < (self.active_mode as u8) {
            self.spine.mode_retractions = self.spine.mode_retractions.saturating_add(1);
        }
        self.active_mode = next;
        next
    }

    /// Records a single step across `3 + System` telemetry domains according to `self.active_mode`.
    ///
    /// In [`TelemetryVerbosityMode::Coarse`], returns immediately after incrementing `total_steps`
    /// (and `qos_throttle_events` when throttled) so Hot-Path L1D stays clean.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn record_step(
        &mut self,
        decision: StigmergicActionDecision,
        horizon: &EpistemicHorizonReading,
        delta: &LiminalBoundaryDelta,
        overlap: &OverlapDiscoveryReport,
        ego: &LiminalEgoState,
        entropy_gate_mode: u32,
        slid_in_bits: u32,
        slid_out_bits: u32,
        escalated: bool,
        qos_throttled: bool,
    ) {
        self.spine.total_steps = self.spine.total_steps.saturating_add(1);
        if qos_throttled {
            self.spine.qos_throttle_events = self.spine.qos_throttle_events.saturating_add(1);
        }
        if matches!(self.active_mode, TelemetryVerbosityMode::Coarse) {
            return;
        }

        self.spine.continuous_standard_steps =
            self.spine.continuous_standard_steps.saturating_add(1);
        if entropy_gate_mode == 1 {
            self.spine.entropy_gate_bifurcation_events =
                self.spine.entropy_gate_bifurcation_events.saturating_add(1);
        } else if entropy_gate_mode >= 2 {
            self.spine.entropy_gate_repulsion_events =
                self.spine.entropy_gate_repulsion_events.saturating_add(1);
        }

        // Layer 0: Liminal
        let reg_idx = usize::from(horizon.regime_code.min(2));
        self.liminal.regime_counts[reg_idx] = self.liminal.regime_counts[reg_idx].saturating_add(1);
        if delta.chattering_channel_mask != 0 {
            self.liminal.chatter_events = self.liminal.chatter_events.saturating_add(1);
        }
        if delta.stationary_shortcut_ready != 0 {
            self.liminal.stationary_shortcut_hits =
                self.liminal.stationary_shortcut_hits.saturating_add(1);
        }
        if slid_in_bits != 0 {
            self.liminal.slide_in_count = self.liminal.slide_in_count.saturating_add(1);
        }
        if slid_out_bits != 0 {
            self.liminal.slide_out_count = self.liminal.slide_out_count.saturating_add(1);
        }

        // Layer 1: Stigmergic
        let dec_idx = (decision as usize).min(6);
        self.stigmergy.decision_counts[dec_idx] =
            self.stigmergy.decision_counts[dec_idx].saturating_add(1);
        if overlap.discovery_triggered != 0 {
            self.stigmergy.overlap_discovery_hits =
                self.stigmergy.overlap_discovery_hits.saturating_add(1);
        }
        if overlap.overlap_core_mask != 0 {
            self.stigmergy.quorum_core_hits = self.stigmergy.quorum_core_hits.saturating_add(1);
        }

        // Layer 2: Ego & cross-layer distortion accounting
        if ego.last_intervened_mask != 0 {
            self.ego.intervention_count = self.ego.intervention_count.saturating_add(1);
            if (ego.last_intervened_mask
                & (EGO_TRIGGER_INTERNAL_CONFLICT | EGO_TRIGGER_BASIN_CAPTURE))
                != 0
            {
                self.ego.tiebreak_count = self.ego.tiebreak_count.saturating_add(1);
            }
            let unaligned = matches!(decision, StigmergicActionDecision::RepulsionEscape)
                || (horizon.is_internal_conflict() && overlap.overlap_core_mask == 0);
            if unaligned {
                self.ego.unaligned_intervention_count =
                    self.ego.unaligned_intervention_count.saturating_add(1);
            }
        } else {
            let stalled = (horizon.is_internal_conflict() || horizon.is_liminal())
                && delta.stationary_shortcut_ready == 0
                && matches!(
                    decision,
                    StigmergicActionDecision::FullOrchestration
                        | StigmergicActionDecision::CrystallizeMacroBasin
                        | StigmergicActionDecision::RepulsionEscape
                );
            if stalled {
                self.liminal.passive_stall_events =
                    self.liminal.passive_stall_events.saturating_add(1);
            }
        }
        if escalated {
            self.ego.escalation_swaps = self.ego.escalation_swaps.saturating_add(1);
        }

        if matches!(self.active_mode, TelemetryVerbosityMode::Detailed) {
            self.record_detailed_breakdown(delta, overlap, ego);
        }
    }

    #[inline]
    fn record_detailed_breakdown(
        &mut self,
        delta: &LiminalBoundaryDelta,
        overlap: &OverlapDiscoveryReport,
        ego: &LiminalEgoState,
    ) {
        self.detailed.detailed_captured_steps =
            self.detailed.detailed_captured_steps.saturating_add(1);

        let chatter = delta.chattering_channel_mask;
        for (ch, cnt) in self
            .detailed
            .liminal_channel_chatter_counts
            .iter_mut()
            .enumerate()
            .take(NUM_LIMINAL_THRESHOLD_CHANNELS)
        {
            if ((chatter >> ch) & 1) != 0 {
                *cnt = cnt.saturating_add(1);
            }
        }

        let core_mask = overlap.overlap_core_mask;
        for (g, cnt) in self
            .detailed
            .stigmergic_granule_overlap_hits
            .iter_mut()
            .enumerate()
            .take(NUM_STIGMERGIC_GRANULES)
        {
            if ((core_mask >> g) & 1) != 0 {
                *cnt = cnt.saturating_add(1);
            }
        }

        const EGO_FLAGS: [u8; 5] = [
            EGO_TRIGGER_BASIN_CAPTURE,
            EGO_TRIGGER_INTERNAL_CONFLICT,
            EGO_TRIGGER_LIMINAL_EXTERIOR,
            EGO_TRIGGER_BOUNDARY_CHATTER,
            EGO_TRIGGER_BOUNDARY_STAGNATION,
        ];
        for (idx, &flag) in EGO_FLAGS.iter().enumerate() {
            if (ego.last_intervened_mask & flag) != 0 {
                self.detailed.ego_trigger_counts[idx] =
                    self.detailed.ego_trigger_counts[idx].saturating_add(1);
            }
        }
    }

    /// Accumulates counters from a pulled [`CoreTelemetrySnapshot`] into this bank, enabling
    /// multi-pass or multi-probe aggregation using the exact same core distortion/ratio laws.
    #[inline]
    pub fn accumulate_snapshot(&mut self, snap: &CoreTelemetrySnapshot) {
        for (dst, &src) in self
            .liminal
            .regime_counts
            .iter_mut()
            .zip(snap.liminal.regime_counts.iter())
        {
            *dst = dst.saturating_add(src);
        }
        self.liminal.chatter_events = self
            .liminal
            .chatter_events
            .saturating_add(snap.liminal.chatter_events);
        self.liminal.stationary_shortcut_hits = self
            .liminal
            .stationary_shortcut_hits
            .saturating_add(snap.liminal.stationary_shortcut_hits);
        self.liminal.slide_in_count = self
            .liminal
            .slide_in_count
            .saturating_add(snap.liminal.slide_in_count);
        self.liminal.slide_out_count = self
            .liminal
            .slide_out_count
            .saturating_add(snap.liminal.slide_out_count);
        self.liminal.passive_stall_events = self
            .liminal
            .passive_stall_events
            .saturating_add(snap.liminal.passive_stall_events);

        for (dst, &src) in self
            .stigmergy
            .decision_counts
            .iter_mut()
            .zip(snap.stigmergy.decision_counts.iter())
        {
            *dst = dst.saturating_add(src);
        }
        self.stigmergy.overlap_discovery_hits = self
            .stigmergy
            .overlap_discovery_hits
            .saturating_add(snap.stigmergy.overlap_discovery_hits);
        self.stigmergy.quorum_core_hits = self
            .stigmergy
            .quorum_core_hits
            .saturating_add(snap.stigmergy.quorum_core_hits);

        self.ego.intervention_count = self
            .ego
            .intervention_count
            .saturating_add(snap.ego.intervention_count);
        self.ego.tiebreak_count = self
            .ego
            .tiebreak_count
            .saturating_add(snap.ego.tiebreak_count);
        self.ego.escalation_swaps = self
            .ego
            .escalation_swaps
            .saturating_add(snap.ego.escalation_swaps);
        self.ego.unaligned_intervention_count = self
            .ego
            .unaligned_intervention_count
            .saturating_add(snap.ego.unaligned_intervention_count);

        self.spine.total_steps = self
            .spine
            .total_steps
            .saturating_add(snap.spine.total_steps);
        self.spine.continuous_standard_steps = self
            .spine
            .continuous_standard_steps
            .saturating_add(snap.spine.continuous_standard_steps);
        self.spine.qos_throttle_events = self
            .spine
            .qos_throttle_events
            .saturating_add(snap.spine.qos_throttle_events);
        self.spine.entropy_gate_bifurcation_events = self
            .spine
            .entropy_gate_bifurcation_events
            .saturating_add(snap.spine.entropy_gate_bifurcation_events);
        self.spine.entropy_gate_repulsion_events = self
            .spine
            .entropy_gate_repulsion_events
            .saturating_add(snap.spine.entropy_gate_repulsion_events);
        self.spine.mode_expansions = self
            .spine
            .mode_expansions
            .saturating_add(snap.spine.mode_expansions);
        self.spine.mode_retractions = self
            .spine
            .mode_retractions
            .saturating_add(snap.spine.mode_retractions);
        self.spine.experience_cache_hits = snap.spine.experience_cache_hits;
        self.spine.experience_cache_stores = snap.spine.experience_cache_stores;
        self.spine.experience_cache_evictions = snap.spine.experience_cache_evictions;
        for (dst, &src) in self
            .attribution_counts
            .iter_mut()
            .zip(snap.attribution_counts.iter())
        {
            *dst = dst.saturating_add(src);
        }
        self.whole_token_bypass_events = self
            .whole_token_bypass_events
            .saturating_add(snap.whole_token_bypass_events);
        self.immediate_tension_damps = self
            .immediate_tension_damps
            .saturating_add(snap.immediate_tension_damps);
        self.slow_tension_excites = self
            .slow_tension_excites
            .saturating_add(snap.slow_tension_excites);
        self.pframe_chain_evals = self
            .pframe_chain_evals
            .saturating_add(snap.pframe_chain_evals);
        self.pframe_total_hops = self
            .pframe_total_hops
            .saturating_add(snap.pframe_total_hops);
        self.exterior_sandbox_stages = self
            .exterior_sandbox_stages
            .saturating_add(snap.exterior_sandbox_stages);
        self.unobserved_aperture_zeros = self
            .unobserved_aperture_zeros
            .saturating_add(snap.unobserved_aperture_zeros);
    }

    /// Computes a Lazy-Pull cross-layer [`LayerDistortionReport`] diagnosing whether the core
    /// is aligned, passively saturated/stalled, or actively overdriven by unaligned Ego bias.
    #[inline]
    #[must_use]
    pub fn compute_distortion_report(&self) -> LayerDistortionReport {
        if self.spine.continuous_standard_steps == 0 {
            return LayerDistortionReport::default();
        }
        let denom = self.spine.continuous_standard_steps as f32;
        let inv_denom = 1.0_f32 / denom;
        let skip_idx = StigmergicActionDecision::InertialLayerSkip as usize;
        let mut max_decision = 0_u64;
        let mut hhi = 0.0_f32;
        for (idx, &cnt) in self.stigmergy.decision_counts.iter().enumerate() {
            // Intentional stationary shortcuts (`InertialLayerSkip` / `SelfRef` passthrough)
            // are healthy low-entropy skips, not pathological stigmergic over-integration.
            if idx == skip_idx {
                continue;
            }
            if cnt > max_decision {
                max_decision = cnt;
            }
            let p = (cnt as f32) * inv_denom;
            hhi += p * p;
        }
        let action_saturation_ratio =
            branchless_clamp_f32((max_decision as f32) * inv_denom, 0.0, 1.0);
        let concentration = branchless_clamp_f32((hhi - (1.0 / 7.0)) / (6.0 / 7.0), 0.0, 1.0);
        let sat_excess = branchless_clamp_f32((action_saturation_ratio - 0.45) / 0.55, 0.0, 1.0);
        let sat_load = 0.5 * concentration + 0.5 * sat_excess;
        let passive_stall_rate = branchless_clamp_f32(
            (self.liminal.passive_stall_events as f32) * inv_denom,
            0.0,
            1.0,
        );
        let active_mismatch_rate = branchless_clamp_f32(
            (self.ego.unaligned_intervention_count as f32) * inv_denom,
            0.0,
            1.0,
        );
        let ego_ratio =
            branchless_clamp_f32((self.ego.intervention_count as f32) * inv_denom, 0.0, 1.0);

        // Preserve total saturation load without diluting it by `ego_ratio`; attribute
        // whether saturation persists passively (low Ego) or despite/from Ego (active).
        let passive_distortion_ratio = branchless_clamp_f32(
            if passive_stall_rate > sat_load * (1.0 - 0.5 * ego_ratio) {
                passive_stall_rate
            } else {
                sat_load * (1.0 - 0.5 * ego_ratio)
            },
            0.0,
            1.0,
        );
        let active_distortion_ratio = branchless_clamp_f32(
            if active_mismatch_rate > sat_load * ego_ratio {
                active_mismatch_rate
            } else {
                sat_load * ego_ratio
            },
            0.0,
            1.0,
        );
        let mismatch_ratio = branchless_clamp_f32(
            ((self.liminal.passive_stall_events + self.ego.unaligned_intervention_count) as f32)
                * inv_denom,
            0.0,
            1.0,
        );

        let enough_samples = self.spine.continuous_standard_steps >= 16;
        let is_distorted = enough_samples
            && (sat_load >= 0.40 || passive_stall_rate >= 0.35 || active_mismatch_rate >= 0.35);
        let ego_cooldown_recommended = is_distorted
            && ((active_distortion_ratio >= passive_distortion_ratio && ego_ratio >= 0.40)
                || active_mismatch_rate >= 0.35);
        let ego_wakeup_recommended = is_distorted && !ego_cooldown_recommended;
        let kind = if ego_cooldown_recommended {
            LayerDistortionKind::ActiveOverdrive
        } else if ego_wakeup_recommended {
            LayerDistortionKind::PassiveSaturation
        } else {
            LayerDistortionKind::Aligned
        };

        LayerDistortionReport {
            action_saturation_ratio,
            passive_distortion_ratio,
            active_distortion_ratio,
            mismatch_ratio,
            kind,
            ego_wakeup_recommended,
            ego_cooldown_recommended,
        }
    }

    /// Computes a Lazy-Pull [`CoreTelemetrySnapshot`] combining `3 + System` counters, derived ratios,
    /// and live `stack` topological state without performing allocations.
    #[inline]
    #[must_use]
    pub fn snapshot(
        &self,
        stack: &CoreStateStack,
        horizon: &EpistemicHorizonReading,
        active_crystals: u8,
        spawned_daughters: u8,
        active_horizon_slots_mask: u32,
    ) -> CoreTelemetrySnapshot {
        let denom = self.spine.continuous_standard_steps.max(1) as f32;
        let has_samples = self.spine.continuous_standard_steps > 0;
        let mut decision_ratios_pct = [0.0_f32; 7];
        if has_samples {
            for (dst, &cnt) in decision_ratios_pct
                .iter_mut()
                .zip(self.stigmergy.decision_counts.iter())
            {
                *dst = (cnt as f32 / denom) * 100.0;
            }
        }
        let excursion_cnt =
            self.liminal.regime_counts[1].saturating_add(self.liminal.regime_counts[2]);
        let topology = LiminalTopologyTelemetrySnapshot::capture(
            &stack.liminal,
            horizon,
            active_crystals,
            spawned_daughters,
        );
        let distortion = self.compute_distortion_report();
        let detailed = if matches!(self.active_mode, TelemetryVerbosityMode::Detailed)
            || self.detailed.detailed_captured_steps > 0
        {
            Some(self.detailed)
        } else {
            None
        };

        CoreTelemetrySnapshot {
            mode: self.active_mode,
            policy: self.policy,
            liminal: self.liminal,
            stigmergy: self.stigmergy,
            ego: self.ego,
            spine: self.spine,
            detailed,
            decision_ratios_pct,
            liminal_excursion_ratio: if has_samples {
                excursion_cnt as f32 / denom
            } else {
                0.0
            },
            stationary_shortcut_ratio: if has_samples {
                self.liminal.stationary_shortcut_hits as f32 / denom
            } else {
                0.0
            },
            ego_intervention_ratio: if has_samples {
                self.ego.intervention_count as f32 / denom
            } else {
                0.0
            },
            topology,
            distortion,
            active_horizon_slots_mask,
            entropy_ema: f32::from(self.entropy_ema_q15) * EMA_Q15_TO_F32,
            coherence_ema: f32::from(self.coherence_ema_q15) * EMA_Q15_TO_F32,
            guard_ema: f32::from(self.guard_ema_q15) * EMA_Q15_TO_F32,
            assimilation_drift: f32::from(self.assimilation_drift_q15) * EMA_Q15_TO_F32,
            assimilation_drift_dwell: self.assimilation_drift_dwell,
            attribution_counts: self.attribution_counts,
            whole_token_bypass_events: self.whole_token_bypass_events,
            immediate_tension_damps: self.immediate_tension_damps,
            slow_tension_excites: self.slow_tension_excites,
            pframe_chain_evals: self.pframe_chain_evals,
            pframe_total_hops: self.pframe_total_hops,
            exterior_sandbox_stages: self.exterior_sandbox_stages,
            unobserved_aperture_zeros: self.unobserved_aperture_zeros,
        }
    }

    /// Evaluates bounded Slow-Loop autonomous threshold adjustments at an epoch boundary.
    ///
    /// Skips adjustment (`applied = false`) when `active_mode == Coarse` or fewer than
    /// `min_samples.max(1)` continuous `Standard`/`Detailed` steps have accumulated so denominator
    /// skew never destabilizes thresholds.
    #[inline]
    pub fn evaluate_slow_loop_epoch(
        &mut self,
        base: &StigmergicThresholds,
        min_samples: u64,
    ) -> SlowLoopTelemetryAdjustment {
        let distortion = self.compute_distortion_report();
        let req = min_samples.max(1);
        if matches!(self.active_mode, TelemetryVerbosityMode::Coarse)
            || self.spine.continuous_standard_steps < req
        {
            return SlowLoopTelemetryAdjustment {
                applied: false,
                sampled_steps: self.spine.continuous_standard_steps,
                tuned_thresholds: *base,
                suggested_slide_out_dwell_delta: 0,
                suggested_repulse_bias_q7: 0,
                suggested_derivative_alpha_q7: self.derivative_alpha_q7,
                distortion,
            };
        }

        let denom = self.spine.continuous_standard_steps as f32;
        let chatter_ratio = self.liminal.chatter_events as f32 / denom;
        let excursion_ratio = (self.liminal.regime_counts[1]
            .saturating_add(self.liminal.regime_counts[2])) as f32
            / denom;
        let skip_ratio = self.stigmergy.decision_counts[1] as f32 / denom;
        let rank1_ratio = self.stigmergy.decision_counts[4] as f32 / denom;
        let repulse_ratio = self.stigmergy.decision_counts[6] as f32 / denom;

        // Layer 0 chatter -> bounded dwell hysteresis extension (prevents slot flapping).
        let dwell_delta: i8 = if chatter_ratio > 0.50 {
            2
        } else if chatter_ratio > 0.25 {
            1
        } else {
            0
        };

        // Layer 1 decision balance -> homeostatic +/-15% adjustments keeping the monotone ladder intact.
        let density_scale = branchless_clamp_f32(
            1.0 - (repulse_ratio - 0.35) * 0.25 + (excursion_ratio - 0.40) * 0.10,
            0.88,
            1.12,
        );
        let rank1_scale = branchless_clamp_f32(
            1.0 + (rank1_ratio - 0.40) * 0.50 - (skip_ratio - 0.25) * 0.20,
            0.85,
            1.25,
        );
        let skip_entropy_scale = branchless_clamp_f32(
            1.0 + (rank1_ratio - 0.55) * 0.20 - (skip_ratio - 0.10) * 0.35,
            0.85,
            1.08,
        );
        let contradict_scale = branchless_clamp_f32(
            1.0 + (repulse_ratio - 0.45) * 0.25 - (excursion_ratio - 0.30) * 0.15,
            0.88,
            1.12,
        );
        let next_crystallize = branchless_clamp_f32(
            base.crystallize_density_thresh * branchless_clamp_f32(rank1_scale, 0.90, 1.25),
            base.subhead_route_density_thresh,
            4.50,
        );

        let next_express = branchless_clamp_f32(
            base.express_steer_density_thresh * density_scale,
            0.05,
            next_crystallize,
        );
        let next_subhead = branchless_clamp_f32(
            base.subhead_route_density_thresh * density_scale,
            next_express,
            next_crystallize,
        );
        let next_rank1 = branchless_clamp_f32(
            base.rank1_inject_density_thresh * rank1_scale,
            next_subhead,
            next_crystallize,
        );
        let next_skip_entropy = branchless_clamp_f32(
            base.inertial_skip_entropy_thresh * skip_entropy_scale,
            0.05,
            0.55,
        );

        let tuned_thresholds = StigmergicThresholds {
            inertial_skip_entropy_thresh: next_skip_entropy,
            express_steer_density_thresh: next_express,
            subhead_route_density_thresh: next_subhead,
            rank1_inject_density_thresh: next_rank1,
            crystallize_density_thresh: next_crystallize,
            contradiction_coherence_thresh: branchless_clamp_f32(
                base.contradiction_coherence_thresh * contradict_scale,
                0.05,
                0.95,
            ),
            ..*base
        };

        // Layer 2 / System bounded repulsion bias in [-16, 16].
        let raw_bias = ((excursion_ratio - 0.35) * 32.0).clamp(-16.0, 16.0) as i8;

        // Slow-Loop relative derivative/integral balance alpha (Q0.7, read-only on Hot Path):
        // PassiveSaturation (over-integration) -> push toward derivative (+12);
        // ActiveOverdrive (over-differentiation) -> pull back toward integral (-12);
        // Aligned -> gently relax toward nominal default.
        let next_alpha = match distortion.kind {
            LayerDistortionKind::PassiveSaturation => self
                .derivative_alpha_q7
                .saturating_add(12)
                .min(crate::stigmergy::substrate::MAX_DERIVATIVE_ALPHA_Q7),
            LayerDistortionKind::ActiveOverdrive => self
                .derivative_alpha_q7
                .saturating_sub(12)
                .max(crate::stigmergy::substrate::MIN_DERIVATIVE_ALPHA_Q7),
            LayerDistortionKind::Aligned => {
                let nominal = crate::stigmergy::substrate::DEFAULT_DERIVATIVE_ALPHA_Q7;
                if self.derivative_alpha_q7 > nominal {
                    self.derivative_alpha_q7.saturating_sub(2).max(nominal)
                } else {
                    self.derivative_alpha_q7.saturating_add(2).min(nominal)
                }
            }
        };
        self.derivative_alpha_q7 = next_alpha;

        self.spine.slow_loop_epochs = self.spine.slow_loop_epochs.saturating_add(1);
        self.spine.slow_loop_dwell_delta = dwell_delta;
        self.spine.slow_loop_repulse_bias_q7 = raw_bias;

        SlowLoopTelemetryAdjustment {
            applied: true,
            sampled_steps: self.spine.continuous_standard_steps,
            tuned_thresholds,
            suggested_slide_out_dwell_delta: dwell_delta,
            suggested_repulse_bias_q7: raw_bias,
            suggested_derivative_alpha_q7: next_alpha,
            distortion,
        }
    }
}

/// Compile-time check pinning [`EpistemicRegime`] indices to `[0, 1, 2]`.
const _: () = {
    assert!(EpistemicRegime::KnownCoherent as usize == 0);
    assert!(EpistemicRegime::InternalConflict as usize == 1);
    assert!(EpistemicRegime::Liminal as usize == 2);
};

// SAFETY: `LiminalTopologyTelemetrySnapshot` is `#[repr(C, align(16))]` of 32 bytes with zero padding traps.
unsafe impl Pod for LiminalTopologyTelemetrySnapshot {}
