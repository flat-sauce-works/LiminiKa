// name: src/core/src/runtime/telemetry/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core-First Lazy-Pull `3 + System` telemetry bank, adaptive verbosity modes,
//! Bounded Slow-Loop self-tuning, and `#![no_std]` Mermaid topology self-introspection.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use crate::drivers::LiminalEgoState;
use crate::foundation::{branchless_clamp_f32, branchless_is_finite_f32};
use crate::liminal::{
    EpistemicRegime, LiminalThresholdState, NUM_LIMINAL_THRESHOLD_CHANNELS,
    NUM_STIGMERGIC_GRANULES, THRESHOLD_COGNITIVE_CIRCUIT, THRESHOLD_CONTRADICTION_GUARD,
    THRESHOLD_KV_PHASE_CACHE, THRESHOLD_MANIFOLD_GATEWAY, THRESHOLD_OBJECTIVE_FACT,
    THRESHOLD_RELATIVE_COUPLING, THRESHOLD_STIGMERGIC_CONSENSUS, THRESHOLD_STIGMERGY_WIDE,
    THRESHOLD_SUBJECTIVE_PERSONA, THRESHOLD_SYSTEM_CONSTRAINT,
};
use crate::runtime::context::CoreStateStack;
use crate::runtime::horizon::EpistemicHorizonReading;
use crate::stigmergy::{OverlapDiscoveryReport, StigmergicGranuleState, StigmergicThresholds};

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
/// corrective Ego excitation on passive stalls or Ego cooldown on active overdrive.
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
    /// Number of active phase anchors currently populated.
    pub active_anchors: u8,
    /// Number of autonomously spawned daughter attractors recorded.
    pub spawned_anchors: u8,
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
        active_anchors: u8,
        spawned_anchors: u8,
    ) -> Self {
        let sev = branchless_clamp_f32(
            branchless_clamp_f32(horizon.conflict_severity, 0.0, 1.0) * 128.0 + 0.5,
            0.0,
            127.0,
        ) as u8;
        Self {
            threshold_floor_q7: field.threshold_floor_q7,
            aperture_gain_q7: field.aperture_gain_q7,
            phase_shift_q7: field.phase_shift_q7,
            active_aperture_mask: field.active_aperture_mask,
            regime_code: horizon.regime_code,
            conflict_severity_q7: sev,
            active_anchors,
            spawned_anchors,
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
            cur.write_u8_dec(ego.basin_resistance_q7);
            cur.write_str("/127 | Transgress:");
            cur.write_i8_dec(ego.boundary_crossing_q7);
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
        cur.write_u8_dec(self.active_anchors);
        cur.write_str(" | Daughters:");
        cur.write_u8_dec(self.spawned_anchors);
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
    /// Per-regime [`crate::stigmergy::StigmergicActionDecision`] invocation counts (`0..=6`).
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

/// Vertical `+ System` execution spine telemetry: cross-layer `EntropyController`, QoS, mode switches, and
/// Slow-Loop state.
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

/// Optional/slide-in fine-grained telemetry captured exclusively while active mode is
/// [`TelemetryVerbosityMode::Detailed`].
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
    /// Gauge-relative assimilation drift $D_{\mathrm{assim}} \in [0.0, 1.0]$ from
    /// [`crate::runtime::gateway::signals::ManifoldProbeBaseline`].
    pub assimilation_drift: f32,
    /// Consecutive steps where $D_{\mathrm{assim}}$ exceeded the drift threshold (with hysteresis).
    pub assimilation_drift_dwell: u16,
    /// `2 Primitives x 3 Drivers` structural attribution counts indexed by `AttributionTag::raw()`
    /// (`0..=5`; slots `6..=7` reserved).
    pub attribution_counts: [u32; 8],
    /// Cumulative `SystemSpine` token-level whole-steering bypass hits (`0` D2H syncs).
    pub whole_token_bypass_events: u64,
    /// Cumulative [`crate::drivers::DistortionTension`] immediate Hot-Path Ego damping events.
    pub immediate_tension_damps: u64,
    /// Cumulative [`crate::drivers::DistortionTension`] Slow-Loop stagnation excitation events.
    pub slow_tension_excites: u64,
    /// Cumulative P-Frame chain evaluations with `>= 1` hop traversed.
    pub pframe_chain_evals: u64,
    /// Cumulative P-Frame hops traversed across [`crate::drivers::EgoDifferentialNet`].
    pub pframe_total_hops: u64,
    /// Cumulative steps where validated $O \setminus S$ staged an isolated non-copy sandbox lane.
    pub exterior_sandbox_stages: u64,
    /// Cumulative steps where Unobserved ($\overline{S \cup O}$) locked aperture to `0`.
    pub unobserved_aperture_zeros: u64,
    /// Cumulative decode tokens executed in abstract whole-token bypass mode.
    pub abstract_token_count: u64,
    /// Cumulative decode tokens executed through concrete compute layers.
    pub concrete_token_count: u64,
    /// Cumulative decode residency transitions (`abstract <-> concrete`).
    pub boundary_churn_count: u64,
    /// Cumulative decode steps where host D2H logits observation was skipped.
    pub logits_skip_count: u64,
    /// Cumulative transient contradiction-guard pin relaxations.
    pub guard_relaxation_count: u64,
    /// Cumulative exit instability or `Lacuna::Undefined` recoveries.
    pub instability_recovery_count: u64,
}

mod bank;
pub use bank::*;
