// name: src/adapter/src/timeline.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dual-track `.lmnkc` / `.lmnkp` GOP container timeline (`LmnkTimeline`) built on
//! [`LmnkContainerView`] (`"LMNK"` header + section directory), GOP cut detection,
//! zero-forward seeking, explicit `TopologyMismatch` rebake signaling, and cross-model re-baking.

use std::fmt;
use std::path::Path;

use liminika_core::Pod;

use super::*;

/// Timeline metadata payload format version (`v2`) stored inside [`LMNK_SECTION_RECALL_METADATA`].
pub const LMNK_TIMELINE_FORMAT_VERSION: u32 = 2;

const MAX_LOGICAL_PFRAME_TOPOLOGY: usize = 64;
const MAX_TIMELINE_FRAMES: usize = 16_384;

/// Structured errors returned when building, loading, or rebaking an [`LmnkTimeline`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LmnkTimelineError {
    /// Static `LiminalTopologyMatrix` CRC32 mismatched (`LMNK_ERR_TOPOLOGY_MISMATCH`),
    /// indicating a clean cache invalidation that can be resolved by re-baking.
    TopologyMismatch {
        /// Active host [`CORE_TOPOLOGY`] CRC32.
        expected_crc32: u32,
        /// Container header `topology_crc32` loaded from the artifact.
        found_crc32: u32,
    },
    /// Container structure, version, checksums, or serialized timeline metadata failed validation.
    Corrupted(String),
    /// Buffer or section payload failed required alignment invariants.
    Alignment,
    /// Outer `.lmnkp` Store ZIP package envelope validation or construction failed.
    Package(String),
    /// Filesystem read/write failure.
    Io(String),
}

impl fmt::Display for LmnkTimelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TopologyMismatch {
                expected_crc32,
                found_crc32,
            } => write!(
                f,
                "topology mismatch (rebake required): expected 0x{expected_crc32:08x}, found 0x{found_crc32:08x}"
            ),
            Self::Corrupted(msg) => write!(f, "corrupted .lmnkc timeline: {msg}"),
            Self::Alignment => write!(f, "timeline alignment violation"),
            Self::Package(msg) => write!(f, "timeline .lmnkp package error: {msg}"),
            Self::Io(msg) => write!(f, "timeline I/O error: {msg}"),
        }
    }
}

impl std::error::Error for LmnkTimelineError {}

impl From<LmnkPackageError> for LmnkTimelineError {
    fn from(err: LmnkPackageError) -> Self {
        Self::Package(err.to_string())
    }
}

/// Transition telemetry evaluated by [`TimelineCutDetector`] to choose between a P-Frame delta
/// and a full I-Frame keyframe (Scene Cut) in an [`LmnkTimeline`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTransitionMetrics {
    /// Mean normalized L1 phase change relative to the previous frame (`0.0..=2.0`).
    pub phase_l1_delta: f32,
    /// Current query boundary classification (`LiminalExterior` triggers scene-cut keyframe).
    pub lacuna_state: LacunaBoundaryState,
    /// Number of P-Frames elapsed since the last I-Frame.
    pub frames_since_keyframe: usize,
    /// Whether a new attractor cluster crystallized at this step.
    pub new_cluster_spawned: bool,
}

/// Trait abstracting GOP scene-cut / keyframe insertion policy for [`LmnkTimeline`].
pub trait TimelineCutDetector {
    /// Returns `true` when the timeline should emit a full I-Frame keyframe instead of a P-Frame.
    fn should_emit_keyframe(&self, metrics: &FrameTransitionMetrics) -> bool;
}

/// Default adaptive GOP detector inserting I-Frames on max GOP length, cluster crystallization,
/// `LiminalExterior` boundary transitions, or large phase shifts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveGopCutDetector {
    /// Maximum P-Frames allowed between I-Frames.
    pub max_gop_len: usize,
    /// Phase L1 delta threshold triggering a scene-cut I-Frame.
    pub phase_delta_cut_thresh: f32,
}

impl Default for AdaptiveGopCutDetector {
    fn default() -> Self {
        Self {
            max_gop_len: 16,
            phase_delta_cut_thresh: 0.35,
        }
    }
}

impl TimelineCutDetector for AdaptiveGopCutDetector {
    fn should_emit_keyframe(&self, metrics: &FrameTransitionMetrics) -> bool {
        use liminika_core::drivers::{
            FrameDirective, SpineFrameTransitionInput, SystemSpineState,
            SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED,
        };
        use liminika_core::liminal::Lacuna;

        let gop_reached = metrics.frames_since_keyframe >= self.max_gop_len;
        let (steps_since_keyframe, max_gop_steps) = if gop_reached {
            (1, 1)
        } else {
            (
                metrics.frames_since_keyframe.min(254) as u8,
                self.max_gop_len.clamp(1, 255) as u8,
            )
        };
        let lacuna_code = if !metrics.phase_l1_delta.is_finite() {
            Lacuna::Undefined.code()
        } else if matches!(metrics.lacuna_state, LacunaBoundaryState::LiminalExterior) {
            Lacuna::Exterior.code()
        } else {
            Lacuna::SelfRef.code()
        };
        let delta_exceeded_raw = metrics.phase_l1_delta >= self.phase_delta_cut_thresh;
        let phase_delta_q7 = (metrics.phase_l1_delta * 64.0).clamp(0.0, 127.0) as u8;
        let mut reanchor_thresh_q7 = (self.phase_delta_cut_thresh * 64.0).clamp(0.0, 127.0) as u8;
        if delta_exceeded_raw && reanchor_thresh_q7 == 0 {
            reanchor_thresh_q7 = 1;
        }
        let eff_phase_delta_q7 = if delta_exceeded_raw {
            phase_delta_q7.max(reanchor_thresh_q7)
        } else if phase_delta_q7 >= reanchor_thresh_q7 && reanchor_thresh_q7 > 0 {
            phase_delta_q7.saturating_sub(1)
        } else {
            phase_delta_q7
        };
        let flags = if metrics.new_cluster_spawned {
            SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED
        } else {
            0
        };

        let (directive, _) = SystemSpineState::new().evaluate_frame_transition_branchless(
            SpineFrameTransitionInput {
                lacuna_code,
                phase_delta_q7: eff_phase_delta_q7,
                reanchor_delta_thresh_q7: reanchor_thresh_q7,
                steps_since_keyframe,
                max_gop_steps,
                candidate_granule_mask: 0xFF,
                flags,
                reserved: 0,
            },
        );
        matches!(directive, FrameDirective::ReAnchorKeyframe)
    }
}

/// Lightweight natural-language source record stored in the Source Track for
/// model-agnostic reproducibility and cross-model transcoding.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceTextRecord {
    /// Monotonic frame index (`0..`).
    pub frame_index: u64,
    /// Unix epoch timestamp in seconds.
    pub timestamp_epoch_sec: u64,
    /// Attractor role (`Fact`, `Persona`, or `Guard`).
    pub role: AttractorRole,
    /// Raw natural-language chunk.
    pub text: String,
    /// Base gravitational energy associated with this text chunk.
    pub energy: f32,
}

/// Video-codec-inspired GOP frame payload stored inside an [`LmnkTimeline`].
#[derive(Debug, Clone, PartialEq)]
pub enum GopFrameKind {
    /// Full I-Frame snapshot containing a validated unified `.lmnk` container and active policy.
    KeyframeI {
        /// Serialized unified mapped-storage container bytes (`LmnkContainerView`).
        container_bytes: Vec<u8>,
        /// Calibrated recall policy at this keyframe.
        recall_policy: RecallPolicy,
    },
    /// Incremental P-Frame storing Q7 phase motion delta and energy delta relative to the prior frame.
    DeltaP {
        /// Per-head Q7 phase motion delta (`64` heads).
        delta_q7: [lmnk_q7_t; MAX_ATTENTION_HEADS],
        /// Incremental energy change.
        energy_delta: f32,
        /// Target anchor ID updated by this step.
        target_anchor_id: u32,
    },
}

/// Single frame in an [`LmnkTimeline`] pairing a Source record with a GOP payload.
#[derive(Debug, Clone, PartialEq)]
pub struct GopTimelineFrame {
    /// Monotonic frame index.
    pub frame_index: u64,
    /// Model-agnostic natural-language source track entry.
    pub source: SourceTextRecord,
    /// Compiled I-Frame or P-Frame payload.
    pub kind: GopFrameKind,
}

/// Reconstructed state returned by zero-forward GOP seek (`AttractorContainerCodec::seek_zero_forward`).
#[derive(Debug, Clone, PartialEq)]
pub struct SeekedContainerState {
    /// Requested target frame index.
    pub target_frame_index: u64,
    /// Frame index of the nearest preceding I-Frame used as reconstruction base.
    pub keyframe_index: u64,
    /// Number of P-Frames accumulated on top of the I-Frame.
    pub p_frames_applied: usize,
    /// Active [`RecallPolicy`] restored from the keyframe.
    pub recall_policy: RecallPolicy,
    /// Reconstructed 64-head Q7 phase state after applying P-Frame deltas.
    pub active_phase_q7: [lmnk_q7_t; MAX_ATTENTION_HEADS],
    /// Reconstructed total gravitational energy.
    pub total_energy: f32,
    /// Anchor IDs present in the keyframe's mapped-storage anchor section.
    pub anchor_ids: Vec<u32>,
}

/// Context passed into [`AttractorContainerCodec::record_step`] to keep recording extensible.
pub struct TimelineStepContext<'a> {
    /// Active steering adapter providing model geometry.
    pub adapter: &'a ModelSteeringAdapter,
    /// Active compiled attractor corpus.
    pub corpus: &'a TextAttractorCorpus,
    /// Active calibrated recall policy.
    pub policy: &'a RecallPolicy,
    /// Scene-cut / keyframe detector trait object.
    pub detector: &'a dyn TimelineCutDetector,
}

/// Trait abstracting dual-track `.lmnkc` container recording, zero-forward GOP seek, and
/// cross-model natural-language track export.
pub trait AttractorContainerCodec {
    /// Appends a timeline frame (emitting an I-Frame or P-Frame per `ctx.detector`).
    /// Returns `Ok(true)` if an I-Frame keyframe was emitted, `Ok(false)` for a P-Frame.
    ///
    /// # Errors
    /// Returns an error if mapped-storage serialization or frame ordering fails.
    fn record_step(
        &mut self,
        ctx: &TimelineStepContext<'_>,
        source: SourceTextRecord,
        metrics: &FrameTransitionMetrics,
    ) -> Result<bool, String>;

    /// Reconstructs attractor phase state at `target_frame_index` by seeking to the nearest
    /// preceding I-Frame and applying subsequent P-Frames with zero LLM forward passes.
    ///
    /// # Errors
    /// Returns an error if `target_frame_index` is out of range or no preceding I-Frame exists.
    fn seek_zero_forward(&self, target_frame_index: u64) -> Result<SeekedContainerState, String>;

    /// Exports the model-agnostic Subtitle / Source Track up to `up_to_frame` (inclusive) for
    /// cross-model re-baking when `EnvProbe` detects a model change.
    fn export_subtitle_track(&self, up_to_frame: Option<u64>) -> Vec<SourceTextRecord>;

    /// Serializes the dual-track container into a unified `.lmnkc` byte stream.
    fn to_bytes(&self) -> Vec<u8>;
}

/// Abstract encoder projecting a [`SourceTextRecord`] from the Source Track into
/// per-layer `q`-space activations for a target model's geometry during cross-model re-baking.
pub trait SubtitleActivationEncoder {
    /// Encodes `record` into per-layer `q`-space activation vectors (`[layer][rotary_dim]`).
    ///
    /// # Errors
    /// Returns an error if tokenization, forward-pass activation capture, or encoding fails.
    fn encode_subtitle_activations(
        &mut self,
        record: &SourceTextRecord,
    ) -> Result<Vec<Vec<f32>>, String>;
}

/// Dependencies required to resume or re-bake an [`LmnkTimeline`] for a target model.
pub struct RebakeContext<'a> {
    /// Target model's steering adapter (defines the target [`RopeGeometry`]).
    pub adapter: &'a ModelSteeringAdapter,
    /// Recall boundary policy applied when constructing re-baked mapped-storage keyframes.
    pub policy: &'a RecallPolicy,
    /// Adaptive GOP scene-cut detector used during re-baking.
    pub detector: &'a dyn TimelineCutDetector,
    /// Optional per-layer reference centroids (`[layer][rotary_dim]`). When empty, the mean of
    /// the encoded subtitle activations is computed automatically as the reference centroid.
    pub centroid_layers: &'a [&'a [f32]],
    /// Optional inclusive upper frame limit for partial re-baking.
    pub up_to_frame: Option<u64>,
}

/// Outcome of [`LmnkTimeline::resume_or_rebake`] after `EnvProbe` fingerprint checking.
#[derive(Debug, Clone)]
pub enum TimelineResumeOutcome {
    /// Target model geometry matched `model_fingerprint`; state was restored via zero-forward seek.
    FastResumed {
        /// Reconstructed state at the requested target frame.
        seek_state: SeekedContainerState,
    },
    /// Target model geometry differed; Source Track records were re-baked into the new space.
    Rebaked {
        /// Newly re-baked dual-track timeline bound to `target_fingerprint`.
        timeline: LmnkTimeline,
        /// Re-baked dynamic attractor corpus aligned with the target model.
        corpus: TextAttractorCorpus,
        /// Reconstructed state at the requested target frame on the re-baked timeline.
        seek_state: SeekedContainerState,
        /// Fingerprint recorded in the source container.
        source_fingerprint: u64,
        /// Fingerprint of the target model's [`RopeGeometry`].
        target_fingerprint: u64,
        /// Number of source records re-baked into the new timeline.
        rebaked_frames: usize,
    },
}

/// Model-agnostic logical P-Frame node descriptor persisted in unified `.lmnkc` / `.lmnkp`
/// containers so portable [`EgoDifferentialNet`] chains can be re-baked to any target model's
/// 64-head Q7 space upon load without modifying base weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LogicalPFrameDescriptor {
    /// Logical P-Frame node identifier (`1..`).
    pub node_id: u32,
    /// Reference logical I-Frame anchor identifier.
    pub anchor_ref_id: u32,
    /// Next hop index (`0..8`, or `0xFFFF` if terminal).
    pub next_hop_idx: u16,
    /// Hop depth along the chain (`1..=4`).
    pub hop_depth: u8,
    /// 8-granule lacuna trigger bitmask.
    pub lacuna_trigger_mask: u8,
    /// Scope overlap region selector bitmask (`SCOPE_REGION_*`).
    pub scope_region_selector: u8,
    /// Stigmergic pheromone density in Q0.7 (`0..=127`).
    pub pheromone_q7: u8,
    /// Directed delta scaling gain in Q0.7 (`0..=127`).
    pub gain_q7: u8,
    /// Model-agnostic semantic Ego intent axes transition (`16 B`).
    pub intent_delta: EgoIntentAxes,
}

/// Dual-Track (Source Natural-Language Track + KV-keyframe Video-Style GOP Track) unified `.lmnkc`
/// / `.lmnkp` container timeline built on [`LmnkContainerView`].
#[derive(Debug, Clone, PartialEq)]
pub struct LmnkTimeline {
    model_fingerprint: u64,
    frames: Vec<GopTimelineFrame>,
    seek_index: Vec<usize>,
    persona_circuit: Option<CognitiveTrajectoryLedger>,
    logical_pframe_topology: Vec<LogicalPFrameDescriptor>,
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn read_exact(&mut self, n: usize) -> Result<&'a [u8], LmnkTimelineError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or_else(|| LmnkTimelineError::Corrupted("cursor offset overflow".to_string()))?;
        let slice = self.bytes.get(self.pos..end).ok_or_else(|| {
            LmnkTimelineError::Corrupted("unexpected EOF in timeline section".to_string())
        })?;
        self.pos = end;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, LmnkTimelineError> {
        Ok(*self
            .read_exact(1)?
            .first()
            .ok_or_else(|| LmnkTimelineError::Corrupted("missing u8".to_string()))?)
    }

    fn read_u16_le(&mut self) -> Result<u16, LmnkTimelineError> {
        let s = self.read_exact(2)?;
        let arr: [u8; 2] = s
            .try_into()
            .map_err(|_| LmnkTimelineError::Corrupted("u16 read failed".to_string()))?;
        Ok(u16::from_le_bytes(arr))
    }

    fn read_u32_le(&mut self) -> Result<u32, LmnkTimelineError> {
        let s = self.read_exact(4)?;
        let arr: [u8; 4] = s
            .try_into()
            .map_err(|_| LmnkTimelineError::Corrupted("u32 read failed".to_string()))?;
        Ok(u32::from_le_bytes(arr))
    }

    fn read_u64_le(&mut self) -> Result<u64, LmnkTimelineError> {
        let s = self.read_exact(8)?;
        let arr: [u8; 8] = s
            .try_into()
            .map_err(|_| LmnkTimelineError::Corrupted("u64 read failed".to_string()))?;
        Ok(u64::from_le_bytes(arr))
    }

    fn read_f32_le(&mut self) -> Result<f32, LmnkTimelineError> {
        Ok(f32::from_bits(self.read_u32_le()?))
    }
}

fn map_container_err(bytes: &[u8], err: LmnkStatus) -> LmnkTimelineError {
    if err == LMNK_ERR_TOPOLOGY_MISMATCH {
        let found_crc32 = bytes
            .get(8..12)
            .and_then(|s| <[u8; 4]>::try_from(s).ok())
            .map(u32::from_le_bytes)
            .unwrap_or(0);
        LmnkTimelineError::TopologyMismatch {
            expected_crc32: CORE_TOPOLOGY.topology_crc32,
            found_crc32,
        }
    } else if err == LMNK_ERR_ALIGNMENT {
        LmnkTimelineError::Alignment
    } else {
        LmnkTimelineError::Corrupted(format!(
            "LmnkContainerView rejected container: {} ({})",
            err.as_str(),
            err.code()
        ))
    }
}

impl LmnkTimeline {
    /// Computes a deterministic 64-bit fingerprint for a [`RopeGeometry`] so `EnvProbe` can detect
    /// model switches requiring subtitle-track transcoding.
    #[must_use]
    pub fn geometry_fingerprint(geom: &RopeGeometry) -> u64 {
        let mut h = 0xCBF2_9CE4_8422_2325_u64;
        for val in [
            geom.num_layers as u64,
            geom.num_q_heads as u64,
            geom.num_kv_heads as u64,
            geom.head_dim as u64,
            geom.rotary_dim as u64,
            geom.layer_rope_mask,
            match geom.layout {
                RopeLayout::Interleaved => 1,
                RopeLayout::HalfSplit => 2,
            },
        ] {
            h ^= val;
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        h
    }

    /// Creates an empty [`LmnkTimeline`] bound to `model_fingerprint`.
    #[must_use]
    pub const fn new(model_fingerprint: u64) -> Self {
        Self {
            model_fingerprint,
            frames: Vec::new(),
            seek_index: Vec::new(),
            persona_circuit: None,
            logical_pframe_topology: Vec::new(),
        }
    }

    /// Attaches model-agnostic logical P-Frame topology descriptors for unified container persistence.
    pub fn set_logical_pframe_topology(&mut self, topology: Vec<LogicalPFrameDescriptor>) {
        self.logical_pframe_topology = topology;
    }

    /// Returns persisted model-agnostic logical P-Frame topology descriptors.
    #[must_use]
    pub fn logical_pframe_topology(&self) -> &[LogicalPFrameDescriptor] {
        &self.logical_pframe_topology
    }

    /// Attaches a [`CognitiveTrajectoryLedger`] (stigmergic persona reasoning circuit) to be
    /// persisted inside unified `.lmnkc` / `.lmnkp` containers.
    pub fn set_persona_circuit(&mut self, circuit: CognitiveTrajectoryLedger) {
        self.persona_circuit = Some(circuit);
    }

    /// Returns the active [`CognitiveTrajectoryLedger`] if set in memory, or extracts the most
    /// recent persisted circuit from the timeline's I-frame `.lmnk` container snapshots.
    #[must_use]
    pub fn persona_circuit(&self) -> Option<CognitiveTrajectoryLedger> {
        self.persona_circuit
            .or_else(|| Self::extract_circuit_from_frames(&self.frames))
    }

    /// Composes this container's persona circuit (`self`) with `overlay` using
    /// [`CognitiveTrajectoryLedger::compose_persona_circuit`] (`blend_q7` in `0..=128`).
    #[must_use]
    pub fn compose_persona_circuits(
        &self,
        overlay: &Self,
        blend_q7: u8,
    ) -> Option<CognitiveTrajectoryLedger> {
        match (self.persona_circuit(), overlay.persona_circuit()) {
            (Some(base), Some(ov)) => Some(base.compose_persona_circuit(&ov, blend_q7)),
            (Some(base), None) => Some(base),
            (None, Some(ov)) => Some(ov),
            (None, None) => None,
        }
    }

    fn extract_circuit_from_snapshot(snapshot: &[u8]) -> Option<CognitiveTrajectoryLedger> {
        if snapshot.is_empty() {
            return None;
        }
        LmnkContainerView::parse(snapshot)
            .ok()
            .and_then(|v| v.cognitive_trajectory_ledger())
    }

    fn extract_circuit_from_frames(
        frames: &[GopTimelineFrame],
    ) -> Option<CognitiveTrajectoryLedger> {
        frames.iter().rev().find_map(|f| match &f.kind {
            GopFrameKind::KeyframeI {
                container_bytes, ..
            } => Self::extract_circuit_from_snapshot(container_bytes),
            GopFrameKind::DeltaP { .. } => None,
        })
    }

    /// Returns the geometry fingerprint this compiled GOP track was baked for.
    #[must_use]
    pub const fn model_fingerprint(&self) -> u64 {
        self.model_fingerprint
    }

    /// Returns all recorded timeline frames.
    #[must_use]
    pub fn frames(&self) -> &[GopTimelineFrame] {
        &self.frames
    }

    /// Returns frame indices of all I-Frame keyframes in the seek table.
    #[must_use]
    pub fn keyframe_positions(&self) -> &[usize] {
        &self.seek_index
    }

    fn slot_for_role(role: AttractorRole) -> u8 {
        match role {
            AttractorRole::Fact => ATTRACTOR_SLOT_KNOWLEDGE,
            AttractorRole::Persona => ATTRACTOR_SLOT_PERSONA,
            AttractorRole::Guard => ATTRACTOR_SLOT_REPULSION,
        }
    }

    fn build_mapped_storage_snapshot(
        corpus: &TextAttractorCorpus,
        timestamp_epoch_sec: u64,
        persona_circuit: Option<&CognitiveTrajectoryLedger>,
    ) -> Result<Vec<u8>, String> {
        let mut anchors = Vec::with_capacity(corpus.entries().len().max(1));
        for entry in corpus.entries() {
            let mut rec =
                PhaseAnchorRecord::new_empty(entry.anchor_id, Self::slot_for_role(entry.role), 0);
            rec.base_energy = entry.energy;
            rec.phase_gravity_mass = entry.energy;
            rec.sample_count = 1;
            if let Some(last_layer) = entry.per_layer_q7.last() {
                rec.phase_profile = *last_layer;
            }
            rec.checksum_crc32 = 0;
            anchors.push(rec);
        }
        if anchors.is_empty() {
            anchors.push(PhaseAnchorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0));
        }
        let trail = lmnk_pointer_trail_t::default();
        let trails = [trail];
        let ledger_slot = persona_circuit.map(CognitiveTrajectoryLedger::to_persona_ledger_slot);
        let req_len = LmnkContainerView::required_standard_byte_len(
            anchors.len(),
            trails.len(),
            false,
            ledger_slot.is_some(),
            0,
        )
        .map_err(|e| format!("required_standard_byte_len error: {e}"))?;
        let mut out = vec![0_u8; req_len];
        let written = LmnkContainerView::write_standard_sections_to_bytes(
            LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            &anchors,
            &trails,
            None,
            ledger_slot.as_ref(),
            &[],
            timestamp_epoch_sec,
            &mut out,
        )
        .map_err(|e| format!("write_standard_sections_to_bytes error: {e}"))?;
        out.truncate(written);
        Ok(out)
    }

    fn latest_phase_q7(&self) -> [lmnk_q7_t; MAX_ATTENTION_HEADS] {
        if let Some(last) = self.frames.last() {
            if let Ok(state) = self.seek_zero_forward(last.frame_index) {
                return state.active_phase_q7;
            }
        }
        [0_i8; MAX_ATTENTION_HEADS]
    }

    fn encode_recall_metadata_payload(&self) -> Result<Vec<u8>, LmnkTimelineError> {
        let frame_count_u32 = u32::try_from(self.frames.len())
            .map_err(|_| LmnkTimelineError::Corrupted("frame count exceeds u32".to_string()))?;
        let mut out = Vec::new();
        out.extend_from_slice(&LMNK_TIMELINE_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.model_fingerprint.to_le_bytes());
        out.extend_from_slice(&frame_count_u32.to_le_bytes());
        for f in &self.frames {
            out.extend_from_slice(&f.frame_index.to_le_bytes());
            out.extend_from_slice(&f.source.timestamp_epoch_sec.to_le_bytes());
            let role_byte = match f.source.role {
                AttractorRole::Fact => 0_u8,
                AttractorRole::Persona => 1_u8,
                AttractorRole::Guard => 2_u8,
            };
            out.push(role_byte);
            out.extend_from_slice(&f.source.energy.to_le_bytes());
            let tb = f.source.text.as_bytes();
            let tb_len = u32::try_from(tb.len())
                .map_err(|_| LmnkTimelineError::Corrupted("text length exceeds u32".to_string()))?;
            out.extend_from_slice(&tb_len.to_le_bytes());
            out.extend_from_slice(tb);
            match &f.kind {
                GopFrameKind::KeyframeI {
                    container_bytes,
                    recall_policy,
                } => {
                    out.push(0_u8);
                    let tkf = u32::try_from(recall_policy.top_k_facts).map_err(|_| {
                        LmnkTimelineError::Corrupted("top_k_facts exceeds u32".to_string())
                    })?;
                    let tkp = u32::try_from(recall_policy.top_k_personas).map_err(|_| {
                        LmnkTimelineError::Corrupted("top_k_personas exceeds u32".to_string())
                    })?;
                    let clen = u32::try_from(container_bytes.len()).map_err(|_| {
                        LmnkTimelineError::Corrupted("keyframe bytes exceed u32".to_string())
                    })?;
                    out.extend_from_slice(&tkf.to_le_bytes());
                    out.extend_from_slice(&tkp.to_le_bytes());
                    out.extend_from_slice(&recall_policy.min_coherence.to_le_bytes());
                    out.extend_from_slice(&recall_policy.boundary_margin.to_le_bytes());
                    out.extend_from_slice(&recall_policy.guard_suppress_margin.to_le_bytes());
                    out.extend_from_slice(&clen.to_le_bytes());
                    out.extend_from_slice(container_bytes);
                }
                GopFrameKind::DeltaP {
                    delta_q7,
                    energy_delta,
                    target_anchor_id,
                } => {
                    out.push(1_u8);
                    for &q in delta_q7 {
                        out.push(q as u8);
                    }
                    out.extend_from_slice(&energy_delta.to_le_bytes());
                    out.extend_from_slice(&target_anchor_id.to_le_bytes());
                }
            }
        }
        Ok(out)
    }

    fn decode_recall_metadata_payload(
        bytes: &[u8],
    ) -> Result<(u64, Vec<GopTimelineFrame>, Vec<usize>), LmnkTimelineError> {
        let mut cur = ByteCursor::new(bytes);
        let ver = cur.read_u32_le()?;
        if ver != LMNK_TIMELINE_FORMAT_VERSION {
            return Err(LmnkTimelineError::Corrupted(format!(
                "unsupported timeline version: {ver}"
            )));
        }
        let model_fingerprint = cur.read_u64_le()?;
        let frame_count = cur.read_u32_le()? as usize;
        if frame_count > MAX_TIMELINE_FRAMES {
            return Err(LmnkTimelineError::Corrupted(format!(
                "frame count {frame_count} exceeds limit {MAX_TIMELINE_FRAMES}"
            )));
        }

        let mut frames = Vec::with_capacity(frame_count);
        let mut seek_index = Vec::new();
        for idx in 0..frame_count {
            let frame_index = cur.read_u64_le()?;
            let ts = cur.read_u64_le()?;
            let role = match cur.read_u8()? {
                0 => AttractorRole::Fact,
                1 => AttractorRole::Persona,
                2 => AttractorRole::Guard,
                r => {
                    return Err(LmnkTimelineError::Corrupted(format!(
                        "unknown AttractorRole byte: {r}"
                    )))
                }
            };
            let energy = cur.read_f32_le()?;
            let text_len = cur.read_u32_le()? as usize;
            let text_bytes = cur.read_exact(text_len)?;
            let text = std::str::from_utf8(text_bytes)
                .map_err(|e| {
                    LmnkTimelineError::Corrupted(format!("invalid UTF-8 in source track: {e}"))
                })?
                .to_string();

            let kind_tag = cur.read_u8()?;
            let kind = match kind_tag {
                0 => {
                    let top_k_facts = cur.read_u32_le()? as usize;
                    let top_k_personas = cur.read_u32_le()? as usize;
                    let min_coherence = cur.read_f32_le()?;
                    let boundary_margin = cur.read_f32_le()?;
                    let guard_suppress_margin = cur.read_f32_le()?;
                    let c_len = cur.read_u32_le()? as usize;
                    let container_bytes = cur.read_exact(c_len)?.to_vec();
                    let _ = LmnkContainerView::parse(&container_bytes)
                        .map_err(|e| map_container_err(&container_bytes, e))?;
                    seek_index.push(idx);
                    GopFrameKind::KeyframeI {
                        container_bytes,
                        recall_policy: RecallPolicy {
                            top_k_facts,
                            top_k_personas,
                            min_coherence,
                            boundary_margin,
                            guard_suppress_margin,
                        },
                    }
                }
                1 => {
                    let q7_bytes = cur.read_exact(MAX_ATTENTION_HEADS)?;
                    let mut delta_q7 = [0_i8; MAX_ATTENTION_HEADS];
                    for (dst, &b) in delta_q7.iter_mut().zip(q7_bytes.iter()) {
                        *dst = b as i8;
                    }
                    let energy_delta = cur.read_f32_le()?;
                    let target_anchor_id = cur.read_u32_le()?;
                    GopFrameKind::DeltaP {
                        delta_q7,
                        energy_delta,
                        target_anchor_id,
                    }
                }
                other => {
                    return Err(LmnkTimelineError::Corrupted(format!(
                        "unknown GopFrameKind tag: {other}"
                    )))
                }
            };
            frames.push(GopTimelineFrame {
                frame_index,
                source: SourceTextRecord {
                    frame_index,
                    timestamp_epoch_sec: ts,
                    role,
                    text,
                    energy,
                },
                kind,
            });
        }
        if cur.remaining() != 0 {
            return Err(LmnkTimelineError::Corrupted(format!(
                "trailing {} unparsed bytes in recall metadata section",
                cur.remaining()
            )));
        }
        Ok((model_fingerprint, frames, seek_index))
    }

    fn encode_logical_pframes_payload(&self) -> Result<Vec<u8>, LmnkTimelineError> {
        if self.logical_pframe_topology.len() > MAX_LOGICAL_PFRAME_TOPOLOGY {
            return Err(LmnkTimelineError::Corrupted(format!(
                "logical P-Frame topology count {} exceeds {MAX_LOGICAL_PFRAME_TOPOLOGY}",
                self.logical_pframe_topology.len()
            )));
        }
        let count_u32 = self.logical_pframe_topology.len() as u32;
        let mut out = Vec::with_capacity(4 + self.logical_pframe_topology.len() * 32);
        out.extend_from_slice(&count_u32.to_le_bytes());
        for d in &self.logical_pframe_topology {
            out.extend_from_slice(&d.node_id.to_le_bytes());
            out.extend_from_slice(&d.anchor_ref_id.to_le_bytes());
            out.extend_from_slice(&d.next_hop_idx.to_le_bytes());
            out.push(d.hop_depth);
            out.push(d.lacuna_trigger_mask);
            out.push(d.scope_region_selector);
            out.push(d.pheromone_q7);
            out.push(d.gain_q7);
            out.push(d.intent_delta.autonomy_q7 as u8);
            out.push(d.intent_delta.transgression_q7 as u8);
            out.push(d.intent_delta.devotion_q7 as u8);
            out.push(d.intent_delta.logic_q7 as u8);
            out.push(d.intent_delta.caution_q7 as u8);
            out.push(d.intent_delta.defiance_q7 as u8);
            out.push(d.intent_delta.synergy_q7 as u8);
            out.push(d.intent_delta.vitality_q7 as u8);
            out.push(d.intent_delta.preferred_slot_mask);
            out.push(d.intent_delta.active_granule_bias_mask);
            out.push(d.intent_delta.confidence_q7);
            out.push(d.intent_delta.flags);
            out.extend_from_slice(&d.intent_delta.signature_hash.to_le_bytes());
        }
        Ok(out)
    }

    fn decode_logical_pframes_payload(
        bytes: &[u8],
    ) -> Result<Vec<LogicalPFrameDescriptor>, LmnkTimelineError> {
        let mut cur = ByteCursor::new(bytes);
        let topo_count = cur.read_u32_le()? as usize;
        if topo_count > MAX_LOGICAL_PFRAME_TOPOLOGY {
            return Err(LmnkTimelineError::Corrupted(format!(
                "excessive logical P-Frame topology count: {topo_count}"
            )));
        }
        let mut out = Vec::with_capacity(topo_count);
        for _ in 0..topo_count {
            let node_id = cur.read_u32_le()?;
            let anchor_ref_id = cur.read_u32_le()?;
            let next_hop_idx = cur.read_u16_le()?;
            let hop_depth = cur.read_u8()?;
            let lacuna_trigger_mask = cur.read_u8()?;
            let scope_region_selector = cur.read_u8()?;
            let pheromone_q7 = cur.read_u8()?;
            let gain_q7 = cur.read_u8()?;
            let ax = cur.read_exact(12)?;
            let sig_hash = cur.read_u32_le()?;
            let intent_delta = EgoIntentAxes {
                autonomy_q7: *ax.first().unwrap_or(&0) as i8,
                transgression_q7: *ax.get(1).unwrap_or(&0) as i8,
                devotion_q7: *ax.get(2).unwrap_or(&0) as i8,
                logic_q7: *ax.get(3).unwrap_or(&0) as i8,
                caution_q7: *ax.get(4).unwrap_or(&0) as i8,
                defiance_q7: *ax.get(5).unwrap_or(&0) as i8,
                synergy_q7: *ax.get(6).unwrap_or(&0) as i8,
                vitality_q7: *ax.get(7).unwrap_or(&0) as i8,
                preferred_slot_mask: *ax.get(8).unwrap_or(&0),
                active_granule_bias_mask: *ax.get(9).unwrap_or(&0),
                confidence_q7: *ax.get(10).unwrap_or(&0),
                flags: *ax.get(11).unwrap_or(&0),
                signature_hash: sig_hash,
            };
            out.push(LogicalPFrameDescriptor {
                node_id,
                anchor_ref_id,
                next_hop_idx,
                hop_depth,
                lacuna_trigger_mask,
                scope_region_selector,
                pheromone_q7,
                gain_q7,
                intent_delta,
            });
        }
        if cur.remaining() != 0 {
            return Err(LmnkTimelineError::Corrupted(format!(
                "trailing {} unparsed bytes in logical P-Frame section",
                cur.remaining()
            )));
        }
        Ok(out)
    }

    /// Serializes this timeline directly into a unified `"LMNK"` `.lmnkc` binary container
    /// (`LmnkContainerHeader` + `LmnkSectionEntry` directory). Empty optional sections are omitted
    /// so every emitted section entry satisfies `length > 0`.
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError`] if any inner payload or container boundary rule is violated.
    pub fn to_container_bytes(&self, container_flags: u32) -> Result<Vec<u8>, LmnkTimelineError> {
        let mut anchor_sec: Option<Vec<u8>> = None;
        let mut trail_sec: Option<Vec<u8>> = None;
        let mut bloom_sec: Option<Vec<u8>> = None;
        let mut ledger_sec: Option<Vec<u8>> = None;
        let mut prov_sec: Option<Vec<u8>> = None;

        // Extract latest valid KeyframeI mapped-storage sections when available.
        for frame in self.frames.iter().rev() {
            if let GopFrameKind::KeyframeI {
                container_bytes, ..
            } = &frame.kind
            {
                if let Ok(view) = LmnkContainerView::parse(container_bytes) {
                    anchor_sec = view
                        .section_bytes(LMNK_SECTION_PHASE_ANCHORS)
                        .filter(|s| !s.is_empty())
                        .map(<[u8]>::to_vec);
                    trail_sec = view
                        .section_bytes(LMNK_SECTION_POINTER_TRAILS)
                        .filter(|s| !s.is_empty())
                        .map(<[u8]>::to_vec);
                    bloom_sec = view
                        .section_bytes(LMNK_SECTION_BLOOM_FILTER)
                        .filter(|s| !s.is_empty())
                        .map(<[u8]>::to_vec);
                    ledger_sec = view
                        .section_bytes(LMNK_SECTION_LEDGER_SLOT)
                        .filter(|s| !s.is_empty())
                        .map(<[u8]>::to_vec);
                    prov_sec = view
                        .section_bytes(LMNK_SECTION_CAUSAL_PROVENANCE)
                        .filter(|s| !s.is_empty())
                        .map(<[u8]>::to_vec);
                    break;
                }
            }
        }
        if let Some(ref circuit) = self.persona_circuit {
            let slot = circuit.to_persona_ledger_slot();
            ledger_sec = Some(slot.as_bytes().to_vec());
        }

        let pframes_sec = if self.logical_pframe_topology.is_empty() {
            None
        } else {
            Some(self.encode_logical_pframes_payload()?)
        };
        let recall_meta_sec = self.encode_recall_metadata_payload()?;

        let mut sections: Vec<LmnkSectionInput<'_>> = Vec::with_capacity(7);
        if let Some(ref bytes) = anchor_sec {
            sections.push(LmnkSectionInput {
                kind: LMNK_SECTION_PHASE_ANCHORS,
                flags: 0,
                payload: bytes,
            });
        }
        if let Some(ref bytes) = trail_sec {
            sections.push(LmnkSectionInput {
                kind: LMNK_SECTION_POINTER_TRAILS,
                flags: 0,
                payload: bytes,
            });
        }
        if let Some(ref bytes) = bloom_sec {
            sections.push(LmnkSectionInput {
                kind: LMNK_SECTION_BLOOM_FILTER,
                flags: 0,
                payload: bytes,
            });
        }
        if let Some(ref bytes) = ledger_sec {
            sections.push(LmnkSectionInput {
                kind: LMNK_SECTION_LEDGER_SLOT,
                flags: 0,
                payload: bytes,
            });
        }
        if let Some(ref bytes) = prov_sec {
            sections.push(LmnkSectionInput {
                kind: LMNK_SECTION_CAUSAL_PROVENANCE,
                flags: 0,
                payload: bytes,
            });
        }
        if let Some(ref bytes) = pframes_sec {
            sections.push(LmnkSectionInput {
                kind: LMNK_SECTION_LOGICAL_PFRAMES,
                flags: 0,
                payload: bytes,
            });
        }
        sections.push(LmnkSectionInput {
            kind: LMNK_SECTION_RECALL_METADATA,
            flags: 0,
            payload: &recall_meta_sec,
        });

        let latest_ts = self
            .frames
            .last()
            .map_or(0, |f| f.source.timestamp_epoch_sec);
        let req_len = required_container_byte_len(&sections)
            .map_err(|e| LmnkTimelineError::Corrupted(format!("container sizing failed: {e}")))?;
        let mut out = vec![0_u8; req_len];
        let written =
            write_container_sections_to_bytes(container_flags, latest_ts, &sections, &mut out)
                .map_err(|e| {
                    LmnkTimelineError::Corrupted(format!("container write failed: {e}"))
                })?;
        out.truncate(written);
        Ok(out)
    }

    /// Exports this timeline as a 64B-aligned PKZIP Store-only `.lmnkp` package byte buffer
    /// containing `manifest.json`, optional `source.lmnk`, and 64B-aligned `compiled.lmnkc`.
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError`] if container or `.lmnkp` encoding fails.
    pub fn export_lmnkp_bytes(
        &self,
        source_lmnk: Option<&str>,
    ) -> Result<Vec<u8>, LmnkTimelineError> {
        let compiled_bytes = self.to_container_bytes(LMNK_CONTAINER_FLAG_PORTABLE_PACKAGE)?;
        let header = LmnkContainerHeader::from_le_bytes(&compiled_bytes)
            .map_err(|e| map_container_err(&compiled_bytes, e))?;
        let manifest = format_package_manifest_json(
            LMNK_ABI_VERSION_STRING,
            self.model_fingerprint,
            self.frames.len(),
            header.topology_crc32,
            LMNKP_ENTRY_COMPILED,
            source_lmnk.map(|_| LMNKP_ENTRY_SOURCE),
        );

        let mut builder = LmnkPackageBuilder::new();
        builder.add_manifest_json(manifest)?;
        if let Some(src) = source_lmnk {
            builder.add_source_lmnk(src.as_bytes().to_vec())?;
        }
        builder.add_compiled_lmnkc(compiled_bytes)?;
        Ok(builder.build()?)
    }

    /// Validates and imports an [`LmnkTimeline`] from an `.lmnkp` Store ZIP package byte slice,
    /// verifying both the 64B alignment of `compiled.lmnkc` and [`LmnkContainerView::parse`].
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError::TopologyMismatch`], [`LmnkTimelineError::Package`], or
    /// [`LmnkTimelineError::Corrupted`].
    pub fn import_lmnkp_bytes(bytes: &[u8]) -> Result<Self, LmnkTimelineError> {
        let pkg = LmnkPackageView::parse(bytes)?;
        let compiled = pkg.compiled_lmnkc()?;
        Self::from_bytes(compiled)
    }

    /// Writes the serialized unified `.lmnkc` container to `path`, returning bytes written.
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError`] if serialization or filesystem creation/write fails.
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<usize, LmnkTimelineError> {
        let p = path.as_ref();
        if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                LmnkTimelineError::Io(format!(
                    "failed to create parent directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let bytes = self.to_container_bytes(LMNK_CONTAINER_FLAG_COMPILED_CACHE)?;
        std::fs::write(p, &bytes).map_err(|e| {
            LmnkTimelineError::Io(format!(
                "failed to write .lmnkc container {}: {e}",
                p.display()
            ))
        })?;
        Ok(bytes.len())
    }

    /// Writes a 64B-aligned `.lmnkp` package archive to `path`, returning bytes written.
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError`] if package creation or filesystem write fails.
    pub fn save_lmnkp_to_path(
        &self,
        path: impl AsRef<Path>,
        source_lmnk: Option<&str>,
    ) -> Result<usize, LmnkTimelineError> {
        let p = path.as_ref();
        if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                LmnkTimelineError::Io(format!(
                    "failed to create parent directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let bytes = self.export_lmnkp_bytes(source_lmnk)?;
        std::fs::write(p, &bytes).map_err(|e| {
            LmnkTimelineError::Io(format!(
                "failed to write .lmnkp package {}: {e}",
                p.display()
            ))
        })?;
        Ok(bytes.len())
    }

    /// Loads and validates an `.lmnkc` container or `.lmnkp` package into an owned buffer from `path`.
    ///
    /// Reads into an owned `Vec<u8>` (never OS-`mmap`ping untrusted external packages) and
    /// distinguishes [`LmnkTimelineError::TopologyMismatch`] for clean rebake fallback.
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError`] on I/O, ZIP, topology mismatch, or container validation errors.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, LmnkTimelineError> {
        let p = path.as_ref();
        let meta_len = std::fs::metadata(p)
            .map_err(|e| {
                LmnkTimelineError::Io(format!("failed to stat container {}: {e}", p.display()))
            })?
            .len();
        if meta_len > LMNKP_MAX_PACKAGE_BYTES as u64 {
            return Err(LmnkTimelineError::Corrupted(format!(
                "container {} ({meta_len} B) exceeds hard cap {LMNKP_MAX_PACKAGE_BYTES}",
                p.display()
            )));
        }
        let bytes = std::fs::read(p).map_err(|e| {
            LmnkTimelineError::Io(format!("failed to read container {}: {e}", p.display()))
        })?;
        if bytes.starts_with(b"PK\x03\x04") || p.extension().is_some_and(|e| e == "lmnkp") {
            Self::import_lmnkp_bytes(&bytes)
        } else {
            Self::from_bytes(&bytes)
        }
    }

    /// Compiles a sequence of model-agnostic [`SourceTextRecord`] entries into a new
    /// [`LmnkTimeline`] and [`TextAttractorCorpus`] for `ctx.adapter.geometry()`
    /// using `encoder`.
    ///
    /// # Errors
    /// Returns an error if `subtitles` is empty, activation encoding fails, or GOP recording fails.
    pub fn bake_from_subtitle_records<E: SubtitleActivationEncoder + ?Sized>(
        subtitles: &[SourceTextRecord],
        ctx: &RebakeContext<'_>,
        encoder: &mut E,
    ) -> Result<(Self, TextAttractorCorpus), String> {
        if subtitles.is_empty() {
            return Err("cannot bake an empty subtitle track".to_string());
        }

        let target_fp = Self::geometry_fingerprint(ctx.adapter.geometry());
        let mut encoded_records = Vec::with_capacity(subtitles.len());
        for record in subtitles {
            encoded_records.push(encoder.encode_subtitle_activations(record)?);
        }

        let num_layers = encoded_records[0].len();
        let auto_centroids: Vec<Vec<f32>> = if ctx.centroid_layers.is_empty() {
            let mut sums: Vec<Vec<f32>> = encoded_records[0]
                .iter()
                .map(|layer| vec![0.0_f32; layer.len()])
                .collect();
            for rec_layers in &encoded_records {
                if rec_layers.len() != num_layers {
                    return Err(
                        "inconsistent layer count from SubtitleActivationEncoder".to_string()
                    );
                }
                for (l, src) in rec_layers.iter().enumerate() {
                    if src.len() != sums[l].len() {
                        return Err(format!(
                            "inconsistent rotary dimension at layer {l}: {} vs {}",
                            src.len(),
                            sums[l].len()
                        ));
                    }
                    for (dst, &val) in sums[l].iter_mut().zip(src.iter()) {
                        *dst += val;
                    }
                }
            }
            let inv_n = 1.0_f32 / (encoded_records.len() as f32);
            for layer_sum in &mut sums {
                for val in layer_sum.iter_mut() {
                    *val *= inv_n;
                }
            }
            sums
        } else {
            Vec::new()
        };

        let auto_refs: Vec<&[f32]> = auto_centroids.iter().map(Vec::as_slice).collect();
        let ref_layers: &[&[f32]] = if ctx.centroid_layers.is_empty() {
            &auto_refs
        } else {
            ctx.centroid_layers
        };

        let mut rebaked_corpus = TextAttractorCorpus::new();
        for (record, q_layers) in subtitles.iter().zip(encoded_records.iter()) {
            let q_refs: Vec<&[f32]> = q_layers.iter().map(Vec::as_slice).collect();
            rebaked_corpus.ingest_text_activations(
                ctx.adapter,
                record.role,
                &record.text,
                &q_refs,
                ref_layers,
                record.energy,
            )?;
        }
        rebaked_corpus.fit_common_mode_subspace(3);

        let mut rebaked_timeline = Self::new(target_fp);
        let mut frames_since_keyframe = 0_usize;
        let step_ctx = TimelineStepContext {
            adapter: ctx.adapter,
            corpus: &rebaked_corpus,
            policy: ctx.policy,
            detector: ctx.detector,
        };
        for (idx, record) in subtitles.iter().enumerate() {
            let scene_cut = idx > 0 && record.role != subtitles[idx - 1].role;
            let emitted_i = rebaked_timeline.record_step(
                &step_ctx,
                record.clone(),
                &FrameTransitionMetrics {
                    phase_l1_delta: if idx == 0 { 0.0 } else { 0.08 },
                    lacuna_state: if scene_cut {
                        LacunaBoundaryState::LiminalExterior
                    } else {
                        LacunaBoundaryState::InsideBasin
                    },
                    frames_since_keyframe,
                    new_cluster_spawned: scene_cut,
                },
            )?;
            frames_since_keyframe = if emitted_i {
                0
            } else {
                frames_since_keyframe + 1
            };
        }

        Ok((rebaked_timeline, rebaked_corpus))
    }

    /// Checks `EnvProbe` compatibility against `ctx.adapter.geometry()` and either performs a
    /// zero-forward GOP seek (`FastResumed`) or automatically re-bakes the Source Track through
    /// `encoder` into the target model's geometry (`Rebaked`).
    ///
    /// # Errors
    /// Returns an error if seeking, encoding subtitle activations, or recording re-baked frames fails.
    pub fn resume_or_rebake<E: SubtitleActivationEncoder + ?Sized>(
        &self,
        ctx: &RebakeContext<'_>,
        encoder: &mut E,
        target_frame: u64,
    ) -> Result<TimelineResumeOutcome, String> {
        let target_fp = Self::geometry_fingerprint(ctx.adapter.geometry());
        if self.model_fingerprint == target_fp {
            let seek_state = self.seek_zero_forward(target_frame)?;
            return Ok(TimelineResumeOutcome::FastResumed { seek_state });
        }

        let subtitles = self.export_subtitle_track(ctx.up_to_frame);
        let (mut rebaked_timeline, rebaked_corpus) =
            Self::bake_from_subtitle_records(&subtitles, ctx, encoder)?;
        rebaked_timeline.set_logical_pframe_topology(self.logical_pframe_topology.clone());
        if let Some(circuit) = self.persona_circuit() {
            rebaked_timeline.set_persona_circuit(circuit);
        }

        let effective_target = subtitles
            .iter()
            .filter(|r| r.frame_index <= target_frame)
            .map(|r| r.frame_index)
            .next_back()
            .unwrap_or(subtitles[0].frame_index);
        let seek_state = rebaked_timeline.seek_zero_forward(effective_target)?;

        Ok(TimelineResumeOutcome::Rebaked {
            timeline: rebaked_timeline,
            corpus: rebaked_corpus,
            seek_state,
            source_fingerprint: self.model_fingerprint,
            target_fingerprint: target_fp,
            rebaked_frames: subtitles.len(),
        })
    }

    /// Parses and validates a unified `.lmnkc` container byte stream produced by
    /// [`Self::to_container_bytes`] or [`AttractorContainerCodec::to_bytes`].
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError::TopologyMismatch`] when `topology_crc32` mismatches
    /// [`CORE_TOPOLOGY`] (signaling rebake), or [`LmnkTimelineError::Corrupted`] /
    /// [`LmnkTimelineError::Alignment`] on structural/CRC failure.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LmnkTimelineError> {
        let view = LmnkContainerView::parse(bytes).map_err(|e| map_container_err(bytes, e))?;
        let meta_bytes = view
            .section_bytes(LMNK_SECTION_RECALL_METADATA)
            .ok_or_else(|| {
                LmnkTimelineError::Corrupted("missing RecallMetadata section".to_string())
            })?;
        let (model_fingerprint, frames, seek_index) =
            Self::decode_recall_metadata_payload(meta_bytes)?;

        let logical_pframe_topology =
            if let Some(pframe_bytes) = view.section_bytes(LMNK_SECTION_LOGICAL_PFRAMES) {
                Self::decode_logical_pframes_payload(pframe_bytes)?
            } else {
                Vec::new()
            };

        let persona_circuit = view
            .cognitive_trajectory_ledger()
            .or_else(|| Self::extract_circuit_from_frames(&frames));

        Ok(Self {
            model_fingerprint,
            frames,
            seek_index,
            persona_circuit,
            logical_pframe_topology,
        })
    }
}

impl AttractorContainerCodec for LmnkTimeline {
    fn record_step(
        &mut self,
        ctx: &TimelineStepContext<'_>,
        source: SourceTextRecord,
        metrics: &FrameTransitionMetrics,
    ) -> Result<bool, String> {
        let frame_index = source.frame_index;
        let emit_i = self.frames.is_empty() || ctx.detector.should_emit_keyframe(metrics);
        let fp = Self::geometry_fingerprint(ctx.adapter.geometry());
        if self.frames.is_empty() {
            self.model_fingerprint = fp;
        }
        if emit_i {
            let container_bytes = Self::build_mapped_storage_snapshot(
                ctx.corpus,
                source.timestamp_epoch_sec,
                self.persona_circuit.as_ref(),
            )?;
            let idx = self.frames.len();
            self.seek_index.push(idx);
            self.frames.push(GopTimelineFrame {
                frame_index,
                source,
                kind: GopFrameKind::KeyframeI {
                    container_bytes,
                    recall_policy: *ctx.policy,
                },
            });
            Ok(true)
        } else {
            let prev_q7 = self.latest_phase_q7();
            let target_entry = ctx.corpus.entries().last();
            let curr_q7 = target_entry
                .and_then(|e| e.per_layer_q7.last().copied())
                .unwrap_or([0_i8; MAX_ATTENTION_HEADS]);
            let mut delta_q7 = [0_i8; MAX_ATTENTION_HEADS];
            for h in 0..MAX_ATTENTION_HEADS {
                delta_q7[h] = curr_q7[h].saturating_sub(prev_q7[h]);
            }
            let target_anchor_id = target_entry.map_or(0, |e| e.anchor_id);
            let energy_delta = source.energy;
            self.frames.push(GopTimelineFrame {
                frame_index,
                source,
                kind: GopFrameKind::DeltaP {
                    delta_q7,
                    energy_delta,
                    target_anchor_id,
                },
            });
            Ok(false)
        }
    }

    fn seek_zero_forward(&self, target_frame_index: u64) -> Result<SeekedContainerState, String> {
        let target_pos = self
            .frames
            .iter()
            .position(|f| f.frame_index == target_frame_index)
            .ok_or_else(|| format!("frame_index {target_frame_index} not found"))?;
        let key_pos = self
            .seek_index
            .iter()
            .copied()
            .rfind(|&pos| pos <= target_pos)
            .ok_or_else(|| format!("no preceding I-Frame for target {target_frame_index}"))?;

        let key_frame = &self.frames[key_pos];
        let (container_bytes, recall_policy) = match &key_frame.kind {
            GopFrameKind::KeyframeI {
                container_bytes,
                recall_policy,
            } => (container_bytes, *recall_policy),
            GopFrameKind::DeltaP { .. } => {
                return Err("corrupt seek index pointing to DeltaP".to_string())
            }
        };

        let view = LmnkContainerView::parse(container_bytes)
            .map_err(|e| format!("mapped-storage container validation failed: {e}"))?;
        let anchor_count = view.phase_anchor_count();
        let mut active_phase_q7 = if anchor_count > 0 {
            view.read_phase_anchor(anchor_count - 1)
                .map_or([0_i8; MAX_ATTENTION_HEADS], |c| c.phase_profile)
        } else {
            [0_i8; MAX_ATTENTION_HEADS]
        };
        let mut total_energy: f32 = (0..anchor_count)
            .filter_map(|i| view.read_phase_anchor(i))
            .map(|c| c.base_energy)
            .sum();
        let anchor_ids: Vec<u32> = (0..anchor_count)
            .filter_map(|i| view.read_phase_anchor(i))
            .map(|c| c.anchor_id)
            .collect();

        let mut p_frames_applied = 0_usize;
        for f in &self.frames[(key_pos + 1)..=target_pos] {
            if let GopFrameKind::DeltaP {
                delta_q7,
                energy_delta,
                ..
            } = &f.kind
            {
                for h in 0..MAX_ATTENTION_HEADS {
                    active_phase_q7[h] = active_phase_q7[h].saturating_add(delta_q7[h]);
                }
                total_energy += *energy_delta;
                p_frames_applied += 1;
            }
        }

        Ok(SeekedContainerState {
            target_frame_index,
            keyframe_index: key_frame.frame_index,
            p_frames_applied,
            recall_policy,
            active_phase_q7,
            total_energy,
            anchor_ids,
        })
    }

    fn export_subtitle_track(&self, up_to_frame: Option<u64>) -> Vec<SourceTextRecord> {
        self.frames
            .iter()
            .filter(|f| up_to_frame.is_none_or(|limit| f.frame_index <= limit))
            .map(|f| f.source.clone())
            .collect()
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.to_container_bytes(LMNK_CONTAINER_FLAG_COMPILED_CACHE)
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyEncoder {
        layers: usize,
        rot_len: usize,
    }

    impl SubtitleActivationEncoder for DummyEncoder {
        fn encode_subtitle_activations(
            &mut self,
            record: &SourceTextRecord,
        ) -> Result<Vec<Vec<f32>>, String> {
            let seed = record.text.bytes().fold(0x1337_u64, |acc, b| {
                acc.wrapping_mul(131).wrapping_add(u64::from(b))
            });
            Ok((0..self.layers)
                .map(|l| {
                    (0..self.rot_len)
                        .map(|d| {
                            (((seed ^ ((l as u64) << 6) ^ (d as u64)) % 100) as f32) * 0.01 - 0.5
                        })
                        .collect()
                })
                .collect())
        }
    }

    #[test]
    fn adaptive_gop_cut_detector_delegates_to_spine_transition_controller() {
        let detector = AdaptiveGopCutDetector::default();
        let calm = FrameTransitionMetrics {
            phase_l1_delta: 0.10,
            lacuna_state: LacunaBoundaryState::InsideBasin,
            frames_since_keyframe: 2,
            new_cluster_spawned: false,
        };
        assert!(!detector.should_emit_keyframe(&calm));

        assert!(detector.should_emit_keyframe(&FrameTransitionMetrics {
            lacuna_state: LacunaBoundaryState::LiminalExterior,
            ..calm
        }));
        assert!(detector.should_emit_keyframe(&FrameTransitionMetrics {
            new_cluster_spawned: true,
            ..calm
        }));
        assert!(detector.should_emit_keyframe(&FrameTransitionMetrics {
            phase_l1_delta: 0.35,
            ..calm
        }));
        assert!(!detector.should_emit_keyframe(&FrameTransitionMetrics {
            phase_l1_delta: 0.349,
            ..calm
        }));
        assert!(!detector.should_emit_keyframe(&FrameTransitionMetrics {
            phase_l1_delta: f32::NAN,
            ..calm
        }));

        // Large max_gop_len (> 255) must not wrap or prematurely expire below max_gop_len.
        let long_gop = AdaptiveGopCutDetector {
            max_gop_len: 300,
            phase_delta_cut_thresh: 0.35,
        };
        assert!(!long_gop.should_emit_keyframe(&FrameTransitionMetrics {
            frames_since_keyframe: 299,
            ..calm
        }));
        assert!(long_gop.should_emit_keyframe(&FrameTransitionMetrics {
            frames_since_keyframe: 300,
            ..calm
        }));

        // Zero cut threshold always triggers keyframe on non-negative deltas.
        let zero_thresh = AdaptiveGopCutDetector {
            max_gop_len: 16,
            phase_delta_cut_thresh: 0.0,
        };
        assert!(zero_thresh.should_emit_keyframe(&FrameTransitionMetrics {
            phase_l1_delta: 0.0,
            ..calm
        }));
    }

    #[test]
    fn unified_lmnkc_and_lmnkp_roundtrip_rebake_and_topology_mismatch() {
        let geom_src = RopeGeometry {
            layout: RopeLayout::HalfSplit,
            head_dim: 64,
            rotary_dim: 32,
            num_q_heads: 8,
            num_kv_heads: 4,
            num_layers: 4,
            layer_rope_mask: 0x0F,
        };
        let mut adapter_src = ModelSteeringAdapter::new(geom_src.clone()).expect("src adapter");
        adapter_src.configure_ego_from_preset(LiminalEgoPreset::Medium);
        let policy = RecallPolicy::default();
        let detector = AdaptiveGopCutDetector::default();
        let mut enc_src = DummyEncoder {
            layers: geom_src.num_layers,
            rot_len: geom_src.q_rot_len(),
        };
        let records = vec![
            SourceTextRecord {
                frame_index: 0,
                timestamp_epoch_sec: 1_700_000_000,
                role: AttractorRole::Fact,
                text: "Fact alpha boundary anchor".to_string(),
                energy: 2.0,
            },
            SourceTextRecord {
                frame_index: 1,
                timestamp_epoch_sec: 1_700_000_010,
                role: AttractorRole::Fact,
                text: "Fact beta coherence continuation".to_string(),
                energy: 1.5,
            },
        ];
        let ctx_src = RebakeContext {
            adapter: &adapter_src,
            policy: &policy,
            detector: &detector,
            centroid_layers: &[],
            up_to_frame: None,
        };
        let (mut tl, _) =
            LmnkTimeline::bake_from_subtitle_records(&records, &ctx_src, &mut enc_src)
                .expect("bake");
        tl.set_logical_pframe_topology(adapter_src.export_logical_pframe_topology());

        // 1. Unified .lmnkc container roundtrip & inner LmnkContainerView verification.
        let mut bytes = tl
            .to_container_bytes(LMNK_CONTAINER_FLAG_COMPILED_CACHE)
            .expect("to_container_bytes");
        let view = LmnkContainerView::parse(&bytes).expect("LmnkContainerView::parse");
        assert!(view.section_bytes(LMNK_SECTION_RECALL_METADATA).is_some());
        assert!(view.section_bytes(LMNK_SECTION_LOGICAL_PFRAMES).is_some());
        let decoded = LmnkTimeline::from_bytes(&bytes).expect("from_bytes");
        assert_eq!(decoded, tl);

        // 2. Empty optional sections cleanly omitted (topology = empty, no persona circuit).
        let mut tl_min = tl.clone();
        tl_min.set_logical_pframe_topology(Vec::new());
        let min_bytes = tl_min
            .to_container_bytes(LMNK_CONTAINER_FLAG_COMPILED_CACHE)
            .expect("min container");
        let min_view = LmnkContainerView::parse(&min_bytes).expect("min parse");
        assert!(min_view
            .section_bytes(LMNK_SECTION_LOGICAL_PFRAMES)
            .is_none());
        assert_eq!(LmnkTimeline::from_bytes(&min_bytes).unwrap(), tl_min);

        // 3. .lmnkp 64B-aligned Store ZIP roundtrip.
        let pkg_bytes = tl
            .export_lmnkp_bytes(Some("persona Demo {}"))
            .expect("export_lmnkp_bytes");
        let pkg_view = LmnkPackageView::parse(&pkg_bytes).expect("package parse");
        let compiled_entry = pkg_view
            .require_aligned64_entry(LMNKP_ENTRY_COMPILED)
            .expect("aligned compiled.lmnkc");
        assert_eq!(compiled_entry.data_offset % 64, 0);
        let _ = LmnkContainerView::parse(compiled_entry.payload).expect("inner container parse");
        let from_pkg = LmnkTimeline::import_lmnkp_bytes(&pkg_bytes).expect("import_lmnkp_bytes");
        assert_eq!(from_pkg, tl);

        // 4. Cross-model rebake on geometry mismatch.
        let mut geom_dst = geom_src;
        geom_dst.rotary_dim = 64;
        let adapter_dst = ModelSteeringAdapter::new(geom_dst.clone()).expect("dst adapter");
        let mut enc_dst = DummyEncoder {
            layers: geom_dst.num_layers,
            rot_len: geom_dst.q_rot_len(),
        };
        let ctx_dst = RebakeContext {
            adapter: &adapter_dst,
            policy: &policy,
            detector: &detector,
            centroid_layers: &[],
            up_to_frame: None,
        };
        match from_pkg
            .resume_or_rebake(&ctx_dst, &mut enc_dst, 1)
            .expect("rebake")
        {
            TimelineResumeOutcome::Rebaked {
                timeline: rebaked,
                rebaked_frames,
                ..
            } => {
                assert_eq!(rebaked_frames, 2);
                assert_eq!(
                    rebaked.model_fingerprint(),
                    LmnkTimeline::geometry_fingerprint(&geom_dst)
                );
            }
            TimelineResumeOutcome::FastResumed { .. } => {
                panic!("expected Rebaked on altered rotary_dim")
            }
        }

        // 5. Mutated topology_crc32 with valid recomputed header_crc32 yields TopologyMismatch.
        let mut hdr = LmnkContainerHeader::from_le_bytes(&bytes).expect("hdr");
        hdr.topology_crc32 ^= 0x00FF_00AA;
        let _ = hdr.finalize_crc32();
        bytes[..64].copy_from_slice(&hdr.to_le_bytes());
        assert!(matches!(
            LmnkTimeline::from_bytes(&bytes),
            Err(LmnkTimelineError::TopologyMismatch { .. })
        ));
    }
}
