// name: src/adapter/src/timeline.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dual-track `.gcso` GOP container timeline (`GcsoDualTrackTimeline`), GOP cut
//! detection, FastResume seeking, and cross-model re-baking codecs.

use super::*;

/// Transition telemetry evaluated by [`TimelineCutDetector`] to choose between a P-Frame delta
/// and a full I-Frame keyframe (Scene Cut) in a `.gcso` GOP timeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameTransitionMetrics {
    /// Mean normalized L1 phase change relative to the previous frame (`0.0..=2.0`).
    pub phase_l1_delta: f32,
    /// Current query boundary classification (`Liminal` triggers scene-cut keyframe).
    pub void_state: VoidBoundaryState,
    /// Number of P-Frames elapsed since the last I-Frame.
    pub frames_since_keyframe: usize,
    /// Whether a new attractor cluster crystallized at this step.
    pub new_cluster_spawned: bool,
}

/// Trait abstracting GOP scene-cut / keyframe insertion policy for `.gcso` timelines.
pub trait TimelineCutDetector {
    /// Returns `true` when the timeline should emit a full I-Frame keyframe instead of a P-Frame.
    fn should_emit_keyframe(&self, metrics: &FrameTransitionMetrics) -> bool;
}

/// Default adaptive GOP detector inserting I-Frames on max GOP length, cluster crystallization,
/// `Liminal` boundary transitions, or large phase shifts.
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
        metrics.frames_since_keyframe >= self.max_gop_len
            || metrics.new_cluster_spawned
            || matches!(metrics.void_state, VoidBoundaryState::LiminalExterior)
            || metrics.phase_l1_delta >= self.phase_delta_cut_thresh
    }
}

/// Lightweight natural-language source record stored in the `.gcso` Subtitle / Source Track for
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

/// Video-codec-inspired GOP frame payload stored in the `.gcso` KV-keyframe Track.
#[derive(Debug, Clone, PartialEq)]
pub enum GopFrameKind {
    /// Full I-Frame snapshot containing a validated 4-section mapped-storage `.gcso` container and active policy.
    KeyframeI {
        /// Serialized 4-section mapped-storage container bytes (`GcsoFourSectionView`).
        zimms_4section: Vec<u8>,
        /// Calibrated recall policy at this keyframe.
        recall_policy: RecallPolicy,
    },
    /// Incremental P-Frame storing Q7 phase motion delta and energy delta relative to the prior frame.
    DeltaP {
        /// Per-head Q7 phase motion delta (`64` heads).
        delta_q7: [gcso_q7_t; MAX_ATTENTION_HEADS],
        /// Incremental energy change.
        energy_delta: f32,
        /// Target anchor ID updated by this step.
        target_anchor_id: u32,
    },
}

/// Single frame in a [`GcsoDualTrackTimeline`] pairing a Subtitle/Source record with a GOP payload.
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
    pub active_phase_q7: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Reconstructed total gravitational energy.
    pub total_energy: f32,
    /// Anchor IDs present in the keyframe's mapped-storage crystal section.
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

/// Trait abstracting dual-track `.gcso` container recording, zero-forward GOP seek, and
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

    /// Serializes the dual-track container into a portable byte stream.
    fn to_bytes(&self) -> Vec<u8>;
}

/// Abstract encoder projecting a [`SourceTextRecord`] from the `.gcso` Subtitle Track into
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

/// Dependencies required to resume or re-bake a [`GcsoDualTrackTimeline`] for a target model.
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

/// Outcome of [`GcsoDualTrackTimeline::resume_or_rebake`] after `EnvProbe` fingerprint checking.
#[derive(Debug, Clone)]
pub enum TimelineResumeOutcome {
    /// Target model geometry matched `model_fingerprint`; state was restored via zero-forward seek.
    FastResumed {
        /// Reconstructed state at the requested target frame.
        seek_state: SeekedContainerState,
    },
    /// Target model geometry differed; Subtitle Track records were re-baked into the new space.
    Rebaked {
        /// Newly re-baked dual-track timeline bound to `target_fingerprint`.
        timeline: GcsoDualTrackTimeline,
        /// Re-baked dynamic attractor corpus aligned with the target model.
        corpus: TextAttractorCorpus,
        /// Reconstructed state at the requested target frame on the re-baked timeline.
        seek_state: SeekedContainerState,
        /// Fingerprint recorded in the source container.
        source_fingerprint: u64,
        /// Fingerprint of the target model's [`RopeGeometry`].
        target_fingerprint: u64,
        /// Number of subtitle records re-baked into the new timeline.
        rebaked_frames: usize,
    },
}

pub(crate) const GCSO_DUAL_MAGIC: &[u8; 8] = b"GCSODUAL";
pub(crate) const GCSO_DUAL_VERSION: u32 = 2;

/// Model-agnostic logical P-Frame node descriptor persisted in `.gcso` v2 containers so
/// portable [`EgoDifferentialNet`] chains can be re-baked to any target model's 64-head Q7 space
/// upon load without modifying base weights.
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
    /// 8-granule Venn trigger bitmask.
    pub venn_trigger_mask: u8,
    /// Scope overlap region selector bitmask (`SCOPE_REGION_*`).
    pub venn_region_selector: u8,
    /// Stigmergic pheromone density in Q0.7 (`0..=127`).
    pub pheromone_q7: u8,
    /// Directed delta scaling gain in Q0.7 (`0..=127`).
    pub gain_q7: u8,
    /// Model-agnostic semantic Ego intent axes transition (`16 B`).
    pub intent_delta: EgoIntentAxes,
}

/// Dual-Track (Subtitle Natural-Language Track + KV-keyframe Video-Style GOP Track) `.gcso` container.
#[derive(Debug, Clone, PartialEq)]
pub struct GcsoDualTrackTimeline {
    model_fingerprint: u64,
    frames: Vec<GopTimelineFrame>,
    seek_index: Vec<usize>,
    persona_circuit: Option<CognitiveTrajectoryLedger>,
    logical_pframe_topology: Vec<LogicalPFrameDescriptor>,
}

impl GcsoDualTrackTimeline {
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

    /// Creates an empty [`GcsoDualTrackTimeline`] bound to `model_fingerprint`.
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

    /// Attaches model-agnostic logical P-Frame topology descriptors for `.gcso` v2 persistence.
    pub fn set_logical_pframe_topology(&mut self, topology: Vec<LogicalPFrameDescriptor>) {
        self.logical_pframe_topology = topology;
    }

    /// Returns persisted model-agnostic logical P-Frame topology descriptors.
    #[must_use]
    pub fn logical_pframe_topology(&self) -> &[LogicalPFrameDescriptor] {
        &self.logical_pframe_topology
    }

    /// Attaches a [`CognitiveTrajectoryLedger`] (stigmergic persona reasoning circuit) to be
    /// persisted inside subsequent I-frame `.gcso` 4-section snapshots.
    pub fn set_persona_circuit(&mut self, circuit: CognitiveTrajectoryLedger) {
        self.persona_circuit = Some(circuit);
    }

    /// Returns the active [`CognitiveTrajectoryLedger`] if set in memory, or extracts the most
    /// recent persisted circuit from the timeline's I-frame `.gcso` 4-section snapshots.
    #[must_use]
    pub fn persona_circuit(&self) -> Option<CognitiveTrajectoryLedger> {
        self.persona_circuit
            .or_else(|| Self::extract_circuit_from_frames(&self.frames))
    }

    /// Composes this container's persona circuit (`self`) with `overlay` using the core
    /// [`CognitiveCircuitPlasticity::compose_persona_circuit`] trait (`blend_q7` in `0..=128`).
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
        if (snapshot.as_ptr() as usize).is_multiple_of(128) {
            return GcsoFourSectionView::from_container_bytes_zero_copy(snapshot)
                .ok()
                .and_then(|v| v.cognitive_trajectory_ledger());
        }
        let mut aligned = vec![0_u8; snapshot.len() + 128];
        let off = aligned.as_ptr().align_offset(128);
        aligned[off..off + snapshot.len()].copy_from_slice(snapshot);
        GcsoFourSectionView::from_container_bytes_zero_copy(&aligned[off..off + snapshot.len()])
            .ok()
            .and_then(|v| v.cognitive_trajectory_ledger())
    }

    fn extract_circuit_from_frames(
        frames: &[GopTimelineFrame],
    ) -> Option<CognitiveTrajectoryLedger> {
        frames.iter().rev().find_map(|f| match &f.kind {
            GopFrameKind::KeyframeI { zimms_4section, .. } => {
                Self::extract_circuit_from_snapshot(zimms_4section)
            }
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

    fn build_zimms_snapshot(
        corpus: &TextAttractorCorpus,
        timestamp_epoch_sec: u64,
        persona_circuit: Option<&CognitiveTrajectoryLedger>,
    ) -> Result<Vec<u8>, String> {
        let mut crystals = Vec::with_capacity(corpus.entries().len().max(1));
        for entry in corpus.entries() {
            let mut rec = CrystallizedAttractorRecord::new_empty(
                entry.anchor_id,
                Self::slot_for_role(entry.role),
                0,
            );
            rec.base_energy = entry.energy;
            rec.phase_gravity_mass = entry.energy;
            rec.sample_count = 1;
            if let Some(first_layer) = entry.per_layer_q7.last() {
                rec.phase_profile = *first_layer;
            }
            rec.checksum_crc32 = 0;
            crystals.push(rec);
        }
        if crystals.is_empty() {
            let empty_rec = CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0);
            crystals.push(empty_rec);
        }
        let trail = gcso_pointer_trail_t::default();
        let trails = [trail];
        let ledger_slot = persona_circuit.map(CognitiveTrajectoryLedger::to_persona_ledger_slot);
        let req_len = GcsoFourSectionView::required_byte_len_with_reports(
            crystals.len(),
            trails.len(),
            false,
            ledger_slot.is_some(),
            0,
        )
        .map_err(|e| format!("required_byte_len error: {e}"))?;
        let mut aligned = vec![0_u8; req_len + 128];
        let off = aligned.as_ptr().align_offset(128);
        let written = GcsoFourSectionView::write_four_section_with_reports_to_bytes(
            &crystals,
            &trails,
            None,
            ledger_slot.as_ref(),
            &[],
            timestamp_epoch_sec,
            &mut aligned[off..off + req_len],
        )
        .map_err(|e| format!("write_four_section error: {e}"))?;
        Ok(aligned[off..off + written].to_vec())
    }

    fn latest_phase_q7(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        if let Some(last) = self.frames.last() {
            if let Ok(state) = self.seek_zero_forward(last.frame_index) {
                return state.active_phase_q7;
            }
        }
        [0_i8; MAX_ATTENTION_HEADS]
    }

    /// Writes the serialized `.gcso` dual-track container to `path`, returning the bytes written.
    ///
    /// # Errors
    /// Returns an error if creating parent directories or writing the file fails.
    pub fn save_to_path(&self, path: impl AsRef<std::path::Path>) -> Result<usize, String> {
        let p = path.as_ref();
        if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "failed to create parent directory {}: {e}",
                    parent.display()
                )
            })?;
        }
        let bytes = self.to_bytes();
        std::fs::write(p, &bytes)
            .map_err(|e| format!("failed to write .gcso container {}: {e}", p.display()))?;
        Ok(bytes.len())
    }

    /// Loads and validates a `.gcso` dual-track container from `path`.
    ///
    /// # Errors
    /// Returns an error if reading the file or parsing the container fails.
    pub fn load_from_path(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let p = path.as_ref();
        let bytes = std::fs::read(p)
            .map_err(|e| format!("failed to read .gcso container {}: {e}", p.display()))?;
        Self::from_bytes(&bytes)
    }

    /// Compiles a sequence of model-agnostic [`SourceTextRecord`] entries into a new
    /// [`GcsoDualTrackTimeline`] and [`TextAttractorCorpus`] for `ctx.adapter.geometry()`
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
                    void_state: if scene_cut {
                        VoidBoundaryState::LiminalExterior
                    } else {
                        VoidBoundaryState::InsideBasin
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
    /// zero-forward GOP seek (`FastResumed`) or automatically re-bakes the Subtitle Track through
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

    /// Deserializes a portable [`GcsoDualTrackTimeline`] byte stream produced by [`Self::to_bytes`].
    ///
    /// # Errors
    /// Returns an error if magic, version, or frame payload lengths are invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut cursor = 0_usize;
        let read_exact = |cur: &mut usize, n: usize| -> Result<&[u8], String> {
            let end = cur.checked_add(n).ok_or_else(|| "overflow".to_string())?;
            if end > bytes.len() {
                return Err("unexpected EOF in GCSO dual-track container".to_string());
            }
            let slice = &bytes[*cur..end];
            *cur = end;
            Ok(slice)
        };
        let magic = read_exact(&mut cursor, 8)?;
        if magic != GCSO_DUAL_MAGIC {
            return Err("invalid GCSO dual-track magic".to_string());
        }
        let ver = u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "ver")?);
        if ver != GCSO_DUAL_VERSION {
            return Err(format!("unsupported GCSO dual-track version: {ver}"));
        }
        let model_fingerprint =
            u64::from_le_bytes(read_exact(&mut cursor, 8)?.try_into().map_err(|_| "fp")?);
        let frame_count =
            u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "cnt")?) as usize;

        let mut frames = Vec::with_capacity(frame_count);
        let mut seek_index = Vec::new();
        for idx in 0..frame_count {
            let frame_index =
                u64::from_le_bytes(read_exact(&mut cursor, 8)?.try_into().map_err(|_| "fidx")?);
            let ts = u64::from_le_bytes(read_exact(&mut cursor, 8)?.try_into().map_err(|_| "ts")?);
            let role = match read_exact(&mut cursor, 1)?[0] {
                0 => AttractorRole::Fact,
                1 => AttractorRole::Persona,
                2 => AttractorRole::Guard,
                r => return Err(format!("unknown AttractorRole byte: {r}")),
            };
            let energy =
                f32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "en")?);
            let text_len =
                u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "tlen")?)
                    as usize;
            let text = std::str::from_utf8(read_exact(&mut cursor, text_len)?)
                .map_err(|e| format!("invalid UTF-8 in source track: {e}"))?
                .to_string();
            let kind_tag = read_exact(&mut cursor, 1)?[0];
            let kind = match kind_tag {
                0 => {
                    let top_k_facts = u32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "tkf")?,
                    ) as usize;
                    let top_k_personas = u32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "tkp")?,
                    ) as usize;
                    let min_coherence = f32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "mc")?,
                    );
                    let boundary_margin = f32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "bm")?,
                    );
                    let guard_suppress_margin = f32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "gsm")?,
                    );
                    let z_len = u32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "zlen")?,
                    ) as usize;
                    let zimms_4section = read_exact(&mut cursor, z_len)?.to_vec();
                    seek_index.push(idx);
                    GopFrameKind::KeyframeI {
                        zimms_4section,
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
                    let q7_bytes = read_exact(&mut cursor, MAX_ATTENTION_HEADS)?;
                    let mut delta_q7 = [0_i8; MAX_ATTENTION_HEADS];
                    for (dst, &b) in delta_q7.iter_mut().zip(q7_bytes.iter()) {
                        *dst = b as i8;
                    }
                    let energy_delta = f32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "ed")?,
                    );
                    let target_anchor_id = u32::from_le_bytes(
                        read_exact(&mut cursor, 4)?.try_into().map_err(|_| "aid")?,
                    );
                    GopFrameKind::DeltaP {
                        delta_q7,
                        energy_delta,
                        target_anchor_id,
                    }
                }
                other => return Err(format!("unknown GopFrameKind tag: {other}")),
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
        let persona_circuit = Self::extract_circuit_from_frames(&frames);
        let topo_count =
            u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "tcnt")?)
                as usize;
        if topo_count > 64 {
            return Err(format!(
                "excessive logical P-Frame topology count: {topo_count}"
            ));
        }
        let mut logical_pframe_topology = Vec::with_capacity(topo_count);
        for _ in 0..topo_count {
            let node_id =
                u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "nid")?);
            let anchor_ref_id =
                u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "ari")?);
            let next_hop_idx =
                u16::from_le_bytes(read_exact(&mut cursor, 2)?.try_into().map_err(|_| "nhi")?);
            let hop_depth = read_exact(&mut cursor, 1)?[0];
            let venn_trigger_mask = read_exact(&mut cursor, 1)?[0];
            let venn_region_selector = read_exact(&mut cursor, 1)?[0];
            let pheromone_q7 = read_exact(&mut cursor, 1)?[0];
            let gain_q7 = read_exact(&mut cursor, 1)?[0];
            let ax = read_exact(&mut cursor, 12)?;
            let sig_hash =
                u32::from_le_bytes(read_exact(&mut cursor, 4)?.try_into().map_err(|_| "sh")?);
            let intent_delta = EgoIntentAxes {
                autonomy_q7: ax[0] as i8,
                transgression_q7: ax[1] as i8,
                devotion_q7: ax[2] as i8,
                logic_q7: ax[3] as i8,
                caution_q7: ax[4] as i8,
                defiance_q7: ax[5] as i8,
                synergy_q7: ax[6] as i8,
                vitality_q7: ax[7] as i8,
                preferred_slot_mask: ax[8],
                active_granule_bias_mask: ax[9],
                confidence_q7: ax[10],
                flags: ax[11],
                signature_hash: sig_hash,
            };
            logical_pframe_topology.push(LogicalPFrameDescriptor {
                node_id,
                anchor_ref_id,
                next_hop_idx,
                hop_depth,
                venn_trigger_mask,
                venn_region_selector,
                pheromone_q7,
                gain_q7,
                intent_delta,
            });
        }
        Ok(Self {
            model_fingerprint,
            frames,
            seek_index,
            persona_circuit,
            logical_pframe_topology,
        })
    }
}

impl AttractorContainerCodec for GcsoDualTrackTimeline {
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
            let zimms_4section = Self::build_zimms_snapshot(
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
                    zimms_4section,
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
                zimms_4section,
                recall_policy,
            } => (zimms_4section, *recall_policy),
            GopFrameKind::DeltaP { .. } => {
                return Err("corrupt seek index pointing to DeltaP".to_string())
            }
        };

        let mut aligned = vec![0_u8; container_bytes.len() + 128];
        let off = aligned.as_ptr().align_offset(128);
        aligned[off..off + container_bytes.len()].copy_from_slice(container_bytes);
        let view = GcsoFourSectionView::from_container_bytes_zero_copy(
            &aligned[off..off + container_bytes.len()],
        )
        .map_err(|e| format!("mapped-storage zero-copy view validation failed: {e}"))?;

        let mut active_phase_q7 = view
            .crystals
            .last()
            .map_or([0_i8; MAX_ATTENTION_HEADS], |c| c.phase_profile);
        let mut total_energy: f32 = view.crystals.iter().map(|c| c.base_energy).sum();
        let anchor_ids: Vec<u32> = view.crystals.iter().map(|c| c.anchor_id).collect();

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
        let mut out = Vec::new();
        out.extend_from_slice(GCSO_DUAL_MAGIC);
        out.extend_from_slice(&GCSO_DUAL_VERSION.to_le_bytes());
        out.extend_from_slice(&self.model_fingerprint.to_le_bytes());
        out.extend_from_slice(&(self.frames.len() as u32).to_le_bytes());
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
            out.extend_from_slice(&(tb.len() as u32).to_le_bytes());
            out.extend_from_slice(tb);
            match &f.kind {
                GopFrameKind::KeyframeI {
                    zimms_4section,
                    recall_policy,
                } => {
                    out.push(0_u8);
                    out.extend_from_slice(&(recall_policy.top_k_facts as u32).to_le_bytes());
                    out.extend_from_slice(&(recall_policy.top_k_personas as u32).to_le_bytes());
                    out.extend_from_slice(&recall_policy.min_coherence.to_le_bytes());
                    out.extend_from_slice(&recall_policy.boundary_margin.to_le_bytes());
                    out.extend_from_slice(&recall_policy.guard_suppress_margin.to_le_bytes());
                    out.extend_from_slice(&(zimms_4section.len() as u32).to_le_bytes());
                    out.extend_from_slice(zimms_4section);
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
        out.extend_from_slice(&(self.logical_pframe_topology.len() as u32).to_le_bytes());
        for d in &self.logical_pframe_topology {
            out.extend_from_slice(&d.node_id.to_le_bytes());
            out.extend_from_slice(&d.anchor_ref_id.to_le_bytes());
            out.extend_from_slice(&d.next_hop_idx.to_le_bytes());
            out.push(d.hop_depth);
            out.push(d.venn_trigger_mask);
            out.push(d.venn_region_selector);
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
        out
    }
}
