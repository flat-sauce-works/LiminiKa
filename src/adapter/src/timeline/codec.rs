// name: src/adapter/src/timeline/codec.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Binary `.lmnkc` / `.lmnkp` serialization, deserialization, and path I/O for [`LmnkTimeline`].

use std::io::Read;
use std::path::Path;

use liminika_core::Pod;

use super::*;

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

pub(super) fn map_container_err(bytes: &[u8], err: LmnkStatus) -> LmnkTimelineError {
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
        const MIN_ENCODED_FRAME_BYTES: usize = 26;
        let min_required_bytes = frame_count
            .checked_mul(MIN_ENCODED_FRAME_BYTES)
            .ok_or_else(|| LmnkTimelineError::Corrupted("frame count byte overflow".to_string()))?;
        if min_required_bytes > cur.remaining() {
            return Err(LmnkTimelineError::Corrupted(format!(
                "frame count {frame_count} requires at least {min_required_bytes} B, only {} B remaining",
                cur.remaining()
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
                    )));
                }
            };
            let energy = cur.read_f32_le()?;
            if !energy.is_finite() {
                return Err(LmnkTimelineError::Corrupted(format!(
                    "non-finite frame energy at index {idx}"
                )));
            }
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
                    if !min_coherence.is_finite()
                        || !boundary_margin.is_finite()
                        || !guard_suppress_margin.is_finite()
                    {
                        return Err(LmnkTimelineError::Corrupted(format!(
                            "non-finite KeyframeI policy float at frame index {idx}"
                        )));
                    }
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
                    if !energy_delta.is_finite() {
                        return Err(LmnkTimelineError::Corrupted(format!(
                            "non-finite DeltaP energy_delta at frame index {idx}"
                        )));
                    }
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
                    )));
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
            out.push(d.intent_delta.boundary_crossing_q7 as u8);
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
                boundary_crossing_q7: *ax.get(1).unwrap_or(&0) as i8,
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
        self.to_container_bytes_with_snapshot(container_flags, None)
    }

    /// Serializes this timeline into a unified v2 `.lmnkc` container while optionally embedding
    /// an `LMNK_SECTION_RUNTIME_SNAPSHOT` payload extracted from a 576-byte snapshot container.
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError`] if the optional snapshot or container boundary rules fail.
    pub fn to_container_bytes_with_snapshot(
        &self,
        container_flags: u32,
        runtime_snapshot_576: Option<&[u8]>,
    ) -> Result<Vec<u8>, LmnkTimelineError> {
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
                && let Ok(view) = LmnkContainerView::parse(container_bytes)
            {
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
        if let Some(ref circuit) = self.persona_circuit {
            let slot = circuit.to_persona_ledger_slot();
            ledger_sec = Some(slot.as_bytes().to_vec());
        }

        let snap_sec = if let Some(snap_bytes) = runtime_snapshot_576 {
            let snap_view = LmnkContainerView::parse(snap_bytes)
                .map_err(|e| map_container_err(snap_bytes, e))?;
            let payload = snap_view
                .section_bytes(LMNK_SECTION_RUNTIME_SNAPSHOT)
                .ok_or_else(|| {
                    LmnkTimelineError::Corrupted(
                        "missing LMNK_SECTION_RUNTIME_SNAPSHOT in snapshot container".to_string(),
                    )
                })?;
            Some(payload.to_vec())
        } else {
            None
        };

        let pframes_sec = if self.logical_pframe_topology.is_empty() {
            None
        } else {
            Some(self.encode_logical_pframes_payload()?)
        };
        let recall_meta_sec = self.encode_recall_metadata_payload()?;

        let mut sections: Vec<LmnkSectionInput<'_>> = Vec::with_capacity(8);
        let mut effective_flags = container_flags;
        if let Some(ref bytes) = snap_sec {
            effective_flags |= LMNK_CONTAINER_FLAG_RUNTIME_SNAPSHOT;
            sections.push(LmnkSectionInput {
                kind: LMNK_SECTION_RUNTIME_SNAPSHOT,
                flags: 0,
                payload: bytes,
            });
        }
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
            write_container_sections_to_bytes(effective_flags, latest_ts, &sections, &mut out)
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
    /// verifying 64B alignment of `compiled.lmnkc`, fail-closed package/container cryptographic
    /// integrity under [`IntegrityPolicy::AllowUnsigned`], and [`LmnkContainerView::parse`].
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError::TopologyMismatch`], [`LmnkTimelineError::Package`], or
    /// [`LmnkTimelineError::Corrupted`].
    pub fn import_lmnkp_bytes(bytes: &[u8]) -> Result<Self, LmnkTimelineError> {
        Self::import_lmnkp_bytes_with_policy(bytes, &IntegrityPolicy::AllowUnsigned)
    }

    /// Validates and imports an [`LmnkTimeline`] from an `.lmnkp` Store ZIP package byte slice
    /// while enforcing an explicit cryptographic [`IntegrityPolicy`].
    ///
    /// The outer `.lmnkp` archive is verified against `policy`, while the embedded `compiled.lmnkc`
    /// is always verified fail-closed under [`IntegrityPolicy::AllowUnsigned`] (and is also
    /// cryptographically bound into the outer `.lmnkp` hash/signature).
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError::TopologyMismatch`], [`LmnkTimelineError::Package`], or
    /// [`LmnkTimelineError::Corrupted`].
    pub fn import_lmnkp_bytes_with_policy(
        bytes: &[u8],
        policy: &IntegrityPolicy,
    ) -> Result<Self, LmnkTimelineError> {
        let pkg = LmnkPackageView::parse(bytes)?;
        let _ = pkg.verify_integrity(policy)?;
        let compiled = pkg.compiled_lmnkc()?;
        Self::from_bytes(compiled)
    }

    /// Writes the serialized unified `.lmnkc` container to `path`, returning bytes written.
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError`] if path validation, serialization, or filesystem write fails.
    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<usize, LmnkTimelineError> {
        let p = path.as_ref();
        validate_local_output_path(p).map_err(|e| {
            LmnkTimelineError::Io(format!("unsafe output path {}: {e}", p.display()))
        })?;
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
    /// Returns [`LmnkTimelineError`] if path validation, package creation, or write fails.
    pub fn save_lmnkp_to_path(
        &self,
        path: impl AsRef<Path>,
        source_lmnk: Option<&str>,
    ) -> Result<usize, LmnkTimelineError> {
        let p = path.as_ref();
        validate_local_output_path(p).map_err(|e| {
            LmnkTimelineError::Io(format!("unsafe output path {}: {e}", p.display()))
        })?;
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
        Self::load_from_path_with_policy(path, &IntegrityPolicy::AllowUnsigned)
    }

    /// Loads and validates an `.lmnkc` container or `.lmnkp` package from `path` under an explicit
    /// cryptographic [`IntegrityPolicy`].
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError`] on I/O, ZIP, integrity/policy, topology mismatch, or
    /// container validation errors.
    pub fn load_from_path_with_policy(
        path: impl AsRef<Path>,
        policy: &IntegrityPolicy,
    ) -> Result<Self, LmnkTimelineError> {
        let p = path.as_ref();
        let file = std::fs::File::open(p).map_err(|e| {
            LmnkTimelineError::Io(format!("failed to open container {}: {e}", p.display()))
        })?;
        let meta_len = file
            .metadata()
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
        let cap_plus_one = (LMNKP_MAX_PACKAGE_BYTES as u64).saturating_add(1);
        let mut bytes = Vec::with_capacity((meta_len as usize).min(LMNKP_MAX_PACKAGE_BYTES));
        file.take(cap_plus_one)
            .read_to_end(&mut bytes)
            .map_err(|e| {
                LmnkTimelineError::Io(format!("failed to read container {}: {e}", p.display()))
            })?;
        if bytes.len() > LMNKP_MAX_PACKAGE_BYTES {
            return Err(LmnkTimelineError::Corrupted(format!(
                "container {} exceeds hard read cap {LMNKP_MAX_PACKAGE_BYTES}",
                p.display()
            )));
        }
        if bytes.starts_with(b"PK\x03\x04")
            || p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("lmnkp"))
        {
            Self::import_lmnkp_bytes_with_policy(&bytes, policy)
        } else {
            Self::from_bytes_with_policy(&bytes, policy)
        }
    }

    /// Parses and validates a unified `.lmnkc` container byte stream produced by
    /// [`Self::to_container_bytes`] or [`AttractorContainerCodec::to_bytes`], enforcing fail-closed
    /// cryptographic verification when `0xFF01`/`0xFF02` sections are present.
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError::TopologyMismatch`] when `topology_crc32` mismatches
    /// [`CORE_TOPOLOGY`] (signaling rebake), or [`LmnkTimelineError::Corrupted`] /
    /// [`LmnkTimelineError::Alignment`] on structural/CRC/integrity failure.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LmnkTimelineError> {
        Self::from_bytes_with_policy(bytes, &IntegrityPolicy::AllowUnsigned)
    }

    /// Parses and validates a unified `.lmnkc` container byte stream under an explicit
    /// cryptographic [`IntegrityPolicy`].
    ///
    /// # Errors
    /// Returns [`LmnkTimelineError::TopologyMismatch`], [`LmnkTimelineError::Alignment`], or
    /// [`LmnkTimelineError::Corrupted`] on structural, integrity, or policy violation.
    pub fn from_bytes_with_policy(
        bytes: &[u8],
        policy: &IntegrityPolicy,
    ) -> Result<Self, LmnkTimelineError> {
        let view = LmnkContainerView::parse(bytes).map_err(|e| map_container_err(bytes, e))?;
        verify_lmnkc_integrity(bytes, policy)
            .map_err(|e| LmnkTimelineError::Corrupted(e.to_string()))?;
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
