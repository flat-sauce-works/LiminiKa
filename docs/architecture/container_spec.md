# `.gcso` Container Specification (Dual-Track + PPRC GOP)

## 1. Overview

The `.gcso` container unifies **Zero-Overhead In-Memory Mapped Storage (ZIMMS)** with a **video-codec-inspired Dual-Track + GOP (Group of Pictures)** architecture. It enables:
1. **Instant Zero-Copy Mount**: Same-model sessions memory-map (`mmap`) compiled 4-section ZIMMS keyframes directly into the Hot Path ($C_{\mathrm{alloc}} = 0$).
2. **Zero-Forward Time-Travel Seek**: Historical persona and memory states are reconstructed by seeking to the nearest preceding **I-Frame** and accumulating subsequent **P-Frame** Q7 phase deltas without re-running the base LLM.
3. **Model-Agnostic Portability**: Every frame carries an append-only **Subtitle / Source Track** containing lightweight natural-language text, allowing `EnvProbe` to transcode (re-bake) the entire attractor manifold whenever the underlying LLM changes.

---

## 2. Dual-Track Architecture

```text
+-------------------------------------------------------------------------+
| .gcso Container (Magic: "GCSODUAL", Version: 1, model_fingerprint: u64) |
+-------------------------------------------------------------------------+
| [Track 1: Subtitle / Source Track] (Model-Agnostic Ground Truth)        |
|   - FrameIndex, Timestamp, AttractorRole (Fact/Persona/Guard), Energy   |
|   - UTF-8 Natural Language Text (lightweight, reproducible)             |
+-------------------------------------------------------------------------+
| [Track 2: PPRC Video-Style GOP Track] (Model-Specific Compiled Phases)  |
|   - KeyframeI: 128B-aligned ZIMMS 4-Section Container + RecallPolicy    |
|   - DeltaP:    64-head Q7 phase delta + energy delta + target anchor ID |
+-------------------------------------------------------------------------+
| [Seek Table]                                                            |
|   - Ordered indices of KeyframeI frames for O(1) nearest-keyframe seek  |
+-------------------------------------------------------------------------+
```

---

## 3. 4-Section ZIMMS Keyframe Layout (`GcsoFourSectionView`)

Each `KeyframeI` embeds a validated, 128-byte aligned ZIMMS 4-section container (`src/core/src/storage.rs`):

| Section | Contents | Alignment |
| :--- | :--- | :--- |
| **Section 0a** | `gcso_snapshot_header_t` (Magic, ABI version, IEEE 802.3 CRC32) | 128 B |
| **Section 0b** | `gcso_zimms_descriptor_t` (Container size, mmap flags) | 64 B |
| **Section 1** | `CrystallizedAttractorRecord[]` (128B macro centroids + Q7 phase profiles) | 64 B |
| **Section 2** | `gcso_pointer_trail_t[]` (128B stigmergic pointer trails) | 128 B |
| **Section 3** | Optional `ContrastivePhaseBloomFilter` (64B), `gcso_daes_slot_t` (64B), `CausalProvenanceReport[]` | 64 B |

---

## 4. Adaptive GOP Scene-Cut & `EnvProbe` Integration

Keyframe insertion is abstracted behind the `TimelineCutDetector` trait (`src/adapter/src/lib.rs`). The default `AdaptiveGopCutDetector` emits a new **I-Frame** when any of the following hold:
1. `frames_since_keyframe >= max_gop_len` (periodic checkpoint).
2. `new_cluster_spawned == true` (bottom-up attractor crystallization).
3. `void_state == VoidBoundaryState::CvoidEmptySpace` (topic/manifold transition into ungrounded `cvoid` space).
4. `phase_l1_delta >= phase_delta_cut_thresh` (abrupt phase shift).

When `EnvProbe` detects at startup or runtime that `GcsoDualTrackTimeline::geometry_fingerprint(active_geometry) != container.model_fingerprint()`, the Cold Path exports the **Subtitle / Source Track** (`export_subtitle_track`) and re-compiles it for the active model's geometry.
