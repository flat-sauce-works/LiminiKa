# LiminiKa Container & Package Specification (`.lmnkc` / `.lmnkp`, `v0.4.0`)

## 1. Container Stack Overview

LiminiKa separates declarative source definitions, compiled topology-bound binary caches, and portable distribution archives into three distinct formats:

| Format | Extension | Layer | Purpose |
| :--- | :--- | :--- | :--- |
| Declarative DSL | `.lmnk` | `liminika-dsl` | Human-readable `target` / `persona` / `session` definitions ([DSL Spec](./dsl_spec.md)). |
| Compiled Binary Container | `.lmnkc` | `liminika-core` + `liminika-adapter` | 64B-aligned Little-Endian `"LMNK"` container storing runtime snapshots, phase anchors, pointer trails, and dual-track GOP timelines (format `v2`). |
| Portable Package | `.lmnkp` | `liminika-adapter` (`package` + `host/integrity`) | Pure-`std` Store-only (Method 0) PKZIP archive bundling `manifest.json`, optional `source.lmnk`, 64B-aligned `compiled.lmnkc`, and optional SHA-256 / Ed25519 integrity entries. |

---

## 2. Unified `.lmnk` / `.lmnkc` Binary Container Format

Implemented in `liminika-core` (`src/core/src/runtime/storage/`) with `#![no_std]` alignment-safe decoding (`LmnkContainerView::parse`).

### 2.1 Header (`LmnkContainerHeader`, 64 B, `align(64)`)

| Offset | Type | Field | Invariant |
| ---: | :--- | :--- | :--- |
| `0..4` | `u32` | `magic` | `0x4B4E_4D4C` (`"LMNK"` Little-Endian) |
| `4..8` | `u32` | `version` | `LMNK_ABI_VERSION_HEX` (`0x0000_0400`); validated via `is_compatible_abi_version` |
| `8..12` | `u32` | `topology_crc32` | Must match active `CORE_TOPOLOGY.topology_crc32` (`0xfcb10184`), else `LMNK_ERR_TOPOLOGY_MISMATCH` (`-52`) |
| `12..16` | `u32` | `container_flags` | Bitmask of `RUNTIME_SNAPSHOT (1<<0)`, `COMPILED_CACHE (1<<1)`, `PORTABLE_PACKAGE (1<<2)` (`known = 0x07`) |
| `16..24` | `u64` | `total_size` | Total container byte length (`<= 64 MiB`, `LMNK_MAX_CONTAINER_BYTES`) |
| `24..32` | `u64` | `timestamp_epoch_sec` | Creation UNIX timestamp (seconds) |
| `32..36` | `u32` | `section_table_offset` | Fixed at `64` (`LMNK_CONTAINER_HEADER_BYTES`) |
| `36..38` | `u16` | `section_count` | `0..=32` (`LMNK_MAX_CONTAINER_SECTIONS`) |
| `38..40` | `u16` | `header_size` | Fixed at `64` |
| `40..60` | `[u8; 20]` | `reserved_future` | Zero-initialized |
| `60..64` | `u32` | `header_crc32` | IEEE 802.3 CRC32 over header bytes `0..60` |

### 2.2 Section Directory (`LmnkSectionEntry`, 16 B each) & Span Invariants

Each entry at `64 + 16 * i` contains `(kind: u16, flags: u16, offset: u32, length: u32, payload_crc32: u32)`. `LmnkContainerView::parse` enforces fail-closed invariants:

- Checked arithmetic across all offset/length sums.
- Every section `offset` is 64-byte aligned (`offset % 64 == 0`), monotonic, non-overlapping, and within `total_size`.
- Duplicate `kind` identifiers within a single container are unconditionally rejected.
- Every payload matches its IEEE 802.3 `payload_crc32`.

### 2.3 Canonical Section Kind Registry

| Kind ID | Constant | Payload Contents |
| ---: | :--- | :--- |
| `1` | `LMNK_SECTION_RUNTIME_SNAPSHOT` | `448 B` `RuntimeContext` snapshot (`576 B` total single-section container; `SystemSpineState @ 320`, `EnvelopeState @ 352`, `EnvelopeBounds @ 416`). |
| `2` | `LMNK_SECTION_PHASE_ANCHORS` | Array of `128 B` `PhaseAnchorRecord` (`"LNKR"`) entries. |
| `3` | `LMNK_SECTION_POINTER_TRAILS` | Array of `128 B` `PointerTrail` entries. |
| `4` | `LMNK_SECTION_BLOOM_FILTER` | `64 B` `ContrastivePhaseBloomFilter`. |
| `5` | `LMNK_SECTION_LEDGER_SLOT` | `64 B` `LedgerSlot`. |
| `6` | `LMNK_SECTION_CAUSAL_PROVENANCE` | Array of `16 B` `CausalProvenanceReport` entries. |
| `7` | `LMNK_SECTION_LOGICAL_PFRAMES` | Array of `32 B` `LogicalPFrameDescriptor` entries. |
| `8` | `LMNK_SECTION_RECALL_METADATA` | `LmnkTimeline` dual-track metadata & subtitle source track (`v2`). |
| `0xFF01` | `LMNK_SECTION_CONTENT_HASH` | `32 B` SHA-256 canonical transcript digest (verified outside `liminika-core`). |
| `0xFF02` | `LMNK_SECTION_SIGNATURE` | `64 B` Ed25519 signature over `LMNKC-SIGNATURE-V1\0 \|\| content_hash`. |

---

## 3. Dual-Track GOP Timeline (`LmnkTimeline`, Format `v2`)

Implemented in `src/adapter/src/timeline/`:

- **Track 1 (Model-Agnostic Subtitle / Source Track)**: Stores lightweight UTF-8 source entries, roles (`Fact`, `Persona`, `Guard`), and energies inside `LMNK_SECTION_RECALL_METADATA` (`LMNK_TIMELINE_FORMAT_VERSION = 2`), enabling cross-model rebaking whenever model geometry or `CORE_TOPOLOGY.topology_crc32` changes (`LmnkTimelineError::TopologyMismatch`).
- **Track 2 (Compiled GOP Track)**: Stores periodic I-Frame keyframes and incremental P-Frame Q0.7 phase deltas (`LMNK_SECTION_LOGICAL_PFRAMES`), allowing zero-forward historical state reconstruction by seeking to the nearest preceding I-Frame.
- **Adaptive Scene-Cut Detection (`AdaptiveGopCutDetector`)**: Inserts a new I-Frame when `frames_since_keyframe >= max_gop_len` (default `16`), `new_cluster_spawned == true`, `lacuna_state == LacunaBoundaryState::LiminalExterior`, or `phase_l1_delta >= phase_delta_cut_thresh` (default `0.35`).

---

## 4. Portable `.lmnkp` Store ZIP Package

Implemented in `src/adapter/src/package/` using pure Rust `std` (zero third-party archive dependencies):

- **Archive Constraints**:
  - Method 0 (**Store-only**, uncompressed); rejects compression, encryption, data descriptors, multi-disk archives, ZIP64 sentinels, trailing polyglot bytes, duplicate or case-colliding entry names, overlapping/gapped spans, LFH-vs-Central-Directory metadata mismatches, and CRC32 mismatches.
  - Hard limits: `LMNKP_MAX_PACKAGE_BYTES = 64 MiB`, `LMNKP_MAX_ENTRIES = 64`, `LMNKP_MAX_NAME_BYTES = 256`.
- **64-Byte Payload Alignment**: Uses PKZIP extra-field ID `0x4C4D` (`"LM"`) to pad the `compiled.lmnkc` payload offset to a 64-byte boundary inside the archive.
- **Canonical Entries**: `manifest.json`, `source.lmnk` (optional), `compiled.lmnkc` (64B aligned), `integrity.sha256` (`32 B`, optional), `signature.ed25519` (`64 B`, optional).
- **TOCTOU / `SIGBUS` Isolation**: External `.lmnkp` files are read via a single file descriptor with `Read::take(cap + 1)` into an owned `Vec<u8>` buffer (`LmnkPackageOwned::load_from_path`). External `.lmnkp` archives are never OS-`mmap`ed; zero-copy mapping is reserved for app-managed local `.lmnkc` caches.
- **Two-Track Path Sanitizer (`src/adapter/src/host/path_guard.rs`)**:
  - **Track A (`validate_archive_relative_path`)**: Enforces POSIX relative archive paths; rejects traversal (`..`), `\`, `:`, control characters, Windows forbidden chars (`< > : " | ? *`), trailing `.`/space, and Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM0..9`, `LPT0..9`, superscript `¹²³` variants, `CONIN$`, `CONOUT$`).
  - **Track B (`validate_local_output_path`)**: Permits OS drive prefixes while rejecting Win32 device namespaces (`\\.\`), trailing separators, and per-segment reserved/forbidden characters.

---

## 5. Cryptographic Integrity & Signature Boundary

All cryptographic verification lives in `src/adapter/src/host/integrity.rs` (`sha2`, `ed25519-dalek`, `zeroize`, `getrandom`); `#![no_std]` `liminika-core` holds zero cryptographic dependencies.

- **`.lmnkc` Transcript (`0xFF01` / `0xFF02`)**:
  - `0xFF01` stores `SHA-256("LMNKC-CONTENT-HASH-V1\0" || header_invariants || ordered_non_security_sections)`.
  - `0xFF02` stores the Ed25519 signature over `"LMNKC-SIGNATURE-V1\0" || content_hash`, verified via non-malleable `VerifyingKey::verify_strict`.
  - A standalone `0xFF02` without a valid matching `0xFF01` is always rejected.
- **`.lmnkp` Transcript (`integrity.sha256` / `signature.ed25519`)**:
  - Computes domain-separated `"LMNKP-CONTENT-HASH-V1\0"` SHA-256 over all non-integrity entries sorted canonically by UTF-8 name bytes, and signs `"LMNKP-SIGNATURE-V1\0" || content_hash` with Ed25519 `verify_strict`.
- **Fail-Closed Verification Policies (`IntegrityPolicy`)**:
  - `AllowUnsigned`: Accepts unsigned legacy artifacts (`IntegrityTrustLevel::Unsigned`), but still fails closed if any present hash or signature is tampered or malformed.
  - `RequireContentHash`: Requires a verified SHA-256 digest (`ContentHashVerified`).
  - `RequireSignature(VerifyingPublicKey)`: Requires both a verified SHA-256 digest and a strict Ed25519 signature (`SignatureVerified`).
- **Host Pin Floor Preservation**: Restoring `LMNK_SECTION_RUNTIME_SNAPSHOT` unconditionally preserves `HOST_IMMUTABLE_PINNED_COMPUTE_MASK` and host-locked `EnvelopeState` privilege floors even under untrusted snapshots.
- **CLI Commands**: `liminika keygen`, `liminika sign`, `liminika verify`, and `liminika inspect --pubkey <file|hex> [--require-hash]`.
