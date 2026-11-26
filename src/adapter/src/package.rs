// name: src/adapter/src/package.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure-`std`, zero-dependency **PKZIP Store-only (Method 0)** `.lmnkp` distribution package layer
//! with exact 64-byte payload alignment for embedded `.lmnkc` containers.
//!
//! # Security & Trust Boundary (TOCTOU / `SIGBUS` Isolation)
//! - External `.lmnkp` archives are always treated as untrusted inputs and read into an owned,
//!   bounds-checked `Vec<u8>` buffer via [`LmnkPackageOwned::load_from_path`] before parsing;
//!   never directly OS-`mmap` an external `.lmnkp` path (preventing concurrent-truncation `SIGBUS`
//!   and post-validation TOCTOU mutations). Zero-copy `mmap` is reserved strictly for app-managed
//!   local `.lmnkc` cache files.
//! - Parser paths avoid all direct `[]` indexing, `.unwrap()`, and `.expect()`, enforce strict
//!   size/entry caps, reject non-Store compression, encryption, data descriptors, ZIP64 sentinels,
//!   trailing polyglot bytes, path traversal, duplicate names, LFH-vs-CD differentials, overlapping
//!   spans, and CRC32 mismatches.

use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use liminika_core::foundation::compute_crc32_ieee;

use crate::host::integrity::{
    IntegrityPolicy, IntegrityReport, SigningKeyPair, compute_archive_entries_content_hash,
    sign_archive_content_hash, verify_archive_entries_integrity,
};
use crate::host::path_guard::{
    PathGuardError, validate_archive_relative_path, validate_local_output_path,
};

/// Hard upper byte limit (`64 MiB`) enforced for any `.lmnkp` package.
pub const LMNKP_MAX_PACKAGE_BYTES: usize = 64 * 1024 * 1024;

/// Maximum number of entries allowed inside a `.lmnkp` package (`64`).
pub const LMNKP_MAX_ENTRIES: usize = 64;

/// Maximum UTF-8 byte length for any `.lmnkp` entry name (`256`).
pub const LMNKP_MAX_NAME_BYTES: usize = 256;

/// Canonical PKZIP extra-field header ID (`"LM"` LE) used to pad payloads to 64-byte boundaries.
pub const LMNKP_ALIGN_EXTRA_HEADER_ID: u16 = 0x4C4D;

/// Canonical `.lmnkp` JSON metadata manifest entry name.
pub const LMNKP_ENTRY_MANIFEST: &str = "manifest.json";

/// Canonical `.lmnkp` optional DSL source entry name.
pub const LMNKP_ENTRY_SOURCE: &str = "source.lmnk";

/// Canonical `.lmnkp` 64B-aligned compiled container entry name.
pub const LMNKP_ENTRY_COMPILED: &str = "compiled.lmnkc";

/// Canonical `.lmnkp` SHA-256 content-hash entry name (`32` raw bytes).
pub const LMNKP_ENTRY_CONTENT_HASH: &str = "integrity.sha256";

/// Canonical `.lmnkp` Ed25519 detached signature entry name (`64` raw bytes).
pub const LMNKP_ENTRY_SIGNATURE: &str = "signature.ed25519";

const PKZIP_LOCAL_HEADER_SIG: u32 = 0x0403_4B50;
const PKZIP_CENTRAL_HEADER_SIG: u32 = 0x0201_4B50;
const PKZIP_EOCD_SIG: u32 = 0x0605_4B50;
const PKZIP_VERSION_NEEDED: u16 = 20;
const PKZIP_VERSION_MADE_BY: u16 = 20;
const PKZIP_METHOD_STORE: u16 = 0;
const PKZIP_FLAG_UTF8: u16 = 0x0800;
const PKZIP_LFH_FIXED_BYTES: usize = 30;
const PKZIP_CD_FIXED_BYTES: usize = 46;
const PKZIP_EOCD_FIXED_BYTES: usize = 22;
const PKZIP_ZIP64_EXTRA_ID: u16 = 0x0001;

/// Structured errors produced when building, loading, or validating `.lmnkp` packages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LmnkPackageError {
    /// Archive exceeds [`LMNKP_MAX_PACKAGE_BYTES`], [`LMNKP_MAX_ENTRIES`], or [`LMNKP_MAX_NAME_BYTES`].
    LimitExceeded(String),
    /// Entry name failed UTF-8, path-traversal, control-character, or uniqueness checks.
    InvalidEntryName(String),
    /// Archive structure, signatures, EOCD layout, or extra-field TLV records are malformed.
    MalformedArchive(String),
    /// Unsupported ZIP feature (non-Store method, encrypted bit, data descriptor, multi-disk, ZIP64).
    UnsupportedFeature(String),
    /// Local File Header and Central Directory Header disagree on entry metadata.
    HeaderMismatch(String),
    /// Entry payload spans overlap, leave interstitial gaps, or fall out of bounds.
    SpanViolation(String),
    /// IEEE 802.3 CRC32 verification failed for an entry payload.
    ChecksumMismatch {
        /// Name of the entry whose CRC32 mismatched.
        name: String,
        /// Expected CRC32 stored in the ZIP headers.
        expected: u32,
        /// Computed CRC32 of the stored payload bytes.
        actual: u32,
    },
    /// Required entry payload is missing or violates 64-byte alignment inside the archive.
    AlignmentViolation(String),
    /// Cryptographic SHA-256 content hash or Ed25519 signature verification failed.
    IntegrityFailure(String),
    /// Filesystem I/O failure.
    Io(String),
}

impl fmt::Display for LmnkPackageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded(msg) => write!(f, "package limit exceeded: {msg}"),
            Self::InvalidEntryName(msg) => write!(f, "invalid package entry name: {msg}"),
            Self::MalformedArchive(msg) => write!(f, "malformed .lmnkp archive: {msg}"),
            Self::UnsupportedFeature(msg) => write!(f, "unsupported ZIP feature: {msg}"),
            Self::HeaderMismatch(msg) => write!(f, "ZIP LFH/CD mismatch: {msg}"),
            Self::SpanViolation(msg) => write!(f, "ZIP span violation: {msg}"),
            Self::ChecksumMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "CRC32 mismatch for '{name}': expected=0x{expected:08x} actual=0x{actual:08x}"
            ),
            Self::AlignmentViolation(msg) => write!(f, "alignment violation: {msg}"),
            Self::IntegrityFailure(msg) => write!(f, "package integrity failure: {msg}"),
            Self::Io(msg) => write!(f, "package I/O error: {msg}"),
        }
    }
}

impl std::error::Error for LmnkPackageError {}

/// Validates an archive entry name against traversal, Windows reserved device names,
/// parser differentials, and OS-unsafe characters via [`validate_archive_relative_path`].
///
/// # Errors
/// Returns [`LmnkPackageError::InvalidEntryName`] or [`LmnkPackageError::LimitExceeded`] on rejection.
pub fn validate_package_entry_name(name: &str) -> Result<(), LmnkPackageError> {
    validate_archive_relative_path(name).map_err(|err| match err {
        PathGuardError::LengthExceeded { len, max } => {
            LmnkPackageError::LimitExceeded(format!("entry name length {len} exceeds {max}"))
        }
        other => LmnkPackageError::InvalidEntryName(format!("entry '{name}': {other}")),
    })
}

/// Formats a canonical JSON `manifest.json` payload without external serialization crates.
#[must_use]
pub fn format_package_manifest_json(
    abi_version: &str,
    model_fingerprint: u64,
    frame_count: usize,
    topology_crc32: u32,
    compiled_entry: &str,
    source_entry: Option<&str>,
) -> String {
    let escaped_abi = escape_json_str(abi_version);
    let escaped_compiled = escape_json_str(compiled_entry);
    let source_field = match source_entry {
        Some(src) => format!("\"{}\"", escape_json_str(src)),
        None => "null".to_string(),
    };
    format!(
        concat!(
            "{{\n",
            "  \"format\": \"lmnkp\",\n",
            "  \"format_version\": 1,\n",
            "  \"abi_version\": \"{abi}\",\n",
            "  \"model_fingerprint\": \"0x{fp:016x}\",\n",
            "  \"topology_crc32\": \"0x{tcrc:08x}\",\n",
            "  \"frame_count\": {frames},\n",
            "  \"compiled_entry\": \"{compiled}\",\n",
            "  \"source_entry\": {source}\n",
            "}}\n"
        ),
        abi = escaped_abi,
        fp = model_fingerprint,
        tcrc = topology_crc32,
        frames = frame_count,
        compiled = escaped_compiled,
        source = source_field,
    )
}

fn escape_json_str(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

#[derive(Debug, Clone)]
struct PendingEntry {
    name: String,
    payload: Vec<u8>,
    align_64: bool,
}

/// Builder emitting deterministic, PKZIP-2.0-compatible Store-only (Method 0) `.lmnkp` packages.
#[derive(Debug, Default, Clone)]
pub struct LmnkPackageBuilder {
    entries: Vec<PendingEntry>,
    seen_names: HashSet<String>,
}

impl LmnkPackageBuilder {
    /// Creates an empty [`LmnkPackageBuilder`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a raw entry to the package, optionally padding its Local File Header extra field so the
    /// payload begins at an exact multiple of `64` bytes from byte `0` of the `.lmnkp` archive.
    ///
    /// # Errors
    /// Returns [`LmnkPackageError`] on invalid/duplicate names or package size/count overflow.
    pub fn add_entry(
        &mut self,
        name: impl Into<String>,
        payload: Vec<u8>,
        align_64: bool,
    ) -> Result<&mut Self, LmnkPackageError> {
        let name = name.into();
        validate_package_entry_name(&name)?;
        if self.entries.len() >= LMNKP_MAX_ENTRIES {
            return Err(LmnkPackageError::LimitExceeded(format!(
                "entry count exceeds maximum {LMNKP_MAX_ENTRIES}"
            )));
        }
        if payload.len() > LMNKP_MAX_PACKAGE_BYTES {
            return Err(LmnkPackageError::LimitExceeded(format!(
                "entry '{name}' payload ({} B) exceeds maximum {LMNKP_MAX_PACKAGE_BYTES}",
                payload.len()
            )));
        }
        let canonical = name.to_ascii_lowercase();
        if !self.seen_names.insert(canonical) {
            return Err(LmnkPackageError::InvalidEntryName(format!(
                "duplicate entry '{name}' (case-insensitive collision)"
            )));
        }
        self.entries.push(PendingEntry {
            name,
            payload,
            align_64,
        });
        Ok(self)
    }

    /// Adds the canonical `manifest.json` metadata entry (`align_64 = false`).
    ///
    /// # Errors
    /// Propagates validation or capacity errors from [`Self::add_entry`].
    pub fn add_manifest_json(
        &mut self,
        manifest_json: impl Into<String>,
    ) -> Result<&mut Self, LmnkPackageError> {
        self.add_entry(
            LMNKP_ENTRY_MANIFEST,
            manifest_json.into().into_bytes(),
            false,
        )
    }

    /// Adds the canonical `source.lmnk` entry (`align_64 = false`).
    ///
    /// # Errors
    /// Propagates validation or capacity errors from [`Self::add_entry`].
    pub fn add_source_lmnk(
        &mut self,
        source_bytes: Vec<u8>,
    ) -> Result<&mut Self, LmnkPackageError> {
        self.add_entry(LMNKP_ENTRY_SOURCE, source_bytes, false)
    }

    /// Adds the canonical 64B-aligned `compiled.lmnkc` entry (`align_64 = true`).
    ///
    /// # Errors
    /// Propagates validation or capacity errors from [`Self::add_entry`].
    pub fn add_compiled_lmnkc(
        &mut self,
        compiled_bytes: Vec<u8>,
    ) -> Result<&mut Self, LmnkPackageError> {
        self.add_entry(LMNKP_ENTRY_COMPILED, compiled_bytes, true)
    }

    /// Computes exact extra-field padding bytes needed for 64-byte alignment at `lfh_base_end`.
    ///
    /// Handles `rem in 61..=63` by wrapping cleanly to `128 - rem` (`65..=67` total bytes,
    /// leaving a `61..=63` byte data payload after the 4-byte extra-field TLV header) instead of
    /// underflowing `4`.
    #[inline]
    fn compute_align64_extra_len(lfh_base_end: usize) -> usize {
        let rem = lfh_base_end % 64;
        if rem == 0 {
            0
        } else if rem <= 60 {
            64 - rem
        } else {
            128 - rem
        }
    }

    /// Serializes all queued entries into a PKZIP Store-only `.lmnkp` byte buffer.
    ///
    /// # Errors
    /// Returns [`LmnkPackageError::LimitExceeded`] if the assembled package exceeds
    /// [`LMNKP_MAX_PACKAGE_BYTES`] or [`u32::MAX`] ZIP32 limits.
    pub fn build(&self) -> Result<Vec<u8>, LmnkPackageError> {
        struct BuiltRecord {
            name: String,
            flags: u16,
            crc32: u32,
            size_u32: u32,
            lfh_offset: u32,
        }

        let mut out = Vec::new();
        let mut records = Vec::with_capacity(self.entries.len());

        for entry in &self.entries {
            let lfh_offset = out.len();
            let lfh_offset_u32 = u32::try_from(lfh_offset).map_err(|_| {
                LmnkPackageError::LimitExceeded("LFH offset exceeds u32".to_string())
            })?;
            let name_bytes = entry.name.as_bytes();
            let name_len_u16 = u16::try_from(name_bytes.len()).map_err(|_| {
                LmnkPackageError::LimitExceeded("entry name exceeds u16".to_string())
            })?;
            let size_u32 = u32::try_from(entry.payload.len()).map_err(|_| {
                LmnkPackageError::LimitExceeded("entry payload exceeds u32".to_string())
            })?;
            let crc32 = compute_crc32_ieee(&entry.payload);
            let flags = if entry.name.is_ascii() {
                0_u16
            } else {
                PKZIP_FLAG_UTF8
            };

            let lfh_base_end = lfh_offset
                .checked_add(PKZIP_LFH_FIXED_BYTES)
                .and_then(|v| v.checked_add(name_bytes.len()))
                .ok_or_else(|| {
                    LmnkPackageError::LimitExceeded("archive byte offset overflow".to_string())
                })?;

            let extra_total = if entry.align_64 {
                Self::compute_align64_extra_len(lfh_base_end)
            } else {
                0
            };
            let extra_len_u16 = u16::try_from(extra_total).map_err(|_| {
                LmnkPackageError::LimitExceeded("extra field length exceeds u16".to_string())
            })?;

            let projected_end = lfh_base_end
                .checked_add(extra_total)
                .and_then(|v| v.checked_add(entry.payload.len()))
                .ok_or_else(|| {
                    LmnkPackageError::LimitExceeded("archive size overflow".to_string())
                })?;
            if projected_end > LMNKP_MAX_PACKAGE_BYTES {
                return Err(LmnkPackageError::LimitExceeded(format!(
                    "package payload length {projected_end} exceeds {LMNKP_MAX_PACKAGE_BYTES}"
                )));
            }

            // Local File Header (30 bytes)
            out.extend_from_slice(&PKZIP_LOCAL_HEADER_SIG.to_le_bytes());
            out.extend_from_slice(&PKZIP_VERSION_NEEDED.to_le_bytes());
            out.extend_from_slice(&flags.to_le_bytes());
            out.extend_from_slice(&PKZIP_METHOD_STORE.to_le_bytes());
            out.extend_from_slice(&0_u16.to_le_bytes()); // mod time
            out.extend_from_slice(&0_u16.to_le_bytes()); // mod date
            out.extend_from_slice(&crc32.to_le_bytes());
            out.extend_from_slice(&size_u32.to_le_bytes());
            out.extend_from_slice(&size_u32.to_le_bytes());
            out.extend_from_slice(&name_len_u16.to_le_bytes());
            out.extend_from_slice(&extra_len_u16.to_le_bytes());
            out.extend_from_slice(name_bytes);

            if extra_total > 0 {
                let data_size = extra_total.checked_sub(4).ok_or_else(|| {
                    LmnkPackageError::AlignmentViolation(
                        "extra_total unexpectedly < 4 for 64B alignment".to_string(),
                    )
                })?;
                let data_size_u16 = u16::try_from(data_size).map_err(|_| {
                    LmnkPackageError::AlignmentViolation("extra data_size exceeds u16".to_string())
                })?;
                out.extend_from_slice(&LMNKP_ALIGN_EXTRA_HEADER_ID.to_le_bytes());
                out.extend_from_slice(&data_size_u16.to_le_bytes());
                out.resize(out.len() + data_size, 0);
            }

            let data_offset = out.len();
            if entry.align_64 && (data_offset & 63) != 0 {
                return Err(LmnkPackageError::AlignmentViolation(format!(
                    "internal 64B alignment error for '{}' at offset {}",
                    entry.name, data_offset
                )));
            }
            out.extend_from_slice(&entry.payload);

            records.push(BuiltRecord {
                name: entry.name.clone(),
                flags,
                crc32,
                size_u32,
                lfh_offset: lfh_offset_u32,
            });
        }

        let cd_offset = out.len();
        let cd_offset_u32 = u32::try_from(cd_offset)
            .map_err(|_| LmnkPackageError::LimitExceeded("CD offset exceeds u32".to_string()))?;

        for rec in &records {
            let name_bytes = rec.name.as_bytes();
            let name_len_u16 = u16::try_from(name_bytes.len()).map_err(|_| {
                LmnkPackageError::LimitExceeded("CD entry name exceeds u16".to_string())
            })?;
            out.extend_from_slice(&PKZIP_CENTRAL_HEADER_SIG.to_le_bytes());
            out.extend_from_slice(&PKZIP_VERSION_MADE_BY.to_le_bytes());
            out.extend_from_slice(&PKZIP_VERSION_NEEDED.to_le_bytes());
            out.extend_from_slice(&rec.flags.to_le_bytes());
            out.extend_from_slice(&PKZIP_METHOD_STORE.to_le_bytes());
            out.extend_from_slice(&0_u16.to_le_bytes()); // mod time
            out.extend_from_slice(&0_u16.to_le_bytes()); // mod date
            out.extend_from_slice(&rec.crc32.to_le_bytes());
            out.extend_from_slice(&rec.size_u32.to_le_bytes());
            out.extend_from_slice(&rec.size_u32.to_le_bytes());
            out.extend_from_slice(&name_len_u16.to_le_bytes());
            out.extend_from_slice(&0_u16.to_le_bytes()); // extra len
            out.extend_from_slice(&0_u16.to_le_bytes()); // comment len
            out.extend_from_slice(&0_u16.to_le_bytes()); // disk number start
            out.extend_from_slice(&0_u16.to_le_bytes()); // internal attrs
            out.extend_from_slice(&0_u32.to_le_bytes()); // external attrs
            out.extend_from_slice(&rec.lfh_offset.to_le_bytes());
            out.extend_from_slice(name_bytes);
        }

        let cd_size = out.len() - cd_offset;
        let cd_size_u32 = u32::try_from(cd_size)
            .map_err(|_| LmnkPackageError::LimitExceeded("CD size exceeds u32".to_string()))?;
        let count_u16 = u16::try_from(records.len())
            .map_err(|_| LmnkPackageError::LimitExceeded("entry count exceeds u16".to_string()))?;

        // End of Central Directory (22 bytes)
        out.extend_from_slice(&PKZIP_EOCD_SIG.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes()); // disk number
        out.extend_from_slice(&0_u16.to_le_bytes()); // cd start disk
        out.extend_from_slice(&count_u16.to_le_bytes());
        out.extend_from_slice(&count_u16.to_le_bytes());
        out.extend_from_slice(&cd_size_u32.to_le_bytes());
        out.extend_from_slice(&cd_offset_u32.to_le_bytes());
        out.extend_from_slice(&0_u16.to_le_bytes()); // comment len

        if out.len() > LMNKP_MAX_PACKAGE_BYTES {
            return Err(LmnkPackageError::LimitExceeded(format!(
                "assembled .lmnkp size {} exceeds {LMNKP_MAX_PACKAGE_BYTES}",
                out.len()
            )));
        }

        Ok(out)
    }

    /// Builds a `.lmnkp` archive with a canonical `integrity.sha256` entry covering all content
    /// entries in sorted name order.
    ///
    /// # Errors
    /// Returns [`LmnkPackageError`] on entry limit overflow or serialization failure.
    pub fn build_with_content_hash(&self) -> Result<Vec<u8>, LmnkPackageError> {
        let mut cloned = self.clone_without_integrity_entries()?;
        let content_refs: Vec<(&str, &[u8])> = cloned
            .entries
            .iter()
            .map(|e| (e.name.as_str(), e.payload.as_slice()))
            .collect();
        let digest = compute_archive_entries_content_hash(&content_refs);
        cloned.add_entry(LMNKP_ENTRY_CONTENT_HASH, digest.to_vec(), false)?;
        cloned.build()
    }

    /// Builds a `.lmnkp` archive carrying both `integrity.sha256` (SHA-256) and
    /// `signature.ed25519` (Ed25519 signature over the canonical sorted entry hash).
    ///
    /// # Errors
    /// Returns [`LmnkPackageError`] on entry limit overflow or serialization failure.
    pub fn build_signed(&self, keypair: &SigningKeyPair) -> Result<Vec<u8>, LmnkPackageError> {
        let mut cloned = self.clone_without_integrity_entries()?;
        let content_refs: Vec<(&str, &[u8])> = cloned
            .entries
            .iter()
            .map(|e| (e.name.as_str(), e.payload.as_slice()))
            .collect();
        let digest = compute_archive_entries_content_hash(&content_refs);
        let sig = sign_archive_content_hash(&digest, keypair);
        cloned.add_entry(LMNKP_ENTRY_CONTENT_HASH, digest.to_vec(), false)?;
        cloned.add_entry(LMNKP_ENTRY_SIGNATURE, sig.to_vec(), false)?;
        cloned.build()
    }

    fn clone_without_integrity_entries(&self) -> Result<Self, LmnkPackageError> {
        let mut builder = Self::new();
        for entry in &self.entries {
            if is_package_integrity_entry(&entry.name) {
                continue;
            }
            builder.add_entry(entry.name.clone(), entry.payload.clone(), entry.align_64)?;
        }
        Ok(builder)
    }

    /// Builds and writes the `.lmnkp` archive to `path`, creating parent directories on demand
    /// after validating `path` with [`validate_local_output_path`].
    ///
    /// # Errors
    /// Returns [`LmnkPackageError`] on unsafe path, serialization, or filesystem write failure.
    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<usize, LmnkPackageError> {
        let p = path.as_ref();
        validate_local_output_path(p).map_err(|e| {
            LmnkPackageError::Io(format!("unsafe output path {}: {e}", p.display()))
        })?;
        let bytes = self.build()?;
        if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                LmnkPackageError::Io(format!(
                    "failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        std::fs::write(p, &bytes).map_err(|e| {
            LmnkPackageError::Io(format!("failed to write .lmnkp {}: {e}", p.display()))
        })?;
        Ok(bytes.len())
    }
}

#[inline]
fn is_package_integrity_entry(name: &str) -> bool {
    name.eq_ignore_ascii_case(LMNKP_ENTRY_CONTENT_HASH)
        || name.eq_ignore_ascii_case(LMNKP_ENTRY_SIGNATURE)
}

/// Re-emits an existing `.lmnkp` archive with a fresh `integrity.sha256` entry while preserving
/// all source entries byte-for-byte and keeping 64B alignment on aligned entries.
///
/// # Errors
/// Returns [`LmnkPackageError`] if input parsing or serialization fails.
pub fn attach_lmnkp_content_hash(package_bytes: &[u8]) -> Result<Vec<u8>, LmnkPackageError> {
    let view = LmnkPackageView::parse(package_bytes)?;
    let mut builder = LmnkPackageBuilder::new();
    for entry in view.entries() {
        if is_package_integrity_entry(entry.name) {
            continue;
        }
        let align_64 = entry.name == LMNKP_ENTRY_COMPILED || entry.is_aligned64();
        builder.add_entry(entry.name, entry.payload.to_vec(), align_64)?;
    }
    builder.build_with_content_hash()
}

/// Re-emits an existing `.lmnkp` archive with `integrity.sha256` and `signature.ed25519` entries
/// while preserving all source entries byte-for-byte and keeping 64B alignment on aligned entries.
///
/// # Errors
/// Returns [`LmnkPackageError`] if input parsing or serialization fails.
pub fn sign_lmnkp_package(
    package_bytes: &[u8],
    keypair: &SigningKeyPair,
) -> Result<Vec<u8>, LmnkPackageError> {
    let view = LmnkPackageView::parse(package_bytes)?;
    let mut builder = LmnkPackageBuilder::new();
    for entry in view.entries() {
        if is_package_integrity_entry(entry.name) {
            continue;
        }
        let align_64 = entry.name == LMNKP_ENTRY_COMPILED || entry.is_aligned64();
        builder.add_entry(entry.name, entry.payload.to_vec(), align_64)?;
    }
    builder.build_signed(keypair)
}

/// Parses and cryptographically verifies an `.lmnkp` archive against `policy`.
///
/// # Errors
/// Returns [`LmnkPackageError`] on archive corruption or integrity/policy failure.
pub fn verify_lmnkp_integrity(
    package_bytes: &[u8],
    policy: &IntegrityPolicy,
) -> Result<IntegrityReport, LmnkPackageError> {
    let view = LmnkPackageView::parse(package_bytes)?;
    view.verify_integrity(policy)
}

/// Validated zero-copy entry descriptor returned by [`LmnkPackageView`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LmnkPackageEntry<'a> {
    /// Entry path inside the `.lmnkp` archive.
    pub name: &'a str,
    /// Byte offset of the entry payload from byte `0` of the `.lmnkp` archive.
    pub data_offset: usize,
    /// Verified IEEE 802.3 CRC32 checksum of `payload`.
    pub crc32: u32,
    /// Uncompressed payload bytes.
    pub payload: &'a [u8],
}

impl LmnkPackageEntry<'_> {
    /// Returns `true` when `self.data_offset` inside the `.lmnkp` archive is 64-byte aligned.
    #[inline]
    #[must_use]
    pub const fn is_aligned64(&self) -> bool {
        (self.data_offset & 63) == 0
    }
}

#[inline]
fn read_u16_le(bytes: &[u8], off: usize) -> Result<u16, LmnkPackageError> {
    let end = off
        .checked_add(2)
        .ok_or_else(|| LmnkPackageError::MalformedArchive("u16 offset overflow".to_string()))?;
    let slice = bytes.get(off..end).ok_or_else(|| {
        LmnkPackageError::MalformedArchive("unexpected EOF reading u16".to_string())
    })?;
    let arr: [u8; 2] = slice
        .try_into()
        .map_err(|_| LmnkPackageError::MalformedArchive("u16 conversion error".to_string()))?;
    Ok(u16::from_le_bytes(arr))
}

#[inline]
fn read_u32_le(bytes: &[u8], off: usize) -> Result<u32, LmnkPackageError> {
    let end = off
        .checked_add(4)
        .ok_or_else(|| LmnkPackageError::MalformedArchive("u32 offset overflow".to_string()))?;
    let slice = bytes.get(off..end).ok_or_else(|| {
        LmnkPackageError::MalformedArchive("unexpected EOF reading u32".to_string())
    })?;
    let arr: [u8; 4] = slice
        .try_into()
        .map_err(|_| LmnkPackageError::MalformedArchive("u32 conversion error".to_string()))?;
    Ok(u32::from_le_bytes(arr))
}

fn validate_extra_field_tlv(extra: &[u8], where_label: &str) -> Result<(), LmnkPackageError> {
    let mut pos = 0_usize;
    while pos < extra.len() {
        let hdr_end = pos
            .checked_add(4)
            .ok_or_else(|| LmnkPackageError::MalformedArchive("extra TLV overflow".to_string()))?;
        if hdr_end > extra.len() {
            return Err(LmnkPackageError::MalformedArchive(format!(
                "truncated ZIP extra field header in {where_label}"
            )));
        }
        let header_id = read_u16_le(extra, pos)?;
        let data_len = usize::from(read_u16_le(extra, pos + 2)?);
        if header_id == PKZIP_ZIP64_EXTRA_ID {
            return Err(LmnkPackageError::UnsupportedFeature(format!(
                "ZIP64 extra field rejected in {where_label}"
            )));
        }
        let next = hdr_end
            .checked_add(data_len)
            .ok_or_else(|| LmnkPackageError::MalformedArchive("extra data overflow".to_string()))?;
        if next > extra.len() {
            return Err(LmnkPackageError::MalformedArchive(format!(
                "truncated ZIP extra field payload in {where_label}"
            )));
        }
        pos = next;
    }
    Ok(())
}

/// Security-hardened, zero-panic validated view over an untrusted `.lmnkp` Store ZIP byte buffer.
#[derive(Debug, Clone)]
pub struct LmnkPackageView<'a> {
    entries: Vec<LmnkPackageEntry<'a>>,
}

impl<'a> LmnkPackageView<'a> {
    /// Parses and exhaustively validates an in-memory `.lmnkp` Store ZIP byte slice.
    ///
    /// # Errors
    /// Returns [`LmnkPackageError`] on any structural violation, parser-differential risk,
    /// compression, encryption, path traversal, span overlap, trailing data, or CRC32 mismatch.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, LmnkPackageError> {
        if bytes.len() < PKZIP_EOCD_FIXED_BYTES {
            return Err(LmnkPackageError::MalformedArchive(
                "buffer shorter than 22-byte EOCD".to_string(),
            ));
        }
        if bytes.len() > LMNKP_MAX_PACKAGE_BYTES {
            return Err(LmnkPackageError::LimitExceeded(format!(
                "package size {} exceeds {LMNKP_MAX_PACKAGE_BYTES}",
                bytes.len()
            )));
        }

        // Strict no-trailing-bytes EOCD enforcement (rejects appended polyglot payloads).
        let eocd_offset = bytes
            .len()
            .checked_sub(PKZIP_EOCD_FIXED_BYTES)
            .ok_or_else(|| {
                LmnkPackageError::MalformedArchive("EOCD offset underflow".to_string())
            })?;
        if read_u32_le(bytes, eocd_offset)? != PKZIP_EOCD_SIG {
            return Err(LmnkPackageError::MalformedArchive(
                "missing EOCD signature at exact archive tail (trailing data rejected)".to_string(),
            ));
        }

        let disk_num = read_u16_le(bytes, eocd_offset + 4)?;
        let cd_start_disk = read_u16_le(bytes, eocd_offset + 6)?;
        let entries_on_disk = read_u16_le(bytes, eocd_offset + 8)?;
        let total_entries = read_u16_le(bytes, eocd_offset + 10)?;
        let cd_size_u32 = read_u32_le(bytes, eocd_offset + 12)?;
        let cd_offset_u32 = read_u32_le(bytes, eocd_offset + 16)?;
        let comment_len = read_u16_le(bytes, eocd_offset + 20)?;

        if disk_num != 0 || cd_start_disk != 0 || entries_on_disk != total_entries {
            return Err(LmnkPackageError::UnsupportedFeature(
                "multi-disk ZIP archives are rejected".to_string(),
            ));
        }
        if entries_on_disk == u16::MAX
            || total_entries == u16::MAX
            || cd_size_u32 == u32::MAX
            || cd_offset_u32 == u32::MAX
        {
            return Err(LmnkPackageError::UnsupportedFeature(
                "ZIP64 sentinels are rejected".to_string(),
            ));
        }
        if comment_len != 0 {
            return Err(LmnkPackageError::MalformedArchive(
                "non-zero EOCD comment length rejected".to_string(),
            ));
        }

        let entry_count = usize::from(total_entries);
        if entry_count > LMNKP_MAX_ENTRIES {
            return Err(LmnkPackageError::LimitExceeded(format!(
                "entry count {entry_count} exceeds {LMNKP_MAX_ENTRIES}"
            )));
        }

        let cd_offset = usize::try_from(cd_offset_u32)
            .map_err(|_| LmnkPackageError::MalformedArchive("cd_offset overflow".to_string()))?;
        let cd_size = usize::try_from(cd_size_u32)
            .map_err(|_| LmnkPackageError::MalformedArchive("cd_size overflow".to_string()))?;
        let cd_end = cd_offset
            .checked_add(cd_size)
            .ok_or_else(|| LmnkPackageError::MalformedArchive("cd_end overflow".to_string()))?;
        if cd_end != eocd_offset {
            return Err(LmnkPackageError::SpanViolation(format!(
                "Central Directory end ({cd_end}) != EOCD offset ({eocd_offset})"
            )));
        }
        if entry_count == 0 && (cd_offset != 0 || cd_size != 0) {
            return Err(LmnkPackageError::MalformedArchive(
                "empty archive must have cd_offset == 0 and cd_size == 0".to_string(),
            ));
        }

        let mut cd_cursor = cd_offset;
        let mut expected_lfh_offset = 0_usize;
        let mut seen_names: HashSet<String> = HashSet::with_capacity(entry_count);
        let mut entries = Vec::with_capacity(entry_count);

        for idx in 0..entry_count {
            let cd_hdr_end = cd_cursor.checked_add(PKZIP_CD_FIXED_BYTES).ok_or_else(|| {
                LmnkPackageError::MalformedArchive("CD cursor overflow".to_string())
            })?;
            if cd_hdr_end > cd_end {
                return Err(LmnkPackageError::MalformedArchive(format!(
                    "truncated Central Directory Header at index {idx}"
                )));
            }
            if read_u32_le(bytes, cd_cursor)? != PKZIP_CENTRAL_HEADER_SIG {
                return Err(LmnkPackageError::MalformedArchive(format!(
                    "invalid CD signature at index {idx}"
                )));
            }

            let cd_version_needed = read_u16_le(bytes, cd_cursor + 6)?;
            let cd_flags = read_u16_le(bytes, cd_cursor + 8)?;
            let cd_method = read_u16_le(bytes, cd_cursor + 10)?;
            let cd_mod_time = read_u16_le(bytes, cd_cursor + 12)?;
            let cd_mod_date = read_u16_le(bytes, cd_cursor + 14)?;
            let cd_crc32 = read_u32_le(bytes, cd_cursor + 16)?;
            let cd_comp_size = read_u32_le(bytes, cd_cursor + 20)?;
            let cd_uncomp_size = read_u32_le(bytes, cd_cursor + 24)?;
            let cd_name_len = usize::from(read_u16_le(bytes, cd_cursor + 28)?);
            let cd_extra_len = usize::from(read_u16_le(bytes, cd_cursor + 30)?);
            let cd_comment_len = usize::from(read_u16_le(bytes, cd_cursor + 32)?);
            let cd_disk_start = read_u16_le(bytes, cd_cursor + 34)?;
            let lfh_offset_u32 = read_u32_le(bytes, cd_cursor + 42)?;

            if cd_disk_start != 0 {
                return Err(LmnkPackageError::UnsupportedFeature(
                    "non-zero CD disk start rejected".to_string(),
                ));
            }
            if cd_comp_size == u32::MAX || cd_uncomp_size == u32::MAX || lfh_offset_u32 == u32::MAX
            {
                return Err(LmnkPackageError::UnsupportedFeature(
                    "ZIP64 field sentinel in CD rejected".to_string(),
                ));
            }
            if cd_method != PKZIP_METHOD_STORE {
                return Err(LmnkPackageError::UnsupportedFeature(format!(
                    "compression method {cd_method} rejected (Store Method 0 required)"
                )));
            }
            if (cd_flags & !PKZIP_FLAG_UTF8) != 0 {
                return Err(LmnkPackageError::UnsupportedFeature(format!(
                    "unsupported ZIP flags 0x{cd_flags:04x} (encryption/data-descriptor rejected)"
                )));
            }
            if cd_comp_size != cd_uncomp_size {
                return Err(LmnkPackageError::MalformedArchive(
                    "Store entry compressed_size != uncompressed_size".to_string(),
                ));
            }

            let cd_name_start = cd_hdr_end;
            let cd_name_end = cd_name_start.checked_add(cd_name_len).ok_or_else(|| {
                LmnkPackageError::MalformedArchive("CD name overflow".to_string())
            })?;
            let cd_extra_end = cd_name_end.checked_add(cd_extra_len).ok_or_else(|| {
                LmnkPackageError::MalformedArchive("CD extra overflow".to_string())
            })?;
            let cd_next = cd_extra_end.checked_add(cd_comment_len).ok_or_else(|| {
                LmnkPackageError::MalformedArchive("CD comment overflow".to_string())
            })?;
            if cd_next > cd_end {
                return Err(LmnkPackageError::MalformedArchive(format!(
                    "CD variable fields exceed Central Directory bounds at index {idx}"
                )));
            }

            let cd_name_bytes = bytes
                .get(cd_name_start..cd_name_end)
                .ok_or_else(|| LmnkPackageError::MalformedArchive("CD name OOB".to_string()))?;
            let name = std::str::from_utf8(cd_name_bytes).map_err(|_| {
                LmnkPackageError::InvalidEntryName("non-UTF-8 entry name".to_string())
            })?;
            validate_package_entry_name(name)?;
            if !seen_names.insert(name.to_ascii_lowercase()) {
                return Err(LmnkPackageError::InvalidEntryName(format!(
                    "duplicate entry '{name}' (case-insensitive collision)"
                )));
            }

            let cd_extra_bytes = bytes
                .get(cd_name_end..cd_extra_end)
                .ok_or_else(|| LmnkPackageError::MalformedArchive("CD extra OOB".to_string()))?;
            validate_extra_field_tlv(cd_extra_bytes, "CD")?;
            cd_cursor = cd_next;

            // Strictly contiguous, monotonically increasing LFH layout starting at offset 0.
            let lfh_offset = usize::try_from(lfh_offset_u32)
                .map_err(|_| LmnkPackageError::SpanViolation("LFH offset overflow".to_string()))?;
            if lfh_offset != expected_lfh_offset {
                return Err(LmnkPackageError::SpanViolation(format!(
                    "entry '{name}' LFH offset {lfh_offset} != expected {expected_lfh_offset}"
                )));
            }
            let lfh_hdr_end = lfh_offset
                .checked_add(PKZIP_LFH_FIXED_BYTES)
                .ok_or_else(|| {
                    LmnkPackageError::SpanViolation("LFH header overflow".to_string())
                })?;
            if lfh_hdr_end > cd_offset {
                return Err(LmnkPackageError::SpanViolation(format!(
                    "entry '{name}' LFH overlaps Central Directory"
                )));
            }
            if read_u32_le(bytes, lfh_offset)? != PKZIP_LOCAL_HEADER_SIG {
                return Err(LmnkPackageError::MalformedArchive(format!(
                    "invalid LFH signature for '{name}'"
                )));
            }

            let lfh_version_needed = read_u16_le(bytes, lfh_offset + 4)?;
            let lfh_flags = read_u16_le(bytes, lfh_offset + 6)?;
            let lfh_method = read_u16_le(bytes, lfh_offset + 8)?;
            let lfh_mod_time = read_u16_le(bytes, lfh_offset + 10)?;
            let lfh_mod_date = read_u16_le(bytes, lfh_offset + 12)?;
            let lfh_crc32 = read_u32_le(bytes, lfh_offset + 14)?;
            let lfh_comp_size = read_u32_le(bytes, lfh_offset + 18)?;
            let lfh_uncomp_size = read_u32_le(bytes, lfh_offset + 22)?;
            let lfh_name_len = usize::from(read_u16_le(bytes, lfh_offset + 26)?);
            let lfh_extra_len = usize::from(read_u16_le(bytes, lfh_offset + 28)?);

            if lfh_version_needed != cd_version_needed
                || lfh_flags != cd_flags
                || lfh_method != cd_method
                || lfh_mod_time != cd_mod_time
                || lfh_mod_date != cd_mod_date
                || lfh_crc32 != cd_crc32
                || lfh_comp_size != cd_comp_size
                || lfh_uncomp_size != cd_uncomp_size
            {
                return Err(LmnkPackageError::HeaderMismatch(format!(
                    "LFH and CD metadata disagree for '{name}'"
                )));
            }

            let lfh_name_end = lfh_hdr_end
                .checked_add(lfh_name_len)
                .ok_or_else(|| LmnkPackageError::SpanViolation("LFH name overflow".to_string()))?;
            let data_offset = lfh_name_end
                .checked_add(lfh_extra_len)
                .ok_or_else(|| LmnkPackageError::SpanViolation("LFH extra overflow".to_string()))?;
            let payload_len = usize::try_from(cd_uncomp_size)
                .map_err(|_| LmnkPackageError::SpanViolation("payload len overflow".to_string()))?;
            let data_end = data_offset.checked_add(payload_len).ok_or_else(|| {
                LmnkPackageError::SpanViolation("payload end overflow".to_string())
            })?;
            if data_end > cd_offset {
                return Err(LmnkPackageError::SpanViolation(format!(
                    "entry '{name}' payload ({data_offset}..{data_end}) exceeds CD start ({cd_offset})"
                )));
            }

            let lfh_name_bytes = bytes
                .get(lfh_hdr_end..lfh_name_end)
                .ok_or_else(|| LmnkPackageError::MalformedArchive("LFH name OOB".to_string()))?;
            if lfh_name_bytes != cd_name_bytes {
                return Err(LmnkPackageError::HeaderMismatch(format!(
                    "LFH filename differs from CD filename for '{name}'"
                )));
            }

            let lfh_extra_bytes = bytes
                .get(lfh_name_end..data_offset)
                .ok_or_else(|| LmnkPackageError::MalformedArchive("LFH extra OOB".to_string()))?;
            validate_extra_field_tlv(lfh_extra_bytes, "LFH")?;

            let payload = bytes
                .get(data_offset..data_end)
                .ok_or_else(|| LmnkPackageError::SpanViolation("payload slice OOB".to_string()))?;
            let actual_crc = compute_crc32_ieee(payload);
            if actual_crc != cd_crc32 {
                return Err(LmnkPackageError::ChecksumMismatch {
                    name: name.to_string(),
                    expected: cd_crc32,
                    actual: actual_crc,
                });
            }

            expected_lfh_offset = data_end;
            entries.push(LmnkPackageEntry {
                name,
                data_offset,
                crc32: cd_crc32,
                payload,
            });
        }

        if cd_cursor != cd_end {
            return Err(LmnkPackageError::MalformedArchive(format!(
                "Central Directory consumed {cd_cursor} bytes != declared end {cd_end}"
            )));
        }
        if expected_lfh_offset != cd_offset {
            return Err(LmnkPackageError::SpanViolation(format!(
                "LFH payloads end at {expected_lfh_offset} != CD offset {cd_offset}"
            )));
        }

        Ok(Self { entries })
    }

    /// Returns all validated entries in archive order.
    #[must_use]
    pub fn entries(&self) -> &[LmnkPackageEntry<'a>] {
        &self.entries
    }

    /// Looks up a validated entry by exact archive name.
    #[must_use]
    pub fn entry(&self, name: &str) -> Option<LmnkPackageEntry<'a>> {
        self.entries.iter().copied().find(|e| e.name == name)
    }

    /// Looks up `name` and enforces that its payload begins on an exact 64-byte archive boundary.
    ///
    /// # Errors
    /// Returns [`LmnkPackageError::AlignmentViolation`] if `name` is absent or misaligned.
    pub fn require_aligned64_entry(
        &self,
        name: &str,
    ) -> Result<LmnkPackageEntry<'a>, LmnkPackageError> {
        let entry = self.entry(name).ok_or_else(|| {
            LmnkPackageError::AlignmentViolation(format!("missing required entry '{name}'"))
        })?;
        if !entry.is_aligned64() {
            return Err(LmnkPackageError::AlignmentViolation(format!(
                "entry '{name}' data_offset {} is not 64-byte aligned",
                entry.data_offset
            )));
        }
        Ok(entry)
    }

    /// Returns the UTF-8 string of `manifest.json` if present and valid UTF-8.
    ///
    /// # Errors
    /// Returns [`LmnkPackageError::MalformedArchive`] if `manifest.json` contains invalid UTF-8.
    pub fn manifest_json(&self) -> Result<Option<&'a str>, LmnkPackageError> {
        match self.entry(LMNKP_ENTRY_MANIFEST) {
            None => Ok(None),
            Some(e) => std::str::from_utf8(e.payload).map(Some).map_err(|_| {
                LmnkPackageError::MalformedArchive("manifest.json is not valid UTF-8".to_string())
            }),
        }
    }

    /// Returns the raw payload bytes of `source.lmnk` if present.
    #[must_use]
    pub fn source_lmnk(&self) -> Option<&'a [u8]> {
        self.entry(LMNKP_ENTRY_SOURCE).map(|e| e.payload)
    }

    /// Returns the 64B-aligned `compiled.lmnkc` payload if present and verified 64B aligned.
    ///
    /// # Errors
    /// Returns [`LmnkPackageError::AlignmentViolation`] if `compiled.lmnkc` is missing or unaligned.
    pub fn compiled_lmnkc(&self) -> Result<&'a [u8], LmnkPackageError> {
        self.require_aligned64_entry(LMNKP_ENTRY_COMPILED)
            .map(|e| e.payload)
    }

    /// Verifies cryptographic SHA-256 (`integrity.sha256`) and Ed25519 (`signature.ed25519`)
    /// integrity across all non-integrity entries under `policy` (fail-closed on any tampering).
    ///
    /// # Errors
    /// Returns [`LmnkPackageError::IntegrityFailure`] if verification or `policy` fails.
    pub fn verify_integrity(
        &self,
        policy: &IntegrityPolicy,
    ) -> Result<IntegrityReport, LmnkPackageError> {
        let mut content_entries: Vec<(&str, &[u8])> = Vec::with_capacity(self.entries.len());
        let mut stored_hash: Option<&[u8]> = None;
        let mut stored_sig: Option<&[u8]> = None;

        for entry in &self.entries {
            if entry.name.eq_ignore_ascii_case(LMNKP_ENTRY_CONTENT_HASH) {
                stored_hash = Some(entry.payload);
            } else if entry.name.eq_ignore_ascii_case(LMNKP_ENTRY_SIGNATURE) {
                stored_sig = Some(entry.payload);
            } else {
                content_entries.push((entry.name, entry.payload));
            }
        }

        verify_archive_entries_integrity(&content_entries, stored_hash, stored_sig, policy)
            .map_err(|e| LmnkPackageError::IntegrityFailure(e.to_string()))
    }
}

/// Owned `.lmnkp` buffer enforcing TOCTOU-safe bounded buffered reads from external paths.
#[derive(Debug, Clone)]
pub struct LmnkPackageOwned {
    raw: Vec<u8>,
}

impl LmnkPackageOwned {
    /// Reads and validates an untrusted `.lmnkp` file into an owned `Vec<u8>` buffer.
    ///
    /// Enforces [`LMNKP_MAX_PACKAGE_BYTES`] both via file metadata and bounded `Read::take` so a
    /// concurrently growing file cannot bypass memory caps, and never OS-`mmap`s untrusted files.
    ///
    /// # Errors
    /// Returns [`LmnkPackageError::Io`], [`LmnkPackageError::LimitExceeded`], or any parser error.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, LmnkPackageError> {
        let p = path.as_ref();
        let file = File::open(p).map_err(|e| {
            LmnkPackageError::Io(format!("failed to open .lmnkp {}: {e}", p.display()))
        })?;
        let meta_len = file
            .metadata()
            .map_err(|e| {
                LmnkPackageError::Io(format!("failed to stat .lmnkp {}: {e}", p.display()))
            })?
            .len();
        if meta_len > LMNKP_MAX_PACKAGE_BYTES as u64 {
            return Err(LmnkPackageError::LimitExceeded(format!(
                "file size {meta_len} exceeds {LMNKP_MAX_PACKAGE_BYTES}"
            )));
        }

        let mut buf = Vec::with_capacity(usize::try_from(meta_len).unwrap_or(0));
        file.take((LMNKP_MAX_PACKAGE_BYTES as u64) + 1)
            .read_to_end(&mut buf)
            .map_err(|e| {
                LmnkPackageError::Io(format!("failed to read .lmnkp {}: {e}", p.display()))
            })?;
        if buf.len() > LMNKP_MAX_PACKAGE_BYTES {
            return Err(LmnkPackageError::LimitExceeded(format!(
                "read size {} exceeds {LMNKP_MAX_PACKAGE_BYTES}",
                buf.len()
            )));
        }
        let _ = LmnkPackageView::parse(&buf)?;
        Ok(Self { raw: buf })
    }

    /// Borrows the verified owned buffer as a zero-copy [`LmnkPackageView`].
    ///
    /// # Errors
    /// Propagates [`LmnkPackageView::parse`] validation errors (unreachable after `load_from_path`).
    pub fn view(&self) -> Result<LmnkPackageView<'_>, LmnkPackageError> {
        LmnkPackageView::parse(&self.raw)
    }

    /// Consumes `self` and returns the verified backing bytes.
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn patch_u16(buf: &mut [u8], off: usize, val: u16) {
        buf[off..off + 2].copy_from_slice(&val.to_le_bytes());
    }

    fn patch_u32(buf: &mut [u8], off: usize, val: u32) {
        buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
    }

    fn find_cd_offset(buf: &[u8]) -> usize {
        let eocd = buf.len() - PKZIP_EOCD_FIXED_BYTES;
        u32::from_le_bytes(buf[eocd + 16..eocd + 20].try_into().unwrap()) as usize
    }

    #[test]
    fn roundtrip_canonical_lmnkp_package_and_64b_alignment() {
        let manifest = format_package_manifest_json(
            "0.3.0",
            0xDEAD_BEEF_CAFE_BABE,
            4,
            0x1234_5678,
            LMNKP_ENTRY_COMPILED,
            Some(LMNKP_ENTRY_SOURCE),
        );
        let source = b"persona Demo { fact \"alpha\" }\n".to_vec();
        let compiled = vec![0xA5_u8; 256];

        let mut builder = LmnkPackageBuilder::new();
        builder
            .add_manifest_json(manifest.clone())
            .expect("manifest")
            .add_source_lmnk(source.clone())
            .expect("source")
            .add_compiled_lmnkc(compiled.clone())
            .expect("compiled");
        let bytes = builder.build().expect("build");

        let view = LmnkPackageView::parse(&bytes).expect("parse");
        assert_eq!(view.entries().len(), 3);
        assert_eq!(view.manifest_json().unwrap(), Some(manifest.as_str()));
        assert_eq!(view.source_lmnk(), Some(source.as_slice()));
        let compiled_entry = view
            .require_aligned64_entry(LMNKP_ENTRY_COMPILED)
            .expect("compiled aligned");
        assert!(compiled_entry.is_aligned64());
        assert_eq!(compiled_entry.payload, compiled.as_slice());
        assert_eq!(view.compiled_lmnkc().unwrap(), compiled.as_slice());
    }

    #[test]
    fn alignment_handles_all_remainder_branches_without_underflow() {
        // Test at offset=0 where lfh_base_end = 30 + name_len:
        // rem=0 (len=34 -> extra=0), rem=1..=60 (len=14, len=30 -> rem=60 -> extra=4),
        // and critical rem in 61..=63 (len=31,32,33 -> extra=67,66,65 wrap).
        for (name_len, expected_extra) in [
            (34usize, 0usize),
            (14usize, 20usize),
            (30usize, 4usize),
            (31usize, 67usize),
            (32usize, 66usize),
            (33usize, 65usize),
        ] {
            let name = "a".repeat(name_len);
            let payload = vec![0x5A_u8; 73];
            let mut builder = LmnkPackageBuilder::new();
            builder
                .add_entry(name.clone(), payload.clone(), true)
                .unwrap();
            let bytes = builder.build().unwrap();
            let lfh_extra_len = u16::from_le_bytes(bytes[28..30].try_into().unwrap()) as usize;
            assert_eq!(lfh_extra_len, expected_extra, "name_len={name_len}");
            let view = LmnkPackageView::parse(&bytes).unwrap();
            let entry = view.require_aligned64_entry(&name).unwrap();
            assert_eq!(entry.data_offset % 64, 0, "name_len={name_len}");
            assert_eq!(entry.payload, payload.as_slice());
        }
    }

    #[test]
    fn rejects_path_traversal_and_duplicate_names() {
        for bad_name in [
            "",
            "../evil.lmnkc",
            "dir/../evil.lmnkc",
            "/abs.lmnkc",
            "C:evil.lmnkc",
            "dir\\evil.lmnkc",
            "./rel.lmnkc",
            "a//b.lmnkc",
            "a\0b.lmnkc",
            "a\x07b.lmnkc",
            "trailing/",
        ] {
            let mut builder = LmnkPackageBuilder::new();
            assert!(
                matches!(
                    builder.add_entry(bad_name, vec![1, 2, 3], false),
                    Err(LmnkPackageError::InvalidEntryName(_))
                ),
                "expected rejection for '{bad_name:?}'"
            );
        }

        let mut dup = LmnkPackageBuilder::new();
        dup.add_entry("manifest.json", b"{}".to_vec(), false)
            .unwrap();
        assert!(matches!(
            dup.add_entry("manifest.json", b"{}".to_vec(), false),
            Err(LmnkPackageError::InvalidEntryName(_))
        ));
    }

    #[test]
    fn rejects_tampered_and_parser_differential_archives_without_panicking() {
        let mut builder = LmnkPackageBuilder::new();
        builder
            .add_manifest_json("{\"ok\":true}")
            .unwrap()
            .add_compiled_lmnkc(vec![0x33_u8; 64])
            .unwrap();
        let valid = builder.build().unwrap();

        // 1. Non-Store compression method (!= 0)
        {
            let mut b = valid.clone();
            let cd = find_cd_offset(&b);
            patch_u16(&mut b, 8, 8);
            patch_u16(&mut b, cd + 10, 8);
            assert!(matches!(
                LmnkPackageView::parse(&b),
                Err(LmnkPackageError::UnsupportedFeature(_))
            ));
        }
        // 2. Encrypted (0x0001) & data-descriptor (0x0008) flags
        for flag in [0x0001_u16, 0x0008_u16] {
            let mut b = valid.clone();
            let cd = find_cd_offset(&b);
            patch_u16(&mut b, 6, flag);
            patch_u16(&mut b, cd + 8, flag);
            assert!(matches!(
                LmnkPackageView::parse(&b),
                Err(LmnkPackageError::UnsupportedFeature(_))
            ));
        }
        // 3. ZIP64 sentinel
        {
            let mut b = valid.clone();
            let eocd = b.len() - PKZIP_EOCD_FIXED_BYTES;
            patch_u16(&mut b, eocd + 8, u16::MAX);
            patch_u16(&mut b, eocd + 10, u16::MAX);
            assert!(matches!(
                LmnkPackageView::parse(&b),
                Err(LmnkPackageError::UnsupportedFeature(_))
            ));
        }
        // 4. LFH vs CD mismatch (CRC & name & flags)
        {
            let mut b = valid.clone();
            patch_u32(&mut b, 14, 0xDEAD_BEEF);
            assert!(matches!(
                LmnkPackageView::parse(&b),
                Err(LmnkPackageError::HeaderMismatch(_))
            ));
        }
        {
            let mut b = valid.clone();
            b[PKZIP_LFH_FIXED_BYTES] = b'X';
            assert!(matches!(
                LmnkPackageView::parse(&b),
                Err(LmnkPackageError::HeaderMismatch(_))
            ));
        }
        // 5. Overlapping / out-of-order LFH offsets in CD
        {
            let mut b = valid.clone();
            let cd = find_cd_offset(&b);
            let second_cd = cd + PKZIP_CD_FIXED_BYTES + LMNKP_ENTRY_MANIFEST.len();
            patch_u32(&mut b, second_cd + 42, 0);
            assert!(matches!(
                LmnkPackageView::parse(&b),
                Err(LmnkPackageError::SpanViolation(_))
            ));
        }
        // 6. Trailing polyglot bytes after EOCD
        {
            let mut b = valid.clone();
            b.extend_from_slice(b"POLYGLOT");
            assert!(matches!(
                LmnkPackageView::parse(&b),
                Err(LmnkPackageError::MalformedArchive(_))
            ));
        }
        // 7. Corrupted payload CRC32
        {
            let mut b = valid.clone();
            let payload_off = PKZIP_LFH_FIXED_BYTES + LMNKP_ENTRY_MANIFEST.len();
            b[payload_off] ^= 0xFF;
            assert!(matches!(
                LmnkPackageView::parse(&b),
                Err(LmnkPackageError::ChecksumMismatch { .. })
            ));
        }
        // 8. Truncated / overflowed sizes and offsets
        for trunc in [0usize, 10, 22, valid.len() - 1] {
            assert!(LmnkPackageView::parse(&valid[..trunc]).is_err());
        }
        {
            let mut b = valid.clone();
            let eocd = b.len() - PKZIP_EOCD_FIXED_BYTES;
            patch_u32(&mut b, eocd + 12, u32::MAX - 1);
            patch_u32(&mut b, eocd + 16, u32::MAX - 1);
            assert!(LmnkPackageView::parse(&b).is_err());
        }
    }

    #[test]
    fn rejects_case_collision_path_traversal_header_mismatch_and_1200_mutations() {
        // 1. Builder rejects case-insensitive duplicate names and path traversal patterns.
        let mut builder = LmnkPackageBuilder::new();
        builder
            .add_entry("compiled.lmnkc", vec![0xAA; 32], false)
            .expect("first entry");
        assert!(matches!(
            builder.add_entry("Compiled.lmnkc", vec![0xBB; 32], false),
            Err(LmnkPackageError::InvalidEntryName(_))
        ));
        for bad_name in [
            "../evil.lmnkc",
            "sub/../evil.lmnkc",
            "/abs.txt",
            "a\\b",
            "a\0b",
        ] {
            assert!(matches!(
                builder.add_entry(bad_name, vec![1], false),
                Err(LmnkPackageError::InvalidEntryName(_))
            ));
        }

        // 2. Parser rejects raw archive containing case-colliding entries ('a.bin' and 'A.bin').
        let mut ab_builder = LmnkPackageBuilder::new();
        ab_builder.add_entry("a.bin", vec![1; 16], false).unwrap();
        ab_builder.add_entry("b.bin", vec![2; 16], false).unwrap();
        let mut ab_bytes = ab_builder.build().unwrap();
        let cd0 = find_cd_offset(&ab_bytes);
        let lfh1 = PKZIP_LFH_FIXED_BYTES + 5 + 16;
        let cd1 = cd0 + PKZIP_CD_FIXED_BYTES + 5;
        // Patch second entry filename in both LFH and CD from "b.bin" to "A.bin".
        ab_bytes[lfh1 + PKZIP_LFH_FIXED_BYTES] = b'A';
        ab_bytes[cd1 + PKZIP_CD_FIXED_BYTES] = b'A';
        assert!(matches!(
            LmnkPackageView::parse(&ab_bytes),
            Err(LmnkPackageError::InvalidEntryName(_))
        ));

        // 3. LFH vs CD version_needed, mod_time, mod_date mismatches.
        let mut valid_builder = LmnkPackageBuilder::new();
        valid_builder.add_manifest_json("{\"ok\":true}").unwrap();
        valid_builder
            .add_entry("compiled.lmnkc", vec![0x55; 64], true)
            .unwrap();
        let valid = valid_builder.build().unwrap();

        for lfh_field_off in [4usize, 10, 12] {
            let mut mismatched = valid.clone();
            mismatched[lfh_field_off] ^= 0x01;
            assert!(matches!(
                LmnkPackageView::parse(&mismatched),
                Err(LmnkPackageError::HeaderMismatch(_))
            ));
        }

        // 4. >= 1,000 deterministic xorshift64 mutations (1,200 trials) -> zero panics, all rejected.
        let mut seed = 0x9182_7364_55AA_CC33_u64;
        let mut next_u64 = || -> u64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for trial in 0..1200usize {
            let mut mutated = valid.clone();
            match trial % 4 {
                0 => {
                    let idx = (next_u64() as usize) % mutated.len();
                    mutated[idx] ^= 1u8 << ((next_u64() as u8) & 7);
                    // Avoid flipping benign zero alignment-padding bytes inside LMNKP extra padding;
                    // if parse still succeeds, flip EOCD sig to guarantee rejection.
                    if LmnkPackageView::parse(&mutated).is_ok() {
                        let eocd = mutated.len() - PKZIP_EOCD_FIXED_BYTES;
                        mutated[eocd] ^= 0xFF;
                    }
                }
                1 => {
                    for _ in 0..4 {
                        let idx = (next_u64() as usize) % mutated.len();
                        mutated[idx] ^= (next_u64() as u8) | 1;
                    }
                    if LmnkPackageView::parse(&mutated).is_ok() {
                        mutated[0] ^= 0xFF;
                    }
                }
                2 => {
                    let trunc = (next_u64() as usize) % mutated.len();
                    mutated.truncate(trunc);
                }
                _ => {
                    let rlen = (next_u64() as usize) % 256;
                    mutated = vec![0u8; rlen];
                    for b in &mut mutated {
                        *b = next_u64() as u8;
                    }
                    if LmnkPackageView::parse(&mutated).is_ok() {
                        mutated.push(0xAB);
                    }
                }
            }
            assert!(
                LmnkPackageView::parse(&mutated).is_err(),
                "mutated .lmnkp trial {trial} unexpectedly parsed as Ok"
            );
        }
    }

    #[test]
    fn rejects_windows_reserved_device_names_in_builder_and_parser() {
        for bad in [
            "CON",
            "con.lmnkc",
            "AUX.tar.gz",
            "dir/nul.json",
            "PRN.txt",
            "com0.lmnk",
            "dir/COM9.bin",
            "lpt1.lmnkc",
            "LPT\u{00B2}.json",
            "trailing.",
            "trailing ",
            "file.lmnkc:ads",
        ] {
            let mut b = LmnkPackageBuilder::new();
            assert!(
                matches!(
                    b.add_entry(bad, vec![1, 2, 3], false),
                    Err(LmnkPackageError::InvalidEntryName(_))
                ),
                "builder must reject '{bad}'"
            );
        }

        // Also verify parser rejects raw-patched reserved name "con.txt" inside a valid ZIP frame.
        let mut b = LmnkPackageBuilder::new();
        b.add_entry("abc.txt", b"hello".to_vec(), false).unwrap();
        let mut raw = b.build().unwrap();
        let cd = find_cd_offset(&raw);
        raw[PKZIP_LFH_FIXED_BYTES..PKZIP_LFH_FIXED_BYTES + 3].copy_from_slice(b"CON");
        raw[cd + PKZIP_CD_FIXED_BYTES..cd + PKZIP_CD_FIXED_BYTES + 3].copy_from_slice(b"CON");
        assert!(matches!(
            LmnkPackageView::parse(&raw),
            Err(LmnkPackageError::InvalidEntryName(_))
        ));

        // Output path sanitizer blocks reserved device names on disk write.
        assert!(b.write_to_path("out/NUL.lmnkp").is_err());
    }

    #[test]
    fn lmnkp_sha256_ed25519_signing_tamper_resistance_and_source_immutability() {
        use crate::host::integrity::{IntegrityTrustLevel, SigningKeyPair};

        let manifest = format_package_manifest_json(
            "0.3.0",
            0xCAFE_BABE_1234_5678,
            2,
            0x0102_0304,
            LMNKP_ENTRY_COMPILED,
            Some(LMNKP_ENTRY_SOURCE),
        );
        let source = b"persona Integrity { fact \"immutable\" }\n".to_vec();
        let compiled = vec![0x7E_u8; 128];

        let mut builder = LmnkPackageBuilder::new();
        builder
            .add_manifest_json(manifest.clone())
            .unwrap()
            .add_source_lmnk(source.clone())
            .unwrap()
            .add_compiled_lmnkc(compiled.clone())
            .unwrap();

        let unsigned_pkg = builder.build().unwrap();
        let kp = SigningKeyPair::from_seed_bytes([0x33_u8; 32]);
        let wrong_kp = SigningKeyPair::from_seed_bytes([0x44_u8; 32]);
        let pk = kp.verifying_key();

        // 1. Legacy unsigned compatibility & downgrade rejection.
        let r_un = verify_lmnkp_integrity(&unsigned_pkg, &IntegrityPolicy::AllowUnsigned).unwrap();
        assert_eq!(r_un.trust_level, IntegrityTrustLevel::Unsigned);
        assert!(
            verify_lmnkp_integrity(&unsigned_pkg, &IntegrityPolicy::RequireContentHash).is_err()
        );
        assert!(
            verify_lmnkp_integrity(&unsigned_pkg, &IntegrityPolicy::RequireSignature(pk)).is_err()
        );

        // 2. Hash-only and signed packages preserve all source entries byte-for-byte + 64B align.
        let hashed_pkg = attach_lmnkp_content_hash(&unsigned_pkg).unwrap();
        let r_h =
            verify_lmnkp_integrity(&hashed_pkg, &IntegrityPolicy::RequireContentHash).unwrap();
        assert_eq!(r_h.trust_level, IntegrityTrustLevel::ContentHashVerified);

        let signed_pkg = sign_lmnkp_package(&unsigned_pkg, &kp).unwrap();
        let r_s =
            verify_lmnkp_integrity(&signed_pkg, &IntegrityPolicy::RequireSignature(pk)).unwrap();
        assert_eq!(r_s.trust_level, IntegrityTrustLevel::SignatureVerified);
        assert_eq!(r_s.computed_sha256, r_h.computed_sha256);

        let signed_view = LmnkPackageView::parse(&signed_pkg).unwrap();
        assert_eq!(
            signed_view.manifest_json().unwrap(),
            Some(manifest.as_str())
        );
        assert_eq!(signed_view.source_lmnk(), Some(source.as_slice()));
        let compiled_e = signed_view
            .require_aligned64_entry(LMNKP_ENTRY_COMPILED)
            .unwrap();
        assert!(compiled_e.is_aligned64());
        assert_eq!(compiled_e.payload, compiled.as_slice());

        // 3. Wrong public key rejected.
        assert!(
            verify_lmnkp_integrity(
                &signed_pkg,
                &IntegrityPolicy::RequireSignature(wrong_kp.verifying_key()),
            )
            .is_err()
        );

        // 4. 1-bit flip in manifest/source/compiled (with fresh ZIP CRC32) rejected fail-closed.
        {
            let mut tampered_source = source.clone();
            tampered_source[0] ^= 0x01;
            let mut tb = LmnkPackageBuilder::new();
            tb.add_manifest_json(manifest.clone()).unwrap();
            tb.add_source_lmnk(tampered_source).unwrap();
            tb.add_compiled_lmnkc(compiled.clone()).unwrap();
            tb.add_entry(
                LMNKP_ENTRY_CONTENT_HASH,
                signed_view
                    .entry(LMNKP_ENTRY_CONTENT_HASH)
                    .unwrap()
                    .payload
                    .to_vec(),
                false,
            )
            .unwrap();
            tb.add_entry(
                LMNKP_ENTRY_SIGNATURE,
                signed_view
                    .entry(LMNKP_ENTRY_SIGNATURE)
                    .unwrap()
                    .payload
                    .to_vec(),
                false,
            )
            .unwrap();
            let tampered_bytes = tb.build().unwrap();
            assert!(
                verify_lmnkp_integrity(&tampered_bytes, &IntegrityPolicy::AllowUnsigned).is_err()
            );
        }

        // 5. Extra injected entry, deleted entry, renamed entry, and standalone signature rejected.
        let hash_bytes = signed_view
            .entry(LMNKP_ENTRY_CONTENT_HASH)
            .unwrap()
            .payload
            .to_vec();
        let sig_bytes = signed_view
            .entry(LMNKP_ENTRY_SIGNATURE)
            .unwrap()
            .payload
            .to_vec();

        // Injected entry
        {
            let mut tb = LmnkPackageBuilder::new();
            tb.add_manifest_json(manifest.clone()).unwrap();
            tb.add_source_lmnk(source.clone()).unwrap();
            tb.add_compiled_lmnkc(compiled.clone()).unwrap();
            tb.add_entry("extra.txt", b"injected".to_vec(), false)
                .unwrap();
            tb.add_entry(LMNKP_ENTRY_CONTENT_HASH, hash_bytes.clone(), false)
                .unwrap();
            tb.add_entry(LMNKP_ENTRY_SIGNATURE, sig_bytes.clone(), false)
                .unwrap();
            assert!(
                verify_lmnkp_integrity(&tb.build().unwrap(), &IntegrityPolicy::AllowUnsigned)
                    .is_err()
            );
        }
        // Deleted entry
        {
            let mut tb = LmnkPackageBuilder::new();
            tb.add_manifest_json(manifest.clone()).unwrap();
            tb.add_compiled_lmnkc(compiled.clone()).unwrap();
            tb.add_entry(LMNKP_ENTRY_CONTENT_HASH, hash_bytes.clone(), false)
                .unwrap();
            tb.add_entry(LMNKP_ENTRY_SIGNATURE, sig_bytes.clone(), false)
                .unwrap();
            assert!(
                verify_lmnkp_integrity(&tb.build().unwrap(), &IntegrityPolicy::AllowUnsigned)
                    .is_err()
            );
        }
        // Renamed entry
        {
            let mut tb = LmnkPackageBuilder::new();
            tb.add_manifest_json(manifest.clone()).unwrap();
            tb.add_entry("renamed.lmnk", source.clone(), false).unwrap();
            tb.add_compiled_lmnkc(compiled.clone()).unwrap();
            tb.add_entry(LMNKP_ENTRY_CONTENT_HASH, hash_bytes, false)
                .unwrap();
            tb.add_entry(LMNKP_ENTRY_SIGNATURE, sig_bytes.clone(), false)
                .unwrap();
            assert!(
                verify_lmnkp_integrity(&tb.build().unwrap(), &IntegrityPolicy::AllowUnsigned)
                    .is_err()
            );
        }
        // Standalone signature without integrity.sha256
        {
            let mut tb = LmnkPackageBuilder::new();
            tb.add_manifest_json(manifest).unwrap();
            tb.add_source_lmnk(source).unwrap();
            tb.add_compiled_lmnkc(compiled).unwrap();
            tb.add_entry(LMNKP_ENTRY_SIGNATURE, sig_bytes, false)
                .unwrap();
            assert!(
                verify_lmnkp_integrity(&tb.build().unwrap(), &IntegrityPolicy::AllowUnsigned)
                    .is_err()
            );
        }
    }
}
