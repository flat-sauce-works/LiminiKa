// name: src/adapter/src/host/integrity.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cryptographic integrity (`ContentHash 0xFF01` SHA-256) and detached signature
//! (`Signature 0xFF02` Ed25519 `verify_strict`) layer for `.lmnkc` containers and `.lmnkp`
//! archives, strictly isolated outside `#![no_std]` `liminika-core`.
//!
//! # Canonical Transcript & Fail-Closed Guarantees
//! - **`.lmnkc` (`0xFF01` / `0xFF02`)**:
//!   - `0xFF01` stores the raw 32-byte SHA-256 digest of a domain-separated, length-prefixed
//!     canonical transcript covering header invariants (`magic`, `version`, `topology_crc32`,
//!     `container_flags`, `timestamp_epoch_sec`) and every non-security section (`kind != 0xFF01`
//!     and `kind != 0xFF02`) in exact container order (`kind || flags || len_u64 || payload`).
//!   - `0xFF02` stores the raw 64-byte Ed25519 signature over `LMNKC-SIGNATURE-V1\0 || content_hash`
//!     and is verified with strict non-malleable verification (`VerifyingKey::verify_strict`).
//!   - A standalone `0xFF02` without a valid matching `0xFF01` is unconditionally rejected.
//! - **`.lmnkp` (Store ZIP entries)**:
//!   - Computes a domain-separated, length-prefixed SHA-256 over all non-integrity entries sorted
//!     canonically by UTF-8 name bytes, and signs `LMNKP-SIGNATURE-V1\0 || content_hash` via
//!     Ed25519 `verify_strict`.
//! - **Backward Compatibility & Fail-Closed Policy**:
//!   - Legacy unsigned artifacts succeed under [`IntegrityPolicy::AllowUnsigned`] with
//!     [`IntegrityTrustLevel::Unsigned`], and fail under [`IntegrityPolicy::RequireContentHash`]
//!     or [`IntegrityPolicy::RequireSignature`].
//!   - Whenever an integrity section/entry is present, any 1-bit tampering, malformed length,
//!     header mutation, section reorder/swap, or standalone signature fails closed regardless of
//!     policy.

use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use liminika_core::liminal::LMNK_ERR_TOPOLOGY_MISMATCH;
use liminika_core::runtime::{
    required_container_byte_len, write_container_sections_to_bytes, LmnkContainerHeader,
    LmnkContainerView, LmnkSectionInput, CORE_TOPOLOGY, LMNK_SECTION_CONTENT_HASH,
    LMNK_SECTION_SIGNATURE,
};
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, Zeroizing};

use crate::host::path_guard::{validate_local_output_path, PathGuardError};

/// Exact byte length of a SHA-256 `ContentHash` (`0xFF01`) digest (`32`).
pub const CONTENT_HASH_BYTES: usize = 32;

/// Exact byte length of an Ed25519 `Signature` (`0xFF02`) payload (`64`).
pub const SIGNATURE_BYTES: usize = 64;

/// Exact byte length of an Ed25519 verifying public key (`32`).
pub const PUBLIC_KEY_BYTES: usize = 32;

/// Exact byte length of an Ed25519 secret signing seed (`32`).
pub const SECRET_KEY_SEED_BYTES: usize = 32;

const LMNKC_CONTENT_HASH_DOMAIN: &[u8] = b"LMNKC-CONTENT-HASH-V1\0";
const LMNKC_SIGNATURE_DOMAIN: &[u8] = b"LMNKC-SIGNATURE-V1\0";
const LMNKP_CONTENT_HASH_DOMAIN: &[u8] = b"LMNKP-CONTENT-HASH-V1\0";
const LMNKP_SIGNATURE_DOMAIN: &[u8] = b"LMNKP-SIGNATURE-V1\0";
const MAX_KEY_FILE_BYTES: u64 = 1024;

/// Cryptographic trust level established after verifying an artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityTrustLevel {
    /// Artifact carries neither a content hash nor a signature (permitted only by `AllowUnsigned`).
    Unsigned,
    /// Artifact carries a verified SHA-256 `ContentHash` (`0xFF01` / `integrity.sha256`).
    ContentHashVerified,
    /// Artifact carries both a verified SHA-256 `ContentHash` and a strictly verified Ed25519
    /// signature matching the caller-supplied trusted [`VerifyingPublicKey`].
    SignatureVerified,
}

impl fmt::Display for IntegrityTrustLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsigned => f.write_str("Unsigned"),
            Self::ContentHashVerified => f.write_str("ContentHashVerified"),
            Self::SignatureVerified => f.write_str("SignatureVerified"),
        }
    }
}

/// Caller-enforced verification policy for `.lmnkc` containers and `.lmnkp` packages.
///
/// Regardless of policy, any present but corrupted, mismatched, malformed, or standalone
/// integrity payload is rejected fail-closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityPolicy {
    /// Allows legacy unsigned artifacts while still failing closed on tampered hashes/signatures.
    AllowUnsigned,
    /// Requires a valid SHA-256 `ContentHash` section/entry; rejects unsigned artifacts.
    RequireContentHash,
    /// Requires both a valid SHA-256 `ContentHash` and a strict Ed25519 signature by the given key.
    RequireSignature(VerifyingPublicKey),
}

/// Verification summary emitted by [`verify_lmnkc_integrity`] and
/// [`verify_archive_entries_integrity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntegrityReport {
    /// Established trust tier.
    pub trust_level: IntegrityTrustLevel,
    /// Freshly computed SHA-256 canonical transcript digest.
    pub computed_sha256: [u8; CONTENT_HASH_BYTES],
    /// `true` if a `ContentHash` (`0xFF01` / `integrity.sha256`) payload was present and verified.
    pub has_content_hash: bool,
    /// `true` if a `Signature` (`0xFF02` / `signature.ed25519`) payload was present.
    pub has_signature: bool,
}

impl IntegrityReport {
    /// Formats `computed_sha256` as a lowercase 64-character hexadecimal string.
    #[must_use]
    pub fn sha256_hex(&self) -> String {
        encode_hex_lower(&self.computed_sha256)
    }
}

/// Structured errors returned by cryptographic integrity, key handling, and signing APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityError {
    /// Underlying `.lmnkc` container failed `LmnkContainerView::parse` structural validation.
    ContainerInvalid(String),
    /// Container belongs to a different `CORE_TOPOLOGY` hash.
    TopologyMismatch {
        /// Active host `CORE_TOPOLOGY` CRC32.
        expected: u32,
        /// Observed container `topology_crc32`.
        actual: u32,
    },
    /// Integrity payload length or encoding is malformed.
    MalformedIntegrityPayload(String),
    /// A `Signature` (`0xFF02`) section/entry appeared without a valid `ContentHash` (`0xFF01`).
    SignatureWithoutContentHash,
    /// Computed canonical SHA-256 digest disagreed with stored `ContentHash`.
    ContentHashMismatch {
        /// Stored expected SHA-256 digest.
        expected: [u8; CONTENT_HASH_BYTES],
        /// Freshly computed SHA-256 digest.
        actual: [u8; CONTENT_HASH_BYTES],
    },
    /// Ed25519 strict signature verification failed.
    SignatureVerificationFailed(String),
    /// Artifact did not meet the caller's [`IntegrityPolicy`] (e.g. unsigned downgrade).
    PolicyViolation(String),
    /// Key material failed format or Ed25519 point validation.
    InvalidKey(String),
    /// OS cryptographic RNG failed while generating a keypair.
    RngFailure(String),
    /// Output path rejected by [`validate_local_output_path`].
    PathRejected(PathGuardError),
    /// Filesystem I/O failure.
    Io(String),
}

impl fmt::Display for IntegrityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContainerInvalid(msg) => write!(f, "invalid container: {msg}"),
            Self::TopologyMismatch { expected, actual } => write!(
                f,
                "topology mismatch: expected 0x{expected:08x}, actual 0x{actual:08x}"
            ),
            Self::MalformedIntegrityPayload(msg) => {
                write!(f, "malformed integrity payload: {msg}")
            }
            Self::SignatureWithoutContentHash => f.write_str(
                "signature section/entry present without a valid ContentHash (fail-closed)",
            ),
            Self::ContentHashMismatch { expected, actual } => write!(
                f,
                "SHA-256 ContentHash mismatch: expected={}, actual={}",
                encode_hex_lower(expected),
                encode_hex_lower(actual)
            ),
            Self::SignatureVerificationFailed(msg) => {
                write!(f, "Ed25519 strict signature verification failed: {msg}")
            }
            Self::PolicyViolation(msg) => write!(f, "integrity policy violation: {msg}"),
            Self::InvalidKey(msg) => write!(f, "invalid Ed25519 key material: {msg}"),
            Self::RngFailure(msg) => write!(f, "OS CSPRNG failure: {msg}"),
            Self::PathRejected(err) => write!(f, "unsafe path rejected: {err}"),
            Self::Io(msg) => write!(f, "integrity I/O error: {msg}"),
        }
    }
}

impl std::error::Error for IntegrityError {}

impl From<PathGuardError> for IntegrityError {
    fn from(err: PathGuardError) -> Self {
        Self::PathRejected(err)
    }
}

/// Validated Ed25519 verifying public key (`32` bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerifyingPublicKey {
    bytes: [u8; PUBLIC_KEY_BYTES],
}

impl fmt::Debug for VerifyingPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VerifyingPublicKey({})", self.to_hex())
    }
}

impl fmt::Display for VerifyingPublicKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl VerifyingPublicKey {
    /// Constructs and validates an Ed25519 verifying public key from 32 raw bytes.
    ///
    /// # Errors
    /// Returns [`IntegrityError::InvalidKey`] if the bytes do not encode a valid Ed25519 point.
    pub fn from_bytes(bytes: [u8; PUBLIC_KEY_BYTES]) -> Result<Self, IntegrityError> {
        let _ = VerifyingKey::from_bytes(&bytes)
            .map_err(|e| IntegrityError::InvalidKey(format!("invalid Ed25519 public key: {e}")))?;
        Ok(Self { bytes })
    }

    /// Parses a 64-character hexadecimal Ed25519 public key (whitespace-trimmed).
    ///
    /// # Errors
    /// Returns [`IntegrityError::InvalidKey`] if hex decoding or curve validation fails.
    pub fn from_hex(hex_str: &str) -> Result<Self, IntegrityError> {
        let bytes = decode_hex_exact::<PUBLIC_KEY_BYTES>(hex_str)?;
        Self::from_bytes(bytes)
    }

    /// Returns the raw 32-byte Ed25519 public key encoding.
    #[inline]
    #[must_use]
    pub const fn to_bytes(&self) -> [u8; PUBLIC_KEY_BYTES] {
        self.bytes
    }

    /// Formats the public key as a lowercase 64-character hexadecimal string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        encode_hex_lower(&self.bytes)
    }

    /// Writes the hexadecimal public key to `path` after enforcing [`validate_local_output_path`].
    ///
    /// # Errors
    /// Returns [`IntegrityError::PathRejected`] or [`IntegrityError::Io`].
    pub fn write_hex_file(&self, path: impl AsRef<Path>) -> Result<(), IntegrityError> {
        let p = path.as_ref();
        validate_local_output_path(p)?;
        if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                IntegrityError::Io(format!(
                    "failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let mut content = self.to_hex();
        content.push('\n');
        std::fs::write(p, content.as_bytes()).map_err(|e| {
            IntegrityError::Io(format!("failed to write public key {}: {e}", p.display()))
        })
    }

    /// Reads and validates a hexadecimal Ed25519 public key from `path`.
    ///
    /// # Errors
    /// Returns [`IntegrityError::Io`] or [`IntegrityError::InvalidKey`].
    pub fn read_hex_file(path: impl AsRef<Path>) -> Result<Self, IntegrityError> {
        let raw = read_bounded_key_file(path.as_ref())?;
        let s = std::str::from_utf8(&raw)
            .map_err(|_| IntegrityError::InvalidKey("public key file is not valid UTF-8".into()))?;
        Self::from_hex(s)
    }

    fn as_dalek(&self) -> Result<VerifyingKey, IntegrityError> {
        VerifyingKey::from_bytes(&self.bytes)
            .map_err(|e| IntegrityError::InvalidKey(format!("invalid Ed25519 public key: {e}")))
    }
}

/// Ed25519 signing keypair wrapper guaranteeing redacted `Debug` formatting and zeroization on drop.
///
/// The wrapped [`ed25519_dalek::SigningKey`] implements `ZeroizeOnDrop`, and all intermediate seed
/// or hex serialization buffers are explicitly scrubbed via [`zeroize::Zeroize`] / [`Zeroizing`].
pub struct SigningKeyPair {
    inner: SigningKey,
}

impl fmt::Debug for SigningKeyPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SigningKeyPair([REDACTED])")
    }
}

impl SigningKeyPair {
    /// Generates a fresh Ed25519 signing keypair using the host OS CSPRNG (`getrandom`).
    ///
    /// # Errors
    /// Returns [`IntegrityError::RngFailure`] if the OS CSPRNG returns an error.
    pub fn generate() -> Result<Self, IntegrityError> {
        let mut seed = [0_u8; SECRET_KEY_SEED_BYTES];
        getrandom::fill(&mut seed)
            .map_err(|e| IntegrityError::RngFailure(format!("getrandom failed: {e}")))?;
        let keypair = Self::from_seed_bytes(seed);
        seed.zeroize();
        Ok(keypair)
    }

    /// Constructs a deterministic Ed25519 signing keypair from a 32-byte seed, scrubbing the local
    /// copy immediately after key derivation.
    #[must_use]
    pub fn from_seed_bytes(mut seed: [u8; SECRET_KEY_SEED_BYTES]) -> Self {
        let inner = SigningKey::from_bytes(&seed);
        seed.zeroize();
        Self { inner }
    }

    /// Parses a 64-character hexadecimal Ed25519 secret seed (whitespace-trimmed).
    ///
    /// # Errors
    /// Returns [`IntegrityError::InvalidKey`] on invalid hex length or characters.
    pub fn from_seed_hex(hex_str: &str) -> Result<Self, IntegrityError> {
        let mut seed = decode_hex_exact::<SECRET_KEY_SEED_BYTES>(hex_str)?;
        let keypair = Self::from_seed_bytes(seed);
        seed.zeroize();
        Ok(keypair)
    }

    /// Returns the corresponding [`VerifyingPublicKey`].
    #[must_use]
    pub fn verifying_key(&self) -> VerifyingPublicKey {
        VerifyingPublicKey {
            bytes: self.inner.verifying_key().to_bytes(),
        }
    }

    /// Writes the secret signing seed as lowercase hex to `path` (enforcing
    /// [`validate_local_output_path`] and scrubbing intermediate buffers on drop).
    ///
    /// On Unix platforms, the file is created with mode `0o600`.
    ///
    /// # Errors
    /// Returns [`IntegrityError::PathRejected`] or [`IntegrityError::Io`].
    pub fn write_seed_hex_file(&self, path: impl AsRef<Path>) -> Result<(), IntegrityError> {
        let p = path.as_ref();
        validate_local_output_path(p)?;
        if let Some(parent) = p.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| {
                IntegrityError::Io(format!(
                    "failed to create directory {}: {e}",
                    parent.display()
                ))
            })?;
        }
        let seed_bytes = Zeroizing::new(self.inner.to_bytes());
        let mut hex = Zeroizing::new(encode_hex_lower(&*seed_bytes));
        hex.push('\n');
        write_secret_bytes_to_path(p, hex.as_bytes())
    }

    /// Reads a hexadecimal Ed25519 secret seed from `path`, scrubbing the read buffer on drop.
    ///
    /// # Errors
    /// Returns [`IntegrityError::Io`] or [`IntegrityError::InvalidKey`].
    pub fn read_seed_hex_file(path: impl AsRef<Path>) -> Result<Self, IntegrityError> {
        let raw = read_bounded_key_file(path.as_ref())?;
        let s = std::str::from_utf8(&raw)
            .map_err(|_| IntegrityError::InvalidKey("secret key file is not valid UTF-8".into()))?;
        Self::from_seed_hex(s)
    }

    fn sign_domain_digest(
        &self,
        domain: &[u8],
        digest: &[u8; CONTENT_HASH_BYTES],
    ) -> [u8; SIGNATURE_BYTES] {
        let mut msg = Vec::with_capacity(domain.len() + CONTENT_HASH_BYTES);
        msg.extend_from_slice(domain);
        msg.extend_from_slice(digest);
        let sig: Signature = self.inner.sign(&msg);
        sig.to_bytes()
    }
}

/// Encodes `bytes` as a lowercase hexadecimal string.
#[must_use]
pub fn encode_hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

/// Decodes a whitespace-trimmed fixed-length hexadecimal string into `[u8; N]`.
///
/// # Errors
/// Returns [`IntegrityError::InvalidKey`] if the length or any hex digit is invalid.
pub fn decode_hex_exact<const N: usize>(input: &str) -> Result<[u8; N], IntegrityError> {
    let trimmed = input.trim();
    if trimmed.len() != N * 2 {
        return Err(IntegrityError::InvalidKey(format!(
            "expected {} hex chars, got {}",
            N * 2,
            trimmed.len()
        )));
    }
    let bytes = trimmed.as_bytes();
    let mut out = [0_u8; N];
    for idx in 0..N {
        let hi = decode_hex_nibble(bytes[idx * 2])?;
        let lo = decode_hex_nibble(bytes[idx * 2 + 1])?;
        out[idx] = (hi << 4) | lo;
    }
    Ok(out)
}

#[inline]
fn decode_hex_nibble(b: u8) -> Result<u8, IntegrityError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(IntegrityError::InvalidKey(
            "non-hexadecimal character in key or digest".to_string(),
        )),
    }
}

#[inline]
fn ct_eq_32(a: &[u8; CONTENT_HASH_BYTES], b: &[u8; CONTENT_HASH_BYTES]) -> bool {
    let mut diff = 0_u8;
    for i in 0..CONTENT_HASH_BYTES {
        diff |= a[i] ^ b[i];
    }
    core::hint::black_box(diff) == 0
}

fn read_bounded_key_file(path: &Path) -> Result<Zeroizing<Vec<u8>>, IntegrityError> {
    let file = File::open(path).map_err(|e| {
        IntegrityError::Io(format!("failed to open key file {}: {e}", path.display()))
    })?;
    let mut buf = Zeroizing::new(Vec::new());
    file.take(MAX_KEY_FILE_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| {
            IntegrityError::Io(format!("failed to read key file {}: {e}", path.display()))
        })?;
    if (buf.len() as u64) > MAX_KEY_FILE_BYTES {
        return Err(IntegrityError::InvalidKey(format!(
            "key file {} exceeds {} byte cap",
            path.display(),
            MAX_KEY_FILE_BYTES
        )));
    }
    Ok(buf)
}

fn write_secret_bytes_to_path(path: &Path, data: &[u8]) -> Result<(), IntegrityError> {
    use std::io::Write as _;

    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        opts.mode(0o600);
    }
    let mut file = opts.open(path).map_err(|e| {
        IntegrityError::Io(format!(
            "failed to create secret key {}: {e}",
            path.display()
        ))
    })?;
    file.write_all(data).map_err(|e| {
        IntegrityError::Io(format!(
            "failed to write secret key {}: {e}",
            path.display()
        ))
    })
}

fn parse_container_view(bytes: &[u8]) -> Result<LmnkContainerView<'_>, IntegrityError> {
    LmnkContainerView::parse(bytes).map_err(|status| {
        if status == LMNK_ERR_TOPOLOGY_MISMATCH {
            let actual = LmnkContainerHeader::from_le_bytes(bytes)
                .map(|h| h.topology_crc32)
                .unwrap_or(0);
            IntegrityError::TopologyMismatch {
                expected: CORE_TOPOLOGY.topology_crc32,
                actual,
            }
        } else {
            IntegrityError::ContainerInvalid(format!(
                "LmnkContainerView rejected container: {} ({})",
                status.as_str(),
                status.code()
            ))
        }
    })
}

#[inline]
const fn is_security_section(kind: u16) -> bool {
    kind == LMNK_SECTION_CONTENT_HASH || kind == LMNK_SECTION_SIGNATURE
}

/// Computes the canonical, domain-separated, length-prefixed SHA-256 digest of a validated
/// `.lmnkc` container view across invariant header fields and all non-security sections in exact
/// container order.
///
/// # Errors
/// Returns [`IntegrityError::ContainerInvalid`] if a declared section slice cannot be retrieved.
pub fn compute_lmnkc_content_hash(
    view: &LmnkContainerView<'_>,
) -> Result<[u8; CONTENT_HASH_BYTES], IntegrityError> {
    let hdr = view.header();
    let mut hasher = Sha256::new();
    hasher.update(LMNKC_CONTENT_HASH_DOMAIN);
    hasher.update(hdr.magic.to_le_bytes());
    hasher.update(hdr.version.to_le_bytes());
    hasher.update(hdr.topology_crc32.to_le_bytes());
    hasher.update(hdr.container_flags.to_le_bytes());
    hasher.update(hdr.timestamp_epoch_sec.to_le_bytes());

    let content_count = view
        .sections()
        .iter()
        .filter(|s| !is_security_section(s.kind))
        .count();
    let count_u32 = u32::try_from(content_count)
        .map_err(|_| IntegrityError::ContainerInvalid("section count exceeds u32".into()))?;
    hasher.update(count_u32.to_le_bytes());

    for sec in view.sections() {
        if is_security_section(sec.kind) {
            continue;
        }
        let payload = view.section_bytes(sec.kind).ok_or_else(|| {
            IntegrityError::ContainerInvalid(format!(
                "missing payload bytes for section kind 0x{:04x}",
                sec.kind
            ))
        })?;
        hasher.update(sec.kind.to_le_bytes());
        hasher.update(sec.flags.to_le_bytes());
        hasher.update((payload.len() as u64).to_le_bytes());
        hasher.update(payload);
    }

    let digest: [u8; CONTENT_HASH_BYTES] = hasher.finalize().into();
    Ok(digest)
}

fn verify_ed25519_strict(
    pubkey: &VerifyingPublicKey,
    domain: &[u8],
    digest: &[u8; CONTENT_HASH_BYTES],
    sig_bytes: &[u8],
) -> Result<(), IntegrityError> {
    if sig_bytes.len() != SIGNATURE_BYTES {
        return Err(IntegrityError::MalformedIntegrityPayload(format!(
            "expected {SIGNATURE_BYTES}-byte Ed25519 signature, got {} bytes",
            sig_bytes.len()
        )));
    }
    let signature = Signature::from_slice(sig_bytes).map_err(|e| {
        IntegrityError::MalformedIntegrityPayload(format!("invalid Ed25519 signature bytes: {e}"))
    })?;
    let vk = pubkey.as_dalek()?;
    let mut msg = Vec::with_capacity(domain.len() + CONTENT_HASH_BYTES);
    msg.extend_from_slice(domain);
    msg.extend_from_slice(digest);
    vk.verify_strict(&msg, &signature)
        .map_err(|e| IntegrityError::SignatureVerificationFailed(e.to_string()))
}

fn evaluate_integrity_payloads(
    computed_sha256: [u8; CONTENT_HASH_BYTES],
    stored_hash: Option<&[u8]>,
    stored_sig: Option<&[u8]>,
    signature_domain: &[u8],
    policy: &IntegrityPolicy,
) -> Result<IntegrityReport, IntegrityError> {
    // Fail-closed invariant 1: Signature without a paired ContentHash is unconditionally rejected.
    if stored_sig.is_some() && stored_hash.is_none() {
        return Err(IntegrityError::SignatureWithoutContentHash);
    }

    // Fail-closed invariant 2: Present ContentHash must be exact length and match computed SHA-256.
    if let Some(raw_hash) = stored_hash {
        let expected: [u8; CONTENT_HASH_BYTES] = raw_hash.try_into().map_err(|_| {
            IntegrityError::MalformedIntegrityPayload(format!(
                "expected {CONTENT_HASH_BYTES}-byte SHA-256 ContentHash, got {} bytes",
                raw_hash.len()
            ))
        })?;
        if !ct_eq_32(&computed_sha256, &expected) {
            return Err(IntegrityError::ContentHashMismatch {
                expected,
                actual: computed_sha256,
            });
        }
    }

    // Fail-closed invariant 3: Present Signature must be structurally valid 64B Ed25519 encoding.
    if let Some(raw_sig) = stored_sig {
        if raw_sig.len() != SIGNATURE_BYTES {
            return Err(IntegrityError::MalformedIntegrityPayload(format!(
                "expected {SIGNATURE_BYTES}-byte Ed25519 signature, got {} bytes",
                raw_sig.len()
            )));
        }
        let _ = Signature::from_slice(raw_sig).map_err(|e| {
            IntegrityError::MalformedIntegrityPayload(format!(
                "invalid Ed25519 signature encoding: {e}"
            ))
        })?;
    }

    let has_content_hash = stored_hash.is_some();
    let has_signature = stored_sig.is_some();

    let trust_level = match policy {
        IntegrityPolicy::AllowUnsigned => {
            if has_content_hash {
                IntegrityTrustLevel::ContentHashVerified
            } else {
                IntegrityTrustLevel::Unsigned
            }
        }
        IntegrityPolicy::RequireContentHash => {
            if !has_content_hash {
                return Err(IntegrityError::PolicyViolation(
                    "missing required SHA-256 ContentHash".to_string(),
                ));
            }
            IntegrityTrustLevel::ContentHashVerified
        }
        IntegrityPolicy::RequireSignature(pubkey) => {
            if !has_content_hash {
                return Err(IntegrityError::PolicyViolation(
                    "missing required SHA-256 ContentHash".to_string(),
                ));
            }
            let Some(raw_sig) = stored_sig else {
                return Err(IntegrityError::PolicyViolation(
                    "missing required Ed25519 Signature".to_string(),
                ));
            };
            verify_ed25519_strict(pubkey, signature_domain, &computed_sha256, raw_sig)?;
            IntegrityTrustLevel::SignatureVerified
        }
    };

    Ok(IntegrityReport {
        trust_level,
        computed_sha256,
        has_content_hash,
        has_signature,
    })
}

/// Validates a `.lmnkc` container's structure, `ContentHash (0xFF01)` SHA-256, and
/// `Signature (0xFF02)` Ed25519 (`verify_strict`) against `policy`.
///
/// # Errors
/// Returns [`IntegrityError`] on container corruption, topology mismatch, standalone signature,
/// hash/signature tampering, or policy violation.
pub fn verify_lmnkc_integrity(
    container_bytes: &[u8],
    policy: &IntegrityPolicy,
) -> Result<IntegrityReport, IntegrityError> {
    let view = parse_container_view(container_bytes)?;
    let computed_sha256 = compute_lmnkc_content_hash(&view)?;
    let stored_hash = view.section_bytes(LMNK_SECTION_CONTENT_HASH);
    let stored_sig = view.section_bytes(LMNK_SECTION_SIGNATURE);
    evaluate_integrity_payloads(
        computed_sha256,
        stored_hash,
        stored_sig,
        LMNKC_SIGNATURE_DOMAIN,
        policy,
    )
}

/// Re-emits a validated `.lmnkc` container with a fresh `ContentHash (0xFF01)` SHA-256 section
/// appended (stripping any stale `0xFF01`/`0xFF02` sections while preserving header metadata and
/// exact non-security section order).
///
/// # Errors
/// Returns [`IntegrityError`] if input container parsing or section serialization fails.
pub fn attach_lmnkc_content_hash(container_bytes: &[u8]) -> Result<Vec<u8>, IntegrityError> {
    rewrite_lmnkc_with_security(container_bytes, None)
}

/// Re-emits a validated `.lmnkc` container with both `ContentHash (0xFF01)` (SHA-256) and
/// `Signature (0xFF02)` (Ed25519) sections appended in canonical order.
///
/// # Errors
/// Returns [`IntegrityError`] if input container parsing or section serialization fails.
pub fn sign_lmnkc_container(
    container_bytes: &[u8],
    keypair: &SigningKeyPair,
) -> Result<Vec<u8>, IntegrityError> {
    rewrite_lmnkc_with_security(container_bytes, Some(keypair))
}

fn rewrite_lmnkc_with_security(
    container_bytes: &[u8],
    keypair: Option<&SigningKeyPair>,
) -> Result<Vec<u8>, IntegrityError> {
    let view = parse_container_view(container_bytes)?;
    let hdr = view.header();
    let digest = compute_lmnkc_content_hash(&view)?;
    let sig_opt = keypair.map(|kp| kp.sign_domain_digest(LMNKC_SIGNATURE_DOMAIN, &digest));

    let mut sections: Vec<LmnkSectionInput<'_>> = Vec::with_capacity(view.sections().len() + 2);
    for sec in view.sections() {
        if is_security_section(sec.kind) {
            continue;
        }
        let payload = view.section_bytes(sec.kind).ok_or_else(|| {
            IntegrityError::ContainerInvalid(format!(
                "missing payload bytes for section kind 0x{:04x}",
                sec.kind
            ))
        })?;
        sections.push(LmnkSectionInput {
            kind: sec.kind,
            flags: sec.flags,
            payload,
        });
    }
    sections.push(LmnkSectionInput {
        kind: LMNK_SECTION_CONTENT_HASH,
        flags: 0,
        payload: &digest,
    });
    if let Some(ref sig) = sig_opt {
        sections.push(LmnkSectionInput {
            kind: LMNK_SECTION_SIGNATURE,
            flags: 0,
            payload: sig,
        });
    }

    let req_len = required_container_byte_len(&sections)
        .map_err(|e| IntegrityError::ContainerInvalid(format!("container sizing failed: {e}")))?;
    let mut out = vec![0_u8; req_len];
    let written = write_container_sections_to_bytes(
        hdr.container_flags,
        hdr.timestamp_epoch_sec,
        &sections,
        &mut out,
    )
    .map_err(|e| IntegrityError::ContainerInvalid(format!("container write failed: {e}")))?;
    out.truncate(written);
    Ok(out)
}

/// Computes the canonical, domain-separated, length-prefixed SHA-256 digest over a set of
/// non-integrity archive entries sorted canonically by UTF-8 name bytes.
#[must_use]
pub fn compute_archive_entries_content_hash(entries: &[(&str, &[u8])]) -> [u8; CONTENT_HASH_BYTES] {
    let mut sorted: Vec<(&str, &[u8])> = entries.to_vec();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));

    let mut hasher = Sha256::new();
    hasher.update(LMNKP_CONTENT_HASH_DOMAIN);
    hasher.update((sorted.len() as u32).to_le_bytes());
    for (name, payload) in sorted {
        hasher.update((name.len() as u32).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update((payload.len() as u64).to_le_bytes());
        hasher.update(payload);
    }
    hasher.finalize().into()
}

/// Signs an archive `ContentHash` digest using domain `LMNKP-SIGNATURE-V1\0` and Ed25519.
#[must_use]
pub fn sign_archive_content_hash(
    content_hash: &[u8; CONTENT_HASH_BYTES],
    keypair: &SigningKeyPair,
) -> [u8; SIGNATURE_BYTES] {
    keypair.sign_domain_digest(LMNKP_SIGNATURE_DOMAIN, content_hash)
}

/// Verifies archive entries against optional stored SHA-256 hash and Ed25519 signature payloads
/// under `policy` (fail-closed on any present tampering or standalone signature).
///
/// # Errors
/// Returns [`IntegrityError`] on standalone signature, hash mismatch, strict Ed25519 verification
/// failure, malformed payload, or policy violation.
pub fn verify_archive_entries_integrity(
    content_entries: &[(&str, &[u8])],
    stored_hash: Option<&[u8]>,
    stored_sig: Option<&[u8]>,
    policy: &IntegrityPolicy,
) -> Result<IntegrityReport, IntegrityError> {
    let computed_sha256 = compute_archive_entries_content_hash(content_entries);
    evaluate_integrity_payloads(
        computed_sha256,
        stored_hash,
        stored_sig,
        LMNKP_SIGNATURE_DOMAIN,
        policy,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use liminika_core::runtime::{
        LMNK_CONTAINER_FLAG_COMPILED_CACHE, LMNK_SECTION_PHASE_ANCHORS, LMNK_SECTION_POINTER_TRAILS,
    };
    use liminika_core::stigmergy::{PhaseAnchorRecord, PointerTrail, ATTRACTOR_SLOT_KNOWLEDGE};

    fn make_sample_container() -> Vec<u8> {
        let mut anchor = PhaseAnchorRecord::new_empty(7, ATTRACTOR_SLOT_KNOWLEDGE, 0);
        anchor.base_energy = 1.5;
        anchor.phase_gravity_mass = 1.5;
        anchor.stamp_crc32();
        let mut anchor_bytes = [0_u8; core::mem::size_of::<PhaseAnchorRecord>()];
        PhaseAnchorRecord::write_records_to_container_bytes(&[anchor], &mut anchor_bytes)
            .expect("serialize anchor");

        let trail = PointerTrail::default();
        let mut trail_bytes = [0_u8; core::mem::size_of::<PointerTrail>()];
        liminika_core::foundation::Pod::write_unaligned_to(&trail, &mut trail_bytes, 0)
            .expect("serialize trail");

        let sections = [
            LmnkSectionInput {
                kind: LMNK_SECTION_PHASE_ANCHORS,
                flags: 0,
                payload: &anchor_bytes,
            },
            LmnkSectionInput {
                kind: LMNK_SECTION_POINTER_TRAILS,
                flags: 0,
                payload: &trail_bytes,
            },
        ];
        let req = required_container_byte_len(&sections).expect("req");
        let mut out = vec![0_u8; req];
        write_container_sections_to_bytes(
            LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            1_774_900_000,
            &sections,
            &mut out,
        )
        .expect("write");
        out
    }

    fn rewrite_with_valid_crcs(mut buf: Vec<u8>) -> Vec<u8> {
        let mut hdr = LmnkContainerHeader::from_le_bytes(&buf).expect("hdr");
        hdr.finalize_crc32();
        buf[..64].copy_from_slice(&hdr.to_le_bytes());
        buf
    }

    #[test]
    fn signing_keypair_redacts_debug_and_roundtrips_deterministic_seed() {
        let seed = [0x42_u8; SECRET_KEY_SEED_BYTES];
        let kp = SigningKeyPair::from_seed_bytes(seed);
        let debug_str = format!("{kp:?}");
        assert_eq!(debug_str, "SigningKeyPair([REDACTED])");
        assert!(!debug_str.contains("4242"));

        let vk = kp.verifying_key();
        let reparsed = VerifyingPublicKey::from_hex(&vk.to_hex()).expect("reparse pubkey");
        assert_eq!(vk, reparsed);

        let generated = SigningKeyPair::generate().expect("os csprng keypair");
        assert_ne!(generated.verifying_key(), vk);
    }

    #[test]
    fn lmnkc_unsigned_hashed_signed_and_all_tamper_attacks() {
        let unsigned = make_sample_container();
        let kp = SigningKeyPair::from_seed_bytes([0x11_u8; 32]);
        let wrong_kp = SigningKeyPair::from_seed_bytes([0x22_u8; 32]);
        let pk = kp.verifying_key();
        let wrong_pk = wrong_kp.verifying_key();

        // 1. Legacy unsigned compatibility & downgrade rejection.
        let r_un =
            verify_lmnkc_integrity(&unsigned, &IntegrityPolicy::AllowUnsigned).expect("unsigned");
        assert_eq!(r_un.trust_level, IntegrityTrustLevel::Unsigned);
        assert!(!r_un.has_content_hash);
        assert!(!r_un.has_signature);
        assert!(matches!(
            verify_lmnkc_integrity(&unsigned, &IntegrityPolicy::RequireContentHash),
            Err(IntegrityError::PolicyViolation(_))
        ));
        assert!(matches!(
            verify_lmnkc_integrity(&unsigned, &IntegrityPolicy::RequireSignature(pk)),
            Err(IntegrityError::PolicyViolation(_))
        ));

        // 2. ContentHash (0xFF01) only.
        let hashed = attach_lmnkc_content_hash(&unsigned).expect("attach hash");
        let r_hash =
            verify_lmnkc_integrity(&hashed, &IntegrityPolicy::RequireContentHash).expect("hash ok");
        assert_eq!(r_hash.trust_level, IntegrityTrustLevel::ContentHashVerified);
        assert!(r_hash.has_content_hash);
        assert!(!r_hash.has_signature);
        assert!(matches!(
            verify_lmnkc_integrity(&hashed, &IntegrityPolicy::RequireSignature(pk)),
            Err(IntegrityError::PolicyViolation(_))
        ));

        // 3. Signed (0xFF01 + 0xFF02) with strict verification.
        let signed = sign_lmnkc_container(&unsigned, &kp).expect("sign");
        let r_sig = verify_lmnkc_integrity(&signed, &IntegrityPolicy::RequireSignature(pk))
            .expect("sig ok");
        assert_eq!(r_sig.trust_level, IntegrityTrustLevel::SignatureVerified);
        assert_eq!(r_sig.computed_sha256, r_hash.computed_sha256);

        // Wrong public key rejected.
        assert!(matches!(
            verify_lmnkc_integrity(&signed, &IntegrityPolicy::RequireSignature(wrong_pk)),
            Err(IntegrityError::SignatureVerificationFailed(_))
        ));

        // 4. 1-bit payload flip (even with recomputed section CRC32) fails closed under AllowUnsigned.
        {
            let view = LmnkContainerView::parse(&signed).expect("view");
            let trail_entry = view.find_section(LMNK_SECTION_POINTER_TRAILS).unwrap();
            let mut tampered_trail = view
                .section_bytes(LMNK_SECTION_POINTER_TRAILS)
                .unwrap()
                .to_vec();
            tampered_trail[0] ^= 0x01;
            let anchor_bytes = view.section_bytes(LMNK_SECTION_PHASE_ANCHORS).unwrap();
            let old_hash = view.section_bytes(LMNK_SECTION_CONTENT_HASH).unwrap();
            let old_sig = view.section_bytes(LMNK_SECTION_SIGNATURE).unwrap();
            let sections = [
                LmnkSectionInput {
                    kind: LMNK_SECTION_PHASE_ANCHORS,
                    flags: 0,
                    payload: anchor_bytes,
                },
                LmnkSectionInput {
                    kind: trail_entry.kind,
                    flags: 0,
                    payload: &tampered_trail,
                },
                LmnkSectionInput {
                    kind: LMNK_SECTION_CONTENT_HASH,
                    flags: 0,
                    payload: old_hash,
                },
                LmnkSectionInput {
                    kind: LMNK_SECTION_SIGNATURE,
                    flags: 0,
                    payload: old_sig,
                },
            ];
            let mut buf = vec![0_u8; required_container_byte_len(&sections).unwrap()];
            write_container_sections_to_bytes(
                view.header().container_flags,
                view.header().timestamp_epoch_sec,
                &sections,
                &mut buf,
            )
            .unwrap();
            assert!(matches!(
                verify_lmnkc_integrity(&buf, &IntegrityPolicy::AllowUnsigned),
                Err(IntegrityError::ContentHashMismatch { .. })
            ));
        }

        // 5. Header timestamp tamper (with recomputed header CRC32) fails closed.
        {
            let mut bad_hdr = signed.clone();
            let new_ts = 1_999_999_999_u64;
            bad_hdr[24..32].copy_from_slice(&new_ts.to_le_bytes());
            bad_hdr = rewrite_with_valid_crcs(bad_hdr);
            assert!(matches!(
                verify_lmnkc_integrity(&bad_hdr, &IntegrityPolicy::AllowUnsigned),
                Err(IntegrityError::ContentHashMismatch { .. })
            ));
        }

        // 6. Section reordering attack (swap PhaseAnchors and PointerTrails with fresh CRCs) fails closed.
        {
            let view = LmnkContainerView::parse(&signed).expect("view");
            let sections = [
                LmnkSectionInput {
                    kind: LMNK_SECTION_POINTER_TRAILS,
                    flags: 0,
                    payload: view.section_bytes(LMNK_SECTION_POINTER_TRAILS).unwrap(),
                },
                LmnkSectionInput {
                    kind: LMNK_SECTION_PHASE_ANCHORS,
                    flags: 0,
                    payload: view.section_bytes(LMNK_SECTION_PHASE_ANCHORS).unwrap(),
                },
                LmnkSectionInput {
                    kind: LMNK_SECTION_CONTENT_HASH,
                    flags: 0,
                    payload: view.section_bytes(LMNK_SECTION_CONTENT_HASH).unwrap(),
                },
                LmnkSectionInput {
                    kind: LMNK_SECTION_SIGNATURE,
                    flags: 0,
                    payload: view.section_bytes(LMNK_SECTION_SIGNATURE).unwrap(),
                },
            ];
            let mut reordered = vec![0_u8; required_container_byte_len(&sections).unwrap()];
            write_container_sections_to_bytes(
                view.header().container_flags,
                view.header().timestamp_epoch_sec,
                &sections,
                &mut reordered,
            )
            .unwrap();
            assert!(matches!(
                verify_lmnkc_integrity(&reordered, &IntegrityPolicy::AllowUnsigned),
                Err(IntegrityError::ContentHashMismatch { .. })
            ));
        }

        // 7. Standalone Signature (0xFF02) without ContentHash (0xFF01) rejected fail-closed.
        {
            let view = LmnkContainerView::parse(&signed).expect("view");
            let sections = [
                LmnkSectionInput {
                    kind: LMNK_SECTION_PHASE_ANCHORS,
                    flags: 0,
                    payload: view.section_bytes(LMNK_SECTION_PHASE_ANCHORS).unwrap(),
                },
                LmnkSectionInput {
                    kind: LMNK_SECTION_SIGNATURE,
                    flags: 0,
                    payload: view.section_bytes(LMNK_SECTION_SIGNATURE).unwrap(),
                },
            ];
            let mut solo_sig = vec![0_u8; required_container_byte_len(&sections).unwrap()];
            write_container_sections_to_bytes(
                view.header().container_flags,
                view.header().timestamp_epoch_sec,
                &sections,
                &mut solo_sig,
            )
            .unwrap();
            assert!(matches!(
                verify_lmnkc_integrity(&solo_sig, &IntegrityPolicy::AllowUnsigned),
                Err(IntegrityError::SignatureWithoutContentHash)
            ));
        }

        // 8. Truncated ContentHash payload rejected fail-closed.
        {
            let view = LmnkContainerView::parse(&unsigned).expect("view");
            let sections = [
                LmnkSectionInput {
                    kind: LMNK_SECTION_PHASE_ANCHORS,
                    flags: 0,
                    payload: view.section_bytes(LMNK_SECTION_PHASE_ANCHORS).unwrap(),
                },
                LmnkSectionInput {
                    kind: LMNK_SECTION_CONTENT_HASH,
                    flags: 0,
                    payload: &[0xAA_u8; 16],
                },
            ];
            let mut trunc_hash = vec![0_u8; required_container_byte_len(&sections).unwrap()];
            write_container_sections_to_bytes(
                view.header().container_flags,
                view.header().timestamp_epoch_sec,
                &sections,
                &mut trunc_hash,
            )
            .unwrap();
            assert!(matches!(
                verify_lmnkc_integrity(&trunc_hash, &IntegrityPolicy::AllowUnsigned),
                Err(IntegrityError::MalformedIntegrityPayload(_))
            ));
        }
    }
}
