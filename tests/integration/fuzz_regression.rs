// name: tests/integration/fuzz_regression.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deep-reaching deterministic mutation fuzz regression suite covering all four untrusted parser
//! boundaries (`.lmnkc`, `.lmnkp` + `path_guard`, `.lmnk` DSL, and `LmnkTimeline` v2).
//!
//! Kept strictly separate from `tests/core/` (Miri targets) and registered as an integration test
//! under `liminika-cli` so the workspace dependency graph and `Cargo.lock` remain untouched.

use liminika_adapter::{
    AdaptiveGopCutDetector, AttractorRole, EquilibriumConfig, EquilibriumState, IntegrityPolicy,
    IntegrityTrustLevel, LMNKP_ENTRY_COMPILED, LMNKP_ENTRY_CONTENT_HASH, LMNKP_ENTRY_SIGNATURE,
    LMNKP_ENTRY_SOURCE, LmnkPackageBuilder, LmnkPackageView, LmnkTimeline, ModelSteeringAdapter,
    RebakeContext, RecallPolicy, RopeGeometry, RopeLayout, SigningKeyPair, SourceTextRecord,
    SubtitleActivationEncoder, format_package_manifest_json, validate_archive_relative_path,
    verify_lmnkc_integrity, verify_lmnkp_integrity,
};
use liminika_core::drivers::LiminalEgoPreset;
use liminika_core::foundation::compute_crc32_ieee;
use liminika_core::liminal::{Lacuna, MAX_ATTENTION_HEADS};
use liminika_core::runtime::{
    CORE_TOPOLOGY, Config, LMNK_CONTAINER_FLAG_COMPILED_CACHE, LMNK_CONTAINER_HEADER_BYTES,
    LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES, LMNK_SECTION_CONTENT_HASH, LMNK_SECTION_ENTRY_BYTES,
    LMNK_SECTION_SIGNATURE, LmnkContainerView, MappedStorage, RuntimeContext,
};
use liminika_core::stigmergy::{ATTRACTOR_SLOT_KNOWLEDGE, PhaseAnchorRecord};
use liminika_dsl::{parse_and_validate_lmnk, parse_lmnk_program};

#[derive(Clone, Copy)]
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            0
        } else {
            (self.next_u64() as usize) % bound
        }
    }

    fn next_byte(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }
}

fn recompute_lmnkc_header_crc(buf: &mut [u8]) {
    if buf.len() < LMNK_CONTAINER_HEADER_BYTES {
        return;
    }
    let crc = compute_crc32_ieee(&buf[..60]);
    buf[60..64].copy_from_slice(&crc.to_le_bytes());
}

fn recompute_lmnkc_section_and_header_crcs(buf: &mut [u8]) {
    if buf.len() < LMNK_CONTAINER_HEADER_BYTES {
        return;
    }
    let sec_count = u16::from_le_bytes([buf[10], buf[11]]) as usize;
    let table_off = u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]) as usize;
    for idx in 0..sec_count.min(32) {
        let entry_off = table_off.saturating_add(idx.saturating_mul(LMNK_SECTION_ENTRY_BYTES));
        if entry_off + LMNK_SECTION_ENTRY_BYTES > buf.len() {
            break;
        }
        let off = u32::from_le_bytes([
            buf[entry_off + 4],
            buf[entry_off + 5],
            buf[entry_off + 6],
            buf[entry_off + 7],
        ]) as usize;
        let len = u32::from_le_bytes([
            buf[entry_off + 8],
            buf[entry_off + 9],
            buf[entry_off + 10],
            buf[entry_off + 11],
        ]) as usize;
        if let Some(end) = off.checked_add(len)
            && end <= buf.len()
        {
            let sec_crc = compute_crc32_ieee(&buf[off..end]);
            buf[entry_off + 12..entry_off + 16].copy_from_slice(&sec_crc.to_le_bytes());
        }
    }
    recompute_lmnkc_header_crc(buf);
}

fn build_valid_equilibrium_seeds() -> (Vec<u8>, Vec<u8>, SigningKeyPair) {
    let keypair = SigningKeyPair::from_seed_bytes([0x42_u8; 32]);
    let mut eq = EquilibriumState::new(EquilibriumConfig::default());
    let ctx = RuntimeContext::new(Config::new()).expect("runtime context");
    let mut snap = [0_u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut req = 0usize;
    ctx.serialize_snapshot(&mut snap, &mut req)
        .expect("snapshot");
    eq.update_runtime_snapshot_from_container(&snap)
        .expect("attach snapshot");

    let mut rec = PhaseAnchorRecord::new_empty(7, ATTRACTOR_SLOT_KNOWLEDGE, 0);
    rec.base_energy = 1.75;
    rec.phase_gravity_mass = 1.75;
    rec.sample_count = 3;
    rec.phase_profile = [24_i8; MAX_ATTENTION_HEADS];
    eq.insert_anchor(rec, Lacuna::SelfRef, true)
        .expect("insert anchor");

    let unsigned = eq.to_lmnkc_bytes(1_774_900_000).expect("unsigned lmnkc");
    let signed = eq
        .to_signed_lmnkc_bytes(1_774_900_000, &keypair)
        .expect("signed lmnkc");
    (unsigned, signed, keypair)
}

#[test]
fn fuzz_regression_lmnkc_container_deep_mutations() {
    let (unsigned_seed, signed_seed, keypair) = build_valid_equilibrium_seeds();
    let pubkey = keypair.verifying_key();
    let req_sig = IntegrityPolicy::RequireSignature(pubkey);

    assert_eq!(
        verify_lmnkc_integrity(&unsigned_seed, &IntegrityPolicy::AllowUnsigned)
            .expect("unsigned ok")
            .trust_level,
        IntegrityTrustLevel::Unsigned
    );
    assert_eq!(
        verify_lmnkc_integrity(&signed_seed, &req_sig)
            .expect("signed ok")
            .trust_level,
        IntegrityTrustLevel::SignatureVerified
    );
    assert!(EquilibriumState::from_lmnkc_bytes(&signed_seed, EquilibriumConfig::default()).is_ok());

    let mut rng = XorShift64::new(0x1145_1419_1981_0001);
    let seeds = [&unsigned_seed, &signed_seed];

    // 1. Targeted deep mutations (800 iterations) penetrating header CRC, section CRCs,
    //    EquilibriumState `0x0101` / `RuntimeSnapshot`, and `0xFF01`/`0xFF02` security sections.
    for iter in 0..800_usize {
        let base = seeds[iter & 1];
        let mut buf = base.clone();
        match iter % 8 {
            0 => {
                // Header invariant mutation (past magic) + header CRC recompute.
                let off = 4 + rng.next_usize(56);
                buf[off] ^= 1_u8 << (rng.next_usize(8));
                recompute_lmnkc_header_crc(&mut buf);
            }
            1 => {
                // Section directory entry mutation + header CRC recompute.
                let table_end = buf.len().min(LMNK_CONTAINER_HEADER_BYTES + 8 * 16);
                if table_end > LMNK_CONTAINER_HEADER_BYTES {
                    let off = LMNK_CONTAINER_HEADER_BYTES
                        + rng.next_usize(table_end - LMNK_CONTAINER_HEADER_BYTES);
                    buf[off] ^= 0x01_u8.wrapping_shl((rng.next_usize(8)) as u32);
                    recompute_lmnkc_header_crc(&mut buf);
                }
            }
            2 | 3 => {
                // Deep payload mutation with BOTH section CRC and header CRC patched so the
                // mutation bypasses container CRC32 and hits Equilibrium / Integrity decoding.
                if buf.len() > 128 {
                    let flips = 1 + rng.next_usize(4);
                    for _ in 0..flips {
                        let off = 128 + rng.next_usize(buf.len() - 128);
                        buf[off] ^= (rng.next_byte()).max(1);
                    }
                    recompute_lmnkc_section_and_header_crcs(&mut buf);
                }
            }
            4 => {
                // Truncation across header, section directory, and payload boundaries.
                let new_len = rng.next_usize(buf.len());
                buf.truncate(new_len);
            }
            5 => {
                // Trailing polyglot bytes or padding corruption.
                let extra = 1 + rng.next_usize(96);
                for _ in 0..extra {
                    buf.push(rng.next_byte());
                }
            }
            6 => {
                // Rewrite section kind to standalone 0xFF02 or bogus security kind + patch CRCs.
                if buf.len() >= LMNK_CONTAINER_HEADER_BYTES + 16 {
                    let kind = match rng.next_usize(3) {
                        0 => LMNK_SECTION_SIGNATURE,
                        1 => LMNK_SECTION_CONTENT_HASH,
                        _ => 0x0101_u16,
                    };
                    buf[LMNK_CONTAINER_HEADER_BYTES..LMNK_CONTAINER_HEADER_BYTES + 2]
                        .copy_from_slice(&kind.to_le_bytes());
                    recompute_lmnkc_section_and_header_crcs(&mut buf);
                }
            }
            _ => {
                // Pure pseudo-random blob with optional valid magic prefix.
                let len = rng.next_usize(512);
                buf = (0..len).map(|_| rng.next_byte()).collect();
                if buf.len() >= 4 && (iter & 2) != 0 {
                    buf[0..4].copy_from_slice(b"LMNK");
                }
            }
        }

        let _ = LmnkContainerView::parse(&buf);
        let _ = verify_lmnkc_integrity(&buf, &IntegrityPolicy::AllowUnsigned);
        let _ = verify_lmnkc_integrity(&buf, &IntegrityPolicy::RequireContentHash);
        let _ = verify_lmnkc_integrity(&buf, &req_sig);
        let _ = EquilibriumState::from_lmnkc_bytes(&buf, EquilibriumConfig::default());
    }

    // 2. Fail-closed guarantee: any 1-bit flip in signed payload (even with CRCs recomputed)
    //    must fail both RequireSignature and AllowUnsigned.
    for bit_step in 0..64_usize {
        let mut tampered = signed_seed.clone();
        let off = 128 + (bit_step * 17) % (tampered.len() - 128);
        tampered[off] ^= 1_u8 << (bit_step & 7);
        recompute_lmnkc_section_and_header_crcs(&mut tampered);
        assert!(verify_lmnkc_integrity(&tampered, &req_sig).is_err());
        assert!(verify_lmnkc_integrity(&tampered, &IntegrityPolicy::AllowUnsigned).is_err());
    }
}

fn patch_zip_entry_crcs(buf: &mut [u8], payload_offset: usize, payload_len: usize) {
    if payload_offset + payload_len > buf.len() || buf.len() < 22 {
        return;
    }
    let crc = compute_crc32_ieee(&buf[payload_offset..payload_offset + payload_len]);
    let crc_bytes = crc.to_le_bytes();
    // Locate EOCD and patch matching CD + LFH CRC32 fields when layout is intact.
    let eocd = buf.len() - 22;
    if &buf[eocd..eocd + 4] != b"PK\x05\x06" {
        return;
    }
    let cd_off = u32::from_le_bytes([
        buf[eocd + 16],
        buf[eocd + 17],
        buf[eocd + 18],
        buf[eocd + 19],
    ]) as usize;
    let cd_size = u32::from_le_bytes([
        buf[eocd + 12],
        buf[eocd + 13],
        buf[eocd + 14],
        buf[eocd + 15],
    ]) as usize;
    let mut cur = cd_off;
    let cd_end = cd_off.saturating_add(cd_size).min(eocd);
    while cur + 46 <= cd_end && &buf[cur..cur + 4] == b"PK\x01\x02" {
        let name_len = u16::from_le_bytes([buf[cur + 28], buf[cur + 29]]) as usize;
        let extra_len = u16::from_le_bytes([buf[cur + 30], buf[cur + 31]]) as usize;
        let cmt_len = u16::from_le_bytes([buf[cur + 32], buf[cur + 33]]) as usize;
        let lfh_off =
            u32::from_le_bytes([buf[cur + 42], buf[cur + 43], buf[cur + 44], buf[cur + 45]])
                as usize;
        if lfh_off + 30 <= cd_off && &buf[lfh_off..lfh_off + 4] == b"PK\x03\x04" {
            let lfh_name_len = u16::from_le_bytes([buf[lfh_off + 26], buf[lfh_off + 27]]) as usize;
            let lfh_extra_len = u16::from_le_bytes([buf[lfh_off + 28], buf[lfh_off + 29]]) as usize;
            let data_off = lfh_off + 30 + lfh_name_len + lfh_extra_len;
            if data_off == payload_offset {
                buf[cur + 16..cur + 20].copy_from_slice(&crc_bytes);
                buf[lfh_off + 14..lfh_off + 18].copy_from_slice(&crc_bytes);
                return;
            }
        }
        cur = cur.saturating_add(46 + name_len + extra_len + cmt_len);
    }
}

#[test]
fn fuzz_regression_lmnkp_package_and_path_guard_deep_mutations() {
    let (unsigned_lmnkc, _, keypair) = build_valid_equilibrium_seeds();
    let pubkey = keypair.verifying_key();
    let req_sig = IntegrityPolicy::RequireSignature(pubkey);

    let manifest = format_package_manifest_json(
        env!("CARGO_PKG_VERSION"),
        0xCAFE_BABE_1234_5678,
        2,
        CORE_TOPOLOGY.topology_crc32,
        LMNKP_ENTRY_COMPILED,
        Some(LMNKP_ENTRY_SOURCE),
    );
    let mut builder = LmnkPackageBuilder::new();
    builder
        .add_manifest_json(manifest)
        .expect("manifest")
        .add_source_lmnk(b"persona Seed {}\nsession S using Seed {}\n".to_vec())
        .expect("source")
        .add_compiled_lmnkc(unsigned_lmnkc)
        .expect("compiled");
    let unsigned_pkg = builder.build().expect("unsigned pkg");
    let signed_pkg = builder.build_signed(&keypair).expect("signed pkg");

    assert_eq!(
        verify_lmnkp_integrity(&signed_pkg, &req_sig)
            .expect("signed pkg ok")
            .trust_level,
        IntegrityTrustLevel::SignatureVerified
    );

    // Record exact entry spans of the signed package so we can patch ZIP CRCs after tampering.
    let signed_spans: Vec<(usize, usize)> = LmnkPackageView::parse(&signed_pkg)
        .expect("parse signed")
        .entries()
        .iter()
        .map(|e| (e.data_offset, e.payload.len()))
        .collect();

    let mut rng = XorShift64::new(0x2266_88AA_CCEE_0002);
    let seeds = [&unsigned_pkg, &signed_pkg];

    // 1. Targeted ZIP / .lmnkp deep mutations (800 iterations).
    for iter in 0..800_usize {
        let is_signed = (iter & 1) == 1;
        let mut buf = seeds[usize::from(is_signed)].clone();
        match iter % 8 {
            0 => {
                // EOCD field mutation (last 22 bytes).
                let start = buf.len().saturating_sub(22);
                let off = start + rng.next_usize(buf.len() - start);
                buf[off] ^= 1_u8 << (rng.next_usize(8));
            }
            1 => {
                // Central Directory header mutation (method, flags, lengths, offsets).
                let eocd = buf.len().saturating_sub(22);
                let cd_off = u32::from_le_bytes([
                    buf[eocd + 16],
                    buf[eocd + 17],
                    buf[eocd + 18],
                    buf[eocd + 19],
                ]) as usize;
                if cd_off < eocd {
                    let off = cd_off + rng.next_usize(eocd - cd_off);
                    buf[off] ^= (rng.next_byte()).max(1);
                }
            }
            2 => {
                // LFH header / extra-field TLV differential mutation.
                let cap = buf.len().min(128);
                if cap > 0 {
                    let off = rng.next_usize(cap);
                    buf[off] ^= (rng.next_byte()).max(1);
                }
            }
            3 | 4 => {
                // Payload tamper + ZIP CRC32 patch so it penetrates into SHA-256 / Ed25519 layer.
                if is_signed && !signed_spans.is_empty() {
                    let (data_off, data_len) = signed_spans[rng.next_usize(signed_spans.len())];
                    if data_len > 0 && data_off + data_len <= buf.len() {
                        let pos = data_off + rng.next_usize(data_len);
                        buf[pos] ^= (rng.next_byte()).max(1);
                        patch_zip_entry_crcs(&mut buf, data_off, data_len);
                    }
                } else if !buf.is_empty() {
                    let pos = rng.next_usize(buf.len());
                    buf[pos] ^= 0xFF;
                }
            }
            5 => {
                // Truncation across LFH / payload / CD / EOCD boundaries.
                let cut = rng.next_usize(buf.len());
                buf.truncate(cut);
            }
            6 => {
                // Trailing polyglot bytes after EOCD.
                let extra = 1 + rng.next_usize(64);
                for _ in 0..extra {
                    buf.push(rng.next_byte());
                }
            }
            _ => {
                // Random byte blob.
                let len = rng.next_usize(384);
                buf = (0..len).map(|_| rng.next_byte()).collect();
            }
        }

        if let Ok(view) = LmnkPackageView::parse(&buf) {
            let _ = view.manifest_json();
            let _ = view.compiled_lmnkc();
            let _ = view.verify_integrity(&IntegrityPolicy::AllowUnsigned);
            let _ = view.verify_integrity(&req_sig);
        }
        let _ = verify_lmnkp_integrity(&buf, &IntegrityPolicy::AllowUnsigned);
        let _ = verify_lmnkp_integrity(&buf, &IntegrityPolicy::RequireContentHash);
        let _ = verify_lmnkp_integrity(&buf, &req_sig);
    }

    // 2. Targeted `validate_archive_relative_path` mutations (500 cases): Windows reserved device
    //    names, superscript digits, traversal, ADS, control chars, trailing dots/spaces, Unicode.
    let reserved_stems = [
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CONIN$",
        "CONOUT$",
        "COM0",
        "COM1",
        "COM9",
        "LPT0",
        "LPT1",
        "LPT9",
        "COM\u{00B9}",
        "COM\u{00B2}",
        "COM\u{00B3}",
        "LPT\u{00B9}",
        "LPT\u{00B2}",
        "LPT\u{00B3}",
    ];
    let suffixes = ["", ".txt", ".lmnkc", " .lmnk", ".", " ", ":stream", "\\sub"];
    for stem in reserved_stems {
        for suf in suffixes {
            let candidate = format!("dir/{stem}{suf}");
            assert!(
                validate_archive_relative_path(&candidate).is_err(),
                "must reject reserved/unsafe path '{candidate}'"
            );
        }
    }
    let alphabet = [
        'a', 'Z', '0', '/', '\\', '.', ':', ' ', '\0', '\x1f', '\x7f', '<', '>', '|', '?', '*',
        '"', '-', '_', '\u{00B9}', '\u{202E}',
    ];
    for i in 0..500_usize {
        let len = 1 + rng.next_usize(if i % 25 == 0 { 300 } else { 48 });
        let mut s = String::with_capacity(len);
        for _ in 0..len {
            s.push(alphabet[rng.next_usize(alphabet.len())]);
        }
        let _ = validate_archive_relative_path(&s);
    }
    assert!(validate_archive_relative_path(LMNKP_ENTRY_CONTENT_HASH).is_ok());
    assert!(validate_archive_relative_path(LMNKP_ENTRY_SIGNATURE).is_ok());
}

#[test]
fn fuzz_regression_lmnk_dsl_deep_mutations() {
    let poc_seed = include_str!("../../examples/dsl/limina_poc.lmnk");
    let rich_seed = concat!(
        "persona Alpha extends \"presets/base\" {\n",
        "  attractors {\n",
        "    attract \"facts/a.lmnk\" @ 0.75\n",
        "    repel \"guards/g.lmnk\" @ 0.90\n",
        "  }\n",
        "}\n",
        "target LocalGpu { vram_budget_mb = 4096 model = \"models/local.gguf\" }\n",
        "session Main using Alpha on LocalGpu { mode = strict }\n",
    );
    assert!(parse_and_validate_lmnk(poc_seed).is_ok());
    assert!(parse_and_validate_lmnk(rich_seed).is_ok());

    let seeds = [poc_seed, rich_seed];
    let splice_tokens = [
        "{",
        "}",
        "\"",
        "@",
        "=",
        "\n",
        "\0",
        "\x1b",
        "..",
        "../",
        "\\",
        "1e500",
        "-0.0",
        "NaN",
        "inf",
        "65537",
        "persona",
        "session",
        "target",
        "attractors",
        "attract",
        "repel",
        "extends",
        "using",
        "on",
        "/*unterminated",
        "//comment\n",
    ];
    let mut rng = XorShift64::new(0x3377_99BB_DDFF_0003);

    for iter in 0..600_usize {
        let base = seeds[iter & 1];
        let mut s = base.to_string();
        match iter % 6 {
            0 => {
                // Insert syntax/edge token at a valid UTF-8 char boundary.
                let chars: Vec<usize> = s.char_indices().map(|(idx, _)| idx).collect();
                let pos = chars[rng.next_usize(chars.len().max(1))];
                let tok = splice_tokens[rng.next_usize(splice_tokens.len())];
                s.insert_str(pos, tok);
            }
            1 => {
                // Delete a random slice between char boundaries.
                let chars: Vec<usize> = s.char_indices().map(|(idx, _)| idx).collect();
                if chars.len() >= 2 {
                    let a = rng.next_usize(chars.len() - 1);
                    let b = a + 1 + rng.next_usize(chars.len() - a - 1);
                    s.replace_range(chars[a]..chars[b], "");
                }
            }
            2 => {
                // Replace a substring with adversarial numeric/path/control payload.
                let chars: Vec<usize> = s.char_indices().map(|(idx, _)| idx).collect();
                if chars.len() >= 2 {
                    let a = rng.next_usize(chars.len() - 1);
                    let b = (a + 1 + rng.next_usize(16)).min(chars.len() - 1);
                    let tok = splice_tokens[rng.next_usize(splice_tokens.len())];
                    s.replace_range(chars[a]..chars[b], tok);
                }
            }
            3 => {
                // Deep brace / nested token stress.
                let depth = 1 + rng.next_usize(128);
                s = "{ ".repeat(depth) + &" }".repeat(depth);
            }
            4 => {
                // Truncation at char boundary.
                let chars: Vec<usize> = s.char_indices().map(|(idx, _)| idx).collect();
                if !chars.is_empty() {
                    let end = chars[rng.next_usize(chars.len())];
                    s.truncate(end);
                }
            }
            _ => {
                // Raw byte mutation filtered through UTF-8 lossy / strict conversion.
                let mut bytes = s.into_bytes();
                let flips = 1 + rng.next_usize(8);
                for _ in 0..flips {
                    if !bytes.is_empty() {
                        let off = rng.next_usize(bytes.len());
                        bytes[off] ^= (rng.next_byte()).max(1);
                    }
                }
                if let Ok(valid) = std::str::from_utf8(&bytes) {
                    let _ = parse_lmnk_program(valid);
                    let _ = parse_and_validate_lmnk(valid);
                }
                s = String::from_utf8_lossy(&bytes).into_owned();
            }
        }
        let _ = parse_lmnk_program(&s);
        let _ = parse_and_validate_lmnk(&s);
    }
}

struct DeterministicDummyEncoder {
    layers: usize,
    rot_len: usize,
}

impl SubtitleActivationEncoder for DeterministicDummyEncoder {
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
                    .map(|d| (((seed ^ ((l as u64) << 6) ^ (d as u64)) % 100) as f32) * 0.01 - 0.5)
                    .collect()
            })
            .collect())
    }
}

fn build_valid_timeline_v2_seed() -> Vec<u8> {
    let geom = RopeGeometry {
        layout: RopeLayout::HalfSplit,
        head_dim: 64,
        rotary_dim: 32,
        num_q_heads: 8,
        num_kv_heads: 4,
        num_layers: 4,
        layer_rope_mask: 0x0F,
    };
    let mut adapter = ModelSteeringAdapter::new(geom.clone()).expect("adapter");
    adapter.configure_ego_from_preset(LiminalEgoPreset::Medium);
    let policy = RecallPolicy::default();
    let detector = AdaptiveGopCutDetector::default();
    let mut encoder = DeterministicDummyEncoder {
        layers: geom.num_layers,
        rot_len: geom.q_rot_len(),
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
            role: AttractorRole::Persona,
            text: "Persona beta continuation".to_string(),
            energy: 1.5,
        },
        SourceTextRecord {
            frame_index: 2,
            timestamp_epoch_sec: 1_700_000_020,
            role: AttractorRole::Guard,
            text: "Guard gamma repulsion boundary".to_string(),
            energy: 1.2,
        },
    ];
    let ctx = RebakeContext {
        adapter: &adapter,
        policy: &policy,
        detector: &detector,
        centroid_layers: &[],
        up_to_frame: None,
    };
    let (mut timeline, _) =
        LmnkTimeline::bake_from_subtitle_records(&records, &ctx, &mut encoder).expect("bake");
    timeline.set_logical_pframe_topology(adapter.export_logical_pframe_topology());
    timeline
        .to_container_bytes(LMNK_CONTAINER_FLAG_COMPILED_CACHE)
        .expect("serialize timeline v2")
}

#[test]
fn fuzz_regression_lmnk_timeline_v2_deep_mutations() {
    let seed = build_valid_timeline_v2_seed();
    assert!(LmnkTimeline::from_bytes(&seed).is_ok());

    let mut rng = XorShift64::new(0x4488_CCEE_1122_0004);

    // 600 deep mutations targeting timeline v2 metadata, embedded I-frame containers,
    // and logical P-Frame topology with section + header CRC32 patched to penetrate deep decoding.
    for iter in 0..600_usize {
        let mut buf = seed.clone();
        match iter % 6 {
            0..=2 => {
                // Deep interior payload mutation + full section/header CRC recompute.
                if buf.len() > 128 {
                    let flips = 1 + rng.next_usize(6);
                    for _ in 0..flips {
                        let off = 128 + rng.next_usize(buf.len() - 128);
                        buf[off] ^= (rng.next_byte()).max(1);
                    }
                    recompute_lmnkc_section_and_header_crcs(&mut buf);
                }
            }
            3 => {
                // Splice extreme u32 counts (frame_count / text_len / topo_count = u32::MAX)
                // inside payload + patch CRCs to test overflow/OOM guards.
                if buf.len() > 192 {
                    let off = 128 + rng.next_usize(buf.len() - 132);
                    let extreme = if (iter & 1) == 0 {
                        u32::MAX
                    } else {
                        0x7FFF_FFFF
                    };
                    buf[off..off + 4].copy_from_slice(&extreme.to_le_bytes());
                    recompute_lmnkc_section_and_header_crcs(&mut buf);
                }
            }
            4 => {
                // Truncation across section table and metadata payloads.
                let new_len = rng.next_usize(buf.len());
                buf.truncate(new_len);
            }
            _ => {
                // Random fuzz blob.
                let len = rng.next_usize(512);
                buf = (0..len).map(|_| rng.next_byte()).collect();
            }
        }
        let _ = LmnkTimeline::from_bytes(&buf);
    }
}
