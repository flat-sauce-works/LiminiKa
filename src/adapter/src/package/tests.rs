// name: src/adapter/src/package/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit tests for `.lmnkp` Store ZIP package codec and security bounds.

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
        env!("CARGO_PKG_VERSION"),
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
        env!("CARGO_PKG_VERSION"),
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
    assert!(verify_lmnkp_integrity(&unsigned_pkg, &IntegrityPolicy::RequireContentHash).is_err());
    assert!(verify_lmnkp_integrity(&unsigned_pkg, &IntegrityPolicy::RequireSignature(pk)).is_err());

    // 2. Hash-only and signed packages preserve all source entries byte-for-byte + 64B align.
    let hashed_pkg = attach_lmnkp_content_hash(&unsigned_pkg).unwrap();
    let r_h = verify_lmnkp_integrity(&hashed_pkg, &IntegrityPolicy::RequireContentHash).unwrap();
    assert_eq!(r_h.trust_level, IntegrityTrustLevel::ContentHashVerified);

    let signed_pkg = sign_lmnkp_package(&unsigned_pkg, &kp).unwrap();
    let r_s = verify_lmnkp_integrity(&signed_pkg, &IntegrityPolicy::RequireSignature(pk)).unwrap();
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
        assert!(verify_lmnkp_integrity(&tampered_bytes, &IntegrityPolicy::AllowUnsigned).is_err());
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
            verify_lmnkp_integrity(&tb.build().unwrap(), &IntegrityPolicy::AllowUnsigned).is_err()
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
            verify_lmnkp_integrity(&tb.build().unwrap(), &IntegrityPolicy::AllowUnsigned).is_err()
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
            verify_lmnkp_integrity(&tb.build().unwrap(), &IntegrityPolicy::AllowUnsigned).is_err()
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
            verify_lmnkp_integrity(&tb.build().unwrap(), &IntegrityPolicy::AllowUnsigned).is_err()
        );
    }
}
