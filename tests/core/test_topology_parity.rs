// name: tests/core/test_topology_parity.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure Rust integration test verifying compile-time `CORE_TOPOLOGY` (`LiminalTopologyMatrix`)
//! CRC32 parity, 5-layer Y-shaped DAG directory/import invariants, and `BitCut8` admission rules
//! with zero external dependencies.

use std::fs;
use std::path::{Path, PathBuf};

use liminika_core::foundation::{compute_crc32_ieee, Pod};
use liminika_core::liminal::{
    BitCut8, LMNK_ABI_COMPAT_MASK, LMNK_ABI_COMPAT_VERSION_HEX, LMNK_ABI_VERSION_HEX,
};
use liminika_core::runtime::{
    CORE_TOPOLOGY, LMNK_TOPOLOGY_CHANNEL_COUNT, LMNK_TOPOLOGY_DAG_EDGE_MATRIX,
    LMNK_TOPOLOGY_LAYER_COUNT, LMNK_TOPOLOGY_MAGIC,
};

fn collect_non_test_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = fs::read_dir(dir).expect("read_dir");
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            if path.file_name().and_then(|n| n.to_str()) == Some("tests") {
                continue;
            }
            collect_non_test_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs")
            && path.file_name().and_then(|n| n.to_str()) != Some("tests.rs")
        {
            out.push(path);
        }
    }
}

fn non_test_body(src: &str) -> &str {
    for (idx, line) in src.match_indices('\n') {
        let _ = (idx, line);
    }
    if let Some(pos) = src.find("\n#[cfg(test)]") {
        &src[..pos]
    } else {
        src
    }
}

#[test]
fn test_core_topology_matrix_and_filesystem_dag_parity() {
    // 1. CRC32 & canonical field parity
    assert_eq!(core::mem::size_of_val(&CORE_TOPOLOGY), 64);
    assert_eq!(core::mem::align_of_val(&CORE_TOPOLOGY), 64);
    assert_eq!(CORE_TOPOLOGY.magic, LMNK_TOPOLOGY_MAGIC);
    assert_eq!(LMNK_ABI_COMPAT_MASK, 0xFFFF_FF00);
    assert_eq!(
        LMNK_ABI_COMPAT_VERSION_HEX,
        LMNK_ABI_VERSION_HEX & LMNK_ABI_COMPAT_MASK
    );
    assert_eq!(CORE_TOPOLOGY.version, LMNK_ABI_COMPAT_VERSION_HEX);
    assert_eq!(CORE_TOPOLOGY.layer_count, LMNK_TOPOLOGY_LAYER_COUNT);
    assert_eq!(CORE_TOPOLOGY.channel_count, LMNK_TOPOLOGY_CHANNEL_COUNT);
    assert_eq!(CORE_TOPOLOGY.dag_edge_matrix, LMNK_TOPOLOGY_DAG_EDGE_MATRIX);
    assert!(CORE_TOPOLOGY.verify_crc32());

    let prefix = CORE_TOPOLOGY.to_prefix_le_bytes();
    let runtime_crc = compute_crc32_ieee(&prefix);
    assert_eq!(CORE_TOPOLOGY.topology_crc32, runtime_crc);
    if cfg!(target_endian = "little") {
        assert_eq!(&CORE_TOPOLOGY.as_bytes()[..60], &prefix[..]);
    }

    // 2. Filesystem 5-layer directory & retired-path parity
    let core_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let core_src = core_root.join("src");
    let mut layer_dirs: Vec<String> = fs::read_dir(&core_src)
        .expect("read core/src")
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.is_dir() {
                Some(p.file_name()?.to_string_lossy().into_owned())
            } else {
                None
            }
        })
        .collect();
    layer_dirs.sort();
    assert_eq!(
        layer_dirs,
        vec!["drivers", "foundation", "liminal", "runtime", "stigmergy"]
    );
    assert_eq!(layer_dirs.len(), usize::from(CORE_TOPOLOGY.layer_count));
    assert!(core_src.join("stigmergy/anchor/mod.rs").is_file());
    assert!(core_src.join("runtime/topology.rs").is_file());
    assert!(!core_src.join("stigmergy/crystal").exists());
    assert!(!core_src.join("stigmergy/attractor").exists());
    assert!(!core_root.join("../kernels").exists());

    // 3. Y-shaped DAG edge bitmask parity & cross-root / driver isolation
    assert_eq!(CORE_TOPOLOGY.dag_edge_matrix[0], 0b0000_0000);
    assert_eq!(CORE_TOPOLOGY.dag_edge_matrix[1], 0b0000_0001);
    assert_eq!(CORE_TOPOLOGY.dag_edge_matrix[2], 0b0000_0011);
    assert_eq!(CORE_TOPOLOGY.dag_edge_matrix[3], 0b0000_0111);
    assert_eq!(CORE_TOPOLOGY.dag_edge_matrix[4], 0b0000_1111);
    assert_eq!(&CORE_TOPOLOGY.dag_edge_matrix[5..8], &[0, 0, 0]);

    let mut foundation_files = Vec::new();
    collect_non_test_rs_files(&core_src.join("foundation"), &mut foundation_files);
    for path in foundation_files {
        let src = fs::read_to_string(&path).expect("read foundation file");
        let prod = non_test_body(&src);
        assert!(
            !prod.contains("liminal::axiom"),
            "foundation file {} must not import liminal::axiom",
            path.display()
        );
        assert!(
            !prod.contains("crate::stigmergy")
                && !prod.contains("crate::drivers")
                && !prod.contains("crate::runtime"),
            "foundation file {} must not depend on upper layers",
            path.display()
        );
    }

    let mut spine_files = Vec::new();
    collect_non_test_rs_files(&core_src.join("drivers/spine"), &mut spine_files);
    for path in spine_files {
        let src = fs::read_to_string(&path).expect("read spine file");
        let prod = non_test_body(&src);
        assert!(
            !prod.contains("crate::drivers::ego"),
            "drivers/spine file {} must stay isolated from drivers/ego",
            path.display()
        );
    }

    let mut ego_files = Vec::new();
    collect_non_test_rs_files(&core_src.join("drivers/ego"), &mut ego_files);
    for path in ego_files {
        let src = fs::read_to_string(&path).expect("read ego file");
        let prod = non_test_body(&src);
        assert!(
            !prod.contains("crate::drivers::spine"),
            "drivers/ego file {} must stay isolated from drivers/spine",
            path.display()
        );
    }

    // 4. BitCut8 admission invariants
    assert!(CORE_TOPOLOGY.admits_channel_branchless(0, BitCut8::split(0x01, 0xFF)));
    assert!(!CORE_TOPOLOGY.admits_channel_branchless(0, BitCut8::split(0x02, 0xFF)));
    assert!(!CORE_TOPOLOGY.admits_channel_branchless(7, BitCut8::split(0x09, 0xFF)));
    assert!(!CORE_TOPOLOGY.admits_channel_branchless(8, BitCut8::split(0xFF, 0xFF)));
}
