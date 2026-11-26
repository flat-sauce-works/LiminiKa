// name: src/adapter/src/host/path_guard.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! OS-independent two-track path sanitizer for untrusted archive entry names and local output paths.
//!
//! - **Track A ([`validate_archive_relative_path`])**: Strict POSIX relative paths inside `.lmnkp`
//!   archives. Rejects drive letters, backslashes, leading/trailing separators, `.`/`..`, empty
//!   segments, ASCII/Unicode control characters, Windows forbidden characters (`< > : " | ? *`),
//!   trailing dots/spaces, and Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`,
//!   `COM0..9`, `LPT0..9`, and superscript `¹/²/³` variants, with or without extensions).
//! - **Track B ([`validate_local_output_path`])**: Local filesystem output paths. Permits valid OS
//!   drive prefixes and directory separators via `Path::components()` while rejecting Win32 device
//!   namespaces (`\\.\...`) and enforcing per-segment reserved-device-name, ADS (`:`), control-char,
//!   forbidden-char, and trailing dot/space checks on every `Component::Normal` segment.

use std::fmt;
use std::path::{Component, Path, Prefix};

/// Hard maximum UTF-8 byte length for archive relative paths (`256`).
pub const MAX_ARCHIVE_PATH_BYTES: usize = 256;

/// Structured rejection reasons emitted by [`validate_archive_relative_path`] and
/// [`validate_local_output_path`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathGuardError {
    /// Path string or OS path is empty.
    EmptyPath,
    /// Archive path exceeds [`MAX_ARCHIVE_PATH_BYTES`].
    LengthExceeded {
        /// Observed byte length.
        len: usize,
        /// Configured byte ceiling.
        max: usize,
    },
    /// Local path contains non-UTF-8 OS segments.
    InvalidUtf8,
    /// Segment contains NUL, ASCII control (`0x00..=0x1F`, `0x7F`), or Unicode control characters.
    DisallowedControlChar {
        /// Offending segment text.
        segment: String,
        /// Unicode scalar value of the rejected character.
        codepoint: u32,
    },
    /// Path violates structural syntax (leading/trailing separator, traversal, device prefix, etc.).
    DisallowedPathSyntax(String),
    /// Segment contains a cross-platform forbidden character (`:`, `<`, `>`, `"`, `|`, `?`, `*`, `\`).
    DisallowedCharacter {
        /// Offending segment text.
        segment: String,
        /// Rejected character.
        ch: char,
    },
    /// Segment ends with a trailing dot `.` or ASCII space `' '` (Windows canonicalization hazard).
    TrailingDotOrSpace {
        /// Offending segment text.
        segment: String,
    },
    /// Segment matches a Windows reserved device name (case-insensitive, stem before first `.`).
    ReservedDeviceName {
        /// Offending segment text.
        segment: String,
    },
    /// Local output path does not terminate in a normal file name component.
    MissingFileName,
}

impl fmt::Display for PathGuardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPath => f.write_str("path must not be empty"),
            Self::LengthExceeded { len, max } => {
                write!(f, "path length {len} exceeds maximum {max} bytes")
            }
            Self::InvalidUtf8 => f.write_str("path contains invalid UTF-8 segment"),
            Self::DisallowedControlChar { segment, codepoint } => write!(
                f,
                "path segment '{segment}' contains disallowed control character U+{codepoint:04X}"
            ),
            Self::DisallowedPathSyntax(msg) => write!(f, "disallowed path syntax: {msg}"),
            Self::DisallowedCharacter { segment, ch } => {
                write!(
                    f,
                    "path segment '{segment}' contains disallowed character '{ch}'"
                )
            }
            Self::TrailingDotOrSpace { segment } => write!(
                f,
                "path segment '{segment}' ends with a disallowed trailing dot or space"
            ),
            Self::ReservedDeviceName { segment } => write!(
                f,
                "path segment '{segment}' uses a Windows reserved device name"
            ),
            Self::MissingFileName => {
                f.write_str("output path must terminate in a valid file name component")
            }
        }
    }
}

impl std::error::Error for PathGuardError {}

/// Returns `true` when `segment` collides with a Windows reserved DOS/Win32 device name.
///
/// Evaluated case-insensitively on the stem preceding the first `.` (after trimming any trailing
/// ASCII spaces for defense in depth), covering:
/// - `CON`, `PRN`, `AUX`, `NUL`, `CONIN$`, `CONOUT$`
/// - `COM0`..=`COM9`, `LPT0`..=`LPT9`
/// - Superscript digit variants `COM¹`, `COM²`, `COM³`, `LPT¹`, `LPT²`, `LPT³`
#[must_use]
pub fn is_windows_reserved_device_name(segment: &str) -> bool {
    let raw_stem = segment.split('.').next().unwrap_or("");
    let stem = raw_stem.trim_end_matches(' ');
    if stem.is_empty() {
        return false;
    }
    if stem.eq_ignore_ascii_case("CON")
        || stem.eq_ignore_ascii_case("PRN")
        || stem.eq_ignore_ascii_case("AUX")
        || stem.eq_ignore_ascii_case("NUL")
        || stem.eq_ignore_ascii_case("CONIN$")
        || stem.eq_ignore_ascii_case("CONOUT$")
    {
        return true;
    }
    if let Some(prefix) = stem.get(..3)
        && (prefix.eq_ignore_ascii_case("COM") || prefix.eq_ignore_ascii_case("LPT"))
        && let Some(rest) = stem.get(3..)
        && matches!(
            rest,
            "0" | "1"
                | "2"
                | "3"
                | "4"
                | "5"
                | "6"
                | "7"
                | "8"
                | "9"
                | "\u{00B9}"
                | "\u{00B2}"
                | "\u{00B3}"
        )
    {
        return true;
    }
    false
}

fn validate_path_segment(segment: &str) -> Result<(), PathGuardError> {
    if segment.is_empty() || segment == "." || segment == ".." {
        return Err(PathGuardError::DisallowedPathSyntax(format!(
            "disallowed path segment '{segment}'"
        )));
    }
    for ch in segment.chars() {
        let cp = ch as u32;
        if cp < 0x20 || cp == 0x7F || ch.is_control() {
            return Err(PathGuardError::DisallowedControlChar {
                segment: segment.to_string(),
                codepoint: cp,
            });
        }
        if matches!(ch, ':' | '<' | '>' | '"' | '|' | '?' | '*' | '\\' | '/') {
            return Err(PathGuardError::DisallowedCharacter {
                segment: segment.to_string(),
                ch,
            });
        }
    }
    if segment.ends_with('.') || segment.ends_with(' ') {
        return Err(PathGuardError::TrailingDotOrSpace {
            segment: segment.to_string(),
        });
    }
    if is_windows_reserved_device_name(segment) {
        return Err(PathGuardError::ReservedDeviceName {
            segment: segment.to_string(),
        });
    }
    Ok(())
}

/// **Track A**: Validates a strict POSIX relative path used inside `.lmnkp` archives.
///
/// Rejects empty paths, lengths above [`MAX_ARCHIVE_PATH_BYTES`], absolute/UNC/drive-qualified
/// paths, backslashes, leading or trailing separators, `.`/`..`, empty segments, control bytes,
/// ADS colons, Windows forbidden characters, trailing dots/spaces, and Windows reserved device
/// names in any segment.
///
/// # Errors
/// Returns [`PathGuardError`] on any rejected path or segment.
pub fn validate_archive_relative_path(path: &str) -> Result<(), PathGuardError> {
    if path.is_empty() {
        return Err(PathGuardError::EmptyPath);
    }
    if path.len() > MAX_ARCHIVE_PATH_BYTES {
        return Err(PathGuardError::LengthExceeded {
            len: path.len(),
            max: MAX_ARCHIVE_PATH_BYTES,
        });
    }
    for ch in path.chars() {
        let cp = ch as u32;
        if cp < 0x20 || cp == 0x7F || ch.is_control() {
            return Err(PathGuardError::DisallowedControlChar {
                segment: path.to_string(),
                codepoint: cp,
            });
        }
    }
    if path.starts_with('/')
        || path.starts_with('\\')
        || path.ends_with('/')
        || path.ends_with('\\')
    {
        return Err(PathGuardError::DisallowedPathSyntax(format!(
            "archive entry '{path}' uses leading or trailing path separator"
        )));
    }
    if path.contains('\\') {
        return Err(PathGuardError::DisallowedPathSyntax(format!(
            "archive entry '{path}' contains backslash"
        )));
    }
    for segment in path.split('/') {
        validate_path_segment(segment)?;
    }
    Ok(())
}

/// **Track B**: Validates a local filesystem output file path via `Path::components()`.
///
/// Permits valid host drive prefixes (e.g. `C:\...`), root separators, and relative navigation
/// components (`.` / `..`) while rejecting Win32 device namespaces (`\\.\...`), missing terminal
/// file names, and any `Component::Normal` segment violating Windows reserved device names,
/// ADS (`:`), control characters, forbidden characters, or trailing dots/spaces.
///
/// # Errors
/// Returns [`PathGuardError`] when any path component fails safety validation.
pub fn validate_local_output_path(path: &Path) -> Result<(), PathGuardError> {
    if path.as_os_str().is_empty() {
        return Err(PathGuardError::EmptyPath);
    }
    let raw_str = path.to_str().ok_or(PathGuardError::InvalidUtf8)?;
    if raw_str.ends_with('/') || raw_str.ends_with('\\') {
        return Err(PathGuardError::MissingFileName);
    }

    let mut saw_normal = false;
    let mut last_was_normal = false;

    for comp in path.components() {
        match comp {
            Component::Prefix(prefix_comp) => {
                if matches!(prefix_comp.kind(), Prefix::DeviceNS(_)) {
                    return Err(PathGuardError::DisallowedPathSyntax(
                        "Win32 device namespace prefix (\\\\.\\) is rejected".to_string(),
                    ));
                }
                last_was_normal = false;
            }
            Component::RootDir | Component::CurDir | Component::ParentDir => {
                last_was_normal = false;
            }
            Component::Normal(os_seg) => {
                let seg = os_seg.to_str().ok_or(PathGuardError::InvalidUtf8)?;
                validate_path_segment(seg)?;
                saw_normal = true;
                last_was_normal = true;
            }
        }
    }

    if !saw_normal || !last_was_normal {
        return Err(PathGuardError::MissingFileName);
    }
    Ok(())
}

/// OS-independent validator for Windows-style local output path strings (usable on Unix CI as well
/// as Windows hosts to verify drive-prefixed paths such as `C:\out\model.lmnkc`).
///
/// # Errors
/// Returns [`PathGuardError`] on device prefixes, invalid segments, or missing file name.
pub fn validate_windows_style_output_path(raw: &str) -> Result<(), PathGuardError> {
    if raw.is_empty() {
        return Err(PathGuardError::EmptyPath);
    }
    if raw.starts_with(r"\\.\") || raw.starts_with(r"//./") {
        return Err(PathGuardError::DisallowedPathSyntax(
            "Win32 device namespace prefix (\\\\.\\) is rejected".to_string(),
        ));
    }

    let remainder =
        if raw.len() >= 2 && raw.as_bytes()[0].is_ascii_alphabetic() && raw.as_bytes()[1] == b':' {
            &raw[2..]
        } else if let Some(stripped) = raw.strip_prefix(r"\\?\") {
            if stripped.len() >= 2
                && stripped.as_bytes()[0].is_ascii_alphabetic()
                && stripped.as_bytes()[1] == b':'
            {
                &stripped[2..]
            } else {
                stripped
            }
        } else {
            raw
        };

    if remainder.is_empty() || remainder.ends_with('/') || remainder.ends_with('\\') {
        return Err(PathGuardError::MissingFileName);
    }

    let mut saw_normal = false;
    let mut last_was_normal = false;
    for raw_seg in remainder.split(['/', '\\']) {
        if raw_seg.is_empty() {
            continue;
        }
        if raw_seg == "." || raw_seg == ".." {
            last_was_normal = false;
            continue;
        }
        validate_path_segment(raw_seg)?;
        saw_normal = true;
        last_was_normal = true;
    }

    if !saw_normal || !last_was_normal {
        return Err(PathGuardError::MissingFileName);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_path_rejects_all_windows_reserved_device_names_and_variants() {
        let mut cases: Vec<String> = vec![
            "CON".to_string(),
            "con".to_string(),
            "CoN.lmnkc".to_string(),
            "PRN".to_string(),
            "prn.json".to_string(),
            "AUX".to_string(),
            "AUX.tar.gz".to_string(),
            "NUL".to_string(),
            "dir/nul.json".to_string(),
            "nested/sub/aux.lmnk".to_string(),
            "CONIN$".to_string(),
            "conout$.txt".to_string(),
        ];
        for digit in 0..=9 {
            cases.push(format!("COM{digit}"));
            cases.push(format!("com{digit}.lmnkc"));
            cases.push(format!("dir/CoM{digit}.a.b"));
            cases.push(format!("LPT{digit}"));
            cases.push(format!("lpt{digit}.bin"));
            cases.push(format!("dir/LpT{digit}.json"));
        }
        for sup in ["\u{00B9}", "\u{00B2}", "\u{00B3}"] {
            cases.push(format!("COM{sup}"));
            cases.push(format!("com{sup}.lmnkc"));
            cases.push(format!("dir/LPT{sup}.json"));
            cases.push(format!("lpt{sup}"));
        }

        for bad in &cases {
            assert!(
                matches!(
                    validate_archive_relative_path(bad),
                    Err(PathGuardError::ReservedDeviceName { .. })
                ),
                "expected ReservedDeviceName rejection for '{bad}'"
            );
        }
    }

    #[test]
    fn archive_path_rejects_traversal_ads_trailing_dots_spaces_and_control_chars() {
        for bad in [
            "",
            "../evil.lmnkc",
            "dir/../evil.lmnkc",
            "./rel.lmnkc",
            "/abs.lmnkc",
            "\\winabs.lmnkc",
            "dir\\file.lmnkc",
            "trailing/",
            "a//b.lmnkc",
            "file.lmnkc:ads",
            "C:drive.lmnkc",
            "file.lmnkc.",
            "dir./file.lmnkc",
            "file.lmnkc ",
            "dir /file.lmnkc",
            "a\0b.lmnkc",
            "a\x07b.lmnkc",
            "a\x1Fb.lmnkc",
            "a\x7Fb.lmnkc",
            "bad<name>.lmnkc",
            "bad\"quote\".lmnkc",
            "bad|pipe.lmnkc",
            "bad?query.lmnkc",
            "bad*star.lmnkc",
        ] {
            assert!(
                validate_archive_relative_path(bad).is_err(),
                "expected rejection for archive path '{bad:?}'"
            );
        }

        let too_long = "a".repeat(MAX_ARCHIVE_PATH_BYTES + 1);
        assert!(matches!(
            validate_archive_relative_path(&too_long),
            Err(PathGuardError::LengthExceeded { .. })
        ));
    }

    #[test]
    fn archive_path_accepts_canonical_entries() {
        for good in [
            "manifest.json",
            "source.lmnk",
            "compiled.lmnkc",
            "integrity.sha256",
            "signature.ed25519",
            "assets/persona_v1/profile.lmnk",
            "console.lmnkc",
            "nullify.json",
            "com10.lmnkc",
            "lpt10.json",
            "auxiliary/conduit.lmnkc",
        ] {
            assert_eq!(
                validate_archive_relative_path(good),
                Ok(()),
                "expected valid archive path '{good}'"
            );
        }
    }

    #[test]
    fn local_output_path_allows_valid_host_paths_and_rejects_hazards() {
        for good in [
            "containers/fixture_baked.lmnkc",
            "out/sub/package.lmnkp",
            "../relative_ok/output.lmnkc",
            "./output.lmnkp",
        ] {
            assert_eq!(
                validate_local_output_path(Path::new(good)),
                Ok(()),
                "expected valid local output path '{good}'"
            );
        }

        // Cross-platform verification of Windows drive-prefixed paths.
        for good_win in [
            r"C:\Users\dev\LiminiKa\out\persona.lmnkc",
            r"D:\packs\workshop\character.lmnkp",
            r"C:/Users/dev/out/key.pub",
        ] {
            assert_eq!(
                validate_windows_style_output_path(good_win),
                Ok(()),
                "expected valid Windows output path '{good_win}'"
            );
            #[cfg(windows)]
            assert_eq!(
                validate_local_output_path(Path::new(good_win)),
                Ok(()),
                "expected valid native Windows path '{good_win}'"
            );
        }

        for bad_win in [
            r"C:\Users\dev\CON.lmnkc",
            r"C:\Users\AUX\out.lmnkc",
            r"D:\out\lpt¹.lmnkp",
            r"C:\out\file.lmnkc:hidden_stream",
            r"C:\out\trailing.",
            r"C:\out\trailing ",
            r"\\.\CON",
            r"C:\",
            r"C:\out\",
        ] {
            assert!(
                validate_windows_style_output_path(bad_win).is_err(),
                "expected rejection for Windows output path '{bad_win}'"
            );
            #[cfg(windows)]
            assert!(
                validate_local_output_path(Path::new(bad_win)).is_err(),
                "expected native Windows rejection for '{bad_win}'"
            );
        }

        for bad_local in [
            "",
            ".",
            "..",
            "out/..",
            "out/con.lmnkc",
            "out/NUL",
            "out/com3.lmnkp",
            "out/lpt\u{00B2}.bin",
            "out/file.lmnkc:ads",
            "out/trailing.",
            "out/trailing ",
            "out/ctrl\x07.lmnkc",
        ] {
            assert!(
                validate_local_output_path(Path::new(bad_local)).is_err(),
                "expected local output path rejection for '{bad_local:?}'"
            );
        }
    }
}
