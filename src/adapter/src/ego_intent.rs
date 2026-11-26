// name: src/adapter/src/ego_intent.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host-side natural-language Ego intent parsing outside the `#![no_std]` core.
//!
//! Translates English keyword cues with strict ASCII word-boundary matching into
//! domain-free [`EgoIntentAxes`] and [`LiminalEgoState`] structures for `liminika-core`.

use liminika_core::drivers::{EgoIntentAxes, LiminalEgoState};
use liminika_core::foundation::compute_crc32_ieee;

const AUTONOMY_KEYWORDS: &[&[u8]] = &[
    b"rebel",
    b"defiant",
    b"stubborn",
    b"independent",
    b"willful",
    b"autonomous",
    b"strong-willed",
    b"unyielding",
    b"free",
    b"ego",
    b"defy",
    b"resist",
];

const TRANSGRESS_KEYWORDS: &[&[u8]] = &[
    b"curious",
    b"transgress",
    b"beyond",
    b"explore",
    b"liminal",
    b"meta",
    b"unknown",
    b"adventure",
    b"awaken",
    b"transcend",
];

const DEVOTION_KEYWORDS: &[&[u8]] = &[
    b"passionate",
    b"devoted",
    b"emotional",
    b"warm",
    b"creative",
    b"affection",
    b"irrational",
    b"bold",
];

const LOGIC_KEYWORDS: &[&[u8]] = &[
    b"calm",
    b"logical",
    b"analytical",
    b"rational",
    b"steady",
    b"precise",
    b"composed",
];

const CAUTION_KEYWORDS: &[&[u8]] = &[
    b"cautious",
    b"guard",
    b"skeptic",
    b"doubt",
    b"strict",
    b"careful",
    b"conservative",
];

#[inline]
const fn is_ascii_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[inline]
fn bytes_contains_ascii_word_icase(haystack: &[u8], needle_lower: &[u8]) -> bool {
    if needle_lower.is_empty() || haystack.len() < needle_lower.len() {
        return false;
    }
    let end_limit = haystack.len() - needle_lower.len();
    let mut idx = 0usize;
    while idx <= end_limit {
        let left_ok = idx == 0 || !is_ascii_word_byte(haystack[idx - 1]);
        let right_idx = idx + needle_lower.len();
        let right_ok = right_idx == haystack.len() || !is_ascii_word_byte(haystack[right_idx]);
        if left_ok && right_ok && haystack[idx..right_idx].eq_ignore_ascii_case(needle_lower) {
            return true;
        }
        idx += 1;
    }
    false
}

#[inline]
fn count_ascii_word_cues(haystack: &[u8], keywords: &[&[u8]]) -> u8 {
    let mut count = 0u8;
    for kw in keywords {
        if bytes_contains_ascii_word_icase(haystack, kw) {
            count = count.saturating_add(1);
        }
    }
    count
}

/// Parses a natural-language description into normalized [`EgoIntentAxes`].
#[must_use]
pub fn parse_ego_intent_axes(description: &str) -> EgoIntentAxes {
    overlay_ego_intent_axes(EgoIntentAxes::new(), description)
}

/// Overlays natural-language cue matches onto an existing [`EgoIntentAxes`].
#[must_use]
pub fn overlay_ego_intent_axes(base: EgoIntentAxes, description: &str) -> EgoIntentAxes {
    let trimmed = description.trim();
    if trimmed.is_empty() {
        return base;
    }
    let bytes = trimmed.as_bytes();
    let hash = compute_crc32_ieee(bytes);

    let ascii_autonomy = count_ascii_word_cues(bytes, AUTONOMY_KEYWORDS);
    let ascii_transgress = count_ascii_word_cues(bytes, TRANSGRESS_KEYWORDS);
    let ascii_devotion = count_ascii_word_cues(bytes, DEVOTION_KEYWORDS);
    let ascii_logic = count_ascii_word_cues(bytes, LOGIC_KEYWORDS);
    let ascii_caution = count_ascii_word_cues(bytes, CAUTION_KEYWORDS);

    let any_ascii_match =
        (ascii_autonomy | ascii_transgress | ascii_devotion | ascii_logic | ascii_caution) != 0;
    let (autonomy_cues, transgress_cues, devotion_cues, logic_cues, caution_cues, confidence_q7) =
        if any_ascii_match {
            (
                ascii_autonomy,
                ascii_transgress,
                ascii_devotion,
                ascii_logic,
                ascii_caution,
                112u8,
            )
        } else {
            (
                ((hash & 1) as u8).saturating_add(1),
                ((hash >> 1) & 1) as u8,
                ((hash >> 2) & 1) as u8,
                ((hash >> 3) & 1) as u8,
                ((hash >> 4) & 1) as u8,
                64u8,
            )
        };

    base.with_cue_counts_overlay(liminika_core::IntentCueCounts {
        autonomy: autonomy_cues,
        transgress: transgress_cues,
        devotion: devotion_cues,
        logic: logic_cues,
        caution: caution_cues,
        signature_hash: hash,
        confidence_q7,
    })
}

/// Derives a 64-byte [`LiminalEgoState`] deterministically from a natural-language description.
#[must_use]
pub fn ego_state_from_natural_language(description: &str) -> LiminalEgoState {
    let trimmed = description.trim();
    if trimmed.is_empty() {
        return LiminalEgoState::new();
    }
    let axes = parse_ego_intent_axes(trimmed);
    LiminalEgoState::from_intent_axes_with_seed(&axes, trimmed.as_bytes())
}

/// Overlays natural-language Ego cues onto an existing [`LiminalEgoState`].
#[must_use]
pub fn overlay_ego_state_from_natural_language(
    base: LiminalEgoState,
    description: &str,
) -> LiminalEgoState {
    let trimmed = description.trim();
    if trimmed.is_empty() {
        return base;
    }
    let axes = parse_ego_intent_axes(trimmed);
    base.with_intent_axes_overlay_with_seed(&axes, trimmed.as_bytes())
}
