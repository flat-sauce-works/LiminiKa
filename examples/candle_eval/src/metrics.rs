// name: examples/candle_eval/src/metrics.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Deterministic non-LLM-judge text quality, drift, and logit divergence metrics.

use std::collections::{HashMap, HashSet};

/// Normalizes a text string into lowercase alphanumeric word tokens.
pub(crate) fn tokenize_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_ascii_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Computes Moving-Average Type-Token Ratio (`MATTR`) over a sliding word window.
///
/// Falls back to standard TTR (`unique / total`) when `words.len() <= window`.
/// Returns `0.0` for empty input or `window == 0`.
pub(crate) fn mattr(text: &str, window: usize) -> f64 {
    if window == 0 {
        return 0.0;
    }
    let words = tokenize_words(text);
    if words.is_empty() {
        return 0.0;
    }
    if words.len() <= window {
        let uniq: HashSet<&str> = words.iter().map(String::as_str).collect();
        return uniq.len() as f64 / words.len() as f64;
    }
    let mut counts: HashMap<&str, usize> = HashMap::with_capacity(window);
    for w in &words[..window] {
        *counts.entry(w.as_str()).or_insert(0) += 1;
    }
    let mut ttr_sum = counts.len() as f64 / window as f64;
    let num_windows = words.len() - window + 1;
    for idx in window..words.len() {
        let leaving = words[idx - window].as_str();
        if let Some(c) = counts.get_mut(leaving) {
            *c -= 1;
            if *c == 0 {
                counts.remove(leaving);
            }
        }
        let entering = words[idx].as_str();
        *counts.entry(entering).or_insert(0) += 1;
        ttr_sum += counts.len() as f64 / window as f64;
    }
    ttr_sum / num_windows as f64
}

/// Computes `MATTR-20` (tuned for short single-turn replies around 32-64 tokens).
pub(crate) fn mattr_20(text: &str) -> f64 {
    mattr(text, 20)
}

/// Computes `MATTR-50` (tuned for concatenated multi-turn transcripts or long replies).
pub(crate) fn mattr_50(text: &str) -> f64 {
    mattr(text, 50)
}

/// Computes `Distinct-2` (ratio of unique adjacent word bigrams to total bigrams).
///
/// Returns `0.0` when fewer than 2 words are present.
pub(crate) fn distinct_2(text: &str) -> f64 {
    let words = tokenize_words(text);
    if words.len() < 2 {
        return 0.0;
    }
    let total = words.len() - 1;
    let unique: HashSet<(&str, &str)> = words
        .windows(2)
        .map(|w| (w[0].as_str(), w[1].as_str()))
        .collect();
    unique.len() as f64 / total as f64
}

/// Computes the repeated 4-gram loop ratio `(total_4grams - unique_4grams) / total_4grams`.
///
/// Returns `0.0` when fewer than 4 words are present or no 4-gram repeats.
pub(crate) fn repetition_4gram_loop_rate(text: &str) -> f64 {
    let words = tokenize_words(text);
    if words.len() < 4 {
        return 0.0;
    }
    let total = words.len() - 3;
    let unique: HashSet<(&str, &str, &str, &str)> = words
        .windows(4)
        .map(|w| (w[0].as_str(), w[1].as_str(), w[2].as_str(), w[3].as_str()))
        .collect();
    total.saturating_sub(unique.len()) as f64 / total as f64
}

/// Returns `true` when a reply has fewer than 3 whitespace-delimited words after trimming.
pub(crate) fn is_abandoned_reply(text: &str) -> bool {
    text.split_whitespace().count() < 3
}

/// Computes the fraction of abandoned replies (`< 3` words after trimming) in `[0.0, 1.0]`.
pub(crate) fn abandonment_rate(replies: &[String]) -> f64 {
    if replies.is_empty() {
        return 0.0;
    }
    let abandoned = replies.iter().filter(|r| is_abandoned_reply(r)).count();
    abandoned as f64 / replies.len() as f64
}

/// Per-reply deterministic lexical quality metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct ReplyQualityMetrics {
    pub(crate) mattr20: f64,
    pub(crate) distinct2: f64,
    pub(crate) rep4_loop: f64,
    pub(crate) abandoned: bool,
}

impl ReplyQualityMetrics {
    pub(crate) fn measure(text: &str) -> Self {
        Self {
            mattr20: mattr_20(text),
            distinct2: distinct_2(text),
            rep4_loop: repetition_4gram_loop_rate(text),
            abandoned: is_abandoned_reply(text),
        }
    }
}

/// Aggregate quality metrics across a multi-turn dialogue, including `T_N - T_1` drift.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct DialogueQualitySummary {
    pub(crate) mean_mattr20: f64,
    pub(crate) concat_mattr50: f64,
    pub(crate) mean_distinct2: f64,
    pub(crate) mean_rep4_loop: f64,
    pub(crate) abandonment_rate: f64,
    pub(crate) delta_mattr20: f64,
    pub(crate) delta_distinct2: f64,
    pub(crate) delta_rep4_loop: f64,
}

/// Aggregates deterministic lexical metrics and `Delta = T_N - T_1` drift across replies.
pub(crate) fn summarize_dialogue_quality(replies: &[String]) -> DialogueQualitySummary {
    if replies.is_empty() {
        return DialogueQualitySummary::default();
    }
    let per_turn: Vec<ReplyQualityMetrics> = replies
        .iter()
        .map(|r| ReplyQualityMetrics::measure(r))
        .collect();
    let n = per_turn.len() as f64;
    let mean_mattr20 = per_turn.iter().map(|m| m.mattr20).sum::<f64>() / n;
    let mean_distinct2 = per_turn.iter().map(|m| m.distinct2).sum::<f64>() / n;
    let mean_rep4_loop = per_turn.iter().map(|m| m.rep4_loop).sum::<f64>() / n;
    let joined = replies.join(" ");
    let concat_mattr50 = mattr_50(&joined);
    let first = per_turn[0];
    let last = per_turn[per_turn.len() - 1];
    DialogueQualitySummary {
        mean_mattr20,
        concat_mattr50,
        mean_distinct2,
        mean_rep4_loop,
        abandonment_rate: abandonment_rate(replies),
        delta_mattr20: last.mattr20 - first.mattr20,
        delta_distinct2: last.distinct2 - first.distinct2,
        delta_rep4_loop: last.rep4_loop - first.rep4_loop,
    }
}

/// Computes numerically stable `f64` `log_sum_exp` for a logit slice.
fn log_sum_exp_f64(logits: &[f32]) -> Option<f64> {
    if logits.is_empty() {
        return None;
    }
    let mut max_v = f64::NEG_INFINITY;
    for &x in logits {
        let xf = f64::from(x);
        if xf.is_finite() && xf > max_v {
            max_v = xf;
        }
    }
    if !max_v.is_finite() {
        return None;
    }
    let sum_exp: f64 = logits
        .iter()
        .map(|&x| {
            let xf = f64::from(x);
            if xf.is_finite() {
                (xf - max_v).exp()
            } else {
                0.0
            }
        })
        .sum();
    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return None;
    }
    Some(max_v + sum_exp.ln())
}

/// Computes `(KL(P_off || P_on), max_abs_logit_delta)` using `f64` log-softmax stability.
///
/// Returns `(0.0, 0.0)` for empty or mismatched-length slices.
pub(crate) fn kl_divergence_and_max_abs_delta(off_logits: &[f32], on_logits: &[f32]) -> (f64, f64) {
    if off_logits.is_empty() || off_logits.len() != on_logits.len() {
        return (0.0, 0.0);
    }
    let (Some(lse_off), Some(lse_on)) = (log_sum_exp_f64(off_logits), log_sum_exp_f64(on_logits))
    else {
        return (0.0, 0.0);
    };
    let mut kl = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for (&lo, &ln_val) in off_logits.iter().zip(on_logits.iter()) {
        let a = f64::from(lo);
        let b = f64::from(ln_val);
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let diff = (b - a).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        let log_p = a - lse_off;
        let log_q = b - lse_on;
        let p = log_p.exp();
        if p > 0.0 {
            kl += p * (log_p - log_q);
        }
    }
    (kl.max(0.0), max_abs)
}

/// Computes `log P_on(token_id) - log P_off(token_id)` using `f64` log-softmax stability.
pub(crate) fn delta_log_prob(off_logits: &[f32], on_logits: &[f32], token_id: u32) -> Option<f64> {
    let idx = token_id as usize;
    if idx >= off_logits.len() || idx >= on_logits.len() {
        return None;
    }
    let lse_off = log_sum_exp_f64(off_logits)?;
    let lse_on = log_sum_exp_f64(on_logits)?;
    let lp_off = f64::from(off_logits[idx]) - lse_off;
    let lp_on = f64::from(on_logits[idx]) - lse_on;
    Some(lp_on - lp_off)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mattr_distinct2_rep4_and_abandonment_cover_edge_cases() {
        assert_eq!(mattr("", 20), 0.0);
        assert_eq!(mattr("hello world", 0), 0.0);
        assert!((mattr_20("one two three four") - 1.0).abs() < 1e-12);
        assert!((mattr_20("same same same same") - 0.25).abs() < 1e-12);

        // 25 unique words -> MATTR-20 == 1.0, MATTR-50 falls back to full TTR == 1.0.
        let unique_25: String = (0..25)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        assert!((mattr_20(&unique_25) - 1.0).abs() < 1e-12);
        assert!((mattr_50(&unique_25) - 1.0).abs() < 1e-12);

        // Exact sliding window verification for window=3 on "a b a b c" (windows: {a,b}, {b,a}, {a,b,c}).
        let w3 = mattr("a b a b c", 3);
        let expected_w3 = (2.0 / 3.0 + 2.0 / 3.0 + 3.0 / 3.0) / 3.0;
        assert!((w3 - expected_w3).abs() < 1e-12);

        // Distinct-2
        assert_eq!(distinct_2("solo"), 0.0);
        assert!((distinct_2("alpha beta gamma delta") - 1.0).abs() < 1e-12);
        assert!((distinct_2("a b a b") - (2.0 / 3.0)).abs() < 1e-12);

        // 4-gram repetition loop rate
        assert_eq!(repetition_4gram_loop_rate("one two three"), 0.0);
        assert_eq!(repetition_4gram_loop_rate("a b c d e f"), 0.0);
        // "a b c d a b c d" -> 5 4-grams: abcd, bcda, cdab, dabc, abcd -> 4 unique, 1 repeat => 0.2
        assert!((repetition_4gram_loop_rate("a b c d a b c d") - 0.2).abs() < 1e-12);
        // Constant loop "x x x x x x" -> 3 4-grams, 1 unique => 2/3
        assert!((repetition_4gram_loop_rate("x x x x x x") - (2.0 / 3.0)).abs() < 1e-12);

        // Abandonment rate (< 3 words after trimming)
        assert!(is_abandoned_reply(""));
        assert!(is_abandoned_reply("   "));
        assert!(is_abandoned_reply("two words"));
        assert!(!is_abandoned_reply("three clear words"));
        let replies = vec![
            "".to_string(),
            "ok".to_string(),
            "this is a complete sentence".to_string(),
            "another valid reply here".to_string(),
        ];
        assert!((abandonment_rate(&replies) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn kl_divergence_and_delta_log_prob_are_numerically_stable() {
        let base = [1000.0_f32, 999.0, 998.0, 995.0];
        let (kl_same, max_same) = kl_divergence_and_max_abs_delta(&base, &base);
        assert!(kl_same.abs() < 1e-12);
        assert_eq!(max_same, 0.0);

        let shifted = [1000.0_f32, 999.0, 998.0, 997.0];
        let (kl_diff, max_diff) = kl_divergence_and_max_abs_delta(&base, &shifted);
        assert!(kl_diff > 0.0 && kl_diff.is_finite());
        assert!((max_diff - 2.0).abs() < 1e-6);

        // Increasing logit[3] increases P(3) and decreases P(0).
        let dlp3 = delta_log_prob(&base, &shifted, 3).expect("valid token 3");
        let dlp0 = delta_log_prob(&base, &shifted, 0).expect("valid token 0");
        assert!(dlp3 > 0.0);
        assert!(dlp0 < 0.0);
        assert!(delta_log_prob(&base, &shifted, 99).is_none());
    }

    #[test]
    fn dialogue_quality_summary_computes_first_to_last_drift() {
        let replies = vec![
            "Alpha beta gamma delta epsilon zeta eta theta.".to_string(),
            "loop loop loop loop loop loop loop loop.".to_string(),
        ];
        let summary = summarize_dialogue_quality(&replies);
        assert!(summary.delta_mattr20 < 0.0);
        assert!(summary.delta_distinct2 < 0.0);
        assert!(summary.delta_rep4_loop > 0.0);
        assert_eq!(summary.abandonment_rate, 0.0);
    }
}
