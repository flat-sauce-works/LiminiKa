// name: src/core/src/traits/telemetry/ledger.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic adaptive telemetry scratchpad (`FastPathLedger`) over `gcso_ledger_slot_t`.

use super::*;

// ===================================================================
// 7. Dynamic Adaptive Extension Scratchpad (`FastPathLedger`)
// ===================================================================

/// Lv.1 Capability Trait for the 64-byte Dynamic Adaptive Extension Scratchpad (FastPathLedger):
/// $\mathcal{O}(1)$ fast-path tagged pointer shortcuts, 8-byte circular telemetry mini-ledger,
/// and dynamic parameter auto-tuning.
pub trait FastPathLedger: BranchlessKernel {
    /// Resets active telemetry mini ledger and cache counters in-place.
    ///
    /// # Errors
    /// Returns an error status only if a custom implementor rejects resetting a locked slot.
    #[inline(always)]
    fn reset_telemetry(&mut self, slot: &mut gcso_ledger_slot_t) -> GcsoResult<()> {
        slot.telemetry_ring_head = 0;
        slot.telemetry_ring_tail = 0;
        slot.cache_hit_count = 0;
        slot.telemetry_mini_ledger = [0u8; LEDGER_HISTORY_LEN];
        Ok(())
    }

    /// Executes fast-path shortcut lookup in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `slot.mode != 0` or no matching non-zero shortcut is registered.
    #[inline(always)]
    fn fast_path_lookup(&self, slot: &gcso_ledger_slot_t, input_key: u64) -> GcsoResult<u64> {
        let (hit, val) = self.fast_path_lookup_branchless(slot, input_key);
        if hit {
            Ok(val)
        } else {
            Err(GCSO_ERROR_INVALID_STATE)
        }
    }

    /// Executes strictly branchless $\mathcal{O}(1)$ fast-path lookup across all 4 shortcut registers.
    #[inline(always)]
    #[must_use]
    fn fast_path_lookup_branchless(&self, slot: &gcso_ledger_slot_t, input_key: u64) -> (bool, u64) {
        let mask_hit = ((slot.fast_path_bypass_mask & input_key) != 0) & (slot.mode == 0);
        let idx = ((input_key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
        let candidate = slot.fast_path_shortcuts[idx & (FAST_PATH_SHORTCUT_SLOTS - 1)];
        let fallback = slot.fast_path_shortcuts[0];
        let resolved = branchless_select_u64(candidate != 0, candidate, fallback);
        let valid_hit = mask_hit & (resolved != 0);
        (valid_hit, branchless_select_u64(valid_hit, resolved, 0))
    }

    /// Branchlessly resolves `input_ptr` against the FastPathLedger fast-path shortcut table in $\mathcal{O}(1)$.
    #[inline(always)]
    fn resolve_or_pass_tagged_ptr(
        &self,
        slot: &mut gcso_ledger_slot_t,
        input_ptr: gcso_tagged_ptr_t,
    ) -> (bool, gcso_tagged_ptr_t) {
        let (hit, shortcut_raw) = self.fast_path_lookup_branchless(slot, input_ptr.raw());
        slot.cache_hit_count = slot.cache_hit_count.wrapping_add(u32::from(hit));
        let eff_raw = branchless_select_u64(hit, shortcut_raw, input_ptr.raw());
        (hit, gcso_tagged_ptr_t(eff_raw))
    }

    /// Registers a 64-bit tagged pointer shortcut into the FastPathLedger fast-path table in $\mathcal{O}(1)$.
    #[inline(always)]
    fn register_fast_path_shortcut(
        &mut self,
        slot: &mut gcso_ledger_slot_t,
        input_key: u64,
        target_shortcut: u64,
    ) {
        let idx = ((input_key.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
        slot.fast_path_shortcuts[idx & (FAST_PATH_SHORTCUT_SLOTS - 1)] = target_shortcut;
        slot.fast_path_bypass_mask |= branchless_select_u64(target_shortcut != 0, input_key, 0);
    }

    /// Pushes a profiling metric byte code (`LEDGER_METRIC_*`) into the FastPathLedger telemetry ring ledger in-place.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_LEDGER_FULL`] only if non-overwriting ring semantics are enforced.
    #[inline(always)]
    fn telemetry_push(&mut self, slot: &mut gcso_ledger_slot_t, metric_code: u8) -> GcsoResult<()> {
        let head = (slot.telemetry_ring_head as usize) & (LEDGER_HISTORY_LEN - 1);
        slot.telemetry_mini_ledger[head] = metric_code;
        slot.telemetry_ring_head = slot.telemetry_ring_head.wrapping_add(1);
        Ok(())
    }

    /// Summarizes the 8-byte `telemetry_mini_ledger` in $\mathcal{O}(1)$ using 8-lane SWAR,
    /// returning `(combined_metric_or_mask, active_entry_count)`.
    #[inline(always)]
    #[must_use]
    fn telemetry_snapshot_metrics(&self, slot: &gcso_ledger_slot_t) -> (u8, u32) {
        let word = u64::from_le_bytes(slot.telemetry_mini_ledger);
        let fold32 = (word as u32) | ((word >> 32) as u32);
        let fold16 = (fold32 & 0xFFFF) | (fold32 >> 16);
        let combined_mask = ((fold16 & 0xFF) | (fold16 >> 8)) as u8;
        (combined_mask, swar_u8_nonzero_count_x8(word))
    }

    /// Sets the operating mode of the ledger slot (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC Buffer).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `mode > 2`.
    #[inline(always)]
    fn set_mode(&mut self, slot: &mut gcso_ledger_slot_t, mode: u32) -> GcsoResult<()> {
        if mode > 2 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        slot.mode = mode;
        Ok(())
    }

    /// Evaluates telemetry ledger to auto-tune sub-head router ratios, RIPA clamps, and EDBC thresholds.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `slot` is in an incompatible mode.
    fn evaluate_auto_tune(
        &self,
        slot: &gcso_ledger_slot_t,
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()>;

    /// Returns the cumulative cache hit count recorded in the slot.
    #[inline(always)]
    fn cache_hit_count(&self, slot: &gcso_ledger_slot_t) -> u32 {
        slot.cache_hit_count
    }
}

