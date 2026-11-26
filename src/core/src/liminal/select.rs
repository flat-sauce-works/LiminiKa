// name: src/core/src/liminal/select.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constant-time branchless primitive selectors shared from Layer 0 upward.

#![forbid(unsafe_code)]

use crate::liminal::constants::PhaseQ7;

/// Branchless bitwise selection for 8-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
#[must_use]
pub const fn branchless_select_u8(cond: bool, if_true: u8, if_false: u8) -> u8 {
    let m = (cond as u8).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 8-bit signed integers (`cond ? if_true : if_false`).
#[inline(always)]
#[must_use]
pub const fn branchless_select_i8(cond: bool, if_true: i8, if_false: i8) -> i8 {
    let m = (cond as i8).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 16-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
#[must_use]
pub const fn branchless_select_u16(cond: bool, if_true: u16, if_false: u16) -> u16 {
    let m = (cond as u16).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 32-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
#[must_use]
pub const fn branchless_select_u32(cond: bool, if_true: u32, if_false: u32) -> u32 {
    let m = (cond as u32).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 64-bit unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
#[must_use]
pub const fn branchless_select_u64(cond: bool, if_true: u64, if_false: u64) -> u64 {
    let m = (cond as u64).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for pointer-sized unsigned integers (`cond ? if_true : if_false`).
#[inline(always)]
#[must_use]
pub const fn branchless_select_usize(cond: bool, if_true: usize, if_false: usize) -> usize {
    let m = (cond as usize).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 32-bit signed integers (`cond ? if_true : if_false`).
#[inline(always)]
#[must_use]
pub const fn branchless_select_i32(cond: bool, if_true: i32, if_false: i32) -> i32 {
    let m = (cond as i32).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for 64-bit signed integers (`cond ? if_true : if_false`).
#[inline(always)]
#[must_use]
pub const fn branchless_select_i64(cond: bool, if_true: i64, if_false: i64) -> i64 {
    let m = (cond as i64).wrapping_neg();
    (if_true & m) | (if_false & !m)
}

/// Branchless bitwise selection for Q7 fixed-point integers (`cond ? if_true : if_false`).
#[inline(always)]
#[must_use]
pub const fn branchless_select_q7(cond: bool, if_true: PhaseQ7, if_false: PhaseQ7) -> PhaseQ7 {
    branchless_select_i8(cond, if_true, if_false)
}

/// Branchless selection for IEEE-754 single-precision floats (`cond ? if_true : if_false`),
/// remaining in the floating-point XMM register file (`vminss` / `vmaxss` / `vblendvps`)
/// without XMM-to-GPR (`vmovd`) cross-domain bypass stalls.
#[inline(always)]
#[must_use]
pub const fn branchless_select_f32(cond: bool, if_true: f32, if_false: f32) -> f32 {
    if cond { if_true } else { if_false }
}
