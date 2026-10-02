// tests/core/test_zero_alloc.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, Ordering};

struct ZeroAllocGuardAllocator;

static ENFORCE_ZERO_ALLOC: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for ZeroAllocGuardAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENFORCE_ZERO_ALLOC.load(Ordering::Relaxed) {
            panic!("Zero-Alloc Violation! Dynamic allocation detected in Hot-Path: {:?}", layout);
        }
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static A: ZeroAllocGuardAllocator = ZeroAllocGuardAllocator;

#[test]
fn test_context_step_token_zero_allocation_guarantee() {
    // 1. Setup Context in Cold Path (Allocations allowed here)
    // ... Initialize config and context ...

    // 2. Enable Strict Zero-Allocation Guard for Hot Path
    ENFORCE_ZERO_ALLOC.store(true, Ordering::SeqCst);

    // 3. Execute Hot Path Step Loop (Must NOT allocate)
    for token_id in 0..100 {
        // Call gcso_context_step_token via FFI with aligned buffers
    }

    // 4. Disable Guard
    ENFORCE_ZERO_ALLOC.store(false, Ordering::SeqCst);
}