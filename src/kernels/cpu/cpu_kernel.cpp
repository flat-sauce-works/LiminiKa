// src/kernels/cpu/cpu_kernel.cpp
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "liminika/gcso_abi.h"

// CPU kernel fallback implementation for GCSO phase steering and SIMD operations.

GCSO_EXTERN_C_BEGIN

GCSO_API gcso_status_t GCSO_CALL gcso_cpu_kernel_ping(void) GCSO_NOEXCEPT {
    return GCSO_SUCCESS;
}

GCSO_EXTERN_C_END