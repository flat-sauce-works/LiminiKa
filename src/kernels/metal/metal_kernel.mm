// src/kernels/metal/metal_kernel.mm
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "liminika/gcso_abi.h"

// Metal compute backend stub for GCSO runtime on macOS platforms.

GCSO_EXTERN_C_BEGIN

GCSO_API gcso_status_t GCSO_CALL gcso_metal_kernel_ping(void) GCSO_NOEXCEPT {
    return GCSO_SUCCESS;
}

GCSO_EXTERN_C_END