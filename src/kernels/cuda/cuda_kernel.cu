// src/kernels/cuda/cuda_kernel.cu
// SPDX-License-Identifier: MIT OR Apache-2.0

#include "liminika/gcso_abi.h"

// CUDA compute kernel stub for GCSO runtime acceleration.

GCSO_EXTERN_C_BEGIN

GCSO_API gcso_status_t GCSO_CALL gcso_cuda_kernel_ping(void) GCSO_NOEXCEPT {
    return GCSO_SUCCESS;
}

GCSO_EXTERN_C_END