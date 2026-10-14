# name: tests/fixtures/generate_dummy_model.py
# SPDX-License-Identifier: MIT OR Apache-2.0

import os
from pathlib import Path

import numpy as np

# Dummy tensor array configuration matching C-ABI memory alignment
HEAD_DIM = 128
NUM_HEADS = 32
TENSOR_ELEMS = HEAD_DIM * NUM_HEADS  # 4096 float32 elements (16,384 bytes)

# Script directory for relative path resolution
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "dummy_weights.bin"


def generate_dummy_raw_tensors(output_path: Path = DEFAULT_OUTPUT_PATH):
    """Generates flat raw binary tensor data for kernel benchmarks and SIMD alignment testing.

    Guarantees strict 32-byte SIMD cacheline alignment for zero-copy ZIMMS mmap testing.
    """
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)

    np.random.seed(42)

    # 1. Generate float32 tensor buffers (Query/Key) - 16,384 bytes each (32-byte aligned)
    query_data = np.random.randn(TENSOR_ELEMS).astype(np.float32).tobytes()
    key_data = np.random.randn(TENSOR_ELEMS).astype(np.float32).tobytes()

    # 2. Generate int8 Q7 phase delta buffer (64 heads) - 64 bytes
    q7_phase_data = np.random.randint(-128, 127, size=64, dtype=np.int8).tobytes()

    # 3. Concatenate into a contiguous flat binary stream
    flat_payload = query_data + key_data + q7_phase_data

    # Pad payload to guarantee overall file size is a multiple of 32 bytes
    alignment_remainder = len(flat_payload) % 32
    if alignment_remainder != 0:
        padding_bytes = 32 - alignment_remainder
        flat_payload += b"\x00" * padding_bytes

    with open(output_path, "wb") as f:
        f.write(flat_payload)

    print(f"[+] Dummy raw weight binary successfully created: {output_path}")
    print(f"    Total Payload Size: {len(flat_payload)} bytes (32-byte aligned)")
    print(f"    - Query Tensor: {len(query_data)} bytes (f32)")
    print(f"    - Key Tensor:   {len(key_data)} bytes (f32)")
    print(f"    - Phase Delta:  {len(q7_phase_data)} bytes (i8)")


if __name__ == "__main__":
    generate_dummy_raw_tensors()
