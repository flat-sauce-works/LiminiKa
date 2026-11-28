# Testing & Verification

This document describes the complete local and CI verification matrix for LiminiKa `v0.4.0`. On Windows, pass `-j 1` to Cargo commands to keep build parallelism deterministic.

---

## 1. Formatting, Dependency & License Hygiene

```bash
# Rust Edition 2024 formatting across root workspace, candle_eval, and fuzz targets
cargo fmt --all -- --check
cargo fmt --manifest-path examples/candle_eval/Cargo.toml -- --check
rustfmt --edition 2024 --check fuzz/fuzz_targets/*.rs

# Python benchmark scripts syntax verification
python -m py_compile scripts/run_benchmarks.py scripts/generate_bench_report.py

# Lockfile integrity and strict permissive license audit
cargo metadata --format-version 1 --locked > /dev/null
cargo metadata --manifest-path examples/candle_eval/Cargo.toml --format-version 1 --locked > /dev/null
cargo deny --offline check licenses bans sources
```

---

## 2. Static Architecture Gate (`cargo xtask`)

Zero-dependency Rust `xtask` (`xtask/src/`) verifies structural invariants directly against `liminika_core::CORE_TOPOLOGY`:

```bash
cargo xtask --check --audit --hotspots --mermaid --mermaid-workspace --liminal-map
```

`--check` enforces hard errors for:

- 5-layer unidirectional DAG edges (`liminal` || `foundation` -> `stigmergy` -> `drivers` -> `runtime`).
- 3-driver 6-pair mutual isolation (`drivers/{spine,ego,envelope}`).
- `liminal/axiom/cut.rs` vs `liminal/axiom/lacuna.rs` orthogonality and `foundation/` zero-`Scoped` rule.
- App-agnostic `src/adapter/src/host/` LLM-vocabulary and reverse-import isolation.
- Rust-to-C-header 27/27 1:1 C-ABI export parity (`include/liminika/lmnk_abi.h`) and `discarded-return == 0`.
- File hygiene: `<= 2,000` prod LOC/file, `<= 120` columns, pure LF, and zero raw CJK characters in source files.

`--audit` and `--hotspots` emit non-blocking diagnostics (exit 0).

---

## 3. Core Targets, Lints & Documentation

```bash
# Strict #![no_std] host build
cargo rustc --locked -p liminika-core --no-default-features --lib --crate-type rlib -- -D warnings

# WebAssembly portability build (0 warnings, RUSTFLAGS unset)
cargo rustc --locked -p liminika-core --no-default-features --target wasm32-unknown-unknown --lib --crate-type rlib -- -D warnings

# Clippy across all configurations (-D warnings)
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo clippy --locked --manifest-path examples/candle_eval/Cargo.toml --all-targets -- -D warnings
cargo clippy --locked --manifest-path examples/candle_eval/Cargo.toml --all-targets --features cuda -- -D warnings

# Rustdoc with warnings denied
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --workspace --all-features
```

---

## 4. Unit, Integration & SIMD Test Suites

```bash
# Workspace unit, integration, and doc tests + candle_eval tests
cargo test --locked --workspace --all-targets
cargo test --locked --workspace --doc
cargo test --locked --manifest-path examples/candle_eval/Cargo.toml
```

Key integration suites under `tests/`:

- `tests/core/test_zero_alloc.rs`: Verifies $C_{\mathrm{alloc}} = 0$ on hot-path `liminika-core` routines.
- `tests/core/test_topology_parity.rs`: Verifies compile-time `CORE_TOPOLOGY` (`64 B`, CRC32) against filesystem layers.
- `tests/core/test_security_adversarial.rs`: Container tampering, ABI version/size/alignment rejections, pin-floor preservation, and teardown scrubbing.
- `tests/ffi/test_abi_layout.rs`: 1:1 layout, alignment, and field offset invariants for all C-ABI types.
- `tests/integration/host_envelope_coordination.rs`: Non-LLM host integration testing `EnvelopeState` + `BoundarySignal`.
- `tests/integration/fuzz_regression.rs`: Deterministic regression corpus (3,508 CRC32-patched container/package/DSL mutations).

### AVX2 + FMA Verification (x86_64)

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo clippy --locked -p liminika-core --all-targets --all-features -- -D warnings
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --locked -p liminika-core --lib --all-features
```

Unset `RUSTFLAGS` immediately afterward so subsequent builds (such as `wasm32-unknown-unknown` or Miri) do not inherit x86_64 target features.

---

## 5. Miri Strict Provenance (UB = 0)

Pinned nightly Miri validates strict pointer provenance and symbolic alignment across all 6 CI targets:

```bash
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" \
  cargo +nightly-2026-11-20 miri test --locked -p liminika-core --lib
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" \
  cargo +nightly-2026-11-20 miri test --locked -p liminika-core --test test_abi_layout
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" \
  cargo +nightly-2026-11-20 miri test --locked -p liminika-core --test test_zero_alloc
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" \
  cargo +nightly-2026-11-20 miri test --locked -p liminika-core --test test_security_adversarial
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" \
  cargo +nightly-2026-11-20 miri test --locked -p liminika-dsl --lib
MIRIFLAGS="-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check" \
  cargo +nightly-2026-11-20 miri test --locked -p liminika-adapter --lib -- host::path_guard
```

---

## 6. Native C11 / C++20 CTest & Header Self-Containment

### CMake / CTest (`examples/c_api/` & `tests/ffi/test_c_abi_smoke.cpp`)

```bash
cargo build --locked --release -p liminika-core
cmake -B build-check -S .
cmake --build build-check --config Release
ctest --test-dir build-check -C Release --output-on-failure --no-tests=error
```

On Linux, CI additionally runs CTest with `-fsanitize=address,undefined -fno-omit-frame-pointer`.

### Standalone Header Self-Containment (3 Headers × C11 & C++20)

Each header under `include/liminika/` (`lmnk_config.h`, `lmnk_types.h`, `lmnk_abi.h`) must compile cleanly on its own in both C11 and C++20 with zero warnings:

- **MSVC (`vcvars64.bat`)**:
  ```cmd
  for %H in (lmnk_config.h lmnk_types.h lmnk_abi.h) do (
    cl.exe /nologo /Zs /std:c11 /TC /W4 /WX /wd4324 /utf-8 /Zc:preprocessor /Iinclude include\liminika\%H
    cl.exe /nologo /Zs /std:c++20 /TP /W4 /WX /wd4324 /permissive- /utf-8 /Zc:preprocessor /Iinclude include\liminika\%H
  )
  ```
- **Clang / GCC**:
  ```bash
  for h in include/liminika/lmnk_config.h include/liminika/lmnk_types.h include/liminika/lmnk_abi.h; do
    cc -std=c11 -Wall -Wextra -Wpedantic -Werror -Iinclude -fsyntax-only -x c "$h"
    c++ -std=c++20 -Wall -Wextra -Wpedantic -Werror -Iinclude -fsyntax-only -x c++ "$h"
  done
  ```

---

## 7. Benchmarks & Fuzz Targets

### Criterion Microbenchmarks

```bash
# Compile all benchmark harnesses without executing
cargo bench --locked --workspace --no-run

# Run via the unified runner (core affinity pinning, EcoQoS suppression, Markdown/JSON/HTML report)
python scripts/run_benchmarks.py --profile quick
python scripts/run_benchmarks.py --profile standard
```

### Continuous Fuzzing (`fuzz/`)

`fuzz/` is an isolated Cargo workspace defining four `libfuzzer-sys` targets under `fuzz/fuzz_targets/`:

- `lmnkc_container`: Untrusted `.lmnkc` binary container parsing (`LmnkContainerView::parse`).
- `lmnkp_package`: Untrusted `.lmnkp` Store ZIP archive parsing (`LmnkPackageView::parse`).
- `lmnk_dsl`: `.lmnk` declarative DSL lexer, parser, and semantic validator.
- `lmnk_timeline`: Dual-track `.lmnkc` timeline decoding and GOP seeking.

Run on Linux/macOS with nightly `cargo-fuzz`:

```bash
cd fuzz && cargo +nightly fuzz run lmnkc_container
```
