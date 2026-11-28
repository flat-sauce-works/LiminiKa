# Getting Started

This guide walks through building LiminiKa locally, exercising `liminika-cli` and the native C-ABI checks, and optionally running local GPU evaluation. Command examples use POSIX shell syntax; on Windows PowerShell, pass `-j 1` to Cargo commands and run CMake/CTest from a Visual Studio x64 Developer Command Prompt (`vcvars64.bat`).

## 1. Prerequisites

- **Rust toolchain**: Edition **2024**, MSRV **`1.100.0`** (`resolver = "3"`), with `rustfmt` and `clippy`.
- **Native C-ABI builds (optional)**: CMake >= 3.18 and a C11 / C++20 compiler:
  - Windows: Visual Studio / Build Tools with `vcvars64.bat` (MSVC `cl.exe`).
  - Linux / macOS: Clang or GCC (plus Ninja recommended).
- **WebAssembly check (optional)**: `rustup target add wasm32-unknown-unknown`.
- **Local GPU evaluation (optional)**: NVIDIA CUDA toolkit and local model weights (not bundled) for `examples/candle_eval`.

---

## 2. Build & Verify the Rust Workspace

```bash
# Run all workspace unit, integration, and doc tests
cargo test --locked --workspace --all-targets
cargo test --locked --workspace --doc

# Verify strict #![no_std] build of liminika-core
cargo rustc --locked -p liminika-core --no-default-features --lib --crate-type rlib -- -D warnings

# Verify 5-layer DAG, 3-driver isolation, C-ABI 27/27 parity, and file hygiene
cargo xtask --check
```

On Windows, passing `-j 1` to Cargo commands avoids transient file-locking overhead when running multiple build gates sequentially.

---

## 3. Using `liminika-cli`

The `liminika` binary (`src/cli/`) provides DSL validation, container baking/rebaking, inspection, and cryptographic signing/verification:

```bash
# Default C-ABI round-trip smoke check (no subcommand)
cargo run --locked -p liminika-cli

# Validate a .lmnk declarative DSL file
cargo run --locked -p liminika-cli -- check examples/dsl/limina_poc.lmnk

# Parse and dry-run an evaluation fixture
cargo run --locked -p liminika-cli -- eval-check
cargo run --locked -p liminika-cli -- eval-dry-run

# Bake a .lmnkc container or .lmnkp package and inspect it
cargo run --locked -p liminika-cli -- bake tests/fixtures/gcso_eval_samples.txt containers/demo.lmnkp phi4-mini --package
cargo run --locked -p liminika-cli -- inspect containers/demo.lmnkp

# Generate an Ed25519 keypair, sign, and verify an artifact
cargo run --locked -p liminika-cli -- keygen containers/dev.key containers/dev.pub
cargo run --locked -p liminika-cli -- sign containers/demo.lmnkp containers/demo_signed.lmnkp --key containers/dev.key
cargo run --locked -p liminika-cli -- verify containers/demo_signed.lmnkp --pubkey containers/dev.pub
```

Run `cargo run --locked -p liminika-cli -- --help` for the full subcommand list (`check`, `eval-check`, `eval-dry-run`, `list`, `inspect`, `verify`, `sign`, `keygen`, `bake`, `rebake`).

---

## 4. Native C11 / C++20 C-ABI Build & CTest

Build the Rust `liminika-core` static library first so CMake can locate and link it:

```bash
cargo build --locked --release -p liminika-core
cmake -B build -S .
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure --no-tests=error
```

On Windows, run the CMake and CTest commands inside a Visual Studio x64 Developer Command Prompt (`vcvars64.bat`). The C-ABI headers live under [`include/liminika/`](../../include/liminika/) (`lmnk_config.h`, `lmnk_types.h`, `lmnk_abi.h`). See the [C-ABI Specification](../architecture/c_abi_spec.md) for API usage.

---

## 5. Local LLM Evaluation (`examples/candle_eval`)

`examples/candle_eval/` is an isolated Cargo workspace for evaluating RoPE phase steering, Rank-1 residual injection, entropy control, and boundary classification against local models (`Phi-4-mini-instruct`, `Ministral-3-3B-Instruct-2512`).

```bash
# Compile and run CPU unit tests (no model weights required)
cargo test --locked --manifest-path examples/candle_eval/Cargo.toml

# Build with CUDA acceleration
cargo build --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda
```

Example live evaluation invocations (supply paths to your local weights):

```bash
# 4-axis steering physics evaluation (automatic layer selection default)
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PATH_TO_MODEL> --quant=q4k --layer-mask=auto --physics-eval

# Step 2 + 8-turn multi-turn benchmark (3 median trials)
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PATH_TO_MODEL> --quant=q4k --layer-mask=auto --fair-bench --bench-trials=3 --multi-turn

# Extended English, Japanese multilingual, and VLM evaluations
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PATH_TO_MINISTRAL3> --quant=q4k --layer-mask=auto --extended-eval --ja-eval
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PATH_TO_MINISTRAL3> --quant=q4k --layer-mask=auto --vlm-eval

# Long-document streaming distillation and cross-restart equilibrium benchmarks
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PATH_TO_MODEL> --quant=q4k --long-doc-bench=docs/paper/en/GCSO.md
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PATH_TO_MODEL> --quant=q4k --layer-mask=auto --equilibrium-bench
```

See [Benchmarks & Evaluation Results](../benchmark/benchmarks.md) for recorded reference metrics and [Testing & Verification](./testing.md) for the full quality gate suite.
