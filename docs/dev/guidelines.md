# Development Guidelines

This document outlines a lightweight set of guidelines for the LiminiKa project. Designed for solo development and open-source collaboration, it aims to serve as a helpful guide rather than a strict set of rules—prioritizing developer comfort, creative freedom, and mutual respect over rigid enforcement.

---

## 1. Core Principles

* **Progress over Perfection**: Don't worry about writing perfectly clean code from day one. Prioritize building working prototypes (PoC) and experimental ideas, then refactor later when needed.
* **Respect Every Contribution**: Whether it's a bug report, documentation fix, question, or experimental PR, all contributions and ideas are warmly welcomed and appreciated.
* **Rely on Automation**: Avoid manual formatting. Standard CI workflows (`.github/workflows/ci.yml`) automatically handle build, linting, and formatting checks.
* **Write for Your Future Self & Collaborators**: Leave brief inline notes when implementing complex logic or mathematical models (especially within GCSO modules).
* **Keep FFI Boundaries Safe**: Since LiminiKa bridges C++ cores and Rust via C-ABI, pay extra attention when modifying files under `include/liminika/` or `src/dsl/src/ffi/`. Always verify ABI compatibility before pushing changes.

---

## 2. Git Workflow

For everyday development, feel free to **commit directly to the `develop` branch**.

> ⚠️ **Note on Direct Commits & CI Health ("Fix-it-first" Guidelines)**: Rapid iteration means things will occasionally break—and that is completely fine! Failing a CI check is never something to worry about or feel bad about. While we encourage testing locally when convenient, these guidelines are just helpful habits, not rigid rules. If `develop` turns red, simply fix it whenever you have time before stacking new features.

### Branch Structure & Protection

* **`develop` (Primary Workspace)**: The main branch for daily development and quick iterations. Direct commits are allowed here. Branch protection is kept minimal to maintain solo development velocity.
* **`main` (Stable Release)**: Reserved for working milestones. Merges into `main` require stable, passing builds. Strict branch protection rules are enforced here to prevent direct accidental pushes.

### Temporary Branches & PRs (Optional)

When attempting larger features, risky refactoring, or submitting a PR, creating temporary branches is recommended:

* **`feat/xxx`** : New features or experiments (e.g., `feat/dpsr-kernel`)
* **`fix/xxx`**  : Bug fixes (e.g., `fix/abi-header`)
* **`docs/xxx`** : Documentation updates (e.g., `docs/update-readme`)

> 💡 **For Collaborators:** If you plan to work on significant architectural changes or new features, opening an Issue or starting a thread in GitHub Discussions beforehand is greatly appreciated to avoid overlapping effort!

### Commit Messages (Recommended)

Using simple prefixes helps keep the history readable, though it is not strictly enforced:

* `feat:` A new feature
* `fix:` A bug fix
* `docs:` Documentation changes
* `refactor:` Code changes that neither fix a bug nor add a feature
* `style:` Formatting adjustments (`cargo fmt`, etc.)
* `chore:` Maintenance tasks (updating configs, dependencies, etc.)

*(Example: `feat: Add basic parser for LiminiKa DSL`)*

---

## 3. Naming Conventions

To maintain readability and cross-platform consistency across different operating systems (Linux, macOS, Windows):

* **Prefer `snake_case` for filenames**: Lowercase `snake_case` is recommended for standard source files, headers, scripts, and documentation (e.g., `c_abi_spec.md`, `gcso_abi.h`).
* **Tooling & Ecosystem Exceptions**: Standard names mandated or universally expected by tools and platforms should follow their respective conventions (e.g., `CMakeLists.txt`, `Cargo.toml`, `LICENSE-MIT`, `README.md`, or canonical paper titles like `GCSO.md`).
* **Guideline over Rule**: This is a flexible recommendation aimed at preventing file-path issues, not a rigid constraint that should hinder rapid experimentation.

---

## 4. Code Style & Formatting

Formatting checks are automatically verified by CI. Manual formatting is not required, but running these tools locally before pushing is recommended.

> 💡 **Editor Integration Note**: A `.clang-format` file is placed in the repository root. Editor extensions (VS Code, CLion, Neovim, etc.) can automatically format C/C++ files on save using this config. Rust formatting relies on standard `cargo fmt` without requiring any custom `rustfmt.toml`.

### Rust (`src/dsl/`, `src/cli/`)

* **Formatter**: Follows standard Rust community conventions (`cargo fmt`). No project-specific `rustfmt.toml` is used to minimize maintenance overhead.
* **Linter**: CI enforces strict warnings as errors for `clippy`. Run the check locally to catch common issues early.

```bash
cargo fmt                                             # Auto-format Rust code
cargo clippy --workspace --all-targets -- -D warnings # Linter (matching CI checks)

```

### C / C++ / GPU Kernels (`src/core/`, `src/kernels/`, `include/`)

Formatting rules are governed by the root `.clang-format` file. A target is integrated into CMake for convenience:

```bash
# Format C/C++/CUDA/Metal files via CMake target
cmake --build build --target format

# (Alternative) Direct script / command execution
find include src/core src/kernels -type f \( -name "*.h" -o -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.metal" \) -exec clang-format -i {} +

```

---

## 5. C-ABI / Memory Layout Safety Checklist

Because LiminiKa bridges C++ core implementations and Rust DSL via C-ABI, modifying shared definitions requires extra caution:

When modifying files under `include/liminika/` or `src/dsl/src/ffi/`:

1. **Check Alignment & Struct Packing**: Verify memory alignment and type definitions in `include/liminika/types.h` (e.g., Q7 fixed-point representation).
2. **Run Dual Test Suites**: Ensure both native C++ tests and Rust FFI bindings pass locally:

```bash
ctest --test-dir build     # Run C++ Core unit tests
cargo test --workspace     # Run Rust DSL & FFI integration tests

```

---

## 6. Directory Layout Guide

When adding new files, place them according to the repository structure:

* **Public C-ABI Headers**: `include/liminika/` (GCSO ABI definitions and types)
* **C/C++ Core Engine (GCSO Modules)**: `src/core/` (`abi/`, `edbc/`, `dpsr/`, `swarm/`, `srl/`, `storage/`, `math/`)
* **GPU/CPU Acceleration Kernels**: `src/kernels/` (`common/`, `cuda/`, `metal/`, `vulkan/`, `cpu/`)
* **Rust DSL Parser & Compiler**: `src/dsl/`
* **Command Line Interface (CLI)**: `src/cli/`
* **Documentation**: `docs/`
* Whitepaper: `docs/paper/` (`ja/`, `en/`)
* Philosophy: `docs/philosophy/` (`ja/`, `en/`)
* Architecture & Specs: `docs/architecture/`
* Developer Guides: `docs/dev/`
* Roadmap & Tasks: `docs/roadmap/`


* **Examples & DSL Scripts**: `examples/` (`c_api/`, `dsl/`)
* **Test Suite**: `tests/` (`core/`, `ffi/`, `fixtures/`)
* **Benchmarks**: `benches/`

---

## 7. Licensing for Contributions

By contributing to LiminiKa, you agree that your contributions will be licensed as follows:

* **Code (`src/`, `include/`, `examples/`, `tests/`, `benches/`)**: Dual-licensed under **MIT** or **Apache-2.0**.
* **Documentation (`docs/`, `*.md`)**: Licensed under **CC BY 4.0**.

*By submitting a pull request, patch, or contribution, you explicitly agree that your work will be covered by the project's respective open-source licenses listed above (Inbound = Outbound Licensing).*

---

## 8. Quick Command Cheatsheet

**Daily Workflow (Working directly on `develop`):**

```bash
git checkout develop
# ... make changes ...

# 1. Auto-format & Lint code
cargo fmt
cargo clippy --workspace --all-targets -- -D warnings
cmake --build build --target format

# 2. Run test suites
cargo test --workspace
ctest --test-dir build

# 3. Commit and push
git add .
git commit -m "feat: Add xxx"

```

**Creating a temporary branch for risky experiments or PRs:**

```bash
git checkout develop
git checkout -b feat/my-experiment
# ... work and complete experiment ...
git checkout develop
git merge feat/my-experiment
git branch -d feat/my-experiment  # Clean up temporary branch

```
