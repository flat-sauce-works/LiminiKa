# Development Guidelines

Welcome to the LiminiKa development guidelines! This document provides a light, welcoming, and low-pressure framework for working on LiminiKa.

LiminiKa was born as a casual, experimental project. Our primary goal is to maintain a relaxed, comfortable, and pressure-free space where anyone can experiment, test ideas, and contribute at their own pace. The **only essential, non-negotiable requirement** is maintaining clean licensing and intellectual property safety.

---

## 1. Core Principles

* 🛡️ **Clean Licensing & IP Safety (The Primary Requirement)**: Keeping the codebase strictly free of third-party copyright issues and incompatible licenses is our only strictly enforced rule.
* ☕ **No Pressure & Relaxed Workspace**: We intentionally avoid rigid processes, deadlines, or formal expectations. Feel free to work whenever and however you like.
* 🛠️ **Freedom of Tooling & Workflow**: Use whatever editors, environments, or tools you feel most comfortable with. There are no restrictions on your personal workflow.
* 🤝 **Welcoming Communication**: Questions, ideas, documentation tweaks, and PRs are warmly welcomed in **either English or Japanese**.
* 🌐 **English Code Comments**: Please write in-code comments in English to keep the codebase consistent and globally accessible.
* ⚙️ **Automated Checks via CI**: Formatting and basic checks are automated via GitHub Actions (`.github/workflows/ci.yml`). Don't worry about local lint perfection—just push your work and let CI handle it.

---

## 2. License Compliance & Clean Code (The Only Strict Rule)

To protect LiminiKa and its users, all committed code must be completely free of incompatible licenses or third-party copyrighted code.

### What is NOT Allowed

* **No Copyleft Code**: Do **NOT** copy, paste, or directly port code snippets from repositories or sources licensed under Copyleft terms (e.g., **GPL, AGPL, SSPL**).
* **No Proprietary or Restrictive Code**: Do **NOT** include code under commercial/proprietary licenses, Non-Commercial restrictions (e.g., **CC-BY-NC**), or unknown terms.

### Permissive Code & Workflow Freedom

* All third-party dependencies must strictly use permissive licenses compatible with **MIT / Apache-2.0** (e.g., MIT, Apache-2.0, BSD, zlib).
* Ensure that all committed code represents original or permissively licensed work.

```bash
# Automated Rust dependency license check
cargo deny check licenses

```

---

## 3. Git & Branching Workflow

To keep development lightweight, allow maintainers to iterate quickly, and prevent frustrating merge conflicts for everyone, we use a simple branching structure.

### Branch Roles

* **`main` (Stable Releases)**: Holds stable, tagged release states. Direct pushes are restricted (Pull Requests required).
* **`develop` (Integration Branch)**: The primary integration branch for ongoing work. Direct pushes to `develop` are allowed for maintainers to support fast personal iteration.

### How to Contribute (Preventing Merge Conflicts)

To make sure you can work comfortably on your own ideas without worrying about merge conflicts on `develop`:

1. **Use a Personal Branch or Fork**: Non-maintainer contributions should always be developed on a personal feature branch (e.g., `feature/my-idea`) or from a personal fork.
2. **Experiment Freely**: Feel free to commit and push WIP (Work In Progress) code on your branch. Broken builds during active experimentation are completely normal!
3. **Open a PR to `develop**`: When you're ready to share your work, open a Pull Request targeting the `develop` branch.

### Low-Friction PR Expectations

* **No Lengthy PR Descriptions Required**: A single sentence explaining what was changed is more than enough.
* **Draft / WIP PRs Welcome**: Feel free to open a Draft PR anytime to share progress or get early feedback, without any pressure to finish immediately.

---

## 4. Local Quick-Start Cheatsheet (Optional)

CI automatically handles formatting and checks upon push. If you want to run quick local checks for your own convenience, here is a reference:

### Rust Crates (`src/dsl/`, `src/cli/`)

```bash
cargo check --workspace                               # Fast syntax & type check
cargo fmt --all                                       # Auto-format all Rust code
cargo clippy --workspace --all-targets -- -D warnings # Run linter
cargo test --workspace                                # Run test suite

```

### C/C++ Core & Kernels (`src/core/`, `src/kernels/`)

```bash
cmake -B build -S .                                   # Configure CMake build directory
cmake --build build                                   # Build C/C++ targets
ctest --test-dir build                                # Run C/C++ unit tests
clang-format -i include/**/*.h src/core/**/*.cpp      # Auto-format C/C++ code

```
