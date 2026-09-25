# Contributing to LiminiKa

Welcome, and thank you for considering contributing to LiminiKa!

LiminiKa is an experimental, low-pressure project. Our goal is to maintain a flexible and comfortable space where anyone can experiment, test ideas, and contribute at their own pace—without formal deadlines, strict expectations, or rigid process requirements.

Whether you are fixing a typo, testing a feature, or optimizing a kernel, all contributions are warmly welcomed.

---

## Flexible & Low-Pressure Workflow

Contributing to LiminiKa is designed to be simple and friction-free. Aside from maintaining **Clean Licensing & IP Safety** (detailed below), feel free to work in whatever way suits you best:

* 🌿 **Low Friction**: PR descriptions can be as simple as a single sentence. Small fixes, single-line cleanups, and documentation tweaks are always welcome.
* 🛠️ **Draft & WIP PRs Encouraged**: Feel free to open a Draft PR early to test ideas, trigger CI builds, or collect early feedback. There is no pressure to finish quickly or complete every draft.
* 🤖 **Automated Checks via CI**: Local formatting and linting perfection are not required. GitHub Actions workflows will handle style checks automatically upon push.
* 💬 **Bilingual Communication**: Feel free to participate in Issues, PR comments, and Discussions in **either English or Japanese**. *(Note: Please write **in-code comments in English** to maintain codebase consistency).*

---

## License Compliance & IP Safety (The Primary Requirement)

To protect the project and its contributors from legal uncertainties, **maintaining clean licensing and intellectual property (IP) integrity is our primary requirement**.

### What to Keep Clean
* **No Copyleft Code**: Please do **NOT** copy, paste, port, or adapt code from repositories or sources bound by Copyleft terms (e.g., **GPL, AGPL, SSPL**).
* **No Proprietary, Restrictive, or Decompiled Code**: Please do **NOT** submit code derived from proprietary software, sources with unknown licensing terms, non-commercial restrictions (e.g., **CC-BY-NC**), or reverse-engineered binaries.
* **Clean & Permissive Provenance**: Ensure all contributed code is either your original creation or strictly derived from permissively licensed sources compatible with **MIT / Apache-2.0**.

### Contribution Licensing & Dependencies
* **Dual-Licensing Agreement**: By submitting a Pull Request, you agree that your code contributions will be dual-licensed under **MIT OR Apache-2.0**, and documentation under **CC BY 4.0**.
* **Dependencies**: If introducing new third-party dependencies (`Cargo.toml` or `CMakeLists.txt`), please verify that their licenses are strictly permissive. You can check Rust dependencies locally via `cargo deny check licenses`.

---

## Development Guidelines & Workflow

For details on project architecture, coding guidelines, and repository design, please see:

👉 **[Development Guidelines](docs/dev/guidelines.md)**

### Target Branch
* Please target your Pull Requests to the **`develop`** branch (our primary integration branch).

---

## Quick Start Commands (Optional)

Running checks locally is completely optional—CI runs them automatically. If you wish to run checks locally for your own convenience:

* **Rust (DSL & CLI)**:

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo test --workspace
cargo deny check licenses

```

* **C/C++ Core & Kernels**:

```bash
cmake -B build -S .
cmake --build build
cmake --build build --target format
ctest --test-dir build

```

---

## Planning Major Changes?

> 💡 If you are considering a major structural refactoring or a large feature, opening an Issue or starting a thread in [GitHub Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions) beforehand is recommended. It helps align overall direction and saves time, though casual PRs are always welcome as well.
