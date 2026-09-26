# Contributing to LiminiKa

LiminiKa is an asynchronous research and development project focused on local LLM geometries and runtime architectures. Development is structured around a low-pressure, friction-free model. 

Work can be submitted in any state, left incomplete, or picked up by others at any time. There are no deadlines, maintenance obligations, or code-review formalities.

---

## Intellectual Property & License Boundaries (Mandatory)

To keep the repository permanently open, legally unencumbered, and safe for downstream use, maintaining strict IP cleanliness is our only non-negotiable requirement.

### Submission Agreement
By submitting a Pull Request to this repository, you agree that your contributions will be licensed as follows:
* **Code contributions**: Dual-licensed under **MIT OR Apache-2.0**.
* **Documentation contributions**: Licensed under **CC BY 4.0**.

### Prohibited Sources
Do not submit, adapt, or derive code from:
* **Copyleft Licenses**: GPL, LGPL, AGPL, SSPL, or any licenses requiring reciprocal derivative terms.
* **Non-Commercial / Restrictive Terms**: Any code containing non-commercial clauses (e.g., CC-BY-NC), proprietary source code, or reverse-engineered binaries.
* **Ambiguous Origins**: Code with unclear or missing licensing information.

All code must be your original work or sourced strictly from permissively licensed projects (MIT, Apache-2.0, BSD, ISC, Zlib).

### Dependencies
Any new dependencies added to `Cargo.toml` or `CMakeLists.txt` must strictly adhere to permissive licenses. Rust dependencies are automatically verified via `cargo-deny` in CI.

---

## Asynchronous Workflow Guidelines

* **Target Branch**: All Pull Requests should target the **`develop`** branch.
* **Minimal PR Descriptions**: A single sentence explaining what was changed or attempted is sufficient.
* **Draft & Experimental Work**: Feel free to open Draft PRs at any time to run CI tests or save partial progress. You are under no obligation to finish or maintain submitted work.
* **Failing CI Checks (Red Status)**: A failing CI build is not treated as an error, rejection, or failure. It simply indicates an unfinished state. PRs with failing checks may sit indefinitely until someone else chooses to address them.
* **Automated Formatting**: CI automatically formats C/C++ source files. If a formatting commit is added by CI to your PR branch, simply run `git pull` if you intend to continue working locally.
* **Bilingual Discussions**: Issues, PR comments, and Discussions are welcome in either **English or Japanese**. *(Technical comments inside source code should remain in English for codebase uniformity).*

---

## Reference Commands (Optional)

Running tests or linters locally before submitting a PR is entirely optional. Validation is handled automatically by CI. The commands below are provided solely for local reference:

### Rust Workspace

```bash
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all
cargo test --workspace
cargo deny check licenses

```

### C/C++ Core & Kernels

```bash
cmake -B build -S .
cmake --build build
cmake --build build --target format
ctest --test-dir build

```

---

## Technical & Architectural Proposals

For major architectural adjustments or structural redesigns, opening an Issue or a thread in [GitHub Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions) is recommended to align technical directions, though exploratory PRs are always welcome.
