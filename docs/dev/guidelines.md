# Development Guidelines

Welcome to the LiminiKa development guidelines. This document serves as an open, low-pressure orientation for exploring, experimenting with, and building upon LiminiKa.

LiminiKa is an open-source research project exploring local LLM geometries, runtime dynamics, and declarative dialogue systems. Our objective is to cultivate an unhurried, friction-free environment where technical ideas and dynamic prototypes can be shared naturally without procedural overhead.

Aside from keeping our licensing and intellectual property clean—which is handled automatically on the repository side—there are virtually no rigid rules here. Feel free to participate at whatever pace, depth, or state fits your current momentum.

---

## 1. Core Operating Principles

* 🛡️ **Automated IP Safety (The Primary Boundary)**
  Our primary structural boundary is maintaining standard permissive licensing (**MIT / Apache-2.0** for code, **CC BY 4.0** for documentation). Automated checks (`deny.toml`) run silently in the background to catch incompatible dependencies automatically. You do not need to perform manual license audits.

* 🛠️ **Tooling as a Convenience, Not a Gatekeeper**
  CI pipelines and automated utilities in this repository exist solely to keep your local environment lightweight and save your machine's compute resources. They are not pass/fail gatekeepers. If a CI run turns red, **simply leave it as-is**. Red checks are never treated as errors, failures, or rejections—they sit quietly as unfinished puzzle pieces.

* ☕ **As-Is & Asynchronous Dynamics**
  Ideas and code are welcome in any state—whether as rough notes, experimental sketches, or partial implementations. There is zero obligation to polish, maintain, or follow up on anything you submit. You are always free to push a draft branch or prototype and walk away whenever you wish.

* 🌐 **Complete Workflow & Tooling Autonomy**
  You are entirely free to write, test, or assemble code using whatever system, editor, environment, or personal setup fits you best. No standard runtime, specific platform, or prescribed toolchain is assumed or required.

* 📝 **Casual Communication & Conventions**
  Do not worry about precise commit messages, strict comment formatting, or exact code layouts. As long as the general intent is roughly discernible, adjustments can easily be made down the road whenever someone feels inclined. Japanese and English are both warmly welcomed for notes, issues, and discussions.

---

## 2. Licensing & Intellectual Property Safety

To ensure LiminiKa remains permanently open and legally straightforward, all contributions must align with our permissive open-source foundation.

### Restricted Code Sources
* **No Copyleft Code**: Do not copy, adapt, or port code subject to Copyleft terms (e.g., **GPL, LGPL, AGPL, SSPL**).
* **No Restrictive Clauses**: Avoid code bound by non-commercial or proprietary restrictions (e.g., **CC-BY-NC**).
* **Permissive or Original Works Only**: Ensure contributions are your own original work or derived from permissively licensed projects.

### Automated Checks
Dependencies are automatically validated via `cargo-deny` in CI to ensure compliance with permissive licenses.

If you ever wish to run a quick local check for your own reference:

```bash
# Optional local license check
cargo deny check licenses

```

---

## 3. Casual Git & PR Flow

Our workflow is designed to be completely low-friction, allowing concepts to be shared without worrying about maintaining build statuses or repository state.

### Branch Structure

* **`main`**: Stable, release-ready state.
* **`develop`**: Active integration and exploration branch.

### Pull Request Flow

1. **Work Anywhere**: Branch off from `develop` or work directly from your own fork.
2. **Push Early & Leave It Be**: Push incomplete, draft, or experimental branches at any point.
3. **Open a Draft PR**: A single sentence or brief note describing what you were experimenting with is more than enough.
4. **Red CI Status is Fully Expected & Fine**:
* If builds, linters, or test checks fail, **simply leave the PR open as-is**.
* There is no expectation to investigate, fix, or respond to failing status indicators.
* Unfinished PRs and failing checks can sit indefinitely without issue. Anyone with spare time or curiosity can pop in, format code, resolve build errors, or pick up parts of an idea whenever convenient.

---

## 4. Optional Local Commands

Executing validation commands locally is **entirely optional**. You do not need to run these before opening a Pull Request. They are listed below strictly as a convenience for local testing if you prefer:

### Rust Workspace (`src/core/`, `src/dsl/`, `src/cli/`)

```bash
cargo check --workspace                               # Quick syntax and type check
cargo fmt --all                                       # Auto-format Rust code
cargo clippy --workspace --all-targets -- -D warnings # Run linter
cargo test --workspace                                # Run test suite

```

### C/C++ Core & Kernels (`include/`, `src/kernels/`)

```bash
cmake -B build -S . -GNinja                            # Configure build
cmake --build build                                    # Build C/C++ targets
ctest --test-dir build                                 # Run C/C++ tests
cmake --build build --target format                    # Apply C/C++ formatting locally

```
