# Contributing to LiminiKa

LiminiKa is an experimental, asynchronous research project exploring local LLM steering and runtime architectures. Contributions are structured around a low-pressure, no-obligation workflow: you can share work in any state, leave it unfinished, or pick up someone else's partial idea at any time.

Aside from keeping licensing and intellectual property clean, there are no rigid rules, deadlines, or maintenance obligations.

---

## Intellectual Property & License Boundaries (Mandatory)

Keeping the repository permanently open and legally clear is the one non-negotiable requirement.

- **Submission agreement**: By opening a Pull Request, you agree that code contributions are dual-licensed under **[MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE)**, and documentation contributions under **[CC BY 4.0](LICENSE-CC-BY-4.0)**.
- **Prohibited sources**: Do not submit, adapt, or derive code from:
  - **Copyleft licenses** (GPL, LGPL, AGPL, SSPL, or similar reciprocal licenses)
  - **Non-commercial or restrictive terms** (e.g., CC-BY-NC, proprietary code, reverse-engineered binaries)
  - **Ambiguous origins** (code with unclear or missing license terms)
- **Original or permissive only**: All submitted code must be original or sourced strictly from permissively licensed projects (MIT, Apache-2.0, BSD, ISC, Zlib).
- **Dependencies**: Any new dependencies in `Cargo.toml` or `CMakeLists.txt` must use permissive licenses. Rust dependencies are checked automatically by `cargo-deny` in CI.

---

## Asynchronous Workflow

- **Target branch**: Open Pull Requests against the **`develop`** branch (`main` holds release-ready snapshots).
- **Minimal PR descriptions**: A single sentence explaining what you changed or tried is plenty.
- **Drafts & partial work**: Draft PRs and unfinished experiments are welcome anytime, with zero obligation to polish or maintain them.
- **Red CI is fine**: Failing checks are not treated as rejections—leave them as-is if you do not want to chase them; someone else can pick them up later.
- **Automated formatting**: CI (`auto-format`) may push formatting commits for LF line endings, Rust, C/C++, and Python. Run `git pull` if you continue working on that branch locally.
- **Language**: Issues, PRs, and [Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions) are welcome in **English or Japanese**. Please keep comments and identifiers inside source code in English.

---

## Optional Local Checks

Running checks locally before opening a PR is completely optional—CI handles validation automatically. If you prefer to test locally:

### Rust Workspace

```bash
cargo test --locked --workspace
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo deny check
python scripts/verify_architecture.py --check
```

### Native C11 / C++20 C-ABI Checks

Build `liminika-core` in release mode first so CMake can link the static library (on Windows, run from a Developer Command Prompt / `vcvars64.bat`):

```bash
cargo build --locked --release -p liminika-core
cmake -B build -S .
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

---

## Larger Proposals

For major architectural changes or redesigns, opening an [Issue](https://github.com/flat-sauce-works/LiminiKa/issues) or a thread in [GitHub Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions) first is helpful to avoid duplicated effort, though exploratory PRs are always welcome.
