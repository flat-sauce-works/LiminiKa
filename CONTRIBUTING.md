# Contributing to LiminiKa

Thank you for your interest in contributing to LiminiKa! 
We warmly welcome all contributions, including bug reports, documentation updates, feature proposals, and experimental PRs.

## Development Guidelines

For detailed guidelines on code style, Git workflow, and architecture boundaries, please refer to:

👉 **[Development Guidelines (docs/dev/guidelines.md)](docs/dev/guidelines.md)**

## Quick Tips for Submitting PRs

Don't worry about breaking things or making mistakes—our automated CI workflows will run tests and check formatting for you! 

If you'd like to check things locally when convenient, here are the commands (all optional):

* **Formatting**: `cargo fmt` and `cmake --build build --target format`
* **Tests**: `cargo test --workspace` and `ctest --test-dir build`

> 💡 **Planning a major change?** If you plan to introduce significant features or architectural changes, opening an Issue or starting a thread in [GitHub Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions) beforehand is greatly appreciated so we can align on direction!