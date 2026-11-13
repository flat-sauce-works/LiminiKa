// name: src/dsl/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Declarative DSL frontend crate for LiminiKa (staged for post-core integration).

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
