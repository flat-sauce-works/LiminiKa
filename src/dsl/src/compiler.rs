// src/dsl/src/compiler.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use liminika_core::abi::gcso_config_t;

pub fn compile_dsl_to_config(persona: &crate::ast::PersonaDecl) -> gcso_config_t {
    let mut config = gcso_config_t::default();
    config.head_dim = 128;
    config.num_heads = 32;
    config.ripa_clamp_max_rad = 0.087266; // 5 degrees
    config.qdps_min_step_rad = 0.01;
    config
}