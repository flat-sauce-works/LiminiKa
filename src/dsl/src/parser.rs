// src/dsl/src/parser.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use winnow::ascii::multispace0;
use winnow::combinator::delimited;
use winnow::token::take_until;
use winnow::PResult;
use winnow::Parser;

#[derive(Debug, PartialEq)]
pub struct PersonaDecl {
    pub name: String,
    pub attractors: Vec<String>,
}

// Parses persona definitions from .lmnk DSL
pub fn parse_persona_name<'s>(input: &mut &'s str) -> PResult<&'s str> {
    delimited("persona ", take_until(0.., " "), " extends").parse_next(input)
}