# LiminiKa Declarative DSL Specification (`.lmnk`, `v0.4.0`)

## 1. Scope & Implementation Status

The `liminika-dsl` crate (`src/dsl/`) implements a zero-dependency recursive-descent lexer, parser, AST, and semantic validator for `.lmnk` specification files (`parse_and_validate_lmnk`).

- **Implemented**: Parsing and semantic validation of `target`, `persona` (`extends`, `attractors` with `attract` / `repel`), and `session` (`using`, optional `on`, optional `mode`) declarations, verified via `liminika check <file.lmnk>`.
- **Not yet implemented**: Direct `.lmnk`-to-`.lmnkc`/`.lmnkp` compilation. Binary containers are currently built via `liminika bake` from evaluation fixtures.

---

## 2. Lexical Rules & Resource Bounds

### 2.1 Comments & Tokens

- **Line comments**: Both `# ...` and `// ...` skip to the next `\n`.
- **Whitespace**: ASCII whitespace separates tokens.
- **Punctuation**: `{`, `}`, `@`, `=`.
- **Identifiers**: Start with `[A-Za-z_]` followed by zero or more `[A-Za-z0-9_-]`.
- **String literals**: Enclosed in double quotes `"..."` on a single line. Embedded newlines, NUL (`0x00`), ASCII control bytes (`< 0x20`), and DEL (`0x7F`) are rejected.
- **Numeric literals**: Optional leading `-`, digits with optional decimal point and scientific exponent (`e`/`E` with optional `+`/`-`), parsed as finite `f32`. Non-finite values (e.g., `1e50`) are rejected.

### 2.2 Hard Parser & Validator Limits

| Constant | Limit | Enforced At |
| :--- | ---: | :--- |
| `LMNK_DSL_MAX_INPUT_BYTES` | `1,048,576` B (`1 MiB`) | Lexer (`tokenize`) |
| `LMNK_DSL_MAX_TOKENS` | `65,536` tokens | Lexer (`push_token`) |
| `LMNK_DSL_MAX_DECLARATIONS` | `256` top-level declarations | Parser & Validator |
| `LMNK_DSL_MAX_RULES_PER_PERSONA` | `1,024` combined `attract` + `repel` rules | Parser & Validator |

---

## 3. Grammar & AST

A `.lmnk` file consists of one or more top-level declarations (`Declaration::{Target, Persona, Session}`). Empty files are rejected.

```ebnf
program        := declaration+ ;
declaration    := target_decl | persona_decl | session_decl ;

target_decl    := "target" IDENT "{" target_field* "}" ;
target_field   := "model" "=" STRING
                | "vram_budget_mb" "=" NUMBER
                | "export" "=" STRING ;

persona_decl   := "persona" IDENT [ "extends" STRING ] "{" [ attractors_blk ] "}" ;
attractors_blk := "attractors" "{" rule* "}" ;
rule           := "attract" STRING "@" NUMBER
                | "repel" STRING "@" NUMBER ;

session_decl   := "session" IDENT "using" IDENT [ "on" IDENT ] "{" [ "mode" "=" IDENT ] "}" ;
```

### 3.1 AST Types (`src/dsl/src/ast.rs`)

- **`TargetDecl`**: `name: String`, `model_path: String`, `vram_budget_mb: u32`, `export_path: String`.
- **`PersonaDecl`**: `name: String`, `parent_preset: Option<String>`, `attractors: AttractorsBlock`.
- **`AttractorsBlock`**: `attract_rules: Vec<AttractRule>` (`source_path: String`, `weight: f32`), `repel_rules: Vec<RepelRule>` (`target_pattern: String`, `weight: f32`).
- **`SessionDecl`**: `name: String`, `persona_ref: String`, `target_ref: Option<String>`, `mode: Option<String>`.
- **`CompiledDslProgram`**: Validated summary grouping `targets`, `personas`, and `sessions`.

---

## 4. Semantic Validation Rules (`compile_lmnk_declarations`)

1. **Non-empty specification**: The program must contain at least one `persona` or `session` declaration (a `target`-only file is rejected).
2. **Unique names & fields**:
   - Declaration names must be unique within each namespace (`target`, `persona`, `session`).
   - Duplicate fields inside `target` (`model`, `vram_budget_mb`, `export`), duplicate `attractors` blocks inside `persona`, and duplicate `mode` fields inside `session` are rejected.
3. **Numeric constraints**:
   - `vram_budget_mb` must be a finite integer in `1..=65536`.
   - Every `attract` and `repel` weight must be finite and strictly positive (`> 0.0`).
4. **Path & string sanitization**:
   - Paths (`target.model`, `target.export`, `persona.extends`, `attract` source path) must be non-empty, free of ASCII control bytes (`< 0x20`, `0x7F`), and must not contain any `..` path traversal segment (split on `/` or `\`).
   - `repel` target patterns must be non-empty after trimming and free of ASCII control bytes.
5. **Reference resolution**:
   - Every `session ... using <Persona>` must reference a `persona` declared in the same program.
   - Every optional `session ... on <Target>` must reference a `target` declared in the same program.

---

## 5. Verified Example

From [`examples/dsl/limina_poc.lmnk`](../../examples/dsl/limina_poc.lmnk):

```text
// name: examples/dsl/limina_poc.lmnk
// SPDX-License-Identifier: MIT OR Apache-2.0

persona Limina extends "presets/persona/limina_base" {
    attractors {
        attract "presets/knowledge/core_baseline.lmnk" @ 0.8
    }
}

session DesktopAssistant using Limina {
    mode = creative_exploration
}
```

Validate with the CLI:

```bash
cargo run --locked -p liminika-cli -- check examples/dsl/limina_poc.lmnk
```
