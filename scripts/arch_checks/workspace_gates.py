# name: scripts/arch_checks/workspace_gates.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Workspace inter-crate gate, downstream vocabulary, and file-hygiene checks."""

from __future__ import annotations

import pathlib
import re

from .common import (
    QUARANTINED_CLI_FILES,
    extract_non_test_code,
    find_banned_public_decls,
    find_function_line_span,
    split_ident_tokens,
    strip_cfg_test_tail,
)
from .core_dag import BANNED_LEGACY_DECL_TOKENS

MAX_PROD_LOC_PER_FILE = 2000

DOWNSTREAM_BANNED_LEGACY_TOKENS: frozenset[str] = BANNED_LEGACY_DECL_TOKENS | frozenset(
    {
        "society",
        "domino",
    }
)

ADAPTER_MUT_BACKDOOR_RE = re.compile(
    r"\.\s*(?:runtime_context_mut|substrate_mut|compute_pool_mut|"
    r"entropy_controller_mut|residual_adapter_mut)\s*\("
)
CORE_MUTABLE_ENGINE_RE = re.compile(
    r"\bliminika_core::(?:runtime::RuntimeContext\b|"
    r"runtime::CoreStateStack\b|"
    r"drivers::(?:SystemSpineState|AnonymousComputePool|EntropyController|"
    r"EgoDifferentialNet|DistortionTension)\b|"
    r"stigmergy::(?:PhaseSteeringEngine|ResidualAdapterEngine)\b)"
    r"|\b(?:RuntimeContext|CoreStateStack|SystemSpineState|AnonymousComputePool|"
    r"EntropyController|EgoDifferentialNet|DistortionTension|"
    r"PhaseSteeringEngine|ResidualAdapterEngine|StigmergicSubstrate)::(?:new|new_boxed|new_empty)\b"
    r"|\.\s*(?:reanchor_baseline_pressure|reset_turn_bypass_state|"
    r"begin_turn_spine_policy|step_hot_path_branchless|"
    r"evaluate_whole_token_steering_bypass_with_spine|"
    r"apply_preset_lanes|allocate_lane|release_all_lanes)\s*\("
)
RAW_C_ABI_CALL_RE = re.compile(r"\blmnk_[a-z0-9_]+\s*\(")
CORE_ABI_FN_PATH_RE = re.compile(r"\bliminika_core::abi::(lmnk_[a-z0-9_]+)\b")
DSL_CORE_REF_RE = re.compile(r"\bliminika_core\b")
ADAPTER_MUT_GATE_CALL_RE = re.compile(
    r"\b(?:TurnOrchestrator::(?:prepare_turn|commit_turn_circuit)|"
    r"ModelSteeringAdapter::new|"
    r"LmnkTimeline::(?:new|load_from_path|from_bytes|save_to_path|record_step|resume_or_rebake)|"
    r"LmnkPackageOwned::load_from_path|"
    r"CompiledTokenCorpus::(?:capture_entries|compile|compile_with_base_prior)|"
    r"ContrastiveAnchorBank::from_entries)\b"
    r"|\.\s*(?:step_decode_control|reanchor_compute_pressure|reset_turn_decode_state|"
    r"orchestrate_layer_rope_with_substrate|orchestrate_layer_rope_with_spillover|"
    r"steer_layer_with_gateway|steer_layer_q_rot|apply_rank1_with_gateway_mut|"
    r"apply_rank1_hidden_fused|orchestrate_layer_rank1_with_substrate|"
    r"calibrate_manifold_probe|apply_slow_loop_telemetry_epoch|"
    r"commit_spillover_experience|reset_horizon_and_core_state|"
    r"step_compute_dispatch_with_spine|commit_compute_exit_reward|"
    r"measure_entropy_state|step_entropy_logits|set_layer_rope_mask|"
    r"set_ego_max_hops|set_guard_hold_tokens|set_max_observe_stride|"
    r"set_pre_bypass_enabled|configure_ego_from_preset|configure_ego_from_natural_language|"
    r"bake_logical_pframe_topology|drain_crystallized_anchors|"
    r"restore_runtime_snapshot|sync_equilibrium_state|restore_from_equilibrium)\s*\("
)
CORE_CRATE_LAYER_RE = re.compile(
    r"\bliminika_core::(liminal|foundation|stigmergy|drivers|runtime|abi)\b"
)
RUST_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
HOST_LLM_BANNED_TOKENS: frozenset[str] = frozenset(
    {
        "candle",
        "tokenizer",
        "tokenizers",
        "prompt",
        "prompts",
        "logit",
        "logits",
        "vocab",
        "rotary",
        "rope",
        "phi4",
        "phi3",
        "gguf",
        "safetensors",
    }
)
HOST_LLM_BANNED_IDENT_EXACT: frozenset[str] = frozenset(
    {
        "kv_cache",
        "compiledtokencorpus",
        "modelsteeringadapter",
        "turnorchestrator",
        "textattractorcorpus",
        "tokensequenceencoder",
        "turnkvcachestepping",
        "ropegeometry",
        "attractorrole",
    }
)
HOST_ADAPTER_LLM_IMPORT_RE = re.compile(
    r"\bcrate::(?:pooling|stratum|corpus_recall|model_adapter|"
    r"turn_orchestrator|reflection|ego_intent|timeline)\b"
)


def scan_host_llm_isolation_in_source(code_only: str, rel_path: str, raw_lines: list[str]) -> list[str]:
    """Scan stripped Rust code under `host/` for disallowed LLM identifier tokens or imports."""
    violations: list[str] = []
    for match in RUST_IDENT_RE.finditer(code_only):
        ident = match.group(0)
        lower_ident = ident.lower()
        tokens = split_ident_tokens(ident)
        banned_toks = sorted(tokens & HOST_LLM_BANNED_TOKENS)
        hit_exact = lower_ident in HOST_LLM_BANNED_IDENT_EXACT or "kv_cache" in lower_ident
        if banned_toks or hit_exact:
            lineno = code_only.count("\n", 0, match.start()) + 1
            snippet = raw_lines[lineno - 1].strip() if 1 <= lineno <= len(raw_lines) else ""
            reason = ", ".join(banned_toks) if banned_toks else ident
            violations.append(
                f"Host LLM isolation (disallowed LLM token `{reason}` in `{ident}`): "
                f"{rel_path}:{lineno}: {snippet}"
            )
    for match in HOST_ADAPTER_LLM_IMPORT_RE.finditer(code_only):
        lineno = code_only.count("\n", 0, match.start()) + 1
        snippet = raw_lines[lineno - 1].strip() if 1 <= lineno <= len(raw_lines) else ""
        violations.append(
            f"Host LLM isolation (disallowed adapter import `{match.group(0)}`): "
            f"{rel_path}:{lineno}: {snippet}"
        )
    return violations


def check_host_llm_isolation(repo_root: pathlib.Path) -> list[str]:
    """Enforce mechanical isolation of `src/adapter/src/host/` from LLM vocabulary and modules."""
    violations: list[str] = []
    host_dir = repo_root / "src" / "adapter" / "src" / "host"
    if not host_dir.is_dir():
        return violations
    for rs_file in sorted(host_dir.rglob("*.rs")):
        if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
            continue
        rel_path = rs_file.relative_to(repo_root).as_posix()
        raw_text = rs_file.read_text(encoding="utf-8")
        code_only = extract_non_test_code(raw_text)
        violations.extend(
            scan_host_llm_isolation_in_source(code_only, rel_path, raw_text.splitlines())
        )
    return violations


def check_downstream_vocabulary(repo_root: pathlib.Path) -> list[str]:
    """Verify legacy declared-identifier bans across `src/adapter/src/` and `src/cli/src/`."""
    violations: list[str] = []
    for crate in ("adapter", "cli"):
        crate_src = repo_root / "src" / crate / "src"
        if not crate_src.is_dir():
            continue
        for rs_file in sorted(crate_src.rglob("*.rs")):
            if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
                continue
            if crate == "cli" and rs_file.name in QUARANTINED_CLI_FILES:
                continue
            rel_path = rs_file.relative_to(repo_root).as_posix()
            code_only = extract_non_test_code(rs_file.read_text(encoding="utf-8"))
            violations.extend(
                find_banned_public_decls(
                    code_only,
                    rel_path,
                    DOWNSTREAM_BANNED_LEGACY_TOKENS,
                    "Downstream vocabulary",
                )
            )
    return violations


def check_workspace_crate_gates(repo_root: pathlib.Path) -> list[str]:
    """Enforce inter-crate boundaries across `src/dsl/`, `src/cli/`, and `examples/candle_eval/`."""
    violations: list[str] = []

    dsl_src = repo_root / "src" / "dsl" / "src"
    if dsl_src.is_dir():
        for rs_file in sorted(dsl_src.rglob("*.rs")):
            if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
                continue
            rel_path = rs_file.relative_to(repo_root).as_posix()
            code_only = extract_non_test_code(rs_file.read_text(encoding="utf-8"))
            for match in DSL_CORE_REF_RE.finditer(code_only):
                lineno = code_only.count("\n", 0, match.start()) + 1
                violations.append(
                    f"Workspace gate (dsl -> liminika_core): {rel_path}:{lineno}: "
                    "disallowed `liminika_core` reference in `liminika-dsl`"
                )

    host_roots = [
        ("cli", repo_root / "src" / "cli" / "src"),
        ("candle_eval", repo_root / "examples" / "candle_eval" / "src"),
    ]
    for crate_tag, src_dir in host_roots:
        if not src_dir.is_dir():
            continue
        for rs_file in sorted(src_dir.rglob("*.rs")):
            if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
                continue
            if crate_tag == "cli" and rs_file.name in QUARANTINED_CLI_FILES:
                continue
            rel_path = rs_file.relative_to(repo_root).as_posix()
            raw_text = rs_file.read_text(encoding="utf-8")
            code_only = extract_non_test_code(raw_text)
            raw_lines = raw_text.splitlines()

            smoke_allowed_lines: set[int] = set()
            if rel_path == "src/cli/src/main.rs":
                smoke_allowed_lines = find_function_line_span(
                    code_only, "run_default_c_abi_smoke"
                )

            for match in ADAPTER_MUT_BACKDOOR_RE.finditer(code_only):
                lineno = code_only.count("\n", 0, match.start()) + 1
                snippet = raw_lines[lineno - 1].strip() if 1 <= lineno <= len(raw_lines) else ""
                violations.append(
                    f"Workspace gate (adapter *_mut backdoor): {rel_path}:{lineno}: {snippet}"
                )

            for match in CORE_MUTABLE_ENGINE_RE.finditer(code_only):
                lineno = code_only.count("\n", 0, match.start()) + 1
                if lineno in smoke_allowed_lines:
                    continue
                snippet = raw_lines[lineno - 1].strip() if 1 <= lineno <= len(raw_lines) else ""
                violations.append(
                    f"Workspace gate (direct core mutable state): {rel_path}:{lineno}: {snippet}"
                )

            for match in RAW_C_ABI_CALL_RE.finditer(code_only):
                lineno = code_only.count("\n", 0, match.start()) + 1
                if lineno in smoke_allowed_lines:
                    continue
                snippet = raw_lines[lineno - 1].strip() if 1 <= lineno <= len(raw_lines) else ""
                violations.append(
                    f"Workspace gate (raw C-ABI call outside allowlist): {rel_path}:{lineno}: "
                    f"{snippet}"
                )

            for match in CORE_ABI_FN_PATH_RE.finditer(code_only):
                sym = match.group(1)
                if sym.endswith("_t"):
                    continue
                lineno = code_only.count("\n", 0, match.start()) + 1
                if lineno in smoke_allowed_lines:
                    continue
                snippet = raw_lines[lineno - 1].strip() if 1 <= lineno <= len(raw_lines) else ""
                violations.append(
                    f"Workspace gate (raw C-ABI symbol outside allowlist): {rel_path}:{lineno}: "
                    f"{snippet}"
                )

    return violations


def check_file_hygiene_and_loc(repo_root: pathlib.Path) -> list[str]:
    """Verify `< 2,000` prod LOC/file, 2-line SPDX headers, and pure LF on source/script trees."""
    violations: list[str] = []
    rust_roots = [
        repo_root / "src",
        repo_root / "examples" / "candle_eval" / "src",
    ]
    for root in rust_roots:
        if not root.is_dir():
            continue
        for rs_file in sorted(root.rglob("*.rs")):
            rel_path = rs_file.relative_to(repo_root).as_posix()
            raw_bytes = rs_file.read_bytes()
            if b"\r\n" in raw_bytes:
                violations.append(f"Line ending (CRLF): {rel_path}")
            text = raw_bytes.decode("utf-8")
            lines = text.splitlines()
            expected_l1 = f"// name: {rel_path}"
            if not lines or lines[0].rstrip() != expected_l1:
                violations.append(f"Header mismatch (line 1): {rel_path} (expected `{expected_l1}`)")
            if len(lines) < 2 or "SPDX-License-Identifier: MIT OR Apache-2.0" not in lines[1]:
                violations.append(f"SPDX header missing (line 2): {rel_path}")
            if rs_file.name != "tests.rs" and "tests" not in rs_file.parts:
                prod_loc = len(strip_cfg_test_tail(lines))
                if prod_loc >= MAX_PROD_LOC_PER_FILE:
                    violations.append(
                        f"Prod LOC ceiling (>= {MAX_PROD_LOC_PER_FILE}): {rel_path}: {prod_loc} prod LOC"
                    )

    scripts_dir = repo_root / "scripts"
    if scripts_dir.is_dir():
        for py_file in sorted(scripts_dir.rglob("*.py")):
            if "__pycache__" in py_file.parts:
                continue
            rel_path = py_file.relative_to(repo_root).as_posix()
            raw_bytes = py_file.read_bytes()
            if b"\r\n" in raw_bytes:
                violations.append(f"Line ending (CRLF): {rel_path}")
            text = raw_bytes.decode("utf-8")
            lines = text.splitlines()
            expected_l1 = f"# name: {rel_path}"
            if not lines or lines[0].rstrip() != expected_l1:
                violations.append(f"Header mismatch (line 1): {rel_path} (expected `{expected_l1}`)")
            if len(lines) < 2 or "SPDX-License-Identifier: MIT OR Apache-2.0" not in lines[1]:
                violations.append(f"SPDX header missing (line 2): {rel_path}")

    return violations


def run_workspace_checks(repo_root: pathlib.Path) -> list[str]:
    """Run all downstream vocabulary, host isolation, workspace gate, and file-hygiene checks."""
    violations: list[str] = []
    violations.extend(check_downstream_vocabulary(repo_root))
    violations.extend(check_host_llm_isolation(repo_root))
    violations.extend(check_workspace_crate_gates(repo_root))
    violations.extend(check_file_hygiene_and_loc(repo_root))
    return violations
