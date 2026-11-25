# name: scripts/arch_checks/coupling_audit.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Non-blocking meta-audit radar (`--audit`) for core-primitive adoption and feedback loops."""

from __future__ import annotations

import pathlib
import re

from .common import QUARANTINED_CLI_FILES, extract_non_test_code, strip_cfg_test_tail

DETERMINISTIC_INFRA_FILES: frozenset[str] = frozenset(
    {
        "src/adapter/src/package.rs",
        "examples/candle_eval/src/metrics.rs",
        "examples/candle_eval/src/cli.rs",
        "examples/candle_eval/src/model.rs",
    }
)

CORE_PRIMITIVE_REF_RE = re.compile(
    r"\b(?:liminika_core::|LacunaPartition8|Lacuna::|BitCut(?:8|32|64)|Scoped|Indivisible|"
    r"SystemSpineState|DistortionTension|EgoDifferentialNet|CoreTelemetryBank|"
    r"StigmergicSubstrate|PhaseAnchorRecord|GatewayFrame|GuardPhaseAnchor|"
    r"ObjectiveFact|SubjectivePersona|TurnOrchestrator|TurnSteeringPlan)\b"
)

FLOAT_THRESHOLD_CMP_RE = re.compile(r"(?:>=|<=|>|<)\s*-?\d+\.\d+(?:_f32|_f64)?")
STATE_COUNTER_RE = re.compile(r"\b(?:streak|counter|dwell|retries|epoch|hits|steps)\b")

DISCARDED_CALL_RE = re.compile(
    r"\blet\s+_\s*=\s*(?P<expr>[^;]{1,220});",
    re.DOTALL,
)
CRITICAL_RETURN_METHODS_RE = re.compile(
    r"\b(?:consolidate_and_evaporate_cold_path|step_decode_control|prepare_turn|"
    r"apply_slow_loop_telemetry_epoch|evaluate_slow_loop_telemetry|"
    r"evaluate_pframe_chain(?:_with_mask)?_branchless|evaluate_ego_pframe_chain_mut|"
    r"step_sandbox_hypothesis_lifecycle|apply_preset_lanes|evaluate_policy|"
    r"update_assimilation_ema|modulate_gateway_frame_at_scale|"
    r"evaluate_whole_token_bypass_with_spine|step_compute_dispatch(?:_with_spine)?|"
    r"orchestrate_layer_rope_with_substrate|orchestrate_layer_rope_with_spillover|"
    r"steer_layer_with_gateway|steer_layer_with_substrate)\b"
)

PUB_CONTROL_FN_RE = re.compile(
    r"\bpub\s+(?:const\s+)?fn\s+(?P<name>(?:step|evaluate|apply|consolidate|settle|"
    r"orchestrate|reanchor|enforce|adapt|intervene|begin_turn|signal_|configure_)[a-z0-9_]*)\b"
)

DOWNWARD_SIGNAL_RE = re.compile(
    r"\b(?:TurnSteeringPlan|SpineStepPlan|ComputeDispatchMask|GatewayFrame|"
    r"CoreStateStack|LacunaPartition8|RecallPolicy|CompiledTokenCorpus|"
    r"GuardPhaseAnchor|ObjectiveFact|SubjectivePersona|EquilibriumState)\b"
)
UPWARD_FEEDBACK_RE = re.compile(
    r"\b(?:step_decode_control|commit_turn_circuit|commit_compute_exit_reward|"
    r"apply_slow_loop_telemetry_epoch|drain_crystallized_anchors|record_step|"
    r"sync_core_layers_from_recall|evaluate_ego_pframe_chain_mut|"
    r"consolidate_and_evaporate_cold_path|step_sandbox_hypothesis_lifecycle|"
    r"reinforce_active_circuit_with_bank|calibrate_manifold_probe|"
    r"signal_compute_instability|update_environment_telemetry|"
    r"step_distortion_tension_hot_path|step_working_set|step_hot_path_branchless|"
    r"ingest_crystallized_anchors|sync_equilibrium_state|restore_from_equilibrium|"
    r"record_tension_feedback)\b"
)


def _iter_prod_files(repo_root: pathlib.Path, dirs: tuple[str, ...]) -> list[tuple[str, pathlib.Path]]:
    out: list[tuple[str, pathlib.Path]] = []
    for rel_dir in dirs:
        base = repo_root / rel_dir
        if not base.is_dir():
            continue
        for rs_file in sorted(base.rglob("*.rs")):
            if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
                continue
            if rel_dir.startswith("src/cli") and rs_file.name in QUARANTINED_CLI_FILES:
                continue
            out.append((rs_file.relative_to(repo_root).as_posix(), rs_file))
    return out


def audit_unintegrated_hotspots(repo_root: pathlib.Path) -> tuple[list[str], list[str]]:
    """Find large non-infra modules with zero core-primitive refs and dense float thresholds."""
    warnings: list[str] = []
    infra_notes: list[str] = []
    scan_dirs = ("src/adapter/src", "src/cli/src", "examples/candle_eval/src")

    for rel_path, rs_file in _iter_prod_files(repo_root, scan_dirs):
        raw = rs_file.read_text(encoding="utf-8")
        prod_loc = len(strip_cfg_test_tail(raw.splitlines()))
        if prod_loc < 400:
            continue
        code = extract_non_test_code(raw)
        core_refs = len(CORE_PRIMITIVE_REF_RE.findall(code))
        float_cmps = len(FLOAT_THRESHOLD_CMP_RE.findall(code))
        counters = len(STATE_COUNTER_RE.findall(code))

        if rel_path in DETERMINISTIC_INFRA_FILES:
            infra_notes.append(
                f"  [infra/deterministic] {rel_path} ({prod_loc} prod LOC, "
                f"core_refs={core_refs}, float_cmps={float_cmps})"
            )
            continue

        if core_refs == 0 and (float_cmps >= 10 or counters >= 10):
            warnings.append(
                f"  [WARN:no-core-refs] {rel_path}: {prod_loc} prod LOC, "
                f"core_refs=0, float_cmps={float_cmps}, state_tokens={counters}"
            )
        elif float_cmps >= 20:
            tag = "WARN:low-core-density" if core_refs < 6 else "INFO:dense-f32-thresholds"
            warnings.append(
                f"  [{tag}] {rel_path}: {prod_loc} prod LOC, "
                f"core_refs={core_refs}, float_cmps={float_cmps}, state_tokens={counters}"
            )
    return warnings, infra_notes


def audit_discarded_returns(repo_root: pathlib.Path) -> list[str]:
    """Detect `let _ =` discards of critical control or crystallization return values."""
    warnings: list[str] = []
    scan_dirs = ("src/adapter/src", "src/cli/src", "examples/candle_eval/src")
    for rel_path, rs_file in _iter_prod_files(repo_root, scan_dirs):
        code = extract_non_test_code(rs_file.read_text(encoding="utf-8"))
        for match in DISCARDED_CALL_RE.finditer(code):
            expr = " ".join(match.group("expr").split())
            hit = CRITICAL_RETURN_METHODS_RE.search(expr)
            if hit is not None:
                lineno = code.count("\n", 0, match.start()) + 1
                warnings.append(
                    f"  [WARN:discarded-return] {rel_path}:{lineno}: "
                    f"`{hit.group(0)}` discarded via `let _ =`"
                )
    return warnings


def audit_unreferenced_core_control_surfaces(repo_root: pathlib.Path) -> list[str]:
    """List public core control methods unreferenced outside `liminika-core`."""
    core_src = repo_root / "src" / "core" / "src"
    if not core_src.is_dir():
        return []

    downstream_chunks: list[str] = []
    for _rel, rs_file in _iter_prod_files(
        repo_root, ("src/adapter/src", "src/cli/src", "examples/candle_eval/src")
    ):
        downstream_chunks.append(extract_non_test_code(rs_file.read_text(encoding="utf-8")))
    downstream_blob = "\n".join(downstream_chunks)

    defined: dict[str, str] = {}
    core_code_chunks: list[str] = []
    for rs_file in sorted(core_src.rglob("*.rs")):
        if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
            continue
        code = extract_non_test_code(rs_file.read_text(encoding="utf-8"))
        core_code_chunks.append(code)
        if "abi" in rs_file.parts:
            continue
        rel = rs_file.relative_to(core_src).as_posix()
        for match in PUB_CONTROL_FN_RE.finditer(code):
            name = match.group("name")
            if name.startswith("lmnk_"):
                continue
            defined.setdefault(name, rel)

    core_blob = "\n".join(core_code_chunks)
    unreferenced: list[str] = []
    for name, rel in sorted(defined.items()):
        if re.search(rf"\b{re.escape(name)}\b", downstream_blob):
            continue
        # Exclude helpers already wired internally inside `liminika-core` (e.g., via RuntimeContext/ABI).
        core_hits = len(re.findall(rf"\b{re.escape(name)}\b", core_blob))
        if core_hits <= 1:
            unreferenced.append(f"  [INFO:core-control-uncalled] {rel}: fn {name}")
    return unreferenced


def audit_module_flow_balance(repo_root: pathlib.Path) -> list[str]:
    """Report per-module Downward (plan/type consume) vs Upward (feedback/crystallize) balance."""
    rows: list[str] = []
    one_way: list[str] = []
    scan_dirs = ("src/adapter/src", "examples/candle_eval/src")
    for rel_path, rs_file in _iter_prod_files(repo_root, scan_dirs):
        if rel_path in DETERMINISTIC_INFRA_FILES:
            continue
        code = extract_non_test_code(rs_file.read_text(encoding="utf-8"))
        down = len(DOWNWARD_SIGNAL_RE.findall(code))
        up = len(UPWARD_FEEDBACK_RE.findall(code))
        if down == 0 and up == 0:
            continue
        tag = "ONE-WAY" if (down > 0 and up == 0) else "CLOSED"
        rows.append(f"  {rel_path:<46} | down={down:>4} | up={up:>4} | {tag}")
        if tag == "ONE-WAY":
            one_way.append(rel_path)

    out = list(rows)
    if one_way:
        out.append(f"  Summary: {len(one_way)} one-way consumer module(s) without upward feedback.")
    else:
        out.append("  Summary: 0 one-way consumer modules detected.")
    return out


def render_coupling_audit(repo_root: pathlib.Path) -> str:
    """Render the complete non-blocking Meta-Audit report (always exits 0)."""
    hotspot_warns, infra_notes = audit_unintegrated_hotspots(repo_root)
    discard_warns = audit_discarded_returns(repo_root)
    uncalled_core = audit_unreferenced_core_control_surfaces(repo_root)
    flow_rows = audit_module_flow_balance(repo_root)

    lines = [
        "=== Architecture Meta-Audit Radar (Non-Blocking Warnings, Exit 0) ===",
        "",
        "1. Unintegrated Control Hotspots (>= 400 prod LOC, low/zero core refs + dense float thresholds):",
    ]
    if hotspot_warns:
        lines.extend(hotspot_warns)
    else:
        lines.append("  None.")
    if infra_notes:
        lines.append("  Exempt deterministic/IO/harness files:")
        lines.extend(infra_notes)

    lines.append("")
    lines.append("2. Discarded Control / Crystallization Returns (`let _ =`):")
    if discard_warns:
        lines.extend(discard_warns)
    else:
        lines.append("  None.")

    lines.append("")
    lines.append(
        f"3. Public `core` Control Surfaces Unreferenced Downstream ({len(uncalled_core)} total, showing up to 15):"
    )
    if uncalled_core:
        lines.extend(uncalled_core[:15])
        if len(uncalled_core) > 15:
            lines.append(f"  ... ({len(uncalled_core) - 15} more)")
    else:
        lines.append("  None.")

    lines.append("")
    lines.append("4. Module Downward-vs-Upward Feedback Balance:")
    lines.extend(flow_rows)
    lines.append("")
    return "\n".join(lines)
