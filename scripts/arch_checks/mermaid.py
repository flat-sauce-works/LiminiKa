# name: scripts/arch_checks/mermaid.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Mermaid DAG/workspace/liminal diagram renderers and core hotspot summaries."""

from __future__ import annotations

import pathlib
import re
from collections import Counter, defaultdict

from .common import QUARANTINED_CLI_FILES, scan_crate_prod_stats, strip_cfg_test_tail
from .workspace_gates import ADAPTER_MUT_GATE_CALL_RE, CORE_CRATE_LAYER_RE, RAW_C_ABI_CALL_RE

CRATE_DEP_RE = re.compile(r"\bcrate::(liminal|foundation|stigmergy|drivers(?:::(?:spine|ego))?|runtime)\b")


def bucket_for_path(core_src: pathlib.Path, rs_file: pathlib.Path) -> str | None:
    """Map a core source file to its Y-shaped DAG node bucket."""
    rel = rs_file.relative_to(core_src).parts
    if not rel:
        return None
    top = rel[0]
    if top == "drivers" and len(rel) >= 2 and rel[1] in ("spine", "ego"):
        return f"drivers::{rel[1]}"
    if top in ("liminal", "foundation", "stigmergy", "drivers", "runtime"):
        return top
    return None


def collect_dag_metrics(
    core_src: pathlib.Path,
) -> tuple[Counter[str], Counter[str], dict[tuple[str, str], int]]:
    """Collect per-layer file counts, LOC, and inter-layer `crate::` import edge counts."""
    files_per_node: Counter[str] = Counter()
    loc_per_node: Counter[str] = Counter()
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)

    for rs_file in sorted(core_src.rglob("*.rs")):
        src_bucket = bucket_for_path(core_src, rs_file)
        if src_bucket is None:
            continue
        text = rs_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        files_per_node[src_bucket] += 1
        loc_per_node[src_bucket] += len(lines)
        for _lineno, line in strip_cfg_test_tail(lines):
            code = line.split("//", 1)[0]
            for target in CRATE_DEP_RE.findall(code):
                dst_bucket = target if target in ("drivers::spine", "drivers::ego") else target.split("::")[0]
                if dst_bucket == "drivers" and src_bucket in ("drivers::spine", "drivers::ego"):
                    dst_bucket = "drivers"
                if dst_bucket != src_bucket:
                    edge_counts[(src_bucket, dst_bucket)] += 1

    return files_per_node, loc_per_node, edge_counts


def render_mermaid(core_src: pathlib.Path) -> str:
    """Render the live Y-shaped 5-layer DAG as a Mermaid `flowchart TD` diagram."""
    files_per_node, loc_per_node, edge_counts = collect_dag_metrics(core_src)

    def node_label(key: str, title: str) -> str:
        return f"{title} - {files_per_node[key]} files, {loc_per_node[key]} LOC"

    node_ids = {
        "liminal": "L0_liminal",
        "foundation": "L1_foundation",
        "stigmergy": "L2_stigmergy",
        "drivers::spine": "L3_spine",
        "drivers::ego": "L3_ego",
        "drivers": "L3_drivers_hub",
        "runtime": "L4_runtime",
    }

    lines = [
        "flowchart TD",
        f'    L0_liminal["{node_label("liminal", "Lv.0 liminal (Boundary Axioms & AbstractDomain)")}"]',
        f'    L1_foundation["{node_label("foundation", "Lv.1 foundation (Pure SWAR, RoPE & Cohomology)")}"]',
        f'    L2_stigmergy["{node_label("stigmergy", "Lv.2 stigmergy (Substrate & PhaseAnchors)")}"]',
        '    subgraph L3_drivers["Lv.3 drivers (Orthogonal Axes & Tension)"]',
        f'        L3_spine["{node_label("drivers::spine", "drivers/spine (Reflex & EntropyController)")}"]',
        f'        L3_ego["{node_label("drivers::ego", "drivers/ego (Intent & EgoDifferentialNet)")}"]',
        f'        L3_drivers_hub["{node_label("drivers", "drivers/mod (DistortionTension)")}"]',
        "    end",
        f'    L4_runtime["{node_label("runtime", "Lv.4 runtime (Context, Gateway, Horizon, ABI)")}"]',
        "",
    ]

    ordered_pairs = [
        ("liminal", "foundation"),
        ("liminal", "stigmergy"),
        ("foundation", "stigmergy"),
        ("liminal", "drivers::spine"),
        ("foundation", "drivers::spine"),
        ("stigmergy", "drivers::spine"),
        ("liminal", "drivers::ego"),
        ("foundation", "drivers::ego"),
        ("stigmergy", "drivers::ego"),
        ("drivers", "drivers::spine"),
        ("drivers", "drivers::ego"),
        ("liminal", "runtime"),
        ("foundation", "runtime"),
        ("stigmergy", "runtime"),
        ("drivers::spine", "runtime"),
        ("drivers::ego", "runtime"),
        ("drivers", "runtime"),
    ]
    for provider, consumer in ordered_pairs:
        refs = edge_counts.get((consumer, provider), 0)
        if refs > 0:
            lines.append(f"    {node_ids[provider]} -->|{refs} refs| {node_ids[consumer]}")

    return "\n".join(lines) + "\n"


def render_mermaid_workspace(repo_root: pathlib.Path) -> str:
    """Render workspace crates -> adapter canonical gates -> core 5-layer DAG as Mermaid."""
    core_src = repo_root / "src" / "core" / "src"
    files_per_node, loc_per_node, core_edges = collect_dag_metrics(core_src)

    dsl_files, dsl_loc, _ = scan_crate_prod_stats(repo_root / "src" / "dsl" / "src")
    cli_files, cli_loc, cli_code = scan_crate_prod_stats(
        repo_root / "src" / "cli" / "src", QUARANTINED_CLI_FILES
    )
    eval_files, eval_loc, eval_code = scan_crate_prod_stats(
        repo_root / "examples" / "candle_eval" / "src"
    )
    ad_files, ad_loc, ad_code = scan_crate_prod_stats(repo_root / "src" / "adapter" / "src")
    host_files, host_loc, _ = scan_crate_prod_stats(
        repo_root / "src" / "adapter" / "src" / "host"
    )

    cli_dsl_refs = len(re.findall(r"\bliminika_dsl\b", cli_code))
    cli_adapter_gates = len(ADAPTER_MUT_GATE_CALL_RE.findall(cli_code))
    cli_core_vals = len(CORE_CRATE_LAYER_RE.findall(cli_code))
    cli_cabi_smoke = len(RAW_C_ABI_CALL_RE.findall(cli_code))

    eval_adapter_gates = len(ADAPTER_MUT_GATE_CALL_RE.findall(eval_code))
    eval_core_vals = len(CORE_CRATE_LAYER_RE.findall(eval_code))

    ad_core_counts: Counter[str] = Counter(CORE_CRATE_LAYER_RE.findall(ad_code))
    ad_runtime_gates = ad_core_counts["runtime"]
    ad_liminal_vals = ad_core_counts["liminal"]
    ad_foundation_vals = ad_core_counts["foundation"]
    ad_stigmergy_vals = ad_core_counts["stigmergy"]
    ad_drivers_vals = ad_core_counts["drivers"]

    def plural_files(count: int) -> str:
        return f"{count} file" if count == 1 else f"{count} files"

    def core_label(key: str, title: str) -> str:
        return f"{title} - {plural_files(files_per_node[key])}, {loc_per_node[key]} LOC"

    lines = [
        "flowchart TD",
        '    %% Legend: solid arrow (-->) = Mutable State Gate; dashed arrow (-.->) = Value/Pod Ref',
        '    subgraph WS_HOSTS["Workspace Host Crates & Evaluation Harness"]',
        f'        W_dsl["liminika-dsl (Parser/AST) - {plural_files(dsl_files)}, {dsl_loc} prod LOC"]',
        f'        W_cli["liminika-cli (CLI & C-ABI Smoke) - {plural_files(cli_files)}, {cli_loc} prod LOC"]',
        (
            f'        W_eval["candle_eval (Thin Eval Harness) - '
            f'{plural_files(eval_files)}, {eval_loc} prod LOC"]'
        ),
        "    end",
        '    subgraph WS_ADAPTER["Canonical Adapter Gate (liminika-adapter)"]',
        (
            '        W_adapter["ModelSteeringAdapter + TurnOrchestrator + CompiledTokenCorpus '
            f'- {plural_files(ad_files - host_files)}, {ad_loc - host_loc} prod LOC"]'
        ),
        (
            '        W_host["App-Agnostic Host (host/ - LLM-isolated) '
            f'- {plural_files(host_files)}, {host_loc} prod LOC"]'
        ),
        "    end",
        '    subgraph WS_CORE["liminika-core (#![no_std] 5-Layer Unidirectional DAG)"]',
        f'        L0_liminal["{core_label("liminal", "Lv.0 liminal (Axioms & Lacuna)")}"]',
        f'        L1_foundation["{core_label("foundation", "Lv.1 foundation (SWAR, RoPE, Pod)")}"]',
        f'        L2_stigmergy["{core_label("stigmergy", "Lv.2 stigmergy (Substrate & Anchors)")}"]',
        '        subgraph WS_L3["Lv.3 drivers (Orthogonal Axes & Tension)"]',
        f'            L3_spine["{core_label("drivers::spine", "Lv.3 drivers/spine (Spine SSOT)")}"]',
        f'            L3_ego["{core_label("drivers::ego", "Lv.3 drivers/ego (Ego P-Frame)")}"]',
        f'            L3_drivers_hub["{core_label("drivers", "Lv.3 drivers/mod (Tension)")}"]',
        "        end",
        f'        L4_runtime["{core_label("runtime", "Lv.4 runtime (Context, Gateway, ABI)")}"]',
        "    end",
        "",
    ]

    if cli_dsl_refs > 0:
        lines.append(f'    W_cli -.->|"Value/AST ({cli_dsl_refs} refs)"| W_dsl')
    lines.append(f'    W_cli -->|"Mutable Gate ({cli_adapter_gates} calls)"| W_adapter')
    if cli_core_vals > 0:
        lines.append(f'    W_cli -.->|"Value/Pod & Inspect ({cli_core_vals} refs)"| L4_runtime')
    if cli_cabi_smoke > 0:
        lines.append(
            f'    W_cli -.->|"C-ABI Smoke Allowlist ({cli_cabi_smoke} calls)"| L4_runtime'
        )
    lines.append(f'    W_eval -->|"Mutable Gate ({eval_adapter_gates} calls)"| W_adapter')
    if eval_core_vals > 0:
        lines.append(f'    W_eval -.->|"Value/Pod & Anchors ({eval_core_vals} refs)"| L4_runtime')

    lines.append(f'    W_adapter -->|"Mutable RuntimeContext Gate ({ad_runtime_gates} refs)"| L4_runtime')
    if host_files > 0:
        lines.append('    W_adapter -.->|"Host Working-Set Bridge"| W_host')
        lines.append('    W_host -.->|"Core Pod & Container View"| L4_runtime')
    if ad_liminal_vals > 0:
        lines.append(f'    W_adapter -.->|"Value/Pod ({ad_liminal_vals} refs)"| L0_liminal')
    if ad_foundation_vals > 0:
        lines.append(f'    W_adapter -.->|"Value/Trait ({ad_foundation_vals} refs)"| L1_foundation')
    if ad_stigmergy_vals > 0:
        lines.append(f'    W_adapter -.->|"Value/Trait ({ad_stigmergy_vals} refs)"| L2_stigmergy')
    if ad_drivers_vals > 0:
        lines.append(f'    W_adapter -.->|"Value/Pod ({ad_drivers_vals} refs)"| L3_spine')

    node_ids = {
        "liminal": "L0_liminal",
        "foundation": "L1_foundation",
        "stigmergy": "L2_stigmergy",
        "drivers::spine": "L3_spine",
        "drivers::ego": "L3_ego",
        "drivers": "L3_drivers_hub",
        "runtime": "L4_runtime",
    }
    ordered_pairs = [
        ("liminal", "foundation"),
        ("liminal", "stigmergy"),
        ("foundation", "stigmergy"),
        ("liminal", "drivers::spine"),
        ("foundation", "drivers::spine"),
        ("stigmergy", "drivers::spine"),
        ("liminal", "drivers::ego"),
        ("foundation", "drivers::ego"),
        ("stigmergy", "drivers::ego"),
        ("drivers", "drivers::spine"),
        ("drivers", "drivers::ego"),
        ("liminal", "runtime"),
        ("foundation", "runtime"),
        ("stigmergy", "runtime"),
        ("drivers::spine", "runtime"),
        ("drivers::ego", "runtime"),
        ("drivers", "runtime"),
    ]
    for provider, consumer in ordered_pairs:
        refs = core_edges.get((consumer, provider), 0)
        if refs > 0:
            lines.append(f"    {node_ids[provider]} -->|{refs} refs| {node_ids[consumer]}")

    lines.append("    L2_stigmergy ~~~ L3_drivers_hub")
    lines.append("    L3_spine ~~~ L4_runtime")
    lines.append("    L3_ego ~~~ L4_runtime")

    return "\n".join(lines) + "\n"


def format_hotspots(core_src: pathlib.Path, top_n: int = 12) -> str:
    """Analyze complexity, `unsafe`, and boundary density across production files in `src/core/src`."""
    rows: list[tuple[int, int, int, int, int, str]] = []
    for rs_file in sorted(core_src.rglob("*.rs")):
        if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
            continue
        lines = rs_file.read_text(encoding="utf-8").splitlines()
        total_loc = len(lines)
        non_test = strip_cfg_test_tail(lines)
        prod_loc = len(non_test)
        unsafe_hits = 0
        raw_mask_hits = 0
        scoped_refs = 0
        for _lineno, line in non_test:
            code = line.split("//", 1)[0]
            if re.search(r"\bunsafe\b", code):
                unsafe_hits += 1
            if "& !" in code:
                raw_mask_hits += 1
            if re.search(r"\b(?:Scoped|Indivisible)\b", code):
                scoped_refs += 1
        rel = rs_file.relative_to(core_src).as_posix()
        rows.append((prod_loc, total_loc, unsafe_hits, raw_mask_hits, scoped_refs, rel))

    rows.sort(reverse=True)
    header = [
        f"Core Architecture Hotspots (top {min(top_n, len(rows))} production files by non-test LOC):",
        f"{'File':<46} | {'ProdLOC':>7} | {'TotalLOC':>8} | {'Unsafe':>6} | {'Raw&!':>5} | {'Axiom':>5}",
        "-" * 92,
    ]
    for prod_loc, total_loc, unsafe_hits, raw_mask_hits, scoped_refs, rel in rows[:top_n]:
        header.append(
            f"{rel:<46} | {prod_loc:>7} | {total_loc:>8} | {unsafe_hits:>6} | {raw_mask_hits:>5} | {scoped_refs:>5}"
        )
    total_files = len(rows)
    total_prod = sum(r[0] for r in rows)
    total_all = sum(r[1] for r in rows)
    total_unsafe = sum(r[2] for r in rows)
    header.append("-" * 92)
    header.append(
        f"Summary: {total_files} prod files | {total_prod} prod LOC "
        f"({total_all} total LOC) | {total_unsafe} non-test unsafe tokens"
    )
    return "\n".join(header) + "\n"


def collect_liminal_map_metrics(core_src: pathlib.Path) -> dict[str, int]:
    """Scan live `src/core/src` production files for Abstraction <-> Concretion metrics."""
    metrics: dict[str, int] = {
        "foundation_files": 0,
        "foundation_prod_loc": 0,
        "axiom_files": 0,
        "axiom_gate_refs": 0,
        "grounded_iframe_refs": 0,
        "active_concretion_refs": 0,
        "abstract_bypass_refs": 0,
        "exterior_sandbox_refs": 0,
        "pframe_ego_refs": 0,
        "runtime_gate_refs": 0,
    }
    patterns = {
        "axiom_gate_refs": re.compile(
            r"\b(?:Scoped|Indivisible|rescope_from_|cut_by_lacuna|"
            r"bridge_to_shared|BitCut(?:8|32|64)?)\b"
        ),
        "grounded_iframe_refs": re.compile(
            r"\b(?:Grounded|PhaseAnchorRecord|PhaseProfile64|PhaseProfileRef64|"
            r"PhaseWordsRef64)\b"
        ),
        "active_concretion_refs": re.compile(
            r"\b(?:ActiveScope|PhaseSteeringEngine|ResidualAdapterEngine|"
            r"PhaseDeltaSlice|ResidualScale|Indivisible\s*<\s*Active)\b"
        ),
        "abstract_bypass_refs": re.compile(
            r"\b(?:Lacuna::Unobserved|Lacuna::SelfRef|"
            r"stationary_shortcut_ready|whole_token_steering_bypass)\b"
        ),
        "exterior_sandbox_refs": re.compile(
            r"\b(?:Lacuna::Exterior|slot_sandbox_mask|settle_sandbox_lane|"
            r"SandboxLifecycleEvent)\b"
        ),
        "pframe_ego_refs": re.compile(
            r"\b(?:DistortionDelta|EgoDifferentialNet|DistortionTension)\b"
        ),
        "runtime_gate_refs": re.compile(
            r"\b(?:LmnkContainerView|SystemSpineState|pinned_compute_mask|"
            r"CORE_TOPOLOGY)\b"
        ),
    }

    for rs_file in sorted(core_src.rglob("*.rs")):
        if rs_file.name == "tests.rs" or "tests" in rs_file.parts:
            continue
        rel = rs_file.relative_to(core_src).parts
        non_test = strip_cfg_test_tail(rs_file.read_text(encoding="utf-8").splitlines())
        if rel and rel[0] == "foundation":
            metrics["foundation_files"] += 1
            metrics["foundation_prod_loc"] += len(non_test)
        if len(rel) >= 2 and rel[0] == "liminal" and rel[1] == "axiom":
            metrics["axiom_files"] += 1
        for _lineno, line in non_test:
            code = line.split("//", 1)[0]
            for key, pat in patterns.items():
                metrics[key] += len(pat.findall(code))

    return metrics


def render_liminal_map(core_src: pathlib.Path) -> str:
    """Render the static `liminal` Abstraction <-> Concretion Mermaid map and table."""
    m = collect_liminal_map_metrics(core_src)
    lines = [
        "```mermaid",
        "flowchart LR",
        '    subgraph ABS["Abstract Realm (Low-Cost Ground State)"]',
        (
            '        ABS_FND["Lv.1 foundation/ Pure Compute '
            f'({m["foundation_files"]} files, {m["foundation_prod_loc"]} prod LOC)"]'
        ),
        (
            '        ABS_IFR["Indivisible(Grounded) I-Frame PhaseAnchorRecord '
            f'({m["grounded_iframe_refs"]} refs)"]'
        ),
        (
            '        ABS_BYP["Lacuna::Unobserved and SelfRef Bypass '
            f'({m["abstract_bypass_refs"]} refs)"]'
        ),
        "    end",
        '    subgraph LIM["liminal Boundary Gates (Lv.0 Axiom + Lv.4 Runtime)"]',
        (
            '        LIM_AX["Lv.0 liminal/axiom seal / rescope / BitCut '
            f'({m["axiom_files"]} files, {m["axiom_gate_refs"]} refs)"]'
        ),
        (
            '        LIM_RT["Lv.4 LmnkContainerView + SystemSpineState '
            f'({m["runtime_gate_refs"]} refs)"]'
        ),
        "    end",
        '    subgraph CON["Concrete Realm (High-Cost Excited State)"]',
        (
            '        CON_ACT["Scoped + Indivisible(Active) RoPE / Rank-1 '
            f'({m["active_concretion_refs"]} refs)"]'
        ),
        (
            '        CON_EGO["EgoDifferentialNet P-Frame + DistortionTension '
            f'({m["pframe_ego_refs"]} refs)"]'
        ),
        (
            '        CON_EXT["Lacuna::Exterior Isolated Sandbox Lanes '
            f'({m["exterior_sandbox_refs"]} refs)"]'
        ),
        "    end",
        '    ABS_FND -.->|"Pure Kernel Supply (Zero-Scoped)"| CON_ACT',
        '    ABS_IFR <-->|"seal / ground I-Frame"| LIM_AX',
        '    ABS_BYP <-->|"4p(1-p) Aperture & Spine Bypass"| LIM_RT',
        '    LIM_AX -->|"rescope_from_delta / Activate"| CON_ACT',
        '    LIM_RT <-->|"Host Pin Floor & Tension Damping"| CON_EGO',
        '    LIM_AX <-->|"cut_by_lacuna / settle_sandbox_lane"| CON_EXT',
        "```",
        "",
        "## Static `liminal` Abstraction <-> Concretion Correspondence Table",
        "",
        (
            "| Abstract Realm (Ground State) | `liminal` Boundary Gate / Op | "
            "Concrete Realm (Excited State) | Live `src/core/src` Counts | "
            "`CoreTelemetryBank` Counter |"
        ),
        "| :--- | :--- | :--- | ---: | :--- |",
        (
            "| Lv.1 `foundation/` non-`Scoped` pure compute | "
            "Direct pure kernel call (no `liminal/axiom` import) | "
            "Lv.2+ `Indivisible(Active)` RoPE / Rank-1 engines | "
            f"{m['foundation_files']} files / {m['foundation_prod_loc']} LOC -> "
            f"{m['active_concretion_refs']} active refs | "
            "`stigmergy.decision_counts[0..7]` |"
        ),
        (
            "| `Indivisible(Grounded)` I-Frame `PhaseAnchorRecord` | "
            "`seal` / `rescope_from_boundary` / `CORE_TOPOLOGY` | "
            "`EgoDifferentialNet` P-Frame `DistortionDelta` | "
            f"{m['grounded_iframe_refs']} grounded refs <-> "
            f"{m['pframe_ego_refs']} P-Frame/tension refs | "
            "`pframe_chain_evals`, `pframe_total_hops`, `ego.intervention_count` |"
        ),
        (
            "| `Lacuna::SelfRef` stationary bypass | "
            "`SystemSpineState` + `pinned_compute_mask` | "
            "`DistortionTension` damped/excited steering | "
            f"{m['abstract_bypass_refs']} bypass refs <-> "
            f"{m['runtime_gate_refs']} spine/container refs | "
            "`whole_token_bypass_events`, `liminal.stationary_shortcut_hits`, "
            "`immediate_tension_damps`, `slow_tension_excites` |"
        ),
        (
            "| `Lacuna::Unobserved` zero-aperture | "
            "`cut_by_lacuna` / `LacunaPartition8` | "
            "`Lacuna::Exterior` sandbox (`slot_sandbox_mask`) | "
            f"{m['axiom_gate_refs']} axiom gate refs <-> "
            f"{m['exterior_sandbox_refs']} sandbox refs | "
            "`unobserved_aperture_zeros`, `exterior_sandbox_stages`, "
            "`liminal.slide_in_count`, `liminal.slide_out_count` |"
        ),
    ]
    return "\n".join(lines) + "\n"
