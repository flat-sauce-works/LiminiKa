# name: scripts/arch_checks/__init__.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Modular architecture verification, workspace gate, Mermaid, and meta-audit checks."""

from __future__ import annotations

from .common import CORE_SRC, QUARANTINED_CLI_FILES, REPO_ROOT
from .core_dag import (
    check_abstract_domain_invariant,
    check_cross_observer_scoped_boundaries,
    check_topology_matrix_parity,
    check_vocabulary_ownership,
    run_core_checks,
    scan_scoped_boundary_violations_in_source,
    scan_vocabulary_ownership_in_source,
)
from .coupling_audit import render_coupling_audit
from .mermaid import (
    bucket_for_path,
    collect_dag_metrics,
    collect_liminal_map_metrics,
    format_hotspots,
    render_liminal_map,
    render_mermaid,
    render_mermaid_workspace,
)
from .workspace_gates import (
    check_downstream_vocabulary,
    check_file_hygiene_and_loc,
    check_workspace_crate_gates,
    run_workspace_checks,
)

__all__ = [
    "CORE_SRC",
    "QUARANTINED_CLI_FILES",
    "REPO_ROOT",
    "bucket_for_path",
    "check_abstract_domain_invariant",
    "check_cross_observer_scoped_boundaries",
    "check_downstream_vocabulary",
    "check_file_hygiene_and_loc",
    "check_topology_matrix_parity",
    "check_vocabulary_ownership",
    "check_workspace_crate_gates",
    "collect_dag_metrics",
    "collect_liminal_map_metrics",
    "format_hotspots",
    "render_coupling_audit",
    "render_liminal_map",
    "render_mermaid",
    "render_mermaid_workspace",
    "run_core_checks",
    "run_workspace_checks",
    "scan_scoped_boundary_violations_in_source",
    "scan_vocabulary_ownership_in_source",
]
