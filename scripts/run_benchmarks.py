# name: scripts/run_benchmarks.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""LiminiKa Unified Benchmark Runner & Pipeline Coordinator.

Executes Rust micro-benchmarks via Cargo with optimal zero-option defaults so
running `python scripts/run_benchmarks.py` with no arguments automatically
achieves maximum determinism, lowest CoV/MAD jitter, peak SIMD throughput, and
generates Criterion's interactive HTML index (`target/criterion/report/index.html`):
- Clean Core vs. Provisional `.gcso` Separation: the default `standard` profile
  runs all finalized Core (`PhaseSteering` + `EntropyControl`) benchmark groups while isolating
  `Provisional_GCSO_Container` via `LIMINIKA_RUN_PROVISIONAL_GCSO=0` and
  Criterion positive group-prefix regex filtering (avoiding `libtest`'s
  `--skip` flag, which Criterion's `configure_from_args()` rejects with exit
  code 2), unless `--include-provisional-gcso`, `--profile provisional-gcso`,
  or `--profile all` is specified.
- Two-stage Pre-Compilation & Thermal/IO Settle (`--precompile` enabled by
  default; disable via `--no-precompile`): builds all benchmark binaries via
  `cargo bench --no-run` first, flushes OS filesystem buffers, runs Python GC,
  and pauses briefly so compiler disk I/O and CPU turbo-boost thermal decay do
  not contaminate the first benchmark group.
- OS Process Priority & Power Throttling Stabilization (`--high-priority`
  enabled by default; disable via `--no-high-priority`): elevates scheduling
  priority (`HIGH_PRIORITY_CLASS` on Windows, `os.nice(-10)` on POSIX),
  disables Windows EcoQoS Power Throttling (`PROCESS_POWER_THROTTLING_INFORMATION`)
  to prevent P-core -> E-core migration, and disables dynamic priority boost
  jitter (`SetProcessPriorityBoost`).
- Topology-Aware Physical CPU Core Pinning (`--pin-core` auto-selected by
  default; disable via `--no-pin-core`): parses Linux sysfs
  `thread_siblings_list` (or `os.sched_getaffinity` / `os.cpu_count()`) to pin
  execution to a dedicated non-zero primary physical core (avoiding Core #0 OS
  IRQ/DPC noise and SMT sibling contention).
- Deterministic Codegen & Thread-Pool Pinning (`--native-cpu` enabled by
  default; disable via `--no-native-cpu`): sets `CARGO_INCREMENTAL=0`,
  `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `-C opt-level=3 -C target-cpu=native
  -C codegen-units=1`.
- Criterion Noise-Threshold, Settle Window & HTML Index Generation
  (`--noise-threshold 0.025`, `--settle-ms 400`, and `--plot` enabled by default;
  pass `--noplot` for headless CI): suppresses sub-30ns RDTSC timer-quantum and
  background desktop OS context-switch false alarms while respecting per-group
  Rust `sample_size` configurations.
"""

import argparse
import ctypes
import gc
import os
import platform
import shutil
import subprocess
import sys
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

# Pre-defined execution profiles for tailored benchmarking runs
PROFILES: dict[str, dict[str, str | None]] = {
    "standard": {
        "description": "Finalized Core benchmark suite covering PhaseSteering, EntropyControl, Online Toroidal k-Means, Coarse-to-Fine Routing, Unified-Field, and Stigmergic Micro-Swarm hot paths (excludes provisional .gcso container benchmarks by default)",
        "filter": None,
        "skip": "Provisional_GCSO_Container",
    },
    "provisional-gcso": {
        "description": "Isolated pre-v1.0 provisional .gcso storage 4-section container serialization, CRC32 verification, and zero-copy mapping benchmarks",
        "filter": "Provisional_GCSO_Container",
        "skip": None,
    },
    "quick": {
        "description": "Fast smoke test run for build validation and core PhaseSteering/EntropyControl kernel verification",
        "filter": "steering_32b|safe_ripa|normal_mode|eval_cvoid|swarm_quorum|unified_field|bayesian|toroidal_kmeans|coarse_to_fine|top2_liminal|merge_and_prune|autonomous_swarm_aperture|gop_budget",
        "skip": "Provisional_GCSO_Container",
    },
    "ripa": {
        "description": "Focused comparison of RIPA LUT/pre-inverted trig vs Full-Channel RoPE, Swarm Quorum, Unified Field, Slerp, and PSPM Pseudo-MoE",
        "filter": "steering_32b|safe_ripa|swarm_quorum|closed_loop_swarm|unified_field|slot_bank_swarm|slerp_geodesic|pspm_subhead|autonomous_swarm_aperture|ripa_extreme",
        "skip": "Provisional_GCSO_Container",
    },
    "kmeans-coarse-to-fine": {
        "description": "Focuses on Classical Online Toroidal k-Means / DP-Means macro clustering and Coarse-to-Fine routing (8 candidates across 4 centroids)",
        "filter": "toroidal_kmeans|coarse_to_fine|top2_liminal|merge_and_prune",
        "skip": "Provisional_GCSO_Container",
    },
    "bayesian-limina": {
        "description": "Focuses on Joint Probability x Attention phase interference, ActionHub Bayesian pointer-trail rewiring, Limina bridging, Unified Field, and Pitchfork Tunneling",
        "filter": "bayesian|liminal|unified_field|inv_temp|tunneling|coherence|coarse_to_fine|top2_liminal",
        "skip": "Provisional_GCSO_Container",
    },
    "swarm-stigmergy": {
        "description": "Focuses on Core 8-Cell SWAR Micro-Swarm quorum, Unified-Field Swarm, Bayesian Pointer Rewiring, Limina Bridging, Online Toroidal k-Means, Coarse-to-Fine routing, 8-Slot Bank lateral recruitment, PSPM Pseudo-MoE, Visual Co-Watching, Metastable Tunneling, and Cohomology",
        "filter": "swarm|quorum|unified_field|bayesian|liminal|toroidal_kmeans|coarse_to_fine|top2_liminal|merge_and_prune|inv_temp|slerp|coherence|pspm|visual|pointer|tunneling|cohomology|chebyshev|hodge|qdpt|gop_budget",
        "skip": "Provisional_GCSO_Container",
    },
    "long-dialogue": {
        "description": "Focuses on long-context endurance, S^1 geodesic phase unwrap, dwell-step anti-chattering, Bloom self-normalization, and multi-turn stream simulations",
        "filter": "long_context|circular|stream|long_dialogue|500k|1m|50k|100k|chattering|dwell|seam_crossing",
        "skip": "Provisional_GCSO_Container",
    },
    "extreme": {
        "description": "Focuses on extreme stress cases, bit saturation, subnormal rescue, RIPA geodesic clamps, and mathematical singularities",
        "filter": "extreme|edge|saturation|underflow|subnormal|singularity|odd_remainder|spike|prefill",
        "skip": "Provisional_GCSO_Container",
    },
    "long-horizon-edge": {
        "description": "Combined focus on long-horizon degradation (100K-1M tokens, Bloom self-normalization, seam crossing) and edge-case bottlenecks (subnormals, Q7 saturation, odd remainders)",
        "filter": "long_context|circular|stream|long_dialogue|500k|1m|50k|100k|chattering|dwell|seam_crossing|extreme|edge|saturation|underflow|subnormal|singularity|odd_remainder|spike|prefill",
        "skip": "Provisional_GCSO_Container",
    },
    "alignment": {
        "description": "Focuses on 32B/128B/256B/512B hardware memory alignment boundary comparisons",
        "filter": "32b|128b|256b|512b",
        "skip": "Provisional_GCSO_Container",
    },
    "all": {
        "description": "Full exhaustive execution across all Core (PhaseSteering + EntropyControl) and Provisional (.gcso Storage) benchmark groups",
        "filter": None,
        "skip": None,
    },
}


class _WinProcessPowerThrottlingState(ctypes.Structure):
    """Win32 PROCESS_POWER_THROTTLING_STATE structure for disabling EcoQoS throttling."""

    _fields_ = [
        ("Version", ctypes.c_ulong),
        ("ControlMask", ctypes.c_ulong),
        ("StateMask", ctypes.c_ulong),
    ]


def discover_benchmark_targets(project_root: Path) -> list[str]:
    """Scan benches/ or src/core/benches/ directory for Cargo benchmark binaries."""
    candidate_dirs = [
        project_root / "benches",
        project_root / "src" / "core" / "benches",
    ]
    targets: set[str] = set()
    for benches_dir in candidate_dirs:
        if benches_dir.exists():
            for entry in benches_dir.glob("*.rs"):
                targets.add(entry.stem)

    return sorted(targets) if targets else ["bench_phase_steering", "bench_entropy_control"]


def resolve_manifest_path(
    project_root: Path, explicit_manifest: Path | None
) -> Path | None:
    """Auto-resolve Cargo.toml manifest path when running from root or sub-crate layout."""
    if explicit_manifest is not None:
        return explicit_manifest
    root_manifest = project_root / "Cargo.toml"
    if root_manifest.exists():
        return None
    core_manifest = project_root / "src" / "core" / "Cargo.toml"
    if core_manifest.exists():
        return core_manifest
    return None


def resolve_criterion_dir(project_root: Path, requested_dir: Path) -> Path:
    """Resolve the active Criterion output directory across workspace and package layouts."""
    candidate = (
        requested_dir if requested_dir.is_absolute() else (project_root / requested_dir)
    )
    if candidate.exists() and any(candidate.glob("**/new/estimates.json")):
        return requested_dir

    fallbacks = [
        project_root / "target" / "criterion",
        project_root / "src" / "core" / "target" / "criterion",
    ]
    for fb in fallbacks:
        if fb.exists() and any(fb.glob("**/new/estimates.json")):
            return fb
    return requested_dir


LEGACY_PROVISIONAL_BENCH_SUBDIRS: tuple[str, ...] = (
    "gateway_signal_steer_and_crystallize_attractor_64h",
    "gateway_swarm_bayesian_pspm_and_crystallize_full_64h",
    "zimms_zero_copy_crystallized_ensemble_gateway_steer_64h",
    "zero_copy_register_crystallized_ensemble_steer_64h",
    "four_section_gcso_zero_copy_map_and_register_steer_64h",
    "step_gateway_signal_and_crystallize_attractor_64h",
    "step_gateway_4axis_contradiction_guard_and_provenance_64h",
    "step_crystallized_ensemble_gravity_fold_4records",
    "zimms_zero_copy_crystallized_ensemble_step_4records",
    "four_section_gcso_zero_copy_write_verify_and_step_4records",
)


def clean_criterion_artifacts(
    project_root: Path,
    requested_dir: Path,
    json_summary_out: Path | None = None,
    full_clean: bool = True,
    remove_provisional_groups: bool = True,
) -> None:
    """Clean stale Criterion directories before a fresh run (or prune provisional groups only)."""
    candidate_roots = [
        requested_dir
        if requested_dir.is_absolute()
        else (project_root / requested_dir),
        project_root / "target" / "criterion",
        project_root / "src" / "core" / "target" / "criterion",
    ]
    if full_clean:
        seen: set[Path] = set()
        for root in candidate_roots:
            resolved = root.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
        if json_summary_out is not None:
            summary_path = (
                json_summary_out
                if json_summary_out.is_absolute()
                else (project_root / json_summary_out)
            )
            if summary_path.exists():
                summary_path.unlink(missing_ok=True)
        return

    for root in candidate_roots:
        if not root.exists():
            continue
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            if remove_provisional_groups and "provisional_gcso" in entry.name.lower():
                shutil.rmtree(entry, ignore_errors=True)
                continue
            if "provisional_gcso" not in entry.name.lower():
                for legacy_sub in LEGACY_PROVISIONAL_BENCH_SUBDIRS:
                    legacy_path = entry / legacy_sub
                    if legacy_path.exists():
                        shutil.rmtree(legacy_path, ignore_errors=True)


def _discover_linux_primary_physical_cores() -> list[int]:
    """Return sorted primary (non-SMT-sibling) logical CPU IDs on Linux via sysfs topology."""
    cpu_root = Path("/sys/devices/system/cpu")
    if not cpu_root.exists():
        return []
    primaries: set[int] = set()
    for topo_file in cpu_root.glob("cpu[0-9]*/topology/thread_siblings_list"):
        try:
            raw = topo_file.read_text(encoding="utf-8").strip()
            first_token = raw.split(",")[0].split("-")[0].strip()
            primaries.add(int(first_token))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(primaries)


def auto_select_optimal_core() -> int | None:
    """Select a dedicated non-zero primary physical CPU core to avoid Core #0 IRQ/DPC and SMT noise."""
    allowed_cores: set[int] | None = None
    sched_getaffinity = cast(
        Callable[[int], set[int]] | None,
        getattr(os, "sched_getaffinity", None),
    )
    if callable(sched_getaffinity):
        try:
            allowed_cores = set(sched_getaffinity(0))
        except OSError:
            allowed_cores = None

    primary_phys = _discover_linux_primary_physical_cores()
    if primary_phys:
        candidates = [
            c
            for c in primary_phys
            if c > 0 and (allowed_cores is None or c in allowed_cores)
        ]
        if candidates:
            return candidates[1] if len(candidates) >= 2 else candidates[0]

    if allowed_cores:
        sorted_allowed = sorted(allowed_cores)
        non_zero = [c for c in sorted_allowed if c > 0]
        if len(non_zero) >= 2:
            return non_zero[1]
        if non_zero:
            return non_zero[0]
        return None

    cpu_cnt = os.cpu_count() or 1
    if cpu_cnt >= 8:
        return 4
    elif cpu_cnt >= 4:
        return 2
    elif cpu_cnt >= 2:
        return 1
    return None


def _win32_configure_kernel32(kernel32: Any) -> None:
    """Configure explicit 64-bit Win32 HANDLE/argtypes on ctypes.windll.kernel32."""
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentThread.argtypes = []
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
    kernel32.SetThreadPriority.restype = ctypes.c_int
    kernel32.SetProcessPriorityBoost.argtypes = [ctypes.c_void_p, ctypes.c_int]
    kernel32.SetProcessPriorityBoost.restype = ctypes.c_int
    kernel32.SetProcessInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    kernel32.SetProcessInformation.restype = ctypes.c_int
    kernel32.SetThreadIdealProcessor.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.SetThreadIdealProcessor.restype = ctypes.c_ulong


def apply_process_stabilization(high_priority: bool, pin_core: int | None) -> None:
    """Elevate process scheduling priority, disable EcoQoS throttling, and pin CPU core affinity."""
    if pin_core is not None:
        if pin_core < 0:
            print(
                f"Warning: Ignoring negative CPU core index #{pin_core}.",
                file=sys.stderr,
            )
        else:
            try:
                sched_setaffinity = cast(
                    Callable[[int, set[int]], None] | None,
                    getattr(os, "sched_getaffinity", None)
                    and getattr(os, "sched_setaffinity", None),
                )
                if callable(sched_setaffinity):
                    sched_setaffinity(0, {int(pin_core)})
                    print(
                        f"CPU Affinity    : Pinned to physical core #{pin_core} (sched_setaffinity, avoids Core #0 IRQ & SMT jitter)"
                    )
                elif platform.system() == "Windows":
                    windll = getattr(ctypes, "windll", None)
                    if windll is not None:
                        kernel32 = windll.kernel32
                        _win32_configure_kernel32(kernel32)
                        handle = kernel32.GetCurrentProcess()
                        mask = ctypes.c_size_t(1 << int(pin_core))
                        if kernel32.SetProcessAffinityMask(handle, mask):
                            print(
                                f"CPU Affinity    : Pinned to core #{pin_core} (Win32 AffinityMask, avoids Core #0 DPC jitter)"
                            )
            except (OSError, ValueError, AttributeError, OverflowError) as e:
                print(
                    f"Warning: Could not pin CPU core #{pin_core}: {e}",
                    file=sys.stderr,
                )

    if high_priority:
        try:
            if platform.system() == "Windows":
                windll = getattr(ctypes, "windll", None)
                if windll is not None:
                    kernel32 = windll.kernel32
                    _win32_configure_kernel32(kernel32)
                    handle = kernel32.GetCurrentProcess()
                    if kernel32.SetPriorityClass(handle, 0x00000080):
                        print(
                            "Process Priority: Elevated to HIGH_PRIORITY_CLASS (Win32 optimal default)"
                        )
                    try:
                        thread_handle = kernel32.GetCurrentThread()
                        kernel32.SetThreadPriority(thread_handle, 2)
                    except (AttributeError, OSError):
                        pass
                    try:
                        kernel32.SetProcessPriorityBoost(handle, 0)
                    except (AttributeError, OSError):
                        pass
                    try:
                        throttling = _WinProcessPowerThrottlingState(
                            Version=1,
                            ControlMask=0x5,
                            StateMask=0x0,
                        )
                        if kernel32.SetProcessInformation(
                            handle,
                            4,
                            ctypes.byref(throttling),
                            ctypes.sizeof(throttling),
                        ):
                            print(
                                "Power QoS       : Disabled Windows EcoQoS & Background Timer Throttling (High-QoS P-Core locked)"
                            )
                    except (AttributeError, OSError):
                        pass
                    try:
                        if pin_core is not None and pin_core >= 0:
                            thread_handle = kernel32.GetCurrentThread()
                            kernel32.SetThreadIdealProcessor(
                                thread_handle, ctypes.c_ulong(int(pin_core))
                            )
                    except (AttributeError, OSError):
                        pass
            else:
                os_nice = cast(
                    Callable[[int], int] | None,
                    getattr(os, "nice", None),
                )
                if callable(os_nice):
                    os_nice(-10)
                    print(
                        "Process Priority: Elevated via os.nice(-10) (POSIX optimal default)"
                    )
        except (OSError, ValueError, AttributeError) as e:
            print(
                f"Warning: Could not elevate process priority (continuing with normal priority): {e}",
                file=sys.stderr,
            )


def prepare_host_for_measurement(settle_ms: int = 150) -> None:
    """Flush OS filesystem caches, run Python GC, and pause briefly after compilation before timing."""
    gc.collect()
    os_sync = cast(Callable[[], None] | None, getattr(os, "sync", None))
    if callable(os_sync):
        try:
            os_sync()
        except OSError:
            pass
    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)


def build_optimal_cargo_env(
    native_cpu: bool,
    extra_rustflags: str | None,
    run_provisional_gcso: bool = False,
    noise_threshold: float | None = 0.025,
) -> dict[str, str]:
    """Construct a deterministic, single-thread-pool Cargo/Criterion environment dictionary."""
    custom_env: dict[str, str] = {
        "CARGO_INCREMENTAL": "0",
        "CARGO_TERM_COLOR": "always",
        "RAYON_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKIO_WORKER_THREADS": "1",
        "RUST_BACKTRACE": "0",
        "LIMINIKA_RUN_PROVISIONAL_GCSO": "1" if run_provisional_gcso else "0",
    }
    if noise_threshold is not None and noise_threshold > 0.0:
        custom_env["LIMINIKA_NOISE_THRESHOLD"] = str(noise_threshold)

    rustflags_list: list[str] = []
    if os.environ.get("RUSTFLAGS"):
        rustflags_list.append(os.environ["RUSTFLAGS"])
    joined_existing = " ".join(rustflags_list)

    if native_cpu:
        if "opt-level" not in joined_existing:
            rustflags_list.append("-C opt-level=3")
        if "target-cpu" not in joined_existing:
            rustflags_list.append("-C target-cpu=native")
        if "codegen-units" not in joined_existing:
            rustflags_list.append("-C codegen-units=1")

    if extra_rustflags:
        rustflags_list.append(extra_rustflags)

    if rustflags_list:
        custom_env["RUSTFLAGS"] = " ".join(rustflags_list)

    return custom_env


def run_command(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
    phase_label: str | None = None,
) -> None:
    """Execute shell command with stream pass-through and strict status validation."""
    if phase_label:
        print(f"\nExecuting [{phase_label}]: {' '.join(cmd)}")
    else:
        print(f"\nExecuting: {' '.join(cmd)}")
    if env:
        if "RUSTFLAGS" in env:
            print(f"Environment RUSTFLAGS        : {env['RUSTFLAGS']}")
        if "CARGO_INCREMENTAL" in env:
            print(f"Environment CARGO_INCREMENTAL: {env['CARGO_INCREMENTAL']}")
    sys.stdout.flush()

    if dry_run:
        return

    current_env = os.environ.copy()
    if env:
        current_env.update(env)

    creationflags = 0x00000080 if sys.platform == "win32" else 0
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=current_env,
        creationflags=creationflags,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"\nError: Command failed with exit code {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def detect_cpu_governor() -> str | None:
    """Read Linux CPU frequency scaling governor if available."""
    gov_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        if gov_path.exists():
            return gov_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return None


def print_system_diagnostics() -> None:
    """Print hardware and execution environment information."""
    print("=== System Environment Diagnostics ===")
    print(
        f"OS Platform     : {platform.system()} {platform.release()} ({platform.machine()})"
    )
    print(f"CPU Cores       : {os.cpu_count() or 'Unknown'}")
    governor = detect_cpu_governor()
    if governor:
        print(f"CPU Governor    : {governor}")
    print(f"Python Runtime  : {platform.python_version()}")
    try:
        rustc_ver = subprocess.check_output(
            ["rustc", "--version"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        print(f"Rust Toolchain  : {rustc_ver}")
    except (OSError, subprocess.SubprocessError):
        print("Rust Toolchain  : Unable to detect rustc")

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        print(f"Git Commit Hash : {git_hash}")
    except (OSError, subprocess.SubprocessError):
        print("Git Commit Hash : N/A")
    print("======================================\n")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    discovered_benches = discover_benchmark_targets(project_root)
    default_core = auto_select_optimal_core()

    parser = argparse.ArgumentParser(
        description="Unified LiminiKa Benchmark & Report Pipeline Runner (Core PhaseSteering/EntropyControl Default + Isolated Provisional .gcso)"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=discovered_benches,
        help=f"List of benchmark target binaries to execute (discovered: {', '.join(discovered_benches)})",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="standard",
        help="Select pre-configured benchmarking profile (default: standard = Core PhaseSteering + EntropyControl only)",
    )
    parser.add_argument(
        "--include-provisional-gcso",
        action="store_true",
        default=False,
        help="Also run provisional .gcso storage container benchmarks alongside the selected profile",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Custom Criterion filter regex (overrides profile filter if specified)",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional explicit path to Cargo.toml (auto-detected if omitted)",
    )
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        default=Path("target/criterion"),
        help="Path to Criterion target output directory (default: target/criterion)",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/benchmark/benchmarks.md"),
        help="Path to output Markdown report file (default: docs/benchmark/benchmarks.md)",
    )
    parser.add_argument(
        "--json-summary-out",
        type=Path,
        default=Path("target/criterion_summary.json"),
        help="Path to output JSON summary for CI pipelines (default: target/criterion_summary.json)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Compare execution against specified Criterion baseline name",
    )
    parser.add_argument(
        "--save-baseline",
        type=str,
        default=None,
        help="Save measurement results under specified Criterion baseline name",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional Criterion --sample-size override (default: None so per-group Rust sample_size(30) on 1M-token streams is preserved)",
    )
    parser.add_argument(
        "--warm-up-time",
        type=float,
        default=None,
        help="Optional Criterion --warm-up-time in seconds (default: None to respect per-group Rust warm_up_time)",
    )
    parser.add_argument(
        "--measurement-time",
        type=float,
        default=None,
        help="Optional Criterion --measurement-time in seconds (default: None to respect per-group Rust measurement_time)",
    )
    parser.add_argument(
        "--noise-threshold",
        type=float,
        default=0.025,
        help="Criterion --noise-threshold relative tolerance (default: 0.025 = 2.5%% to suppress sub-30ns RDTSC & background OS scheduler false alarms)",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=400,
        help="Thermal & disk I/O settle pause in milliseconds after pre-compilation and between benchmark binaries (default: 400)",
    )
    parser.add_argument(
        "--noplot",
        action="store_true",
        default=False,
        help="Pass --noplot to Criterion for headless CI (note: generate_bench_report.py still synthesizes target/criterion/report/index.html)",
    )
    parser.add_argument(
        "--plot",
        action="store_false",
        dest="noplot",
        help="Enable Criterion HTML report & SVG plot generation (default: enabled)",
    )
    parser.add_argument(
        "--open-index",
        action="store_true",
        default=False,
        help="Automatically open target/criterion/report/index.html in the default web browser after completion",
    )
    parser.add_argument(
        "--precompile",
        action="store_true",
        default=True,
        help="Pre-compile benchmark binaries via 'cargo bench --no-run' and settle I/O before timing (default: enabled)",
    )
    parser.add_argument(
        "--no-precompile",
        action="store_false",
        dest="precompile",
        help="Skip the two-stage 'cargo bench --no-run' pre-compilation step",
    )
    parser.add_argument(
        "--native-cpu",
        action="store_true",
        default=True,
        help="Append '-C opt-level=3 -C target-cpu=native -C codegen-units=1' to RUSTFLAGS (default: enabled)",
    )
    parser.add_argument(
        "--no-native-cpu",
        action="store_false",
        dest="native_cpu",
        help="Disable default '-C opt-level=3 -C target-cpu=native -C codegen-units=1' flags for portable codegen testing",
    )
    parser.add_argument(
        "--high-priority",
        action="store_true",
        default=True,
        help="Elevate process scheduling priority and disable EcoQoS throttling to reduce OS preemption jitter (default: enabled)",
    )
    parser.add_argument(
        "--no-high-priority",
        action="store_false",
        dest="high_priority",
        help="Disable automatic process scheduling priority elevation",
    )
    parser.add_argument(
        "--pin-core",
        type=int,
        default=default_core,
        help=f"Pin benchmark runner to a specific CPU core index (default auto-selected: {default_core})",
    )
    parser.add_argument(
        "--no-pin-core",
        action="store_true",
        help="Disable automatic CPU core affinity pinning",
    )
    parser.add_argument(
        "--rustflags",
        type=str,
        default=None,
        help="Custom RUSTFLAGS to append to cargo bench",
    )
    parser.add_argument(
        "--fail-on-high-noise",
        action="store_true",
        help="Fail pipeline if genuine high measurement variance/noise is detected",
    )
    parser.add_argument(
        "--fail-on-regression-pct",
        type=float,
        default=None,
        help="Fail pipeline if any benchmark regresses beyond this percentage vs --baseline",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate summary tables only in Markdown report",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Remove existing Criterion artifacts before measuring so every run starts completely fresh (default: enabled, automatically disabled when --baseline is set)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_false",
        dest="clean",
        help="Keep existing Criterion artifacts for incremental updates or baseline comparisons",
    )
    parser.add_argument(
        "--skip-execution",
        action="store_true",
        help="Skip cargo bench execution and re-generate report from existing target/criterion artifacts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved commands without executing them",
    )

    args = parser.parse_args()
    effective_pin_core = None if args.no_pin_core else args.pin_core

    print_system_diagnostics()

    raw_profile_filter = args.filter
    effective_skip: str | None = None
    if not raw_profile_filter and args.profile in PROFILES:
        raw_profile_filter = PROFILES[args.profile]["filter"]
        if not args.include_provisional_gcso:
            effective_skip = PROFILES[args.profile].get("skip")

    run_provisional_gcso = bool(
        args.include_provisional_gcso
        or args.profile in ("provisional-gcso", "all")
        or (
            raw_profile_filter is not None
            and "provisional" in raw_profile_filter.lower()
        )
    )

    # Criterion's `configure_from_args()` accepts a single positive positional regex `[FILTER]`
    # and does not support `libtest`'s `--skip` flag. Provisional `.gcso` groups are gated
    # in-process via `LIMINIKA_RUN_PROVISIONAL_GCSO=0`, so we do not restrict group names to a
    # hardcoded prefix allowlist—allowing newly added benchmark files/groups to run automatically.
    effective_filter: str | None = raw_profile_filter

    custom_env = build_optimal_cargo_env(
        args.native_cpu,
        args.rustflags,
        run_provisional_gcso=run_provisional_gcso,
        noise_threshold=args.noise_threshold,
    )
    resolved_manifest = resolve_manifest_path(project_root, args.manifest_path)

    if not args.skip_execution:
        if not args.dry_run:
            do_full_clean = bool(args.clean and not args.baseline)
            clean_criterion_artifacts(
                project_root,
                args.criterion_dir,
                json_summary_out=args.json_summary_out,
                full_clean=do_full_clean,
                remove_provisional_groups=not run_provisional_gcso,
            )

        base_cargo_cmd = ["cargo", "bench"]
        if resolved_manifest is not None:
            base_cargo_cmd.extend(["--manifest-path", str(resolved_manifest)])
        for bench_target in args.benchmarks:
            base_cargo_cmd.extend(["--bench", bench_target])

        if args.precompile:
            print(
                "=== Step 0: Pre-Compiling Benchmark Binaries Across All Cores (Separating Build I/O & Thermal Spike from Timing) ==="
            )
            precompile_cmd = [*base_cargo_cmd, "--no-run"]
            run_command(
                precompile_cmd,
                cwd=project_root,
                env=custom_env,
                dry_run=args.dry_run,
                phase_label="Step 0/2: Pre-Compile Only (--no-run; actual benchmarks execute in Step 1 after single-core pinning)",
            )
            if not args.dry_run:
                print(
                    "Pre-compilation complete. Flushing OS caches and settling CPU thermals before Step 1 measurement..."
                )
                prepare_host_for_measurement(
                    settle_ms=max(args.settle_ms, 800) if args.settle_ms > 0 else 0
                )

        if not args.dry_run:
            apply_process_stabilization(args.high_priority, effective_pin_core)

        print(
            f"\n=== Step 1: Running GCSO Core Micro-Benchmarks (Profile: {args.profile}) ==="
        )
        print(f"Description: {PROFILES[args.profile]['description']}")
        print(f"Targets    : {', '.join(args.benchmarks)}")
        if effective_filter:
            print(f"Filter     : '{effective_filter}'")
        if effective_skip:
            print(
                f"Skipped    : '{effective_skip}' (Provisional .gcso isolated via LIMINIKA_RUN_PROVISIONAL_GCSO=0)"
            )

        cargo_args: list[str] = []
        if effective_filter:
            cargo_args.append(effective_filter)
        if args.baseline:
            cargo_args.extend(["--baseline", args.baseline])
        if args.save_baseline:
            cargo_args.extend(["--save-baseline", args.save_baseline])
        if args.sample_size is not None:
            cargo_args.extend(["--sample-size", str(args.sample_size)])
        if args.warm_up_time is not None:
            cargo_args.extend(["--warm-up-time", str(args.warm_up_time)])
        if args.measurement_time is not None:
            cargo_args.extend(["--measurement-time", str(args.measurement_time)])
        if args.noise_threshold is not None and args.noise_threshold > 0.0:
            cargo_args.extend(["--noise-threshold", str(args.noise_threshold)])
        if args.noplot:
            cargo_args.append("--noplot")

        for idx, bench_target in enumerate(args.benchmarks):
            if idx > 0 and not args.dry_run:
                prepare_host_for_measurement(settle_ms=max(0, args.settle_ms))

            target_cmd = ["cargo", "bench"]
            if resolved_manifest is not None:
                target_cmd.extend(["--manifest-path", str(resolved_manifest)])
            target_cmd.extend(["--bench", bench_target])
            if cargo_args:
                target_cmd.append("--")
                target_cmd.extend(cargo_args)

            run_command(
                target_cmd,
                cwd=project_root,
                env=custom_env,
                dry_run=args.dry_run,
                phase_label=f"Step 1/2: Actual Benchmark Measurement ({idx + 1}/{len(args.benchmarks)}: {bench_target})",
            )

    report_script = project_root / "scripts" / "generate_bench_report.py"
    if not report_script.exists() and not args.dry_run:
        print(
            f"Error: Report generator script missing at {report_script}",
            file=sys.stderr,
        )
        sys.exit(1)

    resolved_criterion_dir = (
        args.criterion_dir
        if args.dry_run
        else resolve_criterion_dir(project_root, args.criterion_dir)
    )

    print(
        "\n=== Step 2: Generating Comprehensive Performance, Bottleneck, HTML Index & Long-Context Report ==="
    )
    report_cmd = [
        sys.executable,
        str(report_script),
        "--criterion-dir",
        str(resolved_criterion_dir),
        "--output",
        str(args.output_md),
        "--json-summary-out",
        str(args.json_summary_out),
    ]

    if args.baseline:
        report_cmd.extend(["--baseline", args.baseline])

    if args.fail_on_regression_pct is not None:
        report_cmd.extend(
            ["--fail-on-regression-pct", str(args.fail_on_regression_pct)]
        )

    if args.fail_on_high_noise:
        report_cmd.append("--fail-on-high-noise")

    if args.summary_only:
        report_cmd.append("--summary-only")

    run_command(
        report_cmd,
        cwd=project_root,
        dry_run=args.dry_run,
        phase_label="Step 2/2: Report & HTML Index Synthesis",
    )

    abs_crit_dir = (
        resolved_criterion_dir
        if resolved_criterion_dir.is_absolute()
        else (project_root / resolved_criterion_dir)
    )
    index_html_path = abs_crit_dir / "report" / "index.html"
    if not args.dry_run and index_html_path.exists():
        resolved_html = index_html_path.resolve()
        print(f"🌐 Criterion HTML Index : {resolved_html}")
        print(f"🔗 Browser URI          : {resolved_html.as_uri()}")
        if args.open_index:
            webbrowser.open(resolved_html.as_uri())

    print("\n=== LiminiKa Benchmark Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
