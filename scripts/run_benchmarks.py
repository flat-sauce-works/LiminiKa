# name: scripts/run_benchmarks.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""LiminiKa unified benchmark runner and Criterion report coordinator."""

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

PROFILES: dict[str, dict[str, str | None]] = {
    "standard": {
        "description": "Core PhaseSteering and EntropyControl groups (excludes provisional .lmnk)",
        "filter": None,
        "skip": "Provisional_LMNK_Container",
    },
    "quick": {
        "description": "Fast smoke subset across PhaseSteering and EntropyControl core kernels",
        "filter": (
            "apply_phase_steering_32b|apply_phase_steering_safe_clamped|"
            "eval_stateful_normal_mode|eval_lacuna_energy_dyadic128|"
            "subhead_8slot_pseudo_moe_ensemble|"
            "online_toroidal_centroid_merge_and_prune_8centroids_64h|"
            "evaluate_adaptive_compute_and_gop_budget"
        ),
        "skip": "Provisional_LMNK_Container",
    },
    "extreme": {
        "description": "Extreme stress, boundary saturation, subnormal rescue, and singularity cases",
        "filter": (
            "Extreme_And_Edge_Cases|extreme_head_dim|q7_boundary_|extreme_phase_spike_clamp_branchless|"
            "slerp_norm_guard_|subnormal|1m|chattering_|spike_recovery|"
            "odd_remainder|pitchfork|singularity"
        ),
        "skip": "Provisional_LMNK_Container",
    },
    "all": {
        "description": "All Core and Provisional .lmnk benchmark groups",
        "filter": None,
        "skip": None,
    },
}


class _WinProcessPowerThrottlingState(ctypes.Structure):
    """Win32 PROCESS_POWER_THROTTLING_STATE for disabling EcoQoS throttling."""

    _fields_ = [
        ("Version", ctypes.c_ulong),
        ("ControlMask", ctypes.c_ulong),
        ("StateMask", ctypes.c_ulong),
    ]


def discover_benchmark_targets(project_root: Path) -> list[str]:
    """Scan `benches/` or `src/core/benches/` for Cargo benchmark targets."""
    targets: set[str] = set()
    for benches_dir in (project_root / "benches", project_root / "src" / "core" / "benches"):
        if benches_dir.exists():
            for entry in benches_dir.glob("*.rs"):
                if entry.stem != "common":
                    targets.add(entry.stem)
    return sorted(targets) if targets else ["bench_entropy_control", "bench_phase_steering"]


def resolve_manifest_path(project_root: Path, explicit_manifest: Path | None) -> Path | None:
    """Auto-resolve `Cargo.toml` path when running from root or sub-crate layout."""
    if explicit_manifest is not None:
        return explicit_manifest
    if (project_root / "Cargo.toml").exists():
        return None
    core_manifest = project_root / "src" / "core" / "Cargo.toml"
    return core_manifest if core_manifest.exists() else None


def resolve_criterion_dir(project_root: Path, requested_dir: Path) -> Path:
    """Resolve the active Criterion output directory across workspace and package layouts."""
    candidate = requested_dir if requested_dir.is_absolute() else (project_root / requested_dir)
    if candidate.exists() and any(candidate.glob("**/new/estimates.json")):
        return requested_dir
    for fb in (project_root / "target" / "criterion", project_root / "src" / "core" / "target" / "criterion"):
        if fb.exists() and any(fb.glob("**/new/estimates.json")):
            return fb
    return requested_dir


def clean_criterion_artifacts(
    project_root: Path,
    requested_dir: Path,
    json_summary_out: Path | None = None,
    full_clean: bool = True,
    remove_provisional_groups: bool = True,
) -> None:
    """Clean Criterion artifacts before a fresh run, or prune provisional groups only."""
    roots = [
        requested_dir if requested_dir.is_absolute() else (project_root / requested_dir),
        project_root / "target" / "criterion",
        project_root / "src" / "core" / "target" / "criterion",
    ]
    if full_clean:
        seen: set[Path] = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                if root.exists():
                    shutil.rmtree(root, ignore_errors=True)
        if json_summary_out is not None:
            summary = json_summary_out if json_summary_out.is_absolute() else (project_root / json_summary_out)
            summary.unlink(missing_ok=True)
        return
    if remove_provisional_groups:
        for root in roots:
            if root.exists():
                for entry in root.iterdir():
                    if entry.is_dir() and "provisional_lmnk" in entry.name.lower():
                        shutil.rmtree(entry, ignore_errors=True)


def _discover_linux_primary_physical_cores() -> list[int]:
    cpu_root = Path("/sys/devices/system/cpu")
    if not cpu_root.exists():
        return []
    primaries: set[int] = set()
    for topo in cpu_root.glob("cpu[0-9]*/topology/thread_siblings_list"):
        try:
            first_token = topo.read_text(encoding="utf-8").strip().split(",")[0].split("-")[0].strip()
            primaries.add(int(first_token))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(primaries)


def auto_select_optimal_core() -> int | None:
    """Select a dedicated non-zero primary physical CPU core to reduce IRQ/SMT noise."""
    allowed_cores: set[int] | None = None
    sched_getaffinity = cast(Callable[[int], set[int]] | None, getattr(os, "sched_getaffinity", None))
    if callable(sched_getaffinity):
        try:
            allowed_cores = set(sched_getaffinity(0))
        except OSError:
            allowed_cores = None

    primary_phys = _discover_linux_primary_physical_cores()
    if primary_phys:
        candidates = [c for c in primary_phys if c > 0 and (allowed_cores is None or c in allowed_cores)]
        if candidates:
            return candidates[1] if len(candidates) >= 2 else candidates[0]
    if allowed_cores:
        non_zero = [c for c in sorted(allowed_cores) if c > 0]
        return non_zero[1] if len(non_zero) >= 2 else (non_zero[0] if non_zero else None)

    cpu_cnt = os.cpu_count() or 1
    if cpu_cnt >= 8:
        return 4
    if cpu_cnt >= 4:
        return 2
    return 1 if cpu_cnt >= 2 else None


def _win32_configure_kernel32(kernel32: Any) -> None:
    kernel32.GetCurrentProcess.argtypes, kernel32.GetCurrentProcess.restype = [], ctypes.c_void_p
    kernel32.GetCurrentThread.argtypes, kernel32.GetCurrentThread.restype = [], ctypes.c_void_p
    kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
    kernel32.SetThreadPriority.restype = ctypes.c_int
    kernel32.SetProcessPriorityBoost.argtypes = [ctypes.c_void_p, ctypes.c_int]
    kernel32.SetProcessPriorityBoost.restype = ctypes.c_int
    kernel32.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
    kernel32.SetProcessInformation.restype = ctypes.c_int
    kernel32.SetThreadIdealProcessor.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel32.SetThreadIdealProcessor.restype = ctypes.c_ulong


def apply_process_stabilization(high_priority: bool, pin_core: int | None) -> None:
    """Elevate scheduling priority, disable EcoQoS throttling, and pin CPU core affinity."""
    if pin_core is not None:
        if pin_core < 0:
            print(f"Warning: Ignoring negative CPU core index #{pin_core}.", file=sys.stderr)
        else:
            try:
                sched_setaffinity = cast(
                    Callable[[int, set[int]], None] | None,
                    getattr(os, "sched_getaffinity", None) and getattr(os, "sched_setaffinity", None),
                )
                if callable(sched_setaffinity):
                    sched_setaffinity(0, {int(pin_core)})
                    print(f"CPU Affinity    : Pinned to core #{pin_core} (sched_setaffinity)")
                elif platform.system() == "Windows" and getattr(ctypes, "windll", None) is not None:
                    k32 = ctypes.windll.kernel32
                    _win32_configure_kernel32(k32)
                    if k32.SetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.c_size_t(1 << int(pin_core))):
                        print(f"CPU Affinity    : Pinned to core #{pin_core} (Win32 AffinityMask)")
            except (OSError, ValueError, AttributeError, OverflowError) as exc:
                print(f"Warning: Could not pin CPU core #{pin_core}: {exc}", file=sys.stderr)

    if high_priority:
        try:
            if platform.system() == "Windows" and getattr(ctypes, "windll", None) is not None:
                k32 = ctypes.windll.kernel32
                _win32_configure_kernel32(k32)
                proc, thr = k32.GetCurrentProcess(), k32.GetCurrentThread()
                if k32.SetPriorityClass(proc, 0x00000080):
                    print("Process Priority: Elevated to HIGH_PRIORITY_CLASS")
                for fn in (lambda: k32.SetThreadPriority(thr, 2), lambda: k32.SetProcessPriorityBoost(proc, 0)):
                    try:
                        fn()
                    except (AttributeError, OSError):
                        pass
                try:
                    throttling = _WinProcessPowerThrottlingState(Version=1, ControlMask=0x5, StateMask=0x0)
                    if k32.SetProcessInformation(proc, 4, ctypes.byref(throttling), ctypes.sizeof(throttling)):
                        print("Power QoS       : Disabled Windows EcoQoS throttling")
                except (AttributeError, OSError):
                    pass
                if pin_core is not None and pin_core >= 0:
                    try:
                        k32.SetThreadIdealProcessor(thr, ctypes.c_ulong(int(pin_core)))
                    except (AttributeError, OSError):
                        pass
            else:
                os_nice = cast(Callable[[int], int] | None, getattr(os, "nice", None))
                if callable(os_nice):
                    os_nice(-10)
                    print("Process Priority: Elevated via os.nice(-10)")
        except (OSError, ValueError, AttributeError) as exc:
            print(f"Warning: Could not elevate process priority: {exc}", file=sys.stderr)


def prepare_host_for_measurement(settle_ms: int = 150) -> None:
    """Flush OS caches, run Python GC, and pause briefly before timing."""
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
    run_provisional_lmnk: bool = False,
    noise_threshold: float | None = 0.025,
) -> dict[str, str]:
    """Construct a deterministic single-thread-pool Cargo/Criterion environment dictionary."""
    env: dict[str, str] = {
        "CARGO_INCREMENTAL": "0",
        "CARGO_TERM_COLOR": "always",
        "RAYON_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "TOKIO_WORKER_THREADS": "1",
        "RUST_BACKTRACE": "0",
        "LIMINIKA_RUN_PROVISIONAL_LMNK": "1" if run_provisional_lmnk else "0",
    }
    if noise_threshold is not None and noise_threshold > 0.0:
        env["LIMINIKA_NOISE_THRESHOLD"] = str(noise_threshold)

    flags: list[str] = [os.environ["RUSTFLAGS"]] if os.environ.get("RUSTFLAGS") else []
    joined = " ".join(flags)
    if native_cpu:
        for flag, key in (("-C opt-level=3", "opt-level"), ("-C target-cpu=native", "target-cpu"),
                          ("-C codegen-units=1", "codegen-units")):
            if key not in joined:
                flags.append(flag)
    if extra_rustflags:
        flags.append(extra_rustflags)
    if flags:
        env["RUSTFLAGS"] = " ".join(flags)
    return env


def run_command(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None,
    dry_run: bool = False, phase_label: str | None = None,
) -> None:
    """Execute command with inherited stdio and strict exit validation."""
    prefix = f"[{phase_label}] " if phase_label else ""
    print(f"\nExecuting {prefix}: {' '.join(cmd)}")
    if env and "RUSTFLAGS" in env:
        print(f"RUSTFLAGS: {env['RUSTFLAGS']}")
    sys.stdout.flush()
    if dry_run:
        return
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    flags = 0x00000080 if sys.platform == "win32" else 0
    res = subprocess.run(cmd, cwd=cwd, env=merged_env, creationflags=flags, check=False)
    if res.returncode != 0:
        print(f"\nError: Command failed with exit code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)


def print_system_diagnostics() -> None:
    """Print hardware and toolchain diagnostics."""
    print("=== System Environment Diagnostics ===")
    print(f"OS Platform     : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"CPU Cores       : {os.cpu_count() or 'Unknown'}")
    gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov.exists():
        try:
            print(f"CPU Governor    : {gov.read_text(encoding='utf-8').strip()}")
        except OSError:
            pass
    print(f"Python Runtime  : {platform.python_version()}")
    for label, cmd in (
        ("Rust Toolchain", ["rustc", "--version"]),
        ("Git Commit Hash", ["git", "rev-parse", "--short", "HEAD"]),
    ):
        try:
            val = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            val = "N/A"
        print(f"{label:<16}: {val}")
    print("======================================\n")


def parse_args(discovered_benches: list[str], default_core: int | None) -> argparse.Namespace:
    """Parse benchmark runner CLI flags."""
    p = argparse.ArgumentParser(description="Unified LiminiKa benchmark runner and report pipeline.")
    p.add_argument("--benchmarks", nargs="+", default=discovered_benches,
                   help=f"Benchmark targets (default: {', '.join(discovered_benches)})")
    p.add_argument("--profile", choices=list(PROFILES.keys()), default="standard",
                   help="Pre-configured benchmark profile (default: standard)")
    p.add_argument("--include-provisional-lmnk", action="store_true", default=False,
                   help="Also execute provisional .lmnk storage benchmarks alongside the profile")
    p.add_argument("--filter", type=str, default=None, help="Custom Criterion filter regex")
    p.add_argument("--manifest-path", type=Path, default=None, help="Explicit Cargo.toml path")
    p.add_argument("--criterion-dir", type=Path, default=Path("target/criterion"), help="Criterion output dir")
    p.add_argument("--output-md", type=Path, default=Path("docs/benchmark/benchmarks.md"), help="Output Markdown path")
    p.add_argument("--json-summary-out", type=Path, default=Path("target/criterion_summary.json"),
                   help="Output JSON summary path")
    p.add_argument("--baseline", type=str, default=None, help="Compare against Criterion baseline")
    p.add_argument("--save-baseline", type=str, default=None, help="Save results as Criterion baseline")
    p.add_argument("--sample-size", type=int, default=None, help="Override Criterion --sample-size")
    p.add_argument("--warm-up-time", type=float, default=None, help="Override Criterion --warm-up-time (s)")
    p.add_argument("--measurement-time", type=float, default=None, help="Override Criterion --measurement-time (s)")
    p.add_argument("--noise-threshold", type=float, default=0.025, help="Criterion --noise-threshold (default: 0.025)")
    p.add_argument("--settle-ms", type=int, default=400, help="Settle pause in ms (default: 400)")
    p.add_argument("--noplot", action="store_true", default=False, help="Pass --noplot to Criterion")
    p.add_argument("--plot", action="store_false", dest="noplot", help="Enable Criterion plots (default)")
    p.add_argument("--open-index", action="store_true", default=False, help="Open HTML index after completion")
    p.add_argument("--precompile", action="store_true", default=True, help="Pre-compile via --no-run (default)")
    p.add_argument("--no-precompile", action="store_false", dest="precompile", help="Skip pre-compilation")
    p.add_argument("--native-cpu", action="store_true", default=True, help="Enable target-cpu=native RUSTFLAGS")
    p.add_argument("--no-native-cpu", action="store_false", dest="native_cpu", help="Disable target-cpu=native")
    p.add_argument("--high-priority", action="store_true", default=True, help="Elevate priority and disable EcoQoS")
    p.add_argument("--no-high-priority", action="store_false", dest="high_priority", help="Keep default priority")
    p.add_argument("--pin-core", type=int, default=default_core, help=f"Pin CPU core (default: {default_core})")
    p.add_argument("--no-pin-core", action="store_true", help="Disable CPU core affinity pinning")
    p.add_argument("--rustflags", type=str, default=None, help="Extra RUSTFLAGS to append")
    p.add_argument("--fail-on-high-noise", action="store_true", help="Fail if high variance is detected")
    p.add_argument("--fail-on-regression-pct", type=float, default=None, help="Fail above regression percent")
    p.add_argument("--summary-only", action="store_true", help="Generate summary only in Markdown report")
    p.add_argument("--clean", action="store_true", default=True, help="Clean Criterion dir before running (default)")
    p.add_argument("--no-clean", action="store_false", dest="clean", help="Keep existing Criterion artifacts")
    p.add_argument("--skip-execution", action="store_true", help="Skip cargo bench and regenerate reports")
    p.add_argument("--dry-run", action="store_true", help="Print resolved commands without executing")
    return p.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    args = parse_args(discover_benchmark_targets(project_root), auto_select_optimal_core())
    pin_core = None if args.no_pin_core else args.pin_core
    print_system_diagnostics()

    raw_filter = args.filter or PROFILES[args.profile]["filter"]
    effective_skip = None if (args.filter or args.include_provisional_lmnk) else PROFILES[args.profile].get("skip")
    run_provisional = bool(
        args.include_provisional_lmnk
        or args.profile == "all"
        or (raw_filter is not None and "provisional" in raw_filter.lower())
    )
    custom_env = build_optimal_cargo_env(args.native_cpu, args.rustflags, run_provisional, args.noise_threshold)
    manifest = resolve_manifest_path(project_root, args.manifest_path)

    if not args.skip_execution:
        if not args.dry_run:
            clean_criterion_artifacts(
                project_root, args.criterion_dir, args.json_summary_out,
                full_clean=bool(args.clean and not args.baseline),
                remove_provisional_groups=not run_provisional,
            )
        base_cmd = ["cargo", "bench"] + (["--manifest-path", str(manifest)] if manifest else [])
        for target in args.benchmarks:
            base_cmd.extend(["--bench", target])
        if args.precompile:
            run_command([*base_cmd, "--no-run"], cwd=project_root, env=custom_env,
                        dry_run=args.dry_run, phase_label="Step 0/2: Pre-Compile (--no-run)")
            if not args.dry_run:
                prepare_host_for_measurement(max(args.settle_ms, 800) if args.settle_ms > 0 else 0)
        if not args.dry_run:
            apply_process_stabilization(args.high_priority, pin_core)

        print(f"\n=== Step 1: Running Benchmarks (Profile: {args.profile}) ===")
        print(f"Description: {PROFILES[args.profile]['description']}")
        print(f"Targets    : {', '.join(args.benchmarks)}")
        if raw_filter:
            print(f"Filter     : '{raw_filter}'")
        if effective_skip:
            print(f"Skipped    : '{effective_skip}' (LIMINIKA_RUN_PROVISIONAL_LMNK=0)")

        cargo_args: list[str] = [raw_filter] if raw_filter else []
        for flag, val in (("--baseline", args.baseline), ("--save-baseline", args.save_baseline),
                          ("--sample-size", args.sample_size), ("--warm-up-time", args.warm_up_time),
                          ("--measurement-time", args.measurement_time)):
            if val is not None:
                cargo_args.extend([flag, str(val)])
        if args.noise_threshold is not None and args.noise_threshold > 0.0:
            cargo_args.extend(["--noise-threshold", str(args.noise_threshold)])
        if args.noplot:
            cargo_args.append("--noplot")

        for idx, target in enumerate(args.benchmarks):
            if idx > 0 and not args.dry_run:
                prepare_host_for_measurement(max(0, args.settle_ms))
            cmd = ["cargo", "bench"] + (["--manifest-path", str(manifest)] if manifest else []) + ["--bench", target]
            if cargo_args:
                cmd.extend(["--", *cargo_args])
            run_command(cmd, cwd=project_root, env=custom_env, dry_run=args.dry_run,
                        phase_label=f"Step 1/2: Measurement ({idx + 1}/{len(args.benchmarks)}: {target})")

    report_script = project_root / "scripts" / "generate_bench_report.py"
    if not report_script.exists() and not args.dry_run:
        print(f"Error: Report generator missing at {report_script}", file=sys.stderr)
        sys.exit(1)

    crit_dir = args.criterion_dir if args.dry_run else resolve_criterion_dir(project_root, args.criterion_dir)
    report_cmd = [
        sys.executable, str(report_script),
        "--criterion-dir", str(crit_dir),
        "--output", str(args.output_md),
        "--json-summary-out", str(args.json_summary_out),
    ]
    if args.baseline:
        report_cmd.extend(["--baseline", args.baseline])
    if args.fail_on_regression_pct is not None:
        report_cmd.extend(["--fail-on-regression-pct", str(args.fail_on_regression_pct)])
    if args.fail_on_high_noise:
        report_cmd.append("--fail-on-high-noise")
    if args.summary_only:
        report_cmd.append("--summary-only")
    run_command(report_cmd, cwd=project_root, dry_run=args.dry_run,
                phase_label="Step 2/2: Report and HTML Index Synthesis")

    abs_crit = crit_dir if crit_dir.is_absolute() else (project_root / crit_dir)
    index_html = abs_crit / "report" / "index.html"
    if not args.dry_run and index_html.exists():
        resolved = index_html.resolve()
        print(f"Criterion HTML Index : {resolved}")
        print(f"Browser URI          : {resolved.as_uri()}")
        if args.open_index:
            webbrowser.open(resolved.as_uri())
    print("\n=== LiminiKa Benchmark Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()
