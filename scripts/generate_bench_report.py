# name: scripts/generate_bench_report.py
# SPDX-License-Identifier: MIT OR Apache-2.0

"""Generic Criterion JSON benchmark report, OLS scaling, and HTML index generator."""

import argparse
import html
import json
import math
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Normal-distribution consistency scale `1 / Phi^-1(0.75)` applied by Criterion to `median_abs_dev`.
CRITERION_MAD_NORMAL_SCALE: float = 1.482602218505602
NORMAL_STD_TO_MAD_SCALE: float = 0.6744897501960817

_RE_DIM_HEADS = re.compile(r"^(.*)_dim_(\d+)_heads_(\d+)$")
_RE_PARAM_FAMILY = re.compile(r"^(.*)_(tokens|seq|dim)_(\d+)$")
_RE_ANCHOR_IDENT = re.compile(r"[^a-zA-Z0-9_-]")
_SYNTHESIZED_HTML_MARKER = "<!-- synthesized-by: generate_bench_report.py -->"

_COMMON_CSS = (
    "body{font-family:system-ui,-apple-system,sans-serif;margin:2rem auto;max-width:1440px;padding:0 1.5rem;"
    "background:#0f172a;color:#e2e8f0;line-height:1.5}h1{margin-bottom:.25rem;font-size:1.7rem}"
    "h2{margin-top:0;font-size:1.15rem;display:flex;align-items:center;gap:.5rem}.meta{color:#94a3b8;"
    "margin-bottom:1.25rem;font-size:.92rem}a{color:#38bdf8;text-decoration:none}a:hover{text-decoration:underline}"
    ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.9rem;margin-bottom:1.5rem}"
    ".kpi-card{background:#1e293b;border:1px solid #334155;border-radius:10px;padding:.95rem 1.1rem}"
    ".kpi-label{color:#94a3b8;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}"
    ".kpi-value{font-size:1.25rem;font-weight:700;color:#f8fafc;margin-top:.2rem}.kpi-sub{color:#94a3b8;"
    "font-size:.8rem;margin-top:.2rem}.toolbar{display:flex;flex-wrap:wrap;gap:.75rem;align-items:center;"
    "margin-bottom:1.5rem;background:#1e293b;border:1px solid #334155;padding:.75rem 1rem;border-radius:10px}"
    ".search-input{flex:1;min-width:240px;background:#0f172a;border:1px solid #475569;color:#f8fafc;"
    "padding:.5rem .85rem;border-radius:6px;font-size:.92rem}.search-input:focus{outline:none;border-color:#38bdf8}"
    ".group-nav{display:flex;flex-wrap:wrap;gap:.45rem}.nav-chip{font-size:.8rem;background:#0f172a;"
    "border:1px solid #334155;color:#cbd5e1;padding:.25rem .65rem;border-radius:999px}"
    ".nav-chip:hover{border-color:#38bdf8;color:#38bdf8}.group-card{background:#1e293b;border:1px solid #334155;"
    "border-radius:10px;padding:1.2rem;margin-bottom:1.4rem}.badge{font-size:.78rem;background:#0284c7;color:#fff;"
    "padding:.15rem .6rem;border-radius:999px;font-weight:600}table{border-collapse:collapse;width:100%;"
    "margin-top:.75rem;background:#1e293b}th,td{padding:.55rem .8rem;border-bottom:1px solid #334155;"
    "text-align:left;font-size:.88rem}th{background:#0f172a;color:#94a3b8;font-weight:600}"
    "tr:hover td{background:rgba(56,189,248,.05)}.lat-cell{min-width:160px}.lat-bar-wrap{width:100%;height:5px;"
    "background:#0f172a;border-radius:999px;margin-top:.25rem;overflow:hidden}.lat-bar{height:100%;"
    "background:linear-gradient(90deg,#38bdf8,#818cf8);border-radius:999px}.plot{margin-top:1.5rem;background:#fff;"
    "color:#0f172a;padding:1rem;border-radius:8px}"
)


def format_time(nanoseconds: float) -> str:
    """Format nanoseconds into ASCII-safe time units."""
    for thresh, div, unit in ((1.0, 1e-3, "ps"), (1e3, 1.0, "ns"), (1e6, 1e3, "us"), (1e9, 1e6, "ms")):
        if nanoseconds < thresh:
            return f"{nanoseconds / div:.2f} {unit}"
    return f"{nanoseconds / 1e9:.2f} s"


def format_rate(value: float, unit_kind: str) -> str:
    """Format throughput rate for `Bytes` or `Elements`."""
    scales = (
        ((1e12, "TB/s"), (1e9, "GB/s"), (1e6, "MB/s"), (1e3, "KB/s"), (1.0, "B/s"))
        if unit_kind == "Bytes"
        else ((1e9, "GElem/s"), (1e6, "MElem/s"), (1e3, "KElem/s"), (1.0, "Elem/s"))
    )
    for thresh, suffix in scales:
        if value >= thresh:
            return f"{value / thresh:.2f} {suffix}"
    return f"{value:.2f} {scales[-1][1]}"


def relative_posix_link(from_file: Path, target_path: Path) -> str:
    """Compute a POSIX-normalized relative path from `from_file.parent` to `target_path`."""
    try:
        return Path(os.path.relpath(target_path.resolve(), start=from_file.resolve().parent)).as_posix()
    except (OSError, ValueError):
        return target_path.as_posix()


def is_jitter_bounded(
    mean_ns: float, std_dev_ns: float, moe_pct: float, mad_ns: float,
    mad_pct: float, median_ns: float, tp_kind: str | None = None, tp_count: int | None = None,
) -> tuple[bool, bool]:
    """Classify variance as `(rdtsc_quantum_bounded, os_scheduler_jitter_bounded)`."""
    right_skewed = median_ns <= 0.0 or median_ns <= mean_ns * 1.020
    if mean_ns < 30.0 and right_skewed and (std_dev_ns <= 4.0 or mad_ns <= 2.5) and moe_pct <= 3.5:
        return True, False
    os_jitter = right_skewed and moe_pct <= 3.5 and mad_pct <= 6.5
    if not os_jitter and tp_count is not None and moe_pct <= 3.5 and mad_pct <= 8.5:
        os_jitter = (tp_kind == "Elements" and tp_count >= 100_000) or (tp_kind == "Bytes" and tp_count >= 32_768)
    return False, os_jitter


def parse_estimates(est_path: Path) -> dict[str, Any] | None:
    """Parse Criterion `estimates.json` including unscaled non-parametric MAD."""
    try:
        data = json.loads(est_path.read_text(encoding="utf-8"))
        m = data.get("mean", {})
        mean_pt = float(m.get("point_estimate", 0.0))
        ci = m.get("confidence_interval", {})
        low = float(ci.get("lower_bound", m.get("lower_bound", mean_pt)))
        high = float(ci.get("upper_bound", m.get("upper_bound", mean_pt)))
        moe_pct = ((high - low) / (2.0 * mean_pt)) * 100.0 if mean_pt > 0.0 else 0.0
        std_dev = float(data.get("std_dev", {}).get("point_estimate", 0.0))
        median = float(data.get("median", {}).get("point_estimate", mean_pt))
        mad_obj = data.get("median_abs_dev", {})
        mad = (
            float(mad_obj["point_estimate"]) / CRITERION_MAD_NORMAL_SCALE
            if isinstance(mad_obj, dict) and "point_estimate" in mad_obj
            else std_dev * NORMAL_STD_TO_MAD_SCALE
        )
        cov_pct = (std_dev / mean_pt * 100.0) if mean_pt > 0.0 else 0.0
        mad_pct = (mad / median * 100.0) if median > 0.0 else cov_pct
        return {
            "mean_ns": mean_pt, "mean_formatted": format_time(mean_pt),
            "std_dev_ns": std_dev, "std_dev_formatted": format_time(std_dev),
            "median_ns": median, "median_formatted": format_time(median),
            "mad_ns": mad, "mad_formatted": format_time(mad),
            "mad_pct": mad_pct, "margin_of_error_pct": moe_pct, "cov_pct": cov_pct,
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Warning: Failed to parse estimates from {est_path}: {exc}", file=sys.stderr)
        return None


def parse_throughput(bench_info_path: Path, mean_ns: float) -> tuple[str | None, float, str | None, int | None]:
    """Parse Criterion `benchmark.json` throughput (`Bytes` or `Elements`)."""
    if not bench_info_path.exists() or mean_ns <= 0.0:
        return None, 0.0, None, None
    try:
        tp = json.loads(bench_info_path.read_text(encoding="utf-8")).get("throughput")
        if isinstance(tp, dict):
            for kind in ("Bytes", "Elements"):
                if kind in tp:
                    count = int(tp[kind])
                    rate = (count / mean_ns) * 1_000_000_000.0
                    return format_rate(rate, kind), rate, kind, count
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Warning: Failed to parse throughput from {bench_info_path}: {exc}", file=sys.stderr)
    return None, 0.0, None, None


def parse_baseline_comparison(dir_path: Path, current_mean_ns: float, baseline_name: str) -> dict[str, Any] | None:
    """Compare current mean latency against `<baseline>/estimates.json`."""
    base_est = dir_path.parent / baseline_name / "estimates.json"
    if not base_est.exists():
        return None
    base = parse_estimates(base_est)
    if not base or base["mean_ns"] <= 0.0:
        return None
    diff_pct = ((current_mean_ns - base["mean_ns"]) / base["mean_ns"]) * 100.0
    status = ("Neutral (within 1%)" if abs(diff_pct) < 1.0
              else (f"Improved ({abs(diff_pct):.2f}% faster)" if diff_pct < 0.0
                    else f"Degraded ({diff_pct:.2f}% slower)"))
    return {"baseline_name": baseline_name, "base_mean_ns": base["mean_ns"],
            "base_mean_formatted": base["mean_formatted"], "diff_pct": diff_pct, "status_str": status}


def extract_scaling_parameter(name: str) -> tuple[str, int] | None:
    """Extract `(family_prefix, numeric_scale)` for Log-Log OLS complexity fitting."""
    lower = name.lower()
    if "unwrap" in lower or "vocab" in lower:
        return None
    if m := _RE_DIM_HEADS.search(lower):
        return f"{m.group(1)}_head_dim", int(m.group(2))
    if m := _RE_PARAM_FAMILY.search(lower):
        return f"{m.group(1)}_{m.group(2)}", int(m.group(3))
    return None


def compute_log_log_ols_exponent(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Compute OLS slope `alpha` and `R^2` for `ln(T) = alpha * ln(N) + beta`."""
    valid = [(math.log(n), math.log(t)) for n, t in points if n > 0.0 and t > 0.0]
    m = len(valid)
    if m < 2:
        return 0.0, 1.0
    mx, my = sum(x for x, _ in valid) / m, sum(y for _, y in valid) / m
    sxx = sum((x - mx) ** 2 for x, _ in valid)
    syy = sum((y - my) ** 2 for _, y in valid)
    sxy = sum((x - mx) * (y - my) for x, y in valid)
    if sxx <= 1e-18:
        return 0.0, 1.0
    alpha = sxy / sxx
    r_sq = (sxy**2) / (sxx * syy) if syy > 1e-18 else 1.0
    return alpha, max(0.0, min(1.0, r_sq))


def analyze_complexity_scaling(benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect parameter-scaling families and fit Log-Log OLS power-law complexity."""
    families: dict[str, list[dict[str, Any]]] = {}
    for bench in benchmarks:
        if extracted := extract_scaling_parameter(bench["name"]):
            prefix, param_val = extracted
            families.setdefault(f"{bench['group']}::{prefix}", []).append({**bench, "param_val": param_val})
    reports: list[dict[str, Any]] = []
    for family_key, items in sorted(families.items()):
        ordered = sorted({it["param_val"]: it for it in items}.values(), key=lambda x: x["param_val"])
        if len(ordered) < 2:
            continue
        first, last = ordered[0], ordered[-1]
        ratio = last["mean_ns"] / first["mean_ns"] if first["mean_ns"] > 0.0 else 1.0
        alpha, r_sq = compute_log_log_ols_exponent([(float(it["param_val"]), float(it["mean_ns"])) for it in ordered])
        if abs(alpha) <= 0.15 or ratio <= 1.35:
            label, status = f"O(1) Constant (alpha={alpha:.2f})", "Optimal"
        elif alpha < 0.80:
            label, status = f"Sub-Linear O(N^{alpha:.2f}) (R^2={r_sq:.2f})", "SubLinear"
        elif alpha <= 1.15:
            label, status = f"O(N) Linear (alpha={alpha:.2f}, R^2={r_sq:.2f})", "Linear"
        elif (first.get("tp_kind") == "Bytes" and (first.get("tp_count") or 0) <= 65_536
              and (last.get("tp_count") or 0) >= 4_194_304 and alpha <= 1.25 and r_sq >= 0.98):
            label, status = f"O(N) Linear (Cache->DRAM step, alpha={alpha:.2f}, R^2={r_sq:.2f})", "Linear"
        else:
            label, status = f"Super-Linear O(N^{alpha:.2f}) (R^2={r_sq:.2f})", "Bottleneck"
        reports.append({
            "family": family_key, "min_param": first["param_val"], "max_param": last["param_val"],
            "min_time": first["mean_formatted"], "max_time": last["mean_formatted"],
            "param_ratio": last["param_val"] / first["param_val"], "time_ratio": ratio,
            "alpha": alpha, "r_squared": r_sq, "complexity": label, "status": status,
        })
    return reports


def collect_criterion_benchmarks(
    criterion_dir: Path, baseline_name: str | None = None, verbose: bool = False
) -> dict[str, list[dict[str, Any]]]:
    """Scan `criterion_dir` for `**/new/estimates.json` and group records by Criterion group."""
    if not criterion_dir.exists():
        print(f"Error: Criterion output directory '{criterion_dir}' not found.", file=sys.stderr)
        sys.exit(1)
    estimates_files = sorted(criterion_dir.glob("**/new/estimates.json"))
    if not estimates_files:
        print(f"Error: No benchmark estimates found in '{criterion_dir}'.", file=sys.stderr)
        sys.exit(1)
    groups: dict[str, list[dict[str, Any]]] = {}
    for est_path in estimates_files:
        dir_path, bench_dir = est_path.parent, est_path.parent.parent
        try:
            parts = dir_path.relative_to(criterion_dir).parts
            group_name = parts[0] if parts else "Default"
            bench_name = ("_".join(parts[1:-1]) if len(parts) >= 4 and parts[-1] == "new"
                          else (parts[1] if len(parts) > 1 else "Unknown"))
            stats = parse_estimates(est_path)
            if not stats:
                continue
            tp_str, raw_rate, tp_kind, tp_count = parse_throughput(dir_path / "benchmark.json", stats["mean_ns"])
            rdtsc, os_jitter = is_jitter_bounded(
                stats["mean_ns"], stats["std_dev_ns"], stats["margin_of_error_pct"],
                stats["mad_ns"], stats["mad_pct"], stats["median_ns"], tp_kind, tp_count,
            )
            comp = parse_baseline_comparison(dir_path, stats["mean_ns"], baseline_name) if baseline_name else None
            groups.setdefault(group_name, []).append({
                **stats, "group": group_name, "name": bench_name, "bench_dir": bench_dir,
                "bench_report_index": bench_dir / "report" / "index.html",
                "rdtsc_quantum_bounded": rdtsc, "os_scheduler_jitter_bounded": os_jitter,
                "throughput": tp_str or "N/A", "raw_rate": raw_rate,
                "tp_kind": tp_kind, "tp_count": tp_count, "comparison": comp,
            })
            if verbose:
                print(f"Processed: [{group_name}] {bench_name} -> {stats['mean_formatted']}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Warning: Error processing '{est_path}': {exc}", file=sys.stderr)
    return groups


def _should_write_html(index_path: Path) -> bool:
    if not index_path.exists():
        return True
    try:
        return _SYNTHESIZED_HTML_MARKER in index_path.read_text(encoding="utf-8", errors="ignore")[:512]
    except OSError:
        return True


def _html_doc(title: str, body: str) -> str:
    return (f"<!DOCTYPE html>\n{_SYNTHESIZED_HTML_MARKER}\n<html lang=\"en\"><head><meta charset=\"utf-8\"/>"
            f"<title>{html.escape(title)}</title><style>{_COMMON_CSS}</style></head><body>{body}</body></html>")


def _render_html_rows(benchmarks: list[dict[str, Any]], from_index: Path) -> str:
    ordered = sorted(benchmarks, key=lambda x: x["mean_ns"])
    min_ns = max(0.1, ordered[0]["mean_ns"]) if ordered else 1.0
    max_ns = max(min_ns * 1.01, ordered[-1]["mean_ns"]) if ordered else 10.0
    span = max(1e-6, math.log10(max_ns) - math.log10(min_ns))
    rows: list[str] = []
    for b in ordered:
        b_idx = b.get("bench_report_index")
        b_link = relative_posix_link(from_index, b_idx) if isinstance(b_idx, Path) else "#"
        pct = 35.0 if len(ordered) == 1 else min(
            100.0, max(6.0, ((math.log10(max(0.1, b["mean_ns"])) - math.log10(min_ns)) / span) * 100.0)
        )
        rows.append(
            f'<tr data-bench="{html.escape(b["name"].lower())}"><td><a href="{html.escape(b_link)}">'
            f'<code>{html.escape(b["name"])}</code></a></td><td class="lat-cell"><strong>'
            f'{html.escape(b["mean_formatted"])}</strong><div class="lat-bar-wrap">'
            f'<div class="lat-bar" style="width:{pct:.1f}%"></div></div></td>'
            f'<td>{html.escape(b["median_formatted"])}</td><td>{html.escape(b["std_dev_formatted"])}</td>'
            f'<td>{b["mad_pct"]:.2f}%</td><td>&plusmn;{b["margin_of_error_pct"]:.2f}%</td>'
            f'<td>{b["cov_pct"]:.2f}%</td><td>{html.escape(b["throughput"])}</td></tr>'
        )
    return "".join(rows)


def ensure_criterion_html_index(
    criterion_dir: Path, groups: dict[str, list[dict[str, Any]]],
    utc_now: str, high_variance_count: int, regression_count: int,
) -> Path:
    """Synthesize `<criterion_dir>/report/index.html` plus per-group and per-bench reports."""
    root_index = criterion_dir / "report" / "index.html"
    thead = ("<table><thead><tr><th>Benchmark</th><th>Mean</th><th>Median</th><th>Std Dev</th>"
             "<th>MAD (%)</th><th>MoE (95% CI)</th><th>CoV (%)</th><th>Throughput</th></tr></thead>")
    for group_name, benchmarks in sorted(groups.items()):
        g_dir = criterion_dir / group_name / "report"
        g_index = g_dir / "index.html"
        for b in benchmarks:
            b_dir = b.get("bench_dir")
            if not isinstance(b_dir, Path) or not _should_write_html(b_dir / "report" / "index.html"):
                continue
            b_report, b_index = b_dir / "report", b_dir / "report" / "index.html"
            b_report.mkdir(parents=True, exist_ok=True)
            svgs = "".join(
                f'<div class="plot"><h4>{s}</h4><img src="{s}" alt="{s}"/></div>'
                for s in ("pdf.svg", "violin.svg", "lines.svg", "typical.svg", "MAD.svg", "SD.svg")
                if (b_report / s).exists()
            ) or "<p><em>Estimates read from ../new/estimates.json.</em></p>"
            back = html.escape(relative_posix_link(b_index, root_index))
            rows = (
                f"<tr><td>Mean</td><td><strong>{html.escape(b['mean_formatted'])}</strong></td></tr>"
                f"<tr><td>Median</td><td>{html.escape(b['median_formatted'])}</td></tr>"
                f"<tr><td>Std Dev</td><td>{html.escape(b['std_dev_formatted'])}</td></tr>"
                f"<tr><td>MAD</td><td>{html.escape(b['mad_formatted'])} ({b['mad_pct']:.2f}%)</td></tr>"
                f"<tr><td>95% CI MoE</td><td>&plusmn;{b['margin_of_error_pct']:.2f}%</td></tr>"
                f"<tr><td>CoV</td><td>{b['cov_pct']:.2f}%</td></tr>"
                f"<tr><td>Throughput</td><td>{html.escape(b['throughput'])}</td></tr>"
            )
            b_index.write_text(_html_doc(
                f"{group_name} / {b['name']}",
                f"<h2>{html.escape(group_name)} / <code>{html.escape(b['name'])}</code></h2>"
                f'<p><a href="{back}">Back to Root Index</a></p>'
                f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>{svgs}",
            ), encoding="utf-8")
        if _should_write_html(g_index):
            g_dir.mkdir(parents=True, exist_ok=True)
            violin = '<div class="plot"><img src="violin.svg" alt="violin.svg"/></div>' if (
                g_dir / "violin.svg"
            ).exists() else ""
            back = html.escape(relative_posix_link(g_index, root_index))
            g_index.write_text(_html_doc(
                f"Group: {group_name}",
                f"<h1>Benchmark Group: <code>{html.escape(group_name)}</code></h1>"
                f'<p><a href="{back}">Back to Root Index</a></p>{violin}'
                f"{thead}<tbody>{_render_html_rows(benchmarks, g_index)}</tbody></table>",
            ), encoding="utf-8")

    if _should_write_html(root_index):
        root_index.parent.mkdir(parents=True, exist_ok=True)
        chips, sections = [], []
        for group_name, benchmarks in sorted(groups.items()):
            g_link = html.escape(relative_posix_link(root_index, criterion_dir / group_name / "report" / "index.html"))
            anchor = f"group-{_RE_ANCHOR_IDENT.sub('-', group_name.lower())}"
            chips.append(f'<a class="nav-chip" href="#{anchor}">{html.escape(group_name)} ({len(benchmarks)})</a>')
            sections.append(
                f'<section class="group-card" id="{anchor}" data-group="{html.escape(group_name.lower())}">'
                f'<h2><a href="{g_link}">{html.escape(group_name)}</a> '
                f'<span class="badge">{len(benchmarks)} workloads</span></h2>'
                f"{thead}<tbody>{_render_html_rows(benchmarks, root_index)}</tbody></table></section>"
            )
        all_b = [b for b_list in groups.values() for b in b_list]
        fastest = min(all_b, key=lambda x: x["mean_ns"]) if all_b else None
        slowest = max(all_b, key=lambda x: x["mean_ns"]) if all_b else None
        js = (
            "function filterBenchmarks(q){const qr=(q||'').trim().toLowerCase();"
            "document.querySelectorAll('.group-card').forEach(sec=>{const g=sec.dataset.group||'';let v=0;"
            "sec.querySelectorAll('tbody tr').forEach(r=>{const ok=!qr||r.dataset.bench.includes(qr)||g.includes(qr);"
            "r.style.display=ok?'':'none';if(ok)v++;});sec.style.display=v>0?'':'none';});}"
        )
        cards = "".join([
            f'<div class="kpi-card"><div class="kpi-label">Groups / Workloads</div>'
            f'<div class="kpi-value">{len(groups)} / {len(all_b)}</div>'
            f'<div class="kpi-sub">High-variance: {high_variance_count}</div></div>',
            f'<div class="kpi-card"><div class="kpi-label">Fastest</div>'
            f'<div class="kpi-value">{html.escape(fastest["mean_formatted"] if fastest else "N/A")}</div>'
            f'<div class="kpi-sub">{html.escape(fastest["name"] if fastest else "-")}</div></div>',
            f'<div class="kpi-card"><div class="kpi-label">Slowest</div>'
            f'<div class="kpi-value">{html.escape(slowest["mean_formatted"] if slowest else "N/A")}</div>'
            f'<div class="kpi-sub">{html.escape(slowest["name"] if slowest else "-")}</div></div>',
            f'<div class="kpi-card"><div class="kpi-label">Regressions</div>'
            f'<div class="kpi-value">{regression_count}</div><div class="kpi-sub">Baseline check</div></div>',
        ])
        root_index.write_text(_html_doc(
            "LiminiKa Criterion Benchmark Index",
            f"<h1>LiminiKa Criterion Benchmark Index</h1>"
            f'<p class="meta">Generated on <code>{html.escape(utc_now)}</code></p>'
            f'<div class="kpi-grid">{cards}</div><div class="toolbar"><input type="search" class="search-input" '
            'placeholder="Filter benchmarks by name or group..." oninput="filterBenchmarks(this.value)"/>'
            f'<div class="group-nav">{"".join(chips)}</div></div>{"".join(sections)}<script>{js}</script>',
        ), encoding="utf-8")
    return root_index


def _render_group_tables(
    groups: dict[str, list[dict[str, Any]]], criterion_dir: Path, output_file: Path, baseline_name: str | None
) -> list[str]:
    lines = ["## Benchmark Metrics by Group", ""]
    for group_name, benchmarks in sorted(groups.items()):
        rel_g = relative_posix_link(output_file, criterion_dir / group_name / "report" / "index.html")
        lines.extend([f"### `{group_name}` ([HTML Report]({rel_g}))", ""])
        if baseline_name:
            lines.extend(["| Benchmark | Mean | Std Dev | Throughput | Baseline | Change | HTML |",
                          "|:----------|-----:|--------:|-----------:|---------:|:-------|:----:|"])
        else:
            lines.extend([
                "| Benchmark | Mean | Median | Std Dev | MAD (%) | MoE (95% CI) | CoV (%) | Throughput | HTML |",
                "|:----------|-----:|-------:|--------:|--------:|-------------:|--------:|-----------:|:-----|",
            ])
        for b in sorted(benchmarks, key=lambda x: x["mean_ns"]):
            b_idx = b.get("bench_report_index")
            rel_b = relative_posix_link(output_file, b_idx) if isinstance(b_idx, Path) else rel_g
            if baseline_name:
                comp = b["comparison"]
                lines.append(
                    f"| `{b['name']}` | **{b['mean_formatted']}** | {b['std_dev_formatted']} | "
                    f"{b['throughput']} | {comp['base_mean_formatted'] if comp else 'N/A'} | "
                    f"{comp['status_str'] if comp else 'N/A'} | [HTML]({rel_b}) |"
                )
            else:
                lines.append(
                    f"| `{b['name']}` | **{b['mean_formatted']}** | {b['median_formatted']} | "
                    f"{b['std_dev_formatted']} | {b['mad_pct']:.2f}% | +/-{b['margin_of_error_pct']:.2f}% | "
                    f"{b['cov_pct']:.2f}% | {b['throughput']} | [HTML]({rel_b}) |"
                )
        lines.append("")
    return lines


def _strip_paths(rec: dict[str, Any] | None) -> dict[str, Any] | None:
    return {k: v for k, v in rec.items() if k not in ("bench_dir", "bench_report_index")} if rec else None


def generate_markdown_report(
    criterion_dir: Path, output_file: Path, verbose: bool = False, summary_only: bool = False,
    fail_on_high_noise: bool = False, baseline_name: str | None = None,
    fail_on_regression_pct: float | None = None, json_summary_out: Path | None = None,
) -> None:
    """Scan Criterion artifacts, synthesize HTML index, and write Markdown and JSON reports."""
    groups = collect_criterion_benchmarks(criterion_dir, baseline_name=baseline_name, verbose=verbose)
    all_b = [b for b_list in groups.values() for b in b_list]
    utc_now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    fastest = min(all_b, key=lambda x: x["mean_ns"]) if all_b else None
    slowest = max(all_b, key=lambda x: x["mean_ns"]) if all_b else None
    noisy = [b for b in all_b if b["margin_of_error_pct"] > 3.5 or b["cov_pct"] > 10.0]
    high_var = [b for b in noisy if not b["rdtsc_quantum_bounded"] and not b["os_scheduler_jitter_bounded"]]
    bounded = [b for b in noisy if b["rdtsc_quantum_bounded"] or b["os_scheduler_jitter_bounded"]]
    regressed = (
        [b for b in all_b if b["comparison"] and b["comparison"]["diff_pct"] > fail_on_regression_pct]
        if baseline_name and fail_on_regression_pct is not None else []
    )
    scaling = analyze_complexity_scaling(all_b)
    root_html = ensure_criterion_html_index(criterion_dir, groups, utc_now, len(high_var), len(regressed))
    rel_html = relative_posix_link(output_file, root_html)

    md = [
        "# LiminiKa Benchmark Report", "", f"Generated from Criterion estimates on `{utc_now}`.", "",
        "## Executive Summary", "",
        f"- **Total Groups**: `{len(groups)}`", f"- **Total Workloads**: `{len(all_b)}`",
        f"- **High-Variance Workloads**: `{len(high_var)}` (jitter-bounded: `{len(bounded)}`)",
        f"- **Criterion Artifact Directory**: `{criterion_dir}`",
        f"- **Criterion HTML Index**: [`{(criterion_dir / 'report' / 'index.html').as_posix()}`]({rel_html})",
    ]
    if baseline_name:
        md.append(f"- **Baseline**: `{baseline_name}` (regressions above threshold: `{len(regressed)}`)")
    if fastest:
        md.append(f"- **Fastest**: `{fastest['group']} / {fastest['name']}` ({fastest['mean_formatted']})")
    if slowest:
        md.append(f"- **Slowest**: `{slowest['group']} / {slowest['name']}` ({slowest['mean_formatted']})")
    md.append("")
    if scaling:
        md.extend(["## Complexity Scaling (Log-Log OLS)", "",
                   "| Family / Group | Min Param | Max Param | Min Time | Max Time | Ratio | Fit |",
                   "|:---------------|----------:|----------:|---------:|---------:|------:|:----|"])
        for sa in scaling:
            md.append(f"| `{sa['family']}` | `{sa['min_param']}` | `{sa['max_param']}` | "
                      f"{sa['min_time']} | {sa['max_time']} | {sa['time_ratio']:.1f}x | {sa['complexity']} |")
        md.append("")
    if noisy:
        md.extend([
            "## Variance and Jitter Diagnostics", "",
            "| Group | Benchmark | Mean | Median | Std Dev | MAD (%) | MoE (95% CI) | CoV (%) | Class |",
            "|:------|:----------|-----:|-------:|--------:|--------:|-------------:|--------:|:------|",
        ])
        for hv in sorted(noisy, key=lambda x: x["cov_pct"], reverse=True):
            cls = ("RDTSC Quantum Floor" if hv["rdtsc_quantum_bounded"]
                   else ("OS Scheduler Jitter Floor" if hv["os_scheduler_jitter_bounded"] else "High Variance"))
            md.append(f"| `{hv['group']}` | `{hv['name']}` | {hv['mean_formatted']} | {hv['median_formatted']} | "
                      f"{hv['std_dev_formatted']} | {hv['mad_pct']:.2f}% | +/-{hv['margin_of_error_pct']:.2f}% | "
                      f"{hv['cov_pct']:.2f}% | {cls} |")
        md.append("")
    if not summary_only:
        md.extend(_render_group_tables(groups, criterion_dir, output_file, baseline_name))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] Generated report ({len(all_b)} workloads, {len(groups)} groups) at '{output_file}'.")
    print(f"[OK] Criterion HTML index ready at '{root_html}'.")

    if json_summary_out:
        json_summary_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": utc_now, "criterion_html_index": root_html.as_posix(),
            "total_groups": len(groups), "total_benchmarks": len(all_b),
            "high_variance_count": len(high_var), "bounded_jitter_count": len(bounded),
            "bottleneck_count": sum(1 for sa in scaling if sa["status"] == "Bottleneck"),
            "regression_count": len(regressed),
            "fastest_bench": _strip_paths(fastest), "slowest_bench": _strip_paths(slowest),
            "complexity_scaling": scaling,
        }
        json_summary_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] Exported JSON benchmark summary at '{json_summary_out}'.")

    if fail_on_high_noise and high_var:
        print(f"Error: High variance detected in {len(high_var)} benchmark(s).", file=sys.stderr)
        sys.exit(1)
    if regressed:
        print(f"Error: {len(regressed)} benchmark(s) exceeded --fail-on-regression-pct ({fail_on_regression_pct}%).",
              file=sys.stderr)
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Markdown, JSON, and HTML reports from Criterion outputs.")
    p.add_argument("--criterion-dir", type=Path, default=Path("target/criterion"), help="Criterion output dir")
    p.add_argument("--output", type=Path, default=Path("docs/benchmark/benchmarks.md"), help="Output Markdown path")
    p.add_argument("--verbose", action="store_true", help="Print verbose processing logs")
    p.add_argument("--summary-only", action="store_true", help="Omit per-group metric tables")
    p.add_argument("--fail-on-high-noise", action="store_true", help="Fail if high variance is detected")
    p.add_argument("--baseline", type=str, default=None, help="Criterion baseline name to compare against")
    p.add_argument("--fail-on-regression-pct", type=float, default=None, help="Fail above regression percent")
    p.add_argument("--json-summary-out", type=Path, default=None, help="Optional JSON summary output path")
    args = p.parse_args()
    generate_markdown_report(
        args.criterion_dir, args.output, args.verbose, args.summary_only, args.fail_on_high_noise,
        args.baseline, args.fail_on_regression_pct, args.json_summary_out,
    )


if __name__ == "__main__":
    main()
