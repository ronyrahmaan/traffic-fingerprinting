"""Validate and report on the quality of the captured website dataset.

Run this after capture to get a full quality report:
    - Per-site statistics (packet count, duration, bytes)
    - Per-category breakdown
    - Missing/failed trace detection
    - Outlier detection (abnormally small or large traces)
    - Comparison with Alexa voice command dataset
    - Dataset readiness assessment

Usage:
    cd traffic-fingerprinting
    uv run python scripts/1b_validate_website_dataset.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_fingerprinting.data_loader import load_dataset, load_trace

BASE_DIR = Path(__file__).resolve().parent.parent
WEBSITE_CSV_DIR = BASE_DIR / "data" / "collected" / "website_csv"
ALEXA_CSV_DIR = BASE_DIR / "data" / "raw" / "VCFingerprinting" / "data" / "trace_csv"
MANIFEST_PATH = BASE_DIR / "results" / "website_dataset" / "capture_manifest.json"
REPORT_PATH = BASE_DIR / "results" / "website_dataset" / "quality_report.json"

# Quality thresholds
MIN_PACKETS_PER_TRACE = 50
MIN_TRACES_PER_SITE = 80    # Allow up to 20% failure rate
OUTLIER_IQR_FACTOR = 3.0    # Flag traces beyond 3× IQR


def load_manifest() -> dict | None:
    """Load the capture manifest if available."""
    if not MANIFEST_PATH.exists():
        return None
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def compute_site_stats(
    traces_by_site: dict[str, list],
) -> dict[str, dict]:
    """Compute per-site statistics."""
    stats = {}
    for site_name, traces in sorted(traces_by_site.items()):
        packets = [t.num_packets for t in traces]
        durations = [t.duration for t in traces]
        bytes_list = [sum(abs(p.size) for p in t.packets) for t in traces]

        stats[site_name] = {
            "n_traces": len(traces),
            "packets_mean": round(np.mean(packets), 1),
            "packets_std": round(np.std(packets), 1),
            "packets_min": int(np.min(packets)),
            "packets_max": int(np.max(packets)),
            "packets_median": round(float(np.median(packets)), 1),
            "duration_mean": round(np.mean(durations), 3),
            "duration_std": round(np.std(durations), 3),
            "bytes_mean": round(np.mean(bytes_list), 0),
            "bytes_std": round(np.std(bytes_list), 0),
        }
    return stats


def detect_outliers(
    traces_by_site: dict[str, list],
) -> list[dict]:
    """Detect traces with abnormal packet counts using IQR method."""
    outliers = []
    for site_name, traces in traces_by_site.items():
        packets = np.array([t.num_packets for t in traces])
        if len(packets) < 4:
            continue

        q1 = np.percentile(packets, 25)
        q3 = np.percentile(packets, 75)
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_FACTOR * iqr
        upper = q3 + OUTLIER_IQR_FACTOR * iqr

        for t in traces:
            if t.num_packets < lower or t.num_packets > upper:
                outliers.append({
                    "file": t.filename,
                    "site": site_name,
                    "packets": t.num_packets,
                    "range": f"[{lower:.0f}, {upper:.0f}]",
                    "reason": "below IQR" if t.num_packets < lower else "above IQR",
                })
    return outliers


def compare_with_alexa() -> dict | None:
    """Compare website dataset statistics with Alexa voice command dataset."""
    if not ALEXA_CSV_DIR.exists():
        return None

    try:
        alexa_traces = load_dataset(ALEXA_CSV_DIR)
    except FileNotFoundError:
        return None

    alexa_packets = [t.num_packets for t in alexa_traces]
    alexa_durations = [t.duration for t in alexa_traces]
    alexa_labels = set(t.label for t in alexa_traces)

    return {
        "alexa_n_traces": len(alexa_traces),
        "alexa_n_classes": len(alexa_labels),
        "alexa_packets_mean": round(np.mean(alexa_packets), 1),
        "alexa_packets_std": round(np.std(alexa_packets), 1),
        "alexa_duration_mean": round(np.mean(alexa_durations), 3),
        "alexa_duration_std": round(np.std(alexa_durations), 3),
    }


def main() -> None:
    """Run the full dataset validation and print report."""
    print("=" * 65)
    print("  Website Fingerprinting Dataset — Quality Report")
    print("=" * 65)
    print()

    # Check if dataset exists
    if not WEBSITE_CSV_DIR.exists():
        print(f"ERROR: No dataset found at {WEBSITE_CSV_DIR}")
        print("Run 1b_capture_website_traffic.py first.")
        sys.exit(1)

    csv_files = sorted(WEBSITE_CSV_DIR.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {WEBSITE_CSV_DIR}")
        sys.exit(1)

    # Load all traces
    print(f"Loading {len(csv_files)} traces...")
    traces = []
    load_errors = []
    for f in csv_files:
        try:
            t = load_trace(f)
            traces.append(t)
        except Exception as e:
            load_errors.append({"file": f.name, "error": str(e)})

    # Group by site
    traces_by_site: dict[str, list] = defaultdict(list)
    for t in traces:
        traces_by_site[t.label].append(t)

    # Group by category (from manifest or filename)
    manifest = load_manifest()
    site_to_category: dict[str, str] = {}
    if manifest and "traces" in manifest:
        for entry in manifest["traces"]:
            site_to_category[entry["site_name"]] = entry["category"]

    # ---- Overall Statistics ----
    all_packets = [t.num_packets for t in traces]
    all_durations = [t.duration for t in traces]
    n_sites = len(traces_by_site)

    print(f"\n{'─' * 65}")
    print("  OVERALL STATISTICS")
    print(f"{'─' * 65}")
    print(f"  Total traces:       {len(traces)}")
    print(f"  Total sites:        {n_sites}")
    print(f"  Load errors:        {len(load_errors)}")
    print(f"  Packets/trace:      {np.mean(all_packets):.0f} ± {np.std(all_packets):.0f}")
    print(f"  Packets range:      [{np.min(all_packets)}, {np.max(all_packets)}]")
    print(f"  Duration/trace:     {np.mean(all_durations):.2f}s ± {np.std(all_durations):.2f}s")
    print(f"  Total packets:      {sum(all_packets):,}")

    # ---- Per-Site Statistics ----
    site_stats = compute_site_stats(traces_by_site)

    print(f"\n{'─' * 65}")
    print("  PER-SITE STATISTICS (sorted by mean packets)")
    print(f"{'─' * 65}")
    print(f"  {'Site':<22s} {'Traces':>6s} {'Packets':>10s} {'Std':>8s} {'Duration':>10s}")
    print(f"  {'─' * 58}")

    sorted_sites = sorted(site_stats.items(), key=lambda x: x[1]["packets_mean"])
    for site_name, s in sorted_sites:
        cat = site_to_category.get(site_name, "?")
        print(
            f"  {site_name:<22s} {s['n_traces']:>6d} "
            f"{s['packets_mean']:>10.0f} {s['packets_std']:>8.0f} "
            f"{s['duration_mean']:>9.2f}s"
        )

    # ---- Per-Category Statistics ----
    if site_to_category:
        print(f"\n{'─' * 65}")
        print("  PER-CATEGORY STATISTICS")
        print(f"{'─' * 65}")

        cat_traces: dict[str, list] = defaultdict(list)
        for t in traces:
            cat = site_to_category.get(t.label, "unknown")
            cat_traces[cat].append(t)

        hdr = f"  {'Category':<15s} {'Traces':>7s} {'Sites':>6s} {'Avg Pkts':>10s} {'Avg Dur':>10s}"
        print(hdr)
        print(f"  {'─' * 50}")
        for cat in sorted(cat_traces.keys()):
            ct = cat_traces[cat]
            cat_sites = set(t.label for t in ct)
            pkts = [t.num_packets for t in ct]
            durs = [t.duration for t in ct]
            print(
                f"  {cat:<15s} {len(ct):>7d} {len(cat_sites):>6d} "
                f"{np.mean(pkts):>10.0f} {np.mean(durs):>9.2f}s"
            )

    # ---- Missing Traces ----
    print(f"\n{'─' * 65}")
    print("  MISSING / INCOMPLETE SITES")
    print(f"{'─' * 65}")

    issues = []
    for site_name, site_traces in sorted(traces_by_site.items()):
        n = len(site_traces)
        low_packet = [t for t in site_traces if t.num_packets < MIN_PACKETS_PER_TRACE]
        if n < MIN_TRACES_PER_SITE:
            issues.append(f"  {site_name}: only {n} traces (need {MIN_TRACES_PER_SITE}+)")
        if low_packet:
            issues.append(
                f"  {site_name}: {len(low_packet)} traces below "
                f"{MIN_PACKETS_PER_TRACE}-packet threshold"
            )

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("  None — all sites have sufficient traces and packet counts.")

    # ---- Outlier Detection ----
    outliers = detect_outliers(traces_by_site)

    print(f"\n{'─' * 65}")
    print(f"  OUTLIER TRACES ({OUTLIER_IQR_FACTOR}× IQR)")
    print(f"{'─' * 65}")

    if outliers:
        print(f"  {len(outliers)} outliers detected:")
        for o in outliers[:20]:
            print(
                f"    {o['file']:<35s} {o['packets']:>5d} pkts  "
                f"expected {o['range']}  ({o['reason']})"
            )
        if len(outliers) > 20:
            print(f"    ... and {len(outliers) - 20} more")
    else:
        print("  No outliers detected.")

    # ---- Comparison with Alexa Dataset ----
    alexa_comp = compare_with_alexa()
    if alexa_comp:
        print(f"\n{'─' * 65}")
        print("  CROSS-DOMAIN COMPARISON (Website vs Alexa Voice Commands)")
        print(f"{'─' * 65}")
        print(f"  {'Metric':<25s} {'Website':>12s} {'Alexa':>12s}")
        print(f"  {'─' * 51}")
        print(f"  {'Total traces':<25s} {len(traces):>12d} {alexa_comp['alexa_n_traces']:>12d}")
        print(f"  {'Classes':<25s} {n_sites:>12d} {alexa_comp['alexa_n_classes']:>12d}")
        print(
            f"  {'Mean packets':<25s} {np.mean(all_packets):>12.0f} "
            f"{alexa_comp['alexa_packets_mean']:>12.0f}"
        )
        print(
            f"  {'Mean duration (s)':<25s} {np.mean(all_durations):>12.2f} "
            f"{alexa_comp['alexa_duration_mean']:>12.2f}"
        )

    # ---- Readiness Assessment ----
    print(f"\n{'─' * 65}")
    print("  DATASET READINESS")
    print(f"{'─' * 65}")

    checks = []
    checks.append(("100 sites", n_sites >= 100, f"{n_sites}/100"))
    checks.append((
        f"{MIN_TRACES_PER_SITE}+ traces/site",
        all(len(v) >= MIN_TRACES_PER_SITE for v in traces_by_site.values()),
        f"min {min(len(v) for v in traces_by_site.values())}",
    ))
    checks.append((
        f"All traces >= {MIN_PACKETS_PER_TRACE} packets",
        all(t.num_packets >= MIN_PACKETS_PER_TRACE for t in traces),
        f"{sum(1 for t in traces if t.num_packets < MIN_PACKETS_PER_TRACE)} below",
    ))
    checks.append((
        "10 categories",
        len(set(site_to_category.values())) >= 10 if site_to_category else n_sites >= 100,
        f"{len(set(site_to_category.values())) if site_to_category else '?'}",
    ))

    all_pass = True
    for check_name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check_name:<40s} ({detail})")

    print()
    if all_pass:
        print("  READY for fingerprinting evaluation.")
    else:
        print("  NOT READY — fix the FAIL items above before evaluation.")

    # Save report as JSON
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_traces": len(traces),
        "total_sites": n_sites,
        "load_errors": len(load_errors),
        "packets_mean": round(np.mean(all_packets), 1),
        "packets_std": round(np.std(all_packets), 1),
        "duration_mean": round(np.mean(all_durations), 3),
        "site_stats": site_stats,
        "outliers": outliers,
        "alexa_comparison": alexa_comp,
        "readiness": {name: passed for name, passed, _ in checks},
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved to: {REPORT_PATH}")
    print()


if __name__ == "__main__":
    main()
