"""Evaluate all fingerprinting algorithms on the website traffic dataset.

Runs the same 7 classifiers used on the Alexa voice command data:
  Paper algorithms:  LL-Jaccard, LL-NB, VNG++, AdaBoost
  My additions:  CUMUL+XGB, Timing+XGB, Combined+RF

Also generates:
  - Cross-domain comparison (website vs Alexa voice commands)
  - Per-category accuracy breakdown
  - Publication-quality figures

Results saved to:
  results/website_dataset/evaluation_results.json
  figures/website/*.png

Usage:
    cd traffic-fingerprinting
    uv run python scripts/1b_run_website_evaluation.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from traffic_fingerprinting.classifiers import (
    cross_validate_jaccard_detailed,
    cross_validate_sklearn_detailed,
)
from traffic_fingerprinting.data_loader import (
    get_labels_and_ids,
    load_dataset,
    traces_to_arrays,
)
from traffic_fingerprinting.features import (
    build_combined_matrix,
    build_cumul_matrix,
    build_llnb_matrix,
    build_svm_matrix,
    build_timing_matrix,
    build_vngpp_matrix,
    compute_bursts,
)

# =====================================================================
# Paths
# =====================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
WEBSITE_CSV_DIR = BASE_DIR / "data" / "collected" / "website_csv"
ALEXA_CSV_DIR = BASE_DIR / "data" / "raw" / "VCFingerprinting" / "data" / "trace_csv"
RESULTS_DIR = BASE_DIR / "results" / "website_eval"
FIGURES_DIR = BASE_DIR / "figures" / "website"
ALEXA_RESULTS_PATH = BASE_DIR / "results" / "alexa_dataset" / "expanded_results.json"
MANIFEST_PATH = RESULTS_DIR / "capture_manifest.json"

# Minimum traces per site to include in evaluation
# (filters out stray traces from sites 51-100 that have only 1-2 traces)
MIN_TRACES_PER_SITE = 10

# Website categories (from capture script)
SITE_CATEGORIES: dict[str, str] = {
    "nytimes": "news", "cnn": "news", "bbc": "news", "reuters": "news",
    "apnews": "news", "foxnews": "news", "nbcnews": "news",
    "usatoday": "news", "washingtonpost": "news", "npr": "news",
    "reddit": "social", "twitter": "social", "facebook": "social",
    "linkedin": "social", "pinterest": "social", "tumblr": "social",
    "quora": "social", "mastodon": "social", "threads": "social",
    "discord": "social",
    "amazon": "ecommerce", "ebay": "ecommerce", "walmart": "ecommerce",
    "etsy": "ecommerce", "target": "ecommerce", "bestbuy": "ecommerce",
    "homedepot": "ecommerce", "costco": "ecommerce", "wayfair": "ecommerce",
    "newegg": "ecommerce",
    "github": "tech", "stackoverflow": "tech", "hackernews": "tech",
    "gitlab": "tech", "devto": "tech", "medium": "tech",
    "techcrunch": "tech", "arstechnica": "tech", "theverge": "tech",
    "wired": "tech",
    "youtube": "entertainment", "netflix": "entertainment",
    "spotify": "entertainment", "imdb": "entertainment",
    "twitch": "entertainment", "hulu": "entertainment",
    "rottentomatoes": "entertainment", "soundcloud": "entertainment",
    "disneyplus": "entertainment", "crunchyroll": "entertainment",
}


# =====================================================================
# Helpers
# =====================================================================


def load_alexa_results() -> dict | None:
    """Load previous Alexa evaluation results for cross-domain comparison."""
    if not ALEXA_RESULTS_PATH.exists():
        return None
    with open(ALEXA_RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def compute_per_class_accuracy(
    y_true: list, y_pred: list, id_to_label: dict,
) -> dict:
    """Compute accuracy per class."""
    per_class = {}
    for cls in sorted(set(y_true)):
        mask = [i for i, y in enumerate(y_true) if y == cls]
        correct = sum(1 for i in mask if y_true[i] == y_pred[i])
        total = len(mask)
        label = id_to_label.get(cls, str(cls))
        per_class[label] = round(correct / total * 100, 1) if total > 0 else 0.0

    sorted_items = sorted(per_class.items(), key=lambda x: x[1])
    return {
        "all": per_class,
        "top_10": dict(sorted_items[-10:]),
        "bottom_10": dict(sorted_items[:10]),
    }


def compute_per_category_accuracy(
    y_true: list,
    y_pred: list,
    id_to_label: dict,
) -> dict[str, float]:
    """Compute accuracy per website category."""
    cat_correct: dict[str, int] = defaultdict(int)
    cat_total: dict[str, int] = defaultdict(int)

    for t, p in zip(y_true, y_pred):
        label = id_to_label.get(t, str(t))
        cat = SITE_CATEGORIES.get(label, "unknown")
        cat_total[cat] += 1
        if t == p:
            cat_correct[cat] += 1

    return {
        cat: round(cat_correct[cat] / cat_total[cat] * 100, 1)
        for cat in sorted(cat_total.keys())
    }


def compute_dataset_stats(traces, labels) -> dict:
    """Compute overall dataset statistics."""
    all_pkts = [t.num_packets for t in traces]
    all_durs = [t.duration for t in traces]
    all_bytes = [sum(abs(p.size) for p in t.packets) for t in traces]
    all_burst_counts = [len(compute_bursts(t)) for t in traces]

    return {
        "num_traces": len(traces),
        "num_classes": len(labels),
        "packets": {
            "mean": round(float(np.mean(all_pkts)), 1),
            "median": int(np.median(all_pkts)),
            "min": int(min(all_pkts)),
            "max": int(max(all_pkts)),
            "std": round(float(np.std(all_pkts)), 1),
        },
        "duration": {
            "mean": round(float(np.mean(all_durs)), 3),
            "min": round(float(min(all_durs)), 3),
            "max": round(float(max(all_durs)), 3),
        },
        "total_bytes": {
            "mean": round(float(np.mean(all_bytes)), 0),
            "min": int(min(all_bytes)),
            "max": int(max(all_bytes)),
        },
        "bursts": {
            "mean": round(float(np.mean(all_burst_counts)), 1),
            "min": int(min(all_burst_counts)),
            "max": int(max(all_burst_counts)),
        },
        "packet_counts": all_pkts,
        "durations": [round(d, 4) for d in all_durs],
    }


# =====================================================================
# Figure generation
# =====================================================================


def fig_algorithm_comparison(algo_results: dict) -> None:
    """Bar chart of all algorithm accuracies on website data."""
    paper_algos = ["LL-Jaccard", "LL-NB", "VNG++", "AdaBoost"]
    improvement_algos = [a for a in algo_results if a not in paper_algos]
    all_algos = paper_algos + improvement_algos
    accs = [algo_results[a]["accuracy"] for a in all_algos]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    colors = ["#4C72B0"] * len(paper_algos) + ["#55A868"] * len(improvement_algos)
    bars = ax.bar(range(len(all_algos)), accs, color=colors,
                  edgecolor="black", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(y=2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(len(all_algos) - 0.5, 2.8, "Random Guess (2%)", fontsize=8, color="gray")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Website Traffic Fingerprinting: All Algorithms (50 sites)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(all_algos)))
    ax.set_xticklabels(all_algos, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, max(accs) + 8)
    ax.legend(
        [plt.Rectangle((0, 0), 1, 1, fc="#4C72B0"),
         plt.Rectangle((0, 0), 1, 1, fc="#55A868")],
        ["Paper Algorithms", "My Improvements"],
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "website_algorithm_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: website_algorithm_comparison.png")


def fig_cross_domain_comparison(website_results: dict, alexa_results: dict) -> None:
    """Side-by-side bar chart: website vs Alexa accuracy for each algorithm."""
    algos = ["LL-Jaccard", "LL-NB", "VNG++", "AdaBoost",
             "CUMUL+XGB", "Timing+XGB", "Combined+XGB", "Combined+RF"]
    # Filter to algorithms present in both
    algos = [a for a in algos if a in website_results and a in alexa_results]

    web_accs = [website_results[a]["accuracy"] for a in algos]
    alexa_accs = [alexa_results[a]["accuracy"] for a in algos]

    x = np.arange(len(algos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    bars1 = ax.bar(x - width / 2, alexa_accs, width,
                   label="Alexa Voice Commands (100 classes, 1K traces)",
                   color="#C44E52", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, web_accs, width,
                   label="Website Traffic (50 sites, 5K traces)",
                   color="#4C72B0", edgecolor="black", linewidth=0.5)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Cross-Domain Comparison: Voice Commands vs Website Traffic",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, max(max(web_accs), max(alexa_accs)) + 10)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_domain_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: cross_domain_comparison.png")


def fig_per_category_accuracy(cat_accs: dict[str, float], algo_name: str) -> None:
    """Horizontal bar chart of accuracy per website category."""
    cats = sorted(cat_accs.keys(), key=lambda c: cat_accs[c])
    accs = [cat_accs[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cats)))
    bars = ax.barh(range(len(cats)), accs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, accs):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=11)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy by Website Category — {algo_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, max(accs) + 10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "per_category_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: per_category_accuracy.png")


def fig_per_site_accuracy(per_class: dict, algo_name: str) -> None:
    """Horizontal bar chart: top 10 and bottom 10 sites."""
    top_10 = per_class["top_10"]
    bottom_10 = per_class["bottom_10"]

    labels = list(bottom_10.keys()) + ["..."] + list(top_10.keys())
    values = list(bottom_10.values()) + [0] + list(top_10.values())

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#C44E52"] * len(bottom_10) + ["white"] + ["#55A868"] * len(top_10)

    bars = ax.barh(range(len(labels)), values, color=colors,
                   edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", fontsize=9)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Per-Site Accuracy — {algo_name} (Top 10 + Bottom 10)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "per_site_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: per_site_accuracy.png")


def fig_per_fold_variance(algo_results: dict) -> None:
    """Box plot showing per-fold accuracy variance."""
    fig, ax = plt.subplots(figsize=(10, 5))

    algos = list(algo_results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3"]

    for i, algo in enumerate(algos):
        folds = algo_results[algo]["fold_accuracies"]
        bp = ax.boxplot([folds], positions=[i], widths=0.4,
                        patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 6})
        bp["boxes"][0].set_facecolor(colors[i % len(colors)])
        bp["boxes"][0].set_alpha(0.7)
        ax.scatter([i] * len(folds), folds, color=colors[i % len(colors)],
                   zorder=5, s=40, edgecolor="black", linewidth=0.5)

    ax.set_xticks(list(range(len(algos))))
    ax.set_xticklabels(algos, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Accuracy per Fold (%)", fontsize=12)
    ax.set_title("Per-Fold Accuracy Variance — Website Data (5-Fold CV)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "per_fold_variance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: per_fold_variance.png")


def fig_dataset_stats(stats: dict) -> None:
    """Histograms of packet counts and durations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.hist(stats["packet_counts"], bins=40, color="#4C72B0",
             edgecolor="black", linewidth=0.5)
    ax1.axvline(x=stats["packets"]["mean"], color="red", linestyle="--",
                linewidth=1.5, label=f"Mean: {stats['packets']['mean']:.0f}")
    ax1.set_xlabel("Packets per Trace", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of Trace Lengths (Website)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.hist(stats["durations"], bins=40, color="#55A868",
             edgecolor="black", linewidth=0.5)
    ax2.axvline(x=stats["duration"]["mean"], color="red", linestyle="--",
                linewidth=1.5, label=f"Mean: {stats['duration']['mean']:.2f}s")
    ax2.set_xlabel("Duration (seconds)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of Trace Durations (Website)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_stats.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: dataset_stats.png")


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    """Run full evaluation on website dataset."""
    print("=" * 65)
    print("  WEBSITE TRAFFIC FINGERPRINTING — Full Evaluation")
    print("=" * 65)

    # ------------------------------------------------------------------
    # [1] Load website data
    # ------------------------------------------------------------------
    print("\n[1/8] Loading website dataset...")
    all_traces = load_dataset(WEBSITE_CSV_DIR)
    print(f"  Loaded {len(all_traces)} traces total")

    # Filter out sites with too few traces (stray captures from sites 51-100)
    site_counts: dict[str, int] = defaultdict(int)
    for t in all_traces:
        site_counts[t.label] += 1

    excluded = {s for s, c in site_counts.items() if c < MIN_TRACES_PER_SITE}
    if excluded:
        print(f"  Excluding {len(excluded)} sites with < {MIN_TRACES_PER_SITE} traces: {excluded}")
        all_traces = [t for t in all_traces if t.label not in excluded]

    labels, label_to_id = get_labels_and_ids(all_traces)
    id_to_label = {v: k for k, v in label_to_id.items()}
    traces, y = traces_to_arrays(all_traces, label_to_id)
    print(f"  Using {len(traces)} traces, {len(labels)} sites")
    print(f"  Sites: {', '.join(labels[:10])}... ({len(labels)} total)")

    # ------------------------------------------------------------------
    # [2] Dataset statistics
    # ------------------------------------------------------------------
    print("\n[2/8] Computing dataset statistics...")
    dataset_stats = compute_dataset_stats(traces, labels)
    print(f"  Avg {dataset_stats['packets']['mean']:.0f} packets/trace, "
          f"{dataset_stats['duration']['mean']:.2f}s duration")
    print(f"  Range: {dataset_stats['packets']['min']}–{dataset_stats['packets']['max']} packets")

    # ------------------------------------------------------------------
    # [3] Build feature matrices
    # ------------------------------------------------------------------
    print("\n[3/8] Building feature matrices...")
    feature_sets = {
        "LL-NB": ("llnb", build_llnb_matrix(traces)),
        "VNG++": ("vngpp", build_vngpp_matrix(traces)),
        "AdaBoost": ("adaboost", build_svm_matrix(traces)),
        "CUMUL+XGB": ("xgboost", build_cumul_matrix(traces)),
        "Timing+XGB": ("xgboost", build_timing_matrix(traces)),
        "Combined+XGB": ("xgboost", build_combined_matrix(traces)),
        "Combined+RF": ("random_forest", build_combined_matrix(traces)),
    }
    for name, (_, X) in feature_sets.items():
        print(f"  {name}: {X.shape}")

    # ------------------------------------------------------------------
    # [4] Run all algorithms
    # ------------------------------------------------------------------
    print("\n[4/8] Running algorithms (5-fold stratified CV)...")
    algo_results: dict[str, dict] = {}

    # LL-Jaccard (special — no feature matrix, pairwise similarity)
    print("  LL-Jaccard...", end=" ", flush=True)
    t0 = time.time()
    jac = cross_validate_jaccard_detailed(traces, y)
    jac_time = time.time() - t0
    algo_results["LL-Jaccard"] = {
        "accuracy": round(jac["mean_accuracy"] * 100, 1),
        "fold_accuracies": [round(a * 100, 1) for a in jac["fold_accuracies"]],
        "time_sec": round(jac_time, 1),
        "y_true": jac["y_true"],
        "y_pred": jac["y_pred"],
    }
    print(f"{algo_results['LL-Jaccard']['accuracy']}%  ({jac_time:.1f}s)")

    # All sklearn-based algorithms
    for name, (clf_name, X) in feature_sets.items():
        print(f"  {name}...", end=" ", flush=True)
        t0 = time.time()
        detail = cross_validate_sklearn_detailed(X, y, clf_name)
        elapsed = time.time() - t0
        algo_results[name] = {
            "accuracy": round(detail["mean_accuracy"] * 100, 1),
            "fold_accuracies": [round(a * 100, 1) for a in detail["fold_accuracies"]],
            "features": X.shape[1],
            "time_sec": round(elapsed, 1),
            "y_true": detail["y_true"],
            "y_pred": detail["y_pred"],
        }
        print(f"{algo_results[name]['accuracy']}%  ({elapsed:.1f}s, {X.shape[1]} features)")

    # ------------------------------------------------------------------
    # [5] Per-class and per-category analysis
    # ------------------------------------------------------------------
    print("\n[5/8] Computing per-site and per-category analysis...")
    best_algo = max(algo_results, key=lambda k: algo_results[k]["accuracy"])

    best_per_class = compute_per_class_accuracy(
        algo_results[best_algo]["y_true"],
        algo_results[best_algo]["y_pred"],
        id_to_label,
    )
    best_per_category = compute_per_category_accuracy(
        algo_results[best_algo]["y_true"],
        algo_results[best_algo]["y_pred"],
        id_to_label,
    )
    print(f"  Best algorithm: {best_algo} ({algo_results[best_algo]['accuracy']}%)")
    print(f"  Per-category accuracy ({best_algo}):")
    for cat, acc in sorted(best_per_category.items(), key=lambda x: x[1], reverse=True):
        print(f"    {cat:<15s} {acc:>6.1f}%")

    # ------------------------------------------------------------------
    # [6] Load Alexa results for cross-domain comparison
    # ------------------------------------------------------------------
    print("\n[6/8] Loading Alexa results for cross-domain comparison...")
    alexa_data = load_alexa_results()
    alexa_algo_results = {}
    if alexa_data:
        alexa_algo_results = alexa_data.get("algorithm_results", {})
        print(f"  Loaded Alexa results: {list(alexa_algo_results.keys())}")
    else:
        print("  No Alexa results found — skipping cross-domain comparison")

    # ------------------------------------------------------------------
    # [7] Generate figures
    # ------------------------------------------------------------------
    print("\n[7/8] Generating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Clean results for figures/JSON (remove y_true/y_pred)
    algo_results_clean = {}
    for name, data in algo_results.items():
        algo_results_clean[name] = {
            k: v for k, v in data.items() if k not in ("y_true", "y_pred")
        }

    fig_algorithm_comparison(algo_results_clean)
    fig_per_fold_variance(algo_results_clean)
    fig_per_site_accuracy(best_per_class, best_algo)
    fig_per_category_accuracy(best_per_category, best_algo)
    fig_dataset_stats(dataset_stats)

    if alexa_algo_results:
        fig_cross_domain_comparison(algo_results_clean, alexa_algo_results)

    # ------------------------------------------------------------------
    # [8] Save results
    # ------------------------------------------------------------------
    print("\n[8/8] Saving results...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "dataset": "Website Traffic (My Collection)",
            "num_traces": len(traces),
            "num_classes": len(labels),
            "traces_per_class": f"~{len(traces) // len(labels)}",
            "random_seed": 42,
            "cross_validation": "5-fold stratified",
            "capture_period": "March 1-3, 2026",
        },
        "algorithm_results": algo_results_clean,
        "dataset_stats": {
            k: v for k, v in dataset_stats.items()
            if k not in ("packet_counts", "durations")
        },
        "per_site_accuracy": {
            "algorithm": best_algo,
            "top_10": best_per_class["top_10"],
            "bottom_10": best_per_class["bottom_10"],
        },
        "per_category_accuracy": {
            "algorithm": best_algo,
            "categories": best_per_category,
        },
        "cross_domain_comparison": {},
    }

    # Add cross-domain comparison
    if alexa_algo_results:
        comparison = {}
        for algo in algo_results_clean:
            if algo in alexa_algo_results:
                web_acc = algo_results_clean[algo]["accuracy"]
                alexa_acc = alexa_algo_results[algo]["accuracy"]
                comparison[algo] = {
                    "website": web_acc,
                    "alexa": alexa_acc,
                    "difference": round(web_acc - alexa_acc, 1),
                }
        output["cross_domain_comparison"] = comparison

    out_path = RESULTS_DIR / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Results: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY — Website Traffic Fingerprinting")
    print("=" * 70)
    print(f"  Dataset: {len(traces)} traces, {len(labels)} sites, "
          f"~{len(traces) // len(labels)} traces/site")
    print()

    print(f"{'Algorithm':<20} {'Website':>10} {'Alexa':>10} {'Diff':>10} {'Time':>10}")
    print("-" * 60)
    for name in ["LL-Jaccard", "LL-NB", "VNG++", "AdaBoost",
                 "CUMUL+XGB", "Timing+XGB", "Combined+XGB", "Combined+RF"]:
        if name not in algo_results_clean:
            continue
        web = algo_results_clean[name]["accuracy"]
        alexa = alexa_algo_results.get(name, {}).get("accuracy", 0)
        diff = web - alexa if alexa else 0
        t = algo_results_clean[name]["time_sec"]
        alexa_str = f"{alexa:.1f}%" if alexa else "—"
        diff_str = f"{diff:+.1f}%" if alexa else "—"
        print(f"{name:<20} {web:>9.1f}% {alexa_str:>10} {diff_str:>10} {t:>9.1f}s")

    print()
    print(f"  BEST: {best_algo} ({algo_results[best_algo]['accuracy']}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
