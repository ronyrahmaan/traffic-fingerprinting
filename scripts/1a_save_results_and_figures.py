"""Save evaluation results as JSON and generate comparison figures.

This script runs the full pipeline, saves results with metadata,
and creates publication-quality comparison charts.

Usage:
    cd traffic-fingerprinting
    uv run python scripts/1a_save_results_and_figures.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from traffic_fingerprinting.classifiers import (
    cross_validate_jaccard,
    cross_validate_sklearn,
)
from traffic_fingerprinting.data_loader import (
    get_labels_and_ids,
    load_dataset,
    traces_to_arrays,
)
from traffic_fingerprinting.features import (
    build_llnb_matrix,
    build_svm_matrix,
    build_vngpp_matrix,
)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "VCFingerprinting" / "data" / "trace_csv"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "alexa_dataset"
FIGURES_DIR = Path(__file__).parent.parent / "figures" / "alexa"

# Paper's Table I
PAPER_RESULTS = {
    "LL-Jaccard": 17.4,
    "LL-NB": 33.8,
    "VNG++": 24.9,
    "AdaBoost": 33.4,
    "Random Guess": 1.0,
}


def run_evaluation() -> dict:
    """Run full evaluation and return results dict."""
    print("Loading data...")
    traces = load_dataset(DATA_DIR)
    labels, label_to_id = get_labels_and_ids(traces)
    traces, y = traces_to_arrays(traces, label_to_id)

    results = {}

    # LL-Jaccard
    print("Running LL-Jaccard...")
    t0 = time.time()
    acc = cross_validate_jaccard(traces, y) * 100
    results["LL-Jaccard"] = {"accuracy": round(acc, 1), "time_sec": round(time.time() - t0, 1)}

    # LL-NB
    print("Running LL-NB...")
    t0 = time.time()
    X = build_llnb_matrix(traces)
    acc = cross_validate_sklearn(X, y, "llnb") * 100
    results["LL-NB"] = {"accuracy": round(acc, 1), "time_sec": round(time.time() - t0, 1), "features": X.shape[1]}

    # VNG++
    print("Running VNG++...")
    t0 = time.time()
    X = build_vngpp_matrix(traces)
    acc = cross_validate_sklearn(X, y, "vngpp") * 100
    results["VNG++"] = {"accuracy": round(acc, 1), "time_sec": round(time.time() - t0, 1), "features": X.shape[1]}

    # AdaBoost
    print("Running AdaBoost...")
    t0 = time.time()
    X = build_svm_matrix(traces)
    acc = cross_validate_sklearn(X, y, "adaboost") * 100
    results["AdaBoost"] = {"accuracy": round(acc, 1), "time_sec": round(time.time() - t0, 1), "features": X.shape[1]}

    return results


def save_results(results: dict) -> None:
    """Save results as JSON with metadata."""
    output = {
        "dataset": "Alexa Voice Commands (Kennedy et al., IEEE CNS 2019)",
        "num_traces": 1000,
        "num_classes": 100,
        "traces_per_class": 10,
        "cross_validation": "5-fold stratified",
        "random_seed": 42,
        "timestamp": datetime.now().isoformat(),
        "sklearn_note": "AdaBoost uses HistGradientBoosting (SAMME.R removed in sklearn 1.6+)",
        "results": results,
        "paper_results": PAPER_RESULTS,
    }

    out_path = RESULTS_DIR / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


def make_comparison_figure(results: dict) -> None:
    """Bar chart comparing my results vs paper's Table I."""
    algorithms = ["LL-Jaccard", "LL-NB", "VNG++", "AdaBoost"]
    mine = [results[a]["accuracy"] for a in algorithms]
    paper = [PAPER_RESULTS[a] for a in algorithms]

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, paper, width, label="Paper (Table I)", color="#4C72B0", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, mine, width, label="My Reproduction", color="#DD8452", edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    # Random guess baseline
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(len(algorithms) - 0.5, 1.5, "Random Guess (1%)", fontsize=8, color="gray")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Voice Command Fingerprinting: Paper vs My Reproduction", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=11)
    ax.set_ylim(0, 42)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "alexa_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.close()


def make_feature_figure(results: dict) -> None:
    """Show feature dimensions and accuracy relationship."""
    # Only algorithms with sklearn features
    algos = ["LL-NB", "VNG++", "AdaBoost"]
    features = [results[a]["features"] for a in algos]
    accs = [results[a]["accuracy"] for a in algos]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4C72B0", "#55A868", "#DD8452"]
    bars = ax.bar(algos, accs, color=colors, edgecolor="black", linewidth=0.5, width=0.5)

    # Label each bar with feature count
    for bar, feat, acc in zip(bars, features, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.8, f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, acc / 2, f"{feat} features", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Feature Dimensions vs Classification Accuracy", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 42)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = FIGURES_DIR / "feature_accuracy.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    plt.close()


def main() -> None:
    """Run evaluation, save results, generate figures."""
    results = run_evaluation()
    save_results(results)
    make_comparison_figure(results)
    make_feature_figure(results)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for algo in ["LL-Jaccard", "LL-NB", "VNG++", "AdaBoost"]:
        acc = results[algo]["accuracy"]
        paper = PAPER_RESULTS[algo]
        diff = acc - paper
        sign = "+" if diff >= 0 else ""
        print(f"{algo:<12} Mine: {acc:>5.1f}%  Paper: {paper:>5.1f}%  Diff: {sign}{diff:.1f}%")


if __name__ == "__main__":
    main()
