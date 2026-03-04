"""Expanded evaluation: paper algorithms + my improvements.

Runs all algorithms on the Alexa dataset and generates detailed
results including per-fold accuracy, confusion matrices, bin-width
sweeps, per-class analysis, and dataset statistics.

Saves everything to results/alexa_dataset/expanded_results.json
and generates publication-quality figures to figures/.

Usage:
    cd traffic-fingerprinting
    uv run python scripts/1a_run_expanded_evaluation.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from traffic_fingerprinting.classifiers import (
    cross_validate_jaccard_detailed,
    cross_validate_sklearn,
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
    build_llnb_matrix_with_rounding,
    build_svm_matrix,
    build_timing_matrix,
    build_vngpp_matrix,
    compute_bursts,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "VCFingerprinting" / "data" / "trace_csv"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "alexa_dataset"
FIGURES_DIR = Path(__file__).parent.parent / "figures" / "alexa"

PAPER_RESULTS = {
    "LL-Jaccard": 17.4,
    "LL-NB": 33.8,
    "VNG++": 24.9,
    "AdaBoost": 33.4,
}


def compute_per_class_accuracy(y_true: list, y_pred: list, id_to_label: dict) -> dict:
    """Compute accuracy per class, return all + top/bottom 10."""
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


def compute_confusion_top_pairs(y_true: list, y_pred: list, id_to_label: dict) -> list:
    """Find the most confused class pairs."""
    n_classes = max(max(y_true), max(y_pred)) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i][j] > 0:
                pairs.append({
                    "true": id_to_label.get(i, str(i)),
                    "predicted": id_to_label.get(j, str(j)),
                    "count": int(cm[i][j]),
                })
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:20]


def run_bin_width_sweep(traces, y) -> list:
    """Sweep LL-NB bin width and record accuracy."""
    results = []
    for w in range(10, 160, 10):
        X = build_llnb_matrix_with_rounding(traces, w)
        acc = cross_validate_sklearn(X, y, "llnb") * 100
        results.append({"bin_width": w, "accuracy": round(acc, 1), "features": X.shape[1]})
        print(f"    bin={w:>3}: {acc:.1f}% ({X.shape[1]} features)")
    return results


def compute_dataset_stats(traces, labels) -> dict:
    """Compute overall and per-command dataset statistics."""
    all_pkts = [t.num_packets for t in traces]
    all_durs = [t.duration for t in traces]
    all_bytes = [sum(p.size for p in t.packets) for t in traces]

    per_cmd = {}
    for label in labels:
        cmd_traces = [t for t in traces if t.label == label]
        pkts = [t.num_packets for t in cmd_traces]
        durs = [t.duration for t in cmd_traces]
        per_cmd[label] = {
            "avg_packets": round(float(np.mean(pkts)), 1),
            "avg_duration": round(float(np.mean(durs)), 3),
        }

    # Burst stats
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
        "per_command": per_cmd,
        "packet_counts": all_pkts,
        "durations": [round(d, 4) for d in all_durs],
    }


def get_trace_example(traces) -> dict:
    """Get a median-length trace for visualization."""
    sorted_t = sorted(traces, key=lambda t: t.num_packets)
    trace = sorted_t[len(sorted_t) // 2]
    return {
        "label": trace.label,
        "filename": trace.filename,
        "num_packets": trace.num_packets,
        "duration": round(trace.duration, 3),
        "packets": [
            {"time": round(p.time, 4), "size": p.size, "direction": p.direction}
            for p in trace.packets[:60]
        ],
    }


# =====================================================================
# FIGURE GENERATION
# =====================================================================

def fig_comparison_bar(algo_results: dict, paper_results: dict) -> None:
    """Bar chart: paper vs mine vs improvements."""
    # Paper algorithms
    paper_algos = ["LL-Jaccard", "LL-NB", "VNG++", "AdaBoost"]
    # My improvements
    improvement_algos = [a for a in algo_results if a not in paper_algos]

    all_algos = paper_algos + improvement_algos
    mine = [algo_results[a]["accuracy"] for a in all_algos]
    paper = [paper_results.get(a, 0) for a in all_algos]

    x = np.arange(len(all_algos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))

    # Paper bars (only for paper algorithms)
    paper_x = x[:len(paper_algos)]
    bars1 = ax.bar(paper_x - width / 2, paper[:len(paper_algos)], width,
                   label="Paper (Kennedy et al.)", color="#4C72B0", edgecolor="black", linewidth=0.5)

    # My bars
    colors = ["#DD8452"] * len(paper_algos) + ["#55A868"] * len(improvement_algos)
    bars2 = ax.bar(x[:len(paper_algos)] + width / 2, mine[:len(paper_algos)], width,
                   label="My Reproduction", color="#DD8452", edgecolor="black", linewidth=0.5)

    # Improvement bars (standalone, no paper comparison)
    if improvement_algos:
        bars3 = ax.bar(x[len(paper_algos):], mine[len(paper_algos):], width * 1.5,
                       label="My Improvements", color="#55A868", edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars3, mine[len(paper_algos):]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%",
                ha="center", va="bottom", fontsize=8)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(len(all_algos) - 0.5, 1.8, "Random Guess (1%)", fontsize=8, color="gray")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Voice Command Fingerprinting: Paper vs Reproduction vs Improvements",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(all_algos, fontsize=10, rotation=15, ha="right")
    ax.set_ylim(0, max(mine) + 8)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "comparison_all.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: comparison_all.png")


def fig_per_fold_variance(algo_results: dict) -> None:
    """Strip/box plot showing per-fold variance."""
    fig, ax = plt.subplots(figsize=(10, 5))

    algos = list(algo_results.keys())
    positions = range(len(algos))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]

    for i, algo in enumerate(algos):
        folds = algo_results[algo]["fold_accuracies"]
        # Box
        bp = ax.boxplot([folds], positions=[i], widths=0.4,
                       patch_artist=True, showmeans=True,
                       meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 6})
        bp["boxes"][0].set_facecolor(colors[i % len(colors)])
        bp["boxes"][0].set_alpha(0.7)
        # Scatter individual points
        ax.scatter([i] * len(folds), folds, color=colors[i % len(colors)],
                  zorder=5, s=40, edgecolor="black", linewidth=0.5)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(algos, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Accuracy per Fold (%)", fontsize=12)
    ax.set_title("Per-Fold Accuracy Variance (5-Fold CV)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "per_fold_variance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: per_fold_variance.png")


def fig_bin_width_sweep(sweep_results: list) -> None:
    """Line chart of LL-NB accuracy vs bin width."""
    widths = [r["bin_width"] for r in sweep_results]
    accs = [r["accuracy"] for r in sweep_results]
    feats = [r["features"] for r in sweep_results]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(widths, accs, "o-", color="#4C72B0", linewidth=2, markersize=7, label="Accuracy")
    best_idx = np.argmax(accs)
    ax1.scatter([widths[best_idx]], [accs[best_idx]], color="red", s=120,
               zorder=5, edgecolor="black", linewidth=1.5, label=f"Best: {accs[best_idx]}% (bin={widths[best_idx]})")
    ax1.axhline(y=33.8, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.text(widths[-1], 34.2, "Paper: 33.8%", fontsize=9, color="gray", ha="right")

    ax1.set_xlabel("Bin Width", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")

    # Secondary axis: feature count
    ax2 = ax1.twinx()
    ax2.plot(widths, feats, "s--", color="#DD8452", linewidth=1, markersize=5, alpha=0.7, label="Features")
    ax2.set_ylabel("Number of Features", fontsize=12, color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")

    ax1.set_title("LL-NB: Bin Width Sweep", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "bin_width_sweep.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: bin_width_sweep.png")


def fig_per_class_accuracy(per_class: dict) -> None:
    """Horizontal bar chart: top 10 and bottom 10 classes."""
    top_10 = per_class["top_10"]
    bottom_10 = per_class["bottom_10"]

    # Combine: bottom first, then top
    labels = list(bottom_10.keys()) + ["..."] + list(top_10.keys())
    values = list(bottom_10.values()) + [0] + list(top_10.values())

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#C44E52"] * len(bottom_10) + ["white"] + ["#55A868"] * len(top_10)
    y_pos = range(len(labels))

    bars = ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                   f"{val:.0f}%", va="center", fontsize=9)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("LL-NB Per-Class Accuracy (Top 10 + Bottom 10)", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: per_class_accuracy.png")


def fig_dataset_stats(stats: dict) -> None:
    """Histograms of packet counts and durations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Packet counts
    ax1.hist(stats["packet_counts"], bins=25, color="#4C72B0", edgecolor="black", linewidth=0.5)
    ax1.axvline(x=stats["packets"]["mean"], color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {stats['packets']['mean']}")
    ax1.set_xlabel("Packets per Trace", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Distribution of Trace Lengths", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Durations
    ax2.hist(stats["durations"], bins=25, color="#55A868", edgecolor="black", linewidth=0.5)
    ax2.axvline(x=stats["duration"]["mean"], color="red", linestyle="--", linewidth=1.5,
               label=f"Mean: {stats['duration']['mean']}s")
    ax2.set_xlabel("Duration (seconds)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of Trace Durations", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_stats.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: dataset_stats.png")


def fig_trace_timeline(trace_example: dict) -> None:
    """Plot a single trace as packets over time."""
    packets = trace_example["packets"]
    times = [p["time"] for p in packets]
    sizes = [p["size"] * p["direction"] for p in packets]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Color by direction
    colors = ["#4C72B0" if s > 0 else "#C44E52" for s in sizes]
    ax.bar(times, sizes, width=0.01, color=colors, edgecolor="none")

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Signed Packet Size (bytes)", fontsize=11)
    ax.set_title(f"Trace Timeline: \"{trace_example['label']}\" ({trace_example['num_packets']} packets)",
                fontsize=12, fontweight="bold")

    # Legend
    ax.bar([], [], color="#4C72B0", label="Outgoing (device → cloud)")
    ax.bar([], [], color="#C44E52", label="Incoming (cloud → device)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "trace_timeline.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: trace_timeline.png")


def fig_confusion_heatmap(y_true: list, y_pred: list, id_to_label: dict) -> None:
    """Confusion matrix heatmap for LL-NB (best paper algorithm)."""
    n_classes = max(max(y_true), max(y_pred)) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title("Confusion Matrix — LL-NB (100 classes)", fontsize=13, fontweight="bold")

    # Only label every 10th tick to avoid clutter
    ticks = list(range(0, n_classes, 10))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, fontsize=8)
    ax.set_yticklabels(ticks, fontsize=8)

    plt.colorbar(im, ax=ax, label="Count")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_llnb.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: confusion_matrix_llnb.png")


# =====================================================================
# MAIN
# =====================================================================

def main() -> None:
    """Run full expanded evaluation."""
    print("=" * 65)
    print("  EXPANDED EVALUATION — Paper Algorithms + My Improvements")
    print("=" * 65)

    # Load data
    print("\n[1/8] Loading dataset...")
    traces = load_dataset(DATA_DIR)
    labels, label_to_id = get_labels_and_ids(traces)
    id_to_label = {v: k for k, v in label_to_id.items()}
    traces, y = traces_to_arrays(traces, label_to_id)
    print(f"  {len(traces)} traces, {len(labels)} classes")

    # Dataset statistics
    print("\n[2/8] Computing dataset statistics...")
    dataset_stats = compute_dataset_stats(traces, labels)
    trace_example = get_trace_example(traces)
    print(f"  Avg {dataset_stats['packets']['mean']} packets/trace, "
          f"{dataset_stats['duration']['mean']}s duration")

    # Build ALL feature matrices
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

    # Run all algorithms with detailed output
    print("\n[4/8] Running algorithms (detailed CV)...")
    algo_results = {}

    # LL-Jaccard (special — no feature matrix)
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

    # Bin-width sweep
    print("\n[5/8] Running bin-width sweep (LL-NB)...")
    sweep = run_bin_width_sweep(traces, y)
    best_sweep = max(sweep, key=lambda x: x["accuracy"])
    print(f"  Best: bin={best_sweep['bin_width']}, acc={best_sweep['accuracy']}%")

    # Per-class accuracy (best paper algo = LL-NB, best overall = check)
    print("\n[6/8] Computing per-class analysis...")
    best_algo = max(algo_results, key=lambda k: algo_results[k]["accuracy"])
    llnb_per_class = compute_per_class_accuracy(
        algo_results["LL-NB"]["y_true"], algo_results["LL-NB"]["y_pred"], id_to_label,
    )
    best_per_class = compute_per_class_accuracy(
        algo_results[best_algo]["y_true"], algo_results[best_algo]["y_pred"], id_to_label,
    )
    confusion_pairs = compute_confusion_top_pairs(
        algo_results["LL-NB"]["y_true"], algo_results["LL-NB"]["y_pred"], id_to_label,
    )
    print(f"  Best overall algorithm: {best_algo} ({algo_results[best_algo]['accuracy']}%)")

    # Generate figures
    print("\n[7/8] Generating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Clean y_true/y_pred from results for JSON (too large)
    algo_results_clean = {}
    for name, data in algo_results.items():
        algo_results_clean[name] = {k: v for k, v in data.items() if k not in ("y_true", "y_pred")}

    fig_comparison_bar(algo_results_clean, PAPER_RESULTS)
    fig_per_fold_variance(algo_results_clean)
    fig_bin_width_sweep(sweep)
    fig_per_class_accuracy(llnb_per_class)
    fig_dataset_stats(dataset_stats)
    fig_trace_timeline(trace_example)
    fig_confusion_heatmap(
        algo_results["LL-NB"]["y_true"], algo_results["LL-NB"]["y_pred"], id_to_label,
    )

    # Save JSON results
    print("\n[8/8] Saving results...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "dataset": "Alexa Voice Commands (Kennedy et al., IEEE CNS 2019)",
            "num_traces": len(traces),
            "num_classes": len(labels),
            "random_seed": 42,
            "cross_validation": "5-fold stratified",
        },
        "paper_results": PAPER_RESULTS,
        "algorithm_results": algo_results_clean,
        "dataset_stats": {k: v for k, v in dataset_stats.items()
                         if k not in ("packet_counts", "durations", "per_command")},
        "trace_example": trace_example,
        "bin_width_sweep": sweep,
        "per_class_accuracy_llnb": llnb_per_class,
        "per_class_accuracy_best": {"algorithm": best_algo, **best_per_class},
        "confusion_top_pairs_llnb": confusion_pairs,
    }

    out_path = RESULTS_DIR / "expanded_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

    # Final summary
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'Algorithm':<20} {'Accuracy':>8} {'Paper':>8} {'Diff':>8} {'Features':>8}")
    print("-" * 52)
    for name in ["LL-Jaccard", "LL-NB", "VNG++", "AdaBoost"]:
        acc = algo_results_clean[name]["accuracy"]
        paper = PAPER_RESULTS[name]
        diff = acc - paper
        feats = algo_results_clean[name].get("features", "—")
        print(f"{name:<20} {acc:>7.1f}% {paper:>7.1f}% {diff:>+7.1f}% {feats:>8}")

    print("-" * 52)
    print("MY IMPROVEMENTS:")
    for name in algo_results_clean:
        if name not in ["LL-Jaccard", "LL-NB", "VNG++", "AdaBoost"]:
            acc = algo_results_clean[name]["accuracy"]
            feats = algo_results_clean[name].get("features", "—")
            print(f"{name:<20} {acc:>7.1f}% {'—':>8} {'':>8} {feats:>8}")

    print(f"\nBest overall: {best_algo} ({algo_results[best_algo]['accuracy']}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
