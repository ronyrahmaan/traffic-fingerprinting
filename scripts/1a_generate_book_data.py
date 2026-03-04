"""Generate all data needed for the book HTML.

Runs all algorithms with detailed output: per-fold accuracy, confusion
matrices, bin-width sweeps, dataset statistics, and trace examples.

Saves everything to results/alexa_dataset/book_data.json.

Usage:
    cd traffic-fingerprinting
    uv run python scripts/1a_generate_book_data.py
"""

import json
import time
from datetime import datetime
from pathlib import Path

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
    build_llnb_matrix,
    build_llnb_matrix_with_rounding,
    build_svm_matrix,
    build_vngpp_matrix,
    compute_bursts,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "VCFingerprinting" / "data" / "trace_csv"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "alexa_dataset"

PAPER_RESULTS = {
    "LL-Jaccard": 17.4,
    "LL-NB": 33.8,
    "VNG++": 24.9,
    "AdaBoost": 33.4,
    "Random Guess": 1.0,
}


def compute_dataset_stats(
    traces: list, labels: list[str], label_to_id: dict[str, int],
) -> dict:
    """Compute per-command and overall dataset statistics."""
    per_command: dict[str, dict] = {}

    for label in labels:
        cmd_traces = [t for t in traces if t.label == label]
        pkt_counts = [t.num_packets for t in cmd_traces]
        durations = [t.duration for t in cmd_traces]
        total_bytes_list = [
            sum(p.size for p in t.packets) for t in cmd_traces
        ]
        per_command[label] = {
            "count": len(cmd_traces),
            "avg_packets": round(float(np.mean(pkt_counts)), 1),
            "min_packets": int(min(pkt_counts)),
            "max_packets": int(max(pkt_counts)),
            "avg_duration": round(float(np.mean(durations)), 3),
            "avg_total_bytes": round(float(np.mean(total_bytes_list)), 0),
        }

    all_pkts = [t.num_packets for t in traces]
    all_durs = [t.duration for t in traces]

    return {
        "per_command": per_command,
        "overall": {
            "num_traces": len(traces),
            "num_classes": len(labels),
            "traces_per_class": 10,
            "avg_packets": round(float(np.mean(all_pkts)), 1),
            "min_packets": int(min(all_pkts)),
            "max_packets": int(max(all_pkts)),
            "median_packets": int(np.median(all_pkts)),
            "avg_duration": round(float(np.mean(all_durs)), 3),
            "min_duration": round(float(min(all_durs)), 3),
            "max_duration": round(float(max(all_durs)), 3),
            "packet_count_histogram": np.histogram(
                all_pkts, bins=20,
            )[0].tolist(),
            "packet_count_bin_edges": np.histogram(
                all_pkts, bins=20,
            )[1].tolist(),
            "duration_histogram": np.histogram(
                all_durs, bins=20,
            )[0].tolist(),
            "duration_bin_edges": np.histogram(
                all_durs, bins=20,
            )[1].tolist(),
        },
    }


def extract_trace_example(traces: list) -> dict:
    """Get a real trace example for the timeline visualization."""
    # Pick a medium-length trace for illustration
    sorted_by_len = sorted(traces, key=lambda t: t.num_packets)
    # Pick one near the median
    mid = len(sorted_by_len) // 2
    trace = sorted_by_len[mid]

    packets = []
    for p in trace.packets[:50]:  # First 50 packets for the diagram
        packets.append({
            "time": round(p.time, 4),
            "size": p.size,
            "direction": p.direction,
            "signed_size": p.size * p.direction,
        })

    return {
        "label": trace.label,
        "filename": trace.filename,
        "num_packets": trace.num_packets,
        "duration": round(trace.duration, 3),
        "packets": packets,
    }


def compute_burst_stats(traces: list) -> dict:
    """Compute burst statistics across all traces."""
    all_burst_counts = []
    all_burst_sizes = []

    for t in traces:
        bursts = compute_bursts(t)
        all_burst_counts.append(len(bursts))
        all_burst_sizes.extend([abs(b) for b in bursts])

    return {
        "avg_bursts_per_trace": round(float(np.mean(all_burst_counts)), 1),
        "min_bursts": int(min(all_burst_counts)),
        "max_bursts": int(max(all_burst_counts)),
        "avg_burst_size": round(float(np.mean(all_burst_sizes)), 0),
        "median_burst_size": round(float(np.median(all_burst_sizes)), 0),
        "burst_count_histogram": np.histogram(
            all_burst_counts, bins=15,
        )[0].tolist(),
        "burst_count_bin_edges": np.histogram(
            all_burst_counts, bins=15,
        )[1].tolist(),
    }


def compute_per_class_accuracy(y_true: list, y_pred: list, id_to_label: dict) -> dict:
    """Compute accuracy for each class and return top/bottom 10."""
    classes = sorted(set(y_true))
    per_class = {}

    for cls in classes:
        mask = [i for i, y in enumerate(y_true) if y == cls]
        correct = sum(1 for i in mask if y_true[i] == y_pred[i])
        total = len(mask)
        label = id_to_label.get(cls, str(cls))
        per_class[label] = {
            "accuracy": round(correct / total * 100, 1) if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }

    # Sort by accuracy
    sorted_items = sorted(per_class.items(), key=lambda x: x[1]["accuracy"])
    bottom_10 = dict(sorted_items[:10])
    top_10 = dict(sorted_items[-10:])

    return {
        "all": per_class,
        "top_10": top_10,
        "bottom_10": bottom_10,
    }


def compute_confusion_summary(y_true: list, y_pred: list, id_to_label: dict) -> dict:
    """Compute confusion matrix and extract most confused pairs."""
    n_classes = len(set(y_true))
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # Find most confused pairs (off-diagonal)
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i][j] > 0:
                confused_pairs.append({
                    "true_label": id_to_label.get(i, str(i)),
                    "pred_label": id_to_label.get(j, str(j)),
                    "count": int(cm[i][j]),
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    return {
        "matrix": cm.tolist(),
        "labels": [id_to_label.get(i, str(i)) for i in range(n_classes)],
        "top_20_confused_pairs": confused_pairs[:20],
        "total_correct": int(np.trace(cm)),
        "total_predictions": int(cm.sum()),
    }


def run_bin_width_sweep(traces: list, y: np.ndarray) -> dict:
    """Sweep LL-NB bin width from 10 to 150 and record accuracy."""
    widths = list(range(10, 160, 10))
    results = []

    for w in widths:
        print(f"  Bin width {w}...", end=" ", flush=True)
        t0 = time.time()
        X = build_llnb_matrix_with_rounding(traces, w)
        acc = cross_validate_sklearn(X, y, "llnb") * 100
        elapsed = time.time() - t0
        print(f"{acc:.1f}% ({elapsed:.1f}s)")
        results.append({
            "bin_width": w,
            "accuracy": round(acc, 1),
            "n_features": X.shape[1],
            "time_sec": round(elapsed, 1),
        })

    best = max(results, key=lambda x: x["accuracy"])
    return {
        "sweep_results": results,
        "best_bin_width": best["bin_width"],
        "best_accuracy": best["accuracy"],
    }


def compute_feature_stats(X: np.ndarray, name: str) -> dict:
    """Compute feature matrix statistics."""
    nonzero_ratio = float(np.count_nonzero(X)) / X.size
    return {
        "name": name,
        "shape": list(X.shape),
        "mean": round(float(np.mean(X)), 2),
        "std": round(float(np.std(X)), 2),
        "min": round(float(np.min(X)), 2),
        "max": round(float(np.max(X)), 2),
        "sparsity": round(1.0 - nonzero_ratio, 4),
        "nonzero_ratio": round(nonzero_ratio, 4),
    }


def main() -> None:
    """Generate all book data."""
    print("=" * 60)
    print("Generating Book Data")
    print("=" * 60)

    # Load data
    print("\nLoading traces...")
    traces = load_dataset(DATA_DIR)
    labels, label_to_id = get_labels_and_ids(traces)
    id_to_label = {v: k for k, v in label_to_id.items()}
    traces, y = traces_to_arrays(traces, label_to_id)

    book_data: dict = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "dataset": "Alexa Voice Commands (Kennedy et al., IEEE CNS 2019)",
            "paper_results": PAPER_RESULTS,
        },
    }

    # 1. Dataset statistics
    print("\n--- Dataset Statistics ---")
    book_data["dataset_stats"] = compute_dataset_stats(traces, labels, label_to_id)
    print(f"  {len(traces)} traces, {len(labels)} classes")

    # 2. Trace example
    print("\n--- Trace Example ---")
    book_data["trace_example"] = extract_trace_example(traces)
    print(f"  Selected: {book_data['trace_example']['label']} ({book_data['trace_example']['num_packets']} packets)")

    # 3. Burst statistics
    print("\n--- Burst Statistics ---")
    book_data["burst_stats"] = compute_burst_stats(traces)
    print(f"  Avg {book_data['burst_stats']['avg_bursts_per_trace']} bursts/trace")

    # 4. Build feature matrices
    print("\n--- Feature Matrices ---")
    X_llnb = build_llnb_matrix(traces)
    X_vng = build_vngpp_matrix(traces)
    X_ada = build_svm_matrix(traces)
    book_data["feature_stats"] = {
        "llnb": compute_feature_stats(X_llnb, "LL-NB"),
        "vngpp": compute_feature_stats(X_vng, "VNG++"),
        "adaboost": compute_feature_stats(X_ada, "AdaBoost"),
    }
    print(f"  LL-NB: {X_llnb.shape}, VNG++: {X_vng.shape}, AdaBoost: {X_ada.shape}")

    # 5. Detailed algorithm results
    print("\n--- LL-Jaccard (detailed) ---")
    t0 = time.time()
    jaccard_detail = cross_validate_jaccard_detailed(traces, y)
    jaccard_time = time.time() - t0
    print(f"  Mean: {jaccard_detail['mean_accuracy']*100:.1f}%, folds: {[round(a*100,1) for a in jaccard_detail['fold_accuracies']]}")

    print("\n--- LL-NB (detailed) ---")
    t0 = time.time()
    llnb_detail = cross_validate_sklearn_detailed(X_llnb, y, "llnb")
    llnb_time = time.time() - t0
    print(f"  Mean: {llnb_detail['mean_accuracy']*100:.1f}%, folds: {[round(a*100,1) for a in llnb_detail['fold_accuracies']]}")

    print("\n--- VNG++ (detailed) ---")
    t0 = time.time()
    vng_detail = cross_validate_sklearn_detailed(X_vng, y, "vngpp")
    vng_time = time.time() - t0
    print(f"  Mean: {vng_detail['mean_accuracy']*100:.1f}%, folds: {[round(a*100,1) for a in vng_detail['fold_accuracies']]}")

    print("\n--- AdaBoost (detailed) ---")
    t0 = time.time()
    ada_detail = cross_validate_sklearn_detailed(X_ada, y, "adaboost")
    ada_time = time.time() - t0
    print(f"  Mean: {ada_detail['mean_accuracy']*100:.1f}%, folds: {[round(a*100,1) for a in ada_detail['fold_accuracies']]}")

    book_data["algorithm_results"] = {
        "LL-Jaccard": {
            "accuracy": round(jaccard_detail["mean_accuracy"] * 100, 1),
            "fold_accuracies": [round(a * 100, 1) for a in jaccard_detail["fold_accuracies"]],
            "time_sec": round(jaccard_time, 1),
        },
        "LL-NB": {
            "accuracy": round(llnb_detail["mean_accuracy"] * 100, 1),
            "fold_accuracies": [round(a * 100, 1) for a in llnb_detail["fold_accuracies"]],
            "time_sec": round(llnb_time, 1),
            "features": X_llnb.shape[1],
        },
        "VNG++": {
            "accuracy": round(vng_detail["mean_accuracy"] * 100, 1),
            "fold_accuracies": [round(a * 100, 1) for a in vng_detail["fold_accuracies"]],
            "time_sec": round(vng_time, 1),
            "features": X_vng.shape[1],
        },
        "AdaBoost": {
            "accuracy": round(ada_detail["mean_accuracy"] * 100, 1),
            "fold_accuracies": [round(a * 100, 1) for a in ada_detail["fold_accuracies"]],
            "time_sec": round(ada_time, 1),
            "features": X_ada.shape[1],
        },
    }

    # 6. Per-class accuracy (LL-NB — best algorithm)
    print("\n--- Per-Class Accuracy (LL-NB) ---")
    book_data["per_class_accuracy"] = compute_per_class_accuracy(
        llnb_detail["y_true"], llnb_detail["y_pred"], id_to_label,
    )
    top = list(book_data["per_class_accuracy"]["top_10"].items())
    bottom = list(book_data["per_class_accuracy"]["bottom_10"].items())
    print(f"  Easiest: {top[-1][0]} ({top[-1][1]['accuracy']}%)")
    print(f"  Hardest: {bottom[0][0]} ({bottom[0][1]['accuracy']}%)")

    # 7. Confusion matrix (LL-NB)
    print("\n--- Confusion Matrix (LL-NB) ---")
    book_data["confusion_matrix"] = compute_confusion_summary(
        llnb_detail["y_true"], llnb_detail["y_pred"], id_to_label,
    )
    top_confused = book_data["confusion_matrix"]["top_20_confused_pairs"][:3]
    for pair in top_confused:
        print(f"  {pair['true_label']} → {pair['pred_label']}: {pair['count']} times")

    # 8. Bin-width sweep
    print("\n--- Bin-Width Sweep (LL-NB) ---")
    book_data["bin_width_sweep"] = run_bin_width_sweep(traces, y)
    print(f"  Best: bin={book_data['bin_width_sweep']['best_bin_width']}, acc={book_data['bin_width_sweep']['best_accuracy']}%")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "book_data.json"
    with open(out_path, "w") as f:
        json.dump(book_data, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
