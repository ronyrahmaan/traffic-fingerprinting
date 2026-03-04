"""Run all four attack algorithms on the Alexa dataset.

This is the main script. It:
1. Loads all 1,000 CSV traces
2. Extracts features for each algorithm
3. Runs 5-fold cross validation
4. Prints accuracy for each algorithm
5. Compares against the paper's Table I

Usage:
    cd traffic-fingerprinting
    uv run python scripts/1a_run_evaluation.py
"""

import time
from pathlib import Path

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

# Path to the dataset
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "VCFingerprinting" / "data" / "trace_csv"

# Paper's results (Table I) for comparison
PAPER_RESULTS = {
    "LL-Jaccard": 17.4,
    "LL-NB": 33.8,
    "VNG++": 24.9,
    "AdaBoost": 33.4,
    "Random Guess": 1.0,
}


def main() -> None:
    """Run the full evaluation pipeline."""
    print("=" * 60)
    print("Voice Command Fingerprinting — Reproduction")
    print("Paper: Kennedy et al., IEEE CNS 2019")
    print("=" * 60)

    # --- Load data ---
    print(f"\nLoading traces from {DATA_DIR}...")
    t0 = time.time()
    traces = load_dataset(DATA_DIR)
    print(f"Loaded {len(traces)} traces in {time.time() - t0:.1f}s")

    labels, label_to_id = get_labels_and_ids(traces)
    print(f"Found {len(labels)} unique voice commands")
    traces, y = traces_to_arrays(traces, label_to_id)

    # Quick data summary
    sizes = [t.num_packets for t in traces]
    print(f"Packets per trace: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.0f}")

    # --- Algorithm 1: LL-Jaccard ---
    print("\n--- LL-Jaccard ---")
    t0 = time.time()
    acc_jaccard = cross_validate_jaccard(traces, y) * 100
    print(f"Accuracy: {acc_jaccard:.1f}% (paper: {PAPER_RESULTS['LL-Jaccard']}%)")
    print(f"Time: {time.time() - t0:.1f}s")

    # --- Algorithm 2: LL-NB ---
    print("\n--- LL-NB (Naive Bayes) ---")
    t0 = time.time()
    X_llnb = build_llnb_matrix(traces)
    print(f"Feature matrix shape: {X_llnb.shape}")
    acc_llnb = cross_validate_sklearn(X_llnb, y, "llnb") * 100
    print(f"Accuracy: {acc_llnb:.1f}% (paper: {PAPER_RESULTS['LL-NB']}%)")
    print(f"Time: {time.time() - t0:.1f}s")

    # --- Algorithm 3: VNG++ ---
    print("\n--- VNG++ ---")
    t0 = time.time()
    X_vng = build_vngpp_matrix(traces)
    print(f"Feature matrix shape: {X_vng.shape}")
    acc_vng = cross_validate_sklearn(X_vng, y, "vngpp") * 100
    print(f"Accuracy: {acc_vng:.1f}% (paper: {PAPER_RESULTS['VNG++']}%)")
    print(f"Time: {time.time() - t0:.1f}s")

    # --- Algorithm 4: AdaBoost ---
    print("\n--- AdaBoost (Panchenko features + boosted trees) ---")
    t0 = time.time()
    X_ada = build_svm_matrix(traces)
    print(f"Feature matrix shape: {X_ada.shape}")
    acc_ada = cross_validate_sklearn(X_ada, y, "adaboost") * 100
    print(f"Accuracy: {acc_ada:.1f}% (paper: {PAPER_RESULTS['AdaBoost']}%)")
    print(f"Time: {time.time() - t0:.1f}s")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Algorithm':<20} {'Ours':>8} {'Paper':>8} {'Diff':>8}")
    print("-" * 44)

    results = [
        ("LL-Jaccard", acc_jaccard, PAPER_RESULTS["LL-Jaccard"]),
        ("LL-NB", acc_llnb, PAPER_RESULTS["LL-NB"]),
        ("VNG++", acc_vng, PAPER_RESULTS["VNG++"]),
        ("AdaBoost", acc_ada, PAPER_RESULTS["AdaBoost"]),
    ]

    for name, acc, paper in results:
        diff = acc - paper
        sign = "+" if diff >= 0 else ""
        print(f"{name:<20} {acc:>7.1f}% {paper:>7.1f}% {sign}{diff:>6.1f}%")

    print(f"{'Random Guess':<20} {'1.0%':>8} {'1.0%':>8}")
    print()


if __name__ == "__main__":
    main()
