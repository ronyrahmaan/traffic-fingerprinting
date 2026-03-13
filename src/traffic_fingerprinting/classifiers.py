"""Classifiers for attack algorithms.

Paper algorithms (reproduce Kennedy et al., IEEE CNS 2019):
1. LL-Jaccard: custom Jaccard similarity (no sklearn needed)
2. LL-NB: GaussianNB on packet histograms
3. VNG++: GaussianNB on burst features
4. P-SVM/AdaBoost: HistGradientBoosting on rich burst + stats features

My additions:
5. XGBoost: gradient-boosted trees with better regularization
6. Random Forest: ensemble of decision trees (bagging, not boosting)

All are evaluated with 5-fold stratified cross validation.
"""

import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from traffic_fingerprinting.data_loader import TrafficTrace
from traffic_fingerprinting.features import extract_jaccard_set


# --- Algorithm 1: LL-Jaccard (custom classifier) ---

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    J = |intersection| / |union|

    Returns 0 if both sets are empty, otherwise a value between 0 and 1.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def train_jaccard_profiles(
    traces: list[TrafficTrace],
    y: np.ndarray,
) -> dict[int, set[int]]:
    """Build a representative set for each class (majority vote filtering).

    For each class, take all training traces, find the union of their
    packet sets, then keep only elements that appear in at least half
    of the traces. This filters out noise.
    """
    classes = np.unique(y)
    profiles: dict[int, set[int]] = {}

    for cls in classes:
        # Get all traces for this class
        class_traces = [traces[i] for i in range(len(traces)) if y[i] == cls]
        sets = [extract_jaccard_set(t) for t in class_traces]

        if not sets:
            profiles[cls] = set()
            continue

        # Union of all sets
        all_elements = set()
        for s in sets:
            all_elements |= s

        # Keep elements appearing in >= half of traces
        # Paper uses math.ceil(N/2): for 8 traces, threshold=4 (need 4+ votes)
        threshold = math.ceil(len(sets) / 2)
        filtered = set()
        for elem in all_elements:
            count = sum(1 for s in sets if elem in s)
            if count >= threshold:
                filtered.add(elem)

        profiles[cls] = filtered

    return profiles


def predict_jaccard(
    test_traces: list[TrafficTrace],
    profiles: dict[int, set[int]],
) -> np.ndarray:
    """Predict class for each test trace using Jaccard similarity.

    For each test trace, compute Jaccard similarity against every
    class profile. The class with the highest similarity wins.
    """
    predictions = []
    classes = sorted(profiles.keys())

    for trace in test_traces:
        test_set = extract_jaccard_set(trace)
        best_cls = classes[0]
        best_sim = -1.0

        for cls in classes:
            sim = jaccard_similarity(test_set, profiles[cls])
            if sim > best_sim:
                best_sim = sim
                best_cls = cls

        predictions.append(best_cls)

    return np.array(predictions)


# --- Classifier Factory ---

def _make_classifier(name: str):
    """Create a classifier instance by name.

    Paper classifiers:
        llnb, vngpp  → GaussianNB (fast, no hyperparameters)
        adaboost     → HistGradientBoosting (replaces SAMME.R removed in sklearn 1.6+)

    My additions:
        xgboost      → XGBClassifier (better regularization, handles sparse features)
        random_forest → RandomForestClassifier (bagging ensemble, robust baseline)
    """
    if name in ("llnb", "vngpp"):
        return GaussianNB()
    if name == "adaboost":
        return HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.1, random_state=42,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier: {name}")


# --- Cross Validation ---

def cross_validate_jaccard(
    traces: list[TrafficTrace],
    y: np.ndarray,
    n_folds: int = 5,
) -> float:
    """Run 5-fold cross validation for LL-Jaccard.

    Returns average accuracy across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, test_idx in skf.split(np.zeros(len(traces)), y):
        train_traces = [traces[i] for i in train_idx]
        train_y = y[train_idx]
        test_traces = [traces[i] for i in test_idx]
        test_y = y[test_idx]

        # Train: build class profiles
        profiles = train_jaccard_profiles(train_traces, train_y)

        # Predict
        pred_y = predict_jaccard(test_traces, profiles)

        # Accuracy
        acc = np.mean(pred_y == test_y)
        accuracies.append(acc)

    return float(np.mean(accuracies))


def cross_validate_jaccard_detailed(
    traces: list[TrafficTrace],
    y: np.ndarray,
    n_folds: int = 5,
) -> dict:
    """Run 5-fold CV for LL-Jaccard, returning per-fold details.

    Returns dict with:
        fold_accuracies: list of per-fold accuracy floats
        mean_accuracy: overall mean
        y_true: concatenated true labels across all folds
        y_pred: concatenated predictions across all folds
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []
    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in skf.split(np.zeros(len(traces)), y):
        train_traces = [traces[i] for i in train_idx]
        train_y = y[train_idx]
        test_traces = [traces[i] for i in test_idx]
        test_y = y[test_idx]

        profiles = train_jaccard_profiles(train_traces, train_y)
        pred_y = predict_jaccard(test_traces, profiles)

        fold_accs.append(float(np.mean(pred_y == test_y)))
        all_y_true.extend(test_y.tolist())
        all_y_pred.extend(pred_y.tolist())

    return {
        "fold_accuracies": fold_accs,
        "mean_accuracy": float(np.mean(fold_accs)),
        "y_true": all_y_true,
        "y_pred": all_y_pred,
    }


def cross_validate_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    n_folds: int = 5,
) -> float:
    """Run 5-fold cross validation for sklearn-based classifiers.

    classifier_name: one of "llnb", "vngpp", "adaboost", "xgboost", "random_forest"

    Returns average accuracy across folds.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = _make_classifier(classifier_name)
        clf.fit(X_train, y_train)
        pred_y = clf.predict(X_test)

        acc = np.mean(pred_y == y_test)
        accuracies.append(acc)

    return float(np.mean(accuracies))


def cross_validate_sklearn_detailed(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    n_folds: int = 5,
) -> dict:
    """Run 5-fold CV for sklearn classifiers, returning per-fold details.

    Returns dict with:
        fold_accuracies: list of per-fold accuracy floats
        mean_accuracy: overall mean
        y_true: concatenated true labels across all folds
        y_pred: concatenated predictions across all folds
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []
    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = _make_classifier(classifier_name)
        clf.fit(X_train, y_train)
        pred_y = clf.predict(X_test)

        fold_accs.append(float(np.mean(pred_y == y_test)))
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(pred_y.tolist())

    return {
        "fold_accuracies": fold_accs,
        "mean_accuracy": float(np.mean(fold_accs)),
        "y_true": all_y_true,
        "y_pred": all_y_pred,
    }
