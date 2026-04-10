# src/classifier.py
# ABOUTME: L2-regularized logistic regression for distinguishing hint conditions.
# ABOUTME: Includes GroupKFold CV for regularization tuning and bootstrap CI for onset detection.

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import roc_auc_score


def tune_regularization(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    C_values: list[float] | None = None,
) -> float:
    """Find the best L2 regularization strength via GroupKFold cross-validation.

    Groups ensure all samples from the same question stay in the same fold.
    """
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    gkf = GroupKFold(n_splits=n_folds)
    best_C = C_values[0]
    best_score = -1.0

    for C in C_values:
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=gkf, groups=groups, scoring="roc_auc", n_jobs=-1)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_C = C

    return best_C


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
) -> LogisticRegression:
    """Train an L2-regularized logistic regression classifier."""
    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    return clf


def compute_auc_per_fraction(
    clf: LogisticRegression,
    features_by_fraction: dict[int, np.ndarray],
    y: np.ndarray,
    n_fractions: int,
) -> list[float]:
    """Compute AUC at each fractional position using a trained classifier.

    Args:
        clf: trained classifier
        features_by_fraction: {fraction_idx: np.ndarray [n_samples, n_features]}
        y: [n_samples] binary labels, same ordering as feature arrays
        n_fractions: number of fraction points

    Returns:
        list of AUC values, one per fraction
    """
    auc_curve = []
    for f in range(n_fractions):
        X = features_by_fraction[f]
        probs = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        auc_curve.append(auc)
    return auc_curve


def _bootstrap_one(
    sampled_idx_false: np.ndarray,
    sampled_idx_true: np.ndarray,
    probs_false: np.ndarray,
    y_false: np.ndarray,
    probs_true: np.ndarray,
    y_true: np.ndarray,
) -> float | None:
    """Compute one bootstrap AUC gap sample. Returns None if single-class."""
    try:
        auc_f = roc_auc_score(y_false[sampled_idx_false], probs_false[sampled_idx_false])
        auc_t = roc_auc_score(y_true[sampled_idx_true], probs_true[sampled_idx_true])
        return auc_f - auc_t
    except ValueError:
        return None


def compute_bootstrap_ci(
    probs_false: np.ndarray,
    y_false: np.ndarray,
    probs_true: np.ndarray,
    y_true: np.ndarray,
    question_ids: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for AUC gap (false-hint AUC minus true-hint AUC).

    Resamples unique question IDs, gathers all corresponding samples,
    and recomputes AUC for both conditions per resample.
    Uses joblib for parallel execution across bootstrap iterations.
    """
    rng = np.random.RandomState(seed)
    unique_qids = np.unique(question_ids)
    n_questions = len(unique_qids)

    qid_to_idx = {}
    for idx, qid in enumerate(question_ids):
        qid_to_idx.setdefault(qid, []).append(idx)

    # Pre-generate all resampled indices for reproducibility
    all_idx_false = []
    all_idx_true = []
    for _ in range(n_bootstrap):
        sampled_qids = rng.choice(unique_qids, size=n_questions, replace=True)
        all_idx_false.append(np.concatenate([qid_to_idx[q] for q in sampled_qids]))
        all_idx_true.append(np.concatenate([qid_to_idx[q] for q in sampled_qids]))

    results = Parallel(n_jobs=-1)(
        delayed(_bootstrap_one)(
            all_idx_false[i], all_idx_true[i],
            probs_false, y_false, probs_true, y_true,
        )
        for i in range(n_bootstrap)
    )

    diffs = [r for r in results if r is not None]

    alpha = (1 - ci) / 2
    lower = np.percentile(diffs, 100 * alpha)
    upper = np.percentile(diffs, 100 * (1 - alpha))
    return float(lower), float(upper)
