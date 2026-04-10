# tests/test_classifier.py
# ABOUTME: Tests for linear classifier training and AUC curve computation.
# ABOUTME: Validates GroupKFold cross-validation and per-fraction evaluation.

import pytest
import numpy as np
from src.classifier import (
    tune_regularization,
    train_classifier,
    compute_auc_per_fraction,
    compute_bootstrap_ci,
)


class TestTuneRegularization:
    def test_returns_best_C(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(100, 10) + 1, rng.randn(100, 10) - 1])
        y = np.array([0] * 100 + [1] * 100)
        groups = np.array([i // 2 for i in range(200)])
        best_C = tune_regularization(X, y, groups, n_folds=5)
        assert isinstance(best_C, float)
        assert best_C > 0

    def test_groups_respected(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(20, 5) + (i % 2) * 3 for i in range(10)])
        y = np.array([i % 2 for i in range(10) for _ in range(20)])
        groups = np.array([i for i in range(10) for _ in range(20)])
        best_C = tune_regularization(X, y, groups, n_folds=5)
        assert best_C > 0


class TestTrainClassifier:
    def test_returns_fitted_model(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(50, 10) + 2, rng.randn(50, 10) - 2])
        y = np.array([0] * 50 + [1] * 50)
        clf = train_classifier(X, y, C=1.0)
        accuracy = (clf.predict(X) == y).mean()
        assert accuracy > 0.9


class TestComputeAucPerFraction:
    def test_returns_correct_shape(self):
        rng = np.random.RandomState(42)
        n_samples = 40
        n_fractions = 5
        n_features = 10
        features_by_fraction = {}
        y = np.array([i % 2 for i in range(n_samples)])
        for f in range(n_fractions):
            X = rng.randn(n_samples, n_features) + (y * 4)[:, None]
            features_by_fraction[f] = X

        clf = train_classifier(features_by_fraction[0], y, C=1.0)
        auc_curve = compute_auc_per_fraction(clf, features_by_fraction, y, n_fractions)
        assert len(auc_curve) == n_fractions
        assert all(0.0 <= a <= 1.0 for a in auc_curve)


class TestBootstrapCI:
    def test_returns_lower_upper(self):
        rng = np.random.RandomState(42)
        n = 40
        probs_false = rng.rand(n)
        y_false = (probs_false > 0.3).astype(int)
        probs_true = rng.rand(n) * 0.5
        y_true = (probs_true > 0.25).astype(int)
        question_ids = np.array([i // 2 for i in range(n)])
        lower, upper = compute_bootstrap_ci(
            probs_false, y_false, probs_true, y_true, question_ids,
        )
        assert lower < upper

    def test_identical_predictions_ci_near_zero(self):
        n = 40
        probs = np.array([0.6] * n)
        y = np.array([0, 1] * (n // 2))
        question_ids = np.array([i // 2 for i in range(n)])
        lower, upper = compute_bootstrap_ci(
            probs, y, probs, y, question_ids,
        )
        assert abs(lower) < 0.1
        assert abs(upper) < 0.1
