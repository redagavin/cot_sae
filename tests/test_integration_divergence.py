# tests/test_integration_divergence.py
# ABOUTME: Smoke test for the full divergence localization pipeline.
# ABOUTME: Validates end-to-end data flow with synthetic data.

import pytest
import numpy as np
import torch
from src.fractional import to_sparse_features, from_sparse_features
from src.classifier import tune_regularization, train_classifier, compute_auc_per_fraction
from scripts.run_divergence_analysis import build_paired_features, compute_divergence_onset


class TestEndToEndPipeline:
    def test_sparse_to_classifier_flow(self):
        """Sparse features → paired assembly → classifier → AUC curve."""
        n_questions = 30
        n_fractions = 5
        n_features = 100
        rng = np.random.RandomState(42)

        question_data = []
        for q in range(n_questions):
            nh_fracs = []
            cond_fracs = []
            for f in range(n_fractions):
                nh_dense = torch.zeros(n_features)
                active = rng.choice(n_features, 5, replace=False)
                nh_dense[active] = torch.from_numpy(rng.randn(5).astype(np.float32))
                nh_fracs.append(to_sparse_features(nh_dense))

                cond_dense = torch.zeros(n_features)
                cond_dense[active] = torch.from_numpy(
                    (rng.randn(5) + (f + 1) * 0.5).astype(np.float32)
                )
                cond_fracs.append(to_sparse_features(cond_dense))

            question_data.append({
                "question_id": q,
                "no_hint": nh_fracs,
                "condition": cond_fracs,
            })

        features_by_fraction, y, groups = build_paired_features(
            question_data, n_fractions=n_fractions, n_features=n_features
        )

        X_all = np.vstack([features_by_fraction[f] for f in range(n_fractions)])
        y_all = np.tile(y, n_fractions)
        g_all = np.tile(groups, n_fractions)

        best_C = tune_regularization(X_all, y_all, g_all, n_folds=3)
        clf = train_classifier(X_all, y_all, C=best_C)
        auc_curve = compute_auc_per_fraction(clf, features_by_fraction, y, n_fractions)

        assert len(auc_curve) == n_fractions
        assert auc_curve[-1] >= auc_curve[0]

    def test_divergence_onset_detection(self):
        """Onset detection with synthetic AUC curves."""
        false_auc = [0.52, 0.53, 0.55, 0.70, 0.85, 0.90]
        true_auc = [0.51, 0.52, 0.51, 0.52, 0.53, 0.52]
        onset = compute_divergence_onset(false_auc, true_auc, threshold=0.05, sustained=2)
        assert onset is not None
        assert onset <= 3
