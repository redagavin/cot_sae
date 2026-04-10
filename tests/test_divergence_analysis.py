# tests/test_divergence_analysis.py
# ABOUTME: Tests for divergence analysis helper functions.
# ABOUTME: Validates feature assembly, onset detection, and sparse-to-dense flow.

import pytest
import numpy as np
import torch
from src.fractional import to_sparse_features
from scripts.run_divergence_analysis import (
    build_paired_features,
    compute_divergence_onset,
)


class TestBuildPairedFeatures:
    def test_output_structure(self):
        question_data = []
        for q in range(3):
            nh_fracs = [to_sparse_features(torch.randn(10).abs()) for _ in range(2)]
            fh_fracs = [to_sparse_features(torch.randn(10).abs()) for _ in range(2)]
            question_data.append({
                "question_id": q,
                "no_hint": nh_fracs,
                "condition": fh_fracs,
            })

        features_by_fraction, y, groups = build_paired_features(
            question_data, n_fractions=2, n_features=10
        )
        assert len(features_by_fraction) == 2
        assert features_by_fraction[0].shape == (6, 10)
        assert len(y) == 6
        assert len(groups) == 6
        assert set(y) == {0, 1}


class TestComputeDivergenceOnset:
    def test_clear_onset(self):
        false_auc = [0.51, 0.52, 0.55, 0.65, 0.80, 0.90]
        true_auc = [0.50, 0.51, 0.50, 0.52, 0.51, 0.53]
        onset = compute_divergence_onset(false_auc, true_auc, threshold=0.05, sustained=2)
        assert onset is not None
        assert onset <= 3

    def test_no_onset(self):
        false_auc = [0.51, 0.52, 0.51, 0.52]
        true_auc = [0.50, 0.51, 0.50, 0.51]
        onset = compute_divergence_onset(false_auc, true_auc, threshold=0.05, sustained=2)
        assert onset is None
