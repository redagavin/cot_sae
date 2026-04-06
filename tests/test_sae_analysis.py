# ABOUTME: Tests for SAE feature extraction and differential analysis.
# ABOUTME: Validates pooling strategies and per-layer analysis structure using synthetic data.

import pytest
import torch
from src.sae_analysis import pool_features, analyze_features


class TestPoolFeatures:
    def test_mean_pool_shape(self):
        features = torch.randn(20, 100)
        result = pool_features(features, method="mean")
        assert result.shape == (100,)

    def test_max_pool_shape(self):
        features = torch.randn(20, 100)
        result = pool_features(features, method="max")
        assert result.shape == (100,)

    def test_max_pool_picks_max(self):
        features = torch.zeros(5, 3)
        features[2, 1] = 10.0
        result = pool_features(features, method="max")
        assert result[1].item() == 10.0

    def test_mean_pool_averages(self):
        features = torch.ones(4, 2)
        features[0, 0] = 5.0
        result = pool_features(features, method="mean")
        assert result[0].item() == pytest.approx(2.0)


class TestAnalyzeFeatures:
    def test_returns_expected_keys(self):
        baseline = torch.randn(50, 100)
        condition = baseline.clone()
        condition[:, 0] += 5.0
        result = analyze_features(baseline, condition)
        assert "n_differential" in result
        assert "feature_indices" in result
        assert "effect_sizes" in result

    def test_detects_large_shift(self):
        torch.manual_seed(42)
        baseline = torch.randn(50, 100)
        condition = baseline.clone()
        condition[:, 5] += 5.0
        result = analyze_features(baseline, condition)
        assert 5 in result["feature_indices"]
