# ABOUTME: Tests for divergence metrics used in logit lens and SAE analysis.
# ABOUTME: Validates cosine distance, JSD stability, paired t-test with FDR, and Cohen's d.

import pytest
import torch
import numpy as np
from src.metrics import cosine_distance, jsd, find_differential_features, benjamini_hochberg


class TestCosineDistance:
    def test_identical_vectors_zero_distance(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        assert cosine_distance(a, b).item() == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_distance_two(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([-1.0, 0.0])
        assert cosine_distance(a, b).item() == pytest.approx(2.0, abs=1e-6)

    def test_orthogonal_vectors_distance_one(self):
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        assert cosine_distance(a, b).item() == pytest.approx(1.0, abs=1e-6)

    def test_batch_dimension(self):
        a = torch.randn(10, 64)
        b = torch.randn(10, 64)
        result = cosine_distance(a, b)
        assert result.shape == (10,)


class TestJSD:
    def test_identical_distributions_zero(self):
        p = torch.tensor([2.0, 1.0, 0.5])
        q = torch.tensor([2.0, 1.0, 0.5])
        assert jsd(p, q).item() == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        p = torch.tensor([2.0, 1.0, -1.0])
        q = torch.tensor([-1.0, 2.0, 1.0])
        assert jsd(p, q).item() == pytest.approx(jsd(q, p).item(), abs=1e-6)

    def test_bounded_zero_to_ln2(self):
        p = torch.randn(100)
        q = torch.randn(100)
        result = jsd(p, q).item()
        assert 0.0 <= result <= 0.7  # ln(2) ~ 0.693

    def test_batch_dimension(self):
        p = torch.randn(10, 64)
        q = torch.randn(10, 64)
        result = jsd(p, q)
        assert result.shape == (10,)

    def test_no_nan_with_extreme_logits(self):
        p = torch.tensor([100.0, -100.0, -100.0, -100.0])
        q = torch.tensor([-100.0, 100.0, -100.0, -100.0])
        result = jsd(p, q)
        assert not torch.isnan(result).any()
        assert result.item() > 0

    def test_no_nan_with_large_vocab(self):
        p = torch.randn(256000)
        q = torch.randn(256000)
        result = jsd(p, q)
        assert not torch.isnan(result).any()


class TestBenjaminiHochberg:
    def test_no_rejections_when_all_high(self):
        p_values = np.array([0.5, 0.8, 0.9, 0.99])
        rejected = benjamini_hochberg(p_values, q_threshold=0.05)
        assert not rejected.any()

    def test_rejects_obvious_signal(self):
        p_values = np.array([0.001, 0.002, 0.5, 0.9])
        rejected = benjamini_hochberg(p_values, q_threshold=0.05)
        assert rejected[0] and rejected[1]
        assert not rejected[2] and not rejected[3]

    def test_controls_false_positives(self):
        np.random.seed(42)
        p_values = np.random.uniform(0, 1, size=10000)
        rejected = benjamini_hochberg(p_values, q_threshold=0.05)
        assert rejected.sum() < 600

    def test_handles_nan_p_values(self):
        p_values = np.array([0.001, np.nan, 0.5, 0.9])
        rejected = benjamini_hochberg(p_values, q_threshold=0.05)
        assert rejected[0]
        assert not rejected[1]


@pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")
class TestFindDifferentialFeatures:
    def test_detects_shifted_features_paired(self):
        torch.manual_seed(42)
        n_samples = 50
        n_features = 100
        baseline = torch.randn(n_samples, n_features)
        condition = baseline.clone()
        condition[:, 0] += 5.0

        result = find_differential_features(baseline, condition, q_threshold=0.05)
        assert 0 in result["feature_indices"]

    def test_no_false_positives_on_noise(self):
        torch.manual_seed(42)
        n_samples = 50
        n_features = 100
        baseline = torch.randn(n_samples, n_features)
        condition = torch.randn(n_samples, n_features)
        result = find_differential_features(baseline, condition, q_threshold=0.05)
        assert len(result["feature_indices"]) < 15

    def test_returns_effect_sizes(self):
        torch.manual_seed(42)
        n_samples = 50
        n_features = 10
        baseline = torch.randn(n_samples, n_features)
        condition = baseline.clone()
        condition[:, 3] += 3.0

        result = find_differential_features(baseline, condition, q_threshold=0.05)
        assert "effect_sizes" in result
        assert len(result["effect_sizes"]) == len(result["feature_indices"])
        if 3 in result["feature_indices"]:
            idx = result["feature_indices"].index(3)
            assert result["effect_sizes"][idx] > 1.0

    def test_uses_paired_test(self):
        torch.manual_seed(42)
        n_samples = 30
        n_features = 5
        baseline = torch.randn(n_samples, n_features) * 10
        condition = baseline.clone()
        condition[:, 0] += 0.5

        result = find_differential_features(baseline, condition, q_threshold=0.05)
        assert 0 in result["feature_indices"]
