# ABOUTME: Tests for visualization data preparation and layer recommendation logic.
# ABOUTME: Validates that recommendation narrows to 2-3 layers using both metrics.

import pytest
from src.visualize import compute_layer_recommendation


class TestComputeLayerRecommendation:
    def test_returns_three_layers(self):
        logit_cosine = [0.1 * i for i in range(26)]
        logit_jsd = [0.05 * i for i in range(26)]
        sae_mean = {16: list(range(26)), 65: list(range(26)), 131: list(range(26))}
        sae_max = {16: list(range(26)), 65: list(range(26)), 131: list(range(26))}
        result = compute_layer_recommendation(logit_cosine, logit_jsd, sae_mean, sae_max)
        assert len(result["recommended_layers"]) == 3

    def test_top_layers_when_all_signals_agree(self):
        logit_cosine = [0.0] * 26
        logit_cosine[23], logit_cosine[24], logit_cosine[25] = 1.0, 2.0, 3.0
        logit_jsd = logit_cosine.copy()
        sae_mean = {65: [0] * 26}
        sae_mean[65][23], sae_mean[65][24], sae_mean[65][25] = 10, 20, 30
        result = compute_layer_recommendation(logit_cosine, logit_jsd, sae_mean)
        assert set(result["recommended_layers"]) == {23, 24, 25}

    def test_returns_best_width_normalized(self):
        logit_cosine = list(range(26))
        logit_jsd = list(range(26))
        sae_mean = {16: [1] * 26, 65: [5] * 26, 131: [8] * 26}
        result = compute_layer_recommendation(logit_cosine, logit_jsd, sae_mean)
        assert result["best_sae_width_k"] == 65

    def test_works_without_max_pool(self):
        logit_cosine = list(range(26))
        logit_jsd = list(range(26))
        sae_mean = {65: list(range(26))}
        result = compute_layer_recommendation(logit_cosine, logit_jsd, sae_mean)
        assert len(result["recommended_layers"]) == 3
