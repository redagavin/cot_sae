# ABOUTME: Tests for logit lens projection and divergence computation.
# ABOUTME: Validates LayerNorm application, projection shapes, and masked averaging.

import pytest
import torch
import torch.nn as nn
from src.logit_lens import project_to_logits, compute_token_divergence, masked_mean


class TestProjectToLogits:
    def test_output_shape(self):
        residual = torch.randn(20, 64)
        ln = nn.LayerNorm(64)
        unembed = torch.randn(64, 100)
        logits = project_to_logits(residual, ln, unembed)
        assert logits.shape == (20, 100)

    def test_applies_layernorm(self):
        residual = torch.randn(5, 8)
        ln = nn.LayerNorm(8)
        unembed = torch.eye(8)
        logits = project_to_logits(residual, ln, unembed)
        expected = ln(residual)
        assert torch.allclose(logits, expected, atol=1e-5)


class TestComputeTokenDivergence:
    def test_identical_activations_zero_divergence(self):
        residual = torch.randn(10, 64)
        ln = nn.LayerNorm(64)
        unembed = torch.randn(64, 100)
        result = compute_token_divergence(residual, residual, ln, unembed)
        assert torch.allclose(result["cosine"], torch.zeros(10), atol=1e-5)
        assert torch.allclose(result["jsd"], torch.zeros(10), atol=1e-5)

    def test_different_activations_nonzero(self):
        a = torch.randn(10, 64)
        b = torch.randn(10, 64)
        ln = nn.LayerNorm(64)
        unembed = torch.randn(64, 100)
        result = compute_token_divergence(a, b, ln, unembed)
        assert (result["cosine"] > 0).all()
        assert (result["jsd"] > 0).all()

    def test_output_keys_and_shapes(self):
        a = torch.randn(5, 32)
        b = torch.randn(5, 32)
        ln = nn.LayerNorm(32)
        unembed = torch.randn(32, 50)
        result = compute_token_divergence(a, b, ln, unembed)
        assert "cosine" in result and "jsd" in result
        assert result["cosine"].shape == (5,)
        assert result["jsd"].shape == (5,)


class TestMaskedMean:
    def test_ignores_zeros_in_count(self):
        values = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
        counts = torch.tensor([[1, 1, 0], [1, 0, 0]])
        result = masked_mean(values, counts)
        expected = torch.tensor([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_averages_correctly(self):
        values = torch.tensor([[3.0, 6.0], [1.0, 4.0]])
        counts = torch.tensor([[1, 2], [1, 2]])
        result = masked_mean(values, counts)
        expected = torch.tensor([[3.0, 3.0], [1.0, 2.0]])
        assert torch.allclose(result, expected)
