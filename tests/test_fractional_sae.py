# ABOUTME: Tests for SAE encoding at fractional positions with sparse output.
# ABOUTME: Uses mock SAE to validate encoding logic without GPU.

import pytest
import torch
from unittest.mock import MagicMock
from src.fractional_sae import encode_at_fractions


class TestEncodeAtFractions:
    def _make_mock_sae(self, output_dim):
        sae = MagicMock()
        sae.device = torch.device("cpu")
        def fake_encode(x):
            out = torch.zeros(x.shape[0], output_dim)
            out[:, 0] = 1.0
            out[:, 1] = 0.5
            return out
        sae.encode = fake_encode
        return sae

    def test_output_structure(self):
        sae = self._make_mock_sae(100)
        residual = torch.randn(50, 64)
        fraction_indices = [4, 9, 14, 19, 24]
        result = encode_at_fractions(sae, residual, fraction_indices)
        assert len(result) == 5
        for entry in result:
            assert "indices" in entry
            assert "values" in entry

    def test_correct_number_of_positions(self):
        sae = self._make_mock_sae(100)
        residual = torch.randn(20, 64)
        fraction_indices = [5, 10, 15]
        result = encode_at_fractions(sae, residual, fraction_indices)
        assert len(result) == 3

    def test_sparse_output(self):
        sae = self._make_mock_sae(100)
        residual = torch.randn(50, 64)
        fraction_indices = [24]
        result = encode_at_fractions(sae, residual, fraction_indices)
        assert result[0]["indices"].shape[0] == 2
        assert result[0]["values"].shape[0] == 2
