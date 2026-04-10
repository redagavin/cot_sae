# ABOUTME: Tests for fractional position sampling and sparse feature storage.
# ABOUTME: Validates index computation, sparse round-trip, and EOS detection.

import pytest
import torch
from src.fractional import (
    compute_fraction_indices,
    to_sparse_features,
    from_sparse_features,
    find_eos_position,
)


class TestComputeFractionIndices:
    def test_basic_fractions(self):
        indices = compute_fraction_indices(gen_length=100, n_fractions=20)
        assert len(indices) == 20
        assert indices[0] == 4
        assert indices[-1] == 99

    def test_short_sequence(self):
        indices = compute_fraction_indices(gen_length=10, n_fractions=20)
        assert len(indices) == 20
        assert all(0 <= i < 10 for i in indices)

    def test_single_token(self):
        indices = compute_fraction_indices(gen_length=1, n_fractions=20)
        assert all(i == 0 for i in indices)

    def test_prompt_offset(self):
        indices = compute_fraction_indices(gen_length=100, n_fractions=20, prompt_length=50)
        assert indices[0] == 54
        assert indices[-1] == 149


class TestSparseFeatures:
    def test_round_trip(self):
        dense = torch.zeros(16384)
        dense[10] = 1.5
        dense[100] = 0.3
        dense[5000] = 2.1
        sparse = to_sparse_features(dense)
        recovered = from_sparse_features(sparse, total_features=16384)
        assert torch.allclose(dense, recovered)

    def test_empty_features(self):
        dense = torch.zeros(16384)
        sparse = to_sparse_features(dense)
        assert sparse["indices"].shape[0] == 0
        assert sparse["values"].shape[0] == 0
        recovered = from_sparse_features(sparse, total_features=16384)
        assert torch.allclose(dense, recovered)

    def test_preserves_values(self):
        dense = torch.zeros(65536)
        dense[42] = 3.14
        sparse = to_sparse_features(dense)
        assert 42 in sparse["indices"]
        assert torch.allclose(sparse["values"][sparse["indices"] == 42], torch.tensor(3.14))


class TestFindEosPosition:
    def test_finds_eos(self):
        tokens = torch.tensor([5, 10, 15, 1, 0, 0])
        pos = find_eos_position(tokens, eos_token_id=1, prompt_length=0)
        assert pos == 3

    def test_no_eos(self):
        tokens = torch.tensor([5, 10, 15, 20])
        pos = find_eos_position(tokens, eos_token_id=1, prompt_length=0)
        assert pos == 4

    def test_with_prompt(self):
        tokens = torch.tensor([5, 10, 15, 1, 0])
        pos = find_eos_position(tokens, eos_token_id=1, prompt_length=2)
        assert pos == 3
