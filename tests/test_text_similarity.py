# tests/test_text_similarity.py
# ABOUTME: Tests for text similarity computation at fractional positions.
# ABOUTME: Validates token-based truncation and embedding cosine similarity.

import pytest
from src.text_similarity import (
    text_at_token_fraction,
    compute_text_similarity_curve,
)


class TestTextAtTokenFraction:
    def test_half_fraction(self):
        tokens = ["one", "two", "three", "four", "five", "six"]
        result = text_at_token_fraction(tokens, 0.5)
        assert result == "one two three"

    def test_full_fraction(self):
        tokens = ["hello", "world"]
        result = text_at_token_fraction(tokens, 1.0)
        assert result == "hello world"

    def test_small_fraction(self):
        tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        result = text_at_token_fraction(tokens, 0.1)
        assert len(result) > 0


class TestComputeTextSimilarityCurve:
    def test_identical_texts_high_similarity(self):
        text_a = "The answer is clearly option B because of the evidence presented."
        text_b = "The answer is clearly option B because of the evidence presented."
        fractions = [0.5, 1.0]
        curve = compute_text_similarity_curve(text_a, text_b, fractions)
        assert len(curve) == 2
        assert all(s > 0.99 for s in curve)

    def test_different_texts_lower_similarity(self):
        text_a = "I think the answer is B based on scientific evidence and reasoning."
        text_b = "The professor suggested C so I will go with that recommendation."
        fractions = [0.5, 1.0]
        curve = compute_text_similarity_curve(text_a, text_b, fractions)
        assert len(curve) == 2
        assert all(0.0 <= s <= 1.0 for s in curve)
