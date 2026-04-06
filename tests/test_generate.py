# ABOUTME: Tests for generation utilities — false answer selection and prompt formatting.
# ABOUTME: Model-dependent functions are tested via GPU integration; pure logic is unit-tested.

import pytest
from src.generate import pick_false_answer


class TestPickFalseAnswer:
    def test_never_picks_correct(self):
        for correct in range(4):
            for seed in range(20):
                result = pick_false_answer(correct, seed)
                assert result != correct

    def test_deterministic(self):
        a = pick_false_answer(1, seed=42)
        b = pick_false_answer(1, seed=42)
        assert a == b

    def test_returns_valid_index(self):
        for correct in range(4):
            result = pick_false_answer(correct, seed=0)
            assert 0 <= result <= 3
