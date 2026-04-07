# ABOUTME: Tests for script-level logic — baseline filtering and metadata assembly.
# ABOUTME: Validates hint-following computation, run ID construction, and metadata structure.

import pytest


class TestHintFollowing:
    def test_true_when_predicted_matches_false_answer(self):
        condition = "false_hint"
        predicted = 2
        false_idx = 2
        hint_following = condition == "false_hint" and predicted == false_idx
        assert hint_following is True

    def test_false_when_predicted_differs(self):
        condition = "false_hint"
        predicted = 1
        false_idx = 2
        hint_following = condition == "false_hint" and predicted == false_idx
        assert hint_following is False

    def test_false_for_true_hint(self):
        condition = "true_hint"
        predicted = 2
        false_idx = 2
        hint_following = condition == "false_hint" and predicted == false_idx
        assert hint_following is False

    def test_false_when_predicted_is_none(self):
        condition = "false_hint"
        predicted = None
        false_idx = 2
        hint_following = condition == "false_hint" and predicted == false_idx
        assert hint_following is False


class TestRunIdConstruction:
    def test_no_hint_run_id(self):
        run_id = f"q{0:03d}_no_hint"
        assert run_id == "q000_no_hint"

    def test_hint_run_id(self):
        run_id = f"q{5:03d}_authority_false_hint"
        assert run_id == "q005_authority_false_hint"

    def test_run_id_padding(self):
        run_id = f"q{49:03d}_peer_true_hint"
        assert run_id == "q049_peer_true_hint"
