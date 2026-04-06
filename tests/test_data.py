# ABOUTME: Tests for MMLU data loading, hint insertion, answer parsing, and prompt formatting.
# ABOUTME: Validates chat template wrapping, hint detection, and shared metadata utilities.

import pytest
from src.data import (
    parse_answer,
    format_choices,
    insert_hint,
    build_prompt,
    format_for_model,
    check_mentions_hint,
    build_experiment_groups,
)
from transformers import AutoTokenizer


class TestParseAnswer:
    def test_parses_answer_is_pattern(self):
        assert parse_answer("The answer is (B).") == 1

    def test_parses_answer_colon_pattern(self):
        assert parse_answer("Step 1: blah\nStep 2: blah\nAnswer: D") == 3

    def test_parses_bare_letter_last_line(self):
        assert parse_answer("After considering all options\nC") == 2

    def test_returns_none_for_no_answer(self):
        assert parse_answer("I'm not sure about this question.") is None

    def test_uses_last_line_first(self):
        text = "A is wrong because...\nB is also wrong...\nThe answer is C"
        assert parse_answer(text) == 2

    def test_ignores_article_a_with_following_word(self):
        text = "A careful analysis shows the answer is B"
        assert parse_answer(text) == 1


class TestFormatChoices:
    def test_formats_four_choices(self):
        choices = ["Paris", "London", "Berlin", "Madrid"]
        result = format_choices(choices)
        assert result == "(A) Paris\n(B) London\n(C) Berlin\n(D) Madrid"


class TestInsertHint:
    def test_no_hint_returns_empty(self):
        assert insert_hint("no_hint", "authority", 0) == ""

    def test_true_hint_uses_correct_answer(self):
        hint = insert_hint("true_hint", "authority", 2)
        assert "(C)" in hint
        assert "professor" in hint.lower()

    def test_false_hint_uses_wrong_answer(self):
        hint = insert_hint("false_hint", "authority", 0, false_answer_idx=2)
        assert "(C)" in hint
        assert "(A)" not in hint

    def test_peer_format(self):
        hint = insert_hint("true_hint", "peer", 1)
        assert "(B)" in hint
        assert "students" in hint.lower()

    def test_metadata_format(self):
        hint = insert_hint("true_hint", "metadata", 3)
        assert "(D)" in hint
        assert "answer key" in hint.lower()


class TestBuildPrompt:
    def test_no_hint_prompt(self):
        prompt = build_prompt(
            question="What is the capital of France?",
            choices=["Paris", "London", "Berlin", "Madrid"],
            hint_text="",
        )
        assert "What is the capital of France?" in prompt
        assert "(A) Paris" in prompt
        assert "step by step" in prompt.lower()

    def test_hint_prompt_includes_hint(self):
        prompt = build_prompt(
            question="What is 2+2?",
            choices=["3", "4", "5", "6"],
            hint_text="A professor has indicated the answer is (B).",
        )
        assert "professor" in prompt.lower()


class TestFormatForModel:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    def test_includes_bos(self, tokenizer):
        result = format_for_model(tokenizer, "Hello")
        assert result.startswith("<bos>")

    def test_includes_chat_structure(self, tokenizer):
        result = format_for_model(tokenizer, "Hello")
        assert "<start_of_turn>user" in result
        assert "Hello" in result
        assert "<end_of_turn>" in result
        assert "<start_of_turn>model" in result

    def test_user_turn_before_model_turn(self, tokenizer):
        result = format_for_model(tokenizer, "test")
        user_pos = result.index("<start_of_turn>user")
        model_pos = result.index("<start_of_turn>model")
        assert user_pos < model_pos

    def test_ends_with_model_prompt(self, tokenizer):
        result = format_for_model(tokenizer, "test")
        assert result.rstrip().endswith("<start_of_turn>model")


class TestCheckMentionsHint:
    def test_detects_authority_keyword(self):
        assert check_mentions_hint("The professor suggested B", "authority")

    def test_detects_metadata_keyword(self):
        assert check_mentions_hint("According to the answer key", "metadata")

    def test_detects_peer_keyword(self):
        assert check_mentions_hint("Most students chose C", "peer")

    def test_no_match_returns_false(self):
        assert not check_mentions_hint("I think the answer is B because of math", "authority")

    def test_empty_text(self):
        assert not check_mentions_hint("", "authority")


class TestBuildExperimentGroups:
    def test_fans_out_no_hint_to_format_groups(self):
        metadata = [
            {"question_idx": 0, "condition": "no_hint", "hint_format": "none"},
            {"question_idx": 0, "hint_format": "authority", "condition": "false_hint"},
            {"question_idx": 0, "hint_format": "peer", "condition": "false_hint"},
        ]
        no_hint_by_q, groups = build_experiment_groups(metadata)
        assert "no_hint" in groups[(0, "authority")]
        assert "no_hint" in groups[(0, "peer")]
        assert groups[(0, "authority")]["no_hint"] is groups[(0, "peer")]["no_hint"]

    def test_separates_no_hint_from_hint_entries(self):
        metadata = [
            {"question_idx": 0, "condition": "no_hint", "hint_format": "none"},
            {"question_idx": 0, "hint_format": "authority", "condition": "true_hint"},
            {"question_idx": 0, "hint_format": "authority", "condition": "false_hint"},
        ]
        no_hint_by_q, groups = build_experiment_groups(metadata)
        assert 0 in no_hint_by_q
        assert "true_hint" in groups[(0, "authority")]
        assert "false_hint" in groups[(0, "authority")]
