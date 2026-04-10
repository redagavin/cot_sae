# ABOUTME: Tests for the divergence generation script helper functions.
# ABOUTME: Validates chunk splitting and metadata assembly.

import pytest
from scripts.run_divergence_generation import split_into_chunks, build_run_metadata


class TestSplitIntoChunks:
    def test_even_split(self):
        items = list(range(12))
        chunks = split_into_chunks(items, n_chunks=4)
        assert len(chunks) == 4
        assert [len(c) for c in chunks] == [3, 3, 3, 3]

    def test_uneven_split(self):
        items = list(range(10))
        chunks = split_into_chunks(items, n_chunks=4)
        assert len(chunks) == 4
        total = sum(len(c) for c in chunks)
        assert total == 10

    def test_single_chunk(self):
        items = list(range(5))
        chunks = split_into_chunks(items, n_chunks=1)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5


class TestBuildRunMetadata:
    def test_structure(self):
        meta = build_run_metadata(
            run_id="q00001_authority_false_hint",
            question_idx=1,
            hint_format="authority",
            condition="false_hint",
            correct_answer=2,
            false_answer=0,
            predicted=0,
            response="The answer is A.",
            prompt_length=50,
            gen_length=100,
        )
        assert meta["run_id"] == "q00001_authority_false_hint"
        assert meta["hint_following"] is True
        assert meta["gen_length"] == 100

    def test_no_hint_following(self):
        meta = build_run_metadata(
            run_id="q00001_no_hint",
            question_idx=1,
            hint_format="none",
            condition="no_hint",
            correct_answer=2,
            false_answer=0,
            predicted=2,
            response="The answer is C.",
            prompt_length=50,
            gen_length=80,
        )
        assert meta["hint_following"] is False
