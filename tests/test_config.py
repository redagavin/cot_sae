# ABOUTME: Tests for configuration consistency and invariants.
# ABOUTME: Validates that hint formats, keywords, conditions, and paths are well-formed.

from pathlib import Path
from src.config import (
    HINT_FORMATS,
    HINT_KEYWORDS,
    CONDITIONS,
    ANSWER_LETTERS,
    SAE_WIDTHS,
    N_LAYERS,
    DATA_DIR,
    OUTPUTS_DIR,
)


class TestConfigConsistency:
    def test_hint_format_keys_match_keyword_keys(self):
        assert set(HINT_FORMATS.keys()) == set(HINT_KEYWORDS.keys())

    def test_all_hint_keywords_are_nonempty(self):
        for fmt, keywords in HINT_KEYWORDS.items():
            assert len(keywords) > 0, f"No keywords for format {fmt}"

    def test_conditions_contains_expected(self):
        assert "no_hint" in CONDITIONS
        assert "true_hint" in CONDITIONS
        assert "false_hint" in CONDITIONS

    def test_answer_letters_are_four(self):
        assert ANSWER_LETTERS == ["A", "B", "C", "D"]

    def test_sae_widths_are_positive(self):
        assert all(w > 0 for w in SAE_WIDTHS)

    def test_n_layers_positive(self):
        assert N_LAYERS > 0

    def test_paths_are_path_objects(self):
        assert isinstance(DATA_DIR, Path)
        assert isinstance(OUTPUTS_DIR, Path)
