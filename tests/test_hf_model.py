# ABOUTME: Tests for HuggingFace model utilities — batched tokenization and hooked forward pass.
# ABOUTME: Uses small tensors to validate padding, hook capture, and generation length detection.

import pytest
import torch
from src.hf_model import (
    tokenize_batch,
    extract_generation_lengths,
    register_layer_hooks,
    remove_hooks,
)


class TestTokenizeBatch:
    def test_left_pads(self):
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = 0
        tokenizer.return_value = {
            "input_ids": torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]]),
        }
        result = tokenize_batch(tokenizer, ["abc", "de"])
        assert "input_ids" in result
        assert "attention_mask" in result


class TestExtractGenerationLengths:
    def test_finds_eos(self):
        output_tokens = torch.tensor([
            [0, 1, 2, 3, 10, 11, 99, 0],
            [0, 0, 4, 5, 12, 99, 0, 0],
        ])
        gen_lengths = extract_generation_lengths(
            output_tokens, padded_prompt_length=4, eos_token_id=99
        )
        assert gen_lengths == [2, 1]

    def test_no_eos(self):
        output_tokens = torch.tensor([
            [1, 2, 3, 4, 10, 11, 12, 13],
        ])
        gen_lengths = extract_generation_lengths(
            output_tokens, padded_prompt_length=4, eos_token_id=99
        )
        assert gen_lengths == [4]


class TestRegisterLayerHooks:
    def test_hooks_capture_output(self):
        import torch.nn as nn
        model = nn.Sequential()
        model.add_module("layer_0", nn.Linear(4, 4))
        model.add_module("layer_1", nn.Linear(4, 4))

        captured = {}
        hooks = register_layer_hooks(model, layer_indices=[0, 1], captured=captured,
                                     layer_accessor=lambda m, i: list(m.children())[i])
        x = torch.randn(2, 4)
        model(x)
        assert 0 in captured
        assert 1 in captured
        assert captured[0].shape == (2, 4)
        remove_hooks(hooks)
