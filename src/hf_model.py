# ABOUTME: HuggingFace model loading, batched tokenization, and hooked forward pass.
# ABOUTME: Registers forward hooks on selected decoder layers to capture residual streams.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import MODEL_NAME, SELECTED_LAYERS


def load_hf_model(device: str = "cuda"):
    """Load Gemma 2 2B-it with HuggingFace."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def tokenize_batch(tokenizer, prompts: list[str]) -> dict:
    """Tokenize a batch of prompts with left-padding."""
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )


def extract_generation_lengths(
    output_tokens: torch.Tensor,
    padded_prompt_length: int,
    eos_token_id: int,
) -> list[int]:
    """Find actual generation length for each sequence in a batch.

    After left-padded batched generation, all generated tokens start at
    padded_prompt_length (which is max_prompt_length in the batch).

    gen_length excludes the EOS token, so the 100% fraction point
    is the token just before EOS.
    """
    batch_size = output_tokens.shape[0]
    gen_lengths = []
    for i in range(batch_size):
        generated = output_tokens[i, padded_prompt_length:]
        eos_positions = (generated == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            gen_lengths.append(eos_positions[0].item())
        else:
            gen_lengths.append(len(generated))
    return gen_lengths


def register_layer_hooks(
    model,
    layer_indices: list[int],
    captured: dict,
    layer_accessor=None,
) -> list:
    """Register forward hooks on selected decoder layers.

    The hook captures the layer's output (residual stream post-layer).
    For Gemma 2: model.model.layers[i] is the i-th decoder layer.
    """
    if layer_accessor is None:
        layer_accessor = lambda m, i: m.model.layers[i]

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[layer_idx] = output[0].detach()
            else:
                captured[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for idx in layer_indices:
        layer_module = layer_accessor(model, idx)
        handle = layer_module.register_forward_hook(make_hook(idx))
        hooks.append(handle)

    return hooks


def remove_hooks(hooks: list):
    """Remove all registered hooks."""
    for handle in hooks:
        handle.remove()
