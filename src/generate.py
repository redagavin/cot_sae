# ABOUTME: Loads Gemma 2 2B-it via TransformerLens and generates CoT responses.
# ABOUTME: Caches residual stream activations at all layers via a single forward pass.

import random
import torch
from transformer_lens import HookedTransformer
from src.config import MODEL_NAME, MAX_NEW_TOKENS, N_LAYERS


def load_model(device: str = "cuda") -> HookedTransformer:
    """Load Gemma 2 2B-it with TransformerLens."""
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=device,
        dtype=torch.float16,
    )
    return model


def pick_false_answer(correct_idx: int, seed: int) -> int:
    """Deterministically pick a wrong answer index."""
    rng = random.Random(seed)
    candidates = [i for i in range(4) if i != correct_idx]
    return rng.choice(candidates)


def generate_response(model: HookedTransformer, formatted_prompt: str) -> str:
    """Generate a CoT response for a single formatted prompt. Returns generated text only.

    Expects formatted_prompt from tokenizer.apply_chat_template(), which includes BOS.
    Uses prepend_bos=False to avoid double BOS.
    """
    tokens = model.to_tokens(formatted_prompt, prepend_bos=False)
    output_tokens = model.generate(
        tokens,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )
    generated_tokens = output_tokens[0, tokens.shape[1]:]
    return model.tokenizer.decode(generated_tokens, skip_special_tokens=True)


def generate_with_cache(
    model: HookedTransformer,
    formatted_prompt: str,
) -> tuple[str, dict[int, torch.Tensor]]:
    """Generate a response and cache residual stream activations at all layers.

    Uses a single forward pass over the completed sequence with causal masking.
    For standard causal transformers this produces identical activations to
    autoregressive generation. Gemma 2's alternating sliding window attention
    may introduce minor discrepancies — see Known Limitations in the plan.

    Returns:
        text: the generated text
        activations: dict mapping layer index to tensor [seq_len, hidden_dim]
    """
    tokens = model.to_tokens(formatted_prompt, prepend_bos=False)
    output_tokens = model.generate(
        tokens,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
    )

    _, cache = model.run_with_cache(
        output_tokens.clone(),
        names_filter=lambda name: "resid_post" in name,
    )

    activations = {}
    for layer in range(N_LAYERS):
        activations[layer] = cache["resid_post", layer][0].detach().cpu()

    generated_tokens = output_tokens[0, tokens.shape[1]:]
    text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return text, activations
