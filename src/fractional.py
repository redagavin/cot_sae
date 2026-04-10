# ABOUTME: Utilities for fractional position sampling and sparse SAE feature storage.
# ABOUTME: Converts between dense and sparse representations for storage efficiency.

import torch


def compute_fraction_indices(
    gen_length: int,
    n_fractions: int = 20,
    prompt_length: int = 0,
) -> list[int]:
    """Compute token indices corresponding to fractional positions of generation."""
    indices = []
    for i in range(1, n_fractions + 1):
        frac = i / n_fractions
        gen_idx = min(int(frac * gen_length) - 1, gen_length - 1)
        gen_idx = max(gen_idx, 0)
        indices.append(prompt_length + gen_idx)
    return indices


def to_sparse_features(dense: torch.Tensor) -> dict:
    """Convert a dense feature vector to sparse representation (nonzero only)."""
    nonzero_mask = dense != 0
    indices = torch.where(nonzero_mask)[0].to(torch.int32)
    values = dense[nonzero_mask]
    return {"indices": indices, "values": values}


def from_sparse_features(sparse: dict, total_features: int) -> torch.Tensor:
    """Reconstruct a dense feature vector from sparse representation."""
    dense = torch.zeros(total_features)
    if len(sparse["indices"]) > 0:
        indices = sparse["indices"]
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)
        values = sparse["values"]
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values)
        dense[indices.long()] = values.float()
    return dense


def find_eos_position(
    tokens: torch.Tensor,
    eos_token_id: int,
    prompt_length: int,
) -> int:
    """Find the position of the first EOS token after the prompt."""
    generated = tokens[prompt_length:]
    eos_positions = (generated == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        return prompt_length + eos_positions[0].item()
    return len(tokens)
