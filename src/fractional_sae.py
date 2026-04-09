# ABOUTME: Encodes residual stream activations through SAE at specific fractional positions.
# ABOUTME: Returns sparse feature vectors for storage efficiency.

import torch
from src.fractional import to_sparse_features


def encode_at_fractions(
    sae,
    residual: torch.Tensor,
    fraction_indices: list[int],
) -> list[dict]:
    """Encode residual stream through SAE at specific token positions.

    Args:
        sae: loaded SAE model
        residual: [seq_len, hidden_dim] residual stream tensor for one layer
        fraction_indices: absolute token positions to encode

    Returns:
        list of sparse feature dicts, one per fraction index
    """
    positions = torch.tensor(fraction_indices, dtype=torch.long)
    selected = residual[positions]

    with torch.no_grad():
        features = sae.encode(selected.to(sae.device))
    features = features.cpu()

    return [to_sparse_features(features[i]) for i in range(features.shape[0])]
