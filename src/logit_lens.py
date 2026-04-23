# ABOUTME: Logit lens analysis — projects residual streams through LayerNorm + unembedding.
# ABOUTME: Computes per-token divergence between conditions using cosine distance and JSD.

import torch
from src.metrics import cosine_distance, jsd


def project_to_logits(
    residual: torch.Tensor,
    ln_final: torch.nn.Module,
    unembed_matrix: torch.Tensor,
) -> torch.Tensor:
    """Project residual stream to logit space via final LayerNorm + unembedding."""
    normed = ln_final(residual)
    return normed.to(unembed_matrix.dtype) @ unembed_matrix


def compute_token_divergence(
    baseline_residual: torch.Tensor,
    condition_residual: torch.Tensor,
    ln_final: torch.nn.Module,
    unembed_matrix: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute per-token divergence between two residual streams via logit lens."""
    baseline_logits = project_to_logits(baseline_residual, ln_final, unembed_matrix)
    condition_logits = project_to_logits(condition_residual, ln_final, unembed_matrix)
    return {
        "cosine": cosine_distance(baseline_logits, condition_logits),
        "jsd": jsd(baseline_logits, condition_logits),
    }


def masked_mean(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Compute mean avoiding division by zero for positions with no data."""
    return values / counts.clamp(min=1)
