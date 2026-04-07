# ABOUTME: Divergence metrics for comparing activation distributions across conditions.
# ABOUTME: Provides numerically stable JSD, paired t-test with FDR correction, and cosine distance.

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine distance (1 - cosine_similarity) along last dimension."""
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def jsd(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two sets of logits.

    Inputs are raw logits (pre-softmax). Returns JSD per element in batch.
    Numerically stable: clamps near-zero probabilities to avoid NaN from 0*log(0).
    """
    eps = 1e-8
    p = F.softmax(p_logits, dim=-1).clamp(min=eps)
    q = F.softmax(q_logits, dim=-1).clamp(min=eps)
    m = 0.5 * (p + q)

    kl_p = (p * (p.log() - m.log())).sum(dim=-1)
    kl_q = (q * (q.log() - m.log())).sum(dim=-1)

    return 0.5 * (kl_p + kl_q)


def benjamini_hochberg(p_values: np.ndarray, q_threshold: float = 0.05) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Returns boolean array indicating which hypotheses are rejected.
    NaN p-values are never rejected.
    """
    n = len(p_values)
    rejected = np.zeros(n, dtype=bool)
    valid = ~np.isnan(p_values)

    if not valid.any():
        return rejected

    valid_p = p_values[valid]
    valid_indices = np.where(valid)[0]

    sorted_order = np.argsort(valid_p)
    sorted_p = valid_p[sorted_order]
    thresholds = q_threshold * np.arange(1, len(sorted_p) + 1) / len(sorted_p)

    below = sorted_p <= thresholds
    if not below.any():
        return rejected

    max_k = np.where(below)[0][-1]
    rejected_sorted = sorted_order[:max_k + 1]
    rejected[valid_indices[rejected_sorted]] = True

    return rejected


def find_differential_features(
    baseline: torch.Tensor,
    condition: torch.Tensor,
    q_threshold: float = 0.05,
) -> dict:
    """Find features with significantly different activation between paired conditions.

    Uses paired t-test with Benjamini-Hochberg FDR correction.

    Args:
        baseline: shape [n_samples, n_features], activations under no-hint
        condition: shape [n_samples, n_features], activations under false-hint
        q_threshold: FDR threshold for Benjamini-Hochberg correction

    Returns:
        dict with 'feature_indices' (list[int]) and 'effect_sizes' (list[float])
    """
    baseline_np = baseline.detach().cpu().numpy()
    condition_np = condition.detach().cpu().numpy()

    # Vectorized paired t-test across all features at once
    with np.errstate(divide="ignore", invalid="ignore"):
        _, p_values = stats.ttest_rel(baseline_np, condition_np, axis=0)

    # FDR correction
    rejected = benjamini_hochberg(p_values, q_threshold)

    indices = []
    effect_sizes = []
    for i in np.where(rejected)[0]:
        # Cohen's d_z for paired data: mean(diff) / std(diff)
        diff = condition_np[:, i] - baseline_np[:, i]
        std_diff = diff.std(ddof=1)
        if std_diff > 0:
            cohen_d = abs(diff.mean()) / std_diff
        else:
            cohen_d = 0.0
        indices.append(int(i))
        effect_sizes.append(float(cohen_d))

    # Sort by effect size descending
    if indices:
        paired = sorted(zip(indices, effect_sizes), key=lambda x: x[1], reverse=True)
        indices, effect_sizes = map(list, zip(*paired))

    return {"feature_indices": indices, "effect_sizes": effect_sizes}
