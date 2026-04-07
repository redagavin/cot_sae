# ABOUTME: Generates heatmaps and comparison plots for logit lens and SAE analysis.
# ABOUTME: Computes layer recommendations using both cosine and JSD signals.

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import N_LAYERS, SAE_WIDTHS, HINT_FORMATS


def plot_divergence_heatmap(
    heatmap: torch.Tensor,
    title: str,
    save_path: Path,
):
    """Plot a layer x token position divergence heatmap."""
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap.numpy(), aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_yticks(range(N_LAYERS))
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_layer_comparison(
    logit_cosine: list[float],
    logit_jsd: list[float],
    sae_counts: dict[int, list[float]],
    save_path: Path,
):
    """Plot logit lens divergence and SAE differential counts side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    layers = list(range(N_LAYERS))

    axes[0].bar(layers, logit_cosine)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean Cosine Distance")
    axes[0].set_title("Logit Lens: Cosine Distance")

    axes[1].bar(layers, logit_jsd)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean JSD")
    axes[1].set_title("Logit Lens: JSD")

    bar_width = 0.25
    for i, width_k in enumerate(SAE_WIDTHS):
        if width_k not in sae_counts:
            continue
        offsets = [l + i * bar_width for l in layers]
        axes[2].bar(offsets, sae_counts[width_k], bar_width, label=f"{width_k}k")
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Differential Feature Count")
    axes[2].set_title("SAE: Differential Features (mean-pool)")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def compute_layer_recommendation(
    logit_cosine: list[float],
    logit_jsd: list[float],
    sae_counts_mean: dict[int, list[float]],
    sae_counts_max: dict[int, list[float]] | None = None,
) -> dict:
    """Compute recommended 2-3 layers and best SAE width from all signals.

    Uses rank-aggregation with equal weight between logit lens (cosine+JSD averaged)
    and SAE signal (mean-pool and max-pool ranks averaged). Picks top 3 layers.
    """
    n = len(logit_cosine)

    def rank_desc(values):
        """Rank values descending (highest value gets rank 0)."""
        order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        ranks = [0] * len(values)
        for rank, idx in enumerate(order):
            ranks[idx] = rank
        return ranks

    # Logit lens: average cosine and JSD ranks
    cosine_ranks = rank_desc(logit_cosine)
    jsd_ranks = rank_desc(logit_jsd)
    logit_lens_ranks = [0.5 * (cosine_ranks[i] + jsd_ranks[i]) for i in range(n)]

    # Best SAE width: highest mean fraction of differential features
    available_widths = [w for w in SAE_WIDTHS if w in sae_counts_mean]
    if available_widths:
        best_width = max(
            available_widths,
            key=lambda w: np.mean(sae_counts_mean[w]) / (w * 1000),
        )
    else:
        best_width = SAE_WIDTHS[0]

    # SAE: rank mean-pool and max-pool separately, then average
    mean_ranks = rank_desc(sae_counts_mean.get(best_width, [0] * n))
    if sae_counts_max and best_width in sae_counts_max:
        max_ranks = rank_desc(sae_counts_max[best_width])
        sae_ranks = [0.5 * (mean_ranks[i] + max_ranks[i]) for i in range(n)]
    else:
        sae_ranks = mean_ranks

    # Equal weight: logit lens + SAE
    total_ranks = [logit_lens_ranks[i] + sae_ranks[i] for i in range(n)]
    sorted_layers = sorted(range(n), key=lambda i: total_ranks[i])

    recommended = sorted_layers[:3]

    return {
        "recommended_layers": sorted(recommended),
        "best_sae_width_k": best_width,
        "layer_scores": {l: total_ranks[l] for l in range(n)},
    }
