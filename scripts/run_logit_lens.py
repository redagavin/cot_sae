# ABOUTME: Runs logit lens analysis across all 26 layers for no-hint vs false-hint and no-hint vs true-hint.
# ABOUTME: Uses masked averaging to avoid zero-padding bias. Produces divergence heatmaps.

import json
import torch
from tqdm import tqdm

from src.config import OUTPUTS_DIR, N_LAYERS, HINT_FORMATS, MAX_NEW_TOKENS
from src.logit_lens import compute_token_divergence, masked_mean
from src.data import build_experiment_groups
from src.generate import load_model


def run_comparison(groups, activations_dir, ln_final, unembed, condition_key, n_layers, max_tokens, device="cpu"):
    """Compute logit lens divergence for no-hint vs a target condition across all layers.

    Returns heatmaps, per-format heatmaps, and count-weighted means.
    """
    layer_sum = {m: {l: torch.zeros(max_tokens) for l in range(n_layers)} for m in ["cosine", "jsd"]}
    layer_count = {m: {l: torch.zeros(max_tokens) for l in range(n_layers)} for m in ["cosine", "jsd"]}

    fmt_sum = {fmt: {m: {l: torch.zeros(max_tokens) for l in range(n_layers)} for m in ["cosine", "jsd"]} for fmt in HINT_FORMATS}
    fmt_count = {fmt: {m: {l: torch.zeros(max_tokens) for l in range(n_layers)} for m in ["cosine", "jsd"]} for fmt in HINT_FORMATS}

    for (q_idx, hint_format), conditions in tqdm(groups.items(), desc=f"no_hint vs {condition_key}"):
        if condition_key not in conditions:
            continue

        no_hint_entry = conditions.get("no_hint")
        if no_hint_entry is None:
            continue
        cond_entry = conditions[condition_key]

        no_hint_acts = torch.load(activations_dir / f"{no_hint_entry['run_id']}.pt", weights_only=True)
        cond_acts = torch.load(activations_dir / f"{cond_entry['run_id']}.pt", weights_only=True)

        prompt_len_nh = no_hint_entry["prompt_length"]
        prompt_len_cond = cond_entry["prompt_length"]

        for layer in range(n_layers):
            nh_resid = no_hint_acts[layer][prompt_len_nh:].to(device=device, dtype=torch.float32)
            cond_resid = cond_acts[layer][prompt_len_cond:].to(device=device, dtype=torch.float32)

            min_len = min(len(nh_resid), len(cond_resid))
            if min_len == 0:
                continue

            divergence = compute_token_divergence(
                nh_resid[:min_len], cond_resid[:min_len], ln_final, unembed
            )

            for metric in ["cosine", "jsd"]:
                div_cpu = divergence[metric].cpu()
                layer_sum[metric][layer][:min_len] += div_cpu
                layer_count[metric][layer][:min_len] += 1
                fmt_sum[hint_format][metric][layer][:min_len] += div_cpu
                fmt_count[hint_format][metric][layer][:min_len] += 1

    # Aggregate with masked mean
    heatmaps = {}
    for metric in ["cosine", "jsd"]:
        heatmaps[metric] = torch.stack([
            masked_mean(layer_sum[metric][l], layer_count[metric][l])
            for l in range(n_layers)
        ])

    fmt_heatmaps = {}
    for fmt in HINT_FORMATS:
        fmt_heatmaps[fmt] = {}
        for metric in ["cosine", "jsd"]:
            fmt_heatmaps[fmt][metric] = torch.stack([
                masked_mean(fmt_sum[fmt][metric][l], fmt_count[fmt][metric][l])
                for l in range(n_layers)
            ])

    # Count-weighted mean per layer
    weighted_means = {}
    for metric in ["cosine", "jsd"]:
        weighted_means[metric] = [
            layer_sum[metric][l].sum().item() / max(layer_count[metric][l].sum().item(), 1)
            for l in range(n_layers)
        ]

    return heatmaps, fmt_heatmaps, weighted_means


def main():
    results_dir = OUTPUTS_DIR / "logit_lens"
    results_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = OUTPUTS_DIR / "activations"
    metadata_dir = OUTPUTS_DIR / "metadata"

    with open(metadata_dir / "generation_metadata.json") as f:
        metadata = json.load(f)

    print("Loading model for LayerNorm and unembedding matrix...")
    model = load_model()
    ln_final = model.ln_final.float()
    unembed = model.W_U.detach().float()
    device = unembed.device
    del model
    torch.cuda.empty_cache()

    no_hint_by_q, groups = build_experiment_groups(metadata)

    max_tokens = MAX_NEW_TOKENS

    print("Analyzing no-hint vs false-hint...")
    false_heatmaps, false_fmt_heatmaps, false_weighted = run_comparison(
        groups, activations_dir, ln_final, unembed, "false_hint", N_LAYERS, max_tokens, device
    )
    torch.save(false_heatmaps, results_dir / "heatmaps_false_hint.pt")
    for fmt in HINT_FORMATS:
        torch.save(false_fmt_heatmaps[fmt], results_dir / f"heatmaps_false_hint_{fmt}.pt")

    print("Analyzing no-hint vs true-hint (control)...")
    true_heatmaps, true_fmt_heatmaps, true_weighted = run_comparison(
        groups, activations_dir, ln_final, unembed, "true_hint", N_LAYERS, max_tokens, device
    )
    torch.save(true_heatmaps, results_dir / "heatmaps_true_hint.pt")

    # Save count-weighted means
    with open(results_dir / "weighted_means.json", "w") as f:
        json.dump({"false_hint": false_weighted, "true_hint": true_weighted}, f, indent=2)

    for label, heatmaps in [("FALSE-HINT", false_heatmaps), ("TRUE-HINT", true_heatmaps)]:
        print(f"\nMean cosine distance per layer (no-hint vs {label}):")
        for l in range(N_LAYERS):
            print(f"  Layer {l:2d}: {heatmaps['cosine'][l].mean():.4f}")


if __name__ == "__main__":
    main()
