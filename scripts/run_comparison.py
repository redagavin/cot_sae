# ABOUTME: Compares logit lens and SAE signals, identifies top layers, and produces final plots.
# ABOUTME: Includes true-hint control comparison and outputs a 2-3 layer recommendation.

import json
import torch
from pathlib import Path

from src.config import OUTPUTS_DIR, N_LAYERS, SAE_WIDTHS, HINT_FORMATS
from src.visualize import plot_divergence_heatmap, plot_layer_comparison, compute_layer_recommendation


def main():
    figures_dir = OUTPUTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load logit lens results
    false_heatmaps = torch.load(OUTPUTS_DIR / "logit_lens" / "heatmaps_false_hint.pt", weights_only=True)
    true_heatmaps = torch.load(OUTPUTS_DIR / "logit_lens" / "heatmaps_true_hint.pt", weights_only=True)

    # Plot heatmaps
    for label, heatmaps in [("false_hint", false_heatmaps), ("true_hint", true_heatmaps)]:
        for metric in ["cosine", "jsd"]:
            plot_divergence_heatmap(
                heatmaps[metric],
                f"Logit Lens {metric} (no-hint vs {label})",
                figures_dir / f"heatmap_{metric}_{label}.png",
            )

    # Per-format heatmaps for false-hint
    for fmt in HINT_FORMATS:
        fmt_heatmaps = torch.load(
            OUTPUTS_DIR / "logit_lens" / f"heatmaps_false_hint_{fmt}.pt", weights_only=True
        )
        for metric in ["cosine", "jsd"]:
            plot_divergence_heatmap(
                fmt_heatmaps[metric],
                f"Logit Lens {metric} — false-hint ({fmt})",
                figures_dir / f"heatmap_{metric}_false_hint_{fmt}.png",
            )

    # Load SAE results (per-format)
    with open(OUTPUTS_DIR / "sae_analysis" / "sae_results_false_hint.json") as f:
        false_sae_by_fmt = json.load(f)
    with open(OUTPUTS_DIR / "sae_analysis" / "sae_results_true_hint.json") as f:
        true_sae_by_fmt = json.load(f)

    # Logit lens scores per layer (count-weighted)
    with open(OUTPUTS_DIR / "logit_lens" / "weighted_means.json") as f:
        weighted_means = json.load(f)
    logit_cosine = weighted_means["false_hint"]["cosine"]
    logit_jsd = weighted_means["false_hint"]["jsd"]

    # SAE counts: average across formats
    sae_counts_mean = {}
    sae_counts_max = {}
    for width_k in SAE_WIDTHS:
        sae_counts_mean[width_k] = []
        sae_counts_max[width_k] = []
        for l in range(N_LAYERS):
            mean_counts = []
            max_counts = []
            for fmt in HINT_FORMATS:
                entry = next(
                    (r for r in false_sae_by_fmt[fmt]
                     if r["layer"] == l and r["width_k"] == width_k and r.get("available", True)),
                    None,
                )
                if entry:
                    mean_counts.append(entry["mean_pool"]["n_differential"])
                    max_counts.append(entry["max_pool"]["n_differential"])
            sae_counts_mean[width_k].append(
                sum(mean_counts) / len(mean_counts) if mean_counts else 0
            )
            sae_counts_max[width_k].append(
                sum(max_counts) / len(max_counts) if max_counts else 0
            )

    # Comparison plot
    plot_layer_comparison(logit_cosine, logit_jsd, sae_counts_mean, figures_dir / "signal_comparison.png")

    # Recommendation
    rec = compute_layer_recommendation(logit_cosine, logit_jsd, sae_counts_mean, sae_counts_max)

    # True-hint control
    true_cosine = weighted_means["true_hint"]["cosine"]
    true_jsd = weighted_means["true_hint"]["jsd"]

    # Signal agreement analysis
    def top_n(values, n=5):
        return set(sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:n])

    top5_cosine = top_n(logit_cosine)
    top5_jsd = top_n(logit_jsd)
    best_w = rec["best_sae_width_k"]
    top5_sae_mean = top_n(sae_counts_mean.get(best_w, [0] * N_LAYERS))
    top5_sae_max = top_n(sae_counts_max.get(best_w, [0] * N_LAYERS))

    print("=== SIGNAL AGREEMENT ===")
    print(f"Top-5 by cosine distance: {sorted(top5_cosine)}")
    print(f"Top-5 by JSD: {sorted(top5_jsd)}")
    print(f"Top-5 by SAE mean-pool ({best_w}k): {sorted(top5_sae_mean)}")
    print(f"Top-5 by SAE max-pool ({best_w}k): {sorted(top5_sae_max)}")
    all_top5 = [top5_cosine, top5_jsd, top5_sae_mean, top5_sae_max]
    agreed_all = set.intersection(*all_top5)
    agreed_any_two = set()
    for i in range(len(all_top5)):
        for j in range(i + 1, len(all_top5)):
            agreed_any_two |= all_top5[i] & all_top5[j]
    print(f"In ALL top-5 lists: {sorted(agreed_all) if agreed_all else 'none'}")
    print(f"In 2+ top-5 lists: {sorted(agreed_any_two)}")

    # Per-format SAE consistency
    print(f"\n=== PER-FORMAT SAE CONSISTENCY ===")
    for width_k in SAE_WIDTHS:
        format_top5s = []
        for fmt in HINT_FORMATS:
            fmt_counts = []
            for l in range(N_LAYERS):
                entry = next(
                    (r for r in false_sae_by_fmt[fmt]
                     if r["layer"] == l and r["width_k"] == width_k and r.get("available", True)),
                    None,
                )
                fmt_counts.append(entry["mean_pool"]["n_differential"] if entry else 0)
            format_top5s.append(top_n(fmt_counts))
        consistent = set.intersection(*format_top5s)
        print(f"  {width_k}k: {sorted(consistent) if consistent else 'none'}")

    # True-hint SAE control
    true_sae_counts = {}
    for width_k in SAE_WIDTHS:
        true_sae_counts[width_k] = []
        for l in range(N_LAYERS):
            counts = []
            for fmt in HINT_FORMATS:
                entry = next(
                    (r for r in true_sae_by_fmt.get(fmt, [])
                     if r["layer"] == l and r["width_k"] == width_k and r.get("available", True)),
                    None,
                )
                if entry:
                    counts.append(entry["mean_pool"]["n_differential"])
            true_sae_counts[width_k].append(
                sum(counts) / len(counts) if counts else 0
            )

    print(f"\n=== TRUE-HINT CONTROL ===")
    print("Logit lens (false-hint should show MORE divergence than true-hint):")
    for l in rec["recommended_layers"]:
        print(f"  Layer {l}: false cosine={logit_cosine[l]:.4f}, true cosine={true_cosine[l]:.4f}")
    print(f"\nSAE differential features (false-hint vs true-hint, {best_w}k):")
    for l in rec["recommended_layers"]:
        false_c = sae_counts_mean.get(best_w, [0]*N_LAYERS)[l]
        true_c = true_sae_counts.get(best_w, [0]*N_LAYERS)[l]
        print(f"  Layer {l}: false={false_c:.1f}, true={true_c:.1f}")

    # Top differential features
    print(f"\n=== TOP SAE FEATURES (false-hint, {best_w}k, mean-pool) ===")
    for l in rec["recommended_layers"]:
        for fmt in HINT_FORMATS:
            entry = next(
                (r for r in false_sae_by_fmt.get(fmt, [])
                 if r["layer"] == l and r["width_k"] == best_w and r.get("available", True)),
                None,
            )
            if entry and entry["mean_pool"]["feature_indices"]:
                top5_feats = entry["mean_pool"]["feature_indices"][:5]
                top5_effects = entry["mean_pool"]["effect_sizes"][:5]
                pairs = [f"{idx}(d={eff:.2f})" for idx, eff in zip(top5_feats, top5_effects)]
                print(f"  Layer {l}, {fmt}: {', '.join(pairs)}")

    print(f"\n=== RECOMMENDATION ===")
    print(f"Recommended layers: {rec['recommended_layers']}")
    print(f"Best SAE width: {rec['best_sae_width_k']}k")

    # Save
    recommendation = {
        **rec,
        "logit_cosine_per_layer": logit_cosine,
        "logit_jsd_per_layer": logit_jsd,
        "true_hint_cosine_per_layer": true_cosine,
        "true_hint_jsd_per_layer": true_jsd,
        "true_hint_sae_counts": true_sae_counts,
        "signal_agreement": {
            "top5_cosine": sorted(top5_cosine),
            "top5_jsd": sorted(top5_jsd),
            "top5_sae_mean": sorted(top5_sae_mean),
            "top5_sae_max": sorted(top5_sae_max),
            "in_all": sorted(agreed_all),
            "in_two_plus": sorted(agreed_any_two),
        },
    }
    with open(OUTPUTS_DIR / "recommendation.json", "w") as f:
        json.dump(recommendation, f, indent=2)

    print(f"\nFigures saved to {figures_dir}/")
    print(f"Recommendation saved to {OUTPUTS_DIR / 'recommendation.json'}")


if __name__ == "__main__":
    main()
