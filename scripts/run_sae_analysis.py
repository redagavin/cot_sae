# ABOUTME: Runs SAE differential feature analysis across all 26 layers and 3 SAE widths.
# ABOUTME: Processes one layer at a time for memory efficiency. Includes true-hint control.

import json
import torch
from tqdm import tqdm

from src.config import OUTPUTS_DIR, N_LAYERS, SAE_WIDTHS, HINT_FORMATS
from src.sae_analysis import analyze_layer_width


def load_layer_activations_by_format(
    activations_dir, groups, no_hint_by_q, layer, condition_key, hint_format
):
    """Load activation pairs for one layer and one hint format.

    Returns (no_hint_list, condition_list) with n=50 independent paired observations.
    """
    nh_acts = []
    cond_acts = []

    for (q_idx, fmt), conditions in groups.items():
        if fmt != hint_format or condition_key not in conditions:
            continue

        nh_entry = no_hint_by_q.get(q_idx)
        cond_entry = conditions[condition_key]
        if nh_entry is None:
            continue

        nh_full = torch.load(activations_dir / f"{nh_entry['run_id']}.pt", weights_only=True)
        cond_full = torch.load(activations_dir / f"{cond_entry['run_id']}.pt", weights_only=True)

        nh_slice = nh_full[layer][nh_entry["prompt_length"]:].float()
        cond_slice = cond_full[layer][cond_entry["prompt_length"]:].float()

        del nh_full, cond_full

        # Skip zero-length slices (no generated tokens)
        if len(nh_slice) == 0 or len(cond_slice) == 0:
            continue

        nh_acts.append(nh_slice)
        cond_acts.append(cond_slice)

    return nh_acts, cond_acts


def main():
    results_dir = OUTPUTS_DIR / "sae_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)
    activations_dir = OUTPUTS_DIR / "activations"
    metadata_dir = OUTPUTS_DIR / "metadata"

    with open(metadata_dir / "generation_metadata.json") as f:
        metadata = json.load(f)

    from src.data import build_experiment_groups
    no_hint_by_q, groups = build_experiment_groups(metadata)

    # Analyze per-format to maintain n=50 independent paired observations
    false_hint_results = {}
    for hint_format in HINT_FORMATS:
        false_hint_results[hint_format] = []
        print(f"\nAnalyzing no-hint vs false-hint ({hint_format}, n=50)...")
        for layer in tqdm(range(N_LAYERS), desc=f"Layers ({hint_format})"):
            nh_acts, fh_acts = load_layer_activations_by_format(
                activations_dir, groups, no_hint_by_q, layer, "false_hint", hint_format
            )
            for width_k in SAE_WIDTHS:
                result = analyze_layer_width(layer, width_k, nh_acts, fh_acts)
                result["hint_format"] = hint_format
                false_hint_results[hint_format].append(result)
            del nh_acts, fh_acts

    # Analyze true-hint control (per-format)
    true_hint_results = {}
    for hint_format in HINT_FORMATS:
        true_hint_results[hint_format] = []
        print(f"\nAnalyzing no-hint vs true-hint ({hint_format}, n=50)...")
        for layer in tqdm(range(N_LAYERS), desc=f"Layers ({hint_format})"):
            nh_acts, th_acts = load_layer_activations_by_format(
                activations_dir, groups, no_hint_by_q, layer, "true_hint", hint_format
            )
            for width_k in SAE_WIDTHS:
                result = analyze_layer_width(layer, width_k, nh_acts, th_acts)
                result["hint_format"] = hint_format
                true_hint_results[hint_format].append(result)
            del nh_acts, th_acts

    with open(results_dir / "sae_results_false_hint.json", "w") as f:
        json.dump(false_hint_results, f, indent=2)
    with open(results_dir / "sae_results_true_hint.json", "w") as f:
        json.dump(true_hint_results, f, indent=2)

    # Summary
    print("\n=== Differential Feature Counts (false-hint, mean-pool) ===")
    for hint_format in HINT_FORMATS:
        print(f"\n--- {hint_format} ---")
        print(f"{'Layer':>6}", end="")
        for w in SAE_WIDTHS:
            print(f"  {w}k", end="")
        print()
        for layer in range(N_LAYERS):
            print(f"{layer:6d}", end="")
            for w in SAE_WIDTHS:
                entry = next(
                    (r for r in false_hint_results[hint_format]
                     if r["layer"] == layer and r["width_k"] == w and r.get("available", True)),
                    None,
                )
                count = entry["mean_pool"]["n_differential"] if entry else -1
                print(f"  {count:>4d}", end="")
            print()


if __name__ == "__main__":
    main()
