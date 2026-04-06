# ABOUTME: Extracts SAE features from cached activations using Gemma Scope SAEs.
# ABOUTME: Supports mean and max pooling, with paired t-test and FDR correction.

import torch
from src.config import SAE_RELEASE
from src.metrics import find_differential_features

try:
    from sae_lens import SAE
except ImportError:
    SAE = None


def load_sae(layer: int, width_k: int):
    """Load a Gemma Scope SAE for a given layer and width.

    Note: the exact sae_id format may need adjustment based on available
    SAEs in the Gemma Scope registry. Check SAELens docs if this fails.
    """
    if SAE is None:
        raise ImportError("sae_lens is not installed")
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=f"layer_{layer}/width_{width_k}k/canonical",
    )
    return sae


def extract_sae_features(sae: SAE, residual: torch.Tensor) -> torch.Tensor:
    """Extract SAE feature activations from a residual stream."""
    with torch.no_grad():
        features = sae.encode(residual.to(sae.device))
    return features.cpu()


def pool_features(features: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Pool SAE features across token positions."""
    if method == "mean":
        return features.mean(dim=0)
    elif method == "max":
        return features.max(dim=0).values
    else:
        raise ValueError(f"Unknown pooling method: {method}")


def analyze_features(
    baseline: torch.Tensor,
    condition: torch.Tensor,
    q_threshold: float = 0.05,
) -> dict:
    """Run paired t-test with FDR on pooled feature activations."""
    result = find_differential_features(baseline, condition, q_threshold)
    return {
        "n_differential": len(result["feature_indices"]),
        "feature_indices": result["feature_indices"],
        "effect_sizes": result["effect_sizes"],
    }


def analyze_layer_width(
    layer: int,
    width_k: int,
    no_hint_activations: list[torch.Tensor],
    condition_activations: list[torch.Tensor],
    q_threshold: float = 0.05,
) -> dict:
    """Analyze differential SAE features for one layer/width using both pooling methods."""
    try:
        sae = load_sae(layer, width_k)
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"  WARNING: SAE not available for layer {layer}, width {width_k}k: {e}")
        return {
            "layer": layer,
            "width_k": width_k,
            "mean_pool": {"n_differential": 0, "feature_indices": [], "effect_sizes": []},
            "max_pool": {"n_differential": 0, "feature_indices": [], "effect_sizes": []},
            "available": False,
        }

    # Encode once, pool twice
    nh_features_list = [extract_sae_features(sae, act) for act in no_hint_activations]
    cond_features_list = [extract_sae_features(sae, act) for act in condition_activations]

    results_by_pool = {}
    for method in ["mean", "max"]:
        nh_pooled = [pool_features(f, method) for f in nh_features_list]
        cond_pooled = [pool_features(f, method) for f in cond_features_list]

        baseline = torch.stack(nh_pooled)
        condition = torch.stack(cond_pooled)
        results_by_pool[f"{method}_pool"] = analyze_features(baseline, condition, q_threshold)

    del sae
    torch.cuda.empty_cache()

    return {
        "layer": layer,
        "width_k": width_k,
        "available": True,
        **results_by_pool,
    }
