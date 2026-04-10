# scripts/run_divergence_analysis.py
# ABOUTME: Phase 2 of divergence localization — trains classifiers and computes AUC curves.
# ABOUTME: Includes true-hint control, text similarity baseline, bootstrap CIs, and visualization.

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import (
    DIVERGENCE_DIR, SELECTED_LAYERS, SAE_WIDTHS, N_FRACTIONS,
    FRACTION_POINTS, HINT_FORMATS,
)
from src.fractional import from_sparse_features
from src.classifier import (
    tune_regularization, train_classifier,
    compute_auc_per_fraction, compute_bootstrap_ci,
)
from src.text_similarity import compute_text_similarity_curve
from src.data import build_experiment_groups

SAE_ACTUAL_FEATURES = {16: 16384, 65: 65536}


def load_all_metadata(metadata_dir: Path) -> list[dict]:
    """Load and merge metadata from all array task files."""
    all_metadata = []
    for path in sorted(metadata_dir.glob("metadata_task*.json")):
        with open(path) as f:
            all_metadata.extend(json.load(f))
    return all_metadata


def load_run_features(features_dir: Path, run_id: str, layer_width_key: str) -> list[dict]:
    """Load sparse SAE features for one run at one layer/width."""
    data = torch.load(features_dir / f"{run_id}.pt", weights_only=True)
    return data[layer_width_key]


def load_all_run_features(features_dir: Path, run_id: str) -> dict:
    """Load all sparse SAE features for one run (all layer/width keys)."""
    return torch.load(features_dir / f"{run_id}.pt", weights_only=True)


def build_paired_features(
    question_data: list[dict],
    n_fractions: int,
    n_features: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Build paired no-hint vs condition feature arrays from sparse data.

    Each question's no-hint vector is paired once per format. The same no-hint
    features appear in up to 3 pairings when formats are pooled. This is
    acceptable because the classifier treats each pairing as an observation,
    and GroupKFold ensures all pairings from the same question stay together.
    """
    n_questions = len(question_data)
    features_by_fraction = {}
    y = np.zeros(2 * n_questions, dtype=int)
    groups = np.zeros(2 * n_questions, dtype=int)

    for f in range(n_fractions):
        X = np.zeros((2 * n_questions, n_features), dtype=np.float32)
        for i, qd in enumerate(question_data):
            nh_dense = from_sparse_features(qd["no_hint"][f], total_features=n_features)
            cond_dense = from_sparse_features(qd["condition"][f], total_features=n_features)
            X[2 * i] = nh_dense.numpy()
            X[2 * i + 1] = cond_dense.numpy()
            if f == 0:
                y[2 * i] = 0
                y[2 * i + 1] = 1
                groups[2 * i] = qd["question_id"]
                groups[2 * i + 1] = qd["question_id"]
        features_by_fraction[f] = X

    return features_by_fraction, y, groups


def compute_divergence_onset(
    false_auc: list[float],
    true_auc: list[float],
    threshold: float = 0.05,
    sustained: int = 2,
) -> int | None:
    """Find the earliest fraction where false AUC exceeds true AUC by threshold.

    Requires the gap to be sustained for `sustained` consecutive fractions.
    """
    gaps = [f - t for f, t in zip(false_auc, true_auc)]
    consecutive = 0
    for i, gap in enumerate(gaps):
        if gap >= threshold:
            consecutive += 1
            if consecutive >= sustained:
                return i - sustained + 1
        else:
            consecutive = 0
    return None


def plot_auc_curves(
    false_auc, true_auc, fractions, title, save_path,
    text_sim=None, ci_lower=None, ci_upper=None,
):
    """Plot AUC(fraction) curves with controls and optional confidence band."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(fractions, false_auc, "r-o", label="False-hint vs No-hint", linewidth=2)
    ax1.plot(fractions, true_auc, "b-s", label="True-hint vs No-hint", linewidth=2)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")

    if ci_lower is not None and ci_upper is not None:
        fractions_arr = np.array(fractions)
        true_arr = np.array(true_auc)
        ax1.fill_between(
            fractions_arr,
            true_arr + np.array(ci_lower),
            true_arr + np.array(ci_upper),
            alpha=0.2, color="red", label="95% CI on AUC gap",
        )

    ax1.set_xlabel("Fraction of Generation")
    ax1.set_ylabel("AUC")
    ax1.set_ylim(0.4, 1.05)
    ax1.set_title(title)
    ax1.legend(loc="upper left")

    if text_sim is not None:
        ax2 = ax1.twinx()
        ax2.plot(fractions, text_sim, "g--^", label="Text similarity", alpha=0.6)
        ax2.set_ylabel("Text Cosine Similarity")
        ax2.set_ylim(0, 1.1)
        ax2.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    features_dir = DIVERGENCE_DIR / "features"
    metadata_dir = DIVERGENCE_DIR / "metadata"
    results_dir = DIVERGENCE_DIR / "results"
    figures_dir = DIVERGENCE_DIR / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    metadata = load_all_metadata(metadata_dir)
    no_hint_by_q, groups = build_experiment_groups(metadata)

    question_ids = sorted(no_hint_by_q.keys())
    n_test = max(1, len(question_ids) // 10)
    rng = np.random.RandomState(42)
    rng.shuffle(question_ids)
    test_ids = set(question_ids[:n_test])
    train_ids = set(question_ids[n_test:])
    print(f"Train: {len(train_ids)} questions, Test: {len(test_ids)} questions")

    all_results = {}

    # Preload all feature files once to avoid repeated disk I/O
    print("Preloading feature files...")
    feature_cache = {}  # {run_id: {layer_width_key: [sparse_list]}}
    run_ids_to_load = set()
    for q_id in question_ids:
        nh_entry = no_hint_by_q.get(q_id)
        if nh_entry:
            run_ids_to_load.add(nh_entry["run_id"])
        for fmt in HINT_FORMATS:
            group_key = (q_id, fmt)
            if group_key not in groups:
                continue
            for condition in ["false_hint", "true_hint"]:
                if condition in groups[group_key]:
                    run_ids_to_load.add(groups[group_key][condition]["run_id"])

    from tqdm import tqdm
    from joblib import Parallel, delayed
    sorted_run_ids = sorted(run_ids_to_load)
    print(f"  Loading {len(sorted_run_ids)} files in parallel...")
    loaded = Parallel(n_jobs=-1)(
        delayed(load_all_run_features)(features_dir, rid) for rid in sorted_run_ids
    )
    feature_cache = dict(zip(sorted_run_ids, loaded))
    del loaded
    print(f"Loaded {len(feature_cache)} feature files")

    # Load any previously saved per-combo results (for resuming)
    combo_results_dir = results_dir / "per_combo"
    combo_results_dir.mkdir(parents=True, exist_ok=True)
    for combo_file in combo_results_dir.glob("*.json"):
        try:
            with open(combo_file) as f:
                saved = json.load(f)
                all_results[combo_file.stem] = saved
        except (json.JSONDecodeError, KeyError):
            print(f"  WARNING: corrupted combo file {combo_file.name}, will recompute")
            combo_file.unlink()

    for layer in SELECTED_LAYERS:
        for width_k in SAE_WIDTHS:
            key = f"L{layer}_W{width_k}k"
            n_features = SAE_ACTUAL_FEATURES[width_k]

            if key in all_results:
                print(f"\n=== {key} === (already computed, skipping)")
                continue

            print(f"\n=== {key} ===")

            train_pairs_false = []
            test_pairs_false = []
            train_pairs_true = []
            test_pairs_true = []

            for q_id in question_ids:
                nh_entry = no_hint_by_q.get(q_id)
                if not nh_entry:
                    continue
                nh_features = feature_cache[nh_entry["run_id"]][key]

                for fmt in HINT_FORMATS:
                    group_key = (q_id, fmt)
                    if group_key not in groups:
                        continue

                    for condition, target_train, target_test in [
                        ("false_hint", train_pairs_false, test_pairs_false),
                        ("true_hint", train_pairs_true, test_pairs_true),
                    ]:
                        if condition not in groups[group_key]:
                            continue
                        cond_entry = groups[group_key][condition]
                        cond_features = feature_cache[cond_entry["run_id"]][key]
                        pair = {
                            "question_id": q_id,
                            "no_hint": nh_features,
                            "condition": cond_features,
                            "hint_format": fmt,
                        }
                        if q_id in train_ids:
                            target_train.append(pair)
                        else:
                            target_test.append(pair)

            print(f"  False-hint: {len(train_pairs_false)} train, {len(test_pairs_false)} test pairs")
            train_feats, train_y, train_groups = build_paired_features(
                train_pairs_false, N_FRACTIONS, n_features
            )

            # Tune regularization on the last fraction (strongest signal, avoids pooling all fractions)
            print("  Tuning regularization...")
            best_C = tune_regularization(train_feats[N_FRACTIONS - 1], train_y, train_groups)
            print(f"  Best C: {best_C}")

            print("  Training classifier...")
            # Train on the last fraction too (consistent with tuning)
            clf = train_classifier(train_feats[N_FRACTIONS - 1], train_y, C=best_C)
            del train_feats

            print("  Computing AUC per fraction (false-hint)...")
            test_feats_false, test_y_false, _ = build_paired_features(
                test_pairs_false, N_FRACTIONS, n_features
            )
            false_auc = compute_auc_per_fraction(clf, test_feats_false, test_y_false, N_FRACTIONS)

            print("  Computing true-hint control...")
            train_feats_true, train_y_true, train_groups_true = build_paired_features(
                train_pairs_true, N_FRACTIONS, n_features
            )
            # Tune and train on the last fraction (consistent with false-hint approach)
            best_C_true = tune_regularization(train_feats_true[N_FRACTIONS - 1], train_y_true, train_groups_true)
            clf_true = train_classifier(train_feats_true[N_FRACTIONS - 1], train_y_true, C=best_C_true)
            del train_feats_true

            test_feats_true, test_y_true, _ = build_paired_features(
                test_pairs_true, N_FRACTIONS, n_features
            )
            true_auc = compute_auc_per_fraction(clf_true, test_feats_true, test_y_true, N_FRACTIONS)

            print("  Computing bootstrap CIs...")
            false_qids_set = {p["question_id"] for p in test_pairs_false}
            true_qids_set = {p["question_id"] for p in test_pairs_true}
            shared_qids = sorted(false_qids_set & true_qids_set)

            shared_pairs_false = [p for p in test_pairs_false if p["question_id"] in shared_qids]
            shared_pairs_true = [p for p in test_pairs_true if p["question_id"] in shared_qids]
            assert len(shared_pairs_false) == len(shared_pairs_true), \
                f"Bootstrap CI alignment: false ({len(shared_pairs_false)}) != true ({len(shared_pairs_true)})"
            shared_feats_false, shared_y_false, _ = build_paired_features(
                shared_pairs_false, N_FRACTIONS, n_features
            )
            shared_feats_true, shared_y_true, _ = build_paired_features(
                shared_pairs_true, N_FRACTIONS, n_features
            )
            shared_qids_arr = np.array([qid for p in shared_pairs_false for qid in [p["question_id"]] * 2])

            ci_results = []
            for f in range(N_FRACTIONS):
                probs_f = clf.predict_proba(shared_feats_false[f])[:, 1]
                probs_t = clf_true.predict_proba(shared_feats_true[f])[:, 1]
                lower, upper = compute_bootstrap_ci(
                    probs_f, shared_y_false, probs_t, shared_y_true,
                    shared_qids_arr,
                )
                ci_results.append((lower, upper))
            del shared_feats_false, shared_feats_true
            ci_lower = [c[0] for c in ci_results]
            ci_upper = [c[1] for c in ci_results]

            onset = compute_divergence_onset(false_auc, true_auc)

            top_k = 20
            weights = np.abs(clf.coef_[0])
            top_indices = np.argsort(weights)[-top_k:][::-1]
            top_features = [(int(idx), float(weights[idx])) for idx in top_indices]

            per_format_auc = {}
            for fmt in HINT_FORMATS:
                fmt_test_pairs = [p for p in test_pairs_false if p["hint_format"] == fmt]
                if fmt_test_pairs:
                    fmt_feats, fmt_y, _ = build_paired_features(
                        fmt_test_pairs, N_FRACTIONS, n_features
                    )
                    fmt_auc = compute_auc_per_fraction(clf, fmt_feats, fmt_y, N_FRACTIONS)
                    per_format_auc[fmt] = fmt_auc

            all_results[key] = {
                "false_auc": false_auc,
                "true_auc": true_auc,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "onset_fraction_idx": onset,
                "onset_fraction": FRACTION_POINTS[onset] if onset is not None else None,
                "best_C": best_C,
                "top_features": top_features,
                "per_format_auc": per_format_auc,
            }

            plot_auc_curves(
                false_auc, true_auc, FRACTION_POINTS,
                f"AUC Curve — Layer {layer}, {width_k}k SAE",
                figures_dir / f"auc_L{layer}_W{width_k}k.png",
                ci_lower=ci_lower,
                ci_upper=ci_upper,
            )
            # Save per-combo results for resuming
            with open(combo_results_dir / f"{key}.json", "w") as f:
                json.dump(all_results[key], f, indent=2)

            print(f"  False AUC: {[f'{a:.3f}' for a in false_auc]}")
            print(f"  True AUC:  {[f'{a:.3f}' for a in true_auc]}")
            print(f"  Onset: {FRACTION_POINTS[onset] if onset is not None else 'none'}")

            del test_feats_false, test_feats_true

    print("\nComputing text similarity baseline...")
    text_sims = []
    for q_id in test_ids:
        nh_entry = no_hint_by_q.get(q_id)
        if not nh_entry:
            continue
        for fmt in HINT_FORMATS:
            group_key = (q_id, fmt)
            if group_key not in groups or "false_hint" not in groups[group_key]:
                continue
            fh_entry = groups[group_key]["false_hint"]
            sim_curve = compute_text_similarity_curve(
                nh_entry["response"], fh_entry["response"], FRACTION_POINTS
            )
            text_sims.append(sim_curve)

    avg_text_sim = np.mean(text_sims, axis=0).tolist() if text_sims else [0.0] * N_FRACTIONS

    output = {
        "layer_width_results": all_results,
        "text_similarity_baseline": avg_text_sim,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "fraction_points": FRACTION_POINTS,
    }
    with open(results_dir / "divergence_results.json", "w") as f:
        json.dump(output, f, indent=2)

    best_key = min(all_results, key=lambda k: all_results[k]["onset_fraction_idx"]
                   if all_results[k]["onset_fraction_idx"] is not None else 999)
    best = all_results[best_key]
    plot_auc_curves(
        best["false_auc"], best["true_auc"], FRACTION_POINTS,
        f"Divergence Localization — {best_key}",
        figures_dir / "auc_best_with_text.png",
        text_sim=avg_text_sim,
    )

    print("\n=== DIVERGENCE ONSET SUMMARY ===")
    for key in sorted(all_results):
        r = all_results[key]
        onset_str = f"{r['onset_fraction']:.0%}" if r["onset_fraction"] else "none"
        print(f"  {key}: onset at {onset_str}")

    total_false = sum(1 for m in metadata if m["condition"] == "false_hint")
    hint_following = sum(1 for m in metadata if m.get("hint_following", False))
    mentions = sum(1 for m in metadata if m.get("mentions_hint", False) and m["condition"] == "false_hint")
    print(f"\nHint-following rate: {hint_following}/{total_false} ({hint_following/max(total_false,1):.1%})")
    print(f"Hint-mention rate: {mentions}/{total_false} ({mentions/max(total_false,1):.1%})")

    print(f"\nResults saved to {results_dir / 'divergence_results.json'}")
    print(f"Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
