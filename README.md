# CoT-SAE: Analyzing Unfaithful Chain-of-Thought with Sparse Autoencoders

Research code for the divergence-localization experiment: a single-classifier transfer test for SAE feature specificity on Gemma 2 2B-it with Gemma Scope SAEs.

## Setup

1. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate cot_sae
   ```

2. Verify GPU access (for generation stages):
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

## Pipelines

Two independent pipelines. Both assume a SLURM cluster with an A100 (or similar) partition; edit `PARTITION` in the launch scripts for your setup. Both read/write under `outputs/`.

### 1. Layer selection sweep (`scripts/launch_pipeline.sh`)

Exploratory sweep over all Gemma 2 2B-it layers to pick a subset for the main experiment. Three chained SLURM jobs:

- **Stage 1 (GPU, ~6h)**: `run_baseline.py` + `run_generation.py` — run MMLU with no-hint / true-hint / false-hint and cache residual-stream activations.
- **Stage 2 (GPU, ~6h)**: `run_logit_lens.py` + `run_sae_analysis.py` — per-layer divergence via logit lens and SAE feature differences.
- **Stage 3 (CPU, ~30m)**: `run_comparison.py` — aggregate and recommend layers.

Launch:
```bash
bash scripts/launch_pipeline.sh
```

### 2. Divergence localization (`scripts/launch_divergence.sh`)

Main experiment: fractional-position SAE features, single-classifier transfer test. Two chained SLURM jobs:

- **Phase 1 (GPU array, 4 tasks, ~8h each)**: `run_divergence_generation.py` — generate 7 conditions per MMLU question and extract sparse SAE features at 6 layers × 2 widths × 20 fractional positions.
- **Phase 2 (CPU, 64 cores, up to 14 days)**: `run_divergence_analysis.py` — train per-configuration L2 logistic regression (hint-following false-hint vs no-hint), evaluate on matched true-hint, compute bootstrap CIs, save per-combo JSON + AUC figures.

Launch:
```bash
bash scripts/launch_divergence.sh
```

Phase 2 has resume capability: re-running after a timeout skips configurations that already have saved results in `outputs/divergence/results/per_combo/`.

## Tests

```bash
conda run -n cot_sae env PYTHONPATH=$(pwd) python -m pytest tests/ -q
```

## Config

Key constants live in `src/config.py`: model name, MMLU dataset, layer list, SAE widths, fractional sampling grid, and `MAX_NEW_TOKENS=4096`.
