#!/bin/bash
# ABOUTME: Launches the full layer selection sweep pipeline as chained SLURM jobs.
# ABOUTME: Three stages: (1) data generation, (2) analysis, (3) comparison.

set -euo pipefail

PARTITION="177huntington"
CONDA_ENV="cot_sae"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Working directory: $WORKDIR"
echo "Partition: $PARTITION"
echo "Conda env: $CONDA_ENV"
echo ""

# Stage 1: Baseline + Generation (GPU)
JOB1=$(sbatch --parsable \
    --partition=$PARTITION \
    --gres=gpu:a100:1 \
    --time=06:00:00 \
    --mem=64G \
    --job-name=sweep-generate \
    --output=$WORKDIR/outputs/logs/stage1_%j.log \
    --error=$WORKDIR/outputs/logs/stage1_%j.err \
    --wrap="cd $WORKDIR && conda run -n $CONDA_ENV python scripts/run_baseline.py && conda run -n $CONDA_ENV python scripts/run_generation.py")

echo "Stage 1 (baseline + generation): Job $JOB1"

# Stage 2: Logit lens + SAE analysis (GPU)
JOB2=$(sbatch --parsable \
    --dependency=afterok:$JOB1 \
    --partition=$PARTITION \
    --gres=gpu:a100:1 \
    --time=06:00:00 \
    --mem=64G \
    --job-name=sweep-analysis \
    --output=$WORKDIR/outputs/logs/stage2_%j.log \
    --error=$WORKDIR/outputs/logs/stage2_%j.err \
    --wrap="cd $WORKDIR && conda run -n $CONDA_ENV python scripts/run_logit_lens.py && conda run -n $CONDA_ENV python scripts/run_sae_analysis.py")

echo "Stage 2 (logit lens + SAE):       Job $JOB2 (depends on $JOB1)"

# Stage 3: Comparison + visualization (CPU only)
JOB3=$(sbatch --parsable \
    --dependency=afterok:$JOB2 \
    --partition=$PARTITION \
    --time=00:30:00 \
    --mem=16G \
    --job-name=sweep-comparison \
    --output=$WORKDIR/outputs/logs/stage3_%j.log \
    --error=$WORKDIR/outputs/logs/stage3_%j.err \
    --wrap="cd $WORKDIR && conda run -n $CONDA_ENV python scripts/run_comparison.py")

echo "Stage 3 (comparison):             Job $JOB3 (depends on $JOB2)"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $WORKDIR/outputs/logs/stage1_${JOB1}.log"
