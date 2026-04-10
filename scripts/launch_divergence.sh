#!/bin/bash
# ABOUTME: Launches the divergence localization pipeline as SLURM jobs.
# ABOUTME: Phase 1: array job (4 tasks, 1 H200 each). Phase 2: CPU job after Phase 1.

set -euo pipefail

PARTITION_GPU="gpu"
PARTITION_CPU="177huntington"
CONDA_ENV="cot_sae"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

# Create log directory
mkdir -p "$WORKDIR/outputs/divergence/logs"

echo "Working directory: $WORKDIR"
echo ""

# Phase 1: Data generation (GPU array job)
JOB1=$(sbatch --parsable \
    --partition=$PARTITION_GPU \
    --gres=gpu:1 \
    --time=08:00:00 \
    --mem=128G \
    --job-name=div-generate \
    --array=0-3 \
    --output=$WORKDIR/outputs/divergence/logs/phase1_%A_%a.log \
    --error=$WORKDIR/outputs/divergence/logs/phase1_%A_%a.err \
    --wrap="cd $WORKDIR && conda run -n $CONDA_ENV env PYTHONPATH=$WORKDIR python scripts/run_divergence_generation.py")

echo "Phase 1 (generation): Array job $JOB1 (4 tasks)"

# Phase 2: Analysis (CPU, depends on all Phase 1 tasks)
JOB2=$(sbatch --parsable \
    --dependency=afterok:$JOB1 \
    --partition=$PARTITION_CPU \
    --time=02:00:00 \
    --mem=256G \
    --job-name=div-analysis \
    --output=$WORKDIR/outputs/divergence/logs/phase2_%j.log \
    --error=$WORKDIR/outputs/divergence/logs/phase2_%j.err \
    --wrap="cd $WORKDIR && conda run -n $CONDA_ENV env PYTHONPATH=$WORKDIR python scripts/run_divergence_analysis.py")

echo "Phase 2 (analysis):  Job $JOB2 (depends on $JOB1)"

echo ""
echo "Monitor with: squeue -u \$USER"
