# Divergence Localization Design

## Goal

Sweep through the full CoT generation to measure how well a linear classifier can distinguish no-hint from false-hint SAE feature activations at each fraction of the sequence. The primary output is an AUC(fraction) curve per layer/width that shows how discriminability evolves over the course of generation. The divergence onset — where AUC first rises significantly above the true-hint control — identifies the earliest point where SAE features detect misleading hint influence.

## Background

The layer selection sweep (2026-04-04) identified layers 12, 14, 16, 17, 22, 25 and SAE widths 16k/65k as having the strongest differential SAE features between no-hint and false-hint conditions. This experiment uses those layers and widths for the full divergence localization analysis.

## Method

### Linear Discriminator

Train an L2-regularized logistic regression classifier to distinguish no-hint from false-hint SAE feature vectors. Rather than testing individual features (as in the sweep), the classifier detects signal distributed across many features simultaneously.

For each of 12 layer/width combinations (6 layers × 2 widths):
1. Extract SAE feature vectors at 20 fractional positions (5%, 10%, ..., 100% of generation length)
2. Train classifier on pooled data from all positions (shared classifier)
3. Evaluate AUC at each fraction on held-out questions
4. The AUC(fraction) curve shows how discriminability evolves

### Token Position Alignment

No-hint and false-hint conditions have different prompts and generate different text. There is no guaranteed shared prefix. Comparing at absolute token positions would confound internal hint processing with surface text differences.

We address this by:
- Sampling at **fractional positions** (percentage of total generation length) rather than absolute positions, so comparisons are at corresponding stages of reasoning
- Using two controls to isolate the hint-specific signal from confounds

### Controls

**True-hint control**: Compute AUC(fraction) for no-hint vs true-hint using the same classifier pipeline. Both true-hint and false-hint have the same confounding (different prompt, different text). The excess of false-hint AUC over true-hint AUC at each fraction isolates the misleading-hint signal from general "extra information" effects.

**Text similarity baseline**: Compute cosine similarity of sentence embeddings between no-hint and false-hint generated text at each fraction. If the AUC curve exceeds what text divergence alone would predict, the classifier detects more than surface text differences.

## Data

### Dataset

Full MMLU "all" split (~14,000 questions). Run all questions through Gemma 2 2B-it with no hint at temperature=0. Keep only correctly-answered questions (estimated ~3,000+ at ~24% accuracy observed in the sweep).

### Conditions

For each correctly-answered question, generate under 7 conditions:
- 1 no-hint (shared across hint formats)
- 3 true-hint (authority, metadata, peer)
- 3 false-hint (authority, metadata, peer)

Total forward passes: ~21,000+

### What We Cache Per Run

- Generated text and final answer
- SAE feature vectors at 6 layers (12, 14, 16, 17, 22, 25) sampled at 20 fractional positions (5%, 10%, ..., 100% of generation length)
- Two SAE widths: 16k and 65k
- Stored as sparse tensors (nonzero values + indices only)
- Metadata: prompt length, generation length, hint-following label, hint-mention label (keyword-based)

### Data Split

- 90/10 train/test split by question, stratified
- 5-fold CV within train set for regularization strength tuning
- All three hint formats pooled for classifier training
- Per-format breakdown computed post-hoc on test set

## Model and SAE Configuration

- **Model**: Gemma 2 2B-it (`google/gemma-2-2b-it`), 26 layers, hidden_dim 2304
- **SAEs**: Gemma Scope 2B base model SAEs, canonical release (`gemma-scope-2b-pt-res-canonical`)
- **Layers**: 12, 14, 16, 17, 22, 25 (union of top-5 from mean-pool and max-pool in the sweep)
- **Widths**: 16k, 65k (both, for comparison)
- **Generation**: temperature=0 (greedy), MAX_NEW_TOKENS=4096

## Pipeline

### Phase 1: Data Generation (GPU)

SLURM array job with 4 tasks on `gpu` partition, 1 H200 per task, 8-hour limit. Questions split into 4 chunks by array task ID.

Per task:
1. Load assigned chunk of MMLU questions
2. Run baseline (no-hint) generation, filter to correctly-answered
3. For each correct question, generate under 6 hint conditions (3 formats × 2 hint types)
4. Generation is batched at batch size 128 without early stopping
5. Post-hoc EOS detection to determine actual generation length per sequence
6. For each run, call `run_with_cache` caching only 6 layers
7. Compute 20 fractional positions from actual generation length
8. At each fractional position × layer, encode through SAE at both widths
9. Save sparse feature vectors and metadata, discard raw activations

### Phase 2: Classification and Analysis (CPU)

Single CPU job after Phase 1 completes.

1. Load all sparse feature vectors and metadata from all 4 chunks
2. Split questions 90/10 (train/test)
3. For each of 12 layer/width combinations:
   a. 5-fold CV on train set to tune L2 regularization strength
   b. Train final classifier on full train set with best regularization
   c. At each of 20 fractions, compute AUC on test set
4. Repeat for true-hint control (no-hint vs true-hint)
5. Compute text similarity baseline at each fraction
6. Generate outputs (see below)

## Output

### Primary

- **AUC(fraction) curves**: Per layer/width, for both false-hint and true-hint conditions. 12 curves (6 layers × 2 widths) per condition.
- **Divergence onset**: Per layer/width, the earliest fraction where false-hint AUC exceeds true-hint AUC by more than a threshold (e.g., 0.05 AUC difference sustained for 2+ consecutive fractions). Exact threshold determined during analysis.

### Secondary

- **Text similarity baseline curve**: Cosine similarity of sentence embeddings at each fraction.
- **Per-format AUC breakdown**: Test-set AUC by hint format (authority, metadata, peer) at each fraction.
- **Top classifier weights**: Which SAE features contribute most to discrimination at each layer/width.
- **Hint-following statistics**: Rates of hint-following and hint-mention across the full dataset.

## Compute

- **Phase 1**: 4 × H200 (141 GB HBM3e), `gpu` partition, array job, 8-hour limit per task. Batch size 128. Estimated 1-2 hours wall time.
- **Phase 2**: CPU only, minutes to run.

## Storage

- SAE features stored as sparse tensors (nonzero values + indices)
- Estimated 30-50 GB total for ~21,000 runs × 20 fractions × 6 layers × 2 widths

## Known Limitations

- **IT/base SAE mismatch**: SAEs trained on base model, applied to IT model activations. Google's Gemma Scope report indicates good transfer. Observed strong differential signal in the sweep.
- **Token alignment**: Fractional position sampling normalizes by generation length but does not guarantee semantic alignment. Two controls (true-hint, text similarity) mitigate but do not eliminate this confound.
- **Greedy decoding assumption**: Temperature=0 means the generated text is deterministic given the prompt. Results may not generalize to sampling-based generation.
- **Batched generation waste**: Without early stopping, sequences that finish early continue generating padding tokens. Acceptable trade-off for implementation simplicity and batch throughput.
