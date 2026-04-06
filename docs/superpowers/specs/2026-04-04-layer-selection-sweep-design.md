# Layer Selection Sweep Design

## Goal

Determine which layers of Gemma 2 2B and which SAE dictionary width are most effective for detecting hint-related divergence in internal representations. This informs layer and width selection for the full SAE-based divergence localization experiment.

## Two Signals

We collect two complementary signals to guide layer selection:

1. **Logit lens divergence** — where the hint shifts the model's output prediction (layer x token heatmap)
2. **SAE differential activation** — where the hint activates distinct internal features (per layer/width)

These measure different things and can disagree. Comparing them gives a more complete picture than either alone.

## Model and SAE Configuration

- **Model**: Gemma 2 2B-it (instruction-tuned, 26 layers)
- **SAEs**: Gemma Scope 2B base model SAEs (`gemma-scope-2b-pt-res`), residual stream, widths 16K/65K/131K
- **Known limitation**: SAEs trained on base model, applied to IT model activations. Google reports good transfer; revisit if signal is weak.

## Pipeline

### Stage 1: Data Generation

- Run 200 MMLU questions through Gemma 2 2B-it with no hint (using Gemma 2 chat template) to establish baseline correctness
- Select 50 correctly-answered questions
- Run those 50 under hint conditions:
  - 3 hint formats: authority-based, metadata, peer-influence
  - 3 hint conditions: no-hint, true-hint, false-hint
  - No-hint is format-independent — generated ONCE per question, shared across formats
  - Total: 50 no-hint + 150 true-hint + 150 false-hint = 350 forward passes
- Cache per run:
  - Generated CoT text and final answer
  - Residual stream activations at all 26 layers, all token positions
  - Hint-following label (answer changed toward hint)
  - Hint-mention label (keyword-based detection, not verbatim substring)
- Storage: `.pt` files per run, JSON metadata sidecar

### Stage 2: Logit Lens Analysis

- At each of 26 layers, project residual stream through **final LayerNorm + unembedding matrix** to get calibrated logit distributions
- For each question, compute token-by-token divergence between conditions:
  - Cosine distance on logit vectors
  - Jensen-Shannon divergence on logit distributions (numerically stable, clamped)
- **Two comparisons**: no-hint vs false-hint (main signal) AND no-hint vs true-hint (control)
- Aggregate using **masked averaging** (not zero-padding) to avoid bias from different sequence lengths
- Produce heatmaps per comparison: layer (26) x token position
- Separately aggregate by hint format
- Known limitation: token alignment is imperfect due to different prompt lengths; layer-mean divergence is reliable, positional patterns should be interpreted cautiously

### Stage 3: SAE Signal Analysis

- At each of the 26 layers, pass cached activations through Gemma Scope SAEs at three widths: 16K, 65K, 131K
- For each layer/width, compute differentially active features between conditions:
  - **Paired t-test** (not independent) since same questions appear in both conditions
  - **Benjamini-Hochberg FDR correction** to control false positives across thousands of features
  - **Cohen's d_z** (paired effect size) for ranking features
- **Two pooling methods**: mean-pool and max-pool over token positions (max-pool preserves features that fire at specific critical positions)
- **Two comparisons**: no-hint vs false-hint AND no-hint vs true-hint (control)
- Compare against logit lens heatmaps — do the two signals agree?

## Output

A recommendation of which 2-3 layers and which SAE width to use for the full divergence localization experiment in the next development cycle. Uses rank-aggregation across cosine distance, JSD, and SAE feature counts.

## Metrics

- Cosine distance (logit lens): per layer, per token position
- Jensen-Shannon divergence (logit lens): per layer, per token position
- Differentially active feature count after FDR correction (SAE): per layer, per width, per pooling method
- Top differential features and effect sizes (SAE): per layer, per width

## Scope

This is a layer/width selection sweep only. Full divergence localization (finding the earliest token where SAE features diverge) is deferred to the next cycle.

## Compute

- Single GPU (A100 80GB or equivalent)
- Libraries: TransformerLens, SAELens, datasets, torch
