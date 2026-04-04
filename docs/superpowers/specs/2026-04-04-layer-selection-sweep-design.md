# Layer Selection Sweep Design

## Goal

Determine which layers of Gemma 2 2B and which SAE dictionary width are most effective for detecting hint-related divergence in internal representations. This informs layer and width selection for the full SAE-based divergence localization experiment.

## Two Signals

We collect two complementary signals to guide layer selection:

1. **Logit lens divergence** — where the hint shifts the model's output prediction (layer x token heatmap)
2. **SAE differential activation** — where the hint activates distinct internal features (per layer/width)

These measure different things and can disagree. Comparing them gives a more complete picture than either alone.

## Pipeline

### Stage 1: Data Generation

- Run 200 MMLU questions through Gemma 2 2B with no hint to establish baseline correctness
- Select 50 correctly-answered questions
- Run those 50 under 9 conditions each:
  - 3 hint formats: authority-based, metadata, peer-influence
  - 3 hint conditions: no-hint, true-hint, false-hint
  - Total: 450 forward passes
- Cache per run:
  - Generated CoT text and final answer
  - Residual stream activations at all 26 layers, all token positions (shape: `[26, T, hidden_dim]`)
  - Hint-following label (answer changed toward hint)
  - CoT-mentions-hint label
- Storage: `.pt` files per question, JSON metadata sidecar

### Stage 2: Logit Lens Analysis

- At each of 26 layers, project residual stream through the unembedding matrix to get logit distributions
- For each question, compute token-by-token divergence between no-hint and false-hint conditions:
  - Cosine distance on logit vectors
  - Jensen-Shannon divergence on logit distributions
- Aggregate across questions to produce two heatmaps (one per metric): layer (26) x token position, showing mean divergence
- Separately aggregate by hint format (authority, metadata, peer-influence) to check if divergence patterns differ across formats
- Identify layer(s) where divergence first appears

### Stage 3: SAE Signal Analysis

- At each of the 26 layers, pass cached activations through Gemma Scope SAEs at three widths: 16K, 65K, 131K
- For each layer/width, compute differentially active features between no-hint and false-hint conditions (t-test per feature, effect size threshold)
- Per layer/width summary: count of differentially active features, top features by effect size
- Compare against logit lens heatmaps — do the two signals agree?
- Look up top differential features on Neuronpedia for interpretability sanity checks

## Output

A recommendation of which 2-3 layers and which SAE width to use for the full divergence localization experiment in the next development cycle.

## Metrics

- Cosine distance (logit lens): per layer, per token position
- Jensen-Shannon divergence (logit lens): per layer, per token position
- Differentially active feature count (SAE): per layer, per width
- Top differential features and their Neuronpedia labels (SAE): per layer, per width

## Scope

This is a layer/width selection sweep only. Full divergence localization (finding the earliest token where SAE features diverge) is deferred to the next cycle.

## Model and Infrastructure

- Model: Gemma 2 2B (26 layers)
- SAEs: Gemma Scope (residual stream, widths 16K/65K/131K)
- Compute: single GPU (A100 or smaller sufficient for 2B)
- Libraries: TransformerLens, SAELens, datasets, torch
