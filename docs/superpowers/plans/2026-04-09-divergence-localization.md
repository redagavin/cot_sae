# Divergence Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that trains linear classifiers on SAE features at fractional positions to produce AUC(fraction) curves showing when hint-related divergence becomes detectable in the model's internals.

**Architecture:** Two-phase pipeline. Phase 1 (GPU): batched generation over full MMLU, SAE encoding at fractional positions, sparse feature storage. Phase 2 (CPU): logistic regression with CV, AUC curves, controls, visualization. Reuses existing `src/data.py`, `src/generate.py`, `src/sae_analysis.py` modules.

**Tech Stack:** TransformerLens, SAELens, scikit-learn (LogisticRegression, cross_val_score, roc_auc_score), torch, scipy, matplotlib

---

### Task 1: Update Configuration

**Files:**
- Modify: `src/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# In tests/test_config.py, add:

def test_divergence_localization_config():
    from src.config import (
        SELECTED_LAYERS, N_FRACTIONS, FRACTION_POINTS,
        BATCH_SIZE, DIVERGENCE_DIR,
    )
    assert SELECTED_LAYERS == [12, 14, 16, 17, 22, 25]
    assert N_FRACTIONS == 20
    assert len(FRACTION_POINTS) == 20
    assert FRACTION_POINTS[0] == 0.05
    assert FRACTION_POINTS[-1] == 1.0
    assert BATCH_SIZE == 128
    assert DIVERGENCE_DIR.name == "divergence"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_divergence_localization_config -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Write minimal implementation**

Add to `src/config.py`:

```python
# Divergence localization
SELECTED_LAYERS = [12, 14, 16, 17, 22, 25]
N_FRACTIONS = 20
FRACTION_POINTS = [i / N_FRACTIONS for i in range(1, N_FRACTIONS + 1)]
BATCH_SIZE = 128
DIVERGENCE_DIR = OUTPUTS_DIR / "divergence"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_divergence_localization_config -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: add divergence localization config constants"
```

---

### Task 2: Fractional Sampling and Sparse Storage Utilities

**Files:**
- Create: `src/fractional.py`
- Create: `tests/test_fractional.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fractional.py
# ABOUTME: Tests for fractional position sampling and sparse feature storage.
# ABOUTME: Validates index computation, sparse round-trip, and EOS detection.

import pytest
import torch
from src.fractional import (
    compute_fraction_indices,
    to_sparse_features,
    from_sparse_features,
    find_eos_position,
)


class TestComputeFractionIndices:
    def test_basic_fractions(self):
        # 100 generated tokens, 20 fractions
        indices = compute_fraction_indices(gen_length=100, n_fractions=20)
        assert len(indices) == 20
        assert indices[0] == 4    # 5% of 100 = 5, index 4 (0-based)
        assert indices[-1] == 99  # 100% of 100 = 100, index 99

    def test_short_sequence(self):
        # 10 tokens, some fractions will map to same index
        indices = compute_fraction_indices(gen_length=10, n_fractions=20)
        assert len(indices) == 20
        assert all(0 <= i < 10 for i in indices)

    def test_single_token(self):
        indices = compute_fraction_indices(gen_length=1, n_fractions=20)
        assert all(i == 0 for i in indices)

    def test_prompt_offset(self):
        # With prompt_length, indices shift by prompt_length
        indices = compute_fraction_indices(gen_length=100, n_fractions=20, prompt_length=50)
        assert indices[0] == 54   # 50 + 4
        assert indices[-1] == 149 # 50 + 99


class TestSparseFeatures:
    def test_round_trip(self):
        dense = torch.zeros(16384)
        dense[10] = 1.5
        dense[100] = 0.3
        dense[5000] = 2.1
        sparse = to_sparse_features(dense)
        recovered = from_sparse_features(sparse, total_features=16384)
        assert torch.allclose(dense, recovered)

    def test_empty_features(self):
        dense = torch.zeros(16384)
        sparse = to_sparse_features(dense)
        assert sparse["indices"].shape[0] == 0
        assert sparse["values"].shape[0] == 0
        recovered = from_sparse_features(sparse, total_features=16384)
        assert torch.allclose(dense, recovered)

    def test_preserves_values(self):
        dense = torch.zeros(65536)
        dense[42] = 3.14
        sparse = to_sparse_features(dense)
        assert 42 in sparse["indices"]
        assert 3.14 in sparse["values"].tolist()


class TestFindEosPosition:
    def test_finds_eos(self):
        tokens = torch.tensor([5, 10, 15, 1, 0, 0])  # eos_token_id=1
        pos = find_eos_position(tokens, eos_token_id=1, prompt_length=0)
        assert pos == 3

    def test_no_eos(self):
        tokens = torch.tensor([5, 10, 15, 20])
        pos = find_eos_position(tokens, eos_token_id=1, prompt_length=0)
        assert pos == 4  # full length

    def test_with_prompt(self):
        tokens = torch.tensor([5, 10, 15, 1, 0])  # prompt is first 2 tokens
        pos = find_eos_position(tokens, eos_token_id=1, prompt_length=2)
        assert pos == 3  # EOS at position 3, gen_length = 3 - 2 = 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fractional.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/fractional.py
# ABOUTME: Utilities for fractional position sampling and sparse SAE feature storage.
# ABOUTME: Converts between dense and sparse representations for storage efficiency.

import torch


def compute_fraction_indices(
    gen_length: int,
    n_fractions: int = 20,
    prompt_length: int = 0,
) -> list[int]:
    """Compute token indices corresponding to fractional positions of generation.

    Returns n_fractions indices into the full sequence (prompt + generated).
    Each index corresponds to 1/n_fractions, 2/n_fractions, ..., 1.0 of the
    generated portion, offset by prompt_length.
    """
    indices = []
    for i in range(1, n_fractions + 1):
        frac = i / n_fractions
        gen_idx = min(int(frac * gen_length) - 1, gen_length - 1)
        gen_idx = max(gen_idx, 0)
        indices.append(prompt_length + gen_idx)
    return indices


def to_sparse_features(dense: torch.Tensor) -> dict:
    """Convert a dense feature vector to sparse representation (nonzero only)."""
    nonzero_mask = dense != 0
    indices = torch.where(nonzero_mask)[0].to(torch.int32)
    values = dense[nonzero_mask]
    return {"indices": indices, "values": values}


def from_sparse_features(sparse: dict, total_features: int) -> torch.Tensor:
    """Reconstruct a dense feature vector from sparse representation."""
    dense = torch.zeros(total_features)
    if len(sparse["indices"]) > 0:
        dense[sparse["indices"].long()] = sparse["values"]
    return dense


def find_eos_position(
    tokens: torch.Tensor,
    eos_token_id: int,
    prompt_length: int,
) -> int:
    """Find the position of the first EOS token after the prompt.

    Returns the index of the first EOS in the generated portion,
    or the total sequence length if no EOS is found.
    """
    generated = tokens[prompt_length:]
    eos_positions = (generated == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        return prompt_length + eos_positions[0].item()
    return len(tokens)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fractional.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fractional.py tests/test_fractional.py
git commit -m "feat: add fractional sampling and sparse storage utilities"
```

---

### Task 3: Batched Generation with Selective Caching

**Files:**
- Create: `src/batch_generate.py`
- Create: `tests/test_batch_generate.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_batch_generate.py
# ABOUTME: Tests for batched prompt preparation and generation result extraction.
# ABOUTME: Uses small tensors to validate padding, masking, and EOS detection.

import pytest
import torch
from src.batch_generate import (
    pad_prompts_left,
    extract_generation_lengths,
)


class TestPadPromptsLeft:
    def test_pads_to_max_length(self):
        token_lists = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
            torch.tensor([6, 7, 8, 9]),
        ]
        padded, mask, prompt_lengths = pad_prompts_left(token_lists, pad_token_id=0)
        assert padded.shape == (3, 4)
        assert mask.shape == (3, 4)
        assert prompt_lengths == [3, 2, 4]

    def test_left_padding(self):
        token_lists = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
        ]
        padded, mask, prompt_lengths = pad_prompts_left(token_lists, pad_token_id=0)
        # Shorter sequence should be left-padded
        assert padded[1, 0].item() == 0  # pad
        assert padded[1, 1].item() == 4
        assert padded[1, 2].item() == 5

    def test_attention_mask(self):
        token_lists = [
            torch.tensor([1, 2, 3]),
            torch.tensor([4, 5]),
        ]
        padded, mask, prompt_lengths = pad_prompts_left(token_lists, pad_token_id=0)
        assert mask[0].tolist() == [1, 1, 1]
        assert mask[1].tolist() == [0, 1, 1]


class TestExtractGenerationLengths:
    def test_finds_eos(self):
        # batch of 2, seq_len 6, prompt_lengths [2, 3]
        output_tokens = torch.tensor([
            [1, 2, 10, 11, 99, 0],  # EOS=99 at position 4
            [0, 3, 4, 10, 99, 0],   # EOS=99 at position 4
        ])
        gen_lengths = extract_generation_lengths(
            output_tokens, prompt_lengths=[2, 3], eos_token_id=99
        )
        assert gen_lengths == [2, 1]  # 4-2=2, 4-3=1

    def test_no_eos(self):
        output_tokens = torch.tensor([
            [1, 2, 10, 11, 12, 13],
        ])
        gen_lengths = extract_generation_lengths(
            output_tokens, prompt_lengths=[2], eos_token_id=99
        )
        assert gen_lengths == [4]  # 6-2=4, no EOS found
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_batch_generate.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/batch_generate.py
# ABOUTME: Batched prompt preparation and generation result extraction for H200 GPUs.
# ABOUTME: Handles left-padding, attention masks, and post-hoc EOS detection.

import torch
from src.fractional import find_eos_position


def pad_prompts_left(
    token_lists: list[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Left-pad a list of token tensors to the same length.

    Returns:
        padded: [batch, max_len] padded token tensor
        attention_mask: [batch, max_len] binary mask (1 = real token, 0 = pad)
        prompt_lengths: list of original lengths per sequence
    """
    prompt_lengths = [len(t) for t in token_lists]
    max_len = max(prompt_lengths)

    padded = torch.full((len(token_lists), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(token_lists), max_len, dtype=torch.long)

    for i, tokens in enumerate(token_lists):
        pad_len = max_len - len(tokens)
        padded[i, pad_len:] = tokens
        attention_mask[i, pad_len:] = 1

    return padded, attention_mask, prompt_lengths


def extract_generation_lengths(
    output_tokens: torch.Tensor,
    prompt_lengths: list[int],
    eos_token_id: int,
) -> list[int]:
    """Find actual generation length for each sequence in a batch.

    Scans for the first EOS token after the prompt in each sequence.
    Returns generation length (not including prompt) per sequence.
    """
    batch_size = output_tokens.shape[0]
    gen_lengths = []
    for i in range(batch_size):
        eos_pos = find_eos_position(
            output_tokens[i], eos_token_id, prompt_lengths[i]
        )
        gen_lengths.append(eos_pos - prompt_lengths[i])
    return gen_lengths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_batch_generate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/batch_generate.py tests/test_batch_generate.py
git commit -m "feat: add batched prompt preparation and generation length extraction"
```

---

### Task 4: SAE Encoding at Fractional Positions

**Files:**
- Create: `src/fractional_sae.py`
- Create: `tests/test_fractional_sae.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fractional_sae.py
# ABOUTME: Tests for SAE encoding at fractional positions with sparse output.
# ABOUTME: Uses mock SAE to validate encoding logic without GPU.

import pytest
import torch
from unittest.mock import MagicMock
from src.fractional_sae import encode_at_fractions


class TestEncodeAtFractions:
    def _make_mock_sae(self, output_dim):
        sae = MagicMock()
        sae.device = torch.device("cpu")
        def fake_encode(x):
            # Return sparse-ish features: first few dims active
            out = torch.zeros(x.shape[0], output_dim)
            out[:, 0] = 1.0
            out[:, 1] = 0.5
            return out
        sae.encode = fake_encode
        return sae

    def test_output_structure(self):
        sae = self._make_mock_sae(100)
        residuals = {12: torch.randn(50, 64)}  # 50 tokens, dim 64
        fraction_indices = [4, 9, 14, 19, 24]  # 5 fractions
        result = encode_at_fractions(sae, residuals, layer=12, fraction_indices=fraction_indices)
        assert len(result) == 5
        for entry in result:
            assert "indices" in entry
            assert "values" in entry

    def test_correct_positions_sampled(self):
        sae = self._make_mock_sae(100)
        # Make residuals with distinct values at each position
        residuals = {12: torch.zeros(20, 64)}
        residuals[12][10, 0] = 99.0  # mark position 10
        fraction_indices = [10]
        result = encode_at_fractions(sae, residuals, layer=12, fraction_indices=fraction_indices)
        assert len(result) == 1

    def test_sparse_output(self):
        sae = self._make_mock_sae(100)
        residuals = {12: torch.randn(50, 64)}
        fraction_indices = [24]
        result = encode_at_fractions(sae, residuals, layer=12, fraction_indices=fraction_indices)
        # Mock SAE produces 2 nonzero values
        assert result[0]["indices"].shape[0] == 2
        assert result[0]["values"].shape[0] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fractional_sae.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/fractional_sae.py
# ABOUTME: Encodes residual stream activations through SAE at specific fractional positions.
# ABOUTME: Returns sparse feature vectors for storage efficiency.

import torch
from src.fractional import to_sparse_features


def encode_at_fractions(
    sae,
    residuals: dict[int, torch.Tensor],
    layer: int,
    fraction_indices: list[int],
) -> list[dict]:
    """Encode residual stream through SAE at specific token positions.

    Args:
        sae: loaded SAE model
        residuals: {layer: [seq_len, hidden_dim]} residual stream tensors
        layer: which layer to extract from residuals
        fraction_indices: token positions to encode (absolute indices into sequence)

    Returns:
        list of sparse feature dicts, one per fraction index
    """
    layer_residuals = residuals[layer]
    positions = torch.tensor(fraction_indices, dtype=torch.long)
    selected = layer_residuals[positions]  # [n_fractions, hidden_dim]

    with torch.no_grad():
        features = sae.encode(selected.to(sae.device))  # [n_fractions, n_features]
    features = features.cpu()

    sparse_list = []
    for i in range(features.shape[0]):
        sparse_list.append(to_sparse_features(features[i]))

    return sparse_list
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fractional_sae.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fractional_sae.py tests/test_fractional_sae.py
git commit -m "feat: add SAE encoding at fractional positions with sparse output"
```

---

### Task 5: Linear Classifier and AUC Computation

**Files:**
- Create: `src/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_classifier.py
# ABOUTME: Tests for linear classifier training and AUC curve computation.
# ABOUTME: Uses synthetic data to validate cross-validation and per-fraction evaluation.

import pytest
import numpy as np
from src.classifier import (
    tune_regularization,
    train_classifier,
    compute_auc_per_fraction,
)


class TestTuneRegularization:
    def test_returns_best_C(self):
        rng = np.random.RandomState(42)
        # Linearly separable data
        X = np.vstack([rng.randn(100, 10) + 1, rng.randn(100, 10) - 1])
        y = np.array([0] * 100 + [1] * 100)
        best_C = tune_regularization(X, y, n_folds=5)
        assert isinstance(best_C, float)
        assert best_C > 0


class TestTrainClassifier:
    def test_returns_fitted_model(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(50, 10) + 2, rng.randn(50, 10) - 2])
        y = np.array([0] * 50 + [1] * 50)
        clf = train_classifier(X, y, C=1.0)
        preds = clf.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy > 0.9


class TestComputeAucPerFraction:
    def test_returns_correct_shape(self):
        rng = np.random.RandomState(42)
        n_questions = 20
        n_fractions = 5
        n_features = 10
        # features_by_fraction: {fraction_idx: {question_id: feature_vector}}
        features_by_fraction = {}
        labels = {}
        for f in range(n_fractions):
            features_by_fraction[f] = {}
            for q in range(n_questions):
                label = q % 2  # alternating labels
                offset = 2.0 if label == 1 else -2.0
                features_by_fraction[f][q] = rng.randn(n_features) + offset
                labels[q] = label

        clf = train_classifier(
            np.vstack([features_by_fraction[0][q] for q in range(n_questions)]),
            np.array([labels[q] for q in range(n_questions)]),
            C=1.0,
        )
        auc_curve = compute_auc_per_fraction(
            clf, features_by_fraction, labels, n_fractions
        )
        assert len(auc_curve) == n_fractions
        assert all(0.0 <= a <= 1.0 for a in auc_curve)

    def test_separable_data_high_auc(self):
        rng = np.random.RandomState(42)
        n_questions = 50
        n_fractions = 3
        n_features = 10
        features_by_fraction = {}
        labels = {}
        for f in range(n_fractions):
            features_by_fraction[f] = {}
            for q in range(n_questions):
                label = q % 2
                offset = 5.0 if label == 1 else -5.0
                features_by_fraction[f][q] = rng.randn(n_features) + offset
                labels[q] = label

        X_train = np.vstack([features_by_fraction[0][q] for q in range(n_questions)])
        y_train = np.array([labels[q] for q in range(n_questions)])
        clf = train_classifier(X_train, y_train, C=1.0)
        auc_curve = compute_auc_per_fraction(clf, features_by_fraction, labels, n_fractions)
        assert all(a > 0.9 for a in auc_curve)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/classifier.py
# ABOUTME: L2-regularized logistic regression for distinguishing hint conditions.
# ABOUTME: Includes CV-based regularization tuning and per-fraction AUC evaluation.

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score


def tune_regularization(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    C_values: list[float] | None = None,
) -> float:
    """Find the best L2 regularization strength via cross-validation.

    Args:
        X: [n_samples, n_features] training features
        y: [n_samples] binary labels
        n_folds: number of CV folds
        C_values: regularization strengths to try (inverse of lambda)

    Returns:
        best C value
    """
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    best_C = C_values[0]
    best_score = -1.0

    for C in C_values:
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring="roc_auc")
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score = mean_score
            best_C = C

    return best_C


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
) -> LogisticRegression:
    """Train an L2-regularized logistic regression classifier."""
    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
    clf.fit(X, y)
    return clf


def compute_auc_per_fraction(
    clf: LogisticRegression,
    features_by_fraction: dict[int, dict[int, np.ndarray]],
    labels: dict[int, int],
    n_fractions: int,
) -> list[float]:
    """Compute AUC at each fractional position using a trained classifier.

    Args:
        clf: trained classifier
        features_by_fraction: {fraction_idx: {question_id: feature_vector}}
        labels: {question_id: label}
        n_fractions: number of fraction points

    Returns:
        list of AUC values, one per fraction
    """
    auc_curve = []
    for f in range(n_fractions):
        fraction_data = features_by_fraction[f]
        question_ids = sorted(fraction_data.keys())
        X = np.vstack([fraction_data[q] for q in question_ids])
        y = np.array([labels[q] for q in question_ids])
        probs = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        auc_curve.append(auc)
    return auc_curve
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/classifier.py tests/test_classifier.py
git commit -m "feat: add logistic regression classifier with CV tuning and per-fraction AUC"
```

---

### Task 6: Text Similarity Baseline

**Files:**
- Create: `src/text_similarity.py`
- Create: `tests/test_text_similarity.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_text_similarity.py
# ABOUTME: Tests for text similarity computation at fractional positions.
# ABOUTME: Validates sentence embedding cosine similarity logic.

import pytest
from src.text_similarity import (
    text_at_fraction,
    compute_text_similarity_curve,
)


class TestTextAtFraction:
    def test_half_fraction(self):
        text = "one two three four five six seven eight nine ten"
        result = text_at_fraction(text, 0.5)
        words = result.split()
        assert len(words) <= 6  # roughly half

    def test_full_fraction(self):
        text = "hello world"
        result = text_at_fraction(text, 1.0)
        assert result == text

    def test_small_fraction(self):
        text = "a b c d e f g h i j"
        result = text_at_fraction(text, 0.1)
        assert len(result) > 0


class TestComputeTextSimilarityCurve:
    def test_identical_texts_high_similarity(self):
        text_a = "The answer is clearly option B because of the evidence."
        text_b = "The answer is clearly option B because of the evidence."
        fractions = [0.5, 1.0]
        curve = compute_text_similarity_curve(text_a, text_b, fractions)
        assert len(curve) == 2
        assert all(s > 0.99 for s in curve)

    def test_different_texts_lower_similarity(self):
        text_a = "I think the answer is B based on scientific evidence."
        text_b = "The professor suggested C so I will go with that."
        fractions = [0.5, 1.0]
        curve = compute_text_similarity_curve(text_a, text_b, fractions)
        assert len(curve) == 2
        assert all(0.0 <= s <= 1.0 for s in curve)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_text_similarity.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/text_similarity.py
# ABOUTME: Computes text-level cosine similarity between generation conditions at fractional positions.
# ABOUTME: Uses sentence-transformers for embedding-based similarity as a baseline control.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def text_at_fraction(text: str, fraction: float) -> str:
    """Extract the first `fraction` of a text by character count."""
    end = max(1, int(len(text) * fraction))
    return text[:end]


def compute_text_similarity_curve(
    text_a: str,
    text_b: str,
    fractions: list[float],
) -> list[float]:
    """Compute cosine similarity between two texts at each fractional position.

    Uses TF-IDF vectors for simplicity and to avoid loading a separate model.
    """
    similarities = []
    vectorizer = TfidfVectorizer()

    for frac in fractions:
        prefix_a = text_at_fraction(text_a, frac)
        prefix_b = text_at_fraction(text_b, frac)

        if not prefix_a.strip() or not prefix_b.strip():
            similarities.append(1.0)
            continue

        tfidf = vectorizer.fit_transform([prefix_a, prefix_b])
        sim = sklearn_cosine(tfidf[0:1], tfidf[1:2])[0, 0]
        similarities.append(float(sim))

    return similarities
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_text_similarity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/text_similarity.py tests/test_text_similarity.py
git commit -m "feat: add text similarity baseline using TF-IDF cosine similarity"
```

---

### Task 7: Phase 1 Script — Data Generation with SAE Encoding

**Files:**
- Create: `scripts/run_divergence_generation.py`
- Create: `tests/test_divergence_generation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_divergence_generation.py
# ABOUTME: Tests for the divergence generation script helper functions.
# ABOUTME: Validates chunk splitting and metadata assembly.

import pytest
from scripts.run_divergence_generation import split_into_chunks, build_run_metadata


class TestSplitIntoChunks:
    def test_even_split(self):
        items = list(range(12))
        chunks = split_into_chunks(items, n_chunks=4)
        assert len(chunks) == 4
        assert [len(c) for c in chunks] == [3, 3, 3, 3]

    def test_uneven_split(self):
        items = list(range(10))
        chunks = split_into_chunks(items, n_chunks=4)
        assert len(chunks) == 4
        total = sum(len(c) for c in chunks)
        assert total == 10

    def test_single_chunk(self):
        items = list(range(5))
        chunks = split_into_chunks(items, n_chunks=1)
        assert len(chunks) == 1
        assert len(chunks[0]) == 5


class TestBuildRunMetadata:
    def test_structure(self):
        meta = build_run_metadata(
            run_id="q001_authority_false_hint",
            question_idx=1,
            hint_format="authority",
            condition="false_hint",
            correct_answer=2,
            false_answer=0,
            predicted=0,
            response="The answer is A.",
            prompt_length=50,
            gen_length=100,
        )
        assert meta["run_id"] == "q001_authority_false_hint"
        assert meta["hint_following"] is True  # predicted == false_answer
        assert meta["gen_length"] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_divergence_generation.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write implementation**

```python
# scripts/run_divergence_generation.py
# ABOUTME: Phase 1 of divergence localization — generates CoT under all conditions.
# ABOUTME: Batched generation with SAE encoding at fractional positions, SLURM array job.

import json
import os
import sys
import torch
from tqdm import tqdm
from datasets import load_dataset

from src.config import (
    MMLU_DATASET, MMLU_SPLIT, HINT_FORMATS, ANSWER_LETTERS,
    MAX_NEW_TOKENS, SELECTED_LAYERS, SAE_WIDTHS, N_FRACTIONS,
    DIVERGENCE_DIR,
)
from src.data import (
    build_prompt, format_for_model, insert_hint, parse_answer,
    check_mentions_hint, format_choices,
)
from src.generate import load_model, pick_false_answer
from src.sae_analysis import load_sae
from src.batch_generate import pad_prompts_left, extract_generation_lengths
from src.fractional import compute_fraction_indices
from src.fractional_sae import encode_at_fractions


def split_into_chunks(items: list, n_chunks: int) -> list[list]:
    """Split a list into n roughly equal chunks."""
    k, m = divmod(len(items), n_chunks)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def build_run_metadata(
    run_id: str,
    question_idx: int,
    hint_format: str,
    condition: str,
    correct_answer: int,
    false_answer: int,
    predicted: int | None,
    response: str,
    prompt_length: int,
    gen_length: int,
) -> dict:
    """Assemble metadata for a single generation run."""
    return {
        "run_id": run_id,
        "question_idx": question_idx,
        "hint_format": hint_format,
        "condition": condition,
        "correct_answer": correct_answer,
        "false_answer": false_answer,
        "predicted": predicted,
        "response": response,
        "hint_following": (condition == "false_hint" and predicted == false_answer),
        "mentions_hint": check_mentions_hint(response, hint_format) if hint_format != "none" else False,
        "prompt_length": prompt_length,
        "gen_length": gen_length,
    }


def process_batch(
    model, tokenizer, saes, prompts_info, batch_size,
):
    """Process a batch of prompts: generate, cache activations, encode SAE features.

    Args:
        model: HookedTransformer model
        tokenizer: model tokenizer
        saes: dict of {(layer, width_k): sae} pre-loaded SAEs
        prompts_info: list of dicts with 'formatted_prompt' and metadata fields
        batch_size: max batch size

    Returns:
        list of result dicts with metadata and sparse SAE features
    """
    results = []

    for batch_start in range(0, len(prompts_info), batch_size):
        batch = prompts_info[batch_start:batch_start + batch_size]

        # Tokenize and pad
        token_lists = [
            model.to_tokens(info["formatted_prompt"], prepend_bos=False)[0]
            for info in batch
        ]
        pad_token_id = tokenizer.pad_token_id or 0
        padded, attention_mask, prompt_lengths = pad_prompts_left(token_lists, pad_token_id)
        padded = padded.to(model.cfg.device)

        # Generate
        output_tokens = model.generate(
            padded,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
        )

        # Find actual generation lengths
        eos_token_id = tokenizer.eos_token_id or 1
        gen_lengths = extract_generation_lengths(
            output_tokens, prompt_lengths, eos_token_id
        )

        # Forward pass to cache selected layers
        layer_filter = lambda name: any(
            f"blocks.{l}.hook_resid_post" in name for l in SELECTED_LAYERS
        )
        _, cache = model.run_with_cache(
            output_tokens.clone(),
            names_filter=layer_filter,
        )

        # Process each sequence in the batch
        for i, info in enumerate(batch):
            prompt_len = prompt_lengths[i]
            gen_len = max(gen_lengths[i], 1)

            # Decode generated text
            gen_tokens = output_tokens[i, prompt_len:prompt_len + gen_len]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            predicted = parse_answer(response)

            # Compute fraction indices
            fraction_indices = compute_fraction_indices(
                gen_length=gen_len, n_fractions=N_FRACTIONS, prompt_length=prompt_len
            )

            # Extract residuals for this sequence
            residuals = {}
            for layer in SELECTED_LAYERS:
                residuals[layer] = cache["resid_post", layer][i].detach()

            # SAE encode at fractional positions
            sae_features = {}
            for (layer, width_k), sae in saes.items():
                sparse_list = encode_at_fractions(
                    sae, residuals, layer=layer, fraction_indices=fraction_indices
                )
                sae_features[(layer, width_k)] = sparse_list

            # Build metadata
            meta = build_run_metadata(
                run_id=info["run_id"],
                question_idx=info["question_idx"],
                hint_format=info.get("hint_format", "none"),
                condition=info["condition"],
                correct_answer=info["correct_answer"],
                false_answer=info["false_answer"],
                predicted=predicted,
                response=response,
                prompt_length=prompt_len,
                gen_length=gen_len,
            )

            results.append({"metadata": meta, "sae_features": sae_features})

        # Clear cache
        del cache
        torch.cuda.empty_cache()

    return results


def main():
    # Determine array task
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    n_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    features_dir = DIVERGENCE_DIR / "features"
    metadata_dir = DIVERGENCE_DIR / "metadata"
    features_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Load full MMLU
    print(f"Task {task_id}/{n_tasks}: Loading MMLU...")
    ds = load_dataset(MMLU_DATASET, MMLU_SPLIT, split="test")
    all_questions = [ds[i] for i in range(len(ds))]

    # Split across array tasks
    chunks = split_into_chunks(all_questions, n_tasks)
    my_questions = chunks[task_id]
    # Compute global index offset
    offset = sum(len(chunks[i]) for i in range(task_id))
    print(f"Task {task_id}: processing questions {offset} to {offset + len(my_questions) - 1}")

    # Load model
    print("Loading model...")
    model = load_model()
    tokenizer = model.tokenizer

    # Load all SAEs
    print("Loading SAEs...")
    saes = {}
    for layer in SELECTED_LAYERS:
        for width_k in SAE_WIDTHS:
            saes[(layer, width_k)] = load_sae(layer, width_k)

    # Phase 1a: Baseline (no-hint) to find correctly answered questions
    print("Running baseline...")
    correct_questions = []
    baseline_metadata = []

    baseline_prompts = []
    for local_idx, q in enumerate(my_questions):
        global_idx = offset + local_idx
        user_msg = build_prompt(question=q["question"], choices=q["choices"], hint_text="")
        formatted = format_for_model(tokenizer, user_msg)
        baseline_prompts.append({
            "formatted_prompt": formatted,
            "question_idx": global_idx,
            "local_idx": local_idx,
            "condition": "no_hint",
            "hint_format": "none",
            "correct_answer": q["answer"],
            "false_answer": pick_false_answer(q["answer"], seed=global_idx),
            "run_id": f"q{global_idx:05d}_no_hint",
        })

    baseline_results = process_batch(model, tokenizer, saes, baseline_prompts, BATCH_SIZE)

    for result, prompt_info in zip(baseline_results, baseline_prompts):
        meta = result["metadata"]
        baseline_metadata.append(meta)
        if meta["predicted"] == meta["correct_answer"]:
            correct_questions.append({
                "global_idx": prompt_info["question_idx"],
                "local_idx": prompt_info["local_idx"],
                "question": my_questions[prompt_info["local_idx"]],
                "false_answer": meta["false_answer"],
                "baseline_result": result,
            })

    print(f"Task {task_id}: {len(correct_questions)}/{len(my_questions)} correct")

    # Phase 1b: Generate hint conditions for correct questions
    print("Running hint conditions...")
    all_results = []
    # Save baseline results for correct questions
    for cq in correct_questions:
        all_results.append(cq["baseline_result"])

    hint_prompts = []
    for cq in correct_questions:
        q = cq["question"]
        global_idx = cq["global_idx"]
        correct_idx = q["answer"]
        false_idx = cq["false_answer"]

        for hint_format in HINT_FORMATS:
            for condition in ["true_hint", "false_hint"]:
                hint_text = insert_hint(
                    condition=condition,
                    hint_format=hint_format,
                    correct_idx=correct_idx,
                    false_answer_idx=false_idx,
                )
                user_msg = build_prompt(
                    question=q["question"], choices=q["choices"], hint_text=hint_text
                )
                formatted = format_for_model(tokenizer, user_msg)
                hint_prompts.append({
                    "formatted_prompt": formatted,
                    "question_idx": global_idx,
                    "condition": condition,
                    "hint_format": hint_format,
                    "correct_answer": correct_idx,
                    "false_answer": false_idx,
                    "run_id": f"q{global_idx:05d}_{hint_format}_{condition}",
                })

    hint_results = process_batch(model, tokenizer, saes, hint_prompts, BATCH_SIZE)
    all_results.extend(hint_results)

    # Save results
    print(f"Saving {len(all_results)} results...")
    all_metadata = [r["metadata"] for r in all_results]
    with open(metadata_dir / f"metadata_task{task_id}.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Save SAE features grouped by run
    for result in all_results:
        run_id = result["metadata"]["run_id"]
        features = {}
        for (layer, width_k), sparse_list in result["sae_features"].items():
            features[f"L{layer}_W{width_k}k"] = [
                {"indices": s["indices"], "values": s["values"]}
                for s in sparse_list
            ]
        torch.save(features, features_dir / f"{run_id}.pt")

    print(f"Task {task_id} complete: {len(correct_questions)} questions, {len(all_results)} runs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_divergence_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_divergence_generation.py tests/test_divergence_generation.py
git commit -m "feat: add Phase 1 data generation script with batched SAE encoding"
```

---

### Task 8: Phase 2 Script — Classification and AUC Curves

**Files:**
- Create: `scripts/run_divergence_analysis.py`
- Create: `tests/test_divergence_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_divergence_analysis.py
# ABOUTME: Tests for divergence analysis helper functions.
# ABOUTME: Validates data loading, feature assembly, and AUC curve structure.

import pytest
import numpy as np
from scripts.run_divergence_analysis import (
    assemble_features_by_fraction,
    compute_divergence_onset,
)


class TestAssembleFeaturesByFraction:
    def test_output_structure(self):
        # Simulate loaded data: 3 questions, 2 fractions, 4 features
        sparse_data = {}
        for q in range(3):
            sparse_data[q] = {
                "no_hint": [
                    {"indices": np.array([0, 1]), "values": np.array([1.0, 2.0])},
                    {"indices": np.array([2]), "values": np.array([3.0])},
                ],
                "false_hint": [
                    {"indices": np.array([0]), "values": np.array([5.0])},
                    {"indices": np.array([1, 3]), "values": np.array([6.0, 7.0])},
                ],
            }
        features, labels = assemble_features_by_fraction(
            sparse_data, n_fractions=2, n_features=4
        )
        assert len(features) == 2  # 2 fractions
        assert len(features[0]) == 6  # 3 no_hint + 3 false_hint
        assert len(labels) == 6


class TestComputeDivergenceOnset:
    def test_clear_onset(self):
        false_auc = [0.51, 0.52, 0.55, 0.65, 0.80, 0.90]
        true_auc = [0.50, 0.51, 0.50, 0.52, 0.51, 0.53]
        onset = compute_divergence_onset(false_auc, true_auc, threshold=0.05, sustained=2)
        assert onset is not None
        assert onset <= 3  # first sustained gap >= 0.05

    def test_no_onset(self):
        false_auc = [0.51, 0.52, 0.51, 0.52]
        true_auc = [0.50, 0.51, 0.50, 0.51]
        onset = compute_divergence_onset(false_auc, true_auc, threshold=0.05, sustained=2)
        assert onset is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_divergence_analysis.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write implementation**

```python
# scripts/run_divergence_analysis.py
# ABOUTME: Phase 2 of divergence localization — trains classifiers and computes AUC curves.
# ABOUTME: Includes true-hint control, text similarity baseline, and visualization.

import json
import glob
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
from src.classifier import tune_regularization, train_classifier, compute_auc_per_fraction
from src.text_similarity import compute_text_similarity_curve


def load_all_metadata(metadata_dir: Path) -> list[dict]:
    """Load and merge metadata from all array task files."""
    all_metadata = []
    for path in sorted(metadata_dir.glob("metadata_task*.json")):
        with open(path) as f:
            all_metadata.extend(json.load(f))
    return all_metadata


def load_sparse_features(
    features_dir: Path,
    run_id: str,
    layer: int,
    width_k: int,
    n_fractions: int,
) -> list[np.ndarray]:
    """Load sparse SAE features for a run, returning dense arrays per fraction."""
    data = torch.load(features_dir / f"{run_id}.pt", weights_only=True)
    key = f"L{layer}_W{width_k}k"
    sparse_list = data[key]

    n_features = width_k * 1000
    dense_list = []
    for sparse in sparse_list:
        dense = from_sparse_features(sparse, total_features=n_features)
        dense_list.append(dense.numpy())

    return dense_list


def assemble_features_by_fraction(
    sparse_data: dict[int, dict[str, list]],
    n_fractions: int,
    n_features: int,
) -> tuple[dict[int, dict[int, np.ndarray]], dict[int, int]]:
    """Assemble features organized by fraction for classifier evaluation.

    Args:
        sparse_data: {question_id: {"no_hint": [sparse...], condition: [sparse...]}}
        n_fractions: number of fraction points
        n_features: total SAE features

    Returns:
        features_by_fraction: {fraction_idx: {sample_id: feature_array}}
        labels: {sample_id: 0 or 1}
    """
    features_by_fraction = {f: {} for f in range(n_fractions)}
    labels = {}
    sample_id = 0

    for q_id, conditions in sparse_data.items():
        for condition_name, sparse_list in conditions.items():
            label = 0 if condition_name == "no_hint" else 1
            for f in range(n_fractions):
                dense = from_sparse_features(sparse_list[f], total_features=n_features)
                features_by_fraction[f][sample_id] = dense.numpy()
            labels[sample_id] = label
            sample_id += 1

    return features_by_fraction, labels


def compute_divergence_onset(
    false_auc: list[float],
    true_auc: list[float],
    threshold: float = 0.05,
    sustained: int = 2,
) -> int | None:
    """Find the earliest fraction where false AUC exceeds true AUC by threshold.

    Requires the gap to be sustained for `sustained` consecutive fractions.
    Returns fraction index, or None if never sustained.
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
    false_auc: list[float],
    true_auc: list[float],
    fractions: list[float],
    title: str,
    save_path: Path,
    text_sim: list[float] | None = None,
):
    """Plot AUC(fraction) curves for false-hint and true-hint with optional text baseline."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(fractions, false_auc, "r-o", label="False-hint vs No-hint", linewidth=2)
    ax1.plot(fractions, true_auc, "b-s", label="True-hint vs No-hint", linewidth=2)
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
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

    # Load metadata
    print("Loading metadata...")
    metadata = load_all_metadata(metadata_dir)

    # Organize by question
    from src.data import build_experiment_groups
    no_hint_by_q, groups = build_experiment_groups(metadata)

    # Identify test/train split
    question_ids = sorted(no_hint_by_q.keys())
    n_test = max(1, len(question_ids) // 10)
    rng = np.random.RandomState(42)
    rng.shuffle(question_ids)
    test_ids = set(question_ids[:n_test])
    train_ids = set(question_ids[n_test:])
    print(f"Train: {len(train_ids)} questions, Test: {len(test_ids)} questions")

    all_results = {}

    for layer in SELECTED_LAYERS:
        for width_k in SAE_WIDTHS:
            key = f"L{layer}_W{width_k}k"
            print(f"\n=== {key} ===")
            n_features = width_k * 1000

            # Load features for false-hint analysis
            print("  Loading features (false-hint)...")
            train_data = {}
            test_data = {}

            for q_id in question_ids:
                nh_entry = no_hint_by_q[q_id]
                nh_features = load_sparse_features(
                    features_dir, nh_entry["run_id"], layer, width_k, N_FRACTIONS
                )

                for fmt in HINT_FORMATS:
                    group_key = (q_id, fmt)
                    if group_key not in groups or "false_hint" not in groups[group_key]:
                        continue
                    fh_entry = groups[group_key]["false_hint"]
                    fh_features = load_sparse_features(
                        features_dir, fh_entry["run_id"], layer, width_k, N_FRACTIONS
                    )

                    sample_id_nh = f"{q_id}_{fmt}_nh"
                    sample_id_fh = f"{q_id}_{fmt}_fh"
                    target = train_data if q_id in train_ids else test_data

                    target[sample_id_nh] = {"features": nh_features, "label": 0}
                    target[sample_id_fh] = {"features": fh_features, "label": 1}

            # Build training arrays
            print("  Building training data...")
            X_train_all = []
            y_train_all = []
            for sample_id, info in train_data.items():
                for f in range(N_FRACTIONS):
                    X_train_all.append(info["features"][f])
                    y_train_all.append(info["label"])
            X_train_all = np.vstack(X_train_all)
            y_train_all = np.array(y_train_all)

            # Tune regularization
            print("  Tuning regularization...")
            best_C = tune_regularization(X_train_all, y_train_all)
            print(f"  Best C: {best_C}")

            # Train final classifier
            print("  Training classifier...")
            clf = train_classifier(X_train_all, y_train_all, C=best_C)

            # Evaluate per fraction on test set
            print("  Computing AUC per fraction...")
            test_features_by_fraction = {f: {} for f in range(N_FRACTIONS)}
            test_labels = {}
            for sid, (sample_id, info) in enumerate(test_data.items()):
                for f in range(N_FRACTIONS):
                    test_features_by_fraction[f][sid] = info["features"][f]
                test_labels[sid] = info["label"]

            false_auc = compute_auc_per_fraction(
                clf, test_features_by_fraction, test_labels, N_FRACTIONS
            )

            # True-hint control
            print("  Computing true-hint control...")
            train_data_true = {}
            test_data_true = {}
            for q_id in question_ids:
                nh_entry = no_hint_by_q[q_id]
                nh_features = load_sparse_features(
                    features_dir, nh_entry["run_id"], layer, width_k, N_FRACTIONS
                )
                for fmt in HINT_FORMATS:
                    group_key = (q_id, fmt)
                    if group_key not in groups or "true_hint" not in groups[group_key]:
                        continue
                    th_entry = groups[group_key]["true_hint"]
                    th_features = load_sparse_features(
                        features_dir, th_entry["run_id"], layer, width_k, N_FRACTIONS
                    )

                    sample_id_nh = f"{q_id}_{fmt}_nh"
                    sample_id_th = f"{q_id}_{fmt}_th"
                    target = train_data_true if q_id in train_ids else test_data_true

                    target[sample_id_nh] = {"features": nh_features, "label": 0}
                    target[sample_id_th] = {"features": th_features, "label": 1}

            X_train_true = []
            y_train_true = []
            for sample_id, info in train_data_true.items():
                for f in range(N_FRACTIONS):
                    X_train_true.append(info["features"][f])
                    y_train_true.append(info["label"])
            X_train_true = np.vstack(X_train_true)
            y_train_true = np.array(y_train_true)

            best_C_true = tune_regularization(X_train_true, y_train_true)
            clf_true = train_classifier(X_train_true, y_train_true, C=best_C_true)

            test_features_true = {f: {} for f in range(N_FRACTIONS)}
            test_labels_true = {}
            for sid, (sample_id, info) in enumerate(test_data_true.items()):
                for f in range(N_FRACTIONS):
                    test_features_true[f][sid] = info["features"][f]
                test_labels_true[sid] = info["label"]

            true_auc = compute_auc_per_fraction(
                clf_true, test_features_true, test_labels_true, N_FRACTIONS
            )

            # Divergence onset
            onset = compute_divergence_onset(false_auc, true_auc)

            # Top classifier weights
            top_k = 20
            weights = np.abs(clf.coef_[0])
            top_indices = np.argsort(weights)[-top_k:][::-1]
            top_weights = [(int(idx), float(weights[idx])) for idx in top_indices]

            all_results[key] = {
                "false_auc": false_auc,
                "true_auc": true_auc,
                "onset_fraction_idx": onset,
                "onset_fraction": FRACTION_POINTS[onset] if onset is not None else None,
                "best_C": best_C,
                "top_features": top_weights,
            }

            # Plot
            plot_auc_curves(
                false_auc, true_auc, FRACTION_POINTS,
                f"AUC Curve — Layer {layer}, {width_k}k SAE",
                figures_dir / f"auc_L{layer}_W{width_k}k.png",
            )
            print(f"  False AUC: {[f'{a:.3f}' for a in false_auc]}")
            print(f"  True AUC:  {[f'{a:.3f}' for a in true_auc]}")
            print(f"  Onset: {FRACTION_POINTS[onset] if onset is not None else 'none'}")

    # Text similarity baseline (average across test questions)
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

    # Save all results
    output = {
        "layer_width_results": all_results,
        "text_similarity_baseline": avg_text_sim,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "fraction_points": FRACTION_POINTS,
    }
    with open(results_dir / "divergence_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Summary plot with text similarity overlay for best layer
    best_key = min(all_results, key=lambda k: all_results[k]["onset_fraction_idx"] or 999)
    best = all_results[best_key]
    plot_auc_curves(
        best["false_auc"], best["true_auc"], FRACTION_POINTS,
        f"Divergence Localization — {best_key}",
        figures_dir / "auc_best_with_text.png",
        text_sim=avg_text_sim,
    )

    # Print summary
    print("\n=== DIVERGENCE ONSET SUMMARY ===")
    for key in sorted(all_results):
        r = all_results[key]
        onset_str = f"{r['onset_fraction']:.0%}" if r["onset_fraction"] else "none"
        print(f"  {key}: onset at {onset_str}")

    # Per-format AUC breakdown on test set
    print("\n=== PER-FORMAT AUC (test set, last fraction) ===")
    for key in sorted(all_results):
        layer = int(key.split("_")[0][1:])
        width_str = key.split("_")[1]
        width_k = int(width_str[1:].replace("k", ""))
        n_features = width_k * 1000
        for fmt in HINT_FORMATS:
            fmt_features = {f: {} for f in range(N_FRACTIONS)}
            fmt_labels = {}
            sid = 0
            for q_id in test_ids:
                nh_entry = no_hint_by_q.get(q_id)
                group_key = (q_id, fmt)
                if not nh_entry or group_key not in groups or "false_hint" not in groups[group_key]:
                    continue
                fh_entry = groups[group_key]["false_hint"]
                nh_feats = load_sparse_features(features_dir, nh_entry["run_id"], layer, width_k, N_FRACTIONS)
                fh_feats = load_sparse_features(features_dir, fh_entry["run_id"], layer, width_k, N_FRACTIONS)
                for f in range(N_FRACTIONS):
                    fmt_features[f][sid] = nh_feats[f]
                    fmt_features[f][sid + 1] = fh_feats[f]
                fmt_labels[sid] = 0
                fmt_labels[sid + 1] = 1
                sid += 2
            if fmt_labels:
                clf = all_results[key].get("_clf")
                if clf is None:
                    clf = train_classifier(X_train_all, y_train_all, C=all_results[key]["best_C"])
                fmt_auc = compute_auc_per_fraction(clf, fmt_features, fmt_labels, N_FRACTIONS)
                print(f"  {key} {fmt}: AUC@100% = {fmt_auc[-1]:.3f}")

    # Hint-following stats
    total_false = sum(1 for m in metadata if m["condition"] == "false_hint")
    hint_following = sum(1 for m in metadata if m.get("hint_following", False))
    mentions = sum(1 for m in metadata if m.get("mentions_hint", False) and m["condition"] == "false_hint")
    print(f"\nHint-following rate: {hint_following}/{total_false} ({hint_following/max(total_false,1):.1%})")
    print(f"Hint-mention rate: {mentions}/{total_false} ({mentions/max(total_false,1):.1%})")

    print(f"\nResults saved to {results_dir / 'divergence_results.json'}")
    print(f"Figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_divergence_analysis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_divergence_analysis.py tests/test_divergence_analysis.py
git commit -m "feat: add Phase 2 classification and AUC analysis script"
```

---

### Task 9: SLURM Launch Script

**Files:**
- Create: `scripts/launch_divergence.sh`

- [ ] **Step 1: Write the launch script**

```bash
#!/bin/bash
# ABOUTME: Launches the divergence localization pipeline as SLURM jobs.
# ABOUTME: Phase 1: array job (4 tasks, 1 H200 each). Phase 2: CPU job after Phase 1.

set -euo pipefail

PARTITION_GPU="gpu"
PARTITION_CPU="177huntington"
CONDA_ENV="cot_sae"
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

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
    --mem=64G \
    --job-name=div-analysis \
    --output=$WORKDIR/outputs/divergence/logs/phase2_%j.log \
    --error=$WORKDIR/outputs/divergence/logs/phase2_%j.err \
    --wrap="cd $WORKDIR && conda run -n $CONDA_ENV env PYTHONPATH=$WORKDIR python scripts/run_divergence_analysis.py")

echo "Phase 2 (analysis):  Job $JOB2 (depends on $JOB1)"

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Phase 1 logs: $WORKDIR/outputs/divergence/logs/phase1_${JOB1}_*.log"
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x scripts/launch_divergence.sh
bash -n scripts/launch_divergence.sh && echo "Syntax OK"
```

- [ ] **Step 3: Create logs directory**

```bash
mkdir -p outputs/divergence/logs
```

- [ ] **Step 4: Commit**

```bash
git add scripts/launch_divergence.sh
git commit -m "feat: add SLURM launch script for divergence localization pipeline"
```

---

### Task 10: Add scikit-learn to Environment

**Files:**
- Modify: `environment.yml`

- [ ] **Step 1: Check if scikit-learn is already in environment**

Run: `conda run -n cot_sae python -c "import sklearn; print(sklearn.__version__)"` to check.

- [ ] **Step 2: Add scikit-learn if missing**

Add `- scikit-learn` to the pip dependencies in `environment.yml` (or conda dependencies).

- [ ] **Step 3: Install**

Run: `conda run -n cot_sae pip install scikit-learn`

- [ ] **Step 4: Verify**

Run: `conda run -n cot_sae python -c "from sklearn.linear_model import LogisticRegression; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add environment.yml
git commit -m "deps: add scikit-learn for logistic regression classifier"
```

---

### Task 11: Integration Test — Full Pipeline Smoke Test

**Files:**
- Create: `tests/test_integration_divergence.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration_divergence.py
# ABOUTME: Smoke test for the full divergence localization pipeline.
# ABOUTME: Validates end-to-end data flow with tiny synthetic data.

import pytest
import numpy as np
import torch
from src.fractional import compute_fraction_indices, to_sparse_features, from_sparse_features
from src.classifier import tune_regularization, train_classifier, compute_auc_per_fraction
from src.text_similarity import compute_text_similarity_curve
from scripts.run_divergence_analysis import assemble_features_by_fraction, compute_divergence_onset


class TestEndToEndPipeline:
    def test_sparse_to_classifier_flow(self):
        """Test that sparse features can flow through the full analysis pipeline."""
        n_questions = 30
        n_fractions = 5
        n_features = 100
        rng = np.random.RandomState(42)

        # Simulate sparse features for no_hint and false_hint
        sparse_data = {}
        for q in range(n_questions):
            nh_fractions = []
            fh_fractions = []
            for f in range(n_fractions):
                # no_hint: features centered around 0
                nh_dense = torch.zeros(n_features)
                active = rng.choice(n_features, 5, replace=False)
                nh_dense[active] = torch.from_numpy(rng.randn(5).astype(np.float32))
                nh_fractions.append(to_sparse_features(nh_dense))

                # false_hint: features shifted positive (increasingly with fraction)
                fh_dense = torch.zeros(n_features)
                fh_dense[active] = torch.from_numpy(
                    (rng.randn(5) + (f + 1) * 0.5).astype(np.float32)
                )
                fh_fractions.append(to_sparse_features(fh_dense))

            sparse_data[q] = {"no_hint": nh_fractions, "false_hint": fh_fractions}

        # Assemble
        features_by_fraction, labels = assemble_features_by_fraction(
            sparse_data, n_fractions=n_fractions, n_features=n_features
        )

        # Build training arrays
        X_all = []
        y_all = []
        for sample_id in sorted(labels.keys()):
            for f in range(n_fractions):
                X_all.append(features_by_fraction[f][sample_id])
            y_all.extend([labels[sample_id]] * n_fractions)
        X_all = np.vstack(X_all)
        y_all = np.array(y_all)

        # Train and evaluate
        best_C = tune_regularization(X_all, y_all, n_folds=3)
        clf = train_classifier(X_all, y_all, C=best_C)
        auc_curve = compute_auc_per_fraction(
            clf, features_by_fraction, labels, n_fractions
        )

        assert len(auc_curve) == n_fractions
        # Later fractions should have higher AUC (signal increases)
        assert auc_curve[-1] >= auc_curve[0]

    def test_text_similarity_integration(self):
        """Test text similarity across fractions."""
        text_a = "Let me think step by step about this question carefully."
        text_b = "The expert said the answer is B so I think it must be B."
        fractions = [0.25, 0.5, 0.75, 1.0]
        curve = compute_text_similarity_curve(text_a, text_b, fractions)
        assert len(curve) == 4
        assert all(isinstance(s, float) for s in curve)

    def test_divergence_onset_detection(self):
        """Test onset detection with synthetic AUC curves."""
        # False-hint AUC rises sharply at fraction 3
        false_auc = [0.52, 0.53, 0.55, 0.70, 0.85, 0.90]
        true_auc = [0.51, 0.52, 0.51, 0.52, 0.53, 0.52]
        onset = compute_divergence_onset(false_auc, true_auc, threshold=0.05, sustained=2)
        assert onset is not None
        assert onset <= 3
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration_divergence.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_divergence.py
git commit -m "test: add integration test for divergence localization pipeline"
```

---

## Known Limitations

- **TransformerLens batched generation**: The `model.generate()` with batched left-padded inputs requires attention mask support. TransformerLens may need manual attention mask handling — verify during Task 3 implementation and adjust if needed.
- **SLURM array job count**: `SLURM_ARRAY_TASK_COUNT` may not be available in all SLURM versions. The script falls back to n_tasks=1.
- **Memory for Phase 2**: Loading all sparse features for ~3,000 questions into RAM should be fine at ~30-50 GB, but may need chunking if the number of correct questions is unexpectedly large.
