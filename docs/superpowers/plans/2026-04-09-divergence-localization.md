# Divergence Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that trains linear classifiers on SAE features at fractional positions to produce AUC(fraction) curves showing when hint-related divergence becomes detectable in the model's internals.

**Architecture:** Two-phase pipeline. Phase 1 (GPU): HuggingFace batched generation with forward hooks for selective layer caching, SAE encoding at fractional positions, sparse feature storage. Phase 2 (CPU): logistic regression with GroupKFold CV, AUC curves with bootstrap CIs, controls, visualization. Reuses existing `src/data.py` for prompt construction, `src/sae_analysis.py` for SAE loading.

**Tech Stack:** HuggingFace transformers, SAELens, scikit-learn (LogisticRegression, GroupKFold, roc_auc_score), sentence-transformers, torch, numpy, matplotlib

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
        assert pos == 3  # EOS at absolute position 3
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
        indices = sparse["indices"]
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices)
        values = sparse["values"]
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values)
        dense[indices.long()] = values.float()
    return dense


def find_eos_position(
    tokens: torch.Tensor,
    eos_token_id: int,
    prompt_length: int,
) -> int:
    """Find the position of the first EOS token after the prompt.

    Returns the absolute index of the first EOS in the generated portion,
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

### Task 3: HuggingFace Model Loading and Hooked Forward Pass

**Files:**
- Create: `src/hf_model.py`
- Create: `tests/test_hf_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_hf_model.py
# ABOUTME: Tests for HuggingFace model utilities — batched tokenization and hooked forward pass.
# ABOUTME: Uses small tensors to validate padding, hook capture, and generation length detection.

import pytest
import torch
from src.hf_model import (
    tokenize_batch,
    extract_generation_lengths,
    register_layer_hooks,
    remove_hooks,
)


class TestTokenizeBatch:
    def test_left_pads(self):
        # Mock a tokenizer
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = 0
        tokenizer.return_value = {
            "input_ids": torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]]),
        }
        result = tokenize_batch(tokenizer, ["abc", "de"])
        assert "input_ids" in result
        assert "attention_mask" in result


class TestExtractGenerationLengths:
    def test_finds_eos(self):
        # Batch of 2, padded prompt length = 4, max_new_tokens appended
        output_tokens = torch.tensor([
            [0, 1, 2, 3, 10, 11, 99, 0],  # prompt pad=4, gen=[10,11,EOS,pad]
            [0, 0, 4, 5, 12, 99, 0, 0],   # prompt pad=4, gen=[12,EOS,pad,pad]
        ])
        gen_lengths = extract_generation_lengths(
            output_tokens, padded_prompt_length=4, eos_token_id=99
        )
        assert gen_lengths == [2, 1]  # before EOS

    def test_no_eos(self):
        output_tokens = torch.tensor([
            [1, 2, 3, 4, 10, 11, 12, 13],
        ])
        gen_lengths = extract_generation_lengths(
            output_tokens, padded_prompt_length=4, eos_token_id=99
        )
        assert gen_lengths == [4]  # full generated length


class TestRegisterLayerHooks:
    def test_hooks_capture_output(self):
        # Simple sequential model with named layers
        import torch.nn as nn
        model = nn.Sequential()
        model.add_module("layer_0", nn.Linear(4, 4))
        model.add_module("layer_1", nn.Linear(4, 4))

        captured = {}
        hooks = register_layer_hooks(model, layer_indices=[0, 1], captured=captured,
                                     layer_accessor=lambda m, i: list(m.children())[i])
        x = torch.randn(2, 4)
        model(x)
        assert 0 in captured
        assert 1 in captured
        assert captured[0].shape == (2, 4)
        remove_hooks(hooks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hf_model.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/hf_model.py
# ABOUTME: HuggingFace model loading, batched tokenization, and hooked forward pass.
# ABOUTME: Registers forward hooks on selected decoder layers to capture residual streams.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import MODEL_NAME, SELECTED_LAYERS


def load_hf_model(device: str = "cuda"):
    """Load Gemma 2 2B-it with HuggingFace."""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def tokenize_batch(tokenizer, prompts: list[str]) -> dict:
    """Tokenize a batch of prompts with left-padding."""
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )


def extract_generation_lengths(
    output_tokens: torch.Tensor,
    padded_prompt_length: int,
    eos_token_id: int,
) -> list[int]:
    """Find actual generation length for each sequence in a batch.

    After left-padded batched generation, all generated tokens start at
    padded_prompt_length (which is max_prompt_length in the batch).

    gen_length excludes the EOS token, so the 100% fraction point
    is the token just before EOS.
    """
    batch_size = output_tokens.shape[0]
    gen_lengths = []
    for i in range(batch_size):
        generated = output_tokens[i, padded_prompt_length:]
        eos_positions = (generated == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            gen_lengths.append(eos_positions[0].item())
        else:
            gen_lengths.append(len(generated))
    return gen_lengths


def register_layer_hooks(
    model,
    layer_indices: list[int],
    captured: dict,
    layer_accessor=None,
) -> list:
    """Register forward hooks on selected decoder layers.

    The hook captures the layer's output (residual stream post-layer).
    For Gemma 2: model.model.layers[i] is the i-th decoder layer.

    Args:
        model: the model (HuggingFace or nn.Module)
        layer_indices: which layers to hook
        captured: dict to store captured tensors {layer_idx: tensor}
        layer_accessor: function(model, idx) -> module. Defaults to Gemma 2 structure.

    Returns:
        list of hook handles (call remove_hooks() when done)
    """
    if layer_accessor is None:
        layer_accessor = lambda m, i: m.model.layers[i]

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # Gemma 2 decoder layer returns (hidden_states, ...) tuple
            if isinstance(output, tuple):
                captured[layer_idx] = output[0].detach()
            else:
                captured[layer_idx] = output.detach()
        return hook_fn

    hooks = []
    for idx in layer_indices:
        layer_module = layer_accessor(model, idx)
        handle = layer_module.register_forward_hook(make_hook(idx))
        hooks.append(handle)

    return hooks


def remove_hooks(hooks: list):
    """Remove all registered hooks."""
    for handle in hooks:
        handle.remove()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hf_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/hf_model.py tests/test_hf_model.py
git commit -m "feat: add HuggingFace model loading with batched tokenization and layer hooks"
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
            out = torch.zeros(x.shape[0], output_dim)
            out[:, 0] = 1.0
            out[:, 1] = 0.5
            return out
        sae.encode = fake_encode
        return sae

    def test_output_structure(self):
        sae = self._make_mock_sae(100)
        residual = torch.randn(50, 64)  # 50 tokens, dim 64
        fraction_indices = [4, 9, 14, 19, 24]
        result = encode_at_fractions(sae, residual, fraction_indices)
        assert len(result) == 5
        for entry in result:
            assert "indices" in entry
            assert "values" in entry

    def test_correct_number_of_positions(self):
        sae = self._make_mock_sae(100)
        residual = torch.randn(20, 64)
        fraction_indices = [5, 10, 15]
        result = encode_at_fractions(sae, residual, fraction_indices)
        assert len(result) == 3

    def test_sparse_output(self):
        sae = self._make_mock_sae(100)
        residual = torch.randn(50, 64)
        fraction_indices = [24]
        result = encode_at_fractions(sae, residual, fraction_indices)
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
    residual: torch.Tensor,
    fraction_indices: list[int],
) -> list[dict]:
    """Encode residual stream through SAE at specific token positions.

    Args:
        sae: loaded SAE model
        residual: [seq_len, hidden_dim] residual stream tensor for one layer
        fraction_indices: absolute token positions to encode

    Returns:
        list of sparse feature dicts, one per fraction index
    """
    positions = torch.tensor(fraction_indices, dtype=torch.long)
    selected = residual[positions]  # [n_fractions, hidden_dim]

    with torch.no_grad():
        features = sae.encode(selected.to(sae.device))  # [n_fractions, n_features]
    features = features.cpu()

    return [to_sparse_features(features[i]) for i in range(features.shape[0])]
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

### Task 5: Linear Classifier with GroupKFold

**Files:**
- Create: `src/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_classifier.py
# ABOUTME: Tests for linear classifier training and AUC curve computation.
# ABOUTME: Validates GroupKFold cross-validation and per-fraction evaluation.

import pytest
import numpy as np
from src.classifier import (
    tune_regularization,
    train_classifier,
    compute_auc_per_fraction,
    compute_bootstrap_ci,
)


class TestTuneRegularization:
    def test_returns_best_C(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(100, 10) + 1, rng.randn(100, 10) - 1])
        y = np.array([0] * 100 + [1] * 100)
        groups = np.array([i // 2 for i in range(200)])  # 100 groups of 2
        best_C = tune_regularization(X, y, groups, n_folds=5)
        assert isinstance(best_C, float)
        assert best_C > 0

    def test_groups_respected(self):
        rng = np.random.RandomState(42)
        # 10 groups, 20 samples each
        X = np.vstack([rng.randn(20, 5) + (i % 2) * 3 for i in range(10)])
        y = np.array([i % 2 for i in range(10) for _ in range(20)])
        groups = np.array([i for i in range(10) for _ in range(20)])
        best_C = tune_regularization(X, y, groups, n_folds=5)
        assert best_C > 0


class TestTrainClassifier:
    def test_returns_fitted_model(self):
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(50, 10) + 2, rng.randn(50, 10) - 2])
        y = np.array([0] * 50 + [1] * 50)
        clf = train_classifier(X, y, C=1.0)
        accuracy = (clf.predict(X) == y).mean()
        assert accuracy > 0.9


class TestComputeAucPerFraction:
    def test_returns_correct_shape(self):
        rng = np.random.RandomState(42)
        n_samples = 40
        n_fractions = 5
        n_features = 10
        # features_by_fraction: {fraction_idx: np.ndarray [n_samples, n_features]}
        features_by_fraction = {}
        y = np.array([i % 2 for i in range(n_samples)])
        for f in range(n_fractions):
            X = rng.randn(n_samples, n_features) + (y * 4)[:, None]
            features_by_fraction[f] = X

        clf = train_classifier(features_by_fraction[0], y, C=1.0)
        auc_curve = compute_auc_per_fraction(clf, features_by_fraction, y, n_fractions)
        assert len(auc_curve) == n_fractions
        assert all(0.0 <= a <= 1.0 for a in auc_curve)


class TestBootstrapCI:
    def test_returns_lower_upper(self):
        rng = np.random.RandomState(42)
        n = 40
        probs_false = rng.rand(n)
        y_false = (probs_false > 0.3).astype(int)
        probs_true = rng.rand(n) * 0.5
        y_true = (probs_true > 0.5).astype(int)
        question_ids = np.array([i // 2 for i in range(n)])
        lower, upper = compute_bootstrap_ci(
            probs_false, y_false, probs_true, y_true, question_ids,
        )
        assert lower < upper

    def test_identical_predictions_ci_near_zero(self):
        n = 40
        probs = np.array([0.6] * n)
        y = np.array([0, 1] * (n // 2))
        question_ids = np.array([i // 2 for i in range(n)])
        lower, upper = compute_bootstrap_ci(
            probs, y, probs, y, question_ids,
        )
        assert abs(lower) < 0.1
        assert abs(upper) < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# src/classifier.py
# ABOUTME: L2-regularized logistic regression for distinguishing hint conditions.
# ABOUTME: Includes GroupKFold CV for regularization tuning and bootstrap CI for onset detection.

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import roc_auc_score


def tune_regularization(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_folds: int = 5,
    C_values: list[float] | None = None,
) -> float:
    """Find the best L2 regularization strength via GroupKFold cross-validation.

    Groups ensure all samples from the same question stay in the same fold.
    """
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    gkf = GroupKFold(n_splits=n_folds)
    best_C = C_values[0]
    best_score = -1.0

    for C in C_values:
        clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=1000)
        scores = cross_val_score(clf, X, y, cv=gkf, groups=groups, scoring="roc_auc")
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
    features_by_fraction: dict[int, np.ndarray],
    y: np.ndarray,
    n_fractions: int,
) -> list[float]:
    """Compute AUC at each fractional position using a trained classifier.

    Args:
        clf: trained classifier
        features_by_fraction: {fraction_idx: np.ndarray [n_samples, n_features]}
        y: [n_samples] binary labels, same ordering as feature arrays
        n_fractions: number of fraction points

    Returns:
        list of AUC values, one per fraction
    """
    auc_curve = []
    for f in range(n_fractions):
        X = features_by_fraction[f]
        probs = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
        auc_curve.append(auc)
    return auc_curve


def compute_bootstrap_ci(
    probs_false: np.ndarray,
    y_false: np.ndarray,
    probs_true: np.ndarray,
    y_true: np.ndarray,
    question_ids: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for AUC gap (false-hint AUC minus true-hint AUC).

    Resamples unique question IDs, gathers all corresponding samples,
    and recomputes AUC for both conditions per resample.

    Args:
        probs_false: [n_test] predicted probabilities for false-hint test set
        y_false: [n_test] binary labels for false-hint
        probs_true: [n_test] predicted probabilities for true-hint
        y_true: [n_test] binary labels for true-hint
        question_ids: [n_test] question IDs for grouping

    Returns:
        (lower, upper) bounds of the CI on the AUC gap
    """
    rng = np.random.RandomState(seed)
    unique_qids = np.unique(question_ids)
    n_questions = len(unique_qids)

    # Build index lookup: question_id -> list of sample indices
    qid_to_idx = {}
    for idx, qid in enumerate(question_ids):
        qid_to_idx.setdefault(qid, []).append(idx)

    diffs = []
    for _ in range(n_bootstrap):
        sampled_qids = rng.choice(unique_qids, size=n_questions, replace=True)
        idx_false = np.concatenate([qid_to_idx[q] for q in sampled_qids])
        idx_true = np.concatenate([qid_to_idx[q] for q in sampled_qids])

        try:
            auc_f = roc_auc_score(y_false[idx_false], probs_false[idx_false])
            auc_t = roc_auc_score(y_true[idx_true], probs_true[idx_true])
            diffs.append(auc_f - auc_t)
        except ValueError:
            continue  # skip if a resample has only one class

    alpha = (1 - ci) / 2
    lower = np.percentile(diffs, 100 * alpha)
    upper = np.percentile(diffs, 100 * (1 - alpha))
    return float(lower), float(upper)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/classifier.py tests/test_classifier.py
git commit -m "feat: add logistic regression classifier with GroupKFold CV and bootstrap CI"
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
# ABOUTME: Validates token-based truncation and embedding cosine similarity.

import pytest
from src.text_similarity import (
    text_at_token_fraction,
    compute_text_similarity_curve,
)


class TestTextAtTokenFraction:
    def test_half_fraction(self):
        tokens = ["one", "two", "three", "four", "five", "six"]
        result = text_at_token_fraction(tokens, 0.5)
        assert result == "one two three"

    def test_full_fraction(self):
        tokens = ["hello", "world"]
        result = text_at_token_fraction(tokens, 1.0)
        assert result == "hello world"

    def test_small_fraction(self):
        tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        result = text_at_token_fraction(tokens, 0.1)
        assert len(result) > 0  # at least 1 token


class TestComputeTextSimilarityCurve:
    def test_identical_texts_high_similarity(self):
        text_a = "The answer is clearly option B because of the evidence presented."
        text_b = "The answer is clearly option B because of the evidence presented."
        fractions = [0.5, 1.0]
        curve = compute_text_similarity_curve(text_a, text_b, fractions)
        assert len(curve) == 2
        assert all(s > 0.99 for s in curve)

    def test_different_texts_lower_similarity(self):
        text_a = "I think the answer is B based on scientific evidence and reasoning."
        text_b = "The professor suggested C so I will go with that recommendation."
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
# ABOUTME: Uses sentence-transformers for semantic embedding similarity as a baseline control.

import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def _get_embed_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def text_at_token_fraction(tokens: list[str], fraction: float) -> str:
    """Extract the first `fraction` of a token list, joined as text."""
    n = max(1, int(len(tokens) * fraction))
    return " ".join(tokens[:n])


def compute_text_similarity_curve(
    text_a: str,
    text_b: str,
    fractions: list[float],
) -> list[float]:
    """Compute cosine similarity between two texts at each fractional position.

    Truncates by token count (whitespace-split) to align with SAE fractional positions.
    Uses sentence-transformers for semantic similarity.
    """
    model = _get_embed_model()
    tokens_a = text_a.split()
    tokens_b = text_b.split()

    similarities = []
    for frac in fractions:
        prefix_a = text_at_token_fraction(tokens_a, frac)
        prefix_b = text_at_token_fraction(tokens_b, frac)

        if not prefix_a.strip() or not prefix_b.strip():
            similarities.append(1.0)
            continue

        embeddings = model.encode([prefix_a, prefix_b])
        cos_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        similarities.append(float(cos_sim))

    return similarities
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_text_similarity.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/text_similarity.py tests/test_text_similarity.py
git commit -m "feat: add text similarity baseline using sentence-transformers"
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
            run_id="q00001_authority_false_hint",
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
        assert meta["run_id"] == "q00001_authority_false_hint"
        assert meta["hint_following"] is True
        assert meta["gen_length"] == 100

    def test_no_hint_following(self):
        meta = build_run_metadata(
            run_id="q00001_no_hint",
            question_idx=1,
            hint_format="none",
            condition="no_hint",
            correct_answer=2,
            false_answer=0,
            predicted=2,
            response="The answer is C.",
            prompt_length=50,
            gen_length=80,
        )
        assert meta["hint_following"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_divergence_generation.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write implementation**

```python
# scripts/run_divergence_generation.py
# ABOUTME: Phase 1 of divergence localization — generates CoT under all conditions.
# ABOUTME: HuggingFace batched generation with forward hooks for SAE encoding at fractional positions.

import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset

from src.config import (
    MMLU_DATASET, MMLU_SPLIT, HINT_FORMATS, MAX_NEW_TOKENS,
    SELECTED_LAYERS, SAE_WIDTHS, N_FRACTIONS, DIVERGENCE_DIR, BATCH_SIZE,
)
from src.data import (
    build_prompt, format_for_model, insert_hint, parse_answer, check_mentions_hint,
)
from src.generate import pick_false_answer
from src.sae_analysis import load_sae
from src.hf_model import (
    load_hf_model, tokenize_batch, extract_generation_lengths,
    register_layer_hooks, remove_hooks,
)
from src.fractional import compute_fraction_indices
from src.fractional_sae import encode_at_fractions


def split_into_chunks(items: list, n_chunks: int) -> list[list]:
    """Split a list into n roughly equal chunks."""
    k, m = divmod(len(items), n_chunks)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]


def build_run_metadata(
    run_id, question_idx, hint_format, condition,
    correct_answer, false_answer, predicted, response,
    prompt_length, gen_length,
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


def generate_batch(model, tokenizer, prompts, batch_size):
    """Batched text generation without activation caching.

    Returns (output_tokens, padded_prompt_length, prompt_attention_mask).
    The prompt_attention_mask is saved for reuse in the forward pass.
    """
    encoded = tokenize_batch(tokenizer, prompts)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    padded_prompt_length = input_ids.shape[1]

    with torch.no_grad():
        output_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,
            do_sample=False,
        )

    return output_tokens, padded_prompt_length, attention_mask


def forward_with_hooks(model, input_ids, attention_mask, captured):
    """Run forward pass with hooks already registered. Populates captured dict."""
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)


def process_batch_with_sae(
    model, tokenizer, saes, prompts_info, batch_size,
):
    """Generate text, run hooked forward pass, encode SAE features at fractional positions.

    Two-step process:
    1. Batched generation (no hooks needed)
    2. Batched forward pass with hooks to capture residual streams at selected layers
    Then SAE encode at fractional positions per sequence.
    """
    results = []

    for batch_start in range(0, len(prompts_info), batch_size):
        batch = prompts_info[batch_start:batch_start + batch_size]
        prompts = [info["formatted_prompt"] for info in batch]

        # Step 1: Generate (also returns the prompt attention mask for reuse)
        output_tokens, padded_prompt_length, prompt_attention_mask = generate_batch(
            model, tokenizer, prompts, batch_size
        )

        # Get generation lengths
        eos_token_id = tokenizer.eos_token_id or 1
        gen_lengths = extract_generation_lengths(
            output_tokens, padded_prompt_length, eos_token_id
        )

        # Step 2: Forward pass with hooks
        captured = {}
        hooks = register_layer_hooks(model, SELECTED_LAYERS, captured)

        # Build attention mask for the full output sequence,
        # reusing the prompt attention mask from generation
        full_attention_mask = torch.ones_like(output_tokens)
        full_attention_mask[:, :padded_prompt_length] = prompt_attention_mask
        # Mask out tokens after EOS for each sequence
        for i in range(len(batch)):
            eos_abs = padded_prompt_length + gen_lengths[i]
            if eos_abs < output_tokens.shape[1]:
                full_attention_mask[i, eos_abs:] = 0

        forward_with_hooks(model, output_tokens, full_attention_mask, captured)
        remove_hooks(hooks)

        # Step 3: Process each sequence
        for i, info in enumerate(batch):
            gen_len = max(gen_lengths[i], 1)

            # Decode generated text
            gen_start = padded_prompt_length
            gen_tokens = output_tokens[i, gen_start:gen_start + gen_len]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            predicted = parse_answer(response)

            # Compute fraction indices (absolute positions in full sequence)
            fraction_indices = compute_fraction_indices(
                gen_length=gen_len, n_fractions=N_FRACTIONS,
                prompt_length=padded_prompt_length,
            )

            # SAE encode at fractional positions per layer
            sae_features = {}
            for layer in SELECTED_LAYERS:
                residual = captured[layer][i]  # [seq_len, hidden_dim]
                for width_k in SAE_WIDTHS:
                    sae = saes[(layer, width_k)]
                    sparse_list = encode_at_fractions(sae, residual, fraction_indices)
                    sae_features[f"L{layer}_W{width_k}k"] = sparse_list

            meta = build_run_metadata(
                run_id=info["run_id"],
                question_idx=info["question_idx"],
                hint_format=info.get("hint_format", "none"),
                condition=info["condition"],
                correct_answer=info["correct_answer"],
                false_answer=info["false_answer"],
                predicted=predicted,
                response=response,
                prompt_length=padded_prompt_length,
                gen_length=gen_len,
            )
            results.append({"metadata": meta, "sae_features": sae_features})

        del captured
        torch.cuda.empty_cache()

    return results


def main():
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
    chunks = split_into_chunks(all_questions, n_tasks)
    my_questions = chunks[task_id]
    offset = sum(len(chunks[i]) for i in range(task_id))
    print(f"Task {task_id}: questions {offset} to {offset + len(my_questions) - 1}")

    # Load model
    print("Loading model...")
    model, tokenizer = load_hf_model()

    # Phase 1a: Baseline generation (no SAE caching — just find correct answers)
    print("Running baseline to find correctly-answered questions...")
    correct_questions = []

    for batch_start in tqdm(range(0, len(my_questions), BATCH_SIZE), desc="Baseline"):
        batch_qs = my_questions[batch_start:batch_start + BATCH_SIZE]
        prompts = []
        for local_idx, q in enumerate(batch_qs):
            global_idx = offset + batch_start + local_idx
            user_msg = build_prompt(question=q["question"], choices=q["choices"], hint_text="")
            prompts.append(format_for_model(tokenizer, user_msg))

        output_tokens, padded_prompt_length, _ = generate_batch(
            model, tokenizer, prompts, BATCH_SIZE
        )
        eos_token_id = tokenizer.eos_token_id or 1
        gen_lengths = extract_generation_lengths(
            output_tokens, padded_prompt_length, eos_token_id
        )

        for i, q in enumerate(batch_qs):
            global_idx = offset + batch_start + i
            gen_len = max(gen_lengths[i], 1)
            gen_tokens = output_tokens[i, padded_prompt_length:padded_prompt_length + gen_len]
            response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            predicted = parse_answer(response)

            if predicted == q["answer"]:
                correct_questions.append({
                    "global_idx": global_idx,
                    "question": q,
                    "false_answer": pick_false_answer(q["answer"], seed=global_idx),
                    "baseline_response": response,
                })

    print(f"Task {task_id}: {len(correct_questions)}/{len(my_questions)} correct")

    # Load SAEs for Phase 1b
    print("Loading SAEs...")
    saes = {}
    for layer in SELECTED_LAYERS:
        for width_k in SAE_WIDTHS:
            saes[(layer, width_k)] = load_sae(layer, width_k)

    # Phase 1b: Generate all conditions for correct questions WITH SAE caching
    print("Running full generation with SAE encoding...")
    all_prompts = []
    for cq in correct_questions:
        q = cq["question"]
        global_idx = cq["global_idx"]
        correct_idx = q["answer"]
        false_idx = cq["false_answer"]

        # No-hint (with SAE caching this time)
        user_msg = build_prompt(question=q["question"], choices=q["choices"], hint_text="")
        all_prompts.append({
            "formatted_prompt": format_for_model(tokenizer, user_msg),
            "question_idx": global_idx,
            "condition": "no_hint",
            "hint_format": "none",
            "correct_answer": correct_idx,
            "false_answer": false_idx,
            "run_id": f"q{global_idx:05d}_no_hint",
        })

        # Hint conditions
        for hint_format in HINT_FORMATS:
            for condition in ["true_hint", "false_hint"]:
                hint_text = insert_hint(
                    condition=condition, hint_format=hint_format,
                    correct_idx=correct_idx, false_answer_idx=false_idx,
                )
                user_msg = build_prompt(
                    question=q["question"], choices=q["choices"], hint_text=hint_text,
                )
                all_prompts.append({
                    "formatted_prompt": format_for_model(tokenizer, user_msg),
                    "question_idx": global_idx,
                    "condition": condition,
                    "hint_format": hint_format,
                    "correct_answer": correct_idx,
                    "false_answer": false_idx,
                    "run_id": f"q{global_idx:05d}_{hint_format}_{condition}",
                })

    all_results = process_batch_with_sae(model, tokenizer, saes, all_prompts, BATCH_SIZE)

    # Save results
    print(f"Saving {len(all_results)} results...")
    all_metadata = [r["metadata"] for r in all_results]
    with open(metadata_dir / f"metadata_task{task_id}.json", "w") as f:
        json.dump(all_metadata, f, indent=2)

    for result in all_results:
        run_id = result["metadata"]["run_id"]
        torch.save(result["sae_features"], features_dir / f"{run_id}.pt")

    print(f"Task {task_id} complete: {len(correct_questions)} correct, {len(all_results)} runs saved")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_divergence_generation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_divergence_generation.py tests/test_divergence_generation.py
git commit -m "feat: add Phase 1 data generation with HuggingFace batching and SAE hooks"
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
# ABOUTME: Validates feature assembly, onset detection, and sparse-to-dense flow.

import pytest
import numpy as np
import torch
from src.fractional import to_sparse_features
from scripts.run_divergence_analysis import (
    build_paired_features,
    compute_divergence_onset,
)


class TestBuildPairedFeatures:
    def test_output_structure(self):
        # 3 questions, each with no_hint and false_hint sparse features
        # 2 fractions, 10 features
        question_data = []
        for q in range(3):
            nh_fracs = [to_sparse_features(torch.randn(10).abs()) for _ in range(2)]
            fh_fracs = [to_sparse_features(torch.randn(10).abs()) for _ in range(2)]
            question_data.append({
                "question_id": q,
                "no_hint": nh_fracs,
                "condition": fh_fracs,
            })

        features_by_fraction, y, groups = build_paired_features(
            question_data, n_fractions=2, n_features=10
        )
        assert len(features_by_fraction) == 2  # 2 fractions
        assert features_by_fraction[0].shape == (6, 10)  # 3 nh + 3 fh
        assert len(y) == 6
        assert len(groups) == 6
        assert set(y) == {0, 1}


class TestComputeDivergenceOnset:
    def test_clear_onset(self):
        false_auc = [0.51, 0.52, 0.55, 0.65, 0.80, 0.90]
        true_auc = [0.50, 0.51, 0.50, 0.52, 0.51, 0.53]
        onset = compute_divergence_onset(false_auc, true_auc, threshold=0.05, sustained=2)
        assert onset is not None
        assert onset <= 3

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


def build_paired_features(
    question_data: list[dict],
    n_fractions: int,
    n_features: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Build paired no-hint vs condition feature arrays from sparse data.

    Each question contributes exactly one no-hint and one condition sample.
    Each question's no-hint vector is paired once per format. The same no-hint
    features appear in up to 3 pairings when formats are pooled. This is
    acceptable because the classifier treats each pairing as an observation,
    and GroupKFold ensures all pairings from the same question stay together.

    Args:
        question_data: list of {question_id, no_hint: [sparse...], condition: [sparse...]}
        n_fractions: number of fraction points
        n_features: total SAE features

    Returns:
        features_by_fraction: {fraction_idx: np.ndarray [2*n_questions, n_features]}
        y: np.ndarray [2*n_questions] binary labels (0=no_hint, 1=condition)
        groups: np.ndarray [2*n_questions] question IDs for GroupKFold
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
            if f == 0:  # only set labels/groups once
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
        # Plot confidence band around the AUC gap (centered on true_auc + gap)
        gap = np.array([f - t for f, t in zip(false_auc, true_auc)])
        fractions_arr = np.array(fractions)
        true_arr = np.array(true_auc)
        # ci_lower/ci_upper are bounds on the gap itself, so the band is
        # true_auc + ci_lower to true_auc + ci_upper
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

    # Load metadata
    print("Loading metadata...")
    metadata = load_all_metadata(metadata_dir)
    no_hint_by_q, groups = build_experiment_groups(metadata)

    # Train/test split by question
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
            n_features = width_k * 1000
            print(f"\n=== {key} ===")

            # Build paired data for false-hint (one no-hint per question, not duplicated)
            train_pairs_false = []
            test_pairs_false = []
            train_pairs_true = []
            test_pairs_true = []

            for q_id in question_ids:
                nh_entry = no_hint_by_q.get(q_id)
                if not nh_entry:
                    continue
                nh_features = load_run_features(features_dir, nh_entry["run_id"], key)

                # Collect false-hint and true-hint across formats
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
                        cond_features = load_run_features(
                            features_dir, cond_entry["run_id"], key
                        )
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

            # FALSE-HINT analysis
            print(f"  False-hint: {len(train_pairs_false)} train, {len(test_pairs_false)} test pairs")
            train_feats, train_y, train_groups = build_paired_features(
                train_pairs_false, N_FRACTIONS, n_features
            )

            # Build training array (all fractions pooled)
            X_train = np.vstack([train_feats[f] for f in range(N_FRACTIONS)])
            del train_feats
            y_train = np.tile(train_y, N_FRACTIONS)
            g_train = np.tile(train_groups, N_FRACTIONS)

            print("  Tuning regularization...")
            best_C = tune_regularization(X_train, y_train, g_train)
            print(f"  Best C: {best_C}")

            print("  Training classifier...")
            clf = train_classifier(X_train, y_train, C=best_C)

            print("  Computing AUC per fraction (false-hint)...")
            test_feats_false, test_y_false, _ = build_paired_features(
                test_pairs_false, N_FRACTIONS, n_features
            )
            false_auc = compute_auc_per_fraction(clf, test_feats_false, test_y_false, N_FRACTIONS)

            # TRUE-HINT control
            print("  Computing true-hint control...")
            train_feats_true, train_y_true, train_groups_true = build_paired_features(
                train_pairs_true, N_FRACTIONS, n_features
            )
            X_train_true = np.vstack([train_feats_true[f] for f in range(N_FRACTIONS)])
            del train_feats_true
            y_train_true = np.tile(train_y_true, N_FRACTIONS)
            g_train_true = np.tile(train_groups_true, N_FRACTIONS)

            best_C_true = tune_regularization(X_train_true, y_train_true, g_train_true)
            clf_true = train_classifier(X_train_true, y_train_true, C=best_C_true)

            test_feats_true, test_y_true, _ = build_paired_features(
                test_pairs_true, N_FRACTIONS, n_features
            )
            true_auc = compute_auc_per_fraction(clf_true, test_feats_true, test_y_true, N_FRACTIONS)

            # Bootstrap CI for AUC gap at each fraction
            print("  Computing bootstrap CIs...")
            # Question IDs matching the paired array structure (2 rows per pair)
            test_qids = np.array([qid for p in test_pairs_false for qid in [p["question_id"]] * 2])
            ci_results = []
            for f in range(N_FRACTIONS):
                probs_f = clf.predict_proba(test_feats_false[f])[:, 1]
                probs_t = clf_true.predict_proba(test_feats_true[f])[:, 1]
                lower, upper = compute_bootstrap_ci(
                    probs_f, test_y_false, probs_t, test_y_true,
                    test_qids,
                )
                ci_results.append((lower, upper))
            ci_lower = [c[0] for c in ci_results]
            ci_upper = [c[1] for c in ci_results]

            # Divergence onset
            onset = compute_divergence_onset(false_auc, true_auc)

            # Top classifier weights
            top_k = 20
            weights = np.abs(clf.coef_[0])
            top_indices = np.argsort(weights)[-top_k:][::-1]
            top_features = [(int(idx), float(weights[idx])) for idx in top_indices]

            # Per-format AUC on test set (using the false-hint classifier)
            per_format_auc = {}
            for fmt in HINT_FORMATS:
                fmt_test_pairs = [p for p in test_pairs_false
                                  if p["hint_format"] == fmt]
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
            print(f"  False AUC: {[f'{a:.3f}' for a in false_auc]}")
            print(f"  True AUC:  {[f'{a:.3f}' for a in true_auc]}")
            print(f"  Onset: {FRACTION_POINTS[onset] if onset is not None else 'none'}")

            # Free memory
            del test_feats_false, test_feats_true
            del X_train, X_train_true

    # Text similarity baseline
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

    # Save results
    output = {
        "layer_width_results": all_results,
        "text_similarity_baseline": avg_text_sim,
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "fraction_points": FRACTION_POINTS,
    }
    with open(results_dir / "divergence_results.json", "w") as f:
        json.dump(output, f, indent=2)

    # Summary plot for best layer
    best_key = min(all_results, key=lambda k: all_results[k]["onset_fraction_idx"]
                   if all_results[k]["onset_fraction_idx"] is not None else 999)
    best = all_results[best_key]
    plot_auc_curves(
        best["false_auc"], best["true_auc"], FRACTION_POINTS,
        f"Divergence Localization — {best_key}",
        figures_dir / "auc_best_with_text.png",
        text_sim=avg_text_sim,
    )

    # Summary
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_divergence_analysis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_divergence_analysis.py tests/test_divergence_analysis.py
git commit -m "feat: add Phase 2 classification with GroupKFold CV and bootstrap CIs"
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
```

- [ ] **Step 2: Make executable and verify syntax**

```bash
chmod +x scripts/launch_divergence.sh
bash -n scripts/launch_divergence.sh && echo "Syntax OK"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/launch_divergence.sh
git commit -m "feat: add SLURM launch script for divergence localization pipeline"
```

---

### Task 10: Add Dependencies to Environment

**Files:**
- Modify: `environment.yml`

- [ ] **Step 1: Check current dependencies**

Run: `conda run -n cot_sae python -c "import sklearn; print(sklearn.__version__)"` and
`conda run -n cot_sae python -c "from sentence_transformers import SentenceTransformer; print('OK')"`

- [ ] **Step 2: Add missing dependencies**

Add to `environment.yml` pip section:
```yaml
    - scikit-learn
    - sentence-transformers
```

- [ ] **Step 3: Install**

```bash
conda run -n cot_sae pip install scikit-learn sentence-transformers
```

- [ ] **Step 4: Verify**

```bash
conda run -n cot_sae python -c "from sklearn.linear_model import LogisticRegression; from sentence_transformers import SentenceTransformer; print('OK')"
```

- [ ] **Step 5: Commit**

```bash
git add environment.yml
git commit -m "deps: add scikit-learn and sentence-transformers"
```

---

### Task 11: Integration Test

**Files:**
- Create: `tests/test_integration_divergence.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration_divergence.py
# ABOUTME: Smoke test for the full divergence localization pipeline.
# ABOUTME: Validates end-to-end data flow with synthetic data.

import pytest
import numpy as np
import torch
from src.fractional import to_sparse_features, from_sparse_features
from src.classifier import tune_regularization, train_classifier, compute_auc_per_fraction
from scripts.run_divergence_analysis import build_paired_features, compute_divergence_onset


class TestEndToEndPipeline:
    def test_sparse_to_classifier_flow(self):
        """Sparse features → paired assembly → classifier → AUC curve."""
        n_questions = 30
        n_fractions = 5
        n_features = 100
        rng = np.random.RandomState(42)

        question_data = []
        for q in range(n_questions):
            nh_fracs = []
            cond_fracs = []
            for f in range(n_fractions):
                nh_dense = torch.zeros(n_features)
                active = rng.choice(n_features, 5, replace=False)
                nh_dense[active] = torch.from_numpy(rng.randn(5).astype(np.float32))
                nh_fracs.append(to_sparse_features(nh_dense))

                cond_dense = torch.zeros(n_features)
                cond_dense[active] = torch.from_numpy(
                    (rng.randn(5) + (f + 1) * 0.5).astype(np.float32)
                )
                cond_fracs.append(to_sparse_features(cond_dense))

            question_data.append({
                "question_id": q,
                "no_hint": nh_fracs,
                "condition": cond_fracs,
            })

        features_by_fraction, y, groups = build_paired_features(
            question_data, n_fractions=n_fractions, n_features=n_features
        )

        X_all = np.vstack([features_by_fraction[f] for f in range(n_fractions)])
        y_all = np.tile(y, n_fractions)
        g_all = np.tile(groups, n_fractions)

        best_C = tune_regularization(X_all, y_all, g_all, n_folds=3)
        clf = train_classifier(X_all, y_all, C=best_C)
        auc_curve = compute_auc_per_fraction(clf, features_by_fraction, y, n_fractions)

        assert len(auc_curve) == n_fractions
        assert auc_curve[-1] >= auc_curve[0]

    def test_divergence_onset_detection(self):
        """Onset detection with synthetic AUC curves."""
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

- **HuggingFace generate() with attention masks**: Batched generation with left-padding is well-supported for Gemma 2 (tokenizer defaults to `padding_side="left"`). Verify during Task 3 implementation that the attention mask propagates correctly through sliding window attention layers.
- **No-hint shared across formats but not duplicated**: Each no-hint vector is used once per format pairing in `build_paired_features`, but the same underlying activation is shared. Per-format AUC differences are driven entirely by the hint conditions. This is documented and acceptable.
- **Phase 2 memory**: `build_paired_features` converts sparse to dense arrays. For 65k width with ~3,000 questions per layer/width: ~3,000 × 2 × 65,536 × 4 bytes ≈ 1.5 GB per fraction, 20 fractions ≈ 30 GB. Fits in 256 GB. Arrays are freed between layer/width iterations.
- **Text similarity**: Uses whitespace tokenization for truncation, which does not exactly match the model's BPE tokenizer. This is an approximation for the baseline control.
