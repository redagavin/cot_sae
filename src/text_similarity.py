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
