"""Meaning vectors for words and random drift of those meanings over time."""

import hashlib
import logging

import numpy as np

from .config import SENTENCE_TRANSFORMER_MODEL

logger = logging.getLogger(__name__)


class HashEmbedder:
    """Deterministic stand-in embedder used when sentence-transformers is not installed."""

    def encode(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0
        return _normalise(vec)


class SemanticDrift:
    """Embeds concepts and perturbs their vectors to simulate meaning change."""

    def __init__(self, model_name: str | None = None, rng: np.random.Generator | None = None):
        self.model_name = model_name or SENTENCE_TRANSFORMER_MODEL
        self.rng = rng or np.random.default_rng()
        self._model = None

    @property
    def model(self):
        """Load the sentence-transformers model on first use, falling back to hashing."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                logger.warning(
                    "sentence-transformers is not installed; using hash embeddings. "
                    'Install with: pip install -e ".[semantic]"'
                )
                self._model = HashEmbedder()
            else:
                self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_concept(self, concept_description: str) -> np.ndarray:
        return _normalise(np.asarray(self.model.encode(concept_description), dtype=np.float32))

    def similarity(self, concept_a: str, concept_b: str) -> float:
        """Cosine similarity between two concept descriptions."""
        return float(np.dot(self.embed_concept(concept_a), self.embed_concept(concept_b)))

    def simulate_drift(self, vector: np.ndarray, drift_strength: float = 0.05) -> np.ndarray:
        """Add Gaussian noise to a meaning vector and re-normalise it."""
        return _normalise(vector + self.rng.normal(0, drift_strength, vector.shape))


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm
