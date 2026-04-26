import numpy as np
from typing import Dict, List, Optional
from config import SENTENCE_TRANSFORMER_MODEL

class SemanticDrift:
    """Simulates semantic drift using vector spaces."""
    def __init__(self, model_name: str = None):
        self.model_name = model_name or SENTENCE_TRANSFORMER_MODEL
        self._model = None
        self.word_meanings: Dict[str, np.ndarray] = {}

    @property
    def model(self):
        """Lazy loader for sentence-transformers."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                print("Warning: sentence-transformers not found. Using dummy model.")
                self._model = DummyModel()
        return self._model

    def embed_concept(self, concept_description: str) -> np.ndarray:
        """Converts a concept description to a vector."""
        return self.model.encode(concept_description)

    def calculate_drift(self, initial_concept: str, target_concept: str) -> float:
        """Calculates semantic distance between two concepts."""
        v1 = self.embed_concept(initial_concept)
        v2 = self.embed_concept(target_concept)
        # Cosine similarity
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def simulate_drift(self, initial_vector: np.ndarray, drift_strength: float = 0.05) -> np.ndarray:
        """Adds random noise to a vector to simulate meaning change over time."""
        noise = np.random.normal(0, drift_strength, initial_vector.shape)
        new_vector = initial_vector + noise
        norm = np.linalg.norm(new_vector)
        if norm == 0: return new_vector
        return new_vector / norm

class DummyModel:
    """A placeholder for when sentence-transformers is missing."""
    def encode(self, text: str) -> np.ndarray:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
        norm = np.linalg.norm(vec)
        if norm == 0: return vec
        return vec / norm

class WordMeaning:
    """Stores a word and its associated semantic vector."""
    def __init__(self, word: str, concept_vector: np.ndarray):
        self.word = word
        self.vector = concept_vector

    def __repr__(self):
        return f"WordMeaning({self.word}, vector_dim={self.vector.shape})"
