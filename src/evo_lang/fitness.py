"""Scoring functions: how "Romance-like" and how plausible an evolved word is."""

import math
from collections import Counter
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .genetics import Organism
from .lexicon import ROMANCE_COL

ROMANCE_ENDINGS = ("a", "e", "o", "m", "s")
# Common words that complement the lexicon when training the phonotactic model
EXTRA_TRAINING_WORDS = ["casa", "tempo", "vida", "homem", "mulher", "grande", "fazer", "dizer"]


class PhonotacticFitness:
    """Character n-gram model: how much a word looks like the Romance training words."""

    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams: Counter[str] = Counter()
        self.total_ngrams = 0

    def _grams(self, word: str) -> Iterable[str]:
        padded = f"^{word}$"
        return (padded[i : i + self.n] for i in range(len(padded) - self.n + 1))

    def train(self, words: Iterable[str]) -> None:
        for word in words:
            for gram in self._grams(word):
                self.ngrams[gram] += 1
                self.total_ngrams += 1

    def score(self, word: str) -> float:
        """Length-normalised log-likelihood with Laplace smoothing (higher is better)."""
        if self.total_ngrams == 0:
            return 0.0
        denom = self.total_ngrams + len(self.ngrams)
        log_prob = sum(math.log((self.ngrams.get(g, 0) + 1) / denom) for g in self._grams(word))
        return log_prob / (len(word) + 2)


def train_phonotactic_model(lexicon: pd.DataFrame, n: int = 3) -> PhonotacticFitness:
    """Train the n-gram model on every Romance word in the lexicon."""
    model = PhonotacticFitness(n=n)
    model.train(lexicon[ROMANCE_COL])
    model.train(EXTRA_TRAINING_WORDS)
    return model


class CompositeFitness:
    """
    Weighted score on a 0-100 scale.

    - 50 pts phonotactics: does it sound Romance? (log-likelihood mapped from [-10, -1])
    - 30 pts naturalness: were the sound changes phonetically plausible?
    - 10 pts semantics: is the meaning vector still close to the original? (optional)
    - 10 pts ending: does it end like a typical Romance word?

    With ``source_word`` set, words that eroded below 60% of the original
    length are scaled down, so "novus" cannot collapse into a lone "o".
    """

    def __init__(
        self,
        phonotactics: PhonotacticFitness,
        reference_vector: np.ndarray | None = None,
        source_word: str | None = None,
    ):
        self.phonotactics = phonotactics
        self.reference_vector = reference_vector
        self.min_length = 0.6 * len(source_word) if source_word else 0.0

    def phonotactic_points(self, word: str) -> float:
        return min(1.0, max(0.0, (self.phonotactics.score(word) + 10) / 9.0)) * 50

    def semantic_similarity(self, organism: Organism) -> float:
        if self.reference_vector is None:
            return 1.0
        return float(np.dot(organism.meaning_vector, self.reference_vector))

    def __call__(self, organism: Organism) -> float:
        word = organism.word
        if not word:
            return 0.0
        points = self.phonotactic_points(word)
        points += organism.get_average_naturalness() * 30
        points += max(0.0, self.semantic_similarity(organism)) * 10
        points += 10.0 if word.endswith(ROMANCE_ENDINGS) else 0.0
        if len(word) < self.min_length:
            points *= len(word) / self.min_length
        return points
