import math
import pandas as pd
from collections import Counter
from typing import List
from config import LEXICON_PATH

class PhonotacticFitness:
    """Evaluates how much a word 'looks like' Portuguese using N-gram models."""
    def __init__(self, n: int = 3):
        self.n = n
        self.ngrams = Counter()
        self.total_ngrams = 0

    def train(self, words: List[str]):
        """Trains the model on a list of valid Portuguese words."""
        for word in words:
            if not isinstance(word, str): continue
            word = f"^{word}$"  # Start and end markers
            for i in range(len(word) - self.n + 1):
                gram = word[i:i+self.n]
                self.ngrams[gram] += 1
                self.total_ngrams += 1

    def score(self, word: str) -> float:
        """Calculates the log-likelihood of a word based on trained N-grams."""
        if self.total_ngrams == 0:
            return 0.0
        
        word = f"^{word}$"
        log_prob = 0.0
        for i in range(len(word) - self.n + 1):
            gram = word[i:i+self.n]
            # Laplacian smoothing
            count = self.ngrams.get(gram, 0) + 1
            prob = count / (self.total_ngrams + len(self.ngrams))
            log_prob += math.log(prob)
        
        # Normalize by length to avoid penalizing long words unfairly
        return log_prob / len(word)

def get_portuguese_fitness_model(lexicon_path: str = None) -> PhonotacticFitness:
    """Helper to quickly train a model from the lexicon."""
    path = lexicon_path or LEXICON_PATH
    df = pd.read_csv(path)
    model = PhonotacticFitness(n=3)
    # Train on all romance target words in the lexicon
    target_words = df['portuguese'].dropna().tolist()
    model.train(target_words)
    # Add some common PT words to boost the model
    extra_words = ["casa", "tempo", "vida", "homem", "mulher", "grande", "fazer", "dizer"]
    model.train(extra_words)
    return model
