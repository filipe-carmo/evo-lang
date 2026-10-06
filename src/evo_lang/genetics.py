"""Genetic algorithm over word forms: sound-change rules, organisms and the evolution loop."""

import difflib
import random
from collections import Counter
from collections.abc import Callable

import numpy as np

from .config import CROSSOVER_RATE, MUTATION_RATE
from .phonetics import get_feature_distance

# Distance assigned to rules that delete or insert sounds rather than substitute them.
INDEL_DISTANCE = 3.5


class PhoneticRule:
    """A sound change "source -> target" with its corpus frequency and naturalness score."""

    def __init__(self, source: str, target: str, count: int = 1):
        self.source = source
        self.target = target
        self.count = count
        self.probability = 0.0
        self.nature_score = max(0.1, 1.0 - self._distance() / 10.0)

    def _distance(self) -> float:
        if not self.source or not self.target:
            return INDEL_DISTANCE
        pairs = zip(self.source, self.target, strict=False)
        return sum(get_feature_distance(s, t) for s, t in pairs) / max(len(self.source), len(self.target))

    def apply(self, word: str) -> str:
        """Apply the rule to the first occurrence of its source sequence."""
        return word.replace(self.source, self.target, 1)

    def __repr__(self) -> str:
        return f"Rule({self.source!r} -> {self.target!r}, p={self.probability:.2f}, n={self.nature_score:.2f})"


class RuleExtractor:
    """Counts the edit operations that turn Latin words into their Romance descendants."""

    def __init__(self) -> None:
        self.rules: Counter[tuple[str, str]] = Counter()

    def extract_rules(self, source_word: str, target_word: str) -> None:
        matcher = difflib.SequenceMatcher(None, source_word, target_word)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                self.rules[(source_word[i1:i2], target_word[j1:j2])] += 1

    def get_rules(self, min_count: int = 2) -> list[PhoneticRule]:
        """Rules seen at least ``min_count`` times, most frequent first.

        Pure insertions (empty source) are excluded: they match anywhere and
        produce random junk prefixes.
        """
        kept = {k: v for k, v in self.rules.items() if v >= min_count and k[0]}
        total = sum(kept.values())
        rules = []
        for (src, tgt), count in kept.items():
            rule = PhoneticRule(src, tgt, count)
            rule.probability = count / total
            rules.append(rule)
        return sorted(rules, key=lambda r: r.count, reverse=True)


class Organism:
    """A candidate word form, its meaning vector and the rules that produced it."""

    def __init__(self, word: str, meaning_vector: np.ndarray, history: list[PhoneticRule] | None = None):
        self.word = word
        self.meaning_vector = meaning_vector
        self.history = history or []
        self.fitness = 0.0

    def get_average_naturalness(self) -> float:
        if not self.history:
            return 1.0
        return sum(r.nature_score for r in self.history) / len(self.history)

    def __repr__(self) -> str:
        return f"Organism({self.word!r}, fitness={self.fitness:.2f}, nat={self.get_average_naturalness():.2f})"


FitnessFn = Callable[[Organism], float]
DriftFn = Callable[[np.ndarray], np.ndarray]


class EvolutionaryEngine:
    """Elitist GA: mutation applies a weighted random sound change, crossover splices two words."""

    def __init__(
        self,
        initial_organism: Organism,
        rules: list[PhoneticRule],
        rng: random.Random | None = None,
        mutation_rate: float = MUTATION_RATE,
        crossover_rate: float = CROSSOVER_RATE,
    ):
        self.population = [initial_organism]
        self.rules = rules
        self.rng = rng or random.Random()
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self._weights = [r.probability * r.nature_score for r in rules]

    def mutate(self, organism: Organism, drift_fn: DriftFn | None = None) -> Organism:
        new_word = organism.word
        new_history = list(organism.history)
        if self.rules and self.rng.random() < self.mutation_rate:
            rule = self.rng.choices(self.rules, weights=self._weights)[0]
            new_word = rule.apply(new_word)
            new_history.append(rule)
        new_vector = drift_fn(organism.meaning_vector) if drift_fn else organism.meaning_vector
        return Organism(new_word, new_vector, new_history)

    @staticmethod
    def crossover(parent1: Organism, parent2: Organism) -> Organism:
        """Child = first half of parent1 + second half of parent2."""
        p1, p2 = parent1.word, parent2.word
        new_word = p1[: len(p1) // 2] + p2[len(p2) // 2 :]
        new_vector = (parent1.meaning_vector + parent2.meaning_vector) / 2.0
        # Keep each rule once, preserving order
        new_history = list(dict.fromkeys(parent1.history + parent2.history))
        return Organism(new_word, new_vector, new_history)

    def evolve(
        self,
        fitness_fn: FitnessFn,
        generations: int = 10,
        population_size: int = 50,
        drift_fn: DriftFn | None = None,
    ) -> list[Organism]:
        """Run the GA and return unique word forms, best first."""
        self.population = [self.population[0]] * population_size
        n_elite = max(1, population_size // 10)
        n_parents = max(2, population_size // 2)

        for _ in range(generations):
            self._score(fitness_fn)
            parents = self.population[:n_parents]
            new_population = self.population[:n_elite]
            while len(new_population) < population_size:
                if self.rng.random() < self.crossover_rate:
                    child = self.crossover(*self.rng.sample(parents, 2))
                else:
                    child = self.mutate(self.rng.choice(parents), drift_fn=drift_fn)
                new_population.append(child)
            self.population = new_population

        self._score(fitness_fn)
        best_by_word: dict[str, Organism] = {}
        for org in self.population:
            if org.word not in best_by_word:  # population is sorted, first is best
                best_by_word[org.word] = org
        return list(best_by_word.values())

    def _score(self, fitness_fn: FitnessFn) -> None:
        for org in self.population:
            org.fitness = fitness_fn(org)
        self.population.sort(key=lambda o: o.fitness, reverse=True)
