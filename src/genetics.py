import difflib
import random
import numpy as np
from typing import List, Tuple, Dict, Any, Callable
from phonetics import is_natural_transition, get_feature_distance

class PhoneticRule:
    """Represents a phonetic transition rule with naturalness scoring."""
    def __init__(self, source: str, target: str, count: int = 1):
        self.source = source
        self.target = target
        self.count = count
        self.probability = 0.0
        
        dist = 0.0
        if len(source) == 1 and len(target) == 1:
            dist = get_feature_distance(source, target)
        elif len(source) > 0 and len(target) > 0:
             dist = sum(get_feature_distance(s, t) for s, t in zip(source, target)) / max(len(source), len(target))
        else:
            # Insertion or Deletion
            dist = 3.5
        
        # SOTA: Naturalness score (0.1 to 1.0)
        self.nature_score = max(0.1, 1.0 - (dist / 10.0))

    def apply(self, word: str) -> str:
        if self.source in word:
            return word.replace(self.source, self.target, 1)
        return word

    def __repr__(self):
        return f"Rule({self.source} -> {self.target}, p={self.probability:.2f}, n={self.nature_score:.2f})"

class RuleExtractor:
    """Extracts phonetic transition rules from word pairs."""
    def __init__(self):
        self.rules: Dict[Tuple[str, str], int] = {}

    def extract_rules(self, source_word: str, target_word: str):
        s = difflib.SequenceMatcher(None, source_word, target_word)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag in ['replace', 'delete', 'insert']:
                rule_key = (source_word[i1:i2], target_word[j1:j2])
                self.rules[rule_key] = self.rules.get(rule_key, 0) + 1

    def get_rules(self, min_count: int = 2) -> List[PhoneticRule]:
        # Filter rules by minimum frequency and EXCLUDE empty source (insertions)
        # to prevent random junk prefixes
        filtered_rules = {k: v for k, v in self.rules.items() if v >= min_count and k[0] != ""}
        total = sum(filtered_rules.values())
        if total == 0: return []
        
        extracted = []
        for (src, tgt), count in filtered_rules.items():
            rule = PhoneticRule(src, tgt, count)
            rule.probability = count / total
            extracted.append(rule)
        return sorted(extracted, key=lambda x: x.count, reverse=True)

class Organism:
    """A candidate word in the population."""
    def __init__(self, word: str, meaning_vector: np.ndarray, history: List[PhoneticRule] = None):
        self.word = word
        self.meaning_vector = meaning_vector
        self.history = history or []
        self.fitness = 0.0

    def get_average_naturalness(self) -> float:
        """Returns the average naturalness of applied rules."""
        if not self.history: return 1.0
        return sum(r.nature_score for r in self.history) / len(self.history)

    def __repr__(self):
        return f"Org('{self.word}', fit={self.fitness:.2f}, nat={self.get_average_naturalness():.2f})"

class EvolutionaryEngine:
    """GA Engine with phonetic mutation and linguistic crossover."""
    def __init__(self, initial_organism: Organism, rules: List[PhoneticRule]):
        self.population = [initial_organism]
        self.rules = rules

    def mutate(self, organism: Organism, mutation_rate: float = 0.4, drift_fn=None) -> Organism:
        new_word = organism.word
        new_history = list(organism.history)
        
        if random.random() < mutation_rate and self.rules:
            weights = [r.probability * r.nature_score for r in self.rules]
            rule = random.choices(self.rules, weights=weights)[0]
            new_word = rule.apply(new_word)
            new_history.append(rule)
        
        new_vector = drift_fn(organism.meaning_vector) if drift_fn else organism.meaning_vector
        return Organism(new_word, new_vector, new_history)

    def crossover(self, parent1: Organism, parent2: Organism) -> Organism:
        p1, p2 = parent1.word, parent2.word
        split1 = len(p1) // 2
        split2 = len(p2) // 2
        new_word = p1[:split1] + p2[split2:]
        
        new_vector = (parent1.meaning_vector + parent2.meaning_vector) / 2.0
        new_history = list(set(parent1.history + parent2.history))
        return Organism(new_word, new_vector, new_history)

    def evolve(self, generations: int = 10, population_size: int = 50, 
               fitness_fn=None, drift_fn=None):
        self.population = [self.population[0]] * population_size
        
        for gen in range(generations):
            new_population = []
            
            # Elitism: keep top 10%
            if fitness_fn:
                for org in self.population: org.fitness = fitness_fn(org)
                self.population.sort(key=lambda x: x.fitness, reverse=True)
                new_population.extend(self.population[:max(1, population_size // 10)])
            
            while len(new_population) < population_size:
                if random.random() < 0.3 and len(self.population) > 2:
                    p1, p2 = random.sample(self.population[:population_size//2], 2)
                    child = self.crossover(p1, p2)
                else:
                    parent = random.choice(self.population[:population_size//2])
                    child = self.mutate(parent, drift_fn=drift_fn)
                new_population.append(child)
            
            self.population = new_population

        for org in self.population: org.fitness = fitness_fn(org)
        results = {}
        for org in self.population:
            if org.word not in results or org.fitness > results[org.word].fitness:
                results[org.word] = org
        return sorted(results.values(), key=lambda x: x.fitness, reverse=True)
