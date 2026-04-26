import sys
import os
import argparse
import pandas as pd
import numpy as np

# Add src to the path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from genetics import RuleExtractor, EvolutionaryEngine, Organism
from fitness import get_portuguese_fitness_model
from semantics import SemanticDrift
from config import (
    LEXICON_PATH, 
    DEFAULT_GENERATIONS, 
    DEFAULT_POPULATION_SIZE, 
    SEMANTIC_DRIFT_STRENGTH
)

def run_simulation(word: str, generations: int = None, pop_size: int = None):
    gens = generations or DEFAULT_GENERATIONS
    pop = pop_size or DEFAULT_POPULATION_SIZE
    
    print(f"--- Simulação Evolutiva Neo-Românica: {word} ---")
    
    if not os.path.exists(LEXICON_PATH):
        print(f"Lexicon not found at {LEXICON_PATH}. Run data_fetcher.py first.")
        return

    df = pd.read_csv(LEXICON_PATH)
    extractor = RuleExtractor()
    for _, row in df.iterrows():
        extractor.extract_rules(row['latin'], row['portuguese'])
    
    rules = extractor.get_rules(min_count=2)
    print(f"Rules extracted: {len(rules)}")
    
    fitness_model = get_portuguese_fitness_model()
    
    semantic_module = SemanticDrift()
    initial_vector = semantic_module.embed_concept(f"the concept of {word}")
    
    def composite_fitness(org):
        # 1. Phonotactic Score (Normalized Log-Likelihood)
        # We assume -10 is 'terrible' and -1.0 is 'excellent'
        raw_p = fitness_model.score(org.word)
        p_score = max(0, (raw_p + 10) / 9.0) * 50 # 50% of weight
        
        # 2. Naturalness Index (Based on rule history)
        n_score = org.get_average_naturalness() * 30 # 30% of weight
        
        # 3. Semantic Similarity
        sim = np.dot(org.meaning_vector, initial_vector)
        s_score = max(0, sim) * 10 # 10% of weight
        
        # 4. Romance Conventions (Ending bonus)
        ending_bonus = 10.0 if org.word.endswith(('a', 'e', 'o', 'm', 's')) else 0.0
        
        # Total: 100 points scale
        return p_score + n_score + s_score + ending_bonus

    initial_org = Organism(word, initial_vector)
    engine = EvolutionaryEngine(initial_org, rules)
    
    def drift_fn(vec):
        return semantic_module.simulate_drift(vec, drift_strength=SEMANTIC_DRIFT_STRENGTH)

    results = engine.evolve(
        generations=gens,
        population_size=pop,
        fitness_fn=composite_fitness,
        drift_fn=drift_fn
    )
    
    print(f"\nResultados Neo-Românicos (Top 10):")
    for i, org in enumerate(results[:10]):
        sim = np.dot(org.meaning_vector, initial_vector)
        print(f"{i+1}. {org.word:12} | Fitness: {org.fitness:5.1f} | Nat: {org.get_average_naturalness():.2f} | Sem: {sim:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulação Neo-Românica')
    parser.add_argument('word', type=str, help='Palavra latina inicial')
    parser.add_argument('--gens', type=int, help='Gerações')
    parser.add_argument('--pop', type=int, help='População')
    
    args = parser.parse_args()
    run_simulation(args.word, args.gens, args.pop)
