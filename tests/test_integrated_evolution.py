import sys
import os
import pandas as pd
import numpy as np

# Add src to the path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from genetics import RuleExtractor, EvolutionaryEngine, Organism
from fitness import get_portuguese_fitness_model
from semantics import SemanticDrift

def main():
    print("--- Evolutionary Linguistic Simulation ---")
    
    # 1. Extract phonetic rules
    print("Extracting phonetic rules...")
    lexicon = pd.read_csv('data/lexicon.csv')
    extractor = RuleExtractor()
    for _, row in lexicon.iterrows():
        extractor.extract_rules(row['latin'], row['portuguese'])
    rules = extractor.get_rules()
    
    # 2. Setup Phonotactic Fitness
    print("Training phonotactic fitness model...")
    fitness_model = get_portuguese_fitness_model('data/lexicon.csv')
    
    # 3. Setup Semantic Drift
    print("Initializing semantic drift module...")
    semantic_module = SemanticDrift()
    
    # Initial concept for 'nocte'
    initial_concept = "o período entre o pôr do sol e o nascer do sol"
    initial_vector = semantic_module.embed_concept(initial_concept)
    
    # 4. Define Composite Fitness
    def composite_fitness(organism: Organism) -> float:
        # Phonotactic score (usually negative log-prob, so higher is better)
        phon_score = fitness_model.score(organism.word)
        
        # Penalize length differences if too extreme
        length_penalty = -abs(len(organism.word) - 5) * 0.1
        
        # Add a bonus for common Portuguese endings
        ending_bonus = 0.5 if organism.word.endswith(('e', 'o', 'a')) else 0.0
        
        return (phon_score * 10) + length_penalty + ending_bonus

    # 5. Evolutionary Run
    start_word = "nocte"
    print(f"\nStarting evolution for: '{start_word}'")
    
    initial_org = Organism(start_word, initial_vector)
    engine = EvolutionaryEngine(initial_org, rules)
    
    # Define drift function for the engine
    def drift_fn(vec):
        return semantic_module.simulate_drift(vec, drift_strength=0.02)

    generations = 8
    population_size = 30
    results = engine.evolve(
        generations=generations, 
        population_size=population_size, 
        fitness_fn=composite_fitness,
        drift_fn=drift_fn
    )
    
    print("\nTop 5 Evolved Variants:")
    for org in results[:5]:
        # Calculate semantic drift compared to original
        similarity = np.dot(org.meaning_vector, initial_vector)
        print(f"  Word: {org.word:10} | Fitness: {org.fitness:6.2f} | Semantic Sim: {similarity:4.2f}")

if __name__ == "__main__":
    main()
