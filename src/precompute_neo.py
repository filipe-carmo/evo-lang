import pandas as pd
import json
from src.genetics import RuleExtractor, EvolutionaryEngine, Organism
from src.fitness import get_portuguese_fitness_model
from src.semantics import SemanticDrift
from src.config import LEXICON_PATH, NEO_DICT_PATH

def precompute():
    print("Starting Pre-computation for Neo-Romance Dictionary...")
    if not LEXICON_PATH.exists():
        print(f"Lexicon not found at {LEXICON_PATH}")
        return

    df = pd.read_csv(LEXICON_PATH)
    # Filter for valid string pairs
    df = df.dropna(subset=['latin', 'portuguese'])
    df = df[df['latin'].apply(lambda x: isinstance(x, str))]
    df = df[df['portuguese'].apply(lambda x: isinstance(x, str))]
    
    # 1. Extract rules from ALL languages to create the Neo-Romance 'mixture'
    extractor = RuleExtractor()
    for _, row in df.iterrows():
        extractor.extract_rules(row['latin'], row['portuguese'])
    
    rules = extractor.get_rules(min_count=3) # Filter for consensus rules
    print(f"Consensus rules extracted: {len(rules)}")
    
    # 2. Setup Models
    fitness_model = get_portuguese_fitness_model() # Generic romance phonotactics
    semantic_module = SemanticDrift()
    
    # 3. Process the top unique latin words in our dataset
    unique_latin = df['latin'].unique()[:500] # Top 500 for demo
    
    neo_dict = {}
    
    for i, word in enumerate(unique_latin):
        initial_vector = semantic_module.embed_concept(f"concept of {word}")
        initial_org = Organism(word, initial_vector)
        engine = EvolutionaryEngine(initial_org, rules)
        
        def composite_fitness(org):
            p_score = fitness_model.score(org.word)
            p_score = max(0, (p_score + 10) / 9.0) * 50
            n_score = org.get_average_naturalness() * 30
            ending_bonus = 10.0 if org.word.endswith(('a', 'e', 'o', 'm', 's')) else 0.0
            return p_score + n_score + ending_bonus

        # Higher generations for more drift
        results = engine.evolve(generations=25, population_size=60, fitness_fn=composite_fitness)
        
        if results:
            best_variant = results[0].word
            neo_dict[word] = best_variant
            
        if i % 50 == 0:
            print(f"  Evolved {i}/{len(unique_latin)} words...")

    # Save dictionary
    with NEO_DICT_PATH.open('w', encoding='utf-8') as f:
        json.dump(neo_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Neo-Romance Dictionary saved to {NEO_DICT_PATH}")

if __name__ == "__main__":
    precompute()
