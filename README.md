# 🌍 Neo-Romance Evolution Simulator

This project uses **Genetic Algorithms (GA)** and **NLP** to simulate linguistic drift from Latin to Romance languages, creating a synthesized **"Neo-Romance"** vocabulary.

## 🧬 How it Works

The system treats words as "organisms" that evolve over generations:
1. **Rule Extraction**: Phonetic transition rules (e.g., `t -> d`, `i -> e`) are extracted from a lexicon of Latin-to-Portuguese transformations.
2. **Genetic Algorithm**: For each Latin word, a population of phonetic variants is generated.
3. **Fitness Function**: Variants are scored based on:
    - **Phonotactics**: How much they sound like a Romance language (using N-gram models).
    - **Naturalness**: The phonetic plausibility of the applied transformations.
    - **Semantic Stability**: Using embedding models to ensure the "concept" remains intact during drift.

## 🚀 Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a full evolution (Pre-compute Dictionary):**
   This will evolve 500+ Latin roots and save them to `data/neo_romance_dict.json`.
   ```bash
   python src/precompute_neo.py
   ```

3. **Run the Integrated Test:**
   See a step-by-step evolution of the word *nocte* (night).
   ```bash
   python tests/test_integrated_evolution.py
   ```

## 📂 Project Structure

- `src/genetics.py`: The core GA engine (Selection, Crossover, Mutation).
- `src/fitness.py`: N-gram models for scoring phonetic plausibility.
- `src/phonetics.py`: Mapping of phonetic features and transition naturalness.
- `src/semantics.py`: Semantic drift simulation using Sentence-Transformers.
- `src/precompute_neo.py`: CLI tool for mass word evolution.
- `data/lexicon.csv`: Training data scraped from etymological sources.
- `data/neo_romance_dict.json`: The final evolved "Neo-Romance" dictionary.

---
*Created with Gemini CLI*
