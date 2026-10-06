"""Paths and tunable simulation settings (overridable through environment variables)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("EVO_DATA_DIR", BASE_DIR / "data"))

LEXICON_PATH = DATA_DIR / "lexicon.csv"
NEO_DICT_PATH = DATA_DIR / "neo_romance_dict.json"

DEFAULT_GENERATIONS = int(os.getenv("EVO_GENERATIONS", "15"))
DEFAULT_POPULATION_SIZE = int(os.getenv("EVO_POP_SIZE", "100"))
MUTATION_RATE = float(os.getenv("EVO_MUTATION_RATE", "0.4"))
CROSSOVER_RATE = float(os.getenv("EVO_CROSSOVER_RATE", "0.3"))
SEMANTIC_DRIFT_STRENGTH = float(os.getenv("SEMANTIC_DRIFT_STRENGTH", "0.02"))

SENTENCE_TRANSFORMER_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Max feature distance for a sound change to count as "natural" (e.g. voicing t -> d).
NATURALNESS_THRESHOLD = 2.0
