import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

# Data Files
LEXICON_PATH = DATA_DIR / "lexicon.csv"
NEO_DICT_PATH = DATA_DIR / "neo_romance_dict.json"

# Simulation Constants
DEFAULT_GENERATIONS = int(os.getenv("EVO_GENERATIONS", 15))
DEFAULT_POPULATION_SIZE = int(os.getenv("EVO_POP_SIZE", 100))
MUTATION_RATE = float(os.getenv("EVO_MUTATION_RATE", 0.4))
CROSSOVER_RATE = float(os.getenv("EVO_CROSSOVER_RATE", 0.3))
SEMANTIC_DRIFT_STRENGTH = float(os.getenv("SEMANTIC_DRIFT_STRENGTH", 0.02))

# Model Settings
SENTENCE_TRANSFORMER_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

# Phonetic Naturalness Threshold
NATURALNESS_THRESHOLD = 2.0
