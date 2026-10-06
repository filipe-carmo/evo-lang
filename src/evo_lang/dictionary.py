"""Batch-evolve Latin words into the Neo-Romance dictionary."""

import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LEXICON_PATH, NEO_DICT_PATH
from .fitness import CompositeFitness, train_phonotactic_model
from .genetics import EvolutionaryEngine, Organism
from .lexicon import LATIN_COL, extract_rules, load_lexicon, strip_diacritics

logger = logging.getLogger(__name__)


def build_dictionary(
    words: Iterable[str] | None = None,
    lexicon_path: Path = LEXICON_PATH,
    output_path: Path | None = NEO_DICT_PATH,
    max_words: int = 500,
    generations: int = 25,
    population_size: int = 60,
    seed: int | None = 42,
) -> dict[str, str]:
    """
    Evolve Latin words and return {latin: neo_romance}.

    ``words`` defaults to the vocabulary already in the dictionary at
    ``output_path`` (so a rebuild re-evolves the same words), or else the
    first ``max_words`` Latin roots in the lexicon.

    Meaning vectors are not needed here (no drift), so this runs without an
    embedding model. Results are reproducible for a given ``seed``.
    """
    lexicon = load_lexicon(lexicon_path)
    rules = extract_rules(lexicon, min_count=3)
    logger.info("Extracted %d consensus rules from %d word pairs", len(rules), len(lexicon))

    phonotactics = train_phonotactic_model(lexicon)
    rng = random.Random(seed)
    placeholder_vector = np.zeros(1, dtype=np.float32)

    if words is None:
        words = _default_words(lexicon, output_path, max_words)
    latin_words = list(dict.fromkeys(strip_diacritics(w.lower()) for w in words))
    neo_dict: dict[str, str] = {}
    for i, word in enumerate(latin_words):
        fitness = CompositeFitness(phonotactics, source_word=word)
        engine = EvolutionaryEngine(Organism(word, placeholder_vector), rules, rng=rng)
        results = engine.evolve(fitness, generations=generations, population_size=population_size)
        neo_dict[word] = results[0].word
        if i % 50 == 0:
            logger.info("Evolved %d/%d words", i, len(latin_words))

    if output_path is not None:
        output_path.write_text(json.dumps(neo_dict, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved %d entries to %s", len(neo_dict), output_path)
    return neo_dict


def _default_words(lexicon: pd.DataFrame, output_path: Path | None, max_words: int) -> list[str]:
    if output_path is not None and output_path.exists():
        return list(json.loads(output_path.read_text(encoding="utf-8")))[:max_words]
    return lexicon[LATIN_COL].drop_duplicates().head(max_words).tolist()
