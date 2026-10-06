import pandas as pd
import pytest

from evo_lang.fitness import train_phonotactic_model
from evo_lang.lexicon import extract_rules, load_lexicon


@pytest.fixture(scope="session")
def lexicon() -> pd.DataFrame:
    return load_lexicon()


@pytest.fixture(scope="session")
def rules(lexicon):
    return extract_rules(lexicon, min_count=2)


@pytest.fixture(scope="session")
def phonotactics(lexicon):
    return train_phonotactic_model(lexicon)
