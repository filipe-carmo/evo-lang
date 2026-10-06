"""Load and clean the Latin -> Romance etymology lexicon."""

import unicodedata
from pathlib import Path

import pandas as pd

from .config import LEXICON_PATH
from .genetics import PhoneticRule, RuleExtractor

LATIN_COL = "latin"
ROMANCE_COL = "romance"


def strip_diacritics(word: str) -> str:
    """Remove accents and Latin vowel-length marks (e.g. "aprīlis" -> "aprilis")."""
    decomposed = unicodedata.normalize("NFKD", word)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def load_lexicon(path: Path | str = LEXICON_PATH) -> pd.DataFrame:
    """
    Load the lexicon as clean, single-word, lowercase ASCII pairs.

    Wiktionary entries include affixes ("ad-"), phrases ("achar menos") and
    disambiguation junk ("a/languages m to z"); those are dropped.

    Verbs are also dropped: Wiktionary cites Latin verbs in the 1st person
    singular ("amō") but Romance verbs in the infinitive ("amar"), so comparing
    them would teach the GA a bogus "-o -> -ar" sound change.
    """
    df = pd.read_csv(path).dropna(subset=[LATIN_COL, ROMANCE_COL])
    for col in (LATIN_COL, ROMANCE_COL):
        df[col] = df[col].astype(str).str.lower().map(strip_diacritics)
    is_word = df[LATIN_COL].str.fullmatch(r"[a-z]+") & df[ROMANCE_COL].str.fullmatch(r"[a-z]+")
    is_verb_pair = df[LATIN_COL].str.endswith("o") & df[ROMANCE_COL].str.contains(r"[aei]re?$|ir$")
    return df[is_word & ~is_verb_pair].drop_duplicates().reset_index(drop=True)


def extract_rules(lexicon: pd.DataFrame, min_count: int = 2) -> list[PhoneticRule]:
    """Learn sound-change rules from every (Latin, Romance) pair in the lexicon."""
    extractor = RuleExtractor()
    for latin, romance in zip(lexicon[LATIN_COL], lexicon[ROMANCE_COL], strict=True):
        extractor.extract_rules(latin, romance)
    return extractor.get_rules(min_count=min_count)
