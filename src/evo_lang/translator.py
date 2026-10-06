"""Translate Latin text using the evolved dictionary, with rule-based fallback for unknown words."""

import json
import logging
import re
from pathlib import Path

from .config import LEXICON_PATH, NEO_DICT_PATH
from .genetics import PhoneticRule
from .grammar import GrammarEngine
from .lexicon import extract_rules, load_lexicon

logger = logging.getLogger(__name__)

FALLBACK_RULE_COUNT = 15
FALLBACK_MIN_NATURALNESS = 0.6
ROMANCE_FINAL_LETTERS = "aeioumsh"


def _is_safe_fallback(rule: PhoneticRule) -> bool:
    """Plausible substitutions only: chained deletions or lengthenings would mangle unknown words."""
    return bool(rule.target) and len(rule.target) <= len(rule.source) and rule.nature_score > FALLBACK_MIN_NATURALNESS


class NeoRomanceTranslator:
    def __init__(self, dict_path: Path = NEO_DICT_PATH, lexicon_path: Path = LEXICON_PATH):
        self.dictionary: dict[str, str] = {}
        if dict_path.exists():
            self.dictionary = json.loads(dict_path.read_text(encoding="utf-8"))
        else:
            logger.warning("Neo-Romance dictionary not found at %s", dict_path)

        self.grammar = GrammarEngine()

        self.fallback_rules: list[PhoneticRule] = []
        if lexicon_path.exists():
            rules = extract_rules(load_lexicon(lexicon_path), min_count=4)
            self.fallback_rules = [r for r in rules if _is_safe_fallback(r)][:FALLBACK_RULE_COUNT]

    def translate_word(self, word: str) -> str:
        clean_word = re.sub(r"[^\w]", "", word.lower())
        if clean_word in self.dictionary:
            return self.dictionary[clean_word]

        clean_word = self.grammar.simplify_morphology(clean_word)
        if clean_word in self.dictionary:
            return self.dictionary[clean_word]

        transformed = clean_word
        for rule in self.fallback_rules:
            transformed = rule.apply(transformed)
        # Romance words rarely end in a hard consonant: add a supporting vowel
        if transformed and transformed[-1] not in ROMANCE_FINAL_LETTERS:
            transformed += "e"
        return transformed

    def translate_text(self, text: str) -> str:
        words = text.lower().split()
        return " ".join(self.grammar.particles.get(re.sub(r"[^\w]", "", w)) or self.translate_word(w) for w in words)
