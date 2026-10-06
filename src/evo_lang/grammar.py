"""Coarse grammatical changes from Latin to Romance: case-ending loss and articles."""

import re

# Ordered by priority: the first matching ending wins.
SUFFIX_RULES: list[tuple[str, str]] = [
    (r"orum$", "o"),  # genitive plural
    (r"arum$", "a"),  # genitive plural
    (r"ibus$", "es"),  # dative/ablative plural
    (r"is$", "e"),  # genitive singular
    (r"em$", "e"),  # accusative (loss of final m)
    (r"um$", "o"),  # accusative (loss of final m)
    (r"am$", "a"),  # accusative (loss of final m)
    (r"at$", "a"),  # verb, 3rd person singular
    (r"ant$", "an"),  # verb, 3rd person plural
]

# Latin demonstratives "ille/illa" and the numeral "unus" became Romance articles.
ARTICLES: dict[str, str] = {
    "ille": "lo",
    "illa": "la",
    "illud": "lo",
    "unus": "on",
    "una": "ona",
}


class GrammarEngine:
    def __init__(self) -> None:
        self.suffixes = [(re.compile(p), r) for p, r in SUFFIX_RULES]
        self.particles = dict(ARTICLES)

    def simplify_morphology(self, word: str) -> str:
        """Replace the first matching Latin case/verb ending."""
        for pattern, replacement in self.suffixes:
            if pattern.search(word):
                return pattern.sub(replacement, word)
        return word

    def apply_grammar(self, text: str) -> str:
        """Turn demonstratives into articles and simplify every other word's ending."""
        return " ".join(self.particles.get(word) or self.simplify_morphology(word) for word in text.lower().split())
