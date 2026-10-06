"""Articulatory feature vectors for Romance sounds and distances between them."""

from .config import NATURALNESS_THRESHOLD

# Features: [place, manner, voicing, nasal, vowel_height, vowel_backness, is_vowel]
# Place: 0 labial, 1 dental/alveolar, 2 palatal, 3 velar, 4 glottal
# Manner: 0 stop, 1 fricative, 2 affricate, 3 nasal, 4 liquid/glide
# Vowel height: 0 high, 1 mid, 2 low.  Vowel backness: 0 front, 1 central, 2 back
PHONETIC_FEATURES: dict[str, list[int]] = {
    "p": [0, 0, 0, 0, 0, 0, 0],
    "b": [0, 0, 1, 0, 0, 0, 0],
    "t": [1, 0, 0, 0, 0, 0, 0],
    "d": [1, 0, 1, 0, 0, 0, 0],
    "k": [3, 0, 0, 0, 0, 0, 0],
    "c": [3, 0, 0, 0, 0, 0, 0],  # Latin spelling of /k/
    "q": [3, 0, 0, 0, 0, 0, 0],  # as in "qu" (/kw/)
    "g": [3, 0, 1, 0, 0, 0, 0],
    "f": [0, 1, 0, 0, 0, 0, 0],
    "v": [0, 1, 1, 0, 0, 0, 0],
    "s": [1, 1, 0, 0, 0, 0, 0],
    "x": [1, 1, 0, 0, 0, 0, 0],  # /ks/, simplified to its fricative part
    "z": [1, 1, 1, 0, 0, 0, 0],
    "m": [0, 0, 1, 1, 0, 0, 0],
    "n": [1, 0, 1, 1, 0, 0, 0],
    "l": [1, 4, 1, 0, 0, 0, 0],
    "r": [1, 4, 1, 0, 0, 0, 0],
    "a": [0, 0, 1, 0, 2, 1, 1],
    "e": [0, 0, 1, 0, 1, 0, 1],
    "i": [0, 0, 1, 0, 0, 0, 1],
    "o": [0, 0, 1, 0, 1, 2, 1],
    "u": [0, 0, 1, 0, 0, 2, 1],
    "h": [4, 1, 0, 0, 0, 0, 0],
    "j": [2, 4, 1, 0, 0, 0, 0],
}

UNKNOWN_SOUND_DISTANCE = 10.0


def get_feature_distance(char1: str, char2: str) -> float:
    """Manhattan distance between two sounds' feature vectors (10.0 if either is unknown)."""
    if char1 not in PHONETIC_FEATURES or char2 not in PHONETIC_FEATURES:
        return UNKNOWN_SOUND_DISTANCE
    return float(sum(abs(a - b) for a, b in zip(PHONETIC_FEATURES[char1], PHONETIC_FEATURES[char2], strict=True)))


def is_natural_transition(char1: str, char2: str) -> bool:
    """True when a sound change is small enough to be phonetically plausible."""
    return get_feature_distance(char1, char2) <= NATURALNESS_THRESHOLD
