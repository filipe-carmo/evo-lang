from typing import Dict, List, Tuple

# Simple phonetic feature mapping for Romance languages
# Features: [place, manner, voicing, nasal, vowel_height, vowel_backness, is_vowel]
# Place: 0: labial, 1: dental/alveolar, 2: palatal, 3: velar, 4: glottal
# Manner: 0: stop, 1: fricative, 2: affricate, 3: nasal, 4: liquid/glide
# Vowel Height: 0: high, 1: mid, 2: low
# Vowel Backness: 0: front, 1: central, 2: back

PHONETIC_FEATURES: Dict[str, List[int]] = {
    'p': [0, 0, 0, 0, 0, 0, 0],
    'b': [0, 0, 1, 0, 0, 0, 0],
    't': [1, 0, 0, 0, 0, 0, 0],
    'd': [1, 0, 1, 0, 0, 0, 0],
    'k': [3, 0, 0, 0, 0, 0, 0],
    'g': [3, 0, 1, 0, 0, 0, 0],
    'f': [0, 1, 0, 0, 0, 0, 0],
    'v': [0, 1, 1, 0, 0, 0, 0],
    's': [1, 1, 0, 0, 0, 0, 0],
    'z': [1, 1, 1, 0, 0, 0, 0],
    'm': [0, 0, 1, 1, 0, 0, 0],
    'n': [1, 0, 1, 1, 0, 0, 0],
    'l': [1, 4, 1, 0, 0, 0, 0],
    'r': [1, 4, 1, 0, 0, 0, 0],
    'a': [0, 0, 1, 0, 2, 1, 1],
    'e': [0, 0, 1, 0, 1, 0, 1],
    'i': [0, 0, 1, 0, 0, 0, 1],
    'o': [0, 0, 1, 0, 1, 2, 1],
    'u': [0, 0, 1, 0, 0, 2, 1],
    # Simplified clusters/special chars
    'h': [4, 1, 0, 0, 0, 0, 0],
    'j': [2, 4, 1, 0, 0, 0, 0],
}

def get_feature_distance(char1: str, char2: str) -> float:
    """Calculates Manhattan distance between two phonetic feature vectors."""
    if char1 not in PHONETIC_FEATURES or char2 not in PHONETIC_FEATURES:
        return 10.0 # High penalty for unknown
    
    f1 = PHONETIC_FEATURES[char1]
    f2 = PHONETIC_FEATURES[char2]
    return sum(abs(a - b) for a, b in zip(f1, f2))

def is_natural_transition(char1: str, char2: str) -> bool:
    """Checks if a transition is phonetically 'natural' (small distance)."""
    dist = get_feature_distance(char1, char2)
    return dist <= 2.0 # Threshold for 'natural' change (e.g., voicing or minor place shift)
