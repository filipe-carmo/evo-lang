from typing import List, Tuple, Dict
import re

class GrammarEngine:
    """Simulates high-level grammatical shifts from Latin to Romance."""
    
    def __init__(self):
        # List of tuples to preserve application order (priority)
        self.suffixes: List[Tuple[str, str]] = [
            (r'orum$', 'o'),   # Genitive plural -> o/os
            (r'arum$', 'a'),   # Genitive plural -> a/as
            (r'ibus$', 'es'),  # Dative/Ablative plural
            (r'is$', 'e'),     # Genitive singular
            (r'em$', 'e'),     # Accusative (loss of final m)
            (r'um$', 'o'),     # Accusative (loss of final m)
            (r'am$', 'a'),     # Accusative (loss of final m)
            (r'at$', 'a'),     # Verb 3rd person singular
            (r'ant$', 'an'),   # Verb 3rd person plural
        ]
        
        # Simple articles based on Latin 'ille/illa' (that) and 'unus' (one)
        self.particles: Dict[str, str] = {
            'ille': 'lo',
            'illa': 'la',
            'illud': 'lo',
            'unus': 'on',
            'una': 'ona'
        }

    def simplify_morphology(self, word: str) -> str:
        """Simplifies Latin word endings."""
        for pattern, replacement in self.suffixes:
            if re.search(pattern, word):
                return re.sub(pattern, replacement, word)
        return word

    def apply_grammar(self, text: str) -> str:
        """Applies grammatical transformations to a sentence."""
        words = text.lower().split()
        new_words = []
        
        for i, word in enumerate(words):
            # 1. Substitute demonstratives with articles
            if word in self.particles:
                new_words.append(self.particles[word])
                continue
            
            # 2. Heuristic: Add articles before nouns (very simple logic)
            # If a word is long and doesn't have an article, maybe add one?
            # For now, let's keep it simple.
            
            # 3. Simplify endings
            word = self.simplify_morphology(word)
            new_words.append(word)
            
        return " ".join(new_words)
