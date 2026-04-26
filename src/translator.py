import json
import re
import pandas as pd
from src.genetics import RuleExtractor, PhoneticRule
from src.grammar import GrammarEngine
from src.config import LEXICON_PATH, NEO_DICT_PATH

class NeoRomanceTranslator:
    def __init__(self, dict_path=NEO_DICT_PATH, lexicon_path=LEXICON_PATH):
        # 1. Load the pre-computed dictionary
        if dict_path.exists():
            with dict_path.open('r', encoding='utf-8') as f:
                self.dictionary = json.load(f)
        else:
            self.dictionary = {}
            print(f"Warning: Neo-Romance dictionary not found at {dict_path}.")

        # 2. Grammar Engine
        self.grammar = GrammarEngine()

        # 3. Extract general rules for words not in the dictionary
        self.general_rules = []
        if lexicon_path.exists():
            df = pd.read_csv(lexicon_path)
            df = df.dropna(subset=['latin', 'portuguese'])
            extractor = RuleExtractor()
            for _, row in df.iterrows():
                extractor.extract_rules(str(row['latin']), str(row['portuguese']))
            # Top 15 most common/reliable rules
            self.general_rules = extractor.get_rules(min_count=4)[:15]

    def translate_word(self, word: str) -> str:
        word = word.lower().strip()
        # Remove punctuation
        clean_word = re.sub(r'[^\w\s]', '', word)
        
        # 1. Simplify morphology (e.g., -um -> -o)
        clean_word = self.grammar.simplify_morphology(clean_word)
        
        # 2. Direct Lookup
        if clean_word in self.dictionary:
            return self.dictionary[clean_word]
        
        # 3. Heuristic Transformation for unknown words
        transformed = clean_word
        for rule in self.general_rules:
            if rule.nature_score > 0.6: # Relaxed slightly for more drift
                transformed = rule.apply(transformed)
        
        # Ensure it ends with a romance-like vowel if it ends in a harsh consonant
        if transformed and transformed[-1] not in 'aeioumsh':
            transformed += 'e'
            
        return transformed

    def translate_text(self, text: str) -> str:
        # Pre-process text with grammar rules
        text = self.grammar.apply_grammar(text)
        
        words = text.split()
        translated = [self.translate_word(w) for w in words]
        return " ".join(translated)

if __name__ == "__main__":
    translator = NeoRomanceTranslator()
    sample_text = "luna nocte stella in terra"
    print(f"Original: {sample_text}")
    print(f"Neo-Romance: {translator.translate_text(sample_text)}")
