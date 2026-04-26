import pandas as pd
import sys
import os

# Add src to the path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from genetics import RuleExtractor

def audit():
    df = pd.read_csv('data/lexicon.csv')
    df = df.dropna(subset=['latin', 'portuguese'])
    extractor = RuleExtractor()
    for _, row in df.iterrows():
        extractor.extract_rules(str(row['latin']), str(row['portuguese']))
    
    rules = extractor.get_rules(min_count=5)
    print("Top 20 Rules Audit:")
    for r in rules[:20]:
        print(r)

if __name__ == "__main__":
    audit()
