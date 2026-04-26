import requests
import re
import pandas as pd
import os
import time

HEADERS = {
    "User-Agent": "NeoRomanceBot/1.0 (https://github.com/your-username/evo-lang; your-email@example.com)"
}

def fetch_wiktionary_category(category_name, lang_code, limit=300):
    """Fetches word pairs for a specific romance language."""
    url = "https://en.wiktionary.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category_name,
        "cmlimit": 500,
        "format": "json"
    }
    
    print(f"Fetching {lang_code} from {category_name}...")
    try:
        res = requests.get(url, params=params, headers=HEADERS).json()
        members = [m["title"] for m in res["query"]["categorymembers"] if not m["title"].startswith("Category:")]
    except:
        return []

    data = []
    for i, word in enumerate(members):
        if i >= limit: break
        
        params = {"action": "query", "prop": "revisions", "titles": word, "rvprop": "content", "format": "json"}
        try:
            res = requests.get(url, params=params, headers=HEADERS).json()
            pages = res["query"]["pages"]
            pid = next(iter(pages))
            content = pages[pid]["revisions"][0]["*"]
            
            # Use raw string and careful escaping for f-string
            # Goal: {{ (inh|der|bor) | lang_code | la | (root) }}
            pattern = r"\{\{(?:inh|der|bor)\|" + re.escape(lang_code) + r"\|la\|([^|}}]+)"
            match = re.search(pattern, content)
            if match:
                root = match.group(1).lower()
                root = re.sub(r"^[*-]+", "", root)
                if "{" not in root:
                    data.append({"latin": root, "portuguese": word.lower(), "lang": lang_code})
            
            if i % 50 == 0:
                print(f"  {lang_code}: Processed {i} words...")
            time.sleep(0.05)
        except:
            continue
    return data

def main():
    languages = {
        "pt": "Category:Portuguese_terms_inherited_from_Latin",
        "es": "Category:Spanish_terms_inherited_from_Latin",
        "fr": "Category:French_terms_inherited_from_Latin",
        "it": "Category:Italian_terms_inherited_from_Latin"
    }
    
    all_data = []
    for code, cat in languages.items():
        lang_data = fetch_wiktionary_category(cat, code, limit=200)
        all_data.extend(lang_data)
        print(f"Collected {len(lang_data)} pairs for {code}")

    if all_data:
        df = pd.DataFrame(all_data)
        lex_path = 'data/lexicon.csv'
        
        if os.path.exists(lex_path):
            existing = pd.read_csv(lex_path)
            if 'lang' not in existing.columns: existing['lang'] = 'pt'
            final = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=['latin', 'portuguese', 'lang'])
        else:
            final = df
            
        final.to_csv(lex_path, index=False, encoding='utf-8')
        print(f"Success! Lexicon now has {len(final)} entries across multiple languages.")

if __name__ == "__main__":
    main()
