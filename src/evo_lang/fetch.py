"""Build the etymology lexicon from Wiktionary's "terms inherited from Latin" categories."""

import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

from .config import LEXICON_PATH
from .lexicon import LATIN_COL, ROMANCE_COL

logger = logging.getLogger(__name__)

API_URL = "https://en.wiktionary.org/w/api.php"
HEADERS = {"User-Agent": "evo-lang/0.3 (https://github.com/filipe-carmo/evo-lang)"}
CATEGORIES = {
    "pt": "Category:Portuguese_terms_inherited_from_Latin",
    "es": "Category:Spanish_terms_inherited_from_Latin",
    "fr": "Category:French_terms_inherited_from_Latin",
    "it": "Category:Italian_terms_inherited_from_Latin",
}
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 0.05


def _latin_root(wikitext: str, lang_code: str) -> str | None:
    """Find the Latin etymon in templates like {{inh|pt|la|noctem}}."""
    pattern = r"\{\{(?:inh|der|bor)\|" + re.escape(lang_code) + r"\|la\|([^|}]+)"
    match = re.search(pattern, wikitext)
    if not match:
        return None
    root = re.sub(r"^[*-]+", "", match.group(1).lower())
    return None if "{" in root else root


def fetch_category(session: requests.Session, category: str, lang_code: str, limit: int = 200) -> list[dict]:
    """Return up to ``limit`` {latin, romance, lang} pairs from one Wiktionary category."""
    params = {"action": "query", "list": "categorymembers", "cmtitle": category, "cmlimit": 500, "format": "json"}
    try:
        res = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        members = res.json()["query"]["categorymembers"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        logger.error("Could not list %s: %s", category, exc)
        return []

    titles = [m["title"] for m in members if not m["title"].startswith("Category:")][:limit]
    pairs = []
    for i, title in enumerate(titles):
        params = {"action": "query", "prop": "revisions", "titles": title, "rvprop": "content", "format": "json"}
        try:
            res = session.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            page = next(iter(res.json()["query"]["pages"].values()))
            wikitext = page["revisions"][0]["*"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            logger.debug("Skipping %s: %s", title, exc)
            continue
        root = _latin_root(wikitext, lang_code)
        if root:
            pairs.append({LATIN_COL: root, ROMANCE_COL: title.lower(), "lang": lang_code})
        if i % 50 == 0:
            logger.info("%s: processed %d/%d pages", lang_code, i, len(titles))
        time.sleep(REQUEST_DELAY)
    return pairs


def fetch_lexicon(output_path: Path = LEXICON_PATH, limit_per_language: int = 200) -> pd.DataFrame:
    """Fetch all languages and merge them into the lexicon CSV."""
    rows = []
    with requests.Session() as session:
        session.headers.update(HEADERS)
        for code, category in CATEGORIES.items():
            pairs = fetch_category(session, category, code, limit=limit_per_language)
            logger.info("Collected %d pairs for %s", len(pairs), code)
            rows.extend(pairs)

    lexicon = pd.DataFrame(rows, columns=[LATIN_COL, ROMANCE_COL, "lang"])
    if output_path.exists():
        lexicon = pd.concat([pd.read_csv(output_path), lexicon], ignore_index=True)
    lexicon = lexicon.drop_duplicates(subset=[LATIN_COL, ROMANCE_COL, "lang"])
    lexicon.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Lexicon now has %d entries", len(lexicon))
    return lexicon
