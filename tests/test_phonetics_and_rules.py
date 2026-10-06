import pandas as pd
import pytest

from evo_lang.genetics import PhoneticRule, RuleExtractor
from evo_lang.lexicon import load_lexicon, strip_diacritics
from evo_lang.phonetics import UNKNOWN_SOUND_DISTANCE, get_feature_distance, is_natural_transition


def test_voicing_is_a_natural_sound_change():
    assert get_feature_distance("t", "d") == 1
    assert is_natural_transition("t", "d")
    assert is_natural_transition("p", "b")


def test_distant_sounds_are_not_natural():
    assert not is_natural_transition("p", "a")
    assert get_feature_distance("t", "?") == UNKNOWN_SOUND_DISTANCE


def test_rule_extractor_counts_edits():
    extractor = RuleExtractor()
    for latin, romance in [("rota", "roda"), ("vita", "vida"), ("lupus", "lobo")]:
        extractor.extract_rules(latin, romance)
    rules = {(r.source, r.target): r for r in extractor.get_rules(min_count=2)}
    assert ("t", "d") in rules
    assert rules[("t", "d")].count == 2
    assert sum(r.probability for r in rules.values()) == pytest.approx(1.0)


def test_rule_extractor_ignores_pure_insertions():
    extractor = RuleExtractor()
    for _ in range(3):
        extractor.extract_rules("ab", "xab")
    assert extractor.get_rules(min_count=1) == []


def test_natural_rules_score_higher_than_deletions():
    assert PhoneticRule("t", "d").nature_score > PhoneticRule("t", "").nature_score


def test_rule_applies_to_first_occurrence_only():
    assert PhoneticRule("t", "d").apply("tota") == "dota"
    assert PhoneticRule("x", "y").apply("tota") == "tota"


def test_strip_diacritics_removes_vowel_length():
    assert strip_diacritics("aprīlis") == "aprilis"
    assert strip_diacritics("abóbada") == "abobada"


def test_load_lexicon_drops_affixes_phrases_and_verbs(tmp_path):
    csv = tmp_path / "lexicon.csv"
    pd.DataFrame(
        {
            "latin": ["lupus", "ad-", "afflo", "amō", "aprīlis"],
            "romance": ["lobo", "a-", "achar menos", "amar", "abril"],
            "lang": ["pt"] * 5,
        }
    ).to_csv(csv, index=False)
    lexicon = load_lexicon(csv)
    assert list(zip(lexicon["latin"], lexicon["romance"], strict=True)) == [
        ("lupus", "lobo"),
        ("aprilis", "abril"),
    ]


def test_bundled_lexicon_is_clean(lexicon):
    assert len(lexicon) > 200
    assert lexicon["latin"].str.fullmatch(r"[a-z]+").all()
    assert lexicon["romance"].str.fullmatch(r"[a-z]+").all()
