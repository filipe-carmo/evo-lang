import json

from evo_lang.cli import main
from evo_lang.config import LEXICON_PATH
from evo_lang.translator import NeoRomanceTranslator


def test_translator_uses_dictionary_before_grammar(tmp_path):
    dict_path = tmp_path / "dict.json"
    dict_path.write_text(json.dumps({"factum": "fado"}))
    translator = NeoRomanceTranslator(dict_path=dict_path, lexicon_path=LEXICON_PATH)
    assert translator.translate_word("Factum,") == "fado"


def test_translator_turns_demonstratives_into_articles(tmp_path):
    translator = NeoRomanceTranslator(dict_path=tmp_path / "missing.json", lexicon_path=LEXICON_PATH)
    assert translator.translate_text("ille").split() == ["lo"]


def test_unknown_words_never_vanish(tmp_path):
    translator = NeoRomanceTranslator(dict_path=tmp_path / "missing.json", lexicon_path=LEXICON_PATH)
    for word in ["sed", "sit", "amet", "consectetur"]:
        assert translator.translate_word(word)


def test_cli_evolve_prints_ranked_variants(capsys):
    main(["evolve", "nocte", "--gens", "5", "--pop", "20", "--top", "3", "--seed", "1"])
    out = capsys.readouterr().out
    assert "Evolving 'nocte'" in out
    assert " 1  " in out


def test_cli_rules_and_translate(capsys):
    main(["rules", "--top", "3"])
    main(["translate", "luna", "in", "terra"])
    out = capsys.readouterr().out
    assert "us ->" in out
    assert "Neo-Romance:" in out
