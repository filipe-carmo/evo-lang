import random

import numpy as np

from evo_lang.fitness import CompositeFitness, PhonotacticFitness
from evo_lang.genetics import EvolutionaryEngine, Organism, PhoneticRule
from evo_lang.grammar import GrammarEngine
from evo_lang.semantics import SemanticDrift

VECTOR = np.ones(4, dtype=np.float32) / 2


def test_phonotactics_prefers_romance_like_words(phonotactics):
    assert phonotactics.score("noite") > phonotactics.score("xqzkt")


def test_untrained_model_scores_zero():
    assert PhonotacticFitness().score("casa") == 0.0


def test_composite_fitness_penalises_eroded_words(phonotactics):
    fitness = CompositeFitness(phonotactics, source_word="novus")
    assert fitness(Organism("novo", VECTOR)) > fitness(Organism("o", VECTOR))
    assert fitness(Organism("", VECTOR)) == 0.0


def test_crossover_splices_halves_and_dedupes_history():
    rule = PhoneticRule("t", "d")
    child = EvolutionaryEngine.crossover(Organism("abcd", VECTOR, [rule]), Organism("wxyz", VECTOR * 3, [rule]))
    assert child.word == "abyz"
    assert child.history == [rule]
    np.testing.assert_allclose(child.meaning_vector, VECTOR * 2)


def _run(seed, rules, phonotactics):
    engine = EvolutionaryEngine(Organism("nocte", VECTOR), rules, rng=random.Random(seed))
    fitness = CompositeFitness(phonotactics, source_word="nocte")
    return engine.evolve(fitness, generations=8, population_size=30)


def test_evolution_is_reproducible_and_sorted(rules, phonotactics):
    first = _run(7, rules, phonotactics)
    second = _run(7, rules, phonotactics)
    assert [o.word for o in first] == [o.word for o in second]
    fitnesses = [o.fitness for o in first]
    assert fitnesses == sorted(fitnesses, reverse=True)
    assert len({o.word for o in first}) == len(first)


def test_evolution_actually_changes_words(rules, phonotactics):
    words = {o.word for o in _run(1, rules, phonotactics)}
    assert words - {"nocte"}


def test_semantic_drift_keeps_unit_vectors():
    drift = SemanticDrift(rng=np.random.default_rng(0))
    drifted = drift.simulate_drift(VECTOR, drift_strength=0.1)
    assert np.linalg.norm(drifted) == np.float32(1.0) or abs(np.linalg.norm(drifted) - 1) < 1e-6
    assert not np.allclose(drifted, VECTOR)


def test_grammar_drops_case_endings_and_builds_articles():
    grammar = GrammarEngine()
    assert grammar.simplify_morphology("murum") == "muro"
    assert grammar.simplify_morphology("rosarum") == "rosa"
    assert grammar.apply_grammar("Illa rosam") == "la rosa"
