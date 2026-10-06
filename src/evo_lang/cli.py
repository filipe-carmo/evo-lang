"""Command-line interface: ``evo-lang {evolve,translate,build-dict,rules,fetch}``."""

import argparse
import logging
import random

import numpy as np

from . import __version__
from .config import DEFAULT_GENERATIONS, DEFAULT_POPULATION_SIZE, LEXICON_PATH, SEMANTIC_DRIFT_STRENGTH
from .fitness import CompositeFitness, train_phonotactic_model
from .genetics import EvolutionaryEngine, Organism
from .lexicon import extract_rules, load_lexicon
from .semantics import SemanticDrift

LOREM_IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor"


def run_evolution(
    word: str,
    generations: int = DEFAULT_GENERATIONS,
    population_size: int = DEFAULT_POPULATION_SIZE,
    seed: int | None = None,
    top: int = 10,
) -> list[tuple[Organism, float]]:
    """Evolve one Latin word with semantic drift; return (organism, meaning similarity) pairs."""
    lexicon = load_lexicon(LEXICON_PATH)
    rules = extract_rules(lexicon, min_count=2)

    semantics = SemanticDrift(rng=np.random.default_rng(seed))
    reference = semantics.embed_concept(f"the concept of {word}")
    fitness = CompositeFitness(train_phonotactic_model(lexicon), reference_vector=reference, source_word=word)

    engine = EvolutionaryEngine(Organism(word, reference), rules, rng=random.Random(seed))
    results = engine.evolve(
        fitness,
        generations=generations,
        population_size=population_size,
        drift_fn=lambda vec: semantics.simulate_drift(vec, drift_strength=SEMANTIC_DRIFT_STRENGTH),
    )
    return [(org, fitness.semantic_similarity(org)) for org in results[:top]]


def _cmd_evolve(args: argparse.Namespace) -> None:
    print(f"Evolving '{args.word}' ({args.gens} generations, population {args.pop})\n")
    print(f"{'#':>2}  {'word':12} {'fitness':>7} {'natural':>7} {'meaning':>7}")
    results = run_evolution(args.word, args.gens, args.pop, seed=args.seed, top=args.top)
    for i, (org, sim) in enumerate(results, start=1):
        print(f"{i:>2}  {org.word:12} {org.fitness:7.1f} {org.get_average_naturalness():7.2f} {sim:7.2f}")


def _cmd_translate(args: argparse.Namespace) -> None:
    from .translator import NeoRomanceTranslator

    text = " ".join(args.text) or LOREM_IPSUM
    print(f"Latin:        {text}")
    print(f"Neo-Romance:  {NeoRomanceTranslator().translate_text(text)}")


def _cmd_build_dict(args: argparse.Namespace) -> None:
    from .dictionary import build_dictionary

    neo = build_dictionary(max_words=args.max_words, seed=args.seed)
    print(f"Saved {len(neo)} entries")


def _cmd_rules(args: argparse.Namespace) -> None:
    rules = extract_rules(load_lexicon(LEXICON_PATH), min_count=args.min_count)
    print(f"Top {args.top} of {len(rules)} sound-change rules (min count {args.min_count}):")
    for rule in rules[: args.top]:
        print(f"  {rule.source:>5} -> {rule.target:<5} count={rule.count:<4} naturalness={rule.nature_score:.2f}")


def _cmd_fetch(args: argparse.Namespace) -> None:
    from .fetch import fetch_lexicon

    fetch_lexicon(limit_per_language=args.limit)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evo-lang", description="Evolve Latin into Neo-Romance.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="show progress logs")
    sub = parser.add_subparsers(dest="command", required=True)

    evolve = sub.add_parser("evolve", help="evolve a single Latin word and show the top variants")
    evolve.add_argument("word", help="Latin word, e.g. nocte")
    evolve.add_argument("--gens", type=int, default=DEFAULT_GENERATIONS, help="number of generations")
    evolve.add_argument("--pop", type=int, default=DEFAULT_POPULATION_SIZE, help="population size")
    evolve.add_argument("--top", type=int, default=10, help="variants to show")
    evolve.add_argument("--seed", type=int, help="random seed for reproducible runs")
    evolve.set_defaults(func=_cmd_evolve)

    translate = sub.add_parser("translate", help="translate Latin text with the evolved dictionary")
    translate.add_argument("text", nargs="*", help="Latin text (default: Lorem ipsum)")
    translate.set_defaults(func=_cmd_translate)

    build = sub.add_parser("build-dict", help="re-evolve data/neo_romance_dict.json from the lexicon")
    build.add_argument("--max-words", type=int, default=500)
    build.add_argument("--seed", type=int, default=42)
    build.set_defaults(func=_cmd_build_dict)

    rules = sub.add_parser("rules", help="list the most frequent sound-change rules")
    rules.add_argument("--min-count", type=int, default=5)
    rules.add_argument("--top", type=int, default=20)
    rules.set_defaults(func=_cmd_rules)

    fetch = sub.add_parser("fetch", help="download Latin etymologies from Wiktionary into the lexicon")
    fetch.add_argument("--limit", type=int, default=200, help="pages per language")
    fetch.set_defaults(func=_cmd_fetch)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
