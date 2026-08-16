from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DEFAULT_SPECS, settings

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, RED, YELLOW, CYAN = "\033[32m", "\033[31m", "\033[33m", "\033[36m"


def _plain() -> bool:
    return not sys.stdout.isatty()


def c(text: str, colour: str) -> str:
    return text if _plain() else f"{colour}{text}{RESET}"


def _load_engine():
    from .answer.generator import AnswerEngine
    from .index.store import Index

    index = Index.load(settings.index_dir, settings.lsa_dims)
    return AnswerEngine(index, settings), index


def cmd_fetch(args) -> int:
    from .build import fetch_corpus

    specs = tuple(args.specs) if args.specs else DEFAULT_SPECS
    print(f"Fetching {len(specs)} specification(s) from 3gpp.org")
    manifest = fetch_corpus(specs, args.release)
    print(f"\n{len(manifest)} specification(s) ready in {settings.corpus_dir}")
    return 0 if manifest else 1


def cmd_build(args) -> int:
    from .build import build_index

    index = build_index()
    print(f"\n{c('Index ready', GREEN)}: {len(index)} chunks, backend {index.backend.name}")
    for spec in index.specs:
        print(f"  TS {spec['spec_id']:<8} v{spec['version']:<8} Rel-{spec['release']:<3} {spec['chunks']:5d} chunks")
    return 0


def cmd_ask(args) -> int:
    engine, _ = _load_engine()
    answer = engine.ask(args.question, top_k=args.k, entailment_check=args.entailment)

    if args.json:
        print(json.dumps(answer.as_dict(), indent=2, ensure_ascii=False))
        return 0

    print()
    if answer.abstained:
        print(c("ABSTAINED", YELLOW))
        print(f"  {answer.text}")
        print(f"  {c('reason:', DIM)} {answer.reason}")
    else:
        print(c(f"ANSWER  ({answer.mode}, confidence {answer.confidence:.2f}, groundedness {answer.groundedness:.0%})", GREEN))
        print()
        for line in answer.text.splitlines():
            print(f"  {line}")

    if answer.sources:
        print(f"\n{c('SOURCES', BOLD)}")
        for s in answer.sources:
            print(f"  [S{s.n}] {c(s.citation, CYAN)} | {s.heading}  {c(f'(score {s.score:.2f})', DIM)}")
            if args.verbose:
                print(f"        {s.breadcrumb}")

    dropped = [cl for cl in answer.claims if not cl.supported]
    if dropped:
        print(f"\n{c('REJECTED CLAIMS (failed grounding check)', RED)}")
        for cl in dropped:
            print(f"  - {cl.text[:110]}")
            print(f"    {c(cl.reason, DIM)}")

    if args.verbose:
        print(f"\n{c('DIAGNOSTICS', BOLD)}")
        for key, value in answer.diagnostics.items():
            print(f"  {key}: {value}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    print(f"Serving on http://{args.host}:{args.port}")
    uvicorn.run("telcorag.api:app", host=args.host, port=args.port, reload=args.reload, log_level="info")
    return 0


def cmd_eval(args) -> int:
    from .evaluate import run

    report = run(Path(args.golden) if args.golden else None, limit=args.limit, entailment=args.entailment)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written to {args.out}")
    return 0


def cmd_glossary(args) -> int:
    from .glossary import Glossary

    lex = Glossary.load(settings.index_dir / "glossary.json")
    if not len(lex):
        print("No glossary — build the index first.")
        return 1
    if not args.term:
        print(f"{len(lex)} acronyms indexed")
        return 0
    expansions = lex.lookup(args.term)
    if expansions:
        print(f"{args.term.upper()}: " + "; ".join(expansions))
    else:
        print(f"{args.term.upper()} not found")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="telcorag", description="Grounded question answering over 3GPP specifications")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download specifications from 3gpp.org")
    p.add_argument("--specs", nargs="*", help="spec numbers, e.g. 24.501 38.331")
    p.add_argument("--release", type=int, help="pin to a 3GPP Release, e.g. 18")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build", help="parse the corpus and build the index")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("-k", type=int, help="passages to retrieve")
    p.add_argument("--json", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--entailment", action="store_true", help="add the LLM entailment pass")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("serve", help="run the web UI and API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("eval", help="run the evaluation harness")
    p.add_argument("--golden", help="path to a golden set YAML/JSON")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", help="write the JSON report here")
    p.add_argument("--entailment", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("glossary", help="look up a 3GPP acronym")
    p.add_argument("term", nargs="?")
    p.set_defaults(func=cmd_glossary)

    return parser


def _use_utf8() -> None:
    """Spec text is full of non-ASCII punctuation; a cp1252 console mangles it."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _use_utf8()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(c(f"error: {exc}", RED), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
