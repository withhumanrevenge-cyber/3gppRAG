"""Acronym lexicon harvested from the corpus itself.

Telecom queries and telecom prose rarely use the same surface form: a user asks
about the "Session Management Function", the spec says "SMF"; a user asks about
"5GMM", the spec headings say "5GS mobility management". Pure lexical retrieval
misses on that gap and dense retrieval only partly covers it.

Rather than hand-maintaining a mapping, we mine it: every 3GPP spec carries an
"Abbreviations" clause written as ``ACRONYM<tab>Expansion``, and TR 21.905 is the
project-wide vocabulary. That yields a few thousand authoritative pairs that
stay correct as the corpus is re-fetched.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .corpus.parser import Clause

ABBREV_HEADING_RE = re.compile(r"\babbreviations\b", re.I)
ENTRY_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9\-_/().]{0,24})\t+(.{2,120})$")
ACRONYM_RE = re.compile(r"^(?=.*[A-Z0-9])[A-Za-z0-9][A-Za-z0-9\-/.]{1,20}$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# Expansions too generic to be useful as retrieval hints.
STOP_EXPANSIONS = frozenset({"see", "void", "reserved", "not applicable", "for future use"})
MAX_PHRASE_WORDS = 7


@dataclass
class Glossary:
    expansions: dict[str, list[str]]
    reverse: dict[str, str]

    @classmethod
    def empty(cls) -> "Glossary":
        return cls({}, {})

    def __len__(self) -> int:
        return len(self.expansions)

    def expand(self, query: str) -> list[str]:
        """Return extra retrieval terms implied by acronyms/phrases in ``query``."""
        extra: list[str] = []
        seen: set[str] = set()

        for token in WORD_RE.findall(query):
            for exp in self.expansions.get(token.upper(), [])[:2]:
                if exp.lower() not in seen:
                    seen.add(exp.lower())
                    extra.append(exp)

        words = WORD_RE.findall(query.lower())
        for size in range(min(MAX_PHRASE_WORDS, len(words)), 1, -1):
            for i in range(len(words) - size + 1):
                phrase = " ".join(words[i : i + size])
                acronym = self.reverse.get(phrase)
                if acronym and acronym.lower() not in seen:
                    seen.add(acronym.lower())
                    extra.append(acronym)
        return extra

    def lookup(self, term: str) -> list[str]:
        return self.expansions.get(term.upper(), [])

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"expansions": self.expansions, "reverse": self.reverse}, indent=1, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Glossary":
        if not path.exists():
            return cls.empty()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(raw.get("expansions", {}), raw.get("reverse", {}))


def _entries(clause: Clause) -> Iterable[tuple[str, str]]:
    for line in clause.paragraphs:
        m = ENTRY_RE.match(line.strip())
        if not m:
            continue
        acronym, expansion = m.group(1).strip(), m.group(2).strip()
        expansion = re.sub(r"\s+", " ", expansion).strip(" .;")
        if not ACRONYM_RE.match(acronym):
            continue
        if not expansion or expansion.lower() in STOP_EXPANSIONS:
            continue
        if len(WORD_RE.findall(expansion)) > MAX_PHRASE_WORDS:
            continue
        if expansion.lower() == acronym.lower():
            continue
        yield acronym.upper(), expansion


def build(clauses: Iterable[Clause]) -> Glossary:
    expansions: dict[str, list[str]] = defaultdict(list)
    for clause in clauses:
        if not ABBREV_HEADING_RE.search(clause.heading):
            continue
        for acronym, expansion in _entries(clause):
            bucket = expansions[acronym]
            if not any(e.lower() == expansion.lower() for e in bucket):
                bucket.append(expansion)

    reverse: dict[str, str] = {}
    for acronym, forms in expansions.items():
        for form in forms:
            key = " ".join(WORD_RE.findall(form.lower()))
            if len(key.split()) < 2:
                continue
            reverse.setdefault(key, acronym)

    return Glossary({k: v for k, v in sorted(expansions.items())}, reverse)
