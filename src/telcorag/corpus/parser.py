"""Turn a spec's Word parts into a tree of numbered clauses.

3GPP clause numbering is the backbone of the whole retrieval design: a clause is
the unit a telecom engineer cites ("see 5.4.1.3.2"), it is self-contained by
drafting convention, and its ancestry carries the meaning of its title
("General" is meaningless without "4.4.2 Handling of 5G NAS security contexts").
We therefore keep clauses intact rather than sliding a fixed window over the
text, and attach the full ancestor chain to every clause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .ooxml import Block, Paragraph, Table, read_docx

# Matches "4", "4.2A", "4.5.2A.1", "6.2.4.2a", "A", "A.2.1".
# Annex clauses start with a letter; revision suffixes appear in both cases.
CLAUSE_RE = re.compile(r"^((?:\d+|[A-Z])[A-Za-z]?(?:\.\d+[A-Za-z]?)*)$")
ANNEX_RE = re.compile(r"^Annex\s+([A-Z][A-Z0-9]*)\s*(?:\(([^)]*)\))?\s*:?\s*(.*)$", re.I)

BULLET_STYLES = frozenset({"B1", "B2", "B3", "B4", "B5"})
NOTE_STYLES = frozenset({"NO", "EditorsNote"})
DROP_STYLES_PREFIX = ("TOC", "TAN")
DROP_HEADINGS = frozenset({"foreword", "contents", "change history", "document history"})


@dataclass
class Clause:
    spec_id: str
    version: str
    release: int
    number: str
    heading: str
    level: int
    ancestors: list[tuple[str, str]] = field(default_factory=list)
    paragraphs: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    normative: bool = True
    source: str = ""

    @property
    def label(self) -> str:
        return f"{self.number} {self.heading}".strip()

    @property
    def citation(self) -> str:
        return f"3GPP TS {self.spec_id} v{self.version}, clause {self.number or self.heading}"

    @property
    def breadcrumb(self) -> str:
        trail = [f"{n} {t}".strip() for n, t in self.ancestors]
        return " > ".join(trail + [self.label])

    @property
    def body(self) -> str:
        return "\n\n".join([*self.paragraphs, *self.tables]).strip()

    def is_empty(self) -> bool:
        return not self.paragraphs and not self.tables


def _split_heading(text: str) -> tuple[str, str]:
    parts = [p.strip() for p in text.split("\t") if p.strip()]
    if not parts:
        return "", ""
    head = parts[0]
    rest = " ".join(parts[1:]).strip()

    annex = ANNEX_RE.match(text.replace("\t", " ").strip())
    if annex and not CLAUSE_RE.match(head):
        letter, _kind, title = annex.groups()
        return letter, (title or "").strip()

    if CLAUSE_RE.match(head):
        return head, rest
    return "", " ".join(parts).strip()


def _render(block: Paragraph) -> str | None:
    """Normalise a paragraph, keeping tabs.

    Tabs are load-bearing in 3GPP drafting: abbreviation clauses are written as
    ``5GMM<tab>5GS Mobility Management`` and reference clauses as
    ``[24]<tab>3GPP TS 33.501: "..."``. Collapsing them to spaces would destroy
    the only machine-readable delimiter those lists have.
    """
    style = block.style
    if style.startswith(DROP_STYLES_PREFIX):
        return None
    text = re.sub(r"\t{2,}", "\t", re.sub(r"[ ]{2,}", " ", block.text)).strip()
    if not text:
        return None
    if style in BULLET_STYLES:
        depth = int(style[1]) if style[1:].isdigit() else 1
        return "  " * (depth - 1) + "- " + text.lstrip("-\t ").strip()
    return text


def parse_spec(
    paths: list[Path],
    spec_id: str,
    version: str,
    release: int,
) -> list[Clause]:
    clauses: list[Clause] = []
    stack: list[Clause] = []
    current: Clause | None = None
    in_annex = False
    annex_normative = True

    def push(clause: Clause) -> None:
        nonlocal current
        if current is not None and not current.is_empty():
            clauses.append(current)
        current = clause

    for path in paths:
        if path.suffix.lower() != ".docx":
            continue
        for block in read_docx(path):
            if isinstance(block, Table):
                if current is not None:
                    current.tables.append(block.to_markdown())
                continue

            level = block.heading_level
            if level is None:
                if current is None:
                    continue
                rendered = _render(block)
                if rendered:
                    current.paragraphs.append(rendered)
                continue

            raw = block.text.strip()
            annex = ANNEX_RE.match(raw.replace("\t", " "))
            if annex:
                in_annex = True
                annex_normative = "informative" not in (annex.group(2) or "").lower()

            number, heading = _split_heading(raw)
            if heading.lower() in DROP_HEADINGS or number.lower() in DROP_HEADINGS:
                push(Clause(spec_id, version, release, "", "__drop__", level, source=path.name))
                stack = stack[: level - 1]
                continue

            stack = stack[: level - 1]
            ancestors = [(c.number, c.heading) for c in stack if c.heading != "__drop__"]
            clause = Clause(
                spec_id=spec_id,
                version=version,
                release=release,
                number=number,
                heading=heading,
                level=level,
                ancestors=ancestors,
                normative=annex_normative if in_annex else True,
                source=path.name,
            )
            stack.append(clause)
            push(clause)

    if current is not None and not current.is_empty():
        clauses.append(current)
    return [c for c in clauses if c.heading != "__drop__"]


def load_spec(paths: list[Path], spec_id: str, version: str, release: int) -> list[Clause]:
    return parse_spec(paths, spec_id, version, release)
