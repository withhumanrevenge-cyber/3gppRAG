"""Clause-anchored chunking.

The chunk boundary is the clause boundary, not a fixed token window. A 3GPP
clause is already the smallest self-contained normative unit -- it is what the
specs cross-reference and what an engineer cites -- so cutting across one
reliably severs a condition from its consequence ("the UE shall ... if ...").
Only clauses that exceed ``max_tokens`` are split, and then on paragraph
boundaries with overlap, never mid-table.

Every chunk is prefixed with a provenance header carrying spec, version,
release, clause number and full ancestor path. This is indexed and embedded
along with the body, which is what lets a bare heading like "General" be
retrieved for "5G NAS security context handling", and it is the same string the
generator is later required to cite.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Iterable, Iterator

from ..config import Chunking
from .parser import Clause

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-.'][A-Za-z0-9]+)*")
SENT_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(\[])")


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))


@dataclass
class Chunk:
    id: str
    spec_id: str
    version: str
    release: int
    clause: str
    heading: str
    breadcrumb: str
    level: int
    normative: bool
    text: str
    part: int = 0
    parts: int = 1
    prev_id: str | None = None
    next_id: str | None = None
    source: str = ""
    tokens: int = 0

    @property
    def header(self) -> str:
        tag = f"3GPP TS {self.spec_id} v{self.version} (Rel-{self.release})"
        part = f" [part {self.part + 1}/{self.parts}]" if self.parts > 1 else ""
        return f"{tag} — clause {self.clause} {self.heading}{part}\nPath: {self.breadcrumb}"

    @property
    def indexed_text(self) -> str:
        return f"{self.header}\n\n{self.text}"

    @property
    def citation(self) -> str:
        where = self.clause or self.heading
        return f"3GPP TS {self.spec_id} v{self.version}, clause {where}"

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Chunk":
        return cls(**raw)


def _blocks_of(clause: Clause) -> list[str]:
    return [b for b in [*clause.paragraphs, *clause.tables] if b.strip()]


def _split_long_paragraph(text: str, limit: int) -> list[str]:
    sentences = SENT_SPLIT_RE.split(text)
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for sent in sentences:
        n = count_tokens(sent)
        if buf and size + n > limit:
            out.append(" ".join(buf))
            buf, size = [], 0
        buf.append(sent)
        size += n
    if buf:
        out.append(" ".join(buf))
    return out or [text]


def _split_table(markdown: str, limit: int) -> list[str]:
    """Split an oversized markdown table on row boundaries, repeating the header.

    Some 3GPP tables (IE type lists, cause-code tables) run to thousands of
    tokens. Emitting one as a single chunk both blows the generator's context
    budget and averages its embedding into meaninglessness, so it is banded --
    but every band keeps the header row, without which the cells are unreadable.
    """
    lines = [ln for ln in markdown.split("\n") if ln.strip()]
    if len(lines) < 3:
        return [markdown]

    header, separator, rows = lines[0], lines[1], lines[2:]
    base = count_tokens(header) + count_tokens(separator)
    out: list[str] = []
    buf: list[str] = []
    size = 0
    for row in rows:
        n = count_tokens(row)
        if buf and base + size + n > limit:
            out.append("\n".join([header, separator, *buf]))
            buf, size = [], 0
        buf.append(row)
        size += n
    if buf:
        out.append("\n".join([header, separator, *buf]))
    return out or [markdown]


def _pack(blocks: list[str], cfg: Chunking) -> list[str]:
    units: list[str] = []
    for block in blocks:
        if count_tokens(block) <= cfg.max_tokens:
            units.append(block)
        elif block.lstrip().startswith("|"):
            units.extend(_split_table(block, cfg.target_tokens))
        else:
            units.extend(_split_long_paragraph(block, cfg.target_tokens))

    groups: list[list[str]] = []
    buf: list[str] = []
    size = 0
    for unit in units:
        n = count_tokens(unit)
        if buf and size + n > cfg.target_tokens:
            groups.append(buf)
            carry: list[str] = []
            carried = 0
            for prev in reversed(buf):
                pn = count_tokens(prev)
                if carried + pn > cfg.overlap_tokens:
                    break
                carry.insert(0, prev)
                carried += pn
            buf = list(carry)
            size = carried
        buf.append(unit)
        size += n
    if buf:
        groups.append(buf)
    return ["\n\n".join(g) for g in groups] or [""]


def chunk_clause(clause: Clause, cfg: Chunking) -> list[Chunk]:
    blocks = _blocks_of(clause)
    if not blocks:
        return []

    body = "\n\n".join(blocks)
    pieces = [body] if count_tokens(body) <= cfg.max_tokens else _pack(blocks, cfg)
    pieces = [p for p in pieces if p.strip()]

    base = f"{clause.spec_id}#{clause.number or clause.heading}"
    out: list[Chunk] = []
    for i, piece in enumerate(pieces):
        out.append(
            Chunk(
                id=f"{base}#{i}",
                spec_id=clause.spec_id,
                version=clause.version,
                release=clause.release,
                clause=clause.number,
                heading=clause.heading,
                breadcrumb=clause.breadcrumb,
                level=clause.level,
                normative=clause.normative,
                text=piece.strip(),
                part=i,
                parts=len(pieces),
                source=clause.source,
                tokens=count_tokens(piece),
            )
        )
    return out


def chunk_spec(clauses: Iterable[Clause], cfg: Chunking | None = None) -> list[Chunk]:
    cfg = cfg or Chunking()
    chunks: list[Chunk] = []
    for clause in clauses:
        chunks.extend(chunk_clause(clause, cfg))

    # Clause numbers are meant to be unique within a spec, but unnumbered
    # clauses fall back to their heading and headings repeat ("General").
    # A duplicate id would silently overwrite an entry in the index lookup.
    seen: dict[str, int] = {}
    for chunk in chunks:
        if chunk.id in seen:
            seen[chunk.id] += 1
            chunk.id = f"{chunk.id}~{seen[chunk.id]}"
        else:
            seen[chunk.id] = 0

    for i, chunk in enumerate(chunks):
        if i:
            chunk.prev_id = chunks[i - 1].id
        if i + 1 < len(chunks):
            chunk.next_id = chunks[i + 1].id
    return chunks
