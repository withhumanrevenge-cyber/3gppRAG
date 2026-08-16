"""Minimal WordprocessingML reader.

Parsed straight from the OOXML rather than via python-docx because the clause
parser needs three things that a plain text dump loses: the interleaved order of
paragraphs and tables, the paragraph *style* (3GPP encodes heading depth as
``Heading1``..``Heading9``), and correct handling of revision marks -- several
published specs still carry ``w:del`` runs, whose text must not reach the index.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Subtrees whose text is deleted, field plumbing, or otherwise not body content.
SKIP_TAGS = frozenset({f"{W}del", f"{W}instrText", f"{W}fldChar", f"{W}proofErr", f"{W}delText"})

HEADING_RE = re.compile(r"^heading\s*([1-9])$", re.I)


@dataclass
class Paragraph:
    style: str
    text: str

    @property
    def heading_level(self) -> int | None:
        m = HEADING_RE.match(self.style)
        return int(m.group(1)) if m else None


@dataclass
class Table:
    rows: list[list[str]] = field(default_factory=list)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        width = max(len(r) for r in self.rows)
        norm = [[c.replace("\n", " ").replace("|", "\\|").strip() for c in r] + [""] * (width - len(r)) for r in self.rows]
        head, *body = norm
        out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * width) + "|"]
        out += ["| " + " | ".join(r) + " |" for r in body]
        return "\n".join(out)

    def to_text(self) -> str:
        return "\n".join(" — ".join(c for c in row if c) for row in self.rows if any(row))


Block = Paragraph | Table


def _text_of(node: ET.Element) -> str:
    parts: list[str] = []

    def walk(el: ET.Element) -> None:
        for child in el:
            tag = child.tag
            if tag in SKIP_TAGS:
                continue
            if tag == f"{W}t":
                parts.append(child.text or "")
            elif tag == f"{W}tab":
                parts.append("\t")
            elif tag in (f"{W}br", f"{W}cr"):
                parts.append("\n")
            elif tag == f"{W}noBreakHyphen":
                parts.append("-")
            else:
                walk(child)

    walk(node)
    return "".join(parts).replace("\xa0", " ")


def _style_of(p: ET.Element) -> str:
    node = p.find(f"{W}pPr/{W}pStyle")
    return (node.get(f"{W}val") or "") if node is not None else ""


def _cell_text(tc: ET.Element) -> str:
    lines = [_text_of(p).strip() for p in tc.findall(f"{W}p")]
    return "\n".join(x for x in lines if x)


def _table(tbl: ET.Element) -> Table:
    rows: list[list[str]] = []
    for tr in tbl.findall(f"{W}tr"):
        cells: list[str] = []
        for tc in tr.findall(f"{W}tc"):
            text = _cell_text(tc)
            cells.append(text)
            span = tc.find(f"{W}tcPr/{W}gridSpan")
            if span is not None:
                cells.extend([""] * (int(span.get(f"{W}val") or 1) - 1))
        if cells:
            rows.append(cells)
    return Table(rows)


def read_docx(path: Path) -> list[Block]:
    with zipfile.ZipFile(path) as zf:
        try:
            xml = zf.read("word/document.xml")
        except KeyError:
            return []
    body = ET.fromstring(xml).find(f"{W}body")
    if body is None:
        return []

    blocks: list[Block] = []
    for child in body:
        if child.tag == f"{W}p":
            text = _text_of(child).strip()
            if text:
                blocks.append(Paragraph(_style_of(child), text))
        elif child.tag == f"{W}tbl":
            table = _table(child)
            if table.rows:
                blocks.append(table)
    return blocks
