import io
import zipfile
from pathlib import Path

import pytest

from telcorag.config import Chunking
from telcorag.corpus import download as dl
from telcorag.corpus.chunker import chunk_spec, count_tokens
from telcorag.corpus.ooxml import Paragraph, Table, read_docx
from telcorag.corpus.parser import parse_spec

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def make_docx(tmp_path: Path, body_xml: str, name: str = "t.docx") -> Path:
    document = f'<?xml version="1.0"?><w:document xmlns:w="{W}"><w:body>{body_xml}</w:body></w:document>'
    path = tmp_path / name
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml", document)
    path.write_bytes(buf.getvalue())
    return path


def para(text: str, style: str | None = None) -> str:
    pr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    runs = "".join(f"<w:r><w:t>{part}</w:t></w:r>" if i == 0 else f"<w:r><w:tab/><w:t>{part}</w:t></w:r>"
                   for i, part in enumerate(text.split("\t")))
    return f"<w:p>{pr}{runs}</w:p>"


class TestVersionCodes:
    @pytest.mark.parametrize(
        "code,expected",
        [("k00", (20, 0, 0)), ("i50", (18, 5, 0)), ("g21", (16, 2, 1)), ("900", (9, 0, 0))],
    )
    def test_decode(self, code, expected):
        assert dl.decode_version(code) == expected

    def test_roundtrip(self):
        for code in ("k00", "i50", "j62", "b31"):
            assert dl.encode_version(dl.format_version(code)) == code

    def test_archive_url(self):
        url = dl.archive_url("24.501", "20.0.0")
        assert url.endswith("/24_series/24.501/24501-k00.zip")

    def test_rejects_bad_code(self):
        with pytest.raises(ValueError):
            dl.decode_version("kk")


class TestOoxml:
    def test_reading_order_and_styles(self, tmp_path):
        xml = para("1\tScope", "Heading1") + para("Body text.") + (
            f'<w:tbl><w:tr><w:tc><w:p><w:r><w:t>H1</w:t></w:r></w:p></w:tc>'
            f'<w:tc><w:p><w:r><w:t>H2</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
        )
        blocks = read_docx(make_docx(tmp_path, xml))
        assert isinstance(blocks[0], Paragraph) and blocks[0].heading_level == 1
        assert isinstance(blocks[2], Table)
        assert blocks[2].rows == [["H1", "H2"]]

    def test_tabs_survive(self, tmp_path):
        blocks = read_docx(make_docx(tmp_path, para("5GMM\t5GS Mobility Management")))
        assert "\t" in blocks[0].text

    def test_deleted_runs_excluded(self, tmp_path):
        xml = (
            f'<w:p><w:r><w:t>kept</w:t></w:r>'
            f'<w:del><w:r><w:delText>REMOVED</w:delText></w:r></w:del>'
            f'<w:r><w:t> tail</w:t></w:r></w:p>'
        )
        blocks = read_docx(make_docx(tmp_path, xml))
        assert "REMOVED" not in blocks[0].text
        assert blocks[0].text == "kept tail"


class TestClauseParser:
    def build(self, tmp_path, xml):
        return parse_spec([make_docx(tmp_path, xml)], "24.501", "20.0.0", 20)

    def test_numbering_and_ancestry(self, tmp_path):
        xml = (
            para("4\tGeneral", "Heading1") + para("intro")
            + para("4.4\tNAS security", "Heading2") + para("sec")
            + para("4.4.2\tHandling of contexts", "Heading3") + para("ctx")
            + para("4.4.2.1\tGeneral", "Heading4") + para("leaf text")
        )
        clauses = self.build(tmp_path, xml)
        leaf = [c for c in clauses if c.number == "4.4.2.1"][0]
        assert leaf.heading == "General"
        assert leaf.level == 4
        assert leaf.breadcrumb.startswith("4 General > 4.4 NAS security > 4.4.2 Handling")
        assert leaf.citation == "3GPP TS 24.501 v20.0.0, clause 4.4.2.1"

    @pytest.mark.parametrize("number", ["4.2A", "4.5.2A", "6.2.4.2a", "5.4.1.2.3B"])
    def test_letter_suffixed_clauses(self, tmp_path, number):
        xml = para(f"{number}\tSome heading", "Heading2") + para("content here")
        clauses = self.build(tmp_path, xml)
        assert clauses[0].number == number

    def test_annex_and_informative_flag(self, tmp_path):
        xml = (
            para("Annex A (informative):\tExample flows", "Heading1") + para("body")
            + para("A.1\tFirst", "Heading2") + para("more body")
        )
        clauses = self.build(tmp_path, xml)
        assert clauses[0].number == "A"
        assert all(not c.normative for c in clauses)

    def test_change_history_dropped(self, tmp_path):
        xml = para("1\tScope", "Heading1") + para("real") + para("Change history", "Heading1") + para("noise")
        clauses = self.build(tmp_path, xml)
        assert [c.number for c in clauses] == ["1"]

    def test_empty_clause_dropped(self, tmp_path):
        xml = para("1\tScope", "Heading1") + para("2\tReferences", "Heading1") + para("has body")
        clauses = self.build(tmp_path, xml)
        assert [c.number for c in clauses] == ["2"]


class TestChunker:
    def clause(self, paragraphs, tables=()):
        from telcorag.corpus.parser import Clause

        return Clause("24.501", "20.0.0", 20, "5.1", "Head", 2, [("5", "Top")], list(paragraphs), list(tables))

    def test_short_clause_is_one_chunk(self):
        chunks = chunk_spec([self.clause(["short body text"])], Chunking())
        assert len(chunks) == 1
        assert chunks[0].parts == 1
        assert "clause 5.1 Head" in chunks[0].header

    def test_long_clause_splits_with_overlap(self):
        paras = [f"Sentence number {i} " + "filler word " * 40 for i in range(12)]
        cfg = Chunking(target_tokens=120, max_tokens=160, overlap_tokens=30)
        chunks = chunk_spec([self.clause(paras)], cfg)
        assert len(chunks) > 1
        assert all(c.parts == len(chunks) for c in chunks)
        assert chunks[0].next_id == chunks[1].id

    def test_big_table_splits_and_repeats_header(self):
        header = "| IE | Value |\n|---|---|\n"
        rows = "\n".join(f"| field{i} | {'x' * 40} |" for i in range(120))
        cfg = Chunking(target_tokens=150, max_tokens=200)
        chunks = chunk_spec([self.clause([], [header + rows])], cfg)
        assert len(chunks) > 1
        assert all(c.text.startswith("| IE | Value |") for c in chunks)

    def test_ids_are_stable_and_unique(self):
        chunks = chunk_spec([self.clause(["a" * 50]), self.clause(["b" * 50])], Chunking())
        assert len({c.id for c in chunks}) == len(chunks)
        assert chunks[0].id.startswith("24.501#5.1#")

    def test_token_counter_handles_identifiers(self):
        assert count_tokens("TS 24.501 timer T3510") == 4
