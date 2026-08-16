"""End-to-end tests over a miniature in-memory corpus.

These exercise the generated-answer path without needing an API key, by
substituting an LLM that returns canned output — including output that
deliberately fabricates a value, which is the case the system exists to catch.
"""

import pytest

from telcorag.answer.generator import AnswerEngine
from telcorag.answer.llm import LLMError
from telcorag.config import Settings
from telcorag.corpus.chunker import Chunk
from telcorag.glossary import Glossary
from telcorag.index.embeddings import LSABackend
from telcorag.index.store import Index
from telcorag.retrieval.pipeline import Retriever
from telcorag.retrieval.rerank import LexicalReranker

CLAUSES = [
    (
        "5.5.1.2.2",
        "Initial registration initiation",
        "The UE shall start timer T3510 when it sends the REGISTRATION REQUEST message to the AMF. "
        "The default value of timer T3510 is 15 s. On expiry of T3510 the UE shall abort the initial registration procedure.",
    ),
    (
        "5.4.3.2",
        "Identification procedure initiation by the network",
        "The AMF shall initiate the identification procedure by sending an IDENTITY REQUEST message to the UE "
        "and starting timer T3570. The IDENTITY REQUEST message specifies the requested identity type.",
    ),
    (
        "5.4.2.2",
        "NAS security mode command acceptance by the UE",
        "The UE shall accept the SECURITY MODE COMMAND message if it can be integrity checked with the indicated "
        "5G NAS security context and the selected NAS algorithms are acceptable.",
    ),
    (
        "6.4.1.2",
        "UE requested PDU session establishment procedure initiation",
        "In order to initiate the UE requested PDU session establishment procedure the UE shall send a "
        "PDU SESSION ESTABLISHMENT REQUEST message and start timer T3580.",
    ),
]


class FakeLLM:
    def __init__(self, reply: str, name: str = "fake:test") -> None:
        self.reply = reply
        self.name = name
        self.available = True
        self.calls: list = []

    def complete(self, messages, temperature=0.0, max_tokens=1024):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture(scope="module")
def index():
    chunks = [
        Chunk(
            id=f"24.501#{clause}#0",
            spec_id="24.501",
            version="20.0.0",
            release=20,
            clause=clause,
            heading=heading,
            breadcrumb=f"5 Elementary procedures > {clause} {heading}",
            level=4,
            normative=True,
            text=text,
            tokens=len(text.split()),
        )
        for clause, heading, text in CLAUSES
    ]
    glossary = Glossary({"AMF": ["Access and Mobility Management Function"]}, {"access and mobility management function": "AMF"})
    return Index.build(chunks, LSABackend(dims=16), glossary)


def engine_with(index, llm, cfg=None):
    settings = cfg or Settings()
    retriever = Retriever(index, settings, reranker=LexicalReranker())
    return AnswerEngine(index, settings, llm=llm, retriever=retriever)


class TestRetrieval:
    def test_finds_the_right_clause(self, index):
        retriever = Retriever(index, Settings(), reranker=LexicalReranker())
        result = retriever.retrieve("Which timer starts on sending REGISTRATION REQUEST?")
        assert result.hits[0].chunk.clause == "5.5.1.2.2"

    def test_clause_reference_is_pinned(self, index):
        retriever = Retriever(index, Settings(), reranker=LexicalReranker())
        result = retriever.retrieve("What does TS 24.501 clause 5.4.3.2 say?")
        assert result.hits[0].chunk.clause == "5.4.3.2"
        assert result.hits[0].pinned

    def test_acronym_expansion_recorded(self, index):
        retriever = Retriever(index, Settings(), reranker=LexicalReranker())
        result = retriever.retrieve("what does the Access and Mobility Management Function send")
        assert "AMF" in result.expansions


class TestGeneratedPath:
    def test_faithful_answer_survives(self, index):
        llm = FakeLLM("The UE shall start timer T3510 when it sends the REGISTRATION REQUEST message to the AMF. [S1]")
        answer = engine_with(index, llm).ask("Which timer starts on sending REGISTRATION REQUEST?")
        assert not answer.abstained
        assert answer.mode == "generated"
        assert answer.groundedness == 1.0
        assert "T3510" in answer.text

    def test_fabricated_value_is_stripped(self, index):
        llm = FakeLLM(
            "The UE shall start timer T3510 when it sends the REGISTRATION REQUEST message. [S1]\n"
            "The default value of timer T3510 is 45 s. [S1]"
        )
        answer = engine_with(index, llm).ask("Which timer starts on sending REGISTRATION REQUEST and what is its value?")
        assert not answer.abstained
        assert "45" not in answer.text
        assert answer.groundedness == 0.5
        assert any("45" in c.ungrounded_tokens for c in answer.claims if not c.supported)

    def test_wholly_fabricated_answer_is_withdrawn(self, index):
        llm = FakeLLM("The UE shall start timer T9999 and wait 120 s before retrying. [S1]")
        answer = engine_with(index, llm).ask("Which timer starts on sending REGISTRATION REQUEST?")
        assert answer.abstained
        assert "grounding verification" in answer.reason

    def test_model_declared_insufficiency_is_honoured(self, index):
        llm = FakeLLM("INSUFFICIENT_CONTEXT")
        answer = engine_with(index, llm).ask("Which timer starts on sending REGISTRATION REQUEST?")
        assert answer.abstained
        assert "do not contain" in answer.reason

    def test_llm_failure_falls_back_to_extraction(self, index):
        llm = FakeLLM(LLMError("upstream 503"))
        answer = engine_with(index, llm).ask("Which timer starts on sending REGISTRATION REQUEST?")
        assert answer.mode == "extractive"
        assert not answer.abstained
        assert "llm_error" in answer.diagnostics

    def test_uncited_claims_are_dropped(self, index):
        llm = FakeLLM("The UE shall start timer T3510 when it sends the REGISTRATION REQUEST message. [S1]\nTimers are important.")
        answer = engine_with(index, llm).ask("Which timer starts on sending REGISTRATION REQUEST?")
        assert "Timers are important" not in answer.text


class TestExtractivePath:
    def test_answers_verbatim_and_verifies(self, index):
        from telcorag.answer.llm import NullLLM

        answer = engine_with(index, NullLLM()).ask("Which timer starts on sending REGISTRATION REQUEST?")
        assert answer.mode == "extractive"
        assert not answer.abstained
        assert answer.groundedness == 1.0
        assert answer.sources[0].clause == "5.5.1.2.2"

    def test_sources_carry_archive_links(self, index):
        from telcorag.answer.llm import NullLLM

        answer = engine_with(index, NullLLM()).ask("Which timer starts on sending REGISTRATION REQUEST?")
        assert answer.sources[0].url.endswith("24501-k00.zip")


class TestPremiseCheck:
    def test_suppressed_on_a_small_corpus(self, index):
        """Absence of evidence is only evidence of absence on a full corpus."""
        retriever = Retriever(index, Settings(), reranker=LexicalReranker())
        unknown, _ = retriever._premise("Which timer starts on sending REGISTRATION REQUEST?")
        assert unknown == []

    def test_flags_unknown_identifier_once_vocabulary_is_large(self, index, monkeypatch):
        from telcorag.retrieval import pipeline

        monkeypatch.setattr(pipeline, "MIN_VOCAB_FOR_PREMISE", 0)
        retriever = Retriever(index, Settings(), reranker=LexicalReranker())
        unknown, _ = retriever._premise("What is the value of timer T9999?")
        assert "t9999" in unknown

    def test_flags_specification_outside_the_index(self, index):
        retriever = Retriever(index, Settings(), reranker=LexicalReranker())
        _, missing = retriever._premise("What does TS 36.331 say about reconfiguration?")
        assert missing == ["36.331"]

    def test_indexed_specification_is_not_flagged(self, index):
        retriever = Retriever(index, Settings(), reranker=LexicalReranker())
        _, missing = retriever._premise("What does TS 24.501 say about registration?")
        assert missing == []

    def test_unindexed_spec_question_is_refused_end_to_end(self, index):
        from telcorag.answer.llm import NullLLM

        answer = engine_with(index, NullLLM()).ask("What does TS 36.331 specify about RRC reconfiguration?")
        assert answer.abstained
        assert "36.331" in answer.reason


class TestAbstention:
    def test_out_of_domain_question_is_refused(self, index):
        from telcorag.answer.llm import NullLLM

        answer = engine_with(index, NullLLM()).ask("What is the capital of France and its population?")
        assert answer.abstained

    def test_abstention_still_reports_what_it_looked_at(self, index):
        from telcorag.answer.llm import NullLLM

        answer = engine_with(index, NullLLM()).ask("How do I configure BGP route reflectors on a Cisco router?")
        assert answer.abstained
        assert answer.confidence < 0.6


class TestSerialisation:
    def test_answer_serialises_for_the_api(self, index):
        llm = FakeLLM("The UE shall start timer T3510 when it sends the REGISTRATION REQUEST message. [S1]")
        payload = engine_with(index, llm).ask("Which timer starts on sending REGISTRATION REQUEST?").as_dict()
        assert set(payload) >= {"answer", "abstained", "confidence", "groundedness", "sources", "claims", "diagnostics"}
        assert payload["claims"][0]["citations"] == [1]

    def test_index_roundtrip(self, index, tmp_path):
        index.save(tmp_path)
        restored = Index.load(tmp_path, lsa_dims=16)
        assert len(restored) == len(index)
        assert restored.chunks[0].clause == index.chunks[0].clause
        assert len(restored.glossary) == len(index.glossary)
