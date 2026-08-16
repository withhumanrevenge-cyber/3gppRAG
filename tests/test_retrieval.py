import pytest

from telcorag.config import Guard as GuardConfig
from telcorag.corpus.chunker import Chunk
from telcorag.corpus.parser import Clause
from telcorag.glossary import build as build_glossary
from telcorag.index.bm25 import BM25, tokenize
from telcorag.retrieval.fusion import reciprocal_rank_fusion, top_n
from telcorag.retrieval.pipeline import Retrieved, RetrievalResult
from telcorag.answer.guard import assess
from telcorag.answer.verifier import Verifier, hard_tokens, parse_claims


class TestTokenizer:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("TS 24.501", "24.501"),
            ("timer T3510", "t3510"),
            ("the 5G-GUTI value", "5g-guti"),
            ("5GMM-IDLE mode", "5gmm-idle"),
        ],
    )
    def test_identifiers_kept_whole(self, text, expected):
        assert expected in tokenize(text)

    def test_compound_also_emits_parts(self):
        toks = tokenize("5G-GUTI")
        assert "5g-guti" in toks and "guti" in toks

    def test_stopwords_removed_by_default(self):
        assert "the" not in tokenize("the UE shall register")

    def test_stopwords_kept_when_asked(self):
        assert "the" in tokenize("the UE", keep_stopwords=True)


class TestBM25:
    @pytest.fixture
    def index(self):
        docs = [
            "The UE shall start timer T3510 on sending REGISTRATION REQUEST.",
            "The UE shall start timer T3512 for periodic registration update.",
            "Security context handling for 5G NAS as defined in TS 33.501.",
            "Random access procedure on NR as described in the MAC specification.",
        ]
        return BM25().fit(docs), docs

    def test_exact_identifier_beats_near_neighbour(self, index):
        bm25, _ = index
        idx, _ = bm25.search("T3512", top_k=4)
        assert idx[0] == 1

    def test_ranks_relevant_document_first(self, index):
        bm25, _ = index
        idx, _ = bm25.search("REGISTRATION REQUEST timer", top_k=4)
        assert idx[0] == 0

    def test_unknown_term_returns_nothing(self, index):
        bm25, _ = index
        idx, _ = bm25.search("quantum entanglement zebra", top_k=4)
        assert len(idx) == 0

    def test_expansion_terms_are_downweighted(self, index):
        bm25, _ = index
        plain = bm25.score(["nr"])
        boosted = bm25.score(["nr"], [0.45])
        assert boosted.max() < plain.max()

    def test_roundtrip(self, index, tmp_path):
        bm25, _ = index
        bm25.save(tmp_path)
        restored = BM25.load(tmp_path)
        a, _ = bm25.search("T3510", 3)
        b, _ = restored.search("T3510", 3)
        assert list(a) == list(b)


class TestGlossary:
    @pytest.fixture
    def lexicon(self):
        clause = Clause(
            "24.501", "20.0.0", 20, "3.2", "Abbreviations", 2,
            paragraphs=[
                "5GMM\t5GS Mobility Management",
                "SMF\tSession Management Function",
                "SUCI\tSubscription Concealed Identifier",
                "junk line without a tab",
            ],
        )
        return build_glossary([clause])

    def test_extracts_pairs(self, lexicon):
        assert lexicon.lookup("SMF") == ["Session Management Function"]
        assert len(lexicon) == 3

    def test_acronym_expands_to_phrase(self, lexicon):
        assert "5GS Mobility Management" in lexicon.expand("explain 5GMM states")

    def test_phrase_contracts_to_acronym(self, lexicon):
        assert "SMF" in lexicon.expand("what does the Session Management Function do")

    def test_ignores_clauses_that_are_not_abbreviations(self):
        clause = Clause("24.501", "20.0.0", 20, "3.1", "Definitions", 2, paragraphs=["ABC\tSomething"])
        assert len(build_glossary([clause])) == 0


class TestFusion:
    def test_document_in_both_lists_outranks_either_leader(self):
        fused = reciprocal_rank_fusion([[10, 20, 30], [20, 40, 50]], k=60)
        assert max(fused, key=fused.get) == 20

    def test_weights_shift_the_balance(self):
        fused = reciprocal_rank_fusion([[1], [2]], k=60, weights=[3.0, 1.0])
        assert fused[1] > fused[2]

    def test_top_n_orders_descending(self):
        ranked = top_n({1: 0.1, 2: 0.9, 3: 0.5}, 2)
        assert [doc for doc, _ in ranked] == [2, 3]


def make_hit(score: float, overlap: float = 0.5) -> Retrieved:
    chunk = Chunk("id", "24.501", "20.0.0", 20, "5.1", "H", "5 > 5.1", 2, True, "text")
    return Retrieved(chunk=chunk, score=score, fusion=0.1, overlap=overlap)


class TestAbstentionGate:
    cfg = GuardConfig(min_top_score=0.32, min_support=0.18, min_lexical_overlap=0.08)

    def test_answers_on_strong_retrieval(self):
        result = RetrievalResult("q", hits=[make_hit(0.9), make_hit(0.7), make_hit(0.6)])
        assert assess(result, self.cfg).answerable

    def test_refuses_on_weak_top_score(self):
        result = RetrievalResult("q", hits=[make_hit(0.10), make_hit(0.05)])
        gate = assess(result, self.cfg)
        assert not gate.answerable and "below" in gate.reason

    def test_refuses_when_vocabulary_absent(self):
        result = RetrievalResult("q", hits=[make_hit(0.9, overlap=0.01)] * 3)
        gate = assess(result, self.cfg)
        assert not gate.answerable and "vocabulary" in gate.reason

    def test_refuses_with_no_hits(self):
        assert not assess(RetrievalResult("q", hits=[]), self.cfg).answerable

    def test_refuses_on_unindexed_specification(self):
        result = RetrievalResult("q", hits=[make_hit(0.99)] * 3, unindexed_specs=["36.331"])
        gate = assess(result, self.cfg)
        assert not gate.answerable
        assert "TS 36.331 is not in the indexed corpus" == gate.reason

    def test_refuses_on_term_absent_from_corpus(self):
        result = RetrievalResult("q", hits=[make_hit(0.97)] * 3, unknown_terms=["t9999"])
        gate = assess(result, self.cfg)
        assert not gate.answerable
        assert "t9999" in gate.reason

    def test_premise_failure_beats_high_scores(self):
        """The whole point: strong retrieval must not rescue a false premise."""
        result = RetrievalResult("q", hits=[make_hit(0.999)] * 3, unknown_terms=["blood"])
        assert not assess(result, self.cfg).answerable

    def test_premise_check_can_be_disabled(self):
        cfg = GuardConfig(min_top_score=0.32, min_support=0.18, min_lexical_overlap=0.08, premise_check=False)
        result = RetrievalResult("q", hits=[make_hit(0.99)] * 3, unknown_terms=["t9999"], unindexed_specs=["36.331"])
        assert assess(result, cfg).answerable

    def test_confidence_tracks_score(self):
        low = assess(RetrievalResult("q", hits=[make_hit(0.4)] * 3), self.cfg).confidence
        high = assess(RetrievalResult("q", hits=[make_hit(0.95)] * 3), self.cfg).confidence
        assert high > low


class TestClaimParsing:
    def test_extracts_markers(self):
        claims = parse_claims("The UE starts T3510. [S1][S3]\nAnother statement. [S2]")
        assert claims[0][1] == [1, 3]
        assert "[S1]" not in claims[0][0]

    def test_strips_bullets(self):
        assert parse_claims("- A cited line. [S1]")[0][0] == "A cited line."

    def test_line_without_marker_has_no_citations(self):
        assert parse_claims("Uncited assertion.")[0][1] == []


class TestHardTokens:
    def test_finds_values_that_matter(self):
        found = [t.lower() for t in hard_tokens("Timer T3510 expires after 15 s per TS 24.501.")]
        assert "t3510" in found and "15" in found and "ts 24.501" in found

    def test_ignores_ambient_terms(self):
        assert "UE" not in hard_tokens("The UE responds.")

    def test_captures_message_names(self):
        assert "REGISTRATION REQUEST" in hard_tokens("Send a REGISTRATION REQUEST message.")


class TestVerifier:
    SOURCE = (
        "3GPP TS 24.501 v20.0.0 — clause 5.5.1.2.2\n"
        "The UE shall start timer T3510 when it sends the REGISTRATION REQUEST message. "
        "The default value of T3510 is 15 s. On expiry the UE shall abort the procedure."
    )

    @pytest.fixture
    def verifier(self):
        return Verifier(min_support=0.45)

    def test_faithful_claim_passes(self, verifier):
        out = verifier.verify("The UE shall start timer T3510 when it sends the REGISTRATION REQUEST message. [S1]", [self.SOURCE])
        assert out.claims[0].supported
        assert out.groundedness == 1.0

    def test_wrong_timer_is_rejected(self, verifier):
        out = verifier.verify("The UE shall start timer T3512 when it sends the REGISTRATION REQUEST message. [S1]", [self.SOURCE])
        assert not out.claims[0].supported
        assert "T3512" in out.claims[0].ungrounded_tokens

    def test_fabricated_number_is_rejected(self, verifier):
        out = verifier.verify("The default value of T3510 is 30 s. [S1]", [self.SOURCE])
        assert not out.claims[0].supported
        assert "30" in out.claims[0].ungrounded_tokens

    def test_correct_number_survives_unit_wording(self, verifier):
        out = verifier.verify("The default value of T3510 is 15 seconds. [S1]", [self.SOURCE])
        assert out.claims[0].supported

    def test_wrong_spec_reference_is_rejected(self, verifier):
        out = verifier.verify("This behaviour is defined in TS 24.301. [S1]", [self.SOURCE])
        assert not out.claims[0].supported

    def test_uncited_claim_is_rejected(self, verifier):
        out = verifier.verify("The UE starts a timer.", [self.SOURCE])
        assert not out.claims[0].supported
        assert out.claims[0].reason == "no valid citation"

    def test_out_of_range_citation_is_rejected(self, verifier):
        out = verifier.verify("The UE starts timer T3510. [S7]", [self.SOURCE])
        assert not out.claims[0].supported

    def test_off_topic_claim_fails_lexical_support(self, verifier):
        out = verifier.verify("Handover preparation uses the Xn interface between base stations. [S1]", [self.SOURCE])
        assert not out.claims[0].supported

    def test_groundedness_is_a_ratio(self, verifier):
        answer = "The UE shall start timer T3510 when it sends the REGISTRATION REQUEST message. [S1]\nThe value is 30 s. [S1]"
        out = verifier.verify(answer, [self.SOURCE])
        assert out.groundedness == 0.5
        assert len(out.kept) == 1 and len(out.dropped) == 1
