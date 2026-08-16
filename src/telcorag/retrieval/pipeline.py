"""Query -> ranked clauses.

Order of operations, and why each step is there:

1. glossary expansion   closes the acronym/expansion vocabulary gap
2. structural parsing    "TS 24.501 clause 5.4.1" is an address, not a topic
3. BM25 + dense          exact identifiers and paraphrase respectively
4. RRF                   rank-space fusion, no score normalisation needed
5. sibling expansion     a split clause must be able to win on its right part
6. cross-encoder rerank   turns a candidate pool into a calibrated ordering
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..config import Retrieval, Settings, settings as default_settings
from ..corpus.chunker import Chunk
from ..index.bm25 import tokenize
from ..index.store import Index
from . import rerank as rerank_mod
from .fusion import reciprocal_rank_fusion, top_n

SPEC_REF_RE = re.compile(r"\b(?:TS|TR)\s*(\d{2}\.\d{3})\b", re.I)
CLAUSE_REF_RE = re.compile(r"\b(?:clause|section|subclause)\s+((?:\d+|[A-Z])(?:\.\d+[A-Za-z]?)*)", re.I)

STRUCTURAL_BONUS = 0.55
CLAUSE_BONUS = 1.6

# Below this vocabulary size, a missing word says more about the corpus than
# about the question, so the premise check stands down.
MIN_VOCAB_FOR_PREMISE = 5000


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    fusion: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    overlap: float = 0.0
    pinned: bool = False


@dataclass
class RetrievalResult:
    query: str
    expansions: list[str] = field(default_factory=list)
    hits: list[Retrieved] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    unknown_terms: list[str] = field(default_factory=list)
    unindexed_specs: list[str] = field(default_factory=list)

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def support(self) -> float:
        """Mean score of the supporting hits below the top one."""
        rest = [h.score for h in self.hits[1:3]]
        return sum(rest) / len(rest) if rest else 0.0


def _idf_map(index: Index) -> dict[str, float]:
    return {term: float(index.bm25.idf[tid]) for term, tid in index.bm25.vocab.items()}


class Retriever:
    def __init__(self, index: Index, cfg: Settings | None = None, reranker=None) -> None:
        self.index = index
        self.settings = cfg or default_settings
        self.cfg: Retrieval = self.settings.retrieval
        self.idf = _idf_map(index)
        self.reranker = reranker or rerank_mod.resolve(self.cfg.reranker, self.idf)

    def _premise(self, query: str) -> tuple[list[str], list[str]]:
        """Terms and specifications the question assumes but the corpus lacks.

        The corpus vocabulary is ~50k terms drawn from 17M characters of formal
        telecom prose. A content word absent from *all* of it is strong evidence
        that the question rests on something the specifications never discuss --
        a timer that does not exist, a vendor, another standards body. Catching
        that here is far more reliable than hoping a generator declines, because
        retrieval will always return its best six passages regardless.

        Only *entity-shaped* tokens are flagged: ones carrying a digit or hyphen,
        or capitalised in the question. That covers what this check is actually
        for -- invented identifiers (``T9999``), vendors (``Cisco``, ``Juniper``),
        rival standards (``Wi-Fi``), and plain off-domain subjects (``France``,
        ``Python``) -- while leaving ordinary lowercase English alone.

        An earlier version also flagged any unknown word of four or more letters.
        That was wrong, and measurably so: the specifications are written in
        formal prose and simply never use conversational verbs, so questions
        phrased in plain English were refused for containing "prove" or
        "disagree". Absence from the corpus tells you about the *corpus's
        register* as much as about the question, and only entity-shaped tokens
        carry the signal reliably.

        The check is suppressed below ``MIN_VOCAB_FOR_PREMISE``, because absence
        of evidence is only evidence of absence when the corpus is comprehensive.
        The unindexed-specification check has no such dependency and always
        applies.
        """
        indexed = {c.spec_id for c in self.index.chunks}
        missing = sorted({s for s in SPEC_REF_RE.findall(query) if s not in indexed})

        vocab = self.index.bm25.vocab
        if len(vocab) < MIN_VOCAB_FOR_PREMISE:
            return [], missing

        capitalised = {w.lower() for w in re.findall(r"\b[A-Z][A-Za-z0-9\-]*\b", query)}
        unknown: list[str] = []
        for token in dict.fromkeys(tokenize(query)):
            if token in vocab:
                continue
            if any(ch.isdigit() for ch in token) or "-" in token or token in capitalised:
                unknown.append(token)
        return unknown, missing

    def _structural(self, query: str) -> tuple[set[str], list[tuple[str | None, str]]]:
        specs = {m.group(1) for m in SPEC_REF_RE.finditer(query)}
        clauses = [(next(iter(specs), None), m.group(1)) for m in CLAUSE_REF_RE.finditer(query)]
        return specs, clauses

    def _pinned(self, specs: set[str], clauses: list[tuple[str | None, str]]) -> list[int]:
        if not clauses:
            return []
        out: list[int] = []
        for spec_hint, number in clauses:
            for i, chunk in enumerate(self.index.chunks):
                if chunk.clause != number:
                    continue
                if spec_hint and chunk.spec_id != spec_hint:
                    continue
                if not spec_hint and specs and chunk.spec_id not in specs:
                    continue
                out.append(i)
        return out[:8]

    def _overlap(self, query: str, chunk: Chunk) -> float:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return 0.0
        body = chunk.indexed_text.lower()
        total = sum(self.idf.get(t, 6.0) for t in terms) or 1.0
        hit = sum(self.idf.get(t, 6.0) for t in terms if t in body)
        return hit / total

    def _siblings(self, positions: list[int]) -> list[int]:
        window = self.cfg.neighbour_window
        if window <= 0:
            return []
        extra: set[int] = set()
        for pos in positions:
            chunk = self.index.chunks[pos]
            if chunk.parts <= 1:
                continue
            for offset in range(-window, window + 1):
                neighbour = pos + offset
                if 0 <= neighbour < len(self.index.chunks) and neighbour not in positions:
                    if self.index.chunks[neighbour].clause == chunk.clause:
                        extra.add(neighbour)
        return sorted(extra)

    def _diversify(self, hits: list[Retrieved], limit: int) -> list[Retrieved]:
        """Cap how many parts of one clause may occupy the final results.

        A long clause is indexed as several parts, and sibling expansion puts all
        of them in front of the reranker so the best one can win. They then tend
        to score almost identically, and without a cap a single clause can take
        four of six slots -- burning the generator's context budget on one
        passage and starving the answer of corroborating clauses.
        """
        cap = max(1, self.cfg.max_parts_per_clause)
        counts: dict[tuple[str, str], int] = {}
        out: list[Retrieved] = []
        for hit in hits:
            key = (hit.chunk.spec_id, hit.chunk.clause)
            if counts.get(key, 0) >= cap:
                continue
            counts[key] = counts.get(key, 0) + 1
            out.append(hit)
            if len(out) >= limit:
                break
        return out

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        spec_filter: set[str] | None = None,
        mode: str = "hybrid",
        rerank: bool = True,
    ) -> RetrievalResult:
        top_k = top_k or self.cfg.final_k
        expansions = self.index.glossary.expand(query) if mode != "dense" else []
        specs, clause_refs = self._structural(query)
        unknown_terms, unindexed_specs = self._premise(query)

        bm25_list: list[int] = []
        dense_list: list[int] = []
        if mode in ("hybrid", "bm25"):
            bm25_idx, _ = self.index.lexical_search(query, self.cfg.bm25_candidates, expansions)
            bm25_list = [int(i) for i in bm25_idx]
        if mode in ("hybrid", "dense"):
            query_vec = self.index.backend.encode_queries([query])[0]
            dense_idx, _ = self.index.dense_search(query_vec, self.cfg.dense_candidates)
            dense_list = [int(i) for i in dense_idx]

        fused = reciprocal_rank_fusion(
            [bm25_list, dense_list],
            k=self.cfg.rrf_k,
            weights=[self.cfg.bm25_weight, self.cfg.dense_weight],
        )

        if specs:
            for pos in list(fused):
                if self.index.chunks[pos].spec_id in specs:
                    fused[pos] *= 1.0 + STRUCTURAL_BONUS

        pinned = self._pinned(specs, clause_refs)
        for pos in pinned:
            fused[pos] = fused.get(pos, 0.0) + CLAUSE_BONUS / (self.cfg.rrf_k + 1)

        if spec_filter:
            fused = {p: s for p, s in fused.items() if self.index.chunks[p].spec_id in spec_filter}

        candidates = [p for p, _ in top_n(fused, self.cfg.rerank_depth)]
        candidates.extend(p for p in self._siblings(candidates) if p not in candidates)
        if not candidates:
            return RetrievalResult(query, expansions, [], {"reason": "no candidates"}, unknown_terms, unindexed_specs)

        if rerank:
            scores = self.reranker.score(query, [self.index.chunks[p].indexed_text for p in candidates])
        else:
            ceiling = max(fused.values()) or 1.0
            scores = [fused.get(p, 0.0) / ceiling for p in candidates]
        bm25_rank = {p: r for r, p in enumerate(bm25_list)}
        dense_rank = {p: r for r, p in enumerate(dense_list)}

        hits = [
            Retrieved(
                chunk=self.index.chunks[pos],
                score=float(score),
                fusion=float(fused.get(pos, 0.0)),
                bm25_rank=bm25_rank.get(pos),
                dense_rank=dense_rank.get(pos),
                overlap=self._overlap(query, self.index.chunks[pos]),
                pinned=pos in pinned,
            )
            for pos, score in zip(candidates, scores)
        ]
        hits.sort(key=lambda h: (h.pinned, h.score), reverse=True)
        hits = self._diversify(hits, top_k)

        return RetrievalResult(
            query=query,
            expansions=expansions,
            hits=hits[:top_k],
            unknown_terms=unknown_terms,
            unindexed_specs=unindexed_specs,
            diagnostics={
                "candidates": len(candidates),
                "unknown_terms": unknown_terms,
                "unindexed_specs": unindexed_specs,
                "mode": mode,
                "bm25_hits": len(bm25_list),
                "dense_hits": len(dense_list),
                "reranker": self.reranker.name if rerank else "none",
                "spec_hints": sorted(specs),
                "clause_hints": [c for _, c in clause_refs],
                "pinned": len(pinned),
            },
        )
