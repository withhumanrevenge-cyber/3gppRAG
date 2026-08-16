"""Reranking of fused candidates.

Fusion produces a good candidate pool but a poorly *calibrated* ordering, and
this system needs calibration specifically: the abstention gate refuses to
answer below a score threshold, so the score has to mean something. A
cross-encoder reads query and passage jointly and emits a relevance logit that,
after a sigmoid, separates "this clause answers the question" from "this clause
merely shares vocabulary with it" far more sharply than cosine ever does.

The lexical reranker is the no-download path. It is genuinely weaker at
paraphrase, but it is deliberately built from IDF-weighted coverage so that its
output still spreads across [0, 1] and the same gate logic applies.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from ..index.bm25 import tokenize

WORD_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.name = f"cross-encoder:{model_name}"
        self.batch_size = batch_size
        self._model = CrossEncoder(model_name, max_length=512)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        raw = self._model.predict(
            [(query, p) for p in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [1.0 / (1.0 + math.exp(-float(x))) for x in raw]


class LexicalReranker:
    name = "lexical"

    def __init__(self, idf: dict[str, float] | None = None) -> None:
        self._idf = idf or {}

    def _weight(self, term: str) -> float:
        return self._idf.get(term, 6.0)

    def score(self, query: str, passages: list[str]) -> list[float]:
        q_terms = list(dict.fromkeys(tokenize(query)))
        if not q_terms:
            return [0.0] * len(passages)
        total = sum(self._weight(t) for t in q_terms) or 1.0
        q_lower = query.lower()

        out: list[float] = []
        for passage in passages:
            body = passage.lower()
            present = {t for t in q_terms if t in body}
            coverage = sum(self._weight(t) for t in present) / total

            bigrams = [" ".join(p) for p in zip(q_terms, q_terms[1:])]
            phrase = sum(1 for bg in bigrams if bg in body) / max(len(bigrams), 1)
            head = body.split("\n\n", 1)[0]
            heading_hit = sum(1 for t in present if t in head) / max(len(q_terms), 1)
            exact = 0.15 if len(q_lower) > 12 and q_lower in body else 0.0

            raw = 0.62 * coverage + 0.18 * phrase + 0.20 * heading_hit + exact
            out.append(max(0.0, min(1.0, raw)))
        return out


def resolve(spec: str, idf: dict[str, float] | None = None, quiet: bool = False) -> Reranker:
    spec = (spec or "auto").lower()
    if spec in ("lexical", "none"):
        return LexicalReranker(idf)
    if spec in ("cross-encoder", "ce", "neural"):
        return CrossEncoderReranker()
    try:
        return CrossEncoderReranker()
    except Exception as exc:
        if not quiet:
            print(f"[rerank] falling back to lexical ({type(exc).__name__}: {exc})")
        return LexicalReranker(idf)
