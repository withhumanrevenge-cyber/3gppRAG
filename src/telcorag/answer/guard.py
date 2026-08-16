"""The abstention gate.

"Near-zero hallucination" is mostly a decision about when *not* to answer. A RAG
system that always answers will always eventually invent, because retrieval on
an out-of-scope question still returns its best six passages and a helpful model
will still write something from them.

The gate runs before generation and rejects on three independent signals, so a
question can fail for being off-topic even when reranker scores look reasonable,
and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Guard as GuardConfig
from ..retrieval.pipeline import RetrievalResult

REFUSAL = (
    "I cannot answer this from the indexed 3GPP specifications. "
    "No clause in the corpus supports an answer with sufficient confidence."
)


@dataclass
class Gate:
    answerable: bool
    reason: str
    confidence: float
    signals: dict = field(default_factory=dict)


def assess(result: RetrievalResult, cfg: GuardConfig) -> Gate:
    if not result.hits:
        return Gate(False, "retrieval returned no candidates", 0.0, {})

    top = result.top_score
    support = result.support
    overlap = max((h.overlap for h in result.hits[:3]), default=0.0)
    signals = {
        "top_score": round(top, 4),
        "support": round(support, 4),
        "overlap": round(overlap, 4),
        "unknown_terms": result.unknown_terms,
        "unindexed_specs": result.unindexed_specs,
    }
    confidence = round(0.60 * top + 0.25 * support + 0.15 * overlap, 4)

    # Premise checks run first. They catch the failure the score-based signals
    # structurally cannot: a question built on something the specs never mention
    # still retrieves six topically-adjacent clauses that rerank near 1.0.
    if cfg.premise_check and result.unindexed_specs:
        named = ", ".join(f"TS {s}" for s in result.unindexed_specs)
        return Gate(False, f"{named} is not in the indexed corpus", confidence, signals)

    if cfg.premise_check and result.unknown_terms:
        terms = ", ".join(repr(t) for t in result.unknown_terms[:4])
        return Gate(
            False,
            f"the question relies on {terms}, which does not appear anywhere in the indexed specifications",
            confidence,
            signals,
        )

    if top < cfg.min_top_score:
        return Gate(
            False,
            f"best clause scored {top:.2f}, below the {cfg.min_top_score:.2f} answerability threshold",
            confidence,
            signals,
        )
    if overlap < cfg.min_lexical_overlap:
        return Gate(
            False,
            f"question vocabulary barely occurs in the corpus (overlap {overlap:.2f})",
            confidence,
            signals,
        )
    if support < cfg.min_support and top < cfg.min_top_score * 1.6:
        return Gate(
            False,
            f"only one weak passage supports this (corroboration {support:.2f})",
            confidence,
            signals,
        )
    return Gate(True, "sufficient support", confidence, signals)
