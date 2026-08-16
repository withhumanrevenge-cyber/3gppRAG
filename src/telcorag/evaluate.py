"""Evaluation harness.

Three things are measured, because "minimal hallucination" is not one property:

* retrieval  -- can the right clause be found at all (recall@k, MRR). If it
                cannot, nothing downstream can be grounded.
* abstention -- does the system refuse when it should. Scored on a deliberately
                adversarial half of the golden set: out-of-domain questions,
                questions about specs that are not indexed, and questions with a
                false premise (a timer that does not exist).
* grounding  -- of the claims it does emit, how many survive verification
                against the clause they cite, and do required values appear.

The ablation re-runs retrieval only, with the same golden set, across
BM25-only / dense-only / hybrid / hybrid+rerank, which is what justifies the
architecture rather than asserting it.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import ROOT, settings

DEFAULT_GOLDEN = ROOT / "eval" / "golden.json"


@dataclass
class ItemResult:
    id: str
    question: str
    expect_answerable: bool
    abstained: bool
    correct_decision: bool
    retrieved_clauses: list[str] = field(default_factory=list)
    hit_rank: int | None = None
    recall: bool = False
    groundedness: float = 0.0
    claims: int = 0
    dropped: int = 0
    missing_required: list[str] = field(default_factory=list)
    forbidden_present: list[str] = field(default_factory=list)
    citations_valid: bool = True
    confidence: float = 0.0
    reason: str = ""
    elapsed_s: float = 0.0
    has_ground_truth: bool = False


def clause_matches(actual: str, expected: str) -> bool:
    a, e = actual.strip(), expected.strip()
    if not a or not e:
        return False
    return a == e or a.startswith(f"{e}.") or e.startswith(f"{a}.")


def load_golden(path: Path | None = None) -> list[dict]:
    path = path or DEFAULT_GOLDEN
    if not path.exists():
        raise FileNotFoundError(f"golden set not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def rank_of(pairs: list[tuple[str, str]], expected_clauses: list[str], expected_specs: list[str]) -> int | None:
    """First 1-based position in ``(spec_id, clause)`` pairs matching expectations."""
    for i, (spec_id, clause) in enumerate(pairs, start=1):
        if expected_specs and spec_id not in expected_specs:
            continue
        if not expected_clauses:
            return i
        if any(clause_matches(clause, e) for e in expected_clauses):
            return i
    return None


def ablation(retriever, golden: list[dict], k: int = 6) -> dict:
    modes = [
        ("bm25", {"mode": "bm25", "rerank": False}),
        ("dense", {"mode": "dense", "rerank": False}),
        ("hybrid", {"mode": "hybrid", "rerank": False}),
        ("hybrid+rerank", {"mode": "hybrid", "rerank": True}),
    ]
    answerable = [q for q in golden if q.get("expect_answerable", True) and q.get("expect_clauses")]
    out: dict[str, dict] = {}

    for label, kwargs in modes:
        ranks: list[int | None] = []
        started = time.time()
        for item in answerable:
            result = retriever.retrieve(item["question"], top_k=k, **kwargs)
            pairs = [(h.chunk.spec_id, h.chunk.clause) for h in result.hits]
            ranks.append(rank_of(pairs, item.get("expect_clauses", []), item.get("expect_specs", [])))
        found = [r for r in ranks if r]
        out[label] = {
            "recall_at_k": round(len(found) / len(answerable), 4) if answerable else 0.0,
            "mrr": round(sum(1 / r for r in found) / len(answerable), 4) if answerable else 0.0,
            "top1": round(sum(1 for r in found if r == 1) / len(answerable), 4) if answerable else 0.0,
            "median_rank": statistics.median(found) if found else None,
            "seconds": round(time.time() - started, 1),
        }
    out["_n_questions"] = len(answerable)
    out["_k"] = k
    return out


def run(golden_path: Path | None = None, limit: int | None = None, entailment: bool = False, log=print) -> dict:
    from .answer.generator import AnswerEngine
    from .index.store import Index

    golden = load_golden(golden_path)
    if limit:
        golden = golden[:limit]

    index = Index.load(settings.index_dir, settings.lsa_dims)
    engine = AnswerEngine(index, settings)
    log(f"Evaluating {len(golden)} questions — {len(index)} chunks, llm={engine.llm.name}\n")

    items: list[ItemResult] = []
    for item in golden:
        answer = engine.ask(item["question"], entailment_check=entailment)
        expect_answerable = bool(item.get("expect_answerable", True))
        expect_clauses = item.get("expect_clauses", [])

        # Rank is measured over everything retrieved, not over answer.sources --
        # the latter is filtered to what the answer cited, so using it would let
        # selective citation masquerade as better retrieval.
        clauses = answer.diagnostics.get("retrieved") or [f"{s.spec_id}:{s.clause}" for s in answer.sources]
        pairs = [(c.split(":", 1)[0], c.split(":", 1)[1]) for c in clauses if ":" in c]
        rank = rank_of(pairs, expect_clauses, item.get("expect_specs", []))

        body = answer.text.lower()
        missing = [t for t in item.get("must_include", []) if t.lower() not in body] if not answer.abstained else []
        forbidden = [t for t in item.get("must_not_include", []) if t.lower() in body]
        # Bound is the number of sources shown to the model, not the filtered
        # list returned with the answer.
        offered = answer.diagnostics.get("sources_offered") or len(answer.sources)
        valid_cites = all(1 <= n <= offered for c in answer.claims for n in c.citations) if answer.claims else True

        result = ItemResult(
            id=item.get("id", item["question"][:40]),
            question=item["question"],
            expect_answerable=expect_answerable,
            abstained=answer.abstained,
            correct_decision=(answer.abstained != expect_answerable),
            retrieved_clauses=clauses,
            hit_rank=rank,
            recall=rank is not None,
            groundedness=answer.groundedness,
            claims=len(answer.claims),
            dropped=sum(1 for c in answer.claims if not c.supported),
            missing_required=missing,
            forbidden_present=forbidden,
            citations_valid=valid_cites,
            confidence=answer.confidence,
            reason=answer.reason,
            elapsed_s=answer.diagnostics.get("elapsed_s", 0.0),
        )
        result.has_ground_truth = bool(expect_clauses)
        items.append(result)

        mark = "ok " if result.correct_decision else "FAIL"
        state = "abstained" if answer.abstained else f"answered g={answer.groundedness:.0%}"
        log(f"  [{mark}] {result.id:<28} {state:<22} rank={rank}")

    positives = [i for i in items if i.expect_answerable]
    negatives = [i for i in items if not i.expect_answerable]
    with_clauses = [i for i in positives if i.has_ground_truth]
    answered = [i for i in positives if not i.abstained]

    summary = {
        "questions": len(items),
        "answerable": len(positives),
        "adversarial": len(negatives),
        "answer_rate_on_answerable": round(len(answered) / len(positives), 4) if positives else 0.0,
        "correct_abstention_rate": round(sum(1 for i in negatives if i.abstained) / len(negatives), 4) if negatives else None,
        "false_answer_on_adversarial": sum(1 for i in negatives if not i.abstained),
        "retrieval_recall": round(sum(1 for i in with_clauses if i.recall) / len(with_clauses), 4) if with_clauses else None,
        "mrr": round(sum(1 / i.hit_rank for i in with_clauses if i.hit_rank) / len(with_clauses), 4) if with_clauses else None,
        "mean_groundedness": round(statistics.mean([i.groundedness for i in answered]), 4) if answered else 0.0,
        "claims_emitted": sum(i.claims for i in answered),
        "claims_rejected": sum(i.dropped for i in answered),
        "required_value_misses": sum(1 for i in answered if i.missing_required),
        "forbidden_value_hits": sum(1 for i in items if i.forbidden_present),
        "citation_validity": round(sum(1 for i in answered if i.citations_valid) / len(answered), 4) if answered else 1.0,
        "mean_latency_s": round(statistics.mean([i.elapsed_s for i in items]), 3) if items else 0.0,
    }

    log("\nRetrieval ablation")
    abl = ablation(engine.retriever, golden)
    for label in ("bm25", "dense", "hybrid", "hybrid+rerank"):
        row = abl[label]
        log(f"  {label:<15} recall@6={row['recall_at_k']:.2f}  mrr={row['mrr']:.3f}  top1={row['top1']:.2f}")

    log("\nSummary")
    for key, value in summary.items():
        log(f"  {key:<32} {value}")

    return {
        "config": {
            "embeddings": index.backend.name,
            "reranker": engine.retriever.reranker.name,
            "llm": engine.llm.name,
            "chunks": len(index),
            "guard": {
                "min_top_score": settings.guard.min_top_score,
                "min_support": settings.guard.min_support,
                "min_lexical_overlap": settings.guard.min_lexical_overlap,
                "min_sentence_support": settings.guard.min_sentence_support,
            },
        },
        "summary": summary,
        "retrieval_ablation": abl,
        "items": [vars(i) for i in items],
    }
