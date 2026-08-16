from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict

from ..config import Settings, settings as default_settings
from ..corpus.download import archive_url
from ..index.bm25 import tokenize
from ..index.store import Index
from ..retrieval.pipeline import Retrieved, Retriever
from . import guard, prompts
from .llm import LLM, LLMError, Message, resolve as resolve_llm
from .verifier import ClaimVerdict, Verification, Verifier

INSUFFICIENT = "INSUFFICIENT_CONTEXT"
SENTENCE_RE = re.compile(r"(?<=[.;])\s+(?=[A-Z0-9(\[])")


@dataclass
class SourceRef:
    n: int
    spec_id: str
    version: str
    release: int
    clause: str
    heading: str
    breadcrumb: str
    citation: str
    score: float
    text: str
    url: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Answer:
    question: str
    text: str
    mode: str
    abstained: bool
    reason: str
    confidence: float
    groundedness: float
    sources: list[SourceRef] = field(default_factory=list)
    claims: list[ClaimVerdict] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "mode": self.mode,
            "abstained": self.abstained,
            "reason": self.reason,
            "confidence": self.confidence,
            "groundedness": self.groundedness,
            "sources": [s.as_dict() for s in self.sources],
            "claims": [
                {
                    "text": c.text,
                    "citations": c.citations,
                    "supported": c.supported,
                    "support": round(c.support, 3),
                    "ungrounded_tokens": c.ungrounded_tokens,
                    "entailment": c.entailment,
                    "reason": c.reason,
                }
                for c in self.claims
            ],
            "diagnostics": self.diagnostics,
        }


def _source_refs(hits: list[Retrieved]) -> list[SourceRef]:
    refs: list[SourceRef] = []
    for i, hit in enumerate(hits, start=1):
        c = hit.chunk
        try:
            url = archive_url(c.spec_id, c.version)
        except ValueError:
            url = ""
        refs.append(
            SourceRef(
                n=i,
                spec_id=c.spec_id,
                version=c.version,
                release=c.release,
                clause=c.clause,
                heading=c.heading,
                breadcrumb=c.breadcrumb,
                citation=c.citation,
                score=round(hit.score, 4),
                text=c.text,
                url=url,
            )
        )
    return refs


class AnswerEngine:
    def __init__(
        self,
        index: Index,
        cfg: Settings | None = None,
        llm: LLM | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.index = index
        self.settings = cfg or default_settings
        self.retriever = retriever or Retriever(index, self.settings)
        self.llm = llm or resolve_llm(self.settings)
        self.verifier = Verifier(self.settings.guard.min_sentence_support, self.retriever.idf)

    @staticmethod
    def _quotable(text: str) -> list[str]:
        """Split clause text into spans short enough to quote as single claims.

        Paragraph breaks are split first: 3GPP renders enumerated conditions as
        separate paragraphs, and sentence punctuation alone leaves the whole
        list glued into one unreadable span. Spans outside the length band are
        dropped as either fragments or un-quotable blocks.
        """
        spans: list[str] = []
        for block in text.split("\n"):
            block = block.strip()
            if not block or block.startswith("|"):
                continue
            for span in SENTENCE_RE.split(block):
                span = re.sub(r"\s+", " ", span).strip(" -\t")
                if 40 <= len(span) <= 400:
                    spans.append(span)
        return spans

    def _extractive(self, question: str, hits: list[Retrieved], limit: int = 3) -> str:
        terms = list(dict.fromkeys(tokenize(question)))
        idf = self.retriever.idf
        lines: list[str] = []

        for n, hit in enumerate(hits[:limit], start=1):
            sentences = self._quotable(hit.chunk.text)
            if not sentences:
                continue
            scored = sorted(
                sentences,
                key=lambda s: sum(idf.get(t, 6.0) for t in terms if t in s.lower()),
                reverse=True,
            )
            for sentence in scored[:2]:
                # One claim per line is the contract the verifier parses against.
                # Clause bodies carry hard newlines (3GPP numbered lists are
                # separate paragraphs), so an extracted span must be flattened or
                # it arrives as several lines with only the last one cited.
                flat = re.sub(r"\s+", " ", sentence).strip()
                if flat:
                    lines.append(f"{flat} [S{n}]")
        return "\n".join(lines)

    def _entail(self, claims: list[ClaimVerdict], sources: list[str]) -> None:
        if not claims or not self.llm.available:
            return
        try:
            reply = self.llm.complete(
                [
                    Message("system", prompts.ENTAILMENT_SYSTEM),
                    Message("user", prompts.build_entailment([c.text for c in claims], sources)),
                ],
                temperature=0.0,
                max_tokens=400,
            )
        except LLMError:
            return

        verdicts = {}
        for line in reply.splitlines():
            m = re.match(r"\s*(\d+)\s*[:.)-]\s*(SUPPORTED|NOT_SUPPORTED)", line.strip(), re.I)
            if m:
                verdicts[int(m.group(1))] = m.group(2).upper()

        for i, claim in enumerate(claims, start=1):
            verdict = verdicts.get(i)
            if not verdict:
                continue
            claim.entailment = verdict
            if verdict == "NOT_SUPPORTED":
                claim.supported = False
                claim.reason = "LLM entailment check: not supported by cited clause"

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        spec_filter: set[str] | None = None,
        entailment_check: bool = False,
    ) -> Answer:
        started = time.time()
        result = self.retriever.retrieve(question, top_k, spec_filter)
        gate = guard.assess(result, self.settings.guard)
        diagnostics = {
            **result.diagnostics,
            "gate": gate.signals,
            "expansions": result.expansions,
            "llm": self.llm.name,
        }

        if not gate.answerable:
            diagnostics["elapsed_s"] = round(time.time() - started, 3)
            diagnostics["retrieved"] = [f"{h.chunk.spec_id}:{h.chunk.clause}" for h in result.hits]
            return Answer(
                question=question,
                text=guard.REFUSAL,
                mode="abstained",
                abstained=True,
                reason=gate.reason,
                confidence=gate.confidence,
                groundedness=0.0,
                sources=_source_refs(result.hits[:3]),
                diagnostics=diagnostics,
            )

        context, used = prompts.build_context(result.hits)
        refs = _source_refs(used)
        source_texts = [f"{h.chunk.header}\n{h.chunk.text}" for h in used]

        # Recorded so evaluation can separate retrieval quality from citation
        # behaviour: `sources` is filtered to what the answer actually cited,
        # which would otherwise make both look like the same measurement.
        diagnostics["retrieved"] = [f"{h.chunk.spec_id}:{h.chunk.clause}" for h in result.hits]
        diagnostics["sources_offered"] = len(used)

        mode = "extractive"
        raw = ""
        if self.llm.available:
            try:
                raw = self.llm.complete(
                    [Message("system", prompts.SYSTEM), Message("user", prompts.build_question(question, context))],
                    temperature=self.settings.llm_temperature,
                    max_tokens=900,
                )
                mode = "generated"
            except LLMError as exc:
                diagnostics["llm_error"] = str(exc)
                raw = ""

        if mode == "generated" and INSUFFICIENT in raw.upper():
            diagnostics["elapsed_s"] = round(time.time() - started, 3)
            return Answer(
                question=question,
                text=guard.REFUSAL,
                mode="abstained",
                abstained=True,
                reason="model reported the retrieved clauses do not contain the answer",
                confidence=gate.confidence,
                groundedness=0.0,
                sources=refs,
                diagnostics=diagnostics,
            )

        if not raw.strip():
            raw = self._extractive(question, used)
            mode = "extractive"

        verification: Verification = self.verifier.verify(raw, source_texts)
        if entailment_check and mode == "generated":
            self._entail(verification.claims, source_texts)
            verification.kept = [c for c in verification.claims if c.supported]
            verification.dropped = [c for c in verification.claims if not c.supported]
            verification.groundedness = (
                len(verification.kept) / len(verification.claims) if verification.claims else 0.0
            )

        shown = verification.kept if self.settings.guard.drop_unverified else verification.claims
        if not shown:
            diagnostics["elapsed_s"] = round(time.time() - started, 3)
            diagnostics["dropped_claims"] = len(verification.dropped)
            return Answer(
                question=question,
                text=guard.REFUSAL,
                mode="abstained",
                abstained=True,
                reason="every generated claim failed grounding verification against its cited clause",
                confidence=gate.confidence,
                groundedness=0.0,
                sources=refs,
                claims=verification.claims,
                diagnostics=diagnostics,
            )

        cited = sorted({n for c in shown for n in c.citations})
        text = "\n".join(f"{c.text} " + "".join(f"[S{n}]" for n in c.citations) for c in shown)
        diagnostics["elapsed_s"] = round(time.time() - started, 3)
        diagnostics["claims_total"] = len(verification.claims)
        diagnostics["claims_dropped"] = len(verification.dropped)

        return Answer(
            question=question,
            text=text,
            mode=mode,
            abstained=False,
            reason=gate.reason,
            confidence=gate.confidence,
            groundedness=round(verification.groundedness, 4),
            sources=[r for r in refs if r.n in cited] or refs,
            claims=verification.claims,
            diagnostics=diagnostics,
        )
