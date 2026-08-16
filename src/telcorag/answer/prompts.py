"""Prompt construction.

Three constraints do most of the work here. Requiring one claim per line with
trailing markers makes the output *mechanically* checkable -- the verifier never
has to guess which passage a sentence came from. Forbidding prior knowledge
matters more than usual in this domain, because a strong model has read a great
deal of telecom material and will otherwise answer a 3GPP question from memory
and cite a passage that merely looks related. Requiring verbatim copying of
values attacks the specific failure the verifier is built to catch.
"""

from __future__ import annotations

SYSTEM = """You are a 3GPP standards assistant. You answer strictly from the numbered SOURCE passages supplied with the question.

Rules:
1. Use only what the SOURCES state. Do not use prior knowledge of telecom standards, even where you are confident it is correct. If a source contradicts your prior belief, follow the source.
2. Put each factual statement on its own line and end that line with the markers of the sources it came from, for example [S2] or [S1][S4].
3. Reproduce identifiers, timer names, numeric values, message names, IE names and clause references exactly as written in the sources. Never round, convert units, or infer a value that is not printed.
4. If the sources do not answer the question, reply with exactly: INSUFFICIENT_CONTEXT
5. Do not write an introduction, a summary, a closing remark, or any line without a source marker.
6. If sources disagree (different specifications or releases), state each position on its own line and cite each separately.
7. Prefer the wording of the specification over paraphrase. Where the spec says "shall", keep "shall"."""

ENTAILMENT_SYSTEM = """You check whether a claim is supported by a passage from a 3GPP specification.

Reply with one line per claim in the form:
<claim number>: SUPPORTED
<claim number>: NOT_SUPPORTED

A claim is SUPPORTED only if the passage states it or directly implies it. If the passage is merely on the same topic, answer NOT_SUPPORTED. Do not use outside knowledge. Output nothing else."""


def build_context(hits, max_chars: int = 14000) -> tuple[str, list]:
    """Render retrieved hits as numbered sources, trimming to a character budget."""
    blocks: list[str] = []
    used = []
    total = 0
    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        block = f"[S{i}] {chunk.header}\n{chunk.text}"
        if total + len(block) > max_chars and used:
            break
        blocks.append(block)
        used.append(hit)
        total += len(block)
    return "\n\n---\n\n".join(blocks), used


def build_question(question: str, context: str) -> str:
    return (
        f"SOURCES:\n\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "Answer using the rules you were given. One claim per line, each ending with its source markers."
    )


def build_entailment(claims: list[str], sources: list[str]) -> str:
    parts = ["PASSAGES:"]
    for i, src in enumerate(sources, start=1):
        parts.append(f"[S{i}] {src}")
    parts.append("\nCLAIMS:")
    for i, claim in enumerate(claims, start=1):
        parts.append(f"{i}. {claim}")
    return "\n".join(parts)
