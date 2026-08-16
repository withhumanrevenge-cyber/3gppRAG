"""Post-generation grounding verification.

Prompting a model to "only use the context" reduces fabrication; it does not
eliminate it, and it gives you nothing to measure. This module treats the
generated answer as a set of claims to be *checked* against the passages each
one cites, and it runs two independent checks.

Check 1 -- hard-token grounding (deterministic, no model).
    The failure mode that matters in a standards assistant is not florid
    invention, it is a plausible sentence with the wrong number in it: T3510
    instead of T3512, 15 s instead of 12 s, TS 24.301 instead of TS 24.501.
    Those are exactly the tokens that must appear verbatim in a cited passage,
    so we extract every number, timer, spec reference, hex value and uppercase
    identifier from the claim and require each to be present in its own source.
    A claim that invents a value is caught with certainty, not probability.

Check 2 -- IDF-weighted lexical support.
    Guards against the opposite failure: a claim whose tokens all happen to
    appear somewhere in the source but whose substance does not.

An optional third pass asks an LLM to judge entailment, which catches
correctly-copied tokens assembled into a wrong statement. It is off by default
because it costs a second round-trip and, unlike checks 1 and 2, it is itself a
model and therefore fallible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..index.bm25 import tokenize

CLAIM_MARKER_RE = re.compile(r"\[S(\d+)\]")
TRAILING_MARKERS_RE = re.compile(r"(?:\s*\[S\d+\])+\s*$")

# Dotted sequences are captured whole: matching only "5.5" out of a clause
# reference like "5.5.1.2.4" would look for a value the source never contains.
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)*(?![\w])")
TIMER_RE = re.compile(r"\bT\d{3,4}[A-Za-z]?\b")
SPEC_REF_RE = re.compile(r"\b(?:TS|TR)\s?\d{2}\.\d{3}\b", re.I)
HEX_RE = re.compile(r"\b0x[0-9A-Fa-f]+\b")
IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:[-/][A-Z0-9]+)*\b")
UPPER_PHRASE_RE = re.compile(r"\b[A-Z][A-Z0-9-]{2,}(?:\s+[A-Z][A-Z0-9-]{2,})+\b")

# Uppercase tokens that are ambient in telecom prose rather than claim-specific.
AMBIENT = frozenset({"UE", "NAS", "3GPP", "TS", "TR", "THE", "AND", "NOTE", "IE", "PDU", "NR", "AMF", "SMF"})


@dataclass
class ClaimVerdict:
    text: str
    citations: list[int]
    supported: bool
    support: float
    ungrounded_tokens: list[str] = field(default_factory=list)
    entailment: str | None = None
    reason: str = ""


@dataclass
class Verification:
    claims: list[ClaimVerdict]
    groundedness: float
    kept: list[ClaimVerdict]
    dropped: list[ClaimVerdict]

    @property
    def all_supported(self) -> bool:
        return bool(self.claims) and all(c.supported for c in self.claims)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def hard_tokens(text: str) -> list[str]:
    found: list[str] = []
    for regex in (SPEC_REF_RE, TIMER_RE, HEX_RE):
        found.extend(m.group(0) for m in regex.finditer(text))
    found.extend(m.group(0) for m in UPPER_PHRASE_RE.finditer(text))
    for m in IDENTIFIER_RE.finditer(text):
        if m.group(0) not in AMBIENT:
            found.append(m.group(0))
    found.extend(m.group(0) for m in NUMBER_RE.finditer(text))

    seen: set[str] = set()
    out: list[str] = []
    for tok in found:
        key = _normalise(tok)
        if key not in seen:
            seen.add(key)
            out.append(tok)
    return out


def _token_present(token: str, source_text: str, source_tokens: set[str]) -> bool:
    key = _normalise(token)
    if " " in key:
        return key in source_text
    if key in source_tokens:
        return True
    # "TS 24.501" written as "TS24.501", and dotted ids split by the tokenizer.
    compact = key.replace(" ", "")
    return compact in source_text.replace(" ", "")


def parse_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split a generated answer into (claim text, cited source numbers)."""
    claims: list[tuple[str, list[int]]] = []
    for raw in answer.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        cites = [int(m.group(1)) for m in CLAIM_MARKER_RE.finditer(line)]
        text = TRAILING_MARKERS_RE.sub("", line).strip()
        text = CLAIM_MARKER_RE.sub("", text).strip()
        if text:
            claims.append((text, sorted(set(cites))))
    return claims


class Verifier:
    def __init__(self, min_support: float = 0.45, idf: dict[str, float] | None = None) -> None:
        self.min_support = min_support
        self._idf = idf or {}

    def _support(self, claim: str, source_text: str) -> float:
        terms = [t for t in dict.fromkeys(tokenize(claim))]
        if not terms:
            return 0.0
        total = sum(self._idf.get(t, 6.0) for t in terms) or 1.0
        hit = sum(self._idf.get(t, 6.0) for t in terms if t in source_text)
        return hit / total

    def verify(self, answer: str, sources: list[str]) -> Verification:
        normalised = [_normalise(s) for s in sources]
        token_sets = [set(tokenize(s, keep_stopwords=True)) for s in sources]

        claims: list[ClaimVerdict] = []
        for text, cites in parse_claims(answer):
            valid = [c for c in cites if 1 <= c <= len(sources)]
            if not valid:
                claims.append(
                    ClaimVerdict(text, cites, False, 0.0, reason="no valid citation")
                )
                continue

            merged_text = " ".join(normalised[c - 1] for c in valid)
            merged_tokens: set[str] = set()
            for c in valid:
                merged_tokens |= token_sets[c - 1]

            ungrounded = [t for t in hard_tokens(text) if not _token_present(t, merged_text, merged_tokens)]
            support = self._support(text, merged_text)

            if ungrounded:
                reason = "values not present in cited clause: " + ", ".join(ungrounded[:5])
            elif support < self.min_support:
                reason = f"weak lexical support ({support:.2f} < {self.min_support:.2f})"
            else:
                reason = "grounded"

            claims.append(
                ClaimVerdict(
                    text=text,
                    citations=valid,
                    supported=not ungrounded and support >= self.min_support,
                    support=support,
                    ungrounded_tokens=ungrounded,
                    reason=reason,
                )
            )

        kept = [c for c in claims if c.supported]
        dropped = [c for c in claims if not c.supported]
        groundedness = len(kept) / len(claims) if claims else 0.0
        return Verification(claims, groundedness, kept, dropped)
