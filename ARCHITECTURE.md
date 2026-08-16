# Architecture

This document explains why the system is built the way it is, including the
options that were rejected. The implementation is in `src/telcorag/`.

---

## 1. What "near-zero hallucination" has to mean

"Do not hallucinate" is not implementable as stated. It has to be decomposed
into properties that can be enforced and measured. This system targets four:

| Property | Meaning | Enforced by |
|---|---|---|
| **Attribution** | Every statement names the clause it came from | Generation format + `verifier.parse_claims` |
| **Grounding** | Every value in a statement appears in that clause | `verifier.hard_tokens` |
| **Abstention** | Refuses when the corpus cannot answer | `guard.assess` |
| **Traceability** | A reader can check any claim in one hop | Clause-anchored chunks, source panel, archive links |

The fourth matters more than it looks. In a telecom setting the assistant is
not the authority — TS 24.501 is. A system that lets an engineer verify a claim
in five seconds is more useful than one that is slightly more accurate but
opaque, because the engineer is accountable for the answer, not the tool.

A design consequence follows immediately: **abstention is a feature, not a
failure.** The system is tuned to refuse rather than guess, and the evaluation
scores refusal quality explicitly rather than treating unanswered questions as
losses.

---

## 2. Why standards documents break naive RAG

The default recipe — dump text, split every 512 tokens, embed, top-k, stuff into
a prompt — fails on 3GPP material in five specific ways. Each one drove a
design decision.

**2.1 Fixed-size splitting severs conditions from consequences.**
Normative text reads "if A, the UE shall B; otherwise the UE shall C." Cut that
across a window boundary and retrieval can return the consequence without its
condition. The model then states an unconditional rule that the spec never
made. *Response:* chunk on clause boundaries (§4).

**2.2 Headings are meaningless without ancestry.**
TS 24.501 contains dozens of clauses titled "General". Embedded alone, they are
noise. *Response:* every chunk carries its full ancestor path, indexed and
embedded with the body (§4).

**2.3 Embeddings blur the tokens that carry the meaning.**
`T3510` and `T3512` are different timers; `TS 24.501` and `TS 24.301` are
different specifications; `5GMM` and `5GSM` are different protocols. A
bi-encoder places each pair a hair apart in cosine space. Retrieval by
similarity alone will confidently return the neighbour. *Response:* BM25 with a
tokenizer built for these identifiers, fused with the dense retriever (§6).

**2.4 The question and the corpus use different vocabulary.**
A user writes "Session Management Function"; the spec writes "SMF". *Response:*
an acronym lexicon mined from the corpus (§5).

**2.5 An answer that is well-cited can still be wrong.**
The most dangerous output is a fluent sentence citing a real clause with one
value changed. Prompting does not prevent this, and no amount of prompting lets
you *detect* it. *Response:* verify the generated text against its own cited
passages after generation (§8).

---

## 3. Corpus acquisition

`corpus/download.py` pulls from `3gpp.org/ftp/Specs/archive`. Two details are
domain-specific:

**Version decoding.** 3GPP encodes a version in the filename as three base-36
characters: `24501-k00.zip` is TS 24.501 v20.0.0. For Rel-8 onward the major
version equals the Release number, so the fetcher can offer `--release 18`
pinning. This matters for correctness, not convenience: **the answer to a 3GPP
question is release-dependent**, so release is captured as chunk metadata,
displayed in citations, and available as a retrieval filter.

**Part ordering.** Large specs ship as several `.docx` parts whose names embed
the clause range (`..._2_Main-Body_s05_s0504.docx`). The numeric prefix gives
reading order; sorting by filename alone would interleave the annexes wrongly.

Everything downloaded is the real published artefact, and every citation links
back to the exact archive it came from.

### Why parse OOXML directly

`corpus/ooxml.py` reads `word/document.xml` with the standard library instead of
using `python-docx`. Three requirements drove this:

- **Reading order.** Paragraphs and tables must be interleaved as they appear,
  because a table's meaning depends on the paragraph that introduces it.
- **Style access.** 3GPP encodes heading depth as the paragraph style
  (`Heading1`..`Heading9`). That *is* the clause tree; a plain text dump loses it.
- **Revision marks.** Several published specs still carry `w:del` runs. Their
  text is deleted content and must not reach the index — a retrieval hit on
  withdrawn normative text is a hallucination with a citation attached.

Tabs are preserved rather than collapsed, because 3GPP drafting uses them as the
only machine-readable delimiter in abbreviation lists (`5GMM<tab>5GS Mobility
Management`) and reference lists. That single decision is what makes §5 possible.

---

## 4. Clause-anchored chunking

The chunk boundary is the clause boundary. A clause is already the smallest
self-contained normative unit: it is what the specs cross-reference, what
engineers cite, and what drafting convention keeps coherent.

Only clauses exceeding `max_tokens` are split, and then on paragraph boundaries
with overlap. Oversized tables are banded on row boundaries **with the header
row repeated**, because a table band without its header is unreadable to both
the embedding model and the human checking the citation.

Every chunk is prefixed with a provenance header:

```
3GPP TS 24.501 v20.0.0 (Rel-20) — clause 5.4.1.3.2 Authentication initiation by the network
Path: 5 Elementary procedures for 5GS mobility management > 5.4 5GMM common procedures >
      5.4.1 Primary authentication and key agreement procedure > 5.4.1.3 5G AKA based ...
```

This header is embedded and indexed along with the body. It is what lets a
clause headed "General" be retrieved for a question about NAS security context
handling, and it is the same string the generator must cite.

Measured on the shipped corpus: 5,822 clauses → 11,590 chunks, median 261
tokens.

---

## 5. Acronym lexicon (mined, not written)

Every 3GPP spec carries an "Abbreviations" clause formatted as
`ACRONYM<tab>Expansion`, and TR 21.905 is the project-wide vocabulary. The build
harvests these into a bidirectional map — **689 acronyms** from the current
corpus, with no hand-maintained list to rot.

At query time, expansion runs in both directions: `5GMM` adds "5GS Mobility
Management", and "Session Management Function" adds `SMF`. Expansion terms enter
BM25 at a reduced weight (0.45) so they improve recall without letting a generic
expansion outrank an exact identifier match.

Mining beats hardcoding here for a reason worth stating: the lexicon is
*authoritative* (it is the specs' own definitions), it stays correct when the
corpus is re-fetched at a new release, and it captures genuine ambiguity — `AMF`
resolves to both "Access and Mobility Management Function" and "Authentication
Management Field", which a hand-written list would probably have flattened.

---

## 6. Retrieval

```
query
  ├─ glossary expansion        vocabulary gap
  ├─ structural parsing        "TS 24.501 clause 5.4.1" is an address
  ├─ BM25            ─┐        exact identifiers
  ├─ dense (bi-enc)  ─┴─ RRF   paraphrase
  ├─ sibling expansion         split clauses compete on their right part
  └─ cross-encoder rerank      calibrated ordering
```

**BM25 tokenizer.** The single highest-leverage detail in the retrieval stack. A
stock `\w+` pattern destroys every identifier in the corpus: `24.501` → `24`,
`501`; `5G-GUTI` → `5g`, `guti`. The tokenizer keeps dotted and hyphenated
identifiers whole *and additionally* emits their parts, preserving exact-match
precision without losing partial recall.

**Fusion by RRF, not weighted sum.** BM25 scores are unbounded and
corpus-dependent; cosine similarities are bounded and model-dependent. Combining
them by weighted sum needs a normalisation constant that silently drifts as the
corpus grows or the embedding model changes. RRF discards magnitudes and fuses
on rank position, so it stays stable across both.

**Structural shortcuts.** "What does TS 24.501 clause 5.4.1 say?" is a lookup,
not a semantic search. Named specs boost their chunks; a named clause is pinned
directly into the candidate set so it cannot be lost to fuzzy matching.

**Reranking exists for calibration, not just ordering.** The abstention gate
thresholds on the top score, so that score has to *mean* something. A
cross-encoder reads query and passage jointly and separates "answers the
question" from "shares vocabulary with the question" far more sharply than
cosine does. Its sigmoid output is what the gate consumes.

---

## 7. The abstention gate

Runs **before** generation, in two layers.

### 7.1 Premise checks (categorical)

These run first, because they catch a failure the score-based signals
*structurally cannot*.

- **Unindexed specification.** The question names `TS 36.331`, which is not in
  the corpus. Refuse, naming it.
- **Unknown entity.** An *entity-shaped* token in the question — one carrying a
  digit or hyphen, or capitalised — appears nowhere in the corpus vocabulary.
  Refuse, naming it.

The motivating measurement: questions about timer `T9999` and about `TS 36.331`
reranked between **0.967 and 0.999** — higher than several legitimate questions.
This is not a tuning failure. Retrieval returns its best six passages for any
input, and a question about a nonexistent timer retrieves genuine timer clauses
that a cross-encoder correctly judges to be about timers. No threshold on that
score can separate them. Checking the *premise* does, cheaply.

**Why "entity-shaped" and not "any unknown word".** The first version flagged
any unknown token of four or more letters, and the golden set reported a zero
false-positive rate for it. That measurement was misleading, because the golden
set is written in domain language. Probing with jargon-free phrasing refused
legitimate questions for containing **"prove"** and **"disagree"** — formal
specification prose simply has no conversational verbs, so absence from this
corpus reports its *register* as much as the question's validity. Narrowing to
entity-shaped tokens keeps every genuine catch and drops the false ones.

The cost is real and recorded in EVALUATION.md: a false premise made of ordinary
lowercase words ("which IE carries the subscriber's *blood type*") is no longer
caught here and falls to the generation layer.

The check is suppressed below a minimum vocabulary size
(`MIN_VOCAB_FOR_PREMISE`), because absence of evidence is only evidence of
absence when the corpus is comprehensive.

### 7.2 Score signals (graded)

- `top_score` — best reranked passage; below threshold, nothing good was found
- `support` — mean score of hits 2–3; low means one lucky match, no corroboration
- `overlap` — IDF-weighted share of query terms present in the top passages

Independence is the point: an out-of-domain question can score mediocre-but-not-
terrible on the reranker while its vocabulary is plainly absent, and vice versa.

### 7.3 And two more chances to refuse

The model may emit `INSUFFICIENT_CONTEXT`, and if verification rejects every
claim the answer is withdrawn rather than shown degraded. Measured behaviour of
the whole chain is in EVALUATION.md, including the one adversarial question that
still gets through and why only the generation layer can catch it.

---

## 8. Verification (the part that does the real work)

Prompting reduces fabrication; it cannot detect it, and it produces no number
you can report. So the generated answer is treated as a set of claims to be
checked against the passages each one cites.

**Generation format.** One claim per line, each ending with its source markers
(`[S1][S4]`). This makes attribution mechanical — the verifier never has to
guess which passage a sentence came from — and it is far more robust to parse
than JSON from a mid-size model.

**Check 1 — hard-token grounding (deterministic, no model).**
Extract from each claim every number, timer (`T3510`), spec reference
(`TS 24.501`), hex literal, and uppercase identifier or message name
(`REGISTRATION REQUEST`). Require each to appear verbatim in a passage that the
claim itself cites.

This targets the failure that actually matters. The dangerous output is not
invented prose, it is a correct-looking sentence with `T3512` where the spec says
`T3510`, or `30 s` where it says `15 s`. Because these tokens must match
literally, that class of error is caught **with certainty rather than
probability** — and it costs one pass over a string, no model call.

Unit wording is deliberately tolerated: a claim saying "15 seconds" against a
source saying "15 s" passes, because the *value* is what is checked.

**Check 2 — IDF-weighted lexical support.**
Guards the opposite failure: a claim whose individual tokens all appear in the
source but whose substance does not.

**Check 3 (optional) — LLM entailment.**
Catches correctly-copied tokens assembled into a false statement. Off by default:
it costs a round-trip and, unlike checks 1 and 2, it is itself a model and
therefore fallible. Available via `--entailment`.

Claims failing verification are removed from the answer and shown separately in
the UI with the reason. **Surfacing them is intentional** — it demonstrates the
guard working, and it tells the reader what the model tried to assert. If no
claim survives, the system abstains.

---

## 9. Degradation path

The system runs in three tiers, and each tier is honest about what it gives up:

| Tier | Embeddings | Rerank | Generation | Needs |
|---|---|---|---|---|
| Full | bge-small bi-encoder | cross-encoder | LLM, verified | model download + API key |
| No key | bge-small | cross-encoder | **extractive** | model download |
| Offline | LSA (TF-IDF+SVD) | lexical | extractive | nothing |

Extractive mode quotes retrieved clauses verbatim. It **cannot hallucinate by
construction**, which makes it both a usable fallback and the zero-fabrication
baseline the LLM path is measured against. A reviewer with no API key still sees
the full retrieval and verification stack work end to end.

---

## 10. Alternatives considered and rejected

**Fine-tuning on 3GPP text.** Moves knowledge into weights, where it cannot be
cited, updated, or verified — the opposite of the requirement. Retrieval keeps
the specification as the authority.

**Vector database (FAISS/Chroma/pgvector).** At 11k chunks, brute-force cosine
over a NumPy array is sub-millisecond and exact. A vector DB would add a
dependency, an approximation, and a service to run, for no gain at this scale.
The rewrite point is roughly 10⁶ chunks; `Index.dense_search` is the single
function to change.

**Dense-only retrieval.** Rejected on the identifier-collision argument (§2.3),
and quantified: on identifier-bearing queries dense retrieval scores **top-1 of
0.00** and misses all four timer lookups (`T3512`, `T3560`, `T3570`, `T3580`)
entirely, while BM25 finds them. On natural-language questions the ordering
reverses. Hybrid wins because the two retrievers fail on disjoint query types —
see EVALUATION.md for both tables.

**GraphRAG / knowledge graph over entities.** Genuinely attractive for 3GPP,
whose cross-references form a real graph. Rejected on cost/benefit for this
scope: the extraction step becomes a hallucination source of its own, and the
clause hierarchy already supplies most of the structure a graph would add.

**Semantic (embedding-based) chunking.** Solves a problem this corpus does not
have. 3GPP clause numbering is a better segmentation than any similarity
heuristic would infer, and it is authored by the standards body.

**JSON-structured generation.** More brittle than one-claim-per-line under
mid-size models, for the same verifiability.

---

## 11. Known limitations

- **Figures are not read.** 3GPP message flow diagrams are embedded images;
  their captions are indexed but their content is invisible. Questions answered
  primarily by a figure will under-retrieve. A VLM captioning pass at ingest is
  the natural fix.
- **Cross-reference following is single-hop.** "See clause 5.4.1.3" is retrieved
  as text, not traversed. Multi-hop questions can retrieve a pointer instead of
  the target.
- **Eight specs, not the full corpus.** Adding specs is a config line, but
  questions outside these eight correctly abstain rather than answer.
- **Hard-token checking can over-reject.** A legitimate paraphrase converting
  units ("0.5 s" for "500 ms") is flagged. This is a deliberate bias toward
  false refusal over false assertion, and the eval measures the cost.
- **Release conflicts are surfaced, not resolved.** When indexed specs sit at
  different releases the system reports both positions and cites each; it does
  not decide which the user meant.
- **A false premise built entirely from in-vocabulary words survives the gate.**
  "The 5GMM quantum handover procedure" passes every pre-generation check
  because "quantum" does occur in TS 33.501. Only the generation layer can
  decline it, so extractive mode answers it from real but irrelevant clauses.
- **Heavy paraphrase degrades the reranker.** `ms-marco-MiniLM` is trained on
  web-search queries; on telecom paraphrase carrying no domain vocabulary it
  scored a correct rank-2 clause at 0.04, and the system refused. A reranker
  fine-tuned on telecom query/clause pairs is the obvious next step.
