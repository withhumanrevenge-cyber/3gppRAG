# 3GPP Standards Assistant — Project Design

A retrieval-augmented question answering system over 3GPP telecom specifications,
designed so that every statement it makes is traceable to a clause, and so that
it refuses to answer when the indexed corpus cannot support one.

This document is self-contained: it states the problem, the design decisions and
the reasoning behind each, the alternatives rejected, the measured results, and
the known limitations.

---

## 1. Corpus

Eight specifications, fetched programmatically from the public 3GPP archive at
`3gpp.org/ftp/Specs/archive` — not a curated or hand-cleaned sample.

| Spec | Title | Release |
|---|---|---|
| TS 23.501 | System architecture for the 5G System | Rel-20 |
| TS 23.502 | Procedures for the 5G System | Rel-20 |
| TS 24.501 | NAS protocol for 5GS | Rel-20 |
| TS 33.501 | Security architecture and procedures for 5G | Rel-20 |
| TS 38.300 | NR and NG-RAN overall description | Rel-19 |
| TS 38.321 | NR Medium Access Control | Rel-19 |
| TS 38.331 | NR Radio Resource Control | Rel-19 |
| TR 21.905 | Vocabulary for 3GPP specifications | Rel-19 |

Scale: ~53 MB of source documents produce **5,822 clauses and 11,590 chunks**, a
50,227-term BM25 vocabulary, and 689 acronyms mined from the specifications
themselves.

---

## 2. What "near-zero hallucination" was decomposed into

"Do not hallucinate" is not directly implementable. It was broken into four
properties that can each be enforced and measured:

| Property | Meaning | Enforced by |
|---|---|---|
| Attribution | Every statement names the clause it came from | Generation contract + claim parser |
| Grounding | Every value in a statement appears in that clause | Deterministic hard-token check |
| Abstention | Refuses when the corpus cannot answer | Premise checks + score gate |
| Traceability | A reader can verify any claim in one hop | Clause-anchored chunks, source panel, archive links |

The fourth matters more than it appears. In a telecom setting the assistant is
not the authority — TS 24.501 is. A system that lets an engineer verify a claim
in five seconds is more useful than a marginally more accurate one that is
opaque, because the engineer remains accountable for the answer.

A design consequence follows: **abstention is a feature, not a failure.** The
system is tuned to refuse rather than guess, and the evaluation scores refusal
quality explicitly instead of treating unanswered questions as losses.

---

## 3. Why standards documents break naive RAG

The default recipe — dump text, split every 512 tokens, embed, top-k, stuff into
a prompt — fails on 3GPP material in five specific ways. Each drove a decision.

**3.1 Fixed-size splitting severs normative logic.** Specification text reads
"if A, the UE shall B; otherwise the UE shall C." Cut that across a window
boundary and retrieval can return the consequence without its condition; the
model then states an unconditional rule the spec never made.

**3.2 Headings are meaningless without ancestry.** TS 24.501 contains dozens of
clauses titled "General". Embedded alone they are noise.

**3.3 Embeddings blur the tokens that carry the meaning.** `T3510` and `T3512`
are different timers. `TS 24.501` and `TS 24.301` are different specifications.
`5GMM` and `5GSM` are different protocols. A bi-encoder places each pair a hair
apart in cosine space and will confidently return the neighbour.

**3.4 Question and corpus use different vocabulary.** A user writes "Session
Management Function"; the specification writes "SMF".

**3.5 A well-cited answer can still be wrong.** The most dangerous output is a
fluent sentence citing a real clause with one value changed. Prompting does not
prevent this, and gives you no way to *detect* it.

---

## 4. Architecture

```
             INGEST (offline)
  3gpp.org -> download -> OOXML parse -> clause tree -> chunking -> index
                                                                   |- BM25 postings
                                                                   |- dense vectors
                                                                   |- acronym lexicon

             QUERY (online)
  question -> glossary expansion
           -> premise check ------------------------> REFUSE (unknown entity / unindexed spec)
           -> BM25  ---|
           -> dense ---+-> RRF fusion -> sibling expansion -> cross-encoder rerank
           -> abstention gate ----------------------> REFUSE (weak / uncorroborated)
           -> grounded generation (one claim per line, cited)
           -> claim verification -------------------> REFUSE (nothing survives)
           -> answer + per-claim verdicts + sources
```

### 4.1 Corpus acquisition

3GPP encodes a version in the filename as three base-36 characters: `24501-k00.zip`
is TS 24.501 v20.0.0. For Rel-8 onward the major version equals the Release
number, so the fetcher supports pinning with `--release 18`. This is a
correctness feature, not a convenience: **the answer to a 3GPP question is
release-dependent**, so release is carried as chunk metadata, shown in every
citation, and available as a retrieval filter.

Large specifications ship as several `.docx` parts whose filenames embed the
clause range (`..._2_Main-Body_s05_s0504.docx`); the numeric prefix gives reading
order. Sorting by filename alone interleaves the annexes incorrectly.

### 4.2 Parsing OOXML directly

The reader parses `word/document.xml` with the Python standard library rather
than using `python-docx`, for three reasons:

- **Reading order** — paragraphs and tables must interleave as authored, because
  a table's meaning depends on the paragraph introducing it.
- **Style access** — 3GPP encodes heading depth as the paragraph style
  (`Heading1` through `Heading9`). That *is* the clause tree; a plain text dump
  loses it.
- **Revision marks** — several published specifications still carry `w:del` runs.
  That text is deleted content and must not reach the index; a retrieval hit on
  withdrawn normative text is a hallucination with a citation attached.

Tabs are preserved rather than collapsed, because 3GPP drafting uses them as the
only machine-readable delimiter in abbreviation lists (`5GMM<tab>5GS Mobility
Management`). That single decision is what makes section 4.4 possible.

### 4.3 Clause-anchored chunking

The chunk boundary is the clause boundary. A clause is already the smallest
self-contained normative unit: it is what specifications cross-reference, what
engineers cite, and what drafting convention keeps coherent.

Only clauses exceeding the token ceiling are split, on paragraph boundaries with
overlap, never mid-table. Oversized tables are banded on row boundaries **with
the header row repeated**, because a band without its header is unreadable to
both the embedding model and the human checking the citation.

Every chunk carries a provenance header, which is indexed and embedded with the
body:

```
3GPP TS 24.501 v20.0.0 (Rel-20) — clause 5.4.1.3.2 Authentication initiation by the network
Path: 5 Elementary procedures for 5GS mobility management > 5.4 5GMM common procedures >
      5.4.1 Primary authentication and key agreement procedure > 5.4.1.3 5G AKA based ...
```

This is what lets a clause headed "General" be retrieved for a question about
NAS security context handling, and it is the string the generator must cite.
Result: median 261 tokens per chunk.

### 4.4 Acronym lexicon, mined rather than written

Every 3GPP specification carries an "Abbreviations" clause formatted as
`ACRONYM<tab>Expansion`, and TR 21.905 is the project-wide vocabulary. The build
harvests these into a bidirectional map — **689 acronyms**, with no hand-written
list to rot.

Expansion runs both ways at query time: `5GMM` adds "5GS Mobility Management",
and "Session Management Function" adds `SMF`. Expansion terms enter BM25 at
reduced weight (0.45) so they improve recall without letting a generic expansion
outrank an exact identifier match.

Mining beats hardcoding because the lexicon is *authoritative* (the specs' own
definitions), stays correct when the corpus is re-fetched at a new release, and
captures genuine ambiguity — `AMF` resolves to both "Access and Mobility
Management Function" and "Authentication Management Field".

### 4.5 Retrieval

**BM25 tokenizer.** The highest-leverage detail in the stack. A stock word-character
pattern destroys every identifier in the corpus: `24.501` becomes `24` and `501`;
`5G-GUTI` becomes `5g` and `guti`. The tokenizer keeps dotted and hyphenated
identifiers whole *and additionally* emits their parts, preserving exact-match
precision without losing partial recall.

**Fusion by RRF, not weighted sum.** BM25 scores are unbounded and
corpus-dependent; cosine similarities are bounded and model-dependent. Combining
them by weighted sum requires a normalisation constant that silently drifts as
the corpus grows or the embedding model changes. RRF discards magnitudes and
fuses on rank position, so it stays stable across both.

**Structural shortcuts.** "What does TS 24.501 clause 5.4.1 say?" is an address,
not a topic. Named specifications boost their chunks; a named clause is pinned
directly into the candidate set so it cannot be lost to fuzzy matching.

**Per-clause diversity cap.** A long clause is indexed as several parts which
rerank almost identically; without a cap one clause took four of six slots,
burning the generator's context budget and starving the answer of corroboration.

**Reranking exists for calibration, not only ordering.** The abstention gate
thresholds on the top score, so that score has to mean something. A cross-encoder
reads query and passage jointly and separates "answers the question" from
"shares vocabulary with the question" far more sharply than cosine does.

---

## 5. The four defences against a wrong answer

### 5.1 Premise checks (categorical, before scoring)

- **Unindexed specification** — the question names `TS 36.331`, which is not in
  the corpus. Refuse, naming it.
- **Unknown entity** — an entity-shaped token (carrying a digit or hyphen, or
  capitalised) appears nowhere in the corpus vocabulary. Refuse, naming it.

The motivating measurement: questions about timer `T9999` and about `TS 36.331`
reranked at **0.967 to 0.999**, higher than several legitimate questions. This is
not a tuning failure. Retrieval returns its best six passages for any input, and
a question about a nonexistent timer retrieves genuine timer clauses that a
cross-encoder correctly judges to be about timers. No threshold on that score can
separate them. Checking the premise does.

### 5.2 Abstention gate (graded)

Three independent signals, any of which can refuse: top reranked score;
corroboration from hits 2 and 3; and IDF-weighted overlap between question terms
and the top passages. Independence is the point — an out-of-domain question can
score mediocre-but-not-terrible on the reranker while its vocabulary is plainly
absent from the corpus.

### 5.3 Constrained generation

One claim per line, each ending with the markers of the sources it came from
(`[S1][S4]`). This makes attribution mechanical — the verifier never guesses
which passage a sentence came from — and is far more robust to parse than JSON
from a mid-size model. Prior knowledge is explicitly forbidden, which matters
because a strong model has read a great deal of telecom material and will
otherwise answer from memory and cite a passage that merely looks related.

### 5.4 Grounding verification (after generation)

The generated answer is treated as a set of claims to be checked against the
passages each one cites.

**Check 1 — hard-token grounding, deterministic.** Extract from each claim every
number, timer (`T3510`), specification reference (`TS 24.501`), hex literal, and
uppercase identifier or message name (`REGISTRATION REQUEST`). Require each to
appear verbatim in a passage that claim itself cites.

This targets the failure that actually matters. The dangerous output is not
invented prose, it is a correct-looking sentence with `T3512` where the spec says
`T3510`, or `30 s` where it says `15 s`. Because these tokens must match
literally, that class of error is caught **with certainty rather than
probability**, at the cost of one pass over a string and no model call. Unit
wording is deliberately tolerated: "15 seconds" against a source saying "15 s"
passes, because the *value* is what is checked.

**Check 2 — IDF-weighted lexical support.** Guards the opposite failure: a claim
whose individual tokens all appear in the source but whose substance does not.

**Check 3 (optional) — LLM entailment.** Catches correctly-copied tokens
assembled into a false statement. Off by default: it costs a round-trip and,
unlike checks 1 and 2, it is itself a model and therefore fallible.

Claims failing verification are removed and shown separately with the reason. If
none survive, the answer is withdrawn and the system abstains.

---

## 6. Degradation path

| Tier | Embeddings | Rerank | Generation | Requires |
|---|---|---|---|---|
| Full | bge-small bi-encoder | cross-encoder | LLM, verified | model download + API key |
| No key | bge-small | cross-encoder | **extractive** | model download |
| Offline | LSA (TF-IDF + SVD) | lexical | extractive | nothing |

Extractive mode quotes retrieved clauses verbatim. It **cannot hallucinate by
construction**, making it both a usable fallback and the zero-fabrication
baseline the LLM path is measured against.

---

## 7. Evaluation

25 questions: 17 answerable with ground-truth clauses, 8 adversarial. Ground
truth was derived *from the indexed corpus* — every expected clause was confirmed
to exist and every required token confirmed present in it before being recorded.

Configuration: `groq:llama-3.3-70b-versatile`, `BAAI/bge-small-en-v1.5`,
`cross-encoder/ms-marco-MiniLM-L-6-v2`.

| Metric | Result |
|---|---|
| Retrieval recall@6 | 0.94 |
| Retrieval MRR | 0.843 |
| Answer rate on answerable questions | 0.88 (15/17) |
| Correct abstention on adversarial | 0.875 (7/8) |
| Groundedness of emitted claims | 1.00 (79/79) |
| Citation validity | 1.00 |
| Required values present / forbidden values emitted | 100% / 0 |
| Mean latency | 3.3 s |

### 7.1 Retrieval ablation — natural-language questions

| Configuration | recall@6 | MRR | top-1 |
|---|---|---|---|
| BM25 only | 0.65 | 0.382 | 0.24 |
| Dense only | **1.00** | 0.769 | 0.65 |
| Hybrid (RRF) | 0.88 | 0.623 | 0.47 |
| Hybrid + cross-encoder | 0.94 | **0.843** | **0.76** |

Read honestly: **dense-only wins on raw recall@6.** Equal-weight RRF lets lexical
candidates displace dense ones inside the top 6. Reranking recovers most of it
and produces much the best *ordering*, which is what matters, since the top
passages are what the generator sees and what the gate thresholds on.

### 7.2 Retrieval ablation — identifier queries

The golden set is mostly natural-language questions, which is dense retrieval's
home turf. On identifier-bearing queries — the lookups an engineer actually types
— the result inverts:

| Configuration | recall@6 | MRR | top-1 |
|---|---|---|---|
| BM25 only | 0.70 | 0.533 | 0.40 |
| Dense only | 0.50 | 0.162 | **0.00** |
| **Hybrid (RRF)** | **0.80** | **0.570** | 0.40 |

Dense retrieval never ranked the correct clause first, and **missed all four
timer lookups outright** (`T3512`, `T3560`, `T3570`, `T3580` — absent from the
top 6). Those tokens are near-identical in embedding space; BM25 treats them as
different words.

**This is the entire argument for hybrid retrieval: the two retrievers fail on
disjoint query types.** It is also why the tokenizer keeps identifiers intact.

### 7.3 Abstention

Seven of eight adversarial questions refused, each with a specific reason rather
than a generic decline: `'france'` absent from the corpus; `'bgp'`, `'cisco'`,
`'ios'` absent; `'wi-fi'` absent; TS 36.331 not indexed; `'t9999'` absent;
`'python'` absent; and the 5GMM "quantum handover" question caught by the
generation layer emitting `INSUFFICIENT_CONTEXT`.

### 7.4 A false-positive mode found after the fact

The premise check originally flagged **any** unknown word of four or more
letters. On the golden set that measured a zero false-positive rate, and the
conclusion drawn from it — "ordinary English carries no risk at this vocabulary
size" — was wrong. The golden set is phrased in domain language and so could not
have exposed the failure.

Probing with deliberately jargon-free questions did. Two were refused for relying
on **"prove"** and **"disagree"** — 17M characters of formal specification prose
contain no conversational verbs. Absence from this corpus reports its **register**
as much as the question's validity.

The rule was narrowed to entity-shaped tokens, which keeps every genuine catch.
The cost is recorded rather than hidden: a false premise built from ordinary
lowercase words is no longer caught there and falls to the generator.

The transferable lesson: **a clean number on an evaluation set is not evidence
when the evaluation set shares an assumption with the code.**

---

## 8. Alternatives considered and rejected

**Fine-tuning on 3GPP text.** Moves knowledge into weights, where it cannot be
cited, updated, or verified — the opposite of the requirement. Specifications are
revised quarterly; retraining to fix one clause is untenable. It also makes
hallucination worse here, by making the model confident with no provenance.

**Vector database (FAISS, Chroma, pgvector).** At 11.6k chunks, brute-force
cosine over a NumPy array is exact and sub-millisecond. A vector database adds a
dependency, an approximation, and a service to run, for no gain at this scale.
The rewrite point is roughly one million chunks; one function changes.

**Dense-only retrieval.** Rejected on the identifier-collision argument and
quantified in section 7.2.

**GraphRAG / knowledge graph.** Genuinely attractive for 3GPP, whose
cross-references form a real graph. Rejected on cost/benefit for this scope: the
extraction step becomes a hallucination source of its own, and the clause
hierarchy already supplies most of the structure a graph would add.

**Semantic (embedding-based) chunking.** Solves a problem this corpus does not
have. 3GPP clause numbering is a better segmentation than any similarity
heuristic would infer, and it is authored by the standards body.

**JSON-structured generation.** More brittle than one-claim-per-line under
mid-size models, for the same verifiability.

---

## 9. Known limitations

- **Figures are not read.** 3GPP message-flow diagrams are embedded images;
  captions are indexed, content is not. Figure-answered questions under-retrieve.
  A vision-model captioning pass at ingest is the fix.
- **Cross-reference following is single-hop.** "See clause 5.4.1.3" is retrieved
  as text, not traversed.
- **A false premise built from in-vocabulary words survives the gate.** Only the
  generation layer can decline it, and that is not deterministic — the
  "5GMM quantum handover" question abstained in 4 of 6 runs at `temperature=0`,
  because the inference backend is not bit-reproducible.
- **Heavy paraphrase degrades the reranker.** `ms-marco-MiniLM` is trained on
  web-search queries; on telecom paraphrase carrying no domain vocabulary it
  scored a correct rank-2 clause at 0.04, and the system refused. A reranker
  fine-tuned on telecom query/clause pairs is the obvious next step.
- **Hard-token checking can over-reject.** A legitimate unit conversion ("0.5 s"
  for "500 ms") is flagged. This is a deliberate bias toward false refusal over
  false assertion, and the evaluation measures its cost.
- **Eight specifications, not the full corpus.** Adding more is a configuration
  line; questions outside these eight correctly abstain.
- **The golden set is small (25 questions) and self-authored.** Thresholds are
  tuned on it, so the abstention figures are optimistic. An honest replication
  would hold out a second set.

---

## 10. Running it

```
pip install -e .
python -m telcorag fetch      # ~53 MB from 3gpp.org
python -m telcorag build      # ~23 min CPU, or ~30 s with TELCORAG_EMBEDDINGS=lsa
python -m telcorag serve      # http://127.0.0.1:8000
```

No API key is required — without one the system answers extractively. To enable
generated prose, copy `.env.example` to `.env` and set `GROQ_API_KEY` or
`OPENAI_API_KEY`.

```
python -m telcorag ask "What is timer T3512 used for?" --verbose
python -m telcorag eval --out eval/report.json
python -m pytest                                        # 86 tests, fully offline
```

The corpus and derived index are not included in the archive — they are
reproducible with `fetch` and `build`, and would add roughly 200 MB.

### Layout

```
src/telcorag/
  corpus/      download.py   3GPP archive fetch, version decoding
               ooxml.py      WordprocessingML reader (stdlib only)
               parser.py     clause tree with ancestry
               chunker.py    clause-anchored chunking
  index/       bm25.py       Okapi BM25, telecom-aware tokenizer
               embeddings.py pluggable dense backends
               store.py      index bundle, save and load
  retrieval/   fusion.py     reciprocal rank fusion
               rerank.py     cross-encoder / lexical
               pipeline.py   retrieval pipeline, premise checks
  answer/      guard.py      abstention gate
               prompts.py    generation contract
               generator.py  orchestration
               verifier.py   grounding verification
               llm.py        Groq / OpenAI / Ollama / none
  glossary.py  acronym lexicon mined from the corpus
  evaluate.py  metrics and retrieval ablation
web/index.html the UI
tests/         86 tests
```
