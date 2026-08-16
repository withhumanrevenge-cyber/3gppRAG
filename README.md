# 3GPP Standards Assistant

A retrieval-augmented question answering system over **3GPP telecom specifications**,
built so that every statement it makes is traceable to a clause — and so that it
refuses to answer when the corpus cannot support one.

It indexes the real published specifications, fetched directly from
`3gpp.org/ftp/Specs/archive`:

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

---

## The problem this is built around

Generic RAG fails on standards documents in ways that are easy to miss and
expensive to ship. The three that shaped this design:

- **Fixed-size chunking severs normative logic.** "If A, the UE shall B" split
  across a boundary produces an unconditional rule the spec never stated.
- **Embeddings blur exactly the tokens that matter.** `T3510`/`T3512`,
  `TS 24.501`/`TS 24.301`, `5GMM`/`5GSM` sit a hair apart in cosine space.
  Similarity search returns the neighbour, confidently.
- **A well-cited answer can still be wrong.** The dangerous output is a fluent
  sentence citing a real clause with one value changed. Prompt instructions
  cannot detect that.

So the system chunks on **clause boundaries**, retrieves with **BM25 fused with
dense vectors**, and — the part that does the real work — **verifies the
generated answer against the passages it cites**, sentence by sentence, before
showing it.

Full reasoning, including rejected alternatives: **[ARCHITECTURE.md](ARCHITECTURE.md)**.
Measured results: **[EVALUATION.md](EVALUATION.md)**.

---

## Quickstart

```bash
pip install -e .
```

```bash
python -m telcorag fetch
```

```bash
python -m telcorag build
```

```bash
python -m telcorag serve
```

Then open <http://127.0.0.1:8000>.

`fetch` downloads ~53 MB from 3gpp.org. `build` parses 5,822 clauses into 11,590
chunks; with neural embeddings it takes roughly 23 minutes on CPU, or about 30
seconds with `TELCORAG_EMBEDDINGS=lsa`.

**No API key is required.** Without one the system answers *extractively* —
quoting retrieved clauses verbatim — which cannot fabricate by construction. To
enable generated prose, set a key:

```bash
cp .env.example .env    # then set GROQ_API_KEY or OPENAI_API_KEY
```

---

## Usage

Ask from the command line:

```bash
python -m telcorag ask "Which timer does the UE start when it sends a REGISTRATION REQUEST?"
```

Show retrieval diagnostics, scores and rejected claims:

```bash
python -m telcorag ask "What is the SUCI?" --verbose
```

Add the optional LLM entailment pass on top of deterministic verification:

```bash
python -m telcorag ask "How is the 5G-GUTI reallocated?" --entailment
```

Pin the corpus to a specific 3GPP Release (answers are release-dependent):

```bash
python -m telcorag fetch --specs 24.501 38.331 --release 18
```

Look up an acronym from the mined lexicon:

```bash
python -m telcorag glossary NSSAI
```

Run the evaluation harness:

```bash
python -m telcorag eval --out eval/report.json
```

### HTTP API

`POST /api/ask` returns the answer, per-claim verification verdicts, scored
sources with clause breadcrumbs, and diagnostics.

```bash
curl -s -X POST localhost:8000/api/ask -H "Content-Type: application/json" -d "{\"question\":\"What is a QoS flow?\"}"
```

Also available: `GET /api/health`, `GET /api/glossary/{term}`,
`GET /api/clause/{spec}/{clause}` for direct clause lookup, and `/docs` for the
generated OpenAPI page.

---

## How a wrong answer gets stopped

Four independent chances to refuse, rather than one prompt hoping for the best:

1. **Premise check (before retrieval scoring).** If the question names a spec
   that is not indexed, or an entity-shaped term appearing nowhere in the corpus,
   refuse and say which. This exists because score thresholds cannot catch a
   false premise: questions about timer `T9999` and about `TS 36.331` reranked
   **above 0.96**, since retrieval returns its best passages no matter what is
   asked. (The rule was narrowed after it was caught refusing valid questions for
   containing "prove" — see [EVALUATION.md](EVALUATION.md#a-false-positive-mode-found-after-the-fact).)
2. **Abstention gate.** Top reranked score, corroboration across hits, and
   IDF-weighted vocabulary overlap. Any one can refuse.
3. **Constrained generation.** One claim per line, each ending with the source
   markers it came from. Prior knowledge is explicitly forbidden; values must be
   copied verbatim.
4. **Grounding verification (after generation).** Every number, timer, spec
   reference and message name in a claim must appear in a passage that claim
   cites. This is a string check, not a model — so a fabricated timer value is
   caught *with certainty*. Claims that fail are removed; if none survive, the
   answer is withdrawn and the system abstains.

The UI shows rejected claims deliberately: seeing what the model *tried* to
assert, and why it was blocked, is more informative than a silently clean answer.

---

## Results

From [eval/golden.json](eval/golden.json) — 17 answerable questions with
ground-truth clauses plus 8 adversarial ones. Full analysis, including what
these numbers *don't* show, is in [EVALUATION.md](EVALUATION.md).

Measured with `groq:llama-3.3-70b-versatile` as the generator.

| Metric | Result |
|---|---|
| Retrieval recall@6 / MRR | 0.94 / 0.843 |
| Correct abstention on adversarial questions | 7–8 of 8 |
| Groundedness of emitted claims | 1.00 (84/84) |
| Citation validity | 1.00 |
| Mean latency | 4.3 s |

The retrieval ablation is the part worth reading. On natural-language questions
dense retrieval leads; on **identifier-bearing queries** it collapses — top-1 of
0.00, missing all four timer lookups (`T3512`, `T3560`, `T3570`, `T3580`)
outright, because those tokens are neighbours in embedding space. BM25 finds
them. Hybrid beats both because the two retrievers fail on disjoint query types.

Abstention is reported as a range on purpose. Seven adversarial questions are
refused deterministically by the premise and score checks. The eighth — "the
5GMM quantum handover procedure" — has no out-of-vocabulary term to catch it
(even "quantum" appears in TS 33.501), so only the generator can decline it, and
it does so in **4 of 6 repeat runs**: Groq is not deterministic at
`temperature=0`. The mechanism is right; its reliability is partial, and that is
reported rather than smoothed over with a single lucky run.

---

## Layout

```
src/telcorag/
  corpus/      download.py  3GPP archive fetch, version decoding
               ooxml.py     WordprocessingML reader (stdlib only)
               parser.py    clause tree with ancestry
               chunker.py   clause-anchored chunking
  index/       bm25.py      Okapi BM25, telecom-aware tokenizer
               embeddings.py  pluggable dense backends
               store.py     index bundle, save/load
  retrieval/   fusion.py    reciprocal rank fusion
               rerank.py    cross-encoder / lexical
               pipeline.py  the retrieval pipeline
  answer/      guard.py     abstention gate
               prompts.py   generation contract
               generator.py orchestration
               verifier.py  grounding verification
               llm.py       Groq / OpenAI / Ollama / none
  glossary.py  acronym lexicon mined from the corpus
  evaluate.py  metrics + retrieval ablation
web/index.html the UI
tests/         86 tests
```

## Tests

```bash
python -m pytest
```

All 86 run entirely offline — no corpus, no network, no model downloads. They
cover version decoding, OOXML edge cases (revision marks, tabs, tables), clause
numbering including annexes and letter suffixes, chunk and table splitting, BM25
identifier handling, glossary extraction, fusion, and the abstention gate.

The integration tests drive the **whole pipeline with a stub LLM**, so the
generated-answer path is verified without an API key: a faithful answer passes,
an answer containing one fabricated value has that claim stripped, and a wholly
fabricated answer is withdrawn to an abstention.

## Configuration

Every knob is an environment variable with a working default — see
[.env.example](.env.example). The ones worth knowing:

| Variable | Default | Effect |
|---|---|---|
| `TELCORAG_EMBEDDINGS` | `auto` | `st` (neural) or `lsa` (offline, fast build) |
| `TELCORAG_RERANKER` | `auto` | `cross-encoder` or `lexical` |
| `TELCORAG_MIN_TOP_SCORE` | `0.32` | raise to refuse more often |
| `TELCORAG_DROP_UNVERIFIED` | `1` | `0` keeps failed claims, flagged |
