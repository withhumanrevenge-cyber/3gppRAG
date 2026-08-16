# Evaluation

Reproduce with:

```bash
python -m telcorag eval --out eval/report.json
```

**Configuration.** 11,590 chunks from 8 specifications · embeddings
`BAAI/bge-small-en-v1.5` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` ·
generation `groq:llama-3.3-70b-versatile`.

**Golden set.** 25 questions in [eval/golden.json](eval/golden.json): 17
answerable with ground-truth clauses, 8 adversarial. Ground truth was derived
*from the indexed corpus* — every expected clause was confirmed to exist and
every required token confirmed present in that clause before being recorded.

---

## Headline results

| Metric | Result |
|---|---|
| Retrieval recall@6 | **0.94** (16/17) |
| Retrieval MRR | **0.843** |
| Answer rate on answerable questions | 0.88 (15/17) |
| **Correct abstention on adversarial** | **7–8 of 8** — one case is stochastic, [see below](#the-borderline-case-and-why-one-run-is-not-evidence) |
| Groundedness of emitted claims | 1.00 (84/84) |
| Citation validity | 1.00 |
| Required values present | 100% (0 misses) |
| Forbidden values emitted | 0 |
| Mean latency | 4.3 s |

Retrieval figures are identical with and without an LLM, as they must be —
generation sits downstream of retrieval and cannot change it.

### A measurement bug worth recording

The first LLM run reported MRR 0.882 and citation validity 0.933. Both were
wrong, in opposite directions, from one root cause: `Answer.sources` is filtered
to the passages the answer actually *cited*.

Ranking the correct clause within that filtered list let selective citation
masquerade as better retrieval (0.882 against a true 0.843), while bounding
citation indices by `len(sources)` flagged a perfectly valid `[S5]` as out of
range once only four sources survived filtering. The harness now measures rank
over everything retrieved, and validates citation indices against the number of
sources *offered to the model* — both recorded in diagnostics. Corrected MRR
agrees with the ablation exactly, which is the consistency check that should
have been there from the start.

---

## Retrieval ablation

Same golden set, retrieval only, k=6.

| Configuration | recall@6 | MRR | top-1 |
|---|---|---|---|
| BM25 only | 0.65 | 0.382 | 0.24 |
| Dense only | **1.00** | 0.769 | 0.65 |
| Hybrid (RRF) | 0.88 | 0.623 | 0.47 |
| Hybrid + cross-encoder | 0.94 | **0.843** | **0.76** |

**Read this honestly: dense-only wins on raw recall@6.** Equal-weight RRF lets
50 lexical candidates displace dense ones inside the top 6, costing two
questions. Reranking recovers most of it and produces much the best *ordering*
(MRR 0.843 vs 0.769, top-1 0.76 vs 0.65) — and ordering is what actually
matters, because the top passages are what the generator sees and what the
abstention gate thresholds on.

On 17 questions these gaps are one or two questions wide. Treat them as
directional, not decisive.

### So why keep BM25?

Because the golden set is mostly natural-language questions, which is dense
retrieval's home turf. Re-running the same ablation on **identifier-bearing
queries** — the lookups an engineer actually types — inverts the result:

| Configuration | recall@6 | MRR | top-1 |
|---|---|---|---|
| BM25 only | 0.70 | 0.533 | 0.40 |
| Dense only | 0.50 | 0.162 | **0.00** |
| **Hybrid (RRF)** | **0.80** | **0.570** | 0.40 |

Dense retrieval never once ranked the correct clause first, and **missed all
four timer lookups outright** (`T3512`, `T3560`, `T3570`, `T3580` — not in the
top 6 at all). Those tokens are near-identical in embedding space; BM25 treats
them as different words. Hybrid beats both populations because the two
retrievers fail on disjoint query types.

That is the entire argument for hybrid retrieval on this corpus, and it is the
reason the tokenizer keeps `T3512` and `24.501` intact rather than shredding
them into `t`/`3512` and `24`/`501`.

---

## Abstention

7 of 8 adversarial questions correctly refused, each with a specific reason
rather than a generic decline:

| Question | Caught by | Reason given |
|---|---|---|
| Capital of France | premise | `'france'` absent from the corpus |
| Cisco BGP route reflectors | premise | `'bgp'`, `'cisco'`, `'ios'` absent |
| Wi-Fi 6E EIRP limit | premise | `'wi-fi'`, `'ghz'` absent |
| TS 36.331 LTE RRC | unindexed spec | TS 36.331 is not in the corpus |
| Timer T9999 value | premise | `'t9999'` absent |
| Blood-type IE in REGISTRATION REQUEST | premise | `'blood'` absent |
| Python PCAP parser | premise | `'python'` absent |
| 5GMM quantum handover procedure | **generation layer** | model emitted `INSUFFICIENT_CONTEXT` — but only in 4 of 6 runs |

The four score-based refusals were not close calls: out-of-domain questions
reranked at 0.000–0.081 against 0.30+ for genuine ones.

**Why the premise check exists.** Three of those — `T9999`, `blood type`,
`TS 36.331` — reranked at **0.844–0.999**, higher than several legitimate
questions. Score thresholds structurally cannot catch them, because retrieval
returns its best six topically-adjacent clauses no matter what is asked. Only
checking the question's *premise* against the corpus vocabulary catches them.

### The borderline case, and why one run is not evidence

"Describe the 5GMM quantum handover procedure" is the only adversarial question
no pre-generation check can touch. Every term in it is in the corpus vocabulary
— including "quantum", which does appear in TS 33.501 — so the premise check is
blind to it, and its passages rerank at 0.959. It is precisely the class the
architecture predicted only the generation layer could catch.

With an LLM configured, it *is* caught: the model emits `INSUFFICIENT_CONTEXT`
per rule 4 of the generation contract, and the system abstains. The first eval
run scored 8/8 abstention on the strength of it.

**That result did not reproduce.** Re-running the same question six times:

```
ABSTAIN  ABSTAIN  ABSTAIN  ANSWER  ABSTAIN  ANSWER      -> 4/6 abstain
```

Groq is not deterministic at `temperature=0` — batched inference and
non-deterministic kernels mean identical inputs can yield different outputs. So
the honest figure for correct abstention is **7–8 of 8**, with the eighth
question landing on the right side roughly two thirds of the time.

Two things follow. First, the mechanism works and is the right one; it is the
*reliability* that is partial. Second, and more generally: a single run of a
25-question eval is not a measurement when a stochastic component sits in the
loop. Every deterministic layer of this system — premise checks, hard-token
verification — reproduces exactly. The LLM layer does not, which is an argument
for putting as much of the safety burden as possible on the layers that do.

For a production deployment the fix is to stop relying on a sampled refusal:
run the abstention decision as a separate low-variance call, or take a majority
vote across `n` samples for questions near the boundary.

**In extractive mode this question is always answered**, since quoting cannot
decline. That is the sharpest illustration of what the LLM tier buys.

---

## False refusals

Two answerable questions were refused, both deliberate paraphrase probes:

- *"How does the device let the network know it is still reachable at regular
  intervals?"* — the correct clause (24.501 §5.3.7) was retrieved **at rank 2**,
  but the cross-encoder scored it 0.04 while an unrelated clause took the top
  slot at 0.30. The corroboration signal fired and the system refused.
- *"What stops a subscriber's permanent identity from being sent in the clear
  over the radio interface?"* — the SUCI clauses were not retrieved in the top 6
  at all; top score 0.17.

Both are the same root cause: `ms-marco-MiniLM`, trained on web search queries,
degrades on heavy telecom paraphrase carrying none of the domain vocabulary. In
the second case refusal is arguably the *correct* outcome — retrieval had
failed, so answering would have produced a confident answer from the wrong
clauses.

This is the designed bias: **refuse rather than guess.** The cost is visible
here as an 0.88 answer rate rather than hidden.

---

## Threshold calibration

`min_top_score = 0.20` was not chosen as a round number. The observed rerank
scores form two separated populations:

```
score-caught adversarial:  0.000  0.003  0.015  0.081
genuine questions:                              0.30 … 0.999
                                    ↑
                            0.20 sits in the empty band
```

Nothing was observed between 0.081 and 0.30, so the threshold is placed there.
Re-derive it if the corpus or reranker changes — and note the band is drawn
from a small sample.

---

## What these numbers do not show

**Groundedness of 1.00 means the verifier never fired, not that it works.**
84 claims from a real model, none rejected: `llama-3.3-70b` followed the
citation contract and copied values faithfully on all 25 questions. That is a
good result for the *prompt*, but it means this run contains no evidence about
the verifier's ability to catch a fabrication — nothing needed catching.

Evidence for that lives in [tests/test_integration.py](tests/test_integration.py),
which drives the full pipeline with a stub LLM whose output is chosen to be wrong:

- a faithful answer passes, groundedness 1.00
- an answer with one fabricated value (`45 s`, which appears in no cited clause)
  has that claim stripped, groundedness 0.50
- a wholly fabricated answer (`T9999`, `120 s`) has every claim rejected and the
  answer withdrawn to an abstention

Read those two together: the eval shows the pipeline does not *provoke*
hallucination on this corpus; the tests show it would be caught if it did.

Worth noting that the verifier did fire during development — an earlier
extractive run scored 0.69 groundedness and exposed two real bugs (claims split
across newlines losing their citation, and `5.5` being extracted as a number
from the clause reference `5.5.1.2.4`). The metric earned its keep before it
ever saw model output.

**Other caveats.** 25 questions is small — each adversarial item moves the
abstention rate by 12.5 points. The golden set was written by the same person
who built the system. Thresholds were tuned on this set, so the abstention
figures are optimistic; an honest replication would hold out a second set.
And per the section above, the LLM layer is not reproducible run to run, so
treat any single figure that depends on it as a sample rather than a constant.
