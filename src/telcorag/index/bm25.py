"""Okapi BM25 over a CSR-style posting list.

Lexical scoring is not a legacy baseline in this domain, it is the primary
signal. 3GPP answers hinge on exact tokens that embedding models routinely blur
together: ``T3510`` vs ``T3512``, ``5GMM`` vs ``5GSM``, ``TS 24.501`` vs
``TS 24.301``, ``N1`` vs ``N2``. A bi-encoder places those neighbours within a
hair of each other in cosine space; BM25 treats them as different words.

The tokenizer is the part that matters. A stock ``\\w+`` pattern shreds every
identifier in the corpus (``24.501`` -> ``24``/``501``, ``5G-GUTI`` -> ``5g``/``guti``),
so we keep dotted and hyphenated identifiers whole and *additionally* emit their
parts, which preserves exact-match precision without losing partial recall.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
SPLIT_RE = re.compile(r"[-.]")

STOPWORDS = frozenset(
    """a an the of and or to in for on at by is are be been being as with from that this these those
    it its shall should may can will would could not no if then than when where which who whom what
    such other any all each per via into over under between within""".split()
)


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        if not keep_stopwords and raw in STOPWORDS:
            continue
        out.append(raw)
        if SPLIT_RE.search(raw):
            for part in SPLIT_RE.split(raw):
                if len(part) > 1 and (keep_stopwords or part not in STOPWORDS):
                    out.append(part)
    return out


class BM25:
    def __init__(self, k1: float = 1.4, b: float = 0.72) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.offsets = np.zeros(1, dtype=np.int64)
        self.postings = np.zeros(0, dtype=np.int32)
        self.freqs = np.zeros(0, dtype=np.float32)
        self.idf = np.zeros(0, dtype=np.float32)
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avgdl = 1.0
        self.n_docs = 0

    def fit(self, documents: list[str]) -> "BM25":
        term_docs: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_len = np.zeros(len(documents), dtype=np.float32)

        for doc_id, text in enumerate(documents):
            counts: dict[str, int] = defaultdict(int)
            tokens = tokenize(text)
            for tok in tokens:
                counts[tok] += 1
            doc_len[doc_id] = len(tokens)
            for term, tf in counts.items():
                term_docs[term].append((doc_id, tf))

        self.n_docs = len(documents)
        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if self.n_docs else 1.0

        self.vocab = {term: i for i, term in enumerate(sorted(term_docs))}
        offsets = np.zeros(len(self.vocab) + 1, dtype=np.int64)
        postings = np.zeros(sum(len(v) for v in term_docs.values()), dtype=np.int32)
        freqs = np.zeros(postings.shape[0], dtype=np.float32)
        idf = np.zeros(len(self.vocab), dtype=np.float32)

        cursor = 0
        for term, tid in self.vocab.items():
            entries = term_docs[term]
            offsets[tid] = cursor
            for doc_id, tf in entries:
                postings[cursor] = doc_id
                freqs[cursor] = tf
                cursor += 1
            df = len(entries)
            idf[tid] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
        offsets[-1] = cursor

        self.offsets, self.postings, self.freqs, self.idf = offsets, postings, freqs, idf
        return self

    def score(self, terms: list[str], weights: list[float] | None = None) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return scores
        weights = weights or [1.0] * len(terms)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

        for term, weight in zip(terms, weights):
            tid = self.vocab.get(term)
            if tid is None:
                continue
            lo, hi = self.offsets[tid], self.offsets[tid + 1]
            docs = self.postings[lo:hi]
            tf = self.freqs[lo:hi]
            contrib = self.idf[tid] * (tf * (self.k1 + 1.0)) / (tf + norm[docs])
            np.add.at(scores, docs, weight * contrib)
        return scores

    def search(self, query: str, top_k: int, extra_terms: list[str] | None = None, extra_weight: float = 0.45):
        terms = tokenize(query)
        weights = [1.0] * len(terms)
        for phrase in extra_terms or []:
            for tok in tokenize(phrase):
                terms.append(tok)
                weights.append(extra_weight)
        if not terms:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        scores = self.score(terms, weights)
        k = min(top_k, self.n_docs)
        idx = np.argpartition(-scores, k - 1)[:k] if k < self.n_docs else np.arange(self.n_docs)
        idx = idx[np.argsort(-scores[idx])]
        idx = idx[scores[idx] > 0]
        return idx, scores[idx]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path / "bm25.npz",
            offsets=self.offsets,
            postings=self.postings,
            freqs=self.freqs,
            idf=self.idf,
            doc_len=self.doc_len,
        )
        (path / "bm25_meta.json").write_text(
            json.dumps({"vocab": self.vocab, "avgdl": self.avgdl, "n_docs": self.n_docs, "k1": self.k1, "b": self.b}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BM25":
        meta = json.loads((path / "bm25_meta.json").read_text(encoding="utf-8"))
        obj = cls(k1=meta["k1"], b=meta["b"])
        data = np.load(path / "bm25.npz")
        obj.offsets, obj.postings = data["offsets"], data["postings"]
        obj.freqs, obj.idf, obj.doc_len = data["freqs"], data["idf"], data["doc_len"]
        obj.vocab, obj.avgdl, obj.n_docs = meta["vocab"], meta["avgdl"], meta["n_docs"]
        return obj
