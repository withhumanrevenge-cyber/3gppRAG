"""Pluggable dense-embedding backends.

Two backends ship, and the choice is a deliberate trade rather than a fallback
for its own sake:

``st``   sentence-transformers bi-encoder (default, ``BAAI/bge-small-en-v1.5``).
         Real paraphrase matching -- "how does the phone tell the network it is
         still alive" retrieves the periodic registration update clause. Costs a
         one-off model download and CPU time at build.

``lsa``  TF-IDF + truncated SVD fitted on the corpus. No download, no network,
         builds in seconds. Captures co-occurrence ("AMF" near "registration")
         but not genuine paraphrase, so hybrid retrieval leans harder on BM25.
         Kept because a reviewer must be able to run this offline, and because
         the eval harness quantifies exactly what it costs (see EVALUATION.md).

Both return L2-normalised vectors so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol

import numpy as np

from .bm25 import tokenize


def _analyzer(text: str) -> list[str]:
    """Module-level so a fitted TfidfVectorizer stays picklable."""
    return tokenize(text)

# bge-family models are trained with an asymmetric query instruction.
QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
}


def _query_prefix(model_name: str) -> str:
    low = model_name.lower()
    for key, prefix in QUERY_PREFIXES.items():
        if key in low:
            return prefix
    return ""


def _doc_prefix(model_name: str) -> str:
    return "passage: " if "e5" in model_name.lower() else ""


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return (vecs / np.maximum(norms, 1e-9)).astype(np.float32)


class EmbeddingBackend(Protocol):
    name: str
    dim: int

    def fit(self, corpus: list[str]) -> None: ...
    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_queries(self, texts: list[str]) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...


class SentenceTransformerBackend:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.name = f"st:{model_name}"
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        probe = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(probe())

    def fit(self, corpus: list[str]) -> None:
        return None

    def _encode(self, texts: list[str], prefix: str, progress: bool) -> np.ndarray:
        payload = [prefix + t for t in texts] if prefix else texts
        vecs = self._model.encode(
            payload,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=progress,
        )
        return vecs.astype(np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, _doc_prefix(self.model_name), progress=len(texts) > 512)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, _query_prefix(self.model_name), progress=False)

    def save(self, path: Path) -> None:
        (path / "embedder.txt").write_text(self.name, encoding="utf-8")

    def load(self, path: Path) -> None:
        return None


class LSABackend:
    def __init__(self, dims: int = 256) -> None:
        self.name = f"lsa:{dims}"
        self.dim = dims
        self._vectorizer = None
        self._svd = None

    def fit(self, corpus: list[str]) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        # A real corpus benefits from pruning singletons; a handful of documents
        # would be pruned into an empty vocabulary.
        small = len(corpus) < 25
        self._vectorizer = TfidfVectorizer(
            analyzer=_analyzer,
            min_df=1 if small else 2,
            max_df=1.0 if small else 0.6,
            sublinear_tf=True,
        )
        matrix = self._vectorizer.fit_transform(corpus)
        dims = int(min(self.dim, max(2, min(matrix.shape) - 1)))
        self._svd = TruncatedSVD(n_components=dims, random_state=0, algorithm="randomized", n_iter=7)
        self._svd.fit(matrix)
        self.dim = dims

    def _transform(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("LSABackend used before fit()/load()")
        return _normalize(self._svd.transform(self._vectorizer.transform(texts)))

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._transform(texts)

    def save(self, path: Path) -> None:
        with (path / "lsa.pkl").open("wb") as fh:
            pickle.dump({"vectorizer": self._vectorizer, "svd": self._svd, "dim": self.dim}, fh)

    def load(self, path: Path) -> None:
        with (path / "lsa.pkl").open("rb") as fh:
            state = pickle.load(fh)
        self._vectorizer, self._svd, self.dim = state["vectorizer"], state["svd"], state["dim"]


def resolve(spec: str, model_name: str, lsa_dims: int, quiet: bool = False) -> EmbeddingBackend:
    spec = (spec or "auto").lower()
    if spec.startswith("lsa"):
        return LSABackend(lsa_dims)
    if spec in ("st", "sentence-transformers", "neural"):
        return SentenceTransformerBackend(model_name)
    if spec != "auto":
        raise ValueError(f"unknown embedding backend: {spec}")

    try:
        return SentenceTransformerBackend(model_name)
    except Exception as exc:  # missing package, no network on first download, etc.
        if not quiet:
            print(f"[embeddings] falling back to LSA ({type(exc).__name__}: {exc})")
        return LSABackend(lsa_dims)


def restore(name: str, path: Path, lsa_dims: int) -> EmbeddingBackend:
    if name.startswith("lsa"):
        backend = LSABackend(lsa_dims)
        backend.load(path)
        backend.name = name
        return backend
    return SentenceTransformerBackend(name.split(":", 1)[1])
