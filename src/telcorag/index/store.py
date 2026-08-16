from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..corpus.chunker import Chunk
from ..glossary import Glossary
from .bm25 import BM25
from .embeddings import EmbeddingBackend, restore

FORMAT_VERSION = 2


@dataclass
class Index:
    chunks: list[Chunk]
    bm25: BM25
    vectors: np.ndarray
    backend: EmbeddingBackend
    glossary: Glossary = field(default_factory=Glossary.empty)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_id = {c.id: i for i, c in enumerate(self.chunks)}

    def __len__(self) -> int:
        return len(self.chunks)

    def position(self, chunk_id: str) -> int | None:
        return self._by_id.get(chunk_id)

    @property
    def specs(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for c in self.chunks:
            entry = seen.setdefault(c.spec_id, {"spec_id": c.spec_id, "version": c.version, "release": c.release, "chunks": 0})
            entry["chunks"] += 1
        return sorted(seen.values(), key=lambda e: e["spec_id"])

    @classmethod
    def build(cls, chunks: list[Chunk], backend: EmbeddingBackend, glossary: Glossary, meta: dict | None = None) -> "Index":
        corpus = [c.indexed_text for c in chunks]
        backend.fit(corpus)
        vectors = backend.encode_documents(corpus)
        bm25 = BM25().fit(corpus)
        return cls(chunks, bm25, vectors, backend, glossary, meta or {})

    def dense_search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if not len(self.chunks):
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        scores = self.vectors @ query_vector.astype(np.float32)
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(-scores, k - 1)[:k] if k < scores.shape[0] else np.arange(scores.shape[0])
        idx = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]

    def lexical_search(self, query: str, top_k: int, extra_terms: list[str] | None = None):
        return self.bm25.search(query, top_k, extra_terms)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
        np.save(path / "vectors.npy", self.vectors)
        self.bm25.save(path)
        self.glossary.save(path / "glossary.json")
        self.backend.save(path)
        (path / "index.json").write_text(
            json.dumps(
                {
                    "format": FORMAT_VERSION,
                    "backend": self.backend.name,
                    "dim": int(self.vectors.shape[1]) if self.vectors.size else 0,
                    "chunks": len(self.chunks),
                    **self.meta,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path, lsa_dims: int = 256) -> "Index":
        meta_path = path / "index.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"no index at {path} — run `python -m telcorag build` first")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"index format {meta.get('format')} != {FORMAT_VERSION}; rebuild required")

        chunks = [Chunk.from_dict(json.loads(line)) for line in (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        vectors = np.load(path / "vectors.npy")
        bm25 = BM25.load(path)
        glossary = Glossary.load(path / "glossary.json")
        backend = restore(meta["backend"], path, lsa_dims)
        return cls(chunks, bm25, vectors, backend, glossary, meta)
