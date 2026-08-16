from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def _s(key: str, default: str) -> str:
    return (os.environ.get(key) or default).strip()


def _i(key: str, default: int) -> int:
    try:
        return int(os.environ[key])
    except (KeyError, ValueError):
        return default


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ[key])
    except (KeyError, ValueError):
        return default


# Specs pulled by `telcorag fetch`. Series directory is derived from the spec number.
DEFAULT_SPECS: tuple[str, ...] = (
    "21.905",  # Vocabulary for 3GPP Specifications -- seeds the acronym glossary
    "23.501",  # System architecture for the 5G System
    "23.502",  # Procedures for the 5G System
    "24.501",  # NAS protocol for 5GS
    "33.501",  # Security architecture and procedures for 5G System
    "38.300",  # NR and NG-RAN overall description
    "38.321",  # NR Medium Access Control
    "38.331",  # NR Radio Resource Control
)


@dataclass(frozen=True)
class Retrieval:
    bm25_candidates: int = field(default_factory=lambda: _i("TELCORAG_BM25_K", 50))
    dense_candidates: int = field(default_factory=lambda: _i("TELCORAG_DENSE_K", 50))
    rrf_k: int = field(default_factory=lambda: _i("TELCORAG_RRF_K", 60))
    rerank_depth: int = field(default_factory=lambda: _i("TELCORAG_RERANK_DEPTH", 20))
    final_k: int = field(default_factory=lambda: _i("TELCORAG_FINAL_K", 6))
    neighbour_window: int = field(default_factory=lambda: _i("TELCORAG_NEIGHBOURS", 1))
    max_parts_per_clause: int = field(default_factory=lambda: _i("TELCORAG_MAX_PARTS", 2))
    bm25_weight: float = field(default_factory=lambda: _f("TELCORAG_BM25_WEIGHT", 1.0))
    dense_weight: float = field(default_factory=lambda: _f("TELCORAG_DENSE_WEIGHT", 1.0))
    reranker: str = field(default_factory=lambda: _s("TELCORAG_RERANKER", "auto"))


@dataclass(frozen=True)
class Guard:
    """Thresholds controlling when the system refuses to answer.

    ``min_top_score`` is placed in the empty band of the observed rerank-score
    distribution on eval/golden.json: out-of-domain questions caught by score
    top out at 0.081, while genuine questions start at 0.30. 0.20 sits in the
    gap rather than at a round number, so it separates the two populations
    without sitting on top of either. Re-derive it if the corpus changes --
    see EVALUATION.md.
    """

    min_top_score: float = field(default_factory=lambda: _f("TELCORAG_MIN_TOP_SCORE", 0.20))
    min_support: float = field(default_factory=lambda: _f("TELCORAG_MIN_SUPPORT", 0.18))
    min_lexical_overlap: float = field(default_factory=lambda: _f("TELCORAG_MIN_OVERLAP", 0.08))
    min_sentence_support: float = field(default_factory=lambda: _f("TELCORAG_MIN_SENT_SUPPORT", 0.45))
    drop_unverified: bool = field(default_factory=lambda: _s("TELCORAG_DROP_UNVERIFIED", "1") == "1")
    premise_check: bool = field(default_factory=lambda: _s("TELCORAG_PREMISE_CHECK", "1") == "1")


@dataclass(frozen=True)
class Chunking:
    target_tokens: int = field(default_factory=lambda: _i("TELCORAG_CHUNK_TOKENS", 320))
    max_tokens: int = field(default_factory=lambda: _i("TELCORAG_CHUNK_MAX", 480))
    overlap_tokens: int = field(default_factory=lambda: _i("TELCORAG_CHUNK_OVERLAP", 48))
    min_tokens: int = field(default_factory=lambda: _i("TELCORAG_CHUNK_MIN", 24))


@dataclass(frozen=True)
class Settings:
    raw_dir: Path = DATA / "raw"
    corpus_dir: Path = DATA / "corpus"
    index_dir: Path = DATA / "index"

    embedding_backend: str = field(default_factory=lambda: _s("TELCORAG_EMBEDDINGS", "auto"))
    embedding_model: str = field(default_factory=lambda: _s("TELCORAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    lsa_dims: int = field(default_factory=lambda: _i("TELCORAG_LSA_DIMS", 256))

    llm_provider: str = field(default_factory=lambda: _s("TELCORAG_LLM", "auto"))
    llm_model: str = field(default_factory=lambda: _s("TELCORAG_LLM_MODEL", ""))
    llm_temperature: float = field(default_factory=lambda: _f("TELCORAG_TEMPERATURE", 0.0))
    llm_timeout: float = field(default_factory=lambda: _f("TELCORAG_LLM_TIMEOUT", 90.0))

    groq_api_key: str = field(default_factory=lambda: _s("GROQ_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: _s("OPENAI_API_KEY", ""))
    ollama_host: str = field(default_factory=lambda: _s("OLLAMA_HOST", "http://localhost:11434"))

    retrieval: Retrieval = field(default_factory=Retrieval)
    guard: Guard = field(default_factory=Guard)
    chunking: Chunking = field(default_factory=Chunking)

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.corpus_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_dotenv(path: Path | None = None) -> None:
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()
settings = Settings()
