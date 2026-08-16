from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import settings

WEB = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="3GPP RAG", description="Grounded question answering over 3GPP specifications", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int | None = Field(default=None, ge=1, le=20)
    specs: list[str] | None = None
    entailment: bool = False


@lru_cache(maxsize=1)
def _load():
    from .answer.generator import AnswerEngine
    from .index.store import Index

    index = Index.load(settings.index_dir, settings.lsa_dims)
    return AnswerEngine(index, settings), index


def _engine():
    try:
        return _load()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    page = WEB / "index.html"
    if not page.exists():
        return "<h1>3GPP RAG</h1><p>UI not found; the API is at /docs</p>"
    return page.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    try:
        engine, index = _engine()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ok",
        "chunks": len(index),
        "specs": index.specs,
        "embeddings": index.backend.name,
        "reranker": engine.retriever.reranker.name,
        "llm": engine.llm.name,
        "llm_available": engine.llm.available,
        "acronyms": len(index.glossary),
        "built_at": index.meta.get("built_at"),
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> JSONResponse:
    try:
        engine, _ = _engine()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    answer = engine.ask(
        request.question,
        top_k=request.k,
        spec_filter=set(request.specs) if request.specs else None,
        entailment_check=request.entailment,
    )
    return JSONResponse(answer.as_dict())


@app.get("/api/glossary/{term}")
def glossary(term: str) -> dict:
    _, index = _engine()
    return {"term": term.upper(), "expansions": index.glossary.lookup(term)}


@app.get("/api/clause/{spec_id}/{clause}")
def clause(spec_id: str, clause: str) -> dict:
    _, index = _engine()
    hits = [c for c in index.chunks if c.spec_id == spec_id and c.clause == clause]
    if not hits:
        raise HTTPException(status_code=404, detail=f"clause {clause} not found in TS {spec_id}")
    return {
        "spec_id": spec_id,
        "clause": clause,
        "version": hits[0].version,
        "heading": hits[0].heading,
        "breadcrumb": hits[0].breadcrumb,
        "text": "\n\n".join(h.text for h in sorted(hits, key=lambda h: h.part)),
    }
