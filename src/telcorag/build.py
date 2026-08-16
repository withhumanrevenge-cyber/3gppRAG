"""Corpus acquisition and index construction."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from . import glossary as glossary_mod
from .config import DEFAULT_SPECS, Settings, settings as default_settings
from .corpus import download as dl
from .corpus.chunker import Chunk, chunk_spec
from .corpus.parser import Clause, parse_spec
from .index.embeddings import resolve
from .index.store import Index

MANIFEST = "manifest.json"


def fetch_corpus(
    specs: tuple[str, ...] = DEFAULT_SPECS,
    release: int | None = None,
    cfg: Settings | None = None,
    log=print,
) -> list[dict]:
    cfg = cfg or default_settings
    cfg.ensure_dirs()
    session = requests.Session()
    manifest: list[dict] = []

    for spec_id in specs:
        try:
            archive = dl.pick(dl.list_archives(spec_id, session), release)
        except Exception as exc:
            log(f"  {spec_id}: lookup failed — {type(exc).__name__}: {exc}")
            continue
        try:
            zip_path = dl.download(archive, cfg.raw_dir, session)
            parts = dl.extract(zip_path, cfg.corpus_dir)
        except Exception as exc:
            log(f"  {spec_id}: download failed — {type(exc).__name__}: {exc}")
            continue

        docx = [p for p in parts if p.suffix.lower() == ".docx"]
        if not docx:
            log(f"  {spec_id}: archive holds no .docx parts (legacy .doc), skipped")
            continue
        log(f"  {spec_id} v{archive.version} (Rel-{archive.release}) — {len(docx)} part(s)")
        manifest.append({**archive.as_dict(), "parts": [str(p) for p in docx]})

    (cfg.corpus_dir.parent / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(cfg: Settings | None = None) -> list[dict]:
    cfg = cfg or default_settings
    path = cfg.corpus_dir.parent / MANIFEST
    if not path.exists():
        raise FileNotFoundError("no manifest — run `python -m telcorag fetch` first")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_corpus(manifest: list[dict], log=print) -> tuple[list[Clause], list[Chunk]]:
    all_clauses: list[Clause] = []
    all_chunks: list[Chunk] = []

    for entry in manifest:
        paths = [Path(p) for p in entry["parts"]]
        missing = [p for p in paths if not p.exists()]
        if missing:
            log(f"  {entry['spec_id']}: {len(missing)} missing part(s), skipped")
            continue
        clauses = parse_spec(paths, entry["spec_id"], entry["version"], entry["release"])
        chunks = chunk_spec(clauses)
        all_clauses.extend(clauses)
        all_chunks.extend(chunks)
        log(f"  {entry['spec_id']}: {len(clauses):5d} clauses -> {len(chunks):5d} chunks")

    return all_clauses, all_chunks


def build_index(cfg: Settings | None = None, log=print) -> Index:
    cfg = cfg or default_settings
    cfg.ensure_dirs()

    manifest = load_manifest(cfg)
    log("Parsing specifications")
    clauses, chunks = parse_corpus(manifest, log)
    if not chunks:
        raise RuntimeError("no chunks produced — corpus is empty or unparsable")

    lexicon = glossary_mod.build(clauses)
    log(f"Glossary: {len(lexicon)} acronyms, {len(lexicon.reverse)} reverse phrases")

    backend = resolve(cfg.embedding_backend, cfg.embedding_model, cfg.lsa_dims)
    log(f"Embedding backend: {backend.name}")

    started = time.time()
    index = Index.build(
        chunks,
        backend,
        lexicon,
        meta={
            "specs": [{k: e[k] for k in ("spec_id", "version", "release")} for e in manifest],
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "acronyms": len(lexicon),
        },
    )
    log(f"Indexed {len(index)} chunks in {time.time() - started:.1f}s")

    index.save(cfg.index_dir)
    log(f"Saved to {cfg.index_dir}")
    return index
