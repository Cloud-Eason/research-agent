"""Plain data types shared across the RAG pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def make_doc_id(source: str) -> str:
    """Stable id derived from the source, so re-ingesting replaces instead of duplicating."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Document:
    """One loadable unit: a PDF page, a web page, a text file, or a dataset paragraph."""

    doc_id: str
    title: str
    source: str  # file path, URL, or e.g. "hotpotqa:<title>"
    text: str
    page: int | None = None  # 1-based page number for PDFs


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    page: int | None
    position: int  # index of the chunk within its document/page
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    channel: str  # "dense" | "bm25" | "rrf" | "rerank"
