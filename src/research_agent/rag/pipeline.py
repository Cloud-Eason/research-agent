"""Ingestion: documents -> chunks -> embeddings -> Milvus."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from research_agent.rag.chunking import RecursiveChunker, chunk_document
from research_agent.rag.embedding import Embedder
from research_agent.rag.models import Chunk, Document


class ChunkSink(Protocol):
    """The part of MilvusStore the pipeline needs (kept small for testing)."""

    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...
    def delete_documents(self, doc_ids: Sequence[str]) -> None: ...


@dataclass
class IngestStats:
    documents: int = 0
    chunks: int = 0
    truncated_chunks: int = 0  # chunks longer than the embedder's max input
    seconds: float = 0.0


class IngestionPipeline:
    def __init__(
        self,
        chunker: RecursiveChunker,
        embedder: Embedder,
        sink: ChunkSink,
        *,
        prepend_title: bool = True,
        batch_size: int = 64,
        replace_existing: bool = True,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.sink = sink
        self.prepend_title = prepend_title
        self.batch_size = batch_size
        self.replace_existing = replace_existing
        self._deleted: set[str] = set()

    def ingest(self, documents: Iterable[Document]) -> IngestStats:
        started = time.perf_counter()
        stats = IngestStats()
        buffer: list[Chunk] = []
        for doc in documents:
            stats.documents += 1
            buffer.extend(chunk_document(doc, self.chunker, prepend_title=self.prepend_title))
            if len(buffer) >= self.batch_size:
                self._flush(buffer, stats)
                buffer = []
        if buffer:
            self._flush(buffer, stats)
        stats.seconds = time.perf_counter() - started
        return stats

    def _flush(self, chunks: list[Chunk], stats: IngestStats) -> None:
        # Re-ingesting a source replaces its old chunks. Each doc_id is deleted
        # once per run, so a multi-page PDF split across batches keeps all pages.
        if self.replace_existing:
            new_ids = sorted({c.doc_id for c in chunks} - self._deleted)
            self.sink.delete_documents(new_ids)
            self._deleted.update(new_ids)
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        stats.truncated_chunks += sum(
            1 for c in chunks if self.embedder.count_tokens(c.text) > self.embedder.max_tokens
        )
        self.sink.insert(chunks, vectors)
        stats.chunks += len(chunks)
