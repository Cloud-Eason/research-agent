"""Builds RAG components from Settings."""

from __future__ import annotations

import logging
import re

from research_agent.config import Settings
from research_agent.rag.chunking import RecursiveChunker
from research_agent.rag.embedding import SentenceTransformerEmbedder
from research_agent.rag.store import MilvusStore

log = logging.getLogger(__name__)


def collection_name_for(settings: Settings) -> str:
    """e.g. kb_bge_small_en_v1_5_c256_o32. Each embedding/chunk config gets its own
    collection, so chunk-size ablations switch collections instead of re-ingesting."""
    if settings.kb_collection:
        return settings.kb_collection
    model = settings.embedding_model.split("/")[-1].lower()
    slug = re.sub(r"[^a-z0-9]+", "_", model).strip("_")
    size, overlap = settings.chunk_size, settings.chunk_overlap
    return f"{settings.kb_collection_prefix}_{slug}_c{size}_o{overlap}"


def build_embedder(settings: Settings) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        query_instruction=settings.embedding_query_instruction,
        batch_size=settings.embedding_batch_size,
    )


def build_chunker(settings: Settings, embedder: SentenceTransformerEmbedder) -> RecursiveChunker:
    if settings.chunk_size > embedder.max_tokens:
        log.warning(
            "chunk_size=%d exceeds %s max input of %d tokens; dense vectors will only "
            "see the first %d tokens of each chunk (BM25 still sees all of it)",
            settings.chunk_size,
            settings.embedding_model,
            embedder.max_tokens,
            embedder.max_tokens,
        )
    return RecursiveChunker(embedder.tokenizer, settings.chunk_size, settings.chunk_overlap)


def build_store(settings: Settings, dim: int, collection: str | None = None) -> MilvusStore:
    return MilvusStore(
        uri=settings.milvus_uri,
        collection=collection or collection_name_for(settings),
        dim=dim,
        token=settings.milvus_token.get_secret_value(),
        analyzer=settings.bm25_analyzer,
    )
