"""Builds RAG components from Settings."""

from __future__ import annotations

import logging
import re

from research_agent.config import Settings
from research_agent.rag.chunking import RecursiveChunker
from research_agent.rag.embedding import SentenceTransformerEmbedder
from research_agent.rag.reranker import CrossEncoderReranker
from research_agent.rag.retrieval import HybridRetriever, RetrievalConfig
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


def retrieval_config(settings: Settings) -> RetrievalConfig:
    return RetrievalConfig(
        use_dense=settings.retrieval_use_dense,
        use_bm25=settings.retrieval_use_bm25,
        use_rerank=settings.retrieval_use_rerank,
        dense_top_k=settings.retrieval_dense_top_k,
        bm25_top_k=settings.retrieval_bm25_top_k,
        rrf_k=settings.retrieval_rrf_k,
        rerank_candidates=settings.retrieval_rerank_candidates,
        final_top_n=settings.retrieval_final_top_n,
        min_rerank_score=settings.retrieval_min_rerank_score,
    )


def build_reranker(settings: Settings) -> CrossEncoderReranker:
    return CrossEncoderReranker(
        settings.reranker_model,
        device=settings.reranker_device,
        max_length=settings.reranker_max_length,
    )


def build_retriever(settings: Settings, collection: str | None = None) -> HybridRetriever:
    """Hybrid retriever over an already-ingested collection.

    Models are only loaded for the stages that are switched on, so a BM25-only
    ablation run does not pay for the reranker.
    """
    config = retrieval_config(settings)
    embedder = build_embedder(settings)  # needed for the collection's vector dim
    store = build_store(settings, embedder.dim, collection=collection)
    store.open_existing()
    return HybridRetriever(
        store,
        config,
        embedder=embedder if config.use_dense else None,
        reranker=build_reranker(settings) if config.use_rerank else None,
    )
