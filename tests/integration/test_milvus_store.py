"""Runs against the Docker Milvus: `uv run pytest -m integration`."""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from research_agent.rag.models import Chunk
from research_agent.rag.store import MilvusStore

pytestmark = pytest.mark.integration

DOCS = {
    "redis": "Redis is an in-memory key value store often used as a cache.",
    "milvus": "Milvus is a vector database built for similarity search at scale.",
    "nolan": "Christopher Nolan is a film director born in London.",
}
# Hand-made 4-d unit vectors: each doc points along its own axis.
VECTORS = {
    "redis": [1.0, 0.0, 0.0, 0.0],
    "milvus": [0.0, 1.0, 0.0, 0.0],
    "nolan": [0.0, 0.0, 1.0, 0.0],
}


@pytest.fixture
def store() -> Iterator[MilvusStore]:
    s = MilvusStore(
        uri=os.environ.get("MILVUS_URI", "http://localhost:19530"),
        collection=f"test_{uuid.uuid4().hex[:8]}",
        dim=4,
    )
    s.ensure_collection(recreate=True)
    chunks = [
        Chunk(
            chunk_id=f"{k}-p0-0000",
            doc_id=k,
            title=k.title(),
            source=f"{k}.txt",
            page=None,
            position=0,
            text=text,
        )
        for k, text in DOCS.items()
    ]
    s.insert(chunks, [VECTORS[c.doc_id] for c in chunks])
    yield s
    s.drop()


def test_count(store: MilvusStore) -> None:
    assert store.count() == 3


def test_dense_search_returns_nearest(store: MilvusStore) -> None:
    hits = store.search_dense([0.1, 0.9, 0.0, 0.0], top_k=2)
    assert hits[0].chunk.doc_id == "milvus"
    assert hits[0].channel == "dense"
    assert hits[0].chunk.page is None


def test_bm25_search_matches_keywords(store: MilvusStore) -> None:
    hits = store.search_bm25("which database does vector similarity search", top_k=3)
    assert hits[0].chunk.doc_id == "milvus"
    assert hits[0].score > 0


def test_bm25_ignores_empty_query(store: MilvusStore) -> None:
    assert store.search_bm25("   ", top_k=3) == []


def test_delete_document(store: MilvusStore) -> None:
    store.delete_documents(["redis"])
    assert store.count() == 2
    assert all(h.chunk.doc_id != "redis" for h in store.search_bm25("redis cache", top_k=3))


def test_dim_mismatch_is_detected(store: MilvusStore) -> None:
    other = MilvusStore(
        uri=os.environ.get("MILVUS_URI", "http://localhost:19530"),
        collection=store.collection,
        dim=8,
    )
    with pytest.raises(ValueError, match="dim"):
        other.ensure_collection()
