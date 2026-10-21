"""Milvus collection holding dense vectors, BM25 sparse vectors and metadata.

Schema (one collection per embedding model + chunk config):

    chunk_id  VARCHAR  primary key, deterministic ("<doc_id>-p<page>-<pos>")
    doc_id    VARCHAR  INVERTED scalar index (fast delete-by-document)
    title, source      VARCHAR
    page, position     INT64 (page = -1 when not applicable)
    text      VARCHAR  analyzer enabled -> input of the BM25 function
    dense     FLOAT_VECTOR   HNSW, COSINE
    sparse    SPARSE_FLOAT_VECTOR  produced server-side by the BM25 function

BM25 runs inside Milvus (2.5+): the server tokenizes `text`, maintains term
statistics and scores queries, so no separate keyword index has to be kept in
sync with the vector store.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient

from research_agent.rag.models import Chunk, ScoredChunk

OUTPUT_FIELDS = ["chunk_id", "doc_id", "title", "source", "page", "position", "text"]
_TEXT_MAX_BYTES = 65_000


class MilvusStore:
    def __init__(
        self,
        uri: str,
        collection: str,
        dim: int,
        *,
        token: str = "",
        analyzer: str = "standard",
        insert_batch_size: int = 256,
    ) -> None:
        self._client = MilvusClient(uri=uri, token=token) if token else MilvusClient(uri=uri)
        self.collection = collection
        self.dim = dim
        self._analyzer = analyzer
        self._batch = insert_batch_size

    # ---- lifecycle --------------------------------------------------------
    def ensure_collection(self, *, recreate: bool = False) -> None:
        client = self._client
        if recreate and client.has_collection(self.collection):
            client.drop_collection(self.collection)
        if client.has_collection(self.collection):
            self._check_dim()
        else:
            client.create_collection(
                collection_name=self.collection,
                schema=self._schema(),
                index_params=self._index_params(),
                # Single local node: read-your-writes is worth the small latency cost.
                consistency_level="Strong",
            )
        client.load_collection(self.collection)

    def drop(self) -> None:
        if self._client.has_collection(self.collection):
            self._client.drop_collection(self.collection)

    def count(self) -> int:
        rows = self._client.query(self.collection, filter="", output_fields=["count(*)"])
        return int(rows[0]["count(*)"]) if rows else 0

    # ---- writes -----------------------------------------------------------
    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        rows = [_to_row(c, v) for c, v in zip(chunks, vectors, strict=True)]
        for start in range(0, len(rows), self._batch):
            self._client.insert(self.collection, rows[start : start + self._batch])

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        if doc_ids:
            self._client.delete(self.collection, filter=f"doc_id in {json.dumps(list(doc_ids))}")

    # ---- reads ------------------------------------------------------------
    def search_dense(
        self, vector: Sequence[float], top_k: int, *, expr: str = ""
    ) -> list[ScoredChunk]:
        result = self._client.search(
            collection_name=self.collection,
            data=[list(vector)],
            anns_field="dense",
            limit=top_k,
            filter=expr,
            search_params={"metric_type": "COSINE", "params": {"ef": max(64, top_k)}},
            output_fields=OUTPUT_FIELDS,
        )
        return [_to_scored(hit, "dense") for hit in result[0]]

    def search_bm25(self, query: str, top_k: int, *, expr: str = "") -> list[ScoredChunk]:
        if not query.strip():
            return []
        result = self._client.search(
            collection_name=self.collection,
            data=[query],  # raw text; Milvus applies the same analyzer + BM25
            anns_field="sparse",
            limit=top_k,
            filter=expr,
            search_params={"metric_type": "BM25"},
            output_fields=OUTPUT_FIELDS,
        )
        return [_to_scored(hit, "bm25") for hit in result[0]]

    # ---- schema -----------------------------------------------------------
    def _schema(self) -> Any:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=256)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("title", DataType.VARCHAR, max_length=1024)
        schema.add_field("source", DataType.VARCHAR, max_length=2048)
        schema.add_field("page", DataType.INT64)
        schema.add_field("position", DataType.INT64)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=65_535,
            enable_analyzer=True,
            analyzer_params={"type": self._analyzer},
        )
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse"],
            )
        )
        return schema

    def _index_params(self) -> Any:
        params = self._client.prepare_index_params()
        params.add_index(
            field_name="dense",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
        )
        params.add_index(field_name="doc_id", index_type="INVERTED")
        return params

    def _check_dim(self) -> None:
        info = self._client.describe_collection(self.collection)
        for field in info.get("fields", []):
            if field.get("name") == "dense":
                existing = int(field.get("params", {}).get("dim", 0))
                if existing != self.dim:
                    raise ValueError(
                        f"collection '{self.collection}' has dim {existing}, embedder has "
                        f"{self.dim}; use a different collection or --recreate"
                    )


def _truncate_utf8(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    return text if len(raw) <= max_bytes else raw[:max_bytes].decode("utf-8", errors="ignore")


def _to_row(chunk: Chunk, vector: Sequence[float]) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": _truncate_utf8(chunk.title, 1000),
        "source": _truncate_utf8(chunk.source, 2000),
        "page": chunk.page if chunk.page is not None else -1,
        "position": chunk.position,
        "text": _truncate_utf8(chunk.text, _TEXT_MAX_BYTES),
        "dense": [float(x) for x in vector],
        # "sparse" is omitted on purpose: the BM25 function fills it server-side.
    }


def _to_scored(hit: Any, channel: str) -> ScoredChunk:
    entity = hit["entity"]
    page = int(entity["page"])
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=str(entity["chunk_id"]),
            doc_id=str(entity["doc_id"]),
            title=str(entity["title"]),
            source=str(entity["source"]),
            page=page if page >= 0 else None,
            position=int(entity["position"]),
            text=str(entity["text"]),
        ),
        score=float(hit["distance"]),
        channel=channel,
    )
