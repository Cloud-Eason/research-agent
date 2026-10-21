from __future__ import annotations

from collections.abc import Sequence

from research_agent.rag.chunking import RecursiveChunker
from research_agent.rag.models import Chunk, Document
from research_agent.rag.pipeline import IngestionPipeline
from tests.test_chunking import CharTokenizer


class FakeEmbedder:
    dim = 3
    max_tokens = 12

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t)), 0.0, 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def count_tokens(self, text: str) -> int:
        return len(text)


class RecordingSink:
    def __init__(self) -> None:
        self.inserted: list[Chunk] = []
        self.deleted: list[list[str]] = []
        self.insert_calls = 0

    def insert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        assert len(chunks) == len(vectors)
        self.inserted.extend(chunks)
        self.insert_calls += 1

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        self.deleted.append(list(doc_ids))


def _pipeline(sink: RecordingSink, **kwargs: object) -> IngestionPipeline:
    chunker = RecursiveChunker(CharTokenizer(), chunk_size=10, chunk_overlap=0)
    return IngestionPipeline(
        chunker,
        FakeEmbedder(),
        sink,
        prepend_title=False,
        **kwargs,  # type: ignore[arg-type]
    )


def test_ingest_counts_and_batches() -> None:
    sink = RecordingSink()
    docs = [
        Document(doc_id=f"d{i}", title="t", source="s", text="aaaa bbbb cccc") for i in range(5)
    ]
    stats = _pipeline(sink, batch_size=4).ingest(docs)

    assert stats.documents == 5
    assert stats.chunks == len(sink.inserted) == 10  # 2 chunks per doc
    assert sink.insert_calls == 3  # 4 + 4 + 2


def test_each_doc_deleted_once_even_across_batches() -> None:
    sink = RecordingSink()
    pages = [
        Document(doc_id="pdf1", title="t", source="a.pdf", text="aaaa bbbb cccc", page=p)
        for p in (1, 2, 3)
    ]
    _pipeline(sink, batch_size=2).ingest(pages)

    flat = [d for call in sink.deleted for d in call]
    assert flat == ["pdf1"]
    assert {c.page for c in sink.inserted} == {1, 2, 3}


def test_replace_existing_false_skips_deletes() -> None:
    sink = RecordingSink()
    _pipeline(sink, replace_existing=False).ingest(
        [Document(doc_id="d", title="t", source="s", text="hello")]
    )
    assert sink.deleted == []


def test_truncated_chunks_are_counted() -> None:
    sink = RecordingSink()
    chunker = RecursiveChunker(CharTokenizer(), chunk_size=20, chunk_overlap=0)
    pipeline = IngestionPipeline(chunker, FakeEmbedder(), sink, prepend_title=False)
    stats = pipeline.ingest([Document(doc_id="d", title="t", source="s", text="x" * 18)])
    assert stats.truncated_chunks == 1  # 18 tokens > FakeEmbedder.max_tokens (12)
