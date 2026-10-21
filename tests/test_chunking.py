from __future__ import annotations

import pytest

from research_agent.rag.chunking import RecursiveChunker, chunk_document
from research_agent.rag.models import Document


class CharTokenizer:
    """1 character == 1 token, so expected lengths are easy to reason about."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def chunker(size: int, overlap: int = 0) -> RecursiveChunker:
    return RecursiveChunker(CharTokenizer(), size, overlap)


def test_short_text_is_one_chunk() -> None:
    assert chunker(100).split_text("  hello world  ") == ["hello world"]


def test_empty_text_gives_no_chunks() -> None:
    assert chunker(10).split_text("   \n\n ") == []


def test_every_chunk_respects_size() -> None:
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(10))
    for size in (40, 80, 200):
        chunks = chunker(size, overlap=size // 5).split_text(text)
        assert chunks
        assert all(len(c) <= size for c in chunks), size


def test_prefers_paragraph_boundaries() -> None:
    text = "first paragraph here\n\nsecond paragraph here"
    assert chunker(25).split_text(text) == ["first paragraph here", "second paragraph here"]


def test_overlap_repeats_trailing_pieces() -> None:
    text = "aaaa bbbb cccc dddd eeee"
    chunks = chunker(14, overlap=4).split_text(text)
    assert chunks[0] == "aaaa bbbb cccc"
    assert chunks[1].startswith("cccc")  # last piece of chunk 0 carried over


def test_text_without_separators_is_hard_split_with_overlap() -> None:
    chunks = chunker(10, overlap=2).split_text("x" * 25)
    assert [len(c) for c in chunks] == [10, 10, 9]
    assert sum(len(c) for c in chunks) - 2 * (len(chunks) - 1) == 25


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        RecursiveChunker(CharTokenizer(), 10, 10)


def test_chunk_document_ids_metadata_and_title_prefix() -> None:
    doc = Document(doc_id="abc", title="Ed Wood", source="s.pdf", text="one\n\ntwo", page=3)
    chunks = chunk_document(doc, chunker(4), prepend_title=True)
    assert [c.chunk_id for c in chunks] == ["abc-p3-0000", "abc-p3-0001"]
    assert chunks[1].text == "Ed Wood\ntwo"
    assert all(c.page == 3 and c.source == "s.pdf" for c in chunks)

    plain = chunk_document(doc, chunker(4), prepend_title=False)
    assert plain[0].text == "one"


def test_chunk_ids_are_deterministic() -> None:
    doc = Document(doc_id="d1", title="t", source="s", text="alpha beta gamma delta")
    first = [c.chunk_id for c in chunk_document(doc, chunker(10), prepend_title=False)]
    second = [c.chunk_id for c in chunk_document(doc, chunker(10), prepend_title=False)]
    assert first == second
