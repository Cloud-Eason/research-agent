"""Recursive text splitter that measures length in the embedding model's tokens.

Measuring in the embedder's own tokens (instead of characters or a different
tokenizer such as tiktoken) means `chunk_size` maps directly onto what the
model sees, so chunks are never silently truncated as long as
chunk_size <= the model's max sequence length.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from research_agent.rag.models import Chunk, Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "。", "; ", " ")


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, tokens: list[int]) -> str: ...


class RecursiveChunker:
    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int,
        chunk_overlap: int,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        return [c.strip() for c in self._split(text, self.separators) if c.strip()]

    # -- internals ---------------------------------------------------------
    def _split(self, text: str, separators: Sequence[str]) -> list[str]:
        if self.count(text) <= self.chunk_size:
            return [text]
        for i, sep in enumerate(separators):
            if sep not in text:
                continue
            pieces = [p for p in text.split(sep) if p.strip()]
            remaining = separators[i + 1 :]
            out: list[str] = []
            fitting: list[str] = []
            for piece in pieces:
                if self.count(piece) <= self.chunk_size:
                    fitting.append(piece)
                    continue
                if fitting:
                    out.extend(self._merge(fitting, sep))
                    fitting = []
                out.extend(self._split(piece, remaining))
            if fitting:
                out.extend(self._merge(fitting, sep))
            return out
        return self._hard_split(text)

    def _merge(self, pieces: Sequence[str], sep: str) -> list[str]:
        """Greedily pack pieces up to chunk_size, carrying ~chunk_overlap tokens forward.

        Lengths are measured on the joined text rather than summed per piece:
        subword tokenizers do not count separators consistently, and summing
        lets chunks drift past the limit.
        """
        chunks: list[str] = []
        window: list[str] = []
        for piece in pieces:
            if window and self.count(sep.join([*window, piece])) > self.chunk_size:
                chunks.append(sep.join(window))
                # Drop from the front until what is left fits the overlap budget
                # and leaves room for the incoming piece.
                while window and (
                    self.count(sep.join(window)) > self.chunk_overlap
                    or self.count(sep.join([*window, piece])) > self.chunk_size
                ):
                    window.pop(0)
            window.append(piece)
        if window:
            chunks.append(sep.join(window))
        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Last resort for text with no separators: fixed token windows."""
        tokens = self.tokenizer.encode(text)
        stride = self.chunk_size - self.chunk_overlap
        last_start = max(len(tokens) - self.chunk_overlap, 1)
        return [
            self.tokenizer.decode(tokens[i : i + self.chunk_size])
            for i in range(0, last_start, stride)
        ]


def chunk_document(doc: Document, chunker: RecursiveChunker, *, prepend_title: bool) -> list[Chunk]:
    """Split one document into chunks with deterministic ids.

    With prepend_title, each chunk starts with the document title. Later chunks
    of a page often never name their subject ("He was born in ..."), so the
    title gives both dense and BM25 retrieval the missing entity.
    """
    page_part = f"p{doc.page}" if doc.page is not None else "p0"
    chunks: list[Chunk] = []
    for position, body in enumerate(chunker.split_text(doc.text)):
        text = f"{doc.title}\n{body}" if prepend_title and doc.title else body
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}-{page_part}-{position:04d}",
                doc_id=doc.doc_id,
                title=doc.title,
                source=doc.source,
                page=doc.page,
                position=position,
                text=text,
            )
        )
    return chunks
