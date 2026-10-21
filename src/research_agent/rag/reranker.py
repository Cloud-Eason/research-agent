"""Cross-encoder reranking (e.g. BAAI/bge-reranker-base)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from research_agent.rag.models import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]: ...


class CrossEncoderReranker:
    """Scores each (query, passage) pair jointly.

    Bi-encoders (the dense retriever) embed query and passage separately, which
    is fast enough to search millions of chunks but loses fine-grained
    interaction. A cross-encoder reads both together, which is far more accurate
    but too slow to run over the whole corpus, so it only re-scores the small
    candidate set produced by retrieval.
    """

    def __init__(
        self, model_name: str, *, device: str = "cpu", max_length: int = 512, batch_size: int = 16
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device, max_length=max_length)
        self._batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._model.predict(
            [(query, c.chunk.text) for c in candidates],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        rescored = [
            ScoredChunk(chunk=c.chunk, score=float(s), channel="rerank")
            for c, s in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda h: h.score, reverse=True)
