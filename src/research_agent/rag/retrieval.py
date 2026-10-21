"""Hybrid retrieval: dense + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank.

Every stage can be switched off through RetrievalConfig, and each call returns
the intermediate rankings, so one evaluation run can report Recall@k for
dense-only, BM25-only, fused and reranked lists side by side.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from research_agent.rag.models import ScoredChunk
from research_agent.rag.reranker import Reranker


class SearchableStore(Protocol):
    def search_dense(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]: ...
    def search_bm25(self, query: str, top_k: int) -> list[ScoredChunk]: ...


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class RetrievalConfig:
    use_dense: bool = True
    use_bm25: bool = True
    use_rerank: bool = True
    dense_top_k: int = 20
    bm25_top_k: int = 20
    rrf_k: int = 60
    rerank_candidates: int = 20
    final_top_n: int = 5
    min_rerank_score: float | None = None

    @property
    def label(self) -> str:
        """Short name for logs and ablation tables, e.g. 'dense+bm25+rerank'."""
        parts = [name for name, on in (("dense", self.use_dense), ("bm25", self.use_bm25)) if on]
        if self.use_rerank:
            parts.append("rerank")
        return "+".join(parts)


@dataclass
class RetrievalResult:
    query: str
    hits: list[ScoredChunk]
    stages: dict[str, list[ScoredChunk]] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


def rrf_fuse(rankings: Sequence[Sequence[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank_d).

    RRF uses ranks only, never raw scores. That matters here because cosine
    similarity (roughly 0..1) and BM25 (unbounded, corpus-dependent) are on
    incomparable scales; normalising and weighting them would need tuning per
    corpus, while RRF works out of the box. k=60 is the value from the original
    paper (Cormack et al., 2009); larger k flattens the advantage of top ranks.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(cid, hit)
    fused = [
        ScoredChunk(chunk=first_seen[cid].chunk, score=s, channel="rrf")
        for cid, s in scores.items()
    ]
    # Stable tie-break on chunk_id keeps results reproducible across runs.
    return sorted(fused, key=lambda h: (-h.score, h.chunk.chunk_id))


class HybridRetriever:
    def __init__(
        self,
        store: SearchableStore,
        config: RetrievalConfig,
        *,
        embedder: QueryEmbedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if not (config.use_dense or config.use_bm25):
            raise ValueError("at least one of use_dense / use_bm25 must be enabled")
        if config.use_dense and embedder is None:
            raise ValueError("use_dense requires an embedder")
        if config.use_rerank and reranker is None:
            raise ValueError("use_rerank requires a reranker")
        self.store = store
        self.config = config
        self.embedder = embedder
        self.reranker = reranker

    def retrieve(self, query: str, top_n: int | None = None) -> RetrievalResult:
        cfg = self.config
        n = top_n or cfg.final_top_n
        result = RetrievalResult(query=query, hits=[])
        rankings: list[list[ScoredChunk]] = []

        if cfg.use_dense:
            assert self.embedder is not None
            with _timer(result, "dense"):
                dense = self.store.search_dense(self.embedder.embed_query(query), cfg.dense_top_k)
            result.stages["dense"] = dense
            rankings.append(dense)
        if cfg.use_bm25:
            with _timer(result, "bm25"):
                bm25 = self.store.search_bm25(query, cfg.bm25_top_k)
            result.stages["bm25"] = bm25
            rankings.append(bm25)

        if len(rankings) > 1:
            with _timer(result, "fuse"):
                candidates = rrf_fuse(rankings, k=cfg.rrf_k)
            result.stages["fused"] = candidates
        else:
            candidates = rankings[0]

        if cfg.use_rerank:
            assert self.reranker is not None
            with _timer(result, "rerank"):
                reranked = self.reranker.rerank(query, candidates[: cfg.rerank_candidates])
            result.stages["reranked"] = reranked
            if cfg.min_rerank_score is not None:
                reranked = [h for h in reranked if h.score >= cfg.min_rerank_score]
            candidates = reranked

        result.hits = candidates[:n]
        return result


class _timer:
    """Records the elapsed milliseconds of a block into result.timings_ms[name]."""

    def __init__(self, result: RetrievalResult, name: str) -> None:
        self._result = result
        self._name = name
        self._start = 0.0

    def __enter__(self) -> None:
        self._start = time.perf_counter()

    def __exit__(self, *exc: object) -> None:
        self._result.timings_ms[self._name] = (time.perf_counter() - self._start) * 1000
