from __future__ import annotations

from collections.abc import Sequence

import pytest

from research_agent.agent.tools import DirectToolExecutor, make_kb_search_tool
from research_agent.rag.models import Chunk, ScoredChunk
from research_agent.rag.retrieval import HybridRetriever, RetrievalConfig, rrf_fuse


def hit(
    cid: str, score: float = 1.0, channel: str = "dense", page: int | None = None
) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=cid,
        doc_id=cid.split("-")[0],
        title=f"T-{cid}",
        source=f"/data/{cid}.pdf",
        page=page,
        position=0,
        text=f"text of {cid}",
    )
    return ScoredChunk(chunk=chunk, score=score, channel=channel)


class FakeStore:
    def __init__(self, dense: list[str], bm25: list[str]) -> None:
        self._dense = [hit(c, 1.0 - i / 10, "dense") for i, c in enumerate(dense)]
        self._bm25 = [hit(c, 10.0 - i, "bm25") for i, c in enumerate(bm25)]
        self.dense_calls = 0
        self.bm25_calls = 0

    def search_dense(self, vector: Sequence[float], top_k: int) -> list[ScoredChunk]:
        self.dense_calls += 1
        return self._dense[:top_k]

    def search_bm25(self, query: str, top_k: int) -> list[ScoredChunk]:
        self.bm25_calls += 1
        return self._bm25[:top_k]


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return [0.0, 1.0]


class FakeReranker:
    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.seen: list[str] = []

    def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        self.seen = [c.chunk.chunk_id for c in candidates]
        out = [
            ScoredChunk(
                chunk=c.chunk, score=self.scores.get(c.chunk.chunk_id, 0.0), channel="rerank"
            )
            for c in candidates
        ]
        return sorted(out, key=lambda h: h.score, reverse=True)


def ids(hits: Sequence[ScoredChunk]) -> list[str]:
    return [h.chunk.chunk_id for h in hits]


# ---- RRF ------------------------------------------------------------------
def test_rrf_rewards_agreement_between_lists() -> None:
    dense = [hit("a"), hit("b"), hit("c")]
    bm25 = [hit("b"), hit("d"), hit("a")]
    fused = rrf_fuse([dense, bm25], k=60)
    # b: 1/62 + 1/61 ; a: 1/61 + 1/63 -> b first, then a, then the single-list items
    assert ids(fused)[:2] == ["b", "a"]
    assert set(ids(fused)) == {"a", "b", "c", "d"}
    assert all(h.channel == "rrf" for h in fused)


def test_rrf_score_formula_and_dedup() -> None:
    fused = rrf_fuse([[hit("x")], [hit("x")]], k=60)
    assert len(fused) == 1
    assert fused[0].score == pytest.approx(2 / 61)


def test_rrf_ignores_raw_score_scale() -> None:
    # Same ranks, wildly different raw scores -> same fused order.
    small = rrf_fuse([[hit("a", 0.9), hit("b", 0.8)], [hit("b", 0.2), hit("a", 0.1)]])
    huge = rrf_fuse([[hit("a", 900), hit("b", 800)], [hit("b", 2), hit("a", 1)]])
    assert ids(small) == ids(huge)


def test_rrf_ties_are_deterministic() -> None:
    fused = rrf_fuse([[hit("z")], [hit("a")]])
    assert ids(fused) == ["a", "z"]


# ---- HybridRetriever --------------------------------------------------------
def make_retriever(
    cfg: RetrievalConfig, reranker: FakeReranker | None = None
) -> tuple[HybridRetriever, FakeStore, FakeEmbedder]:
    store = FakeStore(dense=["a", "b", "c"], bm25=["c", "d"])
    embedder = FakeEmbedder()
    retriever = HybridRetriever(
        store, cfg, embedder=embedder, reranker=reranker or FakeReranker({})
    )
    return retriever, store, embedder


def test_hybrid_without_rerank_returns_fused_order() -> None:
    retriever, store, _ = make_retriever(RetrievalConfig(use_rerank=False, final_top_n=2))
    result = retriever.retrieve("q")
    assert ids(result.hits) == ["c", "a"]  # c is in both lists
    assert set(result.stages) == {"dense", "bm25", "fused"}
    assert store.dense_calls == store.bm25_calls == 1


def test_dense_only_never_calls_bm25() -> None:
    retriever, store, embedder = make_retriever(RetrievalConfig(use_bm25=False, use_rerank=False))
    result = retriever.retrieve("q")
    assert ids(result.hits) == ["a", "b", "c"]
    assert store.bm25_calls == 0
    assert embedder.calls == 1
    assert "fused" not in result.stages


def test_bm25_only_never_embeds() -> None:
    retriever, store, embedder = make_retriever(RetrievalConfig(use_dense=False, use_rerank=False))
    assert ids(retriever.retrieve("q").hits) == ["c", "d"]
    assert embedder.calls == 0
    assert store.dense_calls == 0


def test_rerank_reorders_and_limits_candidates() -> None:
    reranker = FakeReranker({"d": 0.9, "b": 0.5})
    cfg = RetrievalConfig(rerank_candidates=3, final_top_n=2)
    retriever, _, _ = make_retriever(cfg, reranker)
    result = retriever.retrieve("q")
    assert len(reranker.seen) == 3  # only the top fused candidates are reranked
    assert result.hits[0].channel == "rerank"
    assert ids(result.hits)[0] in reranker.seen


def test_min_rerank_score_filters_irrelevant_hits() -> None:
    reranker = FakeReranker({"c": 0.95, "a": 0.02, "b": 0.01, "d": 0.03})
    cfg = RetrievalConfig(min_rerank_score=0.5)
    retriever, _, _ = make_retriever(cfg, reranker)
    assert ids(retriever.retrieve("q").hits) == ["c"]


def test_top_n_override_and_timings() -> None:
    retriever, _, _ = make_retriever(RetrievalConfig(use_rerank=False))
    result = retriever.retrieve("q", top_n=1)
    assert len(result.hits) == 1
    assert {"dense", "bm25", "fuse"} <= set(result.timings_ms)


def test_invalid_configs_rejected() -> None:
    store = FakeStore([], [])
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever(store, RetrievalConfig(use_dense=False, use_bm25=False))
    with pytest.raises(ValueError, match="embedder"):
        HybridRetriever(store, RetrievalConfig(use_rerank=False))
    with pytest.raises(ValueError, match="reranker"):
        HybridRetriever(store, RetrievalConfig(use_dense=False))


def test_config_label() -> None:
    assert RetrievalConfig().label == "dense+bm25+rerank"
    assert RetrievalConfig(use_bm25=False, use_rerank=False).label == "dense"


# ---- kb_search tool ----------------------------------------------------------
def test_kb_search_tool_maps_chunks_to_citable_hits() -> None:
    class OneHitRetriever:
        def retrieve(self, query: str, top_n: int | None = None):  # type: ignore[no-untyped-def]
            from research_agent.rag.retrieval import RetrievalResult

            hits = [hit("paper-p3-0001", 0.8, "rerank", page=3)]
            return RetrievalResult(query=query, hits=hits)

    tool = make_kb_search_tool(OneHitRetriever())
    [result] = DirectToolExecutor().execute(
        tool, {"query": "q", "top_k": 3}, thread_id="t", tool_call_id="c"
    )
    assert result.ref == "paper-p3-0001"
    assert result.location == "/data/paper-p3-0001.pdf#page=3"
    assert result.title.endswith("(p.3)")
    assert tool.to_openai_schema()["function"]["name"] == "kb_search"
