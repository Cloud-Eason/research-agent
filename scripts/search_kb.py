"""Inspect every retrieval stage for one query.

    uv run python scripts/search_kb.py "how does checkpointing enable fault tolerance"
    uv run python scripts/search_kb.py "ReAct" --no-rerank --k 3
    uv run python scripts/search_kb.py "query" --no-dense      # BM25 only

Prints the dense, BM25, fused and reranked lists with per-stage latency, which
is the quickest way to see what each stage contributes.
"""

from __future__ import annotations

import argparse
from typing import Any

from research_agent.config import get_settings
from research_agent.rag.factory import build_retriever
from research_agent.rag.models import ScoredChunk


def _show(name: str, hits: list[ScoredChunk], k: int, ms: float | None) -> None:
    timing = f"  ({ms:.0f} ms)" if ms is not None else ""
    print(f"\n=== {name}{timing} ===")
    for rank, hit in enumerate(hits[:k], 1):
        c = hit.chunk
        where = c.source.rsplit("/", 1)[-1] + (f" p.{c.page}" if c.page else "")
        body = " ".join(c.text.split())
        if body.startswith(c.title):
            body = body[len(c.title) :].strip()
        print(f"{rank:>2}. {hit.score:8.4f}  {where}  [{c.chunk_id}]\n    {body[:150]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5, help="rows to print per stage")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--no-dense", action="store_true")
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--min-score", type=float, default=None, help="rerank score cut-off")
    args = parser.parse_args()

    updates: dict[str, Any] = {
        "retrieval_use_dense": not args.no_dense,
        "retrieval_use_bm25": not args.no_bm25,
        "retrieval_use_rerank": not args.no_rerank,
        "retrieval_min_rerank_score": args.min_score,
    }
    settings = get_settings().model_copy(update=updates)
    retriever = build_retriever(settings, collection=args.collection)
    store: Any = retriever.store
    print(f"collection={store.collection}  pipeline={retriever.config.label}")

    result = retriever.retrieve(args.query, top_n=args.k)
    for stage in ("dense", "bm25", "fused", "candidates", "reranked"):
        if stage in result.stages:
            timing_key = {"fused": "fuse", "reranked": "rerank"}.get(stage, stage)
            _show(stage, result.stages[stage], args.k, result.timings_ms.get(timing_key))
    _show("final", result.hits, args.k, sum(result.timings_ms.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
