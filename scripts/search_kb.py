"""Query the knowledge base through each retrieval channel separately.

    uv run python scripts/search_kb.py "what is a checkpointer" --k 5
    uv run python scripts/search_kb.py "BM25 scoring" --mode bm25

Phase 2a sanity check; Phase 2b adds the fused + reranked retriever.
"""

from __future__ import annotations

import argparse
import time

from research_agent.config import get_settings
from research_agent.rag.factory import build_embedder, build_store
from research_agent.rag.models import ScoredChunk


def _show(title: str, hits: list[ScoredChunk], elapsed: float) -> None:
    print(f"\n=== {title}  ({elapsed * 1000:.0f} ms) ===")
    for rank, hit in enumerate(hits, 1):
        c = hit.chunk
        where = f"{c.source}" + (f" p.{c.page}" if c.page else "")
        preview = " ".join(c.text.split())[:160]
        print(f"{rank:>2}. {hit.score:.4f}  [{c.title}] {where}\n    {preview}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mode", choices=["dense", "bm25", "both"], default="both")
    parser.add_argument("--collection", default=None)
    args = parser.parse_args()

    settings = get_settings()
    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dim, collection=args.collection)
    store.ensure_collection()
    print(f"collection={store.collection}  rows={store.count()}")

    if args.mode in ("dense", "both"):
        t = time.perf_counter()
        hits = store.search_dense(embedder.embed_query(args.query), args.k)
        _show("dense", hits, time.perf_counter() - t)
    if args.mode in ("bm25", "both"):
        t = time.perf_counter()
        hits = store.search_bm25(args.query, args.k)
        _show("bm25", hits, time.perf_counter() - t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
