"""Ingest PDFs, text files, directories or URLs into the Milvus knowledge base.

uv run python scripts/ingest.py data/papers/ https://example.com/article
uv run python scripts/ingest.py data/papers/ --chunk-size 512 --overlap 64 --recreate
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator

from research_agent.config import get_settings
from research_agent.rag.factory import build_chunker, build_embedder, build_store
from research_agent.rag.loaders import LoaderError, load_any
from research_agent.rag.models import Document
from research_agent.rag.pipeline import IngestionPipeline


def _documents(targets: list[str], errors: list[str]) -> Iterator[Document]:
    for target in targets:
        try:
            yield from load_any(target)
        except LoaderError as exc:  # skip bad inputs, report at the end
            errors.append(str(exc))
            print(f"  skipped: {exc}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("targets", nargs="+", help="files, directories or URLs")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--collection", default=None, help="override the derived name")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild the collection")
    parser.add_argument("--no-title-prefix", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    updates = {
        k: v
        for k, v in {
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.overlap,
            "kb_collection": args.collection,
        }.items()
        if v is not None
    }
    if args.no_title_prefix:
        updates["chunk_prepend_title"] = False
    settings = get_settings().model_copy(update=updates)

    embedder = build_embedder(settings)
    store = build_store(settings, embedder.dim)
    store.ensure_collection(recreate=args.recreate)
    print(
        f"collection={store.collection}  dim={embedder.dim}  "
        f"chunk_size={settings.chunk_size}  overlap={settings.chunk_overlap}"
    )

    pipeline = IngestionPipeline(
        build_chunker(settings, embedder),
        embedder,
        store,
        prepend_title=settings.chunk_prepend_title,
        replace_existing=not args.recreate,  # a fresh collection has nothing to replace
    )
    errors: list[str] = []
    stats = pipeline.ingest(_documents(args.targets, errors))

    print(
        f"documents={stats.documents}  chunks={stats.chunks}  "
        f"truncated={stats.truncated_chunks}  time={stats.seconds:.1f}s  "
        f"collection_rows={store.count()}"
    )
    if errors:
        print(f"{len(errors)} input(s) skipped", file=sys.stderr)
    return 0 if stats.chunks else 1


if __name__ == "__main__":
    raise SystemExit(main())
