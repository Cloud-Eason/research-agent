"""Sample HotpotQA and build the evaluation knowledge bases (one per chunk config).

    uv run python scripts/prepare_hotpot.py                    # 500 questions, seed 42
    uv run python scripts/prepare_hotpot.py --skip-ingest      # only (re)write the sample

Writes the sample to data/eval/ (gitignored) and creates Milvus collections named
hotpot_<model>_c<size>_o<overlap>, which the ablation YAMLs select by chunk size.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from research_agent.config import Settings, get_settings
from research_agent.eval.datasets import build_corpus, load_examples, sample_hotpot, save_examples
from research_agent.rag.factory import build_chunker, build_embedder, build_store
from research_agent.rag.pipeline import IngestionPipeline

DEFAULT_SAMPLE = Path("data/eval/hotpotqa_validation_500_s42.jsonl")


def _chunk_configs(spec: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in (item.split(":") for item in spec.split(","))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--chunks", default="64:8,128:16,256:32", help="size:overlap,...")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--reuse-sample", action="store_true", help="load --out if present")
    args = parser.parse_args()

    if args.reuse_sample and args.out.exists():
        examples = load_examples(args.out)
        print(f"loaded {len(examples)} examples from {args.out}")
    else:
        examples = sample_hotpot(args.n, args.seed)
        save_examples(args.out, examples)
        print(f"sampled {len(examples)} examples (seed={args.seed}) -> {args.out}")
    n_bridge = sum(e.type == "bridge" for e in examples)
    print(f"  bridge={n_bridge}  comparison={len(examples) - n_bridge}")

    docs = build_corpus(examples)
    print(f"corpus: {len(docs)} unique paragraphs")
    if args.skip_ingest:
        return 0

    base = get_settings()
    embedder = build_embedder(base)  # loaded once, shared by every chunk config
    lengths = [embedder.count_tokens(d.text) for d in docs[:2000]]
    print(
        f"  paragraph length (tokens, first 2000): median={statistics.median(lengths):.0f} "
        f"p90={sorted(lengths)[int(0.9 * len(lengths))]}  max={max(lengths)}"
    )

    for size, overlap in _chunk_configs(args.chunks):
        data = base.model_dump()
        data.update(
            kb_collection_prefix="hotpot",
            kb_collection=None,
            chunk_size=size,
            chunk_overlap=overlap,
        )
        settings = Settings.model_validate(data)
        store = build_store(settings, embedder.dim)
        store.ensure_collection(recreate=True)
        pipeline = IngestionPipeline(
            build_chunker(settings, embedder),
            embedder,
            store,
            prepend_title=settings.chunk_prepend_title,
            replace_existing=False,
        )
        stats = pipeline.ingest(docs)
        print(
            f"  {store.collection}: {stats.chunks} chunks from {stats.documents} paragraphs "
            f"in {stats.seconds:.0f}s (truncated={stats.truncated_chunks}, "
            f"rows={store.count()})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
