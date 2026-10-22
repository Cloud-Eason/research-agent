"""Run an ablation (retrieval or agent layer) from a YAML spec, then write the report.

    uv run python scripts/run_eval.py configs/ablations/retrieval.yaml
    uv run python scripts/run_eval.py configs/ablations/agent.yaml --runs hybrid --limit 20
    uv run python scripts/run_eval.py configs/ablations/agent.yaml          # resumes

Results go to results/<ablation>/<run>/results.jsonl (+ manifest.json). Re-running
skips finished questions and retries failed ones; --fresh starts a run over.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

from research_agent.agent.factory import build_default_deps
from research_agent.agent.graph import build_graph, recursion_limit_for
from research_agent.config import get_settings
from research_agent.eval.agent_eval import run_agent_eval
from research_agent.eval.datasets import load_examples
from research_agent.eval.report import write_reports
from research_agent.eval.retrieval_eval import run_retrieval_eval
from research_agent.eval.spec import load_ablation, settings_for, write_manifest
from research_agent.rag.factory import build_retriever


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument("--runs", default=None, help="comma-separated run names (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--fresh", action="store_true", help="delete previous results first")
    args = parser.parse_args()

    spec = load_ablation(args.spec)
    examples = load_examples(spec.sample_path)[spec.offset : spec.offset + spec.n]
    selected = set(args.runs.split(",")) if args.runs else None
    runs = [r for r in spec.runs if selected is None or r.name in selected]
    if selected and len(runs) != len(selected):
        unknown = selected - {r.name for r in runs}
        raise SystemExit(f"unknown run(s): {sorted(unknown)}")

    base = get_settings()
    eval_dir = args.out / spec.name
    for run in runs:
        settings = settings_for(base, spec, run)
        subset = examples[: run.limit] if run.limit else examples
        if args.limit:
            subset = subset[: args.limit]
        run_dir = eval_dir / run.name
        if args.fresh and run_dir.exists():
            shutil.rmtree(run_dir)
        write_manifest(run_dir, spec, run, settings, [e.qid for e in subset])
        print(f"\n=== {spec.name}/{run.name}: {len(subset)} questions ({spec.layer}) ===")

        if spec.layer == "retrieval":
            retriever = build_retriever(settings)
            collection = getattr(retriever.store, "collection", "?")
            print(f"  collection={collection}  pipeline={retriever.config.label}")
            run_retrieval_eval(retriever, subset, run_dir / "results.jsonl", top_k=spec.top_k)
        else:
            deps = build_default_deps(settings)
            print(f"  tools={sorted(deps.tools)}  model={settings.llm_model}")
            graph = build_graph(deps)  # no checkpointer: the JSONL file is the resume point
            asyncio.run(
                run_agent_eval(
                    graph,
                    subset,
                    run_dir / "results.jsonl",
                    run_name=run.name,
                    kb_enabled=settings.enable_kb_search,
                    concurrency=args.concurrency or spec.concurrency,
                    timeout_s=spec.timeout_s,
                    recursion_limit=recursion_limit_for(settings.max_agent_steps),
                )
            )

    report = write_reports(eval_dir)
    print(f"\nreport: {report}\n")
    print(report.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
