"""Layer 1: retrieval-only evaluation (no LLM calls).

The raw question is the query. One run of the full pipeline yields the ranking
of every stage (dense, bm25, fused, candidates, reranked), so a single pass
measures what each stage contributes. `candidates` is the top
`rerank_candidates` *chunks* of the fused list, i.e. exactly what the reranker
re-orders: fused -> candidates is the truncation loss (larger for small chunks,
which cover fewer paragraphs per chunk), and candidates -> reranked is the
reranker's own effect. candidates and reranked hold the same chunks, so their
recall must agree once k >= rerank_candidates (a built-in correctness check).
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from research_agent.eval.datasets import EvalExample
from research_agent.eval.metrics import all_found_at_k, dedupe_titles, recall_at_k
from research_agent.eval.records import completed_qids
from research_agent.rag.retrieval import RetrievalResult

KS = (1, 2, 5, 10, 20)


class Retriever(Protocol):
    def retrieve(self, query: str, top_n: int | None = None) -> RetrievalResult: ...


def score_retrieval(
    example: EvalExample, result: RetrievalResult, top_k: int, ks: Sequence[int] = KS
) -> dict[str, Any]:
    gold = list(example.supporting_titles)
    stages: dict[str, Any] = {}
    for name, hits in result.stages.items():
        titles = dedupe_titles(h.chunk.title for h in hits)[:top_k]
        stages[name] = {
            "titles": titles,
            **{f"recall@{k}": recall_at_k(titles, gold, k) for k in ks},
            **{f"all@{k}": all_found_at_k(titles, gold, k) for k in ks},
        }
    return {
        "qid": example.qid,
        "question": example.question,
        "type": example.type,
        "level": example.level,
        "gold": gold,
        "stages": stages,
        "timings_ms": result.timings_ms,
    }


def run_retrieval_eval(
    retriever: Retriever,
    examples: Sequence[EvalExample],
    out_path: Path,
    *,
    top_k: int = 20,
    log_every: int = 25,
) -> int:
    """Append one JSON line per question; questions already in the file are skipped."""
    done = completed_qids(out_path)
    todo = [ex for ex in examples if ex.qid not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as f:
        for i, ex in enumerate(todo, 1):
            try:
                record = score_retrieval(ex, retriever.retrieve(ex.question, top_n=top_k), top_k)
            except Exception as exc:
                record = {"qid": ex.qid, "error": f"{type(exc).__name__}: {exc}"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            if i % log_every == 0 or i == len(todo):
                rate = i / (time.perf_counter() - started)
                print(f"  [{i}/{len(todo)}] {rate:.1f} q/s", flush=True)
    return len(todo)
