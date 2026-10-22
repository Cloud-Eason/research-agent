"""Layer 2: end-to-end agent evaluation with bounded concurrency and resume.

Each question runs in its own thread_id with no checkpointer: questions are
independent, and resumability comes from the JSONL file (finished qids are
skipped, errored ones retried), which is cheaper than persisting every step.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from langchain_core.messages import HumanMessage

from research_agent.agent.graph import tool_calls_in_last_turn
from research_agent.agent.state import FinalAnswer
from research_agent.eval.datasets import EvalExample
from research_agent.eval.metrics import (
    all_found_at_k,
    dedupe_titles,
    exact_match,
    f1_score,
    recall_at_k,
    sentences_all_found,
    trajectory_metrics,
)
from research_agent.eval.records import completed_qids


class AsyncGraph(Protocol):
    async def ainvoke(self, input: Any, config: Any, /) -> Any: ...


def score_agent_result(
    example: EvalExample, result: dict[str, Any], latency_s: float, *, kb_enabled: bool
) -> dict[str, Any]:
    final = FinalAnswer.model_validate(result["final"])
    calls = tool_calls_in_last_turn(result["messages"])
    gold_titles = list(example.supporting_titles)
    # Evidence the agent actually retrieved, over all of its (rewritten) queries.
    kb_sources = [s for s in result.get("sources", []) if s["tool"] == "kb_search"]
    kb_titles = dedupe_titles(s["title"] for s in kb_sources)
    usage = result.get("token_usage", {})
    return {
        "qid": example.qid,
        "question": example.question,
        "type": example.type,
        "level": example.level,
        "gold": example.answer,
        "prediction": final.short_answer,
        "answer": final.answer,
        "em": exact_match(final.short_answer, example.answer),
        "f1": f1_score(final.short_answer, example.answer),
        "steps": result.get("step"),
        "hit_step_limit": final.hit_step_limit,
        "n_citations": len(final.citations),
        "n_invalid_citations": len(final.invalid_citation_ids),
        "evidence_recall": recall_at_k(kb_titles, gold_titles, len(kb_titles)),
        "evidence_all_found": all_found_at_k(kb_titles, gold_titles, len(kb_titles)),
        "evidence_sentences_all_found": sentences_all_found(
            example.supporting_sentences(), [s["content"] for s in kb_sources]
        ),
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "llm_calls": int(usage.get("llm_calls", 0)),
        "latency_s": round(latency_s, 3),
        "tool_calls": calls,
        **trajectory_metrics(
            example.question, example.type, gold_titles, calls, kb_enabled=kb_enabled
        ),
    }


async def run_agent_eval(
    graph: AsyncGraph,
    examples: Sequence[EvalExample],
    out_path: Path,
    *,
    run_name: str,
    kb_enabled: bool,
    concurrency: int = 4,
    timeout_s: float = 240.0,
    recursion_limit: int = 25,
) -> int:
    done = completed_qids(out_path)
    todo = [ex for ex in examples if ex.qid not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    finished = 0
    started = time.perf_counter()

    with out_path.open("a", encoding="utf-8") as f:

        async def one(ex: EvalExample) -> None:
            nonlocal finished
            async with semaphore:
                config = {
                    "configurable": {"thread_id": f"eval-{run_name}-{ex.qid}"},
                    "recursion_limit": recursion_limit,
                }
                t0 = time.perf_counter()
                try:
                    result = await asyncio.wait_for(
                        graph.ainvoke({"messages": [HumanMessage(content=ex.question)]}, config),
                        timeout=timeout_s,
                    )
                    record = score_agent_result(
                        ex, result, time.perf_counter() - t0, kb_enabled=kb_enabled
                    )
                except Exception as exc:  # recorded and retried on the next run
                    record = {"qid": ex.qid, "error": f"{type(exc).__name__}: {exc}"}
            async with write_lock:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                finished += 1
                if finished % 10 == 0 or finished == len(todo):
                    elapsed = time.perf_counter() - started
                    print(f"  [{finished}/{len(todo)}] {elapsed:.0f}s elapsed", flush=True)

        await asyncio.gather(*(one(ex) for ex in todo))
    return len(todo)
