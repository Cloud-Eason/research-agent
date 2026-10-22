"""Replay eval questions and show where an answer was won or lost.

uv run python scripts/diagnose_question.py configs/ablations/agent_dev.yaml hybrid_bge QID [QID ...]

Runs each question through the same settings as the eval run, with an in-memory
checkpointer so the agent's draft (which finalize replaces) can be read back. For
every retrieved source it marks SUP (a supporting paragraph) and ANSWER (contains
the gold answer), and lists which supporting sentences were retrieved.
Re-running calls the LLM again; answers can differ from the recorded run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.agent.factory import build_default_deps
from research_agent.agent.graph import build_graph, recursion_limit_for
from research_agent.config import get_settings
from research_agent.eval.datasets import load_examples
from research_agent.eval.metrics import normalize_answer, sentences_all_found
from research_agent.eval.spec import load_ablation, settings_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", type=Path)
    parser.add_argument("run")
    parser.add_argument("qids", nargs="+")
    args = parser.parse_args()

    spec = load_ablation(args.spec)
    run = next((r for r in spec.runs if r.name == args.run), None)
    if run is None:
        raise SystemExit(f"unknown run {args.run!r}; have {[r.name for r in spec.runs]}")
    settings = settings_for(get_settings(), spec, run)
    graph = build_graph(build_default_deps(settings), checkpointer=InMemorySaver())
    examples = {e.qid: e for e in load_examples(spec.sample_path)}

    for qid in args.qids:
        ex = examples[qid]
        config = {
            "configurable": {"thread_id": qid},
            "recursion_limit": recursion_limit_for(settings.max_agent_steps),
        }
        result = graph.invoke({"messages": [HumanMessage(content=ex.question)]}, config)
        draft = next(
            (
                s.values["messages"][-1].content
                for s in graph.get_state_history(config)
                if s.next == ("finalize",)
            ),
            "(none)",
        )
        kb = [s for s in result["sources"] if s["tool"] == "kb_search"]
        gold = normalize_answer(ex.answer)

        print("=" * 100)
        print(f"{qid} [{ex.type}] {ex.question}")
        print(f"gold: {ex.answer}   supporting: {list(ex.supporting_titles)}")
        for m in result["messages"]:
            for call in getattr(m, "tool_calls", None) or []:
                print(f"  call {call['name']}({call['args']})")
        for s in result["sources"]:
            sup = "SUP" if s["title"] in ex.supporting_titles else ""
            hit = "ANSWER" if gold and gold in normalize_answer(s["content"]) else ""
            print(f"  {s['id']:>4} {sup:3} {hit:6} {s['title']}  <- {s['query']!r}")
        for sentence in ex.supporting_sentences():
            found = sentences_all_found([sentence], [s["content"] for s in kb]) == 1.0
            print(f"  sentence {'found  ' if found else 'MISSING'} {sentence.strip()[:110]!r}")
        print(f"--- draft:\n{draft}")
        print(f"--- final short_answer: {result['final']['short_answer']}")
        print(f"--- final answer: {result['final']['answer']}")


if __name__ == "__main__":
    main()
