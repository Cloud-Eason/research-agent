"""Command-line entry point.

    research-agent ask "Who directed the film that won Best Picture in 1998?"
    research-agent chat --thread-id demo -v

Phase 1 uses LangGraph's in-memory checkpointer, so multi-turn memory lasts for
one process. Phase 3 swaps in the Redis checkpointer for crash recovery.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.agent.factory import build_default_deps
from research_agent.agent.graph import build_graph, recursion_limit_for, tool_calls_in_last_turn
from research_agent.agent.state import FinalAnswer
from research_agent.config import ConfigError, get_settings


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="research-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--thread-id", default=None, help="conversation id (default: random)")
    common.add_argument("--max-steps", type=int, default=None, help="override MAX_AGENT_STEPS")
    common.add_argument("-v", "--verbose", action="store_true", help="show tool trajectory")

    ask = sub.add_parser("ask", parents=[common], help="answer one question")
    ask.add_argument("question")
    sub.add_parser("chat", parents=[common], help="interactive multi-turn session")
    return parser.parse_args(argv)


def _run_turn(graph: Any, question: str, config: dict[str, Any], verbose: bool) -> None:
    started = time.perf_counter()
    result = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
    elapsed = time.perf_counter() - started
    final = FinalAnswer.model_validate(result["final"])

    if verbose:
        for i, call in enumerate(tool_calls_in_last_turn(result["messages"]), 1):
            print(f"  [{i}] {call['name']}({call['args']})")
    print(f"\n{result['messages'][-1].content}\n")
    print(f"short answer: {final.short_answer}")
    if final.invalid_citation_ids:
        print(f"dropped invalid citations: {final.invalid_citation_ids}")
    if final.hit_step_limit:
        print("note: step limit reached")
    usage = result.get("token_usage", {})
    print(
        f"steps={result.get('step')}  latency={elapsed:.1f}s  "
        f"thread tokens in/out={usage.get('input_tokens', 0)}/{usage.get('output_tokens', 0)}"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    if args.max_steps is not None:
        settings = settings.model_copy(update={"max_agent_steps": args.max_steps})

    try:
        deps = build_default_deps(settings)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    graph = build_graph(deps, checkpointer=InMemorySaver())
    thread_id = args.thread_id or uuid.uuid4().hex[:8]
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit_for(settings.max_agent_steps),
    }
    print(f"thread_id={thread_id}  model={settings.llm_model}")

    if args.command == "ask":
        _run_turn(graph, args.question, config, args.verbose)
        return 0

    print("Type a question (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            break
        _run_turn(graph, question, config, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
