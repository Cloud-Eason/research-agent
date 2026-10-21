"""Command-line entry point.

    research-agent ask "Who directed the film that won Best Picture in 1998?"
    research-agent chat --thread-id demo -v
    research-agent resume --thread-id demo     # continue a run that was interrupted

With CHECKPOINTER=redis (default) conversations survive process restarts: the
same --thread-id picks up the stored history, and `resume` finishes a run that
crashed mid-way from its last checkpoint. CHECKPOINTER=memory keeps everything
in-process (Phase 1 behaviour).
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any

import redis
from langchain_core.messages import HumanMessage

from research_agent.agent.factory import build_default_deps
from research_agent.agent.graph import build_graph, recursion_limit_for, tool_calls_in_last_turn
from research_agent.agent.state import FinalAnswer
from research_agent.checkpoint.factory import build_persistence
from research_agent.config import ConfigError, get_settings


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="research-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--thread-id", default=None, help="conversation id (default: random)")
    common.add_argument("--max-steps", type=int, default=None, help="override MAX_AGENT_STEPS")
    common.add_argument("-v", "--verbose", action="store_true", help="show tool trajectory")
    common.add_argument("--no-web", action="store_true", help="disable web_search")
    common.add_argument("--no-kb", action="store_true", help="disable kb_search")

    ask = sub.add_parser("ask", parents=[common], help="answer one question")
    ask.add_argument("question")
    sub.add_parser("chat", parents=[common], help="interactive multi-turn session")
    sub.add_parser("resume", parents=[common], help="finish an interrupted run")
    return parser.parse_args(argv)


def print_result(result: dict[str, Any], elapsed: float, verbose: bool) -> None:
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


def invoke_durably(
    graph: Any, payload: dict[str, Any] | None, config: dict[str, Any]
) -> dict[str, Any]:
    """Run the graph with synchronous checkpointing.

    durability="sync" persists each checkpoint before the next step starts. With
    the default async mode a crash can lose the newest checkpoint; the LLM step
    then re-runs, emits new tool_call_ids, and the tool cache no longer matches.
    Passing None as payload resumes from the thread's last checkpoint.
    """
    result: dict[str, Any] = graph.invoke(payload, config, durability="sync")
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    updates: dict[str, Any] = {}
    if args.max_steps is not None:
        updates["max_agent_steps"] = args.max_steps
    if args.no_web:
        updates["enable_web_search"] = False
    if args.no_kb:
        updates["enable_kb_search"] = False
    settings = get_settings().model_copy(update=updates)

    try:
        checkpointer, executor = build_persistence(settings)
        deps = build_default_deps(settings, executor=executor)
    except redis.exceptions.ConnectionError as exc:
        print(f"setup error: cannot reach Redis at {settings.redis_url}: {exc}", file=sys.stderr)
        print("start it (docker compose up -d redis) or set CHECKPOINTER=memory", file=sys.stderr)
        return 2
    except (ConfigError, LookupError) as exc:
        print(f"setup error: {exc}", file=sys.stderr)
        return 2

    graph = build_graph(deps, checkpointer=checkpointer)
    thread_id = args.thread_id or uuid.uuid4().hex[:8]
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit_for(settings.max_agent_steps),
    }
    print(
        f"thread_id={thread_id}  model={settings.llm_model}  tools={sorted(deps.tools)}  "
        f"checkpointer={settings.checkpointer}"
    )

    if args.command == "resume":
        state = graph.get_state(config)
        if not state.values:
            print(f"no checkpoints for thread {thread_id}", file=sys.stderr)
            return 1
        if not state.next:
            print("nothing to resume: the last run finished. Last answer:")
            print(f"\n{state.values['messages'][-1].content}")
            return 0
        print(f"resuming at step {state.values.get('step')}, next: {list(state.next)}")
        started = time.perf_counter()
        result = invoke_durably(graph, None, config)
        print_result(result, time.perf_counter() - started, args.verbose)
        return 0

    def run_turn(question: str) -> None:
        started = time.perf_counter()
        result = invoke_durably(graph, {"messages": [HumanMessage(content=question)]}, config)
        print_result(result, time.perf_counter() - started, args.verbose)

    if args.command == "ask":
        run_turn(args.question)
        return 0

    print("Type a question (empty line or Ctrl-D to quit).")
    while True:
        try:
            question = input("> ").strip()
        except EOFError:
            break
        if not question:
            break
        run_turn(question)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
