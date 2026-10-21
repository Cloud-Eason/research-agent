"""Crash-recovery demo: kill the agent mid-run, restart it, finish without redoing work.

    uv run python scripts/crash_recovery_demo.py                    # real DeepSeek + Tavily
    uv run python scripts/crash_recovery_demo.py --offline          # deterministic, no API keys
    uv run python scripts/crash_recovery_demo.py --crash-at agent:2

What happens:
  1. A child process runs the agent with CRASH_AT set; it dies via os._exit(137)
     at the chosen point (default: right after the 2nd tool result is cached).
  2. The parent inspects Redis: checkpoints, cached tool results, external calls.
  3. A second child process resumes the same thread_id with invoke(None, ...).
  4. The parent verifies three invariants:
       * no external tool call was executed twice,
       * the run was resumed, not restarted (exactly one "input" checkpoint),
       * a final answer was produced.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from research_agent.agent.factory import build_default_deps, build_tools
from research_agent.agent.graph import AgentDeps, build_graph, recursion_limit_for
from research_agent.agent.state import FinalAnswerDraft, SearchHit
from research_agent.agent.tools import ToolSpec, make_web_search_tool
from research_agent.checkpoint.factory import redis_client
from research_agent.checkpoint.redis_saver import RedisCheckpointSaver
from research_agent.checkpoint.tool_cache import CachedToolExecutor
from research_agent.cli import invoke_durably, print_result
from research_agent.config import Settings, get_settings
from research_agent.crash import CRASH_EXIT_CODE

DEFAULT_QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"


# ---------------------------------------------------------------- offline fakes
class OfflineLLM:
    """Decides from the conversation itself, so a fresh process continues correctly."""

    def invoke(self, messages: list[BaseMessage], /) -> AIMessage:
        done = sum(isinstance(m, ToolMessage) for m in messages)
        if done == 0:  # two parallel searches in one message -> one tools node
            return AIMessage(
                content="",
                tool_calls=[
                    _call("Scott Derrickson nationality", "call_a"),
                    _call("Ed Wood nationality", "call_b"),
                ],
            )
        if done == 2:
            follow_up = _call("Derrickson Wood both American", "call_c")
            return AIMessage(content="", tool_calls=[follow_up])
        return AIMessage(content="Both are American [S1][S3].")


class OfflineFinalizer:
    def invoke(self, messages: list[BaseMessage], /) -> FinalAnswerDraft:
        return FinalAnswerDraft(
            answer="Yes. Both are American filmmakers [S1][S3].",
            short_answer="yes",
            cited_source_ids=["S1", "S3"],
        )


class OfflineSearch:
    def search(self, query: str, max_results: int) -> list[SearchHit]:
        time.sleep(0.5)  # pretend to be a network call
        return [
            SearchHit(
                title=f"{query} ({i})",
                location=f"https://example.org/{i}",
                content=f"Offline result {i} for: {query}",
            )
            for i in range(2)
        ]


def _call(query: str, call_id: str) -> dict[str, Any]:
    return {"name": "web_search", "args": {"query": query}, "id": call_id, "type": "tool_call"}


# ------------------------------------------------------------------- the child
def _counting(spec: ToolSpec, client: Any, thread_id: str) -> ToolSpec:
    """Count every real external call in Redis, so both processes add to one counter."""

    def run(args: Any) -> list[SearchHit]:
        record = json.dumps({"tool": spec.name, **args.model_dump()})
        client.rpush(f"demo:calls:{thread_id}", record)
        return spec.run(args)

    return replace(spec, run=run)


def child(args: argparse.Namespace, settings: Settings) -> int:
    client = redis_client(settings)
    saver = RedisCheckpointSaver(client, ttl_s=settings.checkpoint_ttl_s, keep_last=200)
    executor = CachedToolExecutor(client, ttl_s=settings.tool_cache_ttl_s)

    deps: AgentDeps
    if args.offline:
        tool = _counting(make_web_search_tool(OfflineSearch(), 2), client, args.thread_id)
        deps = AgentDeps(
            agent_llm=OfflineLLM(),
            finalizer=OfflineFinalizer(),
            tools={tool.name: tool},
            executor=executor,
            max_steps=8,
        )
    else:
        tools = [_counting(t, client, args.thread_id) for t in build_tools(settings)]
        deps = build_default_deps(settings, executor=executor, tools=tools)

    graph = build_graph(deps, checkpointer=saver)
    config = {
        "configurable": {"thread_id": args.thread_id},
        "recursion_limit": recursion_limit_for(deps.max_steps),
    }
    payload = None if args.child == "resume" else {"messages": [HumanMessage(args.question)]}
    started = time.perf_counter()
    result = invoke_durably(graph, payload, config)
    print_result(result, time.perf_counter() - started, verbose=True)
    print(f"[child] tool cache hits={executor.hits} misses={executor.misses}")
    return 0


# ------------------------------------------------------------------ the parent
def _spawn(mode: str, args: argparse.Namespace, thread_id: str, crash_at: str | None) -> int:
    env = {k: v for k, v in os.environ.items() if k != "CRASH_AT"}
    if crash_at:
        env["CRASH_AT"] = crash_at
    cmd = [
        sys.executable,
        __file__,
        "--child",
        mode,
        "--thread-id",
        thread_id,
        "--question",
        args.question,
    ]
    if args.offline:
        cmd.append("--offline")
    if args.with_kb:
        cmd.append("--with-kb")
    return subprocess.run(cmd, env=env, check=False).returncode


def _snapshot(saver: RedisCheckpointSaver, client: Any, thread_id: str) -> None:
    rows = list(saver.list({"configurable": {"thread_id": thread_id}}))
    print(f"  checkpoints: {len(rows)} (newest first)")
    for row in rows[:6]:
        meta = row.metadata
        writes = len(row.pending_writes or [])
        print(
            f"    step={meta.get('step'):>3}  source={meta.get('source'):<6} "
            f"id=...{row.config['configurable']['checkpoint_id'][-8:]}  pending_writes={writes}"
        )
    cached = client.scard(f"toolcache_idx:{{{thread_id}}}")
    calls = [json.loads(c) for c in client.lrange(f"demo:calls:{thread_id}", 0, -1)]
    print(f"  cached tool results: {cached}")
    print(f"  external calls so far: {len(calls)}")
    for c in calls:
        print(f"    - {c}")


def parent(args: argparse.Namespace, settings: Settings) -> int:
    client = redis_client(settings)
    saver = RedisCheckpointSaver(client, ttl_s=settings.checkpoint_ttl_s, keep_last=200)
    thread_id = f"demo-{uuid.uuid4().hex[:8]}"
    mode = "offline (scripted LLM + fake search)" if args.offline else "live (DeepSeek + Tavily)"
    print(f"thread_id={thread_id}  mode={mode}  crash_at={args.crash_at}\n")

    print("=== 1. run until the simulated crash ===")
    code = _spawn("run", args, thread_id, args.crash_at)
    print(f"\n[parent] child exited with code {code}")
    if code != CRASH_EXIT_CODE:
        print(
            "[parent] the crash point was never reached (the model may have used fewer "
            "tool calls); try --crash-at tool:1 or --offline"
        )
        return 1

    print("\n=== 2. state left in Redis after the crash ===")
    _snapshot(saver, client, thread_id)
    calls_before = client.llen(f"demo:calls:{thread_id}")

    print("\n=== 3. restart and resume the same thread ===")
    code = _spawn("resume", args, thread_id, None)
    if code != 0:
        print(f"[parent] resume failed with exit code {code}")
        return 1

    print("\n=== 4. verify ===")
    _snapshot(saver, client, thread_id)
    final = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    assert final is not None
    messages = final.checkpoint["channel_values"]["messages"]
    executed = [
        m.tool_call_id
        for m in messages
        if isinstance(m, ToolMessage) and not str(m.content).startswith("Not executed")
    ]
    calls_total = client.llen(f"demo:calls:{thread_id}")
    inputs = len(
        list(saver.list({"configurable": {"thread_id": thread_id}}, filter={"source": "input"}))
    )
    checks = {
        "no tool call executed twice": calls_total == len(set(executed)),
        "resumed, not restarted (one input checkpoint)": inputs == 1,
        "final answer produced": bool(final.checkpoint["channel_values"].get("final")),
    }
    print(
        f"  external calls: {calls_before} before crash, {calls_total - calls_before} "
        f"after resume, {calls_total} total for {len(set(executed))} distinct tool calls"
    )
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not args.keep:
        saver.delete_thread(thread_id)
        client.delete(f"demo:calls:{thread_id}")
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--offline", action="store_true", help="no API keys, deterministic")
    parser.add_argument("--crash-at", default="tool:2", help="tool:N or agent:N")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--with-kb", action="store_true", help="also enable kb_search (live)")
    parser.add_argument("--keep", action="store_true", help="keep the thread in Redis afterwards")
    parser.add_argument("--child", choices=["run", "resume"], help=argparse.SUPPRESS)
    parser.add_argument("--thread-id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    settings = get_settings().model_copy(
        update={"enable_kb_search": args.with_kb, "checkpointer": "redis"}
    )
    return child(args, settings) if args.child else parent(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
