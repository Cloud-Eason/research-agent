"""RedisCheckpointSaver and CachedToolExecutor against fakeredis (no Docker needed)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import fakeredis
import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from research_agent import crash
from research_agent.agent.graph import AgentDeps, build_graph
from research_agent.agent.state import FinalAnswer, FinalAnswerDraft
from research_agent.agent.tools import make_web_search_tool
from research_agent.checkpoint import keys
from research_agent.checkpoint.redis_saver import RedisCheckpointSaver
from research_agent.checkpoint.serde import build_serializer
from research_agent.checkpoint.tool_cache import CachedToolExecutor
from tests.conftest import FakeSearchBackend, LoopingToolCaller, ScriptedChatModel, tool_call

MakeDeps = Callable[..., AgentDeps]


class SimulatedCrash(BaseException):
    """BaseException, like SystemExit: ordinary `except Exception` cannot swallow it."""


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def client(server: fakeredis.FakeServer) -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(server=server)


@pytest.fixture(autouse=True)
def _reset_crash_hook(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("CRASH_AT", raising=False)
    yield
    crash.set_handler(None)


def cfg(thread_id: str = "t1") -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}


def ask(graph: Any, question: str | None, thread_id: str = "t1") -> dict[str, Any]:
    payload = None if question is None else {"messages": [HumanMessage(content=question)]}
    result: dict[str, Any] = graph.invoke(payload, cfg(thread_id), durability="sync")
    return result


def simple_llm() -> ScriptedChatModel:
    return ScriptedChatModel([tool_call("q1", "c1"), AIMessage(content="draft")])


# ---------------------------------------------------------------- round trips
def test_state_round_trips_through_redis(make_deps: MakeDeps, client: Any) -> None:
    saver = RedisCheckpointSaver(client)
    draft = FinalAnswerDraft(answer="A [S1]", short_answer="A", cited_source_ids=["S1"])
    ask(build_graph(make_deps(simple_llm(), draft), checkpointer=saver), "question?")

    # A brand-new saver instance (think: new process) sees the same state.
    fresh = build_graph(make_deps(simple_llm()), checkpointer=RedisCheckpointSaver(client))
    state = fresh.get_state(cfg())
    assert FinalAnswer.model_validate(state.values["final"]).short_answer == "A"
    messages = state.values["messages"]
    assert isinstance(messages[0], HumanMessage)
    assert any(isinstance(m, ToolMessage) for m in messages)
    assert state.values["sources"][0]["id"] == "S1"
    assert state.next == ()


def test_multi_turn_matches_in_memory_behaviour(make_deps: MakeDeps, client: Any) -> None:
    llm = ScriptedChatModel(
        [
            tool_call("first", "c1"),
            AIMessage(content="d1"),
            tool_call("second", "c2"),
            AIMessage(content="d2"),
        ]
    )
    drafts = [
        FinalAnswerDraft(answer="a1 [S1]", short_answer="a1", cited_source_ids=["S1"]),
        FinalAnswerDraft(answer="a2 [S3]", short_answer="a2", cited_source_ids=["S3"]),
    ]
    graph = build_graph(make_deps(llm, drafts), checkpointer=RedisCheckpointSaver(client))
    ask(graph, "question one")
    second = ask(graph, "follow-up")

    assert second["step"] == 2
    assert [s["id"] for s in second["sources"]] == ["S1", "S2", "S3", "S4"]
    assert any(isinstance(m, HumanMessage) and m.content == "question one" for m in llm.calls[2])


# ------------------------------------------------------------------ list API
def test_list_order_limit_before_and_filter(make_deps: MakeDeps, client: Any) -> None:
    saver = RedisCheckpointSaver(client)
    ask(build_graph(make_deps(simple_llm()), checkpointer=saver), "q?")
    rows = list(saver.list(cfg()))
    ids = [r.config["configurable"]["checkpoint_id"] for r in rows]

    assert len(rows) >= 5
    assert ids == sorted(ids, reverse=True)  # newest first
    assert len(list(saver.list(cfg(), limit=2))) == 2
    older = list(saver.list(cfg(), before=rows[1].config))
    assert [r.config["configurable"]["checkpoint_id"] for r in older] == ids[2:]
    assert len(list(saver.list(cfg(), filter={"source": "input"}))) == 1
    assert len(list(saver.list(None))) == len(rows)  # cross-thread scan


# -------------------------------------------------------------- pending writes
def test_put_writes_first_writer_wins_but_special_writes_overwrite(
    make_deps: MakeDeps, client: Any
) -> None:
    saver = RedisCheckpointSaver(client)
    ask(build_graph(make_deps(simple_llm()), checkpointer=saver), "q?")
    latest = saver.get_tuple(cfg())
    assert latest is not None
    conf = latest.config

    saver.put_writes(conf, [("ch", 1), ("__error__", "first")], task_id="task")
    saver.put_writes(conf, [("ch", 99), ("__error__", "second")], task_id="task")
    reloaded = saver.get_tuple(conf)
    assert reloaded is not None
    writes = {(t, c): v for t, c, v in reloaded.pending_writes or []}
    assert writes[("task", "ch")] == 1  # regular write kept its first value
    assert writes[("task", "__error__")] == "second"  # special write overwritten


# ------------------------------------------------------- compaction, TTL, delete
def test_compaction_keeps_last_n_and_collects_blobs(make_deps: MakeDeps, client: Any) -> None:
    saver = RedisCheckpointSaver(client, keep_last=3)
    result = ask(build_graph(make_deps(LoopingToolCaller(), max_steps=4), checkpointer=saver), "q?")

    assert client.zcard(keys.index_key("t1", "")) == 3
    assert len(list(saver.list(cfg()))) == 3
    latest = saver.get_tuple(cfg())
    assert latest is not None
    assert latest.checkpoint["channel_values"]["messages"] == result["messages"]

    # Every remaining blob is referenced by one of the 3 kept checkpoints.
    referenced = set()
    for row in saver.list(cfg()):
        for channel, version in row.checkpoint["channel_versions"].items():
            referenced.add(keys.blob_field(channel, version))
    stored = {f.decode() for f in client.hkeys(keys.blobs_key("t1", ""))}
    assert stored <= referenced


def test_ttl_is_applied_and_refreshed(make_deps: MakeDeps, client: Any) -> None:
    saver = RedisCheckpointSaver(client, ttl_s=120)
    ask(build_graph(make_deps(simple_llm()), checkpointer=saver), "q?")
    for key in (keys.index_key("t1", ""), keys.blobs_key("t1", ""), keys.namespaces_key("t1")):
        assert 0 < client.ttl(key) <= 120
    latest = saver.get_tuple(cfg())
    assert latest is not None
    cid = latest.config["configurable"]["checkpoint_id"]
    assert 0 < client.ttl(keys.checkpoint_key("t1", "", cid)) <= 120


def test_delete_thread_removes_every_key(make_deps: MakeDeps, client: Any) -> None:
    saver = RedisCheckpointSaver(client)
    deps = make_deps(simple_llm())
    deps.executor = CachedToolExecutor(client)
    ask(build_graph(deps, checkpointer=saver), "q?", thread_id="gone")
    ask(build_graph(make_deps(simple_llm()), checkpointer=saver), "q?", thread_id="kept")

    assert client.keys("*{gone}*")
    saver.delete_thread("gone")
    assert client.keys("*{gone}*") == []
    assert saver.get_tuple(cfg("kept")) is not None


# ---------------------------------------------------------------- tool cache
def test_tool_cache_hits_on_same_call_and_args_only(client: Any) -> None:
    backend = FakeSearchBackend()
    tool = make_web_search_tool(backend, 3)
    executor = CachedToolExecutor(client)

    first = executor.execute(tool, {"query": "a"}, thread_id="t", tool_call_id="call_0")
    again = executor.execute(tool, {"query": "a"}, thread_id="t", tool_call_id="call_0")
    other = executor.execute(tool, {"query": "b"}, thread_id="t", tool_call_id="call_0")

    assert first == again
    assert backend.queries == ["a", "b"]  # same id + different args is NOT a hit
    assert (executor.hits, executor.misses) == (1, 2)
    assert other[0].title.startswith("b")


# ------------------------------------------------------------ crash & resume
class StateDrivenLLM:
    """Chooses its reply from the conversation, so a 'restarted process' behaves the same."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: list[BaseMessage], /) -> AIMessage:
        self.calls += 1
        done = sum(isinstance(m, ToolMessage) for m in messages)
        if done == 0:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "x"}, "id": "ca", "type": "tool_call"},
                    {"name": "web_search", "args": {"query": "y"}, "id": "cb", "type": "tool_call"},
                ],
            )
        if done == 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "z"}, "id": "cc", "type": "tool_call"}
                ],
            )
        return AIMessage(content="done")


def _raise_crash(event: str) -> None:
    raise SimulatedCrash(event)


@pytest.mark.parametrize("crash_at", ["tool:1", "tool:2", "tool:3", "agent:2"])
def test_crash_then_resume_never_repeats_a_tool_call(
    make_deps: MakeDeps,
    server: fakeredis.FakeServer,
    monkeypatch: pytest.MonkeyPatch,
    crash_at: str,
) -> None:
    backend = FakeSearchBackend()  # shared: counts external calls across "processes"

    def process() -> Any:
        """Everything a fresh process would build: new client, saver, executor, graph."""
        client = fakeredis.FakeRedis(server=server)
        deps = make_deps(StateDrivenLLM(), backend=backend)
        deps.executor = CachedToolExecutor(client)
        return build_graph(deps, checkpointer=RedisCheckpointSaver(client))

    monkeypatch.setenv("CRASH_AT", crash_at)
    crash.set_handler(_raise_crash)
    with pytest.raises(SimulatedCrash):
        ask(process(), "compare x and y")

    monkeypatch.delenv("CRASH_AT")
    result = ask(process(), None)  # resume: no new input

    assert result["final"] is not None
    assert backend.queries.count("x") == 1
    assert backend.queries.count("y") == 1
    assert backend.queries.count("z") == 1
    saver = RedisCheckpointSaver(fakeredis.FakeRedis(server=server))
    assert len(list(saver.list(cfg(), filter={"source": "input"}))) == 1  # resumed, not restarted


# ------------------------------------------------------------------- async
async def test_async_api(make_deps: MakeDeps, client: Any) -> None:
    saver = RedisCheckpointSaver(client)
    graph = build_graph(make_deps(simple_llm()), checkpointer=saver)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="q?")]}, cfg(), durability="sync"
    )
    assert result["final"] is not None
    assert (await saver.aget_tuple(cfg())) is not None
    rows = [row async for row in saver.alist(cfg(), limit=2)]
    assert len(rows) == 2
    await saver.adelete_thread("t1")
    assert await saver.aget_tuple(cfg()) is None


# ---------------------------------------------------------- deserialization
class Untrusted(BaseModel):
    payload: str


def test_strict_serializer_round_trips_messages_but_not_unknown_classes() -> None:
    serde = build_serializer(strict=True)
    message = AIMessage(
        content="hi", tool_calls=[{"name": "t", "args": {}, "id": "1", "type": "tool_call"}]
    )
    restored = serde.loads_typed(serde.dumps_typed([message]))
    assert isinstance(restored[0], AIMessage)
    assert restored[0].tool_calls[0]["id"] == "1"

    try:
        loaded = serde.loads_typed(serde.dumps_typed(Untrusted(payload="x")))
    except Exception:  # rejecting outright is also acceptable
        loaded = None
    assert not isinstance(loaded, Untrusted)
