from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from research_agent.agent.graph import (
    AgentDeps,
    build_graph,
    recursion_limit_for,
    tool_calls_in_last_turn,
)
from research_agent.agent.state import FinalAnswer, FinalAnswerDraft
from tests.conftest import FakeSearchBackend, LoopingToolCaller, ScriptedChatModel, tool_call

MakeDeps = Callable[..., AgentDeps]


def _ask(graph: Any, question: str, thread_id: str = "t1") -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    result: dict[str, Any] = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
    return result


def test_single_search_answer_has_valid_citation(make_deps: MakeDeps) -> None:
    backend = FakeSearchBackend()
    llm = ScriptedChatModel([tool_call("capital of France", "c1"), AIMessage(content="Paris [S1]")])
    draft = FinalAnswerDraft(
        answer="It is Paris [S1].", short_answer="Paris", cited_source_ids=["S1"]
    )
    result = _ask(build_graph(make_deps(llm, draft, backend)), "What is the capital of France?")

    final = FinalAnswer.model_validate(result["final"])
    assert final.short_answer == "Paris"
    assert [c.id for c in final.citations] == ["S1"]
    assert final.citations[0].location == "https://example.com/1/0"
    assert final.invalid_citation_ids == []
    assert result["step"] == 2
    assert backend.queries == ["capital of France"]


def test_multi_hop_sources_are_numbered_across_calls(make_deps: MakeDeps) -> None:
    llm = ScriptedChatModel(
        [
            tool_call("director of Film X", "c1"),
            tool_call("birthplace of Jane Doe", "c2"),
            AIMessage(content="draft"),
        ]
    )
    result = _ask(build_graph(make_deps(llm)), "Where was the director of Film X born?")

    assert [s["id"] for s in result["sources"]] == ["S1", "S2", "S3", "S4"]
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "[S3]" in str(tool_msgs[1].content)
    trajectory = tool_calls_in_last_turn(result["messages"])
    assert [c["args"]["query"] for c in trajectory] == [
        "director of Film X",
        "birthplace of Jane Doe",
    ]


def test_hallucinated_citations_are_dropped(make_deps: MakeDeps) -> None:
    llm = ScriptedChatModel([tool_call("q", "c1"), AIMessage(content="draft")])
    draft = FinalAnswerDraft(
        answer="Claim [S1]. Another claim [S7].", short_answer="x", cited_source_ids=["S1", "S9"]
    )
    final = FinalAnswer.model_validate(_ask(build_graph(make_deps(llm, draft)), "q?")["final"])

    assert [c.id for c in final.citations] == ["S1"]
    assert sorted(final.invalid_citation_ids) == ["S7", "S9"]


def test_step_limit_forces_finalize_and_closes_tool_calls(make_deps: MakeDeps) -> None:
    llm = LoopingToolCaller()
    backend = FakeSearchBackend()
    result = _ask(build_graph(make_deps(llm, backend=backend, max_steps=3)), "endless?")

    assert llm.calls == 3
    assert len(backend.queries) == 2  # third request was never executed
    assert FinalAnswer.model_validate(result["final"]).hit_step_limit is True

    requested = {
        c["id"] for m in result["messages"] if isinstance(m, AIMessage) for c in m.tool_calls
    }
    answered = {m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)}
    assert requested == answered  # history stays valid for the next API call


def test_tool_failure_is_reported_to_llm_not_raised(make_deps: MakeDeps) -> None:
    llm = ScriptedChatModel([tool_call("q", "c1"), AIMessage(content="could not find it")])
    result = _ask(build_graph(make_deps(llm, backend=FakeSearchBackend(fail=True))), "q?")

    error = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert error.status == "error"
    assert "ConnectionError" in str(error.content)
    assert result["sources"] == []


def test_unknown_tool_is_reported(make_deps: MakeDeps) -> None:
    llm = ScriptedChatModel([tool_call("q", "c1", name="nope"), AIMessage(content="draft")])
    result = _ask(build_graph(make_deps(llm)), "q?")

    error = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Unknown tool" in str(error.content)


def test_finalize_replaces_draft_instead_of_appending(make_deps: MakeDeps) -> None:
    llm = ScriptedChatModel([AIMessage(content="draft only")])
    draft = FinalAnswerDraft(answer="Final text", short_answer="x")
    result = _ask(build_graph(make_deps(llm, draft)), "q?")

    plain_ai = [m for m in result["messages"] if isinstance(m, AIMessage) and not m.tool_calls]
    assert len(plain_ai) == 1
    assert plain_ai[0].content == "Final text"


def test_multi_turn_memory_per_thread(make_deps: MakeDeps) -> None:
    llm = ScriptedChatModel(
        [
            tool_call("first", "c1"),
            AIMessage(content="draft 1"),
            tool_call("second", "c2"),
            AIMessage(content="draft 2"),
        ]
    )
    drafts = [
        FinalAnswerDraft(answer="a1 [S1]", short_answer="a1", cited_source_ids=["S1"]),
        FinalAnswerDraft(answer="a2 [S3]", short_answer="a2", cited_source_ids=["S3"]),
    ]
    graph = build_graph(make_deps(llm, drafts), checkpointer=InMemorySaver())

    _ask(graph, "question one", thread_id="t1")
    second = _ask(graph, "follow-up", thread_id="t1")

    assert second["step"] == 2  # per-turn counter was reset
    assert [s["id"] for s in second["sources"]] == ["S1", "S2", "S3", "S4"]
    assert FinalAnswer.model_validate(second["final"]).citations[0].id == "S3"
    first_prompt_of_turn_two = llm.calls[2]
    assert any(
        isinstance(m, HumanMessage) and m.content == "question one"
        for m in first_prompt_of_turn_two
    )
    assert graph.get_state({"configurable": {"thread_id": "other"}}).values == {}


def test_recursion_limit_covers_max_steps() -> None:
    # start + n agent + (n-1) tools + finalize must fit under the limit
    for n in (1, 3, 8, 20):
        assert recursion_limit_for(n) > 1 + n + (n - 1) + 1


def test_no_tools_prompt_names_no_tools(make_deps: MakeDeps) -> None:
    """Named-but-unbound tools make DeepSeek write tool calls as plain text."""
    llm = ScriptedChatModel([AIMessage(content="Paris")])
    deps = dataclasses.replace(make_deps(llm), tools={})
    _ask(build_graph(deps), "What is the capital of France?")
    system = str(llm.calls[0][0].content)
    assert not any(word in system for word in ("kb_search", "web_search", "tool"))


def test_served_models_are_recorded_once(make_deps: MakeDeps) -> None:
    meta = {"model_name": "deepseek-flash", "system_fingerprint": "fp1"}
    llm = ScriptedChatModel(
        [
            AIMessage(
                content="", tool_calls=tool_call("q", "c1").tool_calls, response_metadata=meta
            ),
            AIMessage(content="Paris [S1]", response_metadata=meta),
        ]
    )
    result = _ask(build_graph(make_deps(llm)), "What is the capital of France?")
    assert result["served_models"] == ["deepseek-flash/fp1"]
