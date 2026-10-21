"""ReAct agent as an explicit LangGraph StateGraph.

    START -> start -> agent --(tool_calls & budget left)--> tools -> agent ...
                          \\--(no tool_calls | budget used)--> finalize -> END

Built by hand rather than with a prebuilt agent so that each node is a clear
checkpoint boundary (Phase 3) and the tools node owns execution (idempotency
cache, source numbering, trajectory logging for eval).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal, Protocol

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from research_agent.agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    FINALIZE_SYSTEM_PROMPT,
    STEP_LIMIT_NOTE,
)
from research_agent.agent.state import (
    AgentState,
    Citation,
    FinalAnswer,
    FinalAnswerDraft,
    SourceDict,
)
from research_agent.agent.tools import ToolExecutor, ToolSpec, format_sources

_CITATION_RE = re.compile(r"\[(S\d+)\]")


class ChatModel(Protocol):
    """Anything with `.invoke(messages)`; satisfied by LangChain Runnables and test fakes."""

    def invoke(self, input: list[BaseMessage], /) -> Any: ...


@dataclass
class AgentDeps:
    agent_llm: ChatModel  # chat model already bound to the tool schemas
    finalizer: ChatModel  # returns FinalAnswerDraft, or {"raw", "parsed"} with include_raw
    tools: dict[str, ToolSpec]
    executor: ToolExecutor
    max_steps: int
    today: Callable[[], date] = field(default=date.today)


def usage_from_message(message: BaseMessage) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "llm_calls": 1,
    }


class AgentNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    # ---- start: reset per-turn counters ---------------------------------
    def start(self, state: AgentState) -> dict[str, Any]:
        return {"step": 0, "hit_step_limit": False, "final": None}

    # ---- agent: one LLM call that may request tools ----------------------
    def agent(self, state: AgentState) -> dict[str, Any]:
        step = state.get("step", 0)
        system = SystemMessage(
            AGENT_SYSTEM_PROMPT.format(
                today=self.deps.today().isoformat(),
                step=step,
                max_steps=self.deps.max_steps,
            )
        )
        response = self.deps.agent_llm.invoke([system, *state.get("messages", [])])
        if not isinstance(response, AIMessage):
            raise TypeError(f"agent LLM returned {type(response).__name__}, expected AIMessage")
        return {
            "messages": [response],
            "step": step + 1,
            "token_usage": usage_from_message(response),
        }

    # ---- router -----------------------------------------------------------
    def route_after_agent(self, state: AgentState) -> Literal["tools", "finalize"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            if state.get("step", 0) >= self.deps.max_steps:
                return "finalize"  # budget used: finalize closes the dangling calls
            return "tools"
        return "finalize"

    # ---- tools: execute every call from the last AIMessage ----------------
    def tools(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        thread_id = str((config.get("configurable") or {}).get("thread_id", "default"))
        next_index = len(state.get("sources", [])) + 1

        tool_messages: list[ToolMessage] = []
        new_sources: list[SourceDict] = []
        for call in last.tool_calls:
            call_id = call.get("id") or f"call_{next_index}"
            name = call["name"]
            args = call.get("args", {})
            spec = self.deps.tools.get(name)
            if spec is None:
                tool_messages.append(_error_message(f"Unknown tool '{name}'.", call_id, name))
                continue
            try:
                hits = self.deps.executor.execute(
                    spec, args, thread_id=thread_id, tool_call_id=call_id
                )
            except ValidationError as exc:
                tool_messages.append(_error_message(f"Invalid arguments: {exc}", call_id, name))
                continue
            except Exception as exc:  # surfaced to the LLM so it can retry or move on
                tool_messages.append(
                    _error_message(f"Tool failed: {type(exc).__name__}: {exc}", call_id, name)
                )
                continue

            sources: list[SourceDict] = []
            for hit in hits:
                sources.append(
                    SourceDict(
                        id=f"S{next_index}",
                        tool=name,
                        query=str(args.get("query", "")),
                        title=hit.title,
                        location=hit.location,
                        content=hit.content,
                        score=hit.score,
                    )
                )
                next_index += 1
            new_sources.extend(sources)
            tool_messages.append(
                ToolMessage(
                    content=format_sources(sources) or "No results.",
                    tool_call_id=call_id,
                    name=name,
                )
            )
        return {"messages": tool_messages, "sources": new_sources}

    # ---- finalize: structured answer + citation validation ----------------
    def finalize(self, state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        hit_limit = isinstance(last, AIMessage) and bool(last.tool_calls)

        # Close unanswered tool calls so the thread history stays valid for the
        # OpenAI-style API on the next turn.
        patches: list[ToolMessage] = []
        if hit_limit and isinstance(last, AIMessage):
            patches = [
                ToolMessage(
                    content="Not executed: research step limit reached.",
                    tool_call_id=call.get("id") or "",
                    name=call["name"],
                )
                for call in last.tool_calls
            ]

        sources = state.get("sources", [])
        prompt: list[BaseMessage] = [
            SystemMessage(
                FINALIZE_SYSTEM_PROMPT.format(
                    limit_note=STEP_LIMIT_NOTE if hit_limit else "",
                    sources=format_sources(sources) or "(no sources retrieved)",
                )
            ),
            *_conversation_only(messages),
        ]
        draft, usage = _coerce_draft(self.deps.finalizer.invoke(prompt))
        final = build_final_answer(draft, sources, hit_step_limit=hit_limit)

        # Replace the agent's plain-text draft (same id) instead of appending a
        # second AI message; add_messages replaces messages with a matching id.
        final_message = AIMessage(content=render_answer(final))
        if isinstance(last, AIMessage) and not last.tool_calls and last.id:
            final_message = AIMessage(id=last.id, content=render_answer(final))

        return {
            "messages": [*patches, final_message],
            "final": final.model_dump(),
            "hit_step_limit": hit_limit,
            "token_usage": usage,
        }


def _error_message(text: str, call_id: str, name: str) -> ToolMessage:
    return ToolMessage(content=f"Error: {text}", tool_call_id=call_id, name=name, status="error")


def _conversation_only(messages: Sequence[AnyMessage]) -> list[BaseMessage]:
    """User turns and plain-text AI turns; drops tool-call plumbing."""
    return [
        m
        for m in messages
        if isinstance(m, HumanMessage) or (isinstance(m, AIMessage) and not m.tool_calls)
    ]


def _coerce_draft(result: Any) -> tuple[FinalAnswerDraft, dict[str, int]]:
    """Accept either a parsed draft or with_structured_output(include_raw=True) output."""
    usage: dict[str, int] = {}
    if isinstance(result, dict) and "parsed" in result:
        raw = result.get("raw")
        if isinstance(raw, BaseMessage):
            usage = usage_from_message(raw)
        if result.get("parsing_error") or result["parsed"] is None:
            raise ValueError(f"finalizer output could not be parsed: {result.get('parsing_error')}")
        result = result["parsed"]
    if isinstance(result, FinalAnswerDraft):
        return result, usage
    return FinalAnswerDraft.model_validate(result), usage


def build_final_answer(
    draft: FinalAnswerDraft, sources: Sequence[SourceDict], *, hit_step_limit: bool
) -> FinalAnswer:
    """Keep only citations that point at sources actually retrieved in this thread."""
    known = {s["id"]: s for s in sources}
    cited = list(dict.fromkeys([*draft.cited_source_ids, *_CITATION_RE.findall(draft.answer)]))
    return FinalAnswer(
        answer=draft.answer.strip(),
        short_answer=draft.short_answer.strip(),
        citations=[
            Citation(id=c, title=known[c]["title"], location=known[c]["location"])
            for c in cited
            if c in known
        ],
        invalid_citation_ids=[c for c in cited if c not in known],
        hit_step_limit=hit_step_limit,
    )


def render_answer(final: FinalAnswer) -> str:
    if not final.citations:
        return final.answer
    refs = "\n".join(f"[{c.id}] {c.title} — {c.location}" for c in final.citations)
    return f"{final.answer}\n\nSources:\n{refs}"


def tool_calls_in_last_turn(messages: Sequence[AnyMessage]) -> list[dict[str, Any]]:
    """Tool-call trajectory since the latest user message (used by CLI and eval)."""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    calls: list[dict[str, Any]] = []
    for message in messages[start + 1 :]:
        if isinstance(message, AIMessage):
            calls.extend({"name": c["name"], "args": c.get("args", {})} for c in message.tool_calls)
    return calls


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    nodes = AgentNodes(deps)
    graph = StateGraph(AgentState)
    graph.add_node("start", nodes.start)
    graph.add_node("agent", nodes.agent)
    graph.add_node("tools", nodes.tools)
    graph.add_node("finalize", nodes.finalize)
    graph.add_edge(START, "start")
    graph.add_edge("start", "agent")
    graph.add_conditional_edges(
        "agent", nodes.route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def recursion_limit_for(max_steps: int) -> int:
    """Super-steps per turn: start + max_steps agent + (max_steps-1) tools + finalize."""
    return 2 * max_steps + 4
