"""Wires real dependencies (DeepSeek, Tavily, the Milvus knowledge base) into AgentDeps."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from research_agent.agent.graph import AgentDeps
from research_agent.agent.state import FinalAnswerDraft
from research_agent.agent.tools import (
    DirectToolExecutor,
    TavilyBackend,
    ToolExecutor,
    ToolSpec,
    make_kb_search_tool,
    make_web_search_tool,
)
from research_agent.config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.require_secret("deepseek_api_key"),
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def build_tools(settings: Settings) -> list[ToolSpec]:
    """Tools enabled by config. With none enabled the agent answers from its own
    knowledge, which is the "no retrieval" baseline in the ablations."""
    tools: list[ToolSpec] = []
    if settings.enable_kb_search:
        from research_agent.rag.factory import build_retriever  # loads models; import lazily

        tools.append(make_kb_search_tool(build_retriever(settings)))
    if settings.enable_web_search:
        backend = TavilyBackend(
            api_key=settings.require_secret("tavily_api_key"),
            snippet_chars=settings.web_search_snippet_chars,
        )
        tools.append(make_web_search_tool(backend, settings.web_search_max_results))
    return tools


def build_default_deps(
    settings: Settings,
    executor: ToolExecutor | None = None,
    tools: list[ToolSpec] | None = None,
) -> AgentDeps:
    llm = build_chat_model(settings)
    tool_list = build_tools(settings) if tools is None else tools
    agent_llm = llm.bind_tools([t.to_openai_schema() for t in tool_list]) if tool_list else llm
    return AgentDeps(
        agent_llm=agent_llm,
        # function_calling is the structured-output mode DeepSeek supports reliably;
        # include_raw keeps the AIMessage so token usage can be counted.
        finalizer=llm.with_structured_output(
            FinalAnswerDraft, method="function_calling", include_raw=True
        ),
        tools={t.name: t for t in tool_list},
        executor=executor or DirectToolExecutor(),
        max_steps=settings.max_agent_steps,
    )
