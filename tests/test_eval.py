"""Evaluation: metrics, dataset parsing, runners (with fakes) and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from research_agent.eval.agent_eval import run_agent_eval
from research_agent.eval.datasets import (
    EvalExample,
    build_corpus,
    load_examples,
    parse_hotpot_row,
    save_examples,
)
from research_agent.eval.metrics import (
    all_found_at_k,
    bootstrap_ci,
    bridge_entity,
    dedupe_titles,
    exact_match,
    f1_score,
    normalize_answer,
    recall_at_k,
    trajectory_metrics,
)
from research_agent.eval.records import completed_qids, latest_records
from research_agent.eval.report import summarize_agent, summarize_retrieval, write_reports
from research_agent.eval.retrieval_eval import run_retrieval_eval
from research_agent.rag.models import Chunk, ScoredChunk
from research_agent.rag.retrieval import RetrievalResult

ROW = {
    "id": "q1",
    "question": "Where was the director of Inception born?",
    "answer": "London",
    "type": "bridge",
    "level": "medium",
    "supporting_facts": {
        "title": ["Inception", "Christopher Nolan", "Christopher Nolan"],
        "sent_id": [0, 0, 1],
    },
    "context": {
        "title": ["Inception", "Christopher Nolan", "Paris"],
        "sentences": [
            ["Inception is a film", " by Nolan."],
            ["Nolan was born", " in London."],
            ["Paris is a city."],
        ],
    },
}


def example(qid: str = "q1", **overrides: Any) -> EvalExample:
    row = {**ROW, "id": qid, **overrides}
    return parse_hotpot_row(row)


# ---- metrics ---------------------------------------------------------------------
def test_normalization_matches_official_rules() -> None:
    assert normalize_answer("The  City of London!") == "city of london"
    assert exact_match("the London", "London") == 1.0


def test_f1_partial_and_yes_no_rule() -> None:
    assert f1_score("London, England", "London") == pytest.approx(2 / 3)
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes it is", "yes") == 0.0  # special answers must match exactly
    assert f1_score("Paris", "London") == 0.0


def test_paragraph_level_recall() -> None:
    ranked = dedupe_titles(["A", "A", "B", "C", "A"])
    assert ranked == ["A", "B", "C"]
    assert recall_at_k(ranked, ["A", "C"], 2) == 0.5
    assert all_found_at_k(ranked, ["A", "C"], 2) == 0.0
    assert all_found_at_k(ranked, ["A", "C"], 3) == 1.0


def test_bridge_entity_heuristic() -> None:
    q = "Where was the director of Inception born?"
    assert bridge_entity(q, ["Inception", "Christopher Nolan"], "bridge") == "Christopher Nolan"
    assert bridge_entity(q, ["Inception (film)", "Christopher Nolan"], "bridge") == (
        "Christopher Nolan"
    )
    assert bridge_entity(q, ["Inception", "Christopher Nolan"], "comparison") is None


def test_trajectory_detects_shortcut_vs_discovery() -> None:
    q, gold = "Where was the director of Inception born?", ["Inception", "Christopher Nolan"]

    def kb(query: str) -> dict[str, Any]:
        return {"name": "kb_search", "args": {"query": query}}

    shortcut = trajectory_metrics(
        q, "bridge", gold, [kb("Christopher Nolan birthplace")], kb_enabled=True
    )
    assert shortcut["bridge_in_first_query"] is True
    assert shortcut["sequence_ok"] is False  # only one retrieval

    found = trajectory_metrics(
        q, "bridge", gold, [kb("Inception director"), kb("Christopher Nolan born")], kb_enabled=True
    )
    assert (found["bridge_in_first_query"], found["bridge_in_later_query"]) == (False, True)
    assert found["sequence_ok"] is True
    assert found["kb_first"] is True
    assert trajectory_metrics(q, "bridge", gold, [], kb_enabled=False)["sequence_ok"] is None


def test_bootstrap_ci_is_deterministic_and_brackets_mean() -> None:
    values = [1.0] * 60 + [0.0] * 40
    lo, hi = bootstrap_ci(values)
    assert lo < 0.6 < hi
    assert bootstrap_ci(values) == (lo, hi)


# ---- datasets --------------------------------------------------------------------
def test_parse_round_trip_and_corpus(tmp_path: Path) -> None:
    ex = example()
    assert ex.supporting_titles == ("Inception", "Christopher Nolan")
    assert ex.paragraphs[1].text == "Nolan was born in London."

    path = tmp_path / "sample.jsonl"
    save_examples(path, [ex, example("q2")])
    assert load_examples(path) == [ex, example("q2")]

    corpus = build_corpus([ex, example("q2")])  # same paragraphs twice -> deduplicated
    assert [d.title for d in corpus] == ["Inception", "Christopher Nolan", "Paris"]
    assert corpus[0].source == "hotpotqa:Inception"
    assert corpus[0].doc_id == build_corpus([ex])[0].doc_id  # stable ids


# ---- retrieval runner -------------------------------------------------------------
def scored(title: str) -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"{title}-p0-0000",
        doc_id=title,
        title=title,
        source="s",
        page=None,
        position=0,
        text=title,
    )
    return ScoredChunk(chunk=chunk, score=1.0, channel="x")


class FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, top_n: int | None = None) -> RetrievalResult:
        self.calls += 1
        dense = [scored("Paris"), scored("Inception")]
        reranked = [scored("Inception"), scored("Christopher Nolan"), scored("Paris")]
        return RetrievalResult(
            query=query,
            hits=reranked,
            stages={"dense": dense, "reranked": reranked},
            timings_ms={"dense": 5.0, "rerank": 20.0},
        )


def test_retrieval_eval_scores_each_stage_and_resumes(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    retriever = FakeRetriever()
    run_retrieval_eval(retriever, [example("q1"), example("q2")], out, top_k=20)
    run_retrieval_eval(retriever, [example("q1"), example("q2")], out, top_k=20)
    assert retriever.calls == 2  # second pass skipped finished questions

    summary = summarize_retrieval(latest_records(out))
    assert summary["dense"]["recall@2"] == 0.5
    assert summary["reranked"]["all@2"] == 1.0
    assert summary["reranked"]["ms"] == 20.0


# ---- agent runner --------------------------------------------------------------------
class FakeGraph:
    """Answers correctly except for questions containing 'boom', which fail once."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.failed_once = False

    async def ainvoke(self, input: Any, config: Any, /) -> Any:
        question = input["messages"][0].content
        self.calls.append(question)
        if "boom" in question and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("rate limited")
        return {
            "messages": [
                HumanMessage(content=question),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "kb_search",
                            "args": {"query": "Inception director"},
                            "id": "1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="London [S1]"),
            ],
            "final": {
                "answer": "London [S1]",
                "short_answer": "London",
                "citations": [],
                "invalid_citation_ids": [],
                "hit_step_limit": False,
            },
            "sources": [{"id": "S1", "tool": "kb_search", "title": "Christopher Nolan"}],
            "step": 2,
            "token_usage": {"input_tokens": 100, "output_tokens": 10, "llm_calls": 3},
        }


async def test_agent_eval_scores_records_and_retries_failures(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    graph = FakeGraph()
    exs = [example("q1"), example("q2", question="boom: Where was the director born?")]

    await run_agent_eval(graph, exs, out, run_name="t", kb_enabled=True, concurrency=2)
    assert completed_qids(out) == {"q1"}
    await run_agent_eval(graph, exs, out, run_name="t", kb_enabled=True, concurrency=2)
    assert completed_qids(out) == {"q1", "q2"}
    assert len(graph.calls) == 3  # q1 once, q2 failed then retried

    records = latest_records(out)
    q1 = next(r for r in records if r["qid"] == "q1")
    assert q1["em"] == 1.0
    assert q1["evidence_recall"] == 0.5  # found Nolan, not Inception
    assert q1["kb_first"] is True
    s = summarize_agent(records)
    assert s["n"] == 2
    assert s["errors"] == 0
    assert s["input_tokens"] == 100


def test_write_reports_produces_markdown_and_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "abl" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"layer": "retrieval", "run": "r1"}))
    run_retrieval_eval(FakeRetriever(), [example("q1")], run_dir / "results.jsonl")

    summary = write_reports(tmp_path / "abl").read_text()
    assert "| r1 | dense |" in summary
    assert (tmp_path / "abl" / "summary.csv").exists()
    assert (run_dir / "errors.jsonl").exists()
