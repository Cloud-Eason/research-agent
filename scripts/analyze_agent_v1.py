"""Offline analysis of agent_v1 against retrieval_v1. Reads results/ only, calls no API.

uv run python scripts/analyze_agent_v1.py > docs/agent_v1_analysis.md

1. Does query rewriting make up for single-shot ranking? On the questions where, in
   retrieval_v1, bge reranking found all evidence in the top 5 and dense alone did
   not, compare the dense and dense_bge agent runs. As a supplement, compare runs on
   questions where the agent called kb_search exactly once (a self-selected subset).
2. Error taxonomy of the dense agent run's EM misses, plus a fixed random sample of
   the "reasoning error" class for a manual label audit.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from research_agent.eval.records import latest_records
from research_agent.eval.report import win_loss

RESULTS = Path("results")
AGENT_RUNS = ("dense", "hybrid", "hybrid_minilm", "dense_bge", "hybrid_bge")
AUDIT_SEED = 0
AUDIT_SIZE = 20


def load(path: Path) -> dict[str, dict[str, Any]]:
    return {r["qid"]: r for r in latest_records(path) if "error" not in r}


def pct(values: Sequence[float]) -> str:
    return f"{100 * sum(values) / len(values):.1f}" if values else "-"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A Markdown table followed by a blank line."""
    head = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return [*head, *("| " + " | ".join(row) + " |" for row in rows), ""]


def metric_row(name: str, records: Sequence[dict[str, Any]]) -> list[str]:
    return [
        name,
        str(len(records)),
        pct([float(r["evidence_all_found"]) for r in records]),
        pct([float(r["evidence_sentences_all_found"]) for r in records]),
        pct([r["em"] for r in records]),
        pct([r["f1"] for r in records]),
    ]


METRIC_HEADER = ["run", "n", "evidence all", "evidence sent", "EM", "F1"]


def short(text: str, n: int = 160) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= n else text[: n - 1] + "…"


def analysis_1(agent: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    bge = load(RESULTS / "retrieval_v1/c128_dense_bge/results.jsonl")
    minilm = load(RESULTS / "retrieval_v1/c128_minilm_first100/results.jsonl")
    subset = sorted(
        q
        for q in bge.keys() & minilm.keys()
        if bge[q]["stages"]["reranked"]["all@5"] == 1.0
        and minilm[q]["stages"]["dense"]["all@5"] == 0.0
    )
    reverse = [
        q
        for q in bge.keys() & minilm.keys()
        if bge[q]["stages"]["reranked"]["all@5"] == 0.0
        and minilm[q]["stages"]["dense"]["all@5"] == 1.0
    ]
    md = [
        "## 1. Does query rewriting make up for single-shot ranking?",
        "",
        "### 1a. Questions where bge reranking fixed single-shot retrieval",
        "",
        f"Among the first 100 questions, retrieval_v1 has **{len(subset)}** where "
        "`c128_dense_bge` (reranked) has All@5 = 1 and dense alone "
        "(`c128_minilm_first100`, dense stage) has All@5 = 0; "
        f"{len(reverse)} go the other way. These questions are defined by layer-1 "
        "results only, so the subset is the same for every agent run. Retrieval in "
        "layer 1 uses the original question; in layer 2 the agent writes its own "
        "queries, and keeps top-5 per call (`retrieval_final_top_n: 5`).",
        "",
        *table(
            METRIC_HEADER, [metric_row(run, [agent[run][q] for q in subset]) for run in AGENT_RUNS]
        ),
    ]
    pair_rows = []
    for key in ("evidence_all_found", "evidence_sentences_all_found", "em"):
        base = {q: float(agent["dense"][q][key]) for q in subset}
        other = {q: float(agent["dense_bge"][q][key]) for q in subset}
        wins, losses, ties = win_loss(base, other)
        pair_rows.append([key, str(wins), str(losses), str(ties)])
    md += [
        "Per question, `dense_bge` vs `dense` on this subset:",
        "",
        *table(["metric", "dense_bge better", "dense better", "same"], pair_rows),
        "Per question detail (dense → dense_bge):",
        "",
    ]
    rows = []
    for q in subset:
        d, b = agent["dense"][q], agent["dense_bge"][q]
        rows.append(
            [
                short(d["question"], 90),
                short(d["gold"], 30),
                f"{d['n_kb_calls']} → {b['n_kb_calls']}",
                "{} → {}".format(
                    int(d["evidence_sentences_all_found"]), int(b["evidence_sentences_all_found"])
                ),
                f"{int(d['em'])} → {int(b['em'])}",
            ]
        )
    md += table(["question", "gold", "kb calls", "evidence sent", "EM"], rows)

    # 1b: single-call questions, chosen by the agent itself.
    md += [
        "### 1b. Questions where the agent called kb_search exactly once",
        "",
        "**Selection bias:** whether the agent stops after one search is its own "
        "decision, made after seeing that run's results, so each run's subset is "
        "different and the rows are not a controlled comparison.",
        "",
    ]
    rows = []
    for run in AGENT_RUNS:
        recs = list(agent[run].values())
        n1 = [r for r in recs if r["n_kb_calls"] == 1]
        rows.append(
            [*metric_row(run, n1), pct([1.0 if r["n_kb_calls"] == 1 else 0.0 for r in recs])]
        )
    md += table([*METRIC_HEADER, "% of 300"], rows)
    common = set.intersection(
        *({q for q, r in agent[run].items() if r["n_kb_calls"] == 1} for run in AGENT_RUNS)
    )
    md += [
        f"Questions every run answered with one call ({len(common)}): the same set "
        "across runs, but still chosen by the agent, so it skews to easy questions.",
        "",
        *table(
            METRIC_HEADER,
            [metric_row(run, [agent[run][q] for q in sorted(common)]) for run in AGENT_RUNS],
        ),
        "All questions by number of kb_search calls (`dense` vs `dense_bge`):",
        "",
    ]
    rows = []
    for bucket, test in (
        ("1", lambda n: n == 1),
        ("2", lambda n: n == 2),
        ("3+", lambda n: n >= 3),
    ):
        for run in ("dense", "dense_bge"):
            recs = [r for r in agent[run].values() if test(r["n_kb_calls"])]
            rows.append([f"{bucket} calls", *metric_row(run, recs)])
    md += table(["calls", *METRIC_HEADER], rows)
    return md


Category = tuple[str, Callable[[dict[str, Any]], bool]]
CATEGORIES: list[Category] = [
    ("retrieval missed evidence", lambda r: not r["evidence_sentences_all_found"]),
    ("format / partly right", lambda r: bool(r["evidence_sentences_all_found"]) and r["f1"] >= 0.5),
    ("reasoning error", lambda r: bool(r["evidence_sentences_all_found"]) and r["f1"] < 0.5),
]


def analysis_2(dense: dict[str, dict[str, Any]]) -> list[str]:
    misses = [r for r in dense.values() if r["em"] < 1.0]
    rng = random.Random(AUDIT_SEED)
    md = [
        "## 2. Error taxonomy of the `dense` run",
        "",
        f"{len(misses)} of {len(dense)} questions have EM = 0. Evidence is the "
        "sentence-level metric (every supporting sentence in a retrieved chunk).",
        "",
    ]
    groups = {name: [r for r in misses if test(r)] for name, test in CATEGORIES}
    md += table(
        ["category", "rule", "n", "% of misses"],
        [
            [
                name,
                rule,
                str(len(groups[name])),
                pct([1.0] * len(groups[name]) + [0.0] * (len(misses) - len(groups[name]))),
            ]
            for (name, _), rule in zip(
                CATEGORIES,
                (
                    "evidence sent = false",
                    "evidence sent = true, F1 ≥ 0.5",
                    "evidence sent = true, F1 < 0.5",
                ),
                strict=True,
            )
        ],
    )
    for name, _ in CATEGORIES:
        examples = rng.sample(groups[name], min(3, len(groups[name])))
        md += [f"Examples, {name}:", ""]
        md += table(
            ["question", "gold", "prediction", "F1"],
            [
                [
                    short(r["question"]),
                    short(r["gold"], 40),
                    short(r["prediction"], 40),
                    f"{r['f1']:.2f}",
                ]
                for r in examples
            ],
        )

    audit = random.Random(AUDIT_SEED).sample(
        sorted(groups["reasoning error"], key=lambda r: r["qid"]),
        min(AUDIT_SIZE, len(groups["reasoning error"])),
    )
    md += [
        f"### Label audit sample: {len(audit)} reasoning errors (seed {AUDIT_SEED})",
        "",
        "Full answers as recorded. Judgements are in `docs/agent_v1_label_audit.md`.",
        "",
    ]
    for i, r in enumerate(audit, 1):
        md += [
            f"**A{i}** `{r['qid']}` [{r['type']}] {r['question']}",
            f"- gold: {r['gold']}",
            f"- prediction: {r['prediction']}  (F1 {r['f1']:.2f})",
            f"- answer: {' '.join(r['answer'].split())}",
            "",
        ]
    return md


def main() -> None:
    agent = {run: load(RESULTS / f"agent_v1/{run}/results.jsonl") for run in AGENT_RUNS}
    md = [
        "# agent_v1 analysis",
        "",
        "Generated by `scripts/analyze_agent_v1.py` from `results/` only (no API calls).",
        "",
        *analysis_1(agent),
        *analysis_2(agent["dense"]),
    ]
    print("\n".join(md))


if __name__ == "__main__":
    main()
