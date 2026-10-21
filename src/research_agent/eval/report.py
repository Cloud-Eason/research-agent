"""Turn results.jsonl files into summary tables (Markdown + CSV) and error exports."""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from research_agent.eval.metrics import bootstrap_ci
from research_agent.eval.records import latest_records
from research_agent.eval.retrieval_eval import KS

STAGE_ORDER = ("dense", "bm25", "fused", "reranked")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rate(records: Sequence[dict[str, Any]], key: str) -> float | None:
    """Share of records where `key` is truthy, over records where it applies."""
    applicable = [r[key] for r in records if r.get(key) is not None]
    return _mean([float(bool(v)) for v in applicable]) if applicable else None


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


# ---- agent ----------------------------------------------------------------------
def summarize_agent(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if "error" not in r]
    em = [r["em"] for r in ok]
    f1 = [r["f1"] for r in ok]
    latency = [r["latency_s"] for r in ok]
    return {
        "n": len(ok),
        "errors": len(records) - len(ok),
        "em": _mean(em),
        "em_ci": bootstrap_ci(em),
        "f1": _mean(f1),
        "f1_ci": bootstrap_ci(f1),
        "tool_calls": _mean([r["n_tool_calls"] for r in ok]),
        "kb_first": _rate(ok, "kb_first"),
        "sequence_ok": _rate(ok, "sequence_ok"),
        "evidence_recall": _mean([r["evidence_recall"] for r in ok]),
        "evidence_all_found": _mean([r["evidence_all_found"] for r in ok]),
        "bridge_in_first": _rate(ok, "bridge_in_first_query"),
        "bridge_in_later": _rate(ok, "bridge_in_later_query"),
        "step_limit": _rate(ok, "hit_step_limit"),
        "invalid_citation_rate": _mean([float(r["n_invalid_citations"] > 0) for r in ok]),
        "input_tokens": _mean([r["input_tokens"] for r in ok]),
        "output_tokens": _mean([r["output_tokens"] for r in ok]),
        "latency_mean": _mean(latency),
        "latency_p50": statistics.median(latency) if latency else 0.0,
        "latency_p95": _percentile(latency, 0.95),
    }


AGENT_COLUMNS: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
    ("run", lambda s: s["run"]),
    ("n", lambda s: str(s["n"])),
    ("EM", lambda s: f"{_pct(s['em'])} [{_pct(s['em_ci'][0])}, {_pct(s['em_ci'][1])}]"),
    ("F1", lambda s: f"{_pct(s['f1'])} [{_pct(s['f1_ci'][0])}, {_pct(s['f1_ci'][1])}]"),
    ("evidence R", lambda s: _pct(s["evidence_recall"])),
    ("evidence all", lambda s: _pct(s["evidence_all_found"])),
    ("tool calls", lambda s: f"{s['tool_calls']:.2f}"),
    ("kb first %", lambda s: _pct(s["kb_first"])),
    ("seq ok %", lambda s: _pct(s["sequence_ok"])),
    ("bridge 1st %", lambda s: _pct(s["bridge_in_first"])),
    ("bridge later %", lambda s: _pct(s["bridge_in_later"])),
    ("step limit %", lambda s: _pct(s["step_limit"])),
    ("bad cite %", lambda s: _pct(s["invalid_citation_rate"])),
    ("tokens in", lambda s: f"{s['input_tokens']:.0f}"),
    ("tokens out", lambda s: f"{s['output_tokens']:.0f}"),
    ("latency p50/p95 s", lambda s: f"{s['latency_p50']:.1f} / {s['latency_p95']:.1f}"),
    ("errors", lambda s: str(s["errors"])),
]


def agent_by_type(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for qtype in sorted({r["type"] for r in records if "error" not in r}):
        rows = [r for r in records if "error" not in r and r["type"] == qtype]
        out[qtype] = {
            "n": len(rows),
            "em": _mean([r["em"] for r in rows]),
            "f1": _mean([r["f1"] for r in rows]),
        }
    return out


# ---- retrieval ------------------------------------------------------------------
def summarize_retrieval(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    ok = [r for r in records if "error" not in r]
    stages = [s for s in STAGE_ORDER if ok and s in ok[0]["stages"]]
    timing_key = {"fused": "fuse", "reranked": "rerank"}
    out: dict[str, dict[str, float]] = {}
    for stage in stages:
        row: dict[str, float] = {"n": float(len(ok))}
        for k in KS:
            row[f"recall@{k}"] = _mean([r["stages"][stage][f"recall@{k}"] for r in ok])
            row[f"all@{k}"] = _mean([r["stages"][stage][f"all@{k}"] for r in ok])
        row["ms"] = _mean([r["timings_ms"].get(timing_key.get(stage, stage), 0.0) for r in ok])
        out[stage] = row
    return out


# ---- writing --------------------------------------------------------------------
def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _run_dirs(eval_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for manifest in sorted(eval_dir.glob("*/manifest.json")):
        found.append((manifest.parent, json.loads(manifest.read_text(encoding="utf-8"))))
    return found


def write_reports(eval_dir: Path) -> Path:
    """Write summary.md / summary.csv for an ablation directory and errors.jsonl per run."""
    runs = _run_dirs(eval_dir)
    if not runs:
        raise FileNotFoundError(f"no runs with manifest.json under {eval_dir}")
    layer = runs[0][1]["layer"]
    md: list[str] = [f"# {eval_dir.name} ({layer} layer)", ""]
    csv_rows: list[dict[str, Any]] = []

    if layer == "agent":
        summaries, type_rows = [], []
        for run_dir, manifest in runs:
            records = latest_records(run_dir / "results.jsonl")
            summary = {"run": manifest["run"], **summarize_agent(records)}
            summaries.append(summary)
            for qtype, s in agent_by_type(records).items():
                type_rows.append(
                    [manifest["run"], qtype, str(int(s["n"])), _pct(s["em"]), _pct(s["f1"])]
                )
            _export_agent_errors(run_dir, records)
            csv_rows.append({k: v for k, v in summary.items() if not k.endswith("_ci")})
        md += [
            _md_table(
                [c for c, _ in AGENT_COLUMNS],
                [[fn(s) for _, fn in AGENT_COLUMNS] for s in summaries],
            ),
            "",
        ]
        md += ["EM/F1 in %, with 95% bootstrap confidence intervals.", ""]
        md += [
            "## By question type",
            "",
            _md_table(["run", "type", "n", "EM", "F1"], type_rows),
            "",
        ]
    else:
        header = ["run", "stage", *[f"R@{k}" for k in KS], "All@2", "All@5", "All@10", "ms"]
        rows = []
        for run_dir, manifest in runs:
            records = latest_records(run_dir / "results.jsonl")
            for stage, s in summarize_retrieval(records).items():
                rows.append(
                    [
                        manifest["run"],
                        stage,
                        *[_pct(s[f"recall@{k}"]) for k in KS],
                        _pct(s["all@2"]),
                        _pct(s["all@5"]),
                        _pct(s["all@10"]),
                        f"{s['ms']:.0f}",
                    ]
                )
                csv_rows.append({"run": manifest["run"], "stage": stage, **s})
            _export_retrieval_misses(run_dir, records)
        md += [
            _md_table(header, rows),
            "",
            "Recall at paragraph level (chunks deduplicated by title), in %. All@k: share "
            "of questions with every supporting paragraph in the top k.",
            "",
        ]

    out = eval_dir / "summary.md"
    out.write_text("\n".join(md), encoding="utf-8")
    if csv_rows:
        with (eval_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    return out


def _export_agent_errors(run_dir: Path, records: Sequence[dict[str, Any]]) -> None:
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            if "error" in r or r["em"] < 1.0:
                keep = (
                    "qid",
                    "type",
                    "question",
                    "gold",
                    "prediction",
                    "f1",
                    "tool_calls",
                    "evidence_all_found",
                    "bridge_entity",
                    "error",
                )
                f.write(json.dumps({k: r.get(k) for k in keep}, ensure_ascii=False) + "\n")


def _export_retrieval_misses(run_dir: Path, records: Sequence[dict[str, Any]]) -> None:
    with (run_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            if "error" in r:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                continue
            last = next(s for s in reversed(STAGE_ORDER) if s in r["stages"])
            if r["stages"][last]["all@5"] < 1.0:
                f.write(
                    json.dumps(
                        {
                            "qid": r["qid"],
                            "type": r["type"],
                            "question": r["question"],
                            "gold": r["gold"],
                            "stage": last,
                            "top5": r["stages"][last]["titles"][:5],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
