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

STAGE_ORDER = ("dense", "bm25", "fused", "candidates", "reranked")
UNTIMED_STAGES = {"candidates"}  # a slice of the fused list, no work of its own


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
        # Absent from records scored before this metric existed.
        "evidence_sentences": _rate(ok, "evidence_sentences_all_found"),
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
    ("evidence sent", lambda s: _pct(s["evidence_sentences"])),
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


def _final_stage(record: dict[str, Any]) -> str:
    """The last stage this record has: what the agent would actually be handed."""
    return next(s for s in reversed(STAGE_ORDER) if s in record["stages"])


# ---- paired comparison ----------------------------------------------------------
def _ok_by_qid(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["qid"]: r for r in records if "error" not in r}


def win_loss(base: dict[str, float], other: dict[str, float]) -> tuple[int, int, int]:
    """(wins, losses, ties) of `other` against `base` over the qids both have."""
    shared = base.keys() & other.keys()
    wins = sum(other[q] > base[q] for q in shared)
    losses = sum(other[q] < base[q] for q in shared)
    return wins, losses, len(shared) - wins - losses


def pick_baseline(names: Sequence[str], baseline: str | None = None) -> str:
    if baseline is not None:
        if baseline not in names:
            raise ValueError(f"baseline run {baseline!r} not found; have {sorted(names)}")
        return baseline
    return "dense" if "dense" in names else names[0]


def _paired_section(scores: dict[str, dict[str, float]], baseline: str, metric: str) -> list[str]:
    base = scores[baseline]
    rows = []
    for name, other in scores.items():
        if name == baseline:
            continue
        shared = base.keys() & other.keys()
        wins, losses, ties = win_loss(base, other)
        rows.append(
            [
                name,
                str(len(shared)),
                _pct(_mean([base[q] for q in shared])) if shared else "-",
                _pct(_mean([other[q] for q in shared])) if shared else "-",
                str(wins),
                str(losses),
                str(ties),
            ]
        )
    if not rows:
        return []
    return [
        f"## Per-question {metric} vs `{baseline}`",
        "",
        _md_table(["run", "n shared", baseline, "run", "wins", "losses", "ties"], rows),
        "",
        f"Only qids both runs answered without error. wins: questions where the run's "
        f"{metric} beats `{baseline}`.",
        "",
    ]


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


def write_reports(eval_dir: Path, baseline: str | None = None) -> Path:
    """Write summary.md / summary.csv for an ablation directory and errors.jsonl per run.

    `baseline` names the run every other run is compared against question by question;
    by default `dense` if there is one, else the first run directory.
    """
    runs = _run_dirs(eval_dir)
    if not runs:
        raise FileNotFoundError(f"no runs with manifest.json under {eval_dir}")
    layer = runs[0][1]["layer"]
    base_run = pick_baseline([m["run"] for _, m in runs], baseline)
    md: list[str] = [f"# {eval_dir.name} ({layer} layer)", ""]
    csv_rows: list[dict[str, Any]] = []

    if layer == "agent":
        summaries, type_rows = [], []
        ok_by_run: dict[str, dict[str, dict[str, Any]]] = {}
        for run_dir, manifest in runs:
            records = latest_records(run_dir / "results.jsonl")
            ok_by_run[manifest["run"]] = _ok_by_qid(records)
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
        md += [
            "EM/F1 in %, with 95% bootstrap confidence intervals. evidence all: every "
            "supporting paragraph retrieved (by title); evidence sent: every supporting "
            "sentence appears in a retrieved chunk.",
            "",
        ]
        md += [
            "## By question type",
            "",
            _md_table(["run", "type", "n", "EM", "F1"], type_rows),
            "",
        ]
        if len({len(ok) for ok in ok_by_run.values()}) > 1:
            shared = set.intersection(*(set(ok) for ok in ok_by_run.values()))
            subset_rows = []
            for name, ok in ok_by_run.items():
                s = summarize_agent([ok[q] for q in sorted(shared)])
                subset_rows.append(
                    [
                        name,
                        str(s["n"]),
                        _pct(s["em"]),
                        _pct(s["f1"]),
                        _pct(s["evidence_all_found"]),
                        _pct(s["evidence_sentences"]),
                    ]
                )
            md += [
                "## Same questions",
                "",
                _md_table(["run", "n", "EM", "F1", "evidence all", "evidence sent"], subset_rows),
                "",
                "Every run restricted to the qids all runs answered without error.",
                "",
            ]
        served_rows = []
        for name, ok in ok_by_run.items():
            counts: dict[str, int] = {}
            for r in ok.values():
                for model in r.get("served_models", []):  # absent from older records
                    counts[model] = counts.get(model, 0) + 1
            served = ", ".join(f"{m} ({n})" for m, n in sorted(counts.items())) or "-"
            served_rows.append([name, served])
        md += [
            "## Served models",
            "",
            _md_table(["run", "model/fingerprint (questions)"], served_rows),
            "",
            "As reported by the API for each LLM call. More than one entry, or entries "
            "that differ between runs, means the backend changed mid-evaluation.",
            "",
        ]
        em = {name: {q: r["em"] for q, r in ok.items()} for name, ok in ok_by_run.items()}
        md += _paired_section(em, base_run, "EM")
    else:
        header = ["run", "stage", *[f"R@{k}" for k in KS], "All@2", "All@5", "All@10", "ms"]
        rows = []
        all5: dict[str, dict[str, float]] = {}
        for run_dir, manifest in runs:
            records = latest_records(run_dir / "results.jsonl")
            all5[manifest["run"]] = {
                q: r["stages"][_final_stage(r)]["all@5"] for q, r in _ok_by_qid(records).items()
            }
            for stage, s in summarize_retrieval(records).items():
                rows.append(
                    [
                        manifest["run"],
                        stage,
                        *[_pct(s[f"recall@{k}"]) for k in KS],
                        _pct(s["all@2"]),
                        _pct(s["all@5"]),
                        _pct(s["all@10"]),
                        "-" if stage in UNTIMED_STAGES else f"{s['ms']:.0f}",
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
        md += _paired_section(all5, base_run, "All@5 (final stage)")

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
            last = _final_stage(r)
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
