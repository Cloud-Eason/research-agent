"""Answer, retrieval and trajectory metrics.

Answer metrics follow the official HotpotQA evaluation script: lowercase, strip
punctuation and articles, collapse whitespace; token-level F1 is zero whenever
either side is yes/no/noanswer and the two differ.
"""

from __future__ import annotations

import random
import re
import string
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

_SPECIAL = {"yes", "no", "noanswer"}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if (pred in _SPECIAL or ref in _SPECIAL) and pred != ref:
        return 0.0
    pred_tokens, ref_tokens = pred.split(), ref.split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# ---- retrieval ---------------------------------------------------------------
def dedupe_titles(titles: Iterable[str]) -> list[str]:
    """Paragraph titles in order of first appearance (chunk ranks -> paragraph ranks)."""
    return list(dict.fromkeys(titles))


def recall_at_k(ranked_titles: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    top = set(ranked_titles[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def all_found_at_k(ranked_titles: Sequence[str], gold: Sequence[str], k: int) -> float:
    top = set(ranked_titles[:k])
    return float(bool(gold) and all(g in top for g in gold))


# ---- trajectory ----------------------------------------------------------------
def _core(title: str) -> str:
    """'Ed Wood (film)' -> 'ed wood', for matching titles inside free text."""
    return normalize_answer(re.sub(r"\s*\(.*?\)", "", title))


def bridge_entity(question: str, gold_titles: Sequence[str], qtype: str) -> str | None:
    """For bridge questions, the gold paragraph the question does not name.

    "Where was the director of Inception born?" names Inception; the second gold
    paragraph (Christopher Nolan) is the bridge entity the agent has to discover.
    Returns None when this heuristic cannot identify exactly one such title.
    """
    if qtype != "bridge":
        return None
    q = normalize_answer(question)
    missing = [t for t in gold_titles if _core(t) and _core(t) not in q]
    return missing[0] if len(missing) == 1 else None


def trajectory_metrics(
    question: str,
    qtype: str,
    gold_titles: Sequence[str],
    calls: Sequence[dict[str, Any]],
    *,
    kb_enabled: bool,
) -> dict[str, Any]:
    kb_queries = [
        str(c.get("args", {}).get("query", "")) for c in calls if c["name"] == "kb_search"
    ]
    bridge = bridge_entity(question, gold_titles, qtype)
    in_first = in_later = None
    if bridge is not None and kb_queries:
        core = _core(bridge)
        in_first = core in normalize_answer(kb_queries[0])
        in_later = any(core in normalize_answer(q) for q in kb_queries[1:])
    return {
        "n_tool_calls": len(calls),
        "n_kb_calls": len(kb_queries),
        "kb_first": bool(calls) and calls[0]["name"] == "kb_search",
        "used_web": any(c["name"] == "web_search" for c in calls),
        # Expected trajectory for a 2-hop question with a knowledge base:
        # start with kb_search and retrieve at least twice.
        "sequence_ok": (
            (bool(calls) and calls[0]["name"] == "kb_search" and len(kb_queries) >= 2)
            if kb_enabled
            else None
        ),
        "bridge_entity": bridge,
        # True in the first query = the model already "knew" the bridge entity
        # (a parametric-memory shortcut); true only later = it found it by retrieval.
        "bridge_in_first_query": in_first,
        "bridge_in_later_query": in_later,
    }


# ---- statistics ------------------------------------------------------------------
def bootstrap_ci(
    values: Sequence[float], *, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. With n=300 a 3-point F1 gap can be
    noise; the interval makes that visible in every summary table."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_resamples))
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return (lo, hi)
