"""HotpotQA (distractor) loading, reproducible sampling, and corpus construction.

Each distractor example ships 10 Wikipedia paragraphs: 2 gold (supporting) and
8 distractors. We pool the paragraphs of all sampled questions into one shared
knowledge base, so retrieval has to find the right 2 among ~5000 paragraphs
instead of among 10. That is harder, and closer to a real deployment.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_agent.rag.models import Document, make_doc_id

HF_DATASET = "hotpotqa/hotpot_qa"


@dataclass(frozen=True)
class Paragraph:
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return "".join(self.sentences).strip()


@dataclass(frozen=True)
class EvalExample:
    qid: str
    question: str
    answer: str
    type: str  # "bridge" | "comparison"
    level: str  # "easy" | "medium" | "hard"
    supporting_titles: tuple[str, ...]
    supporting_facts: tuple[tuple[str, int], ...]
    paragraphs: tuple[Paragraph, ...]

    def supporting_sentences(self) -> list[str]:
        """The gold supporting sentences. A few HotpotQA facts point past the end of
        their paragraph (annotation errors); those are skipped."""
        by_title = {p.title: p.sentences for p in self.paragraphs}
        out = []
        for title, idx in self.supporting_facts:
            sentences = by_title.get(title, ())
            if 0 <= idx < len(sentences):
                out.append(sentences[idx])
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "answer": self.answer,
            "type": self.type,
            "level": self.level,
            "supporting_titles": list(self.supporting_titles),
            "supporting_facts": [list(f) for f in self.supporting_facts],
            "paragraphs": [
                {"title": p.title, "sentences": list(p.sentences)} for p in self.paragraphs
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalExample:
        return cls(
            qid=str(data["qid"]),
            question=str(data["question"]),
            answer=str(data["answer"]),
            type=str(data["type"]),
            level=str(data["level"]),
            supporting_titles=tuple(data["supporting_titles"]),
            supporting_facts=tuple((str(t), int(i)) for t, i in data["supporting_facts"]),
            paragraphs=tuple(
                Paragraph(title=p["title"], sentences=tuple(p["sentences"]))
                for p in data["paragraphs"]
            ),
        )


def parse_hotpot_row(row: Mapping[str, Any]) -> EvalExample:
    context = row["context"]
    paragraphs = tuple(
        Paragraph(title=str(t), sentences=tuple(str(s) for s in sents))
        for t, sents in zip(context["title"], context["sentences"], strict=True)
    )
    facts = row["supporting_facts"]
    supporting = tuple(
        (str(t), int(i)) for t, i in zip(facts["title"], facts["sent_id"], strict=True)
    )
    return EvalExample(
        qid=str(row["id"]),
        question=str(row["question"]),
        answer=str(row["answer"]),
        type=str(row["type"]),
        level=str(row["level"]),
        supporting_titles=tuple(dict.fromkeys(t for t, _ in supporting)),
        supporting_facts=supporting,
        paragraphs=paragraphs,
    )


def sample_hotpot(
    n: int, seed: int, *, split: str = "validation", config: str = "distractor"
) -> list[EvalExample]:
    """Uniform random sample without replacement. The sample order is itself
    random, so any prefix (first 300, first 100) is also a uniform sample, and
    smaller runs are strict subsets of larger ones."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, config, split=split)
    indices = random.Random(seed).sample(range(len(ds)), n)
    return [parse_hotpot_row(ds[i]) for i in indices]


def save_examples(path: Path, examples: Iterable[EvalExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


def load_examples(path: Path) -> list[EvalExample]:
    with path.open(encoding="utf-8") as f:
        return [EvalExample.from_dict(json.loads(line)) for line in f if line.strip()]


def build_corpus(examples: Sequence[EvalExample]) -> list[Document]:
    """One Document per unique paragraph title across all sampled questions."""
    docs: dict[str, Document] = {}
    for ex in examples:
        for para in ex.paragraphs:
            if para.title in docs or not para.text:
                continue
            source = f"hotpotqa:{para.title}"
            docs[para.title] = Document(
                doc_id=make_doc_id(source), title=para.title, source=source, text=para.text
            )
    return list(docs.values())
