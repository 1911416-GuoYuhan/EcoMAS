from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class BenchmarkSample:
    uid: str
    task_name: str
    input_text: str
    gold_answer: str
    metadata: dict[str, Any]


def load_samples(task_name: str, benchmark_root: Path, split: str, limit: int | None) -> list[BenchmarkSample]:
    loaders = {
        "mmlu_pro": load_mmlu_pro,
        "math500": load_math500,
        "chaosnli": load_chaosnli,
    }
    if task_name not in loaders:
        raise ValueError(f"Unknown task: {task_name}")
    samples = list(loaders[task_name](benchmark_root, split))
    return samples[:limit] if limit else samples


def load_mmlu_pro(root: Path, split: str) -> Iterable[BenchmarkSample]:
    path = root / "mmlupro" / f"{split}.parquet"
    frame = pd.read_parquet(path)
    for _, row in frame.iterrows():
        options = [
            f"{letter}: {option}"
            for letter, option in zip(string.ascii_uppercase, row["options"])
        ]
        text = (
            f"The following are multiple choice questions (with answers) about "
            f"{row['category']}.\n"
            f"{row['question']}\n"
            + " ".join(options)
        )
        yield BenchmarkSample(
            uid=str(row["question_id"]),
            task_name="mmlu_pro",
            input_text=text,
            gold_answer=str(row["answer"]).strip(),
            metadata={"category": row["category"], "src": row.get("src", "")},
        )


def load_math500(root: Path, split: str) -> Iterable[BenchmarkSample]:
    if split != "test":
        raise ValueError("MATH-500 only provides split 'test' in this benchmark directory")
    path = root / "math500" / "test.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            text = (
                "Solve the following math problem carefully and return the final exact answer:\n"
                + row["problem"]
            )
            yield BenchmarkSample(
                uid=str(row["unique_id"]),
                task_name="math500",
                input_text=text,
                gold_answer=str(row["answer"]).strip(),
                metadata={"subject": row.get("subject"), "level": row.get("level")},
            )


def load_chaosnli(root: Path, split: str) -> Iterable[BenchmarkSample]:
    split_map = {
        "mnli_m": "chaosNLI_mnli_m.jsonl",
        "snli": "chaosNLI_snli.jsonl",
        "alphanli": "chaosNLI_alphanli.jsonl",
    }
    if split not in split_map:
        raise ValueError(f"ChaosNLI split must be one of {sorted(split_map)}")
    path = root / "chaosnli" / split_map[split]
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if split == "alphanli":
                example = row["example"]
                text = (
                    "Choose which hypothesis best explains the observations.\n"
                    f"Observation 1: {example['obs1']}\n"
                    f"Observation 2: {example['obs2']}\n"
                    f"Hypothesis 1: {example['hyp1']}\n"
                    f"Hypothesis 2: {example['hyp2']}\n"
                    "Return exactly one label: entailment for hypothesis 1, "
                    "contradiction for hypothesis 2."
                )
            else:
                text = (
                    "Decide the NLI relation between premise and hypothesis.\n"
                    f"Premise: {row['premise']}\n"
                    f"Hypothesis: {row['hypothesis']}\n"
                    "Return exactly one label: entailment, neutral, or contradiction."
                )
            yield BenchmarkSample(
                uid=str(row["uid"]),
                task_name="chaosnli",
                input_text=text,
                gold_answer=normalize_chaos_label(row["majority_label"]),
                metadata={
                    "entropy": row.get("entropy"),
                    "label_dist": row.get("label_dist"),
                    "source": row.get("source", row.get("example", {}).get("source")),
                },
            )


def normalize_chaos_label(label: Any) -> str:
    mapping = {
        "e": "entailment",
        "n": "neutral",
        "c": "contradiction",
        "0": "entailment",
        "1": "neutral",
        "2": "contradiction",
        0: "entailment",
        1: "neutral",
        2: "contradiction",
        "entailment": "entailment",
        "neutral": "neutral",
        "contradiction": "contradiction",
    }
    key = label if isinstance(label, int) else str(label).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown ChaosNLI label: {label}")
    return mapping[key]
