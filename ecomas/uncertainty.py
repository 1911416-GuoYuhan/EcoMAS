"""Utilities for empirical output distributions and second-order Tsallis entropy."""

from collections import Counter
from typing import Iterable, Mapping


def answer_distribution(answers: Iterable[str]) -> dict[str, float]:
    values = list(answers)
    if not values:
        return {}
    counts = Counter(str(answer) for answer in values)
    total = float(len(values))
    return {answer: count / total for answer, count in sorted(counts.items())}


def second_order_tsallis(probabilities: Mapping[str, float] | Iterable[float]) -> float:
    values = probabilities.values() if isinstance(probabilities, Mapping) else probabilities
    probs = [float(value) for value in values]
    if any(value < 0 for value in probs):
        raise ValueError("Probabilities must be non-negative")
    total = sum(probs)
    if probs and abs(total - 1.0) > 1e-8:
        raise ValueError(f"Probabilities must sum to one, got {total}")
    return 1.0 - sum(value * value for value in probs)


def empirical_second_order_tsallis(answers: Iterable[str]) -> float:
    return second_order_tsallis(answer_distribution(answers))


def records_answer_distribution(records: Iterable[object]) -> dict[str, float]:
    """Estimate the final-answer distribution from RunRecord-like objects."""
    return answer_distribution(getattr(record, "normalized_prediction", "") for record in records)


def grouped_answer_reports(records: Iterable[object]) -> dict[str, dict[str, object]]:
    """Aggregate repeated RunRecords by uid with distribution and Tsallis entropy."""
    grouped: dict[str, list[str]] = {}
    for record in records:
        uid = str(getattr(record, "uid", ""))
        grouped.setdefault(uid, []).append(str(getattr(record, "normalized_prediction", "")))
    return {
        uid: {
            "num_samples": len(answers),
            "distribution": answer_distribution(answers),
            "second_order_tsallis": empirical_second_order_tsallis(answers),
        }
        for uid, answers in sorted(grouped.items())
    }
