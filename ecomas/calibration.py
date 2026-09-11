"""Calibration metrics and reliability-curve data for multiclass outputs."""

from math import log
from typing import Sequence


def _validate(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> None:
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probabilities and labels must be non-empty and have equal length")
    for row in probabilities:
        if any(p < 0 for p in row) or abs(sum(row) - 1.0) > 1e-6:
            raise ValueError("each probability row must be a distribution")


def brier_score(probabilities: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    _validate(probabilities, labels)
    return sum(sum((p - (i == label)) ** 2 for i, p in enumerate(row)) for row, label in zip(probabilities, labels)) / len(labels)


def nll(probabilities: Sequence[Sequence[float]], labels: Sequence[int], eps: float = 1e-12) -> float:
    _validate(probabilities, labels)
    return -sum(log(max(row[label], eps)) for row, label in zip(probabilities, labels)) / len(labels)


def reliability_diagram(probabilities: Sequence[Sequence[float]], labels: Sequence[int], bins: int = 10) -> list[dict[str, float | int]]:
    _validate(probabilities, labels)
    if bins < 1:
        raise ValueError("bins must be positive")
    groups = [[] for _ in range(bins)]
    for row, label in zip(probabilities, labels):
        confidence = max(row)
        prediction = max(range(len(row)), key=row.__getitem__)
        index = min(bins - 1, int(confidence * bins))
        groups[index].append((confidence, int(prediction == label)))
    result = []
    for index, group in enumerate(groups):
        if group:
            result.append({"bin": index, "count": len(group), "confidence": sum(x[0] for x in group) / len(group), "accuracy": sum(x[1] for x in group) / len(group)})
        else:
            result.append({"bin": index, "count": 0, "confidence": 0.0, "accuracy": 0.0})
    return result


def ece(probabilities: Sequence[Sequence[float]], labels: Sequence[int], bins: int = 10) -> float:
    diagram = reliability_diagram(probabilities, labels, bins)
    total = len(labels)
    return sum((entry["count"] / total) * abs(entry["accuracy"] - entry["confidence"]) for entry in diagram)
