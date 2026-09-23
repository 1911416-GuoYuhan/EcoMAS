from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from ecomas.config import METRIC_CONFIG
from ecomas.semantic_clustering import cluster_answers


@dataclass(frozen=True)
class SystemEntropyEstimate:
    task_name: str
    question_uid: str
    num_answers: int
    cluster_ids: tuple[int, ...]
    cluster_distribution: dict[str, float]
    plugin_tsallis: float
    u_statistic_tsallis: float | None
    pair_count: int


def second_order_tsallis_u_statistic(cluster_ids: Iterable[object]) -> float:
    values = list(cluster_ids)
    if len(values) < 2:
        raise ValueError("At least two independent answers are required")
    disagreements = sum(values[left] != values[right] for left in range(len(values)) for right in range(left + 1, len(values)))
    return disagreements / (len(values) * (len(values) - 1) / 2)


def estimate_system_entropy(
    task_name: str,
    question_uid: str,
    answers: Iterable[object],
    *,
    question: str = "",
) -> SystemEntropyEstimate:
    result = cluster_answers(task_name, question_uid, answers, question=question)
    ids = tuple(result.cluster_by_output_id[item.output_id] for item in result.answers)
    distribution = answer_distribution(str(value) for value in ids)
    pair_count = len(ids) * (len(ids) - 1) // 2
    return SystemEntropyEstimate(
        task_name,
        question_uid,
        len(ids),
        ids,
        distribution,
        second_order_tsallis(distribution),
        second_order_tsallis_u_statistic(ids) if pair_count else None,
        pair_count,
    )


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
    if probs and abs(total - 1.0) > METRIC_CONFIG["probability_sum_tolerance"]:
        raise ValueError(f"Probabilities must sum to one, got {total}")
    return 1.0 - sum(value * value for value in probs)


def empirical_second_order_tsallis(answers: Iterable[str]) -> float:
    return second_order_tsallis(answer_distribution(answers))


def records_answer_distribution(records: Iterable[object]) -> dict[str, float]:
    return answer_distribution(getattr(record, "normalized_prediction", "") for record in records)


def grouped_answer_reports(records: Iterable[object]) -> dict[str, dict[str, object]]:
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


def semantic_cluster_report(
    task_name: str,
    question_uid: str,
    answers: Iterable[object],
    *,
    question: str = "",
) -> dict[str, object]:
    result = cluster_answers(task_name, question_uid, answers, question=question)
    cluster_values = [result.cluster_by_output_id[item.output_id] for item in result.answers]
    distribution = answer_distribution(str(value) for value in cluster_values)
    pair_count = len(cluster_values) * (len(cluster_values) - 1) // 2
    return {
        "question_uid": question_uid,
        "task_name": task_name,
        "num_answers": len(result.answers),
        "cluster_distribution": distribution,
        "observed_cluster_count": len(result.members_by_cluster),
        "second_order_tsallis": second_order_tsallis(distribution),
        "second_order_tsallis_u_statistic": (
            second_order_tsallis_u_statistic(cluster_values) if pair_count else None
        ),
        "pair_count": pair_count,
        "invalid_rate": result.invalid_rate,
        "unresolved_rate": result.unresolved_rate,
        "non_transitive_relation": result.non_transitive_relation,
        "semantic_clustering": result.to_dict(),
    }
