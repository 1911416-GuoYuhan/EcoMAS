"""Calibration metrics and reliability-curve data for multiclass outputs."""

from math import log
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ecomas.evaluation import invalid_answer, is_valid_answer, normalize_answer
from ecomas.semantic_clustering import _math_symbol


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


def distribution_brier_score(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
) -> float:
    """Expected multiclass Brier score for possibly soft true labels."""
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("probabilities and targets must be non-empty and have equal length")
    _validate(probabilities, [0] * len(probabilities))
    _validate(targets, [0] * len(targets))
    if any(len(probability) != len(target) for probability, target in zip(probabilities, targets)):
        raise ValueError("probabilities and targets must have equal dimensions")
    return sum(
        sum(probability * probability - 2.0 * probability * target for probability, target in zip(row, target_row)) + 1.0
        for row, target_row in zip(probabilities, targets)
    ) / len(probabilities)


def distribution_nll(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    eps: float = 1e-12,
) -> float:
    """Cross entropy against a true-answer distribution."""
    if len(probabilities) != len(targets) or not probabilities:
        raise ValueError("probabilities and targets must be non-empty and have equal length")
    _validate(probabilities, [0] * len(probabilities))
    _validate(targets, [0] * len(targets))
    if any(len(probability) != len(target) for probability, target in zip(probabilities, targets)):
        raise ValueError("probabilities and targets must have equal dimensions")
    return -sum(
        sum(target * log(max(probability, eps)) for probability, target in zip(row, target_row))
        for row, target_row in zip(probabilities, targets)
    ) / len(probabilities)


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


@dataclass(frozen=True)
class SemanticSpace:
    """A fixed, task-level feature space used by calibration.

    The keys are deliberately independent of question-local cluster IDs.  A
    prediction is represented by a one-hot feature (or a normalized empirical
    distribution), so the same coordinates can be compared on train and test.
    """

    task_name: str
    keys: tuple[str, ...]

    @property
    def dimension(self) -> int:
        return len(self.keys)

    def index(self, key: str) -> int:
        return self.keys.index(key)


def semantic_key(task_name: str, answer: Any) -> str:
    """Map an answer to the fixed symbolic space for its task."""
    text = str(answer).strip()
    if text.startswith("invalid::") or not is_valid_answer(task_name, text):
        return invalid_answer(task_name)
    if task_name == "mmlu_pro":
        return f"mmlu_pro::{normalize_answer(task_name, text)}"
    if task_name == "chaosnli":
        return f"chaosnli::{normalize_answer(task_name, text)}"
    if task_name == "math500":
        return _math_symbol(text)[1]
    return f"{task_name}::{text}"


def build_semantic_space(task_name: str, samples: Iterable[Any] = (), answers: Iterable[Any] = ()) -> SemanticSpace:
    """Construct a deterministic task-level space (方案 A).

    MMLU and ChaosNLI use their complete closed label sets.  Math is open
    vocabulary, therefore the space is the sorted union of observed train/gold
    canonical keys; callers should build it once from train plus test support.
    """
    if task_name == "mmlu_pro":
        keys = tuple(f"mmlu_pro::{letter}" for letter in "ABCDEFGHIJ") + (invalid_answer(task_name),)
    elif task_name == "chaosnli":
        keys = tuple(f"chaosnli::{label}" for label in ("entailment", "neutral", "contradiction")) + (invalid_answer(task_name),)
    else:
        values = list(answers)
        for sample in samples:
            values.extend([getattr(sample, "gold_answer", "")])
        observed = {semantic_key(task_name, value) for value in values if str(value).strip()}
        # Math has open vocabulary; reserve a deterministic unknown bucket so
        # predictions outside the collected gold support still contribute.
        observed.add("math500::__other__")
        observed.add(invalid_answer(task_name))
        keys = tuple(sorted(observed))
    if not keys:
        raise ValueError("semantic space cannot be empty")
    return SemanticSpace(task_name, keys)


def output_distribution(task_name: str, outputs: Iterable[Any], space: SemanticSpace) -> list[float]:
    counts = [0.0] * space.dimension
    values = list(outputs)
    if not values:
        raise ValueError("outputs must be non-empty")
    for output in values:
        key = semantic_key(task_name, output)
        if key not in space.keys and task_name == "math500":
            key = "math500::__other__"
        if key not in space.keys:
            raise ValueError(f"output {key!r} is outside semantic space")
        counts[space.index(key)] += 1.0
    total = sum(counts)
    if total == 0:
        return [0.0] * space.dimension
    return [value / total for value in counts]


def gold_distribution(task_name: str, sample: Any, space: SemanticSpace) -> list[float]:
    """Return q*; ChaosNLI uses its human label distribution when available."""
    if task_name == "chaosnli":
        raw = getattr(sample, "metadata", {}).get("label_dist")
        if isinstance(raw, dict):
            labels = ("entailment", "neutral", "contradiction")
            values = [float(raw.get(label, raw.get(label[0], 0.0))) for label in labels]
            total = sum(values)
            if total > 0:
                return [value / total for value in values] + [0.0]
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            total = sum(float(x) for x in raw)
            if total > 0:
                return [float(x) / total for x in raw] + [0.0]
    vector = [0.0] * space.dimension
    key = semantic_key(task_name, getattr(sample, "gold_answer"))
    if key not in space.keys:
        raise ValueError(f"gold answer {key!r} is outside semantic space")
    vector[space.index(key)] = 1.0
    return vector


def calibration_error(q_theta: Sequence[float], q_star: Sequence[float]) -> float:
    if len(q_theta) != len(q_star):
        raise ValueError("q_theta and q_star must have equal dimension")
    return sum((float(a) - float(b)) ** 2 for a, b in zip(q_theta, q_star))


def estimate_calibration_gradient(
    output_features: "torch.Tensor",
    route_log_probs: "torch.Tensor",
    q_star: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Leave-one-out paired/REINFORCE estimator for the calibration objective.

    ``output_features`` has shape ``[B, D]`` and ``route_log_probs`` is
    ``[B, T]``.  Features and q estimates are detached; only route log-probs
    carry gradients.  The returned loss is the empirical squared MMD objective
    and the scalar surrogate can be backpropagated through the router.
    """
    import torch
    if output_features.ndim != 2 or route_log_probs.ndim != 2:
        raise ValueError("features must be [B,D] and route_log_probs [B,T]")
    bsz = output_features.shape[0]
    if bsz < 2 or route_log_probs.shape[0] != bsz or q_star.numel() != output_features.shape[1]:
        raise ValueError("B must be >= 2 and dimensions must agree")
    features = output_features.detach()
    target = q_star.detach().to(features)
    q_theta = features.mean(dim=0)
    loss = torch.sum((q_theta - target) ** 2)
    terms = []
    for index in range(bsz):
        q_loo = (features.sum(dim=0) - features[index]) / (bsz - 1)
        advantage = 2.0 * torch.dot(q_loo - target, features[index] - q_loo)
        terms.append(advantage.detach() * route_log_probs[index].sum())
    surrogate = torch.stack(terms).mean()
    return loss, surrogate


def estimate_paired_calibration_gradient(
    main_features: "torch.Tensor",
    branch_features: "torch.Tensor",
    route_log_probs: "torch.Tensor",
    q_star: "torch.Tensor",
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Paper-aligned stepwise C/R calibration estimator.

    ``main_features`` is ``[B,D]``; ``branch_features`` is ``[B,T,3,D]`` in
    C/R/Z order; and ``route_log_probs`` is ``[B,T]``.  Z is retained for
    diagnostics, while the calibration gradient uses the required R-minus-C
    suffix feature difference at every decision step.
    """
    import torch
    if main_features.ndim != 2 or branch_features.ndim != 4 or route_log_probs.ndim != 2:
        raise ValueError("expected main [B,D], branches [B,T,3,D], log_probs [B,T]")
    bsz, horizon, branch_count, dimension = branch_features.shape
    if bsz < 2 or branch_count != 3 or main_features.shape != (bsz, dimension) or route_log_probs.shape != (bsz, horizon) or q_star.numel() != dimension:
        raise ValueError("inconsistent paired calibration dimensions")
    features = main_features.detach()
    branches = branch_features.detach()
    target = q_star.detach().to(features)
    q_theta = features.mean(dim=0)
    loss = torch.sum((q_theta - target) ** 2)
    terms = []
    for b in range(bsz):
        q_loo = (features.sum(dim=0) - features[b]) / (bsz - 1)
        for step in range(horizon):
            delta = 2.0 * torch.dot(q_loo - target, branches[b, step, 1] - branches[b, step, 0])
            terms.append(delta.detach() * route_log_probs[b, step])
    return loss, torch.stack(terms).sum() / bsz
