from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from ecomas.datasets import BenchmarkSample
from ecomas.runner import MASRunner, RunRecord, write_jsonl
from ecomas.calibration import build_semantic_space, estimate_calibration_gradient, estimate_paired_calibration_gradient, gold_distribution, semantic_key, output_distribution
from ecomas.paired_sampling import runner_paired_sampler
from ecomas.calibration import calibration_error, brier_score, ece, nll


def accuracy_reward(task_name: str, prediction: str, gold: str) -> float:
    from ecomas.evaluation import is_correct

    return 1.0 if is_correct(task_name, prediction, gold) else 0.0


LOSS_REGISTRY = {
    "accuracy": accuracy_reward,
}


def run_inference(runner: MASRunner, samples: list[BenchmarkSample], result_path: Path,
                  route_mode: str = "argmax", num_samples: int = 1, seed: int = 0) -> list[RunRecord]:
    records: list[RunRecord] = []
    for sample in tqdm(samples, desc=f"infer:{runner.task_name}"):
        for sample_index in range(num_samples):
            record, _ = runner.run_sample(sample, training=False, route_mode=route_mode,
                                          generation_seed=seed + sample_index,
                                          route_seed=seed + sample_index)
            records.append(record)
    write_jsonl(result_path, records)
    import json
    from ecomas.uncertainty import grouped_answer_reports, semantic_cluster_report
    if num_samples > 1:
        report_path = result_path.with_name("uncertainty.json")
        report_path.write_text(json.dumps(grouped_answer_reports(records), ensure_ascii=False, indent=2), encoding="utf-8")
    by_uid: dict[str, list[dict[str, object]]] = {}
    for index, record in enumerate(records):
        by_uid.setdefault(str(record.uid), []).append({
            "output_id": f"run-{index}",
            "raw_answer": record.prediction,
            "normalized_prediction": record.normalized_prediction,
            "parse_valid": record.output_parse_valid,
        })
    sample_by_uid = {str(sample.uid): sample for sample in samples}
    semantic_reports = {
        uid: semantic_cluster_report(
            runner.task_name,
            uid,
            answers,
            question=sample_by_uid[uid].input_text if uid in sample_by_uid else "",
        )
        for uid, answers in sorted(by_uid.items())
    }
    result_path.with_name("semantic_clusters.json").write_text(
        json.dumps(semantic_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records


def run_training(
    runner: MASRunner,
    samples: list[BenchmarkSample],
    result_path: Path,
    checkpoint_path: Path,
    epochs: int = 1,
    lr: float = 1e-3,
    loss_name: str = "accuracy",
    calibration_samples: int = 4,
    seed: int = 0,
) -> list[RunRecord]:
    if loss_name not in {*LOSS_REGISTRY, "calibration", "mixed"}:
        raise ValueError(f"Unknown loss function: {loss_name}")
    optimizer = AdamW(runner.router.model.parameters(), lr=lr)
    all_records: list[RunRecord] = []
    loss_fn = LOSS_REGISTRY.get(loss_name)

    if loss_name in {"calibration", "mixed"}:
        return run_calibration_training(
            runner, samples, result_path, checkpoint_path, epochs=epochs, lr=lr,
            calibration_samples=calibration_samples, seed=seed, mixed=(loss_name == "mixed"),
        )

    runner.router.model.train()
    for epoch in range(1, epochs + 1):
        for sample in tqdm(samples, desc=f"train:{runner.task_name}:epoch{epoch}"):
            optimizer.zero_grad(set_to_none=True)
            record, loss = runner.run_sample(sample, training=True, loss_fn=loss_fn)
            all_records.append(record)
            if loss is not None and torch.isfinite(loss):
                loss.backward()
                optimizer.step()

    runner.router.save(checkpoint_path)
    write_jsonl(result_path, all_records)
    return all_records


def run_calibration_training(
    runner: MASRunner, samples: list[BenchmarkSample], result_path: Path, checkpoint_path: Path,
    *, epochs: int = 1, lr: float = 1e-3, calibration_samples: int = 4, seed: int = 0, mixed: bool = False,
    target_samples: list[BenchmarkSample] | None = None,
) -> list[RunRecord]:
    """Train with the paper's stepwise paired C/R suffix estimator."""
    if calibration_samples < 2:
        raise ValueError("calibration_samples must be at least 2")
    target_support = target_samples or samples
    space = build_semantic_space(runner.task_name, samples=target_support)
    target = torch.tensor([gold_distribution(runner.task_name, s, space) for s in target_support], device=runner.router.device).mean(0)
    optimizer = AdamW(runner.router.model.parameters(), lr=lr)
    from ecomas.batched_runner import BatchedMASRunner
    batched_runner = BatchedMASRunner(runner, micro_batch_size=min(4, calibration_samples))
    records: list[RunRecord] = []
    runner.router.model.train()
    for epoch in range(epochs):
        for sample_index, sample in enumerate(tqdm(samples, desc=f"train:{runner.task_name}:epoch{epoch + 1}")):
            optimizer.zero_grad(set_to_none=True)
            estimate = batched_runner.paired_estimate(
                sample, calibration_samples,
                seed + epoch * 100000 + sample_index * calibration_samples,
            )
            main = torch.stack([_feature_tensor(runner.task_name, x, space, runner.router.device) for x in estimate.main_outputs])
            branches = torch.stack([torch.stack([torch.stack([_feature_tensor(runner.task_name, item[k], space, runner.router.device) for k in ("c", "r", "z")]) for item in row]) for row in estimate.branch_outputs])
            logs = torch.stack([torch.stack(list(row)) for row in estimate.route_log_probs])
            _, loss = estimate_paired_calibration_gradient(main, branches, logs, target)
            if mixed:
                accuracy_term = torch.stack([
                    -float(getattr(record, "correct", False)) * logs[i].sum()
                    for i, record in enumerate(estimate.records or [])
                ]).mean()
                loss = loss + 0.1 * accuracy_term
            loss.backward()
            optimizer.step()
            records.extend([record for record in (estimate.records or []) if record is not None])
            runner.branch_cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    runner.router.save(checkpoint_path)
    write_jsonl(result_path, records)
    return records


def _feature_tensor(task_name, output, space, device):
    vector = torch.zeros(space.dimension, device=device)
    key = semantic_key(task_name, output)
    if key not in space.keys and task_name == "math500":
        key = "math500::__other__"
    if key in space.keys:
        vector[space.index(key)] = 1.0
    return vector


def run_trajectory_calibration_training(
    runner: MASRunner,
    samples: list[BenchmarkSample],
    result_path: Path,
    checkpoint_path: Path,
    *,
    epochs: int = 1,
    lr: float = 1e-3,
    calibration_samples: int = 4,
    seed: int = 0,
    mixed: bool = False,
) -> list[RunRecord]:
    """Train the router against a fixed task-level semantic distribution.

    Each item draws B independent MAS trajectories.  The leave-one-out
    surrogate in :func:`estimate_calibration_gradient` prevents the sampled
    trajectory from appearing in its own q estimate.  LLM/encoder parameters
    stay frozen; gradients flow only through categorical route log-probabilities.
    """
    if calibration_samples < 2:
        raise ValueError("calibration_samples must be at least 2")
    space = build_semantic_space(runner.task_name, samples=samples)
    target_distribution = torch.tensor(
        [gold_distribution(runner.task_name, sample, space) for sample in samples],
        device=runner.router.device,
        dtype=torch.float32,
    ).mean(dim=0)
    optimizer = AdamW(runner.router.model.parameters(), lr=lr)
    all_records: list[RunRecord] = []
    runner.router.model.train()
    for epoch in range(epochs):
        for sample_index, sample in enumerate(tqdm(samples, desc=f"train:{runner.task_name}:epoch{epoch + 1}")):
            optimizer.zero_grad(set_to_none=True)
            features = []
            route_terms = []
            batch_records = []
            for draw in range(calibration_samples):
                record, _ = runner.run_sample(
                    sample, training=True, loss_fn=lambda *_: 1.0, route_mode="sample",
                    generation_seed=seed + epoch * 100000 + sample_index * calibration_samples + draw,
                    route_seed=seed + epoch * 100000 + sample_index * calibration_samples + draw,
                )
                batch_records.append(record)
                vector = torch.zeros(space.dimension, device=runner.router.device)
                key = semantic_key(runner.task_name, record.normalized_prediction or record.prediction)
                if key not in space.keys and runner.task_name == "math500":
                    key = "math500::__other__"
                if key in space.keys:
                    vector[space.index(key)] = 1.0
                features.append(vector)
            # Re-run route decisions with gradients retained is unnecessary:
            # run_sample stores no tensors in JSON records, so obtain a fresh
            # differentiable route log-prob sum for each replayed seed.
            route_log_probs = []
            for draw in range(calibration_samples):
                replay, differentiable_loss = runner.run_sample(
                    sample, training=True, loss_fn=lambda *_: 1.0, route_mode="sample",
                    generation_seed=seed + epoch * 100000 + sample_index * calibration_samples + draw,
                    route_seed=seed + epoch * 100000 + sample_index * calibration_samples + draw,
                )
                # The runner's standard loss is a convenient differentiable
                # carrier; normalize away its reward so calibration controls
                # the coefficient below.
                route_log_probs.append(-differentiable_loss if differentiable_loss is not None else torch.zeros((), device=runner.router.device))
            feature_tensor = torch.stack(features)
            q_star = target_distribution
            route_tensor = torch.stack(route_log_probs).unsqueeze(1)
            _, surrogate = estimate_calibration_gradient(feature_tensor, route_tensor, q_star)
            loss = surrogate
            if mixed:
                loss = loss + 0.1 * torch.stack([
                    -torch.tensor(float(record.correct), device=runner.router.device) * route_tensor[index, 0]
                    for index, record in enumerate(batch_records)
                ]).mean()
            if torch.isfinite(loss):
                loss.backward()
                optimizer.step()
            all_records.extend(batch_records)
    runner.router.save(checkpoint_path)
    write_jsonl(result_path, all_records)
    return all_records


def evaluate_calibration(
    runner: MASRunner,
    samples: list[BenchmarkSample],
    *,
    num_samples: int = 4,
    route_mode: str = "sample",
    seed: int = 0,
) -> dict[str, object]:
    """Evaluate fixed-space predictive distributions without updating weights.

    The returned primary metrics are accuracy and squared semantic calibration
    error.  ECE/Brier/NLL use the same task-level coordinates and are secondary
    diagnostics; they are not substituted for the paper's MMD objective.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    space = build_semantic_space(runner.task_name, samples=samples)
    probabilities: list[list[float]] = []
    labels: list[int] = []
    records: list[RunRecord] = []
    for sample_index, sample in enumerate(samples):
        outputs = []
        for draw in range(num_samples):
            record, _ = runner.run_sample(
                sample, route_mode=route_mode,
                generation_seed=seed + sample_index * num_samples + draw,
                route_seed=seed + sample_index * num_samples + draw,
            )
            records.append(record)
            outputs.append(record.normalized_prediction or record.prediction)
        vector = output_distribution(runner.task_name, outputs, space)
        probabilities.append(vector)
        key = semantic_key(runner.task_name, sample.gold_answer)
        labels.append(space.index(key) if key in space.keys else 0)
    gold = [[1.0 if index == label else 0.0 for index in range(space.dimension)] for label in labels]
    q_theta = [sum(row[index] for row in probabilities) / len(probabilities) for index in range(space.dimension)]
    q_star = [sum(row[index] for row in gold) / len(gold) for index in range(space.dimension)]
    result = {
        "accuracy": sum(record.correct for record in records[::num_samples]) / len(samples) if samples else 0.0,
        "calibration_error": calibration_error(q_theta, q_star),
        "brier": brier_score(probabilities, labels),
        "nll": nll(probabilities, labels),
        "ece": ece(probabilities, labels),
        "semantic_space": list(space.keys),
        "q_theta": q_theta,
        "q_star": q_star,
        "parse_valid_rate": sum(record.output_parse_valid for record in records) / len(records) if records else 0.0,
        "records": records,
    }
    return result


def compare_calibration(
    before_runner: MASRunner,
    after_runner: MASRunner,
    test_samples: list[BenchmarkSample],
    *,
    num_samples: int = 4,
    seed: int = 0,
) -> dict[str, object]:
    """Evaluate two routers on exactly the same test draws/configuration."""
    before = evaluate_calibration(before_runner, test_samples, num_samples=num_samples, seed=seed)
    after = evaluate_calibration(after_runner, test_samples, num_samples=num_samples, seed=seed)
    return {
        "before": {key: value for key, value in before.items() if key != "records"},
        "after": {key: value for key, value in after.items() if key != "records"},
        "delta": {
            "accuracy": float(after["accuracy"]) - float(before["accuracy"]),
            "calibration_error": float(after["calibration_error"]) - float(before["calibration_error"]),
            "brier": float(after["brier"]) - float(before["brier"]),
            "nll": float(after["nll"]) - float(before["nll"]),
            "ece": float(after["ece"]) - float(before["ece"]),
        },
    }
