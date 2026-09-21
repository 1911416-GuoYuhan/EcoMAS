from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from ecomas.datasets import BenchmarkSample
from ecomas.runner import MASRunner, RunRecord, write_jsonl
from ecomas.calibration import build_semantic_space, estimate_paired_calibration_gradient, gold_distribution, semantic_key, output_distribution
from ecomas.paired_sampling import runner_paired_sampler
from ecomas.calibration import calibration_error, distribution_brier_score, distribution_nll, ece


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
    from ecomas.uncertainty import semantic_cluster_report
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
    if num_samples > 1:
        uncertainty_reports = {
            uid: {
                "num_samples": report["num_answers"],
                "cluster_distribution": report["cluster_distribution"],
                "second_order_tsallis": report["second_order_tsallis"],
                "second_order_tsallis_u_statistic": report["second_order_tsallis_u_statistic"],
                "pair_count": report["pair_count"],
                "invalid_rate": report["invalid_rate"],
            }
            for uid, report in semantic_reports.items()
        }
        result_path.with_name("uncertainty.json").write_text(
            json.dumps(uncertainty_reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return records


def run_explanation(
    runner: MASRunner,
    samples: list[BenchmarkSample],
    result_path: Path,
    report_path: Path,
    *,
    sampling_budget: int = 10,
    seed: int = 0,
    parallel_workers: int = 1,
) -> list[RunRecord]:
    """Run the paper's paired C/R/Z estimator and persist auditable samples."""
    if sampling_budget < 2:
        raise ValueError("sampling_budget must be at least 2")
    reports = []
    records: list[RunRecord] = []
    for sample_index, sample in enumerate(tqdm(samples, desc=f"explain:{runner.task_name}")):
        sampler = runner_paired_sampler(
            runner,
            sample,
            route_mode="sample",
            seed=seed + sample_index * 1000003,
            parallel_workers=parallel_workers,
        )
        estimate = sampler.estimate(sampling_budget)
        sample_records = [record for record in (estimate.records or []) if record is not None]
        records.extend(sample_records)
        reports.append({
            "uid": sample.uid,
            "sampling_budget": sampling_budget,
            "horizon": int(runner.task_config["steps"]),
            "system_uncertainty": estimate.system_uncertainty,
            "orchestration": estimate.orchestration,
            "agent_execution": estimate.agent_execution,
            "stepwise_total": [
                orchestration + execution
                for orchestration, execution in zip(estimate.orchestration, estimate.agent_execution)
            ],
            "total_stepwise": estimate.total_stepwise,
            "closure_difference": estimate.total_stepwise - estimate.system_uncertainty,
            "boundary_probability_orchestration": estimate.boundary_probability_orchestration,
            "boundary_probability_agent": estimate.boundary_probability_agent,
            "orchestration_samples": estimate.orchestration_samples,
            "agent_samples": estimate.agent_samples,
            "main_outputs": estimate.main_outputs,
            "branch_outputs": estimate.branch_outputs,
        })
        runner.branch_cache.clear()
    write_jsonl(result_path, records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "task": runner.task_name,
        "sampling_budget": sampling_budget,
        "samples": reports,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
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
    optimizer = AdamW(runner.router.model.parameters(), lr=lr, weight_decay=0.0)
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
) -> list[RunRecord]:
    """Train with the paper's stepwise paired C/R suffix estimator."""
    if calibration_samples < 2:
        raise ValueError("calibration_samples must be at least 2")
    space = build_semantic_space(runner.task_name, samples=samples)
    optimizer = AdamW(runner.router.model.parameters(), lr=lr, weight_decay=0.0)
    records: list[RunRecord] = []
    runner.router.model.train()
    for epoch in range(epochs):
        for sample_index, sample in enumerate(tqdm(samples, desc=f"train:{runner.task_name}:epoch{epoch + 1}")):
            optimizer.zero_grad(set_to_none=True)
            # Transformers generation on one shared CUDA model is not thread
            # safe; keep real-Qwen execution serial unless separate model
            # replicas are provisioned.
            sampler = runner_paired_sampler(runner, sample, route_mode="sample", seed=seed + epoch * 100000 + sample_index * calibration_samples, parallel_workers=1)
            estimate = sampler.estimate(calibration_samples)
            main = torch.stack([_feature_tensor(runner.task_name, x, space, runner.router.device) for x in estimate.main_outputs])
            branches = torch.stack([torch.stack([torch.stack([_feature_tensor(runner.task_name, item[k], space, runner.router.device) for k in ("c", "r", "z")]) for item in row]) for row in estimate.branch_outputs])
            logs = torch.stack([torch.stack(list(row)) for row in estimate.route_log_probs])
            target = torch.tensor(
                gold_distribution(runner.task_name, sample, space),
                device=runner.router.device,
                dtype=main.dtype,
            )
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
    """Compatibility entry point for the paper-aligned paired trainer."""
    return run_calibration_training(
        runner,
        samples,
        result_path,
        checkpoint_path,
        epochs=epochs,
        lr=lr,
        calibration_samples=calibration_samples,
        seed=seed,
        mixed=mixed,
    )


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
    targets: list[list[float]] = []
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
        target = gold_distribution(runner.task_name, sample, space)
        targets.append(target)
        labels.append(max(range(space.dimension), key=target.__getitem__))
    per_sample_calibration = [calibration_error(probability, target) for probability, target in zip(probabilities, targets)]
    q_theta = [sum(row[index] for row in probabilities) / len(probabilities) for index in range(space.dimension)] if probabilities else [0.0] * space.dimension
    q_star = [sum(row[index] for row in targets) / len(targets) for index in range(space.dimension)] if targets else [0.0] * space.dimension
    result = {
        "accuracy": sum(record.correct for record in records) / len(records) if records else 0.0,
        "calibration_error": sum(per_sample_calibration) / len(per_sample_calibration) if per_sample_calibration else 0.0,
        "brier": distribution_brier_score(probabilities, targets),
        "nll": distribution_nll(probabilities, targets),
        "ece": ece(probabilities, labels),
        "semantic_space": list(space.keys),
        "q_theta": q_theta,
        "q_star": q_star,
        "per_sample_calibration_error": per_sample_calibration,
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
