from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW
from tqdm import tqdm

from ecomas.datasets import BenchmarkSample
from ecomas.runner import MASRunner, RunRecord, write_jsonl


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
) -> list[RunRecord]:
    if loss_name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {loss_name}")
    optimizer = AdamW(runner.router.model.parameters(), lr=lr)
    all_records: list[RunRecord] = []
    loss_fn = LOSS_REGISTRY[loss_name]

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
