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


def run_inference(runner: MASRunner, samples: list[BenchmarkSample], result_path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for sample in tqdm(samples, desc=f"infer:{runner.task_name}"):
        record, _ = runner.run_sample(sample, training=False)
        records.append(record)
    write_jsonl(result_path, records)
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
