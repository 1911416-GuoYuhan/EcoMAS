"""Process-level data parallel calibration training.

Each worker owns its own frozen LLM/encoder and router on one CUDA device.
Workers never share CUDA objects or call one model from multiple threads.
After an epoch, the parent averages router state dicts and writes one
checkpoint.  This is deliberately synchronous at epoch boundaries.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import torch

from ecomas.datasets import BenchmarkSample
from ecomas.training import run_calibration_training


def _worker(payload: dict, worker_index: int, queue) -> None:
    try:
        from main import build_runner
        from ecomas.config import RuntimeConfig

        task = payload["task"]
        device = "cuda"
        runtime = RuntimeConfig(
            benchmark_root=Path(payload["benchmark_root"]),
            output_root=Path(payload["output_root"]),
            checkpoint_root=Path(payload["checkpoint_root"]),
            llm_backend=payload["llm_backend"],
            llm_model_path=Path(payload["llm_model_path"]),
            max_new_tokens=int(payload["max_new_tokens"]),
            temperature=float(payload["temperature"]),
            encoder_backend="hf",
            device=device,
        )
        init_checkpoint = Path(payload["init_checkpoint"]) if payload.get("init_checkpoint") else None
        from ecomas.runner import MASRunner
        # build_runner returns agents/encoder/router; reconstruct runner without
        # loading another model.
        agents, encoder, router = build_runner(task, runtime, init_checkpoint, bool(init_checkpoint and init_checkpoint.exists()))
        runner = MASRunner(task, agents, encoder, router)
        subset = [BenchmarkSample(**item) for item in payload["subsets"][worker_index]]
        all_samples = [BenchmarkSample(**item) for item in payload["all_samples"]]
        worker_dir = Path(payload["worker_root"]) / f"worker{worker_index}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        records = run_calibration_training(
            runner, subset, worker_dir / "results.jsonl", worker_dir / "router.pt",
            epochs=int(payload["epochs"]), lr=float(payload["lr"]),
            calibration_samples=int(payload["calibration_samples"]),
            seed=int(payload["seed"]) + worker_index * 1000003,
            target_samples=all_samples,
        )
        # Write the state locally; sending multi-megabyte tensors through a
        # multiprocessing Queue can block the feeder thread at interpreter
        # shutdown.  The parent reads these atomic worker artifacts later.
        (worker_dir / "status.json").write_text(json.dumps({"ok": True, "worker": worker_index, "checkpoint": str(worker_dir / "router.pt"), "records": len(records)}), encoding="utf-8")
    except Exception as exc:  # propagate a readable failure to the parent
        worker_dir = Path(payload["worker_root"]) / f"worker{worker_index}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        (worker_dir / "status.json").write_text(json.dumps({"ok": False, "worker": worker_index, "error": repr(exc)}), encoding="utf-8")


def _sample_payload(sample: BenchmarkSample) -> dict:
    return asdict(sample)


def run_parallel_calibration_training(
    *, task: str, samples: Sequence[BenchmarkSample], gpu_ids: Sequence[int],
    runtime, init_checkpoint: Path | None, output_root: Path, epochs: int = 1,
    lr: float = 1e-3, calibration_samples: int = 10, seed: int = 0,
) -> dict:
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")
    if not samples:
        raise ValueError("samples must be non-empty")
    if epochs != 1:
        raise ValueError("synchronous process trainer currently supports one epoch per call")
    output_root.mkdir(parents=True, exist_ok=True)
    subsets = [list(samples[i::len(gpu_ids)]) for i in range(len(gpu_ids))]
    payload = {
        "task": task, "benchmark_root": str(runtime.benchmark_root),
        "output_root": str(runtime.output_root), "checkpoint_root": str(runtime.checkpoint_root),
        "llm_backend": runtime.llm_backend, "llm_model_path": str(runtime.llm_model_path),
        "max_new_tokens": runtime.max_new_tokens, "temperature": runtime.temperature,
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "subsets": [[_sample_payload(s) for s in subset] for subset in subsets],
        "all_samples": [_sample_payload(s) for s in samples],
        "worker_root": str(output_root / "workers"), "epochs": epochs, "lr": lr,
        "calibration_samples": calibration_samples, "seed": seed,
    }
    ctx = mp.get_context("spawn")
    processes = []
    for index, physical_gpu in enumerate(gpu_ids):
        # CUDA_VISIBLE_DEVICES remaps the selected physical card to cuda:0.
        worker_payload = dict(payload)
        worker_payload["physical_gpu"] = int(physical_gpu)
        process = ctx.Process(target=_worker_with_gpu, args=(worker_payload, index, int(physical_gpu)))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    results = []
    for index in range(len(processes)):
        status_path = Path(payload["worker_root"]) / f"worker{index}" / "status.json"
        if not status_path.exists():
            results.append({"ok": False, "worker": index, "error": f"missing status; exitcode={processes[index].exitcode}"})
        else:
            results.append(json.loads(status_path.read_text(encoding="utf-8")))
    failures = [result for result in results if not result.get("ok")]
    if failures:
        raise RuntimeError("parallel workers failed: " + json.dumps(failures, ensure_ascii=False))
    states = []
    for result in sorted(results, key=lambda item: item["worker"]):
        worker_payload = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
        states.append(worker_payload["model_state_dict"])
    averaged = {}
    for key in states[0]:
        averaged[key] = sum(state[key].float() for state in states) / len(states)
    checkpoint = output_root / "router.pt"
    torch.save({
        "model_state_dict": averaged,
        "input_dim": int(states[0]["fc1.weight"].shape[1]),
        "output_dim": int(states[0]["fc4.weight"].shape[0]),
        "metadata": {"task_name": task, "parallel_gpus": list(gpu_ids)},
    }, checkpoint)
    return {"checkpoint": str(checkpoint), "workers": results, "samples": len(samples), "gpu_ids": list(gpu_ids)}


def _worker_with_gpu(payload, worker_index, physical_gpu):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    _worker(payload, worker_index, None)
