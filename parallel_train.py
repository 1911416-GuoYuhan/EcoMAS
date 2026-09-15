from __future__ import annotations

import argparse
from pathlib import Path

from ecomas.config import RuntimeConfig
from ecomas.datasets import load_samples
from ecomas.parallel_training import run_parallel_calibration_training


def main():
    parser = argparse.ArgumentParser(description="Process-parallel EcoMAS calibration training")
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--gpus", default="5,7")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--calibration-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-root", default="/home_bak/guoyuhan/benchmarks")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()
    samples = load_samples(args.task, Path(args.benchmark_root), args.split, args.limit)
    runtime = RuntimeConfig(
        benchmark_root=Path(args.benchmark_root), output_root=Path(args.output_root),
        checkpoint_root=Path(args.output_root), llm_model_path=Path(args.model_path),
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, device="cuda",
    )
    result = run_parallel_calibration_training(
        task=args.task, samples=samples,
        gpu_ids=[int(x) for x in args.gpus.split(",") if x.strip()],
        runtime=runtime, init_checkpoint=Path(args.init_checkpoint),
        output_root=Path(args.output_root), epochs=args.epochs, lr=args.lr,
        calibration_samples=args.calibration_samples, seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
