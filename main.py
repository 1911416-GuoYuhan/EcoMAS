from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from ecomas.agents import SpecialistAgent
from ecomas.config import PROJECT_ROOT, RuntimeConfig
from ecomas.datasets import load_samples
from ecomas.encoder import build_encoder
from ecomas.llm import build_llm
from ecomas.registry import TASK_REGISTRY
from ecomas.router import ArgmaxRouter
from ecomas.training import run_inference, run_training


def build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        benchmark_root=Path(args.benchmark_root),
        output_root=Path(args.output_root),
        checkpoint_root=Path(args.checkpoint_root),
        llm_backend=args.llm_backend,
        llm_model_path=Path(args.llm_model_path),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        encoder_backend=args.encoder_backend,
        encoder_dim=args.encoder_dim,
        device=args.device,
    )


def build_runner(task_name: str, runtime: RuntimeConfig, checkpoint: Path | None, load_checkpoint: bool):
    task_cfg = TASK_REGISTRY[task_name]
    llm = build_llm(
        runtime.llm_backend,
        runtime.llm_model_path,
        runtime.max_new_tokens,
        runtime.temperature,
        runtime.device,
    )
    agents = [SpecialistAgent(spec, llm) for spec in task_cfg["agents"]]
    encoder = build_encoder(
        runtime.encoder_backend,
        runtime.encoder_dim,
        runtime.llm_model_path,
        runtime.device,
    )
    agent_names = [agent.spec.name for agent in agents]
    router = ArgmaxRouter(task_name, agent_names, encoder.output_dim)
    if load_checkpoint and checkpoint and checkpoint.exists():
        router.load(checkpoint)
    return agents, encoder, router


def summarize(records) -> dict:
    total = len(records)
    correct = sum(1 for record in records if record.correct)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--benchmark-root", default="/home_bak/guoyuhan/benchmarks")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--checkpoint-root", default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--llm-backend", choices=["local_hf", "mock"], default="local_hf")
    parser.add_argument(
        "--llm-model-path",
        default=(
            "/data2/guoyuhan/qwen_semantic_clustering_feasibility/.hf_cache/"
            "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
            "a09a35458c702b33eeacc393d103063234e8bc28"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--encoder-backend", choices=["hash", "hf"], default="hash")
    parser.add_argument("--encoder-dim", type=int, default=768)
    parser.add_argument("--device", default="auto")


def default_split(task: str) -> str:
    if task == "mmlu_pro":
        return "validation"
    if task == "math500":
        return "test"
    if task == "chaosnli":
        return "mnli_m"
    raise ValueError(task)


def main() -> None:
    parser = argparse.ArgumentParser(description="ecoMAS benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tasks")

    run_parser = subparsers.add_parser("run")
    add_common_args(run_parser)
    run_parser.add_argument("--checkpoint", default=None)
    run_parser.add_argument("--require-checkpoint", action="store_true")

    train_parser = subparsers.add_parser("train")
    add_common_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--loss", choices=["accuracy"], default="accuracy")

    args = parser.parse_args()
    if args.command == "list-tasks":
        payload = {
            task: {
                "display_name": cfg["display_name"],
                "steps": cfg["steps"],
                "agents": [agent.name for agent in cfg["agents"]],
            }
            for task, cfg in TASK_REGISTRY.items()
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.split is None:
        args.split = default_split(args.task)
    runtime = build_runtime(args)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    mode = args.command
    run_dir = runtime.output_root / args.task / mode / stamp
    checkpoint_path = (
        Path(args.checkpoint)
        if getattr(args, "checkpoint", None)
        else runtime.checkpoint_root / args.task / "router.pt"
    )
    samples = load_samples(args.task, runtime.benchmark_root, args.split, args.limit)
    if args.command == "run" and args.require_checkpoint and not checkpoint_path.exists():
        raise FileNotFoundError(f"Router checkpoint not found: {checkpoint_path}")

    agents, encoder, router = build_runner(
        args.task,
        runtime,
        checkpoint_path,
        load_checkpoint=(args.command == "run"),
    )
    from ecomas.runner import MASRunner

    runner = MASRunner(args.task, agents, encoder, router)
    result_path = run_dir / "results.jsonl"

    if args.command == "run":
        records = run_inference(runner, samples, result_path)
    else:
        records = run_training(
            runner,
            samples,
            result_path,
            checkpoint_path,
            epochs=args.epochs,
            lr=args.lr,
            loss_name=args.loss,
        )

    summary = summarize(records)
    summary.update(
        {
            "task": args.task,
            "split": args.split,
            "mode": mode,
            "result_path": str(result_path),
            "checkpoint_path": str(checkpoint_path),
            "llm_backend": runtime.llm_backend,
            "encoder_backend": runtime.encoder_backend,
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
