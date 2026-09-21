from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ecomas.agents import SpecialistAgent
from ecomas.config import DEFAULT_ENCODER_MODEL_PATH, PROJECT_ROOT
from ecomas.datasets import load_samples
from ecomas.encoder import build_encoder
from ecomas.llm import build_llm
from ecomas.paired_sampling import runner_paired_sampler
from ecomas.registry import TASK_REGISTRY
from ecomas.router import ArgmaxRouter


def build_runner(
    task: str,
    model_path: Path,
    encoder_model_path: Path,
    checkpoint: Path,
    device: str,
    max_new_tokens: int,
    temperature: float,
):
    llm = build_llm("local_hf", model_path, max_new_tokens, temperature, device)
    agents = [SpecialistAgent(spec, llm) for spec in TASK_REGISTRY[task]["agents"]]
    encoder = build_encoder("hf", encoder_model_path, device)
    router_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    router = ArgmaxRouter(
        task,
        [agent.spec.name for agent in agents],
        encoder.output_dim,
        device=router_device,
        encoder_model_path=encoder_model_path,
    )
    router.load(checkpoint)
    from ecomas.runner import MASRunner

    return MASRunner(task, agents, encoder, router)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_REGISTRY))
    parser.add_argument("--benchmark-root", type=Path, default=Path("/home_bak/guoyuhan/benchmarks"))
    parser.add_argument("--model-path", type=Path, default=Path(
        "/data2/guoyuhan/qwen_semantic_clustering_feasibility/.hf_cache/"
        "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
        "a09a35458c702b33eeacc393d103063234e8bc28"
    ))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-model-path", type=Path, default=DEFAULT_ENCODER_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "paired_validation")
    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--parallel-workers", type=int, default=1)
    args = parser.parse_args()
    split = {"mmlu_pro": "validation", "math500": "test", "chaosnli": "mnli_m"}[args.task]
    samples = load_samples(args.task, args.benchmark_root, split, args.limit)
    runner = build_runner(
        args.task,
        args.model_path,
        args.encoder_model_path,
        args.checkpoint,
        args.device,
        args.max_new_tokens,
        args.temperature,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / args.task / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_index, sample in enumerate(samples):
        sampler = runner_paired_sampler(
            runner,
            sample,
            route_mode="sample",
            seed=sample_index * 1000003,
            parallel_workers=args.parallel_workers,
        )
        estimate = sampler.estimate(args.B)
        rows.append({
            "sample_index": sample_index,
            "uid": sample.uid,
            "B": args.B,
            "horizon": int(runner.task_config["steps"]),
            "orchestration": estimate.orchestration,
            "agent_execution": estimate.agent_execution,
            "stepwise_sum": [o + a for o, a in zip(estimate.orchestration, estimate.agent_execution)],
            "sum_orchestration": sum(estimate.orchestration),
            "sum_agent_execution": sum(estimate.agent_execution),
            "total_stepwise": estimate.total_stepwise,
            "system_uncertainty": estimate.system_uncertainty,
            "closure_difference": estimate.total_stepwise - estimate.system_uncertainty,
            "boundary_probability_orchestration": estimate.boundary_probability_orchestration,
            "boundary_probability_agent": estimate.boundary_probability_agent,
            "orchestration_samples": estimate.orchestration_samples,
            "agent_samples": estimate.agent_samples,
        })
        (output_dir / "progress.json").write_text(json.dumps({"completed": sample_index + 1, "total": len(samples), "rows": rows}, indent=2), encoding="utf-8")
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    report = {"task": args.task, "split": split, "B": args.B, "temperature": args.temperature, "rows": rows}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "task": args.task, "samples": len(rows)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
