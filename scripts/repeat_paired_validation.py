from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ecomas.agents import SpecialistAgent
from ecomas.datasets import load_samples
from ecomas.encoder import build_encoder
from ecomas.llm import build_llm
from ecomas.paired_sampling import runner_paired_sampler
from ecomas.registry import TASK_REGISTRY
from ecomas.router import ArgmaxRouter
from ecomas.runner import MASRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--B", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    task = "mmlu_pro"
    sample = load_samples(task, Path("/home_bak/guoyuhan/benchmarks"), "validation", 1)[0]
    llm = build_llm("local_hf", args.model_path, args.max_new_tokens, args.temperature, "auto")
    agents = [SpecialistAgent(spec, llm) for spec in TASK_REGISTRY[task]["agents"]]
    encoder = build_encoder("hf", args.model_path, "auto")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    router = ArgmaxRouter(task, [agent.spec.name for agent in agents], encoder.output_dim, device=device)
    router.load(args.checkpoint)
    runner = MASRunner(task, agents, encoder, router)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for repeat in range(args.repeats):
        estimate = runner_paired_sampler(
            runner,
            sample,
            route_mode="sample",
            seed=repeat * 100000,
        ).estimate(args.B)
        row = {
            "repeat": repeat,
            "uid": sample.uid,
            "B": args.B,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed_root": repeat * 100000,
            "horizon": int(runner.task_config["steps"]),
            "orchestration": estimate.orchestration,
            "agent_execution": estimate.agent_execution,
            "stepwise_sum": [a + b for a, b in zip(estimate.orchestration, estimate.agent_execution)],
            "total_stepwise": estimate.total_stepwise,
            "system_uncertainty": estimate.system_uncertainty,
            "closure_difference": estimate.total_stepwise - estimate.system_uncertainty,
            "boundary_probability_orchestration": estimate.boundary_probability_orchestration,
            "boundary_probability_agent": estimate.boundary_probability_agent,
        }
        rows.append(row)
        (args.output_dir / "progress.json").write_text(json.dumps({"completed": repeat + 1, "total": args.repeats, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.output_dir / "report.json").write_text(json.dumps({"task": task, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
