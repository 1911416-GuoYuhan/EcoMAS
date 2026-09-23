from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from ecomas.agents import SpecialistAgent
from ecomas.config import ARTIFACT_CONFIG, COMMAND_CONFIG, PATH_CONFIG, RUNTIME_CONFIG, RuntimeConfig, project_path
from ecomas.datasets import load_samples
from ecomas.encoder import build_encoder
from ecomas.llm import build_llm
from ecomas.registry import TASK_REGISTRY
from ecomas.router import ArgmaxRouter
from ecomas.training import run_explanation, run_inference, run_training


def build_runtime(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        benchmark_root=Path(args.benchmark_root),
        output_root=Path(args.output_root),
        checkpoint_root=Path(args.checkpoint_root),
        llm_backend=args.llm_backend,
        llm_model_path=Path(args.llm_model_path),
        encoder_model_path=Path(args.encoder_model_path),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        encoder_backend=args.encoder_backend,
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
        runtime.encoder_model_path,
        runtime.device,
    )
    agent_names = [agent.spec.name for agent in agents]
    router_device = runtime.device if runtime.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    router = ArgmaxRouter(
        task_name,
        agent_names,
        encoder.output_dim,
        device=router_device,
        encoder_model_path=runtime.encoder_model_path,
    )
    if load_checkpoint:
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(f"Router checkpoint not found: {checkpoint}")
        router.load(checkpoint)
    return agents, encoder, router


def summarize(records) -> dict:
    total = len(records)
    correct = sum(1 for record in records if record.correct)
    parsed = sum(1 for record in records if record.output_parse_valid)
    terminal_parsed = sum(1 for record in records if record.terminal_step_parse_valid)
    protocol = sum(1 for record in records if record.terminal_step_protocol_valid)
    fallback = sum(1 for record in records if record.used_answer_fallback)
    total_steps = sum(len(record.steps) for record in records)
    parsed_steps = sum(step.parse_valid for record in records for step in record.steps)
    protocol_steps = sum(step.protocol_valid for record in records for step in record.steps)
    parsed_correct = sum(1 for record in records if record.output_parse_valid and record.correct)
    methods = {}
    for record in records:
        methods[record.match_method] = methods.get(record.match_method, 0) + 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_valid": parsed,
        "parse_valid_rate": parsed / total if total else 0.0,
        "terminal_parse_valid": terminal_parsed,
        "terminal_parse_valid_rate": terminal_parsed / total if total else 0.0,
        "terminal_protocol_valid": protocol,
        "terminal_protocol_valid_rate": protocol / total if total else 0.0,
        "used_answer_fallback": fallback,
        "used_answer_fallback_rate": fallback / total if total else 0.0,
        "step_parse_valid": parsed_steps,
        "step_parse_valid_rate": parsed_steps / total_steps if total_steps else 0.0,
        "step_protocol_valid": protocol_steps,
        "step_protocol_valid_rate": protocol_steps / total_steps if total_steps else 0.0,
        "correct_among_parse_valid": parsed_correct / parsed if parsed else 0.0,
        "match_methods": methods,
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=sorted(TASK_REGISTRY), required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=COMMAND_CONFIG["common"]["limit"])
    parser.add_argument("--benchmark-root", default=str(project_path(PATH_CONFIG["benchmark_root"])))
    parser.add_argument("--output-root", default=str(project_path(PATH_CONFIG["output_root"])))
    parser.add_argument("--checkpoint-root", default=str(project_path(PATH_CONFIG["checkpoint_root"])))
    parser.add_argument("--llm-backend", choices=RUNTIME_CONFIG["llm_backends"], default=RUNTIME_CONFIG["llm_backend"])
    parser.add_argument(
        "--llm-model-path",
        default=str(project_path(PATH_CONFIG["generation_model"])),
    )
    parser.add_argument("--encoder-model-path", default=str(project_path(PATH_CONFIG["encoder_model"])))
    parser.add_argument("--max-new-tokens", type=int, default=RUNTIME_CONFIG["max_new_tokens"])
    parser.add_argument("--temperature", type=float, default=RUNTIME_CONFIG["inference_temperature"])
    parser.add_argument("--encoder-backend", choices=RUNTIME_CONFIG["encoder_backends"], default=RUNTIME_CONFIG["encoder_backend"])
    parser.add_argument("--device", default=RUNTIME_CONFIG["device"])


def default_split(task: str) -> str:
    return str(TASK_REGISTRY[task]["default_split"])


def default_checkpoint(task: str, checkpoint_root: Path) -> Path:
    pretrained = pretrained_checkpoint_path(task)
    return pretrained if pretrained.is_file() else checkpoint_root / task / ARTIFACT_CONFIG["router_checkpoint"]


def pretrained_checkpoint_path(task: str) -> Path:
    return project_path(PATH_CONFIG["pretrained_root"]) / task / ARTIFACT_CONFIG["pretrained_checkpoint"]


def required_pretrained_checkpoint(task: str) -> Path:
    checkpoint = pretrained_checkpoint_path(task)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Pretrained orchestrator not found for {task}: {checkpoint}. "
            "Place the dataset-specific router checkpoint at this path before running explain or train."
        )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="ecoMAS benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-tasks")

    run_parser = subparsers.add_parser("run")
    add_common_args(run_parser)
    run_parser.add_argument("--checkpoint", default=None)
    run_parser.add_argument("--require-checkpoint", action="store_true")
    run_parser.add_argument("--router-mode", choices=RUNTIME_CONFIG["router_modes"], default=COMMAND_CONFIG["run"]["router_mode"])
    run_parser.add_argument("--num-samples", type=int, default=COMMAND_CONFIG["run"]["num_samples"])
    run_parser.add_argument("--seed", type=int, default=COMMAND_CONFIG["run"]["seed"])

    explain_parser = subparsers.add_parser("explain")
    add_common_args(explain_parser)
    explain_parser.add_argument("--sampling-budget", type=int, default=COMMAND_CONFIG["explain"]["sampling_budget"])
    explain_parser.add_argument("--seed", type=int, default=COMMAND_CONFIG["explain"]["seed"])
    explain_parser.add_argument("--parallel-workers", type=int, default=COMMAND_CONFIG["explain"]["parallel_workers"])
    explain_parser.set_defaults(temperature=RUNTIME_CONFIG["training_temperature"])

    train_parser = subparsers.add_parser("train")
    add_common_args(train_parser)
    train_parser.add_argument("--epochs", type=int, default=COMMAND_CONFIG["train"]["epochs"])
    train_parser.add_argument("--lr", type=float, default=COMMAND_CONFIG["train"]["learning_rate"])
    train_parser.add_argument("--loss", choices=COMMAND_CONFIG["train"]["loss_choices"], default=COMMAND_CONFIG["train"]["loss"])
    train_parser.add_argument("--calibration-samples", type=int, default=COMMAND_CONFIG["train"]["calibration_samples"])
    train_parser.add_argument("--seed", type=int, default=COMMAND_CONFIG["train"]["seed"])
    train_parser.add_argument("--test-split", default=COMMAND_CONFIG["train"]["test_split"])
    train_parser.add_argument("--test-limit", type=int, default=None)
    train_parser.set_defaults(temperature=RUNTIME_CONFIG["training_temperature"])

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
    if args.command == "run" and args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")
    if args.command == "explain" and args.sampling_budget < 2:
        raise ValueError("--sampling-budget must be at least 2")
    runtime = build_runtime(args)
    stamp = datetime.utcnow().strftime(ARTIFACT_CONFIG["run_directory_timestamp_format"])
    mode = args.command
    run_dir = runtime.output_root / args.task / mode / stamp
    if args.command == "train":
        checkpoint_path = runtime.checkpoint_root / args.task / ARTIFACT_CONFIG["router_checkpoint"]
        init_checkpoint = required_pretrained_checkpoint(args.task)
    elif args.command == "explain":
        checkpoint_path = required_pretrained_checkpoint(args.task)
        init_checkpoint = checkpoint_path
    else:
        checkpoint_path = Path(args.checkpoint) if getattr(args, "checkpoint", None) else default_checkpoint(args.task, runtime.checkpoint_root)
        init_checkpoint = checkpoint_path
    samples = load_samples(args.task, runtime.benchmark_root, args.split, args.limit)
    if args.command == "run" and args.require_checkpoint and not checkpoint_path.exists():
        raise FileNotFoundError(f"Router checkpoint not found: {checkpoint_path}")

    agents, encoder, router = build_runner(
        args.task,
        runtime,
        init_checkpoint,
        load_checkpoint=args.command in {"explain", "train"} or bool(init_checkpoint and init_checkpoint.exists()),
    )
    from ecomas.runner import MASRunner

    runner = MASRunner(args.task, agents, encoder, router)
    result_path = run_dir / ARTIFACT_CONFIG["results"]
    calibration_report_path = None

    if args.command == "run":
        records = run_inference(runner, samples, result_path, route_mode=args.router_mode,
                                num_samples=args.num_samples, seed=args.seed)
    elif args.command == "explain":
        explanation_path = run_dir / ARTIFACT_CONFIG["explanation"]
        records = run_explanation(
            runner,
            samples,
            result_path,
            explanation_path,
            sampling_budget=args.sampling_budget,
            seed=args.seed,
            parallel_workers=args.parallel_workers,
        )
    else:
        pre_calibration_state = deepcopy(router.model.state_dict())
        records = run_training(
            runner,
            samples,
            result_path,
            checkpoint_path,
            epochs=args.epochs,
            lr=args.lr,
            loss_name=args.loss,
            calibration_samples=getattr(args, "calibration_samples", COMMAND_CONFIG["train"]["calibration_samples"]),
            seed=getattr(args, "seed", COMMAND_CONFIG["train"]["seed"]),
        )
        if args.loss in COMMAND_CONFIG["train"]["calibration_report_losses"]:
            from ecomas.training import evaluate_calibration
            test_samples = load_samples(args.task, runtime.benchmark_root, args.test_split, args.test_limit or args.limit)
            post_metrics = evaluate_calibration(runner, test_samples, num_samples=max(2, args.calibration_samples), seed=args.seed)
            router.model.load_state_dict(pre_calibration_state)
            pre_metrics = evaluate_calibration(runner, test_samples, num_samples=max(2, args.calibration_samples), seed=args.seed)
            router.load(checkpoint_path)
            calibration_report = {
                "task": args.task,
                "train_split": args.split,
                "test_split": args.test_split,
                "before": {key: value for key, value in pre_metrics.items() if key != "records"},
                "after": {key: value for key, value in post_metrics.items() if key != "records"},
            }
            calibration_report_path = run_dir / ARTIFACT_CONFIG["calibration_report"]
            calibration_report_path.write_text(json.dumps(calibration_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = summarize(records)
    summary.update(
        {
            "task": args.task,
            "split": args.split,
            "mode": mode,
            "result_path": str(result_path),
            "checkpoint_path": str(checkpoint_path),
            "initial_checkpoint_path": str(init_checkpoint),
            "llm_backend": runtime.llm_backend,
            "encoder_backend": runtime.encoder_backend,
        }
    )
    if calibration_report_path is not None:
        summary["calibration_report_path"] = str(calibration_report_path)
    if args.command == "explain":
        summary["explanation_path"] = str(explanation_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
