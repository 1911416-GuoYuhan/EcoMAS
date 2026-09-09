from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from ecomas.agents import AgentOutput, SpecialistAgent
from ecomas.datasets import BenchmarkSample
from ecomas.encoder import FrozenTextEncoder
from ecomas.evaluation import is_correct, normalize_answer
from ecomas.registry import TASK_REGISTRY
from ecomas.router import ArgmaxRouter


@dataclass
class StepRecord:
    step: int
    agent_name: str
    probabilities: list[float]
    analysis: str
    candidate_answer: str
    raw_text: str
    route_mode: str = "argmax"
    generation_seed: int | None = None
    route_seed: int | None = None
    state_digest: str = ""


@dataclass
class RunRecord:
    uid: str
    task_name: str
    gold_answer: str
    prediction: str
    normalized_gold: str
    normalized_prediction: str
    correct: bool
    steps: list[StepRecord]
    metadata: dict
    path: list[str] = field(default_factory=list)


class MASRunner:
    def __init__(
        self,
        task_name: str,
        agents: list[SpecialistAgent],
        encoder: FrozenTextEncoder,
        router: ArgmaxRouter,
    ) -> None:
        self.task_name = task_name
        self.agents = {agent.spec.name: agent for agent in agents}
        self.encoder = encoder
        self.router = router
        self.task_config = TASK_REGISTRY[task_name]

    def run_sample(
        self,
        sample: BenchmarkSample,
        training: bool = False,
        loss_fn: Callable[[str, str, str], float] | None = None,
        route_mode: str = "argmax",
        generation_seed: int | None = None,
        route_seed: int | None = None,
    ) -> tuple[RunRecord, torch.Tensor | None]:
        if generation_seed is not None:
            torch.manual_seed(generation_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(generation_seed)
        for agent in self.agents.values():
            agent.reset()

        previous_results: list[str] = []
        state_messages = [
            {
                "role": "system",
                "content": f"You are an assistant. Your task is to {sample.input_text}",
            }
        ]
        step_records: list[StepRecord] = []
        log_probs: list[torch.Tensor] = []
        final_candidate = ""

        for step in range(1, int(self.task_config["steps"]) + 1):
            encoded = self.encoder([state_messages])
            current_route_seed = (route_seed + step - 1) if route_seed is not None else None
            decision = self.router.decide(encoded, mode=route_mode, seed=current_route_seed)
            agent = self.agents[decision.agent_name]
            output: AgentOutput = agent.run(
                sample.input_text,
                previous_results,
                str(self.task_config["display_name"]),
                generation_seed=(generation_seed + step - 1) if generation_seed is not None else None,
            )
            if output.candidate_answer.strip():
                final_candidate = output.candidate_answer
            log_probs.append(decision.log_prob)
            step_records.append(
                StepRecord(
                    step=step,
                    agent_name=decision.agent_name,
                    probabilities=decision.probabilities,
                    analysis=output.analysis,
                    candidate_answer=output.candidate_answer,
                    raw_text=output.raw_text,
                    route_mode=route_mode,
                    generation_seed=(generation_seed + step - 1) if generation_seed is not None else None,
                    route_seed=current_route_seed,
                    state_digest=_digest_messages(state_messages),
                )
            )
            previous_results.append(
                f"Successful Action: reasoning\nResult: {output.raw_text}"
            )
            state_messages = agent.state_messages()

        correct = is_correct(self.task_name, final_candidate, sample.gold_answer)
        record = RunRecord(
            uid=sample.uid,
            task_name=self.task_name,
            gold_answer=sample.gold_answer,
            prediction=final_candidate,
            normalized_gold=normalize_answer(self.task_name, sample.gold_answer),
            normalized_prediction=normalize_answer(self.task_name, final_candidate),
            correct=correct,
            steps=step_records,
            metadata={
                **sample.metadata,
                "route_mode": route_mode,
                "generation_seed": generation_seed,
                "output_parse_valid": bool(final_candidate.strip()),
            },
            path=[step.agent_name for step in step_records],
        )
        loss = None
        if training:
            score = loss_fn(self.task_name, final_candidate, sample.gold_answer) if loss_fn else float(correct)
            reward = torch.tensor(float(score), device=self.router.device)
            loss = -reward * torch.stack(log_probs).sum()
        return record, loss

def write_jsonl(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _digest_messages(messages: list[dict[str, str]]) -> str:
    import hashlib

    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
