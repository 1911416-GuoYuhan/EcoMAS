from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    ) -> tuple[RunRecord, torch.Tensor | None]:
        history = ""
        step_records: list[StepRecord] = []
        log_probs: list[torch.Tensor] = []
        final_candidate = ""

        for step in range(1, int(self.task_config["steps"]) + 1):
            state_text = self._state_text(sample.input_text, history, step)
            encoded = self.encoder([state_text])
            decision = self.router.decide(encoded)
            agent = self.agents[decision.agent_name]
            output: AgentOutput = agent.run(
                sample.input_text, history, str(self.task_config["answer_format"])
            )
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
                )
            )
            history = self._append_history(history, output)

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
            metadata=sample.metadata,
        )
        loss = None
        if training:
            score = loss_fn(self.task_name, final_candidate, sample.gold_answer) if loss_fn else float(correct)
            reward = torch.tensor(float(score), device=self.router.device)
            loss = -reward * torch.stack(log_probs).sum()
        return record, loss

    def _state_text(self, input_text: str, history: str, step: int) -> str:
        return f"task={self.task_name}\nstep={step}\n{input_text}\nhistory:\n{history}"

    def _append_history(self, history: str, output: AgentOutput) -> str:
        new_item = (
            f"[{output.agent_name}]\n"
            f"Analysis: {output.analysis}\n"
            f"Candidate: {output.candidate_answer}"
        )
        return f"{history}\n\n{new_item}".strip()


def write_jsonl(path: Path, records: list[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
